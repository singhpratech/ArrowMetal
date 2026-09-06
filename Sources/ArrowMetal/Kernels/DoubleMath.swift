import Foundation

/// Software IEEE-754 binary64 arithmetic for Metal, which has no `double` type.
///
/// Values travel as raw 64-bit patterns (`ulong`). `d_add`, `d_sub`, `d_mul` are correctly rounded
/// (round to nearest, ties to even) including subnormals, signed zeros, infinities and NaN propagation.
/// `d_div` is a restoring long division on the significands, also correctly rounded.
enum DoubleMath {
    static let msl = """

    #define D_INF  0x7FF0000000000000ul
    #define D_QNAN 0x7FF8000000000000ul
    inline ulong d_exp(ulong x) { return (x >> 52) & 0x7FFul; }
    inline ulong d_mant(ulong x) { return x & 0xFFFFFFFFFFFFFul; }
    inline bool d_is_nan(ulong x) { return d_exp(x) == 0x7FFul && d_mant(x) != 0ul; }
    inline bool d_is_zero(ulong x) { return (x & 0x7FFFFFFFFFFFFFFFul) == 0ul; }

    // Rounds and packs: sign s, biased exponent e for a significand whose leading one sits at bit 55
    // (three guard bits below the 53-bit significand; bit 0 carries sticky).
    inline ulong d_finish(ulong s, long e, ulong m) {
        if (m == 0ul) return s << 63;
        while ((m >> 55) == 0ul) { m <<= 1; e--; }
        while ((m >> 56) != 0ul) { m = (m >> 1) | (m & 1ul); e++; }
        if (e <= 0) {
            long sh = 1 - e;
            if (sh > 60) { m = (m != 0ul) ? 1ul : 0ul; }
            else { ulong st = (m & ((1ul << sh) - 1ul)) ? 1ul : 0ul; m = (m >> sh) | st; }
            e = 0;
        }
        ulong low = m & 7ul; m >>= 3;
        if (low > 4ul || (low == 4ul && (m & 1ul))) m++;
        if ((m >> 53) != 0ul) { m >>= 1; e++; }
        if (e == 0 && (m >> 52) != 0ul) e = 1;
        if (e >= 0x7FF) return (s << 63) | D_INF;
        return (s << 63) | ((ulong)e << 52) | (m & 0xFFFFFFFFFFFFFul);
    }

    inline ulong d_add(ulong a, ulong b) {
        ulong sa = a >> 63, sb = b >> 63;
        long ea = (long)d_exp(a), eb = (long)d_exp(b);
        ulong ma = d_mant(a), mb = d_mant(b);
        if (ea == 0x7FF || eb == 0x7FF) {
            if (ea == 0x7FF && ma) return a | (1ul << 51);
            if (eb == 0x7FF && mb) return b | (1ul << 51);
            if (ea == 0x7FF && eb == 0x7FF) return (sa == sb) ? a : D_QNAN;
            return (ea == 0x7FF) ? a : b;
        }
        bool za = (ea == 0 && ma == 0ul), zb = (eb == 0 && mb == 0ul);
        if (za && zb) return (sa & sb) << 63;
        if (za) return b;
        if (zb) return a;
        if (ea) ma |= 1ul << 52; else ea = 1;
        if (eb) mb |= 1ul << 52; else eb = 1;
        ma <<= 3; mb <<= 3;
        if (ea < eb || (ea == eb && ma < mb)) {
            ulong tm = ma; ma = mb; mb = tm; long te = ea; ea = eb; eb = te; ulong ts = sa; sa = sb; sb = ts;
        }
        long d = ea - eb;
        if (d > 0) {
            if (d >= 64) mb = 1ul;
            else { ulong st = (mb & ((1ul << d) - 1ul)) ? 1ul : 0ul; mb = (mb >> d) | st; }
        }
        ulong m = (sa == sb) ? (ma + mb) : (ma - mb);
        if (m == 0ul) return 0ul;
        return d_finish(sa, ea, m);
    }
    // Exact widening of a float bit pattern to a double bit pattern (subnormals preserved: no float arithmetic).
    inline ulong d_from_float(float f) {
        uint b = as_type<uint>(f);
        ulong s = (ulong)(b >> 31); ulong e = (b >> 23) & 0xFFu; ulong m = b & 0x7FFFFFu;
        if (e == 0xFFu) return (s << 63) | 0x7FF0000000000000ul | (m ? ((1ul << 51) | (m << 29)) : 0ul);
        if (e == 0u) {
            if (m == 0u) return s << 63;
            long ee = 1; while ((m >> 23) == 0ul) { m <<= 1; ee--; }
            m &= 0x7FFFFFu;
            return (s << 63) | ((ulong)(ee - 127 + 1023) << 52) | (m << 29);
        }
        return (s << 63) | ((e - 127 + 1023) << 52) | (m << 29);
    }
    inline ulong d_sub(ulong a, ulong b) { return d_add(a, b ^ 0x8000000000000000ul); }

    inline ulong d_mul(ulong a, ulong b) {
        ulong s = (a ^ b) >> 63;
        long ea = (long)d_exp(a), eb = (long)d_exp(b);
        ulong ma = d_mant(a), mb = d_mant(b);
        if (ea == 0x7FF || eb == 0x7FF) {
            if (ea == 0x7FF && ma) return a | (1ul << 51);
            if (eb == 0x7FF && mb) return b | (1ul << 51);
            if (d_is_zero(a) || d_is_zero(b)) return D_QNAN;
            return (s << 63) | D_INF;
        }
        if (d_is_zero(a) || d_is_zero(b)) return s << 63;
        if (ea) ma |= 1ul << 52; else { ea = 1; while ((ma >> 52) == 0ul) { ma <<= 1; ea--; } }
        if (eb) mb |= 1ul << 52; else { eb = 1; while ((mb >> 52) == 0ul) { mb <<= 1; eb--; } }
        ulong hi = mulhi(ma, mb), lo = ma * mb;
        ulong m = (hi << 15) | (lo >> 49);
        m |= (lo & ((1ul << 49) - 1ul)) ? 1ul : 0ul;
        return d_finish(s, ea + eb - 1023, m);
    }

    // Correctly rounded division by restoring long division on the significands (57 quotient bits:
    // 53 + 3 guard bits, sticky from the remainder). Divisions are rare in analytics; simplicity wins.
    inline ulong d_div(ulong a, ulong b) {
        ulong s = (a ^ b) >> 63;
        long ea = (long)d_exp(a), eb = (long)d_exp(b);
        ulong ma = d_mant(a), mb = d_mant(b);
        if (d_is_nan(a)) return a | (1ul << 51);
        if (d_is_nan(b)) return b | (1ul << 51);
        if (ea == 0x7FF) return (eb == 0x7FF) ? D_QNAN : ((s << 63) | D_INF);
        if (eb == 0x7FF) return s << 63;
        if (d_is_zero(b)) return d_is_zero(a) ? D_QNAN : ((s << 63) | D_INF);
        if (d_is_zero(a)) return s << 63;
        if (ea) ma |= 1ul << 52; else { ea = 1; while ((ma >> 52) == 0ul) { ma <<= 1; ea--; } }
        if (eb) mb |= 1ul << 52; else { eb = 1; while ((mb >> 52) == 0ul) { mb <<= 1; eb--; } }
        // Restoring division needs rem < mb before each step; ma may be up to 2*mb, so take the first bit first.
        ulong rem, q;
        if (ma >= mb) { rem = ma - mb; q = 1ul; } else { rem = ma; q = 0ul; }
        for (int i = 0; i < 56; i++) {
            rem <<= 1; q <<= 1;
            if (rem >= mb) { rem -= mb; q |= 1ul; }
        }
        if (rem != 0ul) q |= 1ul;                       // sticky
        // q = floor(ma * 2^56 / mb) (57 bits at most)  =>  value = (q / 2^55) * 2^(ea - eb - 1)
        return d_finish(s, ea - eb + 1022, q);
    }
    """
}
