import Foundation

/// Software IEEE-754 **binary64 `exp`, `ln`, `log2`, `log10` and `pow`** for Metal, layered on the
/// correctly rounded `d_add` / `d_sub` / `d_mul` / `d_div` / `d_sqrt` of `DoubleMath` and the small
/// helpers (`dt_pow2`, `dt_ldexp`, `dt_from_long`, `dt_to_long`, `dt_round_int`) of
/// `DoubleTranscendental`. Concatenate the three, in that order, into one MSL translation unit.
///
/// These five are what `Kernels/RoundingSource.swift` used to evaluate by narrowing the column to
/// `float`, calling Metal's `exp`/`log`/`log2`/`log10`/`pow` and widening the answer back — seven
/// correct significant decimal digits out of sixteen, and no subnormal or above-`float`-range results
/// at all. Everything below runs in binary64 from end to end.
///
/// **The hard one is `pow`.** `x^y` is `2^(y·log2 x)`, and a relative error of one ulp in the result
/// needs the *product* `y·log2 x` — which reaches 1024 in magnitude — accurate to about `2^-61`
/// absolutely. That is 61 significant bits of an intermediate, more than a double holds, so `log2 x`
/// has to be carried as an unevaluated pair and multiplied by a split `y` in a way that keeps the
/// leading term exact. The layout here is fdlibm's `__ieee754_pow`: every high part has its low 32
/// mantissa bits cleared, so `y1·t1` (21 bits by 21 bits) is an *exact* double, and the correction
/// `(y − y1)·t1 + y·t2` carries the rest. Its polynomial coefficients and the `cp`, `lg2`, `dp`
/// constants are fdlibm's too, reused verbatim rather than re-derived; the accompanying Remez fits are
/// what make the 61 bits reachable with plain double operations instead of a full double-double.
///
/// The same routine yields the three logarithms for free, and more accurately than a direct series
/// would: `dp_log2_hl` returns `log2(x)` as `t1 + t2`, and
///
///   * `log2(x)` is `t1 + t2` rounded once;
///   * `ln(x)` is `t1·ln2_hi + (t1·ln2_lo + t2·ln2)`, where `t1` holds 21 significant bits and
///     `ln2_hi` 32, so the leading product is **exact** and only the tiny correction rounds;
///   * `log10(x)` is the same shape against a `log10 2` split the same way.
///
/// `exp` needs none of that: one argument reduction `x = k·ln2 + r` against the 107-bit `ln 2` pair
/// already in `DoubleTranscendental`, then the `Σ rⁿ/(n+1)!` series that `dt_expm1` uses, then
/// `2^k·(1 + E)`. The scaling is split as `(s·2^(k-1))·2` above `k = 1023` so that a finite result just
/// under `DBL_MAX` does not overflow on the way, and steps through `2^-1022` below it so that a
/// subnormal result rounds exactly once.
///
/// **Accuracy**, measured by `DoubleTranscendentalTests` against Foundation over 10^6 random inputs per
/// function and stated in `docs/DESIGN.md`: `sqrt` is bit-identical (correctly rounded), and `exp`,
/// `ln`, `log2`, `log10` and `power` are within 2 ulp — in practice 1 ulp everywhere the test looks.
enum DoublePower {
    /// A Double bit pattern as an MSL `ulong` literal.
    private static func lit(_ bits: UInt64) -> String { String(format: "0x%016llXul", bits) }
    private static func lit(_ v: Double) -> String { lit(v.bitPattern) }

    /// Horner evaluation of `coefficients` (as bit patterns, lowest order first) in `variable`.
    private static func horner(_ coefficients: [UInt64], _ variable: String) -> String {
        var s = lit(coefficients.last!)
        for c in coefficients.dropLast().reversed() { s = "d_add(d_mul(\(s), \(variable)), \(lit(c)))" }
        return s
    }

    /// `1/(n+1)!` for n = 0...15: `exp(r) - 1 = r·Σ rⁿ/(n+1)!`, ample for `|r| <= ln2/2`.
    private static let expCoefficients: [UInt64] = {
        var c: [UInt64] = [], f = 1.0
        for n in 0...15 { f *= Double(n + 1); c.append((1.0 / f).bitPattern) }
        return c
    }()

    // fdlibm's Remez coefficients. `L*` fits `(3/2)·(log(x) - 2s - (2/3)s³)` in `s²`, `P*` fits
    // `2^z - 1 - z` for the final exponential. Given as bit patterns because the last few bits of each
    // are the whole point of using them.
    private static let logCoefficients: [UInt64] = [0x3FE3_3333_3333_3303, 0x3FDB_6DB6_DB6F_ABFF,
                                                    0x3FD5_5555_518F_264D, 0x3FD1_7460_A91D_4101,
                                                    0x3FCD_864A_93C9_DB65, 0x3FCA_7E28_4A45_4EEF]
    private static let expPolyCoefficients: [UInt64] = [0x3FC5_5555_5555_553E, 0xBF66_C16C_16BE_BD93,
                                                        0x3F11_566A_AF25_DE2C, 0xBEBB_BD41_C5D2_6BF1,
                                                        0x3E66_3769_72BE_A4D0]

    static let msl: String = {
        // log10(2), split so the high part keeps 32 significant bits (its low 21 mantissa bits are
        // zero) and therefore multiplies a 21-bit `t1` exactly. The low part is the *true* tail, not
        // `Double(log10 2) - hi`: the two together carry about 85 bits, and the 61 that `pow`-grade
        // accuracy needs would not survive the shorter split.
        let log10Two = 0.301_029_995_663_981_195_213_738_894_724_493_026_768
        let log10TwoHi = Double(bitPattern: log10Two.bitPattern & ~0x1F_FFFF)
        // log10(2) - log10TwoHi rounded to a double; the pair is then exact to within 2.7e-27.
        let log10TwoLo = Double(bitPattern: 0x3DDF_79FE_F311_F12B)
        return """

        #define DP_ONE     0x3FF0000000000000ul
        #define DP_TWO     0x4000000000000000ul
        #define DP_THREE   0x4008000000000000ul
        #define DP_HALF    0x3FE0000000000000ul
        #define DP_1_5     0x3FF8000000000000ul
        #define DP_TWO53   0x4340000000000000ul
        #define DP_SIGN    0x8000000000000000ul
        #define DP_LN2     0x3FE62E42FEFA39EFul
        #define DP_IVLN2   0x3FF71547652B82FEul
        #define DP_IVLN2_H 0x3FF7154760000000ul
        #define DP_IVLN2_L 0x3E54AE0BF85DDF44ul
        #define DP_LG2     0x3FE62E42FEFA39EFul
        #define DP_LG2_H   0x3FE62E4300000000ul
        #define DP_LG2_L   0xBE205C610CA86C39ul
        #define DP_CP      0x3FEEC709DC3A03FDul
        #define DP_CP_H    0x3FEEC709E0000000ul
        #define DP_CP_L    0xBE3E2FE0145B01F5ul
        #define DP_DP_H1   0x3FE2B80340000000ul
        #define DP_DP_L1   0x3E4CFDEB43CFD006ul
        #define DP_OVT     0x3C971547652B82FEul
        #define DP_LOG10_2   \(lit(log10Two))
        #define DP_LOG10_2_H \(lit(log10TwoHi))
        #define DP_LOG10_2_L \(lit(log10TwoLo))
        #define DP_710     \(lit(710.0))
        #define DP_M746    \(lit(-746.0))
        #define DP_THIRD   \(lit(1.0 / 3.0))
        #define DP_QUARTER \(lit(0.25))

        // The three word games the whole scheme rests on: read the top 32 bits of a double, replace
        // them, or drop the bottom 32 mantissa bits to leave a 21-significant-bit value whose products
        // with another such value are exact.
        inline uint  dp_hiword(ulong x) { return (uint)(x >> 32); }
        inline ulong dp_sethi(ulong x, uint h) { return ((ulong)h << 32) | (x & 0xFFFFFFFFul); }
        inline ulong dp_hi21(ulong x) { return x & 0xFFFFFFFF00000000ul; }
        inline bool  dp_lt(ulong a, ulong b) { return d_key((long)a) < d_key((long)b); }

        // ---- log2(x) as an unevaluated pair t1 + t2 -------------------------------------------------
        // x must be finite and strictly positive. t1 carries 21 significant bits (its low mantissa word
        // is cleared) and t2 the remaining ~53, so the pair is good to roughly 2^-70 of the result.
        inline void dp_log2_hl(ulong ax, thread ulong* pt1, thread ulong* pt2) {
            long n = 0;
            uint ix = dp_hiword(ax) & 0x7FFFFFFFu;
            if (ix < 0x00100000u) { ax = d_mul(ax, DP_TWO53); n -= 53; ix = dp_hiword(ax) & 0x7FFFFFFFu; }
            n += (long)(ix >> 20) - 1023;
            uint j = ix & 0x000FFFFFu;
            ix = j | 0x3FF00000u;                       // the significand alone, in [1, 2)
            // Two sub-intervals around 1 and 1.5, so |s| stays under 0.1716 either way.
            int k;
            if (j <= 0x3988Eu) k = 0;                   // significand < sqrt(3/2)
            else if (j < 0xBB67Au) k = 1;               // significand < sqrt(3)
            else { k = 0; n += 1; ix -= 0x00100000u; }
            ax = dp_sethi(ax, ix);
            ulong bp = k ? DP_1_5 : DP_ONE;
            ulong dpH = k ? DP_DP_H1 : 0ul, dpL = k ? DP_DP_L1 : 0ul;
            ulong u = d_sub(ax, bp);
            ulong v = d_div(DP_ONE, d_add(ax, bp));
            ulong ss = d_mul(u, v);                     // (x - bp) / (x + bp)
            ulong s_h = dp_hi21(ss);
            // The high half of x + bp, assembled from the exponent rather than added: (ix >> 1) halves
            // the exponent field, and the constants re-bias it and fold in the 1.5 of the second
            // interval. t_l is then the exact remainder.
            ulong t_h = dp_sethi(0ul, ((ix >> 1) | 0x20000000u) + 0x00080000u + ((uint)k << 18));
            ulong t_l = d_sub(ax, d_sub(t_h, bp));
            ulong s_l = d_mul(v, d_sub(d_sub(u, d_mul(s_h, t_h)), d_mul(s_h, t_l)));
            ulong s2 = d_mul(ss, ss);
            ulong r = d_mul(d_mul(s2, s2), \(horner(logCoefficients, "s2")));
            r = d_add(r, d_mul(s_l, d_add(s_h, ss)));
            s2 = d_mul(s_h, s_h);
            t_h = dp_hi21(d_add(d_add(DP_THREE, s2), r));
            t_l = d_sub(r, d_sub(d_sub(t_h, DP_THREE), s2));
            u = d_mul(s_h, t_h);
            v = d_add(d_mul(s_l, t_h), d_mul(t_l, ss));
            ulong p_h = dp_hi21(d_add(u, v));
            ulong p_l = d_sub(v, d_sub(p_h, u));
            ulong z_h = d_mul(DP_CP_H, p_h);            // cp = 2/(3 ln2) turns the log into a log2
            ulong z_l = d_add(d_add(d_mul(DP_CP_L, p_h), d_mul(p_l, DP_CP)), dpL);
            ulong t = dt_from_long(n);
            ulong t1 = dp_hi21(d_add(d_add(d_add(z_h, z_l), dpH), t));
            *pt1 = t1;
            *pt2 = d_sub(z_l, d_sub(d_sub(d_sub(t1, t), dpH), z_h));
        }

        // ---- the three logarithms ---------------------------------------------------------------
        // Shared domain handling: 0 is -inf, a negative value NaN, NaN and +inf pass through. Returns
        // true when `out` already holds the answer.
        inline bool dp_log_special(ulong a, thread ulong* out) {
            if (d_is_nan(a)) { *out = a | (1ul << 51); return true; }
            if (d_is_zero(a)) { *out = DT_NINF; return true; }
            if ((a >> 63) != 0ul) { *out = D_QNAN; return true; }
            if (d_exp(a) == 0x7FFul) { *out = a; return true; }
            return false;
        }
        inline ulong dp_log2(ulong a) {
            ulong out;
            if (dp_log_special(a, &out)) return out;
            ulong t1, t2;
            dp_log2_hl(a, &t1, &t2);
            return d_add(t1, t2);                       // exact on a power of two: t2 is then zero
        }
        // ln(x) = log2(x) * ln2. t1 has 21 significant bits and DT_LN2_HI 32, so t1 * DT_LN2_HI is an
        // exact double and only the two small corrections round.
        inline ulong dp_ln(ulong a) {
            ulong out;
            if (dp_log_special(a, &out)) return out;
            ulong t1, t2;
            dp_log2_hl(a, &t1, &t2);
            return d_add(d_mul(t1, DT_LN2_HI), d_add(d_mul(t1, DT_LN2_LO), d_mul(t2, DP_LN2)));
        }
        inline ulong dp_log10(ulong a) {
            ulong out;
            if (dp_log_special(a, &out)) return out;
            ulong t1, t2;
            dp_log2_hl(a, &t1, &t2);
            return d_add(d_mul(t1, DP_LOG10_2_H),
                         d_add(d_mul(t1, DP_LOG10_2_L), d_mul(t2, DP_LOG10_2)));
        }

        // ---- exp ----------------------------------------------------------------------------------
        inline ulong dp_exp(ulong x) {
            if (d_is_nan(x)) return x | (1ul << 51);
            if (d_exp(x) == 0x7FFul) return (x >> 63) ? 0ul : x;
            if (dp_lt(DP_710, x)) return D_INF;         // exp overflows above 709.7827...
            if (dp_lt(x, DP_M746)) return 0ul;          // and underflows to zero below -745.1332...
            long k = 0;
            if (d_exp(x) >= 1022ul) k = dt_to_long(dt_round_int(d_mul(x, DP_IVLN2), 7u));  // |x| >= 1/2
            ulong r = x;
            if (k != 0) {
                ulong dk = dt_from_long(k);
                r = d_sub(d_sub(r, d_mul(dk, DT_LN2_HI)), d_mul(dk, DT_LN2_LO));
            }
            ulong s = d_add(DP_ONE, d_mul(r, \(horner(expCoefficients, "r"))));
            if (k == 0) return s;
            // Split the last doubling out so that a finite result just under DBL_MAX (k = 1024 with a
            // significand below 1) is not turned into infinity by an intermediate 2^1024.
            if (k > 1023) return d_mul(d_mul(s, dt_pow2(k - 1)), DP_TWO);
            return dt_ldexp(s, k);
        }

        // ---- pow ------------------------------------------------------------------------------------
        inline ulong dp_pow(ulong x, ulong y) {
            uint hx = dp_hiword(x), lx = (uint)x, hy = dp_hiword(y), ly = (uint)y;
            uint ix = hx & 0x7FFFFFFFu, iy = hy & 0x7FFFFFFFu;
            if ((iy | ly) == 0u) return DP_ONE;                     // x^0 = 1, even for NaN and inf
            if (hx == 0x3FF00000u && lx == 0u) return DP_ONE;       // 1^y = 1, even for y = NaN (C99)
            if (ix > 0x7FF00000u || (ix == 0x7FF00000u && lx != 0u) ||
                iy > 0x7FF00000u || (iy == 0x7FF00000u && ly != 0u)) return d_add(x, y);   // NaN
            // Is y an integer, and is it odd? Only matters for a negative base.
            uint yisint = 0u;
            if ((x >> 63) != 0ul) {
                if (iy >= 0x43400000u) yisint = 2u;                 // |y| >= 2^53: certainly even
                else if (iy >= 0x3FF00000u) {
                    int ky = (int)(iy >> 20) - 1023;
                    if (ky > 20) {
                        uint jj = ly >> (52 - ky);
                        if ((jj << (52 - ky)) == ly) yisint = 2u - (jj & 1u);
                    } else if (ly == 0u) {
                        uint jj = iy >> (20 - ky);
                        if ((jj << (20 - ky)) == iy) yisint = 2u - (jj & 1u);
                    }
                }
            }
            if (ly == 0u) {
                if (iy == 0x7FF00000u) {                            // y is +-inf
                    if (ix == 0x3FF00000u && lx == 0u) return DP_ONE;              // (-1)^+-inf = 1
                    if (ix >= 0x3FF00000u) return ((y >> 63) == 0ul) ? y : 0ul;    // |x| > 1
                    return ((y >> 63) != 0ul) ? (y ^ DP_SIGN) : 0ul;               // |x| < 1
                }
                if (iy == 0x3FF00000u) return ((y >> 63) != 0ul) ? d_div(DP_ONE, x) : x;
                if (hy == 0x40000000u) return d_mul(x, x);          // y = 2
                if (hy == 0x3FE00000u && (x >> 63) == 0ul) return d_sqrt(x);       // y = 1/2
            }
            ulong ax = x & 0x7FFFFFFFFFFFFFFFul;
            if (lx == 0u && (ix == 0x7FF00000u || ix == 0u || ix == 0x3FF00000u)) {
                ulong z = ax;                                       // x is +-0, +-inf or +-1
                if ((y >> 63) != 0ul) z = d_div(DP_ONE, z);
                if ((x >> 63) != 0ul) {
                    if (ix == 0x3FF00000u && yisint == 0u) return D_QNAN;          // (-1)^non-integer
                    if (yisint == 1u) z ^= DP_SIGN;
                }
                return z;
            }
            if ((x >> 63) != 0ul && yisint == 0u) return D_QNAN;    // negative base, fractional exponent
            ulong sgn = ((x >> 63) != 0ul && yisint == 1u) ? DP_SIGN : 0ul;
            ulong t1, t2;
            if (iy > 0x41E00000u) {                                 // |y| > 2^31
                if (iy > 0x43F00000u) {                             // |y| > 2^64: y is even, sign is +
                    if (ix <= 0x3FEFFFFFu) return ((y >> 63) != 0ul) ? D_INF : 0ul;
                    if (ix >= 0x3FF00000u) return ((y >> 63) == 0ul) ? D_INF : 0ul;
                }
                if (ix < 0x3FEFFFFFu) return sgn | (((y >> 63) != 0ul) ? D_INF : 0ul);
                if (ix > 0x3FF00000u) return sgn | (((y >> 63) == 0ul) ? D_INF : 0ul);
                // |x - 1| <= 2^-20 now, so four terms of the log series carry the pair.
                ulong t = d_sub(ax, DP_ONE);
                ulong w = d_mul(d_mul(t, t),
                                d_sub(DP_HALF, d_mul(t, d_sub(DP_THIRD, d_mul(t, DP_QUARTER)))));
                ulong u = d_mul(DP_IVLN2_H, t);
                ulong v = d_sub(d_mul(t, DP_IVLN2_L), d_mul(w, DP_IVLN2));
                t1 = dp_hi21(d_add(u, v));
                t2 = d_sub(v, d_sub(t1, u));
            } else {
                dp_log2_hl(ax, &t1, &t2);
            }
            // (y1 + y2)(t1 + t2), with y1 * t1 exact: 21 significant bits times 21.
            ulong y1 = dp_hi21(y);
            ulong p_l = d_add(d_mul(d_sub(y, y1), t1), d_mul(y, t2));
            ulong p_h = d_mul(y1, t1);
            ulong z = d_add(p_l, p_h);
            int jw = (int)dp_hiword(z);
            uint iw = (uint)z;
            if (jw >= 0x40900000) {                                 // z >= 1024: overflow
                if ((((uint)jw - 0x40900000u) | iw) != 0u) return sgn | D_INF;
                if (dp_lt(d_sub(z, p_h), d_add(p_l, DP_OVT))) return sgn | D_INF;
            } else if (((uint)jw & 0x7FFFFFFFu) >= 0x4090CC00u) {   // z <= -1075: underflow
                if ((((uint)jw - 0xC090CC00u) | iw) != 0u) return sgn;
                if (!dp_lt(d_sub(z, p_h), p_l)) return sgn;
            }
            // 2^(p_h + p_l): peel off the integer part n, then the fractional part through the series.
            uint iu = (uint)jw & 0x7FFFFFFFu;
            int kk = (int)(iu >> 20) - 1023;
            long n = 0;
            if (iu > 0x3FE00000u) {                                 // |z| > 1/2, so n = round(z)
                uint nn = (uint)jw + (0x00100000u >> (kk + 1));
                kk = (int)((nn & 0x7FFFFFFFu) >> 20) - 1023;
                ulong tn = dp_sethi(0ul, nn & ~(0x000FFFFFu >> kk));
                n = (long)(((nn & 0x000FFFFFu) | 0x00100000u) >> (20 - kk));
                if (jw < 0) n = -n;
                p_h = d_sub(p_h, tn);
            }
            ulong t = dp_hi21(d_add(p_l, p_h));
            ulong u = d_mul(t, DP_LG2_H);
            ulong v = d_add(d_mul(d_sub(p_l, d_sub(t, p_h)), DP_LG2), d_mul(t, DP_LG2_L));
            z = d_add(u, v);
            ulong w = d_sub(v, d_sub(z, u));
            t = d_mul(z, z);
            ulong q = d_sub(z, d_mul(t, \(horner(expPolyCoefficients, "t"))));
            ulong r = d_sub(d_div(d_mul(z, q), d_sub(q, DP_TWO)), d_add(w, d_mul(z, w)));
            z = d_sub(DP_ONE, d_sub(r, z));
            int je = (int)dp_hiword(z) + (int)(n << 20);
            z = ((je >> 20) <= 0) ? dt_ldexp(z, n) : dp_sethi(z, (uint)je);
            return z ^ sgn;
        }
        """
    }()
}
