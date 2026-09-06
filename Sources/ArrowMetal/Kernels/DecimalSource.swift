import Foundation

/// Metal Shading Language for Arrow decimal columns.
///
/// A decimal element is a fixed-width two's-complement little-endian integer: 16 bytes for `decimal128`,
/// 32 bytes for `decimal256`. Metal has no 128-bit type, so every value travels as `DL` 64-bit limbs
/// (`DL` = 2 or 4) in a `dec_t` struct and the arithmetic is written out limb by limb: carries come from
/// unsigned compares (`s < a` means the addition wrapped), the products use `mulhi`/`*` on the limbs, and
/// division is a restoring long division over the 64·DL bits. Everything wraps modulo 2^(64·DL), which is
/// Arrow's unchecked behaviour.
///
/// The generated source is one library per limb count, compiled and cached at runtime like every other
/// kernel family in this package.
enum DecimalSource {

    /// Limb-level helpers shared by every decimal kernel.
    static func core(limbs: Int) -> String { """

    #define DL \(limbs)
    struct dec_t { ulong w[DL]; };

    inline dec_t dec_zero() { dec_t r; for (int k = 0; k < DL; k++) r.w[k] = 0ul; return r; }
    inline dec_t dec_one() { dec_t r = dec_zero(); r.w[0] = 1ul; return r; }
    inline dec_t dec_load(device const ulong* p, uint i) {
        dec_t r; for (int k = 0; k < DL; k++) r.w[k] = p[(ulong)i * DL + k]; return r;
    }
    inline dec_t dec_load_c(constant ulong* p) { dec_t r; for (int k = 0; k < DL; k++) r.w[k] = p[k]; return r; }
    inline void dec_store(device ulong* p, uint i, dec_t v) {
        for (int k = 0; k < DL; k++) p[(ulong)i * DL + k] = v.w[k];
    }
    inline bool dec_is_zero(dec_t a) { for (int k = 0; k < DL; k++) if (a.w[k] != 0ul) return false; return true; }
    inline bool dec_is_neg(dec_t a) { return (long)a.w[DL - 1] < 0L; }

    // Ripple-carry add. Each limb produces a carry when the unsigned sum wraps below either addend.
    inline dec_t dec_add(dec_t a, dec_t b) {
        dec_t r; ulong carry = 0ul;
        for (int k = 0; k < DL; k++) {
            ulong s = a.w[k] + b.w[k];
            ulong c1 = (s < a.w[k]) ? 1ul : 0ul;
            ulong s2 = s + carry;
            ulong c2 = (s2 < s) ? 1ul : 0ul;
            r.w[k] = s2;
            carry = c1 | c2;
        }
        return r;
    }
    // Two's complement negation: ~a + 1, carry rippling only while a limb is all ones.
    inline dec_t dec_neg(dec_t a) {
        dec_t r; ulong carry = 1ul;
        for (int k = 0; k < DL; k++) {
            ulong t = ~a.w[k] + carry;
            carry = (carry != 0ul && t == 0ul) ? 1ul : 0ul;
            r.w[k] = t;
        }
        return r;
    }
    inline dec_t dec_sub(dec_t a, dec_t b) { return dec_add(a, dec_neg(b)); }
    inline dec_t dec_abs(dec_t a) { return dec_is_neg(a) ? dec_neg(a) : a; }

    inline bool dec_eq(dec_t a, dec_t b) { for (int k = 0; k < DL; k++) if (a.w[k] != b.w[k]) return false; return true; }
    // Unsigned compare, most significant limb first.
    inline bool dec_ult(dec_t a, dec_t b) {
        for (int k = DL - 1; k >= 0; k--) if (a.w[k] != b.w[k]) return a.w[k] < b.w[k];
        return false;
    }
    // Signed compare: the top limb is signed, everything below it unsigned.
    inline bool dec_lt(dec_t a, dec_t b) {
        if (a.w[DL - 1] != b.w[DL - 1]) return (long)a.w[DL - 1] < (long)b.w[DL - 1];
        for (int k = DL - 2; k >= 0; k--) if (a.w[k] != b.w[k]) return a.w[k] < b.w[k];
        return false;
    }
    inline dec_t dec_min(dec_t a, dec_t b) { return dec_lt(a, b) ? a : b; }
    inline dec_t dec_max(dec_t a, dec_t b) { return dec_lt(a, b) ? b : a; }

    inline dec_t dec_shl1(dec_t a) {
        dec_t r; ulong carry = 0ul;
        for (int k = 0; k < DL; k++) { ulong v = a.w[k]; r.w[k] = (v << 1) | carry; carry = v >> 63; }
        return r;
    }
    inline bool dec_bit(dec_t a, int i) { return ((a.w[i >> 6] >> (i & 63)) & 1ul) != 0ul; }

    // Schoolbook multiply keeping the low DL limbs (the product wraps, as Arrow's unchecked multiply does).
    inline dec_t dec_mul(dec_t a, dec_t b) {
        dec_t r = dec_zero();
        for (int i = 0; i < DL; i++) {
            ulong carry = 0ul;
            for (int j = 0; i + j < DL; j++) {
                ulong lo = a.w[i] * b.w[j];
                ulong hi = mulhi(a.w[i], b.w[j]);
                ulong s = r.w[i + j] + carry;
                ulong c0 = (s < carry) ? 1ul : 0ul;
                ulong s2 = s + lo;
                ulong c1 = (s2 < lo) ? 1ul : 0ul;
                r.w[i + j] = s2;
                carry = hi + c0 + c1;
            }
        }
        return r;
    }

    // Unsigned restoring long division, one quotient bit per iteration (64*DL iterations).
    // Both operands must be non-negative magnitudes; `d` must not be zero.
    inline dec_t dec_divmod(dec_t a, dec_t d, thread dec_t& rem) {
        dec_t q = dec_zero(), r = dec_zero();
        for (int i = 64 * DL - 1; i >= 0; i--) {
            r = dec_shl1(r);
            if (dec_bit(a, i)) r.w[0] |= 1ul;
            if (!dec_ult(r, d)) { r = dec_sub(r, d); q.w[i >> 6] |= (1ul << (i & 63)); }
        }
        rem = r;
        return q;
    }
    """ }

    /// Compare, reduce, arithmetic and rescale kernels.
    static func source(limbs: Int) -> String {
        var s = KernelSource.prelude + core(limbs: limbs)
        // Comparisons: one thread per 32-bit word of the output bitmap, as everywhere else in the package.
        let ops: [(String, String)] = [("eq", "dec_eq(x, y)"), ("ne", "!dec_eq(x, y)"), ("lt", "dec_lt(x, y)"),
                                       ("le", "!dec_lt(y, x)"), ("gt", "dec_lt(y, x)"), ("ge", "!dec_lt(x, y)")]
        for (name, expr) in ops {
            s += """

            kernel void dec_cmp_scalar_\(name)(device const ulong* a [[buffer(0)]],
                                               constant ulong* scalar [[buffer(1)]],
                                               device const uint* nPtr [[buffer(2)]],
                                               device uint* out [[buffer(3)]],
                                               uint w [[thread_position_in_grid]]) {
                uint n = *nPtr;
                uint base = w * 32u;
                if (base >= n) return;
                uint limit = min(32u, n - base);
                dec_t y = dec_load_c(scalar);
                uint bits = 0u;
                for (uint j = 0; j < limit; j++) { dec_t x = dec_load(a, base + j); if (\(expr)) bits |= (1u << j); }
                out[w] = bits;
            }
            kernel void dec_cmp_array_\(name)(device const ulong* a [[buffer(0)]],
                                              device const ulong* b [[buffer(1)]],
                                              device const uint* nPtr [[buffer(2)]],
                                              device uint* out [[buffer(3)]],
                                              uint w [[thread_position_in_grid]]) {
                uint n = *nPtr;
                uint base = w * 32u;
                if (base >= n) return;
                uint limit = min(32u, n - base);
                uint bits = 0u;
                for (uint j = 0; j < limit; j++) {
                    dec_t x = dec_load(a, base + j), y = dec_load(b, base + j);
                    if (\(expr)) bits |= (1u << j);
                }
                out[w] = bits;
            }
            """
        }
        // Reductions: per-thread 128/256-bit accumulate, threadgroup tree, one partial per threadgroup.
        func reduce(_ name: String, _ initVal: String, _ combine: String) -> String { """

        kernel void dec_reduce_\(name)(device const ulong* vals [[buffer(0)]],
                                       device const uchar* validity [[buffer(1)]],
                                       device const uint* nPtr [[buffer(2)]],
                                       constant uint& hasValidity [[buffer(3)]],
                                       device ulong* partials [[buffer(4)]],
                                       device uint* counts [[buffer(5)]],
                                       uint gid [[thread_position_in_grid]],
                                       uint lid [[thread_index_in_threadgroup]],
                                       uint tgid [[threadgroup_position_in_grid]],
                                       uint gridSize [[threads_per_grid]]) {
            threadgroup ulong sdata[TG * DL];
            threadgroup uint scount[TG];
            uint n = *nPtr;
            dec_t acc = \(initVal);
            uint cnt = 0u;
            for (uint i = gid; i < n; i += gridSize) {
                if (hasValidity != 0u && !bit_get(validity, i)) continue;
                dec_t v = dec_load(vals, i);
                acc = \(combine);
                cnt++;
            }
            for (int k = 0; k < DL; k++) sdata[lid * DL + k] = acc.w[k];
            scount[lid] = cnt;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = TG / 2u; s > 0u; s >>= 1) {
                if (lid < s) {
                    dec_t acc2, v;
                    for (int k = 0; k < DL; k++) { acc2.w[k] = sdata[lid * DL + k]; v.w[k] = sdata[(lid + s) * DL + k]; }
                    dec_t acc = acc2;
                    dec_t r = \(combine);
                    for (int k = 0; k < DL; k++) sdata[lid * DL + k] = r.w[k];
                    scount[lid] += scount[lid + s];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lid == 0u) {
                for (int k = 0; k < DL; k++) partials[tgid * DL + k] = sdata[k];
                counts[tgid] = scount[0];
            }
        }
        """ }
        // Sentinels: the largest / smallest representable signed value, so a thread that saw nothing loses.
        s += """

        inline dec_t dec_min_init() { dec_t r; for (int k = 0; k < DL; k++) r.w[k] = ~0ul; r.w[DL - 1] = 0x7FFFFFFFFFFFFFFFul; return r; }
        inline dec_t dec_max_init() { dec_t r = dec_zero(); r.w[DL - 1] = 0x8000000000000000ul; return r; }
        """
        s += reduce("sum", "dec_zero()", "dec_add(acc, v)")
        s += reduce("min", "dec_min_init()", "dec_min(acc, v)")
        s += reduce("max", "dec_max_init()", "dec_max(acc, v)")
        // Element-wise arithmetic. op: 0 add, 1 subtract, 2 multiply. flags bit 0: b is a broadcast scalar.
        s += """

        kernel void dec_binary(device const ulong* a [[buffer(0)]],
                               device const ulong* b [[buffer(1)]],
                               constant ulong* scalar [[buffer(2)]],
                               device const uint* nPtr [[buffer(3)]],
                               constant uint& op [[buffer(4)]],
                               constant uint& flags [[buffer(5)]],
                               device ulong* out [[buffer(6)]],
                               uint i [[thread_position_in_grid]]) {
            if (i >= *nPtr) return;
            dec_t x = dec_load(a, i);
            dec_t y = (flags & 1u) ? dec_load_c(scalar) : dec_load(b, i);
            dec_t r = (op == 0u) ? dec_add(x, y) : ((op == 1u) ? dec_sub(x, y) : dec_mul(x, y));
            dec_store(out, i, r);
        }

        // op: 0 negate, 1 abs.
        kernel void dec_unary(device const ulong* a [[buffer(0)]],
                              device const uint* nPtr [[buffer(1)]],
                              constant uint& op [[buffer(2)]],
                              device ulong* out [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
            if (i >= *nPtr) return;
            dec_t x = dec_load(a, i);
            dec_store(out, i, (op == 0u) ? dec_neg(x) : dec_abs(x));
        }

        kernel void dec_sign(device const ulong* a [[buffer(0)]],
                             device const uint* nPtr [[buffer(1)]],
                             device int* out [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
            if (i >= *nPtr) return;
            dec_t x = dec_load(a, i);
            out[i] = dec_is_zero(x) ? 0 : (dec_is_neg(x) ? -1 : 1);
        }

        // Scale up: multiply every value by 10^(target - scale), which cannot lose information.
        kernel void dec_scale_up(device const ulong* a [[buffer(0)]],
                                 constant ulong* mul [[buffer(1)]],
                                 device const uint* nPtr [[buffer(2)]],
                                 device ulong* out [[buffer(3)]],
                                 uint i [[thread_position_in_grid]]) {
            if (i >= *nPtr) return;
            dec_store(out, i, dec_mul(dec_load(a, i), dec_load_c(mul)));
        }

        // Scale down: divide the magnitude by 10^(scale - target) and apply the rounding mode.
        // mode: 0 half away from zero, 1 ceil (toward +inf), 2 floor (toward -inf), 3 truncate (toward zero).
        kernel void dec_scale_down(device const ulong* a [[buffer(0)]],
                                   constant ulong* divisor [[buffer(1)]],
                                   device const uint* nPtr [[buffer(2)]],
                                   constant uint& mode [[buffer(3)]],
                                   device ulong* out [[buffer(4)]],
                                   uint i [[thread_position_in_grid]]) {
            if (i >= *nPtr) return;
            dec_t v = dec_load(a, i);
            bool neg = dec_is_neg(v);
            dec_t m = neg ? dec_neg(v) : v;
            dec_t d = dec_load_c(divisor);
            dec_t rem;
            dec_t q = dec_divmod(m, d, rem);
            bool bump = false;
            if (mode == 0u) { bump = !dec_ult(dec_shl1(rem), d); }
            else if (mode == 1u) { bump = !neg && !dec_is_zero(rem); }
            else if (mode == 2u) { bump = neg && !dec_is_zero(rem); }
            if (bump) q = dec_add(q, dec_one());
            dec_store(out, i, neg ? dec_neg(q) : q);
        }
        """
        return s
    }

    /// Byte-width-generic gather (`take`) plus the index generator `filter` and `slice` build on.
    /// One thread per output element; out-of-range indices raise the error flag, as the primitive `take` does.
    static func gather(limbs: Int, I: String) -> String { KernelSource.prelude + core(limbs: limbs) + """

    kernel void dec_iota(device int* out [[buffer(0)]],
                         device const uint* nPtr [[buffer(1)]],
                         constant uint& base [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        out[i] = (int)(base + i);
    }

    // flags: bit0 = source has validity, bit1 = indices have validity.
    kernel void dec_take(device const ulong* vals [[buffer(0)]],
                         device const uchar* validity [[buffer(1)]],
                         device const \(I)* idx [[buffer(2)]],
                         device const uchar* idxValidity [[buffer(3)]],
                         constant uint& n [[buffer(4)]],
                         device const uint* srcLenPtr [[buffer(5)]],
                         constant uint& flags [[buffer(6)]],
                         device ulong* out [[buffer(7)]],
                         device uchar* outValidBytes [[buffer(8)]],
                         device atomic_uint* errorFlag [[buffer(9)]],
                         uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        uint srcLen = *srcLenPtr;
        if ((flags & 2u) && !bit_get(idxValidity, i)) { dec_store(out, i, dec_zero()); outValidBytes[i] = 0; return; }
        long j = (long)idx[i];
        if (j < 0 || j >= (long)srcLen) {
            atomic_store_explicit(errorFlag, 1u, memory_order_relaxed);
            dec_store(out, i, dec_zero()); outValidBytes[i] = 0; return;
        }
        dec_store(out, i, dec_load(vals, (uint)j));
        outValidBytes[i] = (flags & 1u) ? (bit_get(validity, (uint)j) ? 1 : 0) : 1;
    }
    """ }
}
