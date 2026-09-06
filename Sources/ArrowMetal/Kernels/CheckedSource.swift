import Foundation

/// MSL for Arrow's *checked* arithmetic: the kernels that decide whether an element-wise op would have
/// been out of range, so the Swift wrapper can raise instead of returning the wrapped answer.
///
/// **The flag mechanism.** A checked op is the unchecked kernel plus one read-only *check* pass over the
/// same inputs. The check pass writes nothing except, on the rare failing element, into a small shared
/// buffer of eight `uint32`s:
///
///   * word 0 is a bitmask of the failure kinds seen (`CHK_*` below), set with one `atomic_fetch_or`;
///   * words 1...7 hold the first offending row index of each kind, kept with one `atomic_fetch_min`
///     (they start at `0xFFFFFFFF`, which reads back as "no index").
///
/// A conforming element executes no atomic at all, so the common path costs one extra streaming read of
/// the inputs and nothing else. Nulls never reach the predicate, so a null can never raise.
///
/// Splitting the check from the arithmetic buys two guarantees that a fused kernel would have to be
/// argued for: a checked op's values are *bit-identical* to the unchecked op's (it is literally the same
/// kernel), and every unchecked kernel — including the software binary64 ones — gains a checked twin
/// without being touched.
///
/// **What each kind means** (the message text is Arrow's, so a caller can match on it):
///
/// | bit | name          | Arrow message                                            |
/// |----:|---------------|----------------------------------------------------------|
/// |   0 | `OVERFLOW`    | `overflow`                                               |
/// |   1 | `DIV_ZERO`    | `divide by zero`                                         |
/// |   2 | `SQRT_NEG`    | `square root of negative number`                         |
/// |   3 | `LOG_NEG`     | `logarithm of negative number`                           |
/// |   4 | `LOG_ZERO`    | `logarithm of zero`                                      |
/// |   5 | `SHIFT`       | `shift amount must be >= 0 and less than precision of type` |
/// |   6 | `NEG_POW`     | `integers to negative integer powers are not allowed`    |
///
/// Every predicate returns `0` for "fine" or `1 + kind` for a failure, so one `if` in the kernel covers
/// every op.
enum CheckedSource {
    /// The element-type flavours the generator emits. Float columns can only fail on a domain error
    /// (divide by zero, a negative root or logarithm); integer columns can only fail on range.
    enum Kind { case signedInt, unsignedInt, float32, float64 }

    /// Failure kinds, numbered to match the `CHK_*` defines and the flag-word bit layout.
    enum Failure: Int, CaseIterable, Sendable {
        case overflow = 0, divideByZero, sqrtNegative, logNegative, logZero, shiftAmount, negativeExponent

        /// Arrow's own wording for this failure, reused verbatim so callers can match pyarrow's text.
        var message: String {
            switch self {
            case .overflow: return "overflow"
            case .divideByZero: return "divide by zero"
            case .sqrtNegative: return "square root of negative number"
            case .logNegative: return "logarithm of negative number"
            case .logZero: return "logarithm of zero"
            case .shiftAmount: return "shift amount must be >= 0 and less than precision of type"
            case .negativeExponent: return "integers to negative integer powers are not allowed"
            }
        }
    }

    /// `uint32` slots in the flag buffer: one bitmask plus one first-index per failure kind.
    static var flagWords: Int { 1 + Failure.allCases.count }

    /// Shared MSL: the flag words and the one function that raises into them.
    static let prelude = """

    #define CHK_OVERFLOW 0u
    #define CHK_DIV_ZERO 1u
    #define CHK_SQRT_NEG 2u
    #define CHK_LOG_NEG  3u
    #define CHK_LOG_ZERO 4u
    #define CHK_SHIFT    5u
    #define CHK_NEG_POW  6u
    // One `or` for the kind and one `min` for the row: no atomic runs unless an element actually fails.
    inline void chk_raise(device atomic_uint* flags, uint kind, uint i) {
        atomic_fetch_or_explicit(&flags[0], 1u << kind, memory_order_relaxed);
        atomic_fetch_min_explicit(&flags[1u + kind], i, memory_order_relaxed);
    }
    """

    // MARK: - Predicates

    /// Integer predicates, written on the unsigned type of the same width because C defines the wrap
    /// there. `w` is the bit width; the shift bound is Arrow's "precision", one less than the width on a
    /// signed type, matching Arrow's own `shift_left_checked`.
    private static func integerPredicates(T: String, U: String, width w: Int, signed: Bool) -> String {
        let tmin = signed ? "(\(T))((\(U))1 << \(w - 1))" : "(\(T))0"
        let minusOne = "(\(T))(\(U))(~(\(U))0)"
        let precision = signed ? w - 1 : w
        let h = w / 2
        // Whether an unsigned product leaves the type, decided by long multiplication on half-width
        // digits so that every intermediate stays inside `U`.
        //
        // Every textbook spelling of this check miscompiles on 16-bit columns with this Metal front end:
        // `(uint)a * (uint)b > 65535`, `(a * b) / a != b` and even `mulhi(a, b) != 0` all report an
        // overflow for *every* product, `1 * 1` included, unless the intermediate is forced into memory.
        // The front end evidently pattern-matches the widening-multiply-overflow idiom and lowers it
        // wrongly at that width. Half-width digits give it nothing to match, and `CheckedTests` pins the
        // result against a CPU oracle at all eight integer widths.
        var s = """
        inline bool chk_mul_wraps(\(U) x, \(U) y) {
            \(U) mask = (\(U))(((\(U))1 << \(h)) - (\(U))1);
            \(U) ah = (\(U))(x >> \(h)), al = (\(U))(x & mask);
            \(U) bh = (\(U))(y >> \(h)), bl = (\(U))(y & mask);
            if (ah != (\(U))0 && bh != (\(U))0) return true;     // the ah*bh digit alone is out of range
            \(U) mid = (\(U))((\(U))(ah * bl) + (\(U))(al * bh));  // exactly one term is non-zero here
            if ((\(U))(mid >> \(h)) != (\(U))0) return true;
            \(U) shifted = (\(U))(mid << \(h));
            return (\(U))(al * bl) > (\(U))((\(U))(~(\(U))0) - shifted);
        }

        """
        if signed {
            s += """
            inline uint chk_add(\(T) a, \(T) b) {
                \(U) r = (\(U))((\(U))a + (\(U))b);
                return ((((r ^ (\(U))a) & (r ^ (\(U))b)) >> \(w - 1)) & (\(U))1) ? 1u + CHK_OVERFLOW : 0u;
            }
            inline uint chk_sub(\(T) a, \(T) b) {
                \(U) r = (\(U))((\(U))a - (\(U))b);
                return (((((\(U))a ^ (\(U))b) & (r ^ (\(U))a)) >> \(w - 1)) & (\(U))1) ? 1u + CHK_OVERFLOW : 0u;
            }

            """
            s += """
            // Magnitudes first, then the unsigned range test, then the signed bound: the negative side
            // reaches one further than the positive one. |T.min| is representable in U, so the negation
            // below is exact for every input.
            inline uint chk_mul(\(T) a, \(T) b) {
                if (a == (\(T))0 || b == (\(T))0) return 0u;
                \(U) ua = (\(U))(a < (\(T))0 ? (\(T))((\(U))0 - (\(U))a) : a);
                \(U) ub = (\(U))(b < (\(T))0 ? (\(T))((\(U))0 - (\(U))b) : b);
                if (chk_mul_wraps(ua, ub)) return 1u + CHK_OVERFLOW;
                \(U) p = (\(U))(ua * ub);
                \(U) bound = (\(U))((\(U))1 << \(w - 1));
                if ((a < (\(T))0) == (b < (\(T))0)) bound = (\(U))(bound - (\(U))1);
                return (p > bound) ? 1u + CHK_OVERFLOW : 0u;
            }
            inline uint chk_div(\(T) a, \(T) b) {
                if (b == (\(T))0) return 1u + CHK_DIV_ZERO;
                if (a == \(tmin) && b == \(minusOne)) return 1u + CHK_OVERFLOW;
                return 0u;
            }
            inline uint chk_negate(\(T) a) { return (a == \(tmin)) ? 1u + CHK_OVERFLOW : 0u; }
            inline uint chk_abs(\(T) a) { return (a == \(tmin)) ? 1u + CHK_OVERFLOW : 0u; }
            inline uint chk_shl(\(T) a, \(T) b) {
                long k = (long)b;
                return (k < 0 || k >= \(precision)) ? 1u + CHK_SHIFT : 0u;
            }

            """
        } else {
            s += """
            inline uint chk_add(\(T) a, \(T) b) { return ((\(T))(a + b) < a) ? 1u + CHK_OVERFLOW : 0u; }
            inline uint chk_sub(\(T) a, \(T) b) { return (a < b) ? 1u + CHK_OVERFLOW : 0u; }

            """
            s += """
            inline uint chk_mul(\(T) a, \(T) b) { return chk_mul_wraps(a, b) ? 1u + CHK_OVERFLOW : 0u; }
            inline uint chk_div(\(T) a, \(T) b) { return (b == (\(T))0) ? 1u + CHK_DIV_ZERO : 0u; }
            // Arrow has no unsigned `negate_checked` kernel at all; ArrowMetal defines it as "any non-zero
            // value overflows", which is the only answer a modular negation could report.
            inline uint chk_negate(\(T) a) { return (a != (\(T))0) ? 1u + CHK_OVERFLOW : 0u; }
            inline uint chk_abs(\(T) a) { return 0u; }
            inline uint chk_shl(\(T) a, \(T) b) {
                ulong k = (ulong)b;
                return (k >= \(precision)ul) ? 1u + CHK_SHIFT : 0u;
            }

            """
        }
        // Arrow's shifts check the amount only; a value shifted off the top is not an error.
        s += """
        inline uint chk_shr(\(T) a, \(T) b) { return chk_shl(a, b); }
        // Repeated squaring, flagging exactly the multiplies the unchecked kernel would wrap. The base
        // is only squared while another exponent bit remains, so a final harmless square never fires.
        inline uint chk_power(\(T) a, \(T) b) {
            \(signed ? "if ((long)b < 0) return 1u + CHK_NEG_POW;" : "")
            ulong e = (ulong)\(signed ? "(long)b" : "b");
            \(T) r = (\(T))1, base = a;
            while (e != 0ul) {
                if (e & 1ul) {
                    uint k = chk_mul(r, base);
                    if (k) return k;
                    r = (\(T))((\(U))r * (\(U))base);
                }
                e >>= 1;
                if (e != 0ul) {
                    uint k = chk_mul(base, base);
                    if (k) return k;
                    base = (\(T))((\(U))base * (\(U))base);
                }
            }
            return 0u;
        }

        """
        return s
    }

    /// Float32 predicates, decided on the bit pattern rather than on `<` so that a denormal (which Apple
    /// GPUs flush to zero in arithmetic) is still classified by its true sign.
    private static let float32Predicates = """
    inline uint fc_bits(float x) { return as_type<uint>(x); }
    inline bool fc_nan(uint b) { return (b & 0x7FFFFFFFu) > 0x7F800000u; }
    inline bool fc_zero(uint b) { return (b & 0x7FFFFFFFu) == 0u; }
    inline bool fc_neg(uint b) { return (b >> 31) != 0u && !fc_zero(b) && !fc_nan(b); }
    inline int fc_key(uint b) { if (fc_zero(b)) return 0; int k = (int)b; return k ^ (int)(((uint)(k >> 31)) >> 1); }

    inline uint chk_div(float a, float b) { return fc_zero(fc_bits(b)) ? 1u + CHK_DIV_ZERO : 0u; }
    inline uint chk_sqrt(float a) { return fc_neg(fc_bits(a)) ? 1u + CHK_SQRT_NEG : 0u; }
    inline uint chk_log(float a) {
        uint b = fc_bits(a);
        if (fc_zero(b)) return 1u + CHK_LOG_ZERO;
        if (fc_neg(b)) return 1u + CHK_LOG_NEG;
        return 0u;
    }
    // log1p(x) is ln(1 + x): the domain boundary sits at -1, not at 0.
    inline uint chk_log1p(float a) {
        uint b = fc_bits(a);
        if (fc_nan(b)) return 0u;
        if (b == 0xBF800000u) return 1u + CHK_LOG_ZERO;
        if (fc_key(b) < fc_key(0xBF800000u)) return 1u + CHK_LOG_NEG;
        return 0u;
    }
    // logb(x, base) is ln(x) / ln(base): both operands must be in the log domain.
    inline uint chk_logb(float a, float b) {
        uint k = chk_log(a);
        if (k) return k;
        return chk_log(b);
    }

    """

    /// Float64 predicates over raw binary64 patterns (Metal has no `double`).
    private static let float64Predicates = """
    inline bool dc_neg(ulong a) { return (a >> 63) != 0ul && !d_is_zero(a) && !d_is_nan(a); }

    inline uint chk_div(ulong a, ulong b) { return d_is_zero(b) ? 1u + CHK_DIV_ZERO : 0u; }
    inline uint chk_sqrt(ulong a) { return dc_neg(a) ? 1u + CHK_SQRT_NEG : 0u; }
    inline uint chk_log(ulong a) {
        if (d_is_zero(a)) return 1u + CHK_LOG_ZERO;
        if (dc_neg(a)) return 1u + CHK_LOG_NEG;
        return 0u;
    }
    inline uint chk_log1p(ulong a) {
        if (d_is_nan(a)) return 0u;
        if (a == 0xBFF0000000000000ul) return 1u + CHK_LOG_ZERO;
        if (d_key((long)a) < d_key((long)0xBFF0000000000000ul)) return 1u + CHK_LOG_NEG;
        return 0u;
    }
    inline uint chk_logb(ulong a, ulong b) {
        uint k = chk_log(a);
        if (k) return k;
        return chk_log(b);
    }

    """

    // MARK: - Kernels

    /// Kernel bodies. `flagsIn` bit 0 says the left validity bitmap is present, bit 1 the right one.
    ///
    /// Four elements per thread, not one. The predicate is trivial and the pass is memory bound, so at
    /// 50M rows a one-element-per-thread grid spends most of its time launching threads: the check cost
    /// dropped from 2.5 ms to 0.7 ms per 200 MB column when this went 4-wide, which is what the
    /// unchecked arithmetic kernels already do. The block always starts on a multiple of four, so the
    /// four validity bits sit inside one byte and cost one load between them.
    private static let blockPrelude = """
    // The four validity bits for the block starting at `i` (a multiple of 4), or all four set when the
    // column has no bitmap.
    inline uint chk_vbits(device const uchar* v, uint present, uint i) {
        return present ? (uint)((v[i >> 3] >> (i & 7u)) & 0xFu) : 0xFu;
    }

    """

    private static func kernels(T: String, unary: [String], binary: [String], scan: Bool) -> String {
        var s = blockPrelude
        for op in unary {
            s += """
            kernel void chk_unary_\(op)(device const \(T)* a [[buffer(0)]], device const uchar* va [[buffer(1)]],
                                        device const uint* nPtr [[buffer(2)]], constant uint& flagsIn [[buffer(3)]],
                                        device atomic_uint* flags [[buffer(4)]], uint t [[thread_position_in_grid]]) {
                uint n = *nPtr;
                uint i = t * 4u;
                if (i >= n) return;
                uint lim = min(4u, n - i);
                uint vb = chk_vbits(va, flagsIn & 1u, i);
                for (uint j = 0; j < lim; j++) {
                    if (!(vb & (1u << j))) continue;
                    uint k = chk_\(op)(a[i + j]);
                    if (k) chk_raise(flags, k - 1u, i + j);
                }
            }

            """
        }
        for op in binary {
            s += """
            kernel void chk_scalar_\(op)(device const \(T)* a [[buffer(0)]], constant \(T)& scalar [[buffer(1)]],
                                         device const uchar* va [[buffer(2)]], device const uint* nPtr [[buffer(3)]],
                                         constant uint& flagsIn [[buffer(4)]], device atomic_uint* flags [[buffer(5)]],
                                         uint t [[thread_position_in_grid]]) {
                uint n = *nPtr;
                uint i = t * 4u;
                if (i >= n) return;
                uint lim = min(4u, n - i);
                uint vb = chk_vbits(va, flagsIn & 1u, i);
                for (uint j = 0; j < lim; j++) {
                    if (!(vb & (1u << j))) continue;
                    uint k = chk_\(op)(a[i + j], scalar);
                    if (k) chk_raise(flags, k - 1u, i + j);
                }
            }
            kernel void chk_array_\(op)(device const \(T)* a [[buffer(0)]], device const \(T)* b [[buffer(1)]],
                                        device const uchar* va [[buffer(2)]], device const uchar* vb_ [[buffer(3)]],
                                        device const uint* nPtr [[buffer(4)]], constant uint& flagsIn [[buffer(5)]],
                                        device atomic_uint* flags [[buffer(6)]], uint t [[thread_position_in_grid]]) {
                uint n = *nPtr;
                uint i = t * 4u;
                if (i >= n) return;
                uint lim = min(4u, n - i);
                uint bits = chk_vbits(va, flagsIn & 1u, i) & chk_vbits(vb_, flagsIn & 2u, i);
                for (uint j = 0; j < lim; j++) {
                    if (!(bits & (1u << j))) continue;
                    uint k = chk_\(op)(a[i + j], b[i + j]);
                    if (k) chk_raise(flags, k - 1u, i + j);
                }
            }

            """
        }
        guard scan else { return s }
        // The cumulative scan reassociates, so checking inside it would flag sums of interior ranges that
        // a sequential scan never forms. Instead the finished running values are verified against the
        // sequential recurrence: out[i] must be out[i - 1] combined with vals[i], in range. out[i - 1] is
        // the running value even where row i - 1 is null, which is exactly the value Arrow would carry.
        for (name, fn) in [("cum_add", "chk_add"), ("cum_mul", "chk_mul")] {
            s += """
            kernel void chk_\(name)(device const \(T)* vals [[buffer(0)]], device const \(T)* out [[buffer(1)]],
                                    device const uchar* va [[buffer(2)]], device const uint* nPtr [[buffer(3)]],
                                    constant uint& flagsIn [[buffer(4)]], device atomic_uint* flags [[buffer(5)]],
                                    uint t [[thread_position_in_grid]]) {
                uint n = *nPtr;
                uint i = t * 4u;
                if (i >= n) return;
                uint lim = min(4u, n - i);
                uint vb = chk_vbits(va, flagsIn & 1u, i);
                for (uint j = 0; j < lim; j++) {
                    uint r = i + j;
                    if (r == 0u || !(vb & (1u << j))) continue;
                    uint k = \(fn)(out[r - 1u], vals[r]);
                    if (k) chk_raise(flags, k - 1u, r);
                }
            }

            """
        }
        s += """
        // pairwise_diff: out[i] = vals[i] - vals[i - period], only where both rows exist and are valid.
        kernel void chk_pairwise(device const \(T)* vals [[buffer(0)]], device const uchar* va [[buffer(1)]],
                                 device const uint* nPtr [[buffer(2)]], constant int& period [[buffer(3)]],
                                 constant uint& flagsIn [[buffer(4)]], device atomic_uint* flags [[buffer(5)]],
                                 uint t [[thread_position_in_grid]]) {
            uint n = *nPtr;
            uint i = t * 4u;
            if (i >= n) return;
            uint lim = min(4u, n - i);
            uint vb = chk_vbits(va, flagsIn & 1u, i);
            for (uint q = 0; q < lim; q++) {
                uint r = i + q;
                if (!(vb & (1u << q))) continue;
                long j = (long)r - (long)period;
                if (j < 0 || j >= (long)n) continue;
                if ((flagsIn & 1u) && !bit_get(va, (uint)j)) continue;
                uint k = chk_sub(vals[r], vals[(uint)j]);
                if (k) chk_raise(flags, k - 1u, r);
            }
        }

        """
        return s
    }

    // MARK: - Entry point

    /// Generated check kernels for one element type. Float columns get only the domain checks: Arrow's
    /// checked float `add`/`subtract`/`multiply`/`power`/`negate`/`abs` never raise (an overflow to
    /// infinity and a NaN are both ordinary results), so those need no kernel at all.
    static func source(T: String, U: String, width: Int, kind: Kind) -> String {
        var s = KernelSource.prelude + prelude + "\n"
        switch kind {
        case .signedInt, .unsignedInt:
            s += integerPredicates(T: T, U: U, width: width, signed: kind == .signedInt)
            s += kernels(T: T, unary: ["negate", "abs"],
                         binary: ["add", "sub", "mul", "div", "power", "shl", "shr"], scan: true)
        case .float32:
            s += float32Predicates
            s += kernels(T: "float", unary: ["sqrt", "log", "log1p"], binary: ["div", "logb"], scan: false)
        case .float64:
            s += DoubleMath.msl + "\n" + float64Predicates
            s += kernels(T: "ulong", unary: ["sqrt", "log", "log1p"], binary: ["div", "logb"], scan: false)
        }
        return s
    }
}
