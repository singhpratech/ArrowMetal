import Foundation

/// MSL for Arrow's trigonometric, inverse-trigonometric and hyperbolic functions.
///
/// Two flavours are generated:
///
///   * **float32** — the MSL library functions, compiled with `mathMode = .safe`. `sin`, `cos` and
///     `tan` use the `precise::` namespace explicitly, and `asin` / `acos` / `atan` the accurate
///     default overloads (with fast math off, those *are* the precise ones). The six hyperbolics and
///     `atan2` are written out here instead, because Metal's own are inaccurate or wrong on the
///     special values — see `float32Hyperbolics` for the measurements.
///   * **float64** — Metal has no `double`, so values travel as raw `ulong` bit patterns and every
///     function is evaluated in **software binary64** on top of the correctly rounded `d_add`,
///     `d_sub`, `d_mul` and `d_div` in `DoubleMath.swift`. This is a real binary64 implementation,
///     not a widened `float` one: see the accuracy notes below.
///
/// # The software binary64 elementary functions (`dm_*`)
///
/// Everything is built from four primitives, each argument-reduced and then evaluated with a Taylor
/// series whose truncation error is below 2⁻⁶⁵ on the reduced interval (so the series is not the
/// limiting factor — the rounding of the `d_*` operations is):
///
///   * `dm_exp(x)`  — `x = k·ln2 + r`, `|r| ≤ ln2/2`, `e^r` by 16 Taylor terms, then `ldexp` by `k`.
///     `ln2` is split into three 26-bit chunks so every `k·chunk` product is exact.
///   * `dm_log(x)`  — `x = m·2^k`, `m ∈ [√2/2, √2)`, `ln m = 2·atanh(s)` with `s = (m-1)/(m+1)`
///     and 12 series terms, then `k·ln2` added back from the same three chunks, smallest first.
///   * `dm_sin`/`dm_cos`/`dm_tan` — `x = n·(π/2) + r`, `|r| ≤ π/4`, by **Cody-Waite reduction against
///     π/2 split into sixteen 8-bit chunks** (128 bits of π/2). Each `n·chunk` product is exact while
///     `|n| < 2⁴⁵`, and each subtraction is exact by construction, so the reduced argument is exact to
///     about 2⁻⁸³ for `|x| ≤ 2⁴⁵·π/2 ≈ 5.5e13`. Above that the products stop being exact and the
///     reduction degrades linearly (absolute error in `r` of roughly `|x|·2⁻⁵³` radians); `|x| ≥ 2⁶²`
///     returns NaN rather than a meaningless number. `sin(r)` and `cos(r)` are 9- and 10-term series.
///   * `dm_atan(t)` — `|t| > 1` folds through `π/2 - atan(1/t)` and `t > tan(π/12)` through
///     `π/6 + atan((t√3-1)/(t+√3))`, leaving `|t| ≤ 0.268` for a 16-term series.
///
/// The rest are exact identities on those four plus `dm_sqrt` (Newton on the reciprocal square root
/// from a `float` seed, finished with one correctly rounded division) and `dm_expm1` / `dm_log1p`
/// (their own series near zero), chosen so that no branch cancels catastrophically:
///
///     asin(x)  = atan(x / √(1-x²))                      |x| ≤ ½
///              = π/2 - 2·asin(√((1-x)/2))               x > ½
///     acos(x)  = 2·asin(√((1-x)/2))                     x ≥ ½
///              = π - 2·asin(√((1+x)/2))                 x ≤ -½
///              = π/2 - asin(x)                          otherwise
///     sinh(x)  = ½·t·(t+2)/(1+t),  t = expm1(|x|)       |x| ≤ 20,  else e^|x|/2 (scaled inside exp)
///     cosh(x)  = 1 + t²/(2(1+t))                        same split
///     tanh(x)  = t/(t+2),          t = expm1(2|x|)      |x| ≤ 20,  else ±1
///     asinh(x) = log1p(|x| + x²/(1+√(1+x²)))            |x| ≤ 2
///     acosh(x) = log1p(t + √(2t+t²)),  t = x-1          1 ≤ x ≤ 2
///     atanh(x) = ½·log1p(2|x| + 2x²/(1-|x|))            |x| < ½
///
/// Because `sinh`/`cosh` scale inside `dm_exp` rather than halving afterwards, they stay finite over
/// their whole mathematical range instead of overflowing near 710.
///
/// # Special values
///
/// NaN in, NaN out. `sin`/`cos`/`tan` of ±∞ are NaN; `asin`/`acos` outside `[-1, 1]`, `acosh` below 1
/// and `atanh` outside `[-1, 1]` are NaN; `atanh(±1)` is ±∞. Signed zeros are preserved by the odd
/// functions (`sin`, `tan`, `asin`, `atan`, `sinh`, `tanh`, `asinh`, `atanh` all return `-0.0` for
/// `-0.0`). `atan2` follows the C99 table, including the four `±0` and the four `±∞` cases.
enum TrigSource {
    /// The twelve unary functions, in the order the `TrigOp` enum and the C ABI op table use.
    static let unaryOps = ["sin", "cos", "tan", "asin", "acos", "atan",
                           "sinh", "cosh", "tanh", "asinh", "acosh", "atanh"]

    /// The seven functions Arrow also publishes in a `_checked` form, with the MSL expression that
    /// detects a **domain violation** for each. `v` is the loaded value: a `float` for float32, a
    /// raw bit pattern for float64.
    ///
    /// Arrow's rule, confirmed against `pyarrow.compute`: `asin`/`acos` raise outside `[-1, 1]`
    /// (±∞ included), `acosh` below 1, `atanh` at or outside `±1`, and `sin`/`cos`/`tan` on an
    /// infinite input. **NaN never raises** (it is not out of the domain, it is unordered), and a
    /// null row is not inspected at all.
    static let checkedOps: [(op: String, float: String, double: String)] = [
        ("sin", "isinf(v)", "dm_is_inf(v)"),
        ("cos", "isinf(v)", "dm_is_inf(v)"),
        ("tan", "isinf(v)", "dm_is_inf(v)"),
        ("asin", "fabs(v) > 1.0f", "!d_is_nan(v) && dm_lt(DM_ONE, dm_abs(v))"),
        ("acos", "fabs(v) > 1.0f", "!d_is_nan(v) && dm_lt(DM_ONE, dm_abs(v))"),
        ("acosh", "v < 1.0f", "!d_is_nan(v) && dm_lt(v, DM_ONE)"),
        ("atanh", "fabs(v) >= 1.0f", "!d_is_nan(v) && !dm_lt(dm_abs(v), DM_ONE)"),
    ]

    /// The checked kernel for one op: it writes the same values as the unchecked one and, for every
    /// **valid** row that violates the domain, raises a flag and keeps the smallest offending index
    /// so the host can name it. `T` is the MSL value type, `expr` the unchecked expression over `v`.
    static func checkedKernel(op: String, T: String, expr: String, bad: String) -> String { """
    kernel void tg_\(op)_checked(device const \(T)* a [[buffer(0)]],
                                 device const uchar* validity [[buffer(1)]],
                                 device const uint* nPtr [[buffer(2)]],
                                 constant uint& hasValidity [[buffer(3)]],
                                 device \(T)* out [[buffer(4)]],
                                 device atomic_uint* errFlag [[buffer(5)]],
                                 device atomic_uint* errIndex [[buffer(6)]],
                                 uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        \(T) v = a[i];
        out[i] = \(expr);
        if (hasValidity && !bit_get(validity, i)) return;
        if (\(bad)) {
            atomic_store_explicit(errFlag, 1u, memory_order_relaxed);
            atomic_fetch_min_explicit(errIndex, i, memory_order_relaxed);
        }
    }

    """ }

    // MARK: - float32

    /// The six hyperbolic float32 functions, written out rather than taken from the MSL library.
    ///
    /// Metal's own `sinh`, `cosh`, `tanh`, `asinh`, `acosh` and `atanh` are the naive
    /// `(e^x ± e^-x)/2` and `log(x + √(x²+1))` forms: measured against the host libm they lose about
    /// 12 ulp on `sinh(30)`, tens of thousands of ulp on `asinh(-710)`, and return **NaN** for
    /// `tanh(±∞)` and `asinh(±∞)`. The same well-conditioned identities the float64 path uses fix
    /// all of that and stay finite up to the true overflow point. `sin`, `cos` and `tan` do come
    /// from the `precise::` namespace, and `asin` / `acos` / `atan` from the accurate default
    /// overloads (fast math is off, so those are the precise ones).
    static let float32Hyperbolics = """

    #define TGF_LN2  0.69314718055994531f
    #define TGF_EHALF 1.35914091422952262f
    // MSL has no expm1 / log1p, and both identities below need them near zero, where `exp(x) - 1`
    // and `log(1 + x)` cancel. Short Taylor series on the interval each is used over; the first
    // omitted term is below a float ulp there.
    inline float tgf_expm1(float x) {
        if (fabs(x) < 0.35f) {
            float p = 1.0f / 362880.0f;
            p = 1.0f / 40320.0f + x * p;
            p = 1.0f / 5040.0f + x * p;
            p = 1.0f / 720.0f + x * p;
            p = 1.0f / 120.0f + x * p;
            p = 1.0f / 24.0f + x * p;
            p = 1.0f / 6.0f + x * p;
            p = 0.5f + x * p;
            p = 1.0f + x * p;
            return x * p;
        }
        return precise::exp(x) - 1.0f;
    }
    inline float tgf_log1p(float x) {
        if (fabs(x) < 0.29f) {
            float s = x / (2.0f + x), s2 = s * s;
            float p = 1.0f / 9.0f;
            p = 1.0f / 7.0f + s2 * p;
            p = 1.0f / 5.0f + s2 * p;
            p = 1.0f / 3.0f + s2 * p;
            p = 1.0f + s2 * p;
            return 2.0f * s * p;
        }
        return precise::log(1.0f + x);
    }
    inline float tgf_sinh(float v) {
        float a = fabs(v);
        if (isnan(v) || isinf(v)) return v;
        float r;
        if (a > 88.0f) r = precise::exp(a - 1.0f) * TGF_EHALF;      // a - 1 is exact for a >= 64
        else if (a > 20.0f) r = 0.5f * precise::exp(a);
        else { float t = tgf_expm1(a); r = 0.5f * t * (t + 2.0f) / (t + 1.0f); }
        return (v < 0.0f) ? -r : r;
    }
    inline float tgf_cosh(float v) {
        float a = fabs(v);
        if (isnan(v)) return v;
        if (isinf(a)) return INFINITY;
        if (a > 88.0f) return precise::exp(a - 1.0f) * TGF_EHALF;
        if (a > 20.0f) return 0.5f * precise::exp(a);
        float t = tgf_expm1(a);
        return 1.0f + t * t / (2.0f * (1.0f + t));
    }
    inline float tgf_tanh(float v) {
        float a = fabs(v);
        if (isnan(v)) return v;
        if (a > 12.0f) return (v < 0.0f) ? -1.0f : 1.0f;
        float t = tgf_expm1(2.0f * a);
        float r = t / (t + 2.0f);
        return (v < 0.0f) ? -r : r;
    }
    inline float tgf_asinh(float v) {
        float a = fabs(v);
        if (isnan(v) || isinf(v) || a == 0.0f) return v;
        float r;
        if (a > 1.0e9f) r = precise::log(a) + TGF_LN2;
        else if (a > 2.0f) r = precise::log(2.0f * a + 1.0f / (a + sqrt(a * a + 1.0f)));
        else { float a2 = a * a; r = tgf_log1p(a + a2 / (1.0f + sqrt(1.0f + a2))); }
        return (v < 0.0f) ? -r : r;
    }
    inline float tgf_acosh(float v) {
        if (isnan(v)) return v;
        if (v < 1.0f) return NAN;
        if (isinf(v)) return v;
        if (v > 1.0e9f) return precise::log(v) + TGF_LN2;
        if (v > 2.0f) return precise::log(2.0f * v - 1.0f / (v + sqrt(v * v - 1.0f)));
        float t = v - 1.0f;
        return tgf_log1p(t + sqrt(2.0f * t + t * t));
    }
    inline float tgf_atanh(float v) {
        float a = fabs(v);
        if (isnan(v)) return v;
        if (a > 1.0f) return NAN;
        if (a == 1.0f) return (v < 0.0f) ? -INFINITY : INFINITY;
        if (a == 0.0f) return v;
        float a2 = a + a;
        float r = (a < 0.5f) ? 0.5f * tgf_log1p(a2 + a2 * a / (1.0f - a))
                             : 0.5f * tgf_log1p(a2 / (1.0f - a));
        return (v < 0.0f) ? -r : r;
    }
    // Metal's float `atan2` does not follow the C99 special-value table: `atan2(±inf, ±inf)` is NaN
    // and every `x = -0.0` case comes back with the wrong sign or magnitude. Handle those here and
    // delegate the ordinary quadrants, which are accurate, to the library.
    inline float tgf_atan2(float y, float x) {
        if (isnan(y) || isnan(x)) return NAN;
        bool sy = signbit(y), sx = signbit(x);
        if (y == 0.0f) return sx ? (sy ? -M_PI_F : M_PI_F) : y;
        if (x == 0.0f) return sy ? -M_PI_2_F : M_PI_2_F;
        if (isinf(x)) {
            if (isinf(y)) { float v = sx ? 3.0f * M_PI_4_F : M_PI_4_F; return sy ? -v : v; }
            float v = sx ? M_PI_F : 0.0f;
            return sy ? -v : v;
        }
        if (isinf(y)) return sy ? -M_PI_2_F : M_PI_2_F;
        return atan2(y, x);
    }
    """

    /// float32 kernels.
    static let float32: String = {
        var s = KernelSource.prelude + float32Hyperbolics + "\n"
        let expr: [String: String] = [
            "sin": "precise::sin(v)", "cos": "precise::cos(v)", "tan": "precise::tan(v)",
            "asin": "asin(v)", "acos": "acos(v)", "atan": "atan(v)",
            "sinh": "tgf_sinh(v)", "cosh": "tgf_cosh(v)", "tanh": "tgf_tanh(v)",
            "asinh": "tgf_asinh(v)", "acosh": "tgf_acosh(v)", "atanh": "tgf_atanh(v)",
        ]
        for op in unaryOps {
            s += """
            kernel void tg_\(op)(device const float* a [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                                 device float* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
                if (i >= *nPtr) return;
                float v = a[i];
                out[i] = \(expr[op]!);
            }

            """
        }
        for c in checkedOps { s += checkedKernel(op: c.op, T: "float", expr: expr[c.op]!, bad: c.float) }
        s += """
        kernel void tg_atan2_array(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                                   device const uint* nPtr [[buffer(2)]], device float* out [[buffer(3)]],
                                   uint i [[thread_position_in_grid]]) {
            if (i < *nPtr) out[i] = tgf_atan2(a[i], b[i]);
        }
        kernel void tg_atan2_scalar(device const float* a [[buffer(0)]], constant float& scalar [[buffer(1)]],
                                    device const uint* nPtr [[buffer(2)]], device float* out [[buffer(3)]],
                                    uint i [[thread_position_in_grid]]) {
            if (i < *nPtr) out[i] = tgf_atan2(a[i], scalar);
        }
        """
        return s
    }()

    // MARK: - float64

    /// Constants: π/2 as sixteen 8-bit chunks, ln2 as three 26-bit chunks, and the Taylor
    /// coefficient tables. Generated once from a 200-digit decimal expansion; each chunk is an
    /// exactly representable double so that `k · chunk` is exact for a small enough integer `k`.
    static let doubleConstants = """

    constant ulong DM_PIO2_P[16] = {
        0x3FF9200000000000ul, 0x3F3E000000000000ul, 0x3EFB400000000000ul, 0x3E74400000000000ul,
        0x3DD0800000000000ul, 0x3D6A000000000000ul, 0x3CF8400000000000ul, 0x3C5A000000000000ul,
        0x3BF8800000000000ul, 0x3B78C00000000000ul, 0x3AE8800000000000ul, 0x3A71600000000000ul,
        0x39F0000000000000ul, 0x397B800000000000ul, 0x38CC000000000000ul, 0x387A200000000000ul
    };
    constant ulong DM_LN2_P[3] = {
        0x3FE62E42F8000000ul, 0x3E4BE8E7B8000000ul, 0x3CA35793C0000000ul
    };
    #define DM_ONE   0x3FF0000000000000ul
    #define DM_TWO   0x4000000000000000ul
    #define DM_HALF  0x3FE0000000000000ul
    #define DM_PI    0x400921FB54442D18ul
    #define DM_PIO2  0x3FF921FB54442D18ul
    #define DM_PIO4  0x3FE921FB54442D18ul
    #define DM_3PIO4 0x4002D97C7F3321D2ul
    #define DM_PIO6  0x3FE0C152382D7366ul
    #define DM_SQRT3 0x3FFBB67AE8584CAAul
    #define DM_TAN_PIO12 0x3FD126145E9ECD56ul
    #define DM_2_OVER_PI 0x3FE45F306DC9C883ul
    #define DM_INV_LN2   0x3FF71547652B82FEul
    #define DM_LN2       0x3FE62E42FEFA39EFul
    #define DM_SQRT2     0x3FF6A09E667F3BCDul
    #define DM_SIGN      0x8000000000000000ul
    #define DM_ABSMASK   0x7FFFFFFFFFFFFFFFul
    // sin(r)/r, cos(r), e^r, expm1(r)/r, atanh(s)/s and atan(t)/t Taylor coefficients.
    constant ulong DM_SIN_C[9] = {
        0x3FF0000000000000ul, 0xBFC5555555555555ul, 0x3F81111111111111ul, 0xBF2A01A01A01A01Aul,
        0x3EC71DE3A556C734ul, 0xBE5AE64567F544E4ul, 0x3DE6124613A86D09ul, 0xBD6AE7F3E733B81Ful,
        0x3CE952C77030AD4Aul
    };
    constant ulong DM_COS_C[10] = {
        0x3FF0000000000000ul, 0xBFE0000000000000ul, 0x3FA5555555555555ul, 0xBF56C16C16C16C17ul,
        0x3EFA01A01A01A01Aul, 0xBE927E4FB7789F5Cul, 0x3E21EED8EFF8D898ul, 0xBDA93974A8C07C9Dul,
        0x3D2AE7F3E733B81Ful, 0xBCA6827863B97D97ul
    };
    constant ulong DM_EXP_C[16] = {
        0x3FF0000000000000ul, 0x3FF0000000000000ul, 0x3FE0000000000000ul, 0x3FC5555555555555ul,
        0x3FA5555555555555ul, 0x3F81111111111111ul, 0x3F56C16C16C16C17ul, 0x3F2A01A01A01A01Aul,
        0x3EFA01A01A01A01Aul, 0x3EC71DE3A556C734ul, 0x3E927E4FB7789F5Cul, 0x3E5AE64567F544E4ul,
        0x3E21EED8EFF8D898ul, 0x3DE6124613A86D09ul, 0x3DA93974A8C07C9Dul, 0x3D6AE7F3E733B81Ful
    };
    constant ulong DM_EXPM1_C[16] = {
        0x3FF0000000000000ul, 0x3FE0000000000000ul, 0x3FC5555555555555ul, 0x3FA5555555555555ul,
        0x3F81111111111111ul, 0x3F56C16C16C16C17ul, 0x3F2A01A01A01A01Aul, 0x3EFA01A01A01A01Aul,
        0x3EC71DE3A556C734ul, 0x3E927E4FB7789F5Cul, 0x3E5AE64567F544E4ul, 0x3E21EED8EFF8D898ul,
        0x3DE6124613A86D09ul, 0x3DA93974A8C07C9Dul, 0x3D6AE7F3E733B81Ful, 0x3D2AE7F3E733B81Ful
    };
    constant ulong DM_LOG_C[12] = {
        0x3FF0000000000000ul, 0x3FD5555555555555ul, 0x3FC999999999999Aul, 0x3FC2492492492492ul,
        0x3FBC71C71C71C71Cul, 0x3FB745D1745D1746ul, 0x3FB3B13B13B13B14ul, 0x3FB1111111111111ul,
        0x3FAE1E1E1E1E1E1Eul, 0x3FAAF286BCA1AF28ul, 0x3FA8618618618618ul, 0x3FA642C8590B2164ul
    };
    constant ulong DM_ATAN_C[16] = {
        0x3FF0000000000000ul, 0xBFD5555555555555ul, 0x3FC999999999999Aul, 0xBFC2492492492492ul,
        0x3FBC71C71C71C71Cul, 0xBFB745D1745D1746ul, 0x3FB3B13B13B13B14ul, 0xBFB1111111111111ul,
        0x3FAE1E1E1E1E1E1Eul, 0xBFAAF286BCA1AF28ul, 0x3FA8618618618618ul, 0xBFA642C8590B2164ul,
        0x3FA47AE147AE147Bul, 0xBFA2F684BDA12F68ul, 0x3FA1A7B9611A7B96ul, 0xBFA0842108421084ul
    };
    """

    /// The software binary64 elementary functions themselves.
    static let doubleLib = """

    inline ulong dm_abs(ulong a) { return a & DM_ABSMASK; }
    inline ulong dm_neg(ulong a) { return a ^ DM_SIGN; }
    inline bool dm_is_inf(ulong a) { return (a & DM_ABSMASK) == D_INF; }
    // Ordered comparison of two non-NaN values through the order-preserving key.
    inline bool dm_lt(ulong a, ulong b) { return d_key((long)a) < d_key((long)b); }
    inline bool dm_le(ulong a, ulong b) { return d_key((long)a) <= d_key((long)b); }

    // ---- integer bridges
    inline ulong dm_from_long(long v) {
        if (v == 0L) return 0ul;
        ulong s = 0ul, u;
        if (v < 0L) { s = 1ul; u = (ulong)(-v); } else u = (ulong)v;
        int msb = 63;
        while (((u >> msb) & 1ul) == 0ul) msb--;
        ulong m;
        if (msb <= 55) m = u << (55 - msb);
        else { int sh = msb - 55; ulong st = (u & ((1ul << sh) - 1ul)) ? 1ul : 0ul; m = (u >> sh) | st; }
        return d_finish(s, (long)msb + 1023, m);
    }
    // Truncation toward zero; |x| must be below 2^62 (callers guard on that).
    inline long dm_to_long(ulong x) {
        long e = (long)((x >> 52) & 0x7FFul) - 1023;
        if (e < 0) return 0L;
        ulong m = (x & 0xFFFFFFFFFFFFFul) | (1ul << 52);
        long r = (e >= 52) ? (long)(m << (e - 52)) : (long)(m >> (52 - e));
        return (x >> 63) ? -r : r;
    }
    // Nearest integer, ties away from zero. The subtraction below is exact, so no rounding creeps in.
    inline long dm_rint_long(ulong x) {
        long i = dm_to_long(x);
        ulong diff = dm_abs(d_sub(x, dm_from_long(i)));
        if (!dm_lt(diff, DM_HALF)) i += (x >> 63) ? -1L : 1L;
        return i;
    }
    inline ulong dm_pow2(int n) {
        if (n > 1023) n = 1023;
        if (n < -1074) n = -1074;
        if (n >= -1022) return ((ulong)(n + 1023)) << 52;
        return 1ul << (n + 1074);
    }
    inline ulong dm_ldexp(ulong x, int n) {
        while (n > 1023) { x = d_mul(x, dm_pow2(1023)); n -= 1023; }
        while (n < -1022) { x = d_mul(x, dm_pow2(-1022)); n += 1022; }
        return d_mul(x, dm_pow2(n));
    }

    // ---- Horner over a coefficient table (in x, and in x with an outer factor)
    inline ulong dm_poly(ulong x, constant ulong* c, int n) {
        ulong acc = c[n - 1];
        for (int i = n - 2; i >= 0; i--) acc = d_add(c[i], d_mul(acc, x));
        return acc;
    }

    // ---- sqrt: float seed, Newton on the reciprocal root, one correctly rounded division to finish.
    inline ulong dm_sqrt(ulong x) {
        if (d_is_nan(x)) return x | (1ul << 51);
        if (d_is_zero(x)) return x;
        if (x >> 63) return D_QNAN;
        if (dm_is_inf(x)) return x;
        // Scale to m in [1, 4) by an even power of two.
        long e = (long)((x >> 52) & 0x7FFul);
        int scale = 0;
        ulong xn = x;
        if (e == 0) { xn = d_mul(x, dm_pow2(106)); scale = -53; e = (long)((xn >> 52) & 0x7FFul); }
        long ue = e - 1023;
        int j = (int)(ue >> 1);                       // floor(ue / 2), arithmetic shift
        ulong m = dm_ldexp(xn, -2 * j);
        // float seed for 1/sqrt(m): m is in [1, 4), so its float form is one shift away.
        uint me = (uint)((m >> 52) & 0x7FFul);
        float mf = as_type<float>(((me - 1023u + 127u) << 23) | (uint)((m >> 29) & 0x7FFFFFul));
        ulong z = d_from_float(1.0f / sqrt(mf));       // ~24 bits
        ulong hm = d_mul(DM_HALF, m);
        for (int it = 0; it < 3; it++) {
            ulong t = d_sub(0x3FF8000000000000ul, d_mul(hm, d_mul(z, z)));   // 1.5 - m/2 * z^2
            z = d_mul(z, t);
        }
        ulong y = d_mul(m, z);
        y = d_mul(DM_HALF, d_add(y, d_div(m, y)));
        return dm_ldexp(y, j + scale);
    }

    // ---- exp, with an extra power-of-two scale folded into the final ldexp so that e^x/2 near the
    // top of the range does not overflow on the way.
    inline ulong dm_exp_k(ulong x, int extra) {
        if (d_is_nan(x)) return x | (1ul << 51);
        if (dm_is_inf(x)) return (x >> 63) ? 0ul : D_INF;
        if (d_is_zero(x)) return dm_ldexp(DM_ONE, extra);
        long k = dm_rint_long(d_mul(x, DM_INV_LN2));
        if (k > 1200L) return D_INF;
        if (k < -1200L) return 0ul;
        ulong kd = dm_from_long(k);
        ulong r = x;
        for (int i = 0; i < 3; i++) r = d_sub(r, d_mul(kd, DM_LN2_P[i]));
        ulong y = dm_poly(r, &DM_EXP_C[0], 16);
        return dm_ldexp(y, (int)k + extra);
    }
    inline ulong dm_exp(ulong x) { return dm_exp_k(x, 0); }

    // ---- expm1: its own series near zero, where exp(x) - 1 would cancel.
    inline ulong dm_expm1(ulong x) {
        if (d_is_nan(x)) return x | (1ul << 51);
        if (dm_is_inf(x)) return (x >> 63) ? 0xBFF0000000000000ul : D_INF;
        if (dm_lt(dm_abs(x), 0x3FD6666666666666ul)) {           // |x| < 0.35
            return d_mul(x, dm_poly(x, &DM_EXPM1_C[0], 16));
        }
        return d_sub(dm_exp(x), DM_ONE);
    }

    // ---- log
    inline ulong dm_log(ulong x) {
        if (d_is_nan(x)) return x | (1ul << 51);
        if (d_is_zero(x)) return D_INF | DM_SIGN;
        if (x >> 63) return D_QNAN;
        if (dm_is_inf(x)) return x;
        long e = (long)((x >> 52) & 0x7FFul);
        ulong xn = x;
        long k = 0;
        if (e == 0) { xn = d_mul(x, dm_pow2(106)); k = -106; e = (long)((xn >> 52) & 0x7FFul); }
        k += e - 1023;
        ulong m = (xn & 0x800FFFFFFFFFFFFFul) | (1023ul << 52);   // mantissa with exponent 0: [1, 2)
        if (!dm_lt(m, DM_SQRT2)) { m = d_mul(m, DM_HALF); k += 1; }
        ulong s = d_div(d_sub(m, DM_ONE), d_add(m, DM_ONE));
        ulong lnm = d_mul(d_add(s, s), dm_poly(d_mul(s, s), &DM_LOG_C[0], 12));
        if (k == 0L) return lnm;
        ulong kd = dm_from_long(k);
        ulong acc = lnm;
        for (int i = 2; i >= 0; i--) acc = d_add(acc, d_mul(kd, DM_LN2_P[i]));
        return acc;
    }

    inline ulong dm_log1p(ulong y) {
        if (d_is_nan(y)) return y | (1ul << 51);
        if (dm_lt(dm_abs(y), 0x3FD28F5C28F5C28Ful)) {           // |y| < 0.29
            ulong s = d_div(y, d_add(DM_TWO, y));
            return d_mul(d_add(s, s), dm_poly(d_mul(s, s), &DM_LOG_C[0], 12));
        }
        return dm_log(d_add(DM_ONE, y));
    }

    // ---- Cody-Waite reduction against sixteen 8-bit chunks of pi/2.
    // Returns the reduced argument and writes the quadrant to *n. Callers guard NaN / infinity and
    // arguments at or above 2^62, where the quadrant no longer fits a long.
    inline ulong dm_reduce_pio2(ulong x, thread long* n) {
        if (dm_le(dm_abs(x), DM_PIO4)) { *n = 0L; return x; }
        long k = dm_rint_long(d_mul(x, DM_2_OVER_PI));
        ulong kd = dm_from_long(k);
        ulong r = x;
        for (int i = 0; i < 16; i++) r = d_sub(r, d_mul(kd, DM_PIO2_P[i]));
        *n = k;
        return r;
    }
    inline ulong dm_sin_poly(ulong r) { return d_mul(r, dm_poly(d_mul(r, r), &DM_SIN_C[0], 9)); }
    inline ulong dm_cos_poly(ulong r) { return dm_poly(d_mul(r, r), &DM_COS_C[0], 10); }

    #define DM_TOO_BIG 0x43D0000000000000ul   /* 2^62 */

    inline ulong dm_sin(ulong x) {
        if (d_is_nan(x)) return x | (1ul << 51);
        if (dm_is_inf(x) || !dm_lt(dm_abs(x), DM_TOO_BIG)) return D_QNAN;
        long n;
        ulong r = dm_reduce_pio2(x, &n);
        switch ((int)(n & 3L)) {
            case 0: return dm_sin_poly(r);
            case 1: return dm_cos_poly(r);
            case 2: return dm_neg(dm_sin_poly(r));
            default: return dm_neg(dm_cos_poly(r));
        }
    }
    inline ulong dm_cos(ulong x) {
        if (d_is_nan(x)) return x | (1ul << 51);
        if (dm_is_inf(x) || !dm_lt(dm_abs(x), DM_TOO_BIG)) return D_QNAN;
        long n;
        ulong r = dm_reduce_pio2(x, &n);
        switch ((int)((n + 1L) & 3L)) {
            case 0: return dm_sin_poly(r);
            case 1: return dm_cos_poly(r);
            case 2: return dm_neg(dm_sin_poly(r));
            default: return dm_neg(dm_cos_poly(r));
        }
    }
    inline ulong dm_tan(ulong x) {
        if (d_is_nan(x)) return x | (1ul << 51);
        if (dm_is_inf(x) || !dm_lt(dm_abs(x), DM_TOO_BIG)) return D_QNAN;
        long n;
        ulong r = dm_reduce_pio2(x, &n);
        ulong s = dm_sin_poly(r), c = dm_cos_poly(r);
        if (n & 1L) return dm_neg(d_div(c, s));
        return d_div(s, c);
    }

    // ---- atan and friends
    inline ulong dm_atan(ulong x) {
        if (d_is_nan(x)) return x | (1ul << 51);
        ulong sgn = x & DM_SIGN;
        ulong a = dm_abs(x);
        if (dm_is_inf(a)) return DM_PIO2 | sgn;
        if (d_is_zero(a)) return x;
        bool inv = dm_lt(DM_ONE, a);
        if (inv) a = d_div(DM_ONE, a);
        bool six = dm_lt(DM_TAN_PIO12, a);
        if (six) a = d_div(d_sub(d_mul(a, DM_SQRT3), DM_ONE), d_add(a, DM_SQRT3));
        ulong y = d_mul(a, dm_poly(d_mul(a, a), &DM_ATAN_C[0], 16));
        if (six) y = d_add(y, DM_PIO6);
        if (inv) y = d_sub(DM_PIO2, y);
        return y | sgn;
    }

    inline ulong dm_atan2(ulong y, ulong x) {
        if (d_is_nan(x) || d_is_nan(y)) return D_QNAN;
        ulong sy = y & DM_SIGN, sx = x >> 63;
        if (d_is_zero(y)) return sx ? (DM_PI | sy) : y;
        if (d_is_zero(x)) return DM_PIO2 | sy;
        if (dm_is_inf(x)) {
            if (dm_is_inf(y)) return (sx ? DM_3PIO4 : DM_PIO4) | sy;
            return (sx ? DM_PI : 0ul) | sy;
        }
        if (dm_is_inf(y)) return DM_PIO2 | sy;
        ulong z = dm_atan(d_div(dm_abs(y), dm_abs(x)));
        if (sx) z = d_sub(DM_PI, z);
        return z | sy;
    }

    // asin on |u| <= 1/2, where 1 - u^2 loses nothing.
    inline ulong dm_asin_small(ulong u) {
        ulong t = d_sub(DM_ONE, d_mul(u, u));
        return dm_atan(d_div(u, dm_sqrt(t)));
    }
    inline ulong dm_asin(ulong x) {
        if (d_is_nan(x)) return x | (1ul << 51);
        ulong sgn = x & DM_SIGN;
        ulong a = dm_abs(x);
        if (dm_lt(DM_ONE, a)) return D_QNAN;
        if (a == DM_ONE) return DM_PIO2 | sgn;
        if (dm_le(a, DM_HALF)) return dm_asin_small(a) | sgn;
        ulong t = dm_sqrt(d_mul(DM_HALF, d_sub(DM_ONE, a)));
        ulong y = d_sub(DM_PIO2, d_mul(DM_TWO, dm_asin_small(t)));
        return y | sgn;
    }
    inline ulong dm_acos(ulong x) {
        if (d_is_nan(x)) return x | (1ul << 51);
        ulong a = dm_abs(x);
        if (dm_lt(DM_ONE, a)) return D_QNAN;
        if (!dm_lt(x, DM_HALF)) {                       // x >= 0.5
            ulong t = dm_sqrt(d_mul(DM_HALF, d_sub(DM_ONE, x)));
            return d_mul(DM_TWO, dm_asin_small(t));
        }
        if (dm_le(x, 0xBFE0000000000000ul)) {           // x <= -0.5
            ulong t = dm_sqrt(d_mul(DM_HALF, d_add(DM_ONE, x)));
            return d_sub(DM_PI, d_mul(DM_TWO, dm_asin_small(t)));
        }
        return d_sub(DM_PIO2, dm_asin_small(x));
    }

    // ---- hyperbolics
    #define DM_TWENTY 0x4034000000000000ul
    inline ulong dm_sinh(ulong x) {
        if (d_is_nan(x)) return x | (1ul << 51);
        if (dm_is_inf(x) || d_is_zero(x)) return x;
        ulong sgn = x & DM_SIGN, a = dm_abs(x);
        if (dm_lt(DM_TWENTY, a)) return dm_exp_k(a, -1) | sgn;
        ulong t = dm_expm1(a);
        ulong y = d_mul(DM_HALF, d_div(d_mul(t, d_add(t, DM_TWO)), d_add(t, DM_ONE)));
        return y | sgn;
    }
    inline ulong dm_cosh(ulong x) {
        if (d_is_nan(x)) return x | (1ul << 51);
        ulong a = dm_abs(x);
        if (dm_is_inf(a)) return D_INF;
        if (dm_lt(DM_TWENTY, a)) return dm_exp_k(a, -1);
        ulong t = dm_expm1(a);
        ulong num = d_mul(t, t);
        return d_add(DM_ONE, d_div(num, d_mul(DM_TWO, d_add(DM_ONE, t))));
    }
    inline ulong dm_tanh(ulong x) {
        if (d_is_nan(x)) return x | (1ul << 51);
        if (d_is_zero(x)) return x;
        ulong sgn = x & DM_SIGN, a = dm_abs(x);
        if (dm_lt(DM_TWENTY, a)) return DM_ONE | sgn;
        ulong t = dm_expm1(d_mul(DM_TWO, a));
        return d_div(t, d_add(t, DM_TWO)) | sgn;
    }
    inline ulong dm_asinh(ulong x) {
        if (d_is_nan(x)) return x | (1ul << 51);
        if (dm_is_inf(x) || d_is_zero(x)) return x;
        ulong sgn = x & DM_SIGN, a = dm_abs(x);
        ulong y;
        if (dm_lt(0x41B0000000000000ul, a)) {                      // a > 2^28
            y = d_add(dm_log(a), DM_LN2);
        } else if (dm_lt(DM_TWO, a)) {
            ulong s = dm_sqrt(d_add(d_mul(a, a), DM_ONE));
            y = dm_log(d_add(d_mul(DM_TWO, a), d_div(DM_ONE, d_add(a, s))));
        } else {
            ulong a2 = d_mul(a, a);
            ulong s = dm_sqrt(d_add(DM_ONE, a2));
            y = dm_log1p(d_add(a, d_div(a2, d_add(DM_ONE, s))));
        }
        return y | sgn;
    }
    inline ulong dm_acosh(ulong x) {
        if (d_is_nan(x)) return x | (1ul << 51);
        if (dm_lt(x, DM_ONE)) return D_QNAN;
        if (x == DM_ONE) return 0ul;
        if (dm_is_inf(x)) return x;
        if (dm_lt(0x41B0000000000000ul, x)) return d_add(dm_log(x), DM_LN2);
        if (dm_lt(DM_TWO, x)) {
            ulong s = dm_sqrt(d_sub(d_mul(x, x), DM_ONE));
            return dm_log(d_sub(d_mul(DM_TWO, x), d_div(DM_ONE, d_add(x, s))));
        }
        ulong t = d_sub(x, DM_ONE);
        return dm_log1p(d_add(t, dm_sqrt(d_add(d_mul(DM_TWO, t), d_mul(t, t)))));
    }
    inline ulong dm_atanh(ulong x) {
        if (d_is_nan(x)) return x | (1ul << 51);
        ulong sgn = x & DM_SIGN, a = dm_abs(x);
        if (dm_lt(DM_ONE, a)) return D_QNAN;
        if (a == DM_ONE) return D_INF | sgn;
        if (d_is_zero(x)) return x;
        ulong y;
        ulong a2 = d_add(a, a);
        if (dm_lt(a, DM_HALF)) {
            y = d_mul(DM_HALF, dm_log1p(d_add(a2, d_div(d_mul(a2, a), d_sub(DM_ONE, a)))));
        } else {
            y = d_mul(DM_HALF, dm_log1p(d_div(a2, d_sub(DM_ONE, a))));
        }
        return y | sgn;
    }
    """

    /// float64 kernels. Values are raw `ulong` bit patterns; one element per thread.
    static let double: String = {
        var s = KernelSource.prelude + DoubleMath.msl + doubleConstants + doubleLib + "\n"
        for op in unaryOps {
            s += """
            kernel void tg_\(op)(device const ulong* a [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                                 device ulong* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
                if (i < *nPtr) out[i] = dm_\(op)(a[i]);
            }

            """
        }
        for c in checkedOps { s += checkedKernel(op: c.op, T: "ulong", expr: "dm_\(c.op)(v)", bad: c.double) }
        s += """
        kernel void tg_atan2_array(device const ulong* a [[buffer(0)]], device const ulong* b [[buffer(1)]],
                                   device const uint* nPtr [[buffer(2)]], device ulong* out [[buffer(3)]],
                                   uint i [[thread_position_in_grid]]) {
            if (i < *nPtr) out[i] = dm_atan2(a[i], b[i]);
        }
        kernel void tg_atan2_scalar(device const ulong* a [[buffer(0)]], constant ulong& scalar [[buffer(1)]],
                                    device const uint* nPtr [[buffer(2)]], device ulong* out [[buffer(3)]],
                                    uint i [[thread_position_in_grid]]) {
            if (i < *nPtr) out[i] = dm_atan2(a[i], scalar);
        }
        """
        return s
    }()
}
