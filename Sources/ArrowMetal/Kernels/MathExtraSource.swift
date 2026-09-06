import Foundation

/// MSL for the element-wise math Arrow defines that `RoundingSource` does not cover: `expm1`, `log1p`,
/// `logb`, `hypot`, and the rounding family (`round` with `ndigits`, `round_to_multiple`,
/// `round_binary`) in all ten Arrow `RoundMode`s.
///
/// Three flavours, as elsewhere in this package:
///
///   * **float32** — MSL library calls, compiled with `mathMode = .safe`. `expm1` and `log1p` are not in
///     the Metal standard library, and the usual repair for the cancellation near zero — Kahan's factor
///     `(u - 1)·x/log(u)` — cannot be used, because this Metal front end folds `log(exp(x))` back to `x`
///     and collapses it to the naive form. They are Taylor series near zero and the library call outside.
///   * **float64** — values travel as raw `ulong` bit patterns and every step runs through the software
///     binary64 arithmetic of `DoubleMath` and the transcendentals of `DoubleTranscendental`. These are
///     therefore *not* the seven-digit `float`-detour results the older float64 transcendentals give;
///     they carry the full 53-bit significand (measured ulp bounds are on `DoubleTranscendental`).
///   * **integers** — only the rounding family, which Arrow does define on integer columns: `expm1`,
///     `log1p`, `logb` and `hypot` need a floating point column here and throw on an integer one, exactly
///     as `sqrt` and `ln` already do (Arrow instead promotes to float64; cast first).
///
/// **Rounding semantics**, matched element for element against `pyarrow.compute`:
///
///   * `round(x, ndigits, mode)` is `round_int(x·10^ndigits) / 10^ndigits` for `ndigits >= 0` and
///     `round_int(x / 10^-ndigits) · 10^-ndigits` below, which is what Arrow evaluates and is why
///     `round(123.456, 2, HALF_TO_EVEN)` lands on `123.46` rather than on the nearer `123.45`.
///   * `round_to_multiple(x, m, mode)` is `round_int(x / m) · m`; `m` must be strictly positive.
///   * `round_binary(x, ndigits)` is `round` with a per-row `ndigits` column.
///   * On integers, rounding is done on the quotient and remainder rather than through floating point,
///     so an `int64` above 2^53 rounds exactly. `ndigits >= 0` is the identity.
enum MathExtraSource {
    enum Kind { case signedInt, unsignedInt, float32, float64 }

    /// `1/(k+1)!` for k = 0...8: `expm1(x) = x · Σ x^k/(k+1)!`.
    private static let expm1Coefficients: [Float] = {
        var c: [Float] = [], f = 1.0
        for k in 0...8 { f *= Double(k + 1); c.append(Float(1.0 / f)) }
        return c
    }()
    /// `(-1)^k/(k+1)` for k = 0...11: `log1p(x) = x · Σ (-1)^k x^k/(k+1)`.
    private static let log1pCoefficients: [Float] = (0...11).map { Float(($0 % 2 == 0 ? 1.0 : -1.0) / Double($0 + 1)) }

    /// Horner evaluation of `coefficients` in `variable`, in `float`.
    private static func hornerFloat(_ coefficients: [Float], _ variable: String) -> String {
        var s = "\(coefficients.last!)f"
        for c in coefficients.dropLast().reversed() { s = "\(c)f + \(variable) * (\(s))" }
        return s
    }

    /// Largest `|ndigits|` the float tables cover; beyond it the scaling is a no-op or a signed zero.
    static let float32Digits = 38
    static let float64Digits = 308

    private static func pow10Float() -> String {
        let v = (0...float32Digits).map { "\(Float("1e\($0)")!.bitPattern)u" }.joined(separator: ", ")
        return "constant uint mx_p10_bits[\(float32Digits + 1)] = { \(v) };\n"
    }
    private static func pow10Double() -> String {
        let v = (0...float64Digits).map { String(format: "0x%016llXul", Double("1e\($0)")!.bitPattern) }
            .joined(separator: ", ")
        return "constant ulong mx_p10[\(float64Digits + 1)] = { \(v) };\n"
    }
    private static func pow10Integer() -> String {
        var vals: [String] = []
        var p: UInt64 = 1
        for k in 0...18 { vals.append("\(p)ul"); if k < 18 { p *= 10 } }
        return "constant ulong mx_p10i[19] = { \(vals.joined(separator: ", ")) };\n"
    }

    // MARK: - Float32

    private static let float32Body = """
    \(pow10Float())
    inline float mx_p10f(int k) { return as_type<float>(mx_p10_bits[k]); }

    // exp(x) - 1. MSL has no expm1, and the usual repair for the cancellation in `exp(x) - 1` —
    // Kahan's factor `(u - 1)·x/log(u)` — is unusable here: this Metal front end folds `log(exp(x))`
    // straight back to `x`, which collapses the whole expression to the naive `u - 1` and, for
    // |x| ~ 1e-7, returns one ulp of 1.0 instead of x (a 2.9e6 ulp error, caught by MathExtraTests).
    // A Taylor series near zero owes the compiler nothing: `Σ x^k/(k+1)!` truncated after x^8 is good
    // to 1.1e-8 relative over |x| < 0.5, well inside float32's 6e-8.
    inline float mx_expm1(float x) {
        if (isnan(x)) return x;
        if (fabs(x) < 0.5f) return x * (\(hornerFloat(expm1Coefficients, "x")));
        return exp(x) - 1.0f;                       // no cancellation left out here
    }
    // ln(1 + x). Same shape and the same reasoning: an alternating series near zero, the library call
    // outside it. For x in [-1, -0.5] the sum `1 + x` is exact (Sterbenz), so the library branch loses
    // nothing there either.
    inline float mx_log1p(float x) {
        if (isnan(x)) return x;
        if (x == -1.0f) return -INFINITY;
        if (x < -1.0f) return NAN;
        if (isinf(x)) return x;
        if (fabs(x) < 0.25f) return x * (\(hornerFloat(log1pCoefficients, "x")));
        return log(1.0f + x);
    }
    inline float mx_logb(float x, float b) { return log(x) / log(b); }
    inline float mx_hypot(float a, float b) {
        float x = fabs(a), y = fabs(b);
        if (isinf(x) || isinf(y)) return INFINITY;      // IEEE: infinity wins even over NaN
        if (isnan(x) || isnan(y)) return NAN;
        if (x < y) { float t = x; x = y; y = t; }
        if (x == 0.0f) return 0.0f;
        float r = y / x;
        return x * sqrt(1.0f + r * r);
    }

    // Arrow RoundMode: 0 DOWN, 1 UP, 2 TOWARDS_ZERO, 3 TOWARDS_INFINITY, 4 HALF_DOWN, 5 HALF_UP,
    // 6 HALF_TOWARDS_ZERO, 7 HALF_TOWARDS_INFINITY, 8 HALF_TO_EVEN, 9 HALF_TO_ODD.
    inline float mx_round_int(float x, uint mode) {
        if (!isfinite(x)) return x;
        float t = trunc(x);
        if (t == x) return x;
        bool neg = (x < 0.0f);
        float away = t + (neg ? -1.0f : 1.0f);
        if (mode == 0u) return neg ? away : t;
        if (mode == 1u) return neg ? t : away;
        if (mode == 2u) return t;
        if (mode == 3u) return away;
        float frac = fabs(x) - fabs(t);
        if (frac < 0.5f) return t;
        if (frac > 0.5f) return away;
        if (mode == 4u) return neg ? away : t;
        if (mode == 5u) return neg ? t : away;
        if (mode == 6u) return t;
        if (mode == 7u) return away;
        bool even = (fmod(t, 2.0f) == 0.0f);
        if (mode == 8u) return even ? t : away;
        return even ? away : t;
    }
    inline float mx_round_nd(float x, int nd, uint mode) {
        if (!isfinite(x)) return x;
        if (nd == 0) return mx_round_int(x, mode);
        if (nd > \(float32Digits)) return x;
        if (nd < -\(float32Digits)) return x * 0.0f;
        if (nd > 0) { float p = mx_p10f(nd); return mx_round_int(x * p, mode) / p; }
        float p = mx_p10f(-nd);
        return mx_round_int(x / p, mode) * p;
    }
    inline float mx_round_mult(float x, float m, uint mode) {
        if (!isfinite(x)) return x;
        return mx_round_int(x / m, mode) * m;
    }

    """

    // MARK: - Float64

    private static let float64Body = """
    \(pow10Double())
    inline ulong mx_expm1(ulong x) { return dt_expm1(x); }
    inline ulong mx_log1p(ulong x) { return dt_log1p(x); }
    inline ulong mx_logb(ulong x, ulong b) { return dt_logb(x, b); }
    inline ulong mx_hypot(ulong a, ulong b) { return dt_hypot(a, b); }
    inline ulong mx_round_nd(ulong x, int nd, uint mode) {
        if (d_is_nan(x) || d_exp(x) == 0x7FFul) return x;
        if (nd == 0) return dt_round_int(x, mode);
        if (nd > \(float64Digits)) return x;
        if (nd < -\(float64Digits)) return x & 0x8000000000000000ul;
        if (nd > 0) { ulong p = mx_p10[nd]; return d_div(dt_round_int(d_mul(x, p), mode), p); }
        ulong p = mx_p10[-nd];
        return d_mul(dt_round_int(d_div(x, p), mode), p);
    }
    inline ulong mx_round_mult(ulong x, ulong m, uint mode) {
        if (d_is_nan(x) || d_exp(x) == 0x7FFul) return x;
        return d_mul(dt_round_int(d_div(x, m), mode), m);
    }

    """

    // MARK: - Integers

    private static func integerBody(T: String, U: String, signed: Bool, width: Int) -> String {
        let tmax = signed ? "(\(U))((((\(U))1 << \(width - 1)) - (\(U))1))" : "(\(U))(~(\(U))0)"
        let isNeg = signed ? "(v < (\(T))0)" : "false"
        let badMultiple = signed ? "m <= (\(T))0" : "m == (\(T))0"
        return """
        \(pow10Integer())
        // Rounding on the quotient and remainder: exact for every value, including int64 above 2^53.
        // A multiple that pushes the answer past the type's range wraps, like the unchecked arithmetic.
        inline \(T) mx_round_mult(\(T) v, \(T) m, uint mode) {
            if (\(badMultiple)) return v;                    // the host rejects this; be defined anyway
            \(T) r = (\(T))(v % m);
            if (r == (\(T))0) return v;
            bool neg = \(isNeg);
            \(T) toward = (\(T))(v - r);                     // toward zero
            \(T) away = (\(T))((\(U))toward + (neg ? (\(U))((\(U))0 - (\(U))m) : (\(U))m));
            if (mode == 0u) return neg ? away : toward;
            if (mode == 1u) return neg ? toward : away;
            if (mode == 2u) return toward;
            if (mode == 3u) return away;
            \(U) ar = (\(U))(neg ? (\(T))((\(U))0 - (\(U))r) : r);
            \(U) hm = (\(U))m;
            \(U) mhalf = (\(U))(hm >> 1);
            bool odd = ((hm & (\(U))1) != (\(U))0);
            // 2*|r| vs m without ever forming 2*|r| (which would overflow near the type's top):
            // greater iff |r| > m/2; a tie needs an even m and |r| exactly m/2.
            if (ar > mhalf) return away;
            if (ar < mhalf) return toward;
            if (odd) return toward;                          // m odd: 2*|r| == m - 1 < m
            // 2 * |r| == m exactly: a genuine tie.
            if (mode == 4u) return neg ? away : toward;
            if (mode == 5u) return neg ? toward : away;
            if (mode == 6u) return toward;
            if (mode == 7u) return away;
            bool even = (((\(U))(toward / m)) & (\(U))1) == (\(U))0;
            if (mode == 8u) return even ? toward : away;
            return even ? away : toward;
        }
        inline \(T) mx_round_nd(\(T) v, int nd, uint mode) {
            if (nd >= 0) return v;                            // rounding an integer to decimals is the identity
            int k = -nd;
            if (k > 18) return (\(T))0;
            ulong p = mx_p10i[k];
            if (p > (ulong)(\(tmax))) return (\(T))0;          // the multiple does not fit the column's type
            return mx_round_mult(v, (\(T))p, mode);
        }

        """
    }

    // MARK: - Kernels

    private static func kernels(V: String, transcendental: Bool) -> String {
        var s = ""
        if transcendental {
            for op in ["expm1", "log1p"] {
                s += """
                kernel void mx_unary_\(op)(device const \(V)* a [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                                           device \(V)* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
                    if (i < *nPtr) out[i] = mx_\(op)(a[i]);
                }

                """
            }
            for op in ["logb", "hypot"] {
                s += """
                kernel void mx_scalar_\(op)(device const \(V)* a [[buffer(0)]], constant \(V)& scalar [[buffer(1)]],
                                            device const uint* nPtr [[buffer(2)]], device \(V)* out [[buffer(3)]],
                                            uint i [[thread_position_in_grid]]) {
                    if (i < *nPtr) out[i] = mx_\(op)(a[i], scalar);
                }
                kernel void mx_array_\(op)(device const \(V)* a [[buffer(0)]], device const \(V)* b [[buffer(1)]],
                                           device const uint* nPtr [[buffer(2)]], device \(V)* out [[buffer(3)]],
                                           uint i [[thread_position_in_grid]]) {
                    if (i < *nPtr) out[i] = mx_\(op)(a[i], b[i]);
                }

                """
            }
        }
        s += """
        kernel void mx_round(device const \(V)* a [[buffer(0)]], constant int& nd [[buffer(1)]],
                             constant uint& mode [[buffer(2)]], device const uint* nPtr [[buffer(3)]],
                             device \(V)* out [[buffer(4)]], uint i [[thread_position_in_grid]]) {
            if (i < *nPtr) out[i] = mx_round_nd(a[i], nd, mode);
        }
        // round_binary: one ndigits per row. A null ndigits row is masked out by the host-side validity
        // combination, so whatever sits in that slot only has to be harmless, which the clamps make it.
        kernel void mx_round_binary(device const \(V)* a [[buffer(0)]], device const int* nd [[buffer(1)]],
                                    constant uint& mode [[buffer(2)]], device const uint* nPtr [[buffer(3)]],
                                    device \(V)* out [[buffer(4)]], uint i [[thread_position_in_grid]]) {
            if (i < *nPtr) out[i] = mx_round_nd(a[i], nd[i], mode);
        }
        kernel void mx_round_multiple(device const \(V)* a [[buffer(0)]], constant \(V)& mult [[buffer(1)]],
                                      constant uint& mode [[buffer(2)]], device const uint* nPtr [[buffer(3)]],
                                      device \(V)* out [[buffer(4)]], uint i [[thread_position_in_grid]]) {
            if (i < *nPtr) out[i] = mx_round_mult(a[i], mult, mode);
        }

        """
        return s
    }

    // MARK: - Entry point

    static func source(T: String, U: String, width: Int, kind: Kind) -> String {
        switch kind {
        case .float32:
            return KernelSource.prelude + "\n" + float32Body + kernels(V: "float", transcendental: true)
        case .float64:
            return KernelSource.prelude + DoubleMath.msl + DoubleTranscendental.msl + "\n"
                + float64Body + kernels(V: "ulong", transcendental: true)
        case .signedInt, .unsignedInt:
            return KernelSource.prelude + "\n"
                + integerBody(T: T, U: U, signed: kind == .signedInt, width: width)
                + kernels(V: T, transcendental: false)
        }
    }
}
