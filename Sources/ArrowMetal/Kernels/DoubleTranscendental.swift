import Foundation

/// Software IEEE-754 **binary64 transcendentals** for Metal, layered on the correctly rounded
/// `d_add` / `d_sub` / `d_mul` / `d_div` of `DoubleMath`.
///
/// Apple GPUs have no `double` at all. `float64` `sqrt`/`exp`/`ln`/`log2`/`log10` used to convert to
/// `float`, evaluate there and widen back — about seven correct significant digits — and `expm1`,
/// `log1p`, `logb`, `hypot` and the ten `RoundMode`s here were the first not to settle for that.
/// `Kernels/DoublePower.swift` has since taken the same route for the five above and for `power`, so
/// every float64 transcendental in the package now carries the full 53-bit significand.
///
/// Everything is a raw `ulong` bit pattern, and every function is prefixed `dt_` so it can sit beside
/// `DoubleMath`'s `d_*` (whose `d_exp` is the *exponent field*, not the exponential) in one translation
/// unit.
///
/// **Methods and accuracy.** `MathExtraTests.testFloat64PrecisionAgainstFoundation` measures each one
/// against Foundation over 10^6 random inputs and asserts these bounds:
///
/// | function  | method                                                                    | measured |
/// |-----------|---------------------------------------------------------------------------|----------|
/// | `dt_sqrt` | `DoubleMath.d_sqrt`, digit by digit                                        | correctly rounded |
/// | `dt_ln`   | `x = 2^e·m`, `m` folded to `[1/√2, √2]`, `atanh` series in `s = (m-1)/(m+1)`, `e·ln2` in a hi/lo split | (via `logb`) |
/// | `dt_expm1`| `x = k·ln2 + r` (no reduction below 0.5), Taylor `r·Σ rⁿ/(n+1)!`, then `2^k(1+E) − 1` | ≤ 2 ulp |
/// | `dt_log1p`| four-term series below `2^-20`, else `ln(u)·x/(u−1)` (Kahan's correction)  | ≤ 1 ulp  |
/// | `dt_logb` | `dt_ln(x) / dt_ln(base)`                                                   | ≤ 2 ulp  |
/// | `dt_hypot`| scale by `2^-e`, `sqrt(x² + y²)`, scale back — never overflows mid-way     | ≤ 1 ulp  |
///
/// None of these claims correct rounding; they claim a small, measured, bounded ulp error, which is what
/// a GPU library without hardware binary64 can honestly offer.
///
/// The single most valuable line below is the `ln 2` split: reducing `exp`'s argument against the
/// *nearest double* to `ln 2` rather than a 107-bit pair costs 250 ulp at `x ≈ 700`, because the
/// multiplier `k` amplifies that constant's own 5.5e-17 error. The measured 211 ulp before the fix is
/// what made it visible.
enum DoubleTranscendental {
    /// A Double as an MSL `ulong` bit-pattern literal.
    private static func lit(_ v: Double) -> String { String(format: "0x%016llXul", v.bitPattern) }

    /// `1, 1/3, 1/5, ...`: the atanh series `ln((1+s)/(1-s)) = 2s·Σ s^2k/(2k+1)`. Eleven terms cover
    /// `|s| ≤ 0.1716` (`s² ≤ 0.0295`) to well under half an ulp.
    private static let lnCoefficients: [Double] = (0...11).map { 1.0 / Double(2 * $0 + 1) }
    /// `1/(n+1)!`: `exp(r) - 1 = r·Σ rⁿ/(n+1)!`, sixteen terms for `|r| ≤ ln2/2`.
    private static let expCoefficients: [Double] = {
        var c: [Double] = [], f = 1.0
        for n in 0...15 { f *= Double(n + 1); c.append(1.0 / f) }
        return c
    }()

    /// Horner evaluation of `coeffs` in `variable`, innermost first.
    private static func horner(_ coefficients: [Double], _ variable: String) -> String {
        var s = "\(lit(coefficients.last!))"
        for c in coefficients.dropLast().reversed() { s = "d_add(d_mul(\(s), \(variable)), \(lit(c)))" }
        return s
    }

    static let msl: String = {
        // ln 2 to about 107 bits, as a hi/lo pair. `hi` keeps 32 significant bits (its low 21 mantissa
        // bits are zero), so `k * hi` is exact for every `|k| < 2^20` the argument reductions use, and
        // `lo` carries the *true* tail rather than `Double(ln2) - hi`. That distinction matters: the
        // nearest double to ln 2 is off by up to 5.5e-17, and `k` up to 1024 multiplies that into a
        // 5.6e-14 absolute error in the reduced argument — which `exp` turns into 250 ulp. These are
        // fdlibm's constants (0x3FE62E42FEE00000 and 0x3DEA39EF35793C76).
        let ln2Hi = Double(bitPattern: 0x3FE6_2E42_FEE0_0000)
        let ln2Lo = Double(bitPattern: 0x3DEA_39EF_3579_3C76)
        return """

        #define DT_ONE     0x3FF0000000000000ul
        #define DT_MONE    0xBFF0000000000000ul
        #define DT_HALF    0x3FE0000000000000ul
        #define DT_TWO     0x4000000000000000ul
        #define DT_NINF    0xFFF0000000000000ul
        #define DT_SQRT2   0x3FF6A09E667F3BCDul
        #define DT_LN2_HI  \(lit(ln2Hi))
        #define DT_LN2_LO  \(lit(ln2Lo))
        #define DT_INV_LN2 \(lit(1.0 / 0.693147180559945309417232121458))
        #define DT_ABS     0x7FFFFFFFFFFFFFFFul

        // ---- scaling and integer conversion -------------------------------------------------------
        // 2^k as a binary64 pattern; k outside the normal range gives a subnormal, zero or infinity.
        inline ulong dt_pow2(long k) {
            if (k >= -1022 && k <= 1023) return ((ulong)(k + 1023)) << 52;
            if (k > 1023) return D_INF;
            if (k >= -1074) return 1ul << (k + 1074);
            return 0ul;
        }
        // Multiply by 2^k for any k, stepping so that no intermediate over- or underflows.
        inline ulong dt_ldexp(ulong a, long k) {
            while (k > 1023) { a = d_mul(a, dt_pow2(1023)); k -= 1023; }
            while (k < -1022) { a = d_mul(a, dt_pow2(-1022)); k += 1022; }
            return d_mul(a, dt_pow2(k));
        }
        // Exact for |v| < 2^53; larger values round to nearest, ties to even, through d_finish.
        inline ulong dt_from_long(long v) {
            if (v == 0) return 0ul;
            ulong s = (v < 0) ? 1ul : 0ul;
            ulong m = (v < 0) ? (ulong)(-(v + 1)) + 1ul : (ulong)v;
            long p = 63;
            while (((m >> p) & 1ul) == 0ul) p--;
            ulong mm;
            if (p <= 55) mm = m << (55 - p);
            else { ulong lost = m & ((1ul << (p - 55)) - 1ul); mm = (m >> (p - 55)) | (lost ? 1ul : 0ul); }
            return d_finish(s, p + 1023, mm);
        }
        // Truncation toward zero. Only used on values that fit a long.
        inline long dt_to_long(ulong a) {
            long e = (long)d_exp(a) - 1023;
            if (e < 0) return 0;
            if (e > 62) return (a >> 63) ? (long)0x8000000000000000ul : (long)0x7FFFFFFFFFFFFFFFul;
            ulong m = d_mant(a) | (1ul << 52);
            long r = (e >= 52) ? (long)(m << (e - 52)) : (long)(m >> (52 - e));
            return (a >> 63) ? -r : r;
        }

        // ---- rounding to an integer, in every Arrow RoundMode -------------------------------------
        // Clearing the fractional mantissa bits truncates toward zero; infinities and NaN fall out of
        // the `e >= 52` branch untouched.
        inline ulong dt_trunc(ulong a) {
            long e = (long)d_exp(a) - 1023;
            if (e >= 52) return a;
            if (e < 0) return a & 0x8000000000000000ul;
            return a & ~(0xFFFFFFFFFFFFFul >> e);
        }
        // Is an integral-valued double an even integer? (±0 counts as even, as does anything ≥ 2^53.)
        inline bool dt_is_even(ulong t) {
            if (d_is_zero(t)) return true;
            long e = (long)d_exp(t) - 1023;
            if (e >= 53) return true;
            if (e < 0) return true;
            ulong m = d_mant(t) | (1ul << 52);
            return ((m >> (52 - e)) & 1ul) == 0ul;
        }
        // Arrow RoundMode: 0 DOWN, 1 UP, 2 TOWARDS_ZERO, 3 TOWARDS_INFINITY, 4 HALF_DOWN, 5 HALF_UP,
        // 6 HALF_TOWARDS_ZERO, 7 HALF_TOWARDS_INFINITY, 8 HALF_TO_EVEN, 9 HALF_TO_ODD.
        inline ulong dt_round_int(ulong a, uint mode) {
            if (d_is_nan(a)) return a;
            ulong t = dt_trunc(a);
            if (t == a) return a;                       // integral already (covers ±0 and ±inf)
            ulong sgn = a & 0x8000000000000000ul;
            ulong away = d_add(t, sgn ? DT_MONE : DT_ONE);
            if (mode == 0u) return sgn ? away : t;
            if (mode == 1u) return sgn ? t : away;
            if (mode == 2u) return t;
            if (mode == 3u) return away;
            ulong frac = d_sub(a & DT_ABS, t & DT_ABS);  // exact: the fractional part, in (0, 1)
            if (frac < DT_HALF) return t;
            if (frac > DT_HALF) return away;
            if (mode == 4u) return sgn ? away : t;
            if (mode == 5u) return sgn ? t : away;
            if (mode == 6u) return t;
            if (mode == 7u) return away;
            if (mode == 8u) return dt_is_even(t) ? t : away;
            return dt_is_even(t) ? away : t;
        }

        // ---- square root ---------------------------------------------------------------------------
        // `d_sqrt` is correctly rounded and costs less than the three Newton steps (each a software
        // division) this used to take, so `hypot` simply inherits it.
        inline ulong dt_sqrt(ulong a) { return d_sqrt(a); }

        // ---- natural logarithm ---------------------------------------------------------------------
        inline ulong dt_ln(ulong a) {
            if (d_is_nan(a)) return a | (1ul << 51);
            if (d_is_zero(a)) return DT_NINF;
            if ((a >> 63) != 0ul) return D_QNAN;
            if (d_exp(a) == 0x7FFul) return a;
            ulong x = a;
            long bias = 0;
            if (d_exp(x) == 0ul) { x = d_mul(x, dt_pow2(200)); bias = -200; }
            long e = (long)d_exp(x) - 1023 + bias;
            ulong m = d_mant(x) | (1023ul << 52);         // m in [1, 2)
            if (m > DT_SQRT2) { m -= (1ul << 52); e++; }  // fold to [1/sqrt2, sqrt2]: no cancellation
            ulong s = d_div(d_sub(m, DT_ONE), d_add(m, DT_ONE));
            ulong s2 = d_mul(s, s);
            ulong p = \(horner(lnCoefficients, "s2"));
            ulong lnm = d_mul(d_mul(DT_TWO, s), p);
            if (e == 0) return lnm;
            ulong de = dt_from_long(e);
            // e * ln2 in a hi/lo split: e * LN2_HI is exact, so only the tiny lo term rounds.
            ulong t = d_add(lnm, d_mul(de, DT_LN2_LO));
            return d_add(d_mul(de, DT_LN2_HI), t);
        }

        // ---- exp(x) - 1 ------------------------------------------------------------------------------
        inline ulong dt_expm1(ulong x) {
            if (d_is_nan(x)) return x | (1ul << 51);
            if (d_exp(x) == 0x7FFul) return (x >> 63) ? DT_MONE : x;
            if (d_is_zero(x)) return x;                    // preserves -0
            if (d_exp(x) < 969ul) return x;                // |x| < 2^-54: expm1(x) rounds back to x
            if (d_exp(x) >= 1033ul) return (x >> 63) ? DT_MONE : D_INF;   // |x| >= 1024
            // No reduction below 0.5: the series covers |r| <= 0.6 comfortably, and skipping it avoids
            // the `2^k(1 + E) - 1` cancellation, which costs a couple of ulp right around |x| ~ 0.4.
            long k = (d_exp(x) >= 1022ul) ? dt_to_long(dt_round_int(d_mul(x, DT_INV_LN2), 7u)) : 0;
            ulong r = x;
            if (k != 0) {
                ulong dk = dt_from_long(k);
                r = d_sub(r, d_mul(dk, DT_LN2_HI));
                r = d_sub(r, d_mul(dk, DT_LN2_LO));
            }
            ulong q = \(horner(expCoefficients, "r"));
            ulong E = d_mul(r, q);                          // exp(r) - 1, |r| <= ln2/2
            if (k == 0) return E;
            return d_sub(dt_ldexp(d_add(DT_ONE, E), k), DT_ONE);
        }

        // ---- ln(1 + x) -------------------------------------------------------------------------------
        inline ulong dt_log1p(ulong x) {
            if (d_is_nan(x)) return x | (1ul << 51);
            if (d_is_zero(x)) return x;                     // preserves -0
            if (d_exp(x) == 0x7FFul) return (x >> 63) ? D_QNAN : x;
            if (x == DT_MONE) return DT_NINF;
            if (d_key((long)x) < d_key((long)DT_MONE)) return D_QNAN;
            // |x| < 2^-20: four terms of x - x²/2 + x³/3 - x⁴/4 are already exact to the last bit, and
            // they avoid the two extra roundings the correction below costs.
            if (d_exp(x) < 1003ul) {
                ulong x2 = d_mul(x, x);
                ulong t = d_sub(\(lit(1.0 / 3.0)), d_mul(x, \(lit(0.25))));
                t = d_sub(DT_HALF, d_mul(x, t));
                return d_sub(x, d_mul(x2, t));
            }
            ulong u = d_add(DT_ONE, x);
            if (u == DT_ONE) return x;                      // |x| below half an ulp of 1
            ulong lnu = dt_ln(u);
            ulong um1 = d_sub(u, DT_ONE);                   // exact
            if (um1 == x) return lnu;
            // Kahan's correction: rescale ln(1+x) by the ratio the rounding of 1+x lost.
            return d_div(d_mul(lnu, x), um1);
        }

        // ---- log base b, and hypot -------------------------------------------------------------------
        inline ulong dt_logb(ulong x, ulong b) { return d_div(dt_ln(x), dt_ln(b)); }

        inline ulong dt_hypot(ulong a, ulong b) {
            ulong x = a & DT_ABS, y = b & DT_ABS;
            if (x == D_INF || y == D_INF) return D_INF;     // IEEE: inf wins even over NaN
            if (d_is_nan(x)) return x | (1ul << 51);
            if (d_is_nan(y)) return y | (1ul << 51);
            if (x < y) { ulong t = x; x = y; y = t; }
            if (d_is_zero(x)) return 0ul;
            long e;
            if (d_exp(x) == 0ul) { ulong t = d_mul(x, dt_pow2(200)); e = (long)d_exp(t) - 1023 - 200; }
            else e = (long)d_exp(x) - 1023;
            ulong xs = dt_ldexp(x, -e), ys = dt_ldexp(y, -e);
            ulong s = d_add(d_mul(xs, xs), d_mul(ys, ys));
            return dt_ldexp(dt_sqrt(s), e);
        }
        """
    }()
}
