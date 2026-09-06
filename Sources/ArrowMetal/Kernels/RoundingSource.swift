import Foundation

/// MSL for the element-wise arithmetic beyond `add`/`sub`/`mul`/`div`: sign manipulation, roots,
/// exponentials, logarithms, the four rounding functions, `power`, `modulo` and element-wise min/max.
///
/// Three flavours are generated:
///   * integer columns — everything defined on bit patterns and wrapping arithmetic;
///   * `float32` — the MSL library functions, compiled with `mathMode = .safe`;
///   * `float64` — values travel as raw `ulong` bit patterns because Metal has no `double`. Sign, abs,
///     `floor`/`ceil`/`round`/`trunc` and element-wise min/max are **exact**, done on the bit pattern
///     with the software binary64 adder from `DoubleMath` for the one carry each rounding step needs.
///     `sqrt`/`exp`/`ln`/`log10`/`log2` are **not** exact: they convert to `float`, evaluate there and
///     widen back, so expect about 7 correct significant decimal digits (relative error up to ~1e-6)
///     and no subnormal or above-`float`-range results. `power` and `modulo` on `float64` are not
///     implemented at all — `Rounding.swift` throws rather than return a silently poor answer.
enum RoundingSource {
    /// Rounding of a value away from zero adds ±1, which needs a real binary64 add.
    private static let dOne = "0x3FF0000000000000ul"
    private static let dMinusOne = "0xBFF0000000000000ul"

    // MARK: - Integer and float32

    static func source(T: String, U: String, isFloat: Bool, isSigned: Bool) -> String {
        var s = KernelSource.prelude + "\n"
        if isFloat {
            s += """
            inline \(T) m_negate(\(T) a) { return -a; }
            inline \(T) m_abs(\(T) a) { return fabs(a); }
            // Arrow `sign`: NaN and both zeros come back unchanged, everything else is ±1.
            inline \(T) m_sign(\(T) a) { return (a > 0.0f) ? 1.0f : ((a < 0.0f) ? -1.0f : a); }
            inline \(T) m_sqrt(\(T) a) { return sqrt(a); }
            inline \(T) m_exp(\(T) a) { return exp(a); }
            inline \(T) m_ln(\(T) a) { return log(a); }
            inline \(T) m_log10(\(T) a) { return log10(a); }
            inline \(T) m_log2(\(T) a) { return log2(a); }
            inline \(T) m_floor(\(T) a) { return floor(a); }
            inline \(T) m_ceil(\(T) a) { return ceil(a); }
            // MSL `round` is C99 `round`: ties away from zero, which is the mode ArrowMetal defines.
            inline \(T) m_round(\(T) a) { return round(a); }
            inline \(T) m_trunc(\(T) a) { return trunc(a); }
            inline \(T) m_power(\(T) a, \(T) b) { return pow(a, b); }
            inline \(T) m_modulo(\(T) a, \(T) b) { return fmod(a, b); }
            // Element-wise min/max skip NaN (a NaN operand loses), like the reductions in this library.
            inline \(T) m_min_ew(\(T) a, \(T) b) { if (isnan(a)) return b; if (isnan(b)) return a; return a < b ? a : b; }
            inline \(T) m_max_ew(\(T) a, \(T) b) { if (isnan(a)) return b; if (isnan(b)) return a; return a > b ? a : b; }

            """
        } else {
            let absBody = isSigned ? "(a < (\(T))0 ? (\(T))((\(T))0 - a) : a)" : "a"
            let signBody = isSigned ? "(\(T))((a > (\(T))0) - (a < (\(T))0))" : "(\(T))(a > (\(T))0 ? 1 : 0)"
            // `a % -1` overflows for T.min in C; it is 0 for every value, so answer it directly.
            let modGuard = isSigned ? "if (b == (\(T))(~(\(U))0)) return (\(T))0;" : ""
            let negExp = isSigned ? "if ((long)b < 0) return (\(T))0;" : ""
            s += """
            // Wrapping two's complement negate and abs: abs(T.min) is T.min, as in unchecked Arrow.
            inline \(T) m_negate(\(T) a) { return (\(T))((\(U))0 - (\(U))a); }
            inline \(T) m_abs(\(T) a) { return \(absBody); }
            inline \(T) m_sign(\(T) a) { return \(signBody); }
            // Rounding an integer is the identity; the column keeps its type.
            inline \(T) m_floor(\(T) a) { return a; }
            inline \(T) m_ceil(\(T) a) { return a; }
            inline \(T) m_round(\(T) a) { return a; }
            inline \(T) m_trunc(\(T) a) { return a; }
            // Repeated squaring, wrapping. A negative exponent is defined as 0 (Arrow raises instead).
            inline \(T) m_power(\(T) a, \(T) b) {
                \(negExp)
                ulong e = (ulong)b;
                \(U) base = (\(U))a, r = (\(U))1;
                while (e != 0ul) {
                    if (e & 1ul) r = (\(U))(r * base);
                    base = (\(U))(base * base);
                    e >>= 1;
                }
                return (\(T))r;
            }
            // C remainder semantics (the sign follows the dividend). Division by zero is defined as 0,
            // matching the `divide` kernel and its CPU reference.
            inline \(T) m_modulo(\(T) a, \(T) b) {
                if (b == (\(T))0) return (\(T))0;
                \(modGuard)
                return (\(T))(a % b);
            }
            inline \(T) m_min_ew(\(T) a, \(T) b) { return a < b ? a : b; }
            inline \(T) m_max_ew(\(T) a, \(T) b) { return a > b ? a : b; }

            """
        }
        return s + kernels(T: T, unary: isFloat ? allUnary : integerUnary, binary: ["power", "modulo"])
    }

    // MARK: - Float64

    static let double: String = {
        var s = KernelSource.prelude + DoubleMath.msl + """

        // ---- binary64 <-> float conversions, used only by the transcendental kernels.
        inline float d2f(ulong b) {
            ulong sgn = b >> 63;
            long e = (long)((b >> 52) & 0x7FFul);
            ulong m = b & 0xFFFFFFFFFFFFFul;
            if (e == 0x7FF) {
                if (m != 0ul) return as_type<float>(0x7FC00000u);
                return sgn ? -INFINITY : INFINITY;
            }
            float sig = ldexp((float)m, -52);
            if (e == 0) { if (m == 0ul) return sgn ? -0.0f : 0.0f; e = 1; }
            else sig += 1.0f;
            float r = ldexp(sig, (int)(e - 1023));
            return sgn ? -r : r;
        }
        inline ulong f2d(float x) {
            uint b = as_type<uint>(x);
            ulong sgn = (ulong)(b >> 31);
            uint e = (b >> 23) & 0xFFu;
            uint m = b & 0x7FFFFFu;
            if (e == 0xFFu) return (sgn << 63) | (m != 0u ? D_QNAN : D_INF);
            if (e == 0u) {
                if (m == 0u) return sgn << 63;
                long ee = -126;
                while ((m & 0x800000u) == 0u) { m <<= 1; ee--; }
                m &= 0x7FFFFFu;
                return (sgn << 63) | ((ulong)(ee + 1023) << 52) | ((ulong)m << 29);
            }
            return (sgn << 63) | ((ulong)((long)e - 127 + 1023) << 52) | ((ulong)m << 29);
        }

        // ---- exact bit-pattern operations
        inline ulong m_negate(ulong a) { return a ^ 0x8000000000000000ul; }
        inline ulong m_abs(ulong a) { return a & 0x7FFFFFFFFFFFFFFFul; }
        inline ulong m_sign(ulong a) {
            if (d_is_nan(a) || d_is_zero(a)) return a;
            return (a >> 63) ? \(dMinusOne) : \(dOne);
        }
        // Clearing the fractional mantissa bits truncates toward zero; infinities and NaN fall out of
        // the `e >= 52` branch untouched.
        inline ulong m_trunc(ulong a) {
            long e = (long)((a >> 52) & 0x7FFul) - 1023;
            if (e >= 52) return a;
            if (e < 0) return a & 0x8000000000000000ul;
            return a & ~(0xFFFFFFFFFFFFFul >> e);
        }
        inline ulong m_floor(ulong a) {
            if (d_is_nan(a)) return a;
            ulong t = m_trunc(a);
            if (t == a) return a;
            return (a >> 63) ? d_add(t, \(dMinusOne)) : t;
        }
        inline ulong m_ceil(ulong a) {
            if (d_is_nan(a)) return a;
            ulong t = m_trunc(a);
            if (t == a) return a;
            return (a >> 63) ? t : d_add(t, \(dOne));
        }
        // Half away from zero: the top fractional mantissa bit decides, then ±1 is added exactly.
        inline ulong m_round(ulong a) {
            if (d_is_nan(a)) return a;
            long e = (long)((a >> 52) & 0x7FFul) - 1023;
            if (e >= 52) return a;
            ulong sgn = a & 0x8000000000000000ul;
            if (e < 0) return (e == -1) ? (sgn | \(dOne)) : sgn;
            ulong fmask = (1ul << (52 - e)) - 1ul;
            ulong t = a & ~fmask;
            if ((a & fmask) >= (1ul << (51 - e))) t = d_add(t, sgn ? \(dMinusOne) : \(dOne));
            return t;
        }
        inline bool m_dlt(ulong a, ulong b) { return d_key((long)a) < d_key((long)b); }
        inline ulong m_min_ew(ulong a, ulong b) {
            if (d_is_nan(a)) return b;
            if (d_is_nan(b)) return a;
            return m_dlt(a, b) ? a : b;
        }
        inline ulong m_max_ew(ulong a, ulong b) {
            if (d_is_nan(a)) return b;
            if (d_is_nan(b)) return a;
            return m_dlt(b, a) ? a : b;
        }

        // ---- float-precision transcendentals, widened back to binary64
        inline ulong m_sqrt(ulong a) { return f2d(sqrt(d2f(a))); }
        inline ulong m_exp(ulong a) { return f2d(exp(d2f(a))); }
        inline ulong m_ln(ulong a) { return f2d(log(d2f(a))); }
        inline ulong m_log10(ulong a) { return f2d(log10(d2f(a))); }
        inline ulong m_log2(ulong a) { return f2d(log2(d2f(a))); }

        """
        s += kernels(T: "ulong", unary: allUnary, binary: [])
        return s
    }()

    // MARK: - Kernel bodies

    static let allUnary = ["negate", "abs", "sign", "sqrt", "exp", "ln", "log10", "log2", "floor", "ceil", "round", "trunc"]
    static let integerUnary = ["negate", "abs", "sign", "floor", "ceil", "round", "trunc"]

    private static func kernels(T: String, unary: [String], binary: [String]) -> String {
        var s = ""
        for op in unary {
            s += """
            kernel void math_unary_\(op)(device const \(T)* a [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                                         device \(T)* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
                if (i < *nPtr) out[i] = m_\(op)(a[i]);
            }

            """
        }
        for op in binary {
            s += """
            kernel void math_scalar_\(op)(device const \(T)* a [[buffer(0)]], constant \(T)& scalar [[buffer(1)]],
                                          device const uint* nPtr [[buffer(2)]], device \(T)* out [[buffer(3)]],
                                          uint i [[thread_position_in_grid]]) {
                if (i < *nPtr) out[i] = m_\(op)(a[i], scalar);
            }
            kernel void math_array_\(op)(device const \(T)* a [[buffer(0)]], device const \(T)* b [[buffer(1)]],
                                         device const uint* nPtr [[buffer(2)]], device \(T)* out [[buffer(3)]],
                                         uint i [[thread_position_in_grid]]) {
                if (i < *nPtr) out[i] = m_\(op)(a[i], b[i]);
            }

            """
        }
        // Element-wise min/max is the one binary op that must read both validity bitmaps: Arrow skips
        // nulls here, so a null on one side yields the other side's value rather than a null.
        // flags: bit0 = a has validity, bit1 = b has validity.
        s += """
        kernel void math_minmax(device const \(T)* a [[buffer(0)]], device const \(T)* b [[buffer(1)]],
                                device const uchar* va [[buffer(2)]], device const uchar* vb [[buffer(3)]],
                                device const uint* nPtr [[buffer(4)]], constant uint& flags [[buffer(5)]],
                                constant uint& isMax [[buffer(6)]], device \(T)* out [[buffer(7)]],
                                uint i [[thread_position_in_grid]]) {
            if (i >= *nPtr) return;
            bool oka = (flags & 1u) ? bit_get(va, i) : true;
            bool okb = (flags & 2u) ? bit_get(vb, i) : true;
            \(T) x = a[i], y = b[i];
            if (oka && okb) out[i] = isMax ? m_max_ew(x, y) : m_min_ew(x, y);
            else if (oka) out[i] = x;
            else if (okb) out[i] = y;
            else out[i] = (\(T))0;
        }

        """
        return s
    }
}
