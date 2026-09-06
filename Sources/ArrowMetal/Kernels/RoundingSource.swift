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
///     `sqrt` is **correctly rounded** (`d_sqrt`, a digit-by-digit extraction), and `exp`, `ln`,
///     `log10`, `log2` and `power` run entirely in software binary64 through `DoublePower` — within
///     2 ulp, measured. `modulo` on `float64` is still not implemented — `Rounding.swift` throws
///     rather than return a silently poor answer.
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
            // Arrow `sign`: NaN and both zeros come back unchanged, everything else is ±1. Decided on
            // the bit pattern rather than with `>`/`<`, which flush subnormal operands to zero and
            // would report sign(1.4e-45) as 1.4e-45 instead of 1.
            inline \(T) m_sign(\(T) a) {
                uint b = as_type<uint>(a), mag = b & 0x7FFFFFFFu;
                if (mag == 0u || mag > 0x7F800000u) return a;      // ±0 and NaN pass through
                return (b & 0x80000000u) ? -1.0f : 1.0f;
            }
            inline \(T) m_sqrt(\(T) a) { return sqrt(a); }
            inline \(T) m_exp(\(T) a) { return exp(a); }
            inline \(T) m_ln(\(T) a) { return log(a); }
            inline \(T) m_log10(\(T) a) { return log10(a); }
            inline \(T) m_log2(\(T) a) { return log2(a); }
            // A subnormal operand is flushed to zero before the library rounding functions see it,
            // which would make ceil(1.4e-45) zero instead of one. The answer is decided from the bit
            // pattern in that range: |a| < 1, so only the sign matters.
            inline bool f_subnormal(uint b) { return (b & 0x7F800000u) == 0u && (b & 0x007FFFFFu) != 0u; }
            inline \(T) m_floor(\(T) a) {
                uint b = as_type<uint>(a);
                if (f_subnormal(b)) return (b & 0x80000000u) ? -1.0f : 0.0f;
                return floor(a);
            }
            inline \(T) m_ceil(\(T) a) {
                uint b = as_type<uint>(a);
                if (f_subnormal(b)) return (b & 0x80000000u) ? as_type<\(T)>(0x80000000u) : 1.0f;
                return ceil(a);
            }
            // MSL `round` is C99 `round`: ties away from zero, which is the mode ArrowMetal defines.
            // A rounded-to-zero result keeps the sign of its operand -- round(-0.4) is -0.0 -- which
            // the library function drops here but the float64 kernel below preserves.
            inline \(T) m_round(\(T) a) {
                \(T) r = round(a);
                if ((as_type<uint>(r) & 0x7FFFFFFFu) != 0u) return r;
                return as_type<\(T)>(as_type<uint>(a) & 0x80000000u);
            }
            inline \(T) m_trunc(\(T) a) {
                uint b = as_type<uint>(a);
                if (f_subnormal(b)) return as_type<\(T)>(b & 0x80000000u);
                return trunc(a);
            }
            inline \(T) m_power(\(T) a, \(T) b) { return pow(a, b); }
            inline \(T) m_modulo(\(T) a, \(T) b) { return fmod(a, b); }
            // Order-preserving unsigned key: comparing these compares the floats exactly, without the
            // flush-to-zero the arithmetic comparison operators apply to subnormal operands.
            inline uint f_ord(uint b) { return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
            // Element-wise min/max skip NaN (a NaN operand loses), like the reductions in this library,
            // and follow `fmin`/`fmax` on a ±0 tie: min keeps -0.0, max keeps 0.0, whichever side it is
            // on, so the pair is commutative.
            inline \(T) m_min_ew(\(T) a, \(T) b) {
                if (isnan(a)) return b;
                if (isnan(b)) return a;
                uint ka = as_type<uint>(a), kb = as_type<uint>(b);
                if (((ka | kb) & 0x7FFFFFFFu) == 0u) return as_type<\(T)>((ka | kb) & 0x80000000u);
                return f_ord(ka) < f_ord(kb) ? a : b;
            }
            inline \(T) m_max_ew(\(T) a, \(T) b) {
                if (isnan(a)) return b;
                if (isnan(b)) return a;
                uint ka = as_type<uint>(a), kb = as_type<uint>(b);
                if (((ka | kb) & 0x7FFFFFFFu) == 0u) return as_type<\(T)>((ka & kb) & 0x80000000u);
                return f_ord(kb) < f_ord(ka) ? a : b;
            }

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
        var s = KernelSource.prelude + DoubleMath.msl + DoubleTranscendental.msl + DoublePower.msl + """

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
        // On a ±0 tie `d_key` calls the two equal, so the sign is chosen the way `fmin`/`fmax` do:
        // min keeps -0.0 and max keeps 0.0 whichever side it came from, which also makes the pair
        // commutative.
        inline ulong m_min_ew(ulong a, ulong b) {
            if (d_is_nan(a)) return b;
            if (d_is_nan(b)) return a;
            if (((a | b) & 0x7FFFFFFFFFFFFFFFul) == 0ul) return (a | b) & 0x8000000000000000ul;
            return m_dlt(a, b) ? a : b;
        }
        inline ulong m_max_ew(ulong a, ulong b) {
            if (d_is_nan(a)) return b;
            if (d_is_nan(b)) return a;
            if (((a | b) & 0x7FFFFFFFFFFFFFFFul) == 0ul) return (a & b) & 0x8000000000000000ul;
            return m_dlt(b, a) ? a : b;
        }

        // ---- software binary64 transcendentals: no float detour, no lost digits.
        inline ulong m_sqrt(ulong a) { return d_sqrt(a); }
        inline ulong m_exp(ulong a) { return dp_exp(a); }
        inline ulong m_ln(ulong a) { return dp_ln(a); }
        inline ulong m_log10(ulong a) { return dp_log10(a); }
        inline ulong m_log2(ulong a) { return dp_log2(a); }
        inline ulong m_power(ulong a, ulong b) { return dp_pow(a, b); }

        """
        s += kernels(T: "ulong", unary: allUnary, binary: ["power"])
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
