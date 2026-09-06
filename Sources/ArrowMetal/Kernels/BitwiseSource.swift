import Foundation

/// Small MSL type helpers shared by the bit-wise, element-wise math and cumulative kernels.
///
/// The kernels below need three things the `ArrowPrimitive` protocol does not spell out: the unsigned
/// MSL type of the same width (so wrapping shifts and multiplies are written on a type where C defines
/// them), whether the element type is signed, and its bit width.
enum MathTypes {
    /// The unsigned MSL type of the same width. Wrapping arithmetic is written on this type because
    /// signed overflow is undefined in C while unsigned overflow is modular.
    static func unsigned(_ msl: String) -> String {
        switch msl {
        case "char": return "uchar"
        case "short": return "ushort"
        case "int": return "uint"
        case "long": return "ulong"
        default: return msl
        }
    }

    static func isSigned<T: ArrowPrimitive>(_: T.Type) -> Bool { T.minValue < 0 as T }
    static func bitWidth<T: ArrowPrimitive>(_: T.Type) -> Int { T.byteWidth * 8 }

    /// Throws unless `T` is one of the eight integer primitives.
    static func requireInteger<T: ArrowPrimitive>(_: T.Type, _ what: String) throws {
        if T.isFloatingPoint {
            throw ArrowMetalError.unsupportedType("\(what) needs an integer column, got \(T.arrowFormat)")
        }
    }

    /// Throws unless `T` is `float32` or `float64`.
    static func requireFloat<T: ArrowPrimitive>(_: T.Type, _ what: String) throws {
        if !T.isFloatingPoint {
            throw ArrowMetalError.unsupportedType("\(what) needs a floating point column, got \(T.arrowFormat) — cast first")
        }
    }

    /// MSL literal expressions for the neutral element of each cumulative op, in the kernel's value type.
    /// Built from shifts rather than decimal literals so that `int64`'s bounds need no suffix games.
    static func identities<T: ArrowPrimitive>(_: T.Type) -> (sum: String, min: String, max: String) {
        if T.self == Double.self { return ("0ul", "0x7FF0000000000000ul", "0xFFF0000000000000ul") }
        if T.isFloatingPoint { return ("0.0f", "INFINITY", "-INFINITY") }
        let t = T.mslType, u = unsigned(t), w = bitWidth(T.self)
        if isSigned(T.self) {
            return ("(\(t))0",
                    "(\(t))((((\(u))1 << \(w - 1)) - (\(u))1))",   // T.max
                    "(\(t))((\(u))1 << \(w - 1))")                 // T.min
        }
        return ("(\(t))0", "(\(t))(~(\(u))0)", "(\(t))0")
    }
}

/// MSL for Arrow's bit-wise and shift functions over the eight integer primitives.
///
/// **Defined semantics for shifts.** Arrow raises on a shift count that is negative or at least the
/// width of the type, and C leaves the same case undefined. ArrowMetal defines it instead, identically
/// on the GPU and in the CPU oracle: a count outside `[0, bitWidth)` behaves as if it were the bit
/// width, so `shift_left` and unsigned `shift_right` yield 0, and signed `shift_right` yields the sign
/// fill (0 for a non-negative value, -1 for a negative one). Counts are read in the element type, so a
/// negative count on a signed column and a huge count on an unsigned one are both out of range.
/// In-range shifts wrap: bits shifted out of the left are dropped, `shift_right` is arithmetic on
/// signed columns and logical on unsigned ones.
enum BitwiseSource {
    static func source(T: String, U: String, signed: Bool, width: Int) -> String {
        // Out-of-range shift counts collapse to -1, which each shift turns into its defined result.
        let count = signed
            ? "inline long bw_count(\(T) s) { long k = (long)s; return (k < 0 || k >= \(width)) ? -1 : k; }"
            : "inline long bw_count(\(T) s) { ulong k = (ulong)s; return (k >= \(width)ul) ? -1 : (long)k; }"
        let signFill = signed ? "(a < (\(T))0 ? (\(T))(~(\(U))0) : (\(T))0)" : "(\(T))0"

        var s = KernelSource.prelude + """

        \(count)
        inline \(T) bw_and(\(T) a, \(T) b) { return (\(T))(a & b); }
        inline \(T) bw_or (\(T) a, \(T) b) { return (\(T))(a | b); }
        inline \(T) bw_xor(\(T) a, \(T) b) { return (\(T))(a ^ b); }
        // Shift left in the unsigned type: C defines the wrap there, and shifting into a sign bit is
        // undefined on the signed one.
        inline \(T) bw_shl(\(T) a, \(T) b) {
            long k = bw_count(b);
            if (k < 0) return (\(T))0;
            return (\(T))((\(U))((\(U))a << k));
        }
        // Arithmetic on signed columns, logical on unsigned ones: C's `>>` already does exactly that.
        inline \(T) bw_shr(\(T) a, \(T) b) {
            long k = bw_count(b);
            if (k < 0) return \(signFill);
            return (\(T))(a >> k);
        }
        kernel void bw_not(device const \(T)* a [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                           device \(T)* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
            if (i < *nPtr) out[i] = (\(T))(~a[i]);
        }

        """
        for op in ["and", "or", "xor", "shl", "shr"] {
            s += """
            kernel void bw_scalar_\(op)(device const \(T)* a [[buffer(0)]], constant \(T)& scalar [[buffer(1)]],
                                        device const uint* nPtr [[buffer(2)]], device \(T)* out [[buffer(3)]],
                                        uint i [[thread_position_in_grid]]) {
                if (i < *nPtr) out[i] = bw_\(op)(a[i], scalar);
            }
            kernel void bw_array_\(op)(device const \(T)* a [[buffer(0)]], device const \(T)* b [[buffer(1)]],
                                       device const uint* nPtr [[buffer(2)]], device \(T)* out [[buffer(3)]],
                                       uint i [[thread_position_in_grid]]) {
                if (i < *nPtr) out[i] = bw_\(op)(a[i], b[i]);
            }

            """
        }
        return s
    }
}
