import Foundation
import Metal

/// Arrow's floating point classification predicates: `is_nan`, `is_finite` and `is_inf`.
///
/// Arrow defines all three over every numeric type, not only the float ones, so an integer column
/// answers the constant it must (`is_finite` true everywhere, the other two false everywhere) rather
/// than throwing. Nulls propagate: the result shares the input's validity bitmap zero-copy, so a
/// null element gives a **null** predicate, which is what `pyarrow.compute.is_nan` does.
///
/// On the float types the test is done on the raw bit pattern, one thread per 32-bit output word, so
/// Float64 needs no software binary64 and Float32 is unaffected by the GPU's flush-to-zero of
/// subnormal *arithmetic* — nothing here does arithmetic.
public enum FloatClassOp: String, CaseIterable, Sendable {
    case isNan = "is_nan"
    case isFinite = "is_finite"
    case isInf = "is_inf"

    var kernel: String {
        switch self {
        case .isNan: return "fc_is_nan"
        case .isFinite: return "fc_is_finite"
        case .isInf: return "fc_is_inf"
        }
    }
    /// The answer for an integer column, where every value is an ordinary finite number.
    var integerAnswer: Bool { self == .isFinite }
}

enum FloatClassSource {
    /// `B` is the unsigned integer the bit pattern is read as, `absMask` clears the sign bit and
    /// `inf` is the exponent-all-ones, mantissa-zero pattern.
    static func source(B: String, absMask: String, inf: String) -> String {
        var s = KernelSource.prelude + "\n"
        let tests: [(String, String)] = [
            ("fc_is_nan", "(b & \(absMask)) > \(inf)"),
            ("fc_is_finite", "(b & \(absMask)) < \(inf)"),
            ("fc_is_inf", "(b & \(absMask)) == \(inf)"),
        ]
        for (name, test) in tests {
            s += """
            kernel void \(name)(device const \(B)* a [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                                device uint* out [[buffer(2)]], uint w [[thread_position_in_grid]]) {
                uint n = *nPtr;
                uint base = w * 32u;
                if (base >= n) return;
                uint limit = min(32u, n - base);
                uint bits = 0;
                for (uint j = 0; j < limit; j++) {
                    \(B) b = a[base + j];
                    if (\(test)) bits |= (1u << j);
                }
                out[w] = bits;
            }

            """
        }
        return s
    }

    static let float32 = source(B: "uint", absMask: "0x7FFFFFFFu", inf: "0x7F800000u")
    static let float64 = source(B: "ulong", absMask: "0x7FFFFFFFFFFFFFFFul", inf: "0x7FF0000000000000ul")
}

extension MetalArray {
    /// One of `is_nan` / `is_finite` / `is_inf`. Null in, null out.
    public func floatClass(_ op: FloatClassOp) throws -> MetalBooleanArray {
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let ctx = context
        // Integer columns: the answer is a constant, produced as a bitmap fill (still on the GPU).
        guard T.isFloatingPoint else {
            let out = op.integerAnswer
                ? try Structural.filledBitmap(ctx, bits: n, lengthBuffer: lengthBuffer)
                : try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: n), context: ctx)
            return inheritPending(MetalBooleanArray(length: knownLength, nullCount: _nullCount,
                                                    validity: validity, values: out, context: ctx))
        }
        let isDouble = T.self == Double.self
        let src = isDouble ? FloatClassSource.float64 : FloatClassSource.float32
        let out = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: n), zeroed: false, context: ctx)
        let words = BitmapOps.words(bits: n)
        if words > 0 {
            let pso = try Dispatch.pipeline(ctx, family: "floatclass", source: src, function: op.kernel,
                                            type: isDouble ? "float64" : "float32")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                Dispatch.setLength(enc, n, lengthBuffer, index: 1)
                enc.setBuffer(out.mtl, offset: out.offset, index: 2)
                Dispatch.dispatch1D(enc, pso, count: words)
            }
            ctx.retainUntilFlush(self)
        }
        return inheritPending(MetalBooleanArray(length: knownLength, nullCount: _nullCount,
                                                validity: validity, values: out, context: ctx))
    }

    /// Arrow `is_nan`. False on every integer column; null where the input is null.
    public func isNan() throws -> MetalBooleanArray { try floatClass(.isNan) }
    /// Arrow `is_finite`. True on every integer column; null where the input is null.
    public func isFinite() throws -> MetalBooleanArray { try floatClass(.isFinite) }
    /// Arrow `is_inf`. False on every integer column; null where the input is null.
    public func isInf() throws -> MetalBooleanArray { try floatClass(.isInf) }
}
