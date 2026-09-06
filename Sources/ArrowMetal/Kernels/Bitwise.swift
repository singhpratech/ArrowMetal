import Foundation
import Metal

/// Arrow's bit-wise and shift functions. Integer columns only.
public enum BitwiseOp: String, CaseIterable, Sendable {
    case and, or, xor
    /// Arrow `shift_left`.
    case shl
    /// Arrow `shift_right`: arithmetic on signed columns, logical on unsigned ones.
    case shr
}

extension MetalArray {
    /// Bit-wise / shift op against a scalar. Output validity is shared with the input, zero-copy.
    ///
    /// Integer columns only; a float column throws `unsupportedType`. Shift counts outside
    /// `[0, bitWidth)` are defined (see `BitwiseSource`): `shl` and unsigned `shr` give 0, signed `shr`
    /// gives the sign fill.
    public func bitwise(_ op: BitwiseOp, _ scalar: T) throws -> MetalArray<T> {
        try MathTypes.requireInteger(T.self, "bit_wise_\(op.rawValue)")
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: n * T.byteWidth, zeroed: false, context: ctx)
        let pso = try bitwisePipeline("bw_scalar_\(op.rawValue)")
        if n > 0 {
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                Dispatch.setScalar(enc, scalar, index: 1)
                Dispatch.setLength(enc, n, lengthBuffer, index: 2)
                enc.setBuffer(out.mtl, offset: out.offset, index: 3)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        return inheritPending(MetalArray<T>(length: knownLength, nullCount: _nullCount, validity: validity, values: out, context: ctx))
    }

    /// Bit-wise / shift op against another column of the same length. Output validity is the AND of both
    /// inputs, matching Arrow's null propagation for binary kernels.
    public func bitwise(_ op: BitwiseOp, _ other: MetalArray<T>) throws -> MetalArray<T> {
        try MathTypes.requireInteger(T.self, "bit_wise_\(op.rawValue)")
        try checkSameLength(other)
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: n * T.byteWidth, zeroed: false, context: ctx)
        let pso = try bitwisePipeline("bw_array_\(op.rawValue)")
        if n > 0 {
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                enc.setBuffer(other.values.mtl, offset: other.values.offset, index: 1)
                Dispatch.setLength(enc, n, lengthBuffer, index: 2)
                enc.setBuffer(out.mtl, offset: out.offset, index: 3)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        let v = try BitmapOps.combineValidity(ctx, validity, other.validity, bits: n, lengthBuffer: lengthBuffer)
        let res = inheritPending(MetalArray<T>(length: knownLength, nullCount: 0, validity: v, values: out, context: ctx))
        res.recomputeNullCount()
        ctx.retainUntilFlush(self)
        ctx.retainUntilFlush(other)
        return res
    }

    /// Arrow `bit_wise_not`: the one's complement of every value. Validity is shared with the input.
    public func bitwiseNot() throws -> MetalArray<T> {
        try MathTypes.requireInteger(T.self, "bit_wise_not")
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: n * T.byteWidth, zeroed: false, context: ctx)
        let pso = try bitwisePipeline("bw_not")
        if n > 0 {
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                Dispatch.setLength(enc, n, lengthBuffer, index: 1)
                enc.setBuffer(out.mtl, offset: out.offset, index: 2)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        return inheritPending(MetalArray<T>(length: knownLength, nullCount: _nullCount, validity: validity, values: out, context: ctx))
    }

    private func bitwisePipeline(_ function: String) throws -> MTLComputePipelineState {
        let src = BitwiseSource.source(T: T.mslType, U: MathTypes.unsigned(T.mslType),
                                       signed: MathTypes.isSigned(T.self), width: MathTypes.bitWidth(T.self))
        return try Dispatch.pipeline(context, family: "bitwise", source: src, function: function, type: T.mslType)
    }

    // MARK: - Named forms

    public func bitwiseAnd(_ s: T) throws -> MetalArray<T> { try bitwise(.and, s) }
    public func bitwiseOr(_ s: T) throws -> MetalArray<T> { try bitwise(.or, s) }
    public func bitwiseXor(_ s: T) throws -> MetalArray<T> { try bitwise(.xor, s) }
    public func shiftLeft(_ s: T) throws -> MetalArray<T> { try bitwise(.shl, s) }
    public func shiftRight(_ s: T) throws -> MetalArray<T> { try bitwise(.shr, s) }

    public func bitwiseAnd(_ o: MetalArray<T>) throws -> MetalArray<T> { try bitwise(.and, o) }
    public func bitwiseOr(_ o: MetalArray<T>) throws -> MetalArray<T> { try bitwise(.or, o) }
    public func bitwiseXor(_ o: MetalArray<T>) throws -> MetalArray<T> { try bitwise(.xor, o) }
    public func shiftLeft(_ o: MetalArray<T>) throws -> MetalArray<T> { try bitwise(.shl, o) }
    public func shiftRight(_ o: MetalArray<T>) throws -> MetalArray<T> { try bitwise(.shr, o) }
}
