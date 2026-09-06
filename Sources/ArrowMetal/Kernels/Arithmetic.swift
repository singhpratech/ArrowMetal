import Foundation
import Metal

public enum ArithmeticOp: String, CaseIterable, Sendable {
    case add, sub, mul, div
}

extension MetalArray {
    /// Element-wise arithmetic with a scalar. Integer overflow wraps; integer division by zero is unspecified
    /// (the GPU does not trap), matching Arrow's non-checked kernels only for overflow.
    public func arithmetic(_ op: ArithmeticOp, _ scalar: T) throws -> MetalArray<T> {
        guard Dispatch.runsOnGPU(T.self) else { return try CPUReference.arithmetic(self, op, scalar: scalar) }
        try Dispatch.checkLength(length)
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: length * T.byteWidth, zeroed: false, context: ctx)
        let src = KernelSource.arithmetic(T: T.mslType)
        let pso = try Dispatch.pipeline(ctx, family: "arith", source: src, function: "arith_scalar_\(op.rawValue)", type: T.mslType)
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            Dispatch.setScalar(enc, scalar, index: 1)
            Dispatch.setUInt(enc, length, index: 2)
            enc.setBuffer(out.mtl, offset: out.offset, index: 3)
            Dispatch.dispatch1D(enc, pso, count: (length + 3) / 4)
        }
        return MetalArray<T>(length: length, nullCount: nullCount, validity: validity, values: out, context: ctx)
    }

    /// Element-wise arithmetic with another array. Output validity is the AND of both inputs.
    public func arithmetic(_ op: ArithmeticOp, _ other: MetalArray<T>) throws -> MetalArray<T> {
        guard other.length == length else { throw ArrowMetalError.lengthMismatch(length, other.length) }
        guard Dispatch.runsOnGPU(T.self) else { return try CPUReference.arithmetic(self, op, array: other) }
        try Dispatch.checkLength(length)
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: length * T.byteWidth, zeroed: false, context: ctx)
        let src = KernelSource.arithmetic(T: T.mslType)
        let pso = try Dispatch.pipeline(ctx, family: "arith", source: src, function: "arith_array_\(op.rawValue)", type: T.mslType)
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            enc.setBuffer(other.values.mtl, offset: other.values.offset, index: 1)
            Dispatch.setUInt(enc, length, index: 2)
            enc.setBuffer(out.mtl, offset: out.offset, index: 3)
            Dispatch.dispatch1D(enc, pso, count: (length + 3) / 4)
        }
        let v = try BitmapOps.combineValidity(ctx, validity, other.validity, bits: length)
        let res = MetalArray<T>(length: length, nullCount: 0, validity: v, values: out, context: ctx)
        res.recomputeNullCount()
        return res
    }

    public func add(_ s: T) throws -> MetalArray<T> { try arithmetic(.add, s) }
    public func subtract(_ s: T) throws -> MetalArray<T> { try arithmetic(.sub, s) }
    public func multiply(_ s: T) throws -> MetalArray<T> { try arithmetic(.mul, s) }
    public func divide(_ s: T) throws -> MetalArray<T> { try arithmetic(.div, s) }
    public func add(_ o: MetalArray<T>) throws -> MetalArray<T> { try arithmetic(.add, o) }
    public func subtract(_ o: MetalArray<T>) throws -> MetalArray<T> { try arithmetic(.sub, o) }
    public func multiply(_ o: MetalArray<T>) throws -> MetalArray<T> { try arithmetic(.mul, o) }
    public func divide(_ o: MetalArray<T>) throws -> MetalArray<T> { try arithmetic(.div, o) }
}
