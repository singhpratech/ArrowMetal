import Foundation
import Metal

public enum CompareOp: String, CaseIterable, Sendable {
    case eq, ne, lt, le, gt, ge
    func eval<T: Comparable>(_ a: T, _ b: T) -> Bool {
        switch self {
        case .eq: return a == b
        case .ne: return a != b
        case .lt: return a < b
        case .le: return a <= b
        case .gt: return a > b
        case .ge: return a >= b
        }
    }
}

extension MetalArray {
    /// Element-wise comparison with a scalar, producing an Arrow boolean array.
    /// Null inputs produce null outputs (validity bitmap is shared zero-copy with the input).
    public func compare(_ op: CompareOp, _ scalar: T) throws -> MetalBooleanArray {
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let ctx = context
        let words = BitmapOps.words(bits: n)
        let out = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: n), zeroed: false, context: ctx)
        let isDouble = T.self == Double.self
        let src = isDouble ? KernelSource.compareDouble : KernelSource.compare(T: T.mslType)
        let pso = try Dispatch.pipeline(ctx, family: "cmp", source: src, function: "cmp_scalar_\(op.rawValue)", type: isDouble ? "double" : T.mslType)
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            if isDouble { Dispatch.setScalar(enc, Int64(bitPattern: (scalar as! Double).bitPattern), index: 1) }
            else { Dispatch.setScalar(enc, scalar, index: 1) }
            Dispatch.setLength(enc, n, lengthBuffer, index: 2)
            enc.setBuffer(out.mtl, offset: out.offset, index: 3)
            Dispatch.dispatch1D(enc, pso, count: words)
        }
        return inheritPending(MetalBooleanArray(length: knownLength, nullCount: _nullCount, validity: validity, values: out, context: ctx))
    }

    /// Element-wise comparison with another array of the same length.
    public func compare(_ op: CompareOp, _ other: MetalArray<T>) throws -> MetalBooleanArray {
        try checkSameLength(other)
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let ctx = context
        let words = BitmapOps.words(bits: n)
        let out = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: n), zeroed: false, context: ctx)
        let isDouble = T.self == Double.self
        let src = isDouble ? KernelSource.compareDouble : KernelSource.compare(T: T.mslType)
        let pso = try Dispatch.pipeline(ctx, family: "cmp", source: src, function: "cmp_array_\(op.rawValue)", type: isDouble ? "double" : T.mslType)
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            enc.setBuffer(other.values.mtl, offset: other.values.offset, index: 1)
            Dispatch.setLength(enc, n, lengthBuffer, index: 2)
            enc.setBuffer(out.mtl, offset: out.offset, index: 3)
            Dispatch.dispatch1D(enc, pso, count: words)
        }
        let validityOut = try BitmapOps.combineValidity(ctx, validity, other.validity, bits: n, lengthBuffer: lengthBuffer)
        let res = inheritPending(MetalBooleanArray(length: knownLength, nullCount: 0, validity: validityOut, values: out, context: ctx))
        res.recomputeNullCount()
        ctx.retainUntilFlush(self)
        return res
    }
}

extension MetalBooleanArray {
    /// Logical AND (nulls propagate; Kleene logic is not applied, matching Arrow's `and` rather than `and_kleene`).
    public func and(_ other: MetalBooleanArray) throws -> MetalBooleanArray {
        try binary("bitmap_and", other)
    }
    public func or(_ other: MetalBooleanArray) throws -> MetalBooleanArray {
        try binary("bitmap_or", other)
    }
    public func not() throws -> MetalBooleanArray {
        let n = dispatchLength
        let words = BitmapOps.words(bits: n)
        let out = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: n), zeroed: false, context: context)
        let pso = try context.pipeline(source: KernelSource.bitmap, function: "bitmap_not", cacheKey: "bitmap/bitmap_not")
        try context.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            Dispatch.setLength(enc, n, lengthBuffer, index: 2)
            enc.setBuffer(out.mtl, offset: out.offset, index: 3)
            Dispatch.dispatch1D(enc, pso, count: words)
        }
        return inheritPending(MetalBooleanArray(length: knownLength, nullCount: _nullCount, validity: validity, values: out, context: context))
    }
    private func binary(_ fn: String, _ other: MetalBooleanArray) throws -> MetalBooleanArray {
        try checkSameLength(other)
        let n = dispatchLength
        let out = try BitmapOps.binary(context, fn, values, other.values, bits: n, lengthBuffer: lengthBuffer)
        let v = try BitmapOps.combineValidity(context, validity, other.validity, bits: n, lengthBuffer: lengthBuffer)
        let res = inheritPending(MetalBooleanArray(length: knownLength, nullCount: 0, validity: v, values: out, context: context))
        res.recomputeNullCount()
        return res
    }
}
