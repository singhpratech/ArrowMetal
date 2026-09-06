import Foundation
import Metal

/// Index types accepted by `take`.
public protocol ArrowIndex: ArrowPrimitive {}
extension Int32: ArrowIndex {}
extension Int64: ArrowIndex {}
extension UInt32: ArrowIndex {}

extension MetalArray {
    /// Arrow `take`: gathers `indices` from this array. A null index yields a null output element.
    /// Throws `invalidArrowArray` if any index is out of range (checked on the GPU, reported after the dispatch).
    public func take<I: ArrowIndex>(_ indices: MetalArray<I>) throws -> MetalArray<T> {
        try Dispatch.checkLength(length)
        try Dispatch.checkLength(indices.length)
        let ctx = context
        let n = indices.length
        let mslT = Dispatch.moveType(T.self)
        let src = KernelSource.take(T: mslT, I: I.mslType)
        let pso = try Dispatch.pipeline(ctx, family: "take", source: src, function: "take_kernel", type: "\(mslT)/\(I.mslType)")
        let out = try MetalArrowBuffer.allocate(byteCount: n * T.byteWidth, zeroed: false, context: ctx)
        let hasV = validity != nil, hasIV = indices.validity != nil
        let validBytes = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1), zeroed: false, context: ctx)
        let errorFlag = try MetalArrowBuffer.allocate(byteCount: 4, context: ctx)
        if n > 0 {
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                enc.setBuffer((validity ?? values).mtl, offset: (validity ?? values).offset, index: 1)
                enc.setBuffer(indices.values.mtl, offset: indices.values.offset, index: 2)
                enc.setBuffer((indices.validity ?? indices.values).mtl, offset: (indices.validity ?? indices.values).offset, index: 3)
                Dispatch.setUInt(enc, n, index: 4)
                Dispatch.setUInt(enc, length, index: 5)
                Dispatch.setUInt(enc, (hasV ? 1 : 0) | (hasIV ? 2 : 0), index: 6)
                enc.setBuffer(out.mtl, offset: out.offset, index: 7)
                enc.setBuffer(validBytes.mtl, offset: validBytes.offset, index: 8)
                enc.setBuffer(errorFlag.mtl, offset: errorFlag.offset, index: 9)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        let srcLen = length
        try ctx.afterFlush { [errorFlag] in
            if errorFlag.typed(UInt32.self)[0] != 0 { throw ArrowMetalError.invalidArrowArray("take: index out of range (array length \(srcLen))") }
        }
        ctx.retainUntilFlush(errorFlag); ctx.retainUntilFlush(self); ctx.retainUntilFlush(indices); ctx.retainUntilFlush(validBytes)
        var outValidity: MetalArrowBuffer? = nil
        if (hasV || hasIV) && n > 0 { outValidity = try BitmapOps.packBits(ctx, bytes: validBytes, bits: n) }
        let res = MetalArray<T>(length: n, nullCount: 0, validity: outValidity, values: out, context: ctx)
        res.recomputeNullCount()
        return res
    }
}

extension MetalBooleanArray {
    public func take<I: ArrowIndex>(_ indices: MetalArray<I>) throws -> MetalBooleanArray {
        try MetalBooleanArray.fromUInt8Array(try toUInt8Array().take(indices))
    }
}
