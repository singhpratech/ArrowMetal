import Foundation
import Metal

/// Sort-free grouped `min` / `max`, for every element width.
///
/// See `GroupByExtremaSource` for the shape: an order-preserving 64-bit key per value, the extremes of
/// its high word in one pass, then the extremes of its low word among the rows that hold the winning
/// high word. Two linear passes, 32-bit atomics only, and no argsort of the key column — which is what
/// the segmented path this replaces had to pay for a 64-bit `min`.
///
/// Types four bytes wide or narrower run a single pass, because their whole key fits in the low word.
extension GroupBy {

    /// Both extremes of each key's non-null values, from two linear passes. NaN is skipped, so a key
    /// whose only values are NaN is null, exactly as `minMax` promised on the segmented path.
    ///
    /// Both come out of the same passes, so the last column's pair is kept on the `GroupBy` (weakly
    /// keyed by the column): a `min` and a `max` of one column cost one call.
    func extrema<T: ArrowPrimitive>(_ values: MetalArray<T>) throws -> (min: MetalArray<T>, max: MetalArray<T>) {
        guard values.length == keys.length else { throw ArrowMetalError.lengthMismatch(keys.length, values.length) }
        if let hit = cache.lookup(.extrema, values) as? (min: MetalArray<T>, max: MetalArray<T>) { return hit }
        try Dispatch.checkLength(keys.length)
        let ctx = keys.context
        let n = keys.length
        let kc = keyCount
        let shape = GroupByExtremaSource.shape(T.self)
        let src = GroupByExtremaSource.source(shape, KT: K.mslType)
        let cacheType = "\(shape.name)/\(K.mslType)"
        func pso(_ f: String) throws -> MTLComputePipelineState {
            try Dispatch.pipeline(ctx, family: "groupextrema", source: src, function: f, type: cacheType)
        }
        let table = try MetalArrowBuffer.allocate(byteCount: Swift.max(kc, 1) * 16, zeroed: false, context: ctx)
        let outMin = try MetalArrowBuffer.allocate(byteCount: Swift.max(kc, 1) * T.byteWidth, zeroed: false, context: ctx)
        let outMax = try MetalArrowBuffer.allocate(byteCount: Swift.max(kc, 1) * T.byteWidth, zeroed: false, context: ctx)
        let validBytes = try MetalArrowBuffer.allocate(byteCount: Swift.max(kc, 1), zeroed: false, context: ctx)

        let priv = kc <= GroupByExtremaSource.maxPrivateKeys
        let numTG = Swift.max(1, Swift.min(priv ? 1024 : 4096, (n + 4095) / 4096))
        let chunk = (n + numTG - 1) / numTG
        let flags = (keys.validity == nil ? 0 : 1) | (values.validity == nil ? 0 : 2)
        let tg = MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1)
        let grid = MTLSize(width: numTG, height: 1, depth: 1)
        let initPSO = try pso("gxm_init")
        let packPSO = try pso("gxm_pack")
        let hiPSO = shape.wide ? try pso("gxm_hi_\(priv ? "priv" : "dev")") : nil
        let loPSO = try pso("gxm_lo_\(priv ? "priv" : "dev")")

        func bindRows(_ enc: MTLComputeCommandEncoder) {
            enc.setBuffer(keys.values.mtl, offset: keys.values.offset, index: 0)
            let kv = keys.validity ?? keys.values
            enc.setBuffer(kv.mtl, offset: kv.offset, index: 1)
            enc.setBuffer(values.values.mtl, offset: values.values.offset, index: 2)
            let vv = values.validity ?? values.values
            enc.setBuffer(vv.mtl, offset: vv.offset, index: 3)
            Dispatch.setUInt(enc, n, index: 4)
            Dispatch.setUInt(enc, flags, index: 5)
            Dispatch.setUInt(enc, kc, index: 6)
            Dispatch.setUInt(enc, chunk, index: 7)
            enc.setBuffer(table.mtl, offset: 0, index: 8)
        }

        try ctx.run { enc in
            enc.setComputePipelineState(initPSO)
            enc.setBuffer(table.mtl, offset: 0, index: 0)
            Dispatch.setUInt(enc, kc, index: 1)
            Dispatch.dispatch1D(enc, initPSO, count: kc)
            enc.memoryBarrier(scope: .buffers)
            if n > 0, let hiPSO {
                enc.setComputePipelineState(hiPSO)
                bindRows(enc)
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
                enc.memoryBarrier(scope: .buffers)
            }
            if n > 0 {
                enc.setComputePipelineState(loPSO)
                bindRows(enc)
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
                enc.memoryBarrier(scope: .buffers)
            }
            enc.setComputePipelineState(packPSO)
            enc.setBuffer(table.mtl, offset: 0, index: 0)
            Dispatch.setUInt(enc, kc, index: 1)
            enc.setBuffer(outMin.mtl, offset: 0, index: 2)
            enc.setBuffer(outMax.mtl, offset: 0, index: 3)
            enc.setBuffer(validBytes.mtl, offset: 0, index: 4)
            Dispatch.dispatch1D(enc, packPSO, count: kc)
        }
        ctx.retainUntilFlush(keys); ctx.retainUntilFlush(values); ctx.retainUntilFlush(table)
        let bmMin = try BitmapOps.packBits(ctx, bytes: validBytes, bits: kc)
        let bmMax = try BitmapOps.packBits(ctx, bytes: validBytes, bits: kc)
        try ctx.syncPoint()
        let lo = MetalArray<T>(length: kc, nullCount: 0, validity: bmMin, values: outMin, context: ctx)
        let hi = MetalArray<T>(length: kc, nullCount: 0, validity: bmMax, values: outMax, context: ctx)
        lo.recomputeNullCount(); hi.recomputeNullCount()
        cache.store(.extrema, values, (min: lo, max: hi))
        return (lo, hi)
    }
}
