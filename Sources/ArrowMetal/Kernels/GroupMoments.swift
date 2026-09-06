import Foundation
import Metal

/// Grouped central moments in true binary64 over the counting-sort order — see `GroupMomentsSource`.
extension GroupBy {

    /// Arrow `hash_variance` in binary64: the two-pass shifted algorithm, every addition and
    /// multiplication correctly rounded by the software binary64 routines. `ddof` 0 is the population
    /// variance, 1 the sample one; a group with `ddof` or fewer values is null.
    func varianceDouble<T: ArrowPrimitive>(_ values: MetalArray<T>, ddof: Int) throws -> MetalArray<Double> {
        let (dev, counts, ctx) = try deviations(values, powers: 2)
        let kc = keyCount
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(kc, 1) * 8, zeroed: false, context: ctx)
        let validBytes = try MetalArrowBuffer.allocate(byteCount: Swift.max(kc, 1), context: ctx)
        let spec = GroupMomentsSource.loadExpr(T.self)
        let src = GroupMomentsSource.source(valueType: spec.valueType, load: spec.load)
        let pso = try Dispatch.pipeline(ctx, family: "groupmoments", source: src, function: "gm_variance", type: spec.name)
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(dev.mtl, offset: 0, index: 0)
            enc.setBuffer(counts.mtl, offset: 0, index: 1)
            Dispatch.setUInt(enc, kc, index: 2)
            Dispatch.setUInt(enc, ddof, index: 3)
            enc.setBuffer(out.mtl, offset: 0, index: 4)
            enc.setBuffer(validBytes.mtl, offset: 0, index: 5)
            Dispatch.dispatch1D(enc, pso, count: kc)
        }
        let bm = try BitmapOps.packBits(ctx, bytes: validBytes, bits: kc)
        try ctx.syncPoint()
        let res = MetalArray<Double>(length: kc, nullCount: 0, validity: bm, values: out, context: ctx)
        res.recomputeNullCount()
        return res
    }

    /// The per-group sums of `d^2, d^3, d^4` about the exact per-group mean, plus the counts. The third
    /// and fourth central moments of `skew` / `kurtosis`, in binary64 rather than the Float32 pairs the
    /// sort-based path used.
    func centralMoments<T: ArrowPrimitive>(_ values: MetalArray<T>)
        throws -> (sums: MetalArrowBuffer, counts: MetalArrowBuffer) {
        let (dev, counts, _) = try deviations(values, powers: 4)
        return (dev, counts)
    }

    /// Pass one and pass two: the exact per-group mean, then the deviation sums about it.
    /// `powers` is 2 for `(sum d, sum d^2)` and 4 for `(sum d^2, sum d^3, sum d^4)`.
    private func deviations<T: ArrowPrimitive>(_ values: MetalArray<T>, powers: Int)
        throws -> (MetalArrowBuffer, MetalArrowBuffer, MetalContext) {
        guard values.length == keys.length else { throw ArrowMetalError.lengthMismatch(keys.length, values.length) }
        let ctx = values.context
        let seg = try segments()
        let kc = keyCount
        let spec = GroupMomentsSource.loadExpr(T.self)
        let src = GroupMomentsSource.source(valueType: spec.valueType, load: spec.load)
        func pso(_ f: String) throws -> MTLComputePipelineState {
            try Dispatch.pipeline(ctx, family: "groupmoments", source: src, function: f, type: spec.name)
        }
        // A whole threadgroup per group is right while groups are large; at ten million groups of five
        // rows it is a quarter of a billion threads doing nothing, so one thread per group instead.
        let narrow = keys.length < kc * 32
        let sums = try MetalArrowBuffer.allocate(byteCount: Swift.max(kc, 1) * 8, zeroed: false, context: ctx)
        let counts = try MetalArrowBuffer.allocate(byteCount: Swift.max(kc, 1) * 4, context: ctx)
        let means = try MetalArrowBuffer.allocate(byteCount: Swift.max(kc, 1) * 8, zeroed: false, context: ctx)
        let dev = try MetalArrowBuffer.allocate(byteCount: Swift.max(kc, 1) * 8 * (powers == 4 ? 3 : 2),
                                                zeroed: false, context: ctx)
        let sumPSO = try pso(narrow ? "gm_sum_narrow" : "gm_sum_wide")
        let meanPSO = try pso("gm_mean")
        let devPSO = try pso((powers == 4 ? "gm_dev4_" : "gm_dev_") + (narrow ? "narrow" : "wide"))
        let tg = MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1)
        try ctx.run { enc in
            enc.setComputePipelineState(sumPSO)
            ExtraAggregates.bindSegments(enc, seg, values)
            enc.setBuffer(sums.mtl, offset: 0, index: 7)
            enc.setBuffer(counts.mtl, offset: 0, index: 8)
            if narrow { Dispatch.dispatch1D(enc, sumPSO, count: kc) }
            else { enc.dispatchThreadgroups(MTLSize(width: Swift.max(kc, 1), height: 1, depth: 1), threadsPerThreadgroup: tg) }
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(meanPSO)
            enc.setBuffer(sums.mtl, offset: 0, index: 0)
            enc.setBuffer(counts.mtl, offset: 0, index: 1)
            Dispatch.setUInt(enc, kc, index: 2)
            enc.setBuffer(means.mtl, offset: 0, index: 3)
            Dispatch.dispatch1D(enc, meanPSO, count: kc)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(devPSO)
            ExtraAggregates.bindSegments(enc, seg, values)
            enc.setBuffer(means.mtl, offset: 0, index: 7)
            enc.setBuffer(dev.mtl, offset: 0, index: 8)
            if narrow { Dispatch.dispatch1D(enc, devPSO, count: kc) }
            else { enc.dispatchThreadgroups(MTLSize(width: Swift.max(kc, 1), height: 1, depth: 1), threadsPerThreadgroup: tg) }
        }
        ctx.retainUntilFlush(seg.ord); ctx.retainUntilFlush(values)
        ctx.retainUntilFlush(sums); ctx.retainUntilFlush(means)
        return (dev, counts, ctx)
    }
}
