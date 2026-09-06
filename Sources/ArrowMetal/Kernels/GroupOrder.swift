import Foundation
import Metal

/// The counting sort by group id behind `GroupBy.segments()`.
///
/// See `GroupOrderSource` for the two scatters and why there are two. This file picks between them and
/// falls back to the argsort when neither fits: the chunked scatter needs a per-block histogram that
/// stays inside a memory budget, and the atomic scatter needs the longest run to be short enough for
/// the order-restoring pass that follows it.
extension GroupBy {

    /// Rows in group order, with `[start, end)` per group. Counting sort, not a radix sort.
    /// Returns nil when neither counting-sort shape applies, so the caller can fall back.
    ///
    /// Which scatter: the chunked one is exactly stable and needs no read-back, but its per-block
    /// histogram grows with the group count, and its per-row cost includes a sweep of the threadgroup
    /// to rank a row among the rows of its group in the same sub-chunk. The atomic one is one atomic
    /// bump per row plus a sort of each run, which is the cheaper pair once runs are short. So: short
    /// runs and more groups than a private table holds take the atomic path, everything else the
    /// chunked one, and a run too long for either falls through to the argsort.
    func countingSortOrder() throws -> GroupSegments? {
        try Dispatch.checkLength(keys.length)
        let ctx = keys.context
        let n = keys.length
        let kc = keyCount
        guard n > 0 else { return nil }
        let src = GroupOrderSource.source(KT: K.mslType)
        func pso(_ f: String) throws -> MTLComputePipelineState {
            try Dispatch.pipeline(ctx, family: "grouporder", source: src, function: f, type: K.mslType)
        }
        let blocks0 = Swift.max(1, Swift.min(1024, (n + 4095) / 4096))
        // The chunked path partitions the rows one **simdgroup** at a time, so its block count is a
        // multiple of the simdgroups per threadgroup and its histogram has one row per simdgroup.
        let sub = Dispatch.threadgroupSize / 32
        let wanted = Swift.max(sub, Swift.min(2048, (n + 8191) / 8192))
        var logical = Swift.min(wanted, Swift.max(sub, GroupOrderSource.chunkedBudget / kc))
        logical = Swift.min(logical, Swift.max(512, (1 << 20) / kc))
        logical = Swift.max(sub, (logical / sub) * sub)
        let physical = logical / sub
        let chunkedFits = logical == wanted || physical >= 8
        // Once runs are short the atomic scatter plus a per-run sort costs less than the chunked
        // scatter's per-row threadgroup sweep, and it does not allocate a per-block histogram at all.
        let atomicFirst = kc > GroupByExtremaSource.maxPrivateKeys && n < kc * 1024
        let flags = keys.validity == nil ? 0 : 1
        let tg = MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1)

        let total = try MetalArrowBuffer.allocate(byteCount: kc * 4, zeroed: false, context: ctx)
        let segStart = try MetalArrowBuffer.allocate(byteCount: kc * 4, zeroed: false, context: ctx)
        let segEnd = try MetalArrowBuffer.allocate(byteCount: kc * 4, zeroed: false, context: ctx)
        let maxRun = try MetalArrowBuffer.allocate(byteCount: 4, context: ctx)
        let ord = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: false, context: ctx)
        let zeroPSO = try pso("cs_zero"), scanPSO = try pso("cs_scan")

        func bindRows(_ enc: MTLComputeCommandEncoder, _ groupBlocks: Int) {
            enc.setBuffer(keys.values.mtl, offset: keys.values.offset, index: 0)
            let kv = keys.validity ?? keys.values
            enc.setBuffer(kv.mtl, offset: kv.offset, index: 1)
            Dispatch.setUInt(enc, n, index: 2)
            Dispatch.setUInt(enc, flags, index: 3)
            Dispatch.setUInt(enc, kc, index: 4)
            Dispatch.setUInt(enc, (n + groupBlocks - 1) / groupBlocks, index: 5)
        }
        func zero(_ enc: MTLComputeCommandEncoder, _ buf: MetalArrowBuffer, _ count: Int) {
            enc.setComputePipelineState(zeroPSO)
            enc.setBuffer(buf.mtl, offset: 0, index: 0)
            Dispatch.setUInt(enc, count, index: 1)
            Dispatch.dispatch1D(enc, zeroPSO, count: count)
        }
        func scan(_ enc: MTLComputeCommandEncoder) {
            enc.setComputePipelineState(scanPSO)
            enc.setBuffer(total.mtl, offset: 0, index: 0)
            Dispatch.setUInt(enc, kc, index: 1)
            enc.setBuffer(segStart.mtl, offset: 0, index: 2)
            enc.setBuffer(segEnd.mtl, offset: 0, index: 3)
            enc.setBuffer(maxRun.mtl, offset: 0, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
        }

        func runChunked() throws -> GroupSegments {
            let histPSO = try pso("cs_hist_blocks")
            let totalsPSO = try pso("cs_totals"), offPSO = try pso("cs_block_offsets")
            let scatPSO = try pso("cs_scatter")
            let hist = try MetalArrowBuffer.allocate(byteCount: logical * kc * 4, zeroed: false, context: ctx)
            let physGrid = MTLSize(width: physical, height: 1, depth: 1)
            try ctx.run { enc in
                zero(enc, hist, logical * kc)
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(histPSO)
                bindRows(enc, logical)
                enc.setBuffer(hist.mtl, offset: 0, index: 6)
                enc.dispatchThreadgroups(physGrid, threadsPerThreadgroup: tg)
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(totalsPSO)
                enc.setBuffer(hist.mtl, offset: 0, index: 0)
                Dispatch.setUInt(enc, kc, index: 1)
                Dispatch.setUInt(enc, logical, index: 2)
                enc.setBuffer(total.mtl, offset: 0, index: 3)
                Dispatch.dispatch1D(enc, totalsPSO, count: kc)
                enc.memoryBarrier(scope: .buffers)
                zero(enc, maxRun, 1)
                enc.memoryBarrier(scope: .buffers)
                scan(enc)
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(offPSO)
                enc.setBuffer(hist.mtl, offset: 0, index: 0)
                enc.setBuffer(segStart.mtl, offset: 0, index: 1)
                Dispatch.setUInt(enc, kc, index: 2)
                Dispatch.setUInt(enc, logical, index: 3)
                Dispatch.dispatch1D(enc, offPSO, count: kc)
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(scatPSO)
                bindRows(enc, logical)
                enc.setBuffer(hist.mtl, offset: 0, index: 6)
                enc.setBuffer(ord.mtl, offset: 0, index: 7)
                enc.dispatchThreadgroups(physGrid, threadsPerThreadgroup: tg)
            }
            ctx.retainUntilFlush(keys); ctx.retainUntilFlush(hist)
            try ctx.syncPoint()
            let ordArray = MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: ord, context: ctx)
            return GroupSegments(ord: ordArray, segStart: segStart, segEnd: segEnd, keyCount: kc, rows: n)
        }

        // The atomic scatter: count, scan, and look at the longest run before committing to it.
        func runAtomic() throws -> GroupSegments? {
        let histPSO = try pso("cs_hist_dev")
        try ctx.run { enc in
            zero(enc, total, kc)
            zero(enc, maxRun, 1)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(histPSO)
            bindRows(enc, blocks0)
            enc.setBuffer(total.mtl, offset: 0, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: blocks0, height: 1, depth: 1), threadsPerThreadgroup: tg)
            enc.memoryBarrier(scope: .buffers)
            scan(enc)
        }
        ctx.retainUntilFlush(keys)
        try ctx.syncPoint()
        let longest = withExtendedLifetime(maxRun) { Int(maxRun.typed(UInt32.self)[0]) }
        guard longest <= GroupOrderSource.maxGroupFix else { return nil }

        let cursor = try MetalArrowBuffer.allocate(byteCount: kc * 4, zeroed: false, context: ctx)
        let copyPSO = try pso("cs_copy"), scatPSO = try pso("cs_scatter_atomic")
        let fixPSO = try pso(longest <= GroupOrderSource.maxThreadFix ? "cs_fix" : "cs_fix_tg")
        let perThreadFix = longest <= GroupOrderSource.maxThreadFix
        try ctx.run { enc in
            enc.setComputePipelineState(copyPSO)
            enc.setBuffer(segStart.mtl, offset: 0, index: 0)
            enc.setBuffer(cursor.mtl, offset: 0, index: 1)
            Dispatch.setUInt(enc, kc, index: 2)
            Dispatch.dispatch1D(enc, copyPSO, count: kc)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(scatPSO)
            bindRows(enc, blocks0)
            enc.setBuffer(cursor.mtl, offset: 0, index: 6)
            enc.setBuffer(ord.mtl, offset: 0, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: blocks0, height: 1, depth: 1), threadsPerThreadgroup: tg)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(fixPSO)
            enc.setBuffer(segStart.mtl, offset: 0, index: 0)
            enc.setBuffer(segEnd.mtl, offset: 0, index: 1)
            enc.setBuffer(ord.mtl, offset: 0, index: 2)
            Dispatch.setUInt(enc, kc, index: 3)
            if perThreadFix {
                Dispatch.dispatch1D(enc, fixPSO, count: kc)
            } else {
                enc.dispatchThreadgroups(MTLSize(width: kc, height: 1, depth: 1), threadsPerThreadgroup: tg)
            }
        }
        ctx.retainUntilFlush(keys); ctx.retainUntilFlush(cursor)
        try ctx.syncPoint()
        let ordArray = MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: ord, context: ctx)
        return GroupSegments(ord: ordArray, segStart: segStart, segEnd: segEnd, keyCount: kc, rows: n)
        }

        if atomicFirst, let s = try runAtomic() { return s }
        if chunkedFits { return try runChunked() }
        return atomicFirst ? nil : try runAtomic()
    }
}
