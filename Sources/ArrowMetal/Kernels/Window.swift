import Foundation
import Metal

/// SQL window functions, shifts, pairwise differences and trailing rolling windows, all on the GPU.
///
/// **Ranking.** `rowNumber`, `rank`, `denseRank`, `percentRank` and `cumeDist` all come out of one
/// argsort plus a scan. The array is sorted once (the existing stable radix argsort, nulls last), run
/// boundaries are marked in the sorted order, the marks are scanned into a dense rank, each run's first
/// and last sorted positions are recorded, and the answer is scattered back to the row it came from —
/// so the result is aligned to the *original* rows, not to the sorted ones.
///
/// Nulls follow SQL `ORDER BY x NULLS LAST`: they sort after every value and form a single tie group,
/// so `rowNumber` numbers them last in their original order, `rank` and `denseRank` give all of them
/// one rank, and no ranking result is ever itself null. Floating point ties use Arrow value equality:
/// every NaN is one value (ordered after +inf) and `-0.0` equals `0.0`.
///
/// **Shifts, pairwise and cumulative.** `shift`, `pairwiseDiff`, `cumulativeProd` and `cumulativeMean`.
/// The first two are one thread per element; `cumulativeProd` is the two-level scan from
/// `CumulativeSource` with a multiply, and `cumulativeMean` is a binary64 running sum over a running
/// count of non-null rows.
///
/// **Rolling windows.** `rollingSum`, `rollingMin`, `rollingMax` and `rollingMean` over a trailing
/// window of `window` rows ending at each output, with `minPeriods` non-null rows required before a
/// value is produced (fewer gives a null). Min and max are one thread per output scanning the window,
/// which is the right shape up to a few thousand rows per window; sum and mean are O(n), the difference
/// of two prefix sums.
extension MetalArray {

    // MARK: - Ranking

    /// SQL `ROW_NUMBER() OVER (ORDER BY value NULLS LAST)`: 1-based position of each row in the sorted
    /// order, returned aligned to the original rows. Ties keep the input order (the argsort is stable).
    /// The result never contains nulls.
    public func rowNumber() throws -> MetalArray<Int32> { try scatterIntRank(mode: 0) }

    /// SQL `RANK()`: the 1-based position of the *first* row of each tie group, so equal values share a
    /// rank and the following rank skips the gap. Nulls are one tie group at the end.
    public func rank() throws -> MetalArray<Int32> { try scatterIntRank(mode: 1) }

    /// SQL `DENSE_RANK()`: 1-based index of each distinct value in ascending order, with no gaps.
    /// Nulls, being one tie group at the end, take the last index.
    public func denseRank() throws -> MetalArray<Int32> { try scatterIntRank(mode: 2) }

    /// SQL `PERCENT_RANK()`: `(rank - 1) / (n - 1)` as float64, 0 for a single-row column.
    /// Computed with the correctly rounded software binary64 divide, so it matches a host `Double`
    /// division bit for bit.
    public func percentRank() throws -> MetalArray<Double> { try scatterDoubleRank(mode: 0) }

    /// SQL `CUME_DIST()`: the fraction of rows at or before this row's value in the order, as float64.
    /// Equal values share the value, and the largest one (or the nulls, when there are any) gets 1.
    public func cumeDist() throws -> MetalArray<Double> { try scatterDoubleRank(mode: 1) }

    /// Sorted order plus the run marks, scan and run bounds every ranking function shares.
    private func windowOrder() throws -> WindowOrder? {
        let ctx = context
        let n = length
        guard n > 0 else { return nil }
        try Dispatch.checkLength(n)
        let m = n - nullCount                    // the non-null values occupy sorted positions [0, m)
        let uType = UniqueSource.unsignedType(width: T.byteWidth)
        let src = WindowSource.ranking(U: uType)
        func pso(_ f: String) throws -> MTLComputePipelineState {
            try Dispatch.pipeline(ctx, family: "window", source: src, function: f, type: uType)
        }

        // Floats compare as raw bit patterns, so normalise first: one NaN pattern, -0 becomes +0.
        var keyValues = values
        var keyArray: MetalArray<T> = self
        if T.isFloatingPoint {
            let norm = try MetalArrowBuffer.allocate(byteCount: n * T.byteWidth, zeroed: false, context: ctx)
            let normPSO = try pso(T.byteWidth == 8 ? "win_norm_f64" : "win_norm_f32")
            try ctx.run { enc in
                enc.setComputePipelineState(normPSO)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                Dispatch.setLength(enc, n, nil, index: 1)
                enc.setBuffer(norm.mtl, offset: norm.offset, index: 2)
                Dispatch.dispatch1D(enc, normPSO, count: n)
            }
            ctx.retainUntilFlush(self)
            keyValues = norm
            keyArray = MetalArray<T>(length: n, nullCount: nullCount, validity: validity, values: norm, context: ctx)
        }

        let ord = try keyArray.argsort()          // full order, nulls last
        let marks = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: false, context: ctx)
        let ranks = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: false, context: ctx)
        let startPos = try MetalArrowBuffer.allocate(byteCount: n * 4, context: ctx)
        let endPos = try MetalArrowBuffer.allocate(byteCount: n * 4, context: ctx)
        let blocks = (n + Dispatch.threadgroupSize - 1) / Dispatch.threadgroupSize
        let blockTotals = try MetalArrowBuffer.allocate(byteCount: blocks * 4, zeroed: false, context: ctx)
        let marksPSO = try pso("win_marks"), blockPSO = try pso("win_scan_block")
        let totalsPSO = try pso("win_scan_totals"), addPSO = try pso("win_scan_add"), boundsPSO = try pso("win_run_bounds")
        let tg = MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1)
        let grid = MTLSize(width: blocks, height: 1, depth: 1)
        try ctx.run { enc in
            enc.setComputePipelineState(marksPSO)
            enc.setBuffer(keyValues.mtl, offset: keyValues.offset, index: 0)
            enc.setBuffer(ord.values.mtl, offset: ord.values.offset, index: 1)
            Dispatch.setLength(enc, n, nil, index: 2)
            Dispatch.setUInt(enc, m, index: 3)
            enc.setBuffer(marks.mtl, offset: marks.offset, index: 4)
            Dispatch.dispatch1D(enc, marksPSO, count: n)
            enc.memoryBarrier(scope: .buffers)

            enc.setComputePipelineState(blockPSO)
            enc.setBuffer(marks.mtl, offset: marks.offset, index: 0)
            Dispatch.setLength(enc, n, nil, index: 1)
            enc.setBuffer(ranks.mtl, offset: ranks.offset, index: 2)
            enc.setBuffer(blockTotals.mtl, offset: blockTotals.offset, index: 3)
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
            enc.memoryBarrier(scope: .buffers)

            enc.setComputePipelineState(totalsPSO)
            enc.setBuffer(blockTotals.mtl, offset: blockTotals.offset, index: 0)
            Dispatch.setUInt(enc, blocks, index: 1)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
            enc.memoryBarrier(scope: .buffers)

            enc.setComputePipelineState(addPSO)
            enc.setBuffer(ranks.mtl, offset: ranks.offset, index: 0)
            enc.setBuffer(blockTotals.mtl, offset: blockTotals.offset, index: 1)
            Dispatch.setLength(enc, n, nil, index: 2)
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
            enc.memoryBarrier(scope: .buffers)

            enc.setComputePipelineState(boundsPSO)
            enc.setBuffer(marks.mtl, offset: marks.offset, index: 0)
            enc.setBuffer(ranks.mtl, offset: ranks.offset, index: 1)
            Dispatch.setLength(enc, n, nil, index: 2)
            enc.setBuffer(startPos.mtl, offset: startPos.offset, index: 3)
            enc.setBuffer(endPos.mtl, offset: endPos.offset, index: 4)
            Dispatch.dispatch1D(enc, boundsPSO, count: n)
        }
        for o in [keyValues, marks, ranks, startPos, endPos, blockTotals] as [AnyObject] { ctx.retainUntilFlush(o) }
        ctx.retainUntilFlush(ord)
        return WindowOrder(ord: ord, marks: marks, ranks: ranks, startPos: startPos, endPos: endPos, n: n, source: src, unsignedType: uType)
    }

    private func scatterIntRank(mode: Int) throws -> MetalArray<Int32> {
        let ctx = context
        guard let o = try windowOrder() else { return try MetalArray<Int32>([Int32](), context: ctx) }
        let out = try MetalArrowBuffer.allocate(byteCount: o.n * 4, zeroed: false, context: ctx)
        let p = try Dispatch.pipeline(ctx, family: "window", source: o.source, function: "win_scatter_int", type: o.unsignedType)
        try ctx.run { enc in
            enc.setComputePipelineState(p)
            enc.setBuffer(o.ord.values.mtl, offset: o.ord.values.offset, index: 0)
            enc.setBuffer(o.ranks.mtl, offset: o.ranks.offset, index: 1)
            enc.setBuffer(o.marks.mtl, offset: o.marks.offset, index: 2)
            enc.setBuffer(o.startPos.mtl, offset: o.startPos.offset, index: 3)
            Dispatch.setLength(enc, o.n, nil, index: 4)
            Dispatch.setUInt(enc, mode, index: 5)
            enc.setBuffer(out.mtl, offset: out.offset, index: 6)
            Dispatch.dispatch1D(enc, p, count: o.n)
        }
        ctx.retainUntilFlush(o)
        return MetalArray<Int32>(length: o.n, nullCount: 0, validity: nil, values: out, context: ctx)
    }

    private func scatterDoubleRank(mode: Int) throws -> MetalArray<Double> {
        let ctx = context
        guard let o = try windowOrder() else { return try MetalArray<Double>([Double](), context: ctx) }
        let out = try MetalArrowBuffer.allocate(byteCount: o.n * 8, zeroed: false, context: ctx)
        let p = try Dispatch.pipeline(ctx, family: "window", source: o.source, function: "win_scatter_double", type: o.unsignedType)
        try ctx.run { enc in
            enc.setComputePipelineState(p)
            enc.setBuffer(o.ord.values.mtl, offset: o.ord.values.offset, index: 0)
            enc.setBuffer(o.ranks.mtl, offset: o.ranks.offset, index: 1)
            enc.setBuffer(o.marks.mtl, offset: o.marks.offset, index: 2)
            enc.setBuffer(o.startPos.mtl, offset: o.startPos.offset, index: 3)
            enc.setBuffer(o.endPos.mtl, offset: o.endPos.offset, index: 4)
            Dispatch.setLength(enc, o.n, nil, index: 5)
            Dispatch.setUInt(enc, mode, index: 6)
            enc.setBuffer(out.mtl, offset: out.offset, index: 7)
            Dispatch.dispatch1D(enc, p, count: o.n)
        }
        ctx.retainUntilFlush(o)
        return MetalArray<Double>(length: o.n, nullCount: 0, validity: nil, values: out, context: ctx)
    }

    // MARK: - Shift and pairwise

    /// Lag (positive `by`) or lead (negative `by`): `out[i] = self[i - by]`.
    ///
    /// Rows that would read outside the array take `fill`, or become null when `fill` is nil. A null
    /// input row shifts in as a null. `by == 0` is the identity.
    public func shift(by: Int, fill: T? = nil) throws -> MetalArray<T> {
        let ctx = context
        let n = length
        try Dispatch.checkLength(n)
        let (out, validBytes) = try Self.windowOutputs(ctx, n: n, byteWidth: T.byteWidth)
        guard n > 0 else { return MetalArray<T>(length: 0, nullCount: 0, validity: nil, values: out, context: ctx) }
        let p = try valuePipeline("win_shift")
        let hasV = validity != nil
        try ctx.run { enc in
            enc.setComputePipelineState(p)
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            let v = validity ?? values
            enc.setBuffer(v.mtl, offset: v.offset, index: 1)
            Dispatch.setLength(enc, n, nil, index: 2)
            Dispatch.setScalar(enc, Int32(clamping: by), index: 3)
            Dispatch.setUInt(enc, hasV ? 1 : 0, index: 4)
            Dispatch.setUInt(enc, fill != nil ? 1 : 0, index: 5)
            Structural.setMoveScalar(enc, fill ?? .zero, index: 6)
            enc.setBuffer(out.mtl, offset: out.offset, index: 7)
            enc.setBuffer(validBytes.mtl, offset: validBytes.offset, index: 8)
            Dispatch.dispatch1D(enc, p, count: n)
        }
        return try Self.assemble(ctx, self, out: out, validBytes: validBytes, n: n)
    }

    /// Arrow `pairwise_diff`: `out[i] = self[i] - self[i - period]`, null where either side is null or
    /// falls outside the array. A negative `period` differences forwards.
    ///
    /// Integers wrap, like the rest of the unchecked arithmetic here; float32 subtracts in `float` and
    /// float64 through the correctly rounded software binary64 subtract, so both are exact.
    public func pairwiseDiff(period: Int = 1) throws -> MetalArray<T> {
        let ctx = context
        let n = length
        try Dispatch.checkLength(n)
        let (out, validBytes) = try Self.windowOutputs(ctx, n: n, byteWidth: T.byteWidth)
        guard n > 0 else { return MetalArray<T>(length: 0, nullCount: 0, validity: nil, values: out, context: ctx) }
        let p = try valuePipeline("win_pairwise_diff")
        let hasV = validity != nil
        try ctx.run { enc in
            enc.setComputePipelineState(p)
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            let v = validity ?? values
            enc.setBuffer(v.mtl, offset: v.offset, index: 1)
            Dispatch.setLength(enc, n, nil, index: 2)
            Dispatch.setScalar(enc, Int32(clamping: period), index: 3)
            Dispatch.setUInt(enc, hasV ? 1 : 0, index: 4)
            enc.setBuffer(out.mtl, offset: out.offset, index: 5)
            enc.setBuffer(validBytes.mtl, offset: validBytes.offset, index: 6)
            Dispatch.dispatch1D(enc, p, count: n)
        }
        return try Self.assemble(ctx, self, out: out, validBytes: validBytes, n: n)
    }

    // MARK: - Cumulative product and mean

    /// Arrow `cumulative_prod`: the running product, null exactly where the input is, the running value
    /// carrying across nulls unchanged (Arrow's `skip_nulls` behaviour, as `cumulativeSum` has it).
    ///
    /// The two-level scan from `CumulativeSource` with a multiply. Integer products wrap and are exact;
    /// float32 and float64 reassociate, so the last ulp can differ from a strictly sequential product,
    /// and a float32 product that drifts into the subnormals comes back as zero (Apple GPUs flush
    /// float32 denormals; the float64 path is the software multiplier and keeps them).
    public func cumulativeProd() throws -> MetalArray<T> {
        let n = length
        try Dispatch.checkLength(n)
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * T.byteWidth, zeroed: false, context: ctx)
        guard n > 0 else { return MetalArray<T>(length: 0, nullCount: 0, validity: nil, values: out, context: ctx) }
        let blocks = (n + Dispatch.threadgroupSize - 1) / Dispatch.threadgroupSize
        let totals = try MetalArrowBuffer.allocate(byteCount: blocks * T.byteWidth, zeroed: false, context: ctx)
        let (src, cacheType) = Self.prodSource
        func pso(_ f: String) throws -> MTLComputePipelineState {
            try Dispatch.pipeline(ctx, family: "window-prod", source: src, function: f, type: cacheType)
        }
        let blockPSO = try pso("cum_block_prod"), totalsPSO = try pso("cum_totals_prod"), addPSO = try pso("cum_add_prod")
        let tg = MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1)
        let grid = MTLSize(width: blocks, height: 1, depth: 1)
        let hasV = validity != nil
        try ctx.run { enc in
            enc.setComputePipelineState(blockPSO)
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            let v = validity ?? values
            enc.setBuffer(v.mtl, offset: v.offset, index: 1)
            Dispatch.setLength(enc, n, nil, index: 2)
            Dispatch.setUInt(enc, hasV ? 1 : 0, index: 3)
            enc.setBuffer(out.mtl, offset: out.offset, index: 4)
            enc.setBuffer(totals.mtl, offset: totals.offset, index: 5)
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
            enc.memoryBarrier(scope: .buffers)

            enc.setComputePipelineState(totalsPSO)
            enc.setBuffer(totals.mtl, offset: totals.offset, index: 0)
            Dispatch.setUInt(enc, blocks, index: 1)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
            enc.memoryBarrier(scope: .buffers)

            enc.setComputePipelineState(addPSO)
            enc.setBuffer(out.mtl, offset: out.offset, index: 0)
            enc.setBuffer(totals.mtl, offset: totals.offset, index: 1)
            Dispatch.setLength(enc, n, nil, index: 2)
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        }
        ctx.retainUntilFlush(self)
        ctx.retainUntilFlush(totals)
        return MetalArray<T>(length: n, nullCount: nullCount, validity: validity, values: out, context: ctx)
    }

    /// Arrow `cumulative_mean`: the running mean of the non-null values seen so far, as float64.
    ///
    /// Output is null exactly where the input is. The running sum is a binary64 two-level scan and the
    /// running count an int32 one, so integer inputs above 2^53 round on the way in and float sums
    /// reassociate; the final divide is correctly rounded.
    public func cumulativeMean() throws -> MetalArray<Double> {
        let ctx = context
        let n = length
        try Dispatch.checkLength(n)
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * 8, zeroed: false, context: ctx)
        guard n > 0 else { return MetalArray<Double>(length: 0, nullCount: 0, validity: nil, values: out, context: ctx) }
        let sums = try widenedToFloat64().cumulative(.sum)
        let counts = try validCountPrefix(n: n)
        let p = try Dispatch.pipeline(ctx, family: "window", source: WindowSource.doubleOps, function: "win_div_count", type: "f64")
        try ctx.run { enc in
            enc.setComputePipelineState(p)
            enc.setBuffer(sums.values.mtl, offset: sums.values.offset, index: 0)
            enc.setBuffer(counts.values.mtl, offset: counts.values.offset, index: 1)
            Dispatch.setLength(enc, n, nil, index: 2)
            enc.setBuffer(out.mtl, offset: out.offset, index: 3)
            Dispatch.dispatch1D(enc, p, count: n)
        }
        ctx.retainUntilFlush(sums); ctx.retainUntilFlush(counts)
        return MetalArray<Double>(length: n, nullCount: nullCount, validity: validity, values: out, context: ctx)
    }

    // MARK: - Rolling windows

    /// Trailing rolling sum over `window` rows ending at each output, null until `minPeriods` non-null
    /// rows are in the window (`minPeriods` defaults to `window`).
    ///
    /// O(n): the difference of two inclusive prefix sums. Integers wrap, so the difference is exact;
    /// float32 and float64 lose the usual cancellation digits when the window sum is far smaller than
    /// the running total, which is the price of the prefix trick. For the same reason one NaN or
    /// infinity in a float column poisons every later window, not just the ones containing it — use
    /// `rollingMin`/`rollingMax`, which scan the window itself, when that matters.
    public func rollingSum(window: Int, minPeriods: Int? = nil) throws -> MetalArray<T> {
        let (w, mp) = try Self.rollingParams(window, minPeriods)
        let ctx = context
        let n = length
        try Dispatch.checkLength(n)
        let (out, validBytes) = try Self.windowOutputs(ctx, n: n, byteWidth: T.byteWidth)
        guard n > 0 else { return MetalArray<T>(length: 0, nullCount: 0, validity: nil, values: out, context: ctx) }
        let pre = try cumulative(.sum)
        let counts = try validCountPrefix(n: n)
        let p = try valuePipeline("win_rolling_sum")
        try ctx.run { enc in
            enc.setComputePipelineState(p)
            enc.setBuffer(pre.values.mtl, offset: pre.values.offset, index: 0)
            enc.setBuffer(counts.values.mtl, offset: counts.values.offset, index: 1)
            Dispatch.setLength(enc, n, nil, index: 2)
            Dispatch.setUInt(enc, w, index: 3)
            Dispatch.setUInt(enc, mp, index: 4)
            enc.setBuffer(out.mtl, offset: out.offset, index: 5)
            enc.setBuffer(validBytes.mtl, offset: validBytes.offset, index: 6)
            Dispatch.dispatch1D(enc, p, count: n)
        }
        ctx.retainUntilFlush(pre); ctx.retainUntilFlush(counts)
        return try Self.assemble(ctx, self, out: out, validBytes: validBytes, n: n)
    }

    /// Trailing rolling minimum. One thread per output scanning the `window` inputs that end at it, so
    /// the work is O(n · window) — the right shape up to a few thousand rows per window. NaN is skipped,
    /// as it is by the min/max reductions; a window whose non-null values are all NaN yields `+inf`.
    public func rollingMin(window: Int, minPeriods: Int? = nil) throws -> MetalArray<T> {
        try rollingExtremum("win_rolling_min", window, minPeriods)
    }

    /// Trailing rolling maximum, mirroring `rollingMin` (an all-NaN window yields `-inf`).
    public func rollingMax(window: Int, minPeriods: Int? = nil) throws -> MetalArray<T> {
        try rollingExtremum("win_rolling_max", window, minPeriods)
    }

    private func rollingExtremum(_ function: String, _ window: Int, _ minPeriods: Int?) throws -> MetalArray<T> {
        let (w, mp) = try Self.rollingParams(window, minPeriods)
        let ctx = context
        let n = length
        try Dispatch.checkLength(n)
        let (out, validBytes) = try Self.windowOutputs(ctx, n: n, byteWidth: T.byteWidth)
        guard n > 0 else { return MetalArray<T>(length: 0, nullCount: 0, validity: nil, values: out, context: ctx) }
        let p = try valuePipeline(function)
        let hasV = validity != nil
        try ctx.run { enc in
            enc.setComputePipelineState(p)
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            let v = validity ?? values
            enc.setBuffer(v.mtl, offset: v.offset, index: 1)
            Dispatch.setLength(enc, n, nil, index: 2)
            Dispatch.setUInt(enc, w, index: 3)
            Dispatch.setUInt(enc, mp, index: 4)
            Dispatch.setUInt(enc, hasV ? 1 : 0, index: 5)
            enc.setBuffer(out.mtl, offset: out.offset, index: 6)
            enc.setBuffer(validBytes.mtl, offset: validBytes.offset, index: 7)
            Dispatch.dispatch1D(enc, p, count: n)
        }
        return try Self.assemble(ctx, self, out: out, validBytes: validBytes, n: n)
    }

    /// Trailing rolling mean as float64: the same two prefix arrays as `rollingSum`, divided by the
    /// number of non-null rows in the window. O(n), and it inherits `rollingSum`'s cancellation and
    /// NaN-propagation behaviour.
    public func rollingMean(window: Int, minPeriods: Int? = nil) throws -> MetalArray<Double> {
        let (w, mp) = try Self.rollingParams(window, minPeriods)
        let ctx = context
        let n = length
        try Dispatch.checkLength(n)
        let (out, validBytes) = try Self.windowOutputs(ctx, n: n, byteWidth: 8)
        guard n > 0 else { return MetalArray<Double>(length: 0, nullCount: 0, validity: nil, values: out, context: ctx) }
        let pre = try widenedToFloat64().cumulative(.sum)
        let counts = try validCountPrefix(n: n)
        let p = try Dispatch.pipeline(ctx, family: "window", source: WindowSource.doubleOps, function: "win_rolling_mean", type: "f64")
        try ctx.run { enc in
            enc.setComputePipelineState(p)
            enc.setBuffer(pre.values.mtl, offset: pre.values.offset, index: 0)
            enc.setBuffer(counts.values.mtl, offset: counts.values.offset, index: 1)
            Dispatch.setLength(enc, n, nil, index: 2)
            Dispatch.setUInt(enc, w, index: 3)
            Dispatch.setUInt(enc, mp, index: 4)
            enc.setBuffer(out.mtl, offset: out.offset, index: 5)
            enc.setBuffer(validBytes.mtl, offset: validBytes.offset, index: 6)
            Dispatch.dispatch1D(enc, p, count: n)
        }
        ctx.retainUntilFlush(pre); ctx.retainUntilFlush(counts)
        let bm = try BitmapOps.packBits(ctx, bytes: validBytes, bits: n)
        ctx.retainUntilFlush(validBytes)
        let res = MetalArray<Double>(length: n, nullCount: 0, validity: bm, values: out, context: ctx)
        res.recomputeNullCount()
        return res
    }

    // MARK: - Shared pieces

    /// This column widened to binary64 bit patterns, keeping its validity (itself when already float64).
    func widenedToFloat64() throws -> MetalArray<Double> {
        if let d = self as? MetalArray<Double> { return d }
        let ctx = context
        let n = length
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * 8, zeroed: false, context: ctx)
        if n > 0 {
            let p = try valuePipeline("win_to_f64")
            try ctx.run { enc in
                enc.setComputePipelineState(p)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                Dispatch.setLength(enc, n, nil, index: 1)
                enc.setBuffer(out.mtl, offset: out.offset, index: 2)
                Dispatch.dispatch1D(enc, p, count: n)
            }
            ctx.retainUntilFlush(self)
        }
        return MetalArray<Double>(length: n, nullCount: nullCount, validity: validity, values: out, context: ctx)
    }

    /// Inclusive prefix count of the non-null rows, as the values of an int32 array.
    private func validCountPrefix(n: Int) throws -> MetalArray<Int32> {
        let ctx = context
        let ones = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: false, context: ctx)
        let p = try Dispatch.pipeline(ctx, family: "window", source: WindowSource.doubleOps, function: "win_fill_i32", type: "f64")
        try ctx.run { enc in
            enc.setComputePipelineState(p)
            enc.setBuffer(ones.mtl, offset: ones.offset, index: 0)
            Dispatch.setLength(enc, n, nil, index: 1)
            Dispatch.setScalar(enc, Int32(1), index: 2)
            Dispatch.dispatch1D(enc, p, count: n)
        }
        ctx.retainUntilFlush(ones)
        // Nulls contribute the neutral element to the scan, so the running total counts valid rows only.
        return try MetalArray<Int32>(length: n, nullCount: nullCount, validity: validity, values: ones, context: ctx).cumulative(.sum)
    }

    /// Pipeline for one of the element-wise window kernels, specialised for this element type.
    private func valuePipeline(_ function: String) throws -> MTLComputePipelineState {
        let (src, cacheType) = Self.valueSource
        return try Dispatch.pipeline(context, family: "window", source: src, function: function, type: cacheType)
    }

    /// Generated element-wise window source for this element type, plus its pipeline cache key.
    static var valueSource: (source: String, type: String) {
        let ids = MathTypes.identities(T.self)
        if T.self == Double.self {
            return (WindowSource.values(V: "ulong", kind: .float64, unsigned: "ulong", identMin: ids.min, identMax: ids.max), "f64")
        }
        if T.isFloatingPoint {
            return (WindowSource.values(V: "float", kind: .float32, unsigned: "float", identMin: ids.min, identMax: ids.max), "f32")
        }
        let t = T.mslType
        return (WindowSource.values(V: t, kind: .integer, unsigned: MathTypes.unsigned(t), identMin: ids.min, identMax: ids.max), t)
    }

    /// Generated `cumulative_prod` scan source for this element type, plus its pipeline cache key.
    static var prodSource: (source: String, type: String) {
        if T.self == Double.self {
            let ops: [CumulativeSource.Op] = [(name: "prod", identity: "0x3FF0000000000000ul", body: "d_mul(a, b)")]
            return (CumulativeSource.source(V: "ulong", extraPrelude: DoubleMath.msl, ops: ops), "f64")
        }
        let t = T.mslType, u = MathTypes.unsigned(t)
        if T.isFloatingPoint {
            return (CumulativeSource.source(V: t, extraPrelude: "", ops: [(name: "prod", identity: "1.0f", body: "a * b")]), t)
        }
        let ops: [CumulativeSource.Op] = [(name: "prod", identity: "(\(t))1", body: "(\(t))((\(u))a * (\(u))b)")]
        return (CumulativeSource.source(V: t, extraPrelude: "", ops: ops), t)
    }

    /// A values buffer and a one-byte-per-element validity scratch buffer for a window result.
    static func windowOutputs(_ ctx: MetalContext, n: Int, byteWidth: Int) throws -> (MetalArrowBuffer, MetalArrowBuffer) {
        (try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * byteWidth, zeroed: false, context: ctx),
         try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1), zeroed: false, context: ctx))
    }

    /// Packs the per-element validity bytes into an Arrow bitmap and builds the result array.
    static func assemble(_ ctx: MetalContext, _ input: MetalArray<T>, out: MetalArrowBuffer,
                         validBytes: MetalArrowBuffer, n: Int) throws -> MetalArray<T> {
        let bm = try BitmapOps.packBits(ctx, bytes: validBytes, bits: n)
        ctx.retainUntilFlush(input); ctx.retainUntilFlush(validBytes)
        let res = MetalArray<T>(length: n, nullCount: 0, validity: bm, values: out, context: ctx)
        res.recomputeNullCount()
        return res
    }

    /// Validates a rolling window and resolves `minPeriods` (which defaults to the whole window).
    static func rollingParams(_ window: Int, _ minPeriods: Int?) throws -> (Int, Int) {
        guard window >= 1 else { throw ArrowMetalError.invalidArrowArray("rolling window must be at least 1, got \(window)") }
        let mp = minPeriods ?? window
        guard mp >= 1, mp <= window else {
            throw ArrowMetalError.invalidArrowArray("minPeriods must be in 1...\(window), got \(mp)")
        }
        return (window, mp)
    }
}

/// The sorted order and the per-run bookkeeping every ranking function shares: one argsort, one scan.
final class WindowOrder {
    /// Original row index of each sorted position (nulls last).
    let ord: MetalArray<Int32>
    /// One int32 per sorted position, 1 where a new tie group starts (position 0 always reads 0).
    let marks: MetalArrowBuffer
    /// Exclusive prefix sum of `marks`; adding a position's own mark gives its 0-based dense rank.
    let ranks: MetalArrowBuffer
    /// First and one-past-last sorted position of each tie group, indexed by dense rank.
    let startPos: MetalArrowBuffer
    let endPos: MetalArrowBuffer
    let n: Int
    /// Generated MSL for this element width, and its pipeline cache key.
    let source: String
    let unsignedType: String

    init(ord: MetalArray<Int32>, marks: MetalArrowBuffer, ranks: MetalArrowBuffer, startPos: MetalArrowBuffer,
         endPos: MetalArrowBuffer, n: Int, source: String, unsignedType: String) {
        self.ord = ord; self.marks = marks; self.ranks = ranks
        self.startPos = startPos; self.endPos = endPos; self.n = n
        self.source = source; self.unsignedType = unsignedType
    }
}
