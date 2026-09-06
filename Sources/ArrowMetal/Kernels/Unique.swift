import Foundation
import Metal

/// The sorted order of the non-null values plus a run-start mark per sorted position. Shared by
/// `unique`, `valueCounts` and `dictionaryEncode` so each of them is one argsort plus small passes.
final class SortedRuns {
    /// Original row index of each non-null value, ascending by value (stable).
    let ord: MetalArray<Int32>
    /// One byte per sorted position, 1 where a new distinct value starts. Feeds the bitmap packer.
    let markBytes: MetalArrowBuffer
    /// The same marks as int32, input to the rank scan.
    let markInts: MetalArrowBuffer
    /// Number of non-null values (the length of `ord`).
    let count: Int
    /// Generated MSL for this element width, and its cache-key type.
    let source: String
    let unsignedType: String

    init(ord: MetalArray<Int32>, markBytes: MetalArrowBuffer, markInts: MetalArrowBuffer, count: Int,
         source: String, unsignedType: String) {
        self.ord = ord
        self.markBytes = markBytes
        self.markInts = markInts
        self.count = count
        self.source = source
        self.unsignedType = unsignedType
    }
}

extension MetalArray {
    // MARK: - Public API

    /// Arrow `unique`, in ascending order: the distinct non-null values of this array.
    ///
    /// Nulls are excluded entirely (`length - nullCount` tells you how many rows contributed).
    /// Floating point follows Arrow value equality rather than bit equality: all NaNs are one value
    /// (sorted last, after +inf) and `-0.0` equals `0.0`. The representative returned for a group of
    /// equal values is the one from the earliest row, so `-0.0` comes back when it appeared first.
    ///
    /// GPU: normalise (floats only), argsort, mark run boundaries, compact the marks with `filter`,
    /// gather with `take`.
    ///
    /// Above `1 << 16` rows the distinct values come from the GPU hash table in `Kernels/HashTable.swift`
    /// instead, which costs what the *distinct* count costs rather than what the row count costs; only
    /// the `K` distinct values are sorted, so the order and the representatives are unchanged.
    public func unique() throws -> MetalArray<T> {
        if Self.prefersHashTable(rows: length), let d = try hashDistinct() { return d.values }
        guard let runs = try sortedRuns() else { return try MetalArray<T>([T](), context: context) }
        let (firstIdx, _) = try runStarts(runs)
        return try gatherUnique(firstIdx)
    }

    /// Arrow `value_counts`: the distinct non-null values (ascending, as `unique`) and how many rows
    /// carry each of them. Counts are the differences between adjacent run starts in the sorted order.
    ///
    /// Above `1 << 16` rows this runs on the hash table (`Kernels/HashTable.swift`): the counts are then
    /// a dense-key group-by over the codes rather than the gaps between run starts.
    public func valueCounts() throws -> (values: MetalArray<T>, counts: MetalArray<Int64>) {
        if Self.prefersHashTable(rows: length), let d = try hashDistinct() {
            return (d.values, try hashCounts(d))
        }
        guard let runs = try sortedRuns() else {
            return (try MetalArray<T>([T](), context: context), try MetalArray<Int64>([Int64](), context: context))
        }
        let (firstIdx, pos) = try runStarts(runs)
        let values = try gatherUnique(firstIdx)
        let counts = try runLengths(runs, pos: pos)
        return (values, counts)
    }

    /// Arrow `dictionary_encode`: dense Int32 codes into the sorted unique values, plus those values.
    ///
    /// A null row gets a null code. Codes are ranks, so `unique[Int(codes[i]!)]` equals row `i`, and the
    /// codes are exactly the dense keys `GroupBy` wants. The rank of each sorted position is the
    /// exclusive prefix sum of the run-start marks, scanned on the GPU and scattered back to the
    /// original rows.
    ///
    /// Above `1 << 16` rows the codes come from the hash table (`Kernels/HashTable.swift`) rather than a
    /// sort of every row; they index the same ascending dictionary either way.
    public func dictionaryEncode() throws -> (codes: MetalArray<Int32>, unique: MetalArray<T>) {
        let ctx = context
        let n = length
        if Self.prefersHashTable(rows: n), let d = try hashDistinct() {
            return (try hashCodes(d), d.values)
        }
        guard let runs = try sortedRuns() else {
            // Empty, or every row null: every code is null.
            let buf = try MetalArrowBuffer.allocate(byteCount: n * 4, context: ctx)
            let codes = MetalArray<Int32>(length: n, nullCount: n, validity: validity, values: buf, context: ctx)
            return (codes, try MetalArray<T>([T](), context: ctx))
        }
        let (firstIdx, _) = try runStarts(runs)
        let unique = try gatherUnique(firstIdx)
        let codes = try scatterRanks(runs, rows: n)
        return (codes, unique)
    }

    /// Dictionary-encodes this column and wraps the codes in a dense-key `GroupBy`, so aggregation works
    /// over arbitrary Int64 / Int32 / Float keys: `keys.groupBy().0.sum(values)`.
    ///
    /// Group `k` of the result belongs to `unique[k]`. An empty or all-null column has no groups; the
    /// returned `GroupBy` then has the minimum key count of one and every aggregate over it is empty.
    public func groupBy() throws -> (GroupBy<Int32>, unique: MetalArray<T>) {
        let (codes, unique) = try dictionaryEncode()
        return (try GroupBy(keys: codes, keyCount: Swift.max(unique.length, 1)), unique)
    }

    // MARK: - Stages

    /// Sorts the non-null values and marks the run boundaries, or returns nil when nothing is non-null.
    func sortedRuns() throws -> SortedRuns? {
        let ctx = context
        let n = length
        let nonNull = n - nullCount
        guard nonNull > 0 else { return nil }
        try Dispatch.checkLength(n)
        let uType = UniqueSource.unsignedType(width: T.byteWidth)
        let src = UniqueSource.source(U: uType)
        func pso(_ f: String) throws -> MTLComputePipelineState {
            try Dispatch.pipeline(ctx, family: "unique", source: src, function: f, type: uType)
        }

        // Floats are compared as raw bit patterns, so normalise first: every NaN collapses to one pattern
        // and -0 becomes +0. That makes bit equality mean Arrow value equality, and keeps all NaNs
        // adjacent in the sort's total order instead of splitting them across both ends.
        var keyValues = values
        var keyArray: MetalArray<T> = self
        if T.isFloatingPoint {
            let norm = try MetalArrowBuffer.allocate(byteCount: n * T.byteWidth, zeroed: false, context: ctx)
            let normPSO = try pso(T.byteWidth == 8 ? "uq_norm_f64" : "uq_norm_f32")
            try ctx.run { enc in
                enc.setComputePipelineState(normPSO)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                Dispatch.setLength(enc, n, nil, index: 1)
                enc.setBuffer(norm.mtl, offset: norm.offset, index: 2)
                Dispatch.dispatch1D(enc, normPSO, count: n)
            }
            keyValues = norm
            keyArray = MetalArray<T>(length: n, nullCount: nullCount, validity: validity, values: norm, context: ctx)
        }

        // argsort puts nulls last, so the first `nonNull` indices are the sorted non-null rows.
        let ord = try keyArray.argsort().slice(offset: 0, length: nonNull)
        let markBytes = try MetalArrowBuffer.allocate(byteCount: nonNull, zeroed: false, context: ctx)
        let markInts = try MetalArrowBuffer.allocate(byteCount: nonNull * 4, zeroed: false, context: ctx)
        let markPSO = try pso("uq_mark")
        // The normalised buffer is nobody else's after this dispatch, so keep it alive across it.
        try withExtendedLifetime(keyValues) {
            try ctx.run { enc in
                enc.setComputePipelineState(markPSO)
                enc.setBuffer(keyValues.mtl, offset: keyValues.offset, index: 0)
                enc.setBuffer(ord.values.mtl, offset: ord.values.offset, index: 1)
                Dispatch.setLength(enc, nonNull, nil, index: 2)
                enc.setBuffer(markBytes.mtl, offset: markBytes.offset, index: 3)
                enc.setBuffer(markInts.mtl, offset: markInts.offset, index: 4)
                Dispatch.dispatch1D(enc, markPSO, count: nonNull)
            }
            ctx.retainUntilFlush(keyValues)
        }
        return SortedRuns(ord: ord, markBytes: markBytes, markInts: markInts, count: nonNull,
                          source: src, unsignedType: uType)
    }

    /// Compacts the marks into the sorted positions of the run starts (`pos`) and the original rows they
    /// point at (`firstIdx`), using the existing filter and take kernels.
    func runStarts(_ runs: SortedRuns) throws -> (firstIdx: MetalArray<Int32>, pos: MetalArray<Int32>) {
        let ctx = context
        let selection = try BitmapOps.packBits(ctx, bytes: runs.markBytes, bits: runs.count)
        let mask = MetalBooleanArray(length: runs.count, nullCount: 0, validity: nil, values: selection, context: ctx)
        let iotaBuf = try MetalArrowBuffer.allocate(byteCount: runs.count * 4, zeroed: false, context: ctx)
        let iotaPSO = try Dispatch.pipeline(ctx, family: "unique", source: runs.source, function: "uq_iota", type: runs.unsignedType)
        try ctx.run { enc in
            enc.setComputePipelineState(iotaPSO)
            enc.setBuffer(iotaBuf.mtl, offset: iotaBuf.offset, index: 0)
            Dispatch.setLength(enc, runs.count, nil, index: 1)
            Dispatch.dispatch1D(enc, iotaPSO, count: runs.count)
        }
        let iota = MetalArray<Int32>(length: runs.count, nullCount: 0, validity: nil, values: iotaBuf, context: ctx)
        let pos = try iota.filter(mask)
        return (try runs.ord.take(pos), pos)
    }

    /// Gathers the representative value of each run. The result never has nulls, so it carries no bitmap.
    func gatherUnique(_ firstIdx: MetalArray<Int32>) throws -> MetalArray<T> {
        let gathered = try take(firstIdx)
        guard gathered.validity != nil else { return gathered }
        return MetalArray<T>(length: gathered.length, nullCount: 0, validity: nil, values: gathered.values, context: context)
    }

    /// Rows per distinct value: the gap between adjacent run starts, the last one closing at `runs.count`.
    func runLengths(_ runs: SortedRuns, pos: MetalArray<Int32>) throws -> MetalArray<Int64> {
        let ctx = context
        let u = pos.length
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(u, 1) * 8, zeroed: false, context: ctx)
        if u > 0 {
            let p = try Dispatch.pipeline(ctx, family: "unique", source: runs.source, function: "uq_run_lengths", type: runs.unsignedType)
            try ctx.run { enc in
                enc.setComputePipelineState(p)
                enc.setBuffer(pos.values.mtl, offset: pos.values.offset, index: 0)
                Dispatch.setLength(enc, u, nil, index: 1)
                Dispatch.setUInt(enc, runs.count, index: 2)
                enc.setBuffer(out.mtl, offset: out.offset, index: 3)
                Dispatch.dispatch1D(enc, p, count: u)
            }
        }
        return MetalArray<Int64>(length: u, nullCount: 0, validity: nil, values: out, context: ctx)
    }

    /// Exclusive prefix sum of the run marks (the rank of each sorted position) scattered back to the
    /// original rows. Null rows keep the zero the buffer was allocated with and stay null.
    func scatterRanks(_ runs: SortedRuns, rows: Int) throws -> MetalArray<Int32> {
        let ctx = context
        let m = runs.count
        let blocks = Swift.max(1, (m + Dispatch.threadgroupSize - 1) / Dispatch.threadgroupSize)
        let ranks = try MetalArrowBuffer.allocate(byteCount: m * 4, zeroed: false, context: ctx)
        let blockTotals = try MetalArrowBuffer.allocate(byteCount: blocks * 4, zeroed: false, context: ctx)
        let codes = try MetalArrowBuffer.allocate(byteCount: rows * 4, context: ctx)
        func pso(_ f: String) throws -> MTLComputePipelineState {
            try Dispatch.pipeline(ctx, family: "unique", source: runs.source, function: f, type: runs.unsignedType)
        }
        let blockPSO = try pso("uq_scan_block"), totalsPSO = try pso("uq_scan_totals")
        let addPSO = try pso("uq_scan_add"), scatterPSO = try pso("uq_scatter_codes")
        let tg = MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1)
        let grid = MTLSize(width: blocks, height: 1, depth: 1)
        try ctx.run { enc in
            enc.setComputePipelineState(blockPSO)
            enc.setBuffer(runs.markInts.mtl, offset: runs.markInts.offset, index: 0)
            Dispatch.setLength(enc, m, nil, index: 1)
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
            Dispatch.setLength(enc, m, nil, index: 2)
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(scatterPSO)
            enc.setBuffer(runs.ord.values.mtl, offset: runs.ord.values.offset, index: 0)
            enc.setBuffer(ranks.mtl, offset: ranks.offset, index: 1)
            enc.setBuffer(runs.markInts.mtl, offset: runs.markInts.offset, index: 2)
            Dispatch.setLength(enc, m, nil, index: 3)
            enc.setBuffer(codes.mtl, offset: codes.offset, index: 4)
            Dispatch.dispatch1D(enc, scatterPSO, count: m)
        }
        try ctx.syncPoint()
        return MetalArray<Int32>(length: rows, nullCount: nullCount, validity: validity, values: codes, context: ctx)
    }
}
