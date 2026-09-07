import Foundation
import Metal

// Resident top-n with threshold pruning.
//
// The old streaming top-k was correct and slow for one structural reason: every batch paid the full
// selection over all of its rows, and the k-row merge ran on the merge thread where each dispatch is
// its own command buffer. Merging 100 rows into 100 rows measured ~7 ms a batch — microseconds of
// arithmetic behind twenty commit-and-wait round trips.
//
// Both halves are fixed here:
//
// * **The k rows stay on the GPU and the fold happens on the GPU thread**, inside the same command
//   buffer as the batch's own work. The merge stage does nothing at all, and — the part that matters
//   more — the threshold the *next* batch prunes with is always the newest one.
// * **Threshold pruning.** Once k rows are resident, the k-th value is a lower bound on anything that
//   can still enter the answer. Each later batch first runs one comparison kernel over the key
//   column; almost no row passes after the first handful of batches, so the selection, the gather and
//   the fold are skipped entirely. On a 570-batch scan for the top 100, the expected number of
//   survivors in batch i is k / i — the work per batch collapses to a single pass over one column.
//
// The comparison is `>=`, not `>`. A row that *ties* the running k-th value can still be the row the
// total order picks (`topK` breaks ties by row index), so keeping ties is what makes the pruned
// answer identical to the unpruned one rather than merely equal in its values.

/// A boolean mask of the rows of `c` that could still enter a running top-n whose n-th value is
/// `threshold`. Nil when the column's type cannot be compared against the threshold, which simply
/// turns pruning off for that scan.
///
/// A null value never passes: nulls sort last, so once n non-null rows are resident no null row can
/// displace one. (The caller only sets a threshold when the n-th row's key is non-null.)
func topNThresholdMask(_ c: AnyMetalArray, _ threshold: StreamValue, largest: Bool) throws -> MetalBooleanArray? {
    guard !threshold.isNull else { return nil }
    let op: CompareOp = largest ? .ge : .le

    var signed: Int64? = nil
    var unsigned: UInt64? = nil
    switch threshold {
    case .int(let v): signed = v; unsigned = v >= 0 ? UInt64(v) : nil
    case .uint(let v): unsigned = v; signed = v <= UInt64(Int64.max) ? Int64(v) : nil
    case .bool(let b): signed = b ? 1 : 0; unsigned = b ? 1 : 0
    default: break
    }
    let real = threshold.asDouble

    switch c {
    case .int8(let a): guard let v = signed, let t = Int8(exactly: v) else { return nil }; return try a.compare(op, t)
    case .int16(let a): guard let v = signed, let t = Int16(exactly: v) else { return nil }; return try a.compare(op, t)
    case .int32(let a): guard let v = signed, let t = Int32(exactly: v) else { return nil }; return try a.compare(op, t)
    case .int64(let a): guard let v = signed else { return nil }; return try a.compare(op, v)
    case .uint8(let a): guard let v = unsigned, let t = UInt8(exactly: v) else { return nil }; return try a.compare(op, t)
    case .uint16(let a): guard let v = unsigned, let t = UInt16(exactly: v) else { return nil }; return try a.compare(op, t)
    case .uint32(let a): guard let v = unsigned, let t = UInt32(exactly: v) else { return nil }; return try a.compare(op, t)
    case .uint64(let a): guard let v = unsigned else { return nil }; return try a.compare(op, v)
    case .float32(let a):
        guard let v = real, v.isFinite || v.isInfinite else { return nil }
        return try a.compare(op, Float(v))
    case .float64(let a): guard let v = real else { return nil }; return try a.compare(op, v)
    case .temporal(let t):
        guard let v = signed else { return nil }
        switch t.storage {
        case .int32(let a): guard let s = Int32(exactly: v) else { return nil }; return try a.compare(op, s)
        case .int64(let a): return try a.compare(op, v)
        }
    // Everything else keeps the unpruned path: correctness never depends on pruning.
    case .boolean, .string, .binary, .decimal, .smallDecimal, .list, .structure, .map, .union,
         .runEndEncoded, .null, .float16, .interval, .fixedBinary, .extended, .dictionary:
        return nil
    }
}

/// The row numbers where `mask` is a valid `true`, as int32 gather indices, or nil when there are
/// none. Reading the length is the one host round trip a pruned batch pays.
func survivorIndices(_ mask: MetalBooleanArray) throws -> MetalArray<Int32>? {
    let wide = try mask.indicesNonzero()
    guard wide.length > 0 else { return nil }
    return try wide.cast(to: Int32.self)
}

/// `topKIndices` where the value type has a GPU selection, nil where it does not (`utf8`, `binary`,
/// `boolean`, decimals, nested types), so the caller can fall back to the full ordering.
func topKIndicesIfSupported(_ c: AnyMetalArray, k: Int, largest: Bool) throws -> MetalArray<Int32>? {
    switch c {
    case .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64, .float32, .float64, .temporal:
        guard k > 0, k <= c.length else { return nil }
        return try topKIndices(c, k: k, largest: largest)
    case .boolean, .string, .binary, .decimal, .smallDecimal, .list, .structure, .map, .union,
         .runEndEncoded, .null, .float16, .interval, .fixedBinary, .extended, .dictionary:
        return nil
    }
}

/// The value at rank `n - 1` of a resident, already-ranked result — the threshold later batches prune
/// against. Nil when the result is shorter than `n` or that value is null.
func residentThreshold(_ batch: MetalRecordBatch?, column: String, n: Int) throws -> StreamValue {
    guard n > 0, let b = batch, b.length >= n, let c = b[column] else { return .null }
    let vals = try c.streamValues()
    guard vals.count >= n else { return .null }
    let v = vals[n - 1]
    return v.isNull ? .null : v
}

// MARK: - Streaming top-k

/// Streaming top-k: the k best rows of the whole dataset, kept Metal-resident across batches.
///
/// The k best rows of a union are always inside the union of each part's k best, so a batch can never
/// contribute more than k rows. What makes this fast is that it usually contributes none: after the
/// first few batches the running k-th value is already near the dataset's k-th value, and a single
/// comparison kernel rejects the whole batch (see the file comment). State is O(k) whatever the
/// dataset's size, and the merge stage is idle — the fold runs on the GPU thread so that the next
/// batch prunes against the newest threshold.
public final class StreamTopKOperator: StreamOperator {
    public let column: String
    public let k: Int
    public let largest: Bool
    public let filter: Expr?
    /// Set false to measure the unpruned path against the pruned one.
    public var pruning = true

    private var running: MetalRecordBatch?
    private var threshold: StreamValue = .null
    private var filterQuery: ExprQuery?

    /// Batches in which no row passed the threshold, so nothing but the comparison ran.
    public private(set) var prunedBatches = 0
    /// Rows that survived the threshold and reached the selection.
    public private(set) var candidateRows = 0

    public init(column: String, k: Int, largest: Bool = true, filter: Expr? = nil) {
        self.column = column
        self.k = Swift.max(0, k)
        self.largest = largest
        self.filter = filter
    }

    /// The fold happens in `process`, on the GPU thread; `merge` does nothing.
    public var mergeUsesGPU: Bool { false }

    public func process(_ batch: MetalRecordBatch) throws -> Any? {
        guard k > 0, batch.length > 0 else { return nil }
        let ctx = batch.firstContext ?? .shared

        var mask: MetalBooleanArray? = nil
        if let f = filter {
            // One query, built once: the predicate's fused kernel is compiled on the first batch and
            // cached for the other five hundred.
            if filterQuery == nil { filterQuery = ExprQueryBuilder().project([("__mask", f)]) }
            let r = try runExprQuery(filterQuery!, names: batch.names, columns: batch.columns, context: ctx)
            guard let m = r["__mask"]?.asBoolean else {
                throw ArrowMetalError.unsupportedType("a streaming top-k predicate must be boolean")
            }
            mask = m
        }
        if pruning, !threshold.isNull, let c = batch[column],
           let pm = try topNThresholdMask(c, threshold, largest: largest) {
            mask = try mask.map { try $0.and(pm) } ?? pm
        }

        var work = batch
        if let m = mask {
            // Gather the survivors rather than compacting every column: after a few batches there are
            // single digits of them, and a gather of ten rows costs nothing where a compaction of a
            // million-row column costs a pass over every byte of the batch.
            guard let idx = try survivorIndices(m) else { prunedBatches += 1; return nil }
            work = try batch.take(idx)
        }
        guard work.length > 0 else { prunedBatches += 1; return nil }
        candidateRows += work.length

        let cand = try topKRows(work, column: column, k: k, largest: largest)
        // `concatColumns` copies inside unified memory on the host, so the candidate's kernels have to
        // have run before it reads them.
        try ctx.syncPoint()
        if let r = running {
            running = try topKRows(try concatBatches([r, cand]), column: column, k: k, largest: largest)
            try ctx.syncPoint()
        } else {
            running = cand
        }
        threshold = pruning ? try residentThreshold(running, column: column, n: k) : .null
        return nil
    }

    public func merge(_ partial: Any) throws {}

    public func finish() throws -> StreamResult {
        var r = StreamResult()
        r.batch = running
        r.rowsOut = running?.length ?? 0
        return r
    }
}
