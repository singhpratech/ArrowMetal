import Foundation

// External sort: sorted runs to disk, then a k-way merge.
//
// A sort is the one operator that cannot answer from a small running state — the last row read can
// belong at the front of the output — so an out-of-core sort has to spill. Each batch is sorted on
// the GPU (the existing radix argsort, one run per batch) and written to an Arrow IPC stream file;
// `finish()` merges the runs.
//
// **The merge is on the CPU**, deliberately. It is a k-way merge over k cursors, which is a
// branch-per-row loop with no parallelism to give a GPU: at 20 runs the whole merge is one
// comparison of 20 heap entries per output row. What *is* on the GPU is the gather: the merge
// decides the order, and each output batch is materialised with one `take` per run plus one
// permutation `take`, so no column data is ever rebuilt value by value on the host.
//
// Memory during the merge is one batch per run plus one output batch, so it is bounded by
// `runs * batch bytes`, not by the dataset.

/// Streaming external sort. Rows come out in full sort order through `sink`.
public final class ExternalSortOperator: StreamOperator {
    public struct Key: Sendable {
        public var column: String
        public var descending: Bool
        public init(_ column: String, descending: Bool = false) { self.column = column; self.descending = descending }
    }

    public let keys: [Key]
    public let sink: StreamSink
    /// Stop after this many output rows (`ORDER BY ... LIMIT n`).
    public let limit: Int?
    /// Rows per output batch out of the merge.
    public var outputBatchRows = 65_536
    /// Most runs merged at once. Each open run costs a file descriptor and one batch of memory, so a
    /// dataset that spills hundreds of runs is merged in passes rather than all at once.
    public var mergeFanIn = 32
    public private(set) var runURLs: [URL] = []
    /// Number of sorted runs spilled.
    public var runCount: Int { runURLs.count }

    private let scratch: URL
    private var runIndex = 0
    private let context: MetalContext
    private var deleteRuns: Bool

    // MARK: `ORDER BY ... LIMIT n` — top-n, not a sort
    //
    // With a limit the answer is n rows, so nothing ever has to spill: the running n rows stay
    // Metal-resident and each batch folds into them on the GPU thread. That is exactly the argument
    // the bounded fan-in already used — the global first n are inside the union of each part's first
    // n — applied one batch earlier, where it is worth a thousand times more: a run was a whole
    // million-row batch written to disk and read back, and now there are no runs at all.
    //
    // Pruning: once n rows are resident, the n-th row's *first* key bounds anything that can still
    // enter. `>=` keeps the rows that tie it, which is what makes this exact for a multi-key sort
    // too — a row tying on the first key can still win on the second.

    /// The running first n rows, in sort order. Only used when `limit` is set.
    private var running: MetalRecordBatch?
    /// The n-th row's first key, or null while fewer than n rows are resident (or that key is null,
    /// in which case nulls are still in play and nothing can be pruned).
    private var threshold: StreamValue = .null
    /// Set false to measure the unpruned path against the pruned one.
    public var pruning = true
    /// Batches in which no row passed the threshold.
    public private(set) var prunedBatches = 0
    /// Rows that survived the threshold and were sorted.
    public private(set) var candidateRows = 0
    /// True when this sort answers as a resident top-n instead of spilling runs.
    public var usesTopN: Bool { limit != nil }

    public init(keys: [Key], sink: StreamSink, scratch: URL, limit: Int? = nil,
                deleteRuns: Bool = true, context: MetalContext = .shared) throws {
        self.keys = keys
        self.sink = sink
        self.scratch = scratch
        self.limit = limit
        self.deleteRuns = deleteRuns
        self.context = context
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
    }

    public func process(_ batch: MetalRecordBatch) throws -> Any? {
        guard batch.length > 0 else { return nil }
        if let n = limit, n > 0 { try foldTopN(batch, n); return nil }
        let sorted = try sortWhole(batch)
        _ = sorted.length
        return sorted
    }

    /// A whole batch in sort order.
    private func sortWhole(_ batch: MetalRecordBatch) throws -> MetalRecordBatch {
        if keys.count == 1 {
            guard let c = batch[keys[0].column] else {
                throw ArrowMetalError.invalidArrowArray("no column named \(keys[0].column)")
            }
            return try batch.take(try argsortAny(c, descending: keys[0].descending))
        }
        return try batch.sorted(by: keys.map { (column: $0.column, descending: $0.descending) })
    }

    /// The first `n` rows of `batch` in the sort's own total order.
    ///
    /// For a single key the GPU top-k selection replaces the full radix sort: it returns *exactly*
    /// the first k indices `argsort` would (same order-preserving key, same tie-break by row index),
    /// so this is a cheaper way to compute the same prefix, not a different answer.
    private func sortedHead(_ batch: MetalRecordBatch, _ n: Int) throws -> MetalRecordBatch {
        if keys.count == 1, batch.length > n, let c = batch[keys[0].column],
           let idx = try topKIndicesIfSupported(c, k: n, largest: keys[0].descending) {
            return try batch.take(idx)
        }
        let sorted = try sortWhole(batch)
        return sorted.length > n ? try sorted.slice(offset: 0, length: n) : sorted
    }

    /// Folds one batch into the resident first n rows, on the GPU thread.
    private func foldTopN(_ batch: MetalRecordBatch, _ n: Int) throws {
        let ctx = batch.firstContext ?? context
        var work = batch
        if pruning, !threshold.isNull, let c = batch[keys[0].column],
           let m = try topNThresholdMask(c, threshold, largest: keys[0].descending) {
            guard let idx = try survivorIndices(m) else { prunedBatches += 1; return }
            work = try batch.take(idx)
        }
        guard work.length > 0 else { prunedBatches += 1; return }
        candidateRows += work.length

        let head = try sortedHead(work, n)
        // `concatColumns` copies inside unified memory on the host, so the head's kernels must have
        // run before it reads them.
        try ctx.syncPoint()
        if let r = running {
            running = try sortedHead(try concatBatches([r, head]), n)
            try ctx.syncPoint()
        } else {
            running = head
        }
        threshold = pruning ? try residentThreshold(running, column: keys[0].column, n: n) : .null
    }

    /// The run spill is host I/O over batches `process` has already flushed; no command buffer.
    public var mergeUsesGPU: Bool { false }

    public func merge(_ partial: Any) throws {
        guard let b = partial as? MetalRecordBatch, b.length > 0 else { return }
        let url = scratch.appendingPathComponent(String(format: "run-%05d.arrows", runIndex))
        runIndex += 1
        let s = try IPCStreamSink(url: url)
        try s.write(b)
        try s.finish()
        runURLs.append(url)
    }

    public func finish() throws -> StreamResult {
        if usesTopN {
            var written = 0
            if let r = running, r.length > 0 { try sink.write(r); written = r.length }
            try sink.finish()
            var res = StreamResult()
            res.rowsOut = written
            if let c = sink as? CollectingSink { res.batch = try c.table() }
            return res
        }
        var intermediates: [URL] = []
        defer {
            if deleteRuns { for u in runURLs { try? FileManager.default.removeItem(at: u) } }
            for u in intermediates { try? FileManager.default.removeItem(at: u) }
        }
        var written = 0
        if !runURLs.isEmpty {
            // Bounded fan-in. 589 batches means 589 runs, and opening them all at once would want 589
            // file descriptors and 589 resident batches; merging in passes of `mergeFanIn` keeps both
            // constant. `limit` is applied to every pass: the global first n rows are always inside
            // the union of each group's first n, so an intermediate run never needs to be longer.
            var runs = runURLs
            var pass = 0
            while runs.count > mergeFanIn {
                var next: [URL] = []
                var i = 0
                while i < runs.count {
                    let chunk = Array(runs[i..<Swift.min(i + mergeFanIn, runs.count)])
                    i += mergeFanIn
                    if chunk.count == 1 { next.append(chunk[0]); continue }
                    let url = scratch.appendingPathComponent(String(format: "merge-%d-%05d.arrows", pass, next.count))
                    let s = try IPCStreamSink(url: url)
                    _ = try kWayMerge(runs: chunk, keys: keys, sink: s, limit: limit,
                                      outputBatchRows: outputBatchRows, context: context)
                    try s.finish()
                    intermediates.append(url)
                    next.append(url)
                    // The inputs of this pass are done with; free their bytes before the next pass.
                    for u in chunk where u != url {
                        if deleteRuns || intermediates.contains(u) { try? FileManager.default.removeItem(at: u) }
                    }
                }
                runs = next
                pass += 1
            }
            written = try kWayMerge(runs: runs, keys: keys, sink: sink, limit: limit,
                                    outputBatchRows: outputBatchRows, context: context)
        }
        try sink.finish()
        var r = StreamResult()
        r.rowsOut = written
        if let c = sink as? CollectingSink { r.batch = try c.table() }
        return r
    }
}

/// One cursor into a sorted run: the batch it is reading, the row it is on, and that batch's keys.
private final class RunCursor {
    let source: IPCFileSource
    var batch: MetalRecordBatch?
    var row = 0
    /// Key values of the current batch, one column per sort key. `StreamColumn` rather than
    /// `[StreamValue]`: a merge holds one batch per open run, so the key storage is the cursor's
    /// whole memory footprint and plain `[Int64]` / `[Double]` is a third of the boxed form.
    var keyValues: [StreamColumn] = []

    init(url: URL, context: MetalContext) throws {
        self.source = try IPCFileSource(url: url, context: context)
    }

    /// Loads the next non-empty batch and its key columns. Returns false at end of run.
    func advanceBatch(_ keys: [ExternalSortOperator.Key]) throws -> Bool {
        while true {
            guard let b = try source.nextBatch() else { batch = nil; return false }
            if b.length == 0 { continue }
            batch = b
            row = 0
            keyValues = try keys.map { k in
                guard let c = b[k.column] else {
                    throw ArrowMetalError.invalidArrowArray("run is missing sort column \(k.column)")
                }
                return try c.streamColumn()
            }
            return true
        }
    }

    var exhausted: Bool { batch == nil || row >= (batch?.length ?? 0) }

    func key(_ i: Int) -> StreamValue { keyValues[i].value(row) }
}

/// Merges sorted runs into one ordered stream, writing output batches to `sink`.
///
/// Returns the number of rows written. The comparison loop is on the CPU; every output batch is
/// assembled on the GPU with `take`.
func kWayMerge(runs: [URL], keys: [ExternalSortOperator.Key], sink: StreamSink,
               limit: Int?, outputBatchRows: Int, context: MetalContext) throws -> Int {
    var cursors: [RunCursor] = []
    for u in runs {
        let c = try RunCursor(url: u, context: context)
        if try c.advanceBatch(keys) { cursors.append(c) } else { c.source.close() }
    }
    guard !cursors.isEmpty else { return 0 }

    /// True when run `a`'s current row sorts before run `b`'s.
    func before(_ a: RunCursor, _ b: RunCursor) -> Bool {
        for (i, k) in keys.enumerated() {
            let x = a.key(i), y = b.key(i)
            if x == y { continue }
            // Nulls last in both directions, as `argsort` places them.
            if x.isNull { return false }
            if y.isNull { return true }
            let lt = StreamValue.less(x, y)
            return k.descending ? !lt : lt
        }
        return false
    }

    var written = 0
    var plan: [(run: Int, row: Int)] = []
    plan.reserveCapacity(outputBatchRows)

    func flush() throws {
        guard !plan.isEmpty else { return }
        // Group the chunk's rows by run so each run is gathered with a single GPU `take`, then undo the
        // grouping with one permutation `take` so the output really is in merged order.
        var perRun: [Int: [Int32]] = [:]
        var slot: [Int] = []
        slot.reserveCapacity(plan.count)
        for (r, row) in plan {
            perRun[r, default: []].append(Int32(row))
            slot.append(r)
        }
        let runOrder = perRun.keys.sorted()
        var base: [Int: Int] = [:]
        var offset = 0
        for r in runOrder { base[r] = offset; offset += perRun[r]!.count }
        var cursorPer: [Int: Int] = [:]
        var permutation: [Int32] = []
        permutation.reserveCapacity(plan.count)
        for r in slot {
            let i = cursorPer[r, default: 0]
            cursorPer[r] = i + 1
            permutation.append(Int32(base[r]! + i))
        }
        var parts: [MetalRecordBatch] = []
        for r in runOrder {
            let idx = try MetalArray<Int32>(perRun[r]!, context: context)
            parts.append(try cursors[r].batch!.take(idx))
        }
        let grouped = try concatBatches(parts)
        let out = try grouped.take(try MetalArray<Int32>(permutation, context: context))
        try sink.write(out)
        written += out.length
        plan.removeAll(keepingCapacity: true)
    }

    outer: while true {
        // Pick the smallest head. With a handful of runs a linear scan beats a heap's bookkeeping.
        var best = -1
        for (i, c) in cursors.enumerated() where !c.exhausted {
            if best < 0 || before(c, cursors[best]) { best = i }
        }
        if best < 0 { break }
        plan.append((best, cursors[best].row))
        cursors[best].row += 1
        if let limit, written + plan.count >= limit {
            try flush()
            break outer
        }
        if plan.count >= outputBatchRows { try flush() }
        if cursors[best].exhausted {
            // The chunk plan still references this run's current batch, so materialise before moving on.
            try flush()
            if !(try cursors[best].advanceBatch(keys)) { cursors[best].source.close() }
        }
    }
    try flush()
    for c in cursors { c.source.close() }
    return written
}
