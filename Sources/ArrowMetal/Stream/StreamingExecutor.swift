import Foundation
import Metal

// The pipeline. Three stages run at once on three threads:
//
//   stage 1  read      PrefetchingSource's own thread: map / import batch i + 1
//   stage 2  GPU       this thread: record and run batch i's kernels
//   stage 3  merge     the merge thread: fold batch i - 1's small GPU result into the running state
//
// The stages are connected by bounded queues, so a fast SSD cannot outrun the GPU and a slow merge
// cannot let batches pile up: memory stays inside the prefetch budget however large the dataset is.
// Every stage records its own busy time; `StreamStats.overlap` is (read + gpu + merge) / wall, which
// is 1.0 for a serial pipeline and approaches 3.0 for three fully overlapped stages.
//
// The GPU stage always produces a *small* result — a handful of scalars, one row per group, k rows,
// a sketch — because that is what makes streaming work: the merge stage's cost is a function of the
// answer's size, not the dataset's.

/// What each stage of the pipeline cost, and how much of it overlapped.
public struct StreamStats: Sendable {
    /// Record batches pulled from the source.
    public var batches = 0
    /// Rows read.
    public var rows = 0
    /// Bytes the source read from storage.
    public var bytesRead: Int64 = 0
    /// Stage-one busy time (the reader thread inside the source).
    public var readNanos: UInt64 = 0
    /// Stage-two busy time (recording and running kernels).
    public var gpuNanos: UInt64 = 0
    /// Stage-three busy time (the merge thread).
    public var mergeNanos: UInt64 = 0
    /// Time the GPU stage spent waiting for a batch that the reader had not finished.
    public var readStallNanos: UInt64 = 0
    /// Time the GPU stage spent waiting for the merge stage to drain (backpressure).
    public var mergeStallNanos: UInt64 = 0
    /// End to end.
    public var wallNanos: UInt64 = 0

    public var wallSeconds: Double { Double(wallNanos) / 1e9 }
    /// Read throughput from storage.
    public var gigabytesPerSecond: Double {
        wallNanos == 0 ? 0 : Double(bytesRead) / 1e9 / wallSeconds
    }
    /// Sum of the three stages' busy time over wall time: 1.0 serial, up to 3.0 fully overlapped.
    public var overlap: Double {
        wallNanos == 0 ? 0 : Double(readNanos + gpuNanos + mergeNanos) / Double(wallNanos)
    }

    public var description: String {
        String(format: "%d batches / %d rows in %.3f s (read %.3f, gpu %.3f, merge %.3f; overlap %.2fx, %.2f GB/s)",
               batches, rows, wallSeconds, Double(readNanos) / 1e9, Double(gpuNanos) / 1e9,
               Double(mergeNanos) / 1e9, overlap, gigabytesPerSecond)
    }
}

/// Progress reported to a caller-supplied callback as batches go by.
public struct StreamProgress: Sendable {
    public var batches: Int
    public var rows: Int
    public var bytesRead: Int64
    public var totalBytes: Int64
    public var elapsedSeconds: Double
    /// Fraction of the source read, or nil when the source's size is unknown.
    public var fraction: Double? { totalBytes > 0 ? Swift.min(1, Double(bytesRead) / Double(totalBytes)) : nil }
}

/// A streamed query's answer.
public struct StreamResult {
    public var scalarNames: [String] = []
    public var scalars: [ExprScalar] = []
    /// The result rows for a group-by, a top-k or a sorted / filtered result that was collected.
    public var batch: MetalRecordBatch?
    /// Rows the sink received (0 when the terminal is an aggregate).
    public var rowsOut = 0
    public var stats = StreamStats()

    public init() {}

    public func scalar(_ name: String) -> ExprScalar? {
        scalarNames.firstIndex(of: name).map { scalars[$0] }
    }
    public var onlyScalar: ExprScalar? { scalars.count == 1 ? scalars[0] : nil }
}

/// One streaming operator: a GPU half that runs per batch and a merge half that folds the small GPU
/// result into the running state. `process` runs on the GPU thread, `merge` on the merge thread, and
/// `finish` once at the end on the caller's thread.
public protocol StreamOperator: AnyObject {
    /// Runs the per-batch GPU work and returns whatever the merge stage needs (nil to merge nothing).
    func process(_ batch: MetalRecordBatch) throws -> Any?
    /// Folds one batch's result into the running state. Called in batch order, one at a time.
    func merge(_ partial: Any) throws
    /// The finished answer.
    func finish() throws -> StreamResult
    /// True when `merge` records Metal kernels, so the executor wraps the whole merge in **one**
    /// command buffer. `MetalContext.currentBatch` is thread local and the merge runs on its own
    /// thread, so without this every dispatch inside a merge is its own commit-and-wait round trip:
    /// a top-k merge of 200 rows was a concat, a select and a gather over every column — twenty
    /// round trips for microseconds of work. Operators whose merge is pure host arithmetic return
    /// false, so they do not pay for an empty command buffer per batch.
    var mergeUsesGPU: Bool { get }
}

public extension StreamOperator {
    var mergeUsesGPU: Bool { true }
}

// MARK: - The executor

/// Runs a `StreamOperator` over a `BatchSource` as a three-stage pipeline.
public final class StreamingExecutor {
    public let source: BatchSource
    public let context: MetalContext
    /// Called on the GPU thread after each batch. Keep it cheap.
    public var progress: ((StreamProgress) -> Void)?
    /// How many merged results may be queued before the GPU stage blocks.
    public var mergeDepth = 2

    public init(source: BatchSource, context: MetalContext = .shared) {
        self.source = source
        self.context = context
    }

    public func run(_ op: StreamOperator) throws -> StreamResult {
        let t0 = machNow()
        var stats = StreamStats()

        let lock = NSCondition()
        var queue: [Any] = []
        var producerDone = false
        var mergeError: Error?
        var mergeNanos: UInt64 = 0

        let mergeThread = Thread {
            while true {
                lock.lock()
                while queue.isEmpty && !producerDone { lock.wait() }
                if queue.isEmpty && producerDone { lock.unlock(); return }
                let p = queue.removeFirst()
                lock.broadcast()
                lock.unlock()
                let m0 = machNow()
                // One command buffer for the whole merge, not one per kernel. The merge thread has no
                // open batch of its own (`currentBatch` is thread local), so an unbatched merge paid a
                // commit-and-wait round trip *per dispatch* — a top-k merge of 200 rows is a concat,
                // a select and a gather over every column, twenty round trips for microseconds of work.
                do {
                    if op.mergeUsesGPU { try self.context.batch { try op.merge(p) } }
                    else { try op.merge(p) }
                } catch {
                    lock.lock()
                    if mergeError == nil { mergeError = error }
                    producerDone = true
                    queue.removeAll()
                    lock.broadcast()
                    lock.unlock()
                    return
                }
                mergeNanos &+= nanos(since: m0)
            }
        }
        mergeThread.name = "ArrowMetal.merge"
        mergeThread.stackSize = 1 << 21
        mergeThread.start()

        func drain() {
            lock.lock(); producerDone = true; lock.broadcast(); lock.unlock()
            while !mergeThread.isFinished { usleep(50) }
        }

        do {
            while true {
                let r0 = machNow()
                guard let batch = try source.nextBatch() else { break }
                stats.readStallNanos &+= nanos(since: r0)

                let g0 = machNow()
                let partial = try context.batch { try op.process(batch) }
                stats.gpuNanos &+= nanos(since: g0)

                stats.batches += 1
                stats.rows += batch.length

                if let partial {
                    let s0 = machNow()
                    lock.lock()
                    while queue.count >= mergeDepth && mergeError == nil { lock.wait() }
                    if let e = mergeError { lock.unlock(); throw e }
                    queue.append(partial)
                    lock.broadcast()
                    lock.unlock()
                    stats.mergeStallNanos &+= nanos(since: s0)
                }
                if let p = progress {
                    p(StreamProgress(batches: stats.batches, rows: stats.rows, bytesRead: source.bytesRead,
                                     totalBytes: source.totalBytes,
                                     elapsedSeconds: Double(nanos(since: t0)) / 1e9))
                }
            }
        } catch {
            drain()
            throw error
        }
        drain()
        if let e = mergeError { throw e }

        var result = try op.finish()
        stats.mergeNanos = mergeNanos
        stats.bytesRead = source.bytesRead
        if let p = source as? PrefetchingSource { stats.readNanos = p.readNanos }
        else if let p = source as? ParallelIPCSource { stats.readNanos = p.readNanos }
        else { stats.readNanos = stats.readStallNanos }
        stats.wallNanos = nanos(since: t0)
        result.stats = stats
        return result
    }
}

// MARK: - Scalar merge helpers

/// Adds two Arrow scalars of the same shape, widening as Arrow's `sum` does.
func addScalars(_ a: ExprScalar?, _ b: ExprScalar) -> ExprScalar {
    guard let a else { return b }
    switch (a, b) {
    case (.null, _): return b
    case (_, .null): return a
    case (.int(let x), .int(let y)): return .int(x &+ y)
    case (.uint(let x), .uint(let y)): return .uint(x &+ y)
    case (.double(let x), .double(let y)): return .double(x + y)
    default: return .double((a.asDouble ?? 0) + (b.asDouble ?? 0))
    }
}

func minScalars(_ a: ExprScalar?, _ b: ExprScalar) -> ExprScalar {
    guard let a, !a.isNullScalar else { return b }
    if b.isNullScalar { return a }
    switch (a, b) {
    case (.int(let x), .int(let y)): return .int(Swift.min(x, y))
    case (.uint(let x), .uint(let y)): return .uint(Swift.min(x, y))
    default: return (a.asDouble ?? 0) <= (b.asDouble ?? 0) ? a : b
    }
}

func maxScalars(_ a: ExprScalar?, _ b: ExprScalar) -> ExprScalar {
    guard let a, !a.isNullScalar else { return b }
    if b.isNullScalar { return a }
    switch (a, b) {
    case (.int(let x), .int(let y)): return .int(Swift.max(x, y))
    case (.uint(let x), .uint(let y)): return .uint(Swift.max(x, y))
    default: return (a.asDouble ?? 0) >= (b.asDouble ?? 0) ? a : b
    }
}

extension ExprScalar {
    var isNullScalar: Bool { if case .null = self { return true } else { return false } }
}

// MARK: - Aggregate specification

/// One streaming aggregate. `column` is nil only for `count` (which then counts rows).
public struct StreamAggregate: Sendable, Hashable {
    public enum Op: String, Sendable, CaseIterable {
        case sum, count, min, max, mean, variance, stddev
        /// Approximate distinct count via a GPU HyperLogLog sketch merged across batches.
        case countDistinctApprox = "count_distinct_approx"
    }
    public var op: Op
    public var column: String?
    public var name: String

    public init(_ op: Op, _ column: String?, name: String? = nil) {
        self.op = op
        self.column = column
        self.name = name ?? (column.map { "\(op.rawValue)_\($0)" } ?? op.rawValue)
    }
}

// MARK: - Filter / project, streamed to a sink

/// Runs a fused `ExprQuery` with a `project` terminal on every batch and writes the result to a sink.
///
/// The whole predicate and every projected expression compile into **one** Metal kernel per batch
/// (`docs/EXPR.md`), so a 100 GB filter+project reads each column once and materialises no
/// intermediates. Nothing accumulates: memory is one batch in flight plus the sink's own buffer.
public final class FilterProjectOperator: StreamOperator {
    public let query: ExprQuery
    public let sink: StreamSink
    private var rows = 0

    public init(query: ExprQuery, sink: StreamSink) {
        self.query = query
        self.sink = sink
    }

    public func process(_ batch: MetalRecordBatch) throws -> Any? {
        let r = try runExprQuery(query, names: batch.names, columns: batch.columns,
                                 context: batch.firstContext ?? .shared)
        let out = try r.recordBatch()
        // Force the GPU work to finish before this batch leaves the GPU stage: the merge stage writes
        // the bytes and must not race the kernels that produce them.
        _ = out.length
        return out
    }

    /// The sink writes bytes the GPU stage has already flushed; no command buffer.
    public var mergeUsesGPU: Bool { false }

    public func merge(_ partial: Any) throws {
        guard let b = partial as? MetalRecordBatch else { return }
        rows += b.length
        try sink.write(b)
    }

    public func finish() throws -> StreamResult {
        try sink.finish()
        var r = StreamResult()
        r.rowsOut = rows
        if let c = sink as? CollectingSink { r.batch = try c.table() }
        return r
    }
}

// MARK: - Whole-dataset aggregates

/// Streaming `sum` / `count` / `min` / `max` / `mean` / `variance` / `stddev` /
/// `count_distinct_approx` over a whole dataset, with an optional fused filter.
///
/// Every exact aggregate is decomposed into pieces that a batch can produce independently and a merge
/// can combine associatively: `mean` becomes `sum` and `count`, `variance` becomes `sum`, `sum of
/// squares` and `count`. The per-batch pieces come out of the Expr compiler as one fused kernel over
/// the batch, so a filtered aggregate reads the columns once.
public final class StreamAggregateOperator: StreamOperator {
    public let specs: [StreamAggregate]
    public let filter: Expr?
    public let hllPrecision: Int
    /// Delta degrees of freedom for variance / stddev (0 = population, 1 = sample).
    public var ddof = 0

    /// Names of the primitive pieces the Expr query produces, and the running totals.
    private var query: ExprQuery?
    private var totals: [String: ExprScalar] = [:]
    private var sketches: [String: HLLSketch] = [:]

    public init(_ specs: [StreamAggregate], filter: Expr? = nil, hllPrecision: Int = 14) {
        self.specs = specs
        self.filter = filter
        self.hllPrecision = hllPrecision
    }

    /// The primitive aggregates the merge needs, keyed by a private name.
    private func primitives() throws -> [ExprAggregate] {
        var out: [ExprAggregate] = []
        for s in specs {
            switch s.op {
            case .sum:
                out.append(ExprAggregate(.sum, .column(try needColumn(s)), name: "\(s.name)#sum"))
            case .count:
                out.append(ExprAggregate(.count, s.column.map { Expr.column($0) }, name: "\(s.name)#count"))
            case .min:
                out.append(ExprAggregate(.min, .column(try needColumn(s)), name: "\(s.name)#min"))
            case .max:
                out.append(ExprAggregate(.max, .column(try needColumn(s)), name: "\(s.name)#max"))
            case .mean:
                let c = try needColumn(s)
                out.append(ExprAggregate(.sum, .column(c), name: "\(s.name)#sum"))
                out.append(ExprAggregate(.count, .column(c), name: "\(s.name)#count"))
            case .variance, .stddev:
                let c = try needColumn(s)
                let x = Expr.cast(.column(c), .float64)
                out.append(ExprAggregate(.sum, x, name: "\(s.name)#sum"))
                out.append(ExprAggregate(.sum, .binary(.mul, x, x), name: "\(s.name)#sumsq"))
                out.append(ExprAggregate(.count, .column(c), name: "\(s.name)#count"))
            case .countDistinctApprox:
                break       // handled outside the Expr query, by the GPU HLL kernel
            }
        }
        return out
    }

    private func needColumn(_ s: StreamAggregate) throws -> String {
        guard let c = s.column else {
            throw ArrowMetalError.invalidArrowArray("\(s.op.rawValue) needs a column")
        }
        return c
    }

    public func process(_ batch: MetalRecordBatch) throws -> Any? {
        var result: [String: ExprScalar] = [:]
        var batchSketches: [String: HLLSketch] = [:]

        let aggs = try primitives()
        if !aggs.isEmpty {
            if query == nil {
                var b = ExprQueryBuilder()
                if let f = filter { b = b.filter(f) }
                query = b.aggregate(aggs)
            }
            let r = try runExprQuery(query!, names: batch.names, columns: batch.columns,
                                     context: batch.firstContext ?? .shared)
            for (i, n) in r.scalarNames.enumerated() { result[n] = r.scalars[i] }
        }
        for s in specs where s.op == .countDistinctApprox {
            let name = try needColumn(s)
            guard var col = batch[name] else {
                throw ArrowMetalError.invalidArrowArray("no column named \(name)")
            }
            if let f = filter {
                let fr = try streamFilterProject(batch, filter: f, projections: [("v", .column(name))],
                                                 context: batch.firstContext ?? .shared)
                guard let c = fr["v"] else { throw ArrowMetalError.invalidArrowArray("filter produced no column") }
                col = c
            }
            batchSketches[s.name] = try gpuHyperLogLog(col, precision: hllPrecision)
        }
        return (result, batchSketches)
    }

    public func merge(_ partial: Any) throws {
        guard let (scalars, batchSketches) = partial as? ([String: ExprScalar], [String: HLLSketch]) else { return }
        for (k, v) in scalars {
            if k.hasSuffix("#min") { totals[k] = minScalars(totals[k], v) }
            else if k.hasSuffix("#max") { totals[k] = maxScalars(totals[k], v) }
            else { totals[k] = addScalars(totals[k], v) }
        }
        for (k, s) in batchSketches {
            if var existing = sketches[k] { existing.merge(s); sketches[k] = existing }
            else { sketches[k] = s }
        }
    }

    public func finish() throws -> StreamResult {
        var r = StreamResult()
        for s in specs {
            r.scalarNames.append(s.name)
            switch s.op {
            case .sum: r.scalars.append(totals["\(s.name)#sum"] ?? .null)
            case .count: r.scalars.append(totals["\(s.name)#count"] ?? .int(0))
            case .min: r.scalars.append(totals["\(s.name)#min"] ?? .null)
            case .max: r.scalars.append(totals["\(s.name)#max"] ?? .null)
            case .mean:
                let n = totals["\(s.name)#count"]?.asInt64 ?? 0
                let sum = totals["\(s.name)#sum"]?.asDouble
                r.scalars.append(n > 0 && sum != nil ? .double(sum! / Double(n)) : .null)
            case .variance, .stddev:
                let n = Double(totals["\(s.name)#count"]?.asInt64 ?? 0)
                let sum = totals["\(s.name)#sum"]?.asDouble ?? 0
                let sq = totals["\(s.name)#sumsq"]?.asDouble ?? 0
                let denom = n - Double(ddof)
                if n <= 0 || denom <= 0 { r.scalars.append(.null) }
                else {
                    let v = Swift.max(0, (sq - sum * sum / n) / denom)
                    r.scalars.append(.double(s.op == .variance ? v : v.squareRoot()))
                }
            case .countDistinctApprox:
                r.scalars.append(.int(Int64(sketches[s.name]?.count ?? 0)))
            }
        }
        return r
    }

    /// Scalars and a register array: pure host arithmetic, no command buffer needed.
    public var mergeUsesGPU: Bool { false }

    /// The merged sketch of a `count_distinct_approx` aggregate, for callers that want the error bound.
    public func sketch(_ name: String) -> HLLSketch? { sketches[name] }
}

// MARK: - Streaming group-by

/// A group's key, as host values. `Hashable` so the global table is a plain dictionary.
enum StreamGroupKey: Hashable {
    /// The overwhelmingly common shape. Kept out of an array so a lookup allocates nothing: the
    /// 10-million-key merge probes this dictionary once per group per batch, hundreds of millions of
    /// times, and an `[StreamValue]` box per probe dominated everything else.
    case single(StreamValue)
    case multi([StreamValue])
}

/// A running accumulator for one aggregate of one group.
struct StreamAccumulator {
    var sumInt: Int64 = 0
    var sumUInt: UInt64 = 0
    var sumDouble: Double = 0
    var sumSq: Double = 0
    var count: Int64 = 0
    var minV: StreamValue = .null
    var maxV: StreamValue = .null
    var kind: Int = 0     // 0 none, 1 int, 2 uint, 3 double
}

/// Streaming group-by with a global aggregate table that survives across batches.
///
/// Two states, chosen by the shape of the key:
///
/// * **Dense integer keys** (`denseKeyCount` given, one integer key column in `[0, K)`): the global
///   table lives on the **GPU** as one accumulator array per aggregate, `K` entries wide. Each batch
///   runs `GroupBy` over the batch's keys and the per-batch result is folded into the global arrays
///   with element-wise `add` / `min` / `max` kernels — no host round trip at all. When
///   `K * accumulators * 8` exceeds `gpuStateBudgetBytes` the state **spills to the host table**
///   below, which is what keeps a 10-million-key group-by inside a fixed GPU budget.
///
/// * **Arbitrary keys** (any type, any number of columns): each batch runs `GroupByKeys` (which turns
///   arbitrary key columns into dense ids on the GPU) plus the aggregates, and hands the merge stage
///   one row per group — typically thousands of rows for a batch of millions. The merge folds those
///   rows into a host dictionary keyed by the key value itself. Exact, order independent, and its
///   size is the number of *distinct groups*, not the number of rows.
public final class StreamGroupByOperator: StreamOperator {
    public let keyColumns: [String]
    public let aggregates: [StreamAggregate]
    public let filter: Expr?
    /// When set, the keys are integers already in `[0, denseKeyCount)` and the GPU state is used.
    public let denseKeyCount: Int?
    /// Above this many bytes of GPU accumulator state, the dense path spills to the host table.
    public var gpuStateBudgetBytes = 512 << 20
    public var ddof = 0

    /// One partition of the host table. Keys are assigned to a shard by a hash, so the shards hold
    /// disjoint groups and can be merged in parallel with no locking at all.
    ///
    /// Inside a shard the table is a struct of arrays: `intSlot` / `slot` map a key to a group index,
    /// `keyCols` holds the key values column-wise, and `accs` is a flat `groups * aggregates.count`
    /// array of accumulators mutated in place. The obvious `[Key: [Accumulator]]` allocated one array
    /// per group *per batch*; this allocates none.
    final class GroupShard {
        /// The specialised table for a single integer key (the common shape, and the big one).
        var intSlot: [Int64: Int] = [:]
        var slot: [StreamGroupKey: Int] = [:]
        var nullKeyGroup: Int?
        var keyCols: [[StreamValue]] = []
        var accs: [StreamAccumulator] = []
        var count: Int { keyCols.first?.count ?? 0 }
    }

    /// Host table (arbitrary keys, or a dense state that spilled), sharded by key hash.
    ///
    /// A ten-million-group merge is memory-latency bound: nearly every probe of a 10M-entry table is
    /// a DRAM round trip, and one thread can only keep so many in flight. Splitting the table by key
    /// hash lets `mergeShards` threads probe at once, each in arrays no other thread touches.
    private var shards: [GroupShard] = []
    /// Threads the merge stage uses. 1 keeps it strictly single threaded.
    public var mergeShards = 8
    private var keyTemplates: [AnyMetalArray]?
    /// GPU-resident dense state, one array per aggregate.
    private var denseSum: [AnyMetalArray?] = []
    private var denseCount: [MetalArray<Int64>?] = []
    private var denseMin: [AnyMetalArray?] = []
    private var denseMax: [AnyMetalArray?] = []
    private var spilled = false
    private var context: MetalContext = .shared

    /// The GPU-resident table for the arbitrary-key path. Set false to force the host table (the A/B
    /// the streaming tests use to prove the two agree row for row).
    public var residentTable = true
    private var resident: StreamGroupTable?
    private var residentKeyTemplate: AnyMetalArray?

    public init(keys: [String], aggregates: [StreamAggregate], filter: Expr? = nil, denseKeyCount: Int? = nil) {
        self.keyColumns = keys
        self.aggregates = aggregates
        self.filter = filter
        self.denseKeyCount = keys.count == 1 ? denseKeyCount : nil
    }

    /// True while the dense global table is on the GPU (it turns false after a spill).
    public var usesGPUState: Bool { denseKeyCount != nil && !spilled }

    /// True when the arbitrary-key path keeps its global table on the GPU: one integer key column, no
    /// dense key count, and every aggregate expressible as a running (sum, count) pair.
    ///
    /// `min` / `max` would need a per-slot atomic minimum the table does not have, and `variance` a
    /// third accumulator; both keep the host table, which is exact for every key type and every
    /// aggregate and is what the resident path is checked against.
    public var usesResidentTable: Bool {
        residentTable && denseKeyCount == nil && keyColumns.count == 1
            && aggregates.allSatisfy { $0.op == .sum || $0.op == .count || $0.op == .mean }
    }

    /// Whether a sparse key may take the **row-level** resident path — one thread per row into the
    /// global table, no per-batch dense encoding at all. Set false to force the per-batch encoding
    /// (the A/B the tests use to prove the two agree bit for bit).
    public var residentRowLevel = true
    /// Takes the row-level path even for a key the per-batch encoding handles well. Only the tests
    /// set it: it is how the two paths are compared at the cardinalities the chooser would never
    /// send down the row path.
    public var residentRowLevelForced = false
    /// Set false to send an integer-only query down the row path's dense-id branch instead of its
    /// atomic one — the third leg of the same differential test.
    public var residentRowLevelAtomic = true

    /// Both GPU-state paths fold with kernels; the host table's merge is pure host arithmetic over
    /// values `process` already read back.
    public var mergeUsesGPU: Bool { usesGPUState || resident != nil || usesResidentTable }

    /// Whether this batch's key column is cheap to turn into dense ids on its own — an integer
    /// column whose values span a small enough range for `GroupByKeys`' scan (no sort at all).
    ///
    /// That is exactly the shape the per-batch path is good at: a thousand groups over a million rows
    /// become a thousand accumulators in threadgroup memory, where the row-level path would instead
    /// send a million atomic adds at a thousand slots. The sparse key is the other way round — a
    /// million distinct values per batch, one row each — and that is the one the row path takes.
    private func keyIsCheaplyDense(_ c: AnyMetalArray) throws -> Bool {
        guard let r = try GroupByKeys.integerRange(c) else { return false }
        let span = r.hi >= r.lo ? Int(truncatingIfNeeded: r.hi &- r.lo) &+ 1 : 0
        return span > 0 && GroupByKeys.rangeIsWorthIt(span: span, rows: c.length)
    }

    /// Whether every aggregate can be folded by the row-level **atomic** accumulate: counts and
    /// integer sums, which two 32-bit atomic adds with a carry compute exactly and in any order.
    /// A float64 sum cannot — Metal has no 64-bit atomic and no emulation of one rounds a binary64
    /// addition correctly — so a query with one takes the dense-id row path instead.
    private func rowAtomicEligible(_ columns: [AnyMetalArray?]) -> Bool {
        for (a, c) in zip(aggregates, columns) {
            if a.op == .count && a.column == nil { continue }
            guard let c else { return false }
            switch a.op {
            // `count(col)` needs the column's validity, which the accumulate reads out of the widened
            // payload — unless the column has no nulls at all, when counting rows is counting values.
            case .count: if c.nullCount != 0 && !canWiden64(c) { return false }
            case .sum, .mean: if !canWiden64(c) { return false }
            default: return false
            }
        }
        return true
    }

    /// Whether `widen64` has a 64-bit integer payload for this column type. A type test only: it must
    /// not record a cast, because it runs on every batch to choose a path.
    private func canWiden64(_ c: AnyMetalArray) -> Bool {
        switch c {
        case .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64, .temporal: return true
        default: return false
        }
    }

    /// An integer or temporal value column as the 64-bit payload the row-level accumulate adds, in
    /// the width Arrow's `sum` produces; nil for anything a 64-bit integer cannot hold exactly.
    private func widen64(_ c: AnyMetalArray) -> AnyMetalArray? {
        switch c {
        case .int64: return c
        case .uint64: return c
        case .int8(let a): return (try? a.cast(to: Int64.self)).map { .int64($0) }
        case .int16(let a): return (try? a.cast(to: Int64.self)).map { .int64($0) }
        case .int32(let a): return (try? a.cast(to: Int64.self)).map { .int64($0) }
        case .uint8(let a): return (try? a.cast(to: Int64.self)).map { .int64($0) }
        case .uint16(let a): return (try? a.cast(to: Int64.self)).map { .int64($0) }
        case .uint32(let a): return (try? a.cast(to: Int64.self)).map { .int64($0) }
        case .temporal(let t): return (try? t.int64Values()).map { .int64($0) }
        default: return nil
        }
    }

    public func process(_ batch: MetalRecordBatch) throws -> Any? {
        context = batch.firstContext ?? .shared
        var work = batch
        if let f = filter {
            var proj: [(String, Expr)] = keyColumns.map { ($0, .column($0)) }
            for a in aggregates where a.column != nil {
                if !proj.contains(where: { $0.0 == a.column! }) { proj.append((a.column!, .column(a.column!))) }
            }
            work = try streamFilterProject(batch, filter: f, projections: proj, context: context)
        }
        guard work.length > 0 else { return nil }

        var keyCols: [AnyMetalArray] = []
        for k in keyColumns {
            guard let c = work[k] else { throw ArrowMetalError.invalidArrowArray("no key column named \(k)") }
            keyCols.append(c)
        }

        if usesGPUState, let K = denseKeyCount, let idx = try denseIndex(keyCols[0]) {
            // Dense path: the GPU keeps the global table; nothing crosses to the host per batch.
            let gb = try GroupBy(keys: idx, keyCount: K)
            var partials: [(AnyMetalArray?, MetalArray<Int64>?, AnyMetalArray?, AnyMetalArray?)] = []
            for a in aggregates {
                partials.append(try densePartial(a, gb, work))
            }
            return DensePartial(parts: partials)
        }

        // Arbitrary path with an integer key: the global table stays on the GPU, so this batch's one
        // row per group never crosses to the host at all.
        if usesResidentTable, integerKeyWidth(keyCols[0]) != nil {
            // Sparse, high-cardinality key: hand the merge the batch's **rows** and let the resident
            // table do the encoding itself (§4.1). Nothing here sorts or groups anything.
            if try residentRowLevel && (residentRowLevelForced || !keyIsCheaplyDense(keyCols[0])) {
                if residentKeyTemplate == nil { residentKeyTemplate = keyCols[0] }
                var cols: [AnyMetalArray?] = []
                for a in aggregates { cols.append(a.column.flatMap { work[$0] }) }
                return ResidentRowPartial(keys: try int64Keys(keyCols[0]), columns: cols,
                                          atomic: residentRowLevelAtomic && rowAtomicEligible(cols))
            }
            let gk = try GroupByKeys(columns: keyCols)
            guard gk.groupCount > 0 else { return nil }
            if residentKeyTemplate == nil { residentKeyTemplate = keyCols[0] }
            let groupKeys = try int64Keys(try gk.trim(try gk.groupKeys()[0]))
            var vals: [AnyMetalArray?] = [], cnts: [MetalArray<Int64>?] = []
            var kinds: [StreamGroupTable.SumKind] = []
            for a in aggregates {
                let (v, c, _) = try hostPartial(a, gk, work)
                vals.append(v)
                cnts.append(c.flatMap { if case .int64(let x) = $0 { return x } else { return nil } })
                kinds.append(sumKind(v))
            }
            return ResidentPartial(keys: groupKeys, values: vals, counts: cnts, kinds: kinds)
        }

        // Arbitrary path: dense ids on the GPU, one row per group to the host.
        let gk = try GroupByKeys(columns: keyCols)
        guard gk.groupCount > 0 else { return nil }
        if keyTemplates == nil { keyTemplates = keyCols }
        let keyCandidates = try gk.groupKeys().map { try gk.trim($0) }
        var aggCols: [(AnyMetalArray?, AnyMetalArray?, AnyMetalArray?)] = []
        for a in aggregates { aggCols.append(try hostPartial(a, gk, work)) }
        // Every kernel above is recorded into the open batch; run it before any value is read back.
        try context.syncPoint()
        let keyValues = try keyCandidates.map { try $0.streamColumn() }
        var aggValues: [StreamColumn] = []
        var aggCounts: [StreamColumn] = []
        var aggSquares: [StreamColumn] = []
        for (v, c, sq) in aggCols {
            aggValues.append(v == nil ? .empty : try v!.streamColumn())
            aggCounts.append(c == nil ? .empty : try c!.streamColumn())
            aggSquares.append(sq == nil ? .empty : try sq!.streamColumn())
        }
        return HostPartial(groupCount: gk.groupCount, keys: keyValues, values: aggValues,
                           counts: aggCounts, squares: aggSquares)
    }

    /// The int32 dense ids of a key column that is already dense.
    private func denseIndex(_ c: AnyMetalArray) throws -> MetalArray<Int32>? {
        switch c {
        case .int32(let a): return a
        case .int64(let a): return try a.cast(to: Int32.self)
        case .int16(let a): return try a.cast(to: Int32.self)
        case .int8(let a): return try a.cast(to: Int32.self)
        case .uint32(let a): return try a.cast(to: Int32.self)
        case .uint16(let a): return try a.cast(to: Int32.self)
        case .uint8(let a): return try a.cast(to: Int32.self)
        default: return nil
        }
    }

    final class DensePartial {
        let parts: [(AnyMetalArray?, MetalArray<Int64>?, AnyMetalArray?, AnyMetalArray?)]
        init(parts: [(AnyMetalArray?, MetalArray<Int64>?, AnyMetalArray?, AnyMetalArray?)]) { self.parts = parts }
    }
    struct HostPartial {
        let groupCount: Int
        let keys: [StreamColumn]
        let values: [StreamColumn]
        let counts: [StreamColumn]
        let squares: [StreamColumn]
    }

    /// One batch's dense-key aggregate arrays, still on the GPU.
    private func densePartial(_ a: StreamAggregate, _ gb: GroupBy<Int32>, _ batch: MetalRecordBatch)
        throws -> (AnyMetalArray?, MetalArray<Int64>?, AnyMetalArray?, AnyMetalArray?) {
        switch a.op {
        case .count where a.column == nil:
            return (nil, try gb.count(), nil, nil)
        case .count:
            let col = try column(a, batch)
            return (nil, try countValid(col, gb), nil, nil)
        case .sum, .mean:
            let col = try column(a, batch)
            return (try groupSum(col, gb), try countValid(col, gb), nil, nil)
        case .min:
            let col = try column(a, batch)
            return (nil, nil, try groupMinMax(col, gb, isMin: true), nil)
        case .max:
            let col = try column(a, batch)
            return (nil, nil, nil, try groupMinMax(col, gb, isMin: false))
        case .variance, .stddev, .countDistinctApprox:
            throw ArrowMetalError.unsupportedType("\(a.op.rawValue) is not available in a dense-key streaming group-by")
        }
    }

    /// One batch's arbitrary-key aggregate columns, one row per group, still on the GPU.
    private func hostPartial(_ a: StreamAggregate, _ gk: GroupByKeys, _ batch: MetalRecordBatch)
        throws -> (AnyMetalArray?, AnyMetalArray?, AnyMetalArray?) {
        let gb = gk.groupBy
        switch a.op {
        case .count where a.column == nil:
            return (nil, try gk.trim(.int64(try gb.count())), nil)
        case .count:
            let col = try column(a, batch)
            return (nil, try gk.trim(.int64(try countValid(col, gb))), nil)
        case .sum, .mean:
            let col = try column(a, batch)
            return (try gk.trim(try groupSum(col, gb)), try gk.trim(.int64(try countValid(col, gb))), nil)
        case .min:
            let col = try column(a, batch)
            return (try gk.trim(try groupMinMax(col, gb, isMin: true)), nil, nil)
        case .max:
            let col = try column(a, batch)
            return (try gk.trim(try groupMinMax(col, gb, isMin: false)), nil, nil)
        case .variance, .stddev:
            let col = try column(a, batch)
            let dbl = try toDouble(col)
            let sq = try dbl.multiply(dbl)
            return (try gk.trim(.float64(try gb.sumDouble(dbl))),
                    try gk.trim(.int64(try countValid(col, gb))),
                    try gk.trim(.float64(try gb.sumDouble(sq))))
        case .countDistinctApprox:
            throw ArrowMetalError.unsupportedType("count_distinct_approx is a whole-dataset aggregate, not a group-by one")
        }
    }

    private func column(_ a: StreamAggregate, _ batch: MetalRecordBatch) throws -> AnyMetalArray {
        guard let n = a.column, let c = batch[n] else {
            throw ArrowMetalError.invalidArrowArray("\(a.op.rawValue) needs an existing column")
        }
        return c
    }

    public func merge(_ partial: Any) throws {
        if let r = partial as? ResidentPartial {
            try residentTableForMerge().fold(groupKeys: r.keys, values: r.values, groupCounts: r.counts,
                                             kinds: r.kinds)
            return
        }
        if let r = partial as? ResidentRowPartial { try mergeRows(r); return }
        if let d = partial as? DensePartial { try mergeDense(d); return }
        guard let h = partial as? HostPartial else { return }
        try mergeHost(h)
    }

    private func residentTableForMerge() throws -> StreamGroupTable {
        if let r = resident { return r }
        let r = try StreamGroupTable(context: context, aggregateCount: aggregates.count)
        resident = r
        return r
    }

    /// One batch's **rows**, on their way into the resident table: the table does the key encoding
    /// itself, so nothing here has been sorted or grouped.
    final class ResidentRowPartial {
        let keys: MetalArray<Int64>
        /// Per aggregate, the value column as it came off the batch (nil for `count(*)`).
        let columns: [AnyMetalArray?]
        /// True when every aggregate is a count or an integer sum, which the atomic accumulate does
        /// straight from the rows; false sends the batch through the dense-id pass instead.
        let atomic: Bool
        init(keys: MetalArray<Int64>, columns: [AnyMetalArray?], atomic: Bool) {
            self.keys = keys; self.columns = columns; self.atomic = atomic
        }
    }

    /// Folds one batch of rows into the resident table.
    ///
    /// * **atomic** — one thread per row inserts its key, one thread per row folds itself into that
    ///   slot with 64-bit atomic adds built out of two 32-bit ones. Counts and integer sums only.
    /// * **dense** — the insert pass also stamps a batch-local dense id onto every slot it touched,
    ///   so the batch's aggregates run on the ordinary `GroupBy` and the per-group results fold into
    ///   distinct slots with no atomics. That is the path a float64 sum takes, and it still never
    ///   sorts the key column: the resident table is the dictionary.
    private func mergeRows(_ r: ResidentRowPartial) throws {
        guard r.keys.length > 0 else { return }
        let table = try residentTableForMerge()
        if r.atomic {
            var rows: [StreamGroupTable.RowAggregate] = []
            for (a, c) in zip(aggregates, r.columns) {
                var spec = StreamGroupTable.RowAggregate()
                switch a.op {
                case .count where a.column == nil:
                    spec.count = .allRows
                case .count:
                    // `rowAtomicEligible` let this through either because the column widens (so its
                    // validity comes along) or because it has no nulls, when every row is a value.
                    if let w = widen64(c!) { spec.count = .nonNullValues; spec.column = w }
                    else { spec.count = .allRows }
                case .sum, .mean:
                    spec.count = .nonNullValues
                    let w = widen64(c!)
                    spec.column = w
                    spec.sums = true
                    spec.kind = sumKind(w)
                default:
                    throw ArrowMetalError.unsupportedType("\(a.op.rawValue) in a row-level streaming group-by")
                }
                rows.append(spec)
            }
            try table.foldRows(keys: r.keys, aggregates: rows)
            return
        }

        let (ids, groupCount, slotOfDense) = try table.denseIdsForRows(keys: r.keys)
        guard groupCount > 0 else { return }
        let gb = try GroupBy(keys: ids, keyCount: groupCount)
        var vals: [AnyMetalArray?] = [], cnts: [MetalArray<Int64>?] = []
        var kinds: [StreamGroupTable.SumKind] = []
        for (a, c) in zip(aggregates, r.columns) {
            switch a.op {
            case .count where a.column == nil:
                vals.append(nil); cnts.append(try gb.count()); kinds.append(.none)
            case .count:
                vals.append(nil); cnts.append(try countValid(c!, gb)); kinds.append(.none)
            case .sum, .mean:
                // A float64 sum is the reason this branch exists, and with a million groups of one row
                // the ordinary segmented reduction spends a whole threadgroup on each of them.
                if case .float64(let d) = c!, let r = try residentSegmentedSumDouble(d, gb) {
                    vals.append(.float64(r.sum)); cnts.append(r.count); kinds.append(.double)
                } else {
                    let s = try groupSum(c!, gb)
                    vals.append(s); cnts.append(try countValid(c!, gb)); kinds.append(sumKind(s))
                }
            default:
                throw ArrowMetalError.unsupportedType("\(a.op.rawValue) in a row-level streaming group-by")
            }
        }
        try table.fold(slotOfDense: slotOfDense, groupCount: groupCount, values: vals,
                       groupCounts: cnts, kinds: kinds)
    }

    /// One batch's per-group results, still on the GPU, on their way into the resident table.
    final class ResidentPartial {
        let keys: MetalArray<Int64>
        let values: [AnyMetalArray?]
        let counts: [MetalArray<Int64>?]
        let kinds: [StreamGroupTable.SumKind]
        init(keys: MetalArray<Int64>, values: [AnyMetalArray?], counts: [MetalArray<Int64>?],
             kinds: [StreamGroupTable.SumKind]) {
            self.keys = keys; self.values = values; self.counts = counts; self.kinds = kinds
        }
    }

    private func sumKind(_ v: AnyMetalArray?) -> StreamGroupTable.SumKind {
        switch v {
        case .some(.int64): return .int
        case .some(.uint64): return .uint
        case .some(.float64): return .double
        default: return .none
        }
    }

    /// The byte width of an integer key column, or nil when the key is not a plain integer.
    private func integerKeyWidth(_ c: AnyMetalArray) -> Int? {
        switch c {
        case .int8, .uint8: return 1
        case .int16, .uint16: return 2
        case .int32, .uint32: return 4
        case .int64, .uint64: return 8
        default: return nil
        }
    }

    /// An integer key column as int64. Unsigned keys are reinterpreted rather than converted, which
    /// stays injective for the whole uint64 range; `finish` reverses it.
    private func int64Keys(_ c: AnyMetalArray) throws -> MetalArray<Int64> {
        switch c {
        case .int8(let a): return try a.cast(to: Int64.self)
        case .int16(let a): return try a.cast(to: Int64.self)
        case .int32(let a): return try a.cast(to: Int64.self)
        case .int64(let a): return a
        case .uint8(let a): return try a.cast(to: Int64.self)
        case .uint16(let a): return try a.cast(to: Int64.self)
        case .uint32(let a): return try a.cast(to: Int64.self)
        case .uint64(let a):
            return MetalArray<Int64>(length: a.length, nullCount: a.nullCount, validity: a.validity,
                                     values: a.values, context: a.context)
        default: throw ArrowMetalError.unsupportedType("resident group table needs an integer key")
        }
    }

    /// The reverse of `int64Keys`, back to the key column's own type.
    private func keyColumnLike(_ template: AnyMetalArray, _ k: MetalArray<Int64>) throws -> AnyMetalArray {
        switch template {
        case .int8: return .int8(try k.cast(to: Int8.self))
        case .int16: return .int16(try k.cast(to: Int16.self))
        case .int32: return .int32(try k.cast(to: Int32.self))
        case .int64: return .int64(k)
        case .uint8: return .uint8(try k.cast(to: UInt8.self))
        case .uint16: return .uint16(try k.cast(to: UInt16.self))
        case .uint32: return .uint32(try k.cast(to: UInt32.self))
        case .uint64:
            return .uint64(MetalArray<UInt64>(length: k.length, nullCount: k.nullCount,
                                              validity: k.validity, values: k.values, context: k.context))
        default: return .int64(k)
        }
    }

    private func mergeDense(_ d: DensePartial) throws {
        // A partial recorded before the spill can still arrive after it; fold it into the host table.
        if spilled { try foldDensePartialIntoHostTable(d); return }
        if denseSum.isEmpty {
            denseSum = Array(repeating: nil, count: aggregates.count)
            denseCount = Array(repeating: nil, count: aggregates.count)
            denseMin = Array(repeating: nil, count: aggregates.count)
            denseMax = Array(repeating: nil, count: aggregates.count)
        }
        for (i, part) in d.parts.enumerated() {
            let (s, c, mn, mx) = part
            if let s { denseSum[i] = try elementwiseAdd(denseSum[i], s) }
            if let c { denseCount[i] = denseCount[i] == nil ? c : try denseCount[i]!.add(c) }
            if let mn { denseMin[i] = try elementwiseMinMax(denseMin[i], mn, isMin: true) }
            if let mx { denseMax[i] = try elementwiseMinMax(denseMax[i], mx, isMin: false) }
        }
        // Spill check: the state is K entries per aggregate; if it outgrew the budget, move it to the
        // host table and continue there. (In practice K is fixed, so this fires on the first batch.)
        let bytes = (denseKeyCount ?? 0) * aggregates.count * 8 * 2
        if bytes > gpuStateBudgetBytes && !spilled { try spillDenseToHost() }
    }

    /// Creates the shards on first use.
    private func ensureShards() {
        guard shards.isEmpty else { return }
        let n = Swift.max(1, mergeShards)
        shards = (0..<n).map { _ in
            let s = GroupShard()
            s.keyCols = Array(repeating: [], count: Swift.max(keyColumns.count, 1))
            return s
        }
    }

    /// Which shard owns an integer key. Fibonacci hashing on the top bits, so consecutive ids spread.
    @inline(__always) private func shardOf(_ k: Int64) -> Int {
        guard shards.count > 1 else { return 0 }
        let h = UInt64(bitPattern: k) &* 0x9E37_79B9_7F4A_7C15
        return Int(h >> 58) % shards.count
    }
    @inline(__always) private func shardOf(_ key: StreamGroupKey) -> Int {
        shards.count > 1 ? Int(UInt(bitPattern: key.hashValue) % UInt(shards.count)) : 0
    }

    /// Index of `key` inside `shard`, inserting a new group (with the key values in `row`) when new.
    private func groupIndex(_ shard: GroupShard, _ key: StreamGroupKey, _ row: [StreamValue]) -> Int {
        if let i = shard.slot[key] { return i }
        let i = shard.count
        shard.slot[key] = i
        for j in 0..<shard.keyCols.count { shard.keyCols[j].append(j < row.count ? row[j] : .null) }
        shard.accs.append(contentsOf: repeatElement(StreamAccumulator(), count: aggregates.count))
        return i
    }

    private func mergeHost(_ h: HostPartial) throws {
        ensureShards()
        let m = aggregates.count
        let nKeys = keyColumns.count
        guard nKeys == 1, let ks = h.keys[0].asInts else { return try mergeHostSlow(h) }
        var keyValid: [Bool]? = nil
        if case .ints(_, let v) = h.keys[0] { keyValid = v }

        // Pass one, sequential and cache friendly: bucket this batch's groups by the shard that owns
        // their key. Only a hash per group, no table probe — the probes are what has to be spread.
        var buckets = [[Int32]](repeating: [], count: shards.count)
        for b in 0..<buckets.count { buckets[b].reserveCapacity(h.groupCount / buckets.count + 16) }
        for g in 0..<h.groupCount {
            if let v = keyValid, g < v.count, !v[g] { buckets[0].append(Int32(g)); continue }
            buckets[shardOf(ks[g])].append(Int32(g))
        }

        // Pass two, one thread per shard: each walks only its own groups, in its own arrays, so the
        // random probes into a table too large for cache happen `shards` at a time. A ten-million-key
        // merge is latency bound, and this is the only thing that moves it.
        let work = { [self] (sid: Int) in
            let shard = shards[sid]
            let bucket = buckets[sid]
            guard !bucket.isEmpty else { return }
            var bases = [Int](repeating: 0, count: bucket.count)
            for (idx, g) in bucket.enumerated() {
                let gi = Int(g)
                if let v = keyValid, gi < v.count, !v[gi] { bases[idx] = groupIndexNullKey(shard) * m }
                else { bases[idx] = groupIndexInt(shard, ks[gi]) * m }
            }
            for (i, a) in aggregates.enumerated() {
                let counts = h.counts[i].asInts
                switch a.op {
                case .count:
                    if let c = counts {
                        for (idx, g) in bucket.enumerated() where Int(g) < c.count {
                            shard.accs[bases[idx] + i].count += c[Int(g)]
                        }
                    }
                case .sum, .mean:
                    if let c = counts {
                        for (idx, g) in bucket.enumerated() where Int(g) < c.count {
                            shard.accs[bases[idx] + i].count += c[Int(g)]
                        }
                    }
                    accumulate(h.values[i], shard: shard, bucket: bucket, bases: bases, slot: i)
                case .min:
                    let values = h.values[i]
                    for (idx, g) in bucket.enumerated() where values.isValid(Int(g)) {
                        let a = shard.accs[bases[idx] + i].minV
                        shard.accs[bases[idx] + i].minV = minStreamValue(a, values.value(Int(g)))
                    }
                case .max:
                    let values = h.values[i]
                    for (idx, g) in bucket.enumerated() where values.isValid(Int(g)) {
                        let a = shard.accs[bases[idx] + i].maxV
                        shard.accs[bases[idx] + i].maxV = maxStreamValue(a, values.value(Int(g)))
                    }
                case .variance, .stddev:
                    if let c = counts {
                        for (idx, g) in bucket.enumerated() where Int(g) < c.count {
                            shard.accs[bases[idx] + i].count += c[Int(g)]
                        }
                    }
                    if case .doubles(let v, let valid) = h.values[i] {
                        for (idx, g) in bucket.enumerated() where Int(g) < v.count && (valid?[Int(g)] ?? true) {
                            shard.accs[bases[idx] + i].sumDouble += v[Int(g)]
                            shard.accs[bases[idx] + i].kind = 3
                        }
                    }
                    if case .doubles(let q, let valid) = h.squares[i] {
                        for (idx, g) in bucket.enumerated() where Int(g) < q.count && (valid?[Int(g)] ?? true) {
                            shard.accs[bases[idx] + i].sumSq += q[Int(g)]
                        }
                    }
                case .countDistinctApprox:
                    break
                }
            }
        }
        if shards.count == 1 {
            work(0)
        } else {
            DispatchQueue.concurrentPerform(iterations: shards.count, execute: work)
        }
    }

    /// The general path: keys of any type or several of them. Low cardinality in practice, so it
    /// stays on one thread and one shard, which keeps the key-to-shard mapping trivially consistent.
    private func mergeHostSlow(_ h: HostPartial) throws {
        ensureShards()
        let shard = shards[0]
        let m = aggregates.count
        let nKeys = keyColumns.count
        var row = [StreamValue](repeating: .null, count: nKeys)
        var bases = [Int](repeating: 0, count: h.groupCount)
        for g in 0..<h.groupCount {
            for j in 0..<nKeys { row[j] = h.keys[j].value(g) }
            let key: StreamGroupKey = nKeys == 1 ? .single(row[0]) : .multi(row)
            bases[g] = groupIndex(shard, key, row) * m
        }
        for (i, a) in aggregates.enumerated() {
            let counts = h.counts[i].asInts
            let values = h.values[i]
            switch a.op {
            case .count:
                if let c = counts { for g in 0..<Swift.min(h.groupCount, c.count) { shard.accs[bases[g] + i].count += c[g] } }
            case .sum, .mean:
                if let c = counts { for g in 0..<Swift.min(h.groupCount, c.count) { shard.accs[bases[g] + i].count += c[g] } }
                for g in 0..<h.groupCount where values.isValid(g) { addInto(&shard.accs[bases[g] + i], values.value(g)) }
            case .min:
                for g in 0..<h.groupCount where values.isValid(g) {
                    shard.accs[bases[g] + i].minV = minStreamValue(shard.accs[bases[g] + i].minV, values.value(g))
                }
            case .max:
                for g in 0..<h.groupCount where values.isValid(g) {
                    shard.accs[bases[g] + i].maxV = maxStreamValue(shard.accs[bases[g] + i].maxV, values.value(g))
                }
            case .variance, .stddev:
                if let c = counts { for g in 0..<Swift.min(h.groupCount, c.count) { shard.accs[bases[g] + i].count += c[g] } }
                if case .doubles(let v, let valid) = values {
                    for g in 0..<Swift.min(h.groupCount, v.count) where valid?[g] ?? true {
                        shard.accs[bases[g] + i].sumDouble += v[g]
                        shard.accs[bases[g] + i].kind = 3
                    }
                }
                if case .doubles(let q, let valid) = h.squares[i] {
                    for g in 0..<Swift.min(h.groupCount, q.count) where valid?[g] ?? true {
                        shard.accs[bases[g] + i].sumSq += q[g]
                    }
                }
            case .countDistinctApprox:
                break
            }
        }
    }

    /// Adds one aggregate column into a shard's accumulators, switching on its storage once.
    private func accumulate(_ column: StreamColumn, shard: GroupShard, bucket: [Int32],
                            bases: [Int], slot i: Int) {
        switch column {
        case .ints(let v, let valid):
            for (idx, g) in bucket.enumerated() where Int(g) < v.count && (valid?[Int(g)] ?? true) {
                shard.accs[bases[idx] + i].sumInt &+= v[Int(g)]
                shard.accs[bases[idx] + i].kind = Swift.max(shard.accs[bases[idx] + i].kind, 1)
            }
        case .uints(let v, let valid):
            for (idx, g) in bucket.enumerated() where Int(g) < v.count && (valid?[Int(g)] ?? true) {
                shard.accs[bases[idx] + i].sumUInt &+= v[Int(g)]
                shard.accs[bases[idx] + i].kind = Swift.max(shard.accs[bases[idx] + i].kind, 2)
            }
        case .doubles(let v, let valid):
            for (idx, g) in bucket.enumerated() where Int(g) < v.count && (valid?[Int(g)] ?? true) {
                shard.accs[bases[idx] + i].sumDouble += v[Int(g)]
                shard.accs[bases[idx] + i].kind = 3
            }
        case .other(let v):
            for (idx, g) in bucket.enumerated() where Int(g) < v.count {
                addInto(&shard.accs[bases[idx] + i], v[Int(g)])
            }
        case .empty:
            break
        }
    }

    /// Group index for an integer key, through the shard's specialised table.
    @inline(__always) private func groupIndexInt(_ shard: GroupShard, _ k: Int64) -> Int {
        if let i = shard.intSlot[k] { return i }
        let i = shard.count
        shard.intSlot[k] = i
        shard.keyCols[0].append(.int(k))
        for j in 1..<shard.keyCols.count { shard.keyCols[j].append(.null) }
        shard.accs.append(contentsOf: repeatElement(StreamAccumulator(), count: aggregates.count))
        return i
    }

    /// Group index of the null key (Arrow gives nulls a group of their own). It lives in shard 0.
    private func groupIndexNullKey(_ shard: GroupShard) -> Int {
        if let i = shard.nullKeyGroup { return i }
        let i = shard.count
        shard.nullKeyGroup = i
        for j in 0..<shard.keyCols.count { shard.keyCols[j].append(.null) }
        shard.accs.append(contentsOf: repeatElement(StreamAccumulator(), count: aggregates.count))
        return i
    }

    private func addInto(_ acc: inout StreamAccumulator, _ v: StreamValue) {
        switch v {
        case .int(let x): acc.sumInt &+= x; acc.kind = Swift.max(acc.kind, 1)
        case .uint(let x): acc.sumUInt &+= x; acc.kind = Swift.max(acc.kind, 2)
        case .double(let x): acc.sumDouble += x; acc.kind = 3
        default: break
        }
    }

    /// Folds one dense-key partial straight into the host table (used once the state has spilled).
    private func foldDensePartialIntoHostTable(_ d: DensePartial) throws {
        guard let K = denseKeyCount else { return }
        var sums: [[StreamValue]] = [], counts: [[StreamValue]] = []
        var mins: [[StreamValue]] = [], maxs: [[StreamValue]] = []
        for (s, c, mn, mx) in d.parts {
            sums.append(s == nil ? [] : try s!.streamValues())
            counts.append(c == nil ? [] : try AnyMetalArray.int64(c!).streamValues())
            mins.append(mn == nil ? [] : try mn!.streamValues())
            maxs.append(mx == nil ? [] : try mx!.streamValues())
        }
        for k in 0..<K {
            var any = false
            for i in 0..<aggregates.count {
                if counts[i].count > k, case .int(let n) = counts[i][k], n > 0 { any = true }
                if sums[i].count > k, !sums[i][k].isNull { any = true }
                if mins[i].count > k, !mins[i][k].isNull { any = true }
                if maxs[i].count > k, !maxs[i][k].isNull { any = true }
            }
            guard any else { continue }
            ensureShards()
            let shard = shards[shardOf(Int64(k))]
            let base = groupIndexInt(shard, Int64(k)) * aggregates.count
            for i in 0..<aggregates.count {
                if counts[i].count > k, case .int(let n) = counts[i][k], n > 0 { shard.accs[base + i].count += n }
                if sums[i].count > k, !sums[i][k].isNull { addInto(&shard.accs[base + i], sums[i][k]) }
                if mins[i].count > k, !mins[i][k].isNull {
                    shard.accs[base + i].minV = minStreamValue(shard.accs[base + i].minV, mins[i][k])
                }
                if maxs[i].count > k, !maxs[i][k].isNull {
                    shard.accs[base + i].maxV = maxStreamValue(shard.accs[base + i].maxV, maxs[i][k])
                }
            }
        }
    }

    /// Moves the GPU dense state into the host table, then continues on the host path.
    private func spillDenseToHost() throws {
        spilled = true
        guard let K = denseKeyCount else { return }
        var sums: [[StreamValue]] = [], counts: [[StreamValue]] = []
        var mins: [[StreamValue]] = [], maxs: [[StreamValue]] = []
        for i in 0..<aggregates.count {
            sums.append(denseSum[i] == nil ? [] : try denseSum[i]!.streamValues())
            counts.append(denseCount[i] == nil ? [] : try AnyMetalArray.int64(denseCount[i]!).streamValues())
            mins.append(denseMin[i] == nil ? [] : try denseMin[i]!.streamValues())
            maxs.append(denseMax[i] == nil ? [] : try denseMax[i]!.streamValues())
        }
        for k in 0..<K {
            var any = false
            for i in 0..<aggregates.count {
                if counts[i].count > k, case .int(let n) = counts[i][k], n > 0 { any = true }
                if sums[i].count > k, !sums[i][k].isNull { any = true }
                if mins[i].count > k, !mins[i][k].isNull { any = true }
                if maxs[i].count > k, !maxs[i][k].isNull { any = true }
            }
            guard any else { continue }
            ensureShards()
            let shard = shards[shardOf(Int64(k))]
            let base = groupIndexInt(shard, Int64(k)) * aggregates.count
            for i in 0..<aggregates.count {
                if counts[i].count > k, case .int(let n) = counts[i][k], n > 0 { shard.accs[base + i].count = n }
                if sums[i].count > k, !sums[i][k].isNull { addInto(&shard.accs[base + i], sums[i][k]) }
                if mins[i].count > k, !mins[i][k].isNull { shard.accs[base + i].minV = mins[i][k] }
                if maxs[i].count > k, !maxs[i][k].isNull { shard.accs[base + i].maxV = maxs[i][k] }
            }
        }
        denseSum = []; denseCount = []; denseMin = []; denseMax = []
    }

    public func finish() throws -> StreamResult {
        var r = StreamResult()
        if let table = resident {
            r.batch = try residentResultBatch(table)
        } else if usesGPUState, !denseSum.isEmpty {
            r.batch = try denseResultBatch()
        } else {
            r.batch = try hostResultBatch()
        }
        r.rowsOut = r.batch?.length ?? 0
        return r
    }

    /// The resident table read out as a record batch, key column first, in ascending key order with
    /// the null key's group last - the same order the host table produces.
    private func residentResultBatch(_ table: StreamGroupTable) throws -> MetalRecordBatch {
        let (keys, sums, counts) = try table.readOut()
        let order = try keys.argsort(descending: false)
        let template = residentKeyTemplate ?? .int64(keys)
        var names = keyColumns
        var cols: [AnyMetalArray] = [try keyColumnLike(template, try keys.take(order))]
        for (i, a) in aggregates.enumerated() {
            names.append(a.name)
            let c = try counts[i].take(order)
            switch a.op {
            case .count:
                cols.append(.int64(c))
            case .sum:
                // Arrow's `sum` of an all-null group is null, which is what a zero count means here.
                cols.append(try maskByCount(try sums[i].take(order), c))
            case .mean:
                let s = try toDouble(try sums[i].take(order))
                let n = try c.cast(to: Double.self)
                cols.append(try maskByCount(.float64(try s.divide(n)), c))
            default:
                throw ArrowMetalError.unsupportedType("\(a.op.rawValue) in a resident streaming group-by")
            }
        }
        return try MetalRecordBatch(names: names, columns: cols)
    }

    /// Nulls out the groups whose count is zero, as Arrow's `sum` and `mean` do.
    private func maskByCount(_ v: AnyMetalArray, _ counts: MetalArray<Int64>) throws -> AnyMetalArray {
        let live = try counts.compare(.gt, 0)
        return try nullingWhereFalse(v, live)
    }

    /// The GPU-resident dense state read out as a record batch, key column first.
    private func denseResultBatch() throws -> MetalRecordBatch {
        guard let K = denseKeyCount else { throw ArrowMetalError.invalidArrowArray("no dense key count") }
        var names = keyColumns
        var cols: [AnyMetalArray] = [.int32(try MetalArray<Int32>((0..<K).map { Int32($0) }, context: context))]
        let zeros = try MetalArray<Int64>([Int64](repeating: 0, count: K), context: context)
        for (i, a) in aggregates.enumerated() {
            names.append(a.name)
            switch a.op {
            case .count:
                cols.append(.int64(denseCount[i] ?? zeros))
            case .sum:
                cols.append(denseSum[i] ?? .int64(zeros))
            case .mean:
                let s = try (denseSum[i] ?? .int64(zeros)).streamValues()
                let c = try AnyMetalArray.int64(denseCount[i] ?? zeros).streamValues()
                var out: [Double?] = []
                for k in 0..<K {
                    let n = (k < c.count ? c[k].asDouble : 0) ?? 0
                    let sv = k < s.count ? (s[k].asDouble ?? 0) : 0
                    out.append(n > 0 ? sv / n : nil)
                }
                cols.append(.float64(try MetalArray<Double>(out, context: context)))
            case .min:
                cols.append(denseMin[i] ?? .int64(zeros))
            case .max:
                cols.append(denseMax[i] ?? .int64(zeros))
            default: throw ArrowMetalError.unsupportedType("\(a.op.rawValue) in a dense-key streaming group-by")
            }
        }
        // Drop keys no row ever landed on, so the result matches the arbitrary-key path.
        let batch = try MetalRecordBatch(names: names, columns: cols)
        if let ci = aggregates.firstIndex(where: { $0.op == .count || $0.op == .sum || $0.op == .mean }),
           let counts = denseCount[ci] {
            let mask = try counts.compare(.gt, 0)
            return try batch.filter(mask)
        }
        return batch
    }

    /// The host table as a record batch, sorted by key so the output is deterministic.
    private func hostResultBatch() throws -> MetalRecordBatch {
        // Order (shard, group index) pairs, not the keys: the key values stay column-wise where they
        // were written, so a ten-million-group result never builds ten million little key arrays.
        var order: [(Int, Int)] = []
        order.reserveCapacity(shards.reduce(0) { $0 + $1.count })
        for (sid, shard) in shards.enumerated() {
            for g in 0..<shard.count { order.append((sid, g)) }
        }
        order.sort { a, b in
            for j in 0..<Swift.max(keyColumns.count, 1) {
                let x = shards[a.0].keyCols[j][a.1], y = shards[b.0].keyCols[j][b.1]
                if x != y { return StreamValue.less(x, y) }
            }
            return false
        }
        var names = keyColumns
        var cols: [AnyMetalArray] = []
        for j in 0..<keyColumns.count {
            let vals = order.map { shards[$0.0].keyCols[j][$0.1] }
            let fallback = AnyMetalArray.int64(try MetalArray<Int64>([Int64](), context: context))
            let template = keyTemplates.flatMap { $0.count > j ? $0[j] : nil } ?? fallback
            cols.append(try template.rebuild(vals, context: context))
        }
        let m = aggregates.count
        for (i, a) in aggregates.enumerated() {
            names.append(a.name)
            var vals: [StreamValue] = []
            vals.reserveCapacity(order.count)
            for (sid, g) in order {
                let acc = shards[sid].accs[g * m + i]
                switch a.op {
                case .count: vals.append(.int(acc.count))
                case .sum:
                    switch acc.kind {
                    case 1: vals.append(acc.count > 0 ? .int(acc.sumInt) : .null)
                    case 2: vals.append(acc.count > 0 ? .uint(acc.sumUInt) : .null)
                    case 3: vals.append(acc.count > 0 ? .double(acc.sumDouble) : .null)
                    default: vals.append(.null)
                    }
                case .mean:
                    let s: Double
                    switch acc.kind {
                    case 1: s = Double(acc.sumInt)
                    case 2: s = Double(acc.sumUInt)
                    default: s = acc.sumDouble
                    }
                    vals.append(acc.count > 0 ? .double(s / Double(acc.count)) : .null)
                case .min: vals.append(acc.minV)
                case .max: vals.append(acc.maxV)
                case .variance, .stddev:
                    let n = Double(acc.count), denom = n - Double(ddof)
                    if n <= 0 || denom <= 0 { vals.append(.null) }
                    else {
                        let v = Swift.max(0, (acc.sumSq - acc.sumDouble * acc.sumDouble / n) / denom)
                        vals.append(.double(a.op == .variance ? v : v.squareRoot()))
                    }
                case .countDistinctApprox: vals.append(.null)
                }
            }
            cols.append(try aggregateColumn(a, vals))
        }
        return try MetalRecordBatch(names: names, columns: cols)
    }

    private func aggregateColumn(_ a: StreamAggregate, _ vals: [StreamValue]) throws -> AnyMetalArray {
        switch a.op {
        case .count:
            return .int64(try MetalArray<Int64>(vals.map { if case .int(let x) = $0 { return x } else { return 0 } },
                                                context: context))
        case .mean, .variance, .stddev:
            return .float64(try MetalArray<Double>(vals.map { $0.isNull ? nil : $0.asDouble }, context: context))
        default:
            // sum / min / max keep the shape the GPU produced.
            if vals.contains(where: { if case .double = $0 { return true }; return false }) {
                return .float64(try MetalArray<Double>(vals.map { $0.isNull ? nil : $0.asDouble }, context: context))
            }
            if vals.contains(where: { if case .uint = $0 { return true }; return false }) {
                let out: [UInt64?] = vals.map {
                    if case .uint(let x) = $0 { return x }
                    if case .int(let x) = $0 { return UInt64(bitPattern: x) }
                    return nil
                }
                return .uint64(try MetalArray<UInt64>(out, context: context))
            }
            if vals.contains(where: { if case .string = $0 { return true }; return false }) {
                let out: [String?] = vals.map { if case .string(let s) = $0 { return s } else { return nil } }
                return .string(try MetalStringArray(out, context: context))
            }
            return .int64(try MetalArray<Int64>(vals.map { if case .int(let x) = $0 { return x } else { return nil } },
                                                context: context))
        }
    }
}

// MARK: - Shared aggregate plumbing

func minStreamValue(_ a: StreamValue, _ b: StreamValue) -> StreamValue {
    if a.isNull { return b }
    if b.isNull { return a }
    return StreamValue.less(b, a) ? b : a
}
func maxStreamValue(_ a: StreamValue, _ b: StreamValue) -> StreamValue {
    if a.isNull { return b }
    if b.isNull { return a }
    return StreamValue.less(a, b) ? b : a
}

/// The group sum of any supported value column, in the widest type Arrow gives it.
func groupSum(_ col: AnyMetalArray, _ gb: GroupBy<Int32>) throws -> AnyMetalArray {
    switch col {
    case .int8(let a): return .int64(try gb.sum(a))
    case .int16(let a): return .int64(try gb.sum(a))
    case .int32(let a): return .int64(try gb.sum(a))
    case .int64(let a): return .int64(try gb.sum(a))
    case .uint8(let a): return .int64(try gb.sum(a))
    case .uint16(let a): return .int64(try gb.sum(a))
    case .uint32(let a): return .int64(try gb.sum(a))
    case .uint64(let a): return .uint64(try gb.sumUnsigned(a))
    case .float32(let a): return .float64(try gb.sumFloatAsDouble(a))
    case .float64(let a): return .float64(try gb.sumDouble(a))
    case .temporal(let t): return .int64(try gb.sum(try t.int64Values()))
    default: throw ArrowMetalError.unsupportedType("streaming group sum over \(col.arrowFormat)")
    }
}

func groupMinMax(_ col: AnyMetalArray, _ gb: GroupBy<Int32>, isMin: Bool) throws -> AnyMetalArray {
    // Metal has no 64-bit atomics, so `GroupBy.min`/`max` only take 32-bit and narrower values; wider
    // ones go through the segmented (sorted-key) reduction instead.
    func f<T: ArrowPrimitive>(_ a: MetalArray<T>) throws -> MetalArray<T> {
        if T.byteWidth > 4 { return isMin ? try gb.min64(a) : try gb.max64(a) }
        return isMin ? try gb.min(a) : try gb.max(a)
    }
    switch col {
    case .int8(let a): return .int8(try f(a))
    case .int16(let a): return .int16(try f(a))
    case .int32(let a): return .int32(try f(a))
    case .int64(let a): return .int64(try f(a))
    case .uint8(let a): return .uint8(try f(a))
    case .uint16(let a): return .uint16(try f(a))
    case .uint32(let a): return .uint32(try f(a))
    case .uint64(let a): return .uint64(try f(a))
    case .float32(let a): return .float32(try f(a))
    case .float64(let a): return .float64(try f(a))
    case .temporal(let t):
        let v = try t.int64Values()
        return .temporal(try MetalTemporalArray(type: t.type, try f(v)))
    default: throw ArrowMetalError.unsupportedType("streaming group min/max over \(col.arrowFormat)")
    }
}

func countValid(_ col: AnyMetalArray, _ gb: GroupBy<Int32>) throws -> MetalArray<Int64> {
    switch col {
    case .int8(let a): return try gb.countValid(a)
    case .int16(let a): return try gb.countValid(a)
    case .int32(let a): return try gb.countValid(a)
    case .int64(let a): return try gb.countValid(a)
    case .uint8(let a): return try gb.countValid(a)
    case .uint16(let a): return try gb.countValid(a)
    case .uint32(let a): return try gb.countValid(a)
    case .uint64(let a): return try gb.countValid(a)
    case .float32(let a): return try gb.countValid(a)
    case .float64(let a): return try gb.countValid(a)
    case .temporal(let t): return try gb.countValid(try t.int64Values())
    case .string(let a), .binary(let a):
        return try gb.countValid(try a.byteLength())
    default: return try gb.count()
    }
}

func toDouble(_ col: AnyMetalArray) throws -> MetalArray<Double> {
    switch col {
    case .int8(let a): return try a.cast(to: Double.self)
    case .int16(let a): return try a.cast(to: Double.self)
    case .int32(let a): return try a.cast(to: Double.self)
    case .int64(let a): return try a.cast(to: Double.self)
    case .uint8(let a): return try a.cast(to: Double.self)
    case .uint16(let a): return try a.cast(to: Double.self)
    case .uint32(let a): return try a.cast(to: Double.self)
    case .uint64(let a): return try a.cast(to: Double.self)
    case .float32(let a): return try a.cast(to: Double.self)
    case .float64(let a): return a
    default: throw ArrowMetalError.unsupportedType("cannot widen \(col.arrowFormat) to float64")
    }
}

/// Element-wise sum of two accumulator columns (nil left means "first batch").
func elementwiseAdd(_ a: AnyMetalArray?, _ b: AnyMetalArray) throws -> AnyMetalArray {
    guard let a else { return b }
    switch (a, b) {
    case (.int64(let x), .int64(let y)): return .int64(try x.add(y))
    case (.uint64(let x), .uint64(let y)): return .uint64(try x.add(y))
    case (.float64(let x), .float64(let y)): return .float64(try x.add(y))
    default: throw ArrowMetalError.unsupportedType("dense group state add of \(a.arrowFormat) and \(b.arrowFormat)")
    }
}

/// Element-wise min / max of two accumulator columns, with Arrow's `skip_nulls` behaviour.
func elementwiseMinMax(_ a: AnyMetalArray?, _ b: AnyMetalArray, isMin: Bool) throws -> AnyMetalArray {
    guard let a else { return b }
    func f<T: ArrowPrimitive>(_ x: MetalArray<T>, _ y: MetalArray<T>) throws -> MetalArray<T> {
        isMin ? try x.minElementWise(y) : try x.maxElementWise(y)
    }
    switch (a, b) {
    case (.int8(let x), .int8(let y)): return .int8(try f(x, y))
    case (.int16(let x), .int16(let y)): return .int16(try f(x, y))
    case (.int32(let x), .int32(let y)): return .int32(try f(x, y))
    case (.int64(let x), .int64(let y)): return .int64(try f(x, y))
    case (.uint8(let x), .uint8(let y)): return .uint8(try f(x, y))
    case (.uint16(let x), .uint16(let y)): return .uint16(try f(x, y))
    case (.uint32(let x), .uint32(let y)): return .uint32(try f(x, y))
    case (.uint64(let x), .uint64(let y)): return .uint64(try f(x, y))
    case (.float32(let x), .float32(let y)): return .float32(try f(x, y))
    case (.float64(let x), .float64(let y)): return .float64(try f(x, y))
    default: throw ArrowMetalError.unsupportedType("dense group state min/max of \(a.arrowFormat)")
    }
}

// MARK: - Streaming top-k
//
// `StreamTopKOperator` lives in `StreamTopN.swift`, with the threshold pruning it shares with the
// `ORDER BY ... LIMIT n` path.

/// The k best rows of a batch by one column, in ranked order.
func topKRows(_ batch: MetalRecordBatch, column: String, k: Int, largest: Bool) throws -> MetalRecordBatch {
    guard let c = batch[column] else { throw ArrowMetalError.invalidArrowArray("no column named \(column)") }
    let n = batch.length
    if n > k, let idx = try topKIndicesIfSupported(c, k: k, largest: largest) {
        return try batch.take(idx)
    }
    // Fewer rows than k, or a column the selection kernel has no key for (`utf8`, `binary`, boolean):
    // order them all and keep the prefix, so the caller always sees a ranked result.
    let idx = try argsortAny(c, descending: largest)
    let ranked = try batch.take(idx)
    return ranked.length > k ? try ranked.slice(offset: 0, length: k) : ranked
}

func topKIndices(_ c: AnyMetalArray, k: Int, largest: Bool) throws -> MetalArray<Int32> {
    switch c {
    case .int8(let a): return try a.topK(k, largest: largest)
    case .int16(let a): return try a.topK(k, largest: largest)
    case .int32(let a): return try a.topK(k, largest: largest)
    case .int64(let a): return try a.topK(k, largest: largest)
    case .uint8(let a): return try a.topK(k, largest: largest)
    case .uint16(let a): return try a.topK(k, largest: largest)
    case .uint32(let a): return try a.topK(k, largest: largest)
    case .uint64(let a): return try a.topK(k, largest: largest)
    case .float32(let a): return try a.topK(k, largest: largest)
    case .float64(let a): return try a.topK(k, largest: largest)
    case .temporal(let t):
        switch t.storage {
        case .int32(let a): return try a.topK(k, largest: largest)
        case .int64(let a): return try a.topK(k, largest: largest)
        }
    default: throw ArrowMetalError.unsupportedType("top-k over \(c.arrowFormat)")
    }
}

/// Order of one column, nulls last. Numeric, boolean and temporal columns use the GPU radix sort
/// (`AnyMetalArray.argsortIndices`); utf8 and binary columns have no order-preserving GPU key yet, so
/// they fall back to a host sort of the string values (documented in docs/STREAMING.md).
func argsortAny(_ c: AnyMetalArray, descending: Bool) throws -> MetalArray<Int32> {
    switch c {
    case .string, .binary:
        let vals = try c.streamValues()
        var order = Array(0..<vals.count)
        order.sort { i, j in
            let a = vals[i], b = vals[j]
            if a == b { return i < j }
            let lt = StreamValue.less(a, b)
            // Nulls stay last in both directions.
            if a.isNull { return false }
            if b.isNull { return true }
            return descending ? !lt : lt
        }
        return try MetalArray<Int32>(order.map { Int32($0) }, context: c.anyContext)
    default:
        return try c.argsortIndices(descending: descending)
    }
}

// MARK: - Filter and project, with columns the fused compiler cannot materialise

/// True when the fused Expr compiler can *emit* a column of this type as a projection output.
///
/// The compiler evaluates every expression into a fixed-width register, so utf8, binary, temporal,
/// decimal and nested columns cannot be a projection's output — only its input.
func exprCanMaterialise(_ c: AnyMetalArray) -> Bool {
    switch c {
    case .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64,
         .float32, .float64, .boolean:
        return true
    default:
        return false
    }
}

/// Applies a filter and a projection to one batch, choosing between one fused kernel and the
/// two-step path that a string (or other non-register) passthrough column forces.
///
/// * **Fused** — every output is a numeric or boolean expression: the predicate and all projections
///   compile into one kernel, the batch is read once, nothing intermediate is materialised.
/// * **Two-step** — some output is a plain `utf8` / `binary` / temporal / decimal column, which the
///   compiler cannot write. The predicate still compiles into one fused kernel, producing a boolean
///   mask; the mask then drives the GPU `filter`, which carries *any* column type, and only the
///   computed outputs go back through the compiler.
func streamFilterProject(_ batch: MetalRecordBatch, filter: Expr?, projections: [(String, Expr)]?,
                         context: MetalContext) throws -> MetalRecordBatch {
    let projs = projections ?? batch.names.map { ($0, Expr.column($0)) }
    func passthrough(_ e: Expr) -> String? { if case .column(let n) = e { return n }; return nil }

    let fusable = projs.allSatisfy { p in
        guard let n = passthrough(p.1), let c = batch[n] else { return true }
        return exprCanMaterialise(c)
    }
    if fusable {
        var b = ExprQueryBuilder()
        if let f = filter { b = b.filter(f) }
        return try runExprQuery(b.project(projs), names: batch.names, columns: batch.columns,
                                context: context).recordBatch()
    }

    var work = batch
    if let f = filter {
        let mq = ExprQueryBuilder().project([("__mask", f)])
        let r = try runExprQuery(mq, names: batch.names, columns: batch.columns, context: context)
        guard let mask = r["__mask"]?.asBoolean else {
            throw ArrowMetalError.unsupportedType("a streaming filter predicate must be boolean")
        }
        work = try batch.filter(mask)
    }
    var out: [String: AnyMetalArray] = [:]
    var computed: [(String, Expr)] = []
    for (name, e) in projs {
        if let src = passthrough(e), let c = work[src] { out[name] = c } else { computed.append((name, e)) }
    }
    if !computed.isEmpty {
        let r = try runExprQuery(ExprQueryBuilder().project(computed), names: work.names,
                                 columns: work.columns, context: context)
        for (i, n) in r.names.enumerated() { out[n] = r.columns[i] }
    }
    var names: [String] = [], cols: [AnyMetalArray] = []
    for (name, _) in projs {
        guard let c = out[name] else { throw ArrowMetalError.invalidArrowArray("projection \(name) produced nothing") }
        names.append(name)
        cols.append(c)
    }
    return try MetalRecordBatch(names: names, columns: cols)
}
