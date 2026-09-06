import Foundation
import CArrowABI

// The user-facing façade: a source, an optional filter and projection, and one terminal.
//
//     let rows = try StreamQuery(ipc: "/data/events")
//         .filter(col("amount") > 100)
//         .groupBy(["region"], [StreamAggregate(.sum, "amount")])
//
// Every terminal runs the same three-stage pipeline (`StreamingExecutor`) over the same source, so
// the memory profile is identical whatever the question is: one batch in flight, plus the running
// state of the operator.

public final class StreamQuery {
    public let source: BatchSource
    public let context: MetalContext
    private var filterExpr: Expr?
    private var projections: [(String, Expr)]?
    /// The accumulated predicate, or nil.
    var filterExpression: Expr? { filterExpr }
    /// The projection list, or nil for "every column".
    var projectionList: [(String, Expr)]? { projections }
    /// Called after every batch with the progress so far.
    public var progress: ((StreamProgress) -> Void)?

    public init(source: BatchSource, context: MetalContext = .shared) {
        self.source = source
        self.context = context
    }

    /// A query over an Arrow IPC file or a directory of them, with readahead and triple buffering.
    public convenience init(ipc path: String, prefetchDepth: Int = 3, context: MetalContext = .shared) throws {
        self.init(source: try openIPCSource(path, prefetchDepth: prefetchDepth, context: context), context: context)
    }

    // MARK: builders

    @discardableResult
    public func filter(_ e: Expr) -> StreamQuery {
        filterExpr = filterExpr.map { .binary(.and, $0, e) } ?? e
        return self
    }

    /// Filter from the s-expression text the C ABI and Python bindings use.
    @discardableResult
    public func filter(text: String) throws -> StreamQuery {
        filter(try Expr(text: text))
    }

    /// Keep only these columns.
    @discardableResult
    public func select(_ names: [String]) -> StreamQuery {
        projections = names.map { ($0, Expr.column($0)) }
        return self
    }

    /// Materialise computed columns.
    @discardableResult
    public func project(_ pairs: [(String, Expr)]) -> StreamQuery {
        projections = pairs
        return self
    }

    private func executor() -> StreamingExecutor {
        let e = StreamingExecutor(source: source, context: context)
        e.progress = progress
        return e
    }

    /// The filter+project query for the streaming terminals, given the batch's own column names.
    func projectionQuery(names: [String]) -> ExprQuery {
        var b = ExprQueryBuilder()
        if let f = filterExpr { b = b.filter(f) }
        let p = projections ?? names.map { ($0, Expr.column($0)) }
        return b.project(p)
    }

    // MARK: terminals

    /// Streams filtered and projected rows into `sink`.
    @discardableResult
    public func sink(_ s: StreamSink) throws -> StreamResult {
        let op = LazyProjectOperator(query: self, sink: s)
        return try executor().run(op)
    }

    /// Streams filtered and projected rows into an Arrow IPC stream file.
    @discardableResult
    public func sinkIPC(_ url: URL) throws -> StreamResult {
        let s = try IPCStreamSink(url: url)
        return try sink(s)
    }

    /// Collects the filtered and projected rows into one batch. Only for results known to be small.
    public func collect() throws -> MetalRecordBatch? {
        let s = CollectingSink()
        _ = try sink(s)
        return try s.table()
    }

    /// Whole-dataset aggregates.
    public func aggregate(_ specs: [StreamAggregate], hllPrecision: Int = 14, ddof: Int = 0) throws -> StreamResult {
        let op = StreamAggregateOperator(specs, filter: filterExpr, hllPrecision: hllPrecision)
        op.ddof = ddof
        return try executor().run(op)
    }

    public func sum(_ column: String) throws -> ExprScalar? {
        try aggregate([StreamAggregate(.sum, column, name: "sum")]).onlyScalar
    }
    public func count() throws -> Int {
        Int(try aggregate([StreamAggregate(.count, nil, name: "count")]).onlyScalar?.asInt64 ?? 0)
    }
    public func mean(_ column: String) throws -> Double? {
        try aggregate([StreamAggregate(.mean, column, name: "mean")]).onlyScalar?.asDouble
    }
    public func minimum(_ column: String) throws -> ExprScalar? {
        try aggregate([StreamAggregate(.min, column, name: "min")]).onlyScalar
    }
    public func maximum(_ column: String) throws -> ExprScalar? {
        try aggregate([StreamAggregate(.max, column, name: "max")]).onlyScalar
    }
    public func variance(_ column: String, ddof: Int = 0) throws -> Double? {
        try aggregate([StreamAggregate(.variance, column, name: "var")], ddof: ddof).onlyScalar?.asDouble
    }

    /// Approximate distinct count via a GPU HyperLogLog sketch merged across batches.
    public func countDistinctApprox(_ column: String, precision: Int = 14) throws -> Int {
        let r = try aggregate([StreamAggregate(.countDistinctApprox, column, name: "ndv")], hllPrecision: precision)
        return Int(r.onlyScalar?.asInt64 ?? 0)
    }

    /// Streaming group-by with a persistent aggregate state across batches.
    public func groupBy(_ keys: [String], _ aggs: [StreamAggregate], denseKeyCount: Int? = nil,
                        ddof: Int = 0) throws -> StreamResult {
        let op = StreamGroupByOperator(keys: keys, aggregates: aggs, filter: filterExpr, denseKeyCount: denseKeyCount)
        op.ddof = ddof
        return try executor().run(op)
    }

    /// The k rows with the largest (or smallest) value of `column`, merged across batches.
    public func topK(_ column: String, k: Int, largest: Bool = true) throws -> StreamResult {
        try executor().run(StreamTopKOperator(column: column, k: k, largest: largest, filter: filterExpr))
    }

    /// Approximate quantiles via a GPU digest merged across batches.
    public func quantiles(_ column: String, _ qs: [Double], compression: Int = 1000) throws -> [Double?] {
        try quantileResult(column, qs, compression: compression).scalars.map { $0.asDouble }
    }

    /// The same, with the pipeline statistics attached (the shape the C ABI returns).
    public func quantileResult(_ column: String, _ qs: [Double], compression: Int = 1000) throws -> StreamResult {
        try executor().run(StreamQuantileOperator(column: column, quantiles: qs,
                                                  compression: compression, filter: filterExpr))
    }

    /// External sort: sorted runs to `scratch`, then a k-way merge into `sink`.
    @discardableResult
    public func sort(by keys: [ExternalSortOperator.Key], into sink: StreamSink, scratch: URL,
                     limit: Int? = nil) throws -> StreamResult {
        let op = try ExternalSortOperator(keys: keys, sink: sink, scratch: scratch, limit: limit, context: context)
        return try executor().run(op)
    }

    /// External sort collected into one batch (for results known to be small, or with a `limit`).
    public func sorted(by keys: [ExternalSortOperator.Key], scratch: URL, limit: Int? = nil) throws -> MetalRecordBatch? {
        let s = CollectingSink()
        _ = try sort(by: keys, into: s, scratch: scratch, limit: limit)
        return try s.table()
    }

    /// Broadcast join: `build` is held in memory, this query's rows are streamed past it.
    @discardableResult
    public func joinBroadcast(_ build: MetalRecordBatch, on probeKey: String, buildKey: String,
                              kind: JoinKind = .inner, into sink: StreamSink) throws -> StreamResult {
        let op = BroadcastJoinOperator(build: build, probeKey: probeKey, buildKey: buildKey,
                                       kind: kind, sink: sink, filter: filterExpr)
        return try executor().run(op)
    }

    /// A pull-based reader over the filtered and projected rows: one source batch per `nextOutputBatch`.
    public func reader() -> StreamQueryReader {
        StreamQueryReader(query: self)
    }

    /// Exports the filtered and projected rows as an Arrow C Stream, for pyarrow / Polars to consume lazily.
    public func exportArrowArrayStream(into out: UnsafeMutablePointer<ArrowArrayStream>) {
        ArrowStreamExporter(reader()).export(into: out)
    }
}

/// Applies a `StreamQuery`'s filter and projection to every batch and writes to a sink. Built lazily
/// because the projection defaults to "every column", which is only known once a batch has arrived.
final class LazyProjectOperator: StreamOperator {
    let query: StreamQuery
    let sink: StreamSink
    private var rows = 0

    init(query: StreamQuery, sink: StreamSink) {
        self.query = query
        self.sink = sink
    }

    func process(_ batch: MetalRecordBatch) throws -> Any? {
        let out = try streamFilterProject(batch, filter: query.filterExpression,
                                          projections: query.projectionList,
                                          context: batch.firstContext ?? query.context)
        _ = out.length
        return out
    }

    func merge(_ partial: Any) throws {
        guard let b = partial as? MetalRecordBatch else { return }
        rows += b.length
        try sink.write(b)
    }

    func finish() throws -> StreamResult {
        try sink.finish()
        var r = StreamResult()
        r.rowsOut = rows
        if let c = sink as? CollectingSink { r.batch = try c.table() }
        return r
    }
}

/// Pull-based filter+project reader. Each call pulls exactly one source batch and returns the rows
/// that survive, so a consumer (pyarrow, Polars, a C Stream) drives the whole out-of-core pipeline
/// one batch at a time and never materialises more than that.
public final class StreamQueryReader: StreamBatchReader {
    private let query: StreamQuery
    private var finished = false

    init(query: StreamQuery) { self.query = query }

    public func nextOutputBatch() throws -> MetalRecordBatch? {
        while !finished {
            guard let b = try query.source.nextBatch() else { finished = true; return nil }
            let out = try query.context.batch {
                try streamFilterProject(b, filter: query.filterExpression, projections: query.projectionList,
                                        context: b.firstContext ?? query.context)
            }
            // An all-filtered-out batch is skipped rather than handed on as an empty one, so a
            // consumer never sees a zero-row batch in the middle of a stream.
            if out.length > 0 { return out }
        }
        return nil
    }

    public func outputSchema() throws -> MetalRecordBatch? { nil }
}
