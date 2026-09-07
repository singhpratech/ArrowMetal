import Foundation
import CArrowABI
import ArrowMetal

// The C ABI for out-of-core streaming execution (docs/STREAMING.md).
//
// A stream handle owns a `BatchSource` plus the filter and projection built up on it; a terminal call
// runs the three-stage pipeline and returns a result handle carrying columns, scalars and the run's
// statistics. Nothing here materialises the dataset: the only thing that grows with the input is the
// answer.

private let streamErrorKey = "ArrowMetalC.streamLastError"
private func setStreamError(_ e: Error) { Thread.current.threadDictionary[streamErrorKey] = "\(e)" }

/// A stream handle: the source and the query being built on it.
final class StreamBox {
    let query: StreamQuery
    var progress: (@convention(c) (Int64, Int64, Int64, Int64, UnsafeMutableRawPointer?) -> Void)?
    var progressUser: UnsafeMutableRawPointer?
    /// Keeps C strings alive for the caller.
    private var strings: [UnsafeMutablePointer<CChar>] = []
    init(_ q: StreamQuery) { query = q }
    deinit { for s in strings { free(s) } }
    func cString(_ s: String) -> UnsafePointer<CChar> {
        let c = strdup(s)!
        strings.append(c)
        return UnsafePointer(c)
    }
}

/// A finished streamed query.
final class StreamResultBox {
    let result: StreamResult
    private var strings: [UnsafeMutablePointer<CChar>] = []
    init(_ r: StreamResult) { result = r }
    deinit { for s in strings { free(s) } }
    func cString(_ s: String) -> UnsafePointer<CChar> {
        let c = strdup(s)!
        strings.append(c)
        return UnsafePointer(c)
    }
}

@inline(__always) private func streamBox(_ p: OpaquePointer?) -> StreamBox? {
    guard let p else { return nil }
    return Unmanaged<StreamBox>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue()
}
@inline(__always) private func resultBox(_ p: OpaquePointer?) -> StreamResultBox? {
    guard let p else { return nil }
    return Unmanaged<StreamResultBox>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue()
}

private func cstrings(_ p: UnsafeMutablePointer<UnsafePointer<CChar>?>?, _ n: Int64) -> [String]? {
    guard n >= 0 else { return nil }
    guard n == 0 else {
        guard let p else { return nil }
        var out: [String] = []
        for i in 0..<Int(n) {
            guard let s = p[i] else { return nil }
            out.append(String(cString: s))
        }
        return out
    }
    return []
}

/// Aggregate op codes, the C ABI contract (see include/arrowmetal.h).
private func aggregateOp(_ code: Int32) -> StreamAggregate.Op? {
    switch code {
    case 0: return .sum
    case 1: return .count
    case 2: return .min
    case 3: return .max
    case 4: return .mean
    case 5: return .variance
    case 6: return .stddev
    case 7: return .countDistinctApprox
    default: return nil
    }
}

// MARK: - Opening a stream

@_cdecl("am_stream_last_error")
public func am_stream_last_error() -> UnsafePointer<CChar>? {
    let s = (Thread.current.threadDictionary[streamErrorKey] as? String) ?? ""
    if let old = Thread.current.threadDictionary["ArrowMetalC.streamLastErrorC"] as? UnsafeMutablePointer<CChar> { free(old) }
    let c = strdup(s)!
    Thread.current.threadDictionary["ArrowMetalC.streamLastErrorC"] = c
    return UnsafePointer(c)
}

/// Opens an Arrow IPC file, or a directory of them, as a prefetching stream source.
@_cdecl("am_stream_open_ipc")
public func am_stream_open_ipc(_ path: UnsafePointer<CChar>?, _ prefetchDepth: Int32,
                               _ readers: Int32) -> OpaquePointer? {
    guard let path else { return nil }
    do {
        let q = try StreamQuery(ipc: String(cString: path), prefetchDepth: Int(prefetchDepth),
                                readers: Int(Swift.max(1, readers)))
        return OpaquePointer(Unmanaged.passRetained(StreamBox(q)).toOpaque())
    } catch {
        setStreamError(error)
        return nil
    }
}

/// Takes ownership of a foreign `ArrowArrayStream` (a pyarrow RecordBatchReader or dataset scanner,
/// a Polars LazyFrame, DuckDB, ...) and streams batches out of it.
@_cdecl("am_stream_from_c_stream")
public func am_stream_from_c_stream(_ stream: UnsafeMutablePointer<ArrowArrayStream>?,
                                    _ prefetchDepth: Int32) -> OpaquePointer? {
    guard let stream else { return nil }
    do {
        let base = try CStreamSource(stream)
        let src: BatchSource = prefetchDepth <= 0 ? base : PrefetchingSource(base, depth: Int(prefetchDepth))
        return OpaquePointer(Unmanaged.passRetained(StreamBox(StreamQuery(source: src))).toOpaque())
    } catch {
        setStreamError(error)
        return nil
    }
}

/// Streams batches already in Metal memory (a chunked table handed over as a C stream is the usual
/// route; this one takes an array of exported record batches).
@_cdecl("am_stream_release")
public func am_stream_release(_ s: OpaquePointer?) {
    guard let s else { return }
    Unmanaged<StreamBox>.fromOpaque(UnsafeRawPointer(s)).release()
}

// MARK: - Building the query

@_cdecl("am_stream_filter")
public func am_stream_filter(_ s: OpaquePointer?, _ exprText: UnsafePointer<CChar>?) -> Int32 {
    guard let b = streamBox(s), let exprText else { return 2 }
    do {
        _ = b.query.filter(try Expr(text: String(cString: exprText)))
        return 0
    } catch { setStreamError(error); return 1 }
}

@_cdecl("am_stream_select")
public func am_stream_select(_ s: OpaquePointer?, _ names: UnsafeMutablePointer<UnsafePointer<CChar>?>?,
                             _ n: Int64) -> Int32 {
    guard let b = streamBox(s), let ns = cstrings(names, n) else { return 2 }
    _ = b.query.select(ns)
    return 0
}

@_cdecl("am_stream_project")
public func am_stream_project(_ s: OpaquePointer?, _ names: UnsafeMutablePointer<UnsafePointer<CChar>?>?,
                              _ exprs: UnsafeMutablePointer<UnsafePointer<CChar>?>?, _ n: Int64) -> Int32 {
    guard let b = streamBox(s), let ns = cstrings(names, n), let es = cstrings(exprs, n) else { return 2 }
    do {
        _ = b.query.project(try zip(ns, es).map { ($0, try Expr(text: $1)) })
        return 0
    } catch { setStreamError(error); return 1 }
}

/// Registers a progress callback invoked once per batch.
@_cdecl("am_stream_set_progress")
public func am_stream_set_progress(_ s: OpaquePointer?,
                                   _ fn: (@convention(c) (Int64, Int64, Int64, Int64, UnsafeMutableRawPointer?) -> Void)?,
                                   _ user: UnsafeMutableRawPointer?) -> Int32 {
    guard let b = streamBox(s) else { return 2 }
    b.progress = fn
    b.progressUser = user
    if let fn {
        b.query.progress = { p in fn(Int64(p.batches), Int64(p.rows), p.bytesRead, p.totalBytes, user) }
    } else {
        b.query.progress = nil
    }
    return 0
}

// MARK: - Terminals

private func emit(_ r: StreamResult, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    out?.pointee = OpaquePointer(Unmanaged.passRetained(StreamResultBox(r)).toOpaque())
    return 0
}

private func runTerminal(_ out: UnsafeMutablePointer<OpaquePointer?>?,
                         _ body: () throws -> StreamResult) -> Int32 {
    do { return emit(try body(), out) } catch { setStreamError(error); return 1 }
}

/// Runs one whole query given as the `(query ...)` s-expression of docs/EXPR.md.
///
/// A `project` terminal streams rows: with `sink_path` they go to an Arrow IPC stream file, without
/// one they are collected into the result's columns. An `aggregate` terminal streams the aggregates
/// and returns them as scalars, decomposed so that each batch contributes independently.
@_cdecl("am_stream_query")
public func am_stream_query(_ s: OpaquePointer?, _ queryText: UnsafePointer<CChar>?,
                            _ sinkPath: UnsafePointer<CChar>?,
                            _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let b = streamBox(s), let queryText, let out else { return 2 }
    return runTerminal(out) {
        let q = try ExprQuery(text: String(cString: queryText))
        if let f = q.filter { _ = b.query.filter(f) }
        switch q.terminal {
        case .project(let ps):
            _ = b.query.project(ps.map { ($0.name, $0.expr) })
            if let sinkPath {
                return try b.query.sinkIPC(URL(fileURLWithPath: String(cString: sinkPath)))
            }
            let sink = CollectingSink()
            var r = try b.query.sink(sink)
            r.batch = try sink.table()
            return r
        case .aggregate(let aggs):
            let specs: [StreamAggregate] = try aggs.map { a in
                let op: StreamAggregate.Op
                switch a.op {
                case .sum: op = .sum
                case .count: op = .count
                case .min: op = .min
                case .max: op = .max
                case .mean: op = .mean
                }
                guard let e = a.expr else { return StreamAggregate(op, nil, name: a.name) }
                guard case .column(let c) = e else {
                    throw ArrowMetalError.unsupportedType(
                        "a streamed aggregate takes a column, not the expression \(e); project it first")
                }
                return StreamAggregate(op, c, name: a.name)
            }
            return try b.query.aggregate(specs)
        }
    }
}

/// Whole-dataset aggregates by op code. `columns[i]` may be NULL for `count`.
@_cdecl("am_stream_aggregate")
public func am_stream_aggregate(_ s: OpaquePointer?,
                                _ ops: UnsafeMutablePointer<Int32>?,
                                _ columns: UnsafeMutablePointer<UnsafePointer<CChar>?>?,
                                _ names: UnsafeMutablePointer<UnsafePointer<CChar>?>?,
                                _ n: Int64, _ hllPrecision: Int32, _ ddof: Int32,
                                _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let b = streamBox(s), let ops, let names, let out, n > 0 else { return 2 }
    guard let ns = cstrings(names, n) else { return 2 }
    return runTerminal(out) {
        var specs: [StreamAggregate] = []
        for i in 0..<Int(n) {
            guard let op = aggregateOp(ops[i]) else {
                throw ArrowMetalError.unsupportedType("unknown streaming aggregate op \(ops[i])")
            }
            let col = columns?[i].map { String(cString: $0) }
            specs.append(StreamAggregate(op, col, name: ns[i]))
        }
        return try b.query.aggregate(specs, hllPrecision: Int(hllPrecision), ddof: Int(ddof))
    }
}

/// Streaming group-by. `dense_key_count > 0` keeps the global table on the GPU for a single integer
/// key already inside `[0, dense_key_count)`.
@_cdecl("am_stream_group_by")
public func am_stream_group_by(_ s: OpaquePointer?,
                               _ keys: UnsafeMutablePointer<UnsafePointer<CChar>?>?, _ nKeys: Int64,
                               _ ops: UnsafeMutablePointer<Int32>?,
                               _ columns: UnsafeMutablePointer<UnsafePointer<CChar>?>?,
                               _ names: UnsafeMutablePointer<UnsafePointer<CChar>?>?, _ nAggs: Int64,
                               _ denseKeyCount: Int64, _ ddof: Int32,
                               _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let b = streamBox(s), let ops, let names, let out, nKeys > 0, nAggs > 0 else { return 2 }
    guard let ks = cstrings(keys, nKeys), let ns = cstrings(names, nAggs) else { return 2 }
    return runTerminal(out) {
        var specs: [StreamAggregate] = []
        for i in 0..<Int(nAggs) {
            guard let op = aggregateOp(ops[i]) else {
                throw ArrowMetalError.unsupportedType("unknown streaming aggregate op \(ops[i])")
            }
            let col = columns?[i].map { String(cString: $0) }
            specs.append(StreamAggregate(op, col, name: ns[i]))
        }
        return try b.query.groupBy(ks, specs, denseKeyCount: denseKeyCount > 0 ? Int(denseKeyCount) : nil,
                                   ddof: Int(ddof))
    }
}

@_cdecl("am_stream_top_k")
public func am_stream_top_k(_ s: OpaquePointer?, _ column: UnsafePointer<CChar>?, _ k: Int64,
                            _ largest: Int32, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let b = streamBox(s), let column, let out else { return 2 }
    return runTerminal(out) {
        try b.query.topK(String(cString: column), k: Int(k), largest: largest != 0)
    }
}

@_cdecl("am_stream_quantiles")
public func am_stream_quantiles(_ s: OpaquePointer?, _ column: UnsafePointer<CChar>?,
                                _ qs: UnsafeMutablePointer<Double>?, _ nq: Int64, _ compression: Int64,
                                _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let b = streamBox(s), let column, let qs, let out, nq > 0 else { return 2 }
    return runTerminal(out) {
        let want = (0..<Int(nq)).map { qs[$0] }
        return try b.query.quantileResult(String(cString: column), want,
                                          compression: compression > 0 ? Int(compression) : 1000)
    }
}

/// External sort. `sink_path` NULL collects the rows (use `limit` unless the result is small).
@_cdecl("am_stream_sort")
public func am_stream_sort(_ s: OpaquePointer?,
                           _ columns: UnsafeMutablePointer<UnsafePointer<CChar>?>?,
                           _ descending: UnsafeMutablePointer<Int32>?, _ nKeys: Int64,
                           _ limit: Int64, _ scratch: UnsafePointer<CChar>?,
                           _ sinkPath: UnsafePointer<CChar>?,
                           _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let b = streamBox(s), let out, nKeys > 0, let cols = cstrings(columns, nKeys) else { return 2 }
    return runTerminal(out) {
        let keys = cols.enumerated().map {
            ExternalSortOperator.Key($0.element, descending: (descending?[$0.offset] ?? 0) != 0)
        }
        let dir = scratch.map { URL(fileURLWithPath: String(cString: $0)) }
            ?? FileManager.default.temporaryDirectory.appendingPathComponent("arrowmetal-sort-\(UUID().uuidString)")
        var sink: StreamSink = CollectingSink()
        if let p = sinkPath { sink = try IPCStreamSink(url: URL(fileURLWithPath: String(cString: p))) }
        var r = try b.query.sort(by: keys, into: sink, scratch: dir, limit: limit > 0 ? Int(limit) : nil)
        if let c = sink as? CollectingSink { r.batch = try c.table() }
        return r
    }
}

/// Streams the filtered and projected rows into an Arrow IPC stream file.
@_cdecl("am_stream_sink_ipc")
public func am_stream_sink_ipc(_ s: OpaquePointer?, _ path: UnsafePointer<CChar>?,
                               _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let b = streamBox(s), let path, let out else { return 2 }
    return runTerminal(out) { try b.query.sinkIPC(URL(fileURLWithPath: String(cString: path))) }
}

/// Broadcast join: the build side is drained from `build` into memory, this stream is the probe side.
@_cdecl("am_stream_join_broadcast")
public func am_stream_join_broadcast(_ s: OpaquePointer?, _ build: UnsafeMutablePointer<ArrowArrayStream>?,
                                     _ probeKey: UnsafePointer<CChar>?, _ buildKey: UnsafePointer<CChar>?,
                                     _ kind: Int32, _ sinkPath: UnsafePointer<CChar>?,
                                     _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let b = streamBox(s), let build, let probeKey, let buildKey, let out else { return 2 }
    return runTerminal(out) {
        let buildBatch = try loadBuildSide(try CStreamSource(build))
        var sink: StreamSink = CollectingSink()
        if let p = sinkPath { sink = try IPCStreamSink(url: URL(fileURLWithPath: String(cString: p))) }
        var r = try b.query.joinBroadcast(buildBatch, on: String(cString: probeKey),
                                          buildKey: String(cString: buildKey),
                                          kind: kind == 1 ? .left : .inner, into: sink)
        if let c = sink as? CollectingSink { r.batch = try c.table() }
        return r
    }
}

/// Grace hash join: both sides streamed and partitioned to disk, then joined partition by partition.
@_cdecl("am_stream_join_grace")
public func am_stream_join_grace(_ left: OpaquePointer?, _ right: OpaquePointer?,
                                 _ leftKey: UnsafePointer<CChar>?, _ rightKey: UnsafePointer<CChar>?,
                                 _ kind: Int32, _ partitions: Int64,
                                 _ scratch: UnsafePointer<CChar>?, _ sinkPath: UnsafePointer<CChar>?,
                                 _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let l = streamBox(left), let r = streamBox(right), let leftKey, let rightKey, let out else { return 2 }
    return runTerminal(out) {
        let dir = scratch.map { URL(fileURLWithPath: String(cString: $0)) }
            ?? FileManager.default.temporaryDirectory.appendingPathComponent("arrowmetal-grace-\(UUID().uuidString)")
        var sink: StreamSink = CollectingSink()
        if let p = sinkPath { sink = try IPCStreamSink(url: URL(fileURLWithPath: String(cString: p))) }
        let gs = try graceHashJoin(left: l.query.source, right: r.query.source,
                                   leftKey: String(cString: leftKey), rightKey: String(cString: rightKey),
                                   kind: kind == 1 ? .left : .inner, partitions: Int(partitions),
                                   scratch: dir, sink: sink, context: l.query.context)
        var res = StreamResult()
        res.rowsOut = gs.outputRows
        res.stats.rows = gs.leftRows + gs.rightRows
        res.stats.wallNanos = gs.wallNanos
        res.stats.bytesRead = gs.spilledBytes
        if let c = sink as? CollectingSink { res.batch = try c.table() }
        return res
    }
}

/// Exports the filtered and projected rows as an Arrow C Stream: a pull-based, lazy hand-off.
@_cdecl("am_stream_export_c")
public func am_stream_export_c(_ s: OpaquePointer?, _ out: UnsafeMutablePointer<ArrowArrayStream>?) -> Int32 {
    guard let b = streamBox(s), let out else { return 2 }
    b.query.exportArrowArrayStream(into: out)
    return 0
}

// MARK: - Results

@_cdecl("am_stream_result_column_count")
public func am_stream_result_column_count(_ r: OpaquePointer?) -> Int64 {
    Int64(resultBox(r)?.result.batch?.columnCount ?? 0)
}

@_cdecl("am_stream_result_column_name")
public func am_stream_result_column_name(_ r: OpaquePointer?, _ i: Int64) -> UnsafePointer<CChar>? {
    guard let b = resultBox(r), let batch = b.result.batch, i >= 0, i < Int64(batch.columnCount) else { return nil }
    return b.cString(batch.names[Int(i)])
}

@_cdecl("am_stream_result_column")
public func am_stream_result_column(_ r: OpaquePointer?, _ i: Int64,
                                    _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let b = resultBox(r), let batch = b.result.batch, let out,
          i >= 0, i < Int64(batch.columnCount) else { return 2 }
    out.pointee = OpaquePointer(Unmanaged.passRetained(Box(batch.columns[Int(i)])).toOpaque())
    return 0
}

@_cdecl("am_stream_result_scalar_count")
public func am_stream_result_scalar_count(_ r: OpaquePointer?) -> Int64 {
    Int64(resultBox(r)?.result.scalars.count ?? 0)
}

@_cdecl("am_stream_result_scalar_name")
public func am_stream_result_scalar_name(_ r: OpaquePointer?, _ i: Int64) -> UnsafePointer<CChar>? {
    guard let b = resultBox(r), i >= 0, i < Int64(b.result.scalarNames.count) else { return nil }
    return b.cString(b.result.scalarNames[Int(i)])
}

/// out_kind: 0 int64 in out_i64, 1 uint64 in the same slot, 2 float64 in out_f64.
@_cdecl("am_stream_result_scalar")
public func am_stream_result_scalar(_ r: OpaquePointer?, _ i: Int64,
                                    _ outI64: UnsafeMutablePointer<Int64>?, _ outF64: UnsafeMutablePointer<Double>?,
                                    _ outKind: UnsafeMutablePointer<Int32>?,
                                    _ isNull: UnsafeMutablePointer<Int32>?) -> Int32 {
    guard let b = resultBox(r), i >= 0, i < Int64(b.result.scalars.count) else { return 2 }
    isNull?.pointee = 0
    switch b.result.scalars[Int(i)] {
    case .null: isNull?.pointee = 1; outKind?.pointee = 0; outI64?.pointee = 0; outF64?.pointee = 0
    case .int(let v): outKind?.pointee = 0; outI64?.pointee = v
    case .uint(let v): outKind?.pointee = 1; outI64?.pointee = Int64(bitPattern: v)
    case .double(let v): outKind?.pointee = 2; outF64?.pointee = v
    }
    return 0
}

@_cdecl("am_stream_result_rows_out")
public func am_stream_result_rows_out(_ r: OpaquePointer?) -> Int64 { Int64(resultBox(r)?.result.rowsOut ?? 0) }

/// Pipeline statistics: how many batches and rows went through, how much came off disk, and how much
/// of the three stages' work overlapped (1.0 serial, up to 3.0 fully overlapped).
@_cdecl("am_stream_result_stats")
public func am_stream_result_stats(_ r: OpaquePointer?,
                                   _ batches: UnsafeMutablePointer<Int64>?, _ rows: UnsafeMutablePointer<Int64>?,
                                   _ bytesRead: UnsafeMutablePointer<Int64>?,
                                   _ wall: UnsafeMutablePointer<Double>?, _ read: UnsafeMutablePointer<Double>?,
                                   _ gpu: UnsafeMutablePointer<Double>?, _ merge: UnsafeMutablePointer<Double>?,
                                   _ overlap: UnsafeMutablePointer<Double>?) -> Int32 {
    guard let b = resultBox(r) else { return 2 }
    let s = b.result.stats
    batches?.pointee = Int64(s.batches)
    rows?.pointee = Int64(s.rows)
    bytesRead?.pointee = s.bytesRead
    wall?.pointee = Double(s.wallNanos) / 1e9
    read?.pointee = Double(s.readNanos) / 1e9
    gpu?.pointee = Double(s.gpuNanos) / 1e9
    merge?.pointee = Double(s.mergeNanos) / 1e9
    overlap?.pointee = s.overlap
    return 0
}

@_cdecl("am_stream_result_release")
public func am_stream_result_release(_ r: OpaquePointer?) {
    guard let r else { return }
    Unmanaged<StreamResultBox>.fromOpaque(UnsafeRawPointer(r)).release()
}
