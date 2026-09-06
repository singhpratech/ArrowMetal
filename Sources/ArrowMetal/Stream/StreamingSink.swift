import Foundation
import CArrowABI

// Where a streamed query's output goes. A sink takes one batch at a time and never holds the whole
// result, so `filter -> sink_ipc` over a 100 GB dataset runs in a bounded amount of memory.
//
// Three shapes:
//   * `IPCStreamSink` writes the Arrow IPC *stream* encapsulation incrementally to a file.
//   * `CallbackSink` hands each batch to a closure (the C ABI and Python callback path).
//   * `CollectingSink` keeps the batches (tests, small results, and the build side of a join).
// The pull-based export to a foreign consumer is `StreamBatchReader.exportArrowArrayStream`, at the
// bottom of this file: pyarrow and Polars can then consume ArrowMetal's output lazily.

/// A destination for streamed output batches.
public protocol StreamSink: AnyObject {
    func write(_ batch: MetalRecordBatch) throws
    /// Called once, after the last batch. Safe to call twice.
    func finish() throws
    /// Rows handed to this sink.
    var rowsWritten: Int { get }
}

/// Discards everything; used when a query's terminal is an aggregate.
public final class NullSink: StreamSink {
    public private(set) var rowsWritten = 0
    public init() {}
    public func write(_ batch: MetalRecordBatch) throws { rowsWritten += batch.length }
    public func finish() throws {}
}

/// Keeps every batch in memory. Only for results that are known to be small.
public final class CollectingSink: StreamSink {
    public private(set) var batches: [MetalRecordBatch] = []
    public private(set) var rowsWritten = 0
    public init() {}
    public func write(_ batch: MetalRecordBatch) throws {
        guard batch.length > 0 else { return }
        batches.append(batch)
        rowsWritten += batch.length
    }
    public func finish() throws {}
    public func table() throws -> MetalRecordBatch? { batches.isEmpty ? nil : try concatBatches(batches) }
}

/// Hands each batch to a closure.
public final class CallbackSink: StreamSink {
    private let body: (MetalRecordBatch) throws -> Void
    public private(set) var rowsWritten = 0
    public init(_ body: @escaping (MetalRecordBatch) throws -> Void) { self.body = body }
    public func write(_ batch: MetalRecordBatch) throws { rowsWritten += batch.length; try body(batch) }
    public func finish() throws {}
}

/// Writes batches to a file in the Arrow IPC **stream** encapsulation, incrementally.
///
/// The encapsulation is a schema message, then one message per record batch, then an 8-byte
/// end-of-stream marker — a pure concatenation, which is exactly what makes it appendable. The bytes
/// for each message come from `ArrowIPCWriter.encode`, with the schema prologue (measured once, by
/// encoding the schema with no batches) and the end-of-stream marker trimmed off each call.
///
/// `pyarrow.ipc.open_stream`, `polars.scan_ipc_stream` and this package's own `ArrowIPCReader` all
/// read the result; `ArrowIPCReader` sniffs the magic and takes the stream path automatically.
/// Dictionary-encoded columns are decoded before writing (one dictionary per stream is the only form
/// the writer emits, and an incremental sink cannot promise that across batches).
public final class IPCStreamSink: StreamSink {
    private let handle: FileHandle
    private let url: URL
    private var schema: ArrowIPCSchema?
    private var prologueBytes = 0
    private var finished = false
    public private(set) var rowsWritten = 0
    public private(set) var bytesWritten: Int64 = 0
    /// Number of record batch messages written.
    public private(set) var batchesWritten = 0

    public init(url: URL, schema: ArrowIPCSchema? = nil) throws {
        self.url = url
        let fm = FileManager.default
        if fm.fileExists(atPath: url.path) { try fm.removeItem(at: url) }
        fm.createFile(atPath: url.path, contents: nil)
        guard let h = FileHandle(forWritingAtPath: url.path) else {
            throw ArrowIPCError.malformed("cannot open \(url.path) for writing")
        }
        self.handle = h
        if let schema { try writeSchema(schema) }
    }

    private func writeSchema(_ s: ArrowIPCSchema) throws {
        // Encoding the schema with no batches gives exactly "schema message + end-of-stream marker".
        let prologue = try ArrowIPCWriter.encode([], schema: s, format: .stream)
        precondition(prologue.count >= 8)
        let body = prologue.prefix(prologue.count - 8)
        handle.write(Data(body))
        bytesWritten += Int64(body.count)
        prologueBytes = body.count
        schema = s
    }

    public func write(_ batch: MetalRecordBatch) throws {
        guard !finished else { throw ArrowIPCError.malformed("write after finish()") }
        let batch = try decodeDictionaries(batch)
        if schema == nil {
            try writeSchema(ArrowIPCSchema(fields: zip(batch.names, batch.columns).map {
                ArrowIPCField(name: $0.0, type: $0.1.ipcType)
            }))
        }
        guard batch.length > 0 else { return }
        let encoded = try ArrowIPCWriter.encode([batch], schema: schema, format: .stream)
        // Trim the schema prologue and the trailing end-of-stream marker: what is left is the one
        // record batch message, which appends cleanly to the stream already on disk.
        guard encoded.count >= prologueBytes + 8 else {
            throw ArrowIPCError.malformed("unexpected IPC encoding for a single batch")
        }
        let msg = encoded[(encoded.startIndex + prologueBytes)..<(encoded.endIndex - 8)]
        handle.write(Data(msg))
        bytesWritten += Int64(msg.count)
        rowsWritten += batch.length
        batchesWritten += 1
    }

    public func finish() throws {
        guard !finished else { return }
        finished = true
        if schema == nil {
            // No batch was ever written and no schema was given: leave an empty file.
            try? handle.close()
            return
        }
        var eos = Data()
        eos.append(contentsOf: withUnsafeBytes(of: UInt32(0xFFFF_FFFF).littleEndian) { Array($0) })
        eos.append(contentsOf: withUnsafeBytes(of: Int32(0).littleEndian) { Array($0) })
        handle.write(eos)
        bytesWritten += 8
        try? handle.close()
    }

    deinit { try? finish() }

    /// Dictionary columns are decoded: an incremental stream cannot promise one dictionary per column.
    private func decodeDictionaries(_ b: MetalRecordBatch) throws -> MetalRecordBatch {
        guard b.columns.contains(where: { if case .dictionary = $0 { return true }; return false }) else { return b }
        return try MetalRecordBatch(names: b.names, columns: b.columns.map { try $0.decode() })
    }
}

/// Round-robins output batches across `parts` IPC stream files. The grace hash join partitions both
/// of its inputs with one of these per side.
public final class PartitionedIPCSink {
    public let urls: [URL]
    private var sinks: [IPCStreamSink?]
    private let schema: ArrowIPCSchema?
    public private(set) var rowsWritten = 0

    public init(directory: URL, prefix: String, parts: Int, schema: ArrowIPCSchema? = nil) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        self.urls = (0..<parts).map { directory.appendingPathComponent("\(prefix)-\(String(format: "%04d", $0)).arrows") }
        self.sinks = Array(repeating: nil, count: parts)
        self.schema = schema
    }

    public func write(part: Int, _ batch: MetalRecordBatch) throws {
        guard batch.length > 0 else { return }
        if sinks[part] == nil { sinks[part] = try IPCStreamSink(url: urls[part], schema: schema) }
        try sinks[part]!.write(batch)
        rowsWritten += batch.length
    }

    /// Closes every partition file. Partitions that never received a row have no file.
    public func finish() throws {
        for s in sinks { try s?.finish() }
    }

    /// The file for `part`, or nil when nothing was written to it.
    public func url(part: Int) -> URL? { sinks[part] == nil ? nil : urls[part] }

    public func removeFiles() {
        for u in urls { try? FileManager.default.removeItem(at: u) }
    }
}

// MARK: - Exporting ArrowMetal's output as an Arrow C Stream

/// A pull-based reader over a streamed query's output batches.
///
/// This is the shape a foreign consumer wants: `get_next` pulls exactly one batch through the whole
/// pipeline, so pyarrow or Polars can iterate an out-of-core ArrowMetal query lazily and never
/// materialise more than one batch.
public protocol StreamBatchReader: AnyObject {
    /// The next output batch, or nil at end of stream.
    func nextOutputBatch() throws -> MetalRecordBatch?
    /// Column names and types of the output, known before the first batch when possible.
    func outputSchema() throws -> MetalRecordBatch?
}

/// Bridges a `StreamBatchReader` (or any `BatchSource`) to the Arrow C Stream ABI.
public final class ArrowStreamExporter {
    let next: () throws -> MetalRecordBatch?
    var schema: (names: [String], columns: [AnyMetalArray])?
    var pending: MetalRecordBatch?
    var lastError: UnsafeMutablePointer<CChar>?
    /// Batches handed out, kept alive until the consumer releases them (the C data interface's
    /// per-array release callbacks own the buffers, so nothing extra is retained here).
    public private(set) var batchesProduced = 0

    public init(_ source: BatchSource) {
        self.next = { [weak source] in try source?.nextBatch() }
    }
    public init(_ reader: StreamBatchReader) {
        self.next = { [weak reader] in try reader?.nextOutputBatch() }
    }
    public init(next: @escaping () throws -> MetalRecordBatch?) { self.next = next }

    deinit { if let e = lastError { free(e) } }

    /// Peeks one batch so the schema is known before `get_schema` is answered.
    func ensureSchema() throws {
        guard schema == nil else { return }
        if let b = try next() {
            pending = b
            schema = (b.names, b.columns)
        } else {
            schema = ([], [])
        }
    }

    func take() throws -> MetalRecordBatch? {
        if let p = pending { pending = nil; return p }
        return try next()
    }

    func setError(_ e: Error) {
        if let old = lastError { free(old) }
        lastError = strdup("\(e)")
    }

    /// Fills `out` with a C Stream that pulls from this exporter. The exporter is retained by the
    /// stream and released when the consumer calls `release`.
    public func export(into out: UnsafeMutablePointer<ArrowArrayStream>) {
        out.pointee.get_schema = { s, schemaOut in
            guard let s, let schemaOut, let pd = s.pointee.private_data else { return EINVAL }
            let e = Unmanaged<ArrowStreamExporter>.fromOpaque(pd).takeUnretainedValue()
            do {
                try e.ensureSchema()
                let names = e.schema?.names ?? []
                let cols = e.schema?.columns ?? []
                let batch = try MetalRecordBatch(names: names, columns: cols)
                batch.exportArrowSchema(into: schemaOut)
                return 0
            } catch { e.setError(error); return EIO }
        }
        out.pointee.get_next = { s, arrayOut in
            guard let s, let arrayOut, let pd = s.pointee.private_data else { return EINVAL }
            let e = Unmanaged<ArrowStreamExporter>.fromOpaque(pd).takeUnretainedValue()
            do {
                try e.ensureSchema()
                guard let b = try e.take() else {
                    arrayOut.pointee.release = nil     // end of stream
                    return 0
                }
                b.exportArrowArray(into: arrayOut)
                e.batchesProduced += 1
                return 0
            } catch { e.setError(error); return EIO }
        }
        out.pointee.get_last_error = { s in
            guard let s, let pd = s.pointee.private_data else { return nil }
            let e = Unmanaged<ArrowStreamExporter>.fromOpaque(pd).takeUnretainedValue()
            return e.lastError.map { UnsafePointer($0) }
        }
        out.pointee.release = { s in
            guard let s, let pd = s.pointee.private_data else { return }
            Unmanaged<ArrowStreamExporter>.fromOpaque(pd).release()
            s.pointee.release = nil
            s.pointee.private_data = nil
        }
        out.pointee.private_data = Unmanaged.passRetained(self).toOpaque()
    }
}
