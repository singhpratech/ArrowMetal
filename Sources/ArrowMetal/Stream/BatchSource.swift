import Foundation
import Metal
import CArrowABI

// Sources: where a streamed query's rows come from. Everything downstream — the executor, the
// operators, the sinks — only ever sees `BatchSource.nextBatch()`, so an Arrow IPC file, a directory
// of them, a pyarrow dataset scanner arriving over the Arrow C Stream interface and an in-memory
// chunked table are interchangeable.
//
// The read side is where an out-of-core query spends its time, so this file is where the pipeline's
// first stage lives: `PrefetchingSource` runs the wrapped source on its own thread, keeps a bounded
// number of batches in flight, and blocks the producer when the consumer falls behind (backpressure
// by both batch count and bytes). Buffers come from `MetalContext.pool`, which recycles page-aligned
// shared-memory allocations of the same size — so a steady stream of equally shaped batches
// allocates once and then reuses, instead of mapping and faulting fresh pages per batch.

/// A pull-based source of Metal-resident record batches.
public protocol BatchSource: AnyObject {
    /// The schema every batch shares, when the source knows it before the first batch.
    var streamSchema: ArrowIPCSchema? { get }
    /// The next batch, or nil at end of stream. Called from one thread at a time.
    func nextBatch() throws -> MetalRecordBatch?
    /// Releases file handles / foreign streams. Safe to call more than once.
    func close()
    /// Bytes read from storage so far (0 for in-memory sources), for the GB/s figure.
    var bytesRead: Int64 { get }
    /// Total bytes the source expects to read, or 0 when unknown.
    var totalBytes: Int64 { get }
}

extension BatchSource {
    public var streamSchema: ArrowIPCSchema? { nil }
    public var bytesRead: Int64 { 0 }
    public var totalBytes: Int64 { 0 }

    /// Drains the source. Only for tests and small inputs — the point of the package is not to do this.
    public func collect() throws -> [MetalRecordBatch] {
        var out: [MetalRecordBatch] = []
        while let b = try nextBatch() { out.append(b) }
        return out
    }
}

// MARK: - In-memory

/// A source over batches already in Metal memory (a chunked table).
public final class ChunkedTableSource: BatchSource {
    private var batches: [MetalRecordBatch]
    private var i = 0
    public let streamSchema: ArrowIPCSchema?

    public init(_ batches: [MetalRecordBatch], schema: ArrowIPCSchema? = nil) {
        self.batches = batches
        self.streamSchema = schema ?? batches.first.map { b in
            ArrowIPCSchema(fields: zip(b.names, b.columns).map { ArrowIPCField(name: $0.0, type: $0.1.ipcType) })
        }
    }

    public func nextBatch() throws -> MetalRecordBatch? {
        guard i < batches.count else { return nil }
        defer { i += 1 }
        return batches[i]
    }
    public func close() { batches = []; i = 0 }
}

// MARK: - Arrow IPC files

/// Streams one Arrow IPC file, one record batch at a time.
///
/// The file is memory mapped (`ArrowIPCReader(url:)` maps it and borrows page-aligned body buffers
/// without a copy where the file's layout allows; Arrow only guarantees 8-byte body alignment, so a
/// buffer that does not start on a page boundary is copied into shared memory instead).
///
/// Readahead is issued with `fcntl(F_RDADVISE)` on a second descriptor: the mapped pages and the
/// descriptor share the unified buffer cache, so warming the cache ahead of the read cursor pulls the
/// next batches off the SSD while the GPU is busy with the current one. `F_RDAHEAD` is also enabled,
/// which is macOS's sequential-access hint.
public final class IPCFileSource: BatchSource {
    private let reader: ArrowIPCReader
    private var index = 0
    private var fd: Int32 = -1
    private let fileSize: Int64
    private var read: Int64 = 0
    /// How far ahead of the read cursor to warm the buffer cache, in bytes.
    public var readaheadBytes: Int = 64 << 20
    private var advisedTo: Int64 = 0
    public let url: URL

    public var streamSchema: ArrowIPCSchema? { reader.schema }
    public var bytesRead: Int64 { read }
    public var totalBytes: Int64 { fileSize }
    /// Number of record batches in the file (the IPC file format indexes them).
    public var batchCount: Int { reader.batchCount }
    /// True when the last batch borrowed mapped pages instead of copying them.
    public var lastBatchWasZeroCopy: Bool { reader.lastBatchWasZeroCopy }

    public init(url: URL, context: MetalContext = .shared) throws {
        self.url = url
        self.reader = try ArrowIPCReader(url: url, context: context)
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        self.fileSize = (attrs[.size] as? NSNumber)?.int64Value ?? 0
        self.fd = open(url.path, O_RDONLY)
        if fd >= 0 {
            _ = fcntl(fd, F_RDAHEAD, 1)
            advise(upTo: Int64(readaheadBytes))
        }
    }

    /// Warms the unified buffer cache for `[advisedTo, end)`.
    private func advise(upTo end: Int64) {
        guard fd >= 0, end > advisedTo, fileSize > 0 else { return }
        let start = advisedTo
        let count = Swift.min(end, fileSize) - start
        guard count > 0 else { return }
        var r = radvisory(ra_offset: off_t(start), ra_count: Int32(Swift.min(count, Int64(Int32.max))))
        _ = fcntl(fd, F_RDADVISE, &r)
        advisedTo = start + count
    }

    public func nextBatch() throws -> MetalRecordBatch? {
        guard index < reader.batchCount else { return nil }
        let b = try reader.batch(at: index)
        index += 1
        // Approximate the read cursor from the batch index (the reader does not expose block offsets)
        // and keep the cache warm one readahead window ahead of it.
        if reader.batchCount > 0 {
            read = fileSize * Int64(index) / Int64(reader.batchCount)
            advise(upTo: read + Int64(readaheadBytes))
        }
        return b
    }

    public func close() {
        if fd >= 0 { Darwin.close(fd); fd = -1 }
    }
    deinit { close() }
}

/// Streams a directory of Arrow IPC files as one dataset, in sorted filename order.
///
/// Files are opened lazily, one at a time, so a directory larger than memory never has more than one
/// file mapped. Every file must share a schema; the first file's schema is the dataset's.
public final class IPCDirectorySource: BatchSource {
    public let urls: [URL]
    private var fileIndex = -1
    private var current: IPCFileSource?
    private var schema: ArrowIPCSchema?
    private let context: MetalContext
    private var readBefore: Int64 = 0
    private let total: Int64

    public var streamSchema: ArrowIPCSchema? { schema }
    public var bytesRead: Int64 { readBefore + (current?.bytesRead ?? 0) }
    public var totalBytes: Int64 { total }

    /// Every `.arrow` / `.arrows` / `.ipc` / `.feather` file directly inside `directory`, sorted by name.
    public init(directory: URL, extensions: Set<String> = ["arrow", "arrows", "ipc", "feather"],
                context: MetalContext = .shared) throws {
        let fm = FileManager.default
        let entries = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.fileSizeKey])
        self.urls = entries.filter { extensions.contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !urls.isEmpty else {
            throw ArrowIPCError.malformed("no Arrow IPC files in \(directory.path)")
        }
        self.context = context
        var t: Int64 = 0
        for u in urls { t += ((try? fm.attributesOfItem(atPath: u.path))?[.size] as? NSNumber)?.int64Value ?? 0 }
        self.total = t
    }

    public init(files: [URL], context: MetalContext = .shared) throws {
        guard !files.isEmpty else { throw ArrowIPCError.malformed("no Arrow IPC files given") }
        self.urls = files
        self.context = context
        var t: Int64 = 0
        for u in files { t += ((try? FileManager.default.attributesOfItem(atPath: u.path))?[.size] as? NSNumber)?.int64Value ?? 0 }
        self.total = t
    }

    public func nextBatch() throws -> MetalRecordBatch? {
        while true {
            if let c = current, let b = try c.nextBatch() { return b }
            if let c = current { readBefore += c.totalBytes; c.close() }
            current = nil
            fileIndex += 1
            guard fileIndex < urls.count else { return nil }
            let s = try IPCFileSource(url: urls[fileIndex], context: context)
            if schema == nil { schema = s.streamSchema }
            current = s
        }
    }

    public func close() {
        current?.close()
        current = nil
    }
}

// MARK: - Arrow C Stream interface

/// Pulls batches from a foreign `ArrowArrayStream` — a `pyarrow.RecordBatchReader`, a
/// `pyarrow.dataset` scanner (so Parquet, CSV and partitioned datasets work today through pyarrow's
/// readers), a Polars `LazyFrame`, DuckDB, or anything else that speaks the C Stream ABI.
///
/// The stream is released on `close()` or when this object is deallocated. Buffers the producer hands
/// over are imported zero-copy when they are page aligned, and copied into shared memory otherwise.
public final class CStreamSource: BatchSource {
    private let stream: UnsafeMutablePointer<ArrowArrayStream>
    private var schemaStruct = ArrowSchema()
    private var haveSchema = false
    private var closed = false
    private let context: MetalContext
    private let ownsStream: Bool
    public private(set) var zeroCopyBatches = 0
    public private(set) var copiedBatches = 0

    /// Takes ownership of `stream` (it is released here, and the caller must not use it again).
    public init(_ stream: UnsafeMutablePointer<ArrowArrayStream>, context: MetalContext = .shared,
                takingOwnership: Bool = true) throws {
        guard stream.pointee.release != nil else { throw ArrowMetalError.releasedArray }
        // Move the producer out of the caller's struct so its lifetime is ours alone.
        let owned = UnsafeMutablePointer<ArrowArrayStream>.allocate(capacity: 1)
        if takingOwnership {
            owned.initialize(to: stream.pointee)
            stream.pointee.release = nil
        } else {
            owned.initialize(to: stream.pointee)
        }
        self.stream = owned
        self.ownsStream = takingOwnership
        self.context = context
        guard owned.pointee.get_schema(owned, &schemaStruct) == 0 else {
            throw ArrowMetalError.invalidArrowArray(lastError() ?? "get_schema failed")
        }
        haveSchema = true
    }

    private func lastError() -> String? {
        stream.pointee.get_last_error.flatMap { $0(stream) }.map { String(cString: $0) }
    }

    public var streamSchema: ArrowIPCSchema? { nil }

    public func nextBatch() throws -> MetalRecordBatch? {
        guard !closed else { return nil }
        var arr = ArrowArray()
        guard stream.pointee.get_next(stream, &arr) == 0 else {
            throw ArrowMetalError.invalidArrowArray(lastError() ?? "get_next failed")
        }
        if arr.release == nil { return nil }        // end of stream
        let r = try importArrowRecordBatch(schema: &schemaStruct, array: &arr, context: context)
        if r.zeroCopy { zeroCopyBatches += 1 } else { copiedBatches += 1 }
        return r.batch
    }

    public func close() {
        guard !closed else { return }
        closed = true
        if haveSchema { schemaStruct.release?(&schemaStruct); haveSchema = false }
        if ownsStream { stream.pointee.release?(stream) }
        stream.deallocate()
    }
    deinit { close() }
}

// MARK: - Prefetching: stage one of the pipeline

/// Runs a source on its own thread and keeps up to `depth` batches ready, so the GPU never waits on
/// the SSD (and the SSD never runs ahead of memory).
///
/// Backpressure is two-sided: the reader blocks when `depth` batches are queued *or* when the queued
/// batches exceed `budgetBytes`, whichever comes first, so a stream of very wide batches cannot pin
/// more memory than the budget. Consumed batches are released as soon as the executor is done with
/// them and their shared-memory buffers go back to `MetalContext.pool`, where the next read of the
/// same shape picks them up again — the "bounded pool of shared-memory batch buffers".
public final class PrefetchingSource: BatchSource {
    private let inner: BatchSource
    private let depth: Int
    private let budgetBytes: Int

    private let lock = NSCondition()
    private var queue: [MetalRecordBatch] = []
    private var queuedBytes = 0
    private var done = false
    private var failure: Error?
    private var stopped = false
    private var thread: Thread?

    /// Nanoseconds the reader thread spent inside the wrapped source (stage-one busy time).
    public private(set) var readNanos: UInt64 = 0
    /// Nanoseconds the consumer spent waiting for a batch that was not ready (the pipeline stall).
    public private(set) var stallNanos: UInt64 = 0

    public var streamSchema: ArrowIPCSchema? { inner.streamSchema }
    public var bytesRead: Int64 { inner.bytesRead }
    public var totalBytes: Int64 { inner.totalBytes }

    public init(_ inner: BatchSource, depth: Int = 3, budgetBytes: Int = 2 << 30) {
        self.inner = inner
        self.depth = Swift.max(1, depth)
        self.budgetBytes = Swift.max(1 << 20, budgetBytes)
    }

    private func start() {
        guard thread == nil else { return }
        let t = Thread { [weak self] in self?.readLoop() }
        t.name = "ArrowMetal.prefetch"
        t.stackSize = 1 << 20
        thread = t
        t.start()
    }

    private func readLoop() {
        while true {
            lock.lock()
            while !stopped && (queue.count >= depth || queuedBytes >= budgetBytes) { lock.wait() }
            if stopped { lock.unlock(); return }
            lock.unlock()

            let t0 = machNow()
            do {
                guard let b = try inner.nextBatch() else {
                    lock.lock(); done = true; lock.broadcast(); lock.unlock()
                    return
                }
                readNanos &+= nanos(since: t0)
                lock.lock()
                queue.append(b)
                queuedBytes += batchBytes(b)
                lock.broadcast()
                lock.unlock()
            } catch {
                lock.lock(); failure = error; done = true; lock.broadcast(); lock.unlock()
                return
            }
        }
    }

    public func nextBatch() throws -> MetalRecordBatch? {
        start()
        let t0 = machNow()
        lock.lock()
        while queue.isEmpty && !done { lock.wait() }
        let waited = nanos(since: t0)
        if let e = failure, queue.isEmpty { lock.unlock(); throw e }
        guard !queue.isEmpty else { lock.unlock(); return nil }
        let b = queue.removeFirst()
        queuedBytes -= batchBytes(b)
        lock.broadcast()
        lock.unlock()
        stallNanos &+= waited
        return b
    }

    public func close() {
        lock.lock(); stopped = true; lock.broadcast(); lock.unlock()
        inner.close()
    }
}

/// Approximate shared-memory footprint of a batch, for the prefetch byte budget.
func batchBytes(_ b: MetalRecordBatch) -> Int {
    var n = 0
    for c in b.columns { n += columnBytes(c) }
    return n
}

func columnBytes(_ c: AnyMetalArray) -> Int {
    switch c {
    case .int8(let a): return a.values.byteCount + (a.validity?.byteCount ?? 0)
    case .uint8(let a): return a.values.byteCount + (a.validity?.byteCount ?? 0)
    case .int16(let a): return a.values.byteCount + (a.validity?.byteCount ?? 0)
    case .uint16(let a): return a.values.byteCount + (a.validity?.byteCount ?? 0)
    case .int32(let a): return a.values.byteCount + (a.validity?.byteCount ?? 0)
    case .uint32(let a): return a.values.byteCount + (a.validity?.byteCount ?? 0)
    case .int64(let a): return a.values.byteCount + (a.validity?.byteCount ?? 0)
    case .uint64(let a): return a.values.byteCount + (a.validity?.byteCount ?? 0)
    case .float32(let a): return a.values.byteCount + (a.validity?.byteCount ?? 0)
    case .float64(let a): return a.values.byteCount + (a.validity?.byteCount ?? 0)
    case .boolean(let a): return a.values.byteCount + (a.validity?.byteCount ?? 0)
    case .string(let a), .binary(let a):
        return a.offsets.byteCount + a.data.byteCount + (a.validity?.byteCount ?? 0)
    case .temporal(let a): return a.values.byteCount + (a.validity?.byteCount ?? 0)
    case .dictionary(let codes, let values): return codes.values.byteCount + columnBytes(values)
    default: return c.length * 8
    }
}

/// Nanoseconds since a `machNow()` tick.
@inline(__always) func nanos(since t0: UInt64) -> UInt64 {
    let d = machNow() &- t0
    return d &* UInt64(machTimebase.numer) / UInt64(machTimebase.denom)
}
let machTimebase: mach_timebase_info_data_t = {
    var t = mach_timebase_info_data_t()
    mach_timebase_info(&t)
    return t
}()

// MARK: - Convenience constructors

extension BatchSource where Self == IPCFileSource {
    /// A prefetching source over a file or a directory of Arrow IPC files.
    public static func ipc(_ path: String, prefetchDepth: Int = 3,
                           context: MetalContext = .shared) throws -> BatchSource {
        try openIPCSource(path, prefetchDepth: prefetchDepth, context: context)
    }
}

/// A prefetching source over an Arrow IPC file or a directory of them.
///
/// `readers > 1` on a directory reads several files at once with `ParallelIPCSource`, which scales
/// the read stage with cores but does not preserve batch order (see that type). Everything else
/// keeps the source's order.
public func openIPCSource(_ path: String, prefetchDepth: Int = 3,
                          budgetBytes: Int = 2 << 30,
                          readers: Int = 1,
                          context: MetalContext = .shared) throws -> BatchSource {
    var isDir: ObjCBool = false
    guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir) else {
        throw ArrowIPCError.malformed("no such path: \(path)")
    }
    let url = URL(fileURLWithPath: path)
    if isDir.boolValue, readers > 1 {
        return try ParallelIPCSource(directory: url, readers: readers,
                                     depth: Swift.max(1, prefetchDepth), context: context)
    }
    let base: BatchSource = isDir.boolValue ? try IPCDirectorySource(directory: url, context: context)
                                            : try IPCFileSource(url: url, context: context)
    return prefetchDepth <= 0 ? base : PrefetchingSource(base, depth: prefetchDepth, budgetBytes: budgetBytes)
}

// MARK: - Parallel reading

/// Reads a directory of Arrow IPC files with several threads at once.
///
/// `PrefetchingSource` hides the read behind the GPU, but it is still *one* thread doing the mapping
/// and the import. On a machine whose page cache already holds the dataset that thread is the
/// bottleneck: one core copies at roughly 10 GB/s while the whole chip can do several times that.
/// This source gives each of `readers` threads its own files and its own `IPCFileSource`, so the
/// read stage scales with cores.
///
/// **Batch order is not preserved.** Batches arrive interleaved across files, in whatever order the
/// threads finish. Every streaming operator here is order independent — aggregates, sketches,
/// group-by, top-k, the external sort's run generation, both joins — but a `filter -> sink_ipc` that
/// must preserve the source's row order needs `readers: 1`.
public final class ParallelIPCSource: BatchSource {
    private let urls: [URL]
    private let context: MetalContext
    private let readers: Int
    private let depth: Int
    private let budgetBytes: Int

    private let lock = NSCondition()
    private var queue: [MetalRecordBatch] = []
    private var queuedBytes = 0
    private var nextFile = 0
    private var live = 0
    private var failure: Error?
    private var stopped = false
    private var started = false
    private var read: Int64 = 0
    private let total: Int64
    private var schemaFound: ArrowIPCSchema?

    /// Nanoseconds summed across the reader threads, so it can exceed wall time — which is the point.
    public private(set) var readNanos: UInt64 = 0

    public var streamSchema: ArrowIPCSchema? { lock.lock(); defer { lock.unlock() }; return schemaFound }
    public var bytesRead: Int64 { lock.lock(); defer { lock.unlock() }; return read }
    public var totalBytes: Int64 { total }

    public init(files: [URL], readers: Int = 4, depth: Int = 2, budgetBytes: Int = 2 << 30,
                context: MetalContext = .shared) throws {
        guard !files.isEmpty else { throw ArrowIPCError.malformed("no Arrow IPC files given") }
        self.urls = files
        self.context = context
        self.readers = Swift.max(1, Swift.min(readers, files.count))
        self.depth = Swift.max(1, depth)
        self.budgetBytes = Swift.max(1 << 20, budgetBytes)
        var t: Int64 = 0
        for u in files {
            t += ((try? FileManager.default.attributesOfItem(atPath: u.path))?[.size] as? NSNumber)?.int64Value ?? 0
        }
        self.total = t
    }

    public convenience init(directory: URL, readers: Int = 4, depth: Int = 2,
                            extensions: Set<String> = ["arrow", "arrows", "ipc", "feather"],
                            context: MetalContext = .shared) throws {
        let entries = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        let files = entries.filter { extensions.contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw ArrowIPCError.malformed("no Arrow IPC files in \(directory.path)") }
        try self.init(files: files, readers: readers, depth: depth, context: context)
    }

    private func start() {
        guard !started else { return }
        started = true
        live = readers
        for i in 0..<readers {
            let t = Thread { [weak self] in self?.readLoop() }
            t.name = "ArrowMetal.prefetch.\(i)"
            t.stackSize = 1 << 20
            t.start()
        }
    }

    /// Claims the next unread file, or nil when they are all taken.
    private func claimFile() -> URL? {
        lock.lock(); defer { lock.unlock() }
        guard nextFile < urls.count, !stopped else { return nil }
        defer { nextFile += 1 }
        return urls[nextFile]
    }

    private func readLoop() {
        while let url = claimFile() {
            do {
                let source = try IPCFileSource(url: url, context: context)
                while true {
                    lock.lock()
                    // The queue is shared, so the depth is per reader and multiplied out here.
                    while !stopped && (queue.count >= depth * readers || queuedBytes >= budgetBytes) { lock.wait() }
                    if stopped { lock.unlock(); source.close(); finishReader(); return }
                    if schemaFound == nil { schemaFound = source.streamSchema }
                    lock.unlock()

                    let t0 = machNow()
                    guard let b = try source.nextBatch() else { break }
                    let took = nanos(since: t0)

                    lock.lock()
                    readNanos &+= took
                    queue.append(b)
                    queuedBytes += batchBytes(b)
                    lock.broadcast()
                    lock.unlock()
                }
                lock.lock(); read += source.totalBytes; lock.unlock()
                source.close()
            } catch {
                lock.lock()
                if failure == nil { failure = error }
                stopped = true
                lock.broadcast()
                lock.unlock()
                finishReader()
                return
            }
        }
        finishReader()
    }

    private func finishReader() {
        lock.lock()
        live -= 1
        lock.broadcast()
        lock.unlock()
    }

    public func nextBatch() throws -> MetalRecordBatch? {
        lock.lock()
        start()
        while queue.isEmpty && live > 0 { lock.wait() }
        if let e = failure, queue.isEmpty { lock.unlock(); throw e }
        guard !queue.isEmpty else { lock.unlock(); return nil }
        let b = queue.removeFirst()
        queuedBytes -= batchBytes(b)
        lock.broadcast()
        lock.unlock()
        return b
    }

    public func close() {
        lock.lock(); stopped = true; lock.broadcast(); lock.unlock()
    }
}
