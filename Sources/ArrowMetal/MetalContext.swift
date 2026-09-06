import Foundation
import Metal

/// Errors raised by ArrowMetal.
public enum ArrowMetalError: Error, CustomStringConvertible {
    case noMetalDevice
    case bufferAllocationFailed(bytes: Int)
    case shaderCompilationFailed(String)
    case pipelineCreationFailed(String)
    case unsupportedType(String)
    case invalidArrowArray(String)
    case lengthMismatch(Int, Int)
    case releasedArray
    /// A checked (overflow-raising) kernel found at least one offending element. `op` is the Arrow
    /// function name, `index` the first offending row when the GPU could report it, and `detail` the
    /// Arrow message ("overflow", "divide by zero", "logarithm of zero", ...).
    case overflow(op: String, index: Int?, detail: String)

    public var description: String {
        switch self {
        case .noMetalDevice: return "No Metal device available"
        case .bufferAllocationFailed(let b): return "Failed to allocate MTLBuffer of \(b) bytes"
        case .shaderCompilationFailed(let s): return "Metal shader compilation failed: \(s)"
        case .pipelineCreationFailed(let s): return "Metal pipeline creation failed: \(s)"
        case .unsupportedType(let s): return "Unsupported Arrow type: \(s)"
        case .invalidArrowArray(let s): return "Invalid ArrowArray: \(s)"
        case .lengthMismatch(let a, let b): return "Array length mismatch: \(a) vs \(b)"
        case .releasedArray: return "ArrowArray has already been released"
        case .overflow(let op, let index, let detail):
            return index.map { "\(op): \(detail) at index \($0)" } ?? "\(op): \(detail)"
        }
    }
}

/// Owns the Metal device, command queue and a cache of compiled compute pipelines.
///
/// Shaders are compiled at runtime from Metal Shading Language source, so the package
/// works with Command Line Tools alone (no offline `metal` compiler required).
public final class MetalContext: @unchecked Sendable {
    public let device: MTLDevice
    public let queue: MTLCommandQueue

    private let lock = NSLock()
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private var libraries: [String: MTLLibrary] = [:]

    /// True on virtualised GPUs (GitHub-hosted runners) where Metal pipeline creation is unreliable.
    public var isVirtualDevice: Bool { device.name.localizedCaseInsensitiveContains("paravirtual") }

    /// Set ARROWMETAL_DEBUG_SHADERS=1 to dump generated MSL and full compiler diagnostics on failure.
    static let debugShaders = ProcessInfo.processInfo.environment["ARROWMETAL_DEBUG_SHADERS"] != nil

    /// Full NSError description including the compiler log Metal hides in userInfo.
    static func describe(_ error: Error) -> String {
        let ns = error as NSError
        var parts = [ns.localizedDescription]
        for (k, v) in ns.userInfo where k != NSLocalizedDescriptionKey { parts.append("\(k)=\(v)") }
        return parts.joined(separator: " | ")
    }

    /// Process-wide default context on the system default device.
    public static let shared: MetalContext = {
        do { return try MetalContext() } catch { fatalError("\(error)") }
    }()

    /// Recycles page-aligned buffers so repeated kernels do not pay mmap and page-fault costs.
    public let pool: BufferPool

    public init(device: MTLDevice? = nil, poolLimitBytes: Int? = nil) throws {
        guard let dev = device ?? MTLCreateSystemDefaultDevice() else { throw ArrowMetalError.noMetalDevice }
        guard let q = dev.makeCommandQueue() else { throw ArrowMetalError.noMetalDevice }
        self.device = dev
        self.queue = q
        // Default cap: a quarter of the recommended working set, at most 8 GB.
        let cap = poolLimitBytes ?? Swift.min(Int(dev.recommendedMaxWorkingSetSize) / 4, 8 << 30)
        self.pool = BufferPool(limitBytes: cap)
        self.pool.context = self
    }

    /// Returns a compiled compute pipeline for `function` inside `source`, compiling and caching on first use.
    public func pipeline(source: String, function: String, cacheKey: String) throws -> MTLComputePipelineState {
        lock.lock(); defer { lock.unlock() }
        if let p = pipelines[cacheKey] { return p }
        let lib: MTLLibrary
        if let l = libraries[source] {
            lib = l
        } else {
            let opts = MTLCompileOptions()
            if #available(macOS 15.0, iOS 18.0, *) { opts.mathMode = .safe } else { opts.fastMathEnabled = false }
            do { lib = try device.makeLibrary(source: source, options: opts) }
            catch {
                let detail = Self.describe(error)
                if Self.debugShaders {
                    FileHandle.standardError.write("ArrowMetal: shader compilation failed on \(device.name): \(detail)\n--- source ---\n\(source)\n--- end ---\n".data(using: .utf8)!)
                }
                throw ArrowMetalError.shaderCompilationFailed(detail)
            }
            libraries[source] = lib
        }
        guard let fn = lib.makeFunction(name: function) else {
            throw ArrowMetalError.pipelineCreationFailed("function \(function) not found")
        }
        do {
            let p: MTLComputePipelineState
            do { p = try device.makeComputePipelineState(function: fn) }
            catch {
                // Virtualised GPUs (GitHub's "Apple Paravirtual device") fail pipeline creation sporadically; retry once.
                usleep(20_000)
                p = try device.makeComputePipelineState(function: fn)
            }
            pipelines[cacheKey] = p
            return p
        } catch {
            let detail = Self.describe(error)
            if Self.debugShaders {
                FileHandle.standardError.write("ArrowMetal: pipeline creation failed for \(function) on \(device.name): \(detail)\n--- source ---\n\(source)\n--- end ---\n".data(using: .utf8)!)
            }
            throw ArrowMetalError.pipelineCreationFailed("\(function) on \(device.name): \(detail)")
        }
    }

    /// Encodes `body` into a fresh command buffer, commits it and blocks until the GPU finishes.
    ///
    /// Latency: the command buffer skips resource retain/release (every buffer bound here is owned by the
    /// caller for the duration of the call), and the wait spins briefly before blocking, which saves roughly
    /// 30 to 40 µs per call on Apple silicon compared with `waitUntilCompleted` alone.
    @discardableResult
    public func run(_ body: (MTLComputeCommandEncoder) throws -> Void) throws -> MTLCommandBuffer? {
        if let b = currentBatch {
            // Batched: append to the open command buffer; the serial encoder orders dispatches.
            try body(b.encoder)
            b.encoder.memoryBarrier(scope: .buffers)
            return nil
        }
        guard let cb = queue.makeCommandBufferWithUnretainedReferences(), let enc = cb.makeComputeCommandEncoder() else {
            throw ArrowMetalError.noMetalDevice
        }
        try body(enc)
        enc.endEncoding()
        cb.commit()
        wait(cb)
        if let err = cb.error { throw ArrowMetalError.pipelineCreationFailed("command buffer failed: \(err)") }
        return cb
    }

    // MARK: Batching

    /// One open command buffer per thread plus the fix-ups to run once it completes.
    final class Batch {
        let commandBuffer: MTLCommandBuffer
        let encoder: MTLComputeCommandEncoder
        var afterFlush: [() throws -> Void] = []
        /// Buffers and arrays that must stay alive until the GPU has finished with them.
        var retained: [AnyObject] = []
        init(commandBuffer: MTLCommandBuffer, encoder: MTLComputeCommandEncoder) { self.commandBuffer = commandBuffer; self.encoder = encoder }
    }

    private static let batchKey = "ArrowMetal.batch"
    var currentBatch: Batch? {
        get { Thread.current.threadDictionary[Self.batchKey] as? Batch }
        set { Thread.current.threadDictionary[Self.batchKey] = newValue }
    }
    /// Number of open batches across threads; while non-zero the pool parks returned buffers.
    let openBatches = ManagedAtomicCounter()
    public var isBatching: Bool { currentBatch != nil }

    /// Runs `body` with every kernel appended to one command buffer. The GPU runs once, at the end or at
    /// the first CPU-side read (a reduction result, a filtered length, an export). Nested calls join the
    /// outer batch. Errors from deferred checks (for example an out-of-range `take` index) surface here.
    public func batch<R>(_ body: () throws -> R) throws -> R {
        if currentBatch != nil { return try body() }
        try openBatch()
        var result: R
        do { result = try body() } catch { try? flush(); throw error }
        try flush()
        return result
    }

    /// Explicit form of `batch { }` for foreign-language callers. Must be paired on the same thread.
    public func beginBatch() throws { if currentBatch == nil { try openBatch() } }
    public func endBatch() throws { try flush() }

    func openBatch() throws {
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw ArrowMetalError.noMetalDevice }
        currentBatch = Batch(commandBuffer: cb, encoder: enc)
        openBatches.increment()
    }

    /// Ends encoding on the open batch and detaches it from this thread, *without* committing.
    /// Returns nil when nothing is open. The caller owns the batch from here: it must commit
    /// `b.commandBuffer` and, once that buffer has completed, call `finishBatch(_:)` exactly once.
    /// (`flush` does both back to back; `batchAsync` puts a completion handler in between.)
    func detachBatch() -> Batch? {
        guard let b = currentBatch else { return nil }
        currentBatch = nil
        b.encoder.endEncoding()
        return b
    }

    /// Post-completion half of `flush`: pool bookkeeping and the deferred fix-ups (lengths, null counts,
    /// `take` bounds errors). Call exactly once, after `b.commandBuffer` has completed. Touches no
    /// thread-local state, so it is safe from a command buffer completion handler.
    func finishBatch(_ b: Batch) throws {
        openBatches.decrement()
        pool.releaseParked()
        var firstError: Error? = nil
        if let err = b.commandBuffer.error { firstError = ArrowMetalError.pipelineCreationFailed("command buffer failed: \(err)") }
        for f in b.afterFlush { do { try f() } catch { if firstError == nil { firstError = error } } }
        b.retained.removeAll()
        if let e = firstError { throw e }
    }

    /// Commits the open batch (if any), waits, runs deferred fix-ups, and reopens a fresh batch if `reopen`.
    /// Called automatically by any CPU-side read of a pending result.
    public func flush(reopen: Bool = false) throws {
        guard let b = detachBatch() else { return }
        b.commandBuffer.commit()
        wait(b.commandBuffer)
        var firstError: Error? = nil
        do { try finishBatch(b) } catch { firstError = error }
        if reopen { try openBatch() }
        if let e = firstError { throw e }
    }

    /// If a batch is open: flush it and reopen, so the caller can read GPU results now and keep batching after.
    func syncPoint() throws { if currentBatch != nil { try flush(reopen: true) } }

    /// Registers work to run after the current batch completes, or runs it now when not batching.
    func afterFlush(_ f: @escaping () throws -> Void) throws {
        if let b = currentBatch { b.afterFlush.append(f) } else { try f() }
    }
    /// Keeps `o` alive until the current batch has executed (no-op when not batching).
    func retainUntilFlush(_ o: AnyObject) { currentBatch?.retained.append(o) }

    /// Spin for up to `spinMicroseconds`, then block. Spinning avoids a scheduler round trip for short kernels.
    public var spinMicroseconds: UInt64 = 300
    func wait(_ cb: MTLCommandBuffer) {
        let deadline = DispatchTime.now().uptimeNanoseconds + spinMicroseconds * 1000
        while cb.status != .completed {
            if cb.status == .error { return }
            if DispatchTime.now().uptimeNanoseconds > deadline { cb.waitUntilCompleted(); return }
        }
    }
}

final class ManagedAtomicCounter: @unchecked Sendable {
    private var v = 0
    private let lock = NSLock()
    func increment() { lock.lock(); v += 1; lock.unlock() }
    func decrement() { lock.lock(); v -= 1; lock.unlock() }
    var value: Int { lock.lock(); defer { lock.unlock() }; return v }
}

/// Size-bucketed free list of shared `MTLBuffer`s. Exact-length matches only (lengths are page multiples,
/// so repeated same-size allocations, the common case in a pipeline, always hit).
public final class BufferPool: @unchecked Sendable {
    public let limitBytes: Int
    private let lock = NSLock()
    private var free: [Int: [MTLBuffer]] = [:]
    private var order: [Int] = []          // lengths in insertion order for eviction
    public private(set) var pooledBytes = 0

    init(limitBytes: Int) { self.limitBytes = limitBytes }

    func take(length: Int) -> MTLBuffer? {
        lock.lock(); defer { lock.unlock() }
        guard var list = free[length], let b = list.popLast() else { return nil }
        free[length] = list.isEmpty ? nil : list
        pooledBytes -= length
        return b
    }

    /// Buffers returned while a batch is open; they may still be referenced by pending GPU work.
    private var parked: [MTLBuffer] = []
    weak var context: MetalContext?

    func releaseParked() {
        lock.lock()
        let p = parked; parked.removeAll()
        lock.unlock()
        for b in p { give(b) }
    }

    func give(_ b: MTLBuffer) {
        lock.lock(); defer { lock.unlock() }
        if let ctx = context, ctx.openBatches.value > 0 { parked.append(b); return }
        let len = b.length
        if len > limitBytes { return }
        while pooledBytes + len > limitBytes, let evictLen = order.first {
            order.removeFirst()
            if var list = free[evictLen], let _ = list.popLast() {
                free[evictLen] = list.isEmpty ? nil : list
                pooledBytes -= evictLen
            }
        }
        free[len, default: []].append(b)
        order.append(len)
        pooledBytes += len
    }

    /// Releases every pooled buffer back to the OS.
    public func drain() {
        lock.lock(); defer { lock.unlock() }
        free.removeAll(); order.removeAll(); pooledBytes = 0
    }
}
