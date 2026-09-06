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

    /// Process-wide default context on the system default device.
    public static let shared: MetalContext = {
        do { return try MetalContext() } catch { fatalError("\(error)") }
    }()

    public init(device: MTLDevice? = nil) throws {
        guard let dev = device ?? MTLCreateSystemDefaultDevice() else { throw ArrowMetalError.noMetalDevice }
        guard let q = dev.makeCommandQueue() else { throw ArrowMetalError.noMetalDevice }
        self.device = dev
        self.queue = q
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
            catch { throw ArrowMetalError.shaderCompilationFailed("\(error)") }
            libraries[source] = lib
        }
        guard let fn = lib.makeFunction(name: function) else {
            throw ArrowMetalError.pipelineCreationFailed("function \(function) not found")
        }
        do {
            let p = try device.makeComputePipelineState(function: fn)
            pipelines[cacheKey] = p
            return p
        } catch { throw ArrowMetalError.pipelineCreationFailed("\(error)") }
    }

    /// Encodes `body` into a fresh command buffer, commits it and blocks until the GPU finishes.
    @discardableResult
    public func run(_ body: (MTLComputeCommandEncoder) throws -> Void) throws -> MTLCommandBuffer {
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw ArrowMetalError.noMetalDevice
        }
        try body(enc)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { throw ArrowMetalError.pipelineCreationFailed("command buffer failed: \(err)") }
        return cb
    }
}
