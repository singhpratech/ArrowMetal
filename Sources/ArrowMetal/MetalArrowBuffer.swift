import Foundation
import Metal

/// The page size Metal requires for `makeBuffer(bytesNoCopy:)` on Apple silicon.
@inline(__always) func metalPageSize() -> Int { Int(getpagesize()) }

@inline(__always) func roundUp(_ n: Int, to align: Int) -> Int { (n + align - 1) / align * align }

/// An Arrow buffer whose bytes live in an `MTLBuffer` with `.storageModeShared`.
///
/// On Apple silicon the CPU and GPU share one physical memory, so the same bytes are
/// simultaneously a valid Arrow C Data Interface buffer (CPU) and a valid
/// C Device Data Interface buffer (`ARROW_DEVICE_METAL`). No copies in either direction.
public final class MetalArrowBuffer: @unchecked Sendable {
    public let mtl: MTLBuffer
    /// Number of meaningful bytes (Arrow semantic length). `mtl.length` may be larger due to padding.
    public let byteCount: Int
    /// Byte offset into `mtl` where this buffer's data begins (0 unless wrapping a sub-range).
    public let offset: Int
    /// Object kept alive for the lifetime of this buffer when memory is borrowed (zero-copy import).
    private let keepAlive: AnyObject?

    /// Whether the memory was borrowed without a copy from an external owner.
    public var isBorrowed: Bool { keepAlive != nil }

    public init(mtl: MTLBuffer, byteCount: Int, offset: Int = 0, keepAlive: AnyObject? = nil) {
        precondition(offset + byteCount <= mtl.length)
        self.mtl = mtl
        self.byteCount = byteCount
        self.offset = offset
        self.keepAlive = keepAlive
    }

    /// Allocates a zeroed shared buffer of `byteCount` bytes.
    ///
    /// Memory is page aligned (pointer and length) and wrapped with `makeBuffer(bytesNoCopy:)`.
    /// This guarantees that any pointer ArrowMetal hands out through the C Data / C Device interfaces
    /// can be re-wrapped as an `MTLBuffer` by a foreign Metal consumer without a copy, and that
    /// bitmap kernels may safely read whole trailing 32-bit words.
    public static func allocate(byteCount: Int, context: MetalContext = .shared) throws -> MetalArrowBuffer {
        let page = metalPageSize()
        let padded = max(roundUp(byteCount, to: page), page)
        var raw: UnsafeMutableRawPointer? = nil
        guard posix_memalign(&raw, page, padded) == 0, let mem = raw else {
            throw ArrowMetalError.bufferAllocationFailed(bytes: padded)
        }
        memset(mem, 0, padded)
        guard let b = context.device.makeBuffer(bytesNoCopy: mem, length: padded, options: [.storageModeShared],
                                                deallocator: { ptr, _ in free(ptr) }) else {
            free(mem)
            throw ArrowMetalError.bufferAllocationFailed(bytes: padded)
        }
        return MetalArrowBuffer(mtl: b, byteCount: byteCount)
    }

    /// Creates a buffer holding a copy of `bytes`.
    public static func copy(from ptr: UnsafeRawPointer, byteCount: Int, context: MetalContext = .shared) throws -> MetalArrowBuffer {
        let buf = try allocate(byteCount: byteCount, context: context)
        if byteCount > 0 { memcpy(buf.mutableContents, ptr, byteCount) }
        return buf
    }

    /// Wraps external memory without copying when it is page aligned (pointer and length), otherwise copies.
    ///
    /// `keepAlive` is retained for the lifetime of the returned buffer so that borrowed memory stays valid.
    /// Returns the buffer and whether the wrap was zero-copy.
    public static func wrapOrCopy(_ ptr: UnsafeRawPointer, byteCount: Int, keepAlive: AnyObject?,
                                  context: MetalContext = .shared) throws -> (MetalArrowBuffer, zeroCopy: Bool) {
        let page = metalPageSize()
        let addr = UInt(bitPattern: ptr)
        if byteCount > 0, addr % UInt(page) == 0 {
            let len = roundUp(byteCount, to: page)
            // bytesNoCopy requires the whole [ptr, ptr+len) range to be mapped. Page-aligned pointers from
            // mmap/vm_allocate/posix_memalign(page) satisfy this; we still verify with a cheap probe.
            if rangeIsMapped(ptr, length: len),
               let b = context.device.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(mutating: ptr),
                                                 length: len, options: [.storageModeShared], deallocator: nil) {
                return (MetalArrowBuffer(mtl: b, byteCount: byteCount, keepAlive: keepAlive ?? NSObject()), true)
            }
        }
        return (try copy(from: ptr, byteCount: byteCount, context: context), false)
    }

    public var contents: UnsafeRawPointer { UnsafeRawPointer(mtl.contents()).advanced(by: offset) }
    public var mutableContents: UnsafeMutableRawPointer { mtl.contents().advanced(by: offset) }

    /// Raw typed pointer. The pointer is only valid while this object is alive; prefer `withTyped`.
    public func typed<T>(_: T.Type) -> UnsafePointer<T> { contents.assumingMemoryBound(to: T.self) }
    public func mutableTyped<T>(_: T.Type) -> UnsafeMutablePointer<T> { mutableContents.assumingMemoryBound(to: T.self) }

    /// Scoped access that keeps the buffer alive for the duration of `body`.
    public func withTyped<T, R>(_: T.Type, _ body: (UnsafePointer<T>) throws -> R) rethrows -> R {
        try withExtendedLifetime(self) { try body(typed(T.self)) }
    }
    public func withMutableTyped<T, R>(_: T.Type, _ body: (UnsafeMutablePointer<T>) throws -> R) rethrows -> R {
        try withExtendedLifetime(self) { try body(mutableTyped(T.self)) }
    }
}

/// Uses `mincore` to check whether every page in the range is mapped into the process.
private func rangeIsMapped(_ ptr: UnsafeRawPointer, length: Int) -> Bool {
    let page = metalPageSize()
    let pages = (length + page - 1) / page
    var vec = [CChar](repeating: 0, count: pages)
    return mincore(UnsafeMutableRawPointer(mutating: ptr), length, &vec) == 0
}
