import Foundation
import Metal

/// Bit utilities for Arrow validity bitmaps (LSB bit order).
public enum Bitmap {
    @inline(__always) public static func byteCount(bits: Int) -> Int { (bits + 7) / 8 }
    @inline(__always) public static func isSet(_ p: UnsafePointer<UInt8>, _ i: Int) -> Bool { (p[i >> 3] >> (i & 7)) & 1 == 1 }
    @inline(__always) public static func set(_ p: UnsafeMutablePointer<UInt8>, _ i: Int) { p[i >> 3] |= UInt8(1 << (i & 7)) }
    @inline(__always) public static func clear(_ p: UnsafeMutablePointer<UInt8>, _ i: Int) { p[i >> 3] &= ~UInt8(1 << (i & 7)) }
    /// Count of set bits in the first `bits` bits.
    public static func popcount(_ p: UnsafePointer<UInt8>, bits: Int) -> Int {
        var n = 0
        let fullBytes = bits / 8
        var i = 0
        // 8 bytes at a time
        while i + 8 <= fullBytes {
            var w: UInt64 = 0
            memcpy(&w, p + i, 8)
            n += w.nonzeroBitCount
            i += 8
        }
        while i < fullBytes { n += p[i].nonzeroBitCount; i += 1 }
        let rem = bits % 8
        if rem > 0 { n += (p[fullBytes] & UInt8((1 << rem) - 1)).nonzeroBitCount }
        return n
    }
}

/// A primitive (fixed-width) Arrow array whose buffers live in Metal shared memory.
///
/// Layout follows the Arrow columnar spec exactly: an optional validity bitmap (buffer 0)
/// and a values buffer (buffer 1). `offset` is always 0; imported arrays with a non-zero
/// offset are materialised with the offset applied.
public final class MetalArray<T: ArrowPrimitive>: @unchecked Sendable {
    /// Number of elements. Reading it materialises a pending batched result whose length the GPU decides.
    public var length: Int { ensure(); return _length }
    var _length: Int
    /// Null count. Reading it materialises pending results.
    public var nullCount: Int { get { ensure(); return _nullCount } set { _nullCount = newValue } }
    var _nullCount: Int
    /// Validity bitmap, nil when there are no nulls. Buffers are usable by the GPU while a batch is pending.
    public let validity: MetalArrowBuffer?
    public let values: MetalArrowBuffer
    public let context: MetalContext
    /// Set while this array's contents are still being produced by an open batch.
    var pending = false
    /// Length known without a sync (worst case for pending filter results).
    var capacityLength: Int

    public init(length: Int, nullCount: Int, validity: MetalArrowBuffer?, values: MetalArrowBuffer, context: MetalContext = .shared) {
        precondition(values.byteCount >= length * T.byteWidth)
        if let v = validity { precondition(v.byteCount >= Bitmap.byteCount(bits: length)) }
        self._length = length
        self._nullCount = nullCount
        self.validity = validity
        self.values = values
        self.context = context
        self.capacityLength = length
    }

    /// Forces the batch that produces this array to run (no-op otherwise).
    @inline(__always) func ensure() {
        if pending { try? context.syncPoint() }
    }

    /// Builds an array from Swift values, copying them into Metal shared memory.
    public convenience init(_ vals: [T], context: MetalContext = .shared) throws {
        let vb = try MetalArrowBuffer.allocate(byteCount: vals.count * T.byteWidth, zeroed: false, context: context)
        // Plain loop on purpose: a `withUnsafeBytes` closure here miscompiled under -O with Swift 6.3.3
        // (crash on entry in release builds only). The optimiser turns this into a memcpy anyway.
        let dst = vb.mutableTyped(T.self)
        for i in 0..<vals.count { dst[i] = vals[i] }
        self.init(length: vals.count, nullCount: 0, validity: nil, values: vb, context: context)
    }

    /// Builds a nullable array from optionals.
    public convenience init(_ vals: [T?], context: MetalContext = .shared) throws {
        let n = vals.count
        let vb = try MetalArrowBuffer.allocate(byteCount: n * T.byteWidth, context: context)
        let bm = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: n), context: context)
        let vp = vb.mutableTyped(T.self)
        let bp = bm.mutableTyped(UInt8.self)
        var nulls = 0
        for i in 0..<n {
            if let v = vals[i] { vp[i] = v; Bitmap.set(bp, i) } else { vp[i] = 0; nulls += 1 }
        }
        self.init(length: n, nullCount: nulls, validity: nulls == 0 ? nil : bm, values: vb, context: context)
    }

    /// Allocates an uninitialised (zeroed) array with `length` slots and, optionally, a validity bitmap.
    public static func allocate(length: Int, withValidity: Bool, context: MetalContext = .shared) throws -> MetalArray<T> {
        let vb = try MetalArrowBuffer.allocate(byteCount: length * T.byteWidth, context: context)
        let bm = withValidity ? try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: length), context: context) : nil
        return MetalArray(length: length, nullCount: 0, validity: bm, values: vb, context: context)
    }

    /// Recomputes `nullCount` from the validity bitmap. When a batch is open the count is deferred until it
    /// runs, and the array is marked pending.
    public func recomputeNullCount() {
        guard let v = validity else { _nullCount = 0; return }
        if context.isBatching {
            pending = true
            try? context.afterFlush { [self] in
                self._nullCount = self._length - Bitmap.popcount(v.typed(UInt8.self), bits: self._length)
                self.pending = false
            }
            context.retainUntilFlush(self)
        } else {
            _nullCount = _length - Bitmap.popcount(v.typed(UInt8.self), bits: _length)
        }
    }

    /// Marks this array as produced by the open batch with a length the GPU will report in `lengthBuffer`.
    func deferLength(from lengthBuffer: MetalArrowBuffer, then: @escaping () -> Void) {
        pending = true
        try? context.afterFlush { [self] in
            self._length = Int(lengthBuffer.typed(UInt32.self)[0])
            then()
            self.pending = false
        }
        context.retainUntilFlush(self)
        context.retainUntilFlush(lengthBuffer)
    }

    /// Raw pointers are valid only while the array is alive; prefer `withValues`.
    public var valuePointer: UnsafePointer<T> { ensure(); return values.typed(T.self) }
    public var mutableValuePointer: UnsafeMutablePointer<T> { ensure(); return values.mutableTyped(T.self) }

    /// Scoped, lifetime-safe access to the values.
    public func withValues<R>(_ body: (UnsafeBufferPointer<T>) throws -> R) rethrows -> R {
        try withExtendedLifetime(self) { try body(UnsafeBufferPointer(start: valuePointer, count: length)) }
    }

    public func isValid(_ i: Int) -> Bool {
        ensure()
        guard let v = validity else { return true }
        return Bitmap.isSet(v.typed(UInt8.self), i)
    }

    public subscript(i: Int) -> T? { isValid(i) ? valuePointer[i] : nil }

    /// Copies the values out to a Swift array of optionals.
    public func toArray() -> [T?] { (0..<length).map { self[$0] } }

    /// Copies the raw values out (nulls come back as whatever is in the slot, usually 0).
    public func toRawArray() -> [T] { Array(UnsafeBufferPointer(start: valuePointer, count: length)) }
}

/// A boolean Arrow array: values are a packed bitmap (LSB order), same as the validity bitmap.
public final class MetalBooleanArray: @unchecked Sendable {
    public var length: Int { ensure(); return _length }
    var _length: Int
    public var nullCount: Int { get { ensure(); return _nullCount } set { _nullCount = newValue } }
    var _nullCount: Int
    public let validity: MetalArrowBuffer?
    public let values: MetalArrowBuffer
    public let context: MetalContext
    var pending = false

    public init(length: Int, nullCount: Int, validity: MetalArrowBuffer?, values: MetalArrowBuffer, context: MetalContext = .shared) {
        precondition(values.byteCount >= Bitmap.byteCount(bits: length))
        self._length = length; self._nullCount = nullCount; self.validity = validity; self.values = values; self.context = context
    }

    @inline(__always) func ensure() { if pending { try? context.syncPoint() } }

    public convenience init(_ vals: [Bool], context: MetalContext = .shared) throws {
        let vb = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: vals.count), context: context)
        let p = vb.mutableTyped(UInt8.self)
        for (i, b) in vals.enumerated() where b { Bitmap.set(p, i) }
        self.init(length: vals.count, nullCount: 0, validity: nil, values: vb, context: context)
    }

    public static func allocate(length: Int, withValidity: Bool, context: MetalContext = .shared) throws -> MetalBooleanArray {
        let vb = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: length), context: context)
        let bm = withValidity ? try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: length), context: context) : nil
        return MetalBooleanArray(length: length, nullCount: 0, validity: bm, values: vb, context: context)
    }

    public func recomputeNullCount() {
        guard let v = validity else { _nullCount = 0; return }
        if context.isBatching {
            pending = true
            try? context.afterFlush { [self] in
                self._nullCount = self._length - Bitmap.popcount(v.typed(UInt8.self), bits: self._length)
                self.pending = false
            }
            context.retainUntilFlush(self)
        } else {
            _nullCount = _length - Bitmap.popcount(v.typed(UInt8.self), bits: _length)
        }
    }

    public func isValid(_ i: Int) -> Bool {
        ensure()
        guard let v = validity else { return true }
        return Bitmap.isSet(v.typed(UInt8.self), i)
    }
    public subscript(i: Int) -> Bool? { isValid(i) ? Bitmap.isSet(values.typed(UInt8.self), i) : nil }
    public func toArray() -> [Bool?] { (0..<length).map { self[$0] } }
    /// Number of true values among valid slots.
    public var trueCount: Int {
        ensure()
        guard let v = validity else { return Bitmap.popcount(values.typed(UInt8.self), bits: length) }
        var n = 0
        let vp = v.typed(UInt8.self), bp = values.typed(UInt8.self)
        for i in 0..<Bitmap.byteCount(bits: length) {
            var byte = vp[i] & bp[i]
            if i == length / 8 && length % 8 != 0 { byte &= UInt8((1 << (length % 8)) - 1) }
            n += byte.nonzeroBitCount
        }
        return n
    }
}
