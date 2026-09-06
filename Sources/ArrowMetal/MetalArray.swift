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

    // MARK: - bit ranges (Arrow `offset`)
    //
    // An Arrow array may start at any bit of its validity bitmap. These read a `[from, from + bits)`
    // range without materialising a shifted copy, so a slice stays O(1).

    /// Count of set bits in `[from, from + bits)`. Scans only the bytes that range covers.
    public static func popcount(_ p: UnsafePointer<UInt8>, from: Int, bits: Int) -> Int {
        if bits <= 0 { return 0 }
        if from == 0 { return popcount(p, bits: bits) }
        let startByte = from >> 3, head = from & 7
        if head == 0 { return popcount(p + startByte, bits: bits) }
        // Count the whole head byte's worth and subtract the leading bits that are not ours.
        return popcount(p + startByte, bits: head + bits) - (p[startByte] & UInt8((1 << head) - 1)).nonzeroBitCount
    }

    /// Index (relative to `from`) of the first and last set bit in `[from, from + bits)`, or nil when the
    /// range holds none. Scans 64 bits at a time inwards from each end, so the cost is the distance to the
    /// first (last) set bit, not the length of the range.
    public static func bounds(_ p: UnsafePointer<UInt8>, from: Int, bits: Int) -> (first: Int, last: Int)? {
        guard bits > 0 else { return nil }
        guard let f = firstSet(p, from: from, bits: bits) else { return nil }
        return (f, lastSet(p, from: from, bits: bits)!)
    }

    /// Index (relative to `from`) of the first set bit in `[from, from + bits)`, nil when there is none.
    public static func firstSet(_ p: UnsafePointer<UInt8>, from: Int, bits: Int) -> Int? {
        var i = 0
        while i < bits {
            let take = Swift.min(64, bits - i)
            let w = word(p, at: from + i, count: take)
            if w != 0 { return i + w.trailingZeroBitCount }
            i += take
        }
        return nil
    }

    /// Index (relative to `from`) of the last set bit in `[from, from + bits)`, nil when there is none.
    public static func lastSet(_ p: UnsafePointer<UInt8>, from: Int, bits: Int) -> Int? {
        var end = bits
        while end > 0 {
            let take = Swift.min(64, end)
            let start = end - take
            let w = word(p, at: from + start, count: take)
            if w != 0 { return start + (63 - w.leadingZeroBitCount) }
            end = start
        }
        return nil
    }

    /// The `count` (1...64) bits starting at bit `at`, right aligned, zero above them.
    @inline(__always) static func word(_ p: UnsafePointer<UInt8>, at: Int, count: Int) -> UInt64 {
        let byte = at >> 3, sh = at & 7
        let need = (sh + count + 7) / 8            // bytes this range touches
        var lo: UInt64 = 0
        memcpy(&lo, p + byte, Swift.min(need, 8))
        var w = lo >> UInt64(sh)
        if need > 8 { w |= UInt64(p[byte + 8]) << UInt64(64 - sh) }
        if count < 64 { w &= (UInt64(1) << UInt64(count)) - 1 }
        return w
    }

    /// True when any bit of `a & b` (or of `a` alone when `b` is nil) is set in `[from, from + bits)`.
    /// Returns nil when `limit` bits were scanned without an answer, so the caller can escalate.
    public static func anySet(_ a: UnsafePointer<UInt8>, _ b: UnsafePointer<UInt8>?, from: Int, bits: Int,
                              limit: Int = Int.max) -> Bool? {
        var i = 0
        while i < bits {
            if i >= limit { return nil }
            let take = Swift.min(64, bits - i)
            var w = word(a, at: from + i, count: take)
            if let b { w &= word(b, at: from + i, count: take) }
            if w != 0 { return true }
            i += take
        }
        return false
    }

    /// True when every valid bit is set: no bit of `validity & ~values` is set in the range. With no
    /// validity, every bit of `values` must be set. Returns nil when `limit` bits gave no answer.
    public static func allSet(_ values: UnsafePointer<UInt8>, _ validity: UnsafePointer<UInt8>?, from: Int,
                              bits: Int, limit: Int = Int.max) -> Bool? {
        var i = 0
        while i < bits {
            if i >= limit { return nil }
            let take = Swift.min(64, bits - i)
            let v = word(values, at: from + i, count: take)
            let mask = take == 64 ? UInt64.max : (UInt64(1) << UInt64(take)) - 1
            let valid = validity.map { word($0, at: from + i, count: take) } ?? mask
            if (valid & ~v & mask) != 0 { return false }
            i += take
        }
        return true
    }

    /// Copies `bits` bits starting at bit `from` of `src` to bit 0 of `dst`. `dst` must hold
    /// `byteCount(bits:)` bytes; trailing bits of the last byte are zeroed.
    public static func copyBits(_ src: UnsafePointer<UInt8>, from: Int, into dst: UnsafeMutablePointer<UInt8>, bits: Int) {
        let outBytes = byteCount(bits: bits)
        guard outBytes > 0 else { return }
        let startByte = from >> 3, sh = from & 7
        if sh == 0 {
            memcpy(dst, src + startByte, outBytes)
        } else {
            let srcBytes = byteCount(bits: from + bits)      // last source byte this range may touch
            var i = 0
            // 8 output bytes per iteration; needs 9 source bytes, so stop before the source runs out.
            while i + 8 <= outBytes && startByte + i + 9 <= srcBytes {
                var w: UInt64 = 0
                memcpy(&w, src + startByte + i, 8)
                var out = (w >> UInt64(sh)) | (UInt64(src[startByte + i + 8]) << UInt64(64 - sh))
                memcpy(dst + i, &out, 8)
                i += 8
            }
            while i < outBytes {
                let a = UInt16(src[startByte + i])
                let b = (startByte + i + 1) < srcBytes ? UInt16(src[startByte + i + 1]) : 0
                dst[i] = UInt8(truncatingIfNeeded: (a | (b << 8)) >> UInt16(sh))
                i += 1
            }
        }
        let rem = bits % 8
        if rem > 0 { dst[outBytes - 1] &= UInt8((1 << rem) - 1) }
    }
}

/// A primitive (fixed-width) Arrow array whose buffers live in Metal shared memory.
///
/// Layout follows the Arrow columnar spec exactly: an optional validity bitmap (buffer 0)
/// and a values buffer (buffer 1), plus Arrow's element `offset`.
///
/// A slice is always O(1). When its offset is a multiple of 32 both buffers are expressed exactly as
/// `MetalArrowBuffer` views (bitmap words stay word aligned, values stay vector-load aligned) and `offset`
/// stays 0, so the kernels are unaware a slice happened. At any other offset the array keeps the parent's
/// buffers and carries `offset`; `values` / `validity` then normalise it away on first kernel use and cache
/// the result, while host reads (`subscript`, `valuePointer`, `first`, `last`, `nullCount`) and export read
/// the offset directly and never materialise anything.
public final class MetalArray<T: ArrowPrimitive>: @unchecked Sendable {
    /// Number of elements. Reading it materialises a pending batched result whose length the GPU decides.
    public var length: Int { ensure(); return _length }
    var _length: Int
    /// Null count. Reading it materialises pending results.
    public var nullCount: Int { get { ensure(); return _nullCount } set { _nullCount = newValue } }
    /// Storage for the null count plus the deferred count a slice installs (a popcount over its bit range,
    /// paid only if someone asks). Every internal `_nullCount` read resolves it first.
    private var __nullCount: Int
    private var _nullCountPending: (() -> Int)?
    var _nullCount: Int {
        get { if let f = _nullCountPending { __nullCount = f(); _nullCountPending = nil }; return __nullCount }
        set { __nullCount = newValue; _nullCountPending = nil }
    }
    /// Arrow's element `offset`: this array starts at element `offset` of `rawValues` / bit `offset` of
    /// `rawValidity`. Non-zero only for a slice whose offset is not a multiple of 32 (any other offset is
    /// expressed exactly as a buffer view, so the kernels see aligned buffers with no offset at all).
    public private(set) var offset: Int = 0
    /// Storage as handed in. `values` / `validity` normalise a non-zero offset away on first use.
    public let rawValidity: MetalArrowBuffer?
    public let rawValues: MetalArrowBuffer
    private var _normValidity: MetalArrowBuffer?
    private var _normValues: MetalArrowBuffer?
    private let normLock = NSLock()
    /// Validity bitmap, nil when there are no nulls. Buffers are usable by the GPU while a batch is pending.
    public var validity: MetalArrowBuffer? { offset == 0 ? rawValidity : normalized().validity }
    public var values: MetalArrowBuffer { offset == 0 ? rawValues : normalized().values }
    public let context: MetalContext

    /// Materialises the offset away, once, so every kernel keeps seeing a zero-offset array.
    ///
    /// Only reached for a slice at an offset that is not a multiple of 32: the values are a `memcpy` and
    /// the bitmap a word-wise bit shift (`Bitmap.copyBits`), both on the CPU because a normalisation can be
    /// requested from inside an open command encoder. The result is cached, so a sliced column pays this at
    /// most once no matter how many kernels run over it.
    private func normalized() -> (validity: MetalArrowBuffer?, values: MetalArrowBuffer) {
        normLock.lock(); defer { normLock.unlock() }
        if let v = _normValues { return (_normValidity, v) }
        let n = _length
        // Failing here would hand a kernel the unshifted parent buffer, which is silently wrong; only an
        // out-of-memory condition can cause it, so say so rather than compute the wrong answer.
        guard let vb = try? MetalArrowBuffer.allocate(byteCount: Swift.max(n * T.byteWidth, 1), zeroed: false, context: context) else {
            preconditionFailure("out of memory normalising a sliced array of \(n) elements")
        }
        withExtendedLifetime(rawValues) {
            if n > 0 { memcpy(vb.mutableContents, rawValues.contents.advanced(by: offset * T.byteWidth), n * T.byteWidth) }
        }
        var bm: MetalArrowBuffer? = nil
        if let rv = rawValidity {
            guard let b = try? MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 1), context: context) else {
                preconditionFailure("out of memory normalising a sliced array's validity bitmap")
            }
            withExtendedLifetime(rv) { Bitmap.copyBits(rv.typed(UInt8.self), from: offset, into: b.mutableTyped(UInt8.self), bits: n) }
            bm = b
        }
        _normValues = vb; _normValidity = bm
        return (bm, vb)
    }
    /// Set while this array's contents are still being produced by an open batch.
    var pending = false
    /// Length known without a sync (worst case for pending filter results).
    var capacityLength: Int
    /// Device buffer holding the true length while pending (written by the GPU), nil otherwise.
    var lengthBuffer: MetalArrowBuffer? { pending ? _lengthBuffer : nil }
    var _lengthBuffer: MetalArrowBuffer?
    /// Length to size dispatches with (no sync): worst case while pending.
    var dispatchLength: Int { pending ? capacityLength : _length }
    /// Length without forcing a sync; only meaningful when not pending.
    var knownLength: Int { _length }

    public init(length: Int, nullCount: Int, validity: MetalArrowBuffer?, values: MetalArrowBuffer, context: MetalContext = .shared) {
        precondition(values.byteCount >= length * T.byteWidth)
        if let v = validity { precondition(v.byteCount >= Bitmap.byteCount(bits: length)) }
        self._length = length
        self.__nullCount = nullCount
        self.rawValidity = validity
        self.rawValues = values
        self.context = context
        self.capacityLength = length
    }

    /// Arrow's offset form: this array is `length` elements starting at element `offset` of `values`
    /// (bit `offset` of `validity`). `nullCount` is counted lazily from the bitmap on first read.
    ///
    /// An offset that is a multiple of 32 is turned into a pair of buffer views right here, so the array
    /// ends up with no offset at all and every kernel binds an aligned buffer as before. Only the remaining
    /// offsets are carried.
    init(offset: Int, length: Int, validity: MetalArrowBuffer?, values: MetalArrowBuffer,
         nullCount: Int? = nil, context: MetalContext = .shared) {
        precondition(offset >= 0 && length >= 0 && values.byteCount >= (offset + length) * T.byteWidth)
        if let v = validity { precondition(v.byteCount >= Bitmap.byteCount(bits: offset + length)) }
        var off = offset, vals = values, vld = validity
        if off > 0 && off % 32 == 0 {
            vals = values.view(byteOffset: off * T.byteWidth, byteCount: length * T.byteWidth)
            vld = validity?.view(byteOffset: off / 8, byteCount: Bitmap.byteCount(bits: length))
            off = 0
        }
        self._length = length
        self.__nullCount = nullCount ?? 0
        self.rawValidity = vld
        self.rawValues = vals
        self.context = context
        self.capacityLength = length
        self.offset = off
        if nullCount == nil, let v = vld {
            self._nullCountPending = { [off, len = length] in
                withExtendedLifetime(v) { len - Bitmap.popcount(v.typed(UInt8.self), from: off, bits: len) }
            }
        }
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
        guard let v = rawValidity else { _nullCount = 0; return }
        let off = offset
        if context.isBatching {
            pending = true
            try? context.afterFlush { [self] in
                self._nullCount = self._length - Bitmap.popcount(v.typed(UInt8.self), from: off, bits: self._length)
                self.pending = false
            }
            context.retainUntilFlush(self)
        } else {
            _nullCount = _length - Bitmap.popcount(v.typed(UInt8.self), from: off, bits: _length)
        }
    }

    /// Marks this array as produced by the open batch with a length the GPU will report in `lengthBuffer`.
    func deferLength(from lengthBuffer: MetalArrowBuffer, then: @escaping () -> Void) {
        pending = true
        _lengthBuffer = lengthBuffer
        try? context.afterFlush { [self] in
            self._length = Int(lengthBuffer.typed(UInt32.self)[0])
            then()
            self.pending = false
        }
        context.retainUntilFlush(self)
        context.retainUntilFlush(lengthBuffer)
    }

    /// Raw pointers are valid only while the array is alive; prefer `withValues`.
    /// A sliced array answers straight from its storage, so a host read never triggers a normalisation.
    public var valuePointer: UnsafePointer<T> { ensure(); return rawValues.typed(T.self) + offset }
    public var mutableValuePointer: UnsafeMutablePointer<T> { ensure(); return rawValues.mutableTyped(T.self) + offset }

    /// Scoped, lifetime-safe access to the values.
    public func withValues<R>(_ body: (UnsafeBufferPointer<T>) throws -> R) rethrows -> R {
        try withExtendedLifetime(self) { try body(UnsafeBufferPointer(start: valuePointer, count: length)) }
    }

    public func isValid(_ i: Int) -> Bool {
        ensure()
        guard let v = rawValidity else { return true }
        return Bitmap.isSet(v.typed(UInt8.self), offset + i)
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
    private var __nullCount: Int
    private var _nullCountPending: (() -> Int)?
    var _nullCount: Int {
        get { if let f = _nullCountPending { __nullCount = f(); _nullCountPending = nil }; return __nullCount }
        set { __nullCount = newValue; _nullCountPending = nil }
    }
    /// Arrow's element `offset` into `rawValues` / `rawValidity` (both bitmaps here). See `MetalArray.offset`.
    public private(set) var offset: Int = 0
    public let rawValidity: MetalArrowBuffer?
    public let rawValues: MetalArrowBuffer
    private var _normValidity: MetalArrowBuffer?
    private var _normValues: MetalArrowBuffer?
    private let normLock = NSLock()
    public var validity: MetalArrowBuffer? { offset == 0 ? rawValidity : normalized().validity }
    public var values: MetalArrowBuffer { offset == 0 ? rawValues : normalized().values }
    public let context: MetalContext
    var pending = false
    var _capacityLength: Int = 0
    var capacityLength: Int { pending ? _capacityLength : _length }
    var lengthBuffer: MetalArrowBuffer? { pending ? _lengthBuffer : nil }
    var _lengthBuffer: MetalArrowBuffer?
    var dispatchLength: Int { pending ? _capacityLength : _length }
    var knownLength: Int { _length }

    public init(length: Int, nullCount: Int, validity: MetalArrowBuffer?, values: MetalArrowBuffer, context: MetalContext = .shared) {
        precondition(values.byteCount >= Bitmap.byteCount(bits: length))
        self._length = length; self.__nullCount = nullCount; self.rawValidity = validity; self.rawValues = values; self.context = context
    }

    /// Arrow's offset form: `length` bits starting at bit `offset` of both bitmaps. See `MetalArray`.
    init(offset: Int, length: Int, validity: MetalArrowBuffer?, values: MetalArrowBuffer,
         nullCount: Int? = nil, context: MetalContext = .shared) {
        precondition(offset >= 0 && length >= 0 && values.byteCount >= Bitmap.byteCount(bits: offset + length))
        if let v = validity { precondition(v.byteCount >= Bitmap.byteCount(bits: offset + length)) }
        var off = offset, vals = values, vld = validity
        if off > 0 && off % 32 == 0 {
            vals = values.view(byteOffset: off / 8, byteCount: Bitmap.byteCount(bits: length))
            vld = validity?.view(byteOffset: off / 8, byteCount: Bitmap.byteCount(bits: length))
            off = 0
        }
        self._length = length; self.__nullCount = nullCount ?? 0
        self.rawValidity = vld; self.rawValues = vals; self.context = context
        self.offset = off
        if nullCount == nil, let v = vld {
            self._nullCountPending = { [off, len = length] in
                withExtendedLifetime(v) { len - Bitmap.popcount(v.typed(UInt8.self), from: off, bits: len) }
            }
        }
    }

    /// See `MetalArray.normalized()`: materialises the offset away once, on the CPU, and caches it.
    private func normalized() -> (validity: MetalArrowBuffer?, values: MetalArrowBuffer) {
        normLock.lock(); defer { normLock.unlock() }
        if let v = _normValues { return (_normValidity, v) }
        let n = _length
        let bytes = Swift.max(Bitmap.byteCount(bits: n), 1)
        guard let vb = try? MetalArrowBuffer.allocate(byteCount: bytes, context: context) else {
            preconditionFailure("out of memory normalising a sliced boolean array of \(n) rows")
        }
        withExtendedLifetime(rawValues) {
            Bitmap.copyBits(rawValues.typed(UInt8.self), from: offset, into: vb.mutableTyped(UInt8.self), bits: n)
        }
        var bm: MetalArrowBuffer? = nil
        if let rv = rawValidity {
            guard let b = try? MetalArrowBuffer.allocate(byteCount: bytes, context: context) else {
                preconditionFailure("out of memory normalising a sliced boolean array's validity bitmap")
            }
            withExtendedLifetime(rv) { Bitmap.copyBits(rv.typed(UInt8.self), from: offset, into: b.mutableTyped(UInt8.self), bits: n) }
            bm = b
        }
        _normValues = vb; _normValidity = bm
        return (bm, vb)
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
        guard let v = rawValidity else { _nullCount = 0; return }
        let off = offset
        if context.isBatching {
            pending = true
            try? context.afterFlush { [self] in
                self._nullCount = self._length - Bitmap.popcount(v.typed(UInt8.self), from: off, bits: self._length)
                self.pending = false
            }
            context.retainUntilFlush(self)
        } else {
            _nullCount = _length - Bitmap.popcount(v.typed(UInt8.self), from: off, bits: _length)
        }
    }

    public func isValid(_ i: Int) -> Bool {
        ensure()
        guard let v = rawValidity else { return true }
        return Bitmap.isSet(v.typed(UInt8.self), offset + i)
    }
    public subscript(i: Int) -> Bool? { isValid(i) ? Bitmap.isSet(rawValues.typed(UInt8.self), offset + i) : nil }
    public func toArray() -> [Bool?] { (0..<length).map { self[$0] } }
    /// Number of true values among valid slots. Reads the offset directly, so a slice never materialises.
    public var trueCount: Int {
        ensure()
        let len = _length, off = offset
        return withExtendedLifetime(self) { () -> Int in
            let bp = rawValues.typed(UInt8.self)
            guard let v = rawValidity else { return Bitmap.popcount(bp, from: off, bits: len) }
            let vp = v.typed(UInt8.self)
            var n = 0, i = 0
            while i < len {
                let take = Swift.min(64, len - i)
                n += (Bitmap.word(bp, at: off + i, count: take) & Bitmap.word(vp, at: off + i, count: take)).nonzeroBitCount
                i += take
            }
            return n
        }
    }
}

// MARK: - Pending propagation (batched execution)

extension MetalArray {
    /// Element-wise results of a pending input are pending too, with the same length source.
    func inheritPending<R: PendingCarrier>(_ r: R) -> R {
        if pending, let lb = _lengthBuffer {
            r.markPending(capacity: capacityLength, lengthBuffer: lb)
            context.retainUntilFlush(self)
        }
        return r
    }
    /// Length equality without forcing a sync when both come from the same pending source.
    func checkSameLength<O: PendingCarrier>(_ other: O) throws {
        if pending || other.isPending {
            if let a = _lengthBuffer, let b = other.pendingLengthBuffer, a === b { return }
            if pending && other.isPending && _lengthBuffer == nil && other.pendingLengthBuffer == nil { return }
            _ = length; _ = other.resolvedLength   // sync both and compare
        }
        guard other.resolvedLength == length else { throw ArrowMetalError.lengthMismatch(length, other.resolvedLength) }
    }
}

extension MetalBooleanArray {
    func inheritPending<R: PendingCarrier>(_ r: R) -> R {
        if pending, let lb = _lengthBuffer {
            r.markPending(capacity: capacityLength, lengthBuffer: lb)
            context.retainUntilFlush(self)
        }
        return r
    }
    func checkSameLength<O: PendingCarrier>(_ other: O) throws {
        if pending || other.isPending {
            if let a = _lengthBuffer, let b = other.pendingLengthBuffer, a === b { return }
            _ = length; _ = other.resolvedLength
        }
        guard other.resolvedLength == length else { throw ArrowMetalError.lengthMismatch(length, other.resolvedLength) }
    }
}

protocol PendingCarrier: AnyObject {
    var isPending: Bool { get }
    var pendingLengthBuffer: MetalArrowBuffer? { get }
    var resolvedLength: Int { get }
    func markPending(capacity: Int, lengthBuffer: MetalArrowBuffer)
}

extension MetalArray: PendingCarrier {
    var isPending: Bool { pending }
    var pendingLengthBuffer: MetalArrowBuffer? { _lengthBuffer }
    var resolvedLength: Int { length }
    func markPending(capacity: Int, lengthBuffer: MetalArrowBuffer) {
        capacityLength = capacity
        deferLength(from: lengthBuffer) {}
    }
}
extension MetalBooleanArray: PendingCarrier {
    var isPending: Bool { pending }
    var pendingLengthBuffer: MetalArrowBuffer? { _lengthBuffer }
    var resolvedLength: Int { length }
    func markPending(capacity: Int, lengthBuffer: MetalArrowBuffer) {
        _capacityLength = capacity
        pending = true
        _lengthBuffer = lengthBuffer
        try? context.afterFlush { [self] in
            self._length = Int(lengthBuffer.typed(UInt32.self)[0])
            if let v = self.rawValidity { self._nullCount = self._length - Bitmap.popcount(v.typed(UInt8.self), from: self.offset, bits: self._length) }
            self.pending = false
        }
        context.retainUntilFlush(self)
        context.retainUntilFlush(lengthBuffer)
    }
}
