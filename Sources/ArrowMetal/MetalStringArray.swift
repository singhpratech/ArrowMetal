import Foundation
import Metal
import CArrowABI

/// An Arrow `utf8` array in Metal shared memory: validity bitmap, int32 offsets (length + 1), data bytes.
/// `large_utf8` producers are accepted on import when the data fits in 2 GB (offsets are narrowed).
public final class MetalStringArray: @unchecked Sendable {
    public let length: Int
    public internal(set) var nullCount: Int
    public let validity: MetalArrowBuffer?
    public let offsets: MetalArrowBuffer
    public let data: MetalArrowBuffer
    public let context: MetalContext

    public init(length: Int, nullCount: Int, validity: MetalArrowBuffer?, offsets: MetalArrowBuffer, data: MetalArrowBuffer, context: MetalContext = .shared) {
        precondition(offsets.byteCount >= (length + 1) * 4)
        self.length = length; self.nullCount = nullCount; self.validity = validity; self.offsets = offsets; self.data = data; self.context = context
    }

    public convenience init(_ strings: [String?], context: MetalContext = .shared) throws {
        let n = strings.count
        var total = 0
        for s in strings { total += s?.utf8.count ?? 0 }
        let off = try MetalArrowBuffer.allocate(byteCount: (n + 1) * 4, zeroed: false, context: context)
        let dat = try MetalArrowBuffer.allocate(byteCount: total, zeroed: false, context: context)
        let hasNulls = strings.contains { $0 == nil }
        let bm = hasNulls ? try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: n), context: context) : nil
        let op = off.mutableTyped(Int32.self), dp = dat.mutableTyped(UInt8.self)
        var pos = 0, nulls = 0
        for (i, s) in strings.enumerated() {
            op[i] = Int32(pos)
            if let s {
                for b in s.utf8 { dp[pos] = b; pos += 1 }
                if let bm { Bitmap.set(bm.mutableTyped(UInt8.self), i) }
            } else { nulls += 1 }
        }
        op[n] = Int32(pos)
        self.init(length: n, nullCount: nulls, validity: bm, offsets: off, data: dat, context: context)
    }

    public var totalBytes: Int { Int(offsets.typed(Int32.self)[length]) }
    public func isValid(_ i: Int) -> Bool { validity.map { Bitmap.isSet($0.typed(UInt8.self), i) } ?? true }
    public subscript(i: Int) -> String? {
        guard isValid(i) else { return nil }
        let o = offsets.typed(Int32.self), d = data.typed(UInt8.self)
        return String(decoding: UnsafeBufferPointer(start: d + Int(o[i]), count: Int(o[i + 1] - o[i])), as: UTF8.self)
    }
    public func toArray() -> [String?] { (0..<length).map { self[$0] } }

    // MARK: kernels

    private func pso(_ fn: String) throws -> MTLComputePipelineState {
        try context.pipeline(source: StringSource.source, function: fn, cacheKey: "str/\(fn)")
    }

    /// Byte length per string (null in, null out).
    public func byteLength() throws -> MetalArray<Int32> { try lengths("str_byte_length", withData: false) }
    /// UTF-8 code point count per string.
    public func charLength() throws -> MetalArray<Int32> { try lengths("str_char_length", withData: true) }

    private func lengths(_ fn: String, withData: Bool) throws -> MetalArray<Int32> {
        let out = try MetalArrowBuffer.allocate(byteCount: length * 4, zeroed: false, context: context)
        let p = try pso(fn)
        if length > 0 {
            try context.run { enc in
                enc.setComputePipelineState(p)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                var idx = 1
                if withData { enc.setBuffer(data.mtl, offset: data.offset, index: 1); idx = 2 }
                Dispatch.setLength(enc, length, nil, index: idx)
                enc.setBuffer(out.mtl, offset: out.offset, index: idx + 1)
                Dispatch.dispatch1D(enc, p, count: length)
            }
        }
        return MetalArray<Int32>(length: length, nullCount: nullCount, validity: validity, values: out, context: context)
    }

    public enum Predicate: Int { case equals = 0, startsWith, endsWith, contains }

    /// `equals`, `startsWith`, `endsWith`, `contains` against a scalar pattern (byte-wise, case-sensitive).
    public func matches(_ pred: Predicate, _ pattern: String) throws -> MetalBooleanArray {
        let words = BitmapOps.words(bits: length)
        let out = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: length), zeroed: false, context: context)
        let pat = Array(pattern.utf8)
        let patBuf = try MetalArrowBuffer.allocate(byteCount: Swift.max(pat.count, 1), zeroed: false, context: context)
        pat.withUnsafeBytes { if $0.count > 0 { memcpy(patBuf.mutableContents, $0.baseAddress!, $0.count) } }
        let p = try pso("str_predicate")
        if length > 0 {
            try context.run { enc in
                enc.setComputePipelineState(p)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                Dispatch.setLength(enc, length, nil, index: 2)
                enc.setBuffer(patBuf.mtl, offset: patBuf.offset, index: 3)
                Dispatch.setUInt(enc, pat.count, index: 4)
                Dispatch.setUInt(enc, pred.rawValue, index: 5)
                enc.setBuffer(out.mtl, offset: out.offset, index: 6)
                Dispatch.dispatch1D(enc, p, count: words)
            }
        }
        context.retainUntilFlush(patBuf)
        return MetalBooleanArray(length: length, nullCount: nullCount, validity: validity, values: out, context: context)
    }

    public func equals(_ s: String) throws -> MetalBooleanArray { try matches(.equals, s) }
    public func startsWith(_ s: String) throws -> MetalBooleanArray { try matches(.startsWith, s) }
    public func endsWith(_ s: String) throws -> MetalBooleanArray { try matches(.endsWith, s) }
    public func contains(_ s: String) throws -> MetalBooleanArray { try matches(.contains, s) }

    /// Element-wise equality with another string array.
    public func equals(_ other: MetalStringArray) throws -> MetalBooleanArray {
        guard other.length == length else { throw ArrowMetalError.lengthMismatch(length, other.length) }
        let words = BitmapOps.words(bits: length)
        let out = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: length), zeroed: false, context: context)
        let p = try pso("str_eq_array")
        if length > 0 {
            try context.run { enc in
                enc.setComputePipelineState(p)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                enc.setBuffer(other.offsets.mtl, offset: other.offsets.offset, index: 2)
                enc.setBuffer(other.data.mtl, offset: other.data.offset, index: 3)
                Dispatch.setLength(enc, length, nil, index: 4)
                enc.setBuffer(out.mtl, offset: out.offset, index: 5)
                Dispatch.dispatch1D(enc, p, count: words)
            }
        }
        let v = try BitmapOps.combineValidity(context, validity, other.validity, bits: length)
        let res = MetalBooleanArray(length: length, nullCount: 0, validity: v, values: out, context: context)
        res.recomputeNullCount()
        return res
    }

    /// MurmurHash3 (x86_32, seed 0) of each string's bytes. Nulls hash to 0 and stay null.
    public func hash32() throws -> MetalArray<UInt32> {
        let out = try MetalArrowBuffer.allocate(byteCount: length * 4, zeroed: false, context: context)
        let p = try pso("str_hash32")
        if length > 0 {
            try context.run { enc in
                enc.setComputePipelineState(p)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                Dispatch.setLength(enc, length, nil, index: 2)
                enc.setBuffer(out.mtl, offset: out.offset, index: 3)
                Dispatch.dispatch1D(enc, p, count: length)
            }
        }
        return MetalArray<UInt32>(length: length, nullCount: nullCount, validity: validity, values: out, context: context)
    }

    /// Arrow `filter`.
    public func filter(_ mask: MetalBooleanArray) throws -> MetalStringArray {
        guard mask.length == length else { throw ArrowMetalError.lengthMismatch(length, mask.length) }
        // Source index per kept row = filter over an iota, then gather.
        let iota = try MetalArray<Int32>.iota(length, context: context)
        let src = try iota.filter(mask)
        return try gather(src)
    }

    /// Arrow `take`.
    public func take<I: ArrowIndex>(_ indices: MetalArray<I>) throws -> MetalStringArray {
        let idx32: MetalArray<Int32> = I.self == Int32.self ? (indices as! MetalArray<Int32>) : try indices.cast(to: Int32.self)
        try Dispatch.checkLength(length)
        // Bounds check on the CPU-visible index range is done by take on the lengths array below.
        return try gather(idx32)
    }

    /// Builds a new string array from source indices (null index -> null string).
    func gather(_ src: MetalArray<Int32>) throws -> MetalStringArray {
        let n = src.length
        let ctx = context
        // Lengths of selected strings (bounds checked by take), then offsets via scan.
        let lens = try byteLength().take(src)                       // Int32, null where src null
        let lensNoNull: MetalArray<Int32> = lens.validity == nil ? lens : try lens.fillNull(0)
        let outOffsets = try lensNoNull.exclusiveScanToOffsets()
        let total = Int(withExtendedLifetime(outOffsets) { outOffsets.typed(Int32.self)[n] })
        let outData = try MetalArrowBuffer.allocate(byteCount: total, zeroed: false, context: ctx)
        // Source index with -1 for nulls so the copy kernel skips them.
        let srcFilled: MetalArray<Int32> = src.validity == nil ? src : try src.fillNull(-1)
        let p = try pso("str_gather_bytes")
        if n > 0 {
            try ctx.run { enc in
                enc.setComputePipelineState(p)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                enc.setBuffer(srcFilled.values.mtl, offset: srcFilled.values.offset, index: 2)
                enc.setBuffer(outOffsets.mtl, offset: outOffsets.offset, index: 3)
                Dispatch.setLength(enc, n, nil, index: 4)
                enc.setBuffer(outData.mtl, offset: outData.offset, index: 5)
                Dispatch.dispatch1D(enc, p, count: n)
            }
        }
        // Validity: source validity gathered AND index validity = lens' validity (take already combined them).
        let res = MetalStringArray(length: n, nullCount: lens.nullCount, validity: lens.validity, offsets: outOffsets, data: outData, context: ctx)
        return res
    }

    /// Dictionary-encodes on the GPU (`Kernels/StringDictionary.swift`): returns dense Int32 codes in
    /// [0, unique.count) plus the unique strings in first-seen order. Feeds `GroupBy`. Null strings
    /// become null codes.
    public func dictionaryEncode() throws -> (codes: MetalArray<Int32>, unique: MetalStringArray) {
        try dictionaryEncodeGPU()
    }
}

extension MetalArray where T == Int32 {
    /// 0, 1, 2, ... n-1
    static func iota(_ n: Int, context: MetalContext) throws -> MetalArray<Int32> {
        let out = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: false, context: context)
        let p = out.mutableTyped(Int32.self)
        for i in 0..<n { p[i] = Int32(i) }
        return MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: out, context: context)
    }

    /// Exclusive prefix sum into an (n+1)-element offsets buffer, entirely on the GPU.
    func exclusiveScanToOffsets() throws -> MetalArrowBuffer {
        let n = length
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: (n + 1) * 4, zeroed: false, context: ctx)
        let blocks = Swift.max(1, (n + Dispatch.threadgroupSize - 1) / Dispatch.threadgroupSize)
        let totals = try MetalArrowBuffer.allocate(byteCount: blocks * 4, zeroed: false, context: ctx)
        let grand = try MetalArrowBuffer.allocate(byteCount: 4, zeroed: false, context: ctx)
        let p1 = try ctx.pipeline(source: StringSource.source, function: "scan_block", cacheKey: "str/scan_block")
        let p2 = try ctx.pipeline(source: StringSource.source, function: "scan_totals", cacheKey: "str/scan_totals")
        let p3 = try ctx.pipeline(source: StringSource.source, function: "scan_add", cacheKey: "str/scan_add")
        let tg = MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1)
        try ctx.run { enc in
            enc.setComputePipelineState(p1)
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            Dispatch.setLength(enc, n, nil, index: 1)
            enc.setBuffer(out.mtl, offset: 0, index: 2)
            enc.setBuffer(totals.mtl, offset: 0, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: blocks, height: 1, depth: 1), threadsPerThreadgroup: tg)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(p2)
            enc.setBuffer(totals.mtl, offset: 0, index: 0)
            Dispatch.setUInt(enc, blocks, index: 1)
            enc.setBuffer(grand.mtl, offset: 0, index: 2)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(p3)
            enc.setBuffer(out.mtl, offset: 0, index: 0)
            enc.setBuffer(totals.mtl, offset: 0, index: 1)
            Dispatch.setLength(enc, n, nil, index: 2)
            enc.setBuffer(grand.mtl, offset: 0, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: blocks + 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
        }
        try ctx.syncPoint()
        return out
    }
}

extension MetalArray {
    /// Replaces nulls with `value` and drops the validity bitmap (values buffer is rewritten on the CPU for
    /// simplicity; used for small helper arrays).
    func fillNull(_ value: T) throws -> MetalArray<T> {
        guard let v = validity else { return self }
        let out = try MetalArrowBuffer.allocate(byteCount: length * T.byteWidth, zeroed: false, context: context)
        let src = valuePointer, dst = out.mutableTyped(T.self), bm = v.typed(UInt8.self)
        for i in 0..<length { dst[i] = Bitmap.isSet(bm, i) ? src[i] : value }
        return MetalArray<T>(length: length, nullCount: 0, validity: nil, values: out, context: context)
    }
}
