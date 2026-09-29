import Foundation
import Metal
import CArrowABI

// Chunked import: several Arrow arrays of one type (a pyarrow / Polars ChunkedArray, a column of
// arrow-rs RecordBatches) taken as one ArrowMetal array of their total length.
//
// Each chunk's buffers are written straight into the final buffers (shared, page-aligned
// `MetalArrowBuffer`s), on the CPU cores in parallel; there is no intermediate concatenated copy.
// The work is split by output position, not by chunk, so a column of 10,000 small chunks and a
// column of 16 large ones both spread over every core, and no two threads write the same byte:
//
// * fixed-width values: one `memcpy` per chunk piece, from the chunk's Arrow offset.
// * bitmaps (validity, and boolean values): assembled 64 output bits at a time from each chunk's
//   bits at its own bit offset, since a chunk boundary is generally not byte aligned. A chunk with
//   no bitmap, or a null count of 0, contributes set bits. The null count is counted from the
//   merged bitmap.
// * utf8 / binary (and the large variants): each chunk's offsets are rebased by a prefix sum of the
//   chunks' data lengths and narrowed to int32, and the data bytes copied. A column whose data
//   totals 2 GB or more is refused, as the single-array import refuses a large_utf8 array that
//   size: every string kernel reads int32 offsets.
// * utf8_view / binary_view: the views are copied with each out-of-line view's buffer index and
//   offset rewritten to the merged data buffers; inline views are copied as they are. Data buffers
//   shared between chunks (the chunks of one sliced array) are taken once. A column with up to
//   `maxWrappedViewBuffers` distinct data buffers maps them without a copy where the single-array
//   import would; with more, they are copied back to back into data buffers of under 2 GB each. The
//   byte total of the non-null strings is summed on the CPU during the view pass.
//
// Dictionary, nested, run-end encoded, list-view and extension arrays are not taken
// (`chunkedImportSupported` is false and nothing is read or moved); the caller concatenates those
// and imports the result.

/// Most distinct view data buffers a chunked import maps without a copy; with more, they are copied.
let maxWrappedViewBuffers = 64

/// What a chunked import builds for one Arrow type.
private enum ChunkLayout {
    /// `width` bytes per element plus validity; `make` wraps the merged buffers in the array type.
    case fixed(width: Int, make: (Int, MetalArrowBuffer?, MetalArrowBuffer, Int, MetalContext) throws -> AnyMetalArray)
    case boolean
    case null
    case offsets(large: Bool, binary: Bool)
    case views(binary: Bool)

    var bufferCount: Int? {
        switch self {
        case .fixed, .boolean: return 2
        case .offsets: return 3
        case .null, .views: return nil
        }
    }
}

private func primitive<T: ArrowPrimitive>(_: T.Type, _ wrap: @escaping (MetalArray<T>) throws -> AnyMetalArray) -> ChunkLayout {
    .fixed(width: T.byteWidth) { n, validity, values, nulls, ctx in
        try wrap(MetalArray<T>(length: n, nullCount: nulls, validity: validity, values: values, context: ctx))
    }
}

private func chunkLayout(_ schema: UnsafePointer<ArrowSchema>) -> ChunkLayout? {
    guard let f = schema.pointee.format else { return nil }
    let fmt = String(cString: f)
    guard schema.pointee.dictionary == nil, schema.pointee.n_children == 0, extensionInfo(schema) == nil else { return nil }
    switch fmt {
    case "c": return primitive(Int8.self) { .int8($0) }
    case "C": return primitive(UInt8.self) { .uint8($0) }
    case "s": return primitive(Int16.self) { .int16($0) }
    case "S": return primitive(UInt16.self) { .uint16($0) }
    case "i": return primitive(Int32.self) { .int32($0) }
    case "I": return primitive(UInt32.self) { .uint32($0) }
    case "l": return primitive(Int64.self) { .int64($0) }
    case "L": return primitive(UInt64.self) { .uint64($0) }
    case "f": return primitive(Float.self) { .float32($0) }
    case "g": return primitive(Double.self) { .float64($0) }
    case "e": return primitive(UInt16.self) { .float16(MetalFloat16Array(bits: $0)) }
    case "b": return .boolean
    case "n": return .null
    case "u": return .offsets(large: false, binary: false)
    case "U": return .offsets(large: true, binary: false)
    case "z": return .offsets(large: false, binary: true)
    case "Z": return .offsets(large: true, binary: true)
    case "vu": return .views(binary: false)
    case "vz": return .views(binary: true)
    default: break
    }
    // The order follows `importArrowArray`: the interval formats ("ti…") before the temporal ones,
    // decimal32/64 before decimal128/256.
    if let t = ArrowSmallDecimalType(format: fmt) {
        if t.bitWidth == 32 { return primitive(Int32.self) { .smallDecimal(try MetalSmallDecimalArray(type: t, $0)) } }
        return primitive(Int64.self) { .smallDecimal(try MetalSmallDecimalArray(type: t, $0)) }
    }
    if let unit = ArrowIntervalUnit(rawValue: fmt) {
        return .fixed(width: unit.byteWidth) { n, v, vals, nulls, ctx in
            .interval(MetalIntervalArray(unit: unit, length: n, nullCount: nulls, validity: v, values: vals, context: ctx))
        }
    }
    if fmt.hasPrefix("w:") {
        guard let w = Int(fmt.dropFirst(2)), w >= 0 else { return nil }
        return .fixed(width: w) { n, v, vals, nulls, ctx in
            .fixedBinary(MetalFixedBinaryArray(byteWidth: w, length: n, nullCount: nulls, validity: v, values: vals, context: ctx))
        }
    }
    if fmt.hasPrefix("d:") {
        guard let t = try? ArrowDecimalType.parse(fmt) else { return nil }
        return .fixed(width: t.byteWidth) { n, v, vals, nulls, ctx in
            .decimal(MetalDecimalArray(type: t, length: n, nullCount: nulls, validity: v, values: vals, context: ctx))
        }
    }
    if fmt.hasPrefix("t") {
        guard let t = try? ArrowTemporalType.parse(fmt) else { return nil }
        if t.usesInt64 { return primitive(Int64.self) { .temporal(try MetalTemporalArray(type: t, $0)) } }
        return primitive(Int32.self) { .temporal(try MetalTemporalArray(type: t, $0)) }
    }
    return nil
}

/// Whether `importArrowChunks` takes arrays of this schema's type. When it is false the chunked
/// import refuses before reading or moving anything, and the caller concatenates the chunks instead.
public func chunkedImportSupported(schema: UnsafePointer<ArrowSchema>) -> Bool { chunkLayout(schema) != nil }

/// One non-empty chunk, borrowed from its producer until the import succeeds.
private struct ChunkRef {
    let array: UnsafeMutablePointer<ArrowArray>
    /// First row of this chunk in the merged array.
    let rowStart: Int
    let length: Int
    /// The chunk's Arrow offset (applied on read).
    let offset: Int
    /// The validity bitmap when the chunk may hold nulls; nil when every row is valid.
    let validity: UnsafePointer<UInt8>?
    var rowEnd: Int { rowStart + length }
    func buffer(_ i: Int) -> UnsafeRawPointer? { array.pointee.buffers[i].map { UnsafeRawPointer($0) } }
}

/// Keeps the producers' arrays alive for buffers mapped without a copy.
private final class ChunkOwners {
    var arrays: [ImportedCArray] = []
}

/// Imports several arrays of one type as one array of their total length.
///
/// `schema` describes every chunk. Each chunk's offset, length, validity bitmap (or its absence) and
/// null count (-1 included) are honoured. The result is the array the import of the chunks'
/// concatenation gives, value for value and null for null, built without that concatenation.
///
/// Ownership follows the C Data Interface: on success every chunk has been moved (its `release` is
/// nil on return). On failure a chunk whose `release` is still set remains the caller's. When the
/// type is not supported (`chunkedImportSupported`) nothing is moved. With one chunk, or one
/// non-empty chunk among empty ones, this is `importArrowArray` of that chunk, zero-copy when its
/// buffers allow it.
public func importArrowChunks(schema: UnsafePointer<ArrowSchema>, arrays: [UnsafeMutablePointer<ArrowArray>],
                              context: MetalContext = .shared) throws -> ImportResult {
    guard let layout = chunkLayout(schema) else {
        let fmt = schema.pointee.format.map { String(cString: $0) } ?? "(null format)"
        throw ArrowMetalError.unsupportedType("chunked import does not take \(fmt) arrays; import their concatenation")
    }
    // Validate every chunk before anything is read, copied or moved.
    var seen = Set<UnsafeMutablePointer<ArrowArray>>()
    seen.reserveCapacity(arrays.count)
    for (i, p) in arrays.enumerated() {
        let a = p.pointee
        guard a.release != nil else { throw ArrowMetalError.releasedArray }
        // One struct passed twice would be moved (and released) twice.
        guard seen.insert(p).inserted else { throw ArrowMetalError.invalidArrowArray("chunk \(i): the same ArrowArray twice") }
        guard a.length >= 0, a.offset >= 0 else {
            throw ArrowMetalError.invalidArrowArray("chunk \(i): negative length or offset")
        }
        guard a.n_children == 0, a.dictionary == nil else {
            throw ArrowMetalError.invalidArrowArray("chunk \(i): children or a dictionary on a flat type")
        }
        if let nb = layout.bufferCount {
            guard a.n_buffers == nb, a.buffers != nil else {
                throw ArrowMetalError.invalidArrowArray("chunk \(i): expected \(nb) buffers, got \(a.n_buffers)")
            }
        }
        switch layout {
        case .null:
            guard a.n_buffers <= 1 else { throw ArrowMetalError.invalidArrowArray("chunk \(i): a null array has no buffers") }
        case .views:
            guard a.n_buffers >= 3, a.buffers != nil else {
                throw ArrowMetalError.invalidArrowArray("chunk \(i): expected at least 3 buffers for a view array")
            }
        default:
            if a.length > 0, a.buffers[1] == nil { throw ArrowMetalError.invalidArrowArray("chunk \(i): buffer 1 is null") }
        }
    }
    if arrays.count == 1 { return try importArrowArray(schema: schema, array: arrays[0], context: context) }
    let nonEmpty = arrays.filter { $0.pointee.length > 0 }
    if nonEmpty.count == 1 {
        for p in arrays where p.pointee.length == 0 { releaseChunk(p) }
        return try importArrowArray(schema: schema, array: nonEmpty[0], context: context)
    }

    var refs: [ChunkRef] = []
    refs.reserveCapacity(nonEmpty.count)
    var row = 0
    for p in nonEmpty {
        let a = p.pointee
        let n = Int(a.length)
        var validity: UnsafePointer<UInt8>? = nil
        if case .null = layout {} else if a.null_count != 0, let v = a.buffers[0] {
            validity = UnsafeRawPointer(v).assumingMemoryBound(to: UInt8.self)
        }
        refs.append(ChunkRef(array: p, rowStart: row, length: n, offset: Int(a.offset), validity: validity))
        let (sum, overflow) = row.addingReportingOverflow(n)
        guard !overflow else { throw ArrowMetalError.invalidArrowArray("chunk lengths overflow") }
        row = sum
    }
    let total = row
    let owners = ChunkOwners()
    var keepProducers = false

    let result: AnyMetalArray
    switch layout {
    case .null:
        result = .null(MetalNullArray(length: total, context: context))
    case .fixed(let width, let make):
        let (valueBytes, overflow) = total.multipliedReportingOverflow(by: width)
        guard !overflow else { throw ArrowMetalError.invalidArrowArray("chunk lengths overflow") }
        let values = try MetalArrowBuffer.allocate(byteCount: Swift.max(valueBytes, 1), zeroed: false, context: context)
        if width > 0 {
            let dst = values.mutableContents
            parallelCopy(refs.map { c in
                CopySegment(src: c.buffer(1)!.advanced(by: c.offset * width), dst: dst.advanced(by: c.rowStart * width),
                            count: c.length * width)
            })
        }
        let (validity, nulls) = try mergedValidity(refs, total: total, context: context)
        result = try make(total, validity, values, nulls, context)
    case .boolean:
        let values = try allocateBitmap(bits: total, context: context)
        _ = mergeBits(into: values, bits: total, parts: refs.map {
            BitPart(bits: $0.buffer(1)!.assumingMemoryBound(to: UInt8.self), bitOffset: $0.offset, rowStart: $0.rowStart, length: $0.length)
        })
        let (validity, nulls) = try mergedValidity(refs, total: total, context: context)
        result = .boolean(MetalBooleanArray(length: total, nullCount: nulls, validity: validity, values: values, context: context))
    case .offsets(let large, let binary):
        let s = try mergeOffsetStrings(refs, total: total, large: large, context: context)
        s.isBinary = binary
        result = binary ? .binary(s) : .string(s)
    case .views(let binary):
        let (s, wrapped) = try mergeViewStrings(refs, total: total, binary: binary, owners: owners, context: context)
        keepProducers = wrapped
        result = binary ? .binary(s) : .string(s)
    }

    // Success: every chunk is moved. Copied chunks are released now; chunks whose buffers were mapped
    // stay alive as long as the array does.
    for p in arrays {
        if keepProducers { owners.arrays.append(ImportedCArray(moving: p)) } else { releaseChunk(p) }
    }
    return ImportResult(array: result, zeroCopy: false)
}

/// `importArrowChunks` over `count` contiguous `ArrowArray` structs.
public func importArrowChunks(schema: UnsafePointer<ArrowSchema>, arrays: UnsafeMutablePointer<ArrowArray>?, count: Int,
                              context: MetalContext = .shared) throws -> ImportResult {
    guard count >= 0, count == 0 || arrays != nil else { throw ArrowMetalError.invalidArrowArray("chunk array is null") }
    return try importArrowChunks(schema: schema, arrays: (0..<count).map { arrays! + $0 }, context: context)
}

/// A single `utf8_view` / `binary_view` array (already moved into `owner`) through the chunked view
/// path: its views are copied with their buffer indices rewritten and its data buffers copied into
/// merged ones (or mapped, when at most `maxWrappedViewBuffers` distinct ones remain). The single
/// import takes this path for an array with more than `maxWrappedViewBuffers` data buffers, where
/// mapping each buffer and binding each one for the GPU pass cost more than copying them.
func importViewArrayMerged(_ owner: ImportedCArray, binary: Bool, context: MetalContext) throws -> ImportResult {
    let owners = ChunkOwners()
    owners.arrays.append(owner)
    return try withUnsafeMutablePointer(to: &owner.array) { p in
        let a = p.pointee
        let validity = a.null_count != 0 ? a.buffers[0].map { UnsafeRawPointer($0).assumingMemoryBound(to: UInt8.self) } : nil
        let ref = ChunkRef(array: p, rowStart: 0, length: Int(a.length), offset: Int(a.offset), validity: validity)
        let (s, _) = try mergeViewStrings([ref], total: Int(a.length), binary: binary, owners: owners, context: context)
        return ImportResult(array: binary ? .binary(s) : .string(s), zeroCopy: false)
    }
}

/// Releases a consumed chunk: calls its release callback and marks it released.
private func releaseChunk(_ p: UnsafeMutablePointer<ArrowArray>) {
    if let rel = p.pointee.release { rel(p) }
    p.pointee.release = nil
}

// MARK: - Parallel pieces

private let chunkWorkers = Swift.max(1, Int(ProcessInfo.processInfo.environment["ARROWMETAL_IMPORT_THREADS"] ?? "")
                                        ?? ProcessInfo.processInfo.activeProcessorCount)

/// Splits `[0, n)` into at most one shard per core, each at least `grain` long, runs `body(lo, hi)`
/// on them concurrently and returns the sum of what the shards return.
private func parallelSum(_ n: Int, grain: Int, _ body: (Int, Int) -> Int) -> Int {
    guard n > 0 else { return 0 }
    let shards = Swift.max(1, Swift.min(chunkWorkers, (n + grain - 1) / grain))
    if shards == 1 { return body(0, n) }
    let step = (n + shards - 1) / shards
    let partial = UnsafeMutablePointer<Int>.allocate(capacity: shards)
    partial.initialize(repeating: 0, count: shards)
    defer { partial.deallocate() }
    DispatchQueue.concurrentPerform(iterations: shards) { s in
        let lo = s * step, hi = Swift.min(lo + step, n)
        if lo < hi { partial[s] = body(lo, hi) }
    }
    var sum = 0
    for s in 0..<shards { sum += partial[s] }
    return sum
}

/// Index of the last entry of `starts` (ascending) that is `<= x`.
@inline(__always) private func lastStart(_ starts: UnsafeBufferPointer<Int>, atMost x: Int) -> Int {
    var lo = 0, hi = starts.count - 1
    while lo < hi {
        let mid = (lo + hi + 1) >> 1
        if starts[mid] <= x { lo = mid } else { hi = mid - 1 }
    }
    return lo
}

struct CopySegment {
    let src: UnsafeRawPointer
    let dst: UnsafeMutableRawPointer
    let count: Int
}

/// Copies every segment, split by output byte across the cores (a large segment is shared by
/// several threads, many small ones go to one).
private func parallelCopy(_ segments: [CopySegment]) {
    let segs = segments.filter { $0.count > 0 }
    guard !segs.isEmpty else { return }
    var starts = [Int](repeating: 0, count: segs.count)
    var total = 0
    for (i, s) in segs.enumerated() { starts[i] = total; total += s.count }
    starts.withUnsafeBufferPointer { st in
        segs.withUnsafeBufferPointer { sg in
            _ = parallelSum(total, grain: 1 << 20) { lo, hi in
                var k = lastStart(st, atMost: lo)
                var pos = lo
                while pos < hi {
                    let s = sg[k]
                    let segEnd = st[k] + s.count
                    if pos >= segEnd { k += 1; continue }
                    let n = Swift.min(hi, segEnd) - pos
                    let within = pos - st[k]
                    memcpy(s.dst + within, s.src + within, n)
                    pos += n
                }
                return 0
            }
        }
    }
}

/// A chunk's bits: `bits` from bit `bitOffset`, or all set when `bits` is nil.
private struct BitPart {
    let bits: UnsafePointer<UInt8>?
    let bitOffset: Int
    let rowStart: Int
    let length: Int
}

/// A bitmap buffer for `bits` bits, written whole 64-bit words at a time, with the bytes after the
/// last word zeroed.
private func allocateBitmap(bits: Int, context: MetalContext) throws -> MetalArrowBuffer {
    let bytes = Bitmap.byteCount(bits: bits)
    let buf = try MetalArrowBuffer.allocate(byteCount: Swift.max(bytes, 1), zeroed: false, context: context)
    let written = (bits + 63) / 64 * 8
    let room = buf.mtl.length - buf.offset
    if room > written { memset(buf.mutableContents + written, 0, room - written) }
    return buf
}

/// Assembles `parts` (ascending, back to back from row 0) into `dst`, 64 output bits per step, and
/// returns the number of set bits.
private func mergeBits(into dst: MetalArrowBuffer, bits total: Int, parts: [BitPart]) -> Int {
    guard total > 0, !parts.isEmpty else { return 0 }
    let out = dst.mutableContents.assumingMemoryBound(to: UInt64.self)
    let words = (total + 63) / 64
    let starts = parts.map(\.rowStart)
    return starts.withUnsafeBufferPointer { st in
        parts.withUnsafeBufferPointer { pp in
            parallelSum(words, grain: 1 << 13) { w0, w1 in
                let end = Swift.min(w1 * 64, total)
                var pos = w0 * 64
                var k = lastStart(st, atMost: pos)
                var acc: UInt64 = 0
                var ones = 0
                while pos < end {
                    let p = pp[k]
                    let pEnd = p.rowStart + p.length
                    if pos >= pEnd { k += 1; continue }
                    let sh = pos & 63
                    let take = Swift.min(64 - sh, pEnd - pos, end - pos)
                    let w: UInt64
                    if let b = p.bits {
                        w = Bitmap.word(b, at: p.bitOffset + pos - p.rowStart, count: take)
                    } else {
                        w = take == 64 ? UInt64.max : (UInt64(1) << UInt64(take)) - 1
                    }
                    acc |= w << UInt64(sh)
                    pos += take
                    if pos & 63 == 0 || pos == end {
                        out[(pos - 1) >> 6] = acc
                        ones += acc.nonzeroBitCount
                        acc = 0
                    }
                }
                return ones
            }
        }
    }
}

/// The merged validity bitmap and null count; nil and 0 when no chunk can hold a null.
private func mergedValidity(_ refs: [ChunkRef], total: Int, context: MetalContext) throws -> (MetalArrowBuffer?, Int) {
    guard refs.contains(where: { $0.validity != nil }) else { return (nil, 0) }
    let buf = try allocateBitmap(bits: total, context: context)
    let valid = mergeBits(into: buf, bits: total, parts: refs.map {
        BitPart(bits: $0.validity, bitOffset: $0.offset, rowStart: $0.rowStart, length: $0.length)
    })
    return (buf, total - valid)
}

// MARK: - utf8 / binary

private func mergeOffsetStrings(_ refs: [ChunkRef], total: Int, large: Bool,
                                context: MetalContext) throws -> MetalStringArray {
    // Per chunk: the first and last byte its rows use, and where its bytes start in the merged data.
    var starts = [Int](), bases = [Int]()
    starts.reserveCapacity(refs.count); bases.reserveCapacity(refs.count)
    var bytes = 0
    for (i, c) in refs.enumerated() {
        let op = c.buffer(1)!
        let s: Int, e: Int
        if large {
            let o = op.assumingMemoryBound(to: Int64.self)
            s = Int(o[c.offset]); e = Int(o[c.offset + c.length])
        } else {
            let o = op.assumingMemoryBound(to: Int32.self)
            s = Int(o[c.offset]); e = Int(o[c.offset + c.length])
        }
        guard s >= 0, e >= s else { throw ArrowMetalError.invalidArrowArray("chunk \(i): offsets run backwards") }
        if e > s, c.buffer(2) == nil { throw ArrowMetalError.invalidArrowArray("chunk \(i): data buffer is null") }
        starts.append(s); bases.append(bytes)
        let (sum, overflow) = bytes.addingReportingOverflow(e - s)
        bytes = sum
        // The limit the large_utf8 import has: every string kernel reads int32 offsets.
        guard !overflow, bytes < Int(Int32.max) else {
            throw ArrowMetalError.unsupportedType("\(large ? "large_" : "")utf8 chunks over 2 GB in total")
        }
    }
    let offsets = try MetalArrowBuffer.allocate(byteCount: (total + 1) * 4, zeroed: false, context: context)
    let data = try MetalArrowBuffer.allocate(byteCount: Swift.max(bytes, 1), zeroed: false, context: context)
    let out = offsets.mutableTyped(Int32.self)
    let rowStarts = refs.map(\.rowStart)
    rowStarts.withUnsafeBufferPointer { rs in
        refs.withUnsafeBufferPointer { rf in
            _ = parallelSum(total, grain: 1 << 16) { r0, r1 in
                var k = lastStart(rs, atMost: r0)
                var r = r0
                while r < r1 {
                    let c = rf[k]
                    if r >= c.rowEnd { k += 1; continue }
                    let n = Swift.min(r1, c.rowEnd) - r
                    let i0 = c.offset + (r - c.rowStart)
                    let delta = bases[k] - starts[k]
                    let o = out + r
                    if large {
                        let src = c.buffer(1)!.assumingMemoryBound(to: Int64.self) + i0
                        for j in 0..<n { o[j] = Int32(truncatingIfNeeded: Int(src[j]) &+ delta) }
                    } else {
                        let src = c.buffer(1)!.assumingMemoryBound(to: Int32.self) + i0
                        let d = Int32(truncatingIfNeeded: delta)
                        for j in 0..<n { o[j] = src[j] &+ d }
                    }
                    r += n
                }
                return 0
            }
        }
    }
    out[total] = Int32(bytes)
    let dp = data.mutableContents
    parallelCopy(refs.enumerated().compactMap { (k, c) in
        guard let src = c.buffer(2) else { return nil }
        let e = k + 1 < refs.count ? bases[k + 1] : bytes
        return CopySegment(src: src + starts[k], dst: dp + bases[k], count: e - bases[k])
    })
    let (validity, nulls) = try mergedValidity(refs, total: total, context: context)
    return MetalStringArray(length: total, nullCount: nulls, validity: validity, offsets: offsets, data: data, context: context)
}

// MARK: - utf8_view / binary_view

/// A producer data buffer, identified by address and declared size so chunks sharing it take it once.
private struct DataKey: Hashable {
    let address: UInt
    let size: Int
}

private func mergeViewStrings(_ refs: [ChunkRef], total: Int, binary: Bool, owners: ChunkOwners,
                              context: MetalContext) throws -> (MetalStringArray, wrapped: Bool) {
    // 1. The distinct data buffers, and per chunk the distinct id of each of its buffers (-1: none).
    var ids: [DataKey: Int] = [:]
    var distinct: [(ptr: UnsafeRawPointer, size: Int)] = []
    var chunkIds: [[Int]] = [], chunkSizes: [[Int]] = []
    for (i, c) in refs.enumerated() {
        let nb = Int(c.array.pointee.n_buffers)
        let k = nb - 3
        guard c.buffer(1) != nil else { throw ArrowMetalError.invalidArrowArray("chunk \(i): views buffer is null") }
        var cid = [Int](repeating: -1, count: k), csz = [Int](repeating: 0, count: k)
        if k > 0 {
            guard let sp = c.buffer(nb - 1)?.assumingMemoryBound(to: Int64.self) else {
                throw ArrowMetalError.invalidArrowArray("chunk \(i): utf8_view buffer sizes are null")
            }
            for j in 0..<k {
                let size = Int(sp[j])
                guard size >= 0 else { throw ArrowMetalError.invalidArrowArray("chunk \(i): negative data buffer size") }
                csz[j] = size
                guard size > 0, let p = c.buffer(2 + j) else { continue }
                let key = DataKey(address: UInt(bitPattern: p), size: size)
                if let id = ids[key] { cid[j] = id } else {
                    cid[j] = distinct.count; ids[key] = distinct.count; distinct.append((p, size))
                }
            }
        }
        chunkIds.append(cid); chunkSizes.append(csz)
    }

    // 2. Where each distinct buffer lands: (merged buffer index, byte base inside it).
    var placeIndex = [Int32](repeating: 0, count: distinct.count), placeBase = [Int32](repeating: 0, count: distinct.count)
    var dataBuffers: [MetalArrowBuffer?] = [], dataSizes: [Int] = []
    var segments: [CopySegment] = []
    var copied = 0
    var wrapped = false
    if distinct.count <= maxWrappedViewBuffers {
        // Few buffers: map each as the single-array import does, copying only one that cannot be mapped.
        for (d, b) in distinct.enumerated() {
            placeIndex[d] = Int32(d)
            if let w = MetalArrowBuffer.wrapCovering(b.ptr, byteCount: b.size, keepAlive: owners, context: context) {
                dataBuffers.append(w); wrapped = true
            } else {
                let buf = try MetalArrowBuffer.allocate(byteCount: b.size, zeroed: false, context: context)
                segments.append(CopySegment(src: b.ptr, dst: buf.mutableContents, count: b.size))
                copied += b.size
                dataBuffers.append(buf)
            }
            dataSizes.append(b.size)
        }
    } else {
        // Many buffers: packed back to back into merged buffers of under 2 GB (a view offset is int32).
        let cap = Int(Int32.max)
        var groups: [[Int]] = [[]], fill = [0]
        for (d, b) in distinct.enumerated() {
            if fill[fill.count - 1] > 0, fill[fill.count - 1] + b.size > cap { groups.append([]); fill.append(0) }
            placeIndex[d] = Int32(groups.count - 1)
            placeBase[d] = Int32(fill[fill.count - 1])
            groups[groups.count - 1].append(d)
            fill[fill.count - 1] += b.size
        }
        for (g, members) in groups.enumerated() {
            let buf = try MetalArrowBuffer.allocate(byteCount: Swift.max(fill[g], 1), zeroed: false, context: context)
            for d in members {
                segments.append(CopySegment(src: distinct[d].ptr, dst: buf.mutableContents + Int(placeBase[d]), count: distinct[d].size))
            }
            copied += fill[g]
            dataBuffers.append(buf); dataSizes.append(fill[g])
        }
    }
    parallelCopy(segments)

    // 3. Validity first: the byte total counts the non-null rows only.
    let (validity, nulls) = try mergedValidity(refs, total: total, context: context)
    let views = try MetalArrowBuffer.allocate(byteCount: Swift.max(total, 1) * 16, zeroed: false, context: context)
    copied += total * 16 + (validity.map { $0.byteCount } ?? 0)

    // 4. The views: inline ones copied, out-of-line ones pointed at the merged buffers. A view whose
    //    buffer index or byte range is out of bounds gets an index no buffer has, so it reads as the
    //    empty string, as it does after the single-array import.
    let chunkMap: [[(index: Int32, base: Int32, size: Int)]] = zip(chunkIds, chunkSizes).map { cid, csz in
        zip(cid, csz).map { id, size in id < 0 ? (Int32.max, 0, 0) : (placeIndex[id], placeBase[id], size) }
    }
    let vbits = validity?.typed(UInt8.self)
    let outViews = views.mutableContents
    let rowStarts = refs.map(\.rowStart)
    let logical = rowStarts.withUnsafeBufferPointer { rs in
        refs.withUnsafeBufferPointer { rf in
            parallelSum(total, grain: 1 << 15) { r0, r1 in
                var k = lastStart(rs, atMost: r0)
                var r = r0
                var acc = 0
                while r < r1 {
                    let c = rf[k]
                    if r >= c.rowEnd { k += 1; continue }
                    let n = Swift.min(r1, c.rowEnd) - r
                    let src = c.buffer(1)! + (c.offset + (r - c.rowStart)) * 16
                    let dst = outViews + r * 16
                    chunkMap[k].withUnsafeBufferPointer { map in
                        for j in 0..<n {
                            let s = src + j * 16, d = dst + j * 16
                            let lo = s.loadUnaligned(as: UInt64.self)
                            let hi = s.loadUnaligned(fromByteOffset: 8, as: UInt64.self)
                            let len = UInt32(truncatingIfNeeded: lo)
                            let valid = vbits.map { Bitmap.isSet($0, r + j) } ?? true
                            d.storeBytes(of: lo, as: UInt64.self)
                            if len <= 12 {
                                d.storeBytes(of: hi, toByteOffset: 8, as: UInt64.self)
                                if valid { acc += Int(len) }
                                continue
                            }
                            let idx = Int32(truncatingIfNeeded: hi)
                            let off = Int32(truncatingIfNeeded: hi >> 32)
                            if idx >= 0, Int(idx) < map.count, off >= 0 {
                                let m = map[Int(idx)]
                                if m.index != Int32.max, Int(off) + Int(len) <= m.size {
                                    d.storeBytes(of: m.index, toByteOffset: 8, as: Int32.self)
                                    d.storeBytes(of: m.base &+ off, toByteOffset: 12, as: Int32.self)
                                    if valid { acc += Int(len) }
                                    continue
                                }
                            }
                            d.storeBytes(of: Int32.max, toByteOffset: 8, as: Int32.self)
                            d.storeBytes(of: off, toByteOffset: 12, as: Int32.self)
                        }
                    }
                    r += n
                }
                return acc
            }
        }
    }
    // The same limit the single-array view import has.
    guard logical < Int(Int32.max) else {
        throw ArrowMetalError.unsupportedType("\(binary ? "binary" : "utf8")_view over 2 GB")
    }
    let storage = try StringViewStorage(views: views, dataBuffers: dataBuffers, dataSizes: dataSizes, zeroCopy: false,
                                        copiedBytes: copied, logicalBytes: logical, context: context)
    let s = MetalStringArray(length: total, nullCount: nulls, validity: validity, view: storage, context: context)
    s.isBinary = binary
    return (s, wrapped)
}
