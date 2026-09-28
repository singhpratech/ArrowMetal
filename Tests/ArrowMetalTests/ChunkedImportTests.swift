import XCTest
import CArrowABI
@testable import ArrowMetal

// `importArrowChunks` against `importArrowArray` of the concatenation: value for value and null for
// null over every type the chunked import takes, and over chunk layouts a producer can hand over
// (empty chunks, one-row chunks, sliced chunks at bit offsets that are not byte aligned, absent and
// present validity side by side, null counts of -1, all-null chunks, one chunk, 10,000 chunks),
// then through sum, sort and group-by.

// MARK: - A producer: malloc'd buffers behind a release callback

/// Buffers a test producer allocated. Chunks of one sliced array share one of these.
private final class ProducerMemory {
    nonisolated(unsafe) static var live = 0
    let bases: [UnsafeMutableRawPointer?]
    let pointers: [UnsafeRawPointer?]
    /// `shift` bytes in front of every buffer, so none is page aligned (the reference import copies).
    init(_ buffers: [[UInt8]?], shift: Int = 8) {
        var b: [UnsafeMutableRawPointer?] = [], p: [UnsafeRawPointer?] = []
        for buf in buffers {
            guard let buf else { b.append(nil); p.append(nil); continue }
            let m = malloc(buf.count + shift + 64)!
            memset(m, 0xA5, buf.count + shift + 64)
            buf.withUnsafeBytes { if $0.count > 0 { memcpy(m + shift, $0.baseAddress!, $0.count) } }
            b.append(m); p.append(UnsafeRawPointer(m + shift))
        }
        bases = b; pointers = p
        ProducerMemory.live += 1
    }
    deinit { for m in bases { free(m) }; ProducerMemory.live -= 1 }
}

/// The private data of one exported chunk: its memory and its buffer pointer list.
private final class ProducerChunk {
    let memory: ProducerMemory
    let list: UnsafeMutablePointer<UnsafeRawPointer?>
    init(_ memory: ProducerMemory) {
        self.memory = memory
        list = .allocate(capacity: Swift.max(memory.pointers.count, 1))
        for (i, p) in memory.pointers.enumerated() { list[i] = p }
    }
    deinit { list.deallocate() }
}

private func producerArray(_ memory: ProducerMemory, length: Int, offset: Int, nullCount: Int) -> ArrowArray {
    let holder = ProducerChunk(memory)
    var a = ArrowArray()
    a.length = Int64(length); a.null_count = Int64(nullCount); a.offset = Int64(offset)
    a.n_buffers = Int64(memory.pointers.count); a.n_children = 0
    a.buffers = holder.list
    a.children = nil; a.dictionary = nil
    a.private_data = Unmanaged.passRetained(holder).toOpaque()
    a.release = { p in
        guard let p, let pd = p.pointee.private_data else { return }
        Unmanaged<ProducerChunk>.fromOpaque(pd).release()
        p.pointee.release = nil
    }
    return a
}

// MARK: - Column model

/// One logical row: its value bytes, or nil for null. Booleans are one byte, 0 or 1.
private typealias Row = [UInt8]?

private struct TypeSpec {
    let format: String
    /// Bytes per value for a fixed-width type, 0 for bool, -1 offsets (int32), -2 offsets (int64), -3 views, -4 null.
    let width: Int
    var isString: Bool { width <= -1 && width >= -3 }
}

private let specs: [TypeSpec] = [
    TypeSpec(format: "c", width: 1), TypeSpec(format: "C", width: 1), TypeSpec(format: "s", width: 2),
    TypeSpec(format: "S", width: 2), TypeSpec(format: "i", width: 4), TypeSpec(format: "I", width: 4),
    TypeSpec(format: "l", width: 8), TypeSpec(format: "L", width: 8), TypeSpec(format: "f", width: 4),
    TypeSpec(format: "g", width: 8), TypeSpec(format: "e", width: 2), TypeSpec(format: "b", width: 0),
    TypeSpec(format: "tdD", width: 4), TypeSpec(format: "tdm", width: 8), TypeSpec(format: "tsu:UTC", width: 8),
    TypeSpec(format: "tDn", width: 8), TypeSpec(format: "tts", width: 4), TypeSpec(format: "d:20,3", width: 16),
    TypeSpec(format: "d:40,2,256", width: 32), TypeSpec(format: "d:9,2,32", width: 4), TypeSpec(format: "tiM", width: 4),
    TypeSpec(format: "tin", width: 16), TypeSpec(format: "w:5", width: 5),
    TypeSpec(format: "u", width: -1), TypeSpec(format: "z", width: -1), TypeSpec(format: "U", width: -2),
    TypeSpec(format: "Z", width: -2), TypeSpec(format: "vu", width: -3), TypeSpec(format: "vz", width: -3),
    TypeSpec(format: "n", width: -4),
]

private struct Rng {
    var s: UInt64
    mutating func next() -> UInt64 { s &+= 0x9E3779B97F4A7C15; var z = s; z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9; z = (z ^ (z >> 27)) &* 0x94D049BB133111EB; return z ^ (z >> 31) }
    mutating func int(_ n: Int) -> Int { n <= 0 ? 0 : Int(next() % UInt64(n)) }
    mutating func byte() -> UInt8 { UInt8(truncatingIfNeeded: next()) }
}

private func randomRow(_ t: TypeSpec, _ rng: inout Rng, allowNull: Bool = true) -> Row {
    if t.width == -4 { return nil }
    if allowNull && rng.int(5) == 0 { return nil }
    switch t.width {
    case 0: return [UInt8(rng.int(2))]
    case let w where w > 0:
        var v = (0..<w).map { _ in rng.byte() }
        // Temporal values within range of their unit (date64 days in ms, time32 seconds in a day).
        if t.format == "tdm" { let d = Int64(rng.int(40_000) - 20_000) * 86_400_000; v = withUnsafeBytes(of: d) { Array($0) } }
        if t.format == "tts" { let s = Int32(rng.int(86_400)); v = withUnsafeBytes(of: s) { Array($0) } }
        if t.format == "d:9,2,32" { let s = Int32(rng.int(1_000_000_000) - 500_000_000); v = withUnsafeBytes(of: s) { Array($0) } }
        return v
    default:
        // Short strings stay inline in the view layout, long ones go to a data buffer.
        let len = rng.int(3) == 0 ? 13 + rng.int(40) : rng.int(13)
        return (0..<len).map { _ in UInt8(97 + rng.int(26)) }
    }
}

/// Buffers for `rows` stored after `pad` junk rows, so the Arrow offset is `pad`.
private func buffers(_ t: TypeSpec, rows: [Row], pad: Int, withValidity: Bool, rng: inout Rng,
                     viewBufferBytes: Int = 64) -> [[UInt8]?] {
    let all: [Row] = (0..<pad).map { _ in randomRow(t, &rng) } + rows
    var validity: [UInt8]? = nil
    if withValidity {
        var bm = [UInt8](repeating: 0, count: (all.count + 7) / 8 + 1)
        for (i, r) in all.enumerated() where r != nil { bm[i >> 3] |= 1 << (i & 7) }
        // Junk above the last row: a reader must not look there.
        bm[bm.count - 1] |= 0xF0
        validity = bm
    }
    switch t.width {
    case -4:
        return []
    case 0:
        var bits = [UInt8](repeating: 0, count: (all.count + 7) / 8 + 1)
        for (i, r) in all.enumerated() where (r?[0] ?? UInt8(i & 1)) == 1 { bits[i >> 3] |= 1 << (i & 7) }
        return [validity, bits]
    case let w where w > 0:
        var v: [UInt8] = []
        for r in all { v += r ?? (0..<w).map { _ in rng.byte() } }
        return [validity, v]
    case -1, -2:
        var data: [UInt8] = (0..<rng.int(9)).map { _ in rng.byte() }       // bytes before the first row
        var offs: [Int] = [data.count]
        for r in all { data += r ?? []; offs.append(data.count) }
        let o: [UInt8] = t.width == -1
            ? offs.flatMap { v in withUnsafeBytes(of: Int32(v)) { Array($0) } }
            : offs.flatMap { v in withUnsafeBytes(of: Int64(v)) { Array($0) } }
        return [validity, o, data]
    default:
        var views: [UInt8] = []
        var datas: [[UInt8]] = [[]]
        for r in all {
            var v = [UInt8](repeating: 0, count: 16)
            let b = r ?? (rng.int(2) == 0 ? [] : (0..<20).map { _ in rng.byte() })   // a null row's view is junk
            withUnsafeBytes(of: Int32(b.count)) { for k in 0..<4 { v[k] = $0[k] } }
            if b.count <= 12 {
                for (k, x) in b.enumerated() { v[4 + k] = x }
            } else {
                if datas[datas.count - 1].count + b.count > viewBufferBytes { datas.append([]) }
                for k in 0..<4 { v[4 + k] = b[k] }
                withUnsafeBytes(of: Int32(datas.count - 1)) { for k in 0..<4 { v[8 + k] = $0[k] } }
                withUnsafeBytes(of: Int32(datas[datas.count - 1].count)) { for k in 0..<4 { v[12 + k] = $0[k] } }
                datas[datas.count - 1] += b
            }
            views += v
        }
        let sizes = datas.flatMap { d in withUnsafeBytes(of: Int64(d.count)) { Array($0) } }
        return [validity, views] + datas.map { Optional($0) } + [sizes]
    }
}

// MARK: - Reading an imported array back through the C Data Interface

private func fixedWidth(_ fmt: String) -> Int? {
    switch fmt {
    case "c", "C": return 1
    case "s", "S", "e": return 2
    case "i", "I", "f", "tdD", "tts", "ttm", "tiM": return 4
    case "l", "L", "g", "tdm", "ttu", "ttn", "tiD": return 8
    case "tin": return 16
    default: break
    }
    if fmt.hasPrefix("ts") || fmt.hasPrefix("tD") { return 8 }
    if fmt.hasPrefix("w:") { return Int(fmt.dropFirst(2)) }
    if fmt.hasPrefix("d:") {
        if fmt.hasSuffix(",256") { return 32 }
        if fmt.hasSuffix(",32") { return 4 }
        if fmt.hasSuffix(",64") { return 8 }
        return 16
    }
    return nil
}

/// Every row of `a` as bytes (nil for null), plus its exported format and null count.
private func readBack(_ a: AnyMetalArray) -> (format: String, rows: [Row], nullCount: Int) {
    var s = ArrowSchema(), arr = ArrowArray()
    a.exportArrowSchema(into: &s)
    a.exportArrowArray(into: &arr)
    defer {
        if let r = arr.release { r(&arr) }
        if let r = s.release { r(&s) }
    }
    let fmt = String(cString: s.format)
    let n = Int(arr.length), off = Int(arr.offset)
    let bm = arr.n_buffers > 0 ? arr.buffers[0].map { $0.assumingMemoryBound(to: UInt8.self) } : nil
    func valid(_ i: Int) -> Bool {
        if fmt == "n" { return false }
        return bm.map { Bitmap.isSet($0, off + i) } ?? true
    }
    var rows: [Row] = []
    for i in 0..<n {
        guard valid(i) else { rows.append(nil); continue }
        switch fmt {
        case "b":
            rows.append([Bitmap.isSet(arr.buffers[1]!.assumingMemoryBound(to: UInt8.self), off + i) ? 1 : 0])
        case "u", "z":
            let o = arr.buffers[1]!.assumingMemoryBound(to: Int32.self), d = arr.buffers[2]
            let lo = Int(o[off + i]), hi = Int(o[off + i + 1])
            rows.append(hi > lo ? Array(UnsafeRawBufferPointer(start: d! + lo, count: hi - lo)) : [])
        case "vu", "vz":
            let nb = Int(arr.n_buffers), k = nb - 3
            let v = arr.buffers[1]! + (off + i) * 16
            let len = Int(v.loadUnaligned(as: Int32.self))
            if len <= 12 { rows.append(Array(UnsafeRawBufferPointer(start: v + 4, count: Swift.max(len, 0)))); continue }
            let idx = Int(v.loadUnaligned(fromByteOffset: 8, as: Int32.self))
            let o = Int(v.loadUnaligned(fromByteOffset: 12, as: Int32.self))
            let sizes = arr.buffers[nb - 1]!.assumingMemoryBound(to: Int64.self)
            if idx >= 0, idx < k, o >= 0, o + len <= Int(sizes[idx]), let d = arr.buffers[2 + idx] {
                rows.append(Array(UnsafeRawBufferPointer(start: d + o, count: len)))
            } else {
                rows.append([])
            }
        default:
            let w = fixedWidth(fmt)!
            rows.append(Array(UnsafeRawBufferPointer(start: arr.buffers[1]! + (off + i) * w, count: w)))
        }
    }
    return (fmt, rows, a.nullCount)
}

// MARK: - Tests

final class ChunkedImportTests: XCTestCase {

    /// How one chunk is exported.
    private struct ChunkPlan {
        var rows: [Row]
        var pad: Int
        var absentValidity: Bool
        var unknownNullCount: Bool
    }

    /// Plans chunks of the given sizes over random rows, with random pads and validity choices.
    private func plan(_ t: TypeSpec, sizes: [Int], rng: inout Rng, allNullEvery: Int = 7) -> [ChunkPlan] {
        sizes.enumerated().map { (ci, n) in
            let allNull = allNullEvery > 0 && ci % allNullEvery == 3
            let noNulls = !allNull && rng.int(3) == 0
            let rows: [Row] = (0..<n).map { _ in allNull ? nil : randomRow(t, &rng, allowNull: !noNulls) }
            let anyNull = rows.contains { $0 == nil }
            return ChunkPlan(rows: rows, pad: rng.int(3) == 0 ? 0 : rng.int(70),
                             absentValidity: !anyNull && rng.int(2) == 0, unknownNullCount: rng.int(3) == 0)
        }
    }

    private func export(_ t: TypeSpec, _ p: ChunkPlan, rng: inout Rng, viewBufferBytes: Int = 64) -> ArrowArray {
        let bufs = buffers(t, rows: p.rows, pad: p.pad, withValidity: !p.absentValidity && t.width != -4, rng: &rng,
                           viewBufferBytes: viewBufferBytes)
        let nulls = p.rows.filter { $0 == nil }.count
        return producerArray(ProducerMemory(bufs), length: p.rows.count, offset: p.pad,
                             nullCount: p.unknownNullCount ? -1 : nulls)
    }

    private func schema(_ fmt: String, _ body: (UnsafePointer<ArrowSchema>) throws -> Void) rethrows {
        let f = strdup(fmt)!
        defer { free(f) }
        var s = ArrowSchema()
        s.format = UnsafePointer(f)
        try withUnsafePointer(to: &s) { try body($0) }
    }

    /// The chunked import of `plans` against the import of their concatenation and against the rows.
    @discardableResult
    private func check(_ t: TypeSpec, _ plans: [ChunkPlan], rng: inout Rng, viewBufferBytes: Int = 64,
                       _ what: String, file: StaticString = #filePath, line: UInt = #line) throws -> AnyMetalArray {
        let rows = plans.flatMap(\.rows)
        var chunks = plans.map { export(t, $0, rng: &rng, viewBufferBytes: viewBufferBytes) }
        let concat = ChunkPlan(rows: rows, pad: 0, absentValidity: false, unknownNullCount: false)
        var whole = export(t, concat, rng: &rng)
        var got: AnyMetalArray! = nil, want: AnyMetalArray! = nil
        try schema(t.format) { s in
            got = try chunks.withUnsafeMutableBufferPointer { try importArrowChunks(schema: s, arrays: $0.baseAddress, count: $0.count).array }
            want = try importArrowArray(schema: s, array: &whole).array
        }
        for c in chunks { XCTAssertNil(c.release, "\(what): a chunk was not moved", file: file, line: line) }
        let g = readBack(got), w = readBack(want)
        XCTAssertEqual(g.format, w.format, "\(what): format", file: file, line: line)
        XCTAssertEqual(got.length, rows.count, "\(what): length", file: file, line: line)
        XCTAssertEqual(g.nullCount, w.nullCount, "\(what): null count", file: file, line: line)
        XCTAssertEqual(g.nullCount, rows.filter { $0 == nil }.count, "\(what): null count vs rows", file: file, line: line)
        XCTAssertEqual(g.rows.count, w.rows.count, file: file, line: line)
        var bad = 0
        for i in 0..<Swift.min(g.rows.count, rows.count) where g.rows[i] != w.rows[i] || g.rows[i] != rows[i] {
            if bad < 3 { XCTFail("\(what) [\(t.format)] row \(i): chunked \(String(describing: g.rows[i])), concatenated \(String(describing: w.rows[i])), expected \(String(describing: rows[i]))", file: file, line: line) }
            bad += 1
        }
        return got
    }

    func testEveryTypeOverChunkLayouts() throws {
        var rng = Rng(s: 42)
        let layouts: [(String, [Int])] = [
            ("one chunk", [150]),
            ("mixed with empty and one-row chunks", [0, 1, 7, 0, 64, 1, 100, 13, 0, 3, 33, 1]),
            ("two chunks", [37, 91]),
            ("only empty chunks", [0, 0, 0]),
            ("no chunks", []),
            ("one non-empty among empty", [0, 57, 0]),
        ]
        for t in specs {
            for (name, sizes) in layouts {
                let plans = plan(t, sizes: sizes, rng: &rng)
                try check(t, plans, rng: &rng, "\(name)")
            }
        }
        XCTAssertEqual(ProducerMemory.live, 0, "every producer buffer was released")
    }

    func testTenThousandChunks() throws {
        var rng = Rng(s: 7)
        for f in ["l", "g", "b", "i", "u", "U", "vu", "d:20,3", "tsu:UTC"] {
            let t = specs.first { $0.format == f }!
            let sizes = (0..<10_000).map { _ in rng.int(4) }
            // More than `maxWrappedViewBuffers` distinct data buffers: the copy path.
            try check(t, plan(t, sizes: sizes, rng: &rng, allNullEvery: 11), rng: &rng, "10,000 chunks")
        }
        XCTAssertEqual(ProducerMemory.live, 0)
    }

    func testAllNullAndAllValidChunks() throws {
        var rng = Rng(s: 11)
        for f in ["l", "b", "u", "vu", "g"] {
            let t = specs.first { $0.format == f }!
            let valid = { (n: Int, rng: inout Rng) in (0..<n).map { _ in randomRow(t, &rng, allowNull: false) } }
            var plans = [
                ChunkPlan(rows: valid(20, &rng), pad: 3, absentValidity: true, unknownNullCount: false),
                ChunkPlan(rows: [Row](repeating: nil, count: 17), pad: 5, absentValidity: false, unknownNullCount: true),
                ChunkPlan(rows: valid(9, &rng), pad: 0, absentValidity: false, unknownNullCount: false),
                ChunkPlan(rows: [Row](repeating: nil, count: 64), pad: 64, absentValidity: false, unknownNullCount: false),
            ]
            try check(t, plans, rng: &rng, "all-null next to all-valid")
            // No chunk can hold a null: no validity bitmap at all.
            plans = [ChunkPlan(rows: valid(30, &rng), pad: 1, absentValidity: true, unknownNullCount: false),
                     ChunkPlan(rows: valid(30, &rng), pad: 9, absentValidity: true, unknownNullCount: false)]
            let got = try check(t, plans, rng: &rng, "no nulls")
            if case .int64(let a) = got { XCTAssertNil(a.validity) }
        }
    }

    /// A present bitmap with a declared null count of 0 counts as all valid.
    func testZeroNullCountWithBitmap() throws {
        var rng = Rng(s: 3)
        let t = specs.first { $0.format == "i" }!
        let rows: [Row] = (0..<40).map { _ in randomRow(t, &rng, allowNull: false) }
        var a = producerArray(ProducerMemory(buffers(t, rows: Array(rows[0..<20]), pad: 2, withValidity: true, rng: &rng)),
                              length: 20, offset: 2, nullCount: 0)
        var b = producerArray(ProducerMemory(buffers(t, rows: Array(rows[20...]), pad: 0, withValidity: false, rng: &rng)),
                              length: 20, offset: 0, nullCount: 0)
        try schema("i") { s in
            var ptrs = [UnsafeMutablePointer<ArrowArray>]()
            try withUnsafeMutablePointer(to: &a) { pa in
                try withUnsafeMutablePointer(to: &b) { pb in
                    ptrs = [pa, pb]
                    let r = try importArrowChunks(schema: s, arrays: ptrs).array
                    XCTAssertEqual(readBack(r).rows, rows)
                    XCTAssertEqual(r.nullCount, 0)
                }
            }
        }
    }

    /// Chunks that are slices of one array share its data buffers: each is taken once, and mapped.
    func testSlicesOfOneViewArrayShareDataBuffers() throws {
        var rng = Rng(s: 5)
        for f in ["vu", "vz"] {
            let t = specs.first { $0.format == f }!
            let rows: [Row] = (0..<1000).map { _ in randomRow(t, &rng) }
            let bufs = buffers(t, rows: rows, pad: 0, withValidity: true, rng: &rng, viewBufferBytes: 4096)
            let memory = ProducerMemory(bufs)
            var chunks: [ArrowArray] = []
            var start = 0
            while start < rows.count {
                let n = Swift.min(rows.count - start, 1 + rng.int(90))
                chunks.append(producerArray(memory, length: n, offset: start, nullCount: -1))
                start += n
            }
            try schema(f) { s in
                let got = try chunks.withUnsafeMutableBufferPointer { try importArrowChunks(schema: s, arrays: $0.baseAddress, count: $0.count).array }
                XCTAssertEqual(readBack(got).rows, rows)
                guard let sa = got.asStringArray, let v = sa.view else { return XCTFail("not a view column") }
                XCTAssertEqual(v.dataBuffers.count, bufs.count - 3, "each shared data buffer is taken once")
                XCTAssertEqual(v.logicalBytes, rows.reduce(0) { $0 + ($1?.count ?? 0) })
            }
        }
    }

    /// A single view array with more data buffers than `maxWrappedViewBuffers` is imported through
    /// the merged-copy path: same rows, same nulls, same byte total, a malformed view still empty.
    func testSingleViewArrayWithManyDataBuffers() throws {
        var rng = Rng(s: 13)
        for f in ["vu", "vz"] {
            let t = specs.first { $0.format == f }!
            let rows: [Row] = (0..<3000).map { _ in randomRow(t, &rng) }
            for pad in [0, 5, 64] {
                var bufs = buffers(t, rows: rows, pad: pad, withValidity: true, rng: &rng, viewBufferBytes: 40)
                XCTAssertGreaterThan(bufs.count - 3, maxWrappedViewBuffers)
                // Row 0's view (if out of line) points past its buffer: it must read as empty.
                var want = rows
                if let r = rows[0], r.count > 12 {
                    bufs[1]![(pad * 16) + 12] = 0xFF; bufs[1]![(pad * 16) + 13] = 0x7F
                    want[0] = []
                }
                var a = producerArray(ProducerMemory(bufs), length: rows.count, offset: pad, nullCount: pad == 5 ? -1 : rows.filter { $0 == nil }.count)
                try schema(f) { s in
                    let got = try importArrowArray(schema: s, array: &a).array
                    XCTAssertNil(a.release)
                    let r = readBack(got)
                    XCTAssertEqual(r.rows, want, "\(f) pad \(pad)")
                    XCTAssertEqual(r.nullCount, rows.filter { $0 == nil }.count)
                    guard let v = got.asStringArray?.view else { return XCTFail("not a view column") }
                    XCTAssertEqual(v.logicalBytes, want.reduce(0) { $0 + ($1?.count ?? 0) })
                    XCTAssertLessThan(v.dataBuffers.count, 3, "the data buffers were merged")
                }
            }
        }
        XCTAssertEqual(ProducerMemory.live, 0)
    }

    func testViewLogicalBytesAndMalformedViews() throws {
        var rng = Rng(s: 9)
        let t = specs.first { $0.format == "vu" }!
        let plans = plan(t, sizes: (0..<200).map { _ in rng.int(30) }, rng: &rng)
        let got = try check(t, plans, rng: &rng, "view logical bytes")
        guard case .string(let s) = got, let v = s.view else { return XCTFail("not a view column") }
        XCTAssertEqual(v.logicalBytes, plans.flatMap(\.rows).reduce(0) { $0 + ($1?.count ?? 0) })

        // A view pointing past its buffer reads as the empty string, as after the single import.
        var bufs = buffers(t, rows: [Array("abcdefghijklmnopqrstuvwxyz".utf8)], pad: 0, withValidity: false, rng: &rng)
        bufs[1]![12] = 0xFF   // offset far past the buffer
        var a = producerArray(ProducerMemory(bufs), length: 1, offset: 0, nullCount: 0)
        var b = export(t, ChunkPlan(rows: [Array("short".utf8)], pad: 0, absentValidity: true, unknownNullCount: false), rng: &rng)
        var bad = producerArray(ProducerMemory(bufs), length: 1, offset: 0, nullCount: 0)
        try schema("vu") { s in
            try withUnsafeMutablePointer(to: &a) { pa in
                try withUnsafeMutablePointer(to: &b) { pb in
                    let r = try importArrowChunks(schema: s, arrays: [pa, pb]).array
                    let single = try importArrowArray(schema: s, array: &bad).array
                    XCTAssertEqual(readBack(r).rows, [[], Array("short".utf8)])
                    XCTAssertEqual(readBack(single).rows, [[]])
                }
            }
        }
    }

    func testUnsupportedTypesMoveNothing() throws {
        var rng = Rng(s: 1)
        let t = specs.first { $0.format == "i" }!
        var chunks = [export(t, ChunkPlan(rows: [[1, 0, 0, 0]], pad: 0, absentValidity: true, unknownNullCount: false), rng: &rng),
                      export(t, ChunkPlan(rows: [[2, 0, 0, 0]], pad: 0, absentValidity: true, unknownNullCount: false), rng: &rng)]
        // A dictionary schema (int32 indices): not taken.
        let vf = strdup("u")!, kf = strdup("i")!
        defer { free(vf); free(kf) }
        var dict = ArrowSchema(); dict.format = UnsafePointer(vf)
        var s = ArrowSchema(); s.format = UnsafePointer(kf)
        try withUnsafeMutablePointer(to: &dict) { d in
            s.dictionary = d
            try withUnsafePointer(to: &s) { sp in
                XCTAssertFalse(chunkedImportSupported(schema: sp))
                XCTAssertThrowsError(try chunks.withUnsafeMutableBufferPointer { try importArrowChunks(schema: sp, arrays: $0.baseAddress, count: $0.count) })
            }
        }
        for c in chunks { XCTAssertNotNil(c.release, "an unsupported import moves nothing") }
        for i in chunks.indices { chunks[i].release!(&chunks[i]) }
        schema("+l") { XCTAssertFalse(chunkedImportSupported(schema: $0)) }
        schema("l") { XCTAssertTrue(chunkedImportSupported(schema: $0)) }
    }

    /// utf8 chunks whose data totals 2 GB or more are refused, as one large_utf8 array that size is,
    /// and nothing is moved. The offsets claim the bytes; no data is read before the check.
    func testUtf8OverTwoGigabytesIsRefused() throws {
        let off: [UInt8] = [Int32(0), Int32(1_500_000_000)].flatMap { v in withUnsafeBytes(of: v) { Array($0) } }
        var chunks = (0..<2).map { _ in producerArray(ProducerMemory([nil, off, [1, 2, 3]]), length: 1, offset: 0, nullCount: 0) }
        try schema("u") { s in
            XCTAssertThrowsError(try chunks.withUnsafeMutableBufferPointer { try importArrowChunks(schema: s, arrays: $0.baseAddress, count: $0.count) }) { e in
                XCTAssertTrue("\(e)".contains("2 GB"), "\(e)")
            }
        }
        for i in chunks.indices { XCTAssertNotNil(chunks[i].release); chunks[i].release!(&chunks[i]) }
    }

    /// The merged arrays run through sum, sort and group-by like any imported array.
    func testKernelsOnChunkedArrays() throws {
        try requireRealGPU()
        var rng = Rng(s: 21)
        let n = 50_000
        let sizes: [Int] = { var s: [Int] = [], left = n; while left > 0 { let k = Swift.min(left, 1 + rng.int(3000)); s.append(k); left -= k }; return s }()
        func chunksOf<T>(_ values: [T?], format: String) -> [ChunkPlan] {
            var start = 0
            return sizes.map { k in
                let rows: [Row] = values[start..<start + k].map { v in v.map { x in withUnsafeBytes(of: x) { Array($0) } } }
                start += k
                return ChunkPlan(rows: rows, pad: rng.int(40), absentValidity: !rows.contains { $0 == nil } && rng.int(2) == 0,
                                 unknownNullCount: rng.int(2) == 0)
            }
        }
        let ints: [Int64?] = (0..<n).map { i in i % 13 == 0 ? nil : Int64(rng.int(2_000_000)) - 1_000_000 }
        let dbls: [Double?] = (0..<n).map { i in i % 17 == 0 ? nil : Double(rng.int(1_000_000)) / 8 }
        let keys: [Int32?] = (0..<n).map { _ in Int32(rng.int(100)) }

        let tI = specs.first { $0.format == "l" }!, tD = specs.first { $0.format == "g" }!, tK = specs.first { $0.format == "i" }!
        guard case .int64(let ci) = try check(tI, chunksOf(ints, format: "l"), rng: &rng, "int64 for kernels"),
              case .float64(let cd) = try check(tD, chunksOf(dbls, format: "g"), rng: &rng, "float64 for kernels"),
              case .int32(let ck) = try check(tK, chunksOf(keys, format: "i"), rng: &rng, "keys for kernels") else {
            return XCTFail("unexpected array types")
        }
        let ri = try MetalArray<Int64>(ints), rd = try MetalArray<Double>(dbls), rk = try MetalArray<Int32>(keys)
        // Sum.
        XCTAssertEqual("\(String(describing: try ci.sum()))", "\(String(describing: try ri.sum()))")
        XCTAssertEqual("\(String(describing: try cd.sum()))", "\(String(describing: try rd.sum()))")
        // Sort.
        XCTAssertEqual(try ci.sorted().toArray(), try ri.sorted().toArray())
        XCTAssertEqual(try cd.argsort().toArray(), try rd.argsort().toArray())
        // Group-by.
        XCTAssertEqual(try ck.groupBy(keyCount: 100).sum(ci).toArray(), try rk.groupBy(keyCount: 100).sum(ri).toArray())
        XCTAssertEqual(try ck.groupBy(keyCount: 100).count().toArray(), try rk.groupBy(keyCount: 100).count().toArray())

        // Strings, as offsets and as views: sort by the string key.
        let words: [String?] = (0..<n).map { i in i % 19 == 0 ? nil : "w\(rng.int(5000))" + (i % 3 == 0 ? "-a-longer-suffix" : "") }
        let ref = try MetalStringArray(words)
        for f in ["u", "vu"] {
            let t = specs.first { $0.format == f }!
            var start = 0
            let plans = sizes.map { k -> ChunkPlan in
                let rows: [Row] = words[start..<start + k].map { $0.map { Array($0.utf8) } }
                start += k
                return ChunkPlan(rows: rows, pad: rng.int(40), absentValidity: false, unknownNullCount: true)
            }
            guard case .string(let s) = try check(t, plans, rng: &rng, "\(f) for kernels") else { return XCTFail() }
            XCTAssertEqual(try s.argsort().toArray(), try ref.argsort().toArray(), "\(f) argsort")
        }
    }
}

private extension AnyMetalArray {
    var asStringArray: MetalStringArray? {
        switch self { case .string(let s), .binary(let s): return s; default: return nil }
    }
}
