import Foundation
import Metal

// Host-side dispatch for the JSON kernels in `JSONSource`. Everything here is plumbing: it binds
// buffers, launches grids and reads back the handful of scalars (counts, first-error positions) the
// reader needs to size the next step. The algorithm is described on `JSONSource`.

/// Kinds a walk assigns to a value (the low four bits of `JSONEntry.flags`).
enum JSONKind: UInt32 {
    case null = 0, falseValue = 1, trueValue = 2, int = 3, float = 4, string = 5, object = 6, array = 7

    static let escapeFlag: UInt32 = 0x10
    static let keyEscapeFlag: UInt32 = 0x20
    static let specialFlag: UInt32 = 0x40

    /// The class pyarrow names in "changed from X to Y" messages.
    var className: String {
        switch self {
        case .null: return "null"
        case .falseValue, .trueValue: return "boolean"
        case .int, .float: return "number"
        case .string: return "string"
        case .object: return "object"
        case .array: return "array"
        }
    }
    /// Kinds that share a class (int and float are both "number").
    var classID: Int {
        switch self {
        case .null: return 0
        case .falseValue, .trueValue: return 1
        case .int, .float: return 2
        case .string: return 3
        case .object: return 4
        case .array: return 5
        }
    }
}

/// Mirror of the MSL `JEntry`: one child value found by a walk.
struct JSONEntry {
    var parent: UInt32
    var keyStart: UInt32
    var keyLen: UInt32
    var valStart: UInt32
    var valLen: UInt32
    var flags: UInt32
    var kind: JSONKind { JSONKind(rawValue: flags & 15) ?? .null }
    static let stride = 24
}

/// RapidJSON's error texts, indexed by the kernels' error codes.
enum JSONSyntax {
    static func message(_ code: UInt32) -> String {
        switch code {
        case 1: return "Invalid value."
        case 2: return "Missing a name for object member."
        case 3: return "Missing a colon after a name of object member."
        case 4: return "Missing a comma or '}' after an object member."
        case 5: return "Missing a comma or ']' after an array element."
        case 6: return "Invalid escape character in string."
        case 7: return "Incorrect hex digit after \\u escape in string."
        case 8: return "The surrogate pair in string is invalid."
        case 9: return "Invalid encoding in string."
        case 10: return "Missing a closing quotation mark in string."
        case 11: return "Miss fraction part in number."
        case 12: return "Miss exponent in number."
        case 13: return "Number too big to be stored in double."
        case 14: return "Nesting deeper than 1024 levels is not supported."
        case 15: return "The document is empty."
        case 16: return "Column() changed from object to array"
        case 17: return "Column() changed from object to string"
        case 18: return "Column() changed from object to number"
        case 19: return "Column() changed from object to boolean"
        default: return "Invalid value."
        }
    }
    /// Top-level codes whose message carries no "in row N" suffix.
    static func hasRow(_ code: UInt32) -> Bool { code != 15 }
}

/// The records a file holds, found by the structure pass.
struct JSONRecords {
    let start: MetalArrowBuffer
    let end: MetalArrowBuffer
    let count: Int
    /// First byte at depth 0 that is neither whitespace nor the start of a record, with its code.
    let topError: (position: Int, code: UInt32)?
}

/// The children of a list of spans, found by one walk.
struct JSONLevel {
    let entries: MetalArrowBuffer
    let count: Int
    /// `spans + 1` int32 offsets: span s owns entries offsets[s] ..< offsets[s + 1].
    let offsets: MetalArrowBuffer
    let spans: Int

    func entry(_ i: Int) -> JSONEntry {
        entries.contents.advanced(by: i * JSONEntry.stride).loadUnaligned(as: JSONEntry.self)
    }
    func offset(_ s: Int) -> Int { Int(offsets.typed(Int32.self)[s]) }
}

enum JSONKernels {
    static func pso(_ ctx: MetalContext, _ fn: String) throws -> MTLComputePipelineState {
        try ctx.pipeline(source: JSONSource.source, function: fn, cacheKey: "json/\(fn)")
    }

    static func alloc(_ ctx: MetalContext, _ bytes: Int, zeroed: Bool = false) throws -> MetalArrowBuffer {
        try MetalArrowBuffer.allocate(byteCount: Swift.max(bytes, 4), zeroed: zeroed, context: ctx)
    }

    static func set(_ enc: MTLComputeCommandEncoder, _ b: MetalArrowBuffer, _ i: Int) {
        enc.setBuffer(b.mtl, offset: b.offset, index: i)
    }

    static func bytes<T>(_ enc: MTLComputeCommandEncoder, _ v: T, _ i: Int) {
        var v = v
        enc.setBytes(&v, length: MemoryLayout<T>.stride, index: i)
    }

    /// Exclusive prefix sum of `n` int32 values: `n + 1` entries, the total last.
    static func sumScan(_ ctx: MetalContext, _ values: MetalArrowBuffer, _ n: Int) throws -> MetalArrowBuffer {
        try MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: values, context: ctx)
            .exclusiveScanToOffsets()
    }

    /// Exclusive prefix maximum of `n` uint32 values.
    static func maxScan(_ ctx: MetalContext, _ values: MetalArrowBuffer, _ n: Int) throws -> MetalArrowBuffer {
        let tg = Dispatch.threadgroupSize
        let groups = Swift.max(1, (n + tg - 1) / tg)
        let out = try alloc(ctx, n * 4)
        let totals = try alloc(ctx, groups * 4)
        let p1 = try pso(ctx, "js_max_block")
        try ctx.run { enc in
            enc.setComputePipelineState(p1)
            set(enc, values, 0); bytes(enc, UInt32(n), 1); set(enc, out, 2); set(enc, totals, 3)
            enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        }
        if groups > 1 {
            let scanned = try maxScan(ctx, totals, groups)
            let p2 = try pso(ctx, "js_max_add")
            try ctx.run { enc in
                enc.setComputePipelineState(p2)
                set(enc, out, 0); set(enc, scanned, 1); bytes(enc, UInt32(n), 2)
                enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            }
        }
        return out
    }

    struct Blk { var n: UInt32; var start: UInt32; var nblocks: UInt32 }

    /// Stage 1: string state, depth and record boundaries for the whole file.
    static func records(_ ctx: MetalContext, _ src: MetalArrowBuffer, n: Int, start: Int) throws -> JSONRecords {
        let nblocks = (n + JSONSource.blockBytes - 1) / JSONSource.blockBytes
        let P = Blk(n: UInt32(n), start: UInt32(start), nblocks: UInt32(nblocks))
        let escKey = try alloc(ctx, nblocks * 4)
        try ctx.run { enc in
            let p = try pso(ctx, "jb_escape")
            enc.setComputePipelineState(p)
            set(enc, src, 0); bytes(enc, P, 1); set(enc, escKey, 2)
            Dispatch.dispatch1D(enc, p, count: nblocks)
        }
        let escIn = try maxScan(ctx, escKey, nblocks)
        let qpar = try alloc(ctx, nblocks * 4)
        try ctx.run { enc in
            let p = try pso(ctx, "jb_quotes")
            enc.setComputePipelineState(p)
            set(enc, src, 0); bytes(enc, P, 1); set(enc, escIn, 2); set(enc, qpar, 3)
            Dispatch.dispatch1D(enc, p, count: nblocks)
        }
        let qScan = try sumScan(ctx, qpar, nblocks)
        let delta = try alloc(ctx, nblocks * 4)
        try ctx.run { enc in
            let p = try pso(ctx, "jb_depth")
            enc.setComputePipelineState(p)
            set(enc, src, 0); bytes(enc, P, 1); set(enc, escIn, 2); set(enc, qScan, 3); set(enc, delta, 4)
            Dispatch.dispatch1D(enc, p, count: nblocks)
        }
        let dScan = try sumScan(ctx, delta, nblocks)
        let counts = try alloc(ctx, nblocks * 4)
        let firstErr = try alloc(ctx, 4)
        firstErr.mutableTyped(UInt32.self)[0] = .max
        let errCode = try alloc(ctx, nblocks * 4, zeroed: true)
        try ctx.run { enc in
            let p = try pso(ctx, "jb_records")
            enc.setComputePipelineState(p)
            set(enc, src, 0); bytes(enc, P, 1); set(enc, escIn, 2); set(enc, qScan, 3); set(enc, dScan, 4)
            set(enc, counts, 5); set(enc, firstErr, 6); set(enc, errCode, 7)
            Dispatch.dispatch1D(enc, p, count: nblocks)
        }
        let base = try sumScan(ctx, counts, nblocks)
        let total = Int(base.typed(Int32.self)[nblocks])
        let recStart = try alloc(ctx, total * 4)
        let recEnd = try alloc(ctx, total * 4)
        // A record whose closing bracket never comes runs to the end of the file; the walk reports it.
        var fill = UInt32(n)
        if total > 0 { memset_pattern4(recEnd.mutableContents, &fill, total * 4) }
        if total > 0 {
            let scratch = try alloc(ctx, 4)
            try ctx.run { enc in
                let p = try pso(ctx, "jb_emit")
                enc.setComputePipelineState(p)
                set(enc, src, 0); bytes(enc, P, 1); set(enc, escIn, 2); set(enc, qScan, 3); set(enc, dScan, 4)
                set(enc, base, 5); set(enc, recStart, 6); set(enc, recEnd, 7); set(enc, scratch, 8); set(enc, scratch, 9)
                Dispatch.dispatch1D(enc, p, count: nblocks)
            }
        }
        let fe = firstErr.typed(UInt32.self)[0]
        var topError: (Int, UInt32)? = nil
        if fe != .max {
            topError = (Int(fe), errCode.typed(UInt32.self)[Int(fe) / JSONSource.blockBytes])
        }
        return JSONRecords(start: recStart, end: recEnd, count: total, topError: topError)
    }

    struct Walk { var n: UInt32; var nspans: UInt32; var emit: UInt32 }

    /// Stage 2, first half: children per span, plus the first span whose walk failed, the error code
    /// and the byte where the walk stopped. A failed span still counts the children it had reached
    /// (see `jw_walk`), so the caller can build the columns a sequential parser would have seen.
    static func walkCount(_ ctx: MetalContext, _ src: MetalArrowBuffer, n: Int, spanStart: MetalArrowBuffer,
                          spanEnd: MetalArrowBuffer, spans: Int)
        throws -> (counts: MetalArrowBuffer, firstError: (span: Int, code: UInt32, position: Int)?) {
        let counts = try alloc(ctx, spans * 4)
        let errCode = try alloc(ctx, spans, zeroed: true)
        let errPos = try alloc(ctx, spans * 4)
        let firstErr = try alloc(ctx, 4)
        firstErr.mutableTyped(UInt32.self)[0] = .max
        if spans > 0 {
            let dummy = try alloc(ctx, JSONEntry.stride)
            try ctx.run { enc in
                let p = try pso(ctx, "jw_walk")
                enc.setComputePipelineState(p)
                set(enc, src, 0); bytes(enc, Walk(n: UInt32(n), nspans: UInt32(spans), emit: 0), 1)
                set(enc, spanStart, 2); set(enc, spanEnd, 3); set(enc, counts, 4); set(enc, counts, 5)
                set(enc, dummy, 6); set(enc, errCode, 7); set(enc, firstErr, 8); set(enc, errPos, 9)
                Dispatch.dispatch1D(enc, p, count: spans)
            }
        }
        let fe = firstErr.typed(UInt32.self)[0]
        guard fe != .max else { return (counts, nil) }
        let i = Int(fe)
        return (counts, (i, UInt32(errCode.typed(UInt8.self)[i]), Int(errPos.typed(UInt32.self)[i])))
    }

    /// Stage 2, second half: writes the entries of the first `spans` spans at their scanned offsets.
    static func walkEmit(_ ctx: MetalContext, _ src: MetalArrowBuffer, n: Int, spanStart: MetalArrowBuffer,
                         spanEnd: MetalArrowBuffer, spans: Int, counts: MetalArrowBuffer) throws -> JSONLevel {
        let offsets = try sumScan(ctx, counts, spans)
        let total = Int(offsets.typed(Int32.self)[spans])
        let entries = try alloc(ctx, total * JSONEntry.stride)
        if total > 0 {
            let scratch = try alloc(ctx, Swift.max(spans, 1) * 4, zeroed: true)
            let firstErr = try alloc(ctx, 4)
            try ctx.run { enc in
                let p = try pso(ctx, "jw_walk")
                enc.setComputePipelineState(p)
                set(enc, src, 0); bytes(enc, Walk(n: UInt32(n), nspans: UInt32(spans), emit: 1), 1)
                set(enc, spanStart, 2); set(enc, spanEnd, 3); set(enc, counts, 4); set(enc, offsets, 5)
                set(enc, entries, 6); set(enc, scratch, 7); set(enc, firstErr, 8); set(enc, scratch, 9)
                Dispatch.dispatch1D(enc, p, count: spans)
            }
        }
        return JSONLevel(entries: entries, count: total, offsets: offsets, spans: spans)
    }

    struct Key { var count: UInt32; var K0: UInt32; var refFirst: UInt32 }

    /// Positional key match against the reference span. Returns the field ids (-1 for novel entries)
    /// and the novel entries' indices in document order.
    static func matchKeys(_ ctx: MetalContext, _ src: MetalArrowBuffer, level: JSONLevel, K0: Int, refFirst: Int)
        throws -> (fid: MetalArrowBuffer, novelBuffer: MetalArrowBuffer, novelCount: Int) {
        let E = level.count
        let fid = try alloc(ctx, E * 4)
        let flag = try alloc(ctx, E * 4)
        let count = try alloc(ctx, 4, zeroed: true)
        try ctx.run { enc in
            let p = try pso(ctx, "jk_match")
            enc.setComputePipelineState(p)
            set(enc, src, 0); bytes(enc, Key(count: UInt32(E), K0: UInt32(K0), refFirst: UInt32(refFirst)), 1)
            set(enc, level.entries, 2); set(enc, level.offsets, 3); set(enc, fid, 4); set(enc, flag, 5)
            set(enc, count, 6)
            Dispatch.dispatch1D(enc, p, count: E)
        }
        // Every entry matched its position in the reference object: nothing to compact.
        guard count.typed(UInt32.self)[0] > 0 else { return (fid, try alloc(ctx, 4), 0) }
        let pos = try sumScan(ctx, flag, E)
        let nNovel = Int(pos.typed(Int32.self)[E])
        let list = try alloc(ctx, nNovel * 4)
        try ctx.run { enc in
            let p = try pso(ctx, "jk_compact")
            enc.setComputePipelineState(p)
            set(enc, flag, 0); set(enc, pos, 1); bytes(enc, UInt32(E), 2); set(enc, list, 3)
            Dispatch.dispatch1D(enc, p, count: E)
        }
        return (fid, list, nNovel)
    }

    /// fid[list[j]] = codes[skip + j]
    static func assign(_ ctx: MetalContext, list: MetalArrowBuffer, count: Int, codes: MetalArrowBuffer, skip: Int,
                       fid: MetalArrowBuffer) throws {
        guard count > 0 else { return }
        try ctx.run { enc in
            let p = try pso(ctx, "jk_assign")
            enc.setComputePipelineState(p)
            set(enc, list, 0); set(enc, codes, 1); bytes(enc, UInt32(count), 2); bytes(enc, UInt32(skip), 3)
            set(enc, fid, 4)
            Dispatch.dispatch1D(enc, p, count: count)
        }
    }

    struct Scatter { var count: UInt32; var rows: UInt32; var fidLo: Int32; var fidHi: Int32; var atomicMode: UInt32; var kindsMode: UInt32 }

    /// Fields per group for which the scatter also gathers the kinds (`KINDS_FIELDS` in the kernel).
    static let kindsFields = 64

    /// Slot matrix for fields [lo, hi): `(hi - lo) * rows` entry indices, -1 where the field is missing.
    /// Returns the first entry that named a field its object had already named, if any, and -- when the
    /// group has at most `kindsFields` fields -- every field's kinds, as `kinds` would compute them.
    static func scatter(_ ctx: MetalContext, level: JSONLevel, fid: MetalArrowBuffer, rows: Int, lo: Int, hi: Int,
                        mayRepeat: Bool, wanted: [Bool])
        throws -> (matrix: MetalArrowBuffer, firstDup: Int?, kinds: [(mask: UInt32, flags: UInt32)]?) {
        let slots = (hi - lo) * rows
        let M = try alloc(ctx, slots * 4)
        let withKinds = hi - lo <= kindsFields
        let kinds = try alloc(ctx, (hi - lo) * 12, zeroed: true)
        let P = Scatter(count: UInt32(level.count), rows: UInt32(rows), fidLo: Int32(lo), fidHi: Int32(hi),
                        atomicMode: mayRepeat ? 1 : 0, kindsMode: withKinds ? 1 : 0)
        try ctx.run { enc in
            if slots > 0 {
                let f = try pso(ctx, "jm_fill")
                enc.setComputePipelineState(f)
                set(enc, M, 0); bytes(enc, UInt32(slots), 1)
                Dispatch.dispatch1D(enc, f, count: slots)
                enc.memoryBarrier(scope: .buffers)
            }
            if level.count > 0 {
                let p = try pso(ctx, "jm_scatter")
                enc.setComputePipelineState(p)
                bytes(enc, P, 0); set(enc, level.entries, 1); set(enc, fid, 2); set(enc, M, 3); set(enc, kinds, 4)
                Dispatch.dispatch1D(enc, p, count: level.count)
            }
        }
        var fieldKinds: [(mask: UInt32, flags: UInt32)]? = nil
        if withKinds {
            let k = kinds.typed(UInt32.self)
            fieldKinds = (0..<(hi - lo)).map { j in
                (k[3 * j] | (Int(k[3 * j + 2]) < rows ? 1 : 0), k[3 * j + 1])
            }
        }
        guard mayRepeat, level.count > 0 else { return (M, nil, fieldKinds) }
        let first = try alloc(ctx, 4)
        first.mutableTyped(UInt32.self)[0] = .max
        let table = try alloc(ctx, hi - lo)
        let tp = table.mutableTyped(UInt8.self)
        for j in 0..<(hi - lo) { tp[j] = wanted[j] ? 1 : 0 }
        try ctx.run { enc in
            let p = try pso(ctx, "jm_dups")
            enc.setComputePipelineState(p)
            bytes(enc, P, 0); set(enc, level.entries, 1); set(enc, fid, 2); set(enc, M, 3); set(enc, first, 4)
            set(enc, table, 5)
            Dispatch.dispatch1D(enc, p, count: level.count)
        }
        let f = first.typed(UInt32.self)[0]
        return (M, f == .max ? nil : Int(f), fieldKinds)
    }

    /// First entry whose field id is not marked expected.
    static func firstUnexpected(_ ctx: MetalContext, level: JSONLevel, fid: MetalArrowBuffer, expected: [UInt8]) throws -> Int? {
        guard level.count > 0 else { return nil }
        let table = try alloc(ctx, expected.count)
        expected.withUnsafeBytes { memcpy(table.mutableContents, $0.baseAddress!, $0.count) }
        let first = try alloc(ctx, 4)
        first.mutableTyped(UInt32.self)[0] = .max
        try ctx.run { enc in
            let p = try pso(ctx, "jm_unexpected")
            enc.setComputePipelineState(p)
            bytes(enc, UInt32(level.count), 0); set(enc, fid, 1); set(enc, table, 2); set(enc, first, 3)
            Dispatch.dispatch1D(enc, p, count: level.count)
        }
        let f = first.typed(UInt32.self)[0]
        return f == .max ? nil : Int(f)
    }

    struct Cols { var rows: UInt32; var cols: UInt32; var identity: UInt32; var validMask: UInt32; var boolValues: UInt32 }

    /// Columns over one level: `cols` columns of `rows` rows each, every row mapped to an entry of
    /// `level` through `rowEntry`, column-major (column j's row r at j * rows + r; -1 where the field is
    /// missing). A nil `rowEntry` is a single column whose row r is entry r (a list's elements).
    struct ColumnSet {
        let level: JSONLevel
        let rowEntry: MetalArrowBuffer?
        let rows: Int
        let cols: Int

        func entryIndex(_ j: Int, _ r: Int) -> Int {
            guard let rowEntry else { return r }
            return Int(rowEntry.typed(Int32.self)[j * rows + r])
        }
        func P(validMask: UInt32 = 0, boolValues: Bool = false) -> Cols {
            Cols(rows: UInt32(rows), cols: UInt32(cols), identity: rowEntry == nil ? 1 : 0,
                 validMask: validMask, boolValues: boolValues ? 1 : 0)
        }
        /// Column j alone (a view, no copy).
        func column(_ j: Int) -> ColumnSet {
            guard let rowEntry, cols > 1 else { return self }
            return ColumnSet(level: level, rowEntry: rowEntry.view(byteOffset: j * rows * 4, byteCount: Swift.max(rows * 4, 4)),
                             rows: rows, cols: 1)
        }
    }

    /// OR of (1 << kind) and of the flag bits, per column.
    static func kinds(_ ctx: MetalContext, _ cs: ColumnSet) throws -> [(mask: UInt32, flags: UInt32)] {
        let out = try alloc(ctx, cs.cols * 8, zeroed: true)
        if cs.rows > 0 && cs.cols > 0 {
            let chunks = (cs.rows + 255) / 256
            try ctx.run { enc in
                let p = try pso(ctx, "jc_kinds")
                enc.setComputePipelineState(p)
                bytes(enc, cs.P(), 0); set(enc, cs.level.entries, 1); set(enc, cs.rowEntry ?? cs.level.entries, 2)
                set(enc, out, 3)
                Dispatch.dispatch1D(enc, p, count: chunks * cs.cols)
            }
        }
        let o = out.typed(UInt32.self)
        return (0..<cs.cols).map { (o[2 * $0], o[2 * $0 + 1]) }
    }

    /// The chosen columns of `set` as a set of their own: a view when they are adjacent, else a copy.
    static func pack(_ ctx: MetalContext, _ cs: ColumnSet, _ sel: [Int]) throws -> ColumnSet {
        guard let rowEntry = cs.rowEntry else { return cs }
        if let first = sel.first, sel.enumerated().allSatisfy({ $0.element == first + $0.offset }) {
            let view = rowEntry.view(byteOffset: first * cs.rows * 4, byteCount: Swift.max(sel.count * cs.rows * 4, 4))
            return ColumnSet(level: cs.level, rowEntry: view, rows: cs.rows, cols: sel.count)
        }
        let total = sel.count * cs.rows
        let out = try alloc(ctx, total * 4)
        let selBuf = try alloc(ctx, sel.count * 4)
        let sp = selBuf.mutableTyped(UInt32.self)
        for (k, s) in sel.enumerated() { sp[k] = UInt32(s) }
        if total > 0 {
            try ctx.run { enc in
                let p = try pso(ctx, "jc_pack")
                enc.setComputePipelineState(p)
                bytes(enc, UInt32(cs.rows), 0); bytes(enc, UInt32(sel.count), 1); set(enc, rowEntry, 2)
                set(enc, selBuf, 3); set(enc, out, 4)
                Dispatch.dispatch1D(enc, p, count: total)
            }
        }
        return ColumnSet(level: cs.level, rowEntry: out, rows: cs.rows, cols: sel.count)
    }

    /// Per-column validity bitmaps (and boolean values when asked) for the rows whose kind is in
    /// `validMask`. Column j's bitmap starts at word j * wordsPerColumn.
    static func validity(_ ctx: MetalContext, _ cs: ColumnSet, validMask: UInt32, boolValues: Bool = false)
        throws -> (validity: MetalArrowBuffer, values: MetalArrowBuffer?, nulls: [Int], wordsPerColumn: Int) {
        let wpc = (cs.rows + 31) / 32
        let words = wpc * cs.cols
        let valid = try alloc(ctx, words * 4)
        let values = boolValues ? try alloc(ctx, words * 4) : nil
        let nulls = try alloc(ctx, cs.cols * 4, zeroed: true)
        if words > 0 {
            try ctx.run { enc in
                let p = try pso(ctx, "jc_validity")
                enc.setComputePipelineState(p)
                bytes(enc, cs.P(validMask: validMask, boolValues: boolValues), 0); set(enc, cs.level.entries, 1)
                set(enc, cs.rowEntry ?? cs.level.entries, 2); set(enc, valid, 3); set(enc, values ?? valid, 4)
                set(enc, nulls, 5)
                Dispatch.dispatch1D(enc, p, count: words)
            }
        }
        let np = nulls.typed(UInt32.self)
        return (valid, values, (0..<cs.cols).map { Int(np[$0]) }, wpc)
    }

    /// Spans of the values of kind in `validMask` in a single column (empty spans elsewhere), for the
    /// next walk.
    static func spans(_ ctx: MetalContext, _ col: ColumnSet, validMask: UInt32) throws -> (MetalArrowBuffer, MetalArrowBuffer) {
        let s = try alloc(ctx, col.rows * 4)
        let e = try alloc(ctx, col.rows * 4)
        if col.rows > 0 {
            try ctx.run { enc in
                let p = try pso(ctx, "jc_spans")
                enc.setComputePipelineState(p)
                bytes(enc, col.P(validMask: validMask), 0); set(enc, col.level.entries, 1)
                set(enc, col.rowEntry ?? col.level.entries, 2); set(enc, s, 3); set(enc, e, 4)
                Dispatch.dispatch1D(enc, p, count: col.rows)
            }
        }
        return (s, e)
    }

    struct Gather { var rows: UInt32; var identity: UInt32; var mode: UInt32; var validMask: UInt32 }

    enum GatherMode: UInt32 { case key = 0, stringValue = 1, rawValue = 2 }

    /// Keys, unescaped string values or raw value text for every row of every column of `set`, as one
    /// offsets array (column j's row r at j * rows + r) over one data buffer. Rows whose kind is not in
    /// `validMask` are empty; the validity is the caller's to attach.
    static func gather(_ ctx: MetalContext, _ src: MetalArrowBuffer, _ cs: ColumnSet, mode: GatherMode,
                       validMask: UInt32) throws -> (offsets: MetalArrowBuffer, data: MetalArrowBuffer) {
        let n = cs.rows * cs.cols
        let P = Gather(rows: UInt32(n), identity: cs.rowEntry == nil ? 1 : 0, mode: mode.rawValue, validMask: validMask)
        let lens = try alloc(ctx, n * 4)
        if n > 0 {
            try ctx.run { enc in
                let p = try pso(ctx, "jg_len")
                enc.setComputePipelineState(p)
                set(enc, src, 0); bytes(enc, P, 1); set(enc, cs.level.entries, 2)
                set(enc, cs.rowEntry ?? cs.level.entries, 3); set(enc, lens, 4)
                Dispatch.dispatch1D(enc, p, count: n)
            }
        }
        if cs.level.entries.byteCount > 0, src.byteCount > Int(Int32.max) {
            // The offsets are 32-bit: on an input past 2 GiB, add the lengths up before trusting the scan.
            let lp = lens.typed(Int32.self)
            var sum = 0
            for i in 0..<n { sum += Int(lp[i]) }
            guard sum <= Int(Int32.max) else {
                throw JSONError.unsupported("a column set holds \(sum) bytes of text; utf8 offsets are 32-bit")
            }
        }
        let offsets = try sumScan(ctx, lens, n)
        let total = Int(offsets.typed(Int32.self)[n])
        let data = try alloc(ctx, total)
        if n > 0 && total > 0 {
            try ctx.run { enc in
                let p = try pso(ctx, "jg_write")
                enc.setComputePipelineState(p)
                set(enc, src, 0); bytes(enc, P, 1); set(enc, cs.level.entries, 2)
                set(enc, cs.rowEntry ?? cs.level.entries, 3); set(enc, offsets, 4); set(enc, data, 5)
                Dispatch.dispatch1D(enc, p, count: n)
            }
        }
        return (offsets, data)
    }

    /// A utf8 array of the keys of the given entries (unescaped), in the order given.
    static func keys(_ ctx: MetalContext, _ src: MetalArrowBuffer, level: JSONLevel, indices: MetalArrowBuffer, count: Int)
        throws -> MetalStringArray {
        let cs = ColumnSet(level: level, rowEntry: indices, rows: count, cols: 1)
        let (off, data) = try gather(ctx, src, cs, mode: .key, validMask: 0xFF)
        return MetalStringArray(length: count, nullCount: 0, validity: nil, offsets: off, data: data, context: ctx)
    }

    struct Time { var rows: UInt32; var cols: UInt32; var unitScale: Int64; var fracMax: UInt32 }

    /// ISO-8601 parse (Arrow's rules) of `cols` utf8 columns of `rows` rows sharing one offsets array,
    /// with word-aligned validity bitmaps. Returns the values and, per column, how many valid rows did
    /// not parse and the first of them.
    static func timestamps(_ ctx: MetalContext, offsets: MetalArrowBuffer, data: MetalArrowBuffer,
                           validity: MetalArrowBuffer, rows: Int, cols: Int, unit: ArrowTemporalUnit)
        throws -> (values: MetalArrowBuffer, fails: [(count: Int, first: Int?)]) {
        let values = try alloc(ctx, rows * cols * 8)
        let fails = try alloc(ctx, cols * 8)
        let fp = fails.mutableTyped(UInt32.self)
        for j in 0..<cols { fp[2 * j] = 0; fp[2 * j + 1] = .max }
        let fracMax: UInt32
        switch unit { case .second: fracMax = 0; case .milli: fracMax = 3; case .micro: fracMax = 6; case .nano: fracMax = 9 }
        let P = Time(rows: UInt32(rows), cols: UInt32(cols), unitScale: unit.perSecond, fracMax: fracMax)
        let words = (rows + 31) / 32 * cols
        if words > 0 {
            try ctx.run { enc in
                let p = try pso(ctx, "jt_parse")
                enc.setComputePipelineState(p)
                set(enc, offsets, 0); set(enc, data, 1); set(enc, validity, 2)
                enc.setBytes([P], length: MemoryLayout<Time>.stride, index: 3)
                set(enc, values, 4); set(enc, fails, 5)
                Dispatch.dispatch1D(enc, p, count: words)
            }
        }
        let f = fails.typed(UInt32.self)
        return (values, (0..<cols).map { (Int(f[2 * $0]), f[2 * $0 + 1] == .max ? nil : Int(f[2 * $0 + 1])) })
    }
}

