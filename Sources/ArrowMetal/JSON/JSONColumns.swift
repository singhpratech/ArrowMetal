import Foundation
import Metal

/// Turns the entries of one walk into Arrow columns, recursing into nested objects and arrays.
///
/// A *level* is the output of one walk: the children of a list of spans, one span per row of the
/// column being built (the records at the top, the nested objects or arrays one level down). Object
/// levels become the fields of a struct (the top level is the record batch); array levels become a
/// list's child column with the walk's offsets as the list offsets.
///
/// Errors that pyarrow raises while parsing (type conflicts, repeated fields, unexpected fields) are
/// collected with the byte position of the value that triggered them rather than thrown, so the reader
/// can report the one a sequential parser would have met first. Conversion errors (explicit-schema
/// values that do not fit their type) come after all of them, as in pyarrow.
final class JSONColumnBuilder {
    let ctx: MetalContext
    let source: MetalArrowBuffer
    /// Input bytes (the source buffer may be padded past them).
    let n: Int
    let host: UnsafePointer<UInt8>
    let behavior: JSONUnexpectedFieldBehavior
    var parseErrors: [(position: Int, message: String)] = []
    var conversionErrors: [(position: Int, message: String)] = []

    /// Slot-matrix budget per group of fields; wider tables are processed in groups. A variable so the
    /// tests can force groups on a small file.
    nonisolated(unsafe) static var matrixBudgetBytes = 512 << 20

    init(context: MetalContext, source: MetalArrowBuffer, n: Int, host: UnsafePointer<UInt8>,
         behavior: JSONUnexpectedFieldBehavior) {
        self.ctx = context
        self.source = source
        self.n = n
        self.host = host
        self.behavior = behavior
    }

    // MARK: - host helpers (error paths and small tables only)

    /// Decodes a string body (validated by the walk) on the host.
    func decode(_ start: Int, _ len: Int, escaped: Bool) -> String {
        let p = host + start
        if !escaped { return String(decoding: UnsafeBufferPointer(start: p, count: len), as: UTF8.self) }
        var out: [UInt8] = []
        out.reserveCapacity(len)
        func hex4(_ i: Int) -> UInt32 {
            var v: UInt32 = 0
            for k in 0..<4 {
                let c = p[i + k]
                let h: UInt32 = c >= 0x61 ? UInt32(c) - 0x61 + 10 : (c >= 0x41 ? UInt32(c) - 0x41 + 10 : UInt32(c) - 0x30)
                v = (v << 4) | h
            }
            return v
        }
        var i = 0
        while i < len {
            let c = p[i]
            if c != 0x5C { out.append(c); i += 1; continue }
            let e = p[i + 1]
            if e != 0x75 {
                switch e {
                case 0x62: out.append(0x08)
                case 0x66: out.append(0x0C)
                case 0x6E: out.append(0x0A)
                case 0x72: out.append(0x0D)
                case 0x74: out.append(0x09)
                default: out.append(e)
                }
                i += 2
                continue
            }
            var cp = hex4(i + 2)
            i += 6
            if cp >= 0xD800 && cp <= 0xDBFF {
                let lo = hex4(i + 2)
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
                i += 6
            }
            if cp < 0x80 { out.append(UInt8(cp)) }
            else if cp < 0x800 { out += [UInt8(0xC0 | (cp >> 6)), UInt8(0x80 | (cp & 0x3F))] }
            else if cp < 0x10000 {
                out += [UInt8(0xE0 | (cp >> 12)), UInt8(0x80 | ((cp >> 6) & 0x3F)), UInt8(0x80 | (cp & 0x3F))]
            } else {
                out += [UInt8(0xF0 | (cp >> 18)), UInt8(0x80 | ((cp >> 12) & 0x3F)),
                        UInt8(0x80 | ((cp >> 6) & 0x3F)), UInt8(0x80 | (cp & 0x3F))]
            }
        }
        return String(decoding: out, as: UTF8.self)
    }

    func key(_ e: JSONEntry) -> String {
        decode(Int(e.keyStart), Int(e.keyLen), escaped: e.flags & JSONKind.keyEscapeFlag != 0)
    }

    func rawText(_ e: JSONEntry) -> String {
        String(decoding: UnsafeBufferPointer(start: host + Int(e.valStart), count: Int(e.valLen)), as: UTF8.self)
    }

    // MARK: - struct fields

    /// The fields of the objects spanned by `level` (one span per row), in the output order: the
    /// explicit schema's fields first, then the inferred ones in order of first appearance.
    func fields(level: JSONLevel, rows: Int, schema: [JSONField]?, path: String,
                rowToRecord: @escaping (Int) -> Int) throws -> (names: [String], columns: [AnyMetalArray]) {
        // The schema, with a name given twice kept once (its first type), as pyarrow does.
        var schemaFields: [JSONField] = []
        if let schema {
            var seen = Set<String>()
            for f in schema where seen.insert(f.name).inserted { schemaFields.append(f) }
        }

        // Field ids: positional match against the first object that has fields, dictionary for the rest.
        var names: [String] = []
        var fid: MetalArrowBuffer? = nil
        var mayRepeat = false
        if level.count > 0 {
            var ref = 0
            while ref < level.spans && level.offset(ref + 1) == level.offset(ref) { ref += 1 }
            let refFirst = level.offset(ref)
            var K0 = level.offset(ref + 1) - refFirst
            var refNames = (0..<K0).map { key(level.entry(refFirst + $0)) }
            if Set(refNames).count != refNames.count { K0 = 0; refNames = [] }
            let m = try JSONKernels.matchKeys(ctx, source, level: level, K0: K0, refFirst: refFirst)
            fid = m.fid
            if m.novelCount == 0 {
                names = refNames
            } else {
                mayRepeat = true
                let idx = try JSONKernels.alloc(ctx, (K0 + m.novelCount) * 4)
                let ip = idx.mutableTyped(Int32.self)
                for k in 0..<K0 { ip[k] = Int32(refFirst + k) }
                memcpy(ip + K0, m.novelBuffer.contents, m.novelCount * 4)
                let keys = try JSONKernels.keys(ctx, source, level: level, indices: idx, count: K0 + m.novelCount)
                let (codes, unique) = try keys.dictionaryEncode()
                names = unique.toArray().map { $0 ?? "" }
                try JSONKernels.assign(ctx, list: m.novelBuffer, count: m.novelCount, codes: codes.values,
                                       skip: K0, fid: m.fid)
            }
        }

        // Output plan: (name, field id or nil when absent from the data, explicit type or nil).
        var plan: [(name: String, fid: Int?, type: JSONType?)] = []
        var byName: [String: Int] = [:]
        for (i, nm) in names.enumerated() where byName[nm] == nil { byName[nm] = i }
        var inSchema = [Bool](repeating: false, count: names.count)
        for f in schemaFields {
            let id = byName[f.name]
            if let id { inSchema[id] = true }
            plan.append((f.name, id, f.type))
        }
        let explicit = schema != nil
        var unexpected: [Int] = []
        for (i, nm) in names.enumerated() where !inSchema[i] {
            if !explicit || behavior == .infer { plan.append((nm, i, nil)) } else { unexpected.append(i) }
        }
        if explicit && behavior == .error && !unexpected.isEmpty, let fid {
            var expected = [UInt8](repeating: 1, count: names.count)
            for u in unexpected { expected[u] = 0 }
            if let e = try JSONKernels.firstUnexpected(ctx, level: level, fid: fid, expected: expected) {
                parseErrors.append((Int(level.entry(e).keyStart), "unexpected field"))
            }
        }

        // Slot matrix in groups of fields; each group's columns are built together.
        var built = [AnyMetalArray?](repeating: nil, count: plan.count)
        let wanted = plan.enumerated().compactMap { i, p in p.fid.map { ($0, i) } }.sorted { $0.0 < $1.0 }
        if let fid, !wanted.isEmpty {
            let perGroup = Swift.max(1, JSONColumnBuilder.matrixBudgetBytes / 4 / Swift.max(rows, 1))
            var g = 0
            while g < wanted.count {
                let lo = wanted[g].0
                var hi = lo + 1
                var end = g + 1
                while end < wanted.count && wanted[end].0 - lo < perGroup { hi = wanted[end].0 + 1; end += 1 }
                var inGroup = [Bool](repeating: false, count: hi - lo)
                for k in g..<end { inGroup[wanted[k].0 - lo] = true }
                let (M, dup, groupKinds) = try JSONKernels.scatter(ctx, level: level, fid: fid, rows: rows, lo: lo, hi: hi,
                                                                   mayRepeat: mayRepeat, wanted: inGroup)
                if let dup {
                    let x = level.entry(dup)
                    let f = Int(fid.typed(Int32.self)[dup])
                    parseErrors.append((Int(x.keyStart),
                                        "Column(\(path)/\(names[f])) was specified twice in row \(rowToRecord(Int(x.parent)))"))
                }
                let members = Array(wanted[g..<end])
                let set = JSONKernels.ColumnSet(level: level, rowEntry: M, rows: rows, cols: hi - lo)
                let cols = try columns(set, sel: members.map { $0.0 - lo },
                                       types: members.map { plan[$0.1].type },
                                       paths: members.map { "\(path)/\(plan[$0.1].name)" }, rowToRecord: rowToRecord,
                                       kinds: groupKinds)
                for (k, m) in members.enumerated() { built[m.1] = cols[k] }
                g = end
            }
        }
        // Schema fields that never appear: all-null columns of their type, built through the ordinary
        // path over rows that have no entry.
        let absent = plan.indices.filter { built[$0] == nil }
        if !absent.isEmpty {
            let off = try JSONKernels.alloc(ctx, (rows + 1) * 4, zeroed: true)
            let empty = JSONLevel(entries: try JSONKernels.alloc(ctx, JSONEntry.stride), count: 0, offsets: off, spans: rows)
            let rowEntry = try JSONKernels.alloc(ctx, rows * absent.count * 4)
            memset(rowEntry.mutableContents, 0xFF, rows * absent.count * 4)
            let set = JSONKernels.ColumnSet(level: empty, rowEntry: rowEntry, rows: rows, cols: absent.count)
            let cols = try columns(set, sel: Array(0..<absent.count),
                                   types: absent.map { plan[$0].type == .null ? nil : plan[$0].type },
                                   paths: absent.map { "\(path)/\(plan[$0].name)" }, rowToRecord: rowToRecord)
            for (k, i) in absent.enumerated() { built[i] = cols[k] }
        }
        return (plan.map { $0.name }, built.map { $0! })
    }

    // MARK: - columns

    static let maskNull: UInt32 = 1 << 0
    static let maskBool: UInt32 = (1 << 1) | (1 << 2)
    static let maskNumber: UInt32 = (1 << 3) | (1 << 4)
    static let maskString: UInt32 = 1 << 5
    static let maskObject: UInt32 = 1 << 6
    static let maskArray: UInt32 = 1 << 7

    /// Builds the columns `sel` of `set` (types: explicit, or nil to infer). Columns that come out as
    /// the same scalar type are built together: one validity pass, one text gather and one parse for
    /// all of them, so a wide file costs a handful of dispatches per type rather than per column.
    func columns(_ set: JSONKernels.ColumnSet, sel: [Int], types: [JSONType?], paths: [String],
                 rowToRecord: @escaping (Int) -> Int,
                 kinds known: [(mask: UInt32, flags: UInt32)]? = nil) throws -> [AnyMetalArray] {
        let kinds = try known ?? JSONKernels.kinds(ctx, set)
        var out = [AnyMetalArray?](repeating: nil, count: sel.count)
        var groupOrder: [String] = []
        var groups: [String: (type: JSONType, inferred: Bool, members: [Int], flags: UInt32)] = [:]
        for k in 0..<sel.count {
            let (mask, flags) = kinds[sel[k]]
            var classes = Set<Int>()
            for b in 1...7 where mask & (1 << UInt32(b)) != 0 { classes.insert(JSONKind(rawValue: UInt32(b))!.classID) }
            let target: JSONType
            let inferred: Bool
            if let type = types[k] {
                if type == .null {
                    throw JSONError.unsupported("explicit_schema: \(paths[k]) has type null, which the JSON reader does not convert to")
                }
                if classes.contains(where: { $0 != type.jsonClassID }) {
                    reportConflict(set.column(sel[k]), expected: (type.jsonClass, type.jsonClassID), path: paths[k],
                                   rowToRecord: rowToRecord)
                    out[k] = .null(MetalNullArray(length: set.rows, context: ctx))
                    continue
                }
                target = type
                inferred = false
            } else {
                if classes.count > 1 {
                    reportConflict(set.column(sel[k]), expected: nil, path: paths[k], rowToRecord: rowToRecord)
                    out[k] = .null(MetalNullArray(length: set.rows, context: ctx))
                    continue
                }
                guard let only = classes.first else {
                    out[k] = .null(MetalNullArray(length: set.rows, context: ctx))
                    continue
                }
                switch only {
                case 1: target = .boolean
                case 2: target = mask & (1 << JSONKind.float.rawValue) != 0 ? .float64 : .int64
                case 3: target = .utf8
                case 4: target = .structure([])
                default: target = .list(.null)
                }
                inferred = true
            }
            switch target {
            case .structure, .list:
                out[k] = try nested(set.column(sel[k]), type: target, inferred: inferred, path: paths[k],
                                    rowToRecord: rowToRecord)
            default:
                let key = "\(target)|\(inferred)"
                if groups[key] == nil { groupOrder.append(key); groups[key] = (target, inferred, [], 0) }
                groups[key]!.members.append(k)
                groups[key]!.flags |= flags
            }
        }
        for key in groupOrder {
            let g = groups[key]!
            let packed = try JSONKernels.pack(ctx, set, g.members.map { sel[$0] })
            let cols = try scalars(packed, type: g.type, inferred: g.inferred, flags: g.flags)
            for (i, k) in g.members.enumerated() { out[k] = cols[i] }
        }
        return out.map { $0! }
    }

    /// Finds the first row whose class differs from the first non-null class (or from the expected
    /// class of an explicit type) and records pyarrow's message for it.
    func reportConflict(_ col: JSONKernels.ColumnSet, expected: (name: String, id: Int)?, path: String,
                        rowToRecord: (Int) -> Int) {
        var first = expected
        for r in 0..<col.rows {
            let e = col.entryIndex(0, r)
            guard e >= 0 else { continue }
            let x = col.level.entry(e)
            let k = x.kind
            if k == .null { continue }
            guard let f = first else { first = (k.className, k.classID); continue }
            if k.classID != f.id {
                parseErrors.append((Int(x.valStart),
                                    "Column(\(path)) changed from \(f.name) to \(k.className) in row \(rowToRecord(r))"))
                return
            }
        }
    }

    /// Word-aligned bitmap of column j, or nil when the column has no nulls.
    private func bitmap(_ b: MetalArrowBuffer, _ j: Int, wpc: Int, nulls: Int) -> MetalArrowBuffer? {
        nulls == 0 ? nil : b.view(byteOffset: j * wpc * 4, byteCount: Swift.max(wpc * 4, 4))
    }

    /// Columns of one scalar type, built together.
    func scalars(_ set: JSONKernels.ColumnSet, type: JSONType, inferred: Bool, flags: UInt32) throws -> [AnyMetalArray] {
        let rows = set.rows, cols = set.cols
        switch type {
        case .boolean:
            let v = try JSONKernels.validity(ctx, set, validMask: Self.maskBool, boolValues: true)
            return (0..<cols).map { j in
                .boolean(MetalBooleanArray(length: rows, nullCount: v.nulls[j], validity: bitmap(v.validity, j, wpc: v.wordsPerColumn, nulls: v.nulls[j]),
                                           values: v.values!.view(byteOffset: j * v.wordsPerColumn * 4, byteCount: Swift.max(v.wordsPerColumn * 4, 4)),
                                           context: ctx))
            }
        case .int8: return try numbers(set, Int8.self, type: type, flags: flags).map { .int8($0) }
        case .int16: return try numbers(set, Int16.self, type: type, flags: flags).map { .int16($0) }
        case .int32: return try numbers(set, Int32.self, type: type, flags: flags).map { .int32($0) }
        case .int64: return try numbers(set, Int64.self, type: type, flags: flags).map { .int64($0) }
        case .uint8: return try numbers(set, UInt8.self, type: type, flags: flags).map { .uint8($0) }
        case .uint16: return try numbers(set, UInt16.self, type: type, flags: flags).map { .uint16($0) }
        case .uint32: return try numbers(set, UInt32.self, type: type, flags: flags).map { .uint32($0) }
        case .uint64: return try numbers(set, UInt64.self, type: type, flags: flags).map { .uint64($0) }
        case .float32: return try numbers(set, Float.self, type: type, flags: flags).map { .float32($0) }
        case .float64: return try numbers(set, Double.self, type: type, flags: flags).map { .float64($0) }
        case .utf8, .timestamp:
            let v = try JSONKernels.validity(ctx, set, validMask: Self.maskString)
            let (off, data) = try JSONKernels.gather(ctx, source, set, mode: .stringValue, validMask: Self.maskString)
            let strings: [MetalStringArray] = (0..<cols).map { j in
                MetalStringArray(length: rows, nullCount: v.nulls[j], validity: bitmap(v.validity, j, wpc: v.wordsPerColumn, nulls: v.nulls[j]),
                                 offsets: off.view(byteOffset: j * rows * 4, byteCount: (rows + 1) * 4), data: data, context: ctx)
            }
            var unit = ArrowTemporalUnit.second
            var tz: String? = nil
            if case .timestamp(let u, let z) = type { unit = u; tz = z }
            if case .utf8 = type, !inferred { return strings.map { .string($0) } }
            if case .utf8 = type, !strings.contains(where: { $0.length - $0.nullCount > 0 }) { return strings.map { .string($0) } }
            // pyarrow infers timestamp[s] for a string column whose every value is an ISO-8601 timestamp.
            let t = try JSONKernels.timestamps(ctx, offsets: off, data: data, validity: v.validity, rows: rows, cols: cols, unit: unit)
            var result: [AnyMetalArray] = []
            for j in 0..<cols {
                let s = strings[j]
                if case .utf8 = type, t.fails[j].count > 0 || s.length - s.nullCount == 0 {
                    result.append(.string(s))
                    continue
                }
                if case .timestamp = type, t.fails[j].count > 0, let r = t.fails[j].first {
                    conversionErrors.append((Int(set.level.entry(set.entryIndex(j, r)).valStart),
                                             "Failed to convert JSON to \(type), couldn't parse:\(s[r] ?? "")"))
                }
                let arr = MetalArray<Int64>(length: rows, nullCount: s.nullCount, validity: s.validity,
                                            values: t.values.view(byteOffset: j * rows * 8, byteCount: Swift.max(rows * 8, 8)),
                                            context: ctx)
                result.append(.temporal(try MetalTemporalArray(type: .timestamp(unit, timezone: tz), arr)))
            }
            return result
        default:
            throw JSONError.unsupported("internal: \(type) is not a scalar type")
        }
    }

    /// Number columns: the number text of every column is gathered once and parsed once by
    /// `MetalStringArray.parse`; each column is a view of the result with its own validity.
    func numbers<T: ArrowPrimitive>(_ set: JSONKernels.ColumnSet, _: T.Type, type: JSONType, flags: UInt32) throws -> [MetalArray<T>] {
        let rows = set.rows, cols = set.cols
        let v = try JSONKernels.validity(ctx, set, validMask: Self.maskNumber)
        let (off, data) = try JSONKernels.gather(ctx, source, set, mode: .rawValue, validMask: Self.maskNumber)
        let text = MetalStringArray(length: rows * cols, nullCount: 0, validity: nil, offsets: off, data: data, context: ctx)
        let parsed = try text.parse(T.self)
        let vp = parsed.values.mutableTyped(T.self)
        if !T.isFloatingPoint, parsed.nullCount > v.nulls.reduce(0, +) {
            // Each column's first value that did not convert, with its position in the file; the reader
            // reports the earliest, in pyarrow's words.
            let pv = parsed.validity?.typed(UInt8.self)
            let vv = v.validity.typed(UInt8.self)
            for j in 0..<cols {
                for r in 0..<rows where Bitmap.isSet(vv + j * v.wordsPerColumn * 4, r) {
                    if let pv, !Bitmap.isSet(pv, j * rows + r) {
                        conversionErrors.append((Int(set.level.entry(set.entryIndex(j, r)).valStart),
                                                 "Failed to convert JSON to \(type), couldn't parse:\(text[j * rows + r] ?? "")"))
                        break
                    }
                }
            }
        }
        if T.isFloatingPoint, flags & JSONKind.specialFlag != 0 {
            // NaN, Inf and Infinity are set here rather than left to the text parser.
            for j in 0..<cols {
                for r in 0..<rows {
                    let e = set.entryIndex(j, r)
                    guard e >= 0 else { continue }
                    let x = set.level.entry(e)
                    guard x.flags & JSONKind.specialFlag != 0 else { continue }
                    var p = Int(x.valStart)
                    let negative = host[p] == 0x2D
                    if negative { p += 1 }
                    let d: Double = host[p] == 0x4E ? .nan : (negative ? -.infinity : .infinity)
                    vp[j * rows + r] = T.self == Float.self ? Float(d) as! T : d as! T
                }
            }
        }
        let width = MemoryLayout<T>.stride
        return (0..<cols).map { j in
            MetalArray<T>(length: rows, nullCount: v.nulls[j], validity: bitmap(v.validity, j, wpc: v.wordsPerColumn, nulls: v.nulls[j]),
                          values: parsed.values.view(byteOffset: j * rows * width, byteCount: Swift.max(rows * width, width)),
                          context: ctx)
        }
    }

    /// A struct or list column (one column: nested levels are walked per column).
    func nested(_ col: JSONKernels.ColumnSet, type: JSONType, inferred: Bool, path: String,
                rowToRecord: @escaping (Int) -> Int) throws -> AnyMetalArray {
        let rows = col.rows
        switch type {
        case .structure(let schemaFields):
            let v = try JSONKernels.validity(ctx, col, validMask: Self.maskObject)
            let (s, e) = try JSONKernels.spans(ctx, col, validMask: Self.maskObject)
            let level = try walk(s, e, rows)
            let (names, children) = try fields(level: level, rows: rows, schema: inferred ? nil : schemaFields,
                                               path: path, rowToRecord: rowToRecord)
            return .structure(try MetalStructArray(length: rows, nullCount: v.nulls[0],
                                                   validity: bitmap(v.validity, 0, wpc: v.wordsPerColumn, nulls: v.nulls[0]),
                                                   names: names, children: children, context: ctx))
        case .list(let elementType):
            let v = try JSONKernels.validity(ctx, col, validMask: Self.maskArray)
            let (s, e) = try JSONKernels.spans(ctx, col, validMask: Self.maskArray)
            let level = try walk(s, e, rows)
            let elements = JSONKernels.ColumnSet(level: level, rowEntry: nil, rows: level.count, cols: 1)
            let values = try columns(elements, sel: [0], types: [inferred || elementType == .null ? nil : elementType],
                                     paths: [path + "/[]"],
                                     rowToRecord: { r in rowToRecord(Int(level.entry(r).parent)) })[0]
            return .list(MetalListArray(length: rows, nullCount: v.nulls[0],
                                        validity: bitmap(v.validity, 0, wpc: v.wordsPerColumn, nulls: v.nulls[0]),
                                        offsets: level.offsets, values: values, context: ctx))
        default:
            throw JSONError.unsupported("internal: \(type) is not a nested type")
        }
    }

    /// One nested level: the children of the given spans. The top-level walk validated them; the only
    /// spans that stop early are the partial values of the record holding the first syntax error,
    /// whose error the reader already has.
    func walk(_ start: MetalArrowBuffer, _ end: MetalArrowBuffer, _ spans: Int) throws -> JSONLevel {
        let (counts, _) = try JSONKernels.walkCount(ctx, source, n: n, spanStart: start, spanEnd: end, spans: spans)
        return try JSONKernels.walkEmit(ctx, source, n: n, spanStart: start, spanEnd: end, spans: spans, counts: counts)
    }
}
