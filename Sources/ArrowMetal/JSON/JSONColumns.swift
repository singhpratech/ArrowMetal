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
    var conversionErrors: [String] = []

    /// Slot-matrix budget per group of fields; wider tables are processed in groups.
    static let matrixBudgetBytes = 512 << 20

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
            jprof("pre-match"); let m = try JSONKernels.matchKeys(ctx, source, level: level, K0: K0, refFirst: refFirst)
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

        // Slot matrix in groups of fields, then one column per planned field.
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
                jprof("keys"); let (M, dup) = try JSONKernels.scatter(ctx, level: level, fid: fid, rows: rows, lo: lo, hi: hi,
                                                       mayRepeat: mayRepeat)
                if let dup {
                    let x = level.entry(dup)
                    let f = Int(fid.typed(Int32.self)[dup])
                    parseErrors.append((Int(x.keyStart),
                                        "Column(\(path)/\(names[f])) was specified twice in row \(rowToRecord(Int(x.parent)))"))
                }
                for k in g..<end {
                    let (f, planIndex) = wanted[k]
                    let view = MetalArrowBuffer(mtl: M.mtl, byteCount: Swift.max(rows * 4, 4),
                                                offset: M.offset + (f - lo) * rows * 4, keepAlive: M)
                    let col = JSONKernels.Column(level: level, rowEntry: view, rows: rows); defer { jprof("column \(plan[planIndex].name)") }
                    jprof("before col"); built[planIndex] = try column(col, type: plan[planIndex].type, path: "\(path)/\(plan[planIndex].name)",
                                                  rowToRecord: rowToRecord)
                }
                g = end
            }
        }
        // Schema fields that never appear: all-null columns of their type.
        var columns: [AnyMetalArray] = []
        for (i, p) in plan.enumerated() {
            if let c = built[i] { columns.append(c); continue }
            columns.append(try absentColumn(type: p.type ?? .null, rows: rows, path: "\(path)/\(p.name)"))
        }
        return (plan.map { $0.name }, columns)
    }

    /// An all-null column of `type`, built through the ordinary path over an empty level.
    func absentColumn(type: JSONType, rows: Int, path: String) throws -> AnyMetalArray {
        let off = try JSONKernels.alloc(ctx, (rows + 1) * 4, zeroed: true)
        let empty = JSONLevel(entries: try JSONKernels.alloc(ctx, JSONEntry.stride), count: 0, offsets: off, spans: rows)
        let rowEntry = try JSONKernels.alloc(ctx, rows * 4)
        memset(rowEntry.mutableContents, 0xFF, rows * 4)
        let col = JSONKernels.Column(level: empty, rowEntry: rowEntry, rows: rows)
        return try column(col, type: type == .null ? nil : type, path: path, rowToRecord: { $0 })
    }

    // MARK: - one column

    static let maskNull: UInt32 = 1 << 0
    static let maskBool: UInt32 = (1 << 1) | (1 << 2)
    static let maskNumber: UInt32 = (1 << 3) | (1 << 4)
    static let maskString: UInt32 = 1 << 5
    static let maskObject: UInt32 = 1 << 6
    static let maskArray: UInt32 = 1 << 7

    /// Builds one column. `type` is the explicit type, or nil to infer one.
    func column(_ col: JSONKernels.Column, type: JSONType?, path: String,
                rowToRecord: @escaping (Int) -> Int) throws -> AnyMetalArray {
        let (mask, flags) = try JSONKernels.kinds(ctx, col)
        var classes = Set<Int>()
        for k in 1...7 where mask & (1 << UInt32(k)) != 0 { classes.insert(JSONKind(rawValue: UInt32(k))!.classID) }

        if let type {
            if type == .null {
                throw JSONError.unsupported("explicit_schema: \(path) has type null, which the JSON reader does not convert to")
            }
            let want = type.jsonClassID
            if classes.contains(where: { $0 != want }) {
                reportConflict(col, expected: (type.jsonClass, want), path: path, rowToRecord: rowToRecord)
                return .null(MetalNullArray(length: col.rows, context: ctx))
            }
            return try build(col, type: type, flags: flags, inferred: false, path: path, rowToRecord: rowToRecord)
        }
        if classes.count > 1 {
            reportConflict(col, expected: nil, path: path, rowToRecord: rowToRecord)
            return .null(MetalNullArray(length: col.rows, context: ctx))
        }
        guard let only = classes.first else { return .null(MetalNullArray(length: col.rows, context: ctx)) }
        let inferred: JSONType
        switch only {
        case 1: inferred = .boolean
        case 2: inferred = mask & (1 << JSONKind.float.rawValue) != 0 ? .float64 : .int64
        case 3: inferred = .utf8
        case 4: inferred = .structure([])
        default: inferred = .list(.null)
        }
        return try build(col, type: inferred, flags: flags, inferred: true, path: path, rowToRecord: rowToRecord)
    }

    /// Finds the first row whose class differs from the first non-null class (or from the expected
    /// class of an explicit type) and records pyarrow's message for it.
    func reportConflict(_ col: JSONKernels.Column, expected: (name: String, id: Int)?, path: String,
                        rowToRecord: (Int) -> Int) {
        var first = expected
        for r in 0..<col.rows {
            let e = col.entryIndex(r)
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

    private func bitmapOrNil(_ b: MetalArrowBuffer, nulls: Int) -> MetalArrowBuffer? { nulls == 0 ? nil : b }

    func build(_ col: JSONKernels.Column, type: JSONType, flags: UInt32, inferred: Bool, path: String,
               rowToRecord: @escaping (Int) -> Int) throws -> AnyMetalArray {
        let rows = col.rows
        switch type {
        case .null:
            return .null(MetalNullArray(length: rows, context: ctx))

        case .boolean:
            let (valid, values, nulls) = try JSONKernels.validity(ctx, col, validMask: Self.maskBool, boolValues: true)
            return .boolean(MetalBooleanArray(length: rows, nullCount: nulls, validity: bitmapOrNil(valid, nulls: nulls),
                                              values: values!, context: ctx))

        case .int8: return .int8(try integers(col, Int8.self, type: type))
        case .int16: return .int16(try integers(col, Int16.self, type: type))
        case .int32: return .int32(try integers(col, Int32.self, type: type))
        case .int64: return .int64(try integers(col, Int64.self, type: type))
        case .uint8: return .uint8(try integers(col, UInt8.self, type: type))
        case .uint16: return .uint16(try integers(col, UInt16.self, type: type))
        case .uint32: return .uint32(try integers(col, UInt32.self, type: type))
        case .uint64: return .uint64(try integers(col, UInt64.self, type: type))
        case .float32: return .float32(try floats(col, Float.self, flags: flags))
        case .float64: return .float64(try floats(col, Double.self, flags: flags))

        case .utf8:
            let s = try strings(col)
            if inferred, s.length - s.nullCount > 0 {
                // pyarrow infers timestamp[s] when every string is an ISO-8601 timestamp.
                let t = try JSONKernels.timestamps(ctx, s, unit: .second)
                if t.fails == 0 {
                    let arr = MetalArray<Int64>(length: rows, nullCount: s.nullCount,
                                                validity: bitmapOrNil(t.validity, nulls: s.nullCount),
                                                values: t.values, context: ctx)
                    return .temporal(try MetalTemporalArray(type: .timestamp(.second, timezone: nil), arr))
                }
            }
            return .string(s)

        case .timestamp(let unit, let tz):
            let s = try strings(col)
            let t = try JSONKernels.timestamps(ctx, s, unit: unit)
            if t.fails > 0, let r = t.firstFail {
                conversionErrors.append("Failed to convert JSON to \(type), couldn't parse:\(s[r] ?? "")")
            }
            let arr = MetalArray<Int64>(length: rows, nullCount: s.nullCount,
                                        validity: bitmapOrNil(t.validity, nulls: s.nullCount), values: t.values, context: ctx)
            return .temporal(try MetalTemporalArray(type: .timestamp(unit, timezone: tz), arr))

        case .structure(let schemaFields):
            let (valid, _, nulls) = try JSONKernels.validity(ctx, col, validMask: Self.maskObject)
            let (s, e) = try JSONKernels.spans(ctx, col, validMask: Self.maskObject)
            let level = try walk(s, e, rows)
            let (names, children) = try fields(level: level, rows: rows, schema: inferred ? nil : schemaFields,
                                               path: path, rowToRecord: rowToRecord)
            return .structure(try MetalStructArray(length: rows, nullCount: nulls, validity: bitmapOrNil(valid, nulls: nulls),
                                                   names: names, children: children, context: ctx))

        case .list(let elementType):
            let (valid, _, nulls) = try JSONKernels.validity(ctx, col, validMask: Self.maskArray)
            let (s, e) = try JSONKernels.spans(ctx, col, validMask: Self.maskArray)
            let level = try walk(s, e, rows)
            let child = JSONKernels.Column(level: level, rowEntry: nil, rows: level.count)
            let values = try column(child, type: inferred ? nil : (elementType == .null ? nil : elementType),
                                    path: path + "/[]",
                                    rowToRecord: { r in rowToRecord(Int(level.entry(r).parent)) })
            return .list(MetalListArray(length: rows, nullCount: nulls, validity: bitmapOrNil(valid, nulls: nulls),
                                        offsets: level.offsets, values: values, context: ctx))
        }
    }

    /// One nested level: the children of the given spans. The top-level walk validated them; the only
    /// spans that stop early are the partial values of the record holding the first syntax error,
    /// whose error the reader already has.
    func walk(_ start: MetalArrowBuffer, _ end: MetalArrowBuffer, _ spans: Int) throws -> JSONLevel {
        let (counts, _) = try JSONKernels.walkCount(ctx, source, n: n, spanStart: start,
                                                    spanEnd: end, spans: spans)
        return try JSONKernels.walkEmit(ctx, source, n: n, spanStart: start, spanEnd: end,
                                        spans: spans, counts: counts)
    }

    /// The unescaped string values of a column.
    func strings(_ col: JSONKernels.Column) throws -> MetalStringArray {
        let (valid, _, nulls) = try JSONKernels.validity(ctx, col, validMask: Self.maskString)
        let (off, data) = try JSONKernels.gather(ctx, source, col, mode: .stringValue, validMask: Self.maskString)
        return MetalStringArray(length: col.rows, nullCount: nulls, validity: bitmapOrNil(valid, nulls: nulls),
                                offsets: off, data: data, context: ctx)
    }

    /// The number text of a column, as a utf8 array for the string-to-number parse.
    func numberText(_ col: JSONKernels.Column) throws -> MetalStringArray {
        let (valid, _, nulls) = try JSONKernels.validity(ctx, col, validMask: Self.maskNumber)
        let (off, data) = try JSONKernels.gather(ctx, source, col, mode: .rawValue, validMask: Self.maskNumber)
        return MetalStringArray(length: col.rows, nullCount: nulls, validity: bitmapOrNil(valid, nulls: nulls),
                                offsets: off, data: data, context: ctx)
    }

    func integers<T: ArrowPrimitive>(_ col: JSONKernels.Column, _: T.Type, type: JSONType) throws -> MetalArray<T> {
        let text = try numberText(col)
        let out = try text.parse(T.self)
        if out.nullCount > text.nullCount {
            // The first value that did not convert, for pyarrow's message.
            let tv = text.validity?.typed(UInt8.self)
            let ov = out.validity?.typed(UInt8.self)
            for r in 0..<col.rows {
                let inValid = tv.map { Bitmap.isSet($0, r) } ?? true
                let outValid = ov.map { Bitmap.isSet($0, r) } ?? true
                if inValid && !outValid {
                    conversionErrors.append("Failed to convert JSON to \(type), couldn't parse:\(text[r] ?? "")")
                    break
                }
            }
        }
        return out
    }

    func floats<T: ArrowPrimitive>(_ col: JSONKernels.Column, _: T.Type, flags: UInt32) throws -> MetalArray<T> {
        let text = try numberText(col)
        let out = try text.parse(T.self)
        if flags & JSONKind.specialFlag != 0 {
            // NaN, Inf and Infinity are set here rather than trusted to the text parser.
            let vals = out.values.mutableTyped(T.self)
            let bits = out.validity?.mutableTyped(UInt8.self)
            for r in 0..<col.rows {
                let e = col.entryIndex(r)
                guard e >= 0 else { continue }
                let x = col.level.entry(e)
                guard x.flags & JSONKind.specialFlag != 0 else { continue }
                var p = Int(x.valStart)
                let negative = host[p] == 0x2D
                if negative { p += 1 }
                let v: Double = host[p] == 0x4E ? .nan : (negative ? -.infinity : .infinity)
                if T.self == Float.self { vals[r] = Float(v) as! T } else { vals[r] = v as! T }
                if let bits { Bitmap.set(bits, r) }
            }
            out.recomputeNullCount()
        }
        return out
    }
}
