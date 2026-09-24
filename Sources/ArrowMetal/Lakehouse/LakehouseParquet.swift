import Foundation

// Reading the nested columns of a Delta checkpoint through the GPU Parquet reader.
//
// A checkpoint is an ordinary Parquet file whose top-level columns are structs (`add`, `remove`,
// `metaData`, `protocol`, ...). The reader already reads a struct's leaves by dotted path and a
// `list<primitive>` wherever it sits; a `map<string, string>` (`add.partitionValues`,
// `metaData.configuration`) is read here as two lists over the same repetition levels, one of keys and
// one of values. Everything comes back to the host afterwards, because the log replay that consumes it
// is a CPU walk over a few thousand rows.

/// The host-side values of one checkpoint column.
enum CheckpointColumn {
    case strings([String?])
    case ints([Int64?])
    case bools([Bool?])
    case stringLists([[String]?])
    case maps([[String: String?]?])
}

extension ParquetFile {
    /// Every schema element keyed by its dotted path (root excluded).
    func lakeSchemaElements() -> [String: ParquetSchemaElement] {
        var out: [String: ParquetSchemaElement] = [:]
        let els = metadata.schema
        var index = 1
        func walk(_ prefix: [String]) {
            guard index < els.count else { return }
            let e = els[index]
            index += 1
            let path = prefix + [e.name]
            out[path.joined(separator: ".")] = e
            for _ in 0..<Swift.max(e.numChildren, 0) { walk(path) }
        }
        let root = els.first?.numChildren ?? 0
        for _ in 0..<Swift.max(root, 0) { walk([]) }
        return out
    }

    /// The field at a dotted path, walking struct groups by member name.
    func lakeField(at path: [String]) -> ParquetField? {
        guard let first = path.first, var node = fields.first(where: { $0.name == first }) else { return nil }
        for name in path.dropFirst() {
            guard case .group(let children) = node.kind, let c = children.first(where: { $0.name == name }) else { return nil }
            node = c
        }
        return node
    }

    /// Reads the named checkpoint columns (dotted paths). A path the file does not have is left out of
    /// the result, since writers differ in which optional columns they include.
    func lakeReadCheckpointColumns(_ paths: [String]) throws -> [String: CheckpointColumn] {
        let elements = lakeSchemaElements()
        enum Plan { case leaf(ParquetField), list(ParquetField), map(key: ParquetField, value: ParquetField), nativeMap(ParquetField) }
        var plans: [(String, Plan)] = []
        for p in paths {
            let parts = p.split(separator: ".").map(String.init)
            guard let f = lakeField(at: parts) else { continue }
            switch f.kind {
            case .leaf:
                plans.append((p, .leaf(ParquetField(name: p, kind: f.kind, nullable: true))))
            case .list where f.isMap:
                // The Parquet reader models a MAP as a list of key/value entries and reassembles it
                // itself; read the field as the file describes it.
                plans.append((p, .nativeMap(f)))
            case .list:
                plans.append((p, .list(ParquetField(name: p, kind: f.kind, nullable: true))))
            case .group(let children):
                // MAP: group { repeated group key_value { key, value } }
                guard children.count == 1, case .group(let kv) = children[0].kind, kv.count == 2,
                      case .leaf(let keyLeaf) = kv[0].kind, case .leaf(let valueLeaf) = kv[1].kind else {
                    throw LakehouseError.malformed("checkpoint column \(p) is a struct, not a map")
                }
                // The definition level at which a key/value entry exists: one per optional or repeated
                // node from the root down to (and including) the repeated key_value group.
                var dRep = 0
                var prefix: [String] = []
                for name in parts + [children[0].name] {
                    prefix.append(name)
                    if let e = elements[prefix.joined(separator: ".")], e.repetition != .required { dRep += 1 }
                }
                let key = ParquetField(name: p + ".key",
                                       kind: .list(element: ParquetField(name: "key", kind: .leaf(keyLeaf), nullable: false),
                                                   repeatedDefinition: dRep), nullable: true)
                let value = ParquetField(name: p + ".value",
                                         kind: .list(element: ParquetField(name: "value", kind: .leaf(valueLeaf), nullable: true),
                                                     repeatedDefinition: dRep), nullable: true)
                plans.append((p, .map(key: key, value: value)))
            }
        }
        let groups = Array(0..<metadata.rowGroups.count)
        let opts = ParquetReadOptions(columns: nil, rowGroups: nil, dictionaryEncoded: false, filters: [])
        var arrays: [String: [AnyMetalArray]] = [:]
        try context.batch {
            for (p, plan) in plans {
                switch plan {
                case .leaf(let f), .list(let f), .nativeMap(let f):
                    arrays[p] = [try readField(f, rowGroups: groups, options: opts)]
                case .map(let k, let v):
                    arrays[p] = [try readField(k, rowGroups: groups, options: opts),
                                 try readField(v, rowGroups: groups, options: opts)]
                }
            }
        }
        try context.flush()
        var out: [String: CheckpointColumn] = [:]
        for (p, plan) in plans {
            guard let a = arrays[p] else { continue }
            switch plan {
            case .leaf:
                out[p] = try Self.hostColumn(a[0], path: p)
            case .list:
                out[p] = .stringLists(try Self.hostStringLists(a[0], path: p))
            case .map:
                out[p] = .maps(try Self.hostMaps(a[0], a[1], path: p))
            case .nativeMap:
                out[p] = .maps(try Self.hostNativeMaps(a[0], path: p))
            }
        }
        return out
    }

    private static func hostColumn(_ a: AnyMetalArray, path: String) throws -> CheckpointColumn {
        switch try a.decodedIfDictionary() {
        case .string(let s), .binary(let s): return .strings(s.toArray())
        case .int64(let x): return .ints(x.toArray().map { $0 })
        case .int32(let x): return .ints(x.toArray().map { $0.map(Int64.init) })
        case .int16(let x): return .ints(x.toArray().map { $0.map(Int64.init) })
        case .int8(let x): return .ints(x.toArray().map { $0.map(Int64.init) })
        case .boolean(let b): return .bools(b.toArray())
        case .temporal(let t): return .ints(t.toArray())
        default: throw LakehouseError.malformed("checkpoint column \(path) has unexpected type \(a.arrowFormat)")
        }
    }

    /// A `list<string>` column as Swift arrays (nil for a null row).
    private static func hostStringLists(_ a: AnyMetalArray, path: String) throws -> [[String]?] {
        guard case .list(let l) = a else {
            throw LakehouseError.malformed("checkpoint column \(path) is \(a.arrowFormat), expected a list")
        }
        let child: [String?]
        switch try l.values.decodedIfDictionary() {
        case .string(let s), .binary(let s): child = s.toArray()
        default: throw LakehouseError.malformed("checkpoint column \(path) holds \(l.values.arrowFormat), expected strings")
        }
        let off = l.offsets.typed(Int32.self)
        let valid = l.validity.map { v in (0..<l.length).map { Bitmap.isSet(v.typed(UInt8.self), $0) } }
        var out: [[String]?] = []
        out.reserveCapacity(l.length)
        for i in 0..<l.length {
            if let valid, !valid[i] { out.append(nil); continue }
            out.append((Int(off[i])..<Int(off[i + 1])).map { child[$0] ?? "" })
        }
        return out
    }

    /// A map column from its key and value lists, keeping null values as nil.
    /// A map column as the Parquet reader reassembles it: a list of `{key, value}` entries.
    private static func hostNativeMaps(_ a: AnyMetalArray, path: String) throws -> [[String: String?]?] {
        let entries: MetalListArray
        switch a {
        case .map(let m): entries = m.entries
        case .list(let l): entries = l
        default: throw LakehouseError.malformed("checkpoint map \(path) read as \(a.arrowFormat), not a map")
        }
        guard case .structure(let kv) = entries.values, kv.children.count == 2 else {
            throw LakehouseError.malformed("checkpoint map \(path) entries are not key/value pairs")
        }
        func strings(_ c: AnyMetalArray) throws -> [String?] {
            switch try c.decodedIfDictionary() {
            case .string(let s), .binary(let s): return s.toArray()
            default: throw LakehouseError.malformed("checkpoint map \(path) holds \(c.arrowFormat), expected strings")
            }
        }
        let k = try strings(kv.children[0]), v = try strings(kv.children[1])
        let o = entries.offsets.typed(Int32.self)
        let valid = entries.validity.map { b in (0..<entries.length).map { Bitmap.isSet(b.typed(UInt8.self), $0) } }
        var out: [[String: String?]?] = []
        out.reserveCapacity(entries.length)
        for i in 0..<entries.length {
            if let valid, !valid[i] { out.append(nil); continue }
            var m: [String: String?] = [:]
            for j in Int(o[i])..<Int(o[i + 1]) { m[k[j] ?? ""] = j < v.count ? v[j] : nil }
            out.append(m)
        }
        return out
    }

    private static func hostMaps(_ keys: AnyMetalArray, _ values: AnyMetalArray, path: String) throws -> [[String: String?]?] {
        guard case .list(let kl) = keys, case .list(let vl) = values else {
            throw LakehouseError.malformed("checkpoint map \(path) did not read as two lists")
        }
        func strings(_ a: AnyMetalArray) throws -> [String?] {
            switch try a.decodedIfDictionary() {
            case .string(let s), .binary(let s): return s.toArray()
            default: throw LakehouseError.malformed("checkpoint map \(path) holds \(a.arrowFormat), expected strings")
            }
        }
        let k = try strings(kl.values), v = try strings(vl.values)
        let ko = kl.offsets.typed(Int32.self), vo = vl.offsets.typed(Int32.self)
        let valid = kl.validity.map { b in (0..<kl.length).map { Bitmap.isSet(b.typed(UInt8.self), $0) } }
        var out: [[String: String?]?] = []
        out.reserveCapacity(kl.length)
        for i in 0..<kl.length {
            if let valid, !valid[i] { out.append(nil); continue }
            var m: [String: String?] = [:]
            let n = Int(ko[i + 1]) - Int(ko[i])
            for j in 0..<n {
                let key = k[Int(ko[i]) + j] ?? ""
                let vi = Int(vo[i]) + j
                m[key] = vi < v.count ? v[vi] : nil
            }
            out.append(m)
        }
        return out
    }
}
