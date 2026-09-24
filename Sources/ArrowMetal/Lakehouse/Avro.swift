import Foundation
import Compression

// A small CPU reader for Avro object container files, written for Iceberg's manifest lists and
// manifests. Those files are metadata -- a few kilobytes to a few megabytes describing which Parquet
// files hold the data -- so this reader is plain Swift on the host; the data files themselves go through
// the GPU Parquet reader.
//
// Supported: every Avro type (null, boolean, int, long, float, double, bytes, string, record, enum,
// array, map, union, fixed), named-type references, and the `null`, `deflate` and `snappy` codecs.
// Values are decoded against the writer's schema from the file header; logical types come back as their
// underlying Avro type (a `date` is its int, a `decimal` its bytes).

/// An Avro value decoded against the writer schema.
public indirect enum AvroValue: Sendable, Equatable {
    case null
    case boolean(Bool)
    /// Avro `int` and `long` both come back as a 64-bit integer.
    case long(Int64)
    case float(Float)
    case double(Double)
    case bytes([UInt8])
    case string(String)
    case record(AvroRecord)
    case enumSymbol(String)
    case array([AvroValue])
    case map([String: AvroValue])
    case fixed([UInt8])

    public var isNull: Bool { if case .null = self { return true } else { return false } }
    public var int64: Int64? { if case .long(let v) = self { return v } else { return nil } }
    public var bool: Bool? { if case .boolean(let v) = self { return v } else { return nil } }
    public var string: String? {
        switch self {
        case .string(let s): return s
        case .enumSymbol(let s): return s
        default: return nil
        }
    }
    public var bytes: [UInt8]? {
        switch self {
        case .bytes(let b), .fixed(let b): return b
        default: return nil
        }
    }
    public var record: AvroRecord? { if case .record(let r) = self { return r } else { return nil } }
    public var array: [AvroValue]? { if case .array(let a) = self { return a } else { return nil } }

    /// A field of a record value (nil for a missing field or a non-record).
    public subscript(field: String) -> AvroValue? { record?[field] }
}

/// A record value: field names in schema order and their values.
public struct AvroRecord: Sendable, Equatable {
    public let names: [String]
    public let values: [AvroValue]

    public subscript(field: String) -> AvroValue? {
        guard let i = names.firstIndex(of: field) else { return nil }
        return values[i]
    }
}

public enum AvroError: Error, CustomStringConvertible {
    case malformed(String)
    case unsupported(String)

    public var description: String {
        switch self {
        case .malformed(let s): return "Malformed Avro data: \(s)"
        case .unsupported(let s): return "Unsupported Avro feature: \(s)"
        }
    }
}

/// A parsed Avro schema.
public indirect enum AvroSchema: Sendable {
    case null, boolean, int, long, float, double, bytes, string
    case record(name: String, fields: [(name: String, schema: AvroSchema)])
    case enumeration(name: String, symbols: [String])
    case array(AvroSchema)
    case map(AvroSchema)
    case union([AvroSchema])
    case fixed(name: String, size: Int)
    /// A reference to a named type, resolved at decode time (named types may be recursive).
    case named(String)

    /// Parses the JSON form of a schema.
    public static func parse(json: String) throws -> AvroSchema {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) else {
            throw AvroError.malformed("the schema is not valid JSON")
        }
        var names: [String: AvroSchema] = [:]
        return try parse(obj, namespace: nil, names: &names)
    }

    /// Parses a schema and also returns the named types it declares, for resolving references.
    static func parseWithNames(json: String) throws -> (AvroSchema, [String: AvroSchema]) {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) else {
            throw AvroError.malformed("the schema is not valid JSON")
        }
        var names: [String: AvroSchema] = [:]
        let s = try parse(obj, namespace: nil, names: &names)
        return (s, names)
    }

    private static func fullName(_ name: String, _ ns: String?) -> String {
        if name.contains(".") { return name }
        if let ns, !ns.isEmpty { return ns + "." + name }
        return name
    }

    private static func register(_ name: String, _ ns: String?, _ s: AvroSchema, _ names: inout [String: AvroSchema]) {
        names[fullName(name, ns)] = s
        names[name.split(separator: ".").last.map(String.init) ?? name] = s
    }

    private static func parse(_ obj: Any, namespace: String?, names: inout [String: AvroSchema]) throws -> AvroSchema {
        if let s = obj as? String {
            switch s {
            case "null": return .null
            case "boolean": return .boolean
            case "int": return .int
            case "long": return .long
            case "float": return .float
            case "double": return .double
            case "bytes": return .bytes
            case "string": return .string
            default: return .named(fullName(s, namespace))
            }
        }
        if let branches = obj as? [Any] {
            return .union(try branches.map { try parse($0, namespace: namespace, names: &names) })
        }
        guard let d = obj as? [String: Any] else { throw AvroError.malformed("schema node \(obj)") }
        guard let type = d["type"] else { throw AvroError.malformed("schema object without a type") }
        if let t = type as? String {
            switch t {
            case "record", "error":
                guard let name = d["name"] as? String else { throw AvroError.malformed("record without a name") }
                let ns = (d["namespace"] as? String) ?? (name.contains(".") ? String(name[..<name.lastIndex(of: ".")!]) : namespace)
                // Register a reference first so a recursive field can name the record.
                register(name, ns, .named(fullName(name, ns)), &names)
                guard let fs = d["fields"] as? [[String: Any]] else { throw AvroError.malformed("record \(name) without fields") }
                var fields: [(name: String, schema: AvroSchema)] = []
                for f in fs {
                    guard let fname = f["name"] as? String, let ft = f["type"] else {
                        throw AvroError.malformed("record \(name) has a field without a name or type")
                    }
                    fields.append((fname, try parse(ft, namespace: ns, names: &names)))
                }
                let rec = AvroSchema.record(name: fullName(name, ns), fields: fields)
                register(name, ns, rec, &names)
                return rec
            case "enum":
                guard let name = d["name"] as? String, let symbols = d["symbols"] as? [String] else {
                    throw AvroError.malformed("enum without a name or symbols")
                }
                let ns = (d["namespace"] as? String) ?? namespace
                let e = AvroSchema.enumeration(name: fullName(name, ns), symbols: symbols)
                register(name, ns, e, &names)
                return e
            case "array":
                guard let items = d["items"] else { throw AvroError.malformed("array without items") }
                return .array(try parse(items, namespace: namespace, names: &names))
            case "map":
                guard let values = d["values"] else { throw AvroError.malformed("map without values") }
                return .map(try parse(values, namespace: namespace, names: &names))
            case "fixed":
                guard let name = d["name"] as? String, let size = d["size"] as? Int, size >= 0 else {
                    throw AvroError.malformed("fixed without a name or size")
                }
                let ns = (d["namespace"] as? String) ?? namespace
                let f = AvroSchema.fixed(name: fullName(name, ns), size: size)
                register(name, ns, f, &names)
                return f
            default:
                // A primitive with attributes, e.g. {"type": "int", "logicalType": "date"}.
                return try parse(t, namespace: namespace, names: &names)
            }
        }
        // {"type": {...}} -- a nested schema object.
        return try parse(type, namespace: namespace, names: &names)
    }
}

/// A binary Avro decoder over a byte slice.
struct AvroDecoder {
    let bytes: [UInt8]
    var pos: Int
    let names: [String: AvroSchema]
    /// Current nesting of `decode` calls. A schema whose records nest deeper than `maxDepth` (in
    /// practice a record that contains itself without an array, map or union in between, which no
    /// finite value satisfies) is refused instead of recursing until the stack runs out.
    private var depth = 0
    static let maxDepth = 256

    init(_ bytes: [UInt8], names: [String: AvroSchema], at pos: Int = 0) {
        self.bytes = bytes
        self.pos = pos
        self.names = names
    }

    var atEnd: Bool { pos >= bytes.count }

    mutating func byte() throws -> UInt8 {
        guard pos < bytes.count else { throw AvroError.malformed("unexpected end of data at byte \(pos)") }
        defer { pos += 1 }
        return bytes[pos]
    }

    /// Zig-zag varint (Avro `int` and `long`).
    mutating func long() throws -> Int64 {
        var result: UInt64 = 0
        var shift: UInt64 = 0
        while true {
            let b = try byte()
            guard shift < 64 else { throw AvroError.malformed("varint longer than 10 bytes") }
            result |= UInt64(b & 0x7F) << shift
            if b & 0x80 == 0 { break }
            shift += 7
        }
        return Int64(bitPattern: (result >> 1) ^ (0 &- (result & 1)))
    }

    mutating func take(_ n: Int) throws -> [UInt8] {
        // `n <= count - pos` rather than `pos + n <= count`: a length near Int.max must not overflow.
        guard n >= 0, n <= bytes.count - pos else {
            throw AvroError.malformed("length \(n) at byte \(pos) runs past the end (\(bytes.count) bytes)")
        }
        defer { pos += n }
        return Array(bytes[pos..<(pos + n)])
    }

    mutating func lengthPrefixed() throws -> [UInt8] {
        let n = try long()
        guard n >= 0, n <= Int64(Int32.max) else { throw AvroError.malformed("negative or huge length \(n)") }
        return try take(Int(n))
    }

    mutating func decode(_ s: AvroSchema) throws -> AvroValue {
        depth += 1
        defer { depth -= 1 }
        guard depth <= Self.maxDepth else {
            throw AvroError.malformed("values nest deeper than \(Self.maxDepth) levels at byte \(pos) "
                                      + "(a record that contains itself with nothing optional in between?)")
        }
        return try decodeValue(s)
    }

    private mutating func decodeValue(_ s: AvroSchema) throws -> AvroValue {
        switch s {
        case .null: return .null
        case .boolean: return .boolean(try byte() != 0)
        case .int, .long: return .long(try long())
        case .float:
            let b = try take(4)
            var bits: UInt32 = 0
            for i in 0..<4 { bits |= UInt32(b[i]) << (8 * UInt32(i)) }
            return .float(Float(bitPattern: bits))
        case .double:
            let b = try take(8)
            var bits: UInt64 = 0
            for i in 0..<8 { bits |= UInt64(b[i]) << (8 * UInt64(i)) }
            return .double(Double(bitPattern: bits))
        case .bytes: return .bytes(try lengthPrefixed())
        case .string: return .string(String(decoding: try lengthPrefixed(), as: UTF8.self))
        case .record(_, let fields):
            var values: [AvroValue] = []
            values.reserveCapacity(fields.count)
            for f in fields { values.append(try decode(f.schema)) }
            return .record(AvroRecord(names: fields.map { $0.name }, values: values))
        case .enumeration(let name, let symbols):
            let i = try long()
            guard i >= 0, Int(i) < symbols.count else { throw AvroError.malformed("enum \(name) index \(i)") }
            return .enumSymbol(symbols[Int(i)])
        case .array(let items):
            var out: [AvroValue] = []
            while true {
                var count = try long()
                if count == 0 { break }
                if count < 0 {                                   // negated count, then the block byte size
                    guard count != .min else { throw AvroError.malformed("array block count \(count)") }
                    count = -count
                    _ = try long()
                }
                guard count <= Int64(bytes.count - pos) + 1 else { throw AvroError.malformed("array block of \(count) items") }
                for _ in 0..<count { out.append(try decode(items)) }
            }
            return .array(out)
        case .map(let values):
            var out: [String: AvroValue] = [:]
            while true {
                var count = try long()
                if count == 0 { break }
                if count < 0 {
                    guard count != .min else { throw AvroError.malformed("map block count \(count)") }
                    count = -count
                    _ = try long()
                }
                guard count <= Int64(bytes.count - pos) + 1 else { throw AvroError.malformed("map block of \(count) entries") }
                for _ in 0..<count {
                    let k = String(decoding: try lengthPrefixed(), as: UTF8.self)
                    out[k] = try decode(values)
                }
            }
            return .map(out)
        case .union(let branches):
            let i = try long()
            guard i >= 0, Int(i) < branches.count else { throw AvroError.malformed("union branch \(i) of \(branches.count)") }
            return try decode(branches[Int(i)])
        case .fixed(_, let size): return .fixed(try take(size))
        case .named(let n):
            guard let target = names[n] ?? names[n.split(separator: ".").last.map(String.init) ?? n] else {
                throw AvroError.malformed("reference to undeclared type \(n)")
            }
            if case .named(let again) = target, again == n {
                throw AvroError.malformed("type \(n) is referenced before it is defined")
            }
            return try decode(target)
        }
    }
}

/// An Avro object container file, fully decoded.
public struct AvroFile: Sendable {
    /// The most records a block may declare beyond its byte count (records that encode to no bytes).
    static let maxZeroWidthRecords: Int64 = 1 << 20

    /// Header metadata (`avro.schema`, `avro.codec`, and anything the writer added, e.g. Iceberg's
    /// `format-version`, `partition-spec`, `schema`).
    public let metadata: [String: String]
    public let schema: AvroSchema
    public let codec: String
    /// Every record in the file, in order.
    public let records: [AvroValue]

    public init(path: String) throws {
        guard let d = FileManager.default.contents(atPath: path) else {
            throw AvroError.malformed("cannot read \(path)")
        }
        try self.init(bytes: [UInt8](d), source: path)
    }

    public init(bytes: [UInt8], source: String = "<bytes>") throws {
        guard bytes.count >= 4, bytes[0] == 0x4F, bytes[1] == 0x62, bytes[2] == 0x6A, bytes[3] == 0x01 else {
            throw AvroError.malformed("\(source) is not an Avro object container file (bad magic)")
        }
        var header = AvroDecoder(bytes, names: [:], at: 4)
        guard case .map(let meta) = try header.decode(.map(.bytes)) else {
            throw AvroError.malformed("\(source): header metadata")
        }
        var md: [String: String] = [:]
        for (k, v) in meta { md[k] = String(decoding: v.bytes ?? [], as: UTF8.self) }
        let sync = try header.take(16)
        guard let schemaJSON = md["avro.schema"] else { throw AvroError.malformed("\(source) has no avro.schema") }
        let (schema, names) = try AvroSchema.parseWithNames(json: schemaJSON)
        let codec = md["avro.codec"] ?? "null"
        guard ["null", "deflate", "snappy"].contains(codec) else {
            throw AvroError.unsupported("\(source) uses the \(codec) codec; null, deflate and snappy are supported")
        }
        var records: [AvroValue] = []
        var r = header
        while !r.atEnd {
            let count = try r.long()
            let size = try r.long()
            guard count >= 0, size >= 0 else { throw AvroError.malformed("\(source): block count \(count), size \(size)") }
            let raw = try r.take(Int(size))
            let marker = try r.take(16)
            guard marker == sync else { throw AvroError.malformed("\(source): sync marker mismatch after a block") }
            let block: [UInt8]
            switch codec {
            case "deflate": block = try AvroCodecs.inflateRaw(raw)
            case "snappy": block = try AvroCodecs.snappyBlock(raw)
            default: block = raw
            }
            // A record takes at least one byte unless its schema encodes to nothing (a bare `null`,
            // an empty record); cap the count so a corrupt header cannot loop for ever or exhaust memory.
            guard count <= Swift.max(Int64(block.count), AvroFile.maxZeroWidthRecords) else {
                throw AvroError.malformed("\(source): block of \(count) records in \(block.count) bytes")
            }
            var b = AvroDecoder(block, names: names)
            for _ in 0..<count { records.append(try b.decode(schema)) }
        }
        self.metadata = md
        self.schema = schema
        self.codec = codec
        self.records = records
    }
}

/// The block codecs Avro containers use.
enum AvroCodecs {
    /// Raw DEFLATE (RFC 1951), which is what Avro's `deflate` codec stores. `COMPRESSION_ZLIB` in
    /// Apple's Compression framework is raw DEFLATE, streamed here because the output size is unknown.
    static func inflateRaw(_ src: [UInt8]) throws -> [UInt8] {
        if src.isEmpty { return [] }
        let stream = UnsafeMutablePointer<compression_stream>.allocate(capacity: 1)
        defer { stream.deallocate() }
        guard compression_stream_init(stream, COMPRESSION_STREAM_DECODE, COMPRESSION_ZLIB) == COMPRESSION_STATUS_OK else {
            throw AvroError.malformed("cannot initialise the DEFLATE decoder")
        }
        defer { compression_stream_destroy(stream) }
        let chunk = 64 * 1024
        let dst = UnsafeMutablePointer<UInt8>.allocate(capacity: chunk)
        defer { dst.deallocate() }
        var out: [UInt8] = []
        return try src.withUnsafeBufferPointer { sp -> [UInt8] in
            stream.pointee.src_ptr = sp.baseAddress!
            stream.pointee.src_size = sp.count
            while true {
                stream.pointee.dst_ptr = dst
                stream.pointee.dst_size = chunk
                let status = compression_stream_process(stream, Int32(COMPRESSION_STREAM_FINALIZE.rawValue))
                let produced = chunk - stream.pointee.dst_size
                if produced > 0 { out.append(contentsOf: UnsafeBufferPointer(start: dst, count: produced)) }
                switch status {
                case COMPRESSION_STATUS_END: return out
                case COMPRESSION_STATUS_OK:
                    if produced == 0 && stream.pointee.src_size == 0 {
                        throw AvroError.malformed("DEFLATE stream ended without its final block")
                    }
                default: throw AvroError.malformed("DEFLATE stream is corrupt")
                }
            }
        }
    }

    /// Avro's snappy block: raw snappy data followed by the big-endian CRC-32 of the uncompressed bytes.
    static func snappyBlock(_ src: [UInt8]) throws -> [UInt8] {
        guard src.count >= 4 else { throw AvroError.malformed("snappy block shorter than its checksum") }
        let body = Array(src[0..<(src.count - 4)])
        let out = try snappyDecompress(body)
        let want = src[(src.count - 4)...].reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
        guard crc32(out) == want else { throw AvroError.malformed("snappy block checksum mismatch") }
        return out
    }

    /// Raw snappy (the format, not the framing format).
    static func snappyDecompress(_ src: [UInt8]) throws -> [UInt8] {
        var pos = 0
        func varint() throws -> Int {
            var result = 0, shift = 0
            while true {
                guard pos < src.count, shift < 35 else { throw AvroError.malformed("snappy length preamble") }
                let b = src[pos]; pos += 1
                result |= Int(b & 0x7F) << shift
                if b & 0x80 == 0 { return result }
                shift += 7
            }
        }
        let total = try varint()
        var out = [UInt8]()
        out.reserveCapacity(Swift.min(total, 64 << 20))      // the preamble is not trusted for a huge allocation
        func need(_ n: Int) throws {
            guard pos + n <= src.count else { throw AvroError.malformed("snappy data truncated") }
        }
        func copy(offset: Int, length: Int) throws {
            guard offset > 0, offset <= out.count else { throw AvroError.malformed("snappy copy offset \(offset)") }
            let start = out.count - offset
            for i in 0..<length { out.append(out[start + i]) }
        }
        while pos < src.count {
            let tag = src[pos]; pos += 1
            switch tag & 3 {
            case 0:
                var len = Int(tag >> 2)
                if len >= 60 {
                    let extra = len - 59
                    try need(extra)
                    len = 0
                    for i in 0..<extra { len |= Int(src[pos + i]) << (8 * i) }
                    pos += extra
                }
                len += 1
                try need(len)
                out.append(contentsOf: src[pos..<(pos + len)])
                pos += len
            case 1:
                try need(1)
                let len = Int((tag >> 2) & 7) + 4
                let off = (Int(tag >> 5) << 8) | Int(src[pos]); pos += 1
                try copy(offset: off, length: len)
            case 2:
                try need(2)
                let len = Int(tag >> 2) + 1
                let off = Int(src[pos]) | (Int(src[pos + 1]) << 8); pos += 2
                try copy(offset: off, length: len)
            default:
                try need(4)
                let len = Int(tag >> 2) + 1
                var off = 0
                for i in 0..<4 { off |= Int(src[pos + i]) << (8 * i) }
                pos += 4
                try copy(offset: off, length: len)
            }
            guard out.count <= total else { throw AvroError.malformed("snappy output exceeds its declared length") }
        }
        guard out.count == total else { throw AvroError.malformed("snappy output is \(out.count) bytes, expected \(total)") }
        return out
    }

    private static let crcTable: [UInt32] = (0..<256).map { i -> UInt32 in
        var c = UInt32(i)
        for _ in 0..<8 { c = (c & 1) != 0 ? (0xEDB88320 ^ (c >> 1)) : (c >> 1) }
        return c
    }

    static func crc32(_ data: [UInt8]) -> UInt32 {
        var c: UInt32 = 0xFFFFFFFF
        for b in data { c = crcTable[Int((c ^ UInt32(b)) & 0xFF)] ^ (c >> 8) }
        return c ^ 0xFFFFFFFF
    }
}
