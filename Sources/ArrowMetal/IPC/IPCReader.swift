import Foundation

// MARK: - Logical types

/// Time unit of Arrow's temporal types.
public enum ArrowIPCTimeUnit: Int16, Sendable, CustomStringConvertible {
    case second = 0, millisecond = 1, microsecond = 2, nanosecond = 3
    public var description: String {
        switch self {
        case .second: return "s"
        case .millisecond: return "ms"
        case .microsecond: return "us"
        case .nanosecond: return "ns"
        }
    }
}

/// Unit of Arrow's `interval` type, in the order `Schema.fbs` gives them.
public enum ArrowIPCIntervalUnit: Int16, Sendable, CustomStringConvertible {
    case yearMonth = 0, dayTime = 1, monthDayNano = 2
    public var description: String {
        switch self {
        case .yearMonth: return "year_month"
        case .dayTime: return "day_time"
        case .monthDayNano: return "month_day_nano"
        }
    }
    /// Bytes per element of the values buffer.
    var byteWidth: Int {
        switch self {
        case .yearMonth: return 4
        case .dayTime: return 8
        case .monthDayNano: return 16
        }
    }
    /// The engine's spelling of the same unit, which calls `interval[year_month]` `months`.
    var arrayUnit: ArrowIntervalUnit {
        switch self {
        case .yearMonth: return .months
        case .dayTime: return .dayTime
        case .monthDayNano: return .monthDayNano
        }
    }
}

/// Whether a union stores one slot per row in every child (sparse) or packs each child (dense).
public enum ArrowIPCUnionMode: Int16, Sendable, CustomStringConvertible {
    case sparse = 0, dense = 1
    public var description: String { self == .sparse ? "sparse" : "dense" }
}

/// How an Arrow array of a given logical type is laid out in memory.
enum ArrowIPCStorage: Equatable {
    /// Fixed-width values: validity bitmap + values buffer.
    case fixedWidth(bytes: Int)
    /// Packed bits: validity bitmap + values bitmap.
    case bits
    /// Variable width: validity bitmap + offsets + data. `large` selects 64-bit offsets.
    case varBinary(large: Bool)
    /// `utf8_view` / `binary_view`: validity bitmap + 16-byte views, then as many variadic data buffers as
    /// the record batch's `variadicBufferCounts` gives this field (not counted here).
    case view
    /// A layout whose values live in child field nodes (list, struct, map, union, run-end encoded) or in
    /// no buffer at all (`null`). `buffers` counts only the field's own buffers, its children aside.
    case nested(buffers: Int)

    /// Number of Arrow buffers a field of this storage contributes to a record batch, its children aside
    /// (and, for a view field, its variadic data buffers aside).
    var bufferCount: Int {
        switch self {
        case .fixedWidth, .bits, .view: return 2
        case .varBinary: return 3
        case .nested(let n): return n
        }
    }
}

/// The Arrow logical types ArrowMetal can read from and write to IPC.
///
/// Each one maps to the array class that carries it: temporal types to `MetalTemporalArray`
/// (`.temporal`), `binary` / `large_binary` to a byte-flagged `MetalStringArray` (`.binary`), and a
/// dictionary-encoded column to `.dictionary(codes:values:)` — its codes come from the record batch and
/// its values from a `DictionaryBatch` message. A column written from the storage integers of a temporal
/// type against an explicit schema is accepted too, and reads back as `.temporal`.
public enum ArrowIPCType: Equatable, Sendable, CustomStringConvertible {
    case int(bits: Int, signed: Bool)
    case float(bits: Int)
    case bool
    case utf8
    case largeUtf8
    case binary
    case largeBinary
    case date32
    case date64
    case time32(ArrowIPCTimeUnit)
    case time64(ArrowIPCTimeUnit)
    case timestamp(ArrowIPCTimeUnit, timezone: String?)
    case duration(ArrowIPCTimeUnit)
    /// A dictionary-encoded column: `index` is the type of the codes stored in the record batch,
    /// `value` the type of the dictionary itself (which travels in a separate `DictionaryBatch`).
    indirect case dictionary(index: ArrowIPCType, value: ArrowIPCType)
    /// The Arrow `null` type: a length, every element null, and no buffers at all.
    case null
    /// `float16`: IEEE-754 binary16 bit patterns.
    case float16
    /// `decimal32` / `decimal64` / `decimal128` / `decimal256`; `bits` is the storage width.
    case decimal(precision: Int, scale: Int, bits: Int)
    /// `fixed_size_binary`: `byteWidth` raw bytes per element, no offsets.
    case fixedSizeBinary(byteWidth: Int)
    /// `interval[year_month]` / `[day_time]` / `[month_day_nano]`.
    case interval(ArrowIPCIntervalUnit)
    /// `list`: validity + int32 offsets over one child field (Arrow names it "item").
    indirect case list(ArrowIPCField)
    /// `large_list`: the same with 64-bit offsets.
    indirect case largeList(ArrowIPCField)
    /// `fixed_size_list`: validity only, every row covering exactly `size` child elements.
    indirect case fixedSizeList(ArrowIPCField, size: Int)
    /// `struct`: validity only, one child field per member.
    indirect case structure([ArrowIPCField])
    /// `map`: a list whose child is a non-nullable `entries` struct of `key` and `value`.
    indirect case map(entries: ArrowIPCField, keysSorted: Bool)
    /// `union`: type ids (plus offsets when dense) and one child field per variant.
    indirect case union(mode: ArrowIPCUnionMode, typeIDs: [Int32], children: [ArrowIPCField])
    /// `run_end_encoded`: no buffers, a `run_ends` child and a `values` child.
    indirect case runEndEncoded(runEnds: ArrowIPCField, values: ArrowIPCField)
    /// `utf8_view` (`string_view`): 16-byte views over inline bytes or variadic data buffers. Read only:
    /// the reader materialises it to the `utf8` layout, and the writer writes `utf8`.
    case utf8View
    /// `binary_view`: the same layout, bytes uninterpreted. Read as, and written as, `binary`.
    case binaryView
    /// `list_view`: validity + int32 offsets + int32 sizes over one child. Read as, and written as, `list`.
    indirect case listView(ArrowIPCField)
    /// `large_list_view`: the same with 64-bit offsets and sizes. Read as a list (offsets narrowed, as
    /// `large_list`'s are); an explicit schema that names it writes `large_list`.
    indirect case largeListView(ArrowIPCField)

    public var description: String {
        switch self {
        case .int(let b, let s): return "\(s ? "int" : "uint")\(b)"
        case .float(let b): return "float\(b)"
        case .bool: return "bool"
        case .utf8: return "utf8"
        case .largeUtf8: return "large_utf8"
        case .binary: return "binary"
        case .largeBinary: return "large_binary"
        case .date32: return "date32[day]"
        case .date64: return "date64[ms]"
        case .time32(let u): return "time32[\(u)]"
        case .time64(let u): return "time64[\(u)]"
        case .timestamp(let u, let tz): return "timestamp[\(u)\(tz.map { ", tz=\($0)" } ?? "")]"
        case .duration(let u): return "duration[\(u)]"
        case .dictionary(let i, let v): return "dictionary<\(i), \(v)>"
        case .null: return "null"
        case .float16: return "float16"
        case .decimal(let p, let s, let b): return "decimal\(b)(\(p), \(s))"
        case .fixedSizeBinary(let w): return "fixed_size_binary[\(w)]"
        case .interval(let u): return "interval[\(u)]"
        case .list(let f): return "list<\(f.name): \(f.type)>"
        case .largeList(let f): return "large_list<\(f.name): \(f.type)>"
        case .fixedSizeList(let f, let n): return "fixed_size_list<\(f.name): \(f.type)>[\(n)]"
        case .structure(let fs): return "struct<\(fs.map { "\($0.name): \($0.type)" }.joined(separator: ", "))>"
        case .map(let e, let sorted):
            guard case .structure(let fs) = e.type, fs.count == 2 else { return "map" }
            return "map<\(fs[0].type), \(fs[1].type)\(sorted ? ", keys_sorted" : "")>"
        case .union(let mode, _, let fs):
            return "\(mode)_union<\(fs.map { "\($0.name): \($0.type)" }.joined(separator: ", "))>"
        case .runEndEncoded(let r, let v): return "run_end_encoded<\(r.type), \(v.type)>"
        case .utf8View: return "string_view"
        case .binaryView: return "binary_view"
        case .listView(let f): return "list_view<\(f.name): \(f.type)>"
        case .largeListView(let f): return "large_list_view<\(f.name): \(f.type)>"
        }
    }

    /// The type the writer writes for this one: every view type becomes its classic counterparts
    /// (`utf8_view` to `utf8`, `binary_view` to `binary`, `list_view` to `list`, `large_list_view` to
    /// `large_list`), children included. Every other type is its own classic type.
    public var classic: ArrowIPCType {
        func c(_ f: ArrowIPCField) -> ArrowIPCField { f.classic }
        switch self {
        case .utf8View: return .utf8
        case .binaryView: return .binary
        case .listView(let f): return .list(c(f))
        case .largeListView(let f): return .largeList(c(f))
        case .list(let f): return .list(c(f))
        case .largeList(let f): return .largeList(c(f))
        case .fixedSizeList(let f, let n): return .fixedSizeList(c(f), size: n)
        case .structure(let fs): return .structure(fs.map(c))
        case .map(let e, let sorted): return .map(entries: c(e), keysSorted: sorted)
        case .union(let mode, let ids, let fs): return .union(mode: mode, typeIDs: ids, children: fs.map(c))
        case .runEndEncoded(let r, let v): return .runEndEncoded(runEnds: c(r), values: c(v))
        case .dictionary(let i, let v): return .dictionary(index: i, value: v.classic)
        default: return self
        }
    }

    /// The child fields the schema carries for this type, in the order Arrow writes them.
    var children: [ArrowIPCField] {
        switch self {
        case .list(let f), .largeList(let f), .fixedSizeList(let f, _), .map(let f, _),
             .listView(let f), .largeListView(let f): return [f]
        case .structure(let fs), .union(_, _, let fs): return fs
        case .runEndEncoded(let r, let v): return [r, v]
        // A dictionary field carries the value type's children; the index type has none.
        case .dictionary(_, let value): return value.children
        default: return []
        }
    }

    /// The physical type whose array class carries values of this logical type.
    public var physicalType: ArrowIPCType {
        switch self {
        case .date32, .time32: return .int(bits: 32, signed: true)
        case .date64, .time64, .timestamp, .duration: return .int(bits: 64, signed: true)
        case .binary: return .utf8
        case .largeBinary: return .largeUtf8
        // The view types are materialised to the classic layouts on read.
        case .utf8View, .binaryView: return .utf8
        case .listView(let f): return .list(f)
        case .largeListView(let f): return .largeList(f)
        // The record batch carries the codes; the values ride in a dictionary batch.
        case .dictionary(let index, _): return index.physicalType
        default: return self
        }
    }

    var storage: ArrowIPCStorage {
        switch self {
        case .int(let b, _): return .fixedWidth(bytes: b / 8)
        case .float(let b): return .fixedWidth(bytes: b / 8)
        case .bool: return .bits
        case .utf8, .binary: return .varBinary(large: false)
        case .largeUtf8, .largeBinary: return .varBinary(large: true)
        case .date32, .time32: return .fixedWidth(bytes: 4)
        case .date64, .time64, .timestamp, .duration: return .fixedWidth(bytes: 8)
        case .dictionary(let index, _): return index.storage
        case .float16: return .fixedWidth(bytes: 2)
        case .decimal(_, _, let bits): return .fixedWidth(bytes: bits / 8)
        case .fixedSizeBinary(let w): return .fixedWidth(bytes: w)
        case .interval(let u): return .fixedWidth(bytes: u.byteWidth)
        // Validity plus offsets; the values live in the child field node.
        case .list, .largeList, .map: return .nested(buffers: 2)
        // Validity only: a fixed-size list's offsets are implied and a struct has none.
        case .fixedSizeList, .structure: return .nested(buffers: 1)
        // Type ids, plus the offsets a dense union needs. A union carries no validity bitmap.
        case .union(let mode, _, _): return .nested(buffers: mode == .dense ? 2 : 1)
        // Neither of these has a buffer of its own.
        case .null, .runEndEncoded: return .nested(buffers: 0)
        case .utf8View, .binaryView: return .view
        // Validity, offsets and sizes; the values live in the child field node.
        case .listView, .largeListView: return .nested(buffers: 3)
        }
    }

    /// The value type of a dictionary column, or nil for everything else.
    var dictionaryValueType: ArrowIPCType? {
        if case .dictionary(_, let v) = self { return v }
        return nil
    }

    /// The Arrow temporal type this logical type maps to, or nil when it is not temporal.
    var temporalType: ArrowTemporalType? {
        func u(_ x: ArrowIPCTimeUnit) -> ArrowTemporalUnit {
            switch x {
            case .second: return .second
            case .millisecond: return .milli
            case .microsecond: return .micro
            case .nanosecond: return .nano
            }
        }
        switch self {
        case .date32: return .date32
        case .date64: return .date64
        case .time32(let x): return .time32(u(x))
        case .time64(let x): return .time64(u(x))
        case .timestamp(let x, let tz): return .timestamp(u(x), timezone: tz)
        case .duration(let x): return .duration(u(x))
        default: return nil
        }
    }
}

/// One column of an Arrow IPC schema.
public struct ArrowIPCField: Equatable, Sendable {
    public let name: String
    public let type: ArrowIPCType
    public let nullable: Bool
    /// The dictionary id that ties a dictionary-encoded column to its `DictionaryBatch` message.
    /// Nil for every other column; the writer assigns one per dictionary column when it derives a schema.
    public let dictionaryID: Int64?
    /// The field's `custom_metadata`, in order. An extension column carries `ARROW:extension:name` and
    /// `ARROW:extension:metadata` here, which is what lets a consumer rebuild the extension type.
    public let metadata: [(key: String, value: [UInt8])]
    public init(name: String, type: ArrowIPCType, nullable: Bool = true, dictionaryID: Int64? = nil,
                metadata: [(key: String, value: [UInt8])] = []) {
        self.name = name
        self.type = type
        self.nullable = nullable
        self.dictionaryID = dictionaryID
        self.metadata = metadata
    }

    /// `ARROW:extension:name` from the field's metadata, or nil when the field is not an extension type.
    public var extensionName: String? {
        metadata.first { $0.key == ArrowSchemaMetadata.extensionNameKey }.map { String(decoding: $0.value, as: UTF8.self) }
    }

    /// The same field with every view type replaced by its classic counterpart (see `ArrowIPCType.classic`).
    public var classic: ArrowIPCField {
        ArrowIPCField(name: name, type: type.classic, nullable: nullable, dictionaryID: dictionaryID, metadata: metadata)
    }

    public static func == (a: ArrowIPCField, b: ArrowIPCField) -> Bool {
        a.name == b.name && a.type == b.type && a.nullable == b.nullable && a.dictionaryID == b.dictionaryID
            && a.metadata.count == b.metadata.count
            && zip(a.metadata, b.metadata).allSatisfy { $0.key == $1.key && $0.value == $1.value }
    }
}

/// The schema of an IPC stream or file.
public struct ArrowIPCSchema: Equatable, Sendable {
    public let fields: [ArrowIPCField]
    public init(fields: [ArrowIPCField]) { self.fields = fields }
    public var names: [String] { fields.map(\.name) }
    public subscript(name: String) -> ArrowIPCField? { fields.first { $0.name == name } }
}

/// Which of the two IPC encapsulations to read or write.
public enum ArrowIPCFormat: Sendable {
    /// A bare sequence of encapsulated messages (`.arrows`, sockets, Flight).
    case stream
    /// The random-access file format: magic, messages, footer (`.arrow`).
    case file
}

/// `"ARROW1\0\0"`, the 8 bytes an Arrow IPC file starts with.
let arrowFileMagic: [UInt8] = [0x41, 0x52, 0x52, 0x4F, 0x57, 0x31, 0x00, 0x00]
/// The 6-byte magic the file ends with.
let arrowFileMagicTail: [UInt8] = [0x41, 0x52, 0x52, 0x4F, 0x57, 0x31]
/// Marks the start of an encapsulated message.
let arrowContinuation: UInt32 = 0xFFFF_FFFF

/// Where one encapsulated message lives in the source bytes.
struct ArrowIPCMessageRef {
    let metadataOffset: Int
    let metadataLength: Int
    let bodyOffset: Int
    let bodyLength: Int
}

/// Arrow's `CompressionType`: how the buffers of a message body were compressed.
enum ArrowIPCCodec: UInt8 {
    case lz4Frame = 0, zstd = 1
    var name: String { self == .lz4Frame ? "LZ4_FRAME" : "ZSTD" }
}

/// What one scan of the source found: its encapsulation, its schema and where every message lives.
struct ArrowIPCScan {
    let format: ArrowIPCFormat
    let schema: ArrowIPCSchema
    let messages: [ArrowIPCMessageRef]
    let dictionaries: [ArrowIPCMessageRef]
    /// For each record batch, how many dictionary messages precede it. A dictionary applies to the
    /// batches that follow it, which is what lets a stream replace or extend one part way through.
    let dictionariesBefore: [Int]
    /// Whether the schema declares big-endian buffers (the metadata itself is always little-endian).
    let bigEndian: Bool
}

/// Retains the source `Data` for as long as any buffer borrowed from it is alive.
final class ArrowIPCSourceHolder {
    let data: Data
    init(_ data: Data) { self.data = data }
}

// MARK: - Reader

/// Reads the Arrow IPC streaming and file formats into Metal-resident record batches.
///
/// ```swift
/// let batches = try ArrowIPCReader(url: url).readAll()
/// ```
///
/// A file opened by URL is memory mapped; each batch's buffers are copied into Metal shared
/// memory (Arrow only aligns body buffers to 8 bytes, and `MTLBuffer` needs page alignment to
/// wrap memory without a copy, so the zero-copy path almost never applies to a mapped file).
public final class ArrowIPCReader {
    /// The schema every batch in the source conforms to.
    public let schema: ArrowIPCSchema
    /// Whether the source is a stream or a random-access file.
    public let format: ArrowIPCFormat
    /// Number of record batches in the source.
    public var batchCount: Int { messages.count }
    /// Whether the source's schema declares big-endian buffers. Every buffer is then byte swapped on
    /// read, by the width of the values it holds, so the batches this reader returns are native.
    public let isBigEndian: Bool

    private let data: Data
    private let holder: ArrowIPCSourceHolder
    private let messages: [ArrowIPCMessageRef]
    /// `DictionaryBatch` messages, in the order they appear (stream) or the footer lists them (file).
    private let dictionaryMessages: [ArrowIPCMessageRef]
    /// For each record batch, how many of those messages precede it.
    private let dictionariesBefore: [Int]
    /// Dictionary values by id, for the dictionary messages applied so far.
    private var dictionaries: [Int64: AnyMetalArray] = [:]
    /// How many dictionary messages `dictionaries` reflects, counting from the first.
    private var appliedDictionaries = 0
    private let context: MetalContext
    /// Borrow mapped pages instead of copying when they happen to be page aligned.
    private let allowZeroCopy: Bool
    /// Set when the last batch built borrowed at least one buffer instead of copying it.
    public private(set) var lastBatchWasZeroCopy = false
    /// Scratch flag used while a batch is being materialised.
    private var borrowedAny = false

    public convenience init(data: Data, context: MetalContext = .shared) throws {
        try self.init(data: data, allowZeroCopy: false, context: context)
    }

    /// Memory maps `url` and parses its schema and message index.
    public convenience init(url: URL, context: MetalContext = .shared) throws {
        let mapped = try Data(contentsOf: url, options: .alwaysMapped)
        try self.init(data: mapped, allowZeroCopy: true, context: context)
    }

    private init(data: Data, allowZeroCopy: Bool, context: MetalContext) throws {
        self.data = data
        self.holder = ArrowIPCSourceHolder(data)
        self.context = context
        // Inline `Data` storage would not survive the pointer escaping `withUnsafeBytes`;
        // only a mapped or heap allocation (which is what a real IPC payload is) may be borrowed.
        self.allowZeroCopy = allowZeroCopy && data.count >= 65536
        let parsed = try data.withUnsafeBytes { raw in try ArrowIPCReader.scan(raw) }
        self.format = parsed.format
        self.schema = parsed.schema
        self.messages = parsed.messages
        self.dictionaryMessages = parsed.dictionaries
        self.dictionariesBefore = parsed.dictionariesBefore
        self.isBigEndian = parsed.bigEndian
    }

    /// Reads every record batch.
    public func readAll() throws -> [MetalRecordBatch] {
        try (0..<messages.count).map { try batch(at: $0) }
    }

    /// Reads one record batch by index (random access; the file format's whole point).
    public func batch(at index: Int) throws -> MetalRecordBatch {
        guard index >= 0, index < messages.count else {
            throw ArrowIPCError.malformed("record batch \(index) of \(messages.count)")
        }
        let ref = messages[index]
        return try data.withUnsafeBytes { raw in
            let meta = try FBBuf(raw, from: ref.metadataOffset, count: ref.metadataLength)
            let message = try meta.root()
            guard let header = try message.table(2) else { throw ArrowIPCError.malformed("message has no header") }
            return try buildBatch(header: header, raw: raw, bodyOffset: ref.bodyOffset, bodyLength: ref.bodyLength,
                                  dictionaryCount: dictionariesBefore[index])
        }
    }

    // MARK: message index

    private static func hasPrefix(_ raw: UnsafeRawBufferPointer, _ magic: [UInt8], at start: Int) -> Bool {
        guard start >= 0, start + magic.count <= raw.count else { return false }
        for i in 0..<magic.count where raw[start + i] != magic[i] { return false }
        return true
    }

    private static func scan(_ raw: UnsafeRawBufferPointer) throws -> ArrowIPCScan {
        guard raw.count >= 8 else { throw ArrowIPCError.notArrowIPC }
        if hasPrefix(raw, arrowFileMagic, at: 0) { return try scanFile(raw) }
        return try scanStream(raw)
    }

    private static func scanFile(_ raw: UnsafeRawBufferPointer) throws -> ArrowIPCScan {
        let n = raw.count
        guard n >= 8 + 10, hasPrefix(raw, arrowFileMagicTail, at: n - 6) else {
            throw ArrowIPCError.malformed("file does not end with the ARROW1 magic")
        }
        let footerLength = Int(raw.loadUnaligned(fromByteOffset: n - 10, as: Int32.self))
        guard footerLength > 0, 8 + footerLength + 10 <= n else {
            throw ArrowIPCError.malformed("footer length \(footerLength) does not fit the file")
        }
        let footerStart = n - 10 - footerLength
        let footer = try FBBuf(raw, from: footerStart, count: footerLength).root()
        guard let schemaTable = try footer.table(1) else { throw ArrowIPCError.malformed("footer has no schema") }
        let schema = try parseSchema(schemaTable)
        let bigEndian = try isBigEndian(schemaTable)
        var dictionaries: [ArrowIPCMessageRef] = []
        if let dicts = try footer.vector(2) {
            for i in 0..<dicts.count {
                let p = try dicts.structAt(i, stride: fbBlockStride)
                let offset = Int(try footer.buf.load(Int64.self, at: p))
                guard let msg = try message(raw, at: offset),
                      msg.headerType == FBMessageHeader.dictionaryBatch.rawValue else {
                    throw ArrowIPCError.malformed("dictionary block \(i) is not a dictionary batch")
                }
                dictionaries.append(msg.ref)
            }
        }
        var messages: [ArrowIPCMessageRef] = []
        if let blocks = try footer.vector(3) {
            messages.reserveCapacity(blocks.count)
            for i in 0..<blocks.count {
                // struct Block { offset: long; metaDataLength: int; bodyLength: long; }
                let p = try blocks.structAt(i, stride: fbBlockStride)
                let offset = Int(try footer.buf.load(Int64.self, at: p))
                guard let msg = try message(raw, at: offset) else {
                    throw ArrowIPCError.malformed("record batch block \(i) points at an empty message")
                }
                guard msg.headerType == FBMessageHeader.recordBatch.rawValue else {
                    if msg.headerType == FBMessageHeader.tensor.rawValue
                        || msg.headerType == FBMessageHeader.sparseTensor.rawValue {
                        throw tensorMessageError
                    }
                    throw msg.headerType == FBMessageHeader.dictionaryBatch.rawValue
                        ? ArrowIPCError.unsupported("dictionary batches")
                        : ArrowIPCError.malformed("block \(i) is not a record batch")
                }
                messages.append(msg.ref)
            }
        }
        // In the file format every dictionary is in scope for every batch: the footer indexes them all,
        // and the format forbids replacing one, so message order carries no meaning here.
        return ArrowIPCScan(format: .file, schema: schema, messages: messages, dictionaries: dictionaries,
                            dictionariesBefore: Array(repeating: dictionaries.count, count: messages.count),
                            bigEndian: bigEndian)
    }

    private static func scanStream(_ raw: UnsafeRawBufferPointer) throws -> ArrowIPCScan {
        var pos = 0
        var schema: ArrowIPCSchema? = nil
        var bigEndian = false
        var messages: [ArrowIPCMessageRef] = []
        var dictionaries: [ArrowIPCMessageRef] = []
        var dictionariesBefore: [Int] = []
        while pos < raw.count {
            guard let m = try message(raw, at: pos) else { break }     // end-of-stream marker
            switch m.headerType {
            case FBMessageHeader.schema.rawValue:
                guard schema == nil else { throw ArrowIPCError.malformed("a second schema message in one stream") }
                let meta = try FBBuf(raw, from: m.ref.metadataOffset, count: m.ref.metadataLength)
                guard let table = try meta.root().table(2) else { throw ArrowIPCError.malformed("schema message has no header") }
                schema = try parseSchema(table)
                bigEndian = try isBigEndian(table)
            case FBMessageHeader.recordBatch.rawValue:
                guard schema != nil else { throw ArrowIPCError.malformed("record batch before the schema message") }
                messages.append(m.ref)
                dictionariesBefore.append(dictionaries.count)
            case FBMessageHeader.dictionaryBatch.rawValue:
                guard schema != nil else { throw ArrowIPCError.malformed("dictionary batch before the schema message") }
                dictionaries.append(m.ref)
            case FBMessageHeader.tensor.rawValue, FBMessageHeader.sparseTensor.rawValue:
                throw tensorMessageError
            default:
                throw ArrowIPCError.malformed("unknown message header type \(m.headerType)")
            }
            pos = m.next
        }
        guard let schema else { throw ArrowIPCError.notArrowIPC }
        return ArrowIPCScan(format: .stream, schema: schema, messages: messages, dictionaries: dictionaries,
                            dictionariesBefore: dictionariesBefore, bigEndian: bigEndian)
    }

    /// Decodes one encapsulated message header at `pos`. Returns nil at the end-of-stream marker.
    private static func message(_ raw: UnsafeRawBufferPointer, at pos: Int)
        throws -> (ref: ArrowIPCMessageRef, headerType: UInt8, next: Int)? {
        guard pos >= 0, pos + 4 <= raw.count else { return nil }
        var p = pos
        if raw.loadUnaligned(fromByteOffset: p, as: UInt32.self) == arrowContinuation {
            p += 4
            guard p + 4 <= raw.count else { throw ArrowIPCError.truncated("message length after continuation") }
        }
        let metadataLength = Int(raw.loadUnaligned(fromByteOffset: p, as: Int32.self))
        p += 4
        if metadataLength == 0 { return nil }                          // end of stream
        guard metadataLength > 0, p + metadataLength <= raw.count else {
            throw ArrowIPCError.truncated("message metadata of \(metadataLength) bytes at \(p)")
        }
        let meta = try FBBuf(raw, from: p, count: metadataLength)
        let message = try meta.root()
        let headerType = try message.uint8(1)
        let bodyLength = Int(try message.int64(3))
        let bodyOffset = p + metadataLength
        guard bodyLength >= 0, bodyOffset + bodyLength <= raw.count else {
            throw ArrowIPCError.truncated("message body of \(bodyLength) bytes at \(bodyOffset)")
        }
        let ref = ArrowIPCMessageRef(metadataOffset: p, metadataLength: metadataLength,
                                     bodyOffset: bodyOffset, bodyLength: bodyLength)
        return (ref, headerType, bodyOffset + bodyLength)
    }

    // MARK: schema

    /// The error for an IPC `Tensor` or `SparseTensor` message. Those carry an n-dimensional array
    /// outside any record batch; the tensor that lives in a column is the `arrow.fixed_shape_tensor`
    /// extension type, which this reader does read.
    static let tensorMessageError = ArrowIPCError.unsupported(
        "IPC Tensor and SparseTensor messages are not read; only record batches (and their dictionaries) are. "
        + "A tensor column stored as the arrow.fixed_shape_tensor extension type is read")

    /// `Schema.endianness`: Little = 0, Big = 1.
    static func isBigEndian(_ table: FBTable) throws -> Bool {
        let endianness = try table.int16(0)
        guard endianness == 0 || endianness == 1 else {
            throw ArrowIPCError.malformed("schema endianness \(endianness) is neither Little (0) nor Big (1)")
        }
        return endianness == 1
    }

    static func parseSchema(_ table: FBTable) throws -> ArrowIPCSchema {
        _ = try isBigEndian(table)
        var fields: [ArrowIPCField] = []
        if let vec = try table.vector(1) {
            fields.reserveCapacity(vec.count)
            for i in 0..<vec.count { fields.append(try parseField(vec.table(i))) }
        }
        return ArrowIPCSchema(fields: fields)
    }

    private static func parseField(_ field: FBTable) throws -> ArrowIPCField {
        let name = try field.string(0) ?? ""
        let nullable = try field.bool(1)
        let encoding = try field.table(4)
        // The type is classified before its children are judged, so a type this reader does not handle
        // is named for what it is rather than for merely having children.
        let kindCode = try field.uint8(2)
        guard let kind = FBTypeKind(rawValue: kindCode) else {
            throw ArrowIPCError.malformed("unknown type code \(kindCode) for column '\(name)'")
        }
        guard let type = try field.table(3) else {
            throw ArrowIPCError.malformed("column '\(name)' has no type table")
        }
        var children: [ArrowIPCField] = []
        if let vec = try field.vector(5) {
            children.reserveCapacity(vec.count)
            for i in 0..<vec.count { children.append(try parseField(vec.table(i))) }
        }
        let valueType = try parseType(kind, type, children: children, column: name)
        // custom_metadata: [KeyValue], KeyValue { key: string; value: string }. The value is kept as bytes:
        // `ARROW:extension:metadata` is a byte blob that need not be UTF-8.
        var metadata: [(key: String, value: [UInt8])] = []
        if let vec = try field.vector(6) {
            metadata.reserveCapacity(vec.count)
            for i in 0..<vec.count {
                let kv = try vec.table(i)
                metadata.append((key: try kv.string(0) ?? "", value: try kv.stringBytes(1) ?? []))
            }
        }
        guard let encoding else {
            return ArrowIPCField(name: name, type: valueType, nullable: nullable, metadata: metadata)
        }
        // DictionaryEncoding { id: long; indexType: Int; isOrdered: bool; dictionaryKind: short }
        let id = try encoding.int64(0)
        var index = ArrowIPCType.int(bits: 32, signed: true)
        if let it = try encoding.table(1) {
            let bits = Int(try it.int32(0))
            guard [8, 16, 32, 64].contains(bits) else {
                throw ArrowIPCError.unsupported("\(bits)-bit dictionary indices (column '\(name)')")
            }
            index = .int(bits: bits, signed: try it.bool(1))
        }
        return ArrowIPCField(name: name, type: .dictionary(index: index, value: valueType),
                             nullable: nullable, dictionaryID: id, metadata: metadata)
    }

    private static func parseType(_ kind: FBTypeKind, _ type: FBTable, children: [ArrowIPCField],
                                  column: String) throws -> ArrowIPCType {
        func unit(_ raw: Int16) throws -> ArrowIPCTimeUnit {
            guard let u = ArrowIPCTimeUnit(rawValue: raw) else { throw ArrowIPCError.malformed("bad time unit \(raw)") }
            return u
        }
        /// The single child a list-shaped type has.
        func item() throws -> ArrowIPCField {
            guard children.count == 1 else {
                throw ArrowIPCError.malformed("\(kind.name) column '\(column)' has \(children.count) children, not 1")
            }
            return children[0]
        }
        switch kind {
        case .int:
            let bits = Int(try type.int32(0))
            let signed = try type.bool(1)
            guard [8, 16, 32, 64].contains(bits) else {
                throw ArrowIPCError.unsupported("\(bits)-bit integers (column '\(column)')")
            }
            return .int(bits: bits, signed: signed)
        case .floatingPoint:
            switch try type.int16(0) {
            case fbPrecisionSingle: return .float(bits: 32)
            case fbPrecisionDouble: return .float(bits: 64)
            case fbPrecisionHalf: return .float16
            default: throw ArrowIPCError.malformed("bad float precision for column '\(column)'")
            }
        case .bool: return .bool
        case .utf8: return .utf8
        case .largeUtf8: return .largeUtf8
        case .binary: return .binary
        case .largeBinary: return .largeBinary
        case .date:
            return try type.int16(0, default: fbDateUnitMillisecond) == fbDateUnitDay ? .date32 : .date64
        case .time:
            let u = try unit(type.int16(0, default: fbDateUnitMillisecond))
            return try type.int32(1, default: 32) == 32 ? .time32(u) : .time64(u)
        case .timestamp:
            let u = try unit(type.int16(0))
            let tz = try type.string(1)
            return .timestamp(u, timezone: (tz?.isEmpty ?? true) ? nil : tz)
        case .duration:
            return .duration(try unit(type.int16(0, default: fbDateUnitMillisecond)))
        case .null: return .null
        case .decimal:
            // Decimal { precision: int; scale: int; bitWidth: int = 128 }
            let bits = Int(try type.int32(2, default: 128))
            guard [32, 64, 128, 256].contains(bits) else {
                throw ArrowIPCError.unsupported("decimal\(bits) (column '\(column)')")
            }
            return .decimal(precision: Int(try type.int32(0)), scale: Int(try type.int32(1)), bits: bits)
        case .fixedSizeBinary:
            let width = Int(try type.int32(0))
            guard width >= 0 else { throw ArrowIPCError.malformed("negative fixed_size_binary width for column '\(column)'") }
            return .fixedSizeBinary(byteWidth: width)
        case .interval:
            let raw = try type.int16(0, default: 0)
            guard let u = ArrowIPCIntervalUnit(rawValue: raw) else {
                throw ArrowIPCError.malformed("bad interval unit \(raw) for column '\(column)'")
            }
            return .interval(u)
        case .list: return .list(try item())
        case .largeList: return .largeList(try item())
        case .fixedSizeList:
            let size = Int(try type.int32(0))
            guard size >= 0 else { throw ArrowIPCError.malformed("negative fixed_size_list width for column '\(column)'") }
            return .fixedSizeList(try item(), size: size)
        case .structKind: return .structure(children)
        case .map:
            let entries = try item()
            guard case .structure(let fields) = entries.type, fields.count == 2 else {
                throw ArrowIPCError.malformed("map column '\(column)' has a child that is not a struct of two fields")
            }
            return .map(entries: entries, keysSorted: try type.bool(0))
        case .union:
            // Union { mode: UnionMode; typeIds: [int] }; an absent vector means 0, 1, ... in child order.
            let raw = try type.int16(0, default: 0)
            guard let mode = ArrowIPCUnionMode(rawValue: raw) else {
                throw ArrowIPCError.malformed("bad union mode \(raw) for column '\(column)'")
            }
            var ids: [Int32] = (0..<children.count).map(Int32.init)
            if let vec = try type.vector(1) {
                guard vec.count == children.count else {
                    throw ArrowIPCError.malformed(
                        "union column '\(column)' declares \(vec.count) type ids for \(children.count) children")
                }
                ids = try (0..<vec.count).map { try vec.int32($0) }
            }
            for id in ids where Int8(exactly: id) == nil {
                throw ArrowIPCError.malformed("union column '\(column)' has a type id (\(id)) outside a byte")
            }
            return .union(mode: mode, typeIDs: ids, children: children)
        case .runEndEncoded:
            guard children.count == 2 else {
                throw ArrowIPCError.malformed("run-end encoded column '\(column)' needs run_ends and values children")
            }
            return .runEndEncoded(runEnds: children[0], values: children[1])
        case .utf8View: return .utf8View
        case .binaryView: return .binaryView
        case .listView: return .listView(try item())
        case .largeListView: return .largeListView(try item())
        default:
            throw ArrowIPCError.unsupported("\(kind.name) columns (column '\(column)')")
        }
    }

    // MARK: record batch

    /// The `RecordBatch` table's own fields: logical length, field nodes, buffer ranges and the codec
    /// its buffers were compressed with. Shared by record batch and dictionary batch messages (a
    /// dictionary batch wraps a record batch).
    private struct RecordBatchParts {
        let length: Int
        let nodes: [(length: Int, nullCount: Int)]
        let buffers: [ArrowIPCBodyBuffer]
        let codec: ArrowIPCCodec?
        /// `variadicBufferCounts`: one entry per `utf8_view` / `binary_view` field, in pre-order, giving
        /// how many data buffers follow that field's views.
        let variadicCounts: [Int]
    }

    private func recordBatchParts(header: FBTable, bodyLength: Int) throws -> RecordBatchParts {
        var codec: ArrowIPCCodec? = nil
        if let compression = try header.table(3) {
            // BodyCompression { codec: CompressionType; method: BodyCompressionMethod }
            let raw = try compression.uint8(0)
            guard let c = ArrowIPCCodec(rawValue: raw) else {
                throw ArrowIPCError.unsupported("body compression codec \(raw)")
            }
            let method = try compression.uint8(1)
            guard method == 0 else { throw ArrowIPCError.unsupported("body compression method \(method)") }
            codec = c
        }
        var variadicCounts: [Int] = []
        if let variadic = try header.vector(4) {
            variadicCounts.reserveCapacity(variadic.count)
            for i in 0..<variadic.count {
                let n = try variadic.int64(i)
                guard n >= 0, n <= Int64(Int32.max) else {
                    throw ArrowIPCError.malformed("variadic buffer count \(n) for view field \(i)")
                }
                variadicCounts.append(Int(n))
            }
        }
        let length = Int(try header.int64(0))
        guard length >= 0 else { throw ArrowIPCError.malformed("negative record batch length") }
        var nodes: [(length: Int, nullCount: Int)] = []
        if let vec = try header.vector(1) {
            nodes.reserveCapacity(vec.count)
            for i in 0..<vec.count {
                // struct FieldNode { length: long; null_count: long; }
                let p = try vec.structAt(i, stride: fbFieldNodeStride)
                nodes.append((Int(try header.buf.load(Int64.self, at: p)),
                              Int(try header.buf.load(Int64.self, at: p + 8))))
            }
        }
        var buffers: [ArrowIPCBodyBuffer] = []
        if let vec = try header.vector(2) {
            buffers.reserveCapacity(vec.count)
            for i in 0..<vec.count {
                // struct Buffer { offset: long; length: long; }
                let p = try vec.structAt(i, stride: fbBufferStride)
                let off = Int(try header.buf.load(Int64.self, at: p))
                let len = Int(try header.buf.load(Int64.self, at: p + 8))
                guard off >= 0, len >= 0, off + len <= bodyLength else {
                    throw ArrowIPCError.malformed("buffer \(i) (\(off)..<\(off + len)) escapes the \(bodyLength) byte body")
                }
                buffers.append(ArrowIPCBodyBuffer(offset: off, length: len))
            }
        }
        return RecordBatchParts(length: length, nodes: nodes, buffers: buffers, codec: codec,
                                variadicCounts: variadicCounts)
    }

    private func buildBatch(header: FBTable, raw: UnsafeRawBufferPointer, bodyOffset: Int, bodyLength: Int,
                            dictionaryCount: Int) throws -> MetalRecordBatch {
        let parts = try recordBatchParts(header: header, bodyLength: bodyLength)
        let length = parts.length

        lastBatchWasZeroCopy = false
        borrowedAny = false
        try materialiseDictionaries(upTo: dictionaryCount, raw: raw)
        let body = try messageBody(parts, raw: raw, bodyOffset: bodyOffset, fields: schema.fields)
        let cursor = ArrowIPCCursor(nodes: parts.nodes, body: body, variadicCounts: parts.variadicCounts)
        var columns: [AnyMetalArray] = []
        columns.reserveCapacity(schema.fields.count)
        for field in schema.fields {
            let column = try buildColumn(field: field, cursor: cursor)
            // Only a nested child may have a length of its own; every top-level column is the batch's.
            guard column.length == length else {
                throw ArrowIPCError.malformed("column '\(field.name)' has \(column.length) values in a \(length) row batch")
            }
            columns.append(column)
        }
        // Everything the message declared must have been used: a batch with a field node or a buffer the
        // schema does not account for is malformed, not a batch with something extra to ignore.
        guard cursor.nodeIndex == parts.nodes.count else {
            throw ArrowIPCError.malformed(
                "record batch declares \(parts.nodes.count) field nodes but the schema uses \(cursor.nodeIndex)")
        }
        guard cursor.bufferIndex == parts.buffers.count else {
            throw ArrowIPCError.malformed(
                "record batch declares \(parts.buffers.count) buffers but the schema uses \(cursor.bufferIndex)")
        }
        guard cursor.variadicIndex == parts.variadicCounts.count else {
            throw ArrowIPCError.malformed(
                "record batch declares \(parts.variadicCounts.count) variadic buffer counts but the schema has \(cursor.variadicIndex) view fields")
        }
        lastBatchWasZeroCopy = borrowedAny
        return try MetalRecordBatch(names: schema.fields.map(\.name), columns: columns)
    }

    // MARK: message body

    /// Wraps a message body, decompressing every buffer first when the message declared a codec.
    ///
    /// A big-endian source is copied buffer by buffer into shared memory (after decompression, when there
    /// is a codec) and byte swapped there, following the pre-order of `fields`: the columns the body holds
    /// (for a dictionary batch, the dictionary's value field).
    private func messageBody(_ parts: RecordBatchParts, raw: UnsafeRawBufferPointer, bodyOffset: Int,
                             fields: [ArrowIPCField]) throws -> ArrowIPCMessageBody {
        var plain: [MetalArrowBuffer]? = nil
        if let codec = parts.codec {
            plain = try decompress(codec, parts.buffers, raw: raw, bodyOffset: bodyOffset)
        }
        if isBigEndian {
            let buffers = try plain ?? parts.buffers.map { b in
                guard b.length > 0, let base = raw.baseAddress else {
                    return try MetalArrowBuffer.allocate(byteCount: 0, context: context)
                }
                return try MetalArrowBuffer.copy(from: base.advanced(by: bodyOffset + b.offset), byteCount: b.length,
                                                 context: context)
            }
            let plan = ArrowIPCByteSwap.plan(fields, variadicCounts: parts.variadicCounts)
            ArrowIPCByteSwap.apply(plan, to: buffers)
            plain = buffers
        }
        return ArrowIPCMessageBody(raw: raw, offset: bodyOffset, buffers: parts.buffers, plain: plain)
    }

    /// Arrow compresses a body one buffer at a time: an 8-byte little-endian uncompressed length, then the
    /// codec's own bytes — or, when that length is -1, the uncompressed bytes themselves (which is what a
    /// writer emits for a buffer compression would not shrink). A zero-length buffer carries no prefix.
    private func decompress(_ codec: ArrowIPCCodec, _ buffers: [ArrowIPCBodyBuffer],
                            raw: UnsafeRawBufferPointer, bodyOffset: Int) throws -> [MetalArrowBuffer] {
        guard let base = raw.baseAddress else { throw ArrowIPCError.truncated("empty source") }
        if codec == .zstd && !Zstd.isAvailable {
            throw ArrowIPCError.unsupported(
                "ZSTD body compression needs libzstd, which macOS does not ship and this SDK's Compression "
                + "framework does not implement. Install it (brew install zstd) or point ARROWMETAL_ZSTD at libzstd.1.dylib.")
        }
        return try buffers.enumerated().map { i, b in
            guard b.length > 0 else { return try MetalArrowBuffer.allocate(byteCount: 0, context: context) }
            guard b.length >= 8 else {
                throw ArrowIPCError.malformed("compressed buffer \(i) is \(b.length) bytes, too short for a length prefix")
            }
            let src = base.advanced(by: bodyOffset + b.offset)
            let declared = Int64(littleEndian: src.loadUnaligned(as: Int64.self))
            let payload = src.advanced(by: 8)
            let payloadLength = b.length - 8
            if declared < 0 {
                return try MetalArrowBuffer.copy(from: payload, byteCount: payloadLength, context: context)
            }
            guard declared <= Int64(Int.max) else {
                throw ArrowIPCError.malformed("compressed buffer \(i) claims \(declared) uncompressed bytes")
            }
            let want = Int(declared)
            let out = try MetalArrowBuffer.allocate(byteCount: want, zeroed: false, context: context)
            guard want > 0 else { return out }
            let inPtr = payload.assumingMemoryBound(to: UInt8.self)
            let outPtr = out.mutableTyped(UInt8.self)
            let produced: Int
            switch codec {
            case .lz4Frame:
                produced = try ArrowIPCLZ4.decodeFrame(inPtr, payloadLength, outPtr, want)
            case .zstd:
                do { produced = try Zstd.decompress(inPtr, payloadLength, outPtr, want) }
                catch { throw ArrowIPCError.malformed("ZSTD buffer \(i) failed to decompress") }
            }
            guard produced == want else {
                throw ArrowIPCError.malformed("\(codec.name) buffer \(i) produced \(produced) of \(want) bytes")
            }
            return out
        }
    }

    /// Bytes available in buffer `i`, after decompression when the body was compressed.
    private func byteCount(_ body: ArrowIPCMessageBody, _ i: Int) -> Int {
        body.plain?[i].byteCount ?? body.buffers[i].length
    }

    /// Copies (or borrows) `need` bytes of buffer `i` into Metal shared memory.
    private func take(_ body: ArrowIPCMessageBody, _ i: Int, need: Int, what: String) throws -> MetalArrowBuffer {
        guard need > 0 else { return try MetalArrowBuffer.allocate(byteCount: 0, context: context) }
        let have = byteCount(body, i)
        guard have >= need else {
            throw ArrowIPCError.malformed("\(what) holds \(have) bytes where \(need) are needed")
        }
        // A decompressed buffer is already in shared memory and nobody else holds it: view, do not copy.
        if let plain = body.plain { return plain[i].view(byteOffset: 0, byteCount: need) }
        guard let base = body.raw.baseAddress else { throw ArrowIPCError.truncated("empty source") }
        let src = base.advanced(by: body.offset + body.buffers[i].offset)
        if allowZeroCopy {
            let (buf, zeroCopy) = try MetalArrowBuffer.wrapOrCopy(src, byteCount: need, keepAlive: holder, context: context)
            if zeroCopy { borrowedAny = true }
            return buf
        }
        return try MetalArrowBuffer.copy(from: src, byteCount: need, context: context)
    }

    /// The first byte of buffer `i`. The paths that narrow 64-bit offsets read it directly.
    private func pointer(_ body: ArrowIPCMessageBody, _ i: Int) throws -> UnsafeRawPointer {
        if let plain = body.plain { return plain[i].contents }
        guard let base = body.raw.baseAddress else { throw ArrowIPCError.truncated("empty source") }
        return base.advanced(by: body.offset + body.buffers[i].offset)
    }

    // MARK: columns

    /// Builds one field, consuming its field node and buffers and then, recursively, its children's.
    ///
    /// This is the pre-order the IPC specification defines: a field's own node and buffers come first,
    /// each child's whole subtree after. A dictionary-encoded field is the one exception — the record
    /// batch carries only its codes, and its children (if the value type has any) travel in the
    /// `DictionaryBatch` message instead.
    ///
    /// A field that is the canonical `arrow.fixed_shape_tensor` comes back as `.extended`: the storage
    /// column built from the buffers (checked against the tensor's shape), plus the extension name, its
    /// metadata and the field's other keys. Every other column, including one whose metadata names some
    /// other extension type, comes back as its storage column, so every operator takes it as it takes
    /// the plain type; the field's `ARROW:extension:*` keys stay readable on `schema`.
    private func buildColumn(field: ArrowIPCField, cursor: ArrowIPCCursor) throws -> AnyMetalArray {
        let storage = try buildStorageColumn(field: field, cursor: cursor)
        guard !field.metadata.isEmpty, field.extensionName == ArrowFixedShapeTensorType.extensionName else {
            return storage
        }
        let extensionName = ArrowFixedShapeTensorType.extensionName
        var other = ArrowSchemaMetadata(field.metadata.map { ArrowSchemaMetadata.Pair(key: $0.key, value: $0.value) })
        let extensionMetadata = other[ArrowSchemaMetadata.extensionMetadataKey]
        other[ArrowSchemaMetadata.extensionNameKey] = nil
        other[ArrowSchemaMetadata.extensionMetadataKey] = nil
        // Checked against its storage, so a tensor column always has the shape it declares.
        _ = try ArrowFixedShapeTensorType(metadata: extensionMetadata ?? [], storage: storage, column: field.name)
        return .extended(MetalExtensionArray(storage: storage, name: extensionName, metadata: extensionMetadata,
                                             otherMetadata: other))
    }

    /// The column `field`'s buffers describe, before any extension type is applied.
    private func buildStorageColumn(field: ArrowIPCField, cursor: ArrowIPCCursor) throws -> AnyMetalArray {
        let name = field.name
        let node = try cursor.node(name)
        let length = node.length
        guard length >= 0 else { throw ArrowIPCError.malformed("column '\(name)' has a negative length") }
        // null_count == -1 means "unknown": recompute it from the bitmap.
        let nulls = node.nullCount < 0 ? -1 : node.nullCount
        let slots = try cursor.buffers(field.type.storage.bufferCount, name)
        let body = cursor.body

        func validity(_ slot: Int) throws -> MetalArrowBuffer? {
            guard nulls != 0 else { return nil }
            let bytes = Bitmap.byteCount(bits: length)
            guard bytes > 0 else { return nil }
            guard byteCount(body, slot) >= bytes else {
                if nulls > 0 {
                    throw ArrowIPCError.malformed("column '\(name)' declares \(nulls) nulls but has no validity bitmap")
                }
                return nil
            }
            return try take(body, slot, need: bytes, what: "the validity bitmap of column '\(name)'")
        }

        switch field.type {
        case .null:
            // The null type has no buffers at all, and every one of its values is null.
            return .null(MetalNullArray(length: length, context: context))

        case .list(let item), .largeList(let item):
            var large = false
            if case .largeList = field.type { large = true }
            let bitmap = try validity(slots[0])
            let offsets = try listOffsets(body, slot: slots[1], length: length, large: large, name: name)
            let child = try buildColumn(field: item, cursor: cursor)
            return .list(try makeList(length: length, nulls: nulls, validity: bitmap, offsets: offsets,
                                      child: child, kind: .variable, fieldName: item.name, name: name))

        case .fixedSizeList(let item, let size):
            let bitmap = try validity(slots[0])
            let child = try buildColumn(field: item, cursor: cursor)
            // A fixed-size list has no offsets buffer; the engine materialises the ones its layout implies.
            let offsets = try MetalArrowBuffer.allocate(byteCount: (length + 1) * 4, zeroed: false, context: context)
            let p = offsets.mutableTyped(Int32.self)
            for i in 0...length { p[i] = Int32(i * size) }
            return .list(try makeList(length: length, nulls: nulls, validity: bitmap, offsets: offsets,
                                      child: child, kind: .fixedSize(size), fieldName: item.name, name: name))

        case .structure(let fields):
            let bitmap = try validity(slots[0])
            let children = try fields.map { try buildColumn(field: $0, cursor: cursor) }
            for (child, f) in zip(children, fields) where child.length < length {
                throw ArrowIPCError.malformed(
                    "struct column '\(name)' is \(length) rows but its field '\(f.name)' has \(child.length)")
            }
            let s = try MetalStructArray(length: length, nullCount: 0, validity: bitmap, names: fields.map(\.name),
                                         children: children, context: context)
            if nulls < 0 { s.recomputeNullCount() } else { s.nullCount = nulls }
            return .structure(s)

        case .map(let entries, let keysSorted):
            let bitmap = try validity(slots[0])
            let offsets = try listOffsets(body, slot: slots[1], length: length, large: false, name: name)
            let child = try buildColumn(field: entries, cursor: cursor)
            guard case .structure = child else {
                throw ArrowIPCError.malformed("map column '\(name)' has a child that is not a struct")
            }
            let list = try makeList(length: length, nulls: nulls, validity: bitmap, offsets: offsets,
                                    child: child, kind: .variable, fieldName: entries.name, name: name)
            return .map(try MetalMapArray(entries: list, keysSorted: keysSorted))

        case .union(let mode, let typeIDs, let children):
            // A union carries no validity bitmap of its own: null-ness lives in its children.
            let ids = try take(body, slots[0], need: length, what: "the type ids of column '\(name)'")
            let typeIds = MetalArray<Int8>(length: length, nullCount: 0, validity: nil, values: ids, context: context)
            var offsets: MetalArray<Int32>? = nil
            if mode == .dense {
                let b = try take(body, slots[1], need: length * 4, what: "the offsets of column '\(name)'")
                offsets = MetalArray<Int32>(length: length, nullCount: 0, validity: nil, values: b, context: context)
            }
            let kids = try children.map { try buildColumn(field: $0, cursor: cursor) }
            let codes = typeIDs.map { Int8(truncatingIfNeeded: $0) }
            let u = try MetalUnionArray(mode: mode == .dense ? .dense : .sparse, length: length, typeIds: typeIds,
                                        offsets: offsets, typeCodes: codes, names: children.map(\.name),
                                        children: kids, context: context)
            return .union(u)

        case .utf8View, .binaryView:
            let bitmap = try validity(slots[0])
            let dataSlots = try cursor.buffers(try cursor.variadicCount(name), name)
            let a = try materialiseViews(body, views: slots[1], data: dataSlots, length: length, nulls: nulls,
                                         validity: bitmap, name: name)
            return field.type == .binaryView ? .binary(markBinary(a)) : .string(a)

        case .listView(let item), .largeListView(let item):
            var large = false
            if case .largeListView = field.type { large = true }
            let bitmap = try validity(slots[0])
            let child = try buildColumn(field: item, cursor: cursor)
            return .list(try materialiseListView(body, offsets: slots[1], sizes: slots[2], large: large,
                                                 length: length, nulls: nulls, validity: bitmap, child: child,
                                                 fieldName: item.name, name: name))

        case .runEndEncoded(let endsField, let valuesField):
            // No buffers of its own: the node carries the logical length, the children everything else.
            let ends = try buildColumn(field: endsField, cursor: cursor)
            let values = try buildColumn(field: valuesField, cursor: cursor)
            let runEnds = try narrowToInt32(ends, what: "run ends", column: name)
            guard values.length >= runEnds.length else {
                throw ArrowIPCError.malformed(
                    "run-end encoded column '\(name)' has \(runEnds.length) run ends for \(values.length) values")
            }
            guard runEndLogicalLength(runEnds) == length else {
                throw ArrowIPCError.malformed(
                    "run-end encoded column '\(name)' declares \(length) rows but its runs cover \(runEndLogicalLength(runEnds))")
            }
            return .runEndEncoded(runEnds: runEnds, values: values)

        default:
            return try buildFlatColumn(field: field, length: length, nulls: nulls, slots: slots, body: body,
                                       validity: try validity(slots[0]))
        }
    }

    /// A list array plus the checks the engine's own invariants need.
    private func makeList(length: Int, nulls: Int, validity: MetalArrowBuffer?, offsets: MetalArrowBuffer,
                          child: AnyMetalArray, kind: ArrowListKind, fieldName: String,
                          name: String) throws -> MetalListArray {
        let list = MetalListArray(length: length, nullCount: 0, validity: validity, offsets: offsets,
                                  values: child, kind: kind, fieldName: fieldName, context: context)
        if nulls < 0 { list.recomputeNullCount() } else { list.setNullCount(nulls) }
        let range = list.childRange
        guard range.lowerBound >= 0, range.upperBound >= range.lowerBound, range.upperBound <= child.length else {
            throw ArrowIPCError.malformed(
                "column '\(name)' covers child elements \(range.lowerBound)..<\(range.upperBound) of a \(child.length) element child")
        }
        return list
    }

    /// The int32 offsets of a list-shaped field. 64-bit offsets are narrowed, as `large_utf8`'s are.
    private func listOffsets(_ body: ArrowIPCMessageBody, slot: Int, length: Int, large: Bool,
                             name: String) throws -> MetalArrowBuffer {
        let need = (length + 1) * (large ? 8 : 4)
        if byteCount(body, slot) < need {
            // pyarrow writes no offsets at all for an empty column; every offset is then zero.
            guard length == 0 else { throw ArrowIPCError.malformed("column '\(name)' has a short offsets buffer") }
            return try MetalArrowBuffer.allocate(byteCount: 4, zeroed: true, context: context)
        }
        guard large else { return try take(body, slot, need: need, what: "the offsets of column '\(name)'") }
        let src = try pointer(body, slot)
        let out = try MetalArrowBuffer.allocate(byteCount: (length + 1) * 4, zeroed: false, context: context)
        let dst = out.mutableTyped(Int32.self)
        for i in 0...length {
            let v = src.loadUnaligned(fromByteOffset: i * 8, as: Int64.self)
            guard v >= 0, v <= Int64(Int32.max) else {
                throw ArrowIPCError.unsupported("64-bit offsets over 2 GB (column '\(name)')")
            }
            dst[i] = Int32(v)
        }
        return out
    }

    // MARK: view types

    /// Materialises a `utf8_view` / `binary_view` column into the offsets-plus-data layout.
    ///
    /// A tight CPU pass, sharded over the cores in blocks of 64K rows. The first walk over the 16-byte
    /// views turns lengths into int32 offsets (checking every out-of-line view against the data buffer
    /// it names); the second copies the bytes: a fixed 12-byte move for an inline view, whole words for
    /// a short out-of-line one and a `memcpy` for any other. A null row contributes no bytes whatever
    /// its view says. The engine has no GPU kernel that reads views, and the views and data buffers of
    /// a mapped file are not GPU-visible memory until copied.
    private func materialiseViews(_ body: ArrowIPCMessageBody, views slot: Int, data dataSlots: [Int], length: Int,
                                  nulls: Int, validity: MetalArrowBuffer?, name: String) throws -> MetalStringArray {
        let offsets = try MetalArrowBuffer.allocate(byteCount: (length + 1) * 4, zeroed: false, context: context)
        let o = offsets.mutableTyped(Int32.self)
        o[0] = 0
        guard length > 0 else {
            return MetalStringArray(length: 0, nullCount: 0, validity: nil, offsets: offsets,
                                    data: try MetalArrowBuffer.allocate(byteCount: 0, context: context), context: context)
        }
        let have = byteCount(body, slot)
        guard have >= length * 16 else {
            throw ArrowIPCError.malformed("view column '\(name)' holds \(have) bytes of views where \(length * 16) are needed")
        }
        let v = try pointer(body, slot)
        let starts: [UnsafeRawPointer?] = try dataSlots.map { byteCount(body, $0) > 0 ? try pointer(body, $0) : nil }
        let sizes: [Int] = dataSlots.map { byteCount(body, $0) }
        let valid = validity?.typed(UInt8.self)
        let block = 1 << 16
        let blocks = (length + block - 1) / block
        func each(_ body: (Int, Range<Int>) -> Void) {
            if blocks == 1 { return body(0, 0..<length) }
            DispatchQueue.concurrentPerform(iterations: blocks) { k in
                body(k, (k * block)..<Swift.min(length, (k + 1) * block))
            }
        }

        // Pass 1: each block's lengths to offsets relative to the block, then every block moved to its
        // place. `failures` keeps the first bad view of each block.
        var blockTotals = [Int](repeating: 0, count: blocks)
        var failures = [ArrowIPCError?](repeating: nil, count: blocks)
        blockTotals.withUnsafeMutableBufferPointer { totals in
            failures.withUnsafeMutableBufferPointer { errors in
                each { k, rows in
                    var total = 0
                    for i in rows {
                        let at = i &* 16
                        var n = Int(v.loadUnaligned(fromByteOffset: at, as: Int32.self))
                        if let valid, !Bitmap.isSet(valid, i) {
                            n = 0
                        } else if n > 12 {
                            let b = Int(v.loadUnaligned(fromByteOffset: at + 8, as: Int32.self))
                            let off = Int(v.loadUnaligned(fromByteOffset: at + 12, as: Int32.self))
                            guard b >= 0, b < sizes.count, off >= 0, off + n <= sizes[b] else {
                                errors[k] = .malformed(
                                    "view \(i) of column '\(name)' points at bytes \(off)..<\(off + n) of data buffer \(b), "
                                    + "outside the buffers the batch holds")
                                return
                            }
                        } else if n < 0 {
                            errors[k] = .malformed("view \(i) of column '\(name)' has a negative length")
                            return
                        }
                        total &+= n
                        o[i + 1] = Int32(truncatingIfNeeded: total)
                    }
                    totals[k] = total
                }
            }
        }
        if let first = failures.first(where: { $0 != nil }) { throw first! }
        var bases = [Int](repeating: 0, count: blocks)
        var total = 0
        for k in 0..<blocks { bases[k] = total; total += blockTotals[k] }
        guard total <= Int(Int32.max) else {
            throw ArrowIPCError.unsupported(
                "view column '\(name)' over 2 GB: the engine's utf8 and binary arrays have 32-bit offsets")
        }
        if blocks > 1 {
            each { k, rows in
                let base = Int32(bases[k])
                if base != 0 { for i in rows { o[i + 1] &+= base } }
            }
        }

        // Pass 2: the bytes. A short row moves whole words when they fit before the end of its block's
        // bytes: whatever lands past the row's end is then overwritten by a later row of the same block,
        // on the same thread, and never reaches another block's rows. Anything else is a `memcpy`.
        let data = try MetalArrowBuffer.allocate(byteCount: total, zeroed: false, context: context)
        let d = data.mutableTyped(UInt8.self)
        let raw = UnsafeMutableRawPointer(d)
        each { _, rows in
            let blockEnd = Int(o[rows.upperBound])
            for i in rows {
                let start = Int(o[i]), n = Int(o[i + 1]) &- start
                guard n > 0 else { continue }
                let at = i &* 16
                if n <= 12 {
                    let src = v.advanced(by: at + 4)
                    if start + 12 <= blockEnd {
                        raw.storeBytes(of: src.loadUnaligned(as: UInt64.self), toByteOffset: start, as: UInt64.self)
                        raw.storeBytes(of: src.loadUnaligned(fromByteOffset: 8, as: UInt32.self),
                                       toByteOffset: start + 8, as: UInt32.self)
                    } else {
                        memcpy(d + start, src, n)
                    }
                    continue
                }
                let b = Int(v.loadUnaligned(fromByteOffset: at + 8, as: Int32.self))
                let off = Int(v.loadUnaligned(fromByteOffset: at + 12, as: Int32.self))
                let src = starts[b]!.advanced(by: off)
                let words = (n + 7) & ~7
                if n <= 32 && start + words <= blockEnd && off + words <= sizes[b] {
                    for w in stride(from: 0, to: n, by: 8) {
                        raw.storeBytes(of: src.loadUnaligned(fromByteOffset: w, as: UInt64.self),
                                       toByteOffset: start + w, as: UInt64.self)
                    }
                } else {
                    memcpy(d + start, src, n)
                }
            }
        }
        let a = MetalStringArray(length: length, nullCount: 0, validity: validity, offsets: offsets,
                                 data: data, context: context)
        if nulls < 0 { a.recomputeNullCount() } else { a.setNullCount(nulls) }
        return a
    }

    /// Materialises a `list_view` / `large_list_view` column into the list layout (int32 offsets over a
    /// child whose rows are consecutive).
    ///
    /// One CPU pass over the offsets and sizes checks every row against the child and decides between two
    /// cases. When the valid, non-empty rows are already consecutive and in order (what a `list` cast to
    /// `list_view` gives), the offsets are derived and the child is used as it is, with no copy.
    /// Otherwise a second pass, sharded over the cores, writes the child index of every element and the
    /// rows are gathered from the child with the engine's `take`, a GPU gather for primitive and string
    /// children. A
    /// null row contributes no child elements whatever its size says.
    private func materialiseListView(_ body: ArrowIPCMessageBody, offsets offsetSlot: Int, sizes sizeSlot: Int,
                                     large: Bool, length: Int, nulls: Int, validity: MetalArrowBuffer?,
                                     child: AnyMetalArray, fieldName: String, name: String) throws -> MetalListArray {
        let out = try MetalArrowBuffer.allocate(byteCount: (length + 1) * 4, zeroed: false, context: context)
        let o = out.mutableTyped(Int32.self)
        o[0] = 0
        var values = child
        if length > 0 {
            let width = large ? 8 : 4
            for (slot, what) in [(offsetSlot, "offsets"), (sizeSlot, "sizes")] where byteCount(body, slot) < length * width {
                throw ArrowIPCError.malformed("list view column '\(name)' has a short \(what) buffer")
            }
            let offs = try pointer(body, offsetSlot), lens = try pointer(body, sizeSlot)
            let gather = large
                ? try listViewOffsets(Int64.self, offs, lens, o, length: length, validity: validity, child: child.length, name: name)
                : try listViewOffsets(Int32.self, offs, lens, o, length: length, validity: validity, child: child.length, name: name)
            if gather {
                let total = Int(o[length])
                let indices = try MetalArrowBuffer.allocate(byteCount: Swift.max(total, 1) * 4, zeroed: false,
                                                            context: context)
                let p = indices.mutableTyped(Int32.self)
                // Rows write disjoint ranges of the index buffer, so blocks of rows run on all cores.
                let block = 1 << 16
                DispatchQueue.concurrentPerform(iterations: (length + block - 1) / block) { b in
                    for i in (b * block)..<Swift.min(length, (b + 1) * block) {
                        let from = Int(o[i]), n = Int(o[i + 1]) &- from
                        guard n > 0 else { continue }
                        let start = large ? Int(offs.loadUnaligned(fromByteOffset: i &* 8, as: Int64.self))
                                          : Int(offs.loadUnaligned(fromByteOffset: i &* 4, as: Int32.self))
                        for k in 0..<n { p[from &+ k] = Int32(truncatingIfNeeded: start &+ k) }
                    }
                }
                let idx = MetalArray<Int32>(length: total, nullCount: 0, validity: nil, values: indices, context: context)
                values = try child.take(idx)
            }
        }
        return try makeList(length: length, nulls: nulls, validity: validity, offsets: out, child: values,
                            kind: .variable, fieldName: fieldName, name: name)
    }

    /// The first pass of `materialiseListView`: writes the list offsets of the rows' sizes into `o` and
    /// returns whether the child has to be gathered (false when the rows are consecutive and in order,
    /// in which case `o` already points into the child as it is).
    private func listViewOffsets<T: FixedWidthInteger & SignedInteger>(
        _: T.Type, _ offs: UnsafeRawPointer, _ lens: UnsafeRawPointer, _ o: UnsafeMutablePointer<Int32>,
        length: Int, validity: MetalArrowBuffer?, child childLength: Int, name: String) throws -> Bool {
        let w = MemoryLayout<T>.size
        let valid = validity?.typed(UInt8.self)
        var contiguous = true
        var next = -1             // where the next non-empty row must start to keep the child consecutive
        var base = 0, total = 0
        for i in 0..<length {
            if let valid, !Bitmap.isSet(valid, i) { o[i + 1] = Int32(truncatingIfNeeded: total); continue }
            let start = Int(offs.loadUnaligned(fromByteOffset: i &* w, as: T.self))
            let size = Int(lens.loadUnaligned(fromByteOffset: i &* w, as: T.self))
            guard start >= 0, size >= 0 else {
                throw ArrowIPCError.malformed("row \(i) of list view column '\(name)' has a negative offset or size")
            }
            guard start <= Int(Int32.max), size <= Int(Int32.max) else {
                throw ArrowIPCError.unsupported("64-bit offsets over 2 GB (column '\(name)')")
            }
            guard start + size <= childLength else {
                throw ArrowIPCError.malformed(
                    "row \(i) of list view column '\(name)' covers child elements \(start)..<\(start + size) "
                    + "of a \(childLength) element child")
            }
            if size > 0 {
                if next < 0 { base = start } else if start != next { contiguous = false }
                next = start + size
            }
            total &+= size
            o[i + 1] = Int32(truncatingIfNeeded: total)
        }
        guard total <= Int(Int32.max) else { throw ArrowIPCError.unsupported("64-bit offsets over 2 GB (column '\(name)')") }
        // Consecutive rows: point the offsets at the child as it is.
        if contiguous, base > 0 { for i in 0...length { o[i] &+= Int32(base) } }
        return !contiguous
    }

    /// Every type whose values are in its own buffers: the primitives, bool, utf8 / binary, the
    /// temporal types, the decimals, `float16`, `fixed_size_binary`, `interval`, and dictionary codes.
    private func buildFlatColumn(field: ArrowIPCField, length: Int, nulls: Int, slots: [Int],
                                 body: ArrowIPCMessageBody,
                                 validity bitmap: MetalArrowBuffer?) throws -> AnyMetalArray {
        let name = field.name
        func values(_ width: Int) throws -> MetalArrowBuffer {
            try take(body, slots[1], need: length * width, what: "the values of column '\(name)'")
        }
        func make<T: ArrowPrimitive>(_: T.Type, _ buffer: MetalArrowBuffer) -> MetalArray<T> {
            let a = MetalArray<T>(length: length, nullCount: 0, validity: bitmap, values: buffer, context: context)
            if nulls < 0 { a.recomputeNullCount() } else { a.nullCount = nulls }
            return a
        }

        // The types whose array class is not a MetalArray, but whose layout is still validity + values.
        switch field.type {
        case .decimal(let precision, let scale, let bits):
            let buffer = try values(bits / 8)
            if bits == 128 || bits == 256 {
                let a = MetalDecimalArray(type: try ArrowDecimalType(precision: precision, scale: scale, bitWidth: bits),
                                          length: length, nullCount: 0, validity: bitmap, values: buffer, context: context)
                if nulls < 0 { a.recomputeNullCount() } else { a.setNullCount(nulls) }
                return .decimal(a)
            }
            let t = try ArrowSmallDecimalType(precision: precision, scale: scale, bitWidth: bits)
            let a = bits == 32
                ? try MetalSmallDecimalArray(type: t, make(Int32.self, buffer))
                : try MetalSmallDecimalArray(type: t, make(Int64.self, buffer))
            return .smallDecimal(a)
        case .fixedSizeBinary(let width):
            let a = MetalFixedBinaryArray(byteWidth: width, length: length, nullCount: 0, validity: bitmap,
                                          values: try values(width), context: context)
            if nulls < 0 { a.recomputeNullCount() } else { a.setNullCount(nulls) }
            return .fixedBinary(a)
        case .float16:
            return .float16(MetalFloat16Array(bits: make(UInt16.self, try values(2))))
        case .interval(let unit):
            let a = MetalIntervalArray(unit: unit.arrayUnit, length: length, nullCount: 0, validity: bitmap,
                                       values: try values(unit.byteWidth), context: context)
            if nulls < 0 { a.recomputeNullCount() } else { a.setNullCount(nulls) }
            return .interval(a)
        default: break
        }

        switch field.type.storage {
        case .nested:
            throw ArrowIPCError.malformed("column '\(name)' is \(field.type), which has no buffers of its own")
        case .view:
            throw ArrowIPCError.malformed("column '\(name)' is \(field.type), which the view path reads")

        case .fixedWidth(let width):
            let buffer = try values(width)
            var flat: AnyMetalArray
            switch field.type.physicalType {
            case .int(bits: 8, signed: true): flat = .int8(make(Int8.self, buffer))
            case .int(bits: 8, signed: false): flat = .uint8(make(UInt8.self, buffer))
            case .int(bits: 16, signed: true): flat = .int16(make(Int16.self, buffer))
            case .int(bits: 16, signed: false): flat = .uint16(make(UInt16.self, buffer))
            case .int(bits: 32, signed: true): flat = .int32(make(Int32.self, buffer))
            case .int(bits: 32, signed: false): flat = .uint32(make(UInt32.self, buffer))
            case .int(bits: 64, signed: true): flat = .int64(make(Int64.self, buffer))
            case .int(bits: 64, signed: false): flat = .uint64(make(UInt64.self, buffer))
            case .float(bits: 32): flat = .float32(make(Float.self, buffer))
            case .float(bits: 64): flat = .float64(make(Double.self, buffer))
            default: throw ArrowIPCError.unsupported("\(field.type) columns (column '\(name)')")
            }
            // date / time / timestamp / duration keep their logical type: the storage integers are
            // wrapped in a MetalTemporalArray, so `column.asTemporal` round trips.
            if let t = field.type.temporalType {
                switch flat {
                case .int32(let a): return .temporal(try MetalTemporalArray(type: t, a))
                case .int64(let a): return .temporal(try MetalTemporalArray(type: t, a))
                default: throw ArrowIPCError.malformed("column '\(name)' has the wrong storage for \(field.type)")
                }
            }
            // A dictionary-encoded column: the record batch holds the codes, the values came in a
            // dictionary batch. Codes are narrowed to int32, as everywhere else in ArrowMetal.
            if case .dictionary = field.type {
                guard let id = field.dictionaryID, let values = dictionaries[id] else {
                    throw ArrowIPCError.malformed("column '\(name)' has no dictionary batch")
                }
                return .dictionary(codes: try narrowToInt32(flat, what: "dictionary indices", column: name),
                                   values: values)
            }
            return flat

        case .bits:
            let buffer = try take(body, slots[1], need: Bitmap.byteCount(bits: length),
                                  what: "the values of column '\(name)'")
            let a = MetalBooleanArray(length: length, nullCount: 0, validity: bitmap, values: buffer, context: context)
            if nulls < 0 { a.recomputeNullCount() } else { a.nullCount = nulls }
            return .boolean(a)

        case .varBinary(let large):
            let offsets: MetalArrowBuffer
            let total: Int
            if large {
                // Narrow 64-bit offsets; MetalStringArray always stores 32-bit ones.
                let need = (length + 1) * 8
                guard byteCount(body, slots[1]) >= need || length == 0 else {
                    throw ArrowIPCError.malformed("column '\(name)' has a short offsets buffer")
                }
                let out = try MetalArrowBuffer.allocate(byteCount: (length + 1) * 4, zeroed: true, context: context)
                if byteCount(body, slots[1]) >= need {
                    let src = try pointer(body, slots[1])
                    let dst = out.mutableTyped(Int32.self)
                    for i in 0...length {
                        let v = src.loadUnaligned(fromByteOffset: i * 8, as: Int64.self)
                        guard v >= 0, v <= Int64(Int32.max) else {
                            throw ArrowIPCError.unsupported("large binary column '\(name)' over 2 GB")
                        }
                        dst[i] = Int32(v)
                    }
                }
                offsets = out
                total = Int(out.typed(Int32.self)[length])
            } else {
                let need = (length + 1) * 4
                if byteCount(body, slots[1]) >= need {
                    offsets = try take(body, slots[1], need: need, what: "the offsets of column '\(name)'")
                } else if length == 0 {
                    offsets = try MetalArrowBuffer.allocate(byteCount: need, zeroed: true, context: context)
                } else {
                    throw ArrowIPCError.malformed("column '\(name)' has a short offsets buffer")
                }
                total = Int(offsets.typed(Int32.self)[length])
            }
            guard total >= 0 else { throw ArrowIPCError.malformed("column '\(name)' has a negative final offset") }
            let bytes: MetalArrowBuffer = total == 0
                ? try MetalArrowBuffer.allocate(byteCount: 0, context: context)
                : try take(body, slots[2], need: total, what: "the bytes of column '\(name)'")
            let a = MetalStringArray(length: length, nullCount: 0, validity: bitmap,
                                     offsets: offsets, data: bytes, context: context)
            if nulls < 0 { a.recomputeNullCount() } else { a.setNullCount(nulls) }
            // binary / large_binary keep the utf8 layout but come back as `.binary`, bytes uninterpreted.
            switch field.type {
            case .binary, .largeBinary: return .binary(markBinary(a))
            default: return .string(a)
            }
        }
    }

    /// Narrows an integer column of any width to the int32 array dictionary codes and run ends are held in.
    private func narrowToInt32(_ flat: AnyMetalArray, what: String, column: String) throws -> MetalArray<Int32> {
        switch flat {
        case .int32(let a): return a
        case .int8(let a): return try a.cast(to: Int32.self)
        case .uint8(let a): return try a.cast(to: Int32.self)
        case .int16(let a): return try a.cast(to: Int32.self)
        case .uint16(let a): return try a.cast(to: Int32.self)
        case .uint32(let a): return try a.cast(to: Int32.self)
        case .int64(let a): return try a.cast(to: Int32.self)
        case .uint64(let a): return try a.cast(to: Int32.self)
        default: throw ArrowIPCError.malformed("the \(what) of column '\(column)' are not integers")
        }
    }

    // MARK: dictionaries

    /// Applies the first `count` `DictionaryBatch` messages, in the order the source carries them.
    ///
    /// A dictionary applies to the batches that follow it, so a stream may replace one part way through
    /// or extend it with a delta. Reading batches out of order can ask for fewer messages than are
    /// already applied; the walk then starts again from the first one.
    private func materialiseDictionaries(upTo count: Int, raw: UnsafeRawBufferPointer) throws {
        if appliedDictionaries > count {
            dictionaries.removeAll()
            appliedDictionaries = 0
        }
        while appliedDictionaries < count {
            try applyDictionary(dictionaryMessages[appliedDictionaries], raw: raw)
            appliedDictionaries += 1
        }
    }

    private func applyDictionary(_ ref: ArrowIPCMessageRef, raw: UnsafeRawBufferPointer) throws {
        let meta = try FBBuf(raw, from: ref.metadataOffset, count: ref.metadataLength)
        guard let header = try meta.root().table(2) else {
            throw ArrowIPCError.malformed("dictionary batch has no header")
        }
        // DictionaryBatch { id: long; data: RecordBatch; isDelta: bool }
        let id = try header.int64(0)
        let isDelta = try header.bool(2)
        guard let data = try header.table(1) else {
            throw ArrowIPCError.malformed("dictionary batch \(id) has no record batch")
        }
        guard let field = schema.fields.first(where: { $0.dictionaryID == id }),
              let valueType = field.type.dictionaryValueType else {
            throw ArrowIPCError.malformed("dictionary batch \(id) matches no column")
        }
        // The file format indexes every dictionary in its footer, so a batch anywhere in the file may
        // use any of them and a replacement would be ambiguous; the spec forbids one.
        if !isDelta, dictionaries[id] != nil, format == .file {
            throw ArrowIPCError.malformed("dictionary id \(id) is defined twice in a file, which replaces nothing")
        }
        if isDelta, dictionaries[id] == nil {
            throw ArrowIPCError.malformed("delta dictionary batch \(id) has no dictionary to extend")
        }
        let valueField = ArrowIPCField(name: field.name + ".dictionary", type: valueType, nullable: true)
        let parts = try recordBatchParts(header: data, bodyLength: ref.bodyLength)
        let body = try messageBody(parts, raw: raw, bodyOffset: ref.bodyOffset, fields: [valueField])
        let cursor = ArrowIPCCursor(nodes: parts.nodes, body: body, variadicCounts: parts.variadicCounts)
        let values = try buildColumn(field: valueField, cursor: cursor)
        guard values.length == parts.length else {
            throw ArrowIPCError.malformed("dictionary batch \(id) declares \(parts.length) values but holds \(values.length)")
        }
        guard cursor.nodeIndex == parts.nodes.count, cursor.bufferIndex == parts.buffers.count,
              cursor.variadicIndex == parts.variadicCounts.count else {
            throw ArrowIPCError.malformed(
                "dictionary batch \(id) declares field nodes or buffers a \(valueType) dictionary does not use")
        }
        // A delta appends to the dictionary in force; anything else replaces it for the batches that follow.
        if isDelta, let old = dictionaries[id] {
            dictionaries[id] = try concatMetalArrays([old, values])
        } else {
            dictionaries[id] = values
        }
    }
}

// MARK: - Message body

/// One Arrow buffer's place in a message body.
struct ArrowIPCBodyBuffer {
    let offset: Int
    let length: Int
}

/// Where a message's Arrow buffers are: in the body itself, or — when the body declared a codec — in
/// one freshly decompressed shared-memory buffer per declared buffer.
struct ArrowIPCMessageBody {
    let raw: UnsafeRawBufferPointer
    let offset: Int
    let buffers: [ArrowIPCBodyBuffer]
    let plain: [MetalArrowBuffer]?
}

/// The position of a pre-order walk through one message's field nodes and Arrow buffers.
final class ArrowIPCCursor {
    let nodes: [(length: Int, nullCount: Int)]
    let body: ArrowIPCMessageBody
    let variadicCounts: [Int]
    private(set) var nodeIndex = 0
    private(set) var bufferIndex = 0
    private(set) var variadicIndex = 0

    init(nodes: [(length: Int, nullCount: Int)], body: ArrowIPCMessageBody, variadicCounts: [Int] = []) {
        self.nodes = nodes
        self.body = body
        self.variadicCounts = variadicCounts
    }

    /// How many variadic data buffers the next view field, `name`, has.
    func variadicCount(_ name: String) throws -> Int {
        guard variadicIndex < variadicCounts.count else {
            throw ArrowIPCError.malformed("the batch has no variadic buffer count for view column '\(name)'")
        }
        defer { variadicIndex += 1 }
        return variadicCounts[variadicIndex]
    }

    /// The next field node, which belongs to `name`.
    func node(_ name: String) throws -> (length: Int, nullCount: Int) {
        guard nodeIndex < nodes.count else {
            throw ArrowIPCError.malformed("the batch has no field node for column '\(name)'")
        }
        defer { nodeIndex += 1 }
        return nodes[nodeIndex]
    }

    /// The indices of the next `count` Arrow buffers, which belong to `name`.
    func buffers(_ count: Int, _ name: String) throws -> [Int] {
        guard bufferIndex + count <= body.buffers.count else {
            throw ArrowIPCError.malformed("the batch is missing buffers for column '\(name)'")
        }
        defer { bufferIndex += count }
        return Array(bufferIndex..<(bufferIndex + count))
    }
}

// MARK: - LZ4 frame

/// The LZ4 frame format, which is what Arrow's `LZ4_FRAME` body compression wraps its blocks in.
///
/// Every block decodes into one contiguous output, so a frame written with linked blocks (the LZ4
/// default, and what Arrow C++ writes) decodes as well as one with independent blocks: a match that
/// reaches back past the block boundary still finds bytes this decoder has already written.
enum ArrowIPCLZ4 {
    private static let magic: UInt32 = 0x184D_2204

    /// Decodes a whole frame into `dst`, returning how many bytes it produced.
    static func decodeFrame(_ src: UnsafePointer<UInt8>, _ srcLength: Int,
                            _ dst: UnsafeMutablePointer<UInt8>, _ dstLength: Int) throws -> Int {
        func word(_ at: Int) throws -> UInt32 {
            guard at + 4 <= srcLength else { throw ArrowIPCError.truncated("an LZ4 frame block header") }
            return UInt32(src[at]) | UInt32(src[at + 1]) << 8 | UInt32(src[at + 2]) << 16 | UInt32(src[at + 3]) << 24
        }
        guard srcLength >= 7, try word(0) == magic else {
            throw ArrowIPCError.malformed("a compressed buffer does not start with the LZ4 frame magic")
        }
        let flg = src[4], bd = src[5]
        _ = bd
        guard flg >> 6 == 1 else { throw ArrowIPCError.unsupported("LZ4 frame version \(flg >> 6)") }
        guard flg & 0x02 == 0 else { throw ArrowIPCError.malformed("an LZ4 frame with a reserved flag set") }
        var p = 6
        if flg & 0x08 != 0 { p += 8 }                       // content size
        if flg & 0x01 != 0 { p += 4 }                       // dictionary id
        p += 1                                              // header checksum
        let blockChecksum = flg & 0x10 != 0
        var out = 0
        while true {
            let header = try word(p)
            p += 4
            if header == 0 { break }                        // end mark
            let size = Int(header & 0x7FFF_FFFF)
            guard size >= 0, p + size <= srcLength else { throw ArrowIPCError.truncated("an LZ4 frame block") }
            if header & 0x8000_0000 != 0 {                  // stored uncompressed
                guard out + size <= dstLength else { throw ArrowIPCError.malformed("an LZ4 frame overruns its output") }
                if size > 0 { memcpy(dst + out, src + p, size) }
                out += size
            } else {
                out += try decodeBlock(src + p, size, dst, dstLength, out)
            }
            p += size
            if blockChecksum { p += 4 }
        }
        return out
    }

    /// One raw LZ4 block, appended to `dst` at `start`. Matches may reach back before `start`.
    private static func decodeBlock(_ src: UnsafePointer<UInt8>, _ n: Int, _ dst: UnsafeMutablePointer<UInt8>,
                                    _ capacity: Int, _ start: Int) throws -> Int {
        var i = 0, o = start
        func extend(_ base: Int) throws -> Int {
            var value = base
            while true {
                guard i < n else { throw ArrowIPCError.truncated("an LZ4 length") }
                let b = src[i]
                i += 1
                value += Int(b)
                if b != 255 { return value }
            }
        }
        while i < n {
            let token = src[i]
            i += 1
            var literals = Int(token >> 4)
            if literals == 15 { literals = try extend(literals) }
            guard i + literals <= n else { throw ArrowIPCError.truncated("LZ4 literals") }
            guard o + literals <= capacity else { throw ArrowIPCError.malformed("an LZ4 block overruns its output") }
            if literals > 0 { memcpy(dst + o, src + i, literals); i += literals; o += literals }
            // The last sequence of a block is literals only.
            if i == n { break }
            guard i + 2 <= n else { throw ArrowIPCError.truncated("an LZ4 match offset") }
            let offset = Int(src[i]) | Int(src[i + 1]) << 8
            i += 2
            guard offset > 0, o - offset >= 0 else { throw ArrowIPCError.malformed("an LZ4 match reaches before the output") }
            var match = Int(token & 0x0F)
            if match == 15 { match = try extend(match) }
            match += 4
            guard o + match <= capacity else { throw ArrowIPCError.malformed("an LZ4 match overruns its output") }
            // Byte at a time: a match whose offset is shorter than its length repeats a pattern.
            var from = o - offset
            for _ in 0..<match {
                dst[o] = dst[from]
                o += 1
                from += 1
            }
        }
        return o - start
    }
}
