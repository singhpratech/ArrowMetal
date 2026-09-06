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

/// How an Arrow array of a given logical type is laid out in memory.
enum ArrowIPCStorage: Equatable {
    /// Fixed-width values: validity bitmap + values buffer.
    case fixedWidth(bytes: Int)
    /// Packed bits: validity bitmap + values bitmap.
    case bits
    /// Variable width: validity bitmap + offsets + data. `large` selects 64-bit offsets.
    case varBinary(large: Bool)

    /// Number of Arrow buffers a field of this storage contributes to a record batch.
    var bufferCount: Int {
        switch self {
        case .fixedWidth, .bits: return 2
        case .varBinary: return 3
        }
    }
}

/// The Arrow logical types ArrowMetal can read from and write to IPC.
///
/// Temporal types have no dedicated array class yet, so they are carried by their storage
/// integer array (`date32` as `Int32` days, `timestamp` as `Int64` ticks, ...). The logical
/// type is preserved in `ArrowIPCReader.schema` and written back out unchanged.
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
        }
    }

    /// The physical type whose array class carries values of this logical type.
    public var physicalType: ArrowIPCType {
        switch self {
        case .date32, .time32: return .int(bits: 32, signed: true)
        case .date64, .time64, .timestamp, .duration: return .int(bits: 64, signed: true)
        case .binary: return .utf8
        case .largeBinary: return .largeUtf8
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
        }
    }
}

/// One column of an Arrow IPC schema.
public struct ArrowIPCField: Equatable, Sendable {
    public let name: String
    public let type: ArrowIPCType
    public let nullable: Bool
    public init(name: String, type: ArrowIPCType, nullable: Bool = true) {
        self.name = name
        self.type = type
        self.nullable = nullable
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

    private let data: Data
    private let holder: ArrowIPCSourceHolder
    private let messages: [ArrowIPCMessageRef]
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
            return try buildBatch(header: header, raw: raw, bodyOffset: ref.bodyOffset, bodyLength: ref.bodyLength)
        }
    }

    // MARK: message index

    private static func hasPrefix(_ raw: UnsafeRawBufferPointer, _ magic: [UInt8], at start: Int) -> Bool {
        guard start >= 0, start + magic.count <= raw.count else { return false }
        for i in 0..<magic.count where raw[start + i] != magic[i] { return false }
        return true
    }

    private static func scan(_ raw: UnsafeRawBufferPointer)
        throws -> (format: ArrowIPCFormat, schema: ArrowIPCSchema, messages: [ArrowIPCMessageRef]) {
        guard raw.count >= 8 else { throw ArrowIPCError.notArrowIPC }
        if hasPrefix(raw, arrowFileMagic, at: 0) { return try scanFile(raw) }
        return try scanStream(raw)
    }

    private static func scanFile(_ raw: UnsafeRawBufferPointer)
        throws -> (format: ArrowIPCFormat, schema: ArrowIPCSchema, messages: [ArrowIPCMessageRef]) {
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
        if let dicts = try footer.vector(2), dicts.count > 0 {
            throw ArrowIPCError.unsupported("dictionary encoded columns")
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
                    throw msg.headerType == FBMessageHeader.dictionaryBatch.rawValue
                        ? ArrowIPCError.unsupported("dictionary batches")
                        : ArrowIPCError.malformed("block \(i) is not a record batch")
                }
                messages.append(msg.ref)
            }
        }
        return (.file, schema, messages)
    }

    private static func scanStream(_ raw: UnsafeRawBufferPointer)
        throws -> (format: ArrowIPCFormat, schema: ArrowIPCSchema, messages: [ArrowIPCMessageRef]) {
        var pos = 0
        var schema: ArrowIPCSchema? = nil
        var messages: [ArrowIPCMessageRef] = []
        while pos < raw.count {
            guard let m = try message(raw, at: pos) else { break }     // end-of-stream marker
            switch m.headerType {
            case FBMessageHeader.schema.rawValue:
                guard schema == nil else { throw ArrowIPCError.malformed("a second schema message in one stream") }
                let meta = try FBBuf(raw, from: m.ref.metadataOffset, count: m.ref.metadataLength)
                guard let table = try meta.root().table(2) else { throw ArrowIPCError.malformed("schema message has no header") }
                schema = try parseSchema(table)
            case FBMessageHeader.recordBatch.rawValue:
                guard schema != nil else { throw ArrowIPCError.malformed("record batch before the schema message") }
                messages.append(m.ref)
            case FBMessageHeader.dictionaryBatch.rawValue:
                throw ArrowIPCError.unsupported("dictionary batches")
            case FBMessageHeader.tensor.rawValue, FBMessageHeader.sparseTensor.rawValue:
                throw ArrowIPCError.unsupported("tensor messages")
            default:
                throw ArrowIPCError.malformed("unknown message header type \(m.headerType)")
            }
            pos = m.next
        }
        guard let schema else { throw ArrowIPCError.notArrowIPC }
        return (.stream, schema, messages)
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

    static func parseSchema(_ table: FBTable) throws -> ArrowIPCSchema {
        let endianness = try table.int16(0)
        guard endianness == 0 else { throw ArrowIPCError.unsupported("big-endian Arrow data") }
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
        if try field.field(4) != nil { throw ArrowIPCError.unsupported("dictionary encoded column '\(name)'") }
        if let children = try field.vector(5), children.count > 0 {
            throw ArrowIPCError.unsupported("nested column '\(name)'")
        }
        let kindCode = try field.uint8(2)
        guard let kind = FBTypeKind(rawValue: kindCode) else {
            throw ArrowIPCError.malformed("unknown type code \(kindCode) for column '\(name)'")
        }
        guard let type = try field.table(3) else {
            throw ArrowIPCError.malformed("column '\(name)' has no type table")
        }
        return ArrowIPCField(name: name, type: try parseType(kind, type, column: name), nullable: nullable)
    }

    private static func parseType(_ kind: FBTypeKind, _ type: FBTable, column: String) throws -> ArrowIPCType {
        func unit(_ raw: Int16) throws -> ArrowIPCTimeUnit {
            guard let u = ArrowIPCTimeUnit(rawValue: raw) else { throw ArrowIPCError.malformed("bad time unit \(raw)") }
            return u
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
            default: throw ArrowIPCError.unsupported("half precision floats (column '\(column)')")
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
        default:
            throw ArrowIPCError.unsupported("\(kind.name) columns (column '\(column)')")
        }
    }

    // MARK: record batch

    private struct BodyBuffer {
        let offset: Int
        let length: Int
    }

    private func buildBatch(header: FBTable, raw: UnsafeRawBufferPointer,
                            bodyOffset: Int, bodyLength: Int) throws -> MetalRecordBatch {
        if try header.field(3) != nil { throw ArrowIPCError.unsupported("compressed record batch bodies") }
        if let variadic = try header.vector(4), variadic.count > 0 {
            throw ArrowIPCError.unsupported("variadic buffers (view types)")
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
        var buffers: [BodyBuffer] = []
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
                buffers.append(BodyBuffer(offset: off, length: len))
            }
        }
        guard nodes.count == schema.fields.count else {
            throw ArrowIPCError.malformed("record batch has \(nodes.count) field nodes for \(schema.fields.count) columns")
        }

        lastBatchWasZeroCopy = false
        borrowedAny = false
        var columns: [AnyMetalArray] = []
        columns.reserveCapacity(schema.fields.count)
        var next = 0
        for (i, field) in schema.fields.enumerated() {
            let need = field.type.storage.bufferCount
            guard next + need <= buffers.count else {
                throw ArrowIPCError.malformed("record batch is missing buffers for column '\(field.name)'")
            }
            let slice = Array(buffers[next..<(next + need)])
            next += need
            let node = nodes[i]
            // Only nested children may have a length of their own, and those are rejected in the schema.
            guard node.length == length else {
                throw ArrowIPCError.malformed("column '\(field.name)' has \(node.length) values in a \(length) row batch")
            }
            columns.append(try buildColumn(field: field, length: length, nullCount: node.nullCount,
                                           buffers: slice, raw: raw, bodyOffset: bodyOffset))
        }
        lastBatchWasZeroCopy = borrowedAny
        return try MetalRecordBatch(names: schema.fields.map(\.name), columns: columns)
    }

    /// Copies (or borrows) `need` bytes of the body into Metal shared memory.
    private func take(_ buffer: BodyBuffer, need: Int, raw: UnsafeRawBufferPointer,
                      bodyOffset: Int) throws -> MetalArrowBuffer {
        guard need > 0 else { return try MetalArrowBuffer.allocate(byteCount: 0, context: context) }
        guard buffer.length >= need else {
            throw ArrowIPCError.malformed("buffer holds \(buffer.length) bytes where \(need) are needed")
        }
        guard let base = raw.baseAddress else { throw ArrowIPCError.truncated("empty source") }
        let src = base.advanced(by: bodyOffset + buffer.offset)
        if allowZeroCopy {
            let (buf, zeroCopy) = try MetalArrowBuffer.wrapOrCopy(src, byteCount: need, keepAlive: holder, context: context)
            if zeroCopy { borrowedAny = true }
            return buf
        }
        return try MetalArrowBuffer.copy(from: src, byteCount: need, context: context)
    }

    private func buildColumn(field: ArrowIPCField, length: Int, nullCount: Int, buffers: [BodyBuffer],
                             raw: UnsafeRawBufferPointer, bodyOffset: Int) throws -> AnyMetalArray {
        // null_count == -1 means "unknown": recompute it from the bitmap.
        let nulls = nullCount < 0 ? -1 : nullCount
        func validity() throws -> MetalArrowBuffer? {
            guard nulls != 0 else { return nil }
            let bytes = Bitmap.byteCount(bits: length)
            guard bytes > 0 else { return nil }
            guard buffers[0].length >= bytes else {
                if nulls > 0 { throw ArrowIPCError.malformed("column '\(field.name)' declares \(nulls) nulls but has no validity bitmap") }
                return nil
            }
            return try take(buffers[0], need: bytes, raw: raw, bodyOffset: bodyOffset)
        }

        switch field.type.storage {
        case .fixedWidth(let width):
            let bitmap = try validity()
            let values = try take(buffers[1], need: length * width, raw: raw, bodyOffset: bodyOffset)
            func make<T: ArrowPrimitive>(_: T.Type) -> MetalArray<T> {
                let a = MetalArray<T>(length: length, nullCount: 0, validity: bitmap, values: values, context: context)
                if nulls < 0 { a.recomputeNullCount() } else { a.nullCount = nulls }
                return a
            }
            switch field.type.physicalType {
            case .int(bits: 8, signed: true): return .int8(make(Int8.self))
            case .int(bits: 8, signed: false): return .uint8(make(UInt8.self))
            case .int(bits: 16, signed: true): return .int16(make(Int16.self))
            case .int(bits: 16, signed: false): return .uint16(make(UInt16.self))
            case .int(bits: 32, signed: true): return .int32(make(Int32.self))
            case .int(bits: 32, signed: false): return .uint32(make(UInt32.self))
            case .int(bits: 64, signed: true): return .int64(make(Int64.self))
            case .int(bits: 64, signed: false): return .uint64(make(UInt64.self))
            case .float(bits: 32): return .float32(make(Float.self))
            case .float(bits: 64): return .float64(make(Double.self))
            default: throw ArrowIPCError.unsupported("\(field.type) columns (column '\(field.name)')")
            }

        case .bits:
            let bitmap = try validity()
            let values = try take(buffers[1], need: Bitmap.byteCount(bits: length),
                                  raw: raw, bodyOffset: bodyOffset)
            let a = MetalBooleanArray(length: length, nullCount: 0, validity: bitmap, values: values, context: context)
            if nulls < 0 { a.recomputeNullCount() } else { a.nullCount = nulls }
            return .boolean(a)

        case .varBinary(let large):
            let bitmap = try validity()
            let offsets: MetalArrowBuffer
            let total: Int
            if large {
                // Narrow 64-bit offsets; MetalStringArray always stores 32-bit ones.
                guard let base = raw.baseAddress else { throw ArrowIPCError.truncated("empty source") }
                let need = (length + 1) * 8
                guard buffers[1].length >= need || length == 0 else {
                    throw ArrowIPCError.malformed("column '\(field.name)' has a short offsets buffer")
                }
                let out = try MetalArrowBuffer.allocate(byteCount: (length + 1) * 4, zeroed: true, context: context)
                if buffers[1].length >= need {
                    let src = base.advanced(by: bodyOffset + buffers[1].offset)
                    let dst = out.mutableTyped(Int32.self)
                    for i in 0...length {
                        let v = src.loadUnaligned(fromByteOffset: i * 8, as: Int64.self)
                        guard v >= 0, v <= Int64(Int32.max) else {
                            throw ArrowIPCError.unsupported("large binary column '\(field.name)' over 2 GB")
                        }
                        dst[i] = Int32(v)
                    }
                }
                offsets = out
                total = Int(out.typed(Int32.self)[length])
            } else {
                let need = (length + 1) * 4
                if buffers[1].length >= need {
                    offsets = try take(buffers[1], need: need, raw: raw, bodyOffset: bodyOffset)
                } else if length == 0 {
                    offsets = try MetalArrowBuffer.allocate(byteCount: need, zeroed: true, context: context)
                } else {
                    throw ArrowIPCError.malformed("column '\(field.name)' has a short offsets buffer")
                }
                total = Int(offsets.typed(Int32.self)[length])
            }
            guard total >= 0 else { throw ArrowIPCError.malformed("column '\(field.name)' has a negative final offset") }
            let bytes: MetalArrowBuffer = total == 0
                ? try MetalArrowBuffer.allocate(byteCount: 0, context: context)
                : try take(buffers[2], need: total, raw: raw, bodyOffset: bodyOffset)
            let a = MetalStringArray(length: length, nullCount: 0, validity: bitmap,
                                     offsets: offsets, data: bytes, context: context)
            if nulls < 0 { a.recomputeNullCount() } else { a.setNullCount(nulls) }
            return .string(a)
        }
    }
}
