import Foundation

extension AnyMetalArray {
    /// The Arrow logical type this column is written as unless an explicit schema overrides it.
    public var ipcType: ArrowIPCType {
        switch self {
        case .int8: return .int(bits: 8, signed: true)
        case .uint8: return .int(bits: 8, signed: false)
        case .int16: return .int(bits: 16, signed: true)
        case .uint16: return .int(bits: 16, signed: false)
        case .int32: return .int(bits: 32, signed: true)
        case .uint32: return .int(bits: 32, signed: false)
        case .int64: return .int(bits: 64, signed: true)
        case .uint64: return .int(bits: 64, signed: false)
        case .float32: return .float(bits: 32)
        case .float64: return .float(bits: 64)
        case .boolean: return .bool
        case .string: return .utf8
        }
    }
}

/// Writes Metal-resident record batches as Arrow IPC, in either the streaming or the file format.
///
/// ```swift
/// try ArrowIPCWriter.write(batches, to: url)               // .arrow file, readable by pyarrow
/// let stream = try ArrowIPCWriter.encode(batches, format: .stream)
/// ```
///
/// Pass an explicit `schema` to write logical types that have no dedicated array class yet
/// (`date32`, `timestamp`, `duration`, `binary`, `large_utf8`, ...); the storage of each column
/// must match the physical layout of the type it is given.
public enum ArrowIPCWriter {

    /// Encodes `batches` as Arrow IPC bytes.
    public static func encode(_ batches: [MetalRecordBatch], schema: ArrowIPCSchema? = nil,
                              format: ArrowIPCFormat = .file) throws -> Data {
        let schema = try resolve(schema: schema, batches: batches)
        var out = Data()
        out.reserveCapacity(estimatedSize(batches) + 1024)
        if format == .file { out.append(contentsOf: arrowFileMagic) }

        appendMessage(&out, metadata: schemaMessage(schema), body: nil)
        var blocks: [(offset: Int, metadataLength: Int, bodyLength: Int)] = []
        for batch in batches {
            let start = out.count
            let (metadata, body) = try recordBatchMessage(batch, schema: schema)
            let metadataLength = appendMessage(&out, metadata: metadata, body: body)
            blocks.append((start, metadataLength, body.count))
        }
        // End-of-stream marker: a continuation with a zero metadata length.
        out.append(contentsOf: littleEndian(arrowContinuation))
        out.append(contentsOf: littleEndian(Int32(0)))

        if format == .file {
            let footer = footerMessage(schema, blocks: blocks)
            out.append(contentsOf: footer)
            out.append(contentsOf: littleEndian(Int32(footer.count)))
            out.append(contentsOf: arrowFileMagicTail)
        }
        return out
    }

    /// Writes `batches` to `url` (the file format by default, so readers can seek).
    public static func write(_ batches: [MetalRecordBatch], to url: URL, schema: ArrowIPCSchema? = nil,
                             format: ArrowIPCFormat = .file) throws {
        let data = try encode(batches, schema: schema, format: format)
        try data.write(to: url, options: .atomic)
    }

    // MARK: schema

    /// Derives the schema from the first batch, or validates an explicit one against every batch.
    private static func resolve(schema: ArrowIPCSchema?, batches: [MetalRecordBatch]) throws -> ArrowIPCSchema {
        let schema: ArrowIPCSchema = try {
            if let schema { return schema }
            guard let first = batches.first else {
                throw ArrowIPCError.malformed("cannot derive a schema from an empty batch list; pass one explicitly")
            }
            return ArrowIPCSchema(fields: zip(first.names, first.columns).map {
                ArrowIPCField(name: $0, type: $1.ipcType, nullable: true)
            })
        }()
        for (b, batch) in batches.enumerated() {
            guard batch.columnCount == schema.fields.count else {
                throw ArrowIPCError.malformed("batch \(b) has \(batch.columnCount) columns, the schema has \(schema.fields.count)")
            }
            for (i, field) in schema.fields.enumerated() {
                guard compatible(batch.columns[i], field.type) else {
                    throw ArrowIPCError.malformed(
                        "batch \(b) column '\(batch.names[i])' is \(batch.columns[i].ipcType) but the schema says \(field.type)")
                }
            }
        }
        return schema
    }

    /// Whether a column's storage can carry values of `type`.
    private static func compatible(_ column: AnyMetalArray, _ type: ArrowIPCType) -> Bool {
        if case .varBinary = type.storage, case .string = column { return true }
        return column.ipcType == type.physicalType
    }

    private static func estimatedSize(_ batches: [MetalRecordBatch]) -> Int {
        var n = 0
        for b in batches {
            for c in b.columns {
                switch c {
                case .string(let s): n += s.totalBytes + (s.length + 1) * 4
                case .boolean(let a): n += Bitmap.byteCount(bits: a.length)
                default: n += c.length * 8
                }
                n += Bitmap.byteCount(bits: c.length) + 64
            }
        }
        return n
    }

    // MARK: encapsulation

    private static func littleEndian<T: FixedWidthInteger>(_ v: T) -> [UInt8] {
        withUnsafeBytes(of: v.littleEndian) { Array($0) }
    }

    /// Appends `continuation | metadata length | metadata | body` and returns the total metadata length
    /// (prefix included), which is what a file `Block` records.
    @discardableResult
    private static func appendMessage(_ out: inout Data, metadata: [UInt8], body: Data?) -> Int {
        let prefix = 8
        let padded = roundUp(metadata.count + prefix, to: 8)
        out.append(contentsOf: littleEndian(arrowContinuation))
        out.append(contentsOf: littleEndian(Int32(padded - prefix)))
        out.append(contentsOf: metadata)
        let padding = padded - prefix - metadata.count
        if padding > 0 { out.append(contentsOf: [UInt8](repeating: 0, count: padding)) }
        if let body { out.append(body) }
        return padded
    }

    // MARK: metadata tables

    /// Writes one `Field` table and returns its offset.
    private static func field(_ b: FBBuilder, _ f: ArrowIPCField) -> Int {
        let nameOffset = b.createString(f.name)
        let (kind, typeOffset) = type(b, f.type)
        b.startObject(7)
        b.addOffset(id: 0, nameOffset)
        b.addScalar(id: 1, f.nullable, default: false)
        b.addScalar(id: 2, kind.rawValue, default: FBTypeKind.none.rawValue)
        b.addOffset(id: 3, typeOffset)
        return b.endObject()
    }

    /// Writes the `Type` union member for `t` and returns its discriminant and offset.
    private static func type(_ b: FBBuilder, _ t: ArrowIPCType) -> (FBTypeKind, Int) {
        func empty(_ kind: FBTypeKind) -> (FBTypeKind, Int) {
            b.startObject(0)
            return (kind, b.endObject())
        }
        switch t {
        case .int(let bits, let signed):
            b.startObject(2)
            b.addScalar(id: 0, Int32(bits), default: 0)
            b.addScalar(id: 1, signed, default: false)
            return (.int, b.endObject())
        case .float(let bits):
            b.startObject(1)
            b.addScalar(id: 0, bits == 64 ? fbPrecisionDouble : fbPrecisionSingle, default: 0)
            return (.floatingPoint, b.endObject())
        case .bool: return empty(.bool)
        case .utf8: return empty(.utf8)
        case .largeUtf8: return empty(.largeUtf8)
        case .binary: return empty(.binary)
        case .largeBinary: return empty(.largeBinary)
        case .date32, .date64:
            b.startObject(1)
            // DateUnit: DAY = 0, MILLISECOND = 1 (the schema default).
            b.addScalar(id: 0, t == .date32 ? fbDateUnitDay : fbDateUnitMillisecond, default: fbDateUnitMillisecond)
            return (.date, b.endObject())
        case .time32(let u), .time64(let u):
            let bits: Int32 = { if case .time64 = t { return 64 } else { return 32 } }()
            b.startObject(2)
            b.addScalar(id: 0, u.rawValue, default: fbDateUnitMillisecond)
            b.addScalar(id: 1, bits, default: 32)
            return (.time, b.endObject())
        case .timestamp(let u, let tz):
            let tzOffset = tz.map { b.createString($0) } ?? 0
            b.startObject(2)
            b.addScalar(id: 0, u.rawValue, default: Int16(0))
            b.addOffset(id: 1, tzOffset)
            return (.timestamp, b.endObject())
        case .duration(let u):
            b.startObject(1)
            b.addScalar(id: 0, u.rawValue, default: fbDateUnitMillisecond)
            return (.duration, b.endObject())
        }
    }

    /// Writes a `Schema` table and returns its offset.
    private static func schemaTable(_ b: FBBuilder, _ schema: ArrowIPCSchema) -> Int {
        let fieldOffsets = schema.fields.map { field(b, $0) }
        let fieldsVector = b.createOffsetVector(fieldOffsets)
        b.startObject(4)
        b.addScalar(id: 0, Int16(0), default: 0)       // endianness: Little
        b.addOffset(id: 1, fieldsVector)
        return b.endObject()
    }

    /// A complete `Message` flatbuffer wrapping `header`.
    private static func message(_ b: FBBuilder, header: Int, kind: FBMessageHeader, bodyLength: Int) -> [UInt8] {
        b.startObject(5)
        b.addScalar(id: 0, fbMetadataVersionV5, default: 0)
        b.addScalar(id: 1, kind.rawValue, default: FBMessageHeader.none.rawValue)
        b.addOffset(id: 2, header)
        b.addScalar(id: 3, Int64(bodyLength), default: 0)
        return b.finish(b.endObject())
    }

    private static func schemaMessage(_ schema: ArrowIPCSchema) -> [UInt8] {
        let b = FBBuilder()
        return message(b, header: schemaTable(b, schema), kind: .schema, bodyLength: 0)
    }

    private static func footerMessage(_ schema: ArrowIPCSchema,
                                      blocks: [(offset: Int, metadataLength: Int, bodyLength: Int)]) -> [UInt8] {
        let b = FBBuilder()
        let schemaOffset = schemaTable(b, schema)
        // struct Block { offset: long; metaDataLength: int; <4 bytes padding> bodyLength: long; }
        b.startVector(elementSize: fbBlockStride, count: 0, alignment: 8)
        let dictionaries = b.endVector(0)
        b.startVector(elementSize: fbBlockStride, count: blocks.count, alignment: 8)
        for block in blocks.reversed() {
            b.place(Int64(block.bodyLength))
            b.pad(4)
            b.place(Int32(block.metadataLength))
            b.place(Int64(block.offset))
        }
        let recordBatches = b.endVector(blocks.count)
        b.startObject(5)
        b.addScalar(id: 0, fbMetadataVersionV5, default: 0)
        b.addOffset(id: 1, schemaOffset)
        b.addOffset(id: 2, dictionaries)
        b.addOffset(id: 3, recordBatches)
        return b.finish(b.endObject())
    }

    // MARK: record batch

    /// Accumulates the message body: every Arrow buffer padded to an 8-byte boundary.
    private struct Body {
        var data = Data()
        var buffers: [(offset: Int64, length: Int64)] = []

        mutating func add(_ pointer: UnsafeRawPointer?, byteCount: Int) {
            let offset = data.count
            if byteCount > 0, let pointer {
                data.append(pointer.assumingMemoryBound(to: UInt8.self), count: byteCount)
            }
            let padded = roundUp(byteCount, to: 8)
            if padded > byteCount { data.append(contentsOf: [UInt8](repeating: 0, count: padded - byteCount)) }
            buffers.append((Int64(offset), Int64(padded)))
        }

        /// The validity bitmap, or an empty buffer when the column has no nulls (as Arrow expects).
        mutating func addValidity(_ validity: MetalArrowBuffer?, nullCount: Int, length: Int) {
            guard nullCount > 0, let validity else { return add(nil, byteCount: 0) }
            add(validity.contents, byteCount: Bitmap.byteCount(bits: length))
        }
    }

    private static func recordBatchMessage(_ batch: MetalRecordBatch,
                                           schema: ArrowIPCSchema) throws -> (metadata: [UInt8], body: Data) {
        var body = Body()
        var nodes: [(length: Int64, nullCount: Int64)] = []
        body.data.reserveCapacity(estimatedSize([batch]))

        for (i, column) in batch.columns.enumerated() {
            let type = schema.fields[i].type
            nodes.append((Int64(column.length), Int64(column.nullCount)))
            switch column {
            case .int8(let a): append(&body, a, width: 1)
            case .uint8(let a): append(&body, a, width: 1)
            case .int16(let a): append(&body, a, width: 2)
            case .uint16(let a): append(&body, a, width: 2)
            case .int32(let a): append(&body, a, width: 4)
            case .uint32(let a): append(&body, a, width: 4)
            case .int64(let a): append(&body, a, width: 8)
            case .uint64(let a): append(&body, a, width: 8)
            case .float32(let a): append(&body, a, width: 4)
            case .float64(let a): append(&body, a, width: 8)
            case .boolean(let a):
                body.addValidity(a.validity, nullCount: a.nullCount, length: a.length)
                body.add(a.values.contents, byteCount: Bitmap.byteCount(bits: a.length))
            case .string(let a):
                body.addValidity(a.validity, nullCount: a.nullCount, length: a.length)
                if case .varBinary(true) = type.storage {
                    // Widen the 32-bit offsets this package stores to the 64-bit ones large_* needs.
                    let source = a.offsets.typed(Int32.self)
                    var wide = [Int64](repeating: 0, count: a.length + 1)
                    for j in 0...a.length { wide[j] = Int64(source[j]) }
                    wide.withUnsafeBytes { body.add($0.baseAddress, byteCount: $0.count) }
                } else {
                    body.add(a.offsets.contents, byteCount: (a.length + 1) * 4)
                }
                body.add(a.data.contents, byteCount: a.totalBytes)
            }
        }

        let b = FBBuilder()
        b.startVector(elementSize: fbFieldNodeStride, count: nodes.count, alignment: 8)
        for node in nodes.reversed() {
            b.place(node.nullCount)
            b.place(node.length)
        }
        let nodesVector = b.endVector(nodes.count)
        b.startVector(elementSize: fbBufferStride, count: body.buffers.count, alignment: 8)
        for buffer in body.buffers.reversed() {
            b.place(buffer.length)
            b.place(buffer.offset)
        }
        let buffersVector = b.endVector(body.buffers.count)
        b.startObject(5)
        b.addScalar(id: 0, Int64(batch.length), default: 0)
        b.addOffset(id: 1, nodesVector)
        b.addOffset(id: 2, buffersVector)
        let header = b.endObject()
        return (message(b, header: header, kind: .recordBatch, bodyLength: body.data.count), body.data)
    }

    private static func append<T: ArrowPrimitive>(_ body: inout Body, _ array: MetalArray<T>, width: Int) {
        body.addValidity(array.validity, nullCount: array.nullCount, length: array.length)
        body.add(array.values.contents, byteCount: array.length * width)
    }
}
