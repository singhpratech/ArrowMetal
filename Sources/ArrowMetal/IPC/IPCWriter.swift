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
        case .binary: return .binary
        case .temporal(let t):
            func u(_ x: ArrowTemporalUnit) -> ArrowIPCTimeUnit {
                switch x { case .second: return .second; case .milli: return .millisecond; case .micro: return .microsecond; case .nano: return .nanosecond }
            }
            switch t.type {
            case .date32: return .date32
            case .date64: return .date64
            case .time32(let x): return .time32(u(x))
            case .time64(let x): return .time64(u(x))
            case .timestamp(let x, let tz): return .timestamp(u(x), timezone: tz)
            case .duration(let x): return .duration(u(x))
            }
        // A dictionary column writes int32 codes into the record batch and its values into a
        // DictionaryBatch message (see ArrowIPCWriter.encode).
        case .dictionary(_, let values): return .dictionary(index: .int(bits: 32, signed: true), value: values.ipcType)
        // Arrow IPC has no decimal type in this package yet. The value is a placeholder: `recordBatchMessage`
        // rejects decimal columns before it is ever written to a message.
        case .decimal: return .binary
        // Nested columns have no IPC type here: the writer rejects them before this value is used.
        case .list, .structure, .map, .union: return .binary
        // Run-end encoding has no IPC support here: the writer rejects such a column and asks for a decode.
        case .runEndEncoded(_, let values): return values.ipcType
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
/// Columns carry their own logical type: `.temporal` writes `date32` / `timestamp` / ..., `.binary`
/// writes `binary`, and `.dictionary` writes int32 codes plus one `DictionaryBatch` per column (a
/// complete dictionary, never a delta, and one per column for the whole file or stream).
/// Pass an explicit `schema` to write a type a column's own storage does not name — `large_utf8`,
/// `large_binary`, or a temporal type over a plain integer column; the storage of each column must
/// match the physical layout of the type it is given. Run-end encoded columns are refused: decode first.
public enum ArrowIPCWriter {

    /// Encodes `batches` as Arrow IPC bytes.
    public static func encode(_ batches: [MetalRecordBatch], schema: ArrowIPCSchema? = nil,
                              format: ArrowIPCFormat = .file) throws -> Data {
        let schema = try resolve(schema: schema, batches: batches)
        var out = Data()
        out.reserveCapacity(estimatedSize(batches) + 1024)
        if format == .file { out.append(contentsOf: arrowFileMagic) }

        appendMessage(&out, metadata: schemaMessage(schema), body: nil)
        // One complete dictionary per dictionary-encoded column, written before the record batches.
        // No deltas and no replacements: every batch must share one dictionary per column.
        var dictionaryBlocks: [(offset: Int, metadataLength: Int, bodyLength: Int)] = []
        for (i, field) in schema.fields.enumerated() {
            guard case .dictionary(_, let valueType) = field.type, let id = field.dictionaryID else { continue }
            guard let first = batches.first, case .dictionary(_, let values) = first.columns[i] else {
                if batches.isEmpty { continue }
                throw ArrowIPCError.malformed("column '\(field.name)' is \(field.type) but the batch is not dictionary encoded")
            }
            for (b, batch) in batches.enumerated().dropFirst() {
                guard case .dictionary(_, let other) = batch.columns[i], identity(other) == identity(values) else {
                    throw ArrowIPCError.unsupported(
                        "batch \(b) column '\(field.name)' carries a different dictionary; delta and replacement dictionaries are not written")
                }
            }
            let start = out.count
            let (metadata, body) = try dictionaryBatchMessage(values, type: valueType, id: id)
            let metadataLength = appendMessage(&out, metadata: metadata, body: body)
            dictionaryBlocks.append((start, metadataLength, body.count))
        }
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
            let footer = footerMessage(schema, dictionaries: dictionaryBlocks, blocks: blocks)
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
            return ArrowIPCSchema(fields: zip(first.names, first.columns).enumerated().map { i, pair in
                let type = pair.1.ipcType
                // Dictionary columns need an id to tie them to their DictionaryBatch; the column index does.
                if case .dictionary = type { return ArrowIPCField(name: pair.0, type: type, nullable: true, dictionaryID: Int64(i)) }
                return ArrowIPCField(name: pair.0, type: type, nullable: true)
            })
        }()
        // An explicit schema may name a dictionary column without giving it an id; the column index does.
        let resolved = ArrowIPCSchema(fields: schema.fields.enumerated().map { i, field in
            if case .dictionary = field.type, field.dictionaryID == nil {
                return ArrowIPCField(name: field.name, type: field.type, nullable: field.nullable, dictionaryID: Int64(i))
            }
            return field
        })
        for (b, batch) in batches.enumerated() {
            guard batch.columnCount == resolved.fields.count else {
                throw ArrowIPCError.malformed("batch \(b) has \(batch.columnCount) columns, the schema has \(resolved.fields.count)")
            }
            for (i, field) in resolved.fields.enumerated() {
                guard compatible(batch.columns[i], field.type) else {
                    throw ArrowIPCError.malformed(
                        "batch \(b) column '\(batch.names[i])' is \(batch.columns[i].ipcType) but the schema says \(field.type)")
                }
            }
        }
        return resolved
    }

    /// Whether a column's storage can carry values of `type`.
    private static func compatible(_ column: AnyMetalArray, _ type: ArrowIPCType) -> Bool {
        if column.ipcType == type { return true }
        if case .dictionary(_, let valueType) = type {
            guard case .dictionary(_, let values) = column else { return false }
            return compatible(values, valueType)
        }
        if case .varBinary = type.storage {
            switch column { case .string, .binary: return true; default: break }
        }
        return column.ipcType == type.physicalType || column.ipcType.physicalType == type.physicalType
    }

    /// The array object behind a column: two batches share a dictionary when these match.
    private static func identity(_ a: AnyMetalArray) -> ObjectIdentifier {
        switch a {
        case .int8(let x): return ObjectIdentifier(x)
        case .uint8(let x): return ObjectIdentifier(x)
        case .int16(let x): return ObjectIdentifier(x)
        case .uint16(let x): return ObjectIdentifier(x)
        case .int32(let x): return ObjectIdentifier(x)
        case .uint32(let x): return ObjectIdentifier(x)
        case .int64(let x): return ObjectIdentifier(x)
        case .uint64(let x): return ObjectIdentifier(x)
        case .float32(let x): return ObjectIdentifier(x)
        case .float64(let x): return ObjectIdentifier(x)
        case .boolean(let x): return ObjectIdentifier(x)
        case .string(let x), .binary(let x): return ObjectIdentifier(x)
        case .temporal(let x): return ObjectIdentifier(x)
        case .dictionary(let codes, _): return ObjectIdentifier(codes)
        case .runEndEncoded(let runEnds, _): return ObjectIdentifier(runEnds)
        case .decimal(let x): return ObjectIdentifier(x)
        case .list(let x): return ObjectIdentifier(x)
        case .structure(let x): return ObjectIdentifier(x)
        case .map(let x): return ObjectIdentifier(x)
        case .union(let x): return ObjectIdentifier(x)
        }
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
        // A dictionary field carries the *value* type in the union and the index type under `dictionary`.
        let (kind, typeOffset) = type(b, f.type)
        var encodingOffset = 0
        if case .dictionary(let index, _) = f.type {
            // indexType is written even for the int32 default: Arrow C++ rejects a null one.
            var bits = 32, signed = true
            if case .int(let b2, let s2) = index { bits = b2; signed = s2 }
            b.startObject(2)
            b.addScalar(id: 0, Int32(bits), default: 0)
            b.addScalar(id: 1, signed, default: false)
            let indexOffset = b.endObject()
            // DictionaryEncoding { id: long; indexType: Int; isOrdered: bool; dictionaryKind: short }
            b.startObject(4)
            b.addScalar(id: 0, f.dictionaryID ?? 0, default: 0)
            b.addOffset(id: 1, indexOffset)
            encodingOffset = b.endObject()
        }
        b.startObject(7)
        b.addOffset(id: 0, nameOffset)
        b.addScalar(id: 1, f.nullable, default: false)
        b.addScalar(id: 2, kind.rawValue, default: FBTypeKind.none.rawValue)
        b.addOffset(id: 3, typeOffset)
        b.addOffset(id: 4, encodingOffset)
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
        case .dictionary(_, let value):
            return type(b, value)
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
                                      dictionaries dictionaryBlocks: [(offset: Int, metadataLength: Int, bodyLength: Int)],
                                      blocks: [(offset: Int, metadataLength: Int, bodyLength: Int)]) -> [UInt8] {
        let b = FBBuilder()
        let schemaOffset = schemaTable(b, schema)
        // struct Block { offset: long; metaDataLength: int; <4 bytes padding> bodyLength: long; }
        func blockVector(_ list: [(offset: Int, metadataLength: Int, bodyLength: Int)]) -> Int {
            b.startVector(elementSize: fbBlockStride, count: list.count, alignment: 8)
            for block in list.reversed() {
                b.place(Int64(block.bodyLength))
                b.pad(4)
                b.place(Int32(block.metadataLength))
                b.place(Int64(block.offset))
            }
            return b.endVector(list.count)
        }
        let dictionaries = blockVector(dictionaryBlocks)
        let recordBatches = blockVector(blocks)
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
            nodes.append((Int64(column.length), Int64(column.nullCount)))
            try appendColumn(&body, column, type: schema.fields[i].type)
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

    /// Appends one column's Arrow buffers to the message body, in the order the type prescribes.
    /// A dictionary column contributes its *codes*; the values travel in a dictionary batch.
    private static func appendColumn(_ body: inout Body, _ column: AnyMetalArray, type: ArrowIPCType) throws {
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
        case .temporal(let t):
            switch t.storage {
            case .int32(let a): append(&body, a, width: 4)
            case .int64(let a): append(&body, a, width: 8)
            }
        case .dictionary(let codes, _):
            append(&body, codes, width: 4)
        case .runEndEncoded:
            throw ArrowIPCError.unsupported("run-end encoded columns are not written; call runEndDecode() on the column first")
        case .decimal(let a):
            throw ArrowIPCError.unsupported("decimal columns (\(a.type)) are not written to Arrow IPC yet; export them through the C Data Interface")
        case .list, .structure, .map, .union:
            throw ArrowIPCError.unsupported("nested columns (list, struct, map, union) are not written to IPC yet")
        case .string(let a), .binary(let a):
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

    /// A `DictionaryBatch` message: one complete dictionary (`isDelta = false`) for id `id`.
    private static func dictionaryBatchMessage(_ values: AnyMetalArray, type: ArrowIPCType,
                                               id: Int64) throws -> (metadata: [UInt8], body: Data) {
        var body = Body()
        try appendColumn(&body, values, type: type)
        let b = FBBuilder()
        b.startVector(elementSize: fbFieldNodeStride, count: 1, alignment: 8)
        b.place(Int64(values.nullCount))
        b.place(Int64(values.length))
        let nodesVector = b.endVector(1)
        b.startVector(elementSize: fbBufferStride, count: body.buffers.count, alignment: 8)
        for buffer in body.buffers.reversed() {
            b.place(buffer.length)
            b.place(buffer.offset)
        }
        let buffersVector = b.endVector(body.buffers.count)
        b.startObject(5)
        b.addScalar(id: 0, Int64(values.length), default: 0)
        b.addOffset(id: 1, nodesVector)
        b.addOffset(id: 2, buffersVector)
        let recordBatch = b.endObject()
        // DictionaryBatch { id: long; data: RecordBatch; isDelta: bool }
        b.startObject(3)
        b.addScalar(id: 0, id, default: 0)
        b.addOffset(id: 1, recordBatch)
        let header = b.endObject()
        return (message(b, header: header, kind: .dictionaryBatch, bodyLength: body.data.count), body.data)
    }

    private static func append<T: ArrowPrimitive>(_ body: inout Body, _ array: MetalArray<T>, width: Int) {
        body.addValidity(array.validity, nullCount: array.nullCount, length: array.length)
        body.add(array.values.contents, byteCount: array.length * width)
    }
}
