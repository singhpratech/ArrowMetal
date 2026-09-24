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
        case .decimal(let a): return .decimal(precision: a.type.precision, scale: a.type.scale, bits: a.type.bitWidth)
        case .smallDecimal(let a): return .decimal(precision: a.type.precision, scale: a.type.scale, bits: a.type.bitWidth)
        case .list(let a):
            let item = ArrowIPCField(column: a.values, name: a.fieldName)
            if case .fixedSize(let n) = a.kind { return .fixedSizeList(item, size: n) }
            // `large_list` offsets are narrowed to int32 on import, so a list writes `list` unless an
            // explicit schema asks for `large_list`.
            return .list(item)
        case .structure(let a):
            return .structure(zip(a.names, a.children).map { ArrowIPCField(column: $0.1, name: $0.0) })
        case .map(let a):
            let s = a.entryStruct
            // Arrow names a map's child "entries", and requires it and its key to be non-nullable.
            let entries = ArrowIPCField(name: "entries", nullable: false, type: .structure([
                ArrowIPCField(column: s.children[0], name: s.names[0], nullable: false),
                ArrowIPCField(column: s.children[1], name: s.names[1]),
            ]))
            return .map(entries: entries, keysSorted: a.keysSorted)
        case .union(let a):
            return .union(mode: a.mode == .dense ? .dense : .sparse,
                          typeIDs: a.typeCodes.map(Int32.init),
                          children: zip(a.names, a.children).map { ArrowIPCField(column: $0.1, name: $0.0) })
        case .runEndEncoded(_, let values):
            return .runEndEncoded(runEnds: ArrowIPCField(name: "run_ends", nullable: false,
                                                         type: .int(bits: 32, signed: true)),
                                  values: ArrowIPCField(column: values, name: "values"))
        case .null: return .null
        case .float16: return .float16
        case .interval(let a): return .interval(a.unit.ipcUnit)
        case .fixedBinary(let a): return .fixedSizeBinary(byteWidth: a.byteWidth)
        // An extension type writes its storage type; the extension keys ride in the field's metadata.
        case .extended(let a): return a.storage.ipcType
        }
    }
}

extension ArrowIntervalUnit {
    /// The IPC metadata's `IntervalUnit`, which spells `interval[month]` "year_month".
    var ipcUnit: ArrowIPCIntervalUnit {
        switch self {
        case .months: return .yearMonth
        case .dayTime: return .dayTime
        case .monthDayNano: return .monthDayNano
        }
    }
}

extension ArrowIPCField {
    /// The field a column writes itself as: its own logical type, plus the `ARROW:extension:*` keys when
    /// it is an extension array (a consumer that knows the type rebuilds it, one that does not sees the
    /// storage type and the metadata).
    init(column: AnyMetalArray, name: String, nullable: Bool = true) {
        var metadata: [(key: String, value: [UInt8])] = []
        if case .extended(let e) = column {
            metadata = e.exportMetadata().pairs.map { (key: $0.key, value: $0.value) }
        }
        self.init(name: name, type: column.ipcType, nullable: nullable, dictionaryID: nil, metadata: metadata)
    }

    /// A field with no column behind it (a child Arrow names itself, or an explicit schema's own field).
    init(name: String, nullable: Bool, type: ArrowIPCType) {
        self.init(name: name, type: type, nullable: nullable)
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
/// Every type this package can hold is written, nested children recursively: decimals, `float16`,
/// `fixed_size_binary`, the three interval units, `null`, list / large list / fixed-size list, struct,
/// map, dense and sparse unions, run-end encoded columns, and extension types (whose `ARROW:extension:*`
/// keys travel in the field's metadata).
/// Pass an explicit `schema` to write a type a column's own storage does not name — `large_utf8`,
/// `large_binary`, `large_list`, or a temporal type over a plain integer column; the storage of each
/// column must match the physical layout of the type it is given.
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
                let field = ArrowIPCField(column: pair.1, name: pair.0)
                // Dictionary columns need an id to tie them to their DictionaryBatch; the column index does.
                if case .dictionary = field.type {
                    return ArrowIPCField(name: field.name, type: field.type, nullable: true,
                                         dictionaryID: Int64(i), metadata: field.metadata)
                }
                return field
            })
        }()
        // An explicit schema may name a dictionary column without giving it an id; the column index does.
        // A view type (a schema taken from a reader, say) is written as its classic counterpart.
        let resolved = ArrowIPCSchema(fields: schema.fields.map(\.classic).enumerated().map { i, field in
            if case .dictionary = field.type, field.dictionaryID == nil {
                return ArrowIPCField(name: field.name, type: field.type, nullable: field.nullable,
                                     dictionaryID: Int64(i), metadata: field.metadata)
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
        // Nested layouts match on shape, not on the names an explicit schema gives the children.
        switch column {
        case .list(let a):
            switch type {
            case .list(let f), .largeList(let f): return a.kind == .variable && compatible(a.values, f.type)
            case .fixedSizeList(let f, let n): return a.kind == .fixedSize(n) && compatible(a.values, f.type)
            default: return false
            }
        case .structure(let a):
            guard case .structure(let fields) = type, fields.count == a.children.count else { return false }
            return zip(a.children, fields).allSatisfy { compatible($0.0, $0.1.type) }
        case .map(let a):
            guard case .map(let entries, _) = type, case .structure(let fields) = entries.type,
                  fields.count == 2 else { return false }
            let s = a.entryStruct
            return compatible(s.children[0], fields[0].type) && compatible(s.children[1], fields[1].type)
        case .union(let a):
            guard case .union(let mode, let ids, let fields) = type, fields.count == a.children.count,
                  ids == a.typeCodes.map(Int32.init), (mode == .dense) == (a.mode == .dense) else { return false }
            return zip(a.children, fields).allSatisfy { compatible($0.0, $0.1.type) }
        case .runEndEncoded(_, let values):
            guard case .runEndEncoded(_, let v) = type else { return false }
            return compatible(values, v.type)
        case .extended(let a):
            return compatible(a.storage, type)
        default: break
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
        case .null(let x): return ObjectIdentifier(x)
        case .float16(let x): return ObjectIdentifier(x)
        case .smallDecimal(let x): return ObjectIdentifier(x)
        case .interval(let x): return ObjectIdentifier(x)
        case .fixedBinary(let x): return ObjectIdentifier(x)
        case .extended(let x): return ObjectIdentifier(x)
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
    ///
    /// Everything the table refers to — its name, its children, its type and its metadata — is written
    /// first: a FlatBuffers table may only point at bytes that are already in the buffer.
    private static func field(_ b: FBBuilder, _ f: ArrowIPCField) -> Int {
        let nameOffset = b.createString(f.name)
        // Children, depth first, so that a nested field's whole subtree precedes it.
        let childOffsets = f.type.children.map { field(b, $0) }
        let childrenVector = childOffsets.isEmpty ? 0 : b.createOffsetVector(childOffsets)
        // KeyValue { key: string; value: string }
        let metadataOffsets = f.metadata.map { pair -> Int in
            let key = b.createString(pair.key)
            let value = b.createString(bytes: pair.value)
            b.startObject(2)
            b.addOffset(id: 0, key)
            b.addOffset(id: 1, value)
            return b.endObject()
        }
        let metadataVector = metadataOffsets.isEmpty ? 0 : b.createOffsetVector(metadataOffsets)
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
        b.addOffset(id: 5, childrenVector)
        b.addOffset(id: 6, metadataVector)
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
        case .null: return empty(.null)
        case .float16:
            // FloatingPoint { precision: Precision }; HALF is the schema default, so it is omitted.
            b.startObject(1)
            b.addScalar(id: 0, fbPrecisionHalf, default: 0)
            return (.floatingPoint, b.endObject())
        case .decimal(let precision, let scale, let bits):
            // Decimal { precision: int; scale: int; bitWidth: int = 128 }
            b.startObject(3)
            b.addScalar(id: 0, Int32(precision), default: 0)
            b.addScalar(id: 1, Int32(scale), default: 0)
            b.addScalar(id: 2, Int32(bits), default: 128)
            return (.decimal, b.endObject())
        case .fixedSizeBinary(let width):
            b.startObject(1)
            b.addScalar(id: 0, Int32(width), default: 0)
            return (.fixedSizeBinary, b.endObject())
        case .interval(let unit):
            // Interval { unit: IntervalUnit }; YEAR_MONTH is the default.
            b.startObject(1)
            b.addScalar(id: 0, unit.rawValue, default: Int16(0))
            return (.interval, b.endObject())
        case .list: return empty(.list)
        case .largeList: return empty(.largeList)
        case .fixedSizeList(_, let size):
            b.startObject(1)
            b.addScalar(id: 0, Int32(size), default: 0)
            return (.fixedSizeList, b.endObject())
        case .structure: return empty(.structKind)
        case .map(_, let keysSorted):
            b.startObject(1)
            b.addScalar(id: 0, keysSorted, default: false)
            return (.map, b.endObject())
        case .union(let mode, let typeIDs, _):
            // Union { mode: UnionMode; typeIds: [int] }; the vector is written before the table.
            let ids = b.createInt32Vector(typeIDs)
            b.startObject(2)
            b.addScalar(id: 0, mode.rawValue, default: Int16(0))
            b.addOffset(id: 1, ids)
            return (.union, b.endObject())
        case .runEndEncoded: return empty(.runEndEncoded)
        // `resolve` has already replaced every view type with its classic counterpart.
        case .utf8View, .binaryView, .listView, .largeListView: return type(b, t.classic)
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

    /// One `FieldNode`: the logical length of a field and how many of its values are null.
    private typealias FieldNode = (length: Int64, nullCount: Int64)

    private static func recordBatchMessage(_ batch: MetalRecordBatch,
                                           schema: ArrowIPCSchema) throws -> (metadata: [UInt8], body: Data) {
        var body = Body()
        var nodes: [FieldNode] = []
        body.data.reserveCapacity(estimatedSize([batch]))

        for (i, column) in batch.columns.enumerated() {
            try appendColumn(&body, &nodes, column, type: schema.fields[i].type, name: schema.fields[i].name)
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

    /// Appends one column's field node and Arrow buffers to the message body, then its children, which is
    /// the pre-order Arrow IPC prescribes: a field's own node and buffers first, each child's subtree after.
    ///
    /// A dictionary column contributes its *codes*; the values travel in a dictionary batch.
    private static func appendColumn(_ body: inout Body, _ nodes: inout [FieldNode], _ column: AnyMetalArray,
                                     type: ArrowIPCType, name: String) throws {
        func push(_ length: Int, _ nullCount: Int) { nodes.append((Int64(length), Int64(nullCount))) }
        switch column {
        case .int8(let a): push(a.length, a.nullCount); append(&body, a, width: 1)
        case .uint8(let a): push(a.length, a.nullCount); append(&body, a, width: 1)
        case .int16(let a): push(a.length, a.nullCount); append(&body, a, width: 2)
        case .uint16(let a): push(a.length, a.nullCount); append(&body, a, width: 2)
        case .int32(let a): push(a.length, a.nullCount); append(&body, a, width: 4)
        case .uint32(let a): push(a.length, a.nullCount); append(&body, a, width: 4)
        case .int64(let a): push(a.length, a.nullCount); append(&body, a, width: 8)
        case .uint64(let a): push(a.length, a.nullCount); append(&body, a, width: 8)
        case .float32(let a): push(a.length, a.nullCount); append(&body, a, width: 4)
        case .float64(let a): push(a.length, a.nullCount); append(&body, a, width: 8)
        case .boolean(let a):
            push(a.length, a.nullCount)
            body.addValidity(a.validity, nullCount: a.nullCount, length: a.length)
            body.add(a.values.contents, byteCount: Bitmap.byteCount(bits: a.length))
        case .temporal(let t):
            push(t.length, t.nullCount)
            switch t.storage {
            case .int32(let a): append(&body, a, width: 4)
            case .int64(let a): append(&body, a, width: 8)
            }
        case .dictionary(let codes, _):
            push(codes.length, codes.nullCount)
            append(&body, codes, width: 4)
        case .decimal(let a):
            push(a.length, a.nullCount)
            appendFixedWidth(&body, validity: a.validity, nullCount: a.nullCount, length: a.length,
                             values: a.values, width: a.type.byteWidth)
        case .smallDecimal(let a):
            push(a.length, a.nullCount)
            appendFixedWidth(&body, validity: a.validity, nullCount: a.nullCount, length: a.length,
                             values: a.values, width: a.type.byteWidth)
        case .float16(let a):
            push(a.length, a.nullCount)
            append(&body, a.bits, width: 2)
        case .interval(let a):
            push(a.length, a.nullCount)
            appendFixedWidth(&body, validity: a.validity, nullCount: a.nullCount, length: a.length,
                             values: a.values, width: a.unit.byteWidth)
        case .fixedBinary(let a):
            push(a.length, a.nullCount)
            appendFixedWidth(&body, validity: a.validity, nullCount: a.nullCount, length: a.length,
                             values: a.values, width: a.byteWidth)
        case .null(let a):
            // The null type has no buffers at all, and every one of its values is null.
            push(a.length, a.length)
        case .string(let a), .binary(let a):
            push(a.length, a.nullCount)
            var large = false
            if case .varBinary(true) = type.storage { large = true }
            appendVarBinary(&body, a, large: large)
        case .list(let a):
            try appendList(&body, &nodes, a, type: type, name: name)
        case .map(let a):
            guard case .map(let entries, _) = type else {
                throw ArrowIPCError.malformed("column '\(name)' is a map but the schema says \(type)")
            }
            // A map is a list of entry structs; nothing else about the layout differs.
            try appendListBuffers(&body, &nodes, a.entries, child: entries, large: false, fixed: nil, name: name)
        case .structure(let a):
            guard case .structure(let fields) = type, fields.count == a.children.count else {
                throw ArrowIPCError.malformed("column '\(name)' is a struct of \(a.children.count) fields but the schema says \(type)")
            }
            push(a.length, a.nullCount)
            body.addValidity(a.validity, nullCount: a.nullCount, length: a.length)
            for (child, field) in zip(a.children, fields) {
                try appendColumn(&body, &nodes, try trimmed(child, to: a.length), type: field.type, name: field.name)
            }
        case .union(let a):
            guard case .union(let mode, _, let fields) = type, fields.count == a.children.count,
                  (mode == .dense) == (a.mode == .dense) else {
                throw ArrowIPCError.malformed("column '\(name)' is a \(a.mode) union of \(a.children.count) children but the schema says \(type)")
            }
            // A union has no validity bitmap: null-ness lives in the children.
            push(a.length, 0)
            body.add(a.typeIds.values.contents, byteCount: a.length)
            if let offsets = a.offsets { body.add(offsets.values.contents, byteCount: a.length * 4) }
            for (child, field) in zip(a.children, fields) {
                // A sparse union's children are as long as the union; a dense union's are their own length.
                let c = mode == .sparse ? try trimmed(child, to: a.length) : child
                try appendColumn(&body, &nodes, c, type: field.type, name: field.name)
            }
        case .runEndEncoded(let runEnds, let values):
            guard case .runEndEncoded(let endsField, let valuesField) = type else {
                throw ArrowIPCError.malformed("column '\(name)' is run-end encoded but the schema says \(type)")
            }
            guard values.length >= runEnds.length else {
                throw ArrowIPCError.malformed("run-end encoded column '\(name)' has \(runEnds.length) run ends for \(values.length) values")
            }
            // No buffers of its own: the logical length in the node, and everything else in the children.
            push(runEndLogicalLength(runEnds), 0)
            try appendColumn(&body, &nodes, .int32(runEnds), type: endsField.type, name: endsField.name)
            try appendColumn(&body, &nodes, try trimmed(values, to: runEnds.length), type: valuesField.type,
                             name: valuesField.name)
        case .extended(let a):
            // An extension column writes its storage; the extension keys are in the field's metadata.
            try appendColumn(&body, &nodes, a.storage, type: type, name: name)
        }
    }

    /// A child narrowed to the `length` its parent covers. Children may be longer than their parent —
    /// a sliced struct keeps whichever children it was built from — and IPC has no array offset.
    private static func trimmed(_ column: AnyMetalArray, to length: Int) throws -> AnyMetalArray {
        column.length == length ? column : try column.slice(offset: 0, length: length)
    }

    private static func appendFixedWidth(_ body: inout Body, validity: MetalArrowBuffer?, nullCount: Int,
                                         length: Int, values: MetalArrowBuffer, width: Int) {
        body.addValidity(validity, nullCount: nullCount, length: length)
        body.add(values.contents, byteCount: length * width)
    }

    /// A utf8 / binary column's offsets and bytes, rebased so that the first offset is zero: a sliced
    /// column writes only the bytes its own rows use, never the prefix it happens to share.
    private static func appendVarBinary(_ body: inout Body, _ a: MetalStringArray, large: Bool) {
        body.addValidity(a.validity, nullCount: a.nullCount, length: a.length)
        let source = a.offsets.typed(Int32.self)
        let base = Int(source[0]), end = Int(source[a.length])
        if large {
            // Widen the 32-bit offsets this package stores to the 64-bit ones large_* needs.
            var wide = [Int64](repeating: 0, count: a.length + 1)
            for j in 0...a.length { wide[j] = Int64(Int(source[j]) - base) }
            wide.withUnsafeBytes { body.add($0.baseAddress, byteCount: $0.count) }
        } else if base == 0 {
            body.add(a.offsets.contents, byteCount: (a.length + 1) * 4)
        } else {
            var rebased = [Int32](repeating: 0, count: a.length + 1)
            for j in 0...a.length { rebased[j] = source[j] - Int32(base) }
            rebased.withUnsafeBytes { body.add($0.baseAddress, byteCount: $0.count) }
        }
        body.add(end > base ? a.data.contents.advanced(by: base) : nil, byteCount: end - base)
    }

    private static func appendList(_ body: inout Body, _ nodes: inout [FieldNode], _ a: MetalListArray,
                                   type: ArrowIPCType, name: String) throws {
        switch type {
        case .list(let item):
            try appendListBuffers(&body, &nodes, a, child: item, large: false, fixed: nil, name: name)
        case .largeList(let item):
            try appendListBuffers(&body, &nodes, a, child: item, large: true, fixed: nil, name: name)
        case .fixedSizeList(let item, let size):
            try appendListBuffers(&body, &nodes, a, child: item, large: false, fixed: size, name: name)
        default:
            throw ArrowIPCError.malformed("column '\(name)' is a list but the schema says \(type)")
        }
    }

    /// The list layout: validity, offsets rebased to zero (none for a fixed-size list), then the child
    /// restricted to the range the rows actually cover.
    private static func appendListBuffers(_ body: inout Body, _ nodes: inout [FieldNode], _ a: MetalListArray,
                                          child: ArrowIPCField, large: Bool, fixed: Int?, name: String) throws {
        nodes.append((Int64(a.length), Int64(a.nullCount)))
        body.addValidity(a.validity, nullCount: a.nullCount, length: a.length)
        let source = a.offsets.typed(Int32.self)
        let base = Int(source[0]), end = Int(source[a.length])
        if let width = fixed {
            // A fixed_size_list has no offsets buffer: `offsets[i] == offsets[0] + i * N` is its invariant.
            guard end - base == a.length * width else {
                throw ArrowIPCError.malformed(
                    "fixed_size_list column '\(name)' covers \(end - base) child elements, not \(a.length * width)")
            }
        } else if large {
            var wide = [Int64](repeating: 0, count: a.length + 1)
            for j in 0...a.length { wide[j] = Int64(Int(source[j]) - base) }
            wide.withUnsafeBytes { body.add($0.baseAddress, byteCount: $0.count) }
        } else if base == 0 {
            body.add(a.offsets.contents, byteCount: (a.length + 1) * 4)
        } else {
            var rebased = [Int32](repeating: 0, count: a.length + 1)
            for j in 0...a.length { rebased[j] = source[j] - Int32(base) }
            rebased.withUnsafeBytes { body.add($0.baseAddress, byteCount: $0.count) }
        }
        let values = base == 0 && end == a.values.length ? a.values : try a.values.slice(offset: base, length: end - base)
        try appendColumn(&body, &nodes, values, type: child.type, name: child.name)
    }

    /// A `DictionaryBatch` message: one complete dictionary (`isDelta = false`) for id `id`.
    private static func dictionaryBatchMessage(_ values: AnyMetalArray, type: ArrowIPCType,
                                               id: Int64) throws -> (metadata: [UInt8], body: Data) {
        var body = Body()
        var nodes: [FieldNode] = []
        try appendColumn(&body, &nodes, values, type: type, name: "dictionary")
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
