import Foundation

// The Arrow schema a writer stores beside the Parquet one.
//
// Parquet's type system has no time zone (only `isAdjustedToUTC`), no duration, no extension types and no
// per-field metadata, so Arrow writers (pyarrow, Arrow C++, Polars, arrow-rs) put the original Arrow schema
// in the file's key/value metadata under `ARROW:schema`: the base64 of an encapsulated IPC `Schema`
// message. Arrow's own reader uses it to put back what the Parquet schema lost, and so does this one:
//
//   - a timestamp's time zone, where Parquet only says `isAdjustedToUTC`;
//   - `duration[unit]`, which Parquet stores as a plain INT64;
//   - extension types (`ARROW:extension:name` / `:metadata` in the field's custom metadata), which come
//     back wrapped as `MetalExtensionArray` so the consumer rebuilds them when it knows the type;
//   - every field's custom metadata, served by `arrowFieldMetadata(column:)`.
//
// The `null` type needs none of this: Parquet annotates a null column with the `UNKNOWN` logical type,
// and `ParquetLeafData.arrowArray` reads that as `null`. Absent or malformed metadata is ignored: the file
// reads exactly as the Parquet schema alone describes it. The IPC FlatBuffers reader's table and vector
// views (`IPC/FlatBuffers.swift`) do the decoding; every offset they follow is bounds-checked.

/// One field of the Arrow schema stored in `ARROW:schema`, reduced to what the Parquet reader restores.
public struct ParquetArrowField: Sendable {
    public enum Kind: Sendable, Equatable {
        case timestamp(ArrowTemporalUnit, timezone: String?)
        case duration(ArrowTemporalUnit)
        case null
        case structure
        case list
        case largeList
        case fixedSizeList(Int)
        case map
        case decimal(precision: Int, scale: Int, bitWidth: Int)
        /// Dictionary-encoded, with the value type's kind.
        indirect case dictionary(Kind)
        /// Any type the reader does not need to know more about; the raw `Type` union code.
        case other(UInt8)
    }
    public let name: String
    public let nullable: Bool
    public let kind: Kind
    public let children: [ParquetArrowField]
    /// The field's `custom_metadata`, in order.
    public let metadata: ArrowSchemaMetadata

    /// `ARROW:extension:name`, when the field is an extension type.
    public var extensionName: String? { metadata.string(ArrowSchemaMetadata.extensionNameKey) }
}

extension ParquetFile {
    /// The key under which Arrow writers store their schema.
    public static let arrowSchemaKey = "ARROW:schema"

    /// The fields of the stored Arrow schema, or nil when the file has none or it does not decode.
    public var arrowSchema: [ParquetArrowField]? { cachedArrowSchema }

    /// Decodes `ARROW:schema`; nil when absent or malformed (never an error: the metadata is advisory).
    static func decodeArrowSchema(_ kv: [(String, String)]) -> [ParquetArrowField]? {
        guard let text = kv.first(where: { $0.0 == arrowSchemaKey })?.1,
              let data = Data(base64Encoded: text), data.count >= 8 else {
            return nil
        }
        return data.withUnsafeBytes { raw -> [ParquetArrowField]? in
            try? parseArrowSchemaMessage(raw)
        }
    }

    private static func parseArrowSchemaMessage(_ raw: UnsafeRawBufferPointer) throws -> [ParquetArrowField] {
        // Encapsulated message: [0xFFFFFFFF] <int32 metadata length> <flatbuffer Message>. The continuation
        // marker is absent in files written before Arrow 0.15.
        var p = 0
        if raw.loadUnaligned(fromByteOffset: 0, as: UInt32.self) == arrowContinuation { p = 4 }
        guard p + 4 <= raw.count else { throw ArrowIPCError.truncated("ARROW:schema length") }
        let length = Int(raw.loadUnaligned(fromByteOffset: p, as: Int32.self))
        p += 4
        guard length > 0, p + length <= raw.count else { throw ArrowIPCError.truncated("ARROW:schema message") }
        let buf = try FBBuf(raw, from: p, count: length)
        let message = try buf.root()
        // Message { version: short; header_type: ubyte; header: MessageHeader; ... }
        guard try message.uint8(1) == FBMessageHeader.schema.rawValue, let schema = try message.table(2) else {
            throw ArrowIPCError.malformed("ARROW:schema is not a Schema message")
        }
        guard try schema.int16(0) == 0 else { throw ArrowIPCError.unsupported("big-endian ARROW:schema") }
        var fields: [ParquetArrowField] = []
        if let vec = try schema.vector(1) {
            for i in 0..<vec.count { fields.append(try parseArrowField(vec.table(i), depth: 0)) }
        }
        return fields
    }

    private static func parseArrowField(_ t: FBTable, depth: Int) throws -> ParquetArrowField {
        guard depth < 64 else { throw ArrowIPCError.malformed("ARROW:schema nests more than 64 levels") }
        // Field { name; nullable; type_type; type; dictionary; children; custom_metadata }
        let name = try t.string(0) ?? ""
        let nullable = try t.bool(1)
        let code = try t.uint8(2)
        var kind = ParquetArrowField.Kind.other(code)
        if let k = FBTypeKind(rawValue: code) {
            switch k {
            case .timestamp:
                if let type = try t.table(3) {
                    let tz = try type.string(1)
                    kind = .timestamp(unit(try type.int16(0)), timezone: (tz?.isEmpty ?? true) ? nil : tz)
                }
            case .duration:
                if let type = try t.table(3) { kind = .duration(unit(try type.int16(0, default: 1))) }
            case .null: kind = .null
            case .structKind: kind = .structure
            case .list: kind = .list
            case .largeList: kind = .largeList
            case .fixedSizeList:
                if let type = try t.table(3) { kind = .fixedSizeList(Int(try type.int32(0))) }
            case .decimal:
                // Decimal { precision: int; scale: int; bitWidth: int = 128 }
                if let type = try t.table(3) {
                    kind = .decimal(precision: Int(try type.int32(0)), scale: Int(try type.int32(1)),
                                    bitWidth: Int(try type.int32(2, default: 128)))
                }
            case .map: kind = .map
            default: break
            }
        }
        if try t.table(4) != nil { kind = .dictionary(kind) }
        var children: [ParquetArrowField] = []
        if let vec = try t.vector(5) {
            for i in 0..<vec.count { children.append(try parseArrowField(vec.table(i), depth: depth + 1)) }
        }
        var metadata = ArrowSchemaMetadata()
        if let vec = try t.vector(6) {
            for i in 0..<vec.count {
                // KeyValue { key: string; value: string }; the value may be any bytes.
                let kv = try vec.table(i)
                let key = try kv.string(0) ?? ""
                metadata.pairs.append(.init(key: key, value: try rawString(kv, 1)))
            }
        }
        return ParquetArrowField(name: name, nullable: nullable, kind: kind, children: children, metadata: metadata)
    }

    private static func unit(_ raw: Int16) -> ArrowTemporalUnit {
        switch raw {
        case 0: return .second
        case 1: return .milli
        case 2: return .micro
        default: return .nano
        }
    }

    /// A FlatBuffers string field as raw bytes.
    private static func rawString(_ t: FBTable, _ id: Int) throws -> [UInt8] {
        guard let p = try t.field(id) else { return [] }
        let s = try t.buf.indirect(p)
        let n = Int(try t.buf.load(UInt32.self, at: s))
        guard s + 4 + n <= t.buf.bytes.count else { throw ArrowIPCError.truncated("metadata value of \(n) bytes") }
        return Array(UnsafeRawBufferPointer(rebasing: t.buf.bytes[(s + 4)..<(s + 4 + n)]))
    }

    // MARK: - Applying it

    /// The stored Arrow field for top-level field `f`: by position when the two schemas line up, else by name.
    func arrowField(for f: ParquetField) -> ParquetArrowField? {
        guard let stored = arrowSchema else { return nil }
        if stored.count == fields.count, let i = fields.firstIndex(where: { $0.name == f.name }),
           stored[i].name == f.name {
            return stored[i]
        }
        return stored.first { $0.name == f.name }
    }

    /// Puts back what the Parquet schema could not say, where the stored field agrees with the array.
    func applyArrowField(_ a: AnyMetalArray, _ stored: ParquetArrowField) throws -> AnyMetalArray {
        // A dictionary-typed field's logical type is its value type's.
        var kind = stored.kind
        if case .dictionary(let valueKind) = kind { kind = valueKind }
        // A column read dictionary-encoded carries the logical type on its dictionary.
        if case .dictionary(let codes, let values) = a {
            let valueField = ParquetArrowField(name: stored.name, nullable: stored.nullable, kind: kind,
                                               children: stored.children, metadata: ArrowSchemaMetadata())
            let out = AnyMetalArray.dictionary(codes: codes, values: try applyArrowField(values, valueField))
            guard let ext = stored.extensionName else { return out }
            return .extended(MetalExtensionArray(storage: out, name: ext,
                                                 metadata: stored.metadata[ArrowSchemaMetadata.extensionMetadataKey]))
        }
        var out = a
        switch (a, kind) {
        case (.temporal(let t), .timestamp(_, let tz?)):
            // Arrow's rule: only a UTC-adjusted column takes the stored zone, and it keeps its own unit.
            if case .timestamp(let unit, let current) = t.type, current == "UTC", case .int64(let v) = t.storage {
                out = .temporal(try MetalTemporalArray(type: .timestamp(unit, timezone: tz), v))
            }
        case (.int64(let v), .duration(let unit)):
            out = .temporal(try MetalTemporalArray(type: .duration(unit), v))
        case (.decimal(let d), .decimal(let p, let s, let bits)) where bits == 32 || bits == 64:
            // decimal32 / decimal64 are stored as INT32 / INT64 decimals and read as decimal128.
            out = .smallDecimal(try d.narrowed(to: try ArrowSmallDecimalType(precision: p, scale: s, bitWidth: bits)))
        case (.list(let l), .fixedSizeList(let width)) where stored.children.count == 1:
            let values = try applyArrowField(l.values, stored.children[0])
            out = .list(try fixedSizeList(l, values: values, width: width) ?? l)
        case (.structure(let s), .structure) where s.children.count == stored.children.count:
            let kids = try zip(s.children, stored.children).map { try applyArrowField($0, $1) }
            out = .structure(try MetalStructArray(length: s.length, nullCount: s.nullCount, validity: s.validity,
                                                  names: s.names, children: kids, context: s.context))
        case (.list(let l), .list), (.list(let l), .largeList):
            if stored.children.count == 1 {
                out = .list(MetalListArray(length: l.length, nullCount: l.nullCount, validity: l.validity,
                                           offsets: l.offsets, values: try applyArrowField(l.values, stored.children[0]),
                                           kind: l.kind, fieldName: l.fieldName, context: l.context))
            }
        case (.map(let m), .map):
            if stored.children.count == 1, case .structure(let e) = m.entries.values,
               stored.children[0].children.count == 2 {
                let kids = try zip(e.children, stored.children[0].children).map { try applyArrowField($0, $1) }
                let entries = try MetalStructArray(length: e.length, nullCount: e.nullCount, validity: e.validity,
                                                   names: e.names, children: kids, context: e.context)
                let l = m.entries
                out = .map(try MetalMapArray(
                    entries: MetalListArray(length: l.length, nullCount: l.nullCount, validity: l.validity,
                                            offsets: l.offsets, values: .structure(entries), kind: l.kind,
                                            fieldName: l.fieldName, context: l.context),
                    keysSorted: m.keysSorted))
            }
        default:
            break
        }
        if case .dictionary = stored.kind {
            // The column's Arrow type is a dictionary (a pandas categorical, say): encode it, as Arrow's
            // reader does, whether or not its pages were dictionary encoded.
            out = try out.dictionaryEncoded()
        }
        if let ext = stored.extensionName {
            out = .extended(MetalExtensionArray(storage: out, name: ext,
                                                metadata: stored.metadata[ArrowSchemaMetadata.extensionMetadataKey]))
        }
        return out
    }

    /// A list whose valid rows all hold `width` elements as a `fixed_size_list<width>`: null rows get
    /// `width` null child slots, as the fixed-size layout requires. Nil when a row has another length.
    private func fixedSizeList(_ l: MetalListArray, values: AnyMetalArray, width: Int) throws -> MetalListArray? {
        guard width >= 0 else { return nil }
        try context.syncPoint()
        let off = l.offsets.typed(Int32.self)
        var index: [Int32?] = []
        index.reserveCapacity(l.length * width)
        for i in 0..<l.length {
            if l.isValid(i) {
                guard Int(off[i + 1] - off[i]) == width else { return nil }
                for k in 0..<width { index.append(off[i] + Int32(k)) }
            } else {
                index.append(contentsOf: repeatElement(nil, count: width))
            }
        }
        let child = try values.take(try MetalArray<Int32>(index, context: context))
        let offsets = try MetalArrowBuffer.allocate(byteCount: (l.length + 1) * 4, zeroed: false, context: context)
        let p = offsets.mutableTyped(Int32.self)
        for i in 0...l.length { p[i] = Int32(i * width) }
        return MetalListArray(length: l.length, nullCount: l.nullCount, validity: l.validity, offsets: offsets,
                              values: child, kind: .fixedSize(width), fieldName: l.fieldName, context: context)
    }

    /// The custom metadata of a top-level column as `pyarrow.parquet.read_table` reports it: the stored
    /// Arrow field's metadata, plus `PARQUET:field_id` when the Parquet schema gives the field an id.
    /// Empty when there is none, and for a dotted leaf path.
    public func arrowFieldMetadata(column: String) -> ArrowSchemaMetadata {
        guard let f = fields.first(where: { $0.name == column }) else { return ArrowSchemaMetadata() }
        var m = arrowField(for: f)?.metadata ?? ArrowSchemaMetadata()
        if let id = f.fieldID { m["PARQUET:field_id"] = Array(String(id).utf8) }
        return m
    }

    /// The file's key/value metadata without `ARROW:schema`, which is what Arrow reports as the schema's
    /// own metadata.
    public var arrowSchemaMetadata: ArrowSchemaMetadata {
        ArrowSchemaMetadata(metadata.keyValueMetadata.filter { $0.0 != Self.arrowSchemaKey }
                                .map { .init(key: $0.0, string: $0.1) })
    }
}
