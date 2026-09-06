import Foundation

// The subset of `parquet.thrift` this reader needs, decoded with `ThriftReader`. Field ids below are
// the ones in the format definition; anything not listed is skipped, which is what makes the decoder
// forward compatible with newer writers.

/// Physical types (`parquet.thrift` `Type`).
public enum ParquetPhysicalType: Int32, Sendable {
    case boolean = 0, int32 = 1, int64 = 2, int96 = 3, float = 4, double = 5
    case byteArray = 6, fixedLenByteArray = 7

    /// Fixed byte width, or nil for BYTE_ARRAY (and FIXED_LEN_BYTE_ARRAY, whose width is per column).
    var fixedWidth: Int? {
        switch self {
        case .boolean: return 1          // one *bit*, handled specially
        case .int32, .float: return 4
        case .int64, .double: return 8
        case .int96: return 12
        case .byteArray: return nil
        case .fixedLenByteArray: return nil
        }
    }
    var name: String {
        switch self {
        case .boolean: return "BOOLEAN"
        case .int32: return "INT32"
        case .int64: return "INT64"
        case .int96: return "INT96"
        case .float: return "FLOAT"
        case .double: return "DOUBLE"
        case .byteArray: return "BYTE_ARRAY"
        case .fixedLenByteArray: return "FIXED_LEN_BYTE_ARRAY"
        }
    }
}

public enum ParquetRepetition: Int32, Sendable { case required = 0, optional = 1, repeated = 2 }

/// `parquet.thrift` `Encoding`.
public enum ParquetEncoding: Int32, Sendable {
    case plain = 0, plainDictionary = 2, rle = 3, bitPacked = 4
    case deltaBinaryPacked = 5, deltaLengthByteArray = 6, deltaByteArray = 7
    case rleDictionary = 8, byteStreamSplit = 9

    public var name: String {
        switch self {
        case .plain: return "PLAIN"
        case .plainDictionary: return "PLAIN_DICTIONARY"
        case .rle: return "RLE"
        case .bitPacked: return "BIT_PACKED"
        case .deltaBinaryPacked: return "DELTA_BINARY_PACKED"
        case .deltaLengthByteArray: return "DELTA_LENGTH_BYTE_ARRAY"
        case .deltaByteArray: return "DELTA_BYTE_ARRAY"
        case .rleDictionary: return "RLE_DICTIONARY"
        case .byteStreamSplit: return "BYTE_STREAM_SPLIT"
        }
    }
    var isDictionary: Bool { self == .rleDictionary || self == .plainDictionary }
}

/// `parquet.thrift` `CompressionCodec`.
public enum ParquetCodec: Int32, Sendable {
    case uncompressed = 0, snappy = 1, gzip = 2, lzo = 3, brotli = 4, lz4 = 5, zstd = 6, lz4Raw = 7

    public var name: String {
        switch self {
        case .uncompressed: return "UNCOMPRESSED"
        case .snappy: return "SNAPPY"
        case .gzip: return "GZIP"
        case .lzo: return "LZO"
        case .brotli: return "BROTLI"
        case .lz4: return "LZ4"
        case .zstd: return "ZSTD"
        case .lz4Raw: return "LZ4_RAW"
        }
    }
    /// Codecs whose block decoder runs on the GPU.
    public var isGPUDecompressed: Bool {
        switch self {
        case .uncompressed, .snappy, .lz4, .lz4Raw: return true
        default: return false
        }
    }
}

/// `parquet.thrift` `ConvertedType` (the pre-2.4 logical type annotation).
enum ParquetConvertedType: Int32 {
    case utf8 = 0, map = 1, mapKeyValue = 2, list = 3, `enum` = 4, decimal = 5, date = 6
    case timeMillis = 7, timeMicros = 8, timestampMillis = 9, timestampMicros = 10
    case uint8 = 11, uint16 = 12, uint32 = 13, uint64 = 14
    case int8 = 15, int16 = 16, int32 = 17, int64 = 18
    case json = 19, bson = 20, interval = 21
}

/// The `LogicalType` union, in the shape this reader needs.
public enum ParquetLogicalType: Equatable, Sendable {
    case none
    case string
    case map
    case list
    case `enum`
    case decimal(precision: Int, scale: Int)
    case date
    case time(isUTC: Bool, unit: ParquetTimeUnit)
    case timestamp(isUTC: Bool, unit: ParquetTimeUnit)
    case integer(bitWidth: Int, signed: Bool)
    case unknown
    case json
    case bson
    case uuid
    case float16
}

public enum ParquetTimeUnit: Sendable { case millis, micros, nanos }

/// One node of the Parquet schema tree, flattened exactly as it appears in the footer.
public struct ParquetSchemaElement: Sendable {
    public var type: ParquetPhysicalType? = nil
    public var typeLength: Int = 0
    public var repetition: ParquetRepetition = .required
    public var name: String = ""
    public var numChildren: Int = 0
    var convertedType: ParquetConvertedType? = nil
    public var scale: Int = 0
    public var precision: Int = 0
    public var fieldID: Int32? = nil
    public var logicalType: ParquetLogicalType = .none

    static func read(_ r: inout ThriftReader) throws -> ParquetSchemaElement {
        var e = ParquetSchemaElement()
        try r.readStruct { r, id, t in
            switch id {
            case 1: e.type = ParquetPhysicalType(rawValue: try r.int32()); return true
            case 2: e.typeLength = try r.int(); return true
            case 3: e.repetition = ParquetRepetition(rawValue: try r.int32()) ?? .required; return true
            case 4: e.name = try r.string(); return true
            case 5: e.numChildren = try r.int(); return true
            case 6: e.convertedType = ParquetConvertedType(rawValue: try r.int32()); return true
            case 7: e.scale = try r.int(); return true
            case 8: e.precision = try r.int(); return true
            case 9: e.fieldID = try r.int32(); return true
            case 10: e.logicalType = try readLogicalType(&r); return true
            default: _ = t; return false
            }
        }
        // Fill the logical type in from the converted type when a writer only wrote the old annotation.
        if e.logicalType == .none, let c = e.convertedType { e.logicalType = ParquetSchemaElement.fromConverted(c, e) }
        return e
    }

    static func fromConverted(_ c: ParquetConvertedType, _ e: ParquetSchemaElement) -> ParquetLogicalType {
        switch c {
        case .utf8: return .string
        case .map, .mapKeyValue: return .map
        case .list: return .list
        case .enum: return .enum
        case .decimal: return .decimal(precision: e.precision, scale: e.scale)
        case .date: return .date
        case .timeMillis: return .time(isUTC: true, unit: .millis)
        case .timeMicros: return .time(isUTC: true, unit: .micros)
        case .timestampMillis: return .timestamp(isUTC: true, unit: .millis)
        case .timestampMicros: return .timestamp(isUTC: true, unit: .micros)
        case .uint8: return .integer(bitWidth: 8, signed: false)
        case .uint16: return .integer(bitWidth: 16, signed: false)
        case .uint32: return .integer(bitWidth: 32, signed: false)
        case .uint64: return .integer(bitWidth: 64, signed: false)
        case .int8: return .integer(bitWidth: 8, signed: true)
        case .int16: return .integer(bitWidth: 16, signed: true)
        case .int32: return .integer(bitWidth: 32, signed: true)
        case .int64: return .integer(bitWidth: 64, signed: true)
        case .json: return .json
        case .bson: return .bson
        case .interval: return .unknown
        }
    }

    /// The `LogicalType` union: one field is set and its id says which member.
    static func readLogicalType(_ r: inout ThriftReader) throws -> ParquetLogicalType {
        var out = ParquetLogicalType.none
        try r.readStruct { r, id, t in
            switch id {
            case 1: try r.skip(t); out = .string; return true
            case 2: try r.skip(t); out = .map; return true
            case 3: try r.skip(t); out = .list; return true
            case 4: try r.skip(t); out = .enum; return true
            case 5:
                var p = 0, s = 0
                try r.readStruct { r, fid, _ in
                    if fid == 1 { s = try r.int(); return true }
                    if fid == 2 { p = try r.int(); return true }
                    return false
                }
                out = .decimal(precision: p, scale: s); return true
            case 6: try r.skip(t); out = .date; return true
            case 7, 8:
                var utc = false
                var unit = ParquetTimeUnit.millis
                try r.readStruct { r, fid, ft in
                    if fid == 1 { utc = try r.bool(ft); return true }
                    if fid == 2 { unit = try readTimeUnit(&r); return true }
                    return false
                }
                out = id == 7 ? .time(isUTC: utc, unit: unit) : .timestamp(isUTC: utc, unit: unit)
                return true
            case 10:
                var bw = 32
                var signed = true
                try r.readStruct { r, fid, ft in
                    if fid == 1 { bw = Int(try r.byte()); return true }
                    if fid == 2 { signed = try r.bool(ft); return true }
                    return false
                }
                out = .integer(bitWidth: bw, signed: signed); return true
            case 11: try r.skip(t); out = .unknown; return true
            case 12: try r.skip(t); out = .json; return true
            case 13: try r.skip(t); out = .bson; return true
            case 14: try r.skip(t); out = .uuid; return true
            case 15: try r.skip(t); out = .float16; return true
            default: return false
            }
        }
        return out
    }

    static func readTimeUnit(_ r: inout ThriftReader) throws -> ParquetTimeUnit {
        var u = ParquetTimeUnit.millis
        try r.readStruct { r, id, t in
            switch id {
            case 1: try r.skip(t); u = .millis; return true
            case 2: try r.skip(t); u = .micros; return true
            case 3: try r.skip(t); u = .nanos; return true
            default: return false
            }
        }
        return u
    }
}

/// Column statistics, as raw bytes (they are encoded in the column's physical type).
public struct ParquetStatistics: Sendable {
    public var min: [UInt8]? = nil          // deprecated `min` (signed-comparison unsafe for strings)
    public var max: [UInt8]? = nil
    public var minValue: [UInt8]? = nil     // `min_value`, the correctly ordered one
    public var maxValue: [UInt8]? = nil
    public var nullCount: Int64? = nil
    public var distinctCount: Int64? = nil

    /// The best available lower/upper bound bytes.
    public var lower: [UInt8]? { minValue ?? min }
    public var upper: [UInt8]? { maxValue ?? max }

    static func read(_ r: inout ThriftReader) throws -> ParquetStatistics {
        var s = ParquetStatistics()
        try r.readStruct { r, id, t in
            switch id {
            case 1: s.max = try r.binary(); return true
            case 2: s.min = try r.binary(); return true
            case 3: s.nullCount = try r.int64(); return true
            case 4: s.distinctCount = try r.int64(); return true
            case 5: s.maxValue = try r.binary(); return true
            case 6: s.minValue = try r.binary(); return true
            default: _ = t; return false
            }
        }
        return s
    }
}

public struct ParquetColumnMetadata: Sendable {
    public var type: ParquetPhysicalType = .int32
    public var encodings: [ParquetEncoding] = []
    public var path: [String] = []
    public var codec: ParquetCodec = .uncompressed
    public var numValues: Int64 = 0
    public var totalUncompressedSize: Int64 = 0
    public var totalCompressedSize: Int64 = 0
    public var dataPageOffset: Int64 = 0
    public var indexPageOffset: Int64? = nil
    public var dictionaryPageOffset: Int64? = nil
    public var statistics: ParquetStatistics? = nil

    /// Byte offset where this chunk's pages start (the dictionary page comes first when present).
    public var startOffset: Int64 {
        if let d = dictionaryPageOffset, d > 0, d < dataPageOffset { return d }
        return dataPageOffset
    }

    static func read(_ r: inout ThriftReader) throws -> ParquetColumnMetadata {
        var m = ParquetColumnMetadata()
        try r.readStruct { r, id, t in
            switch id {
            case 1: m.type = ParquetPhysicalType(rawValue: try r.int32()) ?? .int32; return true
            case 2: m.encodings = (try r.readInt32List()).compactMap { ParquetEncoding(rawValue: $0) }; return true
            case 3: m.path = try r.readStringList(); return true
            case 4: m.codec = ParquetCodec(rawValue: try r.int32()) ?? .uncompressed; return true
            case 5: m.numValues = try r.int64(); return true
            case 6: m.totalUncompressedSize = try r.int64(); return true
            case 7: m.totalCompressedSize = try r.int64(); return true
            case 9: m.dataPageOffset = try r.int64(); return true
            case 10: m.indexPageOffset = try r.int64(); return true
            case 11: m.dictionaryPageOffset = try r.int64(); return true
            case 12: m.statistics = try ParquetStatistics.read(&r); return true
            default: _ = t; return false
            }
        }
        return m
    }
}

public struct ParquetColumnChunk: Sendable {
    public var filePath: String? = nil
    public var fileOffset: Int64 = 0
    public var meta = ParquetColumnMetadata()

    static func read(_ r: inout ThriftReader) throws -> ParquetColumnChunk {
        var c = ParquetColumnChunk()
        try r.readStruct { r, id, t in
            switch id {
            case 1: c.filePath = try r.string(); return true
            case 2: c.fileOffset = try r.int64(); return true
            case 3: c.meta = try ParquetColumnMetadata.read(&r); return true
            default: _ = t; return false
            }
        }
        return c
    }
}

public struct ParquetRowGroup: Sendable {
    public var columns: [ParquetColumnChunk] = []
    public var totalByteSize: Int64 = 0
    public var numRows: Int64 = 0
    public var fileOffset: Int64? = nil
    public var totalCompressedSize: Int64? = nil

    static func read(_ r: inout ThriftReader) throws -> ParquetRowGroup {
        var g = ParquetRowGroup()
        try r.readStruct { r, id, t in
            switch id {
            case 1:
                try r.readList { rr in g.columns.append(try ParquetColumnChunk.read(&rr)) }
                return true
            case 2: g.totalByteSize = try r.int64(); return true
            case 3: g.numRows = try r.int64(); return true
            case 5: g.fileOffset = try r.int64(); return true
            case 6: g.totalCompressedSize = try r.int64(); return true
            default: _ = t; return false
            }
        }
        return g
    }
}

public struct ParquetFileMetadata: Sendable {
    public var version: Int32 = 0
    public var schema: [ParquetSchemaElement] = []
    public var numRows: Int64 = 0
    public var rowGroups: [ParquetRowGroup] = []
    public var keyValueMetadata: [(String, String)] = []
    public var createdBy: String? = nil

    static func read(_ r: inout ThriftReader) throws -> ParquetFileMetadata {
        var m = ParquetFileMetadata()
        try r.readStruct { r, id, t in
            switch id {
            case 1: m.version = try r.int32(); return true
            case 2:
                try r.readList { rr in m.schema.append(try ParquetSchemaElement.read(&rr)) }
                return true
            case 3: m.numRows = try r.int64(); return true
            case 4:
                try r.readList { rr in m.rowGroups.append(try ParquetRowGroup.read(&rr)) }
                return true
            case 5:
                try r.readList { rr in
                    var k = "", v = ""
                    try rr.readStruct { r2, fid, _ in
                        if fid == 1 { k = try r2.string(); return true }
                        if fid == 2 { v = try r2.string(); return true }
                        return false
                    }
                    m.keyValueMetadata.append((k, v))
                }
                return true
            case 6: m.createdBy = try r.string(); return true
            default: _ = t; return false
            }
        }
        return m
    }
}

// MARK: - Page headers

public enum ParquetPageType: Int32, Sendable { case dataPage = 0, indexPage = 1, dictionaryPage = 2, dataPageV2 = 3 }

public struct ParquetPageHeader: Sendable {
    public var type: ParquetPageType = .dataPage
    public var uncompressedSize: Int32 = 0
    public var compressedSize: Int32 = 0
    // DataPageHeader (v1)
    public var numValues: Int32 = 0
    public var encoding: ParquetEncoding = .plain
    public var definitionLevelEncoding: ParquetEncoding = .rle
    public var repetitionLevelEncoding: ParquetEncoding = .rle
    // DataPageHeaderV2
    public var numNulls: Int32 = -1
    public var numRows: Int32 = -1
    public var defLevelsByteLength: Int32 = 0
    public var repLevelsByteLength: Int32 = 0
    public var isCompressed = true
    // DictionaryPageHeader
    public var dictNumValues: Int32 = 0
    public var dictEncoding: ParquetEncoding = .plain

    static func read(_ r: inout ThriftReader) throws -> ParquetPageHeader {
        var h = ParquetPageHeader()
        try r.readStruct { r, id, t in
            switch id {
            case 1: h.type = ParquetPageType(rawValue: try r.int32()) ?? .dataPage; return true
            case 2: h.uncompressedSize = try r.int32(); return true
            case 3: h.compressedSize = try r.int32(); return true
            case 5:
                try r.readStruct { r2, fid, ft in
                    switch fid {
                    case 1: h.numValues = try r2.int32(); return true
                    case 2: h.encoding = ParquetEncoding(rawValue: try r2.int32()) ?? .plain; return true
                    case 3: h.definitionLevelEncoding = ParquetEncoding(rawValue: try r2.int32()) ?? .rle; return true
                    case 4: h.repetitionLevelEncoding = ParquetEncoding(rawValue: try r2.int32()) ?? .rle; return true
                    default: _ = ft; return false
                    }
                }
                return true
            case 7:
                try r.readStruct { r2, fid, ft in
                    switch fid {
                    case 1: h.dictNumValues = try r2.int32(); return true
                    case 2: h.dictEncoding = ParquetEncoding(rawValue: try r2.int32()) ?? .plain; return true
                    default: _ = ft; return false
                    }
                }
                return true
            case 8:
                try r.readStruct { r2, fid, ft in
                    switch fid {
                    case 1: h.numValues = try r2.int32(); return true
                    case 2: h.numNulls = try r2.int32(); return true
                    case 3: h.numRows = try r2.int32(); return true
                    case 4: h.encoding = ParquetEncoding(rawValue: try r2.int32()) ?? .plain; return true
                    case 5: h.defLevelsByteLength = try r2.int32(); return true
                    case 6: h.repLevelsByteLength = try r2.int32(); return true
                    case 7: h.isCompressed = try r2.bool(ft); return true
                    default: return false
                    }
                }
                return true
            default: _ = t; return false
            }
        }
        return h
    }
}
