import Foundation
import Metal

// Reading a Parquet column, end to end, on the GPU.
//
// The pipeline for one leaf column, across every selected row group at once:
//
//   page headers (host, metadata only)
//     -> decompression                 GPU for SNAPPY / LZ4 / LZ4_RAW, host for ZSTD / GZIP / BROTLI,
//                                      nothing at all for UNCOMPRESSED (the mapped file is the buffer)
//     -> pq_page_layout                finds each page's level and value sections *inside* the page
//     -> pq_decode_levels              definition levels -> a byte per row, plus each row's rank in the
//                                      page's dense value section
//     -> pq_page_scan                  per-page dense offsets
//     -> one value decoder per encoding, writing dense values
//     -> pq_scatter                    dense values -> row positions (skipped when the column has no nulls)
//     -> pq_levels_to_bitmap           definition levels -> an Arrow validity bitmap
//
// Everything after "page headers" reads the column bytes only from the GPU.

/// Mirrors `PageInfo` in `ParquetDecodeSource`.
struct ParquetPageInfo {
    var dataOffset: UInt32 = 0
    var dataLength: UInt32 = 0
    var numValues: UInt32 = 0
    var levelOffset: UInt32 = 0
    var nonNullOffset: UInt32 = 0
    var nonNullCount: UInt32 = 0
    var repOffset: UInt32 = 0
    var repLength: UInt32 = 0
    var defOffset: UInt32 = 0
    var defLength: UInt32 = 0
    var valuesOffset: UInt32 = 0
    var valuesLength: UInt32 = 0
    var encoding: UInt32 = 0
    var bitWidth: UInt32 = 0
    var flags: UInt32 = 0
    var numRows: UInt32 = 0
    var dictBase: UInt32 = 0
    var pad0: UInt32 = 0
}

/// One page as the footer and its own header describe it, before anything has been decoded.
struct ParquetRawPage {
    var header: ParquetPageHeader
    /// Absolute file offset of the page body (just past its Thrift header).
    var bodyOffset: Int
    var rowGroup: Int
}

/// How a Parquet column should come back.
public struct ParquetReadOptions: Sendable {
    /// Columns to read, by name (dotted path also accepted). `nil` reads them all.
    public var columns: [String]? = nil
    /// Row groups to read, by index. `nil` reads them all.
    public var rowGroups: [Int]? = nil
    /// Keep dictionary-encoded columns dictionary-encoded instead of materialising their values.
    /// Ignored for a column whose pages are not all dictionary encoded.
    public var dictionaryEncoded = true
    /// Row-group filters evaluated against the footer's min/max statistics.
    public var filters: [ParquetFilter] = []

    public init(columns: [String]? = nil, rowGroups: [Int]? = nil,
                dictionaryEncoded: Bool = true, filters: [ParquetFilter] = []) {
        self.columns = columns
        self.rowGroups = rowGroups
        self.dictionaryEncoded = dictionaryEncoded
        self.filters = filters
    }
}

// MARK: - Reading

extension ParquetFile {
    /// Reads the selected columns and row groups into one Metal-resident record batch.
    public func read(_ options: ParquetReadOptions = ParquetReadOptions()) throws -> MetalRecordBatch {
        let groups = try selectedRowGroups(options)
        let wanted = try selectedFields(options.columns)
        var names: [String] = []
        var columns: [AnyMetalArray] = []
        // Wrap the bytes this read will touch as one Metal buffer up front. Column chunks are
        // interleaved by row group, so a single column's chunks span nearly the whole file in a
        // many-row-group file: wrapping per column would map the same pages once per column.
        try prewrap(fields: wanted, rowGroups: groups)
        // One open command buffer for the whole read: the decode is a chain of small kernels per column,
        // and a command buffer per kernel would spend more time on round trips than on the GPU. The
        // handful of places that must read a GPU result (a page scan total, an offsets total) flush and
        // reopen the batch through `MetalContext.syncPoint`.
        try context.batch {
            for f in wanted {
                names.append(f.name)
                columns.append(try readField(f, rowGroups: groups, options: options))
            }
        }
        if columns.isEmpty {
            // A projection of no columns still has a row count; expose it as an empty batch.
            return try MetalRecordBatch(names: [], columns: [])
        }
        return try MetalRecordBatch(names: names, columns: columns)
    }

    /// Maps the byte span of every column chunk this read will touch, in one `MTLBuffer`.
    private func prewrap(fields: [ParquetField], rowGroups: [Int]) throws {
        var lo = Int.max, hi = 0
        for f in fields {
            for leaf in f.leaves {
                for g in rowGroups {
                    let rg = metadata.rowGroups[g]
                    guard leaf.index < rg.columns.count else { continue }
                    let m = rg.columns[leaf.index].meta
                    lo = Swift.min(lo, Int(m.startOffset))
                    hi = Swift.max(hi, Int(m.startOffset) + Int(m.totalCompressedSize))
                }
            }
        }
        guard lo < hi else { return }
        _ = try buffer(covering: lo..<Swift.min(hi, fileSize))
    }

    /// Convenience: read named columns from every row group.
    public func read(columns: [String]?) throws -> MetalRecordBatch {
        try read(ParquetReadOptions(columns: columns))
    }

    /// Row groups that survive `options.rowGroups` and the statistics filters.
    public func selectedRowGroups(_ options: ParquetReadOptions) throws -> [Int] {
        var groups = options.rowGroups ?? Array(0..<metadata.rowGroups.count)
        for g in groups where g < 0 || g >= metadata.rowGroups.count {
            throw ParquetError.malformed("row group \(g) is outside 0..<\(metadata.rowGroups.count)")
        }
        if !options.filters.isEmpty {
            groups = groups.filter { g in options.filters.allSatisfy { $0.mayMatch(rowGroup: metadata.rowGroups[g], file: self) } }
        }
        return groups
    }

    /// Resolves projected names. A top-level field name selects that field; a dotted path selects one
    /// leaf on its own, which is how a struct's members are read (`"addr.city"`).
    func selectedFields(_ names: [String]?) throws -> [ParquetField] {
        guard let names else { return fields }
        return try names.map { n in
            if let f = fields.first(where: { $0.name == n }) { return f }
            if let leaf = leaves.first(where: { $0.dottedPath == n }) {
                return ParquetField(name: n, kind: .leaf(leaf), nullable: leaf.maxDefinition > 0)
            }
            throw ParquetError.malformed("no column named \(n)")
        }
    }

    /// Number of rows in the selected row groups.
    public func rowCount(_ options: ParquetReadOptions = ParquetReadOptions()) throws -> Int {
        try selectedRowGroups(options).reduce(0) { $0 + Int(metadata.rowGroups[$1].numRows) }
    }

    func readField(_ f: ParquetField, rowGroups: [Int], options: ParquetReadOptions) throws -> AnyMetalArray {
        switch f.kind {
        case .leaf(let l):
            let d = try decodeLeaf(l, rowGroups: rowGroups, options: options)
            return try d.arrowArray()
        case .list(let element, let repeatedDefinition):
            guard case .leaf(let l) = element.kind else {
                throw ParquetError.unsupported("list of non-primitive elements (column \(f.name))")
            }
            let d = try decodeLeaf(l, rowGroups: rowGroups, options: options, needRepetition: true)
            return try d.listArray(repeatedDefinition: repeatedDefinition, outerNullable: f.nullable)
        case .group:
            throw ParquetError.unsupported("struct column \(f.name); read its leaves by dotted path instead")
        }
    }
}

// MARK: - Filters

/// A row-group filter evaluated against the footer statistics: `column op literal`.
public struct ParquetFilter: Sendable {
    public enum Op: String, Sendable { case eq = "==", ne = "!=", lt = "<", le = "<=", gt = ">", ge = ">=" }
    public enum Value: Sendable {
        case int(Int64)
        case double(Double)
        case string(String)
    }
    public let column: String
    public let op: Op
    public let value: Value

    public init(column: String, op: Op, value: Value) {
        self.column = column
        self.op = op
        self.value = value
    }

    /// True when the row group *may* contain a matching row. Statistics are a conservative filter: a
    /// false answer means "definitely no match", a true answer means "look at the rows".
    public func mayMatch(rowGroup: ParquetRowGroup, file: ParquetFile) -> Bool {
        guard let leaf = file.leaves.first(where: { $0.dottedPath == column || $0.name == column }),
              leaf.index < rowGroup.columns.count else { return true }
        let meta = rowGroup.columns[leaf.index].meta
        guard let stats = meta.statistics, let lo = stats.lower, let hi = stats.upper,
              let low = decode(lo, leaf), let high = decode(hi, leaf) else { return true }
        // `ne` can only be excluded when the whole group is a single value equal to the literal.
        switch op {
        case .eq: return compare(low, high) <= 0 ? (compare(low, self.value) <= 0 && compare(self.value, high) <= 0) : true
        case .ne: return !(compare(low, self.value) == 0 && compare(high, self.value) == 0)
        case .lt: return compare(low, self.value) < 0
        case .le: return compare(low, self.value) <= 0
        case .gt: return compare(high, self.value) > 0
        case .ge: return compare(high, self.value) >= 0
        }
    }

    /// Decodes a statistics blob in the column's physical type into a comparable value.
    private func decode(_ bytes: [UInt8], _ leaf: ParquetLeaf) -> Value? {
        switch leaf.physical {
        case .int32:
            guard bytes.count >= 4 else { return nil }
            var v: Int32 = 0
            withUnsafeMutableBytes(of: &v) { $0.copyBytes(from: bytes[0..<4]) }
            if case .integer(_, let signed) = leaf.logicalType, !signed { return .int(Int64(UInt32(bitPattern: v))) }
            return .int(Int64(v))
        case .int64:
            guard bytes.count >= 8 else { return nil }
            var v: Int64 = 0
            withUnsafeMutableBytes(of: &v) { $0.copyBytes(from: bytes[0..<8]) }
            return .int(v)
        case .float:
            guard bytes.count >= 4 else { return nil }
            var v: Float = 0
            withUnsafeMutableBytes(of: &v) { $0.copyBytes(from: bytes[0..<4]) }
            return .double(Double(v))
        case .double:
            guard bytes.count >= 8 else { return nil }
            var v: Double = 0
            withUnsafeMutableBytes(of: &v) { $0.copyBytes(from: bytes[0..<8]) }
            return .double(v)
        case .byteArray, .fixedLenByteArray:
            return .string(String(decoding: bytes, as: UTF8.self))
        case .boolean:
            return bytes.first.map { .int(Int64($0 == 0 ? 0 : 1)) }
        case .int96:
            return nil
        }
    }

    private func compare(_ a: Value, _ b: Value) -> Int {
        switch (a, b) {
        case (.int(let x), .int(let y)): return x < y ? -1 : (x == y ? 0 : 1)
        case (.int(let x), .double(let y)): return Double(x) < y ? -1 : (Double(x) == y ? 0 : 1)
        case (.double(let x), .int(let y)): return x < Double(y) ? -1 : (x == Double(y) ? 0 : 1)
        case (.double(let x), .double(let y)): return x < y ? -1 : (x == y ? 0 : 1)
        case (.string(let x), .string(let y)): return x < y ? -1 : (x == y ? 0 : 1)
        default: return 0
        }
    }
}
