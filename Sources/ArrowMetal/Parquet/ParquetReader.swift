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
        ParquetProfile.start()
        let afterStatistics = try selectedRowGroups(options)
        // An equality filter whose value the row group's bloom filter has never seen rules it out.
        let afterBloom = useBloomFilters && options.filters.contains(where: { $0.op == .eq })
            ? afterStatistics.filter { bloomFiltersMayMatch(options.filters, rowGroup: $0) } : afterStatistics
        // The page index narrows each row group to candidate row ranges; a row group left with none is
        // not read at all.
        let ranges = try candidateRowRanges(options, rowGroups: afterBloom)
        let groups = afterBloom.filter { ranges[$0].map { !$0.isEmpty } ?? true }
        let plan = ParquetReadPlan(ranges: ranges.filter { !$0.value.isEmpty })
        let wanted = try selectedFields(options.columns)
        var names: [String] = []
        var columns: [AnyMetalArray] = []
        // Wrap the bytes this read will touch as one Metal buffer up front. Column chunks are
        // interleaved by row group, so a single column's chunks span nearly the whole file in a
        // many-row-group file: wrapping per column would map the same pages once per column.
        ParquetProfile.lap("read.plan")
        try prewrap(fields: wanted, rowGroups: groups)
        // One open command buffer for the whole read: the decode is a chain of small kernels per column,
        // and a command buffer per kernel would spend more time on round trips than on the GPU. The
        // handful of places that must read a GPU result (a page scan total, an offsets total) flush and
        // reopen the batch through `MetalContext.syncPoint`.
        try context.batch {
            for f in wanted {
                names.append(f.name)
                var column = try readField(f, rowGroups: groups, options: options, plan: plan)
                // A top-level column takes back what `ARROW:schema` says the Parquet schema lost; a leaf
                // selected on its own by dotted path reads as the Parquet schema describes it. The stored
                // schema is advisory: a claim the column cannot take leaves it as the Parquet schema says.
                if let stored = arrowField(for: f), let restored = try? applyArrowField(column, stored) {
                    column = restored
                }
                columns.append(column)
            }
            ParquetProfile.lap("read.flush", sync: context)
        }
        ParquetProfile.lap("read.flush", sync: context)
        var stats = ParquetReadStatistics()
        stats.rowGroupsRead = groups.count
        stats.rowGroupsSkippedByStatistics = (options.rowGroups?.count ?? metadata.rowGroups.count) - afterStatistics.count
        stats.rowGroupsSkippedByBloomFilter = afterStatistics.count - afterBloom.count
        stats.rowGroupsSkippedByPageIndex = afterBloom.count - groups.count
        stats.pagesDecoded = plan.pagesDecoded
        stats.pagesSkipped = plan.pagesSkipped
        stats.rows = columns.first?.length ?? 0
        recordReadStatistics(stats)
        if columns.isEmpty {
            // A projection of no columns still has a row count; expose it as an empty batch.
            return try MetalRecordBatch(names: [], columns: [])
        }
        let result = try MetalRecordBatch(names: names, columns: columns)
        ParquetProfile.lap("read.batch")
        ParquetProfile.report("read \(names.count) columns")
        return result
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
            // A row group the statistics rule out is dropped, unless its page index shows those statistics
            // leave pages out (`rowGroupStatisticsTrusted`); that check runs only on a drop.
            groups = groups.filter { g in
                options.filters.allSatisfy {
                    $0.mayMatch(rowGroup: metadata.rowGroups[g], file: self) || !rowGroupStatisticsTrusted(g, column: $0.column)
                }
            }
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

    func readField(_ f: ParquetField, rowGroups: [Int], options: ParquetReadOptions,
                   plan: ParquetReadPlan? = nil) throws -> AnyMetalArray {
        // Whole row groups, for the columns that are not trimmed page by page.
        let whole = rowGroups.map { (group: $0, rows: 0..<rowsIn(group: $0)) }
        let trimming = plan.map { !$0.ranges.isEmpty } ?? false
        switch f.kind {
        case .leaf(let l):
            let d = try decodeLeaf(l, rowGroups: rowGroups, options: options, plan: plan, subset: true)
            let a = try d.arrowArray()
            // A leaf below a list, read on its own by dotted path, has one entry per element rather than
            // per row, so there are no rows to trim it to.
            return trimming && l.maxRepetition == 0 ? try trim(a, spans: d.rowSpans, plan: plan!) : a
        case .list(let element, let repeatedDefinition):
            // A one-level `list<primitive>` keeps its dedicated kernel pair; every other list, map and
            // struct goes through the general assembler.
            let a: AnyMetalArray
            if !f.isMap, case .leaf(let l) = element.kind, l.maxRepetition == 1 {
                let d = try decodeLeaf(l, rowGroups: rowGroups, options: options, needRepetition: true, plan: plan)
                a = try d.listArray(repeatedDefinition: repeatedDefinition, outerNullable: f.nullable)
            } else {
                a = try ParquetNestedAssembler(file: self, rowGroups: rowGroups, options: options, plan: plan).buildTopLevel(f)
            }
            return trimming ? try trim(a, spans: whole, plan: plan!) : a
        case .group:
            let a = try ParquetNestedAssembler(file: self, rowGroups: rowGroups, options: options, plan: plan).buildTopLevel(f)
            return trimming ? try trim(a, spans: whole, plan: plan!) : a
        }
    }
}

// MARK: - Filters

/// A row-group filter evaluated against the footer statistics: `column op literal`.
public struct ParquetFilter: Sendable {
    public enum Op: String, Sendable { case eq = "==", ne = "!=", lt = "<", le = "<=", gt = ">", ge = ">=" }
    public enum Value: Sendable {
        case int(Int64)
        /// An integer above `Int64.max`, for an unsigned 64-bit column.
        case uint(UInt64)
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
        guard let stats = meta.statistics, let lo = stats.lower, let hi = stats.upper else { return true }
        return mayMatch(lower: lo, upper: hi, leaf: leaf)
    }

    /// True when values between `lower` and `upper` (statistics bytes in the leaf's physical type) *may*
    /// satisfy the filter. Used for a row group's statistics and for one page's column-index entry.
    func mayMatch(lower lo: [UInt8], upper hi: [UInt8], leaf: ParquetLeaf) -> Bool {
        // Writers leave NaN out of min / max, so a FLOAT or DOUBLE range whose bounds both equal the
        // literal can still hold a NaN, and NaN satisfies `!=`: on those columns `!=` rules nothing out.
        if op == .ne, leaf.physical == .float || leaf.physical == .double { return true }
        // A decimal's statistics are its unscaled integer (or its big-endian bytes): not comparable with
        // a literal written in the column's units, so look at the rows.
        if case .decimal = leaf.logicalType { return true }
        if leaf.physical == .byteArray || leaf.physical == .fixedLenByteArray {
            return mayMatchBytes(lower: lo, upper: hi, leaf: leaf)
        }
        guard let low = decode(lo, leaf), let high = decode(hi, leaf) else { return true }
        // A NaN bound says nothing (the format asks readers to ignore it), and a literal of another kind
        // than the column's (a string against a number) cannot be ordered against it: look at the rows.
        guard !Self.isNaN(low), !Self.isNaN(high), !Self.isNaN(self.value),
              Self.comparable(low, self.value), Self.comparable(high, self.value) else { return true }
        // `ne` can only be excluded when the whole range is a single value equal to the literal.
        switch op {
        case .eq: return compare(low, high) <= 0 ? (compare(low, self.value) <= 0 && compare(self.value, high) <= 0) : true
        case .ne: return !(compare(low, self.value) == 0 && compare(high, self.value) == 0)
        case .lt: return compare(low, self.value) < 0
        case .le: return compare(low, self.value) <= 0
        case .gt: return compare(high, self.value) > 0
        case .ge: return compare(high, self.value) >= 0
        }
    }

    /// BYTE_ARRAY and FIXED_LEN_BYTE_ARRAY statistics are ordered by unsigned byte comparison, so the
    /// bounds stay raw bytes and a string literal is compared as its UTF-8 bytes. Swift's `String <`
    /// orders by Unicode canonical equivalence instead (a composed and a decomposed accent compare
    /// equal, and the order differs from the bytes'), and decoding a truncated bound into a `String`
    /// would replace a cut-off UTF-8 sequence with U+FFFD, raising a lower bound: either could rule out
    /// a row group or page that matches.
    private func mayMatchBytes(lower lo: [UInt8], upper hi: [UInt8], leaf: ParquetLeaf) -> Bool {
        // Only a string literal has bytes to compare; FLOAT16 bytes are little-endian numbers.
        guard case .string(let s) = value, leaf.logicalType != .float16 else { return true }
        let lit = Array(s.utf8)
        func cmp(_ a: [UInt8], _ b: [UInt8]) -> Int { a == b ? 0 : (a.lexicographicallyPrecedes(b) ? -1 : 1) }
        switch op {
        case .eq: return cmp(lo, hi) <= 0 ? (cmp(lo, lit) <= 0 && cmp(lit, hi) <= 0) : true
        case .ne: return !(cmp(lo, lit) == 0 && cmp(hi, lit) == 0)
        case .lt: return cmp(lo, lit) < 0
        case .le: return cmp(lo, lit) <= 0
        case .gt: return cmp(hi, lit) > 0
        case .ge: return cmp(hi, lit) >= 0
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
            if case .integer(_, let signed) = leaf.logicalType, !signed { return .uint(UInt64(bitPattern: v)) }
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
            return nil                      // compared as bytes by `mayMatchBytes`
        case .boolean:
            return bytes.first.map { .int(Int64($0 == 0 ? 0 : 1)) }
        case .int96:
            return nil
        }
    }

    private static func isNaN(_ v: Value) -> Bool {
        if case .double(let d) = v { return d.isNaN }
        return false
    }

    /// True when `compare` orders the two: numbers against numbers (strings are compared as bytes).
    private static func comparable(_ a: Value, _ b: Value) -> Bool {
        switch (a, b) {
        case (.int, .int), (.int, .uint), (.int, .double), (.uint, .int), (.uint, .uint), (.uint, .double),
             (.double, .int), (.double, .uint), (.double, .double): return true
        default: return false
        }
    }

    /// Orders two numbers exactly, whatever their kinds: a 64-bit integer is never rounded to a double.
    /// Neither may be NaN (the caller has ruled NaN out).
    private static func compareNumbers(_ a: Value, _ b: Value) -> Int? {
        func sign<T: Comparable>(_ x: T, _ y: T) -> Int { x < y ? -1 : (x == y ? 0 : 1) }
        // An integer against a double: compare with the double's integer part, then with its fraction.
        func intVsDouble(_ x: Int64, _ d: Double) -> Int {
            if d >= 0x1p63 { return -1 }
            if d < -0x1p63 { return 1 }
            let whole = d.rounded(.down)
            let t = Int64(whole)
            if x != t { return sign(x, t) }
            return d > whole ? -1 : 0
        }
        func uintVsDouble(_ x: UInt64, _ d: Double) -> Int {
            if d < 0 { return 1 }
            if d >= 0x1p64 { return -1 }
            let whole = d.rounded(.down)
            let t = UInt64(whole)
            if x != t { return sign(x, t) }
            return d > whole ? -1 : 0
        }
        func intVsUInt(_ x: Int64, _ u: UInt64) -> Int { x < 0 ? -1 : sign(UInt64(x), u) }
        switch (a, b) {
        case (.int(let x), .int(let y)): return sign(x, y)
        case (.uint(let x), .uint(let y)): return sign(x, y)
        case (.double(let x), .double(let y)): return sign(x, y)
        case (.int(let x), .uint(let y)): return intVsUInt(x, y)
        case (.uint(let x), .int(let y)): return -intVsUInt(y, x)
        case (.int(let x), .double(let y)): return intVsDouble(x, y)
        case (.double(let x), .int(let y)): return -intVsDouble(y, x)
        case (.uint(let x), .double(let y)): return uintVsDouble(x, y)
        case (.double(let x), .uint(let y)): return -uintVsDouble(y, x)
        default: return nil
        }
    }

    private func compare(_ a: Value, _ b: Value) -> Int {
        Self.compareNumbers(a, b) ?? 0
    }
}
