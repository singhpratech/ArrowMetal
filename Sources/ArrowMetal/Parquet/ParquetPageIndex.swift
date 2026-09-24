import Foundation
import Metal

// Page-level skipping with the column index and the offset index.
//
// A writer may store two small structures per column chunk after the row groups: the `ColumnIndex`
// (for every data page, its min and max value and whether it holds only nulls) and the `OffsetIndex`
// (for every data page, where it starts and the index of its first row). With them a statistics filter
// can do better than keep or drop whole row groups:
//
//   1. For each filter on a flat column, each page whose [min, max] cannot satisfy it (or that holds only
//      nulls, by its null-page flag *and* its null count) rules out its rows. The rows the filters leave,
//      intersected across filters, are the row group's *candidate ranges*; a row group left with none is
//      dropped like a statistics-pruned one.
//   2. Every flat column that has an offset index then decodes only the data pages that overlap a
//      candidate range. The skipped pages are never decompressed, never decoded, and their headers are
//      never read: the page list comes from the offset index, not from walking the chunk.
//   3. Pages have different boundaries in different columns, so each column is finally trimmed to exactly
//      the candidate rows (one `filter` with a mask built from the ranges). Every column of the result
//      therefore covers the same rows, and every matching row is among them.
//
// Nested columns and columns without an offset index decode their row groups whole and are trimmed the
// same way. When no page is ruled out the read is exactly the read without the index.

/// One data page in the `OffsetIndex`.
public struct ParquetPageLocation: Sendable {
    public var offset: Int64 = 0
    public var compressedPageSize: Int32 = 0
    public var firstRowIndex: Int64 = 0
}

/// The `ColumnIndex` of one column chunk: per data page, min / max statistics and a null-page flag.
public struct ParquetColumnIndex: Sendable {
    public var nullPages: [Bool] = []
    public var minValues: [[UInt8]] = []
    public var maxValues: [[UInt8]] = []
    public var nullCounts: [Int64]? = nil
}

/// What one read did with the footer statistics and the page index (`ParquetFile.lastReadStatistics`).
public struct ParquetReadStatistics: Sendable, Equatable {
    /// Row groups the read decoded.
    public var rowGroupsRead = 0
    /// Row groups dropped by the row-group statistics.
    public var rowGroupsSkippedByStatistics = 0
    /// Row groups every page of which the column index ruled out.
    public var rowGroupsSkippedByPageIndex = 0
    /// Row groups an equality filter's value is certainly absent from, by the bloom filter.
    public var rowGroupsSkippedByBloomFilter = 0
    /// Data pages decoded, summed over every leaf column read.
    public var pagesDecoded = 0
    /// Data pages the page index let the read leave untouched, summed over every leaf column read.
    public var pagesSkipped = 0
    /// Rows in the result.
    public var rows = 0
    public init() {}
}

/// Per-read state: the candidate row ranges per row group, and the page counts.
final class ParquetReadPlan {
    /// Row group -> sorted, disjoint candidate row ranges (row-group coordinates). A row group that is
    /// absent keeps all of its rows.
    let ranges: [Int: [Range<Int>]]
    var pagesDecoded = 0
    var pagesSkipped = 0
    private let lock = NSLock()

    init(ranges: [Int: [Range<Int>]]) { self.ranges = ranges }

    func count(decoded: Int, skipped: Int) {
        lock.lock(); pagesDecoded += decoded; pagesSkipped += skipped; lock.unlock()
    }
}

extension ParquetFile {

    // MARK: - Reading the two indexes

    /// The offset index of one column chunk, or nil when the writer did not write one.
    public func offsetIndex(rowGroup g: Int, column c: Int) throws -> [ParquetPageLocation]? {
        guard g >= 0, g < metadata.rowGroups.count, c >= 0, c < metadata.rowGroups[g].columns.count else { return nil }
        let chunk = metadata.rowGroups[g].columns[c]
        guard let off = chunk.offsetIndexOffset, let len = chunk.offsetIndexLength, off >= 0, len > 0,
              Int(off) + Int(len) <= fileSize else { return nil }
        var r = ThriftReader(bytes, at: Int(off))
        var pages: [ParquetPageLocation] = []
        try r.readStruct { r, id, _ in
            guard id == 1 else { return false }
            try r.readList { rr in
                var p = ParquetPageLocation()
                try rr.readStruct { r2, fid, _ in
                    switch fid {
                    case 1: p.offset = try r2.int64(); return true
                    case 2: p.compressedPageSize = try r2.int32(); return true
                    case 3: p.firstRowIndex = try r2.int64(); return true
                    default: return false
                    }
                }
                pages.append(p)
            }
            return true
        }
        return pages
    }

    /// The column index of one column chunk, or nil when the writer did not write one.
    public func columnIndex(rowGroup g: Int, column c: Int) throws -> ParquetColumnIndex? {
        guard g >= 0, g < metadata.rowGroups.count, c >= 0, c < metadata.rowGroups[g].columns.count else { return nil }
        let chunk = metadata.rowGroups[g].columns[c]
        guard let off = chunk.columnIndexOffset, let len = chunk.columnIndexLength, off >= 0, len > 0,
              Int(off) + Int(len) <= fileSize else { return nil }
        var r = ThriftReader(bytes, at: Int(off))
        var ci = ParquetColumnIndex()
        try r.readStruct { r, id, _ in
            switch id {
            case 1:
                let h = try r.listHeader()
                for _ in 0..<h.count { ci.nullPages.append(try r.byte() == 1) }
                return true
            case 2:
                let h = try r.listHeader()
                for _ in 0..<h.count { ci.minValues.append(try r.binary()) }
                return true
            case 3:
                let h = try r.listHeader()
                for _ in 0..<h.count { ci.maxValues.append(try r.binary()) }
                return true
            case 5:
                let h = try r.listHeader()
                var counts: [Int64] = []
                for _ in 0..<h.count { counts.append(try r.int64()) }
                ci.nullCounts = counts
                return true
            default:
                return false
            }
        }
        return ci
    }

    /// A row group's row count as a usable `Int`: never negative, whatever a damaged footer says.
    func rowsIn(group g: Int) -> Int {
        Swift.min(Swift.max(0, Int(metadata.rowGroups[g].numRows)), 1 << 40)
    }

    /// An offset index whose pages start at row 0 and move strictly forward, the only kind worth trusting.
    static func wellOrdered(_ oi: [ParquetPageLocation], rows: Int) -> Bool {
        guard let first = oi.first, first.firstRowIndex == 0 else { return false }
        for i in 1..<Swift.max(oi.count, 1) where i < oi.count {
            guard oi[i].firstRowIndex > oi[i - 1].firstRowIndex, oi[i].firstRowIndex < Int64(rows) else { return false }
        }
        return true
    }

    // MARK: - Candidate rows

    /// For each row group in `groups`, the rows the filters' column indexes cannot rule out, as sorted
    /// disjoint ranges in row-group coordinates. A row group that no page index narrows is absent from
    /// the result; one whose every page is ruled out maps to an empty list.
    public func candidateRowRanges(_ options: ParquetReadOptions, rowGroups groups: [Int]) throws -> [Int: [Range<Int>]] {
        var out: [Int: [Range<Int>]] = [:]
        guard usePageIndex, !options.filters.isEmpty else { return out }
        for g in groups {
            let rows = rowsIn(group: g)
            var keep: [Range<Int>] = [0..<rows]
            var narrowed = false
            for filter in options.filters {
                guard let leaf = leaves.first(where: { $0.dottedPath == filter.column || $0.name == filter.column }),
                      leaf.maxRepetition == 0,
                      let ci = try? columnIndex(rowGroup: g, column: leaf.index),
                      let oi = try? offsetIndex(rowGroup: g, column: leaf.index),
                      !oi.isEmpty, Self.wellOrdered(oi, rows: rows),
                      ci.nullPages.count == oi.count, ci.minValues.count == oi.count,
                      ci.maxValues.count == oi.count else { continue }
                var ranges: [Range<Int>] = []
                for (i, page) in oi.enumerated() {
                    let lo = Swift.max(0, Swift.min(Int(page.firstRowIndex), rows))
                    let hi = i + 1 < oi.count ? Swift.max(lo, Swift.min(Int(oi[i + 1].firstRowIndex), rows)) : rows
                    guard lo < hi else { continue }
                    // An all-null page satisfies no comparison, but only a confirmed one is ruled out: Polars
                    // flags every page holding a NaN as a null page while its null count is 0. A flagged page
                    // whose null count does not cover its rows is kept, since its min / max are placeholders.
                    let keepPage: Bool
                    if ci.nullPages[i] {
                        if Self.certainlyAllNull(ci, page: i, rows: hi - lo) { continue }
                        keepPage = true
                    } else {
                        keepPage = filter.mayMatch(lower: ci.minValues[i], upper: ci.maxValues[i], leaf: leaf)
                    }
                    if keepPage {
                        if let last = ranges.last, last.upperBound == lo { ranges[ranges.count - 1] = last.lowerBound..<hi }
                        else { ranges.append(lo..<hi) }
                    }
                }
                keep = Self.intersect(keep, ranges)
                narrowed = true
            }
            if narrowed, keep != [0..<rows] { out[g] = keep }
        }
        return out
    }

    /// True when page `i`, flagged as a null page, is confirmed to hold nothing but nulls: the index's
    /// `null_counts` entry for it covers all `rows` of it (a flat column has one value per row). Without
    /// null counts the flag alone is not trusted and the page is kept.
    static func certainlyAllNull(_ ci: ParquetColumnIndex, page i: Int, rows: Int) -> Bool {
        guard let counts = ci.nullCounts, i < counts.count else { return false }
        return counts[i] >= Int64(rows)
    }

    /// False when the chunk's column index flags a page as a null page that its null counts say is not all
    /// null. Polars writes such pages for every page holding a NaN and leaves them out of the row group's
    /// min / max, so those statistics do not bound the chunk's values and must not drop the row group
    /// (pyarrow trusts them and drops it). True when there is no column index or nothing looks wrong.
    func rowGroupStatisticsTrusted(_ g: Int, column: String) -> Bool {
        guard let leaf = leaves.first(where: { $0.dottedPath == column || $0.name == column }),
              leaf.maxRepetition == 0,
              let ci = try? columnIndex(rowGroup: g, column: leaf.index), ci.nullPages.contains(true),
              let counts = ci.nullCounts, counts.count == ci.nullPages.count else { return true }
        let rows = rowsIn(group: g)
        let oi = (try? offsetIndex(rowGroup: g, column: leaf.index)).flatMap { $0 }
        let pageRows: ((Int) -> Int)? = oi.flatMap { oi in
            guard oi.count == ci.nullPages.count, Self.wellOrdered(oi, rows: rows) else { return nil }
            return { i in (i + 1 < oi.count ? Int(oi[i + 1].firstRowIndex) : rows) - Int(oi[i].firstRowIndex) }
        }
        for i in ci.nullPages.indices where ci.nullPages[i] {
            if counts[i] <= 0 { return false }
            if let pageRows, counts[i] < Int64(pageRows(i)) { return false }
        }
        return true
    }

    /// Intersection of two sorted lists of disjoint ranges.
    static func intersect(_ a: [Range<Int>], _ b: [Range<Int>]) -> [Range<Int>] {
        var out: [Range<Int>] = []
        var i = 0, j = 0
        while i < a.count && j < b.count {
            let lo = Swift.max(a[i].lowerBound, b[j].lowerBound)
            let hi = Swift.min(a[i].upperBound, b[j].upperBound)
            if lo < hi { out.append(lo..<hi) }
            if a[i].upperBound < b[j].upperBound { i += 1 } else { j += 1 }
        }
        return out
    }

    // MARK: - Pages of one chunk, from the offset index

    /// The data pages of a flat column chunk that overlap `ranges`, located through the offset index, plus
    /// the dictionary page, and the row spans (row-group coordinates) the kept pages cover. Nil when the
    /// chunk has no usable offset index, in which case the caller walks the chunk as usual.
    func indexedPages(of chunk: ParquetColumnMetadata, rowGroup g: Int, column c: Int, ranges: [Range<Int>])
        throws -> (dict: ParquetRawPage?, data: [ParquetRawPage], spans: [Range<Int>], skipped: Int)? {
        let rows = rowsIn(group: g)
        guard let oi = try? offsetIndex(rowGroup: g, column: c), !oi.isEmpty, Self.wellOrdered(oi, rows: rows) else {
            return nil
        }
        var dict: ParquetRawPage? = nil
        let firstData = Int(oi[0].offset)
        let start = Int(chunk.startOffset)
        if start >= 0, start < firstData {
            // Whatever precedes the first data page is the dictionary page.
            var r = ThriftReader(bytes, at: start)
            let h = try ParquetPageHeader.read(&r)
            if h.type == .dictionaryPage {
                try validate(h, at: start, body: r.pos)
                dict = ParquetRawPage(header: h, bodyOffset: r.pos, rowGroup: g)
            }
        }
        var data: [ParquetRawPage] = []
        var spans: [Range<Int>] = []
        var skipped = 0
        var k = 0
        for (i, loc) in oi.enumerated() {
            let lo = Swift.max(0, Swift.min(Int(loc.firstRowIndex), rows))
            let hi = i + 1 < oi.count ? Swift.max(lo, Swift.min(Int(oi[i + 1].firstRowIndex), rows)) : rows
            while k < ranges.count && ranges[k].upperBound <= lo { k += 1 }
            guard k < ranges.count, ranges[k].lowerBound < hi else { skipped += 1; continue }
            let at = Int(loc.offset)
            guard at >= 0, at < fileSize else {
                throw ParquetError.malformed("offset index places a page at \(at), outside the \(fileSize)-byte file")
            }
            var r = ThriftReader(bytes, at: at)
            let h = try ParquetPageHeader.read(&r)
            try validate(h, at: at, body: r.pos)
            guard h.type == .dataPage || h.type == .dataPageV2, Int(h.numValues) == hi - lo else {
                // The index disagrees with the pages; do not trust it for this chunk.
                return nil
            }
            data.append(ParquetRawPage(header: h, bodyOffset: r.pos, rowGroup: g))
            if let last = spans.last, last.upperBound == lo { spans[spans.count - 1] = last.lowerBound..<hi }
            else { spans.append(lo..<hi) }
        }
        return (dict, data, spans, skipped)
    }

    // MARK: - Trimming to the candidate rows

    /// Restricts `array`, whose rows are `spans` (row group, row-group-coordinate range) in order, to the
    /// candidate rows of `plan`. Returns the array unchanged when every one of its rows is a candidate.
    func trim(_ array: AnyMetalArray, spans: [(group: Int, rows: Range<Int>)], plan: ParquetReadPlan) throws -> AnyMetalArray {
        let n = spans.reduce(0) { $0 + $1.rows.count }
        guard n == array.length else {
            throw ParquetError.malformed("a column decoded \(array.length) rows where its pages cover \(n)")
        }
        var all = true
        for s in spans {
            if let r = plan.ranges[s.group], Self.intersect(r, [s.rows]) != [s.rows] { all = false; break }
        }
        if all || n == 0 { return array }
        let ctx = context
        let bytes = try MetalArrowBuffer.allocate(byteCount: n, zeroed: true, context: ctx)
        let p = bytes.mutableContents.assumingMemoryBound(to: UInt8.self)
        var base = 0
        for s in spans {
            let keep = plan.ranges[s.group].map { Self.intersect($0, [s.rows]) } ?? [s.rows]
            for r in keep { memset(p + base + (r.lowerBound - s.rows.lowerBound), 1, r.count) }
            base += s.rows.count
        }
        let bits = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 4), zeroed: true, context: ctx)
        try ParquetTypeMap.run(self, ctx, "pq_bytes_to_bitmap", count: (n + 31) / 32) { enc in
            enc.setBuffer(bytes.mtl, offset: bytes.offset, index: 0)
            Dispatch.setUInt(enc, n, index: 1)
            enc.setBuffer(bits.mtl, offset: bits.offset, index: 2)
        }
        let mask = MetalBooleanArray(length: n, nullCount: 0, validity: nil, values: bits, context: ctx)
        return try array.filter(mask)
    }
}
