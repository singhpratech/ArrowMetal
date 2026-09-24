import XCTest
@testable import ArrowMetal

/// Page-level skipping with the column index and offset index: the pages a filter rules out are never
/// decoded, the matching rows are the same with and without the index, and every column of the result
/// covers the same rows. `python/tests/test_parquet_nested.py` checks the same files against pyarrow.
final class ParquetPageIndexTests: XCTestCase {

    private func open(_ name: String) throws -> ParquetFile { try ParquetNestedTests.open(name) }

    /// `id` is `5 * row + 17` in every page-index fixture, `i32` is `row / 40`.
    private func ids(_ b: MetalRecordBatch) throws -> [Int64] {
        let a = try XCTUnwrap(b["id"]?.asInt64)
        let p = a.values.typed(Int64.self)
        return (0..<a.length).map { p[$0] }
    }

    func testIndexesDecode() throws {
        let f = try open("pageindex__pa_plain_none")
        for g in 0..<f.rowGroupCount {
            let rows = Int(f.metadata.rowGroups[g].numRows)
            for c in 0..<f.leaves.count where f.leaves[c].maxRepetition == 0 {
                let oi = try XCTUnwrap(try f.offsetIndex(rowGroup: g, column: c), "offset index \(g)/\(c)")
                let ci = try XCTUnwrap(try f.columnIndex(rowGroup: g, column: c), "column index \(g)/\(c)")
                XCTAssertGreaterThanOrEqual(oi.count, g == 0 && f.leaves[c].name == "id" ? 3 : 1)
                XCTAssertEqual(ci.minValues.count, oi.count)
                XCTAssertEqual(ci.nullPages.count, oi.count)
                XCTAssertTrue(ParquetFile.wellOrdered(oi, rows: rows))
                // The offset index lists exactly the data pages a walk of the chunk finds.
                let walked = try f.pageHeaders(of: f.metadata.rowGroups[g].columns[c].meta, rowGroup: g).data
                XCTAssertEqual(walked.count, oi.count)
            }
        }
        XCTAssertNil(try open("pageindex__pa_noindex").offsetIndex(rowGroup: 0, column: 0))
    }

    func testSkippedPagesGiveTheSameMatchingRows() throws {
        try requireRealGPU()
        let filters: [[ParquetFilter]] = [
            [ParquetFilter(column: "id", op: .gt, value: .int(20000))],
            [ParquetFilter(column: "id", op: .eq, value: .int(12017))],
            [ParquetFilter(column: "i32", op: .lt, value: .int(30))],
            [ParquetFilter(column: "cat", op: .eq, value: .string("c0010"))],
            [ParquetFilter(column: "id", op: .ge, value: .int(10000)), ParquetFilter(column: "i32", op: .lt, value: .int(120))],
        ]
        for name in ["pageindex__pa_plain_none", "pageindex__pa_dict_snappy", "pageindex__pa_v2_snappy", "pageindex__polars"] {
            let f = try open(name)
            for flt in filters {
                let opts = ParquetReadOptions(columns: ["id", "i32", "f64", "cat"], dictionaryEncoded: false, filters: flt)
                f.usePageIndex = false
                let whole = try f.read(opts)
                let without = f.lastReadStatistics
                f.usePageIndex = true
                let part = try f.read(opts)
                let with = f.lastReadStatistics
                XCTAssertEqual(without.pagesSkipped, 0)
                // The same exact matches: every id in the page-index read is in the whole read, and every
                // whole-read id that matches the filter is in the page-index read.
                let wholeIDs = try ids(whole), partIDs = Set(try ids(part))
                func matches(_ id: Int64) -> Bool {
                    let row = (id - 17) / 5, i32 = row / 40
                    return flt.allSatisfy { f in
                        switch (f.column, f.op, f.value) {
                        case ("id", .gt, .int(let v)): return id > v
                        case ("id", .eq, .int(let v)): return id == v
                        case ("id", .ge, .int(let v)): return id >= v
                        case ("i32", .lt, .int(let v)): return i32 < v
                        case ("cat", .eq, .string(let v)): return String(format: "c%04d", Int(row) / 250) == v
                        default: return true
                        }
                    }
                }
                XCTAssertTrue(partIDs.isSubset(of: Set(wholeIDs)), "\(name) \(flt.map(\.column))")
                XCTAssertEqual(wholeIDs.filter(matches), try ids(part).filter(matches), "\(name) \(flt.map(\.column))")
                XCTAssertLessThanOrEqual(part.length, whole.length)
                XCTAssertEqual(with.rows, part.length)
                // Flat columns: every page of the row groups read is decoded or skipped.
                XCTAssertEqual(with.pagesDecoded + with.pagesSkipped, without.pagesDecoded, "\(name)")
                for c in part.columns { XCTAssertEqual(c.length, part.length) }
            }
        }
    }

    func testPagesAreSkipped() throws {
        try requireRealGPU()
        let f = try open("pageindex__pa_plain_none")
        _ = try f.read(ParquetReadOptions(columns: ["id", "cat"], filters: [ParquetFilter(column: "id", op: .eq, value: .int(12017))]))
        let s = f.lastReadStatistics
        XCTAssertEqual(s.rowGroupsRead, 1)
        XCTAssertEqual(s.rowGroupsSkippedByStatistics, 2)
        XCTAssertGreaterThan(s.pagesSkipped, s.pagesDecoded)
        XCTAssertLessThan(s.rows, Int(f.metadata.rowGroups[1].numRows))
        // Without a filter nothing is skipped, and nothing is trimmed.
        _ = try f.read(ParquetReadOptions(columns: ["id"]))
        XCTAssertEqual(f.lastReadStatistics.pagesSkipped, 0)
        XCTAssertEqual(f.lastReadStatistics.rows, 6000)
    }

    func testRowGroupRuledOutByPagesAlone() throws {
        try requireRealGPU()
        let f = try open("pageindex__pa_plain_none")
        let flt = [ParquetFilter(column: "id", op: .ge, value: .int(20000)), ParquetFilter(column: "i32", op: .lt, value: .int(70))]
        XCTAssertTrue(try f.selectedRowGroups(ParquetReadOptions(filters: flt)).contains(1))
        let ranges = try f.candidateRowRanges(ParquetReadOptions(filters: flt), rowGroups: [1])
        XCTAssertEqual(ranges[1], [])
        _ = try f.read(ParquetReadOptions(columns: ["id", "xs"], filters: flt))
        XCTAssertGreaterThanOrEqual(f.lastReadStatistics.rowGroupsSkippedByPageIndex, 1)
    }

    func testNestedColumnsAreTrimmedToTheSameRows() throws {
        try requireRealGPU()
        let f = try open("pageindex__pa_dict_snappy")
        let b = try f.read(ParquetReadOptions(columns: ["xs", "id"], dictionaryEncoded: false,
                                              filters: [ParquetFilter(column: "id", op: .lt, value: .int(500))]))
        let ids = try self.ids(b)
        XCTAssertLessThan(ids.count, 2500)
        let xs = ParquetNestedTests.rows(try XCTUnwrap(b["xs"]))
        for (k, id) in ids.enumerated() {
            let i = Int((id - 17) / 5)
            let want = i % 31 == 0 ? "null" : "[" + (0..<(i % 3)).map { "\(i + $0)" }.joined(separator: ", ") + "]"
            XCTAssertEqual(xs[k], want, "xs at id \(id)")
        }
    }

    /// Polars flags every page that holds a NaN as a null page, with a null count of 0. Those pages hold
    /// matching rows, so they must not be skipped.
    func testNaNPagesFlaggedNullByPolarsAreKept() throws {
        try requireRealGPU()
        let polars = try open("pageindexnan__polars")
        let ci = try XCTUnwrap(try polars.columnIndex(rowGroup: 0, column: 1))
        XCTAssertTrue(ci.nullPages.contains(true))
        XCTAssertEqual(ci.nullCounts?.allSatisfy { $0 == 0 }, true)
        // Polars' row-group min / max leave the flagged pages out too, so they cannot drop the row group:
        // the least value is about -350, the row group's min about -77.
        let below = [ParquetFilter(column: "f64", op: .lt, value: .double(-150))]
        XCTAssertEqual(try polars.selectedRowGroups(ParquetReadOptions(filters: below)), [0])
        XCTAssertFalse(below[0].mayMatch(rowGroup: polars.metadata.rowGroups[0], file: polars))
        let filters: [[ParquetFilter]] = [
            [ParquetFilter(column: "f64", op: .gt, value: .double(100))],
            [ParquetFilter(column: "f64", op: .lt, value: .double(-150))],
            [ParquetFilter(column: "f32", op: .lt, value: .double(5))],
            [ParquetFilter(column: "f64", op: .ge, value: .double(0)), ParquetFilter(column: "id", op: .lt, value: .int(3000))],
        ]
        for name in ["pageindexnan__polars", "pageindexnan__pa_plain_none"] {
            let f = try open(name)
            for flt in filters {
                let opts = ParquetReadOptions(columns: ["id", "f64", "f32"], filters: flt)
                f.usePageIndex = false
                let whole = try f.read(opts)
                f.usePageIndex = true
                let part = try f.read(opts)
                XCTAssertEqual(try matchingIDs(whole, flt), try matchingIDs(part, flt), "\(name) \(flt.map(\.column))")
                XCTAssertGreaterThan(try matchingIDs(part, flt).count, 0, "\(name) \(flt.map(\.column))")
            }
        }
        // pyarrow leaves NaN out of min / max and flags nothing, so its pages still skip.
        let pa = try open("pageindexnan__pa_plain_none")
        _ = try pa.read(ParquetReadOptions(columns: ["id"], filters: [ParquetFilter(column: "f64", op: .gt, value: .double(100))]))
        XCTAssertGreaterThan(pa.lastReadStatistics.pagesSkipped, 0)
    }

    /// The ids whose `f64`, `f32` and `id` satisfy every filter, from the batch's own values.
    private func matchingIDs(_ b: MetalRecordBatch, _ flt: [ParquetFilter]) throws -> [Int64] {
        let ids = try self.ids(b)
        let f64 = try XCTUnwrap(b["f64"]?.asFloat64).toArray()
        let f32 = try XCTUnwrap(b["f32"]?.asFloat32).toArray()
        return ids.indices.filter { i in
            flt.allSatisfy { f in
                let x: Double
                switch f.column {
                case "f64": x = f64[i] ?? .nan
                case "f32": x = f32[i].map(Double.init) ?? .nan
                default: x = Double(ids[i])
                }
                let v: Double
                switch f.value { case .double(let d): v = d; case .int(let n): v = Double(n); default: return false }
                switch f.op {
                case .gt: return x > v
                case .ge: return x >= v
                case .lt: return x < v
                case .le: return x <= v
                case .eq: return x == v
                case .ne: return x != v
                }
            }
        }.map { ids[$0] }
    }

    /// The format says a NaN min or max must be ignored; a literal of another kind than the column's
    /// cannot rule anything out either.
    func testNaNBoundsAndMismatchedLiteralsRuleNothingOut() throws {
        let f = try open("pageindexnan__pa_plain_none")
        let f64 = try XCTUnwrap(f.leaves.first { $0.name == "f64" })
        let id = try XCTUnwrap(f.leaves.first { $0.name == "id" })
        func bytes(_ d: Double) -> [UInt8] { withUnsafeBytes(of: d) { Array($0) } }
        func bytes(_ n: Int64) -> [UInt8] { withUnsafeBytes(of: n) { Array($0) } }
        // A real bound still rules out.
        XCTAssertFalse(ParquetFilter(column: "f64", op: .gt, value: .double(10)).mayMatch(lower: bytes(1.0), upper: bytes(5.0), leaf: f64))
        XCTAssertTrue(ParquetFilter(column: "f64", op: .gt, value: .double(10)).mayMatch(lower: bytes(1.0), upper: bytes(Double.nan), leaf: f64))
        XCTAssertTrue(ParquetFilter(column: "f64", op: .lt, value: .double(0)).mayMatch(lower: bytes(Double.nan), upper: bytes(5.0), leaf: f64))
        XCTAssertTrue(ParquetFilter(column: "f64", op: .eq, value: .double(7)).mayMatch(lower: bytes(Double.nan), upper: bytes(Double.nan), leaf: f64))
        // A string literal against an int64 column: never ruled out, whatever the op.
        for op in [ParquetFilter.Op.eq, .ne, .lt, .le, .gt, .ge] {
            XCTAssertTrue(ParquetFilter(column: "id", op: op, value: .string("datetime.date(2021, 1, 1)"))
                            .mayMatch(lower: bytes(Int64(5)), upper: bytes(Int64(5)), leaf: id), "\(op)")
        }
    }

    func testRangeIntersection() {
        XCTAssertEqual(ParquetFile.intersect([0..<10, 20..<30], [5..<25]), [5..<10, 20..<25])
        XCTAssertEqual(ParquetFile.intersect([0..<10], [10..<20]), [])
        XCTAssertEqual(ParquetFile.intersect([], [0..<5]), [])
    }
}
