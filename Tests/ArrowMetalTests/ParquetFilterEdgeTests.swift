import XCTest
@testable import ArrowMetal

/// Filter and read edge cases: `!=` against a float page whose min and max leave a NaN out, uint64
/// statistics past the signed range, the `null` type below lists, maps and structs, and the index type
/// of a restored dictionary. `python/tests/test_parquet_nested.py` checks the same files against pyarrow.
final class ParquetFilterEdgeTests: XCTestCase {

    private func open(_ name: String) throws -> ParquetFile { try ParquetNestedTests.open(name) }

    private func int64s(_ a: AnyMetalArray?) throws -> [Int64] {
        let a = try XCTUnwrap(a?.asInt64)
        let p = a.values.typed(Int64.self)
        return (0..<a.length).map { p[$0] }
    }

    private func bytes<T>(_ v: T) -> [UInt8] { withUnsafeBytes(of: v) { Array($0) } }

    /// pyarrow leaves NaN out of a page's min / max, so a page of 5.0 with one NaN has min == max == 5.0.
    /// The NaN row satisfies `!= 5.0`; the page must be kept, and the read must agree with the read
    /// without the index.
    func testNotEqualNeverRulesOutAFloatPage() throws {
        try requireRealGPU()
        let f = try open("pageindexnan__pa_constpage")
        let f64 = try XCTUnwrap(f.leaves.first { $0.name == "f64" })
        let f32 = try XCTUnwrap(f.leaves.first { $0.name == "f32" })
        let i64 = try XCTUnwrap(f.leaves.first { $0.name == "i64" })
        // The fixture has the shape: the first page's index entry says 5.0 .. 5.0.
        let ci = try XCTUnwrap(try f.columnIndex(rowGroup: 0, column: 1))
        XCTAssertEqual(ci.minValues[0], bytes(5.0))
        XCTAssertEqual(ci.maxValues[0], bytes(5.0))
        XCTAssertTrue(ParquetFilter(column: "f64", op: .ne, value: .double(5)).mayMatch(lower: bytes(5.0), upper: bytes(5.0), leaf: f64))
        XCTAssertTrue(ParquetFilter(column: "f32", op: .ne, value: .double(5)).mayMatch(lower: bytes(Float(5)), upper: bytes(Float(5)), leaf: f32))
        // An integer column holds no NaN, so a constant page equal to the literal still goes.
        XCTAssertFalse(ParquetFilter(column: "i64", op: .ne, value: .int(5)).mayMatch(lower: bytes(Int64(5)), upper: bytes(Int64(5)), leaf: i64))

        for col in ["f64", "f32"] {
            let flt = [ParquetFilter(column: col, op: .ne, value: .double(5))]
            let opts = ParquetReadOptions(columns: ["id", "f64", "f32"], filters: flt)
            f.usePageIndex = true
            let on = try f.read(opts)
            f.usePageIndex = false
            let off = try f.read(opts)
            func matches(_ b: MetalRecordBatch) throws -> [Int64] {
                let ids = try int64s(b["id"])
                let xs: [Double?] = col == "f64" ? try XCTUnwrap(b["f64"]?.asFloat64).toArray()
                                                 : try XCTUnwrap(b["f32"]?.asFloat32).toArray().map { $0.map(Double.init) }
                return ids.indices.filter { xs[$0].map { $0 != 5 } ?? false }.map { ids[$0] }
            }
            let got = try matches(on)
            XCTAssertEqual(got, try matches(off), col)
            XCTAssertTrue(got.contains(10), "\(col): the NaN row")
        }
        // The integer column's two constant pages are skipped.
        f.usePageIndex = true
        _ = try f.read(ParquetReadOptions(columns: ["id"], filters: [ParquetFilter(column: "i64", op: .ne, value: .int(5))]))
        XCTAssertGreaterThanOrEqual(f.lastReadStatistics.pagesSkipped, 2)
    }

    /// uint64 statistics are unsigned: 2^63 and up must not read as negative numbers.
    func testUnsignedSixtyFourBitStatistics() throws {
        let f = try open("unsigned__pa_plain_none")
        let top = UInt64(1) << 63
        func groups(_ col: String, _ op: ParquetFilter.Op, _ v: ParquetFilter.Value) throws -> [Int] {
            try f.selectedRowGroups(ParquetReadOptions(filters: [ParquetFilter(column: col, op: op, value: v)]))
        }
        XCTAssertEqual(try groups("u64", .ge, .uint(top)), [0, 1, 2, 3, 4])
        XCTAssertEqual(try groups("u64", .eq, .uint(top + 7)), [0])
        XCTAssertEqual(try groups("u64", .gt, .uint(top + 4990)), [4])
        XCTAssertEqual(try groups("u64", .lt, .uint(top + 10)), [0])
        XCTAssertEqual(try groups("u64", .ne, .uint(top + 7)), [0, 1, 2, 3, 4])
        // A signed or float literal against the unsigned range.
        XCTAssertEqual(try groups("u64", .lt, .int(0)), [])
        XCTAssertEqual(try groups("u64", .ge, .int(0)), [0, 1, 2, 3, 4])
        XCTAssertEqual(try groups("u64", .gt, .double(9.3e18)), [])
        XCTAssertEqual(try groups("u64", .ge, .double(0x1p63)), [0, 1, 2, 3, 4])
        // Small unsigned values and uint32 past the int32 range compare as before.
        XCTAssertEqual(try groups("u64lo", .ge, .int(12_000)), [4])
        XCTAssertEqual(try groups("u32", .lt, .int((1 << 31) + 1000)), [0])
        XCTAssertEqual(try groups("u32", .gt, .uint(top)), [])
        // Pages go by the same order.
        try requireRealGPU()
        let b = try f.read(ParquetReadOptions(columns: ["id", "u64"], filters: [ParquetFilter(column: "u64", op: .eq, value: .uint(top + 7))]))
        XCTAssertTrue(try int64s(b["id"]).contains(7))
        XCTAssertGreaterThan(f.lastReadStatistics.pagesSkipped, 0)
    }

    /// The `null` type (a Parquet UNKNOWN leaf) below a list, a list of lists, a map, a list of structs and
    /// a struct of a list: the null child has one slot per element.
    func testNullLeavesBelowRepeatedFields() throws {
        try requireRealGPU()
        for v in ["pa_plain_none", "pa_dict_snappy", "pa_v2_snappy", "pa_v2_lz4"] {
            let b = try open("nullleaves__" + v).read(ParquetReadOptions(dictionaryEncoded: false))
            XCTAssertEqual(b.length, 240, v)
            guard case .list(let ln)? = b["ln"], case .null(let lnValues) = ln.values else { return XCTFail("\(v): ln") }
            // [[null, null], [], null, [null]] repeated: three elements every four rows.
            XCTAssertEqual(lnValues.length, 180, v)
            XCTAssertEqual(Int(ln.offsets.typed(Int32.self)[240]), 180, v)
            guard case .list(let lln)? = b["lln"], case .list(let inner) = lln.values,
                  case .null(let llnValues) = inner.values else { return XCTFail("\(v): lln") }
            XCTAssertEqual(llnValues.length, Int(inner.offsets.typed(Int32.self)[inner.length]), v)
            guard case .map(let mn)? = b["mn"], case .null(let items) = mn.items else { return XCTFail("\(v): mn") }
            XCTAssertEqual(items.length, mn.keys.length, v)
            guard case .list(let lsn)? = b["lsn"], case .structure(let st) = lsn.values,
                  case .null(let a) = st.children[0] else { return XCTFail("\(v): lsn") }
            XCTAssertEqual(a.length, st.length, v)
            XCTAssertEqual(st.length, Int(lsn.offsets.typed(Int32.self)[240]), v)
            XCTAssertEqual(ParquetTests.fingerprint(st.children[1]).count, st.length, v)
            guard case .structure(let sln)? = b["sln"], case .list(let l) = sln.children[0],
                  case .null(let lv) = l.values else { return XCTFail("\(v): sln") }
            XCTAssertEqual(lv.length, Int(l.offsets.typed(Int32.self)[l.length]), v)
        }
    }

    /// A restored dictionary always has int32 indices and no ordered flag, whatever index width and flag
    /// the stored Arrow type had (pandas: int8, ordered for an ordered categorical; Polars: uint32 for
    /// Categorical, uint8 and ordered for Enum). The values are the stored ones.
    func testRestoredDictionariesHaveInt32Indices() throws {
        try requireRealGPU()
        for (name, cols) in [("catwidth__pandas", ["c", "o"]), ("catwidth__polars", ["cat", "enum"])] {
            let b = try open(name).read(ParquetReadOptions(dictionaryEncoded: false))
            for c in cols {
                guard case .dictionary(let codes, let values)? = b[c] else { return XCTFail("\(name).\(c) is not a dictionary") }
                XCTAssertEqual(codes.length, b.length, "\(name).\(c)")
                XCTAssertEqual(values.arrowFormat, "u", "\(name).\(c)")
            }
        }
    }
}
