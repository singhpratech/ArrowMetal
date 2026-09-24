import XCTest
@testable import ArrowMetal

/// Split-block bloom filters: xxHash64 against its published vectors, the filters pyarrow and DuckDB
/// write, and equality filters that drop the row groups a value is certainly absent from.
final class ParquetBloomFilterTests: XCTestCase {

    private func open(_ name: String) throws -> ParquetFile { try ParquetNestedTests.open(name) }

    func testXXHash64Vectors() {
        XCTAssertEqual(XXHash64.hash([]), 0xEF46DB3751D8E999)
        XCTAssertEqual(XXHash64.hash(Array("a".utf8)), 0xD24EC4F1A98C6E5B)
        XCTAssertEqual(XXHash64.hash(Array("abc".utf8)), 0x44BC2CF5AD770999)
    }

    func testFiltersDecode() throws {
        let f = try open("bloom__pa_snappy")
        for g in 0..<f.rowGroupCount {
            for c in 0..<f.leaves.count {
                let b = try XCTUnwrap(try f.bloomFilter(rowGroup: g, column: c), "bloom \(g)/\(c)")
                XCTAssertGreaterThan(b.blockCount, 0)
            }
        }
        XCTAssertNil(try open("bloom__pa_nobloom").bloomFilter(rowGroup: 0, column: 0))
        let duck = try open("bloom__duckdb")
        let cat = try XCTUnwrap(duck.leaves.firstIndex { $0.name == "cat" })
        XCTAssertNotNil(try duck.bloomFilter(rowGroup: 0, column: cat))
    }

    /// Row group `g` of the pyarrow file holds exactly the values `4 * j + g`, j in 0..<1024.
    func testPresentValuesAreAlwaysFoundAndAbsentOnesMostlyRuledOut() throws {
        let f = try open("bloom__pa_snappy")
        let i64 = try XCTUnwrap(f.leaves.first { $0.name == "i64" })
        let s = try XCTUnwrap(f.leaves.first { $0.name == "s" })
        var falsePositives = 0, lookups = 0
        for g in 0..<4 {
            let bi = try XCTUnwrap(try f.bloomFilter(rowGroup: g, column: i64.index))
            let bs = try XCTUnwrap(try f.bloomFilter(rowGroup: g, column: s.index))
            for v in stride(from: 0, to: 4096, by: 7) {
                let hi = try XCTUnwrap(ParquetFile.bloomHash(.int(Int64(v)), i64))
                let hs = try XCTUnwrap(ParquetFile.bloomHash(.string(String(format: "s%06d", v)), s))
                if v % 4 == g {
                    XCTAssertTrue(bi.mightContain(hash: hi), "i64 \(v) in row group \(g)")
                    XCTAssertTrue(bs.mightContain(hash: hs), "s \(v) in row group \(g)")
                } else {
                    lookups += 2
                    if bi.mightContain(hash: hi) { falsePositives += 1 }
                    if bs.mightContain(hash: hs) { falsePositives += 1 }
                }
            }
        }
        // Written for a 1% false-positive rate.
        XCTAssertLessThan(Double(falsePositives) / Double(lookups), 0.03)
    }

    func testEqualityFiltersSkipRowGroups() throws {
        try requireRealGPU()
        let f = try open("bloom__pa_snappy")
        let opts = ParquetReadOptions(columns: ["i64"], dictionaryEncoded: false,
                                      filters: [ParquetFilter(column: "i64", op: .eq, value: .int(4001))])
        let b = try f.read(opts)
        XCTAssertEqual(f.lastReadStatistics.rowGroupsRead, 1)
        XCTAssertEqual(f.lastReadStatistics.rowGroupsSkippedByBloomFilter, 3)
        let p = try XCTUnwrap(b["i64"]?.asInt64).values.typed(Int64.self)
        XCTAssertTrue((0..<b.length).contains { p[$0] == 4001 })
        f.useBloomFilters = false
        _ = try f.read(opts)
        XCTAssertEqual(f.lastReadStatistics.rowGroupsRead, 4)
        XCTAssertEqual(f.lastReadStatistics.rowGroupsSkippedByBloomFilter, 0)
    }

    func testUnhashableLiteralsKeepTheRowGroup() throws {
        let f = try open("bloom__pa_snappy")
        let f64 = try XCTUnwrap(f.leaves.first { $0.name == "f64" })
        let i32 = try XCTUnwrap(f.leaves.first { $0.name == "i32" })
        XCTAssertNil(ParquetFile.bloomHash(.double(0), f64))          // 0.0 and -0.0 hash differently
        XCTAssertNil(ParquetFile.bloomHash(.double(.nan), f64))
        XCTAssertNil(ParquetFile.bloomHash(.int(1 << 40), i32))       // cannot be an int32 value
        XCTAssertNotNil(ParquetFile.bloomHash(.double(0.5), f64))
    }
}
