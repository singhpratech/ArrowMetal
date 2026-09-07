import XCTest
@testable import ArrowMetal

/// The four rank tiebreakers against the whole option surface, over a column that has ties, nulls and
/// a NaN at once — the three things a rank has to place independently.
///
/// Every expected row is `pc.rank(a, sort_keys=..., null_placement=..., tiebreaker=...)` from
/// pyarrow 25.0.1 over `[3, 1, None, 3, 2, None, 1, nan]` as float64.
final class AdversarialRankTests: XCTestCase {

    private static let vals: [Double?] = [3, 1, nil, 3, 2, nil, 1, Double.nan]

    private func assertRank(_ tiebreaker: RankTiebreaker, descending: Bool, _ placement: NullPlacement,
                            _ expected: [Int32], file: StaticString = #filePath, line: UInt = #line) throws {
        let a = try MetalArray<Double>(Self.vals, context: .shared)
        let got = try a.rank(tiebreaker: tiebreaker, descending: descending, nullPlacement: placement)
        XCTAssertEqual(got.toRawArray(), expected,
                       "rank(\(tiebreaker), descending: \(descending), \(placement))", file: file, line: line)
    }

    func testFirstTiebreaker() throws {
        try requireRealGPU()
        try assertRank(.first, descending: false, .atEnd, [4, 1, 7, 5, 3, 8, 2, 6])
        try assertRank(.first, descending: true, .atEnd, [1, 4, 7, 2, 3, 8, 5, 6])
        try assertRank(.first, descending: false, .atStart, [7, 4, 1, 8, 6, 2, 5, 3])
        try assertRank(.first, descending: true, .atStart, [4, 7, 1, 5, 6, 2, 8, 3])
    }

    func testMinTiebreaker() throws {
        try requireRealGPU()
        try assertRank(.min, descending: false, .atEnd, [4, 1, 7, 4, 3, 7, 1, 6])
        try assertRank(.min, descending: true, .atEnd, [1, 4, 7, 1, 3, 7, 4, 6])
        try assertRank(.min, descending: false, .atStart, [7, 4, 1, 7, 6, 1, 4, 3])
        try assertRank(.min, descending: true, .atStart, [4, 7, 1, 4, 6, 1, 7, 3])
    }

    func testDenseTiebreaker() throws {
        try requireRealGPU()
        try assertRank(.dense, descending: false, .atEnd, [3, 1, 5, 3, 2, 5, 1, 4])
        try assertRank(.dense, descending: true, .atEnd, [1, 3, 5, 1, 2, 5, 3, 4])
        try assertRank(.dense, descending: false, .atStart, [5, 3, 1, 5, 4, 1, 3, 2])
        try assertRank(.dense, descending: true, .atStart, [3, 5, 1, 3, 4, 1, 5, 2])
    }

    func testMaxTiebreaker() throws {
        try requireRealGPU()
        try assertRank(.max, descending: false, .atEnd, [5, 2, 8, 5, 3, 8, 2, 6])
        try assertRank(.max, descending: true, .atEnd, [2, 5, 8, 2, 3, 8, 5, 6])
        try assertRank(.max, descending: false, .atStart, [8, 5, 2, 8, 6, 2, 5, 3])
        try assertRank(.max, descending: true, .atStart, [5, 8, 2, 5, 6, 2, 8, 3])
    }

    /// The SQL-named entry points must be the same four modes.
    func testSQLNamesAgreeWithTheTiebreakers() throws {
        try requireRealGPU()
        let a = try MetalArray<Double>(Self.vals, context: .shared)
        for placement in [NullPlacement.atEnd, .atStart] {
            for descending in [false, true] {
                XCTAssertEqual(try a.rowNumber(descending: descending, nullPlacement: placement).toRawArray(),
                               try a.rank(tiebreaker: .first, descending: descending, nullPlacement: placement).toRawArray())
                XCTAssertEqual(try a.rank(descending: descending, nullPlacement: placement).toRawArray(),
                               try a.rank(tiebreaker: .min, descending: descending, nullPlacement: placement).toRawArray())
                XCTAssertEqual(try a.denseRank(descending: descending, nullPlacement: placement).toRawArray(),
                               try a.rank(tiebreaker: .dense, descending: descending, nullPlacement: placement).toRawArray())
                XCTAssertEqual(try a.maxRank(descending: descending, nullPlacement: placement).toRawArray(),
                               try a.rank(tiebreaker: .max, descending: descending, nullPlacement: placement).toRawArray())
            }
        }
    }

    /// Ranks over a slice must equal the ranks over the same rows built as a column of their own, at
    /// every tiebreaker and both null placements.
    func testRanksOnASlicedColumn() throws {
        try requireRealGPU()
        var vals: [Double?] = []
        for i in 0..<3000 {
            if i % 13 == 0 { vals.append(nil) }
            else if i % 97 == 0 { vals.append(Double.nan) }
            else { vals.append(Double((i &* 7919) % 401)) }
        }
        let full = try MetalArray<Double>(vals, context: .shared)
        for off in [1, 31, 32, 33] {
            let s = try full.slice(offset: off, length: 500)
            let standalone = try MetalArray<Double>(Array(vals[off..<(off + 500)]), context: .shared)
            for tb in [RankTiebreaker.first, .min, .dense, .max] {
                for placement in [NullPlacement.atEnd, .atStart] {
                    XCTAssertEqual(try s.rank(tiebreaker: tb, nullPlacement: placement).toRawArray(),
                                   try standalone.rank(tiebreaker: tb, nullPlacement: placement).toRawArray(),
                                   "rank \(tb) \(placement) at offset \(off)")
                }
            }
        }
    }

    /// A column of one row, a column of all-equal rows and an all-null column.
    func testRankDegenerateColumns() throws {
        try requireRealGPU()
        let one = try MetalArray<Double>([7.0], context: .shared)
        for tb in [RankTiebreaker.first, .min, .dense, .max] {
            XCTAssertEqual(try one.rank(tiebreaker: tb).toRawArray(), [1])
        }
        let same = try MetalArray<Double>([Double](repeating: 2.0, count: 5), context: .shared)
        XCTAssertEqual(try same.rank(tiebreaker: .first).toRawArray(), [1, 2, 3, 4, 5])
        XCTAssertEqual(try same.rank(tiebreaker: .min).toRawArray(), [1, 1, 1, 1, 1])
        XCTAssertEqual(try same.rank(tiebreaker: .dense).toRawArray(), [1, 1, 1, 1, 1])
        XCTAssertEqual(try same.rank(tiebreaker: .max).toRawArray(), [5, 5, 5, 5, 5])
        let allNull = try MetalArray<Double>([Double?](repeating: nil, count: 4), context: .shared)
        XCTAssertEqual(try allNull.rank(tiebreaker: .first).toRawArray(), [1, 2, 3, 4])
        XCTAssertEqual(try allNull.rank(tiebreaker: .min).toRawArray(), [1, 1, 1, 1])
        XCTAssertEqual(try allNull.rank(tiebreaker: .dense).toRawArray(), [1, 1, 1, 1])
        XCTAssertEqual(try allNull.rank(tiebreaker: .max).toRawArray(), [4, 4, 4, 4])
        XCTAssertEqual(try MetalArray<Double>([Double](), context: .shared).rank().length, 0)
    }
}
