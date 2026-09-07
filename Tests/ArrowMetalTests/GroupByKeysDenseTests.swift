import XCTest
@testable import ArrowMetal

/// Several integer key columns whose ranges multiply out small enough are packed into one key in a
/// single pass (`Kernels/GroupByKeysDense.swift`) instead of folded pairwise. The claim these tests
/// make is the strong one: the fused path hands back the *same ids in the same order* as the fold, so
/// nothing downstream — group order, `groupKeys()`, any aggregate — can tell which one ran.
final class GroupByKeysDenseTests: XCTestCase {

    /// The ids and cardinality the pairwise fold produces, which is what `GroupByKeys` did before the
    /// fused path existed: one range or sort encoding per column, combined into an int64 composite and
    /// re-encoded after every fold.
    func foldIds(_ columns: [AnyMetalArray]) throws -> (ids: [Int32?], count: Int) {
        let ctx = MetalContext.shared
        var (ids, K) = try GroupByKeys.denseIds(columns[0], ctx)
        for c in columns.dropFirst() {
            let (ids2, K2) = try GroupByKeys.denseIds(c, ctx)
            let composite = try GroupByKeys.combine(ids, ids2, cardinality: K2, ctx)
            (ids, K) = try GroupByKeys.encodeComposite(composite, span: K * K2, ctx)
        }
        return (ids.toArray(), K)
    }

    /// Asserts the fused path ran (or deliberately did not), and that either way the ids the public
    /// `GroupByKeys` hands out are the fold's.
    func check(_ columns: [AnyMetalArray], fused wantFused: Bool, _ label: String) throws {
        let ctx = MetalContext.shared
        let dense = try GroupByKeysDense.ids(columns, ctx)
        XCTAssertEqual(dense != nil, wantFused, "fused path taken? \(label)")
        let gbk = try GroupByKeys(columns: columns)
        guard columns[0].length > 0 else {
            XCTAssertEqual(gbk.groupCount, 0, label)
            return
        }
        let expected = try foldIds(columns)
        XCTAssertEqual(gbk.groupCount, expected.count, "group count, \(label)")
        XCTAssertEqual(gbk.ids.toArray(), expected.ids, "ids, \(label)")
        if let dense {
            XCTAssertEqual(dense.1, expected.count, "fused group count, \(label)")
            XCTAssertEqual(dense.0.toArray(), expected.ids, "fused ids, \(label)")
        }
    }

    func testFusedIdsMatchTheFold() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        for n in [0, 1, 2, 33, 4097, 100_003] {
            var a: [Int32?] = [], b: [Int32?] = [], c: [Int16?] = []
            for i in 0..<n {
                a.append(i % 11 == 3 ? nil : Int32.random(in: -5...5, using: &g))
                b.append(i % 7 == 2 ? nil : Int32.random(in: 100...131, using: &g))
                c.append(Int16.random(in: 0...3, using: &g))
            }
            let ca = AnyMetalArray.int32(try MetalArray<Int32>(a))
            let cb = AnyMetalArray.int32(try MetalArray<Int32>(b))
            let cc = AnyMetalArray.int16(try MetalArray<Int16>(c))
            try check([ca, cb], fused: n > 0, "two int columns, n=\(n)")
            try check([ca, cb, cc], fused: n > 0, "three int columns, n=\(n)")
        }
    }

    /// The shapes that break a packed key if anything is off by one: nothing to group, one row, every
    /// key null, every key the same, and one group per row. A column with no valid value at all has no
    /// range for `minMax` to report, so it is the fold that answers those.
    func testFusedEdgeShapes() throws {
        try requireRealGPU()
        let n = 5000
        let allNullA = AnyMetalArray.int32(try MetalArray<Int32>([Int32?](repeating: nil, count: n)))
        let allNullB = AnyMetalArray.int64(try MetalArray<Int64>([Int64?](repeating: nil, count: n)))
        try check([allNullA, allNullB], fused: false, "both columns all null")

        let equalA = AnyMetalArray.int32(try MetalArray<Int32>([Int32](repeating: 7, count: n)))
        let equalB = AnyMetalArray.int32(try MetalArray<Int32>([Int32](repeating: -9, count: n)))
        try check([equalA, equalB], fused: true, "both columns constant")
        try check([allNullA, equalB], fused: false, "one all-null column, one constant")

        // A column that is null everywhere but one row still has a range, so it is fused.
        var sparse = [Int32?](repeating: nil, count: n)
        sparse[n / 2] = 4
        try check([AnyMetalArray.int32(try MetalArray<Int32>(sparse)), equalB], fused: true,
                  "one valid key among nulls")

        // One group per row: 5000 slots by one, still inside the bound.
        let perRow = AnyMetalArray.int32(try MetalArray<Int32>((0..<Int32(n)).map { $0 }))
        try check([perRow, equalB], fused: true, "one group per row")

        // One row, and no rows at all.
        try check([AnyMetalArray.int32(try MetalArray<Int32>([5] as [Int32])),
                   AnyMetalArray.int32(try MetalArray<Int32>([3] as [Int32]))], fused: true, "one row")
        try check([AnyMetalArray.int32(try MetalArray<Int32>([Int32]())),
                   AnyMetalArray.int32(try MetalArray<Int32>([Int32]()))], fused: false, "no rows")
    }

    /// The bound itself: a packed range that just fits is fused, one value wider falls back to the
    /// fold, and both give the same answer.
    func testFusedRangeBound() throws {
        try requireRealGPU()
        let n = 4096
        // The row count is below 2^16, so the bound is max(1 << 16, rows) = 65,536 slots: a 256 by 256
        // range fits exactly and a 257 by 256 one does not.
        func columns(_ spanA: Int32, _ spanB: Int32) throws -> [AnyMetalArray] {
            let a = (0..<n).map { Int32($0) % spanA }
            let b = (0..<n).map { (Int32($0) &* 7) % spanB }
            return [.int32(try MetalArray<Int32>(a)), .int32(try MetalArray<Int32>(b))]
        }
        try check(try columns(256, 256), fused: true, "65,536 slots, at the bound")
        try check(try columns(257, 256), fused: false, "65,792 slots, over the bound")

        // The hard 2^24 cap sits above the row-count arm, so it only decides for inputs far larger than
        // a test should allocate; it is asserted on the predicate itself. 4096 by 4096 is exactly 2^24
        // and fits; one value more in either column does not, however many rows there are.
        XCTAssertTrue(GroupByKeysDense.worthIt(span: 4096 * 4096, rows: 50_000_000))
        XCTAssertFalse(GroupByKeysDense.worthIt(span: 4097 * 4096, rows: 50_000_000))
        XCTAssertFalse(GroupByKeysDense.worthIt(span: 1 << 24, rows: 1_000_000))
        XCTAssertFalse(GroupByKeysDense.worthIt(span: 0, rows: 1_000_000))
        XCTAssertTrue(GroupByKeysDense.worthIt(span: 1 << 16, rows: 1))
    }

    /// Booleans, temporal values and dictionary codes are integers underneath, and the packed key
    /// unwraps them exactly as the single-column range path already does.
    func testFusedUnwrapsBooleanAndTemporal() throws {
        try requireRealGPU()
        let n = 3000
        let bits: [Bool] = (0..<n).map { i in i % 3 == 0 }
        let flags = try MetalBooleanArray(bits)
        let days = try MetalArray<Int32>((0..<n).map { Int32($0 % 40) })
        try check([.boolean(flags), .int32(days)], fused: true, "boolean and int32")
    }
}
