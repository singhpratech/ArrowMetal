import XCTest
@testable import ArrowMetal

/// The GPU hash table over primitive keys (`Kernels/HashTable.swift`): the path `unique`,
/// `value_counts`, `count_distinct`, `mode` and `dictionary_encode` take above 65,536 rows.
///
/// `UniqueTests` already checks all five against a Swift oracle for every element type at 300,003 rows,
/// which is over the threshold — so these tests go after what that one does not reach: the threshold
/// itself, float semantics at scale, cardinalities from 1 to a few million, bit patterns chosen to be
/// hostile to the hash, sliced inputs, and the table-growth retry.
final class HashTableTests: XCTestCase {

    /// Distinct values ascending, their counts, and the first-seen codes — the definition all five
    /// functions are measured against.
    func oracle<T: ArrowPrimitive & Hashable & Comparable>(_ vals: [T?]) -> (values: [T], counts: [Int64]) {
        var counts: [T: Int64] = [:]
        for v in vals { if let v { counts[v, default: 0] += 1 } }
        let values = counts.keys.sorted()
        return (values, values.map { counts[$0]! })
    }

    /// Checks every function that routes through the table against the oracle.
    func check<T: ArrowPrimitive & Hashable & Comparable>(_ vals: [T?], _ what: String,
                                                          file: StaticString = #filePath, line: UInt = #line) throws {
        let a = try MetalArray<T>(vals)
        let (expValues, expCounts) = oracle(vals)

        XCTAssertEqual(try a.unique().toArray().map { $0! }, expValues, "\(what): unique", file: file, line: line)
        XCTAssertEqual(try a.countDistinct(), expValues.count, "\(what): count_distinct", file: file, line: line)

        let vc = try a.valueCounts()
        XCTAssertEqual(vc.values.toArray().map { $0! }, expValues, "\(what): value_counts values", file: file, line: line)
        XCTAssertEqual(vc.counts.toArray().map { $0! }, expCounts, "\(what): value_counts counts", file: file, line: line)

        let (codes, dict) = try a.dictionaryEncode()
        XCTAssertEqual(dict.toArray().map { $0! }, expValues, "\(what): dictionary", file: file, line: line)
        let d = dict.toArray()
        let decoded = codes.toArray().map { $0.map { d[Int($0)]! } }
        XCTAssertEqual(decoded, vals, "\(what): codes decode to the input", file: file, line: line)

        if let best = expCounts.max() {
            let m = try XCTUnwrap(try a.mode(), "\(what): mode", file: file, line: line)
            XCTAssertEqual(m.count, best, "\(what): mode count", file: file, line: line)
            // Ties go to the smallest value, as Arrow does.
            XCTAssertEqual(m.value, expValues[expCounts.firstIndex(of: best)!], "\(what): mode value", file: file, line: line)
        } else {
            XCTAssertNil(try a.mode(), "\(what): mode of an empty or all-null column", file: file, line: line)
        }
    }

    // MARK: - the threshold

    /// The hash table takes over at 1 << 16 rows. Both sides of that line must agree, for every type.
    func testBothSidesOfTheThreshold() throws {
        try requireRealGPU()
        for n in [(1 << 16) - 1, 1 << 16, (1 << 16) + 3] {
            func mk<T>(_ f: (Int) -> T) -> [T?] { (0..<n).map { $0 % 17 == 5 ? nil : f($0) } }
            try check(mk { Int8(truncatingIfNeeded: $0 % 251) }, "Int8 n=\(n)")
            try check(mk { UInt8(truncatingIfNeeded: $0) }, "UInt8 n=\(n)")
            try check(mk { Int16(truncatingIfNeeded: $0 % 4000 - 2000) }, "Int16 n=\(n)")
            try check(mk { UInt16(truncatingIfNeeded: $0 % 5000) }, "UInt16 n=\(n)")
            try check(mk { Int32($0 % 900) - 400 }, "Int32 n=\(n)")
            try check(mk { UInt32($0 % 900) }, "UInt32 n=\(n)")
            try check(mk { Int64($0 % 900) &* 1_000_000_007 &- 100 }, "Int64 n=\(n)")
            try check(mk { UInt64($0 % 900) &* 0x0123_4567_89AB_CDEF }, "UInt64 n=\(n)")
            try check(mk { Float($0 % 700) / 4 - 50 }, "Float n=\(n)")
            try check(mk { Double($0 % 700) / 8 - 50 }, "Double n=\(n)")
        }
    }

    // MARK: - cardinality

    func testCardinalitiesAtOneMillionRows() throws {
        try requireRealGPU()
        let n = 1_000_003
        var state: UInt64 = 99
        func next(_ m: Int) -> Int { state = state &* 6_364_136_223_846_793_005 &+ 1; return Int((state >> 33) % UInt64(m)) }
        for distinct in [1, 7, 1000, 100_000] {
            let vals: [Int64?] = (0..<n).map { i in i % 29 == 11 ? nil : Int64(next(distinct)) &* 7 &- 3 }
            try check(vals, "Int64 n=\(n) distinct=\(distinct)")
        }
    }

    /// Millions of distinct values: the table is far too big for cache, the growth path is exercised for
    /// real, and the ranking scan runs over tens of millions of slots.
    func testManyDistinctValues() throws {
        try requireRealGPU()
        let n = 5_000_003
        let distinct = 3_000_000
        var state: UInt64 = 4242
        func next(_ m: Int) -> Int { state = state &* 6_364_136_223_846_793_005 &+ 1; return Int((state >> 33) % UInt64(m)) }
        var seen = [Bool](repeating: false, count: distinct)
        var vals = [Int64](repeating: 0, count: n)
        for i in 0..<n {
            let c = next(distinct)
            seen[c] = true
            vals[i] = Int64(c) &* 11 &- 5
        }
        let expected = seen.filter { $0 }.count
        let a = try MetalArray<Int64>(vals)
        XCTAssertEqual(try a.countDistinct(), expected)
        let u = try a.unique()
        XCTAssertEqual(u.length, expected)
        // Ascending, and every value is one of the generated ones.
        let up = u.valuePointer
        var ordered = true
        for i in 1..<u.length where up[i] <= up[i - 1] { ordered = false; break }
        XCTAssertTrue(ordered, "unique must be ascending")
        XCTAssertEqual(up[0], Int64(seen.firstIndex(of: true)!) &* 11 &- 5)
        let vc = try a.valueCounts()
        XCTAssertEqual(vc.counts.length, expected)
        var total: Int64 = 0
        let cp = vc.counts.valuePointer
        for i in 0..<vc.counts.length { total += cp[i] }
        XCTAssertEqual(total, Int64(n), "the counts must add up to the row count")
    }

    // MARK: - hostile inputs

    /// Arrow float equality at a size that takes the hash table: every NaN is one value sorted last,
    /// `-0.0` equals `0.0`, and the representative comes from the earliest row.
    func testFloatSemanticsAtScale() throws {
        try requireRealGPU()
        let n = 200_003
        var f = [Float?](repeating: nil, count: n)
        for i in 0..<n {
            switch i % 7 {
            case 0: f[i] = -0.0
            case 1: f[i] = 0.0
            case 2: f[i] = Float.nan
            case 3: f[i] = -Float.nan
            case 4: f[i] = .infinity
            case 5: f[i] = -.infinity
            default: f[i] = nil
            }
        }
        let a = try MetalArray<Float>(f)
        let u = try a.unique().toArray().map { $0! }
        XCTAssertEqual(u.count, 4, "-0/0 are one value, both NaNs are one value")
        XCTAssertEqual(u[0], -.infinity)
        XCTAssertEqual(u[1], 0)
        XCTAssertTrue(u[1].sign == .minus, "the representative is the earliest row, which held -0.0")
        XCTAssertEqual(u[2], .infinity)
        XCTAssertTrue(u[3].isNaN, "NaN sorts last")
        let vc = try a.valueCounts()
        let nonNull = Int64(f.filter { $0 != nil }.count)
        XCTAssertEqual(vc.counts.toArray().map { $0! }.reduce(0, +), nonNull)

        // The same for Float64, whose values Metal only ever sees as raw bit patterns.
        let d: [Double?] = (0..<n).map { i in
            switch i % 5 {
            case 0: return -0.0
            case 1: return Double.nan
            case 2: return 0.0
            case 3: return 1e308
            default: return nil
            }
        }
        let da = try MetalArray<Double>(d)
        let du = try da.unique().toArray().map { $0! }
        XCTAssertEqual(du.count, 3)
        XCTAssertEqual(du[0], 0)
        XCTAssertTrue(du[0].sign == .minus)
        XCTAssertEqual(du[1], 1e308)
        XCTAssertTrue(du[2].isNaN)
    }

    /// Keys chosen to be hostile: identical low bits, identical high bits, the extremes of the range.
    func testAdversarialBitPatterns() throws {
        try requireRealGPU()
        let n = 300_003
        // Every key a multiple of 2^20: the low 20 bits are all zero, so a table indexed by raw low bits
        // would put every one of them in one bucket.
        try check((0..<n).map { Int64($0 % 5000) << 20 as Int64? }, "low bits all zero")
        // Only the low bits differ, the high half is constant.
        try check((0..<n).map { (Int64(0x7FFF_0000) << 32) | Int64($0 % 5000) as Int64? }, "high bits constant")
        // Both ends of the type, plus zero, plus the sign boundary.
        let extremes: [Int64] = [.min, .min + 1, -1, 0, 1, .max - 1, .max]
        try check((0..<n).map { extremes[$0 % extremes.count] as Int64? }, "extremes of Int64")
        let uExtremes: [UInt64] = [0, 1, UInt64(Int64.max), UInt64(Int64.max) + 1, .max - 1, .max]
        try check((0..<n).map { uExtremes[$0 % uExtremes.count] as UInt64? }, "extremes of UInt64")
        // Every row the same value, and every row null.
        try check([Int64?](repeating: 12345, count: n), "one value")
        try check([Int64?](repeating: nil, count: n), "all null")
        // Every row distinct: the table is at its worst load factor.
        try check((0..<n).map { Int64($0) &* 2_654_435_761 as Int64? }, "all distinct")
    }

    func testSlicedInput() throws {
        try requireRealGPU()
        let full = try MetalArray<Int32>((0..<300_003).map { $0 % 13 == 0 ? nil : Int32($0 % 977) })
        let s = try full.slice(offset: 65, length: 200_003)
        try check(s.toArray(), "sliced Int32")
    }

    /// Starting from a table far too small forces the growth retry, which must land on the same answer.
    func testForcedGrowthRetry() throws {
        try requireRealGPU()
        let n = 200_003
        let vals: [Int64?] = (0..<n).map { (i: Int) -> Int64? in
            if i % 31 == 7 { return nil }
            return Int64(i % 100_000) &* 3
        }
        let a = try MetalArray<Int64>(vals)
        let (expValues, _) = oracle(vals)
        // 1024 slots for ~100k distinct keys: three growth steps of eight before the table is big enough.
        let groups = try XCTUnwrap(try a.hashGroups(initialSlots: 1024))
        XCTAssertEqual(groups.0.groupCount, expValues.count, "group count after the growth retry")
        let d = try XCTUnwrap(try a.hashDistinct(initialSlots: 1024))
        XCTAssertEqual(d.values.toArray().map { $0! }, expValues, "values after the growth retry")
    }

    /// The hash path and the sort path are the same function. Below the threshold every call takes the
    /// sort; the same data above it takes the table; the answers must not move.
    func testHashAndSortPathsAgree() throws {
        try requireRealGPU()
        var base = [Int64?]()
        for i in 0..<50_000 {
            let v: Int64 = Int64(i % 733) &* 7 &- 11
            base.append(i % 19 == 3 ? nil : v)
        }
        let small = try MetalArray<Int64>(base)
        XCTAssertFalse(MetalArray<Int64>.prefersHashTable(rows: small.length), "50k rows takes the sort path")
        let big = try MetalArray<Int64>(base + base)                    // 100k rows, same value set
        XCTAssertTrue(MetalArray<Int64>.prefersHashTable(rows: big.length), "100k rows takes the hash path")
        XCTAssertEqual(try small.unique().toArray(), try big.unique().toArray())
        XCTAssertEqual(try small.countDistinct(), try big.countDistinct())
        let (sv, sc) = try small.valueCounts()
        let (bv, bc) = try big.valueCounts()
        XCTAssertEqual(sv.toArray(), bv.toArray())
        let smallCounts: [Int64] = sc.toArray().map { $0! * 2 }
        let bigCounts: [Int64] = bc.toArray().map { $0! }
        XCTAssertEqual(bigCounts, smallCounts)
        XCTAssertEqual(try small.dictionaryEncode().unique.toArray(), try big.dictionaryEncode().unique.toArray())
        XCTAssertEqual(try small.mode()?.value, try big.mode()?.value)
    }
}
