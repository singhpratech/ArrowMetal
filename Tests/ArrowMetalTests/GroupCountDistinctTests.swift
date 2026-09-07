import XCTest
@testable import ArrowMetal

/// `hash_count_distinct` over a GPU hash set of the (group, value) pair
/// (`Kernels/GroupCountDistinct.swift`), against a Swift `Set` oracle and against the packed-key path
/// it replaced, which `ARROWMETAL_NO_HASH=1` still selects.
final class GroupCountDistinctTests: XCTestCase {

    /// The Swift oracle: distinct non-null values per key, over the rows whose key is in range.
    func oracle<T: Hashable>(_ keys: [Int32?], _ values: [T?], keyCount: Int) -> [Int64] {
        var sets = [Set<T>](repeating: [], count: keyCount)
        for (i, k) in keys.enumerated() {
            guard let k, k >= 0, Int(k) < keyCount, let v = values[i] else { continue }
            sets[Int(k)].insert(v)
        }
        return sets.map { Int64($0.count) }
    }

    func testGroupedCountDistinctMatchesTheOracle() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        // 300,003 rows is past the 2^17 threshold where the table sizes itself from a pilot pass.
        for n in [0, 1, 2, 33, 4097, 300_003] {
            for keyCount in [1, 7, 1021] {
                var keys: [Int32?] = [], values: [Int64?] = []
                for i in 0..<n {
                    // Every fifth key null, every seventh out of range: neither is ever inserted.
                    if i % 5 == 0 { keys.append(nil) }
                    else if i % 7 == 0 { keys.append(Int32(keyCount + i % 3)) }
                    else { keys.append(Int32.random(in: 0..<Int32(keyCount), using: &g)) }
                    values.append(i % 9 == 4 ? nil : Int64.random(in: -20...20, using: &g))
                }
                let gb = try GroupBy(keys: try MetalArray<Int32>(keys), keyCount: keyCount)
                let got = try gb.countDistinct(try MetalArray<Int64>(values))
                XCTAssertEqual(got.length, keyCount, "n=\(n) K=\(keyCount)")
                XCTAssertEqual(got.nullCount, 0, "n=\(n) K=\(keyCount)")
                XCTAssertEqual(got.toArray().map { $0 ?? -1 }, oracle(keys, values, keyCount: keyCount),
                               "n=\(n) K=\(keyCount)")
            }
        }
    }

    /// The shapes that break a set: nothing at all, one row, every value null, every value equal, and
    /// one group per row with a distinct value in each.
    func testGroupedCountDistinctEdgeShapes() throws {
        try requireRealGPU()
        let n = 5000

        let empty = try GroupBy(keys: try MetalArray<Int32>([Int32]()), keyCount: 4)
        XCTAssertEqual(try empty.countDistinct(try MetalArray<Int64>([Int64]())).toArray(), [0, 0, 0, 0])

        let one = try GroupBy(keys: try MetalArray<Int32>([2] as [Int32]), keyCount: 4)
        XCTAssertEqual(try one.countDistinct(try MetalArray<Int64>([9] as [Int64])).toArray(), [0, 0, 1, 0])
        XCTAssertEqual(try one.countDistinct(try MetalArray<Int64>([nil] as [Int64?])).toArray(), [0, 0, 0, 0])

        let keys = try MetalArray<Int32>((0..<n).map { Int32($0 % 8) })
        let gb = try GroupBy(keys: keys, keyCount: 8)
        XCTAssertEqual(try gb.countDistinct(try MetalArray<Int64>([Int64?](repeating: nil, count: n))).toArray(),
                       [Int64](repeating: 0, count: 8))
        XCTAssertEqual(try gb.countDistinct(try MetalArray<Int64>([Int64](repeating: 42, count: n))).toArray(),
                       [Int64](repeating: 1, count: 8))
        XCTAssertEqual(try gb.countDistinct(try MetalArray<Int64>((0..<n).map { Int64($0) })).toArray(),
                       [Int64](repeating: Int64(n / 8), count: 8))

        // One group per row: every group sees exactly one value.
        let perRow = try GroupBy(keys: try MetalArray<Int32>((0..<Int32(n)).map { $0 }), keyCount: n)
        XCTAssertEqual(try perRow.countDistinct(try MetalArray<Int64>((0..<n).map { Int64($0 % 3) })).toArray(),
                       [Int64](repeating: 1, count: n))
    }

    /// Float value semantics: all NaNs are one value, `-0.0` is `0.0`, and the two infinities are two
    /// values — the normalisation `unique()` and `dictionaryEncode()` already apply.
    func testGroupedCountDistinctFloatSemantics() throws {
        try requireRealGPU()
        let values: [Double] = [0.0, -0.0, .nan, -Double.nan, .infinity, -.infinity, 1.5, 1.5]
        let keys = try MetalArray<Int32>([Int32](repeating: 0, count: values.count))
        let gb = try GroupBy(keys: keys, keyCount: 1)
        // 0.0 (with -0.0), one NaN, +inf, -inf, 1.5 = five distinct values.
        XCTAssertEqual(try gb.countDistinct(try MetalArray<Double>(values)).toArray(), [5])

        let f32 = values.map { Float($0) }
        XCTAssertEqual(try gb.countDistinct(try MetalArray<Float>(f32)).toArray(), [5])
    }

    /// Int64 keys and UInt32 keys go through the same kernel with the key column read as its own type.
    func testGroupedCountDistinctWideAndUnsignedKeys() throws {
        try requireRealGPU()
        let n = 4096
        let want = [Int64](repeating: Int64(n / 16 > 7 ? 7 : n / 16), count: 16)
        let vals = try MetalArray<Int32>((0..<n).map { Int32($0 % 7) })
        let wide = try GroupBy(keys: try MetalArray<Int64>((0..<n).map { Int64($0 % 16) }), keyCount: 16)
        XCTAssertEqual(try wide.countDistinct(vals).toArray(), want)
        let unsigned = try GroupBy(keys: try MetalArray<UInt32>((0..<n).map { UInt32($0 % 16) }), keyCount: 16)
        XCTAssertEqual(try unsigned.countDistinct(vals).toArray(), want)
    }
}
