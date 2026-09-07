import XCTest
@testable import ArrowMetal

/// The grouped aggregates over sliced key and value columns, and the group shapes the counting sort
/// has to get right. A slice carries an Arrow element offset that only a kernel dispatch normalises,
/// so a group-by run over one must agree with the same rows built as columns of their own.
final class AdversarialGroupSliceTests: XCTestCase {

    private struct Column {
        let keys: [Int32?]
        let vals: [Double?]
    }

    private static func sample(_ n: Int, groups K: Int) -> Column {
        var keys: [Int32?] = [], vals: [Double?] = []
        for i in 0..<n {
            keys.append(i % 29 == 0 ? nil : Int32((i &* 7919) % K))
            vals.append(i % 17 == 0 ? nil : Double((i &* 2654435761) % 1009) - 500)
        }
        return Column(keys: keys, vals: vals)
    }

    /// Every grouped aggregate over a slice of both columns, at offsets that are and are not multiples
    /// of 32 (the bitmap-word boundary), against the same rows built standalone.
    func testGroupedAggregatesOverSlicedKeysAndValues() throws {
        try requireRealGPU()
        let n = 6000, K = 64, len = 1500
        let c = Self.sample(n, groups: K)
        let fullKeys = try MetalArray<Int32>(c.keys, context: .shared)
        let fullVals = try MetalArray<Double>(c.vals, context: .shared)
        let ints: [Int32?] = c.vals.map { $0.map { v in Int32(v) } }
        let fullInts = try MetalArray<Int32>(ints, context: .shared)
        for off in [1, 7, 31, 32, 33, 64] {
            let k = try fullKeys.slice(offset: off, length: len)
            let v = try fullVals.slice(offset: off, length: len)
            let sk = try MetalArray<Int32>(Array(c.keys[off..<(off + len)]), context: .shared)
            let sv = try MetalArray<Double>(Array(c.vals[off..<(off + len)]), context: .shared)
            let sliced = try k.groupBy(keyCount: K)
            let standalone = try sk.groupBy(keyCount: K)

            XCTAssertEqual(try sliced.count().toRawArray(), try standalone.count().toRawArray(),
                           "hash_count at offset \(off)")
            // `count(values:)` needs a 32-bit-or-narrower value column (the doc says so), so it gets
            // the same rows as int32.
            let iv = try fullInts.slice(offset: off, length: len)
            let siv = try MetalArray<Int32>(Array(ints[off..<(off + len)]), context: .shared)
            XCTAssertEqual(try sliced.count(iv).toRawArray(), try standalone.count(siv).toRawArray(),
                           "hash_count(values) at offset \(off)")
            XCTAssertEqual(try sliced.sum(iv).toArray(), try standalone.sum(siv).toArray(),
                           "hash_sum at offset \(off)")
            XCTAssertEqual(try sliced.mean(iv).toArray(), try standalone.mean(siv).toArray(),
                           "hash_mean at offset \(off)")
            XCTAssertEqual(try sliced.min(iv).toArray(), try standalone.min(siv).toArray(),
                           "hash_min at offset \(off)")
            XCTAssertEqual(try sliced.max(iv).toArray(), try standalone.max(siv).toArray(),
                           "hash_max at offset \(off)")
            let (mnA, mxA) = try sliced.extrema(v)
            let (mnB, mxB) = try standalone.extrema(sv)
            XCTAssertEqual(mnA.toArray(), mnB.toArray(), "grouped min at offset \(off)")
            XCTAssertEqual(mxA.toArray(), mxB.toArray(), "grouped max at offset \(off)")
            XCTAssertEqual(try sliced.varianceDouble(v, ddof: 1).toArray(),
                           try standalone.varianceDouble(sv, ddof: 1).toArray(),
                           "hash_variance at offset \(off)")
            XCTAssertEqual(try sliced.varianceDouble(v, ddof: 0).toArray(),
                           try standalone.varianceDouble(sv, ddof: 0).toArray(),
                           "hash_variance ddof 0 at offset \(off)")
        }
    }

    /// The grouped aggregates must agree with a plain host reduction, so the slice comparison above is
    /// pinned to a value and not just to itself.
    func testGroupedExtremaAndVarianceAgainstAHostReduction() throws {
        try requireRealGPU()
        let n = 5000, K = 37
        let c = Self.sample(n, groups: K)
        let keys = try MetalArray<Int32>(c.keys, context: .shared)
        let vals = try MetalArray<Double>(c.vals, context: .shared)
        let gb = try keys.groupBy(keyCount: K)

        var byGroup = [[Double]](repeating: [], count: K)
        for i in 0..<n {
            guard let k = c.keys[i], k >= 0, k < Int32(K), let v = c.vals[i] else { continue }
            byGroup[Int(k)].append(v)
        }
        let (mn, mx) = try gb.extrema(vals)
        XCTAssertEqual(mn.toArray(), byGroup.map { $0.min() })
        XCTAssertEqual(mx.toArray(), byGroup.map { $0.max() })

        let variance = try gb.varianceDouble(vals, ddof: 1).toArray()
        for g in 0..<K {
            let xs = byGroup[g]
            guard xs.count > 1 else { XCTAssertNil(variance[g], "group \(g)"); continue }
            let mean = xs.reduce(0, +) / Double(xs.count)
            let want = xs.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(xs.count - 1)
            XCTAssertEqual(variance[g]!, want, accuracy: Swift.max(want * 1e-9, 1e-9), "group \(g)")
        }
    }

    /// Grouped min / max over every integer width, including the 64-bit case the two-pass 32-bit atomic
    /// scheme exists for, and over the extreme values of each.
    func testGroupedExtremaOverEveryWidth() throws {
        try requireRealGPU()
        let keys = try MetalArray<Int32>([0, 0, 1, 1, 2], context: .shared)
        let gb = try keys.groupBy(keyCount: 3)

        let i64 = try MetalArray<Int64>([Int64.min, Int64.max, -1, 1, 0], context: .shared)
        let (mn64, mx64) = try gb.extrema(i64)
        XCTAssertEqual(mn64.toArray(), [Int64.min, -1, 0])
        XCTAssertEqual(mx64.toArray(), [Int64.max, 1, 0])

        let u64 = try MetalArray<UInt64>([UInt64.max, 0, 1 << 63, (1 << 63) - 1, 7], context: .shared)
        let (mnU, mxU) = try gb.extrema(u64)
        XCTAssertEqual(mnU.toArray(), [0, (1 << 63) - 1, 7], "uint64 above 2^63 is the larger")
        XCTAssertEqual(mxU.toArray(), [UInt64.max, 1 << 63, 7])

        let i8 = try MetalArray<Int8>([Int8.min, Int8.max, -1, 1, 0], context: .shared)
        let (mn8, mx8) = try gb.extrema(i8)
        XCTAssertEqual(mn8.toArray(), [Int8.min, -1, 0])
        XCTAssertEqual(mx8.toArray(), [Int8.max, 1, 0])

        let f32 = try MetalArray<Float>([-Float.infinity, Float.infinity, -0.0, 0.0,
                                         Float.leastNonzeroMagnitude], context: .shared)
        let (mnF, mxF) = try gb.extrema(f32)
        XCTAssertEqual(mnF.toArray()[0], -Float.infinity)
        XCTAssertEqual(mxF.toArray()[0], Float.infinity)
        XCTAssertEqual(mnF.toArray()[2], Float.leastNonzeroMagnitude, "a subnormal is not zero")
    }

    /// A key column whose only rows are null or out of range: every group must come back empty, and
    /// nothing may be read past the value column.
    func testEveryKeyDroppedLeavesEveryGroupEmpty() throws {
        try requireRealGPU()
        let keys = try MetalArray<Int32>([nil, -1, 99, nil, 50], context: .shared)
        let vals = try MetalArray<Double>([1, 2, 3, 4, 5], context: .shared)
        let gb = try keys.groupBy(keyCount: 4)
        XCTAssertEqual(try gb.count().toRawArray(), [0, 0, 0, 0])
        let (mn, mx) = try gb.extrema(vals)
        XCTAssertEqual(mn.toArray(), [nil, nil, nil, nil])
        XCTAssertEqual(mx.toArray(), [nil, nil, nil, nil])
        XCTAssertEqual(try gb.varianceDouble(vals, ddof: 1).toArray(), [nil, nil, nil, nil])
    }
}
