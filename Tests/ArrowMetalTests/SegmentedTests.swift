import XCTest
@testable import ArrowMetal

/// Sort-based segmented group-by aggregates, checked element for element against a Swift oracle.
final class SegmentedTests: XCTestCase {
    /// Rows of each key: a null key, or a key outside `[0, K)`, contributes to nothing.
    private func groups(_ keys: [Int32?], K: Int) -> [[Int]] {
        var g = [[Int]](repeating: [], count: K)
        for (i, k) in keys.enumerated() { if let k, k >= 0, Int(k) < K { g[Int(k)].append(i) } }
        return g
    }

    /// Deterministic pseudo-random source, so a failure can be reproduced from the seed alone.
    private struct Rng: RandomNumberGenerator {
        var s: UInt64
        mutating func next() -> UInt64 {
            s &+= 0x9E37_79B9_7F4A_7C15
            var z = s
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    func testSumMeanMinMaxAgainstOracle() throws {
        try requireRealGPU()
        var rng = Rng(s: 0x5EED)
        for K in [1, 5, 1000, 100_000] {
            for n in [0, 1, 1000, 200_000] {
                var keys: [Int32?] = [], dvals: [Double?] = [], ivals: [Int64?] = [], uvals: [UInt64?] = []
                for _ in 0..<n {
                    keys.append(Int.random(in: 0..<20, using: &rng) == 0 ? nil : Int32.random(in: 0..<Int32(K), using: &rng))
                    let drop = Int.random(in: 0..<10, using: &rng) == 0
                    dvals.append(drop ? nil : Double.random(in: -1e6...1e6, using: &rng))
                    ivals.append(drop ? nil : Int64.random(in: Int64.min...Int64.max, using: &rng))
                    uvals.append(drop ? nil : UInt64.random(in: 0...UInt64.max, using: &rng))
                }
                if n > 10 { keys[2] = Int32(K) + 5; keys[3] = -1; dvals[4] = .nan }   // out of range, NaN
                let gb = try MetalArray<Int32>(keys).groupBy(keyCount: K)
                let seg = try gb.segments()
                let g = groups(keys, K: K)

                let dv = try MetalArray<Double>(dvals)
                let sums = try gb.sumDouble(dv, segments: seg).toArray()
                let means = try gb.meanDouble(dv, segments: seg).toArray()
                let dmin = try gb.min64(dv, segments: seg).toArray()
                let dmax = try gb.max64(dv, segments: seg).toArray()
                let iv = try MetalArray<Int64>(ivals)
                let imin = try gb.min64(iv, segments: seg).toArray()
                let imax = try gb.max64(iv, segments: seg).toArray()
                let uv = try MetalArray<UInt64>(uvals)
                let umin = try gb.min64(uv, segments: seg).toArray()
                let umax = try gb.max64(uv, segments: seg).toArray()

                for k in 0..<K {
                    let ds = g[k].compactMap { dvals[$0] }
                    let label = "K=\(K) n=\(n) k=\(k)"
                    if ds.isEmpty {
                        XCTAssertNil(sums[k], "empty sum \(label)")
                        XCTAssertNil(means[k], "empty mean \(label)")
                        XCTAssertNil(dmin[k], "empty min \(label)")
                    } else {
                        let exp = ds.reduce(0.0, +), scale = ds.reduce(0.0) { $0 + abs($1) }
                        if exp.isNaN {
                            XCTAssertTrue(sums[k]!.isNaN, "nan sum \(label)")
                            XCTAssertTrue(means[k]!.isNaN, "nan mean \(label)")
                        } else {
                            XCTAssertEqual(sums[k]!, exp, accuracy: 1e-12 * Swift.max(scale, 1), "sum \(label)")
                            XCTAssertEqual(means[k]!, exp / Double(ds.count),
                                           accuracy: 1e-12 * Swift.max(scale, 1) / Double(ds.count), "mean \(label)")
                        }
                        let nonNaN = ds.filter { !$0.isNaN }
                        XCTAssertEqual(dmin[k], nonNaN.min(), "f64 min \(label)")
                        XCTAssertEqual(dmax[k], nonNaN.max(), "f64 max \(label)")
                    }
                    let ints = g[k].compactMap { ivals[$0] }
                    XCTAssertEqual(imin[k], ints.min(), "i64 min \(label)")
                    XCTAssertEqual(imax[k], ints.max(), "i64 max \(label)")
                    let uints = g[k].compactMap { uvals[$0] }
                    XCTAssertEqual(umin[k], uints.min(), "u64 min \(label)")
                    XCTAssertEqual(umax[k], uints.max(), "u64 max \(label)")
                }
            }
        }
    }

    func testFloat32MeanAndSum() throws {
        try requireRealGPU()
        var rng = Rng(s: 0xC0FFEE)
        for K in [1, 5, 1000] {
            let n = 100_000
            let keys: [Int32?] = (0..<n).map { _ in Int32.random(in: 0..<Int32(K), using: &rng) }
            var vals: [Float?] = (0..<n).map { _ in Float.random(in: -1000...1000, using: &rng) }
            vals[7] = nil
            let gb = try MetalArray<Int32>(keys).groupBy(keyCount: K)
            let seg = try gb.segments()
            let fv = try MetalArray<Float>(vals)
            let sums = try gb.sumFloatAsDouble(fv, segments: seg).toArray()
            let means = try gb.meanFloat(fv, segments: seg).toArray()
            let g = groups(keys, K: K)
            for k in 0..<K {
                let members = g[k].compactMap { vals[$0] }.map { Double($0) }
                if members.isEmpty { XCTAssertNil(sums[k]); XCTAssertNil(means[k]); continue }
                let exp = members.reduce(0.0, +), scale = members.reduce(0.0) { $0 + abs($1) }
                XCTAssertEqual(sums[k]!, exp, accuracy: 1e-12 * Swift.max(scale, 1), "sum K=\(K) k=\(k)")
                XCTAssertEqual(means[k]!, exp / Double(members.count),
                               accuracy: 1e-12 * Swift.max(scale, 1) / Double(members.count), "mean K=\(K) k=\(k)")
            }
        }
    }

    /// Exact 64-bit edges, all-null and all-NaN groups, and empty groups between populated ones.
    func testEdges() throws {
        try requireRealGPU()
        let keys = try MetalArray<Int32>([0, 0, 1, 3, 3, nil, 4, 4])
        let gb = try keys.groupBy(keyCount: 6)
        let i64 = try MetalArray<Int64>([Int64.min, Int64.max, 7, nil, nil, 5, -1, 0])
        XCTAssertEqual(try gb.min64(i64).toArray(), [Int64.min, 7, nil, nil, -1, nil])
        XCTAssertEqual(try gb.max64(i64).toArray(), [Int64.max, 7, nil, nil, 0, nil])
        let u64 = try MetalArray<UInt64>([0, UInt64.max, 9, 1, 2, 3, 8, 8])
        XCTAssertEqual(try gb.min64(u64).toArray(), [0, 9, nil, 1, 8, nil])
        XCTAssertEqual(try gb.max64(u64).toArray(), [UInt64.max, 9, nil, 2, 8, nil])
        let f64 = try MetalArray<Double>([1.5, -2.5, .nan, 3.0, .infinity, 0, -0.0, 4.0])
        XCTAssertEqual(try gb.sumDouble(f64).toArray()[0], -1.0)
        XCTAssertNil(try gb.min64(f64).toArray()[1])                    // the only value is NaN
        XCTAssertEqual(try gb.max64(f64).toArray()[3], .infinity)
        XCTAssertEqual(try gb.meanDouble(f64).toArray()[0], -0.5)
        XCTAssertEqual(try gb.meanDouble(f64).toArray()[4], 2.0)
        XCTAssertNil(try gb.sumDouble(f64).toArray()[2])                // key 2 has no rows
        XCTAssertNil(try gb.sumDouble(f64).toArray()[5])
        // Narrow types forward to the atomic min/max rather than throwing.
        let i32 = try MetalArray<Int32>([5, -7, 9, 1, 2, 3, 4, 4])
        XCTAssertEqual(try gb.min64(i32).toArray(), [-7, 9, nil, 1, 4, nil])
        XCTAssertThrowsError(try gb.sumDouble(try MetalArray<Double>([1, 2])))
    }

    /// One key holding every row: the whole array is a single segment.
    func testSingleAndDegenerateSegments() throws {
        try requireRealGPU()
        let n = 250_000
        let keys = try MetalArray<Int32>([Int32](repeating: 0, count: n))
        let gb = try keys.groupBy(keyCount: 1)
        let vals = try MetalArray<Double>((0..<n).map { Double($0) * 0.5 })
        let exp = (0..<n).reduce(0.0) { $0 + Double($1) * 0.5 }
        XCTAssertEqual(try gb.sumDouble(vals).toArray()[0]!, exp, accuracy: 1e-12 * exp)
        XCTAssertEqual(try gb.meanDouble(vals).toArray()[0]!, exp / Double(n), accuracy: 1e-9)
        // Every key null: every group empty.
        let allNull = try MetalArray<Int32>([Int32?](repeating: nil, count: 100))
        let gb2 = try allNull.groupBy(keyCount: 3)
        XCTAssertEqual(try gb2.sumDouble(try MetalArray<Double>((0..<100).map { Double($0) })).toArray(), [nil, nil, nil])
    }
}
