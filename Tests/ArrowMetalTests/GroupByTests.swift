import XCTest
@testable import ArrowMetal

final class GroupByTests: XCTestCase {
    func refSum(_ keys: [Int32?], _ vals: [Int64?], K: Int) -> ([Int64?], [Int64]) {
        var sums = [Int64](repeating: 0, count: K), counts = [Int64](repeating: 0, count: K), rows = [Int64](repeating: 0, count: K)
        for (k, v) in zip(keys, vals) {
            guard let k, k >= 0, Int(k) < K else { continue }
            rows[Int(k)] += 1
            guard let v else { continue }
            sums[Int(k)] &+= v; counts[Int(k)] += 1
        }
        return ((0..<K).map { counts[$0] > 0 ? sums[$0] : nil }, rows)
    }

    func testSumCountMeanAcrossKeyCounts() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        for K in [1, 5, 1000, 1024, 1025, 5000, 100_000] {
            for n in [0, 1, 1000, 300_000] {
                var keys: [Int32?] = [], vals: [Int64?] = []
                for _ in 0..<n {
                    keys.append(Int.random(in: 0..<20, using: &g) == 0 ? nil : Int32.random(in: 0..<Int32(K), using: &g))
                    vals.append(Int.random(in: 0..<10, using: &g) == 0 ? nil : Int64.random(in: -1_000_000...1_000_000, using: &g))
                }
                if n > 10 { keys[2] = Int32(K) + 5; keys[3] = -1 }   // out of range: skipped
                let gb = try MetalArray<Int32>(keys).groupBy(keyCount: K)
                let values = try MetalArray<Int64>(vals)
                let (expSum, expRows) = refSum(keys, vals, K: K)
                XCTAssertEqual(try gb.sum(values).toArray(), expSum, "sum K=\(K) n=\(n)")
                XCTAssertEqual(try gb.count().toRawArray(), expRows, "count K=\(K) n=\(n)")
                let m = try gb.mean(values).toArray()
                var validPerKey = [Int](repeating: 0, count: K)
                for (k, v) in zip(keys, vals) { if let k, k >= 0, Int(k) < K, v != nil { validPerKey[Int(k)] += 1 } }
                for k in 0..<K {
                    if let e = expSum[k] { XCTAssertEqual(m[k]!, Double(e) / Double(validPerKey[k]), accuracy: 1e-9) } else { XCTAssertNil(m[k]) }
                }
            }
        }
    }

    func testSumCarryAndWrapping() throws {
        try requireRealGPU()
        // Values whose low 32 bits overflow many times, plus wrap past Int64.max.
        let keys = try MetalArray<Int32>([Int32](repeating: 0, count: 5000) + [1, 1])
        let vals = try MetalArray<Int64>([Int64](repeating: 0xFFFF_FFFF, count: 5000) + [Int64.max, 1])
        let s = try keys.groupBy(keyCount: 2).sum(vals).toArray()
        XCTAssertEqual(s[0], 5000 * 0xFFFF_FFFF)
        XCTAssertEqual(s[1], Int64.min)
        let neg = try MetalArray<Int64>([Int64](repeating: -3, count: 5000) + [-5, 2])
        XCTAssertEqual(try keys.groupBy(keyCount: 2).sum(neg).toArray(), [-15000, -3])
        let u = try MetalArray<UInt64>([UInt64](repeating: UInt64.max / 2, count: 5000) + [7, 8])
        XCTAssertEqual(try keys.groupBy(keyCount: 2).sum(u).toArray(), [Int64(bitPattern: (UInt64.max / 2) &* 5000), 15])
    }

    func testFloatSumMinMax() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        for K in [3, 2000] {
            let n = 200_000
            let keys: [Int32] = (0..<n).map { _ in Int32.random(in: 0..<Int32(K), using: &g) }
            var vals: [Float?] = (0..<n).map { _ in Float.random(in: -100...100, using: &g) }
            vals[5] = nil; vals[6] = .nan
            let gb = try MetalArray<Int32>(keys).groupBy(keyCount: K)
            let fv = try MetalArray<Float>(vals)
            let sums = try gb.sumFloat(fv).toArray()
            let mins = try gb.min(fv).toArray(), maxs = try gb.max(fv).toArray()
            for k in 0..<K {
                let members = (0..<n).filter { keys[$0] == Int32(k) }.compactMap { vals[$0] }
                let nonNaN = members.filter { !$0.isNaN }
                if members.isEmpty { XCTAssertNil(sums[k]); continue }
                let exp = members.reduce(0.0) { $0 + Double($1) }
                if exp.isNaN { XCTAssertTrue(sums[k]!.isNaN) } else { XCTAssertEqual(sums[k]!, exp, accuracy: Swift.max(abs(exp) * 1e-3, 0.5), "K=\(K) k=\(k)") }
                XCTAssertEqual(mins[k], nonNaN.min(), "min k=\(k)")
                XCTAssertEqual(maxs[k], nonNaN.max(), "max k=\(k)")
            }
        }
    }

    func testIntMinMaxAndCountValues() throws {
        try requireRealGPU()
        let keys = try MetalArray<Int32>([0, 1, 0, 2, 1, nil, 0])
        let i32 = try MetalArray<Int32>([5, -7, nil, 9, 3, 100, -2])
        let gb = try keys.groupBy(keyCount: 4)
        XCTAssertEqual(try gb.min(i32).toArray(), [-2, -7, 9, nil])
        XCTAssertEqual(try gb.max(i32).toArray(), [5, 3, 9, nil])
        XCTAssertEqual(try gb.count(i32).toRawArray(), [2, 2, 1, 0])
        XCTAssertEqual(try gb.count().toRawArray(), [3, 2, 1, 0])
        let u8 = try MetalArray<UInt8>([200, 1, 255, 0, 2, 9, 7])
        XCTAssertEqual(try gb.min(u8).toArray(), [7, 1, 0, nil])
        XCTAssertEqual(try gb.max(u8).toArray(), [255, 2, 0, nil])
        XCTAssertThrowsError(try gb.min(try MetalArray<Int64>([1, 2, 3, 4, 5, 6, 7])))
        XCTAssertThrowsError(try gb.sum(try MetalArray<Int64>([1, 2])))
        // Int64 keys and sliced inputs
        let k64 = try MetalArray<Int64>((0..<2000).map { Int64($0 % 3) }).slice(offset: 64, length: 1000)
        let v = try MetalArray<Int32>((0..<2000).map { Int32($0) }).slice(offset: 64, length: 1000)
        let s = try k64.groupBy(keyCount: 3).sum(v).toArray()
        var exp = [Int64](repeating: 0, count: 3)
        for i in 64..<1064 { exp[i % 3] += Int64(i) }
        XCTAssertEqual(s.map { $0! }, exp)
    }
}
