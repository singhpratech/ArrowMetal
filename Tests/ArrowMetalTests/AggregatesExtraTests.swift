import XCTest
@testable import ArrowMetal

/// The grouped aggregates built on the segmented path, and the three scalar ones `Aggregates.swift`
/// had left out. Every assertion is against a per-key Swift oracle over the same input.
final class AggregatesExtraTests: XCTestCase {

    /// Random keys (some null, some out of range) and Int64 values (some null).
    func sample(_ n: Int, K: Int, outOfRange: Bool = true) -> (keys: [Int32?], vals: [Int64?]) {
        var g = SystemRandomNumberGenerator()
        var keys: [Int32?] = [], vals: [Int64?] = []
        for _ in 0..<n {
            keys.append(Int.random(in: 0..<12, using: &g) == 0 ? nil : Int32.random(in: 0..<Int32(K), using: &g))
            vals.append(Int.random(in: 0..<9, using: &g) == 0 ? nil : Int64.random(in: -1000...1000, using: &g))
        }
        if outOfRange, n > 4 { keys[1] = Int32(K) + 3; keys[2] = -2 }
        return (keys, vals)
    }

    /// The rows of each key, in row order, as (index, value) pairs — the oracle every test shares.
    func groups(_ keys: [Int32?], _ vals: [Int64?], K: Int) -> [[(Int, Int64?)]] {
        var out = [[(Int, Int64?)]](repeating: [], count: K)
        for (i, k) in keys.enumerated() {
            guard let k, k >= 0, Int(k) < K else { continue }
            out[Int(k)].append((i, vals[i]))
        }
        return out
    }

    // MARK: - fused min_max

    func testMinMaxFusedMatchesOracle() throws {
        try requireRealGPU()
        for n in [0, 1, 33, 4097, 200_003] {
            for K in [1, 7, 1024, 5000] {
                let (keys, vals) = sample(n, K: K)
                let gb = try MetalArray<Int32>(keys).groupBy(keyCount: K)
                let g = groups(keys, vals, K: K)

                let i64 = try MetalArray<Int64>(vals)
                let (lo, hi) = try gb.minMax(i64)
                for k in 0..<K {
                    let members = g[k].compactMap { $0.1 }
                    XCTAssertEqual(lo[k], members.min(), "min64 K=\(K) n=\(n) k=\(k)")
                    XCTAssertEqual(hi[k], members.max(), "max64 K=\(K) n=\(n) k=\(k)")
                }
                // A 32-bit type goes through the same fused kernel, and must agree with the atomic path.
                let i32 = try MetalArray<Int32>(vals.map { $0.map { Int32(truncatingIfNeeded: $0) } })
                let (lo32, hi32) = try gb.minMax(i32)
                XCTAssertEqual(lo32.toArray(), try gb.min(i32).toArray(), "min32 K=\(K) n=\(n)")
                XCTAssertEqual(hi32.toArray(), try gb.max(i32).toArray(), "max32 K=\(K) n=\(n)")

                let s = try gb.minMaxStruct(i64)
                XCTAssertEqual(s.names, ["min", "max"])
                XCTAssertEqual(s.length, K)
            }
        }
    }

    func testMinMaxOverFloatsSkipsNaN() throws {
        try requireRealGPU()
        let keys = try MetalArray<Int32>([0, 0, 0, 1, 1, 2])
        let f64 = try MetalArray<Double>([1.5, .nan, -2.0, .nan, .nan, nil])
        let gb = try keys.groupBy(keyCount: 3)
        let (lo, hi) = try gb.minMax(f64)
        XCTAssertEqual(lo.toArray(), [-2.0, nil, nil])
        XCTAssertEqual(hi.toArray(), [1.5, nil, nil])
        let f32 = try MetalArray<Float>([1.5, .nan, -2.0, 7, 8, nil])
        let (lo32, hi32) = try gb.minMax(f32)
        XCTAssertEqual(lo32.toArray(), [-2.0, 7, nil])
        XCTAssertEqual(hi32.toArray(), [1.5, 8, nil])
    }

    // MARK: - count_all, first_last, one

    func testCountAllFirstLastAndOne() throws {
        try requireRealGPU()
        let keys = try MetalArray<Int32>([0, 1, 0, 2, 1, nil, 0, 9])
        let vals = try MetalArray<Int64>([nil, 20, 30, 40, 50, 60, 70, 80])
        let gb = try keys.groupBy(keyCount: 3)
        XCTAssertEqual(try gb.countAll().toRawArray(), [3, 2, 1])
        let fl = try gb.firstLast(vals)
        XCTAssertEqual(fl.names, ["first", "last"])
        guard case .int64(let f) = fl.children[0], case .int64(let l) = fl.children[1] else { return XCTFail("types") }
        XCTAssertEqual(f.toArray(), [30, 20, 40])       // key 0's first non-null value is row 2
        XCTAssertEqual(l.toArray(), [70, 50, 40])
        // `one` is the first non-null value of the group, matching pyarrow's hash_one.
        XCTAssertEqual(try gb.one(vals).toArray(), [30, 20, 40])
        // The other reading — the lowest row of the group, null included — is still reachable.
        XCTAssertEqual(try gb.oneIncludingNull(vals).toArray(), [nil, 20, 40])
    }

    // MARK: - list and distinct

    func testListAndDistinct() throws {
        try requireRealGPU()
        for n in [0, 1, 33, 4097, 100_003] {
            for K in [1, 7, 1024] {
                let (keys, vals) = sample(n, K: K)
                let gb = try MetalArray<Int32>(keys).groupBy(keyCount: K)
                let values = try MetalArray<Int64>(vals)
                let g = groups(keys, vals, K: K)

                let lists = try gb.list(values)
                XCTAssertEqual(lists.length, K, "list length n=\(n) K=\(K)")
                guard case .int64(let child) = lists.values else { return XCTFail("list child type") }
                for k in 0..<K {
                    guard let r = lists.valueRange(k) else { return XCTFail("null list row") }
                    XCTAssertEqual(r.count, g[k].count, "list row \(k) n=\(n) K=\(K)")
                    XCTAssertEqual(r.map { child[$0] }, g[k].map { $0.1 }, "list values \(k) n=\(n) K=\(K)")
                }

                let distinct = try gb.distinct(values)
                guard case .int64(let dchild) = distinct.values else { return XCTFail("distinct child type") }
                for k in 0..<K {
                    guard let r = distinct.valueRange(k) else { return XCTFail("null distinct row") }
                    let expected = Array(Set(g[k].compactMap { $0.1 })).sorted()
                    XCTAssertEqual(r.map { dchild[$0]! }, expected, "distinct \(k) n=\(n) K=\(K)")
                }
            }
        }
    }

    // MARK: - product on the GPU

    func testProductIsGPUAndWraps() throws {
        try requireRealGPU()
        for n in [0, 1, 33, 4097, 50_003] {
            for K in [1, 7, 1024] {
                let (keys, rawVals) = sample(n, K: K)
                // Keep the factors small so the oracle and the kernel agree bit for bit before wrapping.
                let vals = rawVals.map { $0.map { ($0 % 7) + 1 } }
                let gb = try MetalArray<Int32>(keys).groupBy(keyCount: K)
                let values = try MetalArray<Int64>(vals)
                let g = groups(keys, vals, K: K)
                let p = try gb.product(values)
                for k in 0..<K {
                    let members = g[k].compactMap { $0.1 }
                    if members.isEmpty { XCTAssertNil(p[k], "empty product k=\(k)") }
                    else { XCTAssertEqual(p[k], members.reduce(Int64(1)) { $0 &* $1 }, "product k=\(k) n=\(n) K=\(K)") }
                }
            }
        }
        // Wrapping matches the scalar `product`: Int64.max * 2 wraps to -2.
        let keys = try MetalArray<Int32>([0, 0])
        let p = try keys.groupBy(keyCount: 1).product(try MetalArray<Int64>([Int64.max, 2]))
        XCTAssertEqual(p[0], Int64.max &* 2)
    }

    func testProductOverFloats() throws {
        try requireRealGPU()
        let keys = try MetalArray<Int32>([0, 0, 0, 1, 1, 2])
        let gb = try keys.groupBy(keyCount: 3)
        let f32 = try MetalArray<Float>([2, 3, 4, 0.5, -2, nil])
        XCTAssertEqual(try gb.productFloat(f32).toArray(), [24, -1, nil])
        let f64 = try MetalArray<Double>([2, 3, 4, 0.5, -2, nil])
        let d = try gb.productFloat(f64)
        XCTAssertEqual(d[0]!, 24, accuracy: 1e-12)
        XCTAssertEqual(d[1]!, -1, accuracy: 1e-12)
        XCTAssertNil(d[2])
    }

    // MARK: - median and quantile

    func testGroupedMedianAndQuantile() throws {
        try requireRealGPU()
        for n in [0, 1, 33, 4097, 100_003] {
            for K in [1, 7, 1024] {
                let (keys, vals) = sample(n, K: K)
                let gb = try MetalArray<Int32>(keys).groupBy(keyCount: K)
                let values = try MetalArray<Int64>(vals)
                let g = groups(keys, vals, K: K)
                for q in [0.0, 0.25, 0.5, 0.9, 1.0] {
                    let out = q == 0.5 ? try gb.approximateMedian(values) : try gb.quantile(values, q)
                    for k in 0..<K {
                        let sorted = g[k].compactMap { $0.1 }.sorted()
                        if sorted.isEmpty { XCTAssertNil(out[k], "empty q=\(q) k=\(k)"); continue }
                        let pos = q * Double(sorted.count - 1)
                        let lo = Int(pos.rounded(.down)), hi = Int(pos.rounded(.up))
                        let expected = Double(sorted[lo]) + (Double(sorted[hi]) - Double(sorted[lo])) * (pos - Double(lo))
                        XCTAssertEqual(out[k]!, expected, accuracy: 1e-6, "q=\(q) k=\(k) n=\(n) K=\(K)")
                    }
                }
            }
        }
    }

    func testGroupedMedianOverFloats() throws {
        try requireRealGPU()
        let keys = try MetalArray<Int32>([0, 0, 0, 0, 1, 1, 2])
        let gb = try keys.groupBy(keyCount: 3)
        let f = try MetalArray<Double>([4, 1, 3, 2, 10, nil, nil])
        XCTAssertEqual(try gb.approximateMedian(f).toArray(), [2.5, 10, nil])
        XCTAssertEqual(try gb.quantile(f, 0.0).toArray(), [1, 10, nil])
        XCTAssertEqual(try gb.quantile(f, 1.0).toArray(), [4, 10, nil])
    }

    // MARK: - skew and kurtosis

    /// The biased (population) skew and excess kurtosis of a sample, in Double.
    func refMoments(_ xs: [Double]) -> (skew: Double?, kurtosis: Double?) {
        guard !xs.isEmpty else { return (nil, nil) }
        let n = Double(xs.count)
        let mean = xs.reduce(0, +) / n
        var m2 = 0.0, m3 = 0.0, m4 = 0.0
        for x in xs { let d = x - mean; m2 += d * d; m3 += d * d * d; m4 += d * d * d * d }
        m2 /= n; m3 /= n; m4 /= n
        guard m2 > 0 else { return (nil, nil) }
        return (m3 / (m2 * m2.squareRoot()), m4 / (m2 * m2) - 3)
    }

    func testScalarSkewAndKurtosis() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        for n in [33, 4097, 200_003] {
            let xs: [Double] = (0..<n).map { _ in Double.random(in: -5...5, using: &g) }
            let expected = refMoments(xs)
            let f64 = try MetalArray<Double>(xs)
            XCTAssertEqual(try f64.skew()!, expected.skew!, accuracy: max(abs(expected.skew!) * 1e-6, 1e-6), "f64 skew n=\(n)")
            XCTAssertEqual(try f64.kurtosis()!, expected.kurtosis!, accuracy: max(abs(expected.kurtosis!) * 1e-6, 1e-6), "f64 kurt n=\(n)")
            let f32 = try MetalArray<Float>(xs.map { Float($0) })
            let e32 = refMoments(xs.map { Double(Float($0)) })
            XCTAssertEqual(try f32.skew()!, e32.skew!, accuracy: max(abs(e32.skew!) * 1e-3, 1e-3), "f32 skew n=\(n)")
            XCTAssertEqual(try f32.kurtosis()!, e32.kurtosis!, accuracy: max(abs(e32.kurtosis!) * 1e-3, 1e-3), "f32 kurt n=\(n)")
            let ints: [Int64] = xs.map { Int64($0 * 1000) }
            let ei = refMoments(ints.map { Double($0) })
            let i64 = try MetalArray<Int64>(ints)
            XCTAssertEqual(try i64.skew()!, ei.skew!, accuracy: max(abs(ei.skew!) * 1e-3, 1e-3), "i64 skew n=\(n)")
        }
        // Degenerate inputs have no answer.
        XCTAssertNil(try MetalArray<Double>([1, 1, 1]).skew())
        XCTAssertNil(try MetalArray<Double>([Double?]()).kurtosis())
        XCTAssertNil(try MetalArray<Double>([1, 2, nil]).skew(minCount: 5))
        // The sample-corrected forms need three (four) values.
        XCTAssertNil(try MetalArray<Double>([1, 2]).skew(biased: false))
        XCTAssertNotNil(try MetalArray<Double>([1, 2, 5]).skew(biased: false))
    }

    func testGroupedSkewAndKurtosis() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        for n in [0, 1, 33, 4097, 50_003] {
            for K in [1, 7, 200] {
                var keys: [Int32?] = [], vals: [Float?] = []
                for _ in 0..<n {
                    keys.append(Int.random(in: 0..<12, using: &g) == 0 ? nil : Int32.random(in: 0..<Int32(K), using: &g))
                    vals.append(Int.random(in: 0..<9, using: &g) == 0 ? nil : Float.random(in: -5...5, using: &g))
                }
                let gb = try MetalArray<Int32>(keys).groupBy(keyCount: K)
                let values = try MetalArray<Float>(vals)
                let s = try gb.skew(values), kt = try gb.kurtosis(values)
                var members = [[Double]](repeating: [], count: K)
                for (i, k) in keys.enumerated() {
                    guard let k, k >= 0, Int(k) < K, let v = vals[i] else { continue }
                    members[Int(k)].append(Double(v))
                }
                for k in 0..<K {
                    let e = refMoments(members[k])
                    if let es = e.skew, members[k].count > 8 {
                        XCTAssertEqual(s[k]!, es, accuracy: max(abs(es) * 2e-3, 2e-3), "skew k=\(k) n=\(n) K=\(K)")
                        XCTAssertEqual(kt[k]!, e.kurtosis!, accuracy: max(abs(e.kurtosis!) * 2e-3, 2e-3), "kurt k=\(k)")
                    } else if members[k].isEmpty {
                        XCTAssertNil(s[k], "empty skew k=\(k)")
                    }
                }
            }
        }
    }

    // MARK: - tdigest

    func testScalarTDigestBracketsTheExactQuantile() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        let n = 200_003
        let xs: [Double] = (0..<n).map { _ in Double.random(in: 0...1000, using: &g) }
        let col = try MetalArray<Double>(xs)
        let sorted = xs.sorted()
        for q in [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0] {
            let exact = sorted[Int((Double(n - 1) * q).rounded())]
            let est = try col.tdigest(q)!
            // A t-digest with delta 100 keeps the tails tight and the middle within about 1% of the range.
            XCTAssertEqual(est, exact, accuracy: 12.0, "tdigest q=\(q) est=\(est) exact=\(exact)")
        }
        XCTAssertEqual(try col.tdigest(0.0)!, sorted.first!, accuracy: 1e-9)
        XCTAssertEqual(try col.tdigest(1.0)!, sorted.last!, accuracy: 1e-9)
        XCTAssertNil(try MetalArray<Double>([Double?]()).tdigest(0.5))
        // Several quantiles from one sort agree with one at a time.
        let many = try col.tdigest([0.25, 0.5, 0.75])
        XCTAssertEqual(many[1]!, try col.tdigest(0.5)!, accuracy: 1e-9)
    }

    func testGroupedTDigest() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        let n = 60_000, K = 8
        var keys: [Int32] = [], vals: [Double] = []
        for _ in 0..<n {
            keys.append(Int32.random(in: 0..<Int32(K), using: &g))
            vals.append(Double.random(in: 0...1000, using: &g))
        }
        let gb = try MetalArray<Int32>(keys).groupBy(keyCount: K)
        let est = try gb.tdigest(try MetalArray<Double>(vals), 0.5)
        for k in 0..<K {
            let members = (0..<n).filter { keys[$0] == Int32(k) }.map { vals[$0] }.sorted()
            let exact = members[(members.count - 1) / 2]
            XCTAssertEqual(est[k]!, exact, accuracy: 20.0, "grouped tdigest k=\(k)")
        }
    }

    // MARK: - pivot_wider

    func testPivotWider() throws {
        try requireRealGPU()
        let keys = try MetalArray<Int32>([0, 0, 1, 1, 2])
        let pivot = try MetalStringArray(["a", "b", "a", "b", "b"])
        let values = try MetalArray<Int64>([1, 2, 3, 4, 5])
        let gb = try keys.groupBy(keyCount: 3)
        let s = try gb.pivotWider(pivotKeys: pivot, values: values, names: ["a", "b", "c"])
        XCTAssertEqual(s.names, ["a", "b", "c"])
        guard case .int64(let a) = s.children[0], case .int64(let b) = s.children[1],
              case .int64(let c) = s.children[2] else { return XCTFail("pivot child types") }
        XCTAssertEqual(a.toArray(), [1, 3, nil])
        XCTAssertEqual(b.toArray(), [2, 4, 5])
        XCTAssertEqual(c.toArray(), [nil, nil, nil])
    }

    // MARK: - the whole family over a mapped key

    func testEveryAggregateOverArbitraryKeys() throws {
        try requireRealGPU()
        let region = try MetalStringArray(["west", "east", "west", nil, "east", "west"])
        let amount = try MetalArray<Int64>([10, 20, nil, 40, 50, 60])
        let gbk = try GroupByKeys(columns: [.string(region)])
        XCTAssertEqual(gbk.groupCount, 3)
        let gb = gbk.groupBy
        var byName: [String: Int] = [:]
        guard case .string(let names) = try gbk.groupKeys()[0] else { return XCTFail("key type") }
        for g in 0..<gbk.groupCount { byName[names[g] ?? "null"] = g }

        let sums = try gb.sum(amount)
        XCTAssertEqual(sums[byName["west"]!], 70)
        XCTAssertEqual(sums[byName["east"]!], 70)
        XCTAssertEqual(sums[byName["null"]!], 40)
        XCTAssertEqual(try gb.countAll().valuePointer[byName["west"]!], 3)
        XCTAssertEqual(try gb.count(amount).valuePointer[byName["west"]!], 2)
        let (lo, hi) = try gb.minMax(amount)
        XCTAssertEqual(lo[byName["west"]!], 10)
        XCTAssertEqual(hi[byName["west"]!], 60)
        XCTAssertEqual(try gb.one(amount)[byName["west"]!], 10)
        XCTAssertEqual(try gb.first(amount)[byName["west"]!], 10)
        XCTAssertEqual(try gb.last(amount)[byName["west"]!], 60)
        XCTAssertEqual(try gb.product(amount)[byName["west"]!], 600)
        XCTAssertEqual(try gb.approximateMedian(amount)[byName["west"]!]!, 35, accuracy: 1e-9)
        XCTAssertEqual(try gb.countDistinct(amount).valuePointer[byName["east"]!], 2)
        let lists = try gb.list(amount)
        guard case .int64(let child) = lists.values, let r = lists.valueRange(byName["west"]!) else {
            return XCTFail("list shape")
        }
        XCTAssertEqual(r.map { child[$0] }, [10, nil, 60])
    }

    /// Ten million rows, a hundred thousand keys: the extremes and the positional aggregates against
    /// flat-array oracles, which is what keeps this affordable in a debug build.
    func testTenMillionRowsAcrossTheFamily() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        let n = 10_000_003, K = 100_000
        var keys = [Int32](repeating: 0, count: n)
        var vals = [Int64?](repeating: nil, count: n)
        for i in 0..<n {
            keys[i] = Int32.random(in: 0..<Int32(K), using: &g)
            vals[i] = i % 11 == 0 ? nil : Int64.random(in: -1000...1000, using: &g)
        }
        let gb = try MetalArray<Int32>(keys).groupBy(keyCount: K)
        let values = try MetalArray<Int64>(vals)

        var lo = [Int64](repeating: .max, count: K), hi = [Int64](repeating: .min, count: K)
        var counts = [Int](repeating: 0, count: K), rows = [Int](repeating: 0, count: K)
        var firsts = [Int64?](repeating: nil, count: K), lasts = [Int64?](repeating: nil, count: K)
        for i in 0..<n {
            let k = Int(keys[i])
            rows[k] += 1
            guard let v = vals[i] else { continue }
            counts[k] += 1
            lo[k] = Swift.min(lo[k], v); hi[k] = Swift.max(hi[k], v)
            if firsts[k] == nil { firsts[k] = v }
            lasts[k] = v
        }
        let (gpuLo, gpuHi) = try gb.minMax(values)
        let gpuCount = try gb.countValid(values), gpuRows = try gb.countAll()
        let gpuFirst = try gb.first(values), gpuLast = try gb.last(values), gpuOne = try gb.one(values)
        for k in 0..<K {
            XCTAssertEqual(Int(gpuRows.valuePointer[k]), rows[k], "count_all \(k)")
            XCTAssertEqual(Int(gpuCount.valuePointer[k]), counts[k], "count \(k)")
            if counts[k] == 0 {
                XCTAssertNil(gpuLo[k], "min \(k)")
                XCTAssertNil(gpuFirst[k], "first \(k)")
            } else {
                XCTAssertEqual(gpuLo[k], lo[k], "min \(k)")
                XCTAssertEqual(gpuHi[k], hi[k], "max \(k)")
                XCTAssertEqual(gpuFirst[k], firsts[k], "first \(k)")
                XCTAssertEqual(gpuLast[k], lasts[k], "last \(k)")
                XCTAssertEqual(gpuOne[k], firsts[k], "one \(k)")
            }
        }
        // The median needs a sorted oracle per key, which is quadratic-ish to build in a debug build,
        // so it is checked over a million rows and a thousand keys instead of ten million and 100k.
        let m = 1_000_003, MK = 1000
        var mkeys = [Int32](repeating: 0, count: m), mvals = [Int64?](repeating: nil, count: m)
        var byKey = [[Int64]](repeating: [], count: MK)
        for i in 0..<m {
            let k = Int32.random(in: 0..<Int32(MK), using: &g)
            let v: Int64? = i % 11 == 0 ? nil : Int64.random(in: -1000...1000, using: &g)
            mkeys[i] = k; mvals[i] = v
            if let v { byKey[Int(k)].append(v) }
        }
        let mgb = try MetalArray<Int32>(mkeys).groupBy(keyCount: MK)
        let medians = try mgb.approximateMedian(try MetalArray<Int64>(mvals))
        for k in 0..<MK {
            let sorted = byKey[k].sorted()
            if sorted.isEmpty { XCTAssertNil(medians[k], "median \(k)"); continue }
            let pos = 0.5 * Double(sorted.count - 1)
            let a = Double(sorted[Int(pos.rounded(.down))]), b = Double(sorted[Int(pos.rounded(.up))])
            XCTAssertEqual(medians[k]!, a + (b - a) * (pos - pos.rounded(.down)), accuracy: 1e-6, "median \(k)")
        }
    }
}
