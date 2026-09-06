import XCTest
@testable import ArrowMetal

/// The remaining Arrow selection / sort / random functions, each against a host oracle:
/// `inverse_permutation`, `scatter`, `winsorize`, `rank_quantile`, `rank_normal`, `random`,
/// `count_all`, `first_last`, `true_unless_null`, `utf8_swapcase`, `utf8_zero_fill` and
/// `pivot_wider`.
final class SelectionExtraTests: XCTestCase {

    /// Deterministic splitmix64 so a failure is reproducible.
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

    private let sizes = [0, 1, 33, 4097, 1_000_003]

    // MARK: - inverse_permutation and scatter

    /// For the i-th index, the index-th output is i: unassigned slots null, duplicates last-wins.
    private func oracleInversePermutation(_ idx: [Int32?], outLength: Int) -> [Int32?] {
        var out = [Int32?](repeating: nil, count: outLength)
        for (i, v) in idx.enumerated() { if let v { out[Int(v)] = Int32(i) } }
        return out
    }

    func testInversePermutationRoundTrip() throws {
        try requireRealGPU()
        var g = Rng(s: 7)
        for n in sizes {
            var perm = (0..<n).map { Int32($0) }
            perm.shuffle(using: &g)
            let a = try MetalArray<Int32>(perm)
            let inv = try a.inversePermutation()
            XCTAssertEqual(inv.length, n)
            XCTAssertEqual(inv.nullCount, 0, "a permutation leaves no gaps")
            XCTAssertEqual(inv.toArray(), oracleInversePermutation(perm.map { $0 }, outLength: n), "n=\(n)")
            // The inverse of the inverse is the original permutation.
            XCTAssertEqual(try inv.inversePermutation().toArray(), perm.map { Optional($0) }, "n=\(n)")
        }
    }

    func testInversePermutationDuplicatesGapsAndNulls() throws {
        try requireRealGPU()
        var g = Rng(s: 11)
        for n in sizes {
            let outLen = Swift.max(n / 2, 4)
            // Deliberately lumpy: duplicates, unused slots and null indices.
            let idx: [Int32?] = (0..<n).map { i in
                i % 7 == 0 ? nil : Int32(Int(UInt64.random(in: 0..<UInt64(outLen), using: &g)))
            }
            let a = try MetalArray<Int32>(idx)
            let inv = try a.inversePermutation(maxIndex: Int64(outLen - 1))
            XCTAssertEqual(inv.length, outLen, "n=\(n)")
            XCTAssertEqual(inv.toArray(), oracleInversePermutation(idx, outLength: outLen), "n=\(n)")
        }
        // An index outside [0, max_index] raises.
        let bad = try MetalArray<Int32>([0, 5, 1])
        XCTAssertThrowsError(try bad.inversePermutation(maxIndex: 2))
        XCTAssertThrowsError(try MetalArray<Int32>([-1, 0]).inversePermutation())
        // int64 and uint32 index columns work too.
        XCTAssertEqual(try MetalArray<Int64>([2, 0, 1]).inversePermutation().toArray(), [1, 2, 0])
        XCTAssertEqual(try MetalArray<UInt32>([2, 0, 1]).inversePermutation().toArray(), [1, 2, 0])
    }

    func testScatter() throws {
        try requireRealGPU()
        var g = Rng(s: 13)
        for n in sizes {
            let outLen = Swift.max(n, 4)
            let idx: [Int32?] = (0..<n).map { i in
                i % 5 == 0 ? nil : Int32(Int(UInt64.random(in: 0..<UInt64(outLen), using: &g)))
            }
            let vals: [Int64?] = (0..<n).map { i in i % 11 == 0 ? nil : Int64(i) * 3 }
            let values = try MetalArray<Int64>(vals)
            let got = try values.scattered(to: try MetalArray<Int32>(idx), maxIndex: Int64(outLen - 1))
            let source = oracleInversePermutation(idx, outLength: outLen)
            let want: [Int64?] = source.map { $0.flatMap { vals[Int($0)] } }
            XCTAssertEqual(got.toArray(), want, "n=\(n)")
        }
        // Strings scatter through the same inverse permutation.
        let s = try MetalStringArray(["a", "b", "c"])
        let scattered = try AnyMetalArray.string(s).scattered(to: try MetalArray<Int32>([2, 0, 1]))
        XCTAssertEqual(scattered.asString?.toArray(), ["b", "c", "a"])
    }

    // MARK: - winsorize

    private func oracleWinsorize(_ vals: [Double?], _ lower: Double, _ upper: Double) -> [Double?] {
        let finite = vals.compactMap { $0 }.filter { !$0.isNaN }.sorted()
        let m = finite.count
        guard m > 0 else { return vals }
        // Arrow's bounds are the "nearest" quantiles: round(q * (m - 1)), halfway to the even index.
        func nearest(_ q: Double) -> Int {
            Swift.min(m - 1, Swift.max(0, Int((q * Double(m - 1)).rounded(.toNearestOrEven))))
        }
        let loIndex = nearest(lower), hiIndex = Swift.max(loIndex, nearest(upper))
        let lo = finite[loIndex], hi = finite[hiIndex]
        return vals.map { v in
            guard let v else { return nil }
            if v.isNaN { return v }
            return Swift.min(Swift.max(v, lo), hi)
        }
    }

    func testWinsorizeAgainstOracle() throws {
        try requireRealGPU()
        var g = Rng(s: 17)
        for n in sizes {
            let vals: [Double?] = (0..<n).map { i in
                if i % 13 == 0 { return nil }
                if i % 101 == 0 { return Double.nan }
                return Double(Int64.random(in: -10_000...10_000, using: &g))
            }
            let a = try MetalArray<Double>(vals)
            for (lo, hi) in [(0.0, 1.0), (0.05, 0.95), (0.2, 0.8), (0.5, 0.5)] {
                let got = try a.winsorize(lowerLimit: lo, upperLimit: hi).toArray()
                let want = oracleWinsorize(vals, lo, hi)
                XCTAssertEqual(got.count, want.count)
                for i in 0..<got.count {
                    if let w = want[i], w.isNaN { XCTAssertEqual(got[i]?.isNaN, true, "row \(i) n=\(n)"); continue }
                    XCTAssertEqual(got[i], want[i], "row \(i) n=\(n) limits \(lo)/\(hi)")
                }
            }
        }
        // Integers, and the worked example from pyarrow's docs.
        let ints = try MetalArray<Int32>(Array(1...10))
        XCTAssertEqual(try ints.winsorize(lowerLimit: 0.2, upperLimit: 0.8).toRawArray(),
                       [3, 3, 3, 4, 5, 6, 7, 8, 8, 8])
        let floats = try MetalArray<Float>([1, 2, 3, 4, 5])
        XCTAssertEqual(try floats.winsorize(lowerLimit: 0.1, upperLimit: 0.9).toRawArray(), [1, 2, 3, 4, 5])
        XCTAssertEqual(try floats.winsorize(lowerLimit: 0.25, upperLimit: 0.75).toRawArray(), [2, 2, 3, 4, 4])
        // All null: unchanged.
        XCTAssertEqual(try MetalArray<Int32>([nil, nil]).winsorize(lowerLimit: 0.1, upperLimit: 0.9).toArray(), [nil, nil])
        XCTAssertThrowsError(try ints.winsorize(lowerLimit: 0.9, upperLimit: 0.1))
    }

    // MARK: - rank_quantile and rank_normal

    /// pyarrow's definition: (average 1-based rank of the tie group - 0.5) / n, nulls last as one group.
    private func oracleRankQuantile(_ vals: [Double?]) -> [Double] {
        let n = vals.count
        guard n > 0 else { return [] }
        // Arrow's total order: NaN is one value after +inf, and -0.0 ties with 0.0.
        func less(_ a: Double, _ b: Double) -> Bool { a.isNaN ? false : (b.isNaN ? true : a < b) }
        func same(_ a: Double, _ b: Double) -> Bool { (a.isNaN || b.isNaN) ? (a.isNaN && b.isNaN) : a == b }
        let nonNull = vals.enumerated().filter { $0.element != nil }
        let order = nonNull.sorted { a, b in
            if less(a.element!, b.element!) { return true }
            if less(b.element!, a.element!) { return false }
            return a.offset < b.offset
        }.map { $0.offset } + vals.enumerated().filter { $0.element == nil }.map { $0.offset }
        let m = nonNull.count
        var out = [Double](repeating: 0, count: n)
        var p = 0
        while p < n {
            var q = p + 1
            if p >= m {
                q = n                                  // the nulls are one tie group
            } else {
                while q < m && same(vals[order[q]]!, vals[order[p]]!) { q += 1 }
            }
            let value = Double(p + q) / (2 * Double(n))
            for k in p..<q { out[order[k]] = value }
            p = q
        }
        return out
    }

    func testRankQuantileAndNormal() throws {
        try requireRealGPU()
        var g = Rng(s: 19)
        for n in sizes {
            let vals: [Double?] = (0..<n).map { i in
                if i % 9 == 0 { return nil }
                if i % 97 == 0 { return Double.nan }
                return Double(Int64.random(in: -50...50, using: &g))     // plenty of ties
            }
            let a = try MetalArray<Double>(vals)
            let want = oracleRankQuantile(vals)
            let got = try a.rankQuantile().toRawArray()
            XCTAssertEqual(got.count, want.count, "n=\(n)")
            for i in 0..<got.count { XCTAssertEqual(got[i], want[i], accuracy: 1e-15, "row \(i) n=\(n)") }
            // rank_normal is the normal PPF of the same quantiles.
            let normal = try a.rankNormal().toRawArray()
            for i in 0..<normal.count {
                XCTAssertEqual(normal[i], NormalQuantile.ppf(want[i]), accuracy: 1e-12, "row \(i) n=\(n)")
            }
        }
        // Float32 rank_normal on the GPU against the host inverse normal.
        for n in [1, 33, 4097, 1_000_003] {
            let vals = (0..<n).map { i in Float(i % 1000) }
            let a = try MetalArray<Float>(vals)
            let q = try a.rankQuantile().toRawArray()
            let gpu = try a.rankNormalFloat32().toRawArray()
            for i in 0..<n {
                XCTAssertEqual(Double(gpu[i]), NormalQuantile.ppf(q[i]), accuracy: 1e-6, "row \(i) n=\(n)")
            }
        }
        // The worked example from pyarrow.
        let small = try MetalArray<Int32>([3, 1, 4, 1, 5, nil])
        let q = try small.rankQuantile().toRawArray()
        for (got, want) in zip(q, [5.0 / 12, 2.0 / 12, 7.0 / 12, 2.0 / 12, 9.0 / 12, 11.0 / 12]) {
            XCTAssertEqual(got, want, accuracy: 1e-15)
        }
        XCTAssertEqual(try MetalArray<Int32>([]).rankQuantile().length, 0)
    }

    func testNormalQuantileAgainstKnownValues() throws {
        // Wichura AS 241 against values good to 15 digits.
        let cases: [(Double, Double)] = [
            (0.5, 0.0), (0.975, 1.959963984540054), (0.025, -1.959963984540054),
            (0.99, 2.3263478740408408), (1e-6, -4.753424308822899), (1.0 - 1e-6, 4.753424308822899),
            (0.16666666666666666, -0.9674215661017015), (0.9166666666666666, 1.3829941271006389),
        ]
        for (p, want) in cases {
            // The near-1 cases lose a few digits to `1 - p` before AS 241 ever runs, as Wichura's own
            // formulation does; 1e-9 relative is the honest bound there and far looser than the middle.
            XCTAssertEqual(NormalQuantile.ppf(p), want, accuracy: 1e-9 * Swift.max(1, Swift.abs(want)), "ppf(\(p))")
        }
        XCTAssertEqual(NormalQuantile.ppf(0), -.infinity)
        XCTAssertEqual(NormalQuantile.ppf(1), .infinity)
        XCTAssertTrue(NormalQuantile.ppf(1.5).isNaN)
    }

    // MARK: - random

    func testRandomIsDeterministicAndUniform() throws {
        try requireRealGPU()
        XCTAssertEqual(try ArrowRandom.uniform(count: 0, seed: 1).length, 0)
        let a = try ArrowRandom.uniform(count: 1000, seed: 42).toRawArray()
        let b = try ArrowRandom.uniform(count: 1000, seed: 42).toRawArray()
        XCTAssertEqual(a, b, "the same seed gives the same stream")
        let c = try ArrowRandom.uniform(count: 1000, seed: 43).toRawArray()
        XCTAssertNotEqual(a, c)
        XCTAssertTrue(a.allSatisfy { $0 >= 0 && $0 < 1 }, "values live in [0, 1)")
        // A prefix of a longer draw is the same as a shorter draw: the generator is counter based.
        XCTAssertEqual(Array(try ArrowRandom.uniform(count: 5000, seed: 42).toRawArray().prefix(1000)), a)

        let n = 10_000_000
        let v = try ArrowRandom.uniform(count: n, seed: 20240906).toRawArray()
        var bins = [Int](repeating: 0, count: 100)
        var sum = 0.0, sumSq = 0.0
        for x in v {
            bins[Swift.min(99, Int(x * 100))] += 1
            sum += x
            sumSq += x * x
        }
        let mean = sum / Double(n)
        let variance = sumSq / Double(n) - mean * mean
        XCTAssertEqual(mean, 0.5, accuracy: 0.001, "uniform mean")
        XCTAssertEqual(variance, 1.0 / 12, accuracy: 0.001, "uniform variance")
        let expected = Double(n) / 100
        let chi2 = bins.reduce(0.0) { acc, count in
            let d = Double(count) - expected
            return acc + d * d / expected
        }
        // 99 degrees of freedom: the 0.001 upper tail is 148.2, so 200 is a wide but decisive bound.
        XCTAssertLessThan(chi2, 200.0, "chi-square over 100 bins at 10M samples was \(chi2)")
    }

    // MARK: - count_all, first_last, true_unless_null

    func testCountAllFirstLastTrueUnlessNull() throws {
        try requireRealGPU()
        let a = try MetalArray<Int32>([nil, 7, 3, nil, 9, nil])
        XCTAssertEqual(AnyMetalArray.int32(a).countAll, 6)
        let fl = try a.firstLast()
        XCTAssertEqual(fl.length, 1)
        XCTAssertEqual(fl.names, ["first", "last"])
        XCTAssertEqual(fl.children[0].asInt32?.toArray(), [7])
        XCTAssertEqual(fl.children[1].asInt32?.toArray(), [9])
        let flKeep = try a.firstLast(skipNulls: false)
        XCTAssertEqual(flKeep.children[0].asInt32?.toArray(), [nil])
        XCTAssertEqual(flKeep.children[1].asInt32?.toArray(), [nil])
        XCTAssertEqual(try MetalArray<Int32>([nil, nil]).firstLast().children[0].asInt32?.toArray(), [nil])

        let tun = try AnyMetalArray.int32(a).trueUnlessNull()
        XCTAssertEqual(tun.toArray(), [nil, true, true, nil, true, nil])
        let noNulls = try MetalArray<Int64>([1, 2, 3])
        XCTAssertEqual(try AnyMetalArray.int64(noNulls).trueUnlessNull().toArray(), [true, true, true])
        let strings = try MetalStringArray(["a", nil, "c"])
        XCTAssertEqual(try AnyMetalArray.string(strings).trueUnlessNull().toArray(), [true, nil, true])
        XCTAssertEqual(try AnyMetalArray.int32(try MetalArray<Int32>([])).trueUnlessNull().length, 0)
        // Large, so the bitmap tail is exercised.
        let big = try MetalArray<Int32>((0..<100_003).map { $0 % 3 == 0 ? nil : Int32($0) })
        let bigTun = try AnyMetalArray.int32(big).trueUnlessNull()
        XCTAssertEqual(bigTun.nullCount, big.nullCount)
        XCTAssertEqual(bigTun.trueCount, big.length - big.nullCount)
    }

    // MARK: - string transforms

    func testUtf8SwapcaseAndZeroFill() throws {
        try requireRealGPU()
        let a = try MetalStringArray(["aBc", nil, "Ää", "", "123", "ǅ"])
        XCTAssertEqual(try a.utf8Swapcase().toArray(), ["AbC", nil, "äÄ", "", "123", "ǅ"])
        XCTAssertEqual(try a.utf8Swapcase().nullCount, 1)
        // ı -> I shortens the row from two bytes to one; ÿ -> Ÿ keeps it at two.
        XCTAssertEqual(try MetalStringArray(["ı", "ÿ", "Ÿ", "ſ"]).utf8Swapcase().toArray(), ["I", "Ÿ", "ÿ", "S"])

        let nums = try MetalStringArray(["12", "-3", "+4", "abc", "", nil, "123456"])
        XCTAssertEqual(try nums.utf8ZeroFill(width: 5).toArray(),
                       ["00012", "-0003", "+0004", "00abc", "00000", nil, "123456"])
        XCTAssertEqual(try MetalStringArray(["12", "-3"]).utf8ZeroFill(width: 5, padding: "x").toArray(),
                       ["xxx12", "-xxx3"])
        XCTAssertEqual(try MetalStringArray(["é1"]).utf8ZeroFill(width: 5).toArray(), ["000é1"])
        XCTAssertThrowsError(try nums.utf8ZeroFill(width: 5, padding: "ab"))
        // The Arrow-named padding aliases are the existing pad kernels.
        XCTAssertEqual(try nums.lpad(width: 4, pad: "*").toArray(), try nums.padLeft(width: 4, pad: "*").toArray())
        XCTAssertEqual(try nums.rpad(width: 4, pad: "*").toArray(), try nums.padRight(width: 4, pad: "*").toArray())
        // Bigger than one threadgroup, so the offsets scan is exercised.
        let many = try MetalStringArray((0..<10_000).map { "row\($0)" })
        XCTAssertEqual(try many.utf8Swapcase().toArray()[7], "ROW7")
        XCTAssertEqual(try many.utf8ZeroFill(width: 9).toArray()[7], "00000row7")
    }

    // MARK: - pivot_wider

    func testPivotWider() throws {
        try requireRealGPU()
        let keys = AnyMetalArray.string(try MetalStringArray(["w", "x", "y"]))
        let values = AnyMetalArray.int64(try MetalArray<Int64>([1, 2, 3]))
        let s = try PivotWider.pivot(keys: keys, values: values, keyNames: ["w", "x", "z"])
        XCTAssertEqual(s.names, ["w", "x", "z"])
        XCTAssertEqual(s.length, 1)
        XCTAssertEqual(s.children[0].asInt64?.toArray(), [1])
        XCTAssertEqual(s.children[1].asInt64?.toArray(), [2])
        XCTAssertEqual(s.children[2].asInt64?.toArray(), [nil])
        // An unexpected key raises only when asked to.
        XCTAssertThrowsError(try PivotWider.pivot(keys: keys, values: values, keyNames: ["w", "x"],
                                                  unexpectedKey: .raise))
        // A null value does not claim the field; two non-null values for one key raise.
        let dupKeys = AnyMetalArray.string(try MetalStringArray(["w", "w"]))
        XCTAssertEqual(try PivotWider.pivot(keys: dupKeys,
                                            values: .int64(try MetalArray<Int64>([nil, 5])),
                                            keyNames: ["w"]).children[0].asInt64?.toArray(), [5])
        XCTAssertThrowsError(try PivotWider.pivot(keys: dupKeys,
                                                  values: .int64(try MetalArray<Int64>([1, 2])),
                                                  keyNames: ["w"]))
        // Integer keys match by their decimal rendering.
        let intKeys = AnyMetalArray.int32(try MetalArray<Int32>([10, 20, nil]))
        let ints = try PivotWider.pivot(keys: intKeys, values: .int64(try MetalArray<Int64>([1, 2, 3])),
                                        keyNames: ["10", "20", "30"])
        XCTAssertEqual(ints.children[0].asInt64?.toArray(), [1])
        XCTAssertEqual(ints.children[2].asInt64?.toArray(), [nil])
    }
}
