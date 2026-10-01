import XCTest
@testable import ArrowMetal

/// The correctly rounded grouped Float64 `sum` and `mean` (`GroupBy.exactSumMean`), and the grouped
/// extremes that share passes, against host references.
///
/// The sum reference is an independent correctly rounded sum (Shewchuk's partials with the half-even
/// correction of CPython's `math.fsum`). A mean `m` is checked by its residual: `m * n - sum` is formed
/// exactly and must be no larger in magnitude than for either neighbour of `m`.
final class GroupSumExactTests: XCTestCase {

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

    /// Correctly rounded sum of finite values whose partial sums stay finite (CPython's `msum`).
    static func fsum(_ xs: [Double]) -> Double {
        var partials: [Double] = []
        for x0 in xs {
            var x = x0
            var i = 0
            for y0 in partials {
                var y = y0
                if abs(x) < abs(y) { swap(&x, &y) }
                let hi = x + y
                let lo = y - (hi - x)
                if lo != 0 { partials[i] = lo; i += 1 }
                x = hi
            }
            partials.removeSubrange(i...)
            partials.append(x)
        }
        var n = partials.count
        guard n > 0 else { return 0 }
        n -= 1
        var hi = partials[n], lo = 0.0
        while n > 0 {
            let x = hi
            n -= 1
            let y = partials[n]
            hi = x + y
            let yr = hi - x
            lo = y - yr
            if lo != 0 { break }
        }
        if n > 0 && ((lo < 0 && partials[n - 1] < 0) || (lo > 0 && partials[n - 1] > 0)) {
            let y = lo * 2
            let x = hi + y
            let yr = x - hi
            if y == yr { hi = x }
        }
        return hi
    }

    /// |m * n - sum(xs)|, rounded once (monotone in the exact residual).
    static func residual(_ m0: Double, _ xs0: [Double]) -> Double {
        // Terms near the top of the range are scaled by 2^-64 (exactly, for these tests) first.
        let big = abs(m0) >= 0x1p1000 || xs0.contains { abs($0) >= 0x1p1000 }
        let m = big ? m0 * 0x1p-64 : m0, xs = big ? xs0.map { $0 * 0x1p-64 } : xs0
        let n = Double(xs.count)
        let p = m * n
        let e = (-p).addingProduct(m, n)                  // m * n - p, exact
        return abs(fsum([p, e] + xs.map { -$0 }))
    }

    /// `fsum` with every term scaled by 2^-64 when some term is at least 2^1000, so no partial sum
    /// overflows; the terms of these tests are scaled exactly. `rescale` scales the result back.
    static func scaledFsum(_ xs: [Double], rescale: Bool = true) -> Double {
        guard xs.contains(where: { abs($0) >= 0x1p1000 }) else { return fsum(xs) }
        let s = fsum(xs.map { $0 * 0x1p-64 })
        return rescale ? s * 0x1p64 : s
    }

    /// Correctly rounded sum with IEEE special values and the sign of zero, as the kernel defines them.
    static func expectedSum(_ xs: [Double]) -> Double? {
        guard !xs.isEmpty else { return nil }
        if xs.contains(where: { $0.isNaN }) { return .nan }
        let pinf = xs.contains(.infinity), ninf = xs.contains(-.infinity)
        if pinf && ninf { return .nan }
        if pinf { return .infinity }
        if ninf { return -.infinity }
        if xs.allSatisfy({ $0 == 0 }) { return xs.allSatisfy({ $0.sign == .minus }) ? -0.0 : 0.0 }
        return scaledFsum(xs)
    }

    private func same(_ a: Double?, _ b: Double?) -> Bool {
        switch (a, b) {
        case (nil, nil): return true
        case let (x?, y?): return (x.isNaN && y.isNaN) || x.bitPattern == y.bitPattern
        default: return false
        }
    }

    private func checkMean(_ m: Double?, _ xs: [Double], _ label: String) {
        guard let want = Self.expectedSum(xs) else { XCTAssertNil(m, label); return }
        guard let m else { XCTFail("mean is null \(label)"); return }
        if want.isNaN || xs.contains(where: { $0.isInfinite }) || want == 0 {
            // A special sum divides to the same special value; zero keeps its sign.
            XCTAssertTrue(same(m, want.isNaN ? .nan : want / Double(xs.count)), "mean \(m) want \(want) \(label)")
            return
        }
        // A sum that overflows can still have a finite mean: the mean is the exact one, rounded once.
        XCTAssertTrue(m.isFinite, "mean \(m) of finite values \(label)")
        let r = Self.residual(m, xs)
        if m.nextUp.isFinite { XCTAssertLessThanOrEqual(r, Self.residual(m.nextUp, xs), "mean \(m) not nearest (up) \(label)") }
        if m.nextDown.isFinite { XCTAssertLessThanOrEqual(r, Self.residual(m.nextDown, xs), "mean \(m) not nearest (down) \(label)") }
    }

    private func run(keys: [Int32?], values: [Double?], K: Int, label: String,
                     file: StaticString = #filePath, line: UInt = #line) throws {
        let gb = try MetalArray<Int32>(keys).groupBy(keyCount: K)
        let dv = try MetalArray<Double>(values)
        let r = try gb.exactSumMean(dv)
        let sums = r.sum.toArray(), means = r.mean.toArray()
        // Rows of each group in row order, without a K-sized array of arrays.
        var rows: [Int] = []
        for (i, k) in keys.enumerated() {
            if let k, k >= 0, Int(k) < K, values[i] != nil { rows.append(i) }
        }
        rows.sort { (keys[$0]!, $0) < (keys[$1]!, $1) }
        var bad = 0, j = 0
        for k in 0..<K where bad < 5 {
            var xs: [Double] = []
            while j < rows.count && Int(keys[rows[j]]!) == k { xs.append(values[rows[j]]!); j += 1 }
            let want = Self.expectedSum(xs)
            if !same(sums[k], want) {
                bad += 1
                XCTFail("\(label) group \(k): sum \(String(describing: sums[k])) (\(sums[k]?.bitPattern ?? 0)) want \(String(describing: want)) (\(want?.bitPattern ?? 0)) of \(xs.count) values", file: file, line: line)
            }
            checkMean(means[k], xs, "\(label) group \(k)")
        }
        // The public entry points are the same values.
        XCTAssertTrue(zip(try gb.sumDouble(dv).toArray(), sums).allSatisfy { same($0, $1) }, label)
        XCTAssertTrue(zip(try gb.meanDouble(dv).toArray(), means).allSatisfy { same($0, $1) }, label)
    }

    /// Random data over sizes on both sides of the private-table limit (1,024 groups), with null keys,
    /// out-of-range keys, null values and a wide spread of magnitudes.
    func testRandomAgainstCorrectlyRoundedReference() throws {
        try requireRealGPU()
        var rng = Rng(s: 0xC0FFEE)
        for K in [1, 7, 1024, 1025, 5000, 100_000] {
            for n in [1, 999, 100_000, 1_000_000] where n >= K / 4 || n == 1 {
                for spread in [false, true] {
                    var keys: [Int32?] = [], vals: [Double?] = []
                    keys.reserveCapacity(n); vals.reserveCapacity(n)
                    for _ in 0..<n {
                        let r = Int.random(in: 0..<50, using: &rng)
                        keys.append(r == 0 ? nil : (r == 1 ? Int32(K) + 3 : Int32.random(in: 0..<Int32(K), using: &rng)))
                        if Int.random(in: 0..<10, using: &rng) == 0 { vals.append(nil); continue }
                        let x = Double.random(in: -1...1, using: &rng)
                        let e = spread ? Int.random(in: -300...300, using: &rng) : Int.random(in: -3...20, using: &rng)
                        vals.append(x * pow(2.0, Double(e)))
                    }
                    try run(keys: keys, values: vals, K: K, label: "K=\(K) n=\(n) spread=\(spread)")
                }
            }
        }
    }

    /// Hand-built groups: cancellation, signed zeros, subnormals, special values and overflow.
    func testEdgeCases() throws {
        try requireRealGPU()
        let big = 0x1p1023, tiny = Double.leastNonzeroMagnitude, maxd = Double.greatestFiniteMagnitude
        let groups: [[Double]] = [
            [1e300, 1, -1e300],                               // 1, below the window of 1e300
            [0x1p53, 1, -0x1p53],
            [1, 0x1p-60, -1],                                 // 2^-60
            [1e16, 1, 1, 1, -1e16],
            [0x1p60, 0x1p-60, -0x1p60, 0x1p-61],
            [0.1, 0.2, 0.3, -0.6],
            [-0.0], [-0.0, -0.0], [0.0, -0.0], [-0.0, 0.0], [0.0],
            [1, -1],                                          // exact zero from non-zero values: +0.0
            [-1, 1, -0.0],
            [tiny], [tiny, tiny, -tiny], [tiny, -tiny], [0x1p-1022, -tiny], [-tiny, -tiny],
            [0x1p-1022 - tiny, tiny],                         // largest subnormal + smallest = min normal
            [maxd, maxd],                                     // overflows to +inf
            [maxd, maxd, -maxd],                              // exactly maxd
            [-maxd, -maxd],
            [big, big, -big, -big, 1],
            [maxd, 0x1p970],                                  // exactly half an ulp above maxd: ties to +inf
            [maxd, 0x1p969],                                  // below the half-way point: maxd
            [.nan], [1, .nan, 2], [.infinity, 1], [-.infinity, -1], [.infinity, -.infinity],
            [.infinity, .nan], [.infinity, .infinity],
            [Double(bitPattern: 0xFFF8_0000_0000_0000)],      // negative NaN
            [3, Double(bitPattern: 0x7FF0_0000_0000_0001)],   // signalling NaN payload
            [1, 2, 3, 4, 5], [0x1p52 + 1, 0.5],              // ties to even
            [0x1p53, 1],                                      // 2^53 + 1 ties to 2^53
            [0x1p53, 1, 0x1p-80],                             // just above the tie: 2^53 + 2
        ]
        var keys: [Int32?] = [], vals: [Double?] = []
        for (k, g) in groups.enumerated() {
            for v in g { keys.append(Int32(k)); vals.append(v) }
            keys.append(Int32(k)); vals.append(nil)          // a null value in every group
        }
        let K = groups.count + 2                               // two groups with no rows
        try run(keys: keys, values: vals, K: K, label: "edge")
        // Several of the same values spread over rows of other groups, through the device path.
        var k2: [Int32?] = [], v2: [Double?] = []
        for rep in 0..<3 {
            for (k, g) in groups.enumerated() {
                for v in g { k2.append(Int32(k * 700 + rep)); v2.append(v) }
            }
        }
        try run(keys: k2.reversed(), values: v2.reversed(), K: groups.count * 700, label: "edge-dev")

        // The NaN a group returns: the canonical quiet NaN on the GPU, else the first NaN in row order.
        let gb = try MetalArray<Int32>([0, 0, 1, 1, 2]).groupBy(keyCount: 3)
        let s = try gb.sumDouble(try MetalArray<Double>([
            Double(bitPattern: 0x7FF0_0000_0000_0001), Double(bitPattern: 0xFFF8_0000_0000_0005),
            .infinity, -.infinity, .nan])).toArray()
        XCTAssertEqual(s[0]!.bitPattern, 0x7FF8_0000_0000_0001)
        XCTAssertEqual(s[1]!.bitPattern, 0x7FF8_0000_0000_0000)
        XCTAssertEqual(s[2]!.bitPattern, 0x7FF8_0000_0000_0000)
    }

    /// Long cancellation-heavy groups: many large terms that cancel, leaving small ones.
    func testCancellationHeavy() throws {
        try requireRealGPU()
        var rng = Rng(s: 42)
        for K in [3, 2000] {
            var keys: [Int32?] = [], vals: [Double?] = []
            for _ in 0..<200_000 {
                let k = Int32.random(in: 0..<Int32(K), using: &rng)
                let big = Double.random(in: 1e15...1e17, using: &rng)
                let small = Double.random(in: -1...1, using: &rng)
                keys.append(k); vals.append(big)
                keys.append(k); vals.append(-big)
                keys.append(k); vals.append(small)
            }
            try run(keys: keys, values: vals, K: K, label: "cancel K=\(K)")
        }
    }

    /// Never less accurate than the ordered segmented sum it replaced, group by group.
    func testAtLeastAsAccurateAsTheOrderedSum() throws {
        try requireRealGPU()
        var rng = Rng(s: 7)
        let n = 300_000
        for K in [10, 50_000] {
            var keys: [Int32] = [], vals: [Double] = []
            for _ in 0..<n {
                keys.append(Int32.random(in: 0..<Int32(K), using: &rng))
                vals.append(Double.random(in: -1...1, using: &rng) * pow(10, Double(Int.random(in: -8...8, using: &rng))))
            }
            let gb = try MetalArray<Int32>(keys).groupBy(keyCount: K)
            let dv = try MetalArray<Double>(vals)
            let new = try gb.sumDouble(dv).toArray(), old = try gb.sumDoubleOrdered(dv).toArray()
            var g = [[Double]](repeating: [], count: K)
            for i in 0..<n { g[Int(keys[i])].append(vals[i]) }
            var differ = 0
            for k in 0..<K where !g[k].isEmpty {
                let exact = Self.fsum(g[k])
                XCTAssertEqual(new[k]!.bitPattern, exact.bitPattern, "K=\(K) group \(k)")
                // |new - exact| is zero, so it is never larger than |old - exact|.
                if new[k]!.bitPattern != old[k]!.bitPattern { differ += 1 }
            }
            print("K=\(K): \(differ) of \(K) groups differ from the ordered sum")
        }
    }

    /// Past 2^24 groups (the grid limit of the per-group kernels this path does not use).
    func testManyGroups() throws {
        try requireRealGPU()
        let K = (1 << 24) + 3
        var rng = Rng(s: 99)
        var keys = [Int32?](), vals = [Double?]()
        keys.reserveCapacity(K + 100_000); vals.reserveCapacity(K + 100_000)
        for k in 0..<K { keys.append(Int32(k)); vals.append(Double(k % 1000) * 0.001) }
        for _ in 0..<100_000 { keys.append(Int32.random(in: 0..<Int32(K), using: &rng)); vals.append(Double.random(in: -1...1, using: &rng)) }
        keys[5] = nil; vals[7] = nil
        try run(keys: keys, values: vals, K: K, label: "2^24+3")
    }

    /// `min`/`max` over the interleaved table: NaN skipped, -0.0 below +0.0, all-null groups null,
    /// 64-bit keys with a high word of all ones, and narrow types; against a host reference, on both
    /// the private-table and the device-table path, and shared by one call.
    func testExtremaAgainstReference() throws {
        try requireRealGPU()
        var rng = Rng(s: 5)
        for K in [3, 1024, 1025, 70_000] {
            let n = 300_000
            var keys: [Int32?] = [], d: [Double?] = [], l: [Int64?] = [], f: [Float?] = [], u: [UInt32?] = []
            for i in 0..<n {
                keys.append(i % 97 == 0 ? nil : Int32.random(in: 0..<Int32(K), using: &rng))
                if i % 13 == 0 { d.append(nil); l.append(nil); f.append(nil); u.append(nil); continue }
                let pick = Int.random(in: 0..<20, using: &rng)
                let x: Double = pick == 0 ? .nan : (pick == 1 ? -0.0 : (pick == 2 ? 0.0 : (pick == 3 ? .infinity : Double.random(in: -1e6...1e6, using: &rng))))
                d.append(x)
                l.append(pick < 3 ? Int64.max - Int64(pick) : Int64.random(in: .min ... .max, using: &rng))
                f.append(Float.random(in: -1e3...1e3, using: &rng))
                u.append(UInt32.random(in: 0...UInt32.max, using: &rng))
            }
            // Group 1 holds only NaN and nulls; group 2 only nulls (when it exists).
            if K > 2 {
                for i in 0..<n where keys[i] == 1 { d[i] = i % 2 == 0 ? .nan : nil }
                for i in 0..<n where keys[i] == 2 { d[i] = nil; l[i] = nil; f[i] = nil; u[i] = nil }
            }
            let gb = try MetalArray<Int32>(keys).groupBy(keyCount: K)
            func check<T: ArrowPrimitive & Comparable>(_ vals: [T?], _ name: String, lt: (T, T) -> Bool) throws {
                let a = try MetalArray<T>(vals)
                let e = try gb.minMax(a)
                let mn = e.min.toArray(), mx = e.max.toArray()
                // A second call on the same column is the cached pair.
                XCTAssertTrue(try gb.minMax(a).min === e.min, name)
                var lo = [T?](repeating: nil, count: K), hi = [T?](repeating: nil, count: K)
                for i in 0..<n {
                    guard let k = keys[i], let v = vals[i] else { continue }
                    if let dv = v as? Double, dv.isNaN { continue }
                    let kk = Int(k)
                    if lo[kk] == nil || lt(v, lo[kk]!) { lo[kk] = v }
                    if hi[kk] == nil || lt(hi[kk]!, v) { hi[kk] = v }
                }
                for k in 0..<K {
                    if let a = mn[k] as? Double, let b = lo[k] as? Double {
                        XCTAssertEqual(a.bitPattern, b.bitPattern, "\(name) min K=\(K) k=\(k)")
                        XCTAssertEqual((mx[k] as! Double).bitPattern, (hi[k] as! Double).bitPattern, "\(name) max K=\(K) k=\(k)")
                    } else {
                        XCTAssertEqual(mn[k], lo[k], "\(name) min K=\(K) k=\(k)")
                        XCTAssertEqual(mx[k], hi[k], "\(name) max K=\(K) k=\(k)")
                    }
                }
            }
            // Float64 in total order of the sign bit: -0.0 sorts below +0.0.
            try check(d, "f64") { a, b in a < b || (a == 0 && b == 0 && a.sign == .minus && b.sign == .plus) }
            try check(l, "i64", lt: <)
            try check(f, "f32", lt: <)
            try check(u, "u32", lt: <)
        }
    }
}
