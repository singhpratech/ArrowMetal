import XCTest
@testable import ArrowMetal

/// Window functions, shifts, pairwise differences, rolling windows and multi-column sorts, each checked
/// element for element against a sequential Swift oracle written straight from the definition.
final class WindowTests: XCTestCase {

    // MARK: - Oracles

    /// Three-way comparison in the order the ranking functions use: Arrow value order, every NaN one
    /// value ordered after +inf, `-0.0` equal to `0.0`.
    static func compare<T: ArrowPrimitive>(_ a: T, _ b: T) -> Int {
        if let x = a as? Double, let y = b as? Double {
            if x.isNaN && y.isNaN { return 0 }
            if x.isNaN { return 1 }
            if y.isNaN { return -1 }
            return x < y ? -1 : (x > y ? 1 : 0)
        }
        if let x = a as? Float, let y = b as? Float {
            if x.isNaN && y.isNaN { return 0 }
            if x.isNaN { return 1 }
            if y.isNaN { return -1 }
            return x < y ? -1 : (x > y ? 1 : 0)
        }
        return a < b ? -1 : (a > b ? 1 : 0)
    }

    /// The sorted order the ranking functions rank in: non-nulls ascending and stable, then the nulls
    /// in row order, and the 0-based tie-group index of every sorted position.
    static func order<T: ArrowPrimitive>(_ v: [T?]) -> (order: [Int], group: [Int]) {
        let n = v.count
        let sorted = v.indices.filter { v[$0] != nil }.sorted { i, j in
            let c = compare(v[i]!, v[j]!)
            return c < 0 || (c == 0 && i < j)
        }
        let ord = sorted + v.indices.filter { v[$0] == nil }
        var g = [Int](repeating: 0, count: n)
        for p in 1..<Swift.max(n, 1) where p < n {
            let a = v[ord[p - 1]], b = v[ord[p]]
            let same: Bool
            if a == nil && b == nil { same = true }
            else if a == nil || b == nil { same = false }
            else { same = compare(a!, b!) == 0 }
            g[p] = same ? g[p - 1] : g[p - 1] + 1
        }
        return (ord, g)
    }

    struct Ranking {
        var rowNumber: [Int32] = [], rank: [Int32] = [], dense: [Int32] = []
        var percent: [Double] = [], cume: [Double] = []
    }

    static func rankingOracle<T: ArrowPrimitive>(_ v: [T?]) -> Ranking {
        let n = v.count
        var r = Ranking()
        guard n > 0 else { return r }
        let (ord, g) = order(v)
        var start = [Int](repeating: 0, count: n), end = start
        for p in 0..<n {
            if p == 0 || g[p] != g[p - 1] { start[g[p]] = p }
            if p == n - 1 || g[p] != g[p + 1] { end[g[p]] = p + 1 }
        }
        r.rowNumber = [Int32](repeating: 0, count: n); r.rank = r.rowNumber; r.dense = r.rowNumber
        r.percent = [Double](repeating: 0, count: n); r.cume = r.percent
        for p in 0..<n {
            let row = ord[p], d = g[p]
            r.rowNumber[row] = Int32(p + 1)
            r.rank[row] = Int32(start[d] + 1)
            r.dense[row] = Int32(d + 1)
            r.percent[row] = n <= 1 ? 0 : Double(start[d]) / Double(n - 1)
            r.cume[row] = Double(end[d]) / Double(n)
        }
        return r
    }

    /// Every trailing-window aggregate in one pass, so a big oracle costs one sweep instead of four.
    static func rollingOracle<T: ArrowPrimitive>(_ v: [T?], w: Int, minPeriods: Int)
        -> (sum: [T?], min: [T?], max: [T?], mean: [Double?]) {
        let n = v.count
        var s = [T?](repeating: nil, count: n), lo = s, hi = s
        var mean = [Double?](repeating: nil, count: n)
        for i in 0..<n {
            var cnt = 0
            var acc: T = 0, mn = T.maxValue, mx = T.minValue
            var dsum = 0.0
            for j in Swift.max(0, i - w + 1)...i {
                guard let x = v[j] else { continue }
                cnt += 1
                acc = T.wrappingApply(.add, acc, x)
                dsum += x.asDouble
                if x.asDouble.isNaN { continue }     // NaN is skipped by min/max, as in the kernels
                if x < mn { mn = x }
                if x > mx { mx = x }
            }
            guard cnt >= minPeriods else { continue }
            s[i] = acc; lo[i] = mn; hi[i] = mx
            mean[i] = dsum / Double(cnt)
        }
        return (s, lo, hi, mean)
    }

    // MARK: - Comparison helpers

    func expectEqual<T: ArrowPrimitive>(_ got: [T?], _ want: [T?], _ what: String,
                                        file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(got.count, want.count, what, file: file, line: line)
        for i in 0..<Swift.min(got.count, want.count) where got[i] != want[i] {
            // NaN never equals itself; treat two NaNs as a match.
            if let a = got[i]?.asDouble, let b = want[i]?.asDouble, a.isNaN, b.isNaN { continue }
            XCTFail("\(what): index \(i) got \(String(describing: got[i])) want \(String(describing: want[i]))",
                    file: file, line: line)
            return
        }
    }

    /// Relative comparison for the float paths that reassociate (prefix-sum windows, scanned sums).
    func expectClose(_ got: [Double?], _ want: [Double?], _ what: String, tol: Double = 1e-9,
                     file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(got.count, want.count, what, file: file, line: line)
        for i in 0..<Swift.min(got.count, want.count) {
            switch (got[i], want[i]) {
            case (nil, nil): continue
            case (let a?, let b?):
                if a.isNaN && b.isNaN { continue }
                let scale = Swift.max(1.0, Swift.abs(a), Swift.abs(b))
                if Swift.abs(a - b) <= tol * scale { continue }
                XCTFail("\(what): index \(i) got \(a) want \(b)", file: file, line: line); return
            default:
                XCTFail("\(what): index \(i) nullness differs", file: file, line: line); return
            }
        }
    }

    // MARK: - Ranking

    func checkRanking<T: ArrowPrimitive>(_ v: [T?], file: StaticString = #filePath, line: UInt = #line) throws {
        let a = try MetalArray<T>(v)
        let want = Self.rankingOracle(v)
        let what = "\(T.self) n=\(v.count)"
        XCTAssertEqual(try a.rowNumber().toRawArray(), want.rowNumber, "row_number \(what)", file: file, line: line)
        XCTAssertEqual(try a.rank().toRawArray(), want.rank, "rank \(what)", file: file, line: line)
        XCTAssertEqual(try a.denseRank().toRawArray(), want.dense, "dense_rank \(what)", file: file, line: line)
        XCTAssertEqual(try a.percentRank().toRawArray(), want.percent, "percent_rank \(what)", file: file, line: line)
        XCTAssertEqual(try a.cumeDist().toRawArray(), want.cume, "cume_dist \(what)", file: file, line: line)
    }

    func testRanking() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        for n in [0, 1, 33, 4097] {
            // heavy ties, nulls every seventh row
            let ties: [Int32?] = (0..<n).map { (i: Int) -> Int32? in i % 7 == 0 ? nil : Int32(i % 5) }
            let rnd: [Int32?] = (0..<n).map { _ in Int32.random(in: -50...50, using: &g) }
            let wide: [Int64?] = (0..<n).map { (i: Int) -> Int64? in i % 11 == 0 ? nil : Int64(i % 3) * 1_000_000_007 }
            let f32: [Float?] = (0..<n).map { (i: Int) -> Float? in
                if i % 13 == 0 { return nil }
                if i % 17 == 0 { return Float.nan }
                if i % 19 == 0 { return -Float.nan }
                if i % 23 == 0 { return -0.0 }
                return Float(i % 6)
            }
            let f64: [Double?] = (0..<n).map { (i: Int) -> Double? in
                if i % 9 == 0 { return nil }
                if i % 29 == 0 { return Double.nan }
                if i % 31 == 0 { return -0.0 }
                return Double(i % 4) * 1.5
            }
            let small: [UInt8?] = (0..<n).map { UInt8($0 % 3) }
            try checkRanking(ties); try checkRanking(rnd); try checkRanking(wide)
            try checkRanking(f32); try checkRanking(f64); try checkRanking(small)
        }
        // every row null, and every row the same value
        try checkRanking([Int32?](repeating: nil, count: 40))
        try checkRanking([Int32?](repeating: 7, count: 40))
    }

    func testRankingLarge() throws {
        try requireRealGPU()
        let n = 300_003
        let ints: [Int32?] = (0..<n).map { (i: Int) -> Int32? in i % 100 == 0 ? nil : Int32(i % 977) }
        let doubles: [Double?] = (0..<n).map { (i: Int) -> Double? in i % 50 == 0 ? nil : Double(i % 1013) }
        try checkRanking(ints)
        try checkRanking(doubles)
    }

    // MARK: - Shift and pairwise

    func testShift() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        for n in [0, 1, 33, 4097] {
            let v: [Int32?] = (0..<n).map { (i: Int) -> Int32? in
                i % 5 == 0 ? nil : Int32.random(in: -100...100, using: &g)
            }
            let a = try MetalArray<Int32>(v)
            for by in [-4097, -7, -1, 0, 1, 7, 4097] {
                let got = try a.shift(by: by).toArray()
                let want: [Int32?] = (0..<n).map { i in let j = i - by; return (j >= 0 && j < n) ? v[j] : nil }
                expectEqual(got, want, "shift(by: \(by)) n=\(n)")
                let filled = try a.shift(by: by, fill: -1).toArray()
                let wantFilled: [Int32?] = (0..<n).map { i in let j = i - by; return (j >= 0 && j < n) ? v[j] : -1 }
                expectEqual(filled, wantFilled, "shift(by: \(by), fill:) n=\(n)")
            }
        }
        // float64 keeps the exact bit pattern through the shift
        let d: [Double?] = [1.5, nil, -0.0, 3.25, .infinity]
        expectEqual(try MetalArray<Double>(d).shift(by: 2, fill: 9.5).toArray(), [9.5, 9.5, 1.5, nil, -0.0], "float64 shift")
        expectEqual(try MetalArray<Double>(d).shift(by: -1).toArray(), [nil, -0.0, 3.25, .infinity, nil], "float64 lead")
    }

    func testPairwiseDiff() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        func check<T: ArrowPrimitive>(_ v: [T?], _ period: Int) throws {
            let n = v.count
            let got = try MetalArray<T>(v).pairwiseDiff(period: period).toArray()
            let want: [T?] = (0..<n).map { i in
                let j = i - period
                guard j >= 0, j < n, let a = v[i], let b = v[j] else { return nil }
                return T.wrappingApply(.sub, a, b)
            }
            expectEqual(got, want, "pairwise_diff(\(period)) \(T.self) n=\(n)")
        }
        for n in [0, 1, 33, 4097] {
            let i32: [Int32?] = (0..<n).map { (i: Int) -> Int32? in
                i % 6 == 0 ? nil : Int32.random(in: -1000...1000, using: &g)
            }
            let i8: [Int8?] = (0..<n).map { _ in Int8.random(in: Int8.min...Int8.max, using: &g) }
            let f32: [Float?] = (0..<n).map { (i: Int) -> Float? in i % 8 == 0 ? nil : Float(i % 41) * 0.25 }
            let f64: [Double?] = (0..<n).map { (i: Int) -> Double? in i % 9 == 0 ? nil : Double(i % 37) * 1.0e-3 }
            for p in [-3, -1, 1, 2, 33] {
                try check(i32, p); try check(i8, p); try check(f32, p); try check(f64, p)
            }
        }
        let big: [Int64?] = (0..<300_003).map { Int64($0) * 3 }
        try check(big, 1)
    }

    // MARK: - Cumulative product and mean

    func testCumulativeProd() throws {
        try requireRealGPU()
        func check<T: ArrowPrimitive>(_ v: [T?]) throws {
            let got = try MetalArray<T>(v).cumulativeProd().toArray()
            var acc: T = 1
            let want: [T?] = v.map { x in
                guard let x else { return nil }
                acc = T.wrappingApply(.mul, acc, x)
                return acc
            }
            expectEqual(got, want, "cumulative_prod \(T.self) n=\(v.count)")
        }
        for n in [0, 1, 33, 4097] {
            let i32: [Int32?] = (0..<n).map { (i: Int) -> Int32? in
                i % 5 == 0 ? nil : Int32(i % 7 == 0 ? -1 : (i % 3) + 1)
            }
            let i64: [Int64?] = (0..<n).map { Int64($0 % 4 + 1) }            // wraps well past 2^63
            let u8: [UInt8?] = (0..<n).map { (i: Int) -> UInt8? in i % 11 == 0 ? nil : UInt8(i % 5 + 1) }
            // Sign flips only: a long float32 run of 2.0 and 0.5 drifts into the subnormals, which
            // Apple GPUs flush to zero, so magnitudes are exercised by the short case below instead.
            let f32: [Float?] = (0..<n).map { (i: Int) -> Float? in
                i % 6 == 0 ? nil : (i % 2 == 0 ? Float(-1.0) : Float(1.0))
            }
            try check(i32); try check(i64); try check(u8); try check(f32)
        }
        let alternating: [Int32?] = (0..<300_003).map { Int32($0 % 2 == 0 ? 1 : -1) }
        try check(alternating)
        try check([Float?](arrayLiteral: 2.0, nil, 0.5, 4.0, 0.25, -8.0))
        // float64 goes through the software multiplier
        expectEqual(try MetalArray<Double>([2.0, nil, 0.5, -4.0]).cumulativeProd().toArray(),
                    [2.0, nil, 1.0, -4.0], "cumulative_prod float64")
    }

    func testCumulativeMean() throws {
        try requireRealGPU()
        /// Exact for inputs whose running sums are all representable in binary64.
        func checkExact<T: ArrowPrimitive>(_ v: [T?]) throws {
            let got = try MetalArray<T>(v).cumulativeMean().toArray()
            var sum = 0.0, cnt = 0
            let want: [Double?] = v.map { x in
                guard let x else { return nil }
                sum += x.asDouble; cnt += 1
                return sum / Double(cnt)
            }
            expectEqual(got, want, "cumulative_mean \(T.self) n=\(v.count)")
        }
        for n in [0, 1, 33, 4097] {
            let i32: [Int32?] = (0..<n).map { (i: Int) -> Int32? in i % 5 == 0 ? nil : Int32(i % 17) }
            let i8: [Int8?] = (0..<n).map { Int8($0 % 100 - 50) }
            let u64: [UInt64?] = (0..<n).map { (i: Int) -> UInt64? in i % 3 == 0 ? nil : UInt64(i % 1000) }
            let f64: [Double?] = (0..<n).map { (i: Int) -> Double? in i % 7 == 0 ? nil : Double(i % 8) * 0.25 }
            let f32: [Float?] = (0..<n).map { (i: Int) -> Float? in i % 4 == 0 ? nil : Float(i % 16) * 0.5 }
            try checkExact(i32); try checkExact(i8); try checkExact(u64)
            try checkExact(f64); try checkExact(f32)                        // dyadic values: exact
        }
        let large: [Int32?] = (0..<300_003).map { (i: Int) -> Int32? in i % 13 == 0 ? nil : Int32(i % 101) }
        try checkExact(large)
        // Reassociating scan: irrational-ish float64 values agree to a relative 1e-12.
        let noisy: [Double?] = (0..<4097).map { (i: Int) -> Double? in
            i % 6 == 0 ? nil : Double(i) * 0.1 + 1.0 / Double(i + 1)
        }
        var sum = 0.0, cnt = 0
        let want: [Double?] = noisy.map { x in
            guard let x else { return nil }
            sum += x; cnt += 1
            return sum / Double(cnt)
        }
        expectClose(try MetalArray<Double>(noisy).cumulativeMean().toArray(), want, "cumulative_mean noisy", tol: 1e-12)
    }

    // MARK: - Rolling windows

    func checkRolling<T: ArrowPrimitive>(_ v: [T?], w: Int, minPeriods: Int?, exact: Bool,
                                         file: StaticString = #filePath, line: UInt = #line) throws {
        let a = try MetalArray<T>(v)
        let mp = minPeriods ?? w
        let want = Self.rollingOracle(v, w: w, minPeriods: mp)
        let what = "\(T.self) n=\(v.count) w=\(w) minPeriods=\(mp)"
        expectEqual(try a.rollingMin(window: w, minPeriods: minPeriods).toArray(), want.min, "rolling_min \(what)", file: file, line: line)
        expectEqual(try a.rollingMax(window: w, minPeriods: minPeriods).toArray(), want.max, "rolling_max \(what)", file: file, line: line)
        let sum = try a.rollingSum(window: w, minPeriods: minPeriods).toArray()
        let mean = try a.rollingMean(window: w, minPeriods: minPeriods).toArray()
        if exact {
            expectEqual(sum, want.sum, "rolling_sum \(what)", file: file, line: line)
            expectClose(mean, want.mean, "rolling_mean \(what)", tol: 0, file: file, line: line)
        } else {
            expectClose(sum.map { $0?.asDouble }, want.sum.map { $0?.asDouble }, "rolling_sum \(what)", tol: 1e-6, file: file, line: line)
            expectClose(mean, want.mean, "rolling_mean \(what)", tol: 1e-6, file: file, line: line)
        }
    }

    func testRollingWindows() throws {
        try requireRealGPU()
        for n in [0, 1, 33, 4097] {
            // Small integers: the prefix-sum trick is exact, so every aggregate is checked bit for bit.
            let i32: [Int32?] = (0..<n).map { (i: Int) -> Int32? in i % 5 == 0 ? nil : Int32(i % 61) - 30 }
            let i64: [Int64?] = (0..<n).map { (i: Int) -> Int64? in i % 9 == 0 ? nil : Int64(i % 7) }
            let f32: [Float?] = (0..<n).map { (i: Int) -> Float? in i % 6 == 0 ? nil : Float(i % 33) * 0.5 }
            let f64: [Double?] = (0..<n).map { (i: Int) -> Double? in i % 8 == 0 ? nil : Double(i % 21) * 0.25 }
            for w in [1, 7, 64, 4096] {
                // The oracle is O(n · w); keep the one expensive combination to a single sweep.
                let cheap = w * Swift.max(n, 1) <= 400_000
                let mps: [Int?] = cheap ? [nil, 1, Swift.min(w, 3)] : [nil]
                for mp in mps {
                    try checkRolling(i32, w: w, minPeriods: mp, exact: true)
                    guard cheap else { continue }
                    try checkRolling(i64, w: w, minPeriods: mp, exact: true)
                    try checkRolling(f32, w: w, minPeriods: mp, exact: true)   // dyadic values: exact too
                    try checkRolling(f64, w: w, minPeriods: mp, exact: true)
                }
            }
        }
        // A window with nothing but nulls in it, and one with a single value.
        let sparse: [Int32?] = [nil, nil, 5, nil, nil, nil, nil, 9, nil]
        try checkRolling(sparse, w: 3, minPeriods: nil, exact: true)
        try checkRolling(sparse, w: 3, minPeriods: 1, exact: true)
    }

    /// NaN is skipped by rolling min/max but still counts as a row towards `minPeriods`. It is *not*
    /// checked against rolling sum or mean: those run on prefix sums, and one NaN (or infinity) in the
    /// prefix poisons every window after it, which the documentation on those two calls says outright.
    func testRollingMinMaxWithNaN() throws {
        try requireRealGPU()
        let v: [Float?] = [1.0, .nan, 3.0, nil, .nan, 2.0]
        let a = try MetalArray<Float>(v)
        for (w, mp) in [(2, 1), (3, 2), (6, 1)] {
            let want = Self.rollingOracle(v, w: w, minPeriods: mp)
            expectEqual(try a.rollingMin(window: w, minPeriods: mp).toArray(), want.min, "rolling_min NaN w=\(w)")
            expectEqual(try a.rollingMax(window: w, minPeriods: mp).toArray(), want.max, "rolling_max NaN w=\(w)")
        }
        // A window whose only non-null values are NaN falls back to the neutral element.
        let allNaN = try MetalArray<Float>([Float?](repeating: .nan, count: 4))
        XCTAssertEqual(try allNaN.rollingMin(window: 2, minPeriods: 1).toArray(), [.infinity, .infinity, .infinity, .infinity])
        XCTAssertEqual(try allNaN.rollingMax(window: 2, minPeriods: 1).toArray(), [-.infinity, -.infinity, -.infinity, -.infinity])
    }

    func testRollingLarge() throws {
        try requireRealGPU()
        let n = 300_003
        let v: [Int32?] = (0..<n).map { (i: Int) -> Int32? in i % 11 == 0 ? nil : Int32(i % 251) - 125 }
        try checkRolling(v, w: 1, minPeriods: nil, exact: true)
        try checkRolling(v, w: 64, minPeriods: 1, exact: true)
        // Float64 at scale: the prefix difference reassociates, so compare with a relative tolerance.
        let d: [Double?] = (0..<n).map { (i: Int) -> Double? in i % 7 == 0 ? nil : Double(i % 997) * 0.125 + 0.001 }
        try checkRolling(d, w: 8, minPeriods: nil, exact: false)
    }

    func testRollingRejectsBadParameters() throws {
        let a = try MetalArray<Int32>([1, 2, 3])
        XCTAssertThrowsError(try a.rollingSum(window: 0))
        XCTAssertThrowsError(try a.rollingMin(window: 4, minPeriods: 5))
        XCTAssertThrowsError(try a.rollingMean(window: 4, minPeriods: 0))
    }

    // MARK: - Multi-column sort

    /// Stable lexicographic order of the rows, nulls last in every key, as an index array.
    static func lexOracle(_ keys: [[Int64?]], _ descending: [Bool], count: Int) -> [Int32] {
        (0..<count).sorted { i, j in
            for (k, col) in keys.enumerated() {
                let a = col[i], b = col[j]
                if a == nil && b == nil { continue }
                if a == nil { return false }            // nulls last, both directions
                if b == nil { return true }
                if a! == b! { continue }
                return descending[k] ? (a! > b!) : (a! < b!)
            }
            return i < j                                 // stable
        }.map { Int32($0) }
    }

    func testLexsortIndices() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        for n in [0, 1, 33, 4097, 300_003] {
            // Three keys with heavy ties so the tie-breaking chain is exercised at every level.
            let k0: [Int64?] = (0..<n).map { (i: Int) -> Int64? in i % 13 == 0 ? nil : Int64(i % 3) }
            let k1: [Int64?] = (0..<n).map { (i: Int) -> Int64? in i % 17 == 0 ? nil : Int64(i % 7) }
            let k2: [Int64?] = (0..<n).map { _ in Int64.random(in: 0..<1000, using: &g) }
            let cols: [AnyMetalArray] = [.int64(try MetalArray<Int64>(k0)),
                                         .int32(try MetalArray<Int32>(k1.map { $0.map(Int32.init) })),
                                         .float64(try MetalArray<Double>(k2.map { $0.map(Double.init) }))]
            let combos: [[Bool]] = n > 100_000
                ? [[false, false, false], [true, false, true]]
                : [[false, false, false], [true, false, true], [false, true, false], [true, true, true]]
            for desc in combos {
                let got = try lexsortIndices(cols, descending: desc).toRawArray()
                XCTAssertEqual(got, Self.lexOracle([k0, k1, k2], desc, count: n), "lexsort n=\(n) desc=\(desc)")
            }
        }
    }

    func testRecordBatchMultiKeySort() throws {
        try requireRealGPU()
        let region = try MetalArray<Int32>([2, 1, 2, 1, 3, nil, 2])
        let revenue = try MetalArray<Double>([10.0, 5.0, 30.0, 5.0, 1.0, 99.0, nil])
        let label = try MetalStringArray(["a", "b", "c", "d", "e", "f", "g"])
        let batch = try MetalRecordBatch(names: ["region", "revenue", "label"],
                                         columns: [.int32(region), .float64(revenue), .string(label)])
        let sorted = try batch.sorted(by: [("region", false), ("revenue", true)])
        XCTAssertEqual(sorted["label"]!.asString!.toArray(), ["b", "d", "c", "a", "g", "e", "f"])
        XCTAssertEqual(sorted["region"]!.asInt32!.toArray(), [1, 1, 2, 2, 2, 3, nil])
        // A single key behaves exactly like the one-column overload.
        let one = try batch.sorted(by: [("region", false)])
        XCTAssertEqual(one["label"]!.asString!.toArray(), try batch.sorted(by: "region")["label"]!.asString!.toArray())
        XCTAssertThrowsError(try batch.sorted(by: [("nope", false)]))
        XCTAssertThrowsError(try batch.sorted(by: [("label", false)]))   // no order-preserving key for utf8
    }

    func testPartitionNthIndices() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        for n in [0, 1, 33, 4097] {
            let v: [Int32?] = (0..<n).map { (i: Int) -> Int32? in
                i % 7 == 0 ? nil : Int32.random(in: -500...500, using: &g)
            }
            let a = try MetalArray<Int32>(v)
            for k in Set([0, n / 2, Swift.max(0, n - 1)]) {
                let idx = try a.partitionNthIndices(k).toRawArray()
                XCTAssertEqual(Set(idx).count, n, "permutation n=\(n)")
                guard n > 0, k < n else { continue }
                let pivot = v[Int(idx[k])]
                for p in 0..<k {
                    let x = v[Int(idx[p])]
                    XCTAssertTrue(pivot == nil || (x != nil && x! <= pivot!), "left of pivot n=\(n) k=\(k)")
                }
            }
        }
        XCTAssertThrowsError(try MetalArray<Int32>([1, 2, 3]).partitionNthIndices(4))
    }
}
