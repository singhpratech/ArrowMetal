import XCTest
@testable import ArrowMetal

/// Top-k selection must agree with the full sort it replaces, index for index.
final class TopKTests: XCTestCase {
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

    /// The oracle: what `topK` did before there was a selection kernel — a full argsort, then a slice.
    private func viaSort<T: ArrowPrimitive>(_ a: MetalArray<T>, _ k: Int, largest: Bool) throws -> [Int32] {
        let idx = try a.argsort(descending: largest)
        return try idx.slice(offset: 0, length: Swift.min(k, idx.length)).toRawArray()
    }

    private func check<T: ArrowPrimitive>(_ a: MetalArray<T>, _ k: Int, largest: Bool,
                                          file: StaticString = #filePath, line: UInt = #line) throws {
        let got = try a.topK(k, largest: largest).toRawArray()
        let want = try viaSort(a, k, largest: largest)
        XCTAssertEqual(got, want, "\(T.self) n=\(a.length) k=\(k) largest=\(largest)", file: file, line: line)
    }

    func testAgainstSortOnFiveMillion() throws {
        try requireRealGPU()
        var rng = Rng(s: 0xA11CE)
        let n = 5_000_000
        // Heavy duplication (values repeat about 5000 times) so tie-breaking by row index is exercised.
        let raw: [Int64] = (0..<n).map { _ in Int64.random(in: 0..<1000, using: &rng) }
        let dense = try MetalArray<Int64>(raw)
        let nullable = try MetalArray<Int64>(raw.enumerated().map { $0.offset % 17 == 0 ? nil : $0.element })
        let spread = try MetalArray<Int64>((0..<n).map { _ in Int64.random(in: Int64.min...Int64.max, using: &rng) })
        let doubles = try MetalArray<Double>((0..<n).map { i in
            i % 2003 == 0 ? Double.nan : Double.random(in: -1e9...1e9, using: &rng)
        })
        for k in [1, 10, 100, 1000] {
            for largest in [true, false] {
                try check(dense, k, largest: largest)
                try check(nullable, k, largest: largest)
                try check(spread, k, largest: largest)
                try check(doubles, k, largest: largest)
            }
        }
    }

    func testSmallAndAwkwardShapes() throws {
        try requireRealGPU()
        var rng = Rng(s: 0xBEEF)
        for n in [0, 1, 2, 255, 256, 257, 1023, 1024, 1025, 4096, 100_003] {
            let a = try MetalArray<Int32>((0..<n).map { _ in
                Int.random(in: 0..<8, using: &rng) == 0 ? nil : Int32.random(in: -50...50, using: &rng)
            })
            let f = try MetalArray<Float>((0..<n).map { i in i % 31 == 0 ? Float.nan : Float.random(in: -1...1, using: &rng) })
            let u = try MetalArray<UInt64>((0..<n).map { _ in UInt64.random(in: 0...9, using: &rng) })
            for k in [1, 2, 100, 1000, 1024, 1025] where k <= Swift.max(n, 1) {
                for largest in [true, false] {
                    try check(a, k, largest: largest)
                    try check(f, k, largest: largest)
                    try check(u, k, largest: largest)
                }
            }
            // k larger than the array: both paths clamp to the array length.
            XCTAssertEqual(try a.topK(n + 10).length, n)
        }
        XCTAssertEqual(try MetalArray<Int32>([5, 1, 3]).topK(0).length, 0)
    }

    /// Every row null, or fewer non-null rows than k: the selection kernel steps aside for the sort,
    /// which still has to place the null rows.
    func testMostlyNull() throws {
        try requireRealGPU()
        let a = try MetalArray<Int64>([nil, 7, nil, nil, 3, nil])
        try check(a, 4, largest: true)
        try check(a, 4, largest: false)
        try check(a, 2, largest: true)
        let allNull = try MetalArray<Int64>([Int64?](repeating: nil, count: 500))
        try check(allNull, 10, largest: true)
    }

    /// The result is what the caller actually wants: the k largest values, in order.
    func testValuesNotJustIndices() throws {
        try requireRealGPU()
        var rng = Rng(s: 7)
        let vals = (0..<200_000).map { _ in Int64.random(in: -1_000_000...1_000_000, using: &rng) }
        let a = try MetalArray<Int64>(vals)
        let top = try a.take(try a.topK(50)).toRawArray()
        XCTAssertEqual(top, Array(vals.sorted(by: >).prefix(50)))
        let bottom = try a.take(try a.topK(50, largest: false)).toRawArray()
        XCTAssertEqual(bottom, Array(vals.sorted().prefix(50)))
    }

    // MARK: - Radix select (Kernels/RadixSelect.swift)
    //
    // `topK` picks between three implementations by (n, k); these check the answer against the CPU oracle
    // both through `topK` and through the radix-select entry point directly, so a routing change cannot
    // quietly stop covering a path.

    /// Both the routed answer and the radix-select path, against the oracle.
    private func checkPaths<T: ArrowPrimitive>(_ vals: [T?], _ k: Int, largest: Bool, _ label: String,
                                               file: StaticString = #filePath, line: UInt = #line) throws {
        let a = try MetalArray<T>(vals)
        let want = cpuTopK(vals, k, largest: largest)
        XCTAssertEqual(try a.topK(k, largest: largest).toRawArray(), want, "topK \(label)", file: file, line: line)
        // Nil means the path declined (k close to n, or fewer than k non-null rows): the sort answers those.
        if let direct = try a.topKRadixSelect(k, largest: largest) {
            XCTAssertEqual(direct.toRawArray(), want, "radix \(label)", file: file, line: line)
        }
    }

    /// Every key type, both directions, every null ratio, over the shapes that straddle the sub-block,
    /// threadgroup and candidate-budget boundaries.
    func testRadixSelectAllTypesAndDirections() throws {
        try requireRealGPU()
        var rng = Rng(s: 0xF00D)
        for n in [1, 33, 4097] {
            for nullRatio in [0.0, 0.1, 0.5, 0.99] {
                func nulled<V>(_ make: (Int) -> V) -> [V?] {
                    (0..<n).map { i in Double.random(in: 0..<1, using: &rng) < nullRatio ? nil : make(i) }
                }
                for k in [1, 17, 100, 1024, 1025] where k <= n {
                    for largest in [true, false] {
                        let label = "n=\(n) k=\(k) nulls=\(nullRatio) largest=\(largest)"
                        try checkPaths(nulled { _ in Int8.random(in: .min ... .max, using: &rng) }, k, largest: largest, "i8 " + label)
                        try checkPaths(nulled { _ in UInt8.random(in: .min ... .max, using: &rng) }, k, largest: largest, "u8 " + label)
                        try checkPaths(nulled { _ in Int16.random(in: .min ... .max, using: &rng) }, k, largest: largest, "i16 " + label)
                        try checkPaths(nulled { _ in UInt16.random(in: .min ... .max, using: &rng) }, k, largest: largest, "u16 " + label)
                        try checkPaths(nulled { _ in Int32.random(in: .min ... .max, using: &rng) }, k, largest: largest, "i32 " + label)
                        try checkPaths(nulled { _ in UInt32.random(in: .min ... .max, using: &rng) }, k, largest: largest, "u32 " + label)
                        try checkPaths(nulled { _ in Int64.random(in: .min ... .max, using: &rng) }, k, largest: largest, "i64 " + label)
                        try checkPaths(nulled { _ in UInt64.random(in: .min ... .max, using: &rng) }, k, largest: largest, "u64 " + label)
                        try checkPaths(nulled { i in i % 97 == 0 ? Float.nan : Float.random(in: -1e6...1e6, using: &rng) },
                                       k, largest: largest, "f32 " + label)
                        try checkPaths(nulled { i in i % 89 == 0 ? Double.nan : Double.random(in: -1e9...1e9, using: &rng) },
                                       k, largest: largest, "f64 " + label)
                    }
                }
            }
        }
        // n = 0: every k answers with an empty result.
        XCTAssertEqual(try MetalArray<Int64>([Int64]()).topK(1).length, 0)
        XCTAssertEqual(try MetalArray<Double>([Double]()).topK(100, largest: false).length, 0)
    }

    /// A million rows: k on both sides of the selection kernel's 1024 limit, up to k = n, and k = n - 1 and
    /// k = n, which no selection path can shortcut.
    func testRadixSelectLargeShapes() throws {
        try requireRealGPU()
        var rng = Rng(s: 0xC0FFEE)
        let n = 1_000_003
        let i64 = try MetalArray<Int64>((0..<n).map { _ in Int64.random(in: .min ... .max, using: &rng) })
        let f64 = try MetalArray<Double>((0..<n).map { i in
            i % 2003 == 0 ? Double.nan : Double.random(in: -1e9...1e9, using: &rng)
        })
        let nullable = try MetalArray<Int64>((0..<n).map { i in
            i % 10 == 0 ? nil : Int64.random(in: .min ... .max, using: &rng)
        })
        // At this size the oracle is the full argsort — the definition `topK` has to match — which the CPU
        // oracle in the stress test below independently pins down at the smaller sizes.
        func both<T: ArrowPrimitive>(_ a: MetalArray<T>, _ k: Int, _ largest: Bool, _ label: String) throws {
            let want = try viaSort(a, k, largest: largest)
            XCTAssertEqual(try a.topK(k, largest: largest).toRawArray(), want, "topK " + label)
            if let direct = try a.topKRadixSelect(k, largest: largest) {
                XCTAssertEqual(direct.toRawArray(), want, "radix " + label)
            }
        }
        for k in [1, 17, 100, 1024, 1025, 10_000, 100_000, n - 1, n] {
            for largest in [true, false] {
                let label = "n=\(n) k=\(k) largest=\(largest)"
                try both(i64, k, largest, "i64 " + label)
                if k <= 100_000 {        // k near n falls back to the sort for every type; once is enough
                    try both(f64, k, largest, "f64 " + label)
                    try both(nullable, k, largest, "i64-nulls " + label)
                }
            }
        }
    }

    /// Data the histogram cannot split: three distinct values, every value the same, and the float values
    /// the key deliberately merges.
    func testRadixSelectTiesAndSpecialValues() throws {
        try requireRealGPU()
        var rng = Rng(s: 0x7E5)
        for n in [4097, 300_007] {
            let three: [Int64?] = (0..<n).map { _ in [Int64.min, 0, Int64.max].randomElement(using: &rng)! }
            let same = [Int64?](repeating: 42, count: n)
            let sameNulls: [Int64?] = (0..<n).map { i in i % 3 == 0 ? nil : 42 }
            for k in [1, 17, 1024, 10_000] where k <= n / 2 {
                for largest in [true, false] {
                    try checkPaths(three, k, largest: largest, "3-distinct n=\(n) k=\(k) largest=\(largest)")
                    try checkPaths(same, k, largest: largest, "all-equal n=\(n) k=\(k) largest=\(largest)")
                    try checkPaths(sameNulls, k, largest: largest, "all-equal+nulls n=\(n) k=\(k) largest=\(largest)")
                }
            }
        }
        // -0.0 ties with +0.0 and every NaN is one value after +inf, in both directions.
        let odd: [Double?] = (0..<20_000).map { i in
            switch i % 5 {
            case 0: return -0.0
            case 1: return 0.0
            case 2: return Double.nan
            case 3: return i % 10 == 3 ? Double.infinity : -Double.infinity
            default: return Double(i % 7) - 3
            }
        }
        let oddF: [Float?] = odd.map { $0.map { Float($0) } }
        for k in [1, 17, 1024, 5000] {
            for largest in [true, false] {
                try checkPaths(odd, k, largest: largest, "f64-odd k=\(k) largest=\(largest)")
                try checkPaths(oddF, k, largest: largest, "f32-odd k=\(k) largest=\(largest)")
            }
        }
    }

    /// 50M rows, the size the benchmark and the performance target are stated at. Release-only and opt-in:
    /// set `ARROWMETAL_BIG=1`.
    func testRadixSelectFiftyMillionRows() throws {
        guard ProcessInfo.processInfo.environment["ARROWMETAL_BIG"] != nil else { return }
        try requireRealGPU()
        var rng = Rng(s: 0x5150)
        let n = 50_000_000
        var raw = [Int64](repeating: 0, count: n)
        for i in 0..<n { raw[i] = Int64(bitPattern: rng.next()) }
        let a = try MetalArray<Int64>(raw)
        // The oracle here is the full argsort, which is the definition `topK` has to match.
        for k in [1, 100, 1024, 1025, 10_000, 100_000] {
            for largest in [true, false] {
                let want = try a.argsort(descending: largest).slice(offset: 0, length: k).toRawArray()
                XCTAssertEqual(try a.topK(k, largest: largest).toRawArray(), want, "n=\(n) k=\(k) largest=\(largest)")
            }
        }
    }

    // MARK: - kthElement and the quantile that rides on it

    func testKthElementMatchesSortedValue() throws {
        try requireRealGPU()
        var rng = Rng(s: 0x0BADF00D)
        func check<T: ArrowPrimitive>(_ vals: [T?], _ label: String) throws {
            let a = try MetalArray<T>(vals)
            let sortedAsc = try a.sorted().toRawArray()
            let sortedDesc = try a.sorted(descending: true).toRawArray()
            let m = a.validCount
            guard m > 0 else {
                XCTAssertNil(try a.kthElement(1), label)
                return
            }
            for r in Set([1, 2, 3, m / 2, m - 1, m].filter { $0 >= 1 && $0 <= m }) {
                XCTAssertEqual(try a.kthElement(r)?.asDouble, sortedAsc[r - 1].asDouble, "\(label) asc rank \(r)")
                XCTAssertEqual(try a.kthElement(r, largest: true)?.asDouble, sortedDesc[r - 1].asDouble,
                               "\(label) desc rank \(r)")
            }
            XCTAssertNil(try a.kthElement(0), label)
            XCTAssertNil(try a.kthElement(m + 1), label)
        }
        for n in [1, 33, 4097, 200_003] {
            try check((0..<n).map { _ in Int64.random(in: .min ... .max, using: &rng) as Int64? }, "i64 n=\(n)")
            try check((0..<n).map { i in i % 7 == 0 ? nil : Int32.random(in: -1000...1000, using: &rng) }, "i32-nulls n=\(n)")
            try check((0..<n).map { _ in Double.random(in: -1e9...1e9, using: &rng) as Double? }, "f64 n=\(n)")
            try check((0..<n).map { _ in UInt8.random(in: .min ... .max, using: &rng) as UInt8? }, "u8 n=\(n)")
            try check([Int64?](repeating: 7, count: n), "all-equal n=\(n)")
        }
        try check([Int64?](), "empty")
        try check([Int64?](repeating: nil, count: 100), "all-null")
    }

    /// `quantile` now selects instead of sorting; it has to answer exactly what the sort answered.
    func testQuantileMatchesTheSortedAnswer() throws {
        try requireRealGPU()
        var rng = Rng(s: 0x9_1A5)
        for n in [4096, 4097, 100_003] {
            let d: [Double?] = (0..<n).map { i in i % 11 == 0 ? nil : Double.random(in: -1e9...1e9, using: &rng) }
            let i64: [Int64?] = (0..<n).map { i in i % 13 == 0 ? nil : Int64.random(in: -1_000_000...1_000_000, using: &rng) }
            let da = try MetalArray<Double>(d), ia = try MetalArray<Int64>(i64)
            for q in [0.0, 0.001, 0.25, 0.5, 0.75, 0.999, 1.0] {
                XCTAssertEqual(try da.quantile(q)!, sortedQuantile(d.compactMap { $0 }, q), accuracy: 1e-9,
                               "f64 n=\(n) q=\(q)")
                XCTAssertEqual(try ia.quantile(q)!, sortedQuantile(i64.compactMap { $0 }.map(Double.init), q), accuracy: 1e-9,
                               "i64 n=\(n) q=\(q)")
            }
            XCTAssertEqual(try da.approximateMedian()!, sortedQuantile(d.compactMap { $0 }, 0.5), accuracy: 1e-9)
        }
    }

    /// The definition `quantile` has always had: sort, then read the interpolated position.
    private func sortedQuantile(_ vals: [Double], _ q: Double) -> Double {
        let s = vals.sorted()
        let position = Swift.max(0, Swift.min(1, q)) * Double(s.count - 1)
        let lo = Int(position.rounded(.down)), hi = Int(position.rounded(.up))
        if lo == hi { return s[lo] }
        return s[lo] + (s[hi] - s[lo]) * (position - Double(lo))
    }

    // MARK: - Randomised stress against a CPU oracle
    //
    // `topKSelect` was once seen to disagree with pyarrow on one cell of a 13k-case differential run and
    // then passed on every rerun, so the interesting failures are the rare ones that depend on what the
    // GPU and the buffer pool did just before. This loop therefore varies shape, type, k, null ratio and
    // direction, and churns the pool with other kernels between calls. It compares against a CPU oracle
    // built from the same total order, not against `argsort`, so a shared bug cannot hide.
    //
    // Short by default (a few seconds). `ARROWMETAL_STRESS=1` runs 5000 iterations; any other number runs
    // that many.

    /// Order-preserving 64-bit key, and a NaN flag, matching `TopKSource.tk_map` / `SortSource`.
    private static func oracleKey<T: ArrowPrimitive>(_ v: T) -> (key: UInt64, nan: Bool) {
        switch v {
        case let x as Int8: return (UInt64(UInt8(bitPattern: x) ^ 0x80), false)
        case let x as UInt8: return (UInt64(x), false)
        case let x as Int16: return (UInt64(UInt16(bitPattern: x) ^ 0x8000), false)
        case let x as UInt16: return (UInt64(x), false)
        case let x as Int32: return (UInt64(UInt32(bitPattern: x) ^ 0x8000_0000), false)
        case let x as UInt32: return (UInt64(x), false)
        case let x as Int64: return (UInt64(bitPattern: x) ^ 0x8000_0000_0000_0000, false)
        case let x as UInt64: return (x, false)
        case let x as Float:
            var b = x.bitPattern
            let nan = (b & 0x7FFF_FFFF) > 0x7F80_0000
            if (b & 0x7FFF_FFFF) == 0 { b = 0 }
            if nan { b = 0x7F80_0001 }
            return (UInt64((b & 0x8000_0000) != 0 ? ~b : (b | 0x8000_0000)), nan)
        case let x as Double:
            var b = x.bitPattern
            let nan = (b & 0x7FFF_FFFF_FFFF_FFFF) > 0x7FF0_0000_0000_0000
            if (b & 0x7FFF_FFFF_FFFF_FFFF) == 0 { b = 0 }
            if nan { b = 0x7FF0_0000_0000_0000 | 1 }
            return ((b & 0x8000_0000_0000_0000) != 0 ? ~b : (b | 0x8000_0000_0000_0000), nan)
        default: fatalError("no oracle key for \(T.self)")
        }
    }

    /// The k winning row indices computed on the CPU: valid rows only, ordered by the same total order
    /// (key, then row index), with NaN last in both directions, then the null rows in row order.
    private func cpuTopK<T: ArrowPrimitive>(_ vals: [T?], _ k: Int, largest: Bool) -> [Int32] {
        var valid: [(key: UInt64, nan: Bool, row: Int32)] = []
        var nulls: [Int32] = []
        valid.reserveCapacity(vals.count)
        for (i, v) in vals.enumerated() {
            guard let v else { nulls.append(Int32(i)); continue }
            let (key, nan) = Self.oracleKey(v)
            valid.append((key, nan, Int32(i)))
        }
        valid.sort { a, b in
            if a.nan != b.nan { return !a.nan }               // NaN last, ascending or descending
            if a.key != b.key { return largest ? a.key > b.key : a.key < b.key }
            return a.row < b.row
        }
        var out = valid.prefix(k).map { $0.row }
        if out.count < k { out += nulls.prefix(k - out.count) }
        return out
    }

    private func checkAgainstCPU<T: ArrowPrimitive>(_ vals: [T?], _ k: Int, largest: Bool, _ label: String,
                                                    file: StaticString = #filePath, line: UInt = #line) throws {
        let a = try MetalArray<T>(vals)
        let want = cpuTopK(vals, k, largest: largest)
        func compare(_ got: [Int32], _ via: String) {
            guard got != want else { return }
            var where_ = "lengths \(got.count) vs \(want.count)"
            for i in 0..<Swift.min(got.count, want.count) where got[i] != want[i] {
                where_ = "first mismatch at \(i): got \(got[i]) want \(want[i])"
                break
            }
            XCTFail("\(label) [\(via)]: \(where_)\ngot  \(got.prefix(24))\nwant \(want.prefix(24))", file: file, line: line)
        }
        compare(try a.topK(k, largest: largest).toRawArray(), "topK")
        // Whatever the routing picked, the radix-select path has to agree wherever it applies.
        if let direct = try a.topKRadixSelect(k, largest: largest) { compare(direct.toRawArray(), "radix") }
    }

    /// Kernels run between two top-k calls so the buffer pool hands back recycled, dirty memory.
    private func churn(_ rng: inout Rng) throws {
        let n = Int.random(in: 1_000...40_000, using: &rng)
        let a = try MetalArray<Int32>((0..<n).map { _ in Int32.random(in: -1000...1000, using: &rng) })
        let mask = try a.compare(.gt, 0)
        let kept = try a.filter(mask)
        _ = try kept.argsort()
        let idx = try MetalArray<Int32>((0..<Swift.min(n, 5_000)).map { Int32($0) })
        _ = try a.take(idx).toRawArray().first
        _ = try a.cast(to: Int64.self).sum()
    }

    func testStressAgainstCPUOracle() throws {
        try requireRealGPU()
        let env = ProcessInfo.processInfo.environment["ARROWMETAL_STRESS"]
        var iterations = 40
        var seed: UInt64 = 0x0D15_EA5E          // fixed by default, so the short run is reproducible
        if let env {
            let asked = Int(env) ?? 0
            iterations = asked > 1 ? asked : 5000
            seed = DispatchTime.now().uptimeNanoseconds
        }
        var rng = Rng(s: seed)

        for it in 0..<iterations {
            // The shape from the one observed differential failure, run often; other shapes fill in around it.
            let failingShape = it % 4 == 0
            let n = failingShape ? 100_003 : [1, 2, 17, 255, 256, 257, 1023, 4096, 32_768, 32_769,
                                              100_003, 262_144, 1_000_003].randomElement(using: &rng)!
            // k spans both sides of the per-threadgroup kernel's 1024 limit so the radix path is exercised too.
            let k = failingShape ? 17 : Swift.max(1, Swift.min(n, [1, 2, 3, 17, 64, 255, 256, 257, 1000, 1024,
                                                                   1025, 4096, 10_000, 100_000]
                                                    .randomElement(using: &rng)!))
            guard k <= n else { continue }
            let nullRatio = failingShape ? 0.3 : [0.0, 0.05, 0.3, 0.5, 0.9].randomElement(using: &rng)!
            let largest = failingShape ? true : Bool.random(using: &rng)
            let kind = failingShape ? 5 : Int.random(in: 0..<6, using: &rng)
            func nulled<T>(_ make: (Int) -> T) -> [T?] {
                (0..<n).map { i in Double.random(in: 0..<1, using: &rng) < nullRatio ? nil : make(i) }
            }
            let label = "it=\(it) n=\(n) k=\(k) nulls=\(nullRatio) largest=\(largest) kind=\(kind)"
            switch kind {
            case 0: try checkAgainstCPU(nulled { _ in Int32.random(in: -1_000_000...1_000_000, using: &rng) }, k, largest: largest, label)
            case 1: try checkAgainstCPU(nulled { _ in UInt32.random(in: 0...9, using: &rng) }, k, largest: largest, label)
            case 2: try checkAgainstCPU(nulled { i in i % 997 == 0 ? Float.nan : Float.random(in: -1e6...1e6, using: &rng) }, k, largest: largest, label)
            case 3: try checkAgainstCPU(nulled { _ in Int64.random(in: Int64.min...Int64.max, using: &rng) }, k, largest: largest, label)
            case 4: try checkAgainstCPU(nulled { _ in UInt64.random(in: 0...50, using: &rng) }, k, largest: largest, label)
            default: try checkAgainstCPU(nulled { i in i % 2003 == 0 ? Double.nan : Double.random(in: -1e9...1e9, using: &rng) }, k, largest: largest, label)
            }
            if it % 3 == 0 { try churn(&rng) }
            if it % 97 == 96 { MetalContext.shared.pool.drain() }
        }
    }
}
