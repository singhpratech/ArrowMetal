import XCTest
@testable import ArrowMetal

/// `unique`, `valueCounts`, `dictionaryEncode` and the dense-key `groupBy` convenience against a plain
/// Swift oracle, over every primitive type, every interesting size, with and without nulls.
final class UniqueTests: XCTestCase {
    static let sizes = [0, 1, 31, 4096, 4097, 300_003]

    // MARK: - Oracle

    /// Arrow value equality for floats: every NaN is one value, -0 is +0. Identity elsewhere.
    func canonical<T: ArrowPrimitive>(_ v: T) -> T {
        if let f = v as? Float { return (f.isNaN ? Float.nan : (f == 0 ? Float(0) : f)) as! T }
        if let d = v as? Double { return (d.isNaN ? Double.nan : (d == 0 ? Double(0) : d)) as! T }
        return v
    }

    /// Ascending order with NaN last (the order `unique` promises).
    func less<T: ArrowPrimitive>(_ a: T, _ b: T) -> Bool {
        if let x = a as? Float, let y = b as? Float {
            if x.isNaN { return false }
            if y.isNaN { return true }
            return x < y
        }
        if let x = a as? Double, let y = b as? Double {
            if x.isNaN { return false }
            if y.isNaN { return true }
            return x < y
        }
        return a < b
    }

    /// Equality that holds for NaN, so expected and actual values can be compared directly.
    func same<T: ArrowPrimitive>(_ a: T?, _ b: T?) -> Bool {
        switch (a, b) {
        case (nil, nil): return true
        case (let x?, let y?):
            if let p = x as? Float, let q = y as? Float { return (p.isNaN && q.isNaN) || p == q }
            if let p = x as? Double, let q = y as? Double { return (p.isNaN && q.isNaN) || p == q }
            return x == y
        default: return false
        }
    }

    /// Distinct non-null values ascending, their row counts, and the dictionary code of every row.
    func oracle<T: ArrowPrimitive>(_ vals: [T?]) -> (values: [T], counts: [Int64], codes: [Int32?]) {
        var rows: [Int] = []
        for (i, v) in vals.enumerated() where v != nil { rows.append(i) }
        let canon = rows.map { canonical(vals[$0]!) }
        let order = (0..<canon.count).sorted { a, b in
            if less(canon[a], canon[b]) { return true }
            if less(canon[b], canon[a]) { return false }
            return a < b                                   // stable: earliest row wins
        }
        var values: [T] = [], counts: [Int64] = []
        var codes = [Int32?](repeating: nil, count: vals.count)
        for k in order {
            if values.isEmpty || !same(canonical(values[values.count - 1]), canon[k]) {
                values.append(vals[rows[k]]!)
                counts.append(0)
            }
            counts[counts.count - 1] += 1
            codes[rows[k]] = Int32(values.count - 1)
        }
        return (values, counts, codes)
    }

    // MARK: - Checks

    func check<T: ArrowPrimitive>(_ vals: [T?], label: String, file: StaticString = #filePath, line: UInt = #line) throws {
        let a = try MetalArray<T>(vals)
        let (expValues, expCounts, expCodes) = oracle(vals)

        let u = try a.unique().toArray()
        XCTAssertEqual(u.count, expValues.count, "\(label) unique count", file: file, line: line)
        if u.count == expValues.count {
            for i in 0..<u.count where !same(u[i], expValues[i]) {
                XCTFail("\(label) unique[\(i)] = \(String(describing: u[i])) expected \(expValues[i])", file: file, line: line)
                break
            }
        }

        let (vcValues, vcCounts) = try a.valueCounts()
        let vv = vcValues.toArray()
        XCTAssertEqual(vv.count, expValues.count, "\(label) valueCounts values", file: file, line: line)
        if vv.count == expValues.count {
            for i in 0..<vv.count where !same(vv[i], expValues[i]) {
                XCTFail("\(label) valueCounts value[\(i)]", file: file, line: line)
                break
            }
        }
        XCTAssertEqual(vcCounts.toRawArray(), expCounts, "\(label) counts", file: file, line: line)
        XCTAssertEqual(expCounts.reduce(0, +), Int64(vals.compactMap { $0 }.count), "\(label) counts sum", file: file, line: line)

        let (codes, dictValues) = try a.dictionaryEncode()
        XCTAssertEqual(codes.length, vals.count, "\(label) codes length", file: file, line: line)
        XCTAssertEqual(codes.toArray(), expCodes, "\(label) codes", file: file, line: line)
        let dv = dictValues.toArray()
        XCTAssertEqual(dv.count, expValues.count, "\(label) dict values", file: file, line: line)
        if dv.count == expValues.count {
            for i in 0..<dv.count where !same(dv[i], expValues[i]) {
                XCTFail("\(label) dict value[\(i)]", file: file, line: line)
                break
            }
        }
        // Codes really index the dictionary.
        for (i, c) in expCodes.enumerated() where c != nil {
            if !same(canonical(dv[Int(c!)]!), canonical(vals[i]!)) {
                XCTFail("\(label) code \(i) points at the wrong value", file: file, line: line)
                break
            }
        }

        // The GroupBy convenience must count exactly the rows per distinct value.
        let (gb, gbUnique) = try a.groupBy()
        XCTAssertEqual(gbUnique.length, expValues.count, "\(label) groupBy unique", file: file, line: line)
        if !expValues.isEmpty {
            XCTAssertEqual(try gb.count().toRawArray(), expCounts, "\(label) groupBy count", file: file, line: line)
        }
    }

    // MARK: - Tests

    func testAllTypesAndSizes() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        for n in Self.sizes {
            for nulls in [false, true] {
                let label = "n=\(n) nulls=\(nulls)"
                func maybe<V>(_ v: V) -> V? { nulls && Int.random(in: 0..<5, using: &g) == 0 ? nil : v }
                // Ranges are deliberately narrow relative to n so large sizes are heavy on duplicates.
                try check((0..<n).map { _ in maybe(Int8.random(in: -100...100, using: &g)) }, label: "Int8 " + label)
                try check((0..<n).map { _ in maybe(UInt16.random(in: 0...1000, using: &g)) }, label: "UInt16 " + label)
                try check((0..<n).map { _ in maybe(Int32.random(in: -1000...1000, using: &g)) }, label: "Int32 " + label)
                try check((0..<n).map { _ in maybe(Int64.random(in: -1_000_000...1_000_000, using: &g)) }, label: "Int64 " + label)
                try check((0..<n).map { _ in maybe(UInt64.random(in: 0...1_000_000, using: &g)) }, label: "UInt64 " + label)
                try check((0..<n).map { _ in maybe(Float(Int.random(in: -500...500, using: &g)) / 4) }, label: "Float " + label)
                try check((0..<n).map { _ in maybe(Double(Int.random(in: -500...500, using: &g)) / 8) }, label: "Double " + label)
            }
        }
    }

    /// Every distinct value appears exactly once, and every value appears many times.
    func testDuplicateDensity() throws {
        try requireRealGPU()
        try check((0..<4097).map { Int64($0) as Int64? }, label: "all distinct")
        try check([Int64?](repeating: 7, count: 300_003), label: "one value")
        try check((0..<300_003).map { Int32($0 % 1000) as Int32? }, label: "1000 of 300003")
        try check((0..<300_003).map { $0 % 3 == 0 ? nil : Int32($0 % 7) }, label: "7 keys with nulls")
        try check([Int32?](repeating: nil, count: 4097), label: "all null")
        try check([Int8?](repeating: nil, count: 1), label: "one null")
        // Full range of a narrow type.
        try check((0..<300_003).map { Int8(truncatingIfNeeded: $0) as Int8? }, label: "every Int8")
        try check((0..<300_003).map { UInt64($0 % 5) &* (UInt64.max / 5) as UInt64? }, label: "huge UInt64")
    }

    /// Arrow float semantics: all NaNs are one value sorted last, -0.0 equals 0.0.
    func testFloatNaNAndSignedZero() throws {
        try requireRealGPU()
        let f: [Float?] = [0.0, -0.0, .nan, -Float.nan, 1.0, .nan, -0.0, nil, .infinity, -.infinity, 0.0]
        let (fu, fc) = try MetalArray<Float>(f).valueCounts()
        let fv = fu.toArray()
        XCTAssertEqual(fv.count, 5)
        XCTAssertEqual(fv[0], -.infinity)
        XCTAssertEqual(fv[1]!, 0.0)                       // -0.0 == 0.0; the first row wins the slot
        XCTAssertEqual(fv[2], 1.0)
        XCTAssertEqual(fv[3], .infinity)
        XCTAssertTrue(fv[4]!.isNaN, "NaN sorts last")
        XCTAssertEqual(fc.toRawArray(), [1, 4, 1, 1, 3])
        // Codes: both zeros share a code, all three NaNs share a code, the null row stays null.
        let (fcodes, _) = try MetalArray<Float>(f).dictionaryEncode()
        let c = fcodes.toArray()
        XCTAssertEqual(c[0], c[1]); XCTAssertEqual(c[0], c[6]); XCTAssertEqual(c[0], c[10])
        XCTAssertEqual(c[2], c[3]); XCTAssertEqual(c[2], c[5]); XCTAssertEqual(c[2], 4)
        XCTAssertNil(c[7])

        let d: [Double?] = [.nan, -0.0, 0.0, -Double.nan, 5.5, nil, .nan, -1e308, 1e308]
        let (du, dc) = try MetalArray<Double>(d).valueCounts()
        let dv = du.toArray()
        XCTAssertEqual(dv.count, 5)
        XCTAssertEqual(dv[0], -1e308)
        XCTAssertEqual(dv[1]!, 0.0)
        XCTAssertEqual(dv[2], 5.5)
        XCTAssertEqual(dv[3], 1e308)
        XCTAssertTrue(dv[4]!.isNaN)
        XCTAssertEqual(dc.toRawArray(), [1, 2, 1, 1, 3])
        // A column of nothing but NaN collapses to one value.
        let allNaN = try MetalArray<Float>([Float](repeating: .nan, count: 4096))
        XCTAssertEqual(try allNaN.unique().length, 1)
        XCTAssertEqual(try allNaN.valueCounts().counts.toRawArray(), [4096])
    }

    /// The point of the API: aggregate over arbitrary (non-dense) keys of any numeric type.
    func testGroupByArbitraryKeys() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        let n = 200_000
        let raw: [Int64] = (0..<n).map { _ in Int64.random(in: 0..<400, using: &g) &* 1_000_003 &- 5_000_000 }
        let vals: [Int64] = (0..<n).map { _ in Int64.random(in: -1000...1000, using: &g) }
        let values = try MetalArray<Int64>(vals)

        func expect(_ keys: [Int64?]) -> (uniques: [Int64], sums: [Int64?]) {
            let (u, _, codes) = oracle(keys)
            var s = [Int64](repeating: 0, count: u.count), c = [Int](repeating: 0, count: u.count)
            for (i, code) in codes.enumerated() { if let code { s[Int(code)] &+= vals[i]; c[Int(code)] += 1 } }
            return (u, (0..<u.count).map { c[$0] > 0 ? s[$0] : nil })
        }

        // Int64 keys.
        let (eu, es) = expect(raw.map { $0 })
        let (gb, uniq) = try MetalArray<Int64>(raw).groupBy()
        XCTAssertEqual(uniq.toRawArray(), eu)
        XCTAssertEqual(try gb.sum(values).toArray(), es)
        XCTAssertEqual(try gb.count().toRawArray().reduce(0, +), Int64(n))

        // Int32 keys with nulls: null-keyed rows contribute nowhere.
        let k32: [Int32?] = (0..<n).map { $0 % 11 == 0 ? nil : Int32(truncatingIfNeeded: raw[$0] % 977) }
        let (u32, _, c32) = oracle(k32)
        let (gb32, uq32) = try MetalArray<Int32>(k32).groupBy()
        XCTAssertEqual(uq32.toRawArray(), u32)
        var s32 = [Int64](repeating: 0, count: u32.count), n32 = [Int](repeating: 0, count: u32.count)
        for (i, code) in c32.enumerated() { if let code { s32[Int(code)] &+= vals[i]; n32[Int(code)] += 1 } }
        XCTAssertEqual(try gb32.sum(values).toArray(), (0..<u32.count).map { n32[$0] > 0 ? s32[$0] : nil })

        // Float keys.
        let kf: [Float?] = (0..<n).map { Float(raw[$0] % 251) / 8 }
        let (uf, _, cf) = oracle(kf)
        let (gbf, uqf) = try MetalArray<Float>(kf).groupBy()
        XCTAssertEqual(uqf.toRawArray(), uf)
        var sf = [Int64](repeating: 0, count: uf.count)
        for (i, code) in cf.enumerated() { if let code { sf[Int(code)] &+= vals[i] } }
        XCTAssertEqual(try gbf.sum(values).toArray().map { $0 ?? 0 }, sf)
    }

    /// Sliced (offset) inputs go through the same path; slices share buffers with a non-zero offset.
    func testSlicedInput() throws {
        try requireRealGPU()
        let full = try MetalArray<Int32>((0..<8192).map { $0 % 13 == 0 ? nil : Int32($0 % 97) })
        let s = try full.slice(offset: 64, length: 5000)
        try check(s.toArray(), label: "sliced Int32")
        let f = try MetalArray<Float>((0..<8192).map { Float($0 % 31) - 5 }).slice(offset: 128, length: 4097)
        try check(f.toArray(), label: "sliced Float")
    }

    /// 50M Int64 rows over 1000 distinct values. Off by default (the sort dominates and takes seconds);
    /// run with ARROWMETAL_UNIQUE_BENCH=1.
    func testUniqueThroughput() throws {
        try requireRealGPU()
        try XCTSkipUnless(ProcessInfo.processInfo.environment["ARROWMETAL_UNIQUE_BENCH"] != nil,
                          "set ARROWMETAL_UNIQUE_BENCH=1 to run the 50M-row throughput measurement")
        let n = 50_000_000, distinct = 1000
        var g = SystemRandomNumberGenerator()
        let a = try MetalArray<Int64>((0..<n).map { _ in Int64.random(in: 0..<Int64(distinct), using: &g) &* 7 })
        for round in 0..<2 {
            let t0 = DispatchTime.now().uptimeNanoseconds
            let u = try a.unique()
            let seconds = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
            XCTAssertEqual(u.length, distinct)
            let gb = Double(n * 8) / 1e9
            print("unique 50M Int64 / \(distinct) distinct: round \(round) \(String(format: "%.3f", seconds)) s, "
                  + "\(String(format: "%.2f", gb / seconds)) GB/s, \(String(format: "%.1f", Double(n) / seconds / 1e6)) M rows/s")
        }
    }
}
