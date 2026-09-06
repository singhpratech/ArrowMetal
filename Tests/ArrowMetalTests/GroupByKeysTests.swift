import XCTest
@testable import ArrowMetal

/// Group-by over arbitrary key columns: the mapping stage in front of `GroupBy`.
///
/// Every test labels the groups with `groupKeys()` and compares against a per-key Swift dictionary
/// oracle, so the group *order* never enters the assertion — it is deliberately not pyarrow's.
final class GroupByKeysTests: XCTestCase {

    // MARK: - helpers

    /// (key description -> summed value) from a Swift oracle, nulls keyed as "null".
    func oracleSum(_ keys: [[String]], _ vals: [Int64?]) -> [String: Int64] {
        var out: [String: Int64] = [:]
        for (i, k) in keys.enumerated() {
            let label = k.joined(separator: "\u{1}")
            guard let v = vals[i] else { out[label] = out[label] ?? 0; continue }
            out[label] = (out[label] ?? 0) &+ v
        }
        return out
    }

    /// The (label -> value) map an ArrowMetal group-by produced, labels built the same way.
    func labels(_ gbk: GroupByKeys, describe: (AnyMetalArray, Int) -> String) throws -> [String] {
        let cols = try gbk.groupKeys()
        return (0..<gbk.groupCount).map { g in cols.map { describe($0, g) }.joined(separator: "\u{1}") }
    }

    func describeAny(_ a: AnyMetalArray, _ i: Int) -> String {
        switch a {
        case .int8(let x): return x[i].map { "\($0)" } ?? "null"
        case .uint8(let x): return x[i].map { "\($0)" } ?? "null"
        case .int16(let x): return x[i].map { "\($0)" } ?? "null"
        case .uint16(let x): return x[i].map { "\($0)" } ?? "null"
        case .int32(let x): return x[i].map { "\($0)" } ?? "null"
        case .uint32(let x): return x[i].map { "\($0)" } ?? "null"
        case .int64(let x): return x[i].map { "\($0)" } ?? "null"
        case .uint64(let x): return x[i].map { "\($0)" } ?? "null"
        case .float32(let x): return x[i].map { $0.isNaN ? "nan" : "\(Double($0))" } ?? "null"
        case .float64(let x): return x[i].map { $0.isNaN ? "nan" : "\($0)" } ?? "null"
        case .boolean(let x): return x[i].map { "\($0)" } ?? "null"
        case .string(let x), .binary(let x): return x[i] ?? "null"
        case .temporal(let t):
            switch t.storage {
            case .int32(let x): return x[i].map { "\($0)" } ?? "null"
            case .int64(let x): return x[i].map { "\($0)" } ?? "null"
            }
        case .decimal(let d): return d[i].map { "\($0)" } ?? "null"
        case .dictionary(let codes, _): return codes[i].map { "\($0)" } ?? "null"
        case .list, .structure, .map, .union, .runEndEncoded: return "?"
        }
    }

    // MARK: - single-column keys, every supported type

    func testSparseAndNegativeIntegerKeys() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        for n in [0, 1, 33, 4097, 200_003] {
            for span in [1, 7, 1024, 100_000] {
                var keys: [Int64?] = [], vals: [Int64?] = []
                for _ in 0..<n {
                    keys.append(Int.random(in: 0..<8, using: &g) == 0 ? nil
                                : Int64.random(in: -Int64(span)...Int64(span), using: &g) &* 1_000_003)
                    vals.append(Int.random(in: 0..<10, using: &g) == 0 ? nil : Int64.random(in: -1000...1000, using: &g))
                }
                let column = AnyMetalArray.int64(try MetalArray<Int64>(keys))
                let gbk = try GroupByKeys(columns: [column])
                let values = try MetalArray<Int64>(vals)
                let sums = try gbk.groupBy.sum(values)
                let names = try labels(gbk, describe: describeAny)
                let expected = oracleSum(keys.map { [$0.map { "\($0)" } ?? "null"] }, vals)
                var valid: [String: Int] = [:]
                for i in 0..<n where vals[i] != nil { valid[keys[i].map { "\($0)" } ?? "null", default: 0] += 1 }
                XCTAssertEqual(gbk.groupCount, expected.count, "group count n=\(n) span=\(span)")
                for (g, name) in names.enumerated() {
                    guard let e = expected[name] else { XCTFail("unexpected group \(name)"); continue }
                    if (valid[name] ?? 0) == 0 { XCTAssertNil(sums[g], "empty group \(name)") }
                    else { XCTAssertEqual(sums[g], e, "sum for \(name) n=\(n) span=\(span)") }
                }
            }
        }
    }

    func testFloatKeysNormaliseZeroAndNaN() throws {
        try requireRealGPU()
        let keys: [Double?] = [0.0, -0.0, .nan, .nan, 1.5, nil, 1.5, -0.0, .infinity, -.infinity]
        let vals = try MetalArray<Int64>([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
        let gbk = try GroupByKeys(columns: [.float64(try MetalArray<Double>(keys))])
        let sums = try gbk.groupBy.sum(vals)
        var byLabel: [String: Int64] = [:]
        let names = try labels(gbk, describe: describeAny)
        for (g, name) in names.enumerated() { byLabel[name] = sums[g] }
        // -0.0 and 0.0 are one group (1 + 2 + 128), both NaNs are one group (4 + 8).
        XCTAssertEqual(byLabel["0.0"] ?? byLabel["-0.0"], 131)
        XCTAssertEqual(byLabel["nan"], 12)
        XCTAssertEqual(byLabel["1.5"], 80)
        XCTAssertEqual(byLabel["null"], 32)
        XCTAssertEqual(byLabel["inf"], 256)
        XCTAssertEqual(byLabel["-inf"], 512)
        XCTAssertEqual(gbk.groupCount, 6)
    }

    func testBooleanStringAndTemporalKeys() throws {
        try requireRealGPU()
        let vals = try MetalArray<Int64>([1, 2, 3, 4, 5, 6])
        let bools = try MetalBooleanArray.fromUInt8Array(try MetalArray<UInt8>([1, 0, 1, nil, 0, 1]))
        let bg = try GroupByKeys(columns: [.boolean(bools)])
        XCTAssertEqual(bg.groupCount, 3)
        var byLabel: [String: Int64?] = [:]
        for (g, name) in try labels(bg, describe: describeAny).enumerated() { byLabel[name] = try bg.groupBy.sum(vals)[g] }
        XCTAssertEqual(byLabel["true"], 10)
        XCTAssertEqual(byLabel["false"], 7)
        XCTAssertEqual(byLabel["null"], 4)

        let strings = try MetalStringArray(["a", "bb", "a", nil, "bb", "ccc"])
        let sg = try GroupByKeys(columns: [.string(strings)])
        XCTAssertEqual(sg.groupCount, 4)
        var byString: [String: Int64?] = [:]
        for (g, name) in try labels(sg, describe: describeAny).enumerated() { byString[name] = try sg.groupBy.sum(vals)[g] }
        XCTAssertEqual(byString["a"], 4)
        XCTAssertEqual(byString["bb"], 7)
        XCTAssertEqual(byString["ccc"], 6)
        XCTAssertEqual(byString["null"], 4)

        let days = try MetalTemporalArray(type: ArrowTemporalType.date32, [10, 20, 10, nil, 20, 30])
        let tg = try GroupByKeys(columns: [.temporal(days)])
        XCTAssertEqual(tg.groupCount, 4)
    }

    func testDictionaryAndDecimalKeys() throws {
        try requireRealGPU()
        let vals = try MetalArray<Int64>([1, 2, 3, 4, 5])
        let codes = try MetalArray<Int32>([2, 0, 2, nil, 0])
        let dict = AnyMetalArray.dictionary(codes: codes, values: .string(try MetalStringArray(["x", "y", "z"])))
        let dg = try GroupByKeys(columns: [dict])
        XCTAssertEqual(dg.groupCount, 3)          // codes 0 and 2 are used, plus the null group
        let sums = try dg.groupBy.sum(vals).toArray().compactMap { $0 }.sorted()
        XCTAssertEqual(sums, [4, 4, 7])

        let dec = try MetalDecimalArray(type: try ArrowDecimalType(precision: 20, scale: 2),
                                        [ArrowDecimal128(lo: 100, hi: 0), ArrowDecimal128(lo: 200, hi: 0),
                                         ArrowDecimal128(lo: 100, hi: 0), nil, ArrowDecimal128(lo: 0, hi: 1)])
        let cg = try GroupByKeys(columns: [.decimal(dec)])
        XCTAssertEqual(cg.groupCount, 4)
        XCTAssertEqual(try cg.groupBy.sum(vals).toArray().compactMap { $0 }.sorted(), [4, 4, 5, 2].sorted())
    }

    // MARK: - several key columns

    func testTwoAndThreeColumnKeys() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        for n in [0, 1, 33, 4097, 100_003] {
            var a: [Int32?] = [], b: [String?] = [], c: [Float?] = [], vals: [Int64?] = []
            for _ in 0..<n {
                a.append(Int.random(in: 0..<9, using: &g) == 0 ? nil : Int32.random(in: -3...3, using: &g))
                b.append(Int.random(in: 0..<9, using: &g) == 0 ? nil : ["r1", "r2", "r3"].randomElement(using: &g))
                c.append(Int.random(in: 0..<9, using: &g) == 0 ? nil : Float([1.5, -2.5][Int.random(in: 0..<2, using: &g)]))
                vals.append(Int.random(in: 0..<7, using: &g) == 0 ? nil : Int64.random(in: -500...500, using: &g))
            }
            let values = try MetalArray<Int64>(vals)
            let colA = AnyMetalArray.int32(try MetalArray<Int32>(a))
            let colB = AnyMetalArray.string(try MetalStringArray(b))
            let colC = AnyMetalArray.float32(try MetalArray<Float>(c))

            for columns in [[colA, colB], [colA, colB, colC]] {
                let gbk = try GroupByKeys(columns: columns)
                let sums = try gbk.groupBy.sum(values)
                let counts = try gbk.groupBy.count()
                let names = try labels(gbk, describe: describeAny)
                var expectedSum: [String: Int64] = [:], expectedRows: [String: Int] = [:], expectedValid: [String: Int] = [:]
                for i in 0..<n {
                    var parts = [a[i].map { "\($0)" } ?? "null", b[i] ?? "null"]
                    if columns.count == 3 { parts.append(c[i].map { "\(Double($0))" } ?? "null") }
                    let label = parts.joined(separator: "\u{1}")
                    expectedRows[label, default: 0] += 1
                    if let v = vals[i] { expectedSum[label] = (expectedSum[label] ?? 0) &+ v; expectedValid[label, default: 0] += 1 }
                }
                XCTAssertEqual(gbk.groupCount, expectedRows.count, "n=\(n) cols=\(columns.count)")
                for (gi, name) in names.enumerated() {
                    XCTAssertEqual(Int(counts.valuePointer[gi]), expectedRows[name] ?? -1, "rows \(name)")
                    if (expectedValid[name] ?? 0) == 0 { XCTAssertNil(sums[gi]) }
                    else { XCTAssertEqual(sums[gi], expectedSum[name], "sum \(name)") }
                }
            }
        }
    }

    func testDenseKeyFastPathStillWorks() throws {
        try requireRealGPU()
        let keys = try MetalArray<Int32>([0, 1, 0, 2, 1, nil, 0])
        let vals = try MetalArray<Int64>([1, 2, 3, 4, 5, 6, 7])
        let gb = try GroupBy(keys: [.int32(keys)], denseKeyCount: 4)
        XCTAssertEqual(gb.keyCount, 4)
        XCTAssertEqual(try gb.sum(vals).toArray(), [11, 7, 4, nil])
        // Without denseKeyCount the same column is mapped: three observed keys plus the null group.
        let mapped = try GroupBy(keys: [.int32(keys)])
        XCTAssertEqual(mapped.keyCount, 4)
        XCTAssertEqual(try mapped.sum(vals).toArray().compactMap { $0 }.sorted(), [4, 6, 7, 11])
    }

    func testEmptyAndAllNullKeys() throws {
        try requireRealGPU()
        let empty = try GroupByKeys(columns: [.int64(try MetalArray<Int64>([Int64?]()))])
        XCTAssertEqual(empty.groupCount, 0)
        XCTAssertEqual(try empty.groupKeys()[0].length, 0)

        let allNull = try GroupByKeys(columns: [.int64(try MetalArray<Int64>([nil, nil, nil]))])
        XCTAssertEqual(allNull.groupCount, 1)
        XCTAssertEqual(try allNull.groupBy.sum(try MetalArray<Int64>([1, 2, 3])).toArray(), [6])
    }

    func testMismatchedLengthsAndUnsupportedKeys() throws {
        try requireRealGPU()
        XCTAssertThrowsError(try GroupByKeys(columns: []))
        XCTAssertThrowsError(try GroupByKeys(columns: [.int64(try MetalArray<Int64>([1, 2])),
                                                       .int64(try MetalArray<Int64>([1, 2, 3]))]))
        let list = try MetalListArray(counts: [1, 1], values: .int64(try MetalArray<Int64>([1, 2])))
        XCTAssertThrowsError(try GroupByKeys(columns: [.list(list)]))
    }

    func testLargeMultiColumnMatchesOracle() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        let n = 1_000_003
        let a: [Int32] = (0..<n).map { _ in Int32.random(in: 0..<1000, using: &g) }
        let b: [Int32] = (0..<n).map { _ in Int32.random(in: 0..<100, using: &g) }
        let vals: [Int64] = (0..<n).map { _ in Int64.random(in: -100...100, using: &g) }
        let gbk = try GroupByKeys(columns: [.int32(try MetalArray<Int32>(a)), .int32(try MetalArray<Int32>(b))])
        let sums = try gbk.groupBy.sum(try MetalArray<Int64>(vals))
        var expected: [Int64: Int64] = [:]
        for i in 0..<n { expected[Int64(a[i]) * 1000 + Int64(b[i]), default: 0] &+= vals[i] }
        XCTAssertEqual(gbk.groupCount, expected.count)
        let cols = try gbk.groupKeys()
        guard case .int32(let ka) = cols[0], case .int32(let kb) = cols[1] else { return XCTFail("key types") }
        for gi in 0..<gbk.groupCount {
            let key = Int64(ka.valuePointer[gi]) * 1000 + Int64(kb.valuePointer[gi])
            XCTAssertEqual(sums[gi], expected[key], "group \(gi)")
        }
    }

    /// Ten million rows over five million distinct keys — the shape the benchmark runs, and the one
    /// where the composite key and the device-atomic path both matter.
    ///
    /// The keys are `id * 7 - 3` for a random `id`, so the oracle is a flat array indexed by `id`
    /// rather than a five-million-entry dictionary: the same check, but it also finishes in a debug build.
    func testTenMillionRowsFiveMillionKeys() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        let n = 10_000_003, K = 5_000_000
        var ids = [Int32](repeating: 0, count: n)
        var keys = [Int64?](repeating: nil, count: n)
        var vals = [Int64](repeating: 0, count: n)
        for i in 0..<n {
            let id = Int32.random(in: 0..<Int32(K), using: &g)
            ids[i] = i % 97 == 0 ? -1 : id
            keys[i] = i % 97 == 0 ? nil : Int64(id) &* 7 &- 3
            vals[i] = Int64.random(in: -100...100, using: &g)
        }
        let gbk = try GroupByKeys(columns: [.int64(try MetalArray<Int64>(keys))])
        let sums = try gbk.groupBy.sum(try MetalArray<Int64>(vals))
        var expected = [Int64](repeating: 0, count: K), seen = [Bool](repeating: false, count: K)
        var nullSum: Int64 = 0, distinct = 0
        for i in 0..<n {
            let id = Int(ids[i])
            if id < 0 { nullSum &+= vals[i]; continue }
            if !seen[id] { seen[id] = true; distinct += 1 }
            expected[id] &+= vals[i]
        }
        XCTAssertEqual(gbk.groupCount, distinct + 1)
        guard case .int64(let labels) = try gbk.groupKeys()[0] else { return XCTFail("key type") }
        for gi in 0..<gbk.groupCount {
            guard let k = labels[gi] else { XCTAssertEqual(sums[gi], nullSum, "null group"); continue }
            XCTAssertEqual(sums[gi], expected[Int((k + 3) / 7)], "group \(gi)")
        }
    }
}
