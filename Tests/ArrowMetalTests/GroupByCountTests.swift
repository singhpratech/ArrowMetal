import XCTest
@testable import ArrowMetal

/// `count(expr)` is the number of non-null values of `expr` in each group, for every Arrow type, on
/// every path that computes it: the plan executor's fused group-by, its per-aggregate group-by (taken
/// when another aggregate needs it, e.g. a Float64 sum), the whole-input aggregate, the dense
/// `GroupBy` API, the streaming group-by and the fused stream join aggregate. Every answer is checked
/// against a CPU count of the validity the column was built from, at null fractions 0, 0.1 and 1.0.
final class GroupByCountTests: XCTestCase {

    enum Kind: String, CaseIterable {
        case int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64, boolean
        case utf8, utf8View, binary, date32, timestamp, decimal128, list, structure, dictionary
        case dictionaryNullValues, fixedBinary, float16, null, runEndEncoded, union, extensionInt64
    }

    static let fractions = [0.0, 0.1, 1.0]

    /// Deterministic validity: `frac` of the rows null (all valid at 0, all null at 1).
    func validity(_ n: Int, _ frac: Double, seed: UInt64 = 1) -> [Bool] {
        var s = seed &* 0x9E3779B97F4A7C15 &+ 17
        return (0..<n).map { _ in
            s = s &* 6364136223846793005 &+ 1442695040888963407
            return Double(s >> 11) / Double(1 << 53) >= frac
        }
    }

    /// A column of `kind` whose logical validity is `valid` (a `null` column is all null whatever
    /// `valid` says, so the caller takes the returned validity as the truth).
    func column(_ kind: Kind, _ valid: [Bool]) throws -> (AnyMetalArray, [Bool]) {
        let n = valid.count
        func opt<T>(_ f: (Int) -> T) -> [T?] { (0..<n).map { valid[$0] ? f($0) : nil } }
        switch kind {
        case .int8: return (.int8(try MetalArray<Int8>(opt { Int8(truncatingIfNeeded: $0) })), valid)
        case .int16: return (.int16(try MetalArray<Int16>(opt { Int16(truncatingIfNeeded: $0) })), valid)
        case .int32: return (.int32(try MetalArray<Int32>(opt { Int32($0) - 7 })), valid)
        case .int64: return (.int64(try MetalArray<Int64>(opt { Int64($0) * 3 })), valid)
        case .uint8: return (.uint8(try MetalArray<UInt8>(opt { UInt8(truncatingIfNeeded: $0) })), valid)
        case .uint16: return (.uint16(try MetalArray<UInt16>(opt { UInt16(truncatingIfNeeded: $0) })), valid)
        case .uint32: return (.uint32(try MetalArray<UInt32>(opt { UInt32($0) })), valid)
        case .uint64: return (.uint64(try MetalArray<UInt64>(opt { UInt64($0) })), valid)
        // NaN is a value, not a null: count it.
        case .float32: return (.float32(try MetalArray<Float>(opt { $0 % 5 == 0 ? .nan : Float($0) / 4 })), valid)
        case .float64: return (.float64(try MetalArray<Double>(opt { $0 % 5 == 0 ? .nan : Double($0) / 8 })), valid)
        case .boolean:
            let b = try MetalBooleanArray.allocate(length: n, withValidity: true)
            let vp = b.values.mutableTyped(UInt8.self), bp = b.validity!.mutableTyped(UInt8.self)
            for i in 0..<n {
                if i % 3 == 0 { Bitmap.set(vp, i) }
                if valid[i] { Bitmap.set(bp, i) }
            }
            b.recomputeNullCount()
            return (.boolean(b), valid)
        case .utf8: return (.string(try MetalStringArray(opt { "s\($0 % 17)" })), valid)
        case .utf8View:
            return (.string(try MetalStringArray.viewLayout(opt { $0 % 2 == 0 ? "short\($0)" : "a longer string \($0)" },
                                                            bufferBytes: 4096)), valid)
        case .binary: return (.binary(markBinary(try MetalStringArray(opt { "b\($0 % 3)" }))), valid)
        case .date32: return (.temporal(try MetalTemporalArray(type: .date32, opt { Int64($0 % 900) })), valid)
        case .timestamp:
            return (.temporal(try MetalTemporalArray(type: .timestamp(.micro, timezone: "UTC"), opt { Int64($0) * 1_000_003 })), valid)
        case .decimal128:
            return (.decimal(try MetalDecimalArray(type: try ArrowDecimalType(precision: 12, scale: 2),
                                                   unscaled: opt { Int64($0) * 101 - 5000 })), valid)
        case .list:
            let child = AnyMetalArray.int64(try MetalArray<Int64>((0..<(2 * n)).map { Int64($0) }))
            return (.list(try MetalListArray(counts: (0..<n).map { valid[$0] ? $0 % 3 : nil }, values: child)), valid)
        case .structure:
            return (.structure(try MetalStructArray(names: ["a"], children: [.int32(try MetalArray<Int32>((0..<n).map { Int32($0) }))],
                                                    valid: valid)), valid)
        case .dictionary:
            let codes = try MetalArray<Int32>(opt { Int32($0 % 3) })
            return (.dictionary(codes: codes, values: .string(try MetalStringArray(["a", "b", "c"]))), valid)
        case .dictionaryNullValues:
            // Every code is valid; a null row points at the dictionary's null entry.
            let codes = try MetalArray<Int32>((0..<n).map { valid[$0] ? ($0 % 2 == 0 ? 0 : 2) : 1 } as [Int32])
            return (.dictionary(codes: codes, values: .string(try MetalStringArray(["x", nil, "y"]))), valid)
        case .fixedBinary:
            return (.fixedBinary(try MetalFixedBinaryArray(byteWidth: 4, opt { [UInt8($0 & 255), 1, 2, 3] })), valid)
        case .float16: return (.float16(try MetalFloat16Array(opt { Float($0 % 100) / 2 })), valid)
        case .null: return (.null(MetalNullArray(length: n)), [Bool](repeating: false, count: n))
        case .runEndEncoded:
            // Runs of up to three rows that share a validity.
            var ends: [Int32] = [], vals: [Int64?] = []
            var i = 0
            while i < n {
                var j = i + 1
                while j < n, j - i < 3, valid[j] == valid[i] { j += 1 }
                ends.append(Int32(j)); vals.append(valid[i] ? Int64(i) : nil)
                i = j
            }
            return (.runEndEncoded(runEnds: try MetalArray<Int32>(ends), values: .int64(try MetalArray<Int64>(vals))), valid)
        case .union:
            // Sparse: row i selects child i % 2; the child it does not select holds the opposite
            // validity, so reading the wrong child gives the wrong count.
            let ids = try MetalArray<Int8>((0..<n).map { Int8($0 % 2) })
            let c0 = try MetalArray<Int32>((0..<n).map { ($0 % 2 == 0) == valid[$0] ? Int32($0) : nil })
            let c1 = try MetalStringArray((0..<n).map { ($0 % 2 == 1) == valid[$0] ? "u\($0)" : nil })
            return (.union(try MetalUnionArray(mode: .sparse, length: n, typeIds: ids, offsets: nil, typeCodes: [0, 1],
                                               names: ["i", "s"], children: [.int32(c0), .string(c1)])), valid)
        case .extensionInt64:
            return (AnyMetalArray.int64(try MetalArray<Int64>(opt { Int64($0) })).asExtensionType(name: "test.ext"), valid)
        }
    }

    /// Group keys: 13 groups plus a null-key group every 97th row.
    func keys(_ n: Int) -> [Int32?] {
        (0..<n).map { (i: Int) -> Int32? in
            if i % 97 == 50 { return nil }
            return Int32((i &* 7919) % 13)
        }
    }

    func reference(_ keys: [Int32?], _ valid: [Bool]) -> [Int32?: Int64] {
        var out: [Int32?: Int64] = [:]
        for (k, v) in zip(keys, valid) { out[k, default: 0] += v ? 1 : 0 }
        return out
    }

    func i64(_ b: MetalRecordBatch, _ name: String) throws -> [Int64?] {
        guard case .int64(let a)? = b[name] else { XCTFail("\(name) is not int64"); return [] }
        return a.toArray()
    }

    func i32(_ b: MetalRecordBatch, _ name: String) throws -> [Int32?] {
        guard case .int32(let a)? = b[name] else { XCTFail("\(name) is not int32"); return [] }
        return a.toArray()
    }

    /// The batch every plan test runs on: key `k`, the column under test `v`, a Float64 `f` and an
    /// int32 `i`, both with nulls.
    func batch(_ v: AnyMetalArray, _ n: Int) throws -> MetalRecordBatch {
        let fv = validity(n, 0.2, seed: 99), iv = validity(n, 0.3, seed: 5)
        return try MetalRecordBatch(
            names: ["k", "v", "f", "i"],
            columns: [.int32(try MetalArray<Int32>(keys(n))), v,
                      .float64(try MetalArray<Double>((0..<n).map { fv[$0] ? Double($0) / 3 : nil })),
                      .int32(try MetalArray<Int32>((0..<n).map { iv[$0] ? Int32($0 % 50) : nil }))])
    }

    /// The aggregate lists `count(v)` is checked in: alone, next to each Float64 aggregate (the
    /// per-aggregate path), and next to an int sum and a row count (the fused path).
    static let contexts: [(String, [ExprAggregate])] = [
        ("alone", []),
        ("f64 sum", [ExprAggregate(.sum, col("f"), name: "x")]),
        ("f64 mean", [ExprAggregate(.mean, col("f"), name: "x")]),
        ("f64 min", [ExprAggregate(.min, col("f"), name: "x")]),
        ("f64 max", [ExprAggregate(.max, col("f"), name: "x")]),
        ("int sum", [ExprAggregate(.sum, col("i"), name: "x"), ExprAggregate(.count, nil, name: "rows")]),
    ]

    // MARK: - plan executor

    func testPlanGroupByCountEveryTypeEveryContext() throws {
        try requireRealGPU()
        let n = 3001
        let ks = keys(n)
        var rowsPerKey: [Int32?: Int64] = [:]
        for k in ks { rowsPerKey[k, default: 0] += 1 }
        for kind in Kind.allCases {
            for frac in Self.fractions {
                let (v, valid) = try column(kind, validity(n, frac, seed: UInt64(frac * 10) + 3))
                let want = reference(ks, valid)
                let b = try batch(v, n)
                for (label, others) in Self.contexts {
                    let aggs = others + [ExprAggregate(.count, col("v"), name: "c")]
                    let out = try LazyFrame(PlanSource(name: "t", batch: b)).groupBy(["k"], aggs).collect()
                    let gotKeys = try i32(out, "k"), got = try i64(out, "c")
                    XCTAssertEqual(out.length, want.count, "\(kind) \(frac) \(label)")
                    for (j, k) in gotKeys.enumerated() {
                        XCTAssertEqual(got[j], want[k], "\(kind) frac=\(frac) \(label) key=\(String(describing: k))")
                    }
                    if label == "int sum" {
                        let rows = try i64(out, "rows")
                        for (j, k) in gotKeys.enumerated() { XCTAssertEqual(rows[j], rowsPerKey[k], "\(kind) rows") }
                    }
                }
            }
        }
    }

    /// The two differential-grid cases, minimal: `count(v)` over Float64 next to each Float64
    /// aggregate, and `count(s)` over utf8 alone and next to a Float64 sum (SQL's answer is 2).
    func testMinimalReproductions() throws {
        try requireRealGPU()
        let f = try MetalRecordBatch(names: ["k", "v"], columns: [.int32(try MetalArray<Int32>([0, 1, 0, 1])),
                                                                  .float64(try MetalArray<Double>([1, 2, 3, 4]))])
        for op in [ExprAggregate.Op.sum, .min, .max, .mean] {
            let out = try LazyFrame(PlanSource(name: "t", batch: f))
                .groupBy(["k"], [ExprAggregate(op, col("v"), name: "a"), ExprAggregate(.count, col("v"), name: "n")])
                .collect()
            XCTAssertEqual(try i64(out, "n"), [2, 2], "\(op)")
        }
        let s = try MetalRecordBatch(names: ["k", "s", "v"],
                                     columns: [.int64(try MetalArray<Int64>([1, 1, 1, 1])),
                                               .string(try MetalStringArray(["a", nil, "b", nil])),
                                               .float64(try MetalArray<Double>([1, 2, 3, 4]))])
        let alone = try LazyFrame(PlanSource(name: "t", batch: s))
            .groupBy(["k"], [ExprAggregate(.count, col("s"), name: "n")]).collect()
        XCTAssertEqual(try i64(alone, "n"), [2])
        let withSum = try LazyFrame(PlanSource(name: "t", batch: s))
            .groupBy(["k"], [ExprAggregate(.count, col("s"), name: "n"), ExprAggregate(.sum, col("v"), name: "t")]).collect()
        XCTAssertEqual(try i64(withSum, "n"), [2])
        guard case .float64(let t)? = withSum["t"] else { return XCTFail("t is not float64") }
        XCTAssertEqual(t.toArray(), [10])
    }

    func testCountOfAComputedExpression() throws {
        try requireRealGPU()
        let n = 5003
        let b = try batch(.int32(try MetalArray<Int32>((0..<n).map { $0 % 4 == 0 ? nil : Int32($0) })), n)
        guard case .int32(let v)? = b["v"], case .int32(let iv)? = b["i"] else { return XCTFail() }
        let va = v.toArray(), ia = iv.toArray(), ks = keys(n)
        // (v + i) is null where either is; (if_else (v > 100) v null) is null where v is or v <= 100.
        let sumValid = (0..<n).map { va[$0] != nil && ia[$0] != nil }
        let gtValid = (0..<n).map { (va[$0] ?? 0) > 100 }
        let exprs: [(Expr, [Bool])] = [
            (col("v") + col("i"), sumValid),
            (.ifElse(col("v") > 100, col("v"), nullLit(.int32)), gtValid),
            (col("v").isValidExpr, [Bool](repeating: true, count: n)),
        ]
        for (e, valid) in exprs {
            let want = reference(ks, valid)
            for (label, others) in Self.contexts {
                let out = try LazyFrame(PlanSource(name: "t", batch: b))
                    .groupBy(["k"], others + [ExprAggregate(.count, e, name: "c")]).collect()
                let gotKeys = try i32(out, "k"), got = try i64(out, "c")
                for (j, k) in gotKeys.enumerated() { XCTAssertEqual(got[j], want[k], "\(e) \(label)") }
            }
        }
    }

    func testEmptyInputAndAllNullGroups() throws {
        try requireRealGPU()
        for kind in [Kind.utf8, .float64, .boolean, .decimal128, .null] {
            let (v, _) = try column(kind, [])
            let b = try batch(v, 0)
            for (label, others) in Self.contexts {
                let out = try LazyFrame(PlanSource(name: "t", batch: b))
                    .groupBy(["k"], others + [ExprAggregate(.count, col("v"), name: "c")]).collect()
                XCTAssertEqual(out.length, 0, "\(kind) \(label)")
                XCTAssertNotNil(out["c"], "\(kind) \(label)")
            }
            let whole = try LazyFrame(PlanSource(name: "t", batch: b)).aggregate([ExprAggregate(.count, col("v"), name: "c")]).collect()
            XCTAssertEqual(try i64(whole, "c"), [0], "\(kind) whole input")
        }
    }

    func testWholeInputCountEveryType() throws {
        try requireRealGPU()
        let n = 4099
        for kind in Kind.allCases {
            for frac in Self.fractions {
                let (v, valid) = try column(kind, validity(n, frac, seed: 11))
                let b = try batch(v, n)
                let want = Int64(valid.filter { $0 }.count)
                let out = try LazyFrame(PlanSource(name: "t", batch: b))
                    .aggregate([ExprAggregate(.count, col("v"), name: "c"), ExprAggregate(.sum, col("f"), name: "s")]).collect()
                XCTAssertEqual(try i64(out, "c"), [want], "\(kind) \(frac)")
                // With a filter the planner keeps the filter out of the reduce kernel when it counts.
                let ks = keys(n)
                let wantF = Int64((0..<n).filter { valid[$0] && ks[$0] == 3 }.count)
                let outF = try LazyFrame(PlanSource(name: "t", batch: b)).filter(col("k") == 3)
                    .aggregate([ExprAggregate(.count, col("v"), name: "c"), ExprAggregate(.count, nil, name: "r")]).collect()
                XCTAssertEqual(try i64(outF, "c"), [wantF], "\(kind) \(frac) filtered")
            }
        }
    }

    // MARK: - dense GroupBy API (what the C ABI's hash_count calls)

    func testDenseGroupByCountValidEveryType() throws {
        try requireRealGPU()
        for (n, K) in [(3001, 13), (40_000, 5000)] {
            let ks = (0..<n).map { Int32(($0 &* 7919) % K) }
            let gb = try MetalArray<Int32>(ks).groupBy(keyCount: K)
            for kind in Kind.allCases {
                for frac in Self.fractions {
                    let (v, valid) = try column(kind, validity(n, frac, seed: 21))
                    var want = [Int64](repeating: 0, count: K)
                    for i in 0..<n where valid[i] { want[Int(ks[i])] += 1 }
                    XCTAssertEqual(try gb.countValid(v).toRawArray(), want, "\(kind) \(frac) K=\(K)")
                }
            }
            // The typed API takes Float64 now too.
            let (d, dv) = try column(.float64, validity(n, 0.1, seed: 4))
            guard case .float64(let da) = d else { return XCTFail() }
            var want = [Int64](repeating: 0, count: K)
            for i in 0..<n where dv[i] { want[Int(ks[i])] += 1 }
            XCTAssertEqual(try gb.count(da).toRawArray(), want)
            XCTAssertEqual(try gb.countValid(da).toRawArray(), want)
        }
    }

    // MARK: - streaming

    func testStreamingGroupByCount() throws {
        try requireRealGPU()
        let n = 6000
        for kind in [Kind.int32, .float64, .boolean, .utf8, .decimal128, .dictionaryNullValues, .null, .list, .date32] {
            for frac in Self.fractions {
                let (v, valid) = try column(kind, validity(n, frac, seed: 8))
                let ks = (0..<n).map { Int32(($0 &* 31) % 9) }
                let b = try MetalRecordBatch(names: ["k", "v"], columns: [.int32(try MetalArray<Int32>(ks)), v])
                let parts = try stride(from: 0, to: n, by: 1700).map { try b.slice(offset: $0, length: Swift.min(1700, n - $0)) }
                var want = [Int32: Int64]()
                for i in 0..<n { want[ks[i], default: 0] += valid[i] ? 1 : 0 }
                for dense in [nil, 9] as [Int?] {
                    let r = try StreamQuery(source: ChunkedTableSource(parts)).groupBy(
                        ["k"], [StreamAggregate(.count, "v", name: "c"), StreamAggregate(.count, nil, name: "n")],
                        denseKeyCount: dense)
                    let out = try XCTUnwrap(r.batch)
                    let gotKeys = try i32(out, "k"), got = try i64(out, "c")
                    XCTAssertEqual(out.length, 9, "\(kind) \(frac) dense=\(String(describing: dense))")
                    for (j, k) in gotKeys.enumerated() {
                        XCTAssertEqual(got[j], want[k!], "\(kind) \(frac) dense=\(String(describing: dense))")
                    }
                }
            }
        }
    }

    func testStreamJoinAggregateCountOverAnyType() throws {
        try requireRealGPU()
        let n = 5000
        for kind in [Kind.int64, .float64, .boolean, .utf8, .decimal128, .null] {
            for frac in Self.fractions {
                let (v, valid) = try column(kind, validity(n, frac, seed: 13))
                let ks = (0..<n).map { Int32($0 % 40) }
                let probe = try MetalRecordBatch(names: ["k", "v"], columns: [.int32(try MetalArray<Int32>(ks)), v])
                // Keys 0..<30 have one build row each, so a probe row matches at most once.
                let (bv, bvalid) = try column(kind, validity(30, frac, seed: 17))
                let build = try MetalRecordBatch(names: ["k", "w"], columns: [.int32(try MetalArray<Int32>((0..<30).map { Int32($0) })), bv])
                let r = try StreamQuery(source: ChunkedTableSource([probe])).join(build, on: "k")
                    .aggregate([StreamAggregate(.count, "v", name: "cv"), StreamAggregate(.count, "w", name: "cw"),
                                StreamAggregate(.count, nil, name: "n")])
                let wantV = (0..<n).filter { ks[$0] < 30 && valid[$0] }.count
                let wantW = (0..<n).filter { ks[$0] < 30 && bvalid[Int(ks[$0])] }.count
                XCTAssertEqual(r.scalar("cv")?.asInt64, Int64(wantV), "\(kind) \(frac) probe side")
                XCTAssertEqual(r.scalar("cw")?.asInt64, Int64(wantW), "\(kind) \(frac) build side")
                XCTAssertEqual(r.scalar("n")?.asInt64, Int64((0..<n).filter { ks[$0] < 30 }.count))
            }
        }
    }
}
