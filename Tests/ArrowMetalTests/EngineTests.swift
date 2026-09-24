import XCTest
@testable import ArrowMetal

/// The lazy query engine: plan construction and type checking, the optimizer's rewrites asserted on
/// `explain()`, and every operator against a Swift oracle at 0 / 1 / 33 / 4097 / 1_000_003 rows with
/// nulls — plus the whole join matrix, the as-of join and the window functions.
final class EngineTests: XCTestCase {

    let sizes = [0, 1, 33, 4097, 1_000_003]

    struct RNG {
        var s: UInt64
        init(_ seed: UInt64) { s = seed &* 0x9E3779B97F4A7C15 &+ 999 }
        mutating func next() -> UInt64 {
            s = s &+ 0x9E3779B97F4A7C15
            var z = s
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
        mutating func int(_ bound: Int) -> Int { bound <= 0 ? 0 : Int(next() % UInt64(bound)) }
    }

    // MARK: - fixtures

    /// region (int32, some nulls), amount (float32, some nulls), qty (int64), name (utf8).
    func sales(_ n: Int, seed: UInt64 = 7, nulls: Bool = true) throws -> MetalRecordBatch {
        var rng = RNG(seed)
        var region: [Int32?] = [], amount: [Float?] = [], qty: [Int64?] = [], name: [String?] = []
        for i in 0..<n {
            region.append(nulls && i % 23 == 5 ? nil : Int32(rng.int(7)))
            amount.append(nulls && i % 17 == 3 ? nil : Float(rng.int(20000)) / 8.0 - 500)
            qty.append(Int64(rng.int(50)) - 10)
            name.append(nulls && i % 31 == 9 ? nil : "k\(rng.int(11))")
        }
        return try MetalRecordBatch(
            names: ["region", "amount", "qty", "name"],
            columns: [.int32(try MetalArray<Int32>(region)),
                      .float32(try MetalArray<Float>(amount)),
                      .int64(try MetalArray<Int64>(qty)),
                      .string(try MetalStringArray(name))])
    }

    func ints(_ v: [Int32?]) throws -> AnyMetalArray { .int32(try MetalArray<Int32>(v)) }
    func longs(_ v: [Int64?]) throws -> AnyMetalArray { .int64(try MetalArray<Int64>(v)) }
    func strs(_ v: [String?]) throws -> AnyMetalArray { .string(try MetalStringArray(v)) }

    func i32col(_ b: MetalRecordBatch, _ n: String) -> [Int32?] {
        guard case .int32(let a)? = b[n] else { return [] }
        return a.toArray()
    }
    func i64col(_ b: MetalRecordBatch, _ n: String) -> [Int64?] {
        guard case .int64(let a)? = b[n] else { return [] }
        return a.toArray()
    }
    func f64col(_ b: MetalRecordBatch, _ n: String) -> [Double?] {
        guard case .float64(let a)? = b[n] else { return [] }
        return a.toArray()
    }
    func f32col(_ b: MetalRecordBatch, _ n: String) -> [Float?] {
        guard case .float32(let a)? = b[n] else { return [] }
        return a.toArray()
    }
    func strcol(_ b: MetalRecordBatch, _ n: String) -> [String?] {
        guard case .string(let a)? = b[n] else { return [] }
        return a.toArray()
    }

    func frame(_ b: MetalRecordBatch, _ name: String = "t") -> LazyFrame {
        LazyFrame(PlanSource(name: name, batch: b))
    }

    // MARK: - 1. plan construction, schema, type checking

    func testSchemaInference() throws {
        let b = try sales(64)
        let f = frame(b)
            .filter(col("amount") > 100)
            .select([NamedExpr("region", col("region")),
                     NamedExpr("double", col("amount") * 2),
                     NamedExpr("name", col("name"))])
        let s = try f.schema()
        XCTAssertEqual(s.names, ["region", "double", "name"])
        XCTAssertEqual(s["region"]?.exprType, .int32)
        XCTAssertEqual(s["double"]?.exprType, .float32)
        XCTAssertEqual(s["name"]?.exprType, .utf8)
    }

    func testAggregateAndGroupBySchema() throws {
        let b = try sales(64)
        let s = try frame(b)
            .groupBy(["region"], [ExprAggregate(.sum, col("qty"), name: "total"),
                                  ExprAggregate(.mean, col("amount"), name: "avg"),
                                  ExprAggregate(.count, nil, name: "n")])
            .schema()
        XCTAssertEqual(s.names, ["region", "total", "avg", "n"])
        XCTAssertEqual(s["total"]?.exprType, .int64)
        XCTAssertEqual(s["avg"]?.exprType, .float64)
        XCTAssertEqual(s["n"]?.exprType, .int64)
    }

    func testUnknownColumnIsAnError() throws {
        let b = try sales(8)
        XCTAssertThrowsError(try frame(b).filter(col("nope") > 1).schema())
        XCTAssertThrowsError(try frame(b).select([NamedExpr("x", col("missing"))]).schema())
        XCTAssertThrowsError(try frame(b).sort("missing").schema())
    }

    func testNonBooleanFilterIsAnError() throws {
        let b = try sales(8)
        XCTAssertThrowsError(try frame(b).filter(col("qty") + 1).schema())
    }

    // MARK: - 2. optimizer rewrites, asserted on explain()

    func testPredicatePushdownThroughProjection() throws {
        let b = try sales(64)
        let f = frame(b)
            .select([NamedExpr("r", col("region")), NamedExpr("a", col("amount"))])
            .filter(col("r") == 3)
        let text = try f.explain()
        XCTAssertTrue(text.contains("predicate_pushdown"), text)
        // The filter now sits under the projection, so the fused kernel does both in one pass.
        XCTAssertTrue(text.contains("FUSED-FILTER-PROJECT"), text)
    }

    func testFilterFusion() throws {
        let b = try sales(64)
        let f = frame(b).filter(col("qty") > 0).filter(col("qty") < 30)
        let text = try f.explain()
        XCTAssertTrue(text.contains("filter_fusion"), text)
        // One node, one predicate: the two filters became one `and`.
        let logical = text.components(separatedBy: "PHYSICAL PLAN")[0]
        XCTAssertEqual(logical.components(separatedBy: "FILTER").count - 1, 1, text)
        XCTAssertTrue(logical.contains("(and (gt (col \"qty\") (int 0)) (lt (col \"qty\") (int 30)))"), text)
    }

    func testConstantFolding() throws {
        let b = try sales(64)
        let f = frame(b).filter((col("qty") > (Expr.lit(2) + Expr.lit(3))) && Expr.bool(true))
        let text = try f.explain()
        XCTAssertTrue(text.contains("constant_folding"), text)
        XCTAssertTrue(text.contains("(int 5)"), text)
        XCTAssertFalse(text.contains("(bool true)"), text)
    }

    func testProjectionPruningNarrowsTheScan() throws {
        let b = try sales(64)
        let f = frame(b).select([NamedExpr("r", col("region"))])
        let text = try f.explain()
        XCTAssertTrue(text.contains("projection_pruning"), text)
        XCTAssertTrue(text.contains("1/4 columns"), text)
    }

    func testPredicatePushdownBelowSortAndIntoAJoin() throws {
        let left = try MetalRecordBatch(names: ["k", "v"], columns: [try ints([1, 2, 3, 4]), try ints([10, 20, 30, 40])])
        let right = try MetalRecordBatch(names: ["k", "w"], columns: [try ints([2, 3, 5]), try ints([200, 300, 500])])
        let f = frame(left, "l")
            .join(frame(right, "r"), on: ["k"], how: .inner)
            .sort("v")
            .filter(col("w") > 200)
        let text = try f.explain()
        XCTAssertTrue(text.contains("predicate_pushdown"), text)
        // The predicate touches only right-side columns, so it must appear below the join, on `r`.
        let lines = text.components(separatedBy: "PHYSICAL PLAN")[0].components(separatedBy: "\n")
        guard let joinLine = lines.firstIndex(where: { $0.contains("JOIN INNER") }),
              let filterLine = lines.firstIndex(where: { $0.contains("FILTER") && $0.contains("\"w\"") })
        else { return XCTFail("no pushed filter in\n\(text)") }
        XCTAssertGreaterThan(filterLine, joinLine, text)
    }

    func testExplainIsStableAndMentionsBothPlans() throws {
        let b = try sales(64)
        let f = frame(b).filter(col("qty") > 0)
            .groupBy(["region"], [ExprAggregate(.sum, col("qty"), name: "total")])
            .sort([SortKey("total", descending: true)])
            .limit(5)
        let text = try f.explain()
        XCTAssertTrue(text.contains("LOGICAL PLAN"), text)
        XCTAssertTrue(text.contains("PHYSICAL PLAN"), text)
        XCTAssertTrue(text.contains("HASH-AGGREGATE"), text)
        XCTAssertEqual(text, try f.explain())
    }

    func testJoinReorderPutsTheSmallerSideOnTheBuild() throws {
        var big: [Int32?] = [], small: [Int32?] = []
        for i in 0..<10_000 { big.append(Int32(i % 500)) }
        for i in 0..<50 { small.append(Int32(i)) }
        let l = try MetalRecordBatch(names: ["k"], columns: [try ints(small)])
        let r = try MetalRecordBatch(names: ["k", "v"], columns: [try ints(big), try ints(big)])
        let joined = frame(l, "small").join(frame(r, "big"), on: ["k"], how: .inner)

        // Swapping the sides permutes the rows, and docs/ENGINE.md promises an inner join keeps
        // probe (left) order — so the rule only fires where nothing above can see the order.
        let bare = try joined.explain()
        XCTAssertFalse(bare.contains("join_reorder"), bare)

        let grouped = joined.groupBy(["k"], [ExprAggregate(.sum, col("v"), name: "total")])
        let text = try grouped.explain()
        XCTAssertTrue(text.contains("join_reorder"), text)
        // The column order the caller asked for is restored by a projection on top.
        XCTAssertEqual(try joined.schema().names, ["k", "v"])
        XCTAssertEqual(try grouped.schema().names, ["k", "total"])
    }

    // MARK: - 3. filter + project against an oracle

    func testFilterProjectAgainstOracle() throws {
        try requireRealGPU()
        for n in sizes {
            let b = try sales(n)
            let region = i32col(b, "region"), qty = i64col(b, "qty"), amount = f32col(b, "amount")
            let out = try frame(b)
                .filter((col("region") == 3) && (col("qty") > 0))
                .select([NamedExpr("region", col("region")),
                         NamedExpr("scaled", col("qty") * 2 + 1)])
                .collect()
            var wantRegion: [Int32?] = [], wantScaled: [Int64?] = []
            for i in 0..<n {
                guard let r = region[i], r == 3, let q = qty[i], q > 0 else { continue }
                wantRegion.append(r)
                wantScaled.append(q * 2 + 1)
            }
            XCTAssertEqual(i32col(out, "region"), wantRegion, "n=\(n)")
            XCTAssertEqual(i64col(out, "scaled"), wantScaled, "n=\(n)")
            XCTAssertEqual(out.length, wantRegion.count)
            _ = amount
        }
    }

    func testFilterCarryingAStringColumn() throws {
        try requireRealGPU()
        for n in [0, 1, 33, 4097] {
            let b = try sales(n)
            let qty = i64col(b, "qty"), name = strcol(b, "name")
            let out = try frame(b).filter(col("qty") > 20).select(["qty", "name"]).collect()
            var wq: [Int64?] = [], wn: [String?] = []
            for i in 0..<n where (qty[i] ?? -1) > 20 { wq.append(qty[i]); wn.append(name[i]) }
            XCTAssertEqual(i64col(out, "qty"), wq, "n=\(n)")
            XCTAssertEqual(strcol(out, "name"), wn, "n=\(n)")
        }
    }

    func testWholeInputAggregate() throws {
        try requireRealGPU()
        for n in [0, 1, 33, 4097, 1_000_003] {
            let b = try sales(n)
            let qty = i64col(b, "qty"), region = i32col(b, "region")
            let out = try frame(b).filter(col("region") == 2)
                .aggregate([ExprAggregate(.sum, col("qty"), name: "s"),
                            ExprAggregate(.count, nil, name: "c")]).collect()
            var s: Int64 = 0, c: Int64 = 0
            for i in 0..<n where region[i] == 2 { s += qty[i] ?? 0; c += 1 }
            XCTAssertEqual(i64col(out, "s").first ?? nil, c == 0 ? nil : s, "n=\(n)")
            XCTAssertEqual(i64col(out, "c").first ?? nil, c, "n=\(n)")
        }
    }

    // MARK: - 4. group-by

    func testGroupByIntKeyAgainstOracle() throws {
        try requireRealGPU()
        for n in [0, 1, 33, 4097, 1_000_003] {
            let b = try sales(n)
            let region = i32col(b, "region"), qty = i64col(b, "qty")
            let out = try frame(b)
                .groupBy(["region"], [ExprAggregate(.sum, col("qty"), name: "total"),
                                      ExprAggregate(.count, nil, name: "n")])
                .collect()
            var sums: [Int32?: Int64] = [:], counts: [Int32?: Int64] = [:]
            for i in 0..<n {
                sums[region[i], default: 0] += qty[i] ?? 0
                counts[region[i], default: 0] += 1
            }
            let keys = i32col(out, "region")
            XCTAssertEqual(Set(keys.map { $0 }), Set(sums.keys), "n=\(n)")
            let totals = i64col(out, "total"), ns = i64col(out, "n")
            for (j, k) in keys.enumerated() {
                XCTAssertEqual(totals[j], sums[k], "n=\(n) key=\(String(describing: k))")
                XCTAssertEqual(ns[j], counts[k], "n=\(n) key=\(String(describing: k))")
            }
        }
    }

    func testGroupByStringAndMultiKey() throws {
        try requireRealGPU()
        for n in [33, 4097] {
            let b = try sales(n)
            let region = i32col(b, "region"), name = strcol(b, "name"), qty = i64col(b, "qty")
            let out = try frame(b)
                .groupBy(["name", "region"], [ExprAggregate(.max, col("qty"), name: "hi")])
                .collect()
            struct Key: Hashable { var name: String?; var region: Int32? }
            var want: [Key: Int64] = [:]
            for i in 0..<n {
                let k = Key(name: name[i], region: region[i])
                if let q = qty[i] { want[k] = Swift.max(want[k] ?? Int64.min, q) }
                else if want[k] == nil { want[k] = Int64.min }
            }
            let outNames = strcol(out, "name"), outRegions = i32col(out, "region"), his = i64col(out, "hi")
            XCTAssertEqual(out.length, want.count, "n=\(n)")
            for j in 0..<out.length {
                let k = Key(name: outNames[j], region: outRegions[j])
                let expected = want[k]
                XCTAssertEqual(his[j], expected == Int64.min ? nil : expected, "n=\(n)")
            }
        }
    }

    func testGroupByOverAnExpressionAndFloat64Sum() throws {
        try requireRealGPU()
        let n = 4097
        let b = try sales(n)
        let region = i32col(b, "region"), qty = i64col(b, "qty")
        let out = try frame(b)
            .groupBy([NamedExpr("bucket", col("region") / 2)],
                     [ExprAggregate(.sum, col("qty").cast(to: .float64), name: "s"),
                      ExprAggregate(.mean, col("qty"), name: "m")])
            .collect()
        var sums: [Int32?: Double] = [:], counts: [Int32?: Int] = [:]
        for i in 0..<n {
            let k = region[i].map { $0 / 2 }
            sums[k, default: 0] += Double(qty[i] ?? 0)
            counts[k, default: 0] += 1
        }
        let keys = i32col(out, "bucket"), ss = f64col(out, "s"), ms = f64col(out, "m")
        for (j, k) in keys.enumerated() {
            XCTAssertEqual(ss[j] ?? .nan, sums[k] ?? .nan, accuracy: 1e-6)
            XCTAssertEqual(ms[j] ?? .nan, (sums[k] ?? 0) / Double(counts[k] ?? 1), accuracy: 1e-6)
        }
    }

    // MARK: - 5. sort, limit, distinct

    func testMultiKeySortAndLimit() throws {
        try requireRealGPU()
        for n in [0, 1, 33, 4097] {
            let b = try sales(n)
            let out = try frame(b)
                .sort([SortKey("region"), SortKey("qty", descending: true)])
                .limit(10)
                .collect()
            let region = i32col(out, "region"), qty = i64col(out, "qty")
            XCTAssertEqual(out.length, Swift.min(n, 10))
            for i in 1..<Swift.max(out.length, 1) {
                let a = region[i - 1], c = region[i]
                if a == nil { XCTAssertNil(c, "nulls last") ; continue }
                guard let a, let c else { continue }
                XCTAssertLessThanOrEqual(a, c)
                if a == c { XCTAssertGreaterThanOrEqual(qty[i - 1] ?? .min, qty[i] ?? .min) }
            }
        }
    }

    func testTopKMatchesTheFullSort() throws {
        try requireRealGPU()
        let b = try sales(50_000, nulls: false)
        let viaTopK = try frame(b).sort([SortKey("qty", descending: true)]).limit(20).collect()
        let viaSort = try frame(b).sort([SortKey("qty", descending: true)]).collect().slice(offset: 0, length: 20)
        XCTAssertEqual(i64col(viaTopK, "qty"), i64col(viaSort, "qty"))
        XCTAssertTrue(try frame(b).sort([SortKey("qty", descending: true)]).limit(20).explain().contains("TOP-K"))
    }

    func testDistinctKeepsTheFirstOccurrence() throws {
        try requireRealGPU()
        let b = try MetalRecordBatch(names: ["k", "v"],
                                     columns: [try ints([3, 1, 3, nil, 1, nil, 2]),
                                               try ints([10, 11, 12, 13, 14, 15, 16])])
        let out = try frame(b).unique(subset: ["k"]).collect()
        XCTAssertEqual(i32col(out, "k"), [3, 1, nil, 2])
        XCTAssertEqual(i32col(out, "v"), [10, 11, 13, 16])
    }

    // MARK: - 6. joins

    /// A Swift oracle for the whole join matrix over one integer key.
    func joinOracle(_ lk: [Int32?], _ rk: [Int32?], _ how: JoinHow) -> [(Int?, Int?)] {
        var pairs: [(Int?, Int?)] = []
        switch how {
        case .inner, .left, .semi, .anti, .full:
            for (i, a) in lk.enumerated() {
                var matches: [Int] = []
                if let a { for (j, b) in rk.enumerated() where b == a { matches.append(j) } }
                switch how {
                case .inner: for j in matches { pairs.append((i, j)) }
                case .left, .full:
                    if matches.isEmpty { pairs.append((i, nil)) } else { for j in matches { pairs.append((i, j)) } }
                case .semi: if !matches.isEmpty { pairs.append((i, nil)) }
                case .anti: if matches.isEmpty { pairs.append((i, nil)) }
                default: break
                }
            }
            if how == .full {
                for (j, b) in rk.enumerated() {
                    let matched = b != nil && lk.contains { $0 == b }
                    if !matched { pairs.append((nil, j)) }
                }
            }
        case .right:
            for (j, b) in rk.enumerated() {
                var matches: [Int] = []
                if let b { for (i, a) in lk.enumerated() where a == b { matches.append(i) } }
                if matches.isEmpty { pairs.append((nil, j)) } else { for i in matches { pairs.append((i, j)) } }
            }
        }
        return pairs
    }

    func testJoinMatrixOnAnIntegerKey() throws {
        try requireRealGPU()
        var rng = RNG(11)
        for n in [0, 1, 33, 4097] {
            var lk: [Int32?] = [], lv: [Int32?] = []
            var rk: [Int32?] = [], rv: [Int32?] = []
            for i in 0..<n {
                lk.append(i % 13 == 4 ? nil : Int32(rng.int(n / 3 + 2)))
                lv.append(Int32(i))
                rk.append(i % 11 == 6 ? nil : Int32(rng.int(n / 3 + 2)))
                rv.append(Int32(1000 + i))
            }
            let l = try MetalRecordBatch(names: ["k", "lv"], columns: [try ints(lk), try ints(lv)])
            let r = try MetalRecordBatch(names: ["k", "rv"], columns: [try ints(rk), try ints(rv)])
            for how in JoinHow.allCases {
                let out = try frame(l, "l").join(frame(r, "r"), on: ["k"], how: how).collect()
                let want = joinOracle(lk, rk, how)
                XCTAssertEqual(out.length, want.count, "n=\(n) how=\(how.rawValue)")
                // Compare as multisets of (lv, rv): the pair order inside one probe row is unspecified.
                let gotLV = i32col(out, "lv")
                var got: [String] = []
                if how == .semi || how == .anti {
                    got = gotLV.map { "\(String(describing: $0))" }
                } else {
                    let gotRV = i32col(out, "rv")
                    for i in 0..<out.length { got.append("\(String(describing: gotLV[i]))/\(String(describing: gotRV[i]))") }
                }
                var expect: [String] = []
                for (i, j) in want {
                    let a = i.map { lv[$0] } ?? nil
                    if how == .semi || how == .anti { expect.append("\(String(describing: a))") }
                    else { expect.append("\(String(describing: a))/\(String(describing: j.map { rv[$0] } ?? nil))") }
                }
                XCTAssertEqual(got.sorted(), expect.sorted(), "n=\(n) how=\(how.rawValue)")
            }
        }
    }

    func testJoinOnUtf8AndMultipleKeys() throws {
        try requireRealGPU()
        let l = try MetalRecordBatch(names: ["a", "b", "lv"],
                                     columns: [try strs(["x", "y", nil, "x", "z"]),
                                               try ints([1, 2, 3, 2, 1]),
                                               try ints([10, 20, 30, 40, 50])])
        let r = try MetalRecordBatch(names: ["a", "b", "rv"],
                                     columns: [try strs(["x", "x", "y", nil]),
                                               try ints([1, 2, 2, 3]),
                                               try ints([100, 200, 300, 400])])
        let out = try frame(l, "l").join(frame(r, "r"), on: ["a", "b"], how: .inner).collect()
        var got: [String] = []
        let lv = i32col(out, "lv"), rv = i32col(out, "rv")
        for i in 0..<out.length { got.append("\(lv[i]!)/\(rv[i]!)") }
        XCTAssertEqual(got.sorted(), ["10/100", "20/300", "40/200"].sorted())

        // utf8 alone, left join, nulls never match.
        let outL = try frame(l, "l").join(frame(r, "r"), leftOn: ["a"], rightOn: ["a"], how: .left).collect()
        XCTAssertEqual(outL.length, 7)          // x:2, y:1, null:1 row, x:2, z:1 row
        XCTAssertEqual(i32col(outL, "rv").filter { $0 == nil }.count, 2) // the null key and "z"
    }

    func testJoinKeyColumnOfAFullOuterJoinIsFilled() throws {
        try requireRealGPU()
        let l = try MetalRecordBatch(names: ["k", "lv"], columns: [try ints([1, 2]), try ints([10, 20])])
        let r = try MetalRecordBatch(names: ["k", "rv"], columns: [try ints([2, 3]), try ints([200, 300])])
        let out = try frame(l, "l").join(frame(r, "r"), on: ["k"], how: .full).collect()
        XCTAssertEqual(Set(i32col(out, "k").compactMap { $0 }), [1, 2, 3])
        XCTAssertEqual(out.length, 3)
    }

    // MARK: - 7. as-of join

    func testAsofBackwardForwardNearest() throws {
        try requireRealGPU()
        let probe = try MetalRecordBatch(names: ["t"], columns: [try longs([1, 5, 9, 12, 20, nil])])
        let build = try MetalRecordBatch(names: ["t", "v"],
                                         columns: [try longs([2, 6, 10, 10, 15]), try ints([1, 2, 3, 4, 5])])
        func run(_ s: AsofStrategy) throws -> [Int32?] {
            let out = try frame(probe, "p")
                .joinAsof(frame(build, "b"), AsofSpec(leftOn: "t", rightOn: "t", strategy: s))
                .collect()
            return i32col(out, "v")
        }
        // backward: last build key <= probe key. Duplicate build keys resolve to the sorted-last one.
        let back = try run(.backward)
        XCTAssertEqual(back[0], nil)          // 1 has nothing at or before it
        XCTAssertEqual(back[1], 1)            // 5 -> 2
        XCTAssertEqual(back[2], 2)            // 9 -> 6
        XCTAssertTrue(back[3] == 3 || back[3] == 4)   // 12 -> one of the two 10s
        XCTAssertEqual(back[4], 5)            // 20 -> 15
        XCTAssertEqual(back[5], nil)          // null key never matches

        let fwd = try run(.forward)
        XCTAssertEqual(fwd[0], 1)             // 1 -> 2
        XCTAssertEqual(fwd[1], 2)             // 5 -> 6
        XCTAssertTrue(fwd[2] == 3 || fwd[2] == 4)
        XCTAssertEqual(fwd[3], 5)             // 12 -> 15
        XCTAssertEqual(fwd[4], nil)           // nothing at or after 20

        let near = try run(.nearest)
        XCTAssertEqual(near[0], 1)            // |1-2| = 1
        XCTAssertEqual(near[1], 2)            // |5-2| = 3 vs |5-6| = 1, so 6 wins
        XCTAssertEqual(try run(.nearest)[4], 5)
    }

    func testAsofWithPartitionsAndTolerance() throws {
        try requireRealGPU()
        let probe = try MetalRecordBatch(names: ["g", "t"],
                                         columns: [try strs(["a", "a", "b", "b"]), try longs([10, 30, 10, 30])])
        let build = try MetalRecordBatch(names: ["g", "t", "v"],
                                         columns: [try strs(["a", "a", "b"]),
                                                   try longs([5, 25, 1]),
                                                   try ints([1, 2, 3])])
        let out = try frame(probe, "p")
            .joinAsof(frame(build, "b"), AsofSpec(leftOn: "t", rightOn: "t", by: ["g"]))
            .collect()
        XCTAssertEqual(i32col(out, "v"), [1, 2, 3, 3])

        let tol = try frame(probe, "p")
            .joinAsof(frame(build, "b"), AsofSpec(leftOn: "t", rightOn: "t", by: ["g"], tolerance: 6))
            .collect()
        XCTAssertEqual(i32col(tol, "v"), [1, 2, nil, nil])
    }

    func testAsofAtScaleAgainstAnOracle() throws {
        try requireRealGPU()
        var rng = RNG(23)
        let nP = 4097, nB = 1301
        var pt: [Int64?] = [], bt: [Int64?] = [], bv: [Int32?] = []
        for _ in 0..<nP { pt.append(Int64(rng.int(100_000))) }
        var t: Int64 = 0
        for i in 0..<nB { t += Int64(rng.int(150)); bt.append(t); bv.append(Int32(i)) }
        let probe = try MetalRecordBatch(names: ["t"], columns: [try longs(pt)])
        let build = try MetalRecordBatch(names: ["t", "v"], columns: [try longs(bt), try ints(bv)])
        let out = try frame(probe, "p")
            .joinAsof(frame(build, "b"), AsofSpec(leftOn: "t", rightOn: "t"))
            .collect()
        let got = i32col(out, "v")
        for i in 0..<nP {
            var want: Int32? = nil
            for j in 0..<nB where bt[j]! <= pt[i]! { want = bv[j] }
            XCTAssertEqual(got[i], want, "row \(i)")
        }
    }

    // MARK: - 8. windows

    func testWindowRankFamily() throws {
        try requireRealGPU()
        let b = try MetalRecordBatch(names: ["g", "v"],
                                     columns: [try ints([1, 1, 1, 2, 2, 2]),
                                               try ints([5, 5, 9, 7, 1, 7])])
        let out = try frame(b).window([
            WindowSpec(name: "rn", function: .rowNumber, partitionBy: ["g"], orderBy: [SortKey("v")]),
            WindowSpec(name: "rk", function: .rank, partitionBy: ["g"], orderBy: [SortKey("v")]),
            WindowSpec(name: "dr", function: .denseRank, partitionBy: ["g"], orderBy: [SortKey("v")]),
        ]).collect()
        XCTAssertEqual(i32col(out, "rn"), [1, 2, 3, 2, 1, 3])
        XCTAssertEqual(i32col(out, "rk"), [1, 1, 3, 2, 1, 2])
        XCTAssertEqual(i32col(out, "dr"), [1, 1, 2, 2, 1, 2])
    }

    func testWindowLagLeadAndRolling() throws {
        try requireRealGPU()
        let b = try MetalRecordBatch(names: ["g", "t", "v"],
                                     columns: [try ints([1, 1, 1, 2, 2]),
                                               try ints([1, 2, 3, 1, 2]),
                                               try longs([10, 20, 30, 40, 50])])
        let out = try frame(b).window([
            WindowSpec(name: "lag1", function: .lag("v", 1), partitionBy: ["g"], orderBy: [SortKey("t")]),
            WindowSpec(name: "lead1", function: .lead("v", 1), partitionBy: ["g"], orderBy: [SortKey("t")]),
            WindowSpec(name: "cs", function: .cumSum("v"), partitionBy: ["g"], orderBy: [SortKey("t")]),
            WindowSpec(name: "rs", function: .rollingSum("v", 2), partitionBy: ["g"], orderBy: [SortKey("t")]),
            WindowSpec(name: "tot", function: .partitionAggregate(.sum, "v"), partitionBy: ["g"]),
        ]).collect()
        XCTAssertEqual(i64col(out, "lag1"), [nil, 10, 20, nil, 40])
        XCTAssertEqual(i64col(out, "lead1"), [20, 30, nil, 50, nil])
        XCTAssertEqual(i64col(out, "cs"), [10, 30, 60, 40, 90])
        XCTAssertEqual(i64col(out, "rs"), [nil, 30, 50, nil, 90])
        XCTAssertEqual(i64col(out, "tot"), [60, 60, 60, 90, 90])
    }

    func testWindowAtScaleAgainstAnOracle() throws {
        try requireRealGPU()
        var rng = RNG(31)
        let n = 20_003
        var g: [Int32?] = [], v: [Int64?] = []
        for _ in 0..<n { g.append(Int32(rng.int(64))); v.append(Int64(rng.int(1000))) }
        let b = try MetalRecordBatch(names: ["g", "v"], columns: [try ints(g), try longs(v)])
        let out = try frame(b).window([
            WindowSpec(name: "rn", function: .rowNumber, partitionBy: ["g"], orderBy: [SortKey("v")])
        ]).collect()
        // ROW_NUMBER over (partition g order v) is a permutation of 1...count within each partition.
        var seen: [Int32: Set<Int32>] = [:]
        let rn = i32col(out, "rn")
        for i in 0..<n { seen[g[i]!, default: []].insert(rn[i]!) }
        for (key, set) in seen {
            XCTAssertEqual(set, Set(1...Int32(set.count)), "partition \(key)")
        }
    }

    // MARK: - 9. concat and explode

    func testConcat() throws {
        try requireRealGPU()
        let a = try MetalRecordBatch(names: ["k", "s"], columns: [try ints([1, nil]), try strs(["a", "b"])])
        let b = try MetalRecordBatch(names: ["k", "s"], columns: [try ints([3]), try strs([nil])])
        let out = try frame(a, "a").concat([frame(b, "b")]).collect()
        XCTAssertEqual(i32col(out, "k"), [1, nil, 3])
        XCTAssertEqual(strcol(out, "s"), ["a", "b", nil])
    }

    func testExplode() throws {
        try requireRealGPU()
        let child = try MetalArray<Int32>([1, 2, 3, 4])
        let offsets = try MetalArrowBuffer.allocate(byteCount: 4 * 4, zeroed: false)
        let op = offsets.mutableTyped(Int32.self)
        op[0] = 0; op[1] = 2; op[2] = 2; op[3] = 4
        let list = MetalListArray(length: 3, nullCount: 0, validity: nil, offsets: offsets,
                                  values: .int32(child), context: .shared)
        let b = try MetalRecordBatch(names: ["id", "xs"],
                                     columns: [try ints([10, 20, 30]), .list(list)])
        let out = try frame(b).explode(["xs"]).collect()
        XCTAssertEqual(i32col(out, "id"), [10, 10, 20, 30, 30])
        XCTAssertEqual(i32col(out, "xs"), [1, 2, nil, 3, 4])
    }

    // MARK: - 10. the whole thing, and batching

    func testTPCHShapedQuery() throws {
        try requireRealGPU()
        let n = 200_003
        let b = try sales(n)
        let region = i32col(b, "region"), qty = i64col(b, "qty"), amount = f32col(b, "amount")
        let out = try frame(b, "sales")
            .filter((col("amount") > 0) && (col("qty") > 5))
            .groupBy(["region"], [ExprAggregate(.sum, col("qty"), name: "total"),
                                  ExprAggregate(.count, nil, name: "n")])
            .sort([SortKey("total", descending: true)])
            .limit(3)
            .collect()
        var sums: [Int32?: Int64] = [:], counts: [Int32?: Int64] = [:]
        for i in 0..<n {
            guard let a = amount[i], a > 0, let q = qty[i], q > 5 else { continue }
            sums[region[i], default: 0] += q
            counts[region[i], default: 0] += 1
        }
        let want = sums.sorted { $0.value > $1.value }.prefix(3)
        XCTAssertEqual(i64col(out, "total"), want.map { $0.value })
        XCTAssertEqual(i32col(out, "region"), want.map { $0.key })
        XCTAssertEqual(i64col(out, "n"), want.map { counts[$0.key]! })
    }

    func testCollectInsideAnOpenBatchMatches() throws {
        try requireRealGPU()
        let b = try sales(4097)
        let plain = try frame(b).filter(col("qty") > 10).select(["qty"]).collect()
        let inside = try MetalContext.shared.batch {
            try frame(b).filter(col("qty") > 10).select(["qty"]).collect()
        }
        XCTAssertEqual(i64col(plain, "qty"), i64col(inside, "qty"))
    }

    func testUnoptimizedAndOptimizedAgree() throws {
        try requireRealGPU()
        let b = try sales(4097)
        let f = frame(b)
            .select([NamedExpr("r", col("region")), NamedExpr("q", col("qty"))])
            .filter(col("r") == 4)
            .sort([SortKey("q")])
        let opt = try f.collect(optimize: true)
        let raw = try f.collect(optimize: false)
        XCTAssertEqual(i64col(opt, "q"), i64col(raw, "q"))
        XCTAssertEqual(i32col(opt, "r"), i32col(raw, "r"))
    }

    // MARK: - findings from the Polars engine lane's differential suite

    /// A null utf8 slot may keep the bytes it held (valid Arrow: a null slot's contents are
    /// unspecified; Polars exports them). A gather gives it length 0 in the output, so its bytes must
    /// not be copied: they used to land on the next kept row ("xanana" for "banana").
    func testStringGatherIgnoresBytesUnderANullSlot() throws {
        try requireRealGPU()
        func withBytesUnderNulls() throws -> MetalStringArray {
            let off = try MetalArrowBuffer.allocate(byteCount: 16)
            let o = off.mutableTyped(Int32.self)
            for (i, v) in [0, 1, 7, 13].enumerated() { o[i] = Int32(v) }
            let text = Array("xcherrybanana".utf8)
            let dat = try MetalArrowBuffer.allocate(byteCount: text.count)
            for (i, b) in text.enumerated() { dat.mutableTyped(UInt8.self)[i] = b }
            let bm = try MetalArrowBuffer.allocate(byteCount: 1)
            bm.mutableTyped(UInt8.self)[0] = 0b110
            return MetalStringArray(length: 3, nullCount: 1, validity: bm, offsets: off, data: dat)
        }
        let a = try withBytesUnderNulls()
        XCTAssertEqual(try a.filter(try MetalBooleanArray([true, false, true])).toArray(), [nil, "banana"])
        XCTAssertEqual(try a.take(try MetalArray<Int32>([0, 2, 0, 1])).toArray(), [nil, "banana", nil, "cherry"])
        // Through the plan's filter as well.
        let b = try MetalRecordBatch(names: ["s", "k"], columns: [.string(a), try ints([1, 0, 1])])
        XCTAssertEqual(strcol(try frame(b).filter(col("k") > 0).collect(), "s"), [nil, "banana"])
    }

    /// A Boolean column carried through the plan's sort keeps its null count. The sort gathers it with
    /// `take`, whose null count the open batch computes only at the flush; the Boolean repack copied the
    /// count before that and reported 0 over a bitmap with nulls.
    func testBooleanThroughSortKeepsItsNullCount() throws {
        try requireRealGPU()
        let n = 33
        let flags: [Bool?] = (0..<n).map { $0 % 3 == 0 ? nil : $0 % 2 == 0 }
        let bm = try MetalArrowBuffer.allocate(byteCount: 8), vb = try MetalArrowBuffer.allocate(byteCount: 8)
        for (i, f) in flags.enumerated() {
            if f != nil { Bitmap.set(bm.mutableTyped(UInt8.self), i) }
            if f == true { Bitmap.set(vb.mutableTyped(UInt8.self), i) }
        }
        let g = MetalBooleanArray(length: n, nullCount: 11, validity: bm, values: vb)
        let b = try MetalRecordBatch(names: ["a", "g"],
                                     columns: [try ints((0..<n).map { Int32(n - 1 - $0) }), .boolean(g)])
        let out = try frame(b).sort("a").collect()
        guard case .boolean(let s)? = out["g"] else { return XCTFail("g is not boolean") }
        XCTAssertEqual(s.nullCount, 11)
        XCTAssertEqual(s.toArray(), Array(flags.reversed()))

        // The same hand-off for a numeric element-wise result: of a `take` (count due at the flush)
        // and of a `filter` (length and count due at the flush).
        let v = try MetalArray<Int32>((0..<n).map { $0 % 4 == 0 ? nil : Int32($0) })
        let (taken, filtered) = try MetalContext.shared.batch {
            (try v.take(try MetalArray<Int32>((0..<n).map { Int32($0) })).cast(to: Int64.self),
             try v.filter(try MetalBooleanArray((0..<n).map { $0 % 2 == 0 })).cast(to: Int64.self))
        }
        XCTAssertEqual(taken.nullCount, 9)
        XCTAssertEqual(filtered.nullCount, 9)
        XCTAssertEqual(filtered.length, 17)
    }

    /// A filter over a batch that carries a date32 column it never reads. The expression compiler
    /// bound every column of the batch as a kernel input and refused the date32 one.
    func testFilterCarriesAColumnTheCompilerDoesNotRead() throws {
        try requireRealGPU()
        let days = try MetalTemporalArray(type: .date32, (0..<10).map { Int64($0) })
        let b = try MetalRecordBatch(names: ["t", "a"],
                                     columns: [.temporal(days), try longs((-5..<5).map { Int64($0) })])
        let out = try frame(b).filter(col("a") > 0).collect()
        XCTAssertEqual(out.length, 4)
        XCTAssertEqual(out["t"]?.asTemporal?.toArray(), [6, 7, 8, 9])
        XCTAssertEqual(i64col(out, "a"), [1, 2, 3, 4])
    }

    /// A utf8 sort with a null and a row of 8+ bytes (two prefix chunks) inside a batch. The null
    /// partition read the index array on the CPU before the GPU had written it.
    func testStringSortWithANullAndALongRowInABatch() throws {
        try requireRealGPU()
        let xs: [String?] = ["b", "a", nil, "c", "ab", "xxxxxxxx"]
        let b = try MetalRecordBatch(names: ["s"], columns: [try strs(xs)])
        XCTAssertEqual(strcol(try frame(b).sort("s").collect(), "s"), ["a", "ab", "b", "c", "xxxxxxxx", nil])
        XCTAssertEqual(strcol(try frame(b).sort("s", descending: true).collect(), "s"),
                       ["xxxxxxxx", "c", "b", "ab", "a", nil])
        let a = try MetalStringArray(xs)
        let sorted = try MetalContext.shared.batch { try a.sorted() }
        XCTAssertEqual(sorted.toArray(), ["a", "ab", "b", "c", "xxxxxxxx", nil])
    }

    /// The null partition runs in the prefix keys on the GPU: every direction and placement, inside and
    /// outside a batch, with ties, empty strings and rows of up to three prefix chunks, against a CPU
    /// reference (stable byte order for the valid rows, the null rows in row order at the named end).
    func testStringArgsortNullPlacementOnTheGPU() throws {
        try requireRealGPU()
        var rng = SystemRandomNumberGenerator()
        let alphabet: [Character] = ["a", "b", "\u{0}", "z"]
        var xs: [String?] = (0..<3000).map { _ in
            if Int.random(in: 0..<10, using: &rng) == 0 { return nil }
            return String((0..<Int.random(in: 0...20, using: &rng)).map { _ in alphabet.randomElement(using: &rng)! })
        }
        xs += [nil, "", "", nil, "abcdefghijklmnopqrst", "abcdefghijklmnopqrss"]
        func reference(_ desc: Bool, _ placement: NullPlacement) -> [Int32] {
            let valid = xs.indices.filter { xs[$0] != nil }.sorted { l, r in
                let a = Array(xs[l]!.utf8), b = Array(xs[r]!.utf8)
                if a == b { return l < r }
                return desc ? b.lexicographicallyPrecedes(a) : a.lexicographicallyPrecedes(b)
            }
            let nulls = xs.indices.filter { xs[$0] == nil }
            return (placement == .atStart ? nulls + valid : valid + nulls).map { Int32($0) }
        }
        let a = try MetalStringArray(xs)
        for desc in [false, true] {
            for placement in [NullPlacement.atEnd, .atStart] {
                let want = reference(desc, placement)
                XCTAssertEqual(try a.argsort(descending: desc, nullPlacement: placement).toArray(), want)
                let inBatch = try MetalContext.shared.batch {
                    try a.argsort(descending: desc, nullPlacement: placement)
                }
                XCTAssertEqual(inBatch.toArray(), want)
                let sorted = try MetalContext.shared.batch { try a.sorted(descending: desc, nullPlacement: placement) }
                XCTAssertEqual(sorted.toArray(), want.map { xs[Int($0)] })
            }
        }
        // Valid rows that are all empty still need a pass to move the nulls; an all-null column is the identity.
        let empties = try MetalStringArray(["", nil, "", nil, ""])
        XCTAssertEqual(try empties.argsort().toArray(), [0, 2, 4, 1, 3])
        XCTAssertEqual(try empties.argsort(nullPlacement: .atStart).toArray(), [1, 3, 0, 2, 4])
        let allNull = try MetalStringArray([nil, nil, nil] as [String?])
        XCTAssertEqual(try allNull.argsort(descending: true, nullPlacement: .atStart).toArray(), [0, 1, 2])
    }
}
