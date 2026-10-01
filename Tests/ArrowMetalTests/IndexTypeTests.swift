import XCTest
import Metal
@testable import ArrowMetal

/// Index arrays are UInt32: argsort, top-k, partition_nth, the lexsort, join indices, the ranks, the
/// group-by row numbers. A row number of 2^31 or more comes back as itself (an Int32 index array showed
/// it as a negative number), and a call whose row numbers would pass 2^32 - 1 is refused.
final class IndexTypeTests: XCTestCase {

    // MARK: - the type and the values

    func testIndexArraysAreUInt32() throws {
        try requireRealGPU()
        let vals: [Int64?] = [5, nil, 3, 9, 1, 9, 3]
        let a = try MetalArray<Int64>(vals)

        let order: MetalArray<UInt32> = try a.argsort()
        XCTAssertEqual(order.toRawArray(), [4, 2, 6, 0, 3, 5, 1])
        let top: MetalArray<UInt32> = try a.topK(2)
        XCTAssertEqual(top.toRawArray(), [3, 5])
        let bottom: MetalArray<UInt32> = try a.topK(3, largest: false, nullPlacement: .atStart)
        XCTAssertEqual(bottom.toRawArray(), [1, 4, 2])
        let part: MetalArray<UInt32> = try a.partitionNthIndices(3)
        let p = part.toRawArray()
        XCTAssertEqual(Set(p), Set((0..<7).map(UInt32.init)))
        XCTAssertEqual(Set(p.prefix(3)), [4, 2, 6], "the three smallest first")

        let lex: MetalArray<UInt32> = try lexsortIndices([.int64(a), .int64(a)], descending: [true, false])
        XCTAssertEqual(lex.toRawArray(), [3, 5, 0, 2, 6, 4, 1])
        let any: MetalArray<UInt32> = try AnyMetalArray.int64(a).argsortIndices(descending: true)
        XCTAssertEqual(any.toRawArray(), [3, 5, 0, 2, 6, 4, 1])

        let rowNumber: MetalArray<UInt32> = try a.rowNumber()
        XCTAssertEqual(rowNumber.toRawArray(), [4, 7, 2, 5, 1, 6, 3])
        XCTAssertEqual(try a.rank().toRawArray(), [4, 7, 2, 5, 1, 5, 2])
        XCTAssertEqual(try a.denseRank().toRawArray(), [3, 5, 2, 4, 1, 4, 2])
        XCTAssertEqual(try a.maxRank().toRawArray(), [4, 7, 3, 6, 1, 6, 3])
        XCTAssertEqual(try a.rank(tiebreaker: .first).toRawArray(), [4, 7, 2, 5, 1, 6, 3])

        let s = try MetalStringArray(["pear", nil, "apple", "fig"])
        let sOrder: MetalArray<UInt32> = try s.argsort()
        XCTAssertEqual(sOrder.toRawArray(), [2, 3, 0, 1])
        let t = try MetalTemporalArray(type: .date32, try MetalArray<Int32>([30, 10, 20]))
        let tOrder: MetalArray<UInt32> = try t.argsort()
        XCTAssertEqual(tOrder.toRawArray(), [1, 2, 0])

        let l = try MetalArray<Int32>([1, 2, 3, 2]), r = try MetalArray<Int32>([2, 3, 4])
        let (li, ri): (MetalArray<UInt32>, MetalArray<UInt32>) = try hashJoin(left: l, right: r, kind: .inner)
        let pairs = zip(li.toRawArray(), ri.toRawArray()).map { [$0.0, $0.1] }.sorted { $0.lexicographicallyPrecedes($1) }
        XCTAssertEqual(pairs, [[1, 0], [2, 1], [3, 0]])
        let full: JoinIndexPairs = try JoinExtra.indices(leftKeys: [.int32(l)], rightKeys: [.int32(r)], how: .full)
        let fullLeft: MetalArray<UInt32> = full.left
        XCTAssertEqual(fullLeft.length, 5)                      // 3 pairs, unmatched left row 0, unmatched right row 2
        XCTAssertEqual(full.right.toArray().compactMap { $0 }.sorted(), [0, 0, 1, 2])

        let gk = try GroupByKeys(columns: [.int64(a)])
        let reps: MetalArray<UInt32> = try gk.representativeRows()
        XCTAssertEqual(reps.length, gk.groupCount)
        XCTAssertTrue(Set<UInt32>([0, 2, 3, 4]).isSubset(of: Set(reps.toRawArray())))   // the lowest row of 5, 3, 9, 1
        let rows: MetalArray<UInt32> = try GroupByKeys.rowIndices(4, .shared)
        XCTAssertEqual(rows.toRawArray(), [0, 1, 2, 3])
    }

    /// The outputs that are not row numbers into the input keep their types: indices_nonzero is uint64
    /// as in Arrow, dictionary codes and group ids are int32 (a group-by holds at most 2^30 groups), and
    /// inverse_permutation is Arrow's signed int32.
    func testOtherIntegerOutputsKeepTheirTypes() throws {
        try requireRealGPU()
        let a = try MetalArray<Int64>([5, 0, 3, 0])
        let nz: MetalArray<UInt64> = try a.compare(.ne, 0).indicesNonzero()
        XCTAssertEqual(nz.toRawArray(), [0, 2])
        let (codes, _): (MetalArray<Int32>, MetalArray<Int64>) = try a.dictionaryEncode()
        XCTAssertEqual(codes.toArray(), [2, 0, 1, 0])
        let ids: MetalArray<Int32> = try GroupByKeys(columns: [.int64(a)]).ids
        XCTAssertEqual(ids.length, 4)
        let inv: MetalArray<Int32> = try MetalArray<UInt32>([2, 0, 1]).inversePermutation()
        XCTAssertEqual(inv.toArray(), [1, 2, 0])
    }

    // MARK: - take keeps every index type

    func testTakeAcceptsInt32Int64AndUInt32Indices() throws {
        try requireRealGPU()
        let nums = try MetalArray<Int64>([10, nil, 30, 40, 50])
        let strs = try MetalStringArray(["a", "bb", nil, "dddd", "e"])
        let bools = try MetalBooleanArray([true, false, true, false, true])
        let list = try MetalListArray(counts: [1, 2, nil, 0, 3], values: .int64(try MetalArray<Int64>([1, 2, 3, 4, 5, 6])))
        let i32 = try MetalArray<Int32>([4, 1, 0, nil, 3])
        let i64 = try MetalArray<Int64>([4, 1, 0, nil, 3])
        let u32 = try MetalArray<UInt32>([4, 1, 0, nil, 3])

        func check<I: ArrowIndex>(_ idx: MetalArray<I>, _ name: String) throws {
            XCTAssertEqual(try nums.take(idx).toArray(), [50, nil, 10, nil, 40], name)
            XCTAssertEqual(try strs.take(idx).toArray(), ["e", "bb", "a", nil, "dddd"], name)
            XCTAssertEqual(try bools.take(idx).toArray(), [true, false, true, nil, false], name)
            XCTAssertEqual(try list.take(idx).listValueLength().toArray(), [3, 2, 1, nil, 0], name)
            let batch = try MetalRecordBatch(names: ["n", "s"], columns: [.int64(nums), .string(strs)])
            XCTAssertEqual(try batch.take(idx)["n"]!.asInt64!.toArray(), [50, nil, 10, nil, 40], name)
        }
        try check(i32, "int32")
        try check(i64, "int64")
        try check(u32, "uint32")
        // The index arrays this package returns go straight back into take.
        XCTAssertEqual(try nums.take(try nums.argsort()).toArray(), [10, 30, 40, 50, nil])
    }

    /// An index outside the array is an error in every index type, never another row: a negative int32,
    /// a uint32 at the length, and an int64 that a 32-bit narrowing would wrap onto a valid row.
    func testTakeRefusesOutOfRangeIndicesOfEveryType() throws {
        try requireRealGPU()
        let nums = try MetalArray<Int64>([10, 20, 30, 40, 50])
        let strs = try MetalStringArray(["a", "bb", "c", "dddd", "e"])
        let list = try MetalListArray(counts: [1, 2, 0, 0, 3], values: .int64(try MetalArray<Int64>([1, 2, 3, 4, 5, 6])))
        let wraps = Int64(1) << 32 | 1                          // 2^32 + 1 narrows to row 1
        let bad: [(String, AnyMetalArray)] = [
            ("int32 -1", .int32(try MetalArray<Int32>([0, -1]))),
            ("uint32 5", .uint32(try MetalArray<UInt32>([0, 5]))),
            ("int64 2^32 + 1", .int64(try MetalArray<Int64>([0, wraps]))),
        ]
        for (name, idx) in bad {
            func take(_ c: AnyMetalArray) throws {
                switch idx {
                case .int32(let i): _ = try c.take(i)
                case .uint32(let i): _ = try c.take(i)
                case .int64(let i): _ = try c.take(i)
                default: XCTFail("index type")
                }
                try MetalContext.shared.syncPoint()
            }
            XCTAssertThrowsError(try take(.int64(nums)), "primitive \(name)")
            XCTAssertThrowsError(try take(.string(strs)), "utf8 \(name)")
            XCTAssertThrowsError(try take(.list(list)), "list \(name)")
        }
    }

    // MARK: - the refusal past the index type

    /// With the largest row number lowered to 99, a 100-row input is still answered and a 101-row input
    /// is refused by every index-returning call, with the message naming the row that does not fit.
    func testIndexProducersRefuseRowsPastTheLimit() throws {
        try requireRealGPU()
        let saved = Dispatch.maxRowIndex
        Dispatch.maxRowIndex = 99
        defer { Dispatch.maxRowIndex = saved }

        let fits = try MetalArray<Int32>((0..<100).map { Int32(99 - $0) })
        XCTAssertEqual(try fits.argsort().toRawArray().first, 99)
        XCTAssertEqual(try fits.topK(1, largest: false).toRawArray(), [99])

        let n = 101
        let big = try MetalArray<Int32>((0..<n).map { Int32($0 % 7) })
        let bigF = try MetalArray<Double>((0..<n).map { Double($0 % 5) })
        let strs = try MetalStringArray((0..<n).map { "s\($0 % 9)" })
        func refuses(_ what: String, _ body: () throws -> Any) {
            XCTAssertThrowsError(try body(), what) { e in
                XCTAssertTrue("\(e)".contains("row 100 does not fit the UInt32 index type (row numbers go up to 99)"),
                              "\(what): \(e)")
            }
        }
        refuses("argsort") { try big.argsort() }
        refuses("argsort float64") { try bigF.argsort(floatOrder: .total) }
        refuses("top_k") { try big.topK(3) }
        refuses("partition_nth_indices") { try big.partitionNthIndices(5) }
        refuses("lexsort") { try lexsortIndices([.int32(big), .float64(bigF)]) }
        refuses("utf8 argsort") { try strs.argsort() }
        refuses("row_number") { try big.rowNumber() }
        refuses("rank") { try big.rank(tiebreaker: .dense) }
        refuses("row indices") { try GroupByKeys.rowIndices(n, .shared) }
        refuses("representative rows") { try GroupByKeys(columns: [.int32(big)]).representativeRows() }
        refuses("first") { try GroupBy(keys: big, keyCount: 7).first(bigF) }
        refuses("batch sort") {
            try MetalRecordBatch(names: ["k", "v"], columns: [.int32(big), .float64(bigF)]).sorted(by: [("k", false), ("v", true)])
        }
        let sources = ["t": PlanSource(name: "t", batch: try MetalRecordBatch(names: ["k"], columns: [.int32(big)]))]
        refuses("plan window") {
            try PlanJSON.run(#"{"op":"window","input":{"op":"scan","source":"t"},"specs":[{"name":"rn","fn":"row_number","order_by":[["k",false]]}]}"#,
                             sources: sources)
        }
    }

    /// The engine's window ranks are UInt32 like the kernels'.
    func testPlanWindowRanksAreUInt32() throws {
        try requireRealGPU()
        let k = try MetalArray<Int32>([3, 1, 2, 1])
        let sources = ["t": PlanSource(name: "t", batch: try MetalRecordBatch(names: ["k"], columns: [.int32(k)]))]
        let out = try PlanJSON.run(#"{"op":"window","input":{"op":"scan","source":"t"},"specs":[{"name":"rn","fn":"row_number","order_by":[["k",false]]},{"name":"rk","fn":"rank","order_by":[["k",false]]},{"name":"dr","fn":"dense_rank","order_by":[["k",false]]}]}"#,
                                   sources: sources)
        XCTAssertEqual(out["rn"]?.asUInt32?.toRawArray(), [4, 1, 3, 2])
        XCTAssertEqual(out["rk"]?.asUInt32?.toRawArray(), [4, 1, 3, 1])
        XCTAssertEqual(out["dr"]?.asUInt32?.toRawArray(), [3, 1, 2, 1])
    }

    // MARK: - past 2^31 rows

    /// A row number of 2^31 or more round-trips as itself. The input is 2^31 + 2^20 UInt8 rows mapped
    /// from one 1 GiB period (`IndexWrapTests.periodic`): 255 sits at row `o` of every period and nowhere
    /// else, so the three largest values are rows `o`, `o + 2^30` and `o + 2^31`. Needs
    /// `ARROWMETAL_BIG_TESTS=1`.
    func testRowNumbersPast2To31ComeBackPositive() throws {
        try requireRealGPU()
        guard ProcessInfo.processInfo.environment["ARROWMETAL_BIG_TESTS"] == "1" else {
            throw XCTSkip("set ARROWMETAL_BIG_TESTS=1 to run over 2^31 + 2^20 rows")
        }
        let n = (1 << 31) + (1 << 20)
        let o = 12_345
        let (a, pe) = try IndexWrapTests.periodic(UInt8.self, n: n) { i in i == o ? 255 : UInt8(i % 200) }
        XCTAssertEqual(pe, 1 << 30)
        func value(_ r: Int) -> UInt8 { r % pe == o ? 255 : UInt8((r % pe) % 200) }
        let want: [UInt32] = [UInt32(o), UInt32(o + pe), UInt32(o + 2 * pe)]
        XCTAssertGreaterThan(want[2], UInt32(Int32.max))

        // top_k: the three 255s, ties in row order.
        XCTAssertEqual(try a.topK(3, largest: true).toRawArray(), want)

        // take with UInt32 and Int64 row numbers past 2^31.
        let probe = [o + 2 * pe, (1 << 31) + 5, n - 1, 7]
        let wantValues = probe.map(value)
        XCTAssertEqual(try a.take(try MetalArray<UInt32>(probe.map(UInt32.init))).toRawArray(), wantValues)
        XCTAssertEqual(try a.take(try MetalArray<Int64>(probe.map(Int64.init))).toRawArray(), wantValues)

        // indices_nonzero (UInt64) over the same rows.
        XCTAssertEqual(try a.compare(.eq, 255).indicesNonzero().toRawArray(), want.map(UInt64.init))

        // The group-by row numbers: row 2^31 + 5 is 2^31 + 5.
        let rows = try GroupByKeys.rowIndices(n, .shared)
        try MetalContext.shared.syncPoint()
        withExtendedLifetime(rows) {
            let p = rows.values.typed(UInt32.self)
            for r in [0, Int(Int32.max), (1 << 31) + 5, n - 1] { XCTAssertEqual(Int(p[r]), r) }
        }
    }
}
