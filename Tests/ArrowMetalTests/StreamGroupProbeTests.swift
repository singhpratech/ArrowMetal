import XCTest
@testable import ArrowMetal

/// The row-level resident group-by: one thread per row into the GPU-resident table, no per-batch
/// dense encoding at all (`docs/STREAMING.md` §4.1).
///
/// Every test here is differential. The same batches go through four implementations —
///
/// * `host` — the host dictionary, which is the oracle for every key type and every aggregate;
/// * `distinct` — the per-batch dense encoding folded into the resident table (the path this work
///   replaces for sparse keys);
/// * `rowAtomic` — one thread per row, 64-bit atomic adds out of two 32-bit ones;
/// * `rowDense` — one thread per row to claim the slot, then a batch-local dense id so the ordinary
///   `GroupBy` can run a correctly-rounded float64 sum;
///
/// — and the results have to agree row for row. Integer aggregates agree *bit for bit* on all four,
/// which is the whole claim of the atomic path: integer addition commutes, so the order the rows
/// reach a slot in cannot change the answer.
final class StreamGroupProbeTests: XCTestCase {

    enum Path { case host, distinct, rowAtomic, rowDense }

    struct Row {
        var key: Int64?
        var amount: Double?
        var qty: Int32?
    }

    // MARK: - Building batches

    static func batch(_ rows: [Row], keyType: String = "int64") throws -> MetalRecordBatch {
        let k64 = try MetalArray<Int64>(rows.map { $0.key })
        let key: AnyMetalArray
        switch keyType {
        case "int32": key = .int32(try k64.cast(to: Int32.self))
        case "uint32": key = .uint32(try k64.cast(to: UInt32.self))
        default: key = .int64(k64)
        }
        return try MetalRecordBatch(names: ["key", "amount", "qty"], columns: [
            key,
            .float64(try MetalArray<Double>(rows.map { $0.amount })),
            .int64(try MetalArray<Int64>(rows.map { $0.qty.map { Int64($0) } })),
        ])
    }

    /// Ragged batches with an empty one in the middle, so batch boundaries are never uniform.
    static func batches(_ rows: [Row], batchRows: Int, keyType: String = "int64") throws -> [MetalRecordBatch] {
        var out: [MetalRecordBatch] = []
        var i = 0, k = 0
        while i < rows.count {
            if k == 2 { out.append(try batch([], keyType: keyType)); k += 1; continue }
            let take = Swift.min(rows.count - i, k % 3 == 1 ? Swift.max(1, batchRows / 2) : batchRows)
            out.append(try batch(Array(rows[i..<(i + take)]), keyType: keyType))
            i += take
            k += 1
        }
        if out.isEmpty { out.append(try batch([], keyType: keyType)) }
        return out
    }

    /// `n` rows over `keys` distinct values, with duplicates inside every batch, nulls in the key
    /// (their own group) and nulls in both value columns (so some groups are all-null).
    static func rows(_ n: Int, keys: Int, nullKeyEvery: Int = 0, allNullFrom: Int = Int.max) -> [Row] {
        (0..<n).map { i in
            let k: Int64? = (nullKeyEvery > 0 && i % nullKeyEvery == 3)
                ? nil : Int64((i &* 2_654_435_761) % keys)
            let dead = (k ?? 0) >= Int64(allNullFrom)
            return Row(key: k,
                       amount: (dead || i % 11 == 4) ? nil : Double((i &* 48_271) % 10_007) / 7.0,
                       qty: (dead || i % 17 == 9) ? nil : Int32(i % 97))
        }
    }

    // MARK: - Running one path

    func run(_ bs: [MetalRecordBatch], _ aggs: [StreamAggregate], _ path: Path) throws -> MetalRecordBatch {
        let op = StreamGroupByOperator(keys: ["key"], aggregates: aggs)
        switch path {
        case .host: op.residentTable = false
        case .distinct: op.residentRowLevel = false
        case .rowAtomic: op.residentRowLevelForced = true
        case .rowDense: op.residentRowLevelForced = true; op.residentRowLevelAtomic = false
        }
        let r = try StreamingExecutor(source: ChunkedTableSource(bs)).run(op)
        return try XCTUnwrap(r.batch, "\(path) produced no batch")
    }

    /// Every column of two results, bit for bit (float64 compared by bit pattern, nulls included).
    func expectSame(_ got: MetalRecordBatch, _ want: MetalRecordBatch, _ what: String) throws {
        XCTAssertEqual(got.names, want.names, what)
        XCTAssertEqual(got.length, want.length, "row count: \(what)")
        guard got.length == want.length else { return }
        for (i, name) in want.names.enumerated() {
            let a = got.columns[i], b = want.columns[i]
            XCTAssertEqual(a.arrowFormat, b.arrowFormat, "\(name) type: \(what)")
            if let x = a.asFloat64, let y = b.asFloat64 {
                let xs = x.toArray(), ys = y.toArray()
                for j in 0..<xs.count {
                    XCTAssertEqual(xs[j]?.bitPattern, ys[j]?.bitPattern, "\(name)[\(j)]: \(what)")
                }
            } else if let x = a.asInt64, let y = b.asInt64 {
                XCTAssertEqual(x.toArray(), y.toArray(), "\(name): \(what)")
            } else {
                XCTAssertEqual(try a.streamValues().map { String(describing: $0) },
                               try b.streamValues().map { String(describing: $0) }, "\(name): \(what)")
            }
        }
    }

    /// The integer-only aggregate set, which every path computes bit for bit.
    static let integerAggs = [StreamAggregate(.count, nil, name: "n"),
                              StreamAggregate(.count, "qty", name: "cq"),
                              StreamAggregate(.sum, "qty", name: "sq"),
                              StreamAggregate(.mean, "qty", name: "mq")]
    /// With a float64 sum in it, which the atomic path declines and the dense one rounds like the
    /// per-batch path does.
    static let floatAggs = [StreamAggregate(.sum, "amount", name: "s"),
                            StreamAggregate(.count, nil, name: "n"),
                            StreamAggregate(.sum, "qty", name: "sq"),
                            StreamAggregate(.mean, "amount", name: "avg")]

    // MARK: - Cardinalities

    /// Cardinality 1, 1k, 100k and 2M with duplicates in every batch: the atomic row path against the
    /// host oracle and against the per-batch encoding, bit for bit on every integer aggregate.
    func testRowAtomicPathAcrossCardinalities() throws {
        try requireRealGPU()
        for (n, keys, batchRows) in [(1, 1, 64), (4_000, 1, 700), (50_000, 1_000, 900),
                                     (300_000, 100_000, 20_000), (2_400_000, 2_000_000, 300_000)] {
            let bs = try Self.batches(Self.rows(n, keys: keys), batchRows: batchRows)
            let what = "n=\(n) keys=\(keys)"
            let atomic = try run(bs, Self.integerAggs, .rowAtomic)
            try expectSame(atomic, try run(bs, Self.integerAggs, .host), "row-atomic vs host, \(what)")
            try expectSame(atomic, try run(bs, Self.integerAggs, .distinct), "row-atomic vs distinct, \(what)")
        }
    }

    /// The same shapes through the dense-id branch of the row path, which is what a float64 sum
    /// takes. Its per-group sums come out of the same correctly-rounded adder in the same row order
    /// as the per-batch path's, so even the float column matches bit for bit.
    func testRowDensePathAcrossCardinalities() throws {
        try requireRealGPU()
        for (n, keys, batchRows) in [(1, 1, 64), (50_000, 1_000, 900), (300_000, 100_000, 20_000)] {
            let bs = try Self.batches(Self.rows(n, keys: keys), batchRows: batchRows)
            let what = "n=\(n) keys=\(keys)"
            let dense = try run(bs, Self.floatAggs, .rowDense)
            try expectSame(dense, try run(bs, Self.floatAggs, .distinct), "row-dense vs distinct, \(what)")
            // The host table adds with Swift's `+` in the same order, so it agrees to the last bit too.
            try expectSame(dense, try run(bs, Self.floatAggs, .host), "row-dense vs host, \(what)")
        }
    }

    /// A query with a float64 sum picks the dense branch on its own; an integer-only one picks the
    /// atomic branch. Both still answer like the host table.
    func testTheChooserPicksABranchAndBothAnswerTheSame() throws {
        try requireRealGPU()
        let bs = try Self.batches(Self.rows(120_000, keys: 90_000, nullKeyEvery: 13), batchRows: 4_000)
        let mixed = Self.floatAggs + [StreamAggregate(.count, "qty", name: "cq")]
        try expectSame(try run(bs, mixed, .rowAtomic), try run(bs, mixed, .host), "mixed aggregates")
        try expectSame(try run(bs, Self.integerAggs, .rowAtomic),
                       try run(bs, Self.integerAggs, .host), "integer aggregates")
    }

    // MARK: - Nulls

    /// Null keys (their own group), all-null value groups (a null `sum` and a zero count), and a
    /// batch whose keys are *all* null.
    func testNullKeysAndAllNullGroups() throws {
        try requireRealGPU()
        var rows = Self.rows(80_000, keys: 40_000, nullKeyEvery: 7, allNullFrom: 30_000)
        rows.append(contentsOf: (0..<500).map { _ in Row(key: nil, amount: nil, qty: nil) })
        let bs = try Self.batches(rows, batchRows: 3_000)
        for aggs in [Self.integerAggs, Self.floatAggs] {
            let host = try run(bs, aggs, .host)
            try expectSame(try run(bs, aggs, .rowAtomic), host, "nulls, atomic")
            try expectSame(try run(bs, aggs, .rowDense), host, "nulls, dense")
            try expectSame(try run(bs, aggs, .distinct), host, "nulls, distinct")
        }
    }

    /// Every row's key null, so the answer is the single null group.
    func testEveryKeyNull() throws {
        try requireRealGPU()
        let rows = (0..<5_000).map { i in Row(key: nil, amount: Double(i), qty: Int32(i % 5)) }
        let bs = try Self.batches(rows, batchRows: 800)
        let got = try run(bs, Self.floatAggs, .rowAtomic)
        XCTAssertEqual(got.length, 1)
        XCTAssertNil(try XCTUnwrap(got["key"]?.asInt64).toArray()[0])
        try expectSame(got, try run(bs, Self.floatAggs, .host), "all-null keys")
    }

    /// Only empty batches: the operator has nothing to answer with and must not invent a group.
    func testOnlyEmptyBatches() throws {
        try requireRealGPU()
        let bs = try [Self.batch([]), Self.batch([]), Self.batch([])]
        let got = try run(bs, Self.integerAggs, .rowAtomic)
        XCTAssertEqual(got.length, 0)
    }

    // MARK: - Growth

    /// Growth across the rehash threshold, several times over: the table starts at 2^16 slots and
    /// grows at a load factor of 0.55, so a quarter of a million distinct keys rehashes it twice.
    /// Nothing may be lost, duplicated or moved to the wrong slot.
    func testGrowthAcrossTheRehashThreshold() throws {
        try requireRealGPU()
        let n = 250_000
        let rows: [Row] = (0..<n).map { (i: Int) -> Row in
            Row(key: Int64(i &* 7 &+ 3), amount: Double(i), qty: Int32(i % 13))
        }
        // Small batches, so the table grows in the middle of the scan rather than once up front.
        let bs = try Self.batches(rows, batchRows: 5_000)
        let got = try run(bs, Self.integerAggs, .rowAtomic)
        XCTAssertEqual(got.length, n)
        let keys = try XCTUnwrap(got["key"]?.asInt64).toArray()
        for i in 0..<n { XCTAssertEqual(keys[i], Int64(i &* 7 &+ 3)) }
        let counts = try XCTUnwrap(got["n"]?.asInt64).toArray()
        XCTAssertEqual(counts.compactMap { $0 }.reduce(0, +), Int64(n))
        try expectSame(got, try run(bs, Self.integerAggs, .host), "growth")
        try expectSame(try run(bs, Self.floatAggs, .rowDense),
                       try run(bs, Self.floatAggs, .host), "growth, dense")
    }

    // MARK: - Awkward key values

    /// The keys a hash table usually has to reserve. This one reserves none: the slot's three words
    /// each hold `field + 1`, so zero means "not written" and every one of the 2^64 keys is legal.
    func testExtremeKeyValues() throws {
        try requireRealGPU()
        var special: [Int64] = [0, -1, 1, -2, Int64.min, Int64.max]
        special.append(Int64.min + 1)
        special.append(Int64.max - 1)
        special.append(0x3F_FFFF)
        special.append(0x40_0000)
        special.append(0xFFFF_FFFF)
        special.append(-0xFFFF_FFFF)
        special.append(Int64(bitPattern: 0x8000_0000_0000_0000))
        special.append(Int64(bitPattern: 0xFFFF_FFFF_0000_0000))
        special.append(Int64(bitPattern: 0x0000_0000_FFFF_FFFF))
        special = Array(Set(special)).sorted()
        var rows: [Row] = []
        for (i, k) in special.enumerated() {
            for r in 0..<7 { rows.append(Row(key: k, amount: Double(i * 10 + r), qty: Int32(r))) }
        }
        rows.append(Row(key: nil, amount: 1, qty: 1))
        let bs = try Self.batches(rows, batchRows: 9)
        let got = try run(bs, Self.floatAggs, .rowAtomic)
        XCTAssertEqual(got.length, special.count + 1)
        try expectSame(got, try run(bs, Self.floatAggs, .host), "extreme keys")
        try expectSame(try run(bs, Self.integerAggs, .rowDense),
                       try run(bs, Self.integerAggs, .host), "extreme keys, dense")
    }

    /// Both key widths the sparse path is likely to see, and an unsigned one, back in their own type.
    func testKeyWidths() throws {
        try requireRealGPU()
        for keyType in ["int32", "int64", "uint32"] {
            let rows = Self.rows(120_000, keys: 70_000, nullKeyEvery: 19)
            let bs = try Self.batches(rows, batchRows: 6_000, keyType: keyType)
            let host = try run(bs, Self.floatAggs, .host)
            XCTAssertEqual(host["key"]?.arrowFormat, try Self.batch([], keyType: keyType)["key"]?.arrowFormat)
            try expectSame(try run(bs, Self.floatAggs, .rowAtomic), host, "\(keyType) atomic")
            try expectSame(try run(bs, Self.floatAggs, .rowDense), host, "\(keyType) dense")
        }
    }

    /// The float64 sum on the row path is a one-thread-per-group emulation of the segmented
    /// reduction, so it has to reproduce that reduction's *order*, not merely its value. These shapes
    /// are the ones where an order slip would show: runs of one, of two, of three (where the
    /// butterfly pairs lanes 0 and 2 before lane 1), runs with a null in the middle (an empty lane is
    /// replaced, not added to), and a skewed key whose hot group runs past the 256 lanes and folds
    /// several rows into each of them.
    func testFloatSumOrderMatchesTheSegmentedReduction() throws {
        try requireRealGPU()
        for runLength in [1, 2, 3, 4, 5, 7, 8, 9, 17, 33, 64] {
            var rows: [Row] = []
            for g in 0..<3_000 {
                for r in 0..<runLength {
                    // Values chosen so the order of the additions changes the last bits: a large one
                    // first and small ones after it round away, in the other order they do not.
                    let v: Double? = (r % 5 == 3) ? nil
                        : (r == 0 ? 1e16 : Double(r) * 0.1 + Double(g % 7))
                    rows.append(Row(key: Int64(g &* 8_675_309), amount: v, qty: Int32(r)))
                }
            }
            let bs = try Self.batches(rows, batchRows: 3_000 * runLength)
            try expectSame(try run(bs, Self.floatAggs, .rowDense),
                           try run(bs, Self.floatAggs, .distinct), "run length \(runLength)")
        }
        // Skew: one group far past 256 rows alongside thousands of singletons, so the run is folded
        // lane by lane and only then reduced.
        var rows: [Row] = []
        for r in 0..<4_000 { rows.append(Row(key: 42, amount: r == 0 ? 1e16 : Double(r) * 0.5, qty: 1)) }
        for g in 1..<60_000 { rows.append(Row(key: Int64(g &* 1_000_003), amount: Double(g) * 0.25, qty: 1)) }
        let bs = try Self.batches(rows, batchRows: 64_000)
        try expectSame(try run(bs, Self.floatAggs, .rowDense),
                       try run(bs, Self.floatAggs, .distinct), "skewed run")
    }

    /// `count(col)` over columns the atomic accumulate cannot widen — a nullable string, a float64 —
    /// still has to count Arrow's way. A column with no nulls counts every row; one with nulls that
    /// does not widen sends the batch down the dense branch instead.
    func testCountOverColumnsTheAtomicPathCannotWiden() throws {
        try requireRealGPU()
        var batches: [MetalRecordBatch] = []
        for b in 0..<12 {
            let n = 4_000
            var keys: [Int64?] = [], amounts: [Double?] = [], labels: [String?] = [], solid: [Double?] = []
            for i in 0..<n {
                let g = b * n + i
                keys.append(g % 37 == 5 ? nil : Int64(g &* 2_654_435_761))
                amounts.append(g % 5 == 1 ? nil : Double(g) * 0.5)
                labels.append(g % 3 == 2 ? nil : "label-\(g % 97)")
                solid.append(Double(g))                       // never null
            }
            batches.append(try MetalRecordBatch(names: ["key", "amount", "label", "solid"], columns: [
                .int64(try MetalArray<Int64>(keys)),
                .float64(try MetalArray<Double>(amounts)),
                .string(try MetalStringArray(labels)),
                .float64(try MetalArray<Double>(solid)),
            ]))
        }
        let aggs = [StreamAggregate(.count, "amount", name: "ca"),
                    StreamAggregate(.count, "label", name: "cl"),
                    StreamAggregate(.count, "solid", name: "cs"),
                    StreamAggregate(.count, nil, name: "n")]
        let host = try run(batches, aggs, .host)
        try expectSame(try run(batches, aggs, .rowAtomic), host, "count over unwidenable columns")
        try expectSame(try run(batches, aggs, .rowDense), host, "count over unwidenable columns, dense")
        // The same aggregates over a column with no nulls at all take the atomic branch.
        let solidOnly = [StreamAggregate(.count, "solid", name: "cs"), StreamAggregate(.count, nil, name: "n")]
        try expectSame(try run(batches, solidOnly, .rowAtomic),
                       try run(batches, solidOnly, .host), "count over a column with no nulls")
    }

    // MARK: - The chooser

    /// A dense key column keeps the per-batch encoding; a sparse one takes the row path. Both are
    /// only an internal choice, so the check is that the answers still match.
    func testSparseKeyTakesTheRowPathAndDenseKeepsTheEncoding() throws {
        try requireRealGPU()
        // 1000 contiguous values over 200k rows: the range scan is far cheaper than a million atomics.
        let dense = (0..<200_000).map { Row(key: Int64($0 % 1_000), amount: Double($0), qty: Int32($0 % 7)) }
        // 200k values spread over the whole int64 range: nothing to scan, so the rows go straight in.
        let sparse = (0..<200_000).map { i in
            Row(key: Int64(bitPattern: UInt64(i) &* 0x9E37_79B9_7F4A_7C15),
                amount: Double(i), qty: Int32(i % 7))
        }
        for rows in [dense, sparse] {
            let bs = try Self.batches(rows, batchRows: 20_000)
            try expectSame(try run(bs, Self.floatAggs, .rowAtomic),
                           try run(bs, Self.floatAggs, .host), "chooser")
        }
    }
}
