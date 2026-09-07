import XCTest
@testable import ArrowMetal

/// Fused broadcast join + aggregate, checked against the in-memory join followed by the in-memory
/// aggregate — the same two operators the fusion replaces, run over the same rows.
///
/// Integer aggregates must agree bit for bit. Float64 sums are summed in a different order (the
/// fused kernel reduces per threadgroup, the oracle reduces over one materialised column), so they
/// are compared within one ulp per element.
final class StreamJoinFusionTests: XCTestCase {

    // MARK: - fixtures

    /// A probe side split into ragged batches, including an empty one.
    private func probeSource(_ batches: [MetalRecordBatch]) -> BatchSource {
        ChunkedTableSource(batches)
    }

    /// `n` probe rows: an int32 and an int64 key with the same value, a float64 and an int64 value.
    /// `nullKeyEvery > 0` makes every k-th key null; `keyMod` bounds the key space.
    private func probeBatch(_ n: Int, offset: Int = 0, keyMod: Int = 40, nullKeyEvery: Int = 0,
                            nullValueEvery: Int = 0) throws -> MetalRecordBatch {
        var k32: [Int32?] = [], k64: [Int64?] = [], amount: [Double] = [], qty: [Int64?] = []
        for j in 0..<n {
            let i = j + offset
            let isNullKey = nullKeyEvery > 0 && i % nullKeyEvery == 0
            let key = Int64((i &* 2654435761) % keyMod)
            k32.append(isNullKey ? nil : Int32(key))
            k64.append(isNullKey ? nil : key)
            amount.append(Double((i &* 48271) % 10_007) / 7.0)
            qty.append(nullValueEvery > 0 && i % nullValueEvery == 0 ? nil : Int64((i &* 7919) % 997))
        }
        return try MetalRecordBatch(names: ["k32", "k64", "amount", "qty"], columns: [
            .int32(try MetalArray<Int32>(k32)),
            .int64(try MetalArray<Int64>(k64)),
            .float64(try MetalArray<Double>(amount)),
            .int64(try MetalArray<Int64>(qty)),
        ])
    }

    /// A build side whose keys are `keys` (duplicates allowed, nil for a null key), with a float64
    /// `weight`, an int64 `bonus` and an int32 `bucket` to group on.
    private func buildBatch(_ keys: [Int64?]) throws -> MetalRecordBatch {
        try MetalRecordBatch(names: ["b32", "b64", "weight", "bonus", "bucket"], columns: [
            .int32(try MetalArray<Int32>(keys.map { $0.map { Int32($0) } })),
            .int64(try MetalArray<Int64>(keys)),
            .float64(try MetalArray<Double>(keys.enumerated().map { i, _ in Double(i % 37) / 3.0 + 0.5 })),
            .int64(try MetalArray<Int64>(keys.enumerated().map { i, _ in Int64(i % 13) })),
            .int32(try MetalArray<Int32>(keys.enumerated().map { i, _ in Int32(i % 5) })),
        ])
    }

    /// Splits a batch into ragged pieces, one of them empty.
    private func ragged(_ b: MetalRecordBatch, _ size: Int) throws -> [MetalRecordBatch] {
        var out: [MetalRecordBatch] = []
        var i = 0, k = 0
        while i < b.length {
            if k == 2 { out.append(try b.slice(offset: 0, length: 0)); k += 1; continue }
            let take = Swift.min(k % 3 == 1 ? Swift.max(1, size / 2) : size, b.length - i)
            out.append(try b.slice(offset: i, length: take))
            i += take
            k += 1
        }
        if out.isEmpty { out.append(try b.slice(offset: 0, length: 0)) }
        return out
    }

    // MARK: - the oracle

    /// The unfused plan: join every probe batch in memory, concatenate, aggregate the result.
    private func oracle(_ batches: [MetalRecordBatch], _ build: MetalRecordBatch,
                        probeKey: String, buildKey: String, _ specs: [StreamAggregate]) throws -> StreamResult {
        var joined: [MetalRecordBatch] = []
        for b in batches where b.length > 0 {
            joined.append(try b.join(build, on: probeKey, rightKey: buildKey, kind: .inner))
        }
        let source = ChunkedTableSource(joined.isEmpty ? [try build.slice(offset: 0, length: 0)] : joined)
        return try StreamQuery(source: source).aggregate(specs)
    }

    private func fused(_ batches: [MetalRecordBatch], _ build: MetalRecordBatch,
                       probeKey: String, buildKey: String, _ specs: [StreamAggregate]) throws -> StreamResult {
        try StreamQuery(source: probeSource(batches))
            .join(build, on: probeKey, buildKey: buildKey)
            .aggregate(specs)
    }

    /// Integer scalars bit for bit; float64 within one ulp per element folded in.
    private func assertAgrees(_ got: StreamResult, _ want: StreamResult, elements: Int,
                              file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(got.scalarNames, want.scalarNames, file: file, line: line)
        for (i, name) in want.scalarNames.enumerated() {
            let a = got.scalars[i], b = want.scalars[i]
            if b.isNullScalar || a.isNullScalar {
                XCTAssertEqual(a.isNullScalar, b.isNullScalar, "\(name)", file: file, line: line)
                continue
            }
            switch (a, b) {
            case (.int(let x), .int(let y)):
                XCTAssertEqual(x, y, "\(name)", file: file, line: line)          // integers: bit for bit
            case (.uint(let x), .uint(let y)):
                XCTAssertEqual(x, y, "\(name)", file: file, line: line)
            default:
                let x = a.asDouble ?? .nan, y = b.asDouble ?? .nan
                // One ulp per element folded in: the two plans sum in different orders.
                let tol = Swift.max(abs(y).ulp * Double(Swift.max(elements, 1)), 1e-12)
                XCTAssertEqual(x, y, accuracy: tol, "\(name)", file: file, line: line)
            }
        }
    }

    private func check(_ probe: MetalRecordBatch, _ build: MetalRecordBatch, probeKey: String,
                       buildKey: String, _ specs: [StreamAggregate], batchRows: Int = 700,
                       file: StaticString = #filePath, line: UInt = #line) throws {
        let batches = try ragged(probe, batchRows)
        let want = try oracle(batches, build, probeKey: probeKey, buildKey: buildKey, specs)
        let got = try fused(batches, build, probeKey: probeKey, buildKey: buildKey, specs)
        assertAgrees(got, want, elements: probe.length, file: file, line: line)
    }

    private static let allAggregates: [StreamAggregate] = [
        StreamAggregate(.sum, "amount", name: "sum_amount"),        // probe side, float64
        StreamAggregate(.sum, "weight", name: "sum_weight"),        // build side, float64
        StreamAggregate(.sum, "qty", name: "sum_qty"),              // probe side, int64, nullable
        StreamAggregate(.count, nil, name: "rows"),                 // matched pairs
        StreamAggregate(.min, "bonus", name: "min_bonus"),          // build side, int64
        StreamAggregate(.max, "amount", name: "max_amount"),        // probe side, float64
        StreamAggregate(.mean, "qty", name: "mean_qty"),
    ]

    // MARK: - scalar aggregates

    func testFusedJoinSumMatchesJoinThenSumInt64Key() throws {
        try requireRealGPU()
        let probe = try probeBatch(20_000, keyMod: 40, nullValueEvery: 11)
        let build = try buildBatch((0..<40).map { Int64($0) })
        try check(probe, build, probeKey: "k64", buildKey: "b64", Self.allAggregates)
    }

    func testFusedJoinSumMatchesJoinThenSumInt32Key() throws {
        try requireRealGPU()
        let probe = try probeBatch(20_000, keyMod: 40, nullValueEvery: 11)
        let build = try buildBatch((0..<40).map { Int64($0) })
        try check(probe, build, probeKey: "k32", buildKey: "b32", Self.allAggregates)
    }

    func testDuplicateBuildKeysMultiplyRows() throws {
        try requireRealGPU()
        // Each key 0...19 appears three times on the build side, so every matching probe row
        // contributes three times — the count is the pair count, not the probe row count.
        let probe = try probeBatch(9_000, keyMod: 20)
        let build = try buildBatch((0..<60).map { Int64($0 % 20) })
        try check(probe, build, probeKey: "k64", buildKey: "b64", Self.allAggregates)
        let got = try fused(try ragged(probe, 700), build, probeKey: "k64", buildKey: "b64",
                            [StreamAggregate(.count, nil, name: "rows")])
        XCTAssertEqual(got.onlyScalar?.asInt64, Int64(probe.length * 3))
    }

    func testNullKeysNeverMatchOnEitherSide() throws {
        try requireRealGPU()
        // Nulls on the probe side...
        let probe = try probeBatch(12_000, keyMod: 30, nullKeyEvery: 7)
        // ...and on the build side, plus a key that only the build side has.
        var keys: [Int64?] = (0..<30).map { Int64($0) }
        keys.append(nil)
        keys.append(nil)
        keys.append(9_999)
        let build = try buildBatch(keys)
        try check(probe, build, probeKey: "k64", buildKey: "b64", Self.allAggregates)
        try check(probe, build, probeKey: "k32", buildKey: "b32", Self.allAggregates)
    }

    func testBatchWithNoMatchesContributesNothing() throws {
        try requireRealGPU()
        // Two batches: the first matches, the second's keys are all outside the build side.
        let matching = try probeBatch(3_000, keyMod: 20)
        let shifted = try MetalRecordBatch(names: ["k32", "k64", "amount", "qty"], columns: [
            .int32(try MetalArray<Int32>((0..<3_000).map { Int32(10_000 + $0 % 20) })),
            .int64(try MetalArray<Int64>((0..<3_000).map { Int64(10_000 + $0 % 20) })),
            .float64(try MetalArray<Double>((0..<3_000).map { Double($0) / 7.0 })),
            .int64(try MetalArray<Int64>((0..<3_000).map { Int64($0 % 97) })),
        ])
        let build = try buildBatch((0..<20).map { Int64($0) })
        let batches = [matching, try matching.slice(offset: 0, length: 0), shifted]
        let want = try oracle(batches, build, probeKey: "k64", buildKey: "b64", Self.allAggregates)
        let got = try fused(batches, build, probeKey: "k64", buildKey: "b64", Self.allAggregates)
        assertAgrees(got, want, elements: 6_000)
    }

    func testEveryBatchEmpty() throws {
        try requireRealGPU()
        let probe = try probeBatch(16, keyMod: 8)
        let build = try buildBatch((0..<8).map { Int64($0) })
        let batches = [try probe.slice(offset: 0, length: 0), try probe.slice(offset: 0, length: 0)]
        let got = try fused(batches, build, probeKey: "k64", buildKey: "b64", Self.allAggregates)
        XCTAssertEqual(got.scalar("rows")?.asInt64, 0)
        XCTAssertTrue(got.scalar("sum_amount")!.isNullScalar)
        XCTAssertTrue(got.scalar("min_bonus")!.isNullScalar)
        XCTAssertTrue(got.scalar("mean_qty")!.isNullScalar)
    }

    func testBuildSideOfOneRow() throws {
        try requireRealGPU()
        let probe = try probeBatch(10_000, keyMod: 25)
        let build = try buildBatch([7])
        try check(probe, build, probeKey: "k64", buildKey: "b64", Self.allAggregates)
        try check(probe, build, probeKey: "k32", buildKey: "b32", Self.allAggregates)
    }

    func testBuildSideOfAMillionRows() throws {
        try requireRealGPU()
        let build = try buildBatch((0..<1_000_000).map { Int64($0) })
        let probe = try probeBatch(50_000, keyMod: 1_000_000)
        try check(probe, build, probeKey: "k64", buildKey: "b64",
                  [StreamAggregate(.sum, "amount", name: "sum_amount"),
                   StreamAggregate(.sum, "weight", name: "sum_weight"),
                   StreamAggregate(.count, nil, name: "rows")],
                  batchRows: 6_000)
    }

    func testAllValuesNullGivesNull() throws {
        try requireRealGPU()
        let n = 2_000
        let probe = try MetalRecordBatch(names: ["k64", "v"], columns: [
            .int64(try MetalArray<Int64>((0..<n).map { Int64($0 % 10) })),
            .int64(try MetalArray<Int64>([Int64?](repeating: nil, count: n))),
        ])
        let build = try buildBatch((0..<10).map { Int64($0) })
        let got = try fused(try ragged(probe, 300), build, probeKey: "k64", buildKey: "b64",
                            [StreamAggregate(.sum, "v", name: "s"), StreamAggregate(.count, "v", name: "c"),
                             StreamAggregate(.min, "v", name: "mn"), StreamAggregate(.max, "v", name: "mx")])
        XCTAssertTrue(got.scalar("s")!.isNullScalar)
        XCTAssertEqual(got.scalar("c")?.asInt64, 0)
        XCTAssertTrue(got.scalar("mn")!.isNullScalar)
        XCTAssertTrue(got.scalar("mx")!.isNullScalar)
    }

    func testFilterBeforeTheJoinIsApplied() throws {
        try requireRealGPU()
        let probe = try probeBatch(12_000, keyMod: 40)
        let build = try buildBatch((0..<40).map { Int64($0) })
        let batches = try ragged(probe, 700)
        var filtered: [MetalRecordBatch] = []
        for b in batches where b.length > 0 {
            filtered.append(try streamFilterProject(b, filter: col("k64") < 10,
                                                    projections: nil, context: .shared))
        }
        let specs = [StreamAggregate(.sum, "amount", name: "sum_amount"),
                     StreamAggregate(.sum, "weight", name: "sum_weight"),
                     StreamAggregate(.count, nil, name: "rows")]
        let want = try oracle(filtered, build, probeKey: "k64", buildKey: "b64", specs)
        let q = StreamQuery(source: probeSource(batches))
        _ = q.filter(col("k64") < 10)
        let got = try q.join(build, on: "k64", buildKey: "b64").aggregate(specs)
        assertAgrees(got, want, elements: probe.length)
    }

    // MARK: - group-by

    /// The fused group-by against the in-memory join followed by the in-memory group-by.
    private func checkGroup(_ probe: MetalRecordBatch, _ build: MetalRecordBatch, probeKey: String,
                            buildKey: String, keys: [String], _ aggs: [StreamAggregate],
                            file: StaticString = #filePath, line: UInt = #line) throws {
        let batches = try ragged(probe, 700)
        var joined: [MetalRecordBatch] = []
        for b in batches where b.length > 0 {
            joined.append(try b.join(build, on: probeKey, rightKey: buildKey, kind: .inner))
        }
        let want = try StreamQuery(source: ChunkedTableSource(joined)).groupBy(keys, aggs)
        let got = try StreamQuery(source: probeSource(batches))
            .join(build, on: probeKey, buildKey: buildKey)
            .groupBy(keys, aggs)
        let a = try XCTUnwrap(got.batch, file: file, line: line)
        let b = try XCTUnwrap(want.batch, file: file, line: line)
        XCTAssertEqual(a.length, b.length, "group count", file: file, line: line)
        XCTAssertEqual(a.names, b.names, file: file, line: line)
        for (i, name) in b.names.enumerated() {
            let x = a.columns[i], y = b.columns[i]
            if let xi = x.asInt64, let yi = y.asInt64 {
                XCTAssertEqual(xi.toArray(), yi.toArray(), name, file: file, line: line)
            } else if let xi = x.asInt32, let yi = y.asInt32 {
                XCTAssertEqual(xi.toArray(), yi.toArray(), name, file: file, line: line)
            } else if case .float64(let xd) = x, case .float64(let yd) = y {
                let xs = xd.toArray(), ys = yd.toArray()
                XCTAssertEqual(xs.count, ys.count, name, file: file, line: line)
                for j in 0..<Swift.min(xs.count, ys.count) {
                    if xs[j] == nil || ys[j] == nil { XCTAssertEqual(xs[j], ys[j], name, file: file, line: line); continue }
                    XCTAssertEqual(xs[j]!, ys[j]!, accuracy: Swift.max(abs(ys[j]!).ulp * 4096, 1e-9),
                                   "\(name)[\(j)]", file: file, line: line)
                }
            } else {
                XCTFail("uncompared column \(name)", file: file, line: line)
            }
        }
    }

    func testFusedGroupByProbeSideKey() throws {
        try requireRealGPU()
        let probe = try probeBatch(12_000, keyMod: 20, nullValueEvery: 11)
        let build = try buildBatch((0..<20).map { Int64($0) })
        try checkGroup(probe, build, probeKey: "k64", buildKey: "b64", keys: ["k64"],
                       [StreamAggregate(.sum, "amount", name: "total"),
                        StreamAggregate(.count, nil, name: "n"),
                        StreamAggregate(.sum, "weight", name: "w")])
    }

    func testFusedGroupByBuildSideKey() throws {
        try requireRealGPU()
        // `bucket` is a build-side column with five values; each build key maps to one of them.
        let probe = try probeBatch(12_000, keyMod: 20, nullValueEvery: 11)
        let build = try buildBatch((0..<20).map { Int64($0) })
        try checkGroup(probe, build, probeKey: "k64", buildKey: "b64", keys: ["bucket"],
                       [StreamAggregate(.sum, "amount", name: "total"),
                        StreamAggregate(.count, nil, name: "n"),
                        StreamAggregate(.min, "bonus", name: "mn")])
    }

    func testFusedGroupByDuplicateBuildKeysAndNulls() throws {
        try requireRealGPU()
        var keys: [Int64?] = (0..<15).map { Int64($0 % 5) }
        keys.append(nil)
        let build = try buildBatch(keys)
        let probe = try probeBatch(9_000, keyMod: 8, nullKeyEvery: 5)
        try checkGroup(probe, build, probeKey: "k32", buildKey: "b32", keys: ["k32"],
                       [StreamAggregate(.sum, "amount", name: "total"),
                        StreamAggregate(.count, nil, name: "n")])
    }

    func testFusedGroupByInt32Key() throws {
        try requireRealGPU()
        let probe = try probeBatch(9_000, keyMod: 12)
        let build = try buildBatch((0..<12).map { Int64($0) })
        try checkGroup(probe, build, probeKey: "k32", buildKey: "b32", keys: ["bucket"],
                       [StreamAggregate(.sum, "weight", name: "w"),
                        StreamAggregate(.count, nil, name: "n")])
    }

    // MARK: - what is refused

    func testLeftJoinPlusAggregateRaisesRatherThanAnswering() throws {
        try requireRealGPU()
        let probe = try probeBatch(100, keyMod: 8)
        let build = try buildBatch((0..<8).map { Int64($0) })
        XCTAssertThrowsError(try StreamQuery(source: probeSource([probe]))
            .join(build, on: "k64", buildKey: "b64", kind: .left)
            .sum("amount")) { error in
            XCTAssertTrue("\(error)".contains("inner"), "\(error)")
        }
        XCTAssertThrowsError(try StreamQuery(source: probeSource([probe]))
            .join(build, on: "k64", buildKey: "b64", kind: .left)
            .groupBy(["k64"], [StreamAggregate(.count, nil, name: "n")])) { error in
            XCTAssertTrue("\(error)".contains("inner"), "\(error)")
        }
    }

    func testUnsupportedAggregatesRaise() throws {
        try requireRealGPU()
        let probe = try probeBatch(100, keyMod: 8)
        let build = try buildBatch((0..<8).map { Int64($0) })
        for op in [StreamAggregate.Op.variance, .stddev, .countDistinctApprox] {
            XCTAssertThrowsError(try StreamQuery(source: probeSource([probe]))
                .join(build, on: "k64", buildKey: "b64")
                .aggregate([StreamAggregate(op, "amount", name: "x")])) { error in
                XCTAssertTrue("\(error)".contains("not implemented"), "\(error)")
            }
        }
    }

    func testMismatchedKeyTypesRaise() throws {
        try requireRealGPU()
        let probe = try probeBatch(100, keyMod: 8)
        let build = try buildBatch((0..<8).map { Int64($0) })
        XCTAssertThrowsError(try StreamQuery(source: probeSource([probe]))
            .join(build, on: "k32", buildKey: "b64").sum("amount"))
    }

    func testStringValueColumnRaises() throws {
        try requireRealGPU()
        let n = 64
        let probe = try MetalRecordBatch(names: ["k64", "label"], columns: [
            .int64(try MetalArray<Int64>((0..<n).map { Int64($0 % 8) })),
            .string(try MetalStringArray((0..<n).map { Optional("L\($0 % 3)") })),
        ])
        let build = try buildBatch((0..<8).map { Int64($0) })
        XCTAssertThrowsError(try StreamQuery(source: probeSource([probe]))
            .join(build, on: "k64", buildKey: "b64").sum("label"))
    }
}
