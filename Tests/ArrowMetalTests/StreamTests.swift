import XCTest
import CArrowABI
@testable import ArrowMetal

/// Out-of-core streaming execution: every operator is checked against an in-memory whole-dataset
/// oracle at sizes that span 1 to 50 batches, with ragged last batches, empty batches and nulls.
final class StreamTests: XCTestCase {

    // MARK: - fixtures

    private var scratch: URL!

    override func setUpWithError() throws {
        scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("arrowmetal-stream-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let s = scratch { try? FileManager.default.removeItem(at: s) }
    }

    /// Deterministic pseudo-random data: `id`, a dense `region` key, a sparse `label` key, a value
    /// with nulls, and a float. Row `i` is a pure function of `i`, so the oracle is trivial.
    struct Row {
        var id: Int64
        var region: Int32
        var label: String
        var value: Int64?
        var amount: Double
    }

    static func rows(_ n: Int, offset: Int = 0, regions: Int = 7, labels: Int = 5) -> [Row] {
        (0..<n).map { j in
            let i = j + offset
            let v: Int64? = (i % 11 == 3) ? nil : Int64((i &* 2654435761) % 1000)
            return Row(id: Int64(i),
                       region: Int32(i % regions),
                       label: "L\(i % labels)",
                       value: v,
                       amount: Double((i &* 48271) % 10_007) / 7.0)
        }
    }

    static func batch(_ rows: [Row]) throws -> MetalRecordBatch {
        try MetalRecordBatch(names: ["id", "region", "label", "value", "amount"], columns: [
            .int64(try MetalArray<Int64>(rows.map { $0.id })),
            .int32(try MetalArray<Int32>(rows.map { $0.region })),
            .string(try MetalStringArray(rows.map { Optional($0.label) })),
            .int64(try MetalArray<Int64>(rows.map { $0.value })),
            .float64(try MetalArray<Double>(rows.map { $0.amount })),
        ])
    }

    /// Splits `rows` into `sizes.count` batches with the given sizes (0 makes an empty batch).
    static func batches(_ rows: [Row], sizes: [Int]) throws -> [MetalRecordBatch] {
        var out: [MetalRecordBatch] = []
        var i = 0
        for s in sizes {
            let take = Swift.min(s, rows.count - i)
            out.append(try batch(Array(rows[i..<(i + Swift.max(0, take))])))
            i += Swift.max(0, take)
        }
        if i < rows.count { out.append(try batch(Array(rows[i...]))) }
        return out
    }

    /// A ragged batch layout for `n` rows: a few full batches, an empty one, and a short tail.
    static func raggedSizes(_ n: Int, batchRows: Int) -> [Int] {
        var sizes: [Int] = []
        var left = n
        var k = 0
        while left > 0 {
            if k == 2 { sizes.append(0); k += 1; continue }       // an empty batch in the middle
            let s = Swift.min(left, k % 3 == 1 ? Swift.max(1, batchRows / 2) : batchRows)
            sizes.append(s)
            left -= s
            k += 1
        }
        return sizes
    }

    private func source(_ rows: [Row], sizes: [Int]) throws -> BatchSource {
        ChunkedTableSource(try Self.batches(rows, sizes: sizes))
    }

    /// Writes the batches to one IPC stream file and returns a prefetching source over it.
    private func ipcSource(_ batches: [MetalRecordBatch], name: String = "data.arrows",
                           prefetch: Int = 3) throws -> (BatchSource, URL) {
        let url = scratch.appendingPathComponent(name)
        let sink = try IPCStreamSink(url: url)
        for b in batches { try sink.write(b) }
        try sink.finish()
        return (try openIPCSource(url.path, prefetchDepth: prefetch), url)
    }

    // MARK: - sources

    func testIPCStreamSinkRoundTripsThroughTheReader() throws {
        try requireRealGPU()
        let rows = Self.rows(5_000)
        let bs = try Self.batches(rows, sizes: Self.raggedSizes(5_000, batchRows: 700))
        let (src, url) = try ipcSource(bs)
        let read = try src.collect()
        XCTAssertEqual(read.reduce(0) { $0 + $1.length }, rows.count)
        let all = try concatBatches(read)
        XCTAssertEqual(all.names, ["id", "region", "label", "value", "amount"])
        XCTAssertEqual(all[0].asInt64?.toArray().compactMap { $0 }, rows.map { $0.id })
        XCTAssertEqual(all["label"]?.asString?.toArray().compactMap { $0 }, rows.map { $0.label })
        XCTAssertEqual(all["value"]?.asInt64?.toArray(), rows.map { $0.value })
        // The file is a valid Arrow IPC stream in its own right.
        let reader = try ArrowIPCReader(url: url)
        XCTAssertEqual(reader.format, .stream)
        XCTAssertEqual(try reader.readAll().reduce(0) { $0 + $1.length }, rows.count)
    }

    func testDirectorySourceReadsEveryFileInOrder() throws {
        try requireRealGPU()
        let dir = scratch.appendingPathComponent("parts")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var expected: [Int64] = []
        for f in 0..<4 {
            let rows = Self.rows(300, offset: f * 300)
            expected += rows.map { $0.id }
            let sink = try IPCStreamSink(url: dir.appendingPathComponent(String(format: "part-%03d.arrows", f)))
            try sink.write(try Self.batch(rows))
            try sink.finish()
        }
        let src = try IPCDirectorySource(directory: dir)
        let all = try concatBatches(try src.collect())
        XCTAssertEqual(all["id"]?.asInt64?.toArray().compactMap { $0 }, expected)
        XCTAssertGreaterThan(src.totalBytes, 0)
    }

    func testParallelSourceReadsEveryRowAcrossFiles() throws {
        try requireRealGPU()
        let dir = scratch.appendingPathComponent("parallel")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var expected: Set<Int64> = []
        for f in 0..<6 {
            let rows = Self.rows(500, offset: f * 500)
            expected.formUnion(rows.map { $0.id })
            let sink = try IPCStreamSink(url: dir.appendingPathComponent(String(format: "p-%03d.arrows", f)))
            // Several batches per file, so the readers interleave.
            for chunk in stride(from: 0, to: rows.count, by: 137) {
                try sink.write(try Self.batch(Array(rows[chunk..<Swift.min(chunk + 137, rows.count)])))
            }
            try sink.finish()
        }
        let src = try ParallelIPCSource(directory: dir, readers: 4, depth: 2)
        let all = try concatBatches(try src.collect())
        // Order is deliberately not preserved; every row must still arrive exactly once.
        let got = try XCTUnwrap(all["id"]?.asInt64).toArray().compactMap { $0 }
        XCTAssertEqual(got.count, expected.count)
        XCTAssertEqual(Set(got), expected)
        XCTAssertGreaterThan(src.totalBytes, 0)
        src.close()
    }

    func testParallelSourceFeedsAnAggregateIdentically() throws {
        try requireRealGPU()
        let dir = scratch.appendingPathComponent("parallel-agg")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var all: [Row] = []
        for f in 0..<5 {
            let rows = Self.rows(900, offset: f * 900)
            all += rows
            let sink = try IPCStreamSink(url: dir.appendingPathComponent(String(format: "p-%03d.arrows", f)))
            try sink.write(try Self.batch(rows))
            try sink.finish()
        }
        let serial = try StreamQuery(source: try openIPCSource(dir.path, readers: 1)).sum("value")
        let parallel = try StreamQuery(source: try openIPCSource(dir.path, readers: 4)).sum("value")
        XCTAssertEqual(serial?.asInt64, all.compactMap { $0.value }.reduce(0, &+))
        XCTAssertEqual(parallel?.asInt64, serial?.asInt64)
    }

    func testExternalSortMergesInPassesWhenThereAreManyRuns() throws {
        try requireRealGPU()
        let n = 12_000
        let rows = Self.rows(n)
        // 120 batches means 120 runs, which is past the default fan-in of 32: the merge has to run
        // in passes rather than open every run at once.
        let src = try source(rows, sizes: Array(repeating: n / 120, count: 120))
        let collected = CollectingSink()
        let op = try ExternalSortOperator(keys: [ExternalSortOperator.Key("amount")],
                                          sink: collected, scratch: scratch)
        op.mergeFanIn = 8
        let r = try StreamingExecutor(source: src).run(op)
        XCTAssertEqual(op.runCount, 120)
        XCTAssertEqual(r.rowsOut, n)
        let out = try XCTUnwrap(try collected.table())
        XCTAssertEqual(try XCTUnwrap(out["amount"]?.asFloat64).toArray().compactMap { $0 },
                       rows.map { $0.amount }.sorted())
        // Every intermediate run file is cleaned up.
        let left = try FileManager.default.contentsOfDirectory(atPath: scratch.path)
            .filter { $0.hasPrefix("run-") || $0.hasPrefix("merge-") }
        XCTAssertEqual(left, [])
    }

    func testExternalSortWithLimitAcrossManyRuns() throws {
        try requireRealGPU()
        let n = 9_000
        let rows = Self.rows(n)
        let src = try source(rows, sizes: Array(repeating: 100, count: 90))
        let collected = CollectingSink()
        let op = try ExternalSortOperator(keys: [ExternalSortOperator.Key("amount", descending: true)],
                                          sink: collected, scratch: scratch, limit: 17)
        op.mergeFanIn = 4
        _ = try StreamingExecutor(source: src).run(op)
        let out = try XCTUnwrap(try collected.table())
        XCTAssertEqual(try XCTUnwrap(out["amount"]?.asFloat64).toArray().compactMap { $0 },
                       Array(rows.map { $0.amount }.sorted(by: >).prefix(17)))
    }

    func testPrefetchingSourceYieldsTheSameBatches() throws {
        try requireRealGPU()
        let rows = Self.rows(4_000)
        let bs = try Self.batches(rows, sizes: Self.raggedSizes(4_000, batchRows: 333))
        let plain = ChunkedTableSource(bs)
        let pre = PrefetchingSource(ChunkedTableSource(bs), depth: 3, budgetBytes: 1 << 20)
        let a = try plain.collect().map { $0.length }
        let b = try pre.collect().map { $0.length }
        XCTAssertEqual(a, b)
    }

    // MARK: - aggregates

    func testStreamingAggregatesMatchTheWholeDatasetOracle() throws {
        try requireRealGPU()
        for n in [1, 999, 10_000, 50_000] {
            for batchRows in [n, 1_000, 250] {
                let rows = Self.rows(n)
                let src = try source(rows, sizes: Self.raggedSizes(n, batchRows: batchRows))
                let r = try StreamQuery(source: src).aggregate([
                    StreamAggregate(.sum, "value", name: "sum"),
                    StreamAggregate(.count, nil, name: "n"),
                    StreamAggregate(.count, "value", name: "nv"),
                    StreamAggregate(.min, "value", name: "mn"),
                    StreamAggregate(.max, "value", name: "mx"),
                    StreamAggregate(.mean, "amount", name: "avg"),
                    StreamAggregate(.variance, "amount", name: "var"),
                ])
                let valid = rows.compactMap { $0.value }
                XCTAssertEqual(r.scalar("sum")?.asInt64, valid.reduce(0, &+), "n=\(n) b=\(batchRows)")
                XCTAssertEqual(r.scalar("n")?.asInt64, Int64(n))
                XCTAssertEqual(r.scalar("nv")?.asInt64, Int64(valid.count))
                XCTAssertEqual(r.scalar("mn")?.asInt64, valid.min())
                XCTAssertEqual(r.scalar("mx")?.asInt64, valid.max())
                let amounts = rows.map { $0.amount }
                let mean = amounts.reduce(0, +) / Double(amounts.count)
                XCTAssertEqual(r.scalar("avg")!.asDouble!, mean, accuracy: 1e-6 * Swift.max(1, abs(mean)))
                let variance = amounts.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(amounts.count)
                XCTAssertEqual(r.scalar("var")!.asDouble!, variance, accuracy: 1e-4 * Swift.max(1, variance))
            }
        }
    }

    func testFilteredAggregateMatchesTheOracle() throws {
        try requireRealGPU()
        let n = 20_000
        let rows = Self.rows(n)
        let src = try source(rows, sizes: Self.raggedSizes(n, batchRows: 512))
        let q = StreamQuery(source: src).filter(col("region") == 3)
        let r = try q.aggregate([StreamAggregate(.sum, "value", name: "s"),
                                 StreamAggregate(.count, nil, name: "c")])
        let kept = rows.filter { $0.region == 3 }
        XCTAssertEqual(r.scalar("c")?.asInt64, Int64(kept.count))
        XCTAssertEqual(r.scalar("s")?.asInt64, kept.compactMap { $0.value }.reduce(0, &+))
    }

    func testAggregateOverAnEmptyDataset() throws {
        try requireRealGPU()
        let src = ChunkedTableSource([try Self.batch([])])
        let r = try StreamQuery(source: src).aggregate([
            StreamAggregate(.sum, "value", name: "s"),
            StreamAggregate(.count, nil, name: "c"),
            StreamAggregate(.min, "value", name: "m"),
        ])
        XCTAssertEqual(r.scalar("c")?.asInt64, 0)
        XCTAssertTrue(r.scalar("m")?.isNullScalar ?? true)
    }

    // MARK: - HyperLogLog

    func testHyperLogLogAccuracyBounds() throws {
        try requireRealGPU()
        // Distinct counts spanning the linear-counting and the HLL estimator regimes.
        for distinct in [10, 1_000, 100_000, 1_000_000] {
            let p = 14
            var sketch = HLLSketch(precision: p)
            var produced = 0
            while produced < distinct {
                let take = Swift.min(200_000, distinct - produced)
                let col = AnyMetalArray.int64(try MetalArray<Int64>((0..<take).map { Int64($0 + produced) }))
                sketch.merge(try gpuHyperLogLog(col, precision: p))
                produced += take
            }
            let error = abs(sketch.estimate - Double(distinct)) / Double(distinct)
            // Three standard errors is the bound a correct implementation must hold to.
            XCTAssertLessThan(error, Swift.max(0.03, 3 * sketch.standardError),
                              "distinct=\(distinct) estimate=\(sketch.estimate)")
        }
    }

    func testHyperLogLogIsOrderAndSplitIndependent() throws {
        try requireRealGPU()
        let values = (0..<50_000).map { Int64(($0 &* 2654435761) % 20_000) }
        let whole = try gpuHyperLogLog(.int64(try MetalArray<Int64>(values)), precision: 12)
        var split = HLLSketch(precision: 12)
        for chunk in stride(from: 0, to: values.count, by: 3_000) {
            let part = Array(values[chunk..<Swift.min(chunk + 3_000, values.count)])
            split.merge(try gpuHyperLogLog(.int64(try MetalArray<Int64>(part)), precision: 12))
        }
        XCTAssertEqual(whole.registers, split.registers)
    }

    func testStreamingCountDistinctApprox() throws {
        try requireRealGPU()
        let n = 200_000
        let rows = Self.rows(n, regions: 512, labels: 97)
        let src = try source(rows, sizes: Self.raggedSizes(n, batchRows: 4_096))
        let ndv = try StreamQuery(source: src).countDistinctApprox("id", precision: 14)
        XCTAssertLessThan(abs(Double(ndv) - Double(n)) / Double(n), 0.05)

        let src2 = try source(rows, sizes: Self.raggedSizes(n, batchRows: 4_096))
        let labels = try StreamQuery(source: src2).countDistinctApprox("label", precision: 14)
        XCTAssertEqual(labels, 97, accuracy: 3)
    }

    // MARK: - group-by

    func testStreamingGroupByArbitraryKeysMatchesTheOracle() throws {
        try requireRealGPU()
        for n in [1, 5_000, 60_000] {
            let rows = Self.rows(n, regions: 7, labels: 5)
            let src = try source(rows, sizes: Self.raggedSizes(n, batchRows: 900))
            let r = try StreamQuery(source: src).groupBy(["label"], [
                StreamAggregate(.sum, "value", name: "s"),
                StreamAggregate(.count, nil, name: "c"),
                StreamAggregate(.min, "value", name: "mn"),
                StreamAggregate(.max, "value", name: "mx"),
                StreamAggregate(.mean, "amount", name: "avg"),
            ])
            let batch = try XCTUnwrap(r.batch)
            var oracle: [String: (Int64, Int, Int64?, Int64?, Double, Int)] = [:]
            for row in rows {
                var e = oracle[row.label] ?? (0, 0, nil, nil, 0, 0)
                if let v = row.value {
                    e.0 &+= v
                    e.2 = e.2.map { Swift.min($0, v) } ?? v
                    e.3 = e.3.map { Swift.max($0, v) } ?? v
                }
                e.1 += 1
                e.4 += row.amount
                e.5 += 1
                oracle[row.label] = e
            }
            XCTAssertEqual(batch.length, oracle.count, "n=\(n)")
            let keys = try XCTUnwrap(batch["label"]?.asString).toArray()
            let sums = try XCTUnwrap(batch["s"]?.asInt64).toArray()
            let counts = try XCTUnwrap(batch["c"]?.asInt64).toArray()
            let mins = try XCTUnwrap(batch["mn"]?.asInt64).toArray()
            let maxs = try XCTUnwrap(batch["mx"]?.asInt64).toArray()
            let avgs = try XCTUnwrap(batch["avg"]?.asFloat64).toArray()
            for (i, k) in keys.enumerated() {
                let e = try XCTUnwrap(oracle[try XCTUnwrap(k)])
                XCTAssertEqual(sums[i], e.0 == 0 && e.2 == nil ? nil : e.0)
                XCTAssertEqual(counts[i], Int64(e.1))
                XCTAssertEqual(mins[i], e.2)
                XCTAssertEqual(maxs[i], e.3)
                XCTAssertEqual(try XCTUnwrap(avgs[i]), e.4 / Double(e.5), accuracy: 1e-6)
            }
            // Keys come back sorted, so the result is deterministic across runs and batch layouts.
            XCTAssertEqual(keys.compactMap { $0 }, keys.compactMap { $0 }.sorted())
        }
    }

    func testStreamingGroupByMultipleKeys() throws {
        try requireRealGPU()
        let n = 30_000
        let rows = Self.rows(n, regions: 6, labels: 4)
        let src = try source(rows, sizes: Self.raggedSizes(n, batchRows: 700))
        let r = try StreamQuery(source: src).groupBy(["region", "label"], [
            StreamAggregate(.count, nil, name: "c"),
            StreamAggregate(.sum, "value", name: "s"),
        ])
        let batch = try XCTUnwrap(r.batch)
        var oracle: [String: (Int, Int64)] = [:]
        for row in rows {
            var e = oracle["\(row.region)|\(row.label)"] ?? (0, 0)
            e.0 += 1
            e.1 &+= row.value ?? 0
            oracle["\(row.region)|\(row.label)"] = e
        }
        XCTAssertEqual(batch.length, oracle.count)
        let regions = try XCTUnwrap(batch["region"]?.asInt32).toArray()
        let labels = try XCTUnwrap(batch["label"]?.asString).toArray()
        let counts = try XCTUnwrap(batch["c"]?.asInt64).toArray()
        let sums = try XCTUnwrap(batch["s"]?.asInt64).toArray()
        for i in 0..<batch.length {
            let e = try XCTUnwrap(oracle["\(try XCTUnwrap(regions[i]))|\(try XCTUnwrap(labels[i]))"])
            XCTAssertEqual(counts[i], Int64(e.0))
            XCTAssertEqual(sums[i] ?? 0, e.1)
        }
    }

    func testStreamingGroupByDenseKeysKeepsStateOnTheGPU() throws {
        try requireRealGPU()
        let n = 40_000
        let regions = 7
        let rows = Self.rows(n, regions: regions)
        let src = try source(rows, sizes: Self.raggedSizes(n, batchRows: 800))
        let op = StreamGroupByOperator(keys: ["region"], aggregates: [
            StreamAggregate(.sum, "value", name: "s"),
            StreamAggregate(.count, nil, name: "c"),
            StreamAggregate(.max, "value", name: "mx"),
        ], denseKeyCount: regions)
        let r = try StreamingExecutor(source: src).run(op)
        XCTAssertTrue(op.usesGPUState, "the dense path should keep the global table on the GPU")
        let batch = try XCTUnwrap(r.batch)
        var sums = [Int64](repeating: 0, count: regions)
        var counts = [Int64](repeating: 0, count: regions)
        var maxs = [Int64?](repeating: nil, count: regions)
        for row in rows {
            let k = Int(row.region)
            counts[k] += 1
            if let v = row.value {
                sums[k] &+= v
                maxs[k] = maxs[k].map { Swift.max($0, v) } ?? v
            }
        }
        let keys = try XCTUnwrap(batch["region"]?.asInt32).toArray()
        let gotS = try XCTUnwrap(batch["s"]?.asInt64).toArray()
        let gotC = try XCTUnwrap(batch["c"]?.asInt64).toArray()
        let gotM = try XCTUnwrap(batch["mx"]?.asInt64).toArray()
        XCTAssertEqual(batch.length, regions)
        for (i, k) in keys.enumerated() {
            let k = Int(try XCTUnwrap(k))
            XCTAssertEqual(gotS[i], sums[k])
            XCTAssertEqual(gotC[i], counts[k])
            XCTAssertEqual(gotM[i], maxs[k])
        }
    }

    func testDenseGroupBySpillsToHostWhenTheBudgetIsTiny() throws {
        try requireRealGPU()
        let n = 12_000
        let regions = 9
        let rows = Self.rows(n, regions: regions)
        let src = try source(rows, sizes: Self.raggedSizes(n, batchRows: 500))
        let op = StreamGroupByOperator(keys: ["region"], aggregates: [
            StreamAggregate(.sum, "value", name: "s"),
            StreamAggregate(.count, nil, name: "c"),
        ], denseKeyCount: regions)
        op.gpuStateBudgetBytes = 1        // force an immediate spill to the host table
        let r = try StreamingExecutor(source: src).run(op)
        XCTAssertFalse(op.usesGPUState)
        let batch = try XCTUnwrap(r.batch)
        var sums = [Int64](repeating: 0, count: regions)
        var counts = [Int64](repeating: 0, count: regions)
        for row in rows { counts[Int(row.region)] += 1; sums[Int(row.region)] &+= row.value ?? 0 }
        // After the spill the key column takes the shape of the source key column (int32).
        XCTAssertEqual(batch.length, regions)
        let keys = try XCTUnwrap(batch["region"]?.asInt32).toArray()
        let gotS = try XCTUnwrap(batch["s"]?.asInt64).toArray()
        let gotC = try XCTUnwrap(batch["c"]?.asInt64).toArray()
        for (i, k) in keys.enumerated() {
            let k = Int(try XCTUnwrap(k))
            XCTAssertEqual(gotS[i], sums[k])
            XCTAssertEqual(gotC[i], counts[k])
        }
    }

    // MARK: - top-k

    func testStreamingTopKMatchesTheWholeDatasetTopK() throws {
        try requireRealGPU()
        for (n, k) in [(500, 10), (25_000, 32), (25_000, 1)] {
            let rows = Self.rows(n)
            let src = try source(rows, sizes: Self.raggedSizes(n, batchRows: 700))
            let r = try StreamQuery(source: src).topK("amount", k: k, largest: true)
            let got = try XCTUnwrap(r.batch)
            XCTAssertEqual(got.length, k)
            let expected = rows.map { $0.amount }.sorted(by: >).prefix(k)
            let actual = try XCTUnwrap(got["amount"]?.asFloat64).toArray().compactMap { $0 }
            XCTAssertEqual(actual, Array(expected), "n=\(n) k=\(k)")
        }
    }

    func testStreamingTopKSmallest() throws {
        try requireRealGPU()
        let n = 9_000
        let rows = Self.rows(n)
        let src = try source(rows, sizes: Self.raggedSizes(n, batchRows: 400))
        let r = try StreamQuery(source: src).topK("amount", k: 12, largest: false)
        let actual = try XCTUnwrap(try XCTUnwrap(r.batch)["amount"]?.asFloat64).toArray().compactMap { $0 }
        XCTAssertEqual(actual, Array(rows.map { $0.amount }.sorted().prefix(12)))
    }

    // MARK: - quantiles

    func testStreamingQuantilesAreWithinTheirRankErrorBound() throws {
        try requireRealGPU()
        let n = 200_000
        let rows = Self.rows(n)
        let sorted = rows.map { $0.amount }.sorted()
        let src = try source(rows, sizes: Self.raggedSizes(n, batchRows: 8_192))
        let qs = [0.01, 0.1, 0.5, 0.9, 0.99]
        let got = try StreamQuery(source: src).quantiles("amount", qs, compression: 1000)
        for (i, q) in qs.enumerated() {
            let exact = sorted[Swift.min(n - 1, Int(q * Double(n)))]
            let value = try XCTUnwrap(got[i])
            // Compare in rank space: how far off is the reported value's true rank?
            let rank = sorted.firstIndex { $0 >= value } ?? n
            let rankError = abs(Double(rank) - q * Double(n)) / Double(n)
            XCTAssertLessThan(rankError, 0.01, "q=\(q) value=\(value) exact=\(exact)")
        }
    }

    // MARK: - filter / project into a sink

    func testFilterProjectToIPCSinkMatchesTheOracle() throws {
        try requireRealGPU()
        let n = 30_000
        let rows = Self.rows(n)
        let src = try source(rows, sizes: Self.raggedSizes(n, batchRows: 1_100))
        let out = scratch.appendingPathComponent("filtered.arrows")
        let r = try StreamQuery(source: src)
            .filter(col("value") > 500)
            .project([("id", col("id")), ("doubled", col("value") * 2)])
            .sinkIPC(out)
        let expected = rows.filter { ($0.value ?? 0) > 500 }
        XCTAssertEqual(r.rowsOut, expected.count)
        let back = try concatBatches(try ArrowIPCReader(url: out).readAll())
        XCTAssertEqual(back.names, ["id", "doubled"])
        XCTAssertEqual(back["id"]?.asInt64?.toArray().compactMap { $0 }, expected.map { $0.id })
        XCTAssertEqual(back["doubled"]?.asInt64?.toArray().compactMap { $0 }, expected.map { $0.value! * 2 })
    }

    func testReaderPullsOneBatchAtATime() throws {
        try requireRealGPU()
        let n = 8_000
        let rows = Self.rows(n)
        let src = try source(rows, sizes: Self.raggedSizes(n, batchRows: 900))
        let reader = StreamQuery(source: src).filter(col("region") == 1).reader()
        var total = 0, batches = 0
        while let b = try reader.nextOutputBatch() {
            total += b.length
            batches += 1
            XCTAssertGreaterThan(b.length, 0)
        }
        XCTAssertEqual(total, rows.filter { $0.region == 1 }.count)
        XCTAssertGreaterThan(batches, 1)
    }

    // MARK: - Arrow C Stream round trip

    func testCStreamRoundTrip() throws {
        try requireRealGPU()
        let n = 6_000
        let rows = Self.rows(n)
        let src = try source(rows, sizes: Self.raggedSizes(n, batchRows: 800))

        // Export a streamed query as an ArrowArrayStream, then read it back through CStreamSource.
        var stream = ArrowArrayStream()
        StreamQuery(source: src).filter(col("value") > 300).exportArrowArrayStream(into: &stream)
        let ptr = UnsafeMutablePointer<ArrowArrayStream>.allocate(capacity: 1)
        ptr.initialize(to: stream)
        defer { ptr.deallocate() }

        let back = try CStreamSource(ptr)
        let all = try concatBatches(try back.collect())
        let expected = rows.filter { ($0.value ?? 0) > 300 }
        XCTAssertEqual(all.length, expected.count)
        XCTAssertEqual(all["id"]?.asInt64?.toArray().compactMap { $0 }, expected.map { $0.id })
        XCTAssertEqual(all["label"]?.asString?.toArray().compactMap { $0 }, expected.map { $0.label })
        back.close()
    }

    func testCStreamSourceFeedsAStreamingAggregate() throws {
        try requireRealGPU()
        let n = 12_000
        let rows = Self.rows(n)
        let src = try source(rows, sizes: Self.raggedSizes(n, batchRows: 1_000))
        var stream = ArrowArrayStream()
        ArrowStreamExporter(src).export(into: &stream)
        let ptr = UnsafeMutablePointer<ArrowArrayStream>.allocate(capacity: 1)
        ptr.initialize(to: stream)
        defer { ptr.deallocate() }
        let r = try StreamQuery(source: try CStreamSource(ptr)).aggregate([
            StreamAggregate(.sum, "value", name: "s"),
            StreamAggregate(.count, nil, name: "c"),
        ])
        XCTAssertEqual(r.scalar("c")?.asInt64, Int64(n))
        XCTAssertEqual(r.scalar("s")?.asInt64, rows.compactMap { $0.value }.reduce(0, &+))
    }

    // MARK: - external sort

    func testExternalSortOverTwentyRunsMatchesAnInMemorySort() throws {
        try requireRealGPU()
        let n = 20_000
        let rows = Self.rows(n)
        // 20 batches means 20 sorted runs on disk.
        let sizes = (0..<20).map { _ in n / 20 }
        let src = try source(rows, sizes: sizes)
        let collected = CollectingSink()
        let r = try StreamQuery(source: src).sort(by: [ExternalSortOperator.Key("amount")],
                                                  into: collected, scratch: scratch)
        XCTAssertEqual(r.rowsOut, n)
        let out = try XCTUnwrap(try collected.table())
        XCTAssertEqual(out.length, n)
        let got = try XCTUnwrap(out["amount"]?.asFloat64).toArray().compactMap { $0 }
        XCTAssertEqual(got, rows.map { $0.amount }.sorted())
        // The payload columns travelled with their rows.
        let ids = try XCTUnwrap(out["id"]?.asInt64).toArray().compactMap { $0 }
        let expectedIds = rows.sorted { ($0.amount, $0.id) < ($1.amount, $1.id) }.map { $0.id }
        XCTAssertEqual(Set(ids), Set(expectedIds))
        for i in 1..<out.length {
            XCTAssertLessThanOrEqual(got[i - 1], got[i])
        }
    }

    func testExternalSortDescendingWithLimit() throws {
        try requireRealGPU()
        let n = 15_000
        let rows = Self.rows(n)
        let src = try source(rows, sizes: Self.raggedSizes(n, batchRows: 800))
        let collected = CollectingSink()
        _ = try StreamQuery(source: src).sort(by: [ExternalSortOperator.Key("amount", descending: true)],
                                              into: collected, scratch: scratch, limit: 25)
        let out = try XCTUnwrap(try collected.table())
        XCTAssertEqual(out.length, 25)
        let got = try XCTUnwrap(out["amount"]?.asFloat64).toArray().compactMap { $0 }
        XCTAssertEqual(got, Array(rows.map { $0.amount }.sorted(by: >).prefix(25)))
    }

    func testExternalSortWithNullsPutsThemLast() throws {
        try requireRealGPU()
        let n = 6_000
        let rows = Self.rows(n)
        let src = try source(rows, sizes: Self.raggedSizes(n, batchRows: 700))
        let collected = CollectingSink()
        _ = try StreamQuery(source: src).sort(by: [ExternalSortOperator.Key("value")],
                                              into: collected, scratch: scratch)
        let out = try XCTUnwrap(try collected.table())
        let got = try XCTUnwrap(out["value"]?.asInt64).toArray()
        let nulls = rows.filter { $0.value == nil }.count
        XCTAssertEqual(got.count, n)
        XCTAssertEqual(got.suffix(nulls).compactMap { $0 }.count, 0)
        XCTAssertEqual(got.prefix(n - nulls).compactMap { $0 }, rows.compactMap { $0.value }.sorted())
    }

    // MARK: - joins

    func testBroadcastJoinEqualsAnInMemoryJoin() throws {
        try requireRealGPU()
        let n = 20_000
        let rows = Self.rows(n, regions: 40)
        let src = try source(rows, sizes: Self.raggedSizes(n, batchRows: 900))
        let build = try MetalRecordBatch(names: ["rid", "rname"], columns: [
            .int32(try MetalArray<Int32>((0..<40).map { Int32($0) })),
            .int64(try MetalArray<Int64>((0..<40).map { Int64($0 * 100) })),
        ])
        let collected = CollectingSink()
        let r = try StreamQuery(source: src).joinBroadcast(build, on: "region", buildKey: "rid",
                                                           kind: .inner, into: collected)
        XCTAssertEqual(r.rowsOut, n)
        let out = try XCTUnwrap(try collected.table())
        let regions = try XCTUnwrap(out["region"]?.asInt32).toArray()
        let names = try XCTUnwrap(out["rname"]?.asInt64).toArray()
        for i in 0..<out.length {
            XCTAssertEqual(names[i], Int64(try XCTUnwrap(regions[i])) * 100)
        }
    }

    func testGraceHashJoinEqualsAnInMemoryJoin() throws {
        try requireRealGPU()
        let n = 15_000, m = 4_000
        let left = try MetalRecordBatch(names: ["k", "lv"], columns: [
            .int64(try MetalArray<Int64>((0..<n).map { Int64(($0 &* 7919) % 5_000) })),
            .int64(try MetalArray<Int64>((0..<n).map { Int64($0) })),
        ])
        let right = try MetalRecordBatch(names: ["k", "rv"], columns: [
            .int64(try MetalArray<Int64>((0..<m).map { Int64(($0 &* 104729) % 5_000) })),
            .int64(try MetalArray<Int64>((0..<m).map { Int64($0 * 3) })),
        ])
        let inMemory = try left.join(right, on: "k", rightKey: "k", kind: .inner)

        // Same inputs, streamed in many small batches through the grace join.
        func chunks(_ b: MetalRecordBatch, _ size: Int) throws -> [MetalRecordBatch] {
            var out: [MetalRecordBatch] = []
            var i = 0
            while i < b.length {
                out.append(try b.slice(offset: i, length: Swift.min(size, b.length - i)))
                i += size
            }
            return out
        }
        let collected = CollectingSink()
        let stats = try graceHashJoin(left: ChunkedTableSource(try chunks(left, 700)),
                                      right: ChunkedTableSource(try chunks(right, 500)),
                                      leftKey: "k", rightKey: "k", kind: .inner,
                                      partitions: 8, scratch: scratch, sink: collected)
        XCTAssertEqual(stats.leftRows, n)
        XCTAssertEqual(stats.rightRows, m)
        XCTAssertEqual(stats.outputRows, inMemory.length)
        let out = try XCTUnwrap(try collected.table())
        func pairs(_ b: MetalRecordBatch) throws -> Set<[Int64]> {
            let lv = try XCTUnwrap(b["lv"]?.asInt64).toArray()
            let rv = try XCTUnwrap(b["rv"]?.asInt64).toArray()
            var s = Set<[Int64]>()
            for i in 0..<b.length { s.insert([lv[i] ?? -1, rv[i] ?? -1]) }
            return s
        }
        XCTAssertEqual(try pairs(out), try pairs(inMemory))
    }

    // MARK: - pipeline

    func testPipelineOverlapsItsThreeStages() throws {
        try requireRealGPU()
        let n = 400_000
        let rows = Self.rows(n)
        let bs = try Self.batches(rows, sizes: Array(repeating: 20_000, count: 20))
        let (src, _) = try ipcSource(bs, name: "overlap.arrows", prefetch: 3)
        let r = try StreamQuery(source: src).groupBy(["label"], [
            StreamAggregate(.sum, "value", name: "s"),
            StreamAggregate(.count, nil, name: "c"),
        ])
        XCTAssertEqual(r.stats.batches, 20)
        XCTAssertEqual(r.stats.rows, n)
        XCTAssertGreaterThan(r.stats.wallNanos, 0)
        // The reader thread did real work concurrently with the GPU stage.
        XCTAssertGreaterThan(r.stats.readNanos, 0)
        XCTAssertGreaterThan(r.stats.gpuNanos, 0)
    }

    func testProgressCallbackSeesEveryBatch() throws {
        try requireRealGPU()
        let n = 10_000
        let rows = Self.rows(n)
        let src = try source(rows, sizes: Array(repeating: 1_000, count: 10))
        let q = StreamQuery(source: src)
        var seen: [Int] = []
        q.progress = { seen.append($0.batches) }
        _ = try q.count()
        XCTAssertEqual(seen, Array(1...10))
    }

    // MARK: - concat / values helpers

    func testConcatColumnsPreservesValuesAndNulls() throws {
        try requireRealGPU()
        let a = AnyMetalArray.int64(try MetalArray<Int64>([1, nil, 3]))
        let b = AnyMetalArray.int64(try MetalArray<Int64>([nil, 5]))
        let c = try concatColumns([a, b])
        XCTAssertEqual(c.asInt64?.toArray(), [1, nil, 3, nil, 5])

        let s1 = AnyMetalArray.string(try MetalStringArray(["a", nil, "ccc"]))
        let s2 = AnyMetalArray.string(try MetalStringArray([""]))
        XCTAssertEqual(try concatColumns([s1, s2]).asString?.toArray(), ["a", nil, "ccc", ""])

        let b1 = AnyMetalArray.boolean(try makeBooleanArray([true, nil, false]))
        let b2 = AnyMetalArray.boolean(try makeBooleanArray([false, true]))
        XCTAssertEqual(try concatColumns([b1, b2]).asBoolean?.toArray(), [true, nil, false, false, true])
    }
    // MARK: - the resident GPU state, against the implementation it replaces

    /// A key column with duplicates and nulls, and a value column with nulls: the shapes the resident
    /// table has to agree with the host table on.
    struct KeyRow {
        var key: Int64?
        var amount: Double?
        var qty: Int32
    }

    static func keyRows(_ n: Int, keys: Int, nullEvery: Int = 13) -> [KeyRow] {
        (0..<n).map { (i: Int) -> KeyRow in
            let key: Int64? = i % nullEvery == 5 ? nil : Int64((i &* 2654435761) % keys)
            let amount: Double? = i % 7 == 2 ? nil : Double((i &* 48271) % 10_007) / 7.0
            return KeyRow(key: key, amount: amount, qty: Int32(i % 97))
        }
    }

    static func keyBatch(_ rows: [KeyRow]) throws -> MetalRecordBatch {
        try MetalRecordBatch(names: ["key", "amount", "qty"], columns: [
            .int64(try MetalArray<Int64>(rows.map { $0.key })),
            .float64(try MetalArray<Double>(rows.map { $0.amount })),
            .int32(try MetalArray<Int32>(rows.map { Optional($0.qty) })),
        ])
    }

    static func keyBatches(_ rows: [KeyRow], sizes: [Int]) throws -> [MetalRecordBatch] {
        var out: [MetalRecordBatch] = []
        var i = 0
        for s in sizes {
            let take = Swift.max(0, Swift.min(s, rows.count - i))
            out.append(try keyBatch(Array(rows[i..<(i + take)])))
            i += take
        }
        if i < rows.count { out.append(try keyBatch(Array(rows[i...]))) }
        return out
    }

    private func groupByBoth(_ bs: [MetalRecordBatch], _ aggs: [StreamAggregate])
        throws -> (MetalRecordBatch, MetalRecordBatch) {
        let residentOp = StreamGroupByOperator(keys: ["key"], aggregates: aggs)
        let resident = try StreamingExecutor(source: ChunkedTableSource(bs)).run(residentOp)
        XCTAssertTrue(residentOp.usesResidentTable,
                      "an integer key with sum/count aggregates is the resident path")
        let hostOp = StreamGroupByOperator(keys: ["key"], aggregates: aggs)
        hostOp.residentTable = false
        let host = try StreamingExecutor(source: ChunkedTableSource(bs)).run(hostOp)
        return (try XCTUnwrap(resident.batch), try XCTUnwrap(host.batch))
    }

    /// The GPU-resident group table against the host table it replaces, row for row and bit for bit:
    /// duplicate keys, a null-key group, all-null groups, empty batches and a ragged tail.
    func testResidentGroupTableMatchesTheHostTable() throws {
        try requireRealGPU()
        let aggs = [StreamAggregate(.sum, "amount", name: "s"),
                    StreamAggregate(.count, nil, name: "c"),
                    StreamAggregate(.count, "amount", name: "cv"),
                    StreamAggregate(.sum, "qty", name: "sq"),
                    StreamAggregate(.mean, "amount", name: "avg")]
        for (n, keys) in [(1, 1), (5_000, 37), (60_000, 4_001), (120_000, 90_000)] {
            let rows = Self.keyRows(n, keys: keys)
            let bs = try Self.keyBatches(rows, sizes: Self.raggedSizes(n, batchRows: 900))
            let (a, b) = try groupByBoth(bs, aggs)
            XCTAssertEqual(a.length, b.length, "n=\(n) keys=\(keys)")
            XCTAssertEqual(a.names, b.names)
            XCTAssertEqual(try XCTUnwrap(a["key"]?.asInt64).toArray(),
                           try XCTUnwrap(b["key"]?.asInt64).toArray(), "keys differ at n=\(n)")
            XCTAssertEqual(try XCTUnwrap(a["c"]?.asInt64).toArray(),
                           try XCTUnwrap(b["c"]?.asInt64).toArray(), "count differs at n=\(n)")
            XCTAssertEqual(try XCTUnwrap(a["cv"]?.asInt64).toArray(),
                           try XCTUnwrap(b["cv"]?.asInt64).toArray(), "count(col) differs at n=\(n)")
            XCTAssertEqual(try XCTUnwrap(a["sq"]?.asInt64).toArray(),
                           try XCTUnwrap(b["sq"]?.asInt64).toArray(), "int sum differs at n=\(n)")
            // Both fold the same per-batch partials in the same order — one with Swift's `+`, one with
            // the correctly-rounded software adder — so the float64 sums agree to the last bit.
            let sa = try XCTUnwrap(a["s"]?.asFloat64).toArray()
            let sb = try XCTUnwrap(b["s"]?.asFloat64).toArray()
            XCTAssertEqual(sa.count, sb.count)
            for i in 0..<sa.count {
                XCTAssertEqual(sa[i]?.bitPattern, sb[i]?.bitPattern, "sum row \(i), n=\(n)")
            }
            let ma = try XCTUnwrap(a["avg"]?.asFloat64).toArray()
            let mb = try XCTUnwrap(b["avg"]?.asFloat64).toArray()
            for i in 0..<ma.count {
                XCTAssertEqual(ma[i]?.bitPattern, mb[i]?.bitPattern, "mean row \(i), n=\(n)")
            }
        }
    }

    /// The resident table grows and rehashes when the key space outruns it, without losing a group.
    func testResidentGroupTableRehashesWithoutLosingGroups() throws {
        try requireRealGPU()
        let n = 200_000
        let rows = (0..<n).map { KeyRow(key: Int64($0), amount: Double($0), qty: 1) }
        let bs = try Self.keyBatches(rows, sizes: Array(repeating: 4_096, count: 60))
        let (a, b) = try groupByBoth(bs, [StreamAggregate(.sum, "amount", name: "s"),
                                          StreamAggregate(.count, nil, name: "c")])
        XCTAssertEqual(a.length, n)
        XCTAssertEqual(a.length, b.length)
        XCTAssertEqual(try XCTUnwrap(a["key"]?.asInt64).toArray(),
                       try XCTUnwrap(b["key"]?.asInt64).toArray())
        let sa = try XCTUnwrap(a["s"]?.asFloat64).toArray()
        for i in 0..<n { XCTAssertEqual(sa[i], Double(i)) }
    }

    /// Every integer key type the resident path accepts, against the host table.
    func testResidentGroupTableAcrossIntegerKeyTypes() throws {
        try requireRealGPU()
        let n = 20_000
        let base = Self.keyRows(n, keys: 251)
        let plain = try Self.keyBatches(base, sizes: Self.raggedSizes(n, batchRows: 700))
        for cast in ["int16", "int32", "uint32", "uint64"] {
            let bs: [MetalRecordBatch] = try plain.map { b in
                let k = try XCTUnwrap(b["key"]?.asInt64)
                let narrowed: AnyMetalArray
                switch cast {
                case "int16": narrowed = .int16(try k.cast(to: Int16.self))
                case "int32": narrowed = .int32(try k.cast(to: Int32.self))
                case "uint32": narrowed = .uint32(try k.cast(to: UInt32.self))
                default: narrowed = .uint64(try k.cast(to: UInt64.self))
                }
                return try MetalRecordBatch(names: b.names,
                                            columns: [narrowed, b.columns[1], b.columns[2]])
            }
            let aggs = [StreamAggregate(.sum, "amount", name: "s"),
                        StreamAggregate(.count, nil, name: "c")]
            let (got, want) = try groupByBoth(bs, aggs)
            XCTAssertEqual(got.length, want.length, "\(cast)")
            XCTAssertEqual(got["key"]?.arrowFormat, want["key"]?.arrowFormat, "\(cast)")
            let ga = try XCTUnwrap(got["s"]?.asFloat64).toArray()
            let wa = try XCTUnwrap(want["s"]?.asFloat64).toArray()
            XCTAssertEqual(ga.count, wa.count, "\(cast)")
            for i in 0..<ga.count { XCTAssertEqual(ga[i]?.bitPattern, wa[i]?.bitPattern, "\(cast) row \(i)") }
            XCTAssertEqual(try XCTUnwrap(got["c"]?.asInt64).toArray(),
                           try XCTUnwrap(want["c"]?.asInt64).toArray(), "\(cast)")
        }
    }

    // MARK: - threshold pruning

    /// Threshold-pruned top-k against the same operator with pruning off: identical rows, including
    /// ties at the k-th value and batches in which nothing survives.
    func testTopKPruningMatchesTheUnprunedSelection() throws {
        try requireRealGPU()
        // Front-loaded on purpose: the first batch holds the largest values, so later batches are
        // pruned away entirely, and there are many ties exactly at the k-th value.
        let n = 60_000
        func topAmount(_ i: Int) -> Double? {
            if i < 500 { return 1000.0 - Double(i % 5) }
            if i % 23 == 0 { return nil }
            return Double((i &* 48271) % 900)
        }
        var rows: [KeyRow] = []
        rows.reserveCapacity(n)
        for i in 0..<n { rows.append(KeyRow(key: Int64(i), amount: topAmount(i), qty: Int32(i % 31))) }
        let bs = try Self.keyBatches(rows, sizes: Self.raggedSizes(n, batchRows: 1_100))
        for k in [1, 10, 100] {
            let pruned = StreamTopKOperator(column: "amount", k: k)
            let a = try XCTUnwrap(try StreamingExecutor(source: ChunkedTableSource(bs)).run(pruned).batch)
            let plainOp = StreamTopKOperator(column: "amount", k: k)
            plainOp.pruning = false
            let b = try XCTUnwrap(try StreamingExecutor(source: ChunkedTableSource(bs)).run(plainOp).batch)
            XCTAssertGreaterThan(pruned.prunedBatches, 0, "k=\(k): the point is that batches drop out")
            XCTAssertEqual(a.length, k)
            XCTAssertEqual(a.length, b.length)
            for name in ["key", "amount", "qty"] {
                let x: [StreamValue] = try XCTUnwrap(a[name]).streamValues()
                let y: [StreamValue] = try XCTUnwrap(b[name]).streamValues()
                XCTAssertEqual(x, y, "k=\(k) column \(name)")
            }
        }
    }

    /// Smallest-first pruning, over a column that is mostly null after the first batch.
    func testTopKSmallestPruningMatchesTheUnprunedSelection() throws {
        try requireRealGPU()
        let n = 30_000
        func smallAmount(_ i: Int) -> Double? {
            if i < 200 { return Double(i) }
            if i % 5 == 0 { return nil }
            return Double(1_000 + i)
        }
        var rows: [KeyRow] = []
        rows.reserveCapacity(n)
        for i in 0..<n { rows.append(KeyRow(key: Int64(i), amount: smallAmount(i), qty: Int32(i % 17))) }
        let bs = try Self.keyBatches(rows, sizes: Self.raggedSizes(n, batchRows: 900))
        let pruned = StreamTopKOperator(column: "amount", k: 50, largest: false)
        let a = try XCTUnwrap(try StreamingExecutor(source: ChunkedTableSource(bs)).run(pruned).batch)
        let plainOp = StreamTopKOperator(column: "amount", k: 50, largest: false)
        plainOp.pruning = false
        let b = try XCTUnwrap(try StreamingExecutor(source: ChunkedTableSource(bs)).run(plainOp).batch)
        XCTAssertGreaterThan(pruned.prunedBatches, 0)
        let amountA: [StreamValue] = try XCTUnwrap(a["amount"]).streamValues()
        let amountB: [StreamValue] = try XCTUnwrap(b["amount"]).streamValues()
        XCTAssertEqual(amountA, amountB)
        let keyA: [StreamValue] = try XCTUnwrap(a["key"]).streamValues()
        let keyB: [StreamValue] = try XCTUnwrap(b["key"]).streamValues()
        XCTAssertEqual(keyA, keyB)
    }

    /// `ORDER BY ... LIMIT n` answers as a resident top-n and spills no runs, and its rows are the
    /// external sort's own first n — same order, same tie-breaking, nulls in the same place.
    func testOrderByLimitTopNMatchesTheSortedOracle() throws {
        try requireRealGPU()
        let n = 40_000
        func sortAmount(_ i: Int) -> Double? {
            if i < 300 { return 500.0 - Double(i % 4) }
            if i % 19 == 0 { return nil }
            return Double(i % 400)
        }
        var rows: [KeyRow] = []
        rows.reserveCapacity(n)
        for i in 0..<n { rows.append(KeyRow(key: Int64(i), amount: sortAmount(i), qty: Int32(i % 11))) }
        let bs = try Self.keyBatches(rows, sizes: Self.raggedSizes(n, batchRows: 1_000))
        for (limit, desc) in [(1, true), (25, true), (1_000, false)] {
            let sink = CollectingSink()
            let op = try ExternalSortOperator(keys: [.init("amount", descending: desc)], sink: sink,
                                              scratch: scratch.appendingPathComponent("topn-\(limit)-\(desc)"),
                                              limit: limit)
            _ = try StreamingExecutor(source: ChunkedTableSource(bs)).run(op)
            XCTAssertEqual(op.runCount, 0, "a limited sort spills nothing")
            XCTAssertLessThan(op.candidateRows, n, "the threshold should reject most rows")
            // Descending, the first batch already holds every large value, so later batches lose
            // every row they have. Ascending, small values keep arriving, so batches only shrink.
            if desc { XCTAssertGreaterThan(op.prunedBatches, 0, "limit=\(limit)") }
            let got = try XCTUnwrap(try sink.table())

            // Oracle: the whole dataset in the order `argsort` puts it in, nulls last.
            var order = Array(0..<n)
            order.sort { i, j in
                let x = rows[i].amount, y = rows[j].amount
                if x == y { return i < j }
                guard let x else { return false }
                guard let y else { return true }
                return desc ? x > y : x < y
            }
            let want = Array(order.prefix(limit))
            XCTAssertEqual(got.length, want.count, "limit=\(limit) desc=\(desc)")
            let keys = try XCTUnwrap(got["key"]?.asInt64).toArray()
            for (i, w) in want.enumerated() { XCTAssertEqual(keys[i], Int64(w), "row \(i) limit=\(limit)") }
        }
    }

    /// A two-key limited sort: the prune keeps the rows that tie on the first key, so the second key
    /// still decides between them.
    func testOrderByLimitTopNWithTwoKeys() throws {
        try requireRealGPU()
        let n = 20_000
        // `qty` has four values, so every row ties five thousand others on the first key.
        let rows = (0..<n).map { KeyRow(key: Int64($0), amount: Double($0 % 977), qty: Int32($0 % 4)) }
        let bs = try Self.keyBatches(rows, sizes: Self.raggedSizes(n, batchRows: 700))
        let sink = CollectingSink()
        let op = try ExternalSortOperator(
            keys: [.init("qty", descending: true), .init("amount", descending: false)],
            sink: sink, scratch: scratch.appendingPathComponent("topn2"), limit: 40)
        _ = try StreamingExecutor(source: ChunkedTableSource(bs)).run(op)
        let got = try XCTUnwrap(try sink.table())
        var order = Array(0..<n)
        order.sort { i, j in
            if rows[i].qty != rows[j].qty { return rows[i].qty > rows[j].qty }
            if rows[i].amount != rows[j].amount { return (rows[i].amount ?? 0) < (rows[j].amount ?? 0) }
            return i < j
        }
        let keys = try XCTUnwrap(got["key"]?.asInt64).toArray()
        XCTAssertEqual(got.length, 40)
        for i in 0..<40 { XCTAssertEqual(keys[i], Int64(order[i]), "row \(i)") }
    }
}

private func XCTAssertEqual(_ a: Int, _ b: Int, accuracy: Int, file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertLessThanOrEqual(abs(a - b), accuracy, "\(a) vs \(b)", file: file, line: line)
}
