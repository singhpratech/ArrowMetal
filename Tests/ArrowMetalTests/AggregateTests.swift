import XCTest
import CArrowABI
@testable import ArrowMetal

/// Scalar aggregates, run-end encoding and dictionary compute, each checked against a plain Swift loop.
///
/// Tolerances: 1e-6 relative for Float32 and integer statistics (the GPU accumulates squared deviations
/// in compensated float pairs), 1e-12 relative for Float64 (software binary64 on the GPU).
final class AggregateTests: XCTestCase {

    /// The sizes every aggregate is checked at: empty, single, sub-threadgroup, multi-threadgroup, large.
    static let sizes = [0, 1, 33, 4097, 300_003]

    // MARK: - fixtures

    /// Deterministic pseudo-random values, with every third element null.
    private func doubles(_ n: Int, nulls: Bool) -> [Double?] {
        var state = UInt64(0x9E37_79B9_7F4A_7C15)
        return (0..<n).map { i in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let v = Double(Int64(bitPattern: state >> 11) % 20_000) / 128.0
            return (nulls && i % 3 == 1) ? nil : v
        }
    }

    private func ints(_ n: Int, nulls: Bool) -> [Int64?] {
        var state = UInt64(0x243F_6A88_85A3_08D3)
        return (0..<n).map { i in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let v = Int64(bitPattern: state >> 40) - 4_000_000
            return (nulls && i % 3 == 1) ? nil : v
        }
    }

    private func assertClose(_ got: Double?, _ want: Double?, tolerance: Double, _ what: String,
                             file: StaticString = #filePath, line: UInt = #line) {
        guard let want else { return XCTAssertNil(got, what, file: file, line: line) }
        guard let got else { return XCTFail("\(what): got nil, want \(want)", file: file, line: line) }
        let scale = Swift.max(1.0, Swift.abs(want))
        XCTAssertLessThanOrEqual(Swift.abs(got - want) / scale, tolerance,
                                 "\(what): got \(got), want \(want)", file: file, line: line)
    }

    // MARK: - oracles

    private func oracleVariance(_ values: [Double], ddof: Int) -> Double? {
        guard values.count > ddof else { return nil }
        let mean = values.reduce(0, +) / Double(values.count)
        let ss = values.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) }
        return ss / Double(values.count - ddof)
    }

    private func oracleQuantile(_ values: [Double], _ q: Double) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let position = q * Double(sorted.count - 1)
        let low = Int(position.rounded(.down)), high = Int(position.rounded(.up))
        if low == high { return sorted[low] }
        return sorted[low] + (sorted[high] - sorted[low]) * (position - Double(low))
    }

    // MARK: - product

    func testProductAgainstSwift() throws {
        try requireRealGPU()
        for n in [0, 1, 33, 4097] {
            // Small factors keep the oracle exact; the point is the reduction, not overflow.
            var values: [Int32?] = []
            for i in 0..<n { values.append(i % 3 == 1 ? nil : Int32((i % 5) - 2)) }
            let a = try MetalArray<Int32>(values)
            var want: Int64 = 1
            var any = false
            for v in values.compactMap({ $0 }) { want &*= Int64(v); any = true }
            if any { XCTAssertEqual(try a.product(), .int(want), "int32 product n=\(n)") }
            else { XCTAssertNil(try a.product(), "int32 product n=\(n)") }

            var floats: [Float?] = []
            for i in 0..<n { floats.append(i % 3 == 1 ? nil : Float(1.0 + Double(i % 7) / 64.0)) }
            let f = try MetalArray<Float>(floats)
            var wantF = 1.0
            for v in floats.compactMap({ $0 }) { wantF *= Double(v) }
            // A float32 product of thousands of factors reassociates across threads: the tolerance is
            // the accumulated rounding of the multiplications, not the 1e-6 the statistics hold to.
            if n > 0 { assertClose(try f.product()?.asDouble, wantF, tolerance: 1e-4, "float32 product n=\(n)") }
            else { XCTAssertNil(try f.product()) }

            var ds: [Double?] = []
            for i in 0..<n { ds.append(i % 3 == 1 ? nil : 1.0 + Double(i % 7) / 64.0) }
            let d = try MetalArray<Double>(ds)
            var wantD = 1.0
            for v in ds.compactMap({ $0 }) { wantD *= v }
            if n > 0 { assertClose(try d.product()?.asDouble, wantD, tolerance: 1e-12, "float64 product n=\(n)") }
            else { XCTAssertNil(try d.product()) }
        }
    }

    func testProductWrapsLikeArrow() throws {
        try requireRealGPU()
        let a = try MetalArray<Int64>([Int64.max, 3, 5])
        XCTAssertEqual(try a.product(), .int(Int64.max &* 3 &* 5))
        let u = try MetalArray<UInt32>([4_000_000_000, 3])
        XCTAssertEqual(try u.product(), .uint(4_000_000_000 &* 3))
    }

    // MARK: - variance / stddev

    func testVarianceAndStddevAgainstSwift() throws {
        try requireRealGPU()
        for n in AggregateTests.sizes {
            for nulls in [false, true] {
                let values = doubles(n, nulls: nulls)
                let valid = values.compactMap { $0 }
                let d = try MetalArray<Double>(values)
                assertClose(try d.variance(), oracleVariance(valid, ddof: 0), tolerance: 1e-12,
                            "float64 variance n=\(n) nulls=\(nulls)")
                assertClose(try d.variance(ddof: 1), oracleVariance(valid, ddof: 1), tolerance: 1e-12,
                            "float64 sample variance n=\(n) nulls=\(nulls)")
                assertClose(try d.stddev(), oracleVariance(valid, ddof: 0).map { $0.squareRoot() }, tolerance: 1e-12,
                            "float64 stddev n=\(n) nulls=\(nulls)")
                assertClose(try d.stddev(ddof: 1), oracleVariance(valid, ddof: 1).map { $0.squareRoot() }, tolerance: 1e-12,
                            "float64 sample stddev n=\(n) nulls=\(nulls)")

                let f = try MetalArray<Float>(values.map { $0.map(Float.init) })
                let validF = values.compactMap { $0 }.map { Double(Float($0)) }
                assertClose(try f.variance(), oracleVariance(validF, ddof: 0), tolerance: 1e-6,
                            "float32 variance n=\(n) nulls=\(nulls)")
                assertClose(try f.stddev(ddof: 1), oracleVariance(validF, ddof: 1).map { $0.squareRoot() },
                            tolerance: 1e-6, "float32 sample stddev n=\(n) nulls=\(nulls)")

                let intValues = ints(n, nulls: nulls)
                let i64 = try MetalArray<Int64>(intValues)
                let validI = intValues.compactMap { $0 }.map(Double.init)
                assertClose(try i64.variance(), oracleVariance(validI, ddof: 0), tolerance: 1e-6,
                            "int64 variance n=\(n) nulls=\(nulls)")
                assertClose(try i64.stddev(), oracleVariance(validI, ddof: 0).map { $0.squareRoot() },
                            tolerance: 1e-6, "int64 stddev n=\(n) nulls=\(nulls)")
            }
        }
        // Fewer values than degrees of freedom: null, as Arrow returns.
        XCTAssertNil(try MetalArray<Double>([1.0]).variance(ddof: 1))
        XCTAssertNil(try MetalArray<Double>([Double?]()).variance())
        XCTAssertNil(try MetalArray<Double>([nil, nil]).stddev())
    }

    // MARK: - min_max

    func testMinMaxInOnePass() throws {
        try requireRealGPU()
        for n in AggregateTests.sizes {
            let values = doubles(n, nulls: true)
            let valid = values.compactMap { $0 }
            let d = try MetalArray<Double>(values)
            let got = try d.minMax()
            if valid.isEmpty {
                XCTAssertNil(got, "float64 min_max n=\(n)")
            } else {
                XCTAssertEqual(got?.min, valid.min(), "float64 min n=\(n)")
                XCTAssertEqual(got?.max, valid.max(), "float64 max n=\(n)")
                XCTAssertEqual(got?.min, try d.min(), "min_max agrees with min n=\(n)")
                XCTAssertEqual(got?.max, try d.max(), "min_max agrees with max n=\(n)")
            }
            let intValues = ints(n, nulls: true)
            let i = try MetalArray<Int64>(intValues)
            let validI = intValues.compactMap { $0 }
            XCTAssertEqual(try i.minMax()?.min, validI.min(), "int64 min n=\(n)")
            XCTAssertEqual(try i.minMax()?.max, validI.max(), "int64 max n=\(n)")

            let u = try MetalArray<UInt64>(intValues.map { $0.map { UInt64(bitPattern: $0) } })
            let validU = intValues.compactMap { $0 }.map { UInt64(bitPattern: $0) }
            XCTAssertEqual(try u.minMax()?.min, validU.min(), "uint64 min n=\(n)")
            XCTAssertEqual(try u.minMax()?.max, validU.max(), "uint64 max n=\(n)")
        }
        // NaN is skipped, as in min / max.
        let withNaN = try MetalArray<Float>([1, .nan, -2, 3])
        XCTAssertEqual(try withNaN.minMax()?.min, -2)
        XCTAssertEqual(try withNaN.minMax()?.max, 3)
        XCTAssertNil(try MetalArray<Float>([.nan, .nan]).minMax())
    }

    // MARK: - quantile / median / mode / count_distinct

    func testQuantileAndMedianAreExact() throws {
        try requireRealGPU()
        for n in AggregateTests.sizes {
            let values = doubles(n, nulls: true)
            let valid = values.compactMap { $0 }
            let d = try MetalArray<Double>(values)
            for q in [0.0, 0.25, 0.5, 0.75, 1.0] {
                assertClose(try d.quantile(q), oracleQuantile(valid, q), tolerance: 1e-12,
                            "float64 quantile \(q) n=\(n)")
            }
            assertClose(try d.approximateMedian(), oracleQuantile(valid, 0.5), tolerance: 1e-12,
                        "float64 median n=\(n)")

            let intValues = ints(n, nulls: true)
            let i = try MetalArray<Int64>(intValues)
            assertClose(try i.approximateMedian(), oracleQuantile(intValues.compactMap { $0 }.map(Double.init), 0.5),
                        tolerance: 1e-9, "int64 median n=\(n)")
        }
        XCTAssertNil(try MetalArray<Int32>([Int32?]()).quantile(0.5))
        XCTAssertNil(try MetalArray<Int32>([nil, nil]).approximateMedian())
        // Clamped, and exact on an even count (the midpoint interpolates).
        let four = try MetalArray<Int32>([1, 2, 3, 4])
        XCTAssertEqual(try four.quantile(0.5), 2.5)
        XCTAssertEqual(try four.quantile(-1), 1)
        XCTAssertEqual(try four.quantile(2), 4)
    }

    func testModeAndCountDistinct() throws {
        try requireRealGPU()
        for n in [0, 1, 33, 4097, 300_003] {
            // Small alphabet so the mode is unambiguous and easy to count.
            var values: [Int32?] = []
            for i in 0..<n { values.append(i % 4 == 1 ? nil : Int32((i * i) % 7)) }
            let a = try MetalArray<Int32>(values)
            var histogram: [Int32: Int64] = [:]
            for v in values.compactMap({ $0 }) { histogram[v, default: 0] += 1 }
            let best = histogram.max { ($0.value, $1.key) < ($1.value, $0.key) }
            if let best {
                let got = try XCTUnwrap(try a.mode(), "mode n=\(n)")
                XCTAssertEqual(got.count, best.value, "mode count n=\(n)")
                XCTAssertEqual(got.value, best.key, "mode value n=\(n)")
            } else {
                XCTAssertNil(try a.mode(), "mode n=\(n)")
            }
            XCTAssertEqual(try a.countDistinct(), histogram.count, "count_distinct n=\(n)")
        }
        // Ties go to the smaller value.
        XCTAssertEqual(try MetalArray<Int32>([5, 1, 5, 1, 9]).mode()?.value, 1)
    }

    // MARK: - first / last / index

    func testFirstLastAndIndex() throws {
        try requireRealGPU()
        for n in AggregateTests.sizes {
            let values = ints(n, nulls: true)
            let a = try MetalArray<Int64>(values)
            XCTAssertEqual(try a.first(), values.compactMap { $0 }.first, "first n=\(n)")
            XCTAssertEqual(try a.last(), values.compactMap { $0 }.last, "last n=\(n)")
            XCTAssertEqual(try a.first(skipNulls: false), n > 0 ? values[0] : nil, "first keep-nulls n=\(n)")
            XCTAssertEqual(try a.last(skipNulls: false), n > 0 ? values[n - 1] : nil, "last keep-nulls n=\(n)")
            if let target = values.compactMap({ $0 }).last {
                let want = Int64(values.firstIndex { $0 == target } ?? -1)
                XCTAssertEqual(try a.index(of: target), want, "index n=\(n)")
            }
            XCTAssertEqual(try a.index(of: 4_000_000_007), -1, "absent index n=\(n)")
        }
        // A null row never matches, and NaN equals nothing.
        let f = try MetalArray<Float>([nil, 2, .nan, 2])
        XCTAssertEqual(try f.index(of: 2), 1)
        XCTAssertEqual(try f.index(of: .nan), -1)
        XCTAssertEqual(try MetalArray<Float>([-0.0, 0.0]).index(of: 0.0), 0)
        XCTAssertNil(try MetalArray<Int32>([nil, nil]).first())
        XCTAssertNil(try MetalArray<Int32>([nil, nil]).last())
    }

    // MARK: - any / all

    func testBooleanAnyAllOnTheGPU() throws {
        try requireRealGPU()
        for n in AggregateTests.sizes {
            for pattern in 0..<4 {
                var values: [Bool?] = []
                for i in 0..<n {
                    switch pattern {
                    case 0: values.append(true)
                    case 1: values.append(false)
                    case 2: values.append(i % 5 == 0)
                    default: values.append(i % 3 == 1 ? nil : (i % 7 == 0))
                    }
                }
                let a = try Self.booleans(values)
                let valid = values.compactMap { $0 }
                XCTAssertEqual(try a.anyTrue(), valid.contains(true), "any n=\(n) pattern=\(pattern)")
                XCTAssertEqual(try a.allTrue(), !valid.contains(false), "all n=\(n) pattern=\(pattern)")
                XCTAssertEqual(try a.trueAndValidCounts().trueCount, valid.filter { $0 }.count,
                               "trueCount n=\(n) pattern=\(pattern)")
                XCTAssertEqual(try a.trueAndValidCounts().validCount, valid.count, "validCount n=\(n)")
                // The GPU form agrees with the host popcount properties.
                XCTAssertEqual(try a.anyTrue(), a.any, "any matches the host form n=\(n)")
                XCTAssertEqual(try a.allTrue(), a.all, "all matches the host form n=\(n)")
            }
        }
    }

    // MARK: - run-end encoding

    /// Values with long runs, nulls included, so the encoder has something to collapse.
    private func runValues(_ n: Int) -> [Int32?] {
        var out: [Int32?] = []
        out.reserveCapacity(n)
        var i = 0
        while out.count < n {
            let runLength = 1 + (i * 7) % 13
            let value: Int32? = (i % 5 == 3) ? nil : Int32((i * 3) % 11)
            for _ in 0..<runLength where out.count < n { out.append(value) }
            i += 1
        }
        return out
    }

    func testRunEndEncodeDecodeRoundTrip() throws {
        try requireRealGPU()
        for n in AggregateTests.sizes {
            let values = runValues(n)
            let column = AnyMetalArray.int32(try MetalArray<Int32>(values))
            let encoded = try column.runEndEncode()
            XCTAssertEqual(encoded.length, n, "logical length n=\(n)")
            XCTAssertEqual(encoded.arrowFormat, "+r")
            // Runs really are collapsed, and the run ends are strictly increasing.
            let (runEnds, runValuesArray) = try XCTUnwrap(encoded.asRunEndEncoded)
            XCTAssertEqual(runEnds.length, runValuesArray.length, "one value per run n=\(n)")
            var previous: Int32 = 0
            for j in 0..<runEnds.length {
                let end = try XCTUnwrap(runEnds[j])
                XCTAssertGreaterThan(end, previous, "run ends increase n=\(n)")
                previous = end
            }
            if n > 100 { XCTAssertLessThan(runEnds.length, n, "runs are collapsed n=\(n)") }
            XCTAssertEqual(try encoded.runEndDecode().asInt32?.toArray(), values, "decode n=\(n)")
            XCTAssertEqual(encoded.nullCount, values.filter { $0 == nil }.count, "null count n=\(n)")
        }
        // Every element distinct: one run each, and the decode is still the identity.
        let distinct = AnyMetalArray.int64(try MetalArray<Int64>((0..<257).map { Int64($0) }))
        let encoded = try distinct.runEndEncode()
        XCTAssertEqual(encoded.runCount, 257)
        XCTAssertEqual(try encoded.runEndDecode().asInt64?.toArray(), (0..<257).map { Int64($0) })
    }

    func testRunEndEncodedCDataRoundTrip() throws {
        try requireRealGPU()
        for n in [0, 1, 33, 4097] {
            let values = runValues(n)
            let encoded = try AnyMetalArray.int32(try MetalArray<Int32>(values)).runEndEncode()
            var schema = ArrowSchema(), array = ArrowArray()
            encoded.exportArrowSchema(name: "ree", into: &schema)
            encoded.exportArrowArray(into: &array)
            XCTAssertEqual(String(cString: schema.format!), "+r")
            XCTAssertEqual(schema.n_children, 2)
            XCTAssertEqual(array.length, Int64(n))
            let imported = try importArrowArray(schema: &schema, array: &array).array
            schema.release?(&schema)
            XCTAssertEqual(imported.arrowFormat, "+r", "round trip n=\(n)")
            XCTAssertEqual(imported.length, n)
            XCTAssertEqual(try imported.runEndDecode().asInt32?.toArray(), values, "decoded round trip n=\(n)")
        }
    }

    func testRunEndEncodedSelection() throws {
        try requireRealGPU()
        let values = runValues(4097)
        let plain = try MetalArray<Int32>(values)
        let encoded = try AnyMetalArray.int32(plain).runEndEncode()
        // take, filter and slice all decode first, so they must match the plain column exactly.
        let indices = try MetalArray<Int32>((0..<300).map { Int32(($0 * 13) % 4097) })
        XCTAssertEqual(try encoded.take(indices).asInt32?.toArray(), try plain.take(indices).toArray())
        let mask = try plain.compare(.gt, 4)
        XCTAssertEqual(try encoded.filter(mask).asInt32?.toArray(), try plain.filter(mask).toArray())
        XCTAssertEqual(try encoded.slice(offset: 64, length: 500).asInt32?.toArray(),
                       try plain.slice(offset: 64, length: 500).toArray())
        // Temporal and boolean columns encode too.
        let stamps = AnyMetalArray.temporal(try MetalTemporalArray(type: .timestamp(.micro, timezone: nil),
                                                                   try MetalArray<Int64>([5, 5, 5, nil, nil, 7])))
        XCTAssertEqual(try stamps.runEndEncode().runCount, 3)
        XCTAssertEqual(try stamps.runEndEncode().runEndDecode().asTemporal?.toArray(), [5, 5, 5, nil, nil, 7])
        let flags = AnyMetalArray.boolean(try AggregateTests.booleans([true, true, false, nil, false]))
        XCTAssertEqual(try flags.runEndEncode().runEndDecode().asBoolean?.toArray(), [true, true, false, nil, false])
    }

    // MARK: - dictionary compute

    /// A dictionary column and the same data decoded, so every function can be checked both ways.
    private func dictionaryPair(_ n: Int) throws -> (dictionary: AnyMetalArray, decoded: MetalArray<Int32>) {
        var values: [Int32?] = []
        for i in 0..<n { values.append(i % 7 == 2 ? nil : Int32((i * 31) % 23)) }
        let plain = try MetalArray<Int32>(values)
        let dictionary = try AnyMetalArray.int32(plain).dictionaryEncoded()
        return (dictionary, plain)
    }

    func testDictionaryComputeMatchesDecodedCompute() throws {
        try requireRealGPU()
        for n in AggregateTests.sizes {
            let (dictionary, plain) = try dictionaryPair(n)
            XCTAssertEqual(dictionary.length, n)
            XCTAssertEqual(try dictionary.decode().asInt32?.toArray(), plain.toArray(), "decode n=\(n)")

            // compare on the dictionary vs compare on the decoded column
            for op in CompareOp.allCases {
                let onCodes = try dictionary.dictionaryCompare(op, Int32(11))
                let onValues = try plain.compare(op, 11)
                XCTAssertEqual(onCodes.toArray(), onValues.toArray(), "\(op.rawValue) n=\(n)")
            }

            // filter and take touch the codes only, and the values array is shared, not copied
            let mask = try plain.compare(.lt, 11)
            let filtered = try dictionary.filter(mask)
            XCTAssertEqual(try filtered.decode().asInt32?.toArray(), try plain.filter(mask).toArray(), "filter n=\(n)")
            XCTAssertEqual(filtered.asDictionary?.values.length, dictionary.asDictionary?.values.length,
                           "filter keeps the dictionary n=\(n)")
            if n > 0 {
                let indices = try MetalArray<Int32>((0..<Swift.min(n, 128)).map { Int32(($0 * 7) % n) })
                XCTAssertEqual(try dictionary.take(indices).decode().asInt32?.toArray(),
                               try plain.take(indices).toArray(), "take n=\(n)")
                XCTAssertEqual(try dictionary.slice(offset: 0, length: Swift.min(n, 64)).decode().asInt32?.toArray(),
                               try plain.slice(offset: 0, length: Swift.min(n, 64)).toArray(), "slice n=\(n)")
            }

            // unique and value_counts on the codes
            XCTAssertEqual(try dictionary.dictionaryUnique().asInt32?.toArray(), try plain.unique().toArray(),
                           "unique n=\(n)")
            let dictionaryCounts = try dictionary.dictionaryValueCounts()
            let plainCounts = try plain.valueCounts()
            XCTAssertEqual(dictionaryCounts.values.asInt32?.toArray(), plainCounts.values.toArray(), "value_counts n=\(n)")
            XCTAssertEqual(dictionaryCounts.counts.toArray(), plainCounts.counts.toArray(), "value_counts counts n=\(n)")

            // group_by over the codes vs group_by over the decoded column
            let (dictionaryGroups, dictionaryValues) = try dictionary.dictionaryGroupBy()
            let payload = try MetalArray<Int64>((0..<n).map { Int64($0 % 5) })
            let (plainGroups, plainKeys) = try plain.groupBy()
            XCTAssertEqual(dictionaryValues.asInt32?.toArray(), plainKeys.toArray(), "group keys n=\(n)")
            XCTAssertEqual(try dictionaryGroups.sum(payload).toArray(), try plainGroups.sum(payload).toArray(),
                           "grouped sum n=\(n)")
        }
    }

    func testDictionaryEncodeForTemporalAndBinary() throws {
        try requireRealGPU()
        let stamps = try MetalTemporalArray(type: .timestamp(.milli, timezone: "UTC"),
                                            try MetalArray<Int64>([7, 3, 7, nil, 3, 7]))
        let encoded = try AnyMetalArray.temporal(stamps).dictionaryEncoded()
        let pair = try XCTUnwrap(encoded.asDictionary)
        XCTAssertEqual(pair.values.asTemporal?.toArray(), [3, 7])
        XCTAssertEqual(pair.values.asTemporal?.type, .timestamp(.milli, timezone: "UTC"))
        XCTAssertEqual(pair.codes.toArray(), [1, 0, 1, nil, 0, 1])
        XCTAssertEqual(try encoded.decode().asTemporal?.toArray(), [7, 3, 7, nil, 3, 7])
        // Compare without decoding, against the storage integers.
        XCTAssertEqual(try encoded.dictionaryCompare(.gt, Int64(5)).toArray(), [true, false, true, nil, false, true])

        let blobs = try MetalStringArray(bytes: [[1, 2], [3], [1, 2], nil, [3]])
        let encodedBinary = try AnyMetalArray.binary(blobs).dictionaryEncoded()
        let binaryPair = try XCTUnwrap(encodedBinary.asDictionary)
        XCTAssertEqual(binaryPair.values.asBinary?.toByteArrays(), [[1, 2], [3]])
        XCTAssertEqual(binaryPair.codes.toArray(), [0, 1, 0, nil, 1])
        XCTAssertEqual(try encodedBinary.decode().asBinary?.toByteArrays(), [[1, 2], [3], [1, 2], nil, [3]])

        let words = AnyMetalArray.string(try MetalStringArray(["b", "a", "b", nil]))
        let encodedWords = try words.dictionaryEncoded()
        XCTAssertEqual(try encodedWords.dictionaryCompare(.eq, "b").toArray(), [true, false, true, nil])
        XCTAssertEqual(try encodedWords.dictionaryCompare(.ne, "b").toArray(), [false, true, false, nil])
        XCTAssertEqual(try encodedWords.decode().asString?.toArray(), ["b", "a", "b", nil])
    }

    // MARK: - grouped aggregates

    func testGroupedAggregatesAgainstSwift() throws {
        try requireRealGPU()
        for n in [0, 1, 33, 4097, 300_003] {
            let keyCount = 7
            var keys: [Int32?] = []
            var values: [Int32?] = []
            for i in 0..<n {
                keys.append(i % 11 == 4 ? nil : Int32(i % keyCount))
                values.append(i % 5 == 2 ? nil : Int32((i * 13) % 97) - 40)
            }
            let keyArray = try MetalArray<Int32>(keys)
            let valueArray = try MetalArray<Int32>(values)
            let groups = try GroupBy(keys: keyArray, keyCount: keyCount)

            // Swift oracle: the rows of each key, in order.
            var perKey = [[Int32]](repeating: [], count: keyCount)
            var firstRow = [Int32?](repeating: nil, count: keyCount)
            var lastRow = [Int32?](repeating: nil, count: keyCount)
            for i in 0..<n {
                guard let k = keys[i], let v = values[i] else { continue }
                perKey[Int(k)].append(v)
                if firstRow[Int(k)] == nil { firstRow[Int(k)] = v }
                lastRow[Int(k)] = v
            }

            XCTAssertEqual(try groups.first(valueArray).toArray(), firstRow, "hash_first n=\(n)")
            XCTAssertEqual(try groups.last(valueArray).toArray(), lastRow, "hash_last n=\(n)")

            let products = try groups.product(valueArray)
            for k in 0..<keyCount {
                if perKey[k].isEmpty {
                    XCTAssertNil(products[k], "hash_product empty key \(k) n=\(n)")
                } else {
                    var want: Int64 = 1
                    for v in perKey[k] { want &*= Int64(v) }
                    XCTAssertEqual(products[k], want, "hash_product key \(k) n=\(n)")
                }
            }

            let variances = try groups.variance(valueArray)
            let deviations = try groups.stddev(valueArray, ddof: 1)
            let distinct = try groups.countDistinct(valueArray)
            for k in 0..<keyCount {
                let rows = perKey[k].map(Double.init)
                assertClose(variances[k], oracleVariance(rows, ddof: 0), tolerance: 1e-6,
                            "hash_variance key \(k) n=\(n)")
                assertClose(deviations[k], oracleVariance(rows, ddof: 1).map { $0.squareRoot() }, tolerance: 1e-6,
                            "hash_stddev key \(k) n=\(n)")
                XCTAssertEqual(distinct[k], Int64(Set(perKey[k]).count), "hash_count_distinct key \(k) n=\(n)")
            }

            // Booleans: any / all per key.
            var flags: [Bool?] = []
            for i in 0..<n { flags.append(i % 5 == 2 ? nil : (i % 9 == 0)) }
            let flagArray = try AggregateTests.booleans(flags)
            let anyPerKey = try groups.any(flagArray)
            let allPerKey = try groups.all(flagArray)
            var trues = [Int](repeating: 0, count: keyCount), valid = [Int](repeating: 0, count: keyCount)
            for i in 0..<n {
                guard let k = keys[i], let f = flags[i] else { continue }
                valid[Int(k)] += 1
                if f { trues[Int(k)] += 1 }
            }
            for k in 0..<keyCount {
                if valid[k] == 0 {
                    XCTAssertNil(anyPerKey[k], "hash_any empty key \(k) n=\(n)")
                    XCTAssertNil(allPerKey[k], "hash_all empty key \(k) n=\(n)")
                } else {
                    XCTAssertEqual(anyPerKey[k], trues[k] > 0, "hash_any key \(k) n=\(n)")
                    XCTAssertEqual(allPerKey[k], trues[k] == valid[k], "hash_all key \(k) n=\(n)")
                }
            }
        }
        // The grouped median used to throw; the segmented sort in Kernels/AggregatesExtra.swift now
        // answers it exactly (AggregatesExtraTests covers it against a per-key oracle at scale).
        let groups = try GroupBy(keys: try MetalArray<Int32>([0, 0, 1]), keyCount: 2)
        XCTAssertEqual(try groups.approximateMedian(try MetalArray<Int32>([1, 4, 9])).toArray(), [2.5, 9])
    }

    /// A nullable boolean array (`MetalBooleanArray` only builds non-null ones from Swift values).
    static func booleans(_ values: [Bool?]) throws -> MetalBooleanArray {
        let n = values.count
        let bits = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: n))
        let validity = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: n))
        let v = bits.mutableTyped(UInt8.self), b = validity.mutableTyped(UInt8.self)
        var nulls = 0
        for (i, value) in values.enumerated() {
            guard let value else { nulls += 1; continue }
            Bitmap.set(b, i)
            if value { Bitmap.set(v, i) }
        }
        return MetalBooleanArray(length: n, nullCount: nulls, validity: nulls == 0 ? nil : validity, values: bits)
    }
}
