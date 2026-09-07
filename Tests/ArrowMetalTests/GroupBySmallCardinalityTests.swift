import XCTest
@testable import ArrowMetal

/// The grouped aggregates at the cardinalities and boundaries the small-group work moved.
///
/// Three things are pinned here. **The key mapping now reads the key column as the type it is** rather
/// than widening it to int64, so every integer width has to produce the same dense ids and the same
/// answers, sliced inputs included. **`sum` hands back the accumulator's own buffer with a GPU-built
/// bitmap** instead of copying it group by group on the host, so the null pattern of a key that counted
/// no valid value has to be exactly what it was. **`mean` divides on the GPU** from the counts the
/// accumulation already produced, so its answer has to be bit for bit `Double(sum) / Double(count)`.
///
/// The cardinalities are the ones the paths switch on: 1, 2, 1,000, 1,023 / 1,024 / 1,025 (the
/// threadgroup-private table in `GroupBySource.maxPrivateKeys` holds 1,024 keys and no more), and one
/// group per row.
final class GroupBySmallCardinalityTests: XCTestCase {

    /// (sum, count) per dense key from a Swift oracle. Null keys and keys outside `[0, K)` contribute
    /// nothing; a null value is counted by neither.
    private func oracle(_ keys: [Int32?], _ vals: [Int64?], _ K: Int) -> (sums: [Int64], counts: [Int]) {
        var sums = [Int64](repeating: 0, count: K), counts = [Int](repeating: 0, count: K)
        for (i, k) in keys.enumerated() {
            guard let k, k >= 0, Int(k) < K, let v = vals[i] else { continue }
            sums[Int(k)] = sums[Int(k)] &+ v
            counts[Int(k)] += 1
        }
        return (sums, counts)
    }

    private func check(_ label: String, keys: [Int32?], vals: [Int64?], K: Int) throws {
        let gb = try GroupBy(keys: try MetalArray<Int32>(keys), keyCount: K)
        let values = try MetalArray<Int64>(vals)
        let (sums, counts) = oracle(keys, vals, K)
        let gotSum = try gb.sum(values)
        let gotCount = try gb.count(values)
        let gotMean = try gb.mean(values)
        let gotMin = try gb.minMax(values).min
        let gotMax = try gb.minMax(values).max
        for k in 0..<K {
            let n = counts[k]
            if n == 0 {
                XCTAssertNil(gotSum[k], "\(label): sum of empty group \(k)")
                XCTAssertNil(gotMean[k], "\(label): mean of empty group \(k)")
                XCTAssertNil(gotMin[k], "\(label): min of empty group \(k)")
                XCTAssertNil(gotMax[k], "\(label): max of empty group \(k)")
            } else {
                XCTAssertEqual(gotSum[k], sums[k], "\(label): sum of group \(k)")
                // The mean is the host formula's bit pattern, not an approximation of it.
                let expected = Double(sums[k]) / Double(n)
                XCTAssertEqual(gotMean[k]?.bitPattern, expected.bitPattern, "\(label): mean of group \(k)")
                let members = keys.indices.filter { keys[$0] == Int32(k) && vals[$0] != nil }.map { vals[$0]! }
                XCTAssertEqual(gotMin[k], members.min(), "\(label): min of group \(k)")
                XCTAssertEqual(gotMax[k], members.max(), "\(label): max of group \(k)")
            }
            XCTAssertEqual(gotCount[k], Int64(n), "\(label): count of group \(k)")
        }
    }

    // MARK: - cardinalities, including both sides of the private-table threshold

    func testEveryCardinalityAroundThePrivateTableThreshold() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        // 1024 is `GroupBySource.maxPrivateKeys`: at or below it the accumulation uses a threadgroup
        // table, above it device atomics. 1023/1024/1025 straddle that; 1 and 2 are the degenerate ends.
        for K in [1, 2, 10, 1_000, 1_023, 1_024, 1_025, 4_096] {
            for n in [0, 1, 33, 5_000] {
                var keys: [Int32?] = [], vals: [Int64?] = []
                for _ in 0..<n {
                    keys.append(Int.random(in: 0..<10, using: &g) == 0 ? nil
                                : Int32.random(in: 0..<Int32(K), using: &g))
                    vals.append(Int.random(in: 0..<10, using: &g) == 0 ? nil
                                : Int64.random(in: -1_000_000...1_000_000, using: &g))
                }
                try check("K=\(K) n=\(n)", keys: keys, vals: vals, K: K)
            }
        }
    }

    func testOneGroupPerRow() throws {
        try requireRealGPU()
        for n in [1, 33, 1_024, 1_025, 20_000] {
            let keys: [Int32?] = (0..<n).map { Int32($0) }
            let vals: [Int64?] = (0..<n).map { Int64($0) * 7 - 11 }
            try check("one group per row n=\(n)", keys: keys, vals: vals, K: n)
        }
    }

    func testAllNullValuesLeaveEveryGroupNull() throws {
        try requireRealGPU()
        for K in [1, 2, 1_000, 1_025] {
            let n = 3 * K
            let keys: [Int32?] = (0..<n).map { Int32($0 % K) }
            let vals = [Int64?](repeating: nil, count: n)
            try check("all-null values K=\(K)", keys: keys, vals: vals, K: K)
            let gb = try GroupBy(keys: try MetalArray<Int32>(keys), keyCount: K)
            XCTAssertEqual(try gb.sum(try MetalArray<Int64>(vals)).nullCount, K)
            XCTAssertEqual(try gb.mean(try MetalArray<Int64>(vals)).nullCount, K)
            // `count()` counts rows, not values, so it is unaffected by the values being null.
            XCTAssertEqual(try gb.count().toArray().compactMap { $0 }.reduce(0, +), Int64(n))
        }
    }

    func testAllNullKeysProduceNoGroupAtAll() throws {
        try requireRealGPU()
        let keys = [Int32?](repeating: nil, count: 1_000)
        let vals: [Int64?] = (0..<1_000).map { Int64($0) }
        try check("all-null keys", keys: keys, vals: vals, K: 1_000)
    }

    // MARK: - the mapping stage: every integer width, same ids

    /// The range encoder is generated per key element type. Ids and answers must not depend on the width
    /// the caller happened to store the keys in.
    func testEveryIntegerKeyWidthMapsToTheSameGroups() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        let n = 30_000
        var raw: [Int64?] = []
        for _ in 0..<n {
            raw.append(Int.random(in: 0..<12, using: &g) == 0 ? nil : Int64.random(in: -120...120, using: &g))
        }
        let vals = try MetalArray<Int64>((0..<n).map { Int64($0 % 97) - 40 })

        func summary(_ column: AnyMetalArray) throws -> [String: Int64?] {
            let gbk = try GroupByKeys(columns: [column])
            let sums = try gbk.groupBy.sum(vals)
            let keys = try gbk.groupKeys()[0]
            var out: [String: Int64?] = [:]
            for gIdx in 0..<gbk.groupCount {
                let label: String
                switch keys {
                case .int8(let a): label = a[gIdx].map { "\($0)" } ?? "null"
                case .int16(let a): label = a[gIdx].map { "\($0)" } ?? "null"
                case .int32(let a): label = a[gIdx].map { "\($0)" } ?? "null"
                case .int64(let a): label = a[gIdx].map { "\($0)" } ?? "null"
                case .uint8(let a): label = a[gIdx].map { "\(Int64($0) - 128)" } ?? "null"
                case .uint16(let a): label = a[gIdx].map { "\(Int64($0) - 128)" } ?? "null"
                case .uint32(let a): label = a[gIdx].map { "\(Int64($0) - 128)" } ?? "null"
                case .uint64(let a): label = a[gIdx].map { "\(Int64($0) - 128)" } ?? "null"
                default: label = "?"
                }
                out[label] = sums[gIdx]
            }
            return out
        }

        let reference = try summary(.int64(try MetalArray<Int64>(raw)))
        XCTAssertGreaterThan(reference.count, 200)
        // Signed widths hold the values as they are; unsigned ones hold them shifted by 128, which the
        // labelling above undoes, so every column describes the same partition of the rows.
        try XCTAssertEqual(summary(.int8(try MetalArray<Int8>(raw.map { $0.map { Int8($0) } }))), reference)
        try XCTAssertEqual(summary(.int16(try MetalArray<Int16>(raw.map { $0.map { Int16($0) } }))), reference)
        try XCTAssertEqual(summary(.int32(try MetalArray<Int32>(raw.map { $0.map { Int32($0) } }))), reference)
        try XCTAssertEqual(summary(.uint8(try MetalArray<UInt8>(raw.map { $0.map { UInt8($0 + 128) } }))), reference)
        try XCTAssertEqual(summary(.uint16(try MetalArray<UInt16>(raw.map { $0.map { UInt16($0 + 128) } }))), reference)
        try XCTAssertEqual(summary(.uint32(try MetalArray<UInt32>(raw.map { $0.map { UInt32($0 + 128) } }))), reference)
        try XCTAssertEqual(summary(.uint64(try MetalArray<UInt64>(raw.map { $0.map { UInt64($0 + 128) } }))), reference)
    }

    /// A sliced key column: the typed kernels read the buffer at the slice's own offset, and an offset
    /// that is not a multiple of the element width is exactly where a vectorised read would go wrong.
    func testSlicedKeysAndValuesAgreeWithTheStandaloneRows() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        let n = 50_000
        var keys: [Int32?] = [], vals: [Int64?] = []
        for _ in 0..<n {
            keys.append(Int.random(in: 0..<9, using: &g) == 0 ? nil : Int32.random(in: 0..<997, using: &g))
            vals.append(Int.random(in: 0..<9, using: &g) == 0 ? nil : Int64.random(in: -9_999...9_999, using: &g))
        }
        let keyColumn = try MetalArray<Int32>(keys), valColumn = try MetalArray<Int64>(vals)
        for offset in [1, 7, 31, 32, 33, 63, 64] {
            let length = n - offset - 5
            let sliceKeys = try keyColumn.slice(offset: offset, length: length)
            let sliceVals = try valColumn.slice(offset: offset, length: length)
            let gbk = try GroupByKeys(columns: [.int32(sliceKeys)])
            let sums = try gbk.groupBy.sum(sliceVals)
            let rows = offset..<(offset + length)
            let standalone = try GroupByKeys(columns: [.int32(try MetalArray<Int32>(Array(keys[rows])))])
            let standaloneSums = try standalone.groupBy.sum(try MetalArray<Int64>(Array(vals[rows])))
            XCTAssertEqual(gbk.groupCount, standalone.groupCount, "offset \(offset)")
            XCTAssertEqual(sums.toArray(), standaloneSums.toArray(), "offset \(offset)")
        }
    }

    /// The range path is taken when the key span is small enough to scan; past that the column is
    /// sorted instead. Both sides of that decision have to give the same groups and the same sums.
    func testTheRangeAndSortMappingsAgreeAcrossTheSpanThreshold() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        let n = 20_000
        // `rangeIsWorthIt` takes the range path while the span is at most max(2^16, 4 * rows); with
        // 20,000 rows that is 80,000, so a stride of 3 stays inside it and a stride of 5,000 does not.
        for stride in [3, 5_000] {
            var keys: [Int64?] = [], vals: [Int64?] = []
            for i in 0..<n {
                keys.append(i % 11 == 0 ? nil : Int64((i % 40) * stride) - 20)
                vals.append(Int64(i % 13) - 6)
            }
            let span = 39 * stride + 1
            XCTAssertEqual(GroupByKeys.rangeIsWorthIt(span: span, rows: n), stride == 3,
                           "the span threshold moved; this test is picking the paths by hand")
            let gbk = try GroupByKeys(columns: [.int64(try MetalArray<Int64>(keys))])
            let sums = try gbk.groupBy.sum(try MetalArray<Int64>(vals))
            var expected: [Int64: Int64] = [:], nullSum: Int64 = 0
            for (i, k) in keys.enumerated() {
                if let k { expected[k, default: 0] &+= vals[i]! } else { nullSum &+= vals[i]! }
            }
            XCTAssertEqual(gbk.groupCount, expected.count + 1, "stride \(stride)")
            guard case .int64(let labels) = try gbk.groupKeys()[0] else { return XCTFail("key type") }
            for gIdx in 0..<gbk.groupCount {
                if let k = labels[gIdx] { XCTAssertEqual(sums[gIdx], expected[k], "stride \(stride) key \(k)") }
                else { XCTAssertEqual(sums[gIdx], nullSum, "stride \(stride) null group") }
            }
        }
    }

    /// `mean` reuses the accumulation's counts instead of running a second pass; the two have to agree
    /// on which keys are null and on every bit of the quotient, for signed and unsigned columns alike.
    func testMeanAgreesWithSumOverCountBitForBit() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        for K in [1, 1_000, 1_025] {
            let n = 40_000
            var keys: [Int32?] = [], signed: [Int64?] = [], unsigned: [UInt64?] = []
            for _ in 0..<n {
                keys.append(Int.random(in: 0..<8, using: &g) == 0 ? nil : Int32.random(in: 0..<Int32(K), using: &g))
                let v = Int64.random(in: -(1 << 60)...(1 << 60), using: &g)
                let keep = Int.random(in: 0..<8, using: &g) != 0
                signed.append(keep ? v : nil)
                unsigned.append(keep ? UInt64(bitPattern: v) : nil)
            }
            let gb = try GroupBy(keys: try MetalArray<Int32>(keys), keyCount: K)
            let s = try MetalArray<Int64>(signed), u = try MetalArray<UInt64>(unsigned)
            let sSum = try gb.sum(s), sCount = try gb.count(s), sMean = try gb.mean(s)
            let uSum = try gb.sumUnsigned(u), uCount = try gb.count(u), uMean = try gb.mean(u)
            for k in 0..<K {
                if sCount[k] == 0 {
                    XCTAssertNil(sMean[k]); XCTAssertNil(uMean[k])
                    continue
                }
                XCTAssertEqual(sMean[k]?.bitPattern, (Double(sSum[k]!) / Double(sCount[k]!)).bitPattern,
                               "signed mean K=\(K) group \(k)")
                XCTAssertEqual(uMean[k]?.bitPattern, (Double(uSum[k]!) / Double(uCount[k]!)).bitPattern,
                               "unsigned mean K=\(K) group \(k)")
            }
        }
    }
}
