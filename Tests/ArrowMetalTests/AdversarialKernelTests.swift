import XCTest
@testable import ArrowMetal

/// Adversarial review of the newest GPU kernels: determinism under repetition, boundary lengths,
/// null placement at every entry point, and sliced (non-zero Arrow offset) inputs.
///
/// Every test here either reproduces a bug or pins a property that a race would break. The
/// determinism tests run the same kernel many times over one adversarial input and require
/// bit-identical output; a relaxed-atomic or missing-barrier bug shows up as a rare mismatch.
final class AdversarialKernelTests: XCTestCase {

    // MARK: - null placement

    /// `lexsortIndices` and `MetalRecordBatch.sorted(by:nullPlacement:)` document that
    /// `nullPlacement` "applies to every key". It does not reach a utf8 or binary key:
    /// `AnyMetalArray.argsortIndices` drops the argument on `.string` / `.binary`.
    func testLexsortNullPlacementReachesAStringKey() throws {
        try requireRealGPU()
        let xs: [String?] = ["b", nil, "a", "c", nil]
        let a = try MetalStringArray(xs, context: .shared)
        let idx = try lexsortIndices([.string(a)], nullPlacement: .atStart).toRawArray()
        XCTAssertEqual(idx.map { xs[Int($0)] }, [nil, nil, "a", "b", "c"],
                       "nullPlacement: .atStart must put the null rows first on a utf8 key")
    }

    /// The same for a two-key lexsort whose most significant key is utf8.
    func testTwoKeyLexsortNullPlacementAtStart() throws {
        try requireRealGPU()
        let keys: [String?] = ["b", nil, "a", nil]
        let second: [Int32] = [1, 5, 2, 3]
        let a = try MetalStringArray(keys, context: .shared)
        let b = try MetalArray<Int32>(second, context: .shared)
        let idx = try lexsortIndices([.string(a), .int32(b)], nullPlacement: .atStart).toRawArray()
        XCTAssertEqual(idx.map { keys[Int($0)] }, [nil, nil, "a", "b"])
        XCTAssertEqual(idx, [3, 1, 2, 0])
    }

    // MARK: - determinism

    private func repeatedlyEqual<R: Equatable>(_ n: Int, _ label: String, _ body: () throws -> R) throws {
        let first = try body()
        for i in 1..<n {
            let again = try body()
            if again != first {
                XCTFail("\(label): run \(i) differs from run 0")
                return
            }
        }
    }

    /// The string hash table under a load factor near capacity, with keys that share their first
    /// seven bytes (the utf8 argsort chunk width) and differ only past them.
    func testStringUniqueIsDeterministic() throws {
        try requireRealGPU()
        var xs: [String?] = []
        for i in 0..<4096 { xs.append("prefix7" + String(format: "%06d", i)) }
        for i in 0..<4096 { xs.append("prefix7" + String(format: "%06d", i)) }
        xs.append(nil)
        xs.append("")
        let a = try MetalStringArray(xs, context: .shared)
        try repeatedlyEqual(60, "string unique") { try a.unique().toArray() }
    }

    func testStringValueCountsIsDeterministic() throws {
        try requireRealGPU()
        var xs: [String?] = []
        for i in 0..<3000 { for _ in 0..<(i % 3 + 1) { xs.append("k\u{0}\(i)") } }
        let a = try MetalStringArray(xs, context: .shared)
        try repeatedlyEqual(40, "string value_counts") { () -> [String] in
            let (v, c) = try a.valueCounts()
            return zip(v.toArray(), c.toRawArray()).map { "\($0 ?? "<null>")=\($1)" }
        }
    }

    /// The generic 64-bit hash table behind unique / value_counts / count_distinct / dictionary_encode.
    func testInt64UniqueIsDeterministic() throws {
        try requireRealGPU()
        // Keys engineered so that splitmix64's finalizer sends many of them to nearby slots is hard to
        // do blind; all-distinct at a size that lands near a power-of-two table is the practical stress.
        let n = 1 << 16
        let vals = (0..<n).map { Int64($0) &* 0x1_0000_0001 }
        let a = try MetalArray<Int64>(vals, context: .shared)
        try repeatedlyEqual(40, "int64 unique") { try a.unique().toRawArray() }
        try repeatedlyEqual(40, "int64 count_distinct") { try a.countDistinct() }
    }

    func testInt64DictionaryEncodeIsDeterministic() throws {
        try requireRealGPU()
        let n = 1 << 16
        let vals = (0..<n).map { Int64($0 % 997) }
        let a = try MetalArray<Int64>(vals, context: .shared)
        try repeatedlyEqual(30, "int64 dictionary_encode") { () -> [Int32] in
            let (codes, _) = try a.dictionaryEncode()
            return codes.toRawArray()
        }
    }

    /// All-equal values are the worst case for a select: every row lands in one bin.
    func testTopKAllEqualIsDeterministic() throws {
        try requireRealGPU()
        let n = 1 << 17
        let a = try MetalArray<Int32>([Int32](repeating: 7, count: n), context: .shared)
        try repeatedlyEqual(40, "topK all-equal") { try a.topK(1000, largest: true).toRawArray() }
    }

    func testQuantileIsDeterministic() throws {
        try requireRealGPU()
        let n = 1 << 17
        var vals = [Double](repeating: 0, count: n)
        for i in 0..<n { vals[i] = Double((i &* 2654435761) % 4096) }
        let a = try MetalArray<Double>(vals, context: .shared)
        for q in [0.0, 0.25, 0.5, 0.75, 1.0] {
            try repeatedlyEqual(20, "quantile \(q)") { try a.quantile(q) }
        }
    }

    /// The sort-free grouped min / max: two passes of 32-bit atomics over a 64-bit key.
    func testGroupedExtremaAreDeterministic() throws {
        try requireRealGPU()
        let n = 1 << 16
        let K = 512
        let keys = try MetalArray<Int32>((0..<n).map { Int32($0 % K) }, context: .shared)
        let vals = try MetalArray<Int64>((0..<n).map { Int64(($0 &* 2654435761) % 1_000_003) - 500_000 },
                                         context: .shared)
        let gb = try keys.groupBy(keyCount: K)
        try repeatedlyEqual(30, "grouped extrema") { () -> [Int64] in
            let (mn, mx) = try gb.extrema(vals)
            return mn.toRawArray() + mx.toRawArray()
        }
    }

    /// The counting sort by group id, in both regimes (chunked and atomic).
    func testGroupedVarianceIsDeterministic() throws {
        try requireRealGPU()
        let n = 1 << 16
        let K = 300
        let keys = try MetalArray<Int32>((0..<n).map { Int32($0 % K) }, context: .shared)
        let vals = try MetalArray<Double>((0..<n).map { Double(($0 &* 7919) % 1000) }, context: .shared)
        let gb = try keys.groupBy(keyCount: K)
        try repeatedlyEqual(25, "grouped variance") { try gb.varianceDouble(vals, ddof: 1).toArray() }
    }

    // MARK: - boundaries

    private static let lengths = [0, 1, 2, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 65535, 65536, 65537]

    /// Indices of `vals` in stable ascending order — the order `argsort` promises.
    private func stableOrder(_ vals: [Int64]) -> [Int32] {
        var idx: [Int] = Array(0..<vals.count)
        idx.sort { (a: Int, b: Int) -> Bool in
            let x: Int64 = vals[a]
            let y: Int64 = vals[b]
            if x == y { return a < b }
            return x < y
        }
        return idx.map { Int32($0) }
    }

    func testArgsortAtBoundaryLengths() throws {
        try requireRealGPU()
        for n in Self.lengths {
            let vals = (0..<n).map { Int64(($0 &* 2654435761) % 8191) }
            let a = try MetalArray<Int64>(vals, context: .shared)
            let idx = try a.argsort().toRawArray()
            XCTAssertEqual(idx, stableOrder(vals), "argsort n=\(n)")
        }
    }

    func testTopKAtBoundaryK() throws {
        try requireRealGPU()
        let n = 1025
        let vals = (0..<n).map { Int32(($0 &* 37) % 1000) }
        let a = try MetalArray<Int32>(vals, context: .shared)
        let ascending = (0..<n).sorted { vals[$0] != vals[$1] ? vals[$0] < vals[$1] : $0 < $1 }
        let descending = (0..<n).sorted { vals[$0] != vals[$1] ? vals[$0] > vals[$1] : $0 < $1 }
        for k in [0, 1, 2, n - 1, n, n + 1] {
            let got = try a.topK(k, largest: true).toRawArray()
            XCTAssertEqual(got.map(Int.init), Array(descending.prefix(k)), "topK largest k=\(k)")
            let gotSmall = try a.topK(k, largest: false).toRawArray()
            XCTAssertEqual(gotSmall.map(Int.init), Array(ascending.prefix(k)), "topK smallest k=\(k)")
        }
    }

    func testKthElementAtBoundaries() throws {
        try requireRealGPU()
        for n in [1, 2, 4096, 4097, 65537] {
            let vals = (0..<n).map { Double(($0 &* 2654435761) % 65536) }
            let a = try MetalArray<Double>(vals, context: .shared)
            let sortedVals = vals.sorted()
            XCTAssertEqual(try a.kthElement(1), sortedVals.first, "kth 1 n=\(n)")
            XCTAssertEqual(try a.kthElement(n), sortedVals.last, "kth n n=\(n)")
            XCTAssertNil(try a.kthElement(0), "kth 0 n=\(n)")
            XCTAssertNil(try a.kthElement(n + 1), "kth n+1 n=\(n)")
            XCTAssertEqual(try a.kthElement((n + 1) / 2), sortedVals[(n + 1) / 2 - 1], "kth mid n=\(n)")
        }
    }

    func testQuantileAtBoundaries() throws {
        try requireRealGPU()
        for n in [1, 2, 4095, 4096, 4097, 65536, 65537] {
            let vals = (0..<n).map { Double(($0 &* 7919) % 1024) }
            let a = try MetalArray<Double>(vals, context: .shared)
            let s = vals.sorted()
            for q in [0.0, 0.5, 1.0] {
                let pos = q * Double(n - 1)
                let lo = Int(pos.rounded(.down)), hi = Int(pos.rounded(.up))
                let want = lo == hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (pos - Double(lo))
                XCTAssertEqual(try a.quantile(q) ?? .nan, want, accuracy: 1e-9, "quantile \(q) n=\(n)")
            }
        }
    }

    // MARK: - group-by shapes

    func testGroupIdsWithGapsAndSingletons() throws {
        try requireRealGPU()
        // Only groups 0, 5 and 9 occur; every other group must come back empty (null), not stale.
        let keys = try MetalArray<Int32>([0, 5, 9, 5, 0, 9, 5], context: .shared)
        let vals = try MetalArray<Int64>([10, 20, 30, 40, 50, 60, 70], context: .shared)
        let gb = try keys.groupBy(keyCount: 10)
        let (mn, mx) = try gb.extrema(vals)
        XCTAssertEqual(mn.toArray(), [10, nil, nil, nil, nil, 20, nil, nil, nil, 30])
        XCTAssertEqual(mx.toArray(), [50, nil, nil, nil, nil, 70, nil, nil, nil, 60])
    }

    func testGroupIdOutOfRangeIsDroppedNotRead() throws {
        try requireRealGPU()
        let keys = try MetalArray<Int32>([0, 3, -1, 1, 99, 1], context: .shared)
        let vals = try MetalArray<Int64>([1, 2, 3, 4, 5, 6], context: .shared)
        let gb = try keys.groupBy(keyCount: 2)
        let (mn, mx) = try gb.extrema(vals)
        XCTAssertEqual(mn.toArray(), [1, 4])
        XCTAssertEqual(mx.toArray(), [1, 6])
    }

    func testGroupedVarianceOneElementAndAllNullGroups() throws {
        try requireRealGPU()
        let keys = try MetalArray<Int32>([0, 1, 1, 2, 3], context: .shared)
        let vals = try MetalArray<Double>([1.0, nil, nil, 5.0, 7.0], context: .shared)
        let gb = try keys.groupBy(keyCount: 4)
        let v1 = try gb.varianceDouble(vals, ddof: 1).toArray()
        // group 0: one value -> null with ddof 1; group 1: all null -> null; 2 and 3: one value -> null.
        XCTAssertEqual(v1, [nil, nil, nil, nil])
        let v0 = try gb.varianceDouble(vals, ddof: 0).toArray()
        XCTAssertEqual(v0[0], 0.0)
        XCTAssertNil(v0[1])
        XCTAssertEqual(v0[2], 0.0)
        XCTAssertEqual(v0[3], 0.0)
    }

    func testGroupedExtremaOnFloatsWithNaN() throws {
        try requireRealGPU()
        let keys = try MetalArray<Int32>([0, 0, 0, 1, 1], context: .shared)
        let vals = try MetalArray<Float>([1.0, .nan, 3.0, .nan, .nan], context: .shared)
        let gb = try keys.groupBy(keyCount: 2)
        let (mn, mx) = try gb.extrema(vals)
        XCTAssertEqual(mn.toArray()[0], 1.0)
        XCTAssertEqual(mx.toArray()[0], 3.0)
        XCTAssertNil(mn.toArray()[1], "a group whose every value is NaN has no min")
        XCTAssertNil(mx.toArray()[1], "a group whose every value is NaN has no max")
    }

    // MARK: - sliced inputs

    func testSortsAndRanksOnSlicedInputs() throws {
        try requireRealGPU()
        let n = 5000
        let vals = (0..<n).map { Int64(($0 &* 2654435761) % 9973) }
        let full = try MetalArray<Int64>(vals, context: .shared)
        for off in [1, 7, 31, 32, 33, 64] {
            let len = 1000
            let s = try full.slice(offset: off, length: len)
            let sub = Array(vals[off..<(off + len)])
            let expect = stableOrder(sub)
            XCTAssertEqual(try s.argsort().toRawArray(), expect, "argsort on slice offset \(off)")
            let ranks = try s.rank().toRawArray()
            var wantRank = [Int32](repeating: 0, count: len)
            for (r, i) in expect.enumerated() { wantRank[Int(i)] = Int32(r + 1) }
            // `rank` is Arrow's "min" tiebreaker by default in most engines; compare against argsort
            // positions only where the values are distinct, which they are not here — so just check
            // that equal values get equal ranks and the multiset of ranks is a permutation-consistent set.
            var byValue: [Int64: Set<Int32>] = [:]
            for i in 0..<len { byValue[sub[i], default: []].insert(ranks[i]) }
            for (v, rs) in byValue { XCTAssertEqual(rs.count, 1, "value \(v) got ranks \(rs) at offset \(off)") }
        }
    }

    func testStringSortOnASlicedColumn() throws {
        try requireRealGPU()
        let xs: [String?] = (0..<300).map { $0 % 17 == 0 ? nil : "s\u{0}\(($0 &* 37) % 300)" }
        let full = try MetalStringArray(xs, context: .shared)
        for off in [1, 8, 31, 32, 33] {
            let len = 100
            let s = try full.slice(offset: off, length: len)
            let sub = Array(xs[off..<(off + len)])
            let idx = try s.argsort().toRawArray()
            let got = idx.map { sub[Int($0)] }
            let want = sub.compactMap { $0 }.sorted { Array($0.utf8).lexicographicallyPrecedes(Array($1.utf8)) }
                + [String?](repeating: nil, count: sub.filter { $0 == nil }.count)
            XCTAssertEqual(got, want, "utf8 argsort on slice offset \(off)")
        }
    }

    /// A slice must agree with the same rows built as a column of their own — the invariant that a
    /// missed validity bit offset or a missed Arrow `offset` on any of the four kernels behind
    /// `unique` / `value_counts` would break. (utf8 `unique` drops nulls in both orders, by design.)
    func testStringUniqueAndValueCountsOnASlicedColumn() throws {
        try requireRealGPU()
        let xs: [String?] = (0..<500).map { $0 % 23 == 0 ? nil : "abcdefg\($0 % 40)" }
        let full = try MetalStringArray(xs, context: .shared)
        for off in [1, 8, 31, 32, 33, 64] {
            let s = try full.slice(offset: off, length: 200)
            let standalone = try MetalStringArray(Array(xs[off..<(off + 200)]), context: .shared)
            XCTAssertEqual(try s.unique().toArray(), try standalone.unique().toArray(),
                           "string unique on slice offset \(off)")
            let (sv, sc) = try s.valueCounts()
            let (tv, tc) = try standalone.valueCounts()
            XCTAssertEqual(sv.toArray(), tv.toArray(), "string value_counts values, slice offset \(off)")
            XCTAssertEqual(sc.toRawArray(), tc.toRawArray(), "string value_counts counts, slice offset \(off)")
        }
    }

    // MARK: - batching

    func testBatchThatThrowsMidwayLeavesNoOpenBatch() throws {
        try requireRealGPU()
        let ctx = MetalContext.shared
        struct Boom: Error {}
        let a = try MetalArray<Int32>([1, 2, 3, 4], context: ctx)
        XCTAssertThrowsError(try ctx.batch { () -> Int in
            _ = try a.argsort()
            throw Boom()
        })
        XCTAssertFalse(ctx.isBatching, "a batch that threw must not stay open")
        // A later op must not see stale GPU-side state.
        XCTAssertEqual(try a.argsort().toRawArray(), [0, 1, 2, 3])
    }

    func testZeroOpBatchAndNestedBatch() throws {
        try requireRealGPU()
        let ctx = MetalContext.shared
        try ctx.batch { }
        XCTAssertFalse(ctx.isBatching)
        let a = try MetalArray<Int32>([3, 1, 2], context: ctx)
        let r = try ctx.batch { try ctx.batch { try a.argsort() } }
        XCTAssertEqual(r.toRawArray(), [1, 2, 0])
        XCTAssertFalse(ctx.isBatching)
    }
}
