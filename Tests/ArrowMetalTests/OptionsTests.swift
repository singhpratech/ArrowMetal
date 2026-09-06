import XCTest
@testable import ArrowMetal

/// The option surfaces the Arrow compute functions carry: `null_placement`, `tiebreaker`,
/// `null_matching_behavior`, the distinct-value order and the cast flags.
///
/// Every case is checked against a host reference computed straight from the input, at the sizes the
/// rest of the suite uses (0, 1, 33, 4097 and one million-plus run), with nulls sprinkled through.

/// A sort key that carries Arrow's null placement: `bucket` is -1 for a null when nulls go first,
/// +1 when they go last, and 0 for a value.
struct OrderKey: Comparable, Equatable {
    let bucket: Int
    let value: Int32
    static func < (a: OrderKey, b: OrderKey) -> Bool {
        a.bucket != b.bucket ? a.bucket < b.bucket : a.value < b.value
    }
}

final class OptionsTests: XCTestCase {

    static let sizes = [0, 1, 33, 4097]
    static let bigSize = 1_000_003

    /// Values with a healthy number of ties and roughly one null in seven.
    func sample(_ n: Int, nullEvery: Int = 7) -> [Int32?] {
        (0..<n).map { (i: Int) -> Int32? in i % nullEvery == 3 ? nil : Int32((i &* 7919) % 97) }
    }

    // MARK: - Reference implementations

    /// The permutation `argsort` should produce: a stable sort of the non-null rows, with the null rows
    /// as one block in their original order at whichever end.
    func expectedOrder(_ vals: [Int32?], descending: Bool, nullsFirst: Bool) -> [Int32] {
        let nonNull = vals.enumerated().filter { $0.element != nil }
        let sorted = nonNull.sorted { x, y in
            let a = x.element!, b = y.element!
            if a != b { return descending ? a > b : a < b }
            return x.offset < y.offset
        }.map { Int32($0.offset) }
        let nulls = vals.enumerated().filter { $0.element == nil }.map { Int32($0.offset) }
        return nullsFirst ? nulls + sorted : sorted + nulls
    }

    // MARK: - null_placement on the sorts

    func testArgsortNullPlacement() throws {
        try requireRealGPU()
        for n in Self.sizes {
            let vals = sample(n)
            let a = try MetalArray<Int32>(vals)
            for descending in [false, true] {
                for placement in NullPlacement.allCases {
                    let got = try a.argsort(descending: descending, nullPlacement: placement).toRawArray()
                    let want = expectedOrder(vals, descending: descending, nullsFirst: placement == .atStart)
                    XCTAssertEqual(got, want, "n=\(n) desc=\(descending) \(placement)")
                }
            }
        }
    }

    func testArgsortNullPlacementLarge() throws {
        try requireRealGPU()
        let vals = sample(Self.bigSize)
        let a = try MetalArray<Int32>(vals)
        for placement in NullPlacement.allCases {
            let got = try a.argsort(nullPlacement: placement).toRawArray()
            XCTAssertEqual(got, expectedOrder(vals, descending: false, nullsFirst: placement == .atStart))
        }
    }

    func testArgsortNullPlacementNoNulls() throws {
        try requireRealGPU()
        // With no validity bitmap at all the placement must simply not matter.
        let a = try MetalArray<Int32>((0..<257).map { (i: Int) -> Int32 in Int32((i &* 31) % 41) })
        let atEnd = try a.argsort(nullPlacement: .atEnd).toRawArray()
        let atStart = try a.argsort(nullPlacement: .atStart).toRawArray()
        XCTAssertEqual(atEnd, atStart)
    }

    func testLexsortNullPlacement() throws {
        try requireRealGPU()
        let n = 400
        let major: [Int32?] = (0..<n).map { (i: Int) -> Int32? in i % 11 == 2 ? nil : Int32(i % 5) }
        let minor: [Int32?] = (0..<n).map { (i: Int) -> Int32? in i % 13 == 4 ? nil : Int32((i &* 7) % 9) }
        let ma = AnyMetalArray.int32(try MetalArray<Int32>(major))
        let mi = AnyMetalArray.int32(try MetalArray<Int32>(minor))
        for placement in NullPlacement.allCases {
            let idx = try lexsortIndices([ma, mi], descending: [false, true], nullPlacement: placement).toRawArray()
            XCTAssertEqual(Set(idx).count, n, "permutation")
            // Verify the ordering pairwise rather than rebuilding the permutation.
            func key(_ v: Int32?, _ desc: Bool) -> OrderKey {
                guard let v else { return OrderKey(bucket: placement == .atStart ? -1 : 1, value: 0) }
                return OrderKey(bucket: 0, value: desc ? -v : v)
            }
            for i in 1..<n {
                let p = Int(idx[i - 1]), q = Int(idx[i])
                let kp = [key(major[p], false), key(minor[p], true)]
                let kq = [key(major[q], false), key(minor[q], true)]
                XCTAssertFalse(kq.lexicographicallyPrecedes(kp), "out of order at \(i) with \(placement)")
                if kp == kq { XCTAssertLessThan(p, q, "unstable at \(i)") }
            }
        }
    }

    // MARK: - rank tiebreakers and options

    /// `rank` on the host: the sorted order, then the tiebreaker's number for each tie group.
    func expectedRank(_ vals: [Int32?], tiebreaker: RankTiebreaker,
                      descending: Bool, nullsFirst: Bool) -> [Int32] {
        let order = expectedOrder(vals, descending: descending, nullsFirst: nullsFirst)
        var out = [Int32](repeating: 0, count: vals.count)
        var i = 0, dense: Int32 = 0
        while i < order.count {
            var j = i
            let v = vals[Int(order[i])]
            while j < order.count && vals[Int(order[j])] == v { j += 1 }
            dense += 1
            for k in i..<j {
                switch tiebreaker {
                case .min: out[Int(order[k])] = Int32(i + 1)
                case .max: out[Int(order[k])] = Int32(j)
                case .first: out[Int(order[k])] = Int32(k + 1)
                case .dense: out[Int(order[k])] = dense
                }
            }
            i = j
        }
        return out
    }

    func testRankOptions() throws {
        try requireRealGPU()
        for n in Self.sizes {
            let vals = sample(n)
            let a = try MetalArray<Int32>(vals)
            for tb in RankTiebreaker.allCases {
                for descending in [false, true] {
                    for placement in NullPlacement.allCases {
                        let got = try a.rank(tiebreaker: tb, descending: descending,
                                             nullPlacement: placement).toRawArray()
                        let want = expectedRank(vals, tiebreaker: tb, descending: descending,
                                                nullsFirst: placement == .atStart)
                        XCTAssertEqual(got, want, "n=\(n) \(tb) desc=\(descending) \(placement)")
                    }
                }
            }
        }
    }

    func testRankOptionsLarge() throws {
        try requireRealGPU()
        let vals = sample(Self.bigSize)
        let a = try MetalArray<Int32>(vals)
        for tb in RankTiebreaker.allCases {
            let got = try a.rank(tiebreaker: tb, nullPlacement: .atStart).toRawArray()
            XCTAssertEqual(got, expectedRank(vals, tiebreaker: tb, descending: false, nullsFirst: true), "\(tb)")
        }
    }

    /// The SQL-named entry points must stay exactly the four tiebreakers.
    func testRankAliases() throws {
        try requireRealGPU()
        let a = try MetalArray<Int32>(sample(200))
        XCTAssertEqual(try a.rank().toRawArray(), try a.rank(tiebreaker: .min).toRawArray())
        XCTAssertEqual(try a.denseRank().toRawArray(), try a.rank(tiebreaker: .dense).toRawArray())
        XCTAssertEqual(try a.rowNumber().toRawArray(), try a.rank(tiebreaker: .first).toRawArray())
        XCTAssertEqual(try a.maxRank().toRawArray(), try a.rank(tiebreaker: .max).toRawArray())
    }

    // MARK: - rank_quantile / rank_normal

    func testRankQuantileOptions() throws {
        try requireRealGPU()
        for n in [1, 33, 4097] {
            let vals = sample(n)
            let a = try MetalArray<Int32>(vals)
            for descending in [false, true] {
                for placement in NullPlacement.allCases {
                    let got = try a.rankQuantile(descending: descending, nullPlacement: placement).toRawArray()
                    let lo = expectedRank(vals, tiebreaker: .min, descending: descending,
                                          nullsFirst: placement == .atStart)
                    let hi = expectedRank(vals, tiebreaker: .max, descending: descending,
                                          nullsFirst: placement == .atStart)
                    for i in 0..<n {
                        // (average 1-based rank of the tie group - 0.5) / n, which is (s + e) / 2n
                        // with s = min rank - 1 and e = max rank.
                        let want = (Double(lo[i] - 1) + Double(hi[i])) / (2 * Double(n))
                        XCTAssertEqual(got[i], want, accuracy: 1e-12, "n=\(n) row \(i) \(placement)")
                    }
                }
            }
        }
    }

    func testRankNormalOptions() throws {
        try requireRealGPU()
        let vals = sample(129)
        let a = try MetalArray<Int32>(vals)
        for descending in [false, true] {
            for placement in NullPlacement.allCases {
                let q = try a.rankQuantile(descending: descending, nullPlacement: placement).toRawArray()
                let got = try a.rankNormal(descending: descending, nullPlacement: placement).toRawArray()
                for i in 0..<vals.count {
                    XCTAssertEqual(got[i], NormalQuantile.ppf(q[i]), accuracy: 1e-12)
                }
                let f32 = try a.rankNormalFloat32(descending: descending, nullPlacement: placement).toRawArray()
                for i in 0..<vals.count {
                    XCTAssertEqual(Double(f32[i]), NormalQuantile.ppf(q[i]), accuracy: 2e-5)
                }
            }
        }
    }

    // MARK: - partition_nth_indices

    func checkPartition(_ vals: [Int32?], pivot: Int, placement: NullPlacement) throws {
        let a = try MetalArray<Int32>(vals)
        let idx = try a.partitionNthIndices(pivot, nullPlacement: placement).toRawArray()
        XCTAssertEqual(idx.count, vals.count)
        XCTAssertEqual(Set(idx).count, idx.count, "permutation, pivot=\(pivot)")
        guard pivot < vals.count else { return }
        // Ordering key: a null is past every value at the placement's end.
        func key(_ i: Int32) -> OrderKey {
            guard let v = vals[Int(i)] else { return OrderKey(bucket: placement == .atStart ? -1 : 1, value: 0) }
            return OrderKey(bucket: 0, value: v)
        }
        let atPivot = key(idx[pivot])
        for i in 0..<pivot { XCTAssertFalse(atPivot < key(idx[i]), "row \(i) > pivot \(pivot)") }
        for i in (pivot + 1)..<vals.count { XCTAssertFalse(key(idx[i]) < atPivot, "row \(i) < pivot \(pivot)") }
        // And the pivot really is the value a sorted order would put there.
        let want = expectedOrder(vals, descending: false, nullsFirst: placement == .atStart)
        XCTAssertEqual(key(idx[pivot]), key(want[pivot]), "wrong order statistic at \(pivot)")
    }

    func testPartitionNth() throws {
        try requireRealGPU()
        for n in Self.sizes {
            let vals = sample(n)
            for pivot in Set([0, n / 3, n / 2, Swift.max(0, n - 1), n]) {
                for placement in NullPlacement.allCases {
                    try checkPartition(vals, pivot: pivot, placement: placement)
                }
            }
        }
    }

    func testPartitionNthNoNulls() throws {
        try requireRealGPU()
        let vals: [Int32?] = (0..<5000).map { (i: Int) -> Int32? in Int32((i &* 1103515 &+ 12345) % 1000) }
        for pivot in [0, 1, 2499, 4999] {
            try checkPartition(vals, pivot: pivot, placement: .atEnd)
        }
    }

    func testPartitionNthFloatAndWideKeys() throws {
        try requireRealGPU()
        let f: [Float?] = (0..<1000).map { (i: Int) -> Float? in
            i % 9 == 0 ? nil : Float(i % 17) - 8.0
        }
        let fa = try MetalArray<Float>(f)
        let idx = try fa.partitionNthIndices(500).toRawArray()
        XCTAssertEqual(Set(idx).count, 1000)
        let sortedIdx = try fa.argsort().toRawArray()
        XCTAssertEqual(f[Int(idx[500])], f[Int(sortedIdx[500])])

        let l: [Int64?] = (0..<1000).map { (i: Int) -> Int64? in
            i % 11 == 5 ? nil : Int64((i &* 2654435) % 10007)
        }
        let la = try MetalArray<Int64>(l)
        for placement in NullPlacement.allCases {
            let got = try la.partitionNthIndices(333, nullPlacement: placement).toRawArray()
            let want = try la.argsort(nullPlacement: placement).toRawArray()
            XCTAssertEqual(l[Int(got[333])], l[Int(want[333])], "\(placement)")
        }
    }

    func testPartitionNthLarge() throws {
        try requireRealGPU()
        let vals = sample(Self.bigSize)
        let a = try MetalArray<Int32>(vals)
        let idx = try a.partitionNthIndices(Self.bigSize / 2).toRawArray()
        XCTAssertEqual(Set(idx).count, Self.bigSize)
        let want = try a.argsort().toRawArray()
        XCTAssertEqual(vals[Int(idx[Self.bigSize / 2])], vals[Int(want[Self.bigSize / 2])])
    }

    func testPartitionNthRejectsBadPivot() throws {
        try requireRealGPU()
        let a = try MetalArray<Int32>([1, 2, 3] as [Int32])
        XCTAssertThrowsError(try a.partitionNthIndices(-1))
        XCTAssertThrowsError(try a.partitionNthIndices(4))
    }
}

// MARK: - is_in / index_in null_matching_behavior

extension OptionsTests {

    /// Arrow's four behaviours, written out on the host.
    func expectedIsIn(_ vals: [Int32?], set: [Int32?], _ b: SetLookupNullMatching) -> [Bool?] {
        let present = Set(set.compactMap { $0 })
        let setHasNull = set.contains { $0 == nil }
        return vals.map { v -> Bool? in
            guard let v else {
                switch b {
                case .skip: return false
                case .match: return setHasNull
                case .emitNull, .inconclusive: return nil
                }
            }
            if present.contains(v) { return true }
            return (b == .inconclusive && setHasNull) ? nil : false
        }
    }

    func expectedIndexIn(_ vals: [Int32?], set: [Int32?], _ b: SetLookupNullMatching) -> [Int32?] {
        var firstAt: [Int32: Int32] = [:]
        var firstNull: Int32? = nil
        for (i, s) in set.enumerated() {
            if let s {
                if firstAt[s] == nil { firstAt[s] = Int32(i) }
            } else if firstNull == nil {
                firstNull = Int32(i)
            }
        }
        return vals.map { v -> Int32? in
            guard let v else { return b == .match ? firstNull : nil }
            return firstAt[v]
        }
    }

    func testSetLookupNullMatching() throws {
        try requireRealGPU()
        let sets: [[Int32?]] = [[2, nil, 5, 2], [2, 5], [nil], []]
        for n in Self.sizes {
            let vals: [Int32?] = (0..<n).map { (i: Int) -> Int32? in
                i % 5 == 2 ? nil : Int32(i % 8)
            }
            let a = try MetalArray<Int32>(vals)
            for set in sets {
                let s = try MetalArray<Int32>(set)
                for b in SetLookupNullMatching.allCases {
                    let got = try a.isIn(s, nullMatching: b).toArray()
                    XCTAssertEqual(got, expectedIsIn(vals, set: set, b), "is_in n=\(n) set=\(set) \(b)")
                    let gotIdx = try a.indexIn(s, nullMatching: b).toArray()
                    XCTAssertEqual(gotIdx, expectedIndexIn(vals, set: set, b),
                                   "index_in n=\(n) set=\(set) \(b)")
                }
            }
        }
    }

    func testSetLookupNullMatchingLarge() throws {
        try requireRealGPU()
        let vals: [Int32?] = (0..<Self.bigSize).map { (i: Int) -> Int32? in
            i % 7 == 3 ? nil : Int32(i % 64)
        }
        let set: [Int32?] = [1, 2, nil, 63]
        let a = try MetalArray<Int32>(vals)
        let s = try MetalArray<Int32>(set)
        for b in SetLookupNullMatching.allCases {
            XCTAssertEqual(try a.isIn(s, nullMatching: b).toArray(), expectedIsIn(vals, set: set, b), "\(b)")
            XCTAssertEqual(try a.indexIn(s, nullMatching: b).toArray(),
                           expectedIndexIn(vals, set: set, b), "\(b)")
        }
    }

    func testSetLookupNullMatchingStrings() throws {
        try requireRealGPU()
        let words: [String?] = ["a", "bb", nil, "ccc", "a", nil, "dd"]
        let set: [String?] = ["bb", nil, "a"]
        let a = try MetalStringArray(words, context: .shared)
        let s = try MetalStringArray(set, context: .shared)
        let numbering: [String: Int32] = ["a": 0, "bb": 1, "ccc": 2, "dd": 3]
        let asInts: [Int32?] = words.map { w in w.flatMap { numbering[$0] } }
        let setInts: [Int32?] = set.map { w in w.flatMap { numbering[$0] } }
        for b in SetLookupNullMatching.allCases {
            XCTAssertEqual(try a.isIn(s, nullMatching: b).toArray(), expectedIsIn(asInts, set: setInts, b), "\(b)")
            XCTAssertEqual(try a.indexIn(s, nullMatching: b).toArray(),
                           expectedIndexIn(asInts, set: setInts, b), "\(b)")
        }
    }
}

// MARK: - unique / value_counts / dictionary_encode order

extension OptionsTests {

    /// The distinct values in order of first appearance, null included, as Arrow returns them.
    func expectedFirstAppearance(_ vals: [Int32?]) -> [Int32?] {
        var seen = Set<Int32>()
        var sawNull = false
        var out: [Int32?] = []
        for v in vals {
            if let v {
                if seen.insert(v).inserted { out.append(v) }
            } else if !sawNull {
                sawNull = true
                out.append(nil)
            }
        }
        return out
    }

    func testUniqueOrder() throws {
        try requireRealGPU()
        for n in Self.sizes + [Self.bigSize] {
            let vals: [Int32?] = (0..<n).map { (i: Int) -> Int32? in
                i % 6 == 4 ? nil : Int32((i &* 37) % 23)
            }
            let a = try MetalArray<Int32>(vals)
            XCTAssertEqual(try a.unique(order: .firstAppearance).toArray(), expectedFirstAppearance(vals),
                           "unique first_appearance n=\(n)")
            let sorted = try a.unique(order: .sorted).toArray()
            let wantSorted: [Int32?] = Set(vals.compactMap { $0 }).sorted().map { Optional($0) }
            XCTAssertEqual(sorted, wantSorted, "unique sorted n=\(n)")
        }
    }

    func testValueCountsOrder() throws {
        try requireRealGPU()
        for n in Self.sizes {
            let vals: [Int32?] = (0..<n).map { (i: Int) -> Int32? in
                i % 6 == 4 ? nil : Int32((i &* 37) % 23)
            }
            let a = try MetalArray<Int32>(vals)
            let (values, counts) = try a.valueCounts(order: .firstAppearance)
            let want = expectedFirstAppearance(vals)
            XCTAssertEqual(values.toArray(), want, "values n=\(n)")
            let wantCounts: [Int64] = want.map { v in Int64(vals.filter { $0 == v }.count) }
            XCTAssertEqual(counts.toRawArray(), wantCounts, "counts n=\(n)")
        }
    }

    func testDictionaryEncodeOrder() throws {
        try requireRealGPU()
        for n in Self.sizes {
            let vals: [Int32?] = (0..<n).map { (i: Int) -> Int32? in
                i % 6 == 4 ? nil : Int32((i &* 37) % 23)
            }
            let a = try MetalArray<Int32>(vals)
            let (codes, dict) = try a.dictionaryEncode(order: .firstAppearance)
            let wantDict = expectedFirstAppearance(vals).compactMap { $0 }
            XCTAssertEqual(dict.toArray(), wantDict.map { Optional($0) }, "dictionary n=\(n)")
            var position: [Int32: Int32] = [:]
            for (i, v) in wantDict.enumerated() { position[v] = Int32(i) }
            let wantCodes: [Int32?] = vals.map { v in v.flatMap { position[$0] } }
            XCTAssertEqual(codes.toArray(), wantCodes, "codes n=\(n)")
            for (i, c) in codes.toArray().enumerated() {
                XCTAssertEqual(c.map { wantDict[Int($0)] }, vals[i], "row \(i)")
            }
        }
    }

    func testUniqueOrderAllNull() throws {
        try requireRealGPU()
        let vals: [Int32?] = [nil, nil, nil]
        let a = try MetalArray<Int32>(vals)
        XCTAssertEqual(try a.unique(order: .firstAppearance).toArray(), [nil])
        XCTAssertEqual(try a.unique(order: .sorted).toArray(), [])
        let (values, counts) = try a.valueCounts(order: .firstAppearance)
        XCTAssertEqual(values.toArray(), [nil])
        XCTAssertEqual(counts.toRawArray(), [3])
        let (codes, dict) = try a.dictionaryEncode(order: .firstAppearance)
        XCTAssertEqual(dict.length, 0)
        XCTAssertEqual(codes.toArray(), [nil, nil, nil])
    }

    func testStringUniqueOrder() throws {
        try requireRealGPU()
        let words: [String?] = ["pear", "apple", "pear", nil, "fig", "apple"]
        let a = try MetalStringArray(words, context: .shared)
        XCTAssertEqual(try a.unique(order: .firstAppearance).toArray(), ["pear", "apple", "fig"])
        XCTAssertEqual(try a.unique(order: .sorted).toArray(), ["apple", "fig", "pear"])
        let (values, counts) = try a.valueCounts(order: .sorted)
        XCTAssertEqual(values.toArray(), ["apple", "fig", "pear"])
        XCTAssertEqual(counts.toRawArray(), [2, 1, 2])
    }
}

// MARK: - CastOptions

extension OptionsTests {

    func testCastUncheckedIsUnchanged() throws {
        try requireRealGPU()
        let a = try MetalArray<Int64>([300, 5, -1, 0] as [Int64])
        XCTAssertEqual(try a.cast(to: Int8.self, options: .unsafe).toRawArray(), [44, 5, -1, 0])
        XCTAssertEqual(try a.cast(to: Int8.self).toRawArray(), [44, 5, -1, 0])
    }

    func testCastSafeRaisesOnIntegerOverflow() throws {
        try requireRealGPU()
        let a = try MetalArray<Int64>([1, 2, 300, 400] as [Int64])
        XCTAssertThrowsError(try a.cast(to: Int8.self, options: .safe)) { error in
            guard case ArrowMetalError.overflow(_, let index, _) = error else {
                return XCTFail("expected an overflow error, got \(error)")
            }
            XCTAssertEqual(index, 2, "the message must name the first offending row")
        }
        XCTAssertEqual(try a.cast(to: Int16.self, options: .safe).toRawArray(), [1, 2, 300, 400])
        var allowed = CastOptions.safe
        allowed.allowIntOverflow = true
        XCTAssertEqual(try a.cast(to: Int8.self, options: allowed).toRawArray(), [1, 2, 44, -112])
    }

    func testCastSafeCatchesSignCrossing() throws {
        try requireRealGPU()
        // -1 round-trips through uint64 bit for bit and is still out of range, which is why the check
        // is not the round trip alone.
        let signed = try MetalArray<Int64>([-1, 1] as [Int64])
        XCTAssertThrowsError(try signed.cast(to: UInt64.self, options: .safe))
        XCTAssertEqual(try signed.cast(to: UInt64.self).toRawArray(), [UInt64.max, 1])

        let unsigned = try MetalArray<UInt64>([UInt64.max, 1] as [UInt64])
        XCTAssertThrowsError(try unsigned.cast(to: Int64.self, options: .safe))
    }

    func testCastSafeFloatToInt() throws {
        try requireRealGPU()
        let exact = try MetalArray<Float>([0, 1, -2, 3] as [Float])
        XCTAssertEqual(try exact.cast(to: Int32.self, options: .safe).toRawArray(), [0, 1, -2, 3])
        for bad in [[Float(1.5)], [Float(1e30)], [Float.nan], [Float.infinity]] {
            let a = try MetalArray<Float>(bad)
            XCTAssertThrowsError(try a.cast(to: Int32.self, options: .safe), "\(bad)")
            var allowed = CastOptions.safe
            allowed.allowFloatTruncate = true
            XCTAssertNoThrow(try a.cast(to: Int32.self, options: allowed))
        }
    }

    func testCastSafeIntToFloatUsesTheContiguousRange() throws {
        try requireRealGPU()
        // 2^31 is exactly representable in float32 and Arrow still refuses it: the rule is the
        // contiguous integer range, 2^24.
        let a = try MetalArray<Int64>([1 << 31] as [Int64])
        XCTAssertThrowsError(try a.cast(to: Float.self, options: .safe))
        XCTAssertNoThrow(try MetalArray<Int64>([1 << 24] as [Int64]).cast(to: Float.self, options: .safe))
        XCTAssertNoThrow(try a.cast(to: Double.self, options: .safe))
        // A source too narrow to leave the range never pays for a check.
        XCTAssertNil(MetalArray<Int16>.lossPredicate(Float.self))
    }

    func testCastSafeFloatToFloatNeverRaises() throws {
        try requireRealGPU()
        let a = try MetalArray<Double>([1e300, -1e300, 0.5] as [Double])
        let out = try a.cast(to: Float.self, options: .safe).toRawArray()
        XCTAssertEqual(out[0], .infinity)
        XCTAssertEqual(out[1], -.infinity)
        XCTAssertEqual(out[2], 0.5)
    }

    func testCastSafeSkipsNullRows() throws {
        try requireRealGPU()
        // A null row is never evaluated, whatever bytes happen to sit under it.
        let a = try MetalArray<Int64>([1, nil, 2] as [Int64?])
        XCTAssertEqual(try a.cast(to: Int8.self, options: .safe).toArray(), [1, nil, 2])
    }

    func testCastSafeAtEverySize() throws {
        try requireRealGPU()
        for n in Self.sizes + [Self.bigSize] {
            let vals: [Int64?] = (0..<n).map { (i: Int) -> Int64? in i % 11 == 5 ? nil : Int64(i % 100) }
            let a = try MetalArray<Int64>(vals)
            XCTAssertEqual(try a.cast(to: Int8.self, options: .safe).toArray().map { $0.map(Int64.init) },
                           vals, "n=\(n)")
            guard n > 200 else { continue }
            // One bad row anywhere in a big array must still be found.
            var withBad = vals
            withBad[n - 1] = 300
            let b = try MetalArray<Int64>(withBad)
            XCTAssertThrowsError(try b.cast(to: Int8.self, options: .safe), "n=\(n)")
        }
    }

    func testCastOptionBits() {
        XCTAssertEqual(CastOptions.safe.bits, 0)
        XCTAssertEqual(CastOptions.unsafe.bits, 0b111111)
        for bits in UInt32(0)...0b111111 {
            XCTAssertEqual(CastOptions(bits: bits).bits, bits)
        }
    }

    func testCastDispatchReachesEveryFamily() throws {
        try requireRealGPU()
        let ints = AnyMetalArray.int64(try MetalArray<Int64>([1, 0, nil] as [Int64?]))
        XCTAssertEqual(try ints.cast(to: "g").arrowFormat, "g")
        XCTAssertEqual(try ints.cast(to: "b").arrowFormat, "b")
        XCTAssertEqual(try ints.cast(to: "u").arrowFormat, "u")
        XCTAssertEqual(try ints.cast(to: "e").arrowFormat, "e")
        XCTAssertEqual(try ints.cast(to: "tss:").arrowFormat, "tss:")
        XCTAssertEqual(try ints.cast(to: "d:12,2").arrowFormat, "d:12,2")
        // An unknown target is refused rather than silently ignored.
        XCTAssertThrowsError(try ints.cast(to: "+m"))
    }

    func testCastNestedCastsTheChild() throws {
        try requireRealGPU()
        let child = AnyMetalArray.int32(try MetalArray<Int32>([1, 2, 3] as [Int32]))
        let list = try MetalListArray(counts: [2, nil, 1], values: child)
        let out = try AnyMetalArray.list(list).cast(to: "+l", childFormats: ["l"])
        guard case .list(let l) = out else { return XCTFail("expected a list") }
        XCTAssertEqual(l.values.arrowFormat, "l")
        XCTAssertEqual(l.length, 3)
        XCTAssertEqual(l.nullCount, 1)
        // The offsets and the validity bitmap are shared, not copied.
        XCTAssertTrue(l.offsets === list.offsets)

        let s = try MetalStructArray(names: ["a", "b"],
                                     children: [AnyMetalArray.int32(try MetalArray<Int32>([1, 2] as [Int32])),
                                                AnyMetalArray.float32(try MetalArray<Float>([1.5, 2.5] as [Float]))])
        let cast = try AnyMetalArray.structure(s).cast(to: "+s", childFormats: ["l", "g"])
        guard case .structure(let t) = cast else { return XCTFail("expected a struct") }
        XCTAssertEqual(t.children.map { $0.arrowFormat }, ["l", "g"])
        XCTAssertEqual(t.names, ["a", "b"])
    }

    func testCastNestedPropagatesTheChildError() throws {
        try requireRealGPU()
        let child = AnyMetalArray.int64(try MetalArray<Int64>([300] as [Int64]))
        let list = try MetalListArray(counts: [1], values: child)
        XCTAssertThrowsError(try AnyMetalArray.list(list).cast(to: "+l", options: .safe, childFormats: ["c"]))
    }
}

// MARK: - RoundTemporalOptions

extension OptionsTests {

    func timestamps(_ unit: ArrowTemporalUnit = .second) throws -> MetalTemporalArray {
        // 2024-05-17T13:47:33, 1970-01-01, one second before the epoch, and a leap day.
        let seconds: [Int64?] = [1_715_953_653, 0, -1, 1_709_247_000, 951_912_000, nil]
        return try MetalTemporalArray(type: .timestamp(unit, timezone: nil),
                                      try MetalArray<Int64>(seconds))
    }

    func testRoundTemporalUnitsIncludeWeek() throws {
        try requireRealGPU()
        XCTAssertTrue(TemporalRoundUnit.allCases.contains(.week))
        XCTAssertEqual(TemporalRoundUnit.allCases.count, 11, "Arrow has eleven units")
    }

    /// The week grid is anchored on a Monday (or a Sunday) and is a whole number of weeks wide.
    func testFloorTemporalWeekLandsOnTheRightWeekday() throws {
        try requireRealGPU()
        let days: [Int32?] = (0..<40).map { Int32(19_800 + $0) }        // a run of consecutive days
        let a = try MetalTemporalArray(type: .date32, try MetalArray<Int32>(days))
        for startsMonday in [true, false] {
            let opts = RoundTemporalOptions(multiple: 1, unit: .week, weekStartsMonday: startsMonday)
            let got = try a.floorTemporal(opts).asInt32!.toRawArray()
            for (i, d) in days.enumerated() {
                let floored = got[i]
                XCTAssertLessThanOrEqual(floored, d!, "a floor never exceeds its input")
                XCTAssertLessThan(d! - floored, 7)
                // 1970-01-01 was a Thursday, so day 0 mod 7 == 4 counting Monday as 0.
                let weekday = ((floored % 7) + 7 + 3) % 7
                XCTAssertEqual(weekday, startsMonday ? 0 : 6, "day \(d!) floored to \(floored)")
            }
        }
    }

    func testRoundTemporalHalvesGoUp() throws {
        try requireRealGPU()
        let seconds: [Int64?] = [0, 1, 2, 3, 4, 5, -1, -3]
        let a = try MetalTemporalArray(type: .timestamp(.second, timezone: nil),
                                       try MetalArray<Int64>(seconds))
        let got = try a.roundTemporal(RoundTemporalOptions(multiple: 2, unit: .second)).asInt64!.toRawArray()
        XCTAssertEqual(got, [0, 2, 2, 4, 4, 6, 0, -2], "an exact half goes toward +infinity")
    }

    func testCeilIsStrictlyGreater() throws {
        try requireRealGPU()
        let seconds: [Int64?] = [0, 3600 * 3, 3600 * 4]
        let a = try MetalTemporalArray(type: .timestamp(.second, timezone: nil),
                                       try MetalArray<Int64>(seconds))
        let plain = RoundTemporalOptions(multiple: 3, unit: .hour)
        var strict = plain
        strict.ceilIsStrictlyGreater = true
        XCTAssertEqual(try a.ceilTemporal(plain).asInt64!.toRawArray(), [0, 3600 * 3, 3600 * 6])
        XCTAssertEqual(try a.ceilTemporal(strict).asInt64!.toRawArray(), [3600 * 3, 3600 * 6, 3600 * 6])
    }

    func testCalendarBasedOriginStartsAtTheGreaterUnit() throws {
        try requireRealGPU()
        // 2024-05-17T13:47:33 UTC.
        let a = try MetalTemporalArray(type: .timestamp(.second, timezone: nil),
                                       try MetalArray<Int64>([1_715_953_653] as [Int64]))
        var opts = RoundTemporalOptions(multiple: 7, unit: .minute)
        let epochBased = try a.floorTemporal(opts).asInt64!.toRawArray()[0]
        opts.calendarBasedOrigin = true
        let dayBased = try a.floorTemporal(opts).asInt64!.toRawArray()[0]
        // The calendar form starts the grid at the top of the hour: minute 47 floors to 42.
        XCTAssertEqual(dayBased % 3600, 42 * 60)
        XCTAssertNotEqual(epochBased, dayBased)
    }

    func testRoundTemporalRejectsBadOptions() throws {
        try requireRealGPU()
        let a = try timestamps()
        XCTAssertThrowsError(try a.floorTemporal(RoundTemporalOptions(multiple: 0, unit: .day)))
        // A calendar unit needs a date, which a time-of-day column does not carry.
        let t = try MetalTemporalArray(type: .time32(.second), try MetalArray<Int32>([1, 2] as [Int32]))
        XCTAssertThrowsError(try t.floorTemporal(RoundTemporalOptions(multiple: 1, unit: .month)))
        XCTAssertThrowsError(try t.floorTemporal(RoundTemporalOptions(multiple: 1, unit: .week)))
    }

    func testRoundTemporalKeepsNullsAndLength() throws {
        try requireRealGPU()
        for unit in TemporalRoundUnit.allCases {
            let a = try timestamps()
            let out = try a.floorTemporal(RoundTemporalOptions(multiple: 3, unit: unit))
            XCTAssertEqual(out.length, a.length, "\(unit)")
            XCTAssertEqual(out.nullCount, a.nullCount, "\(unit)")
        }
    }

    func testRoundTemporalLarge() throws {
        try requireRealGPU()
        let n = 200_003
        let seconds: [Int64?] = (0..<n).map { (i: Int) -> Int64? in i % 9 == 4 ? nil : Int64(i) * 37 }
        let a = try MetalTemporalArray(type: .timestamp(.second, timezone: nil),
                                       try MetalArray<Int64>(seconds))
        let got = try a.floorTemporal(RoundTemporalOptions(multiple: 5, unit: .hour)).asInt64!.toRawArray()
        for (i, v) in seconds.enumerated() where v != nil {
            let period: Int64 = 5 * 3600
            var lo = (v! / period) * period
            if v! % period != 0 && v! < 0 { lo -= period }
            XCTAssertEqual(got[i], lo, "row \(i)")
        }
    }
}

// MARK: - list_parent_indices in int64

extension OptionsTests {

    func testListParentIndicesWidths() throws {
        try requireRealGPU()
        let child = AnyMetalArray.int64(try MetalArray<Int64>([1, 2, 3, 4, 5, 6] as [Int64]))
        let list = try MetalListArray(counts: [3, 1, nil, 0, 2], values: child)
        let narrow = try list.listParentIndices().toRawArray()
        let wide = try list.listParentIndices64().toRawArray()
        XCTAssertEqual(narrow, [0, 0, 0, 1, 4, 4])
        XCTAssertEqual(wide, narrow.map(Int64.init))
        XCTAssertEqual(try AnyMetalArray.list(list).listParentIndices64().toRawArray(), wide)
    }
}
