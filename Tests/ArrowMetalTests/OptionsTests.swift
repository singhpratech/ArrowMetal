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
