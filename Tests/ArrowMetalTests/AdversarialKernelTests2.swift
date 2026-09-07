import XCTest
@testable import ArrowMetal

/// Second adversarial pass: keys crafted to collide in the GPU hash table, the values a sort has to
/// order exactly, slices at every awkward offset, and the invariants that couple a kernel's compiled
/// constants to the threadgroup size the host dispatches.
final class AdversarialKernelTests2: XCTestCase {

    // MARK: - hash table: crafted collisions

    /// `ht_build` starts every probe at `((uint)h ^ (uint)(h >> 32)) & mask` where
    /// `h = ht_mix(key)` is splitmix64's finalizer. That finalizer is a bijection, so a key can be
    /// chosen for any hash: this inverts it.
    private static func unmix(_ h0: UInt64) -> UInt64 {
        func unshift33(_ x: UInt64) -> UInt64 { x ^ (x >> 33) }   // 33 * 2 > 64, so one step undoes it
        var h = unshift33(h0)
        h &*= 0x9cb4_b2f8_1293_37db                                // inverse of 0xc4ceb9fe1a85ec53
        h = unshift33(h)
        h &*= 0x4f74_430c_22a5_4005                                // inverse of 0xff51afd7ed558ccd
        return unshift33(h)
    }

    /// `count` distinct keys whose mixed hash folds to the *same* 32 bits, so every one of them starts
    /// its probe at the same slot whatever the table size is: one probe chain `count` long, the worst
    /// case open addressing has. Setting the high word to `i` and the low word to `target ^ i` makes
    /// the fold `(lo ^ hi)` equal `target` for every `i`.
    private static func collidingKeys(_ count: Int, target: UInt32 = 0x1234_5678) -> [Int64] {
        (1...count).map { i -> Int64 in
            let hi = UInt64(UInt32(truncatingIfNeeded: i))
            let lo = UInt64(target ^ UInt32(truncatingIfNeeded: i))
            return Int64(bitPattern: unmix((hi << 32) | lo))
        }
    }

    /// Every distinct key sharing one probe chain. The first build attempt has a 96-probe budget, so
    /// this also drives the grow-and-retry loop all the way to the "two slots per row, no budget" table.
    func testHashTableWithOneLongProbeChain() throws {
        try requireRealGPU()
        let distinct = Self.collidingKeys(1000)
        XCTAssertEqual(Set(distinct).count, 1000, "the crafted keys must really be distinct")
        // Above `1 << 16` rows `count_distinct`, `unique` and `value_counts` take the hash-table path.
        let rows = (0..<70_000).map { distinct[$0 % distinct.count] }
        let a = try MetalArray<Int64>(rows, context: .shared)
        XCTAssertEqual(try a.countDistinct(), 1000)
        XCTAssertEqual(Set(try a.unique().toRawArray()), Set(distinct))
        let (values, counts) = try a.valueCounts()
        XCTAssertEqual(values.length, 1000)
        XCTAssertEqual(Set(values.toRawArray()), Set(distinct))
        XCTAssertEqual(counts.toRawArray().reduce(0, +), 70_000)
        // Every key occurs 70 or 71 times; nothing may be merged or lost.
        for c in counts.toRawArray() { XCTAssertTrue(c == 70 || c == 71, "count \(c)") }
    }

    /// The same chain, but with nulls mixed in and the key 0 present — 0 is what an empty slot holds
    /// (`slots[s] == 0` means empty; a slot stores `row + 1`), so a zero *key* must not be confused
    /// with an empty slot.
    func testHashTableWithZeroKeyAndNulls() throws {
        try requireRealGPU()
        var distinct = Self.collidingKeys(400)
        distinct.append(0)
        distinct.append(Int64.min)
        distinct.append(Int64.max)
        var rows: [Int64?] = []
        for i in 0..<70_000 {
            rows.append(i % 37 == 0 ? nil : distinct[i % distinct.count])
        }
        let a = try MetalArray<Int64>(rows, context: .shared)
        let expected = Set(rows.compactMap { $0 })
        XCTAssertEqual(try a.countDistinct(), expected.count)
        XCTAssertEqual(Set(try a.unique().toRawArray()).subtracting([0]).union([0]), expected.union([0]))
        XCTAssertTrue(try a.unique().toRawArray().contains(0), "a zero key is a value, not an empty slot")
    }

    /// A column of one distinct key repeated, and a column where every row is distinct: the two ends
    /// of the load-factor range the estimator sizes for.
    func testHashTableAtBothEndsOfTheCardinalityRange() throws {
        try requireRealGPU()
        let n = 200_000
        let allSame = try MetalArray<Int64>([Int64](repeating: 42, count: n), context: .shared)
        XCTAssertEqual(try allSame.countDistinct(), 1)
        XCTAssertEqual(try allSame.unique().toRawArray(), [42])
        let allDistinct = try MetalArray<Int64>((0..<n).map { Int64($0) &* 0x9E37_79B9 }, context: .shared)
        XCTAssertEqual(try allDistinct.countDistinct(), n)
        XCTAssertEqual(try allDistinct.unique().length, n)
    }

    /// utf8 keys that agree for the first seven bytes — one whole chunk of the radix argsort's key —
    /// plus an empty string, an embedded NUL and non-ASCII bytes above 0x7F, where a signed byte
    /// comparison would invert the order.
    func testStringHashTableOnAdversarialKeys() throws {
        try requireRealGPU()
        var xs: [String?] = []
        for i in 0..<20_000 { xs.append("prefix7" + String(format: "%08d", i)) }
        xs.append("")
        xs.append("a\u{0}b")
        xs.append("a\u{0}c")
        xs.append("\u{00FF}z")          // 0xC3 0xBF: a byte above 0x7F
        xs.append("\u{007F}z")          // 0x7F
        xs.append(nil)
        let repeated = xs + xs
        let a = try MetalStringArray(repeated, context: .shared)
        let (values, counts) = try a.valueCounts()
        let distinct = Set(repeated.compactMap { $0 })
        XCTAssertEqual(Set(values.toArray().compactMap { $0 }), distinct)
        XCTAssertEqual(values.length, distinct.count)
        for c in counts.toRawArray() { XCTAssertEqual(c, 2) }
    }

    // MARK: - sort: the values that have to be ordered exactly

    /// Arrow compares -0.0 and 0.0 equal, so a stable sort must leave them in row order, and the
    /// *bits* of each must survive the round trip through the radix key.
    func testNegativeZeroSortsEqualToZeroAndStably() throws {
        try requireRealGPU()
        let vals: [Double] = [0.0, -0.0, 1.0, -0.0, 0.0, -1.0]
        let a = try MetalArray<Double>(vals, context: .shared)
        let idx = try a.argsort().toRawArray()
        XCTAssertEqual(idx, [5, 0, 1, 3, 4, 2], "the four zeroes keep row order")
        let sorted = try a.sorted().toRawArray()
        XCTAssertEqual(sorted.map { $0.sign == .minus }, [true, false, true, true, false, false],
                       "the sign bit of each zero survives the sort")
    }

    func testExtremeIntegerAndFloatValuesSortExactly() throws {
        try requireRealGPU()
        let i64: [Int64] = [.max, .min, 0, -1, 1, .min + 1, .max - 1]
        let a = try MetalArray<Int64>(i64, context: .shared)
        XCTAssertEqual(try a.sorted().toRawArray(), i64.sorted())
        XCTAssertEqual(try a.sorted(descending: true).toRawArray(), i64.sorted().reversed())

        let u64: [UInt64] = [.max, 0, 1 << 63, (1 << 63) - 1, (1 << 63) + 1, 1]
        let b = try MetalArray<UInt64>(u64, context: .shared)
        XCTAssertEqual(try b.sorted().toRawArray(), u64.sorted(), "uint64 above 2^63 must not sort as negative")

        let f: [Double] = [.leastNonzeroMagnitude, -.leastNonzeroMagnitude, 0.0, .infinity, -.infinity,
                           .leastNormalMagnitude, -.leastNormalMagnitude, .greatestFiniteMagnitude]
        let c = try MetalArray<Double>(f, context: .shared)
        XCTAssertEqual(try c.sorted().toRawArray(), f.sorted(), "subnormals and infinities")

        let f32: [Float] = [.leastNonzeroMagnitude, -.leastNonzeroMagnitude, 0.0, .infinity, -.infinity]
        let d = try MetalArray<Float>(f32, context: .shared)
        XCTAssertEqual(try d.sorted().toRawArray(), f32.sorted())
    }

    /// Nulls sit past the values in *both* directions (Arrow's rule), and `.atStart` moves them to the
    /// front in both directions too.
    func testDescendingWithNullsAtEitherEnd() throws {
        try requireRealGPU()
        let vals: [Int32?] = [3, nil, 1, nil, 2]
        let a = try MetalArray<Int32>(vals, context: .shared)
        XCTAssertEqual(try a.argsort(descending: true).toRawArray().map { vals[Int($0)] },
                       [3, 2, 1, nil, nil])
        XCTAssertEqual(try a.argsort(descending: true, nullPlacement: .atStart).toRawArray().map { vals[Int($0)] },
                       [nil, nil, 3, 2, 1])
        XCTAssertEqual(try a.argsort(nullPlacement: .atStart).toRawArray().map { vals[Int($0)] },
                       [nil, nil, 1, 2, 3])
        // The nulls come back in row order at either end.
        XCTAssertEqual(try a.argsort(nullPlacement: .atStart).toRawArray().prefix(2), [1, 3])
        XCTAssertEqual(try a.argsort(nullPlacement: .atEnd).toRawArray().suffix(2), [1, 3])
    }

    /// A null in the *second* key of a lexsort must break ties the same way it does in the first.
    func testLexsortWithANullInTheSecondKey() throws {
        try requireRealGPU()
        let k1: [Int32?] = [1, 1, 1, 2, 2]
        let k2: [Int32?] = [5, nil, 3, nil, 1]
        let a = try MetalArray<Int32>(k1, context: .shared)
        let b = try MetalArray<Int32>(k2, context: .shared)
        let end = try lexsortIndices([.int32(a), .int32(b)], nullPlacement: .atEnd).toRawArray()
        XCTAssertEqual(end, [2, 0, 1, 4, 3], "within each first-key group the null second key sorts last")
        let start = try lexsortIndices([.int32(a), .int32(b)], nullPlacement: .atStart).toRawArray()
        XCTAssertEqual(start, [1, 2, 0, 3, 4], "and first with .atStart")
    }

    // MARK: - slices

    /// Every sort, rank and selection entry point over a slice at an offset that is not a multiple of
    /// 32 — the case where the validity bitmap's bit offset has to be applied.
    func testEverySelectionEntryPointOnASlice() throws {
        try requireRealGPU()
        let n = 4096
        var vals: [Int32?] = []
        for i in 0..<n {
            if i % 11 == 0 { vals.append(nil) } else { vals.append(Int32((i &* 2654435761) % 1009)) }
        }
        let full = try MetalArray<Int32>(vals, context: .shared)
        for off in [1, 7, 31, 32, 33, 63, 64] {
            let len = 777
            let s = try full.slice(offset: off, length: len)
            let sub = Array(vals[off..<(off + len)])
            let standalone = try MetalArray<Int32>(sub, context: .shared)
            XCTAssertEqual(s.nullCount, standalone.nullCount, "nullCount at offset \(off)")
            XCTAssertEqual(try s.argsort().toRawArray(), try standalone.argsort().toRawArray(),
                           "argsort at offset \(off)")
            XCTAssertEqual(try s.argsort(nullPlacement: .atStart).toRawArray(),
                           try standalone.argsort(nullPlacement: .atStart).toRawArray(),
                           "argsort .atStart at offset \(off)")
            XCTAssertEqual(try s.topK(20).toRawArray(), try standalone.topK(20).toRawArray(),
                           "topK at offset \(off)")
            XCTAssertEqual(try s.rank().toRawArray(), try standalone.rank().toRawArray(),
                           "rank at offset \(off)")
            XCTAssertEqual(try s.denseRank().toRawArray(), try standalone.denseRank().toRawArray(),
                           "denseRank at offset \(off)")
            XCTAssertEqual(try s.partitionNthIndices(100).toRawArray().count, len)
            XCTAssertEqual(try s.kthElement(5), try standalone.kthElement(5), "kth at offset \(off)")
            XCTAssertEqual(try s.quantile(0.5), try standalone.quantile(0.5), "median at offset \(off)")
            XCTAssertEqual(try s.first(), try standalone.first(), "first at offset \(off)")
            XCTAssertEqual(try s.last(), try standalone.last(), "last at offset \(off)")
        }
    }

    /// Slicing a slice adds the offsets; a zero-length slice and a slice that ends exactly at the end
    /// of the parent must both be legal and must read nothing past it.
    func testSliceOfSliceAndDegenerateSlices() throws {
        try requireRealGPU()
        let vals: [Int32?] = (0..<200).map { $0 % 5 == 0 ? nil : Int32($0) }
        let a = try MetalArray<Int32>(vals, context: .shared)
        let s1 = try a.slice(offset: 7, length: 100)
        let s2 = try s1.slice(offset: 9, length: 50)
        let direct = try a.slice(offset: 16, length: 50)
        XCTAssertEqual(s2.toArray(), direct.toArray())
        XCTAssertEqual(try s2.argsort().toRawArray(), try direct.argsort().toRawArray())

        XCTAssertEqual(try a.slice(offset: 200, length: 0).length, 0)
        XCTAssertEqual(try a.slice(offset: 199, length: 1).toArray(), [vals[199]])
        XCTAssertEqual(try a.slice(offset: 0, length: 0).argsort().length, 0)
        XCTAssertEqual(try a.slice(offset: 200, length: 0).argsort().length, 0)

        let xs: [String?] = (0..<200).map { $0 % 5 == 0 ? nil : "s\($0)" }
        let sa = try MetalStringArray(xs, context: .shared)
        let t1 = try sa.slice(offset: 7, length: 100)
        let t2 = try t1.slice(offset: 9, length: 50)
        XCTAssertEqual(t2.toArray(), Array(xs[16..<66]))
        XCTAssertEqual(try sa.slice(offset: 200, length: 0).length, 0)
        XCTAssertEqual(try sa.slice(offset: 199, length: 1).toArray(), [xs[199]])
        XCTAssertEqual(try t2.argsort().toRawArray(),
                       try MetalStringArray(Array(xs[16..<66]), context: .shared).argsort().toRawArray())
    }

    // MARK: - group-by

    /// A group per row, and one group for every row: the two ends of the counting sort's range, plus a
    /// group count large enough to leave the chunked histogram for the atomic scatter.
    func testCountingSortAtBothEndsOfTheGroupCountRange() throws {
        try requireRealGPU()
        let n = 50_000
        let oneGroup = try MetalArray<Int32>([Int32](repeating: 0, count: n), context: .shared)
        let vals = try MetalArray<Double>((0..<n).map { Double($0 % 100) }, context: .shared)
        let gb1 = try oneGroup.groupBy(keyCount: 1)
        XCTAssertEqual(try gb1.count().toRawArray(), [Int64(n)])
        let v1 = try gb1.varianceDouble(vals, ddof: 1).toArray()
        let mean = (0..<n).map { Double($0 % 100) }.reduce(0, +) / Double(n)
        let want = (0..<n).map { pow(Double($0 % 100) - mean, 2) }.reduce(0, +) / Double(n - 1)
        XCTAssertEqual(v1[0]!, want, accuracy: want * 1e-12)

        // Every row its own group: K == n, so no chunked per-block histogram fits and the atomic
        // scatter plus the run fix-up has to carry it.
        let perRow = try MetalArray<Int32>((0..<n).map { Int32($0) }, context: .shared)
        let gbN = try perRow.groupBy(keyCount: n)
        XCTAssertEqual(try gbN.count().toRawArray(), [Int64](repeating: 1, count: n))
        let (mn, mx) = try gbN.extrema(vals)
        XCTAssertEqual(mn.toArray(), (0..<n).map { Double($0 % 100) })
        XCTAssertEqual(mx.toArray(), (0..<n).map { Double($0 % 100) })
    }

    /// Arrow's `first` / `last` with `skip_nulls`: a leading (trailing) run of nulls is skipped when
    /// the flag is set and is the answer when it is not.
    func testFirstAndLastSkipNulls() throws {
        try requireRealGPU()
        let vals: [Int32?] = [nil, nil, 7, 8, nil, 9, nil, nil]
        let a = try MetalArray<Int32>(vals, context: .shared)
        XCTAssertEqual(try a.first(skipNulls: true), 7)
        XCTAssertEqual(try a.last(skipNulls: true), 9)
        XCTAssertNil(try a.first(skipNulls: false))
        XCTAssertNil(try a.last(skipNulls: false))

        let allNull = try MetalArray<Int32>([Int32?](repeating: nil, count: 5), context: .shared)
        XCTAssertNil(try allNull.first())
        XCTAssertNil(try allNull.last())
        let empty = try MetalArray<Int32>([Int32](), context: .shared)
        XCTAssertNil(try empty.first())
        XCTAssertNil(try empty.last())

        // The same over a slice whose first rows are null.
        let s = try a.slice(offset: 1, length: 6)
        XCTAssertEqual(try s.first(skipNulls: true), 7)
        XCTAssertEqual(try s.last(skipNulls: true), 9)
        XCTAssertNil(try s.first(skipNulls: false))
    }

    // MARK: - compiled constants that must track the dispatch

    /// `RadixSelectSource` hard-codes `#define RS_SUBS 8u` and sizes three threadgroup arrays and one
    /// sub-block index with it, while the host dispatches `Dispatch.threadgroupSize` threads and
    /// `KernelSource.prelude` hard-codes `#define TG 256u`. If the three ever disagree, `rs_histogram`
    /// writes `parts[sgid]` and `hist[sgid * RS_RADIX]` out of bounds and `sb = tgid * RS_SUBS + sgid`
    /// collides across threadgroups — silent threadgroup-memory corruption, not a crash.
    func testRadixSelectSubBlockCountTracksTheThreadgroupSize() {
        XCTAssertEqual(RadixSelectSource.subsPerGroup, Dispatch.threadgroupSize / 32,
                       "RS_SUBS must be TG / 32")
        XCTAssertTrue(KernelSource.prelude.contains("#define TG \(Dispatch.threadgroupSize)u"),
                      "the MSL prelude's TG must be Dispatch.threadgroupSize")
        XCTAssertTrue(RadixSelectSource.source(kind: "u32", V: "uint", K: "uint").contains("#define RS_SUBS \(RadixSelectSource.subsPerGroup)u"))
        // `radix_scatter` advances a digit's base only from threads with `lid < RADIX`, so a
        // threadgroup smaller than the 256-entry radix would leave the high digits' bases frozen and
        // later chunks would overwrite earlier ones.
        XCTAssertGreaterThanOrEqual(Dispatch.threadgroupSize, 256,
                                    "radix_scatter's per-digit base update needs TG >= RADIX")
    }
}
