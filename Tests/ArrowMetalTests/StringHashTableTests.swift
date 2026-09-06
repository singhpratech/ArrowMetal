import XCTest
@testable import ArrowMetal

/// The GPU string hash table (`Kernels/StringHashTable.swift`): the path `dictionaryEncode` and
/// `GroupByKeys` take for `utf8` and `binary` columns.
///
/// Every case is checked three ways — against the host hash map (`dictionaryEncodeCPU`, the oracle),
/// against the argsort path it replaced (`dictionaryEncodeSorted`), and, where the shape allows it,
/// against a Swift dictionary group-by — because the whole point of the change is that the answer did
/// not move.
final class StringHashTableTests: XCTestCase {

    // MARK: - helpers

    /// First-seen dictionary encoding on the host: the definition every path is measured against.
    func oracle(_ strings: [String?]) -> (codes: [Int32?], unique: [String]) {
        var map: [String: Int32] = [:]
        var unique: [String] = []
        var codes: [Int32?] = []
        codes.reserveCapacity(strings.count)
        for s in strings {
            guard let s else { codes.append(nil); continue }
            if let c = map[s] { codes.append(c) } else {
                let c = Int32(unique.count); map[s] = c; unique.append(s); codes.append(c)
            }
        }
        return (codes, unique)
    }

    /// Checks the hash table against the oracle, against the sort path, and through `GroupByKeys`.
    func check(_ strings: [String?], _ what: String, file: StaticString = #filePath, line: UInt = #line) throws {
        let a = try MetalStringArray(strings)
        let (expCodes, expUnique) = oracle(strings)

        let (codes, unique) = try a.dictionaryEncodeHashTable()
        XCTAssertEqual(codes.toArray(), expCodes, "\(what): codes", file: file, line: line)
        XCTAssertEqual(unique.toArray().map { $0! }, expUnique, "\(what): dictionary", file: file, line: line)
        XCTAssertEqual(unique.nullCount, 0, "\(what): dictionary has no nulls", file: file, line: line)

        // The path it replaced, and the host hash map, must agree with it element for element.
        let sorted = try a.dictionaryEncodeSorted()
        XCTAssertEqual(codes.toArray(), sorted.codes.toArray(), "\(what): codes vs sort path", file: file, line: line)
        XCTAssertEqual(unique.toArray(), sorted.unique.toArray(), "\(what): dictionary vs sort path", file: file, line: line)
        let host = try a.dictionaryEncodeCPU()
        XCTAssertEqual(codes.toArray(), host.codes.toArray(), "\(what): codes vs host", file: file, line: line)
        XCTAssertEqual(unique.toArray(), host.unique.toArray(), "\(what): dictionary vs host", file: file, line: line)

        // Dense ids for GroupByKeys: the same ids, with nulls moved into their own last group.
        let nullId = Int32(expUnique.count)
        let (ids, cardinality) = try a.hashTableDenseIds()
        XCTAssertEqual(cardinality, expUnique.count + (strings.contains { $0 == nil } ? 1 : 0),
                       "\(what): cardinality", file: file, line: line)
        XCTAssertEqual(ids.nullCount, 0, "\(what): dense ids are never null", file: file, line: line)
        XCTAssertEqual(ids.toArray(), expCodes.map { $0 ?? nullId }, "\(what): dense ids", file: file, line: line)

        // And the aggregate on top: one sum per distinct key, labelled by the key it belongs to.
        guard !strings.isEmpty else { return }
        let values = try MetalArray<Int64>((0..<strings.count).map { Int64($0 % 97) })
        let gbk = try GroupByKeys(columns: [.string(a)])
        let sums = try gbk.groupBy.sum(values)
        guard case .string(let keyCol) = try gbk.groupKeys()[0] else { return XCTFail("key column type") }
        var expected: [String: Int64] = [:]
        for (i, s) in strings.enumerated() { expected[s ?? "\u{0}null", default: 0] += Int64(i % 97) }
        XCTAssertEqual(gbk.groupCount, expected.count, "\(what): group count", file: file, line: line)
        for g in 0..<gbk.groupCount {
            let label = keyCol[g] ?? "\u{0}null"
            XCTAssertEqual(sums[g], expected[label], "\(what): sum for \(label)", file: file, line: line)
        }
    }

    /// A `utf8` column of `n` fixed-width keys drawn from `distinct` values, written straight into the
    /// Arrow buffers: a 50-million-element `[String]` would cost more than the test.
    func wideKeys(_ n: Int, distinct: Int, seed: UInt64) -> (MetalStringArray, [Int32]) {
        let width = 12
        var state = seed &* 6_364_136_223_846_793_005 &+ 1
        func next() -> UInt64 { state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407; return state >> 11 }
        let offsets = try! MetalArrowBuffer.allocate(byteCount: (n + 1) * 4, zeroed: false)
        let data = try! MetalArrowBuffer.allocate(byteCount: Swift.max(n * width, 1), zeroed: false)
        let op = offsets.mutableTyped(Int32.self), dp = data.mutableTyped(UInt8.self)
        var codes = [Int32](repeating: 0, count: n)
        let digits = Array("0123456789abcdefghijklmnopqrstuvwxyz".utf8)
        for i in 0..<n {
            let c = Int32(next() % UInt64(distinct))
            codes[i] = c
            op[i] = Int32(i * width)
            var v = Int(c)
            let base = i * width
            dp[base] = UInt8(ascii: "k")
            for j in stride(from: width - 1, through: 1, by: -1) { dp[base + j] = digits[v % 36]; v /= 36 }
        }
        op[n] = Int32(n * width)
        return (MetalStringArray(length: n, nullCount: 0, validity: nil, offsets: offsets, data: data), codes)
    }

    /// First-seen rank of each code, and how many distinct codes appeared.
    func firstSeenRanks(_ codes: [Int32], distinct: Int) -> (rank: [Int32], count: Int) {
        var rank = [Int32](repeating: -1, count: distinct)
        var next: Int32 = 0
        for c in codes where rank[Int(c)] < 0 { rank[Int(c)] = next; next += 1 }
        return (rank, Int(next))
    }

    // MARK: - shapes

    func testEmptyAndAllNull() throws {
        try requireRealGPU()
        try check([], "empty")
        try check([nil], "one null")
        try check([nil, nil, nil], "all null")
        try check([""], "one empty string")
        try check(["", "", nil, ""], "empty strings and nulls")
    }

    func testSizesAndCardinalities() throws {
        try requireRealGPU()
        var state: UInt64 = 12_345
        func next(_ m: Int) -> Int { state = state &* 6_364_136_223_846_793_005 &+ 1; return Int((state >> 33) % UInt64(m)) }
        for n in [0, 1, 33, 4097] {
            for distinct in [1, 7, 1000, 100_000] {
                var s: [String?] = []
                for i in 0..<n {
                    if i % 11 == 3 { s.append(nil); continue }
                    if i % 23 == 5 { s.append(""); continue }
                    s.append("key-\(next(distinct))")
                }
                try check(s, "n=\(n) distinct=\(distinct)")
            }
        }
    }

    func testOneMillionRowsAgainstTheSortPath() throws {
        try requireRealGPU()
        for distinct in [1, 7, 1000, 100_000] {
            let n = 1_000_003
            let (col, codes) = wideKeys(n, distinct: distinct, seed: UInt64(distinct))
            let (rank, groups) = firstSeenRanks(codes, distinct: distinct)

            let (ids, cardinality) = try col.hashTableDenseIds()
            XCTAssertEqual(cardinality, groups, "distinct=\(distinct)")
            let got = ids.toArray()
            var mismatch = -1
            for i in 0..<n where got[i] != rank[Int(codes[i])] { mismatch = i; break }
            XCTAssertEqual(mismatch, -1, "id mismatch at row \(mismatch), distinct=\(distinct)")

            // The dictionary decodes every row back to itself and matches the sort path exactly.
            let (hcodes, hunique) = try col.dictionaryEncodeHashTable()
            let (scodes, sunique) = try col.dictionaryEncodeSorted()
            XCTAssertEqual(hunique.length, groups, "dictionary size, distinct=\(distinct)")
            XCTAssertEqual(hunique.toArray(), sunique.toArray(), "dictionary vs sort path, distinct=\(distinct)")
            XCTAssertEqual(hcodes.toArray(), scodes.toArray(), "codes vs sort path, distinct=\(distinct)")
        }
    }

    // MARK: - hostile inputs

    func testLongAndEqualPrefixStrings() throws {
        try requireRealGPU()
        // 512 strings that agree on their first 200 bytes and differ only in the last few: the case a
        // prefix-only comparison would get wrong, and the case a weak hash would pile into one bucket.
        let prefix = String(repeating: "p", count: 200)
        var s: [String?] = []
        for i in 0..<2048 { s.append(prefix + String(format: "%04d", i % 512)) }
        s.append(nil)
        s.append(prefix)                                   // a proper prefix of every other string
        try check(s, "equal prefixes")

        // Long strings, including one of 100 KB, repeated so they must collapse onto one group each.
        let long = (0..<4).map { String(repeating: "long\($0)", count: 20_000) }
        var big: [String?] = []
        for i in 0..<200 { big.append(long[i % 4]) }
        big.insert(nil, at: 7)
        try check(big, "long strings")
    }

    func testMultiByteAndBinaryBytes() throws {
        try requireRealGPU()
        let s: [String?] = ["grape🍇", "grape", "🍇", "", nil, "grape🍇", "gräpe", "gräpe", "🍇🍇"]
        try check(s, "multi-byte")
    }

    /// Equality is decided by the bytes, never by the hash. Handing the table a key where *every* string
    /// hashes alike is the strongest form of that claim: the answer must still be the oracle's, it just
    /// costs a longer probe walk.
    func testDeliberatelyCollidingHashesStillSeparateStrings() throws {
        try requireRealGPU()
        var s: [String?] = []
        for i in 0..<600 { s.append(i % 7 == 3 ? nil : "collide-\(i % 50)") }
        let a = try MetalStringArray(s)
        let (expCodes, expUnique) = oracle(s)

        // Every non-null row shares one hash: 50 distinct strings, one bucket.
        let allEqual = try MetalArray<UInt64>([UInt64](repeating: 7, count: s.count))
        let r = try a.hashTableIds(nullId: 0, hashes: allEqual)
        XCTAssertEqual(r.groupCount, expUnique.count)
        let codes = MetalArray<Int32>(length: s.count, nullCount: a.nullCount, validity: a.validity,
                                      values: r.ids, context: a.context)
        XCTAssertEqual(codes.toArray(), expCodes, "colliding hashes must not merge distinct strings")

        // Half the strings collide pairwise, the other half hash normally.
        let real = try a.hash64().toRawArray()
        let mixed = try MetalArray<UInt64>((0..<s.count).map { i in
            guard let v = s[i] else { return UInt64(0) }
            return v.hasSuffix("0") || v.hasSuffix("1") ? 99 : real[i]
        })
        let r2 = try a.hashTableIds(nullId: 0, hashes: mixed)
        XCTAssertEqual(r2.groupCount, expUnique.count)
        let codes2 = MetalArray<Int32>(length: s.count, nullCount: a.nullCount, validity: a.validity,
                                       values: r2.ids, context: a.context)
        XCTAssertEqual(codes2.toArray(), expCodes, "partial collisions must not merge distinct strings")
    }

    /// A real MurmurHash3 seed-0 collision, found by search, driven through the shipping path. The two
    /// strings share the 32-bit hash the sort path keys on; they must land in different groups here too.
    func testRealHash32CollisionSurvives() throws {
        try requireRealGPU()
        let candidates = (0..<400_000).map { "c\($0)" }
        let hashes = try MetalStringArray(candidates).hash32().toRawArray()
        var seen: [UInt32: String] = [:]
        var pair: (String, String)? = nil
        for (i, h) in hashes.enumerated() {
            if let prev = seen[h], prev != candidates[i] { pair = (prev, candidates[i]); break }
            seen[h] = candidates[i]
        }
        let (x, y) = try XCTUnwrap(pair, "no murmur3 seed-0 collision found among 400k candidates")
        var interleaved: [String?] = []
        for i in 0..<2000 { interleaved.append(i % 2 == 0 ? x : y) }
        interleaved.insert(nil, at: 17)
        try check(interleaved, "real 32-bit hash collision")
    }

    /// Starting from a table far too small forces the growth retry, which must land on the same answer.
    func testTableGrowthRetryProducesTheSameIds() throws {
        try requireRealGPU()
        let n = 200_003
        let (col, codes) = wideKeys(n, distinct: 100_000, seed: 99)
        let (rank, groups) = firstSeenRanks(codes, distinct: 100_000)
        // 1024 slots for ~100k distinct keys: three doublings of eight before the table is big enough.
        let r = try col.hashTableIds(nullId: nil, initialSlots: 1024)
        XCTAssertEqual(r.groupCount, groups)
        let ids = MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: r.ids, context: col.context)
        let got = ids.toArray()
        var mismatch = -1
        for i in 0..<n where got[i] != rank[Int(codes[i])] { mismatch = i; break }
        XCTAssertEqual(mismatch, -1, "id mismatch at row \(mismatch) after a table growth retry")
    }

    // MARK: - full scale

    /// 50 million rows and ten million distinct keys: the size the design exists for. The oracle is the
    /// code each row was generated from, so it costs one pass instead of a 50-million-entry dictionary.
    func testFiftyMillionRowsTenMillionKeys() throws {
        try requireRealGPU()
        let n = 50_000_003
        let distinct = 10_000_000
        let (col, codes) = wideKeys(n, distinct: distinct, seed: 7)
        let (ids, cardinality) = try col.hashTableDenseIds()

        // One host pass does the whole oracle: hand out first-seen ranks, count the rows per group and
        // compare every id as it goes. A 50-million-entry dictionary would cost more than the test.
        var rank = [Int32](repeating: -1, count: distinct)
        var rows = [Int64](repeating: 0, count: distinct)
        var groups: Int32 = 0
        var mismatch = -1
        let got = ids.valuePointer
        for i in 0..<n {
            let c = Int(codes[i])
            if rank[c] < 0 { rank[c] = groups; groups += 1 }
            rows[Int(rank[c])] += 1
            if got[i] != rank[c] && mismatch < 0 { mismatch = i }
        }
        XCTAssertEqual(mismatch, -1, "id mismatch at row \(mismatch) of 50M")
        XCTAssertEqual(cardinality, Int(groups))

        // And the aggregate on top: rows per group, against the same oracle.
        let counts = try GroupBy(keys: ids, keyCount: Int(groups)).count()
        let cp = counts.valuePointer
        var bad = -1
        for g in 0..<Int(groups) where cp[g] != rows[g] { bad = g; break }
        XCTAssertEqual(bad, -1, "row count mismatch for group \(bad)")
    }
}
