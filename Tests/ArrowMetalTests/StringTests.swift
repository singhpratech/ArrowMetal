import XCTest
@testable import ArrowMetal

final class StringTests: XCTestCase {
    let sample: [String?] = ["apple", "banana", nil, "", "apricot", "cherry", "app", "grape🍇", "banana", "Apple"]

    func testConstructionAndLengths() throws {
        try requireRealGPU()
        let a = try MetalStringArray(sample)
        XCTAssertEqual(a.toArray(), sample)
        XCTAssertEqual(a.nullCount, 1)
        XCTAssertEqual(try a.byteLength().toArray(), sample.map { $0.map { Int32($0.utf8.count) } })
        XCTAssertEqual(try a.charLength().toArray(), sample.map { $0.map { Int32($0.count) } })
        let empty = try MetalStringArray([String?]())
        XCTAssertEqual(empty.length, 0); XCTAssertEqual(try empty.byteLength().length, 0)
    }

    func testPredicates() throws {
        try requireRealGPU()
        let a = try MetalStringArray(sample)
        XCTAssertEqual(try a.equals("banana").toArray(), sample.map { $0.map { $0 == "banana" } })
        XCTAssertEqual(try a.startsWith("ap").toArray(), sample.map { $0.map { $0.hasPrefix("ap") } })
        XCTAssertEqual(try a.endsWith("e").toArray(), sample.map { $0.map { $0.hasSuffix("e") } })
        XCTAssertEqual(try a.contains("an").toArray(), sample.map { $0.map { $0.contains("an") } })
        XCTAssertEqual(try a.contains("").toArray(), sample.map { $0.map { _ in true } })
        XCTAssertEqual(try a.equals("").toArray(), sample.map { $0.map { $0.isEmpty } })
        let b = try MetalStringArray(sample.reversed())
        // Spelled out with explicit types: the one-line closure form is fine for Swift 6.3 but the
        // 6.1 compiler on the CI runner gives up type-checking it ("unable to type-check this
        // expression in reasonable time").
        let expectedEquals: [Bool?] = zip(sample, sample.reversed()).map { (x: String?, y: String?) -> Bool? in
            guard let x, let y else { return nil }
            return x == y
        }
        XCTAssertEqual(try a.equals(b).toArray(), expectedEquals)
        // large: 200k strings, word boundaries
        let big: [String?] = (0..<200_003).map { $0 % 17 == 0 ? nil : "row\($0 % 1000)" }
        let ba = try MetalStringArray(big)
        XCTAssertEqual(try ba.equals("row7").toArray(), big.map { $0.map { $0 == "row7" } })
        XCTAssertEqual(try ba.startsWith("row99").trueCount, big.compactMap { $0 }.filter { $0.hasPrefix("row99") }.count)
    }

    func testHashMatchesMurmur3Reference() throws {
        try requireRealGPU()
        // Reference values for MurmurHash3_x86_32 with seed 0.
        let known: [(String, UInt32)] = [("", 0), ("a", 0x3c2569b2), ("abc", 0xb3dd93fa), ("hello", 0x248bfa47), ("The quick brown fox jumps over the lazy dog", 0x2e4ff723)]
        let a = try MetalStringArray(known.map { $0.0 })
        XCTAssertEqual(try a.hash32().toRawArray(), known.map { $0.1 })
        // Determinism and distribution sanity
        let many = try MetalStringArray((0..<100_000).map { "key-\($0)" })
        let h = try many.hash32().toRawArray()
        XCTAssertGreaterThanOrEqual(Set(h).count, h.count - 10, "far more collisions than the birthday bound predicts")
    }

    func testFilterTakeGather() throws {
        try requireRealGPU()
        let a = try MetalStringArray(sample)
        let m = try a.startsWith("a").or(try a.equals("banana"))
        let f = try a.filter(m)
        XCTAssertEqual(f.toArray(), ["apple", "banana", "apricot", "app", "banana"])
        let t = try a.take(try MetalArray<Int32>([9, nil, 0, 7, 3]))
        XCTAssertEqual(t.toArray(), ["Apple", nil, "apple", "grape🍇", ""])
        XCTAssertEqual(t.nullCount, 1)
        XCTAssertThrowsError(try a.take(try MetalArray<Int32>([10])))
        let t64 = try a.take(try MetalArray<Int64>([1, 2]))
        XCTAssertEqual(t64.toArray(), ["banana", nil])
        // big
        let big: [String?] = (0..<300_000).map { $0 % 13 == 0 ? nil : String(repeating: "x", count: $0 % 7) + "\($0)" }
        let ba = try MetalStringArray(big)
        let mask = try ba.contains("77")
        XCTAssertEqual(try ba.filter(mask).toArray(), big.filter { $0?.contains("77") ?? false })
        let rev = try MetalArray<Int32>((0..<300_000).reversed().map { Int32($0) })
        XCTAssertEqual(try ba.take(rev).toArray(), big.reversed())
    }

    func testDictionaryEncodeAndGroupBy() throws {
        try requireRealGPU()
        let a = try MetalStringArray(sample)
        let (codes, unique) = try a.dictionaryEncode()
        XCTAssertEqual(unique.toArray(), ["apple", "banana", "", "apricot", "cherry", "app", "grape🍇", "Apple"])
        XCTAssertEqual(codes.toArray(), [0, 1, nil, 2, 3, 4, 5, 6, 1, 7])
        let vals = try MetalArray<Int64>((0..<10).map { Int64($0) })
        let sums = try codes.groupBy(keyCount: unique.length).sum(vals).toArray()
        XCTAssertEqual(sums[1], 1 + 8)
        XCTAssertEqual(sums[0], 0)
    }

    // MARK: - GPU dictionary encoding

    /// Same grouping as the host oracle: same decoded string per row, same partition of rows into
    /// groups, and (because the GPU path relabels into first-seen order) the very same codes.
    private func checkDictionaryEncode(_ strings: [String?], file: StaticString = #filePath, line: UInt = #line) throws {
        let a = try MetalStringArray(strings)
        let (codes, unique) = try a.dictionaryEncodeGPU()
        let (refCodes, refUnique) = try a.dictionaryEncodeCPU()
        let u = unique.toArray(), ru = refUnique.toArray()
        XCTAssertEqual(unique.nullCount, 0, "dictionary carries no nulls", file: file, line: line)
        XCTAssertEqual(Set(u.map { $0! }).count, u.count, "dictionary is distinct", file: file, line: line)
        // Every row decodes to itself, which is the property that actually matters.
        let decoded = codes.toArray().map { $0.map { u[Int($0)]! } }
        XCTAssertEqual(decoded, strings, "decoded rows", file: file, line: line)
        // Same partition of rows into groups as the oracle, and in fact the same labels.
        XCTAssertEqual(codes.toArray(), refCodes.toArray(), "codes", file: file, line: line)
        XCTAssertEqual(u, ru, "dictionary order", file: file, line: line)
    }

    func testDictionaryEncodeGPUMatchesHost() throws {
        try requireRealGPU()
        var rng = SystemRandomNumberGenerator()
        try checkDictionaryEncode(sample)
        try checkDictionaryEncode([])
        try checkDictionaryEncode([nil, nil, nil])
        try checkDictionaryEncode(["only"])
        try checkDictionaryEncode([""])
        try checkDictionaryEncode(["", nil, "", "x", ""])
        try checkDictionaryEncode((0..<5000).map { "v\($0)" })                      // all distinct
        try checkDictionaryEncode((0..<5000).map { _ in "same" })                    // all identical
        try checkDictionaryEncode((0..<200_003).map { i in
            i % 23 == 0 ? nil : "k-\(Int.random(in: 0..<3000, using: &rng))-\(String(repeating: "y", count: i % 5))"
        })
    }

    /// Two different strings that really do share a MurmurHash3 x86_32 seed-0 hash, found by search.
    /// They must still land in different dictionary entries, interleaved or not.
    func testDictionaryEncodeSurvivesRealHashCollision() throws {
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
        XCTAssertEqual(try MetalStringArray([x]).hash32().toRawArray()[0], try MetalStringArray([y]).hash32().toRawArray()[0])
        XCTAssertNotEqual(x, y)
        // Interleaved, which is the arrangement a hash-only grouping would get wrong.
        var interleaved: [String?] = []
        for i in 0..<2000 { interleaved.append(i % 2 == 0 ? x : y) }
        interleaved.insert(nil, at: 17)
        try checkDictionaryEncode(interleaved)
        try checkDictionaryEncode([x, y, x, "z", y, x])
        // And mixed into a large column.
        try checkDictionaryEncode((0..<100_000).map { i in i % 7 == 0 ? x : (i % 7 == 1 ? y : "f\(i % 500)") })
    }

    /// The collision detector itself: hand `encode` a key that deliberately puts distinct strings in one
    /// bucket and it must decline rather than emit wrong codes. This is the path a real 64-bit collision
    /// would take, and the reason `dictionaryEncode` is correct rather than merely probably correct.
    func testDictionaryEncodeRejectsCollidingKeys() throws {
        try requireRealGPU()
        let strings: [String?] = ["a", "b", "a", "b", "c", nil, "a"]
        let a = try MetalStringArray(strings)
        let allSame = try MetalArray<UInt64>(strings.map { $0 == nil ? nil : UInt64(0) })
        XCTAssertNil(try a.encode(keys: allSame), "one bucket for three distinct strings must be rejected")
        // A key that is faithful (equal strings share it, distinct strings do not) is accepted, and the
        // codes it produces are the oracle's.
        let faithful = try MetalArray<UInt64>(strings.map { s in s.map { UInt64($0.utf8.first ?? 0) } })
        let ok = try XCTUnwrap(try a.encode(keys: faithful))
        XCTAssertEqual(ok.codes.toArray(), try a.dictionaryEncodeCPU().codes.toArray())
        XCTAssertEqual(ok.unique.toArray(), ["a", "b", "c"])
        // Every row identical: a single bucket is correct here, so it is accepted.
        let same = try MetalStringArray(["q", "q", "q"])
        XCTAssertNotNil(try same.encode(keys: try MetalArray<UInt64>([0, 0, 0])))
    }

    func testScan() throws {
        try requireRealGPU()
        for n in [0, 1, 255, 256, 257, 70_000, 1_000_001] {
            let lens = try MetalArray<Int32>((0..<n).map { Int32($0 % 5) })
            let off = try lens.exclusiveScanToOffsets()
            let p = off.typed(Int32.self)
            var acc: Int32 = 0
            for i in 0..<n { XCTAssertEqual(p[i], acc, "n=\(n) i=\(i)"); acc += Int32(i % 5); if p[i] != acc - Int32(i % 5) { break } }
            XCTAssertEqual(p[n], acc)
        }
    }
}
