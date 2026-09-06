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
        XCTAssertEqual(try a.equals(b).toArray(), zip(sample, sample.reversed()).map { x, y in (x == nil || y == nil) ? nil : x == y })
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
