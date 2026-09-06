import XCTest
@testable import ArrowMetal

/// The GPU utf8 / binary sort. The reference is Swift's own byte-wise comparison of the UTF-8 bytes,
/// which is what Arrow (and therefore pyarrow's `array_sort_indices` and Polars' `arg_sort`) defines for
/// these types — not Unicode collation.
final class StringSortTests: XCTestCase {
    /// Byte-wise lexicographic order, the Arrow contract.
    private func bytewiseSorted(_ xs: [String]) -> [String] {
        xs.sorted { a, b in Array(a.utf8).lexicographicallyPrecedes(Array(b.utf8)) }
    }

    func testSortsByBytesNotCollation() throws {
        try requireRealGPU()
        // Upper case sorts before lower case by bytes, and "Z" (0x5A) before "a" (0x61) — a collation
        // aware sort would interleave them.
        let xs = ["banana", "Apple", "apple", "Banana", "Zebra", "aardvark", "", "z"]
        let a = try MetalStringArray(xs.map { Optional($0) }, context: .shared)
        XCTAssertEqual(try a.sorted().toArray().map { $0! }, bytewiseSorted(xs))
    }

    func testPrefixesSortBeforeTheirExtensions() throws {
        try requireRealGPU()
        // Also covers the embedded-NUL case a zero-padded key would get wrong: "ab" must precede "ab\0".
        let xs = ["ab", "ab\u{0}", "abc", "a", "abcdefgh", "abcdefg", "abcdefghi", ""]
        let a = try MetalStringArray(xs.map { Optional($0) }, context: .shared)
        XCTAssertEqual(try a.sorted().toArray().map { $0! }, bytewiseSorted(xs))
    }

    func testLongStringsSpanManyChunks() throws {
        try requireRealGPU()
        // Rows that agree for 40 bytes and differ only at the end force several radix passes.
        let prefix = String(repeating: "x", count: 40)
        let xs = (0..<200).map { prefix + String(format: "%04d", (($0 &* 37) % 200)) }
        let a = try MetalStringArray(xs.map { Optional($0) }, context: .shared)
        XCTAssertEqual(try a.sorted().toArray().map { $0! }, bytewiseSorted(xs))
    }

    func testNullsLastAndStableInBothDirections() throws {
        try requireRealGPU()
        let xs: [String?] = ["b", nil, "a", "b", nil, "a", "c"]
        let a = try MetalStringArray(xs, context: .shared)
        let asc = try a.argsort().toRawArray()
        XCTAssertEqual(asc.map { xs[Int($0)] }, ["a", "a", "b", "b", "c", nil, nil])
        // Stable: equal rows keep their original order, and the nulls come back in row order.
        XCTAssertEqual(asc, [2, 5, 0, 3, 6, 1, 4])
        let desc = try a.argsort(descending: true).toRawArray()
        XCTAssertEqual(desc.map { xs[Int($0)] }, ["c", "b", "b", "a", "a", nil, nil])
        XCTAssertEqual(desc, [6, 0, 3, 2, 5, 1, 4])
    }

    func testEmptyAndAllNull() throws {
        try requireRealGPU()
        XCTAssertEqual(try MetalStringArray([], context: .shared).argsort().length, 0)
        let allNull = try MetalStringArray([nil, nil, nil], context: .shared)
        XCTAssertEqual(try allNull.argsort().toRawArray(), [0, 1, 2])
    }

    func testLargeRandomColumnMatchesTheHostOrder() throws {
        try requireRealGPU()
        var seed: UInt64 = 0x9E3779B97F4A7C15
        func next() -> UInt64 { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed }
        let xs = (0..<50_000).map { _ -> String in
            let len = Int(next() % 20)
            return String((0..<len).map { _ in Character(UnicodeScalar(UInt8(32 + next() % 90))) })
        }
        let a = try MetalStringArray(xs.map { Optional($0) }, context: .shared)
        XCTAssertEqual(try a.sorted().toArray().map { $0! }, bytewiseSorted(xs))
    }

    func testBinaryColumnAndLexsortAcceptStringKeys() throws {
        try requireRealGPU()
        let region = try MetalStringArray(["west", "east", "west", "east"].map { Optional($0) }, context: .shared)
        let revenue = try MetalArray<Int64>([10, 30, 20, 5], context: .shared)
        let idx = try lexsortIndices([.string(region), .int64(revenue)], descending: [false, false])
        XCTAssertEqual(idx.toRawArray(), [3, 1, 0, 2])
    }

    func testAnyMetalArrayArgsortIndicesOnStrings() throws {
        try requireRealGPU()
        let a = try MetalStringArray(["c", "a", "b"].map { Optional($0) }, context: .shared)
        XCTAssertEqual(try AnyMetalArray.string(a).argsortIndices().toRawArray(), [1, 2, 0])
    }
}
