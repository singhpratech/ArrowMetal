import XCTest
@testable import ArrowMetal

/// Top-k selection must agree with the full sort it replaces, index for index.
final class TopKTests: XCTestCase {
    private struct Rng: RandomNumberGenerator {
        var s: UInt64
        mutating func next() -> UInt64 {
            s &+= 0x9E37_79B9_7F4A_7C15
            var z = s
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    /// The oracle: what `topK` did before there was a selection kernel — a full argsort, then a slice.
    private func viaSort<T: ArrowPrimitive>(_ a: MetalArray<T>, _ k: Int, largest: Bool) throws -> [Int32] {
        let idx = try a.argsort(descending: largest)
        return try idx.slice(offset: 0, length: Swift.min(k, idx.length)).toRawArray()
    }

    private func check<T: ArrowPrimitive>(_ a: MetalArray<T>, _ k: Int, largest: Bool,
                                          file: StaticString = #filePath, line: UInt = #line) throws {
        let got = try a.topK(k, largest: largest).toRawArray()
        let want = try viaSort(a, k, largest: largest)
        XCTAssertEqual(got, want, "\(T.self) n=\(a.length) k=\(k) largest=\(largest)", file: file, line: line)
    }

    func testAgainstSortOnFiveMillion() throws {
        try requireRealGPU()
        var rng = Rng(s: 0xA11CE)
        let n = 5_000_000
        // Heavy duplication (values repeat about 5000 times) so tie-breaking by row index is exercised.
        let raw: [Int64] = (0..<n).map { _ in Int64.random(in: 0..<1000, using: &rng) }
        let dense = try MetalArray<Int64>(raw)
        let nullable = try MetalArray<Int64>(raw.enumerated().map { $0.offset % 17 == 0 ? nil : $0.element })
        let spread = try MetalArray<Int64>((0..<n).map { _ in Int64.random(in: Int64.min...Int64.max, using: &rng) })
        let doubles = try MetalArray<Double>((0..<n).map { i in
            i % 2003 == 0 ? Double.nan : Double.random(in: -1e9...1e9, using: &rng)
        })
        for k in [1, 10, 100, 1000] {
            for largest in [true, false] {
                try check(dense, k, largest: largest)
                try check(nullable, k, largest: largest)
                try check(spread, k, largest: largest)
                try check(doubles, k, largest: largest)
            }
        }
    }

    func testSmallAndAwkwardShapes() throws {
        try requireRealGPU()
        var rng = Rng(s: 0xBEEF)
        for n in [0, 1, 2, 255, 256, 257, 1023, 1024, 1025, 4096, 100_003] {
            let a = try MetalArray<Int32>((0..<n).map { _ in
                Int.random(in: 0..<8, using: &rng) == 0 ? nil : Int32.random(in: -50...50, using: &rng)
            })
            let f = try MetalArray<Float>((0..<n).map { i in i % 31 == 0 ? Float.nan : Float.random(in: -1...1, using: &rng) })
            let u = try MetalArray<UInt64>((0..<n).map { _ in UInt64.random(in: 0...9, using: &rng) })
            for k in [1, 2, 100, 1000, 1024, 1025] where k <= Swift.max(n, 1) {
                for largest in [true, false] {
                    try check(a, k, largest: largest)
                    try check(f, k, largest: largest)
                    try check(u, k, largest: largest)
                }
            }
            // k larger than the array: both paths clamp to the array length.
            XCTAssertEqual(try a.topK(n + 10).length, n)
        }
        XCTAssertEqual(try MetalArray<Int32>([5, 1, 3]).topK(0).length, 0)
    }

    /// Every row null, or fewer non-null rows than k: the selection kernel steps aside for the sort,
    /// which still has to place the null rows.
    func testMostlyNull() throws {
        try requireRealGPU()
        let a = try MetalArray<Int64>([nil, 7, nil, nil, 3, nil])
        try check(a, 4, largest: true)
        try check(a, 4, largest: false)
        try check(a, 2, largest: true)
        let allNull = try MetalArray<Int64>([Int64?](repeating: nil, count: 500))
        try check(allNull, 10, largest: true)
    }

    /// The result is what the caller actually wants: the k largest values, in order.
    func testValuesNotJustIndices() throws {
        try requireRealGPU()
        var rng = Rng(s: 7)
        let vals = (0..<200_000).map { _ in Int64.random(in: -1_000_000...1_000_000, using: &rng) }
        let a = try MetalArray<Int64>(vals)
        let top = try a.take(try a.topK(50)).toRawArray()
        XCTAssertEqual(top, Array(vals.sorted(by: >).prefix(50)))
        let bottom = try a.take(try a.topK(50, largest: false)).toRawArray()
        XCTAssertEqual(bottom, Array(vals.sorted().prefix(50)))
    }
}
