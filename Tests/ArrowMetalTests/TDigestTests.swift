import XCTest
@testable import ArrowMetal

/// `tdigest` reads its answer out of the sorted column instead of walking every value through
/// `TDigest.add` (`Kernels/TDigestGPU.swift` derives why the two are the same digest). These tests
/// hold the closed form against the walk it replaced, value for value.
final class TDigestTests: XCTestCase {

    static let quantiles: [Double] = [0, 0.001, 0.01, 0.1, 0.25, 1.0 / 3, 0.5, 0.75, 0.9, 0.99, 1, -0.5, 1.5]

    /// The digest of a sorted stream of unit weights, built the slow way.
    func walk(_ sorted: [Double], delta: Double = 100) -> TDigest {
        var d = TDigest(delta: delta)
        for v in sorted { d.add(v) }
        return d
    }

    /// The claim in one line: `sortedQuantile` is `TDigest.quantile` over the digest `add` builds.
    func testSortedQuantileMatchesTheWalk() throws {
        var g = SystemRandomNumberGenerator()
        for n in [0, 1, 2, 3, 4, 5, 17, 64, 1000, 4097] {
            for shape in 0..<3 {
                var values: [Double]
                switch shape {
                case 0: values = (0..<n).map { _ in Double.random(in: -1000...1000, using: &g) }
                case 1: values = (0..<n).map { _ in 7.5 }                       // every value equal
                default: values = (0..<n).map { Double($0) * 1e-9 - 1e-6 }      // tight, near zero
                }
                values.sort()
                let digest = walk(values)
                for q in TDigestTests.quantiles {
                    let want = digest.quantile(q)
                    let got = TDigest.sortedQuantile(q, count: values.count) { values[$0] }
                    XCTAssertEqual(got, want, "n=\(n) shape=\(shape) q=\(q)")
                }
            }
        }
    }

    /// The compression is not a parameter the answer depends on — the digest never merges — so a
    /// different `delta` must give the very same numbers.
    func testSortedQuantileIgnoresCompressionExactlyAsTheWalkDoes() throws {
        var g = SystemRandomNumberGenerator()
        let values = (0..<2001).map { _ in Double.random(in: 0...1, using: &g) }.sorted()
        for delta in [1.0, 20, 100, 500, 1000] {
            let digest = walk(values, delta: delta)
            for q in TDigestTests.quantiles {
                XCTAssertEqual(TDigest.sortedQuantile(q, count: values.count) { values[$0] },
                               digest.quantile(q), "delta=\(delta) q=\(q)")
            }
        }
    }

    /// End to end on the GPU: nulls, NaN and an all-NaN column, against the walk over ArrowMetal's own
    /// sorted output — the stream the previous implementation was fed.
    func testColumnTDigestMatchesTheWalkOverTheSortedColumn() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        for n in [0, 1, 2, 33, 4097, 100_003] {
            for withNaN in [false, true] {
                var values: [Double?] = []
                for i in 0..<n {
                    if i % 9 == 4 { values.append(nil) }
                    else if withNaN && i % 11 == 3 { values.append(.nan) }
                    else { values.append(Double.random(in: -500...500, using: &g)) }
                }
                let a = try MetalArray<Double>(values)
                let m = a.validCount
                let sorted = try a.sorted()
                let stream = withExtendedLifetime(sorted) { () -> [Double] in
                    let p = sorted.valuePointer
                    return (0..<m).map { p[$0] }
                }
                let digest = walk(stream)
                let got = try a.tdigest(TDigestTests.quantiles)
                for (i, q) in TDigestTests.quantiles.enumerated() {
                    XCTAssertEqual(got[i], digest.quantile(q), "n=\(n) nan=\(withNaN) q=\(q)")
                }
            }
        }

        // Every value NaN: `add` counts none of them, so the digest is empty and every quantile is nil.
        let allNaN = try MetalArray<Double>([Double](repeating: .nan, count: 100))
        XCTAssertEqual(try allNaN.tdigest([0, 0.5, 1]), [nil, nil, nil])
        // Every value null: the same, through the `validCount` guard.
        let allNull = try MetalArray<Double>([Double?](repeating: nil, count: 100))
        XCTAssertEqual(try allNull.tdigest([0, 0.5, 1]), [nil, nil, nil])
        // Nothing at all.
        XCTAssertEqual(try MetalArray<Double>([Double]()).tdigest([0.5]), [nil])
    }

    /// Integer columns go through the same path with no NaN boundary to find.
    func testIntegerColumnTDigest() throws {
        try requireRealGPU()
        let values = (0..<5000).map { Int64(($0 &* 7919) % 1000) }
        let a = try MetalArray<Int64>(values)
        let digest = walk(values.map(Double.init).sorted())
        for q in TDigestTests.quantiles {
            XCTAssertEqual(try a.tdigest([q]).first ?? nil, digest.quantile(q), "q=\(q)")
        }
    }
}
