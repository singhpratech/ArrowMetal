import XCTest
@testable import ArrowMetal

/// The radix argsort at the sizes and key shapes where its two scale-dependent decisions bite:
/// the SIMD-ballot rank inside a 256-thread chunk, and the constant-digit pass skip, which only
/// switches on above 2^18 rows because it costs a command-buffer boundary to read the answer back.
///
/// Every case is checked against a host stable sort in Arrow's total order (nulls at the requested
/// end, every NaN one value after +inf, -0.0 tied with 0.0), not against another GPU path, and the
/// comparison is on the whole permutation — the sort is stable, so the permutation is unique and
/// "some valid order" is not the bar.
final class RadixSortScaleTests: XCTestCase {

    struct RNG: RandomNumberGenerator {
        var state: UInt64
        init(_ seed: UInt64) { state = seed }
        mutating func next() -> UInt64 {
            state = state &+ 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    /// Chunk edges (the ballot rank works 32 lanes at a time inside a 256-element chunk), block edges,
    /// and the two rows either side of the pass-skip threshold.
    static let sizes = [0, 1, 2, 31, 32, 33, 63, 64, 65, 255, 256, 257, 511, 512, 513,
                        4095, 4096, 4097, 65_536, (1 << 18) - 1, 1 << 18, (1 << 18) + 1]

    /// The permutation a stable sort in Arrow's order produces, computed on the host.
    func expected(_ vals: [Int64?], descending: Bool, atStart: Bool) -> [Int32] {
        let present = vals.enumerated().filter { $0.element != nil }
        let sorted = present.sorted { x, y in
            let a = x.element!, b = y.element!
            if a != b { return descending ? a > b : a < b }
            return x.offset < y.offset
        }.map { Int32($0.offset) }
        let nulls = vals.enumerated().filter { $0.element == nil }.map { Int32($0.offset) }
        return atStart ? nulls + sorted : sorted + nulls
    }

    func run(_ label: String, _ vals: [Int64?], file: StaticString = #filePath, line: UInt = #line) throws {
        let a = try MetalArray<Int64>(vals)
        for descending in [false, true] {
            for placement in [NullPlacement.atEnd, .atStart] {
                let got = try a.argsort(descending: descending, nullPlacement: placement).toRawArray()
                let want = expected(vals, descending: descending, atStart: placement == .atStart)
                XCTAssertEqual(got, want,
                               "\(label) n=\(vals.count) descending=\(descending) nulls=\(placement)",
                               file: file, line: line)
            }
        }
    }

    // MARK: - key shapes

    /// Full-width random keys: every one of the eight digits varies, so no pass is skipped.
    func testRandomKeysAtEveryBoundary() throws {
        try requireRealGPU()
        var rng = RNG(0xA11C_E501)
        for n in Self.sizes {
            try run("random", (0..<n).map { _ in Int64(bitPattern: rng.next()) })
        }
    }

    /// One repeated value: every digit is constant, every pass is an identity permutation, and the
    /// answer is the identity — which is also the only stable answer.
    func testAllEqualKeys() throws {
        try requireRealGPU()
        for n in Self.sizes {
            let a = try MetalArray<Int64>([Int64](repeating: 42, count: n))
            XCTAssertEqual(try a.argsort().toRawArray(), (0..<n).map { Int32($0) }, "all-equal n=\(n)")
            XCTAssertEqual(try a.argsort(descending: true).toRawArray(), (0..<n).map { Int32($0) },
                           "all-equal descending n=\(n)")
        }
    }

    /// A narrow range (0...999) leaves the top six digits constant above 2^18 rows, so six of the eight
    /// passes are dropped. The result must not change for it.
    func testConstantHighDigitsSkipPasses() throws {
        try requireRealGPU()
        var rng = RNG(0xC0_5A17)
        for n in Self.sizes {
            try run("narrow", (0..<n).map { _ in Int64(rng.next() % 1000) })
        }
    }

    /// The mirror image: the low bits are constant and the high ones vary, so it is the *first* passes
    /// that drop — the case where the already-computed digit-0 histogram has to be thrown away.
    func testConstantLowDigitsSkipPasses() throws {
        try requireRealGPU()
        var rng = RNG(0x10_B175)
        for n in Self.sizes {
            try run("low-constant", (0..<n).map { _ in Int64(bitPattern: rng.next() & 0xFFFF_FFFF_0000_0000) })
        }
    }

    /// Sorted input, reverse-sorted input, and a run-length shape: the cases a comparison sort would
    /// treat specially and a radix sort must not.
    func testMonotonicAndRunLengthShapes() throws {
        try requireRealGPU()
        for n in [4096, 65_536, (1 << 18) + 1] {
            try run("ascending", (0..<n).map { Int64($0) })
            try run("descending", (0..<n).map { Int64(n - $0) })
            try run("runs", (0..<n).map { Int64($0 / 97) })
        }
    }

    /// Nulls at every density, at both placements, at both directions.
    func testNullsAtEveryDensity() throws {
        try requireRealGPU()
        var rng = RNG(0x4E_1177)
        for n in [257, 4097, 65_536, (1 << 18) + 1] {
            for density in [1, 4, 2] {                      // one in 1000, one in 4, one in 2
                let vals: [Int64?] = (0..<n).map { _ in
                    let r = rng.next()
                    let isNull = density == 1 ? (r % 1000 == 0) : (r % UInt64(density) == 0)
                    return isNull ? nil : Int64(bitPattern: rng.next())
                }
                try run("nulls-1-in-\(density)", vals)
            }
        }
        try run("all-null", [Int64?](repeating: nil, count: 4097))
    }

    // MARK: - floats

    /// The float key map folds -0.0 onto +0.0 and every NaN onto one value after +inf. At scale the
    /// pass skip and the ballot rank both see those folded keys, so the special values are checked
    /// against a host order that folds them the same way.
    func testFloatSpecialValuesAtScale() throws {
        try requireRealGPU()
        var rng = RNG(0xF10A_7000)
        let specials: [Double] = [0.0, -0.0, .nan, -.nan, .infinity, -.infinity,
                                  .leastNonzeroMagnitude, -.leastNonzeroMagnitude,
                                  .greatestFiniteMagnitude, -.greatestFiniteMagnitude, 1, -1]
        for n in [513, 4097, 65_536, (1 << 18) + 1] {
            let vals: [Double] = (0..<n).map { i in
                if i % 7 == 0 { return specials[Int(rng.next() % UInt64(specials.count))] }
                return Double(rng.next() % 2_000_001) - 1_000_000.0
            }
            let a = try MetalArray<Double>(vals)
            for descending in [false, true] {
                let got = try a.argsort(descending: descending).toRawArray()
                // Arrow's order: NaN is one value after +inf and stays at the end when reversed;
                // -0.0 ties with 0.0 and the tie is broken by the input position.
                let want = (0..<n).sorted { i, j in
                    let x = vals[i], y = vals[j]
                    if x.isNaN || y.isNaN {
                        if x.isNaN && y.isNaN { return i < j }
                        return y.isNaN
                    }
                    if x != y { return descending ? x > y : x < y }
                    return i < j
                }.map { Int32($0) }
                XCTAssertEqual(got, want, "float specials n=\(n) descending=\(descending)")
            }
        }
    }

    /// `sorted()` is `take(argsort())`, so the values must come back in the permutation's order with
    /// their own bits intact — a -0.0 that tied with 0.0 in the *order* still reads back as -0.0.
    func testSortedValuesKeepTheirBits() throws {
        try requireRealGPU()
        var rng = RNG(0x5017_ED00)
        let n = (1 << 18) + 5
        let vals: [Double] = (0..<n).map { i in i % 11 == 0 ? -0.0 : Double(rng.next() % 1_000_000) }
        let out = try MetalArray<Double>(vals).sorted().toRawArray()
        // The stable permutation, on the host, then the values it selects — bit patterns and all.
        let order = (0..<n).sorted { vals[$0] != vals[$1] ? vals[$0] < vals[$1] : $0 < $1 }
        XCTAssertEqual(out.map { $0.bitPattern }, order.map { vals[$0].bitPattern },
                       "sorted float64 keeps every bit, -0.0 included")
        XCTAssertEqual(out.filter { $0.sign == .minus }.count, vals.filter { $0.sign == .minus }.count,
                       "every -0.0 survived the sort")
    }

    // MARK: - narrower key widths

    /// 32-bit keys take four passes rather than eight and go through the same scatter.
    func testInt32AndFloat32AtScale() throws {
        try requireRealGPU()
        var rng = RNG(0x3232_3232)
        for n in [257, 4097, (1 << 18) + 1] {
            let i32: [Int32] = (0..<n).map { _ in Int32(truncatingIfNeeded: rng.next()) }
            let a = try MetalArray<Int32>(i32)
            let orderI: [Int] = (0..<n).sorted { (x: Int, y: Int) -> Bool in
                if i32[x] != i32[y] { return i32[x] < i32[y] }
                return x < y
            }
            XCTAssertEqual(try a.argsort().toRawArray(), orderI.map { Int32($0) }, "int32 n=\(n)")

            let f32: [Float] = (0..<n).map { _ in Float(rng.next() % 100_000) / 7.0 }
            let b = try MetalArray<Float>(f32)
            let orderF: [Int] = (0..<n).sorted { (x: Int, y: Int) -> Bool in
                if f32[x] != f32[y] { return f32[x] < f32[y] }
                return x < y
            }
            XCTAssertEqual(try b.argsort().toRawArray(), orderF.map { Int32($0) }, "float32 n=\(n)")
        }
    }
}
