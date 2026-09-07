import XCTest
@testable import ArrowMetal

/// `sorted()` when it does not gather.
///
/// Above 2^18 rows the sorted values are rebuilt from the sort's own keys instead of being taken
/// through the permutation, and the two values the key map is not injective on — -0.0, which shares
/// +0.0's key so that the two tie, and NaN, whose payloads all share one key — are copied back from
/// the source over the runs they occupy. So the interesting columns are the ones that hold those, in
/// every proportion from one row to all of them, with and without nulls, in both directions and at
/// both null placements. Every answer is checked against `take(argsort())`, bit pattern for bit
/// pattern, which is what `sorted()` used to be.
final class SortValuesTests: XCTestCase {

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

    /// `sorted()` against `take(argsort())` on the same column, in all four option combinations.
    ///
    /// The values are compared as bit patterns, so a -0.0 that came back as +0.0 or a NaN that came
    /// back canonicalised is a failure; the null slots are compared as nulls, since what a gather
    /// leaves under a null is not defined by Arrow.
    func check<T: ArrowPrimitive>(_ label: String, _ vals: [T?],
                                  file: StaticString = #filePath, line: UInt = #line) throws {
        let a = try MetalArray<T>(vals)
        for descending in [false, true] {
            for placement in [NullPlacement.atEnd, .atStart] {
                let what = "\(label) n=\(vals.count) descending=\(descending) nulls=\(placement)"
                let got = try a.sorted(descending: descending, nullPlacement: placement)
                let want = try a.take(try a.argsort(descending: descending, nullPlacement: placement))
                XCTAssertEqual(got.length, vals.count, what, file: file, line: line)
                XCTAssertEqual(got.nullCount, want.nullCount, "\(what): null count", file: file, line: line)
                let g = got.toArray(), w = want.toArray()
                for i in 0..<vals.count {
                    switch (g[i], w[i]) {
                    case (nil, nil): continue
                    case let (x?, y?):
                        XCTAssertEqual(bits(x), bits(y), "\(what): row \(i)", file: file, line: line)
                    default:
                        XCTFail("\(what): row \(i) is \(String(describing: g[i])), expected \(String(describing: w[i]))",
                                file: file, line: line)
                    }
                }
            }
        }
    }

    private func bits<T: ArrowPrimitive>(_ v: T) -> UInt64 {
        if let d = v as? Double { return d.bitPattern }
        if let f = v as? Float { return UInt64(f.bitPattern) }
        var out: UInt64 = 0
        withUnsafeBytes(of: v) { src in withUnsafeMutableBytes(of: &out) { $0.copyBytes(from: src) } }
        return out
    }

    /// Both sides of the threshold above which the keys are inverted rather than gathered, and one
    /// size that is not a multiple of anything.
    static let sizes = [1, 255, 4097, (1 << 18) - 1, (1 << 18) + 5, 700_003]

    func testFloat64WithNegativeZeroAndNaNAtEverySize() throws {
        try requireRealGPU()
        var rng = RNG(0x501F_0000)
        for n in Self.sizes {
            // Every mixture: plain values, -0.0, +0.0, NaN with two different payloads, infinities.
            let vals: [Double?] = (0..<n).map { i in
                switch i % 13 {
                case 0: return -0.0
                case 1: return 0.0
                case 2: return Double.nan
                case 3: return Double(bitPattern: 0x7FF8_0000_0000_0007)   // a different NaN payload
                case 4: return .infinity
                case 5: return -.infinity
                default: return Double(rng.next() % 1_000_003) - 500_000
                }
            }
            try check("float64 specials", vals)
            // The same values with a tenth of the rows null.
            try check("float64 specials + nulls", vals.enumerated().map { $0.offset % 10 == 3 ? nil : $0.element })
        }
    }

    /// A column that is nothing but the values the map cannot invert: the fix-up covers the whole
    /// output, which is where `sorted()` gives up and gathers all of it.
    func testColumnsMadeEntirelyOfSpecials() throws {
        try requireRealGPU()
        let n = (1 << 18) + 5
        try check("all -0.0", [Double?](repeating: -0.0, count: n))
        try check("all NaN", [Double?](repeating: .nan, count: n))
        try check("half -0.0 half 0.0", (0..<n).map { $0 % 2 == 0 ? -0.0 : 0.0 })
        try check("all null", [Double?](repeating: nil, count: n))
        try check("one value, rest null", (0..<n).map { $0 == 7 ? 1.5 : nil })
        // More than half the output is the null block, but the value block still has to come out of
        // the keys: the row numbers over it are the partition's, not the sort's, unless the sort was
        // carrying them.
        var rng = RNG(0x8A1F_0000)
        try check("more than half null, no specials",
                  (0..<n).map { i in i % 2 == 0 ? nil : Double(rng.next() % 1_000_003) })
    }

    /// float32 takes the 32-bit key and the same inverse.
    func testFloat32() throws {
        try requireRealGPU()
        var rng = RNG(0x3232_0F32)
        let n = (1 << 18) + 5
        let vals: [Float?] = (0..<n).map { i in
            i % 11 == 0 ? -0.0 : (i % 101 == 0 ? Float.nan : Float(rng.next() % 100_003) / 8.0)
        }
        try check("float32 specials", vals)
        try check("float32 specials + nulls", vals.enumerated().map { $0.offset % 7 == 0 ? nil : $0.element })
    }

    /// The integer key maps are bijections, so every integer column takes the inverse with no fix-up
    /// but the null block.
    func testIntegersKeepEveryBit() throws {
        try requireRealGPU()
        var rng = RNG(0x1234_5678)
        let n = (1 << 18) + 5
        try check("int64 full range", (0..<n).map { _ in Int64(bitPattern: rng.next()) as Int64? })
        try check("uint64 above 2^63", (0..<n).map { _ in rng.next() as UInt64? })
        try check("int32", (0..<n).map { _ in Int32(truncatingIfNeeded: rng.next()) as Int32? })
        try check("uint32", (0..<n).map { _ in UInt32(truncatingIfNeeded: rng.next()) as UInt32? })
        try check("int64 with nulls", (0..<n).map { i in i % 3 == 0 ? nil : Int64(bitPattern: rng.next()) })
        try check("int16 widens to a 32-bit key",
                  (0..<n).map { i in i % 5 == 0 ? nil : Int16(truncatingIfNeeded: rng.next()) })
        // Halves the value block, which re-plans the passes' block layout around it.
        try check("int64 half null", (0..<n).map { i in i % 2 == 0 ? nil : Int64(bitPattern: rng.next()) })
    }

    /// A slice whose offset is not a multiple of 32 carries an Arrow element offset, which the key
    /// map, the partition's bitmap read and the fix-up gather must all see through the same
    /// normalisation.
    func testSlicedInput() throws {
        try requireRealGPU()
        var rng = RNG(0x5111_CED0)
        let n = (1 << 18) + 100
        let vals: [Double?] = (0..<n).map { i in
            i % 9 == 0 ? nil : (i % 11 == 0 ? -0.0 : Double(rng.next() % 1_000_003))
        }
        let a = try MetalArray<Double>(vals)
        for off in [1, 33, 64] {
            let s = try a.slice(offset: off, length: n - off)
            for placement in [NullPlacement.atEnd, .atStart] {
                let got = try s.sorted(nullPlacement: placement)
                let want = try s.take(try s.argsort(nullPlacement: placement))
                XCTAssertEqual(got.toArray().map { $0.map { $0.bitPattern } },
                               want.toArray().map { $0.map { $0.bitPattern } },
                               "slice at \(off), nulls \(placement)")
            }
        }
    }
}
