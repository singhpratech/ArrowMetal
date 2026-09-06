import XCTest
@testable import ArrowMetal

/// `expm1`, `log1p`, `logb`, `hypot` and the rounding family (`round(ndigits:mode:)`,
/// `round_to_multiple`, `round_binary`) in all ten Arrow `RoundMode`s.
///
/// The rounding oracles are exact — the kernels do the same scaling Arrow does, in the same order, so
/// the results must be bit-identical to a Swift restatement of the rule. The transcendental oracles are
/// Foundation's, compared in ulp: `float32` goes through the MSL library functions and `float64` through
/// the software binary64 of `Kernels/DoubleTranscendental.swift`, and neither claims correct rounding.
/// `testFloat64PrecisionAgainstFoundation` measures the error over 10^6 random inputs and asserts the
/// bounds documented on `DoubleTranscendental`.
final class MathExtraTests: XCTestCase {

    struct SeededRNG: RandomNumberGenerator {
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

    static let sizes = [0, 1, 33, 4097, 1_000_003]

    // MARK: - Oracles

    /// Distance in units of the last place, treating two NaNs as equal and any other mismatch in
    /// finiteness as infinite.
    static func ulps(_ a: Double, _ b: Double) -> Double {
        if a == b { return 0 }
        if a.isNaN || b.isNaN { return (a.isNaN && b.isNaN) ? 0 : .infinity }
        if !a.isFinite || !b.isFinite { return .infinity }
        return (a - b).magnitude / Swift.max(a.magnitude, b.magnitude).ulp
    }
    static func ulpsF(_ a: Float, _ b: Float) -> Double {
        if a == b { return 0 }
        if a.isNaN || b.isNaN { return (a.isNaN && b.isNaN) ? 0 : .infinity }
        if !a.isFinite || !b.isFinite { return .infinity }
        return Double((a - b).magnitude / Swift.max(a.magnitude, b.magnitude).ulp)
    }

    /// The same decimal literals the kernels' tables hold, so the oracle scales identically.
    static func pow10(_ k: Int) -> Double { Double("1e\(k)")! }

    static func roundToInteger(_ x: Double, _ mode: RoundMode) -> Double {
        guard x.isFinite else { return x }
        let t = x.rounded(.towardZero)
        if t == x { return x }
        let neg = x < 0
        let away = t + (neg ? -1 : 1)
        switch mode {
        case .down: return neg ? away : t
        case .up: return neg ? t : away
        case .towardsZero: return t
        case .towardsInfinity: return away
        default: break
        }
        let frac = Swift.abs(x) - Swift.abs(t)
        if frac < 0.5 { return t }
        if frac > 0.5 { return away }
        let even = t.truncatingRemainder(dividingBy: 2) == 0
        switch mode {
        case .halfDown: return neg ? away : t
        case .halfUp: return neg ? t : away
        case .halfTowardsZero: return t
        case .halfTowardsInfinity: return away
        case .halfToEven: return even ? t : away
        default: return even ? away : t
        }
    }

    static func roundDigits(_ x: Double, _ nd: Int, _ mode: RoundMode) -> Double {
        guard x.isFinite else { return x }
        if nd == 0 { return roundToInteger(x, mode) }
        if nd > 308 { return x }
        if nd < -308 { return x * 0 }
        let p = pow10(Swift.abs(nd))
        return nd > 0 ? roundToInteger(x * p, mode) / p : roundToInteger(x / p, mode) * p
    }

    static func roundToInteger(_ x: Float, _ mode: RoundMode) -> Float {
        Float(roundToInteger(Double(x), mode))
    }
    static func roundDigits(_ x: Float, _ nd: Int, _ mode: RoundMode) -> Float {
        guard x.isFinite else { return x }
        if nd == 0 { return roundToInteger(x, mode) }
        if nd > 38 { return x }
        if nd < -38 { return x * 0 }
        let p = Float("1e\(Swift.abs(nd))")!
        return nd > 0 ? roundToInteger(x * p, mode) / p : roundToInteger(x / p, mode) * p
    }

    /// Integer rounding on the quotient and remainder, the rule the integer kernels implement.
    static func roundToMultiple<T: FixedWidthInteger>(_ v: T, _ m: T, _ mode: RoundMode) -> T {
        guard m > 0 else { return v }
        let r = v % m
        if r == 0 { return v }
        let neg = v < 0
        let toward = v &- r
        let away = toward &+ (neg ? (0 &- m) : m)
        switch mode {
        case .down: return neg ? away : toward
        case .up: return neg ? toward : away
        case .towardsZero: return toward
        case .towardsInfinity: return away
        default: break
        }
        let ar = neg ? (0 &- r) : r
        let half = m / 2
        if ar > half { return away }
        if ar < half { return toward }
        if m % 2 != 0 { return toward }
        let even = (toward / m) % 2 == 0
        switch mode {
        case .halfDown: return neg ? away : toward
        case .halfUp: return neg ? toward : away
        case .halfTowardsZero: return toward
        case .halfTowardsInfinity: return away
        case .halfToEven: return even ? toward : away
        default: return even ? away : toward
        }
    }

    static func roundDigits<T: FixedWidthInteger>(_ v: T, _ nd: Int, _ mode: RoundMode) -> T {
        if nd >= 0 { return v }
        let k = -nd
        if k > 18 { return 0 }
        var p: UInt64 = 1
        for _ in 0..<k { p *= 10 }
        if p > UInt64(T.max) { return 0 }
        return roundToMultiple(v, T(p), mode)
    }

    // MARK: - Input builders

    func doubles(n: Int, nulls: Double, range: ClosedRange<Double>, rng: inout SeededRNG) -> [Double?] {
        (0..<n).map { _ in
            Double.random(in: 0..<1, using: &rng) < nulls ? nil : Double.random(in: range, using: &rng)
        }
    }

    // MARK: - expm1 / log1p / logb / hypot at every size

    func testTranscendentalsAtEverySize() throws {
        try requireRealGPU()
        var rng = SeededRNG(0x3141_5926)
        for n in Self.sizes {
            // expm1 over a range wide enough to exercise both the |x| < 2^-54 shortcut and the k != 0
            // argument reduction; log1p over (-1, big) so the Kahan correction is genuinely needed.
            let xs = doubles(n: n, nulls: 0.2, range: -20...20, rng: &rng)
            let small = doubles(n: n, nulls: 0.2, range: -1e-7...1e-7, rng: &rng)
            let pos = doubles(n: n, nulls: 0.2, range: 0.001...1000, rng: &rng)
            let ys = doubles(n: n, nulls: 0.2, range: -1e30...1e30, rng: &rng)

            let d = try MetalArray<Double>(xs), ds = try MetalArray<Double>(small)
            let dp = try MetalArray<Double>(pos), dy = try MetalArray<Double>(ys)
            try expectClose("float64 expm1 n=\(n)", try d.expm1().toArray(), xs.map { $0.map(Foundation.expm1) }, ulp: 2)
            try expectClose("float64 expm1 small n=\(n)", try ds.expm1().toArray(), small.map { $0.map(Foundation.expm1) }, ulp: 2)
            try expectClose("float64 log1p n=\(n)", try d.log1p().toArray(),
                            xs.map { $0.map { $0 <= -1 ? ($0 == -1 ? -Double.infinity : Double.nan) : Foundation.log1p($0) } }, ulp: 2)
            try expectClose("float64 log1p small n=\(n)", try ds.log1p().toArray(), small.map { $0.map(Foundation.log1p) }, ulp: 2)
            try expectClose("float64 logb n=\(n)", try dp.logb(2.0).toArray(),
                            pos.map { $0.map { Foundation.log($0) / Foundation.log(2.0) } }, ulp: 4)
            try expectClose("float64 hypot n=\(n)", try dy.hypot(dy).toArray(),
                            ys.map { $0.map { Foundation.hypot($0, $0) } }, ulp: 2)
            try expectClose("float64 hypot scalar n=\(n)", try dy.hypot(3.0).toArray(),
                            ys.map { $0.map { Foundation.hypot($0, 3.0) } }, ulp: 2)

            let f = try MetalArray<Float>(xs.map { $0.map(Float.init) })
            let fs = try MetalArray<Float>(small.map { $0.map(Float.init) })
            let fp = try MetalArray<Float>(pos.map { $0.map(Float.init) })
            let fy = try MetalArray<Float>(ys.map { $0.map { Float($0 / 1e25) } })
            try expectCloseF("float32 expm1 n=\(n)", try f.expm1().toArray(),
                             xs.map { $0.map { Float(Foundation.expm1(Double(Float($0)))) } }, ulp: 24)
            try expectCloseF("float32 expm1 small n=\(n)", try fs.expm1().toArray(),
                             small.map { $0.map { Float(Foundation.expm1(Double(Float($0)))) } }, ulp: 4)
            try expectCloseF("float32 log1p small n=\(n)", try fs.log1p().toArray(),
                             small.map { $0.map { Float(Foundation.log1p(Double(Float($0)))) } }, ulp: 8)
            try expectCloseF("float32 logb n=\(n)", try fp.logb(2.0).toArray(),
                             pos.map { $0.map { Float(Foundation.log(Double(Float($0))) / Foundation.log(2.0)) } }, ulp: 16)
            try expectCloseF("float32 hypot n=\(n)", try fy.hypot(3.0).toArray(),
                             ys.map { $0.map { Float(Foundation.hypot(Double(Float($0 / 1e25)), 3.0)) } }, ulp: 4)

            // Nulls come through untouched on every one of them.
            XCTAssertEqual(try d.expm1().nullCount, d.nullCount, "float64 expm1 nulls n=\(n)")
            XCTAssertEqual(try dy.hypot(dy).nullCount, dy.nullCount, "float64 hypot nulls n=\(n)")
        }
    }

    func expectClose(_ what: String, _ got: [Double?], _ want: [Double?], ulp: Double) throws {
        XCTAssertEqual(got.count, want.count, "\(what): length")
        var worst = 0.0, worstAt = -1
        for i in 0..<Swift.min(got.count, want.count) {
            guard let g = got[i], let w = want[i] else {
                XCTAssertTrue(got[i] == nil && want[i] == nil, "\(what): null mismatch at \(i)")
                continue
            }
            let e = Self.ulps(g, w)
            if e > worst { worst = e; worstAt = i }
        }
        XCTAssertLessThanOrEqual(worst, ulp, "\(what): worst \(worst) ulp at row \(worstAt)")
    }

    func expectCloseF(_ what: String, _ got: [Float?], _ want: [Float?], ulp: Double) throws {
        XCTAssertEqual(got.count, want.count, "\(what): length")
        var worst = 0.0, worstAt = -1
        for i in 0..<Swift.min(got.count, want.count) {
            guard let g = got[i], let w = want[i] else {
                XCTAssertTrue(got[i] == nil && want[i] == nil, "\(what): null mismatch at \(i)")
                continue
            }
            let e = Self.ulpsF(g, w)
            if e > worst { worst = e; worstAt = i }
        }
        XCTAssertLessThanOrEqual(worst, ulp, "\(what): worst \(worst) ulp at row \(worstAt)")
    }

    // MARK: - Special values

    func testTranscendentalSpecialValues() throws {
        try requireRealGPU()
        let d = try MetalArray<Double>([0, -0.0, .nan, .infinity, -.infinity, -1, -2, 1e-300, 1e300])
        let expm1 = try d.expm1().toRawArray()
        XCTAssertEqual(expm1[0], 0); XCTAssertEqual(expm1[1].sign, .minus, "expm1(-0) keeps the sign")
        XCTAssertTrue(expm1[2].isNaN)
        XCTAssertEqual(expm1[3], .infinity); XCTAssertEqual(expm1[4], -1)
        XCTAssertEqual(expm1[7], 1e-300, accuracy: 1e-316)
        XCTAssertEqual(expm1[8], .infinity)

        let log1p = try d.log1p().toRawArray()
        XCTAssertEqual(log1p[0], 0); XCTAssertEqual(log1p[1].sign, .minus, "log1p(-0) keeps the sign")
        XCTAssertTrue(log1p[2].isNaN)
        XCTAssertEqual(log1p[3], .infinity)
        XCTAssertTrue(log1p[4].isNaN, "log1p(-inf) is NaN")
        XCTAssertEqual(log1p[5], -.infinity, "log1p(-1) is -inf")
        XCTAssertTrue(log1p[6].isNaN, "log1p(-2) is NaN")

        // hypot: infinity wins even opposite a NaN, as IEEE-754 says.
        let a = try MetalArray<Double>([.infinity, .nan, 3, 0, 1e300])
        let b = try MetalArray<Double>([.nan, .infinity, 4, 0, 1e300])
        let h = try a.hypot(b).toRawArray()
        XCTAssertEqual(h[0], .infinity); XCTAssertEqual(h[1], .infinity)
        XCTAssertEqual(h[2], 5); XCTAssertEqual(h[3], 0)
        XCTAssertEqual(Self.ulps(h[4], Foundation.hypot(1e300, 1e300)), 0, accuracy: 2,
                       "hypot must not overflow on the way")

        let fa = try MetalArray<Float>([.infinity, .nan, 3, 0, 1e38])
        let fb = try MetalArray<Float>([.nan, .infinity, 4, 0, 1e38])
        let fh = try fa.hypot(fb).toRawArray()
        XCTAssertEqual(fh[0], .infinity); XCTAssertEqual(fh[1], .infinity)
        XCTAssertEqual(fh[2], 5); XCTAssertTrue(fh[4].isFinite, "float32 hypot must not overflow either")

        // Integer columns are refused rather than promoted, as `sqrt` and `ln` already are.
        XCTAssertThrowsError(try MetalArray<Int32>([1]).expm1())
        XCTAssertThrowsError(try MetalArray<Int32>([1]).log1p())
        XCTAssertThrowsError(try MetalArray<Int32>([1]).hypot(2))
        XCTAssertThrowsError(try MetalArray<Int32>([1]).logb(2))
    }

    // MARK: - Precision measured against Foundation over 10^6 inputs

    /// The numbers this prints are the ones quoted on `DoubleTranscendental` and in the C header.
    func testFloat64PrecisionAgainstFoundation() throws {
        try requireRealGPU()
        var rng = SeededRNG(0x2718_2818)
        let n = 1_000_000

        func measure(_ name: String, _ inputs: [Double], _ gpu: (MetalArray<Double>) throws -> MetalArray<Double>,
                     _ oracle: (Double) -> Double, bound: Double) throws {
            let got = try gpu(try MetalArray<Double>(inputs)).toRawArray()
            var worst = 0.0, worstAt = 0
            for i in 0..<n {
                let e = Self.ulps(got[i], oracle(inputs[i]))
                if e > worst { worst = e; worstAt = i }
            }
            print("PRECISION float64 \(name): worst \(worst) ulp over \(n) inputs (x = \(inputs[worstAt]))")
            XCTAssertLessThanOrEqual(worst, bound, "\(name) exceeded its documented ulp bound")
        }

        let wide = (0..<n).map { _ in Double.random(in: -700...700, using: &rng) }
        try measure("expm1", wide, { try $0.expm1() }, Foundation.expm1, bound: 2)

        let aboveMinusOne = (0..<n).map { _ in Double.random(in: -0.999999...1e6, using: &rng) }
        try measure("log1p", aboveMinusOne, { try $0.log1p() }, Foundation.log1p, bound: 2)

        let positive = (0..<n).map { _ in Double.random(in: 1e-30...1e30, using: &rng) }
        try measure("logb(2)", positive, { try $0.logb(2.0) }, { Foundation.log($0) / Foundation.log(2.0) }, bound: 4)
        try measure("hypot(x, 3)", positive, { try $0.hypot(3.0) }, { Foundation.hypot($0, 3.0) }, bound: 2)
    }

    // MARK: - Rounding: every mode, every size, floats and integers

    func testRoundModesFloat() throws {
        try requireRealGPU()
        var rng = SeededRNG(0x1618_0339)
        // Halfway cases first: only these tell the ten modes apart.
        let halves: [Double] = [-2.5, -1.5, -0.5, 0.0, -0.0, 0.5, 1.5, 2.5, 3.5, 123.456, -123.456,
                                1e17, .infinity, -.infinity, .nan]
        for n in Self.sizes {
            var vs: [Double?] = (0..<n).map { i in
                if i % 7 == 0 { return nil }
                if i < halves.count { return halves[i] }
                return Double.random(in: -1e6...1e6, using: &rng)
            }
            if n > 0 && n <= halves.count { vs = Array(halves.prefix(n)).map { Optional($0) } }
            let d = try MetalArray<Double>(vs)
            let f = try MetalArray<Float>(vs.map { $0.map(Float.init) })
            for mode in RoundMode.allCases {
                for nd in [0, 1, 2, -1, -2] {
                    let gd = try d.round(ndigits: nd, mode: mode).toRawArray()
                    for i in 0..<n {
                        guard let v = vs[i] else { continue }
                        let want = Self.roundDigits(v, nd, mode)
                        XCTAssertEqual(gd[i].bitPattern, want.bitPattern,
                                       "float64 round nd=\(nd) \(mode.arrowName) n=\(n) row \(i) value \(v): got \(gd[i]) want \(want)")
                    }
                    let gf = try f.round(ndigits: nd, mode: mode).toRawArray()
                    for i in 0..<n {
                        guard let v = vs[i].map(Float.init) else { continue }
                        let want = Self.roundDigits(v, nd, mode)
                        XCTAssertEqual(gf[i].bitPattern, want.bitPattern,
                                       "float32 round nd=\(nd) \(mode.arrowName) n=\(n) row \(i) value \(v)")
                    }
                }
            }
        }
    }

    func testRoundToMultipleAndBinaryFloat() throws {
        try requireRealGPU()
        var rng = SeededRNG(0x2653_5897)
        for n in Self.sizes {
            let vs = doubles(n: n, nulls: 0.2, range: -1000...1000, rng: &rng)
            let d = try MetalArray<Double>(vs)
            let f = try MetalArray<Float>(vs.map { $0.map(Float.init) })
            for mode in RoundMode.allCases {
                for m in [0.5, 1.0, 2.0, 0.1, 3.0] {
                    let g = try d.roundToMultiple(m, mode: mode).toRawArray()
                    for i in 0..<n {
                        guard let v = vs[i] else { continue }
                        let want = Self.roundToInteger(v / m, mode) * m
                        XCTAssertEqual(g[i].bitPattern, want.bitPattern,
                                       "float64 round_to_multiple \(m) \(mode.arrowName) row \(i) value \(v)")
                    }
                    let gf = try f.roundToMultiple(Float(m), mode: mode).toRawArray()
                    for i in 0..<n {
                        guard let v = vs[i].map(Float.init) else { continue }
                        let want = Self.roundToInteger(v / Float(m), mode) * Float(m)
                        XCTAssertEqual(gf[i].bitPattern, want.bitPattern,
                                       "float32 round_to_multiple \(m) \(mode.arrowName) row \(i)")
                    }
                }
            }
            // round_binary: one ndigits per row, null wherever either column is null.
            var nds: [Int32?] = []
            for i in 0..<n { nds.append(i % 11 == 3 ? nil : Int32((i % 7) - 3)) }
            let ndArray = try MetalArray<Int32>(nds)
            let rb = try d.roundBinary(ndArray, mode: .halfToEven)
            let got = rb.toArray()
            for i in 0..<n {
                guard let v = vs[i], let k = nds[i] else {
                    XCTAssertNil(got[i], "round_binary null at \(i)")
                    continue
                }
                XCTAssertEqual(got[i]?.bitPattern, Self.roundDigits(v, Int(k), .halfToEven).bitPattern,
                               "float64 round_binary row \(i) value \(v) ndigits \(k)")
            }
        }
        // A non-positive multiple is refused, as Arrow refuses it.
        XCTAssertThrowsError(try MetalArray<Double>([1]).roundToMultiple(0))
        XCTAssertThrowsError(try MetalArray<Double>([1]).roundToMultiple(-1))
        XCTAssertThrowsError(try MetalArray<Float>([1]).roundToMultiple(0))
        XCTAssertThrowsError(try MetalArray<Int32>([1]).roundToMultiple(0))
    }

    func testRoundIntegers() throws {
        try requireRealGPU()
        var rng = SeededRNG(0x9323_8462)

        func exercise<T: ArrowPrimitive & FixedWidthInteger>(_: T.Type, n: Int, rng: inout SeededRNG) throws {
            let vs: [T?] = (0..<n).map { _ in
                Double.random(in: 0..<1, using: &rng) < 0.2 ? nil : T(truncatingIfNeeded: rng.next())
            }
            let a = try MetalArray<T>(vs)
            let multiples: [T] = [1, 3, 5, 10, T.max / 2 > 0 ? T.max / 2 : 1]
            for mode in RoundMode.allCases {
                for m in multiples where m > 0 {
                    let got = try a.roundToMultiple(m, mode: mode).toArray()
                    let want: [T?] = vs.map { $0.map { Self.roundToMultiple($0, m, mode) } }
                    XCTAssertEqual(got, want, "\(T.arrowFormat) round_to_multiple \(m) \(mode.arrowName) n=\(n)")
                }
                for nd in [0, 2, -1, -2, -20] {
                    let got = try a.round(ndigits: nd, mode: mode).toArray()
                    let want: [T?] = vs.map { $0.map { Self.roundDigits($0, nd, mode) } }
                    XCTAssertEqual(got, want, "\(T.arrowFormat) round nd=\(nd) \(mode.arrowName) n=\(n)")
                }
            }
            // round_binary over the same values.
            var nds: [Int32?] = []
            for i in 0..<n { nds.append(i % 13 == 5 ? nil : Int32((i % 5) - 3)) }
            let got = try a.roundBinary(try MetalArray<Int32>(nds), mode: .halfToEven).toArray()
            for i in 0..<n {
                guard let v = vs[i], let k = nds[i] else { XCTAssertNil(got[i]); continue }
                XCTAssertEqual(got[i], Self.roundDigits(v, Int(k), .halfToEven),
                               "\(T.arrowFormat) round_binary row \(i)")
            }
        }

        for n in [0, 1, 33, 4097] {
            try exercise(Int8.self, n: n, rng: &rng)
            try exercise(UInt8.self, n: n, rng: &rng)
            try exercise(Int16.self, n: n, rng: &rng)
            try exercise(UInt16.self, n: n, rng: &rng)
            try exercise(Int32.self, n: n, rng: &rng)
            try exercise(UInt32.self, n: n, rng: &rng)
            try exercise(Int64.self, n: n, rng: &rng)
            try exercise(UInt64.self, n: n, rng: &rng)
        }
        // The long run, on the two widths where a wrong half-width digit would show up first.
        try exercise(Int64.self, n: 1_000_003, rng: &rng)
        try exercise(UInt64.self, n: 1_000_003, rng: &rng)
    }

    /// The values `pyarrow.compute` returns for the cases the Python differential test also covers, so a
    /// regression shows up here even without pyarrow installed.
    func testMatchesPyarrowFixtures() throws {
        try requireRealGPU()
        let v: [Double] = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
        let expected: [RoundMode: [Double]] = [
            .down: [-3, -2, -1, 0, 1, 2],
            .up: [-2, -1, -0.0, 1, 2, 3],
            .towardsZero: [-2, -1, -0.0, 0, 1, 2],
            .towardsInfinity: [-3, -2, -1, 1, 2, 3],
            .halfDown: [-3, -2, -1, 0, 1, 2],
            .halfUp: [-2, -1, -0.0, 1, 2, 3],
            .halfTowardsZero: [-2, -1, -0.0, 0, 1, 2],
            .halfTowardsInfinity: [-3, -2, -1, 1, 2, 3],
            .halfToEven: [-2, -2, -0.0, 0, 2, 2],
            .halfToOdd: [-3, -1, -1, 1, 1, 3],
        ]
        let a = try MetalArray<Double>(v)
        for (mode, want) in expected {
            let got = try a.round(ndigits: 0, mode: mode).toRawArray()
            for i in 0..<v.count {
                XCTAssertEqual(got[i].bitPattern, want[i].bitPattern,
                               "round \(mode.arrowName) of \(v[i]): got \(got[i]) want \(want[i])")
            }
        }
        // pyarrow: round(123.456, ndigits=2) is 123.46, not the nearer 123.45 — the scaling rounds first.
        XCTAssertEqual(try MetalArray<Double>([123.456]).round(ndigits: 2).toRawArray()[0], 123.46)
        XCTAssertEqual(try MetalArray<Double>([123.456]).round(ndigits: 2, mode: .down).toRawArray()[0], 123.45)
        XCTAssertEqual(try MetalArray<Double>([1.25]).roundToMultiple(0.1).toRawArray()[0], 1.2000000000000002)
        XCTAssertEqual(try MetalArray<Int32>([-7, -5, 5, 7, 125]).roundToMultiple(3).toArray(),
                       [-6, -6, 6, 6, 126])
        XCTAssertEqual(try MetalArray<Int32>([-7, -5, 5, 7, 125]).roundToMultiple(3, mode: .down).toArray(),
                       [-9, -6, 3, 6, 123])
        XCTAssertEqual(try MetalArray<Int32>([125, -125, 7]).round(ndigits: -1).toArray(), [120, -120, 10])
        XCTAssertEqual(try MetalArray<Double>([8]).logb(2).toRawArray()[0], 3.0)
        XCTAssertEqual(try MetalArray<Double>([3, 0]).hypot(try MetalArray<Double>([4, 0])).toRawArray(), [5, 0])
    }
}
