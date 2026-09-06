import XCTest
@testable import ArrowMetal

/// `sqrt`, `exp`, `ln`, `log2`, `log10` and `power` on `float64`, which since
/// `Kernels/DoublePower.swift` run entirely in software binary64 rather than narrowing the column to
/// `float` and widening the answer back.
///
/// The accuracy claim is measured, not asserted from a derivation: every function is compared with
/// Foundation over 10^6 random inputs drawn across its whole domain — subnormals, arguments near 1 for
/// the logarithms, exponents near the overflow and underflow edges for `exp`, negative bases with
/// integer exponents for `power` — and the test prints the ulp histogram it measured. Those printed
/// numbers are the ones quoted in `docs/DESIGN.md`, `docs/COVERAGE.md` and the Python function
/// registry, so a regression in either direction shows up as a diff there.
///
/// `sqrt` is held to a stricter standard than the rest: **bit-identical** to `Foundation.sqrt`, which
/// is IEEE-754's correctly rounded square root. `d_sqrt` extracts the root digit by digit in integers,
/// so there is no approximation left to be off by.
final class DoubleTranscendentalTests: XCTestCase {

    /// 0 and 1 are the degenerate cases, 33 straddles a bitmap word, 4097 crosses several
    /// threadgroups and 1_000_003 is large and not a multiple of anything.
    static let sizes = [0, 1, 33, 4097, 1_000_003]

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

    /// Distance in units of the last place. Two NaNs agree; a finite/non-finite mismatch never does.
    static func ulps(_ a: Double, _ b: Double) -> Double {
        if a == b { return 0 }
        if a.isNaN || b.isNaN { return (a.isNaN && b.isNaN) ? 0 : .infinity }
        if !a.isFinite || !b.isFinite { return .infinity }
        return (a - b).magnitude / Swift.max(a.magnitude, b.magnitude).ulp
    }

    // MARK: - Measurement

    /// Runs `gpu` over `inputs`, compares with `oracle` and reports the worst ulp seen plus the
    /// histogram of how many rows landed in each ulp bucket.
    @discardableResult
    func measure(_ name: String, _ inputs: [Double], bound: Double,
                 _ gpu: (MetalArray<Double>) throws -> MetalArray<Double>,
                 _ oracle: (Double) -> Double) throws -> Double {
        let got = try gpu(try MetalArray<Double>(inputs)).toRawArray()
        var worst = 0.0, worstAt = 0
        var buckets = [Int](repeating: 0, count: 5)     // 0, (0,0.5], (0.5,1], (1,2], >2
        for i in 0..<inputs.count {
            let e = Self.ulps(got[i], oracle(inputs[i]))
            if e > worst { worst = e; worstAt = i }
            let b = e == 0 ? 0 : (e <= 0.5 ? 1 : (e <= 1 ? 2 : (e <= 2 ? 3 : 4)))
            buckets[b] += 1
        }
        let pct = { (k: Int) in String(format: "%.2f%%", 100.0 * Double(buckets[k]) / Double(Swift.max(inputs.count, 1))) }
        print("""
              PRECISION float64 \(name): worst \(worst) ulp over \(inputs.count) inputs \
              (x = \(inputs[worstAt]), got \(got[worstAt]), want \(oracle(inputs[worstAt]))) \
              [exact \(pct(0)) | <=0.5 \(pct(1)) | <=1 \(pct(2)) | <=2 \(pct(3)) | >2 \(pct(4))]
              """)
        XCTAssertLessThanOrEqual(worst, bound, "\(name) exceeded its documented ulp bound")
        return worst
    }

    /// Same, for a two-column op: `oracle` sees both operands.
    @discardableResult
    func measure2(_ name: String, _ xs: [Double], _ ys: [Double], bound: Double,
                  _ oracle: (Double, Double) -> Double) throws -> Double {
        let a = try MetalArray<Double>(xs), b = try MetalArray<Double>(ys)
        let got = try a.power(b).toRawArray()
        var worst = 0.0, worstAt = 0
        for i in 0..<xs.count {
            let e = Self.ulps(got[i], oracle(xs[i], ys[i]))
            if e > worst { worst = e; worstAt = i }
        }
        print("PRECISION float64 \(name): worst \(worst) ulp over \(xs.count) pairs "
              + "(x = \(xs[worstAt]), y = \(ys[worstAt]), got \(got[worstAt]), want \(oracle(xs[worstAt], ys[worstAt])))")
        XCTAssertLessThanOrEqual(worst, bound, "\(name) exceeded its documented ulp bound")
        return worst
    }

    // MARK: - sqrt is correctly rounded

    /// Every random bit pattern, subnormals and the extremes included, must come back bit-identical to
    /// `Foundation.sqrt`. Not "within an ulp": the same 64 bits.
    func testSqrtIsCorrectlyRounded() throws {
        try requireRealGPU()
        var rng = SeededRNG(0x5EED_5417)
        var xs: [Double] = [0, -0.0, 1, 4, 2, 0.5, .infinity,
                            .leastNonzeroMagnitude, .leastNormalMagnitude, .greatestFiniteMagnitude,
                            1e-320, 1e-300, 1e300, 3, 1e308]
        // Uniform over the exponent range as well as over the significand, so subnormals, the top of
        // the range and everything between are all represented.
        for _ in 0..<1_000_000 {
            let bits = rng.next() & 0x7FFF_FFFF_FFFF_FFFF
            let x = Double(bitPattern: bits)
            if x.isNaN || x.isInfinite { continue }
            xs.append(x)
        }
        let got = try MetalArray<Double>(xs).sqrt().toRawArray()
        var mismatches = 0, firstBad = -1
        for i in 0..<xs.count where got[i].bitPattern != Foundation.sqrt(xs[i]).bitPattern {
            mismatches += 1
            if firstBad < 0 { firstBad = i }
        }
        print("PRECISION float64 sqrt: \(mismatches) of \(xs.count) inputs differ from Foundation (bit-exact target)")
        XCTAssertEqual(mismatches, 0,
                       firstBad < 0 ? "" : "first at x = \(xs[firstBad]): got \(got[firstBad]), want \(Foundation.sqrt(xs[firstBad]))")

        // A negative operand is NaN (unchecked Arrow), -0 keeps its sign, +inf stays +inf.
        let special = try MetalArray<Double>([-1, -1e-320, -.infinity, .nan, -0.0]).sqrt().toRawArray()
        XCTAssertTrue(special[0].isNaN); XCTAssertTrue(special[1].isNaN)
        XCTAssertTrue(special[2].isNaN); XCTAssertTrue(special[3].isNaN)
        XCTAssertEqual(special[4].bitPattern, (-0.0 as Double).bitPattern, "sqrt(-0) is -0")
    }

    // MARK: - exp, ln, log2, log10 over 10^6 inputs each

    func testExpPrecision() throws {
        try requireRealGPU()
        var rng = SeededRNG(0x1234_5678)
        let n = 1_000_000
        // The whole finite domain, then the two edges where the answer stops being normal.
        let wide = (0..<n).map { _ in Double.random(in: -745.2...709.78, using: &rng) }
        try measure("exp", wide, bound: 2, { try $0.exp() }, Foundation.exp)
        let underflow = (0..<200_000).map { _ in Double.random(in: -745.2 ... -708.0, using: &rng) }
        try measure("exp (subnormal results)", underflow, bound: 2, { try $0.exp() }, Foundation.exp)
        let overflow = (0..<200_000).map { _ in Double.random(in: 700.0...709.782712893, using: &rng) }
        try measure("exp (near overflow)", overflow, bound: 2, { try $0.exp() }, Foundation.exp)
        let tiny = (0..<200_000).map { _ in Double.random(in: -1e-8...1e-8, using: &rng) }
        try measure("exp (tiny)", tiny, bound: 2, { try $0.exp() }, Foundation.exp)
    }

    /// The logarithms share one reduction, so they share one input set: log-uniform over the whole
    /// positive range, plus a pass concentrated near 1 (where the answer cancels) and one over the
    /// subnormals (which the reduction has to scale up first).
    func testLogarithmPrecision() throws {
        try requireRealGPU()
        var rng = SeededRNG(0x0FED_CBA9)
        let n = 1_000_000
        let wide = (0..<n).map { _ in Foundation.exp(Double.random(in: -744.0...709.0, using: &rng)) }
        let nearOne = (0..<200_000).map { _ in 1.0 + Double.random(in: -0.4...0.5, using: &rng) }
        let subnormal = (0..<200_000).map { _ in
            Double(bitPattern: rng.next() & 0x000F_FFFF_FFFF_FFFF) + Double.leastNonzeroMagnitude
        }
        for (label, xs) in [("", wide), (" near 1", nearOne), (" subnormal", subnormal)] {
            try measure("ln\(label)", xs, bound: 2, { try $0.ln() }, Foundation.log)
            try measure("log2\(label)", xs, bound: 2, { try $0.log2() }, Foundation.log2)
            try measure("log10\(label)", xs, bound: 2, { try $0.log10() }, Foundation.log10)
        }
        // Exact powers of two are exact in log2, and the exponent alone decides them.
        let powers = (-1074...1023).map { Double(sign: .plus, exponent: $0, significand: 1) }
        let got = try MetalArray<Double>(powers).log2().toRawArray()
        for (i, e) in (-1074...1023).enumerated() {
            XCTAssertEqual(got[i], Double(e), "log2(2^\(e)) must be exactly \(e)")
        }
    }

    // MARK: - power

    func testPowerPrecision() throws {
        try requireRealGPU()
        var rng = SeededRNG(0x2468_ACE0)
        let n = 1_000_000
        // Bases log-uniform over the whole positive range against exponents small enough that the
        // result usually stays finite; pairs that overflow or underflow agree trivially and are
        // filtered out by the ulp comparison itself (inf == inf is zero ulp).
        var xs = [Double](), ys = [Double]()
        for _ in 0..<n {
            let x = Foundation.exp(Double.random(in: -700...700, using: &rng))
            let y = Double.random(in: -40...40, using: &rng)
            xs.append(x); ys.append(y)
        }
        try measure2("power (array ^ array)", xs, ys, bound: 2, Foundation.pow)

        // Negative bases with integer exponents: the sign rule and the odd/even test.
        var nx = [Double](), ny = [Double]()
        for _ in 0..<500_000 {
            nx.append(Double.random(in: -100 ... -0.01, using: &rng))
            ny.append(Double(Int.random(in: -40...40, using: &rng)))
        }
        try measure2("power (negative base, integer exponent)", nx, ny, bound: 2, Foundation.pow)

        // A scalar exponent takes a different kernel; check the common ones exactly against Foundation.
        let bases = (0..<200_000).map { _ in Foundation.exp(Double.random(in: -300...300, using: &rng)) }
        for y in [2.0, 0.5, -1.0, 3.0, 1.0 / 3.0, -2.5, 0.0] {
            try measure("power (x ^ \(y), scalar)", bases, bound: 2, { try $0.power(y) }, { Foundation.pow($0, y) })
        }
    }

    /// The values libm pins down exactly, taken straight from C99's `pow` table.
    func testPowerEdgeTable() throws {
        try requireRealGPU()
        let inf = Double.infinity, nan = Double.nan
        let cases: [(Double, Double)] = [
            (0, 0), (0, 1), (0, -1), (-0.0, 3), (-0.0, 2), (-0.0, -3), (-0.0, -2),
            (1, nan), (1, inf), (1, -inf), (nan, 0), (inf, 0), (-inf, 0), (nan, nan),
            (-1, inf), (-1, -inf), (2, inf), (0.5, inf), (2, -inf), (0.5, -inf),
            (inf, 2), (inf, -2), (-inf, 3), (-inf, 2), (-inf, -3), (-inf, -2),
            (-2, 0.5), (-2, 3), (-2, 2), (-2, -3), (2, 1024), (2, -1075), (2, 1023),
            (5e-324, 0.5), (1e308, 2), (1e-308, 2), (10, 308), (10, -308),
            (2, 53), (3, 5), (-1, 3), (-1, 2), (-1, 0.5), (nan, 2), (2, nan),
        ]
        let a = try MetalArray<Double>(cases.map(\.0)), b = try MetalArray<Double>(cases.map(\.1))
        let got = try a.power(b).toRawArray()
        for (i, c) in cases.enumerated() {
            let want = Foundation.pow(c.0, c.1)
            if want.isNaN {
                XCTAssertTrue(got[i].isNaN, "pow(\(c.0), \(c.1)) should be NaN, got \(got[i])")
            } else {
                XCTAssertEqual(got[i].bitPattern, want.bitPattern,
                               "pow(\(c.0), \(c.1)): got \(got[i]), want \(want)")
            }
        }
    }

    /// `exp`, the logarithms and `sqrt` at the values whose answers are pinned rather than approximated.
    func testUnaryEdgeTable() throws {
        try requireRealGPU()
        let inf = Double.infinity
        let xs: [Double] = [0, -0.0, 1, 2, 10, .infinity, -.infinity, .nan, -1, -1e300,
                            .leastNonzeroMagnitude, .leastNormalMagnitude, .greatestFiniteMagnitude,
                            1e-300, 1e300, 709.782712893384, 710, -745.2, -746, 1024, -1024]
        let d = try MetalArray<Double>(xs)

        let e = try d.exp().toRawArray()
        XCTAssertEqual(e[0], 1); XCTAssertEqual(e[1], 1)
        XCTAssertEqual(e[5], inf, "exp(+inf) is +inf")
        XCTAssertEqual(e[6], 0, "exp(-inf) is +0")
        XCTAssertTrue(e[7].isNaN)
        XCTAssertEqual(e[16], inf, "exp overflows above 709.7827...")
        XCTAssertEqual(e[18], 0, "exp underflows to zero below -745.1332...")
        XCTAssertEqual(e[19], inf); XCTAssertEqual(e[20], 0)
        for i in 0..<xs.count where Foundation.exp(xs[i]).isFinite {
            XCTAssertLessThanOrEqual(Self.ulps(e[i], Foundation.exp(xs[i])), 2, "exp(\(xs[i]))")
        }

        let hosts: [(String, (MetalArray<Double>) throws -> MetalArray<Double>, (Double) -> Double)] =
            [("ln", { try $0.ln() }, { Foundation.log($0) }),
             ("log2", { try $0.log2() }, { Foundation.log2($0) }),
             ("log10", { try $0.log10() }, { Foundation.log10($0) })]
        for (name, gpu, host) in hosts {
            let g = try gpu(d).toRawArray()
            XCTAssertEqual(g[0], -inf, "\(name)(0) is -inf")
            XCTAssertEqual(g[1], -inf, "\(name)(-0) is -inf")
            XCTAssertEqual(g[2], 0, "\(name)(1) is 0")
            XCTAssertEqual(g[5], inf, "\(name)(+inf) is +inf")
            XCTAssertTrue(g[6].isNaN, "\(name)(-inf) is NaN")
            XCTAssertTrue(g[7].isNaN); XCTAssertTrue(g[8].isNaN, "\(name)(-1) is NaN")
            XCTAssertTrue(g[9].isNaN)
            for i in 0..<xs.count where host(xs[i]).isFinite {
                XCTAssertLessThanOrEqual(Self.ulps(g[i], host(xs[i])), 2, "\(name)(\(xs[i]))")
            }
        }
        XCTAssertEqual(try d.log2().toRawArray()[3], 1, "log2(2) is exactly 1")
        XCTAssertEqual(try d.log10().toRawArray()[4], 1, "log10(10) is exactly 1")
    }

    // MARK: - Every size, with nulls

    /// The kernels are one thread per element with no cross-lane work, but the wrappers around them are
    /// not: an empty column must not dispatch, a null must survive untouched, and the tail of a grid
    /// that is not a multiple of the threadgroup must still be written.
    func testEverySizeWithNulls() throws {
        try requireRealGPU()
        var rng = SeededRNG(0x0BAD_C0DE)
        for n in Self.sizes {
            let pos: [Double?] = (0..<n).map { _ in
                Double.random(in: 0..<1, using: &rng) < 0.2 ? nil : Foundation.exp(Double.random(in: -300...300, using: &rng))
            }
            let expo: [Double?] = (0..<n).map { _ in
                Double.random(in: 0..<1, using: &rng) < 0.2 ? nil : Double.random(in: -30...30, using: &rng)
            }
            let p = try MetalArray<Double>(pos), q = try MetalArray<Double>(expo)

            func check(_ what: String, _ got: [Double?], _ want: [Double?], ulp: Double) {
                XCTAssertEqual(got.count, want.count, "\(what) n=\(n): length")
                var worst = 0.0, at = -1
                for i in 0..<Swift.min(got.count, want.count) {
                    guard let g = got[i], let w = want[i] else {
                        XCTAssertTrue(got[i] == nil && want[i] == nil, "\(what) n=\(n): null mismatch at \(i)")
                        continue
                    }
                    let e = Self.ulps(g, w)
                    if e > worst { worst = e; at = i }
                }
                XCTAssertLessThanOrEqual(worst, ulp, "\(what) n=\(n): worst \(worst) ulp at row \(at)")
            }
            check("sqrt", try p.sqrt().toArray(), pos.map { $0.map(Foundation.sqrt) }, ulp: 0)
            check("ln", try p.ln().toArray(), pos.map { $0.map(Foundation.log) }, ulp: 2)
            check("log2", try p.log2().toArray(), pos.map { $0.map(Foundation.log2) }, ulp: 2)
            check("log10", try p.log10().toArray(), pos.map { $0.map(Foundation.log10) }, ulp: 2)
            check("exp", try q.exp().toArray(), expo.map { $0.map(Foundation.exp) }, ulp: 2)
            check("power scalar", try p.power(2.5).toArray(), pos.map { $0.map { Foundation.pow($0, 2.5) } }, ulp: 2)

            // Two columns: the result is null wherever either side is.
            let got = try p.power(q).toArray()
            var want = [Double?]()
            for i in 0..<n {
                if let x = pos[i], let y = expo[i] { want.append(Foundation.pow(x, y)) } else { want.append(nil) }
            }
            check("power array", got, want, ulp: 2)

            // Validity is shared zero-copy on the unary ops, so the null count must come through intact.
            XCTAssertEqual(try p.sqrt().nullCount, p.nullCount, "sqrt nulls n=\(n)")
            XCTAssertEqual(try p.ln().nullCount, p.nullCount, "ln nulls n=\(n)")
            XCTAssertEqual(try q.exp().nullCount, q.nullCount, "exp nulls n=\(n)")
        }
    }

    // MARK: - The checked twins

    /// A checked result that does not raise is the unchecked kernel's own output, bit for bit, and the
    /// boundary it raises at is exactly Arrow's: zero and every negative for the logarithms, every
    /// negative for `sqrt`, and nothing at all for `power` on a float column.
    func testCheckedTwinsAtTheExactBoundaries() throws {
        try requireRealGPU()
        let ok = try MetalArray<Double>([1, 2, 0.5, .infinity, .nan, 1e-320, .greatestFiniteMagnitude])
        // NaN, +inf and a subnormal are all in the domain; none of them raises.
        XCTAssertEqual(try ok.sqrtChecked().toRawArray().map(\.bitPattern),
                       try ok.sqrt().toRawArray().map(\.bitPattern), "sqrt_checked is bit-identical")
        XCTAssertEqual(try ok.lnChecked().toRawArray().map(\.bitPattern),
                       try ok.ln().toRawArray().map(\.bitPattern), "ln_checked is bit-identical")
        XCTAssertEqual(try ok.log2Checked().toRawArray().map(\.bitPattern),
                       try ok.log2().toRawArray().map(\.bitPattern), "log2_checked is bit-identical")
        XCTAssertEqual(try ok.log10Checked().toRawArray().map(\.bitPattern),
                       try ok.log10().toRawArray().map(\.bitPattern), "log10_checked is bit-identical")

        // -0.0 is in the domain of sqrt (it gives -0.0) but not of the logarithms (it gives -inf).
        XCTAssertNoThrow(try MetalArray<Double>([-0.0, 0.0]).sqrtChecked())
        for (name, call) in [("ln", { (a: MetalArray<Double>) in try a.lnChecked() }),
                             ("log2", { try $0.log2Checked() }), ("log10", { try $0.log10Checked() })] {
            // Zero raises "logarithm of zero"...
            XCTAssertThrowsError(try call(try MetalArray<Double>([1, 2, 0.0, 3])), "\(name)(0)") { e in
                XCTAssertTrue("\(e)".contains("logarithm of zero"), "\(name): \(e)")
                XCTAssertTrue("\(e)".contains("index 2"), "\(name) must name the first offending row: \(e)")
            }
            XCTAssertThrowsError(try call(try MetalArray<Double>([1, -0.0]))) { e in
                XCTAssertTrue("\(e)".contains("logarithm of zero"), "\(name)(-0): \(e)")
            }
            // ...and anything below it "logarithm of negative number", the smallest subnormal included.
            XCTAssertThrowsError(try call(try MetalArray<Double>([1, 2, -Double.leastNonzeroMagnitude]))) { e in
                XCTAssertTrue("\(e)".contains("logarithm of negative number"), "\(name)(-5e-324): \(e)")
            }
            XCTAssertNoThrow(try call(try MetalArray<Double>([Double.leastNonzeroMagnitude, .infinity, .nan])),
                             "\(name): +5e-324, +inf and NaN are all in the domain")
        }
        // sqrt raises only strictly below zero, and the smallest subnormal is enough.
        XCTAssertThrowsError(try MetalArray<Double>([1, -Double.leastNonzeroMagnitude]).sqrtChecked()) { e in
            XCTAssertTrue("\(e)".contains("square root of negative number"), "\(e)")
        }
        XCTAssertNoThrow(try MetalArray<Double>([.nan, .infinity, -0.0]).sqrtChecked())
        // A null is never checked, on either side.
        XCTAssertNoThrow(try MetalArray<Double>([1, nil, 2] as [Double?]).lnChecked())
        XCTAssertNoThrow(try MetalArray<Double>([1, nil, 2] as [Double?]).sqrtChecked())

        // Arrow's float `power_checked` never raises: an overflow to infinity is an ordinary result.
        let base = try MetalArray<Double>([1e300, -2, 0, .infinity])
        let expo = try MetalArray<Double>([2, 3, -1, 0.5])
        XCTAssertEqual(try base.powerChecked(expo).toRawArray().map(\.bitPattern),
                       try base.power(expo).toRawArray().map(\.bitPattern), "power_checked is bit-identical")
        XCTAssertNoThrow(try base.powerChecked(2.0))
    }

    /// `modulo` is the one float64 binary op still missing, and it must say so rather than answer badly.
    func testModuloStillRefusesFloat64() throws {
        XCTAssertThrowsError(try MetalArray<Double>([1, 2]).modulo(2.0)) { e in
            XCTAssertTrue("\(e)".contains("not implemented for float64"), "\(e)")
        }
    }
}
