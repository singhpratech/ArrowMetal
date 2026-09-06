import XCTest
@testable import ArrowMetal

/// The trigonometric, inverse-trigonometric and hyperbolic kernels, checked element for element
/// against Foundation on the host.
///
/// `float32` runs the MSL library functions and is compared with a small ulp budget. `float64` runs
/// the software binary64 implementation in `TrigSource.swift`; the accuracy test below measures the
/// **maximum ulp error over a million random arguments per function** and asserts a budget, and the
/// measured numbers are the ones quoted in `docs/COVERAGE.md`.
final class TrigTests: XCTestCase {
    /// 0 and 1 are the degenerate cases, 33 straddles a bitmap word, 4097 crosses several
    /// threadgroups and 1_000_003 is large and not a multiple of anything.
    static let sizes = [0, 1, 33, 4097, 1_000_003]

    // MARK: - Oracles

    static func hostF(_ op: TrigOp, _ x: Float) -> Float {
        switch op {
        case .sin: return Foundation.sin(x)
        case .cos: return Foundation.cos(x)
        case .tan: return Foundation.tan(x)
        case .asin: return Foundation.asin(x)
        case .acos: return Foundation.acos(x)
        case .atan: return Foundation.atan(x)
        case .sinh: return Foundation.sinh(x)
        case .cosh: return Foundation.cosh(x)
        case .tanh: return Foundation.tanh(x)
        case .asinh: return Foundation.asinh(x)
        case .acosh: return Foundation.acosh(x)
        case .atanh: return Foundation.atanh(x)
        }
    }

    static func hostD(_ op: TrigOp, _ x: Double) -> Double {
        switch op {
        case .sin: return Foundation.sin(x)
        case .cos: return Foundation.cos(x)
        case .tan: return Foundation.tan(x)
        case .asin: return Foundation.asin(x)
        case .acos: return Foundation.acos(x)
        case .atan: return Foundation.atan(x)
        case .sinh: return Foundation.sinh(x)
        case .cosh: return Foundation.cosh(x)
        case .tanh: return Foundation.tanh(x)
        case .asinh: return Foundation.asinh(x)
        case .acosh: return Foundation.acosh(x)
        case .atanh: return Foundation.atanh(x)
        }
    }

    /// A domain each function is finite on, so a random draw exercises the interesting branches
    /// rather than the NaN one. `acosh` needs `x >= 1`, `asin`/`acos`/`atanh` need `|x| <= 1`.
    static func domain(_ op: TrigOp, _ u: Double) -> Double {
        switch op {
        case .sin, .cos, .tan: return (u - 0.5) * 200.0            // radians, several hundred periods
        case .asin, .acos: return u * 2.0 - 1.0                    // [-1, 1]
        case .atanh: return (u * 2.0 - 1.0) * 0.999_999            // (-1, 1)
        case .atan, .asinh: return (u - 0.5) * 2_000.0
        case .sinh, .cosh, .tanh: return (u - 0.5) * 40.0
        case .acosh: return 1.0 + u * 100.0
        }
    }

    /// Distance in ulps between two doubles, both assumed finite and of the same sign class.
    static func ulps(_ a: Double, _ b: Double) -> Double {
        if a == b { return 0 }
        if a.isNaN && b.isNaN { return 0 }
        if a.isNaN != b.isNaN || a.isInfinite || b.isInfinite { return .infinity }
        let ka = Int64(bitPattern: a.bitPattern), kb = Int64(bitPattern: b.bitPattern)
        func key(_ k: Int64) -> Int64 { k < 0 ? Int64.min &- k : k }   // order-preserving
        return Double((key(ka) &- key(kb)).magnitude)
    }

    static func ulpsF(_ a: Float, _ b: Float) -> Double {
        if a == b { return 0 }
        if a.isNaN && b.isNaN { return 0 }
        if a.isNaN != b.isNaN || a.isInfinite || b.isInfinite { return .infinity }
        let ka = Int32(bitPattern: a.bitPattern), kb = Int32(bitPattern: b.bitPattern)
        func key(_ k: Int32) -> Int32 { k < 0 ? Int32.min &- k : k }
        return Double((Int64(key(ka)) - Int64(key(kb))).magnitude)
    }

    /// Deterministic uniform stream, so a failure is reproducible.
    struct Rng {
        var s: UInt64
        mutating func next() -> UInt64 {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17
            return s
        }
        mutating func unit() -> Double { Double(next() >> 11) * (1.0 / 9_007_199_254_740_992.0) }
    }

    static func nullAt(_ i: Int) -> Bool { i % 7 == 3 }

    // MARK: - float32

    func testFloat32AllSizes() throws {
        try requireRealGPU()
        for n in Self.sizes {
            var rng = Rng(s: 0x2545F4914F6CDD1D)
            for op in TrigOp.allCases {
                let vals: [Float?] = (0..<n).map { i in
                    Self.nullAt(i) ? nil : Float(Self.domain(op, rng.unit()))
                }
                let a = try MetalArray<Float>(vals)
                let got = try a.trig(op).toArray()
                XCTAssertEqual(got.count, n, "\(op) n=\(n)")
                for i in 0..<n {
                    guard let v = vals[i] else { XCTAssertNil(got[i], "\(op) n=\(n) i=\(i)"); continue }
                    let want = Self.hostF(op, v)
                    let g = try XCTUnwrap(got[i], "\(op) n=\(n) i=\(i)")
                    XCTAssertLessThanOrEqual(Self.ulpsF(g, want), 8,
                                             "\(op)(\(v)) = \(g), want \(want) (n=\(n) i=\(i))")
                }
            }
        }
    }

    /// The float32 counterpart of the accuracy measurement below: a million random arguments per
    /// function, the maximum ulp error against Foundation reported and asserted.
    func testFloat32UlpBudget() throws {
        try requireRealGPU()
        let n = 1_000_003
        var report: [String] = []
        for op in TrigOp.allCases {
            var rng = Rng(s: 0xDEADBEEFCAFEBABE)
            let vals: [Float] = (0..<n).map { _ in Float(Self.domain(op, rng.unit())) }
            let got = try MetalArray<Float>(vals).trig(op).toRawArray()
            var worst = 0.0, worstAt = Float(0)
            for i in 0..<n {
                let u = Self.ulpsF(got[i], Self.hostF(op, vals[i]))
                if u > worst { worst = u; worstAt = vals[i] }
            }
            report.append("\(op.rawValue): max \(worst) ulp (worst at \(worstAt))")
            XCTAssertLessThanOrEqual(worst, 4, "f32 \(op) max ulp \(worst) at \(worstAt)")
        }
        print("ArrowMetal float32 trig accuracy over \(n) random arguments:\n  " + report.joined(separator: "\n  "))
    }

    func testFloat32Atan2() throws {
        try requireRealGPU()
        var rng = Rng(s: 0x9E3779B97F4A7C15)
        let n = 4097
        let ys: [Float?] = (0..<n).map { i in Self.nullAt(i) ? nil : Float((rng.unit() - 0.5) * 20) }
        let xs: [Float?] = (0..<n).map { i in i % 11 == 5 ? nil : Float((rng.unit() - 0.5) * 20) }
        let y = try MetalArray<Float>(ys), x = try MetalArray<Float>(xs)
        let got = try y.atan2(x).toArray()
        for i in 0..<n {
            guard let a = ys[i], let b = xs[i] else { XCTAssertNil(got[i], "i=\(i)"); continue }
            let want = Foundation.atan2(a, b)
            XCTAssertLessThanOrEqual(Self.ulpsF(try XCTUnwrap(got[i]), want), 4, "atan2(\(a),\(b)) i=\(i)")
        }
        // Scalar form.
        let got2 = try y.atan2(Float(2.5)).toArray()
        for i in 0..<n {
            guard let a = ys[i] else { XCTAssertNil(got2[i]); continue }
            XCTAssertLessThanOrEqual(Self.ulpsF(try XCTUnwrap(got2[i]), Foundation.atan2(a, 2.5)), 4, "i=\(i)")
        }
    }

    // MARK: - float64

    func testFloat64AllSizes() throws {
        try requireRealGPU()
        for n in Self.sizes where n <= 4097 {
            var rng = Rng(s: 0x123456789ABCDEF)
            for op in TrigOp.allCases {
                let vals: [Double?] = (0..<n).map { i in
                    Self.nullAt(i) ? nil : Self.domain(op, rng.unit())
                }
                let a = try MetalArray<Double>(vals)
                let got = try a.trig(op).toArray()
                XCTAssertEqual(got.count, n)
                for i in 0..<n {
                    guard let v = vals[i] else { XCTAssertNil(got[i]); continue }
                    let want = Self.hostD(op, v)
                    XCTAssertLessThanOrEqual(Self.ulps(try XCTUnwrap(got[i]), want), 4,
                                             "\(op)(\(v)) n=\(n) i=\(i)")
                }
            }
        }
    }

    /// The headline accuracy measurement: one million random arguments per function, the maximum
    /// ulp error against Foundation reported and asserted. These are the numbers in COVERAGE.md.
    func testFloat64UlpBudget() throws {
        try requireRealGPU()
        let n = 1_000_003
        var report: [String] = []
        for op in TrigOp.allCases {
            var rng = Rng(s: 0xDEADBEEFCAFEBABE)
            let vals: [Double] = (0..<n).map { _ in Self.domain(op, rng.unit()) }
            let a = try MetalArray<Double>(vals)
            let got = try a.trig(op).toRawArray()
            var worst = 0.0
            var worstAt = 0.0
            for i in 0..<n {
                let want = Self.hostD(op, vals[i])
                let u = Self.ulps(got[i], want)
                if u > worst { worst = u; worstAt = vals[i] }
            }
            report.append("\(op.rawValue): max \(worst) ulp (worst at \(worstAt))")
            XCTAssertLessThanOrEqual(worst, 6, "\(op) max ulp \(worst) at \(worstAt)")
        }
        print("ArrowMetal float64 trig accuracy over \(n) random arguments:\n  " + report.joined(separator: "\n  "))
    }

    /// `sin`/`cos`/`tan` over the range the Cody-Waite reduction is exact on (|x| <= 2^45 * pi/2).
    func testFloat64LargeArgumentReduction() throws {
        try requireRealGPU()
        var rng = Rng(s: 0x5DEECE66D)
        let vals: [Double] = (0..<100_000).map { _ in (rng.unit() - 0.5) * 2.0 * 5.0e13 }
        let a = try MetalArray<Double>(vals)
        for op in [TrigOp.sin, .cos, .tan] {
            let got = try a.trig(op).toRawArray()
            var worst = 0.0
            for i in 0..<vals.count {
                worst = Swift.max(worst, Self.ulps(got[i], Self.hostD(op, vals[i])))
            }
            // Measured: sin 4, cos 5, tan 7 ulp. tan amplifies the reduced argument's error near
            // its poles, and at |x| ~ 5e13 the input's own ulp is already 0.0078 radians.
            XCTAssertLessThanOrEqual(worst, 8, "\(op) on |x| up to 5e13: max \(worst) ulp")
        }
    }

    func testFloat64Atan2() throws {
        try requireRealGPU()
        var rng = Rng(s: 0xA5A5A5A5A5A5A5)
        let n = 100_003
        let ys: [Double] = (0..<n).map { _ in (rng.unit() - 0.5) * 2000 }
        let xs: [Double] = (0..<n).map { _ in (rng.unit() - 0.5) * 2000 }
        let y = try MetalArray<Double>(ys), x = try MetalArray<Double>(xs)
        let got = try y.atan2(x).toRawArray()
        var worst = 0.0
        for i in 0..<n { worst = Swift.max(worst, Self.ulps(got[i], Foundation.atan2(ys[i], xs[i]))) }
        XCTAssertLessThanOrEqual(worst, 4, "atan2 max \(worst) ulp")
    }

    // MARK: - Special values

    /// NaN, ±inf, ±0 and the domain edges, on both float types, against Foundation.
    func testSpecialValues() throws {
        try requireRealGPU()
        let specialD: [Double] = [.nan, .infinity, -.infinity, 0.0, -0.0, 1.0, -1.0, 2.0, -2.0,
                                  0.5, -0.5, 1.0000000000000002, 0.9999999999999999,
                                  .pi, -.pi, .pi / 2, 20.0, -20.0, 30.0, -30.0, 710.0, -710.0,
                                  1e-300, -1e-300, 5e-324, .greatestFiniteMagnitude]
        let a = try MetalArray<Double>(specialD)
        for op in TrigOp.allCases {
            let got = try a.trig(op).toRawArray()
            for (i, v) in specialD.enumerated() {
                let want = Self.hostD(op, v)
                if want.isNaN {
                    XCTAssertTrue(got[i].isNaN, "\(op)(\(v)) = \(got[i]), want NaN")
                    continue
                }
                // Arguments above the reduction limit are documented as NaN for sin/cos/tan.
                if [TrigOp.sin, .cos, .tan].contains(op) && v.magnitude >= 0x1p62 {
                    XCTAssertTrue(got[i].isNaN, "\(op)(\(v)) beyond the reduction range should be NaN")
                    continue
                }
                XCTAssertLessThanOrEqual(Self.ulps(got[i], want), 4, "\(op)(\(v)) = \(got[i]), want \(want)")
                if want == 0 {
                    XCTAssertEqual(got[i].sign, want.sign, "\(op)(\(v)) signed zero")
                }
            }
        }

        let specialF = specialD.map { Float($0) }
        let af = try MetalArray<Float>(specialF)
        for op in TrigOp.allCases {
            let got = try af.trig(op).toRawArray()
            for (i, v) in specialF.enumerated() {
                let want = Self.hostF(op, v)
                if want.isNaN { XCTAssertTrue(got[i].isNaN, "f32 \(op)(\(v))"); continue }
                XCTAssertLessThanOrEqual(Self.ulpsF(got[i], want), 8, "f32 \(op)(\(v)) = \(got[i]), want \(want)")
            }
        }
    }

    /// The full C99 `atan2` special-value table, on both float types.
    func testAtan2SpecialValues() throws {
        try requireRealGPU()
        let vals: [Double] = [.nan, .infinity, -.infinity, 0.0, -0.0, 1.0, -1.0, 3.0, -3.0]
        var ys: [Double] = [], xs: [Double] = []
        for y in vals { for x in vals { ys.append(y); xs.append(x) } }
        let got = try MetalArray<Double>(ys).atan2(try MetalArray<Double>(xs)).toRawArray()
        for i in 0..<ys.count {
            let want = Foundation.atan2(ys[i], xs[i])
            if want.isNaN { XCTAssertTrue(got[i].isNaN, "atan2(\(ys[i]),\(xs[i]))"); continue }
            XCTAssertLessThanOrEqual(Self.ulps(got[i], want), 2, "atan2(\(ys[i]),\(xs[i])) = \(got[i]), want \(want)")
            if want == 0 { XCTAssertEqual(got[i].sign, want.sign, "atan2(\(ys[i]),\(xs[i])) signed zero") }
        }
        let gotF = try MetalArray<Float>(ys.map { Float($0) })
            .atan2(try MetalArray<Float>(xs.map { Float($0) })).toRawArray()
        for i in 0..<ys.count {
            let want = Foundation.atan2(Float(ys[i]), Float(xs[i]))
            if want.isNaN { XCTAssertTrue(gotF[i].isNaN); continue }
            XCTAssertLessThanOrEqual(Self.ulpsF(gotF[i], want), 2, "f32 atan2(\(ys[i]),\(xs[i]))")
        }
    }

    // MARK: - Checked forms

    /// A value inside each checked op's domain, and one outside it (Arrow's rule).
    static func checkedDomain(_ op: TrigCheckedOp, _ u: Double) -> Double {
        switch op {
        case .sin, .cos, .tan: return (u - 0.5) * 200.0
        case .asin, .acos: return u * 2.0 - 1.0
        case .acosh: return 1.0 + u * 100.0
        case .atanh: return (u * 2.0 - 1.0) * 0.999_999
        }
    }

    static func outOfDomain(_ op: TrigCheckedOp) -> [Double] {
        switch op {
        case .sin, .cos, .tan: return [.infinity, -.infinity]
        case .asin, .acos: return [1.5, -1.5, .infinity, -.infinity]
        case .acosh: return [0.5, -3.0, -.infinity]
        case .atanh: return [1.0, -1.0, 2.0, .infinity]
        }
    }

    /// On an in-domain column the checked op is **bit-identical** to the unchecked one, at every
    /// size, with nulls, and on both float types. NaN never raises.
    func testCheckedPassingPath() throws {
        try requireRealGPU()
        for n in Self.sizes where n <= 4097 {
            for op in TrigCheckedOp.allCases {
                var rng = Rng(s: 0xC0FFEE)
                var vals: [Double?] = (0..<n).map { i in
                    Self.nullAt(i) ? nil : Self.checkedDomain(op, rng.unit())
                }
                // NaN is in the domain for every one of them.
                if n > 5 { vals[5] = Double.nan }
                let a = try MetalArray<Double>(vals)
                let want = try a.trig(op.unchecked).toArray()
                let got = try a.trigChecked(op).toArray()
                XCTAssertEqual(got.count, n)
                for i in 0..<n {
                    if let w = want[i] {
                        let g = try XCTUnwrap(got[i], "\(op.arrowName) n=\(n) i=\(i)")
                        XCTAssertEqual(g.bitPattern, w.bitPattern, "\(op.arrowName) n=\(n) i=\(i)")
                    } else {
                        XCTAssertNil(got[i], "\(op.arrowName) n=\(n) i=\(i)")
                    }
                }

                let af = try MetalArray<Float>(vals.map { $0.map { Float($0) } })
                let wantF = try af.trig(op.unchecked).toArray()
                let gotF = try af.trigChecked(op).toArray()
                for i in 0..<n {
                    if let w = wantF[i] {
                        XCTAssertEqual(try XCTUnwrap(gotF[i]).bitPattern, w.bitPattern, "f32 \(op.arrowName) n=\(n) i=\(i)")
                    } else {
                        XCTAssertNil(gotF[i])
                    }
                }
            }
        }
    }

    /// One out-of-domain value anywhere in the column raises, and the message names its index.
    func testCheckedRaisingPath() throws {
        try requireRealGPU()
        for op in TrigCheckedOp.allCases {
            for bad in Self.outOfDomain(op) {
                for n in [1, 33, 4097] {
                    for badAt in [0, n / 2, n - 1] {
                        var vals: [Double] = (0..<n).map { _ in Self.checkedDomain(op, 0.5) }
                        vals[badAt] = bad
                        let a = try MetalArray<Double>(vals)
                        XCTAssertThrowsError(try a.trigChecked(op), "\(op.arrowName) should raise on \(bad)") { err in
                            let msg = "\(err)"
                            XCTAssertTrue(msg.contains(op.arrowName), msg)
                            XCTAssertTrue(msg.contains("index \(badAt)"), msg)
                        }
                        let af = try MetalArray<Float>(vals.map { Float($0) })
                        XCTAssertThrowsError(try af.trigChecked(op), "f32 \(op.arrowName) on \(bad)")
                    }
                }
            }
        }
    }

    /// A null row is never inspected, so an out-of-domain value sitting in a null slot does not raise.
    func testCheckedIgnoresNullRows() throws {
        try requireRealGPU()
        for op in TrigCheckedOp.allCases {
            let bad = Self.outOfDomain(op)[0]
            let n = 33
            // Build a column whose null slots hold the offending value.
            let raw: [Double] = (0..<n).map { i in i % 3 == 1 ? bad : Self.checkedDomain(op, 0.5) }
            let a = try MetalArray<Double>((0..<n).map { i in i % 3 == 1 ? nil : raw[i] })
            // Write the offending values into the null slots' value bytes.
            let p = a.mutableValuePointer
            for i in 0..<n where i % 3 == 1 { p[i] = bad }
            XCTAssertNoThrow(try a.trigChecked(op), "\(op.arrowName) must ignore null rows")
        }
    }

    func testCheckedRejectsIntegerColumns() throws {
        let a = try MetalArray<Int32>([1, 2, 3])
        for op in TrigCheckedOp.allCases { XCTAssertThrowsError(try a.trigChecked(op)) }
    }

    /// Integer columns are rejected rather than silently promoted, like `sqrt` and `exp`.
    func testIntegerColumnThrows() throws {
        let a = try MetalArray<Int32>([1, 2, 3])
        XCTAssertThrowsError(try a.sin())
        XCTAssertThrowsError(try a.atan2(a))
    }

    /// A slice (word-aligned, so zero-copy, and unaligned, so materialised) gives the same answers.
    func testSlices() throws {
        try requireRealGPU()
        var rng = Rng(s: 7)
        let n = 4097
        let vals: [Double?] = (0..<n).map { i in Self.nullAt(i) ? nil : (rng.unit() - 0.5) * 8 }
        let a = try MetalArray<Double>(vals)
        for off in [32, 5] {
            let s = try a.slice(offset: off, length: n - off)
            let got = try s.trig(.sin).toArray()
            for i in 0..<(n - off) {
                guard let v = vals[i + off] else { XCTAssertNil(got[i]); continue }
                XCTAssertLessThanOrEqual(Self.ulps(try XCTUnwrap(got[i]), Foundation.sin(v)), 2, "off=\(off) i=\(i)")
            }
        }
    }
}
