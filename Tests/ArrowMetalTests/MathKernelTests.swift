import XCTest
@testable import ArrowMetal

/// Bit-wise, element-wise math and cumulative kernels against CPU oracles.
///
/// The oracles below restate the *defined* semantics (documented on `BitwiseSource`, `RoundingSource`
/// and `CumulativeSource`) rather than calling Swift's operators blindly, because the interesting cases
/// — over-wide shifts, `abs(Int8.min)`, `x % 0`, a negative integer exponent — are exactly the ones
/// where Swift traps and Arrow raises.
final class MathKernelTests: XCTestCase {

    // MARK: - Deterministic RNG (SplitMix64), so a failure reproduces

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

    static let sizes = [0, 1, 31, 32, 33, 255, 256, 257, 1000, 8193]
    static let cumulativeSizes = [0, 1, 255, 256, 257, 70_001, 1_000_003]

    // MARK: - Oracles: shifts with ArrowMetal's defined out-of-range behaviour

    /// The shift count in `[0, bitWidth)`, or nil when it is out of range (negative, or too wide).
    static func shiftCount<T: FixedWidthInteger>(_ s: T) -> Int? {
        let w = T.bitWidth
        if T.isSigned {
            let k = Int64(truncatingIfNeeded: s)
            return (k < 0 || k >= Int64(w)) ? nil : Int(k)
        }
        let k = UInt64(truncatingIfNeeded: s)
        return k >= UInt64(w) ? nil : Int(k)
    }
    static func refShl<T: FixedWidthInteger>(_ a: T, _ s: T) -> T {
        guard let k = shiftCount(s) else { return 0 }
        return T(truncatingIfNeeded: UInt64(truncatingIfNeeded: a) << UInt64(k))
    }
    static func refShr<T: FixedWidthInteger>(_ a: T, _ s: T) -> T {
        guard let k = shiftCount(s) else { return a < 0 ? T(truncatingIfNeeded: -1) : 0 }
        if T.isSigned { return T(truncatingIfNeeded: Int64(truncatingIfNeeded: a) >> Int64(k)) }
        return T(truncatingIfNeeded: UInt64(truncatingIfNeeded: a) >> UInt64(k))
    }
    static func refBitwise<T: FixedWidthInteger>(_ op: BitwiseOp, _ a: T, _ b: T) -> T {
        switch op {
        case .and: return a & b
        case .or: return a | b
        case .xor: return a ^ b
        case .shl: return refShl(a, b)
        case .shr: return refShr(a, b)
        }
    }

    // MARK: - Oracles: integer math

    static func refSign<T: FixedWidthInteger>(_ x: T) -> T { x > 0 ? 1 : (x < 0 ? T(truncatingIfNeeded: -1) : 0) }
    static func refAbs<T: FixedWidthInteger>(_ x: T) -> T { x < 0 ? 0 &- x : x }
    static func refPow<T: FixedWidthInteger>(_ a: T, _ b: T) -> T {
        if b < 0 { return 0 }                       // defined here; Arrow raises
        var r: T = 1, base = a, e = UInt64(truncatingIfNeeded: b)
        while e != 0 {
            if e & 1 == 1 { r = r &* base }
            base = base &* base
            e >>= 1
        }
        return r
    }
    static func refMod<T: FixedWidthInteger>(_ a: T, _ b: T) -> T {
        if b == 0 { return 0 }                      // defined here, as for `divide`
        if T.isSigned && b == T(truncatingIfNeeded: -1) { return 0 }
        return a % b
    }

    // MARK: - Input builders

    func ints<T: ArrowPrimitive & FixedWidthInteger>(_: T.Type, n: Int, nulls: Double, rng: inout SeededRNG) -> [T?] {
        (0..<n).map { _ in
            let null = Double.random(in: 0..<1, using: &rng) < nulls
            let v = T(truncatingIfNeeded: rng.next())
            return null ? nil : v
        }
    }
    /// Shift amounts: mostly in range, deliberately out of range about a third of the time.
    func shiftAmounts<T: ArrowPrimitive & FixedWidthInteger>(_: T.Type, n: Int, rng: inout SeededRNG) -> [T?] {
        (0..<n).map { _ in T(truncatingIfNeeded: Int64.random(in: -3...Int64(T.bitWidth + 3), using: &rng)) }
    }
    func doubles(n: Int, nulls: Double, range: ClosedRange<Double>, rng: inout SeededRNG) -> [Double?] {
        (0..<n).map { _ in
            Double.random(in: 0..<1, using: &rng) < nulls ? nil : Double.random(in: range, using: &rng)
        }
    }
    func float32s(n: Int, nulls: Double, range: ClosedRange<Double>, rng: inout SeededRNG) -> [Float?] {
        doubles(n: n, nulls: nulls, range: range, rng: &rng).map { $0.map { Float($0) } }
    }

    // MARK: - Bit-wise and shifts

    func exerciseBitwise<T: ArrowPrimitive & FixedWidthInteger>(_: T.Type, n: Int, rng: inout SeededRNG) throws {
        let label = "\(T.arrowFormat) n=\(n)"
        let av = ints(T.self, n: n, nulls: 0.25, rng: &rng)
        let bv = ints(T.self, n: n, nulls: 0.25, rng: &rng)
        let sv = shiftAmounts(T.self, n: n, rng: &rng)
        let a = try MetalArray<T>(av), b = try MetalArray<T>(bv), s = try MetalArray<T>(sv)
        let scalars: [T] = [0, 1, 3,
                            T(truncatingIfNeeded: T.bitWidth - 1), T(truncatingIfNeeded: T.bitWidth),
                            T(truncatingIfNeeded: T.bitWidth + 5), T(truncatingIfNeeded: -1), T.max, T.min]

        for op in BitwiseOp.allCases {
            let isShift = (op == .shl || op == .shr)
            let other = isShift ? s : b
            let otherVals = isShift ? sv : bv
            let got = try a.bitwise(op, other).toArray()
            let want: [T?] = (0..<n).map { i in
                guard let x = av[i], let y = otherVals[i] else { return nil }
                return Self.refBitwise(op, x, y)
            }
            XCTAssertEqual(got, want, "\(label) \(op) array")

            for sc in scalars {
                let g = try a.bitwise(op, sc).toArray()
                let w: [T?] = av.map { $0.map { Self.refBitwise(op, $0, sc) } }
                XCTAssertEqual(g, w, "\(label) \(op) scalar \(sc)")
            }
        }
        XCTAssertEqual(try a.bitwiseNot().toArray(), av.map { $0.map { ~$0 } }, "\(label) not")
    }

    func testBitwiseAllIntegerTypes() throws {
        try requireRealGPU()
        var rng = SeededRNG(0xA11CE)
        for n in Self.sizes {
            try exerciseBitwise(Int8.self, n: n, rng: &rng)
            try exerciseBitwise(UInt8.self, n: n, rng: &rng)
            try exerciseBitwise(Int16.self, n: n, rng: &rng)
            try exerciseBitwise(UInt16.self, n: n, rng: &rng)
            try exerciseBitwise(Int32.self, n: n, rng: &rng)
            try exerciseBitwise(UInt32.self, n: n, rng: &rng)
            try exerciseBitwise(Int64.self, n: n, rng: &rng)
            try exerciseBitwise(UInt64.self, n: n, rng: &rng)
        }
    }

    /// The documented out-of-range shift semantics, spelled out value by value.
    func testShiftDefinedSemantics() throws {
        try requireRealGPU()
        let signed = try MetalArray<Int32>([-8, -1, 0, 1, 1 << 30, Int32.min, Int32.max])
        XCTAssertEqual(try signed.shiftLeft(32).toArray(), [0, 0, 0, 0, 0, 0, 0], "shl by the width is 0")
        XCTAssertEqual(try signed.shiftLeft(-1).toArray(), [0, 0, 0, 0, 0, 0, 0], "a negative count is 0")
        XCTAssertEqual(try signed.shiftRight(32).toArray(), [-1, -1, 0, 0, 0, -1, 0], "shr by the width sign-fills")
        XCTAssertEqual(try signed.shiftRight(1).toArray(), [-4, -1, 0, 0, 1 << 29, Int32.min / 2, Int32.max / 2],
                       "shr is arithmetic on a signed column")
        XCTAssertEqual(try signed.shiftLeft(1).toArray(), [-16, -2, 0, 2, Int32.min, 0, -2], "shl drops the top bits")

        let unsigned = try MetalArray<UInt32>([1, 0x8000_0000, UInt32.max])
        XCTAssertEqual(try unsigned.shiftRight(1).toArray(), [0, 0x4000_0000, 0x7FFF_FFFF], "shr is logical on unsigned")
        XCTAssertEqual(try unsigned.shiftRight(32).toArray(), [0, 0, 0], "unsigned shr by the width is 0")
        XCTAssertEqual(try unsigned.shiftRight(UInt32.max).toArray(), [0, 0, 0], "a huge unsigned count is out of range")
    }

    // MARK: - Unary math, integers

    func exerciseIntegerUnary<T: ArrowPrimitive & FixedWidthInteger>(_: T.Type, n: Int, rng: inout SeededRNG) throws {
        let label = "\(T.arrowFormat) n=\(n)"
        let av = ints(T.self, n: n, nulls: 0.25, rng: &rng)
        let a = try MetalArray<T>(av)
        XCTAssertEqual(try a.negate().toArray(), av.map { $0.map { 0 &- $0 } }, "\(label) negate")
        XCTAssertEqual(try a.abs().toArray(), av.map { $0.map { Self.refAbs($0) } }, "\(label) abs")
        XCTAssertEqual(try a.sign().toArray(), av.map { $0.map { Self.refSign($0) } }, "\(label) sign")
        for op in [UnaryMathOp.floor, .ceil, .round, .trunc] {
            XCTAssertEqual(try a.unaryMath(op).toArray(), av, "\(label) \(op) is the identity on integers")
        }
    }

    func testUnaryMathIntegers() throws {
        try requireRealGPU()
        var rng = SeededRNG(0xBEEF)
        for n in Self.sizes {
            try exerciseIntegerUnary(Int8.self, n: n, rng: &rng)
            try exerciseIntegerUnary(UInt8.self, n: n, rng: &rng)
            try exerciseIntegerUnary(Int16.self, n: n, rng: &rng)
            try exerciseIntegerUnary(UInt16.self, n: n, rng: &rng)
            try exerciseIntegerUnary(Int32.self, n: n, rng: &rng)
            try exerciseIntegerUnary(UInt32.self, n: n, rng: &rng)
            try exerciseIntegerUnary(Int64.self, n: n, rng: &rng)
            try exerciseIntegerUnary(UInt64.self, n: n, rng: &rng)
        }
        // The wrapping corners, explicitly.
        let edges = try MetalArray<Int8>([Int8.min, -1, 0, 1, Int8.max])
        XCTAssertEqual(try edges.abs().toArray(), [Int8.min, 1, 0, 1, Int8.max], "abs(Int8.min) wraps")
        XCTAssertEqual(try edges.negate().toArray(), [Int8.min, 1, 0, -1, -Int8.max], "negate(Int8.min) wraps")
        XCTAssertEqual(try edges.sign().toArray(), [-1, -1, 0, 1, 1], "signed sign")
        let uedges = try MetalArray<UInt8>([0, 1, 200, UInt8.max])
        XCTAssertEqual(try uedges.negate().toArray(), [0, 255, 56, 1], "unsigned negate is modular")
        XCTAssertEqual(try uedges.sign().toArray(), [0, 1, 1, 1], "unsigned sign is 0 or 1")
        XCTAssertEqual(try uedges.abs().toArray(), [0, 1, 200, 255], "unsigned abs is the identity")
    }

    // MARK: - Unary math, floating point

    func testUnaryMathFloat32() throws {
        try requireRealGPU()
        var rng = SeededRNG(0xF10A7)
        for n in Self.sizes {
            let vals = float32s(n: n, nulls: 0.25, range: -500...500, rng: &rng)
            let a = try MetalArray<Float>(vals)
            XCTAssertEqual(try a.negate().toArray(), vals.map { $0.map { -$0 } }, "negate n=\(n)")
            XCTAssertEqual(try a.abs().toArray(), vals.map { $0.map { Swift.abs($0) } }, "abs n=\(n)")
            XCTAssertEqual(try a.sign().toArray(), vals.map { $0.map { $0 > 0 ? 1 : ($0 < 0 ? -1 : $0) } }, "sign n=\(n)")
            XCTAssertEqual(try a.floor().toArray(), vals.map { $0.map { $0.rounded(.down) } }, "floor n=\(n)")
            XCTAssertEqual(try a.ceil().toArray(), vals.map { $0.map { $0.rounded(.up) } }, "ceil n=\(n)")
            XCTAssertEqual(try a.trunc().toArray(), vals.map { $0.map { $0.rounded(.towardZero) } }, "trunc n=\(n)")
            XCTAssertEqual(try a.round().toArray(), vals.map { $0.map { $0.rounded(.toNearestOrAwayFromZero) } }, "round n=\(n)")

            // Transcendentals on a positive domain, against Foundation (evaluated in double, then narrowed).
            let pos = float32s(n: n, nulls: 0.25, range: 0.001...50, rng: &rng)
            let p = try MetalArray<Float>(pos)
            assertClose(try p.sqrt().toArray(), pos.map { $0.map { Float(Foundation.sqrt(Double($0))) } }, rel: 1e-6, "sqrt n=\(n)")
            assertClose(try p.ln().toArray(), pos.map { $0.map { Float(Foundation.log(Double($0))) } }, rel: 1e-5, "ln n=\(n)")
            assertClose(try p.log10().toArray(), pos.map { $0.map { Float(Foundation.log10(Double($0))) } }, rel: 1e-5, "log10 n=\(n)")
            assertClose(try p.log2().toArray(), pos.map { $0.map { Float(Foundation.log2(Double($0))) } }, rel: 1e-5, "log2 n=\(n)")
            let small = float32s(n: n, nulls: 0.25, range: -20...20, rng: &rng)
            let sm = try MetalArray<Float>(small)
            assertClose(try sm.exp().toArray(), small.map { $0.map { Float(Foundation.exp(Double($0))) } }, rel: 1e-5, "exp n=\(n)")
        }
    }

    func testUnaryMathFloat64() throws {
        try requireRealGPU()
        var rng = SeededRNG(0xD0B1E)
        for n in Self.sizes {
            let vals = doubles(n: n, nulls: 0.25, range: -1e6...1e6, rng: &rng)
            let a = try MetalArray<Double>(vals)
            // Exact: everything below is bit-pattern work on the GPU.
            XCTAssertEqual(try a.negate().toArray(), vals.map { $0.map { -$0 } }, "negate n=\(n)")
            XCTAssertEqual(try a.abs().toArray(), vals.map { $0.map { Swift.abs($0) } }, "abs n=\(n)")
            XCTAssertEqual(try a.sign().toArray(), vals.map { $0.map { $0 > 0 ? 1 : ($0 < 0 ? -1 : $0) } }, "sign n=\(n)")
            XCTAssertEqual(try a.floor().toArray(), vals.map { $0.map { $0.rounded(.down) } }, "floor n=\(n)")
            XCTAssertEqual(try a.ceil().toArray(), vals.map { $0.map { $0.rounded(.up) } }, "ceil n=\(n)")
            XCTAssertEqual(try a.trunc().toArray(), vals.map { $0.map { $0.rounded(.towardZero) } }, "trunc n=\(n)")
            XCTAssertEqual(try a.round().toArray(), vals.map { $0.map { $0.rounded(.toNearestOrAwayFromZero) } }, "round n=\(n)")

            // float precision, widened: about seven significant digits, as documented.
            let pos = doubles(n: n, nulls: 0.25, range: 0.001...50, rng: &rng)
            let p = try MetalArray<Double>(pos)
            assertClose(try p.sqrt().toArray(), pos.map { $0.map { Foundation.sqrt($0) } }, rel: 1e-5, "f64 sqrt n=\(n)")
            assertClose(try p.ln().toArray(), pos.map { $0.map { Foundation.log($0) } }, rel: 1e-5, "f64 ln n=\(n)")
            assertClose(try p.log10().toArray(), pos.map { $0.map { Foundation.log10($0) } }, rel: 1e-5, "f64 log10 n=\(n)")
            assertClose(try p.log2().toArray(), pos.map { $0.map { Foundation.log2($0) } }, rel: 1e-5, "f64 log2 n=\(n)")
            let small = doubles(n: n, nulls: 0.25, range: -20...20, rng: &rng)
            let sm = try MetalArray<Double>(small)
            assertClose(try sm.exp().toArray(), small.map { $0.map { Foundation.exp($0) } }, rel: 1e-5, "f64 exp n=\(n)")
        }
    }

    /// Rounding corners on `float64`, where the kernel works on the raw bit pattern.
    func testFloat64RoundingCorners() throws {
        try requireRealGPU()
        let vals: [Double] = [0.0, -0.0, 0.5, -0.5, 0.49999999999999994, -0.49999999999999994,
                              1.5, -1.5, 2.5, -2.5, 1.0, -1.0, 0.1, -0.1, -0.4, 0.4,
                              4503599627370495.5, 4503599627370496.0, 1e300, -1e300,
                              5e-324, -5e-324, .infinity, -.infinity]
        let a = try MetalArray<Double>(vals)
        XCTAssertEqual(try a.floor().toRawArray(), vals.map { $0.rounded(.down) }, "floor corners")
        XCTAssertEqual(try a.ceil().toRawArray(), vals.map { $0.rounded(.up) }, "ceil corners")
        XCTAssertEqual(try a.trunc().toRawArray(), vals.map { $0.rounded(.towardZero) }, "trunc corners")
        XCTAssertEqual(try a.round().toRawArray(), vals.map { $0.rounded(.toNearestOrAwayFromZero) }, "round corners")
        // Signed zeros survive: `==` cannot see the difference, so compare bit patterns.
        for (op, mode) in [(UnaryMathOp.trunc, FloatingPointRoundingRule.towardZero), (.ceil, .up),
                           (.round, .toNearestOrAwayFromZero), (.floor, .down)] {
            let got = try a.unaryMath(op).toRawArray()
            for (i, v) in vals.enumerated() where !v.isInfinite {
                XCTAssertEqual(got[i].bitPattern, v.rounded(mode).bitPattern, "\(op) bit pattern at \(i) (\(v))")
            }
        }
        // NaN in, NaN out.
        let nan = try MetalArray<Double>([Double.nan, -Double.nan])
        for op in [UnaryMathOp.floor, .ceil, .round, .trunc, .abs, .negate, .sign] {
            XCTAssertTrue(try nan.unaryMath(op).toRawArray().allSatisfy { $0.isNaN }, "\(op) propagates NaN")
        }
        // sign keeps both zeros and both infinities.
        let s = try MetalArray<Double>([0.0, -0.0, .infinity, -.infinity, 3.0, -3.0]).sign().toRawArray()
        XCTAssertEqual(s.map { $0.bitPattern }, [0.0, -0.0, 1.0, -1.0, 1.0, -1.0].map { $0.bitPattern }, "sign")
    }

    // MARK: - Binary math

    func testPowerAndModuloIntegers() throws {
        try requireRealGPU()
        var rng = SeededRNG(0x9E7)
        for n in [0, 1, 33, 257, 1000] {
            let av = ints(Int32.self, n: n, nulls: 0.25, rng: &rng)
            let ev: [Int32?] = (0..<n).map { _ in Int32.random(in: -2...9, using: &rng) }
            let a = try MetalArray<Int32>(av), e = try MetalArray<Int32>(ev)
            XCTAssertEqual(try a.power(e).toArray(), (0..<n).map { i -> Int32? in
                guard let x = av[i], let y = ev[i] else { return nil }
                return Self.refPow(x, y)
            }, "int power array n=\(n)")
            for k: Int32 in [0, 1, 2, 3, 7, -1] {
                XCTAssertEqual(try a.power(k).toArray(), av.map { $0.map { Self.refPow($0, k) } }, "int power \(k) n=\(n)")
            }
            // Divisors include 0 and -1 on purpose.
            let dv: [Int32?] = (0..<n).map { _ in Int32.random(in: -5...5, using: &rng) }
            let d = try MetalArray<Int32>(dv)
            XCTAssertEqual(try a.modulo(d).toArray(), (0..<n).map { i -> Int32? in
                guard let x = av[i], let y = dv[i] else { return nil }
                return Self.refMod(x, y)
            }, "int modulo array n=\(n)")
            for k: Int32 in [1, 3, -7, 0, -1] {
                XCTAssertEqual(try a.modulo(k).toArray(), av.map { $0.map { Self.refMod($0, k) } }, "int modulo \(k) n=\(n)")
            }
        }
        // Every integer width through the same corners.
        var wrng = SeededRNG(0x1234)
        for n in [0, 1, 257] {
            try exerciseIntegerBinary(Int8.self, n: n, rng: &wrng)
            try exerciseIntegerBinary(UInt8.self, n: n, rng: &wrng)
            try exerciseIntegerBinary(Int16.self, n: n, rng: &wrng)
            try exerciseIntegerBinary(UInt16.self, n: n, rng: &wrng)
            try exerciseIntegerBinary(UInt32.self, n: n, rng: &wrng)
            try exerciseIntegerBinary(Int64.self, n: n, rng: &wrng)
            try exerciseIntegerBinary(UInt64.self, n: n, rng: &wrng)
        }
        // Unsigned wrap-around in `power`, and the documented corners.
        let u = try MetalArray<UInt8>([2, 3, 16, 255])
        XCTAssertEqual(try u.power(3).toArray(), [8, 27, 0, 255 &* 255 &* 255], "uint8 power wraps")
        let z = try MetalArray<Int32>([5, -5, 0])
        XCTAssertEqual(try z.power(0).toArray(), [1, 1, 1], "x^0 is 1, including 0^0")
        XCTAssertEqual(try z.power(-3).toArray(), [0, 0, 0], "a negative exponent is defined as 0")
        XCTAssertEqual(try z.modulo(0).toArray(), [0, 0, 0], "x % 0 is defined as 0")
        XCTAssertEqual(try MetalArray<Int32>([Int32.min]).modulo(-1).toArray(), [0], "Int32.min % -1 is 0, not a trap")
        XCTAssertEqual(try MetalArray<Int32>([-7, 7]).modulo(3).toArray(), [-1, 1], "the sign follows the dividend")
    }

    func exerciseIntegerBinary<T: ArrowPrimitive & FixedWidthInteger>(_: T.Type, n: Int, rng: inout SeededRNG) throws {
        let label = "\(T.arrowFormat) n=\(n)"
        let av = ints(T.self, n: n, nulls: 0.25, rng: &rng)
        let ev: [T?] = (0..<n).map { _ in T(truncatingIfNeeded: Int64.random(in: -2...6, using: &rng)) }
        let dv: [T?] = (0..<n).map { _ in T(truncatingIfNeeded: Int64.random(in: -4...4, using: &rng)) }
        let a = try MetalArray<T>(av), e = try MetalArray<T>(ev), d = try MetalArray<T>(dv)
        XCTAssertEqual(try a.power(e).toArray(), (0..<n).map { i -> T? in
            guard let x = av[i], let y = ev[i] else { return nil }
            return Self.refPow(x, y)
        }, "\(label) power")
        XCTAssertEqual(try a.modulo(d).toArray(), (0..<n).map { i -> T? in
            guard let x = av[i], let y = dv[i] else { return nil }
            return Self.refMod(x, y)
        }, "\(label) modulo")
    }

    func testPowerAndModuloFloat32() throws {
        try requireRealGPU()
        var rng = SeededRNG(0x5A17)
        for n in [0, 1, 33, 257, 1000] {
            let av = float32s(n: n, nulls: 0.25, range: 0.1...20, rng: &rng)
            let bv = float32s(n: n, nulls: 0.25, range: -3...3, rng: &rng)
            let a = try MetalArray<Float>(av), b = try MetalArray<Float>(bv)
            assertClose(try a.power(b).toArray(), (0..<n).map { i -> Float? in
                guard let x = av[i], let y = bv[i] else { return nil }
                return Float(Foundation.pow(Double(x), Double(y)))
            }, rel: 1e-5, "float power n=\(n)")
            let dv = float32s(n: n, nulls: 0.25, range: 0.5...5, rng: &rng)
            let d = try MetalArray<Float>(dv)
            assertClose(try a.modulo(d).toArray(), (0..<n).map { i -> Float? in
                guard let x = av[i], let y = dv[i] else { return nil }
                return x.truncatingRemainder(dividingBy: y)
            }, rel: 1e-6, "float modulo n=\(n)")
        }
    }

    func testMinMaxElementWiseSkipsNulls() throws {
        try requireRealGPU()
        var rng = SeededRNG(0x11AA)
        for n in Self.sizes {
            let av = ints(Int64.self, n: n, nulls: 0.35, rng: &rng)
            let bv = ints(Int64.self, n: n, nulls: 0.35, rng: &rng)
            let a = try MetalArray<Int64>(av), b = try MetalArray<Int64>(bv)
            func want(_ pick: (Int64, Int64) -> Int64) -> [Int64?] {
                (0..<n).map { i in
                    switch (av[i], bv[i]) {
                    case (let x?, let y?): return pick(x, y)
                    case (let x?, nil): return x
                    case (nil, let y?): return y
                    default: return nil
                    }
                }
            }
            XCTAssertEqual(try a.minElementWise(b).toArray(), want { Swift.min($0, $1) }, "min_element_wise n=\(n)")
            XCTAssertEqual(try a.maxElementWise(b).toArray(), want { Swift.max($0, $1) }, "max_element_wise n=\(n)")
        }
        // No validity on either side: the fast path must still be right.
        let x = try MetalArray<Float>([1, 5, -3, 2])
        let y = try MetalArray<Float>([4, 2, -8, 2])
        XCTAssertEqual(try x.minElementWise(y).toArray(), [1, 2, -8, 2], "float min_element_wise")
        XCTAssertEqual(try x.maxElementWise(y).toArray(), [4, 5, -3, 2], "float max_element_wise")
        // NaN loses, as it does in the reductions.
        let n1 = try MetalArray<Float>([Float.nan, 3, Float.nan])
        let n2 = try MetalArray<Float>([7, Float.nan, Float.nan])
        let mins = try n1.minElementWise(n2).toRawArray()
        XCTAssertEqual(mins[0], 7, "NaN loses on the left")
        XCTAssertEqual(mins[1], 3, "NaN loses on the right")
        XCTAssertTrue(mins[2].isNaN, "two NaNs stay NaN")
        // float64 min/max is exact bit-pattern work.
        let d1 = try MetalArray<Double>([1.5, nil, -2.25, nil])
        let d2 = try MetalArray<Double>([0.5, 9.0, nil, nil])
        XCTAssertEqual(try d1.minElementWise(d2).toArray(), [0.5, 9.0, -2.25, nil], "float64 min_element_wise")
        XCTAssertEqual(try d1.maxElementWise(d2).toArray(), [1.5, 9.0, -2.25, nil], "float64 max_element_wise")
    }

    /// -0.0 and 0.0 are equal, so which of the two comes back is a tie-break: `fmin` keeps the
    /// negative zero and `fmax` the positive one, whichever side it is on, which also makes the pair
    /// commutative. Subnormal operands must survive the comparison on `float32`, where the arithmetic
    /// `<` flushes them to zero.
    func testMinMaxElementWiseZeroTiesAndSubnormals() throws {
        try requireRealGPU()
        let tiny = Float.leastNonzeroMagnitude                       // 1.4e-45, subnormal
        let a = try MetalArray<Float>([0.0, -0.0, -0.0, tiny])
        let b = try MetalArray<Float>([-0.0, 0.0, -0.0, -0.0])
        XCTAssertEqual(try a.minElementWise(b).toRawArray().map { $0.bitPattern },
                       [Float](repeating: -0.0, count: 4).map { $0.bitPattern }, "float32 min keeps -0.0")
        XCTAssertEqual(try a.maxElementWise(b).toRawArray().map { $0.bitPattern },
                       [0.0, 0.0, -0.0, tiny].map { $0.bitPattern }, "float32 max keeps 0.0, and the subnormal wins")
        let ta = Double.leastNonzeroMagnitude
        let c = try MetalArray<Double>([0.0, -0.0, -0.0, ta])
        let d = try MetalArray<Double>([-0.0, 0.0, -0.0, -0.0])
        XCTAssertEqual(try c.minElementWise(d).toRawArray().map { $0.bitPattern },
                       [Double](repeating: -0.0, count: 4).map { $0.bitPattern }, "float64 min keeps -0.0")
        XCTAssertEqual(try c.maxElementWise(d).toRawArray().map { $0.bitPattern },
                       [0.0, 0.0, -0.0, ta].map { $0.bitPattern }, "float64 max keeps 0.0")
    }

    /// The `float32` kernels run in a flush-to-zero math mode, so `sign`, `floor`, `ceil` and `trunc`
    /// decide the subnormal and signed-zero cases from the bit pattern instead of from `<` and the
    /// library rounding functions -- the same answers the exact `float64` kernels give.
    func testFloat32SignAndRoundingCorners() throws {
        try requireRealGPU()
        let tiny = Float.leastNonzeroMagnitude                        // 1.4e-45
        let vals: [Float] = [0.0, -0.0, tiny, -tiny, Float.leastNormalMagnitude, -0.4, 0.4, -1.5,
                             2.5, -2.5, 1.0, -1.0, .infinity, -.infinity]
        let a = try MetalArray<Float>(vals)
        XCTAssertEqual(try a.sign().toRawArray().map { $0.bitPattern },
                       vals.map { ($0 == 0 ? $0 : ($0 < 0 ? -1 : 1) as Float).bitPattern },
                       "sign: ±0 pass through, every other magnitude is ±1")
        for (op, mode) in [(UnaryMathOp.floor, FloatingPointRoundingRule.down), (.ceil, .up),
                           (.trunc, .towardZero), (.round, .toNearestOrAwayFromZero)] {
            let got = try a.unaryMath(op).toRawArray()
            for (i, v) in vals.enumerated() where !v.isInfinite {
                XCTAssertEqual(got[i].bitPattern, v.rounded(mode).bitPattern,
                               "\(op) bit pattern at \(i) (\(v))")
            }
        }
        let nan = try MetalArray<Float>([Float.nan])
        for op in [UnaryMathOp.floor, .ceil, .round, .trunc, .sign] {
            XCTAssertTrue(try nan.unaryMath(op).toRawArray()[0].isNaN, "\(op) propagates NaN")
        }
    }

    // MARK: - Cumulative

    func cumulativeOracle<T>(_ vals: [T?], identity: T, _ combine: (T, T) -> T) -> [T?] {
        var acc = identity
        return vals.map { v in
            guard let v else { return nil }
            acc = combine(acc, v)
            return acc
        }
    }

    func exerciseCumulativeInts<T: ArrowPrimitive & FixedWidthInteger>(_: T.Type, n: Int, rng: inout SeededRNG) throws {
        let label = "\(T.arrowFormat) n=\(n)"
        let vals = ints(T.self, n: n, nulls: 0.25, rng: &rng)
        let a = try MetalArray<T>(vals)
        XCTAssertEqual(try a.cumulativeSum().toArray(), cumulativeOracle(vals, identity: 0) { $0 &+ $1 },
                       "\(label) cumulative_sum")
        XCTAssertEqual(try a.cumulativeMin().toArray(), cumulativeOracle(vals, identity: T.max) { Swift.min($0, $1) },
                       "\(label) cumulative_min")
        XCTAssertEqual(try a.cumulativeMax().toArray(), cumulativeOracle(vals, identity: T.min) { Swift.max($0, $1) },
                       "\(label) cumulative_max")
    }

    func testCumulativeIntegers() throws {
        try requireRealGPU()
        var rng = SeededRNG(0xC0FFEE)
        for n in Self.cumulativeSizes {
            try exerciseCumulativeInts(Int32.self, n: n, rng: &rng)
            try exerciseCumulativeInts(UInt32.self, n: n, rng: &rng)
            try exerciseCumulativeInts(Int64.self, n: n, rng: &rng)
        }
        var small = SeededRNG(0x5EED)
        for n in [0, 1, 255, 256, 257, 70_001] {
            try exerciseCumulativeInts(Int8.self, n: n, rng: &small)
            try exerciseCumulativeInts(UInt16.self, n: n, rng: &small)
            try exerciseCumulativeInts(UInt64.self, n: n, rng: &small)
        }
        // No validity bitmap at all.
        let dense = try MetalArray<Int32>([3, 1, 4, 1, 5, 9, 2, 6])
        XCTAssertEqual(try dense.cumulativeSum().toArray(), [3, 4, 8, 9, 14, 23, 25, 31])
        XCTAssertEqual(try dense.cumulativeMin().toArray(), [3, 1, 1, 1, 1, 1, 1, 1])
        XCTAssertEqual(try dense.cumulativeMax().toArray(), [3, 3, 4, 4, 5, 9, 9, 9])
        // A null keeps its slot null and does not break the run.
        let holes = try MetalArray<Int32>([1, nil, 2, nil, 3])
        XCTAssertEqual(try holes.cumulativeSum().toArray(), [1, nil, 3, nil, 6])
        XCTAssertEqual(try holes.cumulativeMax().toArray(), [1, nil, 2, nil, 3])
        // Every value null: every output slot null.
        let allNull = try MetalArray<Int32>([nil, nil, nil])
        XCTAssertEqual(try allNull.cumulativeSum().toArray(), [nil, nil, nil])
        // Empty.
        XCTAssertEqual(try MetalArray<Int32>([Int32]()).cumulativeSum().length, 0)
    }

    func testCumulativeFloat32() throws {
        try requireRealGPU()
        var rng = SeededRNG(0x3F32)
        for n in Self.cumulativeSizes {
            // Quarter integers keep every partial sum exactly representable, so the reassociation the
            // two-level scan performs cannot show up as a difference.
            let vals: [Float?] = (0..<n).map { _ in
                Double.random(in: 0..<1, using: &rng) < 0.25 ? nil : Float(Int.random(in: -4...4, using: &rng)) * 0.25
            }
            let a = try MetalArray<Float>(vals)
            XCTAssertEqual(try a.cumulativeSum().toArray(), cumulativeOracle(vals, identity: 0) { $0 + $1 },
                           "f32 cumulative_sum n=\(n)")
            XCTAssertEqual(try a.cumulativeMin().toArray(), cumulativeOracle(vals, identity: .infinity) { Swift.min($0, $1) },
                           "f32 cumulative_min n=\(n)")
            XCTAssertEqual(try a.cumulativeMax().toArray(), cumulativeOracle(vals, identity: -.infinity) { Swift.max($0, $1) },
                           "f32 cumulative_max n=\(n)")
        }
    }

    func testCumulativeFloat64() throws {
        try requireRealGPU()
        var rng = SeededRNG(0x6F64)
        for n in Self.cumulativeSizes {
            let vals = doubles(n: n, nulls: 0.25, range: -1000...1000, rng: &rng)
            let a = try MetalArray<Double>(vals)
            // The two-level scan reassociates the additions, so the running sum is compared with a
            // relative tolerance — relative to the running sum of magnitudes, which is what conditions
            // the answer when the signed sum passes near zero.
            let want = cumulativeOracle(vals, identity: 0) { $0 + $1 }
            let scale = cumulativeOracle(vals, identity: 0) { $0 + Swift.abs($1) }
            let got = try a.cumulativeSum().toArray()
            XCTAssertEqual(got.count, want.count, "f64 cumulative_sum count n=\(n)")
            for i in 0..<n {
                guard let g = got[i], let w = want[i], let sc = scale[i] else {
                    XCTAssertEqual(got[i] == nil, want[i] == nil, "f64 cumulative_sum nullness at \(i)")
                    continue
                }
                XCTAssertEqual(g, w, accuracy: Swift.max(sc * 1e-12, 1e-12), "f64 cumulative_sum n=\(n) at \(i)")
            }
            XCTAssertEqual(try a.cumulativeMin().toArray(), cumulativeOracle(vals, identity: .infinity) { Swift.min($0, $1) },
                           "f64 cumulative_min n=\(n)")
            XCTAssertEqual(try a.cumulativeMax().toArray(), cumulativeOracle(vals, identity: -.infinity) { Swift.max($0, $1) },
                           "f64 cumulative_max n=\(n)")
        }
    }

    // MARK: - Rejections

    func testUnsupportedCombinationsThrow() throws {
        try requireRealGPU()
        let f = try MetalArray<Float>([1, 2, 3])
        XCTAssertThrowsError(try f.bitwiseAnd(1), "bit-wise needs an integer column")
        XCTAssertThrowsError(try f.shiftLeft(1), "shifts need an integer column")
        XCTAssertThrowsError(try f.bitwiseNot(), "bit_wise_not needs an integer column")
        let i = try MetalArray<Int32>([1, 2, 3])
        XCTAssertThrowsError(try i.sqrt(), "sqrt needs a float column")
        XCTAssertThrowsError(try i.ln(), "ln needs a float column")
        XCTAssertThrowsError(try i.exp(), "exp needs a float column")
        let d = try MetalArray<Double>([1, 2, 3])
        XCTAssertThrowsError(try d.power(2), "float64 power is not implemented")
        XCTAssertThrowsError(try d.modulo(2), "float64 modulo is not implemented")
        XCTAssertThrowsError(try i.binaryMath(.minElementWise, 3), "min_element_wise has no scalar form")
        XCTAssertThrowsError(try i.bitwise(.and, try MetalArray<Int32>([1, 2])), "length mismatch")
    }

    // MARK: - Helpers

    func assertClose<F: BinaryFloatingPoint>(_ got: [F?], _ want: [F?], rel: Double, _ label: String,
                                             file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(got.count, want.count, "\(label) count", file: file, line: line)
        for i in 0..<Swift.min(got.count, want.count) {
            switch (got[i], want[i]) {
            case (nil, nil): continue
            case (let g?, let w?):
                let gd = Double(g), wd = Double(w)
                if gd.isNaN && wd.isNaN { continue }
                XCTAssertEqual(gd, wd, accuracy: Swift.max(Swift.abs(wd) * rel, rel), "\(label) at \(i)",
                               file: file, line: line)
            default:
                XCTFail("\(label) nullness differs at \(i)", file: file, line: line)
            }
        }
    }
}
