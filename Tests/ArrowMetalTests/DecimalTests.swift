import XCTest
import CArrowABI
@testable import ArrowMetal

// MARK: - Oracle

/// A two-limb signed 128-bit integer written from scratch for the tests, so that a bug in the library's
/// `ArrowDecimal128` cannot hide behind the same bug in the expected values. Everything wraps modulo 2^128,
/// exactly as the kernels do.
struct I128: Equatable, Comparable, CustomStringConvertible {
    var lo: UInt64
    var hi: UInt64

    init(lo: UInt64, hi: UInt64) { self.lo = lo; self.hi = hi }
    init(_ v: Int64) { lo = UInt64(bitPattern: v); hi = v < 0 ? UInt64.max : 0 }
    init(_ v: Int) { self.init(Int64(v)) }

    static let zero = I128(0)
    var isNegative: Bool { hi >> 63 == 1 }
    var isZero: Bool { lo == 0 && hi == 0 }

    var negated: I128 {
        let l = ~lo &+ 1
        return I128(lo: l, hi: ~hi &+ (l == 0 ? 1 : 0))
    }
    var magnitude: I128 { isNegative ? negated : self }

    static func + (a: I128, b: I128) -> I128 {
        let l = a.lo &+ b.lo
        return I128(lo: l, hi: a.hi &+ b.hi &+ (l < a.lo ? 1 : 0))
    }
    static func - (a: I128, b: I128) -> I128 { a + b.negated }
    static func * (a: I128, b: I128) -> I128 {
        let p = a.lo.multipliedFullWidth(by: b.lo)
        return I128(lo: p.low, hi: p.high &+ (a.lo &* b.hi) &+ (a.hi &* b.lo))
    }
    static func < (a: I128, b: I128) -> Bool {
        if a.hi != b.hi { return Int64(bitPattern: a.hi) < Int64(bitPattern: b.hi) }
        return a.lo < b.lo
    }

    /// Unsigned divide-with-remainder. Uses the O(1) form when the divisor fits 64 bits and a bitwise
    /// long division otherwise.
    func unsignedDivMod(_ d: I128) -> (q: I128, r: I128) {
        precondition(!d.isZero)
        if d.hi == 0 {
            let (qh, rh) = hi.quotientAndRemainder(dividingBy: d.lo)
            let (ql, r) = d.lo.dividingFullWidth((high: rh, low: lo))
            return (I128(lo: ql, hi: qh), I128(lo: r, hi: 0))
        }
        var q = I128.zero, r = I128.zero
        for i in stride(from: 127, through: 0, by: -1) {
            r = I128(lo: r.lo << 1, hi: (r.hi << 1) | (r.lo >> 63))
            let bit = i < 64 ? (lo >> UInt64(i)) & 1 : (hi >> UInt64(i - 64)) & 1
            r.lo |= bit
            if !unsignedLess(r, d) {
                r = r - d
                if i < 64 { q.lo |= (1 << UInt64(i)) } else { q.hi |= (1 << UInt64(i - 64)) }
            }
        }
        return (q, r)
    }
    private func unsignedLess(_ a: I128, _ b: I128) -> Bool { a.hi != b.hi ? a.hi < b.hi : a.lo < b.lo }

    /// Rescales by `delta` decimal places (positive multiplies, negative divides with `mode`).
    func rescaled(delta: Int, mode: DecimalRoundMode) -> I128 {
        if delta == 0 { return self }
        let factor = I128.pow10(abs(delta))
        if delta > 0 { return self * factor }
        let neg = isNegative
        let (q, r) = magnitude.unsignedDivMod(factor)
        var out = q
        var bump = false
        switch mode {
        case .round: bump = !I128(lo: r.lo << 1, hi: (r.hi << 1) | (r.lo >> 63)).unsignedLess2(factor)
        case .ceil: bump = !neg && !r.isZero
        case .floor: bump = neg && !r.isZero
        case .truncate: bump = false
        }
        if bump { out = out + I128(1) }
        return neg ? out.negated : out
    }
    func unsignedLess2(_ b: I128) -> Bool { hi != b.hi ? hi < b.hi : lo < b.lo }

    static func pow10(_ k: Int) -> I128 {
        var v = I128(1)
        for _ in 0..<k { v = v * I128(10) }
        return v
    }

    var doubleValue: Double {
        let m = magnitude
        let d = Double(m.hi) * 0x1p64 + Double(m.lo)
        return isNegative ? -d : d
    }

    var description: String {
        if isZero { return "0" }
        var m = magnitude
        var s = ""
        while !m.isZero {
            let (q, r) = m.unsignedDivMod(I128(10))
            s.append(Character(UnicodeScalar(UInt8(48 + r.lo))))
            m = q
        }
        return (isNegative ? "-" : "") + String(s.reversed())
    }

    var asDecimal128: ArrowDecimal128 { ArrowDecimal128(lo: lo, hi: hi) }
    var littleEndianBytes: [UInt8] {
        (0..<8).map { UInt8((lo >> (8 * UInt64($0))) & 0xFF) } + (0..<8).map { UInt8((hi >> (8 * UInt64($0))) & 0xFF) }
    }
    /// Sign-extended to 32 little-endian bytes (a decimal256 element).
    var littleEndianBytes256: [UInt8] {
        littleEndianBytes + [UInt8](repeating: isNegative ? 0xFF : 0x00, count: 16)
    }
}

private func decimalFromArrow(_ v: ArrowDecimal128?) -> I128? { v.map { I128(lo: $0.lo, hi: $0.hi) } }

final class DecimalTests: XCTestCase {
    let sizes = [0, 1, 33, 4097, 300_003]

    /// Deterministic values spanning both limbs, with negatives and nulls.
    func oracleValues(_ n: Int) -> [I128?] {
        (0..<n).map { i in
            if i % 7 == 3 { return nil }
            var v = I128(Int64(i) &* 1_000_003 &- 500_000)
            if i % 5 == 0 { v = v * I128(1_000_000_007) }           // pushes past 64 bits
            if i % 11 == 2 { v = v * I128(4_294_967_311) }          // and further
            if i % 3 == 0 { v = v.negated }
            return v
        }
    }

    func makeArray(_ vals: [I128?], precision: Int = 38, scale: Int = 4) throws -> MetalDecimalArray {
        try MetalDecimalArray(type: try ArrowDecimalType(precision: precision, scale: scale),
                              vals.map { $0?.asDecimal128 })
    }

    // MARK: - Type metadata

    func testFormatParsing() throws {
        XCTAssertEqual(try ArrowDecimalType.parse("d:10,3").arrowFormat, "d:10,3")
        XCTAssertEqual(try ArrowDecimalType.parse("d:10,3").bitWidth, 128)
        XCTAssertEqual(try ArrowDecimalType.parse("d:38,0").precision, 38)
        XCTAssertEqual(try ArrowDecimalType.parse("d:10,3,128").arrowFormat, "d:10,3")
        let big = try ArrowDecimalType.parse("d:76,20,256")
        XCTAssertEqual(big.arrowFormat, "d:76,20,256")
        XCTAssertEqual(big.bitWidth, 256)
        XCTAssertEqual(big.byteWidth, 32)
        XCTAssertEqual(big.limbCount, 4)
        XCTAssertNil(ArrowDecimalType(format: "l"))
        XCTAssertNil(ArrowDecimalType(format: "d:x,3"))
        XCTAssertThrowsError(try ArrowDecimalType.parse("d:39,2"))     // over decimal128's precision
        XCTAssertThrowsError(try ArrowDecimalType.parse("d:10,3,64"))  // decimal64 is not supported
        XCTAssertThrowsError(try ArrowDecimalType.parse("d:10"))
    }

    func testValueArithmeticHelpers() throws {
        // ArrowDecimal128 against the oracle on a handful of awkward values.
        let cases: [I128] = [I128(0), I128(1), I128(-1), I128(Int64.min), I128(Int64.max),
                             I128(Int64.max) * I128(1_000), I128(Int64.min) * I128(7)]
        for a in cases {
            XCTAssertEqual(a.asDecimal128.description, a.description)
            XCTAssertEqual(a.asDecimal128.negated, a.negated.asDecimal128)
            for b in cases {
                XCTAssertEqual((a.asDecimal128 + b.asDecimal128), (a + b).asDecimal128)
                XCTAssertEqual((a.asDecimal128 - b.asDecimal128), (a - b).asDecimal128)
                XCTAssertEqual((a.asDecimal128 * b.asDecimal128), (a * b).asDecimal128)
                XCTAssertEqual(a.asDecimal128 < b.asDecimal128, a < b)
            }
        }
        XCTAssertEqual(ArrowDecimal128(lo: 0, hi: 0).littleEndianBytes, [UInt8](repeating: 0, count: 16))
        XCTAssertEqual(ArrowDecimal128(-1).littleEndianBytes, [UInt8](repeating: 0xFF, count: 16))
    }

    func testBuildAndRead() throws {
        try requireRealGPU()
        for n in sizes {
            let vals = oracleValues(n)
            let a = try makeArray(vals)
            XCTAssertEqual(a.length, n)
            XCTAssertEqual(a.nullCount, vals.filter { $0 == nil }.count)
            XCTAssertEqual(a.arrowFormat, "d:38,4")
            XCTAssertEqual(a.toArray().map(decimalFromArrow), vals)
            if n > 0 {
                XCTAssertEqual(a.rawBytes(at: 0).count, 16)
                if let v = vals[0] { XCTAssertEqual(a.rawBytes(at: 0), v.littleEndianBytes) }
            }
        }
    }

    // MARK: - Compare (GPU)

    func testCompareScalarAndArray() throws {
        try requireRealGPU()
        for n in sizes {
            let vals = oracleValues(n)
            let a = try makeArray(vals)
            let scalar = I128(1_000_003 &* 400) * I128(3)
            for op in CompareOp.allCases {
                let got = try a.compare(op, scalar.asDecimal128).toArray()
                let want: [Bool?] = vals.map { v in v.map { compareOracle(op, $0, scalar) } }
                XCTAssertEqual(got, want, "scalar \(op) n=\(n)")
            }
            // Array form: compare against a shifted copy so both sides carry nulls.
            let other = try makeArray(vals.enumerated().map { (i, v) in i % 4 == 1 ? nil : v.map { $0 + I128(500) } })
            let otherVals = other.toArray().map(decimalFromArrow)
            for op in CompareOp.allCases {
                let got = try a.compare(op, other).toArray()
                let want: [Bool?] = (0..<n).map { i in
                    guard let x = vals[i], let y = otherVals[i] else { return nil }
                    return compareOracle(op, x, y)
                }
                XCTAssertEqual(got, want, "array \(op) n=\(n)")
            }
        }
    }

    func compareOracle(_ op: CompareOp, _ a: I128, _ b: I128) -> Bool {
        switch op {
        case .eq: return a == b
        case .ne: return a != b
        case .lt: return a < b
        case .le: return a <= b
        case .gt: return a > b
        case .ge: return a >= b
        }
    }

    // MARK: - Reductions (GPU accumulate, CPU combine)

    func testSumMinMax() throws {
        try requireRealGPU()
        for n in sizes {
            let vals = oracleValues(n)
            let a = try makeArray(vals)
            let valid = vals.compactMap { $0 }
            var wantSum = I128.zero
            for v in valid { wantSum = wantSum + v }
            XCTAssertEqual(decimalFromArrow(try a.sum()), valid.isEmpty ? nil : wantSum, "sum n=\(n)")
            XCTAssertEqual(decimalFromArrow(try a.min()), valid.min(), "min n=\(n)")
            XCTAssertEqual(decimalFromArrow(try a.max()), valid.max(), "max n=\(n)")
        }
        // All null and empty return nil, matching Arrow.
        let allNull = try makeArray([nil, nil, nil])
        XCTAssertNil(try allNull.sum()); XCTAssertNil(try allNull.min()); XCTAssertNil(try allNull.max())
    }

    /// The sum wraps modulo 2^128 rather than raising, like Arrow's unchecked `sum`.
    func testSumOverflowWraps() throws {
        try requireRealGPU()
        let big = I128(lo: 0, hi: 0x4000_0000_0000_0000)     // 2^126
        let a = try makeArray([big, big, big], precision: 38, scale: 0)
        XCTAssertEqual(decimalFromArrow(try a.sum()), big + big + big)
        XCTAssertTrue((big + big + big).isNegative, "three times 2^126 must have wrapped into the sign bit")
    }

    // MARK: - Arithmetic (GPU)

    func testAddSubtractAndUnary() throws {
        try requireRealGPU()
        for n in sizes {
            let vals = oracleValues(n)
            let a = try makeArray(vals)
            let bVals: [I128?] = (0..<n).map { i in i % 4 == 1 ? nil : I128(Int64(i) &* 7 &- 13) * I128(1_000_000_007) }
            let b = try makeArray(bVals)

            let add = try a.adding(b)
            XCTAssertEqual(add.type.arrowFormat, "d:38,4")
            XCTAssertEqual(add.toArray().map(decimalFromArrow), zipOracle(vals, bVals) { $0 + $1 }, "add n=\(n)")
            XCTAssertEqual(try a.subtracting(b).toArray().map(decimalFromArrow), zipOracle(vals, bVals) { $0 - $1 }, "sub n=\(n)")

            let s = I128(123_456_789) * I128(1_000_000_007)
            XCTAssertEqual(try a.adding(s.asDecimal128).toArray().map(decimalFromArrow), vals.map { $0.map { $0 + s } })
            XCTAssertEqual(try a.subtracting(s.asDecimal128).toArray().map(decimalFromArrow), vals.map { $0.map { $0 - s } })

            XCTAssertEqual(try a.negated().toArray().map(decimalFromArrow), vals.map { $0?.negated })
            XCTAssertEqual(try a.absoluteValue().toArray().map(decimalFromArrow), vals.map { $0?.magnitude })
            XCTAssertEqual(try a.sign().toArray(), vals.map { v in v.map { Int32($0.isZero ? 0 : ($0.isNegative ? -1 : 1)) } })
        }
    }

    func zipOracle(_ a: [I128?], _ b: [I128?], _ f: (I128, I128) -> I128) -> [I128?] {
        (0..<a.count).map { i in
            guard let x = a[i], let y = b[i] else { return nil }
            return f(x, y)
        }
    }

    /// Addition wraps past 128 bits, as Arrow's unchecked `add` does.
    func testAddOverflowWraps() throws {
        try requireRealGPU()
        let maxV = I128(lo: UInt64.max, hi: 0x7FFF_FFFF_FFFF_FFFF)
        let a = try makeArray([maxV, maxV.negated], precision: 38, scale: 0)
        let b = try makeArray([I128(1), I128(-1)], precision: 38, scale: 0)
        XCTAssertEqual(try a.adding(b).toArray().map(decimalFromArrow), [maxV + I128(1), maxV.negated - I128(1)])
    }

    func testMultiply() throws {
        try requireRealGPU()
        for n in [0, 1, 33, 4097] {
            let vals = oracleValues(n)
            let a = try makeArray(vals, precision: 18, scale: 2)
            let k: Int64 = -1_000_003
            XCTAssertEqual(try a.multiplied(by: k).toArray().map(decimalFromArrow), vals.map { $0.map { $0 * I128(k) } })

            let bVals: [I128?] = (0..<n).map { i in i % 6 == 5 ? nil : I128(Int64(i) &+ 3) }
            let b = try MetalDecimalArray(type: try ArrowDecimalType(precision: 10, scale: 3), bVals.map { $0?.asDecimal128 })
            let p = try a.multiplied(by: b)
            // Arrow's rule: precision p1 + p2 + 1, scale s1 + s2.
            XCTAssertEqual(p.type.precision, 18 + 10 + 1)
            XCTAssertEqual(p.type.scale, 5)
            XCTAssertEqual(p.arrowFormat, "d:29,5")
            XCTAssertEqual(p.toArray().map(decimalFromArrow), zipOracle(vals, bVals) { $0 * $1 }, "mul n=\(n)")
        }
        // A product whose precision does not fit decimal128 is rejected rather than silently truncated.
        let x = try makeArray([I128(1)], precision: 30, scale: 2)
        let y = try makeArray([I128(1)], precision: 30, scale: 2)
        XCTAssertThrowsError(try x.multiplied(by: y)) { e in
            XCTAssertTrue("\(e)".contains("precision"), "\(e)")
        }
    }

    // MARK: - Rounding / rescaling (GPU)

    func testRoundCeilFloorTruncate() throws {
        try requireRealGPU()
        // Values around the .5 boundary in both signs, at scale 4.
        let raw: [Int64] = [0, 5, -5, 15, -15, 4, -4, 6, -6, 12_345, -12_345, 10_000, -10_000,
                            99_995, -99_995, 1, -1, 50_000, -50_000]
        let vals: [I128?] = raw.map { I128($0) } + [nil]
        let a = try makeArray(vals, precision: 20, scale: 4)
        for (mode, target) in [(DecimalRoundMode.round, 3), (.round, 0), (.ceil, 2), (.floor, 2), (.truncate, 2),
                               (.ceil, 0), (.floor, 0), (.truncate, 0)] {
            let got = try a.rescaled(to: target, mode: mode)
            XCTAssertEqual(got.type.scale, target)
            XCTAssertEqual(got.toArray().map(decimalFromArrow),
                           vals.map { $0?.rescaled(delta: target - 4, mode: mode) }, "\(mode) -> \(target)")
        }
        // Scaling up is exact and widens the precision.
        let up = try a.rescaled(to: 8, mode: .round)
        XCTAssertEqual(up.type.arrowFormat, "d:24,8")
        XCTAssertEqual(up.toArray().map(decimalFromArrow), vals.map { $0?.rescaled(delta: 4, mode: .round) })
        // The named helpers agree with rescaled(to:mode:).
        XCTAssertEqual(try a.rounded(toScale: 2).toArray().map(decimalFromArrow),
                       try a.rescaled(to: 2, mode: .round).toArray().map(decimalFromArrow))
        XCTAssertEqual(try a.ceiled(toScale: 2).toArray().map(decimalFromArrow),
                       try a.rescaled(to: 2, mode: .ceil).toArray().map(decimalFromArrow))
        XCTAssertEqual(try a.floored(toScale: 2).toArray().map(decimalFromArrow),
                       try a.rescaled(to: 2, mode: .floor).toArray().map(decimalFromArrow))
        XCTAssertEqual(try a.truncated(toScale: 2).toArray().map(decimalFromArrow),
                       try a.rescaled(to: 2, mode: .truncate).toArray().map(decimalFromArrow))
    }

    /// Rounding at 128-bit magnitudes, where the divisor no longer fits 64 bits.
    func testRoundLargeMagnitudes() throws {
        try requireRealGPU()
        var vals: [I128?] = []
        for k in 0..<12 {
            var v = I128.pow10(30) + I128(Int64(k) &* 5 &+ 5)
            if k % 2 == 1 { v = v.negated }
            vals.append(v)
        }
        vals.append(nil)
        let a = try makeArray(vals, precision: 38, scale: 22)
        for mode in DecimalRoundMode.allCases {
            for target in [21, 12, 0] {
                let got = try a.rescaled(to: target, mode: mode).toArray().map(decimalFromArrow)
                XCTAssertEqual(got, vals.map { $0?.rescaled(delta: target - 22, mode: mode) }, "\(mode) -> \(target)")
            }
        }
    }

    func testRescaleAtSize() throws {
        try requireRealGPU()
        let n = 300_003
        let vals: [I128?] = (0..<n).map { i in i % 7 == 3 ? nil : I128(Int64(i) &* 12_345 &- 999_999) }
        let a = try makeArray(vals, precision: 30, scale: 6)
        let got = try a.rounded(toScale: 2).toArray().map(decimalFromArrow)
        XCTAssertEqual(got, vals.map { $0?.rescaled(delta: -4, mode: .round) })
    }

    // MARK: - Casts (CPU)

    func testCasts() throws {
        try requireRealGPU()
        let vals: [I128?] = [I128(123_456), I128(-123_456), I128(0), nil, I128(1), I128(-1),
                             I128(999_999_999_999)]
        let a = try makeArray(vals, precision: 20, scale: 3)
        let f = try a.toFloat64()
        XCTAssertEqual(f.length, vals.count)
        for (i, v) in vals.enumerated() {
            if let v {
                XCTAssertEqual(try XCTUnwrap(f[i]), v.doubleValue / 1000.0, accuracy: 1e-9)
            } else {
                XCTAssertNil(f[i])
            }
        }
        XCTAssertEqual(a.toDoubleArray().map { $0.map { ($0 * 1000).rounded() } }, vals.map { $0?.doubleValue })

        // float64 -> decimal(_, 3): the scaled value is rounded half away from zero.
        let d = try MetalArray<Double>([1.2345, -1.2345, 0.0, nil, 12345.6789])
        let back = try MetalDecimalArray.fromFloat64(d, type: try ArrowDecimalType(precision: 38, scale: 3))
        XCTAssertEqual(back.toArray().map(decimalFromArrow),
                       [I128(1235), I128(-1235), I128(0), nil, I128(12_345_679)])
        XCTAssertEqual(back.nullCount, 1)

        // int64 -> decimal(_, 4) multiplies by 10^4 exactly.
        let i = try MetalArray<Int64>([7, -7, 0, nil, 1_000_000_000_000])
        let di = try MetalDecimalArray.fromInt64(i, type: try ArrowDecimalType(precision: 38, scale: 4))
        XCTAssertEqual(di.toArray().map(decimalFromArrow),
                       [I128(70_000), I128(-70_000), I128(0), nil, I128(1_000_000_000_000) * I128(10_000)])

        // A non-finite value becomes null rather than an undefined pattern.
        let bad = try MetalArray<Double>([Double.infinity, Double.nan, 1.5])
        let cast = try MetalDecimalArray.fromFloat64(bad, type: try ArrowDecimalType(precision: 38, scale: 1))
        XCTAssertEqual(cast.toArray().map(decimalFromArrow), [nil, nil, I128(15)])
    }

    // MARK: - Selection (GPU)

    func testFilterTakeSlice() throws {
        try requireRealGPU()
        for n in sizes {
            let vals = oracleValues(n)
            let a = try makeArray(vals)
            let maskVals: [Bool?] = (0..<n).map { i in i % 9 == 4 ? nil : (i % 3 != 1) }
            let mask = try MetalBooleanArray.allocate(length: n, withValidity: true)
            for (i, v) in maskVals.enumerated() {
                if let v {
                    Bitmap.set(mask.validity!.mutableTyped(UInt8.self), i)
                    if v { Bitmap.set(mask.values.mutableTyped(UInt8.self), i) }
                }
            }
            mask.recomputeNullCount()
            let kept = try a.filter(mask)
            let want = (0..<n).filter { maskVals[$0] == true }.map { vals[$0] }
            XCTAssertEqual(kept.length, want.count, "filter length n=\(n)")
            XCTAssertEqual(kept.toArray().map(decimalFromArrow), want, "filter n=\(n)")
            XCTAssertEqual(kept.type, a.type)

            guard n > 0 else { continue }
            let idx: [Int32?] = [0, Int32(n - 1), Int32(n / 2), nil, 0]
            let taken = try a.take(try MetalArray<Int32>(idx))
            XCTAssertEqual(taken.toArray().map(decimalFromArrow), idx.map { $0.flatMap { vals[Int($0)] } })

            let off = Swift.min(32, n - 1), len = Swift.min(17, n - off)
            let sl = try a.slice(offset: off, length: len)
            XCTAssertEqual(sl.toArray().map(decimalFromArrow), Array(vals[off..<(off + len)]))
            let odd = Swift.min(7, n - 1), oddLen = Swift.min(11, n - odd)
            let sl2 = try a.slice(offset: odd, length: oddLen)
            XCTAssertEqual(sl2.toArray().map(decimalFromArrow), Array(vals[odd..<(odd + oddLen)]))
        }
    }

    func testTakeOutOfRangeThrows() throws {
        try requireRealGPU()
        let a = try makeArray([I128(1), I128(2)])
        XCTAssertThrowsError(try a.take(try MetalArray<Int32>([0, 5])))
        XCTAssertThrowsError(try a.take(try MetalArray<Int32>([-1])))
    }

    // MARK: - C Data Interface

    func testExportImportRoundTrip() throws {
        try requireRealGPU()
        let vals = oracleValues(1000)
        let a = try makeArray(vals, precision: 30, scale: 7)
        var schema = ArrowSchema(), arr = ArrowArray()
        a.exportArrowSchema(name: "amount", into: &schema)
        a.exportArrowArray(into: &arr)
        XCTAssertEqual(String(cString: schema.format), "d:30,7")
        XCTAssertEqual(String(cString: schema.name), "amount")
        XCTAssertEqual(arr.length, 1000)
        XCTAssertEqual(arr.n_buffers, 2)
        XCTAssertEqual(arr.null_count, Int64(a.nullCount))
        XCTAssertEqual(arr.buffers[1], UnsafeRawPointer(a.values.contents))

        let r = try importArrowArray(schema: &schema, array: &arr)
        XCTAssertTrue(r.zeroCopy)
        XCTAssertNil(arr.release)
        guard case .decimal(let b) = r.array else { return XCTFail("expected a decimal array") }
        XCTAssertEqual(b.type, a.type)
        XCTAssertEqual(b.arrowFormat, "d:30,7")
        XCTAssertEqual(b.toArray().map(decimalFromArrow), vals)
        XCTAssertTrue(b.values.mtl === a.values.mtl, "re-importing our own export must share the MTLBuffer")
        schema.release?(&schema)
    }

    /// A foreign producer's decimal array (page-aligned buffers, a non-zero offset, null_count = -1).
    func testImportForeignWithOffset() throws {
        try requireRealGPU()
        let p = CProducer()
        let vals = oracleValues(64)
        var bytes: [UInt8] = []
        for v in vals { bytes.append(contentsOf: (v ?? I128(0)).littleEndianBytes) }
        let offset = 5
        let arr = p.array(length: vals.count - offset, nullCount: -1,
                          buffers: [p.bitmap(vals.map { $0 != nil }), p.copied(bytes)])
        arr.pointee.offset = Int64(offset)
        let schema = p.schema("d:38,4")
        let r = try importArrowArray(schema: schema, array: arr)
        guard case .decimal(let d) = r.array else { return XCTFail("expected a decimal array") }
        XCTAssertEqual(d.length, vals.count - offset)
        XCTAssertEqual(d.toArray().map(decimalFromArrow), Array(vals[offset...]))
        XCTAssertEqual(d.nullCount, vals[offset...].filter { $0 == nil }.count)
        p.destroy()
    }

    func testAnyMetalArrayAndRecordBatch() throws {
        try requireRealGPU()
        let n = 512
        let vals = oracleValues(n)
        let col = AnyMetalArray.decimal(try makeArray(vals, precision: 20, scale: 2))
        XCTAssertEqual(col.arrowFormat, "d:20,2")
        XCTAssertEqual(col.length, n)
        XCTAssertEqual(col.nullCount, vals.filter { $0 == nil }.count)
        XCTAssertNotNil(col.asDecimal)

        let batch = try MetalRecordBatch(names: ["id", "amount"], columns: [
            .int64(try MetalArray<Int64>((0..<n).map { Int64($0) })),
            col,
        ])
        let mask = try batch["id"]!.asInt64!.compare(.lt, 100)
        let f = try batch.filter(mask)
        XCTAssertEqual(f.length, 100)
        XCTAssertEqual(f["amount"]!.asDecimal!.toArray().map(decimalFromArrow), Array(vals[0..<100]))
        let t = try batch.take(try MetalArray<Int32>([3, 1, 0]))
        XCTAssertEqual(t["amount"]!.asDecimal!.toArray().map(decimalFromArrow), [vals[3], vals[1], vals[0]])
        let s = try batch.slice(offset: 64, length: 10)
        XCTAssertEqual(s["amount"]!.asDecimal!.toArray().map(decimalFromArrow), Array(vals[64..<74]))

        // Struct export carries the decimal child's format.
        var schema = ArrowSchema(), arr = ArrowArray()
        batch.exportArrowSchema(name: "b", into: &schema)
        batch.exportArrowArray(into: &arr)
        XCTAssertEqual(String(cString: schema.children[1]!.pointee.format), "d:20,2")
        let back = try importArrowRecordBatch(schema: &schema, array: &arr)
        XCTAssertEqual(back.batch["amount"]!.asDecimal!.toArray().map(decimalFromArrow), vals)
        schema.release?(&schema)
    }

    // MARK: - decimal256

    func testDecimal256() throws {
        try requireRealGPU()
        let t = try ArrowDecimalType(precision: 50, scale: 6, bitWidth: 256)
        for n in [0, 1, 33, 4097] {
            let vals = oracleValues(n)
            let a = try MetalDecimalArray(type: t, vals.map { $0?.asDecimal128 })
            XCTAssertEqual(a.arrowFormat, "d:50,6,256")
            XCTAssertEqual(a.toArray().map(decimalFromArrow), vals, "values n=\(n)")
            if n > 0 { XCTAssertEqual(a.rawBytes(at: 0).count, 32) }

            // Compare against a scalar (sign-extended to 256 bits) and against another 256-bit column.
            let scalar = I128(0)
            XCTAssertEqual(try a.compare(.gt, scalar.asDecimal128).toArray(),
                           vals.map { v in v.map { !$0.isNegative && !$0.isZero } }, "gt n=\(n)")
            let b = try MetalDecimalArray(type: t, vals.map { $0?.asDecimal128 })
            XCTAssertEqual(try a.compare(.eq, b).toArray(), vals.map { $0 != nil ? true : nil }, "eq n=\(n)")

            // sum, filter and take work on 32-byte elements.
            let valid = vals.compactMap { $0 }
            var wantSum = I128.zero
            for v in valid { wantSum = wantSum + v }
            XCTAssertEqual(decimalFromArrow(try a.sum()), valid.isEmpty ? nil : wantSum, "sum n=\(n)")

            let mask = try MetalBooleanArray((0..<n).map { $0 % 3 == 0 })
            let kept = try a.filter(mask)
            XCTAssertEqual(kept.toArray().map(decimalFromArrow), (0..<n).filter { $0 % 3 == 0 }.map { vals[$0] })
            XCTAssertEqual(kept.type.bitWidth, 256)
            guard n > 0 else { continue }
            let taken = try a.take(try MetalArray<Int32>([Int32(n - 1), 0]))
            XCTAssertEqual(taken.toArray().map(decimalFromArrow), [vals[n - 1], vals[0]])
            XCTAssertEqual(try a.slice(offset: 0, length: Swift.min(9, n)).toArray().map(decimalFromArrow),
                           Array(vals[0..<Swift.min(9, n)]))
        }
        // Everything else says so clearly instead of computing the wrong thing.
        let a = try MetalDecimalArray(type: t, [I128(1).asDecimal128])
        for (name, body) in [("add", { try a.adding(a) }), ("negate", { try a.negated() }),
                             ("abs", { try a.absoluteValue() }),
                             ("round", { try a.rescaled(to: 2, mode: .round) })] as [(String, () throws -> MetalDecimalArray)] {
            XCTAssertThrowsError(try body()) { e in
                XCTAssertTrue("\(e)".contains("decimal128"), "\(name): \(e)")
            }
        }
        XCTAssertThrowsError(try a.min())
        XCTAssertThrowsError(try a.sign())
    }

    func testDecimal256Interop() throws {
        try requireRealGPU()
        let t = try ArrowDecimalType(precision: 60, scale: 3, bitWidth: 256)
        let vals = oracleValues(100)
        let a = try MetalDecimalArray(type: t, vals.map { $0?.asDecimal128 })
        var schema = ArrowSchema(), arr = ArrowArray()
        a.exportArrowSchema(into: &schema)
        a.exportArrowArray(into: &arr)
        XCTAssertEqual(String(cString: schema.format), "d:60,3,256")
        let r = try importArrowArray(schema: &schema, array: &arr)
        guard case .decimal(let b) = r.array else { return XCTFail("expected a decimal array") }
        XCTAssertEqual(b.type.bitWidth, 256)
        XCTAssertEqual(b.toArray().map(decimalFromArrow), vals)
        schema.release?(&schema)

        // A foreign 32-byte-per-element producer.
        let p = CProducer()
        var bytes: [UInt8] = []
        for v in vals { bytes.append(contentsOf: (v ?? I128(0)).littleEndianBytes256) }
        let foreign = p.array(length: vals.count, nullCount: -1,
                              buffers: [p.bitmap(vals.map { $0 != nil }), p.copied(bytes)])
        let r2 = try importArrowArray(schema: p.schema("d:60,3,256"), array: foreign)
        XCTAssertEqual(r2.array.asDecimal!.toArray().map(decimalFromArrow), vals)
        p.destroy()
    }

    // MARK: - Errors

    func testErrors() throws {
        try requireRealGPU()
        let a = try makeArray([I128(1), I128(2)], precision: 20, scale: 4)
        let differentScale = try makeArray([I128(1), I128(2)], precision: 20, scale: 2)
        XCTAssertThrowsError(try a.adding(differentScale)) { e in
            XCTAssertTrue("\(e)".contains("scale"), "\(e)")
        }
        XCTAssertThrowsError(try a.compare(.eq, differentScale)) { e in
            XCTAssertTrue("\(e)".contains("scale"), "\(e)")
        }
        let shorter = try makeArray([I128(1)], precision: 20, scale: 4)
        XCTAssertThrowsError(try a.adding(shorter)) { e in
            XCTAssertTrue("\(e)".contains("length"), "\(e)")
        }
        XCTAssertThrowsError(try a.filter(try MetalBooleanArray([true])))
        XCTAssertThrowsError(try a.slice(offset: 1, length: 5))
        XCTAssertThrowsError(try a.rescaled(to: 39, mode: .round))
        // Widening a decimal(38, s) has nowhere to go.
        let wide = try makeArray([I128(1)], precision: 38, scale: 4)
        XCTAssertThrowsError(try wide.rescaled(to: 10, mode: .round)) { e in
            XCTAssertTrue("\(e)".contains("precision"), "\(e)")
        }
    }
}
