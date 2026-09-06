import XCTest
@testable import ArrowMetal

/// Arrow's checked (overflow-raising) arithmetic against CPU oracles.
///
/// Two claims are tested for every op:
///
///   1. **In range, nothing changes.** The checked op's values are bit-identical to the unchecked op's,
///      because the checked path runs the very same value kernel and only adds a read-only check pass.
///   2. **Out of range, it raises at the right row.** The oracle scans the input sequentially, exactly as
///      Arrow's own kernels do, and the assertion pins both the Arrow message and the first offending
///      index the GPU reported.
///
/// The oracles restate the *defined* semantics rather than calling Swift's operators, because the
/// interesting cases (`Int8.max + 1`, `Int64.min / -1`, an unsigned negate, a shift by the bit width)
/// are precisely the ones where Swift traps.
final class CheckedTests: XCTestCase {

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

    /// The sizes the task asks for: empty, single, a partial threadgroup, several threadgroups, and one
    /// run long enough that the atomic `min` genuinely races across thousands of threadgroups.
    static let sizes = [0, 1, 33, 4097, 1_000_003]
    /// Only the four widths below run the million-row pass; the rest would multiply the suite's runtime
    /// without exercising anything the small sizes do not.
    static let bigSize = 1_000_003

    // MARK: - Oracles

    /// Arrow's shift "precision": the bit width, one less on a signed type.
    static func shiftPrecision<T: FixedWidthInteger>(_: T.Type) -> Int { T.bitWidth - (T.isSigned ? 1 : 0) }

    static func shiftFails<T: FixedWidthInteger>(_ amount: T) -> Bool {
        if T.isSigned {
            let k = Int64(truncatingIfNeeded: amount)
            return k < 0 || k >= Int64(shiftPrecision(T.self))
        }
        return UInt64(truncatingIfNeeded: amount) >= UInt64(shiftPrecision(T.self))
    }

    /// Repeated squaring with an overflow report, squaring only while another exponent bit remains —
    /// the same evaluation order the GPU kernel and Arrow use.
    static func powerFails<T: FixedWidthInteger>(_ a: T, _ b: T) -> Bool {
        if T.isSigned && b < 0 { return true }
        var e = UInt64(truncatingIfNeeded: b)
        var r: T = 1, base = a
        while e != 0 {
            if e & 1 == 1 {
                let m = r.multipliedReportingOverflow(by: base)
                if m.overflow { return true }
                r = m.partialValue
            }
            e >>= 1
            if e != 0 {
                let s = base.multipliedReportingOverflow(by: base)
                if s.overflow { return true }
                base = s.partialValue
            }
        }
        return false
    }

    static func divideFails<T: FixedWidthInteger>(_ a: T, _ b: T) -> Bool {
        if b == 0 { return true }
        return T.isSigned && a == T.min && b == T(truncatingIfNeeded: -1)
    }

    // MARK: - Assertions

    /// Asserts that `checked` raises `expected` (op name, first index) or, when `expected` is nil, that it
    /// returns exactly what `unchecked` returns.
    func expectChecked<T: ArrowPrimitive>(_ label: String, _ opName: String, _ expected: Int?,
                                          checked: () throws -> MetalArray<T>,
                                          unchecked: () throws -> MetalArray<T>) {
        if let want = expected {
            do {
                _ = try checked()
                XCTFail("\(label) \(opName): expected a raise at index \(want)")
            } catch let e as ArrowMetalError {
                guard case .overflow(let op, let index, _) = e else {
                    return XCTFail("\(label) \(opName): expected an overflow error, got \(e)")
                }
                XCTAssertEqual(op, opName, "\(label): error names the wrong op")
                XCTAssertEqual(index, want, "\(label) \(opName): wrong first offending index")
            } catch {
                XCTFail("\(label) \(opName): unexpected error \(error)")
            }
        } else {
            do {
                let got = try checked(), want = try unchecked()
                XCTAssertTrue(Self.identical(got, want), "\(label) \(opName): values differ from the unchecked op")
            } catch {
                XCTFail("\(label) \(opName): unexpected raise \(error)")
            }
        }
    }

    /// Bit-identity: same length, same validity, same raw value bytes (so NaN payloads count too).
    static func identical<T: ArrowPrimitive>(_ a: MetalArray<T>, _ b: MetalArray<T>) -> Bool {
        guard a.length == b.length else { return false }
        for i in 0..<a.length {
            guard a.isValid(i) == b.isValid(i) else { return false }
            guard !a.isValid(i) || withUnsafeBytes(of: a.valuePointer[i], { Array($0) })
                == withUnsafeBytes(of: b.valuePointer[i], { Array($0) }) else { return false }
        }
        return true
    }

    // MARK: - Input builders

    func ints<T: ArrowPrimitive & FixedWidthInteger>(_: T.Type, n: Int, nulls: Double, rng: inout SeededRNG) -> [T?] {
        (0..<n).map { _ in
            let null = Double.random(in: 0..<1, using: &rng) < nulls
            return null ? nil : T(truncatingIfNeeded: rng.next())
        }
    }
    /// Values small enough that add / subtract / multiply / cumulative sums cannot leave the type.
    func smallInts<T: ArrowPrimitive & FixedWidthInteger>(_: T.Type, n: Int, nulls: Double, rng: inout SeededRNG) -> [T?] {
        (0..<n).map { _ in
            Double.random(in: 0..<1, using: &rng) < nulls ? nil : T(truncatingIfNeeded: Int64.random(in: -3...3, using: &rng))
        }
    }
    func shiftAmounts<T: ArrowPrimitive & FixedWidthInteger>(_: T.Type, n: Int, rng: inout SeededRNG) -> [T?] {
        (0..<n).map { _ in T(truncatingIfNeeded: Int64.random(in: -2...Int64(T.bitWidth + 2), using: &rng)) }
    }

    // MARK: - Integer ops, both paths, driven by the oracle

    func exerciseIntegers<T: ArrowPrimitive & FixedWidthInteger>(_: T.Type, n: Int, rng: inout SeededRNG) throws {
        let label = "\(T.arrowFormat) n=\(n)"
        let av = ints(T.self, n: n, nulls: 0.2, rng: &rng)
        var bv = ints(T.self, n: n, nulls: 0.2, rng: &rng)
        // Guarantee divide-by-zero and INT_MIN / -1 rows once the array is big enough to hold them.
        if n > 8 { bv[3] = 0; bv[5] = T(truncatingIfNeeded: -1) }
        let sv = shiftAmounts(T.self, n: n, rng: &rng)
        let a = try MetalArray<T>(av), b = try MetalArray<T>(bv), s = try MetalArray<T>(sv)

        /// First row where both sides are valid and `fails` says so.
        func first(_ other: [T?], _ fails: (T, T) -> Bool) -> Int? {
            for i in 0..<n {
                guard let x = av[i], let y = other[i] else { continue }
                if fails(x, y) { return i }
            }
            return nil
        }
        func firstUnary(_ fails: (T) -> Bool) -> Int? {
            for i in 0..<n { if let x = av[i], fails(x) { return i } }
            return nil
        }

        // add / subtract / multiply / divide, array form
        expectChecked(label, "add_checked", first(bv) { $0.addingReportingOverflow($1).overflow },
                      checked: { try a.addChecked(b) }, unchecked: { try a.add(b) })
        expectChecked(label, "subtract_checked", first(bv) { $0.subtractingReportingOverflow($1).overflow },
                      checked: { try a.subtractChecked(b) }, unchecked: { try a.subtract(b) })
        expectChecked(label, "multiply_checked", first(bv) { $0.multipliedReportingOverflow(by: $1).overflow },
                      checked: { try a.multiplyChecked(b) }, unchecked: { try a.multiply(b) })
        expectChecked(label, "divide_checked", first(bv) { Self.divideFails($0, $1) },
                      checked: { try a.divideChecked(b) }, unchecked: { try a.divide(b) })
        expectChecked(label, "power_checked", first(bv) { Self.powerFails($0, $1) },
                      checked: { try a.powerChecked(b) }, unchecked: { try a.power(b) })
        expectChecked(label, "shift_left_checked", first(sv) { _, k in Self.shiftFails(k) },
                      checked: { try a.shiftLeftChecked(s) }, unchecked: { try a.bitwise(.shl, s) })
        expectChecked(label, "shift_right_checked", first(sv) { _, k in Self.shiftFails(k) },
                      checked: { try a.shiftRightChecked(s) }, unchecked: { try a.bitwise(.shr, s) })

        // Scalar forms, with a scalar chosen to exercise both paths.
        for scalar: T in [0, 1, T(truncatingIfNeeded: -1), T.max, T.min] {
            let constant = [T?](repeating: scalar, count: n)
            let l = "\(label) scalar=\(scalar)"
            expectChecked(l, "add_checked", first(constant) { $0.addingReportingOverflow($1).overflow },
                          checked: { try a.addChecked(scalar) }, unchecked: { try a.add(scalar) })
            expectChecked(l, "multiply_checked", first(constant) { $0.multipliedReportingOverflow(by: $1).overflow },
                          checked: { try a.multiplyChecked(scalar) }, unchecked: { try a.multiply(scalar) })
            expectChecked(l, "divide_checked", first(constant) { Self.divideFails($0, $1) },
                          checked: { try a.divideChecked(scalar) }, unchecked: { try a.divide(scalar) })
        }

        // Unary
        let negFails: (T) -> Bool = T.isSigned ? { $0 == T.min } : { $0 != 0 }
        expectChecked(label, "negate_checked", firstUnary(negFails),
                      checked: { try a.negateChecked() }, unchecked: { try a.negate() })
        expectChecked(label, "abs_checked", firstUnary { T.isSigned && $0 == T.min },
                      checked: { try a.absChecked() }, unchecked: { try a.abs() })

        // Cumulative and pairwise: values kept small so the in-range branch is actually reachable.
        let cv = smallInts(T.self, n: n, nulls: 0.2, rng: &rng)
        let c = try MetalArray<T>(cv)
        expectChecked(label, "cumulative_sum_checked", Self.firstCumulativeFailure(cv, T.zero, { $0.addingReportingOverflow($1) }),
                      checked: { try c.cumulativeSumChecked() }, unchecked: { try c.cumulative(.sum) })
        expectChecked(label, "cumulative_prod_checked", Self.firstCumulativeFailure(cv, 1, { $0.multipliedReportingOverflow(by: $1) }),
                      checked: { try c.cumulativeProdChecked() }, unchecked: { try c.cumulativeProd() })
        for period in [1, 3] {
            expectChecked(label, "pairwise_diff_checked", Self.firstPairwiseFailure(av, period: period),
                          checked: { try a.pairwiseDiffChecked(period: period) },
                          unchecked: { try a.pairwiseDiff(period: period) })
        }
    }

    /// The sequential recurrence Arrow evaluates: the running value carries across nulls, and the first
    /// step that leaves the type is the failure.
    static func firstCumulativeFailure<T: FixedWidthInteger>(_ v: [T?], _ identity: T,
                                                             _ combine: (T, T) -> (partialValue: T, overflow: Bool)) -> Int? {
        var run = identity
        for i in 0..<v.count {
            guard let x = v[i] else { continue }
            let r = combine(run, x)
            if r.overflow { return i }
            run = r.partialValue
        }
        return nil
    }

    static func firstPairwiseFailure<T: FixedWidthInteger>(_ v: [T?], period: Int) -> Int? {
        for i in 0..<v.count {
            let j = i - period
            guard j >= 0, j < v.count, let x = v[i], let y = v[j] else { continue }
            if x.subtractingReportingOverflow(y).overflow { return i }
        }
        return nil
    }

    func testCheckedIntegersAllTypes() throws {
        try requireRealGPU()
        var rng = SeededRNG(0xC4EC_4ED0)
        for n in Self.sizes {
            // The million-row pass runs on one width of each signedness: it is there to race the atomics,
            // not to re-cover semantics the small sizes already pin down.
            if n == Self.bigSize {
                try exerciseIntegers(Int32.self, n: n, rng: &rng)
                try exerciseIntegers(UInt64.self, n: n, rng: &rng)
                continue
            }
            try exerciseIntegers(Int8.self, n: n, rng: &rng)
            try exerciseIntegers(UInt8.self, n: n, rng: &rng)
            try exerciseIntegers(Int16.self, n: n, rng: &rng)
            try exerciseIntegers(UInt16.self, n: n, rng: &rng)
            try exerciseIntegers(Int32.self, n: n, rng: &rng)
            try exerciseIntegers(UInt32.self, n: n, rng: &rng)
            try exerciseIntegers(Int64.self, n: n, rng: &rng)
            try exerciseIntegers(UInt64.self, n: n, rng: &rng)
        }
    }

    // MARK: - Exact boundary values

    /// Every boundary the task calls out, one array each, so the reported index is unambiguous.
    func testExactBoundaries() throws {
        try requireRealGPU()

        func expectRaise<T: ArrowPrimitive>(_ what: String, _ op: String, _ index: Int?,
                                            _ body: () throws -> MetalArray<T>) {
            do {
                _ = try body()
                XCTFail("\(what): expected \(op) to raise")
            } catch let e as ArrowMetalError {
                guard case .overflow(let gotOp, let gotIndex, _) = e else {
                    return XCTFail("\(what): expected an overflow error, got \(e)")
                }
                XCTAssertEqual(gotOp, op, what)
                XCTAssertEqual(gotIndex, index, what)
            } catch {
                XCTFail("\(what): unexpected error \(error)")
            }
        }

        // Int8.max + 1, at a non-zero row so the index is not accidentally right.
        expectRaise("int8 127 + 1", "add_checked", 2) { try MetalArray<Int8>([1, 2, 127, 3]).addChecked(1) }
        expectRaise("int8 -128 - 1", "subtract_checked", 0) { try MetalArray<Int8>([-128]).subtractChecked(1) }
        expectRaise("uint8 255 + 1", "add_checked", 1) { try MetalArray<UInt8>([0, 255]).addChecked(1) }
        expectRaise("uint8 0 - 1", "subtract_checked", 0) { try MetalArray<UInt8>([0, 9]).subtractChecked(1) }
        // Int64.min / -1 is the one division that overflows rather than dividing by zero.
        expectRaise("int64 min / -1", "divide_checked", 1) { try MetalArray<Int64>([4, .min]).divideChecked(-1) }
        expectRaise("int32 / 0", "divide_checked", 0) { try MetalArray<Int32>([1, 2]).divideChecked(0) }
        expectRaise("uint16 / 0", "divide_checked", 0) { try MetalArray<UInt16>([7]).divideChecked(0) }
        expectRaise("int64 min negate", "negate_checked", 3) { try MetalArray<Int64>([1, 2, 3, .min]).negateChecked() }
        expectRaise("uint32 negate", "negate_checked", 1) { try MetalArray<UInt32>([0, 5]).negateChecked() }
        expectRaise("int8 min abs", "abs_checked", 0) { try MetalArray<Int8>([-128, 1]).absChecked() }
        // 1 << 63 on int64: Arrow's precision for a signed type is one less than the bit width.
        expectRaise("int64 1 << 63", "shift_left_checked", 0) { try MetalArray<Int64>([1]).shiftLeftChecked(63) }
        expectRaise("int64 shift by 64", "shift_left_checked", 0) { try MetalArray<Int64>([1]).shiftLeftChecked(64) }
        expectRaise("int64 shift by -1", "shift_right_checked", 0) { try MetalArray<Int64>([1]).shiftRightChecked(-1) }
        expectRaise("uint64 shift by 64", "shift_left_checked", 0) { try MetalArray<UInt64>([1]).shiftLeftChecked(64) }
        expectRaise("int8 2 ^ 10", "power_checked", 0) { try MetalArray<Int8>([2]).powerChecked(10) }
        expectRaise("int8 negative exponent", "power_checked", 0) { try MetalArray<Int8>([2]).powerChecked(-1) }
        expectRaise("float sqrt(-1)", "sqrt_checked", 1) { try MetalArray<Float>([4, -1]).sqrtChecked() }
        expectRaise("float64 sqrt(-1)", "sqrt_checked", 0) { try MetalArray<Double>([-1]).sqrtChecked() }
        expectRaise("float ln(0)", "ln_checked", 1) { try MetalArray<Float>([1, 0]).lnChecked() }
        expectRaise("float64 ln(0)", "ln_checked", 0) { try MetalArray<Double>([0]).lnChecked() }
        expectRaise("float64 ln(-1)", "ln_checked", 0) { try MetalArray<Double>([-1]).lnChecked() }
        expectRaise("float64 -inf ln", "ln_checked", 0) { try MetalArray<Double>([-.infinity]).lnChecked() }
        expectRaise("float log1p(-1)", "log1p_checked", 0) { try MetalArray<Float>([-1]).log1pChecked() }
        expectRaise("float64 log1p(-2)", "log1p_checked", 0) { try MetalArray<Double>([-2]).log1pChecked() }
        expectRaise("float64 logb base 0", "logb_checked", 0) { try MetalArray<Double>([8]).logbChecked(0) }
        expectRaise("float divide by zero", "divide_checked", 0) { try MetalArray<Float>([1, 2]).divideChecked(0) }
        expectRaise("float divide by zero column", "divide_checked", 1) {
            try MetalArray<Float>([1, 2]).divideChecked(try MetalArray<Float>([4, 0]))
        }
        expectRaise("float64 divide by zero", "divide_checked", 0) { try MetalArray<Double>([1]).divideChecked(0) }

        // These are exactly on the boundary and must NOT raise.
        XCTAssertEqual(try MetalArray<Int8>([126]).addChecked(1).toArray(), [127])
        XCTAssertEqual(try MetalArray<Int64>([1]).shiftLeftChecked(62).toArray(), [1 << 62])
        XCTAssertEqual(try MetalArray<UInt64>([1]).shiftLeftChecked(63).toArray(), [1 << 63])
        XCTAssertEqual(try MetalArray<Int64>([.min]).divideChecked(1).toArray(), [.min])
        XCTAssertEqual(try MetalArray<UInt8>([0]).negateChecked().toArray(), [0])
        XCTAssertEqual(try MetalArray<Int8>([-127]).absChecked().toArray(), [127])
        XCTAssertEqual(try MetalArray<Float>([0]).sqrtChecked().toArray(), [0])
        XCTAssertEqual(try MetalArray<Double>([-0.0]).sqrtChecked().toArray(), [-0.0])
    }

    // MARK: - Nulls never raise

    /// A null row holds whatever the builder left in the slot, so a checked op must not look at it. Every
    /// array below would raise if the predicate ran on the null.
    func testNullsNeverRaise() throws {
        try requireRealGPU()
        // Each array holds a value in a null slot that would raise if the predicate looked at it: a
        // MetalArray built from optionals writes 0 into a null slot, so the arrays below put the
        // dangerous value in a *valid* row and the harmless one in the null.
        XCTAssertEqual(try MetalArray<Int8>([nil, 1]).addChecked(1).toArray(), [nil, 2])
        XCTAssertEqual(try MetalArray<Int8>([1, nil]).addChecked(126).toArray(), [127, nil])
        XCTAssertEqual(try MetalArray<Int8>([nil]).negateChecked().toArray(), [nil])
        XCTAssertEqual(try MetalArray<Int32>([1, nil]).divideChecked(try MetalArray<Int32>([1, 0])).toArray(), [1, nil])
        XCTAssertEqual(try MetalArray<Int32>([1, 2]).divideChecked(try MetalArray<Int32>([1, nil])).toArray(), [1, nil])
        XCTAssertEqual(try MetalArray<Float>([1, nil]).sqrtChecked().toArray(), [1, nil])
        // A null contributes the neutral element, so 100 + 0 + 27 stays in range...
        XCTAssertEqual(try MetalArray<Int8>([100, nil, 27]).cumulativeSumChecked().toArray(), [100, nil, 127])
        // ...but the running value carries *across* the null, so 100 + 28 still overflows at row 2.
        XCTAssertThrowsError(try MetalArray<Int8>([100, nil, 28]).cumulativeSumChecked())
    }

    // MARK: - Floats: Arrow's checked float kernels do not raise on inf or NaN

    func testFloatCheckedDoesNotRaiseOnInfinityOrNaN() throws {
        try requireRealGPU()
        // The four arithmetic ops never raise on a float column, whatever the operands.
        let f = try MetalArray<Float>([3.0e38, .nan, .infinity, -.infinity])
        XCTAssertEqual(try f.addChecked(3.0e38).toRawArray()[0], .infinity)
        XCTAssertEqual(try f.multiplyChecked(1.0e38).toRawArray()[0], .infinity)
        XCTAssertEqual(try f.subtractChecked(-3.0e38).toRawArray()[0], .infinity)
        XCTAssertTrue(try f.absChecked().toRawArray()[1].isNaN)
        XCTAssertTrue(try f.negateChecked().toRawArray()[1].isNaN)
        XCTAssertEqual(try f.absChecked().toRawArray()[3], .infinity)

        let d = try MetalArray<Double>([1.0e308, .nan, .infinity, -.infinity])
        XCTAssertEqual(try d.addChecked(1.0e308).toRawArray()[0], .infinity)
        XCTAssertEqual(try d.multiplyChecked(10).toRawArray()[0], .infinity)
        XCTAssertTrue(try d.absChecked().toRawArray()[1].isNaN)
        XCTAssertTrue(try d.negateChecked().toRawArray()[1].isNaN)

        // The domain checks pass NaN and +inf through and raise only on a genuinely negative value, so
        // -inf *does* raise (it is a negative number) while NaN does not.
        let ok = try MetalArray<Float>([4, .nan, .infinity])
        XCTAssertTrue(try ok.sqrtChecked().toRawArray()[1].isNaN)
        XCTAssertEqual(try ok.sqrtChecked().toRawArray()[2], .infinity)
        XCTAssertTrue(try ok.lnChecked().toRawArray()[1].isNaN)
        XCTAssertEqual(try ok.lnChecked().toRawArray()[2], .infinity)
        let okd = try MetalArray<Double>([4, .nan, .infinity])
        XCTAssertTrue(try okd.lnChecked().toRawArray()[1].isNaN)
        XCTAssertEqual(try okd.lnChecked().toRawArray()[2], .infinity)
        XCTAssertTrue(try okd.log1pChecked().toRawArray()[1].isNaN)
        XCTAssertThrowsError(try MetalArray<Float>([-.infinity]).sqrtChecked())
        XCTAssertThrowsError(try MetalArray<Double>([-.infinity]).lnChecked())
    }

    // MARK: - Float domain errors, all sizes

    func testFloatDomainChecksAtEverySize() throws {
        try requireRealGPU()
        var rng = SeededRNG(0xF10A_7)
        for n in Self.sizes {
            for isDouble in [false, true] {
                let label = "\(isDouble ? "float64" : "float32") n=\(n)"
                // Strictly positive values with nulls: nothing may raise, and the values must match.
                var v: [Double?] = (0..<n).map { _ in
                    Double.random(in: 0..<1, using: &rng) < 0.2 ? nil : Double.random(in: 0.001...1000, using: &rng)
                }
                if isDouble {
                    let a = try MetalArray<Double>(v)
                    XCTAssertTrue(Self.identical(try a.sqrtChecked(), try a.sqrt()), "\(label) sqrt")
                    XCTAssertTrue(Self.identical(try a.lnChecked(), try a.ln()), "\(label) ln")
                    XCTAssertTrue(Self.identical(try a.log2Checked(), try a.log2()), "\(label) log2")
                    XCTAssertTrue(Self.identical(try a.log10Checked(), try a.log10()), "\(label) log10")
                    XCTAssertTrue(Self.identical(try a.log1pChecked(), try a.log1p()), "\(label) log1p")
                    XCTAssertTrue(Self.identical(try a.logbChecked(2.0), try a.logb(2.0)), "\(label) logb")
                    XCTAssertTrue(Self.identical(try a.divideChecked(2.0), try a.divide(2.0)), "\(label) divide")
                } else {
                    let a = try MetalArray<Float>(v.map { $0.map(Float.init) })
                    XCTAssertTrue(Self.identical(try a.sqrtChecked(), try a.sqrt()), "\(label) sqrt")
                    XCTAssertTrue(Self.identical(try a.lnChecked(), try a.ln()), "\(label) ln")
                    XCTAssertTrue(Self.identical(try a.log1pChecked(), try a.log1p()), "\(label) log1p")
                    XCTAssertTrue(Self.identical(try a.logbChecked(2.0), try a.logb(2.0)), "\(label) logb")
                }
                guard n > 1 else { continue }
                // One negative in the last row: the reported index must be that row, not an earlier one.
                v[n - 1] = -1
                if isDouble {
                    let a = try MetalArray<Double>(v)
                    expectChecked(label, "sqrt_checked", n - 1, checked: { try a.sqrtChecked() }, unchecked: { try a.sqrt() })
                    expectChecked(label, "ln_checked", n - 1, checked: { try a.lnChecked() }, unchecked: { try a.ln() })
                } else {
                    let a = try MetalArray<Float>(v.map { $0.map(Float.init) })
                    expectChecked(label, "sqrt_checked", n - 1, checked: { try a.sqrtChecked() }, unchecked: { try a.sqrt() })
                    expectChecked(label, "ln_checked", n - 1, checked: { try a.lnChecked() }, unchecked: { try a.ln() })
                }
            }
        }
    }

    // MARK: - Batched mode

    /// Inside a batch the check joins the open command buffer, so the error surfaces from the flush at
    /// the end of `batch { }` rather than from the call that queued it.
    func testCheckedInsideABatchRaisesAtTheSyncPoint() throws {
        try requireRealGPU()
        let ctx = MetalContext.shared
        var queued = false
        XCTAssertThrowsError(try ctx.batch {
            let a = try MetalArray<Int8>([1, 2, 127])
            _ = try a.addChecked(1)
            queued = true          // the call itself returned: the check has not been read yet
            _ = try a.multiply(2)  // more work still encodes fine behind it
        }) { error in
            guard case ArrowMetalError.overflow(let op, let index, _) = error else {
                return XCTFail("expected an overflow error, got \(error)")
            }
            XCTAssertEqual(op, "add_checked")
            XCTAssertEqual(index, 2)
        }
        XCTAssertTrue(queued, "the checked call should return inside a batch and raise at the flush")
        // The context must be usable again afterwards.
        XCTAssertEqual(try MetalArray<Int8>([1]).addChecked(1).toArray(), [2])
    }

    // MARK: - C ABI and error text

    func testErrorTextNamesTheOpTheMessageAndTheRow() throws {
        try requireRealGPU()
        do {
            _ = try MetalArray<Int32>([1, 2, 3]).divideChecked(try MetalArray<Int32>([1, 0, 1]))
            XCTFail("expected a raise")
        } catch let e as ArrowMetalError {
            XCTAssertEqual("\(e)", "divide_checked: divide by zero at index 1")
        }
        do {
            _ = try MetalArray<Int8>([2]).powerChecked(-1)
            XCTFail("expected a raise")
        } catch let e as ArrowMetalError {
            XCTAssertEqual("\(e)", "power_checked: integers to negative integer powers are not allowed at index 0")
        }
        do {
            _ = try MetalArray<Int64>([1]).shiftLeftChecked(63)
            XCTFail("expected a raise")
        } catch let e as ArrowMetalError {
            XCTAssertEqual("\(e)", "shift_left_checked: shift amount must be >= 0 and less than precision of type at index 0")
        }
    }

    /// The empty array is the one case where no kernel runs at all; it must still round-trip.
    func testEmptyArraysNeverRaise() throws {
        try requireRealGPU()
        XCTAssertEqual(try MetalArray<Int8>([]).addChecked(1).length, 0)
        XCTAssertEqual(try MetalArray<Int8>([]).negateChecked().length, 0)
        XCTAssertEqual(try MetalArray<Int8>([]).cumulativeSumChecked().length, 0)
        XCTAssertEqual(try MetalArray<Int8>([]).pairwiseDiffChecked().length, 0)
        XCTAssertEqual(try MetalArray<Double>([]).sqrtChecked().length, 0)
    }
}
