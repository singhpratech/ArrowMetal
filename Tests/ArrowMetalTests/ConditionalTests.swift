import XCTest
@testable import ArrowMetal

/// The logical, float-classification, conditional and hashing kernels added in
/// `LogicalExtra.swift`, `FloatClass.swift`, `Conditional.swift` and `Hash64.swift`, each checked
/// element for element against a Swift CPU oracle.
///
/// Sizes cover the degenerate cases (0, 1), a partial bitmap word (33), several threadgroups (4097)
/// and a size past the multi-block scan paths (1_000_003), with nulls, NaN, ±inf and -0.0 wherever
/// the function can see them.
final class ConditionalTests: XCTestCase {
    static let sizes = [0, 1, 33, 4097, 1_000_003]
    static let smallSizes = [0, 1, 33, 4097]
    static let ratios: [Double] = [0, 0.3, 1.0]

    // MARK: - Helpers

    /// Deterministic null pattern, so a failure is reproducible.
    static func nullAt(_ i: Int, _ ratio: Double, _ salt: Int) -> Bool {
        if ratio <= 0 { return false }
        if ratio >= 1 { return true }
        let h = (UInt64(i) &* 2_654_435_761 &+ UInt64(salt) &* 40_503) % 1000
        return Double(h) / 1000.0 < ratio
    }

    static func column<T: ArrowPrimitive>(_ n: Int, _ ratio: Double, _ salt: Int, _ gen: (Int) -> T) -> [T?] {
        (0..<n).map { nullAt($0, ratio, salt) ? nil : gen($0) }
    }

    /// `MetalBooleanArray` has no `[Bool?]` initialiser, so build one bit by bit.
    static func boolArray(_ vals: [Bool?]) throws -> MetalBooleanArray {
        let n = vals.count
        let b = try MetalBooleanArray.allocate(length: n, withValidity: vals.contains { $0 == nil })
        let vp = b.values.mutableTyped(UInt8.self)
        for (i, v) in vals.enumerated() {
            guard let v else { continue }
            if let bm = b.validity { Bitmap.set(bm.mutableTyped(UInt8.self), i) }
            if v { Bitmap.set(vp, i) }
        }
        b.recomputeNullCount()
        return b
    }

    static func boolPattern(_ n: Int, _ ratio: Double, _ salt: Int, _ gen: (Int) -> Bool) -> [Bool?] {
        (0..<n).map { nullAt($0, ratio, salt) ? nil : gen($0) }
    }

    // MARK: - xor / and_not / and_not_kleene

    func testLogicalExtra() throws {
        try requireRealGPU()
        for n in Self.sizes {
            for ratio in Self.ratios {
                let av = Self.boolPattern(n, ratio, 1) { $0 % 3 == 0 }
                let bv = Self.boolPattern(n, ratio, 2) { $0 % 5 < 2 }
                let a = try Self.boolArray(av), b = try Self.boolArray(bv)
                let tag = "n=\(n) ratio=\(ratio)"

                let xorGot = try a.xor(b).toArray()
                let andNotGot = try a.andNot(b).toArray()
                let kleeneGot = try a.andNotKleene(b).toArray()
                for i in 0..<n {
                    // xor and and_not propagate nulls.
                    let bothValid = av[i] != nil && bv[i] != nil
                    XCTAssertEqual(xorGot[i], bothValid ? (av[i]! != bv[i]!) : nil, "xor \(tag) i=\(i)")
                    XCTAssertEqual(andNotGot[i], bothValid ? (av[i]! && !bv[i]!) : nil, "and_not \(tag) i=\(i)")
                    // and_not_kleene: false wins over a null on either side.
                    let want: Bool?
                    if av[i] == false || bv[i] == true { want = false }
                    else if bothValid { want = av[i]! && !bv[i]! }
                    else { want = nil }
                    XCTAssertEqual(kleeneGot[i], want, "and_not_kleene \(tag) i=\(i) a=\(String(describing: av[i])) b=\(String(describing: bv[i]))")
                }
            }
        }
    }

    /// The truth table Arrow publishes for `and_not_kleene`, spelled out row by row.
    func testAndNotKleeneTruthTable() throws {
        try requireRealGPU()
        let cases: [(Bool?, Bool?, Bool?)] = [
            (true, true, false), (true, false, true), (true, nil, nil),
            (false, true, false), (false, false, false), (false, nil, false),
            (nil, true, false), (nil, false, nil), (nil, nil, nil),
        ]
        let a = try Self.boolArray(cases.map { $0.0 })
        let b = try Self.boolArray(cases.map { $0.1 })
        let got = try a.andNotKleene(b).toArray()
        for (i, c) in cases.enumerated() {
            XCTAssertEqual(got[i], c.2, "and_not_kleene(\(String(describing: c.0)), \(String(describing: c.1)))")
        }
    }

    // MARK: - is_nan / is_finite / is_inf

    func testFloatClassFloat64() throws {
        try requireRealGPU()
        let pool: [Double] = [0.0, -0.0, 1.0, -1.5, .nan, -Double.nan, .infinity, -.infinity,
                              .leastNonzeroMagnitude, .greatestFiniteMagnitude, 1e300, -1e-300]
        for n in Self.sizes {
            for ratio in Self.ratios {
                let vals = Self.column(n, ratio, 4) { pool[$0 % pool.count] }
                let a = try MetalArray<Double>(vals)
                let nanGot = try a.isNan().toArray()
                let finGot = try a.isFinite().toArray()
                let infGot = try a.isInf().toArray()
                for i in 0..<n {
                    guard let v = vals[i] else {
                        XCTAssertNil(nanGot[i]); XCTAssertNil(finGot[i]); XCTAssertNil(infGot[i])
                        continue
                    }
                    XCTAssertEqual(nanGot[i], v.isNaN, "is_nan(\(v)) n=\(n) i=\(i)")
                    XCTAssertEqual(finGot[i], v.isFinite, "is_finite(\(v)) n=\(n) i=\(i)")
                    XCTAssertEqual(infGot[i], v.isInfinite, "is_inf(\(v)) n=\(n) i=\(i)")
                }
            }
        }
    }

    func testFloatClassFloat32() throws {
        try requireRealGPU()
        let pool: [Float] = [0.0, -0.0, 1.0, -1.5, .nan, .infinity, -.infinity,
                             .leastNonzeroMagnitude, .greatestFiniteMagnitude]
        for n in Self.smallSizes {
            let vals = Self.column(n, 0.3, 5) { pool[$0 % pool.count] }
            let a = try MetalArray<Float>(vals)
            let nanGot = try a.isNan().toArray(), finGot = try a.isFinite().toArray(), infGot = try a.isInf().toArray()
            for i in 0..<n {
                guard let v = vals[i] else { XCTAssertNil(nanGot[i]); continue }
                XCTAssertEqual(nanGot[i], v.isNaN, "n=\(n) i=\(i)")
                XCTAssertEqual(finGot[i], v.isFinite, "n=\(n) i=\(i)")
                XCTAssertEqual(infGot[i], v.isInfinite, "n=\(n) i=\(i)")
            }
        }
    }

    /// Arrow defines the three predicates on integers too: everything is finite, nothing is NaN or
    /// infinite, and nulls still propagate.
    func testFloatClassIntegers() throws {
        try requireRealGPU()
        for n in [0, 1, 33, 4097] {
            let vals = Self.column(n, 0.3, 6) { Int32($0 &* 7 &- 3) }
            let a = try MetalArray<Int32>(vals)
            let nanGot = try a.isNan().toArray(), finGot = try a.isFinite().toArray(), infGot = try a.isInf().toArray()
            for i in 0..<n {
                if vals[i] == nil {
                    XCTAssertNil(nanGot[i]); XCTAssertNil(finGot[i]); XCTAssertNil(infGot[i])
                } else {
                    XCTAssertEqual(nanGot[i], false); XCTAssertEqual(finGot[i], true); XCTAssertEqual(infGot[i], false)
                }
            }
        }
    }

    // MARK: - fill_null_forward / fill_null_backward

    static func oracleFill<T>(_ v: [T?], backward: Bool) -> [T?] {
        var out = v
        if backward {
            var carry: T? = nil
            for i in stride(from: v.count - 1, through: 0, by: -1) {
                if let x = v[i] { carry = x } else { out[i] = carry }
            }
        } else {
            var carry: T? = nil
            for i in 0..<v.count {
                if let x = v[i] { carry = x } else { out[i] = carry }
            }
        }
        return out
    }

    func fillMatrix<T: ArrowPrimitive>(_ gen: @escaping (Int) -> T) throws {
        for n in Self.sizes {
            for ratio in Self.ratios {
                let vals = Self.column(n, ratio, 11, gen)
                let a = try MetalArray<T>(vals)
                for backward in [false, true] {
                    let got = backward ? try a.fillNullBackward().toArray() : try a.fillNullForward().toArray()
                    let want = Self.oracleFill(vals, backward: backward)
                    XCTAssertEqual(got.count, n)
                    for i in 0..<n {
                        XCTAssertEqual(got[i], want[i], "\(T.self) fill \(backward ? "backward" : "forward") n=\(n) ratio=\(ratio) i=\(i)")
                    }
                }
            }
        }
    }

    func testFillNullInt64() throws { try requireRealGPU(); try fillMatrix { Int64($0 &* 31 &- 7) } }
    func testFillNullInt8() throws { try requireRealGPU(); try fillMatrix { Int8(truncatingIfNeeded: $0 &* 13) } }
    func testFillNullUInt32() throws { try requireRealGPU(); try fillMatrix { UInt32($0 &* 3) } }
    func testFillNullFloat32() throws { try requireRealGPU(); try fillMatrix { Float($0) * 0.5 } }

    /// Float64 travels as a raw 64-bit move here, so NaN and -0.0 must come through bit-identically.
    func testFillNullFloat64() throws {
        try requireRealGPU()
        let pool: [Double] = [1.0, -0.0, .nan, .infinity, -.infinity, 2.5]
        for n in Self.smallSizes {
            for ratio in Self.ratios {
                let vals = Self.column(n, ratio, 12) { pool[$0 % pool.count] }
                let a = try MetalArray<Double>(vals)
                for backward in [false, true] {
                    let got = backward ? try a.fillNullBackward().toArray() : try a.fillNullForward().toArray()
                    let want = Self.oracleFill(vals, backward: backward)
                    for i in 0..<n {
                        if let w = want[i] {
                            let g = try XCTUnwrap(got[i], "n=\(n) i=\(i)")
                            XCTAssertEqual(g.bitPattern, w.bitPattern, "float64 fill n=\(n) i=\(i)")
                        } else {
                            XCTAssertNil(got[i], "n=\(n) i=\(i)")
                        }
                    }
                }
            }
        }
    }

    func testFillNullBoolean() throws {
        try requireRealGPU()
        for n in Self.smallSizes {
            for ratio in Self.ratios {
                let vals = Self.boolPattern(n, ratio, 13) { $0 % 4 < 2 }
                let a = try Self.boolArray(vals)
                for backward in [false, true] {
                    let got = backward ? try a.fillNullBackward().toArray() : try a.fillNullForward().toArray()
                    let want = Self.oracleFill(vals, backward: backward)
                    for i in 0..<n { XCTAssertEqual(got[i], want[i], "bool fill n=\(n) i=\(i)") }
                }
            }
        }
    }

    /// An array with no validity bitmap comes back unchanged, and a fully null one stays fully null.
    func testFillNullEdges() throws {
        try requireRealGPU()
        let noNulls = try MetalArray<Int32>([1, 2, 3])
        XCTAssertEqual(try noNulls.fillNullForward().toArray(), [1, 2, 3])
        XCTAssertEqual(try noNulls.fillNullBackward().toArray(), [1, 2, 3])
        let allNull = try MetalArray<Int32>([nil, nil, nil])
        XCTAssertEqual(try allNull.fillNullForward().toArray(), [nil, nil, nil])
        XCTAssertEqual(try allNull.fillNullBackward().nullCount, 3)
    }

    // MARK: - case_when

    func testCaseWhen() throws {
        try requireRealGPU()
        for n in Self.sizes {
            for ratio in Self.ratios {
                let c1 = Self.boolPattern(n, ratio, 21) { $0 % 3 == 0 }
                let c2 = Self.boolPattern(n, ratio, 22) { $0 % 5 < 2 }
                let v1 = Self.column(n, ratio, 23) { Int64($0) }
                let v2 = Self.column(n, ratio, 24) { Int64($0) * 100 }
                let dv = Self.column(n, ratio, 25) { Int64(-$0) }
                let conds = [try Self.boolArray(c1), try Self.boolArray(c2)]
                let values = [try MetalArray<Int64>(v1), try MetalArray<Int64>(v2)]
                let def = try MetalArray<Int64>(dv)

                // With a default column.
                let got = try MetalArray<Int64>.caseWhen(conds: conds, values: values, else: def).toArray()
                // Without one: no match gives null.
                let got2 = try MetalArray<Int64>.caseWhen(conds: conds, values: values).toArray()
                for i in 0..<n {
                    // Arrow reads a null condition as false.
                    let want: Int64?
                    let want2: Int64?
                    if c1[i] == true { want = v1[i]; want2 = v1[i] }
                    else if c2[i] == true { want = v2[i]; want2 = v2[i] }
                    else { want = dv[i]; want2 = nil }
                    XCTAssertEqual(got[i], want, "case_when n=\(n) ratio=\(ratio) i=\(i)")
                    XCTAssertEqual(got2[i], want2, "case_when(no default) n=\(n) ratio=\(ratio) i=\(i)")
                }
            }
        }
    }

    func testCaseWhenErrors() throws {
        let c = try Self.boolArray([true, false])
        let v = try MetalArray<Int64>([1, 2])
        XCTAssertThrowsError(try MetalArray<Int64>.caseWhen(conds: [c, c], values: [v]))
        XCTAssertThrowsError(try MetalArray<Int64>.caseWhen(conds: [c], values: [try MetalArray<Int64>([1, 2, 3])]))
        XCTAssertThrowsError(try MetalArray<Int64>.caseWhen(conds: [], values: []))
    }

    // MARK: - choose

    func testChoose() throws {
        try requireRealGPU()
        for n in Self.sizes {
            for ratio in Self.ratios {
                let k = 3
                let idx = Self.column(n, ratio, 31) { Int32($0 % k) }
                let cols = (0..<k).map { j in Self.column(n, ratio, 32 + j) { Int64($0 &* 10 &+ j) } }
                let indices = try MetalArray<Int32>(idx)
                let arrays = try cols.map { try MetalArray<Int64>($0) }
                let got = try MetalArray<Int64>.choose(indices, arrays).toArray()
                for i in 0..<n {
                    let want: Int64? = idx[i].map { cols[Int($0)][i] } ?? nil
                    XCTAssertEqual(got[i], want, "choose n=\(n) ratio=\(ratio) i=\(i)")
                }
            }
        }
    }

    func testChooseInt64IndicesAndErrors() throws {
        try requireRealGPU()
        let a = try MetalArray<Int64>([10, 20, 30])
        let b = try MetalArray<Int64>([40, 50, 60])
        let idx = try MetalArray<Int64>([0, 1, 0])
        XCTAssertEqual(try MetalArray<Int64>.choose(idx, [a, b]).toArray(), [10, 50, 30])
        // Out of range, both directions, as in Arrow.
        XCTAssertThrowsError(try MetalArray<Int64>.choose(try MetalArray<Int64>([0, 2, 0]), [a, b]))
        XCTAssertThrowsError(try MetalArray<Int64>.choose(try MetalArray<Int64>([0, -1, 0]), [a, b]))
        XCTAssertThrowsError(try MetalArray<Int64>.choose(idx, []))
        // A null index that is out of range in its value slot must not raise: nulls are skipped.
        XCTAssertEqual(try MetalArray<Int64>.choose(try MetalArray<Int64>([0, nil, 1]), [a, b]).toArray(),
                       [10, nil, 60])
    }

    // MARK: - replace_with_mask

    func testReplaceWithMask() throws {
        try requireRealGPU()
        for n in Self.sizes {
            for ratio in Self.ratios {
                let vals = Self.column(n, ratio, 41) { Int64($0) }
                let mask = Self.boolPattern(n, ratio, 42) { $0 % 4 == 1 }
                let trueCount = mask.filter { $0 == true }.count
                let repl = (0..<trueCount).map { $0 % 5 == 3 ? nil : Int64(-1000 - $0) }
                let a = try MetalArray<Int64>(vals)
                let m = try Self.boolArray(mask)
                let r = try MetalArray<Int64>(repl)
                let got = try a.replaceWithMask(m, r).toArray()

                var next = 0
                for i in 0..<n {
                    let want: Int64?
                    if mask[i] == nil { want = nil }
                    else if mask[i] == true { want = repl[next]; next += 1 }
                    else { want = vals[i] }
                    XCTAssertEqual(got[i], want, "replace_with_mask n=\(n) ratio=\(ratio) i=\(i)")
                }
            }
        }
    }

    func testReplaceWithMaskErrors() throws {
        try requireRealGPU()
        let a = try MetalArray<Int64>([1, 2, 3, 4])
        let m = try Self.boolArray([true, false, true, nil])
        // Two valid trues need at least two replacements.
        XCTAssertThrowsError(try a.replaceWithMask(m, try MetalArray<Int64>([9])))
        XCTAssertEqual(try a.replaceWithMask(m, try MetalArray<Int64>([9, 8])).toArray(), [9, 2, 8, nil])
        // A surplus of replacements is ignored, as pyarrow does.
        XCTAssertEqual(try a.replaceWithMask(m, try MetalArray<Int64>([9, 8, 7])).toArray(), [9, 2, 8, nil])
        XCTAssertThrowsError(try a.replaceWithMask(try Self.boolArray([true, false]), try MetalArray<Int64>([9])))
    }

    func testReplaceWithMaskBoolean() throws {
        try requireRealGPU()
        let a = try Self.boolArray([true, true, false, false])
        let m = try Self.boolArray([false, true, true, nil])
        let r = try Self.boolArray([false, true])
        XCTAssertEqual(try a.replaceWithMask(m, r).toArray(), [true, false, true, nil])
    }

    // MARK: - indices_nonzero

    func testIndicesNonzero() throws {
        try requireRealGPU()
        for n in Self.sizes {
            for ratio in Self.ratios {
                let vals = Self.column(n, ratio, 51) { Int32($0 % 3 == 0 ? 0 : $0) }
                let a = try MetalArray<Int32>(vals)
                let got = try a.indicesNonzero().toArray()
                let want = (0..<n).compactMap { i -> UInt64? in
                    guard let v = vals[i], v != 0 else { return nil }
                    return UInt64(i)
                }
                XCTAssertEqual(got.map { $0! }, want, "indices_nonzero n=\(n) ratio=\(ratio)")
            }
        }
    }

    /// -0.0 is zero, every NaN is non-zero, ±inf is non-zero.
    func testIndicesNonzeroFloat() throws {
        try requireRealGPU()
        let vals: [Double?] = [0.0, -0.0, 1.0, .nan, -.infinity, nil, 0.0, 2.0]
        let got = try MetalArray<Double>(vals).indicesNonzero().toArray().map { $0! }
        XCTAssertEqual(got, [2, 3, 4, 7])
        let f: [Float?] = [0.0, -0.0, .nan, 3.0, nil]
        XCTAssertEqual(try MetalArray<Float>(f).indicesNonzero().toArray().map { $0! }, [2, 3])
    }

    func testIndicesNonzeroBoolean() throws {
        try requireRealGPU()
        let b = try Self.boolArray([true, false, nil, true, false, true])
        XCTAssertEqual(try b.indicesNonzero().toArray().map { $0! }, [0, 3, 5])
    }

    // MARK: - hash64

    /// The MurmurHash3 finaliser, on the host, over the same normalisation the kernel uses.
    static func hostHash64(_ w: UInt64) -> UInt64 {
        var k = w ^ 0x9E37_79B9_7F4A_7C15
        k ^= k >> 33
        k = k &* 0xFF51_AFD7_ED55_8CCD
        k ^= k >> 33
        k = k &* 0xC4CE_B9FE_1A85_EC53
        k ^= k >> 33
        return k
    }

    func hashMatrix<T: ArrowPrimitive>(_ gen: @escaping (Int) -> T, word: @escaping (T) -> UInt64) throws {
        for n in Self.smallSizes {
            for ratio in Self.ratios {
                let vals = Self.column(n, ratio, 61, gen)
                let got = try MetalArray<T>(vals).hash64().toArray()
                for i in 0..<n {
                    guard let v = vals[i] else { XCTAssertNil(got[i], "\(T.self) n=\(n) i=\(i)"); continue }
                    XCTAssertEqual(got[i], Self.hostHash64(word(v)), "hash64(\(v)) \(T.self) n=\(n) i=\(i)")
                }
            }
        }
    }

    func testHash64Integers() throws {
        try requireRealGPU()
        try hashMatrix({ Int8(truncatingIfNeeded: $0 &* 37) }, word: { UInt64(UInt8(bitPattern: $0)) })
        try hashMatrix({ UInt8(truncatingIfNeeded: $0 &* 11) }, word: { UInt64($0) })
        try hashMatrix({ Int16(truncatingIfNeeded: $0 &* 501) }, word: { UInt64(UInt16(bitPattern: $0)) })
        try hashMatrix({ UInt16(truncatingIfNeeded: $0 &* 7) }, word: { UInt64($0) })
        try hashMatrix({ Int32(truncatingIfNeeded: $0 &* 2_654_435 &- 17) }, word: { UInt64(UInt32(bitPattern: $0)) })
        try hashMatrix({ UInt32(truncatingIfNeeded: $0 &* 2_654_435_761) }, word: { UInt64($0) })
        try hashMatrix({ Int64($0) &* -6_364_136_223_846_793_005 }, word: { UInt64(bitPattern: $0) })
        try hashMatrix({ UInt64($0) &* 11_400_714_819_323_198_485 }, word: { $0 })
    }

    func testHash64Floats() throws {
        try requireRealGPU()
        let dpool: [Double] = [0.0, -0.0, 1.0, -1.0, .nan, .infinity, -.infinity, 3.25, 1e300]
        try hashMatrix({ dpool[$0 % dpool.count] }, word: { v in
            if v == 0 { return 0 }                                  // -0.0 hashes as +0.0
            if v.isNaN { return 0x7FF8_0000_0000_0000 }             // every NaN is one value
            return v.bitPattern
        })
        let fpool: [Float] = [0.0, -0.0, 1.0, -1.0, .nan, .infinity, -.infinity, 3.25]
        try hashMatrix({ fpool[$0 % fpool.count] }, word: { v in
            if v == 0 { return 0 }
            if v.isNaN { return 0x7FC0_0000 }
            return UInt64(v.bitPattern)
        })
    }

    func testHash64Boolean() throws {
        try requireRealGPU()
        for n in Self.smallSizes {
            let vals = Self.boolPattern(n, 0.3, 62) { $0 % 3 == 0 }
            let got = try Self.boolArray(vals).hash64().toArray()
            for i in 0..<n {
                guard let v = vals[i] else { XCTAssertNil(got[i]); continue }
                XCTAssertEqual(got[i], Self.hostHash64(v ? 1 : 0), "n=\(n) i=\(i)")
            }
        }
    }

    /// Equal values hash equal (that is the property a hash join needs), distinct values collide
    /// rarely, and a slice of a column hashes the same as the column.
    func testHash64Properties() throws {
        try requireRealGPU()
        let n = 1_000_003
        let a = try MetalArray<Int64>((0..<n).map { Int64($0) })
        let h = try a.hash64().toRawArray()
        XCTAssertEqual(Set(h).count, n, "hash64 collided on \(n) distinct int64 values")
        // Nulls hash to 0 in the values buffer and stay null.
        let withNulls = try MetalArray<Int64>([1, nil, 3])
        let hn = try withNulls.hash64()
        XCTAssertNil(hn.toArray()[1])
        XCTAssertEqual(hn.toRawArray()[1], 0)
        // A slice hashes identically, aligned and unaligned.
        for off in [32, 5] {
            let s = try a.slice(offset: off, length: 1000)
            let hs = try s.hash64().toRawArray()
            for i in 0..<1000 { XCTAssertEqual(hs[i], h[i + off], "slice+\(off) i=\(i)") }
        }
        // Both zeros and two different NaN payloads hash alike.
        let zeros = try MetalArray<Double>([0.0, -0.0])
        let hz = try zeros.hash64().toRawArray()
        XCTAssertEqual(hz[0], hz[1])
        let nans = try MetalArray<Double>([Double(bitPattern: 0x7FF8_0000_0000_0001),
                                           Double(bitPattern: 0xFFF8_0000_0000_0000)])
        let hnan = try nans.hash64().toRawArray()
        XCTAssertEqual(hnan[0], hnan[1])
    }
}
