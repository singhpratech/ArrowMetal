import XCTest
@testable import ArrowMetal

/// GPU structural, conditional and set-lookup functions checked element-for-element against a CPU
/// oracle: `is_null`, `is_valid`, `fill_null`, `drop_null`, `if_else`, `coalesce`, `is_in`, `index_in`,
/// `and_kleene` and `or_kleene`.
final class StructuralTests: XCTestCase {
    /// 0 and 1 are the degenerate cases, 31/32/33 straddle a bitmap word, 4097 crosses several
    /// threadgroups and 300_003 is large enough to exercise the multi-block paths.
    static let sizes = [0, 1, 31, 32, 33, 4097, 300_003]
    static let ratios: [Double] = [0, 0.3, 1.0]

    // MARK: - Oracle inputs

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

    // MARK: - The per-type matrix

    /// Runs every structural function over one element type across all sizes, null ratios and both
    /// slice shapes (word aligned, so zero copy, and unaligned, so materialised).
    func matrix<T: ArrowPrimitive>(_ gen: @escaping (Int) -> T, other: @escaping (Int) -> T,
                                   fill: T, fill2: T, set: [T]) throws {
        for n in Self.sizes {
            for ratio in Self.ratios {
                let av = Self.column(n, ratio, 1, gen)
                let bv = Self.column(n, ratio, 2, other)
                let cv: [Bool?] = (0..<n).map { Self.nullAt($0, ratio, 3) ? nil : ($0 % 3 == 0) }
                let a = try MetalArray<T>(av)
                let b = try MetalArray<T>(bv)
                let full = try MetalArray<T>((0..<n).map(other))          // never null
                let cond = try Self.boolArray(cv)
                let tag = "\(T.self) n=\(n) ratio=\(ratio)"

                try check(a, av, b, bv, full, cond, cv, fill: fill, fill2: fill2, set: set, tag: tag)

                guard n >= 33, n <= 4097 else { continue }
                // Word-aligned slice (shares the parent buffers) and an unaligned one (copied).
                for off in [32, 5] {
                    let len = n - off
                    let sa = try a.slice(offset: off, length: len)
                    let sb = try b.slice(offset: off, length: len)
                    let sf = try full.slice(offset: off, length: len)
                    let sc = try cond.slice(offset: off, length: len)
                    try check(sa, Array(av[off...]), sb, Array(bv[off...]), sf, sc, Array(cv[off...]),
                              fill: fill, fill2: fill2, set: set, tag: tag + " slice+\(off)")
                }
            }
        }
    }

    private func check<T: ArrowPrimitive>(_ a: MetalArray<T>, _ av: [T?],
                                          _ b: MetalArray<T>, _ bv: [T?],
                                          _ full: MetalArray<T>,
                                          _ cond: MetalBooleanArray, _ cv: [Bool?],
                                          fill: T, fill2: T, set: [T], tag: String) throws {
        let n = av.count
        XCTAssertEqual(try a.isNull().toArray(), av.map { Optional($0 == nil) }, "isNull \(tag)")
        XCTAssertEqual(try a.isValid().toArray(), av.map { Optional($0 != nil) }, "isValid \(tag)")
        XCTAssertEqual(try a.isNull().nullCount, 0, "isNull has no nulls \(tag)")
        XCTAssertEqual(try a.isValid().nullCount, 0, "isValid has no nulls \(tag)")

        let filled = try a.fillingNull(fill)
        XCTAssertEqual(filled.toArray(), av.map { Optional($0 ?? fill) }, "fillNull \(tag)")
        XCTAssertEqual(filled.nullCount, 0, "fillNull nullCount \(tag)")

        let dropped = try a.dropNull()
        XCTAssertEqual(dropped.toArray(), av.compactMap { $0 }.map { Optional($0) }, "dropNull \(tag)")
        XCTAssertEqual(dropped.nullCount, 0, "dropNull nullCount \(tag)")

        // if_else: a null condition yields a null output; otherwise the chosen branch, nulls and all.
        let pick: ([T?], [T?]) -> [T?] = { l, r in (0..<n).map { i in cv[i].map { $0 ? l[i] : r[i] } ?? nil } }
        XCTAssertEqual(try MetalArray<T>.ifElse(cond, a, b).toArray(), pick(av, bv), "ifElse array/array \(tag)")
        XCTAssertEqual(try MetalArray<T>.ifElse(cond, a, fill).toArray(),
                       pick(av, [T?](repeating: fill, count: n)), "ifElse array/scalar \(tag)")
        XCTAssertEqual(try MetalArray<T>.ifElse(cond, fill, b).toArray(),
                       pick([T?](repeating: fill, count: n), bv), "ifElse scalar/array \(tag)")
        XCTAssertEqual(try MetalArray<T>.ifElse(cond, fill, fill2).toArray(),
                       pick([T?](repeating: fill, count: n), [T?](repeating: fill2, count: n)), "ifElse scalar/scalar \(tag)")
        XCTAssertEqual(try cond.ifElse(a, b).toArray(), pick(av, bv), "cond.ifElse \(tag)")

        // coalesce: first non-null across the inputs.
        XCTAssertEqual(try MetalArray<T>.coalesce([a]).toArray(), av, "coalesce one \(tag)")
        XCTAssertEqual(try MetalArray<T>.coalesce([a, b]).toArray(),
                       (0..<n).map { av[$0] ?? bv[$0] }, "coalesce two \(tag)")
        let fv = full.toArray()
        let three = try MetalArray<T>.coalesce([a, b, full])
        XCTAssertEqual(three.toArray(), (0..<n).map { av[$0] ?? bv[$0] ?? fv[$0] }, "coalesce three \(tag)")
        XCTAssertEqual(three.nullCount, 0, "coalesce three nullCount \(tag)")

        // is_in / index_in against a host set and against a Metal set.
        let setArray = try MetalArray<T>(set, context: a.context)
        let inExpected: [Bool?] = av.map { v in Optional(v.map { set.contains($0) } ?? false) }
        XCTAssertEqual(try a.isIn(set).toArray(), inExpected, "isIn [T] \(tag)")
        XCTAssertEqual(try a.isIn(setArray).toArray(), inExpected, "isIn array \(tag)")
        let idxExpected: [Int32?] = av.map { v in v.flatMap { x in set.firstIndex(of: x).map { Int32($0) } } }
        XCTAssertEqual(try a.indexIn(set).toArray(), idxExpected, "indexIn [T] \(tag)")
        XCTAssertEqual(try a.indexIn(setArray).toArray(), idxExpected, "indexIn array \(tag)")
    }

    // MARK: - One test per Arrow primitive type

    func testInt8() throws {
        try requireRealGPU()
        try matrix({ Int8(truncatingIfNeeded: $0 % 61 - 30) }, other: { Int8(truncatingIfNeeded: $0 % 11) },
                   fill: -7, fill2: 21, set: [-30, 0, 7, 7, 29])
    }
    func testUInt8() throws {
        try requireRealGPU()
        try matrix({ UInt8(truncatingIfNeeded: $0 % 61) }, other: { UInt8(truncatingIfNeeded: $0 % 11) },
                   fill: 200, fill2: 3, set: [0, 5, 5, 60, 250])
    }
    func testInt16() throws {
        try requireRealGPU()
        try matrix({ Int16(truncatingIfNeeded: $0 % 97 - 40) }, other: { Int16(truncatingIfNeeded: $0 % 13) },
                   fill: -1000, fill2: 12, set: [-40, 0, 13, 13, 56])
    }
    func testUInt16() throws {
        try requireRealGPU()
        try matrix({ UInt16(truncatingIfNeeded: $0 % 97) }, other: { UInt16(truncatingIfNeeded: $0 % 13) },
                   fill: 60000, fill2: 4, set: [0, 9, 9, 96, 40000])
    }
    func testInt32() throws {
        try requireRealGPU()
        try matrix({ Int32($0 % 211) - 100 }, other: { Int32($0 % 17) },
                   fill: -99999, fill2: 5, set: [-100, 0, 33, 33, 110])
    }
    func testUInt32() throws {
        try requireRealGPU()
        try matrix({ UInt32($0 % 211) }, other: { UInt32($0 % 17) },
                   fill: 4_000_000_000, fill2: 6, set: [0, 33, 33, 210, 999_999])
    }
    func testInt64() throws {
        try requireRealGPU()
        try matrix({ Int64($0 % 211) - 100 }, other: { Int64($0 % 17) },
                   fill: -9_000_000_000, fill2: 7, set: [-100, 0, 33, 33, 110])
    }
    func testUInt64() throws {
        try requireRealGPU()
        try matrix({ UInt64($0 % 211) }, other: { UInt64($0 % 17) },
                   fill: 18_000_000_000_000_000_000, fill2: 8, set: [0, 33, 33, 210, 12_000_000_000])
    }
    func testFloat32() throws {
        try requireRealGPU()
        try matrix({ Float($0 % 211) * 0.25 - 25 }, other: { Float($0 % 17) * 0.5 },
                   fill: -1.5, fill2: 9.75, set: [-25, 0, 8.25, 8.25, 27.5])
    }
    func testFloat64() throws {
        try requireRealGPU()
        try matrix({ Double($0 % 211) * 0.25 - 25 }, other: { Double($0 % 17) * 0.5 },
                   fill: -1.5, fill2: 9.75, set: [-25, 0, 8.25, 8.25, 27.5])
    }

    // MARK: - Floating point corner cases

    func testFloatNaNAndSignedZero() throws {
        try requireRealGPU()
        // Set lookup uses Arrow value equality (as `unique()`): all NaNs are one value, -0 == 0.
        let f = try MetalArray<Float>([.nan, -0.0, 0.0, 1.5, nil, -.infinity, .infinity])
        XCTAssertEqual(try f.isIn([Float.nan, 0.0]).toArray(),
                       [true, true, true, false, false, false, false])
        XCTAssertEqual(try f.isIn([Float]()).toArray(), [Bool?](repeating: false, count: 7))
        XCTAssertEqual(try f.isIn([-.infinity, .infinity]).toArray(),
                       [false, false, false, false, false, true, true])
        XCTAssertEqual(try f.indexIn([Float(1.5), -0.0]).toArray(), [nil, 1, 1, 0, nil, nil, nil])

        let d = try MetalArray<Double>([.nan, -0.0, 0.0, 1.5, nil, -.infinity, .infinity])
        XCTAssertEqual(try d.isIn([Double.nan, 0.0]).toArray(),
                       [true, true, true, false, false, false, false])
        XCTAssertEqual(try d.indexIn([Double(1.5), -0.0]).toArray(), [nil, 1, 1, 0, nil, nil, nil])

        // fill_null and if_else move the bits through unchanged, NaN included.
        let filled = try d.fillingNull(Double.nan)
        XCTAssertTrue(filled[4]!.isNaN)
        XCTAssertEqual(filled.nullCount, 0)
        XCTAssertEqual(filled[1]!.sign, .minus, "-0.0 survives fill_null")
    }

    // MARK: - Boolean columns

    func testBooleanStructural() throws {
        try requireRealGPU()
        for n in Self.sizes {
            for ratio in Self.ratios {
                let vals: [Bool?] = (0..<n).map { Self.nullAt($0, ratio, 4) ? nil : ($0 % 5 < 2) }
                let other: [Bool?] = (0..<n).map { Self.nullAt($0, ratio, 5) ? nil : ($0 % 7 < 4) }
                let cv: [Bool?] = (0..<n).map { Self.nullAt($0, ratio, 6) ? nil : ($0 % 3 == 0) }
                let a = try Self.boolArray(vals), b = try Self.boolArray(other), cond = try Self.boolArray(cv)
                let tag = "bool n=\(n) ratio=\(ratio)"
                XCTAssertEqual(try a.isNull().toArray(), vals.map { Optional($0 == nil) }, "isNull \(tag)")
                XCTAssertEqual(try a.isValid().toArray(), vals.map { Optional($0 != nil) }, "isValid \(tag)")
                XCTAssertEqual(try a.fillingNull(true).toArray(), vals.map { Optional($0 ?? true) }, "fillNull true \(tag)")
                XCTAssertEqual(try a.fillingNull(false).toArray(), vals.map { Optional($0 ?? false) }, "fillNull false \(tag)")
                XCTAssertEqual(try a.dropNull().toArray(), vals.compactMap { $0 }.map { Optional($0) }, "dropNull \(tag)")
                let expected: [Bool?] = (0..<n).map { i in cv[i].map { $0 ? vals[i] : other[i] } ?? nil }
                XCTAssertEqual(try MetalBooleanArray.ifElse(cond, a, b).toArray(), expected, "ifElse \(tag)")
                XCTAssertEqual(try cond.ifElse(a, b).toArray(), expected, "cond.ifElse \(tag)")
                XCTAssertEqual(try MetalBooleanArray.ifElse(cond, true, false).toArray(),
                               (0..<n).map { i in cv[i] }, "ifElse scalars \(tag)")
            }
        }
    }

    // MARK: - Kleene logic

    func testKleeneTruthTableExhaustive() throws {
        try requireRealGPU()
        let states: [Bool?] = [true, false, nil]
        var l: [Bool?] = [], r: [Bool?] = []
        for x in states { for y in states { l.append(x); r.append(y) } }
        let a = try Self.boolArray(l), b = try Self.boolArray(r)

        func andK(_ x: Bool?, _ y: Bool?) -> Bool? {
            if x == false || y == false { return false }
            guard let x, let y else { return nil }
            return x && y
        }
        func orK(_ x: Bool?, _ y: Bool?) -> Bool? {
            if x == true || y == true { return true }
            guard let x, let y else { return nil }
            return x || y
        }
        XCTAssertEqual(try a.andKleene(b).toArray(), (0..<l.count).map { andK(l[$0], r[$0]) })
        XCTAssertEqual(try a.orKleene(b).toArray(), (0..<l.count).map { orK(l[$0], r[$0]) })
        // Arrow's documented asymmetry with plain `and`/`or`, which propagate nulls.
        XCTAssertEqual(try a.and(b).toArray()[2], nil)         // true AND null
        XCTAssertEqual(try a.andKleene(b).toArray()[5], false) // false AND null
        XCTAssertEqual(try a.orKleene(b).toArray()[2], true)   // true OR null
        XCTAssertNil(try a.orKleene(b).toArray()[8])           // null OR null

        // Null counts and the null-free fast path.
        XCTAssertEqual(try a.andKleene(b).nullCount, (0..<l.count).filter { andK(l[$0], r[$0]) == nil }.count)
        XCTAssertEqual(try a.orKleene(b).nullCount, (0..<l.count).filter { orK(l[$0], r[$0]) == nil }.count)
    }

    func testKleeneAcrossSizes() throws {
        try requireRealGPU()
        for n in Self.sizes {
            for ratio in Self.ratios {
                let lv: [Bool?] = (0..<n).map { Self.nullAt($0, ratio, 7) ? nil : ($0 % 2 == 0) }
                let rv: [Bool?] = (0..<n).map { Self.nullAt($0, ratio, 8) ? nil : ($0 % 3 == 0) }
                let a = try Self.boolArray(lv), b = try Self.boolArray(rv)
                let tag = "n=\(n) ratio=\(ratio)"
                let expectedAnd: [Bool?] = (0..<n).map { i in
                    if lv[i] == false || rv[i] == false { return false }
                    guard let x = lv[i], let y = rv[i] else { return nil }
                    return x && y
                }
                let expectedOr: [Bool?] = (0..<n).map { i in
                    if lv[i] == true || rv[i] == true { return true }
                    guard let x = lv[i], let y = rv[i] else { return nil }
                    return x || y
                }
                XCTAssertEqual(try a.andKleene(b).toArray(), expectedAnd, "andKleene \(tag)")
                XCTAssertEqual(try a.orKleene(b).toArray(), expectedOr, "orKleene \(tag)")
            }
        }
        // With no nulls anywhere, Kleene logic is ordinary boolean logic.
        let x = try MetalBooleanArray((0..<100).map { $0 % 2 == 0 })
        let y = try MetalBooleanArray((0..<100).map { $0 % 3 == 0 })
        XCTAssertEqual(try x.andKleene(y).toArray(), try x.and(y).toArray())
        XCTAssertEqual(try x.orKleene(y).toArray(), try x.or(y).toArray())
    }

    // MARK: - Set lookup corner cases

    func testIsInEmptySetNullsAllHitAndNoHit() throws {
        try requireRealGPU()
        let a = try MetalArray<Int32>([1, 2, nil, 3, 2, 1])

        // Empty set: nothing matches, and index_in is null everywhere.
        XCTAssertEqual(try a.isIn([Int32]()).toArray(), [false, false, false, false, false, false])
        XCTAssertEqual(try a.isIn([Int32]()).nullCount, 0)
        XCTAssertEqual(try a.indexIn([Int32]()).toArray(), [nil, nil, nil, nil, nil, nil])

        // A set that itself contains nulls: the nulls are ignored, and a null element never matches.
        let setWithNulls = try MetalArray<Int32>([nil, 2, nil, 3])
        XCTAssertEqual(try a.isIn(setWithNulls).toArray(), [false, true, false, true, true, false])
        XCTAssertEqual(try a.isIn(setWithNulls).nullCount, 0)
        XCTAssertEqual(try a.indexIn(setWithNulls).toArray(), [nil, 1, nil, 3, 1, nil])

        // An all-null set behaves like an empty one.
        let allNull = try MetalArray<Int32>([Int32?](repeating: nil, count: 4))
        XCTAssertEqual(try a.isIn(allNull).toArray(), [false, false, false, false, false, false])
        XCTAssertEqual(try a.indexIn(allNull).toArray(), [nil, nil, nil, nil, nil, nil])

        // All hit and no hit.
        XCTAssertEqual(try a.isIn([1, 2, 3]).toArray(), [true, true, false, true, true, true])
        XCTAssertEqual(try a.isIn([7, 8, 9]).toArray(), [false, false, false, false, false, false])
        XCTAssertEqual(try a.indexIn([7, 8, 9]).toArray(), [nil, nil, nil, nil, nil, nil])

        // index_in reports the first occurrence in the caller's set, unsorted and with duplicates.
        XCTAssertEqual(try a.indexIn([3, 2, 2, 1, 3]).toArray(), [3, 1, nil, 0, 1, 3])

        // Empty input array.
        let empty = try MetalArray<Int32>([Int32]())
        XCTAssertEqual(try empty.isIn([1, 2]).length, 0)
        XCTAssertEqual(try empty.indexIn([1, 2]).length, 0)
    }

    // MARK: - Batched execution

    /// Everything a batch produces, flattened to host arrays so batched and unbatched runs compare directly.
    struct StructuralResults: Equatable {
        var isNull: [Bool?] = [], isValid: [Bool?] = [], isIn: [Bool?] = [], andK: [Bool?] = [], orK: [Bool?] = []
        var fill: [Int64?] = [], drop: [Int64?] = [], ifElse: [Int64?] = [], coalesce: [Int64?] = []
        var indexIn: [Int32?] = []
    }

    func testBatchedMatchesUnbatched() throws {
        try requireRealGPU()
        for n in [0, 1, 33, 4097, 300_003] {
            let av = Self.column(n, 0.3, 11) { Int64($0 % 211) - 100 }
            let bv = Self.column(n, 0.3, 12) { Int64($0 % 17) }
            let cv: [Bool?] = (0..<n).map { Self.nullAt($0, 0.3, 13) ? nil : ($0 % 3 == 0) }
            let a = try MetalArray<Int64>(av), b = try MetalArray<Int64>(bv)
            let cond = try Self.boolArray(cv)
            let set: [Int64] = [-100, 0, 33, 110]

            /// Issues every kernel; the host reads happen in `harvest`, after the batch has run.
            func issue() throws -> (MetalBooleanArray, MetalBooleanArray, MetalArray<Int64>, MetalArray<Int64>,
                                    MetalArray<Int64>, MetalArray<Int64>, MetalBooleanArray, MetalArray<Int32>,
                                    MetalBooleanArray, MetalBooleanArray) {
                (try a.isNull(), try a.isValid(), try a.fillingNull(-1), try a.dropNull(),
                 try MetalArray<Int64>.ifElse(cond, a, b), try MetalArray<Int64>.coalesce([a, b]),
                 try a.isIn(set), try a.indexIn(set), try cond.andKleene(cond), try cond.orKleene(cond))
            }
            func harvest(_ t: (MetalBooleanArray, MetalBooleanArray, MetalArray<Int64>, MetalArray<Int64>,
                               MetalArray<Int64>, MetalArray<Int64>, MetalBooleanArray, MetalArray<Int32>,
                               MetalBooleanArray, MetalBooleanArray)) -> StructuralResults {
                var r = StructuralResults()
                r.isNull = t.0.toArray(); r.isValid = t.1.toArray(); r.fill = t.2.toArray()
                r.drop = t.3.toArray(); r.ifElse = t.4.toArray(); r.coalesce = t.5.toArray()
                r.isIn = t.6.toArray(); r.indexIn = t.7.toArray()
                r.andK = t.8.toArray(); r.orK = t.9.toArray()
                return r
            }

            let plain = harvest(try issue())
            let batched = harvest(try MetalContext.shared.batch { try issue() })
            XCTAssertEqual(batched, plain, "batched result differs from unbatched, n=\(n)")
            // And the unbatched run is right: spot-check against the oracle.
            XCTAssertEqual(plain.coalesce, (0..<n).map { av[$0] ?? bv[$0] }, "coalesce oracle n=\(n)")
        }
    }

    /// Structural functions applied to a result whose length the GPU is still deciding (a filter inside
    /// the same batch), which is the case `Dispatch.setLength` exists for.
    func testStructuralOnPendingBatchedResult() throws {
        try requireRealGPU()
        for n in [1, 33, 4097, 300_003] {
            let av = Self.column(n, 0.3, 21) { Int32($0 % 211) - 100 }
            let a = try MetalArray<Int32>(av)
            let mask = try a.compare(.gt, 0)
            let expectedKept: [Int32?] = av.compactMap { $0 }.filter { $0 > 0 }.map { Optional($0) }
            // Both results are issued against a length the GPU has not decided yet, and read after the batch.
            let (nulls, filled, valid) = try MetalContext.shared.batch {
                () -> (MetalBooleanArray, MetalArray<Int32>, MetalBooleanArray) in
                let kept = try a.filter(mask)          // length decided on the GPU
                return (try kept.isNull(), try kept.fillingNull(-1), try kept.isValid())
            }
            XCTAssertEqual(nulls.toArray(), expectedKept.map { _ in Optional(false) }, "pending isNull n=\(n)")
            XCTAssertEqual(valid.toArray(), expectedKept.map { _ in Optional(true) }, "pending isValid n=\(n)")
            XCTAssertEqual(filled.toArray(), expectedKept, "pending fillNull n=\(n)")
        }
    }
}
