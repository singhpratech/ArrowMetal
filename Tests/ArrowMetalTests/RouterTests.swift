import XCTest
@testable import ArrowMetal

/// Deterministic generator so a failure reproduces.
struct RouterRNG: RandomNumberGenerator {
    var state: UInt64
    init(_ seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// Byte-level Arrow comparison: length, null count, validity presence and bits, and the value bytes
/// of every slot (including slots under nulls, which both paths compute).
func assertArrowIdentical<T: ArrowPrimitive>(_ x: MetalArray<T>, _ y: MetalArray<T>, _ what: String,
                                             file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(x.length, y.length, "\(what): length", file: file, line: line)
    XCTAssertEqual(x.nullCount, y.nullCount, "\(what): null count", file: file, line: line)
    XCTAssertEqual(x.validity == nil, y.validity == nil, "\(what): validity presence", file: file, line: line)
    XCTAssertEqual(x.values.byteCount, y.values.byteCount, "\(what): values buffer size", file: file, line: line)
    guard x.length == y.length else { return }
    let n = x.length
    withExtendedLifetime((x, y)) {
        if n > 0 {
            XCTAssertEqual(memcmp(x.values.contents, y.values.contents, n * T.byteWidth), 0, "\(what): value bytes", file: file, line: line)
        }
        if let a = x.validity, let b = y.validity {
            let p = a.typed(UInt8.self), q = b.typed(UInt8.self)
            let bad = (0..<n).first { Bitmap.isSet(p, $0) != Bitmap.isSet(q, $0) }
            XCTAssertNil(bad, "\(what): validity bit", file: file, line: line)
        }
    }
}

func assertArrowIdentical(_ x: MetalBooleanArray, _ y: MetalBooleanArray, _ what: String,
                          file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(x.length, y.length, "\(what): length", file: file, line: line)
    XCTAssertEqual(x.nullCount, y.nullCount, "\(what): null count", file: file, line: line)
    XCTAssertEqual(x.validity == nil, y.validity == nil, "\(what): validity presence", file: file, line: line)
    guard x.length == y.length else { return }
    let n = x.length
    withExtendedLifetime((x, y)) {
        let p = x.values.typed(UInt8.self), q = y.values.typed(UInt8.self)
        XCTAssertNil((0..<n).first { Bitmap.isSet(p, $0) != Bitmap.isSet(q, $0) }, "\(what): value bit", file: file, line: line)
        if let a = x.validity, let b = y.validity {
            let pv = a.typed(UInt8.self), qv = b.typed(UInt8.self)
            XCTAssertNil((0..<n).first { Bitmap.isSet(pv, $0) != Bitmap.isSet(qv, $0) }, "\(what): validity bit", file: file, line: line)
        }
    }
}

final class RouterTests: XCTestCase {
    override func setUpWithError() throws {
        try requireRealGPU()
        Router.clearLastDecision()
    }

    /// Runs `body` on the GPU, then on the CPU, asserting each ran where it was sent.
    func both<R>(_ op: RoutedOp, file: StaticString = #filePath, line: UInt = #line, _ body: () throws -> R) throws -> (gpu: R, cpu: R) {
        let g = try Router.withMode(.gpu, body)
        XCTAssertEqual(Router.lastDecision?.op, op, file: file, line: line)
        XCTAssertEqual(Router.lastDecision?.path, .gpu, file: file, line: line)
        let c = try Router.withMode(.cpu, body)
        XCTAssertEqual(Router.lastDecision?.op, op, file: file, line: line)
        XCTAssertEqual(Router.lastDecision?.path, .cpu, "\(Router.lastDecision.map { "\($0)" } ?? "no decision")", file: file, line: line)
        return (g, c)
    }

    static let sizes = [0, 1, 3, 4, 5, 31, 32, 33, 63, 64, 65, 255, 256, 257, 1023, 1025, 4099, 65_537, 100_003]

    func column<T: ArrowPrimitive>(_ n: Int, nulls: Double, seed: UInt64, _ gen: (inout RouterRNG) -> T) throws -> MetalArray<T> {
        var g = RouterRNG(seed)
        var v: [T?] = []
        v.reserveCapacity(n)
        for _ in 0..<n { v.append(Double.random(in: 0..<1, using: &g) < nulls ? nil : gen(&g)) }
        return try MetalArray<T>(v)
    }

    /// Plain arrays, arrays with nulls, and slices at an offset that is not a multiple of 32.
    func variants<T: ArrowPrimitive>(_ n: Int, seed: UInt64, _ gen: @escaping (inout RouterRNG) -> T) throws -> [(String, MetalArray<T>)] {
        var out: [(String, MetalArray<T>)] = [
            ("n=\(n) no nulls", try column(n, nulls: 0, seed: seed, gen)),
            ("n=\(n) 10% nulls", try column(n, nulls: 0.1, seed: seed &+ 1, gen)),
            ("n=\(n) 90% nulls", try column(n, nulls: 0.9, seed: seed &+ 2, gen)),
        ]
        if n > 0 {
            let base = try column(n + 7, nulls: 0.2, seed: seed &+ 3, gen)
            out.append(("n=\(n) slice at 7", try base.slice(offset: 7, length: n)))
        }
        return out
    }

    // MARK: both paths, byte-identical

    func checkAll<T: ArrowPrimitive>(_: T.Type, scalar: T, seed: UInt64, _ gen: @escaping (inout RouterRNG) -> T) throws {
        for n in Self.sizes {
            for (label, a) in try variants(n, seed: seed &+ UInt64(n), gen) {
                let what = "\(T.self) \(label)"
                let s = try both(.sum) { try a.sum() }
                if T.isFloatingPoint, case .float(let x)? = s.gpu, case .float(let y)? = s.cpu {
                    XCTAssertEqual(x.bitPattern, y.bitPattern, "\(what): sum bits")
                } else {
                    XCTAssertEqual(s.gpu, s.cpu, "\(what): sum")
                }
                if T.self != Float.self {
                    let mn = try both(.min) { try a.min() }, mx = try both(.max) { try a.max() }
                    XCTAssertEqual(mn.gpu.map { $0.asDouble.bitPattern }, mn.cpu.map { $0.asDouble.bitPattern }, "\(what): min")
                    XCTAssertEqual(mx.gpu.map { $0.asDouble.bitPattern }, mx.cpu.map { $0.asDouble.bitPattern }, "\(what): max")
                    if !T.isFloatingPoint { XCTAssertEqual(mn.gpu, mn.cpu, "\(what): min"); XCTAssertEqual(mx.gpu, mx.cpu, "\(what): max") }
                }
                let b = try column(n, nulls: 0.15, seed: seed &+ 99, gen)
                for op in CompareOp.allCases {
                    let c1 = try both(.compare) { try a.compare(op, scalar) }
                    assertArrowIdentical(c1.gpu, c1.cpu, "\(what): compare \(op) scalar")
                    let c2 = try both(.compare) { try a.compare(op, b) }
                    assertArrowIdentical(c2.gpu, c2.cpu, "\(what): compare \(op) array")
                    if T.self != Float.self && T.self != Double.self {
                        let fw = try both(.filter) { try a.filter(where: op, scalar) }
                        assertArrowIdentical(fw.gpu, fw.cpu, "\(what): filter where \(op)")
                    }
                }
                if T.self != Float.self {
                    for op in [ArithmeticOp.add, .sub, .mul] {
                        let r1 = try both(.arithmetic) { try a.arithmetic(op, scalar) }
                        assertArrowIdentical(r1.gpu, r1.cpu, "\(what): \(op) scalar")
                        let r2 = try both(.arithmetic) { try a.arithmetic(op, b) }
                        assertArrowIdentical(r2.gpu, r2.cpu, "\(what): \(op) array")
                    }
                }
                let mask = try column(n, nulls: 0.2, seed: seed &+ 7) { g in Bool.random(using: &g) ? Int8(1) : Int8(0) }.compare(.eq, 1)
                let f = try both(.filter) { try a.filter(mask) }
                assertArrowIdentical(f.gpu, f.cpu, "\(what): filter mask")
                let noNullMask = try MetalBooleanArray((0..<n).map { $0 % 3 != 0 })
                let f2 = try both(.filter) { try a.filter(noNullMask) }
                assertArrowIdentical(f2.gpu, f2.cpu, "\(what): filter mask without nulls")
            }
        }
    }

    func testIntegersBothPathsIdentical() throws {
        try checkAll(Int8.self, scalar: 3, seed: 1) { Int8.random(in: .min ... .max, using: &$0) }
        try checkAll(UInt8.self, scalar: 200, seed: 2) { UInt8.random(in: .min ... .max, using: &$0) }
        try checkAll(Int16.self, scalar: -5, seed: 3) { Int16.random(in: .min ... .max, using: &$0) }
        try checkAll(UInt16.self, scalar: 7, seed: 4) { UInt16.random(in: .min ... .max, using: &$0) }
        try checkAll(Int32.self, scalar: 0, seed: 5) { Int32.random(in: .min ... .max, using: &$0) }
        try checkAll(UInt32.self, scalar: 1 << 31, seed: 6) { UInt32.random(in: .min ... .max, using: &$0) }
        try checkAll(Int64.self, scalar: 1, seed: 7) { Int64.random(in: .min ... .max, using: &$0) }
        try checkAll(UInt64.self, scalar: .max, seed: 8) { UInt64.random(in: .min ... .max, using: &$0) }
    }

    static let doubleSpecials: [UInt64] = [
        0, 0x8000_0000_0000_0000,                              // +0, -0
        0x7FF0_0000_0000_0000, 0xFFF0_0000_0000_0000,          // +inf, -inf
        0x7FF8_0000_0000_0000, 0xFFF8_0000_0000_0123,          // quiet NaNs, one negative with a payload
        0x7FF0_0000_0000_0001, 0xFFF4_0000_0000_0042,          // signaling NaNs
        0x0000_0000_0000_0001, 0x800F_FFFF_FFFF_FFFF,          // subnormals
        0x7FEF_FFFF_FFFF_FFFF, 0xFFEF_FFFF_FFFF_FFFF,          // +-max
        0x0010_0000_0000_0000,                                 // min normal
    ]

    func doubleValue(_ g: inout RouterRNG) -> Double {
        switch Int.random(in: 0..<10, using: &g) {
        case 0: return Double(bitPattern: Self.doubleSpecials[Int.random(in: 0..<Self.doubleSpecials.count, using: &g)])
        case 1: return Double(bitPattern: g.next())                          // any pattern at all
        case 2: return Double.random(in: -1e-300...1e-300, using: &g)
        default: return Double.random(in: -1e6...1e6, using: &g)
        }
    }

    func testFloat64BothPathsIdentical() throws {
        try checkAll(Double.self, scalar: 0.5, seed: 11) { self.doubleValue(&$0) }
        try checkAll(Double.self, scalar: -.infinity, seed: 12) { Double.random(in: -1...1, using: &$0) }
        try checkAll(Double.self, scalar: .nan, seed: 13) { self.doubleValue(&$0) }
    }

    func testFloat32BothPathsIdentical() throws {
        let specials: [UInt32] = [0, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC0_0000, 0xFFC0_0123, 0x7F80_0001,
                                  0x0000_0001, 0x807F_FFFF, 0x7F7F_FFFF]
        try checkAll(Float.self, scalar: 0.25, seed: 21) { g in
            switch Int.random(in: 0..<8, using: &g) {
            case 0: return Float(bitPattern: specials[Int.random(in: 0..<specials.count, using: &g)])
            case 1: return Float(bitPattern: UInt32(truncatingIfNeeded: g.next()))
            default: return Float.random(in: -1e4...1e4, using: &g)
            }
        }
    }

    /// Float sums use the GPU's order: past 2,097,152 rows every one of the 2,048 x 256 threads takes
    /// more than one block, which is the wrap-around branch of the emulation.
    func testFloatSumOrderPastOneBlockPerThread() throws {
        for n in [2_097_152, 2_097_155, 2_500_003] {
            for nulls in [0.0, 0.1] {
                let d = try column(n, nulls: nulls, seed: UInt64(n)) { Double.random(in: -1e3...1e3, using: &$0) * 1.000000001 }
                let s = try both(.sum) { try d.sum() }
                guard case .float(let x)? = s.gpu, case .float(let y)? = s.cpu else { return XCTFail("float sum") }
                XCTAssertEqual(x.bitPattern, y.bitPattern, "float64 n=\(n) nulls=\(nulls)")
                let f = try column(n, nulls: nulls, seed: UInt64(n) &+ 5) { Float.random(in: -1e3...1e3, using: &$0) }
                let t = try both(.sum) { try f.sum() }
                guard case .float(let u)? = t.gpu, case .float(let v)? = t.cpu else { return XCTFail("float sum") }
                XCTAssertEqual(u.bitPattern, v.bitPattern, "float32 n=\(n) nulls=\(nulls)")
            }
        }
    }

    func testAllNullAndAllNaN() throws {
        let nulls = try MetalArray<Double>([Double?](repeating: nil, count: 100))
        XCTAssertNil(try Router.withMode(.cpu) { try nulls.sum() })
        XCTAssertNil(try Router.withMode(.cpu) { try nulls.min() })
        let nans = try MetalArray<Double>([Double](repeating: .nan, count: 100))
        let g = try Router.withMode(.gpu) { try nans.min() }, c = try Router.withMode(.cpu) { try nans.max() }
        XCTAssertNil(g); XCTAssertNil(c)
        let zeros = try MetalArray<Double>([-0.0, 0.0, -0.0])
        let zg = try Router.withMode(.gpu) { try zeros.min() }!, zc = try Router.withMode(.cpu) { try zeros.min() }!
        XCTAssertEqual(zg.bitPattern, zc.bitPattern)
    }

    // MARK: group-by sum

    func checkGroupBy<K: ArrowIndex, T: ArrowPrimitive & FixedWidthInteger>(_: K.Type, _: T.Type, keyCount: Int, n: Int, seed: UInt64) throws {
        var g = RouterRNG(seed)
        var keys: [K?] = [], vals: [T?] = []
        for _ in 0..<n {
            // Some keys out of range (negative or >= keyCount) and some null: both are skipped.
            let r = Int.random(in: 0..<100, using: &g)
            keys.append(r < 5 ? nil : K(truncatingIfNeededInt64: Int64(r < 8 ? keyCount + r : Int.random(in: 0..<keyCount, using: &g))))
            vals.append(Int.random(in: 0..<10, using: &g) == 0 ? nil : T.random(in: .min ... .max, using: &g))
        }
        let kc = try MetalArray<K>(keys), vc = try MetalArray<T>(vals)
        let gb = try kc.groupBy(keyCount: keyCount)
        let r = try both(.groupBySum) { try gb.sum(vc) }
        assertArrowIdentical(r.gpu, r.cpu, "group-by sum \(K.self)/\(T.self) keys=\(keyCount) n=\(n)")
        // Without any validity bitmaps.
        let kc2 = try MetalArray<K>(keys.map { $0 ?? 0 }), vc2 = try MetalArray<T>(vals.map { $0 ?? 1 })
        let gb2 = try kc2.groupBy(keyCount: keyCount)
        let r2 = try both(.groupBySum) { try gb2.sum(vc2) }
        assertArrowIdentical(r2.gpu, r2.cpu, "group-by sum no nulls \(K.self)/\(T.self) keys=\(keyCount) n=\(n)")
    }

    func testGroupBySumBothPathsIdentical() throws {
        for n in [0, 1, 63, 64, 65, 1000, 100_003] {
            for kcount in [1, 7, 1000, 1024] {
                try checkGroupBy(Int32.self, Int64.self, keyCount: kcount, n: n, seed: UInt64(n * 31 + kcount))
                try checkGroupBy(Int64.self, Int32.self, keyCount: kcount, n: n, seed: UInt64(n * 37 + kcount))
                try checkGroupBy(UInt32.self, UInt64.self, keyCount: kcount, n: n, seed: UInt64(n * 41 + kcount))
                try checkGroupBy(Int32.self, Int8.self, keyCount: kcount, n: n, seed: UInt64(n * 43 + kcount))
                try checkGroupBy(Int32.self, UInt16.self, keyCount: kcount, n: n, seed: UInt64(n * 47 + kcount))
            }
        }
    }

    /// `GroupBy.sumUnsigned` (uint64 values kept unsigned: C op 0 over uint64, Python's `sum` on a
    /// uint64 column) is routed like `sum`, byte-identical on both paths.
    func testGroupBySumUnsignedBothPathsIdentical() throws {
        for n in [0, 1, 65, 1000, 100_003] {
            for kcount in [1, 7, 1024] {
                var g = RouterRNG(UInt64(n * 53 + kcount))
                var keys: [Int32?] = [], vals: [UInt64?] = []
                for _ in 0..<n {
                    let r = Int.random(in: 0..<100, using: &g)
                    keys.append(r < 5 ? nil : Int32(r < 8 ? kcount + r : Int.random(in: 0..<kcount, using: &g)))
                    vals.append(Int.random(in: 0..<10, using: &g) == 0 ? nil : UInt64.random(in: .min ... .max, using: &g))
                }
                let gb = try MetalArray<Int32>(keys).groupBy(keyCount: kcount)
                let vc = try MetalArray<UInt64>(vals)
                let r = try both(.groupBySum) { try gb.sumUnsigned(vc) }
                assertArrowIdentical(r.gpu, r.cpu, "group-by sumUnsigned keys=\(kcount) n=\(n)")
                // Same bits as the signed sum's CPU loop.
                let signed = try Router.withMode(.cpu) { try gb.sum(vc) }
                withExtendedLifetime((signed, r.cpu)) {
                    XCTAssertEqual(memcmp(signed.values.contents, r.cpu.values.contents, kcount * 8), 0)
                }
            }
        }
    }

    func testGroupBySumAboveLimitStaysOnGPU() throws {
        let keys = try MetalArray<Int32>((0..<100).map { Int32($0 % 2000) })
        let vals = try MetalArray<Int64>((0..<100).map { Int64($0) })
        let gb = try keys.groupBy(keyCount: Router.groupBySumMaxKeys + 1)
        _ = try Router.withMode(.cpu) { try gb.sum(vals) }
        XCTAssertEqual(Router.lastDecision?.path, .gpu)
        XCTAssertEqual(Router.lastDecision?.reason, .noCPUPath("more than \(Router.groupBySumMaxKeys) keys"))
    }

    // MARK: the CPU loops against the oracle

    /// `CPUReference` is the tests' oracle; the router's loops must agree with it wherever the oracle's
    /// semantics are the GPU's (no NaN in min/max; float sums compared within the oracle's tolerance,
    /// since the oracle adds in row order and the GPU does not).
    func testCPULoopsMatchCPUReference() throws {
        for n in [0, 1, 33, 1000, 65_537] {
            let a = try column(n, nulls: 0.1, seed: UInt64(n)) { Int64.random(in: -1_000_000...1_000_000, using: &$0) }
            let b = try column(n, nulls: 0.1, seed: UInt64(n) + 1) { Int64.random(in: -1_000_000...1_000_000, using: &$0) }
            XCTAssertEqual(RouterCPU.sum(a), CPUReference.sum(a))
            XCTAssertEqual(RouterCPU.minMax(a, isMin: true), CPUReference.min(a))
            XCTAssertEqual(RouterCPU.minMax(a, isMin: false), CPUReference.max(a))
            for op in CompareOp.allCases {
                let c = try RouterCPU.compare(a, op, 17), r = try CPUReference.compare(a, op, scalar: 17)
                XCTAssertEqual(c.toArray(), r.toArray(), "compare \(op)")
                XCTAssertEqual(try RouterCPU.compare(a, op, b).toArray(), try CPUReference.compare(a, op, array: b).toArray())
            }
            for op in [ArithmeticOp.add, .sub, .mul] {
                XCTAssertEqual(try RouterCPU.arithmetic(a, op, 3).toArray(), try CPUReference.arithmetic(a, op, scalar: 3).toArray())
                XCTAssertEqual(try RouterCPU.arithmetic(a, op, b).toArray(), try CPUReference.arithmetic(a, op, array: b).toArray())
            }
            let mask = try RouterCPU.compare(b, .gt, 0)
            XCTAssertEqual(try RouterCPU.filter(a, mask).toArray(), try CPUReference.filter(a, mask).toArray())
            let d = try column(n, nulls: 0.1, seed: UInt64(n) + 2) { Double.random(in: -100...100, using: &$0) }
            if let s = RouterCPU.sum(d), let r = CPUReference.sum(d) {
                XCTAssertEqual(s.asDouble, r.asDouble, accuracy: 1e-9 * Swift.max(1, abs(r.asDouble)))
            } else { XCTAssertEqual(n == 0 || d.validCount == 0, true) }
            XCTAssertEqual(RouterCPU.minMax(d, isMin: true), CPUReference.min(d))
            XCTAssertEqual(RouterCPU.minMax(d, isMin: false), CPUReference.max(d))
            XCTAssertEqual(try RouterCPU.arithmetic(d, .mul, 2.5).toArray(), try CPUReference.arithmetic(d, .mul, scalar: 2.5).toArray())
        }
    }

    // MARK: decisions

    func testAutoFollowsTheTable() throws {
        try Router.withMode(.auto) { try autoFollowsTheTable() }
    }

    func autoFollowsTheTable() throws {
        for op in RoutedOp.allCases {
            let c = Router.crossoverRows(op)
            XCTAssertGreaterThan(c, RouterTable.bracketLowRows(op), "\(op) fitted inside its bracket")
            XCTAssertLessThanOrEqual(c, RouterTable.measuredStepRows(op), "\(op) fitted inside its bracket")
            let below = Router.decide(op, rows: c - 1, cpuPath: nil, measured: true, pending: false, batching: false, typeName: "long")
            XCTAssertEqual(below.path, .cpu); XCTAssertEqual(below.reason, .belowCrossover(crossover: c))
            let at = Router.decide(op, rows: c, cpuPath: nil, measured: true, pending: false, batching: false, typeName: "long")
            XCTAssertEqual(at.path, .gpu); XCTAssertEqual(at.reason, .atOrAboveCrossover(crossover: c))
        }
        let small = try MetalArray<Int64>([1, 2, 3])
        _ = try Router.withMode(.auto) { try small.sum() }
        XCTAssertEqual(Router.lastDecision?.path, .cpu)
        XCTAssertEqual(Router.lastDecision?.rows, 3)
        // Floating-point columns have no measured crossover: auto keeps them on the GPU.
        let f = try MetalArray<Double>([1, 2, 3])
        _ = try Router.withMode(.auto) { try f.sum() }
        XCTAssertEqual(Router.lastDecision?.path, .gpu)
        XCTAssertEqual(Router.lastDecision?.reason, .notMeasuredForType("float64"))
        // The arithmetic row was measured on add and subtract follows it; multiply has its own row.
        let addRow = Router.crossoverRows(.arithmetic)
        XCTAssertEqual(Router.crossoverRows(arithmetic: .add), addRow)
        XCTAssertEqual(Router.crossoverRows(arithmetic: .sub), addRow)
        XCTAssertNil(Router.crossoverRows(arithmetic: .div))
        _ = try small.subtract(1)
        XCTAssertEqual(Router.lastDecision?.path, .cpu)
        XCTAssertEqual(Router.lastDecision?.reason, .belowCrossover(crossover: addRow))
        let mul = try XCTUnwrap(Router.crossoverRows(arithmetic: .mul))
        XCTAssertEqual(mul, RouterTable.multiplyCrossoverRows)
        XCTAssertGreaterThan(mul, RouterTable.multiplyBracketLowRows, "multiply fitted inside its bracket")
        XCTAssertLessThanOrEqual(mul, RouterTable.multiplyMeasuredStepRows, "multiply fitted inside its bracket")
        _ = try small.multiply(2)
        XCTAssertEqual(Router.lastDecision?.path, .cpu)
        XCTAssertEqual(Router.lastDecision?.reason, .belowCrossover(crossover: mul))
        _ = try small.multiply(small)
        XCTAssertEqual(Router.lastDecision?.path, .cpu)
    }

    /// Multiply under `auto` switches at its own row, not at the add/subtract row.
    func testAutoMultiplyUsesItsOwnRow() throws {
        let mul = try XCTUnwrap(Router.crossoverRows(arithmetic: .mul))
        for n in [mul - 1, mul] {
            let a = try MetalArray<Int64>((0..<n).map { Int64($0 % 1000) - 500 })
            let out = try Router.withMode(.auto) { try a.multiply(3) }
            let d = try XCTUnwrap(Router.lastDecision)
            XCTAssertEqual(d.op, .arithmetic)
            XCTAssertEqual(d.rows, n)
            if n < mul {
                XCTAssertEqual(d.path, .cpu); XCTAssertEqual(d.reason, .belowCrossover(crossover: mul))
            } else {
                XCTAssertEqual(d.path, .gpu); XCTAssertEqual(d.reason, .atOrAboveCrossover(crossover: mul))
            }
            let gpu = try Router.withMode(.gpu) { try a.multiply(3) }
            XCTAssertEqual(try out.toArray(), try gpu.toArray())
        }
        // Int32 columns use the same row (integer widths share the int64 rows).
        let small = try MetalArray<Int32>([1, 2, 3])
        _ = try Router.withMode(.auto) { try small.multiply(2) }
        XCTAssertEqual(Router.lastDecision?.reason, .belowCrossover(crossover: mul))
        // Float64 has no measured crossover: auto keeps it on the GPU.
        let f = try MetalArray<Double>([1, 2, 3])
        _ = try Router.withMode(.auto) { try f.multiply(2) }
        XCTAssertEqual(Router.lastDecision?.reason, .notMeasuredForType("float64"))
    }

    func testAutoAtScaleRunsTheGPU() throws {
        let n = Router.crossoverRows(.sum)
        let a = try MetalArray<Int64>((0..<n).map { Int64($0) })
        let s = try Router.withMode(.auto) { try a.sum() }
        XCTAssertEqual(Router.lastDecision?.path, .gpu)
        XCTAssertEqual(s, .int(Int64(n) * Int64(n - 1) / 2))
    }

    func testBatchAndPendingStayOnGPU() throws {
        let a = try MetalArray<Int64>((0..<100).map { Int64($0) })
        let ctx = MetalContext.shared
        let total: SumResult? = try Router.withMode(.cpu) {
            try ctx.batch {
                let m = try a.compare(.gt, 10)
                XCTAssertEqual(Router.lastDecision?.reason, .batchOpen)
                let f = try a.filter(m)
                XCTAssertEqual(Router.lastDecision?.reason, .batchOpen)
                XCTAssertTrue(f.pending)
                return try f.sum()
            }
        }
        XCTAssertEqual(total, .int((11..<100).reduce(0) { $0 + Int64($1) }))
        XCTAssertEqual(Router.lastDecision?.path, .gpu)
        // The decision function itself: pending input, and a batch, win over a forced cpu mode.
        let d = Router.withMode(.cpu) { Router.decide(.sum, rows: 10, cpuPath: nil, measured: true, pending: true, batching: false, typeName: "long") }
        XCTAssertEqual(d.reason, .pendingInput)
    }

    func testNoCPUPathIsRecorded() throws {
        let f = try MetalArray<Float>([1, 2, 3])
        _ = try Router.withMode(.cpu) { try f.min() }
        XCTAssertEqual(Router.lastDecision?.path, .gpu)
        if case .noCPUPath? = Router.lastDecision?.reason {} else { XCTFail("float32 min: \(String(describing: Router.lastDecision))") }
        _ = try Router.withMode(.cpu) { try f.add(1) }
        if case .noCPUPath? = Router.lastDecision?.reason {} else { XCTFail("float32 add") }
        _ = try Router.withMode(.cpu) { try f.filter(where: .gt, 1) }
        if case .noCPUPath? = Router.lastDecision?.reason {} else { XCTFail("float32 filter where") }
        let i = try MetalArray<Int64>([1, 2, 3])
        _ = try Router.withMode(.cpu) { try i.divide(2) }
        XCTAssertEqual(Router.lastDecision?.reason, .noCPUPath("divide is not routed"))
    }

    func testModesAndOverrides() throws {
        let saved = Router.mode
        defer { Router.mode = saved }
        let a = try MetalArray<Int64>([5, 6, 7])
        Router.mode = .cpu
        _ = try a.max()
        XCTAssertEqual(Router.lastDecision?.reason, .forced(.cpu, thread: false))
        _ = try Router.withMode(.gpu) { try a.max() }
        XCTAssertEqual(Router.lastDecision?.reason, .forced(.gpu, thread: true))
        Router.mode = .gpu
        _ = try Router.withMode(.auto) { try a.max() }
        XCTAssertEqual(Router.lastDecision?.path, .cpu)
        // Nested overrides restore the outer one.
        Router.withMode(.cpu) {
            Router.withMode(.gpu) { XCTAssertEqual(Router.threadMode, .gpu) }
            XCTAssertEqual(Router.threadMode, .cpu)
        }
        XCTAssertNil(Router.threadMode)
        // The override is per thread.
        let other = expectation(description: "other thread")
        Router.withMode(.cpu) {
            Thread.detachNewThread {
                XCTAssertNil(Router.threadMode)
                _ = try? a.max()
                XCTAssertEqual(Router.lastDecision?.reason, .forced(.gpu, thread: false))
                other.fulfill()
            }
            wait(for: [other], timeout: 10)
        }
        XCTAssertEqual(RouterMode(rawValue: "cpu"), .cpu)
        XCTAssertNil(RouterMode(rawValue: "fastest"))
    }

    // MARK: determinism and the table

    /// 200 (operation, size) pairs over the routed operations and the sizes around every crossover.
    static func decisionPairs() -> [(RoutedOp, Int)] {
        var sizes: Set<Int> = [0, 1, 1_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
        for op in RoutedOp.allCases {
            let c = RouterCrossovers.shipped.crossover(op)
            sizes.formUnion([c - 1, c, c + 1, RouterTable.bracketLowRows(op), RouterTable.measuredStepRows(op)])
        }
        var k = 0
        while sizes.count * RoutedOp.allCases.count < 200 { sizes.insert(7 * k + 3); k += 1 }
        let pairs = sizes.sorted().flatMap { n in RoutedOp.allCases.map { ($0, n) } }
        return Array(pairs.prefix(200))
    }

    func testRouteIsPure() throws {
        Router.useShippedTable()
        let pairs = Self.decisionPairs()
        XCTAssertEqual(pairs.count, 200)
        let table = Router.table
        let first = pairs.map { Router.route($0.0, rows: $0.1, mode: .auto, crossover: table.crossover($0.0)) }
        Router.clearLastDecision()
        for _ in 0..<1000 {
            for (k, (op, n)) in pairs.enumerated() {
                let d = Router.route(op, rows: n, mode: .auto, crossover: Router.crossoverRows(op))
                if d != first[k] { XCTFail("\(op) \(n): \(d) != \(first[k])"); return }
            }
        }
        XCTAssertNil(Router.lastDecision, "route records nothing")
        XCTAssertEqual(Router.table, table, "route leaves the table alone")
        // The rule is the table's comparison and nothing else.
        for (k, (op, n)) in pairs.enumerated() {
            XCTAssertEqual(first[k].path, n < table.crossover(op) ? .cpu : .gpu, "\(op) \(n)")
        }
        // `explain` gives what a routed call records.
        let a = try MetalArray<Int64>(Array(0..<5000))
        try Router.withMode(.auto) {
            _ = try a.sum()
            let e = try XCTUnwrap(Router.explain(operation: "sum", dtype: "int64", rows: 5000))
            XCTAssertEqual(Router.lastDecision, e.decision)
            _ = try a.multiply(3)
            XCTAssertEqual(Router.lastDecision, try XCTUnwrap(Router.explain(operation: "multiply", dtype: "int64", rows: 5000)).decision)
        }
    }

    func testShippedTableIsTheGeneratedLiteral() throws {
        let t = RouterCrossovers.shipped
        XCTAssertTrue(t.isShipped)
        for op in RoutedOp.allCases {
            let r = try XCTUnwrap(t.rows[op])
            XCTAssertEqual(r.crossover, RouterTable.crossoverRows(op))
            XCTAssertEqual(r.stepRows, RouterTable.measuredStepRows(op))
            XCTAssertEqual(r.bracketLowRows, RouterTable.bracketLowRows(op))
            XCTAssertEqual(r.points.count, 2, "\(op)")
            XCTAssertGreaterThan(r.points[0].gpuMicros, r.points[0].cpuMicros, "\(op): CPU ahead at the low end")
            XCTAssertLessThanOrEqual(r.points[1].gpuMicros, r.points[1].cpuMicros, "\(op): GPU ahead at the step")
        }
        XCTAssertEqual(t.multiply.crossover, RouterTable.multiplyCrossoverRows)
        XCTAssertEqual(t.source, RouterTable.source)
    }

    func testJSONTableLoadsAndFallsBack() throws {
        defer { Router.useShippedTable() }
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("am-router-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let good = dir.appendingPathComponent("t.json").path
        let json = """
        {"format": "arrowmetal-router-table/1", "machine": {"chip": "Test Chip"}, "date": "2026-09-25T00:00:00Z",
         "grid": {"name": "quick"}, "source": "test",
         "crossovers": {"sum": {"label": "sum(int64)", "crossover_rows": 1234, "step_rows": 3000, "bracket_low_rows": 1000,
                                "points": [{"rows": 1000, "gpu_us": 10.5, "cpu_us": 5}, {"rows": 3000, "gpu_us": 11, "cpu_us": 15}]},
                        "compare": {"label": "compare(int64 > 0)", "crossover_rows": null},
                        "multiply": {"crossover_rows": 777}}}
        """
        try json.write(toFile: good, atomically: true, encoding: .utf8)
        try Router.loadTable(path: good)
        let t = Router.table
        XCTAssertEqual(t.origin, .file(good))
        XCTAssertEqual(Router.crossoverRows(.sum), 1234)
        XCTAssertEqual(Router.crossoverRows(arithmetic: .mul), 777)
        XCTAssertEqual(Router.crossoverRows(.compare), RouterTable.crossoverRows(.compare), "null keeps the shipped row")
        XCTAssertEqual(Router.crossoverRows(.filter), RouterTable.crossoverRows(.filter), "missing keeps the shipped row")
        XCTAssertEqual(t.shippedOps, ["min", "max", "compare", "arithmetic", "filter", "group_by_sum"])
        XCTAssertEqual(t.chip, "Test Chip"); XCTAssertEqual(t.grid, "quick")
        XCTAssertEqual(t.rows[.sum]?.points, [RouterPoint(rows: 1000, gpuMicros: 10.5, cpuMicros: 5), RouterPoint(rows: 3000, gpuMicros: 11, cpuMicros: 15)])
        XCTAssertEqual(Router.route(.sum, rows: 1233, mode: .auto, crossover: Router.crossoverRows(.sum)).path, .cpu)
        XCTAssertEqual(Router.route(.sum, rows: 1234, mode: .auto, crossover: Router.crossoverRows(.sum)).path, .gpu)

        // A bad file leaves the table in force unchanged.
        for (name, text) in [("fmt", #"{"format": "other", "crossovers": {}}"#), ("junk", "not json"),
                             ("neg", #"{"format": "arrowmetal-router-table/1", "crossovers": {"sum": {"crossover_rows": -5}}}"#),
                             ("unknown", #"{"format": "arrowmetal-router-table/1", "crossovers": {"sort": {"crossover_rows": 5}}}"#),
                             ("empty", #"{"format": "arrowmetal-router-table/1", "crossovers": {}}"#)] {
            let p = dir.appendingPathComponent("\(name).json").path
            try text.write(toFile: p, atomically: true, encoding: .utf8)
            XCTAssertThrowsError(try Router.loadTable(path: p), name)
            XCTAssertEqual(Router.table, t, name)
        }
        XCTAssertThrowsError(try Router.loadTable(path: dir.appendingPathComponent("missing.json").path))
        Router.useShippedTable()
        XCTAssertTrue(Router.table.isShipped)
        XCTAssertEqual(Router.crossoverRows(.sum), RouterTable.crossoverRows(.sum))

        // Which table a process starts with.
        let home = dir.appendingPathComponent("home").path
        let machine = RouterCrossovers.machineTablePath(home: home, chip: "Apple M9 Ultra")
        XCTAssertTrue(machine.hasSuffix("/.arrowmetal/router/apple-m9-ultra.json"), machine)
        XCTAssertTrue(RouterCrossovers.initial(environment: [:], home: home, chip: "Apple M9 Ultra").0.isShipped)
        try FileManager.default.createDirectory(atPath: (machine as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
        try FileManager.default.copyItem(atPath: good, toPath: machine)
        let (fromHome, e1) = RouterCrossovers.initial(environment: [:], home: home, chip: "Apple M9 Ultra")
        XCTAssertEqual(fromHome.origin, .file(machine)); XCTAssertNil(e1)
        XCTAssertTrue(RouterCrossovers.initial(environment: ["ARROWMETAL_ROUTER_TABLE": "shipped"], home: home, chip: "Apple M9 Ultra").0.isShipped)
        XCTAssertTrue(RouterCrossovers.initial(environment: [:], home: home, chip: "Apple M1").0.isShipped, "another chip's file is not loaded")
        let (fromEnv, e2) = RouterCrossovers.initial(environment: ["ARROWMETAL_ROUTER_TABLE": good], home: nil, chip: "x")
        XCTAssertEqual(fromEnv.origin, .file(good)); XCTAssertNil(e2)
        let (bad, e3) = RouterCrossovers.initial(environment: ["ARROWMETAL_ROUTER_TABLE": dir.appendingPathComponent("junk.json").path], home: home, chip: "Apple M9 Ultra")
        XCTAssertTrue(bad.isShipped); XCTAssertNotNil(e3)
        XCTAssertEqual(RouterCrossovers.chipID("Apple M4 Max"), "apple-m4-max")
        XCTAssertEqual(RouterCrossovers.chipID("  Apple  M2 (Pro) "), "apple-m2-pro")
    }

    func testDecisionDescriptions() throws {
        let d = Router.decide(.filter, rows: 10, cpuPath: nil, measured: true, pending: false, batching: false, typeName: "long")
        XCTAssertTrue(d.description.hasPrefix("filter: "), d.description)
        XCTAssertEqual(RoutedOp.allCases.map(\.code), Array(0..<Int32(RoutedOp.allCases.count)))
        XCTAssertEqual(RoutedOp.groupBySum.rawValue, "group_by_sum")
    }
}
