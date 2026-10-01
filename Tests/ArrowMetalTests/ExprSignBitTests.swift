import XCTest
@testable import ArrowMetal

/// `signbit` and `is_nan` in the fused expression grammar, and the totalOrder comparisons they make
/// expressible in one expression (docs/EXPR.md, "totalOrder comparisons"), against references built from
/// the bit patterns: every special value (±0.0, ±inf, NaN of both signs with several payloads,
/// subnormals) with nulls, over Float32 and Float64, at sizes on both sides of a threadgroup.
final class ExprSignBitTests: XCTestCase {

    static let doubleBits: [UInt64] = [
        0x0000_0000_0000_0000, 0x8000_0000_0000_0000,          // +0.0, -0.0
        0x3FF0_0000_0000_0000, 0xBFF0_0000_0000_0000,          // 1, -1
        0x4004_0000_0000_0000, 0xC004_0000_0000_0000,          // 2.5, -2.5
        0x0000_0000_0000_0001, 0x8000_0000_0000_0001,          // ± smallest subnormal
        0x7FF0_0000_0000_0000, 0xFFF0_0000_0000_0000,          // +inf, -inf
        0x7FF8_0000_0000_0000, 0xFFF8_0000_0000_0000,          // ± quiet NaN
        0x7FF0_0000_0000_0001, 0xFFF0_0000_0000_0001,          // ± signalling NaN, payload 1
        0x7FFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF,          // ± NaN, all payload bits
        0x7FF8_DEAD_BEEF_0042, 0xFFF4_0000_0000_1234,          // ± NaN, other payloads
        0x7FEF_FFFF_FFFF_FFFF, 0xFFEF_FFFF_FFFF_FFFF,          // ± largest finite
    ]
    static let floatBits: [UInt32] = [
        0x0000_0000, 0x8000_0000, 0x3F80_0000, 0xBF80_0000, 0x4020_0000, 0xC020_0000,
        0x0000_0001, 0x8000_0001, 0x7F80_0000, 0xFF80_0000, 0x7FC0_0000, 0xFFC0_0000,
        0x7F80_0001, 0xFF80_0001, 0x7FFF_FFFF, 0xFFFF_FFFF, 0x7FC0_BEEF, 0xFFA0_1234,
        0x7F7F_FFFF, 0xFF7F_FFFF,
    ]

    /// `n` values cycling through the special patterns, with a null every 7th row from row 3.
    static func doubles(_ n: Int) -> [Double?] {
        (0..<n).map { i in i % 7 == 3 ? nil : Double(bitPattern: doubleBits[(i * 5 + i / 3) % doubleBits.count]) }
    }
    static func floats(_ n: Int) -> [Float?] {
        (0..<n).map { i in i % 7 == 3 ? nil : Float(bitPattern: floatBits[(i * 5 + i / 3) % floatBits.count]) }
    }

    /// The IEEE totalOrder key (arrow-rs `total_cmp`): the sign-flip transform of the bit pattern.
    static func totalKey(_ bits: UInt64, width: Int) -> UInt64 {
        let sign: UInt64 = 1 << UInt64(width - 1)
        let mask: UInt64 = width == 64 ? .max : (1 << UInt64(width)) - 1
        return (bits & sign) != 0 ? (~bits & mask) : (bits | sign)
    }

    static func compare(_ op: ExprBinaryOp, _ a: UInt64, _ b: UInt64) -> Bool {
        switch op {
        case .eq: return a == b
        case .ne: return a != b
        case .lt: return a < b
        case .le: return a <= b
        case .gt: return a > b
        default: return a >= b
        }
    }

    /// `x OP c` in totalOrder, for a non-NaN literal `c`, as one expression (the docs/EXPR.md table).
    static func totalOrderComparison(_ op: ExprBinaryOp, _ x: Expr, _ c: Double, _ t: ExprType) -> Expr {
        let lit = Expr.typedDouble(c, t)
        let sb = x.signBit
        if c == 0 {
            // ±0.0: the sign bit splits what the IEEE comparison calls equal.
            let negZero = c.sign == .minus
            let below = negZero ? (sb && Expr.binary(.ne, x, lit)) : sb              // x < c
            let atOrBelow = negZero ? sb : (sb || Expr.binary(.eq, x, lit))          // x <= c
            let equal = negZero ? (Expr.binary(.eq, x, lit) && sb) : (Expr.binary(.eq, x, lit) && !sb)
            switch op {
            case .eq: return equal
            case .ne: return !equal
            case .lt: return below
            case .le: return atOrBelow
            case .gt: return !atOrBelow
            default: return !below
            }
        }
        // Finite non-zero or infinite: the IEEE comparison is right except for NaN, which totalOrder
        // puts below -inf when its sign bit is set and above +inf when it is not.
        let negNaN = x.isNaN && sb, posNaN = x.isNaN && !sb
        switch op {
        case .eq, .ne: return .binary(op, x, lit)
        case .lt, .le: return .binary(op, x, lit) || negNaN
        default: return .binary(op, x, lit) || posNaN
        }
    }

    func booleans(_ a: AnyMetalArray?) -> [Bool?] {
        guard case .boolean(let b)? = a else { XCTFail("expected boolean column"); return [] }
        return b.toArray()
    }

    // MARK: - the two predicates

    func testSignBitAndIsNaNFloat64() throws {
        try requireRealGPU()
        for n in [1, 33, 4097, 300_007] {
            let v = Self.doubles(n)
            let rb = try MetalRecordBatch(names: ["x"], columns: [.float64(try MetalArray<Double>(v))])
            let r = try rb.query(query().project([("s", col("x").signBit), ("n", col("x").isNaN),
                                                  ("ns", !col("x").signBit)]))
            let s = booleans(r["s"]), nan = booleans(r["n"]), ns = booleans(r["ns"])
            for i in 0..<n {
                guard let x = v[i] else { XCTAssertNil(s[i]); XCTAssertNil(nan[i]); XCTAssertNil(ns[i]); continue }
                XCTAssertEqual(s[i], x.bitPattern >> 63 == 1, "signbit \(String(x.bitPattern, radix: 16)) n=\(n)")
                XCTAssertEqual(ns[i], x.bitPattern >> 63 == 0, "not signbit \(String(x.bitPattern, radix: 16))")
                XCTAssertEqual(nan[i], x.isNaN, "is_nan \(String(x.bitPattern, radix: 16)) n=\(n)")
            }
        }
    }

    func testSignBitAndIsNaNFloat32() throws {
        try requireRealGPU()
        for n in [1, 33, 4097, 300_007] {
            let v = Self.floats(n)
            let rb = try MetalRecordBatch(names: ["x"], columns: [.float32(try MetalArray<Float>(v))])
            let r = try rb.query(query().project([("s", col("x").signBit), ("n", col("x").isNaN)]))
            let s = booleans(r["s"]), nan = booleans(r["n"])
            for i in 0..<n {
                guard let x = v[i] else { XCTAssertNil(s[i]); XCTAssertNil(nan[i]); continue }
                XCTAssertEqual(s[i], x.bitPattern >> 31 == 1, "signbit \(String(x.bitPattern, radix: 16)) n=\(n)")
                XCTAssertEqual(nan[i], x.isNaN, "is_nan \(String(x.bitPattern, radix: 16)) n=\(n)")
            }
        }
    }

    func testIntegersLiteralsAndErrors() throws {
        try requireRealGPU()
        let iv: [Int32?] = [0, -1, 1, Int32.min, Int32.max, nil, -7]
        let uv: [UInt8?] = [0, 255, 128, nil, 1, 2, 3]
        let rb = try MetalRecordBatch(names: ["i", "u"], columns: [.int32(try MetalArray<Int32>(iv)),
                                                                   .uint8(try MetalArray<UInt8>(uv))])
        let r = try rb.query(query().project([
            ("is", col("i").signBit), ("in", col("i").isNaN), ("us", col("u").signBit), ("un", col("u").isNaN),
            ("lit", Expr.typedDouble(-0.0, .float64).signBit), ("lit32", Expr.typedDouble(-.nan, .float32).signBit),
            ("cast", col("i").cast(to: .float64).signBit),
        ]))
        let isb = booleans(r["is"]), inan = booleans(r["in"]), usb = booleans(r["us"]), unan = booleans(r["un"])
        let lit = booleans(r["lit"]), lit32 = booleans(r["lit32"]), cast = booleans(r["cast"])
        for i in 0..<iv.count {
            XCTAssertEqual(isb[i], iv[i].map { $0 < 0 })
            XCTAssertEqual(inan[i], iv[i].map { _ in false })
            XCTAssertEqual(usb[i], uv[i].map { _ in false })
            XCTAssertEqual(unan[i], uv[i].map { _ in false })
            XCTAssertEqual(lit[i], true)
            XCTAssertEqual(lit32[i], true)
            XCTAssertEqual(cast[i], iv[i].map { $0 < 0 })
        }
        let sb = try MetalRecordBatch(names: ["b"], columns: [.boolean(try MetalBooleanArray([true, false]))])
        XCTAssertThrowsError(try sb.query(query().project([("s", col("b").signBit)])))
        XCTAssertThrowsError(try sb.query(query().project([("s", col("b").isNaN)])))
    }

    func testTextRoundTrip() throws {
        let e = (col("x").signBit && col("x").isNaN) || !col("x").signBit
        XCTAssertEqual(e.description,
                       #"(or (and (signbit (col "x")) (is_nan (col "x"))) (not (signbit (col "x"))))"#)
        XCTAssertEqual(try Expr(text: e.description), e)
    }

    // MARK: - totalOrder comparisons in one expression

    func checkTotalOrder(width: Int) throws {
        let n = 4_099
        let t: ExprType = width == 64 ? .float64 : .float32
        let d = Self.doubles(n), f = Self.floats(n)
        let column: AnyMetalArray = width == 64 ? .float64(try MetalArray<Double>(d)) : .float32(try MetalArray<Float>(f))
        let bits: [UInt64?] = width == 64 ? d.map { $0.map(\.bitPattern) } : f.map { $0.map { UInt64($0.bitPattern) } }
        let rb = try MetalRecordBatch(names: ["x"], columns: [column])
        let literals: [Double] = [2.5, -2.5, 1, -1, 0.0, -0.0, .infinity, -.infinity, 5e-324, -5e-324]
        let ops: [ExprBinaryOp] = [.eq, .ne, .lt, .le, .gt, .ge]
        for c in literals {
            if width == 32 && c.magnitude == 5e-324 { continue }   // not a Float32 value
            let cBits = width == 64 ? c.bitPattern : UInt64(Float(c).bitPattern)
            let ck = Self.totalKey(cBits, width: width)
            var projections: [(String, Expr)] = []
            for op in ops { projections.append((op.rawValue, Self.totalOrderComparison(op, col("x"), c, t))) }
            let r = try rb.query(query().project(projections))
            for op in ops {
                let got = booleans(r[op.rawValue])
                for i in 0..<n {
                    guard let b = bits[i] else { XCTAssertNil(got[i], "null row \(op) \(c)"); continue }
                    let want = Self.compare(op, Self.totalKey(b, width: width), ck)
                    XCTAssertEqual(got[i], want, "f\(width) \(String(b, radix: 16)) \(op.rawValue) \(c)")
                }
            }
            // The same predicate as a filter keeps exactly the rows the reference keeps.
            let pred = Self.totalOrderComparison(.lt, col("x"), c, t)
            let kept = try rb.query(query().filter(pred).project([("x", col("x"))]))
            let wantCount = bits.filter { $0.map { Self.totalKey($0, width: width) < ck } ?? false }.count
            XCTAssertEqual(kept["x"]?.length, wantCount, "filter lt \(c) f\(width)")
        }
    }

    func testTotalOrderComparisonsFloat64() throws {
        try requireRealGPU()
        try checkTotalOrder(width: 64)
    }

    func testTotalOrderComparisonsFloat32() throws {
        try requireRealGPU()
        try checkTotalOrder(width: 32)
    }

    /// The plan JSON takes the same text: a filter node whose predicate is a totalOrder `x > 1.0`.
    func testPlanJSONFilterWithSignBit() throws {
        try requireRealGPU()
        let n = 4_099
        let d = Self.doubles(n)
        let row = AnyMetalArray.int32(try MetalArray<Int32>((0..<Int32(n)).map { $0 }))
        let batch = try MetalRecordBatch(names: ["x", "row"], columns: [.float64(try MetalArray<Double>(d)), row])
        let sources = ["t": PlanSource(name: "t", batch: batch)]
        let pred = Self.totalOrderComparison(.gt, col("x"), 1.0, .float64).description
        let escaped = pred.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
        let plan = #"{"op":"filter","predicate":""# + escaped + #"","input":{"op":"scan","source":"t"}}"#
        let out = try PlanJSON.run(plan, sources: sources)
        let one = Self.totalKey(1.0.bitPattern, width: 64)
        let want = (0..<n).filter { d[$0].map { Self.totalKey($0.bitPattern, width: 64) > one } ?? false }.map { Int32($0) }
        XCTAssertEqual(out["row"]!.asInt32!.toRawArray(), want)
        XCTAssertTrue(want.contains { d[Int($0)]!.isNaN }, "a positive NaN passes")
        XCTAssertFalse(want.contains { d[Int($0)]!.bitPattern >> 63 == 1 }, "no negative value or NaN passes")
    }
}
