import XCTest
@testable import ArrowMetal

/// The fused expression compiler: every operator against a Swift oracle, at 0 / 1 / 33 / 4097 /
/// 1_000_003 rows, with and without nulls, batched and unbatched.
final class ExprTests: XCTestCase {

    let sizes = [0, 1, 33, 4097, 1_000_003]

    // MARK: helpers

    func i32(_ n: Int, nulls: Bool = false, seed: UInt64 = 1) -> ([Int32?], MetalArray<Int32>) {
        var rng = SplitMix(seed)
        var v: [Int32?] = []
        for i in 0..<n {
            if nulls && i % 7 == 3 { v.append(nil) } else { v.append(Int32(truncatingIfNeeded: Int64(rng.next() % 2000)) - 1000) }
        }
        return (v, try! MetalArray<Int32>(v))
    }
    func i64(_ n: Int, nulls: Bool = false, seed: UInt64 = 2) -> ([Int64?], MetalArray<Int64>) {
        var rng = SplitMix(seed)
        var v: [Int64?] = []
        for i in 0..<n {
            if nulls && i % 11 == 5 { v.append(nil) } else { v.append(Int64(rng.next() % 100000) - 50000) }
        }
        return (v, try! MetalArray<Int64>(v))
    }
    func f32(_ n: Int, nulls: Bool = false, seed: UInt64 = 3) -> ([Float?], MetalArray<Float>) {
        var rng = SplitMix(seed)
        var v: [Float?] = []
        for i in 0..<n {
            if nulls && i % 5 == 2 { v.append(nil) } else { v.append(Float(rng.next() % 100000) / 97.0 - 500) }
        }
        return (v, try! MetalArray<Float>(v))
    }
    func f64(_ n: Int, nulls: Bool = false, seed: UInt64 = 4) -> ([Double?], MetalArray<Double>) {
        var rng = SplitMix(seed)
        var v: [Double?] = []
        for i in 0..<n {
            if nulls && i % 9 == 4 { v.append(nil) } else { v.append(Double(rng.next() % 1000000) / 7919.0 - 60) }
        }
        return (v, try! MetalArray<Double>(v))
    }

    struct SplitMix {
        var s: UInt64
        init(_ seed: UInt64) { s = seed &* 0x9E3779B97F4A7C15 &+ 12345 }
        mutating func next() -> UInt64 {
            s = s &+ 0x9E3779B97F4A7C15
            var z = s
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
    }

    func batch(_ pairs: [(String, AnyMetalArray)]) throws -> MetalRecordBatch {
        try MetalRecordBatch(names: pairs.map(\.0), columns: pairs.map(\.1))
    }

    func optionals<T: ArrowPrimitive>(_ a: AnyMetalArray?, _: T.Type) throws -> [T?] {
        guard let a else { throw XCTSkip("missing column") }
        guard let m = unwrapAny(a, T.self) else {
            XCTFail("column is \(a.arrowFormat), expected \(T.arrowFormat)"); return []
        }
        return m.toArray()
    }

    func unwrapAny<T: ArrowPrimitive>(_ a: AnyMetalArray, _: T.Type) -> MetalArray<T>? {
        switch a {
        case .int8(let x): return x as? MetalArray<T>
        case .uint8(let x): return x as? MetalArray<T>
        case .int16(let x): return x as? MetalArray<T>
        case .uint16(let x): return x as? MetalArray<T>
        case .int32(let x): return x as? MetalArray<T>
        case .uint32(let x): return x as? MetalArray<T>
        case .int64(let x): return x as? MetalArray<T>
        case .uint64(let x): return x as? MetalArray<T>
        case .float32(let x): return x as? MetalArray<T>
        case .float64(let x): return x as? MetalArray<T>
        default: return nil
        }
    }

    func booleans(_ a: AnyMetalArray?) -> [Bool?] {
        guard case .boolean(let b)? = a else { XCTFail("expected boolean column"); return [] }
        return b.toArray()
    }

    // MARK: - element-wise operators, one at a time

    func testIntegerArithmetic() throws {
        try requireRealGPU()
        for n in sizes {
            let (av, aa) = i32(n, nulls: true), (bv, ba) = i32(n, nulls: true, seed: 9)
            let rb = try batch([("a", .int32(aa)), ("b", .int32(ba))])
            let r = try rb.query(query().project([
                ("add", col("a") + col("b")), ("sub", col("a") - col("b")),
                ("mul", col("a") * col("b")), ("div", col("a") / col("b")),
                ("neg", Expr.unary(.negate, col("a"))), ("abs", col("a").absolute),
            ]))
            let add = try optionals(r["add"], Int32.self)
            let sub = try optionals(r["sub"], Int32.self)
            let mul = try optionals(r["mul"], Int32.self)
            let div = try optionals(r["div"], Int32.self)
            let neg = try optionals(r["neg"], Int32.self)
            let abs = try optionals(r["abs"], Int32.self)
            for i in 0..<n {
                guard let x = av[i], let y = bv[i] else {
                    XCTAssertNil(add[i]); XCTAssertNil(mul[i]); continue
                }
                XCTAssertEqual(add[i], x &+ y, "row \(i)")
                XCTAssertEqual(sub[i], x &- y)
                XCTAssertEqual(mul[i], x &* y)
                XCTAssertEqual(div[i], y == 0 ? 0 : x / y)
                XCTAssertEqual(neg[i], 0 &- x)
                XCTAssertEqual(abs[i], x < 0 ? 0 &- x : x)
            }
        }
    }

    func testComparisonsAndLogic() throws {
        try requireRealGPU()
        for n in sizes {
            let (av, aa) = i32(n, nulls: true), (bv, ba) = i32(n, nulls: true, seed: 21)
            let rb = try batch([("a", .int32(aa)), ("b", .int32(ba))])
            let r = try rb.query(query().project([
                ("lt", col("a") < col("b")), ("ge", col("a") >= col("b")),
                ("eq", col("a") == col("b")), ("ne", col("a") != 0),
                ("and", (col("a") > 0) && (col("b") > 0)),
                ("or", (col("a") > 0) || (col("b") > 0)),
                ("not", !(col("a") > 0)),
                ("andk", Expr.binary(.andKleene, col("a") > 0, col("b") > 0)),
                ("ork", Expr.binary(.orKleene, col("a") > 0, col("b") > 0)),
                ("isnull", Expr.isNull(col("a"))),
            ]))
            let lt = booleans(r["lt"]), andv = booleans(r["and"]), andk = booleans(r["andk"])
            let ork = booleans(r["ork"]), isnull = booleans(r["isnull"]), notv = booleans(r["not"])
            for i in 0..<n {
                let pa = av[i].map { $0 > 0 }, pb = bv[i].map { $0 > 0 }
                if let x = av[i], let y = bv[i] { XCTAssertEqual(lt[i], x < y) } else { XCTAssertNil(lt[i]) }
                XCTAssertEqual(isnull[i], av[i] == nil)
                XCTAssertEqual(notv[i], pa.map { !$0 })
                if pa == nil || pb == nil { XCTAssertNil(andv[i]) } else { XCTAssertEqual(andv[i], pa! && pb!) }
                // Kleene
                if pa == false || pb == false { XCTAssertEqual(andk[i], false) }
                else if pa == nil || pb == nil { XCTAssertNil(andk[i]) }
                else { XCTAssertEqual(andk[i], true) }
                if pa == true || pb == true { XCTAssertEqual(ork[i], true) }
                else if pa == nil || pb == nil { XCTAssertNil(ork[i]) }
                else { XCTAssertEqual(ork[i], false) }
            }
        }
    }

    func testFloat64ArithmeticIsBitExact() throws {
        try requireRealGPU()
        for n in [0, 1, 33, 4097, 100_003] {
            let (av, aa) = f64(n, nulls: true), (bv, ba) = f64(n, nulls: false, seed: 77)
            let rb = try batch([("a", .float64(aa)), ("b", .float64(ba))])
            let r = try rb.query(query().project([
                ("add", col("a") + col("b")), ("sub", col("a") - col("b")),
                ("mul", col("a") * col("b")), ("div", col("a") / col("b")),
                ("neg", Expr.unary(.negate, col("a"))), ("abs", col("a").absolute),
                ("round", col("a").rounded),
            ]))
            let add = try optionals(r["add"], Double.self), sub = try optionals(r["sub"], Double.self)
            let mul = try optionals(r["mul"], Double.self), div = try optionals(r["div"], Double.self)
            let neg = try optionals(r["neg"], Double.self), ab = try optionals(r["abs"], Double.self)
            let rd = try optionals(r["round"], Double.self)
            for i in 0..<n {
                guard let x = av[i], let y = bv[i] else { XCTAssertNil(add[i]); continue }
                XCTAssertEqual(add[i]!.bitPattern, (x + y).bitPattern, "add row \(i)")
                XCTAssertEqual(sub[i]!.bitPattern, (x - y).bitPattern, "sub row \(i)")
                XCTAssertEqual(mul[i]!.bitPattern, (x * y).bitPattern, "mul row \(i)")
                XCTAssertEqual(div[i]!.bitPattern, (x / y).bitPattern, "div row \(i)")
                XCTAssertEqual(neg[i]!.bitPattern, (-x).bitPattern)
                XCTAssertEqual(ab[i]!.bitPattern, Swift.abs(x).bitPattern)
                XCTAssertEqual(rd[i]!.bitPattern, x.rounded(.toNearestOrAwayFromZero).bitPattern, "round row \(i)")
            }
        }
    }

    func testFloat64ComparisonsAndCasts() throws {
        try requireRealGPU()
        let vals: [Double?] = [0, -0.0, 1, -1, 1e300, -1e300, 5e-324, .infinity, -.infinity, .nan, 3.5, -3.5, nil]
        let a = try MetalArray<Double>(vals)
        let rb = try batch([("a", .float64(a))])
        let r = try rb.query(query().project([
            ("gt", col("a") > Expr.typedDouble(1.0, .float64)),
            ("eq", col("a") == Expr.typedDouble(0.0, .float64)),
            ("toi", col("a").cast(to: .int64)),
            ("tof", col("a").cast(to: .float32)),
            ("fromi", Expr.typedInt(1234567890123, .int64).cast(to: .float64) + col("a")),
        ]))
        let gt = booleans(r["gt"]), eq = booleans(r["eq"])
        let toi = try optionals(r["toi"], Int64.self), tof = try optionals(r["tof"], Float.self)
        let fromi = try optionals(r["fromi"], Double.self)
        for (i, v) in vals.enumerated() {
            guard let x = v else { XCTAssertNil(gt[i]); continue }
            XCTAssertEqual(gt[i], x > 1.0, "gt \(x)")
            XCTAssertEqual(eq[i], x == 0.0, "eq \(x)")
            if x.isFinite && Swift.abs(x) < 9e18 { XCTAssertEqual(toi[i], Int64(x.rounded(.towardZero)), "toi \(x)") }
            XCTAssertEqual(tof[i]!.bitPattern, Float(x).bitPattern, "tof \(x)")
            XCTAssertEqual(fromi[i]!.bitPattern, (Double(1234567890123) + x).bitPattern, "fromi \(x)")
        }
    }

    func testMixedTypePromotion() throws {
        try requireRealGPU()
        let n = 4097
        let (iv, ia) = i32(n, nulls: true), (fv, fa) = f32(n, nulls: true, seed: 31)
        let (dv, da) = f64(n, nulls: false, seed: 32)
        let rb = try batch([("i", .int32(ia)), ("f", .float32(fa)), ("d", .float64(da))])
        let r = try rb.query(query().project([
            ("if", col("i") + col("f")),        // int32 + float32 -> float32 (Arrow)
            ("id", col("i") * col("d")),        // int32 * float64 -> float64
            ("lit", col("i") + 1),              // an untyped literal takes the column's type
        ]))
        let iff = try optionals(r["if"], Float.self)
        let idd = try optionals(r["id"], Double.self)
        let lit = try optionals(r["lit"], Int32.self)
        for i in 0..<n {
            if let x = iv[i], let y = fv[i] {
                XCTAssertEqual(Double(iff[i]!), Double(Float(x) + y), accuracy: 1e-3)
            } else { XCTAssertNil(iff[i]) }
            if let x = iv[i] {
                XCTAssertEqual(idd[i]!.bitPattern, (Double(x) * dv[i]!).bitPattern, "row \(i)")
                XCTAssertEqual(lit[i], x &+ 1)
            } else { XCTAssertNil(idd[i]); XCTAssertNil(lit[i]) }
        }
    }

    func testConditionalAndNullHandling() throws {
        try requireRealGPU()
        for n in [0, 1, 33, 4097] {
            let (av, aa) = i32(n, nulls: true), (bv, ba) = i32(n, nulls: true, seed: 44)
            let rb = try batch([("a", .int32(aa)), ("b", .int32(ba))])
            let r = try rb.query(query().project([
                ("ie", Expr.ifElse(col("a") > 0, col("a"), col("b"))),
                ("co", Expr.coalesce([col("a"), col("b"), Expr.typedInt(-1, .int32)])),
                ("fn", col("a").fillNull(Expr.typedInt(7, .int32))),
                ("iv", Expr.isValid(col("a"))),
                ("in", col("a").isIn([Int64(0), 1, 2, -5])),
            ]))
            let ie = try optionals(r["ie"], Int32.self), co = try optionals(r["co"], Int32.self)
            let fn = try optionals(r["fn"], Int32.self)
            let iv = booleans(r["iv"]), isin = booleans(r["in"])
            for i in 0..<n {
                let cond = av[i].map { $0 > 0 }
                if cond == nil { XCTAssertNil(ie[i]) } else { XCTAssertEqual(ie[i], cond! ? av[i] : bv[i]) }
                XCTAssertEqual(co[i], av[i] ?? bv[i] ?? -1)
                XCTAssertEqual(fn[i], av[i] ?? 7)
                XCTAssertEqual(iv[i], av[i] != nil)
                XCTAssertEqual(isin[i], av[i].map { [0, 1, 2, -5].contains(Int($0)) } ?? false)
            }
        }
    }

    func testBitwiseAndShifts() throws {
        try requireRealGPU()
        let n = 4097
        let (av, aa) = i32(n, nulls: true), (bv, ba) = i32(n, nulls: false, seed: 55)
        let rb = try batch([("a", .int32(aa)), ("b", .int32(ba))])
        let r = try rb.query(query().project([
            ("and", col("a") & col("b")), ("or", col("a") | col("b")), ("xor", col("a") ^ col("b")),
            ("not", Expr.unary(.bitNot, col("a"))),
            ("shl", Expr.binary(.shl, col("a"), Expr.typedInt(3, .int32))),
            ("shr", Expr.binary(.shr, col("a"), Expr.typedInt(2, .int32))),
        ]))
        let and = try optionals(r["and"], Int32.self), orv = try optionals(r["or"], Int32.self)
        let xor = try optionals(r["xor"], Int32.self), notv = try optionals(r["not"], Int32.self)
        let shl = try optionals(r["shl"], Int32.self), shr = try optionals(r["shr"], Int32.self)
        for i in 0..<n {
            guard let x = av[i] else { XCTAssertNil(and[i]); continue }
            let y = bv[i]!
            XCTAssertEqual(and[i], x & y); XCTAssertEqual(orv[i], x | y); XCTAssertEqual(xor[i], x ^ y)
            XCTAssertEqual(notv[i], ~x)
            XCTAssertEqual(shl[i], x << 3)
            XCTAssertEqual(shr[i], x >> 2)
        }
    }

    func testMathUnaries() throws {
        try requireRealGPU()
        let n = 4097
        let (fv, fa) = f32(n, nulls: true, seed: 61)
        let rb = try batch([("f", .float32(fa))])
        let r = try rb.query(query().project([
            ("sqrt", col("f").absolute.squareRoot),
            ("exp", (col("f") / Expr.typedDouble(1000, .float32)).exponential),
            ("ln", (col("f").absolute + Expr.typedDouble(1, .float32)).naturalLog),
            ("round", col("f").rounded),
        ]))
        let sq = try optionals(r["sqrt"], Float.self), ex = try optionals(r["exp"], Float.self)
        let ln = try optionals(r["ln"], Float.self), rd = try optionals(r["round"], Float.self)
        for i in 0..<n {
            guard let x = fv[i] else { XCTAssertNil(sq[i]); continue }
            XCTAssertEqual(Double(sq[i]!), Double(Swift.abs(x)).squareRoot(), accuracy: 1e-3)
            XCTAssertEqual(Double(ex[i]!), Foundation.exp(Double(x) / 1000), accuracy: 1e-4)
            XCTAssertEqual(Double(ln[i]!), Foundation.log(Double(Swift.abs(x)) + 1), accuracy: 1e-4)
            XCTAssertEqual(rd[i], x.rounded(.toNearestOrAwayFromZero))
        }
    }

    func testStringPredicates() throws {
        try requireRealGPU()
        let words: [String?] = ["north", "south", "northwest", nil, "", "cust_042_east", "the north pole", "NORTH"]
        let s = try MetalStringArray(words)
        let rb = try batch([("s", .string(s))])
        let r = try rb.query(query().project([
            ("eq", col("s").stringEquals("north")),
            ("sw", col("s").startsWith("north")),
            ("ct", col("s").contains("orth")),
        ]))
        let eq = booleans(r["eq"]), sw = booleans(r["sw"]), ct = booleans(r["ct"])
        for (i, w) in words.enumerated() {
            guard let w else { XCTAssertNil(eq[i]); continue }
            XCTAssertEqual(eq[i], w == "north", w)
            XCTAssertEqual(sw[i], w.hasPrefix("north"), w)
            XCTAssertEqual(ct[i], w.contains("orth"), w)
        }
    }

    // MARK: - deep trees and CSE

    func testDeepMixedTreeWithSharedSubtrees() throws {
        try requireRealGPU()
        for n in sizes {
            let (av, aa) = f32(n, nulls: true, seed: 101)
            let (bv, ba) = f32(n, nulls: true, seed: 102)
            let (cv, ca) = i32(n, nulls: true, seed: 103)
            let rb = try batch([("a", .float32(aa)), ("b", .float32(ba)), ("c", .int32(ca))])
            // shared subtree (a * 2 + b), used four times; > 20 nodes in total
            let shared = col("a") * Expr.typedDouble(2, .float32) + col("b")
            let e = Expr.ifElse(shared > Expr.typedDouble(0, .float32),
                                shared * shared - col("c").cast(to: .float32),
                                (shared.absolute + Expr.typedDouble(1, .float32)).naturalLog + shared)
                .fillNull(Expr.typedDouble(-1, .float32))
            XCTAssertGreaterThan(e.nodeCount, 20)
            let r = try rb.query(query().project([("out", e)]))
            let got = try optionals(r["out"], Float.self)
            for i in 0..<n {
                let s: Float? = (av[i] != nil && bv[i] != nil) ? av[i]! * 2 + bv[i]! : nil
                var expect: Float? = nil
                if let s {
                    if s > 0 { if let c = cv[i] { expect = s * s - Float(c) } }
                    else { expect = Foundation.logf(Swift.abs(s) + 1) + s }
                }
                let want = expect ?? -1
                XCTAssertEqual(got[i]!, want, accuracy: Swift.max(1e-3, Swift.abs(want) * 1e-5), "row \(i)")
            }
        }
    }

    // MARK: - terminals

    func testFilterProject() throws {
        try requireRealGPU()
        for n in sizes {
            let (av, aa) = i32(n, nulls: true, seed: 201)
            let (bv, ba) = f32(n, nulls: true, seed: 202)
            let rb = try batch([("a", .int32(aa)), ("b", .float32(ba))])
            let pred = (col("a") > 0) && (col("b") > Expr.typedDouble(0, .float32))
            let r = try rb.query(query().filter(pred).project([
                ("a", col("a")), ("двa", col("a") * 2), ("b", col("b")),
            ]))
            var wantA: [Int32] = [], wantB: [Float?] = []
            for i in 0..<n {
                guard let x = av[i], let y = bv[i], x > 0, y > 0 else { continue }
                wantA.append(x); wantB.append(y)
            }
            let gotA = try optionals(r["a"], Int32.self)
            let gotD = try optionals(r["двa"], Int32.self)
            let gotB = try optionals(r["b"], Float.self)
            XCTAssertEqual(gotA.count, wantA.count, "n=\(n)")
            for i in 0..<wantA.count {
                XCTAssertEqual(gotA[i], wantA[i])
                XCTAssertEqual(gotD[i], wantA[i] &* 2)
                XCTAssertEqual(gotB[i], wantB[i])
            }
        }
    }

    func testAggregates() throws {
        try requireRealGPU()
        for n in sizes {
            let (av, aa) = i64(n, nulls: true, seed: 301)
            let (bv, ba) = f32(n, nulls: true, seed: 302)
            let rb = try batch([("a", .int64(aa)), ("b", .float32(ba))])
            let r = try rb.query(query().aggregate([
                ExprAggregate(.sum, col("a"), name: "sa"),
                ExprAggregate(.min, col("a"), name: "mi"),
                ExprAggregate(.max, col("a"), name: "ma"),
                ExprAggregate(.count, col("a"), name: "ca"),
                ExprAggregate(.count, nil, name: "rows"),
                ExprAggregate(.mean, col("a"), name: "me"),
                ExprAggregate(.sum, col("b"), name: "sb"),
            ]))
            var sum: Int64 = 0, cnt = 0
            var mn = Int64.max, mx = Int64.min
            for v in av { if let v { sum &+= v; cnt += 1; mn = Swift.min(mn, v); mx = Swift.max(mx, v) } }
            var fsum = 0.0
            for v in bv { if let v { fsum += Double(v) } }
            if cnt == 0 {
                XCTAssertEqual(r.scalar("sa"), .null)
            } else {
                XCTAssertEqual(r.scalar("sa"), .int(sum))
                XCTAssertEqual(r.scalar("mi"), .int(mn))
                XCTAssertEqual(r.scalar("ma"), .int(mx))
                XCTAssertEqual(r.scalar("me")!.asDouble!, Double(sum) / Double(cnt), accuracy: 1e-9)
                XCTAssertEqual(r.scalar("sb")!.asDouble!, fsum, accuracy: Swift.abs(fsum) * 1e-9 + 1e-6)
            }
            XCTAssertEqual(r.scalar("ca"), .int(Int64(cnt)))
            XCTAssertEqual(r.scalar("rows"), .int(Int64(n)))
        }
    }

    func testAggregateUnderPredicate() throws {
        try requireRealGPU()
        for n in sizes {
            let (av, aa) = i64(n, nulls: true, seed: 311)
            let (rv, ra) = i32(n, nulls: false, seed: 312)
            let rb = try batch([("amount", .int64(aa)), ("region", .int32(ra))])
            let q = query().filter((col("region") > 0) && (col("amount") > 100)).sum(col("amount"))
            let r = try rb.query(q)
            var sum: Int64 = 0, any = false
            for i in 0..<n {
                guard let a = av[i], let rr = rv[i], rr > 0, a > 100 else { continue }
                sum &+= a; any = true
            }
            XCTAssertEqual(r.onlyScalar, any ? .int(sum) : .null, "n=\(n)")
        }
    }

    func testGroupBy() throws {
        try requireRealGPU()
        for n in [0, 1, 33, 4097, 1_000_003] {
            let K = 97
            var rng = SplitMix(401)
            var keys: [Int32?] = [], vals: [Int32?] = []
            for i in 0..<n {
                keys.append(i % 13 == 6 ? nil : Int32(rng.next() % UInt64(K + 3)) - 1)
                vals.append(i % 7 == 2 ? nil : Int32(rng.next() % 1000))
            }
            let rb = try batch([("k", .int32(try MetalArray<Int32>(keys))),
                                ("v", .int32(try MetalArray<Int32>(vals)))])
            let r = try rb.query(query().groupBy(col("k"), keyCount: K).aggregate([
                ExprAggregate(.sum, col("v") * 2, name: "s"),
                ExprAggregate(.count, col("v"), name: "c"),
                ExprAggregate(.min, col("v"), name: "mn"),
                ExprAggregate(.max, col("v"), name: "mx"),
            ]))
            var sums = [Int64](repeating: 0, count: K), counts = [Int64](repeating: 0, count: K)
            var mins = [Int32](repeating: .max, count: K), maxs = [Int32](repeating: .min, count: K)
            for i in 0..<n {
                guard let k = keys[i], k >= 0, k < Int32(K), let v = vals[i] else { continue }
                sums[Int(k)] &+= Int64(v) * 2
                counts[Int(k)] += 1
                mins[Int(k)] = Swift.min(mins[Int(k)], v)
                maxs[Int(k)] = Swift.max(maxs[Int(k)], v)
            }
            let gs = try optionals(r["s"], Int64.self), gc = try optionals(r["c"], Int64.self)
            let gmn = try optionals(r["mn"], Int32.self), gmx = try optionals(r["mx"], Int32.self)
            for k in 0..<K {
                XCTAssertEqual(gc[k], counts[k], "count key \(k) n=\(n)")
                if counts[k] == 0 { XCTAssertNil(gs[k]); XCTAssertNil(gmn[k]); continue }
                XCTAssertEqual(gs[k], sums[k], "sum key \(k) n=\(n)")
                XCTAssertEqual(gmn[k], mins[k]); XCTAssertEqual(gmx[k], maxs[k])
            }
        }
    }

    func testGroupByLargeKeySpaceUsesDeviceTables() throws {
        try requireRealGPU()
        let K = 5000, n = 200_003
        var rng = SplitMix(501)
        var keys: [Int32] = [], vals: [Float] = []
        for _ in 0..<n { keys.append(Int32(rng.next() % UInt64(K))); vals.append(Float(rng.next() % 1000) / 8) }
        let rb = try batch([("k", .int32(try MetalArray<Int32>(keys))),
                            ("v", .float32(try MetalArray<Float>(vals)))])
        let r = try rb.query(query().groupBy(col("k"), keyCount: K).sum(col("v"), name: "s"))
        var want = [Double](repeating: 0, count: K)
        for i in 0..<n { want[Int(keys[i])] += Double(vals[i]) }
        let got = try optionals(r["s"], Double.self)
        for k in 0..<K { XCTAssertEqual(got[k] ?? 0, want[k], accuracy: Swift.abs(want[k]) * 1e-4 + 1e-3, "key \(k)") }
    }

    // MARK: - batching, caching, errors

    func testBatchedMatchesUnbatched() throws {
        try requireRealGPU()
        let n = 100_003
        let (_, aa) = i64(n, nulls: true, seed: 601)
        let (_, ra) = i32(n, nulls: false, seed: 602)
        let rb = try batch([("amount", .int64(aa)), ("region", .int32(ra))])
        let q = query().filter((col("region") > 0) && (col("amount") > 100)).sum(col("amount"))
        let plain = try rb.query(q).onlyScalar
        let batched = try MetalContext.shared.batch { try rb.query(q).onlyScalar }
        XCTAssertEqual(plain, batched)

        // A filtered project inside a batch produces pending arrays whose length the GPU decides.
        let pq = query().filter(col("region") > 0).project([("amount", col("amount"))])
        let direct = try rb.query(pq)
        let inBatch = try MetalContext.shared.batch { try rb.query(pq) }
        XCTAssertEqual(direct["amount"]!.length, inBatch["amount"]!.length)
        let a = try optionals(direct["amount"], Int64.self), b = try optionals(inBatch["amount"], Int64.self)
        XCTAssertEqual(a, b)
    }

    func testChainedQueriesInOneBatch() throws {
        try requireRealGPU()
        let n = 50_021
        let (av, aa) = i32(n, nulls: true, seed: 701)
        let rb = try batch([("a", .int32(aa))])
        let sum = try MetalContext.shared.batch { () -> ExprScalar? in
            let kept = try rb.query(query().filter(col("a") > 0).project([("a", col("a"))]))
            let inner = try MetalRecordBatch(names: kept.names, columns: kept.columns)
            return try inner.query(query().sum(col("a"))).onlyScalar
        }
        var want: Int64 = 0, any = false
        for v in av { if let v, v > 0 { want &+= Int64(v); any = true } }
        XCTAssertEqual(sum, any ? .int(want) : .null)
    }

    func testPipelineCacheHit() throws {
        try requireRealGPU()
        let n = 1000
        let (_, aa) = i32(n, nulls: true, seed: 801)
        let rb = try batch([("a", .int32(aa))])
        let q = query().filter(col("a") > 3).sum(col("a") * 7 + 1)
        _ = try rb.query(q)
        let before = ExprCompiler.compileCount
        for _ in 0..<5 { _ = try rb.query(q) }
        XCTAssertEqual(ExprCompiler.compileCount, before, "the same query shape must not recompile")
        // A different shape does compile.
        _ = try rb.query(query().filter(col("a") > 4).sum(col("a") * 7 + 1))
        XCTAssertGreaterThan(ExprCompiler.compileCount, before)
    }

    func testErrors() throws {
        let n = 16
        let (_, aa) = i32(n)
        let s = try MetalStringArray((0..<n).map { "s\($0)" })
        let rb = try batch([("a", .int32(aa)), ("s", .string(s))])
        // unknown column
        XCTAssertThrowsError(try rb.query(query().project([("x", col("nope"))]))) {
            XCTAssertTrue("\($0)".contains("nope"), "\($0)")
        }
        // boolean where numeric is needed
        XCTAssertThrowsError(try rb.query(query().project([("x", col("a") + col("s"))]))) {
            XCTAssertTrue("\($0)".contains("utf8"), "\($0)")
        }
        // string output
        XCTAssertThrowsError(try rb.query(query().project([("x", col("s"))]))) {
            XCTAssertTrue("\($0)".contains("utf8"), "\($0)")
        }
        // non-boolean filter
        XCTAssertThrowsError(try rb.query(query().filter(col("a")).sum(col("a")))) {
            XCTAssertTrue("\($0)".contains("boolean"), "\($0)")
        }
        // group-by 64-bit min
        XCTAssertThrowsError(try rb.query(query().groupBy(col("a"), keyCount: 4)
            .min(col("a").cast(to: .int64)))) {
            XCTAssertTrue("\($0)".contains("64-bit atomics"), "\($0)")
        }
    }

    // MARK: - serialisation round trip

    func testCanonicalTextRoundTrip() throws {
        let e = ((col("a") * Expr.typedDouble(2, .float32) + col("b")) > 100.0)
            && Expr.isValid(col("c")) && col("d").startsWith("north")
        let text = e.description
        let back = try Expr(text: text)
        XCTAssertEqual(back.description, text)
        let q = query().filter(e).groupBy(col("k"), keyCount: 8).sum(col("v"), name: "s")
        let qt = q.canonical
        XCTAssertEqual(try ExprQuery(text: qt).canonical, qt)
    }
}
