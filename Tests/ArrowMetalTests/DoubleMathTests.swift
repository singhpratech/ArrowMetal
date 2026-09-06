import XCTest
@testable import ArrowMetal

/// Software binary64 on the GPU must match Swift's Double bit for bit for add/sub/mul, and within 1 ulp for div.
final class DoubleMathTests: XCTestCase {
    func randomDoubles(_ n: Int, _ g: inout SystemRandomNumberGenerator) -> [Double] {
        var out: [Double] = []
        out.reserveCapacity(n)
        let specials: [Double] = [0, -0.0, 1, -1, 0.5, 2, 1e308, -1e308, 5e-324, -5e-324, 2.2250738585072014e-308, 1.7976931348623157e308,
                                  .infinity, -.infinity, .nan, 3.141592653589793, 1e-310, 123456789.123456789, 0.1, 0.2, 0.3, 1024, 1.0000000000000002]
        for i in 0..<n {
            switch i % 5 {
            case 0: out.append(specials[Int.random(in: 0..<specials.count, using: &g)])
            case 1: out.append(Double(bitPattern: UInt64.random(in: 0...UInt64.max, using: &g)))       // any pattern incl. subnormal/NaN
            case 2: out.append(Double.random(in: -1e6...1e6, using: &g))
            case 3: out.append(Double(bitPattern: UInt64.random(in: 0...UInt64.max, using: &g) & 0x800F_FFFF_FFFF_FFFF)) // subnormals/zeros
            default: out.append(Double.random(in: -1...1, using: &g) * pow(10, Double(Int.random(in: -300...300, using: &g))))
            }
        }
        return out
    }

    func same(_ a: Double, _ b: Double) -> Bool { (a.isNaN && b.isNaN) || a.bitPattern == b.bitPattern }

    func testAddSubMulBitExact() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        let n = 1_000_000
        let a = randomDoubles(n, &g), b = randomDoubles(n, &g)
        let ma = try MetalArray<Double>(a), mb = try MetalArray<Double>(b)
        for (op, f) in [(ArithmeticOp.add, { (x: Double, y: Double) in x + y }), (.sub, { $0 - $1 }), (.mul, { $0 * $1 })] {
            let r = try ma.arithmetic(op, mb).toRawArray()
            var mismatches = 0, first = ""
            for i in 0..<n where !same(r[i], f(a[i], b[i])) {
                mismatches += 1
                if first.isEmpty { first = "\(op) \(a[i]) (\(String(a[i].bitPattern, radix: 16))) \(b[i]) (\(String(b[i].bitPattern, radix: 16))) gpu=\(r[i]) (\(String(r[i].bitPattern, radix: 16))) cpu=\(f(a[i], b[i])) (\(String(f(a[i], b[i]).bitPattern, radix: 16)))" }
            }
            XCTAssertEqual(mismatches, 0, "\(op): \(mismatches) mismatches, first: \(first)")
            // scalar form
            let s = b[7]
            let rs = try ma.arithmetic(op, s).toRawArray()
            XCTAssertTrue((0..<n).allSatisfy { same(rs[$0], f(a[$0], s)) }, "\(op) scalar")
        }
    }

    func testDivBitExact() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        let n = 300_000
        let a = randomDoubles(n, &g), b = randomDoubles(n, &g)
        let r = try MetalArray<Double>(a).divide(try MetalArray<Double>(b)).toRawArray()
        var exact = 0, oneUlp = 0, bad = 0, firstBad = ""
        for i in 0..<n {
            let e = a[i] / b[i]
            if same(r[i], e) { exact += 1; continue }
            if r[i].isFinite && e.isFinite && (r[i] == e.nextUp || r[i] == e.nextDown) { oneUlp += 1; continue }
            bad += 1
            if firstBad.isEmpty { firstBad = "\(a[i]) / \(b[i]) gpu=\(r[i]) cpu=\(e)" }
        }
        XCTAssertEqual(bad, 0, "division off by more than 1 ulp: \(bad), first: \(firstBad)")
        XCTAssertEqual(oneUlp, 0, "division off by 1 ulp: \(oneUlp)")
        XCTAssertEqual(exact, n)
    }

    func testDoubleSumOnGPU() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        for n in [0, 1, 7, 1000, 2_000_003] {
            var vals: [Double?] = []
            for i in 0..<n { vals.append(i % 11 == 0 ? nil : Double.random(in: -1000...1000, using: &g)) }
            let a = try MetalArray<Double>(vals)
            let s = try a.sum(), ref = CPUReference.sum(a)
            if let ref { XCTAssertEqual(s!.asDouble, ref.asDouble, accuracy: max(Double(n) * 1000 * 1e-15 * 4, 1e-9), "n=\(n)")   // reassociation error grows with n and magnitude } else { XCTAssertNil(s) }
            if let ref { XCTAssertEqual(try a.mean()!, ref.asDouble / Double(a.validCount), accuracy: 1e-9) } else { XCTAssertNil(try a.mean()) }
        }
        let big = try MetalArray<Double>([1e308, 1e308]); XCTAssertEqual(try big.sum(), .float(.infinity))
        let nan = try MetalArray<Double>([1, .nan, 2]); XCTAssertTrue(try nan.sum()!.asDouble.isNaN)
        let tiny = try MetalArray<Double>([Double](repeating: 5e-324, count: 4096)); XCTAssertEqual(try tiny.sum(), .float(5e-324 * 4096))
    }
}
