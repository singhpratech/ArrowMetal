import XCTest
@testable import ArrowMetal

/// The GPU float parse behind `MetalStringArray.parse(Double.self)` / `parse(Float.self)` against the
/// CPU path it replaces (Swift's `Double(_:)` / `Float(_:)`), bit for bit, over a large randomized and
/// adversarial corpus. The CSV reader uses the same Eisel-Lemire core (`CSVFloatSource.fp_parse`).
final class CSVFloatParseTests: XCTestCase {

    /// Deterministic generator so a failure reproduces.
    struct SplitMix64 {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        mutating func int(_ n: Int) -> Int { Int(next() % UInt64(n)) }
    }

    /// Known hard inputs: halfway points, the subnormal boundary, the overflow boundary, long mantissas,
    /// every inf/nan spelling, and the shapes the GPU hands to the CPU.
    static let adversarial: [String] = [
        "0", "-0", "+0", "0.0", "-0.0", "00.5", "1", "-1", "+1", "1.", ".5", "-.5", "+.5", ".", "-", "+", "",
        "1e5", "1E+05", "1e-5", "1e", "1e+", "e5", "1.5.5", "1_0", " 1", "1 ", "\t1", "1\t",
        "0.1", "0.2", "0.3", "0.7", "1e23", "8.98846567431158e307", "7.3177701707893310e+15",
        "9007199254740992", "9007199254740993", "9007199254740994", "9007199254740995",
        "9007199254740993.0000000001", "9007199254740992.9999999999", "18014398509481985",
        "2.2250738585072011e-308", "2.2250738585072012e-308", "2.2250738585072014e-308",
        "2.225073858507201136057409796709131975934819546351645648023426109724822222021076945516529523908135087914149158913039621106870086438694594645527657207407820621743379988141063267329253552286881372149012981122451451889849057222307285255133155755015914397476397983411801999323962548289017107081850690630666655994938275772572015763062690663332647565300009245888316433037779791869612049497390377829704905051080609940730262937128958950003583799967207254304360284078895771796150945516748243471030702609144621572289880258182545180325707018860872113128079512233426288368622321503775666622503982534335974568884423900265498198385487948292206894721689831099698365846814022854243330660339850886445804001034933970427567186443383770486037861622771738545623065874679014086723327636718751234567890123456789012345678901234567890e-308",
        "4.9406564584124654e-324", "4.9e-324", "5e-324", "2.4703282292062327e-324", "2.4703282292062328e-324",
        "1e-320", "1e-400", "-1e-400", "1.7976931348623157e308", "1.7976931348623158e308",
        "1.7976931348623159e308", "1e308", "1e309", "1e400", "-1e400", "1e99999999999999999999",
        "0e99999999999999999999", "0.000000000000000000000000000000000000001e39",
        "123456789012345678901234567890", "0.1000000000000000055511151231257827021181583404541015625",
        "3.4028234663852886e38", "3.4028235677973366e38", "3.4028236e38", "1.1754943508222875e-38",
        "1.401298464324817e-45", "7.006492321624085e-46", "7.006492321624086e-46", "16777217", "16777219",
        "0.100000001490116119384765625", "0.100000001490116119384765624", "0.100000001490116119384765626",
        "inf", "-inf", "+inf", "Inf", "INF", "infinity", "Infinity", "INFINITY", "-infinity", "infin", "infx",
        "nan", "-nan", "+nan", "NaN", "NAN", "nan(123)", "nan()", "nan(", "snan", "-snan", "nanx", "na", "n",
        "0x1p3", "0x10", "0X1P-2", "-0x1.8p1", "1\u{0}", "\u{0}1", "abc", "NA", "null", "true", "١",
    ]

    /// Random text in the float alphabet: exercises every branch of the grammar, valid or not.
    static func junk(_ rng: inout SplitMix64, count: Int) -> [String] {
        let alphabet = Array("0123456789+-.eEinfatyINFATYxXpP() _s\t".utf8) + [0]
        return (0..<count).map { _ in
            let len = rng.int(14)
            return String(decoding: (0..<len).map { _ in alphabet[rng.int(alphabet.count)] }, as: UTF8.self)
        }
    }

    /// Random decimal strings: 1 to 30 digits, a decimal point anywhere, exponents across both formats'
    /// whole range and beyond.
    static func decimals(_ rng: inout SplitMix64, count: Int) -> [String] {
        (0..<count).map { _ in
            let nd = 1 + rng.int(30)
            var digits = (0..<nd).map { _ in Character(String(rng.int(10))) }
            if rng.int(4) == 0 { digits[0] = "0" }
            var s = String(digits)
            if rng.int(3) != 0 { s.insert(".", at: s.index(s.startIndex, offsetBy: rng.int(nd + 1))) }
            if rng.int(3) != 0 {
                let e = rng.int(720) - 360
                s += (rng.int(2) == 0 ? "e" : "E") + (e >= 0 && rng.int(2) == 0 ? "+" : "") + String(e)
            }
            if rng.int(4) == 0 { s = (rng.int(2) == 0 ? "-" : "+") + s }
            return s
        }
    }

    /// Random doubles printed several ways: shortest round-trip, 17 and 20+ significant digits, and
    /// exact halfway points between neighbours (odd integers in [2^53, 2^64) are halfway for Double).
    static func printed(_ rng: inout SplitMix64, count: Int) -> [String] {
        var out: [String] = []
        for k in 0..<count {
            let d = Double(bitPattern: rng.next())
            guard d.isFinite else { continue }
            switch k % 5 {
            case 0: out.append("\(d)")
            case 1: out.append(String(format: "%.17g", d))
            case 2: out.append(String(format: "%.25e", d))
            case 3:
                let odd = (UInt64(1) << 53) | (rng.next() >> 11) | 1
                out.append(String(odd << UInt64(rng.int(11))))
            default:
                let f = Float(bitPattern: UInt32(truncatingIfNeeded: rng.next()))
                if f.isFinite { out.append("\(f)"); out.append(String(format: "%.9g", Double(f))) }
                let odd = (UInt64(1) << 24) | (rng.next() >> 40) | 1
                out.append(String(odd))
            }
        }
        return out
    }

    static func corpus(seed: UInt64) -> [String?] {
        var rng = SplitMix64(state: seed)
        var all: [String?] = adversarial
        all += junk(&rng, count: 60_000)
        all += decimals(&rng, count: 120_000)
        all += printed(&rng, count: 100_000)
        // Some nulls, scattered.
        for i in stride(from: 7, to: all.count, by: 997) { all[i] = nil }
        return all
    }

    private func assertBitIdentical<T: ArrowPrimitive>(_ strings: [String?], _: T.Type,
                                                       file: StaticString = #filePath, line: UInt = #line) throws {
        let arr = try MetalStringArray(strings)
        let (gpu, hostRows) = try arr.parseFloatGPUCounting(T.self)
        let host = try arr.parseFloatHost(T.self)
        // The comparison only means something if the GPU decided the bulk of the corpus itself.
        let decided = strings.count - hostRows
        XCTAssertGreaterThan(decided, strings.count * 3 / 4, "\(T.self): only \(decided) of \(strings.count) decided on the GPU",
                             file: file, line: line)
        XCTAssertEqual(gpu.length, host.length, file: file, line: line)
        var mismatches = 0
        for i in 0..<strings.count {
            let g = gpu[i], h = host[i]
            let same: Bool
            switch (g, h) {
            case (nil, nil): same = true
            case (let a?, let b?):
                if T.self == Float.self { same = (a as! Float).bitPattern == (b as! Float).bitPattern }
                else { same = (a as! Double).bitPattern == (b as! Double).bitPattern }
            default: same = false
            }
            if !same {
                mismatches += 1
                if mismatches <= 20 {
                    XCTFail("\(T.self) row \(i) \(String(reflecting: strings[i])): gpu \(String(describing: g)) host \(String(describing: h))",
                            file: file, line: line)
                }
            }
        }
        XCTAssertEqual(mismatches, 0, "\(T.self): \(mismatches) of \(strings.count) rows differ", file: file, line: line)
        XCTAssertEqual(gpu.nullCount, host.nullCount, file: file, line: line)
    }

    func testAdversarialDouble() throws {
        try assertBitIdentical(Self.adversarial, Double.self)
    }

    func testAdversarialFloat() throws {
        try assertBitIdentical(Self.adversarial, Float.self)
    }

    func testRandomizedCorpusDouble() throws {
        try assertBitIdentical(Self.corpus(seed: 0xC5F0_0001), Double.self)
    }

    func testRandomizedCorpusFloat() throws {
        try assertBitIdentical(Self.corpus(seed: 0xC5F0_0002), Float.self)
    }

    /// The GPU decides the plain decimal form itself: numbers of up to 19 significant digits (every
    /// shortest round-trip print, every `%.17g`) never need the CPU. A 20th nonzero digit can straddle a
    /// rounding boundary, and those rows are the ones the GPU hands back.
    func testOrdinaryNumbersStayOnTheGPU() throws {
        var rng = SplitMix64(state: 42)
        let strings: [String?] = Self.printed(&rng, count: 20_000).filter { s in
            let mantissa = s.split(whereSeparator: { $0 == "e" || $0 == "E" }).first ?? ""
            return mantissa.drop(while: { !("1"..."9").contains($0) }).filter(\.isNumber).count <= 19
        }
        XCTAssertGreaterThan(strings.count, 15_000)
        let arr = try MetalStringArray(strings)
        let (gpu, hostRows) = try arr.parseFloatGPUCounting(Double.self)
        XCTAssertEqual(gpu.nullCount, 0)
        XCTAssertEqual(hostRows, 0, "ordinary decimal text should never need the CPU")
        let (gpu32, hostRows32) = try arr.parseFloatGPUCounting(Float.self)
        XCTAssertEqual(gpu32.nullCount, 0)
        XCTAssertEqual(hostRows32, 0)
        try assertBitIdentical(strings, Double.self)
        try assertBitIdentical(strings, Float.self)
    }

    func testPublicParseUsesTheSameContract() throws {
        let s = try MetalStringArray(["1.5", "abc", nil, "0x1p3", "-inf", "1e400"])
        let d = try s.parse(Double.self)
        XCTAssertEqual(d.toArray().map { $0.map { $0.bitPattern } },
                       [1.5, nil, nil, 8.0, -Double.infinity, Double.infinity].map { $0.map { $0.bitPattern } })
        XCTAssertThrowsError(try s.parse(Double.self, strict: true))
    }

    func testInsideABatch() throws {
        let ctx = MetalContext.shared
        let strings: [String?] = ["1.25", "0x10", "nan(7)", "2"]
        let s = try MetalStringArray(strings)
        let d = try ctx.batch { try s.parse(Double.self) }
        XCTAssertEqual(d[0], 1.25)
        XCTAssertEqual(d[1], 16)
        XCTAssertEqual(d[2]?.bitPattern, Double("nan(7)")?.bitPattern)
        XCTAssertEqual(d[3], 2)
    }
}
