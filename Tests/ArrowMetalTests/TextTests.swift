import XCTest
import Foundation
@testable import ArrowMetal

/// Regex, `LIKE`, splitting, the string ↔ number casts and temporal rounding / arithmetic.
///
/// Every regex assertion is checked against `NSRegularExpression` run directly on a Swift array, so
/// the GPU fast paths (a literal pattern routed to `contains`, `^literal` to `startsWith`, a pure
/// prefix/suffix `LIKE` to `startsWith`/`endsWith`) are proved to agree with the CPU engine rather
/// than merely with themselves. Temporal rounding is checked against Foundation's `Calendar` in UTC.
final class TextTests: XCTestCase {

    /// Deterministic so a failure can be reproduced.
    struct TextRNG: RandomNumberGenerator {
        var state: UInt64
        init(_ seed: UInt64) { state = seed &* 0x9E3779B97F4A7C15 &+ 1 }
        mutating func next() -> UInt64 {
            state = state &+ 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
    }

    /// The sizes every shape-sensitive test walks: empty, one row, a partial bitmap word, several
    /// threadgroups, and a size that is no multiple of anything.
    static let sizes = [0, 1, 33, 4097, 200_003]

    // MARK: - Sample data

    /// Random ASCII-and-beyond strings with a scattering of nulls. Deliberately full of the pieces
    /// the patterns below look for, so matches are common rather than rare.
    func sampleStrings(_ n: Int, seed: UInt64, nullEvery: Int = 7) -> [String?] {
        var g = TextRNG(seed)
        let pieces = ["ab", "abc", "a", "b", "", " ", "  ", "12", "007", "-3", "x", "AB",
                      "ab.c", "cafe\u{301}", "\u{00E9}", "a\nb", "\t", "zz"]
        var out: [String?] = []
        out.reserveCapacity(n)
        for i in 0..<n {
            if nullEvery > 0 && i % nullEvery == 3 { out.append(nil); continue }
            var s = ""
            let parts = Int(g.next() % 4)
            for _ in 0...parts { s += pieces[Int(g.next() % UInt64(pieces.count))] }
            out.append(s)
        }
        return out
    }

    func oracleRegex(_ pattern: String, ignoreCase: Bool = false) throws -> NSRegularExpression {
        try NSRegularExpression(pattern: pattern, options: ignoreCase ? [.caseInsensitive] : [])
    }

    func full(_ s: String) -> NSRange { NSRange(s.startIndex..., in: s) }

    // MARK: - Regex against NSRegularExpression

    func testMatchSubstringRegexMatchesOracle() throws {
        try requireRealGPU()
        // "ab" and "^ab" take the GPU fast path; the rest exercise the CPU engine.
        let patterns = ["ab", "^ab", "a.c", "a+b", "(ab|zz)", "ab$", "[0-9]+", "^$", "\\d{2}"]
        for n in Self.sizes {
            let values = sampleStrings(n, seed: UInt64(n) &+ 11)
            let array = try MetalStringArray(values)
            for p in patterns {
                let re = try oracleRegex(p)
                let expected = values.map { s in s.map { re.firstMatch(in: $0, range: full($0)) != nil } }
                let got = try array.matchSubstringRegex(p).toArray()
                XCTAssertEqual(got, expected, "match_substring_regex \(p) at n=\(n)")
            }
        }
    }

    func testMatchSubstringRegexIgnoreCase() throws {
        try requireRealGPU()
        let values = sampleStrings(4097, seed: 21)
        let array = try MetalStringArray(values)
        let re = try oracleRegex("AB", ignoreCase: true)
        let expected = values.map { s in s.map { re.firstMatch(in: $0, range: full($0)) != nil } }
        XCTAssertEqual(try array.matchSubstringRegex("AB", ignoreCase: true).toArray(), expected)
    }

    /// The documented reason `$` is kept off the GPU fast path: ICU matches it before a final newline.
    func testDollarAnchorStaysOnTheCPU() throws {
        try requireRealGPU()
        let array = try MetalStringArray(["ab", "ab\n", "ab\nx", "zab"])
        XCTAssertEqual(try array.matchSubstringRegex("ab$").toArray(), [true, true, false, true])
        XCTAssertEqual(try array.endsWith("ab").toArray(), [true, false, false, true])
        XCTAssertNil(MetalStringArray.literalPredicate("ab$", ignoreCase: false))
    }

    func testCountAndFindSubstringRegexMatchOracle() throws {
        try requireRealGPU()
        let patterns = ["ab", "a.", "[0-9]", "(ab)+"]
        for n in [0, 1, 33, 4097] {
            let values = sampleStrings(n, seed: UInt64(n) &+ 31)
            let array = try MetalStringArray(values)
            for p in patterns {
                let re = try oracleRegex(p)
                let counts = values.map { s in s.map { Int32(re.numberOfMatches(in: $0, range: full($0))) } }
                XCTAssertEqual(try array.countSubstringRegex(p).toArray(), counts, "count \(p) n=\(n)")
                let finds: [Int32?] = values.map { s in
                    guard let s else { return nil }
                    guard let m = re.firstMatch(in: s, range: full(s)), let r = Range(m.range, in: s) else { return -1 }
                    return Int32(s.utf8.distance(from: s.utf8.startIndex, to: r.lowerBound))
                }
                XCTAssertEqual(try array.findSubstringRegex(p).toArray(), finds, "find \(p) n=\(n)")
            }
        }
    }

    func testReplaceSubstringRegexMatchesOracle() throws {
        try requireRealGPU()
        for n in [0, 1, 33, 4097] {
            let values = sampleStrings(n, seed: UInt64(n) &+ 41)
            let array = try MetalStringArray(values)
            // A literal pattern (GPU) and a real regex with a capture group (CPU).
            let re = try oracleRegex("ab")
            let expectedLiteral = values.map { s in
                s.map { re.stringByReplacingMatches(in: $0, range: full($0), withTemplate: "XY") }
            }
            XCTAssertEqual(try array.replaceSubstringRegex("ab", with: "XY").toArray(), expectedLiteral)

            let re2 = try oracleRegex("(a)(b)")
            let expectedGroups = values.map { s in
                s.map { re2.stringByReplacingMatches(in: $0, range: full($0), withTemplate: "$2$1") }
            }
            XCTAssertEqual(try array.replaceSubstringRegex("(a)(b)", with: "$2$1").toArray(), expectedGroups)
        }
    }

    func testReplaceSubstringRegexRespectsMaxReplacements() throws {
        try requireRealGPU()
        let array = try MetalStringArray(["ababab", nil, "ab", ""])
        XCTAssertEqual(try array.replaceSubstringRegex("ab", with: "-", maxReplacements: 2).toArray(),
                       ["--ab", nil, "-", ""])
        XCTAssertEqual(try array.replaceSubstringRegex("(a)b", with: "$1!", maxReplacements: 1).toArray(),
                       ["a!abab", nil, "a!", ""])
    }

    func testExtractRegexNamedGroups() throws {
        try requireRealGPU()
        let values: [String?] = ["2024-05-06", "not a date", nil, "1999-12-31", ""]
        let array = try MetalStringArray(values)
        let groups = try array.extractRegex("(?<y>[0-9]{4})-(?<m>[0-9]{2})-(?<d>[0-9]{2})")
        XCTAssertEqual(Set(groups.keys), ["y", "m", "d"])
        XCTAssertEqual(groups["y"]!.toArray(), ["2024", nil, nil, "1999", nil])
        XCTAssertEqual(groups["m"]!.toArray(), ["05", nil, nil, "12", nil])
        XCTAssertEqual(groups["d"]!.toArray(), ["06", nil, nil, "31", nil])
        XCTAssertThrowsError(try array.extractRegex("[0-9]+"))
        XCTAssertEqual(MetalStringArray.namedGroups(in: "(?<a>x)(?<=y)(?<b>z)\\(?<c>"), ["a", "b"])
    }

    // MARK: - SQL LIKE

    func testMatchLikeSemantics() throws {
        try requireRealGPU()
        let values: [String?] = ["abc", "abcd", "xabc", "", "a", nil, "abc\n", "a_c", "a%c"]
        let array = try MetalStringArray(values)
        XCTAssertEqual(try array.matchLike("abc").toArray(),
                       [true, false, false, false, false, nil, false, false, false])
        XCTAssertEqual(try array.matchLike("abc%").toArray(),
                       [true, true, false, false, false, nil, true, false, false])
        XCTAssertEqual(try array.matchLike("%abc").toArray(),
                       [true, false, true, false, false, nil, false, false, false])
        XCTAssertEqual(try array.matchLike("%b%").toArray(),
                       [true, true, true, false, false, nil, true, false, false])
        XCTAssertEqual(try array.matchLike("a_c").toArray(),
                       [true, false, false, false, false, nil, false, true, true])
        // A backslash makes a wildcard literal.
        XCTAssertEqual(try array.matchLike("a\\_c").toArray(),
                       [false, false, false, false, false, nil, false, true, false])
        XCTAssertEqual(try array.matchLike("a\\%c").toArray(),
                       [false, false, false, false, false, nil, false, false, true])
        XCTAssertEqual(try array.matchLike("%").toArray(),
                       [true, true, true, true, true, nil, true, true, true])
        XCTAssertEqual(try array.matchLike("").toArray(),
                       [false, false, false, true, false, nil, false, false, false])
    }

    /// The GPU predicates and the translated regex must agree, newline-terminated values included.
    func testMatchLikeFastPathAgreesWithRegex() throws {
        try requireRealGPU()
        let values = sampleStrings(4097, seed: 55)
        let array = try MetalStringArray(values)
        for pattern in ["ab", "ab%", "%ab", "%ab%", "%"] {
            let tokens = MetalStringArray.parseLike(pattern)
            XCTAssertNotNil(MetalStringArray.likePredicate(tokens), "\(pattern) should take the GPU path")
            let re = try NSRegularExpression(pattern: MetalStringArray.likeRegex(tokens),
                                             options: [.dotMatchesLineSeparators])
            let expected = values.map { s in s.map { re.firstMatch(in: $0, range: full($0)) != nil } }
            XCTAssertEqual(try array.matchLike(pattern).toArray(), expected, "match_like \(pattern)")
        }
    }

    // MARK: - Splitting

    func testSplitPatternMatchesComponents() throws {
        try requireRealGPU()
        for n in [0, 1, 33, 4097] {
            let values = sampleStrings(n, seed: UInt64(n) &+ 61)
            let array = try MetalStringArray(values)
            let (offsets, pieces) = try array.splitPatternPair("b")
            let offs = offsets.toArray().map { Int($0!) }
            XCTAssertEqual(offs.count, n + 1)
            let flat = pieces.toArray()
            for i in 0..<n {
                let got = Array(flat[offs[i]..<offs[i + 1]]).map { $0! }
                let expected = values[i].map { $0.components(separatedBy: "b") } ?? []
                XCTAssertEqual(got, expected, "row \(i) of \(n)")
            }
        }
        XCTAssertThrowsError(try MetalStringArray(["a"]).splitPattern(""))
    }

    func testSplitPatternMaxSplits() throws {
        try requireRealGPU()
        let array = try MetalStringArray(["a,b,c,d"])
        func pieces(_ r: MetalStringArray.SplitResult) -> [String] { r.values.toArray().map { $0! } }
        XCTAssertEqual(pieces(try array.splitPatternPair(",", maxSplits: 1)), ["a", "b,c,d"])
        XCTAssertEqual(pieces(try array.splitPatternPair(",", maxSplits: 1, reverse: true)), ["a,b,c", "d"])
        XCTAssertEqual(pieces(try array.splitPatternPair(",")), ["a", "b", "c", "d"])
    }

    /// Arrow's whitespace split cuts at every maximal whitespace run, so a leading or trailing run
    /// leaves an empty piece behind and the empty string splits to one empty piece. (Python's
    /// no-argument `str.split()` drops those; `str.split(sep)` does not, and Arrow follows the latter.)
    func testSplitWhitespaceKeepsEndPieces() throws {
        try requireRealGPU()
        let values: [String?] = ["a b  c", "  lead", "trail  ", "   ", "", nil, "one"]
        let array = try MetalStringArray(values)
        let (offsets, pieces) = try array.splitWhitespacePair()
        let offs = offsets.toArray().map { Int($0!) }
        let flat = pieces.toArray().map { $0! }
        let expected = [["a", "b", "c"], ["", "lead"], ["trail", ""], ["", ""], [""], [], ["one"]]
        for i in 0..<values.count {
            XCTAssertEqual(Array(flat[offs[i]..<offs[i + 1]]), expected[i], "row \(i)")
        }
        XCTAssertEqual(MetalStringArray.splitWhitespace(" a b  ", unicode: false, maxSplits: 1, reverse: false),
                       ["", "a b  "])
        XCTAssertEqual(MetalStringArray.splitWhitespace(" a b  ", unicode: false, maxSplits: 1, reverse: true),
                       [" a b", ""])
    }

    func testSplitPatternRegex() throws {
        try requireRealGPU()
        let array = try MetalStringArray(["a1b22c", nil, "abc"])
        let (offsets, pieces) = try array.splitPatternRegexPair("[0-9]+")
        XCTAssertEqual(offsets.toArray(), [0, 3, 3, 4])
        XCTAssertEqual(pieces.toArray(), ["a", "b", "c", "abc"])
        XCTAssertThrowsError(try array.splitPatternRegex("[0-9]+", maxSplits: 1, reverse: true))
    }

    // MARK: - Integer ↔ string

    func checkIntegerRoundTrip<T>(_: T.Type, seed: UInt64) throws
    where T: ArrowPrimitive & FixedWidthInteger {
        var g = TextRNG(seed)
        for n in [0, 1, 33, 4097] {
            var values: [T?] = [T.min, T.max, 0, 1]
            if T.isSigned { values.append(contentsOf: [-1, T.min + 1]) }
            while values.count < n { values.append(values.count % 5 == 2 ? nil : T.random(in: T.min...T.max, using: &g)) }
            values = Array(values.prefix(n))
            let array = try MetalArray<T>(values)
            let text = try array.toStrings()
            XCTAssertEqual(text.toArray(), values.map { $0.map { String($0) } }, "\(T.self) itoa n=\(n)")
            XCTAssertEqual(try text.parse(T.self).toArray(), values, "\(T.self) round trip n=\(n)")
        }
    }

    func testIntegerToStringsAndBack() throws {
        try requireRealGPU()
        try checkIntegerRoundTrip(Int8.self, seed: 1)
        try checkIntegerRoundTrip(UInt8.self, seed: 2)
        try checkIntegerRoundTrip(Int16.self, seed: 3)
        try checkIntegerRoundTrip(UInt16.self, seed: 4)
        try checkIntegerRoundTrip(Int32.self, seed: 5)
        try checkIntegerRoundTrip(UInt32.self, seed: 6)
        try checkIntegerRoundTrip(Int64.self, seed: 7)
        try checkIntegerRoundTrip(UInt64.self, seed: 8)
    }

    func testIntegerToStringsLargeArray() throws {
        try requireRealGPU()
        let n = 200_003
        var g = TextRNG(99)
        var values: [Int64?] = []
        values.reserveCapacity(n)
        for i in 0..<n { values.append(i % 11 == 4 ? nil : Int64.random(in: Int64.min...Int64.max, using: &g)) }
        let text = try MetalArray<Int64>(values).toStrings()
        XCTAssertEqual(text.length, n)
        XCTAssertEqual(text.nullCount, values.filter { $0 == nil }.count)
        XCTAssertEqual(try text.parse(Int64.self).toArray(), values)
    }

    func checkParseEdges<T>(_: T.Type) throws where T: ArrowPrimitive & FixedWidthInteger {
        let overflowHigh = String(T.max) + "0"                 // ten times the maximum
        let overflowLow = T.isSigned ? String(T.min) + "0" : "-1"
        let inputs: [String?] = ["0", "007", "+7", "  7", "7 ", "", "abc", "7a", "+", "-", "0x10",
                                 "1_0", overflowHigh, overflowLow, "-0", nil, String(T.max)]
        let parsed = try MetalStringArray(inputs).parse(T.self).toArray()
        XCTAssertEqual(parsed[0], 0)
        XCTAssertEqual(parsed[1], 7, "leading zeros")
        XCTAssertEqual(parsed[2], 7, "leading plus")
        for (i, label) in [(3, "leading space"), (4, "trailing space"), (5, "empty"), (6, "letters"),
                           (7, "trailing letter"), (8, "lone plus"), (9, "lone minus"),
                           (10, "hex prefix"), (11, "underscore"), (12, "overflow high"),
                           (13, "overflow low"), (15, "null in")] {
            XCTAssertNil(parsed[i], "\(T.self) \(label) should be null")
        }
        XCTAssertEqual(parsed[14], T.isSigned ? 0 : nil, "\(T.self) \"-0\"")
        XCTAssertEqual(parsed[16], T.max)
        // The strict variant reports the same failures as an error.
        XCTAssertThrowsError(try MetalStringArray(inputs).parse(T.self, strict: true))
        XCTAssertNoThrow(try MetalStringArray(["1", "2", nil]).parse(T.self, strict: true))
    }

    func testParseEdgeCases() throws {
        try requireRealGPU()
        try checkParseEdges(Int8.self)
        try checkParseEdges(UInt8.self)
        try checkParseEdges(Int16.self)
        try checkParseEdges(UInt16.self)
        try checkParseEdges(Int32.self)
        try checkParseEdges(UInt32.self)
        try checkParseEdges(Int64.self)
        try checkParseEdges(UInt64.self)
    }

    func testFloatAndBooleanCasts() throws {
        try requireRealGPU()
        let doubles: [Double?] = [0, -0.0, 1, 0.1, -2.5, 1e20, .infinity, -.infinity, .nan, nil]
        let text = try MetalArray<Double>(doubles).toStrings()
        XCTAssertEqual(text.toArray(), doubles.map { $0.map { "\($0)" } })
        let back = try text.parse(Double.self).toArray()
        for (i, d) in doubles.enumerated() {
            guard let d else { XCTAssertNil(back[i]); continue }
            if d.isNaN { XCTAssertEqual(back[i]?.isNaN, true) } else { XCTAssertEqual(back[i], d) }
        }
        let floats: [Float?] = [0, 1, 0.1, -3.25, nil]
        let ftext = try MetalArray<Float>(floats).toStrings()
        XCTAssertEqual(ftext.toArray(), floats.map { $0.map { "\($0)" } })
        XCTAssertEqual(try ftext.parse(Float.self).toArray(), floats)

        let boolArray = try MetalBooleanArray([true, false, true])
        XCTAssertEqual(try boolArray.toStrings().toArray(), ["true", "false", "true"])
        let parsedBools = try MetalStringArray(["true", "TRUE", "false", "1", "0", "yes", "", nil]).parseBool()
        XCTAssertEqual(parsedBools.toArray(), [true, true, false, true, false, nil, nil, nil])
        XCTAssertThrowsError(try MetalStringArray(["maybe"]).parseBool(strict: true))
    }

    // MARK: - Temporal rounding against Foundation

    private func utcCalendar() -> Calendar {
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(secondsFromGMT: 0)!
        return cal
    }

    /// 100k random second-resolution timestamps between 1900 and 2100 — inside the window where
    /// Foundation's Gregorian calendar has no Julian cutover and a `Date` holds whole seconds exactly.
    private func randomSeconds(_ n: Int, seed: UInt64) -> [Int64] {
        var g = TextRNG(seed)
        let lo: Int64 = -2_208_988_800, hi: Int64 = 4_102_444_800
        return (0..<n).map { _ in lo + Int64(g.next() % UInt64(hi - lo)) }
    }

    func testFloorTemporalMatchesCalendar() throws {
        try requireRealGPU()
        let n = 100_000
        let seconds = randomSeconds(n, seed: 2024)
        let ts = try MetalTemporalArray(type: .timestamp(.second, timezone: nil), seconds.map { Optional($0) })
        let cal = utcCalendar()
        let cases: [(TemporalRoundUnit, Set<Calendar.Component>)] = [
            (.day, [.year, .month, .day]),
            (.month, [.year, .month]),
            (.year, [.year]),
        ]
        for (unit, components) in cases {
            let got = try ts.floorTemporal(to: unit).toArray()
            for i in 0..<n {
                let date = Date(timeIntervalSince1970: TimeInterval(seconds[i]))
                let floored = cal.date(from: cal.dateComponents(components, from: date))!
                XCTAssertEqual(got[i], Int64(floored.timeIntervalSince1970),
                               "floor to \(unit.rawValue) of \(seconds[i])")
                if got[i] != Int64(floored.timeIntervalSince1970) { return }
            }
        }
    }

    func testCeilAndRoundTemporalMatchCalendar() throws {
        try requireRealGPU()
        let n = 5_000
        let seconds = randomSeconds(n, seed: 7)
        let ts = try MetalTemporalArray(type: .timestamp(.second, timezone: nil), seconds.map { Optional($0) })
        let cal = utcCalendar()
        for (unit, components, step) in [(TemporalRoundUnit.day, Set<Calendar.Component>([.year, .month, .day]), DateComponents(day: 1)),
                                         (TemporalRoundUnit.month, Set([.year, .month]), DateComponents(month: 1)),
                                         (TemporalRoundUnit.year, Set([.year]), DateComponents(year: 1))] {
            let ceiled = try ts.ceilTemporal(to: unit).toArray()
            let rounded = try ts.roundTemporal(to: unit).toArray()
            for i in 0..<n {
                let date = Date(timeIntervalSince1970: TimeInterval(seconds[i]))
                let lo = cal.date(from: cal.dateComponents(components, from: date))!
                let hi = cal.date(byAdding: step, to: lo)!
                let loS = Int64(lo.timeIntervalSince1970), hiS = Int64(hi.timeIntervalSince1970)
                XCTAssertEqual(ceiled[i], seconds[i] == loS ? loS : hiS, "ceil \(unit.rawValue)")
                let expected = 2 * (seconds[i] - loS) >= (hiS - loS) ? hiS : loS
                XCTAssertEqual(rounded[i], expected, "round \(unit.rawValue)")
                if ceiled[i] != (seconds[i] == loS ? loS : hiS) || rounded[i] != expected { return }
            }
        }
    }

    func testRoundingSubDayUnitsAndMultiples() throws {
        try requireRealGPU()
        // 2021-03-04 05:06:07.008009 UTC
        let base: Int64 = 1_614_834_367_008_009
        let ts = try MetalTemporalArray(type: .timestamp(.micro, timezone: "UTC"), [base, nil, -1])
        XCTAssertEqual(try ts.floorTemporal(to: .second).toArray(), [1_614_834_367_000_000, nil, -1_000_000])
        XCTAssertEqual(try ts.ceilTemporal(to: .second).toArray(), [1_614_834_368_000_000, nil, 0])
        XCTAssertEqual(try ts.floorTemporal(to: .millisecond).toArray(), [1_614_834_367_008_000, nil, -1_000])
        XCTAssertEqual(try ts.floorTemporal(to: .minute, multiple: 15).toArray()[0], 1_614_834_000_000_000)
        // Rounding to a unit finer than the storage is the identity.
        XCTAssertEqual(try ts.floorTemporal(to: .nanosecond).toArray(), [base, nil, -1])
        // A value already on a boundary is left alone by ceil.
        let exact = try MetalTemporalArray(type: .timestamp(.second, timezone: nil), [Int64(86_400)])
        XCTAssertEqual(try exact.ceilTemporal(to: .day).toArray(), [86_400])
        // Halves go up.
        let half = try MetalTemporalArray(type: .timestamp(.second, timezone: nil), [Int64(30), Int64(29)])
        XCTAssertEqual(try half.roundTemporal(to: .minute).toArray(), [60, 0])
        // Calendar units need a date.
        let dur = try MetalTemporalArray(type: .duration(.second), [Int64(5)])
        XCTAssertThrowsError(try dur.floorTemporal(to: .month))
    }

    func testDate32Rounding() throws {
        try requireRealGPU()
        // 2020-02-29 is day 18321.
        let d = try MetalTemporalArray(type: .date32, [Int64(18_321), nil])
        XCTAssertEqual(try d.floorTemporal(to: .day).toArray(), [18_321, nil])
        XCTAssertEqual(try d.floorTemporal(to: .hour).toArray(), [18_321, nil])   // finer than a day
        XCTAssertEqual(try d.floorTemporal(to: .month).toArray(), [18_293, nil])  // 2020-02-01
        XCTAssertEqual(try d.floorTemporal(to: .year).toArray(), [18_262, nil])   // 2020-01-01
        XCTAssertEqual(try d.ceilTemporal(to: .year).toArray(), [18_628, nil])    // 2021-01-01
        XCTAssertEqual(try d.floorTemporal(to: .quarter).toArray(), [18_262, nil])
    }

    // MARK: - Temporal components against Foundation

    func testCalendarFieldsMatchFoundation() throws {
        try requireRealGPU()
        let n = 20_000
        let seconds = randomSeconds(n, seed: 314)
        let ts = try MetalTemporalArray(type: .timestamp(.second, timezone: nil), seconds.map { Optional($0) })
        let cal = utcCalendar()
        var iso = Calendar(identifier: .iso8601)
        iso.timeZone = TimeZone(secondsFromGMT: 0)!

        let dayOfYear = try ts.dayOfYear().toArray()
        let quarter = try ts.quarter().toArray()
        let isoWeek = try ts.isoWeek().toArray()
        let isoYear = try ts.isoYear().toArray()
        let leap = try ts.isLeapYear().toArray()
        for i in 0..<n {
            let date = Date(timeIntervalSince1970: TimeInterval(seconds[i]))
            let c = cal.dateComponents([.year, .month], from: date)
            XCTAssertEqual(Int(dayOfYear[i]!), cal.ordinality(of: .day, in: .year, for: date), "day_of_year")
            XCTAssertEqual(Int(quarter[i]!), (c.month! + 2) / 3, "quarter")
            let isoC = iso.dateComponents([.weekOfYear, .yearForWeekOfYear], from: date)
            XCTAssertEqual(Int(isoWeek[i]!), isoC.weekOfYear!, "iso_week of \(seconds[i])")
            XCTAssertEqual(Int(isoYear[i]!), isoC.yearForWeekOfYear!, "iso_year of \(seconds[i])")
            let y = c.year!
            XCTAssertEqual(leap[i], (y % 4 == 0 && y % 100 != 0) || y % 400 == 0, "is_leap_year")
            if Int(dayOfYear[i]!) != cal.ordinality(of: .day, in: .year, for: date) { return }
        }
    }

    func testSubsecondComponents() throws {
        try requireRealGPU()
        let ns = try MetalTemporalArray(type: .timestamp(.nano, timezone: nil),
                                        [123_456_789, 1_987_654_321, -1, nil])
        XCTAssertEqual(try ns.millisecond().toArray(), [123, 987, 999, nil])
        XCTAssertEqual(try ns.microsecond().toArray(), [456, 654, 999, nil])
        XCTAssertEqual(try ns.nanosecond().toArray(), [789, 321, 999, nil])
        let ms = try MetalTemporalArray(type: .timestamp(.milli, timezone: nil), [Int64(1_042)])
        XCTAssertEqual(try ms.millisecond().toArray(), [42])
        XCTAssertEqual(try ms.microsecond().toArray(), [0])
    }

    // MARK: - Temporal arithmetic

    func testAddDurationAndSubtract() throws {
        try requireRealGPU()
        let ts = try MetalTemporalArray(type: .timestamp(.micro, timezone: "UTC"),
                                        [1_000_000, 2_000_000, nil])
        let dur = try MetalTemporalArray(type: .duration(.second), [Int64(1), nil, 5])
        let sum = try ts.addDuration(dur)
        XCTAssertEqual(sum.arrowFormat, "tsu:UTC")
        XCTAssertEqual(sum.toArray(), [2_000_000, nil, nil])
        XCTAssertEqual(try ts.addDuration(500_000).toArray(), [1_500_000, 2_500_000, nil])

        let other = try MetalTemporalArray(type: .timestamp(.milli, timezone: nil), [Int64(500), 1_000, 0])
        let diff = try ts.subtractTemporal(other)
        XCTAssertEqual(diff.arrowFormat, "tDu")
        XCTAssertEqual(diff.toArray(), [500_000, 1_000_000, nil])

        let d32 = try MetalTemporalArray(type: .date32, [Int64(10), 20])
        let d32b = try MetalTemporalArray(type: .date32, [Int64(7), 25])
        XCTAssertEqual(try d32.subtractTemporal(d32b).toArray(), [259_200, -432_000])   // 3 and -5 days in seconds
        XCTAssertThrowsError(try d32.addDuration(dur.slice(offset: 0, length: 2)))
    }

    func testDaysBetween() throws {
        try requireRealGPU()
        // 2021-03-04T23:59:00Z and 2021-03-05T00:01:00Z are one day apart.
        let a = try MetalTemporalArray(type: .timestamp(.second, timezone: nil),
                                       [1_614_902_340, 0, nil, 86_400])
        let b = try MetalTemporalArray(type: .timestamp(.second, timezone: nil),
                                       [1_614_902_460, 259_200, 5, 0])
        XCTAssertEqual(try a.daysBetween(b).toArray(), [1, 3, nil, -1])
    }

    // MARK: - strftime / strptime

    func testStrftimeAndStrptimeRoundTrip() throws {
        try requireRealGPU()
        let seconds = randomSeconds(5_000, seed: 99)
        let ts = try MetalTemporalArray(type: .timestamp(.second, timezone: nil), seconds.map { Optional($0) })
        let text = try ts.strftime("%Y-%m-%d %H:%M:%S")
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        let got = text.toArray()
        for i in 0..<seconds.count {
            let expected = formatter.string(from: Date(timeIntervalSince1970: TimeInterval(seconds[i])))
            XCTAssertEqual(got[i], expected)
            if got[i] != expected { return }
        }
        let back = try text.strptime("%Y-%m-%d %H:%M:%S", unit: .second)
        XCTAssertEqual(back.toArray(), seconds.map { Optional($0) })
    }

    func testStrftimeSubsecondAndNulls() throws {
        try requireRealGPU()
        let ts = try MetalTemporalArray(type: .timestamp(.micro, timezone: nil), [1_614_834_367_008_009, nil])
        XCTAssertEqual(try ts.strftime("%Y-%m-%dT%H:%M:%S.%f").toArray(),
                       ["2021-03-04T05:06:07.008009", nil])
        XCTAssertEqual(try ts.strftime("%%Y").toArray(), ["%Y", nil])
        let d = try MetalTemporalArray(type: .date32, [Int64(18_321)])
        XCTAssertEqual(try d.strftime("%Y-%m-%d").toArray(), ["2020-02-29"])
    }

    /// Every specifier the GPU formatter claims, byte for byte against the C library it replaced.
    ///
    /// `MetalTemporalArray.formatUTC` is that C library path (`gmtime_r` + `strftime` with `%f`
    /// pre-expanded), so this is the GPU kernel against the implementation it took over from, over
    /// 20k random instants from 1900 to 2100 in all four resolutions.
    func testStrftimeGPUIsByteExactAgainstTheCLibrary() throws {
        try requireRealGPU()
        let formats = ["%Y", "%m", "%d", "%e", "%H", "%I", "%M", "%S", "%j", "%y", "%C",
                       "%b", "%B", "%h", "%a", "%A", "%p", "%G", "%V", "%u", "%w",
                       "%F", "%T", "%D", "%R", "%n", "%t", "%%",
                       "%Y-%m-%d %H:%M:%S", "%A, %B %e, %Y", "%d/%b/%Y %I:%M %p", "%G-W%V-%u",
                       "literal %% text %Y!", "%Y-%m-%dT%H:%M:%S.%f"]
        let n = 20_000
        let seconds = randomSeconds(n, seed: 7_705)
        for unit in [ArrowTemporalUnit.second, .milli, .micro, .nano] {
            let per = unit.perSecond
            var g = TextRNG(31 + UInt64(per % 1_000))
            let sub: [Int64] = (0..<n).map { _ in per == 1 ? 0 : Int64(g.next() % UInt64(per)) }
            let values: [Int64?] = (0..<n).map { $0 % 997 == 5 ? nil : seconds[$0] * per + sub[$0] }
            let ts = try MetalTemporalArray(type: .timestamp(unit, timezone: nil), values)
            for f in formats {
                let got = try ts.strftime(f).toArray()
                for i in stride(from: 0, to: n, by: 7) {
                    guard values[i] != nil else { XCTAssertNil(got[i]); continue }
                    let micros = per <= 1_000_000 ? sub[i] * (1_000_000 / per) : sub[i] / (per / 1_000_000)
                    let want = MetalTemporalArray.formatUTC(seconds: seconds[i], microseconds: micros,
                                                            format: f)
                    XCTAssertEqual(got[i], want, "format \(f) unit \(unit) row \(i) value \(values[i]!)")
                    if got[i] != want { return }
                }
            }
        }
        // date32 goes through the same kernel with whole days.
        let days: [Int64?] = [0, 18_321, -25_567, 47_481, nil]
        let d = try MetalTemporalArray(type: .date32, days)
        for f in ["%Y-%m-%d", "%j", "%A %B %e", "%G-W%V-%u"] {
            let got = try d.strftime(f).toArray()
            for (i, v) in days.enumerated() {
                guard let v else { XCTAssertNil(got[i]); continue }
                XCTAssertEqual(got[i], MetalTemporalArray.formatUTC(seconds: v * 86_400, microseconds: 0,
                                                                    format: f), "\(f) row \(i)")
            }
        }
        // An unknown specifier is not a failure: it falls back to the C library.
        let ts = try MetalTemporalArray(type: .timestamp(.second, timezone: nil), [Int64(0)])
        XCTAssertEqual(try ts.strftime("%c").toArray(),
                       [MetalTemporalArray.formatUTC(seconds: 0, microseconds: 0, format: "%c")])
    }

    /// A timestamp carrying a timezone formats in that zone on the GPU path, which is what makes `%z`
    /// and `%Z` meaningful. Checked against Foundation's own offset, and `%Z` against the abbreviation
    /// the zone's own transition table carries.
    func testStrftimeAppliesTheColumnTimezone() throws {
        try requireRealGPU()
        for tz in ["America/New_York", "Europe/Berlin", "Australia/Sydney", "Asia/Kolkata", "UTC"] {
            let zone = try resolveTimeZone(tz)
            // Two instants a year apart in each hemisphere's summer and winter.
            let instants: [Int64] = [1_704_067_200, 1_719_792_000, -1_000_000_000, 946_684_800]
            let ts = try MetalTemporalArray(type: .timestamp(.second, timezone: tz),
                                            instants.map { Optional($0) })
            let clock = try ts.strftime("%Y-%m-%d %H:%M:%S").toArray()
            let offsets = try ts.strftime("%z").toArray()
            let names = try ts.strftime("%Z").toArray()
            let lookup = try ts.utcOffset().toArray()
            for (i, t) in instants.enumerated() {
                let o = Int64(zone.secondsFromGMT(for: Date(timeIntervalSince1970: Double(t))))
                XCTAssertEqual(clock[i], MetalTemporalArray.formatUTC(seconds: t + o, microseconds: 0,
                                                                      format: "%Y-%m-%d %H:%M:%S"),
                               "\(tz) row \(i)")
                let sign = o < 0 ? "-" : "+"
                let a = abs(o)
                XCTAssertEqual(offsets[i], String(format: "%@%02d%02d", sign, a / 3600, (a / 60) % 60),
                               "\(tz) %z row \(i)")
                XCTAssertEqual(lookup[i], Int32(o), "\(tz) utcOffset row \(i)")
                XCTAssertFalse(names[i]?.isEmpty ?? true, "\(tz) %Z row \(i)")
            }
        }
        // The well-known spellings, which Foundation's own `abbreviation(for:)` does not produce.
        let winter: Int64 = 1_704_067_200, summer: Int64 = 1_719_792_000
        for (tz, expected) in [("America/New_York", ["EST", "EDT"]), ("Europe/Berlin", ["CET", "CEST"]),
                               ("Asia/Kolkata", ["IST", "IST"])] {
            let ts = try MetalTemporalArray(type: .timestamp(.second, timezone: tz), [winter, summer])
            XCTAssertEqual(try ts.strftime("%Z").toArray(), expected.map { Optional($0) }, tz)
        }
    }

    /// The GPU parser against the C library's `strptime` + `timegm`, on well-formed and malformed
    /// input alike. The one deliberate difference is `%f`, which C has no notion of.
    func testStrptimeGPUAgreesWithTheCLibrary() throws {
        try requireRealGPU()
        let formats = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d/%b/%Y %H:%M:%S", "%m/%d/%y",
                       "%Y-%m-%dT%H:%M:%S", "%B %d, %Y", "%A %d %B %Y", "%H:%M:%S", "%I:%M %p"]
        let n = 5_000
        let seconds = randomSeconds(n, seed: 4_242)
        for f in formats {
            let ts = try MetalTemporalArray(type: .timestamp(.second, timezone: nil),
                                            seconds.map { Optional($0) })
            let text = try ts.strftime(f)
            let got = try text.strptime(f, unit: .second).toArray()
            let rows = text.toArray()
            for i in stride(from: 0, to: n, by: 3) {
                XCTAssertEqual(got[i], Self.cStrptime(rows[i]!, f), "format \(f) row \(i) text \(rows[i]!)")
                if got[i] != Self.cStrptime(rows[i]!, f) { return }
            }
        }
        // Malformed input fails the same way it fails on the host.
        let bad = ["2020-01-02", "nonsense", "2020-01-02 extra", "", "2020-13-02", "2020-01-32",
                   "9999-12-31", " 2020-01-02", "2020-01-02 ", "2020-1-2", "20200102",
                   "+2020-01-02", "-0001-01-01", "2020-01-", "2020--1-02"]
        for f in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S"] {
            let text = try MetalStringArray(bad.map { Optional($0) } + [nil])
            let got = try text.strptime(f, unit: .second).toArray()
            for (i, s) in bad.enumerated() {
                XCTAssertEqual(got[i], Self.cStrptime(s, f), "format \(f) input \"\(s)\"")
            }
            XCTAssertNil(got[bad.count])
        }
        // Two documented departures from the host path, both of them the C library falling short.
        // `timegm` reports failure for a year the proleptic Gregorian calendar handles exactly:
        let ancient = try MetalStringArray(["0001-01-01", "0000-01-01"])
        XCTAssertEqual(try ancient.strptime("%Y-%m-%d", unit: .second).toArray(),
                       [-62_135_596_800, -62_167_219_200])
        XCTAssertEqual(Self.cStrptime("0001-01-01", "%Y-%m-%d"), -1)
        // And `%f` is ArrowMetal's own: it round-trips against `strftime`, which C cannot do.
        let micros = try MetalTemporalArray(type: .timestamp(.micro, timezone: nil),
                                            [1_614_834_367_008_009, nil, -1])
        let text = try micros.strftime("%Y-%m-%dT%H:%M:%S.%f")
        XCTAssertEqual(try text.strptime("%Y-%m-%dT%H:%M:%S.%f", unit: .micro).toArray(),
                       [1_614_834_367_008_009, nil, -1])
    }

    /// The C library's `strptime` + `timegm`, the implementation the GPU parser replaced.
    private static func cStrptime(_ s: String, _ format: String) -> Int64? {
        var tmv = tm()
        tmv.tm_mday = 1
        tmv.tm_year = 70
        let parsed: Bool = s.withCString { cs in
            format.withCString { cf in
                guard let rest = Darwin.strptime(cs, cf, &tmv) else { return false }
                return rest.pointee == 0
            }
        }
        guard parsed else { return nil }
        return Int64(timegm(&tmv))
    }

    func testStrptimeFailuresAreNull() throws {
        try requireRealGPU()
        let text = try MetalStringArray(["2020-01-02", "nonsense", "2020-01-02 extra", nil, ""])
        let parsed = try text.strptime("%Y-%m-%d", unit: .second)
        XCTAssertEqual(parsed.toArray(), [1_577_923_200, nil, nil, nil, nil])
        XCTAssertEqual(parsed.arrowFormat, "tss:")
        XCTAssertThrowsError(try text.strptime("%Y-%m-%d", unit: .second, strict: true))
    }

    // MARK: - Shapes

    func testEmptyAndSingleRowShapes() throws {
        try requireRealGPU()
        for n in [0, 1] {
            let values = sampleStrings(n, seed: 5, nullEvery: 1)
            let array = try MetalStringArray(values)
            XCTAssertEqual(try array.matchSubstringRegex("a.b").length, n)
            XCTAssertEqual(try array.countSubstringRegex("a").length, n)
            XCTAssertEqual(try array.matchLike("%a%").length, n)
            XCTAssertEqual(try array.parse(Int32.self).length, n)
            XCTAssertEqual(try MetalArray<Int32>([Int32?]()).toStrings().length, 0)
            let (offsets, pieces) = try array.splitPatternPair(",")
            XCTAssertEqual(offsets.length, n + 1)
            XCTAssertEqual(pieces.length, n == 0 ? 0 : (values[0] == nil ? 0 : 1))
        }
    }

    func testLargeRegexAndLikeShape() throws {
        try requireRealGPU()
        let n = 200_003
        let values = sampleStrings(n, seed: 777)
        let array = try MetalStringArray(values)
        let re = try oracleRegex("a.c")
        let expected = values.map { s in s.map { re.firstMatch(in: $0, range: full($0)) != nil } }
        XCTAssertEqual(try array.matchSubstringRegex("a.c").toArray(), expected)
        let like = try array.matchLike("%ab%").toArray()
        XCTAssertEqual(like, values.map { s in s.map { $0.contains("ab") } })
    }

    // MARK: - Splitting: options, the list column, and the GPU/host agreement

    /// Strings built to make every splitting edge case common rather than rare: leading and trailing
    /// separators, runs of them, empty values, ASCII and Unicode whitespace side by side.
    func splitCorpus(_ n: Int, seed: UInt64) -> [String?] {
        var g = TextRNG(seed)
        let pieces = ["a", "bb", "", " ", "  ", "\t", "\n", ",", ",,", "x,y", "\u{3000}", "\u{A0}",
                      "\u{2003}", "\u{200B}", "é", "日本", "🎉", "\u{1C}", "\u{85}"]
        var out: [String?] = []
        for i in 0..<n {
            if i % 9 == 4 { out.append(nil); continue }
            var s = ""
            for _ in 0...Int(g.next() % 5) { s += pieces[Int(g.next() % UInt64(pieces.count))] }
            out.append(s)
        }
        return out
    }

    /// Reads a `list<utf8>` back as one `[String]` per row (nil for a null row).
    func listRows(_ list: MetalListArray) throws -> [[String]?] {
        let (offsets, values) = try list.stringPair()
        let offs = offsets.toArray().map { Int($0!) }
        let flat = values.toArray()
        return (0..<list.length).map { i in
            guard list.isValid(i) else { return nil }
            return Array(flat[offs[i]..<offs[i + 1]]).map { $0! }
        }
    }

    /// Every splitter against the host oracle, at every size and for every `max_splits` / `reverse`
    /// combination. The GPU splitter runs three passes over two scans, so a row whose piece count or
    /// byte count is computed differently in any pass shows up here immediately.
    func testSplitOptionsMatchTheOracle() throws {
        try requireRealGPU()
        for n in Self.sizes {
            let values = splitCorpus(n, seed: UInt64(n) &+ 821)
            let array = try MetalStringArray(values)
            for maxSplits in [-1, 0, 1, 2, 5] {
                for reverse in [false, true] {
                    for unicode in [false, true] {
                        let got = try listRows(try array.splitWhitespace(unicode: unicode,
                                                                         maxSplits: maxSplits,
                                                                         reverse: reverse))
                        let want = values.map { s in
                            s.map { MetalStringArray.splitWhitespace($0, unicode: unicode,
                                                                     maxSplits: maxSplits, reverse: reverse) }
                        }
                        XCTAssertEqual(got, want,
                                       "split_whitespace(unicode: \(unicode), max: \(maxSplits), rev: \(reverse)) at n=\(n)")
                    }
                    for pattern in [",", ",,", "a", "é"] {
                        let got = try listRows(try array.splitPattern(pattern, maxSplits: maxSplits,
                                                                      reverse: reverse))
                        let want = values.map { s in
                            s.map { MetalStringArray.split($0, on: pattern, maxSplits: maxSplits,
                                                           reverse: reverse) }
                        }
                        XCTAssertEqual(got, want,
                                       "split_pattern(\(pattern), max: \(maxSplits), rev: \(reverse)) at n=\(n)")
                    }
                }
            }
        }
    }

    /// The exact pieces Arrow produces for the shapes that are easy to get wrong: a separator at each
    /// end, a run of separators, the empty string, an all-whitespace value, and a null row.
    func testSplitPinnedAnswers() throws {
        try requireRealGPU()
        let a = try MetalStringArray(["  x  ", "a b  c", "", " ", nil, "one", "a\tb\nc", "a\u{1C}b"])
        XCTAssertEqual(try listRows(try a.splitWhitespace()),
                       [["", "x", ""], ["a", "b", "c"], [""], ["", ""], nil, ["one"], ["a", "b", "c"],
                        ["a\u{1C}b"]])
        // U+001C-U+001F are Unicode whitespace but not ASCII whitespace, so only the utf8 form cuts there.
        XCTAssertEqual(try listRows(try a.splitWhitespace(unicode: true)).last!, ["a", "b"])
        XCTAssertEqual(try listRows(try a.splitWhitespace(maxSplits: 1)),
                       [["", "x  "], ["a", "b  c"], [""], ["", ""], nil, ["one"], ["a", "b\nc"],
                        ["a\u{1C}b"]])
        XCTAssertEqual(try listRows(try a.splitWhitespace(maxSplits: 1, reverse: true)),
                       [["  x", ""], ["a b", "c"], [""], ["", ""], nil, ["one"], ["a\tb", "c"],
                        ["a\u{1C}b"]])
        XCTAssertEqual(try listRows(try a.splitWhitespace(maxSplits: 0)),
                       [["  x  "], ["a b  c"], [""], [" "], nil, ["one"], ["a\tb\nc"], ["a\u{1C}b"]])
        let b = try MetalStringArray(["allbll", "l", "", "lal", nil, "xx"])
        XCTAssertEqual(try listRows(try b.splitPattern("l")),
                       [["a", "", "b", "", ""], ["", ""], [""], ["", "a", ""], nil, ["xx"]])
        XCTAssertEqual(try listRows(try b.splitPattern("l", maxSplits: 1)),
                       [["a", "lbll"], ["", ""], [""], ["", "al"], nil, ["xx"]])
        XCTAssertEqual(try listRows(try b.splitPattern("l", maxSplits: 1, reverse: true)),
                       [["allbl", ""], ["", ""], [""], ["la", ""], nil, ["xx"]])
        XCTAssertEqual(try listRows(try b.splitPatternRegex("l+")),
                       [["a", "b", ""], ["", ""], [""], ["", "a", ""], nil, ["xx"]])
        XCTAssertEqual(try listRows(try b.splitPatternRegex("l+", maxSplits: 1)),
                       [["a", "bll"], ["", ""], [""], ["", "al"], nil, ["xx"]])
    }

    /// The list column is the primary result; the pair is the same buffers seen flat.
    func testSplitListAndPairAgree() throws {
        try requireRealGPU()
        for n in [0, 1, 33, 4097] {
            let values = splitCorpus(n, seed: UInt64(n) &+ 823)
            let array = try MetalStringArray(values)
            let list = try array.splitPattern(",")
            XCTAssertEqual(list.length, n)
            XCTAssertEqual(list.arrowFormat, "+l")
            XCTAssertEqual(list.nullCount, values.filter { $0 == nil }.count)
            let (offsets, pieces) = try array.splitPatternPair(",")
            XCTAssertEqual(offsets.toArray(), try list.stringPair().offsets.toArray())
            XCTAssertEqual(pieces.toArray(), try list.stringPair().values.toArray())
            // A null row owns no pieces and is a null list row.
            for i in 0..<n where values[i] == nil {
                XCTAssertFalse(list.isValid(i))
                XCTAssertEqual(list.valueRange(i), nil)
            }
        }
    }

    // MARK: - SQL LIKE on the GPU

    /// Every `LIKE` shape, including the ones that used to fall through to ICU, against the anchored
    /// regex the host engine would have run. The two must agree row for row.
    func testMatchLikeOnTheGPUMatchesTheEngine() throws {
        try requireRealGPU()
        let patterns = ["abc", "abc%", "%abc", "%abc%", "a_c", "_bc", "ab_", "%a_c%", "a%b%c",
                        "%", "%%", "", "_", "___", "a\\%b", "a\\_b", "\\\\", "%é%", "_é_",
                        "cust\\_1%", "c_st%1", "日%本", "a%", "%c", "%_%"]
        for n in Self.sizes {
            let values = sampleStrings(n, seed: UInt64(n) &+ 901)
            let array = try MetalStringArray(values)
            for p in patterns {
                let tokens = MetalStringArray.parseLike(p)
                let re = try NSRegularExpression(pattern: MetalStringArray.likeRegex(tokens),
                                                 options: [.dotMatchesLineSeparators])
                let expected = values.map { s in s.map { re.firstMatch(in: $0, range: full($0)) != nil } }
                XCTAssertEqual(try array.matchLike(p).toArray(), expected, "match_like \(p) at n=\(n)")
            }
        }
    }

    /// `_` has to consume one whole character, not one byte, or a multi-byte value would match a
    /// pattern that should not fit it.
    func testMatchLikeUnderscoreCountsCodePoints() throws {
        try requireRealGPU()
        let a = try MetalStringArray(["é", "ab", "日", "🎉", "aé", ""])
        XCTAssertEqual(try a.matchLike("_").toArray(), [true, false, true, true, false, false])
        XCTAssertEqual(try a.matchLike("__").toArray(), [false, true, false, false, true, false])
        XCTAssertEqual(try a.matchLike("").toArray(), [false, false, false, false, false, true])
        XCTAssertEqual(try a.matchLike("%_%").toArray(), [true, true, true, true, true, false])
    }

    // MARK: - The GPU literal pre-filter for the regex functions

    /// The pre-filter must never change an answer: for each pattern, what the analysis claims is a
    /// required literal really is required, and the four regex functions agree with the engine run
    /// over every row.
    func testRegexLiteralPrefilterKeepsAnswersIdentical() throws {
        try requireRealGPU()
        let patterns = ["ab", "abc[0-9]+", "\\d{2}-ab-\\d+", "a.c", "(ab|zz)", "ab$", "^abc",
                        "ab*c", "ab+c", "x?abc", "[a-z]+ab", "ab(c|d)ef", "\\.ab\\.", "a{2,3}bcd",
                        "cafe\u{301}.", "ab\\b", "(?:ab)cd"]
        for n in [0, 1, 33, 4097] {
            let values = sampleStrings(n, seed: UInt64(n) &+ 911)
            let array = try MetalStringArray(values)
            for p in patterns {
                let re = try oracleRegex(p)
                // The claim itself: a row that matches must contain the literal.
                if let literal = MetalStringArray.requiredLiteral(p) {
                    for v in values.compactMap({ $0 })
                    where re.firstMatch(in: v, range: full(v)) != nil {
                        XCTAssertTrue(v.contains(literal),
                                      "\(p) matched \(v.debugDescription) without \(literal.debugDescription)")
                    }
                }
                XCTAssertEqual(try array.matchSubstringRegex(p).toArray(),
                               values.map { s in s.map { re.firstMatch(in: $0, range: full($0)) != nil } },
                               "match \(p) at n=\(n)")
                XCTAssertEqual(try array.countSubstringRegex(p).toArray(),
                               values.map { s in s.map { Int32(re.numberOfMatches(in: $0, range: full($0))) } },
                               "count \(p) at n=\(n)")
                XCTAssertEqual(try array.findSubstringRegex(p).toArray(),
                               values.map { s in
                                   s.map { v -> Int32 in
                                       guard let m = re.firstMatch(in: v, range: full(v)),
                                             let r = Range(m.range, in: v) else { return -1 }
                                       return Int32(v.utf8.distance(from: v.utf8.startIndex, to: r.lowerBound))
                                   }
                               },
                               "find \(p) at n=\(n)")
                XCTAssertEqual(try array.replaceSubstringRegex(p, with: "Z").toArray(),
                               values.map { s in
                                   s.map { re.stringByReplacingMatches(in: $0, range: full($0), withTemplate: "Z") }
                               },
                               "replace \(p) at n=\(n)")
            }
        }
    }

    /// What the analysis is allowed to claim, spelled out: a top-level alternation, an inline flag
    /// group and anything inside parentheses give up, and a zero-or-more quantifier drops the
    /// character it binds to.
    func testRequiredLiteralAnalysis() throws {
        XCTAssertEqual(MetalStringArray.requiredLiteral("hello.*world"), "hello")
        XCTAssertEqual(MetalStringArray.requiredLiteral("\\d{4}-cust-\\d+"), "-cust-")
        XCTAssertEqual(MetalStringArray.requiredLiteral("[0-9]+abcd"), "abcd")
        XCTAssertEqual(MetalStringArray.requiredLiteral("ab+cdef"), "cdef")
        XCTAssertNil(MetalStringArray.requiredLiteral("abc|def"))
        XCTAssertNil(MetalStringArray.requiredLiteral("(?i)abcdef"))
        XCTAssertNil(MetalStringArray.requiredLiteral("(abcdef)"))
        XCTAssertNil(MetalStringArray.requiredLiteral("a.c"))
        XCTAssertNil(MetalStringArray.requiredLiteral("ab*c"))
        XCTAssertEqual(MetalStringArray.requiredLiteral("\\.abc\\."), ".abc.")
    }
}
