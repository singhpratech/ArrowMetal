import XCTest
import Foundation
@testable import ArrowMetal

/// The remaining Arrow string surface: the `utf8_is_*` / `ascii_is_*` predicates, `string_is_ascii`,
/// the capitalize / title / center / replace-slice transforms, the Unicode trims, `utf8_normalize`,
/// `extract_regex_span`, `binary_join` and string `is_in` / `index_in`.
///
/// Two layers of checking. A curated table pins the exact answers Arrow gives for the awkward inputs
/// (empty strings, titlecase letters, Roman numerals, ligatures, the information separators), which is
/// what a differential run against `pyarrow.compute` is comparing to. The bulk tests then run every
/// function at 0, 1, 33, 4097 and 500 003 rows against an oracle written straight over Swift's
/// `Unicode.Scalar` view, so a shape bug in the two-pass kernels, the bitmap packing or the CPU
/// fixup shows up at exactly one of those sizes.
final class StringExtraTests: XCTestCase {

    /// Deterministic so a failure can be reproduced.
    struct RNG: RandomNumberGenerator {
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

    static let sizes = [0, 1, 33, 4097, 500_003]

    /// Pieces chosen so that every predicate is true for some rows and false for others, and so that
    /// ASCII-only and non-ASCII rows are interleaved (the predicate kernel decides per row, so the two
    /// paths have to agree inside a single bitmap word).
    static let pieces = ["Hello", "World", "hello", "HELLO", "abc", "ABC", "123", "007", "",
                         "a", " ", "  ", "\t", "\n", "\u{1C}", "\u{85}", "\u{A0}", "\u{2003}",
                         "Ünïcödé", "ünïcödé", "ÜNÏCÖDÉ", "日本語", "😀", "ǅungla", "Ǆ", "ǅ", "ǆ",
                         "ß", "Straße", "İstanbul", "ﬁx", "½", "²", "Ⅷ", "ⅷ", "٣٤", "x-ray",
                         "O'Neil", "a1b2", "  pad  ", "\u{200B}"]

    func sample(_ n: Int, seed: UInt64, nullEvery: Int = 7) -> [String?] {
        var g = RNG(seed)
        var out: [String?] = []
        out.reserveCapacity(n)
        for i in 0..<n {
            if nullEvery > 0 && i % nullEvery == 3 { out.append(nil); continue }
            var s = ""
            let parts = Int(g.next() % 3)
            for _ in 0...parts { s += Self.pieces[Int(g.next() % UInt64(Self.pieces.count))] }
            out.append(s)
        }
        return out
    }

    /// A column with no non-ASCII byte anywhere, to exercise the all-GPU paths.
    func asciiSample(_ n: Int, seed: UInt64) -> [String?] {
        let ascii = Self.pieces.filter { $0.utf8.allSatisfy { b in b < 0x80 } }
        var g = RNG(seed)
        var out: [String?] = []
        for i in 0..<n {
            if i % 5 == 2 { out.append(nil); continue }
            var s = ""
            for _ in 0...Int(g.next() % 3) { s += ascii[Int(g.next() % UInt64(ascii.count))] }
            out.append(s)
        }
        return out
    }

    // MARK: - Oracles

    static func isUpper(_ u: Unicode.Scalar) -> Bool {
        u.properties.changesWhenLowercased || u.properties.generalCategory == .titlecaseLetter
    }
    static func isLower(_ u: Unicode.Scalar) -> Bool {
        if u.value >= 0x2160 && u.value <= 0x216F { return false }
        return u.properties.changesWhenUppercased || u.properties.generalCategory == .lowercaseLetter
    }
    static func isCased(_ u: Unicode.Scalar) -> Bool { isUpper(u) || isLower(u) }

    /// The reference answer for one predicate on one string, written independently of the kernel and
    /// of `StringPredicate.evaluate`.
    static func refPredicate(_ p: StringPredicate, _ s: String) -> Bool {
        let u = Array(s.unicodeScalars), bytes = Array(s.utf8)
        func cat(_ c: Unicode.Scalar) -> Unicode.GeneralCategory { c.properties.generalCategory }
        switch p {
        case .asciiIsPrintable: return bytes.allSatisfy { $0 >= 0x20 && $0 <= 0x7E }
        case .stringIsAscii: return bytes.allSatisfy { $0 < 0x80 }
        case .asciiIsTitle:
            var any = false, prev = false
            for b in bytes {
                let up = b >= 0x41 && b <= 0x5A, lo = b >= 0x61 && b <= 0x7A
                if up { if prev { return false }; prev = true; any = true }
                else if lo { if !prev { return false }; prev = true; any = true }
                else { prev = false }
            }
            return any
        case .utf8IsTitle:
            var any = false, prev = false
            for c in u {
                if isUpper(c) { if prev { return false }; prev = true; any = true }
                else if isLower(c) { if !prev { return false }; prev = true; any = true }
                else { prev = false }
            }
            return any
        case .utf8IsPrintable:
            return u.allSatisfy { c in
                if c.value == 0x20 { return true }
                switch cat(c) {
                case .control, .format, .surrogate, .privateUse, .unassigned,
                     .spaceSeparator, .lineSeparator, .paragraphSeparator: return false
                default: return true
                }
            }
        case .utf8IsUpper:
            if u.contains(where: isLower) { return false }
            return u.contains(where: isUpper)
        case .utf8IsLower:
            if u.contains(where: isUpper) { return false }
            return u.contains(where: isLower)
        case .utf8IsAlpha:
            return !u.isEmpty && u.allSatisfy { c in
                switch cat(c) {
                case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter: return true
                default: return false
                }
            }
        case .utf8IsDecimal:
            return !u.isEmpty && u.allSatisfy { cat($0) == .decimalNumber }
        case .utf8IsDigit:
            return !u.isEmpty && u.allSatisfy { cat($0) == .decimalNumber || cat($0) == .otherNumber }
        case .utf8IsNumeric:
            return !u.isEmpty && u.allSatisfy {
                cat($0) == .decimalNumber || cat($0) == .letterNumber || cat($0) == .otherNumber
            }
        case .utf8IsAlnum:
            return !u.isEmpty && u.allSatisfy { c in
                switch cat(c) {
                case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter,
                     .decimalNumber, .letterNumber, .otherNumber: return true
                default: return false
                }
            }
        case .utf8IsSpace:
            return !u.isEmpty && u.allSatisfy { c in
                switch cat(c) {
                case .spaceSeparator, .lineSeparator, .paragraphSeparator: return true
                default: break
                }
                return (0x09...0x0D).contains(c.value) || (0x1C...0x1F).contains(c.value) || c.value == 0x85
            }
        }
    }

    static func refSimpleUpper(_ c: Unicode.Scalar) -> Unicode.Scalar {
        if c.value == 0xDF { return Unicode.Scalar(0x1E9E)! }
        if (0x1F80...0x1F87).contains(c.value) || (0x1F90...0x1F97).contains(c.value)
            || (0x1FA0...0x1FA7).contains(c.value) { return Unicode.Scalar(c.value + 8)! }
        let m = c.properties.uppercaseMapping.unicodeScalars
        return m.count == 1 ? m.first! : c
    }
    static func refSimpleLower(_ c: Unicode.Scalar) -> Unicode.Scalar {
        if c.value == 0x130 { return Unicode.Scalar(0x69)! }
        let m = c.properties.lowercaseMapping.unicodeScalars
        return m.count == 1 ? m.first! : c
    }
    static func scalarString(_ v: [Unicode.Scalar]) -> String {
        var s = String.UnicodeScalarView()
        for c in v { s.append(c) }
        return String(s)
    }

    static func refCapitalize(_ s: String) -> String {
        scalarString(s.unicodeScalars.enumerated().map { $0.offset == 0 ? refSimpleUpper($0.element) : refSimpleLower($0.element) })
    }
    static func refTitle(_ s: String) -> String {
        var out: [Unicode.Scalar] = [], boundary = true
        for c in s.unicodeScalars {
            if isCased(c) { out.append(boundary ? refSimpleUpper(c) : refSimpleLower(c)); boundary = false }
            else { out.append(c); boundary = true }
        }
        return scalarString(out)
    }
    static func refAsciiTitle(_ s: String) -> String {
        var b = Array(s.utf8), prev = false
        for i in b.indices {
            let up = b[i] >= 0x41 && b[i] <= 0x5A, lo = b[i] >= 0x61 && b[i] <= 0x7A
            if up || lo {
                b[i] = prev ? (up ? b[i] + 32 : b[i]) : (lo ? b[i] - 32 : b[i])
                prev = true
            } else { prev = false }
        }
        return String(decoding: b, as: UTF8.self)
    }
    static func refCenter(_ s: String, _ width: Int, _ pad: String) -> String {
        let n = s.unicodeScalars.count
        guard width > n else { return s }
        let total = width - n, left = total / 2
        return String(repeating: pad, count: left) + s + String(repeating: pad, count: total - left)
    }
    static func refReplaceSlice(_ s: String, _ start: Int, _ stop: Int, _ repl: String) -> String {
        let u = Array(s.unicodeScalars), n = u.count
        var a = start < 0 ? max(n + start, 0) : min(start, n)
        var b = stop < 0 ? max(n + stop, 0) : min(stop, n)
        if b < a { b = a }
        a = min(a, n); b = min(b, n)
        return scalarString(Array(u[0..<a])) + repl + scalarString(Array(u[b..<n]))
    }
    static func refReplaceSliceBytes(_ s: String, _ start: Int, _ stop: Int, _ repl: [UInt8]) -> [UInt8] {
        let u = Array(s.utf8), n = u.count
        let a = start < 0 ? max(n + start, 0) : min(start, n)
        var b = stop < 0 ? max(n + stop, 0) : min(stop, n)
        if b < a { b = a }
        return Array(u[0..<a]) + repl + Array(u[b..<n])
    }
    static func refTrim(_ s: String, left: Bool, right: Bool, _ inSet: (Unicode.Scalar) -> Bool) -> String {
        var u = Array(s.unicodeScalars)[...]
        if left { while let f = u.first, inSet(f) { u = u.dropFirst() } }
        if right { while let l = u.last, inSet(l) { u = u.dropLast() } }
        return scalarString(Array(u))
    }
    static func refIsSpace(_ c: Unicode.Scalar) -> Bool {
        switch c.properties.generalCategory {
        case .spaceSeparator, .lineSeparator, .paragraphSeparator: return true
        default: break
        }
        return (0x09...0x0D).contains(c.value) || (0x1C...0x1F).contains(c.value) || c.value == 0x85
    }

    // MARK: - Curated Arrow semantics

    func testPredicateTable() throws {
        try requireRealGPU()
        // (input, [ascii_printable, ascii_title, string_is_ascii, alnum, alpha, decimal, digit,
        //          lower, numeric, printable, space, title, upper])
        let cases: [(String, [Bool])] = [
            ("", [true, false, true, false, false, false, false, false, false, true, false, false, false]),
            ("Hello World", [true, true, true, false, false, false, false, false, false, true, false, true, false]),
            ("hello", [true, false, true, true, true, false, false, true, false, true, false, false, false]),
            ("HELLO", [true, false, true, true, true, false, false, false, false, true, false, false, true]),
            ("\t x", [false, false, true, false, false, false, false, true, false, false, false, false, false]),
            ("Ünïcödé", [false, false, false, true, true, false, false, false, false, true, false, true, false]),
            ("日本語", [false, false, false, true, true, false, false, false, false, true, false, false, false]),
            ("007", [true, false, true, true, false, true, true, false, true, true, false, false, false]),
            ("٣٤", [false, false, false, true, false, true, true, false, true, true, false, false, false]),
            ("²", [false, false, false, true, false, false, true, false, true, true, false, false, false]),
            ("½", [false, false, false, true, false, false, true, false, true, true, false, false, false]),
            ("Ⅷ", [false, false, false, true, false, false, false, false, true, true, false, true, true]),
            ("ǅ", [false, false, false, true, true, false, false, false, false, true, false, true, false]),
            ("ǅungla", [false, false, false, true, true, false, false, false, false, true, false, true, false]),
            // Byte-wise, the ASCII `X` / `A` opens a word because the bytes of ǅ / ß are not letters.
            ("ǅX", [false, true, false, true, true, false, false, false, false, true, false, false, false]),
            ("Aß", [false, true, false, true, true, false, false, false, false, true, false, true, false]),
            ("\u{00A0}", [false, false, false, false, false, false, false, false, false, false, true, false, false]),
            ("\u{2003}", [false, false, false, false, false, false, false, false, false, false, true, false, false]),
            ("\u{200B}", [false, false, false, false, false, false, false, false, false, false, false, false, false]),
            ("\u{1C}", [false, false, true, false, false, false, false, false, false, false, true, false, false]),
            (" ", [true, false, true, false, false, false, false, false, false, true, true, false, false]),
        ]
        let a = try MetalStringArray(cases.map { $0.0 })
        for p in StringPredicate.allCases {
            let got = try a.predicate(p).toArray()
            for (i, c) in cases.enumerated() {
                XCTAssertEqual(got[i], c.1[p.rawValue],
                               "\(p) on \(c.0.debugDescription) (row \(i))")
                XCTAssertEqual(Self.refPredicate(p, c.0), c.1[p.rawValue],
                               "oracle disagrees for \(p) on \(c.0.debugDescription)")
            }
        }
    }

    func testTransformTable() throws {
        try requireRealGPU()
        let a = try MetalStringArray(["hello world", "aBc dEf", "", "ǆungla", "ünïcödé wörld",
                                      "ß str", "İstanbul", "a1b2 c3", "日本語 abc", "  spaced  "])
        XCTAssertEqual(try a.utf8Capitalize().toArray(),
                       ["Hello world", "Abc def", "", "Ǆungla", "Ünïcödé wörld",
                        "ẞ str", "İstanbul", "A1b2 c3", "日本語 abc", "  spaced  "])
        XCTAssertEqual(try a.utf8Title().toArray(),
                       ["Hello World", "Abc Def", "", "Ǆungla", "Ünïcödé Wörld",
                        "ẞ Str", "İstanbul", "A1B2 C3", "日本語 Abc", "  Spaced  "])
        XCTAssertEqual(try a.asciiTitle().toArray(),
                       ["Hello World", "Abc Def", "", "ǆUngla", "üNïCöDé WöRld",
                        "ß Str", "İStanbul", "A1B2 C3", "日本語 Abc", "  Spaced  "])

        let c = try MetalStringArray(["a", "ab", "abc", "", "ünï", "日本"])
        XCTAssertEqual(try c.center(width: 4, pad: "*").toArray(), ["*a**", "*ab*", "abc*", "****", "ünï*", "*日本*"])
        XCTAssertEqual(try c.center(width: 5, pad: "*").toArray(), ["**a**", "*ab**", "*abc*", "*****", "*ünï*", "*日本**"])
        XCTAssertEqual(try c.center(width: 0, pad: "*").toArray(), ["a", "ab", "abc", "", "ünï", "日本"])

        let r = try MetalStringArray(["abcdef", "", "ab", "ünïcödé", "日本語です"])
        XCTAssertEqual(try r.replaceSlice(start: 1, stop: 3, with: "XY").toArray(),
                       ["aXYdef", "XY", "aXY", "üXYcödé", "日XYです"])
        XCTAssertEqual(try r.replaceSlice(start: 3, stop: 1, with: "XY").toArray(),
                       ["abcXYdef", "XY", "abXY", "ünïXYcödé", "日本語XYです"])
        XCTAssertEqual(try r.replaceSlice(start: -3, stop: -1, with: "XY").toArray(),
                       ["abcXYf", "XY", "XYb", "ünïcXYé", "日本XYす"])
        XCTAssertEqual(try r.replaceSlice(start: 10, stop: 12, with: "XY").toArray(),
                       ["abcdefXY", "XY", "abXY", "ünïcödéXY", "日本語ですXY"])
        // Byte indexing splits multi-byte characters, which is why the result is `binary`.
        let bin = try r.replaceSliceBytes(start: 3, stop: 1, with: Array("XY".utf8))
        XCTAssertTrue(bin.isBinary)
        XCTAssertEqual(bin.toArray(), ["abcXYdef", "XY", "abXY", "ünXYïcödé", "日XY本語です"])
    }

    func testTrimAndNormalizeTable() throws {
        try requireRealGPU()
        let t = try MetalStringArray(["xxhelloxx", "", "xyzabcxyz", "ààbonjourà", "\u{00A0}hi\u{00A0}", "aaa"])
        XCTAssertEqual(try t.utf8Trim(characters: "x").toArray(),
                       ["hello", "", "yzabcxyz", "ààbonjourà", "\u{00A0}hi\u{00A0}", "aaa"])
        XCTAssertEqual(try t.utf8Ltrim(characters: "xyz").toArray(),
                       ["helloxx", "", "abcxyz", "ààbonjourà", "\u{00A0}hi\u{00A0}", "aaa"])
        XCTAssertEqual(try t.utf8Rtrim(characters: "xyz").toArray(),
                       ["xxhello", "", "xyzabc", "ààbonjourà", "\u{00A0}hi\u{00A0}", "aaa"])
        XCTAssertEqual(try t.utf8Trim(characters: "aà").toArray(),
                       ["xxhelloxx", "", "xyzabcxyz", "bonjour", "\u{00A0}hi\u{00A0}", ""])
        XCTAssertEqual(try t.utf8Trim(characters: "").toArray(), t.toArray())

        let w = try MetalStringArray(["\u{00A0} hi \u{00A0}", "\u{2003}x\u{2003}", "\t a \n",
                                      "\u{200B}z\u{200B}", "", "   ", "\u{1C}q\u{1C}"])
        XCTAssertEqual(try w.utf8TrimWhitespace().toArray(),
                       ["hi", "x", "a", "\u{200B}z\u{200B}", "", "", "q"])
        XCTAssertEqual(try w.utf8LtrimWhitespace().toArray(),
                       ["hi \u{00A0}", "x\u{2003}", "a \n", "\u{200B}z\u{200B}", "", "", "q\u{1C}"])
        XCTAssertEqual(try w.utf8RtrimWhitespace().toArray(),
                       ["\u{00A0} hi", "\u{2003}x", "\t a", "\u{200B}z\u{200B}", "", "", "\u{1C}q"])

        let n = try MetalStringArray(["e\u{301}", "\u{e9}", "\u{FB01}x", "\u{2460}", ""])
        XCTAssertEqual(try n.utf8Normalize(.nfc).toArray(), ["\u{e9}", "\u{e9}", "\u{FB01}x", "\u{2460}", ""])
        XCTAssertEqual(try n.utf8Normalize(.nfd).toArray(), ["e\u{301}", "e\u{301}", "\u{FB01}x", "\u{2460}", ""])
        XCTAssertEqual(try n.utf8Normalize(.nfkc).toArray(), ["\u{e9}", "\u{e9}", "fix", "1", ""])
        XCTAssertEqual(try n.utf8Normalize(.nfkd).toArray(), ["e\u{301}", "e\u{301}", "fix", "1", ""])
    }

    // MARK: - Bulk shapes

    func testPredicatesAtEverySize() throws {
        try requireRealGPU()
        for n in Self.sizes {
            let rows = sample(n, seed: 11)
            let a = try MetalStringArray(rows)
            for p in StringPredicate.allCases {
                let got = try a.predicate(p).toArray()
                XCTAssertEqual(got.count, n, "\(p) length at \(n)")
                for (i, r) in rows.enumerated() {
                    guard let r else { XCTAssertNil(got[i], "\(p) null at \(i)/\(n)"); continue }
                    XCTAssertEqual(got[i], Self.refPredicate(p, r), "\(p) row \(i)/\(n): \(r.debugDescription)")
                }
            }
        }
    }

    /// The same predicates over a column with no byte ≥ 0x80: the CPU fixup must never run and the
    /// answers must be identical to the mixed column's.
    func testPredicatesAllASCII() throws {
        try requireRealGPU()
        for n in [1, 33, 4097] {
            let rows = asciiSample(n, seed: 5)
            let a = try MetalStringArray(rows)
            XCTAssertTrue(try a.isAllASCII())
            for p in StringPredicate.allCases {
                let got = try a.predicate(p).toArray()
                for (i, r) in rows.enumerated() {
                    guard let r else { XCTAssertNil(got[i]); continue }
                    XCTAssertEqual(got[i], Self.refPredicate(p, r), "\(p) row \(i)/\(n)")
                }
            }
        }
    }

    func testCaseTransformsAtEverySize() throws {
        try requireRealGPU()
        for n in Self.sizes {
            for rows in [sample(n, seed: 21), asciiSample(n, seed: 22)] {
                let a = try MetalStringArray(rows)
                XCTAssertEqual(try a.utf8Capitalize().toArray(), rows.map { $0.map(Self.refCapitalize) }, "capitalize at \(n)")
                XCTAssertEqual(try a.utf8Title().toArray(), rows.map { $0.map(Self.refTitle) }, "title at \(n)")
                XCTAssertEqual(try a.asciiTitle().toArray(), rows.map { $0.map(Self.refAsciiTitle) }, "ascii title at \(n)")
            }
        }
    }

    func testCenterAtEverySize() throws {
        try requireRealGPU()
        for n in Self.sizes {
            let rows = sample(n, seed: 31)
            let a = try MetalStringArray(rows)
            for (w, pad) in [(0, "*"), (1, "*"), (7, "*"), (12, "é"), (40, " ")] {
                XCTAssertEqual(try a.center(width: w, pad: pad).toArray(),
                               rows.map { $0.map { Self.refCenter($0, w, pad) } }, "center \(w)/\(pad) at \(n)")
            }
        }
        XCTAssertThrowsError(try MetalStringArray(["a"]).center(width: 4, pad: "ab"))
    }

    func testReplaceSliceAtEverySize() throws {
        try requireRealGPU()
        for n in Self.sizes {
            let rows = sample(n, seed: 41)
            let a = try MetalStringArray(rows)
            for (s, e) in [(0, 0), (1, 3), (2, 2), (3, 1), (-3, -1), (-1, 5), (10, 12), (0, Int(Int32.max))] {
                XCTAssertEqual(try a.replaceSlice(start: s, stop: e, with: "<>").toArray(),
                               rows.map { $0.map { Self.refReplaceSlice($0, s, e, "<>") } },
                               "replace_slice \(s)..\(e) at \(n)")
                let gotBytes = try a.replaceSliceBytes(start: s, stop: e, with: Array("<>".utf8))
                let want = rows.map { $0.map { Self.refReplaceSliceBytes($0, s, e, Array("<>".utf8)) } }
                for i in 0..<n {
                    guard let w = want[i] else { XCTAssertNil(gotBytes[i]); continue }
                    XCTAssertEqual(Array(gotBytes[i]!.utf8), Array(String(decoding: w, as: UTF8.self).utf8),
                                   "binary replace_slice \(s)..\(e) row \(i)/\(n)")
                }
            }
        }
    }

    func testTrimsAtEverySize() throws {
        try requireRealGPU()
        let asciiSet = "xa1 ", unicodeSet = "xàé日"
        for n in Self.sizes {
            for rows in [sample(n, seed: 51), asciiSample(n, seed: 52)] {
                let a = try MetalStringArray(rows)
                for set in [asciiSet, unicodeSet, ""] {
                    let member = Set(set.unicodeScalars)
                    XCTAssertEqual(try a.utf8Trim(characters: set).toArray(),
                                   rows.map { $0.map { Self.refTrim($0, left: true, right: true) { member.contains($0) } } },
                                   "trim \(set) at \(n)")
                    XCTAssertEqual(try a.utf8Ltrim(characters: set).toArray(),
                                   rows.map { $0.map { Self.refTrim($0, left: true, right: false) { member.contains($0) } } })
                    XCTAssertEqual(try a.utf8Rtrim(characters: set).toArray(),
                                   rows.map { $0.map { Self.refTrim($0, left: false, right: true) { member.contains($0) } } })
                }
                XCTAssertEqual(try a.utf8TrimWhitespace().toArray(),
                               rows.map { $0.map { Self.refTrim($0, left: true, right: true, Self.refIsSpace) } },
                               "trim whitespace at \(n)")
                XCTAssertEqual(try a.utf8LtrimWhitespace().toArray(),
                               rows.map { $0.map { Self.refTrim($0, left: true, right: false, Self.refIsSpace) } })
                XCTAssertEqual(try a.utf8RtrimWhitespace().toArray(),
                               rows.map { $0.map { Self.refTrim($0, left: false, right: true, Self.refIsSpace) } })
            }
        }
    }

    func testNormalizeAtEverySize() throws {
        try requireRealGPU()
        for n in Self.sizes {
            let rows = sample(n, seed: 61)
            let a = try MetalStringArray(rows)
            for form in UnicodeNormalizationForm.allCases {
                let got = try a.utf8Normalize(form)
                XCTAssertEqual(got.toArray(), rows.map { $0.map { form.apply($0) } }, "\(form) at \(n)")
                // Normalisation is idempotent, whichever form.
                XCTAssertEqual(try got.utf8Normalize(form).toArray(), got.toArray(), "\(form) idempotent at \(n)")
            }
        }
    }

    // MARK: - extract_regex_span

    func testExtractRegexSpan() throws {
        try requireRealGPU()
        let rows: [String?] = ["2024-05-06", "nope", nil, "1999-12-31", "", "ünïcödé2024-01-02"]
        let a = try MetalStringArray(rows)
        let spans = try a.extractRegexSpan(#"(?<y>\d{4})-(?<m>\d{2})"#)
        XCTAssertEqual(Set(spans.keys), ["y", "m"])
        XCTAssertEqual(spans["y"]!.start.toArray(), [0, nil, nil, 0, nil, 11])
        XCTAssertEqual(spans["y"]!.length.toArray(), [4, nil, nil, 4, nil, 4])
        XCTAssertEqual(spans["m"]!.start.toArray(), [5, nil, nil, 5, nil, 16])
        XCTAssertEqual(spans["m"]!.length.toArray(), [2, nil, nil, 2, nil, 2])
        XCTAssertThrowsError(try a.extractRegexSpan(#"\d+"#))

        // The span must always slice back to what extractRegex returns, at every size.
        for n in Self.sizes {
            let big = sample(n, seed: 71)
            let arr = try MetalStringArray(big)
            let pattern = #"(?<w>[A-Za-z]+)"#
            let text = try arr.extractRegex(pattern)["w"]!.toArray()
            let sp = try arr.extractRegexSpan(pattern)["w"]!
            let starts = sp.start.toArray(), lens = sp.length.toArray()
            for i in 0..<n {
                guard let row = big[i] else { XCTAssertNil(starts[i]); continue }
                guard let s = starts[i], let l = lens[i] else {
                    XCTAssertNil(text[i]); continue
                }
                let bytes = Array(row.utf8)
                XCTAssertEqual(String(decoding: bytes[Int(s)..<Int(s + l)], as: UTF8.self), text[i],
                               "span row \(i)/\(n)")
            }
        }
    }

    // MARK: - binary_join

    func testBinaryJoin() throws {
        try requireRealGPU()
        let child = try MetalStringArray(["a", "b", "c", "x", "p", nil, "q"])
        let list = try MetalListArray(counts: [3, 0, 1, nil, 3], values: .string(child))
        XCTAssertEqual(try list.binaryJoin(separator: "-").toArray(), ["a-b-c", "", "x", nil, nil])
        XCTAssertEqual(try list.binaryJoin(separator: "").toArray(), ["abc", "", "x", nil, nil])
        XCTAssertEqual(try list.binaryJoin(separator: "…").toArray(), ["a…b…c", "", "x", nil, nil])
        let seps = try MetalStringArray(["-", "+", "*", "/", "!"])
        XCTAssertEqual(try list.binaryJoin(separator: seps).toArray(), ["a-b-c", "", "x", nil, nil])
        let nullSep = try MetalStringArray(["-", nil, "*", "/", "!"])
        XCTAssertEqual(try list.binaryJoin(separator: nullSep).toArray(), ["a-b-c", nil, "x", nil, nil])
        XCTAssertEqual(try MetalStringArray.binaryJoin(list: list, separator: "-").toArray(),
                       ["a-b-c", "", "x", nil, nil])

        for n in Self.sizes {
            var g = RNG(81)
            let words = sample(n * 2, seed: 82, nullEvery: 23)
            var counts: [Int?] = []
            var used = 0
            for i in 0..<n {
                if i % 11 == 4 { counts.append(nil); continue }
                let c = Swift.min(Int(g.next() % 4), words.count - used)
                counts.append(c); used += c
            }
            let values = try MetalStringArray(Array(words.prefix(used)))
            let l = try MetalListArray(counts: counts, values: .string(values))
            let got = try l.binaryJoin(separator: "|").toArray()
            var pos = 0
            for i in 0..<n {
                guard let c = counts[i] else { XCTAssertNil(got[i], "null row \(i)/\(n)"); continue }
                let slice = Array(words[pos..<(pos + c)])
                pos += c
                if slice.contains(where: { $0 == nil }) { XCTAssertNil(got[i], "null element row \(i)/\(n)") }
                else { XCTAssertEqual(got[i], slice.map { $0! }.joined(separator: "|"), "row \(i)/\(n)") }
            }
        }
    }

    // MARK: - is_in / index_in

    func testIsInAndIndexIn() throws {
        try requireRealGPU()
        let probe = try MetalStringArray(["a", "b", nil, "c", "a"])
        let set = try MetalStringArray(["a", "c", nil, "a"])
        XCTAssertEqual(try probe.isIn(set).toArray(), [true, false, false, true, true])
        XCTAssertEqual(try probe.indexIn(set).toArray(), [0, nil, nil, 1, 0])
        // An empty value set, and a value set that is all nulls, match nothing.
        XCTAssertEqual(try probe.isIn(try MetalStringArray([])).toArray(), [false, false, false, false, false])
        XCTAssertEqual(try probe.indexIn([nil, nil]).toArray(), [nil, nil, nil, nil, nil])
        XCTAssertEqual(try probe.isIn(["b"]).toArray(), [false, true, false, false, false])

        for n in Self.sizes {
            let rows = sample(n, seed: 91)
            let setRows = sample(Swift.max(n / 8, 1), seed: 92, nullEvery: 5)
            let a = try MetalStringArray(rows)
            let s = try MetalStringArray(setRows)
            var first: [String: Int32] = [:]
            for (i, v) in setRows.enumerated() {
                if let v, first[v] == nil { first[v] = Int32(i) }
            }
            let gotIn = try a.isIn(s).toArray()
            let gotIdx = try a.indexIn(s).toArray()
            for i in 0..<n {
                let want = rows[i].flatMap { first[$0] }
                XCTAssertEqual(gotIn[i], want != nil, "is_in row \(i)/\(n)")
                XCTAssertEqual(gotIdx[i], want, "index_in row \(i)/\(n)")
            }
            XCTAssertEqual(try a.isIn(s).nullCount, 0)
        }
    }

    /// Strings whose 64-bit key collides cannot be told apart by the hash, only by the bytes: a value
    /// set built entirely of near-identical long strings exercises the probe's byte comparison and the
    /// lowest-index rule for duplicates.
    func testIsInDuplicatesAndLongStrings() throws {
        try requireRealGPU()
        let base = String(repeating: "abcdefgh", count: 40)
        var setRows: [String?] = []
        for i in 0..<500 { setRows.append(base + String(i)) }
        setRows += setRows                                    // every value twice: index_in takes the first
        let probeRows: [String?] = (0..<1000).map { i in i % 3 == 0 ? nil : base + String(i) }
        let a = try MetalStringArray(probeRows)
        let s = try MetalStringArray(setRows)
        let gotIn = try a.isIn(s).toArray(), gotIdx = try a.indexIn(s).toArray()
        for (i, r) in probeRows.enumerated() {
            guard let r else { XCTAssertEqual(gotIn[i], false); XCTAssertNil(gotIdx[i]); continue }
            let want = setRows.firstIndex(of: r).map { Int32($0) }
            XCTAssertEqual(gotIn[i], want != nil, "row \(i)")
            XCTAssertEqual(gotIdx[i], want, "row \(i)")
        }
    }

    // MARK: - Byte-indexed slicing, reversal and padding (Kernels/StringBytes.swift)

    /// Python's slice over an array of units — the semantics Arrow's `binary_slice` and
    /// `utf8_slice_codeunits` both implement, written here independently of the kernel.
    static func pySlice<T>(_ units: [T], _ start: Int, _ stop: Int?, _ step: Int) -> [T] {
        let n = units.count
        precondition(step != 0)
        var b: Int, e: Int
        if step > 0 {
            b = start < 0 ? Swift.max(n + start, 0) : Swift.min(start, n)
            e = stop.map { $0 < 0 ? Swift.max(n + $0, 0) : Swift.min($0, n) } ?? n
            guard b < e else { return [] }
            return stride(from: b, to: e, by: step).map { units[$0] }
        }
        b = start < 0 ? n + start : Swift.min(start, n - 1)
        if b < -1 { b = -1 }
        e = stop.map { $0 < 0 ? n + $0 : Swift.min($0, n - 1) } ?? -1
        if e < -1 { e = -1 }
        guard b > e else { return [] }
        return stride(from: b, through: e + 1, by: step).map { units[$0] }
    }

    static let sliceCases: [(Int, Int?, Int)] = [
        (0, nil, 1), (1, 4, 1), (-3, nil, 1), (1, -1, 1), (2, 2, 1), (0, 100, 3), (-100, 100, 2),
        (5, 0, -1), (-1, -5, -2), (-1, nil, -1), (0, nil, -1), (100, -100, -3), (9, 9, 1),
    ]

    func testBinarySliceMatchesPythonSlicing() throws {
        try requireRealGPU()
        for n in Self.sizes {
            let rows = sample(n, seed: UInt64(n) &+ 301)
            let a = try MetalStringArray(rows)
            for (start, stop, step) in Self.sliceCases {
                let got = try a.binarySlice(start: start, stop: stop, step: step).toByteArrays()
                let want = rows.map { $0.map { Self.pySlice(Array($0.utf8), start, stop, step) } }
                XCTAssertEqual(got, want, "binary_slice(\(start), \(String(describing: stop)), \(step)) at n=\(n)")
            }
        }
        XCTAssertThrowsError(try MetalStringArray(["a"]).binarySlice(start: 0, stop: 1, step: 0))
    }

    func testSliceCodeunitsWithStepMatchesPythonSlicing() throws {
        try requireRealGPU()
        for n in Self.sizes {
            let rows = sample(n, seed: UInt64(n) &+ 307)
            let a = try MetalStringArray(rows)
            for (start, stop, step) in Self.sliceCases {
                let got = try a.sliceCodeunits(start: start, stop: stop, step: step).toArray()
                let want = rows.map { s -> String? in
                    s.map { v in
                        var out = String.UnicodeScalarView()
                        for u in Self.pySlice(Array(v.unicodeScalars), start, stop, step) { out.append(u) }
                        return String(out)
                    }
                }
                XCTAssertEqual(got, want, "utf8_slice(\(start), \(String(describing: stop)), \(step)) at n=\(n)")
            }
        }
        XCTAssertThrowsError(try MetalStringArray(["a"]).sliceCodeunits(start: 0, stop: 1, step: 0))
    }

    /// `binary_reverse` reverses bytes always; `ascii_reverse` does the same but refuses non-ASCII
    /// input, as pyarrow does rather than emitting invalid UTF-8.
    func testByteReverseAndAsciiReverse() throws {
        try requireRealGPU()
        for n in Self.sizes {
            let rows = sample(n, seed: UInt64(n) &+ 311)
            let a = try MetalStringArray(rows)
            XCTAssertEqual(try a.binaryReverse().toByteArrays(),
                           rows.map { $0.map { Array($0.utf8).reversed().map { b in b } } },
                           "binary_reverse at n=\(n)")
            let ascii = asciiSample(n, seed: UInt64(n) &+ 313)
            let b = try MetalStringArray(ascii)
            XCTAssertEqual(try b.asciiReverse().toArray(),
                           ascii.map { $0.map { String(decoding: Array($0.utf8).reversed(), as: UTF8.self) } },
                           "ascii_reverse at n=\(n)")
            // On ASCII the byte reversal and the code point reversal are the same answer.
            XCTAssertEqual(try b.asciiReverse().toArray(), try b.reverse().toArray())
        }
        XCTAssertThrowsError(try MetalStringArray(["é"]).asciiReverse())
    }

    /// The whole point of the `ascii_*` / `utf8_*` padding pair: one counts bytes, the other counts
    /// code points, and they only agree on ASCII input.
    func testAsciiPaddingCountsBytesAndUtf8CountsCodePoints() throws {
        try requireRealGPU()
        for n in Self.sizes {
            let rows = sample(n, seed: UInt64(n) &+ 317)
            let a = try MetalStringArray(rows)
            for width in [0, 1, 8, 12] {
                XCTAssertEqual(try a.asciiLpad(width: width, pad: "*").toArray(),
                               rows.map { $0.map { Self.padBytes($0, width, "*", left: true, right: false) } },
                               "ascii_lpad \(width) at n=\(n)")
                XCTAssertEqual(try a.asciiRpad(width: width, pad: "*").toArray(),
                               rows.map { $0.map { Self.padBytes($0, width, "*", left: false, right: true) } },
                               "ascii_rpad \(width) at n=\(n)")
                XCTAssertEqual(try a.asciiCenter(width: width, pad: "*").toArray(),
                               rows.map { $0.map { Self.padBytes($0, width, "*", left: true, right: true) } },
                               "ascii_center \(width) at n=\(n)")
            }
        }
        // Six bytes, five code points: the two forms disagree by exactly one pad character.
        let one = try MetalStringArray(["héllo"])
        XCTAssertEqual(try one.asciiLpad(width: 8, pad: "*").toArray(), ["**héllo"])
        XCTAssertEqual(try one.padLeft(width: 8, pad: "*").toArray(), ["***héllo"])
        XCTAssertEqual(try one.asciiCenter(width: 8, pad: "*").toArray(), ["*héllo*"])
        XCTAssertEqual(try one.center(width: 8, pad: "*").toArray(), ["*héllo**"])
        XCTAssertThrowsError(try one.asciiLpad(width: 8, pad: "é"))
    }

    static func padBytes(_ s: String, _ width: Int, _ pad: String, left: Bool, right: Bool) -> String {
        let b = Array(s.utf8), p = Array(pad.utf8)
        let need = Swift.max(width - b.count, 0)
        let l = (left && right) ? need / 2 : (left ? need : 0)
        let r = (left && right) ? need - l : (right ? need : 0)
        var out: [UInt8] = []
        for _ in 0..<l { out += p }
        out += b
        for _ in 0..<r { out += p }
        return String(decoding: out, as: UTF8.self)
    }

    // MARK: - Full Unicode case mapping (Kernels/StringUnicode.swift)

    /// The simple (1:1) mappings, reconstructed from Swift's full ones. Independent of `UnicodeClass`
    /// so this is an oracle rather than an echo of the implementation.
    static func simpleUpper(_ u: Unicode.Scalar) -> Unicode.Scalar {
        if u.value == 0xDF { return Unicode.Scalar(0x1E9E)! }
        if (0x1F80...0x1F87).contains(u.value) || (0x1F90...0x1F97).contains(u.value)
            || (0x1FA0...0x1FA7).contains(u.value) { return Unicode.Scalar(u.value + 8)! }
        let m = u.properties.uppercaseMapping.unicodeScalars
        return m.count == 1 ? m.first! : u
    }
    static func simpleLower(_ u: Unicode.Scalar) -> Unicode.Scalar {
        if u.value == 0x130 { return Unicode.Scalar(0x69)! }
        let m = u.properties.lowercaseMapping.unicodeScalars
        return m.count == 1 ? m.first! : u
    }
    static func refCaseMap(_ op: UnicodeTransform, _ s: String) -> String {
        var out = String.UnicodeScalarView()
        var first = true, boundary = true
        for u in s.unicodeScalars {
            switch op {
            case .upper: out.append(simpleUpper(u))
            case .lower: out.append(simpleLower(u))
            case .swapcase:
                let up = isUpper(u), lo = isLower(u)
                out.append(up && lo ? u : (up ? simpleLower(u) : (lo ? simpleUpper(u) : u)))
            case .capitalize: out.append(first ? simpleUpper(u) : simpleLower(u))
            case .title:
                if isCased(u) { out.append(boundary ? simpleUpper(u) : simpleLower(u)); boundary = false }
                else { out.append(u); boundary = true }
            default: out.append(u)
            }
            first = false
        }
        return String(out)
    }

    /// The per-row GPU/host split has to give the same answer as the host alone, at every size and
    /// with Latin and non-Latin rows interleaved so both paths land inside one dispatch.
    func testUnicodeCaseMappingAtEverySize() throws {
        try requireRealGPU()
        let ops: [(UnicodeTransform, (MetalStringArray) throws -> MetalStringArray)] = [
            (.upper, { try $0.utf8Upper() }), (.lower, { try $0.utf8Lower() }),
            (.swapcase, { try $0.utf8Swapcase() }), (.capitalize, { try $0.utf8Capitalize() }),
            (.title, { try $0.utf8Title() }),
        ]
        for n in Self.sizes {
            let rows = sample(n, seed: UInt64(n) &+ 401)
            let a = try MetalStringArray(rows)
            for (op, call) in ops {
                XCTAssertEqual(try call(a).toArray(), rows.map { $0.map { Self.refCaseMap(op, $0) } },
                               "\(op) at n=\(n)")
            }
            // The same column with every non-ASCII row removed exercises the pure-GPU path.
            let ascii = asciiSample(n, seed: UInt64(n) &+ 403)
            let b = try MetalStringArray(ascii)
            for (op, call) in ops {
                XCTAssertEqual(try call(b).toArray(), ascii.map { $0.map { Self.refCaseMap(op, $0) } },
                               "\(op) ASCII at n=\(n)")
            }
        }
    }

    /// The exact answers pyarrow gives for the code points where a simple mapping and a full one
    /// differ, and for the titlecase letters that `swapcase` has to leave alone.
    func testUnicodeCaseMappingPinnedAnswers() throws {
        try requireRealGPU()
        let rows = ["Straße", "ǅungla", "Ǆungla", "ǆungla", "ΣΊΣΥΦΟΣ", "σίσυφος", "Привет",
                    "日本", "ǰ", "ﬁn", "İstanbul", "ﬀ", "ΐ", "ᾈ", "ŉ", "µ", "ĸ", "ſ", "ı", "Ÿ", "ÿ"]
        let a = try MetalStringArray(rows)
        XCTAssertEqual(try a.utf8Upper().toArray(),
                       ["STRAẞE", "ǄUNGLA", "ǄUNGLA", "ǄUNGLA", "ΣΊΣΥΦΟΣ", "ΣΊΣΥΦΟΣ", "ПРИВЕТ",
                        "日本", "ǰ", "ﬁN", "İSTANBUL", "ﬀ", "ΐ", "ᾈ", "ŉ", "Μ", "ĸ", "S", "I", "Ÿ", "Ÿ"])
        XCTAssertEqual(try a.utf8Lower().toArray(),
                       ["straße", "ǆungla", "ǆungla", "ǆungla", "σίσυφοσ", "σίσυφος", "привет",
                        "日本", "ǰ", "ﬁn", "istanbul", "ﬀ", "ΐ", "ᾀ", "ŉ", "µ", "ĸ", "ſ", "ı", "ÿ", "ÿ"])
        // A titlecase letter is both upper and lower for Arrow, so swapcase leaves it where it is.
        XCTAssertEqual(try a.utf8Swapcase().toArray().prefix(4).map { $0! },
                       ["sTRAẞE", "ǅUNGLA", "ǆUNGLA", "ǄUNGLA"])
        XCTAssertEqual(try a.utf8Capitalize().toArray().prefix(4).map { $0! },
                       ["Straße", "Ǆungla", "Ǆungla", "Ǆungla"])
        // Two bytes in, three out (ß -> ẞ) and two in, one out (ı -> I): both offsets must move.
        XCTAssertEqual(try MetalStringArray(["ß"]).utf8Upper().totalBytes, 3)
        XCTAssertEqual(try MetalStringArray(["ı"]).utf8Upper().totalBytes, 1)
    }

    /// The trims split per row too: an ASCII set never leaves the GPU, a set with a non-ASCII
    /// character sends only the non-ASCII rows to the host, and both must agree with the oracle.
    func testUnicodeTrimSplitsPerRow() throws {
        try requireRealGPU()
        let sets = ["", "Hlo", "éH  ", "\u{3000}x", "½"]
        for n in Self.sizes {
            let rows = sample(n, seed: UInt64(n) &+ 411)
            let a = try MetalStringArray(rows)
            XCTAssertEqual(try a.utf8TrimWhitespace().toArray(),
                           rows.map { $0.map { Self.refTrim($0, left: true, right: true, Self.refIsSpace) } },
                           "utf8_trim_whitespace at n=\(n)")
            XCTAssertEqual(try a.utf8LtrimWhitespace().toArray(),
                           rows.map { $0.map { Self.refTrim($0, left: true, right: false, Self.refIsSpace) } })
            XCTAssertEqual(try a.utf8RtrimWhitespace().toArray(),
                           rows.map { $0.map { Self.refTrim($0, left: false, right: true, Self.refIsSpace) } })
            for set in sets {
                let members = Set(set.unicodeScalars)
                XCTAssertEqual(try a.utf8Trim(characters: set).toArray(),
                               rows.map { $0.map { Self.refTrim($0, left: true, right: true) { members.contains($0) } } },
                               "utf8_trim \(set.debugDescription) at n=\(n)")
                XCTAssertEqual(try a.utf8Ltrim(characters: set).toArray(),
                               rows.map { $0.map { Self.refTrim($0, left: true, right: false) { members.contains($0) } } })
                XCTAssertEqual(try a.utf8Rtrim(characters: set).toArray(),
                               rows.map { $0.map { Self.refTrim($0, left: false, right: true) { members.contains($0) } } })
            }
        }
    }

    // MARK: - binary_join_element_wise over N columns

    func testJoinElementWiseNullHandlingAtEverySize() throws {
        try requireRealGPU()
        for n in Self.sizes {
            let cols = (0..<4).map { k in sample(n, seed: UInt64(n) &+ 500 &+ UInt64(k), nullEvery: 3 + k) }
            let arrays = try cols.map { try MetalStringArray($0) }
            for width in [1, 2, 3, 4] {
                let use = Array(arrays.prefix(width)), rows = Array(cols.prefix(width))
                for (mode, name) in [(JoinNullHandling.emitNull, "emit_null"),
                                     (.skip, "skip"), (.replace, "replace")] {
                    let got = try MetalStringArray.joinElementWise(use, separator: "-",
                                                                   nullHandling: mode,
                                                                   nullReplacement: "?").toArray()
                    let want = (0..<n).map { i -> String? in
                        let cells = rows.map { $0[i] }
                        switch mode {
                        case .emitNull:
                            return cells.contains(where: { $0 == nil }) ? nil
                                                                        : cells.map { $0! }.joined(separator: "-")
                        case .skip:
                            return cells.compactMap { $0 }.joined(separator: "-")
                        case .replace:
                            return cells.map { $0 ?? "?" }.joined(separator: "-")
                        }
                    }
                    XCTAssertEqual(got, want, "\(name) over \(width) columns at n=\(n)")
                }
            }
        }
        XCTAssertThrowsError(try MetalStringArray.joinElementWise([]))
        XCTAssertThrowsError(try MetalStringArray.joinElementWise([try MetalStringArray(["a"]),
                                                                   try MetalStringArray(["a", "b"])]))
    }

    /// An empty separator, an empty replacement and rows that are empty strings rather than nulls:
    /// the three ways a join can produce zero bytes and still be a valid row.
    func testJoinElementWiseEmptyCases() throws {
        try requireRealGPU()
        let a = try MetalStringArray(["", nil, "x", nil])
        let b = try MetalStringArray(["", "y", nil, nil])
        XCTAssertEqual(try MetalStringArray.joinElementWise([a, b], separator: "").toArray(),
                       ["", nil, nil, nil])
        XCTAssertEqual(try MetalStringArray.joinElementWise([a, b], separator: "-",
                                                            nullHandling: .skip).toArray(),
                       ["-", "y", "x", ""])
        XCTAssertEqual(try MetalStringArray.joinElementWise([a, b], separator: "-",
                                                            nullHandling: .replace).toArray(),
                       ["-", "-y", "x-", "-"])
        XCTAssertEqual(try MetalStringArray.joinElementWise([a], nullHandling: .replace,
                                                            nullReplacement: "?").toArray(),
                       ["", "?", "x", "?"])
    }

    // MARK: - extract_regex / extract_regex_span as struct columns

    func testExtractRegexStructShape() throws {
        try requireRealGPU()
        let rows: [String?] = ["a1", "b22", nil, "zz", "q7x"]
        let a = try MetalStringArray(rows)
        let s = try a.extractRegexStruct("(?<letter>[a-z])(?<digits>\\d+)")
        XCTAssertEqual(s.names, ["letter", "digits"])
        XCTAssertEqual(s.length, rows.count)
        XCTAssertEqual((0..<rows.count).map { s.isValid($0) }, [true, true, false, false, true])
        guard case .string(let letters) = s.children[0], case .string(let digits) = s.children[1] else {
            return XCTFail("extract_regex struct children should be utf8")
        }
        XCTAssertEqual(letters.toArray(), ["a", "b", nil, nil, "q"])
        XCTAssertEqual(digits.toArray(), ["1", "22", nil, nil, "7"])

        let span = try a.extractRegexSpanStruct("(?<letter>[a-z])(?<digits>\\d+)")
        XCTAssertEqual(span.names, ["letter", "digits"])
        XCTAssertEqual((0..<rows.count).map { span.isValid($0) }, [true, true, false, false, true])
        guard case .list(let letterSpans) = span.children[0] else {
            return XCTFail("extract_regex_span children should be fixed_size_list<int32>[2]")
        }
        XCTAssertEqual(letterSpans.kind, .fixedSize(2))
        guard case .int32(let flat) = letterSpans.values else { return XCTFail("child should be int32") }
        XCTAssertEqual(Array(flat.toArray().prefix(4)), [0, 1, 0, 1])   // rows 0 and 1: start 0, length 1
        XCTAssertFalse(letterSpans.isValid(2))
        XCTAssertThrowsError(try a.extractRegexStruct("[a-z]"))
    }
}
