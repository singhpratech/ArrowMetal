import XCTest
@testable import ArrowMetal

/// GPU string transforms checked element-for-element against Swift string operations.
///
/// The oracles below are deliberately written over UTF-8 bytes and unicode scalars rather than
/// `Character`s: the kernels work on bytes and code points, and Swift's grapheme-cluster view
/// would disagree with them on combining marks and ZWJ emoji by design.
final class StringTransformTests: XCTestCase {

    // MARK: - CPU oracles

    /// Simple (1:1) uppercase over U+0000–U+017F, which is the block the GPU table claims.
    static func cpUpper(_ c: UInt32) -> UInt32 {
        if c == 0xB5 { return 0x39C }
        if c == 0xDF { return 0x1E9E }
        if c >= 0x61 && c <= 0x7A { return c - 32 }
        if c >= 0xE0 && c <= 0xFE && c != 0xF7 { return c - 32 }
        if c == 0xFF { return 0x178 }
        if c == 0x131 { return 0x49 }
        if c == 0x17F { return 0x53 }
        if c >= 0x100 && c <= 0x137 { return c & ~1 }
        if c >= 0x139 && c <= 0x148 { return (c & 1) == 1 ? c : c - 1 }
        if c >= 0x14A && c <= 0x177 { return c & ~1 }
        if c >= 0x179 && c <= 0x17E { return (c & 1) == 1 ? c : c - 1 }
        return c
    }
    static func cpLower(_ c: UInt32) -> UInt32 {
        if c >= 0x41 && c <= 0x5A { return c + 32 }
        if c >= 0xC0 && c <= 0xDE && c != 0xD7 { return c + 32 }
        if c == 0x178 { return 0xFF }
        if c == 0x130 { return 0x69 }
        if c == 0x138 { return c }                 // kra has no case at all
        if c >= 0x100 && c <= 0x137 { return c | 1 }
        if c >= 0x139 && c <= 0x148 { return (c & 1) == 1 ? c + 1 : c }
        if c >= 0x14A && c <= 0x177 { return c | 1 }
        if c >= 0x179 && c <= 0x17E { return (c & 1) == 1 ? c + 1 : c }
        return c
    }
    /// Unicode's **simple** (1:1) case mapping, reconstructed from Swift's full ones exactly as the
    /// documentation on `Kernels/StringUnicode.swift` describes: a full mapping of one scalar is the
    /// simple mapping, a longer one leaves the code point alone, and ß / İ / the Greek
    /// iota-subscript blocks are the three places where the two disagree.
    ///
    /// Written here independently of `UnicodeClass` so the test is an oracle rather than an echo.
    static func refSimpleUpper(_ u: Unicode.Scalar) -> Unicode.Scalar {
        if u.value == 0xDF { return Unicode.Scalar(0x1E9E)! }
        if (0x1F80...0x1F87).contains(u.value) || (0x1F90...0x1F97).contains(u.value)
            || (0x1FA0...0x1FA7).contains(u.value) { return Unicode.Scalar(u.value + 8)! }
        let m = u.properties.uppercaseMapping.unicodeScalars
        return m.count == 1 ? m.first! : u
    }
    static func refSimpleLower(_ u: Unicode.Scalar) -> Unicode.Scalar {
        if u.value == 0x130 { return Unicode.Scalar(0x69)! }
        let m = u.properties.lowercaseMapping.unicodeScalars
        return m.count == 1 ? m.first! : u
    }
    static func refCase(_ s: String, upper: Bool) -> String {
        var v = String.UnicodeScalarView()
        for u in s.unicodeScalars { v.append(upper ? refSimpleUpper(u) : refSimpleLower(u)) }
        return String(v)
    }
    /// The old, Latin-blocks-only mapping, kept so `testCaseMappingStaysInsideTheCoveredBlocks`
    /// can prove the GPU table still agrees with it inside U+0000–U+017F.
    static func refLatinCase(_ s: String, upper: Bool) -> String {
        var v = String.UnicodeScalarView()
        for u in s.unicodeScalars { v.append(Unicode.Scalar(upper ? cpUpper(u.value) : cpLower(u.value))!) }
        return String(v)
    }
    static func refAsciiBytes(_ s: String, _ f: (Int, UInt8) -> UInt8) -> String {
        var b = Array(s.utf8)
        for i in b.indices { b[i] = f(i, b[i]) }
        return String(decoding: b, as: UTF8.self)
    }
    static func upB(_ b: UInt8) -> UInt8 { (b >= 0x61 && b <= 0x7A) ? b - 32 : b }
    static func loB(_ b: UInt8) -> UInt8 { (b >= 0x41 && b <= 0x5A) ? b + 32 : b }
    static func refAsciiUpper(_ s: String) -> String { refAsciiBytes(s) { _, b in upB(b) } }
    static func refAsciiLower(_ s: String) -> String { refAsciiBytes(s) { _, b in loB(b) } }
    static func refSwapcase(_ s: String) -> String {
        refAsciiBytes(s) { _, b in (b >= 0x61 && b <= 0x7A) ? b - 32 : ((b >= 0x41 && b <= 0x5A) ? b + 32 : b) }
    }
    static func refCapitalize(_ s: String) -> String { refAsciiBytes(s) { i, b in i == 0 ? upB(b) : loB(b) } }

    static func isWS(_ b: UInt8) -> Bool { b == 0x20 || (b >= 0x09 && b <= 0x0D) }
    static func refTrim(_ s: String, left: Bool, right: Bool, set: [UInt8]? = nil) -> String {
        let b = Array(s.utf8)
        func inSet(_ x: UInt8) -> Bool { set.map { $0.contains(x) } ?? isWS(x) }
        var lo = 0, hi = b.count
        if left { while lo < hi && inSet(b[lo]) { lo += 1 } }
        if right { while hi > lo && inSet(b[hi - 1]) { hi -= 1 } }
        return String(decoding: b[lo..<hi], as: UTF8.self)
    }
    static func refReplace(_ s: String, _ pattern: String, _ replacement: String, _ maxR: Int) -> String {
        let b = Array(s.utf8), p = Array(pattern.utf8), r = Array(replacement.utf8)
        if p.isEmpty { return s }
        var out: [UInt8] = [], i = 0, reps = 0
        out.reserveCapacity(b.count)
        while i < b.count {
            var hit = (maxR < 0 || reps < maxR) && i + p.count <= b.count
            if hit { for j in 0..<p.count where b[i + j] != p[j] { hit = false; break } }
            if hit { out += r; i += p.count; reps += 1 } else { out.append(b[i]); i += 1 }
        }
        return String(decoding: out, as: UTF8.self)
    }
    static func refSlice(_ s: String, _ start: Int, _ stop: Int) -> String {
        let cps = Array(s.unicodeScalars), n = cps.count
        let a = start < 0 ? Swift.max(n + start, 0) : Swift.min(start, n)
        let b = stop < 0 ? Swift.max(n + stop, 0) : Swift.min(stop, n)
        guard b > a else { return "" }
        return String(String.UnicodeScalarView(cps[a..<b]))
    }
    static func refPad(_ s: String, _ width: Int, _ pad: String, left: Bool) -> String {
        let k = Swift.max(0, width - s.unicodeScalars.count)
        let p = String(repeating: pad, count: k)
        return left ? p + s : s + p
    }
    static func refReverse(_ s: String) -> String { String(String.UnicodeScalarView(s.unicodeScalars.reversed())) }
    static func refCount(_ s: String, _ pattern: String) -> Int32 {
        let b = Array(s.utf8), p = Array(pattern.utf8)
        if p.isEmpty { return Int32(s.unicodeScalars.count + 1) }
        var i = 0, c = 0
        while i + p.count <= b.count {
            var hit = true
            for j in 0..<p.count where b[i + j] != p[j] { hit = false; break }
            if hit { c += 1; i += p.count } else { i += 1 }
        }
        return Int32(c)
    }
    static func refFind(_ s: String, _ pattern: String) -> Int32 {
        let b = Array(s.utf8), p = Array(pattern.utf8)
        if p.isEmpty { return 0 }
        var i = 0
        while i + p.count <= b.count {
            var hit = true
            for j in 0..<p.count where b[i + j] != p[j] { hit = false; break }
            if hit { return Int32(i) }
            i += 1
        }
        return -1
    }
    static func refClass(_ s: String, _ cls: StringClass) -> Bool {
        let b = Array(s.utf8)
        if b.isEmpty { return false }
        var anyCased = false
        for x in b {
            let lo = x >= 0x61 && x <= 0x7A, up = x >= 0x41 && x <= 0x5A, dg = x >= 0x30 && x <= 0x39
            switch cls {
            case .alnum: if !(lo || up || dg) { return false }
            case .alpha: if !(lo || up) { return false }
            case .digit: if !dg { return false }
            case .space: if !isWS(x) { return false }
            case .upper: if lo { return false }; if up { anyCased = true }
            case .lower: if up { return false }; if lo { anyCased = true }
            }
        }
        return (cls == .upper || cls == .lower) ? anyCased : true
    }

    // MARK: - Data

    /// Deterministic SplitMix64 so a failure is reproducible.
    struct RNG {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
        mutating func int(_ n: Int) -> Int { Int(next() % UInt64(n)) }
    }

    /// 200k rows mixing ASCII words, Latin-1 and Latin Extended-A letters, emoji, whitespace-padded
    /// strings, empty strings and nulls.
    static let corpus: [String?] = {
        var rng = RNG(state: 0xA11CE)
        let ascii = Array("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-.,!?\t")
        // Latin-1 Supplement and Latin Extended-A letters, including the shape-changing ones.
        let latin = Array("àéîõüÿÀÉÎÕÜßøØçÇñÑĀāĂăĆćČčĐđĘęĞğİıŁłŃńŘřŠšŤťŮůŹźŻżŽžſĸŉµ")
        let emoji = Array("🍇🚀🎉🐙🌍👩‍💻é\u{301}")
        var out: [String?] = []
        out.reserveCapacity(200_000)
        for i in 0..<200_000 {
            if i % 37 == 0 { out.append(nil); continue }
            if i % 23 == 0 { out.append(""); continue }
            let len = rng.int(18)
            var s = ""
            if i % 11 == 0 { s += String(repeating: " ", count: 1 + rng.int(3)) }
            for _ in 0..<len {
                switch rng.int(10) {
                case 0, 1: s.append(latin[rng.int(latin.count)])
                case 2: s.append(emoji[rng.int(emoji.count)])
                default: s.append(ascii[rng.int(ascii.count)])
                }
            }
            if i % 13 == 0 { s += String(repeating: "\t ", count: 1 + rng.int(2)) }
            out.append(s)
        }
        return out
    }()

    /// Small, hand-picked cases: empty, null, pattern longer than the string, multi-byte edges.
    let edge: [String?] = ["", nil, "a", "aaa", "aaaa", "ab", "ABC", "abc", "  padded  ", "\t\n\r x \r\n\t",
                           "ÿıİſß", "ĀāŽž", "🍇🚀", "e🍇f", "xxxx", "aXbXc", "  ", "0123", "Mixed Case 42"]

    func check(_ got: MetalStringArray, _ expect: [String?], _ what: String) {
        let g = got.toArray()
        XCTAssertEqual(g.count, expect.count, "\(what): length")
        guard g.count == expect.count else { return }
        for i in 0..<g.count where g[i] != expect[i] {
            XCTFail("\(what) row \(i): got \(String(describing: g[i])) expected \(String(describing: expect[i]))")
            return
        }
        XCTAssertEqual(got.nullCount, expect.filter { $0 == nil }.count, "\(what): null count")
    }

    // MARK: - Case mapping

    func testCaseMappingLarge() throws {
        try requireRealGPU()
        let rows = Self.corpus
        let a = try MetalStringArray(rows)
        check(try a.asciiUpper(), rows.map { $0.map(Self.refAsciiUpper) }, "ascii_upper")
        check(try a.asciiLower(), rows.map { $0.map(Self.refAsciiLower) }, "ascii_lower")
        check(try a.asciiSwapcase(), rows.map { $0.map(Self.refSwapcase) }, "ascii_swapcase")
        check(try a.asciiCapitalize(), rows.map { $0.map(Self.refCapitalize) }, "ascii_capitalize")
        check(try a.utf8Upper(), rows.map { $0.map { Self.refCase($0, upper: true) } }, "utf8_upper")
        check(try a.utf8Lower(), rows.map { $0.map { Self.refCase($0, upper: false) } }, "utf8_lower")
    }

    /// Inside the documented coverage, and away from the documented multi-character exceptions,
    /// the kernel agrees with Swift's own `uppercased()` / `lowercased()`.
    func testCaseMappingMatchesSwiftOnCoveredCharacters() throws {
        try requireRealGPU()
        // Every covered code point except ß, ŉ, µ and ĸ, whose full Unicode mappings expand or
        // move outside the covered blocks (see the doc comment on utf8Upper()).
        let excluded: Set<UInt32> = [0xDF, 0x149, 0xB5, 0x138, 0x130, 0x131, 0x17F]
        var covered: [String] = []
        for c in (0x41 as UInt32)...0x17F where !excluded.contains(c) {
            guard let u = Unicode.Scalar(c) else { continue }
            let s = String(Character(u))
            // Only assert where Swift's full mapping is itself a single scalar.
            if s.uppercased().unicodeScalars.count == 1 && s.lowercased().unicodeScalars.count == 1 {
                covered.append(s)
            }
        }
        XCTAssertGreaterThan(covered.count, 200)
        let a = try MetalStringArray(covered)
        XCTAssertEqual(try a.utf8Upper().toArray(), covered.map { $0.uppercased() })
        XCTAssertEqual(try a.utf8Lower().toArray(), covered.map { $0.lowercased() })
        // The irregular entries of the block: ß and µ now map (their simple uppercase leaves the
        // block), ŉ and ĸ do not (ŉ's full uppercase is two characters, ĸ has no case at all), and
        // ı / ſ / İ change the byte length.
        let odd = try MetalStringArray(["ß", "ŉ", "µ", "ĸ", "ı", "İ", "ſ", "ÿ", "Ÿ"])
        XCTAssertEqual(try odd.utf8Upper().toArray(), ["ẞ", "ŉ", "Μ", "ĸ", "I", "İ", "S", "Ÿ", "Ÿ"])
        XCTAssertEqual(try odd.utf8Lower().toArray(), ["ß", "ŉ", "µ", "ĸ", "ı", "i", "ſ", "ÿ", "ÿ"])
        // Two bytes in, one byte out: the output offsets must shrink.
        let shrink = try MetalStringArray(["ıſİ"])
        XCTAssertEqual(shrink.totalBytes, 6)
        XCTAssertEqual(try shrink.utf8Upper().totalBytes, 4)   // I(1) + S(1) + İ(2)
        XCTAssertEqual(try shrink.utf8Lower().totalBytes, 5)   // ı(2) + ſ(2) + i(1)
        // Above U+017F the host takes over, so Greek and Cyrillic map and the rest passes through.
        let beyond = try MetalStringArray(["ΑΒΓ", "АБВ", "日本語", "🍇"])
        XCTAssertEqual(try beyond.utf8Upper().toArray(), ["ΑΒΓ", "АБВ", "日本語", "🍇"])
        XCTAssertEqual(try beyond.utf8Lower().toArray(), ["αβγ", "абв", "日本語", "🍇"])
    }

    // MARK: - Trimming

    func testTrim() throws {
        try requireRealGPU()
        let rows = Self.corpus
        let a = try MetalStringArray(rows)
        check(try a.trim(), rows.map { $0.map { Self.refTrim($0, left: true, right: true) } }, "trim")
        check(try a.ltrim(), rows.map { $0.map { Self.refTrim($0, left: true, right: false) } }, "ltrim")
        check(try a.rtrim(), rows.map { $0.map { Self.refTrim($0, left: false, right: true) } }, "rtrim")
        // Against Swift's own whitespace trimming on the ASCII-whitespace subset.
        let ws = try MetalStringArray(["  a  ", "\t\tb", "c\n", "   ", "", "d"])
        XCTAssertEqual(try ws.trim().toArray(),
                       ["  a  ", "\t\tb", "c\n", "   ", "", "d"].map { $0.trimmingCharacters(in: .whitespacesAndNewlines) })
        // Character sets.
        let set = "aX."
        let bytes = Array(set.utf8)
        let e = try MetalStringArray(edge)
        check(try e.trim(characters: set), edge.map { $0.map { Self.refTrim($0, left: true, right: true, set: bytes) } }, "trim(set)")
        check(try e.ltrim(characters: set), edge.map { $0.map { Self.refTrim($0, left: true, right: false, set: bytes) } }, "ltrim(set)")
        check(try e.rtrim(characters: set), edge.map { $0.map { Self.refTrim($0, left: false, right: true, set: bytes) } }, "rtrim(set)")
        // Empty set is a no-op; a non-ASCII set is rejected.
        check(try e.trim(characters: ""), edge, "trim(empty set)")
        XCTAssertThrowsError(try e.trim(characters: "é"))
        // Trimming never splits a multi-byte character.
        let uni = try MetalStringArray(["aéa", "🍇a🍇"])
        XCTAssertEqual(try uni.trim(characters: "a").toArray(), ["é", "🍇a🍇"])
    }

    // MARK: - Predicates

    func testPredicates() throws {
        try requireRealGPU()
        let rows = Self.corpus
        let a = try MetalStringArray(rows)
        for (name, got, cls) in [("alnum", try a.asciiIsAlnum(), StringClass.alnum),
                                 ("alpha", try a.asciiIsAlpha(), .alpha),
                                 ("digit", try a.asciiIsDigit(), .digit),
                                 ("space", try a.asciiIsSpace(), .space),
                                 ("upper", try a.asciiIsUpper(), .upper),
                                 ("lower", try a.asciiIsLower(), .lower)] {
            XCTAssertEqual(got.toArray(), rows.map { $0.map { Self.refClass($0, cls) } }, name)
            XCTAssertEqual(got.nullCount, rows.filter { $0 == nil }.count, "\(name) nulls")
        }
        // Python/Arrow semantics spelled out.
        let s = try MetalStringArray(["", "abc", "ABC", "AbC", "123", "abc123", " \t", "a b", "é", "ABÉ", "42a"])
        XCTAssertEqual(try s.asciiIsAlnum().toArray(), [false, true, true, true, true, true, false, false, false, false, true])
        XCTAssertEqual(try s.asciiIsAlpha().toArray(), [false, true, true, true, false, false, false, false, false, false, false])
        XCTAssertEqual(try s.asciiIsDigit().toArray(), [false, false, false, false, true, false, false, false, false, false, false])
        XCTAssertEqual(try s.asciiIsSpace().toArray(), [false, false, false, false, false, false, true, false, false, false, false])
        XCTAssertEqual(try s.asciiIsUpper().toArray(), [false, false, true, false, false, false, false, false, false, true, false])
        XCTAssertEqual(try s.asciiIsLower().toArray(), [false, true, false, false, false, true, false, true, false, false, true])
    }

    // MARK: - Substring transforms

    func testReplaceSubstring() throws {
        try requireRealGPU()
        let rows = Self.corpus
        let a = try MetalStringArray(rows)
        // Longer output, shorter output, equal length, and a capped count.
        check(try a.replaceSubstring("a", with: "AAA"), rows.map { $0.map { Self.refReplace($0, "a", "AAA", -1) } }, "replace grow")
        check(try a.replaceSubstring("ab", with: ""), rows.map { $0.map { Self.refReplace($0, "ab", "", -1) } }, "replace shrink")
        check(try a.replaceSubstring("é", with: "e"), rows.map { $0.map { Self.refReplace($0, "é", "e", -1) } }, "replace multibyte")
        check(try a.replaceSubstring("a", with: "b", maxReplacements: 2),
              rows.map { $0.map { Self.refReplace($0, "a", "b", 2) } }, "replace capped")
        // Against Foundation for the unlimited case (no overlap subtleties with a 2-byte pattern).
        let f = try MetalStringArray(edge)
        XCTAssertEqual(try f.replaceSubstring("aa", with: "Z").toArray(),
                       edge.map { $0?.replacingOccurrences(of: "aa", with: "Z") })
        // Empty pattern is the identity, exactly as Foundation does it.
        XCTAssertEqual(try f.replaceSubstring("", with: "Z").toArray(),
                       edge.map { $0?.replacingOccurrences(of: "", with: "Z") })
        // Pattern longer than every string.
        XCTAssertEqual(try f.replaceSubstring("this-pattern-is-far-too-long", with: "!").toArray(), edge)
        // maxReplacements = 0 changes nothing.
        XCTAssertEqual(try f.replaceSubstring("a", with: "!", maxReplacements: 0).toArray(), edge)
        // Non-overlapping, left to right.
        let ov = try MetalStringArray(["aaaa", "aaaaa"])
        XCTAssertEqual(try ov.replaceSubstring("aa", with: "b").toArray(), ["bb", "bba"])
    }

    func testRepeatSliceAndPad() throws {
        try requireRealGPU()
        let rows = Self.corpus
        let a = try MetalStringArray(rows)
        check(try a.repeat(3), rows.map { $0.map { String(repeating: $0, count: 3) } }, "repeat 3")
        check(try a.repeat(0), rows.map { $0.map { _ in "" } }, "repeat 0")
        check(try a.repeat(1), rows, "repeat 1")
        XCTAssertThrowsError(try a.repeat(-1))

        for (s, e) in [(0, 3), (2, 5), (-3, Int.max), (0, Int.max), (-2, -1), (5, 2), (100, 200), (-100, 100), (3, 3)] {
            check(try a.sliceCodeunits(start: s, stop: e), rows.map { $0.map { Self.refSlice($0, s, e) } }, "slice(\(s),\(e))")
        }
        // Slicing lands on code point boundaries, never inside a multi-byte character.
        let uni = try MetalStringArray(["a🍇bé", "🍇🍇🍇"])
        XCTAssertEqual(try uni.sliceCodeunits(start: 1, stop: 3).toArray(), ["🍇b", "🍇🍇"])
        XCTAssertEqual(try uni.sliceCodeunits(start: -1).toArray(), ["é", "🍇"])
        XCTAssertEqual(try uni.sliceCodeunits(start: 0, stop: 0).toArray(), ["", ""])

        for w in [0, 1, 5, 20] {
            check(try a.padLeft(width: w), rows.map { $0.map { Self.refPad($0, w, " ", left: true) } }, "lpad \(w)")
            check(try a.padRight(width: w, pad: "*"), rows.map { $0.map { Self.refPad($0, w, "*", left: false) } }, "rpad \(w)")
        }
        // Padding counts code points, not bytes.
        let p = try MetalStringArray(["🍇", "ab"])
        XCTAssertEqual(try p.padLeft(width: 3, pad: "é").toArray(), ["éé🍇", "éab"])
        XCTAssertThrowsError(try p.padLeft(width: 3, pad: "ab"))
        XCTAssertThrowsError(try p.padLeft(width: 3, pad: ""))
    }

    func testReverse() throws {
        try requireRealGPU()
        let rows = Self.corpus
        let a = try MetalStringArray(rows)
        check(try a.reverse(), rows.map { $0.map(Self.refReverse) }, "reverse")
        // Code points, not grapheme clusters: the combining acute moves ahead of its base letter.
        let g = try MetalStringArray(["abc", "🍇a", "e\u{301}"])
        XCTAssertEqual(try g.reverse().toArray(), ["cba", "a🍇", "\u{301}e"])
        XCTAssertEqual(try a.reverse().reverse().toArray(), rows, "reverse is an involution")
    }

    // MARK: - Search

    func testCountAndFindSubstring() throws {
        try requireRealGPU()
        let rows = Self.corpus
        let a = try MetalStringArray(rows)
        for pat in ["a", "ab", "é", "🍇", "", "not-present-anywhere"] {
            XCTAssertEqual(try a.countSubstring(pat).toArray(), rows.map { $0.map { Self.refCount($0, pat) } }, "count(\(pat))")
            XCTAssertEqual(try a.findSubstring(pat).toArray(), rows.map { $0.map { Self.refFind($0, pat) } }, "find(\(pat))")
        }
        let s = try MetalStringArray(["banana", "", nil, "aaa", "🍇x🍇"])
        XCTAssertEqual(try s.countSubstring("na").toArray(), [2, 0, nil, 0, 0])
        XCTAssertEqual(try s.countSubstring("aa").toArray(), [0, 0, nil, 1, 0])   // non-overlapping
        XCTAssertEqual(try s.findSubstring("na").toArray(), [2, -1, nil, -1, -1])
        XCTAssertEqual(try s.findSubstring("🍇").toArray(), [-1, -1, nil, -1, 0]) // byte offsets
        XCTAssertEqual(try s.findSubstring("x").toArray(), [-1, -1, nil, -1, 4])
        XCTAssertEqual(try s.countSubstring("").toArray(), [7, 1, nil, 4, 4])     // code points + 1
        XCTAssertEqual(try s.findSubstring("").toArray(), [0, 0, nil, 0, 0])
        XCTAssertEqual(try s.findSubstring("longer than the row").toArray(), [-1, -1, nil, -1, -1])
        XCTAssertEqual(try s.countSubstring("na").nullCount, 1)
    }

    // MARK: - Concat

    func testConcat() throws {
        try requireRealGPU()
        let rows = Self.corpus
        let other = Array(rows.reversed())
        let a = try MetalStringArray(rows), b = try MetalStringArray(other)
        let expect = zip(rows, other).map { x, y -> String? in
            guard let x, let y else { return nil }
            return x + "-" + y
        }
        check(try a.concat(b, separator: "-"), expect, "concat")
        let noSep = zip(rows, other).map { x, y -> String? in
            guard let x, let y else { return nil }
            return x + y
        }
        check(try a.concat(b), noSep, "concat no separator")
        // No nulls on either side: no validity bitmap at all.
        let c = try MetalStringArray(["a", "b"]), d = try MetalStringArray(["x", "🍇"])
        let cd = try c.concat(d, separator: "|")
        XCTAssertEqual(cd.toArray(), ["a|x", "b|🍇"])
        XCTAssertNil(cd.validity)
        XCTAssertEqual(cd.nullCount, 0)
        // Length mismatch is an error.
        XCTAssertThrowsError(try c.concat(try MetalStringArray(["only-one"])))
        // Empty input.
        let e = try MetalStringArray([String?]())
        XCTAssertEqual(try e.concat(e, separator: ",").length, 0)
    }

    // MARK: - Edge cases and structure

    func testEmptyAndAllNull() throws {
        try requireRealGPU()
        let empty = try MetalStringArray([String?]())
        XCTAssertEqual(try empty.utf8Upper().length, 0)
        XCTAssertEqual(try empty.trim().length, 0)
        XCTAssertEqual(try empty.reverse().totalBytes, 0)
        XCTAssertEqual(try empty.countSubstring("a").length, 0)
        XCTAssertEqual(try empty.asciiIsAlnum().length, 0)
        let nulls = try MetalStringArray([nil, nil, nil] as [String?])
        XCTAssertEqual(try nulls.utf8Upper().toArray(), [nil, nil, nil])
        XCTAssertEqual(try nulls.utf8Upper().totalBytes, 0)
        XCTAssertEqual(try nulls.repeat(4).toArray(), [nil, nil, nil])
        XCTAssertEqual(try nulls.asciiIsUpper().toArray(), [nil, nil, nil])
        // A null row contributes no bytes even when the transform grows every other row.
        let mixed = try MetalStringArray(["ab", nil, "cd"])
        let grown = try mixed.replaceSubstring("a", with: "aaaa")
        XCTAssertEqual(grown.toArray(), ["aaaab", nil, "cd"])
        XCTAssertEqual(grown.totalBytes, 7)
        XCTAssertEqual(grown.nullCount, 1)
    }

    func testTransformsRoundTripThroughFilterAndTake() throws {
        try requireRealGPU()
        let rows = Self.corpus
        let a = try MetalStringArray(rows)
        let upper = try a.utf8Upper()
        let mask = try upper.startsWith("A")
        XCTAssertEqual(try upper.filter(mask).toArray(),
                       rows.compactMap { $0.map { Self.refCase($0, upper: true) } }.filter { $0.hasPrefix("A") })
        let idx = try MetalArray<Int32>([5, 0, 200, nil])
        XCTAssertEqual(try upper.take(idx).toArray(),
                       [5, 0, 200].map { rows[$0].map { Self.refCase($0, upper: true) } } + [nil])
    }

    // MARK: - Batching

    func testBatchedMatchesUnbatched() throws {
        try requireRealGPU()
        let rows = Array(Self.corpus.prefix(50_000))
        let a = try MetalStringArray(rows)
        let ctx = MetalContext.shared

        let plainUpper = try a.utf8Upper().toArray()
        let plainTrim = try a.trim().toArray()
        let plainReplace = try a.replaceSubstring("a", with: "AA").toArray()
        let plainPad = try a.padLeft(width: 12, pad: "*").toArray()
        let plainReverse = try a.reverse().toArray()
        let plainConcat = try a.concat(a, separator: "+").toArray()
        let plainCount = try a.countSubstring("a").toArray()
        let plainClass = try a.asciiIsAlnum().toArray()

        let (u, t, r, p, v, c, cnt, cl) = try ctx.batch {
            (try a.utf8Upper(), try a.trim(), try a.replaceSubstring("a", with: "AA"),
             try a.padLeft(width: 12, pad: "*"), try a.reverse(), try a.concat(a, separator: "+"),
             try a.countSubstring("a"), try a.asciiIsAlnum())
        }
        XCTAssertEqual(u.toArray(), plainUpper)
        XCTAssertEqual(t.toArray(), plainTrim)
        XCTAssertEqual(r.toArray(), plainReplace)
        XCTAssertEqual(p.toArray(), plainPad)
        XCTAssertEqual(v.toArray(), plainReverse)
        XCTAssertEqual(c.toArray(), plainConcat)
        XCTAssertEqual(cnt.toArray(), plainCount)
        XCTAssertEqual(cl.toArray(), plainClass)

        // Chained inside one batch: the intermediate never leaves the GPU.
        let chained = try ctx.batch { try a.trim().utf8Upper().reverse() }
        XCTAssertEqual(chained.toArray(), rows.map { $0.map { Self.refReverse(Self.refCase(Self.refTrim($0, left: true, right: true), upper: true)) } })
    }
}
