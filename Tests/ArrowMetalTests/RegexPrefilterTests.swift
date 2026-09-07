import XCTest
@testable import ArrowMetal

/// The literal pre-filter in `Kernels/StringLike.swift`. It is an optimisation: a row whose bit the
/// filter clears is never re-examined, so the analysis is only sound while it *under*-claims. These
/// tests pin the ways it over-claimed, each against `NSRegularExpression` — the very engine the
/// unfiltered path uses, so a disagreement is a wrong answer and not a difference of dialect.
final class RegexPrefilterTests: XCTestCase {
    /// What the row would answer with no pre-filter at all.
    private func icu(_ pattern: String, _ s: String) -> Bool {
        guard let re = try? NSRegularExpression(pattern: pattern) else { return false }
        return re.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)) != nil
    }

    private func assertAgreesWithICU(_ pattern: String, _ rows: [String],
                                     file: StaticString = #filePath, line: UInt = #line) throws {
        let a = try MetalStringArray(rows.map { Optional($0) }, context: .shared)
        let got = try a.matchSubstringRegex(pattern).toArray().map { $0 ?? false }
        XCTAssertEqual(got, rows.map { icu(pattern, $0) }, "pattern \(pattern)", file: file, line: line)
    }

    /// `\u002D` is one escape standing for `-`, not `\u` followed by the four literal characters
    /// `002D`. The scanner consumed only two characters and then claimed `002Dnum` as a literal every
    /// match must contain, so the row that really matches was cleared.
    func testFixedWidthEscapesAreNotClaimedAsLiterals() throws {
        try requireRealGPU()
        XCTAssertNil(MetalStringArray.requiredLiteral(#"id\u002Dnum"#),
                     "a fixed-width escape must abandon the analysis, not contribute its digits")
        try assertAgreesWithICU(#"id\u002Dnum"#, ["id-num", "id002Dnum", "other"])
        try assertAgreesWithICU(#"a\x41bcd"#, ["aAbcd", "a41bcd", "zzz"])
    }

    /// The character-class skip stopped at the first `]`, so the outer `]` of `[[:alpha:]]` and of a
    /// nested set fell through to the literal branch and was appended to the claimed run.
    func testNestedAndPosixCharacterClassesAreSkippedWhole() throws {
        try requireRealGPU()
        XCTAssertEqual(MetalStringArray.requiredLiteral("[[:alpha:]]xyz"), "xyz")
        XCTAssertEqual(MetalStringArray.requiredLiteral("[[a-z][0-9]]wxyz"), "wxyz")
        try assertAgreesWithICU("[[:alpha:]]xyz", ["axyz", "]xyz", "1xyz"])
        try assertAgreesWithICU("[[a-z][0-9]]wxyz", ["awxyz", "]wxyz", "_wxyz"])
    }

    /// Under `(?x)` ICU ignores unescaped whitespace in the pattern; the scanner claimed it.
    func testExtendedModeIsNotAnalysed() throws {
        try requireRealGPU()
        XCTAssertNil(MetalStringArray.requiredLiteral("(?x) ab cd"))
        try assertAgreesWithICU("(?x) ab cd", ["abcd", " ab cd", "zz"])
    }

    /// Patterns the analysis handles correctly must keep working, and the filter must still fire.
    func testOrdinaryPatternsStillAgreeAndStillFilter() throws {
        try requireRealGPU()
        XCTAssertEqual(MetalStringArray.requiredLiteral("abc+def"), "abc")
        XCTAssertEqual(MetalStringArray.requiredLiteral("[a-z]+ghij"), "ghij")
        XCTAssertEqual(MetalStringArray.requiredLiteral(#"foo\.bar"#), "foo.bar")
        XCTAssertEqual(MetalStringArray.requiredLiteral("[]]abcd"), "abcd", "a leading ] is literal")
        try assertAgreesWithICU("abc+def", ["abcdef", "abccdef", "abdef"])
        try assertAgreesWithICU("[a-z]+ghij", ["xghij", "GHIJ", "ghij"])
        try assertAgreesWithICU(#"foo\.bar"#, ["foo.bar", "fooXbar", "nope"])
    }
}
