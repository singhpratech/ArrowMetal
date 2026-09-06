import Foundation
import Metal

/// Regular expressions, SQL `LIKE` and splitting over Arrow `utf8`.
///
/// A backtracking engine is a poor fit for SIMT, so the matching itself runs on the **CPU** through
/// `NSRegularExpression` (ICU), sharded over `DispatchQueue.concurrentPerform` chunks of 4096 rows.
/// Every function here first asks whether the pattern is really a regex at all: a pattern with no
/// metacharacter is routed to the existing byte-wise GPU kernels (`contains` / `startsWith` /
/// `equals` / `endsWith`, `countSubstring`, `findSubstring`, `replaceSubstring`), which is the common
/// case in practice and keeps the data on the device.
///
/// ## Which patterns take the GPU fast path
///
/// * `matchSubstringRegex` / `countSubstringRegex` / `findSubstringRegex` / `replaceSubstringRegex`:
///   a pattern containing none of `\ . [ ] { } ( ) * + ? ^ $ |`, and `ignoreCase == false`. A pattern
///   of the form `^literal` also qualifies for `matchSubstringRegex` (ICU's `^` without
///   `.anchorsMatchLines` is exactly "start of input", so `startsWith` is an exact match).
/// * A trailing `$` deliberately does **not** take the fast path: ICU's `$` matches at the end of the
///   input *or immediately before a final line terminator*, so `abc$` matches `"abc\n"` while
///   `endsWith("abc")` does not. Those patterns go to the CPU engine.
/// * `matchLike`: a pattern whose only wildcards are a leading and/or trailing `%` — `abc`,
///   `abc%`, `%abc`, `%abc%` — routes to `equals` / `startsWith` / `endsWith` / `contains`. SQL `LIKE`
///   always anchors to the whole value, so those four are exact, newlines included.
///
/// ## Documented differences from pyarrow
///
/// * pyarrow uses RE2; this uses ICU. The syntaxes agree on the common constructs but differ at the
///   edges (ICU has backreferences and lookaround, RE2 does not; RE2 has `(?P<name>…)`, ICU spells
///   named groups `(?<name>…)`).
/// * `replaceSubstringRegex` takes an **ICU template**: capture groups are `$1`, `$2`, …, and a
///   literal `$` is written `\$`. RE2 (and therefore pyarrow) spells them `\1`, `\2`.
/// * `extractRegex` returns a dictionary of arrays rather than a struct array, because ArrowMetal has
///   no struct-typed column. A row that does not match is null in every group; a group that took part
///   in no alternative is the empty string.
/// * `splitPattern` / `splitWhitespace` return the `(offsets, values)` pair of an Arrow `list<utf8>`
///   rather than a list array, for the same reason.
extension MetalStringArray {

    // MARK: - Pattern analysis

    /// Bytes that make a pattern a real regular expression rather than a literal.
    static let regexMetacharacters: Set<UInt8> = Set(Array(#"\.[]{}()*+?^$|"#.utf8))

    /// True when every byte of `s` is safe to hand to the byte-wise GPU kernels verbatim.
    static func isRegexLiteral<S: StringProtocol>(_ s: S) -> Bool {
        !s.utf8.contains { regexMetacharacters.contains($0) }
    }

    /// The GPU predicate a regex reduces to, or nil when the CPU engine is needed.
    static func literalPredicate(_ pattern: String, ignoreCase: Bool) -> (Predicate, String)? {
        guard !ignoreCase else { return nil }
        if pattern.hasPrefix("^") {
            let rest = String(pattern.dropFirst())
            return isRegexLiteral(rest) ? (.startsWith, rest) : nil
        }
        return isRegexLiteral(pattern) ? (.contains, pattern) : nil
    }

    /// Compiles a pattern, turning ICU's error into an `ArrowMetalError`.
    static func compileRegex(_ pattern: String, ignoreCase: Bool,
                             dotMatchesNewlines: Bool = false) throws -> NSRegularExpression {
        var opts: NSRegularExpression.Options = []
        if ignoreCase { opts.insert(.caseInsensitive) }
        if dotMatchesNewlines { opts.insert(.dotMatchesLineSeparators) }
        do { return try NSRegularExpression(pattern: pattern, options: opts) }
        catch {
            throw ArrowMetalError.invalidArrowArray("invalid regular expression \"\(pattern)\": \(error.localizedDescription)")
        }
    }

    // MARK: - Parallel row iteration

    /// Calls `body(row, decoded string or nil)` for every row, sharded over `DispatchQueue.concurrentPerform`.
    ///
    /// The chunk size is a multiple of 8 so that two workers never write to the same validity/value
    /// *byte* when the caller is packing a bitmap.
    func forEachRowConcurrently(chunk: Int = 4096, _ body: (Int, String?) -> Void) {
        let n = length
        guard n > 0 else { return }
        let o = offsets.typed(Int32.self), d = data.typed(UInt8.self)
        let bm = validity?.typed(UInt8.self)
        let chunks = (n + chunk - 1) / chunk
        func work(_ c: Int) {
            let lo = c * chunk, hi = Swift.min(lo + chunk, n)
            for i in lo..<hi {
                if let bm, !Bitmap.isSet(bm, i) { body(i, nil); continue }
                let start = Int(o[i]), count = Int(o[i + 1]) - Int(o[i])
                body(i, String(decoding: UnsafeBufferPointer(start: d + start, count: count), as: UTF8.self))
            }
        }
        if chunks == 1 { work(0) } else { DispatchQueue.concurrentPerform(iterations: chunks, execute: work) }
        withExtendedLifetime(self) {}
    }

    /// A boolean result: one bit per row, validity shared with the input.
    ///
    /// `candidates`, when given, is the GPU pre-filter from ``regexCandidates(_:ignoreCase:)``: a row
    /// it clears cannot match, so the engine is never asked about it and the bit stays false.
    private func booleanResult(candidates: [Bool]? = nil,
                               _ predicate: @escaping (String) -> Bool) throws -> MetalBooleanArray {
        let n = length
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 1),
                                                zeroed: true, context: context)
        let bits = out.mutableTyped(UInt8.self)
        forEachRowConcurrently { i, s in
            if let candidates, !candidates[i] { return }
            guard let s, predicate(s) else { return }
            Bitmap.set(bits, i)
        }
        return MetalBooleanArray(length: n, nullCount: nullCount, validity: validity, values: out, context: context)
    }

    /// An int32 result, validity shared with the input. A row the pre-filter clears takes `absent`.
    private func int32Result(candidates: [Bool]? = nil, absent: Int32 = 0,
                             _ value: @escaping (String) -> Int32) throws -> MetalArray<Int32> {
        let n = length
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 4, 4), zeroed: true, context: context)
        let p = out.mutableTyped(Int32.self)
        forEachRowConcurrently { i, s in
            if let candidates, !candidates[i] { p[i] = s == nil ? 0 : absent; return }
            p[i] = s.map(value) ?? 0
        }
        return MetalArray<Int32>(length: n, nullCount: nullCount, validity: validity, values: out, context: context)
    }

    /// A `utf8` result built from one new string per row (nil keeps the row null). A row the
    /// pre-filter clears has no match to rewrite, so it comes back unchanged.
    private func stringResult(candidates: [Bool]? = nil,
                              _ value: @escaping (String) -> String) throws -> MetalStringArray {
        let n = length
        var rows = [String?](repeating: nil, count: n)
        rows.withUnsafeMutableBufferPointer { buf in
            forEachRowConcurrently { i, s in
                if let candidates, !candidates[i] { buf[i] = s; return }
                buf[i] = s.map(value)
            }
        }
        return try MetalStringArray(rows, context: context)
    }

    /// Byte offset of a `String.Index` inside `s`, which is what `findSubstring` reports.
    private static func byteOffset(of index: String.Index, in s: String) -> Int32 {
        Int32(s.utf8.distance(from: s.utf8.startIndex, to: index))
    }

    private static func fullRange(_ s: String) -> NSRange { NSRange(s.startIndex..., in: s) }

    // MARK: - Arrow regex functions

    /// Arrow `match_substring_regex`: true where the pattern matches anywhere in the value.
    /// Nulls propagate. Runs on the GPU when the pattern is a literal or `^literal`, on the CPU otherwise.
    public func matchSubstringRegex(_ pattern: String, ignoreCase: Bool = false) throws -> MetalBooleanArray {
        if let (pred, literal) = Self.literalPredicate(pattern, ignoreCase: ignoreCase) {
            return try matches(pred, literal)
        }
        let re = try Self.compileRegex(pattern, ignoreCase: ignoreCase)
        let candidates = try regexCandidates(pattern, ignoreCase: ignoreCase)
        return try booleanResult(candidates: candidates) {
            re.firstMatch(in: $0, range: Self.fullRange($0)) != nil
        }
    }

    /// Arrow `count_substring_regex`: the number of non-overlapping matches per value.
    /// GPU when the pattern is a literal (routing to `countSubstring`), CPU otherwise.
    public func countSubstringRegex(_ pattern: String, ignoreCase: Bool = false) throws -> MetalArray<Int32> {
        if !ignoreCase, Self.isRegexLiteral(pattern) { return try countSubstring(pattern) }
        let re = try Self.compileRegex(pattern, ignoreCase: ignoreCase)
        let candidates = try regexCandidates(pattern, ignoreCase: ignoreCase)
        return try int32Result(candidates: candidates, absent: 0) {
            Int32(re.numberOfMatches(in: $0, range: Self.fullRange($0)))
        }
    }

    /// Arrow `find_substring_regex`: the **byte** offset of the first match, or -1 when there is none.
    /// GPU when the pattern is a literal (routing to `findSubstring`), CPU otherwise.
    public func findSubstringRegex(_ pattern: String, ignoreCase: Bool = false) throws -> MetalArray<Int32> {
        if !ignoreCase, Self.isRegexLiteral(pattern) { return try findSubstring(pattern) }
        let re = try Self.compileRegex(pattern, ignoreCase: ignoreCase)
        let candidates = try regexCandidates(pattern, ignoreCase: ignoreCase)
        return try int32Result(candidates: candidates, absent: -1) { s in
            guard let m = re.firstMatch(in: s, range: Self.fullRange(s)),
                  let r = Range(m.range, in: s) else { return -1 }
            return Self.byteOffset(of: r.lowerBound, in: s)
        }
    }

    /// Arrow `replace_substring_regex`: replaces non-overlapping matches, left to right.
    ///
    /// `replacement` is an **ICU template** — `$1`, `$2`, … refer to capture groups and `\$` is a
    /// literal dollar sign (RE2, and therefore pyarrow, spells these `\1`). `maxReplacements < 0`
    /// replaces every match. GPU when the pattern is a literal and the replacement has no `$`.
    public func replaceSubstringRegex(_ pattern: String, with replacement: String,
                                      maxReplacements: Int = -1,
                                      ignoreCase: Bool = false) throws -> MetalStringArray {
        if !ignoreCase, Self.isRegexLiteral(pattern), !pattern.isEmpty,
           !replacement.contains("$"), !replacement.contains("\\") {
            return try replaceSubstring(pattern, with: replacement, maxReplacements: maxReplacements)
        }
        let re = try Self.compileRegex(pattern, ignoreCase: ignoreCase)
        let limit = maxReplacements
        let candidates = try regexCandidates(pattern, ignoreCase: ignoreCase)
        return try stringResult(candidates: candidates) { s in
            var out = ""
            out.reserveCapacity(s.count)
            var last = s.startIndex
            var done = 0
            re.enumerateMatches(in: s, range: Self.fullRange(s)) { m, _, stop in
                guard let m, let r = Range(m.range, in: s) else { return }
                if limit >= 0 && done >= limit { stop.pointee = true; return }
                out += s[last..<r.lowerBound]
                out += re.replacementString(for: m, in: s, offset: 0, template: replacement)
                last = r.upperBound
                done += 1
            }
            out += s[last...]
            return out
        }
    }

    /// Arrow `extract_regex`, as one `utf8` array per **named** capture group.
    ///
    /// The pattern must contain at least one `(?<name>…)` group. A row that does not match is null in
    /// every returned array; a named group that took part in no alternative comes back as the empty
    /// string. Always CPU.
    public func extractRegex(_ pattern: String, ignoreCase: Bool = false) throws -> [String: MetalStringArray] {
        let names = Self.namedGroups(in: pattern)
        guard !names.isEmpty else {
            throw ArrowMetalError.invalidArrowArray("extractRegex needs at least one named group, e.g. (?<year>\\d+)")
        }
        let re = try Self.compileRegex(pattern, ignoreCase: ignoreCase)
        let n = length
        let candidates = try regexCandidates(pattern, ignoreCase: ignoreCase)
        var columns = [[String?]](repeating: [String?](repeating: nil, count: n), count: names.count)
        for k in columns.indices {
            columns[k].withUnsafeMutableBufferPointer { buf in
                let name = names[k]
                forEachRowConcurrently { i, s in
                    if let candidates, !candidates[i] { return }
                    guard let s, let m = re.firstMatch(in: s, range: Self.fullRange(s)) else { return }
                    let r = m.range(withName: name)
                    guard r.location != NSNotFound, let rr = Range(r, in: s) else { buf[i] = ""; return }
                    buf[i] = String(s[rr])
                }
            }
        }
        var result: [String: MetalStringArray] = [:]
        for (k, name) in names.enumerated() { result[name] = try MetalStringArray(columns[k], context: context) }
        return result
    }

    /// The `(?<name>…)` group names of a pattern, in order, skipping escaped parentheses and the
    /// `(?<=` / `(?<!` lookbehind forms.
    static func namedGroups(in pattern: String) -> [String] {
        var names: [String] = []
        let c = Array(pattern)
        var i = 0
        while i < c.count {
            if c[i] == "\\" { i += 2; continue }
            if c[i] == "(", i + 2 < c.count, c[i + 1] == "?", c[i + 2] == "<" {
                let j0 = i + 3
                if j0 < c.count, c[j0] == "=" || c[j0] == "!" { i += 3; continue }
                var j = j0, name = ""
                while j < c.count, c[j] != ">" { name.append(c[j]); j += 1 }
                if j < c.count, !name.isEmpty, !names.contains(name) { names.append(name) }
                i = j + 1
                continue
            }
            i += 1
        }
        return names
    }

    // MARK: - SQL LIKE

    /// One piece of a parsed SQL `LIKE` pattern.
    enum LikeToken: Equatable { case literal(String), anySequence, anyOne }

    /// Splits a `LIKE` pattern into literals and wildcards. `\` escapes the next character, so `\%`,
    /// `\_` and `\\` are literal. Runs of `%` collapse into one.
    static func parseLike(_ pattern: String) -> [LikeToken] {
        var tokens: [LikeToken] = []
        var literal = ""
        func flush() { if !literal.isEmpty { tokens.append(.literal(literal)); literal = "" } }
        var it = pattern.makeIterator()
        var pending: Character? = it.next()
        while let ch = pending {
            pending = it.next()
            switch ch {
            case "\\":
                if let esc = pending { literal.append(esc); pending = it.next() } else { literal.append("\\") }
            case "%":
                flush()
                if tokens.last != .anySequence { tokens.append(.anySequence) }
            case "_":
                flush(); tokens.append(.anyOne)
            default:
                literal.append(ch)
            }
        }
        flush()
        return tokens
    }

    /// The GPU predicate a `LIKE` pattern reduces to, or nil when it needs the regex engine.
    /// SQL `LIKE` matches the whole value, so these four are exact — newlines included.
    static func likePredicate(_ tokens: [LikeToken]) -> (Predicate, String)? {
        /// The single literal in `tokens[range]`, or nil when that slice is not exactly one literal.
        func literal(_ i: Int) -> String? {
            if case .literal(let l) = tokens[i] { return l }
            return nil
        }
        switch tokens.count {
        case 0:
            return (.equals, "")                                     // "" matches only the empty value
        case 1:
            if tokens[0] == .anySequence { return (.contains, "") }  // "%" matches everything
            if let l = literal(0) { return (.equals, l) }
        case 2:
            if let l = literal(0), tokens[1] == .anySequence { return (.startsWith, l) }
            if tokens[0] == .anySequence, let l = literal(1) { return (.endsWith, l) }
        case 3:
            if tokens[0] == .anySequence, let l = literal(1), tokens[2] == .anySequence {
                return (.contains, l)
            }
        default:
            break
        }
        return nil
    }

    /// The regex a `LIKE` pattern translates to: `\A`/`\z` anchors so a trailing newline cannot
    /// change the answer, `%` as `.*` and `_` as `.`, both with `.` matching line separators because
    /// SQL wildcards match every character.
    static func likeRegex(_ tokens: [LikeToken]) -> String {
        var out = "\\A"
        for t in tokens {
            switch t {
            case .literal(let l): out += NSRegularExpression.escapedPattern(for: l)
            case .anySequence: out += ".*"
            case .anyOne: out += "."
            }
        }
        return out + "\\z"
    }

    /// Arrow `match_like`: SQL `LIKE`, where `%` matches any run of characters and `_` exactly one.
    /// A backslash escapes `%`, `_` and itself. Nulls propagate.
    ///
    /// Every case-sensitive pattern runs on the **GPU**. A pure prefix / suffix / contains / equality
    /// pattern takes the byte-wise `startsWith` / `endsWith` / `contains` / `equals` kernel; anything
    /// else — a `_` anywhere, an interior `%`, any mixture — is compiled into the byte program in
    /// `Kernels/StringLike.swift` and matched by `lk_like`, where `_` and `%` count **code points**.
    /// Only `ignoreCase: true` still goes to ICU, as a `\A…\z` anchored regex on the host.
    public func matchLike(_ pattern: String, ignoreCase: Bool = false) throws -> MetalBooleanArray {
        let tokens = Self.parseLike(pattern)
        if !ignoreCase {
            if let (pred, literal) = Self.likePredicate(tokens) { return try matches(pred, literal) }
            return try likeMatch(program: Self.likeProgram(tokens))
        }
        let re = try Self.compileRegex(Self.likeRegex(tokens), ignoreCase: ignoreCase, dotMatchesNewlines: true)
        return try booleanResult { re.firstMatch(in: $0, range: Self.fullRange($0)) != nil }
    }

    // MARK: - Splitting

    /// The `(offsets, values)` pair of an Arrow `list<utf8>`: row `i` owns
    /// `values[offsets[i] ..< offsets[i + 1]]`. A null input row owns no pieces (its two offsets are
    /// equal); the caller pairs the pair with this array's validity bitmap to build a list array.
    public typealias SplitResult = (offsets: MetalArray<Int32>, values: MetalStringArray)

    /// Builds the pair from one `[String]` per row.
    private func splitResult(_ pieces: @escaping (String) -> [String]) throws -> SplitResult {
        let n = length
        var perRow = [[String]](repeating: [], count: n)
        perRow.withUnsafeMutableBufferPointer { buf in
            forEachRowConcurrently { i, s in if let s { buf[i] = pieces(s) } }
        }
        var offs = [Int32](repeating: 0, count: n + 1)
        var total = 0
        for i in 0..<n { offs[i] = Int32(total); total += perRow[i].count }
        offs[n] = Int32(total)
        var flat: [String?] = []
        flat.reserveCapacity(total)
        for row in perRow { for piece in row { flat.append(piece) } }
        let offsetsArray = try MetalArray<Int32>(offs, context: context)
        return (offsetsArray, try MetalStringArray(flat, context: context))
    }

    /// Arrow `split_pattern`: splits each value on the literal `pattern`, as a `list<utf8>`.
    ///
    /// Every occurrence makes a boundary, so a value that begins or ends with the pattern gains an
    /// empty end piece and a value with no occurrence comes back as one piece. `maxSplits < 0` means
    /// every occurrence; otherwise the first `maxSplits` are used, or the last `maxSplits` when
    /// `reverse` is set. An empty pattern is rejected, as it is in Arrow. Always **GPU**
    /// (`Kernels/StringSplit.swift`).
    public func splitPattern(_ pattern: String, maxSplits: Int = -1,
                             reverse: Bool = false) throws -> MetalListArray {
        guard !pattern.isEmpty else {
            throw ArrowMetalError.invalidArrowArray("splitPattern needs a non-empty pattern")
        }
        return try splitOnGPU(.literal, pattern: Array(pattern.utf8), maxSplits: maxSplits, reverse: reverse)
    }

    /// ``splitPattern(_:maxSplits:reverse:)`` as the flat `(offsets, values)` pair.
    public func splitPatternPair(_ pattern: String, maxSplits: Int = -1,
                                 reverse: Bool = false) throws -> SplitResult {
        try splitPattern(pattern, maxSplits: maxSplits, reverse: reverse).stringPair()
    }

    /// Arrow `split_pattern_regex`: splits each value on the matches of `pattern`, as a `list<utf8>`.
    ///
    /// Always CPU — the regex engine is a host backtracker. `reverse` is refused, exactly as Arrow
    /// refuses it ("Cannot split in reverse with regex"). An empty match is skipped rather than
    /// producing an empty piece, so a pattern that can match nothing terminates instead of looping.
    public func splitPatternRegex(_ pattern: String, maxSplits: Int = -1, reverse: Bool = false,
                                  ignoreCase: Bool = false) throws -> MetalListArray {
        guard !reverse else {
            throw ArrowMetalError.invalidArrowArray("cannot split in reverse with regex")
        }
        let re = try Self.compileRegex(pattern, ignoreCase: ignoreCase)
        let limit = maxSplits
        return try listFromPair(try splitResult { s in
            var out: [String] = []
            var last = s.startIndex
            var done = 0
            re.enumerateMatches(in: s, range: Self.fullRange(s)) { m, _, stop in
                guard let m, let r = Range(m.range, in: s), !r.isEmpty else { return }
                if limit >= 0 && done >= limit { stop.pointee = true; return }
                out.append(String(s[last..<r.lowerBound]))
                last = r.upperBound
                done += 1
            }
            out.append(String(s[last...]))
            return out
        })
    }

    /// ``splitPatternRegex(_:maxSplits:reverse:ignoreCase:)`` as the flat `(offsets, values)` pair.
    public func splitPatternRegexPair(_ pattern: String, maxSplits: Int = -1, reverse: Bool = false,
                                      ignoreCase: Bool = false) throws -> SplitResult {
        try splitPatternRegex(pattern, maxSplits: maxSplits, reverse: reverse,
                              ignoreCase: ignoreCase).stringPair()
    }

    /// Arrow `ascii_split_whitespace` (`unicode == false`) and `utf8_split_whitespace`
    /// (`unicode == true`), as a `list<utf8>`.
    ///
    /// A separator is a **maximal run** of whitespace, so leading and trailing whitespace each
    /// produce one empty piece and the empty string splits to one empty piece — Arrow's behaviour,
    /// and Python's `str.split(sep)` rather than its no-argument form. The ASCII class is the space
    /// and `\t`–`\r`; the Unicode class is ``UnicodeClass/isSpace(_:)``, which adds `Zs`/`Zl`/`Zp`,
    /// U+001C–U+001F and U+0085 and deliberately excludes U+200B.
    ///
    /// Both forms are **GPU**: the Unicode whitespace class is eighteen code points plus three
    /// control ranges, small enough to spell out in MSL, so the kernel decodes UTF-8 as it walks
    /// rather than handing the row to the host.
    ///
    /// **Difference from pyarrow (25.0.1):** with `unicode: true`, `reverse: true` and a finite
    /// `maxSplits`, pyarrow fails to merge a multi-byte whitespace character with the whitespace
    /// beside it — `"\\u0020\\u3000x"` split once in reverse gives it `[" ", "x"]` where the maximal
    /// run gives `["", "x"]`. This keeps the runs maximal in both directions.
    public func splitWhitespace(unicode: Bool = false, maxSplits: Int = -1,
                                reverse: Bool = false) throws -> MetalListArray {
        try splitOnGPU(unicode ? .unicodeWhitespace : .whitespace, pattern: [],
                       maxSplits: maxSplits, reverse: reverse)
    }

    /// ``splitWhitespace(unicode:maxSplits:reverse:)`` as the flat `(offsets, values)` pair.
    public func splitWhitespacePair(unicode: Bool = false, maxSplits: Int = -1,
                                    reverse: Bool = false) throws -> SplitResult {
        try splitWhitespace(unicode: unicode, maxSplits: maxSplits, reverse: reverse).stringPair()
    }

    // MARK: - CPU split helpers (also the test oracle)

    static func split(_ s: String, on pattern: String, maxSplits: Int, reverse: Bool) -> [String] {
        let hay = Array(s.utf8), needle = Array(pattern.utf8)
        guard !needle.isEmpty, hay.count >= needle.count else { return [s] }
        var hits: [Int] = []
        var i = 0
        while i + needle.count <= hay.count {
            if Array(hay[i..<(i + needle.count)]) == needle { hits.append(i); i += needle.count } else { i += 1 }
        }
        if maxSplits >= 0 && hits.count > maxSplits {
            hits = reverse ? Array(hits.suffix(maxSplits)) : Array(hits.prefix(maxSplits))
        }
        var out: [String] = []
        var last = 0
        for h in hits {
            out.append(String(decoding: hay[last..<h], as: UTF8.self))
            last = h + needle.count
        }
        out.append(String(decoding: hay[last...], as: UTF8.self))
        return out
    }

    /// The whitespace split of one value: cut at every maximal whitespace run, keeping the empty
    /// pieces a leading or trailing run produces. Also the oracle the tests hold the kernel to.
    static func splitWhitespace(_ s: String, unicode: Bool, maxSplits: Int, reverse: Bool) -> [String] {
        let scalars = Array(s.unicodeScalars)
        func isWS(_ u: Unicode.Scalar) -> Bool {
            unicode ? UnicodeClass.isSpace(u) : (u.value == 0x20 || (0x09...0x0D).contains(u.value))
        }
        func text(_ r: Range<Int>) -> String {
            var v = String.UnicodeScalarView()
            for k in r { v.append(scalars[k]) }
            return String(v)
        }
        var runs: [Range<Int>] = []                          // maximal whitespace runs
        var i = 0
        while i < scalars.count {
            if isWS(scalars[i]) {
                let start = i
                while i < scalars.count && isWS(scalars[i]) { i += 1 }
                runs.append(start..<i)
            } else { i += 1 }
        }
        if maxSplits >= 0 && runs.count > maxSplits {
            runs = reverse ? Array(runs.suffix(maxSplits)) : Array(runs.prefix(maxSplits))
        }
        var out: [String] = []
        var last = 0
        for r in runs {
            out.append(text(last..<r.lowerBound))
            last = r.upperBound
        }
        out.append(text(last..<scalars.count))
        return out
    }
}
