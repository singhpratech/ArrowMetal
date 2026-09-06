import Foundation
import Metal

/// Arrow `match_like` on the GPU for **every** pattern, and the literal pre-filter that keeps the
/// regex functions off the host for the rows that cannot match.
///
/// ## `match_like`
///
/// `Kernels/Regex.swift` still routes the four anchored shapes (`abc`, `abc%`, `%abc`, `%abc%`) to
/// the byte-wise `equals` / `startsWith` / `endsWith` / `contains` kernels, which are cheaper still.
/// Everything else — a `_` anywhere, an interior `%`, any mixture — is compiled into the byte program
/// `StringLikeSource` documents and matched by `lk_like`, one row per thread. Only
/// `ignoreCase: true` still needs ICU.
///
/// ## The regex pre-filter
///
/// A backtracking regex engine is a host cost per row. Most real patterns, though, contain a literal
/// run that **every** match must contain: `\\d{4}-cust-\\d+` must contain `-cust-`. ``requiredLiteral(_:)``
/// finds the longest such run with a deliberately conservative reading of the pattern, the GPU
/// `contains` kernel marks the rows that hold it, and only those rows reach ICU. Rows that do not
/// hold it take the answer a non-match implies — false, 0, -1, the value unchanged, or null — so the
/// result is identical to running the engine everywhere.
///
/// ### What the analysis will and will not claim
///
/// It only collects literal bytes at **paren depth 0**, so nothing inside a group — which may be
/// optional, alternated or a lookaround — is ever claimed. It gives up entirely (returns nil) on a
/// top-level `|`, on an inline flag group `(?i)` and on `\\Q`. A quantifier that can match zero times
/// (`*`, `?`, `{…}`) drops the character it binds to; `+` keeps it. A character class, a `.`, an
/// anchor and a class escape (`\\d`, `\\b`, …) all end the current run without contributing. A
/// backslash before a non-alphanumeric contributes that character literally. The result is used only
/// when it is at least two bytes long, since a one-byte filter rarely pays for the pass.
extension MetalStringArray {

    // MARK: - match_like on the GPU

    /// Compiles a parsed `LIKE` pattern into the byte program `lk_like` runs.
    static func likeProgram(_ tokens: [LikeToken]) -> [UInt8] {
        var prog: [UInt8] = []
        for t in tokens {
            switch t {
            case .literal(let l):
                var bytes = Array(l.utf8)[...]
                while !bytes.isEmpty {
                    let chunk = bytes.prefix(255)
                    prog.append(0)
                    prog.append(UInt8(chunk.count))
                    prog.append(contentsOf: chunk)
                    bytes = bytes.dropFirst(chunk.count)
                }
            case .anySequence: prog.append(1)
            case .anyOne: prog.append(2)
            }
        }
        return prog
    }

    /// Runs a compiled `LIKE` program over every row. Nulls propagate.
    func likeMatch(program: [UInt8]) throws -> MetalBooleanArray {
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 1),
                                                zeroed: true, context: ctx)
        let prog = try sxArgBuffer(program)
        if n > 0 {
            let p = try ctx.pipeline(source: StringLikeSource.source, function: "lk_like",
                                     cacheKey: "strl/lk_like")
            try ctx.run { enc in
                enc.setComputePipelineState(p)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                Dispatch.setLength(enc, n, nil, index: 2)
                enc.setBuffer(prog.mtl, offset: prog.offset, index: 3)
                Dispatch.setUInt(enc, program.count, index: 4)
                enc.setBuffer(out.mtl, offset: out.offset, index: 5)
                Dispatch.dispatch1D(enc, p, count: BitmapOps.words(bits: n))
            }
        }
        ctx.retainUntilFlush(prog)
        ctx.retainUntilFlush(self)
        return MetalBooleanArray(length: n, nullCount: nullCount, validity: validity, values: out, context: ctx)
    }

    // MARK: - The regex literal pre-filter

    /// The longest literal run every match of `pattern` must contain, or nil when the analysis
    /// cannot claim one. See the type comment for exactly what it does and does not claim.
    static func requiredLiteral(_ pattern: String) -> String? {
        if pattern.contains("\\Q") || pattern.contains("(?i") { return nil }
        let c = Array(pattern)
        var runs: [String] = []
        var cur = ""
        var depth = 0
        func flush() { if !cur.isEmpty { runs.append(cur); cur = "" } }
        var i = 0
        while i < c.count {
            let ch = c[i]
            switch ch {
            case "\\":
                guard i + 1 < c.count else { flush(); i += 1; continue }
                let next = c[i + 1]
                if next.isLetter || next.isNumber { flush() }        // \d \w \b \1 …: a class, not a literal
                else if depth == 0 { cur.append(next) } else { flush() }
                i += 2
                continue
            case "(":
                depth += 1; flush()
            case ")":
                depth -= 1; flush()
            case "[":
                flush()
                var j = i + 1
                if j < c.count, c[j] == "^" { j += 1 }
                if j < c.count, c[j] == "]" { j += 1 }               // a leading ] is literal
                while j < c.count, c[j] != "]" { if c[j] == "\\" { j += 1 }; j += 1 }
                i = j + 1
                continue
            case "|":
                if depth == 0 { return nil }
                flush()
            case "*", "?":
                if !cur.isEmpty { cur.removeLast() }                 // the bound character is optional
                flush()
            case "+":
                flush()                                              // the bound character is required
            case "{":
                if !cur.isEmpty { cur.removeLast() }                 // {0,n} may drop it; be conservative
                flush()
                var j = i + 1
                while j < c.count, c[j] != "}" { j += 1 }
                i = j + 1
                continue
            case ".", "^", "$":
                flush()
            default:
                if depth == 0 { cur.append(ch) } else { flush() }
            }
            i += 1
        }
        flush()
        let best = runs.max(by: { $0.utf8.count < $1.utf8.count })
        guard let best, best.utf8.count >= 2 else { return nil }
        return best
    }

    /// The rows a regex could possibly match, as a plain `[Bool]`, or nil when no pre-filter applies.
    ///
    /// Computed with the GPU `contains` kernel, so the scan over the bytes never leaves the device;
    /// only the one-bit-per-row answer comes back.
    func regexCandidates(_ pattern: String, ignoreCase: Bool) throws -> [Bool]? {
        guard !ignoreCase, let literal = Self.requiredLiteral(pattern) else { return nil }
        let mask = try matches(.contains, literal)
        try context.syncPoint()
        return withExtendedLifetime(mask) { () -> [Bool] in
            let bits = mask.values.typed(UInt8.self)
            return (0..<length).map { Bitmap.isSet(bits, $0) }
        }
    }
}
