import Foundation
import Metal

/// The remaining Arrow string predicates and transforms: the `utf8_is_*` family, `string_is_ascii`,
/// `ascii_is_printable` / `ascii_is_title`, `utf8_capitalize` / `ascii_title` / `utf8_title`,
/// `ascii_center` / `utf8_center`, `binary_replace_slice` / `utf8_replace_slice`, the `utf8_trim*`
/// family, `utf8_normalize` and `extract_regex_span`.
///
/// ## Where the work runs
///
/// Byte-level work runs on the GPU; anything that needs a Unicode table runs on the CPU, sharded over
/// `DispatchQueue.concurrentPerform` chunks of 4096 rows exactly as `Kernels/Regex.swift` does.
///
/// The `utf8_is_*` predicates split **per row**. One GPU pass (`sx_pred`) answers every row and, in a
/// second bitmap, reports which rows carry a byte ≥ 0x80. A string made only of bytes < 0x80 is
/// classified identically by the byte rules and by the Unicode tables, so the host only revisits the
/// rows that second bitmap marks — an all-ASCII column never touches the CPU at all.
///
/// The case transforms and the `utf8_trim*` family split **per row** as well, in
/// `Kernels/StringUnicode.swift`: an exact GPU table covers every code point at or below U+017F, and
/// the length kernel declines any row above it so the host can redo just those.
///
/// ## Documented differences from pyarrow
///
/// pyarrow classifies with utf8proc; this uses Swift's `Unicode.Scalar.Properties`, so the two agree
/// wherever their Unicode data versions agree. Three deliberate reconstructions of utf8proc's rules:
///
/// * **Cased.** utf8proc calls a code point upper case when its simple lower-case mapping changes it
///   or its category is `Lt`, and lower case when its simple upper-case mapping changes it or its
///   category is `Ll`, minus the Roman numerals U+2160–U+216F. This file spells those as
///   `changesWhenLowercased` / `changesWhenUppercased` plus the same category and range tests. That
///   is *not* Unicode's `Uppercase` / `Lowercase` derived properties, which would (wrongly, for
///   Arrow) also call modifier letters such as U+02B0 (ʰ) lower case. A titlecase letter is both
///   upper and lower for Arrow, which is why `utf8_is_upper("ǅ")` and `utf8_is_lower("ǅ")` are both
///   false.
/// * **Whitespace.** Arrow's Unicode whitespace is the `Zs`/`Zl`/`Zp` categories plus U+0009–U+000D,
///   U+001C–U+001F and U+0085. U+200B (zero-width space, category `Cf`) is deliberately not
///   whitespace. Note U+001C–U+001F are *not* ASCII whitespace for `ascii_is_space`, but they are for
///   `utf8_is_space` and for the `utf8_trim*` whitespace family.
/// * **Printable.** Everything except the `Cc`, `Cf`, `Cs`, `Co`, `Cn`, `Zs`, `Zl` and `Zp`
///   categories, with U+0020 (the space) added back. The empty string is printable; every other
///   predicate here is false on it.
///
/// Case *mapping* (`utf8_upper`, `utf8_lower`, `utf8_swapcase`, `utf8_capitalize`, `utf8_title`) uses
/// Unicode's **simple** 1:1 mappings, as utf8proc does, reconstructed from Swift's full mappings: a full mapping of exactly one scalar is
/// the simple mapping, a longer one leaves the code point alone, and U+00DF (ß → ẞ), U+0130 (İ → i)
/// and the Greek iota-subscript blocks U+1F80–U+1F87 / U+1F90–U+1F97 / U+1FA0–U+1FA7 (which map +8)
/// are the exceptions where the two disagree. Everything else that Swift would expand to several
/// characters — U+0149 (ŉ), U+01F0 (ǰ), U+1E96 (ẖ), U+0587 (և) … — is left unchanged, which is what
/// utf8proc does too.
public enum UnicodeClass {

    static func category(_ u: Unicode.Scalar) -> Unicode.GeneralCategory { u.properties.generalCategory }

    /// utf8proc's `IsUpperCaseCharacterUnicode`: the simple lower-case mapping changes it, or it is
    /// a titlecase letter.
    public static func isUpper(_ u: Unicode.Scalar) -> Bool {
        u.properties.changesWhenLowercased || category(u) == .titlecaseLetter
    }
    /// utf8proc's `IsLowerCaseCharacterUnicode`: the simple upper-case mapping changes it or it is a
    /// lower-case letter — minus the uppercase Roman numerals, which Arrow excludes explicitly.
    public static func isLower(_ u: Unicode.Scalar) -> Bool {
        if (0x2160...0x216F).contains(u.value) { return false }
        return u.properties.changesWhenUppercased || category(u) == .lowercaseLetter
    }
    public static func isCased(_ u: Unicode.Scalar) -> Bool { isUpper(u) || isLower(u) }

    public static func isAlpha(_ u: Unicode.Scalar) -> Bool {
        switch category(u) {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter: return true
        default: return false
        }
    }
    public static func isDecimal(_ u: Unicode.Scalar) -> Bool { category(u) == .decimalNumber }
    public static func isDigit(_ u: Unicode.Scalar) -> Bool {
        let c = category(u)
        return c == .decimalNumber || c == .otherNumber
    }
    public static func isNumeric(_ u: Unicode.Scalar) -> Bool {
        switch category(u) {
        case .decimalNumber, .letterNumber, .otherNumber: return true
        default: return false
        }
    }
    public static func isAlnum(_ u: Unicode.Scalar) -> Bool { isAlpha(u) || isNumeric(u) }
    public static func isSpace(_ u: Unicode.Scalar) -> Bool {
        switch category(u) {
        case .spaceSeparator, .lineSeparator, .paragraphSeparator: return true
        default: break
        }
        return (0x09...0x0D).contains(u.value) || (0x1C...0x1F).contains(u.value) || u.value == 0x85
    }
    public static func isPrintable(_ u: Unicode.Scalar) -> Bool {
        if u.value == 0x20 { return true }
        switch category(u) {
        case .control, .format, .surrogate, .privateUse, .unassigned,
             .spaceSeparator, .lineSeparator, .paragraphSeparator: return false
        default: return true
        }
    }

    /// Unicode's simple (1:1) upper-case mapping; see the type comment for the exceptions.
    public static func simpleUpper(_ u: Unicode.Scalar) -> Unicode.Scalar {
        if u.value == 0xDF { return Unicode.Scalar(0x1E9E)! }                      // ß -> ẞ
        if (0x1F80...0x1F87).contains(u.value) || (0x1F90...0x1F97).contains(u.value)
            || (0x1FA0...0x1FA7).contains(u.value) { return Unicode.Scalar(u.value + 8)! }
        let m = u.properties.uppercaseMapping.unicodeScalars
        return m.count == 1 ? m.first! : u
    }
    /// Unicode's simple (1:1) lower-case mapping.
    public static func simpleLower(_ u: Unicode.Scalar) -> Unicode.Scalar {
        if u.value == 0x130 { return Unicode.Scalar(0x69)! }                       // İ -> i
        let m = u.properties.lowercaseMapping.unicodeScalars
        return m.count == 1 ? m.first! : u
    }
}

/// The character-class predicates in ``MetalStringArray/predicate(_:)``. Ops 0–2 are byte-wise and
/// always run on the GPU; ops 3–12 are the Unicode `utf8_is_*` family.
public enum StringPredicate: Int, Sendable, CaseIterable {
    case asciiIsPrintable = 0
    case asciiIsTitle = 1
    case stringIsAscii = 2
    case utf8IsAlnum = 3
    case utf8IsAlpha = 4
    case utf8IsDecimal = 5
    case utf8IsDigit = 6
    case utf8IsLower = 7
    case utf8IsNumeric = 8
    case utf8IsPrintable = 9
    case utf8IsSpace = 10
    case utf8IsTitle = 11
    case utf8IsUpper = 12

    /// True for the ten `utf8_is_*` predicates, which need the CPU for rows with a byte ≥ 0x80.
    var needsUnicode: Bool { rawValue >= StringPredicate.utf8IsAlnum.rawValue }

    /// The host implementation, also the fallback for a row the GPU could not decide.
    func evaluate(_ s: String) -> Bool {
        let scalars = s.unicodeScalars
        switch self {
        case .asciiIsPrintable:
            return s.utf8.allSatisfy { $0 >= 0x20 && $0 <= 0x7E }
        case .stringIsAscii:
            return s.utf8.allSatisfy { $0 < 0x80 }
        case .asciiIsTitle:
            return Self.title(s.utf8.map { $0 },
                              isUpper: { $0 >= 0x41 && $0 <= 0x5A },
                              isLower: { $0 >= 0x61 && $0 <= 0x7A })
        case .utf8IsTitle:
            return Self.title(Array(scalars), isUpper: UnicodeClass.isUpper, isLower: UnicodeClass.isLower)
        case .utf8IsPrintable:
            return scalars.allSatisfy(UnicodeClass.isPrintable)
        case .utf8IsUpper, .utf8IsLower:
            var anyCased = false
            for u in scalars {
                if self == .utf8IsUpper {
                    if UnicodeClass.isLower(u) { return false }
                    if UnicodeClass.isUpper(u) { anyCased = true }
                } else {
                    if UnicodeClass.isUpper(u) { return false }
                    if UnicodeClass.isLower(u) { anyCased = true }
                }
            }
            return anyCased
        case .utf8IsAlnum: return !s.isEmpty && scalars.allSatisfy(UnicodeClass.isAlnum)
        case .utf8IsAlpha: return !s.isEmpty && scalars.allSatisfy(UnicodeClass.isAlpha)
        case .utf8IsDecimal: return !s.isEmpty && scalars.allSatisfy(UnicodeClass.isDecimal)
        case .utf8IsDigit: return !s.isEmpty && scalars.allSatisfy(UnicodeClass.isDigit)
        case .utf8IsNumeric: return !s.isEmpty && scalars.allSatisfy(UnicodeClass.isNumeric)
        case .utf8IsSpace: return !s.isEmpty && scalars.allSatisfy(UnicodeClass.isSpace)
        }
    }

    /// Arrow's title rule: at least one cased character, every word opens with an upper-case (or
    /// titlecase) one, and every later character of a word is lower case. A non-cased character ends
    /// the word.
    static func title<C: Collection>(_ units: C, isUpper: (C.Element) -> Bool,
                                     isLower: (C.Element) -> Bool) -> Bool {
        var anyCased = false, previousCased = false
        for u in units {
            if isUpper(u) {
                if previousCased { return false }
                previousCased = true; anyCased = true
            } else if isLower(u) {
                if !previousCased { return false }
                previousCased = true; anyCased = true
            } else {
                previousCased = false
            }
        }
        return anyCased
    }
}

/// Op codes for the extra GPU transforms; must match the `SX_*` defines in `StringExtraSource`.
enum StringExtraTransform: Int, Sendable {
    case asciiTitle = 0
    case center = 1
    case replaceSliceCodeunits = 2
    case replaceSliceBytes = 3
}

/// The four Unicode normalisation forms Arrow's `utf8_normalize` accepts.
public enum UnicodeNormalizationForm: Int, Sendable, CaseIterable {
    case nfc = 0, nfkc = 1, nfd = 2, nfkd = 3

    public init?(name: String) {
        switch name.uppercased() {
        case "NFC": self = .nfc
        case "NFKC": self = .nfkc
        case "NFD": self = .nfd
        case "NFKD": self = .nfkd
        default: return nil
        }
    }

    func apply(_ s: String) -> String {
        switch self {
        case .nfc: return s.precomposedStringWithCanonicalMapping
        case .nfkc: return s.precomposedStringWithCompatibilityMapping
        case .nfd: return s.decomposedStringWithCanonicalMapping
        case .nfkd: return s.decomposedStringWithCompatibilityMapping
        }
    }
}

extension MetalStringArray {

    // MARK: - Plumbing

    private func sxPipeline(_ fn: String) throws -> MTLComputePipelineState {
        try context.pipeline(source: StringExtraSource.source, function: fn, cacheKey: "strx/\(fn)")
    }

    /// A device buffer holding `bytes` (never zero sized, so an empty argument still binds).
    func sxArgBuffer(_ bytes: [UInt8]) throws -> MetalArrowBuffer {
        let buf = try MetalArrowBuffer.allocate(byteCount: Swift.max(bytes.count, 1), zeroed: false, context: context)
        if !bytes.isEmpty { bytes.withUnsafeBytes { memcpy(buf.mutableContents, $0.baseAddress!, $0.count) } }
        return buf
    }

    /// 24 bytes matching MSL `struct SxParams { uint op; uint n1; uint n2; int p1; int p2; uint flags; }`.
    static func sxParams(_ op: Int, _ n1: Int, _ n2: Int, _ p1: Int, _ p2: Int, _ flags: Int) -> [UInt32] {
        [UInt32(op), UInt32(n1), UInt32(n2),
         UInt32(bitPattern: Int32(clamping: p1)), UInt32(bitPattern: Int32(clamping: p2)), UInt32(flags)]
    }

    /// True when every byte the array references is < 0x80, so the byte kernels and the Unicode
    /// tables cannot disagree. Reads the shared data buffer on the host, sharded over 64 KiB chunks.
    public func isAllASCII() throws -> Bool {
        try context.syncPoint()
        return withExtendedLifetime(self) { () -> Bool in
            let o = offsets.typed(Int32.self)
            let start = Int(o[0]), end = Int(o[length])
            guard end > start else { return true }
            let d = data.typed(UInt8.self)
            let chunk = 1 << 16
            let chunks = (end - start + chunk - 1) / chunk
            var high = [Bool](repeating: false, count: chunks)
            high.withUnsafeMutableBufferPointer { buf in
                func work(_ c: Int) {
                    let lo = start + c * chunk, hi = Swift.min(lo + chunk, end)
                    for p in lo..<hi where d[p] >= 0x80 { buf[c] = true; return }
                }
                if chunks == 1 { work(0) } else { DispatchQueue.concurrentPerform(iterations: chunks, execute: work) }
            }
            return !high.contains(true)
        }
    }

    /// The two-pass driver for the transforms in `StringExtraSource`, shaped exactly like
    /// `StringTransforms.transform(_:)`: length kernel, GPU scan into offsets, byte kernel.
    func extraTransform(_ op: StringExtraTransform, arg1: [UInt8] = [],
                        p1: Int = 0, p2: Int = 0) throws -> MetalStringArray {
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let lens = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 4, 4), zeroed: true, context: ctx)
        let a1 = try sxArgBuffer(arg1)
        let vb = validity ?? a1                                  // never read when flags bit 0 is clear
        let scratch = try MetalArrowBuffer.allocate(byteCount: 1, zeroed: false, context: ctx)
        var prm = Self.sxParams(op.rawValue, arg1.count, 0, p1, p2, validity == nil ? 0 : 1)

        if n > 0 {
            let pLen = try sxPipeline("sx_tf_len")
            try ctx.run { enc in
                enc.setComputePipelineState(pLen)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 2)
                Dispatch.setLength(enc, n, nil, index: 3)
                enc.setBytes(&prm, length: 24, index: 4)
                enc.setBuffer(a1.mtl, offset: a1.offset, index: 5)
                enc.setBuffer(lens.mtl, offset: lens.offset, index: 6)
                enc.setBuffer(scratch.mtl, offset: scratch.offset, index: 7)
                Dispatch.dispatch1D(enc, pLen, count: n)
            }
        }
        let outOffsets = try MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: lens, context: ctx)
            .exclusiveScanToOffsets()
        let total = Int(withExtendedLifetime(outOffsets) { outOffsets.typed(Int32.self)[n] })
        let outData = try MetalArrowBuffer.allocate(byteCount: total, zeroed: false, context: ctx)
        if n > 0 {
            let pWrite = try sxPipeline("sx_tf_write")
            try ctx.run { enc in
                enc.setComputePipelineState(pWrite)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 2)
                Dispatch.setLength(enc, n, nil, index: 3)
                enc.setBytes(&prm, length: 24, index: 4)
                enc.setBuffer(a1.mtl, offset: a1.offset, index: 5)
                enc.setBuffer(outOffsets.mtl, offset: outOffsets.offset, index: 6)
                enc.setBuffer(outData.mtl, offset: outData.offset, index: 7)
                Dispatch.dispatch1D(enc, pWrite, count: n)
            }
        }
        for b in [a1, scratch, lens] { ctx.retainUntilFlush(b) }
        let out = MetalStringArray(length: n, nullCount: nullCount, validity: validity,
                                   offsets: outOffsets, data: outData, context: ctx)
        out.isBinary = isBinary
        return out
    }

    /// Builds a `utf8` array by mapping every non-null row on the host, sharded over 4096-row chunks.
    func mapRowsConcurrently(_ f: @escaping (String) -> String) throws -> MetalStringArray {
        let n = length
        var rows = [String?](repeating: nil, count: n)
        rows.withUnsafeMutableBufferPointer { buf in
            forEachRowConcurrently { i, s in buf[i] = s.map(f) }
        }
        let out = try MetalStringArray(rows, context: context)
        out.isBinary = isBinary
        return out
    }

    // MARK: - Predicates

    /// One of Arrow's character-class predicates as a packed boolean bitmap. Nulls propagate.
    ///
    /// Ops 0–2 are byte-wise and entirely GPU. The ten `utf8_is_*` ops run the same GPU pass, which
    /// also reports which rows hold a byte ≥ 0x80; only those rows are re-decided on the CPU with
    /// ``UnicodeClass``, so an all-ASCII column never leaves the device.
    public func predicate(_ p: StringPredicate) throws -> MetalBooleanArray {
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let words = BitmapOps.words(bits: n)
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 1),
                                                zeroed: true, context: ctx)
        let nonAscii = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 1),
                                                     zeroed: true, context: ctx)
        if n > 0 {
            let pso = try sxPipeline("sx_pred")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                Dispatch.setLength(enc, n, nil, index: 2)
                Dispatch.setUInt(enc, p.rawValue, index: 3)
                enc.setBuffer(out.mtl, offset: out.offset, index: 4)
                enc.setBuffer(nonAscii.mtl, offset: nonAscii.offset, index: 5)
                Dispatch.dispatch1D(enc, pso, count: words)
            }
            if p.needsUnicode { try fixUpNonASCII(nonAscii, into: out, words: words, p.evaluate) }
        }
        ctx.retainUntilFlush(nonAscii)
        return MetalBooleanArray(length: n, nullCount: nullCount, validity: validity, values: out, context: ctx)
    }

    /// Re-decides, on the host, only the rows the GPU marked as holding a byte ≥ 0x80.
    ///
    /// A chunk is 128 bitmap words = 4096 rows = 512 result bytes, so two workers never touch the
    /// same byte of `out`.
    private func fixUpNonASCII(_ nonAscii: MetalArrowBuffer, into out: MetalArrowBuffer,
                               words: Int, _ evaluate: @escaping (String) -> Bool) throws {
        try context.syncPoint()
        try withExtendedLifetime(self) {
            let na = nonAscii.typed(UInt32.self)
            var any = false
            for w in 0..<words where na[w] != 0 { any = true; break }
            guard any else { return }
            let bits = out.mutableTyped(UInt8.self)
            let o = offsets.typed(Int32.self), d = data.typed(UInt8.self)
            let n = length
            let wordsPerChunk = 128
            let chunks = (words + wordsPerChunk - 1) / wordsPerChunk
            func work(_ c: Int) {
                let lo = c * wordsPerChunk, hi = Swift.min(lo + wordsPerChunk, words)
                for w in lo..<hi {
                    var m = na[w]
                    while m != 0 {
                        let i = w * 32 + m.trailingZeroBitCount
                        m &= m &- 1
                        guard i < n else { break }
                        let s = String(decoding: UnsafeBufferPointer(start: d + Int(o[i]),
                                                                     count: Int(o[i + 1] - o[i])), as: UTF8.self)
                        if evaluate(s) { Bitmap.set(bits, i) } else { Bitmap.clear(bits, i) }
                    }
                }
            }
            if chunks == 1 { work(0) } else { DispatchQueue.concurrentPerform(iterations: chunks, execute: work) }
        }
    }

    /// Arrow `ascii_is_printable`: every byte is in 0x20–0x7E. The **empty string is true**, unlike
    /// the other `ascii_is_*` predicates.
    public func asciiIsPrintable() throws -> MetalBooleanArray { try predicate(.asciiIsPrintable) }
    /// Arrow `ascii_is_title`: byte-wise title case over runs of ASCII letters, at least one letter.
    public func asciiIsTitle() throws -> MetalBooleanArray { try predicate(.asciiIsTitle) }
    /// Arrow `string_is_ascii`: every byte is < 0x80. The empty string is true.
    public func stringIsAscii() throws -> MetalBooleanArray { try predicate(.stringIsAscii) }

    /// Arrow `utf8_is_alnum`: non-empty and every code point is a letter or a number.
    public func utf8IsAlnum() throws -> MetalBooleanArray { try predicate(.utf8IsAlnum) }
    /// Arrow `utf8_is_alpha`: non-empty and every code point is in an `L*` category.
    public func utf8IsAlpha() throws -> MetalBooleanArray { try predicate(.utf8IsAlpha) }
    /// Arrow `utf8_is_decimal`: non-empty and every code point is category `Nd`.
    public func utf8IsDecimal() throws -> MetalBooleanArray { try predicate(.utf8IsDecimal) }
    /// Arrow `utf8_is_digit`: non-empty and every code point is category `Nd` or `No`.
    public func utf8IsDigit() throws -> MetalBooleanArray { try predicate(.utf8IsDigit) }
    /// Arrow `utf8_is_lower`: at least one cased code point and no upper-case one.
    public func utf8IsLower() throws -> MetalBooleanArray { try predicate(.utf8IsLower) }
    /// Arrow `utf8_is_numeric`: non-empty and every code point is category `Nd`, `Nl` or `No`.
    public func utf8IsNumeric() throws -> MetalBooleanArray { try predicate(.utf8IsNumeric) }
    /// Arrow `utf8_is_printable`. The empty string is true.
    public func utf8IsPrintable() throws -> MetalBooleanArray { try predicate(.utf8IsPrintable) }
    /// Arrow `utf8_is_space`: non-empty and every code point is Unicode whitespace.
    public func utf8IsSpace() throws -> MetalBooleanArray { try predicate(.utf8IsSpace) }
    /// Arrow `utf8_is_title`: at least one cased code point, in title case.
    public func utf8IsTitle() throws -> MetalBooleanArray { try predicate(.utf8IsTitle) }
    /// Arrow `utf8_is_upper`: at least one cased code point and no lower-case one.
    public func utf8IsUpper() throws -> MetalBooleanArray { try predicate(.utf8IsUpper) }

    // MARK: - Case transforms

    /// Arrow `ascii_title`: byte-wise, the first ASCII letter of every run of ASCII letters is
    /// upper-cased and the rest lower-cased. Bytes ≥ 0x80 are copied through and end a word, so
    /// `"ünïcödé"` becomes `"üNïCöDé"` — exactly what Arrow does. Always GPU.
    public func asciiTitle() throws -> MetalStringArray { try extraTransform(.asciiTitle) }

    /// Arrow `utf8_capitalize`: the first code point upper-cased, every later one lower-cased.
    ///
    /// Split **per row** (`Kernels/StringUnicode.swift`): a row entirely inside the Latin blocks
    /// (U+0000–U+017F) is mapped by the GPU table, any other row on the host.
    public func utf8Capitalize() throws -> MetalStringArray { try unicodeCapitalize() }

    /// Arrow `utf8_title`: the first **cased** code point of every word is upper-cased and the rest
    /// lower-cased, where a word is a maximal run of cased code points. Same per-row GPU/host split as
    /// ``utf8Capitalize()``.
    public func utf8Title() throws -> MetalStringArray { try unicodeTitle() }

    // MARK: - Centering

    /// Arrow `utf8_center`: pads with `pad` on both sides until the string is `width` **code points**
    /// wide, putting the odd character on the **right** (`"a"` centred in 4 is `"*a**"`). Strings
    /// already `width` or wider come back unchanged. Always GPU.
    ///
    /// `ascii_center` is the same function on ASCII input. It is not exposed separately, for the same
    /// reason `ascii_lpad` is not: Arrow's ASCII form counts bytes, and counting code points is the
    /// answer callers want for a `utf8` column.
    public func center(width: Int, pad: String = " ") throws -> MetalStringArray {
        try extraTransform(.center, arg1: try Self.padBytes(pad), p1: Swift.max(width, 0))
    }

    // MARK: - Slice replacement

    /// Arrow `utf8_replace_slice`: replaces the code points in `start ..< stop` with `replacement`.
    ///
    /// Negative indices count from the end, both ends clamp into range, and a `stop` below `start`
    /// inserts without deleting anything. Always GPU; the cut always lands on a code point boundary.
    public func replaceSlice(start: Int, stop: Int, with replacement: String) throws -> MetalStringArray {
        try extraTransform(.replaceSliceCodeunits, arg1: Array(replacement.utf8), p1: start, p2: stop)
    }

    /// Arrow `binary_replace_slice`: the same substitution indexed in **bytes**, which can split a
    /// UTF-8 sequence and is why the result is `binary` rather than `utf8`. Always GPU.
    public func replaceSliceBytes(start: Int, stop: Int, with replacement: [UInt8]) throws -> MetalStringArray {
        let out = try extraTransform(.replaceSliceBytes, arg1: replacement, p1: start, p2: stop)
        out.isBinary = true
        return out
    }

    // MARK: - Trimming

    /// The ASCII members of Arrow's Unicode whitespace set: `\t \n \v \f \r`, the four information
    /// separators U+001C–U+001F, and the space.
    static let unicodeWhitespaceASCII: [UInt8] = [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x1C, 0x1D, 0x1E, 0x1F, 0x20]

    /// Arrow `utf8_trim`: strips leading and trailing code points that appear in `characters`.
    ///
    /// An all-ASCII set never leaves the **GPU** at all — byte-wise trimming cannot split a UTF-8
    /// sequence, since every continuation byte is ≥ 0x80. A set with a non-ASCII character splits
    /// **per row**: the GPU trims the rows whose bytes are all < 0x80 with the set's ASCII part (the
    /// only part such a row could match), the host trims the rest. An empty set is the identity.
    public func utf8Trim(characters: String) throws -> MetalStringArray {
        try unicodeTrimRows(.trim, set: Set(characters.unicodeScalars))
    }
    /// Leading-only form of ``utf8Trim(characters:)``.
    public func utf8Ltrim(characters: String) throws -> MetalStringArray {
        try unicodeTrimRows(.ltrim, set: Set(characters.unicodeScalars))
    }
    /// Trailing-only form of ``utf8Trim(characters:)``.
    public func utf8Rtrim(characters: String) throws -> MetalStringArray {
        try unicodeTrimRows(.rtrim, set: Set(characters.unicodeScalars))
    }

    /// Arrow `utf8_trim_whitespace`: strips leading and trailing **Unicode** whitespace (see
    /// ``UnicodeClass/isSpace(_:)``), split **per row**: the GPU takes the rows whose bytes are all
    /// < 0x80, where that class is the ten bytes of ``unicodeWhitespaceASCII``, and the host takes the
    /// rest.
    public func utf8TrimWhitespace() throws -> MetalStringArray { try unicodeTrimWhitespaceRows(.trim) }
    /// Leading-only form of ``utf8TrimWhitespace()``.
    public func utf8LtrimWhitespace() throws -> MetalStringArray { try unicodeTrimWhitespaceRows(.ltrim) }
    /// Trailing-only form of ``utf8TrimWhitespace()``.
    public func utf8RtrimWhitespace() throws -> MetalStringArray { try unicodeTrimWhitespaceRows(.rtrim) }

    // MARK: - Normalisation

    /// Arrow `utf8_normalize`: NFC, NFKC, NFD or NFKD, through Foundation's normalisation. Always
    /// CPU — a full Unicode normalisation table in MSL buys nothing over the host implementation.
    ///
    /// **Difference from pyarrow:** `pyarrow.compute.utf8_normalize` (checked against 25.0.1) never
    /// composes, so its `NFC` output equals its `NFD` output and its `NFKC` equals its `NFKD`
    /// (`"é"` comes back as `U+0065 U+0301`). This implementation follows the Unicode standard, and
    /// agrees with Python's `unicodedata.normalize` on all four forms.
    public func utf8Normalize(_ form: UnicodeNormalizationForm) throws -> MetalStringArray {
        try mapRowsConcurrently { form.apply($0) }
    }

    // MARK: - Regular expression spans

    /// One capture group's span: the **byte** offset of the group inside the value and its **byte**
    /// length, both null where the group did not take part in the match.
    public typealias RegexSpan = (start: MetalArray<Int32>, length: MetalArray<Int32>)

    /// Arrow `extract_regex_span`, as one `(start, length)` pair of int32 arrays per **named** capture
    /// group — the shape ``extractRegex(_:ignoreCase:)`` uses, since ArrowMetal has no struct column.
    ///
    /// Offsets and lengths count **bytes**, as Arrow's do. A row that does not match, a row that is
    /// null, and a group that took part in no alternative are all null in both arrays. Always CPU
    /// (`NSRegularExpression`, sharded over 4096-row chunks).
    public func extractRegexSpan(_ pattern: String, ignoreCase: Bool = false) throws -> [String: RegexSpan] {
        let names = Self.namedGroups(in: pattern)
        guard !names.isEmpty else {
            throw ArrowMetalError.invalidArrowArray("extractRegexSpan needs at least one named group, e.g. (?<year>\\d+)")
        }
        let re = try Self.compileRegex(pattern, ignoreCase: ignoreCase)
        let n = length
        let mask = try regexPrefilter(pattern, ignoreCase: ignoreCase)
        let candidates = mask?.bitsPointer
        defer { withExtendedLifetime(mask) {} }
        var result: [String: RegexSpan] = [:]
        for name in names {
            var starts = [Int32?](repeating: nil, count: n)
            var lengths = [Int32?](repeating: nil, count: n)
            starts.withUnsafeMutableBufferPointer { sb in
                lengths.withUnsafeMutableBufferPointer { lb in
                    forEachRowConcurrently { i, s in
                        if let candidates, !Bitmap.isSet(candidates, i) { return }
                        guard let s, let m = re.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)) else { return }
                        let r = m.range(withName: name)
                        guard r.location != NSNotFound, let rr = Range(r, in: s) else { return }
                        let u = s.utf8
                        let start = u.distance(from: u.startIndex, to: rr.lowerBound)
                        let end = u.distance(from: u.startIndex, to: rr.upperBound)
                        sb[i] = Int32(start)
                        lb[i] = Int32(end - start)
                    }
                }
            }
            result[name] = (try MetalArray<Int32>(starts, context: context),
                            try MetalArray<Int32>(lengths, context: context))
        }
        return result
    }
}
