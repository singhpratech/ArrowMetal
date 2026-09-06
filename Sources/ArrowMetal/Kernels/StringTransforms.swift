import Foundation
import Metal

/// GPU string transforms producing new `MetalStringArray`s.
///
/// Output strings have data-dependent byte lengths, so each transform runs the two-pass pattern:
/// one kernel computes the output byte length of every row, `exclusiveScanToOffsets` turns those
/// lengths into the Arrow int32 offsets buffer on the GPU, and a second kernel writes the bytes.
/// Both passes call the same MSL routine (`tf_apply` in `StringTransformSource`), so a length and
/// the bytes that fill it can never disagree.
///
/// Nulls: the validity bitmap is shared with the input (zero copy), a null row produces zero
/// output bytes, and the null count is carried over unchanged. Binary operations AND the two
/// validity bitmaps, which is Arrow's `EMIT_NULL` null handling.
public enum StringTransform: Int, Sendable, CaseIterable {
    case asciiUpper = 0
    case asciiLower = 1
    case utf8Upper = 2
    case utf8Lower = 3
    case asciiSwapcase = 4
    case asciiCapitalize = 5
    case trimWhitespace = 6
    case ltrimWhitespace = 7
    case rtrimWhitespace = 8
    case trimCharacters = 9
    case ltrimCharacters = 10
    case rtrimCharacters = 11
    case replaceSubstring = 12
    case repeatCopies = 13
    case sliceCodeunits = 14
    case padLeft = 15
    case padRight = 16
    case reverse = 17
}

/// ASCII character classes for `MetalStringArray.classify(_:)`. Arrow/Python semantics: an empty
/// string is false everywhere; `upper`/`lower` need at least one cased character and no character
/// of the opposite case, and treat non-ASCII bytes as uncased.
public enum StringClass: Int, Sendable, CaseIterable {
    case alnum = 0, alpha = 1, digit = 2, space = 3, upper = 4, lower = 5
}

extension MetalStringArray {

    // MARK: - Plumbing

    private func tfPipeline(_ fn: String) throws -> MTLComputePipelineState {
        try context.pipeline(source: StringTransformSource.source, function: fn, cacheKey: "strtf/\(fn)")
    }

    /// A device buffer holding `bytes` (at least one page, so an empty argument still binds).
    private func argBuffer(_ bytes: [UInt8]) throws -> MetalArrowBuffer {
        let buf = try MetalArrowBuffer.allocate(byteCount: Swift.max(bytes.count, 1), zeroed: false, context: context)
        if !bytes.isEmpty { bytes.withUnsafeBytes { memcpy(buf.mutableContents, $0.baseAddress!, $0.count) } }
        return buf
    }

    /// 24 bytes matching MSL `struct TfParams { uint op; uint n1; uint n2; int p1; int p2; uint flags; }`.
    private static func params(_ op: Int, _ n1: Int, _ n2: Int, _ p1: Int, _ p2: Int, _ flags: Int) -> [UInt32] {
        [UInt32(op), UInt32(n1), UInt32(n2),
         UInt32(bitPattern: Int32(clamping: p1)), UInt32(bitPattern: Int32(clamping: p2)), UInt32(flags)]
    }

    /// The two-pass transform driver: length kernel, GPU scan to offsets, byte kernel.
    func transform(_ op: StringTransform, arg1: [UInt8] = [], arg2: [UInt8] = [],
                   p1: Int = 0, p2: Int = 0) throws -> MetalStringArray {
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let lens = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 4, 4), zeroed: true, context: ctx)
        let a1 = try argBuffer(arg1), a2 = try argBuffer(arg2)
        let vb = validity ?? a1                                  // never read when flags bit 0 is clear
        let scratch = try MetalArrowBuffer.allocate(byteCount: 1, zeroed: false, context: ctx)
        var prm = Self.params(op.rawValue, arg1.count, arg2.count, p1, p2, validity == nil ? 0 : 1)

        if n > 0 {
            let pLen = try tfPipeline("str_tf_len")
            try ctx.run { enc in
                enc.setComputePipelineState(pLen)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 2)
                Dispatch.setLength(enc, n, nil, index: 3)
                enc.setBytes(&prm, length: 24, index: 4)
                enc.setBuffer(a1.mtl, offset: a1.offset, index: 5)
                enc.setBuffer(a2.mtl, offset: a2.offset, index: 6)
                enc.setBuffer(lens.mtl, offset: lens.offset, index: 7)
                enc.setBuffer(scratch.mtl, offset: scratch.offset, index: 8)
                Dispatch.dispatch1D(enc, pLen, count: n)
            }
        }
        // Scan flushes any open batch and reopens it, so the total below is readable on the host.
        let outOffsets = try MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: lens, context: ctx)
            .exclusiveScanToOffsets()
        let total = Int(withExtendedLifetime(outOffsets) { outOffsets.typed(Int32.self)[n] })
        let outData = try MetalArrowBuffer.allocate(byteCount: total, zeroed: false, context: ctx)
        if n > 0 {
            let pWrite = try tfPipeline("str_tf_write")
            try ctx.run { enc in
                enc.setComputePipelineState(pWrite)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 2)
                Dispatch.setLength(enc, n, nil, index: 3)
                enc.setBytes(&prm, length: 24, index: 4)
                enc.setBuffer(a1.mtl, offset: a1.offset, index: 5)
                enc.setBuffer(a2.mtl, offset: a2.offset, index: 6)
                enc.setBuffer(outOffsets.mtl, offset: outOffsets.offset, index: 7)
                enc.setBuffer(outData.mtl, offset: outData.offset, index: 8)
                Dispatch.dispatch1D(enc, pWrite, count: n)
            }
        }
        for b in [a1, a2, scratch, lens] { ctx.retainUntilFlush(b) }
        ctx.retainUntilFlush(self)
        let out = MetalStringArray(length: n, nullCount: nullCount, validity: validity,
                                   offsets: outOffsets, data: outData, context: ctx)
        // A `binary` column stays `binary`: `binary_length`, `binary_repeat` and the byte-wise trims
        // are defined on it, and Arrow gives them a `binary` result.
        out.isBinary = isBinary
        return out
    }

    // MARK: - Case mapping

    /// Arrow `ascii_upper`: byte-wise `a`–`z` → `A`–`Z`, every other byte untouched.
    public func asciiUpper() throws -> MetalStringArray { try transform(.asciiUpper) }
    /// Arrow `ascii_lower`: byte-wise `A`–`Z` → `a`–`z`.
    public func asciiLower() throws -> MetalStringArray { try transform(.asciiLower) }
    /// Arrow `ascii_swapcase`: byte-wise ASCII case flip.
    public func asciiSwapcase() throws -> MetalStringArray { try transform(.asciiSwapcase) }
    /// Arrow `ascii_capitalize`: first byte upper-cased, the rest lower-cased (ASCII only).
    public func asciiCapitalize() throws -> MetalStringArray { try transform(.asciiCapitalize) }

    /// Arrow `utf8_upper`: Unicode's **simple** (1:1 code point) upper-case mapping, over every script.
    ///
    /// Split per row (`Kernels/StringUnicode.swift`): a row whose code points are all at or below
    /// U+017F — Basic Latin, the Latin-1 Supplement and Latin Extended-A, `"ß"` → `"ẞ"` and `"µ"` →
    /// `"Μ"` included — is mapped by the GPU table; any row above that block is mapped on the host
    /// with Swift's Unicode tables, sharded over 4096-row chunks. Both paths agree with pyarrow.
    ///
    /// Simple, not full: `"ﬁ"` and `"ŉ"` pass through, because their full upper-case mappings are two
    /// characters long and utf8proc — which is what Arrow uses — leaves them alone too.
    public func utf8Upper() throws -> MetalStringArray { try unicodeUpper() }

    /// Arrow `utf8_lower`: Unicode's simple lower-case mapping, with the same per-row GPU/host split
    /// as ``utf8Upper()``.
    ///
    /// `"İ"` lower-cases to `"i"` alone, and `"Σ"` to `"σ"` in every position — the contextual
    /// final-sigma rule is not part of the simple mapping and Arrow does not apply it either.
    public func utf8Lower() throws -> MetalStringArray { try unicodeLower() }

    // MARK: - Trimming

    /// Arrow `ascii_trim_whitespace`: strips leading and trailing ASCII whitespace
    /// (space, `\t`, `\n`, `\v`, `\f`, `\r`). Bytes ≥ 0x80 are never trimmed, so UTF-8 sequences
    /// stay intact.
    public func trim() throws -> MetalStringArray { try transform(.trimWhitespace) }
    /// Leading ASCII whitespace only (Arrow `ascii_ltrim_whitespace`).
    public func ltrim() throws -> MetalStringArray { try transform(.ltrimWhitespace) }
    /// Trailing ASCII whitespace only (Arrow `ascii_rtrim_whitespace`).
    public func rtrim() throws -> MetalStringArray { try transform(.rtrimWhitespace) }

    /// Arrow `ascii_trim`: strips leading and trailing bytes that appear in `characters`.
    /// The set is matched byte-wise, so only ASCII characters can be trimmed; an empty set is a
    /// no-op. Throws when `characters` contains a non-ASCII character.
    public func trim(characters: String) throws -> MetalStringArray {
        try transform(.trimCharacters, arg1: try Self.asciiSet(characters))
    }
    /// Leading-only form of ``trim(characters:)``.
    public func ltrim(characters: String) throws -> MetalStringArray {
        try transform(.ltrimCharacters, arg1: try Self.asciiSet(characters))
    }
    /// Trailing-only form of ``trim(characters:)``.
    public func rtrim(characters: String) throws -> MetalStringArray {
        try transform(.rtrimCharacters, arg1: try Self.asciiSet(characters))
    }

    static func asciiSet(_ s: String) throws -> [UInt8] {
        let b = Array(s.utf8)
        guard b.allSatisfy({ $0 < 0x80 }) else {
            throw ArrowMetalError.unsupportedType("trim character sets are ASCII only, got \(s)")
        }
        return b
    }

    // MARK: - Substring and shape

    /// Arrow `replace_substring`: replaces non-overlapping occurrences of `pattern`, scanning
    /// left to right. `maxReplacements` < 0 means every occurrence. An empty `pattern` is the
    /// identity, matching Foundation's `replacingOccurrences(of: "", with:)`.
    public func replaceSubstring(_ pattern: String, with replacement: String,
                                 maxReplacements: Int = -1) throws -> MetalStringArray {
        try transform(.replaceSubstring, arg1: Array(pattern.utf8), arg2: Array(replacement.utf8),
                      p1: maxReplacements < 0 ? -1 : maxReplacements)
    }

    /// Arrow `binary_repeat`: `n` copies of each string concatenated. `n == 0` gives empty strings.
    public func `repeat`(_ n: Int) throws -> MetalStringArray {
        guard n >= 0 else { throw ArrowMetalError.invalidArrowArray("repeat needs n >= 0, got \(n)") }
        return try transform(.repeatCopies, p1: n)
    }

    /// Arrow `utf8_slice_codeunits`: the substring `value[start:stop:step]`, counted in **code
    /// points**, with Python's slice rules.
    ///
    /// Negative indices count from the end of the string, both ends clamp into range, and a negative
    /// `step` walks backwards (so `step: -1` reverses). `stop == nil` means "to the end" going
    /// forwards and "to the beginning" going backwards. Slicing always lands on UTF-8 code point
    /// boundaries, so a multi-byte character is never split. Always GPU; `step == 1` takes the
    /// dedicated kernel in `StringTransformSource`, any other step the one in `StringBytesSource`.
    public func sliceCodeunits(start: Int, stop: Int? = nil, step: Int = 1) throws -> MetalStringArray {
        guard step != 0 else { throw ArrowMetalError.invalidArrowArray("utf8_slice_codeunits step cannot be zero") }
        let end = Self.sliceStop(stop, step: step)
        if step == 1 { return try transform(.sliceCodeunits, p1: start, p2: end) }
        return try byteTransform(.sliceCodepoints, p1: start, p2: end, p3: step)
    }

    /// Arrow `utf8_lpad`: left-pads with `pad` until the string is `width` code points wide.
    /// Strings already `width` or wider are unchanged. `pad` must be exactly one character.
    public func padLeft(width: Int, pad: String = " ") throws -> MetalStringArray {
        try transform(.padLeft, arg1: try Self.padBytes(pad), p1: Swift.max(width, 0))
    }
    /// Arrow `utf8_rpad`: right-pads with `pad` to `width` code points.
    public func padRight(width: Int, pad: String = " ") throws -> MetalStringArray {
        try transform(.padRight, arg1: try Self.padBytes(pad), p1: Swift.max(width, 0))
    }

    static func padBytes(_ s: String) throws -> [UInt8] {
        guard s.unicodeScalars.count == 1 else {
            throw ArrowMetalError.invalidArrowArray("padding must be exactly one character, got \(s.unicodeScalars.count)")
        }
        return Array(s.utf8)
    }

    /// Arrow `utf8_reverse`: reverses the code points of each string (not grapheme clusters, so a
    /// combining mark or a ZWJ emoji sequence comes back in reverse code point order).
    public func reverse() throws -> MetalStringArray { try transform(.reverse) }

    // MARK: - Binary

    /// Arrow `binary_join_element_wise`: `self[i] + separator + other[i]`. Validities are ANDed on
    /// the GPU, so a null on either side gives a null output (Arrow's `EMIT_NULL`).
    public func concat(_ other: MetalStringArray, separator: String = "") throws -> MetalStringArray {
        guard other.length == length else { throw ArrowMetalError.lengthMismatch(length, other.length) }
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let sepBytes = Array(separator.utf8)
        let sep = try argBuffer(sepBytes)
        let lens = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 4, 4), zeroed: true, context: ctx)
        let va = validity ?? sep, vb = other.validity ?? sep
        let flags = (validity == nil ? 0 : 1) | (other.validity == nil ? 0 : 2)
        var prm = Self.params(0, sepBytes.count, 0, 0, 0, flags)

        if n > 0 {
            let pLen = try tfPipeline("str_cat_len")
            try ctx.run { enc in
                enc.setComputePipelineState(pLen)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(other.offsets.mtl, offset: other.offsets.offset, index: 1)
                enc.setBuffer(va.mtl, offset: va.offset, index: 2)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 3)
                Dispatch.setLength(enc, n, nil, index: 4)
                enc.setBytes(&prm, length: 24, index: 5)
                enc.setBuffer(lens.mtl, offset: lens.offset, index: 6)
                Dispatch.dispatch1D(enc, pLen, count: n)
            }
        }
        let outOffsets = try MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: lens, context: ctx)
            .exclusiveScanToOffsets()
        let total = Int(withExtendedLifetime(outOffsets) { outOffsets.typed(Int32.self)[n] })
        let outData = try MetalArrowBuffer.allocate(byteCount: total, zeroed: false, context: ctx)
        if n > 0 {
            let pWrite = try tfPipeline("str_cat_write")
            try ctx.run { enc in
                enc.setComputePipelineState(pWrite)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                enc.setBuffer(other.offsets.mtl, offset: other.offsets.offset, index: 2)
                enc.setBuffer(other.data.mtl, offset: other.data.offset, index: 3)
                enc.setBuffer(va.mtl, offset: va.offset, index: 4)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 5)
                Dispatch.setLength(enc, n, nil, index: 6)
                enc.setBytes(&prm, length: 24, index: 7)
                enc.setBuffer(sep.mtl, offset: sep.offset, index: 8)
                enc.setBuffer(outOffsets.mtl, offset: outOffsets.offset, index: 9)
                enc.setBuffer(outData.mtl, offset: outData.offset, index: 10)
                Dispatch.dispatch1D(enc, pWrite, count: n)
            }
        }
        for b in [sep, lens] { ctx.retainUntilFlush(b) }
        let outValidity = try BitmapOps.combineValidity(ctx, validity, other.validity, bits: n)
        var nulls = 0
        if let v = outValidity {
            try ctx.syncPoint()
            nulls = n - Bitmap.popcount(v.typed(UInt8.self), bits: n)
        }
        return MetalStringArray(length: n, nullCount: nulls, validity: outValidity,
                                offsets: outOffsets, data: outData, context: ctx)
    }

    // MARK: - Search

    /// Arrow `count_substring`: non-overlapping occurrences of `pattern` in each string. An empty
    /// pattern counts the code point boundaries, that is `charLength() + 1`, matching Arrow.
    /// Nulls propagate.
    public func countSubstring(_ pattern: String) throws -> MetalArray<Int32> {
        try search(pattern, mode: 0)
    }

    /// Arrow `find_substring`: the **byte** offset of the first occurrence of `pattern` in each
    /// string, or -1 when it does not occur. An empty pattern finds 0. Nulls propagate.
    public func findSubstring(_ pattern: String) throws -> MetalArray<Int32> {
        try search(pattern, mode: 1)
    }

    private func search(_ pattern: String, mode: Int) throws -> MetalArray<Int32> {
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let pat = Array(pattern.utf8)
        let patBuf = try argBuffer(pat)
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 4, 4), zeroed: true, context: ctx)
        if n > 0 {
            let p = try tfPipeline("str_tf_search")
            try ctx.run { enc in
                enc.setComputePipelineState(p)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                Dispatch.setLength(enc, n, nil, index: 2)
                enc.setBuffer(patBuf.mtl, offset: patBuf.offset, index: 3)
                Dispatch.setUInt(enc, pat.count, index: 4)
                Dispatch.setUInt(enc, mode, index: 5)
                enc.setBuffer(out.mtl, offset: out.offset, index: 6)
                Dispatch.dispatch1D(enc, p, count: n)
            }
        }
        ctx.retainUntilFlush(patBuf)
        return MetalArray<Int32>(length: n, nullCount: nullCount, validity: validity, values: out, context: ctx)
    }

    // MARK: - Predicates

    /// One of Arrow's `ascii_is_*` predicates as a packed boolean bitmap. Nulls propagate.
    public func classify(_ cls: StringClass) throws -> MetalBooleanArray {
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let out = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: n), zeroed: true, context: ctx)
        if n > 0 {
            let p = try tfPipeline("str_tf_class")
            try ctx.run { enc in
                enc.setComputePipelineState(p)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                Dispatch.setLength(enc, n, nil, index: 2)
                Dispatch.setUInt(enc, cls.rawValue, index: 3)
                enc.setBuffer(out.mtl, offset: out.offset, index: 4)
                Dispatch.dispatch1D(enc, p, count: BitmapOps.words(bits: n))
            }
        }
        return MetalBooleanArray(length: n, nullCount: nullCount, validity: validity, values: out, context: ctx)
    }

    /// Arrow `ascii_is_alnum`: non-empty and every byte is an ASCII letter or digit.
    public func asciiIsAlnum() throws -> MetalBooleanArray { try classify(.alnum) }
    /// Arrow `ascii_is_alpha`: non-empty and every byte is an ASCII letter.
    public func asciiIsAlpha() throws -> MetalBooleanArray { try classify(.alpha) }
    /// Arrow `ascii_is_decimal`: non-empty and every byte is an ASCII digit.
    public func asciiIsDigit() throws -> MetalBooleanArray { try classify(.digit) }
    /// Arrow `ascii_is_space`: non-empty and every byte is ASCII whitespace.
    public func asciiIsSpace() throws -> MetalBooleanArray { try classify(.space) }
    /// Arrow `ascii_is_upper`: at least one uppercase ASCII letter and no lowercase one.
    public func asciiIsUpper() throws -> MetalBooleanArray { try classify(.upper) }
    /// Arrow `ascii_is_lower`: at least one lowercase ASCII letter and no uppercase one.
    public func asciiIsLower() throws -> MetalBooleanArray { try classify(.lower) }
}
