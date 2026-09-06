import Foundation
import Metal

/// Arrow's `utf8_swapcase` and `utf8_zero_fill`, plus the Arrow-named aliases for the padding
/// transforms that already ship.
///
/// Both new transforms run the two-pass GPU pattern the other string transforms use: one kernel
/// computes every output row's byte length, a GPU scan turns those lengths into the Arrow int32
/// offsets buffer, and a second kernel writes the bytes. Nulls share the input's validity bitmap
/// with no copy and produce zero output bytes.
extension MetalStringArray {

    /// Ops understood by `StringAliasesSource`. The numbering is local to that file.
    enum ExtraTransform: Int {
        case utf8Swapcase = 0
        case utf8ZeroFill = 1
    }

    /// The two-pass driver for the transforms in `StringAliasesSource`.
    func extraTransform(_ op: ExtraTransform, arg1: [UInt8] = [], p1: Int = 0) throws -> MetalStringArray {
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let lens = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 4, 4), zeroed: true, context: ctx)
        let a1 = try MetalArrowBuffer.allocate(byteCount: Swift.max(arg1.count, 1), zeroed: false, context: ctx)
        if !arg1.isEmpty { arg1.withUnsafeBytes { memcpy(a1.mutableContents, $0.baseAddress!, $0.count) } }
        let scratch = try MetalArrowBuffer.allocate(byteCount: 1, zeroed: false, context: ctx)
        let vb = validity ?? a1                              // never read when flags bit 0 is clear
        var prm: [UInt32] = [UInt32(op.rawValue), UInt32(arg1.count),
                             UInt32(bitPattern: Int32(clamping: p1)), validity == nil ? 0 : 1]
        func pso(_ f: String) throws -> MTLComputePipelineState {
            try ctx.pipeline(source: StringAliasesSource.source, function: f, cacheKey: "strx/\(f)")
        }
        if n > 0 {
            let p = try pso("sx_len")
            try ctx.run { enc in
                enc.setComputePipelineState(p)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 2)
                Dispatch.setLength(enc, n, nil, index: 3)
                enc.setBytes(&prm, length: 16, index: 4)
                enc.setBuffer(a1.mtl, offset: a1.offset, index: 5)
                enc.setBuffer(lens.mtl, offset: lens.offset, index: 6)
                enc.setBuffer(scratch.mtl, offset: scratch.offset, index: 7)
                Dispatch.dispatch1D(enc, p, count: n)
            }
        }
        // The scan flushes any open batch and reopens it, so the total below is readable on the host.
        let outOffsets = try MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: lens, context: ctx)
            .exclusiveScanToOffsets()
        let total = Int(withExtendedLifetime(outOffsets) { outOffsets.typed(Int32.self)[n] })
        let outData = try MetalArrowBuffer.allocate(byteCount: Swift.max(total, 1), zeroed: false, context: ctx)
        if n > 0 {
            let p = try pso("sx_write")
            try ctx.run { enc in
                enc.setComputePipelineState(p)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 2)
                Dispatch.setLength(enc, n, nil, index: 3)
                enc.setBytes(&prm, length: 16, index: 4)
                enc.setBuffer(a1.mtl, offset: a1.offset, index: 5)
                enc.setBuffer(outOffsets.mtl, offset: outOffsets.offset, index: 6)
                enc.setBuffer(outData.mtl, offset: outData.offset, index: 7)
                Dispatch.dispatch1D(enc, p, count: n)
            }
        }
        for b in [a1, scratch, lens] { ctx.retainUntilFlush(b) }
        ctx.retainUntilFlush(self)
        return MetalStringArray(length: n, nullCount: nullCount, validity: validity,
                                offsets: outOffsets, data: outData, context: ctx)
    }

    /// Arrow `utf8_swapcase`: upper-case characters become lower case and vice versa, one code point
    /// at a time, over every script.
    ///
    /// Split per row (`Kernels/StringUnicode.swift`): a row whose code points are all at or below
    /// U+017F is swapped by the GPU table — `"ß"` → `"ẞ"` and `"µ"` → `"Μ"` included — and any row
    /// above that block on the host. A **titlecase** letter is both upper and lower case for Arrow and
    /// so stays put: `"ǅ"` swaps to `"ǅ"`, while `"Ǆ"` swaps to `"ǆ"`.
    public func utf8Swapcase() throws -> MetalStringArray { try unicodeSwapcase() }

    /// Arrow `utf8_zero_fill`: left-pads each string to `width` **code points** with `padding`,
    /// inserting the padding *after* a leading `+` or `-` so a signed number keeps its sign in front.
    ///
    /// Strings already at or over `width` are returned unchanged, an empty string becomes `width`
    /// padding characters, and the content is not required to be numeric (`"abc"` at width 5 becomes
    /// `"00abc"`, as Arrow does). `padding` must be exactly one character.
    public func utf8ZeroFill(width: Int, padding: String = "0") throws -> MetalStringArray {
        guard padding.unicodeScalars.count == 1 else {
            throw ArrowMetalError.invalidArrowArray("utf8_zero_fill padding must be exactly one character, got \(padding.debugDescription)")
        }
        return try extraTransform(.utf8ZeroFill, arg1: Array(padding.utf8), p1: Swift.max(0, width))
    }

    // MARK: - Arrow-named aliases for transforms that already ship

    /// Arrow `ascii_lpad` / `utf8_lpad`: the existing `padLeft(width:pad:)`. `width` counts code
    /// points and `pad` must be one character; a string already at or over `width` is unchanged.
    public func lpad(width: Int, pad: String = " ") throws -> MetalStringArray {
        try padLeft(width: width, pad: pad)
    }
    /// Arrow `ascii_rpad` / `utf8_rpad`: the existing `padRight(width:pad:)`.
    public func rpad(width: Int, pad: String = " ") throws -> MetalStringArray {
        try padRight(width: width, pad: pad)
    }
}
