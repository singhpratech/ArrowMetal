import Foundation
import Metal

/// Arrow's four splitting functions as a real `list<utf8>` column.
///
/// `Kernels/Regex.swift` produced the `(offsets, values)` **pair** of a list array because there was
/// no list type to put it in; `Sources/ArrowMetal/Nested.swift` now has one, so the list is the
/// primary result everywhere — Swift, the C ABI and Python — and the pair stays as a convenience for
/// callers that only want the flat pieces.
///
/// ## Where the work runs
///
/// | function | where | why |
/// |---|---|---|
/// | `ascii_split_whitespace` | **GPU** | a separator is a run of bytes in `\\t`–`\\r` and the space |
/// | `split_pattern` | **GPU** | a separator is a literal byte string |
/// | `utf8_split_whitespace` | **GPU** | the Unicode whitespace class is small enough to spell out in MSL |
/// | `split_pattern_regex` | CPU, sharded | the regex engine is a host backtracker |
///
/// ## Semantics
///
/// Every separator makes a boundary, so leading and trailing separators produce **empty end pieces**
/// and the empty string splits to one empty piece — Arrow's behaviour, and Python's `str.split(sep)`
/// rather than its no-argument form. `maxSplits < 0` means every separator; otherwise the first
/// `maxSplits` are used, or the last `maxSplits` when `reverse` is set, and whatever is left over
/// stays inside the final piece. A null input row is a null list row owning no pieces.
extension MetalStringArray {

    /// Which separator the byte-wise splitter looks for.
    enum SplitKind: Int { case whitespace = 0, literal = 1, unicodeWhitespace = 2 }

    private func spPipeline(_ fn: String) throws -> MTLComputePipelineState {
        try context.pipeline(source: StringSplitSource.source, function: fn, cacheKey: "strs/\(fn)")
    }

    /// The three-pass GPU splitter: count pieces, scan, measure pieces, scan, copy bytes.
    func splitOnGPU(_ kind: SplitKind, pattern: [UInt8], maxSplits: Int, reverse: Bool) throws -> MetalListArray {
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let pat = try sxArgBuffer(pattern)
        let vb = validity ?? pat                                 // never read when flags bit 1 is clear
        // 16 bytes matching MSL `struct SpParams { uint op; uint n1; int maxSplits; uint flags; }`.
        var prm: [UInt32] = [UInt32(kind.rawValue), UInt32(pattern.count),
                             UInt32(bitPattern: Int32(clamping: maxSplits)),
                             UInt32((reverse ? 1 : 0) | (validity == nil ? 0 : 2))]

        let counts = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 4, 4), zeroed: true, context: ctx)
        if n > 0 {
            let p = try spPipeline("sp_count")
            try ctx.run { enc in
                enc.setComputePipelineState(p)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 2)
                Dispatch.setLength(enc, n, nil, index: 3)
                enc.setBytes(&prm, length: 16, index: 4)
                enc.setBuffer(pat.mtl, offset: pat.offset, index: 5)
                enc.setBuffer(counts.mtl, offset: counts.offset, index: 6)
                Dispatch.dispatch1D(enc, p, count: n)
            }
        }
        let listOffsets = try MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: counts, context: ctx)
            .exclusiveScanToOffsets()
        let pieces = Int(withExtendedLifetime(listOffsets) { listOffsets.typed(Int32.self)[n] })

        let childLens = try MetalArrowBuffer.allocate(byteCount: Swift.max(pieces * 4, 4), zeroed: true, context: ctx)
        if n > 0 {
            let p = try spPipeline("sp_lens")
            try ctx.run { enc in
                enc.setComputePipelineState(p)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 2)
                Dispatch.setLength(enc, n, nil, index: 3)
                enc.setBytes(&prm, length: 16, index: 4)
                enc.setBuffer(pat.mtl, offset: pat.offset, index: 5)
                enc.setBuffer(listOffsets.mtl, offset: listOffsets.offset, index: 6)
                enc.setBuffer(childLens.mtl, offset: childLens.offset, index: 7)
                Dispatch.dispatch1D(enc, p, count: n)
            }
        }
        let valueOffsets = try MetalArray<Int32>(length: pieces, nullCount: 0, validity: nil,
                                                 values: childLens, context: ctx).exclusiveScanToOffsets()
        let bytes = Int(withExtendedLifetime(valueOffsets) { valueOffsets.typed(Int32.self)[pieces] })
        let outData = try MetalArrowBuffer.allocate(byteCount: Swift.max(bytes, 1), zeroed: false, context: ctx)
        if n > 0 {
            let p = try spPipeline("sp_write")
            try ctx.run { enc in
                enc.setComputePipelineState(p)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 2)
                Dispatch.setLength(enc, n, nil, index: 3)
                enc.setBytes(&prm, length: 16, index: 4)
                enc.setBuffer(pat.mtl, offset: pat.offset, index: 5)
                enc.setBuffer(listOffsets.mtl, offset: listOffsets.offset, index: 6)
                enc.setBuffer(valueOffsets.mtl, offset: valueOffsets.offset, index: 7)
                enc.setBuffer(outData.mtl, offset: outData.offset, index: 8)
                Dispatch.dispatch1D(enc, p, count: n)
            }
        }
        for b in [pat, counts, childLens] { ctx.retainUntilFlush(b) }
        ctx.retainUntilFlush(self)
        try ctx.syncPoint()
        let values = MetalStringArray(length: pieces, nullCount: 0, validity: nil,
                                      offsets: valueOffsets, data: outData, context: ctx)
        return MetalListArray(length: n, nullCount: nullCount, validity: validity,
                              offsets: listOffsets, values: .string(values), context: ctx)
    }

    /// Wraps a host-built `(offsets, values)` pair as a `list<utf8>` sharing this array's validity.
    func listFromPair(_ pair: SplitResult) throws -> MetalListArray {
        MetalListArray(length: length, nullCount: nullCount, validity: validity,
                       offsets: pair.offsets.values, values: .string(pair.values), context: context)
    }
}

extension MetalListArray {
    /// The `(offsets, values)` pair behind a `list<utf8>`, for callers that only want the flat pieces.
    public func stringPair() throws -> MetalStringArray.SplitResult {
        guard case .string(let v) = values else {
            throw ArrowMetalError.unsupportedType("stringPair needs a list<utf8>, got \(values.arrowFormat)")
        }
        let o = MetalArray<Int32>(length: length + 1, nullCount: 0, validity: nil,
                                  values: offsets, context: context)
        return (o, v)
    }
}
