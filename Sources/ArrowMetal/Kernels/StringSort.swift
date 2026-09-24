import Foundation
import Metal

// GPU sort for Arrow `utf8` and `binary` columns.
//
// The order is byte-wise lexicographic, exactly what pyarrow's `array_sort_indices` and Polars' `arg_sort`
// produce for utf8: Arrow compares string bytes, not Unicode collation elements, so this is the whole
// contract. Nulls form one block, in row order, at the end `nullPlacement` names (last by default), and
// equal rows keep their original order (stable), in both directions.
//
// How it runs. A string has no fixed-width order-preserving key, so the sort is an LSD radix over
// fixed-width *prefix chunks*: `StringSortSource` packs seven of a row's bytes into one 63-bit integer
// whose numeric order is those bytes' lexicographic order, and the column is sorted once per chunk, from
// the last chunk to the first, with the existing stable 64-bit radix argsort (`Kernels/Sort.swift`) doing
// each pass. Because every pass is stable, sorting by the least significant chunk first and the most
// significant chunk last leaves the rows in full lexicographic order — the same argument that makes a
// digit-by-digit radix sort work, with a 7-byte "digit".
//
// The number of passes is `ceil(longest row / 7)`, measured on the GPU before the first pass, so a column
// of short keys (customer ids, country codes, ISO dates) costs one or two radix sorts and never touches
// the bytes past the longest row. A column with one very long row pays a pass for it; that is the honest
// bound for a radix sort over a variable-width key, and it is still linear in the rows per pass.
//
// The alternative — sort by the first chunk, then re-sort only the tie runs — was rejected: the segmented
// pass needs a second sort per round to restore the run order, so it only wins once rows share more than
// about fourteen leading bytes, and it costs a run-mark scan on every round.

extension MetalStringArray {
    /// Arrow `array_sort_indices` for utf8 / binary: int32 indices putting the rows in byte-wise
    /// lexicographic order, stable, with the null rows at whichever end `nullPlacement` names (past the
    /// values in both directions, as Arrow specifies).
    public func argsort(descending: Bool = false,
                        nullPlacement: NullPlacement = .atEnd) throws -> MetalArray<Int32> {
        let n = length
        let ctx = context
        try Dispatch.checkLength(n)
        if n == 0 { return try MetalArray<Int32>([Int32](), context: ctx) }

        // The null rows ride in bit 63 of every prefix key (see `StringSortSource`), so the stable passes
        // leave them as one block, in row order, at `nullPlacement`'s end: no separate partition and no
        // CPU readback of the index array inside an open batch. A column whose valid rows are all empty
        // strings still takes one pass, to move its nulls.
        let nulls: (MetalArrowBuffer, NullPlacement)? = nullCount > 0 ? validity.map { ($0, nullPlacement) } : nil
        var chunks = try chunkCount()
        if nulls != nil && nullCount < n { chunks = max(chunks, 1) }
        var perm: MetalArray<Int32>? = nil                  // nil means "the identity so far"
        if chunks > 0 {
            for chunk in stride(from: chunks - 1, through: 0, by: -1) {
                let keys = try prefixKeys(chunk: chunk, order: perm, descending: descending, nulls: nulls)
                let step = try keys.argsort()
                perm = try perm.map { try $0.take(step) } ?? step
            }
        }
        return try perm ?? MetalArray<Int32>.iota(n, context: ctx)
    }

    /// The rows in byte-wise lexicographic order (`argsort` then `take`).
    public func sorted(descending: Bool = false,
                       nullPlacement: NullPlacement = .atEnd) throws -> MetalStringArray {
        let res = try take(try argsort(descending: descending, nullPlacement: nullPlacement))
        res.isBinary = isBinary
        return res
    }

    /// Radix passes needed to cover every byte of the longest row.
    private func chunkCount() throws -> Int {
        guard let longest = try byteLength().max() else { return 0 }   // every row null
        let per = StringSortSource.bytesPerChunk
        return (Int(longest) + per - 1) / per
    }

    /// The 63-bit key of chunk `chunk` for each row, read in `order` (the identity when it is nil). With
    /// `nulls`, bit 63 moves the null rows to the placement's end (`StringSortSource`).
    private func prefixKeys(chunk: Int, order: MetalArray<Int32>?, descending: Bool,
                            nulls: (MetalArrowBuffer, NullPlacement)?) throws -> MetalArray<UInt64> {
        let n = length
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: n * 8, zeroed: false, context: ctx)
        let pso = try ctx.pipeline(source: StringSortSource.source, function: "str_prefix_key",
                                  cacheKey: "strsort/str_prefix_key")
        let permBuf = order?.values ?? out                  // an unused binding still needs a buffer
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
            enc.setBuffer(data.mtl, offset: data.offset, index: 1)
            enc.setBuffer(permBuf.mtl, offset: permBuf.offset, index: 2)
            Dispatch.setUInt(enc, n, index: 3)
            Dispatch.setUInt(enc, chunk, index: 4)
            Dispatch.setUInt(enc, order == nil ? 0 : 1, index: 5)
            Dispatch.setUInt(enc, descending ? 1 : 0, index: 6)
            enc.setBuffer(out.mtl, offset: out.offset, index: 7)
            let bm = nulls?.0 ?? out                        // unused when there are no nulls
            enc.setBuffer(bm.mtl, offset: bm.offset, index: 8)
            Dispatch.setUInt(enc, nulls.map { $0.1 == .atStart ? 2 : 1 } ?? 0, index: 9)
            Dispatch.dispatch1D(enc, pso, count: n)
        }
        if let order { ctx.retainUntilFlush(order) }
        return MetalArray<UInt64>(length: n, nullCount: 0, validity: nil, values: out, context: ctx)
    }
}
