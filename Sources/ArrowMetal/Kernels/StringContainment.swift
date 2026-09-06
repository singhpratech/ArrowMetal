import Foundation
import Metal

/// Arrow `is_in` / `index_in` over `utf8` columns, and `binary_join` over a `list<utf8>`.
///
/// ## is_in / index_in
///
/// The value set is hashed with the same 64-bit key `Kernels/StringDictionary.swift` builds — two
/// independent MurmurHash3 x86_32 passes with different seeds, side by side — and inserted into an
/// open-addressing table of **row indices** with linear probing. Probing hashes the same way and
/// confirms every candidate by comparing the full bytes, so a hash collision costs one extra probe
/// and can never produce a wrong answer; the table needs no retry loop the way `dictionaryEncode`
/// does, because nothing here depends on distinct strings landing in distinct buckets.
///
/// Duplicates in the value set collapse onto one slot, and the insert keeps the **lowest** row index,
/// which is exactly the first occurrence `index_in` reports.
///
/// ## Null handling
///
/// Arrow's `null_matching_behavior = "skip"`, the same rule the primitive `isIn` / `indexIn` in
/// `Kernels/Structural.swift` follow: nulls in the value set are ignored, and a null probe is never in
/// the set (`is_in` gives false, `index_in` gives null). pyarrow's default is the opposite
/// (`skip_nulls=False`, where a null probe matches a null in the value set); that option is **not
/// implemented** here — pass `skip_nulls=True` to pyarrow to compare.
extension MetalStringArray {

    private func containmentPipeline(_ fn: String) throws -> MTLComputePipelineState {
        try context.pipeline(source: StringExtraSource.source, function: fn, cacheKey: "strx/\(fn)")
    }

    /// A value set prepared for GPU probing: a power-of-two table of set row indices, `0xFFFFFFFF`
    /// meaning empty.
    struct StringLookupTable {
        let set: MetalStringArray
        let table: MetalArrowBuffer
        let mask: Int
    }

    /// The 64-bit grouping key: two independently seeded 32-bit hashes side by side, as
    /// `dictionaryEncodeGPU` uses.
    func lookupKeys() throws -> MetalArray<UInt64> {
        try Self.compose(try hash32(), try seededHash32(0x9E37_79B9))
    }

    /// Builds the probe table, or nil when the value set holds no non-null string (then nothing can
    /// ever match).
    static func buildLookup(_ set: MetalStringArray) throws -> StringLookupTable? {
        let nonNull = set.length - set.nullCount
        guard nonNull > 0 else { return nil }
        try Dispatch.checkLength(set.length)
        let ctx = set.context
        var slots = 64
        while slots < nonNull * 2 { slots <<= 1 }
        let table = try MetalArrowBuffer.allocate(byteCount: slots * 4, zeroed: false, context: ctx)
        memset(table.mutableContents, 0xFF, slots * 4)
        let keys = try set.lookupKeys()
        let vb = set.validity ?? table
        let pso = try set.containmentPipeline("sx_hash_insert")
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(set.offsets.mtl, offset: set.offsets.offset, index: 0)
            enc.setBuffer(set.data.mtl, offset: set.data.offset, index: 1)
            enc.setBuffer(keys.values.mtl, offset: keys.values.offset, index: 2)
            enc.setBuffer(vb.mtl, offset: vb.offset, index: 3)
            Dispatch.setLength(enc, set.length, nil, index: 4)
            Dispatch.setUInt(enc, slots - 1, index: 5)
            Dispatch.setUInt(enc, set.validity == nil ? 0 : 1, index: 6)
            enc.setBuffer(table.mtl, offset: table.offset, index: 7)
            Dispatch.dispatch1D(enc, pso, count: set.length)
        }
        ctx.retainUntilFlush(keys); ctx.retainUntilFlush(set)
        return StringLookupTable(set: set, table: table, mask: slots - 1)
    }

    /// Arrow `is_in` for strings: true where the value appears among the non-null values of `set`.
    /// The result never contains nulls (a null value is simply not in the set).
    public func isIn(_ set: MetalStringArray) throws -> MetalBooleanArray {
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let words = BitmapOps.words(bits: n)
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 1),
                                                zeroed: true, context: ctx)
        if n > 0, let lookup = try Self.buildLookup(set) {
            let keys = try lookupKeys()
            let vb = validity ?? out
            let pso = try containmentPipeline("sx_is_in")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                enc.setBuffer(keys.values.mtl, offset: keys.values.offset, index: 2)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 3)
                enc.setBuffer(lookup.set.offsets.mtl, offset: lookup.set.offsets.offset, index: 4)
                enc.setBuffer(lookup.set.data.mtl, offset: lookup.set.data.offset, index: 5)
                enc.setBuffer(lookup.table.mtl, offset: lookup.table.offset, index: 6)
                Dispatch.setLength(enc, n, nil, index: 7)
                Dispatch.setUInt(enc, lookup.mask, index: 8)
                Dispatch.setUInt(enc, validity == nil ? 0 : 1, index: 9)
                enc.setBuffer(out.mtl, offset: out.offset, index: 10)
                Dispatch.dispatch1D(enc, pso, count: words)
            }
            ctx.retainUntilFlush(keys); ctx.retainUntilFlush(lookup.table); ctx.retainUntilFlush(self)
        }
        return MetalBooleanArray(length: n, nullCount: 0, validity: nil, values: out, context: ctx)
    }

    /// Arrow `is_in` against a host-side value set.
    public func isIn(_ set: [String?]) throws -> MetalBooleanArray {
        try isIn(try MetalStringArray(set, context: context))
    }

    /// Arrow `index_in` for strings: the int32 position in `set` of each value's **first** occurrence
    /// there, null where the value is null or absent from the set.
    public func indexIn(_ set: MetalStringArray) throws -> MetalArray<Int32> {
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let outValues = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * 4, context: ctx)
        let validBytes = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1), context: ctx)
        if n > 0, let lookup = try Self.buildLookup(set) {
            let keys = try lookupKeys()
            let vb = validity ?? validBytes
            let pso = try containmentPipeline("sx_index_in")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                enc.setBuffer(keys.values.mtl, offset: keys.values.offset, index: 2)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 3)
                enc.setBuffer(lookup.set.offsets.mtl, offset: lookup.set.offsets.offset, index: 4)
                enc.setBuffer(lookup.set.data.mtl, offset: lookup.set.data.offset, index: 5)
                enc.setBuffer(lookup.table.mtl, offset: lookup.table.offset, index: 6)
                Dispatch.setLength(enc, n, nil, index: 7)
                Dispatch.setUInt(enc, lookup.mask, index: 8)
                Dispatch.setUInt(enc, validity == nil ? 0 : 1, index: 9)
                enc.setBuffer(outValues.mtl, offset: outValues.offset, index: 10)
                enc.setBuffer(validBytes.mtl, offset: validBytes.offset, index: 11)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
            ctx.retainUntilFlush(keys); ctx.retainUntilFlush(lookup.table); ctx.retainUntilFlush(self)
        }
        let outValidity = n > 0 ? try BitmapOps.packBits(ctx, bytes: validBytes, bits: n) : nil
        ctx.retainUntilFlush(validBytes)
        let res = MetalArray<Int32>(length: n, nullCount: 0, validity: outValidity, values: outValues, context: ctx)
        res.recomputeNullCount()
        return res
    }

    /// Arrow `index_in` against a host-side value set.
    public func indexIn(_ set: [String?]) throws -> MetalArray<Int32> {
        try indexIn(try MetalStringArray(set, context: context))
    }

    /// Arrow `binary_join`: joins every row of a `list<utf8>` with one scalar separator.
    public static func binaryJoin(list: MetalListArray, separator: String) throws -> MetalStringArray {
        try list.binaryJoin(separator: separator)
    }
    /// Arrow `binary_join` with a per-row separator column.
    public static func binaryJoin(list: MetalListArray, separator: MetalStringArray) throws -> MetalStringArray {
        try list.binaryJoin(separator: separator)
    }
}

extension MetalListArray {

    /// Arrow `binary_join`: concatenates the child strings of every row, separated by `separator`.
    ///
    /// Two GPU passes, the same shape the string transforms use: one kernel sums the child byte
    /// lengths (plus `count - 1` separators) into a per-row output length, the host scans those into
    /// the Arrow offsets buffer, and a second kernel copies the bytes. An empty row joins to the empty
    /// string. A null row, **any** null element inside a row, and a null separator all give a null
    /// output row — Arrow's default `EMIT_NULL` null handling; the `REPLACE` / `SKIP` options are not
    /// implemented.
    public func binaryJoin(separator: String) throws -> MetalStringArray {
        try join(scalar: Array(separator.utf8), array: nil)
    }

    /// Arrow `binary_join` with one separator per row. A null separator gives a null output row.
    public func binaryJoin(separator: MetalStringArray) throws -> MetalStringArray {
        guard separator.length == length else { throw ArrowMetalError.lengthMismatch(length, separator.length) }
        return try join(scalar: [], array: separator)
    }

    private func join(scalar: [UInt8], array sepArray: MetalStringArray?) throws -> MetalStringArray {
        let child: MetalStringArray
        switch values {
        case .string(let c), .binary(let c): child = c
        default:
            throw ArrowMetalError.unsupportedType("binaryJoin needs a list of utf8, got \(arrowFormat) of \(values.arrowFormat)")
        }
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let lens = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 4, 4), zeroed: true, context: ctx)
        let validBytes = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1), context: ctx)
        let sepBuf = try MetalArrowBuffer.allocate(byteCount: Swift.max(scalar.count, 1), zeroed: false, context: ctx)
        if !scalar.isEmpty { scalar.withUnsafeBytes { memcpy(sepBuf.mutableContents, $0.baseAddress!, $0.count) } }

        let lv = validity ?? sepBuf                       // never read when the flag bit is clear
        let cv = child.validity ?? sepBuf
        let so = sepArray?.offsets ?? sepBuf
        let sd = sepArray?.data ?? sepBuf
        let sv = sepArray?.validity ?? sepBuf
        let flags = (validity == nil ? 0 : 1) | (child.validity == nil ? 0 : 2)
            | (sepArray == nil ? 0 : 4) | (sepArray?.validity == nil ? 0 : 8)
        var prm = MetalStringArray.sxParams(0, scalar.count, 0, 0, 0, flags)

        if n > 0 {
            let pLen = try ctx.pipeline(source: StringExtraSource.source, function: "sx_join_len",
                                        cacheKey: "strx/sx_join_len")
            try ctx.run { enc in
                enc.setComputePipelineState(pLen)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(child.offsets.mtl, offset: child.offsets.offset, index: 1)
                enc.setBuffer(lv.mtl, offset: lv.offset, index: 2)
                enc.setBuffer(cv.mtl, offset: cv.offset, index: 3)
                enc.setBuffer(so.mtl, offset: so.offset, index: 4)
                enc.setBuffer(sv.mtl, offset: sv.offset, index: 5)
                Dispatch.setLength(enc, n, nil, index: 6)
                enc.setBytes(&prm, length: 24, index: 7)
                enc.setBuffer(lens.mtl, offset: lens.offset, index: 8)
                enc.setBuffer(validBytes.mtl, offset: validBytes.offset, index: 9)
                Dispatch.dispatch1D(enc, pLen, count: n)
            }
        }
        let outOffsets = try MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: lens, context: ctx)
            .exclusiveScanToOffsets()
        let total = Int(withExtendedLifetime(outOffsets) { outOffsets.typed(Int32.self)[n] })
        let outData = try MetalArrowBuffer.allocate(byteCount: total, zeroed: false, context: ctx)
        if n > 0 {
            let pWrite = try ctx.pipeline(source: StringExtraSource.source, function: "sx_join_write",
                                          cacheKey: "strx/sx_join_write")
            try ctx.run { enc in
                enc.setComputePipelineState(pWrite)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(child.offsets.mtl, offset: child.offsets.offset, index: 1)
                enc.setBuffer(child.data.mtl, offset: child.data.offset, index: 2)
                enc.setBuffer(lv.mtl, offset: lv.offset, index: 3)
                enc.setBuffer(cv.mtl, offset: cv.offset, index: 4)
                enc.setBuffer(so.mtl, offset: so.offset, index: 5)
                enc.setBuffer(sd.mtl, offset: sd.offset, index: 6)
                enc.setBuffer(sv.mtl, offset: sv.offset, index: 7)
                Dispatch.setLength(enc, n, nil, index: 8)
                enc.setBytes(&prm, length: 24, index: 9)
                enc.setBuffer(outOffsets.mtl, offset: outOffsets.offset, index: 10)
                enc.setBuffer(outData.mtl, offset: outData.offset, index: 11)
                Dispatch.dispatch1D(enc, pWrite, count: n)
            }
        }
        for b in [sepBuf, lens] { ctx.retainUntilFlush(b) }
        ctx.retainUntilFlush(self); ctx.retainUntilFlush(child)
        if let sepArray { ctx.retainUntilFlush(sepArray) }
        let outValidity = n > 0 ? try BitmapOps.packBits(ctx, bytes: validBytes, bits: n) : nil
        ctx.retainUntilFlush(validBytes)
        var nulls = 0
        if let v = outValidity {
            try ctx.syncPoint()
            nulls = n - Bitmap.popcount(v.typed(UInt8.self), bits: n)
        }
        return MetalStringArray(length: n, nullCount: nulls, validity: outValidity,
                                offsets: outOffsets, data: outData, context: ctx)
    }
}
