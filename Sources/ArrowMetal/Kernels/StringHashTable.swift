import Foundation
import Metal

/// Dense group ids for a `utf8` / `binary` column, built by a GPU hash table instead of a sort.
///
/// ## Why
///
/// `Kernels/StringDictionary.swift` maps strings to ids by argsorting a 64-bit hash of every row. That
/// is O(rows log rows) with eight radix passes and, crucially, its cost does not depend on how many
/// distinct strings there are: at 50 million rows the argsort alone is ~185 ms whether the column holds
/// a thousand keys or ten million. Real group-by keys are low cardinality, and a hash table's cost
/// scales with the *distinct* count, not the row count, because the table fits in cache when the keys
/// are few.
///
/// ## How
///
/// 1. **Hash** every non-null row to 64 bits, one pass over the bytes (`sht_hash64`).
/// 2. **Estimate** the distinct count by building a table over a 1/64 slice of the *hash space*
///    (`sht_build` with a sample mask). Each distinct string decides once whether it is in the slice,
///    so the occupied-slot count times 64 estimates the cardinality however skewed the rows are. The
///    estimate only sizes the real table; being wrong costs a retry, never a wrong answer.
/// 3. **Build** the real table: open addressing, linear probing, `slots[s] = row + 1`, equality decided
///    by comparing the **bytes** of the candidate against the slot's representative row. Every row also
///    records the slot it landed in and folds itself into that slot's lowest row index.
/// 4. **Rank** the occupied slots with the same GPU scan `unique()` uses, and compact one representative
///    row per slot.
/// 5. **Relabel** into first-seen order — two argsorts over the *distinct* count — and write each row's
///    id in one pass.
///
/// The result is identical to the sort path: same group ids, same group keys, same first-seen
/// dictionary order, same null semantics. Byte comparison is what makes that true rather than merely
/// likely: a 64-bit hash collision costs one extra probe and cannot merge two different strings, so
/// unlike `dictionaryEncodeGPU` this path has no re-hash retry and no host fallback.
struct StringHashTableIds {
    /// One int32 per row: the dense group id, with null rows holding the caller's `nullId`.
    let ids: MetalArrowBuffer
    let rows: Int
    /// Number of distinct non-null strings.
    let groupCount: Int
    /// The lowest row index of each group, in group order (which is first-seen order).
    let firstRows: MetalArray<Int32>
}

extension MetalStringArray {

    // MARK: - Entry points

    /// Whether the hash table beats the sort for a column of this many rows.
    ///
    /// It wins everywhere it has been measured — the table is fewer passes than the argsort even at a
    /// hundred rows — but the sort path stays reachable for a zero-row column, where there is no table
    /// to build, and as the reference the tests compare against.
    static func prefersHashTable(rows: Int) -> Bool { rows > 0 && !HashTable.disabled }

    /// Dense ids `0 ..< groupCount` for `GroupByKeys`, null rows taking the dedicated id `groupCount`.
    /// Returns the ids and the group count *including* the null group, exactly as `densify` does.
    func hashTableDenseIds() throws -> (MetalArray<Int32>, Int) {
        let r = try hashTableIds(nullId: nil)
        let ids = MetalArray<Int32>(length: length, nullCount: 0, validity: nil, values: r.ids, context: context)
        return (ids, r.groupCount + (nullCount > 0 ? 1 : 0))
    }

    /// Arrow `dictionary_encode` through the hash table: dense int32 codes plus the distinct strings in
    /// first-seen order, byte for byte what `dictionaryEncodeCPU` produces. Null rows get a null code.
    func dictionaryEncodeHashTable() throws -> (codes: MetalArray<Int32>, unique: MetalStringArray) {
        let ctx = context
        let r = try hashTableIds(nullId: 0)
        guard r.groupCount > 0 else {
            let buf = try MetalArrowBuffer.allocate(byteCount: Swift.max(length, 1) * 4, context: ctx)
            let codes = MetalArray<Int32>(length: length, nullCount: length, validity: validity, values: buf, context: ctx)
            return (codes, try MetalStringArray([], context: ctx))
        }
        // Null rows hold id 0 and are marked null by the input's own bitmap, which is what the sort path
        // hands back too (a null code, not a code into the dictionary).
        let codes = MetalArray<Int32>(length: length, nullCount: nullCount, validity: validity,
                                      values: r.ids, context: ctx)
        let reps = try gather(r.firstRows)
        // The dictionary has no nulls, so it carries no bitmap (the gather leaves an all-ones one).
        let unique = reps.nullCount == 0 && reps.validity != nil
            ? MetalStringArray(length: reps.length, nullCount: 0, validity: nil,
                               offsets: reps.offsets, data: reps.data, context: ctx)
            : reps
        return (codes, unique)
    }

    // MARK: - The table

    static func pso(_ ctx: MetalContext, _ fn: String) throws -> MTLComputePipelineState {
        try ctx.pipeline(source: StringHashTableSource.source, function: fn, cacheKey: "strhash/\(fn)")
    }

    /// MurmurHash-grade 64-bit hash of every row's bytes, both halves from one pass over the data.
    /// Null rows hash to 0 and are never inserted.
    func hash64() throws -> MetalArray<UInt64> {
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(length, 1) * 8, zeroed: false, context: context)
        if length > 0 {
            let p = try Self.pso(context, "sht_hash64")
            let v = validity ?? offsets
            try context.run { enc in
                enc.setComputePipelineState(p)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                enc.setBuffer(v.mtl, offset: v.offset, index: 2)
                Dispatch.setUInt(enc, validity == nil ? 0 : 1, index: 3)
                Dispatch.setLength(enc, length, nil, index: 4)
                enc.setBuffer(out.mtl, offset: out.offset, index: 5)
                Dispatch.dispatch1D(enc, p, count: length)
            }
            context.retainUntilFlush(self)
        }
        return MetalArray<UInt64>(length: length, nullCount: nullCount, validity: validity, values: out, context: context)
    }

    /// Builds the table and turns it into one dense id per row.
    ///
    /// `nullId` is what a null row's id becomes; `nil` means "the group after the last one", which is
    /// the id `GroupByKeys` gives the null group. `hashes` overrides the computed hashes and exists for
    /// the tests, which feed deliberately colliding keys through to prove byte comparison decides
    /// equality. `verify: false` decides equality on the 64-bit hash alone — faster, but only correct up
    /// to the birthday bound, so nothing in the library passes it.
    /// `initialSlots` overrides the estimated table size and exists for the tests, which force the growth
    /// retry by starting from a table far too small for the input.
    func hashTableIds(nullId: Int?, hashes: MetalArray<UInt64>? = nil, verify: Bool = true,
                      initialSlots: Int? = nil) throws -> StringHashTableIds {
        let ctx = context
        let n = length
        try Dispatch.checkLength(n)
        guard n <= Int(Int32.max) else {
            throw ArrowMetalError.invalidArrowArray("string hash table: columns above 2^31 rows are not supported")
        }
        let nonNull = n - nullCount
        guard nonNull > 0 else {
            // Empty, or every row null: with no groups at all, the null id is 0 either way, which is
            // what the zeroed buffer already holds.
            let ids = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * 4, context: ctx)
            return StringHashTableIds(ids: ids, rows: n, groupCount: 0,
                                      firstRows: try MetalArray<Int32>([Int32](), context: ctx))
        }

        let keys = try hashes ?? hash64()
        let v = validity ?? offsets
        let flagsBase = (validity == nil ? 0 : 1) | (verify ? 2 : 0)

        // The estimate, the growth retry and the rank scan are shared with the primitive table; only the
        // insert kernel differs, because only this one compares bytes.
        let g = try HashTable.groups(ctx: ctx, rows: n, nonNull: nonNull, initialSlots: initialSlots) {
            slots, slotCount, sampleMask, maxProbe, slotOf, firstOfSlot in
            try HashTable.runBuild(ctx: ctx, function: "sht_build", source: StringHashTableSource.source,
                                   cacheKey: "strhash/sht_build", slots: slots, slotCount: slotCount,
                                   rows: n, slotOf: slotOf, firstOfSlot: firstOfSlot) { enc, errorFlag in
                enc.setBuffer(self.offsets.mtl, offset: self.offsets.offset, index: 0)
                enc.setBuffer(self.data.mtl, offset: self.data.offset, index: 1)
                enc.setBuffer(keys.values.mtl, offset: keys.values.offset, index: 2)
                enc.setBuffer(v.mtl, offset: v.offset, index: 3)
                Dispatch.setUInt(enc, flagsBase | (slotOf != nil ? 4 : 0), index: 4)
                Dispatch.setLength(enc, n, nil, index: 5)
                Dispatch.setUInt(enc, slotCount - 1, index: 6)
                Dispatch.setUInt(enc, sampleMask, index: 7)
                Dispatch.setUInt(enc, maxProbe, index: 8)
                enc.setBuffer(slots.mtl, offset: slots.offset, index: 9)
                enc.setBuffer((slotOf ?? errorFlag).mtl, offset: (slotOf ?? errorFlag).offset, index: 10)
                enc.setBuffer((firstOfSlot ?? errorFlag).mtl, offset: (firstOfSlot ?? errorFlag).offset, index: 11)
                enc.setBuffer(errorFlag.mtl, offset: errorFlag.offset, index: 12)
            }
        }
        ctx.retainUntilFlush(keys); ctx.retainUntilFlush(self)

        // Slot order into first-seen order: sort the groups by their lowest row, and invert.
        let perm = try g.firstSlotOrder.argsort()
        let relabel = try perm.argsort()
        let firstRows = try g.firstSlotOrder.take(perm)
        let ids = try HashTable.writeIds(ctx: ctx, rows: n, validity: validity, fallbackBuffer: offsets,
                                         groups: g, relabel: relabel, nullId: nullId ?? g.groupCount)
        return StringHashTableIds(ids: ids, rows: n, groupCount: g.groupCount, firstRows: firstRows)
    }
}
