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
    static func prefersHashTable(rows: Int) -> Bool { rows > 0 }

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
        let flagsBase = (validity == nil ? 0 : Self.hasValidityFlag) | (verify ? Self.verifyFlag : 0)
        let v = validity ?? offsets

        // Largest table we would ever build: at least two slots per non-null row, so an insert into it
        // always finds a free slot and the last attempt cannot fail.
        var cap = 1024
        while cap < 2 * nonNull { cap <<= 1 }

        // Cardinality estimate from a 1/64 slice of the hash space. Below the threshold the table is
        // small either way, so the estimate is not worth a pass.
        var estimate = nonNull
        if initialSlots != nil {
            estimate = 0
        } else if nonNull > 1 << 17 {
            let sampleBits = 6
            var pilot = 1024
            while pilot < 4 * (nonNull >> sampleBits) { pilot <<= 1 }
            pilot = Swift.min(pilot, 1 << 21)
            let slots = try MetalArrowBuffer.allocate(byteCount: pilot * 4, zeroed: false, context: ctx)
            let failed = try build(keys: keys, slots: slots, slotCount: pilot, flags: flagsBase,
                                   sampleMask: (1 << sampleBits) - 1, maxProbe: 64,
                                   slotOf: nil, firstOfSlot: nil)
            let sampled = try failed ? 0 : occupancy(slots: slots, count: pilot).total
            // A pilot that ran out of probe budget is one whose slice was denser than the table it was
            // given; fall back to sizing for the worst case rather than trusting a truncated count.
            estimate = sampled == 0 ? nonNull : Swift.min(nonNull, sampled << sampleBits)
            ctx.retainUntilFlush(slots)
        }

        // Three slots per estimated distinct value keeps the load factor near 0.3, where linear probing
        // is still one probe for almost every row.
        var slotCount = initialSlots ?? 1024
        while slotCount < 3 * estimate && slotCount < cap { slotCount <<= 1 }
        slotCount = Swift.min(slotCount, cap)

        let slotOf = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: false, context: ctx)
        var slots: MetalArrowBuffer
        var firstOfSlot: MetalArrowBuffer
        while true {
            let last = slotCount >= cap
            slots = try MetalArrowBuffer.allocate(byteCount: slotCount * 4, zeroed: false, context: ctx)
            firstOfSlot = try MetalArrowBuffer.allocate(byteCount: slotCount * 4, zeroed: false, context: ctx)
            let failed = try build(keys: keys, slots: slots, slotCount: slotCount,
                                   flags: flagsBase | Self.writeSlotsFlag, sampleMask: 0,
                                   maxProbe: last ? Int(UInt32.max) : 96,
                                   slotOf: slotOf, firstOfSlot: firstOfSlot)
            if !failed { break }
            // The estimate was low and probe chains grew past the budget. Retry with a bigger table; the
            // last attempt has two slots per row and no budget, so this terminates.
            slotCount = Swift.min(slotCount << 3, cap)
        }
        ctx.retainUntilFlush(slots); ctx.retainUntilFlush(firstOfSlot); ctx.retainUntilFlush(slotOf)

        let (cum, groupCount) = try occupancy(slots: slots, count: slotCount)
        // One representative row per group, in slot order, then the relabelling that turns slot order
        // into first-seen order.
        let firstSlotOrder = try compact(slots: slots, firstOfSlot: firstOfSlot, cum: cum,
                                         slotCount: slotCount, groupCount: groupCount)
        let perm = try firstSlotOrder.argsort()
        let relabel = try perm.argsort()
        let firstRows = try firstSlotOrder.take(perm)

        let ids = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: false, context: ctx)
        let p = try Self.pso(ctx, "sht_ids")
        try ctx.run { enc in
            enc.setComputePipelineState(p)
            enc.setBuffer(slotOf.mtl, offset: slotOf.offset, index: 0)
            enc.setBuffer(v.mtl, offset: v.offset, index: 1)
            Dispatch.setUInt(enc, validity == nil ? 0 : Self.hasValidityFlag, index: 2)
            Dispatch.setUInt(enc, nullId ?? groupCount, index: 3)
            enc.setBuffer(cum.values.mtl, offset: cum.values.offset, index: 4)
            enc.setBuffer(relabel.values.mtl, offset: relabel.values.offset, index: 5)
            Dispatch.setLength(enc, n, nil, index: 6)
            enc.setBuffer(ids.mtl, offset: ids.offset, index: 7)
            Dispatch.dispatch1D(enc, p, count: n)
        }
        ctx.retainUntilFlush(cum); ctx.retainUntilFlush(relabel); ctx.retainUntilFlush(self)
        try ctx.syncPoint()
        return StringHashTableIds(ids: ids, rows: n, groupCount: groupCount, firstRows: firstRows)
    }

    // MARK: - Stages

    private static let hasValidityFlag = 1
    private static let verifyFlag = 2
    private static let writeSlotsFlag = 4

    private static func pso(_ ctx: MetalContext, _ fn: String) throws -> MTLComputePipelineState {
        try ctx.pipeline(source: StringHashTableSource.source, function: fn, cacheKey: "strhash/\(fn)")
    }

    /// Clears the table and inserts every (sampled, non-null) row. Returns true when a row ran out of
    /// probe budget, which means the table was too small — never that the input was bad.
    private func build(keys: MetalArray<UInt64>, slots: MetalArrowBuffer, slotCount: Int, flags: Int,
                       sampleMask: Int, maxProbe: Int, slotOf: MetalArrowBuffer?,
                       firstOfSlot: MetalArrowBuffer?) throws -> Bool {
        let ctx = context
        let errorFlag = try MetalArrowBuffer.allocate(byteCount: 4, context: ctx)
        let fillPSO = try Self.pso(ctx, "sht_fill")
        let buildPSO = try Self.pso(ctx, "sht_build")
        let v = validity ?? offsets
        try ctx.run { enc in
            enc.setComputePipelineState(fillPSO)
            enc.setBuffer(slots.mtl, offset: slots.offset, index: 0)
            Dispatch.setUInt(enc, 0, index: 1)
            Dispatch.setLength(enc, slotCount, nil, index: 2)
            Dispatch.dispatch1D(enc, fillPSO, count: slotCount)
            if let firstOfSlot {
                enc.setBuffer(firstOfSlot.mtl, offset: firstOfSlot.offset, index: 0)
                Dispatch.setUInt(enc, Int(UInt32.max), index: 1)
                Dispatch.setLength(enc, slotCount, nil, index: 2)
                Dispatch.dispatch1D(enc, fillPSO, count: slotCount)
            }
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(buildPSO)
            enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
            enc.setBuffer(data.mtl, offset: data.offset, index: 1)
            enc.setBuffer(keys.values.mtl, offset: keys.values.offset, index: 2)
            enc.setBuffer(v.mtl, offset: v.offset, index: 3)
            Dispatch.setUInt(enc, flags, index: 4)
            Dispatch.setLength(enc, length, nil, index: 5)
            Dispatch.setUInt(enc, slotCount - 1, index: 6)
            Dispatch.setUInt(enc, sampleMask, index: 7)
            Dispatch.setUInt(enc, maxProbe, index: 8)
            enc.setBuffer(slots.mtl, offset: slots.offset, index: 9)
            enc.setBuffer((slotOf ?? errorFlag).mtl, offset: (slotOf ?? errorFlag).offset, index: 10)
            enc.setBuffer((firstOfSlot ?? errorFlag).mtl, offset: (firstOfSlot ?? errorFlag).offset, index: 11)
            enc.setBuffer(errorFlag.mtl, offset: errorFlag.offset, index: 12)
            Dispatch.dispatch1D(enc, buildPSO, count: length)
        }
        ctx.retainUntilFlush(keys); ctx.retainUntilFlush(self)
        try ctx.syncPoint()
        return withExtendedLifetime(errorFlag) { errorFlag.typed(UInt32.self)[0] != 0 }
    }

    /// Marks the occupied slots and scans the marks: `cum[s] - 1` is the rank of slot `s` among the
    /// occupied ones, and the last entry is how many groups there are.
    private func occupancy(slots: MetalArrowBuffer, count: Int) throws -> (cum: MetalArray<Int32>, total: Int) {
        let ctx = context
        let marks = try MetalArrowBuffer.allocate(byteCount: count * 4, zeroed: false, context: ctx)
        let p = try Self.pso(ctx, "sht_mark")
        try ctx.run { enc in
            enc.setComputePipelineState(p)
            enc.setBuffer(slots.mtl, offset: slots.offset, index: 0)
            Dispatch.setLength(enc, count, nil, index: 1)
            enc.setBuffer(marks.mtl, offset: marks.offset, index: 2)
            Dispatch.dispatch1D(enc, p, count: count)
        }
        ctx.retainUntilFlush(slots)
        let cum = try MetalArray<Int32>(length: count, nullCount: 0, validity: nil, values: marks, context: ctx)
            .cumulativeSum()
        try ctx.syncPoint()
        let total = withExtendedLifetime(cum) { Int(cum.valuePointer[count - 1]) }
        return (cum, total)
    }

    /// One representative row per occupied slot, in slot order.
    private func compact(slots: MetalArrowBuffer, firstOfSlot: MetalArrowBuffer, cum: MetalArray<Int32>,
                         slotCount: Int, groupCount: Int) throws -> MetalArray<Int32> {
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(groupCount, 1) * 4, zeroed: false, context: ctx)
        if groupCount > 0 {
            let p = try Self.pso(ctx, "sht_compact")
            try ctx.run { enc in
                enc.setComputePipelineState(p)
                enc.setBuffer(slots.mtl, offset: slots.offset, index: 0)
                enc.setBuffer(firstOfSlot.mtl, offset: firstOfSlot.offset, index: 1)
                enc.setBuffer(cum.values.mtl, offset: cum.values.offset, index: 2)
                Dispatch.setLength(enc, slotCount, nil, index: 3)
                enc.setBuffer(out.mtl, offset: out.offset, index: 4)
                Dispatch.dispatch1D(enc, p, count: slotCount)
            }
            ctx.retainUntilFlush(slots); ctx.retainUntilFlush(firstOfSlot); ctx.retainUntilFlush(cum)
            try ctx.syncPoint()
        }
        return MetalArray<Int32>(length: groupCount, nullCount: 0, validity: nil, values: out, context: ctx)
    }
}
