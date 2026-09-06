import Foundation
import Metal

/// The GPU hash table: what `unique`, `value_counts`, `count_distinct`, `mode`, `dictionary_encode` and
/// `GroupByKeys` use instead of a sort when the key column has fewer distinct values than rows.
///
/// ## Why
///
/// Every one of those functions used to start with an argsort of the whole column — eight radix passes
/// over 50 million 64-bit keys, ~185 ms — and then find the runs. That cost is the same whether the
/// column holds a thousand distinct values or ten million, which is exactly backwards: the work these
/// functions actually have to do is proportional to the *distinct* count. A hash table sized to the
/// cardinality fits in cache when the keys are few, and only the very last stage (ordering the distinct
/// values) sorts anything — and it sorts `K` values, not `n` rows.
///
/// ## The stages
///
/// 1. **Key**: one 64-bit key per row. Integers widen (sign-extended, so the map is injective); floats
///    are normalised first, every NaN to one pattern and -0 to +0, so bit equality is Arrow value
///    equality. Strings hash their bytes instead (see `Kernels/StringHashTable.swift`).
/// 2. **Estimate**: a throwaway build over a 1/64 slice of the *hash space*. Each distinct key decides
///    once whether it is in the slice, so occupied-slots × 64 estimates the cardinality however skewed
///    the row frequencies are — a *row* sample would only find the frequent keys. The estimate sizes the
///    real table; being wrong costs a retry, never a wrong answer.
/// 3. **Build**: open addressing, linear probing, three slots per estimated distinct value. Each row
///    records the slot it landed in and folds itself into that slot's lowest row index.
/// 4. **Rank**: mark the occupied slots and scan the marks (the same GPU scan `unique()` uses), then
///    compact one representative row per slot.
/// 5. **Order**: the caller decides. `unique` and friends gather the representative values and argsort
///    those `K` values into Arrow's ascending order; `GroupByKeys` over strings argsorts the
///    representative *rows* into first-appearance order. Either way the relabelling is a permutation of
///    `K` elements, and one pass writes each row's final id.
///
/// The answer is identical to the sort path — same value set, same order, same representative for a
/// group of equal-but-distinguishable values (`-0.0` among `0.0`s comes from the earliest row, because
/// the table keeps the lowest row per slot), same null semantics.
///
/// First-appearance order comes free: `HashGroups.firstSlotOrder` is the lowest row of each group, so
/// ordering the groups by it is one argsort of `K` elements rather than a second pass over the rows.
enum HashTable {

    /// The built table, ranked: everything that does not depend on how the groups are ordered.
    struct HashGroups {
        /// Slot each row landed in. Undefined at null rows, which are never inserted.
        let slotOf: MetalArrowBuffer
        /// Inclusive scan of the slot occupancy: `cum[s] - 1` is the rank of slot `s`.
        let cum: MetalArray<Int32>
        /// Number of distinct non-null keys.
        let groupCount: Int
        /// Lowest row index of each group, in slot order.
        let firstSlotOrder: MetalArray<Int32>
    }

    /// Set `ARROWMETAL_NO_HASH=1` to force every caller back onto the sort path. Nothing in the library
    /// needs it — both paths give the same answer — but it is how the before/after benchmark numbers in
    /// `docs/DESIGN.md` were measured, in one binary.
    static let disabled = ProcessInfo.processInfo.environment["ARROWMETAL_NO_HASH"] != nil

    static func pso(_ ctx: MetalContext, _ fn: String) throws -> MTLComputePipelineState {
        try ctx.pipeline(source: HashTableSource.shared, function: fn, cacheKey: "hashtable/\(fn)")
    }

    /// Builds the table and ranks it. `build` clears the slots and inserts the rows; it is handed the
    /// table, its size, the hash-space sample mask (0 for "every row"), the probe budget and the two
    /// output arrays, and returns true when a row ran out of probe budget — which means the table was
    /// too small, never that the input was bad.
    ///
    /// `initialSlots` overrides the estimate and exists for the tests, which force the growth retry by
    /// starting from a table far too small for the input.
    static func groups(ctx: MetalContext, rows: Int, nonNull: Int, initialSlots: Int? = nil,
                       build: (_ slots: MetalArrowBuffer, _ slotCount: Int, _ sampleMask: Int,
                               _ maxProbe: Int, _ slotOf: MetalArrowBuffer?,
                               _ firstOfSlot: MetalArrowBuffer?) throws -> Bool) throws -> HashGroups {
        // Largest table we would ever build: at least two slots per non-null row, so an insert into it
        // always finds a free slot and the last attempt cannot fail.
        var cap = 1024
        while cap < 2 * nonNull { cap <<= 1 }

        var estimate = nonNull
        if initialSlots != nil {
            estimate = 0
        } else if nonNull > 1 << 17 {
            // Below that threshold the table is small either way and the estimate is not worth a pass.
            let sampleBits = 6
            var pilot = 1024
            while pilot < 4 * (nonNull >> sampleBits) { pilot <<= 1 }
            pilot = Swift.min(pilot, 1 << 21)
            let slots = try MetalArrowBuffer.allocate(byteCount: pilot * 4, zeroed: false, context: ctx)
            let failed = try build(slots, pilot, (1 << sampleBits) - 1, 64, nil, nil)
            let sampled = try failed ? 0 : occupancy(ctx: ctx, slots: slots, count: pilot).total
            // A pilot that ran out of probe budget is one whose slice was denser than the table it was
            // given; size for the worst case rather than trust a truncated count.
            estimate = sampled == 0 ? nonNull : Swift.min(nonNull, sampled << sampleBits)
            ctx.retainUntilFlush(slots)
        }

        // Three slots per estimated distinct value keeps the load factor near 0.3, where linear probing
        // is still one probe for almost every row.
        var slotCount = initialSlots ?? 1024
        while slotCount < 3 * estimate && slotCount < cap { slotCount <<= 1 }
        slotCount = Swift.min(slotCount, cap)

        let slotOf = try MetalArrowBuffer.allocate(byteCount: Swift.max(rows, 1) * 4, zeroed: false, context: ctx)
        var slots: MetalArrowBuffer
        var firstOfSlot: MetalArrowBuffer
        while true {
            let last = slotCount >= cap
            slots = try MetalArrowBuffer.allocate(byteCount: slotCount * 4, zeroed: false, context: ctx)
            firstOfSlot = try MetalArrowBuffer.allocate(byteCount: slotCount * 4, zeroed: false, context: ctx)
            if try !build(slots, slotCount, 0, last ? Int(UInt32.max) : 96, slotOf, firstOfSlot) { break }
            // The estimate was low and the probe chains grew past the budget. Retry with a bigger table;
            // the last attempt has two slots per row and no budget, so this terminates.
            slotCount = Swift.min(slotCount << 3, cap)
        }
        ctx.retainUntilFlush(slots); ctx.retainUntilFlush(firstOfSlot); ctx.retainUntilFlush(slotOf)

        let (cum, groupCount) = try occupancy(ctx: ctx, slots: slots, count: slotCount)
        let firstSlotOrder = try compact(ctx: ctx, slots: slots, firstOfSlot: firstOfSlot, cum: cum,
                                         slotCount: slotCount, groupCount: groupCount)
        return HashGroups(slotOf: slotOf, cum: cum, groupCount: groupCount, firstSlotOrder: firstSlotOrder)
    }

    /// Clears a table and runs an insert kernel over it. The caller binds the key-specific arguments in
    /// `bind`, starting at buffer index 0 of the build kernel.
    static func runBuild(ctx: MetalContext, function: String, source: String, cacheKey: String,
                         slots: MetalArrowBuffer, slotCount: Int, rows: Int,
                         slotOf: MetalArrowBuffer?, firstOfSlot: MetalArrowBuffer?,
                         bind: (MTLComputeCommandEncoder, MetalArrowBuffer) throws -> Void) throws -> Bool {
        let errorFlag = try MetalArrowBuffer.allocate(byteCount: 4, context: ctx)
        let fillPSO = try pso(ctx, "ht_fill")
        let buildPSO = try ctx.pipeline(source: source, function: function, cacheKey: cacheKey)
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
            try bind(enc, errorFlag)
            Dispatch.dispatch1D(enc, buildPSO, count: rows)
        }
        try ctx.syncPoint()
        return withExtendedLifetime(errorFlag) { errorFlag.typed(UInt32.self)[0] != 0 }
    }

    /// Marks the occupied slots and scans the marks: `cum[s] - 1` is the rank of slot `s` among the
    /// occupied ones, and the last entry is how many groups there are.
    static func occupancy(ctx: MetalContext, slots: MetalArrowBuffer,
                          count: Int) throws -> (cum: MetalArray<Int32>, total: Int) {
        let marks = try MetalArrowBuffer.allocate(byteCount: count * 4, zeroed: false, context: ctx)
        let p = try pso(ctx, "ht_mark")
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
    static func compact(ctx: MetalContext, slots: MetalArrowBuffer, firstOfSlot: MetalArrowBuffer,
                        cum: MetalArray<Int32>, slotCount: Int, groupCount: Int) throws -> MetalArray<Int32> {
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(groupCount, 1) * 4, zeroed: false, context: ctx)
        if groupCount > 0 {
            let p = try pso(ctx, "ht_compact")
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

    /// One dense id per row: `relabel[rank[slot]]`, with null rows given `nullId`.
    static func writeIds(ctx: MetalContext, rows: Int, validity: MetalArrowBuffer?, fallbackBuffer: MetalArrowBuffer,
                         groups: HashGroups, relabel: MetalArray<Int32>, nullId: Int) throws -> MetalArrowBuffer {
        let ids = try MetalArrowBuffer.allocate(byteCount: Swift.max(rows, 1) * 4, zeroed: false, context: ctx)
        guard rows > 0 else { return ids }
        let p = try pso(ctx, "ht_ids")
        let v = validity ?? fallbackBuffer
        try ctx.run { enc in
            enc.setComputePipelineState(p)
            enc.setBuffer(groups.slotOf.mtl, offset: groups.slotOf.offset, index: 0)
            enc.setBuffer(v.mtl, offset: v.offset, index: 1)
            Dispatch.setUInt(enc, validity == nil ? 0 : 1, index: 2)
            Dispatch.setUInt(enc, nullId, index: 3)
            enc.setBuffer(groups.cum.values.mtl, offset: groups.cum.values.offset, index: 4)
            enc.setBuffer(relabel.values.mtl, offset: relabel.values.offset, index: 5)
            Dispatch.setLength(enc, rows, nil, index: 6)
            enc.setBuffer(ids.mtl, offset: ids.offset, index: 7)
            Dispatch.dispatch1D(enc, p, count: rows)
        }
        ctx.retainUntilFlush(groups.cum); ctx.retainUntilFlush(relabel); ctx.retainUntilFlush(groups.slotOf)
        try ctx.syncPoint()
        return ids
    }
}

// MARK: - Primitive columns

/// The distinct values of a primitive column, found with the hash table rather than a sort.
struct HashDistinct<T: ArrowPrimitive> {
    /// The distinct non-null values, **ascending** — Arrow's `unique` order, and the order the
    /// dictionary codes index into.
    let values: MetalArray<T>
    /// Lowest row index of each distinct value, in the same order as `values`. Ordering the groups by
    /// this instead gives first-appearance order without touching the rows again.
    let firstRows: MetalArray<Int32>
    /// Slot-order rank -> position in `values`.
    let relabel: MetalArray<Int32>
    let groups: HashTable.HashGroups
    var count: Int { groups.groupCount }
}

extension MetalArray {

    /// Whether the hash table beats the sort for this column.
    ///
    /// The estimate is not free (one pass over the keys), so the rule is by shape rather than by
    /// cardinality: below `1 << 16` rows the sort is a handful of small passes and the table's extra
    /// round trips do not pay for themselves, and above it the table has won at every cardinality
    /// measured — including the worst case of every row distinct, where it matches the sort.
    static func prefersHashTable(rows: Int) -> Bool { rows >= 1 << 16 && !HashTable.disabled }

    /// The 64-bit grouping key of every element. Integers widen; floats are normalised so that bit
    /// equality means Arrow value equality (one NaN, `-0.0 == 0.0`).
    func hashKeys() throws -> MetalArray<UInt64> {
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(length, 1) * 8, zeroed: false, context: ctx)
        if length > 0 {
            let (mslType, expr) = Self.keyMapping
            let src = HashTableSource.keySource(T: mslType, expr: expr)
            let p = try ctx.pipeline(source: src, function: "ht_key", cacheKey: "hashtable/key/\(mslType)/\(T.mslType)")
            try ctx.run { enc in
                enc.setComputePipelineState(p)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                Dispatch.setLength(enc, length, nil, index: 1)
                enc.setBuffer(out.mtl, offset: out.offset, index: 2)
                Dispatch.dispatch1D(enc, p, count: length)
            }
            ctx.retainUntilFlush(self)
        }
        return MetalArray<UInt64>(length: length, nullCount: nullCount, validity: validity, values: out, context: ctx)
    }

    /// How this element type becomes a 64-bit key: the MSL type its buffer is read as, and the
    /// expression that maps a value `v` to the key.
    static var keyMapping: (mslType: String, expr: String) {
        if T.isFloatingPoint {
            // Read the bits, not the float: Apple GPUs flush subnormals in float arithmetic, and Arrow
            // equality wants every NaN to be one value and -0 to equal +0.
            if T.byteWidth == 4 {
                return ("uint", "(ulong)(((v & 0x7FFFFFFFu) > 0x7F800000u) ? 0x7FC00000u : (((v & 0x7FFFFFFFu) == 0u) ? 0u : v))")
            }
            precondition(T.byteWidth == 8, "no 64-bit key mapping for a \(T.byteWidth)-byte float")
            return ("ulong", "((v & 0x7FFFFFFFFFFFFFFFUL) > 0x7FF0000000000000UL) ? 0x7FF8000000000000UL : (((v & 0x7FFFFFFFFFFFFFFFUL) == 0UL) ? 0UL : v)")
        }
        // Sign-extend signed integers and zero-extend unsigned ones: both are injective into 64 bits.
        return (T.mslType, T.minValue < 0 as T ? "(ulong)(long)v" : "(ulong)v")
    }

    /// Builds the table over this column's keys and ranks it, or nil when nothing is non-null.
    func hashGroups(initialSlots: Int? = nil) throws -> (HashTable.HashGroups, keys: MetalArray<UInt64>)? {
        let ctx = context
        let nonNull = length - nullCount
        guard nonNull > 0 else { return nil }
        try Dispatch.checkLength(length)
        guard length <= Int(Int32.max) else {
            throw ArrowMetalError.invalidArrowArray("hash table: columns above 2^31 rows are not supported")
        }
        let keys = try hashKeys()
        let v = validity ?? keys.values
        let hasValidity = validity == nil ? 0 : 1
        let g = try HashTable.groups(ctx: ctx, rows: length, nonNull: nonNull, initialSlots: initialSlots) {
            slots, slotCount, sampleMask, maxProbe, slotOf, firstOfSlot in
            try HashTable.runBuild(ctx: ctx, function: "ht_build", source: HashTableSource.shared,
                                   cacheKey: "hashtable/ht_build", slots: slots, slotCount: slotCount,
                                   rows: length, slotOf: slotOf, firstOfSlot: firstOfSlot) { enc, errorFlag in
                enc.setBuffer(keys.values.mtl, offset: keys.values.offset, index: 0)
                enc.setBuffer(v.mtl, offset: v.offset, index: 1)
                Dispatch.setUInt(enc, hasValidity | (slotOf != nil ? 4 : 0), index: 2)
                Dispatch.setLength(enc, length, nil, index: 3)
                Dispatch.setUInt(enc, slotCount - 1, index: 4)
                Dispatch.setUInt(enc, sampleMask, index: 5)
                Dispatch.setUInt(enc, maxProbe, index: 6)
                enc.setBuffer(slots.mtl, offset: slots.offset, index: 7)
                enc.setBuffer((slotOf ?? errorFlag).mtl, offset: (slotOf ?? errorFlag).offset, index: 8)
                enc.setBuffer((firstOfSlot ?? errorFlag).mtl, offset: (firstOfSlot ?? errorFlag).offset, index: 9)
                enc.setBuffer(errorFlag.mtl, offset: errorFlag.offset, index: 10)
            }
        }
        ctx.retainUntilFlush(keys)
        return (g, keys)
    }

    /// The distinct values of this column, ascending, found by the hash table. Nil when nothing is
    /// non-null (the caller's empty case) — never when the table simply is not worth it, which
    /// `prefersHashTable` decides.
    func hashDistinct(initialSlots: Int? = nil) throws -> HashDistinct<T>? {
        guard let (g, _) = try hashGroups(initialSlots: initialSlots) else { return nil }
        // One representative row per group (the lowest, so `-0.0` beats a later `0.0` exactly as the
        // stable sort path would), then the ascending order of those K values.
        let reps = try take(g.firstSlotOrder)
        let repValues = MetalArray<T>(length: reps.length, nullCount: 0, validity: nil,
                                      values: reps.values, context: context)
        let perm = try repValues.argsort()
        let relabel = try perm.argsort()
        let values = try repValues.take(perm)
        let clean = values.validity == nil ? values
            : MetalArray<T>(length: values.length, nullCount: 0, validity: nil, values: values.values, context: context)
        return HashDistinct(values: clean, firstRows: try g.firstSlotOrder.take(perm), relabel: relabel, groups: g)
    }

    /// Dense codes into `distinct.values`, one per row, null where the row is null.
    func hashCodes(_ distinct: HashDistinct<T>) throws -> MetalArray<Int32> {
        let ids = try HashTable.writeIds(ctx: context, rows: length, validity: validity, fallbackBuffer: values,
                                         groups: distinct.groups, relabel: distinct.relabel, nullId: 0)
        return MetalArray<Int32>(length: length, nullCount: nullCount, validity: validity, values: ids, context: context)
    }

    /// Rows per distinct value, in the order of `distinct.values`: the dense-key group-by counting rows,
    /// with null rows sent to a group of their own that is then dropped.
    func hashCounts(_ distinct: HashDistinct<T>) throws -> MetalArray<Int64> {
        let k = distinct.count
        let ids = try HashTable.writeIds(ctx: context, rows: length, validity: validity, fallbackBuffer: values,
                                         groups: distinct.groups, relabel: distinct.relabel, nullId: k)
        let dense = MetalArray<Int32>(length: length, nullCount: 0, validity: nil, values: ids, context: context)
        let counts = try GroupBy(keys: dense, keyCount: k + (nullCount > 0 ? 1 : 0)).count()
        return counts.length == k ? counts : try counts.slice(offset: 0, length: k)
    }
}
