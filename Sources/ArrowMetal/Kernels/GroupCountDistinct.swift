import Foundation
import Metal

// `hash_count_distinct` as a GPU hash **set** over the (group, value) pair.
//
// The previous shape of this aggregate was five whole-column passes: dictionary-encode the values,
// widen the keys and the codes to int64, pack them into `key * uniqueCount + code`, drop the nulls,
// run `unique()` over the packed column, divide the survivors back out and count them per key. Each
// step materialised another 8-byte column, and the dictionary encoding is itself a hash build plus an
// argsort of the distinct values — an ordering nobody downstream asked for. At 10 million rows that
// came to 124 ms whatever the cardinality, against Polars' 25 to 77 ms.
//
// What the answer actually needs is the *size* of the set of distinct pairs, per group. So:
//
// 1. **Insert.** One open-addressing table over the pair. A slot holds `row + 1`, exactly as the
//    generic table does, so every atomic stays 32-bit and the key of an occupied slot is read back out
//    of the two input columns rather than cached beside the slot. Equality is "same group id **and**
//    same 64-bit value key", so a hash collision can never merge two distinct pairs. The value key is
//    the same normalisation `unique()` and `dictionaryEncode()` use — integers sign-extend, every NaN
//    collapses to one pattern and -0 becomes +0 — so the distinct count is unchanged, `-0.0` among
//    `0.0`s included.
// 2. **Count.** One pass over the slots: an occupied slot is one distinct pair, and its group is the
//    group of the row it holds. An atomic increment per occupied slot lands in the output.
//
// Two passes over the rows (the cardinality pilot makes it three) and one over the table, with no
// column materialised in between. Rows with a null key, a null value, or a key outside `[0, keyCount)`
// are never inserted, which is what the packed-arithmetic path achieved by letting the null propagate.
//
// The counts are accumulated straight into the int64 output buffer: a distinct count cannot exceed the
// row count and therefore never reaches 2^32, so the atomic touches the low half of each int64 and the
// high half stays zero. That is little-endian by construction, which every platform this package runs
// on is.

extension GroupBy {

    /// Arrow `hash_count_distinct` over a hash set of the (key, value) pairs. Keys with no valid value
    /// count zero, as Arrow's count aggregates do.
    func countDistinctHashed<T: ArrowPrimitive>(_ values: MetalArray<T>) throws -> MetalArray<Int64>? {
        guard !HashTable.disabled else { return nil }
        let ctx = values.context
        let rows = values.length
        try Dispatch.checkLength(rows)
        guard rows <= Int(Int32.max) else { return nil }
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(keyCount, 1) * 8, context: ctx)
        // Rows that cannot be inserted: a null value, a null key, or a key outside the range. The last
        // two are not counted here, so this is an upper bound on the insertions — which is all the
        // table sizing needs, since it only has to guarantee two slots per inserted row in the end.
        let insertable = rows - Swift.max(values.nullCount, keys.nullCount)
        guard rows > 0, keyCount > 0, insertable > 0 else {
            return MetalArray<Int64>(length: keyCount, nullCount: 0, validity: nil, values: out, context: ctx)
        }

        let (mslType, expr) = MetalArray<T>.keyMapping
        let source = GroupCountDistinctSource.build(readAs: mslType, key: expr, keyType: K.mslType)
        let cacheKey = "groupcountdistinct/\(mslType)/\(T.mslType)/\(K.mslType)"
        let vValidity = values.validity
        let kValidity = keys.validity
        let flags = (vValidity == nil ? 0 : 1) | (kValidity == nil ? 0 : 2)
        let table = try HashTable.sizedBuild(ctx: ctx, rows: rows, nonNull: insertable,
                                             wantSlotOf: false, wantFirstOfSlot: false) {
            slots, slotCount, sampleMask, maxProbe, _, _ in
            try HashTable.runBuild(ctx: ctx, function: "hcd_build", source: source, cacheKey: cacheKey,
                                   slots: slots, slotCount: slotCount, rows: rows,
                                   slotOf: nil, firstOfSlot: nil) { enc, errorFlag in
                enc.setBuffer(values.values.mtl, offset: values.values.offset, index: 0)
                let vv = vValidity ?? values.values
                enc.setBuffer(vv.mtl, offset: vv.offset, index: 1)
                enc.setBuffer(self.keys.values.mtl, offset: self.keys.values.offset, index: 2)
                let kv = kValidity ?? self.keys.values
                enc.setBuffer(kv.mtl, offset: kv.offset, index: 3)
                Dispatch.setUInt(enc, flags, index: 4)
                Dispatch.setUInt(enc, self.keyCount, index: 5)
                Dispatch.setLength(enc, rows, nil, index: 6)
                Dispatch.setUInt(enc, slotCount - 1, index: 7)
                Dispatch.setUInt(enc, sampleMask, index: 8)
                Dispatch.setUInt(enc, maxProbe, index: 9)
                enc.setBuffer(slots.mtl, offset: slots.offset, index: 10)
                enc.setBuffer(errorFlag.mtl, offset: errorFlag.offset, index: 11)
            }
        }
        ctx.retainUntilFlush(values); ctx.retainUntilFlush(keys)

        let countPSO = try ctx.pipeline(source: source, function: "hcd_count", cacheKey: cacheKey + "/count")
        try ctx.run { enc in
            enc.setComputePipelineState(countPSO)
            enc.setBuffer(table.slots.mtl, offset: table.slots.offset, index: 0)
            enc.setBuffer(self.keys.values.mtl, offset: self.keys.values.offset, index: 1)
            Dispatch.setLength(enc, table.slotCount, nil, index: 2)
            enc.setBuffer(out.mtl, offset: out.offset, index: 3)
            Dispatch.dispatch1D(enc, countPSO, count: table.slotCount)
        }
        ctx.retainUntilFlush(table.slots); ctx.retainUntilFlush(keys)
        try ctx.syncPoint()
        return MetalArray<Int64>(length: keyCount, nullCount: 0, validity: nil, values: out, context: ctx)
    }
}

/// MSL for the pair table. One source per element type, because the 64-bit value key is read straight
/// out of the values buffer instead of being materialised into a column of its own.
enum GroupCountDistinctSource {
    static func build(readAs T: String, key expr: String, keyType K: String) -> String { KernelSource.prelude + """

    #define HCD_VALUE_VALIDITY 1u
    #define HCD_KEY_VALIDITY   2u

    // splitmix64's finalizer, as in Kernels/HashTableSource.swift.
    inline ulong hcd_mix(ulong h) {
        h ^= h >> 33; h *= 0xff51afd7ed558ccdUL;
        h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53UL;
        h ^= h >> 33;
        return h;
    }
    inline uint hcd_slot(ulong h, uint mask) { return ((uint)h ^ (uint)(h >> 32)) & mask; }

    // The value half of the pair key: the same normalisation `unique()` and `dictionaryEncode()` apply,
    // so bit equality here is Arrow value equality (one NaN, -0 == 0).
    inline ulong hcd_key(device const \(T)* vals, uint i) {
        \(T) v = vals[i];
        return \(expr);
    }

    // The pair's slot. Mixing the group id in with the golden-ratio constant keeps two rows of the same
    // group, or the same value in different groups, off each other's probe chains.
    inline ulong hcd_hash(ulong k, uint g) {
        return hcd_mix(k ^ ((ulong)g * 0x9E3779B97F4A7C15UL));
    }

    // Inserts one row per thread into the set of (group, value) pairs. `sampleMask` selects a fraction
    // of the hash space for the cardinality pilot, and `maxProbe` bounds the linear-probe walk, both
    // exactly as `ht_build` does — running out of budget means the table was too small, and the host
    // retries with a bigger one.
    kernel void hcd_build(device const \(T)* vals [[buffer(0)]], device const uchar* vValidity [[buffer(1)]],
                          device const \(K)* gids [[buffer(2)]], device const uchar* kValidity [[buffer(3)]],
                          constant uint& flags [[buffer(4)]], constant uint& keyCount [[buffer(5)]],
                          device const uint* nPtr [[buffer(6)]], constant uint& mask [[buffer(7)]],
                          constant uint& sampleMask [[buffer(8)]], constant uint& maxProbe [[buffer(9)]],
                          device atomic_uint* slots [[buffer(10)]], device atomic_uint* errorFlag [[buffer(11)]],
                          uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if ((flags & HCD_VALUE_VALIDITY) != 0u && !bit_get(vValidity, i)) return;
        if ((flags & HCD_KEY_VALIDITY) != 0u && !bit_get(kValidity, i)) return;
        long g = (long)gids[i];
        if (g < 0 || g >= (long)keyCount) return;
        ulong k = hcd_key(vals, i);
        ulong h = hcd_hash(k, (uint)g);
        if (sampleMask != 0u && ((((uint)(h >> 40)) & sampleMask) != 0u)) return;
        uint s = hcd_slot(h, mask);
        uint budget = maxProbe;
        bool placed = false;
        while (budget > 0u) {
            budget--;
            uint head = atomic_load_explicit(&slots[s], memory_order_relaxed);
            if (head == 0u) {
                uint expected = 0u;
                if (atomic_compare_exchange_weak_explicit(&slots[s], &expected, i + 1u,
                                                          memory_order_relaxed, memory_order_relaxed)) {
                    placed = true; break;
                }
                head = expected;                    // lost the race, or a spurious weak failure
                if (head == 0u) continue;           // spurious: read the same slot again
            }
            uint r = head - 1u;
            // Both halves of the pair, read back out of the input columns: no cached copy beside the
            // slot, so nothing can be seen before its writer published it.
            if ((long)gids[r] == g && hcd_key(vals, r) == k) { placed = true; break; }
            s = (s + 1u) & mask;
        }
        if (!placed) atomic_store_explicit(errorFlag, 1u, memory_order_relaxed);
    }

    // One occupied slot is one distinct pair; its group is the group of the row it holds. `out` is the
    // int64 result buffer and a distinct count never reaches 2^32, so the increment lands in the low
    // half of the int64 and the high half stays the zero it was allocated with.
    kernel void hcd_count(device const uint* slots [[buffer(0)]], device const \(K)* gids [[buffer(1)]],
                          device const uint* nPtr [[buffer(2)]], device atomic_uint* out [[buffer(3)]],
                          uint s [[thread_position_in_grid]]) {
        if (s >= *nPtr) return;
        uint head = slots[s];
        if (head == 0u) return;
        atomic_fetch_add_explicit(&out[(uint)gids[head - 1u] * 2u], 1u, memory_order_relaxed);
    }
    """ }
}
