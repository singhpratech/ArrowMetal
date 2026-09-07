import Foundation
import Metal

// The streaming group-by's global table, kept on the GPU for the whole scan.
//
// The host table it replaces is correct and, for a *large* number of groups, slow for a reason that
// has nothing to do with the GPU: a ten-million-group scan folds roughly a million rows per batch
// into a hash map, five hundred and seventy times, and every one of those probes is a DRAM round
// trip on one core. Sharding it across eight threads (which the host table does) moves it; keeping
// the table on the GPU removes it. The merge stage's cost drops to zero and the per-batch readback
// of a million keys, a million sums and a million counts disappears with it.
//
// ## Publishing a 64-bit key with 32-bit atomics
//
// Metal has no 64-bit atomics (verified on an M4 Max: `atomic_ulong` has no `compare_exchange`,
// `fetch_add` or `fetch_max`) and only guarantees `memory_order_relaxed` on the 32-bit ones. So the
// usual "claim the slot, then publish the key beside it" handshake is not safe: nothing orders the
// key's store against the claim, and a reader that sees a claimed slot may read a key that is not
// there yet. `Kernels/HashTable.swift` sidesteps that by storing `row + 1` in the atomic and reading
// the key out of an array nothing writes; a *persistent* table cannot, because its keys have to live
// in the table.
//
// This table publishes the key **through** the atomics instead. A slot is three 32-bit words holding
// the key's 64 bits split 22 / 21 / 21, each stored as `field + 1` so that **0 means "not written"**
// and every one of the 2^64 keys maps to three non-zero words. There is no reserved sentinel value
// and therefore no key a caller may not use. Writing a word is `compare_exchange(0 -> field + 1)`,
// so:
//
// * a word only ever goes from 0 to its final value, and never changes again;
// * a thread accepts a slot only when **all three** words equal its own fields, and since those
//   words are immutable once set, that decision is permanent and every thread agrees with it;
// * a thread that mismatches any word walks on, which is what linear probing wants it to do.
//
// The awkward case a naive scheme gets wrong — two threads with the same key racing on one empty
// slot — resolves without either a spin (which can deadlock a divergent SIMD group) or a duplicate
// slot: the loser's compare-exchange fails *with the winner's value*, which is its own field, so it
// reads the failure as a match and stops on the same slot. And the case where they differ resolves
// the same way, because the loser sees a value that is not its field and walks on.
//
// A thread that wins word 0 and then loses word 1 to another key leaves behind a slot that some
// other thread completes — the last thread to win a word is still inside its own probe step and goes
// on to write the rest — so a slot is never left half written when a dispatch ends.
//
// ## Two ways in
//
// 1. **Distinct keys** (`fold`) — the batch's per-group results are folded in one insert dispatch
//    plus one accumulate. Distinct keys mean distinct slots, so the accumulate needs no atomics and
//    the float64 sum can use the correctly-rounded software adder (`d_add`) that Metal's missing
//    64-bit atomics would otherwise rule out.
// 2. **Rows** (`foldRows`, `denseIdsForRows`) — one thread per *row*, which is what removes the
//    per-batch dense encoding (a radix sort of the key column) from a high-cardinality scan. The
//    insert is the same kernel; what follows it is either
//    * an **atomic** accumulate, one thread per row, into 64-bit counters built from two 32-bit
//      atomic adds with a carry — exact and order-independent, so `count` and integer `sum` come out
//      bit-identical to the distinct path; or
//    * a **dense id** pass, which stamps one batch-local id onto each slot the batch touched and
//      hands the caller `GroupBy`-shaped ids. That is for float64 sums, which no emulation of a
//      64-bit atomic can round correctly, and it still skips the sort: the resident table itself is
//      the dictionary.
//
// Growth doubles the table and rehashes on the GPU under the same claim rule.

enum StreamGroupTableSource {
    static let msl: String = KernelSource.prelude + DoubleMath.msl + """

    #define SGT_NONE 0xFFFFFFFFu
    #define SGT_W 3u

    inline ulong sgt_mix(ulong h) {
        h ^= h >> 33; h *= 0xff51afd7ed558ccdUL;
        h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53UL;
        h ^= h >> 33;
        return h;
    }
    inline uint sgt_home(long k, uint mask) {
        ulong h = sgt_mix((ulong)k);
        return ((uint)h ^ (uint)(h >> 32)) & mask;
    }

    // The key's 64 bits as three non-zero words: 22 + 21 + 21, each stored as `field + 1`.
    inline uint3 sgt_fields(long k) {
        ulong u = (ulong)k;
        return uint3((uint)(u & 0x3FFFFFUL) + 1u,
                     (uint)((u >> 22) & 0x1FFFFFUL) + 1u,
                     (uint)((u >> 43) & 0x1FFFFFUL) + 1u);
    }
    inline long sgt_key(uint w0, uint w1, uint w2) {
        ulong u = (ulong)(w0 - 1u) | ((ulong)(w1 - 1u) << 22) | ((ulong)(w2 - 1u) << 43);
        return (long)u;
    }

    // 0 the word holds a different field, 1 this thread wrote it, 2 it already held this field.
    inline uint sgt_claim(device atomic_uint* w, uint want) {
        uint cur = atomic_load_explicit(w, memory_order_relaxed);
        if (cur != 0u) return cur == want ? 2u : 0u;
        uint expected = 0u;
        while (!atomic_compare_exchange_weak_explicit(w, &expected, want,
                                                      memory_order_relaxed, memory_order_relaxed)) {
            // A weak exchange can fail spuriously while the slot is still empty; only a value that is
            // really there ends the attempt.
            if (expected != 0u) return expected == want ? 2u : 0u;
        }
        return 1u;
    }

    // The slot this key owns, claiming an empty one if it has none yet. `isNew` is true for the one
    // thread that turned an empty slot into this key's slot.
    inline uint sgt_insert_slot(device atomic_uint* W, uint mask, uint budget, long k, thread bool& isNew) {
        uint3 f = sgt_fields(k);
        uint s = sgt_home(k, mask);
        for (uint p = 0u; p < budget; ++p) {
            uint r0 = sgt_claim(&W[s * SGT_W + 0u], f.x);
            if (r0 != 0u) {
                if (sgt_claim(&W[s * SGT_W + 1u], f.y) != 0u) {
                    if (sgt_claim(&W[s * SGT_W + 2u], f.z) != 0u) { isNew = (r0 == 1u); return s; }
                }
            }
            s = (s + 1u) & mask;
        }
        return SGT_NONE;
    }

    // Read-only probe, for a phase in which nothing writes the table.
    inline uint sgt_find_slot(device const uint* W, uint mask, uint budget, long k) {
        uint3 f = sgt_fields(k);
        uint s = sgt_home(k, mask);
        for (uint p = 0u; p < budget; ++p) {
            uint a = W[s * SGT_W + 0u];
            if (a == 0u) return SGT_NONE;
            if (a == f.x && W[s * SGT_W + 1u] == f.y && W[s * SGT_W + 2u] == f.z) return s;
            s = (s + 1u) & mask;
        }
        return SGT_NONE;
    }

    // A 64-bit atomic add out of two 32-bit ones. The high word takes the carry out of the low word,
    // which is exactly two's-complement 64-bit addition, so it wraps like `Int64` and — being plain
    // integer addition — is independent of the order the rows arrive in.
    inline void sgt_add64(device atomic_uint* words, uint slot, ulong v) {
        uint lo = (uint)(v & 0xFFFFFFFFUL);
        uint hi = (uint)(v >> 32);
        if (lo != 0u) {
            uint old = atomic_fetch_add_explicit(&words[slot * 2u], lo, memory_order_relaxed);
            if (old + lo < old) hi += 1u;
        }
        if (hi != 0u) atomic_fetch_add_explicit(&words[slot * 2u + 1u], hi, memory_order_relaxed);
    }

    // Insert phase. One thread per element of `keys` — a batch's distinct group keys on the fold
    // path, every row of the batch on the row path.
    kernel void sgt_insert(device const long* keys [[buffer(0)]],
                           device const uchar* keyValidity [[buffer(1)]],
                           constant uint& hasValidity [[buffer(2)]],
                           device const uint* nPtr [[buffer(3)]],
                           device atomic_uint* W [[buffer(4)]],
                           constant uint& mask [[buffer(5)]],
                           constant uint& nullSlot [[buffer(6)]],
                           constant uint& budget [[buffer(7)]],
                           device uint* slotOut [[buffer(8)]],
                           device atomic_uint* used [[buffer(9)]],
                           device atomic_uint* overflow [[buffer(10)]],
                           constant uint& denseMode [[buffer(11)]],
                           constant uint& stampId [[buffer(12)]],
                           device atomic_uint* stamp [[buffer(13)]],
                           device uint* denseOfSlot [[buffer(14)]],
                           device uint* slotOfDense [[buffer(15)]],
                           device atomic_uint* denseCounter [[buffer(16)]],
                           uint j [[thread_position_in_grid]]) {
        if (j >= *nPtr) return;
        uint s;
        if (hasValidity != 0u && !bit_get(keyValidity, j)) {
            s = nullSlot;                       // every null key shares one group, one slot past the end
        } else {
            bool isNew = false;
            s = sgt_insert_slot(W, mask, budget, keys[j], isNew);
            if (s == SGT_NONE) {
                atomic_store_explicit(overflow, 1u, memory_order_relaxed);
                slotOut[j] = SGT_NONE;
                return;
            }
            if (isNew) atomic_fetch_add_explicit(used, 1u, memory_order_relaxed);
        }
        slotOut[j] = s;
        if (denseMode != 0u) {
            // Exactly one thread per slot per batch swaps in this batch's stamp, and it is the one
            // that hands the slot its batch-local dense id.
            uint prev = atomic_exchange_explicit(&stamp[s], stampId, memory_order_relaxed);
            if (prev != stampId) {
                uint d = atomic_fetch_add_explicit(denseCounter, 1u, memory_order_relaxed);
                denseOfSlot[s] = d;
                slotOfDense[d] = s;
            }
        }
    }

    // Reads back the dense id the insert dispatch stamped onto each row's slot. A separate dispatch,
    // so `denseOfSlot` is a settled array by the time anything reads it.
    kernel void sgt_dense_of_row(device const uint* slotOfRow [[buffer(0)]],
                                 device const uint* denseOfSlot [[buffer(1)]],
                                 device const uint* nPtr [[buffer(2)]],
                                 device int* out [[buffer(3)]],
                                 uint j [[thread_position_in_grid]]) {
        if (j >= *nPtr) return;
        uint s = slotOfRow[j];
        out[j] = (s == SGT_NONE) ? 0 : (int)denseOfSlot[s];
    }

    // Accumulate, one thread per batch group: distinct keys mean distinct slots, so no atomics, and
    // the float64 sum uses the correctly-rounded software adder.
    // kind: 0 count only, 1 int64 sum, 2 uint64 sum, 3 float64 sum.
    kernel void sgt_accumulate(device const uint* slot [[buffer(0)]],
                               device const ulong* vals [[buffer(1)]],
                               device const uchar* valValidity [[buffer(2)]],
                               constant uint& hasValidity [[buffer(3)]],
                               device const long* counts [[buffer(4)]],
                               constant uint& hasCounts [[buffer(5)]],
                               device ulong* sums [[buffer(6)]],
                               device long* cnts [[buffer(7)]],
                               constant uint& kind [[buffer(8)]],
                               device const uint* nPtr [[buffer(9)]],
                               uint j [[thread_position_in_grid]]) {
        if (j >= *nPtr) return;
        uint s = slot[j];
        if (s == SGT_NONE) return;
        if (hasCounts != 0u) cnts[s] += counts[j];
        if (kind == 0u) return;
        if (hasValidity != 0u && !bit_get(valValidity, j)) return;
        ulong v = vals[j];
        if (kind == 3u) sums[s] = d_add(sums[s], v);
        else if (kind == 1u) sums[s] = (ulong)((long)sums[s] + (long)v);
        else sums[s] += v;
    }

    // Accumulate, one thread per **row**, straight into the slot the insert phase gave it.
    // countMode: 0 none, 1 every row, 2 rows whose value is not null.
    // sumMode:   0 none, 1 add the 64-bit value.
    kernel void sgt_accumulate_rows(device const uint* slotOfRow [[buffer(0)]],
                                    device const ulong* vals [[buffer(1)]],
                                    device const uchar* valValidity [[buffer(2)]],
                                    constant uint& hasValidity [[buffer(3)]],
                                    constant uint& countMode [[buffer(4)]],
                                    constant uint& sumMode [[buffer(5)]],
                                    device atomic_uint* sums [[buffer(6)]],
                                    device atomic_uint* cnts [[buffer(7)]],
                                    device const uint* nPtr [[buffer(8)]],
                                    uint j [[thread_position_in_grid]]) {
        if (j >= *nPtr) return;
        uint s = slotOfRow[j];
        if (s == SGT_NONE) return;
        bool valid = (hasValidity == 0u) || bit_get(valValidity, j);
        if (countMode == 1u || (countMode == 2u && valid)) sgt_add64(cnts, s, 1UL);
        if (sumMode != 0u && valid) sgt_add64(sums, s, vals[j]);
    }

    // Growth. One thread per old slot; the keys being moved are distinct, so the claim rule holds.
    kernel void sgt_rehash(device const uint* oldWords [[buffer(0)]],
                           device const uint* nPtr [[buffer(1)]],
                           device atomic_uint* W [[buffer(2)]],
                           constant uint& mask [[buffer(3)]],
                           constant uint& budget [[buffer(4)]],
                           device uint* moved [[buffer(5)]],
                           device atomic_uint* overflow [[buffer(6)]],
                           uint j [[thread_position_in_grid]]) {
        if (j >= *nPtr) return;
        uint w0 = oldWords[j * SGT_W + 0u];
        if (w0 == 0u) { moved[j] = SGT_NONE; return; }
        long k = sgt_key(w0, oldWords[j * SGT_W + 1u], oldWords[j * SGT_W + 2u]);
        bool isNew = false;
        uint s = sgt_insert_slot(W, mask, budget, k, isNew);
        if (s == SGT_NONE) atomic_store_explicit(overflow, 1u, memory_order_relaxed);
        moved[j] = s;
    }

    // Moves one accumulator pair to its new slot after a rehash.
    kernel void sgt_move_acc(device const uint* moved [[buffer(0)]],
                             device const ulong* oldSums [[buffer(1)]],
                             device const long* oldCnts [[buffer(2)]],
                             device ulong* sums [[buffer(3)]],
                             device long* cnts [[buffer(4)]],
                             device const uint* nPtr [[buffer(5)]],
                             uint j [[thread_position_in_grid]]) {
        if (j >= *nPtr) return;
        uint s = moved[j];
        if (s == SGT_NONE) return;
        sums[s] = oldSums[j];
        cnts[s] = oldCnts[j];
    }

    // One group per **thread**, for the float64 sum of a batch with very many small groups.
    //
    // The segmented reduction this replaces gives each group a whole threadgroup: at a million groups
    // of one row that is 256 threads per row, and it is what a sparse key spends its time on. But its
    // answer has to be reproduced *bit for bit*, because a binary64 sum depends on the order it was
    // added in and the host table is checked against it.
    //
    // It is reproducible because the shape of that reduction is fixed. Lane `l` folds the run's rows
    // at positions l, l + 256, l + 512 … in order, and the lanes then combine in the butterfly
    // `shared[i] = combine(shared[i], shared[i + w])` for w = 128, 64 … 1, in which an empty
    // accumulator is replaced rather than added to. Two facts make that a serial loop:
    //
    // * every step with `w >= L` does nothing (the right-hand lane is always empty), so a run of `L`
    //   rows only ever uses the first `P` lanes, `P` the power of two at or above `L`;
    // * the butterfly over `P` lanes is the balanced tree whose leaves, left to right, are the lanes
    //   in **bit-reversed** order — so walking the lanes in that order and merging equal-rank partial
    //   results on a stack of at most nine entries rebuilds exactly the same tree.
    //
    // So the whole thing costs O(L) per group in one thread, with the same additions in the same
    // order as the threadgroup that used to do it.
    kernel void sgt_seg_sum_f64(device const uint* segStart [[buffer(0)]],
                                device const uint* segEnd [[buffer(1)]],
                                device const int* ord [[buffer(2)]],
                                device const ulong* vals [[buffer(3)]],
                                device const uchar* validity [[buffer(4)]],
                                device const uint* nPtr [[buffer(5)]],
                                constant uint& hasValidity [[buffer(6)]],
                                device ulong* out [[buffer(7)]],
                                device uchar* validBytes [[buffer(8)]],
                                device long* cnts [[buffer(9)]],
                                uint k [[thread_position_in_grid]]) {
        if (k >= *nPtr) return;
        uint s = segStart[k], e = segEnd[k];
        out[k] = 0ul; validBytes[k] = 0; cnts[k] = 0;
        if (e <= s) return;
        uint L = e - s;
        uint bits = 0u, P = 1u;
        while (P < L && P < 256u) { P <<= 1; bits += 1u; }

        ulong stackAcc[9];
        uint stackCnt[9], stackRank[9];
        uint top = 0u;
        for (uint t = 0u; t < P; ++t) {
            uint lane = 0u;                       // the low `bits` bits of t, reversed
            for (uint b = 0u; b < bits; ++b) lane |= ((t >> b) & 1u) << (bits - 1u - b);
            ulong acc = 0ul;
            uint cnt = 0u;
            for (uint i = s + lane; i < e; i += 256u) {
                uint row = (uint)ord[i];
                if (hasValidity != 0u && !bit_get(validity, row)) continue;
                ulong v = vals[row];
                acc = cnt ? d_add(acc, v) : v;
                cnt++;
            }
            uint rank = 0u;
            while (top > 0u && stackRank[top - 1u] == rank) {
                ulong a = stackAcc[top - 1u];
                uint ca = stackCnt[top - 1u];
                if (cnt == 0u) acc = a;
                else if (ca != 0u) acc = d_add(a, acc);
                cnt += ca;
                top -= 1u;
                rank += 1u;
            }
            stackAcc[top] = acc; stackCnt[top] = cnt; stackRank[top] = rank; top += 1u;
        }
        // `P` is a power of two and exactly `P` leaves went in, so the stack collapsed to one.
        if (stackCnt[0] == 0u) return;
        out[k] = stackAcc[0];
        validBytes[k] = 1;
        cnts[k] = (long)stackCnt[0];
    }

    // The finished table, slot by slot: the key and whether the slot is occupied.
    kernel void sgt_extract(device const uint* W [[buffer(0)]],
                            device const uint* nPtr [[buffer(1)]],
                            device long* keys [[buffer(2)]],
                            device uint* live [[buffer(3)]],
                            uint j [[thread_position_in_grid]]) {
        if (j >= *nPtr) return;
        uint w0 = W[j * SGT_W + 0u];
        live[j] = (w0 == 0u) ? 0u : 1u;
        keys[j] = (w0 == 0u) ? 0L : sgt_key(w0, W[j * SGT_W + 1u], W[j * SGT_W + 2u]);
    }
    """
}

/// The GPU-resident global table of a streaming group-by over one integer key.
///
/// Holds one slot per distinct key for the whole scan: the key (three 32-bit words), and per
/// aggregate a 64-bit sum and a 64-bit count. Ten million groups with one aggregate is
/// 10M x (12 key + 8 sum + 8 count) at a load factor of one half — about 560 MB of unified memory,
/// which is nothing on a machine that can hold the dataset's page cache.
final class StreamGroupTable {
    /// How a slot's sum is folded: matching the widths Arrow's `sum` produces.
    enum SumKind: Int { case none = 0, int = 1, uint = 2, double = 3 }

    /// What the row-level accumulate counts into a slot's count accumulator.
    enum CountMode: Int { case none = 0, allRows = 1, nonNullValues = 2 }

    /// One aggregate's row-level instructions: what to count, and what (if anything) to sum.
    struct RowAggregate {
        var count: CountMode = .none
        /// The value column, already widened to a 64-bit payload. It carries the validity `count`
        /// and `sum` both skip, so `count(col)` passes a column it does not sum.
        var column: AnyMetalArray?
        var sums = false
        var kind: SumKind = .none
    }

    let context: MetalContext
    let aggregateCount: Int
    /// Slot count, always a power of two. Slot `slots` (one past the end) is the null key's group.
    private(set) var slots: Int
    private var mask: UInt32 { UInt32(slots - 1) }
    /// Occupied slots.
    private(set) var used = 0

    /// `slots * 3` 32-bit words: the key of each slot, or zeros where the slot is empty.
    private var words: MetalArrowBuffer
    /// `aggregateCount` arrays of `slots + 1` sums, and the same of counts.
    private var sums: [MetalArrowBuffer] = []
    private var counts: [MetalArrowBuffer] = []
    private var usedCounter: MetalArrowBuffer
    private var overflow: MetalArrowBuffer
    private(set) var kinds: [SumKind]
    /// True once any row landed on the null key's group.
    private(set) var sawNullKey = false

    /// Grow when the table is this full. Linear probing degrades sharply past two thirds.
    var loadFactor = 0.55
    /// How many distinct keys the last insert added, which is the estimate the next one grows by.
    private var lastNewKeys = 0

    // The dense-id machinery of the row path, allocated the first time a caller asks for dense ids.
    private var stamp: MetalArrowBuffer?
    private var denseOfSlot: MetalArrowBuffer?
    private var denseCounter: MetalArrowBuffer
    private var stampId: UInt32 = 0

    /// How far a probe walks before the table is declared too full. A load factor of 0.55 puts the
    /// mean probe near two; a chain past this is a table that wants to be bigger, not a bad key.
    private var probeBudget: Int { Swift.min(slots, 4096) }

    init(context: MetalContext, aggregateCount: Int, initialSlots: Int = 1 << 16) throws {
        self.context = context
        self.aggregateCount = aggregateCount
        self.kinds = Array(repeating: .none, count: aggregateCount)
        var s = 1
        while s < Swift.max(initialSlots, 16) { s <<= 1 }
        self.slots = s
        self.words = try MetalArrowBuffer.allocate(byteCount: s * 12, context: context)
        self.usedCounter = try MetalArrowBuffer.allocate(byteCount: 4, context: context)
        self.overflow = try MetalArrowBuffer.allocate(byteCount: 4, context: context)
        self.denseCounter = try MetalArrowBuffer.allocate(byteCount: 4, context: context)
        for _ in 0..<aggregateCount {
            sums.append(try MetalArrowBuffer.allocate(byteCount: (s + 1) * 8, context: context))
            counts.append(try MetalArrowBuffer.allocate(byteCount: (s + 1) * 8, context: context))
        }
    }

    private func pipeline(_ function: String) throws -> MTLComputePipelineState {
        try Dispatch.pipeline(context, family: "streamgrouptable", source: StreamGroupTableSource.msl,
                              function: function, type: "shared")
    }

    // MARK: - The distinct-key path

    /// Folds one batch's per-group results into the table.
    ///
    /// `groupKeys` are the batch's **distinct** keys as int64 (null keys marked in its validity
    /// bitmap); `values[i]` and `groupCounts[i]` are aggregate `i`'s per-group sum and count, either
    /// of which may be nil.
    func fold(groupKeys: MetalArray<Int64>, values: [AnyMetalArray?], groupCounts: [MetalArray<Int64>?],
              kinds newKinds: [SumKind]) throws {
        let g = groupKeys.length
        guard g > 0 else { return }
        for (i, k) in newKinds.enumerated() where k != .none { kinds[i] = k }
        if groupKeys.nullCount > 0 { sawNullKey = true }

        let slot = try insert(groupKeys, dense: false, estimatedNew: g).slotOf
        try accumulate(slotOf: slot, count: g, values: values, groupCounts: groupCounts)
        context.retainUntilFlush(groupKeys)
        try context.syncPoint()
    }

    /// Folds one batch's per-group results into slots the caller already holds — the shape
    /// `denseIdsForRows` leaves behind, where dense id `d` belongs to slot `slotOfDense[d]`.
    func fold(slotOfDense: MetalArrowBuffer, groupCount: Int, values: [AnyMetalArray?],
              groupCounts: [MetalArray<Int64>?], kinds newKinds: [SumKind]) throws {
        guard groupCount > 0 else { return }
        for (i, k) in newKinds.enumerated() where k != .none { kinds[i] = k }
        try accumulate(slotOf: slotOfDense, count: groupCount, values: values, groupCounts: groupCounts)
        try context.syncPoint()
    }

    private func accumulate(slotOf slot: MetalArrowBuffer, count g: Int, values: [AnyMetalArray?],
                            groupCounts: [MetalArray<Int64>?]) throws {
        let acc = try pipeline("sgt_accumulate")
        for i in 0..<aggregateCount {
            let kind = kinds[i]
            let vals = values[i]
            let cnts = groupCounts[i]
            if kind == .none && cnts == nil { continue }
            let (valueBuffer, validity) = try sumBuffer(vals)
            try context.run { enc in
                enc.setComputePipelineState(acc)
                enc.setBuffer(slot.mtl, offset: slot.offset, index: 0)
                enc.setBuffer(valueBuffer.mtl, offset: valueBuffer.offset, index: 1)
                let v = validity ?? valueBuffer
                enc.setBuffer(v.mtl, offset: v.offset, index: 2)
                Dispatch.setUInt(enc, validity == nil ? 0 : 1, index: 3)
                let cb = cnts?.values ?? valueBuffer
                enc.setBuffer(cb.mtl, offset: cb.offset, index: 4)
                Dispatch.setUInt(enc, cnts == nil ? 0 : 1, index: 5)
                enc.setBuffer(sums[i].mtl, offset: sums[i].offset, index: 6)
                enc.setBuffer(counts[i].mtl, offset: counts[i].offset, index: 7)
                Dispatch.setUInt(enc, vals == nil ? 0 : kind.rawValue, index: 8)
                Dispatch.setLength(enc, g, nil, index: 9)
                Dispatch.dispatch1D(enc, acc, count: g)
            }
        }
    }

    // MARK: - The row-level path

    /// Folds a batch **row by row**: one thread per row inserts its key, one thread per row folds
    /// that row into its slot with 64-bit atomic adds.
    ///
    /// Exact and order-independent for counts and integer sums, which is every aggregate this path
    /// accepts; a float64 sum has to keep the distinct-key path, because Metal has no 64-bit atomic
    /// and no emulation of one rounds a binary64 addition correctly.
    func foldRows(keys: MetalArray<Int64>, aggregates: [RowAggregate]) throws {
        let n = keys.length
        guard n > 0 else { return }
        for (i, a) in aggregates.enumerated() where a.kind != .none { kinds[i] = a.kind }
        if keys.nullCount > 0 { sawNullKey = true }

        let slot = try insert(keys, dense: false, estimatedNew: estimateNew(rows: n)).slotOf
        let acc = try pipeline("sgt_accumulate_rows")
        for (i, a) in aggregates.enumerated() {
            if a.count == .none && !a.sums { continue }
            let (valueBuffer, validity) = try sumBuffer(a.column)
            try context.run { enc in
                enc.setComputePipelineState(acc)
                enc.setBuffer(slot.mtl, offset: slot.offset, index: 0)
                enc.setBuffer(valueBuffer.mtl, offset: valueBuffer.offset, index: 1)
                let v = validity ?? valueBuffer
                enc.setBuffer(v.mtl, offset: v.offset, index: 2)
                Dispatch.setUInt(enc, validity == nil ? 0 : 1, index: 3)
                Dispatch.setUInt(enc, a.count.rawValue, index: 4)
                Dispatch.setUInt(enc, a.sums ? 1 : 0, index: 5)
                enc.setBuffer(sums[i].mtl, offset: sums[i].offset, index: 6)
                enc.setBuffer(counts[i].mtl, offset: counts[i].offset, index: 7)
                Dispatch.setLength(enc, n, nil, index: 8)
                Dispatch.dispatch1D(enc, acc, count: n)
            }
        }
        context.retainUntilFlush(keys)
        try context.syncPoint()
    }

    /// One batch-local dense id per row, without sorting anything: the resident table *is* the
    /// dictionary, and the insert pass stamps this batch's ids onto the slots it touched.
    ///
    /// Returns ids in `[0, groupCount)` for `GroupBy`, and the slot each id belongs to, which is what
    /// `fold(slotOfDense:…)` folds the resulting per-group aggregates back into.
    func denseIdsForRows(keys: MetalArray<Int64>)
        throws -> (ids: MetalArray<Int32>, groupCount: Int, slotOfDense: MetalArrowBuffer) {
        let n = keys.length
        if keys.nullCount > 0 { sawNullKey = true }
        let r = try insert(keys, dense: true, estimatedNew: estimateNew(rows: n))
        guard let slotOfDense = r.slotOfDense, let denseOfSlot else {
            throw ArrowMetalError.invalidArrowArray("streaming group table: dense ids were not built")
        }
        let ids = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * 4, zeroed: false, context: context)
        let pso = try pipeline("sgt_dense_of_row")
        try context.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(r.slotOf.mtl, offset: r.slotOf.offset, index: 0)
            enc.setBuffer(denseOfSlot.mtl, offset: denseOfSlot.offset, index: 1)
            Dispatch.setLength(enc, n, nil, index: 2)
            enc.setBuffer(ids.mtl, offset: ids.offset, index: 3)
            Dispatch.dispatch1D(enc, pso, count: n)
        }
        context.retainUntilFlush(keys)
        context.retainUntilFlush(r.slotOf)
        return (MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: ids, context: context),
                r.groupCount, slotOfDense)
    }

    // MARK: - Insert

    private struct InsertResult {
        let slotOf: MetalArrowBuffer
        let groupCount: Int
        let slotOfDense: MetalArrowBuffer?
    }

    /// How many new keys the next batch is expected to bring, which is what the pre-emptive growth
    /// sizes for. Being wrong costs one retry of the insert, never a wrong answer.
    private func estimateNew(rows: Int) -> Int {
        Swift.min(rows, Swift.max(1024, 2 * lastNewKeys))
    }

    /// Inserts every element of `keys`, growing and retrying until the whole batch fits.
    private func insert(_ keys: MetalArray<Int64>, dense: Bool, estimatedNew: Int) throws -> InsertResult {
        let n = keys.length
        try Dispatch.checkLength(n)
        while Double(used + estimatedNew) > Double(slots) * loadFactor { try grow() }

        let slotOf = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * 4, zeroed: false, context: context)
        let usedBefore = used
        while true {
            var slotOfDense: MetalArrowBuffer? = nil
            if dense {
                try ensureDenseArrays()
                slotOfDense = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * 4, zeroed: false,
                                                            context: context)
            }
            stampId &+= 1
            if stampId == 0 { stampId = 1 }
            overflow.mutableTyped(UInt32.self)[0] = 0
            denseCounter.mutableTyped(UInt32.self)[0] = 0
            usedCounter.mutableTyped(UInt32.self)[0] = UInt32(used)

            let pso = try pipeline("sgt_insert")
            let vv = keys.validity
            let dummy = words
            try context.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(keys.values.mtl, offset: keys.values.offset, index: 0)
                let v = vv ?? keys.values
                enc.setBuffer(v.mtl, offset: v.offset, index: 1)
                Dispatch.setUInt(enc, vv == nil ? 0 : 1, index: 2)
                Dispatch.setLength(enc, n, nil, index: 3)
                enc.setBuffer(words.mtl, offset: words.offset, index: 4)
                Dispatch.setUInt(enc, Int(mask), index: 5)
                Dispatch.setUInt(enc, slots, index: 6)
                Dispatch.setUInt(enc, probeBudget, index: 7)
                enc.setBuffer(slotOf.mtl, offset: slotOf.offset, index: 8)
                enc.setBuffer(usedCounter.mtl, offset: usedCounter.offset, index: 9)
                enc.setBuffer(overflow.mtl, offset: overflow.offset, index: 10)
                Dispatch.setUInt(enc, dense ? 1 : 0, index: 11)
                Dispatch.setUInt(enc, Int(stampId), index: 12)
                let st = stamp ?? dummy
                enc.setBuffer(st.mtl, offset: st.offset, index: 13)
                let dos = denseOfSlot ?? dummy
                enc.setBuffer(dos.mtl, offset: dos.offset, index: 14)
                let sod = slotOfDense ?? dummy
                enc.setBuffer(sod.mtl, offset: sod.offset, index: 15)
                enc.setBuffer(denseCounter.mtl, offset: denseCounter.offset, index: 16)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
            try context.syncPoint()
            let overflowed = overflow.typed(UInt32.self)[0] != 0
            let nowUsed = Int(usedCounter.typed(UInt32.self)[0])
            if !overflowed && Double(nowUsed) <= Double(slots) * loadFactor {
                used = nowUsed
                lastNewKeys = nowUsed - usedBefore
                let k = Int(denseCounter.typed(UInt32.self)[0])
                return InsertResult(slotOf: slotOf, groupCount: dense ? k : 0, slotOfDense: slotOfDense)
            }
            // Either the table filled up mid-batch or it ended too full to probe cheaply. Growing
            // rehashes what did land; the retry finds those keys present and claims the rest.
            used = nowUsed
            if overflowed {
                while Double(used + n) > Double(slots) * loadFactor { try grow() }
            } else {
                try grow()
            }
        }
    }

    private func ensureDenseArrays() throws {
        if stamp == nil {
            stamp = try MetalArrowBuffer.allocate(byteCount: (slots + 1) * 4, context: context)
        }
        if denseOfSlot == nil {
            denseOfSlot = try MetalArrowBuffer.allocate(byteCount: (slots + 1) * 4, zeroed: false,
                                                        context: context)
        }
    }

    /// The 64-bit payload buffer of an aggregate's values, and its validity bitmap.
    private func sumBuffer(_ v: AnyMetalArray?) throws -> (MetalArrowBuffer, MetalArrowBuffer?) {
        guard let v else { return (words, nil) }
        switch v {
        case .int64(let a): return (a.values, a.validity)
        case .uint64(let a): return (a.values, a.validity)
        case .float64(let a): return (a.values, a.validity)
        default: throw ArrowMetalError.unsupportedType("resident group table takes 64-bit aggregates, got \(v.arrowFormat)")
        }
    }

    // MARK: - Growth

    private func grow() throws {
        let newSlots = slots * 2
        guard newSlots <= Int(Int32.max) else {
            throw ArrowMetalError.invalidArrowArray("streaming group table outgrew 2^31 slots")
        }
        let newWords = try MetalArrowBuffer.allocate(byteCount: newSlots * 12, context: context)
        let moved = try MetalArrowBuffer.allocate(byteCount: slots * 4, zeroed: false, context: context)
        let newMask = UInt32(newSlots - 1)
        overflow.mutableTyped(UInt32.self)[0] = 0

        let rehash = try pipeline("sgt_rehash")
        let old = slots
        try context.run { enc in
            enc.setComputePipelineState(rehash)
            enc.setBuffer(words.mtl, offset: words.offset, index: 0)
            Dispatch.setLength(enc, old, nil, index: 1)
            enc.setBuffer(newWords.mtl, offset: newWords.offset, index: 2)
            Dispatch.setUInt(enc, Int(newMask), index: 3)
            Dispatch.setUInt(enc, Swift.min(newSlots, 4096), index: 4)
            enc.setBuffer(moved.mtl, offset: moved.offset, index: 5)
            enc.setBuffer(overflow.mtl, offset: overflow.offset, index: 6)
            Dispatch.dispatch1D(enc, rehash, count: old)
        }
        var newSums: [MetalArrowBuffer] = [], newCounts: [MetalArrowBuffer] = []
        let move = try pipeline("sgt_move_acc")
        for i in 0..<aggregateCount {
            let ns = try MetalArrowBuffer.allocate(byteCount: (newSlots + 1) * 8, context: context)
            let nc = try MetalArrowBuffer.allocate(byteCount: (newSlots + 1) * 8, context: context)
            try context.run { enc in
                enc.setComputePipelineState(move)
                enc.setBuffer(moved.mtl, offset: moved.offset, index: 0)
                enc.setBuffer(sums[i].mtl, offset: sums[i].offset, index: 1)
                enc.setBuffer(counts[i].mtl, offset: counts[i].offset, index: 2)
                enc.setBuffer(ns.mtl, offset: ns.offset, index: 3)
                enc.setBuffer(nc.mtl, offset: nc.offset, index: 4)
                Dispatch.setLength(enc, old, nil, index: 5)
                Dispatch.dispatch1D(enc, move, count: old)
            }
            newSums.append(ns)
            newCounts.append(nc)
        }
        try context.syncPoint()
        if overflow.typed(UInt32.self)[0] != 0 {
            throw ArrowMetalError.invalidArrowArray("streaming group table overflowed its probe budget while growing")
        }
        // The null key's group lives one past the end of the table and does not move.
        for i in 0..<aggregateCount {
            newSums[i].mutableTyped(UInt64.self)[newSlots] = sums[i].typed(UInt64.self)[slots]
            newCounts[i].mutableTyped(Int64.self)[newSlots] = counts[i].typed(Int64.self)[slots]
        }
        words = newWords; sums = newSums; counts = newCounts; slots = newSlots
        // Both are indexed by slot, so a bigger table needs bigger ones; a fresh (zeroed) stamp array
        // is harmless because the stamp id itself is never zero.
        if stamp != nil { stamp = try MetalArrowBuffer.allocate(byteCount: (newSlots + 1) * 4, context: context) }
        if denseOfSlot != nil {
            denseOfSlot = try MetalArrowBuffer.allocate(byteCount: (newSlots + 1) * 4, zeroed: false,
                                                        context: context)
        }
    }

    // MARK: - Reading the answer

    /// The finished table: distinct keys and, per aggregate, its sum and count, in slot order.
    /// The null key's group (if any row had one) is appended last with an invalid key.
    func readOut() throws -> (keys: MetalArray<Int64>, sums: [AnyMetalArray], counts: [MetalArray<Int64>]) {
        try context.syncPoint()
        let keyBuffer = try MetalArrowBuffer.allocate(byteCount: slots * 8, zeroed: false, context: context)
        let live = try MetalArrowBuffer.allocate(byteCount: slots * 4, zeroed: false, context: context)
        let pso = try pipeline("sgt_extract")
        try context.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(words.mtl, offset: words.offset, index: 0)
            Dispatch.setLength(enc, slots, nil, index: 1)
            enc.setBuffer(keyBuffer.mtl, offset: keyBuffer.offset, index: 2)
            enc.setBuffer(live.mtl, offset: live.offset, index: 3)
            Dispatch.dispatch1D(enc, pso, count: slots)
        }
        let liveArr = MetalArray<UInt32>(length: slots, nullCount: 0, validity: nil, values: live, context: context)
        let mask = try liveArr.compare(.ne, 0)
        let liveKeys = try MetalArray<Int64>(length: slots, nullCount: 0, validity: nil, values: keyBuffer,
                                             context: context).filter(mask)
        var outSums: [AnyMetalArray] = []
        var outCounts: [MetalArray<Int64>] = []
        for i in 0..<aggregateCount {
            let c = try MetalArray<Int64>(length: slots, nullCount: 0, validity: nil, values: counts[i],
                                          context: context).filter(mask)
            let s: AnyMetalArray
            switch kinds[i] {
            case .double: s = .float64(try MetalArray<Double>(length: slots, nullCount: 0, validity: nil,
                                                              values: sums[i], context: context).filter(mask))
            case .uint: s = .uint64(try MetalArray<UInt64>(length: slots, nullCount: 0, validity: nil,
                                                           values: sums[i], context: context).filter(mask))
            default: s = .int64(try MetalArray<Int64>(length: slots, nullCount: 0, validity: nil,
                                                      values: sums[i], context: context).filter(mask))
            }
            outSums.append(s)
            outCounts.append(c)
        }
        try context.syncPoint()
        guard sawNullKey else { return (liveKeys, outSums, outCounts) }

        // Append the null group. One row: cheap enough to build on the host.
        let n = liveKeys.length
        var keyVals = [Int64?](repeating: nil, count: n + 1)
        let kp = liveKeys.valuePointer
        for i in 0..<n { keyVals[i] = kp[i] }
        let allKeys = try MetalArray<Int64>(keyVals, context: context)
        var mergedSums: [AnyMetalArray] = []
        var mergedCounts: [MetalArray<Int64>] = []
        for i in 0..<aggregateCount {
            let nullCount = counts[i].typed(Int64.self)[slots]
            let nullSumBits = sums[i].typed(UInt64.self)[slots]
            let tail: AnyMetalArray
            switch kinds[i] {
            case .double: tail = .float64(try MetalArray<Double>([Double(bitPattern: nullSumBits)], context: context))
            case .uint: tail = .uint64(try MetalArray<UInt64>([nullSumBits], context: context))
            default: tail = .int64(try MetalArray<Int64>([Int64(bitPattern: nullSumBits)], context: context))
            }
            mergedSums.append(try concatColumns([outSums[i], tail]))
            mergedCounts.append(try MetalArray<Int64>(
                (0..<n).map { outCounts[i].valuePointer[$0] } + [nullCount], context: context))
        }
        return (allKeys, mergedSums, mergedCounts)
    }
}

/// The per-group float64 sum of a batch, and the non-null count that comes out of the same pass.
///
/// Bit-identical to `GroupBy.sumDouble` — the kernel reproduces that reduction's exact order — but
/// one thread per group instead of one threadgroup, which is what a batch with a million groups of
/// one row needs. Returns nil when the runs are long enough that a threadgroup each is the better
/// shape; both give the same answer, so this is only a choice of dispatch.
///
/// The `sum` also folds in the count Arrow needs to null an all-null group, so a streaming `sum` no
/// longer pays for a second pass over the rows to count them.
func residentSegmentedSumDouble(_ values: MetalArray<Double>, _ gb: GroupBy<Int32>)
    throws -> (sum: MetalArray<Double>, count: MetalArray<Int64>)? {
    let ctx = values.context
    let k = gb.keyCount
    // One thread walks its whole run, so very long runs want the threadgroup shape back. The mean run
    // is what decides it: a sparse key's runs are one or two rows, which is the case this exists for.
    guard k > 0, gb.keys.length <= 64 * k else { return nil }
    let seg = try gb.segments()
    let out = try MetalArrowBuffer.allocate(byteCount: k * 8, zeroed: false, context: ctx)
    let validBytes = try MetalArrowBuffer.allocate(byteCount: Swift.max(k, 1), context: ctx)
    let cnts = try MetalArrowBuffer.allocate(byteCount: k * 8, zeroed: false, context: ctx)
    let pso = try Dispatch.pipeline(ctx, family: "streamgrouptable", source: StreamGroupTableSource.msl,
                                    function: "sgt_seg_sum_f64", type: "shared")
    let vv = values.validity
    try ctx.run { enc in
        enc.setComputePipelineState(pso)
        enc.setBuffer(seg.segStart.mtl, offset: seg.segStart.offset, index: 0)
        enc.setBuffer(seg.segEnd.mtl, offset: seg.segEnd.offset, index: 1)
        let o = seg.ord.length > 0 ? seg.ord.values : seg.segStart
        enc.setBuffer(o.mtl, offset: o.offset, index: 2)
        enc.setBuffer(values.values.mtl, offset: values.values.offset, index: 3)
        let v = vv ?? values.values
        enc.setBuffer(v.mtl, offset: v.offset, index: 4)
        Dispatch.setLength(enc, k, nil, index: 5)
        Dispatch.setUInt(enc, vv == nil ? 0 : 1, index: 6)
        enc.setBuffer(out.mtl, offset: out.offset, index: 7)
        enc.setBuffer(validBytes.mtl, offset: validBytes.offset, index: 8)
        enc.setBuffer(cnts.mtl, offset: cnts.offset, index: 9)
        Dispatch.dispatch1D(enc, pso, count: k)
    }
    ctx.retainUntilFlush(seg.ord)
    ctx.retainUntilFlush(values)
    let bitmap = try BitmapOps.packBits(ctx, bytes: validBytes, bits: k)
    try ctx.syncPoint()
    let sum = MetalArray<Double>(length: k, nullCount: 0, validity: bitmap, values: out, context: ctx)
    sum.recomputeNullCount()
    return (sum, MetalArray<Int64>(length: k, nullCount: 0, validity: nil, values: cnts, context: ctx))
}

/// A copy of `a` whose rows are null wherever `live` is false, which is how a group with no non-null
/// value comes back as Arrow's null `sum` rather than a zero.
///
/// `live` comes from a comparison against a column with no nulls of its own, so its values buffer is
/// already exactly the validity bitmap the result wants and no kernel has to build one.
func nullingWhereFalse(_ a: AnyMetalArray, _ live: MetalBooleanArray) throws -> AnyMetalArray {
    func f<T: ArrowPrimitive>(_ x: MetalArray<T>) -> MetalArray<T> {
        let out = MetalArray<T>(length: x.length, nullCount: 0, validity: live.values,
                                values: x.values, context: x.context)
        out.recomputeNullCount()
        return out
    }
    switch a {
    case .int64(let x): return .int64(f(x))
    case .uint64(let x): return .uint64(f(x))
    case .float64(let x): return .float64(f(x))
    default: return a
    }
}
