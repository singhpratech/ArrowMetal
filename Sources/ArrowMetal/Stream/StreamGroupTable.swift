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
// ## Why the insert is race-free without a spin lock
//
// Metal only guarantees `memory_order_relaxed` on device atomics, so the usual "claim the slot, then
// publish the key" handshake is not safe: a reader can see a claimed slot before the key beside it is
// visible. The existing `Kernels/HashTable.swift` sidesteps that by storing `row + 1` in the atomic
// and reading the key out of an array nothing writes. A *persistent* table cannot do that — its keys
// have to live in the table.
//
// The way out is the shape of the input. This table is probed with a batch's **distinct group keys**,
// never with its rows, so within one dispatch every key is different, and the work splits in two
// dispatches with a memory barrier between them:
//
// 1. `sgt_lookup` — read-only against a table nothing is writing. Every key that is already in the
//    table gets its slot; the rest are marked not-found.
// 2. `sgt_insert` — only the not-found keys run, and step 1 has already established that **none of
//    them is anywhere in the table**. So the insert never compares keys at all: it claims the first
//    empty slot on its probe chain with a 32-bit compare-exchange and writes its own key there. A
//    thread that walks over a slot another thread claimed in this very dispatch is walking over a
//    *different* key, which is exactly what linear probing wants it to do, so a stale read is not
//    merely tolerable — it is never consulted.
//
// The chain is never broken, because a slot only ever goes from empty to occupied, so a lookup in a
// later batch still terminates at the same first-empty slot it would have before.
//
// 3. `sgt_accumulate` — one thread per batch group, folding that group's sum and count into the
//    slot's accumulators. Distinct keys mean distinct slots, so this needs **no atomics either**, and
//    the float64 sum can use the correctly-rounded software adder (`d_add`) that Metal's missing
//    64-bit atomics would otherwise rule out.
//
// Growth doubles the table and rehashes on the GPU with the same "claim the first empty slot" rule,
// which is race-free for the same reason: the keys being moved are distinct.

enum StreamGroupTableSource {
    static let msl: String = KernelSource.prelude + DoubleMath.msl + """

    #define SGT_NONE 0xFFFFFFFFu

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

    // Phase 1. Read-only: nothing writes the table while this runs.
    kernel void sgt_lookup(device const long* keys [[buffer(0)]],
                           device const uchar* keyValidity [[buffer(1)]],
                           constant uint& hasValidity [[buffer(2)]],
                           device const uint* nPtr [[buffer(3)]],
                           device const uint* state [[buffer(4)]],
                           device const long* slotKeys [[buffer(5)]],
                           constant uint& mask [[buffer(6)]],
                           constant uint& nullSlot [[buffer(7)]],
                           device uint* out [[buffer(8)]],
                           uint j [[thread_position_in_grid]]) {
        if (j >= *nPtr) return;
        if (hasValidity && !bit_get(keyValidity, j)) { out[j] = nullSlot; return; }
        long k = keys[j];
        uint s = sgt_home(k, mask);
        for (uint p = 0; p <= mask; ++p) {
            uint st = state[s];
            if (st == 0u) { out[j] = SGT_NONE; return; }      // first empty slot ends the chain
            if (slotKeys[s] == k) { out[j] = s; return; }
            s = (s + 1u) & mask;
        }
        out[j] = SGT_NONE;
    }

    // Phase 2. Only the keys phase 1 proved absent. No key comparison: claim the first empty slot.
    kernel void sgt_insert(device const long* keys [[buffer(0)]],
                           device const uint* nPtr [[buffer(1)]],
                           device atomic_uint* state [[buffer(2)]],
                           device long* slotKeys [[buffer(3)]],
                           constant uint& mask [[buffer(4)]],
                           device uint* out [[buffer(5)]],
                           device atomic_uint* used [[buffer(6)]],
                           device atomic_uint* overflow [[buffer(7)]],
                           uint j [[thread_position_in_grid]]) {
        if (j >= *nPtr) return;
        if (out[j] != SGT_NONE) return;
        long k = keys[j];
        uint s = sgt_home(k, mask);
        for (uint p = 0; p <= mask; ++p) {
            uint expected = 0u;
            if (atomic_compare_exchange_weak_explicit(&state[s], &expected, 1u,
                                                      memory_order_relaxed, memory_order_relaxed)) {
                slotKeys[s] = k;
                out[j] = s;
                atomic_fetch_add_explicit(used, 1u, memory_order_relaxed);
                return;
            }
            // `expected` now holds what was really there. A weak exchange can fail spuriously on an
            // empty slot, so only advance when the slot is genuinely taken.
            if (expected != 0u) s = (s + 1u) & mask;
        }
        atomic_store_explicit(overflow, 1u, memory_order_relaxed);
    }

    // Phase 3. One thread per batch group; distinct keys mean distinct slots, so no atomics.
    // kind: 0 count only, 1 int64 sum, 2 uint64 sum, 3 float64 sum (correctly rounded `d_add`).
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

    // Growth. One thread per old slot; the keys being moved are distinct, so the same claim rule holds.
    kernel void sgt_rehash(device const uint* oldState [[buffer(0)]],
                           device const long* oldKeys [[buffer(1)]],
                           device const uint* nPtr [[buffer(2)]],
                           device atomic_uint* state [[buffer(3)]],
                           device long* slotKeys [[buffer(4)]],
                           constant uint& mask [[buffer(5)]],
                           device uint* moved [[buffer(6)]],
                           uint j [[thread_position_in_grid]]) {
        if (j >= *nPtr) return;
        if (oldState[j] == 0u) { moved[j] = SGT_NONE; return; }
        long k = oldKeys[j];
        uint s = sgt_home(k, mask);
        for (uint p = 0; p <= mask; ++p) {
            uint expected = 0u;
            if (atomic_compare_exchange_weak_explicit(&state[s], &expected, 1u,
                                                      memory_order_relaxed, memory_order_relaxed)) {
                slotKeys[s] = k;
                moved[j] = s;
                return;
            }
            if (expected != 0u) s = (s + 1u) & mask;
        }
        moved[j] = SGT_NONE;
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
    """
}

/// The GPU-resident global table of a streaming group-by over one integer key.
///
/// Holds one slot per distinct key for the whole scan: the key, and per aggregate a 64-bit sum and a
/// 64-bit count. Ten million groups with one aggregate is 10M x (8 key + 4 state + 8 sum + 8 count)
/// at a load factor of one half — about 560 MB of unified memory, which is nothing on a machine that
/// can hold the dataset's page cache.
final class StreamGroupTable {
    /// How a slot's sum is folded: matching the widths Arrow's `sum` produces.
    enum SumKind: Int { case none = 0, int = 1, uint = 2, double = 3 }

    let context: MetalContext
    let aggregateCount: Int
    /// Slot count, always a power of two. Slot `slots` (one past the end) is the null key's group.
    private(set) var slots: Int
    private var mask: UInt32 { UInt32(slots - 1) }
    /// Occupied slots.
    private(set) var used = 0

    private var state: MetalArrowBuffer
    private var keys: MetalArrowBuffer
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

    init(context: MetalContext, aggregateCount: Int, initialSlots: Int = 1 << 16) throws {
        self.context = context
        self.aggregateCount = aggregateCount
        self.kinds = Array(repeating: .none, count: aggregateCount)
        var s = 1
        while s < Swift.max(initialSlots, 16) { s <<= 1 }
        self.slots = s
        self.state = try MetalArrowBuffer.allocate(byteCount: s * 4, context: context)
        self.keys = try MetalArrowBuffer.allocate(byteCount: s * 8, zeroed: false, context: context)
        self.usedCounter = try MetalArrowBuffer.allocate(byteCount: 4, context: context)
        self.overflow = try MetalArrowBuffer.allocate(byteCount: 4, context: context)
        for _ in 0..<aggregateCount {
            sums.append(try MetalArrowBuffer.allocate(byteCount: (s + 1) * 8, context: context))
            counts.append(try MetalArrowBuffer.allocate(byteCount: (s + 1) * 8, context: context))
        }
    }

    private func pipeline(_ function: String) throws -> MTLComputePipelineState {
        try Dispatch.pipeline(context, family: "streamgrouptable", source: StreamGroupTableSource.msl,
                              function: function, type: "shared")
    }

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

        // Grow before the batch, so the insert below cannot run out of slots.
        while Double(used + g) > Double(slots) * loadFactor { try grow() }

        let slot = try MetalArrowBuffer.allocate(byteCount: g * 4, zeroed: false, context: context)
        let usedBefore = used

        let lookup = try pipeline("sgt_lookup")
        let vv = groupKeys.validity
        try context.run { enc in
            enc.setComputePipelineState(lookup)
            enc.setBuffer(groupKeys.values.mtl, offset: groupKeys.values.offset, index: 0)
            let v = vv ?? groupKeys.values
            enc.setBuffer(v.mtl, offset: v.offset, index: 1)
            Dispatch.setUInt(enc, vv == nil ? 0 : 1, index: 2)
            Dispatch.setLength(enc, g, nil, index: 3)
            enc.setBuffer(state.mtl, offset: state.offset, index: 4)
            enc.setBuffer(keys.mtl, offset: keys.offset, index: 5)
            Dispatch.setUInt(enc, Int(mask), index: 6)
            Dispatch.setUInt(enc, slots, index: 7)
            enc.setBuffer(slot.mtl, offset: slot.offset, index: 8)
            Dispatch.dispatch1D(enc, lookup, count: g)
        }
        let insert = try pipeline("sgt_insert")
        try context.run { enc in
            enc.setComputePipelineState(insert)
            enc.setBuffer(groupKeys.values.mtl, offset: groupKeys.values.offset, index: 0)
            Dispatch.setLength(enc, g, nil, index: 1)
            enc.setBuffer(state.mtl, offset: state.offset, index: 2)
            enc.setBuffer(keys.mtl, offset: keys.offset, index: 3)
            Dispatch.setUInt(enc, Int(mask), index: 4)
            enc.setBuffer(slot.mtl, offset: slot.offset, index: 5)
            enc.setBuffer(usedCounter.mtl, offset: usedCounter.offset, index: 6)
            enc.setBuffer(overflow.mtl, offset: overflow.offset, index: 7)
            Dispatch.dispatch1D(enc, insert, count: g)
        }

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
        context.retainUntilFlush(groupKeys)
        try context.syncPoint()
        if overflow.typed(UInt32.self)[0] != 0 {
            throw ArrowMetalError.invalidArrowArray("streaming group table overflowed its probe budget")
        }
        used = Int(usedCounter.typed(UInt32.self)[0])
        _ = usedBefore
    }

    /// The 64-bit payload buffer of an aggregate's per-group values, and its validity bitmap.
    private func sumBuffer(_ v: AnyMetalArray?) throws -> (MetalArrowBuffer, MetalArrowBuffer?) {
        guard let v else { return (keys, nil) }
        switch v {
        case .int64(let a): return (a.values, a.validity)
        case .uint64(let a): return (a.values, a.validity)
        case .float64(let a): return (a.values, a.validity)
        default: throw ArrowMetalError.unsupportedType("resident group table takes 64-bit aggregates, got \(v.arrowFormat)")
        }
    }

    private func grow() throws {
        let newSlots = slots * 2
        let newState = try MetalArrowBuffer.allocate(byteCount: newSlots * 4, context: context)
        let newKeys = try MetalArrowBuffer.allocate(byteCount: newSlots * 8, zeroed: false, context: context)
        let moved = try MetalArrowBuffer.allocate(byteCount: slots * 4, zeroed: false, context: context)
        let newMask = UInt32(newSlots - 1)

        let rehash = try pipeline("sgt_rehash")
        let old = slots
        try context.run { enc in
            enc.setComputePipelineState(rehash)
            enc.setBuffer(state.mtl, offset: state.offset, index: 0)
            enc.setBuffer(keys.mtl, offset: keys.offset, index: 1)
            Dispatch.setLength(enc, old, nil, index: 2)
            enc.setBuffer(newState.mtl, offset: newState.offset, index: 3)
            enc.setBuffer(newKeys.mtl, offset: newKeys.offset, index: 4)
            Dispatch.setUInt(enc, Int(newMask), index: 5)
            enc.setBuffer(moved.mtl, offset: moved.offset, index: 6)
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
        // The null key's group lives one past the end of the table and does not move.
        for i in 0..<aggregateCount {
            newSums[i].mutableTyped(UInt64.self)[newSlots] = sums[i].typed(UInt64.self)[slots]
            newCounts[i].mutableTyped(Int64.self)[newSlots] = counts[i].typed(Int64.self)[slots]
        }
        state = newState; keys = newKeys; sums = newSums; counts = newCounts; slots = newSlots
    }

    /// The finished table: distinct keys and, per aggregate, its sum and count, in slot order.
    /// The null key's group (if any row had one) is appended last with an invalid key.
    func readOut() throws -> (keys: MetalArray<Int64>, sums: [AnyMetalArray], counts: [MetalArray<Int64>]) {
        try context.syncPoint()
        let stateArr = MetalArray<UInt32>(length: slots, nullCount: 0, validity: nil, values: state, context: context)
        let mask = try stateArr.compare(.ne, 0)
        let liveKeys = try MetalArray<Int64>(length: slots, nullCount: 0, validity: nil, values: keys,
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
