import Foundation

/// MSL for the GPU hash join.
///
/// The build (right) side goes into an open-addressing table in device memory. A slot holds
/// `row + 1` (0 means empty), never the key itself: the slot's key is `keys[slot - 1]`. That keeps
/// every atomic 32-bit, which is all the GPU has, and works unchanged for 64-bit keys. Duplicate
/// keys chain through a per-build-row `next` array: the first row for a key claims the slot with a
/// compare-and-swap, later rows push themselves onto the front of the chain with an atomic exchange
/// on the same slot (the slot's key is unchanged, so the exchange cannot break the table).
///
/// The probe (left) side runs twice over the same chains: once to count matches per left row, then,
/// after an exclusive scan of those counts, once more to write the index pairs at their own offsets.
/// Null keys are never inserted and never match.
enum JoinSource {
    /// `KT` is the key type: "int" or "long".
    static func source(KT: String) -> String {
        // Hash: MurmurHash3's finalizer, applied to the 32-bit key or to a mix of the two halves.
        let hash = KT == "long"
            ? "uint lo = (uint)k, hi = (uint)(((ulong)k) >> 32); return fmix32(fmix32(lo) ^ (hi * 0x9E3779B9u));"
            : "return fmix32((uint)k);"
        return KernelSource.prelude + """
        inline uint fmix32(uint h) { h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16; return h; }
        inline uint hj_hash(\(KT) k) { \(hash) }

        // Zeroes a table (slots). One thread per word.
        kernel void hj_clear(device uint* buf [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {
            if (i < *nPtr) buf[i] = 0u;
        }

        // Inserts every non-null build row into the table. One thread per build row.
        kernel void hj_build(device const \(KT)* keys [[buffer(0)]],
                             device const uchar* validity [[buffer(1)]],
                             constant uint& hasValidity [[buffer(2)]],
                             device const uint* nPtr [[buffer(3)]],
                             constant uint& mask [[buffer(4)]],
                             device atomic_uint* slots [[buffer(5)]],
                             device uint* next [[buffer(6)]],
                             device atomic_uint* errorFlag [[buffer(7)]],
                             uint r [[thread_position_in_grid]]) {
            uint n = *nPtr;
            if (r >= n) return;
            if (hasValidity && !bit_get(validity, r)) return;      // null keys never match, so never insert them
            \(KT) k = keys[r];
            uint s = hj_hash(k) & mask;
            uint limit = 2u * mask + 66u;                          // the table is at least 2x the build rows
            for (uint p = 0; p < limit; p++) {
                uint head = atomic_load_explicit(&slots[s], memory_order_relaxed);
                if (head == 0u) {
                    uint expected = 0u;
                    if (atomic_compare_exchange_weak_explicit(&slots[s], &expected, r + 1u,
                                                              memory_order_relaxed, memory_order_relaxed)) {
                        next[r] = 0u;                              // first row for this key: end of the chain
                        return;
                    }
                    head = expected;                               // lost the race, or a spurious weak failure
                    if (head == 0u) continue;                      // spurious: retry the same slot
                }
                if (keys[head - 1u] == k) {                        // same key: push onto the bucket's chain
                    next[r] = atomic_exchange_explicit(&slots[s], r + 1u, memory_order_relaxed);
                    return;
                }
                s = (s + 1u) & mask;
            }
            atomic_store_explicit(errorFlag, 1u, memory_order_relaxed);   // table full: cannot happen, guard anyway
        }

        // Head of the chain of build rows whose key is `k` (0 when the key is absent).
        inline uint hj_find(device const \(KT)* bkeys, device const uint* slots, uint mask, \(KT) k) {
            uint s = hj_hash(k) & mask;
            for (uint p = 0; p <= mask; p++) {
                uint head = slots[s];
                if (head == 0u) return 0u;
                if (bkeys[head - 1u] == k) return head;
                s = (s + 1u) & mask;
            }
            return 0u;
        }

        // Pass 1: matches per left row (a left join emits one row for an unmatched left row).
        kernel void hj_probe_count(device const \(KT)* lkeys [[buffer(0)]],
                                   device const uchar* lvalidity [[buffer(1)]],
                                   constant uint& hasValidity [[buffer(2)]],
                                   device const uint* nPtr [[buffer(3)]],
                                   constant uint& mask [[buffer(4)]],
                                   device const \(KT)* bkeys [[buffer(5)]],
                                   device const uint* slots [[buffer(6)]],
                                   device const uint* next [[buffer(7)]],
                                   constant uint& isLeft [[buffer(8)]],
                                   device uint* counts [[buffer(9)]],
                                   uint i [[thread_position_in_grid]]) {
            uint n = *nPtr;
            if (i >= n) return;
            uint c = 0u;
            if (!(hasValidity && !bit_get(lvalidity, i))) {
                for (uint r = hj_find(bkeys, slots, mask, lkeys[i]); r != 0u; r = next[r - 1u]) c++;
            }
            counts[i] = (c == 0u && isLeft) ? 1u : c;
        }

        // Pass 2: writes this left row's pairs starting at its scanned offset.
        kernel void hj_probe_write(device const \(KT)* lkeys [[buffer(0)]],
                                   device const uchar* lvalidity [[buffer(1)]],
                                   constant uint& hasValidity [[buffer(2)]],
                                   device const uint* nPtr [[buffer(3)]],
                                   constant uint& mask [[buffer(4)]],
                                   device const \(KT)* bkeys [[buffer(5)]],
                                   device const uint* slots [[buffer(6)]],
                                   device const uint* next [[buffer(7)]],
                                   constant uint& isLeft [[buffer(8)]],
                                   device const uint* offsets [[buffer(9)]],
                                   device int* outLeft [[buffer(10)]],
                                   device int* outRight [[buffer(11)]],
                                   device uchar* rightValid [[buffer(12)]],
                                   constant uint& writeValid [[buffer(13)]],
                                   uint i [[thread_position_in_grid]]) {
            uint n = *nPtr;
            if (i >= n) return;
            uint pos = offsets[i], c = 0u;
            if (!(hasValidity && !bit_get(lvalidity, i))) {
                for (uint r = hj_find(bkeys, slots, mask, lkeys[i]); r != 0u; r = next[r - 1u]) {
                    outLeft[pos] = (int)i;
                    outRight[pos] = (int)(r - 1u);
                    if (writeValid) rightValid[pos] = 1;
                    pos++; c++;
                }
            }
            if (c == 0u && isLeft) {
                outLeft[pos] = (int)i;
                outRight[pos] = 0;
                if (writeValid) rightValid[pos] = 0;               // unmatched left row: null right index
            }
        }

        // Exclusive scan of the per-row counts, two-level (same shape as the string offsets scan):
        // per-block scan + block totals, a single-threadgroup scan of the totals, then add the block offset.
        kernel void hj_scan_block(device const uint* vals [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                                  device uint* out [[buffer(2)]], device uint* blockTotals [[buffer(3)]],
                                  uint i [[thread_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                                  uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
            threadgroup uint simdTotals[32];
            uint n = *nPtr;
            uint v = (i < n) ? vals[i] : 0u;
            uint pre = simd_prefix_exclusive_sum(v);
            uint t = simd_sum(v);
            if (lane == 0) simdTotals[sgid] = t;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint prefix = 0u;
            for (uint k = 0; k < sgid; k++) prefix += simdTotals[k];
            if (i < n) out[i] = prefix + pre;
            if (lid == TG - 1) { uint total = 0u; for (uint k = 0; k < TG / 32u; k++) total += simdTotals[k]; blockTotals[tgid] = total; }
        }
        kernel void hj_scan_totals(device uint* blockTotals [[buffer(0)]], constant uint& blocks [[buffer(1)]],
                                   device uint* grand [[buffer(2)]], uint lid [[thread_index_in_threadgroup]],
                                   uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
            threadgroup uint simdTotals[32];
            uint per = (blocks + TG - 1) / TG;
            uint lo = lid * per, hi = min(blocks, lo + per);
            uint local = 0u;
            for (uint b = lo; b < hi; b++) local += blockTotals[b];
            uint pre = simd_prefix_exclusive_sum(local);
            uint t = simd_sum(local);
            if (lane == 0) simdTotals[sgid] = t;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint prefix = 0u;
            for (uint k = 0; k < sgid; k++) prefix += simdTotals[k];
            uint run = prefix + pre;
            for (uint b = lo; b < hi; b++) { uint c = blockTotals[b]; blockTotals[b] = run; run += c; }
            if (lid == TG - 1) *grand = run;
        }
        kernel void hj_scan_add(device uint* out [[buffer(0)]], device const uint* blockTotals [[buffer(1)]],
                                device const uint* nPtr [[buffer(2)]],
                                uint i [[thread_position_in_grid]], uint tgid [[threadgroup_position_in_grid]]) {
            if (i < *nPtr) out[i] += blockTotals[tgid];
        }
        """
    }
}
