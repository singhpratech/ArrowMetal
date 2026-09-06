import Foundation

/// Metal Shading Language for the GPU string hash table (`Kernels/StringHashTable.swift`).
///
/// The table is open addressing with linear probing over a power-of-two slot array in device memory.
/// A slot holds `row + 1`, never the string and never the hash: 0 means empty, and the key of an
/// occupied slot is the string of row `slots[s] - 1`. That keeps every atomic 32-bit — Metal has no
/// 64-bit atomics — and it means the table needs no second array to stay consistent, which matters
/// because MSL only guarantees `memory_order_relaxed`: a cached copy of the key beside the slot could
/// be read before its writer had published it, and a probe that then walked past its own bucket would
/// hand one string two group ids. Reading `hashes[slots[s] - 1]` cannot go stale that way, because the
/// hashes are written by an earlier kernel and never change.
///
/// Equality is decided by **comparing the bytes**, exactly as `is_in` does, so a 64-bit hash collision
/// costs one extra probe and can never merge two different strings. That is what lets this path drop
/// the retry loop `StringDictionary` needs.
///
/// Only the two kernels that know about strings live here: `sht_hash64` (one 64-bit hash per row, from
/// a single pass over the bytes) and `sht_build` (the insert, which decides equality by comparing bytes).
/// Clearing the table, ranking the occupied slots, compacting the representatives and writing the ids
/// are type-independent and live in `HashTableSource`.
enum StringHashTableSource {
    static let source: String = KernelSource.prelude + """

    // ---------------------------------------------------------------------------------------------
    // Hashing. One pass over the bytes feeds a 64-bit FNV-style accumulator, finished with the
    // splitmix64 avalanche so every output bit depends on every input byte. The table only ever uses
    // the hash to choose a starting slot and to skip obviously-different candidates, so its quality is
    // a speed question, not a correctness one.
    inline ulong sht_hash_bytes(device const uchar* d, int start, int len) {
        ulong h = 0xcbf29ce484222325UL ^ ((ulong)len * 0x9E3779B97F4A7C15UL);
        int blocks = len >> 2;
        for (int b = 0; b < blocks; b++) {
            int p = start + (b << 2);
            uint k = (uint)d[p] | ((uint)d[p + 1] << 8) | ((uint)d[p + 2] << 16) | ((uint)d[p + 3] << 24);
            h ^= (ulong)k;
            h *= 0x100000001B3UL;
            h ^= h >> 29;
        }
        uint tail = 0u;
        int t = start + (blocks << 2);
        switch (len & 3) {
            case 3: tail |= (uint)d[t + 2] << 16;
            case 2: tail |= (uint)d[t + 1] << 8;
            case 1: tail |= (uint)d[t];
        }
        h ^= (ulong)tail;
        h *= 0x100000001B3UL;
        h ^= h >> 33; h *= 0xff51afd7ed558ccdUL;
        h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53UL;
        h ^= h >> 33;
        return h;
    }

    kernel void sht_hash64(device const int* offsets [[buffer(0)]], device const uchar* data [[buffer(1)]],
                           device const uchar* validity [[buffer(2)]], constant uint& hasValidity [[buffer(3)]],
                           device const uint* nPtr [[buffer(4)]], device ulong* out [[buffer(5)]],
                           uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if (hasValidity != 0u && !bit_get(validity, i)) { out[i] = 0UL; return; }
        int start = offsets[i];
        out[i] = sht_hash_bytes(data, start, offsets[i + 1] - start);
    }

    // ---------------------------------------------------------------------------------------------
    // The table.
    #define SHT_HAS_VALIDITY 1u
    #define SHT_VERIFY       2u
    #define SHT_WRITE_SLOTS  4u

    inline bool sht_eq(device const int* o, device const uchar* d, uint i, uint j) {
        int a0 = o[i], la = o[i + 1] - a0;
        int b0 = o[j], lb = o[j + 1] - b0;
        if (la != lb) return false;
        for (int t = 0; t < la; t++) if (d[a0 + t] != d[b0 + t]) return false;
        return true;
    }

    // Start slot. The hash is already avalanched, so folding the halves together is enough.
    inline uint sht_slot(ulong h, uint mask) { return ((uint)h ^ (uint)(h >> 32)) & mask; }

    // Inserts one row per thread.
    //
    // `sampleMask` selects a fraction of the *hash space* rather than a fraction of the rows: a row
    // takes part only when `(h >> 40) & sampleMask` is zero. Every occurrence of a given string makes
    // the same decision, so the number of occupied slots after a sampled build is an unbiased estimate
    // of `distinct / (sampleMask + 1)` however skewed the row frequencies are. That is the cardinality
    // estimate that sizes the real table, and it costs one extra pass over the hashes.
    //
    // `maxProbe` bounds the linear-probe walk. Running out is not an error of the input: it means the
    // table is fuller than the estimate predicted, and the host retries with a bigger one. The last
    // attempt is given a table at least twice the row count and an unbounded budget, so it cannot fail.
    kernel void sht_build(device const int* offsets [[buffer(0)]], device const uchar* data [[buffer(1)]],
                          device const ulong* hashes [[buffer(2)]], device const uchar* validity [[buffer(3)]],
                          constant uint& flags [[buffer(4)]], device const uint* nPtr [[buffer(5)]],
                          constant uint& mask [[buffer(6)]], constant uint& sampleMask [[buffer(7)]],
                          constant uint& maxProbe [[buffer(8)]], device atomic_uint* slots [[buffer(9)]],
                          device uint* slotOf [[buffer(10)]], device atomic_uint* firstOfSlot [[buffer(11)]],
                          device atomic_uint* errorFlag [[buffer(12)]],
                          uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if ((flags & SHT_HAS_VALIDITY) != 0u && !bit_get(validity, i)) return;
        ulong h = hashes[i];
        if (sampleMask != 0u && ((((uint)(h >> 40)) & sampleMask) != 0u)) return;
        uint s = sht_slot(h, mask);
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
            if (hashes[r] == h && (((flags & SHT_VERIFY) == 0u) || sht_eq(offsets, data, i, r))) {
                placed = true; break;
            }
            s = (s + 1u) & mask;
        }
        if (!placed) { atomic_store_explicit(errorFlag, 1u, memory_order_relaxed); return; }
        if ((flags & SHT_WRITE_SLOTS) == 0u) return;
        slotOf[i] = s;
        // Lowest row per slot, so the groups can be relabelled into first-seen order. The load in front
        // of the atomic keeps all but the first few rows of each group off the atomic path entirely.
        if (i < atomic_load_explicit(&firstOfSlot[s], memory_order_relaxed)) {
            atomic_fetch_min_explicit(&firstOfSlot[s], i, memory_order_relaxed);
        }
    }
    """
}
