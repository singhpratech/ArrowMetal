import Foundation

/// Metal Shading Language for the GPU hash table shared by `Kernels/HashTable.swift` (primitive keys)
/// and `Kernels/StringHashTable.swift` (utf8 / binary keys).
///
/// The table is open addressing with linear probing over a power-of-two slot array in device memory.
/// A slot holds `row + 1`, never the key: 0 means empty, and the key of an occupied slot is the key of
/// row `slots[s] - 1`. That keeps every atomic 32-bit — Metal has no 64-bit atomics — and it means the
/// table needs no second array to stay consistent, which matters because MSL only guarantees
/// `memory_order_relaxed`: a cached copy of the key beside the slot could be read before its writer had
/// published it, and a probe that then walked past its own bucket would hand one key two group ids.
/// Reading `keys[slots[s] - 1]` cannot go stale that way, because the keys are written by an earlier
/// kernel and never change.
///
/// | kernel | what it does |
/// |---|---|
/// | `ht_fill` | fills a uint buffer with a constant (clearing the table on the GPU, not with a host memset) |
/// | `ht_build` | inserts every non-null row, keyed by a 64-bit value that *is* the key |
/// | `ht_mark` | 1 per occupied slot, the input to the rank scan |
/// | `ht_compact` | one representative row per occupied slot, written at that slot's rank |
/// | `ht_ids` | each row's dense group id: `relabel[rank[slot]]`, with null rows given a caller-chosen id |
///
/// `ht_key` (from `keySource`) is the one type-specialised kernel: it turns a column of any primitive
/// type into the 64-bit key the table groups by. Integers widen (sign-extended, so the map is
/// injective); floats are normalised first — every NaN to one pattern and -0 to +0 — so that bit
/// equality means Arrow value equality, exactly as `unique()`'s sort path does it.
enum HashTableSource {
    static let shared: String = KernelSource.prelude + """

    #define HT_HAS_VALIDITY 1u
    #define HT_WRITE_SLOTS  4u

    inline uint ht_slot(ulong h, uint mask) { return ((uint)h ^ (uint)(h >> 32)) & mask; }

    // splitmix64's finalizer: the 64-bit key avalanched into a slot index.
    inline ulong ht_mix(ulong h) {
        h ^= h >> 33; h *= 0xff51afd7ed558ccdUL;
        h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53UL;
        h ^= h >> 33;
        return h;
    }

    kernel void ht_fill(device uint* buf [[buffer(0)]], constant uint& value [[buffer(1)]],
                        device const uint* nPtr [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i < *nPtr) buf[i] = value;
    }

    // Inserts one row per thread.
    //
    // `sampleMask` selects a fraction of the *hash space* rather than a fraction of the rows: a row
    // takes part only when `(mixed >> 40) & sampleMask` is zero. Every occurrence of a given key makes
    // the same decision, so the number of occupied slots after a sampled build is an unbiased estimate
    // of `distinct / (sampleMask + 1)` however skewed the row frequencies are. That is the cardinality
    // estimate that sizes the real table, and it costs one extra pass over the keys.
    //
    // `maxProbe` bounds the linear-probe walk. Running out is not a property of the input: it means the
    // table is fuller than the estimate predicted, and the host retries with a bigger one. The last
    // attempt is given a table at least twice the row count and an unbounded budget, so it cannot fail.
    kernel void ht_build(device const ulong* keys [[buffer(0)]], device const uchar* validity [[buffer(1)]],
                         constant uint& flags [[buffer(2)]], device const uint* nPtr [[buffer(3)]],
                         constant uint& mask [[buffer(4)]], constant uint& sampleMask [[buffer(5)]],
                         constant uint& maxProbe [[buffer(6)]], device atomic_uint* slots [[buffer(7)]],
                         device uint* slotOf [[buffer(8)]], device atomic_uint* firstOfSlot [[buffer(9)]],
                         device atomic_uint* errorFlag [[buffer(10)]],
                         uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if ((flags & HT_HAS_VALIDITY) != 0u && !bit_get(validity, i)) return;
        ulong k = keys[i];
        ulong h = ht_mix(k);
        if (sampleMask != 0u && ((((uint)(h >> 40)) & sampleMask) != 0u)) return;
        uint s = ht_slot(h, mask);
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
            if (keys[head - 1u] == k) { placed = true; break; }
            s = (s + 1u) & mask;
        }
        if (!placed) { atomic_store_explicit(errorFlag, 1u, memory_order_relaxed); return; }
        if ((flags & HT_WRITE_SLOTS) == 0u) return;
        slotOf[i] = s;
        // Lowest row per slot, so the groups can be relabelled into first-appearance order (and so the
        // representative of a group of equal-but-distinguishable values, -0.0 among 0.0s, is Arrow's).
        // The load in front of the atomic keeps all but the first few rows of each group off the atomic
        // path entirely.
        if (i < atomic_load_explicit(&firstOfSlot[s], memory_order_relaxed)) {
            atomic_fetch_min_explicit(&firstOfSlot[s], i, memory_order_relaxed);
        }
    }

    kernel void ht_mark(device const uint* slots [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                        device int* out [[buffer(2)]], uint s [[thread_position_in_grid]]) {
        if (s < *nPtr) out[s] = slots[s] != 0u ? 1 : 0;
    }

    // One representative row per occupied slot, written at that slot's rank: the compaction the rank
    // scan makes possible without a separate filter pass.
    kernel void ht_compact(device const uint* slots [[buffer(0)]], device const uint* firstOfSlot [[buffer(1)]],
                           device const int* cum [[buffer(2)]], device const uint* nPtr [[buffer(3)]],
                           device int* out [[buffer(4)]], uint s [[thread_position_in_grid]]) {
        if (s >= *nPtr || slots[s] == 0u) return;
        out[cum[s] - 1] = (int)firstOfSlot[s];
    }

    // Dense group id per row. `cum` is the INCLUSIVE scan of the occupancy marks, so `cum[s] - 1` is
    // the rank of slot `s` among occupied slots; `relabel` turns that slot-order rank into whatever
    // order the caller wants its groups in (first appearance for strings, ascending by value for
    // primitives), so the hash path and the sort path hand back the very same ids.
    kernel void ht_ids(device const uint* slotOf [[buffer(0)]], device const uchar* validity [[buffer(1)]],
                       constant uint& flags [[buffer(2)]], constant uint& nullId [[buffer(3)]],
                       device const int* cum [[buffer(4)]], device const int* relabel [[buffer(5)]],
                       device const uint* nPtr [[buffer(6)]], device int* out [[buffer(7)]],
                       uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if ((flags & HT_HAS_VALIDITY) != 0u && !bit_get(validity, i)) { out[i] = (int)nullId; return; }
        out[i] = relabel[cum[slotOf[i]] - 1];
    }
    """

    /// The 64-bit grouping key for one element type. `T` is the MSL type the values are read as, and
    /// `expr` turns `v` into the key.
    static func keySource(T: String, expr: String) -> String { KernelSource.prelude + """

    kernel void ht_key(device const \(T)* src [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                       device ulong* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        \(T) v = src[i];
        out[i] = \(expr);
    }
    """ }
}
