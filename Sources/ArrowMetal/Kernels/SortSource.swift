import Foundation

/// LSD radix sort on the GPU: per-block histograms, a global digit scan, and a stable scatter that
/// ranks elements within each block against the other lanes of their SIMD group. Keys are 32- or
/// 64-bit unsigned patterns produced by an order-preserving map; a uint payload (the original index)
/// rides along.
///
/// The digit width is a parameter, and eight is what it measures best at. Wider digits are the
/// textbook way to cut passes — eleven bits take a 64-bit key in six passes instead of eight, a
/// quarter of the memory traffic — but measured here they are *slower*: a 2048-bin scatter needs 24 KB
/// of threadgroup memory against 3 KB, which costs most of the occupancy that hides the scatter's
/// scattered writes, and every block still pays to load and clear a bin table eight times the size on
/// every pass. Argsort of 50M int64 (M4 Max, best of five, 128 blocks): 8 bits 37.7 ms, 10 bits
/// 69.8 ms, 11 bits 84.8 ms. Twelve bits does not fit in threadgroup memory at all.
enum SortSource {
    /// Bits per radix digit: eight passes over a 64-bit key, four over a 32-bit one.
    static let digitBits = 8

    static func source(K: String, bits: Int = digitBits) -> String { KernelSource.prelude + """

    #define RADIX \(1 << bits)u
    #define DIGIT_BITS \(bits)u
    #define DIGIT_MASK \(String(format: "0x%Xu", (1 << bits) - 1))
    // Apple GPUs are 32 lanes wide, which the scan below already assumes (`simdTotals[32]`).
    #define SIMDS (TG / 32u)

    // Histogram of digit `shift` per block of `elemsPerBlock` elements, counts laid out digit-major:
    // counts[d * blocks + b]. The same sweep records the bitwise OR and AND of the block's keys, which
    // together say which bits differ anywhere in the column and so which whole passes are identity
    // permutations that can be skipped (see `argsort`). It is free here: the keys are already loaded.
    kernel void radix_histogram(device const \(K)* keys [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                                constant uint& shift [[buffer(2)]], constant uint& elemsPerBlock [[buffer(3)]],
                                constant uint& blocks [[buffer(4)]], device atomic_uint* counts [[buffer(5)]],
                                device \(K)* spanOr [[buffer(6)]], device \(K)* spanAnd [[buffer(7)]],
                                uint lid [[thread_index_in_threadgroup]], uint tgid [[threadgroup_position_in_grid]]) {
        threadgroup atomic_uint hist[RADIX];
        threadgroup atomic_uint orLo, orHi, andLo, andHi;
        for (uint d = lid; d < RADIX; d += TG) atomic_store_explicit(&hist[d], 0u, memory_order_relaxed);
        if (lid == 0) {
            atomic_store_explicit(&orLo, 0u, memory_order_relaxed);
            atomic_store_explicit(&orHi, 0u, memory_order_relaxed);
            atomic_store_explicit(&andLo, 0xFFFFFFFFu, memory_order_relaxed);
            atomic_store_explicit(&andHi, 0xFFFFFFFFu, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint n = *nPtr, start = tgid * elemsPerBlock, end = min(n, start + elemsPerBlock);
        \(K) o = 0, a = ~(\(K))0;
        for (uint i = start + lid; i < end; i += TG) {
            \(K) k = keys[i];
            o |= k; a &= k;
            atomic_fetch_add_explicit(&hist[(uint)((k >> shift) & DIGIT_MASK)], 1u, memory_order_relaxed);
        }
        atomic_fetch_or_explicit(&orLo, (uint)o, memory_order_relaxed);
        atomic_fetch_and_explicit(&andLo, (uint)a, memory_order_relaxed);
        if (sizeof(\(K)) > 4) {
            atomic_fetch_or_explicit(&orHi, (uint)((ulong)o >> 32), memory_order_relaxed);
            atomic_fetch_and_explicit(&andHi, (uint)((ulong)a >> 32), memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = lid; d < RADIX; d += TG)
            atomic_store_explicit(&counts[d * blocks + tgid], atomic_load_explicit(&hist[d], memory_order_relaxed), memory_order_relaxed);
        if (lid == 0) {
            ulong ov = (ulong)atomic_load_explicit(&orLo, memory_order_relaxed);
            ulong av = (ulong)atomic_load_explicit(&andLo, memory_order_relaxed);
            if (sizeof(\(K)) > 4) {
                ov |= (ulong)atomic_load_explicit(&orHi, memory_order_relaxed) << 32;
                av |= (ulong)atomic_load_explicit(&andHi, memory_order_relaxed) << 32;
            }
            // An empty block must not claim every bit is constant, so it reports "nothing set" for the
            // OR and "nothing clear" for the AND, which is what the identities above already give.
            spanOr[tgid] = (\(K))ov; spanAnd[tgid] = (\(K))av;
        }
    }
    // Exclusive scan over the digit-major counts table (RADIX * blocks entries), single threadgroup.
    kernel void radix_scan(device uint* counts [[buffer(0)]], constant uint& total [[buffer(1)]],
                           uint lid [[thread_index_in_threadgroup]], uint sgid [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
        threadgroup uint simdTotals[32];
        uint per = (total + TG - 1) / TG;
        uint lo = lid * per, hi = min(total, lo + per);
        uint local = 0;
        for (uint i = lo; i < hi; i++) local += counts[i];
        uint pre = simd_prefix_exclusive_sum(local);
        uint t = simd_sum(local);
        if (lane == 0) simdTotals[sgid] = t;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint prefix = 0;
        for (uint k = 0; k < sgid; k++) prefix += simdTotals[k];
        uint run = prefix + pre;
        for (uint i = lo; i < hi; i++) { uint c = counts[i]; counts[i] = run; run += c; }
    }
    // Stable scatter. A block walks its elements in chunks of TG, in order; within a chunk an element's
    // rank among the earlier elements carrying the same digit is what makes the sort stable, and it is
    // found in two steps rather than by walking the chunk:
    //
    //   * inside a SIMD group, DIGIT_BITS ballots isolate the lanes holding the same digit as this one
    //     (`peers`), so the rank is a popcount of the peers below this lane and the group's own count
    //     of the digit is a popcount of all of them;
    //   * across the SIMD groups of the chunk, each group's leader for a digit publishes that count in
    //     `peerCount`, and every lane adds up the groups before its own.
    //
    // That is a fixed dozen instructions where the old code ran a TG-iteration loop per element for the
    // rank and another for the per-digit chunk totals — around 1,500 instructions an element per pass,
    // which measured as roughly two thirds of the whole sort. `peerCount` holds counts of at most 32, so
    // a byte each is enough, and every entry a chunk sets is cleared by the same lane that set it, which
    // keeps the 2048-digit table off the critical path.
    kernel void radix_scatter(device const \(K)* keys [[buffer(0)]], device const uint* vals [[buffer(1)]],
                              device const uint* nPtr [[buffer(2)]], constant uint& shift [[buffer(3)]],
                              constant uint& elemsPerBlock [[buffer(4)]], constant uint& blocks [[buffer(5)]],
                              device const uint* offsets [[buffer(6)]],
                              device \(K)* outKeys [[buffer(7)]], device uint* outVals [[buffer(8)]],
                              uint lid [[thread_index_in_threadgroup]], uint tgid [[threadgroup_position_in_grid]],
                              uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
        threadgroup uint base[RADIX];
        threadgroup uchar peerCount[SIMDS * RADIX];
        for (uint d = lid; d < RADIX; d += TG) base[d] = offsets[d * blocks + tgid];
        for (uint j = lid; j < SIMDS * RADIX; j += TG) peerCount[j] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint n = *nPtr, start = tgid * elemsPerBlock, end = min(n, start + elemsPerBlock);
        for (uint chunk = start; chunk < end; chunk += TG) {
            uint i = chunk + lid;
            bool active = i < end;
            \(K) key = active ? keys[i] : (\(K))0;
            uint val = active ? vals[i] : 0u;
            uint d = (uint)((key >> shift) & DIGIT_MASK);
            // The lanes of this SIMD group that are inside the block, then those of them holding this
            // digit. `simd_ballot` must be reached by every lane, so the mask is arithmetic, not a vote.
            int avail = (int)end - (int)(chunk + sgid * 32u);
            uint peers = avail >= 32 ? 0xFFFFFFFFu : (avail <= 0 ? 0u : (0xFFFFFFFFu >> (32 - (uint)avail)));
            for (uint b = 0u; b < DIGIT_BITS; b++) {
                bool one = ((d >> b) & 1u) != 0u;
                uint ones = (uint)((simd_vote::vote_t)simd_ballot(one));
                peers &= one ? ones : ~ones;
            }
            uint rank = popcount(peers & ((1u << lane) - 1u));
            bool leader = active && rank == 0u;
            if (leader) peerCount[sgid * RADIX + d] = (uchar)popcount(peers);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint before = 0u, total = 0u;
            if (active) {
                for (uint s = 0u; s < SIMDS; s++) {
                    uint c = peerCount[s * RADIX + d];
                    before += (s < sgid) ? c : 0u;
                    total += c;
                }
                uint pos = base[d] + before + rank;
                outKeys[pos] = key;
                outVals[pos] = val;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (leader) {
                peerCount[sgid * RADIX + d] = 0;        // ready for the next chunk, no bulk clear needed
                if (before == 0u) base[d] += total;     // the digit's first SIMD group advances the base
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    // Order-preserving key mappings.
    // Order-preserving key mappings; `inv` flips the order (descending) while keeping the sort stable.
    kernel void key_from_i32(device const int* a [[buffer(0)]], device const uint* nPtr [[buffer(1)]], device uint* out [[buffer(2)]], constant uint& inv [[buffer(3)]], uint i [[thread_position_in_grid]]) { if (i < *nPtr) { uint k = (uint)a[i] ^ 0x80000000u; out[i] = inv ? ~k : k; } }
    kernel void key_from_u32(device const uint* a [[buffer(0)]], device const uint* nPtr [[buffer(1)]], device uint* out [[buffer(2)]], constant uint& inv [[buffer(3)]], uint i [[thread_position_in_grid]]) { if (i < *nPtr) { uint k = a[i]; out[i] = inv ? ~k : k; } }
    kernel void key_from_f32(device const uint* a [[buffer(0)]], device const uint* nPtr [[buffer(1)]], device uint* out [[buffer(2)]], constant uint& inv [[buffer(3)]], uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return; uint b = a[i]; if ((b & 0x7FFFFFFFu) == 0u) b = 0u; bool nan = (b & 0x7FFFFFFFu) > 0x7F800000u; if (nan) b = 0x7F800001u; /* -0 == +0; every NaN is one value after +inf */ uint k = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
        /* NaN stays at the end when the order is reversed, next to the nulls, as in Arrow. UINT_MAX is
           free: only a NaN can map to key 0, so only a NaN can invert to UINT_MAX. */
        out[i] = inv ? (nan ? 0xFFFFFFFFu : ~k) : k;
    }
    kernel void key_from_i64(device const long* a [[buffer(0)]], device const uint* nPtr [[buffer(1)]], device ulong* out [[buffer(2)]], constant uint& inv [[buffer(3)]], uint i [[thread_position_in_grid]]) { if (i < *nPtr) { ulong k = (ulong)a[i] ^ 0x8000000000000000ul; out[i] = inv ? ~k : k; } }
    kernel void key_from_u64(device const ulong* a [[buffer(0)]], device const uint* nPtr [[buffer(1)]], device ulong* out [[buffer(2)]], constant uint& inv [[buffer(3)]], uint i [[thread_position_in_grid]]) { if (i < *nPtr) { ulong k = a[i]; out[i] = inv ? ~k : k; } }
    kernel void key_from_f64(device const ulong* a [[buffer(0)]], device const uint* nPtr [[buffer(1)]], device ulong* out [[buffer(2)]], constant uint& inv [[buffer(3)]], uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return; ulong b = a[i]; if ((b & 0x7FFFFFFFFFFFFFFFul) == 0ul) b = 0ul; bool nan = (b & 0x7FFFFFFFFFFFFFFFul) > 0x7FF0000000000000ul; if (nan) b = 0x7FF0000000000001ul; ulong k = (b & 0x8000000000000000ul) ? ~b : (b | 0x8000000000000000ul);
        out[i] = inv ? (nan ? 0xFFFFFFFFFFFFFFFFul : ~k) : k;
    }
    kernel void iota_u32(device uint* out [[buffer(0)]], device const uint* nPtr [[buffer(1)]], uint i [[thread_position_in_grid]]) { if (i < *nPtr) out[i] = i; }
    """ }
}
