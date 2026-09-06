import Foundation

/// LSD radix sort on the GPU: 8-bit digits, per-block histograms, a global digit scan, and a stable
/// scatter that ranks elements within each block with a threadgroup prefix sum. Keys are 32- or 64-bit
/// unsigned patterns produced by an order-preserving map; a uint payload (the original index) rides along.
enum SortSource {
    static func source(K: String) -> String { KernelSource.prelude + """

    #define RADIX 256u
    // Histogram of digit `shift` per block of TG*ELEMS elements. counts laid out digit-major: counts[d * blocks + b].
    kernel void radix_histogram(device const \(K)* keys [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                                constant uint& shift [[buffer(2)]], constant uint& elemsPerBlock [[buffer(3)]],
                                constant uint& blocks [[buffer(4)]], device atomic_uint* counts [[buffer(5)]],
                                uint lid [[thread_index_in_threadgroup]], uint tgid [[threadgroup_position_in_grid]]) {
        threadgroup atomic_uint hist[RADIX];
        for (uint d = lid; d < RADIX; d += TG) atomic_store_explicit(&hist[d], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint n = *nPtr, start = tgid * elemsPerBlock, end = min(n, start + elemsPerBlock);
        for (uint i = start + lid; i < end; i += TG) {
            uint d = (uint)((keys[i] >> shift) & 0xFF);
            atomic_fetch_add_explicit(&hist[d], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = lid; d < RADIX; d += TG)
            atomic_store_explicit(&counts[d * blocks + tgid], atomic_load_explicit(&hist[d], memory_order_relaxed), memory_order_relaxed);
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
    // Stable scatter: each block processes its elements in order in chunks of TG; within a chunk the rank of
    // an element among equal digits is computed with per-digit threadgroup counters in sequence order.
    kernel void radix_scatter(device const \(K)* keys [[buffer(0)]], device const uint* vals [[buffer(1)]],
                              device const uint* nPtr [[buffer(2)]], constant uint& shift [[buffer(3)]],
                              constant uint& elemsPerBlock [[buffer(4)]], constant uint& blocks [[buffer(5)]],
                              device const uint* offsets [[buffer(6)]],
                              device \(K)* outKeys [[buffer(7)]], device uint* outVals [[buffer(8)]],
                              uint lid [[thread_index_in_threadgroup]], uint tgid [[threadgroup_position_in_grid]]) {
        threadgroup uint base[RADIX];
        threadgroup uint chunkDigit[TG];
        for (uint d = lid; d < RADIX; d += TG) base[d] = offsets[d * blocks + tgid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint n = *nPtr, start = tgid * elemsPerBlock, end = min(n, start + elemsPerBlock);
        for (uint chunk = start; chunk < end; chunk += TG) {
            uint i = chunk + lid;
            bool active = i < end;
            uint d = active ? (uint)((keys[i] >> shift) & 0xFF) : 0xFFFFu;
            chunkDigit[lid] = d;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (active) {
                // rank among earlier elements in this chunk with the same digit (stable)
                uint rank = 0;
                for (uint j = 0; j < lid; j++) if (chunkDigit[j] == d) rank++;
                uint pos = base[d] + rank;
                outKeys[pos] = keys[i];
                outVals[pos] = vals[i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // advance bases by the digit counts of this chunk
            if (lid < RADIX) {
                uint c = 0;
                for (uint j = 0; j < TG; j++) if (chunkDigit[j] == lid) c++;
                base[lid] += c;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    // Order-preserving key mappings.
    // Order-preserving key mappings; `inv` flips the order (descending) while keeping the sort stable.
    kernel void key_from_i32(device const int* a [[buffer(0)]], device const uint* nPtr [[buffer(1)]], device uint* out [[buffer(2)]], constant uint& inv [[buffer(3)]], uint i [[thread_position_in_grid]]) { if (i < *nPtr) { uint k = (uint)a[i] ^ 0x80000000u; out[i] = inv ? ~k : k; } }
    kernel void key_from_u32(device const uint* a [[buffer(0)]], device const uint* nPtr [[buffer(1)]], device uint* out [[buffer(2)]], constant uint& inv [[buffer(3)]], uint i [[thread_position_in_grid]]) { if (i < *nPtr) { uint k = a[i]; out[i] = inv ? ~k : k; } }
    kernel void key_from_f32(device const uint* a [[buffer(0)]], device const uint* nPtr [[buffer(1)]], device uint* out [[buffer(2)]], constant uint& inv [[buffer(3)]], uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return; uint b = a[i]; if ((b & 0x7FFFFFFFu) == 0u) b = 0u; if ((b & 0x7FFFFFFFu) > 0x7F800000u) b = 0x7F800001u; /* -0 == +0; every NaN is one value after +inf */ uint k = (b & 0x80000000u) ? ~b : (b | 0x80000000u); out[i] = inv ? ~k : k;
    }
    kernel void key_from_i64(device const long* a [[buffer(0)]], device const uint* nPtr [[buffer(1)]], device ulong* out [[buffer(2)]], constant uint& inv [[buffer(3)]], uint i [[thread_position_in_grid]]) { if (i < *nPtr) { ulong k = (ulong)a[i] ^ 0x8000000000000000ul; out[i] = inv ? ~k : k; } }
    kernel void key_from_u64(device const ulong* a [[buffer(0)]], device const uint* nPtr [[buffer(1)]], device ulong* out [[buffer(2)]], constant uint& inv [[buffer(3)]], uint i [[thread_position_in_grid]]) { if (i < *nPtr) { ulong k = a[i]; out[i] = inv ? ~k : k; } }
    kernel void key_from_f64(device const ulong* a [[buffer(0)]], device const uint* nPtr [[buffer(1)]], device ulong* out [[buffer(2)]], constant uint& inv [[buffer(3)]], uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return; ulong b = a[i]; if ((b & 0x7FFFFFFFFFFFFFFFul) == 0ul) b = 0ul; if ((b & 0x7FFFFFFFFFFFFFFFul) > 0x7FF0000000000000ul) b = 0x7FF0000000000001ul; ulong k = (b & 0x8000000000000000ul) ? ~b : (b | 0x8000000000000000ul); out[i] = inv ? ~k : k;
    }
    kernel void iota_u32(device uint* out [[buffer(0)]], device const uint* nPtr [[buffer(1)]], uint i [[thread_position_in_grid]]) { if (i < *nPtr) out[i] = i; }
    """ }
}
