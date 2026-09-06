import Foundation

/// MSL for GPU radix select: find the k-th best order-preserving key with a digit histogram, then compact
/// only the rows that can still be in the answer.
///
/// Every kernel here partitions the input the same way — one **sub-block per simdgroup**, `eps` consecutive
/// rows each — so a sub-block's output offsets are just the exclusive scan of its own counts and the scatter
/// needs no threadgroup barriers at all. `rs_histogram` writes the digit counts of every sub-block, which is
/// simultaneously (a) the global histogram, once summed over sub-blocks, and (b) the per-sub-block counts the
/// scatter needs. That is why the fast path reads the column exactly twice.
///
/// Keys are the same order-preserving map `SortSource` and `TopKSource` use (`tk_map`, with `inv` flipping it
/// for "largest"), so the rows this selects and the order they come out in match `argsort` exactly.
enum RadixSelectSource {
    /// Sub-blocks per threadgroup: one per simdgroup.
    static let subsPerGroup = 8
    /// Digit width. 256 bins keeps the per-sub-block count table at 256 * subBlocks words (a few MB) and the
    /// candidate bin at about n/256 rows for well-spread keys.
    static let radix = 256

    static func source(kind: String, V: String, K: String) -> String {
        KernelSource.prelude + "\n" + TopKSource.keyMap(kind: kind, V: V, K: K) + """


        #define RS_RADIX 256u
        #define RS_SIMD 32u
        #define RS_SUBS 8u

        // Digit `shift` of every row, counted per sub-block. Each simdgroup owns a private 256-bin histogram
        // in threadgroup memory, so the only atomic contention is between the 32 lanes of one simdgroup.
        // `hasPrefix` restricts the count to rows whose key already matches the narrowing prefix.
        kernel void rs_histogram(device const \(V)* vals [[buffer(0)]],
                                 device const uchar* validity [[buffer(1)]],
                                 device const uint* nPtr [[buffer(2)]],
                                 constant uint& hasValidity [[buffer(3)]],
                                 constant uint& inv [[buffer(4)]],
                                 constant uint& shift [[buffer(5)]],
                                 constant \(K)& prefix [[buffer(6)]],
                                 constant uint& prefixShift [[buffer(7)]],
                                 constant uint& hasPrefix [[buffer(8)]],
                                 constant uint& eps [[buffer(9)]],
                                 constant uint& subBlocks [[buffer(10)]],
                                 device uint* counts [[buffer(11)]],
                                 uint lid [[thread_index_in_threadgroup]],
                                 uint sgid [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint tgid [[threadgroup_position_in_grid]]) {
            threadgroup atomic_uint hist[RS_SUBS * RS_RADIX];
            for (uint i = lid; i < RS_SUBS * RS_RADIX; i += TG) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint n = *nPtr;
            uint sb = tgid * RS_SUBS + sgid;
            uint start = sb * eps;
            uint end = (start >= n) ? start : min(n, start + eps);
            threadgroup atomic_uint* bank = hist + sgid * RS_RADIX;
            for (uint chunk = start; chunk < end; chunk += RS_SIMD) {
                uint i = chunk + lane;
                if (i >= end) continue;
                if (hasValidity && !bit_get(validity, i)) continue;
                \(K) key = tk_map(vals[i], inv);
                if (hasPrefix && (key >> prefixShift) != prefix) continue;
                uint dg = (uint)((key >> shift) & 0xFFu);
                atomic_fetch_add_explicit(&bank[dg], 1u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Digit-major: consecutive sub-blocks are adjacent, which is what the two readers below want.
            // Sub-blocks past the end of the column write zeros, so the table is always fully defined.
            for (uint dg = lane; dg < RS_RADIX; dg += RS_SIMD)
                counts[dg * subBlocks + sb] = atomic_load_explicit(&hist[sgid * RS_RADIX + dg], memory_order_relaxed);
        }

        // Global count per digit: one threadgroup per digit, summing that digit's row of the table.
        kernel void rs_totals(device const uint* counts [[buffer(0)]],
                              constant uint& subBlocks [[buffer(1)]],
                              device uint* totals [[buffer(2)]],
                              uint lid [[thread_index_in_threadgroup]],
                              uint sgid [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint dg [[threadgroup_position_in_grid]]) {
            threadgroup uint parts[RS_SUBS];
            uint s = 0;
            for (uint b = lid; b < subBlocks; b += TG) s += counts[dg * subBlocks + b];
            uint t = simd_sum(s);
            if (lane == 0u) parts[sgid] = t;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid == 0u) { uint tot = 0u; for (uint r = 0u; r < RS_SUBS; r++) tot += parts[r]; totals[dg] = tot; }
        }

        // Per-sub-block counts of "digit < target" and "digit == target", straight out of the histogram table.
        // This is what makes the fast path two passes: no second read of the column to count the survivors.
        kernel void rs_block_counts(device const uint* counts [[buffer(0)]],
                                    constant uint& subBlocks [[buffer(1)]],
                                    constant uint& target [[buffer(2)]],
                                    device uint* blockLt [[buffer(3)]],
                                    device uint* blockEq [[buffer(4)]],
                                    uint b [[thread_position_in_grid]]) {
            if (b >= subBlocks) return;
            uint lt = 0u;
            for (uint dg = 0u; dg < target; dg++) lt += counts[dg * subBlocks + b];
            blockLt[b] = lt;
            blockEq[b] = counts[target * subBlocks + b];
        }

        // Per-sub-block counts for an arbitrary key window. Only needed after a narrowing round, when the
        // window is finer than one top-digit bin and the histogram table can no longer answer it.
        kernel void rs_count(device const \(V)* vals [[buffer(0)]],
                             device const uchar* validity [[buffer(1)]],
                             device const uint* nPtr [[buffer(2)]],
                             constant uint& hasValidity [[buffer(3)]],
                             constant uint& inv [[buffer(4)]],
                             constant \(K)& loKey [[buffer(5)]],
                             constant \(K)& hiKey [[buffer(6)]],
                             constant uint& eps [[buffer(7)]],
                             device uint* blockLt [[buffer(8)]],
                             device uint* blockEq [[buffer(9)]],
                             uint sgid [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint tgid [[threadgroup_position_in_grid]]) {
            uint n = *nPtr;
            uint sb = tgid * RS_SUBS + sgid;
            uint start = sb * eps;
            uint end = (start >= n) ? start : min(n, start + eps);
            uint lt = 0u, eq = 0u;
            for (uint chunk = start; chunk < end; chunk += RS_SIMD) {
                uint i = chunk + lane;
                if (i >= end) continue;
                if (hasValidity && !bit_get(validity, i)) continue;
                \(K) key = tk_map(vals[i], inv);
                if (key < loKey) lt++;
                else if (key <= hiKey) eq++;
            }
            uint tl = simd_sum(lt), te = simd_sum(eq);
            if (lane == 0u) { blockLt[sb] = tl; blockEq[sb] = te; }
        }

        // Exclusive scan of `count` words in place, starting at `base`. One threadgroup; `count` is the number
        // of sub-blocks (a few thousand), so a strided per-thread pass and a threadgroup scan is enough.
        kernel void rs_scan(device uint* arr [[buffer(0)]],
                            constant uint& count [[buffer(1)]],
                            constant uint& base [[buffer(2)]],
                            uint lid [[thread_index_in_threadgroup]],
                            uint sgid [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
            threadgroup uint parts[RS_SUBS];
            uint per = (count + TG - 1u) / TG;
            uint lo = min(count, lid * per), hi = min(count, lo + per);
            uint local = 0u;
            for (uint i = lo; i < hi; i++) local += arr[i];
            uint pre = simd_prefix_exclusive_sum(local);
            uint t = simd_sum(local);
            if (lane == 0u) parts[sgid] = t;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint prefix = 0u;
            for (uint r = 0u; r < sgid; r++) prefix += parts[r];
            uint run = base + prefix + pre;
            for (uint i = lo; i < hi; i++) { uint c = arr[i]; arr[i] = run; run += c; }
        }

        // Compacts the rows that can still be in the answer, in row order, into one array:
        //   [0, mA)      keys strictly better than the candidate window  — all of them are winners
        //   [mA, limit)  keys inside the window                          — the ties to be resolved
        // Every key in the first part is smaller than every key in the second, and each part is written in
        // ascending row order, so one *stable* sort by key turns the whole array into (key, row) order.
        // `limitEq` bounds the second part: when the window has collapsed to a single key the answer needs
        // only the first `k - mA` of the ties, and writing the rest would cost a full extra array.
        kernel void rs_scatter(device const \(V)* vals [[buffer(0)]],
                               device const uchar* validity [[buffer(1)]],
                               device const uint* nPtr [[buffer(2)]],
                               constant uint& hasValidity [[buffer(3)]],
                               constant uint& inv [[buffer(4)]],
                               constant \(K)& loKey [[buffer(5)]],
                               constant \(K)& hiKey [[buffer(6)]],
                               constant uint& eps [[buffer(7)]],
                               constant uint& limitEq [[buffer(8)]],
                               device const uint* offLt [[buffer(9)]],
                               device const uint* offEq [[buffer(10)]],
                               device \(K)* outKeys [[buffer(11)]],
                               device uint* outRows [[buffer(12)]],
                               device const uint* srcRows [[buffer(13)]],
                               constant uint& hasSrcRows [[buffer(14)]],
                               constant uint& wantLt [[buffer(15)]],
                               uint sgid [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint tgid [[threadgroup_position_in_grid]]) {
            uint n = *nPtr;
            uint sb = tgid * RS_SUBS + sgid;
            uint start = sb * eps;
            uint end = (start >= n) ? start : min(n, start + eps);
            uint posLt = offLt[sb], posEq = offEq[sb];
            for (uint chunk = start; chunk < end; chunk += RS_SIMD) {
                uint i = chunk + lane;
                bool isLt = false, isEq = false;
                \(K) key = (\(K))0;
                if (i < end && (!hasValidity || bit_get(validity, i))) {
                    key = tk_map(vals[i], inv);
                    isLt = key < loKey;
                    isEq = !isLt && key <= hiKey;
                }
                // A refinement round runs over an already compacted array, where the row a slot stands for
                // is the payload, not the slot index.
                uint row = i;
                if (hasSrcRows && (isLt || isEq)) row = srcRows[i];
                uint cl = isLt ? 1u : 0u, ce = isEq ? 1u : 0u;
                uint pl = simd_prefix_exclusive_sum(cl), pe = simd_prefix_exclusive_sum(ce);
                uint tl = simd_sum(cl), te = simd_sum(ce);
                // `wantLt` is off when only the candidate window is wanted (the k-th key, not the k best
                // rows): the winners are still counted, so the window lands at offset 0, but not written.
                if (isLt && wantLt) { uint p = posLt + pl; outKeys[p] = key; outRows[p] = row; }
                if (isEq) { uint p = posEq + pe; if (p < limitEq) { outKeys[p] = key; outRows[p] = row; } }
                posLt += tl; posEq += te;
            }
        }

        // Finishes a selection that fits in one threadgroup: an ascending bitonic sort of the compacted
        // (key, slot) pairs, then the first k rows. The slot index is the tie-break, which is exactly the
        // stability the total order wants, and the whole ordering costs one dispatch instead of a radix sort.
        kernel void rs_sort_small(device const \(K)* keys [[buffer(0)]],
                                  device const uint* rows [[buffer(1)]],
                                  constant uint& m [[buffer(2)]],
                                  constant uint& cap [[buffer(3)]],
                                  constant uint& k [[buffer(4)]],
                                  device int* out [[buffer(5)]],
                                  threadgroup \(K)* bk [[threadgroup(0)]],
                                  threadgroup uint* bp [[threadgroup(1)]],
                                  uint lid [[thread_index_in_threadgroup]]) {
            for (uint i = lid; i < cap; i += TG) { bk[i] = \(K == "ulong" ? "ULONG_MAX" : "UINT_MAX"); bp[i] = 0xFFFFFFFFu; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lid; i < m; i += TG) { bk[i] = keys[i]; bp[i] = i; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint len = 2u; len <= cap; len <<= 1) {
                for (uint step = len >> 1; step > 0u; step >>= 1) {
                    for (uint t = lid; t < (cap >> 1); t += TG) {
                        uint low = t & (step - 1u);
                        uint i0 = ((t - low) << 1) + low;
                        uint i1 = i0 + step;
                        bool up = ((i0 & len) == 0u);
                        \(K) a = bk[i0], b = bk[i1];
                        uint ai = bp[i0], bi = bp[i1];
                        bool lt = (a < b) || (a == b && ai < bi);
                        if (up ? !lt : lt) { bk[i0] = b; bp[i0] = bi; bk[i1] = a; bp[i1] = ai; }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            for (uint j = lid; j < k; j += TG) out[j] = (int)rows[bp[j]];
        }
        """
    }
}
