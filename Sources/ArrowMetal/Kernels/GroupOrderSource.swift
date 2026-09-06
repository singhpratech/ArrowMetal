import Foundation

/// MSL for the **counting sort by group id** that puts every row of a group together.
///
/// `GroupBy.segments()` used to get this ordering from a full stable GPU radix argsort of the key
/// column — four passes for a 32-bit key, each of them a histogram, a scan and a scatter of the keys
/// *and* the payload. But the keys are already dense ids in `[0, K)`, so one counting pass is enough:
/// count the rows of every group, scan the counts into the start of each group's run, and scatter the
/// row indices into it. That is one pass over the keys instead of four over keys and payload.
///
/// Two scatters, because keeping the sort **stable** (rows inside a group must stay in row order, which
/// is what `hash_list` and `hash_first` mean by "first") costs differently at the two ends of the
/// cardinality range:
///
/// - **Chunked** (`cs_hist_blocks` + `cs_scatter`), for a group count small enough that a per-block
///   histogram fits: each block gets its own reserved slice of every group's run, so a block can fill
///   its slice in row order with no atomics at all. Stability is exact and free.
/// - **Atomic** (`cs_scatter_atomic` + `cs_fix` / `cs_fix_tg`), for a group count too large for that
///   table: rows land in their group in arbitrary order through one atomic bump per row, and a second
///   kernel sorts each run by row index. That is only cheap while the runs are short, which is exactly
///   the case that forced this path, so the caller checks the longest run first and falls back to the
///   argsort when it is not.
///
/// Nulls and out-of-range keys are dropped from the order, as `seg_bounds` dropped them before.
enum GroupOrderSource {

    /// Entries of the per-block histogram the chunked path is allowed to allocate (4 bytes each).
    static let chunkedBudget = 16 * 1024 * 1024
    /// Longest run `cs_fix` will insertion-sort in one thread, and `cs_fix_tg` in one threadgroup.
    static let maxThreadFix = 32
    static let maxGroupFix = 1024

    static func source(KT: String) -> String { KernelSource.prelude + """

    #define MAXK 1024u
    #define NOKEY 0xFFFFFFFFu
    #define SUBBLOCKS (TG / 32u)

    // Resolves row `i` to its group id, or NOKEY when the row takes part in no group.
    inline uint cs_key(device const \(KT)* keys, device const uchar* kvalid, uint flags, uint K, uint i) {
        if ((flags & 1u) && !bit_get(kvalid, i)) return NOKEY;
        long kk = (long)keys[i];
        if (kk < 0 || kk >= (long)K) return NOKEY;
        return (uint)kk;
    }

    kernel void cs_zero(device uint* p [[buffer(0)]], constant uint& n [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) { if (i < n) p[i] = 0u; }

    kernel void cs_copy(device const uint* src [[buffer(0)]], device uint* dst [[buffer(1)]],
                        constant uint& n [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) { if (i < n) dst[i] = src[i]; }

    // ---------------------------------------------------------------- histograms

    // Rows per group, one global table. The only histogram the atomic path needs.
    kernel void cs_hist_dev(device const \(KT)* keys [[buffer(0)]],
                            device const uchar* kvalid [[buffer(1)]],
                            constant uint& n [[buffer(2)]],
                            constant uint& flags [[buffer(3)]],
                            constant uint& K [[buffer(4)]],
                            constant uint& chunk [[buffer(5)]],
                            device atomic_uint* total [[buffer(6)]],
                            uint lid [[thread_index_in_threadgroup]],
                            uint tgid [[threadgroup_position_in_grid]]) {
        uint start = tgid * chunk, end = min(n, start + chunk);
        for (uint i = start + lid; i < end; i += TG) {
            uint k = cs_key(keys, kvalid, flags, K, i);
            if (k == NOKEY) continue;
            atomic_fetch_add_explicit(&total[k], 1u, memory_order_relaxed);
        }
    }

    // Rows per (block, group), block-major so a block's slice of the table is contiguous. A "block"
    // here is one **simdgroup**, not one threadgroup, because the scatter below ranks a row against the
    // rows of its own block and doing that across 32 lanes instead of 256 is eight times less work.
    kernel void cs_hist_blocks(device const \(KT)* keys [[buffer(0)]],
                               device const uchar* kvalid [[buffer(1)]],
                               constant uint& n [[buffer(2)]],
                               constant uint& flags [[buffer(3)]],
                               constant uint& K [[buffer(4)]],
                               constant uint& chunk [[buffer(5)]],
                               device atomic_uint* hist [[buffer(6)]],
                               uint tgid [[threadgroup_position_in_grid]],
                               uint sgid [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
        uint lb = tgid * SUBBLOCKS + sgid;
        uint base = lb * K;
        uint start = lb * chunk, end = min(n, start + chunk);
        for (uint i = start + lane; i < end; i += 32u) {
            uint k = cs_key(keys, kvalid, flags, K, i);
            if (k == NOKEY) continue;
            atomic_fetch_add_explicit(&hist[base + k], 1u, memory_order_relaxed);
        }
    }

    // ---------------------------------------------------------------- scan

    // Rows per group from the per-block table.
    kernel void cs_totals(device const uint* hist [[buffer(0)]], constant uint& K [[buffer(1)]],
                          constant uint& B [[buffer(2)]], device uint* total [[buffer(3)]],
                          uint k [[thread_position_in_grid]]) {
        if (k >= K) return;
        uint s = 0;
        for (uint b = 0; b < B; b++) s += hist[b * K + k];
        total[k] = s;
    }

    // Exclusive scan of the group sizes into [start, end) per group, plus the longest run. One
    // threadgroup: the group count is at most a few million and each thread walks a contiguous slice.
    kernel void cs_scan(device const uint* total [[buffer(0)]], constant uint& K [[buffer(1)]],
                        device uint* segStart [[buffer(2)]], device uint* segEnd [[buffer(3)]],
                        device atomic_uint* maxRun [[buffer(4)]],
                        uint lid [[thread_index_in_threadgroup]],
                        uint sgid [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
        threadgroup uint simdTotals[32];
        uint per = (K + TG - 1u) / TG;
        uint lo = min(K, lid * per), hi = min(K, lo + per);
        uint local = 0u, mx = 0u;
        for (uint i = lo; i < hi; i++) { uint c = total[i]; local += c; mx = max(mx, c); }
        uint pre = simd_prefix_exclusive_sum(local);
        uint t = simd_sum(local);
        if (lane == 0u) simdTotals[sgid] = t;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint prefix = 0u;
        for (uint q = 0u; q < sgid; q++) prefix += simdTotals[q];
        uint run = prefix + pre;
        for (uint i = lo; i < hi; i++) { uint c = total[i]; segStart[i] = run; run += c; segEnd[i] = run; }
        if (mx) atomic_fetch_max_explicit(maxRun, mx, memory_order_relaxed);
    }

    // Each block's cursor into every group's run: the group's start plus what earlier blocks hold.
    kernel void cs_block_offsets(device uint* hist [[buffer(0)]], device const uint* segStart [[buffer(1)]],
                                 constant uint& K [[buffer(2)]], constant uint& B [[buffer(3)]],
                                 uint k [[thread_position_in_grid]]) {
        if (k >= K) return;
        uint run = segStart[k];
        for (uint b = 0; b < B; b++) { uint c = hist[b * K + k]; hist[b * K + k] = run; run += c; }
    }

    // ---------------------------------------------------------------- scatter

    // Stable: a block owns a reserved slice of every group's run and fills it in row order. Within a
    // sub-chunk of 32 rows a lane's rank among the earlier rows of the same group comes from one sweep
    // of its simdgroup's key array, and the last row of each group advances the cursor. Nothing crosses
    // simdgroups, so the only synchronisation is a simdgroup barrier.
    kernel void cs_scatter(device const \(KT)* keys [[buffer(0)]],
                           device const uchar* kvalid [[buffer(1)]],
                           constant uint& n [[buffer(2)]],
                           constant uint& flags [[buffer(3)]],
                           constant uint& K [[buffer(4)]],
                           constant uint& chunk [[buffer(5)]],
                           device uint* cursor [[buffer(6)]],
                           device int* ord [[buffer(7)]],
                           uint tgid [[threadgroup_position_in_grid]],
                           uint sgid [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
        threadgroup uint skey[TG];
        uint lb = tgid * SUBBLOCKS + sgid;
        uint base = lb * K, off = sgid * 32u;
        uint start = lb * chunk, end = min(n, start + chunk);
        for (uint c = start; c < end; c += 32u) {
            uint i = c + lane;
            uint k = (i < end) ? cs_key(keys, kvalid, flags, K, i) : NOKEY;
            skey[off + lane] = k;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            uint pos = 0u;
            bool last = false;
            bool active = (k != NOKEY);
            if (active) {
                uint rank = 0u;
                last = true;
                for (uint j = 0u; j < 32u; j++) {
                    uint kj = skey[off + j];
                    if (kj != k) continue;
                    if (j < lane) rank++;
                    else if (j > lane) last = false;
                }
                pos = cursor[base + k] + rank;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (active) {
                ord[pos] = (int)i;
                if (last) cursor[base + k] = pos + 1u;
            }
            simdgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
    }

    // Unstable: one atomic bump per row. `cs_fix` / `cs_fix_tg` restore the row order afterwards.
    kernel void cs_scatter_atomic(device const \(KT)* keys [[buffer(0)]],
                                  device const uchar* kvalid [[buffer(1)]],
                                  constant uint& n [[buffer(2)]],
                                  constant uint& flags [[buffer(3)]],
                                  constant uint& K [[buffer(4)]],
                                  constant uint& chunk [[buffer(5)]],
                                  device atomic_uint* cursor [[buffer(6)]],
                                  device int* ord [[buffer(7)]],
                                  uint lid [[thread_index_in_threadgroup]],
                                  uint tgid [[threadgroup_position_in_grid]]) {
        uint start = tgid * chunk, end = min(n, start + chunk);
        for (uint i = start + lid; i < end; i += TG) {
            uint k = cs_key(keys, kvalid, flags, K, i);
            if (k == NOKEY) continue;
            uint pos = atomic_fetch_add_explicit(&cursor[k], 1u, memory_order_relaxed);
            ord[pos] = (int)i;
        }
    }

    // One thread per group: insertion sort of a short run back into row order.
    kernel void cs_fix(device const uint* segStart [[buffer(0)]], device const uint* segEnd [[buffer(1)]],
                       device int* ord [[buffer(2)]], constant uint& K [[buffer(3)]],
                       uint k [[thread_position_in_grid]]) {
        if (k >= K) return;
        uint s = segStart[k], e = segEnd[k];
        for (uint i = s + 1u; i < e; i++) {
            int v = ord[i];
            uint j = i;
            while (j > s && ord[j - 1u] > v) { ord[j] = ord[j - 1u]; j--; }
            ord[j] = v;
        }
    }

    // One threadgroup per group: a bitonic sort of a run of at most 1024 rows in threadgroup memory.
    kernel void cs_fix_tg(device const uint* segStart [[buffer(0)]], device const uint* segEnd [[buffer(1)]],
                          device int* ord [[buffer(2)]], constant uint& K [[buffer(3)]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint tgid [[threadgroup_position_in_grid]]) {
        threadgroup int buf[1024];
        uint k = tgid;
        if (k >= K) return;
        uint s = segStart[k], e = segEnd[k], m = e - s;
        if (m < 2u) return;
        uint p = 1u;
        while (p < m) p <<= 1;
        for (uint i = lid; i < p; i += TG) buf[i] = (i < m) ? ord[s + i] : INT_MAX;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint size = 2u; size <= p; size <<= 1) {
            for (uint stride = size >> 1; stride > 0u; stride >>= 1) {
                for (uint i = lid; i < p; i += TG) {
                    uint partner = i ^ stride;
                    if (partner > i) {
                        bool up = ((i & size) == 0u);
                        int a = buf[i], b = buf[partner];
                        if ((a > b) == up) { buf[i] = b; buf[partner] = a; }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        for (uint i = lid; i < m; i += TG) ord[s + i] = buf[i];
    }
    """ }
}
