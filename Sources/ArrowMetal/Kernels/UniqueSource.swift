import Foundation

/// MSL for distinct-value extraction over a sorted order.
///
/// The pipeline is: normalise float values so that raw bit equality means Arrow value equality
/// (every NaN collapses to one pattern, -0 becomes +0), argsort, then mark run boundaries in the
/// sorted order, scan the marks (two-level, the same shape as the string offset scan) to get a
/// dense rank per distinct value, and scatter those ranks back to the original positions.
///
/// `U` is the unsigned integer type of the same width as the element type, so the boundary test is a
/// plain bit comparison and one kernel covers signed, unsigned and floating point columns.
enum UniqueSource {
    /// MSL unsigned type whose width matches an Arrow primitive; equality on it is bit equality.
    static func unsignedType(width: Int) -> String {
        switch width {
        case 1: return "uchar"
        case 2: return "ushort"
        case 4: return "uint"
        default: return "ulong"
        }
    }

    static func source(U: String) -> String { KernelSource.prelude + """

    kernel void uq_iota(device int* out [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
        if (i < *nPtr) out[i] = (int)i;
    }
    // Arrow value equality for float32 as a bit pattern: every NaN becomes one quiet NaN, -0 becomes +0.
    // Collapsing NaN also keeps every NaN adjacent in the total order the radix sort uses (negative NaNs
    // would otherwise sort before -inf and positive ones after +inf, splitting the run in two).
    kernel void uq_norm_f32(device const uint* a [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                            device uint* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        uint b = a[i], m = b & 0x7FFFFFFFu;
        if (m > 0x7F800000u) b = 0x7FC00000u;
        else if (m == 0u) b = 0u;
        out[i] = b;
    }
    // Same for float64, handled as a raw 64-bit pattern (Metal has no double).
    kernel void uq_norm_f64(device const ulong* a [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                            device ulong* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        ulong b = a[i], m = b & 0x7FFFFFFFFFFFFFFFul;
        if (m > 0x7FF0000000000000ul) b = 0x7FF8000000000000ul;
        else if (m == 0ul) b = 0ul;
        out[i] = b;
    }
    // Run boundaries in the sorted order: position 0 always starts a run, later positions start one when
    // the value differs from the previous sorted value. Written twice: as bytes for the bitmap packer that
    // feeds `filter`, and as int32 for the rank scan. The int32 copy drops the mark at position 0, because
    // the rank of a sorted position is the *inclusive* scan of the marks after it: rank 0 for the first run.
    kernel void uq_mark(device const \(U)* vals [[buffer(0)]], device const int* ord [[buffer(1)]],
                        device const uint* nPtr [[buffer(2)]], device uchar* markBytes [[buffer(3)]],
                        device int* markInts [[buffer(4)]], uint i [[thread_position_in_grid]]) {
        uint n = *nPtr;
        if (i >= n) return;
        uchar f = (i == 0u || vals[ord[i]] != vals[ord[i - 1u]]) ? 1 : 0;
        markBytes[i] = f;
        markInts[i] = (i == 0u) ? 0 : (int)f;
    }
    // Length of each run from the sorted positions of the run starts; the last run ends at `total`.
    kernel void uq_run_lengths(device const int* pos [[buffer(0)]], device const uint* uPtr [[buffer(1)]],
                               constant uint& total [[buffer(2)]], device long* out [[buffer(3)]],
                               uint j [[thread_position_in_grid]]) {
        uint u = *uPtr;
        if (j >= u) return;
        int end = (j + 1u < u) ? pos[j + 1u] : (int)total;
        out[j] = (long)(end - pos[j]);
    }
    // Ranks back to the original row of each sorted position. `ranks` is the exclusive scan of `marks`, so
    // adding the position's own mark makes it the inclusive scan: the 0-based index of its distinct value.
    kernel void uq_scatter_codes(device const int* ord [[buffer(0)]], device const int* ranks [[buffer(1)]],
                                 device const int* marks [[buffer(2)]], device const uint* nPtr [[buffer(3)]],
                                 device int* codes [[buffer(4)]], uint i [[thread_position_in_grid]]) {
        if (i < *nPtr) codes[ord[i]] = ranks[i] + marks[i];
    }
    // Two-level exclusive scan of the int32 marks: per-block scan plus block totals, a single-threadgroup
    // scan of those totals, then the block offset added back.
    kernel void uq_scan_block(device const int* vals [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                              device int* out [[buffer(2)]], device int* blockTotals [[buffer(3)]],
                              uint i [[thread_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                              uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
        threadgroup int simdTotals[32];
        uint n = *nPtr;
        int v = (i < n) ? vals[i] : 0;
        int pre = simd_prefix_exclusive_sum(v);
        int t = simd_sum(v);
        if (lane == 0) simdTotals[sgid] = t;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int prefix = 0;
        for (uint k = 0; k < sgid; k++) prefix += simdTotals[k];
        if (i < n) out[i] = prefix + pre;
        if (lid == TG - 1) { int total = 0; for (uint k = 0; k < TG / 32u; k++) total += simdTotals[k]; blockTotals[tgid] = total; }
    }
    kernel void uq_scan_totals(device int* blockTotals [[buffer(0)]], constant uint& blocks [[buffer(1)]],
                               uint lid [[thread_index_in_threadgroup]], uint sgid [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
        threadgroup int simdTotals[32];
        uint per = (blocks + TG - 1) / TG;
        uint lo = lid * per, hi = min(blocks, lo + per);
        int local = 0;
        for (uint b = lo; b < hi; b++) local += blockTotals[b];
        int pre = simd_prefix_exclusive_sum(local);
        int t = simd_sum(local);
        if (lane == 0) simdTotals[sgid] = t;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int prefix = 0;
        for (uint k = 0; k < sgid; k++) prefix += simdTotals[k];
        int run = prefix + pre;
        for (uint b = lo; b < hi; b++) { int c = blockTotals[b]; blockTotals[b] = run; run += c; }
    }
    kernel void uq_scan_add(device int* out [[buffer(0)]], device const int* blockTotals [[buffer(1)]],
                            device const uint* nPtr [[buffer(2)]], uint i [[thread_position_in_grid]],
                            uint tgid [[threadgroup_position_in_grid]]) {
        if (i < *nPtr) out[i] += blockTotals[tgid];
    }
    """ }
}
