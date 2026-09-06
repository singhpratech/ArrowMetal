import Foundation

/// MSL for the window, shift, pairwise and rolling-window kernels.
///
/// Three families of source live here, each compiled once per type it is specialised for:
///
///   * `ranking(U:)` — everything that works in *sorted position* space: the float normaliser, the run
///     marks, a two-level int32 scan of those marks, the run bounds, and the scatters that write a rank
///     back to the row it came from. `U` is the unsigned integer type of the element width, so the
///     tie test is a plain bit comparison and one kernel covers signed, unsigned and floating point.
///   * `values(...)` — element-wise kernels over one value type: `shift`, `pairwise_diff`, the rolling
///     min/max scan, the rolling sum from a prefix array, and the widening to binary64.
///   * `doubleOps` — kernels whose value type is always a binary64 bit pattern: the running mean, the
///     rolling mean and a small int32 fill.
///
/// Metal has no `double`, so every float64 result is produced as a raw 64-bit pattern through the
/// software arithmetic in `DoubleMath` (`d_add`, `d_sub`, `d_div`), which is correctly rounded. The
/// widening conversions below (`d_from_i64`, `d_from_u64`, `d_from_f32`) are exact or correctly rounded
/// too, so a float64 window result differs from a sequential host computation only where the algorithm
/// itself reassociates — the prefix-sum trick for rolling sums, and the two-level scan behind the
/// running mean.
enum WindowSource {
    /// Software binary64 plus exact widening conversions into it.
    static let doublePrelude = DoubleMath.msl + """

    // Packs a sign and a magnitude into binary64, rounding to nearest (ties to even) when the magnitude
    // needs more than 53 bits. `d_finish` wants the leading one at bit 55 (three guard bits, bit 0 sticky).
    inline ulong d_from_mag(ulong s, ulong u) {
        if (u == 0ul) return s << 63;
        uint hi = (uint)(u >> 32);
        uint k = (hi != 0u) ? (63u - clz(hi)) : (31u - clz((uint)u));
        ulong m;
        if (k <= 55u) { m = u << (55u - k); }
        else { uint sh = k - 55u; ulong lost = u & ((1ul << sh) - 1ul); m = (u >> sh) | (lost ? 1ul : 0ul); }
        return d_finish(s, (long)k + 1023L, m);
    }
    inline ulong d_from_u64(ulong u) { return d_from_mag(0ul, u); }
    inline ulong d_from_i64(long v) {
        ulong u = (ulong)v;
        return (v < 0) ? d_from_mag(1ul, ~u + 1ul) : d_from_mag(0ul, u);
    }
    inline ulong d_from_uint(uint v) { return d_from_mag(0ul, (ulong)v); }
    // float32 -> float64, exact for every input (subnormals included). NaN payloads collapse to one quiet NaN.
    inline ulong d_from_f32(float f) {
        uint b = as_type<uint>(f);
        ulong s = (ulong)(b >> 31);
        uint e = (b >> 23) & 0xFFu, m = b & 0x7FFFFFu;
        if (e == 0xFFu) return (s << 63) | (m ? D_QNAN : D_INF);
        if (e == 0u) {
            if (m == 0u) return s << 63;
            uint k = 31u - clz(m);                       // subnormal: value = m * 2^-149
            return d_finish(s, (long)k - 149L + 1023L, ((ulong)m) << (55u - k));
        }
        return (s << 63) | ((ulong)(e - 127u + 1023u) << 52) | (((ulong)m) << 29);
    }
    """

    /// The two-level exclusive scan of the int32 run marks, shared by every ranking kernel.
    /// Same shape as the scans in `UniqueSource` and `CumulativeSource`: block scan, a single-threadgroup
    /// scan of the block totals, then the block offset added back.
    private static let scan = """

    kernel void win_scan_block(device const int* vals [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
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
    kernel void win_scan_totals(device int* blockTotals [[buffer(0)]], constant uint& blocks [[buffer(1)]],
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
    kernel void win_scan_add(device int* out [[buffer(0)]], device const int* blockTotals [[buffer(1)]],
                             device const uint* nPtr [[buffer(2)]], uint i [[thread_position_in_grid]],
                             uint tgid [[threadgroup_position_in_grid]]) {
        if (i < *nPtr) out[i] += blockTotals[tgid];
    }
    """

    /// Ranking kernels for an element width whose unsigned type is `U`.
    ///
    /// The order they run in is: normalise (floats only), argsort (in `Sort.swift`), mark the run
    /// boundaries of the sorted order, scan the marks, record where each run starts and ends, scatter.
    /// The sorted order is the *full* one, nulls included: `argsort` puts them last, and the marks
    /// kernel makes all of them one tie group, which is what SQL window functions do with NULLS LAST.
    static func ranking(U: String) -> String { KernelSource.prelude + doublePrelude + scan + """

    // Arrow value equality as bit equality: every NaN becomes one quiet NaN, -0 becomes +0. Collapsing
    // NaN also keeps every NaN in one run instead of splitting it across both ends of the total order.
    kernel void win_norm_f32(device const uint* a [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                             device uint* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        uint b = a[i], m = b & 0x7FFFFFFFu;
        if (m > 0x7F800000u) b = 0x7FC00000u; else if (m == 0u) b = 0u;
        out[i] = b;
    }
    kernel void win_norm_f64(device const ulong* a [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                             device ulong* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        ulong b = a[i], m = b & 0x7FFFFFFFFFFFFFFFul;
        if (m > 0x7FF0000000000000ul) b = 0x7FF8000000000000ul; else if (m == 0ul) b = 0ul;
        out[i] = b;
    }
    // Run boundaries over the full sorted order. Sorted positions [0, m) hold the non-null values and
    // [m, n) the nulls; the nulls are one tie group, so only position m starts a run among them.
    // Position 0 writes 0 because the rank of a sorted position is the *exclusive* scan of the marks
    // plus its own mark: the first run must come out as 0.
    kernel void win_marks(device const \(U)* vals [[buffer(0)]], device const int* ord [[buffer(1)]],
                          device const uint* nPtr [[buffer(2)]], constant uint& m [[buffer(3)]],
                          device int* marks [[buffer(4)]], uint i [[thread_position_in_grid]]) {
        uint n = *nPtr;
        if (i >= n) return;
        int f;
        if (i == 0u) f = 0;
        else if (i >= m) f = (i == m) ? 1 : 0;
        else f = (vals[ord[i]] != vals[ord[i - 1u]]) ? 1 : 0;
        marks[i] = f;
    }
    // First and one-past-last sorted position of each run, indexed by the run's dense rank.
    kernel void win_run_bounds(device const int* marks [[buffer(0)]], device const int* ranks [[buffer(1)]],
                               device const uint* nPtr [[buffer(2)]], device int* startPos [[buffer(3)]],
                               device int* endPos [[buffer(4)]], uint i [[thread_position_in_grid]]) {
        uint n = *nPtr;
        if (i >= n) return;
        int d = ranks[i] + marks[i];
        if (i == 0u || marks[i] != 0) startPos[d] = (int)i;
        if (i + 1u == n || marks[i + 1u] != 0) endPos[d] = (int)(i + 1u);
    }
    // mode 0 row_number (1-based sorted position), 1 rank (the run's first position + 1), 2 dense_rank.
    kernel void win_scatter_int(device const int* ord [[buffer(0)]], device const int* ranks [[buffer(1)]],
                                device const int* marks [[buffer(2)]], device const int* startPos [[buffer(3)]],
                                device const uint* nPtr [[buffer(4)]], constant uint& mode [[buffer(5)]],
                                device int* out [[buffer(6)]], uint i [[thread_position_in_grid]]) {
        uint n = *nPtr;
        if (i >= n) return;
        int d = ranks[i] + marks[i];
        int v = (mode == 0u) ? (int)(i + 1u) : ((mode == 1u) ? (startPos[d] + 1) : (d + 1));
        out[ord[i]] = v;
    }
    // mode 0 percent_rank = (rank - 1) / (n - 1), 1 cume_dist = (rows at or before this value) / n.
    kernel void win_scatter_double(device const int* ord [[buffer(0)]], device const int* ranks [[buffer(1)]],
                                   device const int* marks [[buffer(2)]], device const int* startPos [[buffer(3)]],
                                   device const int* endPos [[buffer(4)]], device const uint* nPtr [[buffer(5)]],
                                   constant uint& mode [[buffer(6)]], device ulong* out [[buffer(7)]],
                                   uint i [[thread_position_in_grid]]) {
        uint n = *nPtr;
        if (i >= n) return;
        int d = ranks[i] + marks[i];
        ulong v;
        if (mode == 0u) v = (n <= 1u) ? 0ul : d_div(d_from_uint((uint)startPos[d]), d_from_uint(n - 1u));
        else v = d_div(d_from_uint((uint)endPos[d]), d_from_uint(n));
        out[ord[i]] = v;
    }
    """ }

    /// Which arithmetic the element-wise kernels use for a value type.
    enum Kind { case integer, float32, float64 }

    /// Element-wise window kernels for one value type. `V` is the MSL type of the values buffer
    /// (float64 travels as `ulong` bit patterns), `unsigned` its unsigned counterpart for wrapping
    /// integer arithmetic, and `identMin` / `identMax` the neutral elements of min and max.
    static func values(V: String, kind: Kind, unsigned: String, identMin: String, identMax: String) -> String {
        let sub: String, isNaN: String, less: String, toF64: String
        switch kind {
        case .integer:
            sub = "(\(V))((\(unsigned))a - (\(unsigned))b)"
            isNaN = "false"
            less = "a < b"
            toF64 = unsigned == V ? "d_from_u64((ulong)x)" : "d_from_i64((long)x)"
        case .float32:
            sub = "a - b"; isNaN = "isnan(x)"; less = "a < b"; toF64 = "d_from_f32(x)"
        case .float64:
            // Values are raw binary64 patterns: subtract through the software adder, order through `d_key`.
            sub = "d_sub(a, b)"; isNaN = "d_is_nan(x)"; less = "d_key((long)a) < d_key((long)b)"; toF64 = "x"
        }
        return KernelSource.prelude + doublePrelude + """

        inline \(V) win_sub(\(V) a, \(V) b) { return \(sub); }
        inline bool win_isnan(\(V) x) { return \(isNaN); }
        inline bool win_lt(\(V) a, \(V) b) { return \(less); }

        // lag / lead: out[i] = a[i - by], with `fill` (or a null) outside the array.
        kernel void win_shift(device const \(V)* vals [[buffer(0)]], device const uchar* validity [[buffer(1)]],
                              device const uint* nPtr [[buffer(2)]], constant int& by [[buffer(3)]],
                              constant uint& hasValidity [[buffer(4)]], constant uint& hasFill [[buffer(5)]],
                              constant \(V)& fill [[buffer(6)]], device \(V)* out [[buffer(7)]],
                              device uchar* outValid [[buffer(8)]], uint i [[thread_position_in_grid]]) {
            uint n = *nPtr;
            if (i >= n) return;
            long j = (long)i - (long)by;
            if (j < 0 || j >= (long)n) { out[i] = fill; outValid[i] = (hasFill != 0u) ? 1 : 0; return; }
            bool ok = (hasValidity == 0u) || bit_get(validity, (uint)j);
            out[i] = ok ? vals[(uint)j] : (\(V))0;
            outValid[i] = ok ? 1 : 0;
        }
        // Arrow `pairwise_diff`: out[i] = a[i] - a[i - period], null where either side is null or missing.
        kernel void win_pairwise_diff(device const \(V)* vals [[buffer(0)]], device const uchar* validity [[buffer(1)]],
                                      device const uint* nPtr [[buffer(2)]], constant int& period [[buffer(3)]],
                                      constant uint& hasValidity [[buffer(4)]], device \(V)* out [[buffer(5)]],
                                      device uchar* outValid [[buffer(6)]], uint i [[thread_position_in_grid]]) {
            uint n = *nPtr;
            if (i >= n) return;
            long j = (long)i - (long)period;
            if (j < 0 || j >= (long)n) { out[i] = (\(V))0; outValid[i] = 0; return; }
            bool ok = (hasValidity == 0u) || (bit_get(validity, i) && bit_get(validity, (uint)j));
            out[i] = ok ? win_sub(vals[i], vals[(uint)j]) : (\(V))0;
            outValid[i] = ok ? 1 : 0;
        }
        // Trailing rolling min / max: one thread per output, scanning the w inputs that end at it.
        // NaN is skipped, as it is by the min/max reductions and the cumulative functions.
        kernel void win_rolling_min(device const \(V)* vals [[buffer(0)]], device const uchar* validity [[buffer(1)]],
                                    device const uint* nPtr [[buffer(2)]], constant uint& w [[buffer(3)]],
                                    constant uint& minP [[buffer(4)]], constant uint& hasValidity [[buffer(5)]],
                                    device \(V)* out [[buffer(6)]], device uchar* outValid [[buffer(7)]],
                                    uint i [[thread_position_in_grid]]) {
            uint n = *nPtr;
            if (i >= n) return;
            uint lo = (i + 1u > w) ? (i + 1u - w) : 0u;
            uint cnt = 0u;
            \(V) acc = \(identMin);
            for (uint j = lo; j <= i; j++) {
                if (hasValidity != 0u && !bit_get(validity, j)) continue;
                cnt++;
                \(V) x = vals[j];
                if (win_isnan(x)) continue;
                if (win_lt(x, acc)) acc = x;
            }
            bool ok = cnt >= minP;
            out[i] = ok ? acc : (\(V))0;
            outValid[i] = ok ? 1 : 0;
        }
        kernel void win_rolling_max(device const \(V)* vals [[buffer(0)]], device const uchar* validity [[buffer(1)]],
                                    device const uint* nPtr [[buffer(2)]], constant uint& w [[buffer(3)]],
                                    constant uint& minP [[buffer(4)]], constant uint& hasValidity [[buffer(5)]],
                                    device \(V)* out [[buffer(6)]], device uchar* outValid [[buffer(7)]],
                                    uint i [[thread_position_in_grid]]) {
            uint n = *nPtr;
            if (i >= n) return;
            uint lo = (i + 1u > w) ? (i + 1u - w) : 0u;
            uint cnt = 0u;
            \(V) acc = \(identMax);
            for (uint j = lo; j <= i; j++) {
                if (hasValidity != 0u && !bit_get(validity, j)) continue;
                cnt++;
                \(V) x = vals[j];
                if (win_isnan(x)) continue;
                if (win_lt(acc, x)) acc = x;
            }
            bool ok = cnt >= minP;
            out[i] = ok ? acc : (\(V))0;
            outValid[i] = ok ? 1 : 0;
        }
        // Trailing rolling sum in O(n): the difference of two inclusive prefix sums, with the number of
        // non-null rows in the window coming from a parallel int32 prefix count.
        kernel void win_rolling_sum(device const \(V)* pre [[buffer(0)]], device const int* cnt [[buffer(1)]],
                                    device const uint* nPtr [[buffer(2)]], constant uint& w [[buffer(3)]],
                                    constant uint& minP [[buffer(4)]], device \(V)* out [[buffer(5)]],
                                    device uchar* outValid [[buffer(6)]], uint i [[thread_position_in_grid]]) {
            uint n = *nPtr;
            if (i >= n) return;
            bool hasPrev = (i + 1u > w);
            uint p = hasPrev ? (i - w) : 0u;
            int c = cnt[i] - (hasPrev ? cnt[p] : 0);
            if (c < (int)minP) { out[i] = (\(V))0; outValid[i] = 0; return; }
            out[i] = hasPrev ? win_sub(pre[i], pre[p]) : pre[i];
            outValid[i] = 1;
        }
        // Widening to binary64 bit patterns, for the running and rolling means.
        kernel void win_to_f64(device const \(V)* vals [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                               device ulong* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
            if (i >= *nPtr) return;
            \(V) x = vals[i];
            out[i] = \(toF64);
        }
        """
    }

    /// Kernels whose value type is always a binary64 bit pattern.
    static let doubleOps: String = KernelSource.prelude + doublePrelude + """

    kernel void win_fill_i32(device int* out [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                             constant int& v [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i < *nPtr) out[i] = v;
    }
    // Running mean: the running binary64 sum over the running count of non-null rows.
    kernel void win_div_count(device const ulong* sums [[buffer(0)]], device const int* cnt [[buffer(1)]],
                              device const uint* nPtr [[buffer(2)]], device ulong* out [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        int c = cnt[i];
        out[i] = (c > 0) ? d_div(sums[i], d_from_uint((uint)c)) : D_QNAN;
    }
    // Trailing rolling mean from the same two prefix arrays as the rolling sum.
    kernel void win_rolling_mean(device const ulong* pre [[buffer(0)]], device const int* cnt [[buffer(1)]],
                                 device const uint* nPtr [[buffer(2)]], constant uint& w [[buffer(3)]],
                                 constant uint& minP [[buffer(4)]], device ulong* out [[buffer(5)]],
                                 device uchar* outValid [[buffer(6)]], uint i [[thread_position_in_grid]]) {
        uint n = *nPtr;
        if (i >= n) return;
        bool hasPrev = (i + 1u > w);
        uint p = hasPrev ? (i - w) : 0u;
        int c = cnt[i] - (hasPrev ? cnt[p] : 0);
        if (c < (int)minP || c <= 0) { out[i] = 0ul; outValid[i] = 0; return; }
        ulong s = hasPrev ? d_sub(pre[i], pre[p]) : pre[i];
        out[i] = d_div(s, d_from_uint((uint)c));
        outValid[i] = 1;
    }
    """
}
