import Foundation

/// Metal Shading Language for the aggregates that `Aggregates.swift` left out: the fused grouped
/// `min_max`, the grouped `product`, the third and fourth central moments (`skew` / `kurtosis`) in both
/// their scalar and grouped forms, and the per-group order statistic behind `hash_approximate_median`
/// and `hash_tdigest`.
///
/// Everything grouped here is **segmented**, in the sense `SegmentedSource` established: the rows are
/// argsorted by group id once, which makes each group one contiguous run, and a threadgroup reduces one
/// run. That is what makes 64-bit min/max and per-group order statistics possible at all — Metal has no
/// 64-bit atomics, and a median is not an atomic reduction in the first place.
///
/// | kernel | shape |
/// |---|---|
/// | `gx_bounds` | `[start, end)` per group from an id column already sorted by group |
/// | `gx_seg_minmax` | one threadgroup per group, both extremes from a single read of the values |
/// | `gx_seg_product` | one threadgroup per group, wrapping integer or reassociated float product |
/// | `gx_seg_moment34` | one threadgroup per group: sums of `dev^2`, `dev^3`, `dev^4` about a given mean |
/// | `gx_moment34` | the same three sums over the whole column, grid-strided |
/// | `gx_seg_pick` | the two values bracketing a quantile position inside each group |
///
/// The moment kernels accumulate in compensated (Neumaier) float pairs for integer and Float32 columns
/// and in software binary64 for Float64 ones, exactly as `agg_moment` does for the second moment.
enum AggregatesExtraSource {

    /// Kernels that do not depend on the element type.
    static let common = KernelSource.prelude + """

    // Segment bounds from an id column that is already in sorted-by-id order. Unlike `seg_bounds` there
    // is no `ord` indirection: the caller has materialised the permuted ids.
    kernel void gx_bounds(device const int* sortedIds [[buffer(0)]],
                          device const uint* nPtr [[buffer(1)]],
                          constant uint& K [[buffer(2)]],
                          device uint* segStart [[buffer(3)]],
                          device uint* segEnd [[buffer(4)]],
                          uint i [[thread_position_in_grid]]) {
        uint m = *nPtr;
        if (i >= m) return;
        long kk = (long)sortedIds[i];
        if (kk < 0 || kk >= (long)K) return;
        uint k = (uint)kk;
        bool first = (i == 0u) || ((long)sortedIds[i - 1u] != kk);
        bool last = (i + 1u == m) || ((long)sortedIds[i + 1u] != kk);
        if (first) segStart[k] = i;
        if (last) segEnd[k] = i + 1u;
    }

    // Exclusive prefix sum turned into int32 list offsets: offsets[0] = 0, offsets[k+1] = sum of the
    // first k + 1 lengths. `cum` is the inclusive scan the caller produced with `cumulativeSum`.
    kernel void gx_offsets(device const long* cum [[buffer(0)]],
                           device const uint* nPtr [[buffer(1)]],
                           device int* out [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
        uint K = *nPtr;
        if (i > K) return;
        out[i] = (i == 0u) ? 0 : (int)cum[i - 1u];
    }

    // Concatenates the run of every group, in group order, into one gather index. A group whose run is
    // empty contributes nothing, so the result is dense even when some groups have no rows and even when
    // the sorted order also holds rows whose key falls outside [0, K).
    kernel void gx_gather_runs(device const uint* segStart [[buffer(0)]],
                               device const uint* segEnd [[buffer(1)]],
                               device const int* ord [[buffer(2)]],
                               device const int* offsets [[buffer(3)]],
                               device const uint* nPtr [[buffer(4)]],
                               device int* out [[buffer(5)]],
                               uint lid [[thread_index_in_threadgroup]],
                               uint tgid [[threadgroup_position_in_grid]]) {
        uint K = *nPtr;
        uint k = tgid;
        if (k >= K) return;
        uint s = segStart[k], e = segEnd[k];
        uint base = (uint)offsets[k];
        for (uint t = s + lid; t < e; t += TG) out[base + (t - s)] = ord[t];
    }
    """

    /// How the deviation `x - mean` is formed, mirroring `AggregatesSource.MomentMode`.
    enum Moment {
        case integer, float, double
    }

    /// Full per-element-type source.
    ///
    /// - `T`: MSL type of the values buffer (Float64 arrives as `long` bit patterns).
    /// - `KEYACC` / `key`: the order-preserving accumulator the fused min/max reduces on.
    /// - `ACC` / `productMul`: the product accumulator and its combine.
    /// - `toFloat`: the value as a `float`, used by the grouped moment kernel.
    static func source(T: String, KEYACC: String, ACC: String, key: String, include: String,
                       minInit: String, maxInit: String, productInit: String, productMul: String,
                       productLoad: String, toFloat: String, moment: Moment,
                       extraPrelude: String = "") -> String {
        // Scalar moment body: the deviation, then its square, cube and fourth power.
        let scalarBody: String
        switch moment {
        case .integer:
            scalarBody = """
                float dev = (float)((long)vals[i] - mp.ipart) - mp.hi;
                float d2 = dev * dev;
                a2 = gx_addc(a2, d2); a3 = gx_addc(a3, d2 * dev); a4 = gx_addc(a4, d2 * d2);
            """
        case .float:
            scalarBody = """
                float dev = ((float)vals[i] - mp.hi) - mp.lo;
                float d2 = dev * dev;
                a2 = gx_addc(a2, d2); a3 = gx_addc(a3, d2 * dev); a4 = gx_addc(a4, d2 * d2);
            """
        case .double:
            scalarBody = """
                ulong dev = d_sub((ulong)vals[i], mp.bits);
                ulong d2 = d_mul(dev, dev);
                a2 = d_add(a2, d2); a3 = d_add(a3, d_mul(d2, dev)); a4 = d_add(a4, d_mul(d2, d2));
            """
        }
        let macc = moment == .double ? "ulong" : "float2"
        let minit = moment == .double ? "0ul" : "float2(0.0f, 0.0f)"
        let mmerge = moment == .double ? "d_add(x, y)" : "gx_merge(x, y)"
        let minclude = moment == .double ? "!d_isnan((long)vals[i])" : include

        return KernelSource.prelude + extraPrelude + """

        struct GxMeanParams { long ipart; ulong bits; float hi; float lo; };

        // Compensated (Neumaier) float addition; the pair is (sum, correction).
        inline float2 gx_addc(float2 acc, float x) {
            float s = acc.x + x;
            float c = (fabs(acc.x) >= fabs(x)) ? ((acc.x - s) + x) : ((x - s) + acc.x);
            return float2(s, acc.y + c);
        }
        inline float2 gx_merge(float2 a, float2 b) {
            float s = a.x + b.x;
            float c = (fabs(a.x) >= fabs(b.x)) ? ((a.x - s) + b.x) : ((b.x - s) + a.x);
            return float2(s, (a.y + b.y) + c);
        }

        // Arrow `hash_min_max`: one threadgroup per group, both extremes from one read of the values.
        kernel void gx_seg_minmax(device const uint* segStart [[buffer(0)]],
                                  device const uint* segEnd [[buffer(1)]],
                                  device const int* ord [[buffer(2)]],
                                  device const \(T)* vals [[buffer(3)]],
                                  device const uchar* validity [[buffer(4)]],
                                  device const uint* nPtr [[buffer(5)]],
                                  constant uint& hasValidity [[buffer(6)]],
                                  device \(KEYACC)* mins [[buffer(7)]],
                                  device \(KEYACC)* maxs [[buffer(8)]],
                                  device uchar* validBytes [[buffer(9)]],
                                  uint lid [[thread_index_in_threadgroup]],
                                  uint tgid [[threadgroup_position_in_grid]]) {
            threadgroup \(KEYACC) sharedMin[TG];
            threadgroup \(KEYACC) sharedMax[TG];
            threadgroup uint scount[TG];
            uint K = *nPtr;
            uint k = tgid;
            if (k >= K) return;
            uint s = segStart[k], e = segEnd[k];
            \(KEYACC) lo = \(minInit), hi = \(maxInit);
            uint cnt = 0u;
            for (uint t = s + lid; t < e; t += TG) {
                uint i = (uint)ord[t];
                if (hasValidity != 0u && !bit_get(validity, i)) continue;
                if (!(\(include))) continue;
                \(KEYACC) kk = \(key);
                lo = min(lo, kk); hi = max(hi, kk); cnt++;
            }
            sharedMin[lid] = lo; sharedMax[lid] = hi; scount[lid] = cnt;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint w = TG / 2u; w > 0u; w >>= 1) {
                if (lid < w) {
                    sharedMin[lid] = min(sharedMin[lid], sharedMin[lid + w]);
                    sharedMax[lid] = max(sharedMax[lid], sharedMax[lid + w]);
                    scount[lid] += scount[lid + w];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lid == 0u) {
                mins[k] = sharedMin[0]; maxs[k] = sharedMax[0];
                validBytes[k] = scount[0] > 0u ? 1 : 0;
            }
        }

        // Arrow `hash_product`: the segmented multiply reduction. Integers wrap in 64 bits exactly as
        // the scalar `product` does; floats reassociate across the threads of the group.
        kernel void gx_seg_product(device const uint* segStart [[buffer(0)]],
                                   device const uint* segEnd [[buffer(1)]],
                                   device const int* ord [[buffer(2)]],
                                   device const \(T)* vals [[buffer(3)]],
                                   device const uchar* validity [[buffer(4)]],
                                   device const uint* nPtr [[buffer(5)]],
                                   constant uint& hasValidity [[buffer(6)]],
                                   device \(ACC)* out [[buffer(7)]],
                                   device uchar* validBytes [[buffer(8)]],
                                   uint lid [[thread_index_in_threadgroup]],
                                   uint tgid [[threadgroup_position_in_grid]]) {
            threadgroup \(ACC) shared[TG];
            threadgroup uint scount[TG];
            uint K = *nPtr;
            uint k = tgid;
            if (k >= K) return;
            uint s = segStart[k], e = segEnd[k];
            \(ACC) acc = \(productInit);
            uint cnt = 0u;
            for (uint t = s + lid; t < e; t += TG) {
                uint i = (uint)ord[t];
                if (hasValidity != 0u && !bit_get(validity, i)) continue;
                \(ACC) a = acc, b = \(productLoad);
                acc = \(productMul);
                cnt++;
            }
            shared[lid] = acc; scount[lid] = cnt;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint w = TG / 2u; w > 0u; w >>= 1) {
                if (lid < w) {
                    \(ACC) a = shared[lid], b = shared[lid + w];
                    shared[lid] = \(productMul);
                    scount[lid] += scount[lid + w];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lid == 0u) { out[k] = shared[0]; validBytes[k] = scount[0] > 0u ? 1 : 0; }
        }

        // Second pass of the grouped skew / kurtosis: the sums of the squared, cubed and fourth-power
        // deviations from the per-group mean the first pass produced. Deviations are formed in float,
        // so a column whose values need more than 24 bits of mantissa should be scaled first.
        kernel void gx_seg_moment34(device const uint* segStart [[buffer(0)]],
                                    device const uint* segEnd [[buffer(1)]],
                                    device const int* ord [[buffer(2)]],
                                    device const \(T)* vals [[buffer(3)]],
                                    device const uchar* validity [[buffer(4)]],
                                    device const uint* nPtr [[buffer(5)]],
                                    constant uint& hasValidity [[buffer(6)]],
                                    device const float* means [[buffer(7)]],
                                    device float* out [[buffer(8)]],
                                    device uint* counts [[buffer(9)]],
                                    uint lid [[thread_index_in_threadgroup]],
                                    uint tgid [[threadgroup_position_in_grid]]) {
            threadgroup float2 s2[TG];
            threadgroup float2 s3[TG];
            threadgroup float2 s4[TG];
            threadgroup uint scount[TG];
            uint K = *nPtr;
            uint k = tgid;
            if (k >= K) return;
            uint s = segStart[k], e = segEnd[k];
            float mean = means[k];
            float2 a2 = float2(0.0f, 0.0f), a3 = a2, a4 = a2;
            uint cnt = 0u;
            for (uint t = s + lid; t < e; t += TG) {
                uint i = (uint)ord[t];
                if (hasValidity != 0u && !bit_get(validity, i)) continue;
                if (!(\(include))) continue;
                float dev = \(toFloat) - mean;
                float d2 = dev * dev;
                a2 = gx_addc(a2, d2); a3 = gx_addc(a3, d2 * dev); a4 = gx_addc(a4, d2 * d2);
                cnt++;
            }
            s2[lid] = a2; s3[lid] = a3; s4[lid] = a4; scount[lid] = cnt;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint w = TG / 2u; w > 0u; w >>= 1) {
                if (lid < w) {
                    s2[lid] = gx_merge(s2[lid], s2[lid + w]);
                    s3[lid] = gx_merge(s3[lid], s3[lid + w]);
                    s4[lid] = gx_merge(s4[lid], s4[lid + w]);
                    scount[lid] += scount[lid + w];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lid == 0u) {
                out[3u * k] = s2[0].x + s2[0].y;
                out[3u * k + 1u] = s3[0].x + s3[0].y;
                out[3u * k + 2u] = s4[0].x + s4[0].y;
                counts[k] = scount[0];
            }
        }

        // Scalar `skew` / `kurtosis`: the same three sums over the whole column, one partial per
        // threadgroup, host combine. `mp` carries the mean the first pass (`sum`) produced.
        kernel void gx_moment34(device const \(T)* vals [[buffer(0)]],
                                device const uchar* validity [[buffer(1)]],
                                device const uint* nPtr [[buffer(2)]],
                                constant uint& hasValidity [[buffer(3)]],
                                constant GxMeanParams& mp [[buffer(4)]],
                                device \(macc)* partials [[buffer(5)]],
                                device uint* counts [[buffer(6)]],
                                uint gid [[thread_position_in_grid]],
                                uint lid [[thread_index_in_threadgroup]],
                                uint tgid [[threadgroup_position_in_grid]],
                                uint gridSize [[threads_per_grid]]) {
            threadgroup \(macc) s2[TG];
            threadgroup \(macc) s3[TG];
            threadgroup \(macc) s4[TG];
            threadgroup uint scount[TG];
            uint n = *nPtr;
            \(macc) a2 = \(minit), a3 = \(minit), a4 = \(minit);
            uint cnt = 0u;
            for (uint i = gid; i < n; i += gridSize) {
                if (hasValidity != 0u && !bit_get(validity, i)) continue;
                if (!(\(minclude))) continue;
        \(scalarBody)
                cnt++;
            }
            s2[lid] = a2; s3[lid] = a3; s4[lid] = a4; scount[lid] = cnt;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint w = TG / 2u; w > 0u; w >>= 1) {
                if (lid < w) {
                    { \(macc) x = s2[lid], y = s2[lid + w]; s2[lid] = \(mmerge); }
                    { \(macc) x = s3[lid], y = s3[lid + w]; s3[lid] = \(mmerge); }
                    { \(macc) x = s4[lid], y = s4[lid + w]; s4[lid] = \(mmerge); }
                    scount[lid] += scount[lid + w];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lid == 0u) {
                partials[3u * tgid] = s2[0]; partials[3u * tgid + 1u] = s3[0]; partials[3u * tgid + 2u] = s4[0];
                counts[tgid] = scount[0];
            }
        }

        // The two values bracketing a quantile position inside each group, from an order already sorted
        // by (group, value) with the group's null values last. The host computes the two positions in
        // double (a float multiply of `q` by a 50-million row count would already miss by whole rows)
        // and this kernel only gathers them.
        kernel void gx_seg_pick(device const uint* segStart [[buffer(0)]],
                                device const int* ord [[buffer(1)]],
                                device const \(T)* vals [[buffer(2)]],
                                device const uint* loIdx [[buffer(3)]],
                                device const uint* hiIdx [[buffer(4)]],
                                device const uchar* wanted [[buffer(5)]],
                                device const uint* nPtr [[buffer(6)]],
                                device \(T)* lows [[buffer(7)]],
                                device \(T)* highs [[buffer(8)]],
                                uint k [[thread_position_in_grid]]) {
            if (k >= *nPtr) return;
            if (wanted[k] == 0) return;
            uint s = segStart[k];
            lows[k] = vals[(uint)ord[s + loIdx[k]]];
            highs[k] = vals[(uint)ord[s + hiIdx[k]]];
        }
        """
    }
}

/// The generated MSL for one element type, plus its pipeline cache key.
struct ExtraAggregateSpec {
    let source: String
    let key: String
    let moment: AggregatesExtraSource.Moment

    static func of<T: ArrowPrimitive>(_: T.Type) -> ExtraAggregateSpec {
        if T.self == Double.self {
            return ExtraAggregateSpec(
                source: AggregatesExtraSource.source(
                    T: "long", KEYACC: "long", ACC: "ulong",
                    key: "d_key((long)vals[i])", include: "!d_isnan((long)vals[i])",
                    minInit: "LONG_MAX", maxInit: "LONG_MIN",
                    productInit: "0x3FF0000000000000ul", productMul: "d_mul(a, b)",
                    productLoad: "(ulong)vals[i]", toFloat: "0.0f", moment: .double,
                    extraPrelude: DoubleMath.msl),
                key: "g", moment: .double)
        }
        if T.isFloatingPoint {
            return ExtraAggregateSpec(
                source: AggregatesExtraSource.source(
                    T: "float", KEYACC: "float", ACC: "float",
                    key: "vals[i]", include: "!isnan(vals[i])",
                    minInit: "INFINITY", maxInit: "-INFINITY",
                    productInit: "1.0f", productMul: "a * b",
                    productLoad: "(float)vals[i]", toFloat: "(float)vals[i]", moment: .float),
                key: "f", moment: .float)
        }
        let signed = T.minValue < 0 as T
        return ExtraAggregateSpec(
            source: AggregatesExtraSource.source(
                T: T.mslType, KEYACC: signed ? "long" : "ulong", ACC: signed ? "long" : "ulong",
                key: signed ? "(long)vals[i]" : "(ulong)vals[i]", include: "true",
                minInit: signed ? "LONG_MAX" : "ULONG_MAX", maxInit: signed ? "LONG_MIN" : "0ul",
                productInit: signed ? "1L" : "1ul",
                // Signed multiplication runs on the unsigned representation so overflow wraps.
                productMul: signed ? "(long)((ulong)a * (ulong)b)" : "a * b",
                productLoad: signed ? "(long)vals[i]" : "(ulong)vals[i]",
                toFloat: "(float)vals[i]", moment: .integer),
            key: T.arrowFormat, moment: .integer)
    }
}
