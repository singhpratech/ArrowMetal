import Foundation

/// Metal Shading Language for the scalar aggregates in `Aggregates.swift`.
///
/// One library is generated per element type. Every kernel follows the shape the existing reductions
/// use: a strided loop per thread, a threadgroup tree reduction, one partial (and one valid count) per
/// threadgroup, and a host finalise over those partials.
///
/// The element-type-specific pieces come in as small inline helpers so the kernels themselves are
/// written once:
///
/// | helper | meaning |
/// |---|---|
/// | `ag_include(v)` | whether the value takes part (false only for NaN, where Arrow skips) |
/// | `ag_mul(a, b)` | the product combine on the accumulator (wrapping for integers, `d_mul` for float64) |
/// | `ag_key(v)` | the value mapped into the ordering accumulator used by min/max |
/// | `ag_eq(a, b)` | Arrow value equality, used by `index` |
enum AggregatesSource {

    /// Type-independent kernels: boolean word reductions and the first/last valid index scan.
    static let common = KernelSource.prelude + """
    // any / all: counts of set bits in (values & validity) and in validity, one thread per 32-bit word.
    kernel void agg_bool_counts(device const uint* values [[buffer(0)]],
                                device const uint* validity [[buffer(1)]],
                                constant uint& hasValidity [[buffer(2)]],
                                device const uint* nPtr [[buffer(3)]],
                                device atomic_uint* trueCount [[buffer(4)]],
                                device atomic_uint* validCount [[buffer(5)]],
                                uint w [[thread_position_in_grid]],
                                uint lid [[thread_index_in_threadgroup]],
                                uint sgid [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
        threadgroup uint sharedTrue[32];
        threadgroup uint sharedValid[32];
        uint n = *nPtr;
        uint base = w * 32u;
        uint value = 0u, valid = 0u;
        if (base < n) {
            uint mask = (n - base < 32u) ? ((1u << (n - base)) - 1u) : 0xFFFFFFFFu;
            uint v = values[w] & mask;
            uint m = hasValidity ? (validity[w] & mask) : mask;
            value = popcount(v & m);
            valid = popcount(m);
        }
        uint t = simd_sum(value), q = simd_sum(valid);
        if (lane == 0) { sharedTrue[sgid] = t; sharedValid[sgid] = q; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            uint a = 0, b = 0;
            for (uint k = 0; k < TG / 32u; k++) { a += sharedTrue[k]; b += sharedValid[k]; }
            atomic_fetch_add_explicit(trueCount, a, memory_order_relaxed);
            atomic_fetch_add_explicit(validCount, b, memory_order_relaxed);
        }
    }
    // first / last with skip_nulls: the smallest and largest index whose validity bit is set.
    kernel void agg_valid_bounds(device const uchar* validity [[buffer(0)]],
                                 device const uint* nPtr [[buffer(1)]],
                                 device atomic_uint* lo [[buffer(2)]],
                                 device atomic_uint* hi [[buffer(3)]],
                                 uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if (!bit_get(validity, i)) return;
        atomic_fetch_min_explicit(lo, i, memory_order_relaxed);
        atomic_fetch_max_explicit(hi, i + 1u, memory_order_relaxed);
    }
    """

    /// How the second pass computes `(x - mean)^2`.
    enum MomentMode {
        /// Integer values: the deviation is `(float)(v - floor(mean)) - frac(mean)`, so a large integer
        /// column keeps full precision in the subtraction (the difference is small even when v is not).
        case integer
        /// Float32 values: the mean travels as a two-float pair, so the subtraction is exact to a ulp
        /// of the deviation rather than of the value.
        case float
        /// Float64 values: everything runs through the software binary64 routines in `DoubleMath`.
        case double
    }

    /// Full source for one element type.
    ///
    /// - `T`: the MSL type of the values buffer (float64 arrives as `long` bit patterns).
    /// - `ACC`: the accumulator for `product` (`long`, `ulong`, `float`, or `ulong` bit patterns).
    /// - `KEYACC`: the accumulator min/max order on.
    static func source(T: String, ACC: String, KEYACC: String,
                       productInit: String, productMul: String, load: String,
                       include: String, key: String, minInit: String, maxInit: String,
                       equal: String, moment: MomentMode, extraPrelude: String = "") -> String {
        let momentBody: String
        switch moment {
        case .integer:
            momentBody = """
                float dev = (float)((long)vals[i] - mp.ipart) - mp.hi;
                acc = ag_addc(acc, dev * dev);
            """
        case .float:
            momentBody = """
                float dev = ((float)vals[i] - mp.hi) - mp.lo;
                acc = ag_addc(acc, dev * dev);
            """
        case .double:
            momentBody = """
                ulong dev = d_sub((ulong)vals[i], mp.bits);
                acc = d_add(acc, d_mul(dev, dev));
            """
        }
        let momentAcc = moment == .double ? "ulong" : "float2"
        let momentInit = moment == .double ? "0ul" : "float2(0.0f, 0.0f)"
        let momentMerge = moment == .double ? "d_add(a, b)" : "ag_merge(a, b)"
        let momentInclude = moment == .double ? "!d_isnan((long)vals[i])" : include

        return KernelSource.prelude + extraPrelude + """

        struct AggMeanParams { long ipart; ulong bits; float hi; float lo; };

        inline bool ag_include(\(T) v) { return \(include.replacingOccurrences(of: "vals[i]", with: "v")); }
        inline \(ACC) ag_mul(\(ACC) a, \(ACC) b) { return \(productMul); }
        inline bool ag_eq(\(T) a, \(T) b) { return \(equal); }
        // Compensated (Neumaier) float addition: the pair is (sum, correction).
        inline float2 ag_addc(float2 acc, float x) {
            float s = acc.x + x;
            float c = (fabs(acc.x) >= fabs(x)) ? ((acc.x - s) + x) : ((x - s) + acc.x);
            return float2(s, acc.y + c);
        }
        inline float2 ag_merge(float2 a, float2 b) {
            float s = a.x + b.x;
            float c = (fabs(a.x) >= fabs(b.x)) ? ((a.x - s) + b.x) : ((b.x - s) + a.x);
            return float2(s, (a.y + b.y) + c);
        }

        // Arrow `product`: one partial per threadgroup, combined on the host.
        kernel void agg_product(device const \(T)* vals [[buffer(0)]],
                                device const uchar* validity [[buffer(1)]],
                                device const uint* nPtr [[buffer(2)]],
                                constant uint& hasValidity [[buffer(3)]],
                                device \(ACC)* partials [[buffer(4)]],
                                device uint* counts [[buffer(5)]],
                                uint gid [[thread_position_in_grid]],
                                uint lid [[thread_index_in_threadgroup]],
                                uint tgid [[threadgroup_position_in_grid]],
                                uint gridSize [[threads_per_grid]]) {
            threadgroup \(ACC) shared[TG];
            threadgroup uint scount[TG];
            uint n = *nPtr;
            \(ACC) acc = \(productInit);
            uint cnt = 0;
            for (uint i = gid; i < n; i += gridSize) {
                if (hasValidity && !bit_get(validity, i)) continue;
                \(ACC) v = \(load);
                acc = ag_mul(acc, v);
                cnt++;
            }
            shared[lid] = acc; scount[lid] = cnt;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = TG / 2; s > 0; s >>= 1) {
                if (lid < s) { shared[lid] = ag_mul(shared[lid], shared[lid + s]); scount[lid] += scount[lid + s]; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lid == 0) { partials[tgid] = shared[0]; counts[tgid] = scount[0]; }
        }

        // Arrow `min_max` in one pass: both partials come out of the same read of the values.
        kernel void agg_minmax(device const \(T)* vals [[buffer(0)]],
                               device const uchar* validity [[buffer(1)]],
                               device const uint* nPtr [[buffer(2)]],
                               constant uint& hasValidity [[buffer(3)]],
                               device \(KEYACC)* mins [[buffer(4)]],
                               device \(KEYACC)* maxs [[buffer(5)]],
                               device uint* counts [[buffer(6)]],
                               uint gid [[thread_position_in_grid]],
                               uint lid [[thread_index_in_threadgroup]],
                               uint tgid [[threadgroup_position_in_grid]],
                               uint gridSize [[threads_per_grid]]) {
            threadgroup \(KEYACC) sharedMin[TG];
            threadgroup \(KEYACC) sharedMax[TG];
            threadgroup uint scount[TG];
            uint n = *nPtr;
            \(KEYACC) lo = \(minInit), hi = \(maxInit);
            uint cnt = 0;
            for (uint i = gid; i < n; i += gridSize) {
                if (hasValidity && !bit_get(validity, i)) continue;
                if (!(\(include))) continue;
                \(KEYACC) k = \(key);
                lo = min(lo, k); hi = max(hi, k); cnt++;
            }
            sharedMin[lid] = lo; sharedMax[lid] = hi; scount[lid] = cnt;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = TG / 2; s > 0; s >>= 1) {
                if (lid < s) {
                    sharedMin[lid] = min(sharedMin[lid], sharedMin[lid + s]);
                    sharedMax[lid] = max(sharedMax[lid], sharedMax[lid + s]);
                    scount[lid] += scount[lid + s];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lid == 0) { mins[tgid] = sharedMin[0]; maxs[tgid] = sharedMax[0]; counts[tgid] = scount[0]; }
        }

        // Second pass of variance / stddev: the sum of squared deviations from a mean the host computed
        // from the first pass (`sum`), so no single-pass cancellation is involved.
        kernel void agg_moment(device const \(T)* vals [[buffer(0)]],
                               device const uchar* validity [[buffer(1)]],
                               device const uint* nPtr [[buffer(2)]],
                               constant uint& hasValidity [[buffer(3)]],
                               constant AggMeanParams& mp [[buffer(4)]],
                               device \(momentAcc)* partials [[buffer(5)]],
                               device uint* counts [[buffer(6)]],
                               uint gid [[thread_position_in_grid]],
                               uint lid [[thread_index_in_threadgroup]],
                               uint tgid [[threadgroup_position_in_grid]],
                               uint gridSize [[threads_per_grid]]) {
            threadgroup \(momentAcc) shared[TG];
            threadgroup uint scount[TG];
            uint n = *nPtr;
            \(momentAcc) acc = \(momentInit);
            uint cnt = 0;
            for (uint i = gid; i < n; i += gridSize) {
                if (hasValidity && !bit_get(validity, i)) continue;
                if (!(\(momentInclude))) continue;
        \(momentBody)
                cnt++;
            }
            shared[lid] = acc; scount[lid] = cnt;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = TG / 2; s > 0; s >>= 1) {
                if (lid < s) {
                    \(momentAcc) a = shared[lid], b = shared[lid + s];
                    shared[lid] = \(momentMerge);
                    scount[lid] += scount[lid + s];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lid == 0) { partials[tgid] = shared[0]; counts[tgid] = scount[0]; }
        }

        // Arrow `index`: the smallest row that holds `target`, by a device-wide atomic minimum.
        kernel void agg_index(device const \(T)* vals [[buffer(0)]],
                              device const uchar* validity [[buffer(1)]],
                              device const uint* nPtr [[buffer(2)]],
                              constant uint& hasValidity [[buffer(3)]],
                              constant \(T)& target [[buffer(4)]],
                              device atomic_uint* out [[buffer(5)]],
                              uint i [[thread_position_in_grid]]) {
            if (i >= *nPtr) return;
            if (hasValidity && !bit_get(validity, i)) return;
            if (ag_eq(vals[i], target)) atomic_fetch_min_explicit(out, i, memory_order_relaxed);
        }
        """
    }
}
