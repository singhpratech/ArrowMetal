import Foundation

/// MSL for the grouped central moments — `hash_variance`, `hash_stddev`, `hash_skew`, `hash_kurtosis` —
/// in **true binary64**, over the counting-sort order.
///
/// The old grouped variance formed its deviations in Float32 about a Float32 mean and summed them with
/// a Float32 atomic compare-and-swap per row, which cost about a relative 1e-5 and, because every row of
/// a group contends for one word, ran at a fraction of memory bandwidth. This runs the textbook
/// two-pass shifted algorithm instead, with every addition and multiplication going through the software
/// binary64 routines in `DoubleMath`:
///
/// 1. the per-group mean, from an exact `d_add` sum of the values;
/// 2. the per-group sums of the deviations `d = x - mean` and of `d^2` (and `d^3`, `d^4` for the third
///    and fourth moments), also by `d_add`;
/// 3. `m2 = (sum d^2 - (sum d)^2 / n) / (n - ddof)`, the correction term that removes the error in the
///    mean itself.
///
/// Shifting by the mean first is what keeps this accurate: `sum x^2 - (sum x)^2 / n` cancels
/// catastrophically on data far from zero, and the shifted form does not.
///
/// Two shapes, because a group-by has two regimes: `_wide` gives a whole threadgroup to one group and
/// tree-reduces in threadgroup memory, which is right while groups hold hundreds of rows; `_narrow`
/// gives one thread to one group, which is right when there are ten million groups holding five rows
/// each and a threadgroup per group would launch a quarter of a billion idle threads.
enum GroupMomentsSource {

    /// How a row of this element type becomes binary64 bits.
    static func loadExpr<T: ArrowPrimitive>(_: T.Type) -> (valueType: String, load: String, name: String) {
        if T.self == Double.self { return ("ulong", "(ulong)vals[i]", "f64") }
        if T.isFloatingPoint { return ("float", "d_from_float(vals[i])", "f32") }
        if T.minValue < 0 as T { return (T.mslType, "d_from_long((long)vals[i])", "i\(T.byteWidth * 8)") }
        return (T.mslType, "d_from_ulong((ulong)vals[i])", "u\(T.byteWidth * 8)")
    }

    static func source(valueType: String, load: String) -> String {
        KernelSource.prelude + DoubleMath.msl + """

        // Exact-or-correctly-rounded integer to binary64. `d_from_ucount` in SegmentedSource truncates;
        // a variance over int64 needs the rounding.
        inline ulong d_from_ulong(ulong v) {
            if (v == 0ul) return 0ul;
            uint p = 63u;
            while (((v >> p) & 1ul) == 0ul) p--;
            ulong m;
            if (p <= 52u) {
                m = v << (52u - p);
            } else {
                uint sh = p - 52u;
                ulong t = v >> sh;
                ulong rem = v & ((1ul << sh) - 1ul);
                ulong hf = 1ul << (sh - 1u);
                if (rem > hf || (rem == hf && (t & 1ul) != 0ul)) {
                    t++;
                    if ((t >> 53) != 0ul) { t >>= 1; p++; }
                }
                m = t;
            }
            ulong e = (ulong)((long)p + 1023);
            return (e << 52) | (m & 0xFFFFFFFFFFFFFul);
        }
        inline ulong d_from_long(long v) {
            ulong a = (v < 0) ? (~(ulong)v + 1ul) : (ulong)v;
            ulong b = d_from_ulong(a);
            return (v < 0) ? (b | 0x8000000000000000ul) : b;
        }

        #define GM_ARGS device const uint* segStart [[buffer(0)]], \\
                        device const uint* segEnd [[buffer(1)]], \\
                        device const int* ord [[buffer(2)]], \\
                        device const \(valueType)* vals [[buffer(3)]], \\
                        device const uchar* validity [[buffer(4)]], \\
                        device const uint* nPtr [[buffer(5)]], \\
                        constant uint& hasValidity [[buffer(6)]]

        // Pass one: the exact sum of each group's values, and how many there are.
        kernel void gm_sum_wide(GM_ARGS,
                                device ulong* sums [[buffer(7)]],
                                device uint* counts [[buffer(8)]],
                                uint lid [[thread_index_in_threadgroup]],
                                uint tgid [[threadgroup_position_in_grid]]) {
            threadgroup ulong sh[TG];
            threadgroup uint sc[TG];
            uint K = *nPtr;
            uint k = tgid;
            if (k >= K) return;
            uint s = segStart[k], e = segEnd[k];
            ulong acc = 0ul; uint cnt = 0u;
            for (uint t = s + lid; t < e; t += TG) {
                uint i = (uint)ord[t];
                if (hasValidity != 0u && !bit_get(validity, i)) continue;
                acc = d_add(acc, \(load)); cnt++;
            }
            sh[lid] = acc; sc[lid] = cnt;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint w = TG / 2u; w > 0u; w >>= 1) {
                if (lid < w) { sh[lid] = d_add(sh[lid], sh[lid + w]); sc[lid] += sc[lid + w]; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lid == 0u) { sums[k] = sh[0]; counts[k] = sc[0]; }
        }

        kernel void gm_sum_narrow(GM_ARGS,
                                  device ulong* sums [[buffer(7)]],
                                  device uint* counts [[buffer(8)]],
                                  uint k [[thread_position_in_grid]]) {
            uint K = *nPtr;
            if (k >= K) return;
            uint s = segStart[k], e = segEnd[k];
            ulong acc = 0ul; uint cnt = 0u;
            for (uint t = s; t < e; t++) {
                uint i = (uint)ord[t];
                if (hasValidity != 0u && !bit_get(validity, i)) continue;
                acc = d_add(acc, \(load)); cnt++;
            }
            sums[k] = acc; counts[k] = cnt;
        }

        // The group mean, exactly rounded from the exact sum.
        kernel void gm_mean(device const ulong* sums [[buffer(0)]], device const uint* counts [[buffer(1)]],
                            constant uint& K [[buffer(2)]], device ulong* means [[buffer(3)]],
                            uint k [[thread_position_in_grid]]) {
            if (k >= K) return;
            uint c = counts[k];
            means[k] = c ? d_div(sums[k], d_from_ulong((ulong)c)) : 0ul;
        }

        // Pass two: the sums of d, d^2 (and, in the `4` variants, d^3 and d^4) about that mean.
        kernel void gm_dev_wide(GM_ARGS,
                                device const ulong* means [[buffer(7)]],
                                device ulong* out [[buffer(8)]],
                                uint lid [[thread_index_in_threadgroup]],
                                uint tgid [[threadgroup_position_in_grid]]) {
            threadgroup ulong s1[TG];
            threadgroup ulong s2[TG];
            uint K = *nPtr;
            uint k = tgid;
            if (k >= K) return;
            uint s = segStart[k], e = segEnd[k];
            ulong mean = means[k];
            ulong a1 = 0ul, a2 = 0ul;
            for (uint t = s + lid; t < e; t += TG) {
                uint i = (uint)ord[t];
                if (hasValidity != 0u && !bit_get(validity, i)) continue;
                ulong dv = d_sub(\(load), mean);
                a1 = d_add(a1, dv); a2 = d_add(a2, d_mul(dv, dv));
            }
            s1[lid] = a1; s2[lid] = a2;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint w = TG / 2u; w > 0u; w >>= 1) {
                if (lid < w) { s1[lid] = d_add(s1[lid], s1[lid + w]); s2[lid] = d_add(s2[lid], s2[lid + w]); }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lid == 0u) { out[2u * k] = s1[0]; out[2u * k + 1u] = s2[0]; }
        }

        kernel void gm_dev_narrow(GM_ARGS,
                                  device const ulong* means [[buffer(7)]],
                                  device ulong* out [[buffer(8)]],
                                  uint k [[thread_position_in_grid]]) {
            uint K = *nPtr;
            if (k >= K) return;
            uint s = segStart[k], e = segEnd[k];
            ulong mean = means[k];
            ulong a1 = 0ul, a2 = 0ul;
            for (uint t = s; t < e; t++) {
                uint i = (uint)ord[t];
                if (hasValidity != 0u && !bit_get(validity, i)) continue;
                ulong dv = d_sub(\(load), mean);
                a1 = d_add(a1, dv); a2 = d_add(a2, d_mul(dv, dv));
            }
            out[2u * k] = a1; out[2u * k + 1u] = a2;
        }

        kernel void gm_dev4_wide(GM_ARGS,
                                 device const ulong* means [[buffer(7)]],
                                 device ulong* out [[buffer(8)]],
                                 uint lid [[thread_index_in_threadgroup]],
                                 uint tgid [[threadgroup_position_in_grid]]) {
            threadgroup ulong s2[TG];
            threadgroup ulong s3[TG];
            threadgroup ulong s4[TG];
            uint K = *nPtr;
            uint k = tgid;
            if (k >= K) return;
            uint s = segStart[k], e = segEnd[k];
            ulong mean = means[k];
            ulong a2 = 0ul, a3 = 0ul, a4 = 0ul;
            for (uint t = s + lid; t < e; t += TG) {
                uint i = (uint)ord[t];
                if (hasValidity != 0u && !bit_get(validity, i)) continue;
                ulong dv = d_sub(\(load), mean);
                ulong q = d_mul(dv, dv);
                a2 = d_add(a2, q); a3 = d_add(a3, d_mul(q, dv)); a4 = d_add(a4, d_mul(q, q));
            }
            s2[lid] = a2; s3[lid] = a3; s4[lid] = a4;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint w = TG / 2u; w > 0u; w >>= 1) {
                if (lid < w) {
                    s2[lid] = d_add(s2[lid], s2[lid + w]);
                    s3[lid] = d_add(s3[lid], s3[lid + w]);
                    s4[lid] = d_add(s4[lid], s4[lid + w]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lid == 0u) { out[3u * k] = s2[0]; out[3u * k + 1u] = s3[0]; out[3u * k + 2u] = s4[0]; }
        }

        kernel void gm_dev4_narrow(GM_ARGS,
                                   device const ulong* means [[buffer(7)]],
                                   device ulong* out [[buffer(8)]],
                                   uint k [[thread_position_in_grid]]) {
            uint K = *nPtr;
            if (k >= K) return;
            uint s = segStart[k], e = segEnd[k];
            ulong mean = means[k];
            ulong a2 = 0ul, a3 = 0ul, a4 = 0ul;
            for (uint t = s; t < e; t++) {
                uint i = (uint)ord[t];
                if (hasValidity != 0u && !bit_get(validity, i)) continue;
                ulong dv = d_sub(\(load), mean);
                ulong q = d_mul(dv, dv);
                a2 = d_add(a2, q); a3 = d_add(a3, d_mul(q, dv)); a4 = d_add(a4, d_mul(q, q));
            }
            out[3u * k] = a2; out[3u * k + 1u] = a3; out[3u * k + 2u] = a4;
        }

        // The sample or population variance from the two sums, with the shift correction.
        kernel void gm_variance(device const ulong* dev [[buffer(0)]], device const uint* counts [[buffer(1)]],
                                constant uint& K [[buffer(2)]], constant uint& ddof [[buffer(3)]],
                                device ulong* out [[buffer(4)]], device uchar* valid [[buffer(5)]],
                                uint k [[thread_position_in_grid]]) {
            if (k >= K) return;
            uint c = counts[k];
            if (c <= ddof) { out[k] = 0ul; valid[k] = 0; return; }
            ulong n = d_from_ulong((ulong)c);
            ulong s1 = dev[2u * k], s2 = dev[2u * k + 1u];
            ulong m2 = d_sub(s2, d_div(d_mul(s1, s1), n));
            out[k] = d_div(m2, d_from_ulong((ulong)(c - ddof)));
            valid[k] = 1;
        }
        """
    }
}
