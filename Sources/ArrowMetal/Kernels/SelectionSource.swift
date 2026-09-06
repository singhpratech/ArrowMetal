import Foundation

/// MSL for the remaining Arrow selection, sort and random functions:
/// `inverse_permutation`, `scatter`, `winsorize`, `rank_quantile`, `rank_normal` and `random`.
///
/// Four families, each compiled once per type it needs:
///
///   * `indexOps` — the scatter that builds an inverse permutation. Untyped in its output
///     (always int32 positions), typed in its input by `indexOps(V:)`.
///   * `clamp(V:kind:)` — the winsorising clamp for one value type.
///   * `ranking(U:)` — the quantile-rank pipeline in *sorted position* space: run marks, run
///     bounds and the two scatters that write a rank back to the row it came from. `U` is the
///     unsigned integer of the element width, so one kernel covers signed, unsigned and float.
///   * `random` — a counter-based Philox4x32-10 generator producing binary64 patterns in [0, 1).
enum SelectionSource {

    // MARK: - Inverse permutation

    /// Scatter kernels for an index column of MSL type `V`.
    ///
    /// `sel_fill_i32` primes the output with -1 ("nothing written here yet"). `sel_invperm` writes
    /// the *largest* source position that names each output slot, which is exactly Arrow's
    /// last-write-wins for duplicate indices and is deterministic under parallel execution (an
    /// unordered `atomic_fetch_max` cannot depend on thread order). `sel_invperm_finish` turns the
    /// -1s into Arrow nulls.
    static func indexOps(V: String) -> String { KernelSource.prelude + """

    kernel void sel_fill_i32(device int* out [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                             constant int& v [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i < *nPtr) out[i] = v;
    }
    kernel void sel_invperm(device const \(V)* idx [[buffer(0)]], device const uchar* validity [[buffer(1)]],
                            device const uint* nPtr [[buffer(2)]], constant uint& hasValidity [[buffer(3)]],
                            constant long& maxIndex [[buffer(4)]], device atomic_int* out [[buffer(5)]],
                            device atomic_uint* err [[buffer(6)]], uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if (hasValidity && !bit_get(validity, i)) return;
        long v = (long)idx[i];
        if (v < 0 || v > maxIndex) { atomic_store_explicit(err, 1u, memory_order_relaxed); return; }
        atomic_fetch_max_explicit(&out[v], (int)i, memory_order_relaxed);
    }
    kernel void sel_invperm_finish(device int* out [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                                   device uchar* validBytes [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        int v = out[i];
        validBytes[i] = (v >= 0) ? 1 : 0;
        if (v < 0) out[i] = 0;
    }
    """ }

    // MARK: - Winsorising clamp

    enum Kind { case integer, float32, float64 }

    /// `out[i] = min(max(a[i], lo), hi)` with Arrow's null and NaN rules: a null row keeps its slot
    /// untouched (the validity bitmap is shared with the input) and a NaN passes through unchanged,
    /// because NaN compares false against both limits.
    ///
    /// Float64 values travel as raw binary64 patterns, so the comparison goes through the
    /// order-preserving `d_key` map from the prelude rather than through software arithmetic.
    static func clamp(V: String, kind: Kind) -> String {
        let body: String
        switch kind {
        case .integer:
            body = "\(V) x = a[i]; out[i] = (x < lo) ? lo : ((x > hi) ? hi : x);"
        case .float32:
            body = "float x = a[i]; out[i] = isnan(x) ? x : ((x < lo) ? lo : ((x > hi) ? hi : x));"
        case .float64:
            body = """
            ulong x = a[i];
            if (d_is_nan(x)) { out[i] = x; return; }
            long k = d_key((long)x);
            out[i] = (k < d_key((long)lo)) ? lo : ((k > d_key((long)hi)) ? hi : x);
            """
        }
        return KernelSource.prelude + DoubleMath.msl + """

        kernel void sel_clamp(device const \(V)* a [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                              constant \(V)& lo [[buffer(2)]], constant \(V)& hi [[buffer(3)]],
                              device \(V)* out [[buffer(4)]], uint i [[thread_position_in_grid]]) {
            if (i >= *nPtr) return;
            \(body)
        }
        """
    }

    // MARK: - Quantile and normal ranks

    /// Quantile-rank kernels for an element width whose unsigned type is `U`.
    ///
    /// The pipeline is: normalise floats (one NaN pattern, -0 becomes +0), argsort (`Sort.swift`),
    /// mark the run boundaries of the sorted order, an inclusive int32 scan of those marks
    /// (`cumulative(.sum)`) to number the runs from 1, record where each run starts and ends, then
    /// scatter one value per row back to the row it came from.
    ///
    /// The quantile rank of a row is `(average 1-based rank of its tie group - 0.5) / n`, which for a
    /// run covering sorted positions `[s, e)` is exactly `(s + e) / (2n)` — pyarrow's definition,
    /// including nulls, which `argsort` puts last and the marks kernel keeps as one tie group.
    static func ranking(U: String) -> String { KernelSource.prelude + WindowSource.doublePrelude + """

    kernel void sel_norm_f32(device const uint* a [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                             device uint* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        uint b = a[i], m = b & 0x7FFFFFFFu;
        if (m > 0x7F800000u) b = 0x7FC00000u; else if (m == 0u) b = 0u;
        out[i] = b;
    }
    kernel void sel_norm_f64(device const ulong* a [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                             device ulong* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        ulong b = a[i], m = b & 0x7FFFFFFFFFFFFFFFul;
        if (m > 0x7FF0000000000000ul) b = 0x7FF8000000000000ul; else if (m == 0ul) b = 0ul;
        out[i] = b;
    }
    // 1 where a new tie group starts. Sorted positions [0, m) hold the values and [m, n) the nulls,
    // which are one group. Position 0 writes 1, so an *inclusive* scan numbers the groups from 1.
    kernel void sel_marks(device const \(U)* vals [[buffer(0)]], device const int* ord [[buffer(1)]],
                          device const uint* nPtr [[buffer(2)]], constant uint& m [[buffer(3)]],
                          device int* marks [[buffer(4)]], uint i [[thread_position_in_grid]]) {
        uint n = *nPtr;
        if (i >= n) return;
        int f;
        if (i == 0u) f = 1;
        else if (i >= m) f = (i == m) ? 1 : 0;
        else f = (vals[ord[i]] != vals[ord[i - 1u]]) ? 1 : 0;
        marks[i] = f;
    }
    // First and one-past-last sorted position of every run, indexed by the run's 0-based number.
    kernel void sel_run_bounds(device const int* marks [[buffer(0)]], device const int* groups [[buffer(1)]],
                               device const uint* nPtr [[buffer(2)]], device int* runStart [[buffer(3)]],
                               device int* runEnd [[buffer(4)]], uint i [[thread_position_in_grid]]) {
        uint n = *nPtr;
        if (i >= n) return;
        int g = groups[i] - 1;
        if (marks[i] != 0) runStart[g] = (int)i;
        if (i + 1u == n || marks[i + 1u] != 0) runEnd[g] = (int)(i + 1u);
    }
    // rank_quantile as binary64: (runStart + runEnd) / (2n), correctly rounded by the software divide.
    kernel void sel_quantile_scatter(device const int* ord [[buffer(0)]], device const int* groups [[buffer(1)]],
                                     device const int* runStart [[buffer(2)]], device const int* runEnd [[buffer(3)]],
                                     device const uint* nPtr [[buffer(4)]], device ulong* out [[buffer(5)]],
                                     uint i [[thread_position_in_grid]]) {
        uint n = *nPtr;
        if (i >= n) return;
        int g = groups[i] - 1;
        ulong num = (ulong)((uint)runStart[g]) + (ulong)((uint)runEnd[g]);
        out[ord[i]] = d_div(d_from_u64(num), d_from_u64(2ul * (ulong)n));
    }

    // Metal has no `erfc`, so this is the Chebyshev fit from Numerical Recipes (`erfcc`), whose
    // fractional error stays under 1.2e-7 everywhere — an order of magnitude better than the float32
    // ulp of the quantiles it is used on below.
    inline float sel_erfc(float x) {
        float z = fabs(x), t = 1.0f / (1.0f + 0.5f * z);
        float ans = t * exp(-z * z - 1.26551223f + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f +
                    t * (-0.18628806f + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f +
                    t * (-0.82215223f + t * 0.17087277f)))))))));
        return (x >= 0.0f) ? ans : 2.0f - ans;
    }
    // Inverse normal CDF in float32: Acklam's rational approximation (relative error under 1.15e-9 in
    // exact arithmetic) followed by one Halley refinement through `sel_erfc`, which pulls the result
    // back to about a float32 ulp of the true quantile.
    inline float sel_ppf_f32(float q) {
        const float a0 = -3.969683028665376e+01f, a1 = 2.209460984245205e+02f, a2 = -2.759285104469687e+02f;
        const float a3 = 1.383577518672690e+02f, a4 = -3.066479806614716e+01f, a5 = 2.506628277459239e+00f;
        const float b0 = -5.447609879822406e+01f, b1 = 1.615858368580409e+02f, b2 = -1.556989798598866e+02f;
        const float b3 = 6.680131188771972e+01f, b4 = -1.328068155288572e+01f;
        const float c0 = -7.784894002430293e-03f, c1 = -3.223964580411365e-01f, c2 = -2.400758277161838e+00f;
        const float c3 = -2.549732539343734e+00f, c4 = 4.374664141464968e+00f, c5 = 2.938163982698783e+00f;
        const float d0 = 7.784695709041462e-03f, d1 = 3.224671290700398e-01f, d2 = 2.445134137142996e+00f;
        const float d3 = 3.754408661907416e+00f;
        const float pLow = 0.02425f;
        if (!(q > 0.0f)) return -INFINITY;
        if (!(q < 1.0f)) return INFINITY;
        float x;
        if (q < pLow) {
            float t = sqrt(-2.0f * log(q));
            x = (((((c0 * t + c1) * t + c2) * t + c3) * t + c4) * t + c5) /
                ((((d0 * t + d1) * t + d2) * t + d3) * t + 1.0f);
        } else if (q <= 1.0f - pLow) {
            float t = q - 0.5f, r = t * t;
            x = (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * t /
                (((((b0 * r + b1) * r + b2) * r + b3) * r + b4) * r + 1.0f);
        } else {
            float t = sqrt(-2.0f * log(1.0f - q));
            x = -(((((c0 * t + c1) * t + c2) * t + c3) * t + c4) * t + c5) /
                 ((((d0 * t + d1) * t + d2) * t + d3) * t + 1.0f);
        }
        // Halley step on F(x) = Phi(x) - q.
        float e = 0.5f * sel_erfc(-x * 0.7071067811865476f) - q;
        float u = e * 2.5066282746310002f * exp(x * x * 0.5f);
        return x - u / (1.0f + x * u * 0.5f);
    }
    // rank_normal in float32: the quantile rank of the row, put through the normal PPF.
    kernel void sel_normal_scatter_f32(device const int* ord [[buffer(0)]], device const int* groups [[buffer(1)]],
                                       device const int* runStart [[buffer(2)]], device const int* runEnd [[buffer(3)]],
                                       device const uint* nPtr [[buffer(4)]], device float* out [[buffer(5)]],
                                       uint i [[thread_position_in_grid]]) {
        uint n = *nPtr;
        if (i >= n) return;
        int g = groups[i] - 1;
        ulong num = (ulong)((uint)runStart[g]) + (ulong)((uint)runEnd[g]);
        ulong den = 2ul * (ulong)n;
        // Above the median, work with the complement 1 - q instead: `den - num` is exact integer
        // arithmetic, whereas a float32 q near 1 has already thrown away the digits the PPF needs
        // (its slope there is 1/phi(z), so a 1e-7 error in q is a 1e-4 error in z).
        if (2ul * num <= den) out[ord[i]] = sel_ppf_f32((float)num / (float)den);
        else out[ord[i]] = -sel_ppf_f32((float)(den - num) / (float)den);
    }
    """ }

    // MARK: - Random

    /// Philox4x32-10 (Salmon, Moraes, Dror & Shaw, SC'11): a counter-based generator, so thread `i`
    /// computes its own value from the counter `i` and the 64-bit key without any shared state, and
    /// the stream depends only on the seed — never on the launch geometry.
    ///
    /// Word 0 and word 1 of each 4x32 output form a 64-bit integer; its top 53 bits are turned into a
    /// binary64 in [0, 1) by writing the exponent and mantissa fields directly, which is exact (the
    /// value is a multiple of 2^-53 below 1) and needs no software floating-point arithmetic.
    static let random = KernelSource.prelude + """

    inline uint sel_mulhi(uint a, uint b) { return (uint)(((ulong)a * (ulong)b) >> 32); }

    inline uint4 sel_philox(uint4 c, uint2 k) {
        for (uint r = 0; r < 10u; r++) {
            uint hi0 = sel_mulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x;
            uint hi1 = sel_mulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
            c = uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
            k += uint2(0x9E3779B9u, 0xBB67AE85u);
        }
        return c;
    }
    // 53 random bits -> a binary64 pattern for value * 2^-53, value in [0, 2^53).
    inline ulong sel_bits_to_unit(ulong v) {
        if (v == 0ul) return 0ul;
        uint hi = (uint)(v >> 32);
        uint k = (hi != 0u) ? (63u - clz(hi)) : (31u - clz((uint)v));   // index of the top set bit
        ulong e = (ulong)(1023 + (int)k - 53);
        ulong m = (v << (52u - k)) & 0xFFFFFFFFFFFFFul;
        return (e << 52) | m;
    }
    kernel void sel_random(device const uint* nPtr [[buffer(0)]], constant uint2& key [[buffer(1)]],
                           device ulong* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        uint4 r = sel_philox(uint4(i, 0u, 0u, 0u), key);
        ulong bits = ((ulong)r.y << 32) | (ulong)r.x;
        out[i] = sel_bits_to_unit(bits >> 11);
    }
    """
}
