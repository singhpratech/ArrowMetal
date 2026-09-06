import Foundation

/// MSL for the join operators `Kernels/Join.swift` does not cover: the match-flag scatter that
/// semi / anti / full outer joins need, and the as-of (nearest-key) join.
enum JoinExtraSource {

    /// One thread per index. `flags[idx[i]] = 1` for every valid entry — the "which rows of the other
    /// side did this join touch" bitmap that semi, anti and the right tail of a full outer join need.
    /// Writing a byte (not a bit) means no atomics: two threads that hit the same row write the same 1.
    static let scatter = KernelSource.prelude + """
    kernel void jx_scatter_flag(device const int* idx [[buffer(0)]],
                                device const uchar* validity [[buffer(1)]],
                                constant uint& hasValidity [[buffer(2)]],
                                device const uint* nPtr [[buffer(3)]],
                                constant uint& limit [[buffer(4)]],
                                device uchar* flags [[buffer(5)]],
                                uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if (hasValidity && !bit_get(validity, i)) return;
        int r = idx[i];
        if (r >= 0 && (uint)r < limit) flags[r] = 1;
    }
    """

    /// As-of join: for every probe row, the nearest build row in the same partition.
    ///
    /// The build side arrives sorted by `(partition, key)` ascending with its null keys already
    /// dropped, so both searches are plain binary searches over contiguous memory: first the
    /// partition's `[lo, hi)` range in `rp`, then the key inside it. That is `log2(nR)` dependent
    /// loads per probe row and no table, which is the whole reason an as-of join does not need a hash.
    ///
    /// `strategy` is 0 backward (last key <= probe), 1 forward (first key >= probe), 2 nearest
    /// (whichever of the two is closer, ties to the backward one — Polars' rule). `tolerance` bounds
    /// `|probe - match|`; without it any distance matches. A null probe key never matches.
    static let asof = KernelSource.prelude + """
    // First index in [lo, hi) whose partition is >= p.
    inline uint aj_lower_part(device const int* rp, uint lo, uint hi, int p) {
        while (lo < hi) { uint mid = lo + (hi - lo) / 2u; if (rp[mid] < p) lo = mid + 1u; else hi = mid; }
        return lo;
    }
    // First index in [lo, hi) whose partition is > p.
    inline uint aj_upper_part(device const int* rp, uint lo, uint hi, int p) {
        while (lo < hi) { uint mid = lo + (hi - lo) / 2u; if (rp[mid] <= p) lo = mid + 1u; else hi = mid; }
        return lo;
    }
    inline uint aj_lower_key(device const long* rk, uint lo, uint hi, long k) {
        while (lo < hi) { uint mid = lo + (hi - lo) / 2u; if (rk[mid] < k) lo = mid + 1u; else hi = mid; }
        return lo;
    }
    inline uint aj_upper_key(device const long* rk, uint lo, uint hi, long k) {
        while (lo < hi) { uint mid = lo + (hi - lo) / 2u; if (rk[mid] <= k) lo = mid + 1u; else hi = mid; }
        return lo;
    }

    kernel void jx_asof_probe(device const long* lk [[buffer(0)]],
                              device const uchar* lvalidity [[buffer(1)]],
                              constant uint& hasLValidity [[buffer(2)]],
                              device const int* lp [[buffer(3)]],
                              constant uint& hasPartition [[buffer(4)]],
                              device const uint* nPtr [[buffer(5)]],
                              device const long* rk [[buffer(6)]],
                              device const int* rp [[buffer(7)]],
                              constant uint& nR [[buffer(8)]],
                              constant uint& strategy [[buffer(9)]],
                              constant long& tolerance [[buffer(10)]],
                              constant uint& hasTolerance [[buffer(11)]],
                              device int* outIdx [[buffer(12)]],
                              device uchar* outValid [[buffer(13)]],
                              uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        outIdx[i] = 0;
        outValid[i] = 0;
        if (hasLValidity && !bit_get(lvalidity, i)) return;
        uint lo = 0u, hi = nR;
        if (hasPartition) {
            int p = lp[i];
            lo = aj_lower_part(rp, 0u, nR, p);
            hi = aj_upper_part(rp, lo, nR, p);
        }
        if (lo >= hi) return;
        long k = lk[i];
        uint back = aj_upper_key(rk, lo, hi, k);         // one past the last key <= k
        uint fwd  = aj_lower_key(rk, lo, hi, k);         // first key >= k
        bool hasBack = back > lo;
        bool hasFwd = fwd < hi;
        uint pick;
        if (strategy == 0u) {
            if (!hasBack) return;
            pick = back - 1u;
        } else if (strategy == 1u) {
            if (!hasFwd) return;
            pick = fwd;
        } else {
            if (!hasBack && !hasFwd) return;
            if (!hasFwd) pick = back - 1u;
            else if (!hasBack) pick = fwd;
            else {
                long db = k - rk[back - 1u];
                long df = rk[fwd] - k;
                pick = (db <= df) ? (back - 1u) : fwd;
            }
        }
        if (hasTolerance) {
            long d = rk[pick] - k;
            if (d < 0) d = -d;
            if (d > tolerance) return;
        }
        outIdx[i] = (int)pick;
        outValid[i] = 1;
    }
    """
}
