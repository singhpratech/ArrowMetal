import Foundation

/// MSL for the timezone kernels: `local_timestamp`, `assume_timezone`, `is_dst` and the raw UTC-offset
/// lookup, all driven by the transition table `TimezoneGPU.swift` builds on the host.
///
/// The table is a sorted list of `T` transition instants plus `T + 1` interval records. Interval `i`
/// runs over the UTC seconds `[trans[i-1], trans[i])` (open at both ends for the first and last) and
/// carries one UTC offset and one DST flag. Every kernel here is therefore one branchless binary
/// search — `tz_interval`, about `log2(T)` steps, ten for the ~560 transitions a zone like
/// `America/New_York` has between 1800 and 2200 — followed by an add.
///
/// `assume_timezone` searches a *second* sorted key, `transLocal[i] = trans[i] + offset[i+1]`, the
/// local wall-clock instant at which interval `i + 1` begins. That is enough to decide the three
/// cases Arrow distinguishes without any second lookup:
///
/// * interval `k` (the one the search lands in) is valid for the local time `L` when `L` is still
///   before its own end in local terms, `trans[k] + offset[k]`;
/// * interval `k - 1` is valid when `L` is before *its* end, `trans[k-1] + offset[k-1]` — which can
///   only happen when the offset shrank at `trans[k-1]`, i.e. a fall-back, so both being valid is
///   exactly Arrow's **ambiguous**;
/// * neither being valid means `L` sits in the gap a spring-forward opened at `trans[k]`, which is
///   Arrow's **nonexistent**.
///
/// The sub-second part of a value never takes part: tz offsets are whole seconds and have been for
/// every zone since 1972, so the value is split into seconds and a remainder and the remainder is
/// carried across untouched, exactly as the host path does it.
enum TimezoneGPUSource {

    /// The pieces `TemporalFormatSource` also needs (`strftime`'s `%z` / `%Z` apply the same table).
    static let helpers = """
    // Floor division; values before 1970 are negative.
    inline long tz_floordiv(long a, long b) {
        long q = a / b;
        if ((a % b != 0L) && ((a < 0L) != (b < 0L))) q -= 1L;
        return q;
    }
    // How many of key[0 ..< T) are <= x. With `key` the sorted transition instants that is exactly the
    // index of the interval owning x, in [0, T].
    inline uint tz_interval(device const long* key, uint T, long x) {
        uint lo = 0u, hi = T;
        while (lo < hi) {
            uint mid = lo + ((hi - lo) >> 1);
            if (key[mid] <= x) lo = mid + 1u; else hi = mid;
        }
        return lo;
    }
    // Parameters shared by every kernel here and by the strftime / strptime kernels.
    struct tz_params {
        long per;        // ticks of the column's unit in one second
        long loSecond;   // the first UTC second the table describes
        long hiSecond;   // one past the last
        uint T;          // transition count
        uint hasValidity;
        uint ambiguous;   // 0 raise, 1 earliest, 2 latest
        uint nonexistent; // 0 raise, 1 earliest, 2 latest
    };
    // Flag slots, all atomic minimums over the row index so the reported row is deterministic:
    //   0 a value outside [loSecond, hiSecond), 1 an ambiguous local time, 2 a nonexistent one.
    #define TZ_NONE 0xFFFFFFFFu
    inline void tz_flag(device atomic_uint* flags, uint slot, uint i) {
        atomic_fetch_min_explicit(&flags[slot], i, memory_order_relaxed);
    }
    """

    static let source = KernelSource.prelude + helpers + """

    // local_timestamp: the wall-clock time each instant names in the column's own timezone.
    kernel void tz_local(device const long* vals [[buffer(0)]],
                         device const uchar* validity [[buffer(1)]],
                         device const uint* nPtr [[buffer(2)]],
                         constant tz_params& P [[buffer(3)]],
                         device const long* transUTC [[buffer(4)]],
                         device const int* offsets [[buffer(5)]],
                         device long* out [[buffer(6)]],
                         device atomic_uint* flags [[buffer(7)]],
                         uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if (P.hasValidity != 0u && !bit_get(validity, i)) { out[i] = 0L; return; }
        long v = vals[i];
        long s = tz_floordiv(v, P.per);
        long sub = v - s * P.per;
        if (s < P.loSecond || s >= P.hiSecond) { tz_flag(flags, 0u, i); out[i] = 0L; return; }
        uint k = tz_interval(transUTC, P.T, s);
        out[i] = (s + (long)offsets[k]) * P.per + sub;
    }

    // The UTC offset in seconds that applies to each value, as int32.
    kernel void tz_offset(device const long* vals [[buffer(0)]],
                          device const uchar* validity [[buffer(1)]],
                          device const uint* nPtr [[buffer(2)]],
                          constant tz_params& P [[buffer(3)]],
                          device const long* transUTC [[buffer(4)]],
                          device const int* offsets [[buffer(5)]],
                          device int* out [[buffer(6)]],
                          device atomic_uint* flags [[buffer(7)]],
                          uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if (P.hasValidity != 0u && !bit_get(validity, i)) { out[i] = 0; return; }
        long s = tz_floordiv(vals[i], P.per);
        if (s < P.loSecond || s >= P.hiSecond) { tz_flag(flags, 0u, i); out[i] = 0; return; }
        out[i] = offsets[tz_interval(transUTC, P.T, s)];
    }

    // is_dst -> packed boolean bitmap, one 32-bit word per thread so no atomics are needed.
    kernel void tz_is_dst(device const long* vals [[buffer(0)]],
                          device const uchar* validity [[buffer(1)]],
                          device const uint* nPtr [[buffer(2)]],
                          constant tz_params& P [[buffer(3)]],
                          device const long* transUTC [[buffer(4)]],
                          device const uchar* dstFlags [[buffer(5)]],
                          device uint* out [[buffer(6)]],
                          device atomic_uint* flags [[buffer(7)]],
                          uint w [[thread_position_in_grid]]) {
        uint n = *nPtr, base = w * 32u;
        if (base >= n) return;
        uint limit = min(32u, n - base), bits = 0u;
        for (uint j = 0; j < limit; j++) {
            uint i = base + j;
            if (P.hasValidity != 0u && !bit_get(validity, i)) continue;
            long s = tz_floordiv(vals[i], P.per);
            if (s < P.loSecond || s >= P.hiSecond) { tz_flag(flags, 0u, i); continue; }
            if (dstFlags[tz_interval(transUTC, P.T, s)] != 0u) bits |= (1u << j);
        }
        out[w] = bits;
    }

    // assume_timezone: read the values as wall-clock times in the zone and return the instants they name.
    kernel void tz_assume(device const long* vals [[buffer(0)]],
                          device const uchar* validity [[buffer(1)]],
                          device const uint* nPtr [[buffer(2)]],
                          constant tz_params& P [[buffer(3)]],
                          device const long* transUTC [[buffer(4)]],
                          device const long* transLocal [[buffer(5)]],
                          device const int* offsets [[buffer(6)]],
                          device long* out [[buffer(7)]],
                          device atomic_uint* flags [[buffer(8)]],
                          uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if (P.hasValidity != 0u && !bit_get(validity, i)) { out[i] = 0L; return; }
        long v = vals[i];
        long s = tz_floordiv(v, P.per);          // the local wall clock, in seconds
        long sub = v - s * P.per;
        if (s < P.loSecond || s >= P.hiSecond) { tz_flag(flags, 0u, i); out[i] = 0L; return; }
        uint k = tz_interval(transLocal, P.T, s);
        // Interval k is valid while the local time is still before its own end.
        bool okK = (k >= P.T) || (s < transUTC[k] + (long)offsets[k]);
        // Interval k - 1 can only still be running when the offset shrank at trans[k-1] (a fall-back).
        bool okPrev = (k > 0u) && (s < transUTC[k - 1u] + (long)offsets[k - 1u]);
        if (okK && okPrev) {                                    // ambiguous: the hour happens twice
            if (P.ambiguous == 0u) { tz_flag(flags, 1u, i); out[i] = 0L; return; }
            long a = (long)offsets[k - 1u], b = (long)offsets[k];
            // The earlier of the two instants is the one with the larger offset.
            long o = (P.ambiguous == 1u) ? max(a, b) : min(a, b);
            out[i] = (s - o) * P.per + sub;
        } else if (okK) {
            out[i] = (s - (long)offsets[k]) * P.per + sub;
        } else if (okPrev) {                                    // unreachable for a real zone
            out[i] = (s - (long)offsets[k - 1u]) * P.per + sub;
        } else {                                                // nonexistent: a spring-forward gap
            if (P.nonexistent == 0u) { tz_flag(flags, 2u, i); out[i] = 0L; return; }
            long t = transUTC[k];                               // k < T here, or okK would hold
            out[i] = (P.nonexistent == 1u) ? (t * P.per - 1L) : (t * P.per);
        }
    }
    """
}
