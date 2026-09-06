import Foundation

/// MSL for the type-matrix types added in `TypesExtra.swift`: `float16`, `decimal32` / `decimal64`,
/// `fixed_size_binary` and the `interval` family.
///
/// Everything here works on raw fixed-width records, so one gather kernel (`fw_take`, parameterised by
/// the byte width) serves `fixed_size_binary`, all three interval layouts and any other opaque record
/// this package grows later.
enum TypesExtraSource {

    /// float16 <-> float32. Metal has a native `half`, so the conversion is one `as_type` plus a cast:
    /// widening is exact, narrowing rounds to nearest-even and overflows to +/-infinity.
    static let float16 = KernelSource.prelude + """
    kernel void f16_to_f32(device const ushort* src [[buffer(0)]],
                           device const uint* nPtr [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        out[i] = (float)as_type<half>(src[i]);
    }
    kernel void f32_to_f16(device const float* src [[buffer(0)]],
                           device const uint* nPtr [[buffer(1)]],
                           device ushort* out [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        out[i] = as_type<ushort>((half)src[i]);
    }
    """

    /// decimal32 / decimal64 <-> decimal128. Widening sign-extends the narrow value into two limbs;
    /// narrowing keeps the low limb, which wraps when the value does not fit (Arrow's unchecked cast).
    static let smallDecimal = KernelSource.prelude + """
    kernel void dec32_widen(device const int* src [[buffer(0)]],
                            device const uint* nPtr [[buffer(1)]],
                            device long* out [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        long v = (long)src[i];
        out[2 * i] = v;
        out[2 * i + 1] = (v < 0L) ? -1L : 0L;
    }
    kernel void dec64_widen(device const long* src [[buffer(0)]],
                            device const uint* nPtr [[buffer(1)]],
                            device long* out [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        long v = src[i];
        out[2 * i] = v;
        out[2 * i + 1] = (v < 0L) ? -1L : 0L;
    }
    kernel void dec32_narrow(device const long* src [[buffer(0)]],
                             device const uint* nPtr [[buffer(1)]],
                             device int* out [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        out[i] = (int)src[2 * i];
    }
    kernel void dec64_narrow(device const long* src [[buffer(0)]],
                             device const uint* nPtr [[buffer(1)]],
                             device long* out [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        out[i] = src[2 * i];
    }
    """

    /// Gather, byte compare and hash over fixed-width opaque records of `w` bytes.
    static let fixedWidth = KernelSource.prelude + """
    // flags: bit0 = source has validity, bit1 = indices have validity.
    kernel void fw_take(device const uchar* vals [[buffer(0)]],
                        device const uchar* validity [[buffer(1)]],
                        device const int* idx [[buffer(2)]],
                        device const uchar* idxValidity [[buffer(3)]],
                        constant uint& n [[buffer(4)]],
                        constant uint& srcLen [[buffer(5)]],
                        constant uint& flags [[buffer(6)]],
                        constant uint& w [[buffer(7)]],
                        device uchar* out [[buffer(8)]],
                        device uchar* outValidBytes [[buffer(9)]],
                        device atomic_uint* errorFlag [[buffer(10)]],
                        uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        device uchar* dst = out + (ulong)i * (ulong)w;
        if ((flags & 2u) && !bit_get(idxValidity, i)) {
            for (uint k = 0; k < w; k++) dst[k] = 0;
            outValidBytes[i] = 0;
            return;
        }
        long j = (long)idx[i];
        if (j < 0 || j >= (long)srcLen) {
            atomic_store_explicit(errorFlag, 1u, memory_order_relaxed);
            for (uint k = 0; k < w; k++) dst[k] = 0;
            outValidBytes[i] = 0;
            return;
        }
        device const uchar* src = vals + (ulong)j * (ulong)w;
        for (uint k = 0; k < w; k++) dst[k] = src[k];
        outValidBytes[i] = (flags & 1u) ? (bit_get(validity, (uint)j) ? 1 : 0) : 1;
    }

    // Byte equality against one scalar record. `ne` inverts the answer (not_equal).
    kernel void fw_cmp_scalar(device const uchar* vals [[buffer(0)]],
                              device const uchar* pat [[buffer(1)]],
                              device const uint* nPtr [[buffer(2)]],
                              constant uint& w [[buffer(3)]],
                              constant uint& ne [[buffer(4)]],
                              device uchar* outBytes [[buffer(5)]],
                              uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        device const uchar* src = vals + (ulong)i * (ulong)w;
        bool eq = true;
        for (uint k = 0; k < w; k++) { if (src[k] != pat[k]) { eq = false; break; } }
        outBytes[i] = (ne != 0u ? !eq : eq) ? 1 : 0;
    }

    // Element-wise byte equality between two arrays of the same width.
    kernel void fw_cmp_array(device const uchar* a [[buffer(0)]],
                             device const uchar* b [[buffer(1)]],
                             device const uint* nPtr [[buffer(2)]],
                             constant uint& w [[buffer(3)]],
                             constant uint& ne [[buffer(4)]],
                             device uchar* outBytes [[buffer(5)]],
                             uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        device const uchar* x = a + (ulong)i * (ulong)w;
        device const uchar* y = b + (ulong)i * (ulong)w;
        bool eq = true;
        for (uint k = 0; k < w; k++) { if (x[k] != y[k]) { eq = false; break; } }
        outBytes[i] = (ne != 0u ? !eq : eq) ? 1 : 0;
    }

    // FNV-1a over the record's bytes. Deterministic and endian-independent; not Arrow's `hash64`.
    kernel void fw_hash64(device const uchar* vals [[buffer(0)]],
                          device const uint* nPtr [[buffer(1)]],
                          constant uint& w [[buffer(2)]],
                          device ulong* out [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        device const uchar* src = vals + (ulong)i * (ulong)w;
        ulong h = 14695981039346656037UL;
        for (uint k = 0; k < w; k++) { h ^= (ulong)src[k]; h *= 1099511628211UL; }
        out[i] = h;
    }
    """

    /// Timestamp / date + interval on the civil calendar. `civil_from_days` and `days_from_civil` are
    /// Howard Hinnant's public-domain algorithms; month arithmetic clamps the day to the target month's
    /// length, which is what Arrow does.
    static let interval = KernelSource.prelude + """
    inline long iv_floordiv(long a, long b) {
        long q = a / b;
        if ((a % b != 0L) && ((a < 0L) != (b < 0L))) q -= 1L;
        return q;
    }
    inline void iv_civil_from_days(long z, thread long& y, thread long& m, thread long& d) {
        z += 719468L;
        long era = (z >= 0L ? z : z - 146096L) / 146097L;
        long doe = z - era * 146097L;
        long yoe = (doe - doe / 1460L + doe / 36524L - doe / 146096L) / 365L;
        long yy = yoe + era * 400L;
        long doy = doe - (365L * yoe + yoe / 4L - yoe / 100L);
        long mp = (5L * doy + 2L) / 153L;
        d = doy - (153L * mp + 2L) / 5L + 1L;
        m = mp + (mp < 10L ? 3L : -9L);
        y = yy + (m <= 2L ? 1L : 0L);
    }
    inline long iv_days_from_civil(long y, long m, long d) {
        y -= (m <= 2L) ? 1L : 0L;
        long era = (y >= 0L ? y : y - 399L) / 400L;
        long yoe = y - era * 400L;
        long doy = (153L * (m + (m > 2L ? -3L : 9L)) + 2L) / 5L + d - 1L;
        long doe = yoe * 365L + yoe / 4L - yoe / 100L + doy;
        return era * 146097L + doe - 719468L;
    }
    inline bool iv_is_leap(long y) { return (y % 4L == 0L && y % 100L != 0L) || (y % 400L == 0L); }
    inline long iv_days_in_month(long y, long m) {
        if (m == 2L) return iv_is_leap(y) ? 29L : 28L;
        if (m == 4L || m == 6L || m == 9L || m == 11L) return 30L;
        return 31L;
    }

    // ivUnit: 0 month ("tiM", one int32), 1 day_time ("tiD", two int32: days, milliseconds),
    //         2 month_day_nano ("tin", int32 months, int32 days, int64 nanoseconds).
    // mode:   0 the value is whole days (date32), 1 the value is ticks since the epoch.
    // The sub-day part of the interval is converted to the value's own resolution as
    // `sub * subNum / subDen` (truncating toward zero).
    kernel void temporal_add_interval(device const long* vals [[buffer(0)]],
                                      device const int* ivi [[buffer(1)]],
                                      device const long* ivl [[buffer(2)]],
                                      device const uint* nPtr [[buffer(3)]],
                                      constant uint& ivUnit [[buffer(4)]],
                                      constant uint& broadcast [[buffer(5)]],
                                      constant long& ticksPerDay [[buffer(6)]],
                                      constant long& subNum [[buffer(7)]],
                                      constant long& subDen [[buffer(8)]],
                                      constant uint& mode [[buffer(9)]],
                                      device long* out [[buffer(10)]],
                                      uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        uint j = (broadcast != 0u) ? 0u : i;
        long months = 0L, addDays = 0L, sub = 0L;
        if (ivUnit == 0u) {
            months = (long)ivi[j];
        } else if (ivUnit == 1u) {
            addDays = (long)ivi[2 * j];
            sub = (long)ivi[2 * j + 1];
        } else {
            months = (long)ivi[4 * j];
            addDays = (long)ivi[4 * j + 1];
            sub = ivl[2 * j + 1];
        }
        long v = vals[i];
        long days = (mode == 0u) ? v : iv_floordiv(v, ticksPerDay);
        long rem = (mode == 0u) ? 0L : (v - days * ticksPerDay);
        if (months != 0L) {
            long y, m, d;
            iv_civil_from_days(days, y, m, d);
            long total = y * 12L + (m - 1L) + months;
            long ny = iv_floordiv(total, 12L);
            long nm = total - ny * 12L + 1L;
            long dim = iv_days_in_month(ny, nm);
            long nd = (d > dim) ? dim : d;
            days = iv_days_from_civil(ny, nm, nd);
        }
        days += addDays;
        if (mode == 0u) {
            out[i] = days;
        } else {
            long ticks = (subDen == 1L) ? (sub * subNum) : ((sub * subNum) / subDen);
            out[i] = days * ticksPerDay + rem + ticks;
        }
    }

    // The three interval-producing `*_interval_between` functions. Both inputs are already in the same
    // unit; `ticksPerDay` and `nanoPerTick` describe it.
    //
    // mode 0 month_interval_between        -> int32 months
    //      1 day_time_interval_between     -> int32 days, int32 milliseconds
    //      2 month_day_nano_interval_between -> int32 months, int32 days, int64 nanoseconds
    //
    // Every field is the plain difference of the corresponding truncated field, which is how Arrow
    // defines these: months are month boundaries crossed, days are the difference of the day fields
    // (of the whole day for day_time), and the sub-day part is the difference of the two times of day.
    kernel void temporal_interval_between(device const long* a [[buffer(0)]],
                                          device const long* b [[buffer(1)]],
                                          device const uint* nPtr [[buffer(2)]],
                                          constant long& ticksPerDay [[buffer(3)]],
                                          constant long& nanoPerTick [[buffer(4)]],
                                          constant uint& mode [[buffer(5)]],
                                          device int* outI [[buffer(6)]],
                                          device long* outL [[buffer(7)]],
                                          uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        long va = a[i], vb = b[i];
        long da = iv_floordiv(va, ticksPerDay), db = iv_floordiv(vb, ticksPerDay);
        long ta = va - da * ticksPerDay, tb = vb - db * ticksPerDay;
        long y1, m1, d1, y2, m2, d2;
        iv_civil_from_days(da, y1, m1, d1);
        iv_civil_from_days(db, y2, m2, d2);
        long months = (y2 - y1) * 12L + (m2 - m1);
        long ns = (tb - ta) * nanoPerTick;
        if (mode == 0u) { outI[i] = (int)months; return; }
        if (mode == 1u) {
            outI[2 * i] = (int)(db - da);
            outI[2 * i + 1] = (int)(ns / 1000000L);
            return;
        }
        outI[4 * i] = (int)months;
        outI[4 * i + 1] = (int)(d2 - d1);
        outL[2 * i + 1] = ns;
    }
    """
}
