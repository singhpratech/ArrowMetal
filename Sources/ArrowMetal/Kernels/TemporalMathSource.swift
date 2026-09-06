import Foundation

/// MSL for temporal rounding and the calendar fields that `Temporal.swift`'s extraction kernel does
/// not cover.
///
/// This file carries its own copy of Howard Hinnant's civil-calendar algorithms from
/// "chrono-Compatible Low-Level Date Algorithms" (public domain) — `civil_from_days` and its inverse
/// `days_from_civil` — rather than reaching into `TemporalSource`, so the two kernels stay
/// independently readable and `Temporal.swift` is untouched. They are exact over the whole int64 day
/// range in the proleptic Gregorian calendar with astronomical year numbering (year 0 is 1 BC and is
/// a leap year). UTC only: no timezone and no leap seconds.
enum TemporalMathSource {
    static func source(T: String) -> String { KernelSource.prelude + """
    inline long tm_floordiv(long a, long b) {
        long q = a / b;
        if ((a % b != 0L) && ((a < 0L) != (b < 0L))) q -= 1L;
        return q;
    }
    // Days since 1970-01-01 -> proleptic Gregorian y / m [1,12] / d [1,31].
    inline void tm_civil_from_days(long z, thread long& y, thread long& m, thread long& d) {
        z += 719468L;
        long era = (z >= 0L ? z : z - 146096L) / 146097L;
        long doe = z - era * 146097L;                                          // [0, 146096]
        long yoe = (doe - doe / 1460L + doe / 36524L - doe / 146096L) / 365L;   // [0, 399]
        long yy = yoe + era * 400L;
        long doy = doe - (365L * yoe + yoe / 4L - yoe / 100L);                  // [0, 365]
        long mp = (5L * doy + 2L) / 153L;                                       // [0, 11]
        d = doy - (153L * mp + 2L) / 5L + 1L;
        m = mp + (mp < 10L ? 3L : -9L);
        y = yy + (m <= 2L ? 1L : 0L);
    }
    // The exact inverse: y / m / d -> days since 1970-01-01.
    inline long tm_days_from_civil(long y, long m, long d) {
        y -= (m <= 2L) ? 1L : 0L;
        long era = (y >= 0L ? y : y - 399L) / 400L;
        long yoe = y - era * 400L;                                             // [0, 399]
        long doy = (153L * (m + (m > 2L ? -3L : 9L)) + 2L) / 5L + d - 1L;       // [0, 365]
        long doe = yoe * 365L + yoe / 4L - yoe / 100L + doy;                    // [0, 146096]
        return era * 146097L + doe - 719468L;
    }

    // floor / ceil / round to a multiple of a unit.
    //
    //   mode: 0 floor, 1 ceil, 2 round (halves go up, that is toward +infinity)
    //   kind: 0 = a multiple of `p` ticks in the value's own resolution
    //         1 = a multiple of `p` calendar months, with `ticksPerDay` ticks in a day
    //
    // `ceil` leaves a value that already sits on a boundary alone, which is Arrow's
    // `ceil_is_strictly_greater = false` default.
    kernel void temporal_round(device const \(T)* vals [[buffer(0)]],
                               device const uint* nPtr [[buffer(1)]],
                               constant uint& mode [[buffer(2)]],
                               constant uint& kind [[buffer(3)]],
                               constant long& p [[buffer(4)]],
                               constant long& ticksPerDay [[buffer(5)]],
                               device \(T)* out [[buffer(6)]],
                               uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        long v = (long)vals[i];
        long lo, hi;
        if (kind == 0u) {
            lo = tm_floordiv(v, p) * p;
            hi = lo + p;
        } else {
            long days = tm_floordiv(v, ticksPerDay);
            long y, m, d;
            tm_civil_from_days(days, y, m, d);
            long months = y * 12L + (m - 1L);
            long fm = tm_floordiv(months, p) * p;
            long fy = tm_floordiv(fm, 12L);
            long cm = fm + p;
            long cy = tm_floordiv(cm, 12L);
            lo = tm_days_from_civil(fy, fm - fy * 12L + 1L, 1L) * ticksPerDay;
            hi = tm_days_from_civil(cy, cm - cy * 12L + 1L, 1L) * ticksPerDay;
        }
        long r;
        if (mode == 0u) r = lo;
        else if (mode == 1u) r = (v == lo) ? lo : hi;
        else r = (2L * (v - lo) >= (hi - lo)) ? hi : lo;
        out[i] = (\(T))r;
    }

    // Calendar fields beyond year / month / day / weekday / hour / minute / second.
    //
    //   field: 0 day_of_year (1-based), 1 quarter (1-4), 2 iso_week (1-53), 3 iso_year,
    //          4 millisecond, 5 microsecond, 6 nanosecond
    //   mode:  0 value is whole days, 1 ticks since the epoch, 2 ticks since midnight
    //
    // The subsecond fields follow Arrow: `millisecond` is the count since the last full second,
    // `microsecond` the count since the last full millisecond and `nanosecond` the count since the
    // last full microsecond. `nsPerTick` widens the value's own resolution to nanoseconds.
    kernel void temporal_fields(device const \(T)* vals [[buffer(0)]],
                                device const uint* nPtr [[buffer(1)]],
                                constant uint& field [[buffer(2)]],
                                constant uint& mode [[buffer(3)]],
                                constant long& divisor [[buffer(4)]],
                                constant long& nsPerTick [[buffer(5)]],
                                device int* out [[buffer(6)]],
                                uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        long v = (long)vals[i];
        long days = 0L, sub = 0L;
        if (mode == 0u) {
            days = v;
        } else {
            long s = tm_floordiv(v, divisor);
            sub = v - s * divisor;                      // always in [0, divisor)
            if (mode == 1u) days = tm_floordiv(s, 86400L);
        }
        int r = 0;
        switch (field) {
        case 0u: { long y, m, d; tm_civil_from_days(days, y, m, d);
                   r = (int)(days - tm_days_from_civil(y, 1L, 1L) + 1L); break; }
        case 1u: { long y, m, d; tm_civil_from_days(days, y, m, d); r = (int)((m + 2L) / 3L); break; }
        case 2u:
        case 3u: {
            long wd = ((days + 3L) % 7L + 7L) % 7L;      // Monday = 0
            long th = days - wd + 3L;                    // the Thursday of this ISO week
            long y, m, d; tm_civil_from_days(th, y, m, d);
            r = (field == 3u) ? (int)y : (int)((th - tm_days_from_civil(y, 1L, 1L)) / 7L + 1L);
            break;
        }
        case 4u: { r = (int)((sub * nsPerTick) / 1000000L); break; }
        case 5u: { r = (int)(((sub * nsPerTick) / 1000L) % 1000L); break; }
        default: { r = (int)((sub * nsPerTick) % 1000L); break; }
        }
        out[i] = r;
    }

    // is_leap_year -> packed boolean bitmap, one 32-bit word per thread.
    kernel void temporal_leap(device const \(T)* vals [[buffer(0)]],
                              device const uint* nPtr [[buffer(1)]],
                              constant uint& mode [[buffer(2)]],
                              constant long& divisor [[buffer(3)]],
                              device uint* out [[buffer(4)]],
                              uint w [[thread_position_in_grid]]) {
        uint n = *nPtr, base = w * 32u;
        if (base >= n) return;
        uint limit = min(32u, n - base), bits = 0u;
        for (uint j = 0; j < limit; j++) {
            long v = (long)vals[base + j];
            long days = (mode == 0u) ? v : tm_floordiv(tm_floordiv(v, divisor), 86400L);
            long y, m, d; tm_civil_from_days(days, y, m, d);
            bool leap = ((y % 4L) == 0L && (y % 100L) != 0L) || ((y % 400L) == 0L);
            if (leap) bits |= (1u << j);
        }
        out[w] = bits;
    }
    """ }
}
