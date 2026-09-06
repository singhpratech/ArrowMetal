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

    // floor / ceil / round to a multiple of a unit, with Arrow's whole `RoundTemporalOptions` surface.
    //
    //   mode: 0 floor, 1 ceil, 2 round (halves go up, that is toward +infinity, as Arrow does)
    //   kind: 0 = a multiple of `p` ticks in the value's own resolution (also how `week` arrives,
    //             as 7 * multiple days of ticks with a week-start origin)
    //         1 = a multiple of `p` calendar months counted from 1970-01, `ticksPerDay` ticks in a day
    //         2 = a multiple of `p` calendar years counted from year 0
    //   originKind: where the grid starts, which is what `calendar_based_origin` changes.
    //         kind 0:  0 the constant `originBase` ticks (the epoch, or the week anchor)
    //                  1 floor of the value to `originPeriod` ticks (start of the containing day,
    //                    hour, minute, second, millisecond or microsecond)
    //                  2 start of the containing month
    //                  3 start of the containing year
    //                  4 the week start on or before 1 January of the containing year, with
    //                    `originBase` the anchor *day* of the week grid
    //         kind 1:  0 from 1970-01, 1 from January of the containing year
    //         kind 2:  ignored — a year has no greater calendar unit
    //   flags: bit 0 = ceil_is_strictly_greater
    //   fineScale: when the unit is *finer* than the array's own tick (rounding a timestamp[s] to
    //         milliseconds, say), the whole computation happens in those finer units — the value is
    //         multiplied up, rounded there, and floored back. That is what Arrow does, and it is why
    //         such a call is not simply the identity: 13:47:33 floored to 7 ms is 13:47:32.998, which
    //         truncates back to 13:47:32. The step back is a truncation toward zero, not a floor,
    //         which is what Arrow does. `fineScale` is 1 for every other case.
    //
    // `ceil` leaves a value that already sits on a boundary alone unless bit 0 is set — except on the
    // calendar kinds, where Arrow's own `ceil` always advances a boundary value and its
    // `ceil_is_strictly_greater` makes no difference. That quirk is reproduced here on purpose.
    inline long tm_origin(long v, uint originKind, long originBase, long originPeriod, long ticksPerDay) {
        if (originKind == 0u) return originBase;
        if (originKind == 1u) return tm_floordiv(v, originPeriod) * originPeriod;
        long days = tm_floordiv(v, ticksPerDay);
        long y, m, d;
        tm_civil_from_days(days, y, m, d);
        if (originKind == 2u) return tm_days_from_civil(y, m, 1L) * ticksPerDay;
        long jan1 = tm_days_from_civil(y, 1L, 1L);
        if (originKind == 3u) return jan1 * ticksPerDay;
        long rel = jan1 - originBase;                       // days since the week anchor
        long back = rel - tm_floordiv(rel, 7L) * 7L;        // how far into its week 1 January sits
        return (jan1 - back) * ticksPerDay;
    }
    // Months since 1970-01 -> ticks of the first instant of that month.
    inline long tm_month_start(long months, long ticksPerDay) {
        long fy = tm_floordiv(months, 12L);
        return tm_days_from_civil(1970L + fy, months - fy * 12L + 1L, 1L) * ticksPerDay;
    }
    kernel void temporal_round(device const \(T)* vals [[buffer(0)]],
                               device const uint* nPtr [[buffer(1)]],
                               constant uint& mode [[buffer(2)]],
                               constant uint& kind [[buffer(3)]],
                               constant long& p [[buffer(4)]],
                               constant long& ticksPerDay [[buffer(5)]],
                               device \(T)* out [[buffer(6)]],
                               constant uint& originKind [[buffer(7)]],
                               constant long& originBase [[buffer(8)]],
                               constant long& originPeriod [[buffer(9)]],
                               constant uint& flags [[buffer(10)]],
                               constant long& fineScale [[buffer(11)]],
                               uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        long v = (long)vals[i] * fineScale;
        long lo, hi;
        if (kind == 0u) {
            long base = tm_origin(v, originKind, originBase, originPeriod, ticksPerDay);
            lo = base + tm_floordiv(v - base, p) * p;
            hi = lo + p;
        } else if (kind == 1u) {
            long y, m, d;
            tm_civil_from_days(tm_floordiv(v, ticksPerDay), y, m, d);
            long months = (y - 1970L) * 12L + (m - 1L);
            long base = (originKind == 1u) ? (y - 1970L) * 12L : 0L;
            long fm = base + tm_floordiv(months - base, p) * p;
            lo = tm_month_start(fm, ticksPerDay);
            hi = tm_month_start(fm + p, ticksPerDay);
        } else {
            long y, m, d;
            tm_civil_from_days(tm_floordiv(v, ticksPerDay), y, m, d);
            long fy = tm_floordiv(y, p) * p;
            lo = tm_days_from_civil(fy, 1L, 1L) * ticksPerDay;
            hi = tm_days_from_civil(fy + p, 1L, 1L) * ticksPerDay;
        }
        long r;
        if (mode == 0u) r = lo;
        else if (mode == 1u) {
            if (kind != 0u) r = hi;                                    // Arrow's calendar ceil always advances
            else if (v != lo) r = hi;
            else r = ((flags & 1u) != 0u) ? hi : lo;
        } else r = (2L * (v - lo) >= (hi - lo)) ? hi : lo;
        out[i] = (\(T))(fineScale == 1L ? r : (r / fineScale));   // truncates toward zero, as Arrow does
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
