import Foundation

/// MSL for the Arrow temporal functions `Temporal.swift` and `Kernels/TemporalMath.swift` do not
/// cover: the option-carrying week numbers, the two struct-valued extractors, `subsecond`, and every
/// `*_between` difference.
///
/// Like `TemporalMathSource`, this file carries its own copy of Howard Hinnant's civil-calendar
/// algorithms from "chrono-Compatible Low-Level Date Algorithms" (public domain) — `civil_from_days`
/// and its inverse `days_from_civil` — so the kernels here stay independently readable and neither
/// `Temporal.swift` nor `TemporalMath.swift` is touched. They are exact over the whole int64 day
/// range in the proleptic Gregorian calendar with astronomical year numbering. UTC only: no timezone
/// and no leap seconds.
enum TemporalExtraSource {

    /// The shared calendar helpers, prefixed `tx_` so nothing collides with `TemporalSource`
    /// (`t_`/`civil_from_days`) or `TemporalMathSource` (`tm_`).
    private static let calendar = """
    inline long tx_floordiv(long a, long b) {
        long q = a / b;
        if ((a % b != 0L) && ((a < 0L) != (b < 0L))) q -= 1L;
        return q;
    }
    // Days since 1970-01-01 -> proleptic Gregorian y / m [1,12] / d [1,31].
    inline void tx_civil_from_days(long z, thread long& y, thread long& m, thread long& d) {
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
    inline long tx_days_from_civil(long y, long m, long d) {
        y -= (m <= 2L) ? 1L : 0L;
        long era = (y >= 0L ? y : y - 399L) / 400L;
        long yoe = y - era * 400L;                                             // [0, 399]
        long doy = (153L * (m + (m > 2L ? -3L : 9L)) + 2L) / 5L + d - 1L;       // [0, 365]
        long doe = yoe * 365L + yoe / 4L - yoe / 100L + doy;                    // [0, 146096]
        return era * 146097L + doe - 719468L;
    }
    // ISO weekday: 1 = Monday ... 7 = Sunday. 1970-01-01 was a Thursday.
    inline long tx_iso_weekday(long days) { return ((days + 3L) % 7L + 7L) % 7L + 1L; }
    // Days from the start of this value's week back to that start, for a week beginning on
    // `weekStart` (1 = Monday ... 7 = Sunday).
    inline long tx_week_offset(long days, long weekStart) {
        return ((tx_iso_weekday(days) - weekStart) % 7L + 7L) % 7L;
    }

    // Arrow `week` with all three WeekOptions, and the one formula every variant is a case of.
    //
    // The week owning a date runs from `ws` (its first day) for seven days. That week belongs to the
    // year of its "pivot": its first day when first_week_is_fully_in_year is set (so week 1 is the
    // first week lying wholly inside January), otherwise its fourth day, the ISO majority rule.
    // count_from_zero numbers the weeks relative to the date's own calendar year instead of the
    // week's, which is what makes a leading partial week come out as 0.
    //
    //   opts bit 0: week_starts_monday, bit 1: count_from_zero, bit 2: first_week_is_fully_in_year
    inline long tx_week(long days, uint opts) {
        long weekStart = ((opts & 1u) != 0u) ? 1L : 7L;
        long ws = days - tx_week_offset(days, weekStart);
        long pivot = ws + (((opts & 4u) != 0u) ? 0L : 3L);
        long y, m, d;
        tx_civil_from_days(((opts & 2u) != 0u) ? days : pivot, y, m, d);
        return tx_floordiv(pivot - tx_days_from_civil(y, 1L, 1L), 7L) + 1L;
    }
    """

    /// Week numbers, `us_year` and the option-carrying `day_of_week`, plus the struct extractors.
    /// `T` is the storage type of the input (`int` for date32, `long` otherwise).
    static func source(T: String) -> String { KernelSource.prelude + calendar + """

    // field: 0 week (p1 = WeekOptions bits), 1 us_year, 2 day_of_week (p1 = count_from_zero,
    //        p2 = week start, 1 = Monday ... 7 = Sunday).
    // `ticksPerDay` turns the stored value into days since the epoch (1 for date32).
    kernel void temporal_extra_field(device const \(T)* vals [[buffer(0)]],
                                     device const uint* nPtr [[buffer(1)]],
                                     constant uint& field [[buffer(2)]],
                                     constant long& ticksPerDay [[buffer(3)]],
                                     constant uint& p1 [[buffer(4)]],
                                     constant long& p2 [[buffer(5)]],
                                     device long* out [[buffer(6)]],
                                     uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        long days = tx_floordiv((long)vals[i], ticksPerDay);
        long r = 0L;
        if (field == 0u) {
            r = tx_week(days, p1);
        } else if (field == 1u) {
            // US epidemiological year: the year owning the Wednesday of this Sunday-start week.
            long y, m, d;
            tx_civil_from_days(days - tx_week_offset(days, 7L) + 3L, y, m, d);
            r = y;
        } else {
            r = tx_week_offset(days, p2) + ((p1 != 0u) ? 0L : 1L);
        }
        out[i] = r;
    }

    // Struct-valued extractors, three int64 children written in one pass.
    // field: 0 year_month_day, 1 iso_calendar (iso_year, iso_week, iso_day_of_week).
    kernel void temporal_extra_struct(device const \(T)* vals [[buffer(0)]],
                                      device const uint* nPtr [[buffer(1)]],
                                      constant uint& field [[buffer(2)]],
                                      constant long& ticksPerDay [[buffer(3)]],
                                      device long* o0 [[buffer(4)]],
                                      device long* o1 [[buffer(5)]],
                                      device long* o2 [[buffer(6)]],
                                      uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        long days = tx_floordiv((long)vals[i], ticksPerDay);
        long y, m, d;
        if (field == 0u) {
            tx_civil_from_days(days, y, m, d);
            o0[i] = y; o1[i] = m; o2[i] = d;
        } else {
            long wd = tx_iso_weekday(days);                 // 1 = Monday
            long th = days - wd + 4L;                        // the Thursday of this ISO week
            tx_civil_from_days(th, y, m, d);
            o0[i] = y;
            o1[i] = (th - tx_days_from_civil(y, 1L, 1L)) / 7L + 1L;
            o2[i] = wd;
        }
    }

    // Every `*_between`. Both sides arrive widened to int64 and are first mapped onto a common
    // ruler by `floor(v * num / den)`: days since the epoch for the calendar kinds, ticks of the
    // op's own unit for kind 4. Exactly one of num / den is ever different from 1.
    //
    //   kind: 0 years, 1 quarters, 2 months, 3 weeks (param = week start, 1 = Monday ... 7 = Sunday),
    //         4 a fixed unit (hours ... nanoseconds)
    //
    // Every kind counts *boundaries crossed*, which is Arrow's definition: each side is truncated to
    // the unit first and the difference taken afterwards, so it is not the truncated difference.
    kernel void temporal_between(device const long* a [[buffer(0)]],
                                 device const long* b [[buffer(1)]],
                                 device const uint* nPtr [[buffer(2)]],
                                 constant uint& kind [[buffer(3)]],
                                 constant long& aNum [[buffer(4)]],
                                 constant long& aDen [[buffer(5)]],
                                 constant long& bNum [[buffer(6)]],
                                 constant long& bDen [[buffer(7)]],
                                 constant long& param [[buffer(8)]],
                                 device long* out [[buffer(9)]],
                                 uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        long av = tx_floordiv(a[i] * aNum, aDen);
        long bv = tx_floordiv(b[i] * bNum, bDen);
        long r;
        if (kind == 4u) {
            r = bv - av;
        } else if (kind == 3u) {
            long wa = av - tx_week_offset(av, param);
            long wb = bv - tx_week_offset(bv, param);
            r = (wb - wa) / 7L;
        } else {
            long y0, m0, d0, y1, m1, d1;
            tx_civil_from_days(av, y0, m0, d0);
            tx_civil_from_days(bv, y1, m1, d1);
            if (kind == 0u) r = y1 - y0;
            else if (kind == 1u) r = (y1 * 4L + (m1 - 1L) / 3L) - (y0 * 4L + (m0 - 1L) / 3L);
            else r = (y1 * 12L + (m1 - 1L)) - (y0 * 12L + (m0 - 1L));
        }
        out[i] = r;
    }
    """ }

    /// `subsecond`: the fraction of a second, as float64. Metal has no `double`, so the quotient is
    /// built with `DoubleMath`'s software binary64 — `d_div` is correctly rounded, so the result is
    /// bit-for-bit the `(double)sub / (double)ticksPerSecond` Arrow computes on the host.
    static func subsecondSource(T: String) -> String {
        KernelSource.prelude + DoubleMath.msl + """

        inline long txs_floordiv(long a, long b) {
            long q = a / b;
            if ((a % b != 0L) && ((a < 0L) != (b < 0L))) q -= 1L;
            return q;
        }
        // Exact ulong -> binary64. d_finish normalises the significand, so any magnitude works; the
        // exponent 1078 = 1023 + 55 is the one that makes the leading bit land on 2^0.
        inline ulong txs_d_from_ulong(ulong v) { return (v == 0ul) ? 0ul : d_finish(0ul, 1078L, v); }

        kernel void temporal_subsecond(device const \(T)* vals [[buffer(0)]],
                                       device const uint* nPtr [[buffer(1)]],
                                       constant long& divisor [[buffer(2)]],
                                       device ulong* out [[buffer(3)]],
                                       uint i [[thread_position_in_grid]]) {
            if (i >= *nPtr) return;
            long v = (long)vals[i];
            long sub = v - txs_floordiv(v, divisor) * divisor;       // always in [0, divisor)
            out[i] = d_div(txs_d_from_ulong((ulong)sub), txs_d_from_ulong((ulong)divisor));
        }
        """
    }
}
