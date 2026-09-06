import Foundation

/// MSL for `strftime` and `strptime`.
///
/// The format string is compiled on the host (`TemporalFormat.compile`) into a flat list of `uint4`
/// operations uploaded as one small buffer, so a single generic kernel serves every format: op `x` is
/// the kind, `y` `z` `w` its arguments. Literal runs are merged and their bytes live in a second buffer.
/// Nothing about the format is baked into the shader, so no format ever triggers a recompile.
///
/// `strftime` is the same two-pass shape as the integer→string cast: one kernel measures each row's
/// output (`%Y` widens past four digits, `%B` and `%Z` are variable, `%e` pads with a space), the host
/// scans the lengths into the Arrow offsets buffer on the GPU, and a second kernel emits the bytes.
/// Both passes run the *same* `fmt_row`, once with `emit` false and once true, so a length and the
/// bytes that fill it cannot disagree.
///
/// The month and day names are C-locale constants baked into the source as one byte blob plus offset
/// and length tables. `%z` and `%Z` read the timezone transition table `TimezoneGPU.swift` uploads, so
/// a `timestamp` carrying a timezone formats in that zone, exactly as pyarrow's `strftime` does.
enum TemporalFormatSource {

    static let monthAbbrev = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    static let monthFull = ["January", "February", "March", "April", "May", "June",
                            "July", "August", "September", "October", "November", "December"]
    static let dayAbbrev = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    static let dayFull = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    static let meridiem = ["AM", "PM"]

    /// Every name in one blob, in the order the `fmt_name_base` table indexes them.
    static let names: [String] = monthAbbrev + monthFull + dayAbbrev + dayFull + meridiem
    /// First index of each table inside `names`: month abbrev, month full, day abbrev, day full, AM/PM.
    static let nameBase = [0, 12, 24, 31, 38]

    private static var nameTables: String {
        var bytes: [UInt8] = []
        var offs: [Int] = []
        var lens: [Int] = []
        for n in names {
            offs.append(bytes.count)
            let b = Array(n.utf8)
            lens.append(b.count)
            bytes += b
        }
        func list(_ a: [Int]) -> String { a.map(String.init).joined(separator: ",") }
        return """
        constant uchar fmt_name_bytes[\(bytes.count)] = {\(list(bytes.map(Int.init)))};
        constant uint fmt_name_off[\(offs.count)] = {\(list(offs))};
        constant uint fmt_name_len[\(lens.count)] = {\(list(lens))};
        constant uint fmt_name_base[\(nameBase.count)] = {\(list(nameBase))};

        """
    }

    static let source = KernelSource.prelude + TimezoneGPUSource.helpers + nameTables + """
    // Howard Hinnant's civil-calendar algorithms (public domain), exact over the whole int64 day range
    // in the proleptic Gregorian calendar with astronomical year numbering.
    inline void fmt_civil_from_days(long z, thread long& y, thread long& m, thread long& d) {
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
    inline long fmt_days_from_civil(long y, long m, long d) {
        y -= (m <= 2L) ? 1L : 0L;
        long era = (y >= 0L ? y : y - 399L) / 400L;
        long yoe = y - era * 400L;
        long doy = (153L * (m + (m > 2L ? -3L : 9L)) + 2L) / 5L + d - 1L;
        long doe = yoe * 365L + yoe / 4L - yoe / 100L + doy;
        return era * 146097L + doe - 719468L;
    }

    struct fmt_params {
        long divisor;      // ticks of the column's unit in one second (1 for date32)
        uint mode;         // 0 the value is whole days, 1 ticks since the epoch
        uint nops;
        uint T;            // timezone transition count
        uint hasValidity;
        uint hasTZ;
        uint pad;
    };

    // Right-aligns `value` in at least `width` characters, padding with '0' (pad 0) or ' ' (pad 1).
    // A negative value takes a leading '-' before the padding, which only `%Y` can produce.
    inline uint fmt_num(long value, uint width, uint pad, bool emit, device uchar* out, uint pos) {
        bool neg = value < 0L;
        ulong m = neg ? ((ulong)(-(value + 1L)) + 1UL) : (ulong)value;
        uint d = 1u;
        for (ulong t = m; t >= 10UL; t /= 10UL) d++;
        uint body = max(d, width);
        if (emit) {
            uint p = pos;
            if (neg) out[p++] = 0x2Du;
            uchar padc = (pad == 0u) ? 0x30u : 0x20u;
            for (uint k = d; k < body; k++) out[p++] = padc;
            for (int k = (int)d - 1; k >= 0; k--) { out[p + (uint)k] = (uchar)(0x30u + (uint)(m % 10UL)); m /= 10UL; }
        }
        return body + (neg ? 1u : 0u);
    }

    // One row of strftime. Returns the byte count; writes at out[pos0 ...] when `emit`.
    //
    // op.x kinds: 0 literal (y = offset, z = length), 1 number (y = field, z = width, w = pad),
    //             2 name (y = table), 3 six-digit fraction, 4 %z, 5 %Z.
    inline uint fmt_row(long v, constant fmt_params& P,
                        device const uint4* ops, device const uchar* lit,
                        device const long* transUTC, device const int* zoffs, device const uchar* zabbr,
                        bool emit, device uchar* out, uint pos0) {
        long seconds, sub = 0L;
        if (P.mode == 0u) { seconds = v * 86400L; }
        else { long s = tz_floordiv(v, P.divisor); sub = v - s * P.divisor; seconds = s; }
        long zoff = 0L;
        uint zi = 0u;
        if (P.hasTZ != 0u) { zi = tz_interval(transUTC, P.T, seconds); zoff = (long)zoffs[zi]; }
        long local = seconds + zoff;
        long days = tz_floordiv(local, 86400L);
        long sod = local - days * 86400L;
        long y, mo, dy;
        fmt_civil_from_days(days, y, mo, dy);
        long hh = sod / 3600L, mi = (sod / 60L) % 60L, ss = sod % 60L;
        long wdSun = ((days + 4L) % 7L + 7L) % 7L;                  // 0 = Sunday
        long micros = (P.divisor <= 1000000L) ? sub * (1000000L / P.divisor) : sub / (P.divisor / 1000000L);

        uint n = 0u;
        for (uint k = 0u; k < P.nops; k++) {
            uint4 op = ops[k];
            if (op.x == 0u) {
                if (emit) { for (uint j = 0u; j < op.z; j++) out[pos0 + n + j] = lit[op.y + j]; }
                n += op.z;
            } else if (op.x == 1u) {
                long val = 0L;
                switch (op.y) {
                case 0u: val = y; break;
                case 1u: val = mo; break;
                case 2u: val = dy; break;
                case 3u: val = hh; break;
                case 4u: val = mi; break;
                case 5u: val = ss; break;
                case 6u: val = days - fmt_days_from_civil(y, 1L, 1L) + 1L; break;         // %j
                case 7u: { long r = y % 100L; val = (r < 0L) ? r + 100L : r; break; }     // %y
                case 8u: { long h = hh % 12L; val = (h == 0L) ? 12L : h; break; }         // %I
                case 9u: val = tz_floordiv(y, 100L); break;                               // %C
                case 10u:                                                                 // %G
                case 11u: {                                                               // %V
                    long wdMon = ((days + 3L) % 7L + 7L) % 7L;
                    long th = days - wdMon + 3L;                                          // this ISO week's Thursday
                    long iy, im, id;
                    fmt_civil_from_days(th, iy, im, id);
                    val = (op.y == 10u) ? iy : ((th - fmt_days_from_civil(iy, 1L, 1L)) / 7L + 1L);
                    break;
                }
                case 12u: { long wdMon = ((days + 3L) % 7L + 7L) % 7L; val = wdMon + 1L; break; }  // %u
                default: val = wdSun; break;                                              // %w
                }
                n += fmt_num(val, op.z, op.w, emit, out, pos0 + n);
            } else if (op.x == 2u) {
                uint idx;
                if (op.y == 0u || op.y == 1u) idx = (uint)(mo - 1L);
                else if (op.y == 2u || op.y == 3u) idx = (uint)wdSun;
                else idx = (hh < 12L) ? 0u : 1u;
                uint e = fmt_name_base[op.y] + idx;
                uint off = fmt_name_off[e], len = fmt_name_len[e];
                if (emit) { for (uint j = 0u; j < len; j++) out[pos0 + n + j] = fmt_name_bytes[off + j]; }
                n += len;
            } else if (op.x == 3u) {
                n += fmt_num(micros, 6u, 0u, emit, out, pos0 + n);                        // %f
            } else if (op.x == 4u) {                                                      // %z
                long a = zoff < 0L ? -zoff : zoff;
                if (emit) {
                    uint p = pos0 + n;
                    out[p] = (zoff < 0L) ? 0x2Du : 0x2Bu;
                    long hrs = a / 3600L, mins = (a / 60L) % 60L;
                    out[p + 1u] = (uchar)(0x30u + (uint)(hrs / 10L));
                    out[p + 2u] = (uchar)(0x30u + (uint)(hrs % 10L));
                    out[p + 3u] = (uchar)(0x30u + (uint)(mins / 10L));
                    out[p + 4u] = (uchar)(0x30u + (uint)(mins % 10L));
                }
                n += 5u;
            } else {                                                                      // %Z
                uint base = zi * 8u, len = 0u;
                while (len < 7u && zabbr[base + len] != 0u) len++;
                if (emit) { for (uint j = 0u; j < len; j++) out[pos0 + n + j] = zabbr[base + j]; }
                n += len;
            }
        }
        return n;
    }

    // Pass 1: the byte length of every row. A null row produces no bytes.
    kernel void fmt_lengths(device const long* vals [[buffer(0)]],
                            device const uchar* validity [[buffer(1)]],
                            device const uint* nPtr [[buffer(2)]],
                            constant fmt_params& P [[buffer(3)]],
                            device const uint4* ops [[buffer(4)]],
                            device const uchar* lit [[buffer(5)]],
                            device const long* transUTC [[buffer(6)]],
                            device const int* zoffs [[buffer(7)]],
                            device const uchar* zabbr [[buffer(8)]],
                            device int* outLens [[buffer(9)]],
                            uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if (P.hasValidity != 0u && !bit_get(validity, i)) { outLens[i] = 0; return; }
        // `emit` is false, so the sink is never written; it only has to be a valid device pointer.
        outLens[i] = (int)fmt_row(vals[i], P, ops, lit, transUTC, zoffs, zabbr,
                                  false, (device uchar*)outLens, 0u);
    }

    // Pass 2: the bytes, at outOffsets[i].
    kernel void fmt_write(device const long* vals [[buffer(0)]],
                          device const uchar* validity [[buffer(1)]],
                          device const uint* nPtr [[buffer(2)]],
                          constant fmt_params& P [[buffer(3)]],
                          device const uint4* ops [[buffer(4)]],
                          device const uchar* lit [[buffer(5)]],
                          device const long* transUTC [[buffer(6)]],
                          device const int* zoffs [[buffer(7)]],
                          device const uchar* zabbr [[buffer(8)]],
                          device const int* outOffsets [[buffer(9)]],
                          device uchar* outData [[buffer(10)]],
                          uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if (P.hasValidity != 0u && !bit_get(validity, i)) return;
        fmt_row(vals[i], P, ops, lit, transUTC, zoffs, zabbr, true, outData, (uint)outOffsets[i]);
    }

    // ------------------------------------------------------------------ strptime

    struct prs_params {
        long scale;        // ticks of the target unit in one second
        long subScale;     // ticks of the target unit in one microsecond, or 0 when the unit is coarser
        long subDivide;    // microseconds per tick when the unit is coarser than a microsecond
        uint nops;
        uint hasValidity;
    };

    inline bool prs_space(uchar c) { return c == 0x20u || (c >= 0x09u && c <= 0x0Du); }

    // BSD `conv_num`: at least one digit, at most `width`, stopping early once another digit would
    // exceed `hi`; the result must land in [lo, hi]. Returns false on failure.
    inline bool prs_num(device const uchar* data, thread int& p, int e, uint width, uint lo, uint hi,
                        thread uint& outv) {
        if (p >= e) return false;
        uchar c = data[p];
        if (c < 0x30u || c > 0x39u) return false;
        uint r = 0u, used = 0u;
        while (p < e && used < width) {
            c = data[p];
            if (c < 0x30u || c > 0x39u) break;
            uint next = r * 10u + (uint)(c - 0x30u);
            if (used > 0u && next > hi) break;
            r = next; p++; used++;
        }
        if (r < lo || r > hi) return false;
        outv = r;
        return true;
    }

    // Case-insensitive match of one of the `count` names starting at `base`; returns the index or -1.
    inline int prs_name(device const uchar* data, thread int& p, int e, uint base, uint count) {
        for (uint k = 0u; k < count; k++) {
            uint off = fmt_name_off[base + k], len = fmt_name_len[base + k];
            if (p + (int)len > e) continue;
            bool ok = true;
            for (uint j = 0u; j < len; j++) {
                uchar a = data[p + (int)j], b = fmt_name_bytes[off + j];
                if (a >= 0x41u && a <= 0x5Au) a += 32u;
                if (b >= 0x41u && b <= 0x5Au) b += 32u;
                if (a != b) { ok = false; break; }
            }
            if (ok) { p += (int)len; return (int)k; }
        }
        return -1;
    }

    // One thread per 32 rows, so the thread owns a whole validity word and no atomics are needed.
    //
    // op.x kinds: 0 literal (y = offset, z = length), 1 number (y = field, z = width, w = lo | hi << 16),
    //             2 name (y = table), 3 fraction, 4 %z, 6 whitespace run.
    kernel void prs_parse(device const int* offsets [[buffer(0)]],
                          device const uchar* data [[buffer(1)]],
                          device const uint* nPtr [[buffer(2)]],
                          device const uchar* inValidity [[buffer(3)]],
                          constant prs_params& P [[buffer(4)]],
                          device const uint4* ops [[buffer(5)]],
                          device const uchar* lit [[buffer(6)]],
                          device long* outVals [[buffer(7)]],
                          device uint* outValid [[buffer(8)]],
                          uint w [[thread_position_in_grid]]) {
        uint n = *nPtr, base = w * 32u;
        if (base >= n) return;
        uint limit = min(32u, n - base), bits = 0u;
        for (uint j = 0u; j < limit; j++) {
            uint i = base + j;
            outVals[i] = 0L;
            if (P.hasValidity != 0u && !bit_get(inValidity, i)) continue;
            int p = offsets[i], e = offsets[i + 1];
            long year = 1970L, month = 1L, day = 1L, hour = 0L, minute = 0L, second = 0L, micros = 0L;
            long zoff = 0L;
            int hour12 = -1, pm = -1;
            bool ok = true;
            for (uint k = 0u; k < P.nops && ok; k++) {
                uint4 op = ops[k];
                if (op.x == 0u) {
                    if (p + (int)op.z > e) { ok = false; break; }
                    for (uint q = 0u; q < op.z; q++) if (data[p + (int)q] != lit[op.y + q]) { ok = false; break; }
                    p += (int)op.z;
                } else if (op.x == 6u) {
                    while (p < e && prs_space(data[p])) p++;
                } else if (op.x == 1u) {
                    uint lo = op.w & 0xFFFFu, hi = op.w >> 16;
                    uint val = 0u;
                    // No sign is accepted, exactly as BSD `conv_num` requires a digit first, so "+2020"
                    // and "-0001" fail the same way the C library fails them.
                    if (!prs_num(data, p, e, op.z, lo, hi, val)) { ok = false; break; }
                    long v = (long)val;
                    switch (op.y) {
                    case 0u: year = v; break;
                    case 1u: month = v; break;
                    case 2u: day = v; break;
                    case 3u: hour = v; break;
                    case 4u: minute = v; break;
                    case 5u: second = v; break;
                    case 7u: year = (v <= 68L) ? (2000L + v) : (1900L + v); break;
                    default: hour12 = (int)v; break;                     // %I
                    }
                } else if (op.x == 2u) {
                    uint tbl = op.y;
                    uint count = (tbl == 0u || tbl == 1u) ? 12u : ((tbl == 4u) ? 2u : 7u);
                    // Month and weekday names accept the full or the abbreviated spelling, as C does;
                    // the full names are tried first so "June" is not eaten as "Jun".
                    int idx = -1;
                    if (tbl == 0u || tbl == 1u) {
                        idx = prs_name(data, p, e, fmt_name_base[1], 12u);
                        if (idx < 0) idx = prs_name(data, p, e, fmt_name_base[0], 12u);
                        if (idx >= 0) month = (long)idx + 1L;
                    } else if (tbl == 2u || tbl == 3u) {
                        idx = prs_name(data, p, e, fmt_name_base[3], 7u);
                        if (idx < 0) idx = prs_name(data, p, e, fmt_name_base[2], 7u);
                    } else {
                        idx = prs_name(data, p, e, fmt_name_base[4], count);
                        if (idx >= 0) pm = idx;
                    }
                    if (idx < 0) { ok = false; break; }
                } else if (op.x == 3u) {
                    uint val = 0u, used = 0u;
                    while (p < e && used < 6u && data[p] >= 0x30u && data[p] <= 0x39u) {
                        val = val * 10u + (uint)(data[p] - 0x30u); p++; used++;
                    }
                    if (used == 0u) { ok = false; break; }
                    for (uint q = used; q < 6u; q++) val *= 10u;
                    micros = (long)val;
                } else {                                                  // %z
                    if (p >= e) { ok = false; break; }
                    uchar s = data[p];
                    if (s == 0x5Au) { p++; zoff = 0L; }                   // "Z"
                    else if (s == 0x2Bu || s == 0x2Du) {
                        p++;
                        uint hh = 0u, mm = 0u;
                        if (!prs_num(data, p, e, 2u, 0u, 23u, hh)) { ok = false; break; }
                        if (p < e && data[p] == 0x3Au) p++;
                        if (!prs_num(data, p, e, 2u, 0u, 59u, mm)) { ok = false; break; }
                        zoff = (long)(hh * 3600u + mm * 60u) * (s == 0x2Du ? -1L : 1L);
                    } else { ok = false; break; }
                }
            }
            if (!ok || p != e) continue;                                  // the whole value must be consumed
            if (hour12 >= 0) {
                long h = (long)(hour12 % 12);
                hour = (pm == 1) ? h + 12L : h;
            } else if (pm == 1 && hour < 12L) {
                hour += 12L;
            } else if (pm == 0 && hour == 12L) {
                hour = 0L;
            }
            long days = fmt_days_from_civil(year, month, day);
            long seconds = days * 86400L + hour * 3600L + minute * 60L + second - zoff;
            long ticks = seconds * P.scale;
            if (P.subScale > 0L) ticks += micros * P.subScale;
            else if (P.subDivide > 0L) ticks += micros / P.subDivide;
            outVals[i] = ticks;
            bits |= (1u << j);
        }
        outValid[w] = bits;
    }
    """
}
