import Foundation

/// MSL for the CSV reader (`Sources/ArrowMetal/CSV/`).
///
/// **Structure scan.** RFC 4180 plus Arrow's reading of it is a five-state machine over byte classes
/// (other, delimiter, quote, newline):
///
/// | state | other | delimiter | quote | `\n` / `\r` |
/// |---|---|---|---|---|
/// | 0 line start | 2 | 1, *field end* | 3 | 0 (empty line, skipped) |
/// | 1 field start | 2 | 1, *field end* | 3 | 0, *record end* |
/// | 2 unquoted field | 2 | 1, *field end* | 2 (a literal quote) | 0, *record end* |
/// | 3 inside quotes | 3 | 3 | 4 | 3 |
/// | 4 quote inside quotes | 2 | 1, *field end* | 3 (doubled quote) or 2 | 0, *record end* |
///
/// A quote only opens a quoted section at the start of a field; anywhere else it is an ordinary byte,
/// and after a closing quote the rest of the field is literal (`"ab"cd` is `abcd`, `ab"c` is `ab"c`).
/// A plain quote-parity scan cannot express that, so the scan is the exact generalisation of it: each
/// thread runs its block of bytes from **every** start state at once (`csv_summarize`: the end state
/// and the number of boundaries for each of the five), the blocks' transfer functions are composed by
/// a prefix scan (`csv_scan_local`, then `csv_scan_groups` over the threadgroup totals), and each
/// thread then re-runs its block from its now-known start state writing the boundary positions
/// (`csv_emit`). The table is data (`trans` / `emit` in `CsvScan`), so the delimiter, the quote
/// character, quoting off and `double_quote` are all the same kernels.
///
/// A boundary is one `uint`: the byte position with bit 31 set for a record end. Field `k` spans from
/// just past boundary `k - 1` (past any run of newlines when that one ended a record) to boundary `k`.
///
/// **Columns.** `csv_spans` turns boundary pairs into each row's *content* span: the bytes between the
/// quotes for a quoted field, the whole field otherwise. A field whose value differs from any single
/// span (a doubled quote, or bytes after the closing quote) is *complex*; those are unescaped into a
/// side buffer (`csv_side_len`, a scan, `csv_unescape`) and their span re-pointed there. Every later
/// kernel reads a span as (offset, length | quoted << 31 | side << 30).
///
/// `csv_classify` runs Arrow's type inference: each row reports which of the candidate types it
/// converts as, the column ANDs them together (`simd_and`, one atomic per simdgroup), and the host
/// picks the first survivor in Arrow's order. The `csv_conv_*` kernels then write the values, one
/// thread per row, with validity packed 32 rows at a time by `simd_ballot`.
enum CSVSource {

    static let source = KernelSource.prelude + CSVFloatSource.functions + """

    // ------------------------------------------------------------------ structure scan

    struct CsvScan {
        uint dataStart;
        uint dataEnd;
        uint blockBytes;
        uint nBlocks;
        uint delim;
        uint quote;          // 256 when quoting is off
        uint base;           // dataStart rounded down to 16: block b starts at base + b * blockBytes
        uint pad1;
        uint trans[4];       // per byte class: next state of each of the 5 states, 3 bits apiece
        uint emit[4];        // per byte class: 5-bit mask of the states that emit a boundary on it
    };

    struct BlockSum { uint ends; uint cnt[5]; };

    #define CSV_IDENTITY (0u | (1u << 3) | (2u << 6) | (3u << 9) | (4u << 12))

    inline uint csv_class(uchar b, constant CsvScan& P) {
        if ((uint)b == P.delim) return 1u;
        if ((uint)b == P.quote) return 2u;
        if (b == (uchar)0x0Au || b == (uchar)0x0Du) return 3u;
        return 0u;
    }

    // Visits the bytes [from, to) of `data` in order as `ch`, sixteen per load where the position is
    // 16-byte aligned (block starts are, except the first block's, when blockBytes is a multiple of 16).
    #define CSV_FOR_BYTES(from, to, ...) \\
        for (uint pos = (from); pos < (to); ) { \\
            if ((pos & 15u) == 0u && pos + 16u <= (to)) { \\
                uint4 v16 = *(device const uint4*)(data + pos); \\
                for (uint q = 0u; q < 16u; q++, pos++) { \\
                    uint ch = (v16[q >> 2] >> ((q & 3u) * 8u)) & 0xFFu; \\
                    __VA_ARGS__ \\
                } \\
            } else { \\
                uint ch = data[pos]; \\
                __VA_ARGS__ \\
                pos++; \\
            } \\
        }

    inline uint2 csv_block(uint b, constant CsvScan& P) {
        uint lo = max(P.dataStart, P.base + b * P.blockBytes);
        uint hi = min(P.base + (b + 1u) * P.blockBytes, P.dataEnd);
        return uint2(lo, max(lo, hi));
    }

    kernel void csv_summarize(device const uchar* data [[buffer(0)]],
                              constant CsvScan& P [[buffer(1)]],
                              device BlockSum* out [[buffer(2)]],
                              uint b [[thread_position_in_grid]]) {
        if (b >= P.nBlocks) return;
        uint2 span = csv_block(b, P);
        uint lo = span.x, hi = span.y;
        uint s0 = 0u, s1 = 1u, s2 = 2u, s3 = 3u, s4 = 4u;
        uint c0 = 0u, c1 = 0u, c2 = 0u, c3 = 0u, c4 = 0u;
        // All five runs until they have collapsed onto at most two states (in practice: the first
        // newline outside quotes sends every run but "inside quotes" to line start) ...
        uint i = lo;
        bool merged = false;
        uint a = 0u, bb = 0u;
        while (i < hi) {
            uint stop = min((i + 16u) & ~15u, hi);
            CSV_FOR_BYTES(i, stop, {
                uint k = csv_class((uchar)ch, P);
                uint t = P.trans[k], e = P.emit[k];
                c0 += (e >> s0) & 1u; s0 = (t >> (3u * s0)) & 7u;
                c1 += (e >> s1) & 1u; s1 = (t >> (3u * s1)) & 7u;
                c2 += (e >> s2) & 1u; s2 = (t >> (3u * s2)) & 7u;
                c3 += (e >> s3) & 1u; s3 = (t >> (3u * s3)) & 7u;
                c4 += (e >> s4) & 1u; s4 = (t >> (3u * s4)) & 7u;
            })
            i = stop;
            a = s0; bb = s0;
            bool two = true;
            uint ss[4] = {s1, s2, s3, s4};
            for (uint j = 0u; j < 4u; j++) {
                if (ss[j] != a) { if (bb == a) bb = ss[j]; else if (ss[j] != bb) two = false; }
            }
            if (two) { merged = true; break; }
        }
        // ... then only the (at most) two distinct runs, each standing for the runs that share its state.
        if (merged && i < hi) {
            uint sa = a, sb = bb, ca = 0u, cb = 0u;
            CSV_FOR_BYTES(i, hi, {
                uint k = csv_class((uchar)ch, P);
                uint t = P.trans[k], e = P.emit[k];
                ca += (e >> sa) & 1u; sa = (t >> (3u * sa)) & 7u;
                cb += (e >> sb) & 1u; sb = (t >> (3u * sb)) & 7u;
            })
            if (s0 == a) { c0 += ca; s0 = sa; } else { c0 += cb; s0 = sb; }
            if (s1 == a) { c1 += ca; s1 = sa; } else { c1 += cb; s1 = sb; }
            if (s2 == a) { c2 += ca; s2 = sa; } else { c2 += cb; s2 = sb; }
            if (s3 == a) { c3 += ca; s3 = sa; } else { c3 += cb; s3 = sb; }
            if (s4 == a) { c4 += ca; s4 = sa; } else { c4 += cb; s4 = sb; }
        }
        BlockSum r;
        r.ends = s0 | (s1 << 3) | (s2 << 6) | (s3 << 9) | (s4 << 12);
        r.cnt[0] = c0; r.cnt[1] = c1; r.cnt[2] = c2; r.cnt[3] = c3; r.cnt[4] = c4;
        out[b] = r;
    }

    // `a` then `b`: state i goes to b(a(i)), and counts a.cnt[i] + b.cnt[a(i)].
    inline void csv_compose(uint am, thread const uint* ac, uint bm, thread const uint* bc,
                            thread uint& om, thread uint* oc) {
        uint m = 0u;
        for (uint i = 0u; i < 5u; i++) {
            uint mid = (am >> (3u * i)) & 7u;
            m |= ((bm >> (3u * mid)) & 7u) << (3u * i);
            oc[i] = ac[i] + bc[mid];
        }
        om = m;
    }

    // Inclusive Hillis-Steele scan of 256 transfer functions in threadgroup memory.
    inline void csv_tg_scan(thread uint& m, thread uint* c, uint lid,
                            threadgroup uint* tm, threadgroup uint* tc) {
        tm[lid] = m;
        for (uint k = 0u; k < 5u; k++) tc[k * 256u + lid] = c[k];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = 1u; off < 256u; off <<= 1u) {
            uint pm = 0u; uint pc[5] = {0u, 0u, 0u, 0u, 0u};
            bool has = lid >= off;
            if (has) {
                pm = tm[lid - off];
                for (uint k = 0u; k < 5u; k++) pc[k] = tc[k * 256u + lid - off];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (has) {
                uint nm; uint nc[5];
                csv_compose(pm, pc, m, c, nm, nc);
                m = nm;
                for (uint k = 0u; k < 5u; k++) c[k] = nc[k];
                tm[lid] = m;
                for (uint k = 0u; k < 5u; k++) tc[k * 256u + lid] = c[k];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    kernel void csv_scan_local(device const BlockSum* sums [[buffer(0)]],
                               constant uint& nBlocks [[buffer(1)]],
                               device BlockSum* incl [[buffer(2)]],
                               device BlockSum* groupTotals [[buffer(3)]],
                               uint gid [[thread_position_in_grid]],
                               uint lid [[thread_index_in_threadgroup]],
                               uint tg [[threadgroup_position_in_grid]]) {
        threadgroup uint tm[256];
        threadgroup uint tc[5 * 256];
        uint m = CSV_IDENTITY; uint c[5] = {0u, 0u, 0u, 0u, 0u};
        if (gid < nBlocks) {
            BlockSum s = sums[gid];
            m = s.ends;
            for (uint k = 0u; k < 5u; k++) c[k] = s.cnt[k];
        }
        csv_tg_scan(m, c, lid, tm, tc);
        BlockSum r; r.ends = m;
        for (uint k = 0u; k < 5u; k++) r.cnt[k] = c[k];
        if (gid < nBlocks) incl[gid] = r;
        if (lid == 255u) groupTotals[tg] = r;
    }

    // One threadgroup: each thread folds a run of group totals, the runs are scanned, and each thread
    // walks its run again from the true start (state 0 at offset 0) writing every group's start.
    kernel void csv_scan_groups(device const BlockSum* totals [[buffer(0)]],
                                constant uint& nGroups [[buffer(1)]],
                                device uint2* groupStart [[buffer(2)]],
                                uint lid [[thread_index_in_threadgroup]]) {
        threadgroup uint tm[256];
        threadgroup uint tc[5 * 256];
        uint per = (nGroups + 255u) / 256u;
        uint lo = min(lid * per, nGroups), hi = min(lo + per, nGroups);
        uint m = CSV_IDENTITY; uint c[5] = {0u, 0u, 0u, 0u, 0u};
        for (uint g = lo; g < hi; g++) {
            BlockSum t = totals[g];
            uint nm; uint nc[5];
            csv_compose(m, c, t.ends, t.cnt, nm, nc);
            m = nm;
            for (uint k = 0u; k < 5u; k++) c[k] = nc[k];
        }
        csv_tg_scan(m, c, lid, tm, tc);
        uint st = 0u, off = 0u;
        if (lid > 0u) { st = tm[lid - 1u] & 7u; off = tc[lid - 1u]; }
        for (uint g = lo; g < hi; g++) {
            groupStart[g] = uint2(st, off);
            BlockSum t = totals[g];
            off += t.cnt[st];
            st = (t.ends >> (3u * st)) & 7u;
        }
        if (lo < hi && hi == nGroups) groupStart[nGroups] = uint2(st, off);
    }

    kernel void csv_emit(device const uchar* data [[buffer(0)]],
                         constant CsvScan& P [[buffer(1)]],
                         device const BlockSum* incl [[buffer(2)]],
                         device const uint2* groupStart [[buffer(3)]],
                         device uint* events [[buffer(4)]],
                         uint b [[thread_position_in_grid]]) {
        if (b >= P.nBlocks) return;
        uint2 gs = groupStart[b >> 8];
        uint st = gs.x, off = gs.y;
        if ((b & 255u) != 0u) {
            BlockSum p = incl[b - 1u];
            off += p.cnt[st];
            st = (p.ends >> (3u * st)) & 7u;
        }
        uint2 span = csv_block(b, P);
        CSV_FOR_BYTES(span.x, span.y, {
            uint k = csv_class((uchar)ch, P);
            if ((P.emit[k] >> st) & 1u) events[off++] = pos | (k == 1u ? 0u : 0x80000000u);
            st = (P.trans[k] >> (3u * st)) & 7u;
        })
    }

    // ------------------------------------------------------------------ columns

    struct CsvCol {
        uint nRows;
        uint nCols;
        uint col;
        uint firstRecord;
        uint dataStart;
        uint dataEnd;
        uint quote;          // 256 when quoting is off
        uint doubleQuote;
        uint flags;          // bit 0 quoted_strings_can_be_null, bit 1 strings_can_be_null, bit 2 check_utf8
        uint decimalPoint;
        uint nNull;
        uint nTrue;
        uint nFalse;
        uint unitDigits;     // 0 s, 3 ms, 6 us, 9 ns
        uint expectZone;     // timestamps: 1 when the target type carries a timezone
        uint isSigned;
        uint maxHex;         // integers: at most this many hex digits after 0x
        uint hasSpans;       // 1 when the column's spans were materialised (it has complex fields)
        ulong limPos;
        ulong limNeg;
    };

    #define CSV_QSCBN 1u
    #define CSV_SCBN 2u
    #define CSV_UTF8 4u
    #define CSV_SIDE 0x40000000u
    #define CSV_QUOTED 0x80000000u
    #define CSV_LEN 0x3FFFFFFFu

    inline uint csv_skip_nl(device const uchar* d, uint p, uint end) {
        while (p < end && (d[p] == (uchar)0x0Au || d[p] == (uchar)0x0Du)) p++;
        return p;
    }

    // Raw bytes of field k: [start, end).
    inline uint2 csv_raw(device const uchar* d, device const uint* ev, uint k, constant CsvCol& P) {
        uint e = ev[k] & 0x7FFFFFFFu;
        uint s;
        if (k == 0u) s = csv_skip_nl(d, P.dataStart, P.dataEnd);
        else {
            uint pv = ev[k - 1u];
            uint pp = pv & 0x7FFFFFFFu;
            s = (pv >> 31) != 0u ? csv_skip_nl(d, pp + 1u, P.dataEnd) : pp + 1u;
        }
        return uint2(s, max(s, e));
    }

    // Walks a quoted field. Returns the unescaped length; `simple` is true when the value is one span of
    // the input (then [cs, cs + len) is it).
    inline uint csv_quoted(device const uchar* d, uint s, uint e, constant CsvCol& P, thread bool& simple, thread uint& cs) {
        uint q = P.quote;
        bool inQ = true, closed = false, cx = false;
        uint len = 0u, closeAt = e;
        for (uint i = s + 1u; i < e; i++) {
            uint c = d[i];
            if (inQ) {
                if (c == q) {
                    if (P.doubleQuote != 0u && i + 1u < e && (uint)d[i + 1u] == q) { len++; i++; cx = true; }
                    else { inQ = false; closed = true; closeAt = i; }
                } else len++;
            } else { len++; cx = true; }
        }
        simple = !cx;
        cs = s + 1u;
        if (simple) len = (closed ? closeAt : e) - (s + 1u);
        return len;
    }

    // One pass over every boundary. Every record has nCols fields exactly when the record ends sit at
    // k = nCols - 1, 2 nCols - 1, ...; the first boundary that breaks the pattern is reported. The same
    // pass counts each column's complex fields (a doubled quote, or text after the closing quote): a
    // column with none, nearly all of them, never materialises spans, its kernels derive each row's
    // span from the boundaries.
    kernel void csv_check_fields(device const uchar* d [[buffer(0)]],
                                 device const uint* ev [[buffer(1)]],
                                 constant CsvCol& P [[buffer(2)]],
                                 constant uint& nEvents [[buffer(3)]],
                                 device atomic_uint* firstBad [[buffer(4)]],
                                 device atomic_uint* complexCounts [[buffer(5)]],
                                 uint k [[thread_position_in_grid]]) {
        if (k >= nEvents) return;
        bool isEnd = (ev[k] >> 31) != 0u;
        bool expect = ((k + 1u) % P.nCols) == 0u;
        if (isEnd != expect) atomic_fetch_min_explicit(firstBad, k, memory_order_relaxed);
        uint2 raw = csv_raw(d, ev, k, P);
        // A span keeps its length in 30 bits; slot nCols counts the fields that would not fit.
        if (raw.y - raw.x > CSV_LEN) atomic_fetch_add_explicit(complexCounts + P.nCols, 1u, memory_order_relaxed);
        if (P.quote > 255u) return;
        if (raw.x < raw.y && (uint)d[raw.x] == P.quote) {
            bool simple; uint cs;
            csv_quoted(d, raw.x, raw.y, P, simple, cs);
            if (!simple) atomic_fetch_add_explicit(complexCounts + (k % P.nCols), 1u, memory_order_relaxed);
        }
    }

    // Materialised spans, for a column with complex fields; those are marked for `csv_unescape`.
    kernel void csv_spans(device const uchar* d [[buffer(0)]],
                          device const uint* ev [[buffer(1)]],
                          constant CsvCol& P [[buffer(2)]],
                          device uint2* spans [[buffer(3)]],
                          uint r [[thread_position_in_grid]]) {
        if (r >= P.nRows) return;
        uint k = (P.firstRecord + r) * P.nCols + P.col;
        uint2 raw = csv_raw(d, ev, k, P);
        if (P.quote > 255u || raw.x >= raw.y || (uint)d[raw.x] != P.quote) {
            spans[r] = uint2(raw.x, raw.y - raw.x);
            return;
        }
        bool simple; uint cs;
        uint len = csv_quoted(d, raw.x, raw.y, P, simple, cs);
        if (simple) { spans[r] = uint2(cs, len | CSV_QUOTED); return; }
        spans[r] = uint2(raw.x, len | CSV_QUOTED | CSV_SIDE);                 // unescaped later
    }

    kernel void csv_side_len(device const uint2* spans [[buffer(0)]],
                             constant CsvCol& P [[buffer(1)]],
                             device int* lens [[buffer(2)]],
                             uint r [[thread_position_in_grid]]) {
        if (r >= P.nRows) return;
        uint2 sp = spans[r];
        lens[r] = (sp.y & CSV_SIDE) != 0u ? (int)(sp.y & CSV_LEN) : 0;
    }

    kernel void csv_unescape(device const uchar* d [[buffer(0)]],
                             device const uint* ev [[buffer(1)]],
                             constant CsvCol& P [[buffer(2)]],
                             device uint2* spans [[buffer(3)]],
                             device const int* sideOffsets [[buffer(4)]],
                             device uchar* side [[buffer(5)]],
                             uint r [[thread_position_in_grid]]) {
        if (r >= P.nRows) return;
        uint2 sp = spans[r];
        if ((sp.y & CSV_SIDE) == 0u) return;
        uint k = (P.firstRecord + r) * P.nCols + P.col;
        uint2 raw = csv_raw(d, ev, k, P);
        uint q = P.quote;
        uint o = (uint)sideOffsets[r], w = o;
        bool inQ = true;
        for (uint i = raw.x + 1u; i < raw.y; i++) {
            uint c = d[i];
            if (inQ && c == q) {
                if (P.doubleQuote != 0u && i + 1u < raw.y && (uint)d[i + 1u] == q) { side[w++] = (uchar)q; i++; }
                else inQ = false;
            } else side[w++] = (uchar)c;
        }
        spans[r] = uint2(o, (sp.y & CSV_LEN) | CSV_QUOTED | CSV_SIDE);
    }

    // ------------------------------------------------------------------ value grammars

    inline device const uchar* csv_ptr(uint2 sp, device const uchar* d, device const uchar* side) {
        return ((sp.y & CSV_SIDE) != 0u ? side : d) + sp.x;
    }

    // Row r's content span: the materialised one, or derived from the boundaries (a column without
    // complex fields, where every quoted field is one span of the file).
    inline uint2 csv_span(uint r, device const uint2* spans, device const uint* ev, device const uchar* d,
                          constant CsvCol& P) {
        if (P.hasSpans != 0u) return spans[r];
        uint k = (P.firstRecord + r) * P.nCols + P.col;
        uint2 raw = csv_raw(d, ev, k, P);
        if (P.quote > 255u || raw.x >= raw.y || (uint)d[raw.x] != P.quote) return uint2(raw.x, raw.y - raw.x);
        bool simple; uint cs;
        uint len = csv_quoted(d, raw.x, raw.y, P, simple, cs);
        return uint2(cs, len | CSV_QUOTED);
    }

    inline bool csv_match(device const uchar* p, uint len, uint first, uint count,
                          device const uint* lists, device const uchar* lbytes) {
        for (uint j = first; j < first + count; j++) {
            uint off = lists[2u * j], l = lists[2u * j + 1u];
            if (l != len) continue;
            bool eq = true;
            for (uint t = 0u; t < l; t++) { if (p[t] != lbytes[off + t]) { eq = false; break; } }
            if (eq) return true;
        }
        return false;
    }

    inline bool csv_is_null(device const uchar* p, uint len, bool quoted, constant CsvCol& P,
                            device const uint* lists, device const uchar* lbytes) {
        if (quoted && (P.flags & CSV_QSCBN) == 0u) return false;
        return csv_match(p, len, 0u, P.nNull, lists, lbytes);
    }

    // Arrow trims spaces and tabs (only) around numbers, dates and times; never around timestamps,
    // booleans or null values.
    #define CSV_TRIM(p, len) \
        while (len > 0u && (p[0] == (uchar)0x20u || p[0] == (uchar)0x09u)) { p++; len--; } \
        while (len > 0u && (p[len - 1u] == (uchar)0x20u || p[len - 1u] == (uchar)0x09u)) len--;

    // Arrow's integer grammar after trimming: -?[0-9]+ in range, or 0x / 0X and 1..maxHex hex digits
    // taken as the bit pattern of the target width. No '+'.
    inline bool csv_int(device const uchar* p, uint len, bool isSigned, uint maxHex, ulong limPos, ulong limNeg,
                        thread ulong& out) {
        CSV_TRIM(p, len)
        if (len == 0u) return false;
        if (len > 2u && p[0] == (uchar)0x30u && (p[1] | 0x20u) == 0x78u) {
            uint nd = len - 2u;
            if (nd > maxHex) return false;
            ulong v = 0UL;
            for (uint i = 2u; i < len; i++) {
                uint c = p[i], h;
                if (c >= 0x30u && c <= 0x39u) h = c - 0x30u;
                else if ((c | 0x20u) >= 0x61u && (c | 0x20u) <= 0x66u) h = (c | 0x20u) - 0x61u + 10u;
                else return false;
                v = (v << 4) | (ulong)h;
            }
            out = v;
            return true;
        }
        uint i = 0u;
        bool neg = false;
        if (p[0] == (uchar)0x2Du) { if (!isSigned) return false; neg = true; i = 1u; }
        if (i >= len) return false;
        ulong lim = neg ? limNeg : limPos;
        // No 64-bit division (slow on the GPU): up to 19 significant digits cannot overflow a ulong
        // (10^19 - 1 < 2^64), so they accumulate freely and only the final value is compared with
        // the limit; a 20th significant digit is checked by hand.
        ulong acc = 0UL;
        uint sig = 0u;
        for (; i < len; i++) {
            uint dd = (uint)p[i] - 48u;
            if (dd > 9u) return false;
            if (sig < 19u) {
                acc = acc * 10UL + (ulong)dd;
                if (acc != 0UL) sig++;
            } else {
                if (sig > 19u || acc > 1844674407370955161UL) return false;
                ulong t = acc * 10UL;
                ulong nv = t + (ulong)dd;
                if (nv < t) return false;
                acc = nv;
                sig++;
            }
        }
        if (acc > lim) return false;
        out = neg ? (0UL - acc) : acc;
        return true;
    }

    inline bool csv_digits(device const uchar* p, uint n, thread uint& v) {
        uint a = 0u;
        for (uint i = 0u; i < n; i++) { uint dd = (uint)p[i] - 48u; if (dd > 9u) return false; a = a * 10u + dd; }
        v = a;
        return true;
    }

    inline int csv_days_from_civil(int y, uint m, uint d) {
        y -= (m <= 2u) ? 1 : 0;
        int era = (y >= 0 ? y : y - 399) / 400;
        uint yoe = (uint)(y - era * 400);
        uint doy = (153u * (m > 2u ? m - 3u : m + 9u) + 2u) / 5u + d - 1u;
        uint doe = yoe * 365u + yoe / 4u - yoe / 100u + doy;
        return era * 146097 + (int)doe - 719468;
    }

    // YYYY-MM-DD at p (10 bytes), validated as a calendar date.
    inline bool csv_ymd(device const uchar* p, thread int& days) {
        if (p[4] != (uchar)0x2Du || p[7] != (uchar)0x2Du) return false;
        uint y, m, d;
        if (!csv_digits(p, 4u, y) || !csv_digits(p + 5, 2u, m) || !csv_digits(p + 8, 2u, d)) return false;
        if (m < 1u || m > 12u || d < 1u) return false;
        uint dim = 31u;
        if (m == 4u || m == 6u || m == 9u || m == 11u) dim = 30u;
        else if (m == 2u) dim = ((y % 4u == 0u && y % 100u != 0u) || y % 400u == 0u) ? 29u : 28u;
        if (d > dim) return false;
        days = csv_days_from_civil((int)y, m, d);
        return true;
    }

    inline bool csv_hh(device const uchar* p, thread int& secs) {
        uint h; if (!csv_digits(p, 2u, h) || h >= 24u) return false;
        secs = (int)h * 3600; return true;
    }
    inline bool csv_hh_mm(device const uchar* p, thread int& secs) {
        if (p[2] != (uchar)0x3Au) return false;
        uint h, m; if (!csv_digits(p, 2u, h) || !csv_digits(p + 3, 2u, m) || h >= 24u || m >= 60u) return false;
        secs = (int)(h * 3600u + m * 60u); return true;
    }
    inline bool csv_hhmm(device const uchar* p, thread int& secs) {
        uint h, m; if (!csv_digits(p, 2u, h) || !csv_digits(p + 2, 2u, m) || h >= 24u || m >= 60u) return false;
        secs = (int)(h * 3600u + m * 60u); return true;
    }
    inline bool csv_hh_mm_ss(device const uchar* p, thread int& secs) {
        if (p[2] != (uchar)0x3Au || p[5] != (uchar)0x3Au) return false;
        uint h, m, s;
        if (!csv_digits(p, 2u, h) || !csv_digits(p + 3, 2u, m) || !csv_digits(p + 6, 2u, s)) return false;
        if (h >= 24u || m >= 60u || s >= 60u) return false;
        secs = (int)(h * 3600u + m * 60u + s); return true;
    }

    inline long csv_pow10(uint n) { long r = 1L; for (uint i = 0u; i < n; i++) r *= 10L; return r; }

    // Arrow's ParseSubSeconds: at most `unitDigits` digits (0 means the unit takes no fraction), scaled
    // to the unit.
    inline bool csv_subsec(device const uchar* p, uint n, uint unitDigits, thread long& ticks) {
        if (unitDigits == 0u || n > unitDigits) return false;
        uint v; if (!csv_digits(p, n, v)) return false;
        ticks = (long)v * csv_pow10(unitDigits - n);
        return true;
    }

    // Arrow's ParseTimestampISO8601, value in ticks of 10^-unitDigits s. `zone` reports an offset or Z.
    // `frac` reports whether a fractional part was present.
    inline bool csv_iso(device const uchar* s, uint length, uint unitDigits, thread long& out,
                        thread bool& zone, thread bool& frac) {
        zone = false; frac = false;
        if (length < 10u) return false;
        int days;
        if (!csv_ymd(s, days)) return false;
        long scale = csv_pow10(unitDigits);
        long secs = (long)days * 86400L;
        if (length == 10u) { out = secs * scale; return true; }
        if (s[10] != (uchar)0x20u && s[10] != (uchar)0x54u) return false;
        int zoneOff = 0;
        if (s[length - 1u] == (uchar)0x5Au) { length -= 1u; zone = true; }
        else if (s[length - 3u] == (uchar)0x2Bu || s[length - 3u] == (uchar)0x2Du) {
            length -= 3u;
            if (!csv_hh(s + length + 1u, zoneOff)) return false;
            if (s[length] == (uchar)0x2Bu) zoneOff = -zoneOff;
            zone = true;
        } else if (s[length - 5u] == (uchar)0x2Bu || s[length - 5u] == (uchar)0x2Du) {
            length -= 5u;
            if (!csv_hhmm(s + length + 1u, zoneOff)) return false;
            if (s[length] == (uchar)0x2Bu) zoneOff = -zoneOff;
            zone = true;
        } else if ((s[length - 6u] == (uchar)0x2Bu || s[length - 6u] == (uchar)0x2Du) && s[length - 3u] == (uchar)0x3Au) {
            length -= 6u;
            if (!csv_hh_mm(s + length + 1u, zoneOff)) return false;
            if (s[length] == (uchar)0x2Bu) zoneOff = -zoneOff;
            zone = true;
        }
        int sm = 0;
        if (length == 13u) { if (!csv_hh(s + 11, sm)) return false; }
        else if (length == 16u) { if (!csv_hh_mm(s + 11, sm)) return false; }
        else if (length == 19u || (length >= 21u && length <= 29u)) { if (!csv_hh_mm_ss(s + 11, sm)) return false; }
        else return false;
        secs += (long)sm + (long)zoneOff;
        if (length <= 19u) { out = secs * scale; return true; }
        if (s[19] != (uchar)0x2Eu) return false;
        long sub;
        if (!csv_subsec(s + 20, length - 20u, unitDigits, sub)) return false;
        frac = true;
        out = secs * scale + sub;
        return true;
    }

    // Arrow's time32 / time64 parse (after trimming): hh:mm, hh:mm:ss, hh:mm:ss.fff...
    inline bool csv_time(device const uchar* p, uint len, uint unitDigits, thread long& out) {
        CSV_TRIM(p, len)
        if (len < 5u) return false;
        int sm;
        if (len == 5u) { if (!csv_hh_mm(p, sm)) return false; }
        else if (len >= 8u) { if (!csv_hh_mm_ss(p, sm)) return false; }
        else return false;
        long scale = csv_pow10(unitDigits);
        out = (long)sm * scale;
        if (len == 5u || len == 8u) return true;
        if (p[8] != (uchar)0x2Eu) return false;
        long sub = 0L;
        if (len - 9u > 0u) { if (!csv_subsec(p + 9, len - 9u, unitDigits, sub)) return false; }
        else if (unitDigits == 0u) return false;
        out += sub;
        return true;
    }

    inline bool csv_date(device const uchar* p, uint len, thread int& days) {
        CSV_TRIM(p, len)
        if (len != 10u) return false;
        return csv_ymd(p, days);
    }

    // Strict UTF-8 (no overlongs, no surrogates, nothing above U+10FFFF).
    inline bool csv_utf8(device const uchar* p, uint len) {
        uint i = 0u;
        while (i < len) {
            uint c = p[i];
            if (c < 0x80u) { i++; continue; }
            uint n; uint lo = 0x80u, hi = 0xBFu;
            if (c >= 0xC2u && c <= 0xDFu) n = 1u;
            else if (c >= 0xE0u && c <= 0xEFu) { n = 2u; if (c == 0xE0u) lo = 0xA0u; if (c == 0xEDu) hi = 0x9Fu; }
            else if (c >= 0xF0u && c <= 0xF4u) { n = 3u; if (c == 0xF0u) lo = 0x90u; if (c == 0xF4u) hi = 0x8Fu; }
            else return false;
            if (i + n >= len) return false;
            uint c1 = p[i + 1u];
            if (c1 < lo || c1 > hi) return false;
            for (uint k = 2u; k <= n; k++) { uint ck = p[i + k]; if (ck < 0x80u || ck > 0xBFu) return false; }
            i += n + 1u;
        }
        return true;
    }

    // ------------------------------------------------------------------ inference

    #define K_NULL 1u
    #define K_INT 2u
    #define K_BOOL 4u
    #define K_DATE 8u
    #define K_TIME 16u
    #define K_TS 32u
    #define K_TSNS 64u
    #define K_TSZ 128u
    #define K_TSZNS 256u
    #define K_REAL 512u
    #define K_TEXT 1024u
    #define K_BIN 2048u
    #define K_TYPED (K_INT | K_BOOL | K_DATE | K_TIME | K_TS | K_TSNS | K_TSZ | K_TSZNS | K_REAL)

    // The first kind (in Arrow's order) a non-null value converts as; 0xFFFFFFFF for a null value.
    inline uint csv_first_kind(device const uchar* p, uint len, constant CsvCol& P,
                               device const uint* lists, device const uchar* lbytes) {
        ulong iv;
        if (csv_int(p, len, true, 16u, 0x7FFFFFFFFFFFFFFFUL, 0x8000000000000000UL, iv)) return 1u;
        if (csv_match(p, len, P.nNull, P.nTrue, lists, lbytes) || csv_match(p, len, P.nNull + P.nTrue, P.nFalse, lists, lbytes)) return 2u;
        int days;
        if (csv_date(p, len, days)) return 3u;
        long tv;
        if (csv_time(p, len, 0u, tv)) return 4u;
        bool zone, frac;
        if (csv_iso(p, len, 9u, tv, zone, frac)) return zone ? (frac ? 8u : 7u) : (frac ? 6u : 5u);
        ulong bits;
        if (fp_parse(p, len, FP_CSV, (uchar)P.decimalPoint, false, bits) != FP_INVALID) return 9u;
        if ((P.flags & CSV_UTF8) == 0u || csv_utf8(p, len)) return 10u;
        return 11u;
    }

    // Speculative inference: the smallest and largest first-kind over the non-null rows. When they are
    // equal, that kind is the column's type (every row converts as it, and no row converts as any
    // earlier kind); otherwise the host runs the full `csv_classify`.
    kernel void csv_kind_range(device const uchar* d [[buffer(0)]],
                               device const uchar* side [[buffer(1)]],
                               device const uint2* spans [[buffer(2)]],
                               constant CsvCol& P [[buffer(3)]],
                               device const uint* lists [[buffer(4)]],
                               device const uchar* lbytes [[buffer(5)]],
                               device atomic_uint* range [[buffer(6)]],
                               device const uint* ev [[buffer(7)]],
                               uint r [[thread_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
        uint lo = 0xFFFFFFFFu, hi = 0u;
        if (r < P.nRows) {
            uint2 sp = csv_span(r, spans, ev, d, P);
            device const uchar* p = csv_ptr(sp, d, side);
            uint len = sp.y & CSV_LEN;
            bool quoted = (sp.y & CSV_QUOTED) != 0u;
            if (!csv_is_null(p, len, quoted, P, lists, lbytes)) {
                uint k = csv_first_kind(p, len, P, lists, lbytes);
                lo = k; hi = k;
            }
        }
        lo = simd_min(lo); hi = simd_max(hi);
        if (lane == 0u && lo != 0xFFFFFFFFu) {
            atomic_fetch_min_explicit(range, lo, memory_order_relaxed);
            atomic_fetch_max_explicit(range + 1, hi, memory_order_relaxed);
        }
    }

    kernel void csv_classify(device const uchar* d [[buffer(0)]],
                             device const uchar* side [[buffer(1)]],
                             device const uint2* spans [[buffer(2)]],
                             constant CsvCol& P [[buffer(3)]],
                             device const uint* lists [[buffer(4)]],
                             device const uchar* lbytes [[buffer(5)]],
                             device atomic_uint* colMask [[buffer(6)]],
                             device const uint* ev [[buffer(7)]],
                             uint r [[thread_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
        uint ok = 0xFFFFFFFFu;
        if (r < P.nRows) {
            uint alive = atomic_load_explicit(colMask, memory_order_relaxed);
            uint2 sp = csv_span(r, spans, ev, d, P);
            device const uchar* p = csv_ptr(sp, d, side);
            uint len = sp.y & CSV_LEN;
            bool quoted = (sp.y & CSV_QUOTED) != 0u;
            ok = K_BIN;
            if (csv_is_null(p, len, quoted, P, lists, lbytes)) {
                ok |= K_NULL | K_TYPED;
                if ((P.flags & CSV_SCBN) != 0u) ok |= K_TEXT;
                else if ((alive & K_TEXT) != 0u && ((P.flags & CSV_UTF8) == 0u || csv_utf8(p, len))) ok |= K_TEXT;
            } else {
                ulong iv;
                if ((alive & K_INT) != 0u && csv_int(p, len, true, 16u, 0x7FFFFFFFFFFFFFFFUL, 0x8000000000000000UL, iv)) ok |= K_INT;
                if ((alive & K_BOOL) != 0u && (csv_match(p, len, P.nNull, P.nTrue, lists, lbytes)
                                              || csv_match(p, len, P.nNull + P.nTrue, P.nFalse, lists, lbytes))) ok |= K_BOOL;
                int days;
                if ((alive & K_DATE) != 0u && csv_date(p, len, days)) ok |= K_DATE;
                long tv;
                if ((alive & K_TIME) != 0u && csv_time(p, len, 0u, tv)) ok |= K_TIME;
                if ((alive & (K_TS | K_TSNS | K_TSZ | K_TSZNS)) != 0u) {
                    bool zone, frac;
                    if (csv_iso(p, len, 9u, tv, zone, frac)) {
                        if (!zone) ok |= K_TSNS | (frac ? 0u : K_TS);
                        else ok |= K_TSZNS | (frac ? 0u : K_TSZ);
                    }
                }
                if ((alive & K_REAL) != 0u) {
                    ulong bits;
                    if (fp_parse(p, len, FP_CSV, (uchar)P.decimalPoint, false, bits) != FP_INVALID) ok |= K_REAL;
                }
                if ((alive & K_TEXT) != 0u && ((P.flags & CSV_UTF8) == 0u || csv_utf8(p, len))) ok |= K_TEXT;
            }
        }
        uint m = simd_and(ok);
        if (lane == 0u && m != 0xFFFFFFFFu) atomic_fetch_and_explicit(colMask, m, memory_order_relaxed);
    }

    // ------------------------------------------------------------------ conversion

    #define CSV_ROW_PROLOGUE \\
        bool valid = false; bool bad = false; \\
        device const uchar* p = d; uint len = 0u; bool quoted = false; bool isNull = false; \\
        if (r < P.nRows) { \\
            uint2 sp = csv_span(r, spans, ev, d, P); \\
            p = csv_ptr(sp, d, side); len = sp.y & CSV_LEN; quoted = (sp.y & CSV_QUOTED) != 0u; \\
            isNull = csv_is_null(p, len, quoted, P, lists, lbytes); \\
        }

    #define CSV_ROW_EPILOGUE \\
        uint vw = (uint)(simd_vote::vote_t)simd_ballot(valid); \\
        if (lane == 0u && r < P.nRows) outValid[r >> 5] = vw; \\
        if (bad) atomic_fetch_min_explicit(err, r, memory_order_relaxed);

    #define CSV_CONV_ARGS \\
        device const uchar* d [[buffer(0)]], \\
        device const uchar* side [[buffer(1)]], \\
        device const uint2* spans [[buffer(2)]], \\
        constant CsvCol& P [[buffer(3)]], \\
        device const uint* lists [[buffer(4)]], \\
        device const uchar* lbytes [[buffer(5)]], \\
        device uint* outValid [[buffer(7)]], \\
        device atomic_uint* err [[buffer(8)]], \\
        device const uint* ev [[buffer(11)]], \\
        uint r [[thread_position_in_grid]], \\
        uint lane [[thread_index_in_simdgroup]]

    kernel void csv_conv_int(CSV_CONV_ARGS, device ulong* out [[buffer(6)]]) {
        CSV_ROW_PROLOGUE
        ulong v = 0UL;
        if (r < P.nRows && !isNull) {
            if (csv_int(p, len, P.isSigned != 0u, P.maxHex, P.limPos, P.limNeg, v)) valid = true; else bad = true;
        }
        if (r < P.nRows) out[r] = valid ? v : 0UL;
        CSV_ROW_EPILOGUE
    }

    // The integer kernel writes 64-bit lanes; this narrows them to the target width.
    kernel void csv_narrow(device const ulong* in [[buffer(0)]],
                           device uchar* out [[buffer(1)]],
                           constant uint& n [[buffer(2)]],
                           constant uint& width [[buffer(3)]],
                           uint r [[thread_position_in_grid]]) {
        if (r >= n) return;
        ulong v = in[r];
        if (width == 1u) out[r] = (uchar)v;
        else if (width == 2u) ((device ushort*)out)[r] = (ushort)v;
        else if (width == 4u) ((device uint*)out)[r] = (uint)v;
        else ((device ulong*)out)[r] = v;
    }

    kernel void csv_conv_f64(CSV_CONV_ARGS, device ulong* out [[buffer(6)]],
                             device uint* outHost [[buffer(9)]], device atomic_uint* hostCount [[buffer(10)]]) {
        CSV_ROW_PROLOGUE
        ulong bits = 0UL; bool host = false;
        if (r < P.nRows && !isNull) {
            uint st = fp_parse(p, len, FP_CSV, (uchar)P.decimalPoint, false, bits);
            if (st == FP_VALUE) valid = true; else if (st == FP_HOST) host = true; else bad = true;
        }
        if (r < P.nRows) out[r] = valid ? bits : 0UL;
        uint hw = (uint)(simd_vote::vote_t)simd_ballot(host);
        if (lane == 0u && r < P.nRows) {
            outHost[r >> 5] = hw;
            if (hw != 0u) atomic_fetch_add_explicit(hostCount, popcount(hw), memory_order_relaxed);
        }
        CSV_ROW_EPILOGUE
    }

    kernel void csv_conv_f32(CSV_CONV_ARGS, device uint* out [[buffer(6)]],
                             device uint* outHost [[buffer(9)]], device atomic_uint* hostCount [[buffer(10)]]) {
        CSV_ROW_PROLOGUE
        ulong bits = 0UL; bool host = false;
        if (r < P.nRows && !isNull) {
            uint st = fp_parse(p, len, FP_CSV, (uchar)P.decimalPoint, true, bits);
            if (st == FP_VALUE) valid = true; else if (st == FP_HOST) host = true; else bad = true;
        }
        if (r < P.nRows) out[r] = valid ? (uint)bits : 0u;
        uint hw = (uint)(simd_vote::vote_t)simd_ballot(host);
        if (lane == 0u && r < P.nRows) {
            outHost[r >> 5] = hw;
            if (hw != 0u) atomic_fetch_add_explicit(hostCount, popcount(hw), memory_order_relaxed);
        }
        CSV_ROW_EPILOGUE
    }

    kernel void csv_conv_bool(CSV_CONV_ARGS, device uint* out [[buffer(6)]]) {
        CSV_ROW_PROLOGUE
        bool v = false;
        if (r < P.nRows && !isNull) {
            if (csv_match(p, len, P.nNull, P.nTrue, lists, lbytes)) { v = true; valid = true; }
            else if (csv_match(p, len, P.nNull + P.nTrue, P.nFalse, lists, lbytes)) valid = true;
            else bad = true;
        }
        uint bw = (uint)(simd_vote::vote_t)simd_ballot(v);
        if (lane == 0u && r < P.nRows) out[r >> 5] = bw;
        CSV_ROW_EPILOGUE
    }

    kernel void csv_conv_date(CSV_CONV_ARGS, device int* out [[buffer(6)]]) {
        CSV_ROW_PROLOGUE
        int v = 0;
        if (r < P.nRows && !isNull) { if (csv_date(p, len, v)) valid = true; else bad = true; }
        if (r < P.nRows) out[r] = valid ? v : 0;
        CSV_ROW_EPILOGUE
    }

    kernel void csv_conv_time32(CSV_CONV_ARGS, device int* out [[buffer(6)]]) {
        CSV_ROW_PROLOGUE
        long v = 0L;
        if (r < P.nRows && !isNull) { if (csv_time(p, len, P.unitDigits, v)) valid = true; else bad = true; }
        if (r < P.nRows) out[r] = valid ? (int)v : 0;
        CSV_ROW_EPILOGUE
    }

    kernel void csv_conv_time64(CSV_CONV_ARGS, device long* out [[buffer(6)]]) {
        CSV_ROW_PROLOGUE
        long v = 0L;
        if (r < P.nRows && !isNull) { if (csv_time(p, len, P.unitDigits, v)) valid = true; else bad = true; }
        if (r < P.nRows) out[r] = valid ? v : 0L;
        CSV_ROW_EPILOGUE
    }

    kernel void csv_conv_ts(CSV_CONV_ARGS, device long* out [[buffer(6)]]) {
        CSV_ROW_PROLOGUE
        long v = 0L; bool zoneBad = false;
        if (r < P.nRows && !isNull) {
            bool zone, frac;
            if (!csv_iso(p, len, P.unitDigits, v, zone, frac)) bad = true;
            else if (zone != (P.expectZone != 0u)) zoneBad = true;
            else valid = true;
        }
        if (r < P.nRows) out[r] = valid ? v : 0L;
        if (zoneBad) atomic_fetch_min_explicit(err + 1, r, memory_order_relaxed);
        CSV_ROW_EPILOGUE
    }

    // Forced `null` type: every row must be one of the null values.
    kernel void csv_conv_null(CSV_CONV_ARGS) {
        CSV_ROW_PROLOGUE
        if (r < P.nRows && !isNull) bad = true;
        CSV_ROW_EPILOGUE
    }

    // Strings, pass 1: byte length per row (0 for a null row), validity, and the UTF-8 check when the
    // column was forced to utf8 (an inferred utf8 column was already checked by `csv_classify`).
    kernel void csv_str_len(CSV_CONV_ARGS, device int* lens [[buffer(6)]], constant uint& checkUtf8 [[buffer(9)]]) {
        CSV_ROW_PROLOGUE
        bool nullRow = isNull && (P.flags & CSV_SCBN) != 0u;
        if (r < P.nRows) {
            valid = !nullRow;
            lens[r] = nullRow ? 0 : (int)len;
            if (valid && checkUtf8 != 0u && !csv_utf8(p, len)) bad = true;
        }
        CSV_ROW_EPILOGUE
    }

    // ------------------------------------------------------------------ all columns in one pass

    // One converted column of `csv_convert_rows`: where its values, validity and flags go in the
    // shared output buffers, and how to parse it.
    struct CsvConvCol {
        uint col;
        uint kind;           // 0 null check, 1 int, 2 float64, 3 float32, 4 bool, 5 date32, 6 time32,
                             // 7 time64, 8 timestamp, 9 string lengths
        uint unitDigits;
        uint expectZone;
        uint isSigned;
        uint maxHex;
        uint width;          // integers: bytes per value
        uint checkUtf8;      // strings: validate
        ulong limPos;
        ulong limNeg;
        uint valuesOff;      // bytes into `values`
        uint validOff;       // words into `valid` (and into `host` for floats)
        uint pad0;
        uint pad1;
    };

    // Converts every row of every column in `descs` (none of them with complex fields). One thread
    // per row walks the row's fields left to right, so the boundaries and the file bytes are read once
    // for the whole table rather than once per column; with one kernel per column, each kernel touched
    // every cache line of both.
    kernel void csv_convert_rows(device const uchar* d [[buffer(0)]],
                                 device const uint* ev [[buffer(1)]],
                                 constant CsvCol& P [[buffer(2)]],
                                 device const CsvConvCol* descs [[buffer(3)]],
                                 constant uint& nDescs [[buffer(4)]],
                                 device const uint* lists [[buffer(5)]],
                                 device const uchar* lbytes [[buffer(6)]],
                                 device uchar* values [[buffer(7)]],
                                 device uint* valid [[buffer(8)]],
                                 device uint* hostFlags [[buffer(9)]],
                                 device atomic_uint* err [[buffer(10)]],
                                 device atomic_uint* hostCount [[buffer(11)]],
                                 uint r [[thread_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
        bool live = r < P.nRows;
        uint rec = P.firstRecord + (live ? r : 0u);
        for (uint j = 0u; j < nDescs; j++) {
            CsvConvCol D = descs[j];
            device const uchar* p = d; uint len = 0u; bool isNull = false;
            bool ok = false, bad = false, zoneBad = false, host = false, bit = false;
            if (live) {
                uint k = rec * P.nCols + D.col;
                uint2 raw = csv_raw(d, ev, k, P);
                bool quoted = false;
                if (P.quote <= 255u && raw.x < raw.y && (uint)d[raw.x] == P.quote) {
                    bool simple; uint cs;
                    len = csv_quoted(d, raw.x, raw.y, P, simple, cs);
                    p = d + cs;
                    quoted = true;
                } else { p = d + raw.x; len = raw.y - raw.x; }
                isNull = csv_is_null(p, len, quoted, P, lists, lbytes);
                device uchar* out = values + D.valuesOff;
                switch (D.kind) {
                case 0u:
                    if (!isNull) bad = true;
                    break;
                case 1u: {
                    ulong v = 0UL;
                    if (!isNull) { if (csv_int(p, len, D.isSigned != 0u, D.maxHex, D.limPos, D.limNeg, v)) ok = true; else bad = true; }
                    if (!ok) v = 0UL;
                    if (D.width == 8u) ((device ulong*)out)[r] = v;
                    else if (D.width == 4u) ((device uint*)out)[r] = (uint)v;
                    else if (D.width == 2u) ((device ushort*)out)[r] = (ushort)v;
                    else out[r] = (uchar)v;
                    break;
                }
                case 2u:
                case 3u: {
                    ulong bits = 0UL;
                    if (!isNull) {
                        uint st = fp_parse(p, len, FP_CSV, (uchar)P.decimalPoint, D.kind == 3u, bits);
                        if (st == FP_VALUE) ok = true; else if (st == FP_HOST) host = true; else bad = true;
                    }
                    if (!ok) bits = 0UL;
                    if (D.kind == 2u) ((device ulong*)out)[r] = bits; else ((device uint*)out)[r] = (uint)bits;
                    break;
                }
                case 4u:
                    if (!isNull) {
                        if (csv_match(p, len, P.nNull, P.nTrue, lists, lbytes)) { bit = true; ok = true; }
                        else if (csv_match(p, len, P.nNull + P.nTrue, P.nFalse, lists, lbytes)) ok = true;
                        else bad = true;
                    }
                    break;
                case 5u: {
                    int v = 0;
                    if (!isNull) { if (csv_date(p, len, v)) ok = true; else bad = true; }
                    ((device int*)out)[r] = ok ? v : 0;
                    break;
                }
                case 6u:
                case 7u: {
                    long v = 0L;
                    if (!isNull) { if (csv_time(p, len, D.unitDigits, v)) ok = true; else bad = true; }
                    if (!ok) v = 0L;
                    if (D.kind == 6u) ((device int*)out)[r] = (int)v; else ((device long*)out)[r] = v;
                    break;
                }
                case 8u: {
                    long v = 0L;
                    if (!isNull) {
                        bool zone, frac;
                        if (!csv_iso(p, len, D.unitDigits, v, zone, frac)) bad = true;
                        else if (zone != (D.expectZone != 0u)) zoneBad = true;
                        else ok = true;
                    }
                    ((device long*)out)[r] = ok ? v : 0L;
                    break;
                }
                default: {
                    bool nullRow = isNull && (P.flags & CSV_SCBN) != 0u;
                    ok = !nullRow;
                    ((device int*)out)[r] = nullRow ? 0 : (int)len;
                    if (ok && D.checkUtf8 != 0u && !csv_utf8(p, len)) bad = true;
                    break;
                }
                }
            }
            uint vw = (uint)(simd_vote::vote_t)simd_ballot(ok);
            uint bw = (uint)(simd_vote::vote_t)simd_ballot(bit);
            uint hw = (uint)(simd_vote::vote_t)simd_ballot(host);
            if (lane == 0u && live) {
                uint w = D.validOff + (r >> 5);
                valid[w] = vw;
                if (D.kind == 4u) ((device uint*)(values + D.valuesOff))[r >> 5] = bw;
                if (D.kind == 2u || D.kind == 3u) {
                    hostFlags[w] = hw;
                    if (hw != 0u) atomic_fetch_add_explicit(hostCount + j, popcount(hw), memory_order_relaxed);
                }
            }
            if (bad) atomic_fetch_min_explicit(err + 2u * j, r, memory_order_relaxed);
            if (zoneBad) atomic_fetch_min_explicit(err + 2u * j + 1u, r, memory_order_relaxed);
        }
    }

    // Strings, pass 2: copy each row's bytes to its offset.
    kernel void csv_str_copy(device const uchar* d [[buffer(0)]],
                             device const uchar* side [[buffer(1)]],
                             device const uint2* spans [[buffer(2)]],
                             constant CsvCol& P [[buffer(3)]],
                             device const int* offsets [[buffer(4)]],
                             device uchar* out [[buffer(5)]],
                             device const uint* ev [[buffer(6)]],
                             uint r [[thread_position_in_grid]]) {
        if (r >= P.nRows) return;
        int o = offsets[r], n = offsets[r + 1] - o;
        if (n <= 0) return;
        device const uchar* p = csv_ptr(csv_span(r, spans, ev, d, P), d, side);
        for (int i = 0; i < n; i++) out[o + i] = p[i];
    }
    """
}
