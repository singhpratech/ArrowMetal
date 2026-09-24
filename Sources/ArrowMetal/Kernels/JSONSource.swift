import Foundation

/// MSL for the newline-delimited JSON reader (`Sources/ArrowMetal/JSON/`, docs/JSON.md).
///
/// The reader runs in three stages, all on the GPU:
///
/// 1. **Structure** (`jb_*`, one thread per 64-byte block). A backslash escapes the byte after it, so
///    whether a block starts inside an escape depends only on the parity of the backslash run that ends
///    the last block before it that is not all backslashes: `jb_escape` writes that as a key, and a
///    max-scan (`js_*`) hands every block its carry. `jb_quotes` counts unescaped quotes, a sum-scan of
///    their parity gives the in-string state at every block start, `jb_depth` counts brackets outside
///    strings, and a sum-scan of those gives the nesting depth at every block start. `jb_records` and
///    `jb_emit` then find the top-level values: a `{` at depth 0 opens a record, the bracket that brings
///    the depth back to 0 closes it, and anything else at depth 0 that is not whitespace is an error.
/// 2. **Walk** (`jw_walk`, one thread per span). Each record (or, one level down, each nested object or
///    array) is walked by a single thread with an explicit container stack. The walk validates the full
///    JSON grammar with the error texts of RapidJSON (the parser pyarrow uses) and emits one `JEntry`
///    per immediate child: its key span, its value span and its kind. It runs twice, once to count and
///    once to write at the scanned offsets.
/// 3. **Columns** (`jk_*`, `jm_*`, `jc_*`, `jg_*`, `jt_*`). Keys are matched against the first object's
///    layout by a byte compare; the rest are dictionary-encoded on the GPU. Entries are scattered into a
///    row-by-field slot matrix, kinds are OR-reduced per column, strings are unescaped in a length pass
///    and a write pass, number text is gathered for the string-to-number parse, and ISO-8601 strings are
///    parsed to timestamps.
enum JSONSource {
    static let source: String = KernelSource.prelude + """

    // ---------------------------------------------------------------------------------------------
    // Shared definitions

    #define K_NULL   0u
    #define K_FALSE  1u
    #define K_TRUE   2u
    #define K_INT    3u
    #define K_FLOAT  4u
    #define K_STRING 5u
    #define K_OBJECT 6u
    #define K_ARRAY  7u
    #define F_ESC     0x10u   // the string value holds at least one escape
    #define F_KEYESC  0x20u   // the key holds at least one escape
    #define F_SPECIAL 0x40u   // NaN, Inf or Infinity

    #define E_NONE          0u
    #define E_VALUE         1u
    #define E_OBJ_NAME      2u
    #define E_OBJ_COLON     3u
    #define E_OBJ_COMMA     4u
    #define E_ARR_COMMA     5u
    #define E_STR_ESCAPE    6u
    #define E_STR_HEX       7u
    #define E_STR_SURROGATE 8u
    #define E_STR_ENCODING  9u
    #define E_STR_QUOTE     10u
    #define E_NUM_FRACTION  11u
    #define E_NUM_EXPONENT  12u
    #define E_NUM_TOO_BIG   13u
    #define E_DEPTH         14u
    #define E_DOC_EMPTY     15u
    #define E_TOP_ARRAY     16u
    #define E_TOP_STRING    17u
    #define E_TOP_NUMBER    18u
    #define E_TOP_BOOLEAN   19u

    #define MAX_DEPTH 1024u

    struct JEntry { uint parent; uint keyStart; uint keyLen; uint valStart; uint valLen; uint flags; };

    inline bool j_ws(uchar c) { return c == 0x20 || c == 0x09 || c == 0x0A || c == 0x0D; }
    inline bool j_digit(uchar c) { return c >= 0x30 && c <= 0x39; }
    inline int j_hex(uchar c) {
        if (c >= 0x30 && c <= 0x39) return (int)c - 0x30;
        if (c >= 0x41 && c <= 0x46) return (int)c - 0x41 + 10;
        if (c >= 0x61 && c <= 0x66) return (int)c - 0x61 + 10;
        return -1;
    }
    // Four hex digits at p (bounded by end); -1 when they are not all there.
    inline int j_hex4(device const uchar* s, uint p, uint end) {
        if (p + 4u > end) {
            // Report the first missing or bad digit the way a byte-at-a-time reader would.
            return -1;
        }
        int v = 0;
        for (uint k = 0; k < 4u; k++) { int h = j_hex(s[p + k]); if (h < 0) return -1; v = (v << 4) | h; }
        return v;
    }
    inline bool j_lit(device const uchar* s, uint p, uint end, uchar a, uchar b, uchar c, uchar d) {
        return p + 4u <= end && s[p] == a && s[p + 1] == b && s[p + 2] == c && s[p + 3] == d;
    }

    // ---------------------------------------------------------------------------------------------
    // Stage 1: structure, one thread per 64-byte block

    struct JBlk { uint n; uint start; uint nblocks; };

    // Escape carry key: 0 for a block of nothing but backslashes (it passes the carry through: 64 is
    // even), otherwise (block + 1) * 2 + (parity of the backslash run that ends the block).
    kernel void jb_escape(device const uchar* s [[buffer(0)]], constant JBlk& P [[buffer(1)]],
                          device uint* key [[buffer(2)]], uint b [[thread_position_in_grid]]) {
        if (b >= P.nblocks) return;
        uint lo = b * 64u, hi = min(lo + 64u, P.n);
        uint t = 0u; bool all = true;
        for (uint i = hi; i > lo; i--) { if (s[i - 1u] == 0x5C) t++; else { all = false; break; } }
        key[b] = all ? 0u : (((b + 1u) << 1) | (t & 1u));
    }

    // Unescaped quotes per block (parity only).
    kernel void jb_quotes(device const uchar* s [[buffer(0)]], constant JBlk& P [[buffer(1)]],
                          device const uint* escIn [[buffer(2)]], device int* qpar [[buffer(3)]],
                          uint b [[thread_position_in_grid]]) {
        if (b >= P.nblocks) return;
        uint lo = b * 64u, hi = min(lo + 64u, P.n);
        bool esc = (escIn[b] & 1u) != 0u;
        uint q = 0u;
        for (uint i = lo; i < hi; i++) {
            uchar c = s[i];
            if (esc) { esc = false; continue; }
            if (c == 0x5C) { esc = true; continue; }
            if (c == 0x22) q++;
        }
        qpar[b] = (int)(q & 1u);
    }

    // Net bracket depth change per block, counted outside strings.
    kernel void jb_depth(device const uchar* s [[buffer(0)]], constant JBlk& P [[buffer(1)]],
                         device const uint* escIn [[buffer(2)]], device const int* qScan [[buffer(3)]],
                         device int* delta [[buffer(4)]], uint b [[thread_position_in_grid]]) {
        if (b >= P.nblocks) return;
        uint lo = b * 64u, hi = min(lo + 64u, P.n);
        bool esc = (escIn[b] & 1u) != 0u;
        bool ins = (qScan[b] & 1) != 0;
        int d = 0;
        for (uint i = lo; i < hi; i++) {
            uchar c = s[i];
            if (esc) { esc = false; continue; }
            if (c == 0x5C) { esc = true; continue; }
            if (c == 0x22) { ins = !ins; continue; }
            if (ins) continue;
            if (c == 0x7B || c == 0x5B) d++;
            else if (c == 0x7D || c == 0x5D) d--;
        }
        delta[b] = d;
    }

    // A string starting at the quote at p. Returns the position after the closing quote.
    inline uint j_string(device const uchar* s, uint p, uint end, thread uint& err, thread bool& esc) {
        p++;
        while (true) {
            if (p >= end) { err = E_STR_QUOTE; return p; }
            uchar c = s[p];
            if (c == 0x22) return p + 1u;
            if (c == 0x5C) {
                esc = true;
                uchar e = (p + 1u < end) ? s[p + 1u] : (uchar)0;
                if (e == 0x22 || e == 0x5C || e == 0x2F || e == 0x62 || e == 0x66 || e == 0x6E || e == 0x72 || e == 0x74) { p += 2u; continue; }
                if (e == 0x75) {
                    int cp = j_hex4(s, p + 2u, end);
                    if (cp < 0) { err = E_STR_HEX; return p; }
                    p += 6u;
                    if (cp >= 0xD800 && cp <= 0xDFFF) {
                        if (cp <= 0xDBFF) {
                            if (!(p + 1u < end && s[p] == 0x5C && s[p + 1u] == 0x75)) { err = E_STR_SURROGATE; return p; }
                            int lo = j_hex4(s, p + 2u, end);
                            if (lo < 0) { err = E_STR_HEX; return p; }
                            if (lo < 0xDC00 || lo > 0xDFFF) { err = E_STR_SURROGATE; return p; }
                            p += 6u;
                        } else { err = E_STR_SURROGATE; return p; }
                    }
                    continue;
                }
                err = E_STR_ESCAPE; return p;
            }
            if (c < 0x20) { err = (c == 0) ? E_STR_QUOTE : E_STR_ENCODING; return p; }
            p++;
        }
    }

    // A number, following RapidJSON's reader with kParseNumbersAsStringsFlag and kParseNanAndInfFlag:
    // the same grammar, the same errors, and the same "Number too big" rule for positive exponents
    // (exponent above 308 plus the fraction digits that reached the significand).
    inline uint j_number(device const uchar* s, uint p, uint end, thread uint& err, thread uint& kind,
                         thread uint& flags) {
        bool minus = false;
        if (p < end && s[p] == 0x2D) { minus = true; p++; }
        uint i32 = 0u; ulong i64 = 0ul; bool use64 = false, useDouble = false, special = false;
        int sig = 0;
        uchar c = (p < end) ? s[p] : (uchar)0;
        if (c == 0x30) { p++; }
        else if (c >= 0x31 && c <= 0x39) {
            i32 = (uint)(c - 0x30); p++;
            uint lim = minus ? 214748364u : 429496729u;
            uchar last = minus ? (uchar)0x38 : (uchar)0x35;
            while (p < end && j_digit(s[p])) {
                if (i32 >= lim) {
                    if (i32 != lim || s[p] > last) { i64 = (ulong)i32; use64 = true; break; }
                }
                i32 = i32 * 10u + (uint)(s[p] - 0x30); p++; sig++;
            }
        } else if (c == 0x4E || c == 0x49) {
            bool ok = false;
            if (c == 0x4E) {
                p++;
                if (p < end && s[p] == 0x61) { p++; if (p < end && s[p] == 0x4E) { p++; ok = true; } }
            } else {
                p++;
                if (p < end && s[p] == 0x6E) { p++; if (p < end && s[p] == 0x66) { p++; ok = true;
                    if (p < end && s[p] == 0x69) {
                        const uchar rest[5] = {0x69, 0x6E, 0x69, 0x74, 0x79};
                        for (uint k = 0; k < 5u; k++) {
                            if (p < end && s[p] == rest[k]) p++;
                            else { err = E_VALUE; return p; }
                        }
                    }
                } }
            }
            if (!ok) { err = E_VALUE; return p; }
            special = true;
        } else { err = E_VALUE; return p; }

        if (use64) {
            ulong lim = minus ? 0x0CCCCCCCCCCCCCCCul : 0x1999999999999999ul;
            uchar last = minus ? (uchar)0x38 : (uchar)0x35;
            while (p < end && j_digit(s[p])) {
                if (i64 >= lim) {
                    if (i64 != lim || s[p] > last) { useDouble = true; break; }
                }
                i64 = i64 * 10ul + (ulong)(s[p] - 0x30); p++; sig++;
            }
        }
        if (useDouble) { while (p < end && j_digit(s[p])) p++; }

        bool isFloat = special;
        long expFrac = 0;
        if (p < end && s[p] == 0x2E) {
            p++;
            if (!(p < end && j_digit(s[p]))) { err = E_NUM_FRACTION; return p; }
            bool nonZero;
            if (!useDouble) {
                if (!use64) i64 = (ulong)i32;
                while (p < end && j_digit(s[p])) {
                    if (i64 > 0x1FFFFFFFFFFFFFul) break;
                    i64 = i64 * 10ul + (ulong)(s[p] - 0x30); p++; expFrac--;
                    if (i64 != 0ul) sig++;
                }
                nonZero = (i64 != 0ul);
                useDouble = true;
            } else {
                nonZero = true;
            }
            while (p < end && j_digit(s[p])) {
                if (sig < 17) {
                    if (s[p] != 0x30) nonZero = true;
                    p++; expFrac--;
                    if (nonZero) sig++;
                } else p++;
            }
            isFloat = true;
        }
        if (p < end && (s[p] == 0x65 || s[p] == 0x45)) {
            p++;
            isFloat = true;
            bool expMinus = false;
            if (p < end && s[p] == 0x2B) p++;
            else if (p < end && s[p] == 0x2D) { expMinus = true; p++; }
            if (p < end && j_digit(s[p])) {
                long e = (long)(s[p] - 0x30); p++;
                if (expMinus) { while (p < end && j_digit(s[p])) p++; }
                else {
                    long maxExp = 308 - expFrac;
                    while (p < end && j_digit(s[p])) {
                        e = e * 10 + (long)(s[p] - 0x30); p++;
                        if (e > maxExp) { err = E_NUM_TOO_BIG; return p; }
                    }
                }
            } else { err = E_NUM_EXPONENT; return p; }
        }
        bool fits = true;
        if (useDouble) fits = false;
        else if (use64) fits = minus ? (i64 <= 0x8000000000000000ul) : (i64 <= 0x7FFFFFFFFFFFFFFFul);
        kind = (isFloat || !fits) ? K_FLOAT : K_INT;
        if (special) flags |= F_SPECIAL;
        return p;
    }

    // Is byte i (at depth 0, outside strings) part of a top-level `null` literal that starts at i - k?
    inline bool j_in_null(device const uchar* s, uint i, uint n, uint start) {
        for (uint k = 1u; k <= 3u; k++) {
            if (i < start + k) break;
            uint p = i - k;
            if (j_lit(s, p, n, 0x6E, 0x75, 0x6C, 0x6C)) return true;
        }
        return false;
    }

    // The error for a value found at depth 0, where only objects (and `null`) may start a record. A
    // sequential parser reads a string or a number to its end before it reports the value's type, so
    // an invalid token reports its own error; an array is reported at its opening bracket.
    inline uint j_top_error(device const uchar* s, uint i, uint n, uchar c) {
        if (c == 0x5B) return E_TOP_ARRAY;
        if (c == 0x7D || c == 0x5D || c == 0x2C || c == 0x3A) return E_DOC_EMPTY;
        uint err = E_NONE;
        if (c == 0x22) {
            bool esc = false;
            j_string(s, i, n, err, esc);
            return err != E_NONE ? err : E_TOP_STRING;
        }
        if (c == 0x74) return j_lit(s, i, n, 0x74, 0x72, 0x75, 0x65) ? E_TOP_BOOLEAN : E_VALUE;
        if (c == 0x66) {
            return (i + 5u <= n && s[i+1] == 0x61 && s[i+2] == 0x6C && s[i+3] == 0x73 && s[i+4] == 0x65) ? E_TOP_BOOLEAN : E_VALUE;
        }
        uint kind = K_NULL, flags = 0u;
        j_number(s, i, n, err, kind, flags);
        return err != E_NONE ? err : E_TOP_NUMBER;
    }

    // Walks one block with its carries. mode 0 counts record starts and reports top-level errors;
    // mode 1 writes record starts and ends.
    inline void j_block_records(device const uchar* s, constant JBlk& P, uint b, bool esc, bool ins, int d,
                                uint mode, thread uint& count, device uint* recStart, device uint* recEnd,
                                uint base, device atomic_uint* firstErr, device uint* errCode) {
        uint lo = b * 64u, hi = min(lo + 64u, P.n);
        bool reported = false;
        for (uint i = lo; i < hi; i++) {
            uchar c = s[i];
            if (esc) { esc = false; continue; }
            bool top = (d == 0 && !ins && i >= P.start);
            if (c == 0x5C) {
                esc = true;
                if (top && mode == 0u && !reported) { reported = true; errCode[b] = E_VALUE; atomic_fetch_min_explicit(firstErr, i, memory_order_relaxed); }
                continue;
            }
            if (c == 0x22) {
                if (top && mode == 0u && !reported) { reported = true; errCode[b] = j_top_error(s, i, P.n, c); atomic_fetch_min_explicit(firstErr, i, memory_order_relaxed); }
                ins = !ins; continue;
            }
            if (ins) continue;
            if (i < P.start) continue;
            if (d == 0) {
                if (j_ws(c)) continue;
                if (c == 0x7B) {
                    if (mode == 1u) recStart[base + count] = i;
                    count++; d = 1; continue;
                }
                if (c == 0x6E && j_lit(s, i, P.n, 0x6E, 0x75, 0x6C, 0x6C)) {
                    if (mode == 1u) { recStart[base + count] = i; recEnd[base + count] = i + 4u; }
                    count++; continue;
                }
                if ((c == 0x75 || c == 0x6C) && j_in_null(s, i, P.n, P.start)) continue;
                if (mode == 0u && !reported) {
                    reported = true;
                    errCode[b] = j_top_error(s, i, P.n, c);
                    atomic_fetch_min_explicit(firstErr, i, memory_order_relaxed);
                }
                if (c == 0x5B) d++;
                else if (c == 0x7D || c == 0x5D) d--;
                continue;
            }
            if (c == 0x7B || c == 0x5B) d++;
            else if (c == 0x7D || c == 0x5D) {
                d--;
                if (d == 0 && mode == 1u && count + base > 0u) recEnd[base + count - 1u] = i + 1u;
            }
        }
    }

    kernel void jb_records(device const uchar* s [[buffer(0)]], constant JBlk& P [[buffer(1)]],
                           device const uint* escIn [[buffer(2)]], device const int* qScan [[buffer(3)]],
                           device const int* dScan [[buffer(4)]], device int* counts [[buffer(5)]],
                           device atomic_uint* firstErr [[buffer(6)]], device uint* errCode [[buffer(7)]],
                           uint b [[thread_position_in_grid]]) {
        if (b >= P.nblocks) return;
        uint count = 0u;
        j_block_records(s, P, b, (escIn[b] & 1u) != 0u, (qScan[b] & 1) != 0, dScan[b], 0u, count,
                        (device uint*)errCode, (device uint*)errCode, 0u, firstErr, errCode);
        counts[b] = (int)count;
    }

    kernel void jb_emit(device const uchar* s [[buffer(0)]], constant JBlk& P [[buffer(1)]],
                        device const uint* escIn [[buffer(2)]], device const int* qScan [[buffer(3)]],
                        device const int* dScan [[buffer(4)]], device const int* recBase [[buffer(5)]],
                        device uint* recStart [[buffer(6)]], device uint* recEnd [[buffer(7)]],
                        device atomic_uint* unused [[buffer(8)]], device uint* unusedCode [[buffer(9)]],
                        uint b [[thread_position_in_grid]]) {
        if (b >= P.nblocks) return;
        uint count = 0u;
        j_block_records(s, P, b, (escIn[b] & 1u) != 0u, (qScan[b] & 1) != 0, dScan[b], 1u, count,
                        recStart, recEnd, (uint)recBase[b], unused, unusedCode);
    }

    // ---------------------------------------------------------------------------------------------
    // Exclusive max-scan over uint (the escape carries): a per-threadgroup scan that also writes each
    // group's maximum, the same scan applied to those maxima (recursively, from the host), then an add.

    kernel void js_max_block(device const uint* v [[buffer(0)]], constant uint& n [[buffer(1)]],
                             device uint* out [[buffer(2)]], device uint* totals [[buffer(3)]],
                             uint i [[thread_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                             uint tgid [[threadgroup_position_in_grid]]) {
        threadgroup uint buf[TG];
        uint x = (i < n) ? v[i] : 0u;
        buf[lid] = x;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = 1u; off < TG; off <<= 1) {
            uint y = (lid >= off) ? buf[lid - off] : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            buf[lid] = max(buf[lid], y);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // exclusive within the group
        if (i < n) out[i] = (lid == 0u) ? 0u : buf[lid - 1u];
        if (lid == TG - 1u) totals[tgid] = buf[TG - 1u];
    }
    kernel void js_max_add(device uint* out [[buffer(0)]], device const uint* totals [[buffer(1)]],
                           constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]],
                           uint tgid [[threadgroup_position_in_grid]]) {
        if (i < n) out[i] = max(out[i], totals[tgid]);
    }

    // ---------------------------------------------------------------------------------------------
    // Stage 2: the walk. One thread per span; emits one entry per immediate child.

    #define S_VALUE     0u
    #define S_OBJ_FIRST 1u
    #define S_OBJ_KEY   2u
    #define S_COLON     3u
    #define S_ARR_FIRST 4u
    #define S_AFTER     5u

    struct JWalk { uint n; uint nspans; uint emit; };

    // Emits (or counts) one child of the span. `partialEnd` is the error position for a child the walk
    // stopped inside, which is still reported so the columns see what a sequential parser saw before
    // the error: the kind of a container it had opened, or a key whose value never came.
    inline void j_child(constant JWalk& P, device JEntry* entries, uint base, thread uint& cnt, uint t,
                        uint keyStart, uint keyLen, uint valStart, uint valEnd, uint flags) {
        if (P.emit) {
            JEntry x; x.parent = t; x.keyStart = keyStart; x.keyLen = keyLen;
            x.valStart = valStart; x.valLen = valEnd - valStart; x.flags = flags;
            entries[base + cnt] = x;
        }
        cnt++;
    }

    kernel void jw_walk(device const uchar* s [[buffer(0)]], constant JWalk& P [[buffer(1)]],
                        device const uint* spanStart [[buffer(2)]], device const uint* spanEnd [[buffer(3)]],
                        device int* counts [[buffer(4)]], device const int* offsets [[buffer(5)]],
                        device JEntry* entries [[buffer(6)]], device uchar* errCode [[buffer(7)]],
                        device atomic_uint* firstErr [[buffer(8)]], device uint* errPos [[buffer(9)]],
                        uint t [[thread_position_in_grid]]) {
        if (t >= P.nspans) return;
        uint p = spanStart[t], end = min(spanEnd[t], P.n);
        uint cnt = 0u;
        uint base = P.emit ? (uint)offsets[t] : 0u;
        if (p >= end || s[p] == 0x6E) { if (!P.emit) counts[t] = 0; return; }
        uint stk[MAX_DEPTH / 32u];
        uint d = 0u;
        uint state = S_VALUE;
        uint err = E_NONE;
        uint keyStart = 0u, keyLen = 0u, keyFlags = 0u, valStart = 0u;
        bool haveKey = false;
        bool done = false;
        while (!done && err == E_NONE) {
            while (p < end && j_ws(s[p])) p++;
            uchar c = (p < end) ? s[p] : (uchar)0;
            bool close = false;
            if (state == S_VALUE) {
                if (d == 1u) valStart = p;
                if (c == 0x7B || c == 0x5B) {
                    if (d >= MAX_DEPTH) { err = E_DEPTH; break; }
                    uint w = d >> 5, bit = 1u << (d & 31u);
                    if (c == 0x5B) stk[w] |= bit; else stk[w] &= ~bit;
                    d++; p++;
                    state = (c == 0x7B) ? S_OBJ_FIRST : S_ARR_FIRST;
                    continue;
                }
                uint kind = K_NULL, fl = 0u;
                if (c == 0x22) {
                    bool esc = false;
                    p = j_string(s, p, end, err, esc);
                    kind = K_STRING; if (esc) fl |= F_ESC;
                } else if (c == 0x2D || j_digit(c) || c == 0x4E || c == 0x49) {
                    p = j_number(s, p, end, err, kind, fl);
                } else if (c == 0x74) {
                    if (j_lit(s, p, end, 0x74, 0x72, 0x75, 0x65)) { p += 4u; kind = K_TRUE; } else err = E_VALUE;
                } else if (c == 0x66) {
                    if (p + 5u <= end && s[p+1] == 0x61 && s[p+2] == 0x6C && s[p+3] == 0x73 && s[p+4] == 0x65) { p += 5u; kind = K_FALSE; } else err = E_VALUE;
                } else if (c == 0x6E) {
                    if (j_lit(s, p, end, 0x6E, 0x75, 0x6C, 0x6C)) { p += 4u; kind = K_NULL; } else err = E_VALUE;
                } else {
                    err = E_VALUE;
                }
                if (err != E_NONE) break;
                if (d == 0u) { done = true; break; }       // a scalar span (never produced by the reader)
                if (d == 1u) {
                    j_child(P, entries, base, cnt, t, keyStart, keyLen, valStart, p, kind | fl | keyFlags);
                    haveKey = false;
                }
                state = S_AFTER;
                continue;
            }
            if (state == S_OBJ_FIRST || state == S_OBJ_KEY) {
                if (state == S_OBJ_FIRST && c == 0x7D) { close = true; }
                else if (c == 0x22) {
                    bool esc = false;
                    uint ks = p + 1u;
                    p = j_string(s, p, end, err, esc);
                    if (err != E_NONE) break;
                    if (d == 1u) { keyStart = ks; keyLen = p - 1u - ks; keyFlags = esc ? F_KEYESC : 0u; haveKey = true; }
                    state = S_COLON;
                    continue;
                } else { err = E_OBJ_NAME; break; }
            } else if (state == S_COLON) {
                if (c == 0x3A) { p++; state = S_VALUE; continue; }
                err = E_OBJ_COLON; break;
            } else if (state == S_ARR_FIRST) {
                if (c == 0x5D) close = true;
                else { state = S_VALUE; continue; }
            } else { // S_AFTER
                uint w = (d - 1u) >> 5, bit = 1u << ((d - 1u) & 31u);
                bool isArr = (stk[w] & bit) != 0u;
                if (c == 0x2C) { p++; state = isArr ? S_VALUE : S_OBJ_KEY; continue; }
                if (c == (isArr ? 0x5D : 0x7D)) close = true;
                else { err = isArr ? E_ARR_COMMA : E_OBJ_COMMA; break; }
            }
            if (close) {
                uint w = (d - 1u) >> 5, bit = 1u << ((d - 1u) & 31u);
                bool wasArr = (stk[w] & bit) != 0u;
                d--; p++;
                if (d == 0u) { done = true; break; }
                if (d == 1u) {
                    j_child(P, entries, base, cnt, t, keyStart, keyLen, valStart, p, (wasArr ? K_ARRAY : K_OBJECT) | keyFlags);
                    haveKey = false;
                }
                state = S_AFTER;
            }
        }
        if (err != E_NONE) {
            // What a sequential parser had already reported for the child it stopped inside.
            if (d >= 2u) {
                bool childArr = (stk[0] & 2u) != 0u;
                j_child(P, entries, base, cnt, t, keyStart, keyLen, valStart, p, (childArr ? K_ARRAY : K_OBJECT) | keyFlags);
            } else if (d == 1u && haveKey) {
                j_child(P, entries, base, cnt, t, keyStart, keyLen, p, p, K_NULL | keyFlags);
            }
        }
        if (!P.emit) {
            counts[t] = (int)cnt;
            if (err != E_NONE) {
                errCode[t] = (uchar)err;
                errPos[t] = p;
                atomic_fetch_min_explicit(firstErr, t, memory_order_relaxed);
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Stage 3: keys and columns

    struct JKey { uint count; uint K0; uint refFirst; };

    // Field id by position: entry k of a span is field k when its raw key bytes equal those of entry k
    // of the reference span. Everything else is novel and goes to the dictionary encoder.
    kernel void jk_match(device const uchar* s [[buffer(0)]], constant JKey& P [[buffer(1)]],
                         device const JEntry* ent [[buffer(2)]], device const int* off [[buffer(3)]],
                         device int* fid [[buffer(4)]], device int* novel [[buffer(5)]],
                         uint e [[thread_position_in_grid]]) {
        if (e >= P.count) return;
        JEntry x = ent[e];
        uint k = e - (uint)off[x.parent];
        int f = -1;
        if (k < P.K0) {
            JEntry r = ent[P.refFirst + k];
            if (r.keyLen == x.keyLen) {
                bool eq = true;
                for (uint i = 0u; i < x.keyLen; i++) { if (s[x.keyStart + i] != s[r.keyStart + i]) { eq = false; break; } }
                if (eq) f = (int)k;
            }
        }
        fid[e] = f;
        novel[e] = (f < 0) ? 1 : 0;
    }

    // Compacts the novel entries: out[pos[e]] = e.
    kernel void jk_compact(device const int* flag [[buffer(0)]], device const int* pos [[buffer(1)]],
                           constant uint& count [[buffer(2)]], device int* out [[buffer(3)]],
                           uint e [[thread_position_in_grid]]) {
        if (e < count && flag[e] != 0) out[pos[e]] = (int)e;
    }

    // fid[list[j]] = codes[skip + j]
    kernel void jk_assign(device const int* list [[buffer(0)]], device const int* codes [[buffer(1)]],
                          constant uint& count [[buffer(2)]], constant uint& skip [[buffer(3)]],
                          device int* fid [[buffer(4)]], uint j [[thread_position_in_grid]]) {
        if (j < count) fid[list[j]] = codes[skip + j];
    }

    struct JScatter { uint count; uint rows; int fidLo; int fidHi; uint atomicMode; };

    // Slot matrix, one column of `rows` slots per field: slot = min entry index holding that field.
    kernel void jm_scatter(constant JScatter& P [[buffer(0)]], device const JEntry* ent [[buffer(1)]],
                           device const int* fid [[buffer(2)]], device uint* M [[buffer(3)]],
                           uint e [[thread_position_in_grid]]) {
        if (e >= P.count) return;
        int f = fid[e];
        if (f < P.fidLo || f >= P.fidHi) return;
        uint slot = (uint)(f - P.fidLo) * P.rows + ent[e].parent;
        if (P.atomicMode) atomic_fetch_min_explicit((device atomic_uint*)&M[slot], e, memory_order_relaxed);
        else M[slot] = e;
    }
    // A field named twice in one object: the later entry lost the slot to the earlier one.
    kernel void jm_dups(constant JScatter& P [[buffer(0)]], device const JEntry* ent [[buffer(1)]],
                        device const int* fid [[buffer(2)]], device const uint* M [[buffer(3)]],
                        device atomic_uint* firstDup [[buffer(4)]], uint e [[thread_position_in_grid]]) {
        if (e >= P.count) return;
        int f = fid[e];
        if (f < P.fidLo || f >= P.fidHi) return;
        uint slot = (uint)(f - P.fidLo) * P.rows + ent[e].parent;
        if (M[slot] != e) atomic_fetch_min_explicit(firstDup, e, memory_order_relaxed);
    }
    // First entry whose field is not wanted (unexpected_field_behavior="error").
    kernel void jm_unexpected(constant uint& count [[buffer(0)]], device const int* fid [[buffer(1)]],
                              device const uchar* expected [[buffer(2)]], device atomic_uint* first [[buffer(3)]],
                              uint e [[thread_position_in_grid]]) {
        if (e >= count) return;
        int f = fid[e];
        if (f >= 0 && expected[f] == 0) atomic_fetch_min_explicit(first, e, memory_order_relaxed);
    }

    struct JCol { uint rows; uint identity; uint validMask; };

    inline int j_row_entry(constant JCol& P, device const int* rowEntry, uint r) {
        return P.identity ? (int)r : rowEntry[r];
    }

    // OR of (1 << kind) and of the flag bits over one column.
    kernel void jc_kinds(constant JCol& P [[buffer(0)]], device const JEntry* ent [[buffer(1)]],
                         device const int* rowEntry [[buffer(2)]], device atomic_uint* out [[buffer(3)]],
                         uint r [[thread_position_in_grid]]) {
        uint km = 0u, fl = 0u;
        if (r < P.rows) {
            int e = j_row_entry(P, rowEntry, r);
            if (e < 0) km = 1u << K_NULL;
            else { uint f = ent[e].flags; km = 1u << (f & 15u); fl = f & 0xF0u; }
        }
        km = simd_or(km); fl = simd_or(fl);
        if (simd_is_first()) {
            if (km) atomic_fetch_or_explicit(&out[0], km, memory_order_relaxed);
            if (fl) atomic_fetch_or_explicit(&out[1], fl, memory_order_relaxed);
        }
    }

    // Validity bitmap (32 rows per thread): valid when the row has an entry whose kind is in validMask.
    // With `boolValues` set it also writes the boolean values (kind true).
    kernel void jc_validity(constant JCol& P [[buffer(0)]], device const JEntry* ent [[buffer(1)]],
                            device const int* rowEntry [[buffer(2)]], device uint* valid [[buffer(3)]],
                            device uint* values [[buffer(4)]], constant uint& boolValues [[buffer(5)]],
                            device atomic_uint* nulls [[buffer(6)]], uint w [[thread_position_in_grid]]) {
        uint words = (P.rows + 31u) / 32u;
        uint nn = 0u;
        if (w < words) {
            uint vb = 0u, tb = 0u;
            uint lo = w * 32u, hi = min(lo + 32u, P.rows);
            for (uint r = lo; r < hi; r++) {
                int e = j_row_entry(P, rowEntry, r);
                uint k = (e < 0) ? K_NULL : (ent[e].flags & 15u);
                if ((P.validMask >> k) & 1u) { vb |= 1u << (r - lo); if (k == K_TRUE) tb |= 1u << (r - lo); }
                else nn++;
            }
            valid[w] = vb;
            if (boolValues) values[w] = tb;
        }
        nn = simd_sum(nn);
        if (simd_is_first() && nn) atomic_fetch_add_explicit(nulls, nn, memory_order_relaxed);
    }

    // Spans of the nested objects or arrays of a column (empty for rows of another kind).
    kernel void jc_spans(constant JCol& P [[buffer(0)]], device const JEntry* ent [[buffer(1)]],
                         device const int* rowEntry [[buffer(2)]], device uint* spanStart [[buffer(3)]],
                         device uint* spanEnd [[buffer(4)]], uint r [[thread_position_in_grid]]) {
        if (r >= P.rows) return;
        int e = j_row_entry(P, rowEntry, r);
        if (e >= 0 && ((P.validMask >> (ent[e].flags & 15u)) & 1u)) {
            spanStart[r] = ent[e].valStart; spanEnd[r] = ent[e].valStart + ent[e].valLen;
        } else { spanStart[r] = 0u; spanEnd[r] = 0u; }
    }

    // ---- string gathers. mode 0: key (unescaped); 1: string value (unescaped, quotes dropped);
    // 2: raw value text (numbers).

    inline uint j_utf8_len(uint cp) { return cp < 0x80u ? 1u : (cp < 0x800u ? 2u : (cp < 0x10000u ? 3u : 4u)); }

    // Decoded length of an escaped string body (validated by the walk).
    inline uint j_decoded_len(device const uchar* s, uint p, uint len) {
        uint o = 0u, i = 0u;
        while (i < len) {
            uchar c = s[p + i];
            if (c != 0x5C) { o++; i++; continue; }
            uchar e = s[p + i + 1u];
            if (e != 0x75) { o++; i += 2u; continue; }
            uint cp = (uint)j_hex4(s, p + i + 2u, p + len);
            i += 6u;
            if (cp >= 0xD800u && cp <= 0xDBFFu) { cp = 0x10000u; i += 6u; }
            o += j_utf8_len(cp);
        }
        return o;
    }
    inline void j_decode(device const uchar* s, uint p, uint len, device uchar* out) {
        uint o = 0u, i = 0u;
        while (i < len) {
            uchar c = s[p + i];
            if (c != 0x5C) { out[o++] = c; i++; continue; }
            uchar e = s[p + i + 1u];
            if (e != 0x75) {
                uchar v = e;
                if (e == 0x62) v = 0x08; else if (e == 0x66) v = 0x0C; else if (e == 0x6E) v = 0x0A;
                else if (e == 0x72) v = 0x0D; else if (e == 0x74) v = 0x09;
                out[o++] = v; i += 2u; continue;
            }
            uint cp = (uint)j_hex4(s, p + i + 2u, p + len);
            i += 6u;
            if (cp >= 0xD800u && cp <= 0xDBFFu) {
                uint lo = (uint)j_hex4(s, p + i + 2u, p + len);
                cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
                i += 6u;
            }
            if (cp < 0x80u) out[o++] = (uchar)cp;
            else if (cp < 0x800u) { out[o++] = (uchar)(0xC0u | (cp >> 6)); out[o++] = (uchar)(0x80u | (cp & 0x3Fu)); }
            else if (cp < 0x10000u) {
                out[o++] = (uchar)(0xE0u | (cp >> 12)); out[o++] = (uchar)(0x80u | ((cp >> 6) & 0x3Fu));
                out[o++] = (uchar)(0x80u | (cp & 0x3Fu));
            } else {
                out[o++] = (uchar)(0xF0u | (cp >> 18)); out[o++] = (uchar)(0x80u | ((cp >> 12) & 0x3Fu));
                out[o++] = (uchar)(0x80u | ((cp >> 6) & 0x3Fu)); out[o++] = (uchar)(0x80u | (cp & 0x3Fu));
            }
        }
    }

    struct JGather { uint rows; uint identity; uint mode; uint validMask; };

    inline void j_source(constant JGather& P, JEntry x, thread uint& p, thread uint& len, thread bool& esc) {
        if (P.mode == 0u) { p = x.keyStart; len = x.keyLen; esc = (x.flags & F_KEYESC) != 0u; }
        else if (P.mode == 1u) { p = x.valStart + 1u; len = x.valLen - 2u; esc = (x.flags & F_ESC) != 0u; }
        else { p = x.valStart; len = x.valLen; esc = false; }
    }

    kernel void jg_len(device const uchar* s [[buffer(0)]], constant JGather& P [[buffer(1)]],
                       device const JEntry* ent [[buffer(2)]], device const int* rowEntry [[buffer(3)]],
                       device int* lens [[buffer(4)]], uint r [[thread_position_in_grid]]) {
        if (r >= P.rows) return;
        int e = P.identity ? (int)r : rowEntry[r];
        if (e < 0) { lens[r] = 0; return; }
        JEntry x = ent[e];
        if (P.mode != 0u && !((P.validMask >> (x.flags & 15u)) & 1u)) { lens[r] = 0; return; }
        uint p, len; bool esc;
        j_source(P, x, p, len, esc);
        lens[r] = (int)(esc ? j_decoded_len(s, p, len) : len);
    }

    kernel void jg_write(device const uchar* s [[buffer(0)]], constant JGather& P [[buffer(1)]],
                         device const JEntry* ent [[buffer(2)]], device const int* rowEntry [[buffer(3)]],
                         device const int* offsets [[buffer(4)]], device uchar* out [[buffer(5)]],
                         uint r [[thread_position_in_grid]]) {
        if (r >= P.rows) return;
        int e = P.identity ? (int)r : rowEntry[r];
        if (e < 0) return;
        JEntry x = ent[e];
        if (P.mode != 0u && !((P.validMask >> (x.flags & 15u)) & 1u)) return;
        uint p, len; bool esc;
        j_source(P, x, p, len, esc);
        device uchar* dst = out + offsets[r];
        if (esc) j_decode(s, p, len, dst);
        else for (uint i = 0u; i < len; i++) dst[i] = s[p + i];
    }

    // ---- ISO-8601 timestamps, following Arrow's ParseTimestampISO8601 byte for byte.

    inline bool j_num2(device const uchar* d, uint p, thread uint& v) {
        if (!j_digit(d[p]) || !j_digit(d[p + 1u])) return false;
        v = (uint)(d[p] - 0x30) * 10u + (uint)(d[p + 1u] - 0x30); return true;
    }
    inline bool j_hh(device const uchar* d, uint p, thread long& secs) {
        uint h; if (!j_num2(d, p, h) || h >= 24u) return false; secs = (long)h * 3600; return true;
    }
    inline bool j_hh_mm(device const uchar* d, uint p, thread long& secs) {
        uint h, m; if (d[p + 2u] != 0x3A) return false;
        if (!j_num2(d, p, h) || !j_num2(d, p + 3u, m) || h >= 24u || m >= 60u) return false;
        secs = (long)h * 3600 + (long)m * 60; return true;
    }
    inline bool j_hhmm(device const uchar* d, uint p, thread long& secs) {
        uint h, m;
        if (!j_num2(d, p, h) || !j_num2(d, p + 2u, m) || h >= 24u || m >= 60u) return false;
        secs = (long)h * 3600 + (long)m * 60; return true;
    }
    inline bool j_hh_mm_ss(device const uchar* d, uint p, thread long& secs) {
        uint h, m, x; if (d[p + 2u] != 0x3A || d[p + 5u] != 0x3A) return false;
        if (!j_num2(d, p, h) || !j_num2(d, p + 3u, m) || !j_num2(d, p + 6u, x)) return false;
        if (h >= 24u || m >= 60u || x >= 60u) return false;
        secs = (long)h * 3600 + (long)m * 60 + (long)x; return true;
    }
    inline long j_days_from_civil(long y, uint m, uint d) {
        y -= (m <= 2u) ? 1 : 0;
        long era = (y >= 0 ? y : y - 399) / 400;
        long yoe = y - era * 400;
        long mp = (long)((m + 9u) % 12u);
        long doy = (153 * mp + 2) / 5 + (long)d - 1;
        long doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        return era * 146097 + doe - 719468;
    }
    inline bool j_leap(uint y) { return (y % 4u == 0u && y % 100u != 0u) || y % 400u == 0u; }

    // unitScale: ticks per second (1, 1e3, 1e6, 1e9); fracMax: fraction digits the unit holds (0, 3, 6, 9).
    inline bool j_iso8601(device const uchar* d, uint p, uint length, long unitScale, uint fracMax, thread long& out) {
        if (length < 10u) return false;
        if (d[p + 4u] != 0x2D || d[p + 7u] != 0x2D) return false;
        uint yh, yl, mo, dy;
        if (!j_num2(d, p, yh) || !j_num2(d, p + 2u, yl) || !j_num2(d, p + 5u, mo) || !j_num2(d, p + 8u, dy)) return false;
        uint year = yh * 100u + yl;
        if (mo < 1u || mo > 12u || dy < 1u) return false;
        const uint mdays[12] = {31u, 28u, 31u, 30u, 31u, 30u, 31u, 31u, 30u, 31u, 30u, 31u};
        uint dim = mdays[mo - 1u] + ((mo == 2u && j_leap(year)) ? 1u : 0u);
        if (dy > dim) return false;
        long secs = j_days_from_civil((long)year, mo, dy) * 86400;
        if (length == 10u) { out = secs * unitScale; return true; }
        if (d[p + 10u] != 0x20 && d[p + 10u] != 0x54) return false;
        long zone = 0;
        if (d[p + length - 1u] == 0x5A) {
            length--;
        } else if (d[p + length - 3u] == 0x2B || d[p + length - 3u] == 0x2D) {
            length -= 3u;
            if (!j_hh(d, p + length + 1u, zone)) return false;
            if (d[p + length] == 0x2B) zone = -zone;
        } else if (length >= 5u && (d[p + length - 5u] == 0x2B || d[p + length - 5u] == 0x2D)) {
            length -= 5u;
            if (!j_hhmm(d, p + length + 1u, zone)) return false;
            if (d[p + length] == 0x2B) zone = -zone;
        } else if (length >= 6u && (d[p + length - 6u] == 0x2B || d[p + length - 6u] == 0x2D) && d[p + length - 3u] == 0x3A) {
            length -= 6u;
            if (!j_hh_mm(d, p + length + 1u, zone)) return false;
            if (d[p + length] == 0x2B) zone = -zone;
        }
        long tod = 0;
        if (length == 13u) { if (!j_hh(d, p + 11u, tod)) return false; }
        else if (length == 16u) { if (!j_hh_mm(d, p + 11u, tod)) return false; }
        else if (length == 19u || (length >= 21u && length <= 29u)) { if (!j_hh_mm_ss(d, p + 11u, tod)) return false; }
        else return false;
        secs += tod + zone;
        if (length <= 19u) { out = secs * unitScale; return true; }
        if (d[p + 19u] != 0x2E) return false;
        uint fl = length - 20u;
        if (fl > fracMax) return false;
        long sub = 0;
        for (uint k = 0u; k < fl; k++) { uchar c = d[p + 20u + k]; if (!j_digit(c)) return false; sub = sub * 10 + (long)(c - 0x30); }
        for (uint k = fl; k < fracMax; k++) sub *= 10;
        out = secs * unitScale + sub;
        return true;
    }

    struct JTime { uint rows; uint hasValidity; long unitScale; uint fracMax; };

    // Parses a utf8 column. Writes values and a validity bitmap (32 rows per thread); counts the valid
    // input rows that fail and keeps the first of them.
    kernel void jt_parse(device const int* offsets [[buffer(0)]], device const uchar* data [[buffer(1)]],
                         device const uchar* validity [[buffer(2)]], constant JTime& P [[buffer(3)]],
                         device long* values [[buffer(4)]], device uint* valid [[buffer(5)]],
                         device atomic_uint* fails [[buffer(6)]], uint w [[thread_position_in_grid]]) {
        uint words = (P.rows + 31u) / 32u;
        uint nf = 0u; uint firstFail = 0xFFFFFFFFu;
        if (w < words) {
            uint vb = 0u;
            uint lo = w * 32u, hi = min(lo + 32u, P.rows);
            for (uint r = lo; r < hi; r++) {
                if (P.hasValidity && !bit_get(validity, r)) { values[r] = 0; continue; }
                long v = 0;
                uint st = (uint)offsets[r], len = (uint)(offsets[r + 1u] - offsets[r]);
                if (j_iso8601(data, st, len, P.unitScale, P.fracMax, v)) { values[r] = v; vb |= 1u << (r - lo); }
                else { values[r] = 0; nf++; if (firstFail == 0xFFFFFFFFu) firstFail = r; }
            }
            valid[w] = vb;
        }
        uint total = simd_sum(nf);
        uint fmin = simd_min(firstFail);
        if (simd_is_first()) {
            if (total) atomic_fetch_add_explicit(&fails[0], total, memory_order_relaxed);
            if (fmin != 0xFFFFFFFFu) atomic_fetch_min_explicit(&fails[1], fmin, memory_order_relaxed);
        }
    }
    """
}
