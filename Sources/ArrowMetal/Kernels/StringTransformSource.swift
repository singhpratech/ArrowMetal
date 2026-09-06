import Foundation

/// MSL for string transforms over Arrow `utf8`: offsets (int32, n+1) + data bytes + validity.
///
/// Output strings have data-dependent byte lengths, so every transform is a two-pass job:
/// `str_tf_len` writes one int32 output length per row, the host scans those into offsets with
/// `exclusiveScanToOffsets`, and `str_tf_write` fills the bytes. Both passes call the same
/// `tf_apply`, once with `write = false` (count) and once with `write = true` (emit), so the two
/// passes cannot disagree about a length.
///
/// ## Case mapping coverage
///
/// `tf_cp_upper` / `tf_cp_lower` implement *simple* (1:1 code point) case mapping over exactly
/// three blocks; every other code point is copied through unchanged:
///
/// * Basic Latin `a`–`z` / `A`–`Z` (U+0041–U+005A, U+0061–U+007A).
/// * Latin-1 Supplement letters U+00C0–U+00DE and U+00E0–U+00FE, skipping the two mathematical
///   signs U+00D7 (×) and U+00F7 (÷), plus U+00FF (ÿ) ↔ U+0178 (Ÿ).
/// * Latin Extended-A U+0100–U+017F in its alternating upper/lower pairs, with the four
///   irregular entries handled explicitly: U+0130 (İ) lowercases to `i`, U+0131 (ı) uppercases
///   to `I`, U+017F (ſ) uppercases to `S`, and U+0138 (ĸ) has no mapping.
///
/// Deliberate deviations from full Unicode case mapping, which needs multi-character expansions:
/// U+00DF (ß) is left as-is (full mapping gives `SS`), U+0149 (ŉ) is left as-is (full mapping
/// gives `ʼN`), and U+00B5 (µ) is left as-is (its simple uppercase U+039C is outside the covered
/// blocks). Nothing above U+017F is ever changed, so Greek, Cyrillic and everything else pass
/// through byte-for-byte.
enum StringTransformSource {
    static let source = KernelSource.prelude + """
    // Transform op codes; must match StringTransform in StringTransforms.swift.
    #define TF_ASCII_UPPER   0u
    #define TF_ASCII_LOWER   1u
    #define TF_UTF8_UPPER    2u
    #define TF_UTF8_LOWER    3u
    #define TF_ASCII_SWAP    4u
    #define TF_ASCII_CAP     5u
    #define TF_TRIM_WS       6u
    #define TF_LTRIM_WS      7u
    #define TF_RTRIM_WS      8u
    #define TF_TRIM_SET      9u
    #define TF_LTRIM_SET    10u
    #define TF_RTRIM_SET    11u
    #define TF_REPLACE      12u
    #define TF_REPEAT       13u
    #define TF_SLICE        14u
    #define TF_PAD_LEFT     15u
    #define TF_PAD_RIGHT    16u
    #define TF_REVERSE      17u

    struct TfParams { uint op; uint n1; uint n2; int p1; int p2; uint flags; };

    inline uchar tf_up_b(uchar b) { return (b >= 0x61u && b <= 0x7Au) ? (uchar)(b - 32u) : b; }
    inline uchar tf_lo_b(uchar b) { return (b >= 0x41u && b <= 0x5Au) ? (uchar)(b + 32u) : b; }

    // Simple (1:1) uppercase over Basic Latin, Latin-1 Supplement and Latin Extended-A. See the
    // Swift doc comment on StringTransformSource for the exact coverage and the deviations.
    inline uint tf_cp_upper(uint c) {
        if (c >= 0x61u && c <= 0x7Au) return c - 32u;
        if (c >= 0xE0u && c <= 0xFEu && c != 0xF7u) return c - 32u;
        if (c == 0xFFu) return 0x178u;                      // small y with diaeresis -> capital
        if (c == 0x131u) return 0x49u;                      // dotless i -> I
        if (c == 0x17Fu) return 0x53u;                      // long s -> S
        if (c >= 0x100u && c <= 0x137u) return c & ~1u;     // even = capital, odd = small
        if (c >= 0x139u && c <= 0x148u) return (c & 1u) ? c : c - 1u;  // odd = capital
        if (c >= 0x14Au && c <= 0x177u) return c & ~1u;
        if (c >= 0x179u && c <= 0x17Eu) return (c & 1u) ? c : c - 1u;
        return c;
    }
    inline uint tf_cp_lower(uint c) {
        if (c >= 0x41u && c <= 0x5Au) return c + 32u;
        if (c >= 0xC0u && c <= 0xDEu && c != 0xD7u) return c + 32u;
        if (c == 0x178u) return 0xFFu;
        if (c == 0x130u) return 0x69u;                      // I with dot above -> i
        if (c >= 0x100u && c <= 0x137u) return c | 1u;
        if (c >= 0x139u && c <= 0x148u) return (c & 1u) ? c + 1u : c;
        if (c >= 0x14Au && c <= 0x177u) return c | 1u;
        if (c >= 0x179u && c <= 0x17Eu) return (c & 1u) ? c + 1u : c;
        return c;
    }
    // Byte length of the UTF-8 sequence starting with b0 (1 for a stray continuation byte).
    inline int tf_adv(uchar b0) {
        if (b0 < 0x80u) return 1;
        if ((b0 & 0xE0u) == 0xC0u) return 2;
        if ((b0 & 0xF0u) == 0xE0u) return 3;
        if ((b0 & 0xF8u) == 0xF0u) return 4;
        return 1;
    }
    inline bool tf_is_ws(uchar b) { return b == 0x20u || (b >= 0x09u && b <= 0x0Du); }
    inline bool tf_in_set(uchar b, device const uchar* set, uint n1, bool useSet) {
        if (!useSet) return tf_is_ws(b);
        for (uint j = 0; j < n1; j++) if (set[j] == b) return true;
        return false;
    }
    // Number of UTF-8 code points (bytes that are not continuation bytes).
    inline int tf_ncp(device const uchar* d, int start, int end) {
        int c = 0;
        for (int p = start; p < end; p++) if ((d[p] & 0xC0) != 0x80) c++;
        return c;
    }

    // The single source of truth for every transform. Returns the number of output bytes, and
    // writes them at out[outPos...] when `write` is true. Called once per pass.
    inline int tf_apply(device const uchar* d, int start, int len,
                        device const uchar* a1, uint n1, device const uchar* a2, uint n2,
                        uint op, int p1, int p2,
                        device uchar* out, int outPos, bool write) {
        int end = start + len;
        int n = 0;
        switch (op) {
        case TF_ASCII_UPPER: case TF_ASCII_LOWER: case TF_ASCII_SWAP: case TF_ASCII_CAP: {
            for (int p = start; p < end; p++) {
                uchar b = d[p], r;
                if (op == TF_ASCII_UPPER) r = tf_up_b(b);
                else if (op == TF_ASCII_LOWER) r = tf_lo_b(b);
                else if (op == TF_ASCII_SWAP) r = (b >= 0x61u && b <= 0x7Au) ? (uchar)(b - 32u)
                                                : ((b >= 0x41u && b <= 0x5Au) ? (uchar)(b + 32u) : b);
                else r = (p == start) ? tf_up_b(b) : tf_lo_b(b);
                if (write) out[outPos + n] = r;
                n++;
            }
            return n;
        }
        case TF_UTF8_UPPER: case TF_UTF8_LOWER: {
            bool up = (op == TF_UTF8_UPPER);
            int p = start;
            while (p < end) {
                uchar b0 = d[p];
                if (b0 < 0x80u) {
                    if (write) out[outPos + n] = up ? tf_up_b(b0) : tf_lo_b(b0);
                    n++; p++;
                } else if ((b0 & 0xE0u) == 0xC0u && p + 1 < end) {
                    uint cp = ((uint)(b0 & 0x1Fu) << 6) | (uint)(d[p + 1] & 0x3Fu);
                    if (cp < 0x80u) {                        // overlong / invalid: pass through
                        if (write) { out[outPos + n] = b0; out[outPos + n + 1] = d[p + 1]; }
                        n += 2;
                    } else {
                        uint m = up ? tf_cp_upper(cp) : tf_cp_lower(cp);
                        if (m < 0x80u) { if (write) out[outPos + n] = (uchar)m; n++; }
                        else {
                            if (write) { out[outPos + n] = (uchar)(0xC0u | (m >> 6));
                                         out[outPos + n + 1] = (uchar)(0x80u | (m & 0x3Fu)); }
                            n += 2;
                        }
                    }
                    p += 2;
                } else {                                     // 3- and 4-byte sequences and stray bytes
                    if (write) out[outPos + n] = b0;
                    n++; p++;
                }
            }
            return n;
        }
        case TF_TRIM_WS: case TF_LTRIM_WS: case TF_RTRIM_WS:
        case TF_TRIM_SET: case TF_LTRIM_SET: case TF_RTRIM_SET: {
            bool useSet = (op >= TF_TRIM_SET);
            bool doL = (op == TF_TRIM_WS || op == TF_LTRIM_WS || op == TF_TRIM_SET || op == TF_LTRIM_SET);
            bool doR = (op == TF_TRIM_WS || op == TF_RTRIM_WS || op == TF_TRIM_SET || op == TF_RTRIM_SET);
            int s = start, e = end;
            if (doL) while (s < e && tf_in_set(d[s], a1, n1, useSet)) s++;
            if (doR) while (e > s && tf_in_set(d[e - 1], a1, n1, useSet)) e--;
            for (int p = s; p < e; p++) { if (write) out[outPos + n] = d[p]; n++; }
            return n;
        }
        case TF_REPLACE: {
            if (n1 == 0u) {                                  // empty pattern: identity
                for (int p = start; p < end; p++) { if (write) out[outPos + n] = d[p]; n++; }
                return n;
            }
            int reps = 0, p = start;
            while (p < end) {
                bool hit = false;
                if ((p1 < 0 || reps < p1) && p + (int)n1 <= end) {
                    hit = true;
                    for (uint j = 0; j < n1; j++) if (d[p + j] != a1[j]) { hit = false; break; }
                }
                if (hit) {
                    for (uint j = 0; j < n2; j++) { if (write) out[outPos + n] = a2[j]; n++; }
                    p += (int)n1; reps++;
                } else { if (write) out[outPos + n] = d[p]; n++; p++; }
            }
            return n;
        }
        case TF_REPEAT: {
            int times = p1 < 0 ? 0 : p1;
            for (int r = 0; r < times; r++)
                for (int p = start; p < end; p++) { if (write) out[outPos + n] = d[p]; n++; }
            return n;
        }
        case TF_SLICE: {
            int ncp = tf_ncp(d, start, end);
            int s = p1 < 0 ? max(ncp + p1, 0) : min(p1, ncp);
            int e = p2 < 0 ? max(ncp + p2, 0) : min(p2, ncp);
            if (e <= s) return 0;
            int b0 = end, b1 = end, cnt = 0;
            for (int p = start; p < end; p++) {
                if ((d[p] & 0xC0) != 0x80) {
                    if (cnt == s) b0 = p;
                    if (cnt == e) { b1 = p; break; }
                    cnt++;
                }
            }
            for (int p = b0; p < b1; p++) { if (write) out[outPos + n] = d[p]; n++; }
            return n;
        }
        case TF_PAD_LEFT: case TF_PAD_RIGHT: {
            int ncp = tf_ncp(d, start, end);
            int pad = (p1 > ncp) ? (p1 - ncp) : 0;
            if (op == TF_PAD_LEFT)
                for (int k = 0; k < pad; k++)
                    for (uint j = 0; j < n1; j++) { if (write) out[outPos + n] = a1[j]; n++; }
            for (int p = start; p < end; p++) { if (write) out[outPos + n] = d[p]; n++; }
            if (op == TF_PAD_RIGHT)
                for (int k = 0; k < pad; k++)
                    for (uint j = 0; j < n1; j++) { if (write) out[outPos + n] = a1[j]; n++; }
            return n;
        }
        case TF_REVERSE: {
            int to = outPos + len, p = start;
            while (p < end) {
                int adv = tf_adv(d[p]);
                if (p + adv > end) adv = 1;
                to -= adv;
                if (write) for (int j = 0; j < adv; j++) out[to + j] = d[p + j];
                p += adv;
            }
            return len;
        }
        default: return 0;
        }
    }

    // Pass 1: output byte length per row. Null rows produce 0 bytes.
    kernel void str_tf_len(device const int* offsets [[buffer(0)]], device const uchar* data [[buffer(1)]],
                           device const uchar* validity [[buffer(2)]], device const uint* nPtr [[buffer(3)]],
                           constant TfParams& prm [[buffer(4)]], device const uchar* a1 [[buffer(5)]],
                           device const uchar* a2 [[buffer(6)]], device int* outLens [[buffer(7)]],
                           device uchar* scratch [[buffer(8)]], uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if ((prm.flags & 1u) != 0u && !bit_get(validity, i)) { outLens[i] = 0; return; }
        int start = offsets[i], len = offsets[i + 1] - start;
        outLens[i] = tf_apply(data, start, len, a1, prm.n1, a2, prm.n2, prm.op, prm.p1, prm.p2, scratch, 0, false);
    }
    // Pass 2: the bytes, at outOffsets[i].
    kernel void str_tf_write(device const int* offsets [[buffer(0)]], device const uchar* data [[buffer(1)]],
                             device const uchar* validity [[buffer(2)]], device const uint* nPtr [[buffer(3)]],
                             constant TfParams& prm [[buffer(4)]], device const uchar* a1 [[buffer(5)]],
                             device const uchar* a2 [[buffer(6)]], device const int* outOffsets [[buffer(7)]],
                             device uchar* outData [[buffer(8)]], uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if ((prm.flags & 1u) != 0u && !bit_get(validity, i)) return;
        int start = offsets[i], len = offsets[i + 1] - start;
        tf_apply(data, start, len, a1, prm.n1, a2, prm.n2, prm.op, prm.p1, prm.p2, outData, outOffsets[i], true);
    }

    // binary_join_element_wise with one separator: a + sep + b, null if either side is null.
    // prm.n1 is the separator length; prm.flags bit 0 = a has validity, bit 1 = b has validity.
    inline bool cat_valid(device const uchar* va, device const uchar* vb, uint flags, uint i) {
        if ((flags & 1u) != 0u && !bit_get(va, i)) return false;
        if ((flags & 2u) != 0u && !bit_get(vb, i)) return false;
        return true;
    }
    kernel void str_cat_len(device const int* oa [[buffer(0)]], device const int* ob [[buffer(1)]],
                            device const uchar* va [[buffer(2)]], device const uchar* vb [[buffer(3)]],
                            device const uint* nPtr [[buffer(4)]], constant TfParams& prm [[buffer(5)]],
                            device int* outLens [[buffer(6)]], uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if (!cat_valid(va, vb, prm.flags, i)) { outLens[i] = 0; return; }
        outLens[i] = (oa[i + 1] - oa[i]) + (int)prm.n1 + (ob[i + 1] - ob[i]);
    }
    kernel void str_cat_write(device const int* oa [[buffer(0)]], device const uchar* da [[buffer(1)]],
                              device const int* ob [[buffer(2)]], device const uchar* db [[buffer(3)]],
                              device const uchar* va [[buffer(4)]], device const uchar* vb [[buffer(5)]],
                              device const uint* nPtr [[buffer(6)]], constant TfParams& prm [[buffer(7)]],
                              device const uchar* sep [[buffer(8)]], device const int* outOffsets [[buffer(9)]],
                              device uchar* outData [[buffer(10)]], uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if (!cat_valid(va, vb, prm.flags, i)) return;
        int to = outOffsets[i];
        for (int p = oa[i]; p < oa[i + 1]; p++) outData[to++] = da[p];
        for (uint j = 0; j < prm.n1; j++) outData[to++] = sep[j];
        for (int p = ob[i]; p < ob[i + 1]; p++) outData[to++] = db[p];
    }

    // count_substring (mode 0) and find_substring (mode 1, byte index or -1).
    kernel void str_tf_search(device const int* offsets [[buffer(0)]], device const uchar* data [[buffer(1)]],
                              device const uint* nPtr [[buffer(2)]], device const uchar* pat [[buffer(3)]],
                              constant uint& plen [[buffer(4)]], constant uint& mode [[buffer(5)]],
                              device int* out [[buffer(6)]], uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        int start = offsets[i], end = offsets[i + 1];
        if (plen == 0u) { out[i] = (mode == 0u) ? tf_ncp(data, start, end) + 1 : 0; return; }
        int found = 0, p = start;
        while (p + (int)plen <= end) {
            bool hit = true;
            for (uint j = 0; j < plen; j++) if (data[p + j] != pat[j]) { hit = false; break; }
            if (hit) {
                if (mode != 0u) { out[i] = p - start; return; }
                found++; p += (int)plen;
            } else p++;
        }
        out[i] = (mode == 0u) ? found : -1;
    }

    // ASCII character-class predicates -> packed boolean bitmap, one 32-bit word per thread.
    // 0 alnum, 1 alpha, 2 digit (decimal), 3 space, 4 upper, 5 lower. Empty strings are false.
    inline bool tf_class(device const uchar* d, int start, int len, uint op) {
        if (len == 0) return false;
        bool anyCased = false;
        for (int p = start; p < start + len; p++) {
            uchar b = d[p];
            bool lo = (b >= 0x61u && b <= 0x7Au), up = (b >= 0x41u && b <= 0x5Au), dg = (b >= 0x30u && b <= 0x39u);
            if (op == 0u) { if (!(lo || up || dg)) return false; }
            else if (op == 1u) { if (!(lo || up)) return false; }
            else if (op == 2u) { if (!dg) return false; }
            else if (op == 3u) { if (!tf_is_ws(b)) return false; }
            else if (op == 4u) { if (lo) return false; if (up) anyCased = true; }
            else { if (up) return false; if (lo) anyCased = true; }
        }
        return (op >= 4u) ? anyCased : true;
    }
    kernel void str_tf_class(device const int* offsets [[buffer(0)]], device const uchar* data [[buffer(1)]],
                             device const uint* nPtr [[buffer(2)]], constant uint& op [[buffer(3)]],
                             device uint* out [[buffer(4)]], uint w [[thread_position_in_grid]]) {
        uint n = *nPtr, base = w * 32u;
        if (base >= n) return;
        uint limit = min(32u, n - base), bits = 0u;
        for (uint j = 0; j < limit; j++) {
            uint i = base + j;
            if (tf_class(data, offsets[i], offsets[i + 1] - offsets[i], op)) bits |= (1u << j);
        }
        out[w] = bits;
    }
    """
}
