import Foundation

/// MSL for the per-row GPU half of the Unicode case transforms and trims (`Kernels/StringUnicode.swift`).
///
/// The kernels here answer a row **only when they can answer it exactly**, and say so: `su_apply`
/// returns -1 for a row it declines, the length kernel records that in a per-row flag byte, and the
/// host recomputes those rows with Swift's Unicode tables. A row is declined when
///
/// * a case transform meets a code point above U+017F, or a byte sequence that is not clean 1- or
///   2-byte UTF-8 — everything Greek, Cyrillic, CJK, emoji or combining goes to the host; or
/// * a trim whose character set has a non-ASCII member meets a row with a byte ≥ 0x80.
///
/// ## The case table, U+0000–U+017F
///
/// Exact, not approximate: `su_upper` / `su_lower` are Unicode's **simple** (1:1) mappings over Basic
/// Latin, the Latin-1 Supplement and Latin Extended-A, matching utf8proc — and therefore pyarrow —
/// on every code point in that range, including the five irregular entries:
///
/// | code point | upper | lower | note |
/// |---|---|---|---|
/// | U+00B5 µ | U+039C Μ | U+00B5 | the simple uppercase leaves the block |
/// | U+00DF ß | U+1E9E ẞ | U+00DF | two bytes become three |
/// | U+00FF ÿ | U+0178 Ÿ | U+00FF | |
/// | U+0130 İ | U+0130 | U+0069 i | two bytes become one |
/// | U+0131 ı | U+0049 I | U+0131 | two bytes become one |
/// | U+0138 ĸ | U+0138 | U+0138 | no case at all, and it sits inside an alternating pair run |
/// | U+017F ſ | U+0053 S | U+017F | |
/// | U+0149 ŉ | U+0149 | U+0149 | full uppercase `ʼN` is two characters, so the simple one is identity |
///
/// `su_swap` is Arrow's `utf8_swapcase`: lower-case it when it has a lowercase mapping, otherwise
/// upper-case it when it has an uppercase one, otherwise leave it. Over this range that is exactly
/// pyarrow's answer; above it, where a titlecase letter is both upper and lower and must stay put,
/// the host decides instead.
///
/// `su_cased` is the set of cased code points in the range — U+0041–U+005A, U+0061–U+007A, U+00B5,
/// U+00C0–U+00D6, U+00D8–U+00F6 and U+00F8–U+017F — which is what `utf8_title` splits words on.
enum StringUnicodeSource {
    static let source = KernelSource.prelude + """

    // Op codes; must match UnicodeTransform in StringUnicode.swift.
    #define SU_UPPER      0u
    #define SU_LOWER      1u
    #define SU_SWAP       2u
    #define SU_CAPITALIZE 3u
    #define SU_TITLE      4u
    #define SU_TRIM       5u
    #define SU_LTRIM      6u
    #define SU_RTRIM      7u

    struct SuParams { uint op; uint n1; int p1; int p2; int p3; uint flags; };

    inline uint su_upper(uint c) {
        if (c >= 0x61u && c <= 0x7Au) return c - 32u;
        if (c == 0xB5u) return 0x39Cu;
        if (c == 0xDFu) return 0x1E9Eu;
        if (c >= 0xE0u && c <= 0xFEu && c != 0xF7u) return c - 32u;
        if (c == 0xFFu) return 0x178u;
        if (c == 0x131u) return 0x49u;
        if (c == 0x17Fu) return 0x53u;
        if (c == 0x138u) return c;                          // kra has no case
        if (c >= 0x100u && c <= 0x137u) return c & ~1u;     // even = capital, odd = small
        if (c >= 0x139u && c <= 0x148u) return (c & 1u) ? c : c - 1u;
        if (c >= 0x14Au && c <= 0x177u) return c & ~1u;
        if (c >= 0x179u && c <= 0x17Eu) return (c & 1u) ? c : c - 1u;
        return c;
    }
    inline uint su_lower(uint c) {
        if (c >= 0x41u && c <= 0x5Au) return c + 32u;
        if (c >= 0xC0u && c <= 0xDEu && c != 0xD7u) return c + 32u;
        if (c == 0x178u) return 0xFFu;                      // capital Y with diaeresis -> small
        if (c == 0x130u) return 0x69u;
        if (c == 0x138u) return c;
        if (c >= 0x100u && c <= 0x137u) return c | 1u;
        if (c >= 0x139u && c <= 0x148u) return (c & 1u) ? c + 1u : c;
        if (c >= 0x14Au && c <= 0x177u) return c | 1u;
        if (c >= 0x179u && c <= 0x17Eu) return (c & 1u) ? c + 1u : c;
        return c;
    }
    inline uint su_swap(uint c) {
        uint l = su_lower(c);
        if (l != c) return l;
        uint u = su_upper(c);
        return (u != c) ? u : c;
    }
    inline bool su_cased(uint c) {
        return (c >= 0x41u && c <= 0x5Au) || (c >= 0x61u && c <= 0x7Au) || c == 0xB5u
            || (c >= 0xC0u && c <= 0xD6u) || (c >= 0xD8u && c <= 0xF6u) || (c >= 0xF8u && c <= 0x17Fu);
    }
    inline bool su_in_set(uchar b, device const uchar* set, uint n1) {
        for (uint j = 0; j < n1; j++) if (set[j] == b) return true;
        return false;
    }

    // Returns -1 when the host has to decide the row; otherwise the output byte count, written at
    // out[outPos...] when `write` is true.
    inline int su_apply(device const uchar* d, int start, int len,
                        device const uchar* a1, uint n1, uint op, int mode,
                        device uchar* out, int outPos, bool write) {
        int end = start + len;
        if (op >= SU_TRIM) {
            if (mode != 0) { for (int p = start; p < end; p++) if (d[p] >= 0x80u) return -1; }
            bool doL = (op == SU_TRIM || op == SU_LTRIM);
            bool doR = (op == SU_TRIM || op == SU_RTRIM);
            int s = start, e = end;
            if (doL) while (s < e && su_in_set(d[s], a1, n1)) s++;
            if (doR) while (e > s && su_in_set(d[e - 1], a1, n1)) e--;
            int n = 0;
            for (int p = s; p < e; p++) { if (write) out[outPos + n] = d[p]; n++; }
            return n;
        }
        int n = 0, idx = 0, p = start;
        bool boundary = true;
        while (p < end) {
            uchar b0 = d[p];
            uint cp; int w;
            if (b0 < 0x80u) { cp = (uint)b0; w = 1; }
            else if ((b0 & 0xE0u) == 0xC0u && p + 1 < end && (d[p + 1] & 0xC0u) == 0x80u) {
                cp = ((uint)(b0 & 0x1Fu) << 6) | (uint)(d[p + 1] & 0x3Fu);
                w = 2;
                if (cp < 0x80u || cp > 0x17Fu) return -1;   // overlong, or past the covered blocks
            } else return -1;                               // 3-/4-byte or malformed: the host decides
            uint m = cp;
            if (op == SU_UPPER) m = su_upper(cp);
            else if (op == SU_LOWER) m = su_lower(cp);
            else if (op == SU_SWAP) m = su_swap(cp);
            else if (op == SU_CAPITALIZE) m = (idx == 0) ? su_upper(cp) : su_lower(cp);
            else {                                          // SU_TITLE
                if (su_cased(cp)) { m = boundary ? su_upper(cp) : su_lower(cp); boundary = false; }
                else { m = cp; boundary = true; }
            }
            if (m < 0x80u) { if (write) out[outPos + n] = (uchar)m; n += 1; }
            else if (m < 0x800u) {
                if (write) { out[outPos + n] = (uchar)(0xC0u | (m >> 6));
                             out[outPos + n + 1] = (uchar)(0x80u | (m & 0x3Fu)); }
                n += 2;
            } else {
                if (write) { out[outPos + n] = (uchar)(0xE0u | (m >> 12));
                             out[outPos + n + 1] = (uchar)(0x80u | ((m >> 6) & 0x3Fu));
                             out[outPos + n + 2] = (uchar)(0x80u | (m & 0x3Fu)); }
                n += 3;
            }
            idx++;
            p += w;
        }
        return n;
    }

    // Pass 1: the output byte length, plus a flag byte naming the rows the host must redo.
    kernel void su_tf_len(device const int* offsets [[buffer(0)]], device const uchar* data [[buffer(1)]],
                          device const uchar* validity [[buffer(2)]], device const uint* nPtr [[buffer(3)]],
                          constant SuParams& prm [[buffer(4)]], device const uchar* a1 [[buffer(5)]],
                          device int* outLens [[buffer(6)]], device uchar* hostRows [[buffer(7)]],
                          device uchar* scratch [[buffer(8)]], device atomic_uint* declined [[buffer(9)]],
                          uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        hostRows[i] = 0;
        if ((prm.flags & 1u) != 0u && !bit_get(validity, i)) { outLens[i] = 0; return; }
        int start = offsets[i], len = offsets[i + 1] - start;
        int r = su_apply(data, start, len, a1, prm.n1, prm.op, prm.p1, scratch, 0, false);
        if (r < 0) { outLens[i] = 0; hostRows[i] = 1; atomic_store_explicit(declined, 1u, memory_order_relaxed); return; }
        outLens[i] = r;
    }
    // Pass 2: the bytes, for the rows the GPU claimed in pass 1.
    kernel void su_tf_write(device const int* offsets [[buffer(0)]], device const uchar* data [[buffer(1)]],
                            device const uchar* validity [[buffer(2)]], device const uint* nPtr [[buffer(3)]],
                            constant SuParams& prm [[buffer(4)]], device const uchar* a1 [[buffer(5)]],
                            device const uchar* hostRows [[buffer(6)]], device const int* outOffsets [[buffer(7)]],
                            device uchar* outData [[buffer(8)]], uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if (hostRows[i] != 0) return;
        if ((prm.flags & 1u) != 0u && !bit_get(validity, i)) return;
        int start = offsets[i], len = offsets[i + 1] - start;
        su_apply(data, start, len, a1, prm.n1, prm.op, prm.p1, outData, outOffsets[i], true);
    }
    """
}
