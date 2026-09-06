import Foundation

/// MSL for the byte-wise splitters in `Kernels/StringSplit.swift`: Arrow `ascii_split_whitespace`
/// and `split_pattern`, both with `max_splits` and `reverse`.
///
/// Splitting is the one string kernel whose output is *two* levels deep — a `list<utf8>` has row
/// offsets over pieces and piece offsets over bytes — so it is a three-pass job with two GPU scans:
///
/// 1. `sp_count` writes the number of pieces in each row. The host scans those into the list offsets.
/// 2. `sp_lens` writes the byte length of every piece, at its own place in the flat child array. The
///    host scans those into the child's value offsets.
/// 3. `sp_write` copies the bytes.
///
/// All three call `sp_walk`, so the three passes cannot disagree about where a piece begins or ends.
///
/// A **separator** is a maximal run of ASCII whitespace (`ascii_split_whitespace`) or one
/// non-overlapping occurrence of a literal byte string (`split_pattern`). Every separator produces a
/// piece boundary, so leading and trailing separators produce empty end pieces and the empty string
/// splits to one empty piece — which is what Arrow does, and what Python's `str.split(sep)` does
/// rather than its no-argument form. `max_splits` keeps the first `max_splits` separators, or the
/// last `max_splits` when `reverse` is set; everything past the budget stays inside the final piece.
enum StringSplitSource {
    static let source = KernelSource.prelude + """

    #define SP_WHITESPACE  0u
    #define SP_LITERAL     1u
    #define SP_UWHITESPACE 2u

    struct SpParams { uint op; uint n1; int maxSplits; uint flags; };

    inline bool sp_ws(uchar b) { return b == 0x20u || (b >= 0x09u && b <= 0x0Du); }

    // Arrow's Unicode whitespace class in full: the Zs, Zl and Zp categories plus U+0009-U+000D,
    // U+001C-U+001F and U+0085. U+200B (zero-width space, category Cf) deliberately is not in it.
    // This is the same set `UnicodeClass.isSpace` computes from Swift's tables, spelled out — Zs has
    // been these eighteen code points for the whole of Unicode's modern history.
    inline bool sp_uws(uint c) {
        if (c < 0x80u) return c == 0x20u || (c >= 0x09u && c <= 0x0Du) || (c >= 0x1Cu && c <= 0x1Fu);
        return c == 0x85u || c == 0xA0u || c == 0x1680u || (c >= 0x2000u && c <= 0x200Au)
            || c == 0x2028u || c == 0x2029u || c == 0x202Fu || c == 0x205Fu || c == 0x3000u;
    }
    // Decodes the UTF-8 sequence at p, returning its byte width and the code point in *cp.
    // A malformed byte decodes as itself over one byte, which can never be whitespace above U+007F.
    inline int sp_decode(device const uchar* d, int p, int end, thread uint* cp) {
        uchar b0 = d[p];
        if (b0 < 0x80u) { *cp = (uint)b0; return 1; }
        if ((b0 & 0xE0u) == 0xC0u && p + 1 < end) {
            *cp = ((uint)(b0 & 0x1Fu) << 6) | (uint)(d[p + 1] & 0x3Fu); return 2;
        }
        if ((b0 & 0xF0u) == 0xE0u && p + 2 < end) {
            *cp = ((uint)(b0 & 0x0Fu) << 12) | ((uint)(d[p + 1] & 0x3Fu) << 6) | (uint)(d[p + 2] & 0x3Fu);
            return 3;
        }
        if ((b0 & 0xF8u) == 0xF0u && p + 3 < end) {
            *cp = ((uint)(b0 & 0x07u) << 18) | ((uint)(d[p + 1] & 0x3Fu) << 12)
                | ((uint)(d[p + 2] & 0x3Fu) << 6) | (uint)(d[p + 3] & 0x3Fu);
            return 4;
        }
        *cp = 0xFFFDu;
        return 1;
    }

    // The length of the separator starting at p, or 0 when none does.
    inline int sp_sep_at(device const uchar* d, int p, int end, uint op,
                         device const uchar* pat, uint plen) {
        if (op == SP_WHITESPACE) {
            if (!sp_ws(d[p])) return 0;
            int q = p;
            while (q < end && sp_ws(d[q])) q++;
            return q - p;
        }
        if (op == SP_UWHITESPACE) {
            uint c;
            int w = sp_decode(d, p, end, &c);
            if (!sp_uws(c)) return 0;
            int q = p + w;
            while (q < end) {
                uint c2;
                int w2 = sp_decode(d, q, end, &c2);
                if (!sp_uws(c2)) break;
                q += w2;
            }
            return q - p;
        }
        if (plen == 0u || p + (int)plen > end) return 0;
        for (uint j = 0; j < plen; j++) if (d[p + j] != pat[j]) return 0;
        return (int)plen;
    }

    inline int sp_count_seps(device const uchar* d, int start, int end, uint op,
                             device const uchar* pat, uint plen) {
        int n = 0, p = start;
        while (p < end) {
            int s = sp_sep_at(d, p, end, op, pat, plen);
            if (s > 0) { n++; p += s; } else p++;
        }
        return n;
    }

    // The one walk every pass shares.
    //   mode 0: return the number of pieces.
    //   mode 1: write each piece's byte length at childLens[base + j]; return the count.
    //   mode 2: copy each piece's bytes to outData[valueOffsets[base + j]]; return the count.
    inline int sp_walk(device const uchar* d, int start, int end, uint op,
                       device const uchar* pat, uint plen, int maxSplits, bool reverse,
                       device int* childLens, device const int* valueOffsets, device uchar* outData,
                       int base, int mode) {
        int totalSeps = (maxSplits < 0) ? 0 : sp_count_seps(d, start, end, op, pat, plen);
        int keep = (maxSplits < 0) ? 0x7FFFFFFF : ((totalSeps <= maxSplits) ? totalSeps : maxSplits);
        int skipBefore = (maxSplits >= 0 && reverse) ? (totalSeps - keep) : 0;
        int skipFrom = (maxSplits < 0) ? 0x7FFFFFFF : (reverse ? totalSeps : keep);

        int idx = 0, pieces = 0, pieceStart = start, p = start;
        while (p < end) {
            int s = sp_sep_at(d, p, end, op, pat, plen);
            if (s > 0) {
                bool use = (idx >= skipBefore && idx < skipFrom);
                idx++;
                if (use) {
                    if (mode == 1) childLens[base + pieces] = p - pieceStart;
                    else if (mode == 2) {
                        int to = valueOffsets[base + pieces];
                        for (int q = pieceStart; q < p; q++) outData[to++] = d[q];
                    }
                    pieces++;
                    pieceStart = p + s;
                }
                p += s;
            } else p++;
        }
        if (mode == 1) childLens[base + pieces] = end - pieceStart;
        else if (mode == 2) {
            int to = valueOffsets[base + pieces];
            for (int q = pieceStart; q < end; q++) outData[to++] = d[q];
        }
        return pieces + 1;
    }

    kernel void sp_count(device const int* offsets [[buffer(0)]], device const uchar* data [[buffer(1)]],
                         device const uchar* validity [[buffer(2)]], device const uint* nPtr [[buffer(3)]],
                         constant SpParams& prm [[buffer(4)]], device const uchar* pat [[buffer(5)]],
                         device int* outCounts [[buffer(6)]], uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if ((prm.flags & 2u) != 0u && !bit_get(validity, i)) { outCounts[i] = 0; return; }
        outCounts[i] = sp_walk(data, offsets[i], offsets[i + 1], prm.op, pat, prm.n1, prm.maxSplits,
                               (prm.flags & 1u) != 0u, outCounts, outCounts, nullptr, 0, 0);
    }

    kernel void sp_lens(device const int* offsets [[buffer(0)]], device const uchar* data [[buffer(1)]],
                        device const uchar* validity [[buffer(2)]], device const uint* nPtr [[buffer(3)]],
                        constant SpParams& prm [[buffer(4)]], device const uchar* pat [[buffer(5)]],
                        device const int* listOffsets [[buffer(6)]], device int* childLens [[buffer(7)]],
                        uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if ((prm.flags & 2u) != 0u && !bit_get(validity, i)) return;
        sp_walk(data, offsets[i], offsets[i + 1], prm.op, pat, prm.n1, prm.maxSplits,
                (prm.flags & 1u) != 0u, childLens, childLens, nullptr, listOffsets[i], 1);
    }

    kernel void sp_write(device const int* offsets [[buffer(0)]], device const uchar* data [[buffer(1)]],
                         device const uchar* validity [[buffer(2)]], device const uint* nPtr [[buffer(3)]],
                         constant SpParams& prm [[buffer(4)]], device const uchar* pat [[buffer(5)]],
                         device const int* listOffsets [[buffer(6)]], device const int* valueOffsets [[buffer(7)]],
                         device uchar* outData [[buffer(8)]], uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if ((prm.flags & 2u) != 0u && !bit_get(validity, i)) return;
        sp_walk(data, offsets[i], offsets[i + 1], prm.op, pat, prm.n1, prm.maxSplits,
                (prm.flags & 1u) != 0u, nullptr, valueOffsets, outData, listOffsets[i], 2);
    }
    """
}
