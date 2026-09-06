import Foundation

/// MSL for the two string transforms that are not in `StringTransformSource`: Arrow's
/// `utf8_swapcase` and `utf8_zero_fill`.
///
/// Same two-pass shape as the other string transforms — `sx_apply` returns the output byte length
/// when `write` is false and emits the bytes when it is true, so a length and the bytes that fill it
/// can never disagree — but with its own kernels and its own op numbering, so this file and
/// `StringTransformSource` stay independent.
enum StringExtraSource {
    static let source = KernelSource.prelude + """

    #define SX_SWAPCASE  0u
    #define SX_ZERO_FILL 1u

    struct SxParams { uint op; uint n1; int p1; uint flags; };

    // Byte length of the UTF-8 sequence starting with b0 (1 for a stray continuation byte).
    inline int sx_adv(uchar b0) {
        if (b0 < 0x80u) return 1;
        if ((b0 & 0xE0u) == 0xC0u) return 2;
        if ((b0 & 0xF0u) == 0xE0u) return 3;
        if ((b0 & 0xF8u) == 0xF0u) return 4;
        return 1;
    }
    // Decodes the code point at `p`; `adv` comes back as its byte length. A malformed lead byte
    // decodes as the raw byte, which round-trips unchanged.
    inline uint sx_decode(device const uchar* d, int p, thread int& adv) {
        uchar b0 = d[p];
        adv = sx_adv(b0);
        if (adv == 1) return (uint)b0;
        if (adv == 2) return (((uint)b0 & 0x1Fu) << 6) | ((uint)d[p + 1] & 0x3Fu);
        if (adv == 3) return (((uint)b0 & 0x0Fu) << 12) | (((uint)d[p + 1] & 0x3Fu) << 6) | ((uint)d[p + 2] & 0x3Fu);
        return (((uint)b0 & 0x07u) << 18) | (((uint)d[p + 1] & 0x3Fu) << 12)
             | (((uint)d[p + 2] & 0x3Fu) << 6) | ((uint)d[p + 3] & 0x3Fu);
    }
    inline int sx_enc_len(uint c) {
        if (c < 0x80u) return 1;
        if (c < 0x800u) return 2;
        if (c < 0x10000u) return 3;
        return 4;
    }
    inline void sx_encode(uint c, device uchar* out, int pos) {
        if (c < 0x80u) { out[pos] = (uchar)c; return; }
        if (c < 0x800u) {
            out[pos] = (uchar)(0xC0u | (c >> 6)); out[pos + 1] = (uchar)(0x80u | (c & 0x3Fu)); return;
        }
        if (c < 0x10000u) {
            out[pos] = (uchar)(0xE0u | (c >> 12));
            out[pos + 1] = (uchar)(0x80u | ((c >> 6) & 0x3Fu));
            out[pos + 2] = (uchar)(0x80u | (c & 0x3Fu));
            return;
        }
        out[pos] = (uchar)(0xF0u | (c >> 18));
        out[pos + 1] = (uchar)(0x80u | ((c >> 12) & 0x3Fu));
        out[pos + 2] = (uchar)(0x80u | ((c >> 6) & 0x3Fu));
        out[pos + 3] = (uchar)(0x80u | (c & 0x3Fu));
    }
    // Simple (1:1) case mapping over Basic Latin, Latin-1 Supplement and Latin Extended-A. Identical
    // to the tables behind utf8_upper / utf8_lower; the Swift doc comment lists the deviations.
    inline uint sx_cp_upper(uint c) {
        if (c >= 0x61u && c <= 0x7Au) return c - 32u;
        if (c >= 0xE0u && c <= 0xFEu && c != 0xF7u) return c - 32u;
        if (c == 0xFFu) return 0x178u;
        if (c == 0x131u) return 0x49u;
        if (c == 0x17Fu) return 0x53u;
        if (c >= 0x100u && c <= 0x137u) return c & ~1u;
        if (c >= 0x139u && c <= 0x148u) return (c & 1u) ? c : c - 1u;
        if (c >= 0x14Au && c <= 0x177u) return c & ~1u;
        if (c >= 0x179u && c <= 0x17Eu) return (c & 1u) ? c : c - 1u;
        return c;
    }
    inline uint sx_cp_lower(uint c) {
        if (c >= 0x41u && c <= 0x5Au) return c + 32u;
        if (c >= 0xC0u && c <= 0xDEu && c != 0xD7u) return c + 32u;
        if (c == 0x178u) return 0xFFu;
        if (c == 0x130u) return 0x69u;
        if (c >= 0x100u && c <= 0x137u) return c | 1u;
        if (c >= 0x139u && c <= 0x148u) return (c & 1u) ? c + 1u : c;
        if (c >= 0x14Au && c <= 0x177u) return c | 1u;
        if (c >= 0x179u && c <= 0x17Eu) return (c & 1u) ? c + 1u : c;
        return c;
    }
    // Number of UTF-8 code points (bytes that are not continuation bytes).
    inline int sx_ncp(device const uchar* d, int start, int end) {
        int c = 0;
        for (int p = start; p < end; p++) if ((d[p] & 0xC0) != 0x80) c++;
        return c;
    }

    /// Returns the output byte count, and writes the bytes at out[outPos...] when `write` is true.
    inline int sx_apply(device const uchar* d, int start, int len,
                        device const uchar* a1, uint n1, uint op, int p1,
                        device uchar* out, int outPos, bool write) {
        int end = start + len;
        int n = 0;
        if (op == SX_SWAPCASE) {
            int p = start;
            while (p < end) {
                int adv = 1;
                uint c = sx_decode(d, p, adv);
                // A cased-up code point maps down, everything else maps up; both are the identity
                // for uncased characters, so digits and punctuation pass through.
                uint lo = sx_cp_lower(c);
                uint o = (lo != c) ? lo : sx_cp_upper(c);
                int w = sx_enc_len(o);
                if (write) sx_encode(o, out, outPos + n);
                n += w;
                p += adv;
            }
            return n;
        }
        // SX_ZERO_FILL: left-pad to p1 code points with the pad character, after a leading + or -.
        int have = sx_ncp(d, start, end);
        int need = p1 - have;
        if (need < 0) need = 0;
        int body = start;
        if (need > 0 && len > 0 && (d[start] == 0x2Bu || d[start] == 0x2Du)) {
            if (write) out[outPos] = d[start];
            n += 1;
            body = start + 1;
        }
        for (int k = 0; k < need; k++) {
            if (write) for (uint j = 0; j < n1; j++) out[outPos + n + (int)j] = a1[j];
            n += (int)n1;
        }
        for (int p = body; p < end; p++) {
            if (write) out[outPos + n] = d[p];
            n += 1;
        }
        return n;
    }

    kernel void sx_len(device const int* offsets [[buffer(0)]], device const uchar* data [[buffer(1)]],
                       device const uchar* validity [[buffer(2)]], device const uint* nPtr [[buffer(3)]],
                       constant SxParams& prm [[buffer(4)]], device const uchar* a1 [[buffer(5)]],
                       device int* outLens [[buffer(6)]], device uchar* scratch [[buffer(7)]],
                       uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if ((prm.flags & 1u) && !bit_get(validity, i)) { outLens[i] = 0; return; }
        int start = offsets[i], len = offsets[i + 1] - start;
        outLens[i] = sx_apply(data, start, len, a1, prm.n1, prm.op, prm.p1, scratch, 0, false);
    }
    kernel void sx_write(device const int* offsets [[buffer(0)]], device const uchar* data [[buffer(1)]],
                         device const uchar* validity [[buffer(2)]], device const uint* nPtr [[buffer(3)]],
                         constant SxParams& prm [[buffer(4)]], device const uchar* a1 [[buffer(5)]],
                         device const int* outOffsets [[buffer(6)]], device uchar* outData [[buffer(7)]],
                         uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if ((prm.flags & 1u) && !bit_get(validity, i)) return;
        int start = offsets[i], len = offsets[i + 1] - start;
        sx_apply(data, start, len, a1, prm.n1, prm.op, prm.p1, outData, outOffsets[i], true);
    }
    """
}
