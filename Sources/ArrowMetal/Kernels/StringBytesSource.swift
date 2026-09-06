import Foundation

/// MSL for the **byte-indexed** string and binary functions in `Kernels/StringBytes.swift`.
///
/// Arrow draws a hard line between the `binary_*` / `ascii_*` names, which index and count **bytes**,
/// and the `utf8_*` names, which index and count **code points**. `Kernels/StringTransformSource.swift`
/// only ever counts code points, so this file supplies the byte-counting halves of those pairs —
/// `binary_slice`, `binary_reverse` / `ascii_reverse`, `ascii_lpad` / `ascii_rpad` / `ascii_center` —
/// plus the one slicing routine that understands Arrow's `step`, in both a byte and a code point form.
///
/// Everything here is the same two-pass job the other transforms use: `sb_tf_len` writes one output
/// byte length per row, the host scans those into the Arrow int32 offsets buffer, and `sb_tf_write`
/// fills the bytes. Both passes call `sb_apply`, so a length and the bytes that fill it cannot disagree.
///
/// ## Slicing
///
/// `sb_slice` is Python's slice, evaluated in 64-bit arithmetic so that the sentinel bounds Arrow's
/// `SliceOptions` uses (`INT64_MAX` for "no stop", `-INT64_MAX` for "no stop, counting backwards")
/// clamp instead of overflowing. For `step > 0` the units are emitted front to back; for `step < 0`
/// the row is still walked forwards — a UTF-8 sequence can only be decoded that way — but each
/// selected unit is written at the *end* of the output and the write head moves backwards, which
/// reverses the order exactly.
///
/// `sb_cat_len` / `sb_cat_write` are one step of the `binary_join_element_wise` fold: two columns and
/// a scalar separator, with Arrow's three `null_handling` modes. Folding this over N columns from the
/// left produces the N-column answer with N-1 separators; `skip` keeps "nothing joined yet" as a null
/// accumulator, so a row that skips its way to the end joins as the empty string rather than picking
/// up stray separators.
enum StringBytesSource {
    static let source = KernelSource.prelude + """

    // Transform op codes; must match ByteTransform in StringBytes.swift.
    #define SB_SLICE_BYTES   0u
    #define SB_SLICE_CP      1u
    #define SB_REVERSE_BYTES 2u
    #define SB_LPAD_BYTES    3u
    #define SB_RPAD_BYTES    4u
    #define SB_CENTER_BYTES  5u

    struct SbParams { uint op; uint n1; int p1; int p2; int p3; uint flags; };

    // Byte length of the UTF-8 sequence starting at p (1 for a stray continuation byte or a
    // sequence that runs off the end of the row).
    inline int sb_adv(device const uchar* d, int p, int end) {
        uchar b0 = d[p];
        int w = 1;
        if (b0 >= 0xF0u) w = 4; else if (b0 >= 0xE0u) w = 3; else if (b0 >= 0xC0u) w = 2;
        if (p + w > end) w = 1;
        return w;
    }
    inline int sb_ncp(device const uchar* d, int start, int end) {
        int c = 0;
        for (int p = start; p < end; p++) if ((d[p] & 0xC0) != 0x80) c++;
        return c;
    }

    // Python's slice over `units` units, in 64-bit arithmetic so the INT64 sentinels clamp.
    // Returns the number of output bytes and, when `write`, emits them; `outLen` is the row's
    // already-known output length, needed only to fill backwards for a negative step.
    inline int sb_slice(device const uchar* d, int start, int len, bool cp,
                        long p1, long p2, long p3,
                        device uchar* out, int outPos, int outLen, bool write) {
        int end = start + len;
        long units = cp ? (long)sb_ncp(d, start, end) : (long)len;
        long st = (p3 == 0) ? 1 : p3;
        long b, e;
        if (st > 0) {
            b = (p1 < 0) ? ((units + p1 < 0) ? 0 : units + p1) : ((p1 > units) ? units : p1);
            e = (p2 < 0) ? ((units + p2 < 0) ? 0 : units + p2) : ((p2 > units) ? units : p2);
        } else {
            b = (p1 < 0) ? (units + p1) : ((p1 > units - 1) ? units - 1 : p1);
            if (b < -1) b = -1;
            e = (p2 < 0) ? (units + p2) : ((p2 > units - 1) ? units - 1 : p2);
            if (e < -1) e = -1;
        }
        int n = 0, wpos = outPos + outLen, p = start;
        long i = 0;
        while (p < end) {
            int w = cp ? sb_adv(d, p, end) : 1;
            bool sel;
            if (st > 0) sel = (i >= b && i < e && ((i - b) % st) == 0);
            else        sel = (i <= b && i > e && ((b - i) % (-st)) == 0);
            if (sel) {
                if (write) {
                    if (st > 0) { for (int j = 0; j < w; j++) out[outPos + n + j] = d[p + j]; }
                    else { wpos -= w; for (int j = 0; j < w; j++) out[wpos + j] = d[p + j]; }
                }
                n += w;
            }
            i++;
            p += w;
        }
        return n;
    }

    inline int sb_apply(device const uchar* d, int start, int len,
                        device const uchar* a1, uint n1, uint op, int p1, int p2, int p3,
                        device uchar* out, int outPos, int outLen, bool write) {
        int end = start + len;
        switch (op) {
        case SB_SLICE_BYTES: case SB_SLICE_CP:
            return sb_slice(d, start, len, op == SB_SLICE_CP, (long)p1, (long)p2, (long)p3,
                            out, outPos, outLen, write);
        case SB_REVERSE_BYTES: {
            if (write) for (int j = 0; j < len; j++) out[outPos + j] = d[end - 1 - j];
            return len;
        }
        case SB_LPAD_BYTES: case SB_RPAD_BYTES: case SB_CENTER_BYTES: {
            // Arrow's ascii_* padding counts BYTES, and the pad character is one byte wide.
            int pad = (p1 > len) ? (p1 - len) : 0;
            int left = 0, right = 0;
            if (op == SB_LPAD_BYTES) left = pad;
            else if (op == SB_RPAD_BYTES) right = pad;
            else { left = pad / 2; right = pad - left; }   // the odd character goes on the right
            int n = 0;
            for (int k = 0; k < left; k++)
                for (uint j = 0; j < n1; j++) { if (write) out[outPos + n] = a1[j]; n++; }
            for (int p = start; p < end; p++) { if (write) out[outPos + n] = d[p]; n++; }
            for (int k = 0; k < right; k++)
                for (uint j = 0; j < n1; j++) { if (write) out[outPos + n] = a1[j]; n++; }
            return n;
        }
        default: return 0;
        }
    }

    kernel void sb_tf_len(device const int* offsets [[buffer(0)]], device const uchar* data [[buffer(1)]],
                          device const uchar* validity [[buffer(2)]], device const uint* nPtr [[buffer(3)]],
                          constant SbParams& prm [[buffer(4)]], device const uchar* a1 [[buffer(5)]],
                          device int* outLens [[buffer(6)]], device uchar* scratch [[buffer(7)]],
                          uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if ((prm.flags & 1u) != 0u && !bit_get(validity, i)) { outLens[i] = 0; return; }
        int start = offsets[i], len = offsets[i + 1] - start;
        outLens[i] = sb_apply(data, start, len, a1, prm.n1, prm.op, prm.p1, prm.p2, prm.p3, scratch, 0, 0, false);
    }
    kernel void sb_tf_write(device const int* offsets [[buffer(0)]], device const uchar* data [[buffer(1)]],
                            device const uchar* validity [[buffer(2)]], device const uint* nPtr [[buffer(3)]],
                            constant SbParams& prm [[buffer(4)]], device const uchar* a1 [[buffer(5)]],
                            device const int* outOffsets [[buffer(6)]], device uchar* outData [[buffer(7)]],
                            uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if ((prm.flags & 1u) != 0u && !bit_get(validity, i)) return;
        int start = offsets[i], len = offsets[i + 1] - start;
        int outPos = outOffsets[i], outLen = outOffsets[i + 1] - outPos;
        sb_apply(data, start, len, a1, prm.n1, prm.op, prm.p1, prm.p2, prm.p3, outData, outPos, outLen, true);
    }

    // -----------------------------------------------------------------------------------------
    // One step of the binary_join_element_wise fold: a[i] + sep + b[i].
    //
    // prm.p1 is Arrow's null_handling (0 emit_null, 1 skip, 2 replace); prm.n1 is the separator
    // length and prm.n2 (carried in p2) the null_replacement length. prm.flags bit 0 = a has a
    // validity bitmap, bit 1 = b has one.
    inline bool sb_valid(device const uchar* v, uint flags, uint bit, uint i) {
        return ((flags & bit) == 0u) || bit_get(v, i);
    }
    kernel void sb_cat_len(device const int* oa [[buffer(0)]], device const int* ob [[buffer(1)]],
                           device const uchar* va [[buffer(2)]], device const uchar* vb [[buffer(3)]],
                           device const uint* nPtr [[buffer(4)]], constant SbParams& prm [[buffer(5)]],
                           device int* outLens [[buffer(6)]], device uchar* outValid [[buffer(7)]],
                           uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        bool av = sb_valid(va, prm.flags, 1u, i), bv = sb_valid(vb, prm.flags, 2u, i);
        int la = oa[i + 1] - oa[i], lb = ob[i + 1] - ob[i];
        int sep = (int)prm.n1, rep = prm.p2;
        if (prm.p1 == 0) {                                   // emit_null
            if (!av || !bv) { outLens[i] = 0; outValid[i] = 0; return; }
            outLens[i] = la + sep + lb; outValid[i] = 1; return;
        }
        if (prm.p1 == 1) {                                   // skip: a null side contributes nothing
            if (!av && !bv) { outLens[i] = 0; outValid[i] = 0; return; }
            if (!av) { outLens[i] = lb; outValid[i] = 1; return; }
            if (!bv) { outLens[i] = la; outValid[i] = 1; return; }
            outLens[i] = la + sep + lb; outValid[i] = 1; return;
        }
        outLens[i] = (av ? la : rep) + sep + (bv ? lb : rep);  // replace
        outValid[i] = 1;
    }
    kernel void sb_cat_write(device const int* oa [[buffer(0)]], device const uchar* da [[buffer(1)]],
                             device const int* ob [[buffer(2)]], device const uchar* db [[buffer(3)]],
                             device const uchar* va [[buffer(4)]], device const uchar* vb [[buffer(5)]],
                             device const uint* nPtr [[buffer(6)]], constant SbParams& prm [[buffer(7)]],
                             device const uchar* sep [[buffer(8)]], device const uchar* repl [[buffer(9)]],
                             device const int* outOffsets [[buffer(10)]], device uchar* outData [[buffer(11)]],
                             uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        bool av = sb_valid(va, prm.flags, 1u, i), bv = sb_valid(vb, prm.flags, 2u, i);
        if (prm.p1 == 0 && (!av || !bv)) return;             // emit_null: the row is null
        if (prm.p1 == 1 && !av && !bv) return;               // skip: nothing joined at all
        int to = outOffsets[i];
        if (prm.p1 == 1 && !(av && bv)) {                    // skip with exactly one live side
            if (av) { for (int p = oa[i]; p < oa[i + 1]; p++) outData[to++] = da[p]; }
            else    { for (int p = ob[i]; p < ob[i + 1]; p++) outData[to++] = db[p]; }
            return;
        }
        // Both sides live (emit_null / skip), or `replace` filling a null side with the replacement.
        if (av) { for (int p = oa[i]; p < oa[i + 1]; p++) outData[to++] = da[p]; }
        else    { for (int j = 0; j < prm.p2; j++) outData[to++] = repl[j]; }
        for (uint j = 0; j < prm.n1; j++) outData[to++] = sep[j];
        if (bv) { for (int p = ob[i]; p < ob[i + 1]; p++) outData[to++] = db[p]; }
        else    { for (int j = 0; j < prm.p2; j++) outData[to++] = repl[j]; }
    }
    """
}
