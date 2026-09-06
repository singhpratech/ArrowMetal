import Foundation

/// MSL for the string functions in `Kernels/StringExtra.swift` and `Kernels/StringContainment.swift`.
///
/// Three independent families live here, all over Arrow `utf8` (int32 offsets + data bytes):
///
/// * **Predicates** (`sx_pred`) — one 32-bit bitmap word per thread. Every thread also reports, in a
///   second bitmap, which of its rows contain a byte ≥ 0x80. The `utf8_is_*` family is exact on the
///   rows whose bit is clear (an ASCII-only string is classified identically by the byte rules below
///   and by the Unicode tables), so the host only has to re-do the rows the second bitmap marks.
/// * **Transforms** (`sx_tf_len` / `sx_tf_write`) — the same two-pass shape `StringTransformSource`
///   uses: one kernel writes an output byte length per row, the host scans those into the Arrow
///   offsets buffer, and a second kernel fills the bytes. Both passes call `sx_apply`, so a length and
///   the bytes that fill it cannot disagree.
/// * **A string hash table** (`sx_hash_insert`, `sx_is_in`, `sx_index_in`) — open addressing with
///   linear probing over the 64-bit key `StringDictionary` already builds. The table stores **row
///   indices**, never hashes, and every probe confirms a hit by comparing the full bytes, so a hash
///   collision costs one extra probe and can never produce a wrong answer.
///
/// `sx_join_len` / `sx_join_write` are the two-pass form of Arrow `binary_join` over a `list<utf8>`.
enum StringExtraSource {
    static let source = KernelSource.prelude + """

    // ---------------------------------------------------------------------------------------------
    // Character-class predicates. Op codes must match StringPredicate in StringExtra.swift.
    #define SXP_ASCII_PRINTABLE  0u
    #define SXP_ASCII_TITLE      1u
    #define SXP_STRING_IS_ASCII  2u
    #define SXP_U_ALNUM          3u
    #define SXP_U_ALPHA          4u
    #define SXP_U_DECIMAL        5u
    #define SXP_U_DIGIT          6u
    #define SXP_U_LOWER          7u
    #define SXP_U_NUMERIC        8u
    #define SXP_U_PRINTABLE      9u
    #define SXP_U_SPACE         10u
    #define SXP_U_TITLE         11u
    #define SXP_U_UPPER         12u

    inline bool sx_low(uchar b)  { return b >= 0x61u && b <= 0x7Au; }
    inline bool sx_up(uchar b)   { return b >= 0x41u && b <= 0x5Au; }
    inline bool sx_dig(uchar b)  { return b >= 0x30u && b <= 0x39u; }
    inline bool sx_alpha(uchar b){ return sx_low(b) || sx_up(b); }
    // Arrow's Unicode whitespace restricted to ASCII: \\t \\n \\v \\f \\r, the four information
    // separators 0x1C-0x1F, and the space. The ascii_is_space kernel deliberately omits 0x1C-0x1F.
    inline bool sx_uspace(uchar b) { return b == 0x20u || (b >= 0x09u && b <= 0x0Du) || (b >= 0x1Cu && b <= 0x1Fu); }

    // Title case, byte-wise: every word starts with an upper-case letter and continues in lower case,
    // where a word is a maximal run of ASCII letters. At least one letter is required.
    inline bool sx_is_title_ascii(device const uchar* d, int start, int end) {
        bool anyCased = false, prevCased = false;
        for (int p = start; p < end; p++) {
            uchar b = d[p];
            if (sx_up(b)) { if (prevCased) return false; prevCased = true; anyCased = true; }
            else if (sx_low(b)) { if (!prevCased) return false; prevCased = true; anyCased = true; }
            else prevCased = false;
        }
        return anyCased;
    }

    /// The predicate for a row that is known to hold only bytes < 0x80.
    inline bool sx_pred_ascii(device const uchar* d, int start, int end, uint op) {
        int len = end - start;
        switch (op) {
        case SXP_STRING_IS_ASCII: return true;                       // caller already checked
        case SXP_ASCII_PRINTABLE: case SXP_U_PRINTABLE: {
            for (int p = start; p < end; p++) if (d[p] < 0x20u || d[p] > 0x7Eu) return false;
            return true;                                             // the empty string is printable
        }
        case SXP_ASCII_TITLE: case SXP_U_TITLE: return sx_is_title_ascii(d, start, end);
        case SXP_U_LOWER: case SXP_U_UPPER: {
            bool anyCased = false;
            for (int p = start; p < end; p++) {
                uchar b = d[p];
                if (op == SXP_U_UPPER) { if (sx_low(b)) return false; if (sx_up(b)) anyCased = true; }
                else { if (sx_up(b)) return false; if (sx_low(b)) anyCased = true; }
            }
            return anyCased;
        }
        default: break;
        }
        if (len == 0) return false;                                  // every remaining class needs a character
        for (int p = start; p < end; p++) {
            uchar b = d[p];
            switch (op) {
            case SXP_U_ALNUM:   if (!(sx_alpha(b) || sx_dig(b))) return false; break;
            case SXP_U_ALPHA:   if (!sx_alpha(b)) return false; break;
            // No ASCII byte is in Unicode's No or Nl categories, so decimal, digit and numeric
            // coincide with [0-9] here.
            case SXP_U_DECIMAL: case SXP_U_DIGIT: case SXP_U_NUMERIC: if (!sx_dig(b)) return false; break;
            case SXP_U_SPACE:   if (!sx_uspace(b)) return false; break;
            default: return false;
            }
        }
        return true;
    }

    // One 32-bit word of the answer per thread, plus one word of "this row has a byte >= 0x80".
    kernel void sx_pred(device const int* offsets [[buffer(0)]], device const uchar* data [[buffer(1)]],
                        device const uint* nPtr [[buffer(2)]], constant uint& op [[buffer(3)]],
                        device uint* out [[buffer(4)]], device uint* nonAscii [[buffer(5)]],
                        uint w [[thread_position_in_grid]]) {
        uint n = *nPtr, base = w * 32u;
        if (base >= n) return;
        uint limit = min(32u, n - base), bits = 0u, high = 0u;
        for (uint j = 0; j < limit; j++) {
            uint i = base + j;
            int start = offsets[i], end = offsets[i + 1];
            bool ascii = true;
            for (int p = start; p < end; p++) if (data[p] >= 0x80u) { ascii = false; break; }
            if (!ascii) { high |= (1u << j); if (op == SXP_STRING_IS_ASCII) continue; }
            if (ascii || op < SXP_U_ALNUM) {
                if (sx_pred_ascii(data, start, end, op)) bits |= (1u << j);
            }
        }
        out[w] = bits;
        nonAscii[w] = high;
    }

    // ---------------------------------------------------------------------------------------------
    // Transforms. Op codes must match StringExtraTransform in StringExtra.swift.
    #define SX_ASCII_TITLE    0u
    #define SX_CENTER         1u
    #define SX_REPLACE_SLICE  2u
    #define SX_REPLACE_BYTES  3u

    struct SxParams { uint op; uint n1; uint n2; int p1; int p2; uint flags; };

    inline uchar sx_upb(uchar b) { return sx_low(b) ? (uchar)(b - 32u) : b; }
    inline uchar sx_lob(uchar b) { return sx_up(b) ? (uchar)(b + 32u) : b; }
    inline int sx_ncp(device const uchar* d, int start, int end) {
        int c = 0;
        for (int p = start; p < end; p++) if ((d[p] & 0xC0) != 0x80) c++;
        return c;
    }
    // Byte position of code point `k`, or `end` when the string is shorter.
    inline int sx_cp_byte(device const uchar* d, int start, int end, int k) {
        int c = 0;
        for (int p = start; p < end; p++) {
            if ((d[p] & 0xC0) != 0x80) { if (c == k) return p; c++; }
        }
        return end;
    }

    // Returns the output byte count, writing the bytes at out[outPos...] when `write` is true.
    inline int sx_apply(device const uchar* d, int start, int len,
                        device const uchar* a1, uint n1, uint op, int p1, int p2,
                        device uchar* out, int outPos, bool write) {
        int end = start + len;
        int n = 0;
        switch (op) {
        case SX_ASCII_TITLE: {
            bool prevCased = false;
            for (int p = start; p < end; p++) {
                uchar b = d[p], r;
                if (sx_alpha(b)) { r = prevCased ? sx_lob(b) : sx_upb(b); prevCased = true; }
                else { r = b; prevCased = false; }
                if (write) out[outPos + n] = r;
                n++;
            }
            return n;
        }
        case SX_CENTER: {
            int ncp = sx_ncp(d, start, end);
            int pad = (p1 > ncp) ? (p1 - ncp) : 0;
            int left = pad / 2, right = pad - left;      // Arrow puts the odd character on the right
            for (int k = 0; k < left; k++)
                for (uint j = 0; j < n1; j++) { if (write) out[outPos + n] = a1[j]; n++; }
            for (int p = start; p < end; p++) { if (write) out[outPos + n] = d[p]; n++; }
            for (int k = 0; k < right; k++)
                for (uint j = 0; j < n1; j++) { if (write) out[outPos + n] = a1[j]; n++; }
            return n;
        }
        case SX_REPLACE_SLICE: case SX_REPLACE_BYTES: {
            bool bytes = (op == SX_REPLACE_BYTES);
            int units = bytes ? len : sx_ncp(d, start, end);
            int s = p1 < 0 ? max(units + p1, 0) : min(p1, units);
            int e = p2 < 0 ? max(units + p2, 0) : min(p2, units);
            if (e < s) e = s;                            // an inverted range inserts, deleting nothing
            int b0 = bytes ? (start + s) : sx_cp_byte(d, start, end, s);
            int b1 = bytes ? (start + e) : sx_cp_byte(d, start, end, e);
            for (int p = start; p < b0; p++) { if (write) out[outPos + n] = d[p]; n++; }
            for (uint j = 0; j < n1; j++) { if (write) out[outPos + n] = a1[j]; n++; }
            for (int p = b1; p < end; p++) { if (write) out[outPos + n] = d[p]; n++; }
            return n;
        }
        default: return 0;
        }
    }

    kernel void sx_tf_len(device const int* offsets [[buffer(0)]], device const uchar* data [[buffer(1)]],
                          device const uchar* validity [[buffer(2)]], device const uint* nPtr [[buffer(3)]],
                          constant SxParams& prm [[buffer(4)]], device const uchar* a1 [[buffer(5)]],
                          device int* outLens [[buffer(6)]], device uchar* scratch [[buffer(7)]],
                          uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if ((prm.flags & 1u) != 0u && !bit_get(validity, i)) { outLens[i] = 0; return; }
        int start = offsets[i], len = offsets[i + 1] - start;
        outLens[i] = sx_apply(data, start, len, a1, prm.n1, prm.op, prm.p1, prm.p2, scratch, 0, false);
    }
    kernel void sx_tf_write(device const int* offsets [[buffer(0)]], device const uchar* data [[buffer(1)]],
                            device const uchar* validity [[buffer(2)]], device const uint* nPtr [[buffer(3)]],
                            constant SxParams& prm [[buffer(4)]], device const uchar* a1 [[buffer(5)]],
                            device const int* outOffsets [[buffer(6)]], device uchar* outData [[buffer(7)]],
                            uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if ((prm.flags & 1u) != 0u && !bit_get(validity, i)) return;
        int start = offsets[i], len = offsets[i + 1] - start;
        sx_apply(data, start, len, a1, prm.n1, prm.op, prm.p1, prm.p2, outData, outOffsets[i], true);
    }

    // ---------------------------------------------------------------------------------------------
    // binary_join over a list<utf8>: row i is the concatenation of its child strings separated by
    // `sep`. prm.flags bit 0 = the list has a validity bitmap, bit 1 = the child has one, bit 2 = the
    // separator is a per-row array (otherwise a scalar of prm.n1 bytes), bit 3 = that array has one.
    // A null list row, a null element inside a row, or a null separator gives a null output row.
    inline bool sx_join_valid(device const int* lo, device const uchar* lv, device const uchar* cv,
                              device const int* so, device const uchar* sv, constant SxParams& prm, uint i) {
        if ((prm.flags & 1u) != 0u && !bit_get(lv, i)) return false;
        if ((prm.flags & 12u) == 12u && !bit_get(sv, i)) return false;
        if ((prm.flags & 2u) != 0u) {
            for (int k = lo[i]; k < lo[i + 1]; k++) if (!bit_get(cv, (uint)k)) return false;
        }
        return true;
    }
    kernel void sx_join_len(device const int* lo [[buffer(0)]], device const int* co [[buffer(1)]],
                            device const uchar* lv [[buffer(2)]], device const uchar* cv [[buffer(3)]],
                            device const int* so [[buffer(4)]], device const uchar* sv [[buffer(5)]],
                            device const uint* nPtr [[buffer(6)]], constant SxParams& prm [[buffer(7)]],
                            device int* outLens [[buffer(8)]], device uchar* outValid [[buffer(9)]],
                            uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if (!sx_join_valid(lo, lv, cv, so, sv, prm, i)) { outLens[i] = 0; outValid[i] = 0; return; }
        int a = lo[i], b = lo[i + 1], total = 0;
        for (int k = a; k < b; k++) total += co[k + 1] - co[k];
        int sepLen = ((prm.flags & 4u) != 0u) ? (so[i + 1] - so[i]) : (int)prm.n1;
        if (b > a) total += (b - a - 1) * sepLen;
        outLens[i] = total;
        outValid[i] = 1;
    }
    kernel void sx_join_write(device const int* lo [[buffer(0)]], device const int* co [[buffer(1)]],
                              device const uchar* cd [[buffer(2)]], device const uchar* lv [[buffer(3)]],
                              device const uchar* cv [[buffer(4)]], device const int* so [[buffer(5)]],
                              device const uchar* sd [[buffer(6)]], device const uchar* sv [[buffer(7)]],
                              device const uint* nPtr [[buffer(8)]], constant SxParams& prm [[buffer(9)]],
                              device const int* outOffsets [[buffer(10)]], device uchar* outData [[buffer(11)]],
                              uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if (!sx_join_valid(lo, lv, cv, so, sv, prm, i)) return;
        bool sepArray = ((prm.flags & 4u) != 0u);
        int sepStart = sepArray ? so[i] : 0;
        int sepLen = sepArray ? (so[i + 1] - so[i]) : (int)prm.n1;
        int to = outOffsets[i];
        for (int k = lo[i]; k < lo[i + 1]; k++) {
            if (k > lo[i]) for (int j = 0; j < sepLen; j++) outData[to++] = sd[sepStart + j];
            for (int p = co[k]; p < co[k + 1]; p++) outData[to++] = cd[p];
        }
    }

    // ---------------------------------------------------------------------------------------------
    // is_in / index_in over strings: an open-addressing table of row indices keyed by the 64-bit
    // string hash. EMPTY is 0xFFFFFFFF; the table never removes an entry, so once a slot is taken it
    // stays taken and every thread hashing the same string walks the same probe sequence and lands on
    // the same slot. Equality is always decided by comparing bytes, never by comparing hashes.
    #define SX_EMPTY 0xFFFFFFFFu

    inline bool sx_eq(device const int* oa, device const uchar* da, uint i,
                      device const int* ob, device const uchar* db, uint j) {
        int a0 = oa[i], la = oa[i + 1] - a0;
        int b0 = ob[j], lb = ob[j + 1] - b0;
        if (la != lb) return false;
        for (int t = 0; t < la; t++) if (da[a0 + t] != db[b0 + t]) return false;
        return true;
    }
    inline uint sx_slot(ulong k, uint mask) { return ((uint)(k >> 32) ^ (uint)k) & mask; }

    kernel void sx_hash_insert(device const int* so [[buffer(0)]], device const uchar* sd [[buffer(1)]],
                               device const ulong* keys [[buffer(2)]], device const uchar* sv [[buffer(3)]],
                               device const uint* nPtr [[buffer(4)]], constant uint& mask [[buffer(5)]],
                               constant uint& hasValidity [[buffer(6)]], device atomic_uint* table [[buffer(7)]],
                               uint i [[thread_position_in_grid]]) {
        uint n = *nPtr;
        if (i >= n) return;
        if (hasValidity != 0u && !bit_get(sv, i)) return;            // nulls in the value set are ignored
        uint slot = sx_slot(keys[i], mask);
        uint budget = 4u * (mask + 1u) + 64u;
        while (budget-- > 0u) {
            uint cur = atomic_load_explicit(&table[slot], memory_order_relaxed);
            if (cur == SX_EMPTY) {
                uint expected = SX_EMPTY;
                if (atomic_compare_exchange_weak_explicit(&table[slot], &expected, i,
                                                          memory_order_relaxed, memory_order_relaxed)) return;
                continue;                                            // lost the race (or a spurious
            }                                                        // failure): re-read the same slot
            if (sx_eq(so, sd, i, so, sd, cur)) {
                // Same string already present: keep the lowest row index, which index_in reports.
                uint c = cur;
                while (c > i) {
                    if (atomic_compare_exchange_weak_explicit(&table[slot], &c, i,
                                                              memory_order_relaxed, memory_order_relaxed)) return;
                }
                return;
            }
            slot = (slot + 1u) & mask;
        }
    }

    // The set row matching probe row i, or SX_EMPTY.
    inline uint sx_lookup(device const int* po, device const uchar* pd, uint i, ulong key,
                          device const int* so, device const uchar* sd,
                          device const uint* table, uint mask) {
        uint slot = sx_slot(key, mask);
        for (uint probe = 0; probe <= mask; probe++) {
            uint cur = table[slot];
            if (cur == SX_EMPTY) return SX_EMPTY;
            if (sx_eq(po, pd, i, so, sd, cur)) return cur;
            slot = (slot + 1u) & mask;
        }
        return SX_EMPTY;
    }

    kernel void sx_is_in(device const int* po [[buffer(0)]], device const uchar* pd [[buffer(1)]],
                         device const ulong* keys [[buffer(2)]], device const uchar* pv [[buffer(3)]],
                         device const int* so [[buffer(4)]], device const uchar* sd [[buffer(5)]],
                         device const uint* table [[buffer(6)]], device const uint* nPtr [[buffer(7)]],
                         constant uint& mask [[buffer(8)]], constant uint& hasValidity [[buffer(9)]],
                         device uint* out [[buffer(10)]], uint w [[thread_position_in_grid]]) {
        uint n = *nPtr, base = w * 32u;
        if (base >= n) return;
        uint limit = min(32u, n - base), bits = 0u;
        for (uint j = 0; j < limit; j++) {
            uint i = base + j;
            if (hasValidity != 0u && !bit_get(pv, i)) continue;      // a null is never in the set
            if (sx_lookup(po, pd, i, keys[i], so, sd, table, mask) != SX_EMPTY) bits |= (1u << j);
        }
        out[w] = bits;
    }

    kernel void sx_index_in(device const int* po [[buffer(0)]], device const uchar* pd [[buffer(1)]],
                            device const ulong* keys [[buffer(2)]], device const uchar* pv [[buffer(3)]],
                            device const int* so [[buffer(4)]], device const uchar* sd [[buffer(5)]],
                            device const uint* table [[buffer(6)]], device const uint* nPtr [[buffer(7)]],
                            constant uint& mask [[buffer(8)]], constant uint& hasValidity [[buffer(9)]],
                            device int* outValues [[buffer(10)]], device uchar* outValid [[buffer(11)]],
                            uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        outValues[i] = 0;
        outValid[i] = 0;
        if (hasValidity != 0u && !bit_get(pv, i)) return;
        uint hit = sx_lookup(po, pd, i, keys[i], so, sd, table, mask);
        if (hit == SX_EMPTY) return;
        outValues[i] = (int)hit;
        outValid[i] = 1;
    }
    """
}
