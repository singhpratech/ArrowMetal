import Foundation

/// MSL for Arrow `utf8` arrays. The per-row kernels are templated over the string layout
/// (`StringLayoutSource`): each exists as `name` (offsets + bytes) and `name_v` (utf8_view), and the
/// two-array `str_eq_array` in all four combinations.
enum StringSource {
    static let source = KernelSource.prelude + StringLayoutSource.accessors + """
    // Byte length of each string.
    template <typename S> inline void str_byte_length_t(S s, device const uint* nPtr, device int* out, uint i) {
        if (i < *nPtr) out[i] = s.len(i);
    }
    // UTF-8 code point count (bytes that are not continuation bytes).
    template <typename S> inline void str_char_length_t(S s, device const uint* nPtr, device int* out, uint i) {
        if (i >= *nPtr) return;
        int len; device const uchar* p = s.row(i, len);
        int c = 0;
        for (int k = 0; k < len; k++) if ((p[k] & 0xC0) != 0x80) c++;
        out[i] = c;
    }
    // Pattern predicates -> packed boolean bitmap. op: 0 equals, 1 starts_with, 2 ends_with, 3 contains.
    inline bool str_match(device const uchar* data, int start, int len, device const uchar* pat, uint plen, uint op) {
        if (op == 0u) { if ((uint)len != plen) return false; for (uint j = 0; j < plen; j++) if (data[start + j] != pat[j]) return false; return true; }
        if (plen > (uint)len) return false;
        if (op == 1u) { for (uint j = 0; j < plen; j++) if (data[start + j] != pat[j]) return false; return true; }
        if (op == 2u) { int base = start + len - (int)plen; for (uint j = 0; j < plen; j++) if (data[base + j] != pat[j]) return false; return true; }
        if (plen == 0u) return true;
        for (int s = start; s + (int)plen <= start + len; s++) {
            bool ok = true;
            for (uint j = 0; j < plen && ok; j++) if (data[s + j] != pat[j]) ok = false;
            if (ok) return true;
        }
        return false;
    }
    template <typename S> inline void str_predicate_t(S s, device const uint* nPtr, device const uchar* pat,
                                                      uint plen, uint op, device uint* out, uint w) {
        uint n = *nPtr, base = w * 32u;
        if (base >= n) return;
        uint limit = min(32u, n - base), bits = 0;
        for (uint j = 0; j < limit; j++) {
            uint i = base + j;
            int len; device const uchar* p = s.row(i, len);
            if (str_match(p, 0, len, pat, plen, op)) bits |= (1u << j);
        }
        out[w] = bits;
    }
    // Element-wise equality of two string arrays.
    template <typename A, typename B> inline void str_eq_array_t(A a, B b, device const uint* nPtr,
                                                                 device uint* out, uint w) {
        uint n = *nPtr, base = w * 32u;
        if (base >= n) return;
        uint limit = min(32u, n - base), bits = 0;
        for (uint j = 0; j < limit; j++) {
            uint i = base + j;
            int la, lb;
            device const uchar* pa = a.row(i, la);
            device const uchar* pb = b.row(i, lb);
            if (la != lb) continue;
            bool eq = true;
            for (int k = 0; k < la && eq; k++) if (pa[k] != pb[k]) eq = false;
            if (eq) bits |= (1u << j);
        }
        out[w] = bits;
    }
    // MurmurHash3 x86_32 per string (seed 0), matching the reference implementation.
    inline uint rotl32(uint x, int r) { return (x << r) | (x >> (32 - r)); }
    inline uint fmix32(uint h) { h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16; return h; }
    template <typename S> inline void str_hash32_t(S s, device const uint* nPtr, device uint* out, uint i) {
        if (i >= *nPtr) return;
        int len; device const uchar* data = s.row(i, len);
        int start = 0;
        uint h = 0u; const uint c1 = 0xcc9e2d51u, c2 = 0x1b873593u;
        int nblocks = len / 4;
        for (int b = 0; b < nblocks; b++) {
            int p = start + b * 4;
            uint k = (uint)data[p] | ((uint)data[p + 1] << 8) | ((uint)data[p + 2] << 16) | ((uint)data[p + 3] << 24);
            k *= c1; k = rotl32(k, 15); k *= c2;
            h ^= k; h = rotl32(h, 13); h = h * 5u + 0xe6546b64u;
        }
        uint k1 = 0u; int tail = start + nblocks * 4;
        switch (len & 3) {
            case 3: k1 ^= (uint)data[tail + 2] << 16;
            case 2: k1 ^= (uint)data[tail + 1] << 8;
            case 1: k1 ^= (uint)data[tail]; k1 *= c1; k1 = rotl32(k1, 15); k1 *= c2; h ^= k1;
        }
        h ^= (uint)len;
        out[i] = fmix32(h);
    }
    // Copies string bytes for a gather: out string i comes from source string src[i], at outOffsets[i].
    template <typename S> inline void str_gather_bytes_t(S s, device const int* src, device const int* outOffsets,
                                                         device const uint* nPtr, device uchar* outData, uint i) {
        if (i >= *nPtr) return;
        int from = src[i];
        if (from < 0) return;
        // The output slot's own width, not the source row's: a null source row keeps whatever bytes it
        // held (valid Arrow) but was given length 0, so copying its source bytes would write them over
        // the next row's slot.
        int to = outOffsets[i], len = outOffsets[i + 1] - to;
        int srcLen; device const uchar* p = s.row((uint)from, srcLen);
        for (int k = 0; k < len; k++) outData[to + k] = p[k];
    }

    """ + StringLayoutSource.variants("str_byte_length", slots: [0],
        params: "device const uint* nPtr [[buffer(2)]], device int* out [[buffer(3)]], uint i [[thread_position_in_grid]]",
        call: "str_byte_length_t(S0, nPtr, out, i)")
    + StringLayoutSource.variants("str_char_length", slots: [0],
        params: "device const uint* nPtr [[buffer(2)]], device int* out [[buffer(3)]], uint i [[thread_position_in_grid]]",
        call: "str_char_length_t(S0, nPtr, out, i)")
    + StringLayoutSource.variants("str_predicate", slots: [0],
        params: "device const uint* nPtr [[buffer(2)]], device const uchar* pat [[buffer(3)]], constant uint& plen [[buffer(4)]], constant uint& op [[buffer(5)]], device uint* out [[buffer(6)]], uint w [[thread_position_in_grid]]",
        call: "str_predicate_t(S0, nPtr, pat, plen, op, out, w)")
    + StringLayoutSource.variants("str_eq_array", slots: [0, 2],
        params: "device const uint* nPtr [[buffer(4)]], device uint* out [[buffer(5)]], uint w [[thread_position_in_grid]]",
        call: "str_eq_array_t(S0, S1, nPtr, out, w)")
    + StringLayoutSource.variants("str_hash32", slots: [0],
        params: "device const uint* nPtr [[buffer(2)]], device uint* out [[buffer(3)]], uint i [[thread_position_in_grid]]",
        call: "str_hash32_t(S0, nPtr, out, i)")
    + StringLayoutSource.variants("str_gather_bytes", slots: [0],
        params: "device const int* src [[buffer(2)]], device const int* outOffsets [[buffer(3)]], device const uint* nPtr [[buffer(4)]], device uchar* outData [[buffer(5)]], uint i [[thread_position_in_grid]]",
        call: "str_gather_bytes_t(S0, src, outOffsets, nPtr, outData, i)")
    + """

    // Exclusive scan of int32 lengths into offsets (n+1), two-level: per-block scan + block totals.
    kernel void scan_block(device const int* vals [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                           device int* out [[buffer(2)]], device int* blockTotals [[buffer(3)]],
                           uint i [[thread_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                           uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
        threadgroup int simdTotals[32];
        uint n = *nPtr;
        int v = (i < n) ? vals[i] : 0;
        int pre = simd_prefix_exclusive_sum(v);
        int t = simd_sum(v);
        if (lane == 0) simdTotals[sgid] = t;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int prefix = 0;
        for (uint k = 0; k < sgid; k++) prefix += simdTotals[k];
        if (i < n) out[i] = prefix + pre;
        if (lid == TG - 1) { int total = 0; for (uint k = 0; k < TG / 32u; k++) total += simdTotals[k]; blockTotals[tgid] = total; }
    }
    // Single threadgroup: exclusive scan of block totals in place; writes grand total.
    kernel void scan_totals(device int* blockTotals [[buffer(0)]], constant uint& blocks [[buffer(1)]],
                            device int* grand [[buffer(2)]], uint lid [[thread_index_in_threadgroup]],
                            uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
        threadgroup int simdTotals[32];
        uint per = (blocks + TG - 1) / TG;
        uint lo = lid * per, hi = min(blocks, lo + per);
        int local = 0;
        for (uint b = lo; b < hi; b++) local += blockTotals[b];
        int pre = simd_prefix_exclusive_sum(local);
        int t = simd_sum(local);
        if (lane == 0) simdTotals[sgid] = t;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int prefix = 0;
        for (uint k = 0; k < sgid; k++) prefix += simdTotals[k];
        int run = prefix + pre;
        for (uint b = lo; b < hi; b++) { int c = blockTotals[b]; blockTotals[b] = run; run += c; }
        if (lid == TG - 1) *grand = run;
    }
    kernel void scan_add(device int* out [[buffer(0)]], device const int* blockTotals [[buffer(1)]],
                         device const uint* nPtr [[buffer(2)]], device const int* grand [[buffer(3)]],
                         uint i [[thread_position_in_grid]], uint tgid [[threadgroup_position_in_grid]]) {
        uint n = *nPtr;
        if (i < n) out[i] += blockTotals[tgid];
        if (i == n) out[n] = *grand;     // offsets[n] = total
    }
    """
}
