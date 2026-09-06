import Foundation

/// Metal Shading Language source templates. Kernels are generated per element type at runtime.
enum KernelSource {
    static let prelude = """
    #include <metal_stdlib>
    using namespace metal;
    #define TG 256u
    inline bool bit_get(device const uchar* bm, uint i) { return (bm[i >> 3] >> (i & 7)) & 1; }
    """

    /// Reduction kernels: sum / min / max with null awareness. `ACC` is the accumulator type.
    /// Each threadgroup writes one partial and one valid-count; the CPU finalises the partials.
    static func reductions(T: String, ACC: String, minInit: String, maxInit: String) -> String {
        func body(_ name: String, _ initVal: String, _ combine: String) -> String { """
        kernel void reduce_\(name)(device const \(T)* vals [[buffer(0)]],
                                  device const uchar* validity [[buffer(1)]],
                                  constant uint& n [[buffer(2)]],
                                  constant uint& hasValidity [[buffer(3)]],
                                  device \(ACC)* partials [[buffer(4)]],
                                  device uint* counts [[buffer(5)]],
                                  uint gid [[thread_position_in_grid]],
                                  uint lid [[thread_index_in_threadgroup]],
                                  uint tgid [[threadgroup_position_in_grid]],
                                  uint gridSize [[threads_per_grid]]) {
            threadgroup \(ACC) shared[TG];
            threadgroup uint scount[TG];
            \(ACC) acc = \(initVal);
            uint cnt = 0;
            if (hasValidity) {
                for (uint i = gid; i < n; i += gridSize) {
                    if (bit_get(validity, i)) { \(ACC) v = (\(ACC))vals[i]; acc = \(combine); cnt++; }
                }
            } else {
                for (uint i = gid; i < n; i += gridSize) { \(ACC) v = (\(ACC))vals[i]; acc = \(combine); cnt++; }
            }
            shared[lid] = acc; scount[lid] = cnt;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = TG / 2; s > 0; s >>= 1) {
                if (lid < s) { \(ACC) v = shared[lid + s]; \(ACC) acc = shared[lid]; shared[lid] = \(combine); scount[lid] += scount[lid + s]; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lid == 0) { partials[tgid] = shared[0]; counts[tgid] = scount[0]; }
        }
        """ }
        return prelude
            + body("sum", "0", "acc + v")
            + body("min", minInit, "min(acc, v)")
            + body("max", maxInit, "max(acc, v)")
    }

    /// Comparison kernels producing a packed Arrow boolean bitmap. One thread per 32-bit output word.
    static func compare(T: String) -> String {
        let ops: [(String, String)] = [("eq", "=="), ("ne", "!="), ("lt", "<"), ("le", "<="), ("gt", ">"), ("ge", ">=")]
        var s = prelude
        for (name, op) in ops {
            s += """
            kernel void cmp_scalar_\(name)(device const \(T)* a [[buffer(0)]],
                                          constant \(T)& scalar [[buffer(1)]],
                                          constant uint& n [[buffer(2)]],
                                          device uint* out [[buffer(3)]],
                                          uint w [[thread_position_in_grid]]) {
                uint base = w * 32u;
                if (base >= n) return;
                uint limit = min(32u, n - base);
                uint bits = 0;
                for (uint j = 0; j < limit; j++) { if (a[base + j] \(op) scalar) bits |= (1u << j); }
                out[w] = bits;
            }
            kernel void cmp_array_\(name)(device const \(T)* a [[buffer(0)]],
                                         device const \(T)* b [[buffer(1)]],
                                         constant uint& n [[buffer(2)]],
                                         device uint* out [[buffer(3)]],
                                         uint w [[thread_position_in_grid]]) {
                uint base = w * 32u;
                if (base >= n) return;
                uint limit = min(32u, n - base);
                uint bits = 0;
                for (uint j = 0; j < limit; j++) { if (a[base + j] \(op) b[base + j]) bits |= (1u << j); }
                out[w] = bits;
            }

            """
        }
        return s
    }

    /// Element-wise arithmetic. One thread per element.
    static func arithmetic(T: String) -> String {
        let ops: [(String, String)] = [("add", "+"), ("sub", "-"), ("mul", "*"), ("div", "/")]
        var s = prelude
        for (name, op) in ops {
            s += """
            kernel void arith_scalar_\(name)(device const \(T)* a [[buffer(0)]],
                                            constant \(T)& scalar [[buffer(1)]],
                                            constant uint& n [[buffer(2)]],
                                            device \(T)* out [[buffer(3)]],
                                            uint i [[thread_position_in_grid]]) {
                if (i < n) out[i] = a[i] \(op) scalar;
            }
            kernel void arith_array_\(name)(device const \(T)* a [[buffer(0)]],
                                           device const \(T)* b [[buffer(1)]],
                                           constant uint& n [[buffer(2)]],
                                           device \(T)* out [[buffer(3)]],
                                           uint i [[thread_position_in_grid]]) {
                if (i < n) out[i] = a[i] \(op) b[i];
            }

            """
        }
        return s
    }

    /// Bitmap word operations (validity combination) and bit packing.
    static let bitmap = prelude + """
    kernel void bitmap_and(device const uint* a [[buffer(0)]], device const uint* b [[buffer(1)]],
                           constant uint& words [[buffer(2)]], device uint* out [[buffer(3)]],
                           uint w [[thread_position_in_grid]]) {
        if (w < words) out[w] = a[w] & b[w];
    }
    kernel void bitmap_and_not(device const uint* a [[buffer(0)]], device const uint* b [[buffer(1)]],
                               constant uint& words [[buffer(2)]], device uint* out [[buffer(3)]],
                               uint w [[thread_position_in_grid]]) {
        if (w < words) out[w] = a[w] & ~b[w];
    }
    kernel void bitmap_or(device const uint* a [[buffer(0)]], device const uint* b [[buffer(1)]],
                          constant uint& words [[buffer(2)]], device uint* out [[buffer(3)]],
                          uint w [[thread_position_in_grid]]) {
        if (w < words) out[w] = a[w] | b[w];
    }
    kernel void bitmap_not(device const uint* a [[buffer(0)]],
                           constant uint& words [[buffer(2)]], device uint* out [[buffer(3)]],
                           uint w [[thread_position_in_grid]]) {
        if (w < words) out[w] = ~a[w];
    }
    // Packs one byte-per-element (0/1) buffer into a bitmap. One thread per output word.
    kernel void pack_bits(device const uchar* bytes [[buffer(0)]], constant uint& n [[buffer(1)]],
                          device uint* out [[buffer(2)]], uint w [[thread_position_in_grid]]) {
        uint base = w * 32u;
        if (base >= n) return;
        uint limit = min(32u, n - base);
        uint bits = 0;
        for (uint j = 0; j < limit; j++) { if (bytes[base + j]) bits |= (1u << j); }
        out[w] = bits;
    }
    """

    /// Stream compaction (Arrow `filter`). The selection bitmap has one bit per element.
    /// Pass 1 counts selected elements per block of TG*32 elements; the host scans the block counts;
    /// pass 2 scatters values (and validity as bytes, packed afterwards) using an in-threadgroup prefix sum.
    static func filter(T: String) -> String { prelude + """
    inline uint masked_word(device const uint* sel, uint w, uint n) {
        uint base = w * 32u;
        if (base >= n) return 0u;
        uint word = sel[w];
        uint limit = n - base;
        if (limit < 32u) word &= (1u << limit) - 1u;
        return word;
    }
    kernel void filter_count(device const uint* sel [[buffer(0)]],
                             constant uint& n [[buffer(1)]],
                             device uint* blockCounts [[buffer(2)]],
                             uint w [[thread_position_in_grid]],
                             uint lid [[thread_index_in_threadgroup]],
                             uint tgid [[threadgroup_position_in_grid]],
                             uint sgid [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
        threadgroup uint simdTotals[32];
        uint c = popcount(masked_word(sel, w, n));
        uint t = simd_sum(c);
        if (lane == 0) simdTotals[sgid] = t;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            uint total = 0;
            for (uint k = 0; k < TG / 32u; k++) total += simdTotals[k];
            blockCounts[tgid] = total;
        }
    }
    kernel void filter_scatter(device const \(T)* vals [[buffer(0)]],
                               device const uchar* validity [[buffer(1)]],
                               device const uint* sel [[buffer(2)]],
                               constant uint& n [[buffer(3)]],
                               constant uint& hasValidity [[buffer(4)]],
                               device const uint* blockOffsets [[buffer(5)]],
                               device \(T)* out [[buffer(6)]],
                               device uchar* outValidBytes [[buffer(7)]],
                               uint w [[thread_position_in_grid]],
                               uint lid [[thread_index_in_threadgroup]],
                               uint tgid [[threadgroup_position_in_grid]],
                               uint sgid [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
        threadgroup uint simdTotals[32];
        uint word = masked_word(sel, w, n);
        uint c = popcount(word);
        uint local = simd_prefix_exclusive_sum(c);
        uint t = simd_sum(c);
        if (lane == 0) simdTotals[sgid] = t;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint prefix = 0;
        for (uint k = 0; k < sgid; k++) prefix += simdTotals[k];
        uint pos = blockOffsets[tgid] + prefix + local;
        uint base = w * 32u;
        while (word) {
            uint j = ctz(word);
            word &= word - 1u;
            uint i = base + j;
            out[pos] = vals[i];
            if (hasValidity) outValidBytes[pos] = bit_get(validity, i) ? 1 : 0;
            pos++;
        }
    }
    """ }
}
