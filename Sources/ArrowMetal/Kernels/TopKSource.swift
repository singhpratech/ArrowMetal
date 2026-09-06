import Foundation

/// MSL for per-threadgroup top-k selection.
///
/// Each threadgroup owns a block of rows and keeps a candidate buffer in threadgroup memory holding the
/// best entries it has seen, plus a threshold: the worst of the k it is currently holding. A row only
/// enters the buffer if it beats that threshold, which after the first few chunks rejects almost
/// everything, so the pass costs about one read per row. When the buffer is close to full it is sorted
/// with a bitonic network and truncated back to k, which also refreshes the threshold.
///
/// Entries are `(key, row)` pairs ordered lexicographically, where `key` is the same order-preserving
/// map `SortSource` uses (with `inv` flipping it for "largest"). Row indices are unique, so the order is
/// total and the result matches a full stable argsort exactly. The buffer is dynamic threadgroup memory
/// so that a small k costs a small allocation and keeps occupancy up.
///
/// The count of held entries is read once per chunk by thread 0 and broadcast through `shared_c`. Every
/// thread needs the *same* value: it decides a branch containing `threadgroup_barrier`, and it is the base
/// of a strided fill whose strides only tile `[c, cap)` when they all start from one `c`. Reading the
/// atomic per thread does not guarantee that (Round 8 in docs/FINDINGS.md).
enum TopKSource {
    /// `kind` names the value mapping, `V` is the MSL type of a row, `K` the key type (uint or ulong).
    static func source(kind: String, V: String, K: String) -> String {
        let keyMax = K == "ulong" ? "ULONG_MAX" : "UINT_MAX"
        // The float mappings canonicalise exactly as `SortSource` does, or top_k would order the ties
        // this library calls equal differently from argsort: -0.0 collapses onto 0.0, every NaN onto
        // one value after +inf, and a reversed order keeps NaN at the end rather than mirroring it to
        // the front. `nan` is a separate flag because the descending key is the free maximum, not ~k.
        let map: String
        var nanFlag = "bool nan = false;"
        switch kind {
        case "i32": map = "\(K) k = (uint)v ^ 0x80000000u;"
        case "u32": map = "\(K) k = (uint)v;"
        case "f32":
            nanFlag = "bool nan = (as_type<uint>(v) & 0x7FFFFFFFu) > 0x7F800000u;"
            map = """
            uint b = as_type<uint>(v); if ((b & 0x7FFFFFFFu) == 0u) b = 0u; if (nan) b = 0x7F800001u;
                \(K) k = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
            """
        case "i64": map = "\(K) k = (ulong)v ^ 0x8000000000000000ul;"
        case "u64": map = "\(K) k = (ulong)v;"
        default:
            nanFlag = "bool nan = ((ulong)v & 0x7FFFFFFFFFFFFFFFul) > 0x7FF0000000000000ul;"
            map = """
            ulong b = (ulong)v; if ((b & 0x7FFFFFFFFFFFFFFFul) == 0ul) b = 0ul; if (nan) b = 0x7FF0000000000001ul;
                \(K) k = (b & 0x8000000000000000ul) ? ~b : (b | 0x8000000000000000ul);
            """
        }
        return KernelSource.prelude + """

        #define TK_NOROW 0xFFFFFFFFu
        inline \(K) tk_map(\(V) v, uint inv) { \(nanFlag) \(map) return inv ? (nan ? \(keyMax) : ~k) : k; }
        // Lexicographic (key, row). Row indices are unique, so this is a strict total order.
        inline bool tk_less(\(K) a, uint ai, \(K) b, uint bi) { return (a < b) || (a == b && ai < bi); }

        kernel void topk_select(device const \(V)* vals [[buffer(0)]],
                                device const uchar* validity [[buffer(1)]],
                                device const uint* nPtr [[buffer(2)]],
                                constant uint& hasValidity [[buffer(3)]],
                                constant uint& inv [[buffer(4)]],
                                constant uint& k [[buffer(5)]],
                                constant uint& cap [[buffer(6)]],
                                constant uint& elemsPerBlock [[buffer(7)]],
                                device \(K)* outKeys [[buffer(8)]],
                                device uint* outRows [[buffer(9)]],
                                threadgroup \(K)* bufKey [[threadgroup(0)]],
                                threadgroup uint* bufRow [[threadgroup(1)]],
                                uint lid [[thread_index_in_threadgroup]],
                                uint tgid [[threadgroup_position_in_grid]]) {
            threadgroup atomic_uint held;
            // The count every thread works from. Reading `held` per thread is not safe: a relaxed atomic
            // load is not guaranteed to give every thread the same answer, and this one decides both a
            // barrier-carrying branch and the range each thread fills. One thread reads it, the rest take
            // it from here. See the `held` note in docs/FINDINGS.md.
            threadgroup uint shared_c;
            threadgroup \(K) thrKey;
            threadgroup uint thrRow;
            if (lid == 0u) {
                atomic_store_explicit(&held, 0u, memory_order_relaxed);
                thrKey = \(keyMax); thrRow = TK_NOROW;      // nothing held yet: everything beats it
            }
            // Sentinels everywhere to start: a slot no one writes then reads as "no row" and is dropped by
            // the host, never as the (key 0, row 0) pair that would sort ahead of every real candidate.
            for (uint i = lid; i < cap; i += TG) { bufKey[i] = \(keyMax); bufRow[i] = TK_NOROW; }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint n = *nPtr;
            uint start = tgid * elemsPerBlock;
            uint end = min(n, start + elemsPerBlock);
            for (uint base = start; base < end; base += TG) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lid == 0u) shared_c = min(atomic_load_explicit(&held, memory_order_relaxed), cap);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                uint c = shared_c;
                if (c + TG > cap) {
                    // Sort the buffer, keep the best k, and tighten the threshold to the k-th best.
                    for (uint i = c + lid; i < cap; i += TG) { bufKey[i] = \(keyMax); bufRow[i] = TK_NOROW; }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    TK_BITONIC
                    if (lid == 0u && c >= k) {
                        atomic_store_explicit(&held, k, memory_order_relaxed);
                        thrKey = bufKey[k - 1u]; thrRow = bufRow[k - 1u];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                uint i = base + lid;
                if (i < end && (!hasValidity || bit_get(validity, i))) {
                    \(K) key = tk_map(vals[i], inv);
                    if (tk_less(key, i, thrKey, thrRow)) {
                        uint slot = atomic_fetch_add_explicit(&held, 1u, memory_order_relaxed);
                        if (slot < cap) { bufKey[slot] = key; bufRow[slot] = i; }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid == 0u) shared_c = min(atomic_load_explicit(&held, memory_order_relaxed), cap);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint c = shared_c;
            for (uint i = c + lid; i < cap; i += TG) { bufKey[i] = \(keyMax); bufRow[i] = TK_NOROW; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            TK_BITONIC
            for (uint j = lid; j < k; j += TG) { outKeys[tgid * k + j] = bufKey[j]; outRows[tgid * k + j] = bufRow[j]; }
        }
        """.replacingOccurrences(of: "TK_BITONIC", with: bitonic(K: K))
    }

    /// Ascending bitonic sorting network over `cap` (a power of two) `(key, row)` pairs in threadgroup
    /// memory. Every thread walks the same uniform loops, so the barriers are reached by all of them.
    private static func bitonic(K: String) -> String { """
    for (uint len = 2u; len <= cap; len <<= 1) {
                        for (uint step = len >> 1; step > 0u; step >>= 1) {
                            for (uint t = lid; t < (cap >> 1); t += TG) {
                                uint low = t & (step - 1u);
                                uint i0 = ((t - low) << 1) + low;
                                uint i1 = i0 + step;
                                bool up = ((i0 & len) == 0u);
                                \(K) a = bufKey[i0], b = bufKey[i1];
                                uint ai = bufRow[i0], bi = bufRow[i1];
                                bool lt = tk_less(a, ai, b, bi);
                                if (up ? !lt : lt) { bufKey[i0] = b; bufRow[i0] = bi; bufKey[i1] = a; bufRow[i1] = ai; }
                            }
                            threadgroup_barrier(mem_flags::mem_threadgroup);
                        }
                    }
    """ }
}
