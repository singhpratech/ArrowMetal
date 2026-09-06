import Foundation

/// MSL for hash-free group-by over dense integer keys in [0, K).
///
/// Two paths generated from one template:
/// - `priv` (K <= 1024): each threadgroup accumulates a private table in threadgroup memory with 32-bit
///   atomics (64-bit sums as lo/hi halves with carry), then writes its partial table to device memory.
///   A finalize kernel merges partials per key without atomics.
/// - `dev` (K > 1024): the same accumulation straight into device-memory atomics.
enum GroupBySource {
    static let maxPrivateKeys = 1024

    /// `T` value MSL type, `KT` key type ("int" or "long"), `kind` one of sum_int, sum_uint, sum_float, count, min_int, max_int, min_uint, max_uint, min_float, max_float
    static func source(T: String, KT: String) -> String {
        var s = KernelSource.prelude + """
        inline int f_key(float f) { int b = as_type<int>(f); return b ^ (int)(((uint)(b >> 31)) >> 1); }
        #define MAXK 1024u

        """
        for space in ["priv", "dev"] {
            s += kernel(space: space, T: T, KT: KT)
        }
        s += """
        // Merge per-threadgroup partial tables (priv path). One thread per key.
        kernel void gb_finalize(device const ulong* partials [[buffer(0)]],
                                device const uint* pcounts [[buffer(1)]],
                                constant uint& K [[buffer(2)]],
                                constant uint& numTG [[buffer(3)]],
                                constant uint& kind [[buffer(4)]],
                                device ulong* out [[buffer(5)]],
                                device ulong* counts [[buffer(6)]],
                                uint k [[thread_position_in_grid]]) {
            if (k >= K) return;
            ulong c = 0;
            for (uint t = 0; t < numTG; t++) c += pcounts[t * K + k];
            counts[k] = c;
            if (kind == 2u) {              // float sum: partials hold float bits in the low word
                float acc = 0.0f;
                for (uint t = 0; t < numTG; t++) if (pcounts[t * K + k]) acc += as_type<float>((uint)partials[t * K + k]);
                out[k] = (ulong)as_type<uint>(acc);
            } else if (kind == 0u || kind == 1u || kind == 3u) {   // integer sums / count
                ulong acc = 0;
                for (uint t = 0; t < numTG; t++) acc += partials[t * K + k];
                out[k] = acc;
            } else if (kind == 4u || kind == 6u || kind == 8u) {   // min (stored as int/uint in low word)
                bool isU = kind == 6u;
                int mi = INT_MAX; uint mu = UINT_MAX;
                for (uint t = 0; t < numTG; t++) if (pcounts[t * K + k]) { uint w = (uint)partials[t * K + k]; if (isU) mu = min(mu, w); else mi = min(mi, (int)w); }
                out[k] = isU ? (ulong)mu : (ulong)(uint)mi;
            } else {                                                 // max
                bool isU = kind == 7u;
                int mi = INT_MIN; uint mu = 0;
                for (uint t = 0; t < numTG; t++) if (pcounts[t * K + k]) { uint w = (uint)partials[t * K + k]; if (isU) mu = max(mu, w); else mi = max(mi, (int)w); }
                out[k] = isU ? (ulong)mu : (ulong)(uint)mi;
            }
        }
        // Convert device-path accumulators (lo/hi/cnt as uint) into the same output layout as gb_finalize.
        kernel void gb_pack_dev(device const uint* lo [[buffer(0)]], device const uint* hi [[buffer(1)]],
                                device const uint* cnt [[buffer(2)]], constant uint& K [[buffer(3)]],
                                device ulong* out [[buffer(4)]], device ulong* counts [[buffer(5)]],
                                uint k [[thread_position_in_grid]]) {
            if (k >= K) return;
            out[k] = ((ulong)hi[k] << 32) | (ulong)lo[k];
            counts[k] = (ulong)cnt[k];
        }
        """
        return s
    }

    private static func kernel(space: String, T: String, KT: String) -> String {
        let isPriv = space == "priv"
        let decl = isPriv
            ? "threadgroup atomic_uint lo[MAXK]; threadgroup atomic_uint hi[MAXK]; threadgroup atomic_uint cnt[MAXK];"
            : ""
        let loRef = isPriv ? "&lo[k]" : "&dlo[k]"
        let hiRef = isPriv ? "&hi[k]" : "&dhi[k]"
        let cntRef = isPriv ? "&cnt[k]" : "&dcnt[k]"
        let devArgs = isPriv ? "" : """
                                device atomic_uint* dlo [[buffer(9)]],
                                device atomic_uint* dhi [[buffer(10)]],
                                device atomic_uint* dcnt [[buffer(11)]],
        """
        return """
        kernel void gb_accumulate_\(space)(device const \(KT)* keys [[buffer(0)]],
                                device const uchar* kvalid [[buffer(1)]],
                                device const \(T)* vals [[buffer(2)]],
                                device const uchar* vvalid [[buffer(3)]],
                                constant uint& n [[buffer(4)]],
                                constant uint& flags [[buffer(5)]],       // bit0 key validity, bit1 value validity
                                constant uint& K [[buffer(6)]],
                                constant uint& kind [[buffer(7)]],
                                constant uint& chunk [[buffer(8)]],
        \(devArgs)
                                device ulong* partials [[buffer(12)]],
                                device uint* pcounts [[buffer(13)]],
                                uint lid [[thread_index_in_threadgroup]],
                                uint tgid [[threadgroup_position_in_grid]]) {
            \(decl)
            \(isPriv ? """
            uint initMin = (kind == 4u) ? (uint)INT_MAX : (kind == 6u ? UINT_MAX : (uint)f_key(INFINITY));
            uint initMax = (kind == 5u) ? (uint)INT_MIN : (kind == 7u ? 0u : (uint)f_key(-INFINITY));
            for (uint k = lid; k < K; k += TG) {
                uint init = (kind >= 4u && kind <= 8u && (kind & 1u) == 0u) ? initMin : ((kind >= 5u && kind <= 9u && (kind & 1u) == 1u) ? initMax : 0u);
                atomic_store_explicit(&lo[k], init, memory_order_relaxed);
                atomic_store_explicit(&hi[k], 0u, memory_order_relaxed);
                atomic_store_explicit(&cnt[k], 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            """ : "")
            uint start = tgid * chunk, end = min(n, start + chunk);
            for (uint i = start + lid; i < end; i += TG) {
                if ((flags & 1u) && !bit_get(kvalid, i)) continue;
                long kk = (long)keys[i];
                if (kk < 0 || kk >= (long)K) continue;
                uint k = (uint)kk;
                if ((flags & 2u) && !bit_get(vvalid, i)) continue;
                atomic_fetch_add_explicit(\(cntRef), 1u, memory_order_relaxed);
                if (kind == 3u) continue;                       // count only
                if (kind == 0u || kind == 1u) {                 // 64-bit sum as lo/hi with carry
                    long v = (long)vals[i];
                    uint vlo = (uint)v; uint vhi = (uint)((ulong)v >> 32);
                    uint old = atomic_fetch_add_explicit(\(loRef), vlo, memory_order_relaxed);
                    uint carry = (old + vlo < old) ? 1u : 0u;
                    atomic_fetch_add_explicit(\(hiRef), vhi + carry, memory_order_relaxed);
                } else if (kind == 2u) {                        // float sum via CAS on bits
                    float v = (float)vals[i];
                    uint old = atomic_load_explicit(\(loRef), memory_order_relaxed);
                    while (!atomic_compare_exchange_weak_explicit(\(loRef), &old, as_type<uint>(as_type<float>(old) + v), memory_order_relaxed, memory_order_relaxed)) {}
                } else if (kind == 4u) { atomic_fetch_min_explicit((\(isPriv ? "threadgroup" : "device") atomic_int*)\(loRef), (int)vals[i], memory_order_relaxed); }
                else if (kind == 5u) { atomic_fetch_max_explicit((\(isPriv ? "threadgroup" : "device") atomic_int*)\(loRef), (int)vals[i], memory_order_relaxed); }
                else if (kind == 6u) { atomic_fetch_min_explicit(\(loRef), (uint)vals[i], memory_order_relaxed); }
                else if (kind == 7u) { atomic_fetch_max_explicit(\(loRef), (uint)vals[i], memory_order_relaxed); }
                else if (kind == 8u) { float f = (float)vals[i]; if (!isnan(f)) atomic_fetch_min_explicit((\(isPriv ? "threadgroup" : "device") atomic_int*)\(loRef), f_key(f), memory_order_relaxed); }
                else if (kind == 9u) { float f = (float)vals[i]; if (!isnan(f)) atomic_fetch_max_explicit((\(isPriv ? "threadgroup" : "device") atomic_int*)\(loRef), f_key(f), memory_order_relaxed); }
            }
            \(isPriv ? """
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = lid; k < K; k += TG) {
                uint l = atomic_load_explicit(&lo[k], memory_order_relaxed);
                uint h = atomic_load_explicit(&hi[k], memory_order_relaxed);
                partials[tgid * K + k] = ((ulong)h << 32) | (ulong)l;
                pcounts[tgid * K + k] = atomic_load_explicit(&cnt[k], memory_order_relaxed);
            }
            """ : "")
        }

        """
    }
}
