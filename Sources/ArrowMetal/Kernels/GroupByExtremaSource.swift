import Foundation

/// MSL for **sort-free** grouped `min` / `max` over dense keys, for every element width.
///
/// The obstacle is that Metal has no 64-bit atomics, which is why `GroupBy.minMax` used to argsort the
/// key column and reduce each group's run. It does not have to: every element type has an
/// order-preserving map into a 64-bit unsigned key, and the minimum of a set of 64-bit unsigned keys can
/// be found with two passes of **32-bit** atomics.
///
/// - Pass 1 takes the minimum (and maximum) of the *high* 32 bits of the key, and counts the values.
/// - Pass 2 takes the minimum of the *low* 32 bits among only those rows whose high word already equals
///   the winning high word. Since some row attains the winning high word, and among those rows the one
///   with the smallest low word is the overall minimum, the answer is exact.
///
/// Two linear passes over the keys and the values, no sort, no reordering, no 64-bit atomic. Types four
/// bytes wide or narrower need only one pass: their whole key fits in the low word, so the high word is
/// zero for every row and pass 1 is skipped.
///
/// Contention is handled the way `GroupBySource` handles it: for `K <= 1024` each threadgroup keeps a
/// private table in threadgroup memory and merges it into the device table once at the end, so the
/// device atomics see `numThreadgroups * K` updates instead of one per row.
///
/// Nulls follow Arrow: a null key or a key outside `[0, K)` contributes nothing, null values are
/// skipped, NaN is skipped, and a key with no valid value comes back null.
enum GroupByExtremaSource {
    static let maxPrivateKeys = 1024

    /// One element type's shape: what to read, how to map it to an ordered 64-bit key, what to skip,
    /// and how the key becomes the element again — the last one so the finalize writes the output
    /// column on the GPU rather than in a host loop over the key count.
    struct Shape {
        let name: String          // pipeline cache key
        let valueType: String     // MSL type read per row
        let outType: String       // MSL type written per group (the element's storage)
        let wide: Bool            // true when the key needs both 32-bit halves
        let keyExpr: String       // `v` -> `ulong`
        let unkeyExpr: String     // `u` (ulong) -> outType
        let skipExpr: String      // `v` -> bool, "false" keeps everything
        let prelude: String       // extra MSL the expressions need
    }

    static func shape<T: ArrowPrimitive>(_: T.Type) -> Shape {
        if T.self == Double.self {
            return Shape(name: "f64", valueType: "ulong", outType: "ulong", wide: true,
                         keyExpr: "((v & 0x8000000000000000ul) ? ~v : (v | 0x8000000000000000ul))",
                         unkeyExpr: "((u & 0x8000000000000000ul) ? (u & 0x7FFFFFFFFFFFFFFFul) : ~u)",
                         skipExpr: "d_isnan((long)v)", prelude: DoubleMath.msl)
        }
        if T.isFloatingPoint {
            return Shape(name: "f32", valueType: "float", outType: "float", wide: false,
                         keyExpr: "(ulong)gx_f32key(v)", unkeyExpr: "gx_f32unkey((uint)u)",
                         skipExpr: "isnan(v)", prelude: "")
        }
        let signed = T.minValue < 0 as T
        let wide = T.byteWidth == 8
        if signed {
            return Shape(name: "i\(T.byteWidth * 8)", valueType: T.mslType, outType: T.mslType, wide: wide,
                         keyExpr: wide ? "((ulong)(long)v ^ 0x8000000000000000ul)"
                                       : "(ulong)((uint)(int)v ^ 0x80000000u)",
                         unkeyExpr: wide ? "(long)(u ^ 0x8000000000000000ul)"
                                         : "(\(T.mslType))(int)((uint)u ^ 0x80000000u)",
                         skipExpr: "false", prelude: "")
        }
        return Shape(name: "u\(T.byteWidth * 8)", valueType: T.mslType, outType: T.mslType, wide: wide,
                     keyExpr: "(ulong)v", unkeyExpr: "(\(T.mslType))u", skipExpr: "false", prelude: "")
    }

    static func source(_ s: Shape, KT: String) -> String {
        let common = KernelSource.prelude + s.prelude + """

        #define MAXK 1024u
        inline uint gx_f32key(float f) { uint b = as_type<uint>(f); return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
        inline float gx_f32unkey(uint o) { return as_type<float>((o & 0x80000000u) ? (o & 0x7FFFFFFFu) : ~o); }

        // Every group's tables back to their identities.
        kernel void gxm_init(device uint* hiMin [[buffer(0)]], device uint* hiMax [[buffer(1)]],
                             device uint* loMin [[buffer(2)]], device uint* loMax [[buffer(3)]],
                             device uint* cnt [[buffer(4)]], constant uint& K [[buffer(5)]],
                             uint k [[thread_position_in_grid]]) {
            if (k >= K) return;
            // A key four bytes wide or narrower has a high word of zero on every row, so pass 1 is
            // skipped and the identities of the high tables are that zero, not the empty extremes.
            hiMin[k] = \(s.wide ? "0xFFFFFFFFu" : "0u"); hiMax[k] = 0u;
            loMin[k] = 0xFFFFFFFFu; loMax[k] = 0u;
            cnt[k] = 0u;
        }

        // (hiMin, loMin) and (hiMax, loMax) back into the two element values, plus a validity byte.
        // Doing the inverse map here is what keeps the whole aggregate off the host: a group count of
        // ten million would otherwise be ten million iterations of a generic Swift loop.
        kernel void gxm_pack(device const uint* hiMin [[buffer(0)]], device const uint* hiMax [[buffer(1)]],
                             device const uint* loMin [[buffer(2)]], device const uint* loMax [[buffer(3)]],
                             device const uint* cnt [[buffer(4)]], constant uint& K [[buffer(5)]],
                             device \(s.outType)* outMin [[buffer(6)]], device \(s.outType)* outMax [[buffer(7)]],
                             device uchar* valid [[buffer(8)]],
                             uint k [[thread_position_in_grid]]) {
            if (k >= K) return;
            uint c = cnt[k];
            valid[k] = c ? 1 : 0;
            if (!c) { outMin[k] = (\(s.outType))0; outMax[k] = (\(s.outType))0; return; }
            ulong u = ((ulong)hiMin[k] << 32) | (ulong)loMin[k];
            outMin[k] = \(s.unkeyExpr);
            u = ((ulong)hiMax[k] << 32) | (ulong)loMax[k];
            outMax[k] = \(s.unkeyExpr);
        }

        """
        var out = common
        for space in ["priv", "dev"] {
            if s.wide { out += pass1(space: space, s: s, KT: KT) }
            out += pass2(space: space, s: s, KT: KT, countHere: !s.wide)
        }
        return out
    }

    /// The row loop shared by both passes: skip invalid keys and values, produce `k` and `u`.
    private static func rowPrologue(_ s: Shape, KT: String) -> String { """
                if ((flags & 1u) && !bit_get(kvalid, i)) continue;
                long kk = (long)keys[i];
                if (kk < 0 || kk >= (long)K) continue;
                uint k = (uint)kk;
                if ((flags & 2u) && !bit_get(vvalid, i)) continue;
                \(s.valueType) v = vals[i];
                if (\(s.skipExpr)) continue;
                ulong u = \(s.keyExpr);
    """ }

    private static func args(_ s: Shape, KT: String) -> String { """
                                 device const \(KT)* keys [[buffer(0)]],
                                 device const uchar* kvalid [[buffer(1)]],
                                 device const \(s.valueType)* vals [[buffer(2)]],
                                 device const uchar* vvalid [[buffer(3)]],
                                 constant uint& n [[buffer(4)]],
                                 constant uint& flags [[buffer(5)]],
                                 constant uint& K [[buffer(6)]],
                                 constant uint& chunk [[buffer(7)]],
    """ }

    /// Pass 1: the extremes of the high word, and the count of contributing values.
    private static func pass1(space: String, s: Shape, KT: String) -> String {
        let priv = space == "priv"
        let decl = priv ? "threadgroup atomic_uint tMin[MAXK]; threadgroup atomic_uint tMax[MAXK]; threadgroup atomic_uint tCnt[MAXK];" : ""
        let minRef = priv ? "&tMin[k]" : "&hiMin[k]"
        let maxRef = priv ? "&tMax[k]" : "&hiMax[k]"
        let cntRef = priv ? "&tCnt[k]" : "&cnt[k]"
        return """
        kernel void gxm_hi_\(space)(\(args(s, KT: KT))
                                 device atomic_uint* hiMin [[buffer(8)]],
                                 device atomic_uint* hiMax [[buffer(9)]],
                                 device atomic_uint* cnt [[buffer(10)]],
                                 uint lid [[thread_index_in_threadgroup]],
                                 uint tgid [[threadgroup_position_in_grid]]) {
            \(decl)
            \(priv ? """
            for (uint k = lid; k < K; k += TG) {
                atomic_store_explicit(&tMin[k], 0xFFFFFFFFu, memory_order_relaxed);
                atomic_store_explicit(&tMax[k], 0u, memory_order_relaxed);
                atomic_store_explicit(&tCnt[k], 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            """ : "")
            uint start = tgid * chunk, end = min(n, start + chunk);
            for (uint i = start + lid; i < end; i += TG) {
        \(rowPrologue(s, KT: KT))
                uint hi = (uint)(u >> 32);
                atomic_fetch_min_explicit(\(minRef), hi, memory_order_relaxed);
                atomic_fetch_max_explicit(\(maxRef), hi, memory_order_relaxed);
                atomic_fetch_add_explicit(\(cntRef), 1u, memory_order_relaxed);
            }
            \(priv ? """
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = lid; k < K; k += TG) {
                uint c = atomic_load_explicit(&tCnt[k], memory_order_relaxed);
                if (!c) continue;
                atomic_fetch_min_explicit(&hiMin[k], atomic_load_explicit(&tMin[k], memory_order_relaxed), memory_order_relaxed);
                atomic_fetch_max_explicit(&hiMax[k], atomic_load_explicit(&tMax[k], memory_order_relaxed), memory_order_relaxed);
                atomic_fetch_add_explicit(&cnt[k], c, memory_order_relaxed);
            }
            """ : "")
        }

        """
    }

    /// Pass 2: the extremes of the low word among the rows that already hold the winning high word.
    /// When the key is 32 bits or narrower the high word is zero everywhere, so this pass also counts.
    private static func pass2(space: String, s: Shape, KT: String, countHere: Bool) -> String {
        let priv = space == "priv"
        let decl = priv
            ? "threadgroup atomic_uint tMin[MAXK]; threadgroup atomic_uint tMax[MAXK]; threadgroup uint sHiMin[MAXK]; threadgroup uint sHiMax[MAXK];"
              + (countHere ? " threadgroup atomic_uint tCnt[MAXK];" : "")
            : ""
        let minRef = priv ? "&tMin[k]" : "&loMin[k]"
        let maxRef = priv ? "&tMax[k]" : "&loMax[k]"
        let hiMinRef = priv ? "sHiMin[k]" : "hiMin[k]"
        let hiMaxRef = priv ? "sHiMax[k]" : "hiMax[k]"
        let cntRef = priv ? "&tCnt[k]" : "&cnt[k]"
        return """
        kernel void gxm_lo_\(space)(\(args(s, KT: KT))
                                 device const uint* hiMin [[buffer(8)]],
                                 device const uint* hiMax [[buffer(9)]],
                                 device atomic_uint* loMin [[buffer(10)]],
                                 device atomic_uint* loMax [[buffer(11)]],
                                 device atomic_uint* cnt [[buffer(12)]],
                                 uint lid [[thread_index_in_threadgroup]],
                                 uint tgid [[threadgroup_position_in_grid]]) {
            \(decl)
            \(priv ? """
            for (uint k = lid; k < K; k += TG) {
                atomic_store_explicit(&tMin[k], 0xFFFFFFFFu, memory_order_relaxed);
                atomic_store_explicit(&tMax[k], 0u, memory_order_relaxed);
                sHiMin[k] = hiMin[k]; sHiMax[k] = hiMax[k];
                \(countHere ? "atomic_store_explicit(&tCnt[k], 0u, memory_order_relaxed);" : "")
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            """ : "")
            uint start = tgid * chunk, end = min(n, start + chunk);
            for (uint i = start + lid; i < end; i += TG) {
        \(rowPrologue(s, KT: KT))
                uint hi = (uint)(u >> 32), lo = (uint)u;
                \(countHere ? "atomic_fetch_add_explicit(\(cntRef), 1u, memory_order_relaxed);" : "")
                if (hi == \(hiMinRef)) atomic_fetch_min_explicit(\(minRef), lo, memory_order_relaxed);
                if (hi == \(hiMaxRef)) atomic_fetch_max_explicit(\(maxRef), lo, memory_order_relaxed);
            }
            \(priv ? """
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = lid; k < K; k += TG) {
                uint a = atomic_load_explicit(&tMin[k], memory_order_relaxed);
                uint b = atomic_load_explicit(&tMax[k], memory_order_relaxed);
                if (a != 0xFFFFFFFFu || b != 0u) {
                    atomic_fetch_min_explicit(&loMin[k], a, memory_order_relaxed);
                    atomic_fetch_max_explicit(&loMax[k], b, memory_order_relaxed);
                }
                \(countHere ? """
                uint c = atomic_load_explicit(&tCnt[k], memory_order_relaxed);
                if (c) atomic_fetch_add_explicit(&cnt[k], c, memory_order_relaxed);
                """ : "")
            }
            """ : "")
        }

        """
    }
}
