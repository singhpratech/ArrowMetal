import Foundation

/// MSL for **sort-free** grouped `min` / `max` over dense keys, for every element width.
///
/// The obstacle is that Metal has no 64-bit atomics, which is why `GroupBy.minMax` used to argsort the
/// key column and reduce each group's run. It does not have to: every element type has an
/// order-preserving map into a 64-bit unsigned key, and the minimum of a set of 64-bit unsigned keys can
/// be found with two passes of **32-bit** atomics.
///
/// - Pass 1 takes the minimum (and maximum) of the *high* 32 bits of the key.
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
/// The device table keeps a group's four words together (`hiMin, hiMax, loMin, loMax`, 16 bytes), so
/// a row touches one cache line rather than one in each of four arrays. A row reads its group's words
/// before it writes them and issues an atomic only when it improves on what it read: once a group has
/// seen a few rows almost no row does, so almost every row costs a read instead of a read-modify-write.
/// A stale read only costs an atomic that changes nothing. No count is kept: a group has a value
/// exactly when its minimum key is at or below its maximum key, which the identities (all ones for the
/// minimum, zero for the maximum) never satisfy.
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

        // Every group's four words back to their identities. A key four bytes wide or narrower has a
        // high word of zero on every row, so pass 1 is skipped and its high words start at that zero.
        kernel void gxm_init(device uint4* t [[buffer(0)]], constant uint& K [[buffer(1)]],
                             uint k [[thread_position_in_grid]]) {
            if (k >= K) return;
            t[k] = uint4(\(s.wide ? "0xFFFFFFFFu" : "0u"), 0u, 0xFFFFFFFFu, 0u);
        }

        // (hiMin, loMin) and (hiMax, loMax) back into the two element values, plus a validity byte.
        // Doing the inverse map here is what keeps the whole aggregate off the host: a group count of
        // ten million would otherwise be ten million iterations of a generic Swift loop.
        kernel void gxm_pack(device const uint4* t [[buffer(0)]], constant uint& K [[buffer(1)]],
                             device \(s.outType)* outMin [[buffer(2)]], device \(s.outType)* outMax [[buffer(3)]],
                             device uchar* valid [[buffer(4)]],
                             uint k [[thread_position_in_grid]]) {
            if (k >= K) return;
            uint4 g = t[k];
            ulong umin = ((ulong)g.x << 32) | (ulong)g.z, umax = ((ulong)g.y << 32) | (ulong)g.w;
            bool c = umin <= umax;
            valid[k] = c ? 1 : 0;
            if (!c) { outMin[k] = (\(s.outType))0; outMax[k] = (\(s.outType))0; return; }
            ulong u = umin;
            outMin[k] = \(s.unkeyExpr);
            u = umax;
            outMax[k] = \(s.unkeyExpr);
        }

        """
        var out = common
        for space in ["priv", "dev"] {
            if s.wide { out += pass1(space: space, s: s, KT: KT) }
            out += pass2(space: space, s: s, KT: KT)
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
                                 device atomic_uint* t [[buffer(8)]],
                                 uint lid [[thread_index_in_threadgroup]],
                                 uint tgid [[threadgroup_position_in_grid]]
    """ }

    /// `atomic_fetch_min` / `max` of `x` into `p`, issued only when `x` improves on what `p` holds now.
    private static func improve(_ op: String, _ p: String, _ x: String) -> String {
        let cmp = op == "min" ? "<" : ">"
        return "if (\(x) \(cmp) atomic_load_explicit(\(p), memory_order_relaxed)) atomic_fetch_\(op)_explicit(\(p), \(x), memory_order_relaxed);"
    }

    /// Pass 1: the extremes of the high word.
    private static func pass1(space: String, s: Shape, KT: String) -> String {
        let priv = space == "priv"
        let decl = priv ? "threadgroup atomic_uint tMin[MAXK]; threadgroup atomic_uint tMax[MAXK];" : ""
        let minRef = priv ? "&tMin[k]" : "&g[0]"
        let maxRef = priv ? "&tMax[k]" : "&g[1]"
        return """
        kernel void gxm_hi_\(space)(\(args(s, KT: KT))) {
            \(decl)
            \(priv ? """
            for (uint k = lid; k < K; k += TG) {
                atomic_store_explicit(&tMin[k], 0xFFFFFFFFu, memory_order_relaxed);
                atomic_store_explicit(&tMax[k], 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            """ : "")
            uint start = tgid * chunk, len = (start < n) ? min(chunk, n - start) : 0u;
            for (uint off = lid; off < len; off += TG) { uint i = start + off;
        \(rowPrologue(s, KT: KT))
                uint hi = (uint)(u >> 32);
                \(priv ? "" : "device atomic_uint* g = &t[(ulong)k * 4ul];")
                \(improve("min", minRef, "hi"))
                \(improve("max", maxRef, "hi"))
            }
            \(priv ? """
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = lid; k < K; k += TG) {
                uint a = atomic_load_explicit(&tMin[k], memory_order_relaxed);
                uint b = atomic_load_explicit(&tMax[k], memory_order_relaxed);
                if (a > b) continue;                          // no value of this key in this threadgroup
                device atomic_uint* g = &t[(ulong)k * 4ul];
                \(improve("min", "&g[0]", "a"))
                \(improve("max", "&g[1]", "b"))
            }
            """ : "")
        }

        """
    }

    /// Pass 2: the extremes of the low word among the rows that already hold the winning high word.
    /// When the key is 32 bits or narrower the high word is zero everywhere, so every row takes part.
    private static func pass2(space: String, s: Shape, KT: String) -> String {
        let priv = space == "priv"
        let decl = priv
            ? "threadgroup atomic_uint tMin[MAXK]; threadgroup atomic_uint tMax[MAXK]; threadgroup uint sHiMin[MAXK]; threadgroup uint sHiMax[MAXK];"
            : ""
        let minRef = priv ? "&tMin[k]" : "&g[2]"
        let maxRef = priv ? "&tMax[k]" : "&g[3]"
        // The high words are final after pass 1 and this pass never writes them, so they are read
        // through a plain (cacheable) view of the table, `th`, rather than as atomics: at a few
        // thousand groups the table stays in cache and an atomic load would go past it on every row.
        let hiMin = priv ? "sHiMin[k]" : "hw.x"
        let hiMax = priv ? "sHiMax[k]" : "hw.y"
        return """
        kernel void gxm_lo_\(space)(\(args(s, KT: KT)),
                                 device const uint2* th [[buffer(9)]]) {
            \(decl)
            \(priv ? """
            for (uint k = lid; k < K; k += TG) {
                atomic_store_explicit(&tMin[k], 0xFFFFFFFFu, memory_order_relaxed);
                atomic_store_explicit(&tMax[k], 0u, memory_order_relaxed);
                uint2 hw = th[(ulong)k * 2ul];
                sHiMin[k] = hw.x; sHiMax[k] = hw.y;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            """ : "")
            uint start = tgid * chunk, len = (start < n) ? min(chunk, n - start) : 0u;
            for (uint off = lid; off < len; off += TG) { uint i = start + off;
        \(rowPrologue(s, KT: KT))
                uint hi = (uint)(u >> 32), lo = (uint)u;
                \(priv ? "" : "device atomic_uint* g = &t[(ulong)k * 4ul]; uint2 hw = th[(ulong)k * 2ul];")
                if (hi == \(hiMin)) { \(improve("min", minRef, "lo")) }
                if (hi == \(hiMax)) { \(improve("max", maxRef, "lo")) }
            }
            \(priv ? """
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = lid; k < K; k += TG) {
                uint a = atomic_load_explicit(&tMin[k], memory_order_relaxed);
                uint b = atomic_load_explicit(&tMax[k], memory_order_relaxed);
                device atomic_uint* g = &t[(ulong)k * 4ul];
                if (a != 0xFFFFFFFFu) { \(improve("min", "&g[2]", "a")) }
                if (b != 0u) { \(improve("max", "&g[3]", "b")) }
            }
            """ : "")
        }

        """
    }
}
