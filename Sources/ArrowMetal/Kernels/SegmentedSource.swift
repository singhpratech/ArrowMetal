import Foundation

/// MSL for sort-based *segmented* group-by aggregation.
///
/// The atomic group-by in `GroupBySource` is limited by what Metal's atomics can express: 32-bit only,
/// so no 64-bit min/max and no Float64 anything. Sorting the keys removes the need for atomics
/// altogether — after an argsort every group is one contiguous run of the sorted order, so a
/// threadgroup can reduce its run privately and write one result. That buys Float64 sums (through the
/// software binary64 adder in `DoubleMath`), 64-bit min/max, and means without a host division.
///
/// Three kernels:
/// - `seg_bounds` turns the sorted key order into `[start, end)` per key (keys outside `[0, K)` are
///   skipped, and a key with no rows keeps the zeroed `start == end == 0`, i.e. an empty segment).
/// - `seg_reduce` reduces one segment per threadgroup: a strided pass over the run into per-thread
///   accumulators, then a tree reduction in threadgroup memory. Null values are skipped.
/// - `seg_finalize` turns the raw accumulator into the output value (identity, an inverse
///   order-preserving map for Float64 min/max, or a division by the count for a mean) and writes the
///   validity byte: a segment with no valid value is null.
enum SegmentedSource {
    /// How the raw accumulator becomes the output value.
    enum Finish: String {
        case raw          // the accumulator already is the output (integer min/max, Float64 sum bits)
        case unkeyDouble  // the accumulator is a `d_key` of the double bits
        case meanDouble   // the accumulator is a Float64 sum; divide by the count
    }

    /// One reduction shape: the MSL value type read per row, the accumulator type, the identity, how a
    /// row becomes an accumulator value, how two accumulators combine, and an optional skip predicate.
    struct Op {
        let name: String
        let valueType: String
        let accType: String
        let identity: String
        let load: String        // `v` is the loaded value of type valueType
        let combine: String     // `a` and `b` are accumulators
        let skip: String        // `v` again; "false" keeps every value
        let finish: Finish
    }

    static func sumDouble() -> Op {
        Op(name: "sum_f64", valueType: "ulong", accType: "ulong", identity: "0ul",
           load: "v", combine: "d_add(a, b)", skip: "false", finish: .raw)
    }
    static func meanDouble() -> Op {
        Op(name: "mean_f64", valueType: "ulong", accType: "ulong", identity: "0ul",
           load: "v", combine: "d_add(a, b)", skip: "false", finish: .meanDouble)
    }
    static func sumFloatAsDouble() -> Op {
        Op(name: "sum_f32d", valueType: "float", accType: "ulong", identity: "0ul",
           load: "d_from_float(v)", combine: "d_add(a, b)", skip: "false", finish: .raw)
    }
    static func meanFloat() -> Op {
        Op(name: "mean_f32", valueType: "float", accType: "ulong", identity: "0ul",
           load: "d_from_float(v)", combine: "d_add(a, b)", skip: "false", finish: .meanDouble)
    }
    /// 64-bit min/max. `signed`/`unsigned` compare the value directly; `double` compares the
    /// order-preserving key of the bit pattern and skips NaN, matching Arrow's `min_max`.
    static func minMax64(isMin: Bool, kind: String) -> Op {
        let cmp = isMin ? "min(a, b)" : "max(a, b)"
        switch kind {
        case "signed":
            return Op(name: (isMin ? "min" : "max") + "_i64", valueType: "long", accType: "long",
                      identity: isMin ? "LONG_MAX" : "LONG_MIN", load: "v", combine: cmp, skip: "false", finish: .raw)
        case "unsigned":
            return Op(name: (isMin ? "min" : "max") + "_u64", valueType: "ulong", accType: "ulong",
                      identity: isMin ? "ULONG_MAX" : "0ul", load: "v", combine: cmp, skip: "false", finish: .raw)
        default:
            return Op(name: (isMin ? "min" : "max") + "_f64", valueType: "ulong", accType: "long",
                      identity: isMin ? "LONG_MAX" : "LONG_MIN", load: "d_key((long)v)", combine: cmp,
                      skip: "d_isnan((long)v)", finish: .unkeyDouble)
        }
    }

    /// Helpers the ops above lean on: float32 -> binary64 bits, uint64 -> binary64 bits (exact for the
    /// counts we divide by, which are far below 2^53), and the inverse of the prelude's `d_key`.
    private static let helpers = """

    inline ulong d_from_float(float f) {
        uint b = as_type<uint>(f);
        ulong s = (ulong)(b >> 31);
        uint e = (b >> 23) & 0xFFu;
        uint m = b & 0x7FFFFFu;
        if (e == 0xFFu) return (s << 63) | 0x7FF0000000000000ul | ((ulong)m << 29);
        if (e == 0u) {
            if (m == 0u) return s << 63;
            int sh = 0;
            while ((m & 0x800000u) == 0u) { m <<= 1; sh++; }
            m &= 0x7FFFFFu;
            long ee = (long)(1 - 127 - sh) + 1023;
            return (s << 63) | ((ulong)ee << 52) | ((ulong)m << 29);
        }
        long ee = (long)e - 127 + 1023;
        return (s << 63) | ((ulong)ee << 52) | ((ulong)m << 29);
    }
    inline ulong d_from_ucount(ulong v) {
        if (v == 0ul) return 0ul;
        uint p = 63u;
        while (((v >> p) & 1ul) == 0ul) p--;
        ulong m = (p >= 52u) ? (v >> (p - 52u)) : (v << (52u - p));
        long e = (long)p + 1023;
        return ((ulong)e << 52) | (m & 0xFFFFFFFFFFFFFul);
    }
    inline long d_unkey(long k) { return (k < 0) ? (k ^ 0x7FFFFFFFFFFFFFFFL) : k; }
    """

    /// Segment boundaries from the sorted key order. `KT` is the key MSL type.
    static func boundsSource(KT: String) -> String { KernelSource.prelude + """

    kernel void seg_bounds(device const \(KT)* keys [[buffer(0)]],
                           device const int* ord [[buffer(1)]],
                           device const uint* nPtr [[buffer(2)]],
                           constant uint& K [[buffer(3)]],
                           device uint* segStart [[buffer(4)]],
                           device uint* segEnd [[buffer(5)]],
                           uint i [[thread_position_in_grid]]) {
        uint m = *nPtr;
        if (i >= m) return;
        long kk = (long)keys[ord[i]];
        if (kk < 0 || kk >= (long)K) return;      // out-of-range keys are skipped, as in the atomic path
        uint k = (uint)kk;
        bool first = (i == 0u) || ((long)keys[ord[i - 1u]] != kk);
        bool last = (i + 1u == m) || ((long)keys[ord[i + 1u]] != kk);
        if (first) segStart[k] = i;
        if (last) segEnd[k] = i + 1u;
    }
    """ }

    /// One reduction kernel plus its finalizer. `nPtr` carries the segment count.
    static func source(_ op: Op) -> String {
        let finishExpr: String
        switch op.finish {
        case .raw: finishExpr = "(ulong)raw"
        case .unkeyDouble: finishExpr = "(ulong)d_unkey((long)raw)"
        case .meanDouble: finishExpr = "d_div((ulong)raw, d_from_ucount(c))"
        }
        return KernelSource.prelude + DoubleMath.msl + helpers + """

        kernel void seg_reduce(device const uint* segStart [[buffer(0)]],
                               device const uint* segEnd [[buffer(1)]],
                               device const int* ord [[buffer(2)]],
                               device const \(op.valueType)* vals [[buffer(3)]],
                               device const uchar* validity [[buffer(4)]],
                               device const uint* nPtr [[buffer(5)]],
                               constant uint& hasValidity [[buffer(6)]],
                               device \(op.accType)* raws [[buffer(7)]],
                               device uint* counts [[buffer(8)]],
                               uint lid [[thread_index_in_threadgroup]],
                               uint tgid [[threadgroup_position_in_grid]]) {
            threadgroup \(op.accType) shared[TG];
            threadgroup uint scount[TG];
            uint K = *nPtr;
            uint k = tgid;
            if (k >= K) return;
            uint s = segStart[k], e = segEnd[k];
            \(op.accType) acc = \(op.identity);
            uint cnt = 0u;
            for (uint i = s + lid; i < e; i += TG) {
                uint row = (uint)ord[i];
                if (hasValidity && !bit_get(validity, row)) continue;
                \(op.valueType) v = vals[row];
                if (\(op.skip)) continue;
                \(op.accType) b = \(op.load);
                \(op.accType) a = acc;
                acc = cnt ? (\(op.combine)) : b;
                cnt++;
            }
            shared[lid] = acc; scount[lid] = cnt;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint w = TG / 2u; w > 0u; w >>= 1) {
                if (lid < w) {
                    uint cb = scount[lid + w];
                    if (cb) {
                        \(op.accType) a = shared[lid], b = shared[lid + w];
                        shared[lid] = scount[lid] ? (\(op.combine)) : b;
                        scount[lid] += cb;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lid == 0u) { raws[k] = shared[0]; counts[k] = scount[0]; }
        }

        // Raw accumulator -> output value (8 bytes) plus a validity byte per segment.
        kernel void seg_finalize(device const \(op.accType)* raws [[buffer(0)]],
                                 device const uint* counts [[buffer(1)]],
                                 device const uint* nPtr [[buffer(2)]],
                                 device ulong* out [[buffer(3)]],
                                 device uchar* validBytes [[buffer(4)]],
                                 uint k [[thread_position_in_grid]]) {
            if (k >= *nPtr) return;
            ulong c = (ulong)counts[k];
            if (c == 0ul) { out[k] = 0ul; validBytes[k] = 0; return; }
            \(op.accType) raw = raws[k];
            out[k] = \(finishExpr);
            validBytes[k] = 1;
        }
        """
    }
}
