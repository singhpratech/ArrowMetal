import Foundation

/// MSL for the grouped Float64 `sum` and `mean` that need no group order: **correctly rounded**,
/// from two linear passes of 32-bit atomics.
///
/// The segmented reduction this replaces summed each group in a fixed tree order, which is why it
/// needed every group's rows together first (a counting sort by group id, then a gather). A correctly
/// rounded sum does not depend on the order of its terms, so the rows can be added where they lie:
///
/// - **Pass A** takes, per group, the largest exponent of its finite non-zero values (an atomic max of
///   the 11-bit field), counts the non-null values, and records the special values it met (NaN, ±inf,
///   and whether a +0.0 was seen) in a flags word.
/// - **Pass B** turns each finite value into an integer in a per-group fixed-point window whose unit is
///   `2^(Emax - G)` of the group's largest exponent, and adds it to a 128-bit two's-complement
///   accumulator with 32-bit atomics that carry (or borrow) into the next word. `G`, the window's guard
///   bits below the largest value's last bit, is `74 - bitlength(rows)`: 53 significand bits plus `G`
///   plus `bitlength(rows)` of headroom stay below the sign bit, so no group's sum can overflow it. A
///   value whose last bits fall below the window is truncated towards zero and counted in `drops`.
/// - **Finalize** (a thread per group) rounds the accumulator to nearest-even once. With no drops the
///   accumulator is the exact sum, so the result is the correctly rounded sum; the mean is the
///   correctly rounded quotient of that exact sum by the count (a long division of the 128-bit value).
///   With `d` drops the exact sum lies strictly between `W - d` and `W + d` units, and the result is
///   taken only when both ends round to the same double. A group whose rounding the window cannot
///   settle, or that holds a NaN other than the canonical quiet NaN, is flagged; the host sums those
///   groups exactly (`GroupSumExact.hostFallback`).
///
/// Special values follow IEEE addition: any NaN makes the sum NaN (a group whose NaNs are all the
/// canonical `0x7FF8000000000000` returns it; any other NaN goes to the host, which returns the first
/// NaN in row order, quieted); +inf with -inf is the canonical quiet NaN; otherwise an infinity wins.
/// An exact sum of zero is +0.0 unless every value of the group is -0.0.
///
/// Per group the state is eight words: `emax, cnt, flags, drops, acc[4]` (acc little-endian).
enum GroupSumExactSource {
    static let maxPrivateKeys = 1024
    static let words = 8

    /// Flags word bits.
    static let flagNaN: UInt32 = 1          // the canonical quiet NaN
    static let flagPosInf: UInt32 = 2
    static let flagNegInf: UInt32 = 4
    static let flagPosZero: UInt32 = 8
    static let flagOtherNaN: UInt32 = 16    // any other NaN bit pattern

    static func source(KT: String) -> String {
        var s = KernelSource.prelude + """

        #define GS_W 8u
        #define MAXK 1024u

        // ---------------------------------------------------------------- 128-bit atomic add / subtract

        """
        for space in ["device", "threadgroup"] {
            let sfx = space == "device" ? "d" : "t"
            s += """

            // Adds (neg == false) or subtracts the 128-bit magnitude (lo, hi) at a[0..3], carrying (or
            // borrowing) word by word. Each word's carry-out comes from its own read-modify-write, so the
            // words hold the exact sum modulo 2^128 however the rows interleave.
            inline void gs_acc_\(sfx)(\(space) atomic_uint* a, ulong lo, ulong hi, bool neg) {
                uint w[4] = { (uint)lo, (uint)(lo >> 32), (uint)hi, (uint)(hi >> 32) };
                uint c = 0u;
                for (uint j = 0u; j < 4u; ++j) {
                    ulong t = (ulong)w[j] + (ulong)c;
                    uint x = (uint)t;
                    c = (uint)(t >> 32);
                    if (x == 0u) continue;
                    if (!neg) {
                        uint old = atomic_fetch_add_explicit(&a[j], x, memory_order_relaxed);
                        c += (old > 0xFFFFFFFFu - x) ? 1u : 0u;
                    } else {
                        uint old = atomic_fetch_sub_explicit(&a[j], x, memory_order_relaxed);
                        c += (old < x) ? 1u : 0u;
                    }
                }
            }

            """
        }
        s += """

        // The fixed-point integer of a finite non-zero value in a window of unit exponent U (biased,
        // >= 1), returned as (lo, hi). `dropped` says whether bits fell below the window.
        inline void gs_fixed(ulong v, uint U, thread ulong& lo, thread ulong& hi, thread bool& dropped) {
            uint e = (uint)((v >> 52) & 0x7FFul);
            ulong m = (v & 0xFFFFFFFFFFFFFul) | (e ? (1ul << 52) : 0ul);
            int sh = (int)(e ? e : 1u) - (int)U;
            dropped = false;
            if (sh >= 64) { lo = 0ul; hi = m << (uint)(sh - 64); }
            else if (sh > 0) { lo = m << (uint)sh; hi = m >> (uint)(64 - sh); }
            else if (sh == 0) { lo = m; hi = 0ul; }
            else {
                uint r = (uint)(-sh);
                hi = 0ul;
                if (r >= 64u) { lo = 0ul; dropped = true; }
                else { lo = m >> r; dropped = (m & ((1ul << r) - 1ul)) != 0ul; }
            }
        }

        inline uint gs_window(uint emax, uint G) { return (emax > G + 1u) ? emax - G : 1u; }

        // ---------------------------------------------------------------- rounding

        inline ulong gs_bits64(thread const uint* w, uint nw, int pos) {   // bits [pos, pos + 64)
            uint j = (uint)pos >> 5, b = (uint)pos & 31u;
            ulong r = 0ul;
            for (uint t = 0u; t < 3u; ++t) {
                uint jj = j + t;
                ulong x = (jj < nw) ? (ulong)w[jj] : 0ul;
                int at = (int)(t * 32u) - (int)b;
                if (at >= 0) { if (at < 64) r |= x << (uint)at; }
                else r |= x >> (uint)(-at);
            }
            return r;
        }
        inline bool gs_any_below(thread const uint* w, uint nw, int pos) {  // any bit in [0, pos)
            if (pos <= 0) return false;
            uint j = (uint)pos >> 5, b = (uint)pos & 31u;
            for (uint t = 0u; t < min(j, nw); ++t) if (w[t]) return true;
            return (b && j < nw) ? ((w[j] & ((1u << b) - 1u)) != 0u) : false;
        }

        // Round-to-nearest-even of (w as an unsigned integer, plus a sticky fraction) * 2^(lsbPos-1074),
        // as binary64 bits with the sign of `neg`. `w` must not be zero.
        inline ulong gs_round(thread const uint* w, uint nw, int lsbPos, bool sticky, bool neg) {
            int top = -1;
            for (int j = (int)nw - 1; j >= 0; --j) if (w[j]) { top = j * 32 + 31 - (int)clz(w[j]); break; }
            ulong sgn = neg ? 0x8000000000000000ul : 0ul;
            int P = top + lsbPos;
            if (P - 52 >= 2046) return sgn | 0x7FF0000000000000ul;
            int s = (P >= 52) ? (top - 52) : (-lsbPos);
            ulong r;
            if (s <= 0) { r = gs_bits64(w, nw, 0) << (uint)(-s); }
            else {
                ulong x = gs_bits64(w, nw, s - 1);
                bool rb = (x & 1ul) != 0ul;
                r = (x >> 1) & ((1ul << 54) - 1ul);
                bool st = sticky || gs_any_below(w, nw, s - 1);
                if (rb && (st || (r & 1ul))) r += 1ul;
            }
            ulong bits = (P >= 52) ? r + ((ulong)(P - 52) << 52) : r;
            if (bits >= 0x7FF0000000000000ul) bits = 0x7FF0000000000000ul;
            return sgn | bits;
        }

        // Sum (or mean when cnt != 0) of the signed 128-bit W * 2^(lsbPos-1074). A zero W gives +0.0.
        inline ulong gs_value(ulong lo, ulong hi, int lsbPos, uint cnt) {
            bool neg = (hi >> 63) != 0ul;
            if (neg) { lo = ~lo + 1ul; hi = ~hi + (lo == 0ul ? 1ul : 0ul); }
            if (lo == 0ul && hi == 0ul) return 0ul;
            // A sum, or a mean over a power-of-two count: W * 2^-j is the same integer one scale lower,
            // rounded once, so no division is needed (the common case of groups of one or two rows).
            if (cnt == 0u || popcount(cnt) == 1u) {
                uint w[4] = { (uint)lo, (uint)(lo >> 32), (uint)hi, (uint)(hi >> 32) };
                return gs_round(w, 4u, lsbPos - (cnt ? (int)ctz(cnt) : 0), false, neg);
            }
            // (W << 128) / cnt: the quotient keeps at least 96 significant bits, the remainder is sticky.
            uint d[8] = { 0u, 0u, 0u, 0u, (uint)lo, (uint)(lo >> 32), (uint)hi, (uint)(hi >> 32) };
            uint q[8];
            ulong rem = 0ul;
            for (int j = 7; j >= 0; --j) {
                ulong cur = (rem << 32) | (ulong)d[j];
                q[j] = (uint)(cur / (ulong)cnt);
                rem = cur % (ulong)cnt;
            }
            return gs_round(q, 8u, lsbPos - 128, rem != 0ul, neg);
        }

        """
        for space in ["dev", "priv"] {
            s += passA(space: space, KT: KT) + passB(space: space, KT: KT)
        }
        s += """

        kernel void gs_init(device uint4* st [[buffer(0)]], constant uint& K [[buffer(1)]],
                            uint k [[thread_position_in_grid]]) {
            if (k >= K) return;
            st[2u * k] = uint4(0u); st[2u * k + 1u] = uint4(0u);
        }

        // One thread per group: the sum and the mean, their validity, and the groups left to the host.
        kernel void gs_finalize(device const uint* st [[buffer(0)]],
                                constant uint& K [[buffer(1)]],
                                constant uint& G [[buffer(2)]],
                                device ulong* outSum [[buffer(3)]],
                                device ulong* outMean [[buffer(4)]],
                                device uchar* valid [[buffer(5)]],
                                device uchar* hostGroup [[buffer(6)]],
                                device atomic_uint* hostCount [[buffer(7)]],
                                uint k [[thread_position_in_grid]]) {
            if (k >= K) return;
            device const uint* g = st + (ulong)k * GS_W;
            uint emax = g[0], cnt = g[1], flags = g[2], drops = g[3];
            hostGroup[k] = 0;
            if (cnt == 0u) { outSum[k] = 0ul; outMean[k] = 0ul; valid[k] = 0; return; }
            valid[k] = 1;
            if (flags & \(flagOtherNaN)u) {
                outSum[k] = 0ul; outMean[k] = 0ul; hostGroup[k] = 1;
                atomic_fetch_add_explicit(&hostCount[0], 1u, memory_order_relaxed);
                atomic_fetch_add_explicit(&hostCount[1], cnt, memory_order_relaxed);
                return;
            }
            if (flags & (\(flagNaN)u | \(flagPosInf)u | \(flagNegInf)u)) {
                ulong r;
                if (flags & \(flagNaN)u) r = 0x7FF8000000000000ul;
                else if ((flags & \(flagPosInf)u) && (flags & \(flagNegInf)u)) r = 0x7FF8000000000000ul;
                else r = (flags & \(flagPosInf)u) ? 0x7FF0000000000000ul : 0xFFF0000000000000ul;
                outSum[k] = r; outMean[k] = r;
                return;
            }
            if (emax == 0u) {                                 // every value is a zero
                ulong z = (flags & \(flagPosZero)u) ? 0ul : 0x8000000000000000ul;
                outSum[k] = z; outMean[k] = z;
                return;
            }
            ulong lo = (ulong)g[4] | ((ulong)g[5] << 32), hi = (ulong)g[6] | ((ulong)g[7] << 32);
            int lsb = (int)gs_window(emax, G) - 1;
            if (drops == 0u) {
                outSum[k] = gs_value(lo, hi, lsb, 0u);
                outMean[k] = gs_value(lo, hi, lsb, cnt);
                return;
            }
            // The exact sum lies strictly between W - drops and W + drops units.
            ulong dlo = lo - (ulong)drops, dhi = hi - ((lo < (ulong)drops) ? 1ul : 0ul);
            ulong ulo = lo + (ulong)drops, uhi = hi + ((ulo < lo) ? 1ul : 0ul);
            ulong s0 = gs_value(dlo, dhi, lsb, 0u), s1 = gs_value(ulo, uhi, lsb, 0u);
            ulong m0 = gs_value(dlo, dhi, lsb, cnt), m1 = gs_value(ulo, uhi, lsb, cnt);
            bool zero0 = (dlo == 0ul && dhi == 0ul), zero1 = (ulo == 0ul && uhi == 0ul);
            if (s0 == s1 && m0 == m1 && !zero0 && !zero1) { outSum[k] = s0; outMean[k] = m0; return; }
            outSum[k] = 0ul; outMean[k] = 0ul; hostGroup[k] = 1;
            atomic_fetch_add_explicit(&hostCount[0], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&hostCount[1], cnt, memory_order_relaxed);
        }

        // The rows of the groups left to the host, in no particular order.
        kernel void gs_host_rows(device const \(KT)* keys [[buffer(0)]],
                                 device const uchar* kvalid [[buffer(1)]],
                                 device const uchar* vvalid [[buffer(2)]],
                                 constant uint& n [[buffer(3)]],
                                 constant uint& flags [[buffer(4)]],
                                 constant uint& K [[buffer(5)]],
                                 device const uchar* hostGroup [[buffer(6)]],
                                 device atomic_uint* cursor [[buffer(7)]],
                                 device uint* rows [[buffer(8)]],
                                 uint i [[thread_position_in_grid]]) {
            if (i >= n) return;
            if ((flags & 1u) && !bit_get(kvalid, i)) return;
            long kk = (long)keys[i];
            if (kk < 0 || kk >= (long)K) return;
            if ((flags & 2u) && !bit_get(vvalid, i)) return;
            if (!hostGroup[(uint)kk]) return;
            rows[atomic_fetch_add_explicit(cursor, 1u, memory_order_relaxed)] = i;
        }

        """
        return s
    }

    private static func args(KT: String) -> String { """
                                device const \(KT)* keys [[buffer(0)]],
                                device const uchar* kvalid [[buffer(1)]],
                                device const ulong* vals [[buffer(2)]],
                                device const uchar* vvalid [[buffer(3)]],
                                constant uint& n [[buffer(4)]],
                                constant uint& flags [[buffer(5)]],
                                constant uint& K [[buffer(6)]],
                                constant uint& chunk [[buffer(7)]],
                                constant uint& G [[buffer(8)]],
                                device atomic_uint* st [[buffer(9)]],
                                uint lid [[thread_index_in_threadgroup]],
                                uint tgid [[threadgroup_position_in_grid]]
    """ }

    private static let rowPrologue = """
                if ((flags & 1u) && !bit_get(kvalid, i)) continue;
                long kk = (long)keys[i];
                if (kk < 0 || kk >= (long)K) continue;
                uint k = (uint)kk;
                if ((flags & 2u) && !bit_get(vvalid, i)) continue;
                ulong v = vals[i];
                uint e = (uint)((v >> 52) & 0x7FFul);
                ulong mag = v & 0x7FFFFFFFFFFFFFFFul;
    """

    /// The flags a special or zero value sets; `F` is the flags word's atomic.
    private static func specials(_ f: String) -> String { """
                if (e == 0x7FFu) {
                    uint b = (mag == 0x7FF0000000000000ul) ? ((v >> 63) ? \(flagNegInf)u : \(flagPosInf)u)
                           : ((v == 0x7FF8000000000000ul) ? \(flagNaN)u : \(flagOtherNaN)u);
                    atomic_fetch_or_explicit(\(f), b, memory_order_relaxed);
                    continue;
                }
                if (mag == 0ul) {
                    if (v == 0ul && !(atomic_load_explicit(\(f), memory_order_relaxed) & \(flagPosZero)u))
                        atomic_fetch_or_explicit(\(f), \(flagPosZero)u, memory_order_relaxed);
                    continue;
                }
    """ }

    private static func passA(space: String, KT: String) -> String {
        let priv = space == "priv"
        if !priv {
            return """

            kernel void gs_a_dev(\(args(KT: KT))) {
                uint start = tgid * chunk, end = min(n, start + chunk);
                for (uint i = start + lid; i < end; i += TG) {
            \(rowPrologue)
                    device atomic_uint* g = st + (ulong)k * GS_W;
                    atomic_fetch_add_explicit(&g[1], 1u, memory_order_relaxed);
            \(specials("&g[2]"))
                    uint ee = e ? e : 1u;
                    if (ee > atomic_load_explicit(&g[0], memory_order_relaxed))
                        atomic_fetch_max_explicit(&g[0], ee, memory_order_relaxed);
                }
            }

            """
        }
        return """

        kernel void gs_a_priv(\(args(KT: KT))) {
            threadgroup atomic_uint tE[MAXK], tC[MAXK], tF[MAXK];
            for (uint k = lid; k < K; k += TG) {
                atomic_store_explicit(&tE[k], 0u, memory_order_relaxed);
                atomic_store_explicit(&tC[k], 0u, memory_order_relaxed);
                atomic_store_explicit(&tF[k], 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint start = tgid * chunk, end = min(n, start + chunk);
            for (uint i = start + lid; i < end; i += TG) {
        \(rowPrologue)
                atomic_fetch_add_explicit(&tC[k], 1u, memory_order_relaxed);
        \(specials("&tF[k]"))
                uint ee = e ? e : 1u;
                if (ee > atomic_load_explicit(&tE[k], memory_order_relaxed))
                    atomic_fetch_max_explicit(&tE[k], ee, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = lid; k < K; k += TG) {
                uint c = atomic_load_explicit(&tC[k], memory_order_relaxed);
                if (!c) continue;
                device atomic_uint* g = st + (ulong)k * GS_W;
                atomic_fetch_add_explicit(&g[1], c, memory_order_relaxed);
                uint e = atomic_load_explicit(&tE[k], memory_order_relaxed);
                if (e) atomic_fetch_max_explicit(&g[0], e, memory_order_relaxed);
                uint f = atomic_load_explicit(&tF[k], memory_order_relaxed);
                if (f) atomic_fetch_or_explicit(&g[2], f, memory_order_relaxed);
            }
        }

        """
    }

    private static func passB(space: String, KT: String) -> String {
        let priv = space == "priv"
        let body = """
                if (e == 0x7FFu || mag == 0ul) continue;
        """
        if !priv {
            return """

            kernel void gs_b_dev(\(args(KT: KT))) {
                uint start = tgid * chunk, end = min(n, start + chunk);
                for (uint i = start + lid; i < end; i += TG) {
            \(rowPrologue)
            \(body)
                    device atomic_uint* g = st + (ulong)k * GS_W;
                    if (atomic_load_explicit(&g[2], memory_order_relaxed) & 0x17u) continue;   // result is special
                    uint U = gs_window(atomic_load_explicit(&g[0], memory_order_relaxed), G);
                    ulong lo, hi; bool dropped;
                    gs_fixed(v, U, lo, hi, dropped);
                    if (dropped) atomic_fetch_add_explicit(&g[3], 1u, memory_order_relaxed);
                    if (lo == 0ul && hi == 0ul) continue;
                    gs_acc_d(&g[4], lo, hi, (v >> 63) != 0ul);
                }
            }

            """
        }
        return """

        kernel void gs_b_priv(\(args(KT: KT))) {
            threadgroup atomic_uint tA[MAXK * 4u];
            threadgroup atomic_uint tD[MAXK];
            threadgroup uint tU[MAXK];
            for (uint k = lid; k < K; k += TG) {
                for (uint j = 0u; j < 4u; ++j) atomic_store_explicit(&tA[k * 4u + j], 0u, memory_order_relaxed);
                atomic_store_explicit(&tD[k], 0u, memory_order_relaxed);
                device atomic_uint* g = st + (ulong)k * GS_W;
                bool special = (atomic_load_explicit(&g[2], memory_order_relaxed) & 0x17u) != 0u;
                tU[k] = special ? 0u : gs_window(atomic_load_explicit(&g[0], memory_order_relaxed), G);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint start = tgid * chunk, end = min(n, start + chunk);
            for (uint i = start + lid; i < end; i += TG) {
        \(rowPrologue)
        \(body)
                uint U = tU[k];
                if (U == 0u) continue;
                ulong lo, hi; bool dropped;
                gs_fixed(v, U, lo, hi, dropped);
                if (dropped) atomic_fetch_add_explicit(&tD[k], 1u, memory_order_relaxed);
                if (lo == 0ul && hi == 0ul) continue;
                gs_acc_t(&tA[k * 4u], lo, hi, (v >> 63) != 0ul);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = lid; k < K; k += TG) {
                device atomic_uint* g = st + (ulong)k * GS_W;
                uint d = atomic_load_explicit(&tD[k], memory_order_relaxed);
                if (d) atomic_fetch_add_explicit(&g[3], d, memory_order_relaxed);
                uint a0 = atomic_load_explicit(&tA[k * 4u + 0u], memory_order_relaxed);
                uint a1 = atomic_load_explicit(&tA[k * 4u + 1u], memory_order_relaxed);
                uint a2 = atomic_load_explicit(&tA[k * 4u + 2u], memory_order_relaxed);
                uint a3 = atomic_load_explicit(&tA[k * 4u + 3u], memory_order_relaxed);
                if ((a0 | a1 | a2 | a3) == 0u) continue;
                // The partial is a 128-bit two's-complement value: adding it modulo 2^128 is exact.
                gs_acc_d(&g[4], (ulong)a0 | ((ulong)a1 << 32), (ulong)a2 | ((ulong)a3 << 32), false);
            }
        }

        """
    }
}
