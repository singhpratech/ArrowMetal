import Foundation

/// Decimal text → IEEE-754 binary64 / binary32 on the GPU, with no floating-point arithmetic at all.
///
/// Metal has no `double`, so the parser never computes in floating point: it reads the decimal digits
/// into a 64-bit integer `w` and a power of ten `q`, then runs the Eisel-Lemire algorithm (the one in
/// fast_float, which is what Arrow's CSV reader and `strtod` replacements in most modern runtimes use):
/// multiply `w` by a 128-bit approximation of 5^q from `CSVFloatTable` with `mulhi` on `ulong`, take the
/// top 54 (or 25) bits, and round to nearest-even from the bits that fell off. For up to 19 significant
/// digits that product is always enough to round correctly (Mushtak & Lemire, "Fast Number Parsing
/// Without Fallback"), so the result is the correctly rounded value, the same one a correct `strtod`
/// returns.
///
/// Every case the parser cannot settle exactly is handed back as **undecided** rather than guessed:
///
/// - more than 19 significant digits where the 19-digit prefix `w` and `w + 1` round to different
///   values (the extra digits decide it, and only a big-integer comparison can);
/// - the one product shape the original Eisel-Lemire paper leaves to a fallback (low word all ones
///   outside q in [-27, 55]) — kept as a conservative guard even though the later proof shows it is
///   never reached for exact 19-digit inputs.
///
/// The caller then parses that element on the CPU. Two grammars share the code:
///
/// - **`FP_SWIFT`** is `MetalStringArray.parse(Double.self)` / `parse(Float.self)`, whose contract is
///   Swift's `Double(_:)` / `Float(_:)` initialisers. The GPU decides the plain decimal form
///   `[+-]?(digits[.digits]|.digits)([eE][+-]?digits)?` and the exact spellings `inf`, `infinity`,
///   `nan` (any case, optional sign); it decides "not a number" only for text Swift's parser cannot
///   consume at all (empty, leading whitespace, a first character that no float spelling starts with).
///   Everything else — hex floats, `nan(payload)`, `snan`, embedded NUL, a malformed exponent — goes to
///   the CPU, so this path cannot disagree with the CPU path by construction.
/// - **`FP_CSV`** is Arrow's CSV float grammar (fast_float's `general` format behind Arrow's own
///   wrapper): spaces and tabs trimmed, one optional `+` or `-`, a configurable decimal point, `inf` /
///   `infinity` / `nan` / `nan(chars)` in any case. A value outside it is invalid.
enum CSVFloatSource {

    /// MSL helpers: `fp_compute` (Eisel-Lemire) and `fp_parse` (grammar + compose the bits).
    static let functions = CSVFloatTable.msl + """

    #define FP_VALUE 0u
    #define FP_INVALID 1u
    #define FP_HOST 2u
    #define FP_SWIFT 0u
    #define FP_CSV 1u

    // Eisel-Lemire for one target format. Returns false when the result cannot be decided exactly.
    // mb: explicit mantissa bits; minExp: minimum exponent; infPow: all-ones exponent; the rest are
    // fast_float's binary_format constants for that format.
    inline bool fp_compute(int q, ulong w, int mb, int minExp, int infPow, int minRTE, int maxRTE,
                           int sp10, int lp10, thread ulong& mant, thread int& p2) {
        if (w == 0UL || q < sp10) { mant = 0UL; p2 = 0; return true; }
        if (q > lp10) { mant = 0UL; p2 = infPow; return true; }
        int lz = (int)clz(w);
        w <<= (ulong)lz;
        uint idx = 2u * (uint)(q + 342);
        ulong p5hi = FP_POW5[idx], p5lo = FP_POW5[idx + 1u];
        ulong fh = mulhi(w, p5hi), fl = w * p5hi;
        ulong precisionMask = 0xFFFFFFFFFFFFFFFFUL >> (ulong)(mb + 3);
        if ((fh & precisionMask) == precisionMask) {
            ulong sh = mulhi(w, p5lo);
            fl += sh;
            if (sh > fl) fh += 1UL;
        }
        if (fl == 0xFFFFFFFFFFFFFFFFUL && (q < -27 || q > 55)) return false;
        int upperbit = (int)(fh >> 63);
        int shift = upperbit + 64 - mb - 3;
        mant = fh >> (ulong)shift;
        p2 = (((152170 + 65536) * q) >> 16) + 63 + upperbit - lz - minExp;
        if (p2 <= 0) {
            if (-p2 + 1 >= 64) { mant = 0UL; p2 = 0; return true; }
            mant >>= (ulong)(-p2 + 1);
            mant += (mant & 1UL);
            mant >>= 1UL;
            p2 = (mant < (1UL << (ulong)mb)) ? 0 : 1;
            return true;
        }
        if (fl <= 1UL && q >= minRTE && q <= maxRTE && (mant & 3UL) == 1UL) {
            if ((mant << (ulong)shift) == fh) mant &= ~1UL;
        }
        mant += (mant & 1UL);
        mant >>= 1UL;
        if (mant >= (2UL << (ulong)mb)) { mant = 1UL << (ulong)mb; p2 += 1; }
        mant &= ~(1UL << (ulong)mb);
        if (p2 >= infPow) { p2 = infPow; mant = 0UL; }
        return true;
    }

    inline bool fp_ieq(device const uchar* p, uint i, uint len, constant char* word, uint n) {
        if (len - i != n) return false;
        for (uint k = 0u; k < n; k++) { if ((p[i + k] | 0x20u) != (uint)word[k]) return false; }
        return true;
    }
    inline bool fp_iprefix(device const uchar* p, uint i, uint len, constant char* word, uint n) {
        if (len - i < n) return false;
        for (uint k = 0u; k < n; k++) { if ((p[i + k] | 0x20u) != (uint)word[k]) return false; }
        return true;
    }

    constant char FP_W_INF[3] = {'i', 'n', 'f'};
    constant char FP_W_INFINITY[8] = {'i', 'n', 'f', 'i', 'n', 'i', 't', 'y'};
    constant char FP_W_NAN[3] = {'n', 'a', 'n'};

    // Parses p[0, len) under `grammar` into the bits of a float64 (f32 == false) or float32 (in the low
    // 32 bits). Returns FP_VALUE, FP_INVALID or FP_HOST.
    inline uint fp_parse(device const uchar* p, uint len, uint grammar, uchar dp, bool f32, thread ulong& bits) {
        bits = 0UL;
        uint i = 0u;
        if (grammar == FP_CSV) {
            while (len > 0u && (p[0] == 0x20u || p[0] == 0x09u)) { p++; len--; }
            while (len > 0u && (p[len - 1u] == 0x20u || p[len - 1u] == 0x09u)) len--;
            if (len == 0u) return FP_INVALID;
        } else {
            if (len == 0u) return FP_INVALID;
            uchar c0 = p[0];
            if (c0 == 0u || c0 == 0x20u || (c0 >= 0x09u && c0 <= 0x0Du)) return FP_INVALID;
        }
        bool neg = false;
        if (p[0] == 0x2Bu || p[0] == 0x2Du) { neg = p[0] == 0x2Du; i = 1u; }
        if (i >= len) return FP_INVALID;
        ulong signBit = neg ? (f32 ? 0x80000000UL : 0x8000000000000000UL) : 0UL;
        ulong infBits = f32 ? 0x7F800000UL : 0x7FF0000000000000UL;
        ulong nanBits = f32 ? 0x7FC00000UL : 0x7FF8000000000000UL;
        uchar c = p[i];
        uchar lc = c | 0x20u;
        if (lc == 0x69u) {                                               // 'i'
            if (fp_ieq(p, i, len, FP_W_INF, 3u) || fp_ieq(p, i, len, FP_W_INFINITY, 8u)) {
                bits = infBits | signBit; return FP_VALUE;
            }
            if (grammar == FP_CSV) return FP_INVALID;
            return fp_iprefix(p, i, len, FP_W_INF, 3u) ? FP_HOST : FP_INVALID;
        }
        if (lc == 0x6Eu) {                                               // 'n'
            if (fp_ieq(p, i, len, FP_W_NAN, 3u)) { bits = nanBits | signBit; return FP_VALUE; }
            if (!fp_iprefix(p, i, len, FP_W_NAN, 3u)) return FP_INVALID;
            if (grammar == FP_SWIFT) return FP_HOST;
            // Arrow: nan(n-char-sequence) with the closing parenthesis ending the value.
            uint k = i + 3u;
            if (p[k] != 0x28u || p[len - 1u] != 0x29u || len - k < 2u) return FP_INVALID;
            for (uint j = k + 1u; j + 1u < len; j++) {
                uchar ch = p[j];
                bool ok = (ch >= 0x30u && ch <= 0x39u) || ((ch | 0x20u) >= 0x61u && (ch | 0x20u) <= 0x7Au) || ch == 0x5Fu;
                if (!ok) return FP_INVALID;
            }
            bits = nanBits | signBit; return FP_VALUE;
        }
        bool startsNumber = (c >= 0x30u && c <= 0x39u) || c == dp;
        if (!startsNumber) {
            if (grammar == FP_CSV) return FP_INVALID;
            // Swift: 's' may be the start of "snan"; everything else cannot start a float.
            return (lc == 0x73u) ? FP_HOST : FP_INVALID;
        }
        // Decimal digits: the first 19 significant ones go into w, the rest only move the exponent.
        ulong w = 0UL; int nd = 0; int e10 = 0; bool trunc = false; uint ndigits = 0u;
        while (i < len) {
            uint d = (uint)p[i] - 48u;
            if (d > 9u) break;
            if (w == 0UL && d == 0u) {}
            else if (nd < 19) { w = w * 10UL + (ulong)d; nd++; }
            else { e10++; if (d != 0u) trunc = true; }
            ndigits++; i++;
        }
        if (i < len && p[i] == dp) {
            i++;
            while (i < len) {
                uint d = (uint)p[i] - 48u;
                if (d > 9u) break;
                if (w == 0UL && d == 0u) { e10--; }
                else if (nd < 19) { w = w * 10UL + (ulong)d; nd++; e10--; }
                else if (d != 0u) { trunc = true; }
                ndigits++; i++;
            }
        }
        if (ndigits == 0u) return grammar == FP_CSV ? FP_INVALID : FP_HOST;
        if (i < len && (p[i] | 0x20u) == 0x65u) {                        // 'e'
            i++;
            bool eneg = false;
            if (i < len && (p[i] == 0x2Bu || p[i] == 0x2Du)) { eneg = p[i] == 0x2Du; i++; }
            uint ed = 0u; int ev = 0;
            while (i < len) {
                uint d = (uint)p[i] - 48u;
                if (d > 9u) break;
                if (ev < 100000000) ev = ev * 10 + (int)d;
                ed++; i++;
            }
            if (ed == 0u) return grammar == FP_CSV ? FP_INVALID : FP_HOST;
            e10 += eneg ? -ev : ev;
        }
        if (i != len) return grammar == FP_CSV ? FP_INVALID : FP_HOST;
        // Clamp far outside both formats' ranges so the int arithmetic cannot overflow.
        int q = clamp(e10, -100000, 100000);
        ulong mant = 0UL; int p2 = 0;
        int mb = f32 ? 23 : 52, minExp = f32 ? -127 : -1023, infPow = f32 ? 0xFF : 0x7FF;
        int minRTE = f32 ? -17 : -4, maxRTE = f32 ? 10 : 23, sp10 = f32 ? -65 : -342, lp10 = f32 ? 38 : 308;
        if (!fp_compute(q, w, mb, minExp, infPow, minRTE, maxRTE, sp10, lp10, mant, p2)) return FP_HOST;
        if (trunc) {
            ulong mant2 = 0UL; int p22 = 0;
            if (!fp_compute(q, w + 1UL, mb, minExp, infPow, minRTE, maxRTE, sp10, lp10, mant2, p22)) return FP_HOST;
            if (mant2 != mant || p22 != p2) return FP_HOST;
        }
        bits = mant | ((ulong)p2 << (ulong)mb) | signBit;
        return FP_VALUE;
    }
    """

    /// `MetalStringArray.parse(Double.self)` / `parse(Float.self)`: one thread per row, validity and the
    /// "ask the CPU" flags packed 32 rows at a time with `simd_ballot`.
    static let stringKernels = KernelSource.prelude + functions + """

    inline void fp_row(device const int* offsets, device const uchar* data, uint n, device const uchar* validity,
                       uint hasValidity, bool f32, uint i, thread ulong& bits, thread bool& ok, thread bool& host) {
        ok = false; host = false; bits = 0UL;
        if (i >= n) return;
        if (hasValidity != 0u && !bit_get(validity, i)) return;
        int s = offsets[i], e = offsets[i + 1];
        uint st = fp_parse(data + s, (uint)(e - s), FP_SWIFT, (uchar)0x2Eu, f32, bits);
        ok = st == FP_VALUE;
        host = st == FP_HOST;
    }

    kernel void str_parse_f64(device const int* offsets [[buffer(0)]],
                              device const uchar* data [[buffer(1)]],
                              device const uint* nPtr [[buffer(2)]],
                              device const uchar* validity [[buffer(3)]],
                              constant uint& hasValidity [[buffer(4)]],
                              device ulong* outVals [[buffer(5)]],
                              device uint* outValid [[buffer(6)]],
                              device uint* outHost [[buffer(7)]],
                              device atomic_uint* hostCount [[buffer(8)]],
                              uint i [[thread_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
        uint n = *nPtr;
        ulong bits; bool ok, host;
        fp_row(offsets, data, n, validity, hasValidity, false, i, bits, ok, host);
        if (i < n) outVals[i] = ok ? bits : 0UL;
        uint vw = (uint)(simd_vote::vote_t)simd_ballot(ok);
        uint hw = (uint)(simd_vote::vote_t)simd_ballot(host);
        if (lane == 0u && i < n) {
            outValid[i >> 5] = vw;
            outHost[i >> 5] = hw;
            if (hw != 0u) atomic_fetch_add_explicit(hostCount, popcount(hw), memory_order_relaxed);
        }
    }

    kernel void str_parse_f32(device const int* offsets [[buffer(0)]],
                              device const uchar* data [[buffer(1)]],
                              device const uint* nPtr [[buffer(2)]],
                              device const uchar* validity [[buffer(3)]],
                              constant uint& hasValidity [[buffer(4)]],
                              device uint* outVals [[buffer(5)]],
                              device uint* outValid [[buffer(6)]],
                              device uint* outHost [[buffer(7)]],
                              device atomic_uint* hostCount [[buffer(8)]],
                              uint i [[thread_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
        uint n = *nPtr;
        ulong bits; bool ok, host;
        fp_row(offsets, data, n, validity, hasValidity, true, i, bits, ok, host);
        if (i < n) outVals[i] = ok ? (uint)bits : 0u;
        uint vw = (uint)(simd_vote::vote_t)simd_ballot(ok);
        uint hw = (uint)(simd_vote::vote_t)simd_ballot(host);
        if (lane == 0u && i < n) {
            outValid[i >> 5] = vw;
            outHost[i >> 5] = hw;
            if (hw != 0u) atomic_fetch_add_explicit(hostCount, popcount(hw), memory_order_relaxed);
        }
    }
    """
}
