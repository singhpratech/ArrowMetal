import Foundation

/// MSL for the string ↔ number casts.
///
/// **Integer → string (`str_itoa_*`)** follows the same two-pass shape as the string transforms: the
/// output byte length of a row is data dependent, so one kernel writes the lengths, the host scans
/// them into the Arrow offsets buffer on the GPU, and a second kernel writes the digits. Both passes
/// derive the digit count from the same `itoa_digits`, so a length and the bytes that fill it cannot
/// disagree. `Int64.min` is handled without ever negating it: the magnitude is built as
/// `(ulong)(-(v + 1)) + 1`.
///
/// **String → integer (`str_parse_int`)** is one thread per 32 rows, so the thread owns a whole
/// 32-bit validity word and no atomics are needed. The grammar is exactly
/// `[+-]? [0-9]+` over the whole value: no whitespace, no underscores, no radix prefix, no exponent.
/// Leading zeros are fine (`"007"` is 7). Anything else — an empty value, a stray character, a value
/// outside the target type's range — clears the row's validity bit and leaves the value 0, which is
/// Arrow's `safe=false` string→integer cast with nulls on failure. A `-` sign on an unsigned target
/// is rejected outright, `"-0"` included.
enum StringCastSource {

    private static let digits = """
    inline int itoa_digits(ulong m) { int d = 1; while (m >= 10UL) { m /= 10UL; d++; } return d; }
    """

    /// Integer → decimal text. `isSigned` picks how the magnitude and the `-` are derived.
    static func itoa(T: String, isSigned: Bool) -> String {
        let split = isSigned
            ? "long v = (long)vals[i]; if (v < 0L) { sign = 1; m = (ulong)(-(v + 1L)) + 1UL; } else { m = (ulong)v; }"
            : "m = (ulong)vals[i];"
        return KernelSource.prelude + digits + """

        // Pass 1: number of output bytes per row (sign + digits). A null row produces no bytes.
        kernel void str_itoa_len(device const \(T)* vals [[buffer(0)]],
                                 device const uchar* validity [[buffer(1)]],
                                 device const uint* nPtr [[buffer(2)]],
                                 constant uint& hasValidity [[buffer(3)]],
                                 device int* outLens [[buffer(4)]],
                                 uint i [[thread_position_in_grid]]) {
            if (i >= *nPtr) return;
            if (hasValidity != 0u && !bit_get(validity, i)) { outLens[i] = 0; return; }
            ulong m = 0UL; int sign = 0;
            \(split)
            outLens[i] = sign + itoa_digits(m);
        }

        // Pass 2: the digits, most significant first, at outOffsets[i].
        kernel void str_itoa_write(device const \(T)* vals [[buffer(0)]],
                                   device const uchar* validity [[buffer(1)]],
                                   device const uint* nPtr [[buffer(2)]],
                                   constant uint& hasValidity [[buffer(3)]],
                                   device const int* outOffsets [[buffer(4)]],
                                   device uchar* outData [[buffer(5)]],
                                   uint i [[thread_position_in_grid]]) {
            if (i >= *nPtr) return;
            if (hasValidity != 0u && !bit_get(validity, i)) return;
            ulong m = 0UL; int sign = 0;
            \(split)
            int d = itoa_digits(m);
            int pos = outOffsets[i];
            if (sign != 0) outData[pos++] = 0x2Du;                 // '-'
            for (int k = d - 1; k >= 0; k--) {
                outData[pos + k] = (uchar)(0x30u + (uint)(m % 10UL));
                m /= 10UL;
            }
        }
        """
    }

    /// Decimal text → integer. `limPos` / `limNeg` are the largest magnitudes the target accepts on
    /// each side of zero; `limNeg == 0` means the target is unsigned and a `-` sign is a parse failure.
    static func parse(T: String, isSigned: Bool) -> String {
        let store = isSigned
            ? "outVals[i] = (\(T))(neg ? (long)(0UL - acc) : (long)acc);"
            : "outVals[i] = (\(T))acc;"
        return KernelSource.prelude + """

        // One thread per 32 rows: the thread owns the whole validity word, so there are no atomics.
        kernel void str_parse_int(device const int* offsets [[buffer(0)]],
                                  device const uchar* data [[buffer(1)]],
                                  device const uint* nPtr [[buffer(2)]],
                                  device const uchar* inValidity [[buffer(3)]],
                                  constant uint& hasValidity [[buffer(4)]],
                                  constant ulong& limPos [[buffer(5)]],
                                  constant ulong& limNeg [[buffer(6)]],
                                  device \(T)* outVals [[buffer(7)]],
                                  device uint* outValid [[buffer(8)]],
                                  uint w [[thread_position_in_grid]]) {
            uint n = *nPtr, base = w * 32u;
            if (base >= n) return;
            uint limit = min(32u, n - base), bits = 0u;
            for (uint j = 0; j < limit; j++) {
                uint i = base + j;
                outVals[i] = (\(T))0;
                if (hasValidity != 0u && !bit_get(inValidity, i)) continue;
                int p = offsets[i], e = offsets[i + 1];
                if (p >= e) continue;                              // empty value
                bool neg = false;
                uchar first = data[p];
                if (first == 0x2Bu || first == 0x2Du) { neg = (first == 0x2Du); p++; }
                if (p >= e) continue;                              // sign with no digits
                if (neg && limNeg == 0UL) continue;                // unsigned target
                ulong lim = neg ? limNeg : limPos;
                ulong acc = 0UL;
                bool ok = true;
                for (; p < e; p++) {
                    uchar c = data[p];
                    if (c < 0x30u || c > 0x39u) { ok = false; break; }
                    ulong dg = (ulong)(c - 0x30u);
                    if (acc > (lim - dg) / 10UL) { ok = false; break; }   // would overflow the target
                    acc = acc * 10UL + dg;
                }
                if (!ok) continue;
                \(store)
                bits |= (1u << j);
            }
            outValid[w] = bits;
        }
        """
    }
}
