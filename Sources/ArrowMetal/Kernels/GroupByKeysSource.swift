import Foundation

/// Metal Shading Language for the key-mapping stage of `GroupByKeys`.
///
/// The stage itself is built out of kernels that already exist — `dictionaryEncode` (argsort, run marks,
/// prefix scan) does the actual work of turning values into dense ids. What is missing is three small
/// element-wise passes, and they are here:
///
/// | kernel | what it does |
/// |---|---|
/// | `gk_densify` | dictionary codes plus a validity bitmap become non-null ids, nulls taking their own group |
/// | `gk_combine` | two dense id columns become one int64 key, `a * K + b` (the radix combine) |
/// | `gk_limb` | one 64-bit limb of a fixed-width value, so decimal128 / decimal256 can be folded limb by limb |
enum GroupByKeysSource {
    static let source: String = KernelSource.prelude + """

    // Dense group ids from dictionary codes. A null row is given the dedicated group `nullId`, so a null
    // key forms its own group exactly as Arrow's hash aggregation does, and the output has no nulls.
    kernel void gk_densify(device const int* codes [[buffer(0)]],
                           device const uchar* validity [[buffer(1)]],
                           constant uint& hasValidity [[buffer(2)]],
                           constant uint& nullId [[buffer(3)]],
                           device const uint* nPtr [[buffer(4)]],
                           device int* out [[buffer(5)]],
                           uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        out[i] = (hasValidity != 0u && !bit_get(validity, i)) ? (int)nullId : codes[i];
    }

    // Radix combine of two dense id columns into one 64-bit key. `K` is the cardinality of `b`, so the
    // map is injective and the result orders lexicographically by (a, b). The host checks that the
    // product of the cardinalities fits in an int64 before dispatching this.
    kernel void gk_combine(device const int* a [[buffer(0)]],
                           device const int* b [[buffer(1)]],
                           constant uint& K [[buffer(2)]],
                           device const uint* nPtr [[buffer(3)]],
                           device long* out [[buffer(4)]],
                           uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        out[i] = (long)a[i] * (long)K + (long)b[i];
    }

    // The two kernels of the RANGE fast path, which skips the sort entirely when an integer key column
    // spans a small enough range: mark which values occur, prefix-scan the marks, and read each row's
    // rank out of the scan. That is three linear passes over the rows and one scan over the range,
    // against the radix sort the general path would otherwise run.
    kernel void gk_occupy(device const long* vals [[buffer(0)]],
                          device const uchar* validity [[buffer(1)]],
                          constant uint& hasValidity [[buffer(2)]],
                          constant long& lo [[buffer(3)]],
                          device const uint* nPtr [[buffer(4)]],
                          device int* occ [[buffer(5)]],
                          uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if (hasValidity != 0u && !bit_get(validity, i)) return;
        occ[(uint)(vals[i] - lo)] = 1;
    }

    // `cum` is the INCLUSIVE scan of the occupancy marks, so `cum[v - lo] - 1` is the dense rank of v
    // among the values that actually occur. A null row takes the dedicated `nullId`.
    kernel void gk_rank(device const long* vals [[buffer(0)]],
                        device const uchar* validity [[buffer(1)]],
                        constant uint& hasValidity [[buffer(2)]],
                        constant long& lo [[buffer(3)]],
                        constant uint& nullId [[buffer(4)]],
                        device const int* cum [[buffer(5)]],
                        device const uint* nPtr [[buffer(6)]],
                        device int* out [[buffer(7)]],
                        uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if (hasValidity != 0u && !bit_get(validity, i)) { out[i] = (int)nullId; return; }
        out[i] = cum[(uint)(vals[i] - lo)] - 1;
    }

    // Row indices 0, 1, ... n - 1, so a 50M-row group-by never pays for a host loop.
    kernel void gk_iota(device const uint* nPtr [[buffer(0)]], device int* out [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
        if (i < *nPtr) out[i] = (int)i;
    }

    // One 64-bit limb of a fixed-width value, little-endian: limb 0 of a decimal128 is its low half.
    kernel void gk_limb(device const ulong* vals [[buffer(0)]],
                        constant uint& limbs [[buffer(1)]],
                        constant uint& which [[buffer(2)]],
                        device const uint* nPtr [[buffer(3)]],
                        device ulong* out [[buffer(4)]],
                        uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        out[i] = vals[i * limbs + which];
    }
    """
}
