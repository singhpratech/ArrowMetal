import Foundation

/// MSL for the utf8 / binary sort (`Kernels/StringSort.swift`).
///
/// One kernel: it turns a fixed-width slice of each row's bytes into a 63-bit integer whose numeric order
/// is the bytes' lexicographic order. Seven source bytes go into seven 9-bit fields, each holding
/// `byte + 1` when the byte exists and `0` when the row has already ended. That extra value is what makes
/// the mapping exact: a row that stops inside the chunk compares below every row that continues, whatever
/// byte comes next — including a NUL, which a plain zero-padded key would confuse with the end of a
/// shorter row ("ab" must sort before "ab\0", and does here).
///
/// Bit 63, which the seven fields leave free, carries the null partition: with `nullMode` 1 a null row's
/// key is exactly `1 << 63` (above every valid key, in both directions), with `nullMode` 2 it is `0` and
/// every valid key gets bit 63 set. All null rows share one key, so the stable passes keep them in row
/// order, as a block at the requested end, with no separate partition step and no CPU readback.
enum StringSortSource {
    /// Bytes packed into one key. 7 x 9 = 63 bits, so the key is always non-negative and `descending`
    /// can mirror it with a subtraction from the 63-bit maximum without losing stability.
    static let bytesPerChunk = 7

    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void str_prefix_key(device const int* offsets [[buffer(0)]],
                               device const uchar* data [[buffer(1)]],
                               device const int* perm [[buffer(2)]],
                               constant uint& n [[buffer(3)]],
                               constant uint& chunk [[buffer(4)]],
                               constant uint& hasPerm [[buffer(5)]],
                               constant uint& descending [[buffer(6)]],
                               device ulong* out [[buffer(7)]],
                               device const uchar* validity [[buffer(8)]],
                               constant uint& nullMode [[buffer(9)]],
                               uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        uint row = hasPerm ? (uint)perm[i] : i;
        uint start = (uint)offsets[row];
        uint end = (uint)offsets[row + 1];
        uint base = start + chunk * 7u;
        ulong k = 0ul;
        for (uint j = 0; j < 7u; j++) {
            uint p = base + j;
            ulong v = (p < end) ? ((ulong)data[p] + 1ul) : 0ul;
            k = (k << 9) | v;
        }
        if (descending) k = 0x7FFFFFFFFFFFFFFFul - k;
        if (nullMode != 0u) {
            bool valid = ((validity[row >> 3] >> (row & 7u)) & 1u) != 0u;
            if (nullMode == 1u) { if (!valid) k = 0x8000000000000000ul; }
            else { k = valid ? (k | 0x8000000000000000ul) : 0ul; }
        }
        out[i] = k;
    }
    """
}
