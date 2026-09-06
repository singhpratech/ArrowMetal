import Foundation

/// MSL for the nested functions in `NestedExtra.swift`: `list_parent_indices`, `list_slice`,
/// `map_lookup` and the `list_view` -> `list` conversion.
///
/// As in `NestedSource`, every kernel here reads offsets and produces either an int32 result or an index
/// array that the child's own `take` gathers, so the child may be of any supported type.
enum NestedExtraSource {

    static let lists = KernelSource.prelude + """
    // Arrow `list_parent_indices`: for child position `base + i`, the row that covers it — the last row
    // whose offset is <= that position. One binary search per output element, so empty rows cost nothing.
    kernel void list_parent_indices(device const int* offsets [[buffer(0)]],
                                    constant uint& nRows [[buffer(1)]],
                                    device const uint* mPtr [[buffer(2)]],
                                    constant int& base [[buffer(3)]],
                                    device int* out [[buffer(4)]],
                                    uint i [[thread_position_in_grid]]) {
        if (i >= *mPtr) return;
        int pos = base + (int)i;
        uint lo = 0u, hi = nRows;
        while (lo + 1u < hi) {
            uint mid = lo + (hi - lo) / 2u;
            if (offsets[mid] <= pos) lo = mid; else hi = mid;
        }
        out[i] = (int)lo;
    }

    // Row lengths of Arrow `list_slice(start, stop, step)`. `stop < 0` means "to the end of the row".
    // A null row keeps length 0 and stays null.
    kernel void list_slice_lengths(device const int* offsets [[buffer(0)]],
                                   device const uchar* validity [[buffer(1)]],
                                   device const uint* nPtr [[buffer(2)]],
                                   constant uint& hasValidity [[buffer(3)]],
                                   constant int& start [[buffer(4)]],
                                   constant int& stop [[buffer(5)]],
                                   constant int& step [[buffer(6)]],
                                   device int* out [[buffer(7)]],
                                   uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        bool ok = (hasValidity == 0u) || bit_get(validity, i);
        int rowLen = offsets[i + 1] - offsets[i];
        if (!ok) { out[i] = 0; return; }
        int s = start > rowLen ? rowLen : start;
        int e = (stop < 0) ? rowLen : (stop > rowLen ? rowLen : stop);
        out[i] = (e > s) ? ((e - s + step - 1) / step) : 0;
    }

    // Expands the slice of every row into one flat child index array.
    kernel void list_slice_gather(device const int* offsets [[buffer(0)]],
                                  device const int* outOffsets [[buffer(1)]],
                                  device const uint* nPtr [[buffer(2)]],
                                  constant int& start [[buffer(3)]],
                                  constant int& step [[buffer(4)]],
                                  device int* out [[buffer(5)]],
                                  uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        int to = outOffsets[i], len = outOffsets[i + 1] - to;
        int from = offsets[i] + start;
        for (int k = 0; k < len; k++) out[to + k] = from + k * step;
    }

    // Arrow `list_view` -> `list`: expands each row's [offset, offset + size) range into a flat child
    // index array laid out by the contiguous offsets the conversion computes.
    kernel void list_view_gather(device const int* viewOffsets [[buffer(0)]],
                                 device const int* viewSizes [[buffer(1)]],
                                 device const int* outOffsets [[buffer(2)]],
                                 device const uint* nPtr [[buffer(3)]],
                                 device int* out [[buffer(4)]],
                                 uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        int to = outOffsets[i], len = outOffsets[i + 1] - to;
        int from = viewOffsets[i];
        for (int k = 0; k < len; k++) out[to + k] = from + k;
    }
    """

    /// `map_lookup` over a map whose keys are utf8/binary (`K` = "str") or an integer widened to int64
    /// (`K` = "int"). Two kernels: one pass that reports the first match, the last match and the number
    /// of matches per row, and one that writes every matching entry index for `occurrence = all`.
    static func mapLookup(K: String) -> String {
        let match: String
        let keyArgs: String
        if K == "str" {
            keyArgs = """
                                 device const int* keyOffsets [[buffer(1)]],
                                 device const uchar* keyData [[buffer(2)]],
                                 device const uchar* pat [[buffer(3)]],
                                 constant uint& patLen [[buffer(4)]],
            """
            match = """
                    int ks = keyOffsets[j], ke = keyOffsets[j + 1];
                    bool hit = ((uint)(ke - ks) == patLen);
                    if (hit) { for (uint k = 0; k < patLen; k++) { if (keyData[ks + k] != pat[k]) { hit = false; break; } } }
            """
        } else {
            keyArgs = """
                                 device const long* keyVals [[buffer(1)]],
                                 device const uchar* keyPad [[buffer(2)]],
                                 device const long* pat [[buffer(3)]],
                                 constant uint& patLen [[buffer(4)]],
            """
            match = """
                    bool hit = (keyVals[j] == pat[0]);
            """
        }
        return KernelSource.prelude + """
        // Per row: the first matching entry index, the last one, and how many matched. -1 when none.
        kernel void map_lookup_scan(device const int* offsets [[buffer(0)]],
        \(keyArgs)
                                    device const uchar* validity [[buffer(5)]],
                                    device const uint* nPtr [[buffer(6)]],
                                    constant uint& hasValidity [[buffer(7)]],
                                    device int* outFirst [[buffer(8)]],
                                    device int* outLast [[buffer(9)]],
                                    device int* outCount [[buffer(10)]],
                                    uint i [[thread_position_in_grid]]) {
            if (i >= *nPtr) return;
            outFirst[i] = -1; outLast[i] = -1; outCount[i] = 0;
            if ((hasValidity != 0u) && !bit_get(validity, i)) return;
            int from = offsets[i], to = offsets[i + 1];
            int first = -1, last = -1, cnt = 0;
            for (int j = from; j < to; j++) {
        \(match)
                if (hit) { if (first < 0) first = j; last = j; cnt++; }
            }
            outFirst[i] = first; outLast[i] = last; outCount[i] = cnt;
        }

        // occurrence = all: every matching entry index, laid out by the scanned offsets.
        kernel void map_lookup_gather(device const int* offsets [[buffer(0)]],
        \(keyArgs)
                                      device const uchar* validity [[buffer(5)]],
                                      device const uint* nPtr [[buffer(6)]],
                                      constant uint& hasValidity [[buffer(7)]],
                                      device const int* outOffsets [[buffer(8)]],
                                      device int* out [[buffer(9)]],
                                      uint i [[thread_position_in_grid]]) {
            if (i >= *nPtr) return;
            if ((hasValidity != 0u) && !bit_get(validity, i)) return;
            int from = offsets[i], to = offsets[i + 1];
            int w = outOffsets[i];
            for (int j = from; j < to; j++) {
        \(match)
                if (hit) { out[w] = j; w++; }
            }
        }
        """
    }
}
