import Foundation

/// MSL for Arrow nested layouts: int32 list offsets over a child array.
///
/// Every list kernel here works on offsets alone and produces either an int32 result or an index array
/// that the child's own `take` then gathers, so one set of kernels covers `list`, `large_list`,
/// `fixed_size_list` and `map` over a child of any supported type, recursively.
enum NestedSource {
    static let source = KernelSource.prelude + """
    // Arrow `list_value_length`: offsets[i + 1] - offsets[i]. Validity is shared with the input.
    kernel void list_value_length(device const int* offsets [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                                  device int* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i < *nPtr) out[i] = offsets[i + 1] - offsets[i];
    }
    // Arrow `list_element`: the child index of element k of each list. The slot is marked invalid when the
    // list is null or has fewer than k + 1 elements, which is what makes the gathered element null.
    kernel void list_element_index(device const int* offsets [[buffer(0)]], device const uchar* validity [[buffer(1)]],
                                   device const uint* nPtr [[buffer(2)]], constant uint& hasValidity [[buffer(3)]],
                                   constant int& k [[buffer(4)]], device int* out [[buffer(5)]],
                                   device uchar* outValid [[buffer(6)]], uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        int from = offsets[i], len = offsets[i + 1] - from;
        bool ok = (hasValidity == 0u) || bit_get(validity, i);
        if (!ok || k < 0 || k >= len) { out[i] = 0; outValid[i] = 0; return; }
        out[i] = from + k;
        outValid[i] = 1;
    }
    // Expands per-row source ranges into one flat child index array, which the child's `take` then gathers.
    // Row i writes outOffsets[i + 1] - outOffsets[i] indices starting at srcOffsets[src[i]]; a negative
    // source index (a null row in the selection) leaves its slots null instead.
    kernel void list_expand_indices(device const int* srcOffsets [[buffer(0)]], device const int* src [[buffer(1)]],
                                    device const int* outOffsets [[buffer(2)]], device const uint* nPtr [[buffer(3)]],
                                    device int* out [[buffer(4)]], device uchar* outValid [[buffer(5)]],
                                    uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        int to = outOffsets[i], len = outOffsets[i + 1] - to;
        int s = src[i];
        int from = (s < 0) ? 0 : srcOffsets[s];
        uchar v = (s < 0) ? 0 : 1;
        for (int k = 0; k < len; k++) { out[to + k] = from + k; outValid[to + k] = v; }
    }
    """
}
