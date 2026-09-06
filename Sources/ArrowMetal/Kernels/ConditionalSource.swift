import Foundation

/// MSL for the conditional and selection functions in `Conditional.swift`:
/// `fill_null_forward`, `fill_null_backward`, `replace_with_mask` and `indices_nonzero`.
///
/// The two null fills and `replace_with_mask` are all *scan*-shaped, and they reuse the two-level
/// GPU scan `CumulativeSource` already generates rather than growing a third copy of it: the kernels
/// here only produce the scan's **input** and consume its **output**.
///
///   * `fill_null_forward` — seed `p+1` at every valid slot and 0 at every null one, run an
///     inclusive **max**-scan, and the scan value at row `i` is one more than the index of the last
///     valid row at or before `i` (0 when there is none, which stays null). `fill_null_backward` is
///     the same scan walked from the far end, so one seed kernel with a direction flag covers both.
///   * `replace_with_mask` — seed 1 where the mask is a valid `true`, run an inclusive **sum**-scan,
///     and row `i`'s replacement is `replacements[scan[i] - 1]`.
///
/// `indices_nonzero` needs no scan of its own: it is `iota` filtered by `value != 0`, and the filter
/// is the existing stream compaction.
enum ConditionalSource {
    /// Seeds and gathers that do not depend on the element type.
    static let common = KernelSource.prelude + """

    // Scan seed for the null fills. `backward` walks the array from the end, so slot p of the scan
    // is row (n - 1 - p). A valid row seeds its own scan position + 1; a null row seeds 0.
    kernel void cn_fill_seed(device const uchar* validity [[buffer(0)]],
                             device const uint* nPtr [[buffer(1)]],
                             constant uint& backward [[buffer(2)]],
                             device uint* out [[buffer(3)]],
                             uint p [[thread_position_in_grid]]) {
        uint n = *nPtr;
        if (p >= n) return;
        uint row = backward ? (n - 1u - p) : p;
        out[p] = bit_get(validity, row) ? (p + 1u) : 0u;
    }

    // Scan seed for replace_with_mask: 1 where the mask is a valid true.
    kernel void cn_mask_seed(device const uchar* maskVal [[buffer(0)]],
                             device const uchar* maskValid [[buffer(1)]],
                             device const uint* nPtr [[buffer(2)]],
                             constant uint& hasValidity [[buffer(3)]],
                             device uint* out [[buffer(4)]],
                             uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        bool sel = bit_get(maskVal, i) && (hasValidity == 0u || bit_get(maskValid, i));
        out[i] = sel ? 1u : 0u;
    }

    // 0, 1, 2, ... as int64, the row numbers `indices_nonzero` compacts.
    kernel void cn_iota64(device const uint* nPtr [[buffer(0)]], device ulong* out [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
        if (i < *nPtr) out[i] = (ulong)i;
    }
    """

    /// The two element-type-dependent gathers. `T` is the *move* type — Float64 travels as `long`,
    /// since neither kernel does arithmetic on a value.
    static func moves(T: String) -> String { KernelSource.prelude + """

    // fill_null_forward / fill_null_backward. `scan[p]` is the max-scan output at scan slot p.
    kernel void cn_fill_gather(device const \(T)* vals [[buffer(0)]],
                               device const uint* scan [[buffer(1)]],
                               device const uint* nPtr [[buffer(2)]],
                               constant uint& backward [[buffer(3)]],
                               device \(T)* out [[buffer(4)]],
                               device uchar* outValid [[buffer(5)]],
                               uint i [[thread_position_in_grid]]) {
        uint n = *nPtr;
        if (i >= n) return;
        uint p = backward ? (n - 1u - i) : i;
        uint m = scan[p];
        if (m == 0u) { out[i] = (\(T))0; outValid[i] = 0; return; }
        uint src = backward ? (n - m) : (m - 1u);
        out[i] = vals[src];
        outValid[i] = 1;
    }

    // replace_with_mask. `scan[i]` is the inclusive count of selected rows up to and including i,
    // so the replacement for a selected row i is replacements[scan[i] - 1].
    // flags: bit0 = mask has validity, bit1 = values have validity, bit2 = replacements have validity.
    kernel void cn_replace(device const \(T)* vals [[buffer(0)]],
                           device const uchar* valsValid [[buffer(1)]],
                           device const uchar* maskVal [[buffer(2)]],
                           device const uchar* maskValid [[buffer(3)]],
                           device const \(T)* repl [[buffer(4)]],
                           device const uchar* replValid [[buffer(5)]],
                           device const uint* scan [[buffer(6)]],
                           device const uint* nPtr [[buffer(7)]],
                           constant uint& flags [[buffer(8)]],
                           device \(T)* out [[buffer(9)]],
                           device uchar* outValid [[buffer(10)]],
                           uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if ((flags & 1u) && !bit_get(maskValid, i)) { out[i] = (\(T))0; outValid[i] = 0; return; }
        if (bit_get(maskVal, i)) {
            uint r = scan[i] - 1u;
            out[i] = repl[r];
            outValid[i] = (flags & 4u) ? (bit_get(replValid, r) ? 1 : 0) : 1;
            return;
        }
        out[i] = vals[i];
        outValid[i] = (flags & 2u) ? (bit_get(valsValid, i) ? 1 : 0) : 1;
    }
    """ }
}
