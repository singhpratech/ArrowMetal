import Foundation

/// Metal Shading Language for the structural, conditional and set-lookup kernels
/// (`is_null`, `is_valid`, `fill_null`, `if_else`, `coalesce`, `is_in`, `index_in`,
/// `and_kleene`, `or_kleene`). Generated per element type at runtime, like the rest of
/// the kernel families.
///
/// Every kernel takes its element count as `device const uint* nPtr`, so a result whose
/// length is still being decided by GPU work in an open batch can bind that length buffer
/// instead of a host-known constant.
enum StructuralSource {
    /// Bitmap-word kernels that do not depend on the element type: a constant word fill
    /// (for `is_valid` on an array with no validity bitmap) and the two Kleene operators.
    ///
    /// Kleene semantics, per element, with `v` the value bit and `p` the validity bit:
    ///
    ///   and: `false AND anything = false`, so the result is valid when either side is a
    ///        valid false or both sides are valid; the value is `va & vb`.
    ///   or:  `true OR anything = true`, so the result is valid when either side is a
    ///        valid true or both sides are valid; the value is `va | vb`.
    ///
    /// Both formulas read the value bits of null slots, which Arrow leaves undefined — that is
    /// safe here because every term that can select such a bit is masked by that side's
    /// validity bit, and where the result is null its value is ignored.
    static let common = KernelSource.prelude + """

    kernel void st_fill_words(device const uint* nPtr [[buffer(0)]], constant uint& value [[buffer(1)]],
                              device uint* out [[buffer(2)]], uint w [[thread_position_in_grid]]) {
        if (w < (*nPtr + 31u) / 32u) out[w] = value;
    }
    // flags: bit0 = a has a validity bitmap, bit1 = b has one.
    kernel void st_and_kleene(device const uint* aVal [[buffer(0)]], device const uint* aValid [[buffer(1)]],
                              device const uint* bVal [[buffer(2)]], device const uint* bValid [[buffer(3)]],
                              device const uint* nPtr [[buffer(4)]], constant uint& flags [[buffer(5)]],
                              device uint* outVal [[buffer(6)]], device uint* outValid [[buffer(7)]],
                              uint w [[thread_position_in_grid]]) {
        if (w >= (*nPtr + 31u) / 32u) return;
        uint av = aVal[w], bv = bVal[w];
        uint ava = (flags & 1u) ? aValid[w] : 0xFFFFFFFFu;
        uint bva = (flags & 2u) ? bValid[w] : 0xFFFFFFFFu;
        outVal[w] = av & bv;
        outValid[w] = (ava & ~av) | (bva & ~bv) | (ava & bva);
    }
    kernel void st_or_kleene(device const uint* aVal [[buffer(0)]], device const uint* aValid [[buffer(1)]],
                             device const uint* bVal [[buffer(2)]], device const uint* bValid [[buffer(3)]],
                             device const uint* nPtr [[buffer(4)]], constant uint& flags [[buffer(5)]],
                             device uint* outVal [[buffer(6)]], device uint* outValid [[buffer(7)]],
                             uint w [[thread_position_in_grid]]) {
        if (w >= (*nPtr + 31u) / 32u) return;
        uint av = aVal[w], bv = bVal[w];
        uint ava = (flags & 1u) ? aValid[w] : 0xFFFFFFFFu;
        uint bva = (flags & 2u) ? bValid[w] : 0xFFFFFFFFu;
        outVal[w] = av | bv;
        outValid[w] = (ava & av) | (bva & bv) | (ava & bva);
    }
    """

    /// Value-moving kernels: `fill_null`, `if_else` and the pairwise step of `coalesce`.
    /// `T` is the *move* type — Float64 columns are moved as raw `long`, since none of these
    /// kernels does arithmetic on the value.
    static func moves(T: String) -> String { KernelSource.prelude + """

    kernel void st_fill_null(device const \(T)* vals [[buffer(0)]],
                             device const uchar* validity [[buffer(1)]],
                             device const uint* nPtr [[buffer(2)]],
                             constant \(T)& scalar [[buffer(3)]],
                             constant uint& hasValidity [[buffer(4)]],
                             device \(T)* out [[buffer(5)]],
                             uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        out[i] = (!hasValidity || bit_get(validity, i)) ? vals[i] : scalar;
    }

    // flags: bit0 cond has validity, bit1 left has validity, bit2 right has validity,
    //        bit3 left is a scalar, bit4 right is a scalar.
    // A null condition yields a null output, matching Arrow's `if_else`.
    kernel void st_if_else(device const uchar* cond [[buffer(0)]],
                           device const uchar* condValid [[buffer(1)]],
                           device const \(T)* left [[buffer(2)]],
                           device const uchar* leftValid [[buffer(3)]],
                           device const \(T)* right [[buffer(4)]],
                           device const uchar* rightValid [[buffer(5)]],
                           constant \(T)& leftScalar [[buffer(6)]],
                           constant \(T)& rightScalar [[buffer(7)]],
                           device const uint* nPtr [[buffer(8)]],
                           constant uint& flags [[buffer(9)]],
                           device \(T)* out [[buffer(10)]],
                           device uchar* outValid [[buffer(11)]],
                           uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if ((flags & 1u) && !bit_get(condValid, i)) { out[i] = (\(T))0; outValid[i] = 0; return; }
        if (bit_get(cond, i)) {
            if (flags & 8u) { out[i] = leftScalar; outValid[i] = 1; }
            else { out[i] = left[i]; outValid[i] = (flags & 2u) ? (bit_get(leftValid, i) ? 1 : 0) : 1; }
        } else {
            if (flags & 16u) { out[i] = rightScalar; outValid[i] = 1; }
            else { out[i] = right[i]; outValid[i] = (flags & 4u) ? (bit_get(rightValid, i) ? 1 : 0) : 1; }
        }
    }

    // One fold step of `coalesce`: take a's value where a is valid, otherwise b's.
    // flags: bit0 a has validity, bit1 b has validity.
    kernel void st_coalesce2(device const \(T)* a [[buffer(0)]],
                             device const uchar* aValid [[buffer(1)]],
                             device const \(T)* b [[buffer(2)]],
                             device const uchar* bValid [[buffer(3)]],
                             device const uint* nPtr [[buffer(4)]],
                             constant uint& flags [[buffer(5)]],
                             device \(T)* out [[buffer(6)]],
                             device uchar* outValid [[buffer(7)]],
                             uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if (!(flags & 1u) || bit_get(aValid, i)) { out[i] = a[i]; outValid[i] = 1; return; }
        out[i] = b[i];
        outValid[i] = (flags & 2u) ? (bit_get(bValid, i) ? 1 : 0) : 1;
    }
    """ }

    /// Order-preserving key for a floating point bit pattern, matching the total order the radix
    /// sort (and therefore `unique()`) uses: every NaN collapses to one quiet NaN that sorts after
    /// +inf, and -0 compares equal to +0.
    static let floatKeys = """
    inline int f_key(int b) {
        int m = b & 0x7FFFFFFF;
        if (m > 0x7F800000) return 0x7FC00000;
        if (m == 0) return 0;
        return b ^ (int)(((uint)(b >> 31)) >> 1);
    }
    inline long d_key_n(long b) {
        long m = b & 0x7FFFFFFFFFFFFFFFL;
        if (m > 0x7FF0000000000000L) return 0x7FF8000000000000L;
        if (m == 0L) return 0L;
        return b ^ (long)(((ulong)(b >> 63)) >> 1);
    }
    """

    /// `is_in` / `index_in` by binary search over a sorted, distinct set.
    ///
    /// `K` is the type the values are read as (the element type itself for integers, the raw
    /// bit-pattern integer for floats) and `toKey` maps one loaded value to the ordered key the
    /// search compares. Null elements never match: the set is built from `unique()`, which drops
    /// nulls, so nulls in the set are ignored and a null input is not in the set.
    static func lookup(K: String, toKey: String) -> String { KernelSource.prelude + floatKeys + """

    inline uint st_lower_bound(device const \(K)* setVals, uint count, \(K) k) {
        uint lo = 0, hi = count;
        while (lo < hi) {
            uint mid = (lo + hi) >> 1;
            if (\(toKey.replacingOccurrences(of: "$0", with: "setVals[mid]")) < k) lo = mid + 1; else hi = mid;
        }
        return lo;
    }

    kernel void st_is_in(device const \(K)* vals [[buffer(0)]],
                         device const uchar* validity [[buffer(1)]],
                         device const uint* nPtr [[buffer(2)]],
                         device const \(K)* setVals [[buffer(3)]],
                         constant uint& setCount [[buffer(4)]],
                         constant uint& hasValidity [[buffer(5)]],
                         device uint* out [[buffer(6)]],
                         uint w [[thread_position_in_grid]]) {
        uint n = *nPtr;
        uint base = w * 32u;
        if (base >= n) return;
        uint limit = min(32u, n - base);
        uint bits = 0;
        for (uint j = 0; j < limit; j++) {
            uint i = base + j;
            if (hasValidity && !bit_get(validity, i)) continue;
            \(K) k = \(toKey.replacingOccurrences(of: "$0", with: "vals[i]"));
            uint p = st_lower_bound(setVals, setCount, k);
            if (p < setCount && \(toKey.replacingOccurrences(of: "$0", with: "setVals[p]")) == k) bits |= (1u << j);
        }
        out[w] = bits;
    }

    // `firstIdx[p]` is the position in the caller's set array of the first occurrence of the
    // p-th distinct value, so the output is an index into that array, as Arrow's `index_in`.
    kernel void st_index_in(device const \(K)* vals [[buffer(0)]],
                            device const uchar* validity [[buffer(1)]],
                            device const uint* nPtr [[buffer(2)]],
                            device const \(K)* setVals [[buffer(3)]],
                            constant uint& setCount [[buffer(4)]],
                            constant uint& hasValidity [[buffer(5)]],
                            device const int* firstIdx [[buffer(6)]],
                            device int* out [[buffer(7)]],
                            device uchar* outValid [[buffer(8)]],
                            uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        out[i] = 0;
        outValid[i] = 0;
        if (hasValidity && !bit_get(validity, i)) return;
        \(K) k = \(toKey.replacingOccurrences(of: "$0", with: "vals[i]"));
        uint p = st_lower_bound(setVals, setCount, k);
        if (p < setCount && \(toKey.replacingOccurrences(of: "$0", with: "setVals[p]")) == k) {
            out[i] = firstIdx[p];
            outValid[i] = 1;
        }
    }
    """ }
}
