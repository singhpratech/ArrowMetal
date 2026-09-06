import Foundation

/// MSL for Arrow `match_like`: a complete SQL `LIKE` matcher on the GPU (`Kernels/StringLike.swift`).
///
/// `Kernels/Regex.swift` used to route only the four anchored shapes — `abc`, `abc%`, `%abc`,
/// `%abc%` — to a byte kernel and send everything with a `_` or an interior `%` to ICU. This kernel
/// takes every pattern instead: the host compiles it once into a tiny byte program and one thread per
/// row runs that program with the classic greedy-plus-backtrack wildcard algorithm.
///
/// ## The program
///
/// | opcode | layout | meaning |
/// |---|---|---|
/// | 0 | `0, len, bytes…` | a literal run of `len` ≤ 255 bytes |
/// | 1 | `1` | `%` — any run of characters, including none |
/// | 2 | `2` | `_` — exactly one character |
///
/// A backslash escapes `%`, `_` and itself, so those become literal bytes at compile time and the
/// program has no escape opcode. Runs of `%` collapse into one.
///
/// ## Matching
///
/// `lk_match` walks the program with a single remembered backtrack point — the most recent `%` and
/// the input position it was last tried at. On a mismatch it advances that position by one
/// **code point** and restarts from just after the `%`. That is what makes `_` and `%` count
/// characters rather than bytes: `_` consumes one whole UTF-8 sequence, and `%` only ever resumes on
/// a code-point boundary, so `"_"` matches `"é"` and not half of it. SQL `LIKE` anchors the whole
/// value, so a match is only reported when the program and the row end together.
///
/// One remembered `%` is enough: with a single wildcard class that can match anything, the greedy
/// algorithm with one backtrack point is complete — an outer `%` never needs to be retried once an
/// inner one has been, because anything the outer one would have to give up, the inner one can take.
enum StringLikeSource {
    static let source = KernelSource.prelude + """

    #define LK_LITERAL 0u
    #define LK_ANY     1u
    #define LK_ONE     2u

    // The byte width of the UTF-8 sequence at p (1 for a stray or truncated byte).
    inline int lk_adv(device const uchar* d, int p, int end) {
        uchar b0 = d[p];
        int w = 1;
        if (b0 >= 0xF0u) w = 4; else if (b0 >= 0xE0u) w = 3; else if (b0 >= 0xC0u) w = 2;
        if (p + w > end) w = 1;
        return w;
    }

    inline bool lk_match(device const uchar* d, int start, int end,
                         device const uchar* prog, uint progLen) {
        int p = start;
        uint t = 0;
        uint starT = 0xFFFFFFFFu;               // program position just after the last '%'
        int starP = -1;                         // input position that '%' was last tried at
        while (true) {
            bool advanced = false;
            if (t < progLen) {
                uchar op = prog[t];
                if (op == LK_ANY) { starT = t + 1u; starP = p; t += 1u; advanced = true; }
                else if (op == LK_ONE) {
                    if (p < end) { p += lk_adv(d, p, end); t += 1u; advanced = true; }
                } else {
                    uint len = (uint)prog[t + 1];
                    bool ok = (p + (int)len <= end);
                    if (ok) for (uint j = 0; j < len; j++) if (d[p + j] != prog[t + 2u + j]) { ok = false; break; }
                    if (ok) { p += (int)len; t += 2u + len; advanced = true; }
                }
            } else if (p == end) return true;
            if (advanced) continue;
            if (starP < 0) return false;         // no '%' to give ground
            if (starP >= end) return false;      // and nothing left to give
            starP += lk_adv(d, starP, end);
            p = starP;
            t = starT;
        }
    }

    // One 32-bit word of the answer per thread, as the other predicate kernels pack them.
    kernel void lk_like(device const int* offsets [[buffer(0)]], device const uchar* data [[buffer(1)]],
                        device const uint* nPtr [[buffer(2)]], device const uchar* prog [[buffer(3)]],
                        constant uint& progLen [[buffer(4)]], device uint* out [[buffer(5)]],
                        uint w [[thread_position_in_grid]]) {
        uint n = *nPtr, base = w * 32u;
        if (base >= n) return;
        uint limit = min(32u, n - base), bits = 0u;
        for (uint j = 0; j < limit; j++) {
            uint i = base + j;
            if (lk_match(data, offsets[i], offsets[i + 1], prog, progLen)) bits |= (1u << j);
        }
        out[w] = bits;
    }
    """
}
