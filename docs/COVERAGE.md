# Apache Arrow compute coverage

ArrowMetal 0.1.0 measured against the [Apache Arrow C++ compute function
list](https://arrow.apache.org/docs/cpp/compute.html) and the [Arrow columnar type
list](https://arrow.apache.org/docs/format/Columnar.html).

Every row below was decided by reading the source in this repository, not by intent. If a row says **GPU**
there is a Metal kernel behind a public API call; if it says **CPU** the work happens on the host but the
call exists; if it says anything else, the function is not there today. Rows carry the file that decides
them so a claim can be checked in one jump.

## Summary

| Arrow function category | GPU | CPU | Partial | Planned | In progress | Not planned | Rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| Aggregations — scalar | 4 | 4 | 1 | 0 | 0 | 12 | 21 |
| Aggregations — grouped (`hash_*`) | 0 | 0 | 6 | 1 | 1 | 7 | 15 |
| Element-wise arithmetic | 5 | 0 | 8 | 1 | 0 | 2 | 16 |
| Bit-wise and shifts | 4 | 0 | 2 | 0 | 0 | 0 | 6 |
| Comparisons | 8 | 0 | 0 | 0 | 0 | 0 | 8 |
| Logical | 4 | 0 | 0 | 0 | 0 | 3 | 7 |
| String predicates | 1 | 0 | 0 | 0 | 0 | 3 | 4 |
| String transforms | 9 | 0 | 2 | 1 | 0 | 7 | 19 |
| String containment and matching | 7 | 0 | 0 | 2 | 0 | 1 | 10 |
| Temporal | 0 | 0 | 0 | 0 | 1 | 5 | 6 |
| Conversions and casts | 0 | 1 | 2 | 0 | 1 | 2 | 6 |
| Selections | 4 | 0 | 1 | 0 | 0 | 0 | 5 |
| Containment / set lookup | 2 | 0 | 0 | 0 | 0 | 1 | 3 |
| Sorts and partitions | 3 | 1 | 1 | 0 | 0 | 2 | 7 |
| Structural and conditional | 4 | 0 | 2 | 0 | 0 | 7 | 13 |
| Associative transforms | 2 | 0 | 0 | 0 | 3 | 0 | 5 |
| Pairwise and cumulative | 1 | 0 | 1 | 0 | 0 | 3 | 5 |
| Hashing | 1 | 0 | 0 | 0 | 0 | 1 | 2 |
| **Total (compute functions)** | **59** | **6** | **26** | **5** | **6** | **56** | **158** |
| Arrow types (matrix below) | 5 | 0 | 3 | 1 | 6 | 11 | 26 |

Interop uses a separate vocabulary and is counted apart: 6 shipped, 1 partial, 3 planned, 1 in progress
(11 rows).

**The scope ArrowMetal 0.1.0 claims 100% of:** flat analytics on primitive, boolean and string columns —
`sum`/`min`/`max`/`mean`, the six comparisons, wrapping `add`/`subtract`/`multiply`/`divide`, boolean
`and`/`or`/`not`, `filter`/`take`/`slice`, numeric `cast`, single-key `sort`/`argsort` and a partial-selection top-k, group-by
`count`/`sum`/`mean`/`min`/`max` over dense integer keys for every primitive value type, `is_null`/`is_valid`/`fill_null`/`drop_null`/
`if_else`/`coalesce`/`is_in`/`index_in`/`and_kleene`/`or_kleene`, `utf8` length/`equals`/`starts_with`/`ends_with`/
`contains`/`count_substring`/`find_substring`/murmur3 hash/`dictionary_encode`, the ASCII case, trim, pad, slice, repeat,
replace, reverse, join and `ascii_is_*` transforms, and Arrow C Data, C Device and C Stream interop for all
of them — over `int8/16/32/64`, `uint8/16/32/64`, `float32`, `float64`, `bool` and `utf8`, null-aware with
Arrow semantics and checked against a CPU oracle in the test suite.

**The scope it does not claim:** decimals; compute over nested types (lists, structs, maps, unions); regex
and Unicode-table string work (full case folding beyond Latin-1 Supplement and Latin Extended-A,
normalisation, Unicode-whitespace trimming, splitting); window and pairwise functions
(`cumulative_sum`/`_min`/`_max` do ship — see Pairwise and cumulative — but `cumulative_prod`,
`cumulative_mean` and `pairwise_diff` do not); temporal component extraction, temporal arithmetic, timezones and
`strftime`/`strptime`; statistical aggregates (`stddev`, `variance`, `quantile`, `mode`, `tdigest`);
set lookup over strings; `case_when`, `replace_with_mask` and the forward/backward null fills; checked
arithmetic and overflow-erroring casts. The long form is at the bottom of this file.

## Legend

| Status | Meaning |
|---|---|
| **GPU** | A Metal kernel, reachable from the public Swift API, the C ABI, or both. |
| **CPU** | Implemented and reachable through the same ArrowMetal API, but the work runs on the host. |
| **Partial** | Available with a stated limitation; the note says exactly what is missing. |
| **Planned** | Not implemented; a [ROADMAP](../ROADMAP.md) item covers it (linked in the note). |
| **In progress** | Being implemented this week on a concurrent branch; not in 0.1.0 as published here. |
| **Not planned** | Not implemented and not on the roadmap. The note says whether it is out of scope for a GPU kernel library or simply unclaimed. |

Counts in the summary are counts of **rows**. A row covers one Arrow function unless it names several
(for example the twenty `ascii_is_*` / `utf8_is_*` predicates share one row).

## Aggregations — scalar

| Arrow function | Status | Notes |
|---|---|---|
| `sum` | **GPU** | `Kernels/Reductions.swift`. Threadgroup partials, host finalise, no atomics. Integers accumulate in Int64/UInt64 and wrap; Float32 accumulates per thread in `float` and finalises in `double`, so the last ulp can differ from a strictly sequential double sum; Float64 uses a software IEEE-754 binary64 adder on the GPU (`Kernels/DoubleMath.swift`). Returns nil when there is no valid value, matching Arrow. |
| `product` | **Not planned** | No roadmap item. Same reduction shape as `sum`; nobody has claimed it. |
| `mean` | **GPU** | GPU sum divided by the valid count on the host (`Reductions.swift`). |
| `min` | **GPU** | NaN is skipped; all-NaN returns null, matching Arrow's `min_max`. Float64 reduces on order-preserving 64-bit keys. |
| `max` | **GPU** | Same as `min`. |
| `min_max` | **Partial** | No fused kernel: call `min()` and `max()`, which is two passes over the data. |
| `count` (valid values) | **CPU** | `validCount` = `length - nullCount`; the null count comes from a host popcount over the validity bitmap (`MetalArray.swift`, `Bitmap.popcount`). O(1) once the count is known. |
| `count_all` (rows) | **CPU** | `length`, O(1) metadata. Inside an open batch, reading it forces a sync point. |
| `count_distinct` | **Not planned** | No roadmap item. Would follow the numeric `unique` work now in flight. |
| `any` | **CPU** | `MetalBooleanArray.any` is a host popcount of `values & validity` (`Slice.swift`, `MetalArray.swift`). There is no kernel for it, and the README's kernel list says so. |
| `all` | **CPU** | As `any`; true for empty and all-null input, matching Arrow's `all` with `skip_nulls`. |
| `index` | **Not planned** | No roadmap item. |
| `first` / `last` / `first_last` | **Not planned** | No roadmap item. Element access (`array[i]`) is not the same function — it does not skip nulls. |
| `mode` | **Not planned** | No roadmap item; needs a hash table, which arrives with hash group-by. |
| `quantile` | **Not planned** | No roadmap item. |
| `approximate_median` | **Not planned** | No roadmap item. |
| `tdigest` | **Not planned** | No roadmap item. |
| `stddev` | **Not planned** | No roadmap item. |
| `variance` | **Not planned** | No roadmap item. |
| `skew` | **Not planned** | No roadmap item. |
| `kurtosis` | **Not planned** | No roadmap item. |

## Aggregations — grouped (`hash_*`)

All grouped aggregates go through `GroupBy`, which takes **dense integer keys in `[0, keyCount)`** — the
shape a dictionary encoding produces. Keys outside the range and null keys are skipped. There are two
implementations behind it.

The **atomic** path (`Kernels/GroupBy.swift`) uses privatised threadgroup tables up to 1024 keys and device
atomics beyond; 64-bit sums use split 32-bit atomics with carry because MSL has no 64-bit atomics. That is
also its ceiling: no 64-bit min/max, no Float64 values.

The **segmented** path (`Kernels/Segmented.swift`) removes atomics from the aggregation. It argsorts the
keys once, which makes each group a contiguous run of the sorted order, then reduces one run per
threadgroup. `segments()` returns the sorted order so several aggregates share the one sort. It covers
exactly what the atomic path could not: Float64 sums and means (through the software binary64 adder in
`Kernels/DoubleMath.swift`), Float32 sums and means accumulated in Float64, and 64-bit min/max.

| Arrow function | Status | Notes |
|---|---|---|
| `hash_sum` | **Partial** | GPU, dense integer keys only — but all value types now. Integers through the atomic `sum`; Float32 through `sumFloat` (Float32 accumulation, host finalise) or `sumFloatAsDouble` (Float64 accumulation, GPU); Float64 through `sumDouble`, which adds with the software binary64 adder on the GPU. |
| `hash_mean` | **Partial** | GPU, dense keys. Integer values through `mean` (GPU sum + GPU count, host division); Float32 and Float64 through `meanFloat` / `meanDouble`, which sum and divide entirely on the GPU. |
| `hash_min` | **Partial** | GPU, dense keys, all ten primitive value types. 32-bit and narrower use the atomic `min`; Int64, UInt64 and Float64 use `min64` on the segmented path, since MSL has no 64-bit atomic min/max. `min64` forwards narrower types to `min`, so it is safe to call for any type. |
| `hash_max` | **Partial** | As `hash_min` (`max` / `max64`). |
| `hash_count` (valid values per key) | **Partial** | GPU, dense keys. |
| `hash_count_all` (rows per key) | **Partial** | GPU, dense keys. |
| `hash_min_max` | **Not planned** | No roadmap item; call `hash_min` and `hash_max`. |
| `hash_any` / `hash_all` | **Not planned** | No roadmap item. |
| `hash_product` | **Not planned** | No roadmap item. |
| `hash_stddev` / `hash_variance` | **Not planned** | No roadmap item. |
| `hash_count_distinct` / `hash_distinct` | **Not planned** | No roadmap item. |
| `hash_first` / `hash_last` / `hash_one` / `hash_list` | **Not planned** | No roadmap item. |
| `hash_approximate_median` / `hash_tdigest` | **Not planned** | No roadmap item. |
| Group-by over arbitrary (non-dense) keys | **Planned** | [ROADMAP → Medium term → Group-by](../ROADMAP.md#medium-term): "hash group-by for arbitrary keys, 64-bit min/max". The 64-bit min/max half of that item is done (`min64` / `max64`); the caller must still dictionary-encode arbitrary keys first, which for `utf8` is now itself a GPU pass. |
| Hash join (Acero, not a compute function) | **In progress** | A concurrent branch is building a GPU hash join this week. Not in 0.1.0 as published here. |

## Element-wise arithmetic

`modulo` (`%`, C remainder semantics, `x % 0` defined as 0) is an ArrowMetal extension rather than an
Arrow function name, so it has no row of its own; it lives beside `power` in `Kernels/Rounding.swift`
and is reachable as `am_binary(op 5)` and `.modulo()` / `__mod__` in Python.

| Arrow function | Status | Notes |
|---|---|---|
| `add` | **GPU** | Scalar and array forms, vectorised 4-wide (`Kernels/Arithmetic.swift`). Integer overflow wraps, like Arrow's unchecked `add`. Float64 runs a software IEEE-754 binary64 adder on the GPU, bit-exact against Swift's `Double`. |
| `subtract` | **GPU** | As `add`. |
| `multiply` | **GPU** | As `add`. |
| `divide` | **Partial** | GPU, but integer division by zero is **defined as 0** here (`KernelSource.swift`, matched by the CPU oracle in `ArrowPrimitive.swift`) rather than raising. Check this against Arrow's `divide` before relying on it. Float division follows IEEE. |
| `add_checked` / `subtract_checked` / `multiply_checked` / `divide_checked` | **Planned** | [ROADMAP → Near term → Checked arithmetic](../ROADMAP.md#near-term-good-first-contributions): report overflow and division by zero like Arrow. |
| `negate` / `negate_checked` | **Partial** | `negate()` is GPU over all ten primitives (`Kernels/Rounding.swift`); integers wrap, so `negate(int8 -128)` is `-128`, and unsigned negation is modular. Float64 flips the sign bit, exactly. `negate_checked` is not implemented — see the checked-arithmetic row above. |
| `abs` / `abs_checked` | **Partial** | `abs()` is GPU over all ten primitives; `abs(int8 -128)` wraps to `-128`, unsigned is the identity, Float64 clears the sign bit exactly. `abs_checked` is not implemented. |
| `sign` | **GPU** | `sign()` over all ten primitives: -1/0/1 in the input's own type (0 or 1 for unsigned). Floats keep NaN and both signed zeros, matching Arrow. |
| `power` / `power_checked` | **Partial** | `power()`, scalar and array forms, GPU. Integers use repeated squaring and wrap; a **negative exponent is defined as 0** here (Arrow raises), and `0^0` is 1. Float32 uses MSL `pow`. Not implemented for Float64 (it would have to drop to float precision) and `power_checked` is not implemented. |
| `sqrt` / `sqrt_checked` | **Partial** | `sqrt()` is GPU on float columns; an integer column throws rather than being promoted to float64 as Arrow does — cast first. On Float64 the value is converted to `float`, evaluated and widened back, so roughly 7 correct significant digits (documented in `Kernels/RoundingSource.swift`). `sqrt_checked` is not implemented. |
| `exp` | **Partial** | `exp()` on float columns only, same Float64 precision caveat as `sqrt`. |
| `ln` / `log2` / `log10` / `log1p` / `logb` (and `_checked`) | **Partial** | `ln()`, `log2()`, `log10()` are GPU on float columns, with the same Float64 precision caveat. `log1p`, `logb` and the `_checked` forms are not implemented. |
| `sin` / `cos` / `tan` / `asin` / `acos` / `atan` / `atan2` | **Not planned** | No roadmap item. |
| `sinh` / `cosh` / `tanh` / `asinh` / `acosh` / `atanh` | **Not planned** | No roadmap item. |
| `ceil` / `floor` / `trunc` | **GPU** | `Kernels/Rounding.swift`, all ten primitives. On an integer column they are the identity and keep its type (Arrow promotes to float64 instead). Float64 clears the fractional mantissa bits on the GPU, exactly, signed zeros and infinities included. |
| `round` / `round_to_multiple` / `round_binary` | **Partial** | `round()` only, and only one mode: **halves away from zero** (Arrow's `HALF_TOWARDS_INFINITY`, not its `HALF_TO_EVEN` default). Exact on Float64. `round_to_multiple`, `round_binary` and the other nine `RoundMode`s are not implemented. |

## Bit-wise and shifts

| Arrow function | Status | Notes |
|---|---|---|
| `bit_wise_and` | **GPU** | `Kernels/Bitwise.swift`, scalar and array forms, over the eight integer types. Note the two different things called "and": boolean `and`/`or`/`not` over *packed bitmaps* are the Logical section below; these are value-level ops on integer columns. A float column throws. |
| `bit_wise_or` | **GPU** | As `bit_wise_and`. |
| `bit_wise_xor` | **GPU** | As `bit_wise_and`. |
| `bit_wise_not` | **GPU** | `bitwiseNot()`; validity is shared zero-copy with the input. |
| `shift_left` / `shift_left_checked` | **Partial** | `shiftLeft()` is GPU, scalar and array forms. Arrow raises on a shift count that is negative or at least the bit width and C leaves it undefined; ArrowMetal **defines** it as 0 instead (`Kernels/BitwiseSource.swift`, matched by the test oracle). In-range shifts drop the bits that leave the top. `shift_left_checked` is not implemented. |
| `shift_right` / `shift_right_checked` | **Partial** | `shiftRight()` is arithmetic on signed columns and logical on unsigned ones. An out-of-range count is **defined** as the sign fill: 0 for a non-negative value or an unsigned column, -1 for a negative one. `shift_right_checked` is not implemented. |

## Comparisons

| Arrow function | Status | Notes |
|---|---|---|
| `equal` | **GPU** | `Kernels/Compare.swift`, scalar and array forms, output is a packed Arrow boolean bitmap written one 32-bit word per thread. Float64 compares on order-preserving bit patterns; IEEE semantics for NaN. |
| `not_equal` | **GPU** | As `equal`. |
| `less` | **GPU** | As `equal`. |
| `less_equal` | **GPU** | As `equal`. |
| `greater` | **GPU** | As `equal`. |
| `greater_equal` | **GPU** | As `equal`. |
| `max_element_wise` | **GPU** | `maxElementWise(_:)` (`Kernels/Rounding.swift`), two columns of the same type. Nulls are skipped, which is Arrow's `skip_nulls` default: a null on one side yields the other side's value and only two nulls make a null, so the output validity is the OR of the inputs', not the AND. NaN loses, as it does in the `min`/`max` reductions. Float64 compares on order-preserving bit patterns, exactly. |
| `min_element_wise` | **GPU** | As `max_element_wise`. |

## Logical

| Arrow function | Status | Notes |
|---|---|---|
| `and` | **GPU** | Word-wise bitmap AND (`Kernels/BitmapOps.swift`). Nulls propagate — this is Arrow's `and`, not `and_kleene`. |
| `or` | **GPU** | As `and`. |
| `invert` (`not`) | **GPU** | Validity is shared zero-copy with the input. |
| `xor` | **Not planned** | No roadmap item; one line of MSL away from the existing bitmap kernels. |
| `and_not` | **Not planned** | No roadmap item. |
| `and_kleene` / `or_kleene` | **GPU** | `Kernels/Structural.swift`, one thread per 32-bit word: the value words are `a & b` / `a | b` and the validity word is computed from both operands' validity, so `false AND null` is `false` and `true OR null` is `true`. With no nulls on either side the call falls through to the plain `and` / `or` kernel. `andKleene` / `orKleene` in Swift, `am_and_kleene` / `am_or_kleene` in C, `and_kleene` / `or_kleene` in Python. |
| `and_not_kleene` | **Not planned** | No roadmap item; expressible as `a.andKleene(b.not())` only when `b` has no nulls, so it needs its own kernel. |

## String predicates

| Arrow function | Status | Notes |
|---|---|---|
| `ascii_is_alnum` / `_alpha` / `_decimal` / `_space` / `_lower` / `_upper` | **GPU** | `Kernels/StringTransforms.swift`: `asciiIsAlnum()`, `asciiIsAlpha()`, `asciiIsDigit()` (Arrow's `_decimal`), `asciiIsSpace()`, `asciiIsLower()`, `asciiIsUpper()` → packed boolean bitmap, one 32-bit word per thread. Python/Arrow semantics: the empty string is false everywhere, and `_lower`/`_upper` need at least one cased ASCII character and none of the opposite case, treating bytes ≥ 0x80 as uncased. Nulls propagate. |
| `ascii_is_printable` / `ascii_is_title` | **Not planned** | No roadmap item. The six predicates above share one kernel and a seventh case is a two-line addition. |
| `utf8_is_alnum` / `_alpha` / `_decimal` / `_digit` / `_lower` / `_numeric` / `_printable` / `_space` / `_title` / `_upper` | **Not planned** | No roadmap item; needs Unicode tables in the kernel. The `ascii_*` predicates above are byte-wise and return false for any non-ASCII byte, which is not the same function. |
| `string_is_ascii` | **Not planned** | No roadmap item. |

## String transforms

`MetalStringArray` (`Sources/ArrowMetal/MetalStringArray.swift`) is Arrow `utf8`: validity bitmap, int32
offsets, data bytes. Everything below is byte-wise and case-sensitive except where a row says otherwise:
the `utf8_*` case, slice, pad and reverse rows work in UTF-8 code points.

The transforms in `Kernels/StringTransforms.swift` produce new string arrays whose bytes are
data-dependent, so each runs the same two-pass shape: one kernel writes the output byte length of every
row, `exclusiveScanToOffsets` scans those into the Arrow offsets buffer on the GPU, and a second kernel
writes the bytes. Both passes call one MSL routine (`tf_apply` in `Kernels/StringTransformSource.swift`),
so a length and the bytes that fill it cannot disagree. The validity bitmap is shared with the input
zero-copy and a null row emits no bytes. All of them are reachable from Swift, from the C ABI
(`am_str_transform`, op table in `include/arrowmetal.h`) and from Python.

| Arrow function | Status | Notes |
|---|---|---|
| `binary_length` | **GPU** | `byteLength()` → Int32, one thread per string, null in / null out. |
| `utf8_length` | **GPU** | `charLength()` counts UTF-8 code points. |
| `ascii_lower` / `ascii_upper` / `ascii_swapcase` / `ascii_capitalize` | **GPU** | `asciiLower()`, `asciiUpper()`, `asciiSwapcase()`, `asciiCapitalize()`. Byte-wise over `a`–`z` / `A`–`Z`; every other byte, UTF-8 continuation bytes included, is copied through, so the output is always valid UTF-8 and the same length as the input. |
| `utf8_lower` / `utf8_upper` | **Partial** | `utf8Lower()` / `utf8Upper()`, GPU, **simple (1:1 code point) case mapping over three blocks only**: Basic Latin; Latin-1 Supplement U+00C0–U+00DE and U+00E0–U+00FE minus U+00D7 (×) and U+00F7 (÷), plus U+00FF ↔ U+0178; and Latin Extended-A U+0100–U+017F in its alternating pairs, with U+0130 (İ) → `i`, U+0131 (ı) → `I` and U+017F (ſ) → `S` — three mappings that shrink a string from two bytes to one, which is why the two-pass shape is not optional. Everything above U+017F is copied through byte-for-byte (Greek, Cyrillic, CJK, emoji). The multi-character expansions Arrow's utf8proc applies are **not** implemented: U+00DF (ß → `SS`), U+0149 (ŉ → `ʼN`) and U+00B5 (µ → U+039C) pass through unchanged. Full Unicode case folding stays on the [ROADMAP](../ROADMAP.md#medium-term). |
| `utf8_capitalize` / `ascii_title` / `utf8_title` | **Not planned** | No roadmap item. `ascii_capitalize` is covered by the row above; the Unicode form needs the same tables `utf8_upper` stops short of, and the title-case functions need word segmentation on top of that. |
| `replace_substring_regex` / `extract_regex` / `extract_regex_span` | **Planned** | [ROADMAP → Medium term → Strings](../ROADMAP.md#medium-term) lists "regex" as open. Nothing regex-shaped exists today, and a backtracking engine is a poor fit for SIMT — treat this as unclaimed until a design lands. |
| `ascii_reverse` / `binary_reverse` / `utf8_reverse` | **GPU** | `reverse()` reverses **code points**, not grapheme clusters: a combining mark or a ZWJ emoji sequence comes back in reverse code point order. That is `utf8_reverse`; `binary_reverse` (byte order) is not exposed separately. |
| `replace_substring` | **GPU** | `replaceSubstring(_:with:maxReplacements:)`, non-overlapping and left to right, `maxReplacements` < 0 meaning all. Byte-wise, so a multi-byte pattern works. An empty pattern is the identity, matching Foundation's `replacingOccurrences(of: "", with:)` rather than Python's insert-everywhere. |
| `binary_replace_slice` / `utf8_replace_slice` | **Not planned** | No roadmap item; expressible as `sliceCodeunits` + `concat`. |
| `binary_slice` / `utf8_slice_codeunits` | **Partial** | `sliceCodeunits(start:stop:)`, GPU, **`step == 1` only** — Arrow's negative and non-unit steps are not implemented. Indices are code points, negative values count from the end, both ends clamp into range, and slices always land on UTF-8 boundaries. `binary_slice` (byte indices) is not exposed separately. |
| `ascii_trim*` / `ascii_ltrim*` / `ascii_rtrim*` (whitespace and character set) | **GPU** | `trim()`/`ltrim()`/`rtrim()` strip ASCII whitespace (space, `\t`, `\n`, `\v`, `\f`, `\r`); `trim(characters:)`/`ltrim(characters:)`/`rtrim(characters:)` strip any byte in an ASCII set, and reject a non-ASCII set rather than splitting a UTF-8 sequence. Bytes ≥ 0x80 are never trimmed. |
| `utf8_trim*` (Unicode whitespace / character set) | **Not planned** | No roadmap item: the Unicode whitespace and character classes need the tables `utf8_lower`/`utf8_upper` also stop short of. The ASCII forms above are a different function, not this one. |
| `ascii_lpad` / `ascii_rpad`, `utf8_lpad` / `utf8_rpad` | **GPU** | `padLeft(width:pad:)` / `padRight(width:pad:)`. `width` counts **code points** (the `utf8_*` behaviour) and `pad` must be exactly one character; strings already at or over `width` are returned unchanged. |
| `ascii_center` / `utf8_center` | **Not planned** | No roadmap item; the same kernel with the pad split across both ends. |
| `binary_repeat` | **GPU** | `repeat(_ n:)`, `n == 0` giving empty strings and `n < 0` raising. |
| `binary_join_element_wise` | **GPU** | `concat(_:separator:)` over two equal-length arrays, one scalar separator. Validities are ANDed on the GPU, so a null on either side gives a null output — Arrow's default `EMIT_NULL` null handling; the `REPLACE`/`SKIP` options are not implemented. |
| `binary_join` (list of strings) | **Not planned** | Out of scope for 0.1.0: the input is a list array, which ArrowMetal has no type for. |
| `split_pattern` / `split_pattern_regex` / `ascii_split_whitespace` / `utf8_split_whitespace` | **Not planned** | No roadmap item; the output is a list array, which ArrowMetal has no type for. |
| `utf8_normalize` | **Not planned** | Out of scope for a GPU kernel library: full Unicode normalisation tables in MSL buy nothing over the CPU. |

## String containment and matching

| Arrow function | Status | Notes |
|---|---|---|
| `equal` (string vs. string scalar) | **GPU** | `equals(_ s: String)` → boolean bitmap. |
| `equal` (string vs. string array) | **GPU** | `equals(_ other: MetalStringArray)`; validities are AND-ed on the GPU. |
| `match_substring` | **GPU** | `contains(_:)`, byte-wise, case-sensitive, no `ignore_case` option. |
| `starts_with` | **GPU** | `startsWith(_:)`. |
| `ends_with` | **GPU** | `endsWith(_:)`. |
| `match_substring_regex` / `match_like` | **Planned** | Regex is listed as open under [ROADMAP → Medium term → Strings](../ROADMAP.md#medium-term). |
| `count_substring_regex` / `find_substring_regex` | **Planned** | Same roadmap item. |
| `count_substring` | **GPU** | `countSubstring(_:)` → Int32, non-overlapping occurrences, byte-wise and case-sensitive (no `ignore_case`). An empty pattern counts the code point boundaries, `charLength() + 1`, matching Arrow. Nulls propagate. |
| `find_substring` | **GPU** | `findSubstring(_:)` → Int32, the **byte** offset of the first occurrence or -1 when absent; an empty pattern finds 0. Byte-wise and case-sensitive. Nulls propagate. |
| `index_in` / `is_in` (strings) | **Not planned** | No roadmap item; see Containment below. |

## Temporal

| Arrow function group | Status | Notes |
|---|---|---|
| Component extraction: `year`, `month`, `day`, `day_of_week`, `day_of_year`, `hour`, `minute`, `second`, `subsecond`, `millisecond`, `microsecond`, `nanosecond`, `quarter`, `week`, `iso_week`, `iso_year`, `iso_calendar`, `us_week`, `us_year`, `year_month_day`, `is_leap_year`, `is_dst` | **Not planned** | No roadmap item. Once temporal types land (below) the existing integer kernels apply to the underlying values, but calendar decomposition itself is unwritten. |
| Differences: `days_between`, `hours_between`, `minutes_between`, `seconds_between`, `weeks_between`, `months_between`, `quarters_between`, `years_between`, `*_interval_between` | **Not planned** | No roadmap item. |
| Rounding: `ceil_temporal`, `floor_temporal`, `round_temporal` | **Not planned** | No roadmap item. |
| Timezones: `assume_timezone`, `local_timestamp` | **Not planned** | Out of scope for a GPU kernel library: the tz database is host data. |
| `strftime` / `strptime` | **Not planned** | Out of scope: string formatting and parsing against locale/tz data belong on the CPU. |
| Temporal **types** (`date32`, `date64`, `time32`, `time64`, `timestamp`, `duration`) | **In progress** | A concurrent branch is adding temporal type import/export and routing them onto the existing fixed-width integer kernels this week. Not in 0.1.0 as published here: `arrowPrimitiveType(forFormat:)` accepts only `c C s S i I l L f g` today. |

## Conversions and casts

| Arrow function | Status | Notes |
|---|---|---|
| `cast` (numeric → numeric) | **Partial** | `Kernels/Cast.swift`, GPU, across all ten primitives. Unchecked only: integer narrowing wraps and float → int truncates toward zero, which is Arrow's `safe=false`. There is no `safe=true` overflow-erroring cast. |
| `cast` involving Float64 | **CPU** | `Dispatch.runsOnGPU` excludes `Double`, so any cast with Float64 on either side runs a host loop. Note that Float64 *arithmetic* and *reductions* do run on the GPU — the cast is the exception. |
| `cast` boolean ↔ integer | **Partial** | `MetalBooleanArray.toUInt8Array()` is a public GPU unpack (bitmap → uint8). The reverse packing exists but is internal. |
| `cast` string ↔ numeric / temporal | **Not planned** | No roadmap item. |
| `cast` to/from decimal | **Not planned** | Out of scope: see the decimal rows in the type matrix. |
| `cast` dictionary | **In progress** | Follows the dictionary type work in flight this week. |

## Selections

| Arrow function | Status | Notes |
|---|---|---|
| `filter` / `array_filter` | **GPU** | `Kernels/Filter.swift`: per-block popcount, GPU scan, scatter, validity pack — all in one command buffer. Null mask entries drop the element (Arrow's `null_selection_behavior = "drop"`); the `"emit_null"` option is not implemented. Works for primitives, booleans and `utf8`. |
| `filter(where:)` (fused predicate + compaction) | **GPU** | ArrowMetal extension, not an Arrow function: the comparison is evaluated inside the counting pass so no boolean array is materialised. |
| `take` / `array_take` | **GPU** | Int32/Int64/UInt32 index arrays. A null index yields a null output element; out-of-range indices set a GPU error flag that is raised after the dispatch. Strings gather through offsets + a GPU byte copy. |
| `drop_null` | **GPU** | `Kernels/Structural.swift`: `is_valid` followed by the existing `filter` compaction, so it is one command buffer with no host round trip. An array with no validity bitmap is returned unchanged. `dropNull()` in Swift (primitive and boolean), `am_drop_null` in C, `drop_null()` in Python. |
| `slice` (array method, not a compute function) | **Partial** | Zero-copy `MTLBuffer` view when the offset is a multiple of 32 (keeps bitmap words and values aligned for the kernels); otherwise one host copy (`Sources/ArrowMetal/Slice.swift`). |

## Containment / set lookup

| Arrow function | Status | Notes |
|---|---|---|
| `is_in` | **GPU** | `Kernels/Structural.swift`, all ten primitive types. The value set is reduced to its sorted distinct non-null values with the existing `unique()`, and each element binary-searches it on the GPU (no hash table). Nulls in the set are ignored and a null element never matches, so the result never has nulls — Arrow's `null_matching_behavior = "skip"`. Float equality is Arrow value equality, as in `unique()`: every NaN is one value and `-0.0` equals `0.0`. `isIn(_:)` in Swift (a `[T]` or a `MetalArray<T>`), `am_is_in` in C, `is_in()` in Python. Strings are not covered. |
| `index_in` | **GPU** | Same search, returning the int32 position in the caller's set array of each element's **first** occurrence there, and null where the element is null or absent. The unique-rank-to-first-row map is a group-by min over the set's dictionary codes, so it too runs on the GPU. `indexIn(_:)` in Swift, `am_index_in` in C, `index_in()` in Python. |
| `indices_nonzero` | **Not planned** | No roadmap item; the filter kernel already contains the scan-and-scatter it needs. |

## Sorts and partitions

| Arrow function | Status | Notes |
|---|---|---|
| `array_sort_indices` | **GPU** | `Kernels/Sort.swift`: LSD radix sort, 4 passes for 32-bit keys and 8 for 64-bit, stable. Ascending or descending. Total order for floats (NaN after +inf). |
| `sort_indices` (multiple sort keys) | **Partial** | One key only. [ROADMAP → Medium term → Sort](../ROADMAP.md#medium-term) lists "multi-column sort keys" as open. |
| Sorted copy (`sorted()`) and `MetalRecordBatch.sorted(by:)` | **GPU** | Argsort then take. Not an Arrow compute function name, but it is what callers use. |
| Nulls-last placement in the sorted index array | **CPU** | The radix sort runs on the GPU; the stable partition that moves null rows to the end is a host pass over the index array (`Sort.swift`). |
| `select_k_unstable` (top-k) | **GPU** | `Kernels/TopK.swift`: for k ≤ 1024 each threadgroup keeps the best k of its own block in threadgroup memory (threshold plus a bitonic compaction), and one radix sort over the `blocks * k` candidates orders the winners. Same total order as `argsort` — value key, ties by row — so the result is index-for-index what the full sort would give. Larger k, and the case where fewer than k rows are non-null, fall back to the argsort-and-slice. Single key. 50M Int64, k=100: 4.0 ms against 128 ms for the full sort. |
| `partition_nth_indices` | **Not planned** | No roadmap item. |
| `rank` / `rank_quantile` / `rank_normal` | **Not planned** | No roadmap item. |

## Structural and conditional transforms

| Arrow function | Status | Notes |
|---|---|---|
| `fill_null` | **GPU** | `Kernels/Structural.swift`, one thread per element, all ten primitive types plus `bool`; the result drops the validity bitmap. Float64 moves as a raw 64-bit value, so no software binary64 is involved. Spelled `fillingNull(_:)` in Swift because the internal host-side `MetalArray.fillNull` used by string gather still exists (`MetalStringArray.swift`); `am_fill_null` in C, `fill_null()` in Python. |
| `fill_null_forward` / `fill_null_backward` | **Not planned** | No roadmap item; both are scan-shaped. |
| `if_else` | **GPU** | `Kernels/Structural.swift`. Array/array, array/scalar, scalar/array and scalar/scalar branches, all ten primitive types plus `bool` (booleans go through the existing unpack/repack). A null condition yields a null output; otherwise the chosen branch's value and null-ness are copied through. `MetalArray.ifElse(_:_:_:)` and `cond.ifElse(_:_:)` in Swift, `am_if_else` in C, `if_else()` in Python. |
| `case_when` | **Not planned** | No roadmap item; `if_else` covers the two-branch case. |
| `coalesce` | **GPU** | `Kernels/Structural.swift`: a left fold of a two-input kernel, stopping early once the accumulator has no validity bitmap left. Any number of same-typed, same-length inputs. `MetalArray.coalesce(_:)` in Swift, `am_coalesce` in C, module-level `coalesce()` in Python. |
| `choose` | **Not planned** | No roadmap item. |
| `replace_with_mask` | **Not planned** | No roadmap item. |
| `is_null` / `is_valid` | **GPU** | `Kernels/Structural.swift`, bitmap word kernels over the validity bitmap: `is_null` is a word-wise NOT of it, `is_valid` shares it zero-copy, and an array with no bitmap gets a constant word fill. Primitive and boolean arrays; the result never has nulls itself. `isNull()` / `isValid()` in Swift, `am_is_null` / `am_is_valid` in C, `is_null()` / `is_valid()` in Python. |
| `is_nan` / `is_finite` / `is_inf` | **Not planned** | No roadmap item. |
| `make_struct` | **Partial** | `MetalRecordBatch(names:columns:)` composes equal-length columns and exports as a `+s` struct array, which is the record-batch form of this. It is not a compute function over struct-typed columns. |
| `struct_field` | **Partial** | `batch[name]` and `selecting(_:)` project columns out of a record batch; there is no struct-typed array to extract a field from. |
| `list_element` / `list_flatten` / `list_parent_indices` / `list_slice` / `list_value_length` | **Not planned** | Out of scope for 0.1.0: ArrowMetal has no list type, and ragged nested compute is a different kernel design from flat columnar. |
| `map_lookup` | **Not planned** | Out of scope: no map type. |

## Associative transforms

| Arrow function | Status | Notes |
|---|---|---|
| `dictionary_encode` (utf8) | **GPU** | `Kernels/StringDictionary.swift`. Hash each string, argsort the hashes, mark run boundaries by comparing the full **bytes** of adjacent sorted strings, rank the marks with the same GPU scan `unique()` uses, and gather the dictionary with the string gather. Codes are relabelled into first-seen order, so the result is identical to the host version this replaced — same codes, same dictionary order. Reachable from Swift, the C ABI (`am_str_dictionary_encode`) and Python. 10M strings, 200k distinct: 43 ms against 1.3 s for the host path. |
| `dictionary_encode` (utf8) collision handling | **GPU** | Rows are grouped by a 64-bit key (two independent murmur3 seeds) and the boundary kernel counts content runs against key runs; equal totals prove every bucket holds one distinct string. A bucket that does not is re-hashed under new seeds, and `MetalStringArray.dictionaryEncodeCPU()` remains as the final fallback, so the result is correct rather than probably correct. |
| `dictionary_encode` (numeric) | **In progress** | A concurrent branch is adding numeric dictionary encoding this week. Not in 0.1.0 as published here. |
| `unique` | **In progress** | Same branch: `unique` over numeric columns. |
| `value_counts` | **In progress** | Same branch: `value_counts` over numeric columns. |

## Pairwise and cumulative

| Arrow function | Status | Notes |
|---|---|---|
| `cumulative_sum` / `cumulative_sum_checked` | **Partial** | `cumulativeSum()` is GPU (`Kernels/Cumulative.swift`): a two-level inclusive scan — block scan, exclusive scan of the block totals, add back. Nulls are skipped in Arrow's sense: the output is null exactly where the input is and the running value carries across unchanged. Integers wrap and are exact; Float32 and Float64 reassociate the additions, so the last ulp can differ from a strictly sequential sum (Float64 accumulates through the software binary64 adder). `cumulative_sum_checked` is not implemented. The scan needs an exact count, so a pending batched input is materialised first. |
| `cumulative_prod` | **Not planned** | No roadmap item. The scan in `Cumulative.swift` takes any associative combine, so this is one more entry in its op table. |
| `cumulative_max` / `cumulative_min` | **GPU** | Same two-level scan, exact on every type including Float64 (bit-pattern ordering, NaN skipped). |
| `cumulative_mean` | **Not planned** | No roadmap item. |
| `pairwise_diff` / `pairwise_diff_checked` | **Not planned** | No roadmap item. |

## Hashing

Arrow C++ exposes no public element-wise hash compute function; the `hash_*` names in its catalogue are
grouped aggregates, covered above. The row below is an ArrowMetal extension.

| Function | Status | Notes |
|---|---|---|
| `hash32` over `utf8` (MurmurHash3 x86_32, seed 0) | **GPU** | `Kernels/StringSource.swift`. Nulls hash to 0 and stay null. Reachable as `am_str_unary(kind: 2)` and `.hash32()` in Python. A seeded variant is internal to `Kernels/StringDictionary.swift`, which needs two independent hashes. |
| Hash of primitive values | **Not planned** | No roadmap item; the in-progress hash join will need one and may bring it. |

## Type matrix

"Compute" means the kernels in this document run on the type. "Interop" means the C Data Interface importer
(`Sources/ArrowMetal/CInterop.swift`) accepts it: today that is exactly the format strings
`c C s S i I l L f g b u U`, plus `+s` for record batches. Anything else raises
`ArrowMetalError.unsupportedType`, and arrays carrying a `dictionary` pointer or any children are rejected
outright.

| Arrow type | Status | Notes |
|---|---|---|
| `null` | **Not planned** | No roadmap item; a buffer-less type has nothing for a kernel to do. |
| `bool` | **GPU** | Packed bitmap values. `and`/`or`/`not`, `filter`, `take`, `slice` are GPU; `count`/`any`/`all` are host popcounts. |
| `int8` / `int16` / `int32` / `int64` | **GPU** | Full kernel set. |
| `uint8` / `uint16` / `uint32` / `uint64` | **GPU** | Full kernel set. |
| `float16` (halffloat) | **Not planned** | No roadmap item, though MSL has `half` so it would be cheap. |
| `float32` | **GPU** | Full kernel set. NaN skipped by min/max, propagated by sum. |
| `float64` | **Partial** | Metal has no `double`. Compare/min/max/filter/take/slice/sort run on the GPU over order-preserving bit patterns; sum and add/sub/mul/div run a software IEEE-754 binary64 implementation on the GPU that is correctly rounded and bit-exact against Swift's `Double`. `negate`/`abs`/`sign`/`floor`/`ceil`/`round`/`trunc`, element-wise min/max and the cumulative functions are exact too (bit-pattern kernels); `sqrt`/`exp`/`ln`/`log10`/`log2` drop to `float` precision and widen back (~7 significant digits), and `power`/`modulo` are not implemented for it at all. Only `cast` falls back to the host, and group-by min/max/sum do not accept it. |
| `decimal32` / `decimal64` | **Not planned** | Out of scope for 0.1.0: 128/256-bit fixed-point arithmetic in MSL is a project of its own with no bandwidth win. |
| `decimal128` / `decimal256` | **Not planned** | Same. ArrowMetal explicitly does not claim decimals. |
| `date32` / `date64` | **In progress** | Concurrent branch this week; not in 0.1.0 as published here. |
| `time32` / `time64` | **In progress** | Same branch. |
| `timestamp` | **In progress** | Same branch. |
| `duration` | **In progress** | Same branch. |
| `interval` (month, day_time, month_day_nano) | **Not planned** | No roadmap item. |
| `binary` / `large_binary` | **In progress** | Concurrent branch this week. The `utf8` layout kernels apply unchanged (byte length, equality, prefix/suffix, hash, filter, take); only the importer and the char-length kernel are utf8-specific. |
| `fixed_size_binary` | **Not planned** | No roadmap item. |
| `utf8` | **GPU** | Byte/char length, equals/starts_with/ends_with/contains, count_substring/find_substring, murmur3 hash, `dictionary_encode` (GPU), filter, take, C Data import/export, and the transforms that build new string arrays: ASCII and Latin case mapping, trim/ltrim/rtrim, pad, slice, repeat, replace, reverse, element-wise join and the `ascii_is_*` predicates. `dictionary_encode` is CPU. |
| `large_utf8` | **Partial** | Import only, and only when the data is under 2 GB: 64-bit offsets are narrowed to int32 in one pass. Exports come back out as `utf8`. |
| `utf8_view` / `binary_view` | **Planned** | [ROADMAP → Medium term → Strings](../ROADMAP.md#medium-term) lists `utf8_view` as open. |
| `list` / `large_list` / `fixed_size_list` / `list_view` / `large_list_view` | **Not planned** | Out of scope for 0.1.0: ragged nested data needs a different kernel design, and the flat analytics case is not finished yet. |
| `struct` | **Partial** | Supported only as the record-batch container: `+s` import and export with one child per column, non-nested children, no top-level nulls and no offset. [ROADMAP → Medium term → RecordBatch](../ROADMAP.md#medium-term) lists "nested struct children" as open. There is no compute over struct-typed columns. |
| `map` | **Not planned** | Out of scope: nested. |
| `union` (dense and sparse) | **Not planned** | Out of scope: a type-id-dispatched layout defeats the uniform-thread model kernels rely on. |
| `dictionary` | **In progress** | Concurrent branch this week adds the type. [ROADMAP → Medium term](../ROADMAP.md#medium-term) covers the compute half: "compare and filter on codes without decoding". Today the importer rejects any array with a `dictionary` pointer. |
| `run_end_encoded` | **Not planned** | No roadmap item. |
| Extension types | **Not planned** | No roadmap item; the importer reads `format` only and would need `ARROW:extension:name` metadata handling. |

## Interop (not compute — separate vocabulary)

| Capability | Status | Notes |
|---|---|---|
| C Data Interface import | **Shipped** | Zero-copy when the producer's buffers are page aligned and the offset is 0, otherwise one copy; the result reports which (`ImportResult.zeroCopy`). Moves the array per spec. |
| C Data Interface export | **Shipped** | Primitive, boolean and `utf8` arrays. |
| C Device Data Interface import / export | **Shipped** | `ARROW_DEVICE_METAL`; a `sync_event` on import is waited on with an empty command buffer. |
| C Stream Interface import | **Shipped** | `importArrowArrayStream` drains a stream into record batches. |
| C Stream Interface export | **Planned** | [ROADMAP → Medium term → RecordBatch](../ROADMAP.md#medium-term): "Open: C Stream export, C Device Stream". |
| Record batch as `+s` struct array | **Partial** | Import and export both work; import rejects struct-level nulls, a non-zero offset, and nested children. |
| `MTLBuffer` recovery from our own exports | **Shipped** | `metalBuffers(of:)` for device arrays this process produced. |
| Arrow IPC (file and stream) read / write | **In progress** | A concurrent branch is adding IPC this week. Nothing in 0.1.0 as published here reads or writes IPC; callers go through the C Data Interface. |
| Python: PyCapsule `__arrow_c_array__` | **Shipped** | `python/arrowmetal/__init__.py` over the C ABI. |
| Python: `__arrow_c_device_array__`, wheel with the dylib inside | **Planned** | [ROADMAP → Integrations](../ROADMAP.md#integrations). |
| arrow-swift and MLX bridges, DuckDB/DataFusion UDF | **Planned** | [ROADMAP → Integrations](../ROADMAP.md#integrations). |

## What ArrowMetal 0.1.0 claims, and what it does not

**The claim.** ArrowMetal 0.1.0 claims complete, GPU-resident, null-correct coverage of exactly one thing:
**flat analytics on primitive, boolean and string columns** — that is, over `int8/16/32/64`,
`uint8/16/32/64`, `float32`, `float64`, `bool` and `utf8`: the reductions `sum`, `min`, `max`, `mean`; the
six comparisons against a scalar or another column; wrapping `add`/`subtract`/`multiply`/`divide`; boolean
`and`/`or`/`not`; `filter` (including a fused predicate form), `take` and `slice`; numeric `cast`;
single-key `sort`, `argsort` and top-k; group-by `count`/`sum`/`mean`/`min`/`max` over dense integer keys,
over every primitive value type;
`is_null`, `is_valid`, `fill_null`, `drop_null`, `if_else`, `coalesce`, `is_in`, `index_in` and the Kleene
`and_kleene`/`or_kleene`; `utf8` byte and character length, `equals`/`starts_with`/`ends_with`/`contains`,
`count_substring`, `find_substring`, murmur3 hash and `dictionary_encode`, plus the string transforms that build new `utf8`
arrays — ASCII and Latin case mapping, trim, pad, slice, repeat, replace, reverse, element-wise join and the
`ascii_is_*` predicates; and Arrow
C Data, C Device and C Stream interop for all of it — every one of them null-aware with Arrow semantics and
checked element-for-element against a CPU oracle in the test suite. Temporal columns join that sentence when
the in-progress temporal types land, because they are fixed-width integers underneath and the same kernels
apply unchanged; they are not part of the claim as published here.

**What it does not claim.** ArrowMetal does not do decimals (`decimal32/64/128/256`); it does not do compute
over nested types — lists, structs, maps, unions (struct appears only as the record-batch container, and
there is no compute over struct-typed columns); it does not do regex, full Unicode case folding beyond the
Latin-1 Supplement and Latin Extended-A blocks (the multi-character expansions of ß, ŉ and µ are left
alone), normalisation, Unicode-whitespace trimming, splitting or any other Unicode-table-driven string
transform; it does not do window or pairwise functions, and of the cumulative family only `cumulative_sum`,
`cumulative_min` and `cumulative_max` ship; it does not do temporal component extraction, temporal arithmetic,
timezones or `strftime`/`strptime`; it does not do statistical aggregates (`stddev`, `variance`, `quantile`,
`mode`, `tdigest`, `approximate_median`); it does not do set lookup over strings, nor the structural
functions it has no kernel for (`case_when`, `choose`, `replace_with_mask`, `fill_null_forward`/`_backward`,
`is_nan`/`is_finite`/`is_inf`); it does not do checked arithmetic or overflow-erroring casts; and it is not a query planner, a SQL engine or a
tensor library. Several of those are near-term roadmap items rather than refusals — the rows above say which
is which, one function at a time.

---

Version 0.1.0. Read alongside [ROADMAP.md](../ROADMAP.md), [DESIGN.md](DESIGN.md) and
[BENCHMARKS.md](BENCHMARKS.md).
