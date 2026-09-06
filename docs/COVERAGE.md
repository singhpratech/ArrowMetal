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
| String transforms | 9 | 2 | 2 | 0 | 0 | 7 | 20 |
| String containment and matching | 7 | 2 | 0 | 0 | 0 | 1 | 10 |
| Temporal | 2 | 1 | 1 | 0 | 1 | 1 | 6 |
| Conversions and casts | 0 | 2 | 3 | 0 | 1 | 0 | 6 |
| Selections | 4 | 0 | 1 | 0 | 0 | 0 | 5 |
| Containment / set lookup | 2 | 0 | 0 | 0 | 0 | 1 | 3 |
| Sorts and partitions | 5 | 1 | 2 | 0 | 0 | 0 | 8 |
| Structural and conditional | 6 | 1 | 0 | 0 | 0 | 7 | 14 |
| Associative transforms | 2 | 0 | 0 | 0 | 3 | 0 | 5 |
| Pairwise and cumulative | 4 | 0 | 2 | 0 | 0 | 0 | 6 |
| Hashing | 1 | 0 | 0 | 0 | 0 | 1 | 2 |
| **Total (compute functions)** | **68** | **13** | **28** | **2** | **6** | **45** | **162** |
| Arrow types (matrix below) | 7 | 0 | 6 | 1 | 6 | 8 | 28 |

Interop uses a separate vocabulary and is counted apart: 6 shipped, 1 partial, 3 planned, 1 in progress
(11 rows).

**The scope ArrowMetal 0.1.0 claims 100% of:** flat analytics on primitive, boolean and string columns —
`sum`/`min`/`max`/`mean`, the six comparisons, wrapping `add`/`subtract`/`multiply`/`divide`, boolean
`and`/`or`/`not`, `filter`/`take`/`slice`, numeric `cast`, single- and multi-key `sort`/`argsort` and a partial-selection top-k, group-by
`count`/`sum`/`mean`/`min`/`max` over dense integer keys for every primitive value type, `is_null`/`is_valid`/`fill_null`/`drop_null`/
`if_else`/`coalesce`/`is_in`/`index_in`/`and_kleene`/`or_kleene`, `utf8` length/`equals`/`starts_with`/`ends_with`/
`contains`/`count_substring`/`find_substring`/murmur3 hash/`dictionary_encode`, the ASCII case, trim, pad, slice, repeat,
replace, reverse, join and `ascii_is_*` transforms, and Arrow C Data, C Device and C Stream interop for all
of them — over `int8/16/32/64`, `uint8/16/32/64`, `float32`, `float64`, `bool` and `utf8`, null-aware with
Arrow semantics and checked against a CPU oracle in the test suite.

**The scope it does not claim:** decimals beyond decimal128 arithmetic and decimal256 selection; compute over
nested types beyond list/struct access, selection and child navigation;
Unicode-table string work (full case folding beyond Latin-1 Supplement and Latin Extended-A,
normalisation, Unicode-whitespace trimming and splitting — the ASCII splits and the regex functions do
ship, on the CPU); the statistical ranking transforms (`rank_quantile`, `rank_normal`) and the checked
(overflow-raising) forms of the cumulative and pairwise functions — the ranking, shift, pairwise-difference,
cumulative and rolling-window families themselves do ship, see Sorts and partitions and Pairwise and
cumulative; timezones; statistical aggregates (`stddev`, `variance`, `quantile`, `mode`, `tdigest`);
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
| `ascii_is_printable` / `ascii_is_title` | **GPU** | `Kernels/StringExtra.swift`: `asciiIsPrintable()` (every byte in 0x20–0x7E) and `asciiIsTitle()` (byte-wise title case over runs of ASCII letters, at least one letter). One bitmap word per thread, like the row above. `ascii_is_printable` is the one predicate here that is **true on the empty string**, matching Arrow. |
| `utf8_is_alnum` / `_alpha` / `_decimal` / `_digit` / `_lower` / `_numeric` / `_printable` / `_space` / `_title` / `_upper` | **GPU / CPU** | `Kernels/StringExtra.swift`, split **per row**: the `sx_pred` kernel answers every row byte-wise and reports, in a second bitmap, which rows carry a byte ≥ 0x80. A string of bytes < 0x80 is classified identically by the byte rules and by the Unicode tables, so only the marked rows are re-decided on the host with `UnicodeClass` (Swift's `Unicode.Scalar.Properties`), sharded over 4096-row chunks — an all-ASCII column never leaves the device. Categories: alpha `L*`, decimal `Nd`, digit `Nd`∪`No`, numeric `Nd`∪`Nl`∪`No`, alnum letters ∪ numbers, space `Zs`/`Zl`/`Zp` plus U+0009–U+000D, **U+001C–U+001F** and U+0085 (U+200B is *not* whitespace), printable everything but `Cc`/`Cf`/`Cs`/`Co`/`Cn`/`Zs`/`Zl`/`Zp` with U+0020 added back. `utf8_is_printable` is true on the empty string; the rest are false. Arrow's cased rule is reconstructed from utf8proc: upper = the simple lower-case mapping changes it or category `Lt`; lower = the simple upper-case mapping changes it or category `Ll`, minus the Roman numerals U+2160–U+216F — **not** Unicode's `Uppercase`/`Lowercase` derived properties, which would call modifier letters such as U+02B0 (ʰ) lower case. A titlecase letter is therefore both, which is why `utf8_is_upper("ǅ")` and `utf8_is_lower("ǅ")` are both false. Checked row-for-row against `pyarrow.compute` on a mixed 5 000-row column. |
| `string_is_ascii` | **GPU** | `Kernels/StringExtra.swift`: `stringIsAscii()`, every byte < 0x80, true on the empty string. Shares the `sx_pred` kernel with the rows above. |

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
| `utf8_capitalize` / `ascii_title` / `utf8_title` | **GPU / CPU** | `Kernels/StringExtra.swift`. `asciiTitle()` is always **GPU** (byte-wise: the first ASCII letter of every run of ASCII letters is upper-cased and the rest lower-cased, so `"ünïcödé"` → `"üNïCöDé"`, exactly as Arrow does). `utf8Capitalize()` (first code point up, the rest down) and `utf8Title()` (the first **cased** code point of every maximal run of cased code points up, the rest down) take the GPU byte kernel when the whole column is ASCII — an output length that depends on a Unicode table cannot be computed in the GPU length pass — and the CPU, sharded over 4096-row chunks, otherwise. Case *mapping* is Unicode's **simple** 1:1 mapping, as utf8proc's is, reconstructed from Swift's full mappings: a full mapping of exactly one scalar is the simple mapping, a longer one leaves the code point alone (U+0149 ŉ, U+01F0 ǰ, U+1E96 ẖ, U+0587 և …), and U+00DF (ß → ẞ), U+0130 (İ → i) and the Greek iota-subscript blocks U+1F80–U+1F87 / U+1F90–U+1F97 / U+1FA0–U+1FA7 (which map +8) are the explicit exceptions where the two would otherwise disagree. Checked against `pyarrow.compute` on a mixed 5 000-row column. |
| `replace_substring_regex` / `extract_regex` | **CPU** | `Kernels/Regex.swift`. A backtracking engine is a poor fit for SIMT, so matching runs on the host through `NSRegularExpression` (ICU), sharded over `DispatchQueue.concurrentPerform` chunks of 4096 rows. `replaceSubstringRegex(_:with:maxReplacements:)` falls through to the **GPU** `replaceSubstring` kernel when the pattern has no metacharacter and the template has no `$`. Two documented differences from pyarrow, which uses RE2: the replacement template is ICU's (`$1`, not `\1`), and `extractRegex(_:)` returns a `[String: MetalStringArray]` of the `(?<name>…)` groups rather than a struct array, because ArrowMetal has no struct-typed column. A row that does not match is null in every group. |
| `extract_regex_span` | **CPU** | `Kernels/StringExtra.swift`: `extractRegexSpan(_:ignoreCase:)` returns one `(start, length)` **pair of int32 arrays per named capture group**, the same `[String: …]` shape `extractRegex` uses, because ArrowMetal has no struct column. Offsets and lengths count **bytes**, as Arrow's do. A row that does not match, a null row and a group that took part in no alternative are null in both arrays. Same ICU engine and 4096-row sharding as `extractRegex`, so the same RE2-vs-ICU syntax differences apply — notably ICU's `\d` matches every Unicode decimal digit where RE2's is ASCII only. |
| `ascii_reverse` / `binary_reverse` / `utf8_reverse` | **GPU** | `reverse()` reverses **code points**, not grapheme clusters: a combining mark or a ZWJ emoji sequence comes back in reverse code point order. That is `utf8_reverse`; `binary_reverse` (byte order) is not exposed separately. |
| `replace_substring` | **GPU** | `replaceSubstring(_:with:maxReplacements:)`, non-overlapping and left to right, `maxReplacements` < 0 meaning all. Byte-wise, so a multi-byte pattern works. An empty pattern is the identity, matching Foundation's `replacingOccurrences(of: "", with:)` rather than Python's insert-everywhere. |
| `binary_replace_slice` / `utf8_replace_slice` | **GPU** | `Kernels/StringExtra.swift`: `replaceSlice(start:stop:with:)` indexes **code points** and `replaceSliceBytes(start:stop:with:)` **bytes** (returning `binary`, since a byte cut can split a UTF-8 sequence). Negative indices count from the end, both ends clamp into range, and a `stop` below `start` inserts without deleting — all three checked against pyarrow. Two-pass like every other transform, so the length and the bytes cannot disagree. |
| `binary_slice` / `utf8_slice_codeunits` | **Partial** | `sliceCodeunits(start:stop:)`, GPU, **`step == 1` only** — Arrow's negative and non-unit steps are not implemented. Indices are code points, negative values count from the end, both ends clamp into range, and slices always land on UTF-8 boundaries. `binary_slice` (byte indices) is not exposed separately. |
| `ascii_trim*` / `ascii_ltrim*` / `ascii_rtrim*` (whitespace and character set) | **GPU** | `trim()`/`ltrim()`/`rtrim()` strip ASCII whitespace (space, `\t`, `\n`, `\v`, `\f`, `\r`); `trim(characters:)`/`ltrim(characters:)`/`rtrim(characters:)` strip any byte in an ASCII set, and reject a non-ASCII set rather than splitting a UTF-8 sequence. Bytes ≥ 0x80 are never trimmed. |
| `utf8_trim*` (Unicode whitespace / character set) | **GPU / CPU** | `Kernels/StringExtra.swift`: `utf8Trim(characters:)` / `utf8Ltrim` / `utf8Rtrim` and `utf8TrimWhitespace()` / `utf8LtrimWhitespace()` / `utf8RtrimWhitespace()`. A character set that is itself ASCII routes to the existing **GPU** trim kernel — byte-wise trimming can never split a UTF-8 sequence, since every continuation byte is ≥ 0x80 — and a set with a non-ASCII character runs on the CPU over code points. The whitespace family routes to the GPU with a ten-byte set (`\t`, `\n`, `\v`, `\f`, `\r`, U+001C–U+001F and the space) when the column is all ASCII and to the CPU otherwise; the Unicode set is `Zs`/`Zl`/`Zp` plus U+0009–U+000D, U+001C–U+001F and U+0085, so U+00A0 and U+2003 are trimmed and U+200B is not. An empty character set is the identity, as in Arrow. |
| `ascii_lpad` / `ascii_rpad`, `utf8_lpad` / `utf8_rpad` | **GPU** | `padLeft(width:pad:)` / `padRight(width:pad:)`. `width` counts **code points** (the `utf8_*` behaviour) and `pad` must be exactly one character; strings already at or over `width` are returned unchanged. |
| `ascii_center` / `utf8_center` | **GPU** | `Kernels/StringExtra.swift`: `center(width:pad:)`, `width` counting **code points** and `pad` being exactly one character, with the odd pad character on the **right** (`"a"` centred in 4 is `"*a**"`), which is what Arrow does. Strings already at or over `width` come back unchanged. `ascii_center`, which counts bytes, is not exposed separately — the same choice the `ascii_lpad` / `ascii_rpad` row above makes. |
| `binary_repeat` | **GPU** | `repeat(_ n:)`, `n == 0` giving empty strings and `n < 0` raising. |
| `binary_join_element_wise` | **GPU** | `concat(_:separator:)` over two equal-length arrays, one scalar separator. Validities are ANDed on the GPU, so a null on either side gives a null output — Arrow's default `EMIT_NULL` null handling; the `REPLACE`/`SKIP` options are not implemented. |
| `binary_join` (list of strings) | **GPU** | `Kernels/StringContainment.swift`: `MetalListArray.binaryJoin(separator:)` over a `list<utf8>` (`MetalListArray` lives in `Sources/ArrowMetal/Nested.swift`), with a scalar separator or a per-row `utf8` column. Two passes: one kernel sums the child byte lengths plus `count - 1` separators into a per-row output length, the host scans those into the offsets buffer, and a second kernel copies the bytes. An empty row joins to the empty string; a null row, **any** null element inside a row, and a null separator all give a null output row — Arrow's `EMIT_NULL`; the `REPLACE` / `SKIP` options are not implemented. `am_binary_join` in C, `binary_join()` in Python. |
| `split_pattern` / `split_pattern_regex` / `ascii_split_whitespace` | **CPU** | `Kernels/Regex.swift`: `splitPattern(_:maxSplits:reverse:)`, `splitPatternRegex(_:maxSplits:)` and `splitWhitespace(maxSplits:reverse:)`. ArrowMetal still has no list type, so the result is the `(offsets, values)` **pair** of an Arrow `list<utf8>` — row `i` owns `values[offsets[i] ..< offsets[i+1]]` — and a null input row owns no pieces. `splitWhitespace` splits on runs of ASCII whitespace and drops the empty pieces at both ends, which is Python's `str.split()`; `maxSplits` keeps the remaining whitespace inside the last piece, as Python does. `utf8_split_whitespace` (Unicode whitespace) is not implemented. |
| `utf8_normalize` | **CPU** | `Kernels/StringExtra.swift`: `utf8Normalize(_:)` for NFC, NFKC, NFD and NFKD through Foundation, sharded over 4096-row chunks — full Unicode normalisation tables in MSL buy nothing over the host. **Difference from pyarrow:** `pyarrow.compute.utf8_normalize` (checked against 25.0.1) never composes, so its `NFC` output equals its `NFD` and its `NFKC` equals its `NFKD` (`"é"` comes back as `U+0065 U+0301`); this follows the Unicode standard and agrees with Python's `unicodedata.normalize` on all four forms. |

## String containment and matching

| Arrow function | Status | Notes |
|---|---|---|
| `equal` (string vs. string scalar) | **GPU** | `equals(_ s: String)` → boolean bitmap. |
| `equal` (string vs. string array) | **GPU** | `equals(_ other: MetalStringArray)`; validities are AND-ed on the GPU. |
| `match_substring` | **GPU** | `contains(_:)`, byte-wise, case-sensitive, no `ignore_case` option. |
| `starts_with` | **GPU** | `startsWith(_:)`. |
| `ends_with` | **GPU** | `endsWith(_:)`. |
| `match_substring_regex` / `match_like` | **CPU** | `Kernels/Regex.swift`: `matchSubstringRegex(_:ignoreCase:)` and `matchLike(_:)`, matched on the host with `NSRegularExpression` over concurrent chunks. Both have a **GPU** fast path: a pattern with no metacharacter (none of `\ . [ ] { } ( ) * + ? ^ $ \|`) routes to the `contains` kernel and `^literal` to `startsWith`, since ICU's `^` is exactly "start of input". A trailing `$` deliberately does not — ICU also matches it before a final line terminator, so `abc$` matches `"abc\n"` while `endsWith("abc")` does not. `match_like` translates `%`/`_` to a `\A…\z`-anchored regex, with `\` escaping a wildcard; a pure prefix, suffix, contains or equality pattern routes to `startsWith`/`endsWith`/`contains`/`equals`, which is exact because SQL `LIKE` anchors to the whole value. |
| `count_substring_regex` / `find_substring_regex` | **CPU** | Same file and the same sharding. A literal pattern routes to the existing **GPU** `countSubstring` / `findSubstring` kernels. `findSubstringRegex` reports the **byte** offset of the first match, or -1, matching `find_substring`. |
| `count_substring` | **GPU** | `countSubstring(_:)` → Int32, non-overlapping occurrences, byte-wise and case-sensitive (no `ignore_case`). An empty pattern counts the code point boundaries, `charLength() + 1`, matching Arrow. Nulls propagate. |
| `find_substring` | **GPU** | `findSubstring(_:)` → Int32, the **byte** offset of the first occurrence or -1 when absent; an empty pattern finds 0. Byte-wise and case-sensitive. Nulls propagate. |
| `index_in` / `is_in` (strings) | **GPU** | `Kernels/StringContainment.swift`. The value set is hashed with the same 64-bit key `dictionaryEncode` builds (two independently seeded MurmurHash3 x86_32 passes side by side) and inserted into an open-addressing table of **row indices** with linear probing; every probe confirms its candidate by comparing the full **bytes**, so a hash collision costs one extra probe and can never give a wrong answer, and duplicates in the set collapse onto the lowest row index — exactly the first occurrence `index_in` reports. Nulls in the value set are ignored and a null value is never in the set, so `is_in` never returns a null and `index_in` is null exactly where the value is null or absent (Arrow's `null_matching_behavior = "skip"`). pyarrow's default is the opposite (`skip_nulls=False`, where a null value matches a null in the set); that option is **not implemented**. `isIn(_:)` / `indexIn(_:)` in Swift, `am_string_is_in` / `am_string_index_in` in C, `is_in()` / `index_in()` in Python. |

## Temporal

| Arrow function group | Status | Notes |
|---|---|---|
| Component extraction: `year`, `month`, `day`, `day_of_week`, `hour`, `minute`, `second`, `day_of_year`, `quarter`, `iso_week`, `iso_year`, `is_leap_year`, `millisecond`, `microsecond`, `nanosecond` | **GPU** | `Temporal.swift` decomposes the calendar with Howard Hinnant's `civil_from_days`; `Kernels/TemporalMath.swift` extends it in its own source file with `dayOfYear()`, `quarter()`, `isoWeek()`, `isoYear()`, `isLeapYear()` (a packed boolean bitmap, one 32-bit word per thread) and the three subsecond components. Arrow's nesting for those: `millisecond` counts from the last full second, `microsecond` from the last full millisecond, `nanosecond` from the last full microsecond. UTC only — a timestamp's timezone is metadata and is never applied. `subsecond`, `week`/`us_week`/`us_year`, `iso_calendar`, `year_month_day` and `is_dst` are not implemented. |
| Differences and arithmetic: `days_between`, `subtract` / `add` over temporal types, `hours_between`, `minutes_between`, `seconds_between`, `weeks_between`, `months_between`, `quarters_between`, `years_between`, `*_interval_between` | **Partial** | `Kernels/TemporalMath.swift`, all GPU: `daysBetween(_:)` floors both sides to their UTC day and returns the int64 day difference; `subtractTemporal(_:)` gives `timestamp − timestamp` (or `duration − duration`, `date32 − date32`, `date64 − date64`) as a `duration` in the finer of the two resolutions; `addDuration(_:)` adds a duration column (rescaled to the receiver's unit) or a scalar of the receiver's own ticks. The other `*_between` functions and calendar-aware `months_between` are not implemented, and `addDuration` is rejected on `date32`, whose tick is a whole day. |
| Rounding: `ceil_temporal`, `floor_temporal`, `round_temporal` | **GPU** | `Kernels/TemporalMath.swift`: `floorTemporal(to:multiple:)`, `ceilTemporal`, `roundTemporal` over `nanosecond` … `day` (integer arithmetic in the value's own resolution; rounding to a unit finer than the storage is the identity) and `month` / `quarter` / `year` (the civil algorithm and its inverse `days_from_civil`, so it needs a column that carries a date). `ceil` leaves a value already on a boundary alone, which is Arrow's `ceil_is_strictly_greater = false`; `round` sends an exact half **up**, toward +infinity. Checked against Foundation's `Calendar` in UTC over 100k random timestamps in `TextTests`. `calendar_based_origin` and week-based units are not implemented. |
| Timezones: `assume_timezone`, `local_timestamp` | **Not planned** | Out of scope for a GPU kernel library: the tz database is host data. |
| `strftime` / `strptime` | **CPU** | `Kernels/TemporalMath.swift`. Formatting and parsing against calendar data belong on the host, so both go through the C library with a UTC `tm` (`gmtime_r` + `strftime`, `strptime` + `timegm`) rather than a `DateFormatter` Unicode pattern — the format string is a **C strftime/strptime format**, which is what Arrow takes. `%f` is an ArrowMetal extension expanding to the six-digit fractional second. `strptime` must consume the whole value; a row that does not parse comes back null, or throws with `strict: true`. `strptime` is sharded over `DispatchQueue.concurrentPerform`. No locale and no timezone offsets: UTC only. |
| Temporal **types** (`date32`, `date64`, `time32`, `time64`, `timestamp`, `duration`) | **In progress** | A concurrent branch is adding temporal type import/export and routing them onto the existing fixed-width integer kernels this week. Not in 0.1.0 as published here: `arrowPrimitiveType(forFormat:)` accepts only `c C s S i I l L f g` today. |

## Conversions and casts

| Arrow function | Status | Notes |
|---|---|---|
| `cast` (numeric → numeric) | **Partial** | `Kernels/Cast.swift`, GPU, across all ten primitives. Unchecked only: integer narrowing wraps and float → int truncates toward zero, which is Arrow's `safe=false`. There is no `safe=true` overflow-erroring cast. |
| `cast` involving Float64 | **CPU** | `Dispatch.runsOnGPU` excludes `Double`, so any cast with Float64 on either side runs a host loop. Note that Float64 *arithmetic* and *reductions* do run on the GPU — the cast is the exception. |
| `cast` boolean ↔ integer | **Partial** | `MetalBooleanArray.toUInt8Array()` is a public GPU unpack (bitmap → uint8). The reverse packing exists but is internal. |
| `cast` string ↔ numeric / temporal | **Partial** | `Kernels/StringCast.swift`. **Integer → string** is GPU (`str_itoa_*`, the same two-pass length/bytes shape as the string transforms), exact for `Int64.min` and `UInt64.max`; **string → integer** is GPU (`str_parse_int`, one thread per 32 rows so the validity word needs no atomics), the whole value matching `[+-]?[0-9]+` with leading zeros allowed and everything else — empty, malformed, out of range, a `-` on an unsigned target — becoming null, which is Arrow's `safe=false`; `strict: true` throws instead. **Float and boolean** conversions are CPU: floats format as the shortest decimal string that round-trips, which differs from Arrow in keeping a `.0` on a whole value and using Swift's exponent form, and parse with Swift's `Double`/`Float` initialiser; booleans are `"true"`/`"false"` out and `"true"`/`"false"`/`"1"`/`"0"` case-insensitively in. **Temporal ↔ string** goes through the `strftime` / `strptime` row above. Reachable as `am_to_strings` / `am_parse` in C and `to_strings()` / `cast("string")` / `parse(type)` in Python. Decimal and the checked (`safe=true`) forms are not implemented. |
| `cast` to/from decimal | **CPU** | `Sources/ArrowMetal/Decimal.swift`, decimal128 only, one host pass each. `toFloat64()` divides the unscaled 128-bit value by 10^scale; `MetalDecimalArray.fromFloat64(_:type:)` multiplies by 10^scale and rounds halves away from zero, turning a non-finite or out-of-range value into null; `fromInt64(_:type:)` multiplies exactly (wrapping past 128 bits). Both directions carry Double's 53 bits of precision, so a cast through float64 is lossy above 2^53 — deliberately host code, since a kernel would buy nothing over the PCIe-free unified memory. `am_decimal_op` ops 16 and 17, `to_float64()` in Python. Decimal ↔ string and decimal ↔ decimal128-with-another-precision are not implemented (use `round`/`ceil`/`floor`/`truncate` to change scale). |
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
| `is_in` | **GPU** | `Kernels/Structural.swift`, all ten primitive types. The value set is reduced to its sorted distinct non-null values with the existing `unique()`, and each element binary-searches it on the GPU (no hash table). Nulls in the set are ignored and a null element never matches, so the result never has nulls — Arrow's `null_matching_behavior = "skip"`. Float equality is Arrow value equality, as in `unique()`: every NaN is one value and `-0.0` equals `0.0`. `isIn(_:)` in Swift (a `[T]` or a `MetalArray<T>`), `am_is_in` in C, `is_in()` in Python. `utf8` columns take a different route — a GPU hash table over the string bytes; see the `index_in` / `is_in` (strings) row under String containment — and `is_in()` / `index_in()` in Python dispatch on the column's type. |
| `index_in` | **GPU** | Same search, returning the int32 position in the caller's set array of each element's **first** occurrence there, and null where the element is null or absent. The unique-rank-to-first-row map is a group-by min over the set's dictionary codes, so it too runs on the GPU. `indexIn(_:)` in Swift, `am_index_in` in C, `index_in()` in Python. |
| `indices_nonzero` | **Not planned** | No roadmap item; the filter kernel already contains the scan-and-scatter it needs. |

## Sorts and partitions

| Arrow function | Status | Notes |
|---|---|---|
| `array_sort_indices` | **GPU** | `Kernels/Sort.swift`: LSD radix sort, 4 passes for 32-bit keys and 8 for 64-bit, stable. Ascending or descending. Total order for floats (NaN after +inf). |
| `sort_indices` (multiple sort keys) | **GPU** | `Kernels/MultiSort.swift`: successive stable radix argsorts from the least significant key upwards, the keys reordered with `take` between passes, so k keys cost k argsorts and no new kernel. Ascending or descending per key; nulls last in every key in both directions. `lexsortIndices(_:descending:)` and `MetalRecordBatch.sorted(by: [(column:descending:)])` in Swift, `am_lexsort` in C, `lexsort_indices()` in Python. utf8, binary and dictionary key columns throw — there is no order-preserving GPU key for them yet. |
| Sorted copy (`sorted()`) and `MetalRecordBatch.sorted(by:)` | **GPU** | Argsort then take, single key or several. Not an Arrow compute function name, but it is what callers use. |
| Nulls-last placement in the sorted index array | **CPU** | The radix sort runs on the GPU; the stable partition that moves null rows to the end is a host pass over the index array (`Sort.swift`). |
| `select_k_unstable` (top-k) | **GPU** | `Kernels/TopK.swift`: for k ≤ 1024 each threadgroup keeps the best k of its own block in threadgroup memory (threshold plus a bitonic compaction), and one radix sort over the `blocks * k` candidates orders the winners. Same total order as `argsort` — value key, ties by row — so the result is index-for-index what the full sort would give. Larger k, and the case where fewer than k rows are non-null, fall back to the argsort-and-slice. Single key. 50M Int64, k=100: 4.0 ms against 128 ms for the full sort. |
| `partition_nth_indices` | **Partial** | `partitionNthIndices(_:)` (`Kernels/MultiSort.swift`) returns a full `argsort`, which trivially satisfies the partition contract but costs a sort rather than the O(length) a selection algorithm would. Documented as such at the call site; the signature is the one a real partition would have. |
| `rank` / `rank_quantile` / `rank_normal` | **Partial** | `rank()` is GPU (`Kernels/Window.swift`): one argsort, run marks over the sorted order, a two-level scan of those marks, and a scatter back to the original rows. Arrow's `tiebreaker` options are separate calls here — `rank()` is `min`, `denseRank()` is `dense`, `rowNumber()` is `first`; `max` is not implemented, and neither is `rank_quantile` or `rank_normal`. |
| SQL window ranking: `row_number`, `dense_rank`, `percent_rank`, `cume_dist` | **GPU** | An ArrowMetal extension, not Arrow compute function names. Same argsort-plus-scan as `rank`, so all five cost one sort. Nulls follow `ORDER BY x NULLS LAST`: they sort after every value and form one tie group, so no ranking result is itself null. Float ties use Arrow value equality (every NaN is one value, ordered after +inf; `-0.0` equals `0.0`). `percentRank()` and `cumeDist()` come back as float64 through the correctly rounded software binary64 divide. `am_window` ops 0-4 in C, `row_number()` / `rank()` / `dense_rank()` / `percent_rank()` / `cume_dist()` in Python. |

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
| `make_struct` | **CPU** | `MetalStructArray(names:children:valid:)` composes equal-length columns into a struct-typed column, and `MetalRecordBatch(names:columns:)` does the record-batch form of the same thing (`Sources/ArrowMetal/Nested.swift`). Metadata only — the children are shared, nothing is copied and no kernel runs. |
| `struct_field` | **GPU** | `MetalStructArray.structField(_:)` (`Nested.swift`), `am_struct_field` in C, `struct_field()` in Python. The struct's own nulls are propagated into the field, matching Arrow: a field of a null row is null, which is one GPU gather when the struct has a validity bitmap and free when it has none. `batch[name]` / `selecting(_:)` still project columns out of a record batch. |
| `list_element` / `list_flatten` / `list_value_length` | **GPU** | `Kernels/NestedSource.swift`, over `list`, `large_list`, `fixed_size_list` and `map`, with a child of any supported type including another nested array. `listValueLength()` → int32, null in / null out; `listFlatten()` is the child restricted to `offsets[0] ..< offsets[length]`; `listElement(_:)` builds child indices on the GPU and gathers through the child's own `take`, giving **null** where the row is null or shorter than the index (Arrow raises instead). `am_list_value_length` / `am_list_flatten` / `am_list_element` in C, `list_value_length()` / `list_flatten()` / `list_element()` in Python. |
| `list_parent_indices` / `list_slice` | **Not planned** | No roadmap item. Both are one more kernel over the offsets buffer the three functions above already use. |
| `map_lookup` | **Not planned** | No roadmap item. The `map` type itself imports, exports and selects (see the type matrix); nothing looks a key up. |

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
| `cumulative_prod` | **GPU** | `cumulativeProd()` (`Kernels/Window.swift`) runs the scan from `CumulativeSource` with a multiply, so it is the same three passes as `cumulative_sum` and skips nulls the same way. Integer products wrap and are exact; Float32 and Float64 reassociate, and a Float32 product that drifts into the subnormals comes back as zero (Apple GPUs flush Float32 denormals — the Float64 path uses the software multiplier and keeps them). `cumulative_prod_checked` is not implemented. |
| `cumulative_max` / `cumulative_min` | **GPU** | Same two-level scan, exact on every type including Float64 (bit-pattern ordering, NaN skipped). |
| `cumulative_mean` | **GPU** | `cumulativeMean()` returns float64 for every input type: a binary64 running sum over an int32 running count of non-null rows, then the correctly rounded software divide. Output null exactly where the input is. Values widen to binary64 first, so int64 magnitudes above 2^53 round on the way in, and the sum reassociates as `cumulative_sum` does. |
| `pairwise_diff` / `pairwise_diff_checked` | **Partial** | `pairwiseDiff(period:)` is GPU, one thread per element: `out[i] = a[i] - a[i - period]`, null where either side is null or falls outside the array, and a negative period differences forwards. Integers wrap; Float32 subtracts in `float` and Float64 through the correctly rounded software binary64 subtract, so both are exact. `pairwise_diff_checked` is not implemented. |
| `shift` (lag / lead) and trailing rolling `sum` / `min` / `max` / `mean` | **GPU** | ArrowMetal extensions, not Arrow compute function names (`Kernels/Window.swift`). `shift(by:fill:)` moves rows forwards or backwards, filling with a scalar or a null. The rolling calls take `window` and `minPeriods` (how many non-null rows the trailing window needs before it produces a value; fewer gives a null). Min and max scan the window, one thread per output — O(n · window), the right shape up to a few thousand rows per window — with NaN skipped as the reductions skip it. Sum and mean are the difference of two prefix sums, so they are O(n); the price is that a float window sum loses cancellation digits, and one NaN or infinity in a float column poisons every later window. `rollingMean` returns float64. `am_window` ops 5 and 9-12 in C, `shift()` / `rolling_sum()` / `rolling_min()` / `rolling_max()` / `rolling_mean()` in Python. |

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
| `decimal32` / `decimal64` | **Not planned** | No roadmap item. The C Data Interface spells them `d:p,s,32` and `d:p,s,64`; the importer rejects those widths with a message naming the two that are supported. Both fit an existing integer column exactly, so there is no kernel work behind them — only type metadata nobody has asked for. |
| `decimal128` | **GPU** | `Sources/ArrowMetal/Decimal.swift` and `Kernels/DecimalSource.swift`. Elements are Arrow's raw 16-byte little-endian two's-complement unscaled values; every kernel works on two 64-bit limbs, with carries from unsigned compares and `mulhi`/`*` for the products. GPU: the six comparisons (scalar and array, signed 128-bit ordering), `sum`/`min`/`max` (per-thread 128-bit accumulate, threadgroup tree, host combine), `add`/`subtract` (array and scalar), `negate`/`abs`/`sign`, multiply by an int64 scalar and element-wise multiply (Arrow's rule: precision `p1+p2+1`, scale `s1+s2`, rejected when the precision does not fit 38), `round`/`ceil`/`floor`/`truncate` to a target scale (scaling up multiplies, scaling down is a restoring 128-bit long division with the rounding mode applied to the magnitude; halves go away from zero, as this package's float `round` does), and `filter`/`take`/`slice` (the int32 filter compacts an index vector, then a byte-width-generic gather moves the values). Arithmetic wraps modulo 2^128, matching Arrow's unchecked kernels; the two sides of a binary op must share a scale or the call raises rather than rescaling silently. Casts are CPU (see Conversions above). C Data Interface import (zero-copy, offsets and foreign producers handled) and export both work, `am_format` reports `d:p,s`, and `MetalRecordBatch` carries decimal columns through `nullCount`/`filter`/`take`/`slice`/`+s` export. `am_decimal_op` in C (op table in `include/arrowmetal.h`), `am_compare_scalar`/`am_compare_array` for the comparisons, and `decimal_add`/`decimal_sub`/`decimal_mul`/`decimal_round`/`to_float64` plus the `==`/`<`/… operators in Python. Not implemented: divide, `cast` between decimal precisions, group-by, sort and `is_in` over decimal columns, and Arrow IPC (the writer rejects a decimal column with a message). |
| `decimal256` | **Partial** | Same files, four limbs instead of two: C Data Interface import and export, the six comparisons (a 16-byte scalar is sign-extended to 256 bits), `filter`, `take`, `slice` and `sum` run on 32-byte elements. `min`/`max`, all arithmetic, `sign`, the rounding family and the casts throw `unsupportedType` naming decimal128 rather than computing something wrong. |
| `date32` / `date64` | **In progress** | Concurrent branch this week; not in 0.1.0 as published here. |
| `time32` / `time64` | **In progress** | Same branch. |
| `timestamp` | **In progress** | Same branch. |
| `duration` | **In progress** | Same branch. |
| `interval` (month, day_time, month_day_nano) | **Not planned** | No roadmap item. |
| `binary` / `large_binary` | **In progress** | Concurrent branch this week. The `utf8` layout kernels apply unchanged (byte length, equality, prefix/suffix, hash, filter, take); only the importer and the char-length kernel are utf8-specific. |
| `fixed_size_binary` | **Not planned** | No roadmap item. |
| `utf8` | **GPU** | Byte/char length, equals/starts_with/ends_with/contains, count_substring/find_substring, murmur3 hash, `dictionary_encode` (GPU), filter, take, C Data import/export, and the transforms that build new string arrays: ASCII and Latin case mapping, trim/ltrim/rtrim, pad, slice, repeat, replace, reverse, element-wise join and the `ascii_is_*` predicates, plus GPU integer↔string casts. The regex functions, SQL `LIKE`, splitting and the float/boolean casts are CPU. |
| `large_utf8` | **Partial** | Import only, and only when the data is under 2 GB: 64-bit offsets are narrowed to int32 in one pass. Exports come back out as `utf8`. |
| `utf8_view` / `binary_view` | **Planned** | [ROADMAP → Medium term → Strings](../ROADMAP.md#medium-term) lists `utf8_view` as open. |
| `list` / `large_list` / `fixed_size_list` | **Partial** | `MetalListArray` (`Sources/ArrowMetal/Nested.swift`): C Data import and export of `+l`, `+L` and `+w:N`, `list_value_length` / `list_flatten` / `list_element`, and `filter` / `take` / `slice`. The child is an `AnyMetalArray`, so it may be any supported type including another list, a struct or a map, recursively. Offsets are always int32 in Metal memory: `large_list` offsets are narrowed on import (and come back out as `+l`, as `large_utf8` comes back out as `utf8`), and a `fixed_size_list` materialises the `i * N` offsets its layout implies, so one set of kernels covers all three. `take` recomputes the offsets with the existing GPU scan and expands the selected rows' source ranges into one child index array that the child's own `take` gathers; a `slice` of a variable-length list shares both the offsets buffer and the child. Not implemented: aggregates or arithmetic over list values, `list_parent_indices`, `list_slice`. |
| `list_view` / `large_list_view` | **Not planned** | No roadmap item; the out-of-order sizes-and-offsets layout is a different importer from the three above. |
| `struct` | **GPU** | `MetalStructArray` (`Nested.swift`) is `+s` as a **column**, not only as the record-batch container: named children of any supported type, its own validity bitmap, arbitrary nesting in either direction (a struct of lists, a list of structs), `structField(_:)`, and `filter` / `take` / `slice` by delegating to the children and gathering the struct's own validity. `importArrowRecordBatch` keeps its top-level meaning and now accepts nested children. No aggregate takes a struct column. |
| `map` | **Partial** | `MetalMapArray` (`Nested.swift`): `+m` import and export — a list of non-nullable `struct<key, value>` entries, with `keys_sorted` carried through — plus `filter` / `take` / `slice` and the `list_*` functions on the entries. There is no `map_lookup` and no compute over keys or values. |
| `union` (dense and sparse) | **Partial** | `MetalUnionArray` (`Nested.swift`): `+ud:` and `+us:` import and export with the type ids, the dense offsets and one child per variant, plus `filter` / `take` / `slice` (dense selects type ids and offsets, sparse moves the children with the selection). No kernel reads a union's values: a type-id-dispatched layout defeats the uniform-thread model the compute kernels rely on. Unions carry no validity bitmap, per Arrow 1.0. |
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
single- and multi-key `sort`, `argsort` and top-k; group-by `count`/`sum`/`mean`/`min`/`max` over dense integer keys,
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
there is no compute over struct-typed columns); it does not do full Unicode case folding beyond the
Latin-1 Supplement and Latin Extended-A blocks (the multi-character expansions of ß, ŉ and µ are left
alone), normalisation, Unicode-whitespace trimming or Unicode splitting — the regex functions, SQL
`LIKE` and the ASCII splits do ship, on the CPU behind the same API, with a GPU fast path for patterns
that are really literals; of the window family it does not do `rank_quantile` or `rank_normal`, nor any of the
checked (overflow-raising) cumulative or pairwise forms — the ranking, shift, pairwise-difference, cumulative and
rolling-window calls themselves ship; it does not do
timezones; it does not do statistical aggregates (`stddev`, `variance`, `quantile`,
`mode`, `tdigest`, `approximate_median`); it does not do set lookup over strings, nor the structural
functions it has no kernel for (`case_when`, `choose`, `replace_with_mask`, `fill_null_forward`/`_backward`,
`is_nan`/`is_finite`/`is_inf`); it does not do checked arithmetic or overflow-erroring casts; and it is not a query planner, a SQL engine or a
tensor library. Several of those are near-term roadmap items rather than refusals — the rows above say which
is which, one function at a time.

---

Version 0.1.0. Read alongside [ROADMAP.md](../ROADMAP.md), [DESIGN.md](DESIGN.md) and
[BENCHMARKS.md](BENCHMARKS.md).
