# Differential evaluation against pyarrow.compute

ArrowMetal 0.1.0 checked kernel by kernel against Apache Arrow's own C++ compute functions, through
`pyarrow.compute`. The point is not to confirm the hand-written tests in `python/tests/test_arrowmetal.py`
but to find the cases nobody thought to write down: the harness generates the input, runs both engines on
it and compares, so a kernel that is wrong in an unanticipated way still fails.

    swift build -c release --product ArrowMetalC
    PYTHONPATH=python python python/tests/differential_report.py       # the matrix, as a table
    PYTHONPATH=python python -m pytest python/tests/test_differential.py -q

| | |
|---|---|
| Files | `python/tests/test_differential.py` (the harness), `python/tests/differential_report.py` (the runner) |
| Oracle | `pyarrow.compute` 25.0.1 |
| Cases in the default matrix | 13,176, about 13 s on an M4 Max; more with `DIFF_LARGE=1` |
| Result at 0.1.0 | 12,617 pass, 46 fail across 4 documented divergences, 513 skip (kernels not implemented), 0 unclassified |

## Method

**One oracle, never a literal.** Every expected value comes from a `pyarrow.compute` call on the same
array: `pc.sum`, `pc.min`/`pc.max`, `pc.equal`…, `pc.add`/`pc.subtract`/`pc.multiply`/`pc.divide`,
`Array.cast(safe=False)`, `pc.and_`/`pc.or_`/`pc.invert`, `Array.filter`, `Array.take`, `Array.slice`,
`pc.array_sort_indices`, `pc.binary_length`, `pc.utf8_length`, `pc.starts_with`, `pc.ends_with`,
`pc.match_substring`, and `pa.Table.group_by(...).aggregate(...)` for the grouped aggregates. The two
exceptions are stated in the code: `hash32` has no Arrow counterpart and is checked for null propagation
and injectivity instead, and integer `sum`/`mean` fall back to a wrapped-64-bit oracle where Arrow refuses
to produce an overflowed answer (see *Divergences* below).

**Generated input, not fixtures.** For every type ArrowMetal imports — `int8 int16 int32 int64`,
`uint8 uint16 uint32 uint64`, `float32`, `float64`, `bool`, `utf8` — the generator produces:

- **Sizes** 0, 1, 33, 1000, 100,003; add 5,000,000 with `DIFF_LARGE=1`, drop 100k with `DIFF_QUICK=1`.
  33 and 100,003 are deliberately not multiples of a threadgroup or SIMD width.
- **Null ratios** 0, 0.3 and 1.0. Nulls are applied with `pa.array(values, mask=...)`, which leaves the
  original numbers *under* the validity bitmap — what a real column looks like after a filter, and what a
  kernel is required to ignore. That choice is what surfaced the argsort null-order bug.
- **Special values** (the `special` flavor): each type's min and max, 0, 1, -1, half-range; for floats
  `-0.0`, NaN, ±inf, the smallest normal, the smallest subnormal and their negatives, `FLT_MAX`/`DBL_MAX`
  and machine epsilon; for strings the empty string, multi-byte UTF-8 (`héllo`, `日本語`, `Ωμέγα`, an
  emoji pair), embedded tabs and newlines, and 300- and 4096-byte strings. The two long strings drop
  out of the pool above 200,000 rows, where they would otherwise make a gigabyte-scale array per case.
- **Sliced arrays** (the `sliced` flavor): the array is built two offsets longer and then
  `pa.Array.slice(offset, size)` at offset 3 or 5 — neither byte- nor 8-bit-aligned, so the import path has
  to honour `ArrowArray.offset` for the values buffer, the validity bitmap and the utf8 offset buffer alike.

That is 27 datasets per (operation, type) by default.

**Every method that exists is in scope.** Operations are a registry in `test_differential.py`, and
`test_every_public_operation_has_a_differential_case` fails if a method is added to `MetalArray` or
`GroupBy` without one, so the harness cannot silently fall behind the library. Optional operations
register themselves the moment the method appears on `MetalArray`, so a new kernel joins the matrix
with no edit here; the ones still missing (`is_nan`, `reverse`, `cumulative_prod` at 0.1.0) are listed
by the report as absent. Two methods can never be reached this way and say so instead of passing
quietly: the temporal kernels (`year`…`second`, `cast_unit`), because the generator makes no timestamp
column — `Tests/ArrowMetalTests/TemporalTests.swift` covers those — and the generic dispatchers
`unary`, `binary` and `cumulative`, every op of which is reached through a named form that *is* in the
matrix. `test_methods_outside_the_matrix_are_reported` prints that list on every run.

**Comparison strictness.** Results are compared as `pyarrow.Array`s, not as Python lists, so the *type* is
part of the comparison: a kernel that returns the right numbers as `int64` instead of `uint64` fails, which
is how the unsigned group-by total turned up. Integers, booleans, strings and index vectors are compared exactly. Float
arithmetic (`+ - * /`, scalar and array, Float32 and Float64) is compared *bit-exact* — `Array.equals` is
deliberately not used for float types, because Arrow calls `-0.0` and `0.0` equal and `NaN` and `NaN`
unequal, and this harness needs the opposite of both. Reductions get a tolerance, because the two engines
legitimately accumulate in a different order: a relative 1e-6 for Float32 and 1e-12 for Float64, plus the
standard roundoff bound `8·eps·sqrt(n)·Σ|x|`, without which a 100k-row Float32 sum cannot be compared at
all (`sqrt(100000)·2^-24 ≈ 2e-5` on its own).

**Not-implemented versus wrong.** An `ArrowMetalError` whose message matches a known gap ("Unsupported
Arrow type", "group-by min/max needs a 32-bit or narrower type", …) is a *skip*; any other error, or a
wrong answer, is a *failure*. The 513 skips in the default run are the group-by combinations the kernels do
not cover — `min`/`max` on 64-bit values, `mean` on Float32, and any aggregate over Float64, the same set
`test_arrowmetal.py` pins as expected errors — plus the primitive-only kernels (`is_null`, `is_valid`,
`fill_null`, `drop_null`, `if_else`, `is_in`, `index_in`) on `utf8` and, for the two set-lookup kernels,
on `bool`.

## Deliberate and documented divergences

These are not bugs. Each has a named test in `test_differential.py` that asserts *both* behaviours, so the
suite notices if either engine changes its mind.

| Case | ArrowMetal | pyarrow | Test |
|---|---|---|---|
| Integer division by zero | 0 | raises `ArrowInvalid` | `test_integer_division_by_zero_returns_zero_where_pyarrow_raises` |
| Integer overflow | wraps | `pc.add` wraps, `pc.add_checked` raises — the harness uses the unchecked variants | `test_integer_overflow_wraps_where_checked_pyarrow_raises` |
| `INT_MIN / -1` (undefined in C) | `INT_MIN` (the hardware answer) | 0 | `test_int_min_divided_by_minus_one_wraps_where_pyarrow_yields_zero` |
| Out-of-range float → int cast (undefined in C) | wraps | saturates to `INT32_MAX` | `test_out_of_range_float_to_int_cast_diverges` |
| Float division by zero | ±inf / NaN | ±inf / NaN — identical | `test_float_division_by_zero_is_ieee_754_in_both` |
| `&` / `\|` with nulls | null-propagating, like `pc.and_`/`pc.or_` — *not* `and_kleene`/`or_kleene` | both available | `test_boolean_and_or_are_not_kleene` |
| NaN in `min`/`max` | skipped | skipped — identical | `test_nan_is_skipped_by_min_and_max_in_both` |
| NaN in `sum` | propagates | propagates — identical | `test_nan_propagates_through_sum_in_both` |
| Sort order of ties | stable | stable — identical | `test_ties_sort_stably_in_both` |
| NaN's position in a sort | after `+inf` | after `+inf` — identical | `test_sort_places_nan_after_positive_infinity_in_both` |
| Group-by keys outside `[0, key_count)` | dropped | given their own group | `test_group_by_ignores_keys_outside_the_declared_range` |
| Integer `mean` when the total overflows | divides the wrapped 64-bit sum, matching its own `sum()` | accumulates in double | `test_integer_mean_wraps_where_pyarrow_widens` |
| `min`/`max` of an all-NaN column | null | NaN | `test_all_nan_min_max_is_null_in_arrowmetal_and_nan_in_pyarrow` |
| `sign` of an integer column | keeps the column's type | narrows to `int8` | `test_sign_keeps_the_column_type_where_pyarrow_narrows_to_int8` |
| `floor`/`ceil`/`trunc` of an integer column | the identity, keeps the type at any magnitude | widens to double, and refuses the column past 2^53 | `test_floor_and_ceil_keep_the_integer_type_where_pyarrow_widens_to_double` |
| Shift count outside `[0, bit width)` | shifts the bits out: 0, or the sign fill for a signed `shift_right` | unchecked returns the operand untouched; checked raises. Arrow's range also excludes the sign bit, so `int32 << 31` already differs | `test_out_of_range_shift_counts_shift_the_bits_out_where_pyarrow_returns_the_operand` |
| Nulls in `cumulative_sum`/`_min`/`_max` | the running value carries across a null, output null where input null — Arrow's `skip_nulls=True` | the same with `skip_nulls=True`; its *default* propagates the first null to the end. ArrowMetal has no such mode | `test_cumulative_functions_skip_nulls_where_arrow_propagates_them` |
| A null in `is_in`'s value set | ignored; a null element never matches, so the result has no nulls — Arrow's `skip_nulls=True` | the same with `skip_nulls=True`; its default matches null to null | `test_is_in_never_matches_a_null_where_arrow_matches_null_to_null` |
| Empty pattern in `replace` | the identity | `pc.replace_substring` does not terminate on an empty pattern — the harness must never call it with one | `test_empty_replace_pattern_is_the_identity` |
| Empty pattern in `count_substring` | code points + 1 | bytes + 1 | `test_empty_pattern_counts_code_points_where_arrow_counts_bytes` |
| `float64` `sqrt`/`exp`/`ln`/`log10`/`log2` | evaluated in `float` and widened: ~7 significant digits, nothing below the smallest float32 normal or above `FLT_MAX` | evaluated in double | `test_float64_transcendentals_are_evaluated_in_float32` |

Two of Arrow's own defaults would make the matrix meaningless if the harness accepted them, so the oracle
states the option instead and a test records why. `pc.cumulative_max`'s default `start` is
`numeric_limits<T>::min()`, which on a float column is the smallest positive *normal* — so Arrow's default
clamps every negative running maximum to 1.18e-38 (`test_arrow_cumulative_max_default_start_clamps_negative_floats`);
the matrix passes ∓inf explicitly. And `pc.round` defaults to half-to-even where ArrowMetal rounds halves
away from zero, so the oracle asks for `half_towards_infinity`, the mode the kernel documents.

Because `min`/`max` skip NaN in both engines but disagree on what is left when *every* value is NaN, the
matrix compares `min`/`max`/`mean` on NaN-free input and the NaN behaviour is pinned in tests of its own.
For the same reason the float `cumulative_sum` comparison drops the values that can overflow a partial sum
(±inf, and anything above `type_max / n`): past that point one association reaches ±inf where the other
does not, and `inf - inf` is NaN, which is not a roundoff difference any bound can express. NaN itself
stays in the input — it poisons every later element in both engines
(`test_cumulative_sum_propagates_nan_and_reassociates_infinities`).

## Open findings

Five divergences the harness found that are not bugs but are not free choices either: each is a place
where a kernel's own consistency was preferred to Arrow's answer, or where a documented limit of the GPU
path shows through. Together they account for all 46 failing cases. Each has an entry in `FINDINGS` in
`test_differential.py`, so the matrix groups the affected cells under the finding instead of burying them,
and an `xfail(strict=True)` reproduction, so the suite turns red the moment a kernel changes its mind.

### 1. Float32 arithmetic flushes subnormals to zero

*8 failing cases: `arith_scalar/float32`, `arith_array/float32`, `special` flavor.*

Metal's default math mode is flush-to-zero and denormals-are-zero. The Float32 **arithmetic** kernels
inherit it, so a subnormal result becomes `0.0` and a subnormal operand contributes nothing — while
import/export round-trips the same values perfectly, which makes the loss look like a data bug rather
than a math-mode one.

```python
tiny = 1.1754943508222875e-38                      # smallest normal float32
a = pa.array([tiny, 1.0], pa.float32())
am.array(a).arith("*", 0.5).to_arrow()             # [0.0, 0.5]
pc.multiply(a, pa.scalar(0.5, pa.float32()))       # [5.877471754111438e-39, 0.5]   (so does numpy)
```

Float64 is unaffected — it runs through the software IEEE-754 path and returns the subnormal, which
`test_float64_arithmetic_and_comparison_keep_subnormals` asserts. The kernels that do not add or multiply
were taken off the flush: `compare` builds bit keys, and `sign`, `floor`, `ceil`, `trunc`, `round` and the
element-wise `min`/`max` decide the subnormal and signed-zero cases from the bit pattern, so
`sign(1.4e-45)` is `1` and `ceil(1.4e-45)` is `1.0`, as in Arrow
(`test_float32_comparison_distinguishes_subnormals_from_zero`, `test_sign_of_a_float32_subnormal_is_one`,
`testFloat32SignAndRoundingCorners`). Only `+ - * /` still flush.

Reproduction: `test_float32_arithmetic_keeps_subnormal_results`.

### 2. `upper` and `lower` map Latin only

*18 failing cases: `upper/utf8`, `lower/utf8`, every dataset whose strings need a mapping outside the
covered blocks.*

`am_str_transform` ops 2 and 3 implement the simple (1:1 code point) case mappings of Basic Latin, Latin-1
Supplement and Latin Extended-A, including the ones that change the byte length (`ſ` → `S`, `İ`/`ı`).
Everything above U+017F is copied through, and the multi-character expansions (`ß` → `SS`, `ŉ`, `µ`) are
not applied. pyarrow uses full Unicode.

```python
a = pa.array(["Ωμέγα", "ÅNGSTRÖM"], pa.string())
am.array(a).upper().to_arrow()      # ['Ωμέγα', 'ÅNGSTRÖM']   <- the Greek passes through
pc.utf8_upper(a)                    # ['ΩΜΈΓΑ', 'ÅNGSTRÖM']
```

A full case table is a data-size decision rather than a kernel one, and it is out of scope at 0.1.0 — the
header says so. The finding is classified *by the data*: a dataset counts under it only if it actually
contains a code point outside the covered blocks, so a regression inside them still shows up as a new
divergence. Reproductions: `test_case_mapping_covers_all_of_unicode` (xfail) and
`test_case_mapping_inside_latin_extended_a_matches_pyarrow` (the blocks that must agree exactly).

### 3. `sign` keeps the sign of `-0.0`

*8 failing cases: `sign/float32`, `sign/float64`, the datasets containing a negative zero.*

`am_unary`'s `sign` returns `-1`/`0`/`1` and leaves NaN and both signed zeros alone, so `sign(-0.0)` is
`-0.0`. pyarrow normalises it:

```python
a = pa.array([-0.0, 0.0, math.nan], pa.float64())
am.array(a).sign().to_arrow()      # [-0.0, 0.0, nan]
pc.sign(a)                         # [ 0.0, 0.0, nan]
```

Keeping the operand is the documented behaviour (`include/arrowmetal.h`) and the one that loses no
information; the two engines agree on every other value, NaN included. Reproduction:
`test_sign_of_negative_zero_matches_pyarrow`.

### 4. `is_in` and `index_in` treat `-0.0` and `0.0` as one value

*6 failing cases: `is_in/float64`, `index_in/float64`, the datasets containing a negative zero.*

The set lookup is a binary search over `unique()`, which orders values by the same total order the radix
sort uses: every NaN is one value, and `-0.0` is `0.0`. Arrow's hash lookup keeps the two zeros apart (it
agrees with us on NaN).

```python
a = pa.array([0.0, -0.0], pa.float64())
s = pa.array([-0.0], pa.float64())
am.array(a).is_in(am.array(s)).to_arrow()               # [True, True]
pc.is_in(a, value_set=s, skip_nulls=True)               # [False, True]
```

Matching Arrow here would mean `is_in` disagreeing with `unique`, `dictionary_encode` and `sort` about how
many distinct values a column has, which is the worse of the two inconsistencies. Reproductions:
`test_is_in_separates_negative_zero_from_zero` (xfail) and `test_is_in_matches_nan_to_nan_in_both`.

### 5. `cumulative_prod` reassociates, so overflow and underflow land differently

*`cumulative_prod/float32` and `cumulative_prod/float64`, the datasets whose running product leaves the
normal range.*

The running product is a two-level parallel scan (`Kernels/Window.swift`), so the multiplications happen
in a different order from Arrow's left-to-right loop. Inside the normal range that only moves the last
ulp, which the float tolerance absorbs (`test_cumulative_prod_matches_arrow_inside_range`). Once an
intermediate overflows or underflows the order decides the answer: Arrow's sequential product turns to
`inf` (or `0`) and stays there, while the scan can pair an overflowed partial with an underflowed one and
produce `inf * 0 = NaN`, or skip the overflow altogether.

```python
a = pa.array([1e30, 1e30, 1e-30, 1e-30] * 8, pa.float32())
pc.cumulative_prod(a)                        # [1e30, inf, inf, inf, ...]
am.array(a).cumulative_prod().to_arrow()     # [1e30, inf, 1e30, nan, ...]
```

Matching Arrow exactly would need a sequential pass, which is the one thing the GPU should not do.
The finding is classified by the data (`_prefix_product_leaves_safe_range`): a dataset counts only when
its sequential running product gets within 2^40 of overflow or of the smallest normal. Reproduction:
`test_cumulative_prod_matches_arrow_past_overflow` (xfail).

## Findings that were fixed

The five bugs the first run of this matrix reported are closed. The reproductions stayed, as plain
assertions, so the suite notices a relapse.

| Was | Now | Test |
|---|---|---|
| `argsort`/`top_k` ordered null indices by the bytes under the validity bitmap | null rows keep their input order | `test_argsort_null_block_is_stable`, `test_top_k_null_block_is_stable` |
| `sort`/`argsort`/`top_k` split the `-0.0` / `0.0` tie | the sort keys canonicalise the sign of zero, so the tie holds input order | `test_negative_zero_is_a_sort_tie` |
| `sum`/`mean` over Float32 accumulated in Float32 | they accumulate in double, as Arrow's do | `test_float32_sum_does_not_overflow_before_arrow_does` |
| Grouped `sum` over UInt64 came back as Int64 | the aggregate keeps the value type | `test_group_by_sum_over_uint64_stays_unsigned` |
| Float32 comparison treated subnormal operands as zero | comparisons are exact, on bit keys; only arithmetic still flushes | `test_float32_comparison_distinguishes_subnormals_from_zero` |

Four more turned up while closing those, and were fixed here rather than written down:

- **`top_k` did not canonicalise floats at all.** The selection kernel carries its own key mapping, which
  the `-0.0` and NaN fix had never reached, so `top_k` could order a column differently from `argsort`. It
  now uses the same mapping (`testNaNAndNegativeZeroPlacement`).
- **A descending sort mirrored NaN to the front.** Arrow's null placement covers NaN, so a reversed order
  keeps NaN at the end, next to the nulls; the key mapping now sends NaN to the maximum key when the order
  is inverted (`test_nan_sorts_last_in_both_directions_in_both`).
- **Element-wise `min`/`max` broke a ±0 tie by position**, so `min(0.0, -0.0)` and `min(-0.0, 0.0)`
  disagreed. Both now answer the way `fmin`/`fmax` do — min keeps `-0.0`, max keeps `0.0`
  (`test_element_wise_min_max_break_a_zero_tie_like_fmin_and_fmax_in_both`).
- **Float32 `sign`, `ceil`, `floor`, `trunc` and `round` inherited the arithmetic flush**, so
  `sign(1.4e-45)` returned the subnormal itself, `ceil(1.4e-45)` was `0.0`, and `round(-0.4)` lost the
  sign of its zero. All five decide those cases on the bit pattern now, which is what the exact float64
  kernels already did (`testFloat32SignAndRoundingCorners`).


## What passed

Everything else, on all 27 datasets per cell:

- **Import/export round trip** for all 12 types, including every sliced offset, all-null and empty arrays,
  4096-byte strings and multi-byte UTF-8 — type, length and `null_count` preserved.
- **Reductions** `sum`, `min`, `max`, `mean` on all 10 numeric types, including UInt64 totals above `2^63`
  and Int64 totals that wrap.
- **Comparisons** all six operators, scalar and array, on all numeric types — including UInt64 against
  `2^63`, where a signed comparison would flip.
- **Arithmetic** `+ - * /`, scalar and array, bit-exact on both float types (outside finding 1) and exact
  under wrapping on all eight integer types; `modulo` and `power` on all eight, against a truncating-division
  oracle and with the exponent folded into [0, 8).
- **Casts** all 100 numeric source/target pairs, matching `Array.cast(safe=False)` including narrowing
  wraps and float→int truncation toward zero.
- **Boolean** `and`, `or`, `not` with nulls.
- **Selection** `filter` (null mask entries drop the row, as in Arrow), `filter_where` for all six
  predicates, `take` with repeated and null indices, `slice`.
- **`sort`, `argsort`, `top_k`** ascending and descending on all numeric types, at every size, including
  `-0.0`, NaN and null placement -- the three now agree with Arrow everywhere in the matrix.
- **Element-wise math** `abs`, `negate`, `sign`, `floor`, `ceil`, `round`, `trunc`, `sqrt`, `exp`, `ln`,
  `log10`, `log2`, element-wise `min`/`max`, the four bit-wise ops and both shifts.
- **Cumulative** `sum`, `min`, `max` on all ten numeric types, nulls included.
- **Structural** `is_null`, `is_valid`, `fill_null`, `drop_null`, `if_else`, `is_in`, `index_in` and the
  Kleene `and`/`or`.
- **Strings** `byte_length`, `char_length`, `starts_with`, `ends_with`, `str_contains`, `str_equals`
  (scalar and array) against 11 patterns including the empty string, a multi-byte prefix, a tab and a
  300-byte needle; the ASCII case and trim transforms, `replace`, `repeat`, `slice_codeunits`, the two
  pads, `str_reverse`, `str_concat`, `count_substring`, `find_substring` and the six ASCII predicates;
  `dictionary_encode` round trip, first-seen dictionary order and `decode`; `hash32` null propagation and
  injectivity.
- **Group-by** `count`, `count_values`, `sum`, `min`, `max`, `mean` on every value type the kernels cover.

## Reading the report

`differential_report.py` prints one row per operation and one column per type. A cell says `ok N` when all
N datasets agree, `known n/N` when n of them hit one of the open findings above, `NEW n/N` for a divergence
that is *not* in this document, `skip N` where the kernel does not exist for that type, and `-` where the
operation does not apply.

**It exits on the `unclassified` count, not on the failure count.** A documented divergence has a FINDINGS
entry, a strict-xfail reproduction and a paragraph above it: it is a decision on the record, and holding CI
red on it would only teach everyone to ignore the gate. An unclassified divergence is one nobody has looked
at, and that is what fails the build -- 0 at 0.1.0, and it should stay there. The totals line prints both
numbers, and the last line reads `PASS (n documented divergence(s))` so the count cannot drift unnoticed.

`--ops` and `--types` narrow the matrix while chasing one cell:

    PYTHONPATH=python python python/tests/differential_report.py --ops argsort,top_k --types int32
