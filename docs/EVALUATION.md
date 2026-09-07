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
| Oracle | `pyarrow.compute` 25.0.1, plus a reference written in the harness for the 16 operations Arrow has no function for |
| Cases in the default matrix | 33,156 — 181 operations x 45 column types x 27 datasets — about 160 s on an M4 Max; more with `DIFF_LARGE=1` |
| Result at 0.1.0 | 30,570 pass, 1,677 fail across 21 documented divergences, 909 skip (kernels not implemented), **0 unclassified** |

## Method

**One oracle, never a literal.** Every expected value comes from a `pyarrow.compute` call on the same
array: `pc.sum`, `pc.min`/`pc.max`, `pc.equal`…, `pc.add`/`pc.subtract`/`pc.multiply`/`pc.divide`,
`Array.cast(safe=False)`, `pc.and_`/`pc.or_`/`pc.invert`, `Array.filter`, `Array.take`, `Array.slice`,
`pc.array_sort_indices`, `pc.binary_length`, `pc.utf8_length`, `pc.starts_with`, `pc.ends_with`,
`pc.match_substring`, and `pa.Table.group_by(...).aggregate(...)` for the grouped aggregates. Every
option is spelled out rather than defaulted, so the oracle says what the kernel implements: `pc.rank`
gets its `tiebreaker`, `pc.week` its three `WeekOptions` flags, `pc.round` its `round_mode`,
`pc.quantile` its `interpolation`, `pc.assume_timezone` its `ambiguous`/`nonexistent`, `pc.is_in` its
`skip_nulls`, `pc.list_slice` its `return_fixed_size_list`.

**Where Arrow has no function.** Sixteen operations have no `pyarrow.compute` counterpart at all. Each
names itself in `_NO_ORACLE`, `test_no_oracle_operations_are_reported` prints the list on every run, and
each is compared against a reference written out in the harness rather than against nothing:

| Operation | Reference |
|---|---|
| `hash32`, `hash64` | no value to compare: null propagation, equal-values-hash-equal, and injectivity over the distinct values |
| `percent_rank_and_cume_dist` | the SQL definitions, computed from the same ascending-nulls-last order `pc.rank` uses |
| `shift` | the shifted Python list |
| `cumulative_mean` | the exact running sum over the running count, on the values a double can hold exactly |
| `rolling_sum`, `rolling_mean`, `rolling_min_max` | each window taken as an Arrow slice and reduced with `pc.sum` / `pc.mean` / `pc.min` / `pc.max`, so only the windowing is the harness's |
| `run_end` | the decoded column, put through pyarrow's own kernels — pyarrow has no filter/take/slice for a run-end array |
| `temporal_round_duration` | integer arithmetic on the tick count, which is what the kernel documents |
| `temporal_clock_on_a_date` | the clock of the midnight the date names: zero where valid, null where not |
| `temporal_field_types` | the recorded result-type contract (`_TEMPORAL_RESULT_TYPES`) |
| `interval_layouts` | the `month_day_nano` result, which pyarrow *does* compute and the matrix compares against it |
| `add_interval` | `pc.add` of the equivalent duration for the day/time half, and a Python calendar reference for the month half |
| `string_normalize` | Python's `unicodedata` — pyarrow 25.0.1's `utf8_normalize` ignores its `form` option |
| `extension_type` | the round trip through pyarrow's own extension registry |
| `nulls_constructor` | `pa.nulls(n)` |

**Generated input, not fixtures.** The generator covers every type ArrowMetal imports — 45 columns:

- **Flat primitives** `int8 int16 int32 int64`, `uint8 uint16 uint32 uint64`, `float32`, `float64`,
  `bool`, `utf8`.
- **Temporal** `timestamp` in all four units, each with and without a timezone (`America/New_York`, for
  its DST rule); `date32`, `date64`; `time32[s]`, `time32[ms]`, `time64[us]`, `time64[ns]`; `duration`
  in all four units. Values are drawn from 1900-01-01 to 2100-01-01, and the special flavor adds the
  epoch and the second either side of it, both US DST transitions of 2024, the leap days of 1900, 2000
  and 2024, the ISO-week corners of 2021 and 2025, and the ends of the window.
- **Decimal** `decimal128(9,2)`, `decimal128(18,6)` and `decimal128(38,10)` — Arrow's maximum precision,
  generated with full 38-digit values, negative values, and the extremes ±(10^p − 1) in the special
  flavor — plus `decimal32(9,2)` and `decimal64(18,4)` for the widening and narrowing casts.
- **Nested** `list<int64>`, `list<utf8>` (rows of 0 to 4 elements, with nulls inside the rows),
  `struct<a: int64, b: utf8>` and `map<utf8, int64>` (with duplicate and empty keys).
- **The rest of the type matrix** `float16`, `fixed_size_binary(8)`, `dictionary<int32, utf8>`,
  `run_end_encoded<int32, int64>` (runs of 1 to 6), `null`, and `month_day_nano_interval`.

For every one of them the generator produces:

- **Sizes** 0, 1, 33, 1000, 100,003; add 5,000,000 with `DIFF_LARGE=1`, drop 100k with `DIFF_QUICK=1`.
  33 and 100,003 are deliberately not multiples of a threadgroup or SIMD width.
- **Null ratios** 0, 0.3 and 1.0. Nulls are applied with `pa.array(values, mask=...)`, which leaves the
  original numbers *under* the validity bitmap — what a real column looks like after a filter, and what a
  kernel is required to ignore. That choice is what surfaced the argsort null-order bug.
- **Special values** (the `special` flavor): each type's min and max, 0, 1, -1, half-range; for floats
  `-0.0`, NaN, ±inf, the smallest normal, the smallest subnormal and their negatives, `FLT_MAX`/`DBL_MAX`
  and machine epsilon; for strings the empty string, multi-byte UTF-8 (`héllo`, `日本語`, `Ωμέγα`, an
  emoji pair, a full-width digit run and a title-case code point), embedded tabs and newlines, and
  300- and 4096-byte strings. The two long strings drop out of the pool above 200,000 rows, where they
  would otherwise make a gigabyte-scale array per case. The temporal, decimal, `float16` and
  `fixed_size_binary` generators have their own special pools, listed above.
- **Sliced arrays** (the `sliced` flavor): the array is built two offsets longer and then
  `pa.Array.slice(offset, size)` at offset 3 or 5 — neither byte- nor 8-bit-aligned, so the import path has
  to honour `ArrowArray.offset` for the values buffer, the validity bitmap and the utf8 offset buffer alike.

That is 27 datasets per (operation, type) by default.

**Every method that exists is in scope.** Operations are a registry in `test_differential.py`, and
`test_every_public_operation_has_a_differential_case` fails if a method is added to `MetalArray` or
`GroupBy` without one, so the harness cannot silently fall behind the library;
`test_every_module_level_function_has_a_differential_case` does the same for `coalesce`, `case_when`,
`choose`, `lexsort_indices` and `nulls`. Optional operations register themselves the moment the method
appears on `MetalArray`, so a new kernel joins the matrix with no edit here; the ones still missing
(`reverse` at 0.1.0) are listed by the report as absent. `_NO_MATRIX_TYPE` — the list of methods the
generator could not reach — is **empty**: it used to hold the temporal kernels, which now have their
own columns. The only members outside the matrix are the six accessors and the six generic dispatchers
(`unary`, `binary`, `cumulative`, `window`, `string_predicate`, `string_transform`), every op of which
is reached through a named form that *is* in the matrix.

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
wrong answer, is a *failure*. The 909 skips in the default run are:

- the group-by combinations the kernels do not cover — `min`/`max` on 64-bit values, `mean` on Float32,
  and any aggregate over Float64, the same set `test_arrowmetal.py` pins as expected errors;
- the primitive-only kernels (`is_null`, `is_valid`, `fill_null`, `drop_null`, `if_else`, `is_in`,
  `index_in`, `case_when`, `choose`, `coalesce`) on `utf8` and `bool`;
- `mode` and `first`/`last`/`index` on a boolean column (`am_reduce_ex` does not define them there);
- `subtract_temporal` on a time-of-day column (two times of day do not make a duration here);
- run-end encoded arrays with a non-zero offset ("decode on the producer side first"), which is the
  `sliced` flavor of `roundtrip_ext`;
- the cells where the operation has nothing to do: `temporal_round_finer` on a nanosecond column,
  because no unit is finer than its own tick.

The report prints all of them, grouped by the message the kernel gave.

**Tolerances, and where each comes from.** Exactness is the default; every departure is named:

| Comparison | Tolerance | Why |
|---|---|---|
| integers, booleans, strings, index vectors, decimals, temporal values, nested values | exact | nothing accumulates |
| float `+ - * /`, scalar and array | bit-exact | the two engines do the same operation in the same order |
| `sum`, `mean`, `cumulative_sum`, `rolling_sum`, `rolling_mean`, the grouped aggregates | relative 1e-6 (float32) / 1e-12 (float64), plus the roundoff bound `8·eps·sqrt(n)·Σ|x|` | the two engines accumulate in a different order; without a bound that scales with the input a 100k-row float32 sum cannot be compared at all |
| `variance`, `stddev` | relative 1e-6 (float32) / 1e-7 | the kernel accumulates the two moments and subtracts them, which costs about six digits against pyarrow's exact two-pass algorithm — measured at 6e-10 over the matrix and 2e-8 on a float32 column |
| `sqrt`, `exp`, `ln`, `log10`, `log2` | relative 1e-6 (plus an absolute 1e-6 for the logarithms, and a `|x|`-scaled bound for `exp`) | they evaluate in `float` and widen back |
| `sin`…`atanh`, `atan2` | relative 1e-6 (float32) / 1e-14 (float64) | the header measures the worst case at 4 ulp and 5 ulp against the host libm; every special value (NaN, ±inf, ±0, the domain errors) is compared exactly by a pinned test instead |
| `quantile`, `median`, `cumulative_mean` | relative 1e-12 | a double division at the end of an exact selection |

Nothing else is given a tolerance. In particular the temporal, decimal, string and nested families are
compared exactly, type included.

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
| `float64` `sqrt`/`exp`/`ln`/`log10`/`log2`/`power` | software IEEE-754 binary64 on the GPU: `sqrt` correctly rounded, the others within 1 ulp of libm over the whole double range (was: evaluated in `float`, fixed 2026-09-06) | evaluated in double | `test_float64_transcendentals_are_true_binary64` |
| `list_element` on a row shorter than the index | null | raises `ArrowInvalid` for the whole column | `test_list_element_of_a_short_row_is_null_where_pyarrow_raises` |
| `parse` of a string that is not a number | null | `cast` raises, even with `safe=False` | `test_parse_returns_null_where_pyarrow_raises` |
| adding a duration to a time of day past midnight | wraps inside the day | raises: the result is outside `[0, 86400)` | `test_time_of_day_addition_wraps_where_pyarrow_raises` |
| `list_parent_indices` | `int32` — list offsets are `int32` throughout this package | `int64` | `test_list_parent_indices_are_int32_where_pyarrow_returns_int64` |
| the temporal extractors (`year`…`nanosecond`, `day_of_week` with the default options) | `int32` | `int64` (`us_week`, `us_year`, `week` and the option-carrying `day_of_week` are `int64` in both) | `test_temporal_extractors_return_int32_where_pyarrow_returns_int64` |
| `null_count` of a run-end encoded column | the logical count | always 0 — the nulls live in the values child | `test_run_end_null_count_is_logical` |
| `filter`/`take`/`slice` of a run-end encoded column | decodes to a flat column | no kernel for the type at all | `test_run_end_selection_decodes` |
| the replacement template in `replace_substring_regex` | ICU's: `$1`, `$2` | RE2's: `\1`, `\2` | `test_regex_replacement_template_is_icu_not_re2` |
| `index`'s needle on a `uint64` value above 2^63 | not found: the needle crosses the C ABI as a `double` | found | `test_index_of_a_large_unsigned_value_is_not_found` |
| `min`/`max` of an all-NaN *rolling window* | the scan's identity, ±inf | NaN, as `pc.min` does | `test_rolling_min_of_an_all_nan_window` |
| decimal `+`/`-`/`*` past the column's precision | wraps modulo 2^128, the storage width | `pc.add` refuses precision 39 outright; its unchecked cast wraps modulo 10^precision | `test_decimal_arithmetic_wraps_modulo_two_to_the_128` |

Three of the divergences the harness turned up are **pyarrow's**, not ArrowMetal's. They are recorded
here because the matrix has to work around them, and each has a test that fails if pyarrow fixes it:

| Case | pyarrow 25.0.1 | Correct | Test |
|---|---|---|---|
| `pc.utf8_normalize` | decomposes whatever `form` says — its NFC is NFD | ArrowMetal and Python's `unicodedata` agree | `test_pyarrow_utf8_normalize_ignores_its_form_option` |
| `pc.pairwise_diff` on a sliced column | reads the values buffer without `ArrowArray.offset` | ArrowMetal honours the offset; the matrix hands the oracle a materialised copy | `test_pyarrow_pairwise_diff_ignores_the_array_offset` |
| `pc.fill_null_forward`/`_backward`/`replace_with_mask` on a sliced **boolean** column | the same offset bug | same | `test_pyarrow_boolean_fill_null_forward_ignores_the_array_offset` |

And one is a crash rather than a wrong answer: `pc.year_month_day` and `pc.iso_calendar` corrupt the
heap in pyarrow 25.0.1 and segfault the process a couple of allocations later, so the matrix never
calls them — `temporal_struct` compares the two struct-valued kernels field by field against
`pc.year`/`pc.month`/`pc.day` and `pc.iso_year`/`pc.iso_week`/`pc.day_of_week` instead, which is a
stronger check anyway. See `docs/FINDINGS.md`.

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

Seventeen divergences the harness found that are not bugs but are not free choices either: each is a
place where a kernel's own consistency was preferred to Arrow's answer, or where a documented limit of
the GPU path shows through. Together they account for all 1,677 failing cases. Each has an entry in
`FINDINGS` in `test_differential.py`, so the matrix groups the affected cells under the finding instead
of burying them, and an `xfail(strict=True)` reproduction, so the suite turns red the moment a kernel
changes its mind. Thirteen of them are classified *by the data* — a `data_check` that looks at the
generated values — so a dataset that does not actually contain the triggering value still has to agree
exactly.

| # | Finding | Cases | Cells |
|---|---|---|---|
| 1 | `float32-subnormal-ftz` | 19 | 5 |
| 2 | `sign-of-negative-zero` | 8 | 2 |
| 3 | `negative-zero-set-lookup` | 14 | 4 |
| 4 | `cumulative-prod-reassociation` | 12 | 2 |
| 5 | `decimal-to-float64-divides` | 38 | 3 |
| 6 | `decimal-round-carry-past-the-precision` | 12 | 3 |
| 7 | `regex-icu-unicode-classes` | 28 | 2 |
| 8 | `regex-anchor-in-a-repeated-search` | 15 | 1 |
| 9 | `split-loses-the-null-row` | 31 | 2 |
| 10 | `split-whitespace-trailing-run` | 14 | 1 |
| 11 | `float-text-swift-format` | 35 | 2 |
| 12 | `temporal-extract-in-utc` | 977 | 58 |
| 13 | `strftime-seconds-carry-the-fraction` | 57 | 3 |
| 14 | `timezone-after-2038` | 96 | 8 |
| 15 | `trig-argument-reduction` | 8 | 2 |
| 16 | `variance-accumulator-overflow` | 8 | 2 |
| 17 | `rolling-min-max-zero-and-subnormal` | 5 | 2 |

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

### 2. `sign` keeps the sign of `-0.0`

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

### 3. `is_in` and `index_in` treat `-0.0` and `0.0` as one value

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

### 4. `cumulative_prod` reassociates, so overflow and underflow land differently

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

### 5. `decimal` → `float64` divides where Arrow multiplies by the reciprocal

*38 failing cases: `decimal_to_float64` on all three decimal128 columns.*

`am_decimal_op` op 16 computes `unscaled / 10^scale` in double, which is correctly rounded — the nearest
double to the exact decimal value. Arrow's cast multiplies by the precomputed reciprocal `10^-scale`,
which is a ulp out on most values.

```python
a = pa.array([decimal.Decimal("99.99")], pa.decimal128(9, 2))
am.array(a).to_float64().to_arrow()   # [99.99]                <- 9999 / 100, exactly
a.cast(pa.float64())                  # [99.99000000000001]
```

The kernel's answer is the better one, and changing it to reproduce Arrow's rounding would mean
introducing an error on purpose. Reproductions: `test_decimal_to_float64_matches_arrows_cast` (xfail)
and `test_decimal_to_float64_is_the_correctly_rounded_quotient`.

### 6. Rounding a decimal down narrows the precision, and a carry can outgrow it

*12 failing cases: `decimal_round` on the three decimal128 columns, the `special` flavor.*

`decimal_round(target)` **rescales**: the result type is `decimal128(precision − (scale − target),
target)` when rounding down and `decimal128(precision + target − scale, target)` when rounding up.
Scaling up is exact. Scaling down is not always: a value whose rounding carries needs one digit more
than the narrowed precision, and the kernel keeps the value rather than wrapping it.

```python
a = pa.array([decimal.Decimal("9999999.99")], pa.decimal128(9, 2))
am.array(a).decimal_round(0).to_arrow()          # decimal128(7, 0): [10000000]  <- 8 digits
pc.round(a, ndigits=0, round_mode="half_towards_infinity")   # decimal128(9, 2): [10000000.00]
```

Arrow keeps the input scale instead and never has to narrow, so the two only part company on the type
and on that carry; casting Arrow's answer into the narrowed type wraps it modulo `10^precision`, which
is not a number anyone wants. Classified by the data (`_decimal_rounding_carries`). Reproductions:
`test_decimal_round_carry_wraps_like_arrows_cast` (xfail) and
`test_decimal_round_narrows_the_precision_by_the_digits_it_drops`.

### 7. ICU's `\d`, `\w` and `\s` are Unicode-aware; RE2's are ASCII

*28 failing cases: `regex_match` and `regex_replace` on `utf8`, the datasets containing a full-width
digit.*

ArrowMetal matches with ICU (`NSRegularExpression`) and pyarrow with RE2. The syntaxes agree on
everything the matrix uses except the character classes: ICU's `\d` is Unicode category Nd, RE2's is
`[0-9]`.

```python
a = pa.array(["２０２４"], pa.string())
am.array(a).match_substring_regex(r"\d").to_arrow()   # [true]
pc.match_substring_regex(a, r"\d")                     # [false]
```

Which is right is a matter of which engine you came from; neither is a bug. The finding is classified by
the data — a column has to contain a non-ASCII code point of category Nd to count under it — so a
column of plain ASCII still has to agree, which `test_regex_digit_class_agrees_on_ascii` asserts.
Reproduction: `test_regex_digit_class_is_ascii_only` (xfail).

### 8. `^` in a repeated search anchors to the input, not to the search

*15 failing cases: `regex_match` on `utf8`, the datasets with a row an anchored pattern can match twice.*

`count_substring_regex` and `replace_substring_regex` search repeatedly. ICU anchors `^` at the start of
the *input* and keeps it there; RE2 re-anchors it at the start of each search, so a second match can
appear where there is no second line.

```python
a = pa.array(["aa"], pa.string())
am.array(a).count_substring_regex("^a").to_arrow()   # [1]
pc.count_substring_regex(a, "^a")                     # [2]
```

Reproduction: `test_anchored_pattern_counts_once_per_input` (xfail).

### 9. `split` has nowhere to put a null row

*31 failing cases: `regex_split` and `split_whitespace` on `utf8`, every dataset with a null.*

ArrowMetal has no list type, so `split_pattern` and `split_whitespace` return the `(offsets, values)`
pair of an Arrow `list<utf8>` instead — and a pair of flat arrays has no validity bitmap for the rows.
A null value splits to an empty row where Arrow's `list<utf8>` keeps the null.

```python
a = pa.array(["a b", None], pa.string())
offsets, values = am.array(a).split_pattern(" ")
pa.ListArray.from_arrays(offsets.to_arrow(), values.to_arrow())   # [["a", "b"], []]
pc.split_pattern(a, " ")                                          # [["a", "b"], null]
```

Giving the pair a third array for the validity would be a list type in all but name; the caller that
needs the distinction has `is_null()`. Classified by the data (the column has to have a null).
Reproduction: `test_split_keeps_the_null_row` (xfail).

### 10. `split_whitespace` gives one empty piece for a trailing run, Arrow gives two

*14 failing cases: `split_whitespace` (the `unicode=True` path) on `utf8`, on the datasets with a value
whose trailing whitespace run is two or more characters long.*

Both engines treat a run of whitespace as one separator and keep the empty pieces at either end:
`"  padded"` is `['', 'padded']` in both, `" "` is `['', '']` in both. Arrow's `utf8_split_whitespace`
is asymmetric on its own — a trailing run of two or more characters produces *two* empty pieces — while
Arrow's `ascii_split_whitespace` produces one, and so does ArrowMetal in both modes. The Unicode flag
lines up with Arrow's pair: `split_whitespace(unicode=True)` is compared with `utf8_split_whitespace`
(U+3000 and U+00A0 separate) and the default with `ascii_split_whitespace`, whose cell passes.

```python
a = pa.array(["padded  "], pa.string())
am.array(a).split_whitespace(unicode=True).to_arrow()   # [['padded', '']]
pc.utf8_split_whitespace(a)                             # [['padded', '', '']]
pc.ascii_split_whitespace(a)                            # [['padded', '']]   <- Arrow's own ASCII variant agrees
```

Classified by the data (`_ends_with_a_whitespace_run`). Reproductions:
`test_split_whitespace_trailing_run_matches_arrow` (xfail),
`test_ascii_split_whitespace_trailing_run_matches_arrows_ascii_variant` and
`test_split_whitespace_agrees_with_arrow_away_from_a_trailing_run`.

### 11. `float` → `utf8` uses Swift's formatting

*35 failing cases: `to_strings_text` on `float32` and `float64`.*

`am_to_strings` formats a float on the CPU with Swift's own `description`, which is a shortest
round-tripping decimal like Arrow's — but shaped differently: an integral value keeps its `.0`, the
exponent has two digits, and the switch to scientific notation happens at a different magnitude.

```python
a = pa.array([1.0, -0.0, 1.1786107e-06, 7.5e-07], pa.float64())
am.array(a).to_strings().to_arrow()   # ['1.0', '-0.0', '1.1786107e-06', '7.5e-07']
a.cast(pa.string())                   # ['1',   '-0',   '0.0000011786107', '7.5e-7']
```

Both texts name the same number, and that is the property the matrix gates on: `to_strings_value`
reads ArrowMetal's own digits back with Python's `float` and compares them with the input **bit for
bit**, `-0.0`, the subnormals and `DBL_MAX` included, and it passes everywhere. Only the *text* is a
finding, classified by the data (`_float_text_differs`). Reproductions:
`test_float_to_text_matches_arrows_formatter` (xfail) and `test_float_to_text_names_the_same_number`.

### 12. The temporal kernels read a zoned timestamp in UTC

*977 failing cases across 58 cells: every temporal operation on the four `timestamp[…, tz]` columns.*

`am_temporal_extract`, `am_temporal_math` and `am_temporal_extra` all work on the instant, in UTC,
whatever timezone the column's type carries — which is what the class documents ("extracted in UTC").
pyarrow reads a zoned timestamp in its own zone, so every field, rounding and difference differs by the
offset.

```python
a = pa.array([0], pa.timestamp("s", "America/New_York"))
am.array(a).hour().to_arrow()   # [0]     <- 1970-01-01T00:00:00Z
pc.hour(a)                       # [19]    <- 1969-12-31T19:00:00-05:00
```

Reading in UTC is the only thing a GPU kernel can do without the tz database, and it is what the two
functions that *do* consult the database (`assume_timezone`, `local_timestamp`) are for: convert first,
extract after. The finding is the largest in the matrix, so it gets a companion operation rather than a
blanket exemption — `temporal_utc_semantics` runs every extractor and `floor_temporal` on the zoned
column and compares them with **Arrow's own answer for the same instants written without a zone**, and
it passes everywhere. The zone is therefore the whole of the difference, which is what
`test_temporal_extract_is_arrows_answer_for_the_same_naive_instant` asserts. Reproduction:
`test_temporal_extract_uses_the_columns_timezone` (xfail).

### 13. `%S` prints whole seconds

*57 failing cases: `strftime_seconds` on the sub-second timestamp columns.*

`am_parse`'s strftime path is C's, where `%S` is the two-digit second. Arrow's appends the sub-second
digits of the column's own resolution to it.

```python
a = pa.array([1_700_000_000_123_456], pa.timestamp("us"))
am.array(a).strftime("%S").to_arrow()      # ['20']
pc.strftime(a, format="%S")                # ['20.123456']
```

`%f` is the ArrowMetal extension for the fractional second, so the information is reachable; a `%S`
that silently changes width with the column's unit is not something a C format string should do. Every
other field agrees (`test_strftime_agrees_on_every_other_field`), which is why the `%S` formats are an
operation of their own. Reproduction: `test_strftime_seconds_carry_the_fraction` (xfail).

### 14. pyarrow's timezone database stops transitioning at 2038

*96 failing cases: `assume_timezone` and `temporal_timezone` on the datasets reaching past 2038.*

This one is pyarrow's, not ours, but it is a divergence the matrix has to carry: Arrow's bundled tz
database stops applying a DST rule once the 32-bit epoch runs out, so every summer instant after
January 2038 comes back with the winter offset. Foundation keeps applying the rule.

```python
a = pa.array([_seconds_of(2050, 7, 15, 12)], pa.timestamp("s"))   # July, so EDT = UTC-4
am.array(a).assume_timezone("America/New_York", "earliest", "earliest")   # offset -14400
pc.assume_timezone(a, "America/New_York", ambiguous="earliest", nonexistent="earliest")  # -18000
```

Before 2038 the two agree everywhere in the matrix, which
`test_assume_timezone_agrees_before_2038_and_keeps_the_rule_after` asserts along with the one-hour gap
after it. Classified by the data (`_after_the_2038_cutoff`). Reproduction:
`test_assume_timezone_agrees_past_2038` (xfail).

### 15. The software binary64 trigonometry loses its argument reduction past 2^49

*8 failing cases: `trig` and `trig_checked` on `float64`, the datasets containing a value at or above
2^49.*

`sin`, `cos` and `tan` on a Float64 column run the software binary64 implementation on the GPU, whose
argument reduction carries a fixed number of bits of π. Below 2^49 the results are bit-comparable with
the host libm; above it they drift, and past about 2^61 the reduction gives up and returns NaN.

```python
a = pa.array([2.0 ** 52 + 0.5], pa.float64())
am.array(a).sin().to_arrow()   # [0.797567…]
pc.sin(a)                       # [0.874217…]
```

A Payne–Hanek reduction would fix it and would cost every ordinary argument the time; the header states
the limit instead. Float32 is unaffected — it runs Metal's own functions, which reduce correctly across
the whole range. Classified by the data. Reproductions: `test_trig_reduces_a_large_argument` (xfail)
and `test_trig_agrees_below_the_reduction_limit`.

### 16. `variance` and `stddev` accumulate the moments in 64 bits

*8 failing cases: `variance_and_stddev` on `int64` and `uint64`, the `special` flavor.*

The two moments are accumulated on the GPU in a 64-bit accumulator. For a column of 64-bit integers
past about 2^40 the sum of squares leaves it, and the answer stops meaning anything — a wrapped number,
or NaN.

```python
a = pa.array([2 ** 62] * 3, pa.int64())
am.array(a).variance(0)            # 1.3611295640251995e+37
pc.variance(a, ddof=0).as_py()     # 0.0                     <- three equal values
```

Inside the accumulator the kernel is accurate to about ten significant digits, which is the tolerance
the matrix gives it (see *Tolerances* above) and what `test_variance_is_accurate_inside_the_accumulator`
asserts. Classified by the data (`_moments_leave_the_accumulator`). Reproduction:
`test_variance_of_a_large_int64_column` (xfail).

### 17. The rolling `min`/`max` scan compares raw values

*5 failing cases: `rolling_min_max` on the float columns containing a negative zero or a float32
subnormal.*

The scalar `min`/`max`, the sort and the element-wise `min`/`max` all decide their ±0 and subnormal
cases from the bit pattern — that was one of the fixes in the first round. The rolling window's scan
kept the plain comparison, so a ±0 tie is broken by position rather than the way `fmin` does it.

```python
a = pa.array([0.0, -0.0], pa.float64())
am.array(a).rolling_min(2).to_arrow()   # [null, 0.0]
pc.min(a).as_py()                        # -0.0
```

Small, and the same shape as the bugs that were fixed in the selection kernels; it is written down here
rather than fixed because the rolling kernels are the newest in the package and the fix belongs with
their next revision. Classified by the data. Reproduction:
`test_rolling_min_breaks_a_zero_tie_like_fmin` (xfail).

## Findings that were fixed

The five bugs the first run of this matrix reported are closed, and four of the original twenty-one findings were closed by later kernel work (full-Unicode case mapping, and the three calendar-rounding corners). The reproductions stayed, as plain
assertions, so the suite notices a relapse.

| Was | Now | Test |
|---|---|---|
| `argsort`/`top_k` ordered null indices by the bytes under the validity bitmap | null rows keep their input order | `test_argsort_null_block_is_stable`, `test_top_k_null_block_is_stable` |
| `sort`/`argsort`/`top_k` split the `-0.0` / `0.0` tie | the sort keys canonicalise the sign of zero, so the tie holds input order | `test_negative_zero_is_a_sort_tie` |
| `sum`/`mean` over Float32 accumulated in Float32 | they accumulate in double, as Arrow's do | `test_float32_sum_does_not_overflow_before_arrow_does` |
| Grouped `sum` over UInt64 came back as Int64 | the aggregate keeps the value type | `test_group_by_sum_over_uint64_stays_unsigned` |
| Float32 comparison treated subnormal operands as zero | comparisons are exact, on bit keys; only arithmetic still flushes | `test_float32_comparison_distinguishes_subnormals_from_zero` |
| `top_k` put row 0 first about once in 900 calls: every thread read the candidate count from a relaxed threadgroup atomic, one simdgroup could see a newer value, and the sentinel padding then left holes that sorted ahead of the real candidates | the count is broadcast through a plain threadgroup variable between barriers, and the buffer starts full of sentinels | `TopKTests.testStressAgainstCPUOracle` (`ARROWMETAL_STRESS=1` for the long run) |
| `upper`/`lower` mapped Basic Latin, Latin-1 and Latin Extended-A only; everything above U+017F passed through | the GPU table covers U+0000–U+017F and any row holding a code point above it is mapped on the host with Unicode's simple mapping, so Greek, Cyrillic, Turkish dotted I and astral scripts agree with pyarrow | `test_case_mapping_covers_all_of_unicode` |
| `ceil_temporal` left a value already on a month, quarter or year boundary alone; Arrow advances it a whole unit | the calendar units advance, the fixed-length units keep the value, as Arrow's do | `test_ceil_temporal_advances_a_value_on_a_month_boundary` |
| a multiple of months or quarters counted from year 0; Arrow counts from 1970-01 | months and quarters count from 1970-01 (years from year 0, as Arrow's do) | `test_calendar_multiples_share_an_origin` |
| rounding to a unit finer than the column's resolution was the identity; Arrow converts, rounds and truncates back | the value is converted to the finer unit, rounded there and truncated back | `test_rounding_to_a_finer_unit_converts_like_arrow` |

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

Everything else — 30,570 of the 33,156 cases — on all 27 datasets per cell:

- **Import/export round trip** for all 45 types, including every sliced offset, all-null and empty arrays,
  4096-byte strings and multi-byte UTF-8, 38-digit decimals, nested lists with nulls inside the rows,
  maps with duplicate keys, a dictionary, a run-end encoded column and the `null` type — type, length and
  `null_count` preserved. `filter`, `take` and `slice` follow on all 45 as well.
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
- **Decimals** the six comparisons scalar and array, `add`/`subtract` against the operand's own type and
  `multiply` against Arrow's widened one (in decimal256 where 38 digits overflow Arrow's own precision),
  `negate`/`abs`/`sign`, `round`/`ceil`/`floor`/`truncate` to five target scales each, `sum`/`min`/`max`,
  and the decimal32/64 widening and narrowing casts — all exact, at 9, 18 and 38 digits of precision.
- **Nested** `list_value_length`, `list_flatten`, `list_element` at three indices, `list_parent_indices`,
  `list_slice` with six start/stop/step triples, `struct_field` by name and `child` by position,
  `map_lookup` for three keys × first/last/all, and `binary_join` with a scalar and a per-row separator.
- **Regular expressions** twelve patterns × match/count/find, case-insensitive matching, five
  replacements, seven splits, three named-group extractions and their byte spans, and twelve LIKE
  patterns — against ICU-vs-RE2, with only the two documented syntax differences.
- **The Arrow string surface** all thirteen character-class predicates including the Unicode ones,
  `ascii_title`/`utf8_capitalize`/`utf8_title`, `utf8_center`, `utf8_replace_slice` and its binary twin
  with negative and reversed bounds, the six trims with and without a character set, all four
  normalisation forms, and string `is_in`/`index_in` through the GPU hash table.
- **utf8 ↔ number** `to_strings` exact on all eight integer types and `bool`; the float text's *value*
  bit-exact; `cast("string")` identical to `to_strings()` on all ten; and `parse` back from Arrow's own
  rendering on all ten plus `bool`.
- **Temporal** every calendar and clock field on all fourteen date- or time-carrying columns; `week` with
  all eight `WeekOptions` combinations and `day_of_week` with six; `iso_calendar` and `year_month_day`
  field by field; `floor`/`ceil`/`round_temporal` for every unit at multiples 1, 2, 3 and 7; all nine
  `*_between` differences plus `weeks_between` at every `week_start`; `months_between`,
  `month_day_nano_interval_between` and the two narrow interval layouts; `add_duration`,
  `subtract_temporal`, `add_interval`, `cast_unit` between all four resolutions, `strftime` and
  `strptime`; `is_dst`, `local_timestamp` and `assume_timezone` in four zones across both DST modes.
- **Window and ranking** `row_number`, `rank` and `dense_rank` against `pc.rank`'s three tiebreakers;
  `percent_rank`, `cume_dist`, `shift` (five offsets, with and without a fill), `pairwise_diff` at three
  periods, `cumulative_prod`/`cumulative_mean`, the rolling sum, mean, min and max over five windows, and
  `lexsort_indices` against `pc.sort_indices` on a two-column table in all four directions.
- **Statistical aggregates** `product`, `variance`/`stddev` at ddof 0 and 1, `quantile` at seven
  quantiles with linear interpolation, `median`, `mode` with its count, `count_distinct`, `first`,
  `last`, `index`, `min_max`, `any` and `all` — and the run-end encode/decode round trip.
- **The remaining type rows** the float16 casts both ways and computation through float32,
  `fixed_size_binary` equality against an array and a scalar, `hash64`'s determinism, canonicalisation
  and injectivity, the dictionary decode, the `null` column, the interval fields and the extension-type
  metadata round trip.
- **Trigonometry** all twelve functions plus `atan2` on both float types, the seven `_checked` twins
  agreeing in-domain and *both raising* out of it, `xor`/`and_not`/`and_not_kleene`,
  `is_nan`/`is_inf`/`is_finite` on integers as well as floats, `fill_null_forward`/`_backward`,
  `case_when`, `choose`, `replace_with_mask`, `indices_nonzero` and `coalesce`.

## What each family compares

One row per family: the operations in it, the columns it runs on, the oracle, and the tolerance. "exact"
means the values *and* the Arrow type have to match.

| Family | Operations | Types | Oracle | Tolerance |
|---|---|---|---|---|
| Interop | `roundtrip`, `roundtrip_ext`, `filter_ext`, `take_ext`, `slice_ext`, `null_column` | all 45 | the input itself; `Array.filter`/`take`/`slice` | exact |
| Reductions | `sum`, `min_max`, `mean`, `product`, `variance_and_stddev`, `quantile`, `mode_and_count_distinct`, `first_last_index_min_max`, `any_all` | the 10 numeric, `bool` | `pc.sum`/`min`/`max`/`mean`/`product`/`variance`/`stddev`/`quantile(linear)`/`mode`/`count_distinct`/`first`/`last`/`index`/`min_max`/`any`/`all`, options spelled out | exact for the positional ones; see *Tolerances* for the rest |
| Element-wise | `compare_*`, `arith_*`, `cast`, `rounding`, `abs`, `negate`, `sign`, `bitwise_*`, `shift_*`, `modulo`, `power`, `element_wise_min_max`, `float_class` | the 10 numeric | the matching `pc.*`, unchecked variants | bit-exact (floats), exact (integers) |
| Transcendental | `sqrt`, `exp`, `logarithm`, `trig`, `trig_checked` | `float32`, `float64` | `pc.sqrt`/`exp`/`ln`/`log10`/`log2`/`sin`…`atanh`/`atan2` and the `_checked` twins | relative, per *Tolerances*; the special values pinned exactly |
| Selection & sort | `filter`, `filter_where`, `take`, `slice`, `sort`, `argsort`, `top_k`, `lexsort`, `indices_nonzero`, `drop_null` | the 10 numeric, `bool`, `utf8` | `Array.filter`/`take`/`slice`, `pc.array_sort_indices`, `pc.sort_indices`, `pc.indices_nonzero`, `pc.drop_null` | exact |
| Structural & conditional | `is_null`, `is_valid`, `fill_null`, `fill_null_direction`, `if_else`, `case_when`, `choose`, `replace_with_mask`, `coalesce`, `is_in`, `index_in`, `kleene`, `logical_extras` | the 10 numeric, `bool`, `utf8` | the matching `pc.*`; `skip_nulls=True` for the set lookups | exact |
| Cumulative & window | `cumulative_sum`/`_min`/`_max`/`_prod`/`_mean`, `ranking`, `percent_rank_and_cume_dist`, `shift`, `pairwise_diff`, `rolling_sum`, `rolling_mean`, `rolling_min_max` | the 10 numeric | `pc.cumulative_*` with `skip_nulls=True` and an explicit `start`, `pc.rank` with a tiebreaker, `pc.pairwise_diff`; a reference in the harness for the six Arrow has no function for | exact for the integer and min/max forms; the reductions' bound for the float sums |
| Strings | `str_length`, `str_match`, `str_equals_array`, `str_hash32`, the ASCII transforms and predicates, `replace`, `repeat`, `slice_codeunits`, `pad`, `substring_search`, `str_concat`, `string_predicates`, `string_case_transforms`, `string_pad_and_slice`, `string_trim`, `string_normalize`, `string_set_lookup` | `utf8` | the matching `pc.*`; `unicodedata` for `utf8_normalize` | exact |
| Regex & LIKE | `regex_match`, `regex_replace`, `regex_split`, `split_whitespace`, `regex_extract`, `regex_extract_span`, `match_like` | `utf8` | `pc.match_substring_regex`, `pc.count_substring_regex`, `pc.find_substring_regex`, `pc.replace_substring_regex`, `pc.split_pattern[_regex]`, `pc.utf8_split_whitespace`, `pc.extract_regex[_span]`, `pc.match_like` | exact |
| utf8 ↔ number | `to_strings`, `to_strings_text`, `to_strings_value`, `cast_to_string`, `parse_numbers` | the 10 numeric, `bool` | `Array.cast(utf8)` and back; the text read with Python's `float` for the value check | exact |
| Decimal | `decimal_compare`, `decimal_arith`, `decimal_unary`, `decimal_round`, `decimal_to_float64`, `decimal_reduce`, `decimal_widen_narrow` | the 5 decimal columns | `pc.equal`…, `pc.add`/`subtract`/`multiply` (in decimal256 where 38 digits overflow Arrow's own precision), `pc.negate`/`abs`/`sign`, `pc.round`, `Array.cast`, `pc.sum`/`min`/`max` | exact |
| Nested | `list_length`, `list_flatten`, `list_element`, `list_parent_indices`, `list_slice`, `struct_field`, `map_lookup`, `binary_join` | `list<int64>`, `list<utf8>`, `struct`, `map` | `pc.list_value_length`/`list_flatten`/`list_element`/`list_parent_indices`/`list_slice`/`struct_field`/`map_lookup`/`binary_join` | exact |
| Temporal fields | `temporal_calendar`, `temporal_clock`, `temporal_clock_on_a_date`, `temporal_week_options`, `temporal_struct`, `temporal_field_types`, `temporal_utc_semantics` | the 8 timestamp, 2 date, 4 time columns | `pc.year`…`pc.subsecond`, `pc.week` with all eight option combinations, `pc.day_of_week` with six | exact, after the documented int32/int64 width cast |
| Temporal rounding | `temporal_round`, `temporal_round_finer`, `temporal_round_calendar`, `temporal_ceil_calendar`, `temporal_round_unaligned`, `temporal_round_duration` | the 8 timestamp, 2 date, 4 time, 4 duration columns | `pc.floor_temporal`/`ceil_temporal`/`round_temporal`, every unit at multiples 1, 2, 3 and 7 | exact |
| Temporal arithmetic | `temporal_between`, `temporal_between_clock`, `weeks_between`, `months_between`, `interval_between`, `interval_layouts`, `add_duration`, `subtract_temporal`, `add_interval`, `cast_unit`, `strftime`, `strftime_seconds`, `strptime`, `strptime_roundtrip` | the temporal columns, `utf8` | the nine `pc.*_between`, `pc.weeks_between` with `week_start` 1–7, `pc.month_day_nano_interval_between`, `pc.add`, `pc.subtract`, `Array.cast`, `pc.strftime`, `pc.strptime(error_is_null=True)` | exact |
| Timezones | `temporal_timezone`, `assume_timezone` | the 4 zoned and 4 naive timestamp columns | `pc.is_dst`, `pc.local_timestamp`, `pc.assume_timezone` in four zones x two ambiguity modes | exact |
| The rest of the type matrix | `float16_casts`, `float16_compute`, `float16_reduce`, `fixed_binary_compare`, `hash64`, `dictionary_ops`, `dictionary_encode`, `dictionary_decode`, `run_end`, `extension_type`, `nulls_constructor` | `float16`, `fixed_size_binary`, `dict<utf8>`, run-end, `null`, the numerics | `Array.cast`, `pc.equal`/`not_equal`, `pc.run_end_encode`/`decode`; the hash by its properties | exact |
| Group-by | `group_by_count`, `_count_values`, `_sum`, `_min`, `_max`, `_mean` | the 10 numeric | `pa.Table.group_by(...).aggregate(...)` | exact for the integers, the reductions' bound for the floats |

## Reading the report

`differential_report.py` prints one row per operation and one column per type — in blocks of ten types,
because the matrix is 45 columns wide, and a block leaves out the operations that do not apply to any of
its types. A cell says `ok N` when all N datasets agree, `known n/N` when n of them hit one of the open
findings above, `NEW n/N` for a divergence that is *not* in this document, `skip N` where the kernel does
not exist for that type, and `-` where the operation does not apply.

**It exits on the `unclassified` count, not on the failure count.** A documented divergence has a FINDINGS
entry, a strict-xfail reproduction and a paragraph above it: it is a decision on the record, and holding CI
red on it would only teach everyone to ignore the gate. An unclassified divergence is one nobody has looked
at, and that is what fails the build -- 0 at 0.1.0, and it should stay there. The totals line prints both
numbers, and the last line reads `PASS (n documented divergence(s))` so the count cannot drift unnoticed.

`--ops` and `--types` narrow the matrix while chasing one cell:

    PYTHONPATH=python python python/tests/differential_report.py --ops argsort,top_k --types int32
