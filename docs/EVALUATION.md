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
| Cases in the default matrix | 6,318, about 7 s on an M4 Max; 7,020 in about 140 s with `DIFF_LARGE=1` |
| Result at 0.1.0 | 5,903 pass, 145 fail across 5 open findings, 270 skip (kernels not implemented), 0 unclassified |

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
  kernel is required to ignore. That choice is what surfaced finding 1.
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
`GroupBy` without one, so the harness cannot silently fall behind the library. Kernels that are on the
roadmap but absent at 0.1.0 — `is_null`, `is_valid`, `is_nan`, `abs`, `negate`, `sign`, `upper`, `lower`,
`trim`, `reverse`, `fill_null`, `if_else`, `is_in`, `cumulative_sum`/`_prod`/`_max`/`_min`,
`bitwise_and`/`_or`/`_xor`/`_not`, `shift_left`, `shift_right` — already have their oracle written and
register themselves the moment the method appears on `MetalArray`. Until then the report lists them as
absent.

**Comparison strictness.** Results are compared as `pyarrow.Array`s, not as Python lists, so the *type* is
part of the comparison: a kernel that returns the right numbers as `int64` instead of `uint64` fails, which
is how finding 5 turned up. Integers, booleans, strings and index vectors are compared exactly. Float
arithmetic (`+ - * /`, scalar and array, Float32 and Float64) is compared *bit-exact* — `Array.equals` is
deliberately not used for float types, because Arrow calls `-0.0` and `0.0` equal and `NaN` and `NaN`
unequal, and this harness needs the opposite of both. Reductions get a tolerance, because the two engines
legitimately accumulate in a different order: a relative 1e-6 for Float32 and 1e-12 for Float64, plus the
standard roundoff bound `8·eps·sqrt(n)·Σ|x|`, without which a 100k-row Float32 sum cannot be compared at
all (`sqrt(100000)·2^-24 ≈ 2e-5` on its own).

**Not-implemented versus wrong.** An `ArrowMetalError` whose message matches a known gap ("Unsupported
Arrow type", "group-by min/max needs a 32-bit or narrower type", …) is a *skip*; any other error, or a
wrong answer, is a *failure*. The 270 skips in the default run are all group-by combinations the kernels do
not cover — `min`/`max` on 64-bit values, `mean` on Float32, and any aggregate over Float64 — the same set
`test_arrowmetal.py` pins as expected errors.

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

Because `min`/`max` skip NaN in both engines but disagree on what is left when *every* value is NaN (see
finding 6), the matrix compares `min`/`max`/`mean` on NaN-free input, and the NaN behaviour is pinned in
tests of its own.

## Open findings

Six divergences the harness found that are not deliberate; the first five account for all 145 failing
cases. **None has been fixed** — this document and the tests are the record. Each has an
`xfail(strict=True)` reproduction in `test_differential.py`, so the suite turns red the moment a kernel
starts behaving, and the first five have an entry in `FINDINGS` in the same file, so the matrix groups the
affected cells under the finding instead of burying them.

### 1. `argsort` and `top_k` order null indices by the bytes under the validity bitmap

*116 of the 145 failing cases. Affects every numeric type, every dataset with nulls.*

`am_argsort` documents a stable sort with nulls last. The values are sorted correctly and the nulls do land
last, but their *indices* come back ordered by whatever is in the values buffer at those positions, rather
than in input order. Two arrays Arrow considers equal therefore argsort differently:

```python
mask = np.array([True, True, True, False])
loud  = pa.array(np.array([30, 20, 10, 7], np.int32), mask=mask, type=pa.int32())
quiet = pa.array(np.array([ 0,  0,  0, 7], np.int32), mask=mask, type=pa.int32())
loud.equals(quiet)                       # True -- both are [None, None, None, 7]

am.array(loud).argsort().to_arrow()      # [3, 2, 1, 0]     <- ordered by 30 > 20 > 10
am.array(quiet).argsort().to_arrow()     # [3, 0, 1, 2]
pc.array_sort_indices(loud)              # [3, 0, 1, 2]     <- stable: null indices in input order
```

`top_k(k, largest=False)` inherits it (`[3, 2, 1, 0]` where Arrow gives `[3, 0, 1, 2]`). `sort()` is not
affected — every element it returns for those positions is null either way. Nothing here is *invalid*: the
result is always a permutation, the nulls are always last, and `take(argsort())` still reproduces
`sort()`. It is the stability guarantee that does not hold, and the index order becomes an unpredictable
function of buffer contents that Arrow says are meaningless. Anything that argsorts one column to reorder
another gets a different, arbitrary permutation of its null rows on each input.

Likely cause: the radix sort keys nulls on their raw payload with a sentinel high bit, instead of on the
row index. Reproductions: `test_argsort_null_block_is_stable`, `test_top_k_null_block_is_stable`. The
properties that *do* hold are asserted, not assumed, in
`test_argsort_is_a_permutation_with_nulls_last_and_values_in_order`.

### 2. Float32 kernels flush subnormals to zero

*11 failing cases: `arith_scalar/float32`, `arith_array/float32`, `compare_array/float32`, `special` flavor.*

Metal's default fast-math mode is flush-to-zero and denormals-are-zero. The Float32 arithmetic and
comparison kernels inherit it, so a subnormal result becomes `0.0` and a subnormal operand compares as
zero — while import/export round-trips the same values perfectly, which makes the loss look like a data
bug rather than a math-mode one.

```python
tiny = 1.1754943508222875e-38                      # smallest normal float32
a = pa.array([tiny, 1.0], pa.float32())
am.array(a).arith("*", 0.5).to_arrow()             # [0.0, 0.5]
pc.multiply(a, pa.scalar(0.5, pa.float32()))       # [5.877471754111438e-39, 0.5]   (so does numpy)

smallest = 1.401298464324817e-45                   # smallest subnormal float32
b = pa.array([smallest], pa.float32())
am.array(b).compare(">", 0.0).to_arrow()           # [False]
pc.greater(b, pa.scalar(0.0, pa.float32()))        # [True]

# and a subnormal compares equal to -0.0:
am.array(pa.array([-0.0], pa.float32())) == am.array(b)   # [True]; pyarrow says [False]
```

Float64 is unaffected — it runs through the software IEEE-754 path and returns the subnormal, which
`test_float64_arithmetic_and_comparison_keep_subnormals` asserts. The divergence is Float32-only and
therefore looks like a missing `-fno-fast-math` (or an explicit denormal mode) on the Float32 kernels
rather than anything algorithmic.

Reproductions: `test_float32_arithmetic_keeps_subnormal_results`,
`test_float32_comparison_distinguishes_subnormals_from_zero`.

### 3. `sort` and `argsort` split the `-0.0` / `0.0` tie

*44 failing cases across `sort`, `argsort` and `top_k` on both float types, `special` flavor.*

IEEE-754 says `-0.0 == 0.0`, so Arrow's stable sort leaves them in input order. The radix sort compares bit
patterns, where `-0.0` has the sign bit set, and puts every `-0.0` before every `0.0`:

```python
a = pa.array([0.0, -0.0, 0.0, -0.0], pa.float64())
am.array(a).argsort().to_arrow()            # [1, 3, 0, 2]
pc.array_sort_indices(a)                    # [0, 1, 2, 3]
am.array(a).sort().to_arrow()               # [-0.0, -0.0, 0.0, 0.0]
a.take(pc.array_sort_indices(a))            # [ 0.0, -0.0, 0.0, -0.0]
```

Descending is wrong in the mirror direction (`[0, 2, 1, 3]` against Arrow's `[0, 1, 2, 3]`). It is the same
stability guarantee as finding 1, broken by a different mechanism, and the sorted *values* are still in
non-decreasing order under `==`. Fixing it means canonicalising `-0.0` to `0.0` in the radix key, which
costs one instruction per element in the key-building pass.

Reproduction: `test_negative_zero_is_a_sort_tie`.

### 4. `sum` and `mean` over Float32 accumulate in Float32

*2 failing cases, `special` flavor only.*

`pc.sum` over a Float32 column returns a **double** — Arrow widens the accumulator on purpose. ArrowMetal
accumulates in Float32, so a total that exceeds `FLT_MAX` becomes ±inf, and once an inf of each sign is in
play the result is NaN where Arrow's is finite or infinite:

```python
big = 3.4028234663852886e+38                       # FLT_MAX
a = pa.array([big, big], pa.float32())
am.array(a).sum()                                  # inf
pc.sum(a).as_py()                                  # 6.805646932770577e+38
```

On the generated `special` dataset that turns into `sum() -> nan` where Arrow gives `inf`. Ordinary data is
unaffected beyond the roundoff the harness's error bound already allows; the divergence needs values within
a factor of two of `FLT_MAX`. Grouped `sum` over Float32 has the same accumulator but no failing case in the
default matrix. Reproduction: `test_float32_sum_does_not_overflow_before_arrow_does`.

### 5. Grouped `sum` over UInt64 values returns Int64

*8 failing cases: `group_by_sum/uint64` and `group_by_mean/uint64`, `special` flavor.*

`am_group_by` with `agg = sum` reports an `int64` array regardless of the value type, so a UInt64 group
total above `2^63` comes back negative. The bits are right; the type is not.

```python
keys   = pa.array([0, 0], pa.int32())
values = pa.array([2**63, 2**63 - 5], pa.uint64())
am.array(keys).group_by(1).sum(am.array(values)).type       # int64  (should be uint64)
am.array(keys).group_by(1).sum(am.array(values)).to_arrow() # [-5]
pa.table({"k": keys, "v": values}).group_by("k").aggregate([("v", "sum")])
                                                            # [18446744073709551611]
```

`group_by(...).mean(...)` divides the same signed total, so it is wrong by the same reinterpretation
(`-0.71` where Arrow gives `2.6e18`). The scalar `MetalArray.sum()` does *not* have this problem: it reads
`out_kind == 1` and masks back to unsigned. Reproduction:
`test_group_by_sum_over_uint64_stays_unsigned`.

### 6. `min` and `max` return null for an all-NaN array

*No failing matrix cases — the matrix compares min/max on NaN-free input. Pinned as a divergence.*

ArrowMetal treats NaN as missing throughout, so with nothing but NaN left it reports null, even though the
array's `null_count` is 0. pyarrow returns NaN.

```python
a = pa.array([math.nan, math.nan], pa.float64())
a.null_count                       # 0
am.array(a).min()                  # None
pc.min(a).as_py()                  # nan
```

Defensible as a design choice, and the more consistent one given that ArrowMetal skips NaN elsewhere — but
it is not written down anywhere, and `min()` returning `None` for an array with no nulls will surprise a
caller that branches on `null_count`. It belongs in `include/arrowmetal.h` next to the sort's NaN note
either way. Pinned by `test_all_nan_min_max_is_null_in_arrowmetal_and_nan_in_pyarrow`.

## What passed

Everything else, on all 27 datasets per cell:

- **Import/export round trip** for all 12 types, including every sliced offset, all-null and empty arrays,
  4096-byte strings and multi-byte UTF-8 — type, length and `null_count` preserved.
- **Reductions** `sum`, `min`, `max`, `mean` on all 10 numeric types, including UInt64 totals above `2^63`
  and Int64 totals that wrap.
- **Comparisons** all six operators, scalar and array, on all numeric types — including UInt64 against
  `2^63`, where a signed comparison would flip.
- **Arithmetic** `+ - * /`, scalar and array, bit-exact on both float types (outside finding 2) and exact
  under wrapping on all eight integer types.
- **Casts** all 100 numeric source/target pairs, matching `Array.cast(safe=False)` including narrowing
  wraps and float→int truncation toward zero.
- **Boolean** `and`, `or`, `not` with nulls.
- **Selection** `filter` (null mask entries drop the row, as in Arrow), `filter_where` for all six
  predicates, `take` with repeated and null indices, `slice`.
- **`sort`** ascending and descending on all numeric types, at every size, outside finding 3.
- **Strings** `byte_length`, `char_length`, `starts_with`, `ends_with`, `str_contains`, `str_equals`
  (scalar and array) against 11 patterns including the empty string, a multi-byte prefix, a tab and a
  300-byte needle; `dictionary_encode` round trip and first-seen dictionary order; `hash32` null
  propagation and injectivity.
- **Group-by** `count`, `count_values`, `sum`, `min`, `max`, `mean` on every value type the kernels cover.

## Reading the report

`differential_report.py` prints one row per operation and one column per type. A cell says `ok N` when all
N datasets agree, `known n/N` when n of them hit one of the open findings above, `NEW n/N` for a divergence
that is *not* in this document, `skip N` where the kernel does not exist for that type, and `-` where the
operation does not apply. It exits 1 whenever anything failed, so it is usable as a gate; the useful signal
in CI is the `unclassified` count in the totals line, which is 0 at 0.1.0 and should stay there.

`--ops` and `--types` narrow the matrix while chasing one cell:

    PYTHONPATH=python python python/tests/differential_report.py --ops argsort,top_k --types int32
