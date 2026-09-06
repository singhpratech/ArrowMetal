# Apache Arrow compute functions, one row per name

<!-- Generated. Do not edit by hand: regenerate with
     PYTHONPATH=python python python/tests/function_table_report.py --page > docs/ARROW_FUNCTIONS.md -->

Every Apache Arrow v25 compute function name — the 307 of them, being the 283 in the
[C++ compute function list](https://arrow.apache.org/docs/cpp/compute.html) plus the 24
`hash_*` grouped aggregates `pyarrow` registers — with what ArrowMetal 0.1.0 does about it.

This is the by-name page. [COVERAGE.md](COVERAGE.md) is the by-family page: it groups these functions
and explains how each family works, with the Arrow **type** matrix and the interop status alongside.
Come here to answer "is `<name>` covered, and how?"; go there for "how does this family work?".

## How the table is produced

`python/arrowmetal/functions.py` holds a registry with exactly one entry per Arrow name. Each entry
carries the status, the Swift file that implements it, the ArrowMetal call that reaches it, a note,
and two executable pieces: a **call adapter** that runs the function through ArrowMetal under Arrow's
own argument and option names, and an **oracle** that produces the answer `pyarrow.compute` gives for
the same input.

`python/tests/test_functions.py` then does four things, and the third is the point:

1. asserts the registry covers every name `pyarrow.compute.list_functions()` reports, so the list
   cannot drift as Arrow grows, and invents none of its own;
2. asserts every row is well formed — a status from the vocabulary below, a note, and, for a row that
   claims to work, a call, an example input and a named source file;
3. **runs** every `gpu` / `cpu` / `partial` row through `arrowmetal.functions.call_function` and
   compares the result to `pyarrow.compute`, value for value, with a second input in a different
   Arrow type family for the rows whose claim spans several. Float comparisons use the tolerance
   recorded for that row in `arrowmetal.functions.TOLERANCE`; where an answer legitimately differs
   from Arrow's, the row carries an oracle that checks the property Arrow actually specifies, and the
   note says what the difference is;
4. asserts that a `missing` row raises rather than quietly doing something.

So a status here is a measurement, not an intention. Nothing in this table is reachable-in-principle:
if it says `gpu`, `cpu` or `partial`, a test called it this run.

## What the statuses mean

| Status | Meaning |
|---|---|
| **GPU** | A Metal kernel does the work. Host code sets up buffers and reads the answer back, nothing more. |
| **CPU** | Implemented and reachable through the ArrowMetal API, but the work happens on the host. Every row here says *why* the host is the right place — a timezone database, a Unicode table, an ICU regex, or an output that is one row wide however long the input is. |
| **Partial** | Reachable, with a stated limitation. Three different things wear this label and each note says which: (a) an option or an input type Arrow supports and this does not; (b) an answer that deliberately differs from Arrow's — group order, ascending `unique`, an int32 where Arrow returns int64; (c) an evaluation genuinely split between the GPU and the host, of which the ten `utf8_is_*` predicates are the clearest case: they answer every row on the GPU and re-decide on the CPU only the rows carrying a byte >= 0x80. |
| **Missing** | Not implemented. The note says why. |
| **Pending** | Reserved for a name landing on an unmerged branch. No row carries it today. |

Precision is recorded, not glossed. Three families of float difference exist and the note on each row
names the one that applies: a float32 evaluation of a float64 column (about 1e-7 relative — Metal has
no `double` transcendentals, so `exp`, the plain logarithms, `sqrt` and `power` widen a `float`
result); the software binary64 routines (within 5 ulp of the host libm — the trigonometric family,
`expm1`, `log1p`, `logb`, `hypot`); and the grouped moments, whose deviations are formed in float32
about a float64 mean (about 1e-5 relative).

## Regenerating this file

No Makefile, one command:

```sh
PYTHONPATH=python python python/tests/function_table_report.py --page > docs/ARROW_FUNCTIONS.md
```

Run it from the repository root, with a Python that has `pyarrow` installed and the ArrowMetal dylib
built (`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift build -c release --product ArrowMetalC`). Check the numbers first with:

```sh
PYTHONPATH=python python -m pytest python/tests/test_functions.py -q
```

## Summary by section

| Section | GPU | CPU | Partial | Missing | Pending | Rows |
|---|---:|---:|---:|---:|---:|---:|
| Aggregations | 20 | 3 | 1 | 0 | 0 | 24 |
| Arithmetic | 16 | 0 | 4 | 0 | 0 | 20 |
| Bitwise | 8 | 0 | 0 | 0 | 0 | 8 |
| Rounding | 6 | 0 | 0 | 0 | 0 | 6 |
| Logarithmic | 4 | 0 | 6 | 0 | 0 | 10 |
| Trigonometric | 20 | 0 | 0 | 0 | 0 | 20 |
| Comparisons | 8 | 0 | 0 | 0 | 0 | 8 |
| Logical | 8 | 0 | 0 | 0 | 0 | 8 |
| StringPredicates | 9 | 0 | 10 | 0 | 0 | 19 |
| StringTransforms | 10 | 2 | 9 | 0 | 0 | 21 |
| StringPadding | 6 | 0 | 1 | 0 | 0 | 7 |
| StringTrimming | 6 | 0 | 6 | 0 | 0 | 12 |
| StringSplitting | 0 | 2 | 2 | 0 | 0 | 4 |
| StringExtraction | 0 | 2 | 0 | 0 | 0 | 2 |
| StringJoining | 1 | 0 | 1 | 0 | 0 | 2 |
| StringSlicing | 1 | 0 | 0 | 1 | 0 | 2 |
| Containment | 7 | 4 | 0 | 0 | 0 | 11 |
| Categorizations | 6 | 1 | 0 | 0 | 0 | 7 |
| Selecting | 4 | 0 | 0 | 0 | 0 | 4 |
| Conversions | 6 | 0 | 0 | 0 | 0 | 6 |
| TemporalExtraction | 21 | 3 | 0 | 0 | 0 | 24 |
| TemporalDifference | 11 | 0 | 2 | 0 | 0 | 13 |
| Timezone | 0 | 2 | 0 | 0 | 0 | 2 |
| Random | 1 | 0 | 0 | 0 | 0 | 1 |
| Associative | 4 | 0 | 0 | 0 | 0 | 4 |
| Selections | 7 | 0 | 0 | 0 | 0 | 7 |
| Sorts | 8 | 0 | 2 | 0 | 0 | 10 |
| NullFilling | 3 | 0 | 0 | 0 | 0 | 3 |
| Structural | 8 | 1 | 0 | 0 | 0 | 9 |
| Pairwise | 2 | 0 | 0 | 0 | 0 | 2 |
| Cumulative | 7 | 0 | 0 | 0 | 0 | 7 |
| GroupedAggregations | 23 | 0 | 1 | 0 | 0 | 24 |
| Total | 241 | 20 | 45 | 1 | 0 | 307 |

Totals: **241 gpu**, **20 cpu**, **45 partial**, **1 missing**, **0 pending** over 307 Arrow function names.

## Every Arrow function name

| Arrow function | Section | Status | ArrowMetal call | Implementation | Notes |
|---|---|---|---|---|---|
| `all` | Aggregations | **GPU** | `all()` | `Kernels/Aggregates.swift` | One GPU pass over the values and validity bitmaps. Null-only input gives None, as Arrow does. |
| `any` | Aggregations | **GPU** | `any()` | `Kernels/Aggregates.swift` | Same pass as `all`. |
| `approximate_median` | Aggregations | **GPU** | `median()` | `Kernels/Aggregates.swift` | Exact, not approximate: a GPU sort and an interpolated read, not a sketch. Answers are therefore at least as good as Arrow's. |
| `count` | Aggregations | **CPU** | `count(mode)` | `Sources/ArrowMetal/MetalArray.swift` | O(1) metadata: the null count already rides on the column. `mode` accepts only_valid / only_null / all. |
| `count_all` | Aggregations | **CPU** | `count_all()` | `Kernels/Selection.swift` | The row count, valid or not. O(1) metadata; inside an open batch, reading it forces a sync point. |
| `count_distinct` | Aggregations | **GPU** | `count_distinct()` | `Kernels/Unique.swift` | The length of `unique()`: one GPU sort and a run scan. |
| `first` | Aggregations | **GPU** | `first()` | `Kernels/Aggregates.swift` | A GPU pass takes the atomic minimum valid index, then one host read fetches the value. |
| `first_last` | Aggregations | **GPU** | `first_last()` | `Kernels/Selection.swift` | `first()` and `last()` packaged as the one-row struct Arrow returns. `min_count` is not implemented. |
| `index` | Aggregations | **GPU** | `index(value)` | `Kernels/Aggregates.swift` | Row of the first occurrence, -1 when absent. The value crosses the C ABI as a double, so an integer above 2^53 cannot be expressed exactly. |
| `kurtosis` | Aggregations | **GPU** | `kurtosis(biased)` | `Kernels/AggregatesExtra.swift` | Excess kurtosis, biased by default as Arrow's is: two GPU passes, the same per-type deviation machinery `variance` uses, so about 1e-15 relative on a float64 column. |
| `last` | Aggregations | **GPU** | `last()` | `Kernels/Aggregates.swift` | The atomic maximum valid index, mirroring `first`. |
| `max` | Aggregations | **GPU** | `max()` | `Kernels/Reductions.swift` | Threadgroup partials, host finalise, no atomics. |
| `mean` | Aggregations | **GPU** | `mean()` | `Kernels/Reductions.swift` | The sum kernel over a valid-count, divided on the host. |
| `min` | Aggregations | **GPU** | `min()` | `Kernels/Reductions.swift` | Threadgroup partials, host finalise. |
| `min_max` | Aggregations | **GPU** | `min_max()` | `Kernels/Aggregates.swift` | One kernel produces both. Returned as a `(min, max)` tuple rather than Arrow's struct scalar. |
| `mode` | Aggregations | **GPU** | `mode()` | `Kernels/Aggregates.swift` | A GPU sort plus a run scan. Returned as `(value, count)`; only the single most common value, not Arrow's top-n list. |
| `pivot_wider` | Aggregations | **CPU** | `am.pivot_wider(...)` | `Sources/ArrowMetal/PivotWider.swift` | One host pass over the key column, then one single-row `take` per field. The output is one row wide however long the input is, so there is no parallel work worth a kernel. |
| `product` | Aggregations | **GPU** | `product()` | `Kernels/Aggregates.swift` | Integers wrap in 64 bits; a float32 product over thousands of factors reassociates. |
| `quantile` | Aggregations | **GPU** | `quantile(q)` | `Kernels/Aggregates.swift` | Exact, with linear interpolation, from a GPU sort. Arrow's other `interpolation` modes and its multi-q form are not implemented. |
| `skew` | Aggregations | **GPU** | `skew(biased)` | `Kernels/AggregatesExtra.swift` | The third standardised central moment, biased by default as Arrow's is. Same two GPU passes as `kurtosis`. |
| `stddev` | Aggregations | **GPU** | `stddev(ddof)` | `Kernels/Aggregates.swift` | Compensated squared deviations on the GPU: about 1e-7 relative for float32, 1e-15 for float64. |
| `sum` | Aggregations | **GPU** | `sum()` | `Kernels/Reductions.swift` | Integers accumulate in int64/uint64 and wrap; float64 uses the software binary64 adder. |
| `tdigest` | Aggregations | **Partial** | `tdigest(q)` | `Kernels/AggregatesExtra.swift` | A GPU sort feeding a **host** centroid merge (delta 100), so the work is mixed: this is a sketch and agrees with Arrow's to within the sketch's error rather than exactly. Returns one q as a float, not Arrow's list; `quantile()` is the exact answer and needs no sketch. |
| `variance` | Aggregations | **GPU** | `variance(ddof)` | `Kernels/Aggregates.swift` | Same pass as `stddev`. |
| `abs` | Arithmetic | **GPU** | `abs()` | `Kernels/Rounding.swift` | One thread per element. |
| `add` | Arithmetic | **GPU** | `a + b` | `Kernels/Arithmetic.swift` | Wrapping on integers, which is Arrow's unchecked `add`. |
| `divide` | Arithmetic | **GPU** | `a / b` | `Kernels/Arithmetic.swift` | Integer division by zero is undefined here rather than an error; that is `divide_checked`'s job. |
| `exp` | Arithmetic | **Partial** | `exp()` | `Kernels/Rounding.swift` | Evaluated in `float` even for a float64 column: Metal has no `double` transcendentals, so the answer is correct to about float32 precision (1e-7 relative) rather than to a float64 ulp. |
| `multiply` | Arithmetic | **GPU** | `a * b` | `Kernels/Arithmetic.swift` | Wrapping on integers. |
| `negate` | Arithmetic | **GPU** | `negate()` | `Kernels/Rounding.swift` | Wrapping on integers. |
| `power` | Arithmetic | **Partial** | `power(other)` | `Kernels/Rounding.swift` | Element-wise `pow` on **float32 only**: a float64 column raises rather than losing precision silently (Metal has no `double` transcendental to call). Cast first. |
| `sign` | Arithmetic | **GPU** | `sign()` | `Kernels/Rounding.swift` | -1 / 0 / 1. |
| `sqrt` | Arithmetic | **Partial** | `sqrt()` | `Kernels/Rounding.swift` | A negative input gives NaN, as Arrow's unchecked `sqrt` does. Evaluated in `float` for float64 columns too, so the last digits differ from a float64 square root. |
| `subtract` | Arithmetic | **GPU** | `a - b` | `Kernels/Arithmetic.swift` | Wrapping on integers. |
| `abs_checked` | Arithmetic | **GPU** | `abs_checked()` | `Kernels/Checked.swift` | The unchecked kernel plus a read-only check pass in the same command buffer, so a checked op costs one GPU round trip. Raises only for INT_MIN on a signed integer column. |
| `add_checked` | Arithmetic | **GPU** | `add_checked(other)` | `Kernels/Checked.swift` | Wrapping is an error rather than a result: an `ArrowMetalError` naming the Arrow message and the first offending row. Float columns never raise, as in Arrow. |
| `divide_checked` | Arithmetic | **GPU** | `divide_checked(other)` | `Kernels/Checked.swift` | Raises `divide by zero` for a zero divisor on any type, and `overflow` for INT_MIN / -1. |
| `multiply_checked` | Arithmetic | **GPU** | `multiply_checked(other)` | `Kernels/Checked.swift` | As `add_checked`. |
| `negate_checked` | Arithmetic | **GPU** | `negate_checked()` | `Kernels/Checked.swift` | Raises for INT_MIN on a signed column and — unlike pyarrow, which has no unsigned kernel at all — for every non-zero value on an unsigned one. |
| `power_checked` | Arithmetic | **GPU** | `power_checked(other)` | `Kernels/Checked.swift` | Raises for a negative integer exponent and for any repeated-squaring step that would wrap. On a float column this is the float32-evaluated `power`, so the same 1e-7 relative applies. |
| `sqrt_checked` | Arithmetic | **Partial** | `sqrt_checked()` | `Kernels/Checked.swift` | Raises `square root of negative number`; NaN, -0.0 and +inf do not raise. Bit-identical to the unchecked `sqrt`, and so evaluated in `float` for a float64 column too. |
| `subtract_checked` | Arithmetic | **GPU** | `subtract_checked(other)` | `Kernels/Checked.swift` | As `add_checked`. |
| `expm1` | Arithmetic | **GPU** | `expm1()` | `Kernels/MathExtra.swift` | exp(x) - 1, accurate for small x, through the software binary64 routine on a float64 column (within about 5 ulp of the host libm). Float columns only, as in Arrow. |
| `hypot` | Arithmetic | **GPU** | `hypot(other)` | `Kernels/MathExtra.swift` | sqrt(x^2 + y^2), scaled so a large or tiny pair neither overflows nor underflows on the way. An infinite operand gives inf even opposite a NaN, as IEEE-754 prescribes. |
| `bit_wise_and` | Bitwise | **GPU** | `bitwise_and(other)` | `Kernels/Bitwise.swift` | One thread per element. |
| `bit_wise_not` | Bitwise | **GPU** | `bitwise_not()` | `Kernels/Bitwise.swift` | One thread per element. |
| `bit_wise_or` | Bitwise | **GPU** | `bitwise_or(other)` | `Kernels/Bitwise.swift` | One thread per element. |
| `bit_wise_xor` | Bitwise | **GPU** | `bitwise_xor(other)` | `Kernels/Bitwise.swift` | One thread per element. |
| `shift_left` | Bitwise | **GPU** | `shift_left(other)` | `Kernels/Bitwise.swift` | A shift at or past the width is undefined here rather than an error. |
| `shift_right` | Bitwise | **GPU** | `shift_right(other)` | `Kernels/Bitwise.swift` | Arithmetic for signed types, logical for unsigned. |
| `shift_left_checked` | Bitwise | **GPU** | `shift_left_checked(other)` | `Kernels/Checked.swift` | Raises when the shift amount is negative or at least the precision of the type (the bit width, less one on a signed column). Bits shifted off the top are not an error, as in Arrow. |
| `shift_right_checked` | Bitwise | **GPU** | `shift_right_checked(other)` | `Kernels/Checked.swift` | The same amount check as `shift_left_checked`. |
| `ceil` | Rounding | **GPU** | `ceil()` | `Kernels/Rounding.swift` | One thread per element. |
| `floor` | Rounding | **GPU** | `floor()` | `Kernels/Rounding.swift` | One thread per element. |
| `trunc` | Rounding | **GPU** | `trunc()` | `Kernels/Rounding.swift` | One thread per element. |
| `round` | Rounding | **GPU** | `round(ndigits, mode)` | `Kernels/MathExtra.swift` | All ten Arrow round modes and any `ndigits`, evaluated as round_int(x * 10^ndigits) / 10^ndigits. `round()` with no argument keeps its historical meaning (halves away from zero); passing either option selects Arrow's kernel, whose defaults are ndigits=0 and half_to_even. |
| `round_binary` | Rounding | **GPU** | `round_binary(ndigits, mode)` | `Kernels/MathExtra.swift` | `round` with one `ndigits` per row, from an int32 column. Null wherever either column is. |
| `round_to_multiple` | Rounding | **GPU** | `round_to_multiple(multiple, mode)` | `Kernels/MathExtra.swift` | round_int(x / multiple) * multiple, for any positive scalar multiple and any Arrow round mode. |
| `ln` | Logarithmic | **Partial** | `ln()` | `Kernels/Rounding.swift` | Metal's `log`, evaluated in `float` even for a float64 column, so the answer is correct to about float32 precision (1e-7 relative) rather than to a float64 ulp. |
| `log10` | Logarithmic | **Partial** | `log10()` | `Kernels/Rounding.swift` | Metal's `log10`, evaluated in `float` even for a float64 column, so the answer is correct to about float32 precision (1e-7 relative) rather than to a float64 ulp. |
| `log2` | Logarithmic | **Partial** | `log2()` | `Kernels/Rounding.swift` | Metal's `log2`, evaluated in `float` even for a float64 column, so the answer is correct to about float32 precision (1e-7 relative) rather than to a float64 ulp. |
| `ln_checked` | Logarithmic | **Partial** | `ln_checked()` | `Kernels/Checked.swift` | Raises `logarithm of zero` / `logarithm of negative number`. Bit-identical to the unchecked `ln`, so a float64 column is still evaluated in `float` (about 1e-7 relative). |
| `log10_checked` | Logarithmic | **Partial** | `log10_checked()` | `Kernels/Checked.swift` | Same domain check and same float32 evaluation as `ln_checked`. |
| `log2_checked` | Logarithmic | **Partial** | `log2_checked()` | `Kernels/Checked.swift` | Same domain check and same float32 evaluation as `ln_checked`. |
| `log1p` | Logarithmic | **GPU** | `log1p()` | `Kernels/MathExtra.swift` | ln(1 + x), accurate for small x, through the software binary64 routine — full float64 precision, unlike the plain `ln`. x == -1 gives -inf and x < -1 gives NaN. |
| `log1p_checked` | Logarithmic | **GPU** | `log1p_checked()` | `Kernels/Checked.swift` | As `log1p`, raising at the domain boundary: -1 gives `logarithm of zero` and anything below it `logarithm of negative number`. |
| `logb` | Logarithmic | **GPU** | `logb(base)` | `Kernels/MathExtra.swift` | ln(x) / ln(base) in software binary64, with a scalar base or a column of bases. |
| `logb_checked` | Logarithmic | **GPU** | `logb_checked(base)` | `Kernels/Checked.swift` | As `logb`, raising when the value or the base is zero or negative. |
| `acos` | Trigonometric | **GPU** | `acos()` | `Kernels/Trig.swift` | Needs \|x\| <= 1; outside that the unchecked form gives NaN. |
| `acos_checked` | Trigonometric | **GPU** | `acos_checked()` | `Kernels/Trig.swift` | As `acos`, raising for \|x\| > 1 on a non-null row. NaN never raises. |
| `acosh` | Trigonometric | **GPU** | `acosh()` | `Kernels/Trig.swift` | Needs x >= 1. |
| `acosh_checked` | Trigonometric | **GPU** | `acosh_checked()` | `Kernels/Trig.swift` | As `acosh`, raising for x < 1. |
| `asin` | Trigonometric | **GPU** | `asin()` | `Kernels/Trig.swift` | Needs \|x\| <= 1. |
| `asin_checked` | Trigonometric | **GPU** | `asin_checked()` | `Kernels/Trig.swift` | As `asin`, raising for \|x\| > 1. |
| `asinh` | Trigonometric | **GPU** | `asinh()` | `Kernels/Trig.swift` | Defined on the whole real line, so Arrow publishes no checked twin. |
| `atan` | Trigonometric | **GPU** | `atan()` | `Kernels/Trig.swift` | Whole real line. |
| `atan2` | Trigonometric | **GPU** | `atan2(other)` | `Kernels/Trig.swift` | The angle of (x, y) in [-pi, pi], following the C99 special-value table including the four +/-0 and four +/-infinity cases. A scalar second argument is broadcast. |
| `atanh` | Trigonometric | **GPU** | `atanh()` | `Kernels/Trig.swift` | Needs \|x\| < 1. |
| `atanh_checked` | Trigonometric | **GPU** | `atanh_checked()` | `Kernels/Trig.swift` | As `atanh`, raising for \|x\| >= 1. |
| `cos` | Trigonometric | **GPU** | `cos()` | `Kernels/Trig.swift` | Whole real line. |
| `cos_checked` | Trigonometric | **GPU** | `cos_checked()` | `Kernels/Trig.swift` | As `cos`, rejecting +/-infinity. |
| `cosh` | Trigonometric | **GPU** | `cosh()` | `Kernels/Trig.swift` | Written out from a well-conditioned identity rather than Metal's own. |
| `sin` | Trigonometric | **GPU** | `sin()` | `Kernels/Trig.swift` | Whole real line. |
| `sin_checked` | Trigonometric | **GPU** | `sin_checked()` | `Kernels/Trig.swift` | As `sin`, rejecting +/-infinity. |
| `sinh` | Trigonometric | **GPU** | `sinh()` | `Kernels/Trig.swift` | As `cosh`. |
| `tan` | Trigonometric | **GPU** | `tan()` | `Kernels/Trig.swift` | Whole real line. |
| `tan_checked` | Trigonometric | **GPU** | `tan_checked()` | `Kernels/Trig.swift` | As `tan`, rejecting +/-infinity. |
| `tanh` | Trigonometric | **GPU** | `tanh()` | `Kernels/Trig.swift` | As `cosh`. |
| `equal` | Comparisons | **GPU** | `compare('==', other)` | `Kernels/Compare.swift` | Bit-exact for float32 through an order-preserving integer key; validity bitmaps are ANDed on the GPU. |
| `greater` | Comparisons | **GPU** | `compare('>', other)` | `Kernels/Compare.swift` | One thread per element. |
| `greater_equal` | Comparisons | **GPU** | `compare('>=', other)` | `Kernels/Compare.swift` | One thread per element. |
| `less` | Comparisons | **GPU** | `compare('<', other)` | `Kernels/Compare.swift` | One thread per element. |
| `less_equal` | Comparisons | **GPU** | `compare('<=', other)` | `Kernels/Compare.swift` | One thread per element. |
| `not_equal` | Comparisons | **GPU** | `compare('!=', other)` | `Kernels/Compare.swift` | One thread per element. |
| `max_element_wise` | Comparisons | **GPU** | `max_element_wise(other)` | `Kernels/Rounding.swift` | Arrow's default `skip_nulls=True` is what this does; `skip_nulls=False` is not implemented. |
| `min_element_wise` | Comparisons | **GPU** | `min_element_wise(other)` | `Kernels/Rounding.swift` | Same null rule as `max_element_wise`. |
| `and_` | Logical | **GPU** | `a & b` | `Kernels/Compare.swift` | Bitmap AND on the GPU, null-propagating (Arrow's `and`, not `and_kleene`). |
| `or_` | Logical | **GPU** | `a \| b` | `Kernels/Compare.swift` | Bitmap OR, null-propagating. |
| `invert` | Logical | **GPU** | `~a` | `Kernels/Compare.swift` | Validity is shared zero-copy with the input. |
| `and_kleene` | Logical | **GPU** | `and_kleene(other)` | `Kernels/Structural.swift` | Three-valued AND: false wins over null. |
| `or_kleene` | Logical | **GPU** | `or_kleene(other)` | `Kernels/Structural.swift` | Three-valued OR: true wins over null. |
| `and_not` | Logical | **GPU** | `and_not(other)` | `Kernels/LogicalExtra.swift` | `a AND NOT b`, word-wise over the packed bitmaps, one thread per 32-bit output word. Nulls propagate. |
| `and_not_kleene` | Logical | **GPU** | `and_not_kleene(other)` | `Kernels/LogicalExtra.swift` | Three-valued `a AND NOT b`: a valid false on the left or a valid true on the right gives false even when the other side is null. |
| `xor` | Logical | **GPU** | `a ^ b` | `Kernels/LogicalExtra.swift` | Word-wise bitmap XOR; the output validity is the AND of the inputs'. |
| `ascii_is_alnum` | StringPredicates | **GPU** | `is_alnum()` | `Kernels/StringTransforms.swift` | One thread per row over the bytes. An empty string is false, as in Arrow. |
| `ascii_is_alpha` | StringPredicates | **GPU** | `is_alpha()` | `Kernels/StringTransforms.swift` | One thread per row. |
| `ascii_is_decimal` | StringPredicates | **GPU** | `is_digit()` | `Kernels/StringTransforms.swift` | One thread per row; `0`-`9` only, which is exactly `ascii_is_decimal`. |
| `ascii_is_lower` | StringPredicates | **GPU** | `is_lower()` | `Kernels/StringTransforms.swift` | Needs at least one cased character and none of the opposite case; non-ASCII bytes count as uncased. |
| `ascii_is_space` | StringPredicates | **GPU** | `is_space()` | `Kernels/StringTransforms.swift` | Space and `\t`-`\r`. |
| `ascii_is_upper` | StringPredicates | **GPU** | `is_upper()` | `Kernels/StringTransforms.swift` | Mirror of `ascii_is_lower`. |
| `ascii_is_printable` | StringPredicates | **GPU** | `ascii_is_printable()` | `Kernels/StringExtra.swift` | Every byte in 0x20-0x7E. The empty string is true, and every other predicate here is false on it. One thread per row. |
| `ascii_is_title` | StringPredicates | **GPU** | `ascii_is_title()` | `Kernels/StringExtra.swift` | Byte-wise title case over runs of ASCII letters; needs at least one letter. |
| `string_is_ascii` | StringPredicates | **GPU** | `string_is_ascii()` | `Kernels/StringExtra.swift` | Every byte < 0x80. The empty string is true. |
| `utf8_is_alnum` | StringPredicates | **Partial** | `utf8_is_alnum()` | `Kernels/StringExtra.swift` | Non-empty and every code point a letter or a number. GPU for the rows whose bytes are all < 0x80; rows with a byte >= 0x80 are re-decided on the host, sharded over 4096-row chunks. |
| `utf8_is_alpha` | StringPredicates | **Partial** | `utf8_is_alpha()` | `Kernels/StringExtra.swift` | Non-empty and every code point in an L* category. Same GPU/host split as `utf8_is_alnum`. |
| `utf8_is_decimal` | StringPredicates | **Partial** | `utf8_is_decimal()` | `Kernels/StringExtra.swift` | Non-empty and every code point category Nd. Same GPU/host split. |
| `utf8_is_digit` | StringPredicates | **Partial** | `utf8_is_digit()` | `Kernels/StringExtra.swift` | Non-empty and every code point category Nd or No. Same GPU/host split. |
| `utf8_is_lower` | StringPredicates | **Partial** | `utf8_is_lower()` | `Kernels/StringExtra.swift` | At least one cased code point and no upper-case one. Same GPU/host split. |
| `utf8_is_numeric` | StringPredicates | **Partial** | `utf8_is_numeric()` | `Kernels/StringExtra.swift` | Non-empty and every code point category Nd, Nl or No. Same GPU/host split. |
| `utf8_is_printable` | StringPredicates | **Partial** | `utf8_is_printable()` | `Kernels/StringExtra.swift` | No Cc/Cf/Cs/Co/Cn/Zs/Zl/Zp code point, except U+0020; the empty string is true. Same GPU/host split. |
| `utf8_is_space` | StringPredicates | **Partial** | `utf8_is_space()` | `Kernels/StringExtra.swift` | Non-empty and every code point Unicode whitespace (U+200B deliberately is not). Same GPU/host split. |
| `utf8_is_title` | StringPredicates | **Partial** | `utf8_is_title()` | `Kernels/StringExtra.swift` | At least one cased code point, in title case. Same GPU/host split. |
| `utf8_is_upper` | StringPredicates | **Partial** | `utf8_is_upper()` | `Kernels/StringExtra.swift` | At least one cased code point and no lower-case one. Same GPU/host split. |
| `ascii_capitalize` | StringTransforms | **GPU** | `capitalize()` | `Kernels/StringTransforms.swift` | First byte upper-cased, the rest lower-cased, ASCII only. |
| `ascii_lower` | StringTransforms | **GPU** | `ascii_lower()` | `Kernels/StringTransforms.swift` | Byte-wise `A`-`Z` to `a`-`z`; every other byte is copied through, so the output stays valid UTF-8. |
| `ascii_upper` | StringTransforms | **GPU** | `ascii_upper()` | `Kernels/StringTransforms.swift` | Byte-wise `a`-`z` to `A`-`Z`. |
| `ascii_swapcase` | StringTransforms | **GPU** | `swapcase()` | `Kernels/StringTransforms.swift` | Byte-wise ASCII case flip. |
| `ascii_reverse` | StringTransforms | **Partial** | `str_reverse()` | `Kernels/StringTransforms.swift` | Reverses **code points**, not bytes. Identical to Arrow for ASCII input — which is what `ascii_reverse` is defined on — but Arrow's byte reversal of non-ASCII input (which produces invalid UTF-8) is not reproduced. |
| `utf8_reverse` | StringTransforms | **GPU** | `str_reverse()` | `Kernels/StringTransforms.swift` | Reverses code points, walking the UTF-8 lead bytes. |
| `binary_reverse` | StringTransforms | **Partial** | `str_reverse()` | `Kernels/StringTransforms.swift` | Same kernel as `utf8_reverse`, so it reverses code points rather than bytes; equal to Arrow only for single-byte content, which is what the check below feeds it. A `binary` column is refused, so the input has to be utf8. |
| `binary_length` | StringTransforms | **Partial** | `byte_length()` | `Sources/ArrowMetal/MetalStringArray.swift` | The offsets difference; no data read at all. Takes a **utf8** column only — a `binary` column is refused, where Arrow's `binary_length` accepts both. The same holds for `binary_repeat` and `binary_reverse`; `is_in` / `index_in` / `take` / `filter` / `binary_replace_slice` do accept binary. |
| `utf8_length` | StringTransforms | **GPU** | `char_length()` | `Sources/ArrowMetal/MetalStringArray.swift` | Counts non-continuation bytes. |
| `binary_repeat` | StringTransforms | **Partial** | `repeat(n)` | `Kernels/StringTransforms.swift` | Two-pass on the GPU: a length kernel, a scan into offsets, a byte kernel. Two limits: `n` is one scalar for the whole column, where Arrow also takes a per-row `num_repeats` array; and the input must be utf8, not `binary`. |
| `replace_substring` | StringTransforms | **GPU** | `replace(pattern, replacement, max_replacements)` | `Kernels/StringTransforms.swift` | Literal replacement on the GPU, two-pass. |
| `replace_substring_regex` | StringTransforms | **CPU** | `replace_substring_regex(pattern, replacement)` | `Kernels/Regex.swift` | The regex engine is a backtracker on the host; a literal pattern is routed to the GPU kernel instead. |
| `utf8_lower` | StringTransforms | **Partial** | `lower()` | `Kernels/StringTransforms.swift` | Simple 1:1 case mapping over Basic Latin, Latin-1 Supplement and Latin Extended-A. Everything above U+017F passes through unchanged; there is no Unicode case table on the GPU. |
| `utf8_upper` | StringTransforms | **Partial** | `upper()` | `Kernels/StringTransforms.swift` | Same coverage as `utf8_lower`. U+00DF, whose full uppercase is `SS`, passes through unchanged. |
| `utf8_swapcase` | StringTransforms | **Partial** | `utf8_swapcase()` | `Kernels/StringExtra.swift` | Same coverage as `utf8_upper` / `utf8_lower`: Basic Latin, Latin-1 Supplement and Latin Extended-A, including the length-changing pairs. U+00DF (which Arrow swaps to U+1E9E) and every code point above U+017F pass through unchanged rather than being mangled. |
| `ascii_title` | StringTransforms | **GPU** | `ascii_title()` | `Kernels/StringExtra.swift` | Byte-wise: the first ASCII letter of every run of ASCII letters is upper-cased and the rest lower-cased. Bytes >= 0x80 are copied through and end a word, so `"ünïcödé"` becomes `"üNïCöDé"` — exactly what Arrow does. |
| `utf8_capitalize` | StringTransforms | **Partial** | `utf8_capitalize()` | `Kernels/StringExtra.swift` | First code point upper-cased, every later one lower-cased, through Unicode's **simple** 1:1 mappings as utf8proc uses. GPU (the `ascii_capitalize` kernel) when the whole column is ASCII and host-side otherwise — an output length that depends on a Unicode table cannot be computed in the GPU length pass. |
| `utf8_title` | StringTransforms | **Partial** | `utf8_title()` | `Kernels/StringExtra.swift` | The first cased code point of every word upper-cased and the rest lower-cased, a word being a maximal run of cased code points. Same per-column GPU/host split as `utf8_capitalize`. |
| `utf8_normalize` | StringTransforms | **CPU** | `utf8_normalize(form)` | `Kernels/StringExtra.swift` | NFC / NFKC / NFD / NFKD through Foundation, on the host: a full Unicode normalisation table in MSL buys nothing over it. Deliberate difference: pyarrow 25 never *composes*, so its NFC equals its NFD and its NFKC equals its NFKD; this follows the Unicode standard and agrees with Python's `unicodedata.normalize` on all four forms. The check below uses NFD, where the two agree. |
| `utf8_replace_slice` | StringTransforms | **GPU** | `utf8_replace_slice(start, stop, replacement)` | `Kernels/StringExtra.swift` | Replaces the code points in [start, stop). Negative indices count from the end, both ends clamp, and a stop below start inserts without deleting. The cut always lands on a code-point boundary. |
| `binary_replace_slice` | StringTransforms | **GPU** | `binary_replace_slice(start, stop, replacement)` | `Kernels/StringExtra.swift` | The same substitution indexed in **bytes**, which can split a UTF-8 sequence and is why the result is `binary` rather than `utf8`. |
| `ascii_lpad` | StringPadding | **GPU** | `pad_left(width, pad)` | `Kernels/StringTransforms.swift` | `width` counts code points and `pad` must be one character; a string already at or over `width` is returned unchanged. |
| `ascii_rpad` | StringPadding | **GPU** | `pad_right(width, pad)` | `Kernels/StringTransforms.swift` | Mirror of `ascii_lpad`. |
| `utf8_lpad` | StringPadding | **GPU** | `pad_left(width, pad)` | `Kernels/StringTransforms.swift` | The same kernel: `width` has always counted code points. |
| `utf8_rpad` | StringPadding | **GPU** | `pad_right(width, pad)` | `Kernels/StringTransforms.swift` | The same kernel as `ascii_rpad`. |
| `utf8_zero_fill` | StringPadding | **GPU** | `utf8_zero_fill(width, padding)` | `Kernels/StringExtra.swift` | Left-pads to `width` code points, inserting the padding after a leading `+` or `-`. The content need not be numeric. |
| `ascii_center` | StringPadding | **Partial** | `utf8_center(width, padding)` | `Kernels/StringExtra.swift` | The same kernel as `utf8_center`, so `width` counts **code points** where Arrow's ASCII form counts bytes — identical on ASCII input, which is what `ascii_center` is defined on. |
| `utf8_center` | StringPadding | **GPU** | `utf8_center(width, padding)` | `Kernels/StringExtra.swift` | Pads on both sides to `width` code points, the odd pad character going on the **right** (`"a"` centred in 4 is `"*a**"`). Strings already at or over `width` come back unchanged. |
| `ascii_trim_whitespace` | StringTrimming | **GPU** | `trim()` | `Kernels/StringTransforms.swift` | Space and `\t`-`\r` from both ends. |
| `ascii_ltrim_whitespace` | StringTrimming | **GPU** | `ltrim()` | `Kernels/StringTransforms.swift` | Leading ASCII whitespace. |
| `ascii_rtrim_whitespace` | StringTrimming | **GPU** | `rtrim()` | `Kernels/StringTransforms.swift` | Trailing ASCII whitespace. |
| `ascii_trim` | StringTrimming | **GPU** | `trim(characters)` | `Kernels/StringTransforms.swift` | Trims any byte in `characters` from both ends. |
| `ascii_ltrim` | StringTrimming | **GPU** | `ltrim(characters)` | `Kernels/StringTransforms.swift` | Leading bytes only. |
| `ascii_rtrim` | StringTrimming | **GPU** | `rtrim(characters)` | `Kernels/StringTransforms.swift` | Trailing bytes only. |
| `utf8_trim_whitespace` | StringTrimming | **Partial** | `utf8_trim()` | `Kernels/StringExtra.swift` | Strips the full **Unicode** whitespace class from both ends (Zs/Zl/Zp plus U+0009-U+000D, U+001C-U+001F and U+0085; U+200B deliberately is not whitespace). GPU when the column is all ASCII — that set restricted to ASCII is a ten-byte set the existing trim kernel handles — and host-side otherwise. |
| `utf8_ltrim_whitespace` | StringTrimming | **Partial** | `utf8_ltrim()` | `Kernels/StringExtra.swift` | Leading-only form of `utf8_trim_whitespace`, with the same GPU/host split. |
| `utf8_rtrim_whitespace` | StringTrimming | **Partial** | `utf8_rtrim()` | `Kernels/StringExtra.swift` | Trailing-only form, with the same GPU/host split. |
| `utf8_trim` | StringTrimming | **Partial** | `utf8_trim(characters)` | `Kernels/StringExtra.swift` | Strips leading and trailing **code points** that appear in `characters`. An ASCII set runs on the GPU (byte-wise trimming can never split a UTF-8 sequence, since every continuation byte is >= 0x80); a set with a non-ASCII character runs on the host. |
| `utf8_ltrim` | StringTrimming | **Partial** | `utf8_ltrim(characters)` | `Kernels/StringExtra.swift` | Leading-only form of `utf8_trim`, with the same GPU/host split. |
| `utf8_rtrim` | StringTrimming | **Partial** | `utf8_rtrim(characters)` | `Kernels/StringExtra.swift` | Trailing-only form, with the same GPU/host split. |
| `ascii_split_whitespace` | StringSplitting | **Partial** | `split_whitespace()` | `Kernels/Regex.swift` | Splits on runs of ASCII whitespace, on the host. Two differences from Arrow: leading and trailing whitespace produce no empty piece (Arrow emits one at each end, and one for an empty string), and the ArrowMetal call returns `(offsets, values)` rather than a list column — the registry stitches those into one. `max_splits` and `reverse` are not implemented. |
| `utf8_split_whitespace` | StringSplitting | **Partial** | `split_whitespace()` | `Kernels/Regex.swift` | The same host split, with the same dropped end pieces as `ascii_split_whitespace`, and **ASCII** whitespace only rather than the Unicode whitespace class. |
| `split_pattern` | StringSplitting | **CPU** | `split_pattern(pattern)` | `Kernels/Regex.swift` | Literal split on the host. `max_splits` and `reverse` are not implemented. |
| `split_pattern_regex` | StringSplitting | **CPU** | `split_pattern(pattern, regex=True)` | `Kernels/Regex.swift` | The host backtracking engine. `max_splits` and `reverse` are not implemented. |
| `extract_regex` | StringExtraction | **CPU** | `extract_regex(pattern)` | `Kernels/Regex.swift` | Named groups on the host, returned as a struct column. ICU spelling `(?<name>...)`; RE2's `(?P<name>...)`, which pyarrow uses, is rewritten by the Python wrapper. |
| `extract_regex_span` | StringExtraction | **CPU** | `extract_regex_span(pattern)` | `Kernels/StringExtra.swift` | One `(start, length)` pair of int32 columns per **named** group, counted in bytes as Arrow's are; the registry reshapes them into Arrow's row-wise struct. A row that does not match, a null row and a group that took part in no alternative are null in both. Host-side throughout (ICU, sharded over 4096-row chunks), and ICU's `(?<name>...)` spelling rather than RE2's `(?P<name>...)`. |
| `binary_join` | StringJoining | **GPU** | `binary_join(separator)` | `Kernels/StringContainment.swift` | Joins the child strings of every row of a `list<utf8>`, two-pass on the GPU. The separator is a scalar or a per-row column. An empty row joins to the empty string; a null row, a null element inside a row and a null separator all give a null output row. |
| `binary_join_element_wise` | StringJoining | **Partial** | `str_concat(other, separator)` | `Kernels/StringTransforms.swift` | Two columns and a scalar separator on the GPU. Arrow's N-column form (the last argument being the separator column) and its `null_handling` options are not implemented. |
| `utf8_slice_codeunits` | StringSlicing | **GPU** | `slice_codeunits(start, stop)` | `Kernels/StringTransforms.swift` | Code-point slicing, two-pass on the GPU. A negative `start`/`stop` and Arrow's `step` are not implemented. |
| `binary_slice` | StringSlicing | Missing | `-` | `-` | The one Arrow compute name with no implementation here. The slicing kernel counts code points and there is no byte-offset variant of it; `binary_replace_slice` does index in bytes, so the machinery exists and this is unclaimed rather than out of scope. |
| `count_substring` | Containment | **GPU** | `count_substring(pattern)` | `Kernels/StringTransforms.swift` | Byte-wise search, one thread per row. |
| `find_substring` | Containment | **GPU** | `find_substring(pattern)` | `Kernels/StringTransforms.swift` | First byte offset, -1 when absent. |
| `match_substring` | Containment | **GPU** | `str_contains(pattern)` | `Sources/ArrowMetal/MetalStringArray.swift` | Byte-wise containment. `ignore_case` is not implemented. |
| `starts_with` | Containment | **GPU** | `starts_with(pattern)` | `Sources/ArrowMetal/MetalStringArray.swift` | Byte-wise prefix test. |
| `ends_with` | Containment | **GPU** | `ends_with(pattern)` | `Sources/ArrowMetal/MetalStringArray.swift` | Byte-wise suffix test. |
| `count_substring_regex` | Containment | **CPU** | `count_substring_regex(pattern)` | `Kernels/Regex.swift` | Host engine; a literal pattern routes to the GPU counter instead. |
| `find_substring_regex` | Containment | **CPU** | `find_substring_regex(pattern)` | `Kernels/Regex.swift` | Host engine. |
| `match_substring_regex` | Containment | **CPU** | `match_substring_regex(pattern)` | `Kernels/Regex.swift` | Host engine; literal and `^literal` patterns route to the GPU kernels. |
| `match_like` | Containment | **CPU** | `match_like(pattern)` | `Kernels/Regex.swift` | SQL LIKE, translated to the host regex engine; a pattern with no metacharacter routes to the GPU. |
| `is_in` | Containment | **GPU** | `is_in(value_set, null_matching_behavior)` | `Kernels/SetLookup.swift` | GPU throughout: a sorted set plus a binary search per row for primitive and temporal columns, the GPU string hash table for utf8 and binary ones. All four of Arrow's `null_matching_behavior` values are implemented — `match`, `skip`, `emit_null` and `inconclusive` — as a rewrite of the validity bitmap over the kernel's own `skip` answer, since the four differ only in what a null row reports. ArrowMetal defaults to `skip`; pyarrow defaults to `match`, so the check below passes it. |
| `index_in` | Containment | **GPU** | `index_in(value_set, null_matching_behavior)` | `Kernels/SetLookup.swift` | The same two paths and the same four behaviours, returning the int32 position of the first occurrence in the set and null where the element is absent. Only `match` differs from the other three for `index_in`: it reports the position of the value set's first null for a null element. |
| `is_null` | Categorizations | **GPU** | `is_null()` | `Kernels/Structural.swift` | The validity bitmap inverted on the GPU. Arrow's `nan_is_null` is not implemented. |
| `is_valid` | Categorizations | **GPU** | `is_valid()` | `Kernels/Structural.swift` | The validity bitmap copied out as a boolean column. |
| `true_unless_null` | Categorizations | **CPU** | `true_unless_null()` | `Kernels/Selection.swift` | A host `memset` of the values bitmap; the validity bitmap is shared with the input with no copy, so no kernel runs. Union and run-end encoded columns are refused. |
| `indices_nonzero` | Categorizations | **GPU** | `indices_nonzero()` | `Kernels/Conditional.swift` | `iota` through the existing stream compaction: the uint64 row numbers where the value is valid and not zero. `-0.0` counts as zero and every NaN as non-zero, as in Arrow. |
| `is_finite` | Categorizations | **GPU** | `is_finite()` | `Kernels/FloatClass.swift` | A raw bit-pattern test, one thread per 32-bit output word, so float64 needs no software binary64. True everywhere on an integer column, and null where the input is null. |
| `is_inf` | Categorizations | **GPU** | `is_inf()` | `Kernels/FloatClass.swift` | As `is_finite`; false everywhere on an integer column. |
| `is_nan` | Categorizations | **GPU** | `is_nan()` | `Kernels/FloatClass.swift` | As `is_finite`; false everywhere on an integer column. |
| `if_else` | Selecting | **GPU** | `if_else(left, right)` | `Kernels/Structural.swift` | One thread per element; a null condition gives a null output. |
| `coalesce` | Selecting | **GPU** | `am.coalesce(*arrays)` | `Kernels/Structural.swift` | First non-null across the inputs, one thread per element. |
| `case_when` | Selecting | **GPU** | `am.case_when(conds, values, default)` | `Kernels/Conditional.swift` | A fold of the existing `if_else` kernel, one GPU pass per branch. A **null condition counts as false** and the row falls through, as in Arrow. ArrowMetal takes the conditions as a list of boolean columns where Arrow takes one struct column of them. |
| `choose` | Selecting | **GPU** | `am.choose(indices, values)` | `Kernels/Conditional.swift` | `values[indices[i]][i]`, element-wise, as the same fold. A null index gives a null output and an index outside [0, len(values)) raises, as in Arrow. |
| `cast` | Conversions | **GPU** | `cast(target, safe=..., allow_*=...)` | `Sources/ArrowMetal/CastDispatch.swift` | One entry point for every target, taking Arrow's whole `CastOptions`. Numeric to numeric, bool to and from numeric, numeric and temporal to utf8 and utf8 back to numeric, temporal resolution changes and the date/timestamp conversions, integer to and from decimal128 and a decimal rescale, and `list<T>` -> `list<U>` and struct casts that cast the children and share the offsets and bitmaps. `safe=True` adds one read-only GPU pass that converts each value back and raises on the first row that does not round-trip, which is exactly the set of losses Arrow objects to; each `allow_*` flag turns one class back off. ArrowMetal defaults to `safe=False`, where pyarrow defaults to `safe=True`. Two things remain: with `safe=False` an out-of-range float -> integer value saturates at 64 bits and truncates where Arrow saturates at the target's width (C leaves it undefined; `safe=True` refuses the row rather than choosing), and dictionary, union, run-end and interval targets, and utf8 -> temporal (that is `strptime`), are refused. |
| `ceil_temporal` | Conversions | **GPU** | `ceil_temporal(unit, multiple, week_starts_monday, ceil_is_strictly_greater, calendar_based_origin)` | `Kernels/TemporalMath.swift` | As `round_temporal`, with the whole `RoundTemporalOptions` surface. A value already on a boundary is left alone unless `ceil_is_strictly_greater` — except on `month`, `quarter` and `year`, where Arrow's own ceil always advances a boundary value and the flag makes no difference; that quirk is reproduced deliberately. |
| `floor_temporal` | Conversions | **GPU** | `floor_temporal(unit, multiple, week_starts_monday, ceil_is_strictly_greater, calendar_based_origin)` | `Kernels/TemporalMath.swift` | As `ceil_temporal`. |
| `round_temporal` | Conversions | **GPU** | `round_temporal(unit, multiple, week_starts_monday, ceil_is_strictly_greater, calendar_based_origin)` | `Kernels/TemporalMath.swift` | The whole of Arrow's `RoundTemporalOptions`, in integer arithmetic in the value's own resolution plus the civil-date algorithm for the calendar units. An exact half rounds **up** (toward +infinity), which is what Arrow does. `week` is a seven-day grid on a Monday or Sunday anchor; `calendar_based_origin` starts the grid at the beginning of the value's own next-greater calendar unit; the month and quarter grids are anchored at 1970-01 and the year grid at year 0, as Arrow anchors them. |
| `run_end_encode` | Conversions | **GPU** | `run_end_encode()` | `Sources/ArrowMetal/RunEndEncoded.swift` | Bit equality collapses adjacent values; nulls form runs of their own. Run ends are int32. |
| `run_end_decode` | Conversions | **GPU** | `run_end_decode()` | `Sources/ArrowMetal/RunEndEncoded.swift` | Expands a run-end encoded column back to a flat one. |
| `year` | TemporalExtraction | **GPU** | `year()` | `Sources/ArrowMetal/Temporal.swift` | The civil-date algorithm on the GPU, UTC. A timestamp's timezone is carried as metadata and never applied. |
| `month` | TemporalExtraction | **GPU** | `month()` | `Sources/ArrowMetal/Temporal.swift` | UTC. |
| `day` | TemporalExtraction | **GPU** | `day()` | `Sources/ArrowMetal/Temporal.swift` | UTC. |
| `day_of_week` | TemporalExtraction | **GPU** | `day_of_week(count_from_zero, week_start)` | `Kernels/TemporalExtra.swift` | Arrow's full `DayOfWeekOptions`; `week_start` uses the ISO numbering (1 = Monday ... 7 = Sunday). The default options keep the int32 result (Monday = 0); any other combination returns int64, as pyarrow does. |
| `hour` | TemporalExtraction | **GPU** | `hour()` | `Sources/ArrowMetal/Temporal.swift` | UTC. |
| `minute` | TemporalExtraction | **GPU** | `minute()` | `Sources/ArrowMetal/Temporal.swift` | UTC. |
| `second` | TemporalExtraction | **GPU** | `second()` | `Sources/ArrowMetal/Temporal.swift` | UTC. |
| `day_of_year` | TemporalExtraction | **GPU** | `day_of_year()` | `Kernels/TemporalMath.swift` | UTC. |
| `quarter` | TemporalExtraction | **GPU** | `quarter()` | `Kernels/TemporalMath.swift` | UTC. |
| `iso_week` | TemporalExtraction | **GPU** | `iso_week()` | `Kernels/TemporalMath.swift` | UTC. |
| `iso_year` | TemporalExtraction | **GPU** | `iso_year()` | `Kernels/TemporalMath.swift` | UTC. |
| `millisecond` | TemporalExtraction | **GPU** | `millisecond()` | `Kernels/TemporalMath.swift` | UTC. |
| `microsecond` | TemporalExtraction | **GPU** | `microsecond()` | `Kernels/TemporalMath.swift` | UTC. |
| `nanosecond` | TemporalExtraction | **GPU** | `nanosecond()` | `Kernels/TemporalMath.swift` | UTC. |
| `is_leap_year` | TemporalExtraction | **GPU** | `is_leap_year()` | `Kernels/TemporalMath.swift` | UTC. |
| `strftime` | TemporalExtraction | **CPU** | `strftime(format)` | `Kernels/TemporalMath.swift` | The C library's `strftime` against a `gmtime_r` struct on the host, plus a `%f` extension for microseconds. Not a Unicode date pattern, and always UTC. |
| `strptime` | TemporalExtraction | **CPU** | `strptime(format)` | `Kernels/TemporalMath.swift` | The C library's `strptime` on the host, UTC. `error_is_null` is not implemented — an unparseable row is null either way. |
| `is_dst` | TemporalExtraction | **CPU** | `is_dst()` | `Kernels/TemporalExtra.swift` | The one function here that does apply a timestamp's timezone, and so the one that needs the IANA tz database — host data with no GPU-resident form. Runs on the host, sharded over `DispatchQueue.concurrentPerform`. A naive timestamp is an error, as in Arrow. |
| `iso_calendar` | TemporalExtraction | **GPU** | `iso_calendar()` | `Kernels/TemporalExtra.swift` | A struct of int64 `iso_year`, `iso_week` and `iso_day_of_week` (1 = Monday), UTC. |
| `subsecond` | TemporalExtraction | **GPU** | `subsecond()` | `Kernels/TemporalExtra.swift` | The fraction of a second in [0, 1) as float64, UTC. date32 and date64 answer 0 (pyarrow has no kernel for them) and duration is rejected. |
| `us_week` | TemporalExtraction | **GPU** | `us_week()` | `Kernels/TemporalExtra.swift` | The US week number: Sunday-start weeks and the majority rule, 1-53. UTC. |
| `us_year` | TemporalExtraction | **GPU** | `us_year()` | `Kernels/TemporalExtra.swift` | The US epidemiological week-numbering year — the year owning the Wednesday of this date's Sunday-start week. UTC. |
| `week` | TemporalExtraction | **GPU** | `week(week_starts_monday, count_from_zero, first_week_is_fully_in_year)` | `Kernels/TemporalExtra.swift` | Arrow's full `WeekOptions`, int64, UTC. The defaults reproduce `iso_week`. |
| `year_month_day` | TemporalExtraction | **GPU** | `year_month_day()` | `Kernels/TemporalExtra.swift` | A struct of int64 `year`, `month` and `day`, UTC. |
| `days_between` | TemporalDifference | **GPU** | `days_between(other)` | `Kernels/TemporalMath.swift` | Whole days between two temporal columns, UTC. |
| `day_time_interval_between` | TemporalDifference | **Partial** | `day_time_interval_between(other).interval_field(...)` | `Sources/ArrowMetal/IntervalBetween.swift` | GPU. pyarrow 25 has no Python type or Array class for `interval[day_time]`, so the result cannot be handed to pyarrow at all: read the fields with `interval_field('days' \| 'nanoseconds')`, which is what the check below compares against `days_between`. |
| `hours_between` | TemporalDifference | **GPU** | `hours_between(other)` | `Kernels/TemporalExtra.swift` | Hour boundaries crossed, int64, UTC. |
| `microseconds_between` | TemporalDifference | **GPU** | `microseconds_between(other)` | `Kernels/TemporalExtra.swift` | Microsecond boundaries crossed. |
| `milliseconds_between` | TemporalDifference | **GPU** | `milliseconds_between(other)` | `Kernels/TemporalExtra.swift` | Millisecond boundaries crossed. |
| `minutes_between` | TemporalDifference | **GPU** | `minutes_between(other)` | `Kernels/TemporalExtra.swift` | Minute boundaries crossed. |
| `month_day_nano_interval_between` | TemporalDifference | **GPU** | `month_day_nano_interval_between(other)` | `Sources/ArrowMetal/IntervalBetween.swift` | The one interval difference pyarrow can express in Python, so this one round-trips as an `interval[month_day_nano]` column. Every field is the difference of the corresponding truncated field, so the day and sub-day parts may carry the opposite sign. |
| `month_interval_between` | TemporalDifference | **Partial** | `month_interval_between(other).interval_field('months')` | `Sources/ArrowMetal/IntervalBetween.swift` | GPU: month boundaries crossed. Same pyarrow gap as `day_time_interval_between` — there is no Python type for `interval[month]`, so the months are read with `interval_field('months')` and checked against the `months` field of `month_day_nano_interval_between`. |
| `nanoseconds_between` | TemporalDifference | **GPU** | `nanoseconds_between(other)` | `Kernels/TemporalExtra.swift` | Nanosecond boundaries crossed; wraps in int64 past about 292 years, as Arrow's does. |
| `quarters_between` | TemporalDifference | **GPU** | `quarters_between(other)` | `Kernels/TemporalExtra.swift` | The difference of year * 4 + quarter. |
| `seconds_between` | TemporalDifference | **GPU** | `seconds_between(other)` | `Kernels/TemporalExtra.swift` | Second boundaries crossed. |
| `weeks_between` | TemporalDifference | **GPU** | `weeks_between(other, count_from_zero, week_start)` | `Kernels/TemporalExtra.swift` | Week boundaries crossed, both sides floored to the start of their week first. `week_start` is 1 = Monday ... 7 = Sunday; `count_from_zero` is accepted for signature parity and does not change the answer, as in Arrow. |
| `years_between` | TemporalDifference | **GPU** | `years_between(other)` | `Kernels/TemporalExtra.swift` | The difference of the two calendar years. |
| `assume_timezone` | Timezone | **CPU** | `assume_timezone(tz, ambiguous, nonexistent)` | `Sources/ArrowMetal/Timezone.swift` | Reads a naive column as wall-clock times in `tz` and returns the instants they name. Host-side deliberately: the offsets are a lookup in the IANA tz database, which has no GPU-resident form, so uploading the transition table per call would cost more than the arithmetic saves. Sharded over `DispatchQueue.concurrentPerform` with a per-shard offset cache. A local time that occurs twice or never raises by default; `"earliest"` / `"latest"` pick one, as Arrow does. |
| `local_timestamp` | Timezone | **CPU** | `local_timestamp()` | `Sources/ArrowMetal/Timezone.swift` | The wall-clock time each instant names in the column's own timezone, as a naive timestamp of the same unit. Host-side for the same reason as `assume_timezone`; a column with no timezone comes back unchanged. |
| `random` | Random | **GPU** | `am.random(n, initializer)` | `Kernels/Selection.swift` | Philox4x32-10 keyed by the seed, one counter per element, so the stream depends only on the seed. The top 53 bits of each draw become a multiple of 2^-53 in [0, 1). ArrowMetal's own stream: it does not reproduce Arrow C++'s pcg32_fast numbers for the same seed. |
| `unique` | Associative | **GPU** | `unique(order)` | `Kernels/UniqueOrder.swift` | One GPU sort plus a run scan gives the distinct values ascending; `order="first_appearance"` (the default, and Arrow's own order) then reorders them with a GPU group-min of the row index per distinct value, a stable argsort of those minima and a gather — and, like Arrow, keeps the null as one entry at the position of the first null row. `order="sorted"` is the ascending pass on its own, nulls dropped, and the cheaper of the two. A utf8 column has no null entry in either order: the GPU string dictionary has no slot for one. |
| `value_counts` | Associative | **GPU** | `value_counts(order)` | `Kernels/UniqueOrder.swift` | The same two orders and the same null handling, returned as a struct of `values` and `counts` (int64). |
| `dictionary_encode` | Associative | **GPU** | `dictionary_encode(order)` | `Kernels/UniqueOrder.swift` | GPU hashing for utf8, a GPU sort for primitives, then the same first-appearance reordering `unique` uses (the default) or the sorted dictionary. Returns `(codes, values)` rather than Arrow's dictionary-typed array. The dictionary never holds a null and a null row gets a null code, as in Arrow. |
| `dictionary_decode` | Associative | **GPU** | `dictionary_decode()` | `Sources/ArrowMetal/DictionaryArray.swift` | A `take` of the values through the codes. |
| `filter` | Selections | **GPU** | `filter(mask)` | `Kernels/Filter.swift` | Per-block popcount, GPU scan, scatter and validity pack, all in one command buffer. A null mask entry drops the row (Arrow's `null_selection_behavior="drop"`); `"emit_null"` is not implemented. |
| `array_filter` | Selections | **GPU** | `array_filter(mask)` | `Kernels/Filter.swift` | The same kernel under Arrow's array-only name. |
| `take` | Selections | **GPU** | `take(indices)` | `Kernels/Take.swift` | One gather kernel. A null index yields a null row; an out-of-range index sets a GPU error flag raised after the dispatch. |
| `array_take` | Selections | **GPU** | `array_take(indices)` | `Kernels/Take.swift` | The same kernel under Arrow's array-only name. |
| `drop_null` | Selections | **GPU** | `drop_null()` | `Kernels/Structural.swift` | The filter kernel driven by the validity bitmap. |
| `inverse_permutation` | Selections | **GPU** | `inverse_permutation(max_index)` | `Kernels/Selection.swift` | An atomic scatter: for the i-th index the index-th output is i. Unassigned slots are null and duplicates resolve to the last source position, deterministically (the scatter is an atomic maximum). Always int32 — Arrow's `output_type` is not implemented. |
| `scatter` | Selections | **GPU** | `scatter(indices, max_index)` | `Kernels/Selection.swift` | The inverse permutation used as a `take`, so it works for every column type. Unassigned positions are null and duplicate indices resolve to the last value. |
| `array_sort_indices` | Sorts | **GPU** | `array_sort_indices(descending, null_placement)` | `Kernels/Sort.swift` | LSD radix sort, stable, total order for floats (NaN after +inf). Both of Arrow's `null_placement` values are implemented, in both directions: the nulls are one block at whichever end, moved there by a stable partition of the index array. |
| `sort_indices` | Sorts | **Partial** | `sort_indices() / am.lexsort_indices(cols)` | `Kernels/MultiSort.swift` | Single key through the radix argsort; multiple keys through `lexsort_indices`, which is successive stable argsorts from the least significant key upwards. Both `null_placement` values are implemented, and apply to every key as Arrow's do. utf8, binary and dictionary key columns are still not sortable, which is what keeps this row `partial`. |
| `partition_nth_indices` | Sorts | **GPU** | `partition_nth_indices(pivot, null_placement)` | `Kernels/PartitionNth.swift` | A real selection, not a sort: an MSB-first GPU radix select finds the pivot value in a fixed four (32-bit keys) or eight (64-bit) histogram passes, and three GPU stream compactions split the row indices around it. O(length). Both `null_placement` values are implemented. The permutation is not the sorted one, and Arrow does not promise it is — only the partition property, which the check below verifies. |
| `select_k_unstable` | Sorts | **GPU** | `select_k_unstable(k, largest)` | `Kernels/TopK.swift` | For k <= 1024 each threadgroup keeps the best k of its own block and one radix sort orders the survivors; larger k falls back to the full sort. Single key. |
| `top_k_unstable` | Sorts | **GPU** | `top_k_unstable(k)` | `Kernels/TopK.swift` | `select_k_unstable` with the descending order. |
| `bottom_k_unstable` | Sorts | **GPU** | `bottom_k_unstable(k)` | `Kernels/TopK.swift` | `select_k_unstable` with the ascending order. |
| `rank` | Sorts | **GPU** | `rank(sort_keys, null_placement, tiebreaker)` | `Kernels/Window.swift` | One argsort, run marks, a scan and a scatter back to the original rows, with Arrow's whole option surface: the sort direction, both `null_placement` values and all four tiebreakers (`min`, `max`, `first`, `dense`, which are also spelled `rank()`, `max_rank()`, `row_number()` and `dense_rank()`). The result never contains nulls. |
| `rank_quantile` | Sorts | **GPU** | `rank_quantile(sort_keys, null_placement)` | `Kernels/Selection.swift` | (average 1-based rank of the tie group - 0.5) / n, computed on the GPU as (s + e) / (2n) over the run's sorted positions with the correctly rounded software binary64 divide. Arrow's `sort_keys` direction and both `null_placement` values are implemented; the nulls are one tie group at whichever end. |
| `rank_normal` | Sorts | **Partial** | `rank_normal(sort_keys, null_placement, float32=False)` | `Kernels/Selection.swift` | The normal percent-point function of `rank_quantile`, with the same `sort_keys` and `null_placement` options. float64 evaluates the inverse CDF **on the host** with Wichura's AS 241 (about 1e-16 relative) because Metal has no `double` and the software binary64 has no log/exp/erfc — that host step is what keeps this row `partial`; `float32=True` runs Acklam plus one Halley refinement entirely on the GPU, within about 1e-6. |
| `winsorize` | Sorts | **GPU** | `winsorize(lower_limit, upper_limit)` | `Kernels/Selection.swift` | One GPU sort for the two nearest quantiles, then a clamp kernel. Nulls stay null and NaNs pass through, taking part in neither the limits nor the comparison. |
| `fill_null` | NullFilling | **GPU** | `fill_null(value)` | `Kernels/Structural.swift` | One thread per element; the result carries no validity bitmap when the fill removes every null. |
| `fill_null_forward` | NullFilling | **GPU** | `fill_null_forward()` | `Kernels/Conditional.swift` | One GPU max-scan over the last valid row index plus a gather. Leading nulls stay null. |
| `fill_null_backward` | NullFilling | **GPU** | `fill_null_backward()` | `Kernels/Conditional.swift` | The same scan run the other way. Trailing nulls stay null. |
| `list_value_length` | Structural | **GPU** | `list_value_length()` | `Sources/ArrowMetal/Nested.swift` | The offsets difference; null lists give null. |
| `list_flatten` | Structural | **GPU** | `list_flatten()` | `Sources/ArrowMetal/Nested.swift` | Concatenates the child ranges of the valid rows. |
| `list_element` | Structural | **GPU** | `list_element(index)` | `Sources/ArrowMetal/Nested.swift` | A gather through offsets + index; a row too short gives null where Arrow raises. |
| `struct_field` | Structural | **GPU** | `struct_field(name)` | `Sources/ArrowMetal/Nested.swift` | The named child with the struct's own nulls propagated into it. Only a single-level field name, not Arrow's nested index path. |
| `make_struct` | Structural | **CPU** | `am.make_struct(arrays, names)` | `Sources/ArrowMetal/Nested.swift` | Metadata only: the children are shared, nothing is copied and no kernel runs. |
| `list_parent_indices` | Structural | **GPU** | `list_parent_indices64() / list_parent_indices()` | `Sources/ArrowMetal/NestedExtra.swift` | GPU, one binary search per child element. `list_parent_indices64()` returns int64, the width pyarrow returns; `list_parent_indices()` keeps the int32 form, which is what the list offsets themselves are and what every caller inside this package wants. |
| `list_slice` | Structural | **GPU** | `list_slice(start, stop, step)` | `Sources/ArrowMetal/NestedExtra.swift` | `row[start:stop:step]` for every row, as a variable-length list. `start` must be >= 0 and `step` >= 1, as Arrow requires; a null row stays null and a row shorter than `start` becomes empty. |
| `map_lookup` | Structural | **GPU** | `map_lookup(key, occurrence)` | `Sources/ArrowMetal/NestedExtra.swift` | One key compare per entry inside each row's range. `occurrence` is `first`, `last` or `all` (which returns a list of the item type); the result is null where the row is null or the key is absent. Keys may be utf8 / binary or any integer type. |
| `replace_with_mask` | Structural | **GPU** | `replace_with_mask(mask, replacements)` | `Kernels/Conditional.swift` | Rows where the mask is true take the next value from `replacements`, in order; rows where the mask is null become null; every other row keeps its own value. A GPU scan supplies the replacement index. Fewer replacements than valid trues raises, a surplus is ignored, both as in pyarrow. |
| `pairwise_diff` | Pairwise | **GPU** | `pairwise_diff(period)` | `Kernels/Window.swift` | One thread per element; null where either side is null or outside the array. Integers wrap. |
| `pairwise_diff_checked` | Pairwise | **GPU** | `pairwise_diff_checked(period)` | `Kernels/Checked.swift` | `self[i] - self[i - period]`, raising where that step would wrap. |
| `cumulative_sum` | Cumulative | **GPU** | `cumulative_sum()` | `Kernels/Cumulative.swift` | A two-level scan. Null in, null out, the running value carrying across nulls (Arrow's `skip_nulls=True`). `start` is not implemented. |
| `cumulative_prod` | Cumulative | **GPU** | `cumulative_prod()` | `Kernels/Window.swift` | The same scan with a multiply; float products reassociate. |
| `cumulative_max` | Cumulative | **GPU** | `cumulative_max()` | `Kernels/Cumulative.swift` | The same scan with a maximum. |
| `cumulative_min` | Cumulative | **GPU** | `cumulative_min()` | `Kernels/Cumulative.swift` | The same scan with a minimum. |
| `cumulative_mean` | Cumulative | **GPU** | `cumulative_mean()` | `Kernels/Window.swift` | A binary64 running sum over a running count of non-null rows; int64 magnitudes above 2^53 round on the way in. |
| `cumulative_sum_checked` | Cumulative | **GPU** | `cumulative_sum_checked()` | `Kernels/Checked.swift` | The running sum, raising where a step would wrap. Null rows are skipped and the running value carries across them (this package's `cumulative_sum` behaviour, Arrow's `skip_nulls=True`); pyarrow's default instead makes every row after a null null. |
| `cumulative_prod_checked` | Cumulative | **GPU** | `cumulative_prod_checked()` | `Kernels/Checked.swift` | The running product, raising where a step would wrap. Same null rule as `cumulative_sum_checked`. |
| `hash_count` | GroupedAggregations | **GPU** | `group_by(keys).count(values)` | `Kernels/GroupByKeys.swift` | Non-null values per group, for any value type. Arrow's `mode` option is not implemented — this is always `only_valid`; `hash_count_all` is the `all` mode. |
| `hash_count_all` | GroupedAggregations | **GPU** | `group_by(keys).count_all()` | `Kernels/AggregatesExtra.swift` | Rows per group, null values included. A group with no row is zero, never null. |
| `hash_sum` | GroupedAggregations | **GPU** | `group_by(keys).sum(values)` | `Kernels/GroupByKeys.swift` | A segmented reduction over the dense ids. Integers accumulate in 64 bits and wrap; `min_count` is not implemented. |
| `hash_mean` | GroupedAggregations | **GPU** | `group_by(keys).mean(values)` | `Kernels/GroupByKeys.swift` | The grouped sum over the grouped valid count. |
| `hash_min` | GroupedAggregations | **GPU** | `group_by(keys).min(values)` | `Kernels/GroupByKeys.swift` | A segmented minimum. |
| `hash_max` | GroupedAggregations | **GPU** | `group_by(keys).max(values)` | `Kernels/GroupByKeys.swift` | A segmented maximum. |
| `hash_min_max` | GroupedAggregations | **GPU** | `group_by(keys).min_max(values)` | `Kernels/AggregatesExtra.swift` | Fused: one segmented kernel produces both extremes from one read of the values, as a `struct<min, max>` column. |
| `hash_all` | GroupedAggregations | **GPU** | `group_by(keys).all(values)` | `Kernels/AggregatesExtra.swift` | Three-valued AND per group over a boolean column, matching Arrow's null handling. |
| `hash_any` | GroupedAggregations | **GPU** | `group_by(keys).any(values)` | `Kernels/AggregatesExtra.swift` | Three-valued OR per group. |
| `hash_approximate_median` | GroupedAggregations | **GPU** | `group_by(keys).approximate_median(values)` | `Kernels/AggregatesExtra.swift` | **Exact**, not approximate: a GPU sort by (group, value) and a GPU per-group pick, because sorting on the GPU is cheaper than sketching. Arrow's is a t-digest, so on a group whose size makes the sketch inexact this answers the true median and pyarrow does not. |
| `hash_count_distinct` | GroupedAggregations | **GPU** | `group_by(keys).count_distinct(values)` | `Kernels/AggregatesExtra.swift` | Dictionary-encode, packed `unique`, then a segmented count. |
| `hash_distinct` | GroupedAggregations | **GPU** | `group_by(keys).distinct(values)` | `Kernels/AggregatesExtra.swift` | The distinct non-null values of each group as a list column, **ascending** — Arrow returns them in order of first appearance, as it does for the scalar `unique`. |
| `hash_first` | GroupedAggregations | **GPU** | `group_by(keys).first(values)` | `Kernels/AggregatesExtra.swift` | A group-by extreme over a masked row index plus a gather. |
| `hash_first_last` | GroupedAggregations | **GPU** | `group_by(keys).first_last(values)` | `Kernels/AggregatesExtra.swift` | Both ends in one `struct<first, last>` column: two group-by extremes over the masked row index plus two gathers. |
| `hash_last` | GroupedAggregations | **GPU** | `group_by(keys).last(values)` | `Kernels/AggregatesExtra.swift` | Mirror of `hash_first`. |
| `hash_list` | GroupedAggregations | **GPU** | `group_by(keys).list(values)` | `Kernels/AggregatesExtra.swift` | Every value of the group in row order, as a list column: a segmented gather after the stable sort by group id. |
| `hash_one` | GroupedAggregations | **GPU** | `group_by(keys).one(values)` | `Kernels/AggregatesExtra.swift` | One value per group — here always the group's lowest row, null included. Arrow leaves which one unspecified. |
| `hash_product` | GroupedAggregations | **GPU** | `group_by(keys).product(values)` | `Kernels/AggregatesExtra.swift` | A segmented multiply reduction. Integers wrap in 64 bits exactly as the scalar `product` does, and float products reassociate across the threads of a group. |
| `hash_stddev` | GroupedAggregations | **GPU** | `group_by(keys).stddev(values, ddof)` | `Kernels/GroupByKeys.swift` | The square root of `hash_variance`, and so carries the same float32 deviations: expect about 1e-5 relative. |
| `hash_variance` | GroupedAggregations | **GPU** | `group_by(keys).variance(values, ddof)` | `Kernels/GroupByKeys.swift` | Two GPU passes: per-group means, then the squared deviations. The deviations are formed in float32 about a float64 mean, so expect about 1e-5 relative on well-conditioned data rather than the 1e-15 of the scalar `variance`. |
| `hash_pivot_wider` | GroupedAggregations | **GPU** | `group_by(keys).pivot_wider(pivot_keys, values, names)` | `Kernels/AggregatesExtra.swift` | One masked `hash_one` per pivot key, giving a struct with one field per name. A (group, key) pair carrying more than one non-null value takes the lowest row here where pyarrow raises. |
| `hash_kurtosis` | GroupedAggregations | **GPU** | `group_by(keys).kurtosis(values)` | `Kernels/AggregatesExtra.swift` | Excess kurtosis, biased, from two GPU passes over the per-group means. Same float32 deviations as `hash_variance`, so about 1e-5 relative. A group with too few values is null here where pyarrow returns NaN. |
| `hash_skew` | GroupedAggregations | **GPU** | `group_by(keys).skew(values)` | `Kernels/AggregatesExtra.swift` | The third standardised central moment, biased. Same passes, precision and null-on-degenerate group as `hash_kurtosis`. |
| `hash_tdigest` | GroupedAggregations | **Partial** | `group_by(keys).tdigest(values, q)` | `Kernels/AggregatesExtra.swift` | Mixed: a GPU sort by (group, value) and a **host** merge of each group's centroids. Returns one q per group as a scalar column where Arrow returns a list, and being a sketch it agrees with Arrow's to within the sketch's error. `group_by(keys).quantile(values, q)` is the exact answer. |

---

Version 0.1.0. Read alongside [COVERAGE.md](COVERAGE.md), [ROADMAP.md](../ROADMAP.md) and [DESIGN.md](DESIGN.md).
