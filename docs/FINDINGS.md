# Findings and gotchas

Things learned the hard way. Add to this whenever something surprises you.

## Round 16 (2026-09-30): Float64 group sums without a group order

**What.** Float64 group-by aggregates were the slowest ones ArrowMetal ran. Over 50M rows a Float64
`sum` took 10.62 ms at 200 groups, 39.78 ms at 1M groups and 227.11 ms at 25M groups (rows / 2),
where an Int64 `sum` took 1.73, 7.94 and 52.16 ms and `count` 1.00, 1.66 and 15.29 ms
(`Benchmarks/results/groupby_float64_profile_2026-09-30.csv`, the Swift API, best of 5).

**Where the time went.** A Float64 sum put each group's rows together first (the counting sort by
group id, `GroupBy.segments()`: 7.02 ms at 200 groups, 29.20 ms at 1M, 92.60 ms at 25M) and then gave
a 256-thread threadgroup to each group, which added its run in a fixed tree order (3.64, 10.38 and
134.15 ms). The order was the reason for the sort: the tree is what made the answer reproducible. At
200 groups that reduction ran 200 threadgroups over 250,000 rows each; at 25M groups it launched 6.4
billion threads for 50M additions. `min` and `max` took two passes of 32-bit atomics, with three
atomic updates per row in the first (the high word's minimum and maximum, and a count) spread over
four arrays, and a plan with a `min` and a `max` of one column ran both passes twice.

**Change.** A correctly rounded sum does not depend on the order of its terms, so Float64 `sum` and
`mean` no longer build a group order (`Kernels/GroupSumExact.swift`, DESIGN.md "Group-by"):

- Pass one takes each group's largest exponent with an atomic max of the 11-bit field, counts its
  values, and records NaN, ±inf and whether a +0.0 was seen.
- Pass two writes each value as an integer in a per-group fixed-point window, `G = 74 -
  bitlength(rows)` bits below the last bit of the group's largest exponent, and adds it to a 128-bit
  two's-complement accumulator with 32-bit atomics that carry (or borrow) word by word. Each word's
  carry comes from its own read-modify-write, so the four words hold the exact sum modulo 2^128 in any
  interleaving, and the headroom keeps every sum inside it. A value whose last bits fall below the
  window is truncated and counted.
- A thread per group rounds once to nearest-even. With nothing truncated that is the correctly
  rounded sum, and the mean is the 128-bit sum divided by the count in a long division, rounded once
  (a power-of-two count is a shift). With `d` truncated values the exact sum lies strictly between
  `W - d` and `W + d` units, and the result stands when both ends round to the same double. A group
  whose rounding that cannot settle, or that holds a NaN other than `0x7FF8000000000000`, is summed on
  the host from its own rows with an exact 2,240-bit accumulator.
- Up to 1,024 groups each threadgroup accumulates in threadgroup memory and merges once.

`min` and `max` keep their two passes and change their table: a group's four words sit in one
16-byte entry, a row in pass one reads them and issues an atomic only when it improves on what it
read, pass two reads the finished high words through a plain view of the table, and no count is
kept (a group has a value exactly when its minimum key is at or below its maximum key). A `sum` and
a `mean`, or a `min` and a `max`, of one column on one grouping share one call.

**Behaviour change.** Grouped Float64 `sum` is now correctly rounded: the binary64 value nearest the
exact sum of the group's non-null values, ties to even, equal to `math.fsum` of those values. It no
longer depends on row order, it differs from the earlier ordered sum in the last bits of some groups,
and it is never further from the exact sum than that sum. `mean` is the exact sum divided by the
count, rounded once. A group whose exact sum is finite returns it where a running sum overflowed
(`[max, max, -max]` gives `max`); one whose exact sum is beyond the largest double gives ±inf. NaN
in a group gives NaN: the canonical quiet NaN when every NaN of the group is that one, else the
group's first NaN in row order, quieted. +inf with -inf gives the canonical quiet NaN; an exact zero
is -0.0 only when every value is -0.0. `min` and `max` return the same bits as before. The streaming
group-by keeps the ordered sum, so its per-batch, resident and host paths stay bit-identical to each
other.

**Measured.** The builds before and after, timed alternately in three rounds, with a warm-up of at
least 100 ms before each case and the first run after 500 ms of idle recorded apart
(`Benchmarks/results/groupby_float64_2026-09-30.csv`; every round in `…_raw_…`, the conditions in
`…_conditions.txt`). The array API is one `am.group_by(keys)` and one aggregate per call; the plan
runner is a warm `collect()`. At 50M rows, one int32 key, no nulls:

| aggregate | groups | array API | plan runner |
|---|---:|---:|---:|
| `sum` | 200 | 12.73 → 5.62 ms (2.27x) | 14.87 → 7.77 ms (1.91x) |
| `sum` | 1,000,000 | 43.50 → 17.69 ms (2.46x) | 47.75 → 22.11 ms (2.16x) |
| `sum` | 25,000,000 | 283.67 → 197.05 ms (1.44x) | 328.73 → 242.34 ms (1.36x) |
| `min` | 1,000,000 | 14.59 → 9.68 ms (1.51x) | 18.31 → 13.59 ms (1.35x) |
| `min` + `max` | 1,000,000 | 14.04 → 9.57 ms (1.47x) | 28.86 → 13.63 ms (2.12x) |
| `min` + `max` | 25,000,000 | 176.05 → 130.96 ms (1.34x) | 300.75 → 175.30 ms (1.72x) |

Over every 50M-row case (one and two keys, 0 and 10% nulls, 200 to 25M groups), best of the rounds:
`sum` and `mean` 1.41x-2.66x (array API) and 1.34x-2.33x (plan runner); `min` + `max` 0.98x-1.49x and
1.29x-2.12x; `min` or `max` alone 0.96x-1.51x and 0.95x-1.35x. `count`, an Int64 `sum` and a Float32
`sum` in the same runs are 0.96x-1.04x. The kernels alone at 50M rows: `sum` 10.62 → 3.48 ms (200
groups), 23.78 → 9.02 ms (10,000), 39.78 → 14.17 ms (1M), 227.11 → 110.43 ms (25M); `min` + `max`
10.64 → 5.91 ms (1M) and 82.82 → 35.65 ms (25M).

Polars `MetalEngine()` (`Benchmarks/results/polars_engine_groupby_float64_2026-09-30.csv`, three
alternating rounds, every result equal to Polars'): at 50M rows the Float64 `min` + `max` grid cases
the default runs on the GPU are 1.19x-2.59x faster (1M groups, one key: 48.36 → 18.67 ms) and `(l)`
Float64 `sum` + `mean` is 46.12 → 29.68 ms. At 2M rows six case-engine pairs were slower in that run;
timed again alone, eight alternating rounds of 100 runs
(`Benchmarks/results/polars_engine_groupby_float64_retime_2026-09-30.csv`), none is slower in both
best and median: the Float64 `min` + `max` cases are 1.18x-1.38x faster cold (`g1minmax10k` 1.90 →
1.43 ms, `g2minmax10k` 2.28 → 1.82 ms, `t7` 2.18 → 1.58 ms), the Int64 `g1sum1M` 1.04x, and the
default's `g1minmax200` and `t4`, which it does not run on the GPU at 2M rows, 0.98x-1.02x. Through the DataFusion rule forced on at 50M rows
(`datafusion/results/datafusion_float64_groupby_2026-09-30.csv`, one run of each build): Float64
`sum` and `avg` 1.33x-1.82x faster than before (1M groups: 58.03 → 31.95 ms against DataFusion
alone's 55.51 ms), `MIN` + `MAX` 1.06x-1.19x.

**The 10M-row, 10,000-group min/max rows.** In the alternating run and in eight further rounds of
100 calls each (`Benchmarks/results/groupby_float64_retime_2026-09-30.csv`), ten Float64 `min`/`max`
rows at 10M rows were slower in both best and median, nine of them at 10,000 groups (best
0.60x-0.92x, median 0.71x-0.88x). Their first run after 500 ms of idle was faster than before in
the same file (one key, `max`: 8.60 → 7.09 ms). The time they lost was in the GPU's state after a
loop of the new kernel, not in the kernel
(`Benchmarks/results/groupby_float64_minmax_state_2026-09-30.csv`, eight rounds per build):

- A `count` over the same grouping, the same code in both builds, took 0.40 and 0.32 ms (best; one
  key, two keys with nulls) after a loop of the earlier `max`, 0.46 and 0.49 ms after a loop of the
  new one, and 0.41 and 0.38 ms on the new build with the earlier `max` kernel in place of the new
  one; run first in the process, before any `max`, it took 0.44-0.47 ms in every build.
- With the same 50M-row Int64 group sum run before every timed call in both builds, the new `max`
  was faster: 1.38 → 1.12 ms best and 1.57 → 1.36 ms median (one key), 1.28 → 1.09 ms and 1.50 →
  1.34 ms (two keys, 10% nulls); the `count` beside it was unchanged (0.42 → 0.40 ms and 0.38 →
  0.39 ms best).
- In fresh processes that time only `am.group_by(keys).max(values)`, at 5M, 10M and 20M rows and
  3,000, 10,000 and 30,000 groups, warm loop and with the same heater, the new build was faster in
  17 of 18 cells in best and 17 of 18 in median (10M rows, 10,000 groups, warm loop: 1.85 → 1.72 ms
  best, 3.21 → 2.39 ms median); no cell was slower in both.

**To improve.** Through the DataFusion rule, a Float64 `MIN`/`MAX` carries four helper aggregates per
extreme (the NaN, value, zero and negative-zero counts, each a sum over an `if_else`), which take
most of the plan: 36.88 ms at 200 groups from 50M rows, where the plan runner's own `min` + `max`
takes 7.34 ms. At 25M groups from 50M rows the grouping itself (`am.group_by(keys)` with a `count`)
is 110.12 ms of the 197.05 ms `sum`.

**Tests.** `GroupSumExactTests` (Swift) checks sums and means against an independent correctly
rounded reference (CPython's `msum` partials with its half-even correction, and a residual check of
each mean against its two neighbours) over 1 to 100,000 groups on both sides of the private-table
limit, with null and out-of-range keys, null values, magnitudes spread over 2^-300 to 2^300,
hand-built groups of cancellation, signed zeros, subnormals, overflow, ties and every special value
on both paths, cancellation-heavy groups of 10^15 to 10^17 terms that cancel, 2^24 + 3 groups, and
`min`/`max` over Float64 (NaN, both zeros, inf), Int64 near its maximum, Float32 and UInt32 against a
host reference. `python/tests/test_group_sum_exact.py` checks the array API and the plan runner
against the exact `Fraction` sum and mean of each group, and two differential rows,
`group_by_sum_correctly_rounded` and `group_by_mean_correctly_rounded`, check every shape of the
matrix bit for bit against the exact value rounded once.

## Round 15 (2026-09-27): a count that depended on its neighbours

**What.** `count(expr)` in a plan's `group_by` gave three different answers for one column depending
on the other aggregates of the same `group_by`. Alone, or next to aggregates the one-kernel group-by
takes, a count of a Float64 column was right and a count of a utf8 column was rejected ("group_by
count of a utf8 expression"). Next to a `sum`, `mean`, `min` or `max` over Float64, the count of the
Float64 column failed ("group-by over Float64 values: cast to Float32 first"), and the count of the
utf8 column returned the group's row count with its nulls: 4 for `["a", null, "b", null]`, where SQL's
`count` is 2. No error was raised for the second. Found by a differential grid of group-by plans.

**Cause.** One aggregate the one-kernel group-by cannot take (its sums finish with 32-bit atomics)
sends the whole `group_by` through `GroupBy`'s per-aggregate kernels (`Executor.aggregateOne`). That
path's `count` switched on the column's type: the numeric types went to `GroupBy.count(_:)`, whose
shared accumulate kernel refuses a Float64 `MetalArray` before it looks at the kind, although a count
reads only the validity bitmap; every type the switch did not list fell to `default:` and to
`GroupBy.count()`, the row count. The one-kernel path reads numeric expressions only and rejected the
rest. The same `default:` counted rows in the streaming group-by's per-batch count, for boolean,
decimal, dictionary and nested columns (667 per group where 500 rows were non-null), and the dense-key
streaming result left out a key whose rows were all null in the first counted or summed column,
because it chose its groups by that aggregate's count of values instead of rows.

**Fix.** A count is computed from validity and nothing else, for every type. `GroupBy.count(validity:)`
runs the existing count kernel with a validity bitmap and the key column's buffer standing in for
the values, which the count kind never loads; `GroupBy.countValid(_: AnyMetalArray)` takes a column of
any type through `AnyMetalArray.logicalValidity()`, which is the column's own bitmap, or, where Arrow's
nulls live elsewhere, one built on the host: all null for the `null` type, the code's and the
dictionary entry's validity for a dictionary, the run's for run-end encoded, the selected child
slot's for a union. The per-aggregate path, the streaming group-by, the C ABI's `hash_count` and the
dense `am_group_by` count all call it. A `count` no longer decides the group-by path; in the one-kernel
path a count of an expression the kernel does not read as a number is taken the same way next to the
kernel, and the whole-table aggregate does the same. The dense-key streaming result keeps a row count
per key, taken from a count the batch already has when its column holds no null.

**Cost.** Over 10,000,000 rows and 200 groups the one-kernel group-by shapes (a filtered `sum` and
row count, `count` of an int32 or Float64 column with 10% nulls, an int32 `sum` with both counts) run
in the same time as before, 2.0-2.7 ms, each within 0.1 ms of its time before (best of 15 per run,
best of six runs of each library, interleaved). On the per-aggregate kernels a Float64 `sum` next to
`count(*)` is 5.63 ms against 6.34, next to the count of an int32 column 6.37 ms against 5.33, inside
those shapes' run-to-run spread of 5.3-9.0 ms; next to the count of the Float64 column, which failed
before, 5.43 ms. The dense-key streaming group-by now counts a column once per batch where `sum` and
`count` share it (21.0-21.2 ms against 21.6-21.8 for the two over a column with nulls); a `sum` alone
over a column with nulls adds one row count per batch, 20.5-21.5 ms against 19.3-20.3 ms best, over
10,000,000 rows in 1,048,576-row batches.

**Tests.** `GroupByCountTests` (Swift) checks 27 column types, the ten numeric types, boolean, utf8,
utf8 views, binary, date32, timestamp, decimal128, list, struct, two dictionaries (null codes, null
entries), fixed-size binary, float16, `null`, run-end encoded, a sparse union and an extension type,
at null fractions 0, 0.1 and 1.0, alone, next to a Float64 `sum`, `mean`, `min` and `max`, and next
to an int32 sum and a row count, against a count of the validity each column was built from; and the
same types through the whole-table aggregate with and without a filter, the dense `GroupBy` API at 13
and 5,000 keys, computed expressions with nulls, empty input, the streaming group-by with and without
a dense key count, and the fused stream join aggregate on both sides.
`python/tests/test_group_count.py` checks the plan runner, `am.group_by(...).count(...)`, the dense
`count_values` and the streaming group-by against pyarrow's `is_valid` per key, and
`rust/arrowmetal/tests/plan.rs` checks the two minimal cases above.

## Round 14 (2026-09-26): what the 0.3.0 reviews and sweeps found

**TL;DR**

- The Polars engine imported NumPy for one reciprocal. NumPy comes with neither the wheel, pyarrow nor
  Polars, so on a clean install the engine raised `ModuleNotFoundError`.
- One 790 KB Snappy dictionary page took about 94 ms on the GPU, most of a 1,000,000-row read.
- A grid of 2^32 threads or more wraps on the GPU, which made group-bys over 2^24 groups or more return
  nulls: [round 13](#round-13-2026-09-26-a-grid-of-232-threads-runs-almost-none-of-them).
- The engine conformance grid found three answers of the Polars engine different from Polars'. All three
  now match.
- The engine's first crossover table was fitted from a sweep that started at a load average of 50.8.
  The sweeps rerun from a quiet start moved it: String shapes from 5,000,000 rows, the rows/2
  group-count bucket never taken, and every shape taken from 1.5 times its fitted crossover.

Each change below is in 0.3.0.

### 1. The Polars engine needed NumPy

Polars divides a column by a scalar as `x * (1 / c)`, with the reciprocal rounded in the result type,
and the engine emits the same multiply so the bits match ([POLARS.md](POLARS.md#where-the-answers-would-differ-and-what-the-engine-emits-instead)).
It computed that reciprocal with NumPy. NumPy is installed by none of the wheel, pyarrow and Polars, so
in a fresh virtualenv holding the wheel and its `polars` extra, tier 4 raised `ModuleNotFoundError`.
Found by installing the wheel alone (`scripts/check_wheel.sh`). The reciprocal is now computed with
Python floats, identical to the NumPy result on 800,046 checked values, zeros, infinities, NaN and
subnormals among them, and `scripts/check_wheel.sh` runs all four tiers with NumPy not installed
([POLARS.md](POLARS.md#install)).

### 2. One Snappy page took 94 ms on the GPU

A 1,000,000-row file with an `int64` and a `float64` column, written with pyarrow's defaults, read in
102-106 ms, 94 ms of it one 790 KB dictionary page. Every Snappy page was decompressed by one GPU SIMD
group, and one worker walks a page's token stream from start to end however the page is decoded. A
token-dense page is tens of thousands of tokens of a few bytes: on the GPU it costs 130 ns per output
byte on a SIMD group, on one host core 0.42 ns ([PARQUET.md](PARQUET.md#decompression)). The first fix
decoded dictionary pages, and dispatches of at most 16 pages, on the host (20-36 ms for that file). The
reader now splits the Snappy and LZ4 pages of a read between the host and the GPU by how token-dense each
page header says it is, from measured costs on each side, and runs both at once: the 10,000,000-row,
7-column pyarrow-default Snappy file reads in 25.4 ms against 67.5 ms, its token-dense pages, the
790 KB dictionary pages among them, on the host.

### 3. A grid of 2^32 threads

[Round 13](#round-13-2026-09-26-a-grid-of-232-threads-runs-almost-none-of-them).

### 4. Three engine answers that differed from Polars'

The engine conformance grid (`python/tests/engine_report.py`) compares the Polars engine with
`lf.collect()` bit for bit over generated shapes, dtypes, null patterns and sizes from 0 to 100,000
rows ([COVERAGE.md](COVERAGE.md#engines)). It found three answers that differed, each now the same as
Polars' ([POLARS.md](POLARS.md#where-the-answers-would-differ-and-what-the-engine-emits-instead)):

- **A column of one row.** Polars divides by a scalar as a reciprocal multiply and multiplies a float
  column by -1 as a negation, except over a column of one row, where the scalar and the column have the
  same length and it computes element-wise. The engine now chooses by the input's row count: from the
  in-memory frame, or counted when the plan runs if a filter or join decides it.
- **-0.0 and 0.0.** Polars' `min` and `max` order -0.0 below 0.0; ArrowMetal treats the two as equal.
  The engine counts the zeros of the sign Polars prefers and takes the sign from that count.
- **A Float32 `mean`.** Polars accumulates it in Float64. The engine casts the column to Float64 first,
  and the two match bit for bit.

The recorded run: 12,597 Polars cases, 12,392 pass, 32 documented (float summation order), 0
unclassified (`Benchmarks/results/engine_conformance_2026-09-25.csv`).

### 5. Crossovers fitted on a busy machine

The default `MetalEngine()` takes a subtree from its measured crossover against the faster Polars engine
([POLARS.md](POLARS.md#which-translatable-subtrees-it-runs-the-defaults)). The first table was fitted
from `Benchmarks/results/polars_engine_crossover_2026-09-26.csv`, a sweep that started at a load
average of 50.8 and ran between 13.6 and 58.2 (`polars_engine_crossover_2026-09-26_conditions.txt`).
There the String sort (o) was 1.66x the faster Polars engine at 1,000,000 rows, 0.69x at 2,000,000 and
0.78x at 5,000,000. The final sweep started at 3.2 (`bench_conditions_2026-09-26-final.txt`), and
there (`polars_engine_crossover_2026-09-26-final.csv`) the same case is 0.98x, 1.20x and 1.37x,
growing with size. The table in 0.3.0 is fitted from the final sweep, with three rules on the fit:

- **String shapes from 5,000,000 rows** (`STRING_FLOOR`), with a 35% margin where a numeric shape has
  15% (`MARGIN`). A String shape's advantage grows slowly with size, and below 5,000,000 rows the String
  sorts, (o) 1.2x and (p) 1.38x at 2,000,000, sit inside the benchmark's noise band of 0.62x to 1.31x.
- **The rows/2 bucket is never taken** (`UNTAKEN_BUCKETS`). The sweep is not monotone for group-bys over
  a quarter of the rows or more: the one-key count over 0.43 times as many groups as rows is 0.78x,
  1.9x, 4.04x, 2.21x and 1.18x the faster Polars engine from 2,000,000 to 50,000,000 rows.
- **1.5 times the fit** (`HEADROOM`). The fit interpolates between sizes measured 2 to 2.5 times apart,
  and shapes just past their crossover were within run-to-run noise of Polars.

In the default benchmark on that table (`Benchmarks/results/polars_engine_bench_2026-09-26-final3.csv`,
194 case-size pairs) the default took 62 pairs, 42 of them group-bys, every one ahead of the faster
Polars engine, 1.21x to 9.42x.

## Round 13 (2026-09-26): a grid of 2^32 threads runs almost none of them

**What.** `am.group_by([key]).sum(values)` and `.mean(values)` over Float64 values, and over Float32
values (summed in Float64), came back null for 2^24 groups and more; 2^24 - 1 groups were right. At
exactly 2^24 groups every group was null, at 2^24 + 1 every group but one, at 2^24 + 2^20 the first
2^20 groups were right and the other 2^24 null. No error was raised. `product` and `list` were wrong
for the same groups. Count, min and max over every type, and the Int64 sum and mean, were right. Found by
the Polars engine's crossover sweep.

**Cause.** Those aggregates reduce one group per 256-thread threadgroup and dispatched a
`(groups, 1, 1)` grid of threadgroups (`seg_reduce`, `Kernels/Segmented.swift`). On the M4 Max the
thread count of a grid dimension, threadgroups times threads per threadgroup, is held in 32 bits: at
2^24 threadgroups of 256 threads it is 2^32 and wraps, and the dispatch runs `groups mod 2^24`
threadgroups. A standalone kernel shows it: 2^24 threadgroups of 32 threads all run, 2^24 of 256 run
none, 2^23 of 512 run none, 2^25 + 3 of 128 run 3. The groups that no threadgroup reached kept a
count of zero and came back null. The atomic path (count, 32-bit min/max, integer sum and mean) and
the two-pass 64-bit min/max dispatch over rows, not groups, and never reach that width. The same
one-threadgroup-per-group grid served the segmented 64-bit min/max, product, the list gather, the
variance passes over groups of 32 rows or more on average, and the per-group run sort of the counting
sort.

**Fix.** `Dispatch.perGroup` dispatches every one-threadgroup-per-group kernel. Below 2^32 threads the
grid is the same `(groups, 1, 1)`; from there it is folded into rows of 65,536 threadgroups, and the
kernels read their group as `(tgid.y << 16) + tgid.x`. Reading the row width from
`threadgroups_per_grid` instead nearly doubled the time of a light per-group kernel (a gather over 10M
groups of 5 rows: 122.6 ms against 64.5 ms), so the width is a power of two known to the kernel. The
50M-row group-by rows run in the same time as before.

**Tests.** `GroupByGridFoldTests` (Swift) checks sum, mean, count, min and max over Float64, Float32
and Int64 at 2^24 - 1, 2^24, 2^24 + 1 and 2^24 + 2^20 groups against a host reference, and runs every
per-group kernel with the fold forced on at 150,000 groups against the plain grid.
`python/tests/test_group_by_2_24.py` checks the same aggregates, product and list at the same four
group counts against pyarrow's `Table.group_by`, and the lazy-plan Float64 sum and mean at 2^24 groups.

## Round 12 (2026-09-24): what the 0.2.0 reviews found

**TL;DR**

- The 0.2.0 work (the CPU/GPU router, the Polars engine, DuckDB plan rewriting, the GPU CSV and NDJSON
  readers, the Parquet reader's nested and page-index gaps, Delta Lake and Iceberg tables, and IPC view
  types, big-endian files and the tensor extension) was reviewed before it merged.
- The reviews found seven bugs older than this work and one in pyarrow, filed as
  [apache/arrow#51491](https://github.com/apache/arrow/issues/51491) ([UPSTREAM.md](UPSTREAM.md)).
  Everything below is fixed on `main` with a test.
- The Python suites had been loading a library built on 2026-09-07, left in the checkout by a wheel
  build. The test run now pins the library it loads.

### 1. The Python suites tested a stale library

The Python package looks for its library in three places, and a copy bundled by a wheel build wins over
the development build. The 2026-09-07 wheel build left such a copy in the checkout, so from 2026-09-08
to 2026-09-24 the Python suites and the differential ran against the 2026-09-07 build. The Swift suite
was unaffected. It surfaced when the IPC work's new Python tests failed with "view columns unsupported",
which only the old library says.

Fixed by pinning the path: `ARROWMETAL_LIB` names the build under test, and the run stops unless Python
reports loading exactly that file ([../python/README.md](../python/README.md)). Rerun against the fresh library, the
differential gave the same counts as before (39,069 cases, 0 unclassified) and the Python suites passed,
so nothing had regressed behind the stale copy. The 2026-09-17 crossover sweep was rerun the same way
on 2026-09-24 (`Benchmarks/router_check.py`; [DESIGN.md](DESIGN.md), "CPU/GPU router").

### 2. A float literal of 2^63 or more ended the process

The fused expression compiler converted every float literal to Int64, even when the target was a float,
and Swift's `Int64(Double)` traps outside its range. `int64_column >= 1e19` or `x * 4.49e307` ended the
host process. Found while reviewing the Polars engine, through a division by a tiny literal. Float targets
no longer compute the integer; an integer-typed literal that does not fit is an error that names it.

### 3. Two same-named columns came back as one twice

With no explicit projection, a stream looked every column up by name, so the second of two columns
called `a` was replaced by a copy of the first: `am.scan_ipc(...).collect()` returned wrong data and no
error. Found while reviewing the IPC reader, on a union fixture. A batch with duplicate names now stays
positional; batches with unique names keep the fused path.

### 4. Skips recorded as failures

Twenty-three IPC test assertions called the pyarrow helper inside `XCTAssertEqual`. When no pyarrow
interpreter exists, the helper throws `XCTSkip`, and a skip thrown inside the assertion's autoclosure is
recorded as a failure. Found while reviewing the router: a full-suite run showed 12 such failures on a clean
environment. The helper now runs before the assertion.

### 5. Four engine bugs the Polars engine had been working around

The Polars engine's differential suite compared every plan it could take against Polars and found four
wrong answers inside the engine. That work routed around them and pinned each with a strict expected
failure; they were then fixed at the root, each with a test that fails on the old code:

- A string filter wrote the bytes a null row still held over the next row's slot, turning "banana"
  into "xanana". The gather kernel now copies the output slot's own width.
- A Boolean column lost its null count through a sort inside a batch: results of an array whose count
  was still pending copied the placeholder. They now count their own nulls after the flush.
- A filter refused any batch that also carried a date32 column, because the expression compiler bound
  every column, read or not. It binds only the ones the query reads.
- A string sort returned wrong rows when a column held a null and a value of eight bytes or more: the
  null partition read the sort's indices on the CPU before the GPU had written them. It first waited for
  them, which made such a sort slower than the old, wrong one; the partition now runs on the GPU, as a
  bit in the sort's own prefix keys, and there is nothing to wait for.

The JSON reader's review found a fifth wrong answer, in new code: an explicit `timestamp[ns]` outside the
years 1678 to 2261 wrapped around instead of raising, as pyarrow raises. It raises now.

### 6. pyarrow drops NaN rows when statistics are present

Found while reviewing the Parquet reader: ArrowMetal's own page skipping dropped a page holding a NaN under
`!=`, because writers leave NaN out of min and max. After the fix, a test showed pyarrow doing the same
at row-group level. Reproduced in plain pyarrow 25.0.1: `x != 5.0` over `[5, NaN, 5, 5]` returns `[nan]`
in memory and from a file without statistics, and `[]` from a file with them; `~(x <= 10.0)` over
`[1, NaN, 10]` behaves the same way. The cause is in
`ParquetFileFragment::EvaluateStatisticsAsExpression`, unchanged on Arrow's main, and is distinct from
the 2023 fix for NaN inside min or max (#28074). The same class of bug was fixed on our side in Delta
partition pruning. Filed as [apache/arrow#51491](https://github.com/apache/arrow/issues/51491)
([UPSTREAM.md](UPSTREAM.md)).

### 7. Two Parquet changes that broke the Delta reader

The Parquet work added a filter value for integers above INT64_MAX and changed how a MAP column is
described. With the Delta and Iceberg work in the same tree, the build failed, and once built, 54
lakehouse tests failed: Delta checkpoints could no longer be read. Both are fixed on main, with tests.

## Round 11 (2026-09-20): The three Metal findings reported to Apple

**TL;DR**

- Metal has no public issue tracker, so the three findings this log had accumulated about Metal itself
  went to Apple through Feedback Assistant on 2026-09-20. Feedback Assistant is private, so this entry
  and [APPLE_REPORTS.md](APPLE_REPORTS.md) are the readable record; the rows move in
  [UPSTREAM.md](UPSTREAM.md) when Apple answers.
- FB24858110: the Feature Set Tables promise "the full set of 64-bit atomic operations" on Apple9; the
  shading language compiles exactly two, min and max. FB24858160: pipeline creation fails at random on
  the "Apple Paravirtual device" of GitHub's macOS runners, 110 times in one test process. FB24858235: a
  running kernel sees CPU stores to shared memory only sporadically, so a persistent GPU worker cannot
  be built; filed as a suggestion for a system-scope primitive.
- Re-verifying each report against its own attachments before filing changed two of them. The rule
  held: nothing was sent that the attachments do not show.

### 1. 64-bit atomics: the documentation and the compiler disagree (FB24858110)

Apple's Metal Feature Set Tables list 64-bit atomics for the Apple9 family, which the M4 series belongs
to, and say in a footnote that "the full set of 64-bit atomic operations is supported on all platforms
starting with Apple9". On an M4 Max the compiler accepts exactly two 64-bit atomic operations, on device
memory only: `atomic_min_explicit` and `atomic_max_explicit` on `device atomic_ulong`, void-returning,
relaxed order. Fetch-add, fetch-sub, the value-returning min and max, exchange, compare-exchange, load
and store are all compile errors at every language version from 3.1 to 4.0, and there is no signed
`atomic_long`. The shipped header and the shading-language specification's own table of 64-bit atomic
functions agree with the compiler, not with the Feature Set Tables. A 64-bit add cannot be built from
parts either, because there is no 64-bit compare-exchange to loop on.

Measured with a short Swift program that compiles one tiny kernel per operation at three language
versions and prints each verdict, then runs the two that compile over ten million elements and checks
the results. It was run again on the day of filing with the same outcome as two weeks earlier.

What it costs here: every 64-bit accumulation carries a workaround. An int64 sum per group is a split
32-bit add with an explicit carry; a hash table publishes a 32-bit row index because a 64-bit key cannot
be published atomically; grouped variance and standard deviation, which accumulate in software binary64,
run a counting sort by group first because they cannot use per-group atomic accumulators. On the
matrix's 50-million-row, 1,000-group rows the grouped sum, where the carry trick suffices, is 3.8x ahead
of pyarrow's threaded engine; grouped variance, which cannot use it, is level with it.

Asked: on Apple9, the operations the footnote already claims; failing that, a per-family statement of
which `ulong` operations exist, in place of "the full set".

### 2. Pipeline creation fails at random on GitHub's virtual GPU (FB24858160)

On GitHub-hosted macOS runners the Metal device is an "Apple Paravirtual device". Compiling a shader
library from source succeeds every time; creating a compute pipeline for a function from that library
then fails sporadically with the text "Compilation failed" and nothing else, for ordinary kernels that
succeed on the same runner image in other runs and in the same process moments later. Rounds 4 and 5
below are where this was first met.

Measured from two complete CI logs on runner image macos-15-arm64 20260829.0321.1 with Xcode 16.4. In
the first run one test process saw 110 pipeline-creation failures across 23 distinct kernels and no
library-compilation failure, the most frequent a filter kernel, 64 times. In the second the engine
retried each creation once after 20 ms and skipped GPU tests on virtual devices; the test step passed
by creating few pipelines, and the benchmark step, which still created them, failed on a group-by
kernel even with the retry. On real Apple silicon the same source has never produced one such failure.

What it costs: GPU code cannot be validated on GitHub-hosted macOS runners. CI here proves the build,
the language interop and the CPU paths; every GPU test and benchmark runs on local hardware.

Asked: deterministic pipeline creation for a function from a library the same device just compiled; or
an error that names the unsupported construct and a device property that reports the limitation before
any work is submitted.

### 3. A running kernel cannot reliably see CPU stores to shared memory (FB24858235)

Small operations are dominated by the fixed cost of a dispatch, about 60 microseconds of submission and
completion notification on this machine. The standard escape elsewhere is a persistent worker: one
long-running kernel spinning on a ring of work descriptors in shared memory, so that submitting work is
a CPU store. That needs a CPU store to become visible to a running kernel within a bounded time.

Measured with a standalone program that compiles one kernel per memory qualifier the shading language
offers, from plain pointers through `volatile`, `coherent(device)`, device-scope atomics and
device-memory barriers; one threadgroup spins on a doorbell word in a shared-storage buffer for three
seconds while the CPU increments it every 5 ms. Five runs across three days: in one, no store was seen
under any qualifier; in another, 24 of 433 under one qualifier and a handful under others, the first
after 160 ms or more; in another, a single store of 598. The behaviour is consistent with visibility
arriving only when a cache line happens to be evicted, which nothing in the language can request.
`coherent(device)` is the widest scope the language has and there is no synchronisation call that can
run mid-dispatch. The full study is [RESIDENT.md](RESIDENT.md).

Asked: a system-scope coherence primitive with a stated visibility bound; or a documented, explicitly
costed way for a running kernel to observe CPU stores without a command-buffer boundary; or, failing
both, a documented submit-and-notify path under 20 microseconds, which removes the motive.

### What changed before filing

Re-running the coherence probe, which on 2026-09-08 had seen no CPU store at all under any qualifier,
saw 24 of 433 in one of three runs that day, a single store in another and none in the third. "Never"
became "sporadic and unbounded": the more accurate claim, and the harder one to dismiss. The paravirtual
draft quoted an error code that appears in neither attached CI log; it was replaced by what the logs
show, "Compilation failed" with an empty userInfo. Verifying every sentence of a report against its
own attachments caught both.

## Round 10 (2026-09-08): arrow-go's span iterator, found while designing its aggregates

**TL;DR**

- Arrow Go's `compute` package has no aggregate functions ([apache/arrow-go#1296](https://github.com/apache/arrow-go/issues/1296)). The maintainer asked for them; a design note with a tested prototype of `sum`, `mean`, `min_max`, `min`, `max`, `count`, `any` and `all` is [on the issue](https://github.com/apache/arrow-go/issues/1296#issuecomment-5593008436).
- Reviewing that prototype the way this project reviews its own kernels found a bug in arrow-go itself: `ArraySpan.SetSlice` carries a stale null count into the next slice, so `and_kleene` and `or_kleene` return `false` where the answer is null once `ExecCtx.ChunkSize` is below the input length, on main and in v18.7.0. Reported as [#1305](https://github.com/apache/arrow-go/issues/1305), fixed by [#1306](https://github.com/apache/arrow-go/pull/1306).
- Two more things the C++ reference does that a straightforward Go port gets wrong: C's `fmin`/`fmax` rank -0.0 below +0.0, and float sums are pairwise, not a left fold.
- Two documentation pull requests from the same day, [#1302](https://github.com/apache/arrow-go/pull/1302) and [#1303](https://github.com/apache/arrow-go/pull/1303), were merged within hours.

### The setting

`compute.GetFunctionRegistry()` in arrow-go v18.7.0 holds 85 functions, none of aggregate kind
(89 on main: 76 scalar, 8 vector, 5 meta). `FuncScalarAgg` exists as a kind; `execInternal`
returns `ErrNotImplemented` for it. The Go binding here therefore uses plain loops as its reference
for reductions ([GO.md](GO.md)). The maintainer's reply to the report was that the kernels had been
on his list for a long time and he would love to see them; the design note was promised before
any code, so the interface would be one he wants to maintain.

### Research steps

1. Read the C++ `ScalarAggregateKernel` (`kernel.h`), the executor (`exec.cc`), the options
   (`api_aggregate.h`) and the kernels (`aggregate_basic.cc`, `aggregate_basic.inc.cc`,
   `aggregate_internal.h`) at f251bc3, and the Go `compute` package at 67ef40b, and write the design.
2. Check the note row by row against pyarrow built from that C++ commit and against the Go
   identifiers it names, compiling the proposed constraint change.
3. Build the prototype from the corrected note, run arrow-go's own lint under the Go version it
   pins, and cross-check every expected test value against pyarrow.
4. Attack the code: sliced inputs at bitmap-hostile offsets, forced small `ChunkSize`, executor pool
   reuse, the race detector, a checked allocator on every call, bad options and input kinds.
5. Differential fuzz: 31,760 cases across the twelve input types, plain, sliced, chunked and scalar,
   under every `skip_nulls`/`min_count` combination, compared bit for bit with pyarrow.
6. Fix what was found, rerun everything, and only then post.

### Finding 1: the note's own errors, caught before posting

| The note said | What C++ does (pyarrow from the same commit agrees) |
|---|---|
| `mean` of an empty input follows the same rule as `sum` (0 with `min_count=0`) | NaN for numeric and boolean inputs, 0.0 for the null type |
| in `min_max`, NaN never wins | true against a value; when every non-null value is NaN the result is NaN, because the state starts at NaN |
| `mode`, `quantile` and `tdigest` are scalar aggregates for later | `mode` and `quantile` are vector functions; `tdigest` returns an array, so a scalar-returning Finalize cannot implement it |
| no other options type in the package inverts a field for the zero value | `ArithmeticOptions.NoCheckOverflow` does, and `TakeOptions` uses a constructor |

### Finding 2: signed zeros in `min`, `max` and `min_max`

One divergence in the first 31,760 fuzz cases, 2,884 times, all the same cause. C's `fmin` and
`fmax`, which the C++ kernels use, order -0.0 below +0.0, so pyarrow returns `min` -0.0 and `max`
+0.0 whenever both appear, in any order. A comparison-based port keeps whichever zero came first:

```go
// before: b < a is false for (-0.0, +0.0), so the incumbent stays
case b < a:
    return b
// after: the sign is observable, so it is a case of its own
case b < a:
    return b
case b == 0 && a == 0 && math.Signbit(float64(b)):
    return b
```

After the fix the rerun matched on every case.

### Finding 3: float sums are pairwise

C++ `SumArray` sums floats pairwise in blocks of sixteen, the same algorithm numpy uses, and
`mean` accumulates in float64 for every input type. A plain left fold over 100,001 values gives
75247.35643756675 where pyarrow gives 75247.35643756694; the port of the pairwise loop matches bit
for bit, and the test that proves it is in the prototype.

### Finding 4: `ArraySpan.SetSlice` carries a stale null count

Every prototype kernel returned wrong answers once `ExecCtx.ChunkSize` was smaller than the input:
`count` of 339 valid values in 493 gave 193 at chunk size 1, 490 at 2, 467 at 4. The cause is not
in the kernels. `iterateExecSpans` reuses one `ArraySpan` per argument and advances it with
`SetSlice`, which on main reads:

```go
if a.Type.ID() != arrow.NULL {
    if a.Nulls != 0 {
        if a.Nulls == a.Len {
            a.Nulls = length          // "all null" carries over
        } else {
            a.Nulls = array.UnknownNullCount
        }
    }                                 // "no nulls" carries over too
} else {
    a.Nulls = length
}
```

That is right only while `Nulls` describes the whole array. A kernel that calls
`UpdateNullCount()` on the shared span stores the current slice's count; the next `SetSlice` then
treats the following slice as all valid or all null. Two kernels on main do exactly that, and the
reproducer shows it without any new code:

| ChunkSize | `and_kleene([T, null, T, null, F, T, null, T], [null, T, T, F, null, null, T, T])` |
|---|---|
| default | `[null null true false false null null true]` |
| 1 | `[null false true false false null false true]` |
| 2 | `[null null true false false false false true]` |
| 3 and above | correct |

`or_kleene` fails the same way; the `SetSlice` code is identical in v18.7.0. The C++
`ArraySpan::SetSlice` never carries a count when a validity bitmap is present, and the fix is that
rule:

```go
if a.Type.ID() != arrow.NULL {
    if a.Nulls != 0 || len(a.Buffers[0].Buf) != 0 {
        a.Nulls = array.UnknownNullCount
    }
} else {
    a.Nulls = length
}
```

With it, both Kleene kernels pass at every chunk size and the whole compute suite still passes.
The prototype's kernels, independently of that fix, count each span from its bitmap and never
write to the shared span, with a test at chunk sizes 1 to 65.

### What shipped, and what is open

- Merged the same day: [#1302](https://github.com/apache/arrow-go/pull/1302) (allocator alignment and
  which allocator to use when C keeps a buffer, closing [#1297](https://github.com/apache/arrow-go/issues/1297))
  and [#1303](https://github.com/apache/arrow-go/pull/1303) (what the compute registry holds).
- Open: [#1296](https://github.com/apache/arrow-go/issues/1296) (design approved 2026-09-21) and the
  framework plus count/sum pull request [#1336](https://github.com/apache/arrow-go/pull/1336);
  [#1305](https://github.com/apache/arrow-go/issues/1305) with its fix
  [#1306](https://github.com/apache/arrow-go/pull/1306) by singhpratech.
- Status of each is on the [Upstream tracker](UPSTREAM.md).

Logs, scripts and raw outputs of every run above (the audit reports, the build and lint log, the
fuzz fixtures and results, the reproducer and its output on both commits) are kept with the
release records.

## Round 9 (2026-09-06): radix select, and the LSD radix sort's blocking at small n

**TL;DR**

- A GPU radix select needs two passes over the column, not three, when the histogram keeps its
  counts per sub-block: at 50M rows that is 1 ms saved out of 3.
- After selection, ordering the ~200k survivors was the bottleneck, and the cause was a fixed
  4096-element block in the LSD radix sort. Halving the block until there are at least 64 blocks took
  `argsort` of 20k from 1.99 ms to 0.41 ms and is most of why `top_k(100)` went from 5.6 ms to 2.9 ms.
- The selected bin's size depends entirely on the key distribution's top byte. Float keys concentrate
  in a handful of bins, so a 50M Float64 median varies from run to run while Int64 keys are stable.

### The setting

A GPU radix select normally costs three passes over the column: histogram the top digit, count how
many rows fall in the selected range per block, then scatter them in order. This round built the
select behind `top_k`, then measured where the time went once selection itself was cheap: the final
ordering of the candidates, and the size of the bin the selection leaves behind.

### Research steps

1. Lay the histogram out per sub-block, so the count pass is redundant and the select is two passes.
2. Make the sub-block a simdgroup's slice rather than a threadgroup's, removing the barriers from the
   scatter.
3. Time the final `argsort` over the ~200k candidates; trace its cost to the fixed
   `elemsPerBlock = 4096`.
4. Halve the block size until there are at least 64 blocks; remeasure `argsort` at 20k and 200k rows
   and `top_k(100)` end to end.
5. Run the histogram and compaction a second time over the compacted candidates.
6. Read the benchmark generator's key distribution, and the top byte of a float64 key, against the
   bin sizes observed.

### Finding 1: the two-pass trick

The count pass is redundant if the histogram keeps its counts **per sub-block** instead of only
globally. The digit-major table `counts[digit * subBlocks + sub]` is simultaneously the global
histogram (summed over sub-blocks) and the offset table the scatter needs (summed over digits <=
target). At 50M rows that is 1 ms saved out of 3.

### Finding 2: one sub-block per simdgroup, not per threadgroup

Making the output granularity a simdgroup's slice rather than a threadgroup's removes every
`threadgroup_barrier` from the scatter: the rank of a selected row within its slice is
`simd_prefix_exclusive_sum` over 32 lanes plus a running base each lane computes identically. The
cost is an 8x larger count table (256 * 8 * groups words, ~8 MB at 50M rows), which is noise next to
the 400 MB the pass reads anyway.

### Finding 3: the final sort was the bottleneck, and it was a blocking bug

After selection there are only ~200k candidates left to order, but `argsort` of 200k UInt64 measured
2.2-2.9 ms, the time of sorting 800k. The cause was `elemsPerBlock = 4096` fixed: 200k rows is 49
threadgroups, 20k rows is *five*, on a GPU with 40 cores, and `radix_scatter` does an O(TG) rank
loop per element that nothing else can overlap. Halving the block size until there are at least 64
blocks (inputs above ~256k rows are untouched, so the 50M argsort is unchanged) gave:

| Input | Before | After |
|---|---|---|
| `argsort` of 20k | 1.99 ms | 0.41 ms |
| `argsort` of 200k | 2.20 ms | 1.34 ms |
| `top_k(100)` | 5.6 ms | 2.9 ms |

The block-size change is most of the `top_k(100)` improvement. The rule as it stands in
`Sources/ArrowMetal/Kernels/Sort.swift`:

```swift
func blockPlan(_ rows: Int) -> (elemsPerBlock: Int, blocks: Int) {
    var e = Swift.max(4096, ((rows + 127) / 128 + 255) / 256 * 256)
    while e > 256 && (rows + e - 1) / e < 64 { e >>= 1 }
    return (e, Swift.max(1, (rows + e - 1) / e))
}
```

### Finding 4: refine on the compacted array, not the column

The selected bin is ~n/256 rows, which still dominates the final ordering. Running the same
histogram + compaction *again* over the compacted candidates (a few hundred thousand keys,
microseconds) shrinks it by another 256x. One extra round trip, and it takes the survivors under the
2048 pairs a single-threadgroup bitonic sort can order in one dispatch.

### Finding 5: benchmark data hides skew

`rng.integers(-(2**62), 2**62)` spreads over only 128 of the 256 top-digit bins, so the candidate bin
is n/128, not n/256. Worth remembering when reading a selection benchmark: the bin size, and
therefore the final sort, depends entirely on the key distribution's top byte.

### Finding 6: a float key's top byte is the exponent, so float columns are the skewed case

The top byte of a float64 key is the sign plus seven exponent bits, so a column of uniform doubles
concentrates in a handful of bins rather than spreading over 256, and the bin holding the wanted rank
can be a large fraction of the column. When it is over the compaction budget the search narrows
another digit over the column instead, one more full pass, which is why the same 50M Float64 median
varies from run to run depending on exactly where the rank lands, while Int64 keys are stable.
Raising the budget so the big bin gets compacted instead is faster still, but a 25M-row bin needs
300 MB of scratch for a 400 MB column, and a median already 80x faster than pyarrow (4.61 ms against
368.88 ms at 50M rows, `Benchmarks/results/full_matrix_2026-09-07-parallel.csv`) is not worth a 75%
memory overhead.

### What shipped

- The two-pass select with per-simdgroup sub-blocks, planned by `TopK.radixPlan(n:)` in
  `Sources/ArrowMetal/Kernels/RadixSelect.swift`; the single-dispatch bitonic limit is `bitonicCap = 2048`
  in the same file.
- The block-size rule `blockPlan` in `Sources/ArrowMetal/Kernels/Sort.swift`, quoted above.
- The second refinement pass over the compacted candidates; the compaction budget left as it is, for the
  memory reason in Finding 6.
- The median figures are the `quantile(float64, 0.5)` rows at 50M in
  `Benchmarks/results/full_matrix_2026-09-07-parallel.csv`.

## Round 8 (2026-09-06): a threadgroup atomic read that is not uniform (`top_k`)

**TL;DR**

- One cell of a 13,000-case differential run diverged and then passed on every rerun. A stress test
  against a CPU oracle reproduced it at roughly 1 in 900 calls.
- Every thread of `topk_select` read the threadgroup atomic count `held` itself with a relaxed load;
  one simdgroup out of eight occasionally saw a newer value, so the sentinel fill left holes of zeroed
  memory that read as `(key 0, row 0)` and sorted to the front.
- Fix: thread 0 reads `held` once per chunk into a plain threadgroup variable between two barriers.
  The cost is below the noise floor: 3.11 / 3.31 / 3.38 ms with the fix against 3.22 / 3.30 / 4.37 ms
  without it for `top_k(100 of 20M Int64)`.
- Rule: never let a barrier-carrying branch, or a loop bound whose strides must tile a range, come from
  a per-thread `atomic_load_explicit(..., memory_order_relaxed)`.

### The setting

**Symptom.** One cell of a 13,000-case differential run diverged and then passed on every rerun:
`top_k / float64`, 100,003 rows, 30% nulls, k = 17, descending. `result[0]` was row 0 where pyarrow
says row 35254. A randomised stress against a CPU oracle (`TopKTests.testStressAgainstCPUOracle`)
reproduced it at roughly 1 in 900 calls across every key type, both directions, n from 32k to 1M and
k from 1 to 1024. The failure always looked the same: a run of zeros at the front of the result, the
true answer after them.

### Research steps

1. A stress test comparing against a CPU oracle (not against `argsort`, which shares the sort keys),
   with pool-churning kernels in between; this is `testStressAgainstCPUOracle` in
   `Tests/ArrowMetalTests/TopKTests.swift`.
2. Poison the candidate buffers before the dispatch, to prove the kernel really wrote those zeros
   rather than leaving stale bytes.
3. Add a debug buffer carrying each threadgroup's `held`, arrival count and compaction count, which
   showed the holes were inside `[0, c)` and `[c, cap)` in simdgroup-sized runs.
4. Instrument the kernel to count the holes per threadgroup.
5. Fix, measure the cost, and check the other `atomic_load_explicit` sites for the same pattern.

### Finding 1: a relaxed atomic load is not uniform across the threadgroup

**Root cause.** `topk_select` keeps a per-threadgroup candidate buffer in threadgroup memory with an
atomic count `held`. Every thread read that count itself:

```metal
threadgroup_barrier(mem_flags::mem_threadgroup);
uint c = atomic_load_explicit(&held, memory_order_relaxed);   // Kernels/TopKSource.swift
...
for (uint i = c + lid; i < cap; i += TG) { bufKey[i] = keyMax; bufRow[i] = TK_NOROW; }
```

`c` bounds the range each thread pads with sentinels, and the union over `lid` covers `[c, cap)`
**only if every thread has the same `c`**. It does not: on an M4 Max, one simdgroup out of eight
occasionally comes back with a newer value than the rest. A relaxed atomic load is not ordered by the
preceding barrier the way a plain threadgroup read is, so it can be satisfied late, after other
threads have already run ahead and incremented `held`. The threads holding the larger `c` start their
stride later and the slots they should have covered are never written. Instrumenting the kernel showed
7 to 63 such holes per threadgroup, clustered at 32 and 64: one and two simdgroups.

### Finding 2: what a hole does to the answer

Those holes hold zeroed pool memory, which reads as `(key 0, row 0)`. That pair is the *minimum* of
the `(key, row)` order, so the bitonic sort moves it to the front of the buffer and it becomes the
block's answer; the host then sees row 0 as the top candidate. Worse, if a block accumulates k or more
holes the compaction sets its threshold to `bufKey[k-1] = (0, 0)`, which nothing can beat, and the
block is blind for the rest of the scan. The same read also decides `if (c + TG > cap)`, a branch
containing `threadgroup_barrier`, so a non-uniform `c` was undefined behaviour in its own right.

### What shipped

- **Fix.** Thread 0 reads `held` once per chunk into a plain `threadgroup uint shared_c` (clamped to
  `cap`) between two barriers, and every thread takes `c` from there; the branch and the fill ranges
  are now uniform by construction. The kernel as it stands in `Sources/ArrowMetal/Kernels/TopKSource.swift`:

```metal
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lid == 0u) shared_c = min(atomic_load_explicit(&held, memory_order_relaxed), cap);
threadgroup_barrier(mem_flags::mem_threadgroup);
uint c = shared_c;
if (c + TG > cap) {
    for (uint i = c + lid; i < cap; i += TG) { bufKey[i] = \(keyMax); bufRow[i] = TK_NOROW; }
```

- In addition, the buffer is filled with sentinels once at kernel entry, so a slot no one writes reads
  as "no row" and the host drops it instead of it masquerading as row 0.
- **Cost.** One extra `threadgroup_barrier` per 256-row chunk, and one `cap`-element sentinel fill per
  block (2 to 8 strided stores per thread, once). Below the noise floor: `top_k(100 of 20M Int64)` on
  M4 Max runs 3.11 / 3.31 / 3.38 ms with the fix against 3.22 / 3.30 / 4.37 ms without it. The pass is
  still about one read per row.
- **Test.** `TopKTests.testStressAgainstCPUOracle`, the stress against the CPU oracle.
- **Rule.** In MSL, never let a value that decides a barrier-carrying branch, or a per-thread loop
  bound whose strides must tile a range, come from a per-thread
  `atomic_load_explicit(..., memory_order_relaxed)`. Broadcast it through a plain threadgroup variable
  between barriers. The other `atomic_load_explicit` sites (`SortSource`, `GroupBySource`,
  `JoinSource`, `StringExtraSource`) were checked: each reads a slot the calling thread owns, so none
  of them tiles or branches on a shared count.

## Round 8b (2026-09-06): the differential matrix over the whole type surface

**TL;DR**

- Extending `python/tests/test_differential.py` to every type ArrowMetal imports (45 columns, 181
  operations, 33,156 cases at the time; 212 operations and 39,069 cases today) turned up three bugs
  and one unreproduced crash **in pyarrow 25.0.1**, not in ArrowMetal.
- Each has a workaround in the harness and a test that fails if a later pyarrow fixes it.
- Two properties of the harness itself: time-of-day cases have to drop their null rows, and a 38-digit
  `Decimal` has to be read off `as_tuple().digits`.

### The setting

The differential matrix compares every ArrowMetal operation against pyarrow over every imported type.
Extending it to the whole type surface is what surfaced the pyarrow behaviours below. They are
recorded here because the harness has to work around them, and each has a test that fails if a later
pyarrow fixes it.

### Research steps

1. Extend the matrix to 45 columns and 181 operations, 33,156 cases (212 operations and 39,069 cases
   today).
2. Trace one segfault back from the faulting frame, `Array.nbytes` inside the harness's own array
   cache, to the preceding `pc.year_month_day` call, which cost an hour.
3. Compare `pc.utf8_normalize` against Python's `unicodedata` and the Unicode annex for each form.
4. Run `pc.pairwise_diff` on a sliced column whose values are not an arithmetic progression, then the
   same slice check on the boolean fill and replace kernels.
5. Give each oracle a workaround, and pin each pyarrow behaviour with a test that fails when it changes.

### Finding 1: a single unreproduced segfault after `pc.year_month_day`

A **single unreproduced segfault** was observed a few allocations after `pc.year_month_day` on a
`timestamp[s]` array with nulls; it has not recurred. It landed in whatever unrelated call happened
next: the faulting frame was `Array.nbytes` inside the harness's own array cache, which cost an hour
to trace back. Out of caution the matrix does not use the two struct-valued temporal kernels as
oracles: it compares `iso_calendar` and `year_month_day` field by field against `pc.iso_year`/
`pc.iso_week`/`pc.day_of_week` and `pc.year`/`pc.month`/`pc.day`, which is a stronger check anyway.

### Finding 2: `pc.utf8_normalize` ignores its `form` option

`pc.utf8_normalize` **ignores its `form` option**: NFC and NFKC come back decomposed, so its NFC is
NFD. Python's `unicodedata` and ArrowMetal agree with each other and with the Unicode annex; the
matrix uses `unicodedata` as the oracle:

```python
for form in ("NFC", "NFKC", "NFD", "NFKD"):
    got.append(arrow(x.utf8_normalize(form)))
    expected.append(pa.array([None if s is None else unicodedata.normalize(form, s)
```

Pinned by `test_pyarrow_utf8_normalize_ignores_its_form_option`.

### Finding 3: `pc.pairwise_diff` ignores `ArrowArray.offset`

`pc.pairwise_diff` **ignores `ArrowArray.offset`**: on a sliced column it reads the values buffer
from the start and answers with the wrong rows. Only visible when the values are not an arithmetic
progression, which is why it hid for a while. The matrix hands that oracle a materialised copy while
ArrowMetal still gets the slice, so the case remains a test of the offset handling:

```python
def _materialised(src):
    """The same values with the Arrow offset folded away."""
    if src.offset == 0:
        return src
    return pa.concat_arrays([src.slice(0, 0), src])
```

Fixed upstream in pyarrow 26.0.0; the workaround stays only while 25.0.1 is the pinned version.

### Finding 4: the same offset bug on boolean columns

`pc.fill_null_forward`, `pc.fill_null_backward` and `pc.replace_with_mask` have the same offset bug
on a **boolean** column (the values bitmap, not the validity one). Same mitigation.

### Finding 5: `pc.add` on time-of-day validates the values under the validity bitmap

`pc.add` on a `time32`/`time64` column validates the *values under the validity bitmap*, so a null
row whose hidden value would leave `[0, 86400)` makes the oracle raise even though the row is null.
The generator deliberately puts real numbers under the null bits, so the time-of-day cases have to
drop their null rows rather than mask them.

### Finding 6: a 38-digit `Decimal` and the default context

A `Decimal` that came out of a 38-digit column cannot be scaled with `Decimal.scaleb` or multiplied
by `10 ** scale` under the default decimal context; 28 digits of precision silently round it. Read
the unscaled magnitude off `as_tuple().digits` instead:

```python
def _unscaled_magnitude(value):
    return int("".join(str(d) for d in value.as_tuple().digits) or "0")
```

### What changed

- The oracles in `python/tests/test_differential.py`: field-by-field comparison for `iso_calendar`
  and `year_month_day`, `unicodedata` for normalisation, `_materialised` copies for `pc.pairwise_diff`
  and the boolean fill and replace kernels, dropped null rows for time-of-day addition, and
  `_unscaled_magnitude` for decimals.
- The pinning tests in the same file: `test_pyarrow_utf8_normalize_ignores_its_form_option`,
  `test_pyarrow_pairwise_diff_ignores_the_array_offset` and
  `test_pyarrow_boolean_fill_null_forward_ignores_the_array_offset`; each fails when a later pyarrow
  changes the behaviour it pins.

## Round 7 (2026-09-06): strings and sort

**TL;DR**

- Generated MSL written through a shell heredoc needs `\(K)` (one backslash) for Swift interpolation.
- LSD radix sort with 8-bit digits, 4096-element blocks and a stable in-chunk ranking is correct
  across all types and sizes; descending order must invert keys, not reverse the ascending result.
- MurmurHash3 x86_32 matches the reference vectors; four collisions among 100k 32-bit hashes is the
  birthday bound, not a bug.

### The setting

The string kernels (hashing) and the LSD radix sort, with their MSL generated from Swift source
written through a shell heredoc.

### Research steps

1. Generate MSL through a shell heredoc and check what reaches the Metal compiler.
2. Run the LSD radix sort across all types and sizes, in both directions, and check ties.
3. Check MurmurHash3 x86_32 against its reference vectors and count collisions among 100k hashes.

### Finding 1: heredoc interpolation

Generated MSL written through a shell heredoc must use `\(K)` (one backslash) for Swift
interpolation; a doubled backslash reaches the Metal compiler as literal text. Same newline rule as
before for `#define` (Round 6, Finding 3).

### Finding 2: the radix sort, and descending order

LSD radix sort with 8-bit digits, 4096-element blocks and a stable in-chunk ranking is correct across
all types and sizes; descending order must invert keys rather than reverse the ascending result, or
ties flip.

### Finding 3: MurmurHash3 reference vectors

MurmurHash3 x86_32 reference vectors (seed 0):

| Input | Hash |
|---|---|
| "" | 0 |
| "a" | 0x3c2569b2 |
| "abc" | 0xb3dd93fa |
| "hello" | 0x248bfa47 |

Four collisions among 100k 32-bit hashes is normal (birthday bound), not a bug.

### What shipped

- The heredoc rule for generated MSL.
- Key inversion for descending sorts in `Sources/ArrowMetal/Kernels/Sort.swift`.
- The reference vectors above are checked in `Tests/ArrowMetalTests/StringTests.swift`, from "" -> 0
  through "hello" -> 0x248bfa47.

## Round 6 (2026-09-06): software IEEE-754 double on the GPU

**TL;DR**

- Add, subtract and multiply on `ulong` with 3 guard bits and sticky are bit-exact against Swift's
  `Double` for 1M random pairs including subnormals, signed zeros, infinities and NaN.
- A float-seeded Newton division was within 1 ulp only 93% of the time and wrong for subnormal
  inputs; restoring long division on the significands (57 quotient bits) replaced it.
- Generated MSL fragments that start with a `#define` must begin with a newline, because Swift
  multi-line strings drop the final newline.
- The Float64 GPU sum accumulates with `d_add` in tree order and differs from a sequential CPU sum
  only by normal floating-point reordering.

### The setting

Float64 arithmetic implemented in MSL on `ulong` bit patterns, tested against Swift's `Double`.

### Research steps

1. Implement add, subtract and multiply with 3 guard bits and sticky; compare against `Double` on 1M
   random pairs including subnormals, signed zeros, infinities and NaN.
2. Implement division as a float-seeded Newton iteration and measure how often it is within 1 ulp.
3. Replace it with restoring long division on the significands, and test on ratios above and below 1.
4. Fix the `#define` that glued to the previous line in the generated source.
5. Sum Float64 on the GPU with `d_add` and combine the partials on the CPU.

### Finding 1: add, subtract and multiply are bit-exact

Add, subtract and multiply implemented on `ulong` with 3 guard bits and sticky are bit-exact against
Swift's `Double` for 1M random pairs including subnormals, signed zeros, infinities and NaN
(`d_finish` handles normalisation, subnormal shift-with-sticky, round-to-nearest-even, and rounding
carry).

### Finding 2: division

A float-seeded Newton iteration was within 1 ulp only 93% of the time and wrong for subnormal inputs.
Replaced with restoring long division on the significands (57 quotient bits). Two bugs on the way: one
extra quotient bit shifted every result by 2x, and the restoring loop needs `rem < mb` before the first
step (take the first quotient bit explicitly). Lesson: test division on ratios above and below 1.

### Finding 3: a `#define` glued to the previous line

A preprocessor `#define` glued to the previous line (`}#define`) because Swift multi-line strings drop
the final newline. Generated MSL fragments that start with a directive must begin with a newline.

### Finding 4: the Float64 sum

Float64 sum on the GPU accumulates with `d_add` in tree order; per-threadgroup partials are combined
on the CPU in `Double`. Results differ from a sequential CPU sum only by normal floating-point
reordering.

### What shipped

- `d_add`, `d_sub`, `d_mul`, `d_div` and `d_finish` in `Sources/ArrowMetal/Kernels/DoubleMath.swift`.
- Tests in `Tests/ArrowMetalTests/DoubleMathTests.swift`: `testAddSubMulBitExact`, `testDivBitExact`
  and `testDoubleSumOnGPU`.
- The newline rule for generated fragments that start with a directive.

## Round 5 (2026-09-06): pipeline creation on the paravirtual GPU, and lengths that stay on the device

**TL;DR**

- The paravirtual GPU on GitHub runners fails `makeComputePipelineState` for arbitrary trivial kernels
  with no diagnostic while identical kernels pass in the same run. Mitigation: one retry, and GPU tests
  `XCTSkip` on devices whose name contains "Paravirtual".
- Kernels now read their element count from a device buffer, so a filter followed by a sum is one
  round trip.

### The setting

Round 4 left two things open: `makeComputePipelineState` failures on GitHub's paravirtual GPU, and the
second round trip a `sum()` on a batched filter result paid to learn the filtered length.

### Research steps

1. Enable full compiler logs on the paravirtual GPU and compare the kernels that fail with those that
   pass in the same run.
2. Add one retry on pipeline creation, and skip GPU tests on virtual devices.
3. Move the element count into a device buffer and bind a pending filter's GPU-written total as that
   buffer.

### Finding 1: the paravirtual GPU fails arbitrary trivial kernels

With full compiler logs enabled, the paravirtual GPU on GitHub runners fails `makeComputePipelineState`
for arbitrary trivial kernels (`bitmap_not`, `cast_kernel`) with no diagnostic while identical kernels
pass in the same run. It is the virtual Metal stack, not a construct. The retry in
`Sources/ArrowMetal/MetalContext.swift`:

```swift
do { p = try device.makeComputePipelineState(function: fn) }
catch {
    // Virtualised GPUs (GitHub's "Apple Paravirtual device") fail pipeline creation sporadically; retry once.
    usleep(20_000)
    p = try device.makeComputePipelineState(function: fn)
}
```

### Finding 2: lengths that stay on the device

Kernels now read their element count from a device buffer (`device const uint* nPtr`). A pending
filter result binds its GPU-written total as that buffer, so compare / arithmetic / cast / bitmap ops /
another filter / a reduction can consume it inside the same command buffer with worst-case dispatch
sizes. Reductions still sync to read partials, but a filter followed by a sum is now one round trip.

### What shipped

- One retry on pipeline creation, quoted above.
- GPU tests `XCTSkip` on devices whose name contains "Paravirtual" (`ARROWMETAL_FORCE_GPU_TESTS=1`
  overrides): `requireRealGPU()` in `Tests/ArrowMetalTests/TestSupport.swift`. CI validates the
  build, interop and CPU paths; GPU correctness runs on real hardware.
- `device const uint* nPtr` as the element count of every kernel, so pending lengths flow on the GPU.

## Round 4 (2026-09-06): the paravirtual GPU on GitHub runners

**TL;DR**

- GitHub's `macos-15` Apple silicon runners expose an "Apple Paravirtual device" GPU with 3 CPU cores,
  ~16 GB/s for a sum. Use CI for correctness only; never publish its numbers.
- The paravirtual GPU failed `makeComputePipelineState` for kernels that pass on M4 Max;
  `ARROWMETAL_DEBUG_SHADERS=1` now dumps the generated MSL and the compiler log on failure.
- The ~120 µs per-call floor is the round trip; batching is what removes it, and a `sum()` on a batched
  filter still paid a second trip, which Round 5 removed.

### The setting

The first CI runs on GitHub's hosted Apple silicon runners, after the Round 3 push, and the per-call
latency of a single dispatch on M4 Max.

### Research steps

1. Measure a sum on the `macos-15` runner's GPU.
2. Read the pipeline creation failures on the runner for kernels that pass on M4 Max, and add a way to
   see the generated source and the compiler log.
3. Try `makeCommandBufferWithUnretainedReferences` plus a spin-wait against the per-call latency.
4. Count the round trips of a `sum()` on a batched filter result.

### Finding 1: the runner GPU

GitHub's `macos-15` Apple silicon runners expose an "Apple Paravirtual device" GPU with 3 CPU cores:
~16 GB/s for a sum. Use CI for correctness only; never publish its numbers.

### Finding 2: pipeline creation fails on the runner

The paravirtual GPU failed `makeComputePipelineState` for kernels that pass on M4 Max (round 3 push).
`ARROWMETAL_DEBUG_SHADERS=1` now dumps generated MSL and the full compiler log on failure; CI sets it.

### Finding 3: the per-call floor

`makeCommandBufferWithUnretainedReferences` plus a spin-wait did not measurably reduce per-call
latency; the ~120 µs floor is the round trip. Batching is what works: chains pay it once.

### Finding 4: a reduction on a batched filter

A `sum()` on a batched filter result still costs a second round trip because the reduction needs the
filtered length on the CPU. Done in Round 5 above: kernels read `n` from a device buffer so pending
lengths flow on the GPU.

### What shipped

- `ARROWMETAL_DEBUG_SHADERS=1`, read in `Sources/ArrowMetal/MetalContext.swift`; the CI job in
  `.github/workflows/ci.yml` runs `ARROWMETAL_DEBUG_SHADERS=1 swift test`.
- The rule that CI numbers are never published.

## Round 3 (2026-09-06): integer division by zero, and the group-by atomics

**TL;DR**

- Metal integer division by zero returns an unspecified value; ArrowMetal defines it as 0 in both the
  GPU kernels and the CPU reference, and `Int.min / -1` wraps.
- Threadgroup-privatised group-by with 32-bit atomics reaches ~300 GB/s for up to 1024 keys, the same
  rate as a plain sum; device atomics at 100k keys halve it.
- Crossing the C boundary from Python, and exporting a result back to pyarrow, cost nothing
  measurable; importing pyarrow buffers is one memcpy.

### The setting

The arithmetic kernels, the group-by, and the Python binding's path across the C boundary.

### Research steps

1. Observe what Metal returns for integer division by zero, and define it the same way in the GPU
   kernels and the CPU reference.
2. Measure the threadgroup-privatised group-by at up to 1024 keys and with device atomics at 100k keys.
3. Time a call across the C boundary from Python, with and without exporting the result to pyarrow.
4. Check the alignment of pyarrow's buffers against what a zero-copy import needs.

### Finding 1: integer division by zero

Metal integer division by zero returns an unspecified value (observed 1 on M4 Max). ArrowMetal now
defines it as 0 in both the GPU kernels and the CPU reference, and `Int.min / -1` wraps instead of
trapping. The generator in `Sources/ArrowMetal/Kernels/KernelSource.swift`:

```swift
/// Integer division by zero yields 0 (defined here, matching the CPU reference); overflow wraps.
if name == "div" && isInt {
    sclArray = "(b[i] == (\(T))0) ? (\(T))0 : a[i] / b[i]"
```

### Finding 2: the group-by atomics

Threadgroup-privatised group-by with 32-bit atomics reaches ~300 GB/s for up to 1024 keys, the same
rate as a plain sum: the atomics are not the bottleneck at that key count. Device atomics at 100k keys
halve it.

### Finding 3: the C boundary from Python

Crossing the C boundary from Python costs nothing measurable per call (ctypes overhead is ~10 µs);
exporting a result back to pyarrow costs nothing measurable:

| Call | Time |
|---|---|
| without export | 3.55 ms |
| with export | 3.57 ms |

### Finding 4: importing pyarrow buffers

Importing pyarrow buffers is one memcpy: pyarrow's allocator is 64-byte aligned, not page aligned.

### What shipped

- Division by zero defined as 0 in the GPU kernels and the CPU reference; `Int.min / -1` wraps.
- The threadgroup-privatised group-by.

## Ecosystem research (2026-09-06)

**TL;DR**

- No existing Arrow library computes on Metal; the pieces that exist are types, IPC, device buffer
  wrappers and tensors.

**What exists**

- `apache/arrow-swift` v21: types, IPC, Flight, C Data Interface. No compute kernels, nothing Metal.
- `arrow-nanoarrow` device extension: C wrapper for `ARROW_DEVICE_METAL` buffers, no compute.
- cuDF: CUDA only. MLX: unified-memory tensors, zero-copy `MTLBuffer` access, no nulls or columnar
  semantics.
- A DuckDB extension with Metal aggregates exists (gpudb); it does not use Arrow buffers.

**The device interface**

- The Arrow C Device Data Interface defines `ARROW_DEVICE_METAL = 8` and expects `MTLEvent*` as
  `sync_event`.

## Toolchain

**TL;DR**

- Run tests with Xcode's `DEVELOPER_DIR` and always in release as well as debug.
- Swift 6.3.3 miscompiles a `withUnsafeBytes` closure in a generic `throws` function under `-O`
  (swiftlang/swift#90477, open); `MetalArray.init(_:)` uses a plain loop instead.
- The `macos-15` runner's Swift 6.1 refuses closures Swift 6.3 accepts; release builds shorten object
  lifetimes; `posix_memalign` memory is not zero.

**Running the tests**

- XCTest is not in the Command Line Tools. Run tests with
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`.
- Always run `swift test -c release`.

**Compilers**

- Swift 6.3.3 miscompiles a `withUnsafeBytes` closure inside a generic `throws` function under `-O`
  when the caller is in another module: the closure clobbers the error register around an Objective-C
  message send, so the function reports a phantom error and the caller crashes retaining it
  (swiftlang/swift#90477, open). `MetalArray.init(_:)` uses a plain element loop instead; the
  Metal-free reproducer is in UPSTREAM.md.
- GitHub `macos-15` runners come with Xcode 16.4 / Swift 6.1, which refuses to type-check dense
  closures that Swift 6.3 accepts. Keep test expressions simple.

**Memory**

- Swift release builds shorten object lifetimes to last use: a raw pointer taken from a buffer object
  can outlive the object. Use `withExtendedLifetime` or the closure accessors.
- `posix_memalign` memory is not zero for small blocks (recycled heap); only fresh mmap pages are
  zero.

## Metal: what the M4 Max gives a kernel

**TL;DR**

- 32 KB threadgroup memory, SIMD width 32, unified memory, no 64-bit atomic add from MSL.
- Runtime `makeLibrary(source:)` needs no `metal` toolchain; `makeBuffer(length:)` buffers are not
  page aligned.
- Memory-bound element-wise kernels run at ~375 GB/s against ~350 GB/s on the 16-core CPU; reductions
  and compaction favour the GPU by 2x to 3x.

**The hardware**

- Apple M4 Max: 32 KB threadgroup memory, SIMD width 32, unified memory, no 64-bit atomic add from MSL
  (`atomic_ulong` fetch_add fails to compile; min and max do). 64-bit sums use split 32-bit atomics
  with carry.

**The API**

- Runtime `makeLibrary(source:)` works without the `metal` toolchain. Errors carry line numbers of the
  generated source.
- `makeBuffer(length:)` buffers are heap sub-allocated and not page aligned; `bytesNoCopy` requires
  page alignment of pointer and length.

**Bandwidth**

- Memory-bound element-wise kernels run at ~375 GB/s on M4 Max once allocation overhead is removed;
  the 16-core CPU reaches ~350 GB/s on the same loops. Reductions and compaction favour the GPU by 2x
  to 3x.
- Fusing the predicate into the filter's counting pass avoids materialising a boolean array.
