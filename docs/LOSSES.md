# Where ArrowMetal needs to improve, and why

The project's bar is 3x over the fastest CPU idiom on every operation at scale. This page is the
honest remainder: every measured row of the full matrix where ArrowMetal is **slower** than a CPU
library, grouped by the measured cause. Nothing here is rounded in ArrowMetal's favour; a row leaves
this page only when a rerun of the matrix moves it.

The matrix now measures every CPU library twice: its plain eager idiom, and the most parallel idiom
it has for the same answer — `polars-lazy` is the same expression through `pl.LazyFrame` collected on
the in-memory or the streaming engine, `pyarrow-threaded` is an Acero plan over the same values split
into 16 record batches (pandas has no parallel idiom and says so in its own row). "Fastest CPU" is the
best wall time of *all* of those idioms. That is the baseline this page is written against:
`Benchmarks/results/full_matrix_2026-09-07-parallel.csv` ([BENCHMARKS_MATRIX.md](BENCHMARKS_MATRIX.md),
idle M4 Max, 16 cores, 64 GB), with the cores each idiom actually used in
`Benchmarks/results/full_matrix_2026-09-07-parallel_cores.txt`: a median of 11.5 cores for Polars lazy
and 11.1 for pyarrow through Acero, against 1.0 for the eager idioms on most families.

**Against the parallel idioms, of 339 measured rows: 145 at or above 3x, 102 faster but under 3x, 77
slower than the fastest CPU idiom, and 15 with no CPU equivalent.** The 77 are the first section
below, grouped into ten causes. The sections after them are this page as it stood against each
library's *eager* idiom, kept as history.

## Against the most parallel idiom

Each table gives the operation, the row count, ArrowMetal's wall time, the fastest CPU idiom with its
wall time, the cores that idiom actually used (its CPU-ms divided by its wall ms), and the ratio.
Every number in this section is computed from the parallel CSV by
`private/keep/2026-09-07/losses_parallel.py`; nothing is typed by hand. Eight ArrowMetal rows in that
CSV were re-measured on a quieter machine after a regression check flagged them and spliced back in
place — their `note` column says so, and the log is
`private/keep/2026-09-07/matrix_remeasure.log`.

**What these rows are, and what they are not.** Twelve to fifteen cores running a vectorised kernel
over a column in unified memory is a fast, well-engineered thing to be, and on a single pass over a
single column it reaches the same memory the GPU reads at close to the same rate. Most of the 77 are
that: a bandwidth tie, where neither side has a 3x to give. The rest are fixed cost — a Metal command
buffer is 140–170 µs whatever it carries, so below about a million rows the dispatch is the
operation — plus three operations that genuinely do more arithmetic per element than hardware doubles
do, one that never reaches the GPU at all, and the rows where the CPU library returns a view and
ArrowMetal materialises a column. None of these is a row where the CPU is doing something slowly. They are the rows where
having the option of unified memory on the GPU buys you nothing yet, and each one names what would
change that.

### 1. The dispatch floor and the per-call fixed cost, at a million rows and below (16 rows)

| operation | rows | ArrowMetal | fastest CPU idiom | its cores | ratio |
|---|---:|---:|---|---:|---:|
| sum(int64) | 1,000 | 0.11 ms | Polars 0.00 ms | 5.0 | 0.00x |
| sum(int64) | 100,000 | 0.13 ms | Polars 0.01 ms | 1.0 | 0.05x |
| sum(int64) | 1,000,000 | 0.20 ms | Polars 0.08 ms | 1.0 | 0.39x |
| filter(int64 > 0) | 1,000 | 0.15 ms | Polars 0.00 ms | 1.1 | 0.02x |
| filter(int64 > 0) | 100,000 | 0.16 ms | Polars 0.03 ms | 1.0 | 0.17x |
| filter(int64 > 0) | 1,000,000 | 0.23 ms | Polars lazy 0.17 ms | 3.7 | 0.75x |
| group-by sum (1000 keys) | 1,000 | 0.40 ms | pandas 0.08 ms | 1.0 | 0.20x |
| group-by sum (1000 keys) | 100,000 | 0.50 ms | pyarrow Acero 0.26 ms | 4.8 | 0.51x |
| group-by sum (1000 keys) | 1,000,000 | 1.08 ms | pyarrow Acero 0.62 ms | 8.4 | 0.58x |
| upper | 1,000,000 | 2.55 ms | pyarrow Acero 1.68 ms | 12.1 | 0.66x |
| lower | 1,000,000 | 2.63 ms | pyarrow Acero 1.71 ms | 11.7 | 0.65x |
| trim (whitespace) | 1,000,000 | 2.50 ms | pyarrow Acero 1.45 ms | 11.5 | 0.58x |
| is_alpha | 1,000,000 | 0.69 ms | pyarrow Acero 0.54 ms | 10.2 | 0.78x |
| is_in (utf8, 100-value set) | 1,000,000 | 2.28 ms | Polars lazy 1.10 ms | 7.8 | 0.48x |
| slice_codeunits [5:10] | 1,000,000 | 1.27 ms | pyarrow Acero 0.86 ms | 11.0 | 0.68x |
| split_pattern("_") | 1,000,000 | 3.36 ms | pyarrow Acero 2.40 ms | 11.3 | 0.72x |

A Metal dispatch costs 140–170 µs of command-buffer creation, encoding, commit and completion wait
before the first byte is touched ([DESIGN.md](DESIGN.md): an empty kernel is ~116 µs), and the
cheapest dispatch in this CSV — `sum(int64)` over a thousand values — is 0.11 ms, which is that floor
and nothing else. Nine of these rows are the latency family, which exists to measure exactly this.

The seven string rows are the same shape with a larger constant: a string kernel encodes an offsets
pass and a bytes pass, and at a million short strings that fixed cost is most of the call. Every one
of them is ahead at ten million rows in the same CSV — `upper` 2.87x, `lower` 2.91x,
`slice_codeunits` 2.50x, `split_pattern` 2.45x, `is_alpha` 2.20x, `trim` 2.14x, `is_in` 1.48x — so
the break-even is between one and ten million, not somewhere past the sizes measured.

**What would change it.** Nothing removes the floor on this hardware; the persistent-kernel approach
that would is impossible here ([RESIDENT.md](RESIDENT.md)). What moves the break-even down is putting
more work in each command buffer: `MetalContext.batch` (8–32% off per call), the fused expression
compiler (one dispatch for a whole expression tree) and the lazy engine (one command buffer for a
whole plan). The opportunity that would actually retire these rows is plan-level batching inside the
engines that call us, where one dispatch carries a whole query rather than one operator.

### 2. Memory-bound single passes, where twelve to fifteen cores reach the same memory (28 rows)

| operation | rows | ArrowMetal | fastest CPU idiom | its cores | ratio |
|---|---:|---:|---|---:|---:|
| abs (float64) | 10,000,000 | 1.01 ms | Polars lazy 0.74 ms | 10.4 | 0.73x |
| case_when (2 conditions) | 10,000,000 | 2.01 ms | Polars lazy 1.56 ms | 12.7 | 0.78x |
| coalesce (2 int64 columns) | 10,000,000 | 2.01 ms | Polars lazy 1.39 ms | 11.5 | 0.69x |
| divide (float64 / float64) | 10,000,000 | 1.47 ms | Polars lazy 1.07 ms | 11.5 | 0.73x |
| exp (float32) | 10,000,000 | 1.15 ms | Polars lazy 1.13 ms | 12.4 | 0.98x |
| if_else (bool ? int64 : int64) | 10,000,000 | 2.18 ms | Polars lazy 1.32 ms | 11.8 | 0.60x |
| negate (int64) | 10,000,000 | 0.96 ms | Polars lazy 0.77 ms | 9.7 | 0.81x |
| power (float32 ** 2) | 10,000,000 | 1.07 ms | Polars lazy 0.47 ms | 9.5 | 0.44x |
| round (float64) | 10,000,000 | 1.01 ms | Polars lazy 0.86 ms | 10.4 | 0.86x |
| sqrt (float64) | 10,000,000 | 1.29 ms | Polars lazy 0.78 ms | 10.7 | 0.60x |
| case_when (2 conditions) | 50,000,000 | 6.92 ms | Polars lazy 6.52 ms | 13.5 | 0.94x |
| power (float32 ** 2) | 50,000,000 | 1.91 ms | Polars lazy 1.82 ms | 11.4 | 0.95x |
| sqrt (float64) | 50,000,000 | 3.92 ms | Polars lazy 3.42 ms | 12.7 | 0.87x |
| compare array (int64 > int64) | 10,000,000 | 1.23 ms | Polars lazy 0.87 ms | 11.7 | 0.71x |
| drop_null (int64, 10% nulls) | 10,000,000 | 1.56 ms | Polars lazy 0.89 ms | 11.4 | 0.57x |
| filter int64 (30% kept) | 10,000,000 | 0.78 ms | Polars lazy 0.65 ms | 9.3 | 0.84x |
| filter int64 (90% kept) | 10,000,000 | 1.64 ms | Polars lazy 0.97 ms | 10.2 | 0.59x |
| replace_with_mask (30% replaced) | 10,000,000 | 2.54 ms | Polars lazy 0.98 ms | 10.2 | 0.39x |
| compare scalar (int64 > 0) | 50,000,000 | 2.41 ms | Polars lazy 1.91 ms | 11.8 | 0.79x |
| drop_null (int64, 10% nulls) | 50,000,000 | 4.20 ms | Polars lazy 3.81 ms | 13.4 | 0.91x |
| filter int64 (90% kept) | 50,000,000 | 4.24 ms | Polars lazy 3.70 ms | 13.5 | 0.87x |
| replace_with_mask (30% replaced) | 50,000,000 | 10.55 ms | Polars lazy 3.78 ms | 13.4 | 0.36x |
| max(int64, 10% nulls) | 10,000,000 | 0.84 ms | Polars lazy 0.54 ms | 10.0 | 0.65x |
| min(int64, 10% nulls) | 10,000,000 | 0.56 ms | Polars lazy 0.53 ms | 10.4 | 0.95x |
| min_max(int64) | 10,000,000 | 1.57 ms | pyarrow Acero 1.29 ms | 11.1 | 0.82x |
| decimal compare (> scalar) | 10,000,000 | 0.92 ms | Polars lazy 0.82 ms | 11.9 | 0.90x |
| decimal round (2 places) | 10,000,000 | 7.64 ms | Polars lazy 6.61 ms | 14.8 | 0.87x |
| decimal round (2 places) | 50,000,000 | 37.28 ms | Polars lazy 32.44 ms | 14.9 | 0.87x |

This is the largest group and the least dramatic one. Every row is a single pass that reads one or two
columns and writes one, and both sides are moving the same bytes through the same memory controller:
`sqrt` at 50M rows is 204 GB/s on the GPU against Polars lazy's 234, `drop_null` at 50M is 181 GB/s
against 199, both within reach of this machine's ~400 GB/s unified-memory ceiling. Twelve to fifteen
cores get there too, and when they do the ratio is a coin toss decided by the dispatch cost on one
side and the thread-pool wake-up on the other. There is no 3x on this shape for anybody.
[BENCHMARKS_MATRIX.md](BENCHMARKS_MATRIX.md) prints the two bandwidths side by side for every one of
these rows.

Two rows in the table are further behind than the rest and the matrix names why:
`replace_with_mask` at 0.36–0.39x is three passes here — the mask's prefix sum, a gather of the
replacements, then the merge — where pyarrow fuses them into one streaming pass over the column.
`power (float32 ** 2)` at 0.44x is the ordinary tie at 10M (74.8 GB/s against 170) and closes to 0.95x
at 50M (210 against 220).

**What would change it.** Not a faster kernel — the bytes are already moving at bandwidth. What
changes a bandwidth tie is reading the bytes fewer times: fusing the operator into an expression so
the intermediate column is never written, which is what `am.query` and the lazy engine are for, and
keeping the column resident on the GPU so the next operator does not re-read it from a fresh
allocation. `replace_with_mask` is the one row here with a kernel-level fix on the table: one streaming
pass instead of three.

### 3. Software binary64 transcendentals and calendar arithmetic (6 rows)

| operation | rows | ArrowMetal | fastest CPU idiom | its cores | ratio |
|---|---:|---:|---|---:|---:|
| ln (float64) | 10,000,000 | 16.46 ms | Polars lazy 1.81 ms | 13.0 | 0.11x |
| ln (float64) | 50,000,000 | 81.31 ms | Polars lazy 8.46 ms | 14.2 | 0.10x |
| sin (float64) | 10,000,000 | 30.19 ms | Polars lazy 5.06 ms | 13.8 | 0.17x |
| sin (float64) | 50,000,000 | 150 ms | Polars lazy 23.28 ms | 15.1 | 0.16x |
| days_between | 10,000,000 | 9.89 ms | Polars lazy 1.46 ms | 12.4 | 0.15x |
| days_between | 50,000,000 | 46.31 ms | Polars lazy 5.75 ms | 13.8 | 0.12x |

This is the one group where the gap is arithmetic, and it is the price of an explicit decision. Apple
GPUs have no binary64 hardware at all, so `ln`, `sin` and every other double transcendental is
emulated: forty-odd software binary64 operations per element, three to four GPU instructions each.
The answers are worth what they cost — arithmetic and `sqrt` correctly rounded, `exp`/`ln` within
1–2 ulp, trigonometry within 5 (asserted 6) ([EVALUATION.md](EVALUATION.md)) — but a vectorised libm
on twelve to fifteen cores does the same element with hardware doubles. `days_between` is the same
story in integers, and the matrix records the diagnosis: days-from-civil and its inverse are a few
dozen integer operations per row on both sides, so the row is compute bound rather than bandwidth
bound and the GPU's only advantage is its lane count.

Against the *eager* idiom in `full_matrix_2026-09-07.csv` these rows were 1.07x, 1.99x and 1.11x at
50M rows, because one core still lost to the GPU's lane count. Against twelve to fifteen cores of
hardware doubles they are 0.10x, 0.16x and 0.12x, and that is the honest number.

**What would change it.** A better polynomial or a cheaper range reduction — this is a self-contained,
well-specified piece of numerical work with a direct path to a 3x row, and anyone who has done it
before can do it here without touching the rest of the engine. Failing that, a second numerical
contract: a `fast_math` variant that computes the transcendentals in float32 or in a reduced-precision
double, documented as such and never the default, for callers who have said they want it. For
`days_between` specifically, hoisting the calendar conversion out of the per-element loop (both
columns are dates, and the difference in days does not need two civil conversions) is a kernel-level
fix that has been costed but not written.

### 4. A real regular expression, matched on the host (2 rows)

| operation | rows | ArrowMetal | fastest CPU idiom | its cores | ratio |
|---|---:|---:|---|---:|---:|
| match_substring_regex (real regex) | 1,000,000 | 16.58 ms | Polars lazy 2.34 ms | 9.3 | 0.14x |
| match_substring_regex (real regex) | 10,000,000 | 167 ms | Polars lazy 14.57 ms | 14.3 | 0.09x |

A genuine regular expression does not run on the GPU here at all. Only a metacharacter-free pattern
(or `^literal`) takes the GPU path; everything else is NSRegularExpression row by row across 4096-row
chunks, which the matrix records as the diagnosis. It is already spread across the machine — 2,542
CPU-ms over a 167 ms wall at 10M rows is 15 cores — so this is not a threading gap: it is ICU matching
row by row against Polars' own compiled regex engine, at 1.05 GB/s against 12.1. LIKE patterns and
literal substrings never take this path; they are GPU kernels and they win.

**What would change it.** A compiled-automaton kernel: translate the pattern to a DFA on the host,
upload the transition table, and step it per row on the GPU, the way the LIKE and literal kernels
already scan bytes. That is the single largest open win on the strings side. Short of the full engine,
widening the set of patterns that reach the existing GPU path — bounded character classes, anchored
alternations — would take the common cases off the host fallback one shape at a time.

### 5. The views a CPU library returns for free (4 rows)

| operation | rows | ArrowMetal | fastest CPU idiom | its cores | ratio |
|---|---:|---:|---|---:|---:|
| shift (lag 1, int64) | 10,000,000 | 2.37 ms | Polars lazy 0.03 ms | 2.2 | 0.01x |
| shift (lag 1, int64) | 50,000,000 | 3.58 ms | Polars lazy 0.03 ms | 2.2 | 0.01x |
| slice (zero-copy view) | 10,000,000 | 0.00 ms | pyarrow 0.00 ms | 0.0 | 0.40x |
| slice (zero-copy view) | 50,000,000 | 0.00 ms | Polars 0.00 ms | 0.0 | 0.33x |

These two rows are not a computation contest. The matrix measures the default `shift`, which writes a
new contiguous column: 813 MB moved at 50M rows in 3.58 ms is 223 GB/s, this machine's bandwidth, so
the kernel is not the problem. Polars answers with a two-chunk view — a null chunk in front of a slice
of the original — and copies nothing; the parallel CSV records it at 27,826 GB/s, which is how you can
tell from the numbers alone that no data moved. `slice` is pointer arithmetic on every side and the
two `slice` rows are measurement noise on calls of a microsecond; they are in the table because the
rule of this page is that every losing row appears.

ArrowMetal already has the same answer: `shift(by, fill, view=True)` returns a `pyarrow.ChunkedArray`
over the same device memory in 0.04 ms at 10M rows (`Benchmarks/loss_sort_shift_sqrt.py`). It is
opt-in, because a chunked array is not a `MetalArray` and cannot re-enter a kernel without being
combined, and the matrix keeps measuring the default.

**What would change it.** A chunked `MetalArray`: a first-class multi-buffer array every kernel
accepts. That makes the view the default answer for `shift`, and turns slices and concatenations into
pointers too. It is the single change that would retire this cause, and it is a substantial one.

### 6. Temporal field extraction and truncation (8 rows)

| operation | rows | ArrowMetal | fastest CPU idiom | its cores | ratio |
|---|---:|---:|---|---:|---:|
| year | 10,000,000 | 5.55 ms | pyarrow Acero 4.70 ms | 10.1 | 0.85x |
| month | 10,000,000 | 5.56 ms | pyarrow Acero 5.17 ms | 8.8 | 0.93x |
| day | 10,000,000 | 5.99 ms | pyarrow Acero 5.06 ms | 10.2 | 0.84x |
| year | 50,000,000 | 26.82 ms | pyarrow Acero 17.91 ms | 13.7 | 0.67x |
| month | 50,000,000 | 26.91 ms | pyarrow Acero 17.84 ms | 13.9 | 0.66x |
| day | 50,000,000 | 29.11 ms | pyarrow Acero 19.36 ms | 13.9 | 0.67x |
| floor_temporal (day) | 10,000,000 | 1.92 ms | Polars lazy 0.74 ms | 10.7 | 0.39x |
| floor_temporal (day) | 50,000,000 | 8.74 ms | Polars lazy 3.35 ms | 11.9 | 0.38x |

Extracting a calendar field is a civil-from-days conversion per element — a few dozen integer
operations, the same on both sides — so these rows are compute bound, not bandwidth bound: ArrowMetal
runs 22.4 GB/s on `year` at 50M rows against this machine's ~400 GB/s ceiling. When an operation is
arithmetic per element rather than bytes per second, the GPU's advantage is its lane count alone, and
thirteen to fourteen cores through Acero close it. Against the eager idioms the same rows were
comfortable wins — `year` at 50M was 7.0x in `full_matrix_2026-09-07.csv` — and against Acero they are
0.66–0.67x at 50M and 0.84–0.93x at 10M. `floor_temporal` is the same conversion and its inverse,
against a lazy Polars path that reaches 216–239 GB/s doing it.

**What would change it.** Fewer integer operations per row, since that is what the row is made of.
Two candidates, neither measured yet: the divisions in the conversion are all by compile-time
constants, so a multiply-and-shift form of each is available; and `year`, `month` and `day` of the
same column each redo the whole conversion, so a kernel that decomposes once and writes the fields a
caller asked for turns three passes into one. Both are contained changes in the temporal kernels, and
both should be measured before anything is claimed for them.

### 7. `unique` and `value_counts` against a threaded hash aggregation (4 rows)

| operation | rows | ArrowMetal | fastest CPU idiom | its cores | ratio |
|---|---:|---:|---|---:|---:|
| unique (int32, 1000 distinct) | 10,000,000 | 5.90 ms | pyarrow Acero 3.37 ms | 10.9 | 0.57x |
| unique (int32, 1000 distinct) | 50,000,000 | 15.72 ms | pyarrow Acero 13.04 ms | 13.6 | 0.83x |
| value_counts (int32, 1000 distinct) | 10,000,000 | 6.67 ms | pyarrow Acero 3.70 ms | 10.8 | 0.55x |
| value_counts (int32, 1000 distinct) | 50,000,000 | 17.04 ms | pyarrow Acero 14.31 ms | 13.7 | 0.84x |

These four rows are an algorithm mismatch, not a bandwidth one. `unique` and `value_counts` go through
the GPU dictionary pipeline: a full radix argsort of the column, run marks, a scan and a gather — work
proportional to *n* log *n* in passes over the data, whatever the number of distinct values.
Acero answers the same question with one threaded hash pass over the column and a table of a thousand
entries. On a column with a thousand distinct values out of ten million, the hash table is the right
data structure and the sort is not; the CPU-ms say the same from the other side (`value_counts` at
10M: 5.3 ArrowMetal CPU-ms against Acero's 40 — the GPU is doing the work, it is simply doing more of
it). The matrix records the same diagnosis and the same fix.

**What would change it.** The group-by path already has the kernel: `am_group_by_keys` builds dense
ids for a narrow-range integer key with a hash-and-scan pass rather than a sort, and it is what took
`sum by int32 key` past 3x. Routing `unique`, `value_counts` and `dictionary_encode` at a low distinct
count through that instead of through `DictionaryCompute.dictionaryEncoded()` is a plumbing change on
top of a kernel that exists, and it is the clearest short-dated win on this page.

### 8. The grouped moments at a thousand groups (1 row)

| operation | rows | ArrowMetal | fastest CPU idiom | its cores | ratio |
|---|---:|---:|---|---:|---:|
| variance by key (1000 groups) | 50,000,000 | 30.36 ms | pyarrow 29.17 ms | 12.7 | 0.96x |

One row, and it is a tie: 30.4 ms against 29.2. `stddev` on the same shape is the row next to it at
1.24x, on the winning side of the same line. Everything the eager-baseline section below says about
this operation still holds and is still the diagnosis: the moments accumulate in software binary64
because Metal has no 64-bit atomic add, which forces a counting sort by group id in front of the two
moment passes, and the moment kernels give a whole threadgroup to one group so a thousand groups
leaves most of the machine idle. Note that the fastest idiom here is `pyarrow` *eager* — Acero's
threaded grouped variance is slower than its single-threaded one on this shape, so the parallel
baseline did not move this row.

**What would change it.** A simdgroup-per-group accumulator, each lane emulating eight of the 256
logical slots, keeps every group resident and preserves the accumulation order; the measured shape
sweep below prices it at about 6 ms of the 15 the two moment passes cost. It would leave the counting
sort's 10.8 ms untouched, so the honest ceiling for this row on this hardware is around 1.5x, not 3x.
A 64-bit atomic add in the shading language would remove the sort as well; that is an ask of Apple,
recorded in [UPSTREAM.md](UPSTREAM.md).

### 9. String conversions with variable-length output (4 rows)

| operation | rows | ArrowMetal | fastest CPU idiom | its cores | ratio |
|---|---:|---:|---|---:|---:|
| parse (utf8 -> int64) | 1,000,000 | 2.84 ms | pyarrow Acero 0.64 ms | 10.0 | 0.22x |
| parse (utf8 -> int64) | 10,000,000 | 19.59 ms | Polars lazy 3.15 ms | 14.6 | 0.16x |
| to_strings (int64 -> utf8) | 1,000,000 | 1.52 ms | pyarrow Acero 0.76 ms | 10.1 | 0.50x |
| to_strings (int64 -> utf8) | 10,000,000 | 11.95 ms | pyarrow Acero 6.76 ms | 11.1 | 0.57x |

Unlike the other string rows, these two do not turn into wins at ten million: `parse` is 0.22x at 1M
and 0.16x at 10M. The output length of each row depends on the row's value, so the kernel measures the
lengths, prefix-sums them and writes in a second pass, against one streaming pass on the CPU — the
matrix records that diagnosis for all four rows. Two passes over the data against one is the whole of
the gap, and it is the same 2x for any number of cores on the other side.

**What would change it.** A single-pass form: an upper bound on each row's output length is known
from the input type (twenty digits for an int64), so the kernel could write into a conservatively
sized buffer and compact once, or fuse the measure pass into whatever produced the column. That the
two conversion rows are the ones left behind while `upper`, `trim`, `split_pattern` and the predicates
are all ahead at 10M says the shape of the pass is the problem, not the string machinery.

### 10. Whole-query chains against a streaming engine (4 rows)

| operation | rows | ArrowMetal | fastest CPU idiom | its cores | ratio |
|---|---:|---:|---|---:|---:|
| filter two columns + sum | 10,000,000 | 1.94 ms | Polars lazy 0.83 ms | 11.3 | 0.43x |
| filter two columns + sum | 50,000,000 | 3.53 ms | Polars lazy 2.92 ms | 13.9 | 0.83x |
| group-by after filter | 10,000,000 | 3.75 ms | Polars lazy 1.86 ms | 10.7 | 0.49x |
| group-by after filter | 50,000,000 | 6.73 ms | Polars lazy 4.85 ms | 13.7 | 0.72x |

This is the honest cost of the change of baseline, and it lands on the operation the project is
proudest of. Against eager Polars, `filter two columns + sum` at 50M rows was 5.96x; against the same
query given to `pl.LazyFrame` on the streaming engine — which fuses the predicate and the aggregate
exactly as our expression compiler does, and then runs it on fourteen cores — it is 0.83x. Both sides
are now doing the same thing: one pass, no intermediate column. What is left is bandwidth (61.8 GB/s
against 145 at 10M) and the fact that the GPU pays its dispatch once per query while the CPU pool is
already awake. The CPU-ms are still the other half of the picture — 1.7 against 40.7 at 50M — and that
is the argument the wall clock does not make: the same answer, with thirteen cores left for the rest
of the application.

**What would change it.** For `group-by after filter` the matrix names the cost precisely: the
chain's group-by rebuilds the dense key mapping after the filter, and that is most of the measured
time — the filter and the aggregate are each well inside the bar on their own. Carrying the key
mapping through the filter instead of rebuilding it is the fix, and it is a change to the chain, not
to a kernel. For `filter two columns + sum`, the same batching answer as cause 1: `[batched]` already
takes `group-by after filter` at 10M from 3.75 ms to 2.04 ms in this CSV. The larger opportunity is
the one the join page describes — these chains are short because the harness hands us two columns and
asks one question; a plan handed down from Polars or DuckDB is longer, and every operator added to it
amortises against the same dispatch.

## Against each library's eager idiom — history, the 2026-09-07 run

Everything from here down is this page as it stood against `full_matrix_2026-09-07.csv`, the run of
the same matrix earlier the same day, when every CPU library was measured only in its plain eager
idiom — one core on most element-wise and whole-column rows, as the cores table now shows. Those
verdicts are no longer the project's headline numbers and the counts below are that run's, not the
parallel run's. It is kept because the diagnoses in it are still the diagnoses, because several rows
above point back into it, and because the record of what a change was worth should not be edited after
the fact. The sort section at the end is measured against `private/results/sort_gather_2026-09-07/`
and stands on its own data.

Of 946 comparisons in that eager matrix: 822 at or above 3x, 93 faster but under 3x, 31 slower than
the fastest eager CPU library. The 31 fall into six causes. The morning run of the same day
(`full_matrix_2026-09-07-am.csv`) had 50 slower and 115 under 3x; what moved is at the end of the page.

### Slower than the eager CPU library

#### 1. The dispatch floor below a million rows (19 rows)

| operation | rows | ArrowMetal | fastest CPU | ratio |
|---|---:|---:|---:|---:|
| sum(int64) | 1,000 | 0.14 ms | 0.00 ms (Polars) | 0.00x |
| sum(int64) | 100,000 | 0.15 ms | 0.01 ms (Polars) | 0.05x |
| sum(int64) | 1,000,000 | 0.22 ms | 0.07 ms (pyarrow) | 0.31x |
| filter(int64 > 0) | 1,000 | 0.18 ms | 0.00 ms (Polars) | 0.02x |
| filter(int64 > 0) | 100,000 | 0.18 ms | 0.03 ms (Polars) | 0.17x |
| group-by sum (1000 keys) | 1,000 | 0.80 ms | 0.08 ms (pandas) | 0.09x |
| group-by sum (1000 keys) | 100,000 | 0.93 ms | 0.35 ms (pandas) | 0.38x |

A Metal dispatch costs 60–140 µs before the first byte is touched: command-buffer creation,
encoding, commit and the completion wait. A CPU library sums a thousand integers in a fraction of a
microsecond. Below about a million rows the GPU cannot win a single operation, and this page will
always carry these rows. What helps: batching several operations into one command buffer
(`MetalContext.batch`, 8–32% off per call), the fused expression compiler (one dispatch for a whole
expression tree), and the lazy engine (one command buffer for a whole plan) — all of which move the
break-even point down, none of which remove the floor. The persistent-kernel approach that would
remove it is impossible on this hardware ([RESIDENT.md](RESIDENT.md)).

#### 2. `shift` as a copy (4 rows)

| operation | rows | ArrowMetal | Polars | pandas | ratio |
|---|---:|---:|---:|---:|---:|
| shift (lag 1, int64) | 10,000,000 | 2.30 ms | 0.05 ms | 0.05 ms | 0.02x |
| shift (lag 1, int64) | 50,000,000 | 3.63 ms | 0.05 ms | 0.15 ms | 0.01x |

The matrix measures the default `shift`, which writes a new contiguous column: 813 MB moved at 50M
rows at 226 GB/s, which is this machine's bandwidth, so the kernel is not the problem. Polars answers
with a two-chunk view (a null chunk in front of a slice of the original) and copies nothing. Since the
morning run `shift(by, fill, view=True)` returns exactly that — a `pyarrow.ChunkedArray` over the same
device memory, 0.04 ms at 10M rows — and it is opt-in because the chunked form is not a `MetalArray`
and cannot re-enter a kernel without being combined. The matrix keeps measuring the default, so the
row stays here; a caller who wants Polars' answer has it.

#### 3. Grouped variance and stddev in software binary64 (2 rows)

| operation | rows | ArrowMetal | pyarrow | ratio |
|---|---:|---:|---:|---:|
| variance by key (1000 groups) | 50,000,000 | 34.1 ms | 26.4 ms | 0.78x |
| stddev by key (1000 groups) | 50,000,000 | 34.1 ms | 26.2 ms | 0.77x |

The grouped moments accumulate in software IEEE-754 binary64, because Apple GPUs have no double
arithmetic and a float32 accumulator is wrong past a few million rows. The answer agrees with Arrow to
about 6e-10 relative (2e-8 on float32) — the moment accumulator's own stated accuracy — not to the
last bit; the price is 3–4 GPU instructions per FLOP. At 10M rows the same rows tie pyarrow (1.00x,
1.02x); against Polars they win by 2.4x; at 100,000 and 10M groups they win by 2.1–4x.

**These two rows are the ones the small-group work below did not fix, and the reason is not the
arithmetic.** The key-mapping change took 3.8 ms off them (33.5 → 29.7 ms at 50M rows against
pyarrow's 19.0 in the same script, 0.64x); what is left is the *shape* the software binary64 forces.
Measured, at 50 million rows and a thousand groups:

| step | cost | the same step with the keys already sorted |
|---|---:|---:|
| the counting sort by group id | 10.8 ms | 4.7 ms |
| the two moment passes over it | 15.3 ms | 7.0 ms |
| one segmented float64 sum (for scale) | 7.4 ms | 1.7 ms |

Both moment passes read their values through `ord`, the counting sort's permutation, and consecutive
positions inside a group are about `K` rows apart, so each read pulls a cache line to use eight bytes
of it. Sorting the key column first makes that permutation nearly the identity and halves the whole
operation — which is a diagnosis, not a fix, since the input is not sorted.

The obvious fix, and the one this work set out to make, is to drop the sort: privatise the
accumulators per threadgroup, as `sum` and `count` do. It cannot be done here. A threadgroup-private
accumulator needs an atomic add, and Metal has 32-bit atomics only, so a binary64 accumulator can only
be updated by a thread that owns it exclusively — which means one private table per *lane*, not per
threadgroup. At a thousand groups that is 8 KB a lane, and 32 KB of threadgroup memory holds four of
them. Every arrangement that fits — one lane owning `g % 32`, or the values shuffled to their owner —
puts one lane's `d_add` under a mask while the other 31 wait, and pays 32 times the arithmetic. The
counting sort exists precisely because there is no 64-bit atomic add (the shading language exposes 64-bit atomic min and max only; UPSTREAM.md).

What is left is the group count itself. The moment kernels give a whole threadgroup to one group, so a
thousand groups is a thousand threadgroups, and only a fraction of them are resident at once: the
resident ones read rows scattered through the column instead of sweeping it. Measured at 50M rows, the
two moment passes cost 49.1 ms at 10 groups (too few threadgroups to fill the machine), 9.7 ms at 100,
15.4 ms at 1,000 and 16.1 ms at 4,000 — the wall is between 100 and 1,000 groups, exactly where these
rows sit. A simdgroup per group instead of a threadgroup would keep every group resident and preserve
the accumulation order (each lane emulating eight of the 256 logical slots), and the K = 100 figure
says that is worth about 6 ms of the 15.4. It is not done here, and it would leave the sort's 10.8 ms
untouched, so the honest ceiling for these two rows on this hardware is around 1.5x pyarrow, not 3x.

#### 4. Grouped min at 1000 groups, 10M rows (1 row)

| operation | rows | ArrowMetal | pyarrow | ratio |
|---|---:|---:|---:|---:|
| min by int32 key (1000 groups) | 10,000,000 | 5.77 ms | 5.05 ms | 0.87x |

The morning run had this row at 4.64 ms (1.09x), the 50M row of the same operation did not move
(9.86 ms, 2.06x over pyarrow), and the grouped min/max kernels were not touched between the runs. The
targeted re-measurement at the end of the page put it back at 4.17 ms (1.22x): the matrix value is
noise on a 5 ms call. The row stays in this table because the matrix is the record.

Since then the key-mapping change below has taken the operation to **1.66 ms** against pyarrow's 4.62
in the same script (2.8x), and 5.71 ms against 18.5 at 50M rows (3.2x). The next matrix run should
remove this row.

#### 5. Regex on the host (1 row)

| operation | rows | ArrowMetal | Polars | ratio |
|---|---:|---:|---:|---:|
| match_substring_regex (real regex) | 1,000,000 | 18.9 ms | 17.4 ms | 0.92x |

A real regex runs on the CPU (RE2-equivalent semantics through Foundation), behind a GPU pre-filter
that clears rows that cannot match. At 10M rows the pre-filter wins 1.15x over Polars and 2.2x over
pyarrow; at 1M rows the host regex dominates. LIKE patterns and literal substrings are GPU kernels and
win by 7–87x; only a genuine regex takes this path.

#### 6. Noise on zero-cost rows (4 rows)

`slice (zero-copy view)` reads 0.00 ms for every library — all are pointer arithmetic, and the ratio
is measurement noise.

### Faster, but under the 3x bar (93 rows)

The full list is in the matrix page under ⚠️. The clusters:

- **Software binary64 math** — `ln` 1.06–1.08x, `sin` 1.98–1.99x, `days_between` 1.04–1.11x,
  `sqrt` at 10M rows 1.83x (3.13x at 50M, where the dispatch floor no longer shows). Within 2 ulp of the host libm
  for `exp`, `ln`, `log2`, `log10` and `power` (the test bound), `sqrt` correctly rounded, 4–5 ulp for
  trigonometry; the CPU has hardware doubles and the GPU does not. These will not reach
  3x without a different numerical contract.
- **Grouped aggregates against pyarrow at 1000 groups** — sum/count/min/max/mean by int32, float64 and
  utf8 key at 1.3–2.2x, two int32 keys at 1.11x (10M rows; 3.8x at 50M), and variance/stddev at
  1.0–2.4x. pyarrow's grouped kernels are memory-bound and 16-thread; the GPU's advantage grows with
  the number of groups (100,000 groups: 3.5–5.5x; 10M groups: 2.9–25x) and with wider values.
  **Most of this cluster is fixed** (see the last section): the mapping from key values to dense group
  ids was 6.8 ms of the 8.8 ms `sum` at 50M rows and a thousand groups, because it widened the key
  column to int64 before it read it twice. Re-measured against pyarrow in the same script, sum/count/
  mean by int32 key are now 3.1–3.9x at 50M rows and min/max 3.2–3.3x, where they were 1.7–2.0x. Two
  rows in the cluster do not move and are not touched by that change: `sum by utf8 key` (2.0x at 50M
  and 1.7x at 10M in the matrix; 2.3x in `loss_groupby_small.py`), whose cost is the string hash table
  in front of the aggregate rather than the group-by, and variance/stddev, which cause 3 above now
  explains in full.
- **Memory-bound element-wise kernels against Polars and numpy** — compare, `is_nan`, `abs`,
  `bit_wise_and`, `if_else`, `negate`, `shift_left`, `replace_with_mask`, `drop_null` at 1.1–2.98x.
  Both sides run at unified-memory bandwidth; the GPU's edge is the dispatch overhead it does not pay
  per thread. Fusing them into one expression (`am.query`) is where the 3x comes from, not from the
  single kernel; even so `filter two columns + sum` at 10M rows is 2.2x over Polars (6x at 50M).
- **`sort float64`** at 2.67x over Polars, both sizes, in the matrix. `argsort` of the same column is
  10x; the difference was the `take` that materialised the sorted values (a random 8-byte gather).
  That gather is gone — the sorted values now come out of the sort's own keys — and the row is
  **6.503 ms at 10M and 31.948 ms at 50M** when measured on its own (the last section of this page),
  which is 4.01x and 4.14x against the Polars figures in this matrix's own column. It stays in this
  list until a rerun of the matrix moves it, which is the rule this page keeps.
- **Joins** at 2.7–2.8x over pyarrow for the materialised inner join (the index-only join and the
  left outer join are 3.0–3.8x).
- **Host-assisted strings** — `parse` 1.4–2.7x, real regex 1.2–2.2x.
- **Small dictionaries** — `unique`, `value_counts`, `dictionary_encode` and `mode` on 1000-distinct
  columns at 1.9–2.5x over pandas/pyarrow at 10M rows (4x at 50M).
- **`list_value_length`** at 1.9–2.1x: 80 MB in 0.37 ms is bandwidth plus the dispatch floor.
- **The latency family at 1M rows** — 1.0–2.1x, the floor again.

### What changed since the morning run

Compared with `full_matrix_2026-09-07-am.csv` on the same machine, 54 rows are faster and 13 read
slower. The wins are the two changes recorded in the next section:

| operation | rows | morning | afternoon | fastest CPU now | ratio now |
|---|---:|---:|---:|---:|---:|
| count_distinct by key (1000 groups) | 50,000,000 | 708 ms | 45.7 ms | 156 ms (Polars) | 3.4x |
| count_distinct by key (10M groups) | 50,000,000 | 715 ms | 61.0 ms | 401 ms (Polars) | 6.6x |
| tdigest(float64, q=0.5) | 50,000,000 | 2,836 ms | 48.5 ms | 1,061 ms (pyarrow) | 21.9x |
| sum by two int32 keys (~1024 groups) | 50,000,000 | 22.7 ms | 6.0 ms | 22.9 ms (pyarrow) | 3.8x |
| argsort int64 | 50,000,000 | 128 ms | 39.2 ms | 301 ms (Polars) | 7.7x |
| argsort float64 | 50,000,000 | 129 ms | 40.0 ms | 429 ms (Polars) | 10.8x |
| lexsort (2 int32 keys) | 50,000,000 | 149 ms | 37.4 ms | 912 ms (Polars) | 24x |
| argsort utf8 | 10,000,000 | 60.8 ms | 13.7 ms | 234 ms (Polars) | 17x |
| sort float64 | 50,000,000 | 139 ms | 49.6 ms | 132 ms (Polars) | 2.67x |
| sqrt (float64) | 50,000,000 | 12.7 ms | 3.95 ms | 12.4 ms (numpy) | 3.1x |
| upper / lower / trim | 10,000,000 | 52–55 ms | 5.2–5.4 ms | 139–173 ms (pyarrow / pandas) | 27–32x |
| list_value_length | 10,000,000 | 1.12 ms | 0.37 ms | 0.71 ms (pyarrow) | 1.9x |

Ten of the 13 slower rows are at 10M rows, one at 1M (`to_strings`, 1.21x) and two at 50M (`min` and
`mean` of int64 with 10% nulls, 1.16x and 1.11x on calls of about a millisecond); none is above 1.7x.
Where the slower row is at 10M, the 50M row of the same operation is unchanged (`partition_nth_indices`
4.9 → 8.0 ms at 10M against 19.8 ms unchanged at 50M; `max by int32 key` 2.65 → 3.85 ms against
9.87 ms unchanged; the rest are 1.1–1.5x on values of 1–4 ms in kernels the two changes did not touch:
LIKE, floor_temporal, coalesce, fill_null_forward). That is the signature of run-to-run noise on
short calls, not of a regression, and the targeted re-measurement below settles it.

The re-measurement (`full_matrix.py --families sort,group-by,chains --sizes 10000000`, idle machine,
5.5 minutes, kept as `private` data and summarised here):

| operation, 10M rows | morning | afternoon matrix | re-measured | fastest CPU | ratio |
|---|---:|---:|---:|---:|---:|
| compare + filter + take | 0.98 ms | 1.52 ms | 0.99 ms | 7.33 ms (Polars) | 7.4x |
| max by int32 key (1000 groups) | 2.65 ms | 3.85 ms | 2.76 ms | 5.07 ms (pyarrow) | 1.8x |
| min by int32 key (1000 groups) | 4.64 ms | 5.77 ms | 4.17 ms | 5.08 ms (pyarrow) | 1.2x |
| mean by int32 key (100000 groups) | 3.50 ms | 4.11 ms | 3.46 ms | 15.8 ms (pyarrow) | 4.6x |
| sum by two int32 keys (~1024 groups) | 6.26 ms | 5.10 ms | 2.24 ms | 5.67 ms (pyarrow) | 2.5x |
| sort float64 | 29.7 ms | 9.73 ms | 9.63 ms | 28.1 ms (Polars) | 2.9x |
| argsort int64 | 27.7 ms | 7.86 ms | 7.76 ms | 56.5 ms (Polars) | 7.3x |
| count_distinct by key (1000 groups) | 124 ms | 7.96 ms | 8.26 ms | 27.2 ms (Polars) | 3.3x |
| **partition_nth_indices (n/2)** | **4.90 ms** | **7.97 ms** | **8.09 ms** | 56.8 ms (pyarrow) | 7.0x |

Every grouped and chained row went back to its morning value, so those were noise; the two-key row's
10M value is the noisiest of all (2.2–6.3 ms across four runs) and its 50M value is stable at 6.0 ms.

**The `partition_nth_indices` row was not a regression, and the investigation of it found a real cost
anyway.** Running the two builds alternately in one process each with `Benchmarks/loss_partition_nth.py`
— full_matrix's columns, seed and rule, with the row measured where the matrix measures it, last in the
sort family — the radix-sort commit moves it by nothing: 5.20 → 5.08 ms at 10M and 19.97 → 19.74 ms at
50M. What the 4.9 against 8.1 actually tracked is *what ran before it*. `partition_nth_indices` allocated
about 280 MB of scratch per call out of the context's buffer pool, and in a fresh process with nothing in
front of it the pool had none of that to give: measured first in the process it cost 7.7–8.0 ms on
**both** builds, and measured after any operation that had already allocated the big buffers it cost
4.9–5.2 ms on both. The afternoon matrix and the re-measurement run put different work in front of the
row, and that is the whole of the difference.

The cost the investigation did find is that the operation was doing far too much work for what it is.
The split around the selected key was three `compare` + `filter` compactions of an index array,
concatenated afterwards — seven command buffers, three reads of the keys, three worst-case index
arrays — and two of its steps never touched the GPU at all: the row numbers it compacted were filled by
a host loop and the output it concatenated into was allocated zeroed, a single-threaded write and a
memset over 40 MB at 10M rows. It is now one stable counting sort over five buckets (null, NaN, below
the key, equal, above) in a single command buffer, and the pool sensitivity mostly goes with it:

| operation | rows | before | after |
|---|---:|---:|---:|
| partition_nth_indices (n/2), int64 | 10,000,000 | 5.13 ms | 3.02 ms |
| partition_nth_indices (n/2), int64 | 50,000,000 | 19.84 ms | 12.27 ms |
| the same, measured first in a fresh process | 10,000,000 | 7.9 ms | 5.5 ms |

Both builds alternated over two rounds, best of five after a warm-up, idle machine. No other row of the
sort family moves: argsort int64 7.84 → 7.80 ms at 10M and 39.0 → 39.0 at 50M, sort float64 9.74 → 9.68
and 49.3 → 49.4, top_k (k=100) 1.06 → 0.96 and 2.59 → 2.62, select_k_unstable 1.03 → 0.96 and
2.63 → 2.65, lexsort 7.14 → 7.14 and 37.4 → 37.4.

### The work between the two runs, as it was designed and measured

This is the record of the two changes that separate the morning run from the afternoon run, written
by the people who made them, with the numbers they measured at the time. Each number comes from a
per-operation script that uses `full_matrix.py`'s columns, seed and rule (one warm-up, best of five),
with the before and after builds run alternately in separate processes; the matrix rows above are the
authoritative re-measurement and agree with these to within a few per cent.

#### Sort, sqrt and shift

Measured with `Benchmarks/loss_sort_shift_sqrt.py`.

| operation | rows | before | after | fastest CPU | before | after |
|---|---:|---:|---:|---|---:|---:|
| sqrt (float64) | 10,000,000 | 2.67 ms | **1.36 ms** | 2.53 ms (numpy) | 0.95x | **1.86x** |
| sqrt (float64) | 50,000,000 | 12.96 ms | **3.89 ms** | 12.54 ms (numpy) | 1.03x | **3.23x** |
| sort float64 | 10,000,000 | 32.3 ms | **9.42 ms** | 30.1 ms (Polars) | 1.09x | **3.20x** |
| sort float64 | 50,000,000 | 154.6 ms | **49.2 ms** | 141.8 ms (Polars) | 1.95x | **2.88x** |
| argsort int64 | 10,000,000 | 27.8 ms | **7.68 ms** | 57.2 ms (Polars) | 2.21x | **7.45x** |
| argsort int64 | 50,000,000 | 142.1 ms | **39.0 ms** | 302.6 ms (Polars) | 5.23x | **7.76x** |
| argsort float64 | 10,000,000 | 28.1 ms | **7.93 ms** | 82.3 ms (Polars) | 3.15x | **10.4x** |
| argsort float64 | 50,000,000 | 143.3 ms | **39.8 ms** | 445.1 ms (Polars) | 6.85x | **11.2x** |
| shift (lag 1, int64), `view=True` | 10,000,000 | — | **0.037 ms** | 0.033 ms (Polars) | — | **0.91x** |
| shift (lag 1, int64), `view=True` | 50,000,000 | — | **0.154 ms** | 0.034 ms (Polars) | — | **0.22x** |

(The CPU figures on the 50M rows of the "before" pass ran while a second process was resident and came
out 2–3x slower than the matrix's; the CPU column above therefore quotes the quiet "after" pass, which
agrees with the matrix to within a few per cent. ArrowMetal's own before-numbers agree with the matrix
in both passes.)

**`sqrt`.** Still correctly rounded — bit-identical to `Foundation.sqrt`, and to numpy and Python's
`math.sqrt` over 10.6M values covering the subnormals, every binade, both sides of exact squares, ±0,
±inf, NaN and negatives. The 54-step restoring extraction is gone; in its place is a hardware `rsqrt`
seed, three Newton steps on the reciprocal square root in Q62 fixed point, and the exact 128-bit
remainder `N - q²` to settle the last bit, which is the same shape `d_div` has used since it stopped
being a long division. At 50M rows the kernel now moves its 800 MB at 206 GB/s, so what is left is
bandwidth, not arithmetic. At 10M rows it is 1.86x rather than 3x for the ordinary reason: 1.36 ms is
close enough to the dispatch floor that the fixed cost shows.

**The radix sort.** Three changes, in order of what they were worth. The stable scatter used to find an
element's rank among the earlier elements of its 256-element chunk carrying the same digit by *walking
the chunk* — a 256-iteration loop per element for the rank and a second one per digit for the chunk
totals, about 1,500 instructions an element per pass, which was roughly two thirds of the whole sort.
It is now eight `simd_ballot`s to isolate the lanes of the SIMD group holding the same digit, a
popcount for the rank, and a byte per (SIMD group, digit) so the groups of a chunk add up across.
Blocks then became far bigger and fewer — about 128 rather than n/4096 — because each block loads and
clears the whole bin table on every pass and the digit scan is a single threadgroup over
`radix × blocks` entries; the scatter is memory bound and does not miss the parallelism. Last, a pass
whose digit is the same in every row is an identity permutation (the sort is stable) and is now
dropped: the first histogram reports the bitwise OR and AND of the keys next to its counts, and
`or ^ and` names every skippable pass at once, so a small-range, sorted-ish or single-valued column
loses most of its passes. Every permutation is unchanged — checked against `pc.array_sort_indices` over
21 lengths × 2 directions × 2 null placements × 10 key shapes.

The "fewer, wider digits" line above was the wrong guess, and it was measured rather than assumed away:
eleven bits do take a 64-bit key in six passes instead of eight, but a 2048-bin scatter needs 24 KB of
threadgroup memory against 3 KB, and the occupancy that costs is worth more than the two passes it
saves. Argsort of 50M int64, same code, digit width only: 8 bits 37.7 ms, 10 bits 84.3 ms, 11 bits
96.3 ms; 12 bits does not fit in threadgroup memory at all.

`sort float64` reaches 2.9x rather than 3x, and the remainder is not the sort: it is `take`. `sorted()`
is `take(argsort())`, and the gather costs 9.4 ms of the 49.2 at 50M rows because a random 8-byte
gather pulls a whole cache line per element. The sorted *keys* are already sitting in the sort's own
buffer and the key map is a bijection for every value except -0.0 and NaN, so inverting them instead of
gathering would remove that 9.4 ms — for a column with no nulls, no NaN and no -0.0, which needs a flag
the key kernel does not raise yet. That is the next step, and unlike the last one it has been costed.
(It was taken; the last section of this page is what it measured.)

**`shift`.** The copy is unchanged and is still the default, because it was never the thing that was
wrong: 3.6 ms for the 813 MB it touches at 50M rows is 226 GB/s, this machine's bandwidth and rather
more than the sqrt kernel next to it reaches, so a better kernel would still be fifty times slower than
a pointer. What was missing was the option not to copy. `shift(by, fill, view=True)` in Python returns
a `pyarrow.ChunkedArray` of two chunks — |by| rows of `fill` or of nulls in front, and a slice of the
input behind, still pointing at the same device memory — and moves nothing, which is exactly the answer
Polars gives. It is opt-in rather than the default because the chunked form is not a `MetalArray` and
cannot go back into an ArrowMetal kernel without being combined first, which costs the copy that was
just avoided; `MetalArray` still has no chunked representation, and that has not changed.

The 50M view costs 0.154 ms rather than the 0.003 ms the pointer arithmetic takes, because
`pa.chunked_array` settles the slice's null count — a popcount over 50M validity bits. A column with no
nulls does not pay it. That is pyarrow's accounting, not a copy.

#### count_distinct by key, tdigest, two-key group-by and list lengths

Measured with `Benchmarks/loss_bench.py`, the "before" being the previous build loaded through
`ARROWMETAL_LIB`.

**`count_distinct` by key** (cause 3) — a GPU hash **set** over the (key, value) pair replaces the
dictionary encoding, the packed int64 column and the `unique()` over it. One insert pass over the
rows, then one pass over the table's occupied slots, each of which is one distinct pair and increments
its group's count. Answers identical to the previous implementation over 162 shapes (float64 / float32
/ int32 values, NaN, ±0.0, inf, 0–100% nulls, 0 to 300k rows, 1 to 997 groups).

| operation | rows | before | after | Polars | pyarrow |
|---|---:|---:|---:|---:|---:|
| count_distinct by key (1000 groups) | 10,000,000 | 122.6 ms | **8.6 ms** | 27.1 ms | 325 ms |
| count_distinct by key (1000 groups) | 50,000,000 | 703 ms | **45.3 ms** | 160 ms | 1,876 ms |
| count_distinct by key (100000 groups) | 10,000,000 | 123.1 ms | **8.6 ms** | 31.8 ms | 372 ms |
| count_distinct by key (100000 groups) | 50,000,000 | 704 ms | **45.5 ms** | 154 ms | 1,998 ms |
| count_distinct by key (10M groups) | 10,000,000 | 125.6 ms | **11.7 ms** | 83.5 ms | 661 ms |
| count_distinct by key (10M groups) | 50,000,000 | 710 ms | **60.5 ms** | 418 ms | 3,235 ms |

**`tdigest`** (cause 4) — the host walk is gone, and so is the centroid merge it was doing, because
there was never anything to merge. This digest scales its weight limit by the weight seen *so far*
rather than by the column's final weight, and the k1 scale function's inverse is bounded by 1, so the
limit never reaches the next value's weight and every centroid holds exactly one value — at any
compression, for any column length. The digest of a sorted column is therefore the sorted column, and
`TDigest.quantile` over unit centroids has a closed form (`TDigest.sortedQuantile`, held against the
walk itself by `TDigestTests`). The nulls are also compacted away before the sort, because sorting a
column that carries a validity bitmap costs three times sorting the same values without one. The
answer is unchanged to the last bit.

| operation | rows | before | after | pyarrow |
|---|---:|---:|---:|---:|
| tdigest(float64, q=0.5) | 10,000,000 | 653 ms | **28.7 ms** | 198–246 ms |
| tdigest(float64, q=0.5) | 50,000,000 | 3,082 ms | **151 ms** | 1,070 ms |

**Two-key group-by** (cause 8) — several integer key columns whose ranges multiply out to at most
2^24 (and at most the row count) are packed into one key in a single pass instead of being folded
pairwise through three range encodings and two materialised int64 columns. The dense ids are the
fold's own, value for value, so the group order is unchanged.

| operation | rows | before | after | Polars | pyarrow |
|---|---:|---:|---:|---:|---:|
| sum by two int32 keys (~1024 groups) | 10,000,000 | 6.6–7.2 ms | **2.2–4.9 ms** | 32.3 ms | 5.8 ms |
| sum by two int32 keys (~1024 groups) | 50,000,000 | 22.3 ms | **6.2 ms** | 168 ms | 22.2 ms |

The 10M row is the one number here that a second job on the machine moves: the operation is now short
enough that GPU contention shows up in it, and the single-key row next to it (untouched code) moved
2.8 ms → 5.6 ms across the same runs. At 50M, where the runs were stable, it is 3.6x pyarrow.

**`list_value_length`** (cause 9) — one thread per row read every offset twice and stored four bytes
at a time, which measured 70 GB/s of real traffic where the element-wise arithmetic kernels next door
(the same 80 MB, 4-wide vectors) run at 225 GB/s. Eight rows per thread through vector loads and
stores; a slice that leaves the offsets pointer misaligned keeps the one-row kernel.

| operation | rows | before | after | Polars | pyarrow |
|---|---:|---:|---:|---:|---:|
| list_value_length | 10,000,000 | 1.05 ms | **0.35 ms** | 5.19 ms | 0.68 ms |

### The work after the afternoon run: grouped aggregates at a small group count

Measured with `Benchmarks/loss_groupby_small.py`, which is the group-by family of `full_matrix.py`
alone — its seed, its distributions, its rule (one warm-up, best of five under a 1.2 s budget). The
two builds ran alternately, one process at a time, two rounds each, on an idle machine; the pool is
warmed with a sort, a sum and a group-by after every column exists, because warming it before the
columns are built leaves the first timed row reading twice its settled value. pyarrow was measured in
the same script and is a little faster there than in the matrix (different draw order, same
distributions), so the ratios below are the conservative ones.

**Where the time actually went.** At 50 million rows and a thousand groups, `sum by int32 key` was
8.8 ms, of which the aggregation kernel was 1.9. The other 6.9 ms was the stage in front of it: the
map from key values to dense ids `0 ..< K`. That stage took the range path — mark which values occur,
scan the marks, read each row's rank — but it first **widened the key column to int64**, because its
two kernels were written for `long` only. At 50M int32 keys that is a 400 MB write nothing else needs,
followed by two passes reading 400 MB where 200 would do: 1.2 GB of the 1.8 GB the mapping moved. The
kernels are now generated per key element type, the conversion happens in a register, and the rank
kernel reads the distinct count out of the scan's own last entry so the three dispatches share one
command buffer instead of needing the host in between. The ids are unchanged, value for value.

| key mapping alone | rows | groups | before | after |
|---|---:|---:|---:|---:|
| `am.group_by([int32])` | 10,000,000 | 1,000 | 2.11 ms | **0.91 ms** |
| | 10,000,000 | 100,000 | 1.98 ms | **0.98 ms** |
| | 10,000,000 | 10,000,000 | 3.51 ms | **3.20 ms** |
| | 50,000,000 | 1,000 | 6.82 ms | **3.03 ms** |
| | 50,000,000 | 100,000 | 9.19 ms | **4.58 ms** |
| | 50,000,000 | 10,000,000 | 9.98 ms | **9.16 ms** |

Two host loops went with it. `hash_mean` over an integer column ran the whole column twice — once for
the sum, once for a count the sum kernel had already produced as the thing that decides which keys are
null — and then divided group by group on the CPU; `hash_sum` copied its own output buffer into a
fresh column in a second loop. Neither matters at a thousand groups and they were 19 ms of a 68 ms
mean and 10 ms of a 43 ms sum at ten million. The mean now reuses the counts and divides on the GPU,
and the sum hands back the accumulator's buffer with a GPU-built bitmap. `d_from_long` /
`d_from_ulong` round exactly as `Double(Int64)` / `Double(UInt64)` do and `d_div` is correctly
rounded, so the quotient is bit for bit the host division's — checked over 1,762,240 group means
across all eight integer element types, 0% and 10% value nulls, five row counts and five group counts,
with zero differing bit patterns. Putting the accumulation and those finalizing kernels in one command
buffer rather than three matters too: a command buffer is 100–150 µs whatever it carries, which is
nothing next to a 50-million-row pass and most of a group-by over a million rows.

| operation, 1,000 groups | rows | before | after | pyarrow | before | after |
|---|---:|---:|---:|---:|---:|---:|
| sum by int32 key | 50,000,000 | 8.75 ms | **4.81 ms** | 16.45 ms | 1.88x | **3.42x** |
| count by int32 key | 50,000,000 | 7.88 ms | **4.05 ms** | 15.58 ms | 1.98x | **3.85x** |
| mean by int32 key | 50,000,000 | 9.78 ms | **4.80 ms** | 16.55 ms | 1.69x | **3.45x** |
| min by int32 key | 50,000,000 | 9.80 ms | **5.71 ms** | 18.48 ms | 1.89x | **3.24x** |
| max by int32 key | 50,000,000 | 9.67 ms | **5.74 ms** | 18.97 ms | 1.96x | **3.30x** |
| min by key, float64 | 50,000,000 | 9.70 ms | **5.79 ms** | 18.84 ms | 1.94x | **3.25x** |
| max by key, float64 | 50,000,000 | 9.62 ms | **5.74 ms** | 18.71 ms | 1.94x | **3.26x** |
| variance by key | 50,000,000 | 33.52 ms | **29.68 ms** | 18.99 ms | 0.57x | 0.64x |
| stddev by key | 50,000,000 | 33.32 ms | **29.79 ms** | 19.05 ms | 0.57x | 0.64x |
| sum by utf8 key | 50,000,000 | 18.18 ms | 18.11 ms | 42.14 ms | 2.32x | 2.33x |
| sum by two int32 keys | 50,000,000 | 5.85 ms | 5.88 ms | 21.21 ms | 3.62x | 3.61x |
| sum by int32 key | 10,000,000 | 2.78 ms | **1.82 ms** | 4.07 ms | 1.46x | **2.24x** |
| count by int32 key | 10,000,000 | 2.55 ms | **1.65 ms** | 3.89 ms | 1.53x | **2.35x** |
| mean by int32 key | 10,000,000 | 3.30 ms | **1.77 ms** | 4.20 ms | 1.27x | **2.37x** |
| min by int32 key | 10,000,000 | 2.55 ms | **1.66 ms** | 4.62 ms | 1.81x | **2.79x** |
| max by int32 key | 10,000,000 | 2.57 ms | **1.70 ms** | 4.81 ms | 1.87x | **2.82x** |
| variance by key | 10,000,000 | 6.35 ms | **5.66 ms** | 5.31 ms | 0.84x | 0.94x |

At 10 million rows the operations are 1.7–1.8 ms and about a fifth of that is the dispatch floor, so
they sit at 2.2–2.8x rather than 3x for the reason cause 1 gives.

The higher group counts move the same way and nothing there gets slower:

| operation | rows | groups | before | after | pyarrow | after |
|---|---:|---:|---:|---:|---:|---:|
| sum by int32 key | 50,000,000 | 100,000 | 11.24 ms | **7.70 ms** | 48.28 ms | **6.3x** |
| count by int32 key | 50,000,000 | 100,000 | 8.46 ms | **4.96 ms** | 36.24 ms | **7.3x** |
| mean by int32 key | 50,000,000 | 100,000 | 12.95 ms | **7.70 ms** | 47.41 ms | **6.2x** |
| min by int32 key | 50,000,000 | 100,000 | 13.47 ms | **9.98 ms** | 52.12 ms | **5.2x** |
| variance by key | 50,000,000 | 100,000 | 40.08 ms | **36.75 ms** | 136.47 ms | **3.7x** |
| sum by int32 key | 10,000,000 | 10,000,000 | 13.78 ms | **10.07 ms** | 332.53 ms | **33x** |
| mean by int32 key | 10,000,000 | 10,000,000 | 28.66 ms | **11.38 ms** | 324.81 ms | **29x** |
| mean by int32 key | 50,000,000 | 10,000,000 | 83.72 ms | **55.85 ms** | 1,319.80 ms | **24x** |
| sum by int32 key | 50,000,000 | 10,000,000 | 54.84 ms | **53.31 ms** | 1,321.68 ms | **25x** |

**The shape sweep.** Timings on the rows a benchmark happens to contain say nothing about the shapes
it does not, so `loss_groupby_small.py --sweep` walks one axis at a time away from a base shape — row
count 1M / 3M / 10M / 27M / 50M, group count 1 / 2 / 10 / 100 / 1,000 / 1,025 / 5,000 / 20,000 /
100,000 / 1M / 10M, keys uniform / 90%-in-one-group / sorted, keys with 10% nulls, values int64 and
float64, values with 10% nulls, and a slice at offset 33 — over sum, count, mean, min, max and
variance. 220 measurements, both builds, two rounds each (sweep output not committed; rerun
`Benchmarks/loss_groupby_small.py --sweep` to reproduce). **218 are faster after the change and none
is slower than main's own spread**: the two that read below 1.0x are `min` and `max` at 3M rows and
1,000 groups, where main measured 1.42 and 2.49 ms across its two rounds and the new build 1.57 and
1.62 — the new build's worst is below main's worst, and the comparison picked main's lucky round.
`--digest` hashes every answer instead of timing it: all 220 are **bit-identical** between the two
builds.

Two shapes in that sweep are slow on both builds and are worth recording as the next thing to look at,
because they are the same wall cause 3 describes: a grouped `sum` or `mean` over a **float64** column
with very few groups runs the segmented path, which gives a whole threadgroup to one group, so 10
million rows in one group is 35 ms and 90% of the rows in one group is 31 ms, against 4.6 ms for the
same column spread over a thousand groups. Integer values do not have this shape — they take the
atomic path — and neither does `min`, `max` or `count`.
### After the afternoon run: `sorted()` stops gathering, and the nulls stop being a host pass

Measured with `Benchmarks/loss_sort_gather.py`, the before build (`main` at `86bd495`) loaded through
`ARROWMETAL_LIB` and its own ctypes package through `ARROWMETAL_PYTHON` — the change adds an entry
point, so the old package has to come with the old library. full_matrix's rule throughout: its seed,
its column builders, one warm-up, best of five. **Every figure below is in
`private/results/sort_gather_2026-09-07/`**, whose `README.md` says which file backs which row; the
whole directory was re-measured on the build this branch ships, so nothing is quoted from an
intermediate one.

Two harnesses, and it matters which is which. **Matrix mode** runs one build per process, which is how
`full_matrix.py` measures, and the tables in the next two sections are its (`matrix_table.txt`, best of
two before-runs and two after-runs). The **shape sweep** loads both builds into one process and
alternates them case by case, which is the only way its 1M-row rows say anything; every sweep figure is
labelled as such and comes from `sweep_table.txt`.

#### The gather

`sorted()` was `take(argsort())`. The radix sort maps every value to an order-preserving unsigned key,
and that map is a bijection on every value except two: -0.0 shares +0.0's key (so that the two tie,
which is what Arrow's order asks for) and every NaN payload shares one key. So for a column with
neither, the sorted values *are* the sorted keys, read backwards through the map — a sequential read
and a sequential write where the gather was a random 8-byte one that pulled a cache line per element.

The key kernel now says which, in a flag word it fills from two `simd_ballot`s while it has the values
loaded anyway; `argsort` does not ask for it and pays nothing. And once the values no longer come out
of the permutation, the sort does not have to carry the permutation: dropping the uint payload takes
every pass from 24 bytes an element to 16, which is worth more than the gather was.

A column that *does* hold a -0.0 or a NaN still gets its answer from the keys, and only the output
positions the inverse cannot produce are copied back through the sorted row numbers: the run of zeros,
the run of NaNs, the null block. Each is contiguous — they share a key — and each is found by a binary
search over the sorted keys. Those columns keep the payload, so they win less: in the sweep at 50M rows
they are 1.14x and 1.15x where the same column without them is 1.35–1.56x.

| operation | rows | before | after | before | after |
|---|---:|---:|---:|---:|---:|
| sort float64 | 10,000,000 | 9.585 ms | **6.503 ms** | 3.19x Polars | **4.70x Polars** |
| sort float64 | 50,000,000 | 50.281 ms | **31.948 ms** | 2.86x | **4.51x** |
| sort int64 | 10,000,000 | 9.390 ms | **6.142 ms** | 3.35x | **5.11x** |
| sort int64 | 50,000,000 | 49.409 ms | **30.626 ms** | 3.40x | **5.48x** |

The Polars column is measured by the same script (`--polars`, recorded in `matrix.log`) in the same
processes as the `after` rows, so ArrowMetal's own numbers there are if anything a little pessimistic:
sort float64 30.551 ms at 10M and 143.964 ms at 50M, sort int64 31.413 ms and 167.743 ms. The
afternoon matrix's own Polars column is faster — 26.1 ms and 132.4 ms for float64, on a machine with
nothing else resident — and against *those* the two float64 rows are 4.01x and 4.14x. Either
accounting puts `sort float64` past the 3x bar at both sizes, which is what this work was for.

#### The nulls

Sorting a column that carries a validity bitmap cost an order of magnitude more than sorting the same
values without one — 139.7 ms against 9.6 ms at 10M float64 rows, and 725.9 ms against 50.3 ms at 50M.
None of that was the GPU. The null rows went through the sort with whatever bytes sat under their
bitmap, and a **host** pass then lifted them back out of the finished permutation: three loops over
the whole index array plus a `sort()` of the null row numbers, because a null's place among the other
nulls is its input position, not wherever its garbage key landed.

The nulls do not enter the sort at all now. A stable three-way partition — values, NaNs when
`.atStart` wants them separated, nulls — runs first and compacts the value block's keys as it goes, so
the sort runs over the value block alone and the nulls keep their input order by construction, which
is the order Arrow asks for. Three kernels: a per-block bucket count, a scan, and a scatter that ranks
a chunk's rows against its own SIMD group with three ballots, the same trick the radix scatter uses.
The passes are then planned around the value block rather than the column, which is what the 50% rows
below are: blocks sized for the column would leave half the GPU idle.

| operation | rows | nulls | before | after | source |
|---|---:|---:|---:|---:|---|
| sort float64 | 10,000,000 | 10% | 139.745 ms | **6.837 ms** | matrix |
| sort float64 | 50,000,000 | 10% | 725.935 ms | **33.669 ms** | matrix |
| argsort float64 | 10,000,000 | 10% | 138.795 ms | **7.654 ms** | matrix |
| argsort float64 | 50,000,000 | 10% | 711.053 ms | **39.225 ms** | matrix |
| sort float64 | 50,000,000 | 1% | 187.955 ms | **36.189 ms** | sweep |
| sort float64 | 50,000,000 | 50% | 3,436.130 ms | **22.879 ms** | sweep |
| argsort int32 | 50,000,000 | 50% | 3,401.301 ms | **11.938 ms** | sweep |

The old cost grew with the null count because the host `sort()` of the null row numbers did; that is
why the 50% rows are three seconds. Every caller of `argsort` gets this, not only `sort`: `unique`,
`value_counts`, `rank`, the window functions, `lexsort` and the dictionary paths all sort nullable
columns.

**`.atStart` also stops being a host pass.** Placing the nulls (and, on a float column, the NaNs) at
the front used to scan every value on the CPU to count the NaNs before it could decide anything. The
partition does that work on the GPU now, and it runs on any float column asked for `.atStart` whether
or not it holds a NaN — so the worst case for it is a clean column, where it has nothing to separate.
Measured on a clean 50M float64 column, both builds alternating in one process (`atstart.txt`): `sort`
64.07 → 36.81 ms and `argsort` 53.96 → 44.58 ms, so even the empty partition is well ahead of the host
scan it replaced. It is not free — `argsort` `.atEnd` on the same column is 40.75 ms, so the partition
costs about 4 ms at 50M rows — but there is nothing to gate it on: whether a NaN exists is not known
until the keys have been mapped, which is the same command buffer the partition has to be encoded in.
With 1% of the rows NaN the same two calls go 542.3 → 47.6 ms and 524.6 → 50.1 ms.

**The tdigest workaround stays.** `tdigest` compacts the nulls out with `drop_null` before sorting,
which was worth an order of magnitude and is now worth 10–12% — but only at 50% nulls, where dropping
them also halves the output buffer and the pass over it (10M float64: 5.14 ms sorting the column
against 4.60 ms compacting first; 50M: 22.80 against 20.69, `tdigest_vs_dropnull.txt`). At 1% and 10%
nulls the general path is now the faster of the two by 7–10% (10M: 7.63 against 8.41 and 7.16 against
7.64; 50M: 36.38 against 40.01 and 34.20 against 36.54). Since the rule was to remove the workaround
only if the general path is at least as fast, it stays, and `Kernels/TDigestGPU.swift` records the
same numbers.

#### What did not move

`argsort` of a column with no nulls is the same code as before and measures the same: int64 7.82 →
7.73 ms at 10M and 39.8 → 39.5 at 50M, float64 7.98 → 7.95 and 40.6 → 40.3. So does `lexsort`
(6.86 → 6.66 at 10M, 37.4 → 37.3 at 50M). `partition_nth_indices` reads 2.91 → 2.51 ms at 10M and
11.67 → 11.65 at 50M, `top_k` 1.78 → 0.79 and 2.52 → 2.48, `select_k_unstable` 1.69 → 0.77 and
2.51 → 2.46; the three of them share the radix select, which this work did not touch, and their 10M
rows are the pool-sensitivity `partition_nth_indices` already has a paragraph about further up this
page rather than anything new.

#### The shape sweep

The matrix's columns are one shape. Every claim above was re-taken over five sizes (1M, 3M, 10M, 27M,
50M) × 29 columns — eleven shapes on float64 (random, sorted, reverse-sorted, all-equal,
1000-distinct, with -0.0, with NaN, 1% / 10% / 50% nulls, and a slice at offset 33) and nine each on
int64 and int32, the two float-special shapes not applying to them — × two operations: **290 measured
cases**. The two builds are loaded into **one** process there and measured alternately, case by case,
because a build measured in its own process is measured at a different minute and at 1M rows the
difference between two minutes is larger than the difference between the two builds.

Every `sort` row at 3M rows and above is faster: 1.02x to 285x. At 1M the range is 0.99x to 38x, the
0.99x being the two float64 columns that hold a -0.0 or a NaN, where 1.858 ms became 1.878 and
1.677 became 1.701 — at that size the fix-up's extra command buffer eats the win. Every `argsort` row
on a column with nulls is faster, from 1.76x to 301x. Every `argsort` row without nulls is unchanged
to within 1% at 10M rows and above, which is what the code says it should be: that path is untouched.
Two rows read below 0.97x, both `argsort` of a 1000-distinct column at 1M rows: int32 0.642 → 0.756
ms and int64 0.433 → 0.456 ms. The same sweep run with the **same build on both sides** —
`noise_control_table.txt`, the harness measuring itself — produces ten such rows, down to 0.86x and
including one at 50M, so that band is the harness and not the change. **Nothing at 10M, 27M or 50M
is slower.**

| operation, 50M rows | shape | before | after |
|---|---|---:|---:|
| sort float64 | random | 50.308 ms | **32.301 ms** |
| sort float64 | sorted | 43.615 ms | **32.298 ms** |
| sort float64 | reverse-sorted | 43.265 ms | **32.244 ms** |
| sort float64 | all-equal | 8.962 ms | **7.579 ms** |
| sort float64 | 1000-distinct | 26.418 ms | **17.413 ms** |
| sort float64 | with -0.0 | 50.309 ms | **44.241 ms** |
| sort float64 | with NaN | 50.903 ms | **44.438 ms** |
| sort float64 | sliced at offset 33 | 50.541 ms | **32.395 ms** |
| sort int64 | random | 48.277 ms | **31.511 ms** |
| sort int32 | random | 24.166 ms | **14.017 ms** |
| argsort float64 | random | 40.707 ms | 41.210 ms |
| argsort int64 | random | 38.780 ms | 38.804 ms |
