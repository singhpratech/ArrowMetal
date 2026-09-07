# Where ArrowMetal needs to improve, and why

The project's bar is 3x over the fastest of Polars, pyarrow and pandas/numpy on every operation at
scale. This page is the honest remainder: every measured row from the full matrix
([BENCHMARKS_MATRIX.md](BENCHMARKS_MATRIX.md), `Benchmarks/results/full_matrix_2026-09-07.csv`, idle
M4 Max, 16 cores, 64 GB) where ArrowMetal is **slower** than a CPU library, grouped by the measured
cause, followed by the rows that win by less than 3x. Nothing here is rounded in ArrowMetal's favour;
a row leaves this page only when a rerun of the matrix moves it.

Of 946 comparisons in the matrix: 822 at or above 3x, 93 faster but under 3x, 31 slower than the
CPU library. The 31 fall into six causes. The morning run of the same day
(`full_matrix_2026-09-07-am.csv`) had 50 slower and 115 under 3x; what moved is at the end of the page.

## Slower than the CPU library

### 1. The dispatch floor below a million rows (19 rows)

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

### 2. `shift` as a copy (4 rows)

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

### 3. Grouped variance and stddev in software binary64 (2 rows)

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
counting sort exists precisely because there are no 64-bit atomics.

What is left is the group count itself. The moment kernels give a whole threadgroup to one group, so a
thousand groups is a thousand threadgroups, and only a fraction of them are resident at once: the
resident ones read rows scattered through the column instead of sweeping it. Measured at 50M rows, the
two moment passes cost 49.1 ms at 10 groups (too few threadgroups to fill the machine), 9.7 ms at 100,
15.4 ms at 1,000 and 16.1 ms at 4,000 — the wall is between 100 and 1,000 groups, exactly where these
rows sit. A simdgroup per group instead of a threadgroup would keep every group resident and preserve
the accumulation order (each lane emulating eight of the 256 logical slots), and the K = 100 figure
says that is worth about 6 ms of the 15.4. It is not done here, and it would leave the sort's 10.8 ms
untouched, so the honest ceiling for these two rows on this hardware is around 1.5x pyarrow, not 3x.

### 4. Grouped min at 1000 groups, 10M rows (1 row)

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

### 5. Regex on the host (1 row)

| operation | rows | ArrowMetal | Polars | ratio |
|---|---:|---:|---:|---:|
| match_substring_regex (real regex) | 1,000,000 | 18.9 ms | 17.4 ms | 0.92x |

A real regex runs on the CPU (RE2-equivalent semantics through Foundation), behind a GPU pre-filter
that clears rows that cannot match. At 10M rows the pre-filter wins 1.15x over Polars and 2.2x over
pyarrow; at 1M rows the host regex dominates. LIKE patterns and literal substrings are GPU kernels and
win by 7–87x; only a genuine regex takes this path.

### 6. Noise on zero-cost rows (4 rows)

`slice (zero-copy view)` reads 0.00 ms for every library — all are pointer arithmetic, and the ratio
is measurement noise.

## Faster, but under the 3x bar (93 rows)

The full list is in the matrix page under ⚠️. The clusters:

- **Software binary64 math** — `ln` 1.06–1.08x, `sin` 1.98–1.99x, `days_between` 1.04–1.11x,
  `sqrt` at 10M rows 1.83x (3.13x at 50M, where the dispatch floor no longer shows). Correct to 1 ulp
  (4–5 ulp for trigonometry); the CPU has hardware doubles and the GPU does not. These will not reach
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
- **`sort float64`** at 2.67x over Polars, both sizes. `argsort` of the same column is 10x; the
  difference is the `take` that materialises the sorted values (a random 8-byte gather). Inverting
  the sort key in place of the gather is the costed next step (below).
- **Joins** at 2.7–2.8x over pyarrow for the materialised inner join (the index-only join and the
  left outer join are 3.0–3.8x).
- **Host-assisted strings** — `parse` 1.4–2.7x, real regex 1.2–2.2x.
- **Small dictionaries** — `unique`, `value_counts`, `dictionary_encode` and `mode` on 1000-distinct
  columns at 1.9–2.5x over pandas/pyarrow at 10M rows (4x at 50M).
- **`list_value_length`** at 1.9–2.1x: 80 MB in 0.37 ms is bandwidth plus the dispatch floor.
- **The latency family at 1M rows** — 1.0–2.1x, the floor again.

## What changed since the morning run

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

## The work between the two runs, as it was designed and measured

This is the record of the two changes that separate the morning run from the afternoon run, written
by the people who made them, with the numbers they measured at the time. Each number comes from a
per-operation script that uses `full_matrix.py`'s columns, seed and rule (one warm-up, best of five),
with the before and after builds run alternately in separate processes; the matrix rows above are the
authoritative re-measurement and agree with these to within a few per cent.

### Sort, sqrt and shift

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

### count_distinct by key, tdigest, two-key group-by and list lengths

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

## The work after the afternoon run: grouped aggregates at a small group count

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
