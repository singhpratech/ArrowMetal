# Where ArrowMetal loses, and why

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
arithmetic and a float32 accumulator is wrong past a few million rows. The answer is bit-exact against
Arrow; the price is 3–4 GPU instructions per FLOP. At 10M rows the same rows tie pyarrow (1.00x,
1.02x); against Polars they win by 2.4x; at 100,000 and 10M groups they win by 2.1–4x.

### 4. Grouped min at 1000 groups, 10M rows (1 row)

| operation | rows | ArrowMetal | pyarrow | ratio |
|---|---:|---:|---:|---:|
| min by int32 key (1000 groups) | 10,000,000 | 5.77 ms | 5.05 ms | 0.87x |

The morning run had this row at 4.64 ms (1.09x), the 50M row of the same operation did not move
(9.86 ms, 2.06x over pyarrow), and the grouped min/max kernels were not touched between the runs. The
targeted re-measurement at the end of the page put it back at 4.17 ms (1.22x): the matrix value is
noise on a 5 ms call. The row stays in this table because the matrix is the record; the operation is
in the "under 3x" cluster on its merits, with the other grouped aggregates at 1000 groups.

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

- **Software binary64 math** — `ln` 1.06–1.33x, `sin` 1.98–2.06x, `days_between` 1.04–1.47x,
  `sqrt` at 10M rows 1.83x (3.13x at 50M, where the dispatch floor no longer shows). Correct to 1 ulp
  (4–5 ulp for trigonometry); the CPU has hardware doubles and the GPU does not. These will not reach
  3x without a different numerical contract.
- **Grouped aggregates against pyarrow at 1000 groups** — sum/count/min/max/mean by int32, float64 and
  utf8 key at 1.3–2.2x, two int32 keys at 1.11x (10M rows; 3.8x at 50M), and variance/stddev at
  1.0–2.4x. pyarrow's grouped kernels are memory-bound and 16-thread; the GPU's advantage grows with
  the number of groups (100,000 groups: 3.5–5.5x; 10M groups: 2.9–25x) and with wider values.
- **Memory-bound element-wise kernels against Polars and numpy** — compare, `is_nan`, `abs`,
  `bit_wise_and`, `if_else`, `negate`, `shift_left`, `replace_with_mask`, `drop_null` at 1.1–2.98x.
  Both sides run at unified-memory bandwidth; the GPU's edge is the dispatch overhead it does not pay
  per thread. Fusing them into one expression (`am.query`) is where the 3x comes from, not from the
  single kernel; even so `filter two columns + sum` at 10M rows is 2.2x over Polars (6x at 50M).
- **`sort float64`** at 2.67x over Polars, both sizes, in the matrix. `argsort` of the same column is
  10x; the difference was the `take` that materialised the sorted values (a random 8-byte gather).
  That gather is gone — the sorted values now come out of the sort's own keys — and the row is
  **4.1x over Polars at 50M rows and 3.9x at 10M** when measured on its own (the last section of this
  page). It stays in this list until a rerun of the matrix moves it, which is the rule this page keeps.
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
| argsort int64 | 50,000,000 | 142 ms | 39.2 ms | 301 ms (Polars) | 7.7x |
| argsort float64 | 50,000,000 | 143 ms | 40.0 ms | 429 ms (Polars) | 10.8x |
| lexsort (2 int32 keys) | 50,000,000 | 149 ms | 37.4 ms | 912 ms (Polars) | 24x |
| argsort utf8 | 10,000,000 | 60.8 ms | 13.7 ms | 234 ms (Polars) | 17x |
| sort float64 | 50,000,000 | 139 ms | 49.6 ms | 132 ms (Polars) | 2.67x |
| sqrt (float64) | 50,000,000 | 12.7 ms | 3.95 ms | 12.4 ms (numpy) | 3.1x |
| upper / lower / trim | 10,000,000 | 52–55 ms | 5.2–5.4 ms | 139–173 ms (pyarrow / pandas) | 27–32x |
| list_value_length | 10,000,000 | 1.12 ms | 0.37 ms | 0.71 ms (pyarrow) | 1.9x |

The 13 slower rows are all at 10M rows and none is above 1.7x, while the 50M row of the same
operation is unchanged in every case (`partition_nth_indices` 4.9 → 8.0 ms at 10M against 19.8 ms
unchanged at 50M; `max by int32 key` 2.65 → 3.85 ms against 9.87 ms unchanged; the rest are 1.1–1.5x
on values of 1–4 ms in kernels the two changes did not touch: LIKE, floor_temporal, coalesce,
fill_null_forward). That is the signature of run-to-run noise on short calls, not of a regression, and
the targeted re-measurement below settles it.

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

## After the afternoon run: `sorted()` stops gathering, and the nulls stop being a host pass

Measured with `Benchmarks/loss_sort_gather.py`, the before build loaded through `ARROWMETAL_LIB` and
its own ctypes package through `ARROWMETAL_PYTHON` (the change adds an entry point, so the old package
has to come with the old library). full_matrix's rule throughout: its seed, its column builders, one
warm-up, best of five. The tables below are the two builds run alternately in separate processes,
twice each, which is how the matrix measures; the shape sweep at the end runs them alternately *inside
one process*, case by case, which is the only way the 1M-row rows say anything at all.

### The gather

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
search over the sorted keys. Those columns keep the payload, so they win less: 1.09–1.14x rather than
1.3–1.5x.

| operation | rows | before | after | fastest CPU (matrix) | before | after |
|---|---:|---:|---:|---|---:|---:|
| sort float64 | 10,000,000 | 9.71 ms | **6.75 ms** | 26.1 ms (Polars) | 2.69x | **3.87x** |
| sort float64 | 50,000,000 | 49.6 ms | **32.2 ms** | 132.4 ms (Polars) | 2.67x | **4.12x** |
| sort int64 | 10,000,000 | 9.50 ms | **6.41 ms** | 29.9 ms (Polars, re-measured) | 3.15x | **4.67x** |
| sort int64 | 50,000,000 | 48.8 ms | **30.7 ms** | 157.3 ms (Polars, re-measured) | 3.22x | **5.12x** |
| sort float64, 10% of the rows -0.0 or NaN | 50,000,000 | 49.5 ms | **43.4 ms** | — | — | 1.14x |

The Polars figures for `sort float64` are the matrix's own; this script does not re-measure the CPU
libraries. Measured here in one process with Polars resident (which costs ArrowMetal a few
milliseconds of its own), the four rows come out 28.7 ms and 140.8 ms for float64 and 29.9 ms and
157.3 ms for int64, against 8.4 / 34.8 and 6.6 / 31.3 — 3.4x and 4.0x for float64, 4.5x and 5.0x for
int64. Either accounting puts `sort float64` past the 3x bar at both sizes.

### The nulls

Sorting a column that carries a validity bitmap cost an order of magnitude more than sorting the same
values without one — 126 ms against 9.7 ms at 10M float64 rows, and 678 ms against 49.6 ms at 50M.
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

| operation | rows | nulls | before | after |
|---|---:|---:|---:|---:|
| sort float64 | 10,000,000 | 10% | 139.3 ms | **7.10 ms** |
| sort float64 | 50,000,000 | 10% | 704.6 ms | **33.7 ms** |
| sort float64 | 50,000,000 | 1% | 186.3 ms | **35.8 ms** |
| sort float64 | 50,000,000 | 50% | 3,274.7 ms | **22.7 ms** |
| argsort float64 | 10,000,000 | 10% | 131.5 ms | **7.86 ms** |
| argsort float64 | 50,000,000 | 10% | 695.8 ms | **39.0 ms** |
| argsort int32 | 50,000,000 | 50% | 3,159.9 ms | **11.4 ms** |

The old cost grew with the null count because the host `sort()` of the null row numbers did; that is
why the 50% rows are three seconds. Every caller of `argsort` gets this, not only `sort`: `unique`,
`value_counts`, `rank`, the window functions, `lexsort` and the dictionary paths all sort nullable
columns.

**The tdigest workaround stays.** `tdigest` compacts the nulls out with `drop_null` before sorting,
which was worth an order of magnitude and is now worth 6–10% — but only at 50% nulls, where dropping
them also halves the output buffer and the pass over it (10M float64: 5.1 ms sorting the column
against 4.9 ms compacting first; 50M: 23.1 against 21.0). At 1% and 10% nulls the general path is now
the faster of the two by 3–10%. Since the rule was to remove the workaround only if the general path
is at least as fast, it stays, and `Kernels/TDigestGPU.swift` records both numbers.

### What did not move

`argsort` of a column with no nulls is the same code as before and measures the same: int64 7.81 →
7.85 ms at 10M and 39.3 → 39.3 at 50M, float64 8.02 → 7.99 and 39.9 → 40.1. So do `lexsort`
(7.19 → 7.11 at 10M, 37.5 → 37.5 at 50M), `partition_nth_indices` (3.04 → 3.07 and 12.2 → 12.3) and
`select_k_unstable` (1.18 → 0.95 and 2.53 → 2.55). `top_k` reads 1.24 → 0.97 ms at 10M and
2.54 → 2.65 at 50M, which is noise on a 2.5 ms call in a kernel this work did not touch.

### The shape sweep

The matrix's columns are one shape. Every claim above was re-taken over five sizes (1M, 3M, 10M, 27M,
50M) × eleven column shapes (random, sorted, reverse-sorted, all-equal, 1000-distinct, with -0.0, with
NaN, 1% / 10% / 50% nulls, and a slice at offset 33) × three types (float64, int64, int32), both
operations: 312 rows. The two builds are loaded into **one** process there and measured alternately,
case by case, because a build measured in its own process is measured at a different minute and at 1M
rows the difference between two minutes is larger than the difference between the two builds.

Every `sort` row is faster, from 1.02x to 265x. Every `argsort` row on a column with nulls is faster,
from 2.6x to 278x. Every `argsort` row without nulls is unchanged to within 1% at 10M rows and above.
Seven rows at 1M and 3M read 0.92–0.97x; the same sweep run with the *same* build on both sides
produces rows at 0.88–0.96x, so that band is the harness, not the change. Nothing at 10M, 27M or 50M
is slower.

| operation, 50M rows | shape | before | after |
|---|---|---:|---:|
| sort float64 | random | 49.5 ms | **32.1 ms** |
| sort float64 | sorted | 42.9 ms | **32.0 ms** |
| sort float64 | reverse-sorted | 42.7 ms | **32.1 ms** |
| sort float64 | all-equal | 9.07 ms | **7.67 ms** |
| sort float64 | 1000-distinct | 26.2 ms | **17.4 ms** |
| sort float64 | with -0.0 | 49.6 ms | **43.4 ms** |
| sort float64 | with NaN | 49.5 ms | **43.3 ms** |
| sort float64 | sliced at offset 33 | 49.5 ms | **32.1 ms** |
| sort int64 | random | 47.6 ms | **31.4 ms** |
| sort int32 | random | 24.0 ms | **14.1 ms** |
| argsort float64 | random | 40.0 ms | 40.0 ms |
| argsort int64 | random | 38.1 ms | 38.1 ms |
