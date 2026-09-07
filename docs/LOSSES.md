# Where ArrowMetal loses, and why

The project's bar is 3x over the fastest of Polars, pyarrow and pandas/numpy on every operation at
scale. This page is the honest remainder: every measured row from the full matrix
([BENCHMARKS_MATRIX.md](BENCHMARKS_MATRIX.md), `Benchmarks/results/full_matrix_2026-09-07.csv`, idle
M4 Max, 16 cores, 64 GB) where ArrowMetal is **slower** than a CPU library, grouped by the measured
cause, followed by the rows that win by less than 3x. Nothing here is rounded in ArrowMetal's favour;
a row leaves this page only when a rerun of the matrix moves it.

Of 946 comparisons in the matrix: 781 at or above 3x, 115 faster but under 3x, 50 slower than the
CPU library. The 50 fall into nine causes.

## Slower than the CPU library

### 1. The dispatch floor below a million rows (17 rows)

| operation | rows | ArrowMetal | fastest CPU | ratio |
|---|---:|---:|---:|---:|
| sum(int64) | 1,000 | 0.14 ms | 0.00 ms (Polars) | 0.00x |
| sum(int64) | 100,000 | 0.15 ms | 0.01 ms (Polars) | 0.05x |
| sum(int64) | 1,000,000 | 0.22 ms | 0.08 ms (Polars) | 0.35x |
| filter(int64 > 0) | 1,000 | 0.18 ms | 0.00 ms (Polars) | 0.02x |
| filter(int64 > 0) | 100,000 | 0.18 ms | 0.03 ms (Polars) | 0.15x |
| group-by sum (1000 keys) | 1,000 | 0.81 ms | 0.08 ms (pandas) | 0.09x |
| group-by sum (1000 keys) | 100,000 | 0.91 ms | 0.35 ms (pandas) | 0.38x |

A Metal dispatch costs 60–140 µs before the first byte is touched: command-buffer creation,
encoding, commit and the completion wait. A CPU library sums a thousand integers in a fraction of a
microsecond. Below about a million rows the GPU cannot win a single operation, and this page will
always carry these rows. What helps: batching several operations into one command buffer
(`MetalContext.batch`, 8–32% off per call), the fused expression compiler (one dispatch for a whole
expression tree), and the lazy engine (one command buffer for a whole plan) — all of which move the
break-even point down, none of which remove the floor. The persistent-kernel approach that would
remove it is impossible on this hardware ([RESIDENT.md](RESIDENT.md)).

### 2. `shift` is a copy here and a view in Polars (4 rows)

| operation | rows | ArrowMetal | Polars | pandas | ratio |
|---|---:|---:|---:|---:|---:|
| shift (lag 1, int64) | 10,000,000 | 2.34 ms | 0.04 ms | 0.05 ms | 0.02x |
| shift (lag 1, int64) | 50,000,000 | 3.59 ms | 0.05 ms | 0.15 ms | 0.01x |

ArrowMetal's `shift` writes a new column: 800 MB moved at 50M rows, at memory bandwidth. Polars
answers with a two-chunk view (a null chunk in front of a slice of the original) and copies nothing;
pandas reuses its block. A view is the right answer and needs a chunked array representation, which
`MetalArray` does not have; the engine's `concat` is the closest thing today. Until then this row is a
genuine 50–70x loss whenever the caller does not need a contiguous result.

### 3. `count_distinct` by key (6 rows)

| operation | rows | ArrowMetal | Polars | ratio |
|---|---:|---:|---:|---:|
| count_distinct by key (1000 groups) | 10,000,000 | 124 ms | 26 ms | 0.21x |
| count_distinct by key (1000 groups) | 50,000,000 | 708 ms | 154 ms | 0.22x |
| count_distinct by key (100000 groups) | 10,000,000 | 124 ms | 31 ms | 0.25x |
| count_distinct by key (100000 groups) | 50,000,000 | 706 ms | 146 ms | 0.21x |
| count_distinct by key (10M groups) | 10,000,000 | 127 ms | 79 ms | 0.62x |
| count_distinct by key (10M groups) | 50,000,000 | 715 ms | 420 ms | 0.59x |

The grouped distinct count sorts the (key, value) pairs and counts runs — a 128-bit radix sort at
50M rows. Polars hashes each (key, value) pair into a per-group set. A GPU hash set over the packed
pair (the generic 64-bit `HashTable` already exists; it needs a two-word key) would replace the sort.
It still beats pyarrow by 2.6–2.9x on the same rows; the loss is to Polars only.

### 4. `tdigest` (2 rows)

| operation | rows | ArrowMetal | pyarrow | ratio |
|---|---:|---:|---:|---:|
| tdigest(float64, q=0.5) | 10,000,000 | 557 ms | 207 ms | 0.37x |
| tdigest(float64, q=0.5) | 50,000,000 | 2,836 ms | 1,053 ms | 0.37x |

The t-digest is built on the host from a GPU sort (0.14 GB/s says so). The sort-free `quantile`
next to it runs 158x faster than pyarrow, so a caller who wants a quantile should use `quantile`;
`tdigest` exists for callers who want the sketch itself. A GPU-side centroid merge is the fix.

### 5. Grouped variance and stddev in software binary64 (4 rows)

| operation | rows | ArrowMetal | pyarrow | ratio |
|---|---:|---:|---:|---:|
| variance by key (1000 groups) | 50,000,000 | 34.2 ms | 25.9 ms | 0.76x |
| stddev by key (1000 groups) | 50,000,000 | 34.2 ms | 25.3 ms | 0.74x |
| stddev by key (1000 groups) | 10,000,000 | 6.4 ms | 6.4 ms | 0.99x |

The grouped moments accumulate in software IEEE-754 binary64, because Apple GPUs have no double
arithmetic and a float32 accumulator is wrong past a few million rows. The answer is bit-exact against
Arrow; the price is 3–4 GPU instructions per FLOP. Against Polars the same rows win by 2.3–2.4x.

### 6. Float64 `sqrt` and `sort` (4 rows)

| operation | rows | ArrowMetal | fastest CPU | ratio |
|---|---:|---:|---:|---:|
| sqrt (float64) | 10,000,000 | 2.72 ms | 2.39 ms (numpy / pyarrow) | 0.88x |
| sqrt (float64) | 50,000,000 | 12.7 ms | 11.7 ms (pyarrow) | 0.92x |
| sort float64 | 10,000,000 | 29.7 ms | 26.1 ms (Polars) | 0.88x |
| sort float64 | 50,000,000 | 139 ms | 133 ms (Polars) | 0.95x |

`sqrt` is correctly rounded software binary64 (a hardware `rsqrt` seed and Newton steps in emulated
double); the previous float32-evaluated version was 4x faster and wrong in the last bits, and the
project chose correct. The float64 sort is an 8-pass LSD radix sort over the order-preserving key;
Polars' multi-threaded sort is within 5–12% of it. Fewer, wider digits (six 11-bit passes) is the
known next step.

### 7. Regex on the host (1 row)

| operation | rows | ArrowMetal | Polars | ratio |
|---|---:|---:|---:|---:|
| match_substring_regex (real regex) | 1,000,000 | 19.8 ms | 17.8 ms | 0.90x |

A real regex runs on the CPU (RE2-equivalent semantics through Foundation), behind a GPU pre-filter
that clears rows that cannot match. At 10M rows the pre-filter wins 1.06x over Polars and 2x over
pyarrow; at 1M rows the host regex dominates. LIKE patterns and literal substrings are GPU kernels and
win by 20–1300x; only a genuine regex takes this path.

### 8. Two-key group-by (2 rows)

| operation | rows | ArrowMetal | pyarrow | ratio |
|---|---:|---:|---:|---:|
| sum by two int32 keys (~1024 groups) | 10,000,000 | 6.26 ms | 5.68 ms | 0.91x |
| sum by two int32 keys (~1024 groups) | 50,000,000 | 22.7 ms | 22.2 ms | 0.98x |

Two keys go through the general hashed-keys path even when both are small integers whose packed
64-bit value would be a dense key for the fast path (2.7 ms at 10M rows for one int32 key). Packing
narrow key pairs is a small, known change. Against Polars the same rows win by 4–5x.

### 9. Noise and small absolute values (10 rows)

`slice (zero-copy view)` reads 0.00 ms for every library — all are pointer arithmetic, and the ratio
is measurement noise. `list_value_length` at 10M rows is 1.12 ms against 0.71 ms: an offsets
difference that should run at bandwidth and does not yet; it is a real, small loss.

## Faster, but under the 3x bar (115 rows)

The full list is in the matrix page under ⚠️. The clusters:

- **Software binary64 math** — `ln` 1.06–1.33x, `sin` 1.96–2.07x, `days_between` 1.04–1.41x,
  `sqrt` 1.9–2.2x against Polars. Correct to 1 ulp (4–5 ulp for trigonometry); the CPU has hardware
  doubles and the GPU does not. These will not reach 3x without a different numerical contract.
- **Grouped aggregates against pyarrow at 1000 groups** — sum/count/min/max/mean by int32 key at
  1.1–2.2x. pyarrow's grouped kernels are memory-bound and 16-thread; the GPU's advantage grows with
  the number of groups (10M groups: 92–96x) and with wider values.
- **Memory-bound element-wise kernels against Polars** — compare, `is_nan`, `abs`, `bit_wise_and`,
  `if_else`, `replace_with_mask` at 1.2–2.7x. Both sides run at unified-memory bandwidth; the GPU's
  edge is the dispatch overhead it does not pay per thread. Fusing them into one expression
  (`am.query`) is where the 3x comes from, not from the single kernel.
- **Sorting and argsort** at 1.8–2.8x over Polars.
- **Host-assisted strings** — `trim`, `lower`, `upper` at 1M rows (1.8–2.2x; the dispatch floor
  again — at 10M rows they win 2.5x–26x), `parse` 1.4–2.7x, real regex 1.7–2.1x.
- **Joins** at 2.6–3.0x over pyarrow, 3–8x over Polars.

## What changed since the previous matrix

Compared with `full_matrix_2026-09-06.csv` on the same machine: 132 rows faster, 27 slower. The large
wins came from the review wave (assume_timezone 2,900x, LIKE with `_` 1,300x, split 600x, strftime
170x, quantile 158x, grouped min/max at 10M groups 95x, count_distinct 70x). The slower rows are the
binary64 transcendental and sqrt change (correct at the cost of throughput, above), and a regression
in `upper`/`lower`/`trim` (3.7 ms → 54 ms at 10M rows) introduced with full-Unicode case mapping:
the hybrid GPU/host driver allocated and walked an n-sized host array even when no row needed the
host. That is fixed in the commit that adds this page (`upper` 4.6 ms, `trim` 3.0 ms at 10M rows,
results identical to pyarrow) and the matrix row will be re-measured in the next full run.

## Fixed since the matrix

Four of the rows above have been fixed and will leave this page when the matrix is next run. The
tables are still the matrix's, so they still show the old numbers; the numbers below were measured
with `Benchmarks/loss_bench.py` — the same generated columns, the same library idioms and the same
rule as `full_matrix.py` (one warm-up, best of five) — with the before and after runs interleaved on
one machine, the "before" being the previous build loaded through `ARROWMETAL_LIB`.

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
