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

Four of the rows above have been re-measured on the same idle M4 Max after the work described here.
The numbers are from `Benchmarks/loss_sort_shift_sqrt.py`, which uses `full_matrix.py`'s columns, seed
and rule (one warm-up, best of five), run before and after alternately in separate processes. The
matrix tables above are the record of the 2026-09-07 run and are not edited; these rows will move when
the matrix is next run in full.

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
