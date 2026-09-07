# Benchmark history

## 2026-09-07, Apple M4 Max, round 9: the parallel baseline

The matrix now measures every CPU library **twice**: its plain eager idiom, and the most parallel
idiom it has for the same answer — `polars-lazy` is the same expression through `pl.LazyFrame`
collected on the in-memory or the streaming engine, `pyarrow-threaded` is an Acero plan over the same
values split into 16 record batches. pandas has no parallel idiom and says so in its own row; numpy's
ufuncs are single-threaded and say the same. "Fastest CPU" is now the best wall time of *all* of those
idioms, and the `note` column of each CSV row names the exact idiom that was run.

The reason is in `Benchmarks/results/full_matrix_2026-09-07-parallel_cores.txt`, which reports
`cpu_ms / wall_ms` per idiom over every measured row: the eager idioms use a median of **1.0 core** on
most families however many threads their pool has, while Polars lazy uses a median of **11.5** and
pyarrow through Acero **11.1**. A comparison against one core is not the comparison this project wants
to make.

Against that baseline, of **339 measured rows: 145 at or above 3x, 102 between 1x and 3x, 77 slower
than the fastest CPU idiom, 15 with no CPU equivalent.** Against the eager idioms alone, the same
build had 247 rows at or above 3x, 62 between, 15 slower and 15 without an equivalent
(`Benchmarks/results/full_matrix_2026-09-07.csv`). Both CSVs are kept. The 77 slower rows are grouped
by measured cause, with what would change each one, in [LOSSES.md](LOSSES.md); the row-by-row tables
are in [BENCHMARKS_MATRIX.md](BENCHMARKS_MATRIX.md).

Ten ArrowMetal rows that a regression check flagged against the eager run were re-measured on a
quieter machine and eight of them spliced back into the parallel CSV in place; those rows say so in
their `note` column.

Reproduce with `PYTHONPATH=python python Benchmarks/full_matrix.py` (add `--cores` for the cores
table, `--verify` to assert the eager and parallel idioms return the same answer).

## 2026-09-06, Apple M4 Max, round 8: what real binary64 costs

> **Superseded by the 2026-09-07 matrix** — `sqrt` is now 3.92 ms at 50M rows, which is 3.2x the
> fastest *eager* CPU library and 0.87x the fastest parallel one (Polars lazy at 3.42 ms on 12.7
> cores; `Benchmarks/results/full_matrix_2026-09-07-parallel.csv`), so re-running the reproduce line
> below will not produce the table that follows it. Kept for the before/after of removing the float
> detour.

50M Float64 rows, best of 5, release build, against numpy 2.5, pyarrow 25 and Polars 1.44 on the same
data. Reproduce with `PYTHONPATH=python python Benchmarks/float64_math_bench.py`.

`sqrt`, `exp`, `ln`, `log2` and `log10` used to narrow a float64 column to `float`, call Metal's own
library and widen the answer back — seven correct significant decimal digits out of sixteen — and `power`
was not implemented for float64 at all. They now run in software binary64 from end to end: `sqrt`
correctly rounded, the rest within 1 ulp. Here is the bill.

| op | before (float detour) | after (binary64) | GB/s after | fastest CPU | ratio |
|---|---:|---:|---:|---:|---:|
| `sqrt` | 3.2 ms | 12.7 ms | 63.1 | 12.7 (numpy) | 1.00x |
| `exp` | 5.2 ms | 59.4 ms | 13.5 | 76.7 (numpy) | 1.29x |
| `ln` | 3.2 ms | 82.3 ms | 9.7 | 92.9 (numpy) | 1.13x |
| `log2` | 4.5 ms | 76.5 ms | 10.5 | 92.7 (numpy) | 1.21x |
| `log10` | 3.1 ms | 81.9 ms | 9.8 | 105.4 (numpy) | 1.29x |
| `power(x, 2.5)` | *not implemented* | 120.9 ms | 6.6 | 242.7 (pyarrow) | 2.01x |
| `power(x, y)` | *not implemented* | 123.8 ms | 9.7 | 238.8 (numpy) | 1.93x |

Read that honestly: the old kernels were memory bound at 250 GB/s because they were doing float32 work on
float64 data. The new ones are compute bound on forty-odd software binary64 operations per element, and a
logarithm costs 25x more than it used to. It still matches or beats every CPU library on the same machine,
by 1.0-2x rather than by the 25x the old `ln` would have shown — `sqrt` is an exact tie — and the old
`ln`'s answer was wrong in the ninth digit. The float32 kernels are untouched and still run at
200-240 GB/s (`ln` on float32: 1.7 ms, 45x the fastest CPU), so a column that does not need sixteen
digits should not be float64.

Two arithmetic ops moved the other way in the same round, from reworking `DoubleMath` itself — `clz`
normalisation instead of shift loops, four 32x32 partial products instead of an emulated 64x64, and a
Newton reciprocal with an exact remainder correction instead of a 57-step restoring long division:

| op | before | after | GB/s before | GB/s after | ratio vs fastest CPU |
|---|---:|---:|---:|---:|---:|
| `divide` (float64) | 9.6 ms | **3.6 ms** | 125.3 | 331.1 | 1.12x -> **3.00x** |
| `multiply` (float64) | 4.6 ms | **3.1 ms** | 260.9 | 390.8 | 2.32x -> **3.46x** |
| `add` (float64) | 3.1 ms | 3.1 ms | 390.1 | 388.4 | 3.33x |

`multiply` and `add` now sit at the ~390 GB/s these single-pass float64 rows reach and `divide` is within 15% of it,
so the software arithmetic has all but stopped being visible. Both rewrites stay correctly rounded —
`DoubleMathTests` compares them with Swift's `Double` bit for bit.

`divide` measured 3.2 ms (377 GB/s, 3.26x) with **two** Newton steps, which is provably enough while
`MetalContext` compiles with `mathMode = .safe`: that makes the `float` seed correctly rounded and good
to 2^-22, and two steps saturate the 63 bits the reciprocal holds. The shipped code takes a third step
and pays 0.4 ms for it, because the two-step version is only correct *given a compile flag set in another
file* — under fast math the seed would be looser, the quotient would land further than the single
correction step covers, and `d_div` would quietly stop being correctly rounded. 0.4 ms on an operation
already near the memory ceiling is a cheap price for removing that coupling.

**Measure on a quiet machine.** These runs were repeated until two agreed: a second Metal process on the
same GPU inflates the short memory-bound rows (`sqrt` 12.7 -> 30.6 ms in one contended run) while barely
touching the compute-bound ones, which reads as a plausible result rather than an obvious error.

## 2026-09-06, Apple M4 Max, round 7: sort and strings
50M Int64/Float64 rows and 10M utf8 values (1000 distinct keys, `cust_NNN_region`, 130 MB of bytes),
best of 5, release build. CPU time is process user+system time consumed by the call (all threads).

Swift, Metal vs all 16 CPU cores on the same Arrow buffers:

| Operation | Metal | CPU-ms | 16-core CPU | CPU-ms |
|---|---:|---:|---:|---:|
| argsort Int64, 50M | **128.52 ms** | 0.4 | 653.97 (chunk sort + merge tree) | 5017.1 |
| sort Float64, 50M | **138.52** | 0.9 | 591.97 (chunk sort + merge tree) | 4468.6 |
| sort Float64, 50M (1 core, for scale) | | | 4283.22 (`[Double].sort()`) | 4276.2 |
| top_k 100 of 50M Int64 | 128.67 | 0.4 | **1.96** (per-core running top-k) | 25.6 |
| string `contains("north")`, 10M (25% hit) | **1.65** | 0.4 | 17.77 (byte scan) | 250.0 |
| string `starts_with("cust_1")`, 10M | **1.75** | 0.4 | 3.10 | 42.7 |
| string `equals("cust_042_east")`, 10M | **1.13** | 0.4 | 3.81 | 52.4 |
| string filter, 10M (~30% kept) | 4.72 | 2.4 | **2.74** (count, prefix, copy) | 34.1 |
| dictionary_encode + group-by sum, 10M | 749.61 | 747.3 | **13.89** (hash dict + group-by) | 184.3 |
| group-by sum on cached codes (GPU part only) | **1.21** | 0.4 | | |

Called from Python on the same in-process data, against Polars 1.44 (16 threads), pyarrow 25 and numpy 2.5.
Wall ms, with CPU-ms in parentheses:

| Operation | ArrowMetal | Polars | pyarrow | numpy |
|---|---:|---:|---:|---:|
| argsort Int64, 50M | **128.51** (0.5) | 334.27 (3538.1) | 5339.10 (5337.4) | 4675.49 (4672.7) |
| sort Float64, 50M | **138.03** (0.9) | 148.55 (1213.8) | 6321.34 (6315.9) | 1782.05 (1781.2) |
| sort Float64 + export to pyarrow | 138.03 (0.9) | | | |
| top_k 100 of 50M Int64 | 128.38 (0.4) | 68.70 (68.8) | **24.27** (24.3, `select_k_unstable`) | 191.48 (191.5) |
| string `contains("north")`, 10M | **1.61** (0.4) | 147.78 (147.8) | 121.91 (121.9) | 122.31 (pandas) |
| string `starts_with("cust_1")`, 10M | **1.46** (0.4) | 36.00 (36.1) | 32.14 (32.1) | |
| string `equals("cust_042_east")`, 10M | **1.79** (0.4) | 42.24 (42.2) | 36.77 (36.8) | |
| string filter, 10M (~30% kept) | 5.17 (2.9) | **4.19** (4.2) | 49.70 (49.7) | |
| dictionary_encode + group-by sum, 10M | 850.04 (847.6) | 40.59 (361.9) | **15.84** (146.2) | |
| group-by sum on cached codes | **2.13** (0.4) | | | |

Findings:
- The GPU LSD radix sort is 4.6x faster than 16 CPU cores at 50M Int64 and 2.6x faster than Polars, and it
  costs the CPU under a millisecond against Polars' 3.5 CPU-seconds. Wall-clock parity with Polars is closer
  on Float64 (138 vs 149 ms) because the sorted copy adds a gather.
- String predicates are where the GPU is furthest ahead: 1.1 to 1.8 ms against 32 to 148 ms, i.e. 20x to 90x
  Polars and pyarrow, at 100 to 155 GB/s over the utf8 data buffer.
- *As of 2026-09-06, since fixed:* `top_k` was a full argsort plus a slice, so a CPU running top-k
  selection (which touches each value once and rarely writes) won by 65x. A partial radix / threadgroup
  selection kernel was the fix. In the 2026-09-07 matrix `top_k (k=100, int64)` at 50M rows is 2.67 ms
  against pyarrow's 24.00 ms, a 9.0x win.
- *As of 2026-09-06, since fixed:* `dictionaryEncode` ran on the CPU and allocated per row, which
  dominated the string group-by (750 ms of the 751). Once codes existed, the GPU group-by over 10M string
  keys was 1.2 ms, 12x the all-core CPU hash group-by and 13x pyarrow. Hashing moved onto the GPU; in the
  2026-09-07 matrix `dictionary_encode (utf8)` at 10M rows is 5.65 ms against 113.44 ms for the fastest
  CPU idiom (Polars lazy), a 20.1x win.
- String `filter` is the one string kernel the CPU still wins (2.7 vs 4.7 ms): the byte gather is
  short-string dominated, one thread per row copying ~13 bytes.

Reproduce: `swift build -c release && .build/release/arrowmetal-bench 50000000 5`,
`python Benchmarks/python_bench.py 50000000 5`, `PYTHONPATH=python python Benchmarks/python_gpu_bench.py 50000000 5`.


## 2026-09-06, Apple M4 Max, round 6: CPU time per operation (the "CPU stays free" claim, measured)
50M rows, best of 5. CPU time is process user+system time consumed by the call (all threads).

| Operation | Implementation | Wall ms | CPU ms |
|---|---|---:|---:|
| sum Int64, 10% nulls | Metal | 1.07 | 0.4 |
| sum Int64, 10% nulls | 16-core Swift | 4.87 | 66.6 |
| sum Int64, 10% nulls | Polars (from Python) | 15.65 | 15.6 |
| filter Int64 > 0 | Metal | 2.95 | 1.0 |
| filter Int64 > 0 | 16-core Swift | 3.49 | 48.8 |
| filter Int64 > 0 | Polars | 22.79 | 22.7 |
| take 25M random indices | Metal | 5.91 | 0.8 |
| take 25M random indices | 16-core Swift | 11.75 | 154.6 |
| take 25M random indices | Polars gather | 163.92 | 163.9 |
| group-by sum, 5 keys | Metal | 1.72 | 0.4 |
| group-by sum, 5 keys | 16-core Swift | 5.67 | 83.1 |
| group-by sum, 1000 keys | Metal (from Python) | 1.91 | 0.4 |
| group-by sum, 1000 keys | Polars | 84.06 | 1182.7 |
| group-by sum, 1000 keys | pyarrow | 18.67 | 250.8 |
| query: filter two columns + sum | Metal, batched | 1.75 | 0.4 |
| query: filter two columns + sum | 16-core Swift fused loop | 6.27 | 86.5 |
| query: filter two columns + sum | Polars lazy | 15.81 | 26.6 |
| Float64 compare + filter | Metal | 3.37 | 0.8 |
| Float64 compare + filter | 16-core Swift | 5.86 | 73.7 |

A GPU query costs the CPU well under a millisecond; the same work on the CPU costs 50 to 1200
CPU-milliseconds, which is time the rest of the application does not get.

Float64 add, sub, mul and div now run on the GPU through software IEEE-754 and are bit-exact against
Swift's `Double`; `sum` accumulates in tree order, so it differs from a sequential CPU sum by reordering
only (see round 6 in docs/FINDINGS.md).


## 2026-09-06, Apple M4 Max, round 5: lengths flow on the GPU
5-op chain (compare, compare, and, filter, sum), µs per call, best of 30:

| rows | unbatched | batched (round 4, two syncs) | batched (round 5, one sync) |
|---:|---:|---:|---:|
| 1,000 | 642 | 346 | 256 |
| 10,000 | 580 | 320 | 232 |
| 100,000 | 573 | 330 | 243 |
| 1,000,000 | 872 | 549 | 428 |
| 10,000,000 | 1,344 | 1,047 | 970 |


## 2026-09-06, Apple M4 Max, round 4: latency and batched execution
Fixed cost per call (µs, best of 30). "5-op chain" is compare, compare, and, filter, sum.

| rows | GPU sum | CPU 1-core sum | GPU filter | CPU 1-core filter | GPU group-by | chain unbatched | chain batched |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 152 | 0 | 167 | 1 | 144 | 624 | 346 |
| 10,000 | 144 | 1 | 169 | 8 | 165 | 575 | 320 |
| 100,000 | 136 | 6 | 153 | 223 | 198 | 590 | 330 |
| 1,000,000 | 228 | 64 | 244 | 2,351 | 497 | 853 | 549 |
| 10,000,000 | 315 | 798 | 625 | 23,726 | 615 | 1,390 | 1,047 |

Raw Metal floor on this machine: empty kernel encode + commit + wait 116 µs; ten kernels in one command
buffer 100 µs total. The GPU overtakes one CPU core at roughly 100k rows for filter and 1M rows for sum.

Query at 50M rows: unbatched 2.21 ms, batched 1.88 ms, 16-core CPU 6.46 ms, Polars lazy 15.68 ms.
From Python: 2.31 ms unbatched, 1.90 ms with `with am.batch():`.


## 2026-09-06, Apple M4 Max, round 3: group-by, fused query, and ArrowMetal called from Python
Swift side (Metal vs 16-core Swift):

| Operation | Metal | 16-core Swift |
|---|---:|---:|
| fused filter(where Int64 > 0) | 3.60 ms | 3.49 (count/prefix/scatter) |
| group-by sum Int64, 5 keys (privatised) | 1.96 | 5.92 |
| group-by sum Int64, 1000 keys (privatised) | 2.02 | |
| group-by sum Int64, 100000 keys (device atomics) | 4.02 | |
| query: sum(amount) where region == 2 and amount > 100 | 2.33 | 6.36 (fused loop) |

Python side, same in-process data, ArrowMetal through the C ABI vs Polars 1.44 (16 threads), pyarrow 25, pandas 3.0:

| Operation | ArrowMetal from Python | Polars | pyarrow.compute | pandas |
|---|---:|---:|---:|---:|
| sum Int64, 10% nulls | 1.05 ms | 15.78 | 48.26 | 47.75 |
| filter Int64 > 0 (fused) | 3.55 | 22.99 | 211.05 | 271.50 |
| filter + export back to pyarrow | 3.57 | | | |
| take 25M random indices | 5.98 | 164.82 | 138.61 | |
| group-by sum, 1000 keys | 2.05 | 84.65 | 18.73 | |
| query: filter two columns + sum | 3.13 | 16.41 (lazy, fused) | | 109.46 (numpy) |

Importing a 50M-row Int64 column from pyarrow into Metal memory (one copy, pyarrow buffers are not page aligned)
took about 60 ms; every later operation on it is zero-copy in and out.


All runs: best of 5, release build, 50,000,000 rows. Hardware is named per run. Reproduce with
`swift run -c release arrowmetal-bench` and `python Benchmarks/python_bench.py` (venv with polars, pyarrow, pandas).

## 2026-09-06, Apple M4 Max (16 CPU cores), round 2: buffer pool, all-core baselines
Metal vs 16-core Swift vs Accelerate (all cores) vs Polars 1.44 (16 threads) vs pyarrow 25 vs pandas 3.0.

| Operation | Metal | 16-core Swift / Accelerate | Polars | pyarrow | pandas |
|---|---:|---:|---:|---:|---:|
| sum Int64, 10% nulls | 1.40 ms | 4.89 | 15.87 | 46.81 | 46.61 |
| min Int64, 10% nulls | 1.34 | 3.73 | 12.65 | 45.97 | 45.61 |
| compare Int64 > 0 | 1.07 | 2.12 | 4.48 | 5.89 | 6.09 |
| filter Int64 (45% kept) | 2.90 | 3.49 | 17.85 | 197.86 | 254.67 |
| compare + filter | 4.07 | 5.62 | 22.54 | 205.38 | 263.24 |
| multiply Int64 * 3 | 2.13 | 2.26 | 23.54 | 12.95 | 53.35 |
| take 25M random indices | 5.85 | 11.93 | 162.47 | 132.94 | 160.72 |
| cast Int64 to Float32 | 3.14 | 1.80 | 71.56 | 54.62 | |
| Float64 compare + filter | 3.40 | 5.96 | 22.31 | 103.00 | 189.81 |
| sum Float32 | 0.90 | 0.84 (vDSP) | 2.17 | 6.27 | 5.31 (numpy) |
| max Float32 | 0.88 | 0.79 (vDSP) | 2.99 | 22.39 | 2.08 (numpy) |
| multiply Float32 * 2.5 | 1.72 | 1.63 (vDSP) | 10.87 | 3.22 | 3.21 (numpy) |
| Float32 compare + filter | 2.14 | 5.73 | 15.40 | 94.50 | 165.43 (numpy) |

Notes: the Metal numbers include allocation of the output from the pool and the CPU-side wait. Polars
numbers are eager single-expression calls; its lazy engine would fuse compare + filter.

## 2026-09-06, Apple M4 Max, round 1 (before the pool, single-core baselines)
Kept for history: sum Int64 1.07 ms, compare 1.18, filter 5.36, multiply 6.84, cast 5.04, take 8.09.
The pool removed 2 to 3x of allocation overhead from write-heavy kernels.
