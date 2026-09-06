# Benchmark history

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

Float64 sum, add, sub, mul, div now run on the GPU through software IEEE-754 (bit-exact; see round 6 in
docs/FINDINGS.md).


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
