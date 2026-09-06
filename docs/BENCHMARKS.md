# Benchmark history

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
- `top_k` is currently a full argsort plus a slice, so a CPU running top-k selection (which touches each
  value once and rarely writes) wins by 65x. A partial radix / threadgroup selection kernel is the fix.
- `dictionaryEncode` still runs on the CPU and allocates per row, which dominates the string group-by
  (750 ms of the 751). Once codes exist, the GPU group-by over 10M string keys is 1.2 ms, 12x the all-core
  CPU hash group-by and 13x pyarrow. Hashing on the GPU (`hash32` already exists) is the obvious next step.
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
