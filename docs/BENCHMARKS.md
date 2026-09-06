# Benchmark history

All runs: best of 5, release build, 50,000,000 rows. Hardware is named per run. Reproduce with
`swift run -c release arrowmetal-bench` and `python Benchmarks/python_bench.py` (venv with polars, pyarrow, pandas).

## 2026-09-06, Apple M4 Max (16 CPU cores), v0.2 + buffer pool
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

## 2026-09-06, Apple M4 Max, v0.1 (before the pool, single-core baselines)
Kept for history: sum Int64 1.07 ms, compare 1.18, filter 5.36, multiply 6.84, cast 5.04, take 8.09.
The pool removed 2 to 3x of allocation overhead from write-heavy kernels.
