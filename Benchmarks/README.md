# Benchmarks

Three complementary benchmarks. Numbers per chip live in `../docs/BENCHMARKS.md`.

| Script | What it measures |
|---|---|
| `swift run -c release arrowmetal-bench [rows] [iters]` | Metal kernels vs 16-core Swift loops vs Accelerate (all cores). Same process, same buffers. |
| `python Benchmarks/python_bench.py [rows] [iters]` | Polars, pyarrow.compute, pandas, numpy on data of identical shape (their own memory). |
| `PYTHONPATH=python python Benchmarks/python_gpu_bench.py [rows] [iters]` | ArrowMetal called **from Python** vs Polars/pyarrow/pandas on the **same in-process data**, including the cost of crossing the C boundary. |

Rules we follow so the comparison is fair:
- Best of N after one warm-up (so shader compilation is excluded, as it would be in steady state).
- CPU baselines use all cores, not one. Single-core numbers are not published.
- Bytes counted are the bytes each operation must touch (input + output), so GB/s is comparable across rows.
- Data shapes: Int64 with 10% nulls in [-1000, 1000]; Float32 in [-1, 1]; Float64 in [0, 1000]; keys uniform.
- Sorting uses separate no-null columns (Int64 over the full range, Float64 in [-1e9, 1e9]).
- Strings: 10M utf8 values drawn uniformly from 1000 distinct keys shaped `cust_NNN_region` (13 bytes each).
- Python: a venv with `polars pyarrow pandas numpy` on Python 3.13.
