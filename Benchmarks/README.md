# Benchmarks

Six complementary benchmarks. Numbers per chip live in `../docs/BENCHMARKS.md`, and the complete
operation-by-operation matrix in `../docs/BENCHMARKS_MATRIX.md`.

| Script | What it measures |
|---|---|
| `swift run -c release arrowmetal-bench [rows] [iters]` | Metal kernels vs 16-core Swift loops vs Accelerate (all cores). Same process, same buffers. |
| `python Benchmarks/python_bench.py [rows] [iters]` | Polars, pyarrow.compute, pandas, numpy on data of identical shape (their own memory). |
| `PYTHONPATH=python python Benchmarks/python_gpu_bench.py [rows] [iters]` | ArrowMetal called **from Python** vs Polars/pyarrow/pandas on the **same in-process data**, including the cost of crossing the C boundary. |
| `PYTHONPATH=python python Benchmarks/expr_bench.py [rows] [iters]` | The fused expression compiler: one runtime-generated kernel for a whole expression, against the same expression built op by op, Polars, pyarrow, pandas and numpy (see `../docs/EXPR.md`). |
| `PYTHONPATH=python python Benchmarks/engine_bench.py [rows] [iters]` | The lazy query engine: eight TPC-H-flavoured query shapes at 50M rows (plus a 10M x 1M join, a 50M as-of join and a window over 1000 partitions) against Polars lazy and DuckDB on the same Arrow buffers (see `../docs/ENGINE.md`). |
| `PYTHONPATH=python python Benchmarks/full_matrix.py [--quick]` | **Every** operation family, against the fastest idiom of each CPU library, with a pass/fail verdict per row. |

## full_matrix.py: the complete comparison

`python_gpu_bench.py` is a sample of headline operations. `full_matrix.py` is the whole surface: every
family the Python package exposes — reductions, element-wise math, compare/filter/take, the sort family,
group-by at 1k / 100k / 10M groups over int, utf8 and two-column keys, joins, strings, temporal, window
and rolling, decimals, nested types, batched query chains and the small-row latency floor — measured
against Polars, pyarrow.compute and pandas (numpy where pandas has no vectorised equivalent).

- Sizes: 10M and 50M rows, 1M and 10M for string columns, plus 1k / 100k / 1M for the latency floor.
- Output: `Benchmarks/results/full_matrix_<date>.csv` (family, op, rows, library, wall_ms, cpu_ms,
  GB/s, iterations, status, note) and `docs/BENCHMARKS_MATRIX.md`, which adds the ratio against the
  fastest baseline per row, a verdict (✅ at or above 3x, ⚠️ 1-3x, ❌ slower) and a SHORTFALL section
  listing every row below 3x with the most likely cause, worst first.
- `--quick` runs the same matrix at 1M rows in a few minutes and writes into `Benchmarks/results/`
  instead of `docs/`. Use it while editing the script; the full run takes roughly half an hour.
- `--families`, `--sizes`, `--iters` and `--budget` narrow a run while investigating one operation.
- `--report-from <csv> --elapsed-min <n>` rebuilds `docs/BENCHMARKS_MATRIX.md` from an existing results
  CSV, so the write-up can be corrected without spending half an hour re-measuring.

Extra fairness rules this script follows on top of the ones below:
- An operation a library does not have is recorded as "no equivalent" with the reason, and one that
  raises is recorded with the exception message. Nothing is dropped silently, on either side.
- Each baseline uses that library's fastest idiom: Polars lazy where it fuses, pyarrow.compute kernels
  rather than table wrappers, pandas vectorised (arrow-backed when the column has nulls, numpy-backed
  when it does not, since that is what pandas is fastest with). No Python loops in any baseline.
- A single call that takes longer than the per-measurement budget is repeated fewer times (never fewer
  than twice), so one six-second sort does not cost a minute of the run. The count is in the CSV.
- numpy rows carry the same values without a validity bitmap, because numpy has no null representation.

Rules we follow so the comparison is fair:
- Best of N after one warm-up (so shader compilation is excluded, as it would be in steady state).
- CPU baselines use all cores, not one. Single-core numbers are not published.
- Bytes counted are the bytes each operation must touch (input + output), so GB/s is comparable across rows.
- Data shapes: Int64 with 10% nulls in [-1000, 1000]; Float32 in [-1, 1]; Float64 in [0, 1000]; keys uniform.
- Sorting uses separate no-null columns (Int64 over the full range, Float64 in [-1e9, 1e9]).
- Strings: 10M utf8 values drawn uniformly from 1000 distinct keys shaped `cust_NNN_region` (13 bytes each).
- Python: a venv with `polars pyarrow pandas numpy` on Python 3.13.
