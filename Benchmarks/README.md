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
| `PYTHONPATH=python python Benchmarks/full_matrix.py [--quick]` | **Every** operation family, against both the eager and the most parallel idiom of each CPU library, with a pass/fail verdict per row and the cores each idiom used. |

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
- `--str-sizes` and `--small-sizes` narrow the string and latency families the way `--sizes` narrows the rest.
- `--verify [--sizes N]` checks answers instead of timing them; `--cores <csv>` prints the cores-per-idiom
  summary for a CSV that has already been measured.

Extra fairness rules this script follows on top of the ones below:
- An operation a library does not have is recorded as "no equivalent" with the reason, and one that
  raises is recorded with the exception message. Nothing is dropped silently, on either side.
- Each baseline uses that library's fastest eager idiom: Polars lazy where it fuses, pyarrow.compute kernels
  rather than table wrappers, pandas vectorised (arrow-backed when the column has nulls, numpy-backed
  when it does not, since that is what pandas is fastest with). No Python loops in any baseline.
- **Every CPU library is measured twice: eager, and in its most parallel idiom.** What the cores table
  shows about the eager idioms is not one number: they use about one core on the element-wise rows and
  the whole-column reductions -- `Series.sum()` and `pc.add(...)` do not fan out however many threads
  the pool has -- and several cores on group-by, sort and join, where the library reaches its own
  parallel machinery unprompted (pyarrow's eager group-by rows sit near six cores, because Acero splits
  even a single chunk into several ExecBatches). So each operation is measured again through
  `pl.LazyFrame` (`polars-lazy`, in-memory or streaming engine) and through Acero over the same values
  in one record batch per hardware thread (`pyarrow-threaded`, `to_table(use_threads=True)`, or
  `pa.Table.group_by` over that chunked table). These land in the CSV as their own library rows with
  the idiom named in the `note` column; the default rows are unchanged, and "fastest CPU" in the report
  is the best of all of them. pandas gets a `pandas-parallel` row recording why it has none: its kernels
  are single-threaded by design, and its two threaded paths -- numexpr behind `pd.eval`, and the numba
  engine with `parallel=True` behind `rolling`, `groupby.agg`/`transform` and `apply` -- are looked up
  at run time and named in the note, present or absent.
- The pyarrow plan's source table carries only the columns the operation reads. An Acero filter node
  emits its whole input schema, so a wider source would materialise columns the eager `pc.filter` row
  never touches, and the parallel row would be handicapped rather than helped.
- `--verify` runs every operation once instead of timing it and asserts each parallel idiom returns the
  same answer as its library's default idiom, within 1e-9 relative for floats (a threaded reduction adds
  in a different order); a grouped result is compared as a set of rows. Two comparisons are deliberately
  skipped and reported as such: t-digest, whose sketch depends on how the values were partitioned, and
  pyarrow's threaded grouped `list`, whose element order inside a group follows batch arrival.
- `--cores`, and `full_matrix_<date>_cores.txt` written next to every results CSV, report cpu_ms/wall_ms
  -- the cores actually used -- per library and idiom, overall and per family. ArrowMetal's own number
  there is host CPU time only: the thread that encodes the command buffer and waits on it. GPU execution
  time is not in it, and neither clock counts it.
- A single call that takes longer than the per-measurement budget is repeated fewer times (never fewer
  than twice), so one six-second sort does not cost a minute of the run. The count is in the CSV.
- numpy rows carry the same values without a validity bitmap, because numpy has no null representation.

Rules we follow so the comparison is fair:
- Best of N after one warm-up (so shader compilation is excluded, as it would be in steady state).
- CPU baselines are handed every core the machine has (Polars and pyarrow both run 16-thread pools here;
  no pool is limited). Whether a given idiom *uses* them is a property of the idiom, not of the harness,
  so `full_matrix.py` measures each library's most parallel idiom alongside its eager one and publishes
  cpu_ms/wall_ms -- the cores each row actually used -- for every row. Nothing is quoted as "on all
  cores" that the CSV does not show running on them.
- Bytes counted are the bytes each operation must touch (input + output), so GB/s is comparable across rows.
- Data shapes: Int64 with 10% nulls in [-1000, 1000]; Float32 in [-1, 1]; Float64 in [0, 1000]; keys uniform.
- Sorting uses separate no-null columns (Int64 over the full range, Float64 in [-1e9, 1e9]).
- Strings: 10M utf8 values drawn uniformly from 1000 distinct keys shaped `cust_NNN_region` (13 bytes each).
- Python: a venv with `polars pyarrow pandas numpy` on Python 3.13.

## polars_bench.py

`PYTHONPATH=python python Benchmarks/polars_bench.py [rows] [iterations]` compares native Polars
against the three ways of reaching the GPU from Polars — the `.arrowmetal` namespaces, the Rust
expression plugin inside a lazy plan, and the same kernels with the columns already resident —
for sum, filter+sum, group-by, top-k and string contains, in wall-ms and CPU-ms. It also prints
the hand-off cost and the buffer addresses that prove the import copies nothing. The published
tables at 10M and 50M rows are in [../docs/POLARS.md](../docs/POLARS.md). The plugin rows are
skipped when `polars-plugin/target/release/libarrowmetal_polars.dylib` has not been built.
