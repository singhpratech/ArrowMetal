# Testing

How ArrowMetal is tested, what each layer compares against, and what "green" means before anything is
pushed. Numbers are from the last gated run of `main` (0.1.0, unreleased) on an M4 Max.

| Layer | Size | Oracle |
|---|---|---|
| Swift suites (`Tests/ArrowMetalTests`) | 764 tests in 60 files, run in release | plain-Swift CPU references, hand-computed vectors, pyarrow 25.0.1 answers pinned as literals where noted |
| Python suites (`python/tests`) | 2,460 collected cases over the ctypes API and the three integrations | `pyarrow.compute`, Polars, DuckDB, pandas |
| Differential matrix (`python/tests/test_differential.py`, `differential_report.py`) | 39,069 generated cases, 45 column types, every public operation | `pyarrow.compute`, option by option ([EVALUATION.md](EVALUATION.md)) |
| TypeScript suites (`node/test`) | 58 tests in 5 files over the N-API addon | Apache Arrow JS 21.2.0 and plain JS over the same rows ([TYPESCRIPT.md](TYPESCRIPT.md)) |
| Adversarial review pass | four independent reviewers plus a coverage pass before release | each finding carries a regression test |
| Benchmarks (`Benchmarks/`) | 339 operation-and-size rows over 173 operations, against four CPU libraries; streaming and engine benches | measured, never estimated ([BENCHMARKS_MATRIX.md](BENCHMARKS_MATRIX.md)) |

Run everything:

```
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
swift build -c release
swift test -c release                                   # release is required: a release-only miscompile has bitten this project once
PYTHONPATH=python python -m pytest python/tests -q      # all Python suites, including the differential file
PYTHONPATH=python python python/tests/differential_report.py   # the matrix as one report; exit 0 = nothing unclassified
(cd node && npm install && npm test)                    # TypeScript: builds the addon, then 58 node:test cases
```

Tests that need a real GPU skip on virtual Metal devices (`requireRealGPU()`), so a hosted CI runner
exercises the host paths only; the numbers above are from a physical Mac.

## 1. Swift suites

A CPU reference sits behind the kernels (`Sources/ArrowMetal/CPUReference.swift` and per-suite oracles) and
each is compared against it across types, sizes, null densities and input shapes. Sizes are chosen to cross
every threadgroup and simdgroup boundary (0, 1, 31, 32, 33, 1023, 1024, 1025, 65535, 65536, 65537, and
odd lengths above a million), null ratios are 0, 0.3 and 1.0, and every selection entry point is run on
a sliced input at offsets 1, 7, 31, 32, 33, 63 and 64 against the same rows built standalone.

| Suite family | Files | What is pinned |
|---|---|---|
| Kernels and types | KernelTestsV2, ArrowMetalTests, TypesExtraTests, DecimalTests, NestedTests, StructuralTests, SliceOffsetTests, ConditionalTests, CheckedTests | every reduction, arithmetic, compare, cast, filter/take/slice, conditional and checked kernel on every type, with nulls, against the CPU reference; nested, decimal, dictionary and run-end arrays |
| Software binary64 | DoubleTranscendentalTests, TrigTests, MathKernelTests, MathExtraTests | sqrt correctly rounded, exp/ln/log2/log10/power within 1 ulp, trigonometry within 4–5 ulp of the host libm, over random and edge-case inputs |
| Sorting and selection | TopKTests, SegmentedTests, StringSortTests, SelectionExtraTests, PartitionNthNaNTests, AdversarialRankTests | stable radix sort in both directions, null placement, IEEE total order, top-k for every k against the full sort, rank with all four tiebreakers against pyarrow's answers, partition_nth with NaN |
| Hashing and grouping | HashTableTests, StringHashTableTests, UniqueTests, GroupByTests, GroupByKeysTests, AggregateTests, AggregatesExtraTests, WindowTests | unique/value_counts/dictionary_encode, dense and arbitrary-key group-by, grouped moments, first/last, window functions; crafted hash collisions (1,000 keys folding to one 32-bit value) |
| Strings | StringTests, StringExtraTests, StringTransformTests, TextTests, RegexPrefilterTests | lengths, predicates, MurmurHash3 vectors, LIKE with every wildcard shape, split, full-Unicode case mapping including byte-length changes, the regex pre-filter's literal claims |
| Temporal | TemporalTests, TemporalExtraTests, OptionsTests | calendar fields, rounding with every RoundTemporalOptions flag, timezone transitions, strftime/strptime |
| Execution model | BatchTests, AsyncTests, ResidentTests, AdversarialStringAndBatchTests | batched chains equal unbatched, a throwing body leaves no batch open, results readable after commit, four workers over two contexts; the persistent-kernel negative result |
| Interop | CInteropTests, IPCTests, RecordBatchTests | C Data / C Device / C Stream import and export with offsets and release callbacks, IPC round trips, struct and stream import |
| Expression compiler and engine | ExprTests, ExprLiteralTypeTests, EngineTests, JoinTests | fused expressions against the unfused kernels, literal promotion, every optimizer rule against the unoptimized plan, six join kinds and as-of against a dictionary oracle |
| Parquet | ParquetTests, ParquetWriterTests | pyarrow-written fixtures across encodings and compressions, damaged files that must error rather than trap |
| Streaming | StreamTests | every streaming operator against the in-memory answer at 1–120 batches with ragged and empty batches, HyperLogLog within 3σ, external sort over 120 runs, grace join against the in-memory join |
| Adversarial pins | AdversarialKernelTests, AdversarialKernelTests2, AdversarialGroupSliceTests | the probes the review pass ran that came back clean, kept so they stay clean: 100–200 repeat determinism loops, boundary lengths, slice-of-slice, group-by with one group and with one group per row, threadgroup-size invariants |

## 2. Python suites

The column below counts `def test_*` functions. The 2,460 in the table at the top of this page is what
pytest *collects*, which is larger because a parametrised function collects once per parameter set.

| File | Test functions | Compares against |
|---|---|---|
| `test_arrowmetal.py` | 114 | `pyarrow.compute` and plain Python over the ctypes API, zero-copy import/export, the wheel loader |
| `test_options.py`, `test_checked.py`, `test_strings_extra.py`, `test_float64_math.py` | 113 | option surfaces (null placement, tiebreakers, cast safety, rounding), checked arithmetic, string kernels, binary64 math |
| `test_functions.py` | 8 | executes every runnable row of the Arrow-name registry through `call_function` against `pyarrow.compute`, with a second input in another type family where the claim spans several |
| `test_expr.py`, `test_lazy.py`, `test_lazy_optimizer.py`, `test_lazy_memory.py` | 84 | fused expressions and lazy plans against Polars and pyarrow; the optimizer against the unoptimized plan; RSS over 30,000 queries |
| `test_parquet.py`, `test_parquet_robustness.py` | 31 | `pyarrow.parquet.read_table` on generated files; fuzzed and damaged files |
| `test_stream.py` | 27 | streaming results against pyarrow and Polars, including a 2 GB IPC directory generated at test time |
| `test_polars.py` | 53 | the bridge and namespaces against native Polars; the 8 Rust-plugin tests run when `polars-plugin/` is built |
| `test_duckdb.py` | 38 | the bridge against DuckDB SQL; the 11 extension tests run when `duckdb-extension/build.sh` has produced the extension, and one test compiles the public C header as C |
| `test_pandas.py` | 72 | the accessor and accel mode against plain pandas across five null-carrying dtype flavours; `install()`/`uninstall()` restore every patched slot |
| `test_differential.py` (standalone part) | 100 plus 17 documented xfails | one test per finding and per fixed finding, plus the guards that every public operation and every module-level function has a matrix case |

## 3. The differential matrix

`differential_report.py` builds, for every public operation, a generated column of each type it accepts
in several shapes (random, sorted, all-equal, special values, three null ratios, plain and sliced) and
compares the ArrowMetal answer with the `pyarrow.compute` answer for the same options. The last run:
39,069 cases, 36,486 pass, 1,566 documented divergences, 1,017 skips (an operation that does not apply
to a type), **0 unclassified**. A divergence counts as documented only if it matches one of the 22
open findings in `FINDINGS`, each of which has a section in [EVALUATION.md](EVALUATION.md) with a
reproduction; a finding may be narrowed by the data (`data_check`) so a dataset that does not contain
the triggering value still has to agree exactly. The report exits non-zero on any unclassified
divergence, which is what gates a push. Findings that were later fixed move to "Findings that were
fixed" with a plain regression test, so a relapse is a failure rather than a re-classified divergence.

`test_every_public_operation_has_a_differential_case` and its module-level twin fail the suite when a
method is added without a matrix case, so the harness cannot silently fall behind the library.

## 4. Adversarial review before release

Four independent reviewers (integrations; engine and expression compiler; GPU kernels; C ABI, bindings
and Parquet) and one coverage pass each received a checklist of hostile inputs (sliced and chunked
inputs, null-only columns, crafted hash collisions, corrupt files, threadgroup-boundary lengths, escape
sequences, Unicode edge cases, 30,000-call leak loops, fuzzed C entry points). Every finding they
reported is either fixed with a regression test or recorded as a finding in EVALUATION.md; the fixes are
listed under "Fixed" in the CHANGELOG. Nothing from that pass was accepted on the reviewer's word: each
finding's test fails on the commit before its fix.

## 5. What "green" means before a push

Every merge to `main` runs, on a quiet machine, in this order, and pushes only if all pass:

1. `swift build -c release` with no errors.
2. `swift test -c release`: every test, 0 failures (skips are the virtual-GPU guards).
3. `differential_report.py` exits 0: 0 unclassified divergences.
4. `pytest python/tests` (every suite): 0 failures; xfails must be strict and tied to a finding.
5. The standalone tests in `test_differential.py`: 0 failures.

A benchmark comparison is never part of the gate, because timings on a loaded machine are noise; the
benchmark matrix is rerun on an idle machine before its numbers are published.

## 6. Benchmark method

`Benchmarks/full_matrix.py` measures every operation family the Python package exposes against the
fastest idiom of pyarrow, Polars and pandas on the same in-process data at 10M and 50M rows (1M and 10M
for strings): one warm-up call, then best-of-5 wall time under a per-measurement budget; CPU time is the
process user+system delta so a 16-thread CPU kernel shows about 16× its wall time; every library gets
the same values (pyarrow the Arrow array, Polars a Series built from it, pandas an Arrow-backed Series
when nulls are present and a numpy-backed one otherwise); an operation ArrowMetal lacks is an error row,
never a skip. One measured caveat: the first operation in a family that asks the buffer pool for
hundreds of megabytes can read up to 60% slow while the pool is cold (`partition_nth_indices` measured
7.9 ms first in a fresh process and 5.1 ms after any large operation, on the same build), so a row that
moves between runs is re-measured in place before it is called a regression. The streaming bench runs
every (workload, engine) cell in its own subprocess so peak RSS is that engine's alone. Results land in `Benchmarks/results/*.csv` and are rendered into
[BENCHMARKS_MATRIX.md](BENCHMARKS_MATRIX.md); losses stay in the table.

## 7. Hardware and toolchain of the published numbers

Apple M4 Max (16 cores, 64 GB unified memory, internal SSD), macOS 26.6.2, Swift 6.3.3 (Xcode
toolchain), Python 3.13, pyarrow 25.0.1, Polars 1.44.1, DuckDB 1.5.5, pandas 3.0.5.
