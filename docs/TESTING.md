# Testing

How ArrowMetal is tested, what each layer compares against, and what "green" means before anything is
pushed. Numbers are from the last gated run of `main` (0.1.0) on an M4 Max.

| Layer | Size | Oracle |
|---|---|---|
| Swift suites (`Tests/ArrowMetalTests`) | 943 tests in 73 files, run in release (all 943 executed, 3 skipped in the merge gate of 2026-09-24 at `89d4ed4`: the three opt-in throughput measurements) | plain-Swift CPU references, hand-computed vectors, pyarrow 25.0.1 answers pinned as literals where noted |
| Python suites (`python/tests`) | 5,892 collected cases over the ctypes API, the integrations and the readers in the merge gate of 2026-09-24 at `89d4ed4` (5,867 passed and 25 skipped; the differential matrix in `test_differential.py` is counted in its own row) | `pyarrow.compute`, Polars, DuckDB, pandas |
| Rust suites (`rust/arrowmetal/tests`, `rust/arrowmetal-sys`) | 48 tests, 46 in the safe crate against arrow-rs plus 2 in `arrowmetal-sys` over the raw ABI, run in release, plus 4 `no_run` doc-tests (compiled, not executed); all passed on 2026-09-24 at `3c3ea1e` | `arrow::compute` (arrow-rs 59) on the same data; a `HashMap` fold where arrow-rs has no kernel; `include/arrowmetal.h` re-parsed for the ABI signatures ([RUST.md](RUST.md)) |
| Differential matrix (`python/tests/test_differential.py`, `differential_report.py`) | 39,069 generated cases, 45 column types, every public operation | `pyarrow.compute`, option by option ([EVALUATION.md](EVALUATION.md)) |
| TypeScript suites (`node/test`) | 62 tests in 6 files over the N-API addon (62 passed on 2026-09-24 at `3c3ea1e`) | Apache Arrow JS 21.2.0 and plain JS over the same rows ([TYPESCRIPT.md](TYPESCRIPT.md)) |
| Go binding (`go/arrowmetal`) | 46 test functions and one `Example`, 47 runnable, and 131 subtests: 178 passing results as `go test -count=1 -v ./...` reported them on 2026-09-24 at `3c3ea1e`, 0 failed, run twice (plain and under the cgo pointer checker, `GOEXPERIMENT=cgocheck2`, with the same counts) | `arrow-go/v18`'s own `compute` where it has the function, plain Go loops where it does not ([GO.md](GO.md)) |
| R suites (`r/arrowmetal/tests/testthat`) | 68 `test_that()` blocks in the sources (69 as testthat runs them: the one in test-dispatch.R runs once per attach order), 273 expectations (0 failed, 0 skipped on 2026-09-17 and again on 2026-09-24 at `3c3ea1e`; the four blocks in test-carriers.R cover the ArrowArray/ArrowSchema carriers the shim allocates: release on drop without import, refusal of a moved or unfilled pair, tag checks), over the 34 ABI entry points the R binding wraps | base R and the `arrow` R package's own kernels on the same data ([R.md](R.md)) |
| Adversarial review pass | four independent reviewers plus a coverage pass before release | each finding carries a regression test |
| Benchmarks (`Benchmarks/`) | 339 operation-and-size rows over 173 operations, against four CPU libraries in two idioms each — the plain eager one and the most parallel one that library has for the same answer (`polars-lazy`, `pyarrow-threaded`); streaming and engine benches | measured, never estimated; the baseline is the fastest idiom of any library, and against it 145 rows are at or above 3x, 102 between 1x and 3x, 77 to improve, where the fastest CPU idiom is ahead, and 15 without a CPU equivalent ([BENCHMARKS_MATRIX.md](BENCHMARKS_MATRIX.md), [TO_IMPROVE.md](TO_IMPROVE.md)) |

Run everything:

```
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
swift build -c release
# release is required: a release-only miscompile has bitten this project once
swift test -c release
# all Python suites, including the differential file
PYTHONPATH=python python -m pytest python/tests -q
# the matrix as one report; exit 0 = nothing unclassified
PYTHONPATH=python python python/tests/differential_report.py
# TypeScript: builds the addon, then 62 node:test cases
(cd node && npm install && npm test)
# the Rust binding, against arrow-rs's own kernels
(cd rust && cargo test --release)
# the Go binding
(cd go/arrowmetal && ARROWMETAL_LIB=$PWD/../../.build/release/libArrowMetalC.dylib go test ./...)
# and with the cgo pointer checker
(cd go/arrowmetal && ARROWMETAL_LIB=$PWD/../../.build/release/libArrowMetalC.dylib GOEXPERIMENT=cgocheck2 go test -count=1 ./...)
# the R binding
ARROWMETAL_LIB=$PWD/.build/release/libArrowMetalC.dylib \
  Rscript -e 'testthat::test_local("r/arrowmetal")'
```

The R and Go bindings compile against copies of `include/arrowmetal.h` and `include/arrow_abi.h` (`r/arrowmetal/src/`, `go/arrowmetal/include/`); when a header changes, copy it over both, and `python/tests/test_header_copies.py` fails until you do.

The Rust suite finds `libArrowMetalC.dylib` in `.build/release` on its own; from outside the
repository, set `ARROWMETAL_LIB` to the dylib's full path.

The R suite needs R with `arrow` and `testthat`; `R CMD INSTALL r/arrowmetal` first, or point
`ARROWMETAL_LIB` at the dylib as above. A conda-built R names its own compiler in `Makeconf`, so
activate the environment (or put its `bin` on `PATH`) before installing; setting `CC = clang` in
`~/.R/Makevars` to use Xcode's clang works too. `R CMD check --no-manual` on the built tarball is
the fuller gate; the run recorded in `private/keep/2026-09-08/final_gate_b2dc7fa.log` is the testthat suite
(266 passed, 0 failed), not `R CMD check`, so no `R CMD check` result is claimed here.

Tests that need a real GPU skip on virtual Metal devices (`requireRealGPU()`), so a hosted CI runner
exercises the host paths only; the numbers above are from a physical Mac.

**The router and the suites.** The CPU/GPU router ([DESIGN.md](DESIGN.md#cpugpu-router)) sends small
inputs of seven operations to a CPU loop under its default `auto` mode, and the suites are full of small
inputs. So three harnesses pin the router to the GPU and keep exercising the kernels: Swift's
`TestSupport.swift` sets the process mode to `gpu` from `requireRealGPU()`, which the GPU test files call
(a test that runs a routed operation without calling it runs under `auto`); Python's
`python/tests/conftest.py` sets it for every pytest run; and `python/tests/differential_report.py`, a
plain script that pytest's conftest does not reach, sets it itself, and prints the mode in its header.
`ARROWMETAL_ROUTER`, when set, wins over each pin, so the same suites run every routed operation through
its CPU loop with

```
ARROWMETAL_ROUTER=cpu swift test -c release
ARROWMETAL_ROUTER=cpu PYTHONPATH=python python -m pytest python/tests -q
ARROWMETAL_ROUTER=cpu PYTHONPATH=python python python/tests/differential_report.py
```

and a green run there means the CPU loops give the answers the GPU tests expect. The router's own
suites (`RouterTests`, `test_router.py`) choose the path per call and run `gpu`, `cpu` and `auto`
regardless of either setting.

The other bindings' suites are not pinned: the Go tests (`go/arrowmetal/*_test.go`), the Rust crate's
tests (`rust/arrowmetal/tests`), the R package's testthat suite and the Node tests (`node/test`) call the
C ABI in their own processes under the process default `auto`, so a routed operation they run on an
input below its crossover runs the CPU loop. Run them with `ARROWMETAL_ROUTER=gpu` to exercise the GPU
kernels, and with `ARROWMETAL_ROUTER=cpu` for the CPU loops. The Polars plugin and the DuckDB extension
are tested from pytest (`test_polars.py`, `test_duckdb.py`); outside it they, too, run under `auto`
unless `ARROWMETAL_ROUTER` says otherwise.

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
| Interop | ArrowMetalTests, KernelTestsV2, SliceOffsetTests, NestedTests, IPCTests, IPCViewTests | C Data / C Device / C Stream import and export with offsets and release callbacks, IPC round trips, struct and stream import; every type the IPC writer emits — decimals, float16, fixed-size binary, the three interval units, null, the three list layouts, struct, map, both unions, run-end encoded and extension columns — written from ArrowMetal and read back by pyarrow in both encapsulations, and each one written by pyarrow and read back by ArrowMetal value for value, with nulls, empty arrays, sliced inputs and nested nulls (a null row, an empty row, a null element, a null struct whose children are not null, a null map); LZ4 and ZSTD compressed bodies; a replacement and a delta dictionary in a stream; a batch declaring a buffer the schema does not use, rejected; the view types as pyarrow writes them (flat, nested in struct and list, sliced, with nulls, out-of-order list views, LZ4 bodies, batches past one 64K-row block) read and written back as classic types; Arrow's big-endian integration files against their little-endian twins and pyarrow; `arrow.fixed_shape_tensor` round trips with pyarrow, its shape product checked for overflow; other extension names read as their storage; tensor messages and offsets past 2 GB refused with their messages |
| Expression compiler and engine | ExprTests, ExprLiteralTypeTests, EngineTests, JoinTests | fused expressions against the unfused kernels, literal promotion, every optimizer rule against the unoptimized plan, six join kinds and as-of against a dictionary oracle |
| Parquet | ParquetTests, ParquetWriterTests, ParquetNestedTests, ParquetArrowSchemaTests, ParquetPageIndexTests, ParquetFilterEdgeTests, ParquetBloomFilterTests | pyarrow-written fixtures across encodings and compressions, damaged files that must error rather than trap; nested structs, maps and lists from pyarrow, DuckDB and Polars against the generator's formulas and against each other; the stored Arrow schema's types and metadata; page skipping with identical matches with and without the index, `!=` over a NaN hidden in a constant float page, uint64 statistics past the int64 range, the `null` type below repeated fields; bloom filters against xxHash64's published vectors |
| Delta Lake and Iceberg | LakehouseTests | the 62 reads in `Tests/Fixtures/lakehouse/expected.json` (what `deltalake` 1.6.5 and pyiceberg 0.12.0 returned for every version, snapshot, projection and filter recorded there) replayed row for row; checkpoint against full log replay, log cleanup, multi-part checkpoints, column mapping, the refused reader features and delete files by name, pruning counters, partition-transform projection, Iceberg bound decoding, the Avro container with the `null`, `deflate` and hand-built `snappy` codecs, malformed Avro containers, Delta partition values and reader-feature lists, Iceberg snapshots without manifests and data files holding none of the table's columns as errors, a NaN Delta partition kept for `!=` only, the row-filter literal rules at the edges of their types, float32 exact comparison, empty-string partitions as null, byte-wise string row-group pruning, Iceberg paths used as written ([LAKEHOUSE.md](LAKEHOUSE.md)) |
| Parquet | ParquetTests, ParquetWriterTests | pyarrow-written fixtures across encodings and compressions, damaged files that must error rather than trap |
| CSV | CSVReaderTests, CSVFloatParseTests | the GPU structure scan against an independent CPU parser over random hazard-filled files at eight scan block sizes and both file-access modes; inference, forced types, errors, the timestamp[ns] range, `skip_rows_after_names` over ragged rows, `scanBlockBytes` past 32 bits; the GPU float parse against Swift's `Double` / `Float` initialisers bit for bit over a randomized and adversarial corpus ([CSV.md](CSV.md)) |
| JSON | JSONReaderTests | the structure scan's record boundaries and top-level errors against a sequential CPU scanner, the max-scan against a CPU loop, the walk's entries, RapidJSON's error texts, timestamps against a day-counting reference, random flat files against Foundation's `JSONSerialization`, grouped slot matrices ([JSON.md](JSON.md)) |
| Streaming | StreamTests | every streaming operator against the in-memory answer at 1–120 batches with ragged and empty batches, HyperLogLog within 3σ, external sort over 120 runs, grace join against the in-memory join |
| CPU/GPU router | RouterTests | both paths of every routed operation byte-identical (values including slots under nulls, validity bits, null count, buffer size) on all ten primitives, sizes across word and threadgroup boundaries, 0/10/90% nulls, slices at offset 7, signed zeros, infinities, quiet and signaling NaNs with payloads, subnormals; float sums bit-identical past one block per GPU thread; group-by sum (and `sumUnsigned`) with null and out-of-range keys; the CPU loops against `CPUReference`; the decision rules (table, multiply's own row, batch, pending input, no CPU path, modes, per-thread override) |
| Adversarial pins | AdversarialKernelTests, AdversarialKernelTests2, AdversarialGroupSliceTests | the probes the review pass ran that came back clean, kept so they stay clean: 100–200 repeat determinism loops, boundary lengths, slice-of-slice, group-by with one group and with one group per row, threadgroup-size invariants |

## 2. Python suites

The column below counts `def test_*` functions. The 5,892 in the table at the top of this page is what
pytest *collected* in the 2026-09-24 gate, which is larger because a parametrised function collects once per parameter set.

| File | Test functions | Compares against |
|---|---|---|
| `test_arrowmetal.py` | 115 | `pyarrow.compute` and plain Python over the ctypes API, zero-copy import/export, the wheel loader |
| `test_options.py`, `test_checked.py`, `test_strings_extra.py`, `test_float64_math.py` | 114 | option surfaces (null placement, tiebreakers, cast safety, rounding), checked arithmetic, string kernels, binary64 math |
| `test_functions.py` | 8 | executes every runnable row of the Arrow-name registry through `call_function` against `pyarrow.compute`, with a second input in another type family where the claim spans several |
| `test_expr.py`, `test_lazy.py`, `test_lazy_optimizer.py`, `test_lazy_memory.py` | 84 | fused expressions and lazy plans against Polars and pyarrow; the optimizer against the unoptimized plan; RSS over 40,000 `explain()` and 10,000 `collect()` calls |
| `test_parquet.py`, `test_parquet_robustness.py` | 31 | `pyarrow.parquet.read_table` on generated files; fuzzed and damaged files |
| `test_lakehouse.py` | 31 | `deltalake`'s `to_pyarrow_table` and pyiceberg's `scan().to_arrow()`: the recorded reads in `expected.json`, the committed tables live (types included), and tables generated at test time read with time travel and with filters of every comparison over every column type; empty-string partitions, decomposed strings in small row groups, partition values pyiceberg escapes on disk, float32 literals, a NaN float partition under every comparison; filter literals at the edges of their types and malformed manifest lists, each read in a subprocess so a crash fails the test; the differences from the reference readers that [LAKEHOUSE.md](LAKEHOUSE.md) lists. The live comparisons skip when `deltalake` or `pyiceberg` is not installed |
| `test_csv.py` | 28 | `pyarrow.csv.read_csv` on generated files: identical tables (floats bit for bit) or identical error messages, one case per inference and parsing rule, every option, 60 seeded random files, block-boundary straddles, the type decided past the inference sample, files written by Polars and DuckDB ([CSV.md](CSV.md)) |
| `test_ipc_views.py` | 12 | view types, big-endian integration files and `arrow.fixed_shape_tensor` through `scan_ipc` against pyarrow, the little-endian twins and the files' JSON; tensor shapes whose product overflows refused; a column naming another extension type through every streaming operator, matching the same column without the keys |
| `test_parquet_nested.py` | 39 | `pyarrow.parquet.read_table` on nested fixtures from pyarrow, DuckDB and Polars, value for value and type for type; the stored Arrow schema's types, field metadata and schema metadata, including crafted `ARROW:schema` values pyarrow ignores or refuses; page skipping and bloom filters against pyarrow's exact matches with the index and filters on and off, including Polars' NaN pages; filter values of unsupported types; damaged nested files and damaged page indexes |
| `test_json.py` | 39 (636 collected) | `pyarrow.json.read_json` on every probe, explicit-schema case, generated, truncated and mutated file (values, nulls, types, or the error text), plus one test per documented difference ([JSON.md](JSON.md)) |
| `test_stream.py` | 27 | streaming results against pyarrow and Polars, including a 2 GB IPC directory generated at test time |
| `test_polars.py` | 53 | the bridge and namespaces against native Polars; the 17 Rust-plugin test functions (28 tests as pytest collects them, the parametrised one expanding to 12) run when `polars-plugin/` is built |
| `test_polars_engine.py` | 74 | `MetalEngine` (tier 4 of [POLARS.md](POLARS.md)) against Polars' own collect of the same LazyFrame, over `test_differential.py`'s generators at five sizes, three null ratios, sliced, special-value, two-chunk and `DataFrame.slice` frames, joins and `unique` included; each fallback with its reason; the Polars surfaces and IR version it relies on; the four ArrowMetal findings it once worked around, fixed and passing; one case reruns `test_polars.py` and `test_lazy.py` with every collect also run through the engine (`metal_engine_everywhere.py`) |
| `test_duckdb.py` | 38 | the bridge against DuckDB SQL; the 11 extension tests run when `duckdb-extension/build.sh` has produced the extension, and one test compiles the public C header as C |
| `test_duckdb_rewrite.py` | 38 (178 collected) | the rewrite extension ([DUCKDB.md](DUCKDB.md) §4b) against DuckDB's own operators: every query with `arrowmetal_rewrite` off and forced, same types and values bit for bit, over generated tables reaching both ends of every integer type, each connection-taking test once in one block and once in 2,048-row streamed blocks; plus the `auto` gate against `router_2026-09-17.json` and the benchmark results (`duckdb_rewrite_2026-09-24.csv`). Skips until `duckdb-extension/build_rewrite.sh` has built the extension |
| `test_pandas.py` | 72 | the accessor and accel mode against plain pandas across five null-carrying dtype flavours; `install()`/`uninstall()` restore every patched slot |
| `test_numpy.py` | 6 | the numpy bridge: which dtypes cross without a copy, NaN as a value, and float64 arithmetic against numpy bit for bit ([NUMPY.md](NUMPY.md)) |
| `test_router.py` | 13 | both router paths byte-identical through the ctypes API and equal to `pyarrow.compute` (and to pyarrow's `hash_sum` for the group-by, signed and uint64), `am.last_route()` and its reasons, `am.router()` / `am.set_router()`, `ARROWMETAL_ROUTER` in a subprocess, `differential_report.py`'s GPU pin, `Benchmarks/router_table.py --check` against the committed table, the table's header naming which CPU loops it measured, `--from-check` fitting a table inside the brackets of a `router_check.py` run, and `multiply` switching at its own row, fitted inside the bracket its check file measured |
| `test_differential.py` (standalone part) | 95 plus 17 documented xfails | one test per finding and per fixed finding, plus the guards that every public operation and every module-level function has a matrix case |

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
sequences, Unicode edge cases, 40,000-call leak loops, fuzzed C entry points). Every finding they
reported is either fixed with a regression test or recorded as a finding in EVALUATION.md; the fixes are
listed under "Fixed" in the CHANGELOG. Nothing from that pass was accepted on the reviewer's word: each
finding's test fails on the commit before its fix.

## 5. What "green" means before a push

Every merge to `main` runs steps 1-5 (`private/tools/gate.sh`), on a quiet machine, in this order, and pushes
only if all pass:

1. `swift build -c release` with no errors.
2. `swift test -c release`: every test, 0 failures (the skips on a physical Mac are the opt-in measurements
   behind `ARROWMETAL_IPC_THROUGHPUT`, `ARROWMETAL_LATENCY_BENCH` and `ARROWMETAL_UNIQUE_BENCH`, and the fourteen
   pyarrow cross-checks in `IPCTests` when no python with pyarrow is found; on a virtual Metal device
   `requireRealGPU()` skips the GPU-only suites as well).
3. `differential_report.py` exits 0: 0 unclassified divergences.
4. `pytest python/tests` (every suite): 0 failures; xfails must be strict and tied to a finding.
5. The standalone tests in `test_differential.py`: 0 failures.

Before a release the binding suites run as well and are recorded in `private/keep/2026-09-08/final_gate_b2dc7fa.log`:

6. `cd rust && cargo test --release`: 48 tests and 4 compile-only doc-tests, 0 failures.
7. `go test` plain and under `GOEXPERIMENT=cgocheck2`.
8. `npm test` (62 tests).
9. testthat (266 expectations).

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
[BENCHMARKS_MATRIX.md](BENCHMARKS_MATRIX.md); rows to improve stay in the table.

## 7. Hardware and toolchain of the published numbers

Apple M4 Max (16 cores, 64 GB unified memory, internal SSD), macOS 26.6.2, Swift 6.3.3 (Xcode
toolchain), Python 3.13, pyarrow 25.0.1, Polars 1.44.1, DuckDB 1.5.5, pandas 3.0.5.
