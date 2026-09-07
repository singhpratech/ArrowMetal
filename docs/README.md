# ArrowMetal documentation

Start here. Everything in this folder is written so that someone who has never seen the code can check
the claims against the source, the tests and the recorded benchmark runs: each page names the file, the
test or the CSV a claim comes from.

| Read this | To learn |
|---|---|
| [../README.md](../README.md) | What ArrowMetal is, the pitch, headline numbers, quick start in Swift and Python |
| [ARROW_FUNCTIONS.md](ARROW_FUNCTIONS.md) | Coverage of Apache Arrow compute by **exact name**: all 307 v25 function names, one row each, generated from a registry the test suite executes against `pyarrow.compute` |
| [COVERAGE.md](COVERAGE.md) | The same ground by **family**, with the Arrow type matrix and interop: which functions and types are supported, and how (GPU, GPU/CPU, CPU, partial) |
| [EVALUATION.md](EVALUATION.md) | How correctness is checked: CPU oracles, the scenario matrix, bit-exact IEEE-754 tests, and the differential harness against pyarrow.compute, with open findings |
| [UPSTREAM.md](UPSTREAM.md) | What the matrix found in other projects (pyarrow, the Arrow tz database, Apple Metal), the test behind each finding, and where every upstream report stands |
| [TESTING.md](TESTING.md) | Every test suite, what it covers, and the commands to run it |
| [BENCHMARKS.md](BENCHMARKS.md) | Every benchmark round with hardware, methodology, wall time and CPU time; the tables the README quotes |
| [LOSSES.md](LOSSES.md) | Every row of the matrix still short of the 3x bar, slower rows first, grouped by the measured cause (dispatch floor, copy vs view, sort-based distinct, software binary64, host regex), plus the rows under the 3x bar and what changed since the previous matrix |
| [BENCHMARKS_MATRIX.md](BENCHMARKS_MATRIX.md) | The complete operation-by-operation comparison against Polars, pyarrow.compute and pandas at 10M and 50M rows, with a verdict per row and a shortfall list of everything below the 3x bar |
| [DESIGN.md](DESIGN.md) | How it works: buffers, kernels, batching, GPU-side lengths, software Float64, where the time goes |
| [PANDAS.md](PANDAS.md) | pandas on the GPU: the `.am` accessor and the zero-code-change accel mode, what routes to the GPU and when, the zero-copy and null rules, the numbers and the limits |
| [NUMPY.md](NUMPY.md) | numpy on the GPU through pyarrow: which dtypes cross without a copy (asserted by `test_numpy.py`), NaN-as-value semantics, the matrix rows measured against numpy, and the NEP 18 step that is ours to take |
| [RUST.md](RUST.md) | The Rust crate over the C ABI: install, the copy rule as measured, what is wrapped and what is not, the timing table with its method |
| [GO.md](GO.md) | The Go module: cgo shim, pinning, the alignment finding, what is wrapped, the timing table |
| [TYPESCRIPT.md](TYPESCRIPT.md) | The Node addon and TypeScript API: Arrow JS interop, lifetimes, the copy rule, timings |
| [R.md](R.md) | The R package: arrow R interop, int64 through bit64, the timing table with both measurement modes |
| [EXPR.md](EXPR.md) | Fused expression queries: one runtime-generated kernel for a whole expression DAG, the grammar, how nulls are compiled, the numbers and the limits |
| [ENGINE.md](ENGINE.md) | The lazy query engine: the logical plan, the optimizer rules with `explain()` examples, fusion planning, the full join matrix (multi-key, utf8, outer, semi/anti, as-of), window functions, the plan grammar, the numbers and the limits |
| [POLARS.md](POLARS.md) | Polars on the GPU in three tiers: the zero-copy bridge and `.arrowmetal` namespaces, the Rust expression plugin for lazy plans, and the streaming hand-off — with install, numbers at 10M and 50M rows, and what a real Metal `engine=` backend would need |
| [DUCKDB.md](DUCKDB.md) | Using ArrowMetal from DuckDB: the Python bridge (copy-free where DuckDB returns one chunk), the loadable SQL extension, streaming tables larger than memory, and an honest account of which shapes the GPU wins and which it loses |
| [STREAMING.md](STREAMING.md) | Out-of-core streaming execution: datasets larger than memory flowing from disk through the GPU, the three-stage pipeline and its measured overlap, every streaming operator with exact-or-approximate marked, and the buffer budget |
| [DECISIONS.md](DECISIONS.md) | Why it is built this way, one dated entry per decision |
| [FINDINGS.md](FINDINGS.md) | Things learned the hard way: toolchain quirks, Metal limits, bugs and their lessons |
| [ROADMAP.md](ROADMAP.md) | What is next and what is open for contributors |
| [../CHANGELOG.md](../CHANGELOG.md) | What is in the (unreleased) 0.1.0 |
| [../Benchmarks/README.md](../Benchmarks/README.md) | The three benchmark programs and the fairness rules |
| [../python/README.md](../python/README.md) | The Python package: install, wheel build, usage |
| [RELEASE.md](RELEASE.md) | The ordered checklist for cutting and publishing 0.1.0: tags, the wheel, PyPI, the plugin crate, the docs to re-verify |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | How to add a kernel, test it, and benchmark it |

## Installing

Nothing here is published yet — no PyPI package, no crates.io crate, no tagged Swift release. Both install
routes go through this checkout, and [RELEASE.md](RELEASE.md) is what turns them into published artefacts.

```
# Swift: add the package by path or git URL in Package.swift, then
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift build -c release

# Python from source
swift build -c release --product ArrowMetalC
PYTHONPATH=python python -c "import arrowmetal as am; print(am.device_name())"

# Python from a wheel that bundles the dylib (macOS 14+, arm64)
pip install build && scripts/build_wheel.sh          # or python/build_wheel.sh, if the dylib is built
pip install python/dist/arrowmetal-0.1.0-*.whl
pip install 'python/dist/arrowmetal-0.1.0-*.whl[polars,duckdb,pandas]'   # optional bridges
```

The package finds `libArrowMetalC.dylib` in one of three places, in order: `$ARROWMETAL_LIB`, which pins
one specific build and wins over everything (the A/B benchmark scripts and the merge gate rely on that),
then the copy bundled inside the wheel (`arrowmetal/_lib/`), then a development build in `.build/release`
beside a source checkout. See [../python/README.md](../python/README.md).

## Reproducing everything

All of it runs on an Apple silicon Mac on macOS 14 or later (the package's floor) with Xcode installed:

```
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test               # Swift suite, debug
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test -c release    # same, optimised
swift build -c release --product ArrowMetalC && PYTHONPATH=python python -m pytest python/tests -q
swift run -c release arrowmetal-bench            # Metal vs all CPU cores vs Accelerate, with CPU-ms
python Benchmarks/python_bench.py                # Polars / pyarrow / pandas on the same data shapes
PYTHONPATH=python python Benchmarks/python_gpu_bench.py   # ArrowMetal from Python vs Polars, in-process
PYTHONPATH=python python Benchmarks/expr_bench.py         # fused expression queries vs one kernel per operator
PYTHONPATH=python python Benchmarks/engine_bench.py       # the lazy query engine vs Polars lazy and DuckDB
cd polars-plugin && cargo build --release                 # the Polars expression plugin (tier 2)
PYTHONPATH=python python -m pytest python/tests/test_polars.py -q   # the three Polars tiers
PYTHONPATH=python python Benchmarks/polars_bench.py       # native Polars vs both Polars tiers
PYTHONPATH=python python Benchmarks/pandas_bench.py       # plain pandas vs the .am accessor vs accel mode
PYTHONPATH=python python Benchmarks/streaming_bench.py --data-dir /tmp/am-stream --size-gb 30
                                                 # out-of-core streaming vs Polars, DuckDB and pyarrow.dataset
swift run -c release arrowmetal-examples         # six end-to-end scenarios
```

Every benchmark table in the repository names the machine it was measured on. Numbers from GitHub-hosted
runners are never published: their GPU is virtual.


## Parquet on the GPU

| Read this | To learn |
|---|---|
| [PARQUET.md](PARQUET.md) | The Parquet reader that decodes on the Apple GPU: the pipeline, the parallel RLE strategy, GPU Snappy and LZ4, the supported encoding/codec/type matrix, projection and statistics pushdown, benchmarks against pyarrow / Polars / pandas, the small writer, and the limits |

```
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --filter "ParquetTests|ParquetWriterTests"
PYTHONPATH=python python -m pytest python/tests/test_parquet.py -q     # every fixture vs pyarrow.parquet
PYTHONPATH=python python Benchmarks/parquet_bench.py --rows 50000000   # vs pyarrow, Polars, pandas
python Tests/Fixtures/generate_parquet.py                              # regenerate the committed fixtures
```
