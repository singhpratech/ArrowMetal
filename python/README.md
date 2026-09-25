# arrowmetal (Python)

**ArrowMetal runs Apache Arrow compute on the Apple silicon GPU: Arrow arrays that live in Metal shared
memory and GPU kernels that keep Arrow's semantics, reached from Python through a ctypes wrapper over
`libArrowMetalC.dylib` whose input and output go through the Arrow C Data Interface, so it composes with
pyarrow, Polars, pandas and DuckDB.**

Apple silicon has one physical memory shared by the CPU and the GPU, so an Arrow buffer placed in a
Metal shared buffer is at once a valid CPU Arrow buffer and a valid GPU buffer: a column is used where
it already is, with no copy across a bus in either direction. While a kernel runs, the CPU is free for
the rest of the application, and every table in the repository shows the CPU time of each call next to
its wall time.

One row, the same in-process data in every column:

| `sum by int32 key (1000 groups)`, 50,000,000 rows | wall ms | CPU ms of that call |
|---|---:|---:|
| ArrowMetal | **4.89** | 1.2 |
| pyarrow Acero (`Table.group_by`, 16 threads) | 18.45 | 250 |
| Polars lazy (16 threads) | 81.93 | 1,186 |
| pandas | 247.93 | 248 |

From [`Benchmarks/results/full_matrix_2026-09-07-parallel.csv`](https://github.com/singhpratech/ArrowMetal/blob/main/Benchmarks/results/full_matrix_2026-09-07-parallel.csv),
Apple M4 Max (16 CPU cores, 64 GB), best of up to five calls after one warm-up, release build; the
339-row matrix that row comes from, including the 77 rows where the CPU idiom is ahead, is in
[docs/BENCHMARKS_MATRIX.md](https://github.com/singhpratech/ArrowMetal/blob/main/docs/BENCHMARKS_MATRIX.md).

```
pip install arrowmetal          # macOS 14 or later on Apple silicon; pyarrow is the only dependency
python -m arrowmetal.bench      # 30 seconds or less
python -m arrowmetal.bench --parquet data.parquet   # read, sum, filter, group-by on your own file
```

The second line generates 10,000,000 rows, runs sum, filter, sort and group-by sum through pyarrow
(and Polars when it is installed) and through ArrowMetal, and checks every ArrowMetal answer against
pyarrow's. It prints one table for your Mac, wall and CPU milliseconds per call, with a block ready to
paste into a [benchmark result](https://github.com/singhpratech/ArrowMetal/issues/new?template=benchmark_result.yml)
issue or the `#benchmarks` channel of the Discord linked from the
[README](https://github.com/singhpratech/ArrowMetal#readme); nothing is sent. The generated dataset
is drawn with NumPy, which pyarrow does not install: without it the command says so and exits.

`python -m arrowmetal.bench --parquet data.parquet` runs on your own file instead. It reads the file's
integer, floating-point and string columns with pyarrow, Polars and ArrowMetal, then runs sum and
`filter > median` on the largest numeric column and group-by sum keyed on the lowest-cardinality
integer or string column, CPU against Metal, with the same protocol and every ArrowMetal answer checked
against pyarrow's. The report, the Share it block and the prefilled issue link carry the file's row
count, column count, row groups, size and codecs and the timings, never its path, column names or
values. A file whose columns would take more than a quarter of physical memory is refused, with the
limit printed (the bench holds the columns up to four times at once), and a file with no integer,
floating-point or string column gets one line saying so. `--parquet` needs no NumPy.

```python
import pyarrow as pa, arrowmetal as am
amount = am.MetalArray.from_arrow(pa.array([10.0, 20.0, 30.0, 40.0]))
region = am.MetalArray.from_arrow(pa.array([1, 2, 1, 2], pa.int32()))
print(am.group_by([region]).sum(amount).to_arrow())     # [40, 60]: GPU group-by sum, one row per region
print(amount.filter_where(">", 15).to_arrow())          # [20, 30, 40]: GPU filter, back as a pyarrow array
```

Repository, documentation and the other language bindings: <https://github.com/singhpratech/ArrowMetal>. Site: <https://arrowmetal.org>.

## Install

Two ways: a wheel built with `scripts/build_wheel.sh` and installed with
`pip install python/dist/arrowmetal-0.2.0-*.whl` (macOS 14 or later on Apple silicon; the wheel bundles
`libArrowMetalC.dylib` and the Polars expression plugin `libarrowmetal_polars.dylib`, and pyarrow is its
only dependency), or from this repository.

**From source**, the development path. Needs the Swift toolchain:

```
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift build -c release --product ArrowMetalC
# that builds .build/release/libArrowMetalC.dylib
pip install pyarrow
PYTHONPATH=python python -c "import arrowmetal as am; print(am.device_name())"
```

**From a wheel**, which carries both dylibs and needs no Swift or Rust toolchain at install time:

```
# the wheel build frontend, once
pip install build
# swift build, cargo build of the Polars plugin, then the wheel
scripts/build_wheel.sh
# pyarrow comes with it
pip install python/dist/arrowmetal-0.2.0-*.whl
# optional bridges
pip install "$(echo python/dist/arrowmetal-0.2.0-*.whl)[polars,duckdb,pandas]"
python -c "import arrowmetal as am; print(am.device_name())"
```

The wheel is `arrowmetal-0.2.0-py3-none-macosx_14_0_arm64.whl`; the build script prints its size. It is macOS arm64 only:
it links Metal and holds an arm64 binary. Extras `polars`, `duckdb`, `pandas` and `test` pull in the
libraries the corresponding bridges and the test suite want; none of them are needed to `import arrowmetal`.

The first thing to run needs nothing beyond the wheel itself (pyarrow comes with it):

```python
import pyarrow as pa, arrowmetal as am

# this small array is copied in; large pyarrow buffers are page aligned
# and are borrowed; out is zero-copy
col = am.array(pa.array([1, None, 3, 40]))
big = col.filter_where(">", 2)                 # GPU
print(big.sum(), big.to_arrow())               # 43  [3, 40]

keys = am.array(pa.array([0, 1, 0, 2], pa.int32()))
print(keys.group_by(3).sum(col).to_arrow())    # [4, null, 40]
```

With the `polars` extra (or any installed Polars), a Series crosses through Arrow the same way:

```python
import polars as pl
col = am.array(pl.Series([1, None, 3, 40]).to_arrow())
print(pl.from_arrow(col.filter_where(">", 2).to_arrow()))
```

Float64 columns (NumPy's and pandas' default) run entirely on the GPU: compare, filter, take, sum and
arithmetic use a software IEEE-754 implementation that is bit-exact with the CPU. Use `with am.batch():`
around a chain of operations to pay one GPU round trip instead of one per call.

See `Benchmarks/python_gpu_bench.py` for a side-by-side with Polars, pyarrow.compute and pandas on the same data.

## Building a wheel

```
pip install build                    # the wheel build frontend
scripts/build_wheel.sh               # swift build, then the wheel (cargo builds the Polars plugin)
# just the wheel, when .build/release/libArrowMetalC.dylib exists
python/build_wheel.sh
# install the wheel and Polars into a fresh virtualenv and run all four Polars tiers
scripts/check_wheel.sh
```

`python/build_wheel.sh` copies `.build/release/libArrowMetalC.dylib` into `python/arrowmetal/_lib/`,
builds the Polars expression plugin with `cargo build --release` in `polars-plugin/` against that same
dylib and copies `libarrowmetal_polars.dylib` beside it, replaces the plugin's build-machine rpath with
`@loader_path` (so it loads the packaged `libArrowMetalC.dylib`), re-signs both, runs
`python -m build --wheel` in `python/` with the platform tag `macosx_14_0_arm64`, checks that both
dylibs really are inside the archive, and prints the wheel's path and size.
`scripts/build_wheel.sh` is the same thing with `swift build -c release --product ArrowMetalC` in front.
The version comes from one place, `__version__` in `arrowmetal/__init__.py`; `pyproject.toml` reads it.

`ARROWMETAL_DYLIB` points the script at a dylib somewhere other than `.build/release`,
`ARROWMETAL_POLARS_PLUGIN_DYLIB` at an already built plugin (then cargo is not needed), and `PLAT_TAG`
overrides the platform tag.

`scripts/check_wheel.sh [wheel]` installs the wheel and Polars from PyPI into a fresh virtualenv in a
temporary directory, and from there, with a scrubbed environment (no `PYTHONPATH`, `ARROWMETAL_*` or
`DYLD_*` variables, no cargo on `PATH`), runs a tier-1 group-by, tier-2 `sum`, `filter_sum` and
`device` expressions, a tier-3 `collect_gpu` and a tier-4 `MetalEngine` collect against Polars' own
answers, checks that the process loaded exactly one `libArrowMetalC.dylib`, the packaged one, and runs
`python -m arrowmetal.bench --parquet` on a 1,000,000-row file.

### Where the dylib is found

`arrowmetal` looks for `libArrowMetalC.dylib` in this order:

1. **`$ARROWMETAL_LIB`**, the full path to a dylib. It wins over everything else, so one specific build
   can be pinned; a bundled `arrowmetal/_lib/` copy in the same checkout never shadows it. Setting it to
   a path that does not exist is an error, not a silent fall-back to some other dylib.
2. **Bundled in the installed package**, `arrowmetal/_lib/libArrowMetalC.dylib` — what a wheel install has.
3. **The development build**, `.build/release/libArrowMetalC.dylib` beside a source checkout
   (then `.build/debug`, then `/usr/local/lib` and `/opt/homebrew/lib`).

When none of the three exists the import fails with an `OSError` that names all three and how to satisfy each.

## Tests

```
pip install pytest
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift build -c release --product ArrowMetalC
PYTHONPATH=python python -m pytest python/tests -q
```

`python/tests/test_arrowmetal.py` checks import and round trip for int64/float64/float32/bool/string with
nulls, reductions, compare/filter/`filter_where`, take, slice, cast, arithmetic (float64 included),
boolean logic, group-by, the string kernels, `with am.batch():` against the unbatched results, and Polars
interop. Every result is compared to `pyarrow.compute` or plain Python. Polars tests skip when Polars is
absent; the suite needs a real Metal device.

Not covered by the dense-key `MetalArray.group_by(key_count)` path: `min`/`max` on 64-bit values, `mean` on
Float32, and Float64 values for any aggregate; `python/tests` pins those as expected errors. `am.group_by(keys)` runs every aggregate on Float64 values
(python/tests/test_arrowmetal.py::test_group_by_arbitrary_keys_matches_pyarrow).

## Polars

Four tiers, all in this repository — see [docs/POLARS.md](https://github.com/singhpratech/ArrowMetal/blob/main/docs/POLARS.md):

```python
import polars as pl, arrowmetal as am

# -> dict[str, MetalArray], zero copy
am.from_polars(df)
# tier 1: the GPU around Polars
df.arrowmetal.group_by("k").sum("v")
# tier 2: inside a lazy plan (the wheel carries the plugin)
pl.col("v").arrowmetal.sum()
# tier 3: Polars runs the plan, the GPU finishes it
lf.arrowmetal.collect_gpu(query)
```

`import arrowmetal` still does not import Polars: the bridge loads on first use (or at import when
Polars is already loaded). Tier 2 is a Rust plugin that a wheel built from this tree carries in
`arrowmetal/_lib/`; from a source checkout it is one extra build,
`cd polars-plugin && cargo build --release`. The plugin is built against the Polars 1.44 plugin ABI.
The other tiers are pure Python over the Arrow C Data Interface. `PYTHONPATH=python python -m pytest python/tests/test_polars.py -q` runs the suite.
