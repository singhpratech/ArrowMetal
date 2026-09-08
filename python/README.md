# arrowmetal (Python)

Apache Arrow arrays on the Apple silicon GPU, from Python. A ctypes wrapper over `libArrowMetalC.dylib`;
input and output go through the Arrow C Data Interface, so it composes with pyarrow, Polars, pandas and DuckDB.

## Install

Two ways: a wheel built with `scripts/build_wheel.sh` and installed with
`pip install python/dist/arrowmetal-0.1.0-*.whl` (macOS 14 or later on Apple silicon; the wheel bundles
`libArrowMetalC.dylib`, and pyarrow is its only dependency), or from this repository.

**From source**, the development path. Needs the Swift toolchain:

```
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift build -c release --product ArrowMetalC
# that builds .build/release/libArrowMetalC.dylib
pip install pyarrow
PYTHONPATH=python python -c "import arrowmetal as am; print(am.device_name())"
```

**From a wheel**, which carries the dylib and needs no Swift toolchain at install time:

```
# the wheel build frontend, once
pip install build
# swift build, then the wheel
scripts/build_wheel.sh
# pyarrow comes with it
pip install python/dist/arrowmetal-0.1.0-*.whl
# optional bridges
pip install "$(echo python/dist/arrowmetal-0.1.0-*.whl)[polars,duckdb,pandas]"
python -c "import arrowmetal as am; print(am.device_name())"
```

The wheel is `arrowmetal-0.1.0-py3-none-macosx_14_0_arm64.whl`; the build script prints its size. It is macOS arm64 only:
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
scripts/build_wheel.sh               # swift build, then the wheel
# just the wheel, when .build/release/libArrowMetalC.dylib exists
python/build_wheel.sh
```

`python/build_wheel.sh` copies `.build/release/libArrowMetalC.dylib` into `python/arrowmetal/_lib/`,
runs `python -m build --wheel` in `python/` with the platform tag `macosx_14_0_arm64`, checks that the
dylib really is inside the archive, and prints the wheel's path and size.
`scripts/build_wheel.sh` is the same thing with `swift build -c release --product ArrowMetalC` in front.
The version comes from one place, `__version__` in `arrowmetal/__init__.py`; `pyproject.toml` reads it.

`ARROWMETAL_DYLIB` points the script at a dylib somewhere other than `.build/release`, and `PLAT_TAG`
overrides the platform tag.

### Where the dylib is found

`arrowmetal` looks for `libArrowMetalC.dylib` in this order:

1. **`$ARROWMETAL_LIB`**, the full path to a dylib. It wins over everything else, because pinning one
   specific build is exactly what the A/B benchmark scripts and the merge gate use it for — a stale
   `arrowmetal/_lib/` left behind by a wheel build in the same checkout must not shadow it. Setting it to
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
Float32, and Float64 values for any aggregate; `python/tests` pins those as expected errors so the tests flag it
when they land. `am.group_by(keys)` runs every aggregate on Float64 values
(python/tests/test_arrowmetal.py::test_group_by_arbitrary_keys_matches_pyarrow).

## Polars

Three tiers, all in this repository — see [docs/POLARS.md](https://github.com/singhpratech/ArrowMetal/blob/main/docs/POLARS.md):

```python
import polars as pl, arrowmetal as am

# -> dict[str, MetalArray], zero copy
am.from_polars(df)
# tier 1: the GPU around Polars
df.arrowmetal.group_by("k").sum("v")
# tier 2: inside a lazy plan (needs the Rust plugin)
pl.col("v").arrowmetal.sum()
# tier 3: Polars runs the plan, the GPU finishes it
lf.arrowmetal.collect_gpu(query)
```

`import arrowmetal` still does not import Polars: the bridge loads on first use (or at import when
Polars is already loaded). Tier 2 needs one extra build,
`cd polars-plugin && cargo build --release`; the other two are pure Python over the Arrow C Data
Interface. `PYTHONPATH=python python -m pytest python/tests/test_polars.py -q` runs the suite.
