# arrowmetal (Python)

Apache Arrow arrays on the Apple silicon GPU, from Python. A ctypes wrapper over `libArrowMetalC.dylib`;
input and output go through the Arrow C Data Interface, so it composes with pyarrow, Polars, pandas and DuckDB.

```
swift build -c release --product ArrowMetalC          # builds .build/release/libArrowMetalC.dylib
pip install pyarrow
PYTHONPATH=python python -c "import arrowmetal as am; print(am.device_name())"
```

```python
import pyarrow as pa, polars as pl, arrowmetal as am

s = pl.Series([1, None, 3, 40])
col = am.array(s.to_arrow())            # one copy in (Polars buffers are not page aligned); results are zero-copy out
big = col.filter_where(">", 2)          # GPU
print(big.sum(), pl.from_arrow(big.to_arrow()))

keys = am.array(pa.array([0, 1, 0, 2], pa.int32()))
print(keys.group_by(3).sum(col).to_arrow())
```

Float64 columns (NumPy's and pandas' default) run entirely on the GPU: compare, filter, take, sum and
arithmetic use a software IEEE-754 implementation that is bit-exact with the CPU. Use `with am.batch():`
around a chain of operations to pay one GPU round trip instead of one per call.

See `Benchmarks/python_gpu_bench.py` for a side-by-side with Polars, pyarrow.compute and pandas on the same data.

## Building a wheel

```
pip install build                    # the wheel build frontend
scripts/build_wheel.sh               # -> python/dist/arrowmetal-0.1.0-*-macosx_*_arm64.whl
```

The script builds `libArrowMetalC.dylib` in release, copies it into `python/arrowmetal/` and runs
`python -m build --wheel` in `python/`, so the wheel carries the dylib as package data and needs no
`swift build` at install time. The loader prefers the bundled dylib, then `.build/release`, then
`.build/debug`, then the usual system prefixes; `$ARROWMETAL_LIB` overrides all of them. The wheel is
macOS arm64 only: it links Metal and holds an arm64 binary.

## Tests

```
pip install pytest
swift build -c release --product ArrowMetalC
PYTHONPATH=python python -m pytest python/tests -q
```

`python/tests/test_arrowmetal.py` checks import and round trip for int64/float64/float32/bool/string with
nulls, reductions, compare/filter/`filter_where`, take, slice, cast, arithmetic (float64 included),
boolean logic, group-by, the string kernels, `with am.batch():` against the unbatched results, and Polars
interop. Every result is compared to `pyarrow.compute` or plain Python. Polars tests skip when Polars is
absent; the suite needs a real Metal device.

Not covered by the group-by kernels today: `min`/`max` on 64-bit values, `mean` on Float32, and Float64
values for any aggregate. `python/tests` pins those as expected errors so the tests flag it when they land.

## Polars

Three tiers, all shipping in this repository — see [docs/POLARS.md](../docs/POLARS.md):

```python
import polars as pl, arrowmetal as am

am.from_polars(df)                              # -> dict[str, MetalArray], zero copy
df.arrowmetal.group_by("k").sum("v")            # tier 1: the GPU around Polars
pl.col("v").arrowmetal.sum()                    # tier 2: inside a lazy plan (needs the Rust plugin)
lf.arrowmetal.collect_gpu(query)                # tier 3: Polars runs the plan, the GPU finishes it
```

`import arrowmetal` still does not import Polars: the bridge loads on first use (or at import when
Polars is already loaded). Tier 2 needs one extra build,
`cd polars-plugin && cargo build --release`; the other two are pure Python over the Arrow C Data
Interface. `PYTHONPATH=python python -m pytest python/tests/test_polars.py -q` runs the suite.
