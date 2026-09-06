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

See `Benchmarks/python_gpu_bench.py` for a side-by-side with Polars, pyarrow.compute and pandas on the same data.
