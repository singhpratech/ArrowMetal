# numpy

How a numpy array reaches the GPU, what crosses without a copy and what does not, what the benchmark
matrix measures against numpy, and the one step that would let existing numpy code run on Metal. Every
number here is from `Benchmarks/results/full_matrix_2026-09-07.csv` or from `python/tests/test_numpy.py`,
which asserts each claim in the first two sections on every run. That CSV is the **eager** run, and the
numpy rows quoted below are numpy's plain eager idiom; the published baseline is the parallel run in
`Benchmarks/results/full_matrix_2026-09-07-parallel.csv`, which measures each CPU library's most
parallel idiom too (numpy has none, so its `numpy-parallel` column is empty), and
[BENCHMARKS_MATRIX.md](BENCHMARKS_MATRIX.md) and [TO_IMPROVE.md](TO_IMPROVE.md) are written against it.

## 1. The crossing

There is no numpy bridge, because none is needed: pyarrow already speaks numpy, and ArrowMetal speaks
Arrow.

```python
import numpy as np, pyarrow as pa, arrowmetal as am

x = np.random.rand(50_000_000)          # float64, the numpy default
col = am.array(pa.array(x))             # no copy: pyarrow adopts the numpy buffer, Metal wraps it
r = col.sqrt()                          # GPU, software IEEE-754 binary64
y = r.to_arrow().to_numpy(zero_copy_only=True)   # no copy out either
```

Two things make this copy-free, and both are checked rather than assumed:

- `pa.array(x)` adopts the numpy buffer for integer and float dtypes: the Arrow array's data buffer has
  the numpy array's address.
- `am.array(...)` wraps a page-aligned buffer in a Metal buffer in place. macOS hands numpy page-aligned
  memory for every allocation we measured from 1,000 elements up to 50M; a buffer that is not page aligned
  (tiny arrays of ten or a hundred elements were not) is copied once on the way in, which costs nothing
  worth measuring at that size.

`test_numpy.py` proves the import is in place by writing to the numpy array *after* the import and reading
the new value back through the Metal column, and proves the export is in place by comparing the address
of `to_numpy(zero_copy_only=True)` with the Arrow buffer's.

## 2. What crosses and what it costs

| numpy dtype | copies in | why |
|---|---|---|
| `int8`…`int64`, `uint8`…`uint64`, `float32`, `float64` | **none** | pyarrow adopts the buffer; Metal wraps it |
| `bool` | one | Arrow packs booleans to one bit per value, so pyarrow has to write a new buffer |
| `datetime64` | none | adopted as an Arrow timestamp |
| `str_` (`U`), `object` strings | one conversion | become an Arrow `string` array; numpy's fixed-width UTF-32 layout is not Arrow's |
| a slice with a stride | one | Arrow has no strides; pyarrow copies to contiguous |

Results come back the same way: any numeric result with no nulls is a numpy view over the Metal buffer
through `to_numpy(zero_copy_only=True)`; a result with nulls needs `to_numpy(zero_copy_only=False)`,
which materialises NaNs, or stays as Arrow.

**NaN is a value, not a null.** `pa.array(np.array([1.0, np.nan]))` has a null count of zero, and ArrowMetal
treats it the way numpy does: `sum` and `mean` return NaN. That is numpy's semantics and Arrow's default; a
pandas column takes the other path (`from_pandas=True`), where NaN becomes null and is skipped, and
[PANDAS.md](PANDAS.md) covers it. If you want numpy's NaNs skipped on the GPU, pass `from_pandas=True` to
`pa.array` and pay one pass to build the validity bitmap.

numpy has no nulls, so the GPU column has no validity bitmap and every kernel takes its no-null fast path.

## 3. Measured against numpy

The benchmark matrix (`Benchmarks/full_matrix.py`) compares each operation with pandas, and where pandas has
no vectorised idiom of its own it measures numpy directly on the same values, without a validity bitmap
because numpy has none. Those rows, at their largest measured size, from the 2026-09-07 run on an Apple
M4 Max (numpy 2.5.3, the eager idiom, best of up to five calls after a warm-up; cores = CPU-ms / wall ms):

| family | operation | rows | numpy ms (cores) | ArrowMetal ms (cores) | speedup |
|---|---|---|---|---|---|
| sort | lexsort (2 int32 keys) | 50,000,000 | 28,585 (1.0) | 37.4 (0.1) | 763.9x |
| sort | argsort float64 | 50,000,000 | 5,489 (1.0) | 40.0 (0.0) | 137.4x |
| sort | top_k (k=100, int64) | 50,000,000 | 328.5 (1.0) | 2.685 (0.5) | 122.4x |
| sort | argsort int64 | 50,000,000 | 4,555 (1.0) | 39.2 (0.0) | 116.4x |
| sort | top_k (k=10000, int64) | 50,000,000 | 328.3 (1.0) | 2.944 (0.6) | 111.5x |
| chains | compare + filter + take | 50,000,000 | 206.7 (1.0) | 4.569 (0.3) | 45.2x |
| sort | sort float64 | 50,000,000 | 1,762 (1.0) | 49.6 (0.0) | 35.5x |
| element-wise | exp (float32) | 50,000,000 | 61.9 (1.0) | 1.912 (0.2) | 32.4x |
| chains | filter two columns + sum | 50,000,000 | 113.6 (1.0) | 3.545 (0.5) | 32.0x |
| element-wise | case_when (2 conditions) | 50,000,000 | 199.5 (1.0) | 6.896 (0.2) | 28.9x |
| element-wise | index_in (int64, 100-value set) | 50,000,000 | 163.0 (1.0) | 5.848 (0.6) | 27.9x |
| sort | partition_nth_indices (n/2) | 50,000,000 | 341.6 (1.0) | 19.8 (0.4) | 17.2x |
| compare+select | indices_nonzero (bool) | 50,000,000 | 16.8 (1.0) | 3.151 (0.3) | 5.3x |
| element-wise | if_else (bool ? int64 : int64) | 50,000,000 | 14.7 (1.0) | 3.699 (0.3) | 4.0x |
| element-wise | is_nan (float64) | 50,000,000 | 4.926 (1.0) | 1.320 (0.3) | 3.7x |
| element-wise | shift_left (int64 << 2) | 50,000,000 | 6.817 (1.0) | 1.915 (0.2) | 3.6x |
| element-wise | bit_wise_and (int64) | 50,000,000 | 10.1 (1.0) | 2.935 (0.2) | 3.4x |
| element-wise | sqrt (float64) | 50,000,000 | 12.3 (1.0) | 3.951 (0.1) | 3.1x |
| element-wise | sin (float64) | 50,000,000 | 301.7 (1.0) | 150.2 (0.0) | 2.0x |
| element-wise | ln (float64) | 50,000,000 | 87.3 (1.0) | 81.2 (0.0) | 1.1x |
Two readings. numpy runs every one of these on one core, which is the honest shape of the comparison:
this is one CPU core against the GPU, and the crossing is free because the memory is shared; a 16-thread
CPU library is the other comparison, and it is on the same Compare tab. And the transcendentals are the
narrow rows: `ln` at 1.1x and `sin` at 2.0x are software binary64 on a GPU with no double hardware against
a vectorised libm, and they are listed in [TO_IMPROVE.md](TO_IMPROVE.md) as open work.

## 4. The step that would matter, and it is ours to take

numpy already dispatches. Since numpy 1.17, an array type that implements `__array_function__` (NEP 18)
answers `np.sum(x)`, `np.sort(x)`, `np.where(...)` and the rest on its own memory, and numpy calls it
instead of its own kernel. An ArrowMetal array type that implements it would let existing numpy code run
on the GPU with no import changed; the hook on numpy's side has been there for years. It is on the
roadmap ([ROADMAP.md](ROADMAP.md)); nothing here claims it exists.

## 5. Limits

- Multi-dimensional arrays are not Arrow arrays; pass one column (`x[:, i]` is strided, so it copies once;
  `np.ascontiguousarray` first if you will reuse it).
- `bool` costs one pass in each direction because of the bit packing.
- Strings are converted, in and out.
- A numpy array smaller than about a thousand elements is usually not page aligned and is copied on
  import; at that size the dispatch floor, not the copy, is the cost, and the GPU is not the right place
  for it anyway (see the latency family in the matrix).
