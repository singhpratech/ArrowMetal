# pandas on the Apple silicon GPU

ArrowMetal gives pandas two ways onto the GPU, and you pick how much of your code changes.

| Tier | You write | What it does | Falls back? |
|---|---|---|---|
| **1. Accessor** | `s.am.sum()`, `df.am.groupby("k").sum("v")` | Explicit. Always dispatched to the GPU, with a documented per-row host fallback for the Unicode case transforms on non-Latin rows. Results come back Arrow-backed, so they go straight out again with no copy. | No — an unsupported dtype raises, so you always know where the work ran. |
| **2. Accel mode** | `import arrowmetal.pandas_accel; arrowmetal.pandas_accel.install()` then **unchanged pandas** | Patches a documented set of pandas methods. Routes to the GPU only when the dtype, the size and the arguments all qualify. | Yes — always, silently, with the reason in `stats()`. |

Both live in the same package and can be used together.

```bash
# macOS arm64. pandas is not a hard dependency.
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
    swift build -c release --product ArrowMetalC
export PYTHONPATH=python
```

`import arrowmetal` never imports pandas: the bridge is loaded the first time you touch
`am.from_pandas`, `am.to_pandas`, `am.pandas_bridge` or `am.pandas_accel`.

---

## Tier 1: the `.am` accessor

```python
import pandas as pd, arrowmetal as am
import arrowmetal.pandas_bridge      # registers .am on Series and DataFrame

df = pd.read_parquet("trades.parquet", dtype_backend="pyarrow")   # Arrow-backed: zero-copy

df["price"].am.sum()                    # scalar
df["price"].am.nlargest(100)            # Series, index taken along
df["symbol"].am.contains("BRK")         # boolean Series
df.am.groupby("symbol").sum("size")     # DataFrame indexed by the key
df.am.sort_values(["symbol", "price"])
df.am.merge(reference, on="symbol")
df.am.query(am.filter(am.col("price") > 100).sum(am.col("size")))   # one fused kernel
```

### Series accessor

| Method | Notes |
|---|---|
| `sum` `min` `max` `mean` `product` `median` `any` `all` | `sum` of an empty or all-null column is `0`, as in pandas — each bridge follows its own host library, so the Polars bridge answers `None` there ([POLARS.md](POLARS.md)) |
| `std(ddof=1)` `var(ddof=1)` `count()` `nunique(dropna=True)` | |
| `value_counts(sort, ascending, dropna)` | counts descending, ties in first-seen order — pandas' own order |
| `sort_values(ascending)` | stable, nulls last; the index is taken along |
| `argsort(ascending)` | int32 positions |
| `top_k(k, largest)` / `nlargest(n)` / `nsmallest(n)` | pandas' `keep="first"` tie behaviour |
| `isin(values)` `abs()` `round(ndigits=0)` | `round` is half-to-even, as pandas is |
| `compare(op, other)` and `>` `>=` `<` `<=` `==` `!=` | scalar or another column |
| `filter(mask)` | boolean-mask selection, index taken along |
| `contains` `startswith` `endswith` `upper` `lower` `len` | utf8 |
| `to_metal()` `to_arrow()` `zero_copy()` | the escape hatches |

### DataFrame accessor

| Method | Notes |
|---|---|
| `groupby(by, dropna=True, sort=True, as_index=True).sum/mean/min/max/count(cols)` and `.size()` | one or several key columns of any type |
| `sort_values(by, ascending)` | one or several columns, per-column direction, stable, nulls last |
| `filter(mask)` | |
| `merge(right, on=..., how="inner")` | needs a unique, null-free key on the right frame |
| `query(expr)` | a fused ArrowMetal expression query over the whole frame — see [EXPR.md](EXPR.md) |
| `to_metal()` `zero_copy()` | |

---

## Tier 2: zero-code-change accel mode

```python
import arrowmetal.pandas_accel
arrowmetal.pandas_accel.install()        # or set ARROWMETAL_PANDAS_ACCEL=1

# ... your existing pandas script, unchanged ...

print(arrowmetal.pandas_accel.stats())
```

or without touching the script at all:

```
python -m arrowmetal.pandas_accel my_script.py
ARROWMETAL_PANDAS_ACCEL=1 python my_script.py
```

A call is routed to the GPU only when **all four** hold:

1. **dtype** — the column is numeric, boolean or utf8, in any of the three pandas storage flavours
   (numpy-backed, `pd.ArrowDtype` / `int64[pyarrow]`, or the masked nullable dtypes `Int64`,
   `Float64`, `boolean`, `string`), including pandas 3's default `str` dtype;
2. **size** — the frame is at least `threshold` rows, 2,000,000 by default
   (`install(threshold=...)`, `set_threshold(...)`, or `ARROWMETAL_PANDAS_ACCEL_THRESHOLD`).
   Below that the launch latency costs more than the kernel saves;
3. **arguments** — the call uses arguments the kernels implement exactly (the "not routed" column
   below);
4. **the operation is worth it** — see the next section. Five reductions, `abs` and the scalar
   comparisons are intercepted but deliberately left to pandas.

Everything else runs in pandas, unchanged. **Any `Exception` inside the GPU path is caught, recorded
in `stats()`, and the original pandas method is run instead**, so a GPU failure does not surface as a
program failure. The one exception is repeated `install()`/`uninstall()` cycling — see Limits.

### Why some operations are intercepted but not routed

Handing a pandas column to Metal is zero-copy — the GPU reads the very bytes pandas holds, and
`am.zero_copy_report` proves it by address — when the buffer is page aligned, which it is for every
allocation at these sizes; a small unaligned buffer costs one copy. But it is not *free*: the pages
have to be mapped into the device's address space once. On an M4 Max, for a 50M-row (400 MB)
`int64[pyarrow]` column:

| Step | Time |
|---|---|
| `Series.array._pa_array` → `pyarrow.Array` | 0.00 ms (the same buffer) |
| that buffer mapped into Metal (`MetalArray.from_arrow`) | 6.7 ms |
| `sum` on the GPU, column already mapped | **1.07 ms** (374 GB/s) |
| `Series.sum()` in pandas | 4.9 ms |

The kernel is 4.6x faster than pandas. The map is not, and one `sum` cannot amortise it. So the
default routing table leaves alone every operation that reads each byte once and writes at most one
byte back — `sum`, `min`, `max`, `mean`, `count`, `abs`, and the six scalar comparisons — and routes
the ones that do enough per byte to pay for the map many times over: sorts, hash group-bys, merges,
string scans, `round`, `isin`, `nunique`, `value_counts`, boolean-mask selection.

That table is data, not a hard-coded rule:

```python
accel.NEVER_BY_DEFAULT     # ('sum', 'min', 'max', 'mean', 'count', 'abs', 'eq', 'ne', ...)
accel.ROW_FACTOR           # op -> rows needed as a multiple of the threshold, or None for "never"
accel.route_all()          # route everything anyway
accel.ROW_FACTOR["sum"] = 1
```

The `.am` accessor has no such table: it always dispatches to the GPU — bar the per-row host fallback
the Unicode case transforms take on non-Latin rows — and `s.am.to_metal()` gives you the
mapped column so that a chain of operations pays the map once. Nothing about correctness changes
either way — only where the work runs.

### What is intercepted

| pandas call | Routed to the GPU when | Not routed (falls through) |
|---|---|---|
| `Series.sum/min/max/mean` | numeric column, and `route_all()` | by default, always — see above; also `skipna=False`, `min_count>0`, `numeric_only` |
| `Series.count` | numeric, boolean or utf8, and `route_all()` | by default, always — see above |
| `Series.nunique` | numeric, boolean or utf8 | — |
| `Series.value_counts` | numeric, boolean or utf8 | `normalize=True`, `bins=` |
| `Series.sort_values` | numeric, boolean or utf8 | `key=`, `na_position="first"`, `inplace=True` |
| `Series.nlargest/nsmallest` | numeric | `keep != "first"` |
| `Series.isin` | numeric or utf8 | any `NaN`/`None` in `values` (pandas matches NaN to NaN; Arrow's `is_in` does not) |
| `Series.abs` | numeric, and `route_all()` | by default, always — see above |
| `Series.round` | float | integer columns (pandas' answer is the identity, so there is nothing to win) |
| `Series.__eq__ __ne__ __lt__ __le__ __gt__ __ge__` | numeric vs a numeric scalar, or utf8 vs a string for `==`/`!=`, and `route_all()` | by default, always — see above; also another Series (pandas aligns indexes first), `NaN` as the operand, ordering comparisons on strings |
| `Series[bool_mask]` | any supported dtype, mask of the same length and index | anything but a full-length boolean mask |
| `Series.str.contains` | literal pattern | a real regex, `case=False`, `flags=`, explicit `na=` |
| `Series.str.startswith/endswith` | a single string pattern | a tuple of patterns |
| `Series.str.upper/lower` | the column is pure ASCII (checked on the GPU: byte length == code point count) | any non-ASCII value |
| `Series.str.len` | utf8 | — |
| `DataFrame[bool_mask]` | every column a supported dtype | a mixed frame with, say, a datetime column |
| `DataFrame.sort_values` | one or several supported columns | `key=`, `na_position="first"`, `inplace=True` |
| `DataFrame.merge` | `how="inner"`, one key column, unique and null-free on the right | outer/left/right joins, several key columns, duplicate or null right keys, `indicator`, `validate` |
| `DataFrame.groupby(...).sum/mean/min/max/count` | key columns of a supported dtype, numeric value columns | `min_count`, `numeric_only`, a callable or array grouper, `level=`, non-numeric values |

Anything not in that table — `apply`, `rolling`, `pivot_table`, `resample`, `describe`, arithmetic
between columns, and every dtype outside the list (categorical, datetime, timedelta, object,
interval, period) — is untouched pandas.

### Control

```python
from arrowmetal import pandas_accel as accel

accel.install(threshold=5_000_000)
accel.set_threshold(1_000_000)
with accel.disabled():                  # a block of guaranteed plain pandas
    ...
accel.stats()                           # what ran where
accel.stats().to_frame()                # the same as a DataFrame: gpu / cpu / errors per op
accel.reset_stats()
accel.uninstall()                       # every original method back
accel.REGISTRY                          # the contract, as data
accel.ROW_FACTOR                        # the routing table, as data
accel.route_all()                       # route the operations pandas is otherwise faster at
```

`stats()` reports four things per operation: `gpu` (routed), `cpu` (fell through, with the reason
implied by the table above), `errors` (the GPU path raised and pandas ran instead — each with the
exception text), and `intercepted`, the total number of calls that reached a patched method at all.

---

## Zero-copy: what happens

`Series.array._pa_array` is a `pyarrow.ChunkedArray` for any Arrow-backed pandas column. The bridge
takes its single chunk and hands those buffers to Metal — the same bytes, no copy. Apple silicon's
unified memory means the GPU then reads the pages the CPU wrote, with no transfer either.

`am.zero_copy_report(df)` does not guess: it compares the buffer address pandas holds with the one
handed over.

```python
>>> am.zero_copy_report(pd.Series(pd.array([1, 2, 3], dtype="int64[pyarrow]")))
{'zero_copy': True, 'reason': 'Arrow-backed pandas column, buffers shared',
 'dtype': 'int64[pyarrow]', 'arrow_type': 'int64'}
```

The "none" rows are copy-free when the buffer is page aligned, which it is for every allocation at these
sizes; a small unaligned buffer costs one copy.

| pandas column | Copy? | Why |
|---|---|---|
| `pd.ArrowDtype` / `int64[pyarrow]`, one chunk | **none** | the Arrow buffers are handed straight over |
| pandas 3 default `str` dtype (`ArrowStringArray`) | **none** | same — it is a `large_string` Arrow array |
| Arrow-backed but split into several chunks | one | combined once into a contiguous array; reported as `"...split into N chunks; combined once"` |
| numpy `int*`/`uint*` | **none** | pyarrow adopts the numpy buffer; there is no validity bitmap to build |
| numpy `bool` | one | Arrow packs booleans to one bit per value, so the buffer cannot be adopted |
| numpy `float*`, `datetime64` | values none, validity one pass | the values buffer is adopted, and one pass builds the validity bitmap from the NaNs/NaTs |
| `Int64`, `Float64`, `boolean`, `string` (masked nullable) | one | pandas keeps values and mask in two arrays; Arrow needs a bitmap |
| `Categorical` | one | becomes an Arrow dictionary array, indices widened to int32 |

Results come back the same way: `am.to_pandas(...)` and every `.am` method wrap the Metal buffers in
a `pd.arrays.ArrowExtensionArray`, so the returned Series shares memory with the GPU result. Accel
mode adds one cast on top, to give you back the dtype pandas itself would have produced.

## Nulls, NaN and the index

- **A numpy float column has no null**, so `NaN` becomes Arrow `null` on the way in. That is exactly
  what `skipna=True` — pandas' default — already means, so `sum`, `mean`, `min` and `max` agree.
- **An Arrow-backed float column has both**, and the bridge does not touch either: a `NaN` stays a
  `NaN` and a null stays a null.
- **Coming back**, accel mode restores pandas' own convention: a null becomes `NaN` for a numpy
  float result, `False` for a plain `bool` result, `True` for a plain `bool` result of `!=`
  (`NaN != x` is `True` in numpy), and the dtype's own NA for a nullable dtype. The rule is not
  hand-written per operation: each wrapper runs the original pandas method over a two-row probe of
  the same dtype and copies the dtype and NA convention it sees.
- **The index is carried along by the same permutation as the values** — `sort_values`, `nlargest`,
  `nsmallest` and boolean-mask selection all return the original index rows, in the new order.
  `ignore_index=True` gives a fresh `RangeIndex`, as in pandas. Group-by results are indexed by the
  key (or by a `MultiIndex` for several keys), and `as_index=False` resets it, as in pandas.
- **Group-by follows pandas**: null keys are dropped unless `dropna=False`, groups come back sorted
  by key unless `sort=False` (then in first-seen order), and a group whose values are all null sums
  to `0`.
- **Inner merge preserves left-row order** and gives the result a fresh `RangeIndex`, as pandas does,
  with the same `_x`/`_y` suffixing for overlapping columns.

### The one deliberate difference

`sort_values` on the GPU is **stable**. pandas' default is `kind="quicksort"`, which is not, so rows
with equal sort keys can come back in a different order than pandas would give them (the same order
as `kind="stable"`). Every other value, dtype and index matches. If that matters, use the accessor
explicitly or keep `sort_values` out of the accelerated set with `accel.disabled()`.

## Numbers

Measured with `PYTHONPATH=python python Benchmarks/pandas_bench.py <rows> <iters>` on an Apple
M4 Max, pandas 3.0.5 / pyarrow 25.0.1 / Python 3.13, best of five, on the same in-process frame.
The accessor and accel rows include the conversion the dtype forces, the map into Metal, the GPU
work, and the trip back into a pandas object — nothing is pre-converted or cached between
iterations, so these are the numbers a pandas user gets, not kernel times.

`routes to` is where accel mode sent the call with the default table; the `accessor` column is
always the GPU, so it also shows what those operations would cost if you routed them.

### Arrow-backed frame (`pd.ArrowDtype`, what `read_parquet(dtype_backend="pyarrow")` gives you)

| Operation | 10M pandas | 10M accessor | 10M accel | 10M accel x | 50M pandas | 50M accessor | 50M accel | 50M accel x | routes to |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `sum` (int64) | 1.0 | 2.4 | 1.0 | 1.01x | 4.9 | 10.5 | 4.8 | 1.02x | pandas |
| `mean` (float64) | 1.2 | 2.6 | 1.2 | 0.96x | 5.6 | 14.5 | 5.7 | 0.98x | pandas |
| `s > 0` | 1.5 | 2.6 | 1.4 | 1.14x | 6.4 | 11.1 | 6.5 | 0.99x | pandas |
| `abs` (float64) | 1.6 | 2.6 | 1.6 | 0.98x | 7.6 | 13.0 | 7.5 | 1.02x | pandas |
| `round(2)` | 22.3 | 4.1 | 4.9 | **4.55x** | 110.9 | 18.5 | 22.6 | **4.91x** | GPU |
| `isin` (10 values) | 66.8 | 3.3 | 8.8 | **7.62x** | 337.4 | 13.8 | 39.3 | **8.59x** | GPU |
| `nunique` | 35.5 | 9.8 | 9.6 | **3.69x** | 176.1 | 18.8 | 19.2 | **9.17x** | GPU |
| `value_counts` (utf8) | 95.5 | 38.9 | 39.5 | **2.42x** | 476.6 | 157.7 | 170.4 | **2.80x** | GPU |
| groupby-sum, 1k keys | 99.4 | 15.4 | 15.9 | **6.26x** | 504.3 | 32.4 | 35.6 | **14.18x** | GPU |
| groupby-sum, 100k keys | 103.0 | 17.7 | 16.4 | **6.26x** | 450.7 | 34.9 | 35.2 | **12.80x** | GPU |
| `sort_values` (int64) | 392.7 | 69.9 | 62.4 | **6.30x** | 2312.1 | 324.9 | 305.3 | **7.57x** | GPU |
| `nlargest(100)` | 53.3 | 7.7 | 8.0 | **6.64x** | 264.0 | 28.9 | 29.0 | **9.10x** | GPU |
| `str.contains` | 97.6 | 3.4 | 3.7 | **26.06x** | 492.6 | 16.3 | 18.8 | **26.22x** | GPU |
| `df[df.i > 0]` | 58.2 | 35.6 | 30.0 | **1.94x** | 284.6 | 124.6 | 105.2 | **2.71x** | GPU |
| `merge`, 100k int keys | 233.4 | 37.9 | 38.0 | **6.15x** | 1258.5 | 132.6 | 134.8 | **9.34x** | GPU |

### numpy-backed frame (plain `pd.DataFrame({...})`)

Every column has to be converted to Arrow first, and pandas' numpy kernels are much faster than its
pyarrow ones, so the speedups are smaller — and `round`/`isin` are left to pandas here.

| Operation | 10M pandas | 10M accessor | 10M accel | 10M accel x | 50M pandas | 50M accessor | 50M accel | 50M accel x | routes to |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `sum` (int64) | 1.1 | 2.7 | 1.1 | 1.00x | 5.1 | 11.2 | 5.2 | 0.97x | pandas |
| `mean` (float64) | 4.1 | 23.3 | 4.1 | 0.99x | 21.1 | 116.3 | 20.6 | 1.02x | pandas |
| `s > 0` | 1.2 | 3.0 | 1.1 | 1.04x | 5.4 | 12.6 | 5.5 | 0.99x | pandas |
| `abs` (float64) | 1.5 | 22.7 | 1.5 | 1.00x | 7.4 | 110.8 | 7.3 | 1.01x | pandas |
| `round(2)` | 4.1 | 25.6 | 4.1 | 0.99x | 20.3 | 120.5 | 20.6 | 0.98x | pandas |
| `isin` (10 values) | 8.0 | 4.6 | 7.7 | 1.05x | 41.5 | 14.2 | 41.2 | 1.01x | pandas |
| `nunique` | 16.7 | 9.6 | 9.8 | **1.69x** | 89.8 | 20.3 | 19.0 | **4.72x** | GPU |
| `value_counts` (utf8) | 91.5 | 43.4 | 43.2 | **2.12x** | 460.6 | 178.1 | 185.1 | **2.49x** | GPU |
| groupby-sum, 1k keys | 38.7 | 9.4 | 9.3 | **4.16x** | 198.9 | 33.2 | 36.7 | **5.42x** | GPU |
| groupby-sum, 100k keys | 77.8 | 18.9 | 18.6 | **4.19x** | 362.7 | 40.9 | 36.8 | **9.85x** | GPU |
| `sort_values` (int64) | 1068.4 | 95.6 | 89.5 | **11.94x** | 6178.0 | 473.1 | 434.6 | **14.22x** | GPU |
| `nlargest(100)` | 52.3 | 11.3 | 11.1 | **4.73x** | 267.6 | 31.3 | 29.8 | **8.98x** | GPU |
| `str.contains` | 98.0 | 8.4 | 13.0 | **7.53x** | 520.2 | 32.0 | 56.6 | **9.19x** | GPU |
| `df[df.i > 0]` | 56.7 | 61.4 | 57.7 | 0.98x | 274.2 | 259.5 | 237.1 | **1.16x** | GPU |
| `merge`, 100k int keys | 147.6 | 120.5 | 77.6 | **1.90x** | 638.0 | 262.1 | 285.3 | **2.24x** | GPU |

Read it this way: **an Arrow-backed frame is where the GPU pays.** Group-by, sort, top-k, merge and
string scanning are 6x to 26x on a 50M-row frame with no code change at all; the operations pandas
already does at memory bandwidth are left alone and cost nothing; and the whole-frame operations
(`sort_values` on a DataFrame, `df[mask]`, `merge`) carry every column across, which is why they
land lower than the single-column ones.

Two caveats visible in the numbers. The GPU CPU-ms column (in the benchmark output, not repeated
here) is roughly a third of the wall time on group-by and sort, so the cores are free while the GPU
works — a real advantage the wall-clock ratio understates. And a numpy **float** column costs about
2 ms per million values to convert, because building the validity bitmap from the NaNs is a
single-threaded pyarrow pass: that is the whole difference between the accessor's 13 ms for `abs` on
an Arrow-backed 50M-row float column and its 111 ms on the numpy one, and it is why every
single-pass float operation stays in pandas there.

## Limits

- **Only these dtypes.** Numeric, boolean and utf8. Categorical, datetime, timedelta, object,
  interval and period columns are left to pandas: their semantics are richer than the kernels here,
  and a wrong answer is worse than a slow one. (The *bridge* converts categoricals and datetimes
  fine — `am.from_pandas` handles them; it is the accel layer that will not route them.)
- **`merge` is the `validate="m:1"` inner join.** Duplicate or null keys on the right frame, and any
  outer/left/right join, fall back. A many-to-many join needs a GPU expansion the C ABI does not have
  yet.
- **`str.upper`/`str.lower` are ASCII-guarded in accel mode.** The kernels implement Unicode's simple
  1:1 mapping over every script, but pandas applies the *full* mapping (`ß` → `SS`), so accel mode
  checks for pure ASCII on the GPU and falls back otherwise; it never returns a different string
  than pandas would. The `.am` accessor runs the simple mapping on anything.
- **`str.contains` with a real regex falls back.** A literal pattern runs on the GPU.
- **Float reductions add in a different order.** A GPU tree reduction is not bit-identical to
  pandas' pairwise sum; expect agreement to about 1e-12 relative, not to the last bit.
- **Below the threshold nothing is accelerated.** At a million rows most of these operations are
  already sub-millisecond in pandas and the GPU launch cannot pay for itself. The default of two
  million rows is deliberately conservative; lower it if your columns are wide.
- **Single-pass operations are not accelerated at all**, at any size: `sum`, `min`, `max`, `mean`,
  `count`, `abs` and the scalar comparisons, plus `round` and `isin` on a numpy-backed column. The
  measurement and the reasoning are above; `route_all()` overrides it, and the `.am` accessor never
  applies it.
- **A numpy float column costs about 2 ms per million values to convert**, because pyarrow builds
  the validity bitmap from the NaNs in a single-threaded pass. Arrow-backed columns skip it
  entirely — which is the single biggest thing you can do to make this faster: read your data with
  `dtype_backend="pyarrow"`.
- **Install once.** `install()` and `uninstall()` rewrite pandas' method slots. Cycling them a
  hundred-odd times in one process can leave an already-specialized `x[mask]` call site bound to
  whichever `__getitem__` CPython first saw — an interpreter-level artifact of repeatedly rewriting a
  dunder. Use `accel.disabled()` to turn the layer off for a block instead.
- **Threads.** The wrappers are re-entrancy-guarded per thread, but the GPU queue underneath is a
  single device queue; several Python threads calling into it will serialise. `disabled()`,
  `set_threshold()` and `route_all()` are **process-wide**, not per thread: while one thread is
  inside a `disabled()` block every thread runs in pandas. The answers do not change — only where
  the work happens, and those calls land in neither `stats().gpu` nor `stats().cpu`.

## Tests

```
swift build -c release --product ArrowMetalC
PYTHONPATH=python python -m pytest python/tests/test_pandas.py -q
```

`python/tests/test_pandas.py` covers: a GPU round trip for twenty pandas dtypes; zero-copy proved by
buffer address for each Arrow-backed flavour and disproved for each copying one; every accessor
method against its pandas equivalent; and, for accel mode, forty-plus operations run twice — once in
plain pandas, once with the layer installed — over numpy-backed, Arrow-backed and masked frames
carrying nulls, NaNs, strings and duplicate keys, with `assert_series_equal` / `assert_frame_equal`
checking values, dtype and index. Each of those also asserts that the GPU path really was taken, so
a test cannot pass by quietly falling back. The rest of the file is the fallback surface: unsupported
dtypes, unsupported arguments, regex patterns, non-ASCII case changes, `NaN` in `isin`, duplicate and
null merge keys, an injected exception inside the GPU path, the row threshold in both directions,
`uninstall()` restoring every original, and the `ARROWMETAL_PANDAS_ACCEL=1` and
`python -m arrowmetal.pandas_accel` entry points in real subprocesses.
