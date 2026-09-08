# ArrowMetal for Polars users

Polars is the reason most people on an Apple silicon Mac have Arrow-shaped data in memory at all.
This document is how you point that data at the GPU.

There are three tiers, all in 0.1.0, and they differ in **where the GPU sits
relative to the Polars plan**:

| Tier | Where the GPU runs | What you write | Needs |
|---|---|---|---|
| 1. Bridge and namespaces | Around Polars: you hand a collected frame over | `df.arrowmetal.group_by("k").sum("v")` | Python only |
| 2. Expression plugin | Inside a Polars lazy plan | `pl.col("v").arrowmetal.sum()` | a Rust build |
| 3. Streaming hand-off | Polars runs the plan, ArrowMetal finishes it | `lf.arrowmetal.collect_gpu(q)` | Python only |

All three move data over the Arrow C Data Interface. For a single-chunk numeric Polars column that
is **no copy at all** -- the GPU reads the buffer Polars already owns. Strings, Categoricals and
multi-chunk Series each cost one conversion pass -- see Limits. The evidence is below.

---

## Install

```bash
# 1. The GPU library (all three tiers need it)
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
    swift build -c release --product ArrowMetalC

# 2. The Python package
pip install polars pyarrow
export PYTHONPATH=python          # or: pip install ./python

# 3. Tier 2 only: the Polars expression plugin
cd polars-plugin && cargo build --release && cd ..
```

That is the whole build, and both halves are checked: on an M4 Max the Swift product and the cargo
release build both go through from an empty `target/`, with no other flags and no
`DYLD_LIBRARY_PATH`. `cargo build --release` is enough for the plugin -- it is a plain `cdylib`
that Polars `dlopen`s, not a Python extension module, so `maturin` is optional, and
`arrowmetal.polars_plugin.plugin_path()` finds `polars-plugin/target/release/` on its own.
`ARROWMETAL_POLARS_PLUGIN` overrides the search.

`maturin develop --release` is the **untested** path: `polars-plugin/` has no `pyproject.toml`,
and `plugin_path()` looks for a maturin install at
`<sys.path>/arrowmetal_polars/libarrowmetal_polars.dylib`, which is not the layout maturin
produces for a `cdylib` crate. Use `cargo build --release`, or set `ARROWMETAL_POLARS_PLUGIN` to
whatever you did build.

The plugin's `build.rs` links `libArrowMetalC.dylib` by **rpath**. The dylib's install name is
`@rpath/libArrowMetalC.dylib`, so `polars-plugin/arrowmetal-sys/build.rs` finds the directory
holding it -- `$ARROWMETAL_LIB_DIR`, `$ARROWMETAL_LIB`, `<repo>/.build/release`,
`<repo>/.build/debug`, `/usr/local/lib`, `/opt/homebrew/lib`, first hit wins -- and bakes it in as
an `LC_RPATH` entry. No `DYLD_LIBRARY_PATH` is needed at run time. It republishes that directory
to the plugin crate through its `links = "ArrowMetalC"` key (`DEP_ARROWMETALC_LIB_DIR`), because
Cargo passes a build script's link arguments only to the crate that owns it.

### Version pinning

`polars-plugin/Cargo.toml` pins `polars` 0.55.1 and `pyo3-polars` 0.28.0. Those are the Rust
crates **py-polars 1.44.x** is built from: the polars workspace at tag `py-1.44.1` carries
`version = "0.55.1"`, and pyo3-polars 0.28 is the release that depends on `polars ^0.55.1`. The
requirement is a caret, so `Cargo.lock` currently resolves polars 0.55.2 and `polars-ffi` 0.55.2;
the ABI check is on `polars_ffi`'s major/minor, so that patch bump loads against py-polars 1.44.1
without complaint.

Polars checks the plugin ABI when it loads the library (`_polars_plugin_get_version`, which
returns `polars_ffi`'s major/minor packed into a `u32`) and refuses a mismatched pair with
`this Polars engine doesn't support plugin version: ...`. Moving to a different Polars means
re-pinning both crates to whatever that release's workspace version is -- read it from
`https://raw.githubusercontent.com/pola-rs/polars/refs/tags/py-<version>/Cargo.toml`, do not
guess. Tiers 1 and 3 have no such constraint: they are pure Python over the C Data Interface.

---

## Tier 1 -- the zero-copy bridge and the `.arrowmetal` namespaces

```python
import polars as pl
import arrowmetal as am
```

Importing `arrowmetal` does **not** import Polars -- the package's only hard dependency is
pyarrow. The Polars half arrives on first use. Any of these arms it:

```python
import arrowmetal.polars_bridge          # explicit
am.from_polars(df)                       # first touch of a bridge function
import polars as pl; import arrowmetal   # Polars already loaded -> registered at import
```

### Two functions

```python
am.from_polars(series)      # -> MetalArray
am.from_polars(dataframe)   # -> dict[str, MetalArray]
am.to_polars(metal_array)   # -> pl.Series
am.to_polars({"a": ...})    # -> pl.DataFrame
```

### Three namespaces

```python
# Series
s.arrowmetal.sum() / .min() / .max() / .mean() / .count()
s.arrowmetal.top_k(100) / .bottom_k(100) / .sort(descending=True) / .arg_sort()
s.arrowmetal.filter(mask) / .unique() / .cum_sum() / .hash64()
s.arrowmetal.contains("x") / .starts_with("x") / .ends_with("x") / .upper() / .lower()
s.arrowmetal.to_metal()                     # keep the column on the GPU

# DataFrame
df.arrowmetal.group_by("k").sum("v")
df.arrowmetal.group_by("k", "region").agg(total=("v", "sum"), avg=("f", "mean"), n=(None, "len"))
df.arrowmetal.query(am.filter(am.col("k") == 2).sum(am.col("v")))
df.arrowmetal.sort(["k", "v"], descending=[False, True])
df.arrowmetal.top_k(10, by="v")
df.arrowmetal.filter(df["k"] > 8)
df.arrowmetal.join(other, on="k", how="inner")
df.arrowmetal.to_metal() / .device()

# LazyFrame
lf.arrowmetal.collect_gpu(query_or_callable)
```

Results come back as Polars objects. Scalar reductions come back as Python scalars, as
`pl.Series.sum()` does -- with one difference, in both tier 1 and tier 2: the sum of an **empty or
all-null** column is `None`, where `pl.Series.sum()` answers `0`. `am_reduce` has nothing to add
up and says so; `min`, `max` and `mean` are `None` on both sides.

Grouped aggregates available through `.agg`: `sum`, `mean`, `min`, `max`, `count`, `len`,
`n_unique`, `first`, `last`, `median`, `std`, `var`, `product`, `any`, `all`, plus
`gb.quantile(column, q)`. One `am_group_by_keys` pass builds the dense group ids and every
aggregate after that reuses it, so `.agg(...)` with six outputs costs **one** group-by.

### What each tier-1 method runs

| Method | ArrowMetal entry point | Note |
|---|---|---|
| `sum` / `min` / `max` / `mean` | `am_reduce` | integers widen to 64 bits, Arrow's rule |
| `top_k` / `bottom_k` | `am_top_k` + `am_take` | GPU radix sort |
| `sort` / `arg_sort` | `am_sort` / `am_argsort` | stable, nulls last, NaN after +inf |
| `filter` | `am_filter` | GPU stream compaction |
| `unique` | `am_unique` | **first-seen** order, nulls kept; Polars' plain `unique()` promises no order |
| `hash64` | `am_hash64` (strings: `am_str_unary` kind 2) | Arrow-equal values hash equal |
| `contains` / `starts_with` / `ends_with` | `am_str_match` | literal, not regex |
| `upper` / `lower` | `am_str_transform` 2/3 | simple 1:1 case mapping, see limits |
| `cum_sum` | `am_cumulative` | two-level GPU scan |
| `df.group_by(...)` | `am_group_by_keys` + `am_group_agg_ex` | any key type; several keys fold |
| `df.sort(...)` | `am_lexsort` + `am_take` | one argsort, one take per column |
| `df.query(...)` | `am_query` | the whole expression DAG as **one** generated kernel |
| `df.join(...)` | `am_index_in` + `am_take` / `am_filter` | see the join section |

### The join

The bridge does not call the C ABI's `am_join` hash join yet; `df.arrowmetal.join` is built out of
two other kernels. `am_index_in` finds, for every left key, the row of the right key column it matches;
`am_take` and `am_filter` then gather both sides. That is a complete **inner** or **left** join
whenever the **right key is unique** -- the usual dimension-table shape -- and the whole thing,
uniqueness check included (one `am_group_by_keys`: as many groups as rows means every key is
distinct), runs on the GPU with no row-by-row work on the host.

Everything else falls back to `pl.DataFrame.join`: a duplicated right key (which changes the row
count in a way `index_in` cannot express), a multi-column key, or an outer/semi/anti join. Pass
`allow_cpu_fallback=False` to get an error instead, so a benchmark can be sure what it measured.
Null keys never match, which is Polars' default `join_nulls=False`.

### Column pruning in `df.arrowmetal.query`

`query` imports only the columns the query names -- it reads them out of the serialised query's
`(col "name")` nodes. This matters more than it sounds: on the benchmark frame, importing all
four columns (one of them 50M strings) cost more than the entire query did.

---

## Tier 2 -- the expression plugin

```python
import polars as pl
import arrowmetal.polars_plugin        # registers the namespace

lf.select(pl.col("amount").arrowmetal.sum())
lf.with_columns(pl.col("name").arrowmetal.upper())
lf.select(pl.col("amount").arrowmetal.filter_sum(pl.col("region") == 2))
lf.group_by("k").agg(pl.col("v").arrowmetal.sum())
```

These are real Polars expressions: they compose with `select`, `with_columns`, `filter`,
`group_by`, `over`, and they take part in projection and predicate pushdown. That is the whole
point of the tier -- tier 1 needs a materialised frame, tier 2 does not.

| Expression | Output | Registered as |
|---|---|---|
| `.sum()` | Int64 / UInt64 / Float64 | `returns_scalar` |
| `.min()` / `.max()` | the column's own dtype | `returns_scalar` |
| `.mean()` | Float64 | `returns_scalar` |
| `.filter_sum(predicate)` | as `.sum()` | `returns_scalar` |
| `.top_k(k)` / `.bottom_k(k)` | the column's own dtype | `changes_length` |
| `.hash64()` | UInt64 | `is_elementwise` |
| `.contains(p)` / `.starts_with(p)` / `.ends_with(p)` | Boolean | `is_elementwise` |
| `.upper()` / `.lower()` | String | `is_elementwise` |
| `.add(x)` / `.sub(x)` / `.mul(x)` / `.truediv(x)` | the column's own dtype | `is_elementwise` |
| `.group_by_sum(values)` | `Struct{key, sum}`, one row per group | `changes_length` |
| `.device()` | String, one row | `returns_scalar` |

`.filter_sum` is the shape that pays for itself: one GPU compaction plus one reduction, with the
filtered column never crossing back into Polars.

### What tier 2 accepts

Narrower than the bridge, and not the same list as the "Types" paragraph under Limits -- that one
is about what *round-trips*, which is a tier-1 and tier-3 question.

| Expression | Dtypes |
|---|---|
| `.sum()` / `.min()` / `.max()` / `.mean()` / `.filter_sum()` / `.top_k()` / `.add` … / `.group_by_sum()` | Int8/16/32/64, UInt8/16/32/64, Float32/64 -- nothing else |
| `.hash64()` | those, plus Boolean, Date, Datetime, Duration, Time and String |
| `.contains` / `.starts_with` / `.ends_with` / `.upper` / `.lower` | String |

Everything else -- Boolean, the temporal types, Binary, Categorical, Enum, Decimal, List, Struct,
Null -- raises a Polars `ComputeError` whose message starts `arrowmetal:`. Nothing panics through
pyo3. **There are two places tier 2 is behind tier 1.** Categorical and Enum: the Python
bridge recodes Polars' `dictionary<uint32>` index buffer to int32 for the GPU, and
`polars-plugin/src/bridge.rs` does not, so a Categorical column reaches the kernel as-is and is
refused with "dictionary indices must be int32 or int64". And Decimal: `am_decimal_op`
backs `s.arrowmetal.sum()` in tier 1, and the plugin does not reach for it.

### The scalar in `.add` / `.sub` / `.mul` / `.truediv`

"Integers wrap" is about the **arithmetic**: `127 + 1` is `-128` on an Int8 column, and integer
division by zero is 0. It is not about the **operand**. A scalar the column's type cannot hold
exactly is an error, the same call the tier-1 bridge makes (it packs the scalar with
`struct.pack` at the column's own width, and `struct.pack` raises):

```python
pl.col("i8").arrowmetal.add(1000)      # raises: 1000 is out of range for an Int8 column
pl.col("u8").arrowmetal.add(-1)        # raises: -1 is out of range for a UInt8 column
pl.col("i64").arrowmetal.add(1.5)      # raises: an integer column takes an integer scalar
pl.col("i64").arrowmetal.add(2**60+1)  # exact -- the scalar does not go through an f64
```

A float column takes either an integer or a float, and an operand too large for Float32 becomes an
infinity, which is what `struct.pack("f", 1e300)` gives tier 1.

### `group_by_sum`, and what the plugin API cannot do

`pl.col("k").arrowmetal.group_by_sum(pl.col("v"))` returns a struct column of `n_groups` rows:

```python
df.select(pl.col("k").arrowmetal.group_by_sum(pl.col("v")).alias("g")).unnest("g")
```

It has to be a struct because a plugin function answers with a single Series. And it runs as a
**projection over the whole frame**, not inside `df.group_by(...).agg(...)`: Polars' expression
plugin API (`polars.plugins.register_plugin_function`) registers *expressions*, and has no hook
for contributing a hash aggregate to the group-by engine itself. The flags it accepts are
`is_elementwise`, `changes_length`, `returns_scalar`, `cast_to_supertype`,
`input_wildcard_expansion` and `pass_name_to_apply` -- none of them says "I am an aggregation the
group-by engine should call per group". `df.arrowmetal.group_by(...)` (tier 1) is the ergonomic
spelling; `pl.col("v").arrowmetal.sum()` inside `.agg(...)` also works and is called once per
group, which is the wrong granularity for a GPU on small groups.

### How the plugin gets the data across

`polars-plugin/src/bridge.rs`, in both directions:

```
Series --rechunk--> polars_arrow::ffi::ArrowArray --am_import--> am_array (Metal-resident)
am_array --am_export--> ArrowArray --import_array_from_c--> Series
```

`polars_arrow::ffi::ArrowArray` and `arrowmetal_sys::ArrowArray` are both `#[repr(C)]`
transcriptions of the same C struct, so the hand-off is a pointer cast. Ownership follows the C
Data Interface's consumer-releases rule: `am_import` takes over the exported array (so the Rust
side `mem::forget`s its copy, exactly as the Python binding does after `_export_to_c`), and
`import_array_from_c` takes over the exported one coming back.

Strings go through `CompatLevel::oldest()` -- Arrow `LargeUtf8`, not the `Utf8View` layout Polars
uses natively -- because ArrowMetal's kernels read offsets plus bytes. That conversion is the
plugin's only copy, and it is why the string rows below are 2.2x rather than 89x.

`arrowmetal-sys` is a hand-written transcription of the header, not bindgen output: the surface
is small, the header is stable, and a checked-in file needs no libclang on the build machine.
`cargo test` in `polars-plugin/arrowmetal-sys` runs 10 tests against the real dylib (import,
export, reductions, compare + filter, scalar arithmetic, top-k + take, hash64, group-by, and the
error path).

---

## Tier 3 -- the streaming hand-off, and what a real Metal backend would need

```python
lf = pl.scan_parquet("trades/*.parquet").filter(pl.col("day") == "2026-09-01")

lf.arrowmetal.collect_gpu(am.filter(am.col("region") == 2).sum(am.col("amount")))
lf.arrowmetal.collect_gpu(lambda df: df.arrowmetal.group_by("k").sum("v"))
lf.arrowmetal.collect_gpu()                     # just lf.collect()
```

`collect_gpu` collects the Polars plan and then runs one ArrowMetal pass over the result. Its
`engine=` and any other keyword go straight to `LazyFrame.collect`, so the Polars half can still
use the streaming engine: projection pushdown, predicate pushdown and `slice` all happen before a
single byte reaches the GPU. It is an explicit hand-off, and it is explicit because Polars'
engine hook is not open to us yet. Here is exactly why.

### The `engine=` hook, as it stands in polars 1.44.1

Read from the installed package, not from memory:

* `polars/_typing.py:443` --
  `EngineTypeName: TypeAlias = Literal["auto", "in-memory", "streaming", "gpu"]`, and
  `EngineType: TypeAlias = Union[EngineTypeName, "Engine"]`. So `collect(engine=...)` does accept
  an `Engine` **object**, not only a name.
* `polars/lazyframe/engine.py:77` -- `class Engine(ABC)`, documented as "Subclass this to plug a
  new backend into Polars", with an abstract `name` property, a `plan_engine` property, and
  `collect` / `execute` / `collect_async` / `collect_batches` / the `sink_*` family.
* `polars/lazyframe/engine.py:330` -- `class _LocalEngine(Engine)`, "Base for in-process engines
  backed by `PyLazyFrame`". Its `collect` ends in `wrap_df(ldf.collect(self.name, callback))`.
* The callback comes from `_LocalEngine._post_opt_callback(*, background, eager)`, typed
  `PostOptCallback | None` where `PostOptCallback: TypeAlias = Callable[[Any, int | None], None]`
  (`polars/_typing.py:450`). The base returns `None`.
* `polars/lazyframe/engine.py:884` -- `class GPUEngine(_LocalEngine)` with `_name = "gpu"`. Its
  `_post_opt_callback` imports `cudf_polars` and returns
  `partial(cudf_polars.execute_with_cudf, config=self)`. It refuses background collection and
  opts out in eager mode.
* `polars/lazyframe/engine_config.py:28` --
  `SUPPORTED_ENGINE_NAMES = ("auto", "in-memory", "streaming", "gpu")`, and `_engine_from_name`
  maps the string `"gpu"` to `GPUEngine()`.
* The callback's first argument is a `NodeTraverser` (`polars/_plr.pyi:2595`), whose surface is:
  `get_exprs()`, `get_inputs()`, `version()`, `get_schema()`, `get_dtype(expr_node)`,
  `set_node(node)`, `get_node()`, `set_udf(function, is_pure=False)`, `view_current_node()`,
  `view_expression(node)`, `add_expressions(expressions)`, `set_expr_mapping(mapping)`,
  `unset_expr_mapping()`.

So a Metal backend is **not** blocked on Polars adding an API -- the API is there. What a
`MetalEngine` would have to do:

1. Subclass `_LocalEngine` (or `Engine`) and give it a `name`. The name is passed to Rust as
   `ldf.collect(self.name, callback)`, and Rust only knows the four in `SUPPORTED_ENGINE_NAMES`,
   so today a third-party backend has to answer `"gpu"` to get the post-optimisation callback
   invoked at all. A first-class Metal backend wants either a fifth name or a Rust-side
   "call the callback for any unknown engine" rule; that is the one genuine upstream change.
2. Return a `PostOptCallback` from `_post_opt_callback`. It receives the `NodeTraverser` sitting
   on the optimised IR plus an optional node id, and returns `None` -- it works by mutation.
3. Walk the IR with `get_node` / `set_node` / `get_inputs` / `view_current_node` /
   `view_expression`, translating each node it recognises. Every node it does not recognise is
   where the backend must decide between falling back (leave the node alone) and raising
   (`raise_on_fail`, which `GPUEngine` exposes as a config flag).
4. Replace the translated subtree with `set_udf(callable)`: the callable is what Polars will
   execute for that node, and it must return a Polars `DataFrame`. This is the seam through
   which ArrowMetal would return `am.to_polars(...)`.
5. Handle the schema contract exactly: `get_schema()` and `get_dtype()` give the dtypes Polars
   will assume downstream, so a backend that widens an integer sum has to cast back.

The pieces ArrowMetal would need for step 3 already exist: `am_query` compiles a whole filter +
projection + aggregate DAG into one Metal kernel and is a close match for a Polars `SELECT` node,
`am_group_by_keys` + `am_group_agg_ex` cover the aggregate nodes, `am_lexsort` covers sort, the
engine's join matrix ([ENGINE.md](ENGINE.md)) covers the join nodes, and the scan nodes stay with
Polars. The missing piece is the IR translator itself, with an explicit unsupported-node list.
Until then, `collect_gpu` is the same idea in its explicit form: Polars owns the plan, ArrowMetal
owns one pass over the result.

---

## Numbers

Apple M4 Max, macOS 26.6.2, polars 1.44.1 (16 threads), pyarrow 25.0.1, ArrowMetal 0.1.0. Best of 5
runs after a warm-up, one process, one data set. Every figure below is from
`Benchmarks/results/polars_bench_50000000_2026-09-07.txt`. Reproduce with:

```
PYTHONPATH=python python Benchmarks/polars_bench.py 50000000 5
```

Columns: `k` Int32 with 1000 distinct values, `v` Int64, `amount` Float64, `name` String drawn
from 4096 distinct values. The "Polars" column is Polars' eager idiom; the lazy engine and
pyarrow's Acero are on the Compare tab and in the benchmark matrix.

### 50M rows

| Operation | Polars | tier 1 namespace | tier 2 plugin | GPU-resident |
|---|---|---|---|---|
| `sum(Int64)` | 4.1 ms / 4.1 CPU-ms | 10.1 ms (0.4x) | 8.5 ms (0.5x) | 1.1 ms (3.8x) |
| `filter(k == 2) + sum(v)` | 4.3 ms / 9.4 CPU-ms | 13.9 ms (0.3x) | 11.8 ms (0.4x) | 0.8 ms (5.1x) |
| group-by `sum(v)` by 1000 keys | 79.3 ms / 1121 CPU-ms | 20.1 ms (4.0x) | 19.7 ms (4.0x) | 1.9 ms, aggregate only, group ids cached |
| `top_k(100)` | 60.0 ms / 60.1 CPU-ms | 18.2 ms (3.3x) | 18.0 ms (3.3x) | 10.5 ms (5.7x) |
| string `contains` (literal) | 634.1 ms / 634.0 CPU-ms | 290.7 ms (2.2x) | 272.8 ms (2.3x) | 7.1 ms (89.3x) |

### Reading the table

* **The "Polars" column is Polars' eager idiom**, which is what `Benchmarks/polars_bench.py` measures
  and what `Benchmarks/results/polars_bench_50000000_2026-09-07.txt` records. The published baseline is the
  parallel run, `Benchmarks/results/full_matrix_2026-09-07-parallel.csv`, which adds Polars' lazy
  engine (`polars-lazy`, a median of 11.5 of the 16 cores); ratios there are lower on the rows where
  the lazy engine parallelises. [BENCHMARKS_MATRIX.md](BENCHMARKS_MATRIX.md) has both.
* **"GPU-resident"** is the same kernel with the column already in Metal memory -- the import is
  outside the timed region, and, for the group-by rows, the `am.group_by([k])` key-mapping pass as
  well: those rows time the aggregate only. It is what a pipeline that stays on the GPU sees, and it
  is the column that shows what the kernels are worth.
* **The group-by row is not a like-for-like ratio against Polars**, whose 79.3 ms includes its whole
  hash group-by. The comparable end-to-end figure is in the benchmark matrix: `sum by int32 key
  (1000 groups)` at 50M rows is **4.89 ms against Polars lazy's 81.93 ms, 16.8x**
  (`Benchmarks/results/full_matrix_2026-09-07-parallel.csv`), and 1.73 ms against 20.24 ms, 11.7x, at
  10M. The fastest CPU idiom on that row is pyarrow's Acero (`pyarrow-threaded`), 18.45 ms at 50M,
  which puts the ratio at 3.8x.
* **The hand-off is the whole difference** between the middle columns and the right one. At 50M
  rows the import is 5.25 ms and the export 0.01 ms, and the run records the import as a no-copy
  -- the Polars source buffer and the Metal buffer are the same address (`zero copy: ...
  -> SAME`). Every tier-1 and tier-2 row pays it once per call, so on a single `sum` Polars is
  ahead, and a group-by is 4x.
* **CPU-ms is the other half of the story.** The 50M group-by costs Polars 1121 CPU-ms across 16
  threads; ArrowMetal costs 14.0 CPU-ms end to end and 0.4 CPU-ms resident. On a laptop that is
  battery, and on a shared box it is 16 cores left free for something else.
* **Strings** are the exception in both directions: 89x resident, 2.2x through the bridge. Polars
  stores strings as `Utf8View` and ArrowMetal reads offsets + bytes, so the conversion is a real
  copy, and at 50M rows it dominates. Keeping a string column resident (`s.arrowmetal.to_metal()`)
  pays for itself immediately.

### Where each tier is worth using

| | Use it when |
|---|---|
| Tier 1, one call | The kernel is expensive relative to 400 MB of page mapping: group-by, sort, top-k, string search. Not a bare `sum`. |
| Tier 1, resident | You run several kernels over the same column. `to_metal()` once, then every kernel in the table above is 0.8-10.5 ms. |
| Tier 2 | The GPU op belongs inside a plan you want Polars to keep optimising -- scans, pushdown, and lazy composition still apply. |
| Tier 3 | Polars should do the IO and the reshaping and ArrowMetal should do one heavy pass at the end. |

---

## Limits

**Types.** This paragraph is about what the **bridge** round-trips, which is a tier-1 and tier-3
question; tier 2's expressions accept a shorter list, set out in "What tier 2 accepts" above.
Every dtype Polars and Arrow share round-trips: all signed and unsigned integer widths,
Float32/64, Boolean, Date, Datetime (all units), Time, Duration, String, Binary. `Categorical` and
`Enum` also work, but Polars encodes them as `dictionary<uint32>` and `dictionary<uint8>` while
ArrowMetal wants int32 or int64 indices, so the bridge recodes the index buffer -- 4 bytes a row,
values untouched. An `Enum` comes back from `to_polars` as a `Categorical`: the dictionary crosses,
the fact that its value set was closed does not. `List` and `Struct` cross and round-trip, but only
the structural kernels operate on them -- you cannot sum or group by one. `Object` is not bridged.

**Chunking.** ArrowMetal takes one Arrow array. A multi-chunk Series is rechunked once, which does
copy; `am.from_polars(s, rechunk=False)` raises instead, so the copy is never silent.

**Order.** Group order is ArrowMetal's, not Polars': ascending by key for numeric, boolean,
temporal and decimal keys, first-seen for utf8 and binary, lexicographic in column order for
several keys. Polars' `group_by` promises no order at all, so sort both sides before comparing.
`s.arrowmetal.unique()` is **first-seen**, the order Polars' `unique(maintain_order=True)` gives,
and it keeps a null as one of the distinct values rather than dropping it. Sorts put nulls last in
**both** directions, where Polars' ascending default is nulls first -- pass `nulls_last=True` when
comparing.

**Strings.** `upper`/`lower` are Unicode's simple 1:1 case mapping over every script (the GPU
table covers U+0000–U+017F exactly; a row holding anything above is mapped on the host), so Greek
and Cyrillic come back mapped. Simple, not full: the multi-character expansions are not applied,
so U+00DF becomes `ẞ` where Polars' `str.to_uppercase()` gives `SS`, and `ﬁ` stays put.
`contains` / `starts_with` / `ends_with` are literal, not regex.

**Arithmetic.** `.add/.sub/.mul/.truediv` in tier 2 keep the column's own type and follow Arrow's
*unchecked* rules for the operation: integers wrap, and integer division by zero yields 0 where
Polars raises. The **operand** is checked -- one the column's type cannot hold raises rather than
being clamped or truncated, and an integer operand is exact past 2^53. See "The scalar in `.add`"
above.

**Joins.** GPU path only for a single-column inner or left join against a unique right key;
everything else falls back to Polars (or raises with `allow_cpu_fallback=False`).

**Aggregation inside `group_by`.** Tier 2's `group_by_sum` is a projection, not a hash aggregate
Polars calls per group -- the plugin API has no hook for that. See the tier-2 section.

**Threads.** ArrowMetal serialises command-buffer commits behind its own lock and keeps its error
state thread-local, so Polars is free to call the plugin from several worker threads. The plugin
takes no lock of its own.

**Version coupling.** Tier 2 only: the plugin is pinned to polars 0.55.1 / pyo3-polars 0.28 for
py-polars 1.44.x. Tiers 1 and 3 speak the C Data Interface and are not coupled to a Polars
version.

---

## Tests

```
PYTHONPATH=python python -m pytest python/tests/test_polars.py -q     # 90 tests
cd polars-plugin/arrowmetal-sys && cargo test --release               # 10 tests
```

`test_polars.py` covers round trips for every shared dtype (plus Categorical, Enum, empty,
all-null and chunked), the namespace methods against native Polars at five sizes from 0 to
100,003 rows, the plugin expressions inside lazy plans, the join against `pl.DataFrame.join`,
zero-copy assertions on buffer addresses, and 50M-row timings as assertions with bounds generous
enough not to flake. The plugin tests skip when the Rust library has not been built, so a
checkout without a Rust toolchain still runs green -- **build it before you trust a green run**,
or 28 of the 90 are skips (`62 passed, 28 skipped`).

The tier-2 block at the end of the file is the adversarial pass: the scalar operand against the
tier-1 bridge, nulls against native Polars, a sliced Series and a two-chunk one, an empty frame,
an all-set validity bitmap with no nulls, the twelve dtypes the numeric expressions refuse (each
one a Polars error carrying ArrowMetal's wording, never a pyo3 panic), and the expression inside
`group_by().agg()` and inside `collect(engine="streaming")`.
