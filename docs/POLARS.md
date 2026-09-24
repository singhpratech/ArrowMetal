# ArrowMetal for Polars users

Polars is the reason most people on an Apple silicon Mac have Arrow-shaped data in memory at all.
This document is how you point that data at the GPU.

There are four tiers, all in 0.1.0, and they differ in **where the GPU sits
relative to the Polars plan**:

| Tier | Where the GPU runs | What you write | Needs |
|---|---|---|---|
| 1. Bridge and namespaces | Around Polars: you hand a collected frame over | `df.arrowmetal.group_by("k").sum("v")` | Python only |
| 2. Expression plugin | Inside a Polars lazy plan | `pl.col("v").arrowmetal.sum()` | a Rust build |
| 3. Streaming hand-off | Polars runs the plan, ArrowMetal finishes it | `lf.arrowmetal.collect_gpu(q)` | Python only |
| 4. `MetalEngine` | In place of whole subtrees of the optimised Polars plan | `lf.collect(engine=am.MetalEngine())` | Python only |

All four move data over the Arrow C Data Interface. For a single-chunk numeric Polars column that
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

## Tier 3 -- the streaming hand-off, and Polars' engine hook

```python
lf = pl.scan_parquet("trades/*.parquet").filter(pl.col("day") == "2026-09-01")

lf.arrowmetal.collect_gpu(am.filter(am.col("region") == 2).sum(am.col("amount")))
lf.arrowmetal.collect_gpu(lambda df: df.arrowmetal.group_by("k").sum("v"))
lf.arrowmetal.collect_gpu()                     # just lf.collect()
```

`collect_gpu` collects the Polars plan and then runs one ArrowMetal pass over the result. Its
`engine=` and any other keyword go straight to `LazyFrame.collect`, so the Polars half can still
use the streaming engine: projection pushdown, predicate pushdown and `slice` all happen before a
single byte reaches the GPU. It is an explicit hand-off. Polars' own `engine=` hook is the other
way in, and tier 4 uses it; this section is what the hook is, read from the installed package.

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
   `ldf.collect(self.name, callback)`, and Rust only knows the four in `SUPPORTED_ENGINE_NAMES`
   (a fifth string raises `ValueError`). When a callback is supplied, Rust invokes it for any
   known name, `"in-memory"` and `"streaming"` included (checked on 1.44.1), and an `Engine`
   object passed to `collect(engine=...)` bypasses the Python-side name check. So a third-party
   backend can run today by passing `"in-memory"` to Rust and reporting itself through
   `plan_engine`; what it cannot do is carry its own name through Rust, so `explain` and the
   callback's error message (`'cuda' conversion failed`) name the wrong engine. That naming
   is the one upstream change worth asking for; it is not a blocker.
2. Return a `PostOptCallback` from `_post_opt_callback`. It receives the `NodeTraverser` sitting
   on the optimised IR plus a second argument that is `None` in a plain `collect` (the
   type alias calls it `int | None`; it is a timing value, not a node id), and returns `None`
   -- it works by mutation.
3. Walk the IR with `get_node` / `set_node` / `get_inputs` / `view_current_node` /
   `view_expression`, translating each node it recognises. Every node it does not recognise is
   where the backend must decide between falling back (leave the node alone) and raising
   (`raise_on_fail`, which `GPUEngine` exposes as a config flag).
4. Replace the translated subtree with `set_udf(callable)`: the callable is what Polars will
   execute for that node, and it must return a Polars `DataFrame`. This is the seam through
   which tier 4 returns its results.
5. Handle the schema contract exactly: `get_schema()` and `get_dtype()` give the dtypes Polars
   will assume downstream, so a backend that widens an integer sum has to cast back. Polars does
   **not check** what the replacement returns: a frame with a wrong dtype is accepted and the wrong
   dtype propagates, so the backend has to assert its own output.

That translator is tier 4, below. `collect_gpu` stays the explicit form of the same idea: Polars
owns the plan, ArrowMetal owns one pass over the result.

---

## Tier 4 -- `MetalEngine`, a Polars engine

```python
import polars as pl
import arrowmetal as am

engine = am.MetalEngine()
df = lf.collect(engine=engine)      # the same frame lf.collect() returns
print(engine.last_report)           # which nodes ran on Metal, and why the rest did not
```

`MetalEngine` is `python/arrowmetal/polars_engine.py`. Polars optimises the plan as it always does
and hands the optimised IR to the engine's post-optimisation callback. The callback walks every
node and expression, translates the subtrees it can into an ArrowMetal plan (the grammar in
[ENGINE.md](ENGINE.md)), and replaces each one with a function that runs that plan on the GPU and
returns a Polars `DataFrame`. Everything it does not take stays with Polars' in-memory engine,
which also runs whatever sits above a replaced subtree. Every plan collects; the most that can
happen is that nothing moves, and then the answer is plain Polars'.

`import arrowmetal` still does not import Polars: `am.MetalEngine` loads the module on first touch,
like the other three tiers.

### How it plugs into Polars 1.44.1

Read from the installed package and checked by `python/tests/test_polars_engine.py`:

* `lf.collect(engine=<an Engine object>)` passes the object through unchanged, so no Polars change
  is needed. The name the engine gives Rust is `"in-memory"`: Rust accepts only its four engine
  names, and with a callback supplied it runs the callback for any of them; the in-memory engine is
  also what runs every node the callback leaves. `engine.name` is `"in-memory"`, `engine.plan_engine`
  and `repr(engine)` say `metal`.
* The callback receives the `NodeTraverser` and a second argument that is `None` under `collect`
  and an integer under `profile` (the time since the query started, which the engine uses to place
  its rows in the profile).
* `set_udf` turns the current node into a `PythonScan` whose function Polars calls as
  `f(with_columns, predicate, n_rows, should_time)`. That function takes no input, so **a replaced
  subtree is a leaf**: the only subtrees that can move are ones whose leaves are all in-memory
  frames (`DataFrameScan`). A file scan (`Scan`, `PythonScan`) stays with Polars, and so does
  everything above it.
* Polars does not check the replacement's output. The engine does: a frame whose schema is not
  the one `get_schema()` promised raises `ArrowMetalError` inside the query.
* An exception from the callback reaches the user as
  `ComputeError: 'cuda' conversion failed: <Type>: <message>`; the `'cuda'` is hardcoded in Polars.
  The engine's own messages start with `ArrowMetal MetalEngine:` so they read correctly inside it.
* `LazyFrame.profile(engine=...)` passes the callback only for a `GPUEngine`, so
  `lf.profile(engine=MetalEngine())` profiles plain Polars. `engine.profile(lf)` passes the callback
  through `profile`'s own keyword, and each replaced subtree appears as a `metal:<Node>#<id>` row.
* `collect_async` and `collect(background=True)` never run the callback; Polars runs the plan
  (background collection warns, as `GPUEngine` does). The `sink_*` family and `collect_batches` do
  run it, and Polars' streaming sink then panics on a replaced subtree ("entered unreachable code"),
  so the engine leaves any plan with a `Sink` node to Polars whole.
* The IR version the engine was written against, `(14, 7)`, is pinned by a test, as is every Polars
  surface it touches (`_LocalEngine`, `_post_opt_callback`, the `NodeTraverser` methods, the node
  classes), so a Polars upgrade that moves one fails a named test instead of changing an answer.
  A different IR major makes the engine leave the whole plan to Polars.

### What it translates

| Polars node | ArrowMetal | Taken when |
|---|---|---|
| `DataFrameScan` | `scan` | every column it reads (after Polars' projection pushdown) is an integer, float, Boolean, String, Date, Datetime, Duration or Time |
| `Filter` | `filter` | the predicate translates; Polars' `dynamic_pred` hints (which `sort().head()` inserts) are dropped |
| `Select`, `HStack` | kept virtual: each output is an s-expression over the physical columns, computed where it is used or in one `select` at the top | every output translates and is numeric or Boolean (a bare column of any carried type is carried) |
| `Select` whose every output is an aggregate | `aggregate` | each output is one of the aggregates in the last row of the expression table |
| `SimpleProjection` | no operator: a column list | always |
| `Slice` | `limit` | offset >= 0 (`tail` counts from the end and stays with Polars) |
| `Sort` | `sort`, with `limit` for a pushed-in slice | keys are columns, none of them String; no `maintain_order=True` together with a slice |
| `GroupBy` | `group_by` | keys are non-float columns; `maintain_order=False`; not rolling or dynamic |
| `Join` | `join` | inner, left, semi or anti; key columns of equal, non-float dtypes (String, multi-column and temporal keys included); `nulls_equal=False` (null keys never match, on both engines); `maintain_order="none"`; no pushed-in slice; the output names are the ones ArrowMetal's join gives (left columns, then right columns without a same-named key, the suffix on a collision), which covers Polars' coalescing defaults and `left_on`/`right_on` with different names |
| `Distinct` (`unique`) | `unique` | `keep="first"` or `"any"` (ArrowMetal keeps each group's first row, a valid `"any"`); `maintain_order=False`; no float column in the subset |
| everything else (`Union`, `HConcat`, `Cache`, `MapFunction`, `MergeSorted`, `ExtContext`, `Sink`, `Scan`, `PythonScan`, and right, full, cross and as-of joins) | -- | stays with Polars, named in the report |

| Expression | ArrowMetal |
|---|---|
| column, alias, typed literal (a Null literal takes the type it meets) | `(col ...)`, the literal at Polars' own dtype |
| `+ - *`, true division | `add sub mul div`, each operand cast to the result dtype Polars' `get_dtype` gives; a literal divisor as `mul` by its reciprocal, which is how Polars divides by a scalar (below); a float multiply by a scalar -1 as `negate`, as Polars does (below); Float32 goes through binary64 (below) |
| `== != < <= > >=` | `eq ne lt le gt ge`; floats in Polars' total order; String against a literal by `str_eq` (`==`, `!=` only) |
| `&`, `\|`, `^`, `~` | `and_kleene`, `or_kleene`, `ne` of the two as integers for Boolean xor, `bit_and`/`bit_or`/`bit_xor` on integers, `not`/`bit_not` |
| `when/then/otherwise` | `if_else`, a null condition taking the `otherwise` branch |
| `cast` | only casts that cannot fail or lose a value (integer widening, unsigned to a wider signed type, integers to Float64, 8/16-bit integers to Float32, Float32 to Float64, Boolean to numbers) |
| `is_null`, `is_not_null`, `fill_null` | `is_null`, `is_valid`, `fill_null` |
| `is_in` a literal list of 1 to 64 values | `is_in` (numbers) or `str_eq` terms (strings), null for a null input; a NaN in the list matches NaN rows, as in Polars' total order (`(ne x x)`) |
| `str.starts_with`, `str.contains(literal=True)` or a pattern without regex characters | `starts_with`, `contains` |
| `sum min max mean count len` | the plan's aggregates, with the fix-ups below (`min`/`max` of a Boolean stay with Polars; a per-group `count` of a Float64 or Boolean column is a sum of validity bits) |

Everything else -- `%`, `//`, `eq_missing`, `str.ends_with`, regex, windows (`over`), `rank`,
`median`, an expression over an aggregate, a String-valued output, a narrowing or fallible cast,
true division by a scalar that is not a plain literal,
a column name holding a NUL byte (Polars' own Arrow export panics on one) -- falls back, and the
report says which one. Categorical, Enum, Decimal, List, Struct, Null, Binary
and Object columns in a subtree's input keep the whole subtree on Polars.

### Where the answers would differ, and what the engine emits instead

Each line is a differential case in `test_polars_engine.py`, run against Polars itself.

* **Float comparisons.** Polars compares floats in a total order: NaN equals NaN and is greater than
  every number, and -0.0 equals 0.0. The engine adds the NaN terms (`(ne x x)` is "x is NaN") so the
  fused comparison gives Polars' answer, nulls included. `is_in` matches the same way: a NaN in the
  value list becomes an `(ne x x)` term, since ArrowMetal's `is_in` compares with IEEE equality.
* **Float32 arithmetic.** The GPU's float adds, multiplies and divides flush subnormals to zero,
  which Polars does not. The engine computes Float32 `+ - * /` in ArrowMetal's correctly rounded
  software binary64 and rounds once back to Float32, which is the correctly rounded Float32 result
  for these four operations, subnormals included. Float64 `+ - * /` is correctly rounded in both.
* **Division by a scalar.** Polars divides a column by a scalar as `x * (1 / c)`, with the
  reciprocal rounded in the result type; that differs from the correctly rounded `x / c` by at most
  one ulp, in a share of rows that depends on the divisor (about a third of Float64 rows for `/ 3.0`,
  none for a power of two). The engine emits the same multiply, so the bits match Polars'
  (`test_true_division_by_a_literal_is_polars_reciprocal_multiply`, which also checks zero, infinite,
  NaN, subnormal and null divisors). A column divisor is a true division in both.
* **Multiplying by -1.** Polars multiplies a float column by a scalar -1 (on either side, and divides
  by -1) as a negation, which flips a NaN's sign bit where a multiply keeps the input NaN. The engine
  emits ArrowMetal's `negate` there, so NaN rows carry Polars' bits too
  (`test_multiply_by_minus_one_is_a_negation_like_polars`, which compares the raw bits).
* **Aggregates.** A `sum` over no values is 0 in Polars (ArrowMetal: null) and gets a `fill_null`; a
  `min`/`max` over only NaN is NaN in Polars (ArrowMetal: null over a whole frame, an infinity per
  group), so the engine counts the non-null and non-NaN values and decides from the two; a `mean` of
  an Int64/UInt64 column is taken over the values cast to Float64, because ArrowMetal's integer mean
  sums in 64-bit integers and wraps on extreme values where Polars does not; every result is cast
  to Polars' dtype (UInt32 counts, the Int32 sum of an Int32 column, Float32 of a Float32, UInt32
  for the sum of a Boolean). A per-group `count` of a Float64 or Boolean column is the sum of its
  validity bits, because ArrowMetal's group-by will not read those values even to count them, and
  `min`/`max` of a Float64 column per group stays with Polars for the same reason.
* **Sort order.** ArrowMetal puts nulls last in both directions and a NaN after the numbers in a
  descending sort; Polars' default is nulls first and NaN above every number. The engine adds a
  validity key or a NaN key in front where it needs one.
* **Integer overflow** wraps in both (checked on Int8 and Int64 extremes), and an integer true
  division by zero is IEEE in both.

### ArrowMetal behaviours this suite found, now fixed in the engine

Each was first worked around here and pinned by a strict `xfail`; each is now fixed in ArrowMetal,
its test in `test_polars_engine.py` passes, and the workaround is gone
(`test_engine_takes_the_plans_it_once_worked_around` runs the plans the engine once changed or
declined):

1. **A null String slot with bytes under it.** Polars exports a null String value with the bytes the
   slot held (valid Arrow), and ArrowMetal's String gather copied those bytes over the next kept
   value (`test_core_string_filter_null_slot`). The engine now hands such a column over as exported.
2. **A Boolean column through the plan's sort** came back with the right validity bitmap and a null
   count of 0 (`test_core_bool_sort_null_count`). Result columns are now used as ArrowMetal returns
   them.
3. **A filter over a scan that carries a `date32` column** was rejected by the expression compiler
   (`test_core_filter_carrying_a_date`). Temporal columns now go to ArrowMetal as their own types.
4. **Sorting by a String column that holds a null and a value of 8 bytes or more** returned wrong
   rows (`test_core_string_sort_with_nulls`, run in a child process because one run ended in a bus
   error). The engine now sorts by String columns.
5. **A finite float literal of magnitude 2^63 or more** trapped the process inside the expression
   compiler, which converted every float literal to Int64 as well. The engine now runs those plans
   on Metal (`test_a_float_literal_of_magnitude_2_63_or_more_runs_on_metal`).

### Which translatable subtrees it runs: the defaults

A subtree the engine can translate still has to be one where the GPU is ahead, because getting a
Polars column onto the GPU is not free: a single-chunk numeric column is imported without a copy,
but mapping its pages into Metal and releasing them costs time on every query, and a String column
is converted on the CPU. `Benchmarks/polars_engine_bench.py` measures the eight shapes of
`Benchmarks/engine_bench.py` plus ten group-by, sort and `unique` shapes, as Polars LazyFrames, through
Polars' in-memory engine, Polars' streaming engine and the engine with everything it can translate
(`shapes="all"`), cold (nothing imported before) and warm (see the import cache below). The run
behind the defaults is `Benchmarks/results/polars_engine_bench_2026-09-23_provisional.csv`; it ran
while other work shared the machine, so its numbers are provisional and not quoted here.

Read cold, against the faster of Polars' two engines, that run says (sizes near a crossover move
between runs on a shared machine, which is why the defaults keep a margin):

* **A full sort** whose keys ArrowMetal orders as Polars does (no helper key): both such shapes were
  ahead at every size measured, from 500,000 rows. (An earlier provisional run had one of them
  behind at 500,000.)
* **A full sort that needs a helper key** (nulls first on a nullable key, or a float key descending)
  was behind at 1,000,000 rows and below, level at 2,000,000, and ahead from 10,000,000.
* **A sort that carries a String column** was behind at every size: the String import is a CPU copy.
* **Group-by** depends on what nobody knows before running it, the number of groups, and on the key
  types: some shapes were ahead from 500,000 rows, others behind at every size below 50,000,000 and
  level with Polars there.
* **Top-k, whole-frame aggregates and row-wise filters and projections** were behind the faster
  Polars engine at every size.
* **Joins and `unique`**: an inner join feeding an aggregate, and a `unique` over a key with ten
  thousand distinct values, were ahead at every size measured; a semi join returning most of its
  probe rows was behind at every size. Like group-by, both run on the key-to-id machine, whose speed
  against Polars depends on how many distinct keys there are, and one shape of each is not enough to
  set a default by.
* **Windows and the as-of join** stay with Polars in this version; their rows record what the
  callback costs a plan it leaves alone.

So `MetalEngine()` takes a subtree when every shape in it is a full sort (`MEASURED_SHAPES`), none of
its inputs is a String column, and its in-memory inputs hold at least 1,000,000 rows -- 10,000,000
when a helper key is needed. 1,000,000 is the sort family's crossover against the fastest CPU
library in `Benchmarks/results/router_2026-09-17.json` (`sort float64`, `argsort int64`, `lexsort (2
int32 keys)`); the provisional run had full sorts ahead below it as well, so the default keeps the
router's figure as its margin. Every other subtree stays with Polars with the reason in the report.

```python
am.MetalEngine()                          # the defaults above
am.MetalEngine(shapes="all")              # every translatable subtree of at least min_rows rows
am.MetalEngine(shapes="all", min_rows=0)  # everything it can translate (what the tests use)
am.MetalEngine(raise_on_fail=True)        # raise instead of leaving anything to Polars
```

`shapes="all"` is there for plans the benchmark did not cover and for moving work off the CPU cores:
the CPU-ms column of the results file is the process CPU time of each run.

### The import cache

Each Polars column the engine imports is kept, keyed by its Arrow type, length, offset and the
address and size of every buffer, and the next query that reads the same column reuses the import.
That is safe because the cached import holds the exported array and, through it, Polars' own buffer:
while an entry lives, Polars can neither free that memory (so no other column can appear at the same
address) nor write to it in place (Polars copies a buffer it shares before writing). Only a column
that was imported without a copy is cached; a String column is converted on every export, and a
multi-chunk column concatenated on every export, so neither is. Entries are evicted least recently
used above a byte budget, a quarter of physical memory by default:

```python
from arrowmetal import polars_engine
polars_engine.import_cache_info()       # {"entries", "bytes", "limit", "hits", "misses"}
polars_engine.import_cache_limit(2 << 30)
polars_engine.clear_import_cache()      # and with it the Polars buffers it kept alive
```

The "warm" column of the results file is a second collect over the same frame; the defaults above
were read from the cold one.

### The report

```
MetalEngine report (polars 1.44.1, IR (14, 7))
  metal:  Sort#3 [Sort > HStack > Filter > DataFrameScan] over 10,000,000 rows, ran in <t> ms -> <n> rows
  polars: Select#1: function rank has no ArrowMetal translation
```

`engine.last_report` is a `MetalPlanReport`: `taken` (one entry per subtree that ran on Metal: the
node at its top, the node kinds inside it, the rows it read, the ArrowMetal plan text, and after the
run its wall time and output rows), `fallbacks` (one `Kind#id: reason` line per node that stayed
with Polars for a reason of its own), `walked` (every node of the optimised plan) and `nodes`. With
`POLARS_VERBOSE=1` the fallback lines are also issued as a `PerformanceWarning`. The report belongs to
the engine object, so two threads collecting through one `MetalEngine` overwrite each other's.

### Tests

```
PYTHONPATH=python python -m pytest python/tests/test_polars_engine.py -q
DIFF_QUICK=1 PYTHONPATH=python python -m pytest python/tests/test_polars_engine.py -q   # without the 100,003-row datasets
```

The differential cases collect each LazyFrame on Polars and through
`MetalEngine(raise_on_fail=True, min_rows=0, shapes="all")` -- so nothing may fall back -- and
compare the frames: schema first, then values, exactly for integers, Booleans, Strings, nulls and
element-wise floats, with a relative tolerance for float aggregates. The inputs are
`test_differential.py`'s generators at 0, 1, 33, 4,097 and 100,003 rows, null ratios 0, 0.3 and 1,
its "sliced" and "special" (extremes, NaN, infinities, subnormals) flavours, and two Polars-side
shapes, a two-chunk frame and a frame sliced with `DataFrame.slice`. Every numeric dtype runs
through the comparisons, arithmetic, null logic, `is_in`, casts, group-by and whole-frame aggregates,
sorts in both directions with nulls at both ends, and top-k; Boolean logic, String predicates and
carried temporal columns have their own cases, and so do the four join kinds on integer, String and
two-column keys with null keys and duplicates on both sides (a two-chunk right side among the
shapes), suffix collisions, different key names, a join inside a filter-join-aggregate plan, and
`unique` with and without a subset. Each fallback in the tables above has a case that
checks the result is still Polars' and the report names the reason. One case reruns
`test_polars.py` and `test_lazy.py` with every `LazyFrame.collect()` also collected through the engine
(`python/tests/metal_engine_everywhere.py`) and requires the two to agree.

### A checklist to run by hand

The plan behind this tier asks for these on real hardware before anything from it goes upstream:

0. `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift build -c release --product ArrowMetalC`,
   then `PYTHONPATH=python python -m pytest python/tests/test_polars.py python/tests/test_lazy.py -q`:
   tiers 1 to 3 still green.
1. `PYTHONPATH=python python -m pytest python/tests/test_polars_engine.py -q`, after that fresh build.
2. In a REPL: `import sys, arrowmetal as am` and check `"polars" not in sys.modules`; then collect one
   plan through `am.MetalEngine(shapes="all", min_rows=0)` and read `engine.last_report`.
3. With `POLARS_VERBOSE=1`, collect a plan with an unsupported node (`pl.col("v").rank()`) and see the
   `PerformanceWarning` listing it.
4. `am.zero_copy_report(df["v"])` on a single-chunk numeric column of a million rows says the two
   addresses are the same.
5. `out, timings = engine.profile(lf)` shows a `metal:` row.
6. Run `PYTHONPATH=python python Benchmarks/polars_engine_bench.py --sizes 1000000,10000000,50000000
   --out a.csv` twice on a quiet machine, compare the two files, and read the `taken` column of the
   eight engine_bench shapes against the defaults above.
7. Read this section against `engine.last_report` on your machine.

---

## Numbers

These are tiers 1 and 2; tier 4's measurements are in
`Benchmarks/results/polars_engine_bench_2026-09-23_provisional.csv` (see "Tier 4" above).

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
| Tier 4 | You want Polars' own `collect()` and its answers, with the parts of the plan the GPU is measured ahead on (by default, large sorts of in-memory frames) run there. |

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

**Version coupling.** Tier 2: the plugin is pinned to polars 0.55.1 / pyo3-polars 0.28 for
py-polars 1.44.x. Tiers 1 and 3 speak the C Data Interface and are not coupled to a Polars
version. Tier 4 is pure Python but reads Polars' optimised IR through an API Polars calls unstable;
it was written against IR version (14, 7) of polars 1.44.1, a test pins both, and a different IR
major makes it leave every plan to Polars.

---

## Tests

```
PYTHONPATH=python python -m pytest python/tests/test_polars.py -q     # 90 tests
cd polars-plugin/arrowmetal-sys && cargo test --release               # 10 tests
PYTHONPATH=python python -m pytest python/tests/test_polars_engine.py -q   # tier 4
```

`test_polars_engine.py` is described under "Tier 4", "Tests".

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
