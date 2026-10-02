# ArrowMetal for Polars users

Polars is the reason most people on an Apple silicon Mac have Arrow-shaped data in memory at all.
This document is how you point that data at the GPU.

There are four tiers, the first three in 0.1.0 and the engine (tier 4) in 0.2.0, and they differ in **where the GPU sits
relative to the Polars plan**:

| Tier | Where the GPU runs | What you write | Needs |
|---|---|---|---|
| 1. Bridge and namespaces | Around Polars: you hand a collected frame over | `df.arrowmetal.group_by("k").sum("v")` | Python only |
| 2. Expression plugin | Inside a Polars lazy plan | `pl.col("v").arrowmetal.sum()` | Python only (the wheel carries the plugin) |
| 3. Streaming hand-off | Polars runs the plan, ArrowMetal finishes it | `lf.arrowmetal.collect_gpu(q)` | Python only |
| 4. `MetalEngine` | In place of whole subtrees of the optimised Polars plan | `lf.collect(engine=am.MetalEngine())` | Python only |

All four move data over the Arrow C Data Interface. For a single-chunk numeric Polars column that
is **no copy at all** -- the GPU reads the buffer Polars already owns. A String column crosses the
same way in tiers 1, 3 and 4: in Polars' own `Utf8View` layout, which the string kernels read
directly (tier 2's plugin still asks Polars for `large_string`). Categoricals and multi-chunk Series
each cost one conversion pass -- see Limits. The evidence is below.

---

## Install

### From the wheel

```bash
scripts/build_wheel.sh                                   # swift build, cargo build, then the wheel
pip install python/dist/arrowmetal-*.whl polars
```

A wheel built from this tree carries both native libraries in `arrowmetal/_lib/`:
`libArrowMetalC.dylib` (the GPU library) and `libarrowmetal_polars.dylib` (the tier-2 expression
plugin, built by cargo during the wheel build, against that same `libArrowMetalC.dylib`). All four
tiers work from that install, with no Xcode, no cargo and no `DYLD_LIBRARY_PATH`. The packaged plugin
has one rpath, `@loader_path`, so its `@rpath/libArrowMetalC.dylib` resolves to the copy beside it,
the same file Python loads. The 0.2.0 wheel on PyPI carries `libArrowMetalC.dylib` only; with it,
tier 2 needs the cargo build below.

`scripts/check_wheel.sh` checks a built wheel: it installs the wheel with its `polars` extra into a
fresh virtualenv outside the repository, with a scrubbed environment and no cargo on `PATH`, runs one
expression or plan per tier, checks that the process loaded exactly one `libArrowMetalC.dylib`, the
packaged one, and runs `python -m arrowmetal.bench` with and without `--parquet`. On an M4 Max
(macOS 26.6.2, Homebrew Python 3.13.9) it passed with polars 1.44.2 and pyarrow 25.0.1 as pip
resolved them, NumPy not installed: the packaged plugin, built against
polars 0.55 crates, loads in py-polars 1.44.2. The plugin adds 5.0 MB to the compressed wheel
(3.2 MB before, 8.2 MB after) and is 21 MB on disk after `strip -x`.

### Which Polars

`pip install 'arrowmetal[polars]'` installs `polars>=1.44,<1.45`, the range every tier works in.
Per tier:

| Tier | Polars |
|---|---|
| 1. Bridge and namespaces | any `polars>=1.0` (pure Python over the Arrow C Data Interface) |
| 3. Streaming hand-off | any `polars>=1.0` (pure Python over the Arrow C Data Interface) |
| 2. Expression plugin | 1.44.x: the plugin is built on the polars 0.55 crates, and Polars refuses a plugin built for another minor's ABI (Version pinning, below) |
| 4. `MetalEngine` | tested on 1.44.1 and 1.44.2 (`TESTED_POLARS`: the full engine suite passes and the capability table is the same on both; 1.44.2 also by `scripts/check_wheel.sh`); it walks Polars' unstable IR, checked against `TESTED_IR_VERSION` (14, 7). `python -m arrowmetal.polars_engine check` reports the installed Polars against these |

A Polars outside 1.44 installed without the extra keeps tiers 1 and 3.

### From source

```bash
# 1. The GPU library (every tier needs it)
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

`plugin_path()` looks, first hit wins, at `$ARROWMETAL_POLARS_PLUGIN` (the full path to a plugin
dylib); the packaged `arrowmetal/_lib/libarrowmetal_polars.dylib` when Python loaded the packaged
`libArrowMetalC.dylib` beside it, which is what a wheel install has; `polars-plugin/target/release/`
and `target/debug/` in a source checkout; the packaged copy in any other case; and a maturin install
on `sys.path`. When `$ARROWMETAL_LIB` pins a development build, a cargo build linked against it
therefore wins over a packaged plugin left in `_lib/` by a wheel build in the same checkout.

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

The bridge does not call the C ABI's `am_join` hash join; `df.arrowmetal.join` is built out of
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
uses natively. That conversion is the plugin's only copy, and it is why the plugin's string row
below is 2.3x where tier 1's, which takes the column as views, is 17.6x. Tiers 1, 3 and 4 hand
String columns over as `Utf8View`, which the kernels read directly (the Strings paragraphs under
Numbers and Limits).

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

So a Metal backend is **not** blocked on Polars adding an API -- the API is there, and tier 4 below
is built on exactly this surface. One fact about names: the engine's `name` is passed to Rust as
`ldf.collect(self.name, callback)`, and Rust only knows the four in `SUPPORTED_ENGINE_NAMES` (a fifth
string raises `ValueError`). When a callback is supplied, Rust invokes it for any known name,
`"in-memory"` and `"streaming"` included (checked on 1.44.1), and an `Engine` object passed to
`collect(engine=...)` bypasses the Python-side name check. So a third-party engine runs by passing
`"in-memory"` to Rust and reporting itself through `plan_engine`; what it cannot do is carry its own
name through Rust, so `explain` and the callback's error message (`'cuda' conversion failed`) name
the wrong engine.

`collect_gpu` stays the explicit form of the same idea: Polars
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
  subtree is a leaf**: the only subtrees that can move are ones whose leaves are in-memory frames
  (`DataFrameScan`) or Parquet files the engine reads itself (`Scan`, below). Any other scan
  (`PythonScan`, a CSV or IPC `Scan`) stays with Polars, and so does everything above it.
* `view_current_node` raises `NotImplementedError: ipc scan` for a `scan_ipc` node. The engine
  leaves such a node, and everything above it, to Polars and names it in the report.
* Polars does not check the replacement's output. The engine does: a frame whose schema is not
  the one `get_schema()` promised raises `ArrowMetalError` inside the query.
* An exception from the callback reaches the user as
  `ComputeError: 'cuda' conversion failed: <Type>: <message>`; the `'cuda'` is hardcoded in Polars.
  The engine's own messages start with `ArrowMetal MetalEngine:` so they read correctly inside it,
  name the node (`Sort#3`) and the reason, and end with the way to run the plan on Polars instead
  (`FORCE_POLARS`: collect without `engine=`, or set `ARROWMETAL_METAL_ENGINE=off`).
* `LazyFrame.profile(engine=...)` passes the callback only for a `GPUEngine`, so
  `lf.profile(engine=MetalEngine())` profiles plain Polars. `engine.profile(lf)` passes the callback
  through `profile`'s own keyword, and each replaced subtree appears as a `metal:<Node>#<id>` row.
* Only `collect` and the paths built on it run the callback; the next section lists every path and
  what the engine does on it.
* The IR version the engine was written against, `(14, 7)`, is pinned by a test, as is every Polars
  surface it touches (`_LocalEngine`, `_post_opt_callback`, the `NodeTraverser` methods, the node
  classes), so a Polars upgrade that moves one fails a named test instead of changing an answer.
  A different IR major makes the engine leave the whole plan to Polars. So does a node kind outside
  `KNOWN_NODE_KINDS` (the 20 node classes of polars 1.44.1) or a node Polars fails to show to the
  engine: the report line is `The plan holds an unknown node <kind> in polars <version>, so the
  whole plan stays with Polars.`, and the query is not an error.
* `ARROWMETAL_METAL_ENGINE=off` (or `0`, `false`, `no`, `polars`) makes every `MetalEngine` leave every
  plan to Polars, `raise_on_fail=True` included; the report says so.

### Collect paths

What each Polars 1.44.1 entry point does with a `MetalEngine` (`eng` below), read from the installed
package and checked by `test_every_collect_path_is_explicit` and
`test_polars_side_entry_points_that_never_call_the_engine`. `last_report.path` names the path; a
path on which the whole plan runs on Polars says why in `last_report.fallbacks` and issues a
`MetalEngineFallbackWarning` (a `UserWarning`) once per process.

| Entry point | Runs on | `last_report.path` | Warning |
|---|---|---|---|
| `lf.collect(engine=eng)` | Metal, the subtrees the engine takes | `collect` | -- |
| `lf.head(n).collect(engine=eng)`, and `lf.fetch(n, engine=eng)` (deprecated; it is `head(n).collect`) | Metal | `collect` | -- |
| `lf.collect()` or `df.lazy().collect()` under `pl.Config(engine_affinity=eng)` or `pl.Config.set_engine_affinity(eng)` | Metal | `collect` | -- |
| `lf.collect(engine=eng)` inside any other `pl.Config` context (`tbl_rows`, `engine_affinity="streaming"`, ...) | Metal | `collect` | -- |
| `pl.collect_all(lfs, engine=eng)` | Metal, frame by frame: each frame is optimised and collected on its own, where Polars' `collect_all` optimises the frames together | `collect_all`; `last_reports` holds one report per frame | -- |
| `eng.profile(lf)` | Metal, with a `metal:<Node>#<id>` row per replaced subtree | `profile` | -- (Polars' own `DeprecationWarning` for `profile`) |
| `eng.explain(lf)` | nothing runs: Polars' optimised plan followed by the report of what would run on Metal (each subtree it would take is still checked over a 64-row prefix) | `explain` | -- |
| `lf.collect_async(engine=eng)` | Polars' in-memory engine: Polars passes no engine callback on this path | `collect_async` | once |
| `pl.collect_all_async(lfs, engine=eng)` | Polars' in-memory engine, for the same reason | `collect_all_async` | once |
| `lf.collect_batches(engine=eng)` | Polars, for the same reason | `collect_batches` | once |
| `lf.collect(engine=eng, background=True)` | Polars' in-memory engine (background collection is not supported, as for `GPUEngine`) | `background` | once |
| `lf.sink_parquet`, `sink_ipc`, `sink_csv`, `sink_ndjson`, `sink_batches` with `engine=eng`, and a `lazy=True` sink collected through `eng` | Polars: its streaming sink panics on a replaced subtree ("entered unreachable code"), so a plan with a `Sink` node stays whole | `sink` | once |
| `lf.explain(engine=eng)` | nothing runs: Polars' plan, the same text as `lf.explain()`; Polars reads only `eng.plan_engine` | not written | none: Polars does not call the engine |
| `lf.profile(engine=eng)` | Polars: polars 1.44.1 passes the callback only to a `GPUEngine` | not written | none: Polars does not call the engine |
| Eager `DataFrame` methods (`df.sort`, `df.group_by(...).agg`, ...) under an engine affinity of `eng` | Polars' in-memory engine, which Polars uses for every eager operation | not written | none: Polars does not call the engine |
| `lf.collect(engine="in-memory")` (or `"streaming"`) under an engine affinity of `eng` | Polars: an explicit engine name wins over the affinity | not written | none: Polars does not call the engine |

`collect(background=False)` with Polars' eager optimisation flag set asks for no callback either;
the report says `eager` and there is no warning, as for `GPUEngine`. For code that cannot change
its `collect` calls, `ARROWMETAL_METAL_ENGINE=off` sends every path to Polars.

### What it translates

[ENGINE_CAPABILITIES.md](ENGINE_CAPABILITIES.md) is the same boundary measured: 29 plan shapes
(filters, projections, `with_columns`, `slice`, sorts, top-k, group-by with `x` as the key and as the
value of each aggregate, whole-frame aggregates, the four joins, `unique`, and Parquet scans) over 29
input dtypes and three null patterns, each run through the engine and marked `Metal` or `Polars`
with the engine's reason. It is generated by `python/tests/engine_capabilities.py`, and
`test_the_capability_table_is_current` fails when the committed file differs from a fresh run.

| Polars node | ArrowMetal | Taken when |
|---|---|---|
| `DataFrameScan` | `scan` | every column it reads (after Polars' projection pushdown) is an integer, float, Boolean, String, Date, Datetime, Duration or Time |
| `Scan` of Parquet | the file read on the GPU (`am.read_parquet`, through the open-file cache), then `scan` over the columns it returned, with the scan's predicate as a `filter` | one local file (a list or a glob that resolves to one file counts); every column it reads (Polars' projection is the reader's column list) is one of the `DataFrameScan` types above and a top-level column of the file; no hive partitions, `row_index_name`, `n_rows` (a `head`, `tail` or `slice` Polars pushed into the scan), `include_file_paths`, `schema=`, deletion files or column mapping; something above it does GPU work. See "Parquet scans" below |
| `Filter` | `filter` | the predicate translates; Polars' `dynamic_pred` hints (which `sort().head()` inserts) are dropped |
| `Select`, `HStack` | kept virtual: each output is an s-expression over the physical columns, computed where it is used or in one `select` at the top | every output translates and is numeric or Boolean (a bare column of any carried type is carried) |
| `Select` whose every output is an aggregate | `aggregate` | each output is one of the aggregates in the last row of the expression table |
| `SimpleProjection` | no operator: a column list | always |
| `Slice` | `limit` | offset >= 0 (`tail` counts from the end and stays with Polars) |
| `Sort` | `sort`, with `limit` for a pushed-in slice (`sort().head()`, `top_k`, `bottom_k`) | keys are columns of any carried dtype, each one plan key with Polars' null placement and float order as its options; no `maintain_order=True` together with a slice |
| `GroupBy` | `group_by` | keys are non-float columns; `maintain_order=False`; not rolling or dynamic |
| `Join` | `join` | inner, left, semi or anti; key columns of equal, non-float dtypes (String, multi-column and temporal keys included); `nulls_equal=False` (null keys never match, on both engines); `maintain_order="none"`; no pushed-in slice; the output names are the ones ArrowMetal's join gives (left columns, then right columns without a same-named key, the suffix on a collision), which covers Polars' coalescing defaults and `left_on`/`right_on` with different names |
| `Distinct` (`unique`) | `unique` | `keep="first"` or `"any"` (ArrowMetal keeps each group's first row, a valid `"any"`); `maintain_order=False`; no float column in the subset |
| everything else (`Union`, `HConcat`, `Cache`, `MapFunction`, `MergeSorted`, `ExtContext`, `Sink`, `PythonScan`, a CSV, IPC or NDJSON `Scan`, and right, full, cross and as-of joins) | -- | stays with Polars, named in the report |

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

Each line is a differential case in `test_polars_engine.py` or `test_engine_conformance.py`, run
against Polars itself.

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
  NaN, subnormal and null divisors). A column divisor is a true division in both. Over a column of
  **one row** Polars divides element-wise instead (the scalar and the column have the same length),
  so there the answer is the correctly rounded `x / c`. The engine emits whichever of the two Polars
  computes for the row count of the node's input: from the in-memory frame when nothing between it
  and the division changes the row count by an unknown amount, and otherwise by counting that input
  when the plan runs, both forms in the plan and the count choosing between them
  (`test_scalar_division_and_minus_one_follow_polars_at_every_length`).
* **Multiplying by -1.** Polars multiplies a float column by a scalar -1 (on either side, and divides
  by -1) as a negation, which flips a NaN's sign bit where a multiply keeps the input NaN. The engine
  emits ArrowMetal's `negate` there, so NaN rows carry Polars' bits too
  (`test_multiply_by_minus_one_is_a_negation_like_polars`, which compares the raw bits). Over a
  column of one row Polars multiplies, and the NaN keeps its sign; the engine chooses by the row
  count as for a division.
* **Aggregates.** A `sum` over no values is 0 in Polars (ArrowMetal: null) and gets a `fill_null`; a
  `min`/`max` over only NaN is NaN in Polars (ArrowMetal: null over a whole frame, an infinity per
  group), so the engine counts the non-null and non-NaN values and decides from the two; a `mean` of
  an Int64/UInt64 column is taken over the values cast to Float64, because ArrowMetal's integer mean
  sums in 64-bit integers and wraps on extreme values where Polars does not; every result is cast
  to Polars' dtype (UInt32 counts, the Int32 sum of an Int32 column, Float32 of a Float32, UInt32
  for the sum of a Boolean). A per-group `count` of a Float64 or Boolean column is the sum of its
  validity bits, because ArrowMetal's group-by will not read those values even to count them, and
  `min`/`max` of a Float64 column per group stays with Polars for the same reason. Polars' `min`
  and `max` order -0.0 below 0.0, so a `min` over both zeros is -0.0 and a `max` 0.0; ArrowMetal
  treats the two as equal and returns whichever it met first over a whole frame, and 0.0 per group.
  The engine counts the zeros of the sign Polars prefers and takes the sign from that count
  (`test_min_and_max_over_both_zeros_are_polars_signed_zeros`).
* **Float sums and means.** The one place the answers differ, and the engine leaves it: a Float32
  or Float64 `sum`, and a Float64 `mean`, add the same values in the GPU's order where Polars adds in
  its own, so the two can differ in the last bits. Both are sums of the same values, so they differ
  by at most twice the rounding bound of a sum, 2(n - 1) u sum(|x|) for a sum of n values and
  2 u sum(|x|) for a mean (u the unit roundoff of the type the sum accumulates in: 2^-24 for a
  Float32 sum, 2^-53 otherwise), and the engine conformance grid holds each such case to that bound.
  In the run recorded in `Benchmarks/results/engine_conformance_2026-10-02_report.txt` (totals in
  `engine_conformance_2026-10-02.csv`) the largest difference was 0.199 u sum(|x|) for a Float32 sum
  (3.55e-5 of the answer), 0.372 u sum(|x|) for a Float64 sum (2.00e-14 of the answer) and
  0.0033 u sum(|x|) for a Float64 mean (7.05e-14 of the answer). A `mean` of an integer or Float32
  column is computed in Float64 by both (the engine casts the column, as Polars does); in that run
  it matched bit for bit in every case but one int64 group mean over the integer extremes (1,000
  rows), where the engine returns the correctly rounded mean of the cast values and the two differ
  by 8.76e-8 u sum(|x|), as much as the answer itself. Every other output of the grid is compared
  bit for bit. `test_polars_engine.py` compares these
  aggregates to a relative tolerance.
* **Sort order.** Polars (1.44 and the 2.0 release candidate alike, pinned by
  `test_polars_sort_order.py`) places the nulls per key, first unless `nulls_last`, in either
  direction; orders a float key with every NaN (of either sign) one value above every number, +inf
  included, in both directions, and -0.0 equal to 0.0; keeps tied rows in input order in both
  directions under `maintain_order=True`, a tie falling through to the next key; orders String keys
  byte-wise and Boolean keys False first. `top_k` is a descending sort with `nulls_last=True` and a
  slice, `bottom_k` the ascending one. The engine writes each Polars key as one ArrowMetal key whose
  options give that order: `"nulls": "first"` where the nulls go first, and `"float_order":
  "nan_largest"` on a float key (docs/ENGINE.md, "Sort key options"). No key column is added, and a
  single-key sort with a slice stays a GPU top-k.
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
6. **A UInt64 literal above 2^63 - 1** (found by the conformance grid below, not worked around first: `is_in`, a comparison or `fill_null` against a large
   unsigned value) was rejected by the expression parser, which read every integer literal as an
   Int64, and the plan stayed with Polars. The parser now keeps such a literal as its bit pattern,
   the way the code generator writes unsigned literals (`test_a_u64_literal_above_int64_max_reaches_the_gpu`,
   and `testWideUnsignedLiteralsParsePrintAndStayUnfolded` in Swift).

### Which translatable subtrees it runs: the defaults

A subtree the engine can translate still has to be one where the GPU is ahead, because getting a
Polars column onto the GPU is not free: a single-chunk numeric column is imported without a copy,
but mapping its pages into Metal and releasing them costs time on every query. String columns are
handed over in Polars' own view layout (see "Strings" under Limits), in the crossover sweep and in
the benchmark of the defaults below alike.
`MetalEngine()` (`shapes="measured"`) decides per subtree from measured crossovers (`python/arrowmetal/_engine_policy.py`). Each translated subtree has shape classes:

* `rowwise` -- filters and projections only;
* `aggregate:<family>` (a whole-frame aggregate), `group_by:<family>` (one key) and
  `group_by_multi:<family>` (two or more keys), the families being `sum`, `count` (`count` and
  `len`), `mean` and `minmax`;
* `sort` (a full sort) and `top_k` (a sort with a slice);
* `join:inner`, `join:left`, `join:semi`, `join:anti`, and `distinct` (`unique`).

It also has a dtype class, `string` when a String column is among the columns it reads and `numeric`
otherwise, and an input, in-memory frames or a Parquet file. A subtree runs on Metal when its input
rows (the rows of its in-memory frames, or of its Parquet file as the footer states them; what a
predicate will keep is not estimated) are at or above the crossover of every class in it for its
dtype class and input; a group-by is judged at the bucket of its estimated number of groups
("Group-bys: the number of groups", below). A class whose row says **not taken** stays with Polars at
every size. The decision reads nothing but the subtree's classes, dtypes, input rows and input, a
group-by's estimate and the tables, and the estimate comes from a fixed-seed sample, so the same
subtree over the same frame always gets the same answer (`python/tests/test_engine_policy.py` checks
both across 200 calls and across processes). Where the policy leaves a node, the placement moves down to its inputs, so a
smaller subtree below it can still be taken: under a whole-frame sum over an inner join, the join
runs on Metal and Polars adds up its output.

**The crossovers.** `Benchmarks/polars_engine_bench.py --crossover` ran 93 LazyFrames over in-memory
frames, the 45 cases below and a group-by grid of 48 (each aggregate family over one int32 key and
over two, the keys drawn from 200, 1,000, 10,000, 100,000 or 1,000,000 values or from half the
frame's rows, `(g1sum200)` to `(g2minmaxR2)`), at 250,000, 500,000, 1,000,000, 2,000,000, 5,000,000,
10,000,000, 20,000,000 and 50,000,000 rows (the probe side of the joins grows with the size, the
build side is 1,000,000 rows), and the four Parquet scan cases over files of 1,000,000, 2,000,000,
5,000,000, 10,000,000, 20,000,000 and 50,000,000 rows, snappy and uncompressed, each through Polars'
in-memory and streaming engines and through `MetalEngine(shapes="all", min_rows=0)` cold, best of 7,
every MetalEngine result equal to Polars': `Benchmarks/results/polars_engine_crossover_2026-09-26-final.csv`,
run conditions (the load average, sampled each minute) in
`Benchmarks/results/bench_conditions_2026-09-26-final.txt`. Each group-by row of the results file also
records the number of groups the case's data holds at that size (`groups`). The in-memory sort and
top-k cases ((b), (m), (n), (o), (p), (q), (x2), (x3), (y6), and (x4), a top 100 by a nullable Float64
key descending, added then) were swept again on 2026-09-29, after each Polars sort key became one plan
key with Polars' null placement and float order as its options (see "Sort order" above), at the same
eight sizes, best of 7: `Benchmarks/results/polars_engine_crossover_2026-09-29-sort.csv` holds those rows
and the 2026-09-26 rows of every other case, run conditions in
`Benchmarks/results/polars_engine_crossover_2026-09-29-sort_conditions.txt`. The group-by cases that run
the grouped Float64 sum and mean or the grouped min/max ((c), (l), (t3), (t4), (t6), (t7), (v3), (v4)
and the grid's mean and min + max cases; a mean of an int64 column runs as a Float64 mean, the engine
casting it first) were swept again on 2026-09-30, after both were rebuilt, at the same eight sizes,
best of 7, and the grid gained a Float64 sum over each of its key sets and group counts
(`(g1fsum200)` to `(g2fsumR2)`, a sum of q / 1e9): `Benchmarks/results/polars_engine_crossover_2026-09-30-groupby.csv`
holds those rows and the earlier rows of every other case, run conditions in
`Benchmarks/results/polars_engine_crossover_2026-09-30-groupby_conditions.txt`, and the table is fitted
from it.
`Benchmarks/polars_engine_crossover.py` fits it the way `Benchmarks/router_table.py` fits the router
table (`python/arrowmetal/_router_fit.py`), with the MetalEngine as the GPU side and the faster Polars
engine as the CPU side, over each case's input rows. A case's fit is the first size from which it is
ahead at every larger size, placed between that size and the one below it where the two straight
lines meet; a case ahead at the largest size alone has none, and a size where the MetalEngine's answer
differed from Polars' counts as behind.

**Four rules on the fit.** The table keeps each case's and each bucket's fit as `fit`, and the rows
the default uses as `rows`:

* **The margin** (`MARGIN`). A case is ahead at a size when its MetalEngine time, raised by a margin,
  is at most the faster Polars engine's time: 15% for a numeric shape, 35% for a shape with a String
  column. A String shape's advantage grows slowly with size: the String sort (o) is 0.98x the faster
  Polars engine at 1,000,000 rows in the 2026-09-26 sweep (`Benchmarks/results/polars_engine_crossover_2026-09-26-final.csv`,
  `vs_fastest_polars`), 1.2x at 2,000,000, 1.37x at 5,000,000 and 1.5x at
  10,000,000, so a fit at 15% would land where the two engines are within noise of each other.
* **The String floor** (`STRING_FLOOR`). A shape with a String column is taken from 5,000,000 rows
  at the earliest, whatever its fit says. Below that the String sorts are close to Polars: (o) 1.2x
  and (p) 1.38x at 2,000,000 rows in the 2026-09-26 sweep, and 1.21x and 1.31x under `shapes="all"` at that size
  in the benchmark below (`Benchmarks/results/polars_engine_bench_2026-09-26-final3.csv`, cold), inside its noise band
  (0.62x to 1.31x); from 5,000,000 rows that sweep has them at 1.37x and up.
* **The headroom** (`HEADROOM`). The default takes a shape from 1.5 times its fit (a String shape
  from the larger of that and the floor), and a shape whose 1.5 times passes the largest size
  measured is not taken. The fit interpolates between sizes measured 2 to 2.5 times apart
  (250,000, 500,000, 1,000,000, 2,000,000, 5,000,000 and so on), and shapes just past their
  crossover were within run-to-run noise of Polars: two runs of the same Polars plan in the
  benchmark below differ by 0.62x to 1.31x (`MetalEngine default, cold` where it took nothing, against
  `polars in-memory`, rows of 5 ms or more in `Benchmarks/results/polars_engine_bench_2026-09-26-final3.csv`).
* **The default benchmark** (`BENCH_RATIO`, `BENCH`). The sweep times each case back to back; the
  default is also held to `Benchmarks/polars_engine_bench.py --idle` (a 100 ms warm-up, a first run
  after 500 ms of idle on its own, another warm-up, then best of 5 and their median), over the files
  `BENCH` names (`Benchmarks/results/polars_engine_default_groupby_raw_2026-09-30.csv`, "Group-bys
  after the grouped Float64 sum, mean and min/max were rebuilt" below). A class, or a group-count
  bucket for a case judged by its estimate, of which a case the default took there is behind the
  faster Polars engine on the best run or on the median is taken only from the smallest benchmarked
  size above the largest size it was behind at, and not at all when it is behind at the largest. A
  case of several classes (a mean and a max in one group-by) that is behind counts against those of
  its classes whose one-class cases at that bucket and size are not all ahead, and against all of
  them only when every one of its classes has one-class cases there that are ahead, so a class that
  is ahead on its own keeps its rows and the case itself is still left. The table records each
  benchmarked row's cases and their ratios as `benchmark`.

A class's crossover is the largest over the cases whose classes all belong to its node, so every case
that measures a class has to be ahead; if one of them has no fit, the class has no crossover. A case
whose classes span two nodes, (b) (a group-by under a top-k) and (e) (a whole-frame sum over a join),
measures no class. The fitted table is `python/arrowmetal/_engine_crossovers.py`, and
`polars_engine_crossover.py --check` fails when it and the results file disagree. The policy then
takes the larger of that crossover and the kernels' own: the router table in force
(`am.router_table()`, [CROSSOVER.md](CROSSOVER.md)) for the kernels it routes, and for the sort
classes the crossover of `argsort int64`, `argsort float64` and `lexsort (2 int32 keys)` against the
fastest CPU library in `Benchmarks/results/router_2026-09-24.json`, 1,000,000 rows. The fits below are
from `Benchmarks/results/polars_engine_crossover_2026-09-30-groupby.csv`.

| class | dtype | input | rows the default uses | the fit it came from | cases (their fit) |
|---|---|---|---:|---|---|
| `sort` | numeric | in-memory | 1,000,000 | the sort kernels, above the table's 947,836 ((q) 631,891 x 1.5) | (m) 456,986; (q) 631,891; (x2) 250,000, the smallest input measured |
| `sort` | numeric | Parquet | 1,500,000 | (s4) 1,000,000, the smallest file measured | (s4) uncompressed 1,000,000; (s4) snappy 1,000,000 |
| `join:left` | numeric | in-memory | 1,875,000 | (w2) 1,250,000, the smallest input measured | (w2) 1,250,000 |
| `join:inner` | numeric | in-memory | 2,029,827 | (w1) 1,353,218 | (w1) 1,353,218 |
| `join:anti` | numeric | in-memory | 3,727,959 | (w4) 2,485,306 | (w4) 2,485,306 |
| `distinct` | numeric | in-memory | 5,494,090 | (x1) 3,662,727 | (r) 1,252,378; (x1) 3,662,727 |
| `sort` | string | in-memory | 5,000,000 | the String floor, above (o) 2,439,140 x 1.5 = 3,658,710 | (o) 2,439,140; (p) 1,131,461 |
| `distinct` | string | in-memory | 5,000,000 | the String floor, above (y5) 2,463,352 x 1.5 = 3,695,028 | (y5) 2,463,352 |
| `group_by_multi:sum` | string | in-memory | per group count (below); with no estimate, 9,744,372 | (k) 6,496,248 | (k) 6,496,248 |
| `group_by_multi:minmax` | numeric | in-memory | per group count (below); with no estimate, 11,675,406 | (v4) 7,783,604 | (v4) 7,783,604; (c) 710,717; the grid's two-key min + max 535,292 to 7,023,044 |
| `group_by:minmax` | numeric | in-memory | per group count (below); with no estimate, 19,740,807 | (t4) 13,160,538 | (t4) 13,160,538; (t7) 1,312,255; the grid's one-key min + max 473,148 to 13,139,506 |
| `group_by_multi:mean` | numeric | in-memory | per group count (below); with no estimate, 24,143,946 | (v3) 16,095,964 | (v3) 16,095,964; (c) 710,717; (l) 778,101; the grid's two-key means 635,715 to 6,615,362 |
| `group_by_multi:count` | numeric | in-memory | per group count (below); with no estimate, 26,547,327 | (j) 17,698,218 | (j) 17,698,218; (v2) 710,626; the grid's two-key counts 661,173 to 6,712,670 |
| `group_by:sum`, `group_by:count`, `group_by:mean`, `group_by_multi:sum` | numeric | in-memory | per group count (below); with no estimate, not taken | | each has a case with no fit: over one key the 200- and 1,000-group grid cases, (t1), (t2) and (t3); over two keys, (g2sum200) |
| `join:semi` | numeric | in-memory | not taken | (f) none | (f) none; (w3) 2,464,828 |
| `aggregate:sum`, `:count`, `:mean`, `:minmax` | numeric | in-memory | not taken | | (a), (a2), (a3), (a4) none |
| `top_k` | numeric | in-memory | not taken | (x4) none | (n) 6,600,767; (x3) 6,210,387; (x4) none |
| `rowwise` | numeric | in-memory | not taken | | (d) none |
| `top_k`, `rowwise`, `aggregate:sum`, `group_by:sum`, `join:inner` | string | in-memory | not taken | | (y6), (y2), (y3), (y1), (y4) none |
| `group_by:sum`, `group_by:count`, `aggregate:sum`, `aggregate:count` | numeric | Parquet | not taken | | (s1) and (s3) none with both codecs; (s2) uncompressed none, snappy 1,000,000 |

A class with no row (a String column in any other class, and every Parquet class but these) has no
measurement and is not taken. `arrowmetal.polars_engine.placement_rules()` returns the table under the router
table in force, and `group_placement_rules()` the group-count buckets below.

**Group-bys: the number of groups.** Where the GPU group-by is ahead depends on the number of groups
more than on the rows. In the sweep (`Benchmarks/results/polars_engine_crossover_2026-09-30-groupby.csv`,
`vs_fastest_polars` of `MetalEngine all, cold`) the one-key sum over 200 groups is behind at every size ((t1) at
best 0.58x, (g1sum200) 0.51x); over 10,000, 100,000 and 1,000,000 groups it is ahead at 50,000,000
rows by 2.65x, 2.97x and 2.74x ((g1sum10k), (g1sum100k), (g1sum1M)); over about 0.43 times as many
groups as rows it is at 1.02x there ((g1sumR2)), inside the margin. So a group-by class is judged at
the bucket of its subtree's number of groups:

| bucket | groups |
|---|---|
| 200 | 1 to 447 |
| 1,000 | 448 to 3,162 |
| 10,000 | 3,163 to 31,622 |
| 100,000 | 31,623 to 316,227 |
| 1,000,000 | 316,228 to 3,162,277 |
| rows/2 | at least a quarter of the input rows, whatever the count |

A count between the 1,000,000 bucket and a quarter of the rows has no bucket and stays with Polars.
Each bucket is fitted like a class, from the points of the group-by cases whose number of groups at
that size falls in it (the grid, (c), (i), (j), (k), (l), (t1) to (t7), (v1) to (v4), (y1), (s1) and
(s2)); at each size the bucket's point is the worst of its cases there, so its step is the largest of
the cases' own, and a bucket ahead at its largest size alone has no crossover. The grid's
1,000,000-group cases hold a quarter of the rows or more below 5,000,000 rows, where they count in the
rows/2 bucket, so the 1,000,000 bucket was measured from 5,000,000 rows, and that is where its fits
start. The rows the default uses, with the fit each came from in brackets (the router table's
`group_by_sum` row is below every one of them):

| class | 200 | 1,000 | 10,000 | 100,000 | 1,000,000 | rows/2 |
|---|---:|---:|---:|---:|---:|---:|
| `group_by:sum` | not taken | not taken | 6,041,361 (4,027,574) | 5,252,956 (3,501,971) | 7,500,000 (5,000,000) | not taken |
| `group_by:count` | not taken | not taken | 3,123,562 (2,082,375) | 3,801,232 (2,534,155) | 7,500,000 (5,000,000) | not taken |
| `group_by:mean` | not taken | not taken | 2,436,520 (1,624,347) | 2,524,249 (1,682,833) | 7,500,000 (5,000,000) | not taken |
| `group_by:minmax` | 19,740,807 (13,160,538) | 19,709,259 (13,139,506) | 2,995,140 (1,996,760) | 2,184,147 (1,456,098) | 7,500,000 (5,000,000) | not taken |
| `group_by_multi:sum` | not taken | 11,993,616 (7,995,744) | 2,777,623 (1,851,749) | 2,288,101 (1,525,401) | 7,500,000 (5,000,000) | not taken |
| `group_by_multi:count` | 5,580,594 (3,720,396) | 10,069,005 (6,712,670) | 1,082,526 (721,684) | 1,118,998 (745,999) | 7,500,000 (5,000,000) | not taken |
| `group_by_multi:mean` | 9,923,043 (6,615,362) | 9,720,262 (6,480,175) | 50,000,000 (1,281,142)\* | 50,000,000 (635,715)\* | 7,500,000 (5,000,000) | not taken |
| `group_by_multi:minmax` | 10,534,566 (7,023,044) | 8,162,302 (5,441,535) | 1,171,750 (781,167) | 802,938 (535,292) | 7,500,000 (5,000,000) | not taken |
| `group_by_multi:sum`, String column | | | | | 9,744,372 (6,496,248) | not taken |
| `group_by:sum`, String column | | not taken | | | | |
| `group_by:sum`, `group_by:count`, Parquet | | not taken | | | | |

An empty cell has no measurement and is not taken. \* Raised by the default benchmark (`BENCH_RATIO`;
`Benchmarks/results/polars_engine_default_groupby_raw_2026-09-30.csv`, the faster Polars engine ÷ the default over
every run in which the default took the case): at 2,000,000 rows the default took (c) (a mean and a max over two keys, 10,000 groups) at 0.82x of
the faster Polars engine on the best run and 0.75x on the median, (g2mean10k) at 0.87x and 0.75x
and (g2mean100k) at 0.65x and 0.57x (both tables' runs where the default took them), all ahead at
50,000,000 rows, so the two-key mean's 10,000 and 100,000 buckets are taken from 50,000,000 rows.
(c)'s min + max class keeps its 10,000 bucket: (g2minmax10k), its one-class case there, was 1.18x and
1.09x. **The rows/2 bucket is never taken**
(`UNTAKEN_BUCKETS`), whatever its fit, because the sweep (`Benchmarks/results/polars_engine_crossover_2026-09-30-groupby.csv`) is not
monotone there: the one-key count over
0.43 times as many groups as rows, (g1countR2), is 0.78x, 1.9x, 4.04x, 2.21x and 1.18x the faster
Polars engine at 2,000,000, 5,000,000, 10,000,000, 20,000,000 and 50,000,000 rows, and the two-key
rows/2 cases peak at 10,000,000 rows the same way ((g2countR2) 6.47x there, 2.38x at 50,000,000).
(j), (v3) and (v4), whose keys range over 100,000 x 1,000 values and which hold 0.79 to 1.0 times as
many groups as rows in the sweep, are 1.22x, 1.17x and 1.51x at 50,000,000 rows. The grid's Float64
sums set two sum buckets: the one-key 10,000 bucket ((g1fsum10k) 1.04x at 2,000,000 rows and 1.25x
at 5,000,000, then 2.63x at 10,000,000) and the two-key 1,000 bucket ((g2fsum1k) 0.74x at 5,000,000
and 1.88x at 10,000,000). The one-key min + max over 200 and 1,000 groups is taken from about
20,000,000 rows: (g1minmax200) is 1.13x, 1.41x and 1.67x at 10,000,000, 20,000,000 and 50,000,000
rows, (g1minmax1k) 1.06x, 1.27x and 1.64x.

**The probe.** For a group-by whose keys are columns of one in-memory input frame, read as they are
(through filters, projections, joins and `unique` below it, which can only drop key values, so the
frame's count is an upper bound), the engine estimates the number of distinct key tuples in that
frame at plan time. It reads the keys of a stratified sample of n of the frame's N rows, row
i·(N / n) + h(i) with h a fixed-seed hash, so the same frame always gets the same sample; counts the
distinct tuples d and those seen exactly once (f1) and twice (f2); and scales them with the
bias-corrected Chao1 estimator (Chao, Biometrics 2005), D = d + f1 (f1 - 1) / (2 (f2 + 1)), clipped
to [d, N]. Its range is D with f2 moved by two Poisson standard deviations (and one or two more)
either way, the high end being the whole frame while fewer than about 6 pairs have been seen. The
samples are 512, 2,048, 8,192 and so on, at most 65,536 rows and a quarter of the frame, and the
probe stops at the first whose range the decision does not depend on: every bucket the range reaches
is taken at these rows, or none is. So a few hundred groups or ten thousand settle on 512 rows, a
hundred thousand on 2,048, and a count near a quarter of the rows takes more. A frame of at most
4,096 rows is counted exactly. One integer key is read as it is; several keys, and a String, Boolean
or temporal key or one holding nulls, through Polars' hash. The samples' counts are cached per frame,
key columns and sample size, keyed by the columns' buffers (the cache keeps those columns alive, so no
other frame can take their addresses while it stands), so collecting the same frame again samples
nothing and gets the same estimate; `arrowmetal.polars_engine.clear_group_estimates()` drops the
cache. The probe runs only when some bucket of the class is taken at the input rows; below every
bucket's crossover the subtree stays with Polars without one. A Parquet file's footer answers when
it states a distinct count for every key column in every row group, and only when the largest row
group's count and the sum over row groups fall in one bucket; the benchmark files state none. With no estimate (a key the plan computes, keys
from an aggregate below, two group-bys in one subtree, a footer without distinct counts) a group-by
class is judged by its row in the class table: `group_by_multi:minmax` from 11,675,406 rows,
`group_by:minmax` from 19,740,807, `group_by_multi:mean` from 24,143,946, `group_by_multi:count` from
26,547,327, the (String, int32) sum from 9,744,372, and every other numeric group-by class at no size.

**What the probe costs, and how close it gets.** `Benchmarks/group_probe_bench.py` collects a
group-by sum over the grid's frames (one int32 key, two, and one String key, at each group count)
through `MetalEngine()` nine times with its caches cleared and reads the probe's time from the
report: `Benchmarks/results/group_probe_2026-09-26.csv`, run conditions in
`Benchmarks/results/group_probe_2026-09-26_conditions.txt`. Against the fastest of Polars' two engines
and the default for the same group-by:

| rows | groups | probe (median of 9) | share of the group-by |
|---|---|---:|---:|
| 50,000,000 | 200 to 100,000 | 66 to 199 µs | 0.30% to 0.77% |
| 50,000,000 | 1,000,000 | 279 µs (one key), 461 µs (two) | 0.84%, 1.19% |
| 50,000,000 | half the rows | 255 µs (one key), 458 µs (two) | 0.12%, 0.10% |
| 2,000,000 | 200 to 100,000, two keys | 50 to 123 µs | 1.09% to 3.17% |
| 2,000,000 | a quarter of the rows or more, two keys | 366 and 393 µs | 4.70%, 4.90% |

That file was measured under the earlier group-count table. Under the table in force, a
group-by at 2,000,000 rows is probed only for the two-key count and min/max, the classes with a
bucket taken at that size; a one-key, String or two-key sum group-by there is not probed,
and neither is a String group-by at 50,000,000 (only the (String, int32) sum has buckets). In the
default benchmark below, where the probe runs in a process that holds every case's frames
(`probe_us` in `Benchmarks/results/polars_engine_bench_2026-09-26-final3.csv`, the probes of one collect summed), its median
is 99 µs at 2,000,000 rows (2.21% of the default's time for the case; 3 of the 14 probes under 1%)
and 144 µs at 50,000,000 (0.53%; 54 of 65 under 1%). The eight probes over 5 ms, 5.2 to 13.5 ms, are
all at 50,000,000 rows on frames with 0.43 times as many groups as rows or more ((g1minmaxR2) 5.2 ms,
0.93% of the default's time; (g2minmaxR2) 13.5 ms, 1.65%), and its largest share is 11.43% at
2,000,000 rows ((g2count100k), 461 µs of 4.0 ms). A second collect of the same frame reads its
samples from the cache. Sampled until its denominator settles instead of until the decision does (`free_estimate` in `Benchmarks/results/group_probe_2026-09-26.csv`), the
estimate falls in the bucket of the true count for every frame and key set of the grid at both sizes,
within 29% of it up to 1,000,000 groups at 50,000,000 rows, and 24% to 31% above it at half the rows.

Each probe is listed in the report, and the rule of a group-by the policy judged by its estimate
names it:

```
  metal:  GroupBy#1 [GroupBy > DataFrameScan] over 50,000,000 rows, ran in <t> ms -> 100,000 rows
          rule: 50,000,000 input rows is at or above the 2,961,043-row crossover for group_by:minmax at an estimated 102,981 groups over (k1), a Chao1 estimate from a 2,048-row sample, 67,758 to 198,392 (engine table, the 100,000-group bucket, ...)
  groups: GroupBy#1 over (k1): 102,981 groups, a Chao1 estimate from a 2,048-row sample, 67,758 to 198,392 (probed in 132 us)

  polars: GroupBy#1: rule: estimated 191 groups over (region), a Chao1 estimate from a 512-row sample, 189 to 196: below the measured band for group_by:sum at 50,000,000 input rows (taken at 3,163 to 3,162,277 groups; no count in the estimate's range is; ...)
  groups: GroupBy#1 over (region): 191 groups, a Chao1 estimate from a 512-row sample, 189 to 196 (probed in 76 us)
```

The other classes in the sweep (`Benchmarks/results/polars_engine_crossover_2026-09-30-groupby.csv`, `vs_fastest_polars`
of `MetalEngine all, cold`):

* **Joins and `unique`** on numeric keys: the left join is ahead from the smallest input measured,
  1,250,000 rows, and taken from 1,875,000; the inner join is fitted at 1,353,218 and taken from
  2,029,827, the anti join at 2,485,306 and from 3,727,959. The semi join is not taken: against a
  1,000-row table, (f), it is behind at every size (at best 0.42x), though against a 1,000,000-row
  table, (w3), it is fitted at 2,464,828. `unique` is fitted at 1,252,378 rows over 10,000 groups (r)
  and at 3,662,727 over about as many groups as rows (x1), and taken from 5,494,090.
* **Sorts** over in-memory frames are fitted at 250,000 rows (x2, the smallest input measured; 1.42x
  there), 456,986 (m) and 631,891 (q). The sort kernels' crossover, 1,000,000 rows, is above (q)'s
  947,836 and sets `sort`.
* **Top-k** is fitted at 6,600,767 rows by a Float64 key descending (n; 1.86x at 10,000,000 rows,
  1.81x at 50,000,000) and at 6,210,387 by an int64 key (x3; 1.1x to 1.35x from 5,000,000), but the
  top 100 by a nullable Float64 key descending (x4), whose answer is 100 of the null rows, is ahead at
  no size (0.75x to 1.05x), so `top_k` is not taken.
* **Whole-frame aggregates and row-wise filters and projections** are behind at every size:
  (a) to (a4) at 0.09x to 0.24x, (d) at 0.14x to 0.42x.
* **String shapes**: the sorts (o) and (p) and `unique` (y5) are taken from the 5,000,000-row
  floor, and the (String, int32) group-by (k), about 1,000,000 groups from
  5,000,000 rows, from 9,744,372. The filter by a String equality (y2, at best 0.13x), the prefix
  filter and sum (y3, 0.5x), the join on a String key (y4, 0.67x) and the top-k with a String column
  (y6, 0.88x) are behind at every size; the group-by on a String key with 1,000 values (y1) is ahead
  by the margin at no size (1.27x at 20,000,000 rows, 1.09x at 50,000,000).
* **Over a Parquet file**, cold (the open-file cache cleared before each run), the sort is ahead from
  the smallest file measured, 1,000,000 rows, with both codecs (4.37x uncompressed and 4.9x snappy
  there), and taken from 1,500,000. The filter and group-by (s1) has no fit with either codec
  (uncompressed at best 1.09x, at 50,000,000 rows; snappy 1.42x and 1.26x at 1,000,000 and
  2,000,000, then 0.79x to 0.92x), nor has the aggregate (s3) (at best 0.84x); the filter that skips
  row groups, (s2), is ahead at every size snappy (1.24x to 2.48x) and at none uncompressed (at best
  0.99x). So the group-by and aggregate classes over a file are not taken.

**The default against Polars and against `shapes="all"`.** `Benchmarks/polars_engine_bench.py` over
the 93 in-memory cases at 2,000,000 and 50,000,000 rows and the four Parquet scan cases over the
50,000,000-row files, snappy and uncompressed, 194 case-size pairs, best of 5, every result equal to
Polars' (the cold runs of `MetalEngine()` clear the group-count cache too, so each pays its probe):
`Benchmarks/results/polars_engine_bench_2026-09-26-final3.csv`, run conditions in
`Benchmarks/results/bench_conditions_2026-09-26-final3.txt`. The default took a subtree in 62 pairs
(10 at 2,000,000 rows and 52 at 50,000,000), 42 of them group-bys, and every one is ahead of the faster
Polars engine, from 1.21x ((t6) at 50M) to 9.42x ((g2count1M) at 50M):

| case | rows | Polars in-memory | Polars streaming | `shapes="all"`, cold | `MetalEngine()`, cold | vs faster Polars |
|---|---|---:|---:|---:|---:|---:|
| (e) inner join then sum (the join taken) | 2M | 7.5 ms | 4.9 ms | 3.7 ms | **2.6 ms** | 1.87 |
| (m) sort 3 columns by an int64 key | 2M | 12.1 ms | 13.0 ms | 3.4 ms | **2.5 ms** | 4.86 |
| (q) filter, then sort by (int32 asc, int64 desc) | 2M | 14.4 ms | 14.4 ms | 4.0 ms | **3.9 ms** | 3.72 |
| (v2) group-by (region, sub), count | 2M | 6.5 ms | 5.0 ms | 3.3 ms | **3.5 ms** | 1.43 |
| (w1) inner join, 1M-row build side | 2M | 7.7 ms | 5.2 ms | 4.3 ms | **2.3 ms** | 2.31 |
| (w2) left join, 1M-row build side | 2M | 10.2 ms | 6.0 ms | 3.2 ms | **2.6 ms** | 2.27 |
| (x2) sort by a nullable Float64 key, descending | 2M | 18.2 ms | 19.3 ms | 6.3 ms | **6.4 ms** | 2.82 |
| (g2count100k) group-by grid, 2 keys, 100,000 groups, count | 2M | 6.9 ms | 5.7 ms | 3.6 ms | **4.0 ms** | 1.42 |
| (g2count10k) group-by grid, 2 keys, 10,000 groups, count | 2M | 5.8 ms | 5.3 ms | 3.4 ms | **3.4 ms** | 1.54 |
| (g2mean10k) group-by grid, 2 keys, 10,000 groups, mean | 2M | 7.6 ms | 5.3 ms | 3.6 ms | **3.8 ms** | 1.39 |
| (c) group-by (region, sub) mean + max | 50M | 208.7 ms | 109.4 ms | 61.4 ms | **61.0 ms** | 1.80 |
| (e) inner join then sum (the join taken) | 50M | 21.6 ms | 15.6 ms | 9.1 ms | **9.0 ms** | 1.74 |
| (i) group-by 1 key, 100,000 groups, sum | 50M | 90.3 ms | 45.6 ms | 24.2 ms | **24.3 ms** | 1.88 |
| (k) group-by (String, int32), sum | 50M | 375.1 ms | 165.2 ms | 74.3 ms | **74.7 ms** | 2.21 |
| (l) group-by (region, sub), Float64 sum + mean | 50M | 212.8 ms | 112.0 ms | 55.3 ms | **55.2 ms** | 2.03 |
| (m) sort 3 columns by an int64 key | 50M | 377.8 ms | 441.9 ms | 72.4 ms | **74.6 ms** | 5.07 |
| (o) sort with a String column, by an int64 key | 50M | 392.7 ms | 502.8 ms | 272.0 ms | **282.4 ms** | 1.39 |
| (p) filter, then sort by (int32 asc, nullable Float64 desc) | 50M | 952.8 ms | 1009.3 ms | 406.9 ms | **394.7 ms** | 2.41 |
| (q) filter, then sort by (int32 asc, int64 desc) | 50M | 720.8 ms | 734.8 ms | 114.7 ms | **116.5 ms** | 6.18 |
| (r) unique over (region, sub), keep first | 50M | 172.5 ms | 172.9 ms | 28.4 ms | **30.3 ms** | 5.69 |
| (s4) Parquet scan, sort 2 columns by a float64 key, uncompressed | 50M | 581.2 ms | 620.5 ms | 117.0 ms | **100.4 ms** | 5.79 |
| (s4) Parquet scan, sort 2 columns by a float64 key, snappy | 50M | 604.3 ms | 632.9 ms | 125.5 ms | **188.6 ms** | 3.20 |
| (t5) group-by 1 key, 100,000 groups, count | 50M | 75.3 ms | 44.2 ms | 14.0 ms | **13.9 ms** | 3.19 |
| (t6) group-by 1 key, 100,000 groups, mean | 50M | 99.1 ms | 60.2 ms | 49.7 ms | **49.9 ms** | 1.21 |
| (t7) group-by 1 key, 100,000 groups, min + max | 50M | 119.9 ms | 69.4 ms | 37.5 ms | **37.6 ms** | 1.85 |
| (v1) group-by (region, sub), sum | 50M | 200.9 ms | 112.2 ms | 28.6 ms | **28.7 ms** | 3.92 |
| (v2) group-by (region, sub), count | 50M | 166.5 ms | 103.7 ms | 18.5 ms | **18.2 ms** | 5.69 |
| (w1) inner join, 1M-row build side | 50M | 95.2 ms | 82.2 ms | 35.1 ms | **35.2 ms** | 2.33 |
| (w2) left join, 1M-row build side | 50M | 267.1 ms | 100.1 ms | 41.4 ms | **41.1 ms** | 2.43 |
| (w4) anti join, 1M-row build side | 50M | 82.1 ms | 63.9 ms | 37.6 ms | **37.4 ms** | 1.71 |
| (x1) unique over (k1, k2), keep first | 50M | 771.6 ms | 477.3 ms | 277.4 ms | **281.2 ms** | 1.70 |
| (x2) sort by a nullable Float64 key, descending | 50M | 551.5 ms | 639.5 ms | 160.8 ms | **249.6 ms** | 2.21 |
| (y5) unique over (String, int32), keep first | 50M | 401.7 ms | 434.4 ms | 75.5 ms | **67.7 ms** | 5.93 |
| (g1count100k) group-by grid, 1 key, 100,000 groups, count | 50M | 74.5 ms | 86.3 ms | 19.2 ms | **16.9 ms** | 4.40 |
| (g1count10k) group-by grid, 1 key, 10,000 groups, count | 50M | 63.6 ms | 84.7 ms | 18.0 ms | **15.8 ms** | 4.03 |
| (g1count1M) group-by grid, 1 key, 1,000,000 groups, count | 50M | 154.3 ms | 96.9 ms | 20.4 ms | **18.7 ms** | 5.18 |
| (g1mean100k) group-by grid, 1 key, 100,000 groups, mean | 50M | 112.0 ms | 86.8 ms | 57.6 ms | **59.3 ms** | 1.46 |
| (g1mean10k) group-by grid, 1 key, 10,000 groups, mean | 50M | 96.5 ms | 94.9 ms | 54.5 ms | **51.6 ms** | 1.84 |
| (g1minmax100k) group-by grid, 1 key, 100,000 groups, min + max | 50M | 124.5 ms | 145.9 ms | 50.8 ms | **43.9 ms** | 2.83 |
| (g1minmax10k) group-by grid, 1 key, 10,000 groups, min + max | 50M | 110.7 ms | 153.7 ms | 42.9 ms | **37.4 ms** | 2.96 |
| (g1minmax1M) group-by grid, 1 key, 1,000,000 groups, min + max | 50M | 210.7 ms | 123.1 ms | 64.5 ms | **60.2 ms** | 2.04 |
| (g1sum100k) group-by grid, 1 key, 100,000 groups, sum | 50M | 103.0 ms | 122.8 ms | 38.1 ms | **30.3 ms** | 3.40 |
| (g1sum10k) group-by grid, 1 key, 10,000 groups, sum | 50M | 96.6 ms | 97.1 ms | 35.9 ms | **29.1 ms** | 3.32 |
| (g1sum1M) group-by grid, 1 key, 1,000,000 groups, sum | 50M | 182.4 ms | 103.2 ms | 38.4 ms | **34.2 ms** | 3.02 |
| (g2count100k) group-by grid, 2 keys, 100,000 groups, count | 50M | 246.8 ms | 180.0 ms | 25.3 ms | **22.3 ms** | 8.07 |
| (g2count10k) group-by grid, 2 keys, 10,000 groups, count | 50M | 209.9 ms | 147.1 ms | 25.3 ms | **21.6 ms** | 6.81 |
| (g2count1M) group-by grid, 2 keys, 1,000,000 groups, count | 50M | 446.2 ms | 224.9 ms | 27.5 ms | **23.9 ms** | 9.42 |
| (g2count1k) group-by grid, 2 keys, 1,000 groups, count | 50M | 189.0 ms | 53.2 ms | 23.6 ms | **19.8 ms** | 2.68 |
| (g2count200) group-by grid, 2 keys, 200 groups, count | 50M | 37.5 ms | 37.4 ms | 19.3 ms | **18.3 ms** | 2.04 |
| (g2mean100k) group-by grid, 2 keys, 100,000 groups, mean | 50M | 313.7 ms | 206.2 ms | 67.6 ms | **69.7 ms** | 2.96 |
| (g2mean10k) group-by grid, 2 keys, 10,000 groups, mean | 50M | 250.8 ms | 174.8 ms | 62.1 ms | **55.8 ms** | 3.13 |
| (g2mean1M) group-by grid, 2 keys, 1,000,000 groups, mean | 50M | 478.9 ms | 268.9 ms | 89.4 ms | **100.0 ms** | 2.69 |
| (g2mean1k) group-by grid, 2 keys, 1,000 groups, mean | 50M | 257.0 ms | 62.4 ms | 56.0 ms | **50.4 ms** | 1.24 |
| (g2minmax100k) group-by grid, 2 keys, 100,000 groups, min + max | 50M | 284.2 ms | 217.7 ms | 52.1 ms | **46.4 ms** | 4.69 |
| (g2minmax10k) group-by grid, 2 keys, 10,000 groups, min + max | 50M | 260.7 ms | 169.5 ms | 48.3 ms | **41.2 ms** | 4.12 |
| (g2minmax1M) group-by grid, 2 keys, 1,000,000 groups, min + max | 50M | 514.6 ms | 295.5 ms | 68.8 ms | **65.5 ms** | 4.51 |
| (g2minmax1k) group-by grid, 2 keys, 1,000 groups, min + max | 50M | 269.8 ms | 66.9 ms | 50.2 ms | **31.9 ms** | 2.09 |
| (g2minmax200) group-by grid, 2 keys, 200 groups, min + max | 50M | 67.4 ms | 97.6 ms | 43.0 ms | **32.9 ms** | 2.05 |
| (g2sum100k) group-by grid, 2 keys, 100,000 groups, sum | 50M | 280.8 ms | 190.0 ms | 41.8 ms | **33.3 ms** | 5.70 |
| (g2sum10k) group-by grid, 2 keys, 10,000 groups, sum | 50M | 237.8 ms | 191.5 ms | 45.7 ms | **32.7 ms** | 5.85 |
| (g2sum1M) group-by grid, 2 keys, 1,000,000 groups, sum | 50M | 463.6 ms | 255.9 ms | 44.5 ms | **40.1 ms** | 6.39 |
| (g2sum1k) group-by grid, 2 keys, 1,000 groups, sum | 50M | 244.4 ms | 53.8 ms | 37.3 ms | **29.4 ms** | 1.83 |

In the other 132 pairs the default took nothing and ran Polars' in-memory plan; there its time was
0.39 to 2.92 times the `polars in-memory` row of the same case (`wall_ms` in
`Benchmarks/results/polars_engine_bench_2026-09-26-final3.csv`), and 0.62 to 1.31 times in the 102 of
them where that row is 5 ms or more. That is the spread of two runs of the same Polars plan in this
run, and the noise band the ratios above are read against. `shapes="all"` took a subtree and was
ahead of the faster Polars engine in 34 of those 132 (`vs_fastest_polars` of `MetalEngine all, cold`), 14 of them above 1.31x, each left by the default
because its class or bucket has a crossover above that size or none:

* at 2,000,000 rows: group-bys below the lowest bucket of their class, (g2minmax100k) 1.98x,
  (g2minmax1M) 1.96x, (g2minmaxR2) 1.81x, (g2minmax10k) 1.70x and (c) 1.08x (`group_by_multi:minmax`
  from 2,029,638), (l) 1.50x, (g2sum100k) 1.35x, (v1) 1.21x, (g2sum10k) 1.20x, (g2sum1M) 1.04x and
  (g2sumR2) 1.04x (`group_by_multi:sum` from 2,288,101), (g1minmax10k) 1.57x, (g1minmax100k) 1.20x,
  (t7) 1.18x and (g1minmax1M) 1.08x (`group_by:minmax` from 2,583,360), (g1count10k) 1.07x (from
  3,123,562) and (g1mean10k) 1.02x (from 4,172,575); two-key counts over about 865,000 groups,
  estimated above every bucket taken at that size, (g2count1M) 1.44x and (g2countR2) 1.35x; `unique`
  (r) 1.57x (below 5,494,090); the String shapes (p) 1.31x and (y5) 1.07x (below the 5,000,000-row
  floor) and (o) 1.21x (below 6,301,531);
* at 50,000,000 rows: group-bys over 0.43 times as many groups as rows, estimated above every bucket
  taken for their class, (g2countR2) 2.02x, (g2sumR2) 1.76x, (g2minmaxR2) 1.25x, (g2meanR2) 1.07x
  and (g1countR2) 1.06x; group-bys in buckets not taken for their class, (g2sum200) 1.47x (200
  groups), (g1mean1M) 1.20x (1,000,000 groups), (g1minmax1k) 1.16x and (g1count1k) 1.04x (1,000
  groups); the semi join (w3) 1.68x (`join:semi` is not taken, (f)); over the
  snappy Parquet file, the filter that skips row groups and group-by (s2) 1.14x (`group_by:sum` over a
  file is not taken, (s1)).

**Sorts with one key per Polars key.** The benchmark above ran when a nulls-first nullable key and a
descending float key still carried an extra key column each. With Polars' order as options of the one
key, the sort and top-k cases and four group-by controls were timed old against new, alternating, three
rounds of best of 5 after a 100 ms warm-up (`Benchmarks/results/polars_engine_sort_options_2026-09-30.csv`,
run conditions in `…_conditions.txt`); best and median of the rounds, in ms:

| case | rows | engine | before, best / median | after, best / median | after 500 ms idle, before → after |
|---|---|---|---:|---:|---:|
| (x4) top 100 by a nullable Float64 key, descending | 50M | `shapes="all"`, cold | 129.53 / 129.89 | **11.07 / 11.09** | 168.14 → 15.84 |
| (x4) top 100 by a nullable Float64 key, descending | 50M | `shapes="all"`, warm | 122.91 / 123.46 | **4.85 / 4.98** | 136.94 → 6.97 |
| (x4) top 100 by a nullable Float64 key, descending | 2M | `shapes="all"`, cold | 12.31 / 13.77 | **2.57 / 2.63** | 20.43 → 6.72 |
| (n) top 100 by Float64, descending | 50M | `shapes="all"`, cold | 79.93 / 80.10 | **11.50 / 13.00** | 103.92 → 33.13 |
| (n) top 100 by Float64, descending | 2M | `shapes="all"`, cold | 8.56 / 10.00 | **2.17 / 2.36** | 11.48 → 4.87 |
| (x2) sort by a nullable Float64 key, descending | 50M | `MetalEngine()`, cold | 134.03 / 134.24 | **93.54 / 93.96** | 163.88 → 106.02 |
| (x2) sort by a nullable Float64 key, descending | 2M | `MetalEngine()`, cold | 15.71 / 16.64 | **8.79 / 10.67** | 20.52 → 15.77 |
| (p) filter, then sort by (int32 asc, nullable Float64 desc) | 50M | `MetalEngine()`, cold | 350.54 / 351.84 | **317.39 / 317.96** | 374.03 → 344.01 |
| (p) filter, then sort by (int32 asc, nullable Float64 desc) | 2M | `shapes="all"`, cold | 25.82 / 26.77 | **16.01 / 16.46** | 43.11 → 36.91 |

The faster Polars engine in the same runs (best of the rounds of the new build, `new_best_ms` of the
Polars rows): (x4) 9.75 ms at 50M
and 1.21 ms at 2M (streaming), (n) 18.43 ms and 2.23 ms (streaming), (x2) 542.42 ms and 17.54 ms,
(p) 978.94 ms and 22.02 ms (in-memory).
The top-k cases are still left to Polars by the default (`top_k` has no crossover, above), so its rows
for them are Polars' in either build. The sorts whose keys carry no option ((m), (o), (q), the Parquet
(s4)), the top-k by an int64 key (x3), (y6), (b) and the group-by controls (c), (i), (j), (t1) are
unchanged: every row that was more than 5% slower in both best and median over the three rounds, all
of them on paths this change does not touch, was timed again alone over eight alternating rounds of
100 runs (the rows `flagged row re-timed alone` in the same file), where before ÷ after is 0.95 to
1.25 in best and 0.96 to 1.22 in median. No row's median
process CPU time rose by more than 1.5x.

**Group-bys after the grouped Float64 sum, mean and min/max were rebuilt.** The 75 group-by cases
(the grid's 60, (c), (i), (j), (l), (t1) to (t7), (v1) to (v4)) at 2,000,000 and 50,000,000 rows,
`MetalEngine()` with the table fitted before the 2026-09-30 re-sweep and with the table fitted from
it (before the default-benchmark rule), the same build, three alternating rounds of best of 5 after a
100 ms warm-up, the first run after 500 ms of idle recorded on its own:
`Benchmarks/results/polars_engine_default_groupby_2026-09-30.csv` (per row best of the rounds and
median of the rounds' medians; every run in `…_raw_2026-09-30.csv`, run conditions in
`…_2026-09-30_conditions.txt`). Every answer was equal to Polars'. The re-swept table took 57 of the
150 case-size pairs (8 at 2,000,000 rows, 49 at 50,000,000), the earlier one 48, and it left none
that the earlier one took. What it took that the earlier one left, in ms (best of the rounds; the
earlier table's default ran Polars' in-memory plan there):

| case | rows | faster Polars engine | earlier table | table in force | vs faster Polars, best / median | after 500 ms idle |
|---|---|---:|---:|---:|---:|---:|
| (g2minmax10k) group-by grid, 2 keys, 10,000 groups, min + max | 2M | 5.50 | 6.37 | **4.66** | 1.18x / 1.09x | 11.50 |
| (g2minmax100k) group-by grid, 2 keys, 100,000 groups, min + max | 2M | 6.14 | 8.09 | **5.46** | 1.12x / 1.12x | 13.36 |
| (c) group-by (region, sub) mean + max | 2M | 5.49 | 6.97 | 6.73 | 0.82x / 0.75x | 9.63 |
| (g2mean100k) group-by grid, 2 keys, 100,000 groups, mean | 2M | 5.82 | 8.38 | 8.90 | 0.65x / 0.57x | 14.84 |
| (g1mean1M) group-by grid, 1 key, 1,000,000 groups, mean | 50M | 58.08 | 172.67 | **31.13** | 1.87x / 2.27x | 60.58 |
| (g2mean200) group-by grid, 2 keys, 200 groups, mean | 50M | 39.25 | 40.50 | **19.31** | 2.03x / 2.03x | 58.85 |
| (g1minmax200) group-by grid, 1 key, 200 groups, min + max | 50M | 16.54 | 16.82 | **12.33** | 1.34x / 1.09x | 32.69 |
| (t4) group-by 1 key, 200 groups, min + max | 50M | 15.61 | 15.87 | **12.02** | 1.30x / 1.13x | 26.90 |
| (g1minmax1k) group-by grid, 1 key, 1,000 groups, min + max | 50M | 16.05 | 97.29 | **12.94** | 1.24x / 1.01x | 34.00 |

Every other pair it took was ahead of the faster Polars engine on the best run and on the median
(`new_vs_fastest_polars_best` and `_median` in the same file): at
50,000,000 rows from 1.24x ((g1minmax1k)) to 10.69x ((g2count1M)), at 2,000,000 rows from 1.12x to
1.63x ((g2count10k)). Taken and behind at 2,000,000 rows were the two-key means (c) at 0.82x,
(g2mean100k) at 0.65x and (g2mean10k), which the earlier table took as well, at 0.87x (5.83 against
5.08 ms; 0.73x under the earlier table). The default-benchmark rule (above) raises the two-key mean's
10,000 and 100,000 buckets to 50,000,000 rows, so the table in force takes 54 of the 150 pairs (5 at
2,000,000 rows, 49 at 50,000,000), every one at 1.0x or more of the faster Polars engine on the best
run and the median in this benchmark, and leaves (c), (g2mean10k) and (g2mean100k) at 2,000,000 rows
to Polars, (g2mean10k) being the one pair the earlier table took that this one leaves. Those three and
(g2minmax10k), which shares (c)'s 10,000-group bucket of the min + max class, re-timed, the table
before the rule against the table in force, three alternating rounds
(`Benchmarks/results/polars_engine_default_groupby_retime_2026-10-01.csv`, run conditions in
`…_retime_2026-10-01_conditions.txt`), in ms, best / median, against the faster Polars engine's best
of 5.49, 4.98, 5.61 and 5.44 ms:

| case | before the rule | table in force |
|---|---:|---:|
| (c) | taken: 8.28 / 9.81 (0.66x / 0.58x) | Polars' in-memory plan: 6.83 / 7.49 (0.80x / 0.76x) |
| (g2mean10k) | taken: 6.54 / 7.28 (0.76x / 0.72x) | Polars' in-memory plan: 6.78 / 7.22 (0.73x / 0.72x) |
| (g2mean100k) | taken: 8.24 / 8.87 (0.68x / 0.66x) | Polars' in-memory plan: 7.39 / 8.32 (0.76x / 0.70x) |
| (g2minmax10k) | taken: 4.63 / 4.83 (1.17x / 1.17x) | taken: 4.04 / 4.35 (1.34x / 1.30x) |

Left to Polars, `MetalEngine()` runs Polars' in-memory plan, which uses the CPU cores (77 to 90 CPU-ms
against 4 to 6 on the GPU path for these three, `new_cpu_ms` and `old_cpu_ms` in the retime file). In this benchmark each engine's timed runs follow Polars' runs
of the same case and a 500 ms idle gap, and these three take 4.75 to 9.79 ms (best of 5) under
`shapes="all"`, cold or warm, as well (the `reported` rounds of `…_raw_2026-09-30.csv`); run back to back, 15 runs of each way of collecting in turn, three
rounds and no idle gap (`Benchmarks/results/polars_engine_groupby_backtoback_2026-10-01.csv`), the
default runs them in 2.89 ms (c), 2.41 ms (g2mean10k) and 3.03 ms (g2mean100k) against Polars'
streaming engine's 5.48, 5.09 and 5.74 ms, which is where the crossover sweep, measured the same
back-to-back way, put them (2.09x, 2.19x and 2.34x at 2,000,000 rows in
`Benchmarks/results/polars_engine_crossover_2026-09-30-groupby.csv`). In the 150-pair benchmark the
pairs whose decision did not change ran the same plan under both tables; the rows of those that were
more than 5% slower in best and median under one table than the other ((g1count10k) and (g1count1M)
at 50,000,000 rows, taken by both; (g2count100k) at 2,000,000, taken by both; (g1minmax100k) and (t4)
at 2,000,000, Polars' plan under both) differ between the two runs of one plan.

**Float64 group sums and means at 2^24 groups.** The (v3) case of the crossover sweep, a mean over two keys with about as many groups as rows, found ArrowMetal's group-by returning null for most groups at 16,777,216 groups and above (16,777,215 were right): the per-group kernels dispatched one threadgroup per group and the grid wrapped past 2^32 threads. Fixed in the core ([FINDINGS.md](FINDINGS.md), round 13; `python/tests/test_group_by_2_24.py`), so no engine rule is needed and every group-by shape follows the policy above. The sweep and the benchmark above ran with the fix: under `shapes="all"` the cases the guard once kept with Polars, `(c)`, `(l)`, `(t3)`, `(t6)`, `(v3)` and the Parquet cases `(s1)` and `(s2)`, run on Metal and equal Polars' answers. At 50M rows, under `shapes="all"`, cold (`Benchmarks/results/polars_engine_bench_2026-09-26-final3.csv`), `(c)` is 1.78, `(l)` 2.03 and `(t6)` 1.21 times the faster Polars engine; `(t3)` is behind at 35.5 ms against 12.4 and `(v3)` at 612.0 ms against 421.7.

```python
am.MetalEngine()                          # shapes="measured": the crossovers and buckets above
am.MetalEngine(shapes="all")              # every translatable subtree of at least 1,000,000 rows
am.MetalEngine(shapes="all", min_rows=0)  # everything it can translate (what the tests use)
am.MetalEngine(shapes={"group_by", "sort"})  # the classes named (or their prefix), any size
am.MetalEngine(min_rows=5_000_000)        # the crossovers, and at least 5,000,000 rows
am.MetalEngine(raise_on_fail=True)        # raise instead of leaving anything to Polars
```

Each subtree the report lists as taken carries the rule that took it, and each node the policy left
has a `rule:` line:

```
  metal:  Sort#2 [Sort > DataFrameScan] over 2,000,000 rows, ran in <t> ms -> 2,000,000 rows
          rule: 2,000,000 input rows is at or above the 1,000,000-row crossover for sort (crossover sweep, ...)
  polars: Select#3: rule: aggregate:sum was not measured ahead of Polars up to 50,000,000 input rows (...)
  polars: Join#2: rule: 900,000 input rows is below the 2,029,827-row crossover for join:inner (...)
```

`shapes="all"` is there for plans the benchmark did not cover and for moving work off the CPU cores:
the CPU-ms column of the results file is the process CPU time of each run.

### The import cache

Each Polars column the engine imports is kept, keyed by its Arrow type, length, offset and the
address and size of every buffer, and the next query that reads the same column reuses the import.
That is safe because the cached import holds the exported array and, through it, Polars' own buffer:
while an entry lives, Polars can neither free that memory (so no other column can appear at the same
address) nor write to it in place (Polars copies a buffer it shares before writing). Only a column
that was imported without a copy is cached; a String column (handed over as views, without a copy,
but outside the cache's key) and a multi-chunk column (concatenated on every export) are not. Entries are evicted least recently
used above a byte budget, a quarter of physical memory by default:

```python
from arrowmetal import polars_engine
polars_engine.import_cache_info()       # {"entries", "bytes", "limit", "hits", "misses"}
polars_engine.import_cache_limit(2 << 30)
polars_engine.clear_import_cache()      # and with it the Polars buffers it kept alive
```

The "warm" column of the results file is a second collect over the same frame; the defaults above
were read from the cold one.

### Parquet scans

```python
lf = (pl.scan_parquet("trades.parquet")
        .filter(pl.col("price") > 500.0)
        .group_by("qty").agg(pl.col("weight").sum()))
lf.collect(engine=am.MetalEngine(shapes="all"))
```

A `Scan` of one local Parquet file becomes a leaf the engine reads itself, so the subtree above it
needs no import at all: the columns are decoded on the GPU straight into Metal memory
([PARQUET.md](PARQUET.md)) and the plan runs over them.

* **Projection.** The columns Polars' projection pushdown left on the `Scan` are the reader's column
  list; no other column chunk is touched. Dictionary-encoded columns are read materialised, which
  is what Polars reads them as.
* **Predicate.** Polars pushes a filter into the `Scan` as its predicate. The engine translates it
  like any `Filter` and runs it on the GPU over the rows the reader returns. The comparisons in it
  that the file's statistics can judge the way Polars compares go to the reader as well, which skips
  row groups (and, with a page index, pages) that cannot hold a matching row: a comparison of a
  column with a literal, joined by `&`, on an integer column (all six operators), a String column
  (`==`, `!=`) or a float column (`<`, `<=` and `==` only, against a literal exact in the column's
  type). Polars orders NaN above every number, so NaN rows pass `>`, `>=` and `!=`, and writers leave
  NaN out of min/max; those three never reach the reader for a float column, so `!= x` keeps a NaN
  in a row group whose statistics say x .. x (the case of apache/arrow#51491). `|`, functions and
  comparisons of two columns stay on the GPU filter only. A `Filter` Polars left directly above the
  `Scan` (with its predicate pushdown off) is handed to the reader the same way.
  `scan_parquet(use_statistics=False)` hands it nothing.
* **Nulls.** A column is treated as nullable unless the footer's statistics say it holds no null
  (`ParquetFile.column_null_count`), so a sort key by a column without nulls carries no null placement. The
  read checks the footer's word, and a file whose data holds nulls its statistics deny fails the
  query with `ArrowMetalError` rather than answering differently.
* **The open file is kept.** The reader goes through the open-file cache
  ([PARQUET.md](PARQUET.md), "The open-file cache"), keyed by the file's path, inode, modification
  time and size, so the second query over a file does not map it again, and a rewritten file is read
  afresh. `am.clear_parquet_cache()` and `am.parquet_cache_limit()` control it.
* **Checked once per plan and file.** As for in-memory inputs, the plan first runs over a prefix
  of the file (64 rows of its first row group) and its output schema is compared with Polars'. A
  file whose stored Arrow schema Polars applies and ArrowMetal's reader does not (one with a
  different number of fields than the file, which Arrow's own reader ignores) fails that check or
  names a column the file does not have, and stays with Polars.
* **A bare scan stays with Polars.** A `Scan` with nothing above it that does GPU work is Polars'
  to read.

What stays with Polars, with the reason in the report: several files (a list, or a glob or
directory that matches more than one), hive partition columns, a URL or cloud path (`file://`
included), `row_index_name`, `n_rows` (and a `head`, `tail` or `slice` Polars pushed into the scan),
`include_file_paths`, `schema=`, deletion files, column mapping, a column whose dtype the plan does
not carry (Decimal, Categorical, Enum, List, Struct, Binary, ...), a predicate that does not
translate, and CSV and NDJSON scans. polars 1.44.1 cannot show an IPC scan to an engine
(`NodeTraverser.view_current_node` raises `NotImplementedError: ipc scan`), so `scan_ipc` stays with
Polars as well.

Each taken subtree's report entry lists what the reader did (`scans`: the file, the filters it was
given, row groups read and skipped, pages skipped):

```
  metal:  GroupBy#2 [GroupBy > SimpleProjection > Scan] over 50,000,000 rows, ran in <t> ms -> 1,000 rows
          parquet /data/bench-none-50000000.parquet: filters [('id', '<', 5000000)], 5 row groups read, 45 skipped, 0 pages skipped
```

`MetalEngine()` judges a scan subtree by the same rule as an in-memory one, with the Parquet rows of
the crossover table and the file's row count from its footer: a sort of a file of at least 1,500,000
rows is taken, and the filter, group-by and aggregate shapes over a file are not taken at any size
("Which translatable subtrees it runs: the defaults" above). The scan cases below are in line with
that: the sort is ahead of both Polars engines cold and warm, and the filter, group-by and aggregate
cases are behind cold.

### The report

```
MetalEngine report for collect (polars 1.44.1, IR (14, 7))
  metal:  Sort#3 [Sort > HStack > Filter > DataFrameScan] over 10,000,000 rows, ran in <t> ms -> <n> rows
  polars: Select#1: Function rank has no ArrowMetal translation.
```

`engine.last_report` is a `MetalPlanReport`: `taken` (one entry per subtree that ran on Metal: the
node at its top, the node kinds inside it, the rows it read, the ArrowMetal plan text, and after the
run its wall time and output rows), `fallbacks` (one `Kind#id: reason` line per node that stayed
with Polars for a reason of its own; every translation reason is a full sentence, which
`test_every_unsupported_reason_is_a_full_sentence` checks, and a node the placement policy left
has a `rule: ...` line, as above), `walked` (every node of the optimised
plan), `nodes`, and `path` (the collect path it is for, "Collect paths" above). With
`POLARS_VERBOSE=1` the fallback lines are also issued as a `PerformanceWarning`. The report belongs to
the engine object, so two threads collecting through one `MetalEngine` overwrite each other's.

### Tests

```
PYTHONPATH=python python -m pytest python/tests/test_polars_engine.py -q
DIFF_QUICK=1 PYTHONPATH=python python -m pytest python/tests/test_polars_engine.py -q   # without the 100,003-row datasets
PYTHONPATH=python python -m pytest python/tests/test_engine_policy.py -q
```

`test_engine_policy.py` checks the default policy: the same decision for the same subtree across 200
calls and in a second process, every row of the crossover table one row below its crossover (left,
with the reason) and at it (taken), a subtree needing every class's crossover, the String and Parquet
rows, the router-table and sort-kernel floors, the overrides, a Parquet scan judged by its footer's
row count under a filter that keeps a handful of rows, the report's `rule` lines, that the committed
table is the fit of the results file it names, and the Float64 group-by guard at 2^24 groups. For
the group-count buckets it checks that the buckets tile the counts, every bucket one row below its
crossover and at it, a losing bucket named below or above the band, an estimate's range taken only
where every count in it is, the fallback to the class row without an estimate, where the probe finds
a group-by's keys, the probe's estimate in the true count's bucket over one int key, two and a String
key at 2,000,000 rows, the same estimate across 200 calls and in a second process, the cache, and the
engine taking a group-by in a winning bucket and leaving one in a losing bucket at the same rows.

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

`test_every_collect_path_is_explicit` runs every row of the collect-path table and checks where the
plan ran, the report's `path` and reason, and that the warning comes once.
`test_the_capability_table_is_current` regenerates [ENGINE_CAPABILITIES.md](ENGINE_CAPABILITIES.md)
and compares it with the committed file; no cell of it may be a Metal answer that differs from
Polars' or an engine exception. `test_an_unknown_node_kind_keeps_the_whole_plan_on_polars` and
`test_a_node_polars_cannot_describe_keeps_the_whole_plan_on_polars` check the fail-closed walk, and
`test_the_check_command_prints_versions_and_the_capability_header` the command below.

`python -m arrowmetal.polars_engine check` prints the installed Polars version and whether it is one
of `TESTED_POLARS`, the IR version against `TESTED_IR_VERSION`, whether the callback API is present,
the IR node kinds of this Polars the engine does not know, whether `ARROWMETAL_METAL_ENGINE` is set,
and the header of the capability table (in a source checkout). It exits 1 when the callback API is
missing or the IR major differs, since then every plan stays with Polars.

The Parquet scan cases (212 tests) run fourteen scan shapes -- filters over every numeric dtype,
Polars' float total order, `!=`, String predicates, projections, group-by on one and two keys,
whole-file aggregates, sorts with nulls at both ends, top-k, a join with an in-memory frame and
`unique` -- over one 20,011-row dataset with nulls, NaN, -0.0 and infinities written by pyarrow
(snappy; uncompressed with a page index and no dictionary; ZSTD with v2 pages), by Polars and by
DuckDB; a filter, a sort and an aggregate over the flat columns of 30 nested fixtures from all three
writers; seventeen predicate-pushdown cases on every writer, each checking the answer, the filters
the reader was given and the row groups it skipped, including NaN under `>`, `>=` and `!=` (the
apache/arrow#51491 shape, also on the `pageindexnan__pa_constpage` fixture); every fallback reason
above; the open-file cache's reuse, invalidation and bounds; and footer null counts, including a file
whose statistics deny its nulls.

**The conformance grid.** `python/tests/engine_polars_grid.py` generates the cases instead of
choosing them: every shape the engine translates (filters, `select` and `with_columns` expressions,
`slice`, sorts in both directions with nulls at both ends and over two keys, top-k, group-by with
each aggregate with the column as the key and as the value, whole-frame aggregates, the four join
kinds, `unique`) over every dtype it carries (the eight integer widths, Float32, Float64, Boolean,
String, Date, Datetime in ms, us and ns and with a time zone, Duration in ms, us and ns, Time), with
no, 5%, 70% or all nulls, at 0, 1, 7, 1,000 and 100,000 rows, plus the special values (integer
extremes, NaN, infinities, subnormals, -0.0). Each case collects through
`MetalEngine(shapes="all", min_rows=0)` and through Polars and compares the frames bit for bit.
The sorts cover each direction with the nulls first and last, a stable sort in each direction
(`maintain_order=True`: tied rows, the two zeros and every NaN keep their input order, compared bit
for bit), two and three keys with a per-key null placement, a tie on one key falling through to the
next, and top-k as `sort().head()`, `top_k`, `bottom_k` and over two keys. In the run recorded in
`Benchmarks/results/engine_conformance_2026-10-02.csv` (the per-shape table in
`engine_conformance_2026-10-02_report.txt`): 15,376 cases, 15,096 identical on Metal, 33 within the
float-summation bound above, none different otherwise, and 247
where Polars' optimised plan had nothing to run (all of them sorts over 0 or 1 row, which Polars
leaves out). `python/tests/engine_report.py --engine polars` reruns it and prints the
per-shape table.

### To verify on your own machine

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
5. `out, timings = engine.profile(lf)` shows a `metal:` row, and
   `PYTHONPATH=python python -m arrowmetal.polars_engine check` prints the versions it was tested with.
6. Run `PYTHONPATH=python python Benchmarks/polars_engine_bench.py --sizes 2000000,50000000 --out a.csv`
   twice on a quiet machine, compare the two files, and read the `taken` and `rule` columns of the
   `MetalEngine default, cold` rows against the table above. `--crossover` with more sizes, then
   `python Benchmarks/polars_engine_crossover.py a.csv --print`, fits the table from your own run.
7. Read this section against `engine.last_report` on your machine.

---

## Numbers

These are tiers 1 and 2, and tier 4 over a Parquet file at the end; tier 4's measurements over
in-memory frames are in `Benchmarks/results/polars_engine_bench_2026-09-26-final3.csv` and
`Benchmarks/results/polars_engine_crossover_2026-09-26-final.csv` (see "Tier 4" above).

Apple M4 Max, macOS 26.6.2, polars 1.44.1 (16 threads), pyarrow 25.0.1, ArrowMetal 0.1.0. Best of 5
runs after a warm-up, one process, one data set. Every figure below is from
`Benchmarks/results/polars_bench_50000000_2026-09-07.txt`, except the String rows: those are ArrowMetal
0.2.0 with String columns handed over in Polars' view layout, best of 5,
`Benchmarks/results/polars_bench_strings_view_2026-09-26-quiet.txt` (run conditions in
`Benchmarks/results/bench_conditions_2026-09-26-quiet.txt`). Reproduce with:

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
| string `contains` (literal) | 671.1 ms / 670.8 CPU-ms | 38.1 ms (17.6x) | 290.4 ms (2.3x) | 6.0 ms (112.4x) |
| string `contains` (literal), `string_layout="offsets"` | 671.1 ms / 670.8 CPU-ms | 316.7 ms (2.1x) | | 6.3 ms (106.9x) |

The String hand-off of that column, 50M rows, in the same run:

| String hand-off | ms | CPU-ms |
|---|---:|---:|
| `pl.Series` -> Metal, view layout (0 bytes copied, 40 data buffers) | 30.0 | 27.7 |
| `pl.Series` -> Metal, offsets (`large_string`) | 309.0 | 308.9 |
| Metal -> `pl.Series`, view layout | 0.0 | 0.0 |

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
* **Strings** cross in Polars' own `Utf8View` layout: 50M rows hand over in 30.0 ms with no bytes
  copied, against 309.0 ms through `large_string` (offsets + bytes), which is a copy. Through the
  namespace `contains` is 17.6x with views and 2.1x with offsets; resident it is 112.4x (106.9x over
  offsets). The tier-2 plugin still asks Polars for `large_string` and is 2.3x. Keeping a string
  column resident (`s.arrowmetal.to_metal()`) takes the 38.1 ms call to 6.0 ms.

### Where each tier is worth using

| | Use it when |
|---|---|
| Tier 1, one call | The kernel is expensive relative to 400 MB of page mapping: group-by, sort, top-k, string search. Not a bare `sum`. |
| Tier 1, resident | You run several kernels over the same column. `to_metal()` once, then every kernel in the table above is 0.8-10.5 ms. |
| Tier 2 | The GPU op belongs inside a plan you want Polars to keep optimising -- scans, pushdown, and lazy composition still apply. |
| Tier 3 | Polars should do the IO and the reshaping and ArrowMetal should do one heavy pass at the end. |
| Tier 4 | You want Polars' own `collect()` and its answers, with the parts of the plan the GPU is measured ahead on (by default, from their measured crossovers: sorts, numeric-key inner, left and anti joins, `unique`, group-bys by their number of groups, sorts of Parquet files, and with a String column sorts, `unique` and the two-key group-by sum) run there. |

### Tier 4 over a Parquet file, 50M rows

`pl.scan_parquet(file)` under a filter, a group-by, an aggregate or a sort, over the 50,000,000-row,
8-column files of `Benchmarks/parquet_bench.py` (snappy, 1.65 GB, and uncompressed, 2.23 GB),
collected by Polars' in-memory and streaming engines and by `MetalEngine(shapes="all", min_rows=0)`.
"cold" clears the open-file cache before every run, so the engine opens and maps the file each time;
"warm" keeps it open between runs. Polars reads the file on every run; the file stays in the OS page
cache throughout. Best of 5, every engine result equal to Polars',
`Benchmarks/results/polars_engine_scan_2026-09-26-quiet.csv` (run conditions in
`Benchmarks/results/bench_conditions_2026-09-26-quiet.txt`).

| case | codec | Polars in-memory | Polars streaming | MetalEngine cold | MetalEngine warm | `MetalEngine()` default, cold |
|---|---|---:|---:|---:|---:|---:|
| (s1) filter `price > 500`, group-by `qty` (1,000 keys), sum + count | snappy | 98.7 ms | 45.6 ms | 101.9 ms | 91.3 ms | 99.7 ms (Polars) |
| | none | 87.6 ms | 36.1 ms | 47.2 ms | **31.6 ms** | 90.3 ms (Polars) |
| (s2) filter `id < 5,000,000` (45 of 50 row groups skipped), group-by, sum | snappy | 21.8 ms | 12.1 ms | 35.0 ms | 33.4 ms | 24.1 ms (Polars) |
| | none | 16.2 ms | 5.1 ms | 8.5 ms | 6.3 ms | 17.2 ms (Polars) |
| (s3) filter on two columns, sum + count | snappy | 14.0 ms | 13.7 ms | 21.0 ms | 13.9 ms | 14.7 ms (Polars) |
| | none | 13.8 ms | 13.1 ms | 13.3 ms | **5.9 ms** | 14.0 ms (Polars) |
| (s4) sort 2 columns by a Float64 key | snappy | 542.3 ms | 559.4 ms | **123.7 ms** | **116.3 ms** | **123.5 ms** (Metal) |
| | none | 562.0 ms | 563.2 ms | **73.7 ms** | **64.7 ms** | **75.0 ms** (Metal) |

* **The sort is ahead cold and warm**: 4.4x (snappy) and 7.6x (uncompressed) the faster Polars
  engine cold, 4.7x and 8.7x warm, and the default takes it (4.4x and 7.5x).
* **Cold, the other three are behind and to improve**, from 35.0 ms against 12.1 ((s2), snappy) to
  13.3 ms against 13.1 ((s3), uncompressed) (`vs_fastest_polars`). The cold runs include what the
  open-file cache removes: opening the file and handing the pages the query reads to Metal; a
  one-column read of these files is 6.9-12.2 ms through a fresh open and 3.2-7.6 ms through the cache
  ([PARQUET.md](PARQUET.md), "The open-file cache"). The default leaves them to Polars.
* **Warm**, over the uncompressed file (s3) is ahead of Polars' streaming engine (5.9 ms against
  13.1, 2.2x) and so is (s1) (31.6 ms against 36.1, 1.1x); (s2) is behind at 6.3 ms against 5.1.
  Over the Snappy file, which the GPU decompresses first ([PARQUET.md](PARQUET.md), "What the
  numbers say"), (s1) is behind at 91.3 ms against 45.6, (s2) at 33.4 against 12.1 and (s3) at 13.9
  against 13.7.
* **CPU time**: the warm engine runs cost 2.9-7.8 ms of process CPU, against 37.5-6733.8 ms for
  Polars (`cpu_ms`).

```
PYTHONPATH=python python Benchmarks/polars_engine_bench.py --scan-only --scan-rows 50000000 \
    --scan-codecs snappy,none --out scan.csv
```

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

**Chunking.** `am.from_polars` takes one Arrow array. A multi-chunk Series is rechunked once, which
does copy; `am.from_polars(s, rechunk=False)` raises instead, so the copy is never silent. The
`MetalEngine` hands a multi-chunk column of an in-memory frame over as its own chunks, which the
chunked import (`am_import_chunks`) copies straight into the final Metal buffers without Polars
concatenating them first; Categorical, Enum and nested columns are still concatenated by `to_arrow`.

**Order.** Group order is ArrowMetal's, not Polars': ascending by key for numeric, boolean,
temporal and decimal keys, first-seen for utf8 and binary, lexicographic in column order for
several keys. Polars' `group_by` promises no order at all, so sort both sides before comparing.
`s.arrowmetal.unique()` is **first-seen**, the order Polars' `unique(maintain_order=True)` gives,
and it keeps a null as one of the distinct values rather than dropping it. Sorts put nulls last in
**both** directions, where Polars' ascending default is nulls first -- pass `nulls_last=True` when
comparing.

**Strings.** A String or Binary column crosses in Polars' own `Utf8View` / `BinaryView` layout
(`to_arrow(compat_level=newest)`): its 16-byte views and data buffers are mapped into Metal without a
copy, and export hands them back to Polars the same way. `s.arrowmetal.to_metal().string_layout` says
`"view"`. The string kernels read the views directly (the list is `am.STRING_VIEW_KERNELS`); the
rest (`am.STRING_VIEW_CONVERTS`: `str_concat`, the splits, `match_like`, `strptime`, the
byte-counting pads, `utf8_zero_fill`, the byte slices and reversals, casts from strings, the file
writers) convert the column to offsets + bytes once, on the GPU, and keep that form, which
`string_layout` then reports as `"view (converted)"`. `am.from_polars(s, string_layout="offsets")`
asks Polars for `large_string` instead, the layout used before the kernels read views; the tier-2
plugin still uses that. `upper`/`lower` are Unicode's simple 1:1 case mapping over every script (the GPU
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

The tier-2 block at the end of the file is the hostile-input pass: the scalar operand against the
tier-1 bridge, nulls against native Polars, a sliced Series and a two-chunk one, an empty frame,
an all-set validity bitmap with no nulls, the twelve dtypes the numeric expressions refuse (each
one a Polars error carrying ArrowMetal's wording, never a pyo3 panic), and the expression inside
`group_by().agg()` and inside `collect(engine="streaming")`.
