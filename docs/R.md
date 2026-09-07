# R

`r/arrowmetal` is an R package that runs ArrowMetal's GPU kernels on columns held by the
[`arrow`](https://arrow.apache.org/docs/r/) R package. Columns cross through the Arrow C Data
Interface; the package links against nothing at build time and opens `libArrowMetalC.dylib` with
`dlopen` when it loads.

Every number on this page was measured in one session on an idle Apple M4 Max (16 cores, 64 GB
unified memory), macOS 26.6.2, R 4.5.3, arrow 25.0.0, ArrowMetal 0.1.0.

## Install

You need the dylib first:

```sh
swift build -c release          # produces .build/release/libArrowMetalC.dylib
```

Then either:

```sh
R CMD INSTALL r/arrowmetal
```

```r
remotes::install_local("r/arrowmetal")
```

The package's `configure` script looks for `../../.build/release/libArrowMetalC.dylib` relative to
the package source and compiles the absolute path it finds into the shim, so an in-tree install
needs nothing else. Otherwise, point `ARROWMETAL_LIB` at the dylib:

```sh
export ARROWMETAL_LIB=/path/to/libArrowMetalC.dylib
```

The search order at load is `ARROWMETAL_LIB`, then the path `configure` recorded, then
`../../.build/release/libArrowMetalC.dylib` relative to the working directory. When none of them
opens, the package still loads and every compute call raises an error listing all three paths and
why each failed; `am_available()`, `am_load_error()` and `am_lib_path()` report the same thing
without raising.

A conda-built R names its own compiler in `Makeconf` (here
`arm64-apple-darwin20.0.0-clang`), which lives in the environment's `bin` but is only on `PATH`
once the environment is activated. Calling such an R by absolute path without activating it fails
with `sh: arm64-apple-darwin20.0.0-clang: command not found`. Either put that `bin` on `PATH`, or
set `CC = clang` in `~/.R/Makevars` to use Xcode's clang instead; the shim compiles clean under
both (conda LLVM 23.1.0 and Apple clang 21.0.0), and the only non-standard C it uses is
`__typeof__`, which both provide.

## Example

```r
library(arrowmetal)
library(arrow)

x  <- Array$create(runif(1e7))                       # an arrow Array
h  <- am_array(x)                                    # onto the GPU
am_sum(h); am_min(h); am_max(h); am_mean(h)          # scalar reductions, nulls skipped

hits <- am_filter(h, am_compare(h, ">", 0.5))        # boolean mask, then gather
sorted <- am_sort(h, descending = TRUE)              # stable radix sort, nulls last
idx <- am_argsort(h)                                 # zero-based int32 indices
top <- am_take(h, am_slice(idx, 0, 10))

g <- am_group_by(sample(c("north", "south"), 1e7, TRUE))
as.vector(as_arrow_array(g$keys()))                  # one row per group
as.vector(as_arrow_array(g$sum(h)))                  # grouped sum

src <- am_plan_source("sales", list(region = c("a", "b", "a"), amount = c(1, 2, 3)))
res <- am_plan_run('{"op":"limit","count":2,"input":{"op":"scan","source":"sales"}}', src)
as.vector(as_arrow_array(res$amount))                # back to plain R
```

## The copy rule

**Out is always copy-free.** `as_arrow_array()` hands `arrow` a C Data Interface array pointing at
the Metal buffers; its release callback keeps them alive, so the `am_array` handle may go out of
scope first.

**In is copy-free when the producer's buffers are page aligned, and one copy otherwise.** What
that means for arrow R, measured with `am_buffer_alignment()` (page size 16,384 bytes):

| Buffer | 1M elements | 10M elements |
|---|---|---|
| `Array$create(<R double>)` values | offset 48, **not aligned** (20/20 trials) | offset 48, **not aligned** (20/20 trials) |
| `Array$create(<R integer>)` values | offset 48, **not aligned** | offset 48, **not aligned** |
| int64 values (`type = int64()`) | offset 0, **page aligned** | offset 0, **page aligned** |
| validity bitmap of a column with nulls | offset 0, **page aligned** | offset 0, **page aligned** |
| a buffer arrow allocates (`$cast()`, string offsets and data) | offset 0, **page aligned** | not measured |
| an mmapped Arrow IPC file | offset 0, **page aligned** | not measured |

arrow R **borrows** an R double or integer vector instead of copying it — `Array$create(x)` twice
on the same `x` returns the identical buffer address — and R's vector data starts 48 bytes past a
`malloc` block that is page aligned at these sizes, because R's vector header is 48 bytes on
64-bit. So **at 1M and 10M elements** a float64 or int32 column that came from an R vector takes
the copying path in, while an int64 column, a validity bitmap, a cast result, a string column and
anything read from a file take the copy-free path.

That size qualifier matters: a buffer is page aligned only when it is large enough for arrow to
give it its own allocation instead of a slice of a pool. Spot-checked, an int64 or string column is
page aligned from about a thousand elements and a validity bitmap from about a hundred thousand (a
bitmap for a thousand rows is 125 bytes); at `n = 3` nothing is aligned at all — int64 values
landed at offset 128 in the page, string buffers at 256 and 384, a validity bitmap at 192. Small
columns always take the copying path, at a cost too small to measure.

The copy costs **1.42 ms (median) for 80 MB** (10M float64), about 56 GB/s.

## Timing

10,000,000 float64, no nulls.

**Method:** **5 fresh R processes**, each building its own data, calling every expression once as a
warm-up, then `microbenchmark(times = 20)` — 100 timed runs per method in five independent
processes. The tables give the **minimum across all 100** and the **median of the five per-process
medians**. A single best-of-5 inside one process is not stable at this size: an earlier run of that
shape put resident `filter` at 1.22 ms, which does not reproduce, so the median is the number to
quote and the minimum is the floor. "resident" means the column is already an `am_array`, "import"
means the timing starts from an `arrow::Array` and includes the transfer. All variants were checked
to produce the same answer in the same script.

`sum`:

| Method | min | median |
|---|---:|---:|
| `am_sum(h)` resident | **0.86 ms** | 1.40 ms |
| `arrow::call_function("sum", a)` | 1.14 ms | **1.26 ms** |
| `am_sum(a)` import + sum | 2.05 ms | 3.29 ms |
| `sum(x)` base R | 11.18 ms | 11.96 ms |

**ArrowMetal loses `sum`**: on the median, **1.11× slower** than arrow resident and **2.6× slower**
with the import counted. On its single fastest run it edges arrow (0.86 ms against 1.14 ms), but
that does not hold across processes. A sum is one pass over 80 MB with no arithmetic to speak of,
so it is bounded by memory bandwidth the 16-thread CPU kernel already saturates.

`filter` (`x > 0.5`, about 5M rows out):

| Method | min | median |
|---|---:|---:|
| `am_filter(h, am_compare(h, ">", 0.5))` resident | **0.78 ms** | **1.51 ms** |
| `am_filter(a, am_compare(a, ">", 0.5))` import + filter | 3.74 ms | 4.65 ms |
| `arrow` `greater` then `filter` | 21.29 ms | 22.40 ms |
| `x[x > 0.5]` base R | 36.62 ms | 39.34 ms |

ArrowMetal wins `filter` **14.9× against arrow resident and 4.8× including the import**, on the
medians.

arrow's kernels are multi-threaded and base R's are not, so a comparison against base R is not a
per-core figure. Timings on a loaded machine are noise; these ran on an idle machine.

## Covered

| Area | Functions |
|---|---|
| Import / export | `am_array()`, `as_arrow_array()`, `as.vector()`, `length()`, `am_null_count()`, `am_format()` |
| Reductions | `am_sum()`, `am_min()`, `am_max()`, `am_mean()` |
| Element-wise | `am_compare()` — `==`, `!=`, `<`, `<=`, `>`, `>=`, against a scalar or a column |
| Selection | `am_filter()`, `am_take()`, `am_slice()` |
| Sorting | `am_argsort()`, `am_sort()` |
| Grouping | `am_group_by()` over any number of key columns of any supported type, with `$sum $min $max $mean $count $count_all $first $last $product $var $sd $median $quantile` and `$agg()` for the remaining `am_group_agg_ex` ops |
| Query engine | `am_plan_source()`, `am_plan_run()`, `am_plan_explain()` — the full JSON plan grammar |
| Environment | `am_available()`, `am_load_error()`, `am_lib_path()`, `am_version()`, `am_device_name()`, `am_buffer_alignment()` |

Types exercised by the tests: float64, float32, int32, int64, boolean and utf8; sliced views of
float64, boolean and utf8 at offsets 0, 1 and 3, plus a slice of a slice and selection on a sliced
column; and multi-chunk, single-chunk and empty ChunkedArrays.

## Not covered

The binding resolves 34 of the ABI's 200-plus entry points. Not wrapped, and reachable only from
Python or Swift for now:

- arithmetic (`am_arith_*`, `am_unary`, `am_binary`, checked variants), casts (`am_cast`),
  boolean logic and Kleene logic;
- every string kernel (`am_str_*`, `am_string_*`, `am_regex`, `am_split`), temporal kernels,
  decimal, list, struct, map and extension types;
- `am_top_k`, `am_lexsort`, `am_rank`, `am_unique`, `am_value_counts`, `am_is_in`,
  `am_cumulative`, `am_window`, `am_hash64`, `am_if_else`, `am_coalesce`, `am_fill_null`;
- joins (`am_join`), the dense-key `am_group_by`, `am_query` (the fused expression compiler),
  Parquet (`am_parquet_*`), the streaming engine (`am_stream_*`), C Device interop
  (`am_import_device` / `am_export_device`), batching (`am_batch_begin` / `am_batch_end`) and
  resident mode.

There is also no dplyr backend and no `RecordBatch`/`Table` surface: everything is per-column.

## Limits

- **int64.** R's integer is 32-bit and its double is 64-bit; there is no native 64-bit integer.
  An int64 Arrow column is carried end to end, but a scalar result comes back as a double, exact
  only to 2^53. `am_sum(x, integer64 = TRUE)` returns an exact `bit64::integer64` instead, and a
  plain call warns when the result is outside 2^53. `arrow` itself follows the same convention:
  `Scalar$as_vector()` on an int64 gives an R `integer` when it fits and a `bit64::integer64`
  when it does not. Note that `as.vector()` on a `bit64::integer64` strips the class and
  reinterprets the bit pattern as a double (2474723182 becomes 1.22e-314); use `as.numeric()`.
- **The `arrow` package is a hard dependency.** It builds every column and reads every result
  back; `am_array(c(1, 2, 3))` calls `arrow::Array$create()` for you.
- **Zero-based indices** in `am_argsort()`, `am_take()` and `am_slice()`, following Arrow rather
  than R. `am_argsort(x)` matches `order(x, na.last = TRUE) - 1L`. `am_take()` rejects a
  fractional index or one outside `[0, 2^31)` with an error rather than coercing it to `NA`.
- **`as_arrow_array()` is `arrow`'s own generic.** `arrow` exports
  `as_arrow_array(x, ..., type = NULL)` with eight methods; this package registers a ninth for
  `am_array` and re-exports the generic unchanged, rather than defining a second one. A second
  generic of that name would mask arrow's (breaking its methods) or be masked by it (never
  dispatching for `am_array`), depending on the order the packages are attached.
- **An `NA` scalar in `am_compare()`** returns an all-null boolean mask, matching base R
  (`c(1, 5) > NA`) and `arrow`'s `greater` kernel. The ABI scalar is a raw value with no validity
  flag, so this case is resolved in R and never reaches the GPU. `NaN` is a value, not a null, and
  does go to the kernel.
- **Group order is not first-seen order**: ascending by key for numeric, boolean, temporal and
  decimal columns (nulls last), first-seen for strings and binary, lexicographic in column order
  for several columns. Label rows with `$keys()`.
- **Sorts put nulls and `NaN` last in both directions**, so a descending sort is not the exact
  reverse of an ascending one.
- **macOS on Apple silicon only.** The dylib is not shipped inside the package.

## Tests

266 testthat tests, all passing, against base R and against `arrow`'s own kernels on the same
data: nulls, all-null and empty columns, sliced input at three offsets, lengths of 1, 33, 1024,
65537 and 1,000,001 (crossing a threadgroup boundary), one group per row and one group for
everything, int64 above 2^53, float32 accumulation, and every documented error path.
`R CMD check --no-manual` is clean: 0 errors, 0 warnings, 0 notes.

```sh
ARROWMETAL_LIB=$PWD/.build/release/libArrowMetalC.dylib \
  Rscript -e 'testthat::test_local("r/arrowmetal")'
```
