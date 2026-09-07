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
64-bit. So a float64 or int32 column that came from an R vector takes the copying path in; an
int64 column, a validity bitmap, a cast result, a string column and anything read from a file take
the copy-free path.

The copy costs **1.51 ms for 80 MB** (10M float64), about 53 GB/s.

## Timing

10,000,000 float64, no nulls. One warm-up call of each expression, then
`microbenchmark(times = 5)`; the figure is the **minimum** wall time of the five. "resident" means
the column is already an `am_array`, "import" means the timing starts from an `arrow::Array` and
includes the transfer. All variants were checked to produce the same answer in the same script.

`sum`:

| Method | Best of 5 | vs base R |
|---|---:|---:|
| `arrow::call_function("sum", a)` | **1.20 ms** | 10.4× |
| `am_sum(h)` resident | 1.39 ms | 9.0× |
| `am_sum(a)` import + sum | 3.58 ms | 3.5× |
| `sum(x)` base R | 12.54 ms | 1.0× |

**ArrowMetal loses `sum`.** 1.39 ms against arrow's 1.20 ms resident (1.16× slower), 3.58 ms
against 1.20 ms with the import counted (3.0× slower). A sum is one pass over 80 MB with no
arithmetic to speak of, so it is bounded by memory bandwidth the 16-thread CPU kernel already
saturates.

`filter` (`x > 0.5`, about 5M rows out):

| Method | Best of 5 | vs base R |
|---|---:|---:|
| `am_filter(h, am_compare(h, ">", 0.5))` resident | **1.22 ms** | 32.7× |
| `am_filter(a, am_compare(a, ">", 0.5))` import + filter | 4.20 ms | 9.5× |
| `arrow` `greater` then `filter` | 24.15 ms | 1.7× |
| `x[x > 0.5]` base R | 40.03 ms | 1.0× |

ArrowMetal wins `filter` 19.8× against arrow resident, 5.8× including the import.

arrow's kernels are multi-threaded and base R's are not, so "vs base R" is not a per-core figure.
Timings on a loaded machine are noise; this ran once, on an idle machine, and has not been
re-measured.

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

Types exercised by the tests: float64, float32, int32, int64, boolean, utf8, and sliced views of
any of them.

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
  than R. `am_argsort(x)` matches `order(x, na.last = TRUE) - 1L`.
- **Group order is not first-seen order**: ascending by key for numeric, boolean, temporal and
  decimal columns (nulls last), first-seen for strings and binary, lexicographic in column order
  for several columns. Label rows with `$keys()`.
- **Sorts put nulls and `NaN` last in both directions**, so a descending sort is not the exact
  reverse of an ascending one.
- **macOS on Apple silicon only.** The dylib is not shipped inside the package.

## Tests

179 testthat tests, all passing, against base R and against `arrow`'s own kernels on the same
data: nulls, all-null and empty columns, sliced input at three offsets, lengths of 1, 33, 1024,
65537 and 1,000,001 (crossing a threadgroup boundary), one group per row and one group for
everything, int64 above 2^53, float32 accumulation, and every documented error path.
`R CMD check --no-manual` is clean: 0 errors, 0 warnings, 0 notes.

```sh
ARROWMETAL_LIB=$PWD/.build/release/libArrowMetalC.dylib \
  Rscript -e 'testthat::test_local("r/arrowmetal")'
```
