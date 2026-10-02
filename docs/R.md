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
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift build -c release --product ArrowMetalC   # produces .build/release/libArrowMetalC.dylib
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
idx <- am_argsort(h)                                 # zero-based uint32 indices
top <- am_take(h, am_slice(idx, 0, 10))

g <- am_group_by(sample(c("north", "south"), 1e7, TRUE))
as.vector(as_arrow_array(g$keys()))                  # one row per group
as.vector(as_arrow_array(g$sum(h)))                  # grouped sum

src <- am_plan_source("sales", list(region = c("a", "b", "a"), amount = c(1, 2, 3)))
res <- am_plan_run('{"op":"limit","count":2,"input":{"op":"scan","source":"sales"}}', src)
as.vector(as_arrow_array(res$amount))                # back to plain R
```

## Sort options

`am_argsort()`, `am_sort()`, `am_top_k()` and `am_lexsort()` take Arrow's `null_placement`
(`"at_end"`, the default, or `"at_start"`) and a `float_order` (`"ieee"`, the default, `"total"` or
`"nan_largest"`):

```r
x <- c(2, NA, NaN, -0, 7)
am_argsort(x, descending = TRUE)                                  # 4 0 3 2 1: NA and NaN last
am_argsort(x, descending = TRUE, null_placement = "at_start",
           float_order = "total")                                 # 1 2 4 0 3
am_top_k(x, 2, null_placement = "at_start", float_order = "total")   # 1 2
am_lexsort(list(c(1, 1, 2), c(3, NA, 1)), descending = c(FALSE, TRUE),
           null_placement = c("at_end", "at_start"))              # 1 0 2
```

(Each is shown through `as.vector()`; `test-sort-options.R` and `test-chunks.R` run these lines.)
With the defaults each function runs the call it always ran: nulls last and `NaN` after `+Inf` in both directions, `-0`
tied with `0`. `"at_start"` puts the nulls first in either direction. `float_order = "ieee"` is Arrow
C++'s order: `-0` ties `0`, every `NaN` is one value, and the `NaN` rows sit next to the nulls in both
directions. `"total"` is IEEE 754 totalOrder, the order arrow-rs and Rust's `total_cmp` use:
`-NaN < -Inf < ... < -0 < 0 < ... < Inf < NaN`, and a descending sort is its exact mirror.
`"nan_largest"` is Polars' and NumPy's order: every `NaN` is one value above `+Inf` in both directions
(last ascending, first descending) and `-0` ties `0`. Integer, string and temporal columns ignore
`float_order`. `am_top_k(x, k, largest, ...)` is the first `k`
indices of `am_argsort(x, descending = largest, ...)`. In `am_lexsort()` each option is one value for
every key or a vector with one per key. A plan's `sort` key takes the same options as JSON:
`{"column": "x", "descending": true, "nulls": "first", "float_order": "total"}` ([ENGINE.md](ENGINE.md)).

## Chunked columns

`am_array_chunks()` imports a column held in chunks — an `arrow::ChunkedArray`, or a list of
Arrays of one type such as one column of several record batches — as one `am_array`, with no
concatenated copy first. `am_array()` keeps concatenating a ChunkedArray first, as it always has
(measured below, that is the faster path in R at 10M and 50M rows).

```r
ca <- arrow::chunked_array(c(1, NA), numeric(0), c(3, 4, 5))
h <- am_array_chunks(ca)
length(h); am_null_count(h); am_sum(h)    # 5, 1, 13
```

The column crosses the C Data Interface once, as an `ArrowArrayStream` of one-column record batches
(`arrow::as_record_batch_reader()`), so the R side does the same work for 10 chunks as for 1,000. The
shim reads the stream in C, moves each batch's column into one block of `ArrowArray` structs it owns
and releases the batch, and `am_import_chunks` copies each chunk's buffers straight into the final
GPU buffers, on the CPU cores in parallel, honouring each chunk's offset, length and validity. Each
chunk's release callback runs exactly once, by ArrowMetal or, for a chunk it did not take, by the
shim. One chunk is `am_array()`. A list of arrays becomes a ChunkedArray first, which checks that
the chunks share one type. A type the chunked import does not take (dictionary, nested, run-end
encoded, extension) is concatenated and imported, as `am_array()` always did.

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
give it its own allocation instead of a slice of a pool, so **a small column's buffers are not page
aligned and take the copying path** whatever their type. Where the changeover happens is allocator
behaviour — it differs by type and is not monotone in `n` — so no threshold is quoted here. The
table above is what was measured, at 1M and 10M; for anything else, run `am_buffer_alignment()` on
your own data rather than inferring. At small sizes the copy costs nothing worth measuring.

The copy costs **1.39 ms (median) for 80 MB** (10M float64; decimal MB, 76 MiB), about 57 GB/s.

## Timing

10,000,000 float64, no nulls.

**Method.** The benchmark is committed at `r/arrowmetal/inst/bench/timing.R` with its driver
`run.R`, so the table below can be re-derived rather than taken on trust:

```sh
ARROWMETAL_LIB=$PWD/.build/release/libArrowMetalC.dylib \
  Rscript r/arrowmetal/inst/bench/run.R 5 2
```

**2 replicates × 5 fresh R processes × `microbenchmark(times = 20)`**, each process building its
own data. Every expression is timed **two ways**, because they disagree by enough to move a
headline:

- **isolated** — each expression in its *own* `microbenchmark()` call, after one warm-up call of
  that expression. The most favourable measurement.
- **interleaved** — all expressions in *one* `microbenchmark()` call, so the runs are shuffled and
  each expression meets the cache and buffer-pool state the others leave behind. The conservative
  measurement, and the right one for a table whose rows are compared with each other.

`min` is the fastest of all 200 runs, `median` the median of the 10 per-process medians.
"resident" means the column is already an `am_array`, "import" means the timing starts from an
`arrow::Array` and includes the transfer. All variants were checked to give the same answer in the
same script.

`sum`:

| Method | isolated min / median | interleaved min / median |
|---|---:|---:|
| `am_sum(h)` resident | 0.52 / 0.82 ms | 1.37 / 1.70 ms |
| `arrow::call_function("sum", a)` | 1.09 / 1.16 ms | 1.12 / 1.31 ms |
| `am_sum(a)` import + sum | 3.29 / 6.36 ms | 2.75 / 3.20 ms |
| `sum(x)` base R | 11.15 / 11.60 ms | 11.15 / 11.61 ms |

**Resident `sum` is not separable from arrow's, and no verdict is claimed for it.** It measures
between **0.57 and 1.54 ms depending on the process**. In the isolated mode: the per-process medians
were 0.57, 0.59, 0.60, 0.60, 0.78 in one replicate and 0.86, 1.38, 1.38, 1.39, 1.54 in the next,
while arrow stayed tight at 1.14–1.27 across all ten. One replicate makes ArrowMetal look twice as
fast, the next makes it look the other way; the two are the same speed to within the noise of this
measurement.

**`sum` including the import is a row where arrow is ahead, in every replicate**: 3.20 ms against
arrow's 1.31 ms interleaved, **2.4×**. A sum is one bandwidth-bound pass over 80 MB (decimal MB;
76 MiB) with no arithmetic to hide the transfer behind.

`filter` (`x > 0.5`, about 5M rows out):

| Method | isolated min / median | interleaved min / median |
|---|---:|---:|
| `am_filter(h, am_compare(h, ">", 0.5))` resident | 1.09 / 1.19 ms | 1.12 / 1.58 ms |
| `am_filter(a, am_compare(a, ">", 0.5))` import + filter | 4.19 / 11.79 ms | 3.95 / 4.49 ms |
| `arrow` `greater` then `filter` | 20.80 / 22.27 ms | 20.84 / 21.92 ms |
| `x[x > 0.5]` base R | 36.12 / 37.57 ms | 34.79 / 37.88 ms |

ArrowMetal is ahead on `filter`, but **the multiple depends on how it is measured**: about **14×**
against arrow resident and about **4.9×** including the import, on the conservative interleaved
medians. The resident row ranges 1.12–1.64 ms across processes here and independent runs on the
same machine have put it above 2 ms, which would make it nearer 9×. Treat it as **roughly an order
of magnitude, not a precise multiple**.

The `import` rows are the one place isolated timing is *worse* than interleaved (11.79 ms against
4.49 ms for filter): twenty imports back to back give the buffer pool no chance to recycle.

`am_array()` alone on the same column: **1.32 ms min, 1.39 ms median** interleaved (80 MB, about
57 GB/s).

arrow's kernels are multi-threaded and base R's are not, so a comparison against base R is not a
per-core figure. Timings on a loaded machine are noise; these ran on an idle machine.

### Chunked columns against concatenating first

`Rscript inst/bench/chunks.R`: a column in chunks, each chunk its own Array, imported with
`am_array_chunks()` against `am_array(do.call(arrow::concat_arrays, chunks))`, which is what
`am_array()` of a ChunkedArray does. M4 Max, 2026-09-29, R 4.5.3, arrow 25.0.0; three rounds, each row
warmed for 100 ms and then timed 10 times; best of the three rounds, the median of the per-round
medians in parentheses; CPU is process CPU time per call. Source:
`Benchmarks/results/bindings_chunked_import_2026-09-29.csv`.

An `am_array` is freed when R's collector finalizes it. With `gc()` run before every timed call
(outside the timed region), each call starts with the previous call's handle and GPU memory gone,
as a program that releases its columns sees it:

| Column | Rows | Chunks | `concat_arrays` + `am_array()` | `am_array_chunks()` | CPU ms (concatenate / chunked) |
|---|---:|---:|---:|---:|---:|
| float64, 10% null | 10,000,000 | 153 | 1.82 (1.89) ms | 1.20 (1.31) ms | 2.1 / 8.4 |
| float64, 10% null | 10,000,000 | 10 | 1.47 (1.51) ms | 1.12 (1.24) ms | 1.7 / 8.0 |
| float64, 10% null | 50,000,000 | 763 | 8.36 (9.03) ms | 3.68 (4.01) ms | 10.5 / 34.6 |
| float64, 10% null | 50,000,000 | 50 | 6.02 (6.12) ms | 3.07 (3.35) ms | 7.2 / 37.0 |
| int64 | 10,000,000 | 153 | 1.59 (1.66) ms | 1.09 (1.23) ms | 1.8 / 6.9 |
| int64 | 10,000,000 | 10 | 1.26 (1.31) ms | 0.97 (1.05) ms | 1.3 / 6.7 |
| int64 | 50,000,000 | 763 | 7.74 (8.42) ms | 3.37 (3.81) ms | 8.6 / 32.3 |
| int64 | 50,000,000 | 50 | 5.57 (5.71) ms | 2.69 (2.98) ms | 5.9 / 30.4 |

Without the `gc()` calls, the handles of earlier calls hold their GPU memory until the collector
runs, and every chunked import writes into newly allocated GPU memory, while `arrow`'s concatenation
is page aligned and borrowed. There the chunked import is behind at 10M rows (see To improve below):

| Column | Rows | Chunks | `concat_arrays` + `am_array()` | `am_array_chunks()` | CPU ms (concatenate / chunked) |
|---|---:|---:|---:|---:|---:|
| float64, 10% null | 10,000,000 | 153 | 2.87 (3.52) ms | 5.15 (5.71) ms | 4.2 / 93.8 |
| float64, 10% null | 10,000,000 | 10 | 2.37 (4.89) ms | 4.57 (5.03) ms | 5.0 / 81.2 |
| float64, 10% null | 50,000,000 | 763 | 13.42 (14.34) ms | 7.23 (12.44) ms | 15.3 / 119.4 |
| float64, 10% null | 50,000,000 | 50 | 11.31 (15.78) ms | 7.70 (86.55) ms | 16.2 / 185.4 |
| int64 | 10,000,000 | 153 | 4.51 (5.22) ms | 4.36 (5.68) ms | 5.6 / 90.8 |
| int64 | 10,000,000 | 10 | 4.32 (5.12) ms | 4.85 (5.59) ms | 5.6 / 93.9 |
| int64 | 50,000,000 | 763 | 16.96 (21.79) ms | 9.73 (10.27) ms | 21.3 / 133.8 |
| int64 | 50,000,000 | 50 | 10.76 (11.72) ms | 10.12 (10.83) ms | 12.0 / 151.0 |

`am_array()` of a ChunkedArray therefore keeps concatenating first, as it did.

### The existing calls, before and after

`Rscript inst/bench/overhead.R` times the calls that existed before, installed from the previous
commit and from this one against the same `libArrowMetalC.dylib`, in four alternating rounds (each row
warmed for 100 ms, a 500 ms idle and one call timed on its own, then 30 calls); best of the four
rounds, the median of the per-round medians in parentheses. At 10M rows:

| Call | Previous commit | This commit |
|---|---:|---:|
| `am_array()`, int64 Array | 0.011 (0.029) ms | 0.011 (0.015) ms |
| `am_array()`, float64 Array with 10% nulls | 4.15 (4.82) ms | 4.15 (4.86) ms |
| `am_array()`, one-chunk ChunkedArray | 3.76 (4.50) ms | 3.91 (4.59) ms |
| `am_argsort(x)`, float64 with 10% nulls | 6.47 (7.74) ms | 6.48 (7.92) ms |
| `am_argsort(x, TRUE)`, int64 | 3.74 (5.46) ms | 4.29 (5.28) ms |
| `am_sort(x)`, float64 with 10% nulls | 7.40 (8.07) ms | 7.26 (9.50) ms |

No row at 1,000, 1,000,000 or 10M rows was slower in both best and median. Every row, with
first-call-after-idle and CPU time: `Benchmarks/results/bindings_call_overhead_2026-09-29_summary.csv`.

### To improve

- `am_array_chunks()` in a loop that leaves its handles to R's collector: at 10M rows it takes 0.97x
  to 1.93x the time of `concat_arrays` + `am_array()` (best of three rounds; int64 in 153 chunks
  4.36 against 4.51 ms, float64 in 10 chunks 4.57 against 2.37 ms), and at 50M rows float64 in 50
  chunks has a median of 86.5 ms against 15.8 ms. It uses 81 to 185 CPU-ms per call there, against
  7 to 37 with `gc()` between calls.

## Covered

| Area | Functions |
|---|---|
| Import / export | `am_array()`, `am_array_chunks()` (a ChunkedArray or a list of arrays, `am_import_chunks`), `as_arrow_array()`, `as.vector()`, `length()`, `am_null_count()`, `am_format()` |
| Reductions | `am_sum()`, `am_min()`, `am_max()`, `am_mean()` |
| Element-wise | `am_compare()` — `==`, `!=`, `<`, `<=`, `>`, `>=`, against a scalar or a column |
| Selection | `am_filter()`, `am_take()`, `am_slice()` |
| Sorting | `am_argsort()`, `am_sort()`, `am_top_k()`, `am_lexsort()`, each with `null_placement` and `float_order` |
| Grouping | `am_group_by()` over any number of key columns of any supported type, with `$sum $min $max $mean $count $count_all $first $last $product $var $sd $median $quantile` and `$agg()` for the remaining `am_group_agg_ex` ops |
| Query engine | `am_plan_source()`, `am_plan_run()`, `am_plan_explain()` — the full JSON plan grammar |
| Environment | `am_available()`, `am_load_error()`, `am_lib_path()`, `am_version()`, `am_device_name()`, `am_buffer_alignment()` |

Types exercised by the tests: float64, float32, int32, int64, boolean and utf8; sliced views of
float64, boolean and utf8 at offsets 0, 1 and 3, plus a slice of a slice and selection on a sliced
column; and multi-chunk, single-chunk and empty ChunkedArrays.

## Not covered

The binding resolves 42 of the ABI's 283 entry points (36 it needs, and 6 newer ones it uses when the
loaded library has them). Not wrapped, and reachable only from
Python or Swift for now:

- arithmetic (`am_arith_*`, `am_unary`, `am_binary`, checked variants), casts (`am_cast`),
  boolean logic and Kleene logic;
- every string kernel (`am_str_*`, `am_string_*`, `am_regex`, `am_split`), temporal kernels,
  decimal, list, struct, map and extension types;
- `am_rank`, `am_unique`, `am_value_counts`, `am_is_in`,
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
  fractional index or one outside `[0, 2^32)` with an error rather than coercing it to `NA`.
- **Index arrays are uint32** (`am_argsort()`, `am_top_k()`, `am_lexsort()`, the ranks). R has no
  unsigned 32-bit type, so they read back the way `arrow` reads a uint32 array: an R `integer` vector
  when every value is at most 2^31 - 1, a `double` vector when one is larger (exact up to 2^32 - 1).
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
- **By default sorts put nulls and `NaN` last in both directions**, so a descending sort is not the
  exact reverse of an ascending one; `null_placement = "at_start"` and `float_order = "total"`
  change that ([Sort options](#sort-options)).
- **macOS on Apple silicon only.** The dylib is not included in the package.

## Tests

`src/arrowmetal.h` and `src/arrow_abi.h` are copies of the repository's `include/` headers; refresh them (`cp include/arrowmetal.h r/arrowmetal/src/`) whenever the header changes, or `python/tests/test_header_copies.py` and `test-header-copy.R` fail.

88 `test_that()` blocks in the sources (89 as testthat runs them: the one in test-dispatch.R runs once per attach order), 920 expectations, against base R and against `arrow`'s own
kernels on the same data: nulls, all-null and empty columns, sliced input at three offsets, lengths
of 1, 33, 1024, 65537 and 1,000,001 (crossing a threadgroup boundary), one group per row and one group for
everything, int64 above 2^53, float32 accumulation, and every documented error path.

```sh
ARROWMETAL_LIB=$PWD/.build/release/libArrowMetalC.dylib \
  Rscript -e 'testthat::test_local("r/arrowmetal")'
```
