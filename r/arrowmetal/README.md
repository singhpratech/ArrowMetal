# arrowmetal for R

Apache Arrow compute on Apple silicon GPUs, from R, through the `arrow` package.

```r
library(arrowmetal)
library(arrow)

x <- Array$create(runif(1e7))
am_sum(x)                                   # one GPU pass
as.vector(as_arrow_array(am_filter(x, am_compare(x, ">", 0.5))))
```

Full documentation, including installation, is in [`docs/R.md`](../../docs/R.md).

## What is wrapped

| Area | R |
|---|---|
| Import / export | `am_array()`, `as_arrow_array()`, `as.vector()`, `length()`, `am_null_count()`, `am_format()` |
| Reductions | `am_sum()`, `am_min()`, `am_max()`, `am_mean()` |
| Element-wise | `am_compare()` (scalar or column, six operators) |
| Selection | `am_filter()`, `am_take()`, `am_slice()` |
| Sorting | `am_argsort()`, `am_sort()` |
| Grouping | `am_group_by()` with 13 named aggregates plus `$agg()` for the rest |
| Query engine | `am_plan_source()`, `am_plan_run()`, `am_plan_explain()` |
| Environment | `am_available()`, `am_load_error()`, `am_lib_path()`, `am_version()`, `am_device_name()`, `am_buffer_alignment()` |

This binding resolves 34 of the C ABI's 222 entry points. Everything not in the table above —
string kernels, temporal kernels, casts, arithmetic, joins, Parquet, the streaming engine — is
reachable from Python or Swift but not yet from R.

## The copy rule

**Out is always copy-free.** `as_arrow_array()` hands the `arrow` package a C Data Interface array
that points straight at the Metal buffers and keeps them alive with its release callback.

**In is copy-free when the producer's buffers are page aligned, and one copy otherwise.**

Measured this session with `am_buffer_alignment()`, R 4.5.3, arrow 25.0.0, macOS on an M4 Max,
page size 16,384 bytes:

| Producer | 1M elements | 10M elements |
|---|---|---|
| `Array$create(<R double vector>)` values buffer | offset 48 in the page — **not aligned**, 20/20 trials | offset 48 — **not aligned**, 20/20 trials |
| `Array$create(<R integer vector>)` values buffer | offset 48 — **not aligned** | offset 48 — **not aligned** |
| int64 values buffer (`type = int64()`) | offset 0 — **page aligned** | offset 0 — **page aligned** |
| validity buffer of a column with nulls | offset 0 — **page aligned** | offset 0 — **page aligned** |
| a buffer arrow allocates itself (`$cast()`, string offsets and data) | offset 0 — **page aligned** | not measured |
| an mmapped Arrow IPC file | offset 0 — **page aligned** | not measured |

The mechanism: arrow R **borrows** an R double or integer vector rather than copying it — calling
`Array$create(x)` twice on the same `x` gives the identical buffer address — and R's vector data
starts 48 bytes past a `malloc` block that is itself page aligned at these sizes, because R's
vector header is 48 bytes on 64-bit. Anything arrow allocates for itself (an int64 column, which R
cannot hold; a validity bitmap; a cast result; string buffers) gets its own large allocation and
lands page aligned.

So, **at the 1M and 10M sizes measured here**: a double or int32 column that came from an R vector
is copied on the way in; an int64 column, a validity bitmap, a cast result, a string column and
anything read from a file are not.

The size qualifier is load-bearing. A buffer lands page aligned only when it is large enough for
arrow to give it its own allocation rather than a slice of a pool, so **a small column's buffers
are not page aligned and take the copying path** whatever their type. Where the changeover happens
is allocator behaviour, it varies by type and is not monotone in `n`, so no threshold is quoted
here; the table above is what was measured, at 1M and 10M. Use `am_buffer_alignment()` on your own
data rather than inferring. At small sizes the copy costs nothing worth measuring anyway.

The copy is measured below at 1.39 ms (median) for 80 MB.

## Measured timing

Apple M4 Max (16 cores, 64 GB unified memory), macOS 26.6.2, R 4.5.3, arrow 25.0.0,
ArrowMetal 0.1.0, idle machine. 10,000,000 float64, no nulls.

**Method.** The script is committed at [`inst/bench/timing.R`](inst/bench/timing.R) with its driver
[`inst/bench/run.R`](inst/bench/run.R), so the table can be re-derived:

```sh
ARROWMETAL_LIB=$PWD/.build/release/libArrowMetalC.dylib \
  Rscript r/arrowmetal/inst/bench/run.R 5 2
```

**2 replicates × 5 fresh R processes × `microbenchmark(times = 20)`**, each process building its own
data. Every expression is timed **two ways**, because they do not agree and the difference is large
enough to move a headline:

- **isolated** — each expression gets its *own* `microbenchmark()` call, after one warm-up call of
  that expression. The most favourable measurement.
- **interleaved** — every expression in *one* `microbenchmark()` call, so the runs are shuffled
  together and each expression meets the cache and buffer-pool state the others leave behind. The
  conservative measurement, and the one the table below uses, since its rows are compared with each other.

`min` is the fastest of all 200 runs; `median` is the median of the 10 per-process medians.
"resident" means the column is already an `am_array`; "import" means the timing starts from an
`arrow::Array` and includes the transfer.

### `sum`

| Method | isolated min / median | interleaved min / median |
|---|---:|---:|
| `am_sum(h)` — resident | 0.52 / 0.82 ms | 1.37 / 1.70 ms |
| `arrow::call_function("sum", a)` | 1.09 / 1.16 ms | 1.12 / 1.31 ms |
| `am_sum(a)` — import + sum | 3.29 / 6.36 ms | 2.75 / 3.20 ms |
| `sum(x)` — base R double vector | 11.15 / 11.60 ms | 11.15 / 11.61 ms |

**Resident `sum` is not separable from arrow's.** It measures anywhere from **0.57 to 1.54 ms
depending on the process**. In the isolated mode: the per-process medians came out 0.57, 0.59, 0.60,
0.60, 0.78 in one replicate and 0.86, 1.38, 1.38, 1.39, 1.54 in the next, straddling arrow's, which
is tight at 1.14 to 1.27 across all ten. So it lands either side of arrow's 1.16 ms isolated median
depending on which replicate you run, and **no verdict is claimed for that row.**

**`sum` including the import is a row where arrow is ahead, in every replicate**: 3.20 ms against
arrow's 1.31 ms interleaved, **2.4×**. A sum is one bandwidth-bound pass over 80 MB
(decimal MB; 76 MiB) with no arithmetic to hide the transfer behind.

### `filter` (`x > 0.5`, ~5M rows out)

| Method | isolated min / median | interleaved min / median |
|---|---:|---:|
| `am_filter(h, am_compare(h, ">", 0.5))` — resident | 1.09 / 1.19 ms | 1.12 / 1.58 ms |
| `am_filter(a, am_compare(a, ">", 0.5))` — import + filter | 4.19 / 11.79 ms | 3.95 / 4.49 ms |
| `arrow` `greater` then `filter` | 20.80 / 22.27 ms | 20.84 / 21.92 ms |
| `x[x > 0.5]` — base R | 36.12 / 37.57 ms | 34.79 / 37.88 ms |

ArrowMetal is ahead on `filter`, but **the exact multiple depends on how you measure**: on the
conservative interleaved medians it is **about 14× against arrow resident and about 4.9× including
the import**. The resident row alone ranges 1.12–1.64 ms across processes here, and independent
runs on the same machine have put it above 2 ms, which would make it nearer 9×. Read it as
**roughly an order of magnitude, not a precise multiple**. All four produce the same answer
(checked in the same script: identical sum, identical output length).

Note the `import` rows are the one place isolated timing is *worse* than interleaved (11.79 ms
against 4.49 ms for filter): repeating an import twenty times with nothing in between gives the
buffer pool no chance to recycle, which is exactly the effect the interleaved mode exists to avoid
reporting.

### Import alone

`am_array(a)` on the same 10M float64 arrow Array: **1.32 ms min, 1.39 ms median** interleaved —
the copying path described above (80 MB, about 57 GB/s at the median).

Method note: `arrow`'s CPU kernels are multi-threaded and base R's are not, so a comparison against
base R is not a per-core figure. Timings on a loaded machine are noise; these ran on an otherwise
idle Mac.

## Limits

- **int64.** R has no native 64-bit integer. An int64 Arrow column works throughout, but a scalar
  result comes back as a double, exact only to 2^53; `am_sum(x, integer64 = TRUE)` returns an
  exact `bit64::integer64` instead, and a plain call warns when the value is outside that range.
- **`arrow` is required**, and is how every column is built and read back. There is no path from a
  plain R vector to the GPU that does not go through an `arrow::Array` (`am_array(c(1,2,3))` calls
  `Array$create` for you).
- **Zero-based indices.** `am_argsort()` and `am_take()` use Arrow's convention, not R's. An index
  outside `[0, 2^31)`, or a fractional one, is an error rather than a silent `NA`.
- **`as_arrow_array()` is `arrow`'s generic**, not a new one: the package registers a method on it
  and re-exports it, so dispatch works whichever order the two packages are attached in.
- **An `NA` scalar** in `am_compare()` gives an all-null mask, as base R and arrow do. The ABI
  scalar carries no validity flag, so this is handled in R rather than on the GPU.
- **macOS on Apple silicon only**, and the Swift core is a separate `libArrowMetalC.dylib` that is
  not bundled inside this package.

## Tests

64 `test_that()` blocks in the sources (65 as testthat runs them: the one in test-dispatch.R runs once per attach order), 266 expectations, comparing against base R and against `arrow`'s
own kernels on the same data: nulls, all-null and empty columns, sliced input, lengths of 1, 33,
1024, 65537 and 1,000,001 (crossing a threadgroup boundary), int64 above 2^53, float32, strings and
booleans, and every error path.
`R CMD build r/arrowmetal && R CMD check --no-manual arrowmetal_0.1.0.tar.gz` is clean: 0 errors,
0 warnings, 0 notes.

```
ARROWMETAL_LIB=/path/to/libArrowMetalC.dylib Rscript -e 'testthat::test_local("r/arrowmetal")'
```
