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

This binding resolves 34 of the C ABI's 200-plus entry points. Everything not in the table above —
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
cannot hold; a validity bitmap; a cast result; string buffers) lands page aligned.

So: **a double or int32 column that came from an R vector is copied on the way in; an int64
column, a validity bitmap, a cast result, a string column and anything read from a file are not.**
The copy is measured below at 1.51 ms for 80 MB.

## Measured timing

One run, this session. Apple M4 Max (16 cores, 64 GB unified memory), macOS 26.6.2, R 4.5.3,
arrow 25.0.0, ArrowMetal 0.1.0. 10,000,000 float64, no nulls, one warm-up call of every
expression, then `microbenchmark(times = 5)`; the number is the **minimum** wall time of the five.
"resident" means the column is already an `am_array`; "import" means the timing starts from an
`arrow::Array` and includes the transfer.

### `sum`

| Method | Best of 5 | vs base R |
|---|---:|---:|
| `arrow::call_function("sum", a)` | **1.20 ms** | 10.4× |
| `am_sum(h)` — resident | 1.39 ms | 9.0× |
| `am_sum(a)` — import + sum | 3.58 ms | 3.5× |
| `sum(x)` — base R double vector | 12.54 ms | 1.0× |

**ArrowMetal loses `sum` to arrow's own CPU kernel**: 1.39 ms against 1.20 ms with the column
already on the GPU (1.16× slower), and 3.58 ms against 1.20 ms once the import is counted
(3.0× slower). A sum is one pass over 80 MB and nothing else, so it is bounded by memory
bandwidth that the CPU already saturates with 16 threads; there is no arithmetic for the GPU to
win back.

### `filter` (`x > 0.5`, ~5M rows out)

| Method | Best of 5 | vs base R |
|---|---:|---:|
| `am_filter(h, am_compare(h, ">", 0.5))` — resident | **1.22 ms** | 32.7× |
| `am_filter(a, am_compare(a, ">", 0.5))` — import + filter | 4.20 ms | 9.5× |
| `arrow` `greater` then `filter` | 24.15 ms | 1.7× |
| `x[x > 0.5]` — base R | 40.03 ms | 1.0× |

ArrowMetal wins `filter` by 19.8× against arrow with the column resident and by 5.8× including
the import. All four produce the same answer (checked in the same script: identical sum, identical
output length).

### Import alone

`am_array(a)` on the same 10M float64 arrow Array: **1.51 ms**, the copying path described above
(80 MB, about 53 GB/s).

Method note: `arrow`'s CPU kernels are multi-threaded and base R's are not, so the "vs base R"
column is not a per-core comparison. Timings on a loaded machine are noise; this ran on an
otherwise idle Mac and has not been re-measured since.

## Limits

- **int64.** R has no native 64-bit integer. An int64 Arrow column works throughout, but a scalar
  result comes back as a double, exact only to 2^53; `am_sum(x, integer64 = TRUE)` returns an
  exact `bit64::integer64` instead, and a plain call warns when the value is outside that range.
- **`arrow` is required**, and is how every column is built and read back. There is no path from a
  plain R vector to the GPU that does not go through an `arrow::Array` (`am_array(c(1,2,3))` calls
  `Array$create` for you).
- **Zero-based indices.** `am_argsort()` and `am_take()` use Arrow's convention, not R's.
- **macOS on Apple silicon only**, and the Swift core is a separate `libArrowMetalC.dylib` that is
  not shipped inside this package.

## Tests

179 testthat tests, all passing, comparing against base R and against `arrow`'s own kernels on the
same data: nulls, all-null and empty columns, sliced input, lengths of 1, 33, 1024, 65537 and
1,000,001 (crossing a threadgroup boundary), int64 above 2^53, float32, strings and booleans, and
every error path. `R CMD check --no-manual` is clean: 0 errors, 0 warnings, 0 notes.

```
ARROWMETAL_LIB=/path/to/libArrowMetalC.dylib Rscript -e 'testthat::test_local("r/arrowmetal")'
```
