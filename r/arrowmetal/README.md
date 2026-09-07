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
cannot hold; a validity bitmap; a cast result; string buffers) gets its own large allocation and
lands page aligned.

So, **at the 1M and 10M sizes measured here**: a double or int32 column that came from an R vector
is copied on the way in; an int64 column, a validity bitmap, a cast result, a string column and
anything read from a file are not.

The size qualifier is load-bearing. A buffer only lands page aligned when it is big enough for
arrow to give it its own allocation rather than a slice of a pool. Spot-checked: an int64 or string
column is page aligned from about a thousand elements, a validity bitmap from about a hundred
thousand (a bitmap for a thousand rows is 125 bytes), and at `n = 3` **nothing** is page aligned —
int64 values landed at offset 128, string buffers at 256 and 384, a validity bitmap at 192. Small
columns always take the copying path, which costs nothing worth measuring at that size.

The copy is measured below at 1.42 ms (median) for 80 MB.

## Measured timing

Apple M4 Max (16 cores, 64 GB unified memory), macOS 26.6.2, R 4.5.3, arrow 25.0.0,
ArrowMetal 0.1.0, idle machine. 10,000,000 float64, no nulls.

**Method:** **5 fresh R processes**, each building its own data, calling every expression once as a
warm-up, then `microbenchmark(times = 20)`. Reported below are the **minimum across all 100 runs**
and the **median of the 5 per-process medians**. A single best-of-5 in one process is not stable
here — an earlier such run put resident `filter` at 1.22 ms, which does not reproduce — so the
median is the number to quote and the minimum is the floor. "resident" means the column is already
an `am_array`; "import" means the timing starts from an `arrow::Array` and includes the transfer.

### `sum`

| Method | min | median |
|---|---:|---:|
| `am_sum(h)` — resident | **0.86 ms** | 1.40 ms |
| `arrow::call_function("sum", a)` | 1.14 ms | **1.26 ms** |
| `am_sum(a)` — import + sum | 2.05 ms | 3.29 ms |
| `sum(x)` — base R double vector | 11.18 ms | 11.96 ms |

**ArrowMetal loses `sum` to arrow's own CPU kernel.** On the median it is **1.11× slower**
resident and **2.6× slower** once the import is counted. (On its single best run it edges arrow —
0.86 ms against 1.14 ms — but that does not hold up across processes, which is exactly why the
median is quoted.) A sum is one pass over 80 MB and nothing else, so it is bounded by memory
bandwidth the 16-thread CPU kernel already saturates; there is no arithmetic for the GPU to win
back.

### `filter` (`x > 0.5`, ~5M rows out)

| Method | min | median |
|---|---:|---:|
| `am_filter(h, am_compare(h, ">", 0.5))` — resident | **0.78 ms** | **1.51 ms** |
| `am_filter(a, am_compare(a, ">", 0.5))` — import + filter | 3.74 ms | 4.65 ms |
| `arrow` `greater` then `filter` | 21.29 ms | 22.40 ms |
| `x[x > 0.5]` — base R | 36.62 ms | 39.34 ms |

ArrowMetal wins `filter` by **14.9× against arrow with the column resident and 4.8× including the
import**, on the medians. All four produce the same answer (checked in the same script: identical
sum, identical output length).

### Import alone

`am_array(a)` on the same 10M float64 arrow Array: **1.31 ms min, 1.42 ms median** — the copying
path described above (80 MB, about 56 GB/s at the median).

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
  not shipped inside this package.

## Tests

266 testthat tests, all passing, comparing against base R and against `arrow`'s own kernels on the
same data: nulls, all-null and empty columns, sliced input, lengths of 1, 33, 1024, 65537 and
1,000,001 (crossing a threadgroup boundary), int64 above 2^53, float32, strings and booleans, and
every error path. `R CMD check --no-manual` is clean: 0 errors, 0 warnings, 0 notes.

```
ARROWMETAL_LIB=/path/to/libArrowMetalC.dylib Rscript -e 'testthat::test_local("r/arrowmetal")'
```
