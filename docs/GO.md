# ArrowMetal for Go users

A Go module that hands an `arrow.Array` from [Apache Arrow Go](https://github.com/apache/arrow-go)
to the GPU and takes the answer back, over the Arrow C Data Interface. It wraps a deliberately small
part of the C ABI: import and export, the four reductions, compare, filter, take, sort, argsort,
lexsort, group-by, and the JSON plan runner. Everything it wraps has a test against Arrow Go's own
compute or against a plain Go loop on the same data. Everything it does not wrap is listed under
[What is not wrapped](#what-is-not-wrapped).

Module path: `github.com/singhpratech/ArrowMetal/go/arrowmetal`. Source: [`go/arrowmetal/`](../go/arrowmetal).

---

## Install

```bash
# 1. The GPU library. It is a build product of the Swift package; `go get` cannot fetch it.
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
    swift build -c release --product ArrowMetalC

# 2. The module.
go get github.com/singhpratech/ArrowMetal/go/arrowmetal
```

That `go get` line starts working once the repository is public **and** carries a
`go/arrowmetal/vX.Y.Z` tag — the module lives in a subdirectory, so the Go module proxy wants the
tag prefixed with the module's path within the repository, not a bare `vX.Y.Z`. Until then, work
against a checkout: clone the repository and add

```
replace github.com/singhpratech/ArrowMetal/go/arrowmetal => /path/to/ArrowMetal/go/arrowmetal
```

to your own `go.mod`, or just run inside `go/arrowmetal`.

Requirements: macOS on Apple silicon, a Metal device, and Go 1.25.0 or newer (the module's `go`
directive, which is what `arrow-go/v18` requires) with **cgo enabled** — `CGO_ENABLED=1`, the default
when a C toolchain is present; Xcode's clang is what this was built and tested with. The module
depends on `github.com/apache/arrow-go/v18` and nothing else. Everything here was measured on
Go 1.27.1.

### Finding the dylib

The binding does not link `libArrowMetalC.dylib` at build time. It `dlopen`s it the first time you
use the package, so the location is a run-time decision and a missing library is a clear Go error
rather than a dyld message. The search, in order:

1. `$ARROWMETAL_LIB` — the **full path to the dylib**, not a directory. If it is set and does not
   load, that is the error; there is no fallback.
2. `.build/release/libArrowMetalC.dylib`, and the same at one, two and three levels up, relative to
   the working directory. (`go test` runs in the package directory, so `../../.build/release` is the
   repository's own build; a program run from the repository root finds `.build/release`.)
3. The same four paths relative to the directory holding the running binary.
4. Plain `libArrowMetalC.dylib`, letting dyld search `DYLD_LIBRARY_PATH`, `/usr/local/lib` and the
   rest.

When none of them loads, the error names `ARROWMETAL_LIB`, every path it tried and why each failed,
and the `swift build` line that produces the file. `arrowmetal.Init()` performs the load and returns
that error, so a program can fail early with it; every other entry point calls `Init` first.
`arrowmetal.LibraryPath()` reports where it came from.

---

## Example

```go
package main

import (
	"fmt"

	"github.com/apache/arrow-go/v18/arrow/array"
	am "github.com/singhpratech/ArrowMetal/go/arrowmetal"
)

func main() {
	// PageAlignedAllocator is a memory.Allocator whose buffers ArrowMetal can borrow (see Copy rule).
	b := array.NewInt64Builder(am.NewPageAlignedAllocator())
	defer b.Release()
	b.AppendValues([]int64{5, 3, 9, 1}, []bool{true, true, false, true}) // 9 is null

	src := b.NewArray()
	defer src.Release()

	gpu, err := am.Import(src)             // arrow.Array -> GPU
	if err != nil {
		panic(err)
	}
	defer gpu.Release()

	sum, _ := gpu.Sum()                    // nulls skipped, like Arrow
	mask, _ := gpu.CompareScalar(am.Gt, int64(2))
	kept, _ := gpu.Filter(mask)
	out, _ := kept.Export()                // back to arrow-go, no copy
	defer out.Release()

	fmt.Println(sum, out)                  // 9 [5 3]
}
```

Run it with `ARROWMETAL_LIB=/path/to/.build/release/libArrowMetalC.dylib go run .`.

---

## Copy rule

The project's rule, unchanged: **copy-free out always; copy-free in when the producer's buffers are
page aligned, one copy otherwise.**

Concretely, on the way in ArrowMetal wraps a buffer with `MTLBuffer(bytesNoCopy:)` when the buffer's
start address is a multiple of the page size — 16384 bytes on Apple silicon, which
`arrowmetal.PageSize()` reports — and copies the bytes into a page-aligned Metal buffer otherwise.
On the way out the exported `arrow.Array` references the same unified memory the GPU wrote, always.

### What Arrow Go's default allocator does — measured

`memory.DefaultAllocator` **is** `memory.NewGoAllocator()`, the Go heap, unless the program is built
with the `mallocator` build tag — the same object, not a similar one, which is why only one of them
appears below. The Go runtime backs large allocations with whole spans of its own 8 KiB pages, so an
Arrow buffer from it is **8 KiB aligned and only incidentally 16 KiB aligned**.

`TestAllocatorAlignment` builds an Int64 array 20 times per size and records the values buffer's
offset past a page boundary. On an M4 Max, Go 1.27.1, arrow-go v18.7.0:

| Allocator | Offset past a 16384-byte page | Page aligned? |
|---|---|---|
| `memory.NewGoAllocator()` (the default) | only ever **0 or 8192**, never any other value | a coin flip |
| `arrowmetal.NewPageAlignedAllocator()` | always **0** | always |

**The hit rate is not a property, so it is not quoted here.** Across fresh runs of the same test the
count at 10M int64 came back 5/20, 15/20 and 14/20 — it depends on the state of the heap when the
buffer is allocated, and it will differ on your machine and between two runs on mine. What is
reproducible, and what the test asserts, is the offset: a Go-heap buffer is either exactly on a page
boundary or exactly 8192 bytes past one. That follows from Go's 8 KiB page meeting Apple silicon's
16 KiB one, and it means a copy-free import with the default allocator is a coin flip whose outcome
you cannot observe from Go.

`arrowmetal.PageAlignedAllocator` implements `memory.Allocator` on top of `posix_memalign` at page
granularity, so the import can always borrow.

### Go pointers, cgo, and why pinning is not optional

Alignment and legality are separate questions, and `PageAlignedAllocator` answers only the first.

`cdata.ExportArrowArray` writes `&buf.Bytes()[0]` straight into a `malloc`'d buffer array
([cdata_exports.go:388](https://github.com/apache/arrow-go/blob/main/arrow/cdata/cdata_exports.go)).
With the default allocator that address is in the Go heap, so this is an unpinned Go pointer stored
into non-Go memory — which the [cgo pointer rules](https://pkg.go.dev/cmd/cgo#hdr-Passing_pointers)
forbid, and which `GOEXPERIMENT=cgocheck2` turns into a hard `fatal error: unpinned Go pointer stored
into non-Go memory`. This is [apache/arrow-go#70](https://github.com/apache/arrow-go/issues/70),
closed upstream without the default path being made legal; arrow-go's own advice is to use
`CgoArrowAllocator`, which needs the `ccalloc` build tag and a C++ Arrow install.

`Import` therefore pins every buffer it publishes with [`runtime.Pinner`](https://pkg.go.dev/runtime#Pinner)
(Go 1.21+) before handing anything to C, and keeps the `Pinner` inside the returned `Array` until
`Release`, unpinning only after `am_release` — ArrowMetal may have been borrowing those exact bytes.
`Pin` ignores anything outside the Go heap, so a buffer from `PageAlignedAllocator` costs nothing
here and needs nothing.

The consequence for you: **any allocator is safe**, including the default. The suite is run under
`GOEXPERIMENT=cgocheck2` as well as normally, and removing the pin makes it fail on the first round
trip. Choose `PageAlignedAllocator` when you want the copy avoided, not to make the code legal.

### What the copy actually costs — measured

Less than you would expect, because at 76 MB the copy is not the dominant cost of crossing the
boundary; setting up the Metal buffer is. From the table below: importing 10M int64 rows takes
1.13 ms when the buffer can be borrowed and 1.41 ms when it must be copied — the copy adds about
0.3 ms to a 1.4 ms operation, roughly 20%. Export is 917 ns, three orders of magnitude cheaper,
which is what "copy-free out always" buys.

**There is no way to ask the library whether a given import copied.** The Swift core computes it
(`ImportResult.zeroCopy`) but the C ABI's `am_import` does not return it, so this binding cannot
report it either and the table above measures alignment in Go instead.

---

## Timing

10M Int64 rows (76 MB), no nulls, one M4 Max, one process, wall clock, **best of 5 timed runs after
one untimed warm-up**. Reproduce with `go run ./cmd/amtiming` from `go/arrowmetal`; the source of
every row is [`cmd/amtiming/main.go`](../go/arrowmetal/cmd/amtiming/main.go).

ArrowMetal 0.1.0, Go 1.27.1, arrow-go v18.7.0. The filter predicate is `x > 0` and keeps 50.0% of
the rows.

| Op | Method | Best of 5 | Notes |
|---|---|---:|---|
| Import | arrow-go → ArrowMetal, page-aligned buffer | **1.13 ms** | borrowed, no copy |
| Import | arrow-go → ArrowMetal, buffer 64 B past a page | **1.41 ms** | one copy of 76 MB |
| Import | arrow-go → ArrowMetal, `memory.NewGoAllocator` | **1.40 ms** | this run's buffer was 8192 B past a page |
| Export | ArrowMetal → arrow-go | **917 ns** | always copy-free |
| Sum | plain Go loop over `[]int64` | 2.46 ms | |
| Sum | Arrow Go `arrow/math` (NEON) | 1.16 ms | arrow-go registers no `sum` compute function |
| Sum | **ArrowMetal, array already resident** | **296 µs** | 3.9× the Arrow Go kernel |
| Sum | ArrowMetal, end to end from a page-aligned `arrow.Array` | 2.22 ms | **slower than Arrow Go** |
| Sum | ArrowMetal, end to end with a copying import | 1.87 ms | **slower than Arrow Go** |
| Filter | plain Go loop (one fused pass into a `[]int64`) | 29.18 ms | |
| Filter | Arrow Go compute (`greater` then `filter`) | 48.93 ms | |
| Filter | **ArrowMetal, array already resident** | **1.51 ms** | 32× Arrow Go compute, 19× the Go loop |
| Filter | ArrowMetal, end to end from a page-aligned `arrow.Array` | 3.22 ms | 15× Arrow Go compute |
| Filter | ArrowMetal, end to end with a copying import | 2.98 ms | 16× Arrow Go compute |

### Reading this honestly

- **ArrowMetal loses at Sum end to end.** 2.22 ms against Arrow Go's 1.16 ms. A single 76 MB sum is
  a memory-bandwidth problem that the CPU is already good at, and the ~1.2 ms of import overhead is
  most of the ArrowMetal number. Only when the array is already on the GPU does Sum win, 296 µs
  against 1.16 ms. If your program's shape is "load an arrow-go array, sum it once, throw it away",
  this binding is the wrong tool.
- **ArrowMetal wins at Filter, in every shape.** Even paying import and export on every call, 3.22 ms
  against 48.93 ms is 15×; resident it is 32×. Filter does more work per byte than Sum and the
  fixed cost stops dominating.
- The end-to-end rows with a copying import came out *faster* than the page-aligned ones in this
  run (1.87 vs 2.22 ms for Sum, 2.98 vs 3.22 ms for Filter), which is the opposite of what the
  Import rows say. The gap is a few tenths of a millisecond either way, the same size as the copy
  itself, and it is run-to-run noise from Metal's buffer allocation. The defensible statement is the
  narrow one: at 76 MB, borrowing saves about 0.3 ms of a 1.4 ms import, and end to end that saving
  is inside the noise. It is not on its own a reason to change allocators — and it is not a
  correctness reason either, since `Import` pins whatever it is given.
- `arrow/math.Int64.Sum` ignores nulls, and the data here has none. It is the fastest thing arrow-go
  has for this and it is what the Sum row compares against, because **arrow-go's compute package
  registers no aggregate function at all** — no `sum`, `mean` or `min_max` in its registry.
  `TestArrowGoHasNoAggregates` records this and will log a nudge if a later release adds them.
- The plain Go filter loop writes into a `[]int64` rather than building an `arrow.Array`, so it is
  doing strictly less work than the other two Filter rows. It is included because it is what a Go
  programmer writes when they have not reached for Arrow yet.

---

## What is covered

Every item below has at least one test in `go/arrowmetal`; the oracle is named. 45 test functions,
164 cases counting subtests, all green.

| Surface | Go API | Oracle |
|---|---|---|
| C Data Interface in and out | `Import`, `(*Array).Export` | value-and-null round trip at lengths 0, 1, 1000, 1,000,001, plain and sliced |
| Sliced input (`offset != 0`) | the same | `array.NewSlice` at offsets 1, 7, 31, 32, 33, 63, 64, 1000 against the same rows read directly |
| Sum, Min, Max, Mean | `(*Array).Sum/Min/Max/Mean` | plain Go loops (arrow-go has no aggregates); Int64 and Float64, with and without nulls |
| Null and NaN rules | the same | all-null and empty arrays report an invalid `Scalar`; min/max skip NaN; an all-NaN column is null |
| Compare against a scalar | `(*Array).CompareScalar` | `compute.CallFunction("equal"/"not_equal"/"less"/"less_equal"/"greater"/"greater_equal")`, all six, with nulls |
| Scalar range checking | the same | a scalar outside the column's element type is an error, at both edges of int8, uint8 and float32; the values that do fit agree with a Go loop |
| Compare two arrays | `(*Array).CompareArray` | length-mismatch error path |
| Filter | `(*Array).Filter` | `compute.FilterArray` with `SelectionDropNulls` |
| Take | `(*Array).Take` | `compute.TakeArray`, including null indices |
| Sort | `(*Array).Sort` | `compute.SortArray`, ascending and descending, nulls at end |
| Argsort | `(*Array).Argsort` | `compute.SortIndicesArray`, index for index, on data with heavy ties |
| Lexsort | `Lexsort` | `sort.SliceStable` over the same two columns |
| Group-by | `NewGroupBy`, `.Sum/.Count/.CountAll/.Mean/.Min/.Max/.Key/.IDs` | plain Go maps; one and two key columns, 1 to 1000 groups, null keys, float values, zero rows |
| JSON plan runner | `NewSource`, `RunPlan`, `ExplainPlan`, `PlanResult.RecordBatch` | plain Go; the header's own group-by/sort/limit example, optimized against unoptimized, a type-check failure |
| Slice on the GPU | `(*Array).Slice` | the same rows read from the source |
| Errors | `*arrowmetal.Error` | a length mismatch and a bad plan both carry `am_last_error()` text |
| Allocator | `PageAlignedAllocator` | alignment at 1M and 10M, allocate/reallocate/free bookkeeping, a full round trip |
| Allocator alignment | the Go heap's shape | every Go-heap offset past a page is 0 or 8192 and nothing else, at 1M and 10M, 20 trials each |
| cgo pointer rules | `Import` | the whole suite again under `GOEXPERIMENT=cgocheck2`; removing the pin makes it die on the first round trip |
| Handle lifecycle | `Release` | repeated import of one `arrow.Array`; a long-lived handle alongside short-lived ones; a released handle errors rather than crashing |
| Leaks | the whole chain | 2,000 import/compare/filter/export/release round trips at 200k rows; `TASK_VM_INFO.phys_footprint` has to stay inside 16 MB of the baseline (it grows about 2.5 MB, and leaking just the mask handle grows 35 MB) |
| The loader | `Init`, `LibraryPath` | a child process with `ARROWMETAL_LIB` pointing at nothing, and a child with nothing set in an empty directory: the error has to name the variable, the paths and the `swift build` line |
| Docs | the example in this file | compiled and run as `Example()`, so it cannot drift from the API |

Sizes are at or below 10M elements throughout.

---

## What is not wrapped

The C ABI has 220 entry points; this binding resolves 33 of them. Not wrapped, and not tested from
Go:

- **Arithmetic and math**: `am_arith_scalar`, `am_arith_array`, `am_unary`, `am_binary`,
  `am_cumulative`, `am_window`, the checked variants.
- **Strings and text**: `am_str_unary`, `am_str_match`, `am_str_transform` (all 26 ops),
  `am_str_concat`, `am_str_dictionary_encode`, the regex surface.
- **Temporal**: `am_temporal_extract`, `am_temporal_cast_unit`, `am_round_temporal_ex`,
  `am_add_interval`.
- **Types**: decimals, dictionaries beyond what import/export carries through, nested types (list,
  struct, map, union), run-end encoding, and every `_ex` option variant (`am_argsort_ex`,
  `am_rank_ex`, `am_is_in_ex`, …).
- **Structural and conditional**: `am_is_null`, `am_fill_null`, `am_drop_null`, `am_if_else`,
  `am_coalesce`, `am_is_in`, `am_index_in`, the Kleene operators.
- **Aggregates beyond the four**: `am_reduce_ex` (product, variance, stddev), and the grouped
  aggregates past sum/count/mean/min/max — the `Agg` method takes any op number from the header's
  table, but only those five are tested from Go.
- **Batching**: `am_batch_begin` / `am_batch_end`. A Go binding for these needs the batch and every
  call inside it pinned to one OS thread, which the current one-call-at-a-time pinning does not give
  you across calls.
- **Whole subsystems**: Parquet (`am_parquet_*`), streaming (`am_stream_*`), the device interface
  (`am_import_device` / `am_export_device`), joins in index form (`am_hash_join`), sources built
  from an `ArrowArrayStream`.
- `arrow.RecordBatch` and `arrow.Table` go in only one at a time, column by column. `PlanResult` can
  produce a `RecordBatch`; nothing consumes one.

Adding any of these is mechanical: a prototype in `amshim.h`, a pointer and a forwarder in
`amshim.c`, a method in Go, and a test with an oracle.

---

## Limits

- **macOS on Apple silicon only.** The module compiles anywhere cgo does, but the dylib it needs
  exists only for Apple silicon, so on any other platform `Init()` returns the loader's error.
- **cgo is required.** `CGO_ENABLED=0` builds will fail to compile the package.
- **One goroutine at a time per handle.** An `Array`, `GroupBy`, `Source` or `PlanResult` is not safe
  for concurrent use. Separate handles on separate goroutines are fine.
- **Every call pins its goroutine to an OS thread for the duration.** `am_last_error()` is
  thread-local, so the failing call and the message read have to happen on the same thread; Go is
  otherwise free to move a goroutine between calls. The cost is a `runtime.LockOSThread` pair per
  call, which is tens of nanoseconds against kernels measured in hundreds of microseconds.
- **Release your handles.** A finalizer is set as a backstop, but GPU memory should not wait for the
  garbage collector, and the finalizer runs at an unpredictable time.
- **Go heap buffers stay pinned for the life of the handle.** Importing an array built with Arrow
  Go's default allocator hands C a pointer into the Go heap that ArrowMetal keeps until the handle
  is released, so `Import` pins those buffers and `Release` unpins them (see
  [Go pointers, cgo, and why pinning is not optional](#go-pointers-cgo-and-why-pinning-is-not-optional)).
  Pinned objects cannot be moved or freed, so a handle you forget to release holds its source
  buffers as well as its GPU memory.
- **Scalar comparison covers the primitive types only.** `CompareScalar` accepts a Go value for
  int8…int64, uint8…uint64, float32, float64 and bool, range-checked against the column's element
  type: `int64(1000)` against an int8 column is an error naming both, not a silent −24. A string,
  decimal or temporal scalar returns an error naming the Arrow format string rather than guessing.
- **The vendored headers can drift.** `go/arrowmetal/include/arrowmetal.h` and `arrow_abi.h` are
  copies of the repository's `include/`, because a Go module can only see files inside its own
  directory. `TestHeadersMatchRepository` compares them when it is run inside a checkout and skips
  otherwise.

---

## Testing

See [TESTING.md](TESTING.md). In short, from `go/arrowmetal`:

```bash
ARROWMETAL_LIB=$PWD/../../.build/release/libArrowMetalC.dylib go test ./...

# and once more with the cgo pointer checker armed, which is where an unpinned
# Go pointer handed to C shows up as a hard failure rather than as luck:
ARROWMETAL_LIB=$PWD/../../.build/release/libArrowMetalC.dylib \
    GOEXPERIMENT=cgocheck2 go test -count=1 ./...
```

The library path is optional inside the repository — `../../.build/release` is on the search list —
but it is the reliable way to say which build you mean.
