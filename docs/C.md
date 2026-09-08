# C

One header, one dynamic library, no Swift in sight for the caller. `include/arrowmetal.h` declares
222 entry points over `libArrowMetalC.dylib`; every other binding in this repository (Python,
Rust, Go, TypeScript, R, the Polars plugin, the DuckDB extension) is built on it, so it is tested by all
of their suites as well as by `python/tests`, which calls it through ctypes, and by a test that compiles
the header as C in the merge gate.

## Build the library

```
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
swift build -c release --product ArrowMetalC        # .build/release/libArrowMetalC.dylib
```

## Conventions

- Handles: `am_array*` is an opaque, Metal-resident Arrow array; `am_release` frees it.
- Every function returns 0 on success and non-zero on error; `am_last_error()` has the message,
  thread-local, valid until the next call on that thread. Argument guards return 2 without setting a
  message.
- In and out through the Arrow C Data Interface: `am_import` moves the caller's `ArrowArray` (its
  `release` is nulled) and borrows the schema; `am_export` fills an `ArrowArray`/`ArrowSchema` pair
  whose release callback owns the buffers. `am_import_device`/`am_export_device` speak the C Device
  interface with `ARROW_DEVICE_METAL`; `am_stream_*` speak the C Stream interface.
- Scalars are passed as a pointer to a value of the array's compute element type: `am_compute_format`
  names it (for a dictionary array `am_format` reports the index type, and the kernels compute on the
  values).
- Copy-free out, always. Copy-free in when the producer's buffers are page aligned; one copy otherwise.

## Example

```c
#include "arrowmetal.h"

struct ArrowSchema schema; struct ArrowArray array;   /* filled by any Arrow producer */
am_array* col = NULL;
if (am_import(&schema, &array, &col) != 0) { fprintf(stderr, "%s\n", am_last_error()); return 1; }

int64_t two = 2; am_array* mask = NULL; am_array* hits = NULL;
am_compare_scalar(col, 4 /* gt */, &two, &mask);
am_filter(col, mask, &hits);

int64_t sum; double f; int kind, is_null;
am_reduce(hits, 0 /* sum */, &sum, &f, &kind, &is_null);

struct ArrowSchema out_schema; struct ArrowArray out_array;
am_export(hits, &out_schema, &out_array);             /* copy-free; the consumer calls out_array.release */
am_release(hits); am_release(mask); am_release(col);
```

## What is in the header

Reductions, element-wise compare and arithmetic, casts, boolean logic, selection, sorts and top-k,
group-by with its aggregates, strings, temporal, decimal, nested types, hashing and set lookup, window
functions, joins, the fused expression runner (`am_query`), the JSON plan runner (`am_plan_*`), Parquet,
the streaming executor (`am_stream_*`), batching (`am_batch_*`) and the device interfaces. The header's
comments are the reference for each; [ARROW_FUNCTIONS.md](ARROW_FUNCTIONS.md) maps every Arrow function
name to its entry point.

## Limits

- Thread-local errors mean a green-threaded runtime must read `am_last_error` on the thread that made
  the call (the Go binding pins the OS thread around every call for this reason).
- `am_import` does not report whether it wrapped or copied; a caller infers it from buffer alignment.
  `am_import_ex` with that flag is on the [roadmap](ROADMAP.md).
- macOS arm64 only: the library links Metal.
