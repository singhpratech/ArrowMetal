# ArrowMetal

**Apache Arrow columnar data on Apple silicon GPUs.** Zero-copy, unified-memory, Metal-accelerated kernels that
understand Arrow's layout natively: validity bitmaps, packed booleans, the C Data Interface and the
C Device Data Interface (`ARROW_DEVICE_METAL`).

> Status: v0.1, early and small on purpose. The core works, is tested against a CPU oracle, and is fast.
> Everything on the [roadmap](ROADMAP.md) is up for grabs.

## Why this exists

Apple silicon has one physical memory shared by CPU and GPU. An Arrow buffer placed in a `MTLBuffer` with
`storageModeShared` is *simultaneously* a valid CPU Arrow buffer and a valid GPU buffer. No upload, no download,
no copies. Nothing in the Arrow ecosystem took advantage of that:

| Project | What it has | What it lacks |
|---|---|---|
| [arrow-swift](https://github.com/apache/arrow-swift) | Types, IPC, Flight, C Data Interface | Zero compute kernels, no Metal |
| [nanoarrow device](https://arrow.apache.org/nanoarrow/latest/reference/device.html) | C wrapper for Metal buffers | No compute, C only |
| [cuDF](https://github.com/rapidsai/cudf) | Full GPU dataframe | CUDA only |
| [MLX](https://github.com/ml-explore/mlx) | Fast unified-memory tensors | No nulls, no columnar semantics, ML focused |

ArrowMetal fills that hole: Arrow arrays whose buffers live in Metal shared memory, GPU kernels that honour
Arrow semantics, and standard Arrow C interfaces in and out so it plugs into arrow-rs, pyarrow, DuckDB, Polars,
arrow-swift or anything else that speaks the C Data Interface.

## Benchmarks

Apple M4 Max (16 CPU cores), 50,000,000 rows, best of 5, release build. Full history and methodology in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md) and [Benchmarks/README.md](Benchmarks/README.md).

**Called from Python, same in-process data**, against Polars (16 threads), pyarrow.compute and pandas:

| Operation | ArrowMetal | Polars | pyarrow | pandas |
|---|---:|---:|---:|---:|
| sum Int64, 10% nulls | **1.05 ms** | 15.78 | 48.26 | 47.75 |
| filter Int64 > 0 | **3.55** | 22.99 | 211.05 | 271.50 |
| take 25M random indices | **5.98** | 164.82 | 138.61 | |
| group-by sum, 1000 keys | **2.05** | 84.65 | 18.73 | |
| filter two columns + sum | **3.13** | 16.41 (lazy) | | 109.46 (numpy) |

**Swift, against all 16 CPU cores** (tight typed loops over the same Arrow layout) and Accelerate:

| Operation | Metal | 16-core CPU / Accelerate |
|---|---:|---:|
| sum Int64, 10% nulls | **1.40 ms** | 4.89 |
| min Int64 | **1.34** | 3.73 |
| compare Int64 > 0 to bitmap | **1.07** | 2.12 |
| filter Int64 (45% kept) | **2.90** | 3.49 |
| take 25M random indices | **5.85** | 11.93 |
| group-by sum, 5 keys | **1.96** | 5.92 |
| query: filter two columns + sum | **2.33** | 6.36 |
| multiply Int64 * 3 | **2.13** | 2.26 |
| cast Int64 to Float32 | 3.14 | **1.80** |
| sum Float32 | 0.90 | **0.84** (vDSP) |
| multiply Float32 * 2.5 | 1.72 | **1.63** (vDSP) |
| Float32 compare + filter | **2.14** | 5.73 |

Takeaways: reductions, comparisons, selection, group-by and query-shaped pipelines beat all 16 CPU cores by
1.5x to 3x and Polars by 5x to 40x. Pure element-wise arithmetic is memory bound on both sides, so
Accelerate on 16 cores ties or edges ahead there. Arrays under about a million rows are dominated by the
fixed cost of a GPU dispatch (see [docs/DESIGN.md](docs/DESIGN.md) for the pipelining plan).

## From Python

```
swift build -c release --product ArrowMetalC        # .build/release/libArrowMetalC.dylib
PYTHONPATH=python python -c "import arrowmetal as am; print(am.device_name())"
```
```python
import pyarrow as pa, polars as pl, arrowmetal as am
col = am.array(pl.Series([1, None, 3, 40]).to_arrow())
print(col.filter_where(">", 2).sum(), pl.from_arrow(col.filter_where(">", 2).to_arrow()))
with am.batch():                                   # several kernels, one GPU round trip
    total = col.filter((col > 1) & (col < 40)).sum()
```
The same C ABI (`include/arrowmetal.h`) serves Rust, Go, C#, R, C++ and C through their Arrow C Data
Interface bindings. See [python/README.md](python/README.md).

## Quick start (Swift)

```swift
import ArrowMetal

let prices = try MetalArray<Float>([9.5, nil, 12.0, 3.25])       // nullable, lives in Metal shared memory
let mask   = try prices.compare(.gt, 5)                          // GPU, Arrow boolean bitmap out
let picked = try prices.filter(mask)                             // GPU stream compaction
print(try picked.sum(), try picked.max(), picked.nullCount)      // .float(21.5), 12.0, 0

// Multi-column batches behave like an Arrow RecordBatch.
let orders = try MetalRecordBatch(names: ["region", "amount"], columns: [.int32(region), .float32(amount)])
let hits = try orders.filter(try region.compare(.eq, 2).and(try amount.compare(.gt, 100)))
let sample = try hits.take(try MetalArray<Int32>([0, 5, 9]))
let window = try hits.slice(offset: 32, length: 1000)             // zero-copy view
```

Chains of operations can share one GPU round trip:

```swift
let total = try MetalContext.shared.batch {
    try amount.filter(try region.compare(.eq, 2).and(try amount.compare(.gt, 100))).sum()   // one command buffer
}
```

Five end-to-end scenarios (analytics query, feature preparation, Float64 with NaN, C Data Interface
interop, sliced windows) live in `Sources/ArrowMetalExamples`: `swift run -c release arrowmetal-examples`.

Interop with any Arrow implementation through the C Data Interface:

```swift
import CArrowABI

var schema = ArrowSchema(); var array = ArrowArray()
producer.export(into: &schema, &array)                            // arrow-rs, pyarrow, nanoarrow, arrow-swift ...
let imported = try importArrowArray(schema: &schema, array: &array)
// imported.zeroCopy == true when the producer's buffers were page aligned

var device = ArrowDeviceArray()
picked.exportArrowDeviceArray(into: &device)                      // device_type == ARROW_DEVICE_METAL
```

Requirements: macOS 14+ / iOS 17+, Swift 5.10+. Shaders compile at runtime so the Command Line Tools are
enough to build and use it. Running the test suite needs Xcode (for XCTest):
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test`.

## What is implemented

- `MetalArrowBuffer`: page-aligned shared-memory buffers, zero-copy wrap of foreign page-aligned memory.
- `MetalArray<T>` for Int8/16/32/64, UInt8/16/32/64, Float32, Float64; `MetalBooleanArray` with packed bits.
- `MetalRecordBatch`: named equal-length columns with `filter`, `take`, `slice`, `selecting`.
- Kernels: `sum`, `min`, `max`, `mean`, `compare` (6 ops, scalar and array), `add/sub/mul/div` (scalar and
  array, vectorised), `filter` and fused `filter(where:)` (single command buffer, GPU scan), `take`
  (Int32/Int64/UInt32 indices, bounds checked), `cast`, `slice` (zero-copy when 32-aligned), boolean
  `and/or/not/count/any/all`. All null-aware with Arrow semantics.
- `GroupBy` over dense integer keys: `count`, `sum`, `mean`, `min`, `max` (privatised threadgroup tables for
  up to 1024 keys, device atomics beyond; 64-bit sums via split 32-bit atomics with carry).
- `libArrowMetalC`: a C ABI over everything above, and a ctypes Python package that speaks the Arrow
  PyCapsule protocol.
- Float64: Metal has no `double`, so compare, min, max, filter, take and slice run on the GPU using an
  order-preserving map of the IEEE bit pattern (exact, NaN and signed zero handled); sum and arithmetic run
  on the CPU through the same API.
- NaN: `min`/`max` skip NaN and return null if only NaN remains; `sum` propagates NaN; comparisons follow IEEE.
- C Data Interface import/export for primitive arrays and struct (`+s`) record batches, C Stream Interface
  import, C Device Data Interface import/export, `MTLBuffer` recovery from our own exports.
- A CPU reference implementation of every kernel, used as the oracle in tests.

## Design notes

- **Runtime shader compilation.** Kernels are MSL strings generated per element type and cached. No `.metal`
  files, no `metal` toolchain, no Xcode required to build.
- **No atomics in reductions.** Each threadgroup writes a partial; the host finalises a few thousand values.
  Deterministic results, works for 64-bit integers where Metal atomics are limited.
- **Bitmaps as 32-bit words.** Compare writes one word per thread; filter uses `popcount` + simdgroup
  prefix sums. Buffers are always padded so whole-word reads are in bounds.
- **Page-aligned allocation.** Every buffer ArrowMetal hands out can be re-wrapped with
  `makeBuffer(bytesNoCopy:)` by another Metal consumer.
- **Lifetime discipline.** Raw pointers are only valid while their owning object lives. Prefer the closure
  accessors (`withValues`, `withTyped`).

See [ROADMAP.md](ROADMAP.md) for what is next and [CONTRIBUTING.md](CONTRIBUTING.md) to get involved.

## License

Apache License 2.0. Apache Arrow is a trademark of the Apache Software Foundation; this project is
independent and not endorsed by the ASF or Apple.
