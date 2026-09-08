# Swift

The native API. The Python package, the C ABI and every other binding call these types; nothing is
faster to reach the kernels than calling them here. Everything on this page is in
[README.md](../README.md), `Sources/ArrowMetalExamples/main.swift` or the tests under
`Tests/ArrowMetalTests`, and the numbers are the ones those cite.

## Install

```swift
// Package.swift
dependencies: [.package(url: "https://github.com/singhpratech/ArrowMetal", from: "0.1.0")],
targets: [.target(name: "MyApp", dependencies: [.product(name: "ArrowMetal", package: "ArrowMetal")])]
```

macOS 14 or later, iOS 17 or later (`Package.swift`); Swift 6.3.3 is the toolchain the tests run under.
Build and test from a checkout with the Xcode toolchain selected:

```
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
swift build -c release
swift test -c release        # release is required: a release-only miscompile has bitten this project once
```

## Example

```swift
import ArrowMetal

let prices = try MetalArray<Float>([9.5, nil, 12.0, 3.25])       // nullable, lives in Metal shared memory
let mask   = try prices.compare(.gt, 5)                          // GPU, Arrow boolean bitmap out
let picked = try prices.filter(mask)                             // GPU stream compaction
print(try picked.sum(), try picked.max(), picked.nullCount)      // .float(21.5), 12.0, 0

// Multi-column batches behave like an Arrow RecordBatch.
let orders = try MetalRecordBatch(names: ["region", "amount"], columns: [.int32(region), .float32(amount)])
let hits = try orders.filter(try region.compare(.eq, 2).and(try amount.compare(.gt, 100)))
let window = try hits.slice(offset: 32, length: 1000)             // zero-copy view

// Chains share one GPU round trip.
let total = try MetalContext.shared.batch {
    try amount.filter(try region.compare(.eq, 2).and(try amount.compare(.gt, 100))).sum()
}
```

`batchAsync` is the same without the wait; return a pending array from the body and use the async
accessors (`sumAsync` and friends) for scalars. Six end-to-end scenarios (analytics on a record batch,
feature preparation, Float64 with NaN, C Data Interface interop, sliced windows, one batched command
buffer) run with `swift run -c release arrowmetal-examples`.

## What the API is

| Type | What it is | Where it is documented |
|---|---|---|
| `MetalArray<T>` | one Arrow array in Metal shared memory: reductions, compare, arithmetic, cast, filter, take, slice, sort, top-k, strings, temporal, decimal | [DESIGN.md](DESIGN.md), [COVERAGE.md](COVERAGE.md) |
| `AnyMetalArray` | the type-erased array every entry point in the C ABI hands around | [ARROW_FUNCTIONS.md](ARROW_FUNCTIONS.md) |
| `MetalRecordBatch` | named columns with filter, take, slice and group-by across them | [ENGINE.md](ENGINE.md) |
| `MetalContext` | the device, the command queue, `batch` and `batchAsync` | [DESIGN.md](DESIGN.md), [RESIDENT.md](RESIDENT.md) |
| Expressions and `LazyFrame` | a fused expression compiler and a lazy engine with an optimizer, six join kinds and an as-of join | [EXPR.md](EXPR.md), [ENGINE.md](ENGINE.md) |
| `ParquetFile` | Parquet decoded on the GPU | [PARQUET.md](PARQUET.md) |
| `StreamQuery` | larger-than-memory scans, disk through the GPU | [STREAMING.md](STREAMING.md) |
| `importArrowArray`, `exportArrowArray`, `exportArrowDeviceArray` (`CArrowABI`) | the Arrow C Data, C Device and C Stream interfaces; `ImportResult.zeroCopy` says whether the import wrapped or copied | [COVERAGE.md](COVERAGE.md) |

## The copy rule

Copy-free out, always. Copy-free in when the producer's buffers are page aligned; one copy otherwise.
`importArrowArray` reports which it did. Details and the measurements behind them are in
[DESIGN.md](DESIGN.md) and [NUMPY.md](NUMPY.md).

## Tests

769 test functions in 61 files under `Tests/ArrowMetalTests`, run in release; the oracles are plain-Swift
CPU references, hand-computed vectors and pyarrow answers pinned as literals ([TESTING.md](TESTING.md)).
Tests that need a real GPU skip on virtual Metal devices.

## Numbers

The Swift-level baselines (one core, all 16 cores, Accelerate) are in [README.md](../README.md) and
[BENCHMARKS.md](BENCHMARKS.md); the operation-by-operation matrix is measured from Python over the same
kernels ([BENCHMARKS_MATRIX.md](BENCHMARKS_MATRIX.md)).

## Limits

- One GPU, one device; no multi-GPU.
- The kernels are generated Metal Shading Language, compiled at first use and cached per process, so the
  first call of each kernel family pays a compile of a few milliseconds.
- Everything the other pages list under limits applies here first, since this is the layer they wrap.
