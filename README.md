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

Apple M4 Max, 50,000,000 rows, best of 5, release build. CPU baselines are tight typed Swift loops over the same
Arrow layout, plus Accelerate where an equivalent exists. Reproduce with `swift run -c release arrowmetal-bench`.

| Operation | Implementation | Time (ms) | Throughput (GB/s) |
|---|---|---:|---:|
| sum(Int64, 10% nulls) | **Metal** | 1.07 | 375.2 |
| sum(Int64, 10% nulls) | CPU 1-core null-aware loop | 55.04 | 7.3 |
| sum(Int64, 10% nulls) | CPU 16-core null-aware loop | 5.01 | 79.9 |
| min(Int64, 10% nulls) | **Metal** | 1.79 | 222.8 |
| min(Int64, 10% nulls) | CPU 1-core | 27.78 | 14.4 |
| compare(Int64 > 0) to bitmap | **Metal** | 1.18 | 339.3 |
| compare(Int64 > 0) to bitmap | CPU 1-core packed bitmap | 7.54 | 53.0 |
| filter(Int64, ~45% kept) | **Metal** | 5.36 | 74.6 |
| filter(Int64, ~45% kept) | CPU 1-core bit-scan loop | 37.64 | 10.6 |
| compare + filter | **Metal** | 6.37 | 62.8 |
| compare + filter | Swift `[Int64].filter` | 133.82 | 3.0 |
| multiply(Int64 * 3) | **Metal** | 6.84 | 116.9 |
| multiply(Int64 * 3) | CPU 1-core | 123.67 | 6.5 |
| sum(Float32) | **Metal** | 1.11 | 180.8 |
| sum(Float32) | Accelerate vDSP | 2.01 | 99.7 |
| max(Float32) | **Metal** | 0.88 | 226.3 |
| max(Float32) | Accelerate vDSP | 2.09 | 95.7 |
| multiply(Float32 * 2.5) | Metal | 3.65 | 109.6 |
| multiply(Float32 * 2.5) | **Accelerate vDSP** | 3.15 | 127.0 |

Takeaways: reductions and comparisons run at memory bandwidth on the GPU and beat 16 CPU cores. Filter is
about 7x a single core. Pure element-wise arithmetic is bandwidth bound on both sides, so Accelerate ties or
wins there; use the GPU when the column is already resident or the operation is part of a larger GPU pipeline.

## Quick start

```swift
import ArrowMetal

let prices = try MetalArray<Float>([9.5, nil, 12.0, 3.25])       // nullable, lives in Metal shared memory
let mask   = try prices.compare(.gt, 5)                          // GPU, Arrow boolean bitmap out
let picked = try prices.filter(mask)                             // GPU stream compaction
print(try picked.sum(), try picked.max(), picked.nullCount)      // .float(21.5), 12.0, 0
```

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
- Kernels: `sum`, `min`, `max`, `mean`, `compare` (6 ops, scalar and array), `add/sub/mul/div` (scalar and
  array), `filter`, boolean `and/or/not`. All null-aware with Arrow semantics.
- C Data Interface import/export, C Device Data Interface import/export, `MTLBuffer` recovery from our own
  exports for Metal consumers.
- Float64 columns run on a CPU path through the same API (Metal has no `double`).
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
