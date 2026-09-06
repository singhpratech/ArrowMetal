# Changelog

## 0.1.0 (unreleased, in development)
Everything below ships together as the first public release.

Core
- Metal shared-memory Arrow buffers (page aligned, pooled) and primitive/boolean arrays.
- Kernels: sum/min/max/mean, compare, add/sub/mul/div (vectorised, defined integer division by zero),
  filter (one command buffer, GPU scan) and fused filter(where:), take, cast, slice, boolean and/or/not/count/any/all.
- Float64 compare/min/max/filter/take/slice on the GPU via order-preserving bit patterns; NaN semantics.
- GroupBy over dense integer keys: count, sum, mean, min, max (privatised and device-atomic paths).
- MetalRecordBatch with filter/take/slice/selecting; struct (+s) C Data import/export; ArrowArrayStream import.
- Arrow C Data Interface and C Device Data Interface (ARROW_DEVICE_METAL) import and export.

Strings and sorting
- `MetalStringArray` (utf8): byte/char length, equals/starts_with/ends_with/contains, MurmurHash3, GPU filter/take,
  dictionary encoding; import/export through the C Data Interface (large_utf8 narrowed on import).
- GPU LSD radix sort: `argsort`, `sorted`, `topK`, `MetalRecordBatch.sorted(by:)`; stable, nulls last, IEEE total order.

Execution model
- `MetalContext.batch { }`: one command buffer per chain; kernels read lengths from device buffers so pending
  filter results chain without a CPU sync; pool parks buffers while a batch is open.
- Software IEEE-754 Float64 (add/sub/mul/div/sum) on the GPU, bit-exact against the CPU.

Bindings
- libArrowMetalC C ABI (include/arrowmetal.h) and python/arrowmetal ctypes package (Arrow PyCapsule protocol).

Quality
- CPU reference for every kernel; 26 tests including a scenario matrix over every type, null density, size
  and sliced input; concurrency and pool tests; CI on hosted Apple silicon in debug and release.
- Benchmarks: Swift vs all-core CPU vs Accelerate; Polars/pyarrow/pandas; ArrowMetal from Python in-process; latency mode.
