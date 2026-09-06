# Changelog

## 0.2.0 (2026-09-06)
- `take` (gather) with Int32/Int64/UInt32 indices, null indices and GPU bounds checking.
- `cast` between all primitive types; `slice` with zero-copy views at 32-element alignment.
- Float64 compare/min/max/filter/take/slice on the GPU via order-preserving bit patterns.
- NaN semantics: min/max skip NaN (null if nothing else), sum propagates, IEEE comparisons.
- Boolean `filter`, `take`, `slice`, `count`, `any`, `all`.
- `MetalRecordBatch` with filter/take/slice/selecting; struct (`+s`) C Data Interface import/export;
  `ArrowArrayStream` import; device export of batches.
- Examples executable with five end-to-end scenarios; benchmark rows for take, Float64 and cast.

## 0.1.0 (2026-09-06)
- Initial release: Metal shared-memory Arrow buffers and primitive/boolean arrays.
- GPU kernels: sum, min, max, mean, compare, add/sub/mul/div, filter, boolean and/or/not.
- Arrow C Data Interface and C Device Data Interface (ARROW_DEVICE_METAL) import and export.
- CPU reference implementation and test suite; benchmark executable.
