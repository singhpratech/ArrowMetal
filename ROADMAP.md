# Roadmap

Ordered roughly by impact divided by effort. Each item is a self-contained contribution. Open an issue to claim one.

## Near term (good first contributions)
- [x] **Take / gather** kernel (`take(indices:)`), the other half of selection.
- [x] **Count / any / all** for boolean arrays on the GPU.
- [x] **Cast** between primitive types, including float to int with Arrow's truncation rules.
- [x] **Float64 on GPU** for compare/min/max/filter/take via order-preserving bit patterns (exact).
      Still open: Float64 `sum` and arithmetic on the GPU (double-float emulation, or Float32 downcast opt-in).
- [ ] **Checked arithmetic** (`add_checked` etc.) that reports overflow and division by zero like Arrow.
- [x] **Slicing with offsets** without materialising (`offset != 0` currently copies on import).
- [x] **NaN semantics** for float min/max matching Arrow's `min_max` (skip NaN vs propagate).
- [ ] Benchmarks on M1/M2/M3 and on iPhone/iPad; a results table per chip.

## Medium term
- [x] **RecordBatch**: multiple columns, struct import/export, C Stream import. Open: C Stream export,
      C Device Stream, nested struct children.
- [x] **Group-by** over dense keys (sum/count/min/max/mean). Open: hash group-by for arbitrary keys, 64-bit min/max.
- [ ] **Sort / argsort** (radix sort on the GPU), then **top-k**.
- [ ] **Strings** (`utf8`, `large_utf8`, and `utf8_view`): equality, prefix match, length, hashing.
- [ ] **Dictionary-encoded** arrays: compare and filter on codes without decoding.
- [ ] **Async / pipelined API** (see docs/DESIGN.md): return command buffers or Swift `async` results instead of blocking per kernel; fuse
      kernels into one command buffer; expose `MTLSharedEvent` through `sync_event` in the device interface.
- [ ] **Metal 4** command-encoding path and residency sets for very large columns.

## Integrations
- [ ] `ArrowMetalSwiftArrow`: convenience conversion to and from `apache/arrow-swift` arrays.
- [ ] `ArrowMetalMLX`: zero-copy bridge to `MLXArray` for feeding columns into models.
- [x] Python package over the C ABI with `__arrow_c_array__`. Open: wheel packaging with the dylib inside, `__arrow_c_device_array__`.
- [ ] A DuckDB or DataFusion user-defined function that offloads a scan+filter+aggregate to ArrowMetal.

## Project
- [ ] Decide the final project name (see the trademark note in the README) and publish to Swift Package Index.
- [ ] GitHub Actions on `macos-15` runners (Metal works on the hosted Apple silicon runners).
- [ ] DocC documentation site.
