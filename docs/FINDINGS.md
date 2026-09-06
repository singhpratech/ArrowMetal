# Findings and gotchas

Things learned the hard way. Add to this whenever something surprises you.

## Ecosystem research (2026-09-06)
- `apache/arrow-swift` v21: types, IPC, Flight, C Data Interface. No compute kernels, nothing Metal. ~32 stars.
- `arrow-nanoarrow` device extension: C wrapper for `ARROW_DEVICE_METAL` buffers, no compute.
- cuDF: CUDA only. MLX: unified-memory tensors, zero-copy `MTLBuffer` access, no nulls or columnar semantics.
- A DuckDB extension with Metal aggregates exists (gpudb); it does not use Arrow buffers.
- The Arrow C Device Data Interface defines `ARROW_DEVICE_METAL = 8` and expects `MTLEvent*` as `sync_event`.

## Toolchain
- XCTest is not in the Command Line Tools. Run tests with `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`.
- Swift 6.3.3 miscompiled a `withUnsafeBytes` closure inside a throwing generic convenience init under `-O`
  (release-only crash on entry). Rewritten as a loop. Always run `swift test -c release`.
- GitHub `macos-15` runners ship Xcode 16.4 / Swift 6.1, which refuses to type-check dense closures that
  Swift 6.3 accepts. Keep test expressions simple.
- Swift release builds shorten object lifetimes to last use: a raw pointer taken from a buffer object can
  outlive the object. Use `withExtendedLifetime` or the closure accessors.
- `posix_memalign` memory is not zero for small blocks (recycled heap); only fresh mmap pages are zero.

## Metal
- Apple M4 Max: 32 KB threadgroup memory, SIMD width 32, unified memory, no 64-bit atomics from MSL
  (`atomic_ulong` fetch_add/max fail to compile). 64-bit sums use split 32-bit atomics with carry.
- Runtime `makeLibrary(source:)` works without the `metal` toolchain. Errors carry line numbers of the
  generated source.
- `makeBuffer(length:)` buffers are heap sub-allocated and not page aligned; `bytesNoCopy` requires page
  alignment of pointer and length.
- Memory-bound element-wise kernels run at ~375 GB/s on M4 Max once allocation overhead is removed; the
  16-core CPU reaches ~350 GB/s on the same loops. Reductions and compaction favour the GPU by 2x to 3x.
- Fusing the predicate into the filter's counting pass avoids materialising a boolean array.
