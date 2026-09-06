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

## Round 3 (2026-09-06)
- Metal integer division by zero returns an unspecified value (observed 1 on M4 Max). ArrowMetal now
  defines it as 0 in both the GPU kernels and the CPU reference, and `Int.min / -1` wraps instead of trapping.
- Threadgroup-privatised group-by with 32-bit atomics reaches ~300 GB/s for up to 1024 keys, the same rate
  as a plain sum: the atomics are not the bottleneck at that key count. Device atomics at 100k keys halve it.
- Crossing the C boundary from Python costs nothing measurable per call (ctypes overhead is ~10 µs);
  exporting a result back to pyarrow is zero-copy (3.55 ms vs 3.57 ms with export).
- Importing pyarrow buffers is one memcpy: pyarrow's allocator is 64-byte aligned, not page aligned.

## Round 4 (2026-09-06)
- GitHub's `macos-15` Apple silicon runners expose an "Apple Paravirtual device" GPU with 3 CPU cores:
  ~16 GB/s for a sum. Use CI for correctness only; never publish its numbers.
- The paravirtual GPU failed `makeComputePipelineState` for kernels that pass on M4 Max (round 3 push).
  `ARROWMETAL_DEBUG_SHADERS=1` now dumps generated MSL and the full compiler log on failure; CI sets it.
- `makeCommandBufferWithUnretainedReferences` plus a spin-wait did not measurably reduce per-call latency;
  the ~120 µs floor is the round trip. Batching is what works: chains pay it once.
- A `sum()` on a batched filter result still costs a second round trip because the reduction needs the
  filtered length on the CPU. Next: kernels read `n` from a device buffer so pending lengths flow on the GPU.
