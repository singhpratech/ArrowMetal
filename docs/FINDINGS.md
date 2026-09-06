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

## Round 5 (2026-09-06)
- With full compiler logs enabled, the paravirtual GPU on GitHub runners fails `makeComputePipelineState`
  for arbitrary trivial kernels (`bitmap_not`, `cast_kernel`) with no diagnostic while identical kernels
  pass in the same run. It is the virtual Metal stack, not a construct. Mitigation: one retry on pipeline
  creation, and GPU tests `XCTSkip` on devices whose name contains "Paravirtual" (`ARROWMETAL_FORCE_GPU_TESTS=1`
  overrides). CI validates the build, interop and CPU paths; GPU correctness runs on real hardware.
- Kernels now read their element count from a device buffer (`device const uint* nPtr`). A pending
  filter result binds its GPU-written total as that buffer, so compare / arithmetic / cast / bitmap ops /
  another filter / a reduction can consume it inside the same command buffer with worst-case dispatch sizes.
  Reductions still sync to read partials, but a filter followed by a sum is now one round trip.

## Round 6 (2026-09-06): software IEEE-754 double on the GPU
- Add, subtract and multiply implemented on `ulong` with 3 guard bits and sticky are bit-exact against Swift's
  `Double` for 1M random pairs including subnormals, signed zeros, infinities and NaN (`d_finish` handles
  normalisation, subnormal shift-with-sticky, round-to-nearest-even, and rounding carry).
- Division: a float-seeded Newton iteration was within 1 ulp only 93% of the time and wrong for subnormal
  inputs. Replaced with restoring long division on the significands (57 quotient bits). Two bugs on the way:
  one extra quotient bit shifted every result by 2x, and the restoring loop needs `rem < mb` before the first
  step (take the first quotient bit explicitly). Lesson: test division on ratios above and below 1.
- A preprocessor `#define` glued to the previous line (`}#define`) because Swift multi-line strings drop the
  final newline. Generated MSL fragments that start with a directive must begin with a newline.
- Float64 sum on the GPU accumulates with `d_add` in tree order; per-threadgroup partials are combined on
  the CPU in `Double`. Results differ from a sequential CPU sum only by normal floating-point reordering.

## Round 7 (2026-09-06): strings and sort
- Generated MSL written through a shell heredoc must use `\(K)` (one backslash) for Swift interpolation; a
  doubled backslash reaches the Metal compiler as literal text. Same newline rule as before for `#define`.
- LSD radix sort with 8-bit digits, 4096-element blocks and a stable in-chunk ranking is correct across all
  types and sizes; descending order must invert keys rather than reverse the ascending result, or ties flip.
- MurmurHash3 x86_32 reference vectors (seed 0): "" -> 0, "a" -> 0x3c2569b2, "abc" -> 0xb3dd93fa, "hello" -> 0x248bfa47.
  Four collisions among 100k 32-bit hashes is normal (birthday bound), not a bug.

## Round 8 (2026-09-06): a threadgroup atomic read that is not uniform (`top_k`)

**Symptom.** One cell of a 13,000-case differential run diverged and then passed on every rerun:
`top_k / float64`, 100,003 rows, 30% nulls, k = 17, descending — `result[0]` was row 0 where pyarrow says
row 35254. A randomised stress against a CPU oracle (`TopKTests.testStressAgainstCPUOracle`) reproduced it
at roughly 1 in 900 calls across every key type, both directions, n from 32k to 1M and k from 1 to 1024.
The failure always looked the same: a run of zeros at the front of the result, the true answer after them.

**Root cause.** `topk_select` keeps a per-threadgroup candidate buffer in threadgroup memory with an atomic
count `held`. Every thread read that count itself:

    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint c = atomic_load_explicit(&held, memory_order_relaxed);   // Kernels/TopKSource.swift
    ...
    for (uint i = c + lid; i < cap; i += TG) { bufKey[i] = keyMax; bufRow[i] = TK_NOROW; }

`c` bounds the range each thread pads with sentinels, and the union over `lid` covers `[c, cap)` **only if
every thread has the same `c`**. It does not: on an M4 Max, one simdgroup out of eight occasionally comes
back with a newer value than the rest — a relaxed atomic load is not ordered by the preceding barrier the
way a plain threadgroup read is, so it can be satisfied late, after other threads have already run ahead
and incremented `held`. The threads holding the larger `c` start their stride later and the slots they
should have covered are never written. Instrumenting the kernel showed 7 to 63 such holes per threadgroup,
clustered at 32 and 64 — one and two simdgroups.

Those holes hold zeroed pool memory, which reads as `(key 0, row 0)`. That pair is the *minimum* of the
`(key, row)` order, so the bitonic sort moves it to the front of the buffer and it becomes the block's
answer; the host then sees row 0 as the top candidate. Worse, if a block accumulates k or more holes the
compaction sets its threshold to `bufKey[k-1] = (0, 0)`, which nothing can beat, and the block is blind for
the rest of the scan. The same read also decides `if (c + TG > cap)`, a branch containing
`threadgroup_barrier`, so a non-uniform `c` was undefined behaviour in its own right.

**Fix.** Thread 0 reads `held` once per chunk into a plain `threadgroup uint shared_c` (clamped to `cap`)
between two barriers, and every thread takes `c` from there; the branch and the fill ranges are now uniform
by construction. Belt and braces, the buffer is filled with sentinels once at kernel entry, so a slot no one
writes reads as "no row" and the host drops it instead of it masquerading as row 0.

**Cost.** One extra `threadgroup_barrier` per 256-row chunk, and one `cap`-element sentinel fill per block
(2 to 8 strided stores per thread, once). Below the noise floor: `top_k(100 of 20M Int64)` on M4 Max runs
3.11 / 3.31 / 3.38 ms with the fix against 3.22 / 3.30 / 4.37 ms without it. The pass is still about one
read per row.

**How it was found.** A stress test comparing against a CPU oracle (not against `argsort`, which shares the
sort keys) with pool-churning kernels in between; then poisoning the candidate buffers before the dispatch
to prove the kernel really wrote those zeros rather than leaving stale bytes; then a debug buffer carrying
each threadgroup's `held`, arrival count and compaction count, which showed the holes were inside `[0, c)`
and `[c, cap)` in simdgroup-sized runs.

**Rule.** In MSL, never let a value that decides a barrier-carrying branch, or a per-thread loop bound whose
strides must tile a range, come from a per-thread `atomic_load_explicit(..., memory_order_relaxed)`.
Broadcast it through a plain threadgroup variable between barriers. The other `atomic_load_explicit` sites
(`SortSource`, `GroupBySource`, `JoinSource`, `StringExtraSource`) were checked: each reads a slot the
calling thread owns, so none of them tiles or branches on a shared count.
