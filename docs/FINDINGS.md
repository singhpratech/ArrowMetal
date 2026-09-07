# Findings and gotchas

Things learned the hard way. Add to this whenever something surprises you.

## Ecosystem research (2026-09-06)
- `apache/arrow-swift` v21: types, IPC, Flight, C Data Interface. No compute kernels, nothing Metal.
- `arrow-nanoarrow` device extension: C wrapper for `ARROW_DEVICE_METAL` buffers, no compute.
- cuDF: CUDA only. MLX: unified-memory tensors, zero-copy `MTLBuffer` access, no nulls or columnar semantics.
- A DuckDB extension with Metal aggregates exists (gpudb); it does not use Arrow buffers.
- The Arrow C Device Data Interface defines `ARROW_DEVICE_METAL = 8` and expects `MTLEvent*` as `sync_event`.

## Toolchain
- XCTest is not in the Command Line Tools. Run tests with `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`.
- Swift 6.3.3 miscompiles a `withUnsafeBytes` closure inside a generic `throws` function under `-O` when the caller is in another module: the closure clobbers the error register around an Objective-C message send, so the function reports a phantom error and the caller crashes retaining it (swiftlang/swift#90477, open). `MetalArray.init(_:)` uses a plain element loop instead; the Metal-free reproducer is in UPSTREAM.md
  (release-only crash on entry). Rewritten as a loop. Always run `swift test -c release`.
- GitHub `macos-15` runners ship Xcode 16.4 / Swift 6.1, which refuses to type-check dense closures that
  Swift 6.3 accepts. Keep test expressions simple.
- Swift release builds shorten object lifetimes to last use: a raw pointer taken from a buffer object can
  outlive the object. Use `withExtendedLifetime` or the closure accessors.
- `posix_memalign` memory is not zero for small blocks (recycled heap); only fresh mmap pages are zero.

## Metal
- Apple M4 Max: 32 KB threadgroup memory, SIMD width 32, unified memory, no 64-bit atomic add from MSL
  (`atomic_ulong` fetch_add fails to compile; min and max do). 64-bit sums use split 32-bit atomics
  with carry.
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
  exporting a result back to pyarrow costs nothing measurable (3.55 ms vs 3.57 ms with export).
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

## Round 8b (2026-09-06): the differential matrix over the whole type surface

Extending `python/tests/test_differential.py` to every type ArrowMetal imports (45 columns, 181
operations, 33,156 cases at the time; 212 operations and 39,069 cases today) turned up three bugs and
one unreproduced crash **in pyarrow 25.0.1**, not in ArrowMetal.
They are recorded here because the harness has to work around them, and each has a test that fails if a
later pyarrow fixes it.

- A **single unreproduced segfault** was observed a few allocations after `pc.year_month_day` on a
  `timestamp[s]` array with nulls; it has not recurred. It landed in whatever unrelated call happened
  next — the faulting frame was `Array.nbytes` inside the harness's own array cache, which cost an
  hour to trace back. Out of caution the matrix does not use the two struct-valued temporal kernels as
  oracles: it compares `iso_calendar` and `year_month_day` field by field against `pc.iso_year`/
  `pc.iso_week`/`pc.day_of_week` and `pc.year`/`pc.month`/`pc.day`, which is a stronger check anyway.
- `pc.utf8_normalize` **ignores its `form` option**: NFC and NFKC come back decomposed, so its NFC is
  NFD. Python's `unicodedata` and ArrowMetal agree with each other and with the Unicode annex; the
  matrix uses `unicodedata` as the oracle. Pinned by
  `test_pyarrow_utf8_normalize_ignores_its_form_option`.
- `pc.pairwise_diff` **ignores `ArrowArray.offset`**: on a sliced column it reads the values buffer
  from the start and answers with the wrong rows. Only visible when the values are not an arithmetic
  progression, which is why it hid for a while. The matrix hands that oracle a materialised copy while
  ArrowMetal still gets the slice, so the case remains a test of the offset handling. Fixed upstream in
  pyarrow 26.0.0; the workaround stays only while 25.0.1 is the pinned version.
- `pc.fill_null_forward`, `pc.fill_null_backward` and `pc.replace_with_mask` have the same offset bug
  on a **boolean** column (the values bitmap, not the validity one). Same mitigation.

Two things about the harness itself that were not obvious:

- `pc.add` on a `time32`/`time64` column validates the *values under the validity bitmap*, so a null
  row whose hidden value would leave `[0, 86400)` makes the oracle raise even though the row is null.
  The generator deliberately puts real numbers under the null bits, so the time-of-day cases have to
  drop their null rows rather than mask them.
- A `Decimal` that came out of a 38-digit column cannot be scaled with `Decimal.scaleb` or multiplied
  by `10 ** scale` under the default decimal context — 28 digits of precision silently round it. Read
  the unscaled magnitude off `as_tuple().digits` instead.

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

## Round 9 (2026-09-06): radix select, and the LSD radix sort's blocking at small n

**The two-pass trick.** A GPU radix select normally costs three passes over the column: histogram the top
digit, count how many rows fall in the selected range per block, then scatter them in order. The count pass
is redundant if the histogram keeps its counts **per sub-block** instead of only globally — the digit-major
table `counts[digit * subBlocks + sub]` is simultaneously the global histogram (summed over sub-blocks) and
the offset table the scatter needs (summed over digits <= target). At 50M rows that is 1 ms saved out of 3.

**One sub-block per simdgroup, not per threadgroup.** Making the output granularity a simdgroup's slice
rather than a threadgroup's removes every `threadgroup_barrier` from the scatter: the rank of a selected row
within its slice is `simd_prefix_exclusive_sum` over 32 lanes plus a running base each lane computes
identically. The cost is an 8x larger count table (256 * 8 * groups words, ~8 MB at 50M rows), which is
noise next to the 400 MB the pass reads anyway.

**The final sort was the bottleneck, and it was a blocking bug.** After selection there are only ~200k
candidates left to order, but `argsort` of 200k UInt64 measured 2.2-2.9 ms — as slow as sorting 800k. The
cause was `elemsPerBlock = 4096` fixed: 200k rows is 49 threadgroups, 20k rows is *five*, on a GPU with 40
cores, and `radix_scatter` does an O(TG) rank loop per element that nothing else can overlap. Halving the
block size until there are at least 64 blocks (inputs above ~256k rows are untouched, so the 50M argsort is
unchanged) took argsort of 20k from 1.99 ms to 0.41 ms and of 200k from 2.20 to 1.34 ms, and it is most of
why `top_k(100)` went from 5.6 ms to 2.9 ms.

**Refine on the compacted array, not the column.** The selected bin is ~n/256 rows, which still dominates
the final ordering. Running the same histogram + compaction *again* over the compacted candidates (a few
hundred thousand keys, microseconds) shrinks it by another 256x. One extra round trip, and it takes the
survivors under the 2048 pairs a single-threadgroup bitonic sort can order in one dispatch.

**Benchmark data hides skew.** `rng.integers(-(2**62), 2**62)` spreads over only 128 of the 256 top-digit
bins, so the candidate bin is n/128, not n/256. Worth remembering when reading a selection benchmark: the
bin size, and therefore the final sort, depends entirely on the key distribution's top byte.

**A float key's top byte is the exponent, so float columns are the skewed case.** The top byte of a float64
key is the sign plus seven exponent bits, so a column of uniform doubles concentrates in a handful of bins
rather than spreading over 256, and the bin holding the wanted rank can be a large fraction of the column.
When it is over the compaction budget the search narrows another digit over the column instead, one more full
pass — which is why the same 50M Float64 median measures anywhere between 2.7 and 3.9 ms depending on exactly
where the rank lands, while Int64 keys are stable. Raising the budget so the big bin gets compacted instead is
faster still, but a 25M-row bin needs 300 MB of scratch for a 400 MB column, and a median already 100x faster
than pyarrow is not worth a 75% memory overhead.
