# Decision log

Architecture and product decisions, newest first. Each entry says what was decided, why, and what it rules out.

## 2026-09-06: Version stays 0.1.0 until the public launch
Everything built before going public is one release. Internal rounds are tracked in the changelog under a
single unreleased 0.1.0 heading and in docs/BENCHMARKS.md as numbered rounds.

## 2026-09-06: Batched execution with lazy materialisation instead of futures
A per-thread open command buffer with sync-on-read keeps the synchronous API (every method still returns a
real array) while removing the ~120 µs round trip between chained kernels. Futures would have changed every
signature. Cost: a reduction inside a batch still syncs; filter results carry a worst-case buffer until read.

## 2026-09-06: One C ABI, thin idiomatic wrappers per language
No per-language ports. `libArrowMetalC` takes and returns Arrow C Data Interface structs plus opaque handles.
Python, Rust, Go, C#, R, C++ all have Arrow bindings that produce and consume those structs.

## 2026-09-06: Name is ArrowMetal
"AppleArrow" and "AppleMetalArrow" were rejected: "Apple" in a product name invites a trademark complaint and
implies Apple built it. "Arrow" is an Apache Software Foundation mark; the README carries a non-endorsement note.

## 2026-09-06: Swift host, MSL kernels, C ABI only at the border
Swift with unsafe pointers matches C for CPU loops (the 16-core baselines reach memory bandwidth), gives native
Metal API access, ARC for buffer lifetime, iOS, and SwiftPM. The hot code is Metal Shading Language regardless.
C matters only for foreign-language bindings, which the Arrow C Data Interface already covers.

## 2026-09-06: No dependency on MLX or arrow-swift
MLX is a tensor library without nulls or columnar semantics; arrow-swift has no compute. Both become optional
bridges through the Arrow C interfaces rather than foundations.

## 2026-09-06: Runtime shader compilation from generated MSL
No `.metal` files, no offline `metal` compiler, builds with Command Line Tools only. Kernels are string
templates specialised per element type and cached per pipeline. Cost: ~100 ms first-use compile per library
(binary archives are on the roadmap).

## 2026-09-06: Page-aligned, pooled buffers via `posix_memalign` + `makeBuffer(bytesNoCopy:)`
Metal sub-allocates small `makeBuffer(length:)` buffers from a heap, so they are not page aligned and cannot
be re-wrapped zero-copy by another Metal consumer. Own allocation guarantees alignment. A size-bucketed pool
avoids mmap and page-fault costs on repeated allocations, which cut element-wise kernel times by 2x to 3x.
Kernel outputs that are fully written skip zeroing.

## 2026-09-06: Reductions without atomics
Each threadgroup writes a partial; the host (or a scan kernel) finalises. Deterministic float sums, works for
64-bit integers (Apple GPUs expose no 64-bit atomics from MSL).

## 2026-09-06: Float64 on the GPU through order-preserving bit patterns
Metal has no `double`. Compare, min, max, filter, take and slice treat Float64 as `long` with an
order-preserving key (NaN and signed zero handled). Sum and arithmetic stay on the CPU until double-float
emulation lands.

## 2026-09-06: NaN semantics
`min`/`max` skip NaN and return null if only NaN remains; `sum` propagates NaN; comparisons follow IEEE.

## 2026-09-06: Benchmarks compare against all cores and against Polars
Single-core baselines were removed. Baselines are 16-core Swift loops, Accelerate on all cores, and Polars,
pyarrow and pandas (multi-threaded) on data of identical shape.
