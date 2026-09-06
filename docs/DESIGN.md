# Design and future capability

How ArrowMetal is put together, what limits it today, and where the next 10x comes from.

## Layers
```
Python / Rust / Go / C# / R / C++  ──(Arrow C Data Interface)──►  libArrowMetalC (C ABI, handles)
                                                                          │
Swift apps ────────────────────────────────────────────────────►  ArrowMetal (Swift API)
                                                                          │
                                       MetalContext ── pipeline cache ── BufferPool
                                                                          │
                                   MSL kernels generated per element type, runtime compiled
                                                                          │
                                       Metal shared-memory buffers (page aligned, unified memory)
```
- **Buffers** are the Arrow columnar layout verbatim: validity bitmap + values (+ offsets later). The same bytes
  serve CPU consumers (C Data Interface) and GPU kernels (C Device Interface) with no copies.
- **Kernels** never assume a length multiple of anything: every buffer is padded to a page so trailing
  32-bit bitmap words are readable; kernels bounds-check the last word/element.
- **Nulls** ride along as bitmaps. Element-wise ops share the input's validity buffer (zero-copy); binary ops
  AND the two bitmaps on the GPU; compaction repacks.

## Where the time goes today (M4 Max, 50M rows)
| Kernel | Passes over data | Achieved | Bound |
|---|---|---:|---|
| sum / min / max | 1 read | 290-375 GB/s | memory |
| compare | 1 read, 1/64 write | 375 GB/s | memory |
| multiply / cast (vectorised) | 1 read, 1 write | 375 / 190 GB/s | memory |
| filter (count, scan, scatter, pack) | 2 reads, 1 write | 138 GB/s | passes |
| take (random gather) | random reads | 85 GB/s | latency |
| group-by, privatised (K <= 1024) | 1 read + TG atomics | see benchmarks | atomics |

Device peak on M4 Max is ~546 GB/s. Single-pass kernels sit at 55-70% of peak; the rest is command
buffer setup and the CPU wait.

## Latency (small inputs)
Every public call today is one command buffer: encode, commit, `waitUntilCompleted`. That costs roughly
100-300 µs per call regardless of size, so for arrays under ~1M elements the fixed cost dominates. The
`filter` kernels are already fused into one command buffer (count, scan, scatter) to avoid a CPU round trip.

Planned: a **pipeline** API where operations append to one command buffer and results are materialised on
first read (`let q = ctx.pipeline(); let s = q.filter(...).sum(); q.run()`), plus completion handlers for
async Swift. Expected to bring per-op overhead below 20 µs when batched, and to let the CPU keep working.

## Roadmap for "no room left"
1. **Pipelined execution** (above). Biggest win for query-shaped work and for Python callers.
2. **Fused expressions**: `filter(where:)` already fuses predicate + compaction. Next: `sum(where:)`,
   `groupBy.sum(where:)`, arithmetic chains compiled into one kernel from an expression tree.
3. **Reductions with vector loads** (`long4`) and tuned threadgroup counts per chip family.
4. **Filter in two passes** instead of four: compute selection + per-block counts in one kernel, then
   scatter with in-kernel decoupled look-back scan.
5. **Strings** (`utf8`, `utf8_view`): equality, prefix, length, hash; dictionary encoding on the GPU so
   group-by over strings maps to the dense-key path.
6. **Sort / top-k**: radix sort on 32/64-bit keys with payload; argsort for record batches.
7. **Hash group-by** for arbitrary keys (open addressing in device memory), feeding the same aggregators.
8. **Float64 sum/arithmetic** on the GPU via double-float (two `float`) arithmetic, or opt-in Float32.
9. **Binary archives** (`MTLBinaryArchive`) so the first call does not pay ~100 ms of shader compilation.
10. **Metal 4** command encoding and residency sets for very large resident datasets.
11. **iOS/visionOS**: the library already targets iOS 17+; add a demo app and thermal-aware sizing.
12. **Chip matrix**: publish benchmarks for M1/M2/M3/M4 base, Pro, Max, Ultra and A17/A18.

## Non-goals
- Not a query planner or SQL engine. It is the compute layer that DuckDB, DataFusion, Polars plugins or
  an app can call.
- Not a tensor library; MLX is that. A zero-copy bridge to MLX is planned.
