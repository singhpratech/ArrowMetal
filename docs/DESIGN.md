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
| grouped min/max (any width) | 2 reads + 32-bit atomics | see below | memory |
| counting sort by group id | 2 reads, 1 write | see below | memory |

Device peak on M4 Max is ~546 GB/s. Single-pass kernels sit at 55-70% of peak; the rest is command
buffer setup and the CPU wait.

## Group-by

Dense group ids `0 ..< K` come out of `GroupByKeys`; everything below aggregates over them. There are
three shapes, and the point of the current design is that **none of them sorts the key column**.

**Atomic accumulation** (`Kernels/GroupBy.swift`, `GroupBySource.swift`) is one linear pass. For
`K <= 1024` each threadgroup keeps a private table in threadgroup memory and merges it into the device
table once; above that the updates go straight to device memory. 64-bit sums are a pair of 32-bit atomic
adds with an explicit carry, because Metal has no 64-bit atomic.

**Sort-free extremes** (`Kernels/GroupByExtrema.swift`) is how `min`, `max` and `hash_min_max` avoid the
missing 64-bit atomic. Every element type maps order-preservingly into a 64-bit unsigned key. Pass one
takes the extremes of that key's **high** word with 32-bit atomics and counts the values; pass two takes
the extremes of the **low** word among only the rows whose high word already equals the winner. Some row
attains the winning high word, and among those the smallest low word is the overall minimum, so the
answer is exact. Types four bytes wide or narrower skip pass one — their whole key is the low word.
`first` / `last` ride the same kernel over a masked row index.

**Counting sort by group id** (`Kernels/GroupOrder.swift`, `GroupOrderSource.swift`) replaces the argsort
for the aggregates that genuinely need each group's rows together — `list`, `distinct`, `product`, the
central moments, the order statistics. Histogram, scan, scatter: one pass over the keys rather than four
over keys and payload. Two scatters share the scan:

- *chunked*, when a per-block histogram fits the budget. A "block" is one **simdgroup**, so a row is
  ranked against 32 lanes rather than 256, and each block owns a reserved slice of every group's run and
  fills it in row order. Stability is exact and needs no atomics.
- *atomic*, when the group count is large. One atomic bump per row places the rows, then `cs_fix`
  (a thread per group) or `cs_fix_tg` (a threadgroup, bitonic) sorts each run back into row order. Cheap
  exactly while runs are short, which is the case that forced this path; a longer run falls through to
  the argsort, so no input is worse off than before.

The ordering is cached on the `GroupBy`, so N aggregates over one grouping pay for it once.

**Segmented reductions** come in two shapes. `_wide` gives a threadgroup to a group and tree-reduces in
threadgroup memory; `_narrow` gives a single thread to a group. Ten million groups of five rows want the
second one — the first would launch a quarter of a billion threads to do fifty million adds.

**Grouped variance, stddev, skew, kurtosis** (`Kernels/GroupMoments.swift`) run the two-pass shifted
algorithm over that ordering in **true binary64**: an exact `d_add` sum for the per-group mean, then the
sums of `d = x - mean` and `d^2` (and `d^3`, `d^4`) about it, finished as
`m2 = (sum d^2 - (sum d)^2 / n) / (n - ddof)`. Shifting first is what makes it accurate — `sum x^2 -
(sum x)^2 / n` cancels catastrophically on data far from zero — and the `(sum d)^2 / n` term removes the
error in the mean itself. On 200k float64 values offset to 1e9 in 997 groups, the worst relative error
against an exact rational reference is **1.9e-16**; pyarrow's own answer on the same input is 2.5e-10
off. The Float32 deviations this replaced cost about 1e-5.

Finalizers write the output column and its validity **on the GPU**. A Swift loop over the group count is
invisible at a thousand groups and is the whole cost at ten million.

### Before and after (Apple M4 Max, best of five, 10M rows, one `group_by` per call)

| grouped aggregate | groups | before | after | change |
|---|---:|---:|---:|---:|
| min (int64 values) | 1,000 | 18.7 ms | 2.7 ms | 6.9x |
| min | 100,000 | 35.8 ms | 3.5 ms | 10.2x |
| min | 10,000,000 | 1189 ms | 17.0 ms | 70x |
| max | 1,000 | 18.8 ms | 2.7 ms | 7.0x |
| max | 100,000 | 36.0 ms | 3.5 ms | 10.2x |
| max | 10,000,000 | 1228 ms | 17.8 ms | 69x |
| mean | 10,000,000 | 338 ms | 30.3 ms | 11x |
| variance (float64, ddof=1) | 1,000 | 512 ms | 6.5 ms | 79x |
| variance | 100,000 | 500 ms | 11.1 ms | 45x |
| variance | 10,000,000 | 823 ms | 34.8 ms | 24x |
| first | 100,000 | 10.7 ms | 4.4 ms | 2.4x |
| first | 10,000,000 | 480 ms | 18.1 ms | 27x |
| list | 1,000 | 20.3 ms | 6.7 ms | 3.0x |
| list | 100,000 | 20.8 ms | 11.6 ms | 1.8x |
| sum | any | unchanged (already the atomic path) | | |
| count | any | unchanged | | |
| count_distinct | any | unchanged — still two full radix sorts | | |

At 50M rows, against the fastest of Polars / pyarrow / pandas: min 2.5x / 3.9x / 5.5x at 1k / 100k / 10M
groups, where it was 0.25x / 0.56x / 0.21x; variance 0.71x / 3.3x / 4.9x, where it was 0.05x at 100k;
`first` 5.9x / 6.0x / 5.2x; `list` 10.2x / 8.1x / 5.2x. Two caveats on those ratios. At a **thousand**
groups every grouped aggregate lands near 2.5x, `sum` and `count` included, because a fresh
`group_by([...])` spends about 7 ms of the 10 ms rebuilding the dense key mapping — reuse the object and
the aggregate itself is 3 ms. And **`count_distinct` is the remaining shortfall** (0.55x): it still
dictionary-encodes the values with one radix sort and collapses the packed `(group, code)` pairs with
another, where the CPU libraries keep a hash set per group. A segmented sort of the values inside each
group's counting-sort run, or a per-group hash, is the fix — but the sort of the *key* column is gone
from every aggregate.

## Latency (small inputs)
Measured floor on M4 Max: an empty kernel with encode + commit + wait costs ~116 µs; ten kernels in one
command buffer cost ~100 µs in total. So the fixed cost is the round trip, not the kernel, and the only lever
is fewer round trips.

Unbatched, every public call is one command buffer (~140-170 µs fixed). `filter` already fuses its three
kernels into one command buffer with a GPU scan.

**Batched execution** (`MetalContext.batch { }`, `am_batch_begin/end` in C, `with am.batch():` in Python):
kernels append to one serial compute encoder; results are created immediately with their buffers, but a
result whose length the GPU decides (filter) or whose null count needs a bitmap read is marked *pending*.
Any CPU-side read (a reduction, `length`, `nullCount`, subscripts, export) triggers a *sync point*: the
open command buffer is committed and waited once, deferred fix-ups run (lengths, null counts, `take`
bounds errors), and a fresh batch is opened so the caller keeps batching. While any batch is open the
pool parks returned buffers instead of recycling them, so pending GPU work can never observe a reused
buffer. Per-thread batches; nested `batch` calls join the outer one.

**Lengths flow on the GPU.** Every kernel takes its element count as `device const uint* nPtr`. A pending
filter result binds the GPU-written total as that pointer and is dispatched at its worst-case size, so the
next kernel (compare, arithmetic, cast, bitmap ops, another filter, a reduction's partial pass) never needs
the CPU to know the length. A reduction still syncs once to read its partials.

**Float64 without hardware doubles.** `Kernels/DoubleMath.swift` carries a software IEEE-754 binary64
(`d_add`, `d_sub`, `d_mul`, `d_div`) over `ulong` bit patterns, correctly rounded. Sum accumulates with
`d_add`; arithmetic kernels run one element per thread. It is slower than native Float32 math but still
memory-bound at 50M rows, and it means no Float64 column ever falls back to the CPU.

### Async
`MetalContext.batchAsync` is `batch { }` without the wait. It records `body`'s kernels into one command
buffer exactly as `batch` does, then commits and hangs an `MTLCommandBuffer.addCompletedHandler` off it
instead of spinning. The deferred fix-ups — `afterFlush` closures (pending lengths, null counts, `take`
bounds errors), `pool.releaseParked()`, the `openBatches` decrement — run on the GPU's completion thread,
so the calling thread is released the moment the work is committed. Two forms:

```swift
let kept = try await ctx.batchAsync { try amount.filter(where: .gt, 100) }   // suspends, no thread held
ctx.batchAsync({ try amount.filter(where: .gt, 100) }) { result in ... }     // returns immediately
```

`flush` is split into `detachBatch()` (end encoding, unhook from the thread) and `finishBatch(_:)` (the
post-completion fix-ups, which touch no thread-local state); the synchronous path calls both back to back
around its wait, the async path puts the completion handler in between. Batches stay per-thread: `body`
runs synchronously, before the first suspension, so the batch is opened, filled and detached without a
thread hop. Nested calls join the enclosing batch, as with `batch`.

Sync points still exist *inside* `body`. A reduction, `length` or `nullCount` of a pending filter result, a
subscript or an export commits the open batch and blocks right there, exactly as in `batch { }`. So `body`
should return something that does not force a sync — a pending array from `filter`, `compare`, `take`,
`cast` or arithmetic. Its length, null count and contents are resolved by the time the `await` returns.
For a scalar there is an async accessor: `MetalArray.sumAsync()` (and `meanAsync()`) records the reduction
kernel in an async batch and combines the per-threadgroup partials on the completion path, so nothing
blocks. Deferred errors are thrown from the `await` (or delivered as `.failure`), not from the recording.

Not yet: async accessors for `min`/`max`/group-by, and cancellation (a committed command buffer runs to
completion).

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
