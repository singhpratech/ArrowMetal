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
(`d_add`, `d_sub`, `d_mul`, `d_div`, `d_sqrt`) over `ulong` bit patterns, correctly rounded. Sum
accumulates with `d_add`; arithmetic kernels run one element per thread. `d_div` is a Newton reciprocal
seeded by one hardware `float` division, with the exact 128-bit remainder `N - q·D` settling the last
bit; `d_sqrt` extracts the root digit by digit in integers. Both are correctly rounded rather than close,
and `DoubleMathTests` holds them to Swift's own `Double` bit for bit. `add` and `multiply` run at this
machine's memory ceiling (≈390 GB/s at 50M rows) and `divide` within 15% of it (331 GB/s), so no Float64
column ever falls back to the CPU and the software arithmetic is all but invisible in a bandwidth-bound
query.

The transcendentals are a different story, and worth being explicit about. `Kernels/DoubleTranscendental.swift`
(`expm1`, `log1p`, `logb`, `hypot`, the ten `RoundMode`s) and `Kernels/DoublePower.swift` (`exp`, `ln`,
`log2`, `log10`, `pow`) evaluate in binary64 from end to end, over that same software arithmetic —
**not** by narrowing the column to `float`, calling Metal's own library and widening back, which is what
these kernels used to do for about seven correct significant decimal digits out of sixteen. `pow` is the
one that sets the bar: a 1-ulp result needs the product `y·log2 x`, which reaches 1024 in magnitude,
accurate to 2⁻⁶¹ absolutely — more than a double holds — so `log2 x` is carried as an unevaluated hi/lo
pair whose high part keeps 21 significant bits, and `y` is split the same way so that `y₁·t₁` is an exact
double. That is fdlibm's layout, and its Remez coefficients and hi/lo constants are reused verbatim; the
three logarithms fall out of the same reduction, more accurately than a direct series would give them.

Accuracy is measured, not derived. `DoubleTranscendentalTests` compares each function with Foundation
over 10⁶ random inputs drawn across its whole domain and prints the ulp histogram; these are those
numbers:

| function | domain sampled                                        | max ulp |
|----------|-------------------------------------------------------|---------|
| `sqrt`   | every finite positive bit pattern, subnormals included | **0** (bit-identical) |
| `exp`    | -745.2 to 709.78, plus the subnormal-result and near-overflow edges | 1 |
| `ln`     | 5e-324 to 1.8e308, plus near 1 and the subnormals      | 1       |
| `log2`   | same                                                   | 1       |
| `log10`  | same                                                   | 1       |
| `power`  | 10⁶ random pairs, 5·10⁵ negative bases with integer exponents | 1 |

The C99 edge table — `x^0`, `0^y`, `1^y`, `(-1)^int`, infinity and NaN propagation — matches libm bit for
bit, and the `_checked` twins are the unchecked kernel plus a read-only check pass, so they inherit every
value and raise at exactly the same boundaries.

The price is throughput, and it is the honest cost of the accuracy. At 50M rows on an M4 Max, the old
`float`-detour `ln` ran at 250 GB/s because it was memory bound; the binary64 one runs at 10 GB/s because
it is compute bound on forty-odd software operations per element. That is a 25x throughput loss for nine
more correct digits, and it is the right trade for a library whose whole claim is that a Float64 column
means Float64. The float32 kernels are untouched and still take the hardware path.

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

## Top-k and order statistics: GPU radix select

`top_k`, `quantile` and `approximate_median` all ask the same question — which rows sit at which rank — and
all of them used to answer it by sorting the whole column. Sorting 50M Int64 is eight LSD radix passes and a
gather, about 130 ms; the answer needs a few hundred bytes of it. Radix select replaces the sort with two
passes over the column, whatever k is.

### The algorithm

Everything runs on the **order-preserving key**: the same map `argsort` uses (`TopKSource.keyMap`), so nulls,
NaN, -0.0 and the ascending/descending flag behave identically and the answer is exactly
`argsort(descending:)[..<k]`, index for index.

1. **Histogram** (`rs_histogram`, `Kernels/RadixSelectSource.swift`). One pass counts the top 8 bits of every
   non-null row's key. The counts are kept **per simdgroup-sized sub-block** — one 256-bin private histogram
   per simdgroup in threadgroup memory, so the only atomic contention is between 32 lanes — and written
   digit-major as `counts[digit * subBlocks + sub]`. Summed over sub-blocks that is the global histogram;
   left per sub-block it is exactly the offset table step 3 needs. That double duty is what keeps the fast
   path at two passes instead of three.
2. **Pick the bin.** The host walks the 256 totals to the first digit whose running total reaches k. Rows
   with a smaller digit are guaranteed winners — there are fewer than k of them — and the rest of the answer
   lives in that one bin, about n/256 rows for well-spread keys.
3. **Compact** (`rs_scatter`). A second pass writes the winners and then the bin's rows, both in ascending
   row order, into one array of `(key, row)` pairs. Every winner's key is smaller than every bin key, so a
   *stable* sort by key alone turns that array into (key, row) order — the total order — with no second sort
   key and no 96-bit comparisons. Because each sub-block owns a contiguous slice of the output, the scatter
   needs no threadgroup barriers: a `simd_prefix_exclusive_sum` per 32 rows is the whole ranking.
4. **Refine.** The bin is still ~n/256 rows, far more than the answer, and it would dominate the final sort.
   So steps 1-3 run again on the *compacted array* and the next digit — microseconds of work that shrinks the
   bin by another factor of 256. One refinement round is enough; the loop stops when the bin is under
   `TopK.refineCap` (4096) rows.
5. **Order.** Up to `TopK.bitonicCap` (2048) survivors are sorted by a bitonic network in a single
   threadgroup (`rs_sort_small`, one dispatch, the slot index as the tie-break), which covers every k up to
   about 1024. Above that the stable LSD radix sort orders them and the first k rows are the answer.

**Ties.** Heavily tied data can put nearly every row in the chosen bin, which would make step 3 copy the whole
column. Step 2 therefore repeats on the next digit *without* compacting while the bin is over the candidate
budget (1M rows). When the key runs out of digits every remaining row is an exact tie, and ties are settled by
row order, so the scatter writes only the first `k - winners` of them: the output is exactly k pairs however
many rows share the key.

The same in-place narrowing covers the other skewed case: a float64 key's top byte is the sign plus seven
exponent bits, so a column of uniform doubles concentrates in a handful of bins rather than spreading over
256, and a rank in the middle of one of them costs one extra pass before the bin fits the budget.

**Nulls.** The histogram and the scatter both skip invalid rows, so the selection is over the non-null rows
only. When fewer than k rows are non-null the answer has to reach into the null rows, which only the sort
places — that case, and k above half the non-null rows (where the compaction would copy most of the column
anyway), fall back to `argsort`.

### Order statistics: selecting a value, not rows

A quantile wants the *value* at a rank, not the rows, so `Kernels/RadixSelectValue.swift` runs the same
narrowing with the winners counted but never written (`wantLt = 0`). Once the bin is under 4096 keys it is
read back and finished on the host, which ends the round trips immediately instead of after another six
digits. `kthElement(k, largest:)` is that search plus the inverse key map; `quantile(q)` asks for the one or
two adjacent ranks that bracket the interpolated position, and because they are adjacent one search usually
settles both. `approximateMedian()` is `quantile(0.5)`.

The reconstructed value is exact for every integer type. For floats it differs from indexing a sorted copy on
exactly the two values the key deliberately merges: -0.0 comes back as +0.0 and a NaN loses its payload. Both
are values Arrow's total order calls equal, and `quantile` returns a `Double`, so its answers are unchanged.

`tdigest` still sorts: its centroids are defined by the whole sorted sequence, not by one rank, so selection
cannot produce the same digest.

### Which path runs

| condition | path |
|---|---|
| k > 1024, or n >= 2^19 | radix select |
| k <= 1024 and n < 2^19 | per-threadgroup selection (`Kernels/TopK.swift`) — one dispatch, no mid-flight readback, so it wins where latency rather than bandwidth decides |
| fewer than k non-null rows, or k > (non-null)/2, or a type with no key map | full `argsort` |

The crossover at 2^19 rows is where radix select's two passes plus one host readback start beating a single
selection dispatch. Below it the per-threadgroup kernel is one command buffer; above it, it is bandwidth that
matters and radix select reads the column at close to peak.

### Numbers (M4 Max, 50M rows, best of 5)

Called from Python on the same in-process data (`PYTHONPATH=python python Benchmarks/python_gpu_bench.py
50000000 5`), against pyarrow 25, Polars 1.44 (16 threads) and numpy 2.5. Wall ms.

| Operation (50M rows) | before | **after** | pyarrow | Polars | numpy |
|---|---:|---:|---:|---:|---:|
| `top_k(100)`, Int64 | 8.42 (threadgroup select) | **4.68** | 24.85 `select_k_unstable` | 63.81 | 192.87 `argpartition` |
| `top_k(10 000)`, Int64 | 139.6 (full argsort) | **4.22** | 37.04 | 63.23 | 192.92 |
| `top_k(100 000)`, Int64 | 139.6 (full argsort) | **4.04** | 193.14 | 62.58 | 194.36 |
| `quantile(0.5)`, Int64 | 139.4 (full sort) | **4.48** | 307.33 | 67.66 `median` | 313.34 `median` |
| `quantile(0.5)`, Float64 | 138.0 (full sort) | **2.69** | 387.05 | 132.00 `median` | 396.31 `median` |

That is 5.3x / 8.8x / 47.8x pyarrow's `select_k_unstable` at the three k, and 69x / 144x pyarrow's
`quantile`. The target was 3x. `top_k` is now flat in k, because k only changes how many of the compacted
candidates the final ordering has to sort, not how much of the column is read — while pyarrow's heap-based
`select_k_unstable` degrades from 25 ms to 193 ms over the same range.

The column is read twice, so the useful bandwidth figure is 2 x 400 MB over the wall time: 171 / 190 / 198
GB/s for the three top-k, against 423 GB/s for a bare `sum` over the same column and a device peak of ~546
GB/s. The gap is the mid-flight readback — the host reads 256 counts to pick the bin, so the two passes
cannot share a command buffer — plus the final ordering, which is a few hundred microseconds at k = 100 and
a couple of milliseconds at k = 100 000.

The CPU stays free throughout: 1.7-1.9 CPU-ms for the top-k calls and 0.9-1.4 for the quantiles, against
25-396 CPU-ms for every CPU library on the same question.

The radix sort's block size is also now adaptive below ~256k rows: a fixed 4096 elements per block left a
20k-element sort — the size top-k's final ordering lands on — running on five threadgroups. Inputs above
~256k rows are unaffected, so the 50M argsort is unchanged.

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
