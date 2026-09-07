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
  serve CPU consumers (C Data Interface) and GPU kernels (C Device Interface): copy-free out always, and
  copy-free in when the producer's buffers are page aligned — one copy otherwise, which the importer
  reports (`ImportResult.zeroCopy`).
- **Kernels** never assume a length multiple of anything: every buffer is padded to a page so trailing
  32-bit bitmap words are readable; kernels bounds-check the last word/element.
- **Nulls** ride along as bitmaps. Element-wise ops share the input's validity buffer (zero-copy); binary ops
  AND the two bitmaps on the GPU; compaction repacks.

## Where the time goes today (M4 Max, 50M rows)

Source: `swift run -c release arrowmetal-bench`, recorded 2026-09-06 on the Apple M4 Max (16 cores,
64 GB unified memory, Darwin 25.6.0) that every table on this page was measured on; the dated rounds
are in [BENCHMARKS.md](BENCHMARKS.md).

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

## Group-by over string keys: a hash table instead of a sort

`GroupByKeys` turns a key column into dense ids `0 ..< K` before any aggregate runs. For `utf8` and
`binary` columns that used to mean the GPU string dictionary: hash every row to 64 bits, **argsort the
hashes**, mark run boundaries by comparing the bytes of adjacent sorted strings. Correct, but the argsort
is eight radix passes over 50 million 64-bit keys — ~185 ms — and it costs the same whether the column
holds a thousand distinct keys or ten million. Real group-by keys are low cardinality, so the sort was
paying row-count prices for distinct-count work.

`Kernels/StringHashTable.swift` replaces it with an open-addressing table sized to the cardinality:

1. **Hash** every non-null row to 64 bits in one pass over the bytes (`sht_hash64`).
2. **Estimate** the distinct count by building a throwaway table over a **1/64 slice of the hash space**
   — a row takes part only when `(h >> 40) & 63 == 0`. Each distinct string makes that decision once, so
   the occupied-slot count times 64 estimates the cardinality no matter how skewed the row frequencies
   are (a *row* sample would not: it sees the frequent keys and misses the rest). The estimate only sizes
   the real table, and being wrong costs a retry, never a wrong answer.
3. **Build** the table with three slots per estimated distinct value: linear probing, `slots[s] = row + 1`
   so every atomic stays 32-bit (Metal has no 64-bit atomic add; it has 64-bit atomic min and max only, see UPSTREAM.md), and the key of an occupied slot is the
   string of row `slots[s] - 1`. Equality is decided by **comparing the bytes** of the candidate against
   that representative, the same rule `is_in` uses, so a 64-bit hash collision costs one extra probe and
   can never merge two different strings. Each row also records the slot it landed in and folds itself
   into that slot's lowest row index with an `atomic_fetch_min` guarded by a plain load.
4. **Rank** the occupied slots with the same GPU scan `unique()` uses, compact one representative row per
   slot, and **relabel** into first-seen order with two argsorts over the *distinct* count.

Two design choices are worth stating plainly. Nothing is cached beside the slot — not the hash, not a
key prefix — because MSL only guarantees `memory_order_relaxed`: a companion array could be read before
its writer had published it, and a probe that then walked past its own bucket would give one string two
group ids. Reading `hashes[slots[s] - 1]` cannot go stale, because the hashes are written by an earlier
kernel and never change. And because byte comparison decides equality, this path needs **no re-hash retry
and no host fallback**, unlike the sort path it replaces. The only retry is a table that turned out too
small; the last attempt gets two slots per row and an unbounded probe budget, so it cannot fail.

The byte comparison was measured against deciding equality on the 64-bit hash alone: 17.5 vs 13.7 ms at
a thousand distinct keys, 22.9 vs 17.0 ms at 100k, and no difference at all at 10 million (where the
random slot access dominates). A 20-30% saving is not worth an answer that is only correct to the
birthday bound, so there is no unverified mode.

The result is identical to the sort path — same group ids, same group keys, same first-seen dictionary
order, same null semantics (a null key still forms its own group, id `K`) — which is what
`Tests/ArrowMetalTests/StringHashTableTests.swift` asserts, case by case, against both the host hash map
and the old path.

### Measured (M4 Max, 50M rows, 12-byte utf8 keys, best of 5), 2026-09-06

Group-by sum over a utf8 key column, key mapping included. Metal from Swift; Polars 1.44 (16 threads),
pyarrow 25 and the 16-core Swift hash group-by on the same buffers.

| distinct keys | Metal before (sort) | **Metal after (hash table)** | pyarrow | Polars | 16-core CPU |
|---|---:|---:|---:|---:|---:|
| 1,000 | 203.3 ms | **20.4** | 58.0 | 208.1 | 117.7 |
| 100,000 | 231.5 | **27.0** | 194.2 | 221.4 | 303.8 |
| 10,000,000 | 322.7 | **154.1** | 3387.6 | 751.7 | 2336.1 |

`dictionary_encode` on the same column, which is the mapping plus the first-seen relabel and the gather
of the distinct strings:

| distinct keys | sort path (old) | **hash table** | pyarrow `dictionary_encode` |
|---|---:|---:|---:|
| 1,000 | 275.8 ms | **20.9** | 422.3 |
| 100,000 | 329.0 | **26.1** | 603.0 |
| 10,000,000 | 438.9 | **268.1** | 4500.4 |

At a thousand keys the whole table is 4096 slots — 16 KB — so every probe is a cache hit and the pass is
memory-bound on the key bytes. At ten million the table is 32 M slots (128 MB) and the cost becomes the
random slot access, which is why the win narrows from 10x to 2x. The 1000- and 100k-key cases are now
**2.9x and 7.2x faster than pyarrow**, against 3.5x and 1.2x *slower* before, and they cost the CPU about
4 ms against pyarrow's 700-2000.

Integer, boolean, temporal and dictionary key columns still take the range path (~22 ms at 50M rows);
they never enter the hash table when they are group-by keys.

### The same table for the distinct-value functions

`unique`, `value_counts`, `count_distinct`, `mode` and `dictionary_encode` over a **primitive** column had
the same problem for the same reason: all five began with an argsort of every row, so all five cost the
same whatever the cardinality, and all five lost to CPU libraries that use a hash table (0.11x to 0.71x
of pyarrow/Polars/pandas at 10M rows before this change). `Kernels/HashTable.swift` is the string table
with the byte comparison removed — the key *is* the 64-bit value, so equality on it is exact — and the
estimate, the growth retry, the rank scan and the id pass are literally the same code.

Turning a column into that key is one kernel: integers widen (sign-extended, so the map is injective),
and floats are normalised first — every NaN to one bit pattern, `-0.0` to `0.0` — so bit equality means
Arrow value equality, which is what `unique`'s sort path achieves by normalising before the sort. The
order does not change either: the distinct values come back **ascending**, because the last stage
argsorts the `K` representative values. Sorting `K` values instead of `n` rows is the whole trick.
`count_distinct` skips even that — it is the occupied-slot count, so it never gathers or orders anything.

Because the table keeps the lowest row per slot, first-appearance order is available for one argsort of
`K` int32s (`HashGroups.firstSlotOrder`, exposed as `HashDistinct.firstRows`) rather than another pass
over the rows — which is what an `order:` option would want.

The threshold is `1 << 16` rows: below it the sort is a handful of small passes and the table's extra
round trips do not pay for themselves. `ARROWMETAL_NO_HASH=1` forces the sort path in a shipping binary,
which is how the before column below was measured.

M4 Max, 50M `int64` rows, Swift, best of 3, 2026-09-06 (the "before" column with `ARROWMETAL_NO_HASH=1`):

| function | distinct | before (sort) | **after (hash table)** |
|---|---|---:|---:|
| `unique` | 1,000 / 100k / 10M | 142.7 / 147.2 / 303.9 ms | **12.3 / 14.5 / 89.7** |
| `value_counts` | 1,000 / 100k / 10M | 141.2 / 235.5 / 443.3 | **14.2 / 17.7 / 110.6** |
| `count_distinct` | 1,000 / 100k / 10M | 142.3 / 282.3 / 265.0 | **9.6 / 10.1 / 42.5** |
| `mode` | 1,000 / 100k / 10M | 142.2 / 199.7 / 278.0 | **13.9 / 17.9 / 121.6** |
| `dictionary_encode` | 1,000 / 100k / 10M | 162.3 / 295.5 / 277.2 | **12.8 / 16.5 / 108.2** |

Against the CPU libraries on the same buffers, called from Python through
`Benchmarks/python_gpu_bench.py` (M4 Max, 50M rows, best of 3, ms, 2026-09-06):

| function | distinct | ArrowMetal | Polars | pyarrow | vs the best of them |
|---|---|---:|---:|---:|---:|
| `unique` | 1,000 | 12.2 | 120.9 | 97.2 | **7.9x** |
| | 100,000 | 14.1 | 141.8 | 89.8 | **6.4x** |
| | 10,000,000 | 91.1 | 191.3 | 715.1 | 2.1x |
| `value_counts` | 1,000 | 14.1 | 130.6 | 154.5 | **9.2x** |
| | 100,000 | 17.6 | 340.1 | 221.4 | **12.6x** |
| | 10,000,000 | 113.6 | 1781.0 | 1620.0 | **14.3x** |
| `count_distinct` | 1,000 | 9.7 | 121.0 | 91.4 | **9.4x** |
| | 100,000 | 9.8 | 137.1 | 90.4 | **9.2x** |
| | 10,000,000 | 41.8 | 183.8 | 798.3 | **4.4x** |
| `mode` | 1,000 | 27.7 | 85.1 | 23.2 | 0.8x |
| | 100,000 | 35.1 | 93.6 | 583.0 | 2.7x |
| | 10,000,000 | 247.1 | 494.3 | 1107.3 | 2.0x |

Two cases are worth being honest about. `unique` at ten million distinct is only 2.1x Polars because
ArrowMetal orders its output and Polars does not: 30 of those 91 ms are the argsort of the ten million
distinct values, which a hash-order `unique` would not pay. And `mode` is measured through a Python
binding that computes it **twice** (`_mode` in `python/arrowmetal/__init__.py` calls the reduction once
for the value and again for the count) — the Swift figures are 13.9 / 17.9 / 121.6 ms, i.e. 1.7x / 32.6x /
9.1x pyarrow. At 1,000 distinct pyarrow's `mode` is genuinely fast because the values span a narrow range
and it counts into a direct-indexed table rather than a hash map; the same range trick already exists here
in `GroupByKeys.rangeIds` and is the obvious next step for these five functions.
## Group-by

Dense group ids `0 ..< K` come out of `GroupByKeys`; everything below aggregates over them. There are
three shapes, and the point of the current design is that **none of them sorts the key column**.

**Atomic accumulation** (`Kernels/GroupBy.swift`, `GroupBySource.swift`) is one linear pass. For
`K <= 1024` each threadgroup keeps a private table in threadgroup memory and merges it into the device
table once; above that the updates go straight to device memory. 64-bit sums are a pair of 32-bit atomic
adds with an explicit carry, because Metal has no 64-bit atomic add (only min and max, UPSTREAM.md).

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

### Before and after (Apple M4 Max, best of five, 10M rows, one `group_by` per call), 2026-09-06

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
| count_distinct | any | GPU hash set over `(group, value)` ([LOSSES.md](LOSSES.md)) | | |

At 50M rows, against the fastest idiom of Polars / pyarrow / pandas
(`Benchmarks/results/full_matrix_2026-09-07-parallel.csv`): min 3.32x / 4.93x / 4.57x at 1k / 100k / 10M
groups, where it was 0.25x / 0.56x / 0.21x; variance 0.96x / 2.52x / 4.06x, where it was 0.05x at 100k;
`first` 7.95x / 5.81x / 5.12x; `list` 10.85x / 9.72x / 5.70x. Two caveats on those ratios. At a
**thousand** groups every numeric grouped aggregate lands near 3.3x to 4.1x, `sum` and `count` included,
because a fresh `group_by([...])` spends about 7 ms of the 10 ms rebuilding the dense key mapping — reuse
the object and the aggregate itself is 3 ms. And `count_distinct` by key was the remaining shortfall and
is now 3.4x to 7.2x the fastest CPU idiom ([LOSSES.md](LOSSES.md)) — the sort of the *key* column is gone
from every aggregate.

## Latency (small inputs)
Measured floor on M4 Max: an empty kernel with encode + commit + wait costs about 60-70 µs measured
([RESIDENT.md](RESIDENT.md)), 110-230 µs as an all-in per-call floor in the matrix's latency family; ten
kernels in one command buffer cost ~100 µs in total. So the fixed cost is the round trip, not the kernel,
and the only lever is fewer round trips.

Unbatched, every public call is one command buffer. `filter` already fuses its three
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
~390 GB/s these single-pass rows reach at 50M rows and `divide` within 15% of it (331 GB/s), so the
software arithmetic is all but invisible in a bandwidth-bound query. `cast` is the one float64 path that
still runs on the host, and `modulo` is the one binary operator with no float64 kernel — both say so in
[COVERAGE.md](COVERAGE.md).

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
numbers, measured 2026-09-06. The **measured** column is what the run reports; the test asserts a
2-ulp bound on everything but `sqrt`, which it holds to bit equality, so a regression of one ulp shows
up as a changed number here before it fails the suite. The trigonometric family is a separate kernel and
a separate budget: measured 2-5 ulp against a 6-ulp assertion (`TrigTests`, [COVERAGE.md](COVERAGE.md)).

| function | domain sampled                                        | max ulp |
|----------|-------------------------------------------------------|---------|
| `sqrt`   | 10⁶ random positive bit patterns drawn uniformly over the exponent range as well as the significand — subnormals, 1e-320 and 1e308 included — plus the perfect squares, their neighbours, every binade and the subnormal range in `testSqrtIsCorrectlyRoundedOnAdversarialInputs` | **0** (bit-identical) |
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

### Numbers (M4 Max, 50M rows, best of 5), 2026-09-06

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

This is the throughput list this page keeps; the project roadmap is [ROADMAP.md](ROADMAP.md). Four of
the items it opened with have since landed and are struck through rather than deleted, so a reader can
see what the design note predicted and where it went.

1. **Pipelined execution** (above). Biggest win for query-shaped work and for Python callers. Still open.
2. ~~**Fused expressions**~~ — done: `Sources/ArrowMetal/Expr` compiles a whole expression DAG into one
   runtime-generated MSL kernel, and filter + aggregate, project and dense-key group-by fuse into one
   dispatch ([EXPR.md](EXPR.md)).
3. **Reductions with vector loads** (`long4`) and tuned threadgroup counts per chip family.
4. **Filter in two passes** instead of four: compute selection + per-block counts in one kernel, then
   scatter with in-kernel decoupled look-back scan.
5. ~~**Strings**~~ — done for `utf8` / `binary`: equality, prefix, length, hash and GPU dictionary
   encoding, so group-by over strings maps to the dense-key path (above). `utf8_view` / `binary_view`
   are the part still open, and are on [ROADMAP.md](ROADMAP.md#types-and-interop).
6. ~~**Sort / top-k**~~ — done: the LSD radix sort on 32/64-bit keys with payload, `MetalRecordBatch
   .sorted(by:)`, and the radix select above.
7. **Hash group-by** for arbitrary keys (open addressing in device memory), feeding the same aggregators.
   Done for `utf8` / `binary` keys (above); the same table would replace the sort for float and
   wide-range integer keys, which still argsort.
8. ~~**Float64 sum/arithmetic** on the GPU~~ — done, but not the way this line guessed: software
   IEEE-754 binary64 (`Kernels/DoubleMath.swift`) rather than double-float or opt-in Float32, because
   an approximate float64 would have been a support burden forever ([DECISIONS.md](DECISIONS.md)).
9. **Binary archives** (`MTLBinaryArchive`) so the first call does not pay ~100 ms of shader compilation.
10. **Metal 4** command encoding and residency sets for very large resident datasets.
11. **iOS/visionOS**: the library already targets iOS 17+ (`Package.swift`); add a demo app and
    thermal-aware sizing. Nothing has been measured on an iPhone or iPad.
12. **Chip matrix**: publish benchmarks for M1/M2/M3/M4 base, Pro, Max, Ultra and A17/A18. Every number
    in this repository is from one M4 Max.

## Non-goals
- Not a query planner or SQL engine. It is the compute layer that DuckDB, DataFusion, Polars plugins or
  an app can call.
- Not a tensor library; MLX is that. A zero-copy bridge to MLX is planned.
