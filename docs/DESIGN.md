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
   so every atomic stays 32-bit (Metal has no 64-bit atomics), and the key of an occupied slot is the
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

### Measured (M4 Max, 50M rows, 12-byte utf8 keys, best of 5)

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
**2.9x and 7.2x faster than pyarrow**, against 5.4x and 1.7x *slower* before, and they cost the CPU about
4 ms against pyarrow's 700-2000.

Integer, boolean, temporal and dictionary key columns still take the range path (~22 ms at 50M rows);
they never enter the hash table. `unique` and `value_counts` are not implemented for string columns, so
there was nothing there to route.

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
   Done for `utf8` / `binary` keys (above); the same table would replace the sort for float and
   wide-range integer keys, which still argsort.
8. **Float64 sum/arithmetic** on the GPU via double-float (two `float`) arithmetic, or opt-in Float32.
9. **Binary archives** (`MTLBinaryArchive`) so the first call does not pay ~100 ms of shader compilation.
10. **Metal 4** command encoding and residency sets for very large resident datasets.
11. **iOS/visionOS**: the library already targets iOS 17+; add a demo app and thermal-aware sizing.
12. **Chip matrix**: publish benchmarks for M1/M2/M3/M4 base, Pro, Max, Ultra and A17/A18.

## Non-goals
- Not a query planner or SQL engine. It is the compute layer that DuckDB, DataFusion, Polars plugins or
  an app can call.
- Not a tensor library; MLX is that. A zero-copy bridge to MLX is planned.
