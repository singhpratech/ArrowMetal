# Out-of-core streaming execution

Datasets larger than memory, answered on the Apple GPU. The rows flow from disk through the GPU one
record batch at a time; the CPU never touches a value column. Only the *answer* grows with the input,
so a 100 GB question runs in a few GB of RAM on a 64 GB machine.

```swift
let rows = try StreamQuery(ipc: "/data/events")
    .filter(col("amount") > 100)
    .groupBy(["region"], [StreamAggregate(.sum, "amount")])
```

```python
import arrowmetal as am
am.scan_ipc("/data/events").filter(am.col("amount") > 100) \
  .group_by("region").agg([("sum", "amount", "total")])
```

Source files: `Sources/ArrowMetal/Stream/`, C ABI in `Sources/ArrowMetalC/ArrowMetalC_Stream.swift`
and the block at the end of `include/arrowmetal.h`, Python in `python/arrowmetal/stream.py`.

---

## 1. The pipeline

Three stages run at the same time on three threads, connected by bounded queues:

```
   ┌──────────────┐   batch i+1   ┌─────────────┐   result i   ┌──────────────┐
   │  stage 1     │ ────────────► │  stage 2    │ ───────────► │  stage 3     │
   │  read        │   queue(3)    │  GPU        │   queue(2)   │  merge       │
   │              │               │             │              │              │
   │ mmap + import│               │ one batch   │              │ fold a small │
   │ next batch   │               │ of kernels  │              │ result into  │
   │ (own thread) │               │ (this one)  │              │ running state│
   └──────────────┘               └─────────────┘              └──────────────┘
        SSD                        Metal                        host memory
```

* **Stage 1** is `PrefetchingSource`. It runs the wrapped source on its own thread and keeps up to
  `depth` batches ready (3 by default). Backpressure is two-sided: the reader blocks when `depth`
  batches are queued **or** when the queued batches exceed `budgetBytes` (2 GB by default), whichever
  comes first, so a stream of very wide batches cannot pin more memory than the budget.
* **Stage 2** is the caller's thread inside `StreamingExecutor.run`. It pulls one batch, records the
  operator's kernels into a single Metal command buffer via `MetalContext.batch { }`, and hands the
  small result on. It never waits for the merge unless the merge queue is full.
* **Stage 3** is the merge thread. It folds each batch's result into the running state, one at a
  time, in batch order.

Every stage records its own busy time. `StreamStats.overlap` is
`(read + gpu + merge) / wall`: **1.0** means the stages ran one after another, and it approaches
**3.0** when all three are saturated at once. `readStallNanos` is how long stage 2 waited for a batch
the reader had not finished; `mergeStallNanos` is how long it waited for the merge to drain.

### Why the merge stage is cheap

The GPU stage always produces a *small* result — a handful of scalars, one row per group, k rows for
a top-k, a 16 KB sketch. That is the whole reason streaming works: the merge's cost is a function of
the **answer's** size, not the dataset's. The one exception is documented below (the external sort's
k-way merge, which is O(rows) on the host by construction).

### Buffer budget

| What | Where | Size |
| --- | --- | --- |
| Batches in flight | `MetalContext.pool` (page-aligned shared memory) | `prefetchDepth + 1` batches, capped by `budgetBytes` (2 GB default) |
| Per-batch kernel scratch | `MetalContext.pool` | recycled by exact length, so a steady stream of equal-shaped batches allocates once |
| Running state | host or GPU, per operator | see the operator table |
| Sink buffer | one batch | the IPC sink writes each message straight through |

`MetalContext.pool` recycles page-aligned `MTLBuffer`s by exact byte length. Batches of the same
shape hit that cache on every read after the first, so an out-of-core scan does not pay `mmap` and
page-fault costs per batch. Its cap defaults to a quarter of the device's recommended working set,
at most 8 GB (`MetalContext(poolLimitBytes:)`).

---

## 2. Sources

`Sources/ArrowMetal/Stream/BatchSource.swift`.

| Source | Reads | Notes |
| --- | --- | --- |
| `IPCFileSource` | one Arrow IPC file | Memory mapped. Body buffers are borrowed without a copy where the file's layout allows; Arrow only guarantees 8-byte body alignment, so a buffer that does not start on a page boundary is copied into shared memory instead. Readahead with `fcntl(F_RDADVISE)` on a second descriptor plus `F_RDAHEAD`, which warms the unified buffer cache ahead of the read cursor. |
| `IPCDirectorySource` | a directory of them | Sorted by filename, opened lazily one at a time, so a directory larger than memory never has more than one file mapped. |
| `CStreamSource` | any `ArrowArrayStream` | A pyarrow `RecordBatchReader`, a `pyarrow.dataset` scanner (so **Parquet, CSV and partitioned datasets work today** through pyarrow's readers), a Polars `LazyFrame`, DuckDB — anything that speaks the Arrow C Stream ABI. Buffers are imported zero-copy when page aligned. |
| `ChunkedTableSource` | batches already in Metal memory | For tests and in-memory tables. |
| `PrefetchingSource` | wraps any of the above | Stage one of the pipeline. |

A GPU Parquet reader lives in `Sources/ArrowMetal/Parquet/` and is developed separately; until it is
the source of record, Parquet arrives through `scan_arrow(pyarrow.dataset(...))`.

## 3. Sinks

`Sources/ArrowMetal/Stream/StreamingSink.swift`.

* **`IPCStreamSink`** writes the Arrow IPC **stream** encapsulation incrementally: a schema message,
  one message per record batch, then the 8-byte end-of-stream marker. That encapsulation is a pure
  concatenation, which is what makes it appendable — the bytes for each message come from
  `ArrowIPCWriter.encode`, with the schema prologue (measured once by encoding the schema with no
  batches) and the end-of-stream marker trimmed off each call. `pyarrow.ipc.open_stream`,
  `polars.scan_ipc_stream` and this package's own `ArrowIPCReader` all read the result.
  Dictionary-encoded columns are decoded before writing, because an incremental sink cannot promise
  one dictionary per column across batches.
* **`CallbackSink`** hands each batch to a closure.
* **`CollectingSink`** keeps the batches (only for results known to be small).
* **`ArrowStreamExporter`** exports a query as an `ArrowArrayStream`. `get_next` pulls exactly one
  source batch through the whole pipeline, so pyarrow or Polars can consume an out-of-core ArrowMetal
  query lazily and never materialise more than one batch — `Stream.to_reader()` in Python.

---

## 4. Operators

| Operator | Exact or approximate | GPU work | Host work | Running state |
| --- | --- | --- | --- | --- |
| `filter` / `project` → sink | **exact** | the predicate and every projection compile into **one** fused kernel per batch (`docs/EXPR.md`) | writing the sink | none |
| `sum`, `count`, `min`, `max` | **exact** | one fused aggregate kernel per batch | one scalar combine per batch | a few scalars |
| `mean` | **exact** | decomposed to `sum` + `count` on the GPU | combine, divide once at the end | 2 scalars |
| `variance`, `stddev` | **exact** | decomposed to `sum(x)`, `sum(x·x)` in float64 and `count`, all in one fused kernel | combine | 3 scalars |
| `count_distinct_approx` | **approximate** — relative standard error `1.04 / sqrt(2^p)`: 0.81 % at the default p = 14, 0.41 % at p = 16 | a new HyperLogLog kernel: one `atomic_fetch_max` per row into a `2^p`-register table | element-wise max of two register arrays | `2^p` bytes (16 KB at p = 14), whatever the dataset's size |
| `group_by` (dense integer key) | **exact** | `GroupBy` per batch, then element-wise `add` / `min` / `max` folds the batch result into the **GPU-resident** global table | none | `K` accumulators per aggregate on the GPU; spills to the host table above `gpuStateBudgetBytes` (512 MB default) |
| `group_by` (arbitrary keys, any type, any number) | **exact** | `GroupByKeys` turns the key columns into dense ids on the GPU, then the aggregates | fold one row per group into a host dictionary keyed by the key value | one entry per **distinct group**, not per row |
| `top_k` | **exact** | per-batch GPU top-k, then a GPU top-k over the 2k candidates | `concat` of two k-row batches | ≤ 2k rows |
| `quantile` | **approximate** — see §6 | one radix argsort plus one gather per batch; only `compression` values reach the host | merge two weighted centroid lists and re-compress | ≤ `compression` centroids (1000 default) |
| `sort` (external) | **exact** | one radix argsort per batch; each output chunk is assembled with `take` | **the k-way merge is on the CPU** (see §5) | one batch per run + one output batch |
| `join` (broadcast) | **exact** | the existing GPU hash join per probe batch | writing the sink | the build side, in memory |
| `join` (grace hash) | **exact** | partition on the GPU, then the GPU hash join per partition | writing the partition files | one partition pair at a time |

Null handling follows Arrow throughout: a null key forms its own group, `count(expr)` counts non-null
values, `sum`/`min`/`max` skip nulls and return null for an all-null input, and a filter drops rows
where the predicate is null.

### Filter and project: two paths

The fused Expr compiler evaluates every expression into a fixed-width register, so it can *read* a
`utf8` column but cannot *write* one. `streamFilterProject` picks between:

* **Fused** — every output is numeric or boolean: predicate and projections compile into one kernel,
  the batch is read once, nothing intermediate is materialised.
* **Two-step** — some output is a plain `utf8` / `binary` / temporal / decimal column. The predicate
  still compiles into one fused kernel, producing a boolean mask; that mask drives the GPU `filter`,
  which carries *any* column type, and only the computed outputs go back through the compiler.

### Streaming group-by: two global tables

* **Dense integer key** (`denseKeyCount` given, one key column already inside `[0, K)`): the global
  table lives on the **GPU** as one accumulator array per aggregate, `K` entries wide. Each batch runs
  `GroupBy` over the batch's keys and the per-batch result is folded into the global arrays with
  element-wise `add` / `min` / `max` kernels — **no host round trip at all**. When
  `K * aggregates * 16` bytes exceeds `gpuStateBudgetBytes` the state **spills to the host table**,
  and partials recorded before the spill are folded into it, so a 10-million-key group-by still runs
  inside a fixed GPU budget.
* **Arbitrary keys**: `GroupByKeys` (which turns any key columns into dense ids on the GPU) plus the
  aggregates give one row per group — typically thousands of rows for a batch of millions. The merge
  folds those rows into a host dictionary keyed by the key value itself. Exact, order independent,
  and its size is the number of distinct groups.

Both paths return the groups in ascending key order, so the result is deterministic across batch
layouts and runs. (This differs from pyarrow, which returns first-seen order; sort both sides before
comparing.)

---

## 5. External sort

Each batch is sorted on the GPU (the existing radix argsort) and written to its own Arrow IPC stream
file — one run per batch. `finish()` merges the runs.

**The merge is on the CPU, deliberately.** A k-way merge is a branch-per-row loop with no parallelism
to give a GPU: at 20 runs the whole merge is one comparison of 20 heap entries per output row. What
*is* on the GPU is the gather. For each output chunk the merge decides the order, then:

1. the rows needed from each run are gathered with **one `take` per run**, giving rows grouped by run;
2. **one permutation `take`** puts them back in merged order.

So no column data is ever rebuilt value by value on the host, and memory during the merge is one
batch per run plus one output batch. `ORDER BY ... LIMIT n` stops the merge at n rows.

Sort keys use the GPU radix sort for numeric, boolean and temporal columns. `utf8` and `binary`
columns have no order-preserving GPU key yet, so they fall back to a host sort of the string values
(correct, but the one place a sort touches every row on the CPU).

## 6. Streaming quantiles

An exact quantile needs the whole column ordered, which an out-of-core query cannot afford. Instead a
mergeable digest keeps a bounded set of weighted centroids: each says "about *w* values sit near *v*".
Two digests merge by concatenating their centroids, ordering them and compressing back to the budget,
which is associative and order independent.

Per batch the GPU does the work: sort the non-null values with the radix argsort, then gather
`compression` sample points with one `take`. **The sample positions are not uniform in rank** — they
follow the inverse of the t-digest k1 scale function `k(q) = (C / 2π) · asin(2q − 1)`, which packs
samples into the tails, where a uniform sample is worst. The merge re-compresses under the same scale
function, so a centroid near q = 0.99 keeps far less weight than one near q = 0.5.

A centroid covering rank interval `[r0, r1)` answers any quantile inside it with rank error at most
`(r1 − r0) / 2`: about `n / C` at the median and `n·π / (2C²)` near the extremes, so p99 is roughly C
times more accurate than the median. The tests assert a rank error under 1 % at C = 1000 across
p1 … p99.

## 7. HyperLogLog on the GPU

`Sources/ArrowMetal/Stream/HyperLogLog.swift`. Every row is hashed to 64 bits (splitmix64 for
fixed-width values, FNV-1a then splitmix64 for bytes); the top `p` bits pick a register and the
position of the first 1 in the rest updates that register with a max. One thread per row,
`atomic_fetch_max_explicit` on a `2^p`-entry `uint` array in device memory (64 KB at p = 14).
Contention is low: the hash spreads rows across 16,384 registers and a max only ever moves a register
upward, so most atomics are no-ops the hardware resolves without a retry.

Two sketches merge by element-wise max, which makes the result **independent of batch boundaries and
batch order** — the tests assert bit-identical registers for a whole-array sketch and the same data
split into 17 chunks. The estimator is the standard one with linear counting below `2.5·m`.

`-0.0` and `0.0` hash the same; nulls are skipped, as Arrow's `count_distinct` does.

## 8. Streaming joins

* **Broadcast** — the build side is read once into one Metal-resident batch and every probe batch runs
  the existing GPU hash join against it, straight to the sink. Memory is the build side plus one
  batch. The right plan whenever one side fits, however large the other one is.
* **Grace hash** — both sides are streamed once and partitioned by
  `((key · 0x9E3779B97F4A7C15) >>> (63 − log₂P)) & (P − 1)` into `P` Arrow IPC files per side. The
  hash takes the *high* bits of the product so an id column that is a multiple of `P` does not land
  entirely in one partition. Equal keys always land in the same partition, so the join of the whole is
  the union of the per-partition joins, and each partition is small enough for the in-memory GPU hash
  join. Two passes over each input plus one write and one read of each; memory is one partition pair.
  Row order differs from an in-memory join, as it does for any hash join; the test compares the
  result as a set of pairs.

Join keys are int32 or int64 on both sides, which is what the GPU hash join takes. Null keys never
match.

---

## 9. Numbers

Measured on an M4 Max (16 cores, 64 GB unified memory, internal SSD), Swift release build, over a
**30.0 GB** Arrow IPC directory: 588,235,294 rows, 37 files, 589 record batches of 1M rows, 8 columns
(`id` int64, `region` int32 with 1000 values, `bigkey` int64 with 10M values, `label` utf8 with 50
values, `amount` float64, `qty` int32, `ts` int64, `flag` int8). Generated by
`Benchmarks/streaming_bench.py`; the machine has 64 GB of RAM, so the dataset is about half of it and
the page cache is warm after the first pass.

### The whole dataset, one pass each

| Query | Wall | GPU stage | Read stage | Merge stage | Overlap | GB/s from disk | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sum(amount) where region < 100` | 5.40 s | 0.56 s | 5.36 s | 0.00 s | 1.10x | 5.6 | 0.97 GB |
| `group by region → sum, count` (1000 groups, host table) | 4.87 s | 4.24 s | 4.81 s | 0.02 s | 1.87x | 6.2 | 1.10 GB |
| `group by region → sum` (1000 groups, **GPU state**) | 4.81 s | 3.17 s | 4.79 s | 0.39 s | 1.74x | 6.3 | 1.10 GB |
| `group by bigkey → sum` (**10,000,000 groups**) | 41.8 s | 17.9 s | 4.2 s | 40.8 s | 1.51x | 0.7 | 3.79 GB |
| `top 100 by amount` | 3.56 s | 2.28 s | 3.21 s | 2.68 s | 2.30x | 8.4 | 4.06 GB |
| `count_distinct_approx(id)` | 2.67 s | 0.49 s | 2.66 s | 0.01 s | 1.18x | 11.2 | 4.06 GB |
| `quantile(amount, [p50, p99])` | 3.42 s | 3.34 s | 2.90 s | 0.02 s | 1.83x | 8.8 | 4.06 GB |

Peak RSS is cumulative for the process (the rows are one Python process running every query in turn),
so the interesting figure is the **first** one: 0.97 GB for a full pass over 30 GB. Nothing but the
answer grows: the 10-million-group query is the only one whose state is large, and 3.8 GB is that
global table, not the data.

Accuracy of the two approximate operators on this data:

| | Exact | ArrowMetal | Error |
| --- | --- | --- | --- |
| `count_distinct(id)` | 588,235,294 | 588,008,929 | **0.038 %** (bound at p = 14 is 0.81 % standard error) |
| `quantile(amount, 0.5)` | 500.000 | 499.989 | rank error < 0.01 % |
| `quantile(amount, 0.99)` | 990.000 | 989.988 | rank error < 0.01 % |

### The overlap, measured

`prefetch=0` makes the read stage run inline on the GPU thread — the same work, serialised. Same
process, same page-cache state, best of two alternating runs each:

| Query | prefetch=0 (serial) | prefetch=3 (pipelined) | Speed-up | Overlap serial → pipelined |
| --- | --- | --- | --- | --- |
| `filter + sum` | 3.57 s | **2.83 s** | 1.26x | 1.00x → 1.14x |
| `group by region` | 8.00 s | **4.93 s** | 1.62x | 1.00x → 1.82x |
| `top 100` | 7.17 s | **3.52 s** | 2.04x | 1.22x → 2.54x |

`filter + sum` gains least because its GPU stage is only 0.5 s against a 3 s read: there is barely
anything to hide. `top 100` gains most because all three stages are busy — 2.5 s of GPU and 3.2 s of
merge run inside a 3.5 s wall. That is the pipeline doing exactly what it is for.

### Where the 10-million-group case goes

The arbitrary-key merge folds one row per group per batch into the host table: 589 batches x ~800k
groups is roughly 470 million probes. Three changes took it from 260 s to 42 s, and they are worth
naming because they are the whole reason the merge stage is viable:

1. columns arrive as `StreamColumn` (`[Int64]` / `[Double]` storage) instead of `[StreamValue]`, so
   the loop is not doing ARC per element;
2. the accumulators are one flat `groups * aggregates` array mutated in place, not one small array
   per group per batch;
3. a single integer key uses an `[Int64: Int]` table, not a boxed key.

The dense-key path is the alternative when the key space is known: `dense_key_count = 10_000_000`
keeps the state on the GPU and never touches the host, but pays an element-wise merge over 10M
accumulators per batch — 92 s here, so the host table wins at this key count. The GPU state is the
right plan at a few thousand keys and the crossover is around a million.

## 10. Limits

* **Sort keys** on `utf8` / `binary` columns fall back to a host sort per batch. Numeric, boolean and
  temporal keys use the GPU radix sort.
* **`count_distinct_approx` is a whole-dataset aggregate**, not a per-group one; a per-group HLL would
  need one sketch per group and is not implemented.
* **`variance` / `stddev` are not available in the dense-key GPU group-by path** (they need a third
  accumulator array); the arbitrary-key path has them.
* **Grace join keys must be integers.** Strings would need the GPU string hash table another agent
  owns.
* **The IPC sink writes the stream encapsulation**, not the random-access file format: the file
  footer's block index cannot be built incrementally without reaching into `IPCWriter`. Every reader
  that matters (pyarrow, Polars, this package) reads it.
* **Dictionary-encoded columns are decoded** by the sink and by the merge helpers. One dictionary per
  column for a whole stream is the only form the writer emits, which an incremental sink cannot
  promise.
* **A `Stream` is single use**: a terminal consumes the source. Open a new scan for a second question.
* **`am_stream_query` supports filter + project and filter + aggregate**; a grouped query goes through
  `am_stream_group_by`, because the Expr compiler's `group_by` takes a dense integer key and the
  streaming group-by takes arbitrary key columns.
* Arrays above 2³² rows are still refused (`Dispatch.checkLength`), which bounds one *batch*, not a
  dataset.
