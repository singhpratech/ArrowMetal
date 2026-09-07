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
| `ParallelIPCSource` | a directory, several files at once | `readers` threads, each with its own `IPCFileSource` and its own files, feeding one bounded queue. **Batch order is not preserved** — see below. |
| `ChunkedTableSource` | batches already in Metal memory | For tests and in-memory tables. |
| `PrefetchingSource` | wraps any of the above | Stage one of the pipeline, one thread. |

### One reader thread is not enough

`PrefetchingSource` hides the read behind the GPU, but it is still *one* thread doing the mapping and
the import. One core copies at roughly 10 GB/s; Polars and DuckDB scan the same directory at 44 to
72 GB/s because they read it with every core. `ParallelIPCSource` closes that gap: it hands each of
`readers` threads its own files, so the read stage scales with cores.

The cost is batch **order**. Batches arrive interleaved across files, in whatever order the threads
finish. Every operator here is order independent — aggregates, the HLL sketch, group-by, top-k, the
external sort's run generation, both joins — so the only thing that needs `readers: 1` is a
`filter -> sink_ipc` (or `to_reader`) that must preserve the source's row order.

```python
am.scan_ipc("/data/events", readers=8).group_by("region").agg([("sum", "amount", "t")])
```

```swift
try StreamQuery(ipc: "/data/events", readers: 8)
```

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
| `group_by` (arbitrary keys, any type, any number) | **exact** | `GroupByKeys` turns the key columns into dense ids on the GPU, then the aggregates | fold one row per group into a host table sharded by key hash across `mergeShards` threads | one entry per **distinct group**, not per row |
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

Runs are merged with a **bounded fan-in**. 570 batches means 570 runs, and opening them all at once
would want 570 file descriptors and 570 resident batches; instead the merge runs in passes of at most
`mergeFanIn` (32 by default), writing intermediate runs and deleting their inputs as it goes, so both
stay constant however many runs there are. A `limit` is applied to *every* pass — the global first n
rows are always inside the union of each group's first n — which is what makes `ORDER BY ... LIMIT`
cheap over hundreds of runs.

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
**30.21 GB** Arrow IPC directory: 570,000,000 rows, 30 files, 570 record batches of 1M rows, 8
columns (`id` int64 unique, `region` int32 with 1000 values, `bigkey` int64 with 10M values, `label`
utf8 with 50 values, `amount` float64, `qty` int32, `ts` int64, `flag` int8). Generated and run by
`Benchmarks/streaming_bench.py`, which runs every (workload, engine) cell in its own subprocess so
peak RSS is that engine's alone and an out-of-memory kill costs one cell rather than the run.

The machine has 64 GB of RAM and the dataset is 30 GB, so after the first pass it is in the page
cache: the GB/s figures are memory bandwidth as much as SSD bandwidth, and they are the same for
every engine.

### Against Polars and DuckDB

Wall-clock milliseconds, then peak RSS. ArrowMetal runs with `readers=8`; Polars uses
`scan_ipc(...).collect(engine="streaming")`, DuckDB queries the directory through a
`pyarrow.dataset`. Every cell finished within memory.

| Workload | ArrowMetal | Polars | DuckDB | ArrowMetal RSS | Polars RSS | DuckDB RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sum(amount) where region < 100` | **527** | 515 | 418 | 8.9 GB | 2.0 GB | 1.1 GB |
| group by `region` -> sum, count (1k groups) | 1,765 | **554** | 625 | 9.5 GB | 3.1 GB | 1.3 GB |
| group by `bigkey` -> sum (10M groups) | 19,946 | **4,031** | 5,232 | 12.4 GB | 13.1 GB | 25.7 GB |
| top 100 by `amount` | 4,060 | 587 | **544** | 8.4 GB | 2.5 GB | 0.9 GB |
| `count_distinct(id)` | 532 | 3,825 | **451** | 8.8 GB | 17.0 GB | 0.9 GB |
| order by `amount` desc limit 1000 | 11,121 | 642 | **567** | 9.3 GB | 2.7 GB | 0.9 GB |
| broadcast join + sum | 1,807 | 764 | **423** | 9.2 GB | 2.3 GB | 1.1 GB |

pyarrow.dataset (measured in a separate single-reader run, so not comparable cell by cell) finished
`groupby_10m` in 557 s and **timed out at 900 s on `count_distinct`**.

**Read the wins and the losses honestly.**

* **`filter_sum` ties Polars** at 57 GB/s — a whole-dataset filtered aggregate over 570 million rows
  in half a second, with the CPU doing nothing but reading bytes.
* **`count_distinct_approx` beats Polars by 7x** (532 ms against 3,825 ms) *and* uses half a
  gigabyte where Polars uses 17. DuckDB's sketch is faster still, but its answer is **12.4 % off**
  against ArrowMetal's **0.007 %** — DuckDB's `approx_count_distinct` is tuned for a much smaller
  sketch. Against an exact count this is the operator where streaming on a GPU clearly pays.
* **Top-k, sort+limit and the joins lose**, by 3x to 17x. The reason is visible in the pipeline table
  below: these are the workloads where the *GPU stage plus the merge stage* is the bottleneck, not
  the read, and a Metal command buffer per batch has a fixed cost that 570 batches multiply out.
  Polars and DuckDB are also simply very good at these.
* **`groupby_10m` is 5x behind** but uses half of DuckDB's memory. See §4 for what the merge does and
  where the remaining time goes.

### The ArrowMetal pipeline on each workload

`overlap` is (read + gpu + merge) / wall: 1.0 for a serial pipeline, and above that by however much
the stages actually ran at the same time. With `readers = 8` the read column is the sum across the
eight reader threads, so it can exceed wall on its own.

| Workload | Overlap | Read (s) | GPU (s) | Merge (s) | Wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `filter_sum` | 7.72x | 3.52 | 0.29 | 0.00 | 0.53 |
| `groupby_1k` | 3.82x | 3.48 | 1.71 | 1.43 | 1.77 |
| `groupby_10m` | 1.47x | 3.79 | 16.50 | 8.88 | 19.95 |
| `topk` | 4.14x | 9.42 | 3.24 | 3.91 | 4.06 |
| `count_distinct` | 7.70x | 3.55 | 0.29 | 0.01 | 0.53 |
| `sort_limit` | 1.20x | 4.60 | 3.88 | 4.84 | 11.12 |
| `broadcast_join` | 3.45x | 3.85 | 1.24 | 0.00 | 1.81 |

### Accuracy of the two approximate operators

| | Exact | ArrowMetal | Error |
| --- | --- | --- | --- |
| `count_distinct(id)` | 570,000,000 | 569,958,240 | **0.0073 %** (the p = 14 bound is 0.81 % standard error) |
| `quantile(amount, 0.5)` | 500.000 | 499.989 | rank error < 0.01 % |
| `quantile(amount, 0.99)` | 990.000 | 989.988 | rank error < 0.01 % |

DuckDB's `approx_count_distinct` on the same column: 640,524,923, an error of **12.37 %**.

### The overlap, measured

`prefetch=0` makes the read stage run inline on the GPU thread — the same work, serialised. Same
process, same page-cache state, one reader thread on both sides, best of two alternating runs each,
on a 30 GB dataset:

| Query | prefetch=0 (serial) | prefetch=3 (pipelined) | Speed-up | Overlap serial -> pipelined |
| --- | ---: | ---: | ---: | --- |
| `filter + sum` | 3.57 s | **2.83 s** | 1.26x | 1.00x -> 1.14x |
| `group by region` | 8.00 s | **4.93 s** | 1.62x | 1.00x -> 1.82x |
| `top 100` | 7.17 s | **3.52 s** | 2.04x | 1.22x -> 2.54x |

`filter + sum` gains least because its GPU stage is only 0.5 s against a 3 s read: there is barely
anything to hide. `top 100` gains most because all three stages are busy. Going from one reader
thread to eight is a second, larger step on top of this: `filter_sum` went from 2.83 s to 0.53 s and
`count_distinct` from 2.36 s to 0.53 s once the read stopped being the bottleneck.

### What the memory figures mean

ArrowMetal's peak RSS at `readers = 8` is around 9 GB against Polars' 2 GB, and almost all of the
difference is **mapped file pages**: eight readers hold eight ~1 GB part files mapped at once, and
`ru_maxrss` counts those. They are page cache, not anonymous memory — the kernel reclaims them under
pressure — but the number is real and it is the price of the parallel read. With `readers = 1` the
same queries run at 1.2 GB of RSS (and two to five times slower). The genuinely private state is what
§4 lists: a few scalars, a 16 KB sketch, k rows, or the group table.

## 10. Limits

* **`readers > 1` does not preserve batch order**, so a row-order-preserving `sink_ipc` or
  `to_reader` needs `readers: 1` (the default).
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
* **Top-k, sort+limit and the joins are 3x to 17x slower than Polars and DuckDB** on a warm 30 GB
  dataset (§9). Those are the workloads where the GPU stage and the merge dominate rather than the
  read, and one Metal command buffer per batch has a fixed cost that 570 batches multiply out;
  a larger `batch_rows` amortises it, at the cost of memory per batch.
* **`readers > 1` raises peak RSS by roughly `readers` x the part-file size**, because that many
  files are mapped at once. Those pages are reclaimable page cache, not anonymous memory, but they
  do show up in `ru_maxrss`.
* **`am_stream_query` supports filter + project and filter + aggregate**; a grouped query goes through
  `am_stream_group_by`, because the Expr compiler's `group_by` takes a dense integer key and the
  streaming group-by takes arbitrary key columns.
* Arrays above 2³² rows are still refused (`Dispatch.checkLength`), which bounds one *batch*, not a
  dataset.
