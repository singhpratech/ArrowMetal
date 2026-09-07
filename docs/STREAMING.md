# Out-of-core streaming execution

Datasets larger than memory, answered on the Apple GPU. The rows flow from disk through the GPU one
record batch at a time; the CPU touches value columns only where §5 and §10 say it does: a `utf8` sort
key and the external sort's k-way merge. Only the *answer* grows with the input, so RAM tracks the
answer rather than the scan: measured to 30 GB on a 64 GB machine, where the private running state
ranges from a few scalars to a 940 MB group table (§4.1) and peak RSS is dominated by mapped file
pages (§9).

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
the import. One core copies at roughly 10 GB/s; Polars and DuckDB scan the same directory at roughly
50-60 GB/s because they read it with every core. `ParallelIPCSource` closes that gap: it hands each of
`readers` threads its own files, so the read stage scales with cores.

The cost is batch **order**. Batches arrive interleaved across files, in whatever order the threads
finish. Every operator here is order independent in its *answer* — aggregates, the HLL sketch, group-by,
top-k, the external sort's run generation, both joins — though a float64 sum's last bits depend on
batch order (§10). So the only thing that needs `readers: 1` is a `filter -> sink_ipc` (or
`to_reader`) that must preserve the source's row order.

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
| `filter` / `project` → sink | **exact** | the predicate and every **numeric** projection compile into one fused kernel per batch (`docs/EXPR.md`); a string, temporal or decimal output takes the two-step path below | writing the sink | none |
| `sum`, `count`, `min`, `max` | **exact** | one fused aggregate kernel per batch | one scalar combine per batch | a few scalars |
| `mean` | **exact** | decomposed to `sum` + `count` on the GPU | combine, divide once at the end | 2 scalars |
| `variance`, `stddev` | **exact** | decomposed to `sum(x)`, `sum(x·x)` in float64 and `count`, all in one fused kernel | combine | 3 scalars |
| `count_distinct_approx` | **approximate** — relative standard error `1.04 / sqrt(2^p)`: 0.81 % at the default p = 14, 0.41 % at p = 16 | a new HyperLogLog kernel: one `atomic_fetch_max` per row into a `2^p`-register table | element-wise max of two register arrays | `2^p` bytes (16 KB at p = 14), whatever the dataset's size |
| `group_by` (dense integer key) | **exact** | `GroupBy` per batch, then element-wise `add` / `min` / `max` folds the batch result into the **GPU-resident** global table | none | `K` accumulators per aggregate on the GPU; spills to the host table above `gpuStateBudgetBytes` (512 MB default) |
| `group_by` (one integer key, `sum` / `count` / `mean`) | **exact** | a dense key encodes per batch and folds into the **GPU-resident** hash table; a **sparse** one skips the encoding entirely — one thread per row probes and inserts into that table directly (§4.1) | **none** | one slot per distinct key on the GPU |
| `group_by` (any other key or aggregate) | **exact** | `GroupByKeys` turns the key columns into dense ids on the GPU, then the aggregates | fold one row per group into a host table sharded by key hash across `mergeShards` threads | one entry per **distinct group**, not per row |
| `top_k` | **exact** | one comparison kernel rejects the batch against the running k-th value (§4.2); survivors go through the GPU top-k and are folded into the **GPU-resident** k rows | **none** | k rows on the GPU |
| `quantile` | **approximate** — see §6 | one radix argsort plus one gather per batch; only `compression` values reach the host | merge two weighted centroid lists and re-compress | ≤ `compression` centroids (1000 default) |
| `sort` with `limit n` | **exact** | top-n, not a sort: the same threshold prune, then the GPU top-k selection over the survivors (§5) | **none** | n rows on the GPU; **nothing spills** |
| `sort` (external, no limit) | **exact** | one radix argsort per batch; each output chunk is assembled with `take` | **the k-way merge is on the CPU** (see §5) | one batch per run + one output batch |
| `join` (broadcast) | **exact** | the existing GPU hash join per probe batch | writing the sink | the build side, in memory |
| `join` (broadcast) + `sum` / `count` / `min` / `max` / `mean` | **exact** | one kernel probes, gathers the build-side value and accumulates (§8.1); the build table is built once for the scan | **none** | the build table plus two 64-bit words per aggregate, on the GPU |
| `join` (broadcast) + `group_by` | **exact** | the hash join, then a gather of the key and value columns of the matched pairs only, into the resident group table (§4.1) | as the group-by's own path | the build table plus the group table |
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

### Streaming group-by: three global tables

* **Dense integer key** (`denseKeyCount` given, one key column already inside `[0, K)`): the global
  table lives on the **GPU** as one accumulator array per aggregate, `K` entries wide. Each batch runs
  `GroupBy` over the batch's keys and the per-batch result is folded into the global arrays with
  element-wise `add` / `min` / `max` kernels — **no host round trip at all**. When
  `K * aggregates * 16` bytes exceeds `gpuStateBudgetBytes` the state **spills to the host table**,
  and partials recorded before the spill are folded into it, so a 10-million-key group-by still runs
  inside a fixed GPU budget.
* **One integer key** with `sum` / `count` / `mean`: the **resident hash table** of §4.1, on the GPU
  for the whole scan. A dense key encodes per batch and folds one row per group into it; a sparse one
  skips the encoding and puts the rows in directly.
* **Anything else** (several key columns, a string or float key, `min` / `max` / `variance`):
  `GroupByKeys` (which turns any key columns into dense ids on the GPU) plus the aggregates give one
  row per group — typically thousands of rows for a batch of millions. The merge folds those rows into
  a host dictionary keyed by the key value itself. Exact, order independent, and its size is the
  number of distinct groups.

Every path returns the groups in ascending key order, so the result is deterministic across batch
layouts and runs. (This differs from pyarrow, which returns first-seen order; sort both sides before
comparing.)

### 4.1 The resident group table

`Sources/ArrowMetal/Stream/StreamGroupTable.swift`. For **one integer key** with `sum` / `count` /
`mean` aggregates, the global table is an open-addressing hash table in device memory that lives for
the whole scan: one slot per distinct key, and per aggregate a 64-bit sum and a 64-bit count. Ten
million groups with one aggregate is about 940 MB at a load factor of one third. Nothing crosses to
the host until `finish()`.

#### Publishing a 64-bit key with 32-bit atomics

Metal has no 64-bit atomics — checked, not assumed: on this M4 Max `atomic_ulong` has no
`compare_exchange`, no `fetch_add` and no `fetch_max` — and only guarantees `memory_order_relaxed` on
the 32-bit ones. So the usual "claim the slot, then publish the key beside it" handshake is not safe:
nothing orders the key's store against the claim, and a reader that sees a claimed slot may read a
key that is not there yet. `Kernels/HashTable.swift` avoids that by storing `row + 1` in the atomic
and reading the key out of an array nothing writes, which a *persistent* table cannot do.

This table publishes the key **through** the atomics instead. A slot is three 32-bit words holding
the key's 64 bits split 22 / 21 / 21, each stored as `field + 1`, so **0 means "not written"** and
every one of the 2^64 keys maps to three non-zero words. There is no reserved sentinel value, and so
no key a caller may not use — `0`, `-1`, `Int64.min` and `Int64.max` are ordinary keys, and the tests
say so. Each word is written with `compare_exchange(0 -> field + 1)`, which gives three properties:

* a word only ever goes from 0 to its final value, and never changes again;
* a thread accepts a slot only when **all three** words equal its own fields, and because those words
  are immutable once set, that decision is permanent and every thread agrees with it;
* a thread that mismatches any word walks on, which is what linear probing wants it to do.

The case a naive scheme gets wrong — two threads with the same key racing on one empty slot —
resolves without a spin (which can deadlock a divergent SIMD group) and without a duplicate slot: the
loser's compare-exchange fails *with the winner's value*, which is its own field, so it reads the
failure as a match and stops on the same slot. Two threads with different keys resolve the same way,
because the loser sees a value that is not its field and walks on. A thread that wins word 0 and then
loses word 1 to another key leaves a slot some other thread completes — the thread that won a word is
still inside its own probe step and goes on to write the rest — so no slot is left half written when
a dispatch ends.

Nothing in that argument needs an ordering Metal does not give, and nothing needs the keys in a
dispatch to be distinct. So one thread per **row** is safe, which is what removes the per-batch dense
encoding from a sparse, high-cardinality scan.

#### Three ways in

| Path | When | Per batch |
| --- | --- | --- |
| **distinct keys** | the key column is dense enough for `GroupByKeys`' range scan (a thousand regions, a date column, dictionary codes) | `GroupByKeys` gives one row per group, one insert dispatch places them, one accumulate folds them |
| **rows, atomic** | sparse key, and every aggregate is a `count` or an **integer** `sum` / `mean` | one thread per row inserts, one thread per row folds itself into its slot |
| **rows, dense ids** | sparse key with a **float64** or float32 sum in it | the insert also stamps a batch-local id onto every slot it touched; the batch's aggregates then run on the ordinary `GroupBy` and fold into distinct slots |

The chooser is one min/max over the key column per batch: a span small enough for
`GroupByKeys.rangeIsWorthIt` keeps the per-batch encoding, because a thousand groups fit in
threadgroup memory and beat a million atomic adds landing on a thousand slots; anything sparser goes
row-level. Both give the same answer, so it is only ever a choice of plan.

**The atomic accumulate** builds a 64-bit add out of two 32-bit ones: add the low word, take the
carry out of it into the high word. That is exactly two's-complement 64-bit addition, so it wraps
like `Int64` does, and being plain integer addition it is **independent of the order the rows reach a
slot in**. Counts and integer sums are therefore bit-identical to the per-batch path and to the host
table, run after run, whatever the batch layout — which the tests assert at cardinalities 1, 1k, 100k
and 2M.

**A float64 sum cannot take that path.** There is no 64-bit atomic to compare-exchange a double into,
and no emulation of one rounds a binary64 addition correctly or reproducibly. It takes the dense-id
branch instead, which still never sorts the key column — the resident table *is* the dictionary — and
keeps the correctly-rounded software adder `d_add`. The batch-local id is assigned by whichever
thread first swaps this batch's stamp into a slot, so exactly one thread per slot per batch assigns
one, and the ids are read back in a **later** dispatch, after the barrier that makes them visible.
The cost is two `uint` arrays the size of the table (about 270 MB at ten million groups), allocated
only when that branch is used.

**The per-group float64 sum is one thread per group**, not the one *threadgroup* per group the
segmented reduction gives it: at a million groups of one row that was 256 threads per row, and three
quarters of the batch. Its answer has to match the old one bit for bit, because a binary64 sum
depends on the order it was added in and the host table is checked against it — and it does, because
the shape of that reduction is fixed. Every butterfly step whose stride is at or above the run length
does nothing, so a run of `L` rows only uses the first `P` lanes (`P` the power of two at or above
`L`); and the butterfly over `P` lanes is the balanced tree whose leaves, left to right, are the
lanes in **bit-reversed** order. Walking the lanes in that order and merging equal-rank partial
results on a stack of at most nine rebuilds the same tree, including the rule that an empty
accumulator is replaced rather than added to. O(L) per group, same additions, same order. Runs long
enough that one thread each is the wrong shape keep the threadgroup version; the mean run length
decides, and since both give the same bits it is only a choice of dispatch.

#### Growth

Growth doubles the table and rehashes on the GPU under the same claim rule, and the row path checks
the load factor *after* its insert, when the number of new keys is known: the pre-emptive growth is
sized by what the last batch added, and a batch that outruns it grows the table and re-runs the
insert, which finds the keys that did land already present and claims the rest. Being wrong about the
estimate costs one extra pass, never an answer.

`min` / `max` / `variance`, multi-column keys and non-integer keys keep the host table, and the tests
check every resident path against it bit for bit — a null-key group, all-null groups, duplicate keys
per batch, empty and ragged batches, growth across the rehash threshold, keys at both ends of the
int64 range, and three integer key widths.

Both fold the same per-batch partials in the same order, one with Swift's `+` and one with `d_add`,
so the float64 sums agree to the last bit.

### 4.2 Threshold pruning

Once a top-k (or an `ORDER BY ... LIMIT n`) has k rows resident, the k-th value is a lower bound on
anything that can still enter the answer. Every later batch therefore starts with **one comparison
kernel over the key column**; the rows that pass are gathered — usually none, or a handful — and only
they reach the selection and the fold. The expected number of survivors in batch *i* of a scan for
the top k is `k / i`, so the work per batch collapses to a single pass over one column.

Two details make it exact rather than merely close:

* The comparison is `>=`, not `>`. A row that *ties* the running k-th value can still be the row the
  total order picks (`topK` breaks ties by row index), and for a multi-key sort it can still win on
  the second key. Keeping ties is what makes the pruned answer *the* answer.
* A threshold is only taken when the k-th row's key is **non-null**. Nulls sort last, so once k
  non-null rows are resident no null row can displace one; while fewer than k are, nothing is pruned.

The fold runs on the GPU thread rather than the merge thread, which leaves the merge stage idle *and*
means the threshold the next batch prunes with is always the newest one.

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
batch per run plus one output batch.

**`ORDER BY ... LIMIT n` never gets here.** With a limit the answer is n rows, so nothing has to
spill: the running n rows stay Metal-resident and each batch folds into them on the GPU thread, under
the threshold prune of §4.2. That is the same argument the bounded fan-in below already used — the
global first n are inside the union of each part's first n — applied one batch earlier, where it is
worth a thousand times more: a run used to be a whole million-row batch written to disk and read
back, and now there are no runs at all. A single-key head uses the GPU top-k selection, which returns
*exactly* the first n indices `argsort` would (same order-preserving key, same tie-break by row
index), so it is a cheaper way to compute the same prefix and not a different answer.

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

### 8.1 A join followed by an aggregate is one kernel

`Sources/ArrowMetal/Stream/StreamJoinAggregate.swift`.

A `join -> sum` used to be two operators with a whole joined table between them: every probe batch
ran the hash join, gathered **every column of both sides** into a new record batch, wrote it to a
sink, and the caller summed the rows it got back. The joined rows are the largest thing in that
pipeline and **none of them is the answer**.

```swift
try StreamQuery(ipc: path).filter(col("region") < 50)
    .join(dim, on: "region", buildKey: "region")
    .sum("amount")
```

```python
am.scan_ipc(path).filter(am.col("region") < 50).join(dim, on="region") \
  .agg([("sum", "amount", "total"), ("sum", "weight", "w")])
```

The terminal picks the plan. `.sink` / `.collect` still stream the joined rows out; `.sum`,
`.aggregate` and `.groupBy` fuse:

1. **The build table is built once.** `MetalRecordBatch.join` builds its hash table per call, which
   for a streamed join is per *batch*. The build side does not change across a scan, so
   `BroadcastBuildTable` builds it at construction with the same `hj_clear` / `hj_build` kernels and
   every probe batch reads it.
2. **The filter projects down to what the kernel reads** — the join key plus the probe-side value
   columns. A filtered join used to gather every column of the surviving rows, including a utf8 one.
3. **`jfa_probe` probes, gathers and accumulates in one kernel.** One thread per probe row walks its
   bucket chain and, for each match, reads the aggregate's value — from the probe row it is standing
   on, or from the build row the chain points at — and folds it into thread registers. A threadgroup
   tree reduction turns 256 of those into one partial per accumulator, and `jfa_fold` folds the
   partials into a **device-resident** accumulator pair that lives for the whole scan. Nothing
   proportional to the number of matches is allocated, nothing is gathered, and nothing crosses to
   the host until `finish()` reads sixteen bytes per aggregate. **The merge stage does nothing.**

Every accumulator is a raw `ulong` — a signed sum, an unsigned sum, or an IEEE-754 binary64 bit
pattern folded by the correctly-rounded software adder `d_add`, since Metal has neither a `double`
type nor a 64-bit atomic. The kernel is generated per (key type, aggregate list) and unrolled, so
there is no per-aggregate branch in the probe loop. A slot's *count* doubles as its "is empty" flag,
which is what makes `min` / `max` free of a per-type sentinel and an all-null input come out as null
rather than as an infinity.

`sum`, `count`, `min`, `max` and `mean` fuse; up to seven of them at once (Metal binds 31 buffers and
the probe itself needs nine). A column name is resolved against the probe side first and then the
build side, so `sum("weight")` reaches a build-side column; a build column shadowed by a probe column
of the same name is reachable as `name_right`, matching the materialising join's output names.

**The grouped form** (`join(...).groupBy(keys, aggs)`) keeps the join's index pairs, because a
per-group accumulator needs a table probed once per matched *pair*, and the resident group table's
race-free insert relies on every key in a dispatch being distinct (§4.1), which rows are not. It
gathers **only the key columns and the aggregated values** of the matched pairs — two or three
columns, never the joined batch's eight — and folds those into the ordinary streaming group-by, whose
global table is already resident. A key may come from either side: a probe-side key is gathered with
the left index array, a build-side key with the right one.

Semantics are the inner join's and match Polars and DuckDB: duplicate build keys multiply rows, a
null key on either side never matches, and a probe row with no match contributes nothing. **A left
join followed by an aggregate raises**, naming what to use instead, rather than returning a number
that counts unmatched rows wrongly; so do `variance`, `stddev`, `count_distinct_approx`, a grace hash
join, and a non-numeric value column.

`Tests/ArrowMetalTests/StreamJoinFusionTests.swift` checks the fused answer against the in-memory
join followed by the in-memory aggregate — bit for bit for integer aggregates, within one ulp per
element for float64 sums — over duplicate build keys, null keys on both sides, empty batches, a batch
with no matches at all, a build side of one row and of a million, a group-by key from each side, and
int32 and int64 keys.

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

### The row-level group-by: 4 GB, before and after

Measured on a **4.03 GB** run of the same generator (`--size-gb 4`: 76,000,000 rows, 4 files, 76
batches of 1M rows) with `readers = 2`, best of three **interleaved A/B rounds** — before, after,
then the reference engines, three times round, one cell process at a time, same dataset and same page
cache, and only the loaded library changes between "before" and "after". "before" is the per-batch
dense encoding of the previous section; "after" is the row-level path of §4.1.

| Workload | before | **after** | Polars | DuckDB | RSS before | **RSS after** | Polars RSS | DuckDB RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| group by `bigkey` -> sum (10M groups) | 2,156 | **614** | 393 | 638 | 4.12 GB | 4.74 GB | 2.90 GB | 4.63 GB |
| group by `region` -> sum, count (1k groups) | 199 | 200 | 106 | 240 | 2.30 GB | 2.32 GB | 2.32 GB | 0.86 GB |

`groupby_10m` is **3.5x** faster and now **beats DuckDB**; it was 5.5x behind Polars and is 1.6x
behind. `groupby_1k` is unchanged to the millisecond, which is the point of the chooser: a thousand
contiguous values is exactly the shape the range encoding is for, the row path is never taken, and
nothing about that workload moved.

Where the batch's time went, per stage, at 76 batches:

| Stage | before | after |
| --- | ---: | ---: |
| read (two reader threads) | 0.40 s | 0.43 s |
| GPU stage | 2.05 s | 0.27 s |
| merge stage | 0.23 s | 0.50 s |
| merge stall (GPU stage waiting on the merge queue) | 0.00 s | 0.22 s |
| overlap | 1.26x | 2.07x |

The work moved rather than only shrank: the resident table can only be touched from the merge stage,
so the insert, the aggregates and the fold all run there now and the GPU stage is left with the read
and the filter. Total GPU-side work went from 2.28 s to 0.77 s. What remains in the merge is the
insert of a million rows into a 33-million-slot table, the batch's `GroupBy`, and the fold.

Two intermediate points, measured the same way on the same dataset, showing what each half bought:

| `group by bigkey` | wall | merge stage |
| --- | ---: | ---: |
| per-batch encoding (before) | 2,156 ms | 0.23 s |
| row-level, dense ids, segmented sum | 1,523 ms | 1.47 s |
| row-level, dense ids, one thread per group | **614 ms** | 0.50 s |
| the same query as `count(*)` (atomic path) | 456 ms | 0.31 s |
| the same query as `sum(qty)`, int32 (atomic path) | **330 ms** | 0.22 s |

The last two lines are the point of the atomic path: with an integer aggregate the batch never builds
dense ids, never sorts anything and never runs a second kernel over its groups, and the whole query
is read-bound. A float64 sum pays for the dense ids and the per-group reduction because Metal has no
64-bit atomic to fold a double with.

**Peak RSS rises by 0.69 GB** on `groupby_10m`, and that is the price of the dense-id branch: two
`uint` arrays the size of the table (2 x 134 MB at 33.5 M slots) plus the batch's segment arrays. The
atomic path allocates neither. Nothing else in the table moved.

### The earlier measurement: 8 GB, before and after the resident-state work

The 30 GB table this section used to carry is kept below, because it is the size at which the read
stops being the limit. The current numbers are from an **8.06 GB** run of the same generator
(`--size-gb 8`: 152,000,000 rows, 8 files, 152 batches of 1M rows) with `readers = 2`, best of three
**interleaved A/B rounds** in which only the loaded library changes — same benchmark script, same
dataset, same page cache, one cell process at a time. "before" is main; "after" is §4.1, §4.2 and §5.
Wall-clock milliseconds, then peak RSS.

| Workload | before | **after** | Polars | DuckDB | RSS before | RSS after | Polars RSS | DuckDB RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sum(amount) where region < 100` | 325 | 324 | **158** | 230 | 2.30 GB | 2.30 GB | 1.86 GB | 0.97 GB |
| group by `region` -> sum, count (1k groups) | 509 | 513 | **164** | 268 | 2.52 GB | 2.52 GB | 2.26 GB | 1.08 GB |
| group by `bigkey` -> sum (10M groups) | 7,629 | **4,110** | 743 | 864 | 5.14 GB | 4.14 GB | 5.31 GB | 8.04 GB |
| top 100 by `amount` | 527 | **386** | 176 | 240 | 2.51 GB | 2.51 GB | 1.79 GB | 0.81 GB |
| `count_distinct(id)` | 329 | 331 | 694 | **223** | 2.30 GB | 2.30 GB | 4.78 GB | 0.79 GB |
| order by `amount` desc limit 1000 | 2,297 | **524** | 177 | 241 | 3.85 GB | 2.52 GB | 2.91 GB | 0.83 GB |
| broadcast join + sum | 564 | 562 | 281 | 244 | 2.47 GB | 2.48 GB | 2.15 GB | 0.95 GB |

`sort_limit` is **4.4x** faster and spills no runs at all (peak RSS falls by 1.3 GB with them),
`groupby_10m` is **1.86x** faster on a gigabyte less memory, `topk` is **1.4x** faster. `filter_sum`
and `count_distinct` are unchanged within noise, which is the point: they were already right.

The four cross-checked workloads (`filter_sum`, `groupby_1k`, `topk`, `sort_limit`) are compared
against Polars' answer with a 1e-9 relative tolerance on float sums, and all four pass.

### Where the pipeline's time goes now

`overlap` is (read + gpu + merge) / wall: 1.0 for a serial pipeline, and above that by however much
the stages actually ran at the same time. With `readers = 2` the read column is the sum across both
reader threads, so it can exceed wall on its own. The stall columns are the GPU stage waiting: for a
batch the readers had not finished, and for the merge queue to drain.

| Workload | Overlap | Read (s) | GPU (s) | Merge (s) before | **Merge (s) after** | Read stall (s) | Merge stall (s) | Wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `filter_sum` | 2.19x | 0.57 | 0.09 | 0.00 | 0.00 | 0.21 | 0.00 | 0.32 |
| `groupby_1k` | 2.58x | 0.56 | 0.48 | 0.30 | **0.22** | 0.02 | 0.00 | 0.51 |
| `groupby_10m` | 1.24x | 0.72 | 4.01 | 2.35 | **0.34** | 0.01 | 0.00 | 4.11 |
| `topk` | 2.60x | 0.60 | 0.34 | 0.48 | **0.00** | 0.03 | 0.00 | 0.39 |
| `count_distinct` | 2.19x | 0.57 | 0.09 | 0.00 | 0.00 | 0.21 | 0.00 | 0.33 |
| `sort_limit` | 2.13x | 0.57 | 0.49 | 1.19 | **0.00** | 0.01 | 0.00 | 0.52 |
| `broadcast_join` | 2.42x | 0.62 | 0.29 | 0.00 | 0.00 | 0.02 | 0.00 | 0.56 |

**The merge stage is zero on every operator that had one**, which was the point: nothing folds a
batch's answer on the host any more except the host group-by table (§4), and that table is only
reached by the keys and aggregates the resident one does not take. **The merge stall is 0.00 s
everywhere**, so the bounded queue between the GPU stage and the merge never fills, and peak RSS
stays flat at 2.3 to 2.5 GB over 8 GB of data.

**Where we still lose, and why.**

* **`filter_sum` and `count_distinct` are read-bound at `readers = 2`.** Their GPU stage is 0.09 s of
  a 0.32 s wall and the read stall is 0.21 s of it, so what is being measured there is two reader
  threads moving 8 GB against Polars reading the same directory with every core at 51 GB/s. The
  30 GB numbers below, taken with `readers = 8`, are what the same operators look like once the read
  is not the limit.
* **`groupby_10m` was still 5.5x behind Polars at this point**, and the reason was precisely located:
  the merge was 0.34 s, but the **GPU stage was 4.0 s**, and essentially all of it was the *per-batch*
  `GroupByKeys` that turns a million int64 keys into dense ids. `bigkey` spans ten million values
  over a million-row batch, too sparse for the range-encoding path, so every batch paid a full radix
  sort of its key column plus an atomic group-by min to pick each group's representative row — with a
  host loop over that batch's ~950,000 groups inside it. That is what the row-level path of §4.1
  removed; the 4 GB table above is the result.
* **`groupby_1k`, `topk`, `sort_limit` and the join are within 2x to 3x** of Polars and DuckDB, and
  are read- or fixed-cost-bound rather than dominated by any one stage.
* **The broadcast join is unchanged** because the benchmark streams its joined rows back to Python
  and sums them there. Fusing the aggregate into the probe, so the join's output never leaves the
  GPU, is measured separately below.

### The fused join + aggregate, measured

A separate 4.03 GB run (`--size-gb 4 --ipc-format file`: 76,000,000 rows, 4 files, 76 batches of 1M
rows) with `readers = 2`, best of **three interleaved rounds**, one cell process at a time. Both
sides are the same binary over the same dataset in the same run: `broadcast_join_unfused` is the old
plan (gather every matched pair into a record batch, collect it, sum it in Python) and
`broadcast_join` is the fused one (§8.1). Both compute `sum(amount)` from the probe side and
`sum(weight)` from the build side after `region < 50`, and both agree with Polars to the last two
digits of a double (the benchmark's own cross-check passes on every round).

| Cell | Wall | Peak RSS | GPU stage | Merge stage | Read stage | Read stall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `broadcast_join` **unfused** | 379 ms | 2,462 MB | 0.14 s | 0.00 s | 0.325 s | 0.03 s |
| `broadcast_join` **fused** | **336 ms** | **2,283 MB** | **0.04 s** | 0.00 s | 0.315 s | 0.13 s |
| Polars | 233 ms | 1,836 MB | | | | |
| DuckDB | 229 ms | 862 MB | | | | |

**The GPU stage is 3.7x smaller** — 0.14 s to 0.04 s — which is what the fusion actually did: the
gather of every column of every matched pair is gone, and so is the joined batch. Peak RSS falls by
180 MB and host CPU by 170 ms (698 to 529 ms), because nothing is collected and nothing is summed in
Python. (The fused GPU stage measures between 0.04 and 0.09 s across rounds; the spread is the
one-off compile of the generated kernel on the first batch, which the unfused path does not pay
because its kernels are already in the pipeline cache. The unfused stage is 0.14 to 0.15 s in every
round.)

**The wall clock only moves from 379 ms to 336 ms, and it still loses to Polars and DuckDB, because
at `readers = 2` this cell is read-bound.** The read stage is 0.315 s of a 0.336 s wall and the GPU
stage waits 0.13 s of it for a batch the two reader threads had not finished: what is being measured
is two threads moving 4 GB against Polars reading the same directory with every core. One
supplementary round at `readers = 4` shows the operator with the read out of the way — **359 ms
unfused to 264 ms fused, a 1.36x speed-up** (GPU 0.147 s to 0.043 s), which brings it within 13 % of
DuckDB's 229 ms; the cost is peak RSS, 4.4 to 4.9 GB, because four part files are mapped at once.

### The earlier 30 GB measurement

Kept because 30 GB with `readers = 8` is the size at which the read stops being the limit and the
per-batch work shows through. Same M4 Max, a **30.21 GB** directory: 570,000,000 rows, 30 files, 570
batches of 1M rows.

| Workload | ArrowMetal (before) | Polars | DuckDB | ArrowMetal RSS | Polars RSS | DuckDB RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sum(amount) where region < 100` | **527** | 515 | 418 | 8.9 GB | 2.0 GB | 1.1 GB |
| group by `region` -> sum, count (1k groups) | 1,765 | **554** | 625 | 9.5 GB | 3.1 GB | 1.3 GB |
| group by `bigkey` -> sum (10M groups) | 19,946 | **4,031** | 5,232 | 12.4 GB | 13.1 GB | 25.7 GB |
| top 100 by `amount` | 4,060 | 587 | **544** | 8.4 GB | 2.5 GB | 0.9 GB |
| `count_distinct(id)` | 532 | 3,825 | **451** | 8.8 GB | 17.0 GB | 0.9 GB |
| order by `amount` desc limit 1000 | 11,121 | 642 | **567** | 9.3 GB | 2.7 GB | 0.9 GB |
| broadcast join + sum | 1,807 | 764 | **423** | 9.2 GB | 2.3 GB | 1.1 GB |

The resident top-n and threshold pruning of §4.2 and §5 were measured once on that dataset before it
was deleted, in a single interleaved A/B round with `readers = 8`, printed to a tenth of a second:

| Workload | before | after |
| --- | ---: | ---: |
| top 100 by `amount` | 4.7 s | **0.8 s** |
| order by `amount` desc limit 1000 | 11.2 s | **1.1 s** |
| group by `region` (1k groups) | 1.3 s | 1.0 s |
| `sum(amount) where region < 100` | 0.5 s | 0.5 s |
| `count_distinct(id)` | 0.5 s | 0.5 s |
| broadcast join + sum | 1.7 s | 1.6 s |

At that size **top-k and `ORDER BY ... LIMIT` come within 1.4x-1.9x of Polars and DuckDB** — 0.8 s
against 0.587 and 0.544, 1.1 s against 0.642 and 0.567 — down from 4.7 s and 11.2 s, because 30 GB is
where the per-batch work, not the read, was the limit. The resident group table (§4.1) landed after
that run, so `groupby_10m` was not re-measured at 30 GB; its 8 GB improvement is in the table above.

pyarrow.dataset (measured in a separate single-reader run, so not comparable cell by cell) finished
`groupby_10m` in 557 s and **timed out at 900 s on `count_distinct`**.

* **`count_distinct_approx` beats Polars by 7x** (532 ms against 3,825 ms) at 8.8 GB of peak RSS
  against Polars' 17.0 GB, and its own running state is a 16 KB sketch. DuckDB's sketch is faster
  still, but its answer is **12.4 % off** against ArrowMetal's **0.007 %** — DuckDB's
  `approx_count_distinct` is tuned for a much smaller sketch. Against an exact count this is the
  operator where streaming on a GPU clearly pays.
* **`filter_sum` ties Polars** at 57 GB/s — a whole-dataset filtered aggregate over 570 million rows
  in half a second, with the CPU doing nothing but reading bytes.

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
* **Threshold pruning is exact but not free of assumptions**: it compares against the running n-th
  value with `>=`, which needs a comparison kernel for the key column's type. Types without one
  (`utf8`, `binary`, boolean, decimals) simply run unpruned — correct, just not faster.
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
* **A group-by over ten million groups is 1.6x slower than Polars** (§9), down from 5.5x. The rows go
  straight into the resident table now, so there is no per-batch encoding left to remove; what is
  left is the insert itself, the batch's `GroupBy` and the fold, all in the merge stage, which the
  GPU stage then waits 0.22 s on. Splitting the resident table into shards a batch could insert into
  from more than one stage is not implemented.
* **A float64 sum cannot use the row-level atomic accumulate**, because Metal has no 64-bit atomic
  (`atomic_ulong` has no compare-exchange, `fetch_add` or `fetch_max` on this hardware) and no
  emulation of one rounds a binary64 addition correctly. It takes the dense-id branch of §4.1
  instead, which costs a second pass over the batch's groups and two `uint` arrays the size of the
  table. Counts and integer sums skip all of that; on the same query and dataset that is 330 ms
  against 614 ms.
* **The row-level path's float64 sum is deterministic but not associative-order-free**: a group's
  rows are added in a fixed order that depends on how the batches fell, so the same dataset read with
  a different `batch_rows` can differ in the last bits, exactly as the per-batch path always could.
  Integer sums and counts have no such dependence — they are 64-bit integer adds and commute — so
  they are bit-identical across paths, batch layouts and runs. There is no non-deterministic option
  to turn on.
* **The resident group table takes one integer key and `sum` / `count` / `mean`.** `min` / `max`
  (they would need a per-slot atomic minimum), `variance` (a third accumulator), multi-column keys
  and non-integer keys all keep the host table, which is exact but folds one row per group per batch
  on the CPU.
* **A fused join + aggregate is an inner broadcast join only.** `join(...).sum(...)` and
  `join(...).groupBy(...)` fuse (§8.1); a **left** join followed by an aggregate raises, because it
  would have to count unmatched probe rows and that is not implemented, and so does a **grace hash**
  join, which writes its partitions to disk and has no fused form. Both name the plan to use
  instead. `variance` / `stddev` / `count_distinct_approx` and non-numeric value columns raise too.
* **A fused aggregate takes at most seven aggregates.** Metal binds 31 buffers and the probe needs
  nine of them, three per aggregate after that.
* **The dense-id branch costs about 270 MB at ten million groups** — two `uint` arrays the size of
  the table — which is why `groupby_10m`'s peak RSS rises by 0.69 GB. Packing the batch stamp and the
  dense id into one word is possible and is not implemented.
* **Each batch is one Metal command buffer, not several.** The fixed cost per batch is paid once per
  batch; a larger `batch_rows` amortises it, at the cost of memory per batch.
* **`readers > 1` raises peak RSS by roughly `readers` x the part-file size**, because that many
  files are mapped at once. Those pages are reclaimable page cache, not anonymous memory, but they
  do show up in `ru_maxrss`.
* **`am_stream_query` supports filter + project and filter + aggregate**; a grouped query goes through
  `am_stream_group_by`, because the Expr compiler's `group_by` takes a dense integer key and the
  streaming group-by takes arbitrary key columns. A *joined* query has no s-expression form either:
  it goes through `am_stream_join_aggregate` / `am_stream_join_group_by`, which take the build side
  as an Arrow C Stream.
* Arrays above 2³² rows are still refused (`Dispatch.checkLength`), which bounds one *batch*, not a
  dataset.
