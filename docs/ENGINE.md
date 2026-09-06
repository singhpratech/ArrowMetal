# The query engine

A lazy, optimizing query engine that runs on the GPU: `filter`, `select`, `group_by`, `sort`, `join`,
`join_asof`, window functions, `unique`, `explode` and `concat`, planned as a whole and executed inside
one Metal command buffer.

```python
import arrowmetal as am

q = (am.scan(table)
       .filter((am.col("amount") > 0) & (am.col("qty") > 5))
       .group_by("region").agg(am.agg.sum("amount", "revenue"), am.agg.count("orders"))
       .sort("revenue", descending=True)
       .limit(10))
q.explain()      # the optimized plan, the way Polars prints one
q.collect()      # -> pyarrow.Table
```

```swift
let df = LazyFrame(PlanSource(name: "sales", batch: batch))
    .filter(col("amount") > 0)
    .groupBy(["region"], [ExprAggregate(.sum, col("amount"), name: "revenue")])
    .sort([SortKey("revenue", descending: true)])
    .limit(10)
let out = try df.collect()
```

---

## Why this layer exists

`docs/EXPR.md` describes the fused expression compiler: it turns a whole element-wise tree plus one
terminal (project, filter+project, aggregate, group-by over dense keys) into **one** runtime-generated
Metal kernel. That is the fast half of a query engine, and it was already here.

What was missing is everything that moves rows around — sorting, joining, grouping by keys that are not
already dense, windows, limits — and, more importantly, the layer that *decides*: which parts of a query
become one kernel, which columns are read at all, where a predicate should run. Without it, a caller
writing a real query gets one Arrow kernel per operator and a CPU round trip between each.

So the engine is a plan, three layers of decision over it, and one of execution:

| File | What lives there |
|---|---|
| `Sources/ArrowMetal/Engine/LogicalPlan.swift` | the plan, its schema inference and type checking, `describe()` |
| `Sources/ArrowMetal/Engine/Optimizer.swift` | the rewrite rules and the cardinality estimates |
| `Sources/ArrowMetal/Engine/PhysicalPlan.swift` | fusion planning: which regions become one kernel |
| `Sources/ArrowMetal/Engine/Executor.swift` | running it, and `LazyFrame` |
| `Sources/ArrowMetal/Engine/WindowOps.swift` | window functions over partitions |
| `Sources/ArrowMetal/Engine/Concat.swift` | column and batch concatenation |
| `Sources/ArrowMetal/Engine/PlanJSON.swift` | the serialised plan grammar |
| `Sources/ArrowMetal/Kernels/JoinExtra.swift` | the rest of the join matrix and the as-of join |
| `Sources/ArrowMetalC/ArrowMetalC_Plan.swift` | `am_plan_*` over the C ABI |
| `python/arrowmetal/lazy.py` | `am.scan(...)`, `am.LazyFrame`, `am.agg` |

## Operators

| Operator | Swift | Python | Runs as |
|---|---|---|---|
| scan | `.scan(source, columns:)` | `am.scan(table)` | no kernel: the columns by reference |
| filter | `.filter(_:)` | `.filter(expr)` | one fused kernel (predicate + compaction) |
| select / project | `.select(_:)` | `.select(...)` | one fused kernel; bare column references cost nothing |
| with_columns | `.withColumns(_:)` | `.with_columns(...)` | as select, keeping the other columns |
| aggregate | `.aggregate(_:)` | `.agg(...)` | one fused kernel, threadgroup partials |
| group-by aggregate | `.groupBy(_:_:)` | `.group_by(...).agg(...)` | `GroupByKeys` for the ids, then one fused group-by kernel (or `GroupBy`'s per-aggregate kernels) |
| sort | `.sort(_:)` | `.sort(by, descending)` | `lexsort` (one stable radix sort per key) + `take` |
| limit / head / slice | `.limit(_:offset:)` | `.limit(n)`, `.head(n)`, `.slice(o, n)` | a slice, or the top-k selection kernel when it follows a single-key sort |
| unique | `.unique(subset:)` | `.unique(subset=)` | `GroupByKeys` + the lowest row index per group |
| join | `.join(_:leftOn:rightOn:how:)` | `.join(other, on=, how=)` | GPU hash join; see the matrix below |
| join_asof | `.joinAsof(_:_:)` | `.join_asof(other, on=, by=, strategy=, tolerance=)` | GPU binary search per probe row |
| concat | `.concat(_:)` | `am.concat([...])` | `memcpy` over unified memory |
| window | `.window(_:)` | `.with_row_number(...)` and friends | sort once, work in sorted order, scatter back |
| explode | `.explode(_:)` | `.explode(columns)` | list offsets to a gather index, then `take` |

Aggregates are `sum`, `min`, `max`, `mean` and `count`, over any expression, whole-table or per group.
Window functions are `row_number`, `rank`, `dense_rank`, `lag`, `lead`, `cum_sum`,
`rolling_sum` / `rolling_mean` / `rolling_min` / `rolling_max`, and any aggregate broadcast over its
partition.

## The optimizer

Every rule preserves the result exactly — same rows, same nulls. `explain()` lists the ones that fired.

| Rule | What it does |
|---|---|
| `constant_folding` | evaluates literal-only subtrees, collapses `and(true, x)`, `x * 1`, `not(not x)`, dead `coalesce` arms |
| `filter_fusion` | `filter(filter(x, a), b)` → `filter(x, and(a, b))`: one compaction pipeline instead of two |
| `predicate_pushdown` | moves each conjunct as far down as it can go: through projections (substituting the projected expression), through `with_columns`, below sorts, below a group-by when it only touches key columns, into each side of a join where the join kind allows, into every branch of a concat, and below an explode |
| `projection_pruning` | works out what each node's parent actually needs and narrows the scan to those columns; drops projection and window outputs nothing reads |
| `expression_cse` | drops duplicate outputs with the same canonical text (the fused kernel's own CSE only sees one query at a time) |
| `join_reorder` | an inner join is commutative and `hashJoin` builds its table from the **right** side, so the smaller estimated input is put there; a projection on top restores the caller's column order |

Two rules are deliberately absent. A filter is **not** pushed below a `limit`, because that changes
which rows survive. A filter is **not** pushed below a `window`, because a window function's value
depends on the rows in its partition.

Cardinality comes from `PlanStats`: row counts flow up from the scans, a filter is assumed to keep a
quarter of its rows, a group-by to fold by 16 per key, an inner join to produce `min(left, right)`.
Setting `stats.useGPUDistinctCounts` replaces the per-column guess with a real `GroupByKeys` pass; it
costs a pass over the column, so it is off unless the plan will run for far longer than the estimate.

### `explain()`

```
LOGICAL PLAN
  LIMIT 10
    SORT BY [total DESC]
      GROUP_BY [region] AGG [(sum "total" (col "amount")), (count "n")]
        FILTER (gt (col "amount") (int 100))
          SCAN sales [region, amount] 2/5 columns, 1000 rows

PHYSICAL PLAN
  HEAD 10
    TOP-K 10 BY total DESC
      HASH-AGGREGATE [region] AGG [(sum "total" (col "amount")), (count "n")] (per-aggregate kernels)
        FUSED-FILTER-PROJECT [region, amount] FILTER (gt (col "amount") (int 100))
          SOURCE sales [region, amount] 2/5 columns, 1000 rows

RULES APPLIED: projection_pruning
```

Three of the five source columns are never read. The filter and the projection are one kernel. The
`sort` + `limit` became a top-k selection, which touches each row once instead of running eight radix
passes over all of them.

A join, with the predicate pushed into the side it belongs to:

```
LOGICAL PLAN
  LIMIT 5
    SORT BY [w DESC]
      SELECT [region_name, (mul (col "amount") (col "weight")) AS "w"]
        JOIN INNER ON [region]
          FILTER (gt (col "qty") (int 5))
            SCAN sales [region, amount, qty] 3/5 columns, 1000 rows
          FILTER (gt (col "weight") (float 1.0))
            SCAN dim [region, region_name, weight] 3/3 columns, 200 rows

PHYSICAL PLAN
  HEAD 5
    TOP-K 5 BY w DESC
      FUSED-PROJECT [(mul (col "amount") (col "weight")) AS "w", region_name]
        HASH-JOIN INNER ON [region]
          FUSED-FILTER-PROJECT [region, amount, qty] FILTER (gt (col "qty") (int 5))
            SOURCE sales [region, amount, qty] 3/5 columns, 1000 rows
          MASK-FILTER (gt (col "weight") (float 1.0))
            SOURCE dim [region, region_name, weight] 3/3 columns, 200 rows

RULES APPLIED: predicate_pushdown, projection_pruning
```

As written, that query was a join of two full tables followed by one `and` of two predicates. The
`dim` side falls back to `MASK-FILTER` — one fused kernel for the predicate, then the ordinary
compaction kernels — because it has to carry a `utf8` column through, and the expression compiler will
not materialise a string output.

## Fusion planning

The physical planner decides what becomes one kernel. The rule is short, because the expression
compiler already defines what it can do:

* An element-wise region — arithmetic, comparisons, null logic, casts, string predicates — is one
  kernel (`FUSED-PROJECT`).
* A `filter` immediately under a `select` or an `aggregate` joins that kernel, so the predicate is
  evaluated inside the same pass that computes the outputs and no boolean column is ever written
  (`FUSED-FILTER-PROJECT`, `FUSED-AGGREGATE`).
* A `select` output that is a bare column reference is not a kernel at all: the column is carried by
  reference (`CARRY`).
* A region stops being fusable the moment an output would be `utf8`, or a column it must read is
  temporal, decimal, list, struct, map, union or dictionary encoded. Then the predicate becomes a mask
  (`MASK-FILTER`) and the ordinary per-column compaction kernels run — which is what the package did
  before fusion existed, so nothing is lost, only the extra pass.
* Everything that moves rows is its own node.

## Execution

The whole tree runs inside one `MetalContext.batch { }`, so every kernel goes into one command buffer
and the ~150 µs round trip is paid once for the query rather than once per operator. Where an
operator's *length* is decided by the GPU — a filter — the result is a pending array whose length lives
in a device buffer and the next kernel binds that buffer instead of a CPU-known count
(`docs/DESIGN.md`, "Lengths flow on the GPU"), so a filter feeding a projection feeding another filter
never returns to the CPU.

Four operators are unavoidable sync points, because the CPU has to know a count before it can size the
next dispatch: `GroupByKeys` (the number of groups), the hash join (the number of pairs), the top-k
selection, and `explode` and `slice` (which read offsets). `flush(reopen: true)` commits, waits and
reopens the batch at each, so batching resumes immediately afterwards. Intermediate buffers come from
and go back to `MetalContext.pool`, which parks rather than recycles while a batch is open.

## The join matrix

`Kernels/Join.swift` does inner and left over one `int32` or `int64` key. `Kernels/JoinExtra.swift`
does the rest.

| | single int32/int64 key | multi-column keys | utf8 keys | mixed types |
|---|---|---|---|---|
| inner | direct hash join | ✓ | ✓ | ✓ |
| left | direct hash join | ✓ | ✓ | ✓ |
| right | probes the right side | ✓ | ✓ | ✓ |
| full outer | left join + the unmatched right rows | ✓ | ✓ | ✓ |
| semi | match flags + `filter` | ✓ | ✓ | ✓ |
| anti | match flags + `filter` | ✓ | ✓ | ✓ |
| as-of (backward / forward / nearest, with `by` and `tolerance`) | int and temporal keys | partition columns of any type | partition columns of any type | ✓ |

**Making arbitrary keys joinable.** A join needs an equality that is exact, not probabilistic. Rather
than hashing a wide or string key into 64 bits and adding a verification pass, the key columns of
*both* sides are concatenated and handed to `GroupByKeys` — the machine that already turns arbitrary
key columns into dense `int32` ids with an injective mapping. Two rows get the same id exactly when
their keys are equal, so the `int32` hash join over those ids *is* the original join, with no
collisions to resolve. It costs one densification pass over `left + right` rows. The common case that
needs none — a single `int32` or `int64` key on both sides — goes straight to `hashJoin`.

**Nulls never match**, in either direction and for every join kind, which is Arrow's, Polars' and SQL's
rule. `GroupByKeys` gives null keys a group of their own, so the ids alone would match them up; the
per-side "every key column is valid" bitmap is attached to the id array instead, and `hashJoin` skips a
null key on both the build and the probe side.

**Row order.** Inner and left keep probe (left) order, and a left row's matches are contiguous. Right
keeps right order. Full is left order followed by the unmatched right rows. Semi and anti keep left
order and emit each row once. The order of several matches *within* one probe row is unspecified.

**Semi, anti and the right tail of a full outer join** all need the same primitive: which rows of one
side the join touched. That is one `jx_scatter_flag` dispatch over the index pairs plus a `filter` of
the row indices — no second hash table.

**As-of.** The build side is sorted by `(partition, key)` on the GPU with its null keys dropped, and
every probe row does two binary searches — one for its partition's range in the sorted array, one for
the key inside it. That is `log2(n)` dependent loads per probe row and no table at all, which is the
whole reason an as-of join does not need a hash. `strategy` is `backward` (last key at or before),
`forward` (first at or after) or `nearest` (ties to backward, as Polars does); `tolerance` bounds
`|probe − match|`. An as-of join is a left join: every probe row comes out, in its original order, with
nulls where nothing matched.

## Window functions

Always the same shape, and it is the shape a GPU likes: **sort once, work in sorted order, scatter
back.**

1. `lexsort` by `(partition keys…, order keys…)`. Every partition is now one contiguous run and inside
   it the rows are in the window's order — one stable radix sort per key column.
2. Compute the answer as a function of a row's *position* in that order, its partition's first
   position, and its tie group's first position. All three are a `GroupBy` minimum over the row
   positions plus a `take`.
3. `take` the result through the inverse permutation (which is `argsort` of the permutation), so the
   output is aligned to the caller's rows.

Ranking, `lag` and `lead` fall out of step 2 with no new kernel: the index a row wants is
`if_else(position - k >= partition_start && position - k <= partition_end, position - k, null)`, which
is one fused expression, and `take` of a null index gives a null. `RANK` is the tie group's first
position minus the partition's first position, plus one; `DENSE_RANK` is a running sum of tie-group
marks minus its value at the partition's first row.

`cum_sum` and the four `rolling_*` functions are the exception: their existing kernels
(`Kernels/Cumulative.swift`, `Kernels/Window.swift`) are already exactly right per partition, so each
partition's contiguous slice is handed to them and the results concatenated. That is one dispatch per
partition — the right trade below a few thousand partitions, and capped at 8192 with an error above.

Nulls follow SQL `ORDER BY x NULLS LAST`, which is what `lexsort` does in both directions: a null key
sorts after every value, and all nulls of one key form one tie group.

## The plan grammar

Expressions already have a text form — the s-expression grammar of `docs/EXPR.md`, which is the cache
key the fused compiler uses — so a plan reuses it verbatim and only spells the *operators*. JSON is the
wrapper, because a plan is a tree of records with optional fields.

```json
{"op": "sort", "by": [["total", true]], "input":
  {"op": "group_by", "keys": [["region", "(col \"region\")"]],
   "aggs": [["sum", "total", "(col \"amount\")"]], "input":
    {"op": "filter", "predicate": "(gt (col \"amount\") (int 100))",
     "input": {"op": "scan", "source": "sales"}}}}
```

| op | fields |
|---|---|
| `scan` | `source`, `columns`? |
| `filter` | `input`, `predicate` |
| `select` / `with_columns` | `input`, `exprs`: `[[name, sexpr], …]` |
| `aggregate` | `input`, `aggs`: `[[op, name, sexpr?], …]` |
| `group_by` | `input`, `keys`, `aggs` |
| `sort` | `input`, `by`: `[[column, descending], …]` |
| `limit` | `input`, `count`, `offset`? |
| `unique` | `input`, `subset`? |
| `join` | `left`, `right`, `left_on`, `right_on`, `how`, `suffix`? |
| `join_asof` | `left`, `right`, `left_on`, `right_on`, `by`?, `by_right`?, `strategy`?, `tolerance`?, `suffix`? |
| `concat` | `inputs` |
| `window` | `input`, `specs`: `[{name, fn, column?, n?, partition_by?, order_by?}, …]` |
| `explode` | `input`, `columns` |

Over the C ABI the tables are registered once (`am_plan_source`, which takes the `am_array` handles the
caller already holds) and the plan text carries no data, so the same text can be re-run against new
sources and `am_plan_explain` can print a plan without touching the GPU. `include/arrowmetal.h` carries
the same grammar for C consumers.

## Numbers

M4 Max, 50 M rows, best of 5, in process, against Polars 1.44 lazy (16 threads) and DuckDB 1.5
(16 threads) on the same Arrow buffers. `Benchmarks/engine_bench.py`. "CPU ms" is process CPU time over
the same run: it is what the query cost the machine, next to what it cost the caller in latency.

| case | implementation | wall ms | CPU ms | vs ArrowMetal |
|---|---|---:|---:|---:|
| **(a)** `sum(amount), count(*)` where `region < 20 and qty > 10` | **ArrowMetal lazy** | **2.54** | 1.0 | — |
| | polars lazy | 16.41 | 26.4 | 6.5x |
| | duckdb | 194.79 | 197.9 | 77x |
| **(b)** group-by 200 keys, `sum` + `count`, order by total desc limit 10 | **ArrowMetal lazy** | **15.11** | 4.2 | — |
| | polars lazy | 43.04 | 211.6 | 2.8x |
| | duckdb | 702.05 | 992.0 | 46x |
| **(c)** group-by `(region, sub)`, 10 000 groups, `mean` + `max` | **ArrowMetal lazy** | **129.01** | 7.4 | — |
| | polars lazy | 256.76 | 2464.0 | 2.0x |
| | duckdb | 583.10 | 1164.9 | 4.5x |
| **(d)** `(amount*2 + qty) / (region + 1) - qty`, then filter | **ArrowMetal lazy (1 fused kernel)** | **4.82** | 0.8 | — |
| | polars lazy | 210.59 | 210.5 | 44x |
| | duckdb | 351.14 | 353.2 | 73x |
| **(e)** inner join 10M x 1M on int64, then `sum` | **ArrowMetal lazy** | **6.44** | 1.2 | — |
| | polars lazy | 30.95 | 234.7 | 4.8x |
| | duckdb | 191.97 | 361.2 | 30x |
| **(f)** semi join 10M against a 1 000-row key set | **ArrowMetal lazy** | **3.35** | 1.3 | — |
| | polars lazy | 5.74 | 42.4 | 1.7x |
| | duckdb | 7.92 | 9.7 | 2.4x |
| **(g)** `row_number() over (partition by g order by v)`, 10M rows, 1 000 partitions | **ArrowMetal lazy** | **66.96** | 10.5 | — |
| | polars lazy | 139.89 | 741.3 | 2.1x |
| | duckdb | 632.91 | 2301.1 | 9.5x |
| **(h)** as-of join 50M trades against 1M quotes | **ArrowMetal lazy** | **16.53** | 3.4 | — |
| | polars | 195.14 | 194.3 | 12x |
| | duckdb | did not finish | | |

Reading it: the widest margins are where the operator count is high relative to the bytes moved — (d)
is six operators over four columns and 44x, (a) is a filter plus a reduction and 6.5x — and where the
work is a scan the GPU does in one pass, like (h)'s binary search. The narrowest are (f), which is
bound by writing 10M output rows, and (c), where 10 000 groups times two aggregates is atomics bound.

The CPU column is the other half of the story: ArrowMetal's queries cost the machine 1 to 10 ms of CPU
where Polars spends 26 to 2464 ms and DuckDB 10 to 2301 ms across 16 threads. A query that costs 7 CPU-ms
instead of 2.5 CPU-seconds leaves the cores free for whatever else the process is doing.

Below about 1M rows the fixed cost dominates and the GPU is the wrong tool; at 2M rows on the same
machine (a) is 0.59 ms against Polars' 0.81 ms and (f) is 3.58 ms against Polars' 1.15 ms. The crossover
for these shapes is between 1M and 5M rows.


## Limits

- **Strings are read, not written.** The fused kernels will not materialise a `utf8` output column, so
  a projection that computes one is rejected; a string column can be carried through a plan, filtered,
  sorted by (through `GroupByKeys`), joined on and grouped by, but not transformed inside a query.
  Use the per-operator string kernels for that.
- **Temporal, decimal, list, struct, map, union and dictionary columns** can be carried, filtered,
  joined on, grouped by and sorted, but not read by a fused expression. A predicate over one of them
  is an error naming the column and its Arrow format.
- **`min` and `max` in a whole-table aggregate come back at the input's width** but through a 64-bit
  scalar, so a `uint64` maximum above `Int64.max` is not representable.
- **Group-by aggregate types.** The one-kernel group-by path is bounded by the 32-bit atomics MSL
  offers (`sum`/`mean` over any integer or `float32`, `min`/`max` over a 32-bit-or-narrower integer or
  `float32`). Anything else — a `float64` sum, a 64-bit `min` — silently takes `GroupBy`'s own
  per-aggregate kernels instead, which `explain()` says.
- **Group order is not Polars'.** `GroupByKeys` emits groups ascending by key for numeric, boolean,
  temporal and decimal keys and in first-seen order for `utf8`; pyarrow and Polars use first-seen for
  everything. Sort both sides before comparing.
- **`join_asof` keys must be integer or temporal.** Float and string as-of keys are not supported.
- **`explode` takes one list column at a time**, and builds its gather index on the CPU, so it is a
  sync point.
- **`cum_sum` and the rolling window functions are capped at 8192 partitions**, because they run one
  dispatch per partition.
- **Concatenation is a `memcpy`,** not a kernel: the buffers are unified memory, so appending is a
  memcpy at ~100 GB/s with no transfer, and concatenation is always a materialising boundary anyway.
- **No SQL front end, no `LIMIT` push into a join, no spilling.** Every input must fit in the working
  set; there is no partitioned or out-of-core execution.
- Arrays above 2^31 rows are not supported by the join, and 2^32 elsewhere, as in the rest of the
  package.

## Correctness

- `Tests/ArrowMetalTests/EngineTests.swift` — 34 tests: schema inference and type checking, every
  optimizer rule asserted on `explain()`, every operator against a Swift oracle at 0 / 1 / 33 / 4097 /
  1 000 003 rows with nulls, all six join kinds against a Swift oracle over an integer key at four
  sizes with null keys on both sides, multi-column and `utf8` keys, the as-of join in all three
  strategies with partitions and a tolerance and against an oracle at 4097 × 1301 rows, the window
  functions, `concat`, `explode`, batched execution, and optimized-vs-unoptimized equivalence.
- `python/tests/test_lazy.py` — 48 queries against Polars lazy, pyarrow and DuckDB on the same data,
  including TPC-H shapes (filter + group-by + sort + limit; join + aggregate; window over partition;
  as-of join on timestamps).

```
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --filter EngineTests
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test -c release --filter EngineTests
swift build -c release --product ArrowMetalC
PYTHONPATH=python python -m pytest python/tests/test_lazy.py -q
PYTHONPATH=python python Benchmarks/engine_bench.py
```
