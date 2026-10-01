# ArrowMetal for DataFusion

[Apache DataFusion](https://datafusion.apache.org) is a query engine in Rust: SQL and DataFrames over
Arrow, with its own planner, optimizer and multi-threaded operators. `datafusion-arrowmetal` (in
[`datafusion/`](../datafusion)) is a **physical optimizer rule for DataFusion 55.1**. Registered on a
`SessionContext`, it replaces DataFusion's full sort (an `ORDER BY` without `LIMIT`) with `MetalExec`,
which runs the sort on the Apple GPU through ArrowMetal's plan runner and hands DataFusion the
`RecordBatch`es it expects. It also replaces the hash aggregates (`GROUP BY`, `DISTINCT`) of the
shapes a measured table takes: `count(*)` over two int32 keys of a `MemTable` of at least
10,000,000 rows and `DISTINCT` over two int32 keys of one of at least 50,000,000 rows. Such an
aggregate estimates its number of groups from a sample of its keys when it runs, runs on the GPU at
the numbers of groups the table takes at that row count, and otherwise hands the node back to
DataFusion's own operators. The SQL does not change, and every
other node of the plan stays DataFusion's.

**The summary.** On an Apple M4 Max with DataFusion's default of one partition per core, a full sort
of 250,000 to 50,000,000 rows is **6.9x to 28.8x faster** with the rule than DataFusion alone, and at
50M rows it uses 96 to 126 CPU-ms where DataFusion uses 6,104 to 8,311. Over DataFusion's own Parquet
reader, a Float64 sort is 10.8x to 16.5x faster at 10M and 50M rows, and a sort over a Parquet file
with a string column 3.6x to 5.5x at 1M to 50M rows. The aggregates the default runs on the GPU,
`count(*)` over two int32 keys at 10M and 50M rows and `DISTINCT` over two int32 keys at 50M rows,
were 2.14x to 4.11x faster than DataFusion alone warm. On the first run after 500 ms of idle they
were 1.34x to 2.08x faster than DataFusion alone's first run after the same idle, and after 5 s of
idle 1.08x to 1.72x; against DataFusion alone's warm time, that first run was 0.67x to 1.60x after
500 ms and 0.53x to 1.23x after 5 s ([Aggregates](#aggregates-with-the-default)). The answers are
DataFusion's: a differential grid of 7,656 query pairs runs every query with and without the rule
and finds 0 mismatches. Top-k (`ORDER BY … LIMIT`), filters and every other aggregate shape are
left to DataFusion by default; the measured numbers for them are under [To improve](#to-improve).

- [Install](#install)
- [Use](#use)
- [What the default takes](#what-the-default-takes)
- [What it leaves, and why](#what-it-leaves-and-why)
- [DataFusion's semantics](#datafusions-semantics)
- [The differential grid](#the-differential-grid)
- [Numbers](#numbers)
- [To improve](#to-improve)
- [Limits](#limits)
- [Tests and files](#tests-and-files)

---

## Install

The crate needs an Apple silicon Mac and the ArrowMetal dylib:

```bash
# 1. The GPU library, from the repository root.
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
    swift build -c release --product ArrowMetalC

# 2. The example, against that build.
cd datafusion
ARROWMETAL_LIB=$PWD/../.build/release/libArrowMetalC.dylib cargo run --example quickstart
```

The crate is not on crates.io; depend on it by path from a checkout:

```toml
# Cargo.toml
[dependencies]
datafusion = { version = "=55.1.0", default-features = false, features = ["sql"] }
datafusion-arrowmetal = { path = "../ArrowMetal/datafusion" }
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

It depends on the [`arrowmetal`](RUST.md) crate, which finds `libArrowMetalC.dylib` through
`ARROWMETAL_LIB` and the other locations listed in [RUST.md, "Finding the dylib"](RUST.md#finding-the-dylib).
A binary that uses the rule needs the dylib's directory as an rpath entry, set the way that section
shows (a `build.rs` or `RUSTFLAGS`); the crate's own `build.rs` does it for its tests and examples.

**Versions.** DataFusion is pinned to `=55.1.0`, the release the grid and the benchmark ran on
(arrow-rs 59.3.0, the same arrow-rs major as the `arrowmetal` crate, so batches pass between the two
with no conversion). DataFusion 55.1.0 declares `rust-version` 1.94; the crate was built with rustc
1.95.0.

---

## Use

```rust
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, Int64Array, RecordBatch};
use datafusion::datasource::MemTable;
use datafusion::error::Result;
use datafusion::physical_plan::{collect, displayable};
use datafusion::prelude::SessionConfig;
use datafusion_arrowmetal::{session_context, ArrowMetalConfig, ArrowMetalRule};

#[tokio::main]
async fn main() -> Result<()> {
    let rule = ArrowMetalRule::new(ArrowMetalConfig::default()); // full sorts from 250,000 rows
    let ctx = session_context(SessionConfig::new(), rule.clone()); // DataFusion's defaults + the rule

    let v: Int64Array = (0..1_000_000i64).map(|i| (i * 7919) % 1_000_003).collect();
    let batch = RecordBatch::try_from_iter([("v", Arc::new(v) as ArrayRef)])?;
    ctx.register_table("t", Arc::new(MemTable::try_new(batch.schema(), vec![vec![batch]])?))?;

    // EXPLAIN shows MetalExec in the physical plan.
    ctx.sql("EXPLAIN SELECT v FROM t ORDER BY v DESC").await?.show().await?;

    // Or plan once, print the plan and the rule's report, and run that plan.
    rule.clear_report();
    let plan = ctx.sql("SELECT v FROM t ORDER BY v DESC").await?.create_physical_plan().await?;
    println!("{}", displayable(plan.as_ref()).indent(false));
    println!("{}", rule.report()); // TAKEN / LEFT, with the reason
    let batches = collect(plan, ctx.task_ctx()).await?;
    println!("{} rows", batches.iter().map(|b| b.num_rows()).sum::<usize>());
    Ok(())
}
```

The `EXPLAIN` it prints:

```text
+---------------+-----------------------------------------------------+
| plan_type     | plan                                                |
+---------------+-----------------------------------------------------+
| logical_plan  | Sort: t.v DESC NULLS FIRST                          |
|               |   TableScan: t projection=[v]                       |
| physical_plan | MetalExec: sort=[v DESC NULLS FIRST]                |
|               |   DataSourceExec: partitions=1, partition_sizes=[1] |
|               |                                                     |
+---------------+-----------------------------------------------------+
```

Three ways to register it, all in [`src/lib.rs`](../datafusion/src/lib.rs):

| Function | Gives |
|---|---|
| `session_context(config, rule)` | a `SessionContext` with DataFusion's default features and optimizer rules plus `rule` |
| `with_arrowmetal(builder, rule)` | the same on a `SessionStateBuilder` you are already configuring |
| `physical_optimizer_rules(rule)` | DataFusion's default physical optimizer rules with `rule` inserted, for `with_physical_optimizer_rules` |

The rule goes in before DataFusion's last two physical rules, the post-optimization `FilterPushdown`
and `SanityCheckPlan`, so the rewritten plan is still checked for distribution and ordering.
After a run, `MetalExec`'s metrics (`ExecutionPlan::metrics`) hold `input_time`, `import_time`,
`kernel_time`, `export_time` and `input_batches`, and for an aggregate `probe_time` (the group-count
estimate), `groups_estimate` and `handed_back`; the benchmark reads its time split from them.

[`examples/quickstart.rs`](../datafusion/examples/quickstart.rs) registers a 1,000,000-row
`MemTable` (an int64, a nullable Float64 and a string column) with the default configuration and 4
partitions, and runs three queries. Its output, from `cargo run --example quickstart` on an M4 Max:

```text
== SELECT name, region, amount FROM sales ORDER BY amount DESC NULLS LAST, name

ProjectionExec: expr=[name@2 as name, region@0 as region, amount@1 as amount]
  MetalExec: sort=[amount DESC NULLS LAST, name ASC NULLS LAST]
    DataSourceExec: partitions=1, partition_sizes=[1]

TAKEN    SortExec: expr=[amount@1 DESC NULLS LAST, name@2 ASC NULLS LAST], preserve_partitioning=[false] -- input rows 1000000 (exact) vs min_rows 250000

1000000 rows; the first 3:
+---------+--------+--------------------+
| name    | region | amount             |
+---------+--------+--------------------+
| n001395 | 0      | 1429.4285714285713 |
| n005102 | 5      | 1429.4285714285713 |
| n015952 | 4      | 1429.4285714285713 |
+---------+--------+--------------------+

== SELECT name, amount FROM sales ORDER BY amount DESC NULLS LAST LIMIT 3

ProjectionExec: expr=[name@1 as name, amount@0 as amount]
  SortExec: TopK(fetch=3), expr=[amount@0 DESC NULLS LAST], preserve_partitioning=[false]
    DataSourceExec: partitions=1, partition_sizes=[1]

LEFT     SortExec: TopK(fetch=3), expr=[amount@0 DESC NULLS LAST], preserve_partitioning=[false] -- top-k (sort with fetch 3) disabled in config

3 rows; the first 3:
+---------+--------------------+
| name    | amount             |
+---------+--------------------+
| n300762 | 1429.4285714285713 |
| n421196 | 1429.4285714285713 |
| n110979 | 1429.4285714285713 |
+---------+--------------------+

== SELECT region, count(*) AS n, avg(amount) AS mean FROM sales GROUP BY region ORDER BY region

SortPreservingMergeExec: [region@0 ASC NULLS LAST]
  ProjectionExec: expr=[region@0 as region, count(Int64(1))@1 as n, avg(sales.amount)@2 as mean]
    SortExec: expr=[region@0 ASC NULLS LAST], preserve_partitioning=[true]
      AggregateExec: mode=FinalPartitioned, gby=[region@0 as region], aggr=[count(Int64(1)), avg(sales.amount)]
        RepartitionExec: partitioning=Hash([region@0], 4), input_partitions=4
          AggregateExec: mode=Partial, gby=[region@0 as region], aggr=[count(Int64(1)), avg(sales.amount)]
            RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=1
              DataSourceExec: partitions=1, partition_sizes=[1]

LEFT     SortPreservingMergeExec: [region@0 ASC NULLS LAST] -- input rows ~1000000 are an estimate (accept_inexact is off)
LEFT     SortExec: expr=[region@0 ASC NULLS LAST], preserve_partitioning=[true] -- per-partition sort (preserve_partitioning) with no replaced merge above it
LEFT     AggregateExec: mode=FinalPartitioned, gby=[region@0 as region], aggr=[count(Int64(1)), avg(sales.amount)] -- input rows 1000000 (exact) vs min_rows 250000; count + sum_avg_f64 over 1 i64 key (memory input): the measured table takes it at no group count and size (results/datafusion_groupby_sweep_2026-09-30.csv)
LEFT     AggregateExec: mode=Partial, gby=[region@0 as region], aggr=[count(Int64(1)), avg(sales.amount)] -- Partial aggregate whose Final was not replaced

13 rows; the first 3:
+--------+-------+-------------------+
| region | n     | mean              |
+--------+-------+-------------------+
| 0      | 76924 | 714.7095888611585 |
| 1      | 76923 | 714.7337580454488 |
| 2      | 76923 | 714.739149199677  |
+--------+-------+-------------------+
```

The report (`rule.report()`, a `Report` of `Decision`s) has one line per node the rule looked at:
`TAKEN` or `LEFT` with the reason; for a replaced aggregate, one line per run, `GPU` or `HANDBACK`,
with the estimated number of groups, its range, the sample it came from and the table row that
decided (`Decision::groups` holds the estimate); and `FALLBACK` for a `MetalExec` that handed its
node back to DataFusion on an ArrowMetal error, on data only DataFusion answers exactly, or when the
memory pool refused its reservation. Clones of a rule share one report. It keeps the decisions of
the last `report_plans` plans the rule optimized (default 64; an `EXPLAIN` and each executed plan
count one each), until `clear_report()` empties it.

---

## What the default takes

`ArrowMetalConfig::default()`:

| Field | Default | Meaning |
|---|---|---|
| `sort` | **on** | full sorts: `ORDER BY` without `LIMIT` |
| `min_rows` | **250,000** | take a node only when its input has at least this many rows |
| `topk` | off | `ORDER BY … LIMIT` |
| `aggregate` | **on** | hash `GROUP BY` and `DISTINCT` of the shapes the measured table takes (below) |
| `filter` | off | `WHERE` |
| `accept_inexact` | off | use an estimated row count as if it were exact |
| `take_when_unknown` | off | take a node whose input row count is unknown |
| `aggregate_choice` | `Measured` | who runs a replaced aggregate: `Measured` (the group-count estimate and the measured table), `ArrowMetal` (always the GPU), `DataFusion` (always handed back) |
| `table_rows` | none | look the measured table up at this row count instead of the input's |
| `report_plans` | 64 | how many plans the report keeps |

`ArrowMetalConfig::all()` switches `topk` and `filter` on as well. The fields are public to read; a
config is built from `default()` or `all()` with the `with_*` setters, e.g.
`ArrowMetalConfig::all().with_min_rows(0)`, which the grid and the benchmark use.

A full sort is taken when:

- every sort key is a column (not an expression) of type int8 to int64, uint8 to uint64, Float32,
  Float64, Utf8, LargeUtf8 or Utf8View;
- every column the sort carries is one of those types or Boolean;
- DataFusion's statistics give the sort's input an **exact** row count of at least `min_rows`
  (`MemTable`s and DataFusion's Parquet scan gave exact counts in the tests and the benchmark);
- the plan shape is `SortExec` (one partition, or `preserve_partitioning = false`),
  `SortPreservingMergeExec` over `SortExec(preserve_partitioning)` with the same ordering, or
  `SortPreservingMergeExec` → `ProjectionExec` → `SortExec`, which DataFusion plans when the
  `SELECT` list reorders or computes columns; the projection stays above `MetalExec`.

`MetalExec` emits one partition. When the replaced node had more and something above it requires a
distribution, the rule adds a `RepartitionExec` with the replaced node's partitioning; with only
projections above it, the output stays at one partition.

**Why 250,000 rows.** In `datafusion_rule_2026-09-29.csv` every key type is ahead of DataFusion at
100,000 rows (1.35x to 2.63x, both layouts), and 250,000 rows is the first measured size at which
every key type is at 2.64x or more. The file measured below with a warm-up
(`datafusion_sort_warm_2026-09-29.csv`) puts every key type at 6.9x or more at 250,000 rows.

**Aggregates.** The rule sees an aggregate's row count but not its number of groups, and the same
aggregate is ahead of DataFusion at some group counts and behind at others. So an aggregate is taken
in two steps:

- At plan time the rule describes it by its shape: the aggregate families (`count`, `sum`/`avg`
  over an integer column, `sum`/`avg` over Float64, `min`/`max` over an integer column, `min`/`max`
  over Float64, and `DISTINCT` for a `GROUP BY` with no aggregate function), one key or several, the
  key type class (up to 32-bit integers, 64-bit integers, floats, strings) and whether it reads a
  `MemTable` scan directly. It replaces the node with a `MetalExec` only when the measured table
  ([`src/agg_table.rs`](../datafusion/src/agg_table.rs)) takes that shape at some number of groups
  at the input's exact row count.
- When it runs, `MetalExec` reads the first batches of each input partition (262,144 rows in all),
  estimates the number of groups from a stratified sample of their keys (the bias-corrected Chao1
  estimator over 512 to 8,192 sampled rows), and looks the shape up at that number of groups. If the
  table takes it, the rest of the input is collected, a 2,048-row sample of the whole input must
  agree with the estimate, and the aggregate runs on the GPU. Otherwise the batches read so far and
  the rest of each partition stream into DataFusion's own partial and final aggregates, over the
  same partitions as without the rule.

The table is generated by [`scripts/groupby_table.py`](../datafusion/scripts/groupby_table.py) from
the measurements it names:

1. The sweep (`datafusion_groupby_sweep_2026-09-30.csv`: every family over one and two int32 and
   int64 keys, at 200 to about rows / 2 groups, 1M to 50M rows, both table layouts, the rule forced
   onto the GPU against DataFusion alone; `sum`/`avg` over Float64 and `min`/`max` over Float64 and
   over int64 measured on 2026-09-30, after the grouped Float64 sum and mean and the grouped min/max
   were rebuilt, the other families on 2026-09-29). A series (shape and number of groups) qualifies from the
   smallest measured row count at which its worst case (over the layouts and the family's queries)
   was at least 1.65x faster than DataFusion alone, at that size and every larger one, at two sizes
   or more. 34 of the 120 measured series qualify, from 2M to 10M rows.
2. Idle against idle, after 500 ms and after 5 s: a series is taken at a size only if, at that
   size and every larger one, the default's first run after 500 ms of idle and its first run after
   5 s of idle (the Metal pipelines already compiled) were each at least as fast as DataFusion
   alone's first run after the same idle, for the same query, size and layout, in every case (the
   sweep's queries and `SELECT DISTINCT region, sub`) and layout (the median of three runs of each, contexts and gaps alternating;
   `datafusion_groupby_idle_gaps_2026-09-29.csv` and `datafusion_groupby_resweep_check_2026-10-01.csv`,
   which measure both gaps in one session, and the earlier files). A case with no measurement at one
   of the gaps counts as below.
3. The hand-back: a shape is replaced at a size only if, at that size and every larger one, every
   case the default handed back at run time was at 0.97x of DataFusion alone or better on the best
   run and on the median run (`datafusion_groupby_resweep_check_2026-10-01.csv`,
   `datafusion_groupby_idle_gaps_recheck_2026-09-29.csv`,
   `datafusion_groupby_idle_vs_idle_2026-09-29.csv` and the files before them).

Ten series are taken, all over integer keys of a `MemTable` scan (an int32 class includes narrower keys):

| aggregate | keys | groups | taken from | sweep, worst case 10M / 50M |
|---|---|---|---|---|
| `count` (`count(*)`, `count(x)`) | two int32 | 31,623 to 316,227 | 10,000,000 rows | 2.7x / 3.2x |
| `count` | two int32 | 316,228 to 3,162,277 | 10,000,000 rows | 2.8x / 4.0x |
| `DISTINCT` | two int32 | 1 to 1,414 | 50,000,000 rows | 1.7x / 2.2x |
| `DISTINCT` | two int32 | 31,623 to 316,227 | 50,000,000 rows | 2.6x / 3.0x |
| `DISTINCT` | two int32 | 316,228 to 3,162,277 | 50,000,000 rows | 2.1x / 3.4x |
| `min`, `max` over an integer column | one int64 | 316,228 to 3,162,277 | 50,000,000 rows | 2.6x / 3.1x |
| `min`, `max` over an integer column | two int32 | 31,623 to 316,227 | 50,000,000 rows | 2.4x / 2.6x |
| `min`, `max` over an integer column | two int32 | 316,228 to 3,162,277 | 50,000,000 rows | 2.7x / 3.4x |
| `min`, `max` over an integer column | two, at least one int64 | 31,623 to 316,227 | 50,000,000 rows | 2.6x / 2.8x |
| `min`, `max` over an integer column | two, at least one int64 | 316,228 to 3,162,277 | 50,000,000 rows | 2.9x / 3.4x |

Each series' row in `src/agg_table.rs` ends with the reason for its threshold. Of the other 24
series that qualify in the sweep, 21 are left by their first run after idle at 50M rows (5 slower
than DataFusion alone's first run after the same idle, 16 with no measurement after 5 s of idle
there), and 3 because a hand-back of the same shape at 50M rows cost more than 3%
([To improve](#aggregates)).

---

## What it leaves, and why

| Shape | Default | Why |
|---|---|---|
| top-k, `ORDER BY … LIMIT` | left | behind DataFusion at every measured size from 250,000 to 50M rows: 0.14x to 0.41x ([To improve](#top-k)) |
| `GROUP BY`, `DISTINCT` of a shape the table does not take at the input's row count | left | behind DataFusion, not 1.65x ahead in the worst case at two sizes, slower than DataFusion alone on the first run after the same idle, or its hand-backs cost more than 3% ([To improve](#aggregates)) |
| `count(*)` over two int32 keys at 10M rows or more, or `DISTINCT` over two int32 keys or `min`/`max` over an integer column with one int64 key or two integer keys at 50M rows or more, whose estimated number of groups is not in a taken range | handed back at run time | `HANDBACK` in the report |
| `WHERE` | left | behind on the measured shapes: 0.51x to 0.74x ([To improve](#filters)) |
| joins | not replaced | the rule has no join operator |
| a sort over an estimated row count (above a filter, a join or an aggregate) | left | the threshold needs an exact count; `accept_inexact` takes it |
| a sort key that is an expression, or a key or carried column of another type (Date, Timestamp, Decimal, Dictionary, nested) | left | not in the types the grid covers |
| a per-partition sort with no merge above it, a merge whose ordering differs from its sort's | left | one sorted partition cannot stand in for several |

Each of these except joins is a `LEFT` line in the report with its reason; the rule does not report
the nodes it has no operator for.

---

## DataFusion's semantics

Each `ORDER BY` key becomes one key of ArrowMetal's sort, with two options that give DataFusion's
order in the sort itself, with no extra key or pass:

- **Nulls** first or last, as the query says, in either direction (`NULLS FIRST`, `NULLS LAST`, or
  DataFusion's default for the direction).
- **Floats** in IEEE 754 totalOrder, the order arrow-rs and DataFusion use:
  -NaN < -inf < … < -0.0 < +0.0 < … < +inf < +NaN, NaNs ordered by payload, and descending the exact
  mirror. So a NaN comes first in a descending sort and -0.0 sorts before +0.0.
  `tests/arrowmetal_repros.rs` (`sort_float_order_total_is_arrow_rs_order`) checks this bit for bit
  against `arrow::compute::sort_to_indices`, for Float64 and Float32, with -NaN, NaN payloads, ±0.0,
  ±inf, subnormals and nulls, in both directions with both null placements.
- **Strings** (Utf8) in byte order, as arrow-rs compares them; the grid's string keys include `""`,
  `"B"` against `"b"`, and `"ü"`.
- **Ties**: SQL leaves the order of rows with equal keys open. The grid checks the key sequence
  exactly and the rows within each run of equal keys as a multiset.

**Handed back to DataFusion at run time.** `MetalExec` keeps the subtree it replaced. If ArrowMetal
returns an error or the GPU path panics, or the session's memory pool refuses the reservation for
the collected input or the result, `MetalExec` runs that subtree over the batches it collected and
the rest of its input, returns DataFusion's answer, and adds a `FALLBACK` line to the report. No
query in the grid or in the benchmarks took that path; `tests/rule.rs` runs a sort and a group-by
in a 3 MiB memory pool, which hand back. With aggregates and filters taken, three checks on the data
also hand the node back, because on such data only DataFusion can give DataFusion's answer:

- a filter that compares a float column with a literal, when the column holds a NaN with the sign bit
  set (totalOrder puts it below -inf; the GPU comparison cannot place it);
- a `GROUP BY` over a float key that holds NaNs of more than one bit pattern (DataFusion keeps each
  pattern as its own group; ArrowMetal puts every NaN in one group);
- a grouped Float64 `MIN` or `MAX` over a group that holds a NaN, or both -0.0 and +0.0 with a zero
  result (DataFusion's answer there depends on the order the rows arrive in).

---

## The differential grid

`tests/grid.rs` runs every query in a plain `SessionContext` and in one with the rule at
`ArrowMetalConfig::all().with_min_rows(0)`, and compares the answers. Every `GROUP BY` runs three
times with the rule: forced onto the GPU (`AggregateChoice::ArrowMetal`), forced back to DataFusion
(`AggregateChoice::DataFusion`), and with the measured choice looked up at 50M rows
(`table_rows`).

- **Tables:** 13 columns: Int32 and Int64 keys; Float64 keys with ±0.0, ±inf and NaN; Float64 keys
  with -NaN and NaN payloads of both signs; Float32 keys with NaN, ±0.0 and ±inf; Utf8 keys and the
  same keys as Utf8View and as LargeUtf8; Int32 and Int64 values; Float64 values with NaN, NaN
  payloads and both zeros; Float64 values whose sums round; Float32 values with -NaN and a NaN
  payload.
- **24 table configurations:** 0, 1, 1,000 and 20,000 rows × null fraction 0, 0.1 and 1.0 on every
  column × two layouts (one partition with `target_partitions` 1; three partitions with
  `target_partitions` 4, which plans per-partition sorts under a merge).
- **221 queries:** 154 `ORDER BY` (12 columns × ASC/DESC × `NULLS FIRST`/`NULLS LAST`/default × with
  and without `LIMIT 7`, plus 10 multi-column orderings), 49 `GROUP BY`, and 18 `WHERE`.
- **Comparison:** floats by bit pattern (so -0.0 ≠ +0.0, and NaN sign and payload must match), except
  float `sum` and every `avg`, compared within 1e-9 relative.

Run on 2026-09-29 (`cargo test`, debug build, 88 s):

| | |
|---|---|
| query pairs | **7,656**: 5,304 with every node forced onto the GPU, 1,176 `GROUP BY` forced back, 1,176 `GROUP BY` with the measured choice |
| pairs with a node replaced | 5,976, including all 3,696 `ORDER BY` pairs |
| run-time choices | forced onto the GPU: 960 on the GPU; forced back: 960 handed back; measured: none replaced (every `GROUP BY` of the grid has one key; the table takes two-key shapes only) |
| mismatches | **0** |
| hand-backs on an ArrowMetal error | **0** |
| hand-backs on the data (the checks above) | 128: 64 Float64 `MIN` over a group with NaN, 32 filters on a column holding -NaN, 32 group keys with several NaN patterns |
| largest relative deviation in a float `sum`/`avg` | 2.8e-13 |

The pairs of the forced variant with nothing taken are Float32 `MIN`/`MAX` group-bys, a `CAST` in a
predicate, and a float comparison against zero, each left by the rule with its reason.
`tests/rule.rs` checks both branches of the measured choice with the table looked up at 10M and at
50M rows: `count(*)` over two int32 keys drawn from 1,000,000 values runs on the GPU, over 150 values
it is handed back, on one batch per partition and on 8,192-row batches. A unit test in `src/exec.rs`
checks that a prefix repeating a few thousand keys, followed by keys seen once, is not confirmed by
the 2,048-row sample of the whole input.

---

## Numbers

**Machine and method.** Apple M4 Max (16 CPU cores: 12 performance, 4 efficiency), 64 GB, macOS 27.0;
rustc 1.95.0, DataFusion 55.1.0, arrow-rs 59.3.0; the benchmark is
[`examples/bench.rs`](../datafusion/examples/bench.rs), built `--release` with thin LTO.

- `SessionConfig::new()`: 16 partitions, 8,192-row batches.
- Each query runs in a context without the rule (**off**) and in one with
  `ArrowMetalConfig::all().with_min_rows(0).with_accept_inexact(true)` (**on**); both answers are
  compared before anything is timed. A third context with `ArrowMetalConfig::default()` is only
  planned, and its decisions are recorded in the CSV's `rule_default` column. The aggregate files
  below time the default configuration itself (**def**).
- Timed: SQL to logical plan, physical planning (the rule runs there) and `collect`. Best of 5
  wall-clock runs. CPU-ms is the process CPU time (user + system, all threads) of the best run; GPU
  time is not in it.
- `datafusion_sort_warm_2026-09-29.csv`: before the timed runs of each context, untimed runs repeat
  until they add up to 100 ms (the GPU runs small plans about three times slower after it idles and
  reaches its fast state after about 20-25 ms of back-to-back work). The file also records
  `on_first_ms`, the first run with the rule, taken with the GPU idle before it. Three passes at
  250,000 to 2M rows, of which the tables show the median, each column on its own; one pass at 10M
  and 50M.
- Tables: `MemTable`s with 16 partitions, of 8,192-row batches dealt round-robin (**b8192**, the layout
  DataFusion's scans produce) or one batch per partition (**part1**). Queries:
  `SELECT q, k1, x FROM extra ORDER BY q` (int64 key), `… ORDER BY x` (Float64), `SELECT q, k1, y …
  ORDER BY y` (Float32), `SELECT name, q, k1 FROM extra ORDER BY name` (String, 1,000 values); keys
  drawn uniformly (int64 in [0, 10^9), floats in [0, 1)).
- Before each block of queries, the harness waited for a 1-minute load average below 3.5 and no
  compiler or test process; the load at block start and end (1.49 to 3.39 in this file) is in every
  row.

Every timed row gave the same answer with the rule as without it, with 0 hand-backs.

### Full sorts, rule off and on

8,192-row batches, from `datafusion/results/datafusion_sort_warm_2026-09-29.csv` (rows `crate = now`):

| key | rows | rule off, ms | rule on, ms | off ÷ on | CPU-ms off / on | first run on, idle GPU, ms |
|---|---:|---:|---:|---:|---:|---:|
| int64 | 250,000 | 8.08 | 0.98 | 8.2x | 16.0 / 0.9 | 31.86* |
| int64 | 500,000 | 15.66 | 1.40 | 11.2x | 34.1 / 1.5 | 31.67* |
| int64 | 1,000,000 | 31.75 | 1.77 | 17.9x | 74.2 / 2.7 | 34.32* |
| int64 | 2,000,000 | 62.49 | 2.58 | 24.2x | 159.4 / 4.9 | 39.54* |
| int64 | 10,000,000 | 312.87 | 13.31 | 23.5x | 974.3 / 25.1 | 68.53* |
| int64 | 50,000,000 | 1,581.22 | 69.74 | 22.7x | 6,104.1 / 125.8 | 176.36* |
| Float64 | 250,000 | 8.43 | 1.11 | 7.6x | 16.8 / 1.0 | 1.95 |
| Float64 | 500,000 | 16.81 | 1.39 | 12.1x | 37.5 / 1.5 | 4.44 |
| Float64 | 1,000,000 | 33.76 | 2.02 | 16.7x | 78.9 / 2.6 | 6.00 |
| Float64 | 2,000,000 | 65.97 | 3.25 | 20.3x | 169.2 / 5.3 | 9.66 |
| Float64 | 10,000,000 | 331.75 | 16.88 | 19.7x | 1,032.6 / 23.5 | 24.06 |
| Float64 | 50,000,000 | 1,672.57 | 87.90 | 19.0x | 6,406.7 / 119.5 | 127.94 |
| Float32 | 250,000 | 8.64 | 0.98 | 8.8x | 16.4 / 0.9 | 2.75 |
| Float32 | 500,000 | 16.75 | 1.35 | 12.4x | 36.5 / 1.4 | 3.83 |
| Float32 | 1,000,000 | 33.47 | 1.61 | 20.8x | 76.5 / 2.3 | 4.70 |
| Float32 | 2,000,000 | 65.55 | 2.51 | 26.1x | 165.7 / 4.3 | 6.93 |
| Float32 | 10,000,000 | 329.33 | 11.63 | 28.3x | 1,015.3 / 20.4 | 21.17 |
| Float32 | 50,000,000 | 1,625.59 | 61.14 | 26.6x | 6,227.0 / 108.2 | 107.53 |
| String | 250,000 | 10.69 | 1.42 | 7.5x | 27.2 / 1.3 | 3.52 |
| String | 500,000 | 19.34 | 1.93 | 10.0x | 51.1 / 2.1 | 3.63 |
| String | 1,000,000 | 38.86 | 2.73 | 14.2x | 107.1 / 3.1 | 7.53 |
| String | 2,000,000 | 77.33 | 3.37 | 22.9x | 226.0 / 5.5 | 9.99 |
| String | 10,000,000 | 375.98 | 16.86 | 22.3x | 1,340.7 / 23.6 | 27.51 |
| String | 50,000,000 | 1,861.23 | 87.62 | 21.2x | 8,310.5 / 119.5 | 129.86 |

\* The int64 sort in this layout was the first query each benchmark process ran, so its first run
also compiled the GPU pipelines. The same query in the one-batch layout, run next in the same
process, took 3.36, 3.44, 4.29, 7.27, 22.25 and 116.46 ms on its first run at the six sizes.

One batch per partition, off ÷ on, same file:

| key | 250,000 | 500,000 | 1,000,000 | 2,000,000 | 10,000,000 | 50,000,000 |
|---|---:|---:|---:|---:|---:|---:|
| int64 | 7.8x | 11.2x | 16.0x | 24.0x | 23.4x | 22.8x |
| Float64 | 7.7x | 11.1x | 16.5x | 19.9x | 19.9x | 19.1x |
| Float32 | 9.2x | 12.3x | 19.6x | 27.1x | 28.8x | 26.9x |
| String | 6.9x | 9.4x | 15.2x | 21.8x | 22.3x | 21.4x |

**The first run from an idle GPU.** A single `ORDER BY` after the GPU has idled takes longer than the
warm figure: 1.95 against 1.11 ms for the Float64 sort at 250,000 rows, 127.94 against 87.90 ms at
50M. Every first-run figure in the 8,192-row table except the starred ones, and every one in the
one-batch layout, is below DataFusion's warm time for the same query.

### Over DataFusion's Parquet reader

`SELECT id, price FROM f ORDER BY price` (a Float64 key), with DataFusion's own Parquet scan feeding
both runs, from `datafusion/results/datafusion_rule_2026-09-29.csv` (rows `crate = now`, case
`p_sort`). The files come from `Benchmarks/parquet_bench.py build`.

| codec | rows | rule off, ms | rule on, ms | off ÷ on | CPU-ms off / on | input / import / sort, ms |
|---|---:|---:|---:|---:|---:|---|
| snappy | 10,000,000 | 289.63 | 21.35 | 13.6x | 921.5 / 83.7 | 7.11 / 1.72 / 11.68 |
| zstd | 10,000,000 | 294.26 | 27.14 | 10.8x | 964.2 / 135.8 | 12.82 / 1.69 / 11.86 |
| snappy | 50,000,000 | 1,643.01 | 99.43 | 16.5x | 5,477.1 / 448.8 | 30.49 / 8.77 / 58.26 |
| zstd | 50,000,000 | 1,659.64 | 120.45 | 13.8x | 5,729.9 / 719.6 | 52.03 / 8.51 / 58.11 |

The default configuration takes this sort: the scan reports an exact row count (the `rule_default`
column reads `TAKEN SortPreservingMergeExec (input rows 50000000 (exact) vs min_rows 250000)`). This
file was measured without the warm-up; in it, the in-memory sorts at 10M and 50M rows are within 13%
of the warm-up file's figures, rule on. With the rule, `input_time` includes the Parquet decode.

### A Parquet file with a string column

DataFusion 55.1 reads Parquet strings as Utf8View, which the rule carries and sorts by. A file the
benchmark writes (the columns above, 1,048,576-row row groups), off and on, two rounds of best of 5
(best of 2 above 2 s), 8,192-row batches, from `datafusion/results/datafusion_parquet_string_sort_2026-09-29.csv`:

| query | rows | rule off, ms | rule on, ms | off ÷ on | CPU-ms off / on |
|---|---:|---:|---:|---:|---:|
| `SELECT name, q, k1 FROM f ORDER BY name` | 1,000,000 | 32.64 | 9.04 | 3.6x | 67.5 / 11.5 |
| | 10,000,000 | 201.05 | 50.96 | 3.9x | 786.9 / 107.5 |
| | 50,000,000 | 1,030.14 | 252.93 | 4.1x | 4,351.5 / 546.8 |
| `SELECT q, name, k1 FROM f ORDER BY q` | 1,000,000 | 45.30 | 9.27 | 4.9x | 70.3 / 12.4 |
| | 10,000,000 | 278.41 | 57.67 | 4.8x | 961.7 / 115.5 |
| | 50,000,000 | 1,590.83 | 288.79 | 5.5x | 5,911.6 / 568.1 |

The default configuration takes both (exact row counts from the file's footer).

### Aggregates with the default

Every `GROUP BY` and `DISTINCT` case of the sweep (with one and two int32 and int64 keys, 200 to about
rows / 2 groups) and `SELECT DISTINCT region, sub FROM fact` (two int32 keys, 10,000 groups), with
`ArrowMetalConfig::default()` timed against DataFusion alone, both layouts, 1M to 50M rows. A case
whose default ran on the GPU: three rounds of best of 5 with the two contexts in rotated order, then
three runs of each after 500 ms of sleep and three after 5 s, the contexts and the gaps alternating
(the median of the three is the first-run time). Any other case (both contexts on the CPU): 30 runs
of each, alternating run by run, after a 100 ms warm-up. The files: `datafusion_groupby_default_2026-09-29.csv`
(every case, an earlier table) and the re-measurements after it, whose rows replace its rows for the
same case, size and layout (`…_taken_…`, `…_cap_…`, `…_idle_…`, `…_handback_…`, `…_recheck_gpu_…`,
`…_recheck_…`, `…_recheck2_…`, `…_idle_vs_idle_…`, `…_idle_vs_idle_recheck_…`, `…_idle_gap_…`,
`…_idle_gaps_…`, `…_idle_gaps_recheck_…`, `…_resweep_check_2026-10-01…`, `…_resweep_recheck_2026-10-01…`).
`…_idle_vs_idle_…` holds every case whose decision an
earlier version of this table changed and a random fifth of the rest; `…_idle_gaps_…` every case
that version ran on the GPU, at both gaps (22 rows, 19:03 to 19:17 EDT, load 0.94 to 3.14 at the
start of each block); `…_idle_gaps_recheck_…` every case whose decision the final table changed
(18 rows, 19:51 to 21:42 EDT, load 1.50 to 3.44 at the start of each block). After the Float64
`sum`/`avg` and the `min`/`max` families were swept again (2026-09-30), `…_resweep_check_…` holds every
case of those families that a table from that sweep alone replaced (256 rows, 03:34 to 08:34 EDT on
2026-10-01), and `…_resweep_recheck_…` every case whose decision the final table changed (the 30
`min`/`max` cases at 50M rows) and a seeded fifth of the rest (303 rows), 09:09 to 12:55 EDT, load
1.18 to 3.50 at the start of each block; in it the idle runs were taken only for the cases the
default ran on the GPU from its 65th row on. Every answer was equal to DataFusion's.

| rows | cases | left at plan time | handed back at run time | run on the GPU |
|---:|---:|---:|---:|---:|
| 1,000,000 to 5,000,000 | 902 | 902 | 0 | 0 |
| 10,000,000 | 322 | 312 | 6 | 4 |
| 50,000,000 | 322 | 270 | 32 | 20 |

**On the GPU** (`datafusion_groupby_idle_gaps_2026-09-29.csv`; the `min`/`max` rows
`datafusion_groupby_resweep_recheck_2026-10-01.csv`). *Warm*: best
run, DataFusion alone ÷ the default. *Idle vs idle*: DataFusion alone's first run after the idle ÷
the default's first run after the same idle. *GPU idle vs DataFusion warm*: DataFusion alone warm ÷
the default's first run after idle, the comparison of a single query on an idle GPU with
DataFusion's time when it has just run the same query. CPU-ms: process CPU time of the best warm run
and of the median run after 500 ms of idle.

| query | groups | rows | layout | DataFusion warm, ms | default warm, ms | warm | after 500 ms idle, DataFusion / default, ms | idle vs idle, 500 ms | after 5 s idle, DataFusion / default, ms | idle vs idle, 5 s | GPU idle vs DataFusion warm, 500 ms / 5 s | CPU-ms warm, DataFusion / default | CPU-ms after 500 ms idle, DataFusion / default |
|---|---:|---:|---|---:|---:|---:|---|---:|---|---:|---|---|---|
| `count(*)` | 100,000 | 10M | 8,192-row batches | 15.45 | 5.58 | 2.77x | 17.37 / 11.68 | 1.49x | 17.92 / 15.39 | 1.16x | 1.32x / 1.00x | 178 / 11 | 180 / 18 |
| `count(*)` | 100,000 | 10M | one per partition | 14.68 | 5.21 | 2.82x | 23.09 / 17.26 | 1.34x | 29.23 / 17.01 | 1.72x | 0.85x / 0.86x | 184 / 9 | 255 / 33 |
| `count(*)` | 999,943 | 10M | 8,192-row batches | 18.02 | 7.14 | 2.52x | 24.94 / 16.13 | 1.55x | 33.86 / 24.41 | 1.39x | 1.12x / 0.74x | 262 / 13 | 350 / 20 |
| `count(*)` | 999,943 | 10M | one per partition | 17.69 | 6.26 | 2.83x | 36.28 / 26.55 | 1.37x | 35.97 / 33.30 | 1.08x | 0.67x / 0.53x | 259 / 13 | 415 / 38 |
| `count(*)` | 100,000 | 50M | 8,192-row batches | 50.62 | 17.34 | 2.92x | 69.71 / 44.41 | 1.57x | 73.41 / 54.66 | 1.34x | 1.14x / 0.93x | 764 / 50 | 893 / 77 |
| `count(*)` | 100,000 | 50M | one per partition | 53.78 | 16.01 | 3.36x | 65.60 / 41.60 | 1.58x | 70.16 / 53.20 | 1.32x | 1.29x / 1.01x | 764 / 50 | 928 / 74 |
| `count(*)` | 1,000,000 | 50M | 8,192-row batches | 74.10 | 19.33 | 3.83x | 88.38 / 46.35 | 1.91x | 91.96 / 66.49 | 1.38x | 1.60x / 1.11x | 1,112 / 52 | 1,267 / 95 |
| `count(*)` | 1,000,000 | 50M | one per partition | 73.40 | 17.84 | 4.11x | 90.01 / 48.84 | 1.84x | 86.45 / 59.90 | 1.44x | 1.50x / 1.23x | 1,108 / 52 | 1,327 / 81 |
| `DISTINCT` | 200 | 50M | 8,192-row batches | 29.93 | 13.99 | 2.14x | 52.07 / 25.01 | 2.08x | 52.10 / 47.26 | 1.10x | 1.20x / 0.63x | 422 / 51 | 620 / 71 |
| `DISTINCT` | 200 | 50M | one per partition | 29.27 | 12.57 | 2.33x | 52.07 / 28.61 | 1.82x | 50.55 / 43.23 | 1.17x | 1.02x / 0.68x | 416 / 49 | 641 / 73 |
| `DISTINCT` | 100,000 | 50M | 8,192-row batches | 45.47 | 15.94 | 2.85x | 67.92 / 47.54 | 1.43x | 69.07 / 58.81 | 1.17x | 0.96x / 0.77x | 611 / 48 | 837 / 92 |
| `DISTINCT` | 100,000 | 50M | one per partition | 40.91 | 14.78 | 2.77x | 66.47 / 45.58 | 1.46x | 56.78 / 52.09 | 1.09x | 0.90x / 0.79x | 637 / 49 | 869 / 79 |
| `DISTINCT` | 1,000,000 | 50M | 8,192-row batches | 56.75 | 17.10 | 3.32x | 69.38 / 45.13 | 1.54x | 70.74 / 48.76 | 1.45x | 1.26x / 1.16x | 860 / 49 | 1,009 / 97 |
| `DISTINCT` | 1,000,000 | 50M | one per partition | 56.08 | 15.42 | 3.64x | 77.91 / 53.99 | 1.44x | 72.58 / 57.18 | 1.27x | 1.04x / 0.98x | 854 / 45 | 1,090 / 82 |
| `min`, `max` over int64 (one int64 key) | 1,000,000 | 50M | 8,192-row batches | 84.25 | 26.60 | 3.17x | 104.11 / 66.53 | 1.56x | 92.92 / 70.16 | 1.32x | 1.27x / 1.20x | 1,218 / 90 | 1,332 / 138 |
| `min`, `max` over int64 (two int32 keys) | 1,000,000 | 50M | 8,192-row batches | 97.86 | 28.42 | 3.44x | 111.22 / 54.72 | 2.03x | 116.11 / 82.26 | 1.41x | 1.79x / 1.19x | 1,472 / 87 | 1,506 / 123 |
| `min`, `max` over int64 (two int64 keys) | 1,000,000 | 50M | 8,192-row batches | 116.41 | 32.91 | 3.54x | 133.20 / 88.90 | 1.50x | 130.57 / 88.17 | 1.48x | 1.31x / 1.32x | 1,729 / 136 | 1,751 / 296 |
| `min`, `max` over int64 (one int64 key) | 1,000,000 | 50M | one per partition | 82.36 | 24.55 | 3.35x | 99.35 / 72.21 | 1.38x | 101.00 / 78.63 | 1.28x | 1.14x / 1.05x | 1,202 / 85 | 1,374 / 204 |
| `min`, `max` over int64 (two int32 keys) | 1,000,000 | 50M | one per partition | 95.09 | 26.45 | 3.60x | 114.02 / 79.07 | 1.44x | 114.63 / 73.61 | 1.56x | 1.20x / 1.29x | 1,428 / 78 | 1,608 / 226 |
| `min`, `max` over int64 (two int64 keys) | 1,000,000 | 50M | one per partition | 109.61 | 30.71 | 3.57x | 129.35 / 80.68 | 1.60x | 129.57 / 88.57 | 1.46x | 1.36x / 1.24x | 1,628 / 123 | 1,778 / 296 |
| `min`, `max` over int64 (two int32 keys) | 100,000 | 50M | 8,192-row batches | 70.24 | 26.60 | 2.64x | 85.46 / 53.50 | 1.60x | 94.14 / 79.02 | 1.19x | 1.31x / 0.89x | 1,024 / 88 | 1,114 / 137 |
| `min`, `max` over int64 (two int64 keys) | 100,000 | 50M | 8,192-row batches | 83.68 | 30.44 | 2.75x | 100.45 / 75.71 | 1.33x | 98.18 / 84.19 | 1.17x | 1.11x / 0.99x | 1,257 / 123 | 1,298 / 275 |
| `min`, `max` over int64 (two int32 keys) | 100,000 | 50M | one per partition | 66.83 | 25.22 | 2.65x | 89.71 / 49.05 | 1.83x | 90.39 / 77.89 | 1.16x | 1.36x / 0.86x | 1,011 / 82 | 1,218 / 119 |
| `min`, `max` over int64 (two int64 keys) | 100,000 | 50M | one per partition | 81.00 | 29.16 | 2.78x | 98.81 / 82.28 | 1.20x | 98.42 / 83.38 | 1.18x | 0.98x / 0.97x | 1,194 / 130 | 1,398 / 305 |

Per size: at 10M rows the default was 2.52x to 2.83x faster warm, 1.34x to 1.55x idle against idle
after 500 ms and 1.08x to 1.72x after 5 s, and on its first run after idle 0.67x to 1.32x (500 ms)
and 0.53x to 1.00x (5 s) of DataFusion's warm time; at 50M rows 2.14x to 4.11x, 1.43x to 2.08x,
1.09x to 1.45x, 0.90x to 1.60x and 0.63x to 1.23x. DataFusion alone's first run took 1.12x to 2.05x
its warm time after 500 ms and 1.16x to 2.03x after 5 s at 10M rows, 1.19x to 1.78x and 1.18x to
1.74x at 50M; the default's took 2.09x to 4.24x and 2.76x to 5.32x at 10M, 1.79x to 3.50x and
2.85x to 3.71x at 50M. On the median of the three rounds instead of the best run, the lowest warm
ratio is 1.58x at 10M rows (`count(*)`, 999,943 groups, 8,192-row batches; 2.52x best) and 2.12x at
50M. The `min`/`max` rows at 50M: 2.64x to 3.60x warm (2.65x lowest on the median of the rounds),
1.20x to 2.03x idle against idle after 500 ms and 1.16x to 1.56x after 5 s, and 0.98x to 1.79x
(500 ms) and 0.86x to 1.32x (5 s) of DataFusion's warm time.

The idle ratios move between measurements of the same case. For the 22 cases in both
`…_idle_vs_idle_…` and `…_idle_gaps_…`, the later 500 ms idle-against-idle ratio is 0.65x to 1.80x
of the earlier one (median 1.00x); for the 6 cases measured after 5 s in both `…_idle_gap_…` and
`…_idle_gaps_…`, 0.74x to 1.09x (median 1.02x).

The cases the earlier version of the table ran on the GPU and this one does not, from
`…_idle_gaps_…` (idle against idle after 500 ms / after 5 s): `DISTINCT`, 200 groups, 10M rows,
1.28x / 0.53x and 1.24x / 0.50x (warm 1.45x and 1.73x); `count(*)`, 10,000 groups, 50M rows,
1.33x / 0.94x and 1.03x / 1.10x; `DISTINCT`, 10,000 groups, 50M rows, 1.37x / 1.08x and
1.58x / 1.10x; `SELECT DISTINCT region, sub`, 50M rows, 1.06x / 1.01x and 1.61x / 0.84x.

**Handed back at run time:** the other group counts of the same two shapes at 10M and 50M rows, both
layouts (18 cases): 0.977x to 1.030x of DataFusion alone on the best run and 0.971x to 1.081x on the median
run; the group-count estimate took 0.017 to 0.169 ms. In `…_resweep_recheck_…`, the 23 cases handed
back (`count(*)` over two int32 keys at 10M rows, the three `min`/`max` shapes at 50M): 0.983x to
1.009x on the best run and 0.971x to 1.049x on the median, the estimate 0.017 to 0.242 ms.

**Left at plan time:** DataFusion's own plan in both contexts, with the rule's walk over the plan in
one. The walk added 0.009 to 0.029 ms to planning (median, `examples/plancost.rs`,
`datafusion_plancost_2026-09-29.csv`: 0.139 against 0.148 ms and 0.120 against 0.131 ms at 1M rows,
0.269 against 0.298 ms and 0.272 against 0.282 ms at 10M; plan and run 0.831 against 0.846 ms,
3.526 against 3.532 ms). The 344 left cases in `…_idle_vs_idle_…`, `…_idle_vs_idle_recheck_…`
and `…_idle_gaps_recheck_…` were at 0.894x to 1.165x of DataFusion alone on the best run and 0.943x to 1.133x
on the median run, none below 0.97x on both. The 298 left cases in `…_resweep_recheck_…` were at
0.901x to 1.143x on the best run and 0.927x to 1.053x on the median, one below 0.97x on both
(`avg(q)` over one int64 key, 10,000 groups, 50M rows, 8,192-row batches: 21.27 against 22.54 ms);
timed again alone, 200 runs of each alternating
(`datafusion_groupby_resweep_left_retime_2026-10-01.csv`), it is at 1.000x best and 0.983x median.

**Pipeline compilation** (`examples/coldstart.rs`, `datafusion_coldstart_2026-09-29.csv`, 10M
rows, three fresh processes per query): the first GPU query of a process compiles its Metal
pipelines, 42 to 66 ms more than the same query's next run after the same 500 ms of idle (median
per query: `count(*)` 52.8 ms, `sum` 54.1, `min`/`max` 49.1, `DISTINCT` 53.9, a full sort 49.7).

### Where the time goes: a 50M-row sort

`MetalExec`'s metrics for the best rule-on run, 8,192-row batches, from
`datafusion_sort_warm_2026-09-29.csv`. **Input** is the wait for and collection of the input stream,
**import** the copy of the collected batches into Metal buffers, **sort** the plan run on the GPU,
**export** the hand-back of the result to arrow-rs; **rest** is the wall time outside those four
(planning, the projection above, slicing the output into batches).

| key | wall, ms | input | import | sort | export | rest | input batches |
|---|---:|---:|---:|---:|---:|---:|---:|
| int64 | 69.74 | 7.31 | 10.21 | 48.21 | 0.02 | 3.99 | 6,104 |
| Float64 | 87.90 | 7.32 | 10.60 | 66.24 | 0.01 | 3.73 | 6,104 |
| Float32 | 61.14 | 7.14 | 8.87 | 41.22 | 0.01 | 3.90 | 6,104 |
| String | 87.62 | 7.54 | 10.73 | 65.46 | 0.01 | 3.88 | 6,104 |

The sort itself is 67% to 75% of the wall time. The import writes each column's 6,104 chunks
straight into that column's Metal buffers (`Array::from_arrow_chunks`), with no `concat_batches` copy
before it.

---

## To improve

The measured shapes where the rule is behind DataFusion alone. The default leaves top-k and filters
to DataFusion, and the aggregate shapes below.

### Top-k

`ORDER BY … LIMIT 100`, 8,192-row batches, rule off / rule on in ms (off ÷ on), from
`datafusion_sort_warm_2026-09-29.csv`:

| rows | int64 | Float64 DESC | Float32 DESC |
|---:|---:|---:|---:|
| 250,000 | 0.39 / 1.12 (0.35x) | 0.37 / 1.15 (0.32x) | 0.40 / 0.98 (0.41x) |
| 500,000 | 0.41 / 1.23 (0.33x) | 0.44 / 1.24 (0.35x) | 0.43 / 1.46 (0.29x) |
| 1,000,000 | 0.46 / 2.86 (0.16x) | 0.52 / 2.12 (0.25x) | 0.55 / 2.01 (0.27x) |
| 2,000,000 | 0.63 / 3.94 (0.16x) | 0.73 / 2.45 (0.30x) | 0.71 / 2.40 (0.30x) |
| 10,000,000 | 1.59 / 7.44 (0.21x) | 1.87 / 6.07 (0.31x) | 1.93 / 5.28 (0.37x) |
| 50,000,000 | 4.81 / 30.46 (0.16x) | 6.21 / 26.70 (0.23x) | 6.00 / 22.04 (0.27x) |

One batch per partition: 0.14x to 0.38x. DataFusion's TopK operator keeps the best 100 rows of each
partition while its input streams; `MetalExec` first collects and imports every row. At 50M rows the Float64 top-k's
`input_time` alone is 7.72 ms and its import 9.83 ms, against DataFusion's 6.21 ms for the whole
query.

### Aggregates

With every replaced aggregate forced onto the GPU (`AggregateChoice::ArrowMetal`), DataFusion alone ÷
with the rule, the range over one and two keys, int32 and int64 keys, both layouts and the family's
queries, from `datafusion_groupby_sweep_2026-09-30.csv` (the Float64 `sum`/`avg` and both `min`/`max`
rows measured on 2026-09-30). Of these cells the default takes the two-int32-key cases of `count(*)`
at 100,000 and 1,000,000 groups from 10M rows, of `DISTINCT` at 200, 100,000 and 1,000,000 groups at
50M, and of `min`/`max` over int64 at 100,000 and 1,000,000 groups at 50M over two int32 keys and
over two int64 keys, and at 1,000,000 groups over one int64 key.

| aggregate | groups | 1M | 2M | 5M | 10M | 50M |
|---|---|---|---|---|---|---|
| `count(*)` | 200 | 0.40-0.58 | 0.40-0.92 | 0.64-1.32 | 0.91-1.68 | 1.32-2.31 |
| `count(*)` | 10,000 | 0.70-0.88 | 0.70-0.95 | 0.67-1.63 | 1.14-1.83 | 1.25-2.38 |
| `count(*)` | 100,000 | 1.17-1.31 | 1.49-2.20 | 1.35-2.76 | 2.45-2.95 | 2.45-3.25 |
| `count(*)` | 1,000,000 | | | 2.03-2.46 | 1.88-2.97 | 3.39-4.10 |
| `count(*)` | rows / 2 | 1.13-1.59 | 1.39-1.76 | 1.25-2.06 | 1.88-2.13 | 0.89-0.94 |
| `sum`, `avg` over int64 | 200 | 0.35-0.62 | 0.45-0.83 | 0.64-1.23 | 0.69-1.62 | 0.90-1.94 |
| `sum`, `avg` over int64 | 10,000 | 0.58-0.76 | 0.66-1.17 | 0.82-1.53 | 0.81-1.68 | 0.94-1.75 |
| `sum`, `avg` over int64 | 100,000 | 0.88-1.20 | 1.23-2.08 | 1.67-2.23 | 1.61-2.57 | 1.55-3.20 |
| `sum`, `avg` over int64 | 1,000,000 | | | 1.31-1.74 | 1.58-2.45 | 2.03-3.27 |
| `sum`, `avg` over int64 | rows / 2 | 0.89-1.24 | 1.06-1.27 | 1.01-1.36 | 1.04-1.42 | 0.63-0.78 |
| `sum`, `avg` over Float64 | 200 | 0.21-0.57 | 0.50-1.01 | 0.62-1.36 | 0.65-1.36 | 0.89-1.75 |
| `sum`, `avg` over Float64 | 10,000 | 0.42-0.97 | 0.55-1.14 | 0.77-1.38 | 0.71-1.42 | 0.74-1.55 |
| `sum`, `avg` over Float64 | 100,000 | 0.63-1.32 | 0.84-2.47 | 1.72-2.30 | 1.41-2.51 | 1.30-2.47 |
| `sum`, `avg` over Float64 | 1,000,000 | | | 1.33-1.75 | 1.42-2.10 | 1.75-2.81 |
| `sum`, `avg` over Float64 | rows / 2 | 0.33-1.55 | 0.79-1.45 | 1.02-1.38 | 0.93-1.33 | 0.59-0.79 |
| `min`, `max` over int64 | 200 | 0.38-0.76 | 0.57-0.98 | 0.73-1.34 | 0.77-1.37 | 1.02-1.75 |
| `min`, `max` over int64 | 10,000 | 0.59-1.08 | 0.93-1.25 | 1.00-1.59 | 0.95-1.55 | 0.87-1.61 |
| `min`, `max` over int64 | 100,000 | 1.06-1.65 | 1.34-2.81 | 2.58-3.07 | 2.41-3.02 | 2.03-2.88 |
| `min`, `max` over int64 | 1,000,000 | | | 2.52-3.00 | 2.56-2.97 | 3.11-3.70 |
| `min`, `max` over int64 | rows / 2 | 0.75-3.24 | 1.78-2.19 | 2.28-2.51 | 2.14-2.32 | 1.03-1.11 |
| `min`, `max` over Float64 | 200 | 0.11-0.25 | 0.15-0.34 | 0.18-0.46 | 0.27-0.56 | 0.37-0.73 |
| `min`, `max` over Float64 | 10,000 | 0.32-0.79 | 0.18-0.58 | 0.37-0.59 | 0.37-0.64 | 0.32-0.61 |
| `min`, `max` over Float64 | 100,000 | 0.42-0.68 | 0.41-0.99 | 0.96-1.23 | 0.90-1.31 | 0.71-1.20 |
| `min`, `max` over Float64 | 1,000,000 | | | 0.62-0.80 | 0.69-0.86 | 0.77-1.10 |
| `min`, `max` over Float64 | rows / 2 | 0.26-0.73 | 0.44-0.65 | 0.48-0.61 | 0.44-0.54 | 0.30-0.34 |
| `DISTINCT` | 200 | 0.43-0.60 | 0.59-0.92 | 0.70-1.48 | 1.00-1.73 | 1.20-2.32 |
| `DISTINCT` | 10,000 | 0.74-0.92 | 0.64-1.04 | 0.94-1.92 | 1.24-1.92 | 1.29-2.34 |
| `DISTINCT` | 100,000 | 1.26-1.36 | 1.50-2.55 | 1.69-2.92 | 2.36-2.82 | 2.04-3.36 |
| `DISTINCT` | 1,000,000 | | | 2.08-2.62 | 2.10-2.93 | 2.73-3.84 |
| `DISTINCT` | rows / 2 | 1.25-1.84 | 1.45-1.99 | 2.13-2.45 | 2.10-2.30 | 0.80-0.88 |

At 1M and 2M rows the 1,000,000-group data holds at least rows / 4 groups and is in the rows / 2
row.

**The first run after the GPU idles.** 34 series are at least 1.65x faster warm at two sizes or
more; the default takes 10. The first run after 500 ms of idle (pipelines compiled) of every case the
default has run on the GPU (including `SELECT DISTINCT region, sub` at 50M rows), over the latest
measurement of each case and layout
(`datafusion_groupby_default_idle_2026-09-29.csv`, `datafusion_groupby_recheck_gpu_2026-09-29.csv`,
`datafusion_groupby_idle_vs_idle_2026-09-29.csv`, `datafusion_groupby_idle_gap_2026-09-29.csv`,
`datafusion_groupby_idle_gaps_2026-09-29.csv`, `datafusion_groupby_resweep_check_2026-10-01.csv`,
`datafusion_groupby_resweep_recheck_2026-10-01.csv`; the first two with DataFusion's idle run always
before the default's, the others alternating; the last two measured the Float64 `sum`/`avg` and the
`min`/`max` families after they were rebuilt):

| rows | cases | idle vs idle: DataFusion idle ÷ default idle, min / median / max | at or above 1.0x | GPU idle vs DataFusion warm, min / median / max | DataFusion idle ÷ its warm time |
|---:|---:|---|---:|---|---|
| 2,000,000 | 10 | 0.79 / 1.35 / 2.57 | 8 | 0.35 / 0.43 / 0.72 | 1.72-4.08 |
| 5,000,000 | 50 | 0.64 / 1.22 / 1.78 | 43 | 0.37 / 0.50 / 0.67 | 1.56-3.63 |
| 10,000,000 | 82 | 0.73 / 1.22 / 1.82 | 68 | 0.41 / 0.63 / 1.32 | 1.11-2.82 |
| 50,000,000 | 84 | 0.87 / 1.43 / 2.37 | 80 | 0.48 / 1.06 / 1.94 | 1.14-1.82 |

The 32 cases measured in both `…_idle_…` (DataFusion's idle run first) and `…_idle_vs_idle_…`
(alternating) differ between the two: the later idle-against-idle ratio is 0.69x to 1.51x of the
earlier one (median 0.93x).

Series left at 50M rows by the first run after idle, lowest DataFusion idle ÷ default idle over the
cases and layouts: after 500 ms, `DISTINCT` over one int32 key, 100,000 groups 0.88; after 5 s
(`datafusion_groupby_idle_gaps_2026-09-29.csv`), `count(*)` over two int32 keys, 10,000 groups 0.94,
and `DISTINCT` over two int32 keys, 10,000 groups 0.84 (`SELECT DISTINCT region, sub`, one batch per
partition); after 5 s (`datafusion_groupby_resweep_check_2026-10-01.csv`), `min`/`max` over int64 with
one int32 key, 100,000 groups 0.84, and with one int64 key, 100,000 groups 0.80, and `sum`/`avg` over
Float64 with two int64 keys, 100,000 groups 0.91. The 16 other series that qualify warm and are left
at 50M have no first run after 5 s measured there (`count(*)` and `DISTINCT` over one key and over two
int64 keys, `sum`/`avg` over int64 over two keys). `DISTINCT` over two int32 keys at 200 groups is left at 10M rows by its first run after
5 s: 0.53x and 0.50x.

**The hand-back.** Handing an aggregate back at run time (reading the first batches, estimating the
groups, then running DataFusion's plan) cost, against DataFusion alone, measured run by run
(`datafusion_groupby_default_handback_2026-09-29.csv`, the shapes the earlier table replaced), on
the best run / the median run: at 2M rows a median of 0.969x / 0.967x, lowest 0.934x and 0.921x
(the latter `count(*)` over one int64 key, 864,909 groups: 5.30 → 5.61 ms best); at 5M a median of
0.984x / 0.984x, lowest 0.936x and 0.914x; at 10M 0.990x / 0.991x, lowest 0.884x and 0.968x; at 50M
0.993x / 0.995x, lowest 0.944x and 0.924x. Series left at 50M because a hand-back of their shape
there is below 0.97x (best / median, `datafusion_groupby_resweep_check_2026-10-01.csv`): `min`/`max`
over int64 with one int32 key, 1,000,000 groups (`minmax_int_1i32_10k`, 0.994x / 0.951x); `sum`/`avg`
over Float64 with two int32 keys, 100,000 groups (`sum_f64_2i32_200`, 0.989x / 0.925x) and with two
int64 keys, 1,000,000 groups (`sum_f64_2i64_200`, 0.986x / 0.942x). The 16 series above with no first
run after 5 s at 50M have hand-backs of their shape there below 0.97x as well, by shape, with the
shape's lowest hand-back at 50M (best / median; the latest measurement of each case in the check
files): `count(*)` over one int32 key,
100,000 and 1,000,000 groups (0.961x / 1.003x); over one int64 key, 100,000 and 1,000,000 groups
(0.948x / 0.943x); over two int64 keys, 100,000 and 1,000,000 groups (0.972x / 0.958x); `DISTINCT`
over one int32 key, 1,000,000 groups (0.968x / 0.996x); over one int64 key, 100,000 and 1,000,000
groups (0.901x / 1.039x); over two int64 keys, 100,000 and 1,000,000 groups (0.953x / 1.002x);
`sum`/`avg` over int64 with two int32 keys, 100,000 and 1,000,000 groups (0.990x / 0.924x); with two
int64 keys, 100,000 and 1,000,000 groups (0.944x / 0.992x).

A Float64 `MIN`/`MAX` through the rule carries four helper aggregates per extreme (the NaN, value,
zero and negative-zero counts, each a sum over an `if_else`) next to the two extremes: at 200 groups
from 50M rows its GPU plan takes 36.88 ms (`datafusion_float64_groupby_2026-09-30.csv`), where the
plan runner's own `min` + `max` of a Float64 column over one int32 key takes 7.34 ms
(`Benchmarks/results/groupby_float64_2026-09-30.csv`).

### Filters

With `ArrowMetalConfig::all()`, from `datafusion_rule_2026-09-29.csv`:

| query | rows | off ÷ on |
|---|---:|---|
| `SELECT sum(amount), count(*) FROM fact WHERE region < 20 AND qty > 10` (`a_sumcnt`) | 1M / 10M / 50M | 0.56-0.59 / 0.62-0.72 / 0.74 |
| `SELECT qty, sum(weight), count(*) FROM f WHERE price > 500.0 GROUP BY qty`, Parquet (`p_filtgroup`) | 10M / 50M | 0.51 snappy, 0.52 zstd / 0.54 snappy, 0.55 zstd |

DataFusion's `FilterExec` streams batch by batch; `MetalExec` collects the whole input first.

### The first query of a process

The first GPU query in a process also compiles the Metal pipelines. In the table above, the int64
sort's first run at 250,000 to 1,000,000 rows (31.67 to 34.32 ms, starred) is slower than DataFusion's
warm time for the same query (8.08 to 31.75 ms).

---

## Limits

- **It collects its input.** `MetalExec` reads every input partition to the end before it sorts or
  runs an aggregate on the GPU (an aggregate it hands back reads only the first batches of each
  partition before DataFusion takes over). It has no spill path of its own. The collected batches
  and the result are counted in DataFusion's memory pool (a `MemoryConsumer` named `MetalExec`, at
  the bytes each batch's slices reference); when the pool refuses, `MetalExec` hands the node back to
  DataFusion, whose operators can spill. The Metal buffers themselves (the imported columns) are not
  counted in the pool.
- **The GPU call is not cancelled.** It runs inside one `tokio::task::spawn_blocking` call; when the
  query is dropped, it runs to its end and its result is discarded.
- **One GPU job at a time per process.** A process-wide lock serialises every `MetalExec`, so two
  queries that both reach the GPU run their GPU parts one after the other.
- **One blocking thread per node.** ArrowMetal handles are not `Send`, so all GPU work of one
  `MetalExec` runs inside one `tokio::task::spawn_blocking` call, off DataFusion's async worker
  threads.
- **Exact row counts.** By default a node is taken only when DataFusion's statistics give its input
  an exact row count. A sort above a filter, a join or an aggregate has an estimate and is left
  unless `accept_inexact` is set; an unknown count is left unless `take_when_unknown` is set.
- **Aggregates.** The measured table takes two shapes: `count` over two int32 (or narrower) keys of
  a `MemTable` scan with at least 10,000,000 rows, and `DISTINCT` over the same keys with at least
  50,000,000 rows. Every other aggregate is left,
  including every aggregate over a Parquet scan, a filter or a join, and every one with a float or
  string key.
- **Pipeline compilation.** The first GPU query of a process compiles its Metal pipelines: 42 to
  66 ms at 10M rows (`datafusion_coldstart_2026-09-29.csv`).
- **macOS on Apple silicon only.** `build.rs` stops the build for any other target with a message
  naming the target; `arrowmetal-sys` refuses non-macOS targets as well.
- **DataFusion 55.1.0 only.** The dependency is pinned with `=`; DataFusion's physical-plan API
  changes between releases.
- **Not on crates.io.** The crate is used by path from a checkout.

---

## Tests and files

```bash
cd datafusion
ARROWMETAL_LIB=/path/to/libArrowMetalC.dylib cargo test                       # 30 tests + 1 doc-test
ARROWMETAL_LIB=/path/to/libArrowMetalC.dylib cargo test --test rule -- --ignored   # default config at 2M rows
```

| File | What it checks |
|---|---|
| `tests/grid.rs` | the differential grid above (1 test) |
| `tests/rule.rs` | 14 tests: the threshold and its reason, inexact statistics, the config switches, what the default leaves, unsupported shapes, `EXPLAIN`, `ORDER BY` through a projection, replaced aggregates under a partitioned join, `count(DISTINCT)`, the forced hand-back, the measured choice's two branches, a refused memory reservation; plus 1 ignored test of the default configuration at 2,000,000 rows |
| `src/probe.rs`, `src/exec.rs` (unit tests) | 7 tests: the group-count estimate (exact small inputs, 200 to 1,000,000 groups, sorted keys, key tuples, the same estimate for the same input, a prefix of the input) and the whole-input check of a prefix that under-counts |
| `tests/arrowmetal_repros.rs` | 8 tests of ArrowMetal's plan runner alone, no DataFusion: totalOrder against arrow-rs, NaN and zero group keys, `count` on every path, the chunked import |

On 2026-09-29: 30 passed, 2 ignored, doc-test passed; the ignored rule test passed when run with
`--ignored`.

| Path | What it is |
|---|---|
| `datafusion/src/rule.rs` | `ArrowMetalRule`, `ArrowMetalConfig`, `AggregateChoice`, `Report`, `Decision` |
| `datafusion/src/exec.rs` | `MetalExec`, `MetalOp`: collection, the run-time choice, the hand-back |
| `datafusion/src/probe.rs` | the group-count estimate |
| `datafusion/src/choice.rs` | an aggregate's shape and the table lookup |
| `datafusion/src/agg_table.rs` | the measured table (generated) |
| `datafusion/scripts/groupby_table.py` | generates the table from the sweep and the default-check CSVs (`--check`, `--print`) |
| `datafusion/src/translate.rs` | the checks on node shapes and types; predicates to ArrowMetal expressions |
| `datafusion/src/gpu.rs` | the plans sent to ArrowMetal, the import, and the run-time checks |
| `datafusion/examples/quickstart.rs` | the example above |
| `datafusion/examples/bench.rs` | the benchmark |
| `datafusion/examples/coldstart.rs` | the first GPU query of a process, against the next one after idle |
| `datafusion/examples/plancost.rs` | the rule's planning cost on a plan it leaves |
| `datafusion/results/datafusion_sort_warm_2026-09-29.csv` | sorts and top-k with the warm-up, 100,000 to 50M rows |
| `datafusion/results/datafusion_rule_2026-09-29.csv` | every case, rule off and on, 100,000 to 50M rows, and the Parquet cases |
| `datafusion/results/datafusion_groupby_sweep_2026-09-30.csv` | the aggregate sweep, rule forced on against off, 1M to 50M rows: the table's source (the Float64 `sum`/`avg` and `min`/`max` and the int64 `min`/`max` cases measured on 2026-09-30, the other cases from `datafusion_groupby_sweep_2026-09-29.csv`, which also has the rule forced back) |
| `datafusion/results/datafusion_groupby_resweep_check_2026-10-01.csv`, `datafusion/results/datafusion_groupby_resweep_recheck_2026-10-01.csv`, `datafusion/results/datafusion_groupby_resweep_left_retime_2026-10-01.csv` | the default-take checks after the 2026-09-30 sweep: every case of the re-swept families the sweep-only table replaced, then every case whose decision the final table changed and a seeded fifth of the rest; one left case timed again alone |
| `datafusion/results/datafusion_groupby_default_2026-09-29.csv` and `…_taken_…`, `…_cap_…`, `…_idle_…`, `…_handback_…`, `…_recheck_gpu_…`, `…_recheck_…`, `…_recheck2_…`, `…_idle_vs_idle_…`, `…_idle_vs_idle_recheck_…`, `…_idle_gap_…`, `…_idle_gaps_…`, `…_idle_gaps_recheck_…` | the aggregates with the default configuration timed; `…_idle_gap_…`: twelve aggregates after 5 s and after 500 ms of idle; `…_idle_gaps_…`: both gaps per case |
| `datafusion/results/datafusion_coldstart_2026-09-29.csv` | pipeline compilation per process |
| `datafusion/results/datafusion_plancost_2026-09-29.csv` | the rule's planning cost |
| `datafusion/results/datafusion_parquet_string_sort_2026-09-29.csv` | Parquet sorts with a string column |

The first two CSVs also hold rows with `crate = before`: an earlier state of this crate, measured in the same
sessions. The tables on this page use the rows with `crate = now`.
