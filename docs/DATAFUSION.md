# ArrowMetal for DataFusion

[Apache DataFusion](https://datafusion.apache.org) is a query engine in Rust: SQL and DataFrames over
Arrow, with its own planner, optimizer and multi-threaded operators. `datafusion-arrowmetal` (in
[`datafusion/`](../datafusion)) is a **physical optimizer rule for DataFusion 55.1**. Registered on a
`SessionContext`, it replaces DataFusion's full sort (an `ORDER BY` without `LIMIT`) with `MetalExec`,
which runs the sort on the Apple GPU through ArrowMetal's plan runner and hands DataFusion the
`RecordBatch`es it expects. It also replaces the hash aggregates (`GROUP BY`) of the one shape a
measured table takes: `count(*)` over two int32 keys of a `MemTable` of at least 50,000,000 rows.
Such an aggregate estimates its number of groups from a sample of its keys when it runs, runs on the
GPU at about 316,000 to 3,160,000 groups, and otherwise hands the node back to DataFusion's own
operators. The SQL does not change, and every other node of the plan stays DataFusion's.

**The summary.** On an Apple M4 Max with DataFusion's default of one partition per core, a full sort
of 250,000 to 50,000,000 rows is **6.9x to 28.8x faster** with the rule than DataFusion alone, and at
50M rows it uses 96 to 126 CPU-ms where DataFusion uses 6,104 to 8,311. Over DataFusion's own Parquet
reader, a Float64 sort is 10.8x to 16.5x faster at 10M and 50M rows, and a sort over a Parquet file
with a string column 3.6x to 5.5x at 1M to 50M rows. The aggregate the default runs on the GPU,
`count(*)` over two int32 keys with 1,000,000 groups at 50M rows, was 3.90x and 4.16x faster than
DataFusion alone, and 1.23x and 1.70x on its first run after the GPU had idled for 500 ms
([Aggregates](#aggregates-with-the-default)). The answers are DataFusion's: a differential grid of
7,656 query pairs runs every query with and without the rule and finds 0 mismatches. Top-k
(`ORDER BY … LIMIT`), filters and every other aggregate shape are left to DataFusion by default; the
measured numbers for them are under [To improve](#to-improve).

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
LEFT     AggregateExec: mode=FinalPartitioned, gby=[region@0 as region], aggr=[count(Int64(1)), avg(sales.amount)] -- input rows 1000000 (exact) vs min_rows 250000; count + sum_avg_f64 over 1 i64 key (memory input): the measured table takes it at no group count and size (results/datafusion_groupby_sweep_2026-09-29.csv)
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

1. The sweep (`datafusion_groupby_sweep_2026-09-29.csv`: every family over one and two int32 and
   int64 keys, at 200 to about rows / 2 groups, 1M to 50M rows, both table layouts, the rule forced
   onto the GPU against DataFusion alone). A series (shape and number of groups) qualifies from the
   smallest measured row count at which its worst case (over the layouts and the family's queries)
   was at least 1.65x faster than DataFusion alone, at that size and every larger one, at two sizes
   or more. 26 of the 120 measured series qualify, from 2M to 10M rows.
2. The first run after the GPU idles: a series is taken at a size only if, at that size and every
   larger one, the default's first run after 500 ms of idle (the Metal pipelines already compiled)
   was at least as fast as DataFusion alone's warm time, in every case and layout
   (`datafusion_groupby_default_idle_2026-09-29.csv`, `datafusion_groupby_recheck_gpu_2026-09-29.csv`).
3. The hand-back: a shape is replaced at a size only if, at that size and every larger one, every
   case the default handed back at run time was at 0.97x of DataFusion alone or better on the best
   run and on the median run (`datafusion_groupby_default_handback_2026-09-29.csv`,
   `datafusion_groupby_recheck_2026-09-29.csv` and the files before them).

One series is taken:

| shape | groups | taken from |
|---|---|---|
| `count` (`count(*)`, `count(x)`) over two int32 keys (or narrower), `MemTable` input | 316,228 to 3,162,277 | 50,000,000 rows |

Of the other 25 series that qualify in the sweep, 20 are not taken because their first run after
idle was slower than DataFusion alone's warm time at every size, and 5 (`count` over one int32, one
int64 or two int64 keys, `DISTINCT` over one int64 key, and `sum`/`avg` over two int32 keys, all at
1,000,000 groups) because a hand-back of the same shape at 50M rows cost more than 3% ([To
improve](#aggregates)).

---

## What it leaves, and why

| Shape | Default | Why |
|---|---|---|
| top-k, `ORDER BY … LIMIT` | left | behind DataFusion at every measured size from 250,000 to 50M rows: 0.14x to 0.41x ([To improve](#top-k)) |
| `GROUP BY`, `DISTINCT` of a shape the table does not take at the input's row count | left | behind DataFusion, not 1.65x ahead in the worst case at two sizes, slower than DataFusion on the first run after idle, or its hand-backs cost more than 3% ([To improve](#aggregates)) |
| `count(*)` over two int32 keys at 50M rows or more whose estimated number of groups is not in the taken range | handed back at run time | `HANDBACK` in the report |
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

Run on 2026-09-29 (`cargo test`, debug build, 85 s):

| | |
|---|---|
| query pairs | **7,656**: 5,304 with every node forced onto the GPU, 1,176 `GROUP BY` forced back, 1,176 `GROUP BY` with the measured choice |
| pairs with a node replaced | 5,976, including all 3,696 `ORDER BY` pairs |
| run-time choices | forced onto the GPU: 960 on the GPU; forced back: 960 handed back; measured: none replaced (the table takes one shape at 50M rows; the grid's tables hold at most 20,000 rows) |
| mismatches | **0** |
| hand-backs on an ArrowMetal error | **0** |
| hand-backs on the data (the checks above) | 128: 64 Float64 `MIN` over a group with NaN, 32 filters on a column holding -NaN, 32 group keys with several NaN patterns |
| largest relative deviation in a float `sum`/`avg` | 2.8e-13 |

The pairs of the forced variant with nothing taken are Float32 `MIN`/`MAX` group-bys, a `CAST` in a
predicate, and a float comparison against zero, each left by the rule with its reason.
`tests/rule.rs` checks both branches of the measured choice with the table looked up at 50M rows:
`count(*)` over two int32 keys drawn from 1,000,000 values runs on the GPU, over 150 values it is
handed back, on one batch per partition and on 8,192-row batches. A unit test in `src/exec.rs`
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
rows / 2 groups) and `SELECT DISTINCT region, sub FROM fact`, with `ArrowMetalConfig::default()`
timed against DataFusion alone, both layouts, 1M to 50M rows. A case whose default ran on the GPU:
three rounds of best of 5 with the two contexts in rotated order, then three runs of each after
500 ms of sleep. Any other case (both contexts on the CPU): 30 runs of each, alternating run by run,
after a 100 ms warm-up. The files: `datafusion_groupby_default_2026-09-29.csv` (every case, the
earlier table) and the re-measurements after it, whose rows replace its rows for the same case,
size and layout (`…_taken_…`, `…_cap_…`, `…_idle_…`, `…_handback_…`, `…_recheck_gpu_…`,
`…_recheck_…`, `…_recheck2_…`); every row whose decision the final table changed was re-measured,
and so was a random fifth of the rest. Every answer was equal to DataFusion's.

| rows | cases | left at plan time | handed back at run time | run on the GPU |
|---:|---:|---:|---:|---:|
| 1,000,000 to 10,000,000 | 1,224 | 1,224 | 0 | 0 |
| 50,000,000 | 322 | 312 | 8 | 2 |

**On the GPU** (`datafusion_groupby_recheck_gpu_2026-09-29.csv`), `count(*)` over two int32 keys,
1,000,000 groups, 50M rows:

| layout | DataFusion alone warm, ms | default warm, ms | off ÷ default | first run after 500 ms idle, ms (median of 3; slowest) | DataFusion alone warm ÷ that | DataFusion alone after the same idle, ms | CPU-ms off / default |
|---|---:|---:|---:|---|---:|---:|---|
| 8,192-row batches | 74.78 | 19.18 | 3.90x | 60.70; 72.02 | 1.23x | 94.94 | 1,129 / 50 |
| one batch per partition | 73.99 | 17.80 | 4.16x | 43.64; 58.56 | 1.70x | 95.18 | 1,120 / 49 |

**Handed back at run time:** the same shape at 50M rows with 200, 10,000, 100,000 and about
rows / 2 groups, both layouts: 0.977x to 1.037x of DataFusion alone on the best run and 0.995x to
1.023x on the median run; the group-count estimate took 0.02 to 0.17 ms.

**Left at plan time:** DataFusion's own plan in both contexts, with the rule's walk over the plan in
one. The walk added 0.009 to 0.029 ms to planning (median, `examples/plancost.rs`,
`datafusion_plancost_2026-09-29.csv`: 0.139 against 0.148 ms and 0.120 against 0.131 ms at 1M rows,
0.269 against 0.298 ms and 0.272 against 0.282 ms at 10M; plan and run 0.831 against 0.846 ms,
3.526 against 3.532 ms). Measured run by run, the 661 left cases were at 0.906x to 1.109x of
DataFusion alone on the best run and 0.936x to 1.068x on the median run, none below 0.97x on both;
the 875 left cases measured only in rotated rounds were at 0.912x to 1.108x and 0.906x to 1.319x.

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
queries, from `datafusion_groupby_sweep_2026-09-29.csv`. The default takes one cell's two-int32-key
cases: `count(*)` at 1,000,000 groups, 50M rows (bold).

| aggregate | groups | 1M | 2M | 5M | 10M | 50M |
|---|---|---|---|---|---|---|
| `count(*)` | 200 | 0.40-0.58 | 0.40-0.92 | 0.64-1.32 | 0.91-1.68 | 1.32-2.31 |
| `count(*)` | 10,000 | 0.70-0.88 | 0.70-0.95 | 0.67-1.63 | 1.14-1.83 | 1.25-2.38 |
| `count(*)` | 100,000 | 1.17-1.31 | 1.49-2.20 | 1.35-2.76 | 2.45-2.95 | 2.45-3.25 |
| `count(*)` | 1,000,000 | | | 2.03-2.46 | 1.88-2.97 | **3.39-4.10** |
| `count(*)` | rows / 2 | 1.13-1.59 | 1.39-1.76 | 1.25-2.06 | 1.88-2.13 | 0.89-0.94 |
| `sum`, `avg` over int64 | 200 | 0.35-0.62 | 0.45-0.83 | 0.64-1.23 | 0.69-1.62 | 0.90-1.94 |
| `sum`, `avg` over int64 | 10,000 | 0.58-0.76 | 0.66-1.17 | 0.82-1.53 | 0.81-1.68 | 0.94-1.75 |
| `sum`, `avg` over int64 | 100,000 | 0.88-1.20 | 1.23-2.08 | 1.67-2.23 | 1.61-2.57 | 1.55-3.20 |
| `sum`, `avg` over int64 | 1,000,000 | | | 1.31-1.74 | 1.58-2.45 | 2.03-3.27 |
| `sum`, `avg` over int64 | rows / 2 | 0.89-1.24 | 1.06-1.27 | 1.01-1.36 | 1.04-1.42 | 0.63-0.78 |
| `sum`, `avg` over Float64 | 200 | 0.22-0.56 | 0.40-0.68 | 0.51-1.00 | 0.57-1.29 | 0.64-1.26 |
| `sum`, `avg` over Float64 | 10,000 | 0.47-0.69 | 0.46-0.85 | 0.55-1.00 | 0.48-0.97 | 0.46-0.98 |
| `sum`, `avg` over Float64 | 100,000 | 0.78-0.98 | 0.94-1.21 | 1.01-1.38 | 1.00-1.55 | 0.77-1.67 |
| `sum`, `avg` over Float64 | 1,000,000 | | | 0.75-1.03 | 0.87-1.39 | 0.92-1.68 |
| `sum`, `avg` over Float64 | rows / 2 | 0.52-0.85 | 0.61-0.80 | 0.53-0.75 | 0.55-0.82 | 0.44-0.57 |
| `min`, `max` over int64 | 200 | 0.37-0.51 | 0.42-0.76 | 0.60-1.07 | 0.69-1.23 | 0.84-1.48 |
| `min`, `max` over int64 | 10,000 | 0.48-0.87 | 0.69-0.99 | 0.80-1.19 | 0.64-1.21 | 0.66-1.27 |
| `min`, `max` over int64 | 100,000 | 1.05-1.28 | 1.60-1.84 | 1.76-2.19 | 1.76-2.07 | 1.31-2.28 |
| `min`, `max` over int64 | 1,000,000 | | | 1.58-1.69 | 1.42-1.82 | 1.35-1.89 |
| `min`, `max` over int64 | rows / 2 | 1.25-1.59 | 1.46-1.81 | 1.21-1.43 | 1.09-1.23 | 0.56-0.60 |
| `min`, `max` over Float64 | 200 | 0.16-0.30 | 0.19-0.33 | 0.19-0.40 | 0.27-0.54 | 0.36-0.69 |
| `min`, `max` over Float64 | 10,000 | 0.29-0.43 | 0.33-0.50 | 0.34-0.58 | 0.32-0.56 | 0.28-0.58 |
| `min`, `max` over Float64 | 100,000 | 0.41-0.64 | 0.57-0.80 | 0.84-1.05 | 0.82-1.05 | 0.64-1.07 |
| `min`, `max` over Float64 | 1,000,000 | | | 0.57-0.70 | 0.62-0.84 | 0.67-0.99 |
| `min`, `max` over Float64 | rows / 2 | 0.39-0.52 | 0.51-0.57 | 0.42-0.54 | 0.40-0.48 | 0.26-0.29 |
| `DISTINCT` | 200 | 0.43-0.60 | 0.59-0.92 | 0.70-1.48 | 1.00-1.73 | 1.20-2.32 |
| `DISTINCT` | 10,000 | 0.74-0.92 | 0.64-1.04 | 0.94-1.92 | 1.24-1.92 | 1.29-2.34 |
| `DISTINCT` | 100,000 | 1.26-1.36 | 1.50-2.55 | 1.69-2.92 | 2.36-2.82 | 2.04-3.36 |
| `DISTINCT` | 1,000,000 | | | 2.08-2.62 | 2.10-2.93 | 2.73-3.84 |
| `DISTINCT` | rows / 2 | 1.25-1.84 | 1.45-1.99 | 2.13-2.45 | 2.10-2.30 | 0.80-0.88 |

At 1M and 2M rows the 1,000,000-group data holds at least rows / 4 groups and is in the rows / 2
row.

**The first run after the GPU idles.** Of the 26 series that are at least 1.65x faster warm at two
sizes or more, 25 are left because of their first run after 500 ms of idle (pipelines compiled) or
the cost of a hand-back of the same shape. The first run after idle against DataFusion alone's warm time, lowest over
the cases and layouts of each size, from `datafusion_groupby_default_idle_2026-09-29.csv` (the
default's GPU runs with the earlier table):

| rows | cases | warm: off ÷ default, min / median | first run after idle: DataFusion warm ÷ it, min / median / max | at or above 1.0x |
|---:|---:|---|---|---:|
| 2,000,000 | 6 | 1.48 / 1.73 | 0.35 / 0.42 / 0.72 | 0 |
| 5,000,000 | 34 | 1.94 / 2.31 | 0.40 / 0.48 / 0.63 | 0 |
| 10,000,000 | 60 | 1.74 / 2.61 | 0.41 / 0.63 / 0.93 | 0 |
| 50,000,000 | 60 | 1.81 / 2.76 | 0.62 / 0.97 / 1.94 | 27 |

DataFusion alone is slower after the same idle too: its first run after 500 ms took 1.16x to 4.08x
its warm time in these cases (median 3.30x at 2M rows, 1.33x at 50M). Against DataFusion alone after
the same idle, DataFusion alone's time ÷ the default's first run was 0.71x to 2.57x (median 1.20x to
1.45x per size).

Series left at 50M rows by the first run after idle, lowest DataFusion warm ÷ first run over the
cases and layouts: `count(*)`, one int32 key, 100,000 groups 0.83; one int64 key, 100,000 groups
0.93; two int32 keys, 10,000 groups 0.72 and 100,000 groups 0.93; two int64 keys, 100,000 groups
0.93; `DISTINCT`, one int32 key, 100,000 groups 0.62 and 1,000,000 groups 0.87; one int64 key,
100,000 groups 0.66; two int32 keys, 200 groups 0.65, 10,000 groups 0.64, 100,000 groups 0.88,
1,000,000 groups 0.997; two int64 keys, 100,000 groups 0.91 and 1,000,000 groups 0.83; `min`/`max`
over int64, two int32 keys, 100,000 groups 0.73; two int64 keys, 100,000 groups 0.84 and 1,000,000
groups 0.85; `sum`/`avg` over int64, two int32 keys, 100,000 groups 0.79; two int64 keys, 100,000
groups 0.77 and 1,000,000 groups 0.92.

**The hand-back.** Handing an aggregate back at run time (reading the first batches, estimating the
groups, then running DataFusion's plan) cost, against DataFusion alone, measured run by run
(`datafusion_groupby_default_handback_2026-09-29.csv`, the shapes the earlier table replaced), on
the best run / the median run: at 2M rows a median of 0.969x / 0.967x, lowest 0.934x and 0.921x
(the latter `count(*)` over one int64 key, 864,909 groups: 5.30 → 5.61 ms best); at 5M a median of
0.984x / 0.984x, lowest 0.936x and 0.914x; at 10M 0.990x / 0.991x, lowest 0.884x and 0.968x; at 50M
0.993x / 0.995x, lowest 0.944x and 0.924x. The series whose first run after idle passes at 50M but
whose shape's hand-backs there do not, with that shape's lowest hand-back at 50M (best / median):
`count(*)` over one int32 key, 1,000,000 groups (0.961x / 1.003x); over one int64 key (0.948x /
0.943x); over two int64 keys (0.972x / 0.958x); `DISTINCT` over one int64 key (0.901x / 1.039x);
`sum`/`avg` over int64 with two int32 keys (0.990x / 0.924x).

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
- **Aggregates.** The measured table takes one shape: `count` over two int32 (or narrower) keys of a
  `MemTable` scan with at least 50,000,000 rows. Every other aggregate is left, including every
  aggregate over a Parquet scan, a filter or a join, and every one with a float or string key.
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
| `datafusion/results/datafusion_groupby_sweep_2026-09-29.csv` | the aggregate sweep, rule forced on, off and forced back, 1M to 50M rows: the table's source |
| `datafusion/results/datafusion_groupby_default_2026-09-29.csv` and `…_taken_…`, `…_cap_…`, `…_idle_…`, `…_handback_…`, `…_recheck_gpu_…`, `…_recheck_…`, `…_recheck2_…` | the aggregates with the default configuration timed |
| `datafusion/results/datafusion_coldstart_2026-09-29.csv` | pipeline compilation per process |
| `datafusion/results/datafusion_plancost_2026-09-29.csv` | the rule's planning cost |
| `datafusion/results/datafusion_parquet_string_sort_2026-09-29.csv` | Parquet sorts with a string column |

The first two CSVs also hold rows with `crate = before`: an earlier state of this crate, measured in the same
sessions. The tables on this page use the rows with `crate = now`.
