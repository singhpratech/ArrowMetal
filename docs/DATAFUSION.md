# ArrowMetal for DataFusion

[Apache DataFusion](https://datafusion.apache.org) is a query engine in Rust: SQL and DataFrames over
Arrow, with its own planner, optimizer and multi-threaded operators. `datafusion-arrowmetal` (in
[`datafusion/`](../datafusion)) is a **physical optimizer rule for DataFusion 55.1**. Registered on a
`SessionContext`, it replaces DataFusion's full sort (an `ORDER BY` without `LIMIT`) with `MetalExec`,
which runs the sort on the Apple GPU through ArrowMetal's plan runner and hands DataFusion the
`RecordBatch`es it expects. The SQL does not change, and every other node of the plan stays
DataFusion's.

**The summary.** On an Apple M4 Max with DataFusion's default of one partition per core, a full sort
of 250,000 to 50,000,000 rows is **6.9x to 28.8x faster** with the rule than DataFusion alone, and at
50M rows it uses 96 to 126 CPU-ms where DataFusion uses 6,104 to 8,311. Over DataFusion's own Parquet
reader, a Float64 sort is 10.8x to 16.5x faster at 10M and 50M rows. The answers are DataFusion's: a
differential grid of 4,656 query pairs runs every query with and without the rule and finds 0
mismatches. Top-k (`ORDER BY … LIMIT`), aggregates and filters are left to DataFusion by default; the
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
`kernel_time`, `export_time` and `input_batches`; the benchmark reads its time split from them.

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
LEFT     AggregateExec: mode=FinalPartitioned, gby=[region@0 as region], aggr=[count(Int64(1)), avg(sales.amount)] -- aggregate disabled in config
LEFT     AggregateExec: mode=Partial, gby=[region@0 as region], aggr=[count(Int64(1)), avg(sales.amount)] -- aggregate disabled in config

13 rows; the first 3:
+--------+-------+-------------------+
| region | n     | mean              |
+--------+-------+-------------------+
| 0      | 76924 | 714.7095888611585 |
| 1      | 76923 | 714.7337580454488 |
| 2      | 76923 | 714.7391491996768 |
+--------+-------+-------------------+
```

The report (`rule.report()`, a `Report` of `Decision`s) has one line per node the rule looked at:
`TAKEN` or `LEFT` with the reason, and `FALLBACK` for a `MetalExec` that handed its node back to
DataFusion at run time. Clones of a rule share one report. It keeps the decisions of every query
planned with the rule, `EXPLAIN` included, until `clear_report()` empties it.

---

## What the default takes

`ArrowMetalConfig::default()`:

| Field | Default | Meaning |
|---|---|---|
| `sort` | **on** | full sorts: `ORDER BY` without `LIMIT` |
| `min_rows` | **250,000** | take a node only when its input has at least this many rows |
| `topk` | off | `ORDER BY … LIMIT` |
| `aggregate` | off | hash `GROUP BY` and `DISTINCT` |
| `filter` | off | `WHERE` |
| `accept_inexact` | off | use an estimated row count as if it were exact |
| `take_when_unknown` | off | take a node whose input row count is unknown |

`ArrowMetalConfig::all()` switches `topk`, `aggregate` and `filter` on as well; the grid and the
benchmark use it with `min_rows: 0`.

A full sort is taken when:

- every sort key is a column (not an expression) of type int8 to int64, uint8 to uint64, Float32,
  Float64 or Utf8;
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

---

## What it leaves, and why

| Shape | Default | Why |
|---|---|---|
| top-k, `ORDER BY … LIMIT` | left | behind DataFusion at every measured size from 250,000 to 50M rows: 0.14x to 0.41x ([To improve](#top-k)) |
| `GROUP BY`, `DISTINCT` | left | ahead on some measured shapes and behind on others with the same row count ([To improve](#aggregates)); the rule sees DataFusion's row count but not the number of groups |
| `WHERE` | left | behind on the measured shapes: 0.51x to 0.74x ([To improve](#filters)) |
| joins | not replaced | the rule has no join operator |
| a sort over an estimated row count (above a filter, a join or an aggregate) | left | the threshold needs an exact count; `accept_inexact` takes it |
| a sort key that is an expression, or a key or carried column of another type (Date, Timestamp, Decimal, LargeUtf8, Utf8View, Dictionary, nested) | left | not in the types the grid covers |
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
returns an error, `MetalExec` runs that subtree over the batches it collected, returns DataFusion's
answer, and adds a `FALLBACK` line to the report. No sort in the grid or in the benchmark took that
path. With aggregates and filters switched on, three checks on the data also hand the node back,
because on such data only DataFusion can give DataFusion's answer:

- a filter that compares a float column with a literal, when the column holds a NaN with the sign bit
  set (totalOrder puts it below -inf; the GPU comparison cannot place it);
- a `GROUP BY` over a float key that holds NaNs of more than one bit pattern (DataFusion keeps each
  pattern as its own group; ArrowMetal puts every NaN in one group);
- a grouped Float64 `MIN` or `MAX` over a group that holds a NaN, or both -0.0 and +0.0 with a zero
  result (DataFusion's answer there depends on the order the rows arrive in).

---

## The differential grid

`tests/grid.rs` runs every query twice, in a plain `SessionContext` and in one with the rule at
`ArrowMetalConfig { min_rows: 0, ..ArrowMetalConfig::all() }`, and compares the answers.

- **Tables:** 11 columns: Int32 and Int64 keys; Float64 keys with ±0.0, ±inf and NaN; Float64 keys
  with -NaN and NaN payloads of both signs; Float32 keys with NaN, ±0.0 and ±inf; Utf8 keys; Int32
  and Int64 values; Float64 values with NaN, NaN payloads and both zeros; Float64 values whose sums
  round; Float32 values with -NaN and a NaN payload.
- **24 table configurations:** 0, 1, 1,000 and 20,000 rows × null fraction 0, 0.1 and 1.0 on every
  column × two layouts (one partition with `target_partitions` 1; three partitions with
  `target_partitions` 4, which plans per-partition sorts under a merge).
- **194 queries:** 128 `ORDER BY` (10 columns × ASC/DESC × `NULLS FIRST`/`NULLS LAST`/default × with
  and without `LIMIT 7`, plus 8 multi-column orderings), 48 `GROUP BY`, and 18 `WHERE`.
- **Comparison:** floats by bit pattern (so -0.0 ≠ +0.0, and NaN sign and payload must match), except
  float `sum` and every `avg`, compared within 1e-9 relative.

Run on 2026-09-29 (`cargo test`, debug build, 57 s):

| | |
|---|---|
| query pairs | **4,656** |
| pairs with a node run on ArrowMetal | 4,368, including all 3,072 `ORDER BY` pairs (1,536 full sorts, 1,536 with `LIMIT`) |
| mismatches | **0** |
| hand-backs on an ArrowMetal error | **0** |
| hand-backs on the data (the checks above) | 128: 64 Float64 `MIN` over a group with NaN, 32 filters on a column holding -NaN, 32 group keys with several NaN patterns |
| largest relative deviation in a float `sum`/`avg` | 2.8e-13 |

The 288 pairs with nothing taken are Float32 `MIN`/`MAX` group-bys, a `CAST` in a predicate, and a
float comparison against zero, each left by the rule with its reason.

---

## Numbers

**Machine and method.** Apple M4 Max (16 CPU cores: 12 performance, 4 efficiency), 64 GB, macOS 27.0;
rustc 1.95.0, DataFusion 55.1.0, arrow-rs 59.3.0; the benchmark is
[`examples/bench.rs`](../datafusion/examples/bench.rs), built `--release` with thin LTO.

- `SessionConfig::new()`: 16 partitions, 8,192-row batches.
- Each query runs in a context without the rule (**off**) and in one with
  `ArrowMetalConfig { min_rows: 0, accept_inexact: true, ..ArrowMetalConfig::all() }` (**on**); both
  answers are compared before anything is timed. A third context with `ArrowMetalConfig::default()`
  is only planned, and its decisions are recorded in the CSV's `rule_default` column.
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

The measured shapes where the rule is behind DataFusion alone. The default leaves top-k, aggregates
and filters to DataFusion; the aggregate table also shows the group counts where the rule is ahead,
because the same SQL falls on both sides depending on the number of groups.

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

`GROUP BY` over one or two int32 keys, with `ArrowMetalConfig::all()`, off ÷ on at 10M / 50M rows,
the range over one and two keys and both layouts, from `datafusion_rule_2026-09-29.csv` (measured
without the warm-up; cases `g1*`, `g2*`, `u_small`):

| aggregate | 200 groups | 10,000 | 100,000 | 1,000,000 | rows / 2 |
|---|---|---|---|---|---|
| `count(*)` | 0.76-1.39 / 0.95-1.72 | 0.97-1.53 / 0.88-1.63 | 1.41-2.54 / 1.62-2.54 | 1.91-2.63 / 2.70-3.23 | 1.61-1.90 / 0.87-0.92 |
| `sum(x)`, Float64 | 0.51-1.02 / 0.56-0.97 | 0.46-0.79 / 0.42-0.73 | 0.86-1.22 / 0.68-1.10 | 0.83-1.07 / 0.92-1.21 | 0.53-0.65 / 0.42-0.47 |
| `avg(x)`, Float64 | 0.52-1.10 / 0.59-1.00 | 0.51-0.89 / 0.45-0.79 | 1.08-1.47 / 0.92-1.42 | 1.04-1.21 / 1.14-1.41 | 0.62-0.74 / 0.49-0.53 |
| `min(x), max(x)`, Float64 | 0.26-0.51 / 0.33-0.60 | 0.32-0.56 / 0.29-0.48 | 0.85-1.07 / 0.62-0.88 | 0.62-0.76 / 0.68-0.83 | 0.37-0.44 / 0.26-0.27 |

`SELECT DISTINCT` over two int32 keys (10,000 groups): 1.60-1.80 at 10M, 1.51-1.65 at 50M, 0.72-0.86
at 1M. A `GROUP BY` over DataFusion's Parquet scan (100,000 groups, `sum` and `count(*)`,
case `p_group`): 1.38x snappy and 1.09x zstd at 10M rows, 1.57x and 1.29x at 50M.

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

- **It collects its input.** `MetalExec` reads every input partition to the end before it sorts, as
  DataFusion's `SortExec` does, but it has no spill path. The collected batches are counted in
  DataFusion's memory pool (a `MemoryConsumer` named `MetalExec`); when the pool refuses, the query
  fails with DataFusion's resources-exhausted error. The Metal buffers (the imported columns and the
  sorted result) are not counted in the pool.
- **One GPU job at a time per process.** A process-wide lock serialises every `MetalExec`, so two
  queries that both reach the GPU run their GPU parts one after the other.
- **One blocking thread per node.** ArrowMetal handles are not `Send`, so all GPU work of one
  `MetalExec` runs inside one `tokio::task::spawn_blocking` call, off DataFusion's async worker
  threads.
- **Exact row counts.** By default a node is taken only when DataFusion's statistics give its input
  an exact row count. A sort above a filter, a join or an aggregate has an estimate and is left
  unless `accept_inexact` is set; an unknown count is left unless `take_when_unknown` is set.
- **Parquet strings.** DataFusion 55.1 reads Parquet string columns as Utf8View
  (`datafusion.execution.parquet.schema_force_view_types`, on by default), and the rule does not
  carry Utf8View, so a sort whose input includes a string column read from Parquet is left.
- **macOS on Apple silicon only.** `build.rs` stops the build for any other target with a message
  naming the target; `arrowmetal-sys` refuses non-macOS targets as well.
- **DataFusion 55.1.0 only.** The dependency is pinned with `=`; DataFusion's physical-plan API
  changes between releases.
- **Not on crates.io.** The crate is used by path from a checkout.

---

## Tests and files

```bash
cd datafusion
ARROWMETAL_LIB=/path/to/libArrowMetalC.dylib cargo test                       # 20 tests + 1 doc-test
ARROWMETAL_LIB=/path/to/libArrowMetalC.dylib cargo test --test rule -- --ignored   # default config at 2M rows
```

| File | What it checks |
|---|---|
| `tests/grid.rs` | the differential grid above (1 test) |
| `tests/rule.rs` | 11 tests: the threshold and its reason, inexact statistics, the config switches, what the default leaves, unsupported shapes, `EXPLAIN`, `ORDER BY` through a projection, replaced aggregates under a partitioned join, `count(DISTINCT)`; plus 1 ignored test of the default configuration at 2,000,000 rows |
| `tests/arrowmetal_repros.rs` | 8 tests of ArrowMetal's plan runner alone, no DataFusion: totalOrder against arrow-rs, NaN and zero group keys, `count` on every path, the chunked import |

On 2026-09-29: 20 passed, 1 ignored, doc-test passed; the ignored test passed when run with `--ignored`.

| Path | What it is |
|---|---|
| `datafusion/src/rule.rs` | `ArrowMetalRule`, `ArrowMetalConfig`, `Report`, `Decision` |
| `datafusion/src/exec.rs` | `MetalExec`, `MetalOp` |
| `datafusion/src/translate.rs` | the checks on node shapes and types; predicates to ArrowMetal expressions |
| `datafusion/src/gpu.rs` | the plans sent to ArrowMetal, the import, and the run-time checks |
| `datafusion/examples/quickstart.rs` | the example above |
| `datafusion/examples/bench.rs` | the benchmark |
| `datafusion/results/datafusion_sort_warm_2026-09-29.csv` | sorts and top-k with the warm-up, 100,000 to 50M rows |
| `datafusion/results/datafusion_rule_2026-09-29.csv` | every case, rule off and on, 100,000 to 50M rows, and the Parquet cases |

Both CSVs also hold rows with `crate = before`: an earlier state of this crate, measured in the same
sessions. The tables on this page use the rows with `crate = now`.
