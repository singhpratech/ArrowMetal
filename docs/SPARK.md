# Apache Spark

`spark/` is a Spark SQL extension that runs Spark's work on the Apple silicon GPU through ArrowMetal.
Queries do not change. Two kinds of operator are replaced:

- **The stage above a Parquet or ORC scan** (`MetalStage`): any chain of filters (`WHERE`) and projections
  (`SELECT` expressions) directly over a columnar scan, optionally topped by the partial (map-side) half of
  a `GROUP BY` or of an ungrouped aggregate, runs as one ArrowMetal plan per partition.
- **A sort on one key** (`MetalSortExec`), for `ORDER BY` and `sortWithinPartitions`.

Whatever does not translate exactly stays on Spark's own operators, so a query always completes with
Spark's answer.

Measured on Spark 4.3.0 in local mode on an M4 Max (16 CPU cores, macOS 27.2.1) with OpenJDK 27.0.1.1 and
ArrowMetal 0.6.1. Recorded runs, written by `spark/check.py` and `spark/fuzz.py`:
`Benchmarks/results/spark_fuzz_2026-11-22.txt`, `spark_stage_2026-11-22.txt`,
`spark_groupby_2026-11-22.txt` and `spark_sort_2026-11-22.txt`.

## Build and enable

Needs a JDK 22 or newer (the plugin calls the C ABI through Java's foreign function API) and a Python with
`pyspark` installed, whose `jars/` folder holds Spark's classes and the Scala 2.13 compiler the plugin is
compiled with.

```bash
JAVA_HOME=<JDK 22+> PYTHON=<python with pyspark> spark/build.sh      # spark/build/arrowmetal-spark.jar
```

```python
spark = (SparkSession.builder
         .config("spark.jars", "spark/build/arrowmetal-spark.jar")
         .config("spark.sql.extensions", "org.arrowmetal.spark.ArrowMetalExtensions")
         .config("spark.driver.extraJavaOptions", "--enable-native-access=ALL-UNNAMED")
         .getOrCreate())
```

The library is found through the `ARROWMETAL_LIB` environment variable (or the `arrowmetal.lib` system
property), the path of `libArrowMetalC.dylib`.

| Setting | Default | Effect |
|---|---|---|
| `spark.arrowmetal.enabled` | `true` | `false` leaves every query to Spark |
| `spark.arrowmetal.stage.enabled` | `true` | `false` leaves filters, projections and aggregates to Spark |
| `spark.arrowmetal.stage.batchRows` | `16000000` | rows a GPU chunk holds; a chunk also ends at 1 GiB of off-heap columns |
| `spark.arrowmetal.stage.ungrouped` | `false` | `true` takes aggregates without `GROUP BY` too |
| `spark.arrowmetal.stage.rows` | `false` | `true` takes filter/projection chains with no aggregate on them |
| `spark.arrowmetal.sort.enabled` | `false` | `true` sorts one-key partitions on the GPU |
| `spark.arrowmetal.sort.minRows` | `1000000` | with sorting on, smaller partitions are sorted by Spark's own ordering |

## How a column reaches the GPU

`ArrowMetalNative.java` calls `libArrowMetalC.dylib` directly with Java's foreign function API: no JNI and
no compiled glue. The stage copies the scan's column batches, column by column, into off-heap Arrow
columns (C Data Interface `ArrowArray` / `ArrowSchema` pairs, `am_import`), runs one lazy-engine plan over
them (`am_plan_run`: filter, `select`, `group_by` or `aggregate`, written by `SparkToArrowMetal.scala`) and
reads the result back through `am_export`. A string column that is only carried to the output stays on
the JVM and is joined back by row number. The sort writes the key of every row into one Arrow column and
emits the rows in the order `am_argsort_ex2` returns. The columns are copied off-heap in chunks;
at Spark's default 1 GB driver heap a 100,000-group query over 50,000,000 rows completes with the plugin and
without it (`Benchmarks/results/spark_heap_2026-11-22.txt`).

## What it takes

| | Taken | Left to Spark |
|---|---|---|
| Column types | byte, short, int, long, float, double, boolean, string, date, timestamp; timestamp_ntz is taken and no test covers it | decimal, binary, nested types |
| Predicates | `=`, `<>`, `<`, `<=`, `>`, `>=` on numbers, dates, timestamps and booleans; `=` of a string with a literal; `AND`, `OR`, `NOT`; `IS [NOT] NULL`; `IN` lists of literals; `LIKE 'abc'`, `'abc%'`, `'%abc%'`; `startswith`, `contains` | string ordering, `LIKE '%abc'` and patterns with `_` |
| Expressions | `+ - *` and unary minus on floats, and on integers when `spark.sql.ansi.enabled` is false; `/` by a non-zero literal; `abs`; widening casts; `coalesce`; `IF`; `CASE WHEN` | integer arithmetic under ANSI (Spark raises on overflow, the GPU wraps), division by a column, narrowing casts, string-valued expressions, everything else |
| Aggregates (partial) | `SUM` of integers and floats (of longs only without ANSI), `COUNT(*)`, `COUNT(col)`, `MIN` and `MAX` of integers and dates, `AVG` of integers and floats; with any number of non-string keys, or one string key, or none | `MIN`/`MAX` of floats, `DISTINCT`, `FILTER`, several keys with a string among them, other aggregates |
| Sort | one key of type long, int, double, float, date, timestamp or timestamp_ntz, either direction, nulls first or last | several keys, other key types |

The translation keeps Spark's semantics where Arrow's differ:

- **Floats.** Spark's comparisons put NaN equal to NaN and above every other value; each float comparison is
  written that way, with a null input giving null. The sort uses ArrowMetal's NaN-largest order, Spark's
  own: every NaN one value above +infinity, -0.0 tying 0.0.
- **Nulls.** `AND`/`OR` are three-valued. `IF` and `CASE WHEN` take the `ELSE` branch on a null condition.
  `x IN (...)` is null when `x` is null.
- **Partial aggregate buffers.** `COUNT` of a group with only nulls is 0; `AVG`'s running sum starts at 0.0;
  an ungrouped aggregate over an empty partition emits Spark's initial buffer.

## Correctness

`spark/fuzz.py` writes a 1,000,000-row Parquet table holding nulls, NaN, -0.0 and 0.0, both infinities, the
int32 extremes, non-ASCII strings and byte, short, date and timestamp columns, generates random queries over
it, runs each with and without the plugin, half under ANSI, and compares the rows as multisets: double sums
and averages to 1e-9 relative (the summation order differs), everything else exactly, -0.0 apart from 0.0.
Three runs, 1,300 queries on the build of `spark_fuzz_2026-11-22_final.txt`: 1,277 with identical rows, 23
where both raise, 0 different, 1,081 with a `MetalStage` in the executed plan. The fuzz of an earlier build
found 76 of 300 queries failing, all on byte or short columns read back from the GPU. The chunk-size
change after it ran 100 more (`spark_fuzz_2026-11-22_smoke16m.txt`).

`spark/check.py check` compares 25 sorted key columns, in order, with Spark's sort (five key types, both
directions, both null placements, global `ORDER BY`): 25 of 25 identical. `spark/check.py agg-check` compares
three `GROUP BY` queries over 20,000,000 Parquet rows group by group: 3 of 3 identical.

## Timings

Median of five runs a side after two warm-up runs, the side that goes first alternating; 50,000,000 rows,
M4 Max, 16 cores, local mode, everything on. Each side at the better of the partition counts tried (16 and
6; 16, 4 and 1 for the sorts). Files: `spark_groupby_2026-11-22_16m.txt`, `spark_stage_2026-11-22_16m.txt`,
`spark_sort_2026-11-22_final.txt` under `Benchmarks/results/`.

| Query | Spark | With ArrowMetal | |
|---|---:|---:|---:|
| `SUM, COUNT, MIN, MAX`, 1,000,000 groups | 2.35 s | 1.01 s | 2.3x |
| `SUM(double)`, 1,000,000 groups | 1.40 s | 0.53 s | 2.6x |
| revenue by part, 100,000 groups | 0.88 s | 0.54 s | 1.6x |
| `SUM, COUNT, MIN, MAX`, 1,001 groups | 0.55 s | 0.49 s | 1.1x |
| `SUM(double)`, 1,001 groups | 0.31 s | 0.29 s | 1.1x |
| pricing summary, 4 groups (TPC-H Q1 shape) | 0.39 s | 0.40 s | 1.0x |
| filtered revenue, no `GROUP BY` (TPC-H Q6 shape), off by default | 0.13 s | 0.23 s | 0.6x |
| `sortWithinPartitions` long, off by default | 0.58 s | 0.45 s | 1.3x |
| `sortWithinPartitions` double, off by default | 0.57 s | 0.69 s | 0.8x |
| `ORDER BY` long, off by default | 2.59 s | 2.69 s | 1.0x |
| `ORDER BY` double, off by default | 2.64 s | 2.98 s | 0.9x |

Runs of the same query on the same day moved: the Q1 shape measured 1.4x in an earlier build's run
(`spark_stage_2026-11-22_warm.txt`) and 1.0x in this one, and at 16 partitions the 1,001-group rows are 1.10x
and 0.97x. The earlier recordings (`spark_*_2026-11-22.txt`, best of two, Spark first on a cold JVM; `_warm`;
`_final`) are kept beside these.
