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
| `spark.arrowmetal.stage.rows` | `true` | `false` takes a filter/projection chain only when an aggregate sits on it |
| `spark.arrowmetal.sort.minRows` | `1000000` | partitions with fewer rows are sorted by Spark's own ordering |

## How a column reaches the GPU

`ArrowMetalNative.java` calls `libArrowMetalC.dylib` directly with Java's foreign function API: no JNI and
no compiled glue. The stage copies the scan's column batches, column by column, into off-heap Arrow
columns (C Data Interface `ArrowArray` / `ArrowSchema` pairs, `am_import`), runs one lazy-engine plan over
them (`am_plan_run`: filter, `select`, `group_by` or `aggregate`, written by `SparkToArrowMetal.scala`) and
reads the result back through `am_export`. A string column that is only carried to the output stays on
the JVM and is joined back by row number. The sort writes the key of every row into one Arrow column and
emits the rows in the order `am_argsort_ex2` returns. A partition is held in memory while it is processed.

## What it takes

| | Taken | Left to Spark |
|---|---|---|
| Column types | byte, short, int, long, float, double, boolean, string, date, timestamp, timestamp_ntz (the fuzz table has no byte, short or timestamp column) | decimal, binary, nested types |
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

`spark/fuzz.py` writes a 1,000,000-row Parquet table holding nulls, NaN, -0.0 and 0.0, both infinities,
the int32 extremes and non-ASCII strings, generates random queries over it (filters nested up to two
levels, projections, grouped and ungrouped aggregates), runs each with and without the plugin, half under
ANSI and half not, and compares the rows as multisets (double sums and averages to 1e-9 relative, the
summation order being different; everything else exactly). Two runs: 300 queries (seed 1) and 500
queries (seed 2), **800 of 800 identical to Spark**, 646 of them with a `MetalStage` in the executed plan.

`spark/check.py check` compares 25 sorted key columns, in order, with Spark's sort (five key types, both
directions, both null placements, global `ORDER BY`): 25 of 25 identical. `spark/check.py agg-check` compares
three `GROUP BY` queries over 20,000,000 Parquet rows group by group: 3 of 3 identical.

## Timings

Best of two runs each, alternating with and without the plugin. Local mode, 16 cores.

**Filter, expressions and aggregate in one stage**, a 698 MB Parquet file of 50,000,000 lineitem-shaped
rows, both answers compared:

| Query | Partitions | Spark | With ArrowMetal | |
|---|---:|---:|---:|---:|
| filtered revenue, no `GROUP BY` (TPC-H Q6 shape) | 16 | 0.24 s | 0.20 s | 1.22x |
| pricing summary by flag (TPC-H Q1 shape, int key) | 16 | 0.51 s | 0.26 s | 1.96x |
| revenue by part, 100,000 groups | 16 | 0.79 s | 0.78 s | 1.01x |
| filtered revenue, no `GROUP BY` | 6 | 0.27 s | 0.25 s | 1.09x |
| pricing summary by flag | 6 | 0.58 s | 0.36 s | 1.60x |
| revenue by part, 100,000 groups | 6 | 0.87 s | 0.46 s | 1.89x |

**`GROUP BY` from Parquet**, a 1,267 MB file of 50,000,000 rows, written to Spark's `noop` sink:

| Query | Partitions | Spark | With ArrowMetal | |
|---|---:|---:|---:|---:|
| `SUM, COUNT, MIN, MAX` by int key (1,001 groups) | 16 | 0.61 s | 0.61 s | 0.99x |
| `SUM(double)` by int key | 16 | 0.63 s | 0.49 s | 1.29x |
| `SUM, COUNT, MIN, MAX` by long key (1,000,000 groups) | 16 | 2.21 s | 1.47 s | 1.50x |
| `SUM(double)` by long key | 16 | 1.55 s | 0.99 s | 1.56x |
| `SUM, COUNT, MIN, MAX` by int key | 6 | 0.61 s | 0.46 s | 1.31x |
| `SUM(double)` by int key | 6 | 0.37 s | 0.29 s | 1.26x |
| `SUM, COUNT, MIN, MAX` by long key | 6 | 3.20 s | 0.95 s | 3.38x |
| `SUM(double)` by long key | 6 | 1.73 s | 0.53 s | 3.24x |

**Sort, 50,000,000 cached rows**, written to Spark's `noop` sink:

| Query | Partitions | Spark | With ArrowMetal | |
|---|---:|---:|---:|---:|
| `sortWithinPartitions` long | 16 | 0.79 s | 0.50 s | 1.60x |
| `sortWithinPartitions` double | 16 | 0.69 s | 0.63 s | 1.09x |
| `ORDER BY` long | 16 | 2.53 s | 2.46 s | 1.03x |
| `ORDER BY` double | 16 | 2.87 s | 2.54 s | 1.13x |
| `sortWithinPartitions` double | 4 | 1.49 s | 1.17 s | 1.28x |
| `ORDER BY` double | 1 | 4.74 s | 3.27 s | 1.45x |

The largest gains are on many groups: on the same file at 6 partitions, Spark's time for the four
aggregates grows from 0.61 s to 3.20 s between 1,001 and 1,000,000 groups, and the plugin's from 0.46 s to
0.95 s. A global `ORDER BY` adds Spark's range exchange in front of the sort: at 16 partitions, 2.53 s
against 0.79 s for the same long key sorted within partitions.
