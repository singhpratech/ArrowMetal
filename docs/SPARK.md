# Apache Spark

`spark/` is a Spark SQL extension that runs two of Spark's operators on the Apple silicon GPU through
ArrowMetal: a sort on one key, and the partial (map-side) half of a `GROUP BY` over a Parquet or ORC scan.
Queries do not change. Every other operator, and every sort or aggregate outside the lists below, stays on
Spark's own implementation, so a query always completes with Spark's answer.

Measured on Spark 4.3.0 in local mode on an M4 Max (16 CPU cores, macOS 27.2.1) with OpenJDK 27.0.1.1 and
ArrowMetal 0.6.1. Both recorded runs: `Benchmarks/results/spark_sort_2026-11-22.txt` and
`Benchmarks/results/spark_groupby_2026-11-22.txt`, written by `spark/check.py`.

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
| `spark.arrowmetal.enabled` | `true` | `false` leaves every plan to Spark |
| `spark.arrowmetal.sort.minRows` | `1000000` | partitions with fewer rows are sorted by Spark's own ordering |
| `spark.arrowmetal.aggregate.enabled` | `true` | `false` leaves `GROUP BY` to Spark |

## How a column reaches the GPU

`ArrowMetalNative.java` calls `libArrowMetalC.dylib` directly with Java's foreign function API: no JNI and
no compiled glue. A column is laid out in off-heap memory as an Arrow C Data Interface `ArrowArray` /
`ArrowSchema` pair, handed to `am_import`, and the result comes back through `am_export`.

- **Sort** (`MetalSortExec`, replacing `SortExec`). The key of every row of a partition is written into one
  Arrow column, `am_argsort_ex2` returns the row order, and the rows are emitted in that order.
- **`GROUP BY`** (`MetalPartialAggregate`, replacing the partial `HashAggregateExec` that sits on a columnar
  scan). The scan's column batches are copied column by column into one Arrow column each, with no row
  conversion; `am_group_by_keys` and `am_group_agg_ex` produce one row per group, which Spark's exchange and
  final aggregate take as they would its own partial result.

## What it takes

| Operator | Taken | Left to Spark |
|---|---|---|
| Sort | one key of type long, int, double, float, date, timestamp or timestamp_ntz; ascending or descending; nulls first or last; `ORDER BY` and `sortWithinPartitions` | several keys, other key types |
| `GROUP BY` (partial) | one key of type int, long or date, directly over a columnar scan; `SUM` of int or double, `COUNT(*)`, `COUNT(col)`, `MIN` and `MAX` of int or long | `SUM` of long (Spark raises on overflow in ANSI mode, the GPU sum wraps), `DISTINCT`, `FILTER`, other aggregates, several keys, a child that is not a columnar scan |

The sort uses ArrowMetal's NaN-largest float order, which is Spark's: every NaN is one value above
+infinity and -0.0 ties 0.0. A partition is held in memory while it is sorted or aggregated.

## Correctness

`spark/check.py check` sorts 3,000,000 rows (4 partitions) whose keys include nulls, NaN, -0.0 and 0.0,
both infinities and the int64 extremes, and compares each sorted key column, in order, with Spark's own sort:
25 of 25 identical, over five key types, both directions, both null placements and global `ORDER BY`.

`spark/check.py agg-check` runs `SUM(int)`, `SUM(double)`, `COUNT(*)`, `COUNT(col)`, `MIN`/`MAX` of int and
long over 20,000,000 Parquet rows grouped by an int key (1,001 groups, one of them null), a long key
(1,000,000 groups) and a date key (3,650 groups), and compares every group with Spark's own aggregation:
3 of 3 identical, double sums to 1e-9 relative (the order of summation differs).

## Timings

Best of two runs each, alternating with and without the plugin, written to Spark's `noop` sink. Sorts read
a cached DataFrame; `GROUP BY` reads a 1,267 MB Parquet file of 50,000,000 rows.

**Sort, 50,000,000 rows**

| Query | Partitions | Spark | With ArrowMetal | |
|---|---:|---:|---:|---:|
| `sortWithinPartitions` long | 16 | 0.79 s | 0.50 s | 1.60x |
| `sortWithinPartitions` double | 16 | 0.69 s | 0.63 s | 1.09x |
| `ORDER BY` long | 16 | 2.53 s | 2.46 s | 1.03x |
| `ORDER BY` double | 16 | 2.87 s | 2.54 s | 1.13x |
| `sortWithinPartitions` double | 4 | 1.49 s | 1.17 s | 1.28x |
| `ORDER BY` double | 1 | 4.74 s | 3.27 s | 1.45x |

**`GROUP BY` from Parquet, 50,000,000 rows**

| Query | Partitions | Spark | With ArrowMetal | |
|---|---:|---:|---:|---:|
| `SUM, COUNT, MIN, MAX` by int key (1,001 groups) | 16 | 0.76 s | 0.55 s | 1.38x |
| `SUM(double)` by int key | 16 | 0.61 s | 0.46 s | 1.32x |
| `SUM, COUNT, MIN, MAX` by long key (1,000,000 groups) | 16 | 2.22 s | 1.51 s | 1.48x |
| `SUM(double)` by long key | 16 | 1.33 s | 0.90 s | 1.47x |
| `SUM, COUNT, MIN, MAX` by int key | 6 | 0.64 s | 0.47 s | 1.34x |
| `SUM(double)` by int key | 6 | 0.35 s | 0.27 s | 1.33x |
| `SUM, COUNT, MIN, MAX` by long key | 6 | 3.02 s | 0.87 s | 3.47x |
| `SUM(double)` by long key | 6 | 1.96 s | 0.49 s | 4.03x |

Every row is ahead. On the same file at 6 partitions, Spark's time for the four aggregates grows from 0.64 s
to 3.02 s between 1,001 and 1,000,000 groups, and the plugin's from 0.47 s to 0.87 s. A global `ORDER BY`
adds Spark's range exchange in front of the sort: at 16 partitions, 2.53 s against 0.79 s for the same long
key sorted within partitions.
