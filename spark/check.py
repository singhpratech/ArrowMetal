"""The ArrowMetal Spark plugin against Spark's own sort: the same answers, and the time each takes.

    JAVA_HOME=<JDK 22+> python spark/check.py check [rows]                  # identical to Spark's sort?
    JAVA_HOME=<JDK 22+> python spark/check.py bench [rows] [partitions]     # Spark's CPU sort vs the GPU
    JAVA_HOME=<JDK 22+> python spark/check.py agg-check [rows]              # GROUP BY from Parquet, same answers?
    JAVA_HOME=<JDK 22+> python spark/check.py agg-bench [rows] [partitions] # Spark's GROUP BY vs the GPU
    JAVA_HOME=<JDK 22+> python spark/check.py stage-bench [rows] [partitions] # filter + expressions + aggregate

Needs pyspark (4.x), spark/build/arrowmetal-spark.jar (spark/build.sh) and libArrowMetalC.dylib
(ARROWMETAL_LIB, default .build/release/libArrowMetalC.dylib). Spark runs in local mode on this Mac.
"""
import datetime as dt
import os
import platform
import subprocess
import sys
import time

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ARROWMETAL_LIB", f"{R}/.build/release/libArrowMetalC.dylib")
if "JAVA_HOME" not in os.environ:
    sys.exit("set JAVA_HOME to a JDK 22 or newer")
JAR = f"{R}/spark/build/arrowmetal-spark.jar"

from pyspark.sql import SparkSession, functions as F  # noqa: E402

mode = sys.argv[1] if len(sys.argv) > 1 else "check"
rows = int(sys.argv[2]) if len(sys.argv) > 2 else 3_000_000

spark = (SparkSession.builder.master("local[*]").appName("arrowmetal-spark")
         .config("spark.jars", JAR)
         .config("spark.sql.extensions", "org.arrowmetal.spark.ArrowMetalExtensions")
         .config("spark.driver.extraJavaOptions", "--enable-native-access=ALL-UNNAMED")
         .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "48g"))
         .config("spark.ui.enabled", "false")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
jvm = spark.sparkContext._jvm
print(f"ArrowMetal Spark plugin {mode}: {chip}, {os.cpu_count()} CPU cores, macOS {platform.mac_ver()[0]}; "
      f"Spark {spark.version} local[*], Java {jvm.System.getProperty('java.version')}, "
      f"ArrowMetal {jvm.org.arrowmetal.spark.ArrowMetalNative.version()}; "
      f"{dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")


def frame(n, parts):
    """Keys of each supported type, with nulls, NaN, -0.0/0.0, infinities and extremes mixed in."""
    df = spark.range(0, n, 1, parts)
    h = F.xxhash64("id")
    special = F.col("id") % 97
    dbl = (F.when(special == 0, F.lit(float("nan")))
           .when(special == 1, F.lit(-0.0)).when(special == 2, F.lit(0.0))
           .when(special == 3, F.lit(float("inf"))).when(special == 4, F.lit(float("-inf")))
           .when(special == 5, F.lit(None))
           .otherwise((h % 1_000_000_007).cast("double") / 7.0))
    lng = (F.when(special == 6, F.lit(None)).when(special == 7, F.lit(-(2 ** 63)).cast("long"))
           .when(special == 8, F.lit(2 ** 63 - 1).cast("long")).otherwise(h))
    return df.select("id", lng.alias("k_long"), dbl.alias("k_double"),
                     (h % 100_000).cast("int").alias("k_int"),
                     F.when(special == 9, F.lit(None)).otherwise(dbl.cast("float")).alias("k_float"),
                     F.date_add(F.lit("1970-01-01"), (h % 40_000).cast("int")).alias("k_date"))


def run(enabled, fn):
    spark.conf.set("spark.arrowmetal.enabled", str(enabled).lower())
    return fn()



def versus(q):
    """Seconds for Spark and for the plugin: 2 unrecorded warm-up runs of each, then 5 recorded runs of
    each, the side that goes first alternating. Returns {enabled: sorted times}."""
    res = {False: [], True: []}
    for i in range(7):
        for enabled in ((False, True) if i % 2 == 0 else (True, False)):
            t0 = time.perf_counter()
            run(enabled, q)
            if i >= 2:
                res[enabled].append(time.perf_counter() - t0)
    return {k: sorted(v) for k, v in res.items()}


def report(label, res, tail=""):
    c, g = res[False], res[True]
    print(f"  {label:50s} Spark {c[2]:.2f} s [{c[0]:.2f}-{c[-1]:.2f}], with ArrowMetal {g[2]:.2f} s "
          f"[{g[0]:.2f}-{g[-1]:.2f}], {c[2] / g[2]:.2f}x{tail}")


def keys(df, col):
    return df.select(col).toArrow().column(0).to_pylist()


def same(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x != y and not (isinstance(x, float) and isinstance(y, float) and x != x and y != y):
            return False
    return True


if mode in ("agg-check", "agg-bench", "stage-bench"):
    pass
elif mode == "check":
    spark.conf.set("spark.arrowmetal.sort.minRows", "1000")      # every partition takes the GPU path
    base = frame(rows, 4).cache()
    base.count()
    print(f"{rows:,} rows in 4 partitions; each sorted key column compared, in order, with Spark's own sort")
    plan_ok, bad, total = None, 0, 0
    for col in ("k_long", "k_double", "k_int", "k_float", "k_date"):
        for asc in (True, False):
            for nulls_first in (True, False):
                o = getattr(F.col(col), ("asc" if asc else "desc") + ("_nulls_first" if nulls_first else "_nulls_last"))()
                checks = [("sortWithinPartitions", lambda: base.sortWithinPartitions(o))]
                if asc and nulls_first:
                    checks.append(("ORDER BY (global)", lambda: base.orderBy(o)))
                for label, q in checks:
                    g = run(True, q)
                    if plan_ok is None:
                        plan_ok = "MetalSort" in g._jdf.queryExecution().executedPlan().toString()
                    ok = same(keys(g, col), keys(run(False, q), col))
                    bad += not ok
                    total += 1
                    order = f"{'asc ' if asc else 'desc'} nulls {'first' if nulls_first else 'last '}"
                    print(f"  {col:9s} {order}  {label:21s} {'identical' if ok else 'DIFFERENT'}")
    print(f"MetalSortExec in the plan: {plan_ok}; {total - bad} of {total} identical to Spark's sort")
elif mode == "bench":
    parts = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    base = frame(rows, parts).select("id", "k_long", "k_double").cache()
    base.count()
    print(f"{rows:,} rows, {parts} partitions, cached; median [min-max] of 5 runs each after 2 warm-up runs, the first side alternating, written to Spark's noop sink")
    for col in ("k_long", "k_double"):
        for label, q in (("sortWithinPartitions", lambda: base.sortWithinPartitions(col)),
                         ("ORDER BY (global)", lambda: base.orderBy(col))):
            report(f"{col} {label}", versus(lambda: q().write.format("noop").mode("overwrite").save()))


def parquet(n):
    """A Parquet file of n rows with three key columns and four value columns, nulls included."""
    path = os.environ.get("SPARK_DATA", f"{R}/spark/build/data") + f"/groupby_{n}.parquet"
    if not os.path.exists(path):
        h = F.xxhash64("id")
        u = F.abs(h)
        (spark.range(0, n, 1, 16).select(
            F.when(F.col("id") % 101 == 0, F.lit(None)).otherwise((u % 1000).cast("int")).alias("k_int"),
            (u % 1_000_000).alias("k_long"),
            F.date_add(F.lit("2000-01-01"), (u % 3650).cast("int")).alias("k_date"),
            F.when(F.col("id") % 13 == 0, F.lit(None)).otherwise((h % 100_000).cast("int")).alias("v_int"),
            F.when(F.col("id") % 17 == 0, F.lit(None)).otherwise((h % 1_000_000_007).cast("double") / 3.0).alias("v_double"),
            F.when(F.col("id") % 19 == 0, F.lit(None)).otherwise(F.xxhash64(F.col("id") + 1)).alias("v_long"))
         .write.mode("overwrite").parquet(path))
    return path


AGG = ("SUM(v_int) s_int, SUM(v_double) s_double, COUNT(*) n, COUNT(v_int) n_int, "
       "MIN(v_int) mn_int, MAX(v_int) mx_int, MIN(v_long) mn_long, MAX(v_long) mx_long")


def agg_rows(df):
    out = {}
    for r in df.collect():
        d = r.asDict()
        k = d.pop(list(d)[0])
        out[k] = d
    return out


def agg_same(a, b):
    if a.keys() != b.keys():
        return f"group keys differ: {len(a)} vs {len(b)} groups"
    for k in a:
        for c, x in a[k].items():
            y = b[k][c]
            if isinstance(x, float) and isinstance(y, float):
                if not (x == y or abs(x - y) <= 1e-9 * max(abs(x), abs(y))):
                    return f"group {k} {c}: {x!r} vs {y!r}"
            elif x != y:
                return f"group {k} {c}: {x!r} vs {y!r}"
    return None


if mode == "agg-check":
    path = parquet(rows)
    spark.read.parquet(path).createOrReplaceTempView("t")
    print(f"{rows:,} rows of Parquet; each GROUP BY compared group by group with Spark's own aggregation "
          "(double sums to 1e-9 relative: the summation order differs)")
    bad = total = 0
    plan_ok = None
    for key in ("k_int", "k_long", "k_date"):
        q = lambda: spark.sql(f"SELECT {key}, {AGG} FROM t GROUP BY {key}")
        g = run(True, q)
        gr = agg_rows(g)
        if plan_ok is None:      # adaptive execution: the final plan exists once the query has run
            plan_ok = "MetalStage" in g._jdf.queryExecution().executedPlan().toString()
        cr = agg_rows(run(False, q))
        d = agg_same(gr, cr)
        bad += d is not None
        total += 1
        print(f"  GROUP BY {key:7s} {len(gr):>9,} groups: {'identical' if d is None else 'DIFFERENT: ' + d}")
    print(f"MetalStage in the plan: {plan_ok}; {total - bad} of {total} identical to Spark")
elif mode == "agg-bench":
    parts = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    path = parquet(rows)
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs)
    spark.conf.set("spark.sql.files.maxPartitionBytes", str(max(1 << 20, size // parts + 1)))
    spark.conf.set("spark.sql.files.minPartitionNum", str(parts))   # else the split size follows the core count
    spark.read.parquet(path).createOrReplaceTempView("t")
    nparts = spark.read.parquet(path).rdd.getNumPartitions()
    print(f"{rows:,} rows of Parquet ({size / 1e6:.0f} MB) read in {nparts} partitions; best of 2 runs each, "
          "alternating, written to Spark's noop sink")
    for key in ("k_int", "k_long"):
        for label, sql in ((f"SUM, COUNT, MIN, MAX by {key}", f"SELECT {key}, {AGG} FROM t GROUP BY {key}"),
                           (f"SUM(v_double) by {key}", f"SELECT {key}, SUM(v_double) FROM t GROUP BY {key}")):
            report(label, versus(lambda: spark.sql(sql).write.format("noop").mode("overwrite").save()))


def lineitem(n):
    """A lineitem-shaped Parquet file: quantities, prices, discounts, dates and two small keys."""
    path = os.environ.get("SPARK_DATA", f"{R}/spark/build/data") + f"/lineitem_{n}.parquet"
    if not os.path.exists(path):
        h = F.abs(F.xxhash64("id"))
        (spark.range(0, n, 1, 16).select(
            (h % 50 + 1).cast("double").alias("quantity"),
            ((h % 9_000_000) / 100.0 + 900.0).alias("price"),
            ((h % 11) / 100.0).alias("discount"),
            ((F.abs(F.xxhash64(F.col("id") + 3)) % 9) / 100.0).alias("tax"),
            F.date_add(F.lit("1992-01-01"), (F.abs(F.xxhash64(F.col("id") + 5)) % 2526).cast("int")).alias("shipdate"),
            (h % 4).cast("int").alias("flag"),
            (F.abs(F.xxhash64(F.col("id") + 9)) % 100_000).alias("partkey"))
         .write.mode("overwrite").parquet(path))
    return path


if mode == "stage-bench":
    parts = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    path = lineitem(rows)
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs)
    spark.conf.set("spark.sql.files.maxPartitionBytes", str(max(1 << 20, size // parts + 1)))
    spark.conf.set("spark.sql.files.minPartitionNum", str(parts))
    spark.read.parquet(path).createOrReplaceTempView("lineitem")
    nparts = spark.read.parquet(path).rdd.getNumPartitions()
    print(f"{rows:,} rows of Parquet ({size / 1e6:.0f} MB) read in {nparts} partitions; best of 2 runs each, "
          "alternating; both answers compared")
    queries = [
        ("filtered revenue, no GROUP BY (TPC-H Q6 shape)",
         "SELECT SUM(price * discount) AS revenue FROM lineitem WHERE shipdate >= DATE'1994-01-01' "
         "AND shipdate < DATE'1995-01-01' AND discount >= 0.05 AND discount <= 0.07 AND quantity < 24.0"),
        ("pricing summary by flag (TPC-H Q1 shape, int key)",
         "SELECT flag, SUM(quantity), SUM(price), SUM(price * (1.0 - discount)), "
         "SUM(price * (1.0 - discount) * (1.0 + tax)), AVG(quantity), AVG(price), AVG(discount), COUNT(*) "
         "FROM lineitem WHERE shipdate <= DATE'1998-09-02' GROUP BY flag"),
        ("revenue by part, 100,000 groups",
         "SELECT partkey, SUM(price * (1.0 - discount)) AS rev, COUNT(*) AS n FROM lineitem "
         "WHERE quantity > 10.0 GROUP BY partkey"),
    ]
    for label, sql in queries:
        answers = {e: run(e, lambda: spark.sql(sql).collect()) for e in (False, True)}
        res = versus(lambda: spark.sql(sql).collect())
        a = {tuple(r)[0] if len(r) > 1 else 0: tuple(r) for r in answers[False]}
        b = {tuple(r)[0] if len(r) > 1 else 0: tuple(r) for r in answers[True]}
        ok = a.keys() == b.keys() and all(
            all((x == y) or (isinstance(x, float) and isinstance(y, float) and abs(x - y) <= 1e-9 * max(abs(x), abs(y)))
                for x, y in zip(a[k], b[k])) for k in a)
        report(label, res, f"  ({'same answer' if ok else 'DIFFERENT ANSWER'})")
spark.stop()
