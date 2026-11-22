"""Random-query conformance for the ArrowMetal Spark plugin: every query runs with the plugin and
without it, and the answers are compared.

    JAVA_HOME=<JDK 22+> python spark/fuzz.py [queries] [seed] [rows]

A table with nulls, NaN, -0.0, infinities, integer extremes and non-ASCII strings in it is written as
Parquet; queries are generated from filters, projections and grouped and ungrouped aggregates over it,
half with spark.sql.ansi.enabled true and half false. Rows are compared as multisets; double sums and
averages to 1e-9 relative (the summation order differs), everything else exactly, NaN equal to NaN.
A query counts as taken when the executed plan contains a MetalStage.
"""
import datetime as dt
import math
import os
import platform
import random
import subprocess
import sys

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ARROWMETAL_LIB", f"{R}/.build/release/libArrowMetalC.dylib")
if "JAVA_HOME" not in os.environ:
    sys.exit("set JAVA_HOME to a JDK 22 or newer")

from pyspark.sql import SparkSession, functions as F  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
ROWS = int(sys.argv[3]) if len(sys.argv) > 3 else 1_000_000

spark = (SparkSession.builder.master("local[*]").appName("arrowmetal-spark-fuzz")
         .config("spark.jars", f"{R}/spark/build/arrowmetal-spark.jar")
         .config("spark.sql.extensions", "org.arrowmetal.spark.ArrowMetalExtensions")
         .config("spark.driver.extraJavaOptions", "--enable-native-access=ALL-UNNAMED")
         .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "24g"))
         .config("spark.ui.enabled", "false")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
jvm = spark.sparkContext._jvm
chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
print(f"ArrowMetal Spark plugin fuzz: {chip}, macOS {platform.mac_ver()[0]}; Spark {spark.version}, "
      f"Java {jvm.System.getProperty('java.version')}, ArrowMetal {jvm.org.arrowmetal.spark.ArrowMetalNative.version()}; "
      f"{N} queries, seed {SEED}, {ROWS:,} rows; {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")

path = os.environ.get("SPARK_DATA", f"{R}/spark/build/data") + f"/fuzz2_{ROWS}.parquet"
if not os.path.exists(path):
    h = F.xxhash64("id")
    u = F.abs(h)
    m = F.col("id") % 50
    d = (F.when(m == 0, F.lit(float("nan"))).when(m == 1, F.lit(-0.0)).when(m == 2, F.lit(0.0))
         .when(m == 3, F.lit(float("inf"))).when(m == 4, F.lit(float("-inf"))).when(m == 5, F.lit(None))
         .otherwise((h % 2_000_001).cast("double") / 8.0))
    strings = F.array(*[F.lit(s) for s in ["", "a", "apple", "banana", "Apple", "an", "日本", "naïve", "ab%c", "x y", "zz"]])
    (spark.range(0, ROWS, 1, 8).select(
        F.when(m == 6, F.lit(None)).when(m == 7, F.lit(-2147483648)).when(m == 8, F.lit(2147483647))
         .otherwise((h % 100_000).cast("int")).cast("int").alias("i"),
        F.when(m == 9, F.lit(None)).otherwise(F.xxhash64(F.col("id") + 7) % 1_000_000_000_000).alias("l"),
        d.alias("d"),
        F.when(m == 10, F.lit(None)).otherwise(d.cast("float")).alias("f"),
        F.when(m == 11, F.lit(None)).otherwise(u % 2 == 0).alias("b"),
        F.when(m == 12, F.lit(None)).otherwise(F.element_at(strings, (u % 11 + 1).cast("int"))).alias("s"),
        F.when(m == 13, F.lit(None)).otherwise(F.date_add(F.lit("2000-01-01"), (u % 9000).cast("int"))).alias("dt"),
        ((h % 2_000_001).cast("double") / 8.0).alias("x"),          # finite: sums that are not NaN
        F.when(m == 14, F.lit(None)).otherwise((h % 128).cast("byte")).alias("y"),
        F.when(m == 15, F.lit(None)).otherwise((h % 32768).cast("short")).alias("sh"),
        F.when(m == 16, F.lit(None)).otherwise(F.timestamp_seconds(u % 1_700_000_000)).alias("ts"),
        (u % 20).cast("int").alias("k1"),
        (u % 1000).alias("k2"))
     .write.mode("overwrite").parquet(path))
spark.read.parquet(path).createOrReplaceTempView("t")

rnd = random.Random(SEED)
NUM = {"i": "int", "l": "long", "d": "double", "f": "float", "x": "double", "y": "int", "sh": "int", "k1": "int", "k2": "long"}
LIT = {"int": ["0", "-1", "7", "1000", "-50000"], "long": ["0", "123456789", "-5"],
       "double": ["0.0", "-0.0", "1.5", "-2.25", "100000.0"], "float": ["0.0", "1.5", "-3.0"]}


def lit(c):
    return rnd.choice(LIT[NUM[c]])


def pred(depth=0):
    r = rnd.random()
    if depth < 2 and r < 0.25:
        return f"({pred(depth + 1)} {rnd.choice(['AND', 'OR'])} {pred(depth + 1)})"
    if depth < 2 and r < 0.32:
        return f"NOT ({pred(depth + 1)})"
    kind = rnd.choice(["cmp", "cmp", "cmp", "colcmp", "null", "in", "str", "bool", "date"])
    op = rnd.choice(["=", "<", "<=", ">", ">=", "<>"])
    if kind == "cmp":
        c = rnd.choice(list(NUM))
        return f"{c} {op} {lit(c)}"
    if kind == "colcmp":
        a, b = rnd.choice([("i", "k1"), ("l", "k2"), ("d", "d"), ("d", "f"), ("f", "f")])
        return f"{a} {op} {b}" if a != "d" or b != "f" else f"d {op} CAST(f AS DOUBLE)"
    if kind == "null":
        return f"{rnd.choice(list(NUM) + ['s', 'b', 'dt'])} IS {rnd.choice(['', 'NOT '])}NULL"
    if kind == "in":
        c = rnd.choice(["i", "k1", "k2", "s"])
        vals = ["'apple'", "'an'", "'日本'", "''"] if c == "s" else ["0", "3", "7", "19", "-1"]
        return f"{c} IN ({', '.join(rnd.sample(vals, 3))})"
    if kind == "str":
        return rnd.choice(["s = 'apple'", "s <> 'an'", "s LIKE 'a%'", "s LIKE '%an%'", "s LIKE 'apple'",
                           "startswith(s, 'ba')", "contains(s, 'p')", "s LIKE '%e'"])
    if kind == "bool":
        return rnd.choice(["b", "NOT b", "b = true", "b IS TRUE"])
    if rnd.random() < 0.3:
        return f"ts {op} TIMESTAMP'{rnd.choice(['1980-01-01 00:00:00', '2001-09-09 01:46:40'])}'"
    return f"dt {op} DATE'{rnd.choice(['2000-01-01', '2010-06-15', '2024-12-31'])}'"


def proj():
    return rnd.choice([
        "d * 2.5", "d + 1.25", "-d", "abs(d)", "d / 4.0", "CAST(i AS DOUBLE) + d", "f * 2.0",
        "i + 1", "l - k2", "k1 * 3", "-i", "CAST(k1 AS BIGINT)", "coalesce(d, 0.0)", "coalesce(i, k1)",
        "IF(d > 0.0, d, -1.0)", "CASE WHEN k1 < 5 THEN 'low' WHEN k1 < 15 THEN 'mid' ELSE s END",
        "IF(b, i, k1)", "s", "dt", "b", "l", "k1 < 10 AND b", "d IS NULL", "isnan(d)", "i % 7", "x * 2.5", "x / 3.0", "y", "sh", "ts", "coalesce(y, sh)"])


def agg():
    return rnd.choice(["SUM(i)", "SUM(d)", "SUM(f)", "SUM(l)", "SUM(k1)", "COUNT(*)", "COUNT(d)", "COUNT(s)",
                       "MIN(i)", "MAX(l)", "MIN(dt)", "MAX(k2)", "AVG(i)", "AVG(d)", "AVG(k1)",
                       "MIN(d)", "SUM(x)", "SUM(x)", "AVG(x)", "SUM(x * 1.5)", "SUM(y)", "AVG(sh)", "MIN(y)", "MAX(sh)", "COUNT(ts)", "SUM(d * 2.0)", "COUNT(DISTINCT k1)"])


def query():
    form = rnd.choice(["select", "group", "group", "global"])
    where = f" WHERE {pred()}" if rnd.random() < 0.85 else ""
    if form == "select":
        cols = ", ".join(f"{proj()} AS p{j}" for j in range(rnd.randint(1, 4)))
        return form, f"SELECT {cols} FROM t{where}"
    aggs = ", ".join(f"{agg()} AS a{j}" for j in range(rnd.randint(1, 4)))
    if form == "global":
        return form, f"SELECT {aggs} FROM t{where}"
    keys = rnd.choice([["k1"], ["k2"], ["k1", "k2"], ["s"], ["b"], ["dt"], ["k1", "s"], ["i"], ["y"], ["sh", "k1"], ["ts"], ["x"], ["d"], ["f"]])
    return form, f"SELECT {', '.join(keys)}, {aggs} FROM t{where} GROUP BY {', '.join(keys)}"


def norm(v):
    if isinstance(v, float):
        return ("nan",) if math.isnan(v) else ("f", v.hex())      # hex keeps -0.0 apart from 0.0
    return ("v", v)


def close(a, b):
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        if math.isinf(a) or math.isinf(b):
            return a == b
        return a == b or abs(a - b) <= 1e-9 * max(abs(a), abs(b))
    return a == b


def same(x, y, tolerant):
    if len(x) != len(y):
        return f"{len(x)} rows vs {len(y)}"
    kx = sorted(x, key=lambda r: [repr(norm(v)) for v in r])
    ky = sorted(y, key=lambda r: [repr(norm(v)) for v in r])
    for rx, ry in zip(kx, ky):
        for a, b in zip(rx, ry):
            if not (close(a, b) if tolerant else (norm(a) == norm(b))):
                return f"row {rx} vs {ry}"
    return None


taken = differ = errors = both = 0
by_form = {}
for qn in range(N):
    form, sql = query()
    ansi = qn % 2 == 0
    spark.conf.set("spark.sql.ansi.enabled", str(ansi).lower())
    try:
        spark.conf.set("spark.arrowmetal.enabled", "false")
        try:
            cpu = [tuple(r) for r in spark.sql(sql).collect()]
            cpu_err = None
        except Exception as e:  # noqa: BLE001
            cpu, cpu_err = None, type(e).__name__
        spark.conf.set("spark.arrowmetal.enabled", "true")
        df = spark.sql(sql)
        try:
            gpu = [tuple(r) for r in df.collect()]
            gpu_err = None
        except Exception as e:  # noqa: BLE001
            gpu, gpu_err = None, f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
        used = "MetalStage" in df._jdf.queryExecution().executedPlan().toString()
    except Exception as e:  # noqa: BLE001
        errors += 1
        print(f"  [{qn}] harness error {e!r}: {sql}")
        continue
    taken += used
    f = by_form.setdefault(form, [0, 0])
    f[0] += 1
    f[1] += used
    if cpu_err or gpu_err:
        both += bool(cpu_err) and bool(gpu_err)
        if bool(cpu_err) != bool(gpu_err):
            differ += 1
            print(f"  [{qn}] DIFFERENT (ansi={ansi}, gpu={used}): Spark {cpu_err or 'ok'}, plugin {gpu_err or 'ok'}: {sql}")
        continue
    d = same(cpu, gpu, tolerant=form != "select")
    if d:
        differ += 1
        print(f"  [{qn}] DIFFERENT (ansi={ansi}, gpu={used}): {d}: {sql}")
print(f"{N} queries: {N - differ - errors - both} with identical rows, {both} where both raise, {differ} different, {errors} harness errors; "
      f"{taken} ran a MetalStage on the GPU ({', '.join(f'{k} {v[1]}/{v[0]}' for k, v in sorted(by_form.items()))})")
spark.stop()
