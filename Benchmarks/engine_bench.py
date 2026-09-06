"""The lazy query engine vs Polars lazy, DuckDB and pyarrow on the same in-process data.

Eight query shapes, TPC-H flavoured: filtered aggregate, filter + group-by + sort + limit, multi-key
group-by, a projection chain, a 10M x 1M hash join with an aggregate on top, a semi join, a window
over partitions, and a 50M-row as-of join against a 1M-row quote table.

Every engine gets the same Arrow buffers, imported once outside the timed loop; ArrowMetal's columns
are imported into Metal shared memory once and reused, which is what a real caller holding GPU-
resident data would do. Wall time is the best of `iters` runs; CPU time is process CPU over the same
run, so a GPU engine's "how much of the machine did this cost" is visible next to its latency.

Usage: PYTHONPATH=python python Benchmarks/engine_bench.py [rows] [iterations]
Requires .build/release/libArrowMetalC.dylib (swift build -c release --product ArrowMetalC).
"""
import os
import resource
import sys
import time

import numpy as np
import polars as pl
import pyarrow as pa

import arrowmetal as am

try:
    import duckdb
except ImportError:                                    # pragma: no cover - optional
    duckdb = None

rows = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000_000
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 5
rng = np.random.default_rng(1234)
results = []


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def bench(section, label, fn):
    try:
        fn()
    except Exception as e:                             # a missing engine must not stop the run
        print(f"  {label:<44} skipped: {type(e).__name__}: {e}")
        return
    best, best_cpu = float("inf"), float("inf")
    for _ in range(iters):
        c0, t0 = cpu_seconds(), time.perf_counter()
        fn()
        wall, cpu = time.perf_counter() - t0, cpu_seconds() - c0
        if wall < best:
            best, best_cpu = wall, cpu
    print(f"  {label:<44} {best * 1000:9.2f} ms   {best_cpu * 1000:9.1f} CPU-ms")
    results.append((section, label, best * 1000, best_cpu * 1000))


def header(s):
    print(f"\n{s}")


print(f"ArrowMetal {am.version()} on {am.device_name()} vs polars {pl.__version__} "
      f"({pl.thread_pool_size()} threads), pyarrow {pa.__version__}"
      + (f", duckdb {duckdb.__version__}" if duckdb else ", duckdb not installed")
      + f"; rows={rows:,}, best of {iters}")

# --------------------------------------------------------------------------------------------------
# data

region_np = rng.integers(0, 200, size=rows, dtype=np.int32)
sub_np = rng.integers(0, 50, size=rows, dtype=np.int32)
amount_np = (rng.random(rows, dtype=np.float32) * 2000 - 500).astype(np.float32)
qty_np = rng.integers(-10, 40, size=rows, dtype=np.int64)

fact = pa.table({
    "region": pa.array(region_np),
    "sub": pa.array(sub_np),
    "amount": pa.array(amount_np),
    "qty": pa.array(qty_np),
})
pl_fact = pl.from_arrow(fact)
am_fact = {n: am.array(fact.column(n).combine_chunks()) for n in fact.column_names}

dim_rows = 200
dim = pa.table({
    "region": pa.array(np.arange(dim_rows, dtype=np.int32)),
    "weight": pa.array(rng.random(dim_rows, dtype=np.float32) * 2),
})
pl_dim = pl.from_arrow(dim)
am_dim = {n: am.array(dim.column(n).combine_chunks()) for n in dim.column_names}

con = None
if duckdb:
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={pl.thread_pool_size()}")
    con.register("fact", fact)
    con.register("dim", dim)


def duck(sql):
    return lambda: con.execute(sql).to_arrow_table()


# --------------------------------------------------------------------------------------------------
# (a) filtered aggregate

sec = "(a) sum(amount), count(*) where region < 20 and qty > 10"
header(sec)
bench(sec, "ArrowMetal lazy",
      lambda: am.scan(am_fact).filter((am.col("region") < 20) & (am.col("qty") > 10))
                .agg(am.agg.sum("amount", "total"), am.agg.count("n")).collect())
bench(sec, "polars lazy",
      lambda: pl_fact.lazy().filter((pl.col("region") < 20) & (pl.col("qty") > 10))
                .select(pl.col("amount").sum().alias("total"), pl.len().alias("n")).collect())
if con:
    bench(sec, "duckdb", duck("SELECT sum(amount) AS total, count(*) AS n FROM fact "
                              "WHERE region < 20 AND qty > 10"))

# --------------------------------------------------------------------------------------------------
# (b) filter + group-by + sort + limit

sec = "(b) group-by 200 keys: sum, count; order by total desc limit 10"
header(sec)
bench(sec, "ArrowMetal lazy",
      lambda: am.scan(am_fact).filter(am.col("amount") > 0)
                .group_by("region").agg(am.agg.sum("amount", "total"), am.agg.count("n"))
                .sort("total", descending=True).limit(10).collect())
bench(sec, "polars lazy",
      lambda: pl_fact.lazy().filter(pl.col("amount") > 0)
                .group_by("region").agg(pl.col("amount").sum().alias("total"), pl.len().alias("n"))
                .sort("total", descending=True).limit(10).collect())
if con:
    bench(sec, "duckdb", duck("SELECT region, sum(amount) AS total, count(*) AS n FROM fact "
                              "WHERE amount > 0 GROUP BY region ORDER BY total DESC LIMIT 10"))

# --------------------------------------------------------------------------------------------------
# (c) multi-key group-by (10 000 groups)

sec = "(c) group-by (region, sub): 10 000 groups, mean + max"
header(sec)
bench(sec, "ArrowMetal lazy",
      lambda: am.scan(am_fact).group_by("region", "sub")
                .agg(am.agg.mean("qty", "avg"), am.agg.max("qty", "hi")).collect())
bench(sec, "polars lazy",
      lambda: pl_fact.lazy().group_by("region", "sub")
                .agg(pl.col("qty").mean().alias("avg"), pl.col("qty").max().alias("hi")).collect())
if con:
    bench(sec, "duckdb", duck("SELECT region, sub, avg(qty) AS avg, max(qty) AS hi FROM fact "
                              "GROUP BY region, sub"))

# --------------------------------------------------------------------------------------------------
# (d) projection chain (six operators, one fused kernel)

sec = "(d) with_columns((amount*2 + qty) / (region + 1) - qty) then filter"
header(sec)
expr_am = (am.col("amount") * 2 + am.col("qty")) / (am.col("region") + 1) - am.col("qty")
expr_pl = (pl.col("amount") * 2 + pl.col("qty")) / (pl.col("region") + 1) - pl.col("qty")
bench(sec, "ArrowMetal lazy (1 fused kernel)",
      lambda: am.scan(am_fact).select(expr_am.alias("r"))
                .filter(am.col("r") > 0).collect())
bench(sec, "polars lazy",
      lambda: pl_fact.lazy().select(expr_pl.alias("r")).filter(pl.col("r") > 0).collect())
if con:
    bench(sec, "duckdb", duck("SELECT r FROM (SELECT (amount*2 + qty) / (region + 1) - qty AS r "
                              "FROM fact) WHERE r > 0"))

# --------------------------------------------------------------------------------------------------
# (e) hash join, 10M probe x 1M build, then an aggregate

probe_rows = min(rows, 10_000_000)
build_rows = 1_000_000
pk = rng.integers(0, build_rows, size=probe_rows, dtype=np.int64)
pv = (rng.random(probe_rows, dtype=np.float32) * 100).astype(np.float32)
bk = np.arange(build_rows, dtype=np.int64)
bw = (rng.random(build_rows, dtype=np.float32) * 2).astype(np.float32)
probe = pa.table({"k": pa.array(pk), "v": pa.array(pv)})
build = pa.table({"k": pa.array(bk), "w": pa.array(bw)})
pl_probe, pl_build = pl.from_arrow(probe), pl.from_arrow(build)
am_probe = {n: am.array(probe.column(n).combine_chunks()) for n in probe.column_names}
am_build = {n: am.array(build.column(n).combine_chunks()) for n in build.column_names}
if con:
    con.register("probe", probe)
    con.register("build", build)

sec = f"(e) inner join {probe_rows:,} x {build_rows:,} on int64, then sum"
header(sec)
bench(sec, "ArrowMetal lazy",
      lambda: am.scan(am_probe).join(am.scan(am_build), on="k", how="inner")
                .agg(am.agg.sum("v", "total")).collect())
bench(sec, "polars lazy",
      lambda: pl_probe.lazy().join(pl_build.lazy(), on="k", how="inner")
                .select(pl.col("v").sum().alias("total")).collect())
if con:
    bench(sec, "duckdb", duck("SELECT sum(v) AS total FROM probe JOIN build USING (k)"))

# --------------------------------------------------------------------------------------------------
# (f) semi join

sec = f"(f) semi join {probe_rows:,} against a 1 000-row key set"
header(sec)
small = pa.table({"k": pa.array(np.arange(1000, dtype=np.int64))})
pl_small = pl.from_arrow(small)
am_small = {"k": am.array(small.column("k").combine_chunks())}
if con:
    con.register("small", small)
bench(sec, "ArrowMetal lazy",
      lambda: am.scan(am_probe).join(am.scan(am_small), on="k", how="semi").collect())
bench(sec, "polars lazy",
      lambda: pl_probe.lazy().join(pl_small.lazy(), on="k", how="semi").collect())
if con:
    bench(sec, "duckdb", duck("SELECT * FROM probe WHERE k IN (SELECT k FROM small)"))

# --------------------------------------------------------------------------------------------------
# (g) window over partitions

win_rows = min(rows, 10_000_000)
wg = rng.integers(0, 1000, size=win_rows, dtype=np.int32)
wv = rng.integers(0, 1_000_000, size=win_rows, dtype=np.int64)
win = pa.table({"g": pa.array(wg), "v": pa.array(wv)})
pl_win = pl.from_arrow(win)
am_win = {n: am.array(win.column(n).combine_chunks()) for n in win.column_names}
if con:
    con.register("win", win)

sec = f"(g) row_number() over (partition by g order by v), {win_rows:,} rows, 1 000 partitions"
header(sec)
bench(sec, "ArrowMetal lazy",
      lambda: am.scan(am_win).with_row_number("rn", partition_by="g", order_by="v").collect())
bench(sec, "polars lazy",
      lambda: pl_win.lazy().with_columns(pl.col("v").rank("ordinal").over("g").alias("rn")).collect())
if con:
    bench(sec, "duckdb", duck("SELECT g, v, row_number() OVER (PARTITION BY g ORDER BY v) AS rn FROM win"))

# --------------------------------------------------------------------------------------------------
# (h) as-of join

asof_rows = rows
quote_rows = 1_000_000
span = 10_000_000_000
trade_t = np.sort(rng.integers(0, span, size=asof_rows).astype(np.int64))
# Strictly increasing quote timestamps: a cumulative sum of positive gaps, not a sample without
# replacement (which would materialise a permutation of the whole timestamp span).
quote_t = np.cumsum(rng.integers(1, 2 * span // quote_rows, size=quote_rows, dtype=np.int64))
quote_p = (rng.random(quote_rows) * 100).astype(np.float64)
trades = pa.table({"t": pa.array(trade_t)})
quotes = pa.table({"t": pa.array(quote_t), "px": pa.array(quote_p)})
pl_trades, pl_quotes = pl.from_arrow(trades), pl.from_arrow(quotes)
am_trades = {"t": am.array(trades.column("t").combine_chunks())}
am_quotes = {n: am.array(quotes.column(n).combine_chunks()) for n in quotes.column_names}
if con:
    con.register("trades", trades)
    con.register("quotes", quotes)

sec = f"(h) as-of join {asof_rows:,} trades against {quote_rows:,} quotes"
header(sec)
bench(sec, "ArrowMetal lazy",
      lambda: am.scan(am_trades).join_asof(am.scan(am_quotes), on="t").collect())
bench(sec, "polars",
      lambda: pl_trades.join_asof(pl_quotes, on="t"))
# DuckDB's ASOF JOIN is off by default here: at these sizes it did not finish in ten minutes on the
# machine these numbers come from, while every other engine took milliseconds. Set
# AM_BENCH_DUCKDB_ASOF=1 to include it.
if con and os.environ.get("AM_BENCH_DUCKDB_ASOF"):
    bench(sec, "duckdb", duck("SELECT t.t, q.px FROM trades t ASOF JOIN quotes q ON t.t >= q.t"))

# --------------------------------------------------------------------------------------------------

print("\n| case | implementation | wall ms | CPU ms |")
print("|---|---|---:|---:|")
last = None
for section, label, wall, cpu in results:
    print(f"| {section if section != last else ''} | {label} | {wall:.2f} | {cpu:.1f} |")
    last = section
