"""The fused expression compiler vs one kernel per operator, Polars, pyarrow, pandas and numpy.

Every case runs on the same in-process data. "ArrowMetal fused" is one runtime-generated Metal kernel
for the whole expression; "ArrowMetal op-by-op" is the same expression built from the per-operator
kernels, batched into one command buffer (today's best without fusion).

Usage: PYTHONPATH=python python Benchmarks/expr_bench.py [rows] [iterations]
Requires .build/release/libArrowMetalC.dylib (swift build -c release --product ArrowMetalC).
"""
import resource
import sys
import time

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc

import arrowmetal as am

rows = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000_000
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 5
rng = np.random.default_rng(42)
results = []


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def bench(section, label, bytes_, fn):
    fn()
    best, best_cpu = float("inf"), float("inf")
    for _ in range(iters):
        c0, t0 = cpu_seconds(), time.perf_counter()
        fn()
        wall, cpu = time.perf_counter() - t0, cpu_seconds() - c0
        if wall < best:
            best, best_cpu = wall, cpu
    print(f"  {label:<46} {best * 1000:9.3f} ms  {bytes_ / best / 1e9:7.1f} GB/s  {best_cpu * 1000:8.1f} CPU-ms")
    results.append((section, label, best * 1000, bytes_ / best / 1e9, best_cpu * 1000))


print(f"ArrowMetal {am.version()} on {am.device_name()} vs polars {pl.__version__} "
      f"({pl.thread_pool_size()} threads), pyarrow {pa.__version__}, pandas {pd.__version__}, "
      f"numpy {np.__version__}; rows={rows}\n")

# ---------------------------------------------------------------------------- (a) filtered sum

region_np = rng.integers(0, 5, size=rows, dtype=np.int32)
amount_np = (rng.random(rows, dtype=np.float32) * 500).astype(np.float32)
region, amount = pa.array(region_np), pa.array(amount_np)
g_region, g_amount = am.array(region), am.array(amount)
tbl_a = pa.table({"region": region, "amount": amount})
pl_a = pl.from_arrow(tbl_a)
lazy_a = pl_a.lazy()
pd_a = pd.DataFrame({"region": region_np, "amount": amount_np})
QB = rows * 8

sec = "(a) sum(amount) where region == 2 and amount > 100"
print(sec)
q_a = am.filter((am.col("region") == 2) & (am.col("amount") > 100)).sum(am.col("amount"))
cols_a = {"region": g_region, "amount": g_amount}
bench(sec, "ArrowMetal fused (1 kernel)", QB, lambda: am.query(cols_a, q_a))


def _chained():
    with am.batch():
        return g_amount.filter((g_region == 2) & (g_amount > 100)).sum()


bench(sec, "ArrowMetal op-by-op (batched, 6 kernels)", QB, _chained)
bench(sec, "polars lazy (fused)", QB,
      lambda: lazy_a.filter((pl.col("region") == 2) & (pl.col("amount") > 100))
                    .select(pl.col("amount").sum()).collect())
bench(sec, "pyarrow.compute", QB,
      lambda: pc.sum(pc.filter(amount, pc.and_(pc.equal(region, 2), pc.greater(amount, 100)))))
bench(sec, "pandas", QB, lambda: pd_a.loc[(pd_a.region == 2) & (pd_a.amount > 100), "amount"].sum())
bench(sec, "numpy masked sum", QB, lambda: amount_np[(region_np == 2) & (amount_np > 100)].sum())

# ---------------------------------------------------------------------------- (b) 6-operator arithmetic

cols_np, cols_pa, cols_g = {}, {}, {}
for name, seed in [("a", 1), ("b", 2), ("c", 3), ("d", 4)]:
    v = rng.standard_normal(rows)
    mask = rng.random(rows) < 0.05
    cols_np[name] = v
    cols_pa[name] = pa.array(v, mask=mask)
    cols_g[name] = am.array(cols_pa[name])
tbl_b = pa.table(cols_pa)
lazy_b = pl.from_arrow(tbl_b).lazy()
EB = rows * 8 * 5           # four float64 columns read, one written

sec = "(b) (a*2 + b) / (c + 1) - d, float64, 5% nulls"
print("\n" + sec)
e_b = (am.col("a") * 2 + am.col("b")) / (am.col("c") + 1) - am.col("d")
q_b = e_b.alias("r").project()
bench(sec, "ArrowMetal fused (1 kernel)", EB, lambda: am.query(cols_g, q_b))


def _opbyop_b():
    with am.batch():
        t = (cols_g["a"] * 2.0 + cols_g["b"]) / (cols_g["c"] + 1.0) - cols_g["d"]
        return len(t)


bench(sec, "ArrowMetal op-by-op (batched, 5 kernels)", EB, _opbyop_b)
bench(sec, "polars lazy", EB,
      lambda: lazy_b.select(((pl.col("a") * 2 + pl.col("b")) / (pl.col("c") + 1) - pl.col("d")).alias("r")).collect())
bench(sec, "pyarrow.compute", EB,
      lambda: pc.subtract(pc.divide(pc.add(pc.multiply(cols_pa["a"], pa.scalar(2.0)), cols_pa["b"]),
                                    pc.add(cols_pa["c"], pa.scalar(1.0))), cols_pa["d"]))
bench(sec, "numpy (no nulls)", EB,
      lambda: (cols_np["a"] * 2 + cols_np["b"]) / (cols_np["c"] + 1) - cols_np["d"])

# ---------------------------------------------------------------------------- (c) filter + project 3

sec = "(c) filter(c < 5) -> project 3 columns"
print("\n" + sec)
sel_np = rng.integers(0, 10, size=rows, dtype=np.int32)
sel = pa.array(sel_np)
g_sel = am.array(sel)
cols_c = {"a": cols_g["a"], "b": cols_g["b"], "sel": g_sel}
tbl_c = pa.table({"a": cols_pa["a"], "b": cols_pa["b"], "sel": sel})
lazy_c = pl.from_arrow(tbl_c).lazy()
CB = rows * 20 + rows // 2 * 20
q_c = am.filter(am.col("sel") < 5).project(
    [am.col("a"), (am.col("b") * 2).alias("b2"), am.col("sel")])
bench(sec, "ArrowMetal fused (count + scan + scatter)", CB, lambda: am.query(cols_c, q_c))


def _opbyop_c():
    with am.batch():
        m = g_sel < 5
        return cols_g["a"].filter(m), (cols_g["b"] * 2.0).filter(m), g_sel.filter(m)


bench(sec, "ArrowMetal op-by-op (batched)", CB, _opbyop_c)
bench(sec, "polars lazy", CB,
      lambda: lazy_c.filter(pl.col("sel") < 5).select([pl.col("a"), (pl.col("b") * 2).alias("b2"), pl.col("sel")]).collect())
bench(sec, "pyarrow.compute", CB, lambda: pc.filter(tbl_c, pc.less(sel, 5)))

# ---------------------------------------------------------------------------- (d) group-by an expression

sec = "(d) group-by sum(v * 3 + 1) by 1000 keys"
print("\n" + sec)
K = 1000
keys_np = rng.integers(0, K, size=rows, dtype=np.int32)
vals_np = rng.integers(0, 1000, size=rows, dtype=np.int32)
keys, vals = pa.array(keys_np), pa.array(vals_np)
g_keys, g_vals = am.array(keys), am.array(vals)
tbl_d = pa.table({"k": keys, "v": vals})
lazy_d = pl.from_arrow(tbl_d).lazy()
DB = rows * 8
q_d = am.group_by(am.col("k"), K).sum(am.col("v") * 3 + 1, "s")
cols_d = {"k": g_keys, "v": g_vals}
bench(sec, "ArrowMetal fused (1 kernel + merge)", DB, lambda: am.query(cols_d, q_d))


def _opbyop_d():
    with am.batch():
        e = g_vals * 3 + 1
        return g_keys.group_by(K).sum(e)


bench(sec, "ArrowMetal op-by-op (batched)", DB, _opbyop_d)
bench(sec, "polars lazy", DB,
      lambda: lazy_d.group_by("k").agg((pl.col("v") * 3 + 1).sum()).collect())
bench(sec, "pyarrow group_by", DB,
      lambda: pa.table({"k": keys, "e": pc.add(pc.multiply(vals, 3), 1)})
                .group_by("k").aggregate([("e", "sum")]))

# ---------------------------------------------------------------------------- (e) the dispatch floor

small = 100_000
print(f"\n(e) the same shapes at {small} rows: the fixed cost is paid once")
s_region, s_amount = pa.array(region_np[:small]), pa.array(amount_np[:small])
sg = {"region": am.array(s_region), "amount": am.array(s_amount)}
s_tbl = pa.table({"region": s_region, "amount": s_amount})
s_lazy = pl.from_arrow(s_tbl).lazy()
sec = f"(e) sum(amount) where region == 2 and amount > 100, {small} rows"
SB = small * 8
bench(sec, "ArrowMetal fused (1 kernel)", SB, lambda: am.query(sg, q_a))


def _small_chained():
    with am.batch():
        return sg["amount"].filter((sg["region"] == 2) & (sg["amount"] > 100)).sum()


bench(sec, "ArrowMetal op-by-op (batched)", SB, _small_chained)
bench(sec, "polars lazy", SB,
      lambda: s_lazy.filter((pl.col("region") == 2) & (pl.col("amount") > 100))
                    .select(pl.col("amount").sum()).collect())
bench(sec, "pyarrow.compute", SB,
      lambda: pc.sum(pc.filter(s_amount, pc.and_(pc.equal(s_region, 2), pc.greater(s_amount, 100)))))

s_cols_b = {k: am.array(cols_pa[k].slice(0, small)) for k in "abcd"}
s_lazy_b = pl.from_arrow(pa.table({k: cols_pa[k].slice(0, small) for k in "abcd"})).lazy()
sec = f"(e) (a*2 + b) / (c + 1) - d, {small} rows"
SEB = small * 8 * 5
bench(sec, "ArrowMetal fused (1 kernel)", SEB, lambda: am.query(s_cols_b, q_b))


def _small_opbyop():
    with am.batch():
        t = (s_cols_b["a"] * 2.0 + s_cols_b["b"]) / (s_cols_b["c"] + 1.0) - s_cols_b["d"]
        return len(t)


bench(sec, "ArrowMetal op-by-op (batched, 5 kernels)", SEB, _small_opbyop)
bench(sec, "polars lazy", SEB,
      lambda: s_lazy_b.select(((pl.col("a") * 2 + pl.col("b")) / (pl.col("c") + 1) - pl.col("d")).alias("r")).collect())

# ---------------------------------------------------------------------------- summary

print("\n| case | implementation | wall ms | GB/s | CPU ms |")
print("|---|---|---:|---:|---:|")
for section, label, ms, gbs, cpu in results:
    print(f"| {section} | {label} | {ms:.3f} | {gbs:.1f} | {cpu:.1f} |")
