"""Same operations as arrowmetal-bench, run with Polars, pyarrow.compute and pandas (all multi-threaded
where the library supports it). Data has the same shape and distribution as the Swift benchmark.

Usage: python Benchmarks/python_bench.py [rows] [iterations]
"""
import sys, time, os, resource
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pandas as pd

rows = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000_000
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 5
rng = np.random.default_rng(42)
results = []

def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def bench(section, label, bytes_, fn):
    fn()
    best = float("inf"); best_cpu = float("inf")
    for _ in range(iters):
        c0 = cpu_seconds(); t0 = time.perf_counter(); fn(); wall = time.perf_counter() - t0; cpu = cpu_seconds() - c0
        if wall < best: best, best_cpu = wall, cpu
    print(f"  {label:<44} {best*1000:9.2f} ms  {bytes_/best/1e9:7.1f} GB/s  {best_cpu*1000:8.1f} CPU-ms")
    results.append((section, label, best * 1000, bytes_ / best / 1e9, best_cpu * 1000))

print(f"Python bench: polars {pl.__version__} ({pl.thread_pool_size()} threads), pyarrow {pa.__version__} "
      f"({pa.cpu_count()} threads), pandas {pd.__version__}; rows={rows}, best of {iters}\n")

# ---- Int64 with 10% nulls
vals = rng.integers(-1000, 1001, size=rows, dtype=np.int64)
valid = rng.random(rows) >= 0.10
arr_i64 = pa.array(vals, mask=~valid)               # Arrow int64 with validity bitmap
pl_i64 = pl.Series("x", arr_i64)
pd_i64 = pd.Series(pd.arrays.ArrowExtensionArray(arr_i64))
B = rows * 8

sec = "sum(Int64, 10% nulls)"; print(sec)
bench(sec, "polars  sum", B, lambda: pl_i64.sum())
bench(sec, "pyarrow sum", B, lambda: pc.sum(arr_i64))
bench(sec, "pandas  sum (arrow-backed)", B, lambda: pd_i64.sum())

sec = "min(Int64, 10% nulls)"; print("\n" + sec)
bench(sec, "polars  min", B, lambda: pl_i64.min())
bench(sec, "pyarrow min", B, lambda: pc.min(arr_i64))
bench(sec, "pandas  min (arrow-backed)", B, lambda: pd_i64.min())

sec = "compare(Int64 > 0) -> boolean bitmap"; print("\n" + sec)
bench(sec, "polars  compare", B, lambda: (pl_i64 > 0))
bench(sec, "pyarrow compare", B, lambda: pc.greater(arr_i64, 0))
bench(sec, "pandas  compare (arrow-backed)", B, lambda: (pd_i64 > 0))

sec = "filter(Int64 where > 0) -> compacted (~45% kept)"; print("\n" + sec)
pl_mask = pl_i64 > 0; pa_mask = pc.greater(arr_i64, 0); pd_mask = pd_i64 > 0
bench(sec, "polars  filter", B, lambda: pl_i64.filter(pl_mask))
bench(sec, "pyarrow filter", B, lambda: pc.filter(arr_i64, pa_mask))
bench(sec, "pandas  filter (arrow-backed)", B, lambda: pd_i64[pd_mask])

sec = "compare + filter pipeline"; print("\n" + sec)
bench(sec, "polars  compare then filter", B, lambda: pl_i64.filter(pl_i64 > 0))
bench(sec, "pyarrow compare then filter", B, lambda: pc.filter(arr_i64, pc.greater(arr_i64, 0)))
bench(sec, "pandas  compare then filter", B, lambda: pd_i64[pd_i64 > 0])

sec = "multiply(Int64 * 3)"; print("\n" + sec)
bench(sec, "polars  multiply scalar", 2 * B, lambda: pl_i64 * 3)
bench(sec, "pyarrow multiply scalar", 2 * B, lambda: pc.multiply(arr_i64, 3))
bench(sec, "pandas  multiply scalar (arrow-backed)", 2 * B, lambda: pd_i64 * 3)

sec = "take(Int64, 25M random indices)"; print("\n" + sec)
idx = rng.integers(0, rows, size=rows // 2, dtype=np.int32)
pa_idx = pa.array(idx); pl_idx = pl.Series(idx)
TB = rows // 2 * (8 + 4 + 8)
bench(sec, "polars  gather", TB, lambda: pl_i64.gather(pl_idx))
bench(sec, "pyarrow take", TB, lambda: pc.take(arr_i64, pa_idx))
bench(sec, "pandas  take (arrow-backed)", TB, lambda: pd_i64.take(idx))

sec = "cast(Int64 -> Float32)"; print("\n" + sec)
bench(sec, "polars  cast", rows * 12, lambda: pl_i64.cast(pl.Float32))
bench(sec, "pyarrow cast", rows * 12, lambda: pc.cast(arr_i64, pa.float32()))

# ---- group-by and end-to-end query
keys5 = rng.integers(0, 5, size=rows, dtype=np.int32)
df = pl.DataFrame({"k": keys5, "x": pl_i64})
pa_tbl = pa.table({"k": pa.array(keys5), "x": arr_i64})
pdf = pd.DataFrame({"k": keys5, "x": pd_i64})
GB = rows * 12
sec = "group-by sum(Int64) by 5 keys"; print("\n" + sec)
bench(sec, "polars  group_by sum", GB, lambda: df.group_by("k").agg(pl.col("x").sum()))
bench(sec, "pyarrow group_by sum", GB, lambda: pa_tbl.group_by("k").aggregate([("x", "sum")]))
bench(sec, "pandas  groupby sum (arrow-backed)", GB, lambda: pdf.groupby("k")["x"].sum())
keys1k = rng.integers(0, 1000, size=rows, dtype=np.int32)
df1k = pl.DataFrame({"k": keys1k, "x": pl_i64}); pa1k = pa.table({"k": pa.array(keys1k), "x": arr_i64})
sec = "group-by sum(Int64) by 1000 keys"; print("\n" + sec)
bench(sec, "polars  group_by sum", GB, lambda: df1k.group_by("k").agg(pl.col("x").sum()))
bench(sec, "pyarrow group_by sum", GB, lambda: pa1k.group_by("k").aggregate([("x", "sum")]))
keys100k = rng.integers(0, 100_000, size=rows, dtype=np.int32)
df100k = pl.DataFrame({"k": keys100k, "x": pl_i64}); pa100k = pa.table({"k": pa.array(keys100k), "x": arr_i64})
sec = "group-by sum(Int64) by 100000 keys"; print("\n" + sec)
bench(sec, "polars  group_by sum", GB, lambda: df100k.group_by("k").agg(pl.col("x").sum()))
bench(sec, "pyarrow group_by sum", GB, lambda: pa100k.group_by("k").aggregate([("x", "sum")]))

amount = (rng.random(rows, dtype=np.float32) * 500).astype(np.float32)
q = pl.DataFrame({"region": keys5, "amount": amount})
qlazy = q.lazy()
pq = pd.DataFrame({"region": keys5, "amount": amount})
QB = rows * 8
sec = "query: sum(amount) where region == 2 and amount > 100 (50M rows)"; print("\n" + sec)
bench(sec, "polars  lazy (fused)", QB, lambda: qlazy.filter((pl.col("region") == 2) & (pl.col("amount") > 100)).select(pl.col("amount").sum()).collect())
bench(sec, "polars  eager", QB, lambda: q.filter((pl.col("region") == 2) & (pl.col("amount") > 100))["amount"].sum())
bench(sec, "pandas  query", QB, lambda: pq.loc[(pq.region == 2) & (pq.amount > 100), "amount"].sum())
bench(sec, "numpy   masked sum", QB, lambda: amount[(keys5 == 2) & (amount > 100)].sum())

# ---- Float64 no nulls
f64 = rng.random(rows) * 1000.0
arr_f64 = pa.array(f64); pl_f64 = pl.Series(arr_f64); pd_f64 = pd.Series(f64)
sec = "Float64: compare(> 500) then filter"; print("\n" + sec)
bench(sec, "polars  compare then filter", B, lambda: pl_f64.filter(pl_f64 > 500))
bench(sec, "pyarrow compare then filter", B, lambda: pc.filter(arr_f64, pc.greater(arr_f64, 500)))
bench(sec, "pandas  compare then filter (numpy-backed)", B, lambda: pd_f64[pd_f64 > 500])
bench(sec, "numpy   boolean index", B, lambda: f64[f64 > 500])

# ---- Float32 no nulls
f32 = (rng.random(rows, dtype=np.float32) * 2 - 1).astype(np.float32)
arr_f32 = pa.array(f32); pl_f32 = pl.Series(arr_f32); pd_f32 = pd.Series(f32)
B4 = rows * 4
sec = "sum(Float32, no nulls)"; print("\n" + sec)
bench(sec, "polars  sum", B4, lambda: pl_f32.sum())
bench(sec, "pyarrow sum", B4, lambda: pc.sum(arr_f32))
bench(sec, "numpy   sum", B4, lambda: f32.sum())
sec = "max(Float32, no nulls)"; print("\n" + sec)
bench(sec, "polars  max", B4, lambda: pl_f32.max())
bench(sec, "pyarrow max", B4, lambda: pc.max(arr_f32))
bench(sec, "numpy   max", B4, lambda: f32.max())
sec = "multiply(Float32 * 2.5)"; print("\n" + sec)
bench(sec, "polars  multiply scalar", 2 * B4, lambda: pl_f32 * 2.5)
bench(sec, "pyarrow multiply scalar", 2 * B4, lambda: pc.multiply(arr_f32, np.float32(2.5)))
bench(sec, "numpy   multiply scalar", 2 * B4, lambda: f32 * np.float32(2.5))
sec = "compare(Float32 > 0) then filter"; print("\n" + sec)
bench(sec, "polars  compare then filter", B4, lambda: pl_f32.filter(pl_f32 > 0))
bench(sec, "pyarrow compare then filter", B4, lambda: pc.filter(arr_f32, pc.greater(arr_f32, 0)))
bench(sec, "numpy   boolean index", B4, lambda: f32[f32 > 0])

print("\n\n| Operation | Implementation | Time (ms) | Throughput (GB/s) | CPU time (ms) |\n|---|---|---:|---:|---:|")
for s, l, ms, gb, cpu in results:
    print(f"| {s} | {l} | {ms:.2f} | {gb:.1f} | {cpu:.1f} |")
