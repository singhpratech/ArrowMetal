"""ArrowMetal from Python vs Polars, pyarrow.compute and pandas on the same in-process data.

Usage: PYTHONPATH=python python Benchmarks/python_gpu_bench.py [rows] [iterations]
Requires .build/release/libArrowMetalC.dylib (swift build -c release --product ArrowMetalC).
"""
import sys, time
import numpy as np, pyarrow as pa, pyarrow.compute as pc, polars as pl, pandas as pd
import arrowmetal as am

rows = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000_000
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 5
rng = np.random.default_rng(42)
results = []


def bench(section, label, bytes_, fn):
    fn()
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    print(f"  {label:<44} {best*1000:9.2f} ms  {bytes_/best/1e9:7.1f} GB/s")
    results.append((section, label, best * 1000, bytes_ / best / 1e9))


print(f"ArrowMetal {am.version()} on {am.device_name()} vs polars {pl.__version__} ({pl.thread_pool_size()} threads), "
      f"pyarrow {pa.__version__}, pandas {pd.__version__}; rows={rows}\n")

vals = rng.integers(-1000, 1001, size=rows, dtype=np.int64)
valid = rng.random(rows) >= 0.10
arr = pa.array(vals, mask=~valid)
t0 = time.perf_counter(); gpu = am.array(arr); print(f"import 50M-row int64 column into Metal memory: {(time.perf_counter()-t0)*1000:.1f} ms\n")
pls = pl.Series("x", arr); pds = pd.Series(pd.arrays.ArrowExtensionArray(arr))
B = rows * 8

sec = "sum(Int64, 10% nulls)"; print(sec)
bench(sec, "ArrowMetal (GPU)", B, lambda: gpu.sum())
bench(sec, "polars", B, lambda: pls.sum())
bench(sec, "pyarrow.compute", B, lambda: pc.sum(arr))
bench(sec, "pandas (arrow-backed)", B, lambda: pds.sum())

sec = "filter(Int64 > 0) -> compacted"; print("\n" + sec)
bench(sec, "ArrowMetal (GPU, fused predicate)", B, lambda: gpu.filter_where(">", 0))
bench(sec, "ArrowMetal (GPU) + to_arrow()", B, lambda: gpu.filter_where(">", 0).to_arrow())
bench(sec, "polars", B, lambda: pls.filter(pls > 0))
bench(sec, "pyarrow.compute", B, lambda: pc.filter(arr, pc.greater(arr, 0)))
bench(sec, "pandas (arrow-backed)", B, lambda: pds[pds > 0])

sec = "take(25M random indices)"; print("\n" + sec)
idx = rng.integers(0, rows, size=rows // 2, dtype=np.int32)
gidx = am.array(pa.array(idx)); pidx = pl.Series(idx); aidx = pa.array(idx)
TB = rows // 2 * 20
bench(sec, "ArrowMetal (GPU)", TB, lambda: gpu.take(gidx))
bench(sec, "polars gather", TB, lambda: pls.gather(pidx))
bench(sec, "pyarrow.compute take", TB, lambda: pc.take(arr, aidx))

sec = "group-by sum(Int64) by 1000 keys"; print("\n" + sec)
keys = rng.integers(0, 1000, size=rows, dtype=np.int32)
gkeys = am.array(pa.array(keys)); gb = gkeys.group_by(1000)
df = pl.DataFrame({"k": keys, "x": pls}); tbl = pa.table({"k": pa.array(keys), "x": arr})
GB = rows * 12
bench(sec, "ArrowMetal (GPU)", GB, lambda: gb.sum(gpu))
bench(sec, "polars group_by", GB, lambda: df.group_by("k").agg(pl.col("x").sum()))
bench(sec, "pyarrow group_by", GB, lambda: tbl.group_by("k").aggregate([("x", "sum")]))

sec = "query: sum(amount) where region == 2 and amount > 100"; print("\n" + sec)
region = rng.integers(0, 5, size=rows, dtype=np.int32)
amount = (rng.random(rows, dtype=np.float32) * 500).astype(np.float32)
gr = am.array(pa.array(region)); ga = am.array(pa.array(amount))
q = pl.DataFrame({"region": region, "amount": amount}); ql = q.lazy()
QB = rows * 8
bench(sec, "ArrowMetal (GPU)", QB, lambda: ga.filter((gr == 2) & (ga > 100)).sum())
def _batched():
    with am.batch():
        return ga.filter((gr == 2) & (ga > 100)).sum()
bench(sec, "ArrowMetal (GPU, batched)", QB, _batched)
bench(sec, "polars lazy (fused)", QB, lambda: ql.filter((pl.col("region") == 2) & (pl.col("amount") > 100)).select(pl.col("amount").sum()).collect())
bench(sec, "numpy masked sum", QB, lambda: amount[(region == 2) & (amount > 100)].sum())

print("\n\n| Operation | Implementation | Time (ms) | Throughput (GB/s) |\n|---|---|---:|---:|")
for s, l, ms, gb_ in results:
    print(f"| {s} | {l} | {ms:.2f} | {gb_:.1f} |")
