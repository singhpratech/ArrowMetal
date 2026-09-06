"""ArrowMetal from Python vs Polars, pyarrow.compute and pandas on the same in-process data.

Usage: PYTHONPATH=python python Benchmarks/python_gpu_bench.py [rows] [iterations]
Requires .build/release/libArrowMetalC.dylib (swift build -c release --product ArrowMetalC).
"""
import sys, time, resource
import numpy as np, pyarrow as pa, pyarrow.compute as pc, polars as pl, pandas as pd
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
    best = float("inf"); best_cpu = float("inf")
    for _ in range(iters):
        c0 = cpu_seconds(); t0 = time.perf_counter(); fn(); wall = time.perf_counter() - t0; cpu = cpu_seconds() - c0
        if wall < best: best, best_cpu = wall, cpu
    print(f"  {label:<44} {best*1000:9.2f} ms  {bytes_/best/1e9:7.1f} GB/s  {best_cpu*1000:8.1f} CPU-ms")
    results.append((section, label, best * 1000, bytes_ / best / 1e9, best_cpu * 1000))


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

# ---- sorting: ArrowMetal's GPU radix sort vs polars, pyarrow and numpy
sort_i64 = rng.integers(-(2 ** 62), 2 ** 62, size=rows, dtype=np.int64)
a_i64 = pa.array(sort_i64); g_i64 = am.array(a_i64); p_i64 = pl.Series("x", a_i64)
sort_f64 = rng.random(rows) * 2e9 - 1e9
a_f64 = pa.array(sort_f64); g_f64 = am.array(a_f64); p_f64 = pl.Series("x", a_f64)

sec = f"argsort(Int64, {rows} rows, no nulls)"; print("\n" + sec)
bench(sec, "ArrowMetal (GPU)", rows * 12, lambda: g_i64.argsort())
bench(sec, "polars arg_sort", rows * 12, lambda: p_i64.arg_sort())
bench(sec, "pyarrow array_sort_indices", rows * 12, lambda: pc.array_sort_indices(a_i64))
bench(sec, "numpy argsort", rows * 12, lambda: np.argsort(sort_i64))

sec = f"sort(Float64, {rows} rows, no nulls)"; print("\n" + sec)
bench(sec, "ArrowMetal (GPU)", rows * 16, lambda: g_f64.sort())
bench(sec, "ArrowMetal (GPU) + to_arrow()", rows * 16, lambda: g_f64.sort().to_arrow())
bench(sec, "polars sort", rows * 16, lambda: p_f64.sort())
bench(sec, "pyarrow sort_indices + take", rows * 16, lambda: pc.take(a_f64, pc.array_sort_indices(a_f64)))
bench(sec, "numpy sort", rows * 16, lambda: np.sort(sort_f64))

sec = f"top_k(100 of {rows} Int64)"; print("\n" + sec)
bench(sec, "ArrowMetal (GPU, full sort)", rows * 8, lambda: g_i64.top_k(100))
bench(sec, "polars top_k", rows * 8, lambda: p_i64.top_k(100))
bench(sec, "numpy argpartition", rows * 8, lambda: np.argpartition(sort_i64, rows - 100)[rows - 100:])

# ---- strings: 10M utf8 values from 1000 distinct keys
str_rows = min(rows, 10_000_000)
distinct = 1000
regions = ["north", "south", "east", "west"]
vocab = [f"cust_{i:03d}_{regions[i % 4]}" for i in range(distinct)]
scodes = rng.integers(0, distinct, size=str_rows, dtype=np.int32)
arr_str = pa.DictionaryArray.from_arrays(pa.array(scodes), pa.array(vocab)).cast(pa.string())
t0 = time.perf_counter(); g_str = am.array(arr_str)
print(f"\nimport {str_rows}-row utf8 column into Metal memory: {(time.perf_counter()-t0)*1000:.1f} ms")
p_str = pl.Series("s", arr_str)
SB = arr_str.nbytes

sec = f'string contains("north") over {str_rows} strings (25% hit)'; print("\n" + sec)
bench(sec, "ArrowMetal (GPU)", SB, lambda: g_str.str_contains("north"))
bench(sec, "polars str.contains (literal)", SB, lambda: p_str.str.contains("north", literal=True))
bench(sec, "pyarrow match_substring", SB, lambda: pc.match_substring(arr_str, "north"))

sec = f'string starts_with("cust_1") over {str_rows} strings (10% hit)'; print("\n" + sec)
bench(sec, "ArrowMetal (GPU)", SB, lambda: g_str.starts_with("cust_1"))
bench(sec, "polars str.starts_with", SB, lambda: p_str.str.starts_with("cust_1"))
bench(sec, "pyarrow starts_with", SB, lambda: pc.starts_with(arr_str, "cust_1"))

sec = f'string equals("cust_042_east") over {str_rows} strings'; print("\n" + sec)
bench(sec, "ArrowMetal (GPU)", SB, lambda: g_str.str_equals("cust_042_east"))
bench(sec, "polars ==", SB, lambda: p_str == "cust_042_east")
bench(sec, "pyarrow equal", SB, lambda: pc.equal(arr_str, "cust_042_east"))

sec = f"string filter over {str_rows} strings (~30% kept)"; print("\n" + sec)
keep = scodes < distinct * 3 // 10
g_keep = am.array(pa.array(keep)); pa_keep = pa.array(keep); pl_keep = pl.Series(keep)
bench(sec, "ArrowMetal (GPU)", SB, lambda: g_str.filter(g_keep))
bench(sec, "polars filter", SB, lambda: p_str.filter(pl_keep))
bench(sec, "pyarrow filter", SB, lambda: pc.filter(arr_str, pa_keep))

sec = f"dictionary_encode + group-by sum over {str_rows} strings ({distinct} distinct)"; print("\n" + sec)
str_amount = sort_i64[:str_rows]
g_amount = am.array(pa.array(str_amount))
DB = SB + str_rows * 8
df_str = pl.DataFrame({"s": p_str, "x": str_amount})
tbl_str = pa.table({"s": arr_str, "x": pa.array(str_amount)})
def _am_dict_group():
    codes, uniq = g_str.dictionary_encode()
    return codes.group_by(len(uniq)).sum(g_amount)
bench(sec, "ArrowMetal (dictionary_encode on CPU + GPU group-by)", DB, _am_dict_group)
_codes, _uniq = g_str.dictionary_encode(); _gb = _codes.group_by(len(_uniq))
bench(sec, "ArrowMetal (GPU group-by on cached codes)", DB, lambda: _gb.sum(g_amount))
bench(sec, "polars group_by(str) sum", DB, lambda: df_str.group_by("s").agg(pl.col("x").sum()))
bench(sec, "pyarrow group_by(str) sum", DB, lambda: tbl_str.group_by("s").aggregate([("x", "sum")]))

print("\n\n| Operation | Implementation | Time (ms) | Throughput (GB/s) | CPU time (ms) |\n|---|---|---:|---:|---:|")
for s, l, ms, gb_, cpu in results:
    print(f"| {s} | {l} | {ms:.2f} | {gb_:.1f} | {cpu:.1f} |")
