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

# ---- distinct-value functions over an int64 column at 1K / 100K / 10M distinct. ArrowMetal runs these
# on the GPU hash table (Kernels/HashTable.swift) above 65,536 rows, so their cost follows the distinct
# count rather than the row count; the CPU libraries all use a hash table too, which is why these were
# the cases where the old sort-based path lost.
for dv_distinct in (1_000, 100_000, 10_000_000):
    if dv_distinct > rows:
        continue
    dv_n = rng.integers(0, dv_distinct, size=rows, dtype=np.int64) * 7
    dv_a = pa.array(dv_n)
    dv_g = am.array(dv_a)
    dv_p = pl.Series("v", dv_a)
    dv_d = pd.Series(pd.arrays.ArrowExtensionArray(dv_a))
    DV = rows * 8

    sec = f"unique(int64) ({dv_distinct} distinct)"; print("\n" + sec)
    bench(sec, "ArrowMetal (GPU hash table)", DV, lambda _g=dv_g: _g.unique())
    bench(sec, "polars", DV, lambda _p=dv_p: _p.unique())
    bench(sec, "pyarrow", DV, lambda _a=dv_a: pc.unique(_a))
    bench(sec, "pandas (numpy)", DV, lambda _n=dv_n: pd.unique(_n))

    sec = f"value_counts(int64) ({dv_distinct} distinct)"; print("\n" + sec)
    bench(sec, "ArrowMetal (GPU hash table)", DV, lambda _g=dv_g: _g.value_counts())
    bench(sec, "polars", DV, lambda _p=dv_p: _p.value_counts())
    bench(sec, "pyarrow", DV, lambda _a=dv_a: pc.value_counts(_a))
    bench(sec, "pandas", DV, lambda _d=dv_d: _d.value_counts())

    sec = f"count_distinct(int64) ({dv_distinct} distinct)"; print("\n" + sec)
    bench(sec, "ArrowMetal (GPU hash table)", DV, lambda _g=dv_g: _g.count_distinct())
    bench(sec, "polars", DV, lambda _p=dv_p: _p.n_unique())
    bench(sec, "pyarrow", DV, lambda _a=dv_a: pc.count_distinct(_a))
    bench(sec, "pandas", DV, lambda _d=dv_d: _d.nunique())

    sec = f"mode(int64) ({dv_distinct} distinct)"; print("\n" + sec)
    bench(sec, "ArrowMetal (GPU hash table)", DV, lambda _g=dv_g: _g.mode())
    bench(sec, "polars", DV, lambda _p=dv_p: _p.mode())
    bench(sec, "pyarrow", DV, lambda _a=dv_a: pc.mode(_a))

    # dictionary_encode over a primitive column is Swift-only: the C ABI has am_str_dictionary_encode
    # (utf8) and nothing for the primitive form, so it is measured in Sources/ArrowMetalBench/main.swift.
    del dv_n, dv_a, dv_g, dv_p, dv_d

# ---- group-by over arbitrary keys: utf8 keys and two int32 key columns, at 1K / 100K / 10M distinct.
# ArrowMetal maps the keys to dense group ids on the GPU, so unlike the dense group-by above there is
# no dictionary_encode step for the caller to do first; polars and pyarrow are given the same columns.
for gb_distinct in (1_000, 100_000, 10_000_000):
    if gb_distinct > rows:
        continue
    codes = rng.integers(0, gb_distinct, size=rows, dtype=np.int32)
    # Fixed-width 12-byte keys, built straight into the Arrow buffers with vectorised numpy: "k" plus
    # the code's low 44 bits in hex. (np.char.mod over 50 million elements would cost more than the
    # benchmark it feeds.)
    width = 12
    hexdigits = np.frombuffer(b"0123456789abcdef", dtype=np.uint8)
    body = np.empty((rows, width), dtype=np.uint8)
    body[:, 0] = ord("k")
    acc = codes.astype(np.uint64)
    for _j in range(width - 1, 0, -1):
        body[:, _j] = hexdigits[(acc & 0xF).astype(np.intp)]
        acc >>= 4
    offsets = np.arange(rows + 1, dtype=np.int32) * width
    key_arr = pa.Array.from_buffers(pa.utf8(), rows,
                                    [None, pa.py_buffer(offsets), pa.py_buffer(body.reshape(-1))])
    g_keys = am.array(key_arr)
    p_keys = pl.Series("k", key_arr)
    KB = rows * width + (rows + 1) * 4 + rows * 8

    sec = f"group-by sum(Int64) over utf8 keys ({gb_distinct} distinct)"
    print("\n" + sec)
    df_k = pl.DataFrame({"k": p_keys, "x": pls})
    tbl_k = pa.table({"k": key_arr, "x": arr})

    def _am_str_group(_g=g_keys):
        return am.group_by([_g]).sum(gpu)

    bench(sec, "ArrowMetal (GPU, key mapping included)", KB, _am_str_group)
    _cached = am.group_by([g_keys])
    bench(sec, "ArrowMetal (GPU, cached key mapping)", KB, lambda _c=_cached: _c.sum(gpu))
    bench(sec, "polars group_by(utf8)", KB, lambda _d=df_k: _d.group_by("k").agg(pl.col("x").sum()))
    bench(sec, "pyarrow group_by(utf8)", KB, lambda _t=tbl_k: _t.group_by("k").aggregate([("x", "sum")]))
    print(f"    groups: ArrowMetal {_cached.group_count}, pyarrow {tbl_k.group_by('k').aggregate([]).num_rows}")

    # The stage underneath that group-by, on its own: utf8 -> dense codes plus the distinct values in
    # first-seen order. ArrowMetal runs the GPU hash table in Kernels/StringHashTable.swift.
    sec = f"dictionary_encode(utf8) ({gb_distinct} distinct)"
    print("\n" + sec)
    bench(sec, "ArrowMetal (GPU hash table)", KB, lambda _g=g_keys: _g.dictionary_encode())
    bench(sec, "pyarrow dictionary_encode", KB, lambda _a=key_arr: pc.dictionary_encode(_a))

    side = max(2, int(np.ceil(np.sqrt(gb_distinct))))
    ka = rng.integers(0, side, size=rows, dtype=np.int32)
    kb = rng.integers(0, side, size=rows, dtype=np.int32)
    g_a, g_b = am.array(pa.array(ka)), am.array(pa.array(kb))
    TB2 = rows * 16
    sec = f"group-by sum(Int64) over two int32 columns (~{side * side} distinct)"
    print("\n" + sec)
    df_2 = pl.DataFrame({"a": ka, "b": kb, "x": pls})
    tbl_2 = pa.table({"a": pa.array(ka), "b": pa.array(kb), "x": arr})

    def _am_two_group(_a=g_a, _b=g_b):
        return am.group_by([_a, _b]).sum(gpu)

    bench(sec, "ArrowMetal (GPU, key mapping included)", TB2, _am_two_group)
    _cached2 = am.group_by([g_a, g_b])
    bench(sec, "ArrowMetal (GPU, cached key mapping)", TB2, lambda _c=_cached2: _c.sum(gpu))
    bench(sec, "polars group_by(a, b)", TB2, lambda _d=df_2: _d.group_by(["a", "b"]).agg(pl.col("x").sum()))
    bench(sec, "pyarrow group_by(a, b)", TB2,
          lambda _t=tbl_2: _t.group_by(["a", "b"]).aggregate([("x", "sum")]))
    print(f"    groups: ArrowMetal {_cached2.group_count}, "
          f"pyarrow {tbl_2.group_by(['a', 'b']).aggregate([]).num_rows}")


print("\n\n| Operation | Implementation | Time (ms) | Throughput (GB/s) | CPU time (ms) |\n|---|---|---:|---:|---:|")
for s, l, ms, gb_, cpu in results:
    print(f"| {s} | {l} | {ms:.2f} | {gb_:.1f} | {cpu:.1f} |")
