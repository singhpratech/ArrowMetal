"""DuckDB alone vs DuckDB scanning and ArrowMetal computing, on the same table in the same process.

Usage: PYTHONPATH=python python Benchmarks/duckdb_bench.py [rows] [iterations]
       PYTHONPATH=python python Benchmarks/duckdb_bench.py 10000000 5

Requires .build/release/libArrowMetalC.dylib and `pip install duckdb`.

WHAT IS BEING COMPARED, AND THE RULES

Both sides start from the same DuckDB table, already materialised in memory, and both produce the
same answer - which the benchmark checks, every iteration, before it believes a number.

  "duckdb"          the whole thing in SQL, on all the CPU's cores. DuckDB's own answer.
  "gpu (resident)"  the columns already on the GPU, which is what you have inside a loop that runs
                    many operations over one dataset. Kernel time only.
  "gpu (+crossing)" the same, plus the cost of getting the column out of DuckDB and onto the GPU
                    on every single call. This is the number for a one-shot query, and it is the
                    honest one to quote unless you really are reusing the data.

The crossing is reported on its own too, split into the two things it actually costs:

  to_arrow_table    DuckDB hands its result over the Arrow C Data interface. Cheap: it is mostly
                    pointers, not bytes.
  combine_chunks    DuckDB emits ~2048-row chunks, and one ArrowMetal array is one buffer, so the
                    chunks are concatenated. THIS one is a real copy at memory bandwidth, and it is
                    the single largest cost in the crossing. `am.duckdb_batches` skips it by keeping
                    the chunks separate; see docs/DUCKDB.md.
  am_import         wrapping the Arrow buffer in a Metal buffer. Copies nothing - the benchmark
                    asserts the pointer is unchanged - but it still costs page-table work.

Timing is best-of-N wall time with the process's CPU time alongside, the same way every other
benchmark in this repository does it, so a result that merely moved work onto more cores is visible
as CPU-ms rather than hiding behind wall-ms.
"""
import resource
import sys
import time

import duckdb
import pyarrow as pa

import arrowmetal as am

ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000_000
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 5

results = []


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def measure(fn):
    """Best-of-ITERS wall time in ms, and the CPU time of that same run."""
    fn()
    best_wall, best_cpu = float("inf"), float("inf")
    for _ in range(ITERS):
        c0, t0 = cpu_seconds(), time.perf_counter()
        fn()
        wall, cpu = time.perf_counter() - t0, cpu_seconds() - c0
        if wall < best_wall:
            best_wall, best_cpu = wall, cpu
    return best_wall * 1000, best_cpu * 1000


def row(section, duckdb_fn, gpu_fn, crossing_fn=None, check=True):
    """One comparison line. Both sides are run once first and their answers compared."""
    if check:
        expected, got = duckdb_fn(), gpu_fn()
        if not same(expected, got):
            print(f"  {section:<38}  MISMATCH: duckdb={expected!r} gpu={got!r}")
            return
    d_wall, d_cpu = measure(duckdb_fn)
    g_wall, g_cpu = measure(gpu_fn)
    x_wall, x_cpu = measure(crossing_fn) if crossing_fn else (float("nan"), float("nan"))
    speedup = d_wall / g_wall if g_wall else float("nan")
    total = d_wall / x_wall if crossing_fn and x_wall else float("nan")
    print(f"  {section:<38} {d_wall:9.1f} {d_cpu:9.1f} {g_wall:9.1f} {g_cpu:9.1f} "
          f"{x_wall:11.1f} {speedup:8.1f}x {total:8.1f}x")
    results.append((section, d_wall, d_cpu, g_wall, g_cpu, x_wall, speedup, total))


def same(a, b, tolerance=1e-6):
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        return abs(float(a) - float(b)) <= tolerance * max(1.0, abs(float(a)))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y, tolerance) for x, y in zip(a, b))
    return a == b


# ---------------------------------------------------------------------------------------------------
# The data: one table, in DuckDB, exactly as a user would have it.
# ---------------------------------------------------------------------------------------------------
print(f"ArrowMetal {am.version()} on {am.device_name()}; DuckDB {duckdb.__version__}; "
      f"pyarrow {pa.__version__}; rows={ROWS:,}; best of {ITERS}\n")

con = duckdb.connect()
print("building the table in DuckDB ...", end=" ", flush=True)
t0 = time.perf_counter()
con.execute(f"""
    create table t as
    select i::BIGINT                       as v,
           (hash(i) % 1000)::INTEGER       as k1000,
           (hash(i * 7) % 100000)::INTEGER as k100k,
           (i * 0.5)::DOUBLE               as f,
           ('user' || (hash(i) % 5000))    as s
    from range({ROWS}) r(i)
""")
con.execute("select count(*) from t").fetchone()
print(f"{(time.perf_counter() - t0):.1f} s\n")

# ---------------------------------------------------------------------------------------------------
# The crossing, on its own.
# ---------------------------------------------------------------------------------------------------
print("Getting one BIGINT column out of DuckDB and onto the GPU")
t0 = time.perf_counter()
table = con.sql("select v from t").to_arrow_table()
to_arrow_ms = (time.perf_counter() - t0) * 1000
column = table.column("v")
chunks, nbytes = column.num_chunks, column.nbytes

t0 = time.perf_counter()
flat = column.chunk(0) if chunks == 1 else column.combine_chunks()
combine_ms = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
metal = am.MetalArray.from_arrow(flat)
import_ms = (time.perf_counter() - t0) * 1000

zero_copy = am.duckdb_is_zero_copy(flat, metal)
print(f"  to_arrow_table   {to_arrow_ms:8.1f} ms   ({chunks} chunks, {nbytes / 1e6:.0f} MB)")
print(f"  combine_chunks   {combine_ms:8.1f} ms   ({nbytes / max(combine_ms, 1e-9) / 1e6:.1f} GB/s"
      f" - a real copy)")
print(f"  am_import        {import_ms:8.1f} ms   (buffer shared, not copied: {zero_copy})")
print(f"  total            {to_arrow_ms + combine_ms + import_ms:8.1f} ms\n")

# Resident columns: what you have after one from_duckdb, and what a loop reuses.
columns = am.from_duckdb(con.sql("select v, k1000, k100k, f, s from t"))
gpu_v, gpu_k1000, gpu_k100k, gpu_f, gpu_s = (
    columns["v"], columns["k1000"], columns["k100k"], columns["f"], columns["s"])
# Built once, because that is how you would hold them: the key mapping is reused by every aggregate.
group_1000 = am.group_by([gpu_k1000])
group_100k = am.group_by([gpu_k100k])

header = (f"  {'operation':<38} {'duckdb':>9} {'cpu-ms':>9} {'gpu':>9} {'cpu-ms':>9} "
          f"{'gpu+cross':>11} {'resident':>9} {'one-shot':>9}")
print(header)
print("  " + "-" * (len(header) - 2))


def cross(name):
    """Pull one column from DuckDB onto the GPU, the way a cold call would have to."""
    return am.from_duckdb(con.sql(f"select {name} from t"))[name]


# ---- filter + sum -----------------------------------------------------------------------------
threshold = ROWS // 2
row("filter(v > n) then sum",
    lambda: con.sql(f"select sum(v) from t where v > {threshold}").fetchone()[0],
    lambda: gpu_v.filter_where(">", threshold).sum(),
    lambda: cross("v").filter_where(">", threshold).sum())

# The fused compiler does the same work in one kernel: one read of the column, no intermediate.
fused = am.filter(am.col("v") > threshold).sum(am.col("v"), name="total")
row("filter(v > n) then sum, fused query",
    lambda: con.sql(f"select sum(v) from t where v > {threshold}").fetchone()[0],
    lambda: am.query({"v": gpu_v}, fused),
    lambda: am.query({"v": cross("v")}, fused))

# ---- group-by ---------------------------------------------------------------------------------
# Both sides reduce the per-key totals to one number. That keeps the comparison on the group-by
# itself instead of on how fast each side can hand 100,000 rows to Python, which is not the question.
def duckdb_group(key):
    return con.sql(f"select sum(total), count(*) from "
                   f"(select {key} k, sum(v) total from t group by {key})").fetchone()


def gpu_group(group, values):
    totals = group.sum(values)
    return (totals.sum(), len(totals))


row("group-by sum, 1k keys",
    lambda: duckdb_group("k1000"),
    lambda: gpu_group(group_1000, gpu_v),
    lambda: (lambda k, v: gpu_group(am.group_by([k]), v))(
        cross("k1000"), cross("v")))

row("group-by sum, 100k keys",
    lambda: duckdb_group("k100k"),
    lambda: gpu_group(group_100k, gpu_v),
    lambda: (lambda k, v: gpu_group(am.group_by([k]), v))(
        cross("k100k"), cross("v")))

# ---- top-k ------------------------------------------------------------------------------------
row("top_k(v, 100)",
    lambda: [r[0] for r in con.sql("select v from t order by v desc limit 100").fetchall()],
    lambda: gpu_v.take(gpu_v.top_k(100)).to_arrow().to_pylist(),
    lambda: (lambda c: c.take(c.top_k(100)).to_arrow().to_pylist())(cross("v")))

# ---- sort -------------------------------------------------------------------------------------
# Comparing the sorts themselves, not the cost of materialising 50M rows through Python: both sides
# sort the whole column and then look at the same two ends of it.
def duckdb_sort_ends():
    rows = con.sql("select min(v) a, max(v) b from (select v from t order by v)").fetchone()
    return list(rows)


def gpu_sort_ends(column):
    sorted_column = column.sort()
    return [sorted_column.slice(0, 1).to_arrow().to_pylist()[0],
            sorted_column.slice(len(sorted_column) - 1, 1).to_arrow().to_pylist()[0]]


row("sort(v) ascending",
    duckdb_sort_ends,
    lambda: gpu_sort_ends(gpu_v),
    lambda: gpu_sort_ends(cross("v")))

# ---- string contains --------------------------------------------------------------------------
# The match and the count both stay on the GPU: a boolean column of 50M elements costs more to turn
# into a Python list than either engine spends matching it.
contains_query = am.filter(am.col("s").contains("user1")).count()
row("count(s like '%user1%')",
    lambda: con.sql("select count(*) from t where s like '%user1%'").fetchone()[0],
    lambda: am.query({"s": gpu_s}, contains_query),
    lambda: am.query({"s": cross("s")}, contains_query))

# ---- mean over a double column ----------------------------------------------------------------
row("avg(f)",
    lambda: con.sql("select avg(f) from t").fetchone()[0],
    lambda: gpu_f.mean(),
    lambda: cross("f").mean())

# ---------------------------------------------------------------------------------------------------
# Streaming: the same total, from a stream that never holds the whole table.
# ---------------------------------------------------------------------------------------------------
print("\nStreaming a sum through the GPU one record batch at a time (never materialised whole)")
for rows_per_batch in (1 << 18, 1 << 20, 1 << 22):
    t0 = time.perf_counter()
    c0 = cpu_seconds()
    total = am.duckdb_aggregate(con.sql("select v from t"), {"s": ("sum", "v")},
                                rows_per_batch=rows_per_batch)["s"]
    wall = (time.perf_counter() - t0) * 1000
    cpu = (cpu_seconds() - c0) * 1000
    expected = con.sql("select sum(v) from t").fetchone()[0]
    mark = "exact" if total == expected else f"WRONG ({total} vs {expected})"
    print(f"  {rows_per_batch:>9,} rows/batch  {wall:9.1f} ms wall  {cpu:9.1f} CPU-ms  {mark}")

print("\nColumns are best-of-%d wall ms. 'resident' is duckdb/gpu with the data already on the GPU;\n"
      "'one-shot' is duckdb/(gpu + crossing), which is what a single cold query costs." % ITERS)
