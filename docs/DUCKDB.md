# ArrowMetal and DuckDB

DuckDB is an excellent query engine. It reads Parquet, plans joins, pushes down predicates, keeps
statistics, spills to disk, and saturates every core you have. ArrowMetal is not a query engine at
all: it is a pile of Metal kernels that run one columnar operation very fast on the GPU that is
already sitting in your Mac.

Those are complementary, and they meet for free. DuckDB hands results out over the Arrow C Data
interface; ArrowMetal takes Arrow buffers in. On Apple silicon the CPU and GPU share one physical
memory pool, so `am_import` copies nothing - it wraps DuckDB's own pages in a Metal buffer. The
address on the far side is the same address. `python/tests/test_duckdb.py` asserts it. Getting to one
contiguous chunk does copy when DuckDB returns many; see §5.

**The summary.** The Python bridge, at 50M rows: a 100,000-key group-by is **16x** faster than
DuckDB's once the column and its group ids are already on the GPU (**1.1x** for a single cold
query), and a string `LIKE` scan is **9.3x** resident (**0.6x** cold). DuckDB is ahead on a plain
`sum` over a column, which is memory-bound and which DuckDB does while it scans. The loadable
extension works and matches DuckDB's answers exactly; it is **behind DuckDB's own SQL in 0.1.0**, and §4 says
why.

- [1. Which tier you want](#1-which-tier-you-want)
- [2. Install](#2-install)
- [3. Tier 1: the Python bridge](#3-tier-1-the-python-bridge)
- [4. Tier 2: the loadable extension](#4-tier-2-the-loadable-extension)
- [5. Numbers](#5-numbers)
- [6. When this is worth it, and when it is not](#6-when-this-is-worth-it-and-when-it-is-not)
- [7. Limits](#7-limits)

---

## 1. Which tier you want

| | Tier 1: Python bridge | Tier 2: loadable extension |
|---|---|---|
| You write | Python around SQL | SQL only |
| Install | `pip install duckdb`, nothing else | build a `.duckdb_extension`, connect with `allow_unsigned_extensions` |
| Data crossing | zero-copy where DuckDB returns one chunk | DataChunks assembled into one buffer (a copy) |
| Types | everything DuckDB emits, including strings, decimals, lists, structs, maps | the fixed-width numeric types, `DATE`, `TIMESTAMP` |
| Speed | resident: 16.0x on 100k-key group-by, 9.3x on `LIKE`, 2.2x on sort; one-shot (crossing included): 1.1x, 0.6x, 0.7x (§5) | **behind DuckDB** in 0.1.0, see §4 |
| Larger than memory | yes, `am.duckdb_batches` | no |

If you are reading this to make something faster: **use tier 1**. Tier 2 exists because "call it from
SQL" is a real requirement for some people, and because the extension is the piece that has to exist
before it can be made fast.

---

## 2. Install

Both tiers need the library built:

```
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift build -c release --product ArrowMetalC
```

### Tier 1

```
pip install duckdb
PYTHONPATH=python python -c "import duckdb, arrowmetal as am; print(am.device_name())"
```

The bridge is imported lazily, so `import arrowmetal` still works with no duckdb installed; the
`am.from_duckdb` attribute is what pulls it in.

### Tier 2

```
./duckdb-extension/build.sh
```

That compiles one translation unit, links `libArrowMetalC.dylib`, and appends the 512-byte metadata
footer DuckDB checks before it will `dlopen` anything. On the first run it downloads `duckdb.h` and
`duckdb_extension.h` into `duckdb-extension/third_party/`; afterwards it needs no network.
`duckdb-extension/CMakeLists.txt` drives the same compile if you would rather use cmake:

```
cmake -S duckdb-extension -B duckdb-extension/build -DCMAKE_BUILD_TYPE=Release
cmake --build duckdb-extension/build
```

Then load it. Unsigned extensions have to be allowed explicitly, and that is a per-connection config
setting, not a `SET` you can issue afterwards:

```python
import duckdb
con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
con.execute("LOAD 'duckdb-extension/build/arrowmetal.duckdb_extension'")
con.sql("SELECT arrowmetal_version(), arrowmetal_device()").show()
```

From the CLI: `duckdb -unsigned`, then `LOAD '.../arrowmetal.duckdb_extension';`.

**Versions.** Built and tested against **DuckDB 1.5.5** (`source_id d8cdaa33fd`), platform
`osx_arm64`, pyarrow 25.0.1, on an Apple M4 Max. The extension declares the **C extension API**
version `v1.2.0`, not the DuckDB version - a `C_STRUCT` extension is compatible with any DuckDB
whose C API is at least what it declares, so it is not pinned to 1.5.5. The **platform** string is
checked strictly: an `osx_arm64` build will not load into an `osx_amd64` DuckDB. Rebuild with
`DUCKDB_PLATFORM=... ./duckdb-extension/build.sh` to target another one.

---

## 3. Tier 1: the Python bridge

`python/arrowmetal/duckdb_bridge.py`. Nine entry points, all reachable straight off `am`; six of
them are below, and the other three are `duckdb_reader`, `duckdb_table` and `duckdb_is_zero_copy`.

### Pull a result onto the GPU

```python
import duckdb, arrowmetal as am
con = duckdb.connect()

cols = am.from_duckdb(con.sql("select region, amount from sales where placed_at >= date '2024-01-01'"))
# -> {"region": MetalArray, "amount": MetalArray}
```

`from_duckdb` takes a `DuckDBPyRelation`, a SQL string with `con=`, a connection holding a pending
result, or any pyarrow table or stream. Columns whose type ArrowMetal cannot lift are left as
pyarrow arrays in the same dict; pass `on_unsupported="raise"` to be told instead.

`test_every_duckdb_type_round_trips` checks 25 of DuckDB's types: every integer width and signedness,
both floats, `BOOLEAN`, `VARCHAR`, `BLOB`, `DATE`, `TIME`, `TIMESTAMP`, `TIMESTAMPTZ`, `INTERVAL`,
`DECIMAL(10,2)`, `DECIMAL(38,2)`, `HUGEINT`, `LIST`, `STRUCT` and `MAP`. `UUID` is checked as its
`VARCHAR` rendering; the native type is not exercised.

### Push results back

```python
gb = am.group_by([cols["region"]])
rel = am.to_duckdb(con, "gpu_totals", {"region": gb.keys()[0], "total": gb.sum(cols["amount"])})

con.sql("select * from gpu_totals order by total desc limit 10").show()
con.sql("select r.name, g.total from gpu_totals g join regions r on r.id = g.region").show()
```

`to_duckdb` registers through `con.register`, which takes the Arrow table over the C Data interface -
DuckDB reads the GPU-produced buffers in place. The view behaves like any other relation: join it,
filter it, insert from it. It lives until `con.unregister(name)`.

### Both halves at once

```python
rel = am.duckdb_gpu_query(
    con,
    """select o.customer, o.amount
       from orders o join customers c on c.id = o.customer
       where o.placed_at >= date '2024-01-01'""",
    then=lambda cols: {
        "customer": am.group_by([cols["customer"]]).keys()[0],
        "total":    am.group_by([cols["customer"]]).sum(cols["amount"]),
    },
    name="per_customer")

rel.order("total desc").limit(10).show()
```

**DuckDB does the scan, the predicate and the join. ArrowMetal does the aggregate.** That is the
whole idea, and it is the arrangement that pays: the join is a planning problem DuckDB is
good at, and the group-by over the join's output is a throughput problem the GPU is good at.

### A fused query over a DuckDB result

`am.query` compiles a whole filter-and-aggregate expression into one Metal kernel, so the column is
read once no matter how many operators the expression has (see [EXPR.md](EXPR.md)):

```python
q = am.filter((am.col("amount") > 100) & am.col("region").is_in([2, 7])).aggregate([
        ("sum", "total", am.col("amount")),
        ("count", "n", None)])
am.query(am.from_duckdb(con.sql("select region, amount from sales")), q)
# {"total": ..., "n": ...}
```

### Tables larger than memory

`fetch_record_batch` gives an Arrow C Stream that produces batches lazily. `am.duckdb_batches`
iterates it as `{name: MetalArray}` with only one batch resident at a time:

```python
total = 0
for batch in am.duckdb_batches(con.sql("select amount from huge_parquet"), rows_per_batch=1 << 22):
    total += batch["amount"].sum()
```

Two wrappers do the merging for you:

```python
am.duckdb_aggregate(con.sql("select amount, region from huge"),
                    {"total": ("sum", "amount"), "n": ("count", "amount"),
                     "regions": ("count_distinct", "region")},
                    rows_per_batch=1 << 22)

am.duckdb_group_by(con.sql("select region, amount from huge"), "region",
                   {"total": ("sum", "amount"), "avg": ("mean", "amount")})
```

**What is exact, and what is refused.** This is the part that usually goes wrong quietly elsewhere,
so the bridge is explicit about it:

| aggregate | exact across batches? | how it merges |
|---|---|---|
| `sum`, `count`, `min`, `max` | **yes** | the merge is the same associative operator as the aggregate |
| `mean` | **yes**, to float64 rounding | total sum over total count - never a mean of batch means, which is wrong for unequal batches |
| `count_distinct` | **yes** | the host unions each batch's distinct values, so memory is proportional to cardinality, not to the batch |
| `median`, `quantile`, `stddev`, `mode`, `tdigest` | **no** | **refused with an error.** These have no exact merge from independent partials, and returning an approximation under the exact aggregate's name would be a lie. Compute them over a whole batch, or over a sample. |

`test_streaming_aggregate_is_batch_size_independent` runs the same sum at 3,000 / 30,000 / 1,000,000
rows per batch and requires all three answers to be identical, and identical to DuckDB's.

---

## 4. Tier 2: the loadable extension

`duckdb-extension/`. Built against DuckDB's **C** extension API, which means it needs no DuckDB
source tree, no C++ ABI match with the host engine, and no rebuild for each DuckDB patch release.
The loader hands it a struct of function pointers and everything after that is plain C.

### The functions

```sql
SELECT arrowmetal_version();                          -- '0.1.0'
SELECT arrowmetal_device();                           -- 'Apple M4 Max'

SELECT * FROM arrowmetal_agg('sales', 'amount');      -- sum, count, min, max, mean
SELECT * FROM arrowmetal_group_by('sales', 'region', 'amount');  -- key, count, sum, min, max
SELECT * FROM arrowmetal_top_k('sales', 'amount', 100);          -- the 100 largest, descending
SELECT * FROM arrowmetal_sort('sales', 'amount');                -- every value, ascending
SELECT * FROM arrowmetal_query('sales',
    '(query (filter (gt (col "amount") (int 100)))
            (aggregate (sum "total" (col "amount")) (count "n")))');
```

The results are ordinary relations, so they join:

```sql
SELECT r.name, g.sum
FROM arrowmetal_group_by('sales', 'region', 'amount') g
JOIN regions r ON r.id = g.key
ORDER BY g.sum DESC;
```

`arrowmetal_query` takes the s-expression from [EXPR.md](EXPR.md) and returns one row per aggregate:
`name`, `value` (a DOUBLE) and `exact` (a BIGINT, non-null when the aggregate is integral). Both
columns exist because a `sum` past 2^53 is exact in `exact` and rounded in `value`, and pretending
otherwise would silently corrupt large integer totals.

Aggregate *functions* - `SELECT am_sum(x) FROM t` - are deliberately **not** registered even though
`duckdb_create_aggregate_function` exists. DuckDB would call them once per 2048-row DataChunk, and a
2048-row GPU dispatch is all latency and no work; the CPU would be ahead and it would look like
ArrowMetal's fault. Table functions get the whole column at once, which is the only shape where a GPU
dispatch amortises.

### It is correct, and DuckDB's own SQL is ahead of it

Every function's answer matches DuckDB's own SQL exactly, nulls included -
`python/tests/test_duckdb.py` checks each one against the equivalent query on the same connection.

**The header compiles**: `am_plan_source` is the typedef and `am_plan_source_create` the function.
`test_the_public_c_header_compiles` compiles the public header as C on every run; the `@extension`
tests skip only when the extension has not been built. Tier 1 does not go through the header and is
unaffected either way.

And plain SQL is ahead of it. The extension's timings are not recorded: `Benchmarks/duckdb_bench.py` measures the Python bridge only, and
`python/tests/test_duckdb.py` checks the extension's answers without timing them.

The same GPU group-by from the Python bridge runs in 4.5 ms at 50M rows (§5), so the time is in the
extension's path. Three things account for it:

1. **The DataChunk assembly is single-threaded.** DuckDB emits ~2048-row chunks and one ArrowMetal
   array is one contiguous buffer, so the extension memcpys chunk after chunk on one thread while
   DuckDB's own aggregate is running on all of them. This is the dominant cost.
2. **`duckdb_query` materialises the whole result first**, so the rows are copied once inside DuckDB
   before the extension has touched them. A streaming result would remove that copy.
3. **The extension re-reads the table on every call**, because a table function has nowhere to cache.
   The bridge pays the crossing once and then runs twenty operations on resident data.

Fixing (1) and (2) is the obvious next work: partition the assembly across threads, and execute the
inner query in streaming mode. None of it changes the extension's interface, so the SQL above is
what it will keep being.

---

## 5. Numbers

Apple M4 Max, DuckDB 1.5.5, pyarrow 25.0.1, ArrowMetal 0.1.0. Best of 5, wall ms, with the process's
CPU time beside it. Every figure in this section is from
`Benchmarks/results/duckdb_bench_50000000_2026-09-07.txt`. Reproduce with:

```
PYTHONPATH=python python Benchmarks/duckdb_bench.py 50000000 5
```

Both sides compute the same answer from the same in-memory DuckDB table, and the benchmark checks
that they agree before it reports a time.

### 50M rows

| operation | DuckDB | CPU-ms | GPU (resident) | CPU-ms | GPU + crossing | resident | one-shot |
|---|---:|---:|---:|---:|---:|---:|---:|
| filter + sum | 1.1 | 13.7 | 3.1 | 0.8 | 42.8 | 0.4x | 0.0x |
| filter + sum, fused query | 1.1 | 13.6 | 1.1 | 0.4 | 40.6 | 1.0x | 0.0x |
| group-by sum, 1k keys | 7.6 | 112.2 | 2.1 | 0.6 | 68.8 | **3.6x** | 0.1x |
| group-by sum, 100k keys | 71.4 | 1056.2 | 4.5 | 0.6 | 65.9 | **16.0x** | **1.1x** |
| top_k(100) | 5.7 | 12.2 | 9.7 | 3.4 | 54.1 | 0.6x | 0.1x |
| sort | 39.2 | 570.4 | 17.6 | 1.4 | 57.3 | **2.2x** | 0.7x |
| `count(s like '%user1%')` | 47.4 | 718.5 | 5.1 | 0.4 | 84.7 | **9.3x** | 0.6x |
| `avg(f)` | 2.8 | 40.8 | 2.5 | 0.4 | 42.8 | **1.1x** | 0.1x |

Look at the CPU-ms columns as well as the wall-ms. DuckDB's 100k-key group-by at 50M rows costs
1,056 CPU-ms to produce 71.4 ms of wall time - it is using about fifteen cores. The GPU's costs 0.6
CPU-ms. If anything else on the machine wants those cores, that difference is the real one.

### What the crossing costs

At 50M rows, one BIGINT column (406 MB):

```
to_arrow_table       19.7 ms   (407 chunks - DuckDB handing over pointers, not bytes)
combine_chunks       26.6 ms   (a real copy: 407 chunks into one buffer, 15.3 GB/s)
am_import             5.7 ms   (buffer shared, not copied - the pointer is unchanged)
                    -------
total                52.1 ms
```

The Metal import copies nothing; **`combine_chunks` is the largest single item**. It is the price
of ArrowMetal wanting one contiguous array per column and DuckDB producing many. `am.duckdb_batches`
avoids it entirely by keeping the chunks separate. Streaming a sum through the GPU one record batch
at a time, never materialising the column:

| rows per batch | wall ms | CPU-ms |
|---:|---:|---:|
| 262,144 | 98.7 | 114.1 |
| 1,048,576 | 67.5 | 78.9 |
| 4,194,304 | 76.1 | 76.0 |

All three are exact. On this run the whole-table crossing (52.1 ms) is ahead of the streaming path
(67.5 ms at its best batch size), so streaming is a memory-footprint argument rather than a speed
one.

This is also why the "one-shot" column above is mostly below 1.0x: for a single operation the
crossing dominates, and DuckDB is ahead. The GPU pays off when you do several things to one dataset,
or one expensive thing.

### 50M rows out of Parquet

```python
con.execute("copy (select i::BIGINT amount, (hash(i) % 1000)::INTEGER region "
            "from range(50000000) r(i)) to 'big.parquet' (format parquet)")

con.sql("select sum(amount) from read_parquet('big.parquet')")
am.from_duckdb(con.sql("select amount from read_parquet('big.parquet')"))["amount"].sum()
am.duckdb_aggregate(con.sql("select amount from read_parquet('big.parquet')"),
                    {"total": ("sum", "amount")}, rows_per_batch=1 << 22)
```

All three give `1249999975000000`. **DuckDB is ahead here**: a single sum over a Parquet file is
exactly the case where DuckDB aggregates during the scan and never materialises anything. Reach for
the GPU when the aggregate is the expensive part, not the scan.

### The matrix rows, DuckDB alongside

[BENCHMARKS_MATRIX.md](BENCHMARKS_MATRIX.md) has no DuckDB column, because DuckDB is a query engine
rather than an array library: the comparison needs a table, a query, and a decision about where the
result lands. `Benchmarks/duckdb_matrix.py` makes those decisions explicit and measures DuckDB on the
matrix's sort, group-by, filter and sum rows, with the matrix's own generators, sizes and best-of-five
protocol, in four idioms: the same SQL over a registered Arrow table in one chunk and in sixteen
batches, and over a native DuckDB table with the result exported to Arrow or kept in a DuckDB temp
table. Every result asserts its row count. Reproduce with:

```
PYTHONPATH=python python Benchmarks/duckdb_matrix.py 10000000 50000000
python Benchmarks/duckdb_matrix.py --report Benchmarks/results/duckdb_matrix_2026-09-12.csv
```

DuckDB rows measured 2026-09-12: duckdb 1.5.5, threads=16, preserve_insertion_order=True, pyarrow 25.0.1, python 3.13.9, Apple M4 Max, macOS 26.6.2, loadavg at start 2.5, 2026-09-12T23:34:30.
ArrowMetal, Polars and pyarrow are the published 2026-09-07 matrix rows (`Benchmarks/results/full_matrix_2026-09-07-parallel.csv`), not re-measured. Wall ms, best of up to 5 after one warm-up, process CPU ms in brackets. "Best" is the faster of a library's eager and parallel idioms. The last column is the fastest CPU number on the row, any library or idiom, over ArrowMetal.

| family | op | rows | ArrowMetal | Polars best | pyarrow best | DuckDB native, to Arrow | DuckDB native, stays in DuckDB | DuckDB Arrow scan, 16 batches | DuckDB Arrow scan, 1 chunk | fastest CPU / ArrowMetal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reductions | sum(int64, 10% nulls) | 10,000,000 | 0.30 (0) | 0.57 (6) | 1.29 (14) | 1.07 (15) | 1.08 (14) | 1.90 (18) | 12.0 | 1.9x |
| compare+select | filter int64 (30% kept) | 10,000,000 | 0.78 (1) | 0.65 (6) | 5.33 (60) | 35.9 (53) | 8.52 (88) | 39.9 (73) | 42.4 | 0.8x |
| sort | argsort int64 | 10,000,000 | 7.81 (1) | 57.6 (655) | 841.2 (841) | 212.7 (647) | 52.3 (674) | 248.6 (688) | 403.7 | 6.7x |
| sort | argsort float64 | 10,000,000 | 8.11 (1) | 78.6 (1,004) | 984.1 (983) | 231.4 (725) | 60.1 (767) | 266.2 (779) | 460.0 | 7.4x |
| sort | sort float64 | 10,000,000 | 6.85 (1) | 26.2 (250) | 1,006.5 (1,006) | 204.9 (628) | 54.6 (718) | 223.6 (669) | 410.9 | 3.8x |
| sort | lexsort (2 int32 keys) | 10,000,000 | 7.29 (2) | 111.4 (1,217) | 1,122.7 (1,122) | 197.5 (635) | 49.9 (660) | 264.6 (893) | 625.9 | 6.8x |
| group-by | sum by int32 key (1000 groups) | 10,000,000 | 1.73 (1) | 20.2 (242) | 4.61 (51) | 2.09 (28) | 2.11 (27) | 7.67 (80) | 53.7 | 1.2x |
| group-by | mean by int32 key (1000 groups) | 10,000,000 | 2.21 (1) | 20.3 (232) | 4.63 (51) | 2.36 (33) | 2.36 (32) | 8.31 (79) | 55.2 | 1.1x |
| group-by | count by int32 key (1000 groups) | 10,000,000 | 1.52 (1) | 21.2 (166) | 4.43 (48) | 2.14 (25) | 2.03 (26) | 6.70 (72) | 48.4 | 1.3x |
| group-by | sum by two int32 keys (~1024 groups) | 10,000,000 | 2.19 (2) | 34.0 (385) | 5.50 (60) | 3.04 (41) | 2.98 (40) | 9.25 (100) | 69.5 | 1.4x |
| group-by | sum by int32 key (100000 groups) | 10,000,000 | 1.95 (1) | 22.7 (275) | 16.4 (98) | 20.5 (292) | 21.0 (295) | 29.3 (357) | 116.4 | 8.4x |
| group-by | mean by int32 key (100000 groups) | 10,000,000 | 2.48 (1) | 23.3 (284) | 16.2 (98) | 24.1 (345) | 24.3 (344) | 29.4 (351) | 112.0 | 6.6x |
| group-by | count by int32 key (100000 groups) | 10,000,000 | 1.96 (1) | 27.4 (208) | 13.0 (81) | 19.8 (279) | 19.9 (282) | 23.3 (287) | 90.1 | 6.6x |
| group-by | sum by int32 key (10000000 groups) | 10,000,000 | 9.46 (2) | 69.5 (942) | 312.4 (627) | 187.4 (815) | 38.0 (524) | 178.0 (880) | 278.4 | 4.0x |
| reductions | sum(int64, 10% nulls) | 50,000,000 | 1.11 (0) | 2.30 (26) | 5.09 (66) | 4.40 (65) | 4.67 (64) | 7.26 (77) | 58.9 | 2.1x |
| compare+select | filter int64 (30% kept) | 50,000,000 | 2.23 (1) | 2.85 (36) | 22.6 (298) | 186.8 (242) | 32.1 (416) | 204.8 (337) | 216.3 | 1.3x |
| sort | argsort int64 | 50,000,000 | 39.1 (1) | 302.0 (3,574) | 5,196.8 (5,196) | 1,525.2 (3,954) | 320.7 (4,340) | 1,471.2 (4,009) | 2,082.6 | 7.7x |
| sort | argsort float64 | 50,000,000 | 39.9 (1) | 420.1 (5,531) | 5,885.3 (5,885) | 1,403.5 (3,854) | 344.0 (4,349) | 1,479.2 (4,014) | 2,358.4 | 8.6x |
| sort | sort float64 | 50,000,000 | 32.1 (1) | 132.9 (1,284) | 6,044.5 (6,044) | 1,308.4 (3,377) | 293.5 (3,866) | 1,248.9 (3,483) | 2,038.2 | 4.1x |
| sort | lexsort (2 int32 keys) | 50,000,000 | 37.6 (2) | 902.9 (9,948) | 7,773.5 (7,773) | 1,253.8 (3,088) | 264.9 (3,532) | 1,557.7 (4,595) | 3,378.6 | 7.0x |
| group-by | sum by int32 key (1000 groups) | 50,000,000 | 4.89 (1) | 81.9 (1,186) | 18.5 (250) | 9.00 (135) | 9.07 (135) | 26.2 (357) | 262.6 | 1.8x |
| group-by | mean by int32 key (1000 groups) | 50,000,000 | 4.91 (1) | 81.5 (1,180) | 18.6 (251) | 10.1 (151) | 10.5 (159) | 30.4 (350) | 265.5 | 2.0x |
| group-by | count by int32 key (1000 groups) | 50,000,000 | 4.23 (1) | 89.0 (810) | 17.4 (235) | 8.35 (126) | 8.37 (126) | 25.6 (318) | 241.0 | 2.0x |
| group-by | sum by two int32 keys (~1024 groups) | 50,000,000 | 5.91 (2) | 164.2 (2,216) | 23.9 (322) | 13.9 (199) | 14.0 (201) | 37.3 (434) | 344.6 | 2.3x |
| group-by | sum by int32 key (100000 groups) | 50,000,000 | 7.41 (1) | 99.4 (1,461) | 46.4 (542) | 76.1 (1,087) | 75.9 (1,130) | 92.3 (1,248) | 444.2 | 6.3x |
| group-by | mean by int32 key (100000 groups) | 50,000,000 | 7.39 (1) | 99.2 (1,458) | 47.0 (540) | 86.5 (1,277) | 85.5 (1,291) | 93.9 (1,290) | 445.1 | 6.4x |
| group-by | count by int32 key (100000 groups) | 50,000,000 | 4.92 (1) | 120.7 (1,060) | 38.3 (421) | 64.9 (974) | 67.2 (992) | 70.8 (937) | 400.0 | 7.8x |
| group-by | sum by int32 key (10000000 groups) | 50,000,000 | 46.1 (2) | 321.7 (4,585) | 1,303.3 (3,682) | 324.4 (2,356) | 148.6 (2,210) | 360.1 (2,712) | 1,371.0 | 3.2x |
| sort | sort utf8 | 1,000,000 | - | - | - | 25.8 (84) | 11.3 (98) | 30.6 (104) | 70.4 | - |
| sort | sort utf8 | 10,000,000 | 23.0 (7) | 133.1 (1,326) | 1,864.9 (1,865) | 295.0 (852) | 79.4 (1,063) | 345.7 (926) | 723.7 | 3.5x |

SQL per operation, over the native table; the Arrow-scan idioms run the same text over the registered Arrow table, and the stays-in-DuckDB idiom wraps it in `CREATE OR REPLACE TEMP TABLE r AS`:

- sum(int64, 10% nulls): `SELECT sum(x) FROM tn` (DuckDB sum(BIGINT) returns HUGEINT)
- filter int64 (30% kept): `SELECT x FROM tn WHERE m`
- argsort int64: `SELECT i FROM tn ORDER BY x` (DuckDB has no argsort)
- argsort float64: `SELECT i FROM tn ORDER BY x` (DuckDB has no argsort)
- sort float64: `SELECT x FROM tn ORDER BY x`
- lexsort (2 int32 keys): `SELECT i FROM tn ORDER BY a, b` (DuckDB has no argsort)
- sum by int32 key (1000 groups): `SELECT k, sum(x) FROM tn GROUP BY k`
- mean by int32 key (1000 groups): `SELECT k, avg(x) FROM tn GROUP BY k`
- count by int32 key (1000 groups): `SELECT k, count(x) FROM tn GROUP BY k`
- sum by two int32 keys (~1024 groups): `SELECT a, b, sum(x) FROM tn GROUP BY a, b`
- sum by int32 key (100000 groups): `SELECT k, sum(x) FROM tn GROUP BY k`
- mean by int32 key (100000 groups): `SELECT k, avg(x) FROM tn GROUP BY k`
- count by int32 key (100000 groups): `SELECT k, count(x) FROM tn GROUP BY k`
- sum by int32 key (10000000 groups): `SELECT k, sum(x) FROM tn GROUP BY k`
- sort utf8: `SELECT s FROM tn ORDER BY s`

What the table says:

- **DuckDB is the fastest CPU engine on low-cardinality group-by and on multi-key sort**, and it is
  a long way ahead of Polars on both: 9.0 ms against Polars' 81.9 for `sum by int32 key` at 1,000
  groups and 50M rows, 265 ms against 903 for `lexsort`. Against DuckDB those two ArrowMetal rows are
  1.8x and 7.0x, where the matrix, whose fastest CPU idiom is Polars or Acero, shows 16.8x and 24.0x.
  At 10M rows and 1,000 groups the margin is 1.2x. This is the honest group-by story: the GPU is ahead
  by 3x or more from 100,000 groups upward and roughly even with DuckDB at a thousand groups.
- **DuckDB does not win every CPU row.** At 100,000 groups Acero is faster than DuckDB (46.4 ms
  against 75.9 at 50M rows), on `sort float64` Polars is (133 against 294), and on the filter and the
  plain sum Polars lazy is the fastest CPU idiom by a wide margin. The fastest-CPU column picks the
  winner per row.
- **Exporting the result to Arrow is a large part of DuckDB's wall time on wide results.** Sorting 50M
  float64 takes 294 ms inside DuckDB and 1,308 ms delivered as an Arrow table; the argsorts add about
  1.1 to 1.2 s the same way. A Python caller pays that; ArrowMetal's result is already an Arrow buffer. For a
  group-by result of a thousand rows the two idioms are the same.
- **The CPU-time gap does not narrow.** DuckDB's 1,000-group sum at 50M rows costs 135 CPU-ms across
  about fifteen cores, its sorts 3,100 to 4,350; the ArrowMetal rows cost 1.2 to 2.4 CPU-ms.
- **A registered Arrow table in one chunk scans on two cores.** That idiom is 1.2x slower than the
  native table on the filter and 29x slower on the 1,000-group sum. Register a chunked table, or
  create a DuckDB table, before benchmarking DuckDB over Arrow data.

Two things to carry with the numbers. The ArrowMetal, Polars and pyarrow rows were measured on
2026-09-07 and the DuckDB rows on 2026-09-12, on the same machine with the same protocol, so the
day differs. And the `sort` row in the 50M table above (39.2 ms) sorts `v = i`, a column that is
already in order; on uniform random float64 of the same size DuckDB's sort takes 294 ms.

---

## 6. When this is worth it, and when it is not

**Worth it**

- High-cardinality group-by over a resident column. 100,000 keys at 50M rows: 16x, at 0.6 CPU-ms
  against DuckDB's 1,056; a single cold query, crossing included, is 1.1x.
- String matching over a column you are keeping resident. `LIKE '%...%'` at 50M rows: 9.3x resident;
  a single cold query is 0.6x.
- Sorting a resident column. 2.2x at 50M rows (17.6 ms against DuckDB's 39.2), and 1.4 CPU-ms
  against 570.
- Several operations over one dataset. The crossing is paid once; every kernel after that is free of
  it. This is the single biggest lever - `from_duckdb` once, then loop.
- Anything where you want the cores back. The GPU group-by uses 0.6 CPU-ms where DuckDB uses 1,056.
- Fused filter-and-aggregate expressions, which read the column once no matter how long they are.

**Not worth it**

- A single `sum`/`avg` over one column, whether from memory or Parquet. Memory-bound, and DuckDB
  does it while scanning.
- Sorting from cold. The GPU radix sort is ahead of DuckDB's parallel sort on a resident column
  (2.2x) but behind once the crossing is included (0.7x one-shot).
- Small data. Below a few million rows the dispatch latency is most of the time — about 60-70 µs
  measured ([RESIDENT.md](RESIDENT.md)), 110-160 µs per call for sum and filter at 1,000 rows in the
  matrix's latency family; use
  `am.batch()` to amortise it, or do not bother.
- Anything the extension can do, if speed is the reason. See §4.

**The rule of thumb**: the GPU is ahead when the compute is large relative to the bytes. Group-by with
many keys and string matching are compute-heavy per byte; `sum` is not.

---

## 7. Limits

**Tier 1**

- One `MetalArray` is one contiguous array, so a multi-chunk DuckDB result is concatenated on the
  way in. At 50M rows that is 26.6 ms (§5). `am.duckdb_batches` skips it.
- `to_duckdb` registers a view on one connection; other connections to the same database will not
  see it. Use `CREATE TABLE ... AS SELECT * FROM the_view` to make it durable.
- Nested columns (list, struct, map) import and round-trip, but only the structural kernels operate
  on them - you cannot group by a struct.
- Streaming refuses `median`, `quantile`, `stddev`, `mode` and `tdigest` by design (§3).

**Tier 2**

- Behind DuckDB's own SQL in 0.1.0 (§4).
- The eight integer widths, `FLOAT`, `DOUBLE`, `DATE` and `TIMESTAMP` only -- exactly what
  `arrow_format_for` in `duckdb-extension/src/arrowmetal_extension.cpp` lists. `BOOLEAN` is **not**
  among them; nor are `VARCHAR`, `DECIMAL`, `TIME`,
  `TIMESTAMPTZ`, `HUGEINT` or the nested types. All of them are refused by name at bind time with a
  message pointing here, and the bridge handles every one.
- The table functions take a **table or view name**, not a subquery - DuckDB's C table-function API
  has no way to accept a relation. A Python-registered relation (`con.register(...)`) is
  connection-local and the extension's own connection cannot see it; `CREATE VIEW` first.
- `arrowmetal_group_by` returns `key` as BIGINT or DOUBLE, so a key column wider than that (there is
  none among the supported types) would not fit.
- `arrowmetal_query` accepts only an `(aggregate ...)` terminal. `(project ...)` and `(group_by ...)`
  produce columns, which this function's fixed output schema cannot carry; use `arrowmetal_group_by`
  or the bridge.
- The extension opens one DuckDB connection when it loads and keeps it for the life of the process,
  serialised by a mutex. It cannot connect lazily instead: the `duckdb_database` handle the loader
  passes does not outlive the load call. That connection is never closed, so the database it belongs
  to is not freed until the process exits.
- Unsigned. `allow_unsigned_extensions` has to be set when the connection is created.
- Built for one platform string at a time; `osx_arm64` unless you set `DUCKDB_PLATFORM`.

## Files

| Path | What |
|---|---|
| `python/arrowmetal/duckdb_bridge.py` | tier 1, the whole bridge |
| `duckdb-extension/src/arrowmetal_extension.cpp` | tier 2, the extension |
| `duckdb-extension/build.sh` | builds the extension without cmake |
| `duckdb-extension/CMakeLists.txt` | the same build, for cmake |
| `duckdb-extension/scripts/append_metadata.py` | writes DuckDB's 512-byte extension footer |
| `python/tests/test_duckdb.py` | both tiers, with DuckDB's own SQL as the oracle |
| `Benchmarks/duckdb_bench.py` | the tables in §5 |
