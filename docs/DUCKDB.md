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

**The honest summary, before anything else.** The Python bridge is real and it wins on the shapes it
should win on: a 100,000-key group-by is **24x** faster than DuckDB's once the column and its group
ids are already on the GPU (**1.2x** for a single cold query), and a string `LIKE` scan **21x**
resident (**0.7x** cold), both at 50M rows. It loses on shapes DuckDB is already excellent at - a
plain `sum` over a column is memory-bound and DuckDB does it while it scans. The loadable extension
works, matches DuckDB's answers exactly, and is **not currently a speedup**; §4 says why, in detail,
without softening it.

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
| Speed | **faster than DuckDB** on high-cardinality group-by, string matching, fused filter+aggregate | **slower than DuckDB** today, see §4 |
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

`python/arrowmetal/duckdb_bridge.py`. Eight entry points, all reachable straight off `am`; six of
them are below.

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
whole idea, and it is the arrangement that actually pays: the join is a planning problem DuckDB is
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
2048-row GPU dispatch is all latency and no work; it would be slower than the CPU and it would look
like ArrowMetal's fault. Table functions get the whole column at once, which is the only shape a GPU
can win in.

### It is correct, and it is slower

Every function's answer matches DuckDB's own SQL exactly, nulls included -
`python/tests/test_duckdb.py` checks each one against the equivalent query on the same connection.

**The header compiles**: `am_plan_source` is the typedef and `am_plan_source_create` the function.
`test_the_public_c_header_compiles` compiles the public header as C on every run; the `@extension`
tests skip only when the extension has not been built. Tier 1 does not go through the header and is
unaffected either way.

And it is slower than just writing the SQL:

| 10M rows | DuckDB SQL | extension |
|---|---:|---:|
| `sum, count, min, max, avg` | 2.4 ms | 15.5 ms |
| group-by, 100k keys | 92.8 ms | 124.8 ms |
| `order by v desc limit 100` | 2.8 ms | 15.6 ms |
| full sort | 11.5 ms | 77.1 ms |

| 50M rows | DuckDB SQL | extension |
|---|---:|---:|
| `sum, count, min, max, avg` | 9.0 ms | 86.5 ms |
| group-by, 100k keys | 197.8 ms | 434.1 ms |
| `order by v desc limit 100` | 7.2 ms | 105.9 ms |
| full sort | 64.0 ms | 540.4 ms |

(The group-by rows compute all four aggregates at once, which is what `arrowmetal_group_by` returns,
so they are not comparable to the single-`sum` group-by in §5.)

The kernels are not the problem - the same GPU group-by from the Python bridge runs in 4.6 ms at 50M
rows, against these hundreds. Three things in the extension's path are:

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
CPU time beside it. Reproduce with:

```
PYTHONPATH=python python Benchmarks/duckdb_bench.py 50000000 5
```

Both sides compute the same answer from the same in-memory DuckDB table, and the benchmark checks
that they agree before it reports a time.

### 50M rows

| operation | DuckDB | CPU-ms | GPU (resident) | CPU-ms | GPU + crossing | resident | one-shot |
|---|---:|---:|---:|---:|---:|---:|---:|
| filter + sum | 2.5 | 13.2 | 3.7 | 1.1 | 76.0 | 0.7x | 0.0x |
| filter + sum, fused query | 2.6 | 13.7 | 1.4 | 0.6 | 59.0 | **1.8x** | 0.0x |
| group-by sum, 1k keys | 15.0 | 140.3 | 2.4 | 0.7 | 85.3 | **6.3x** | 0.2x |
| group-by sum, 100k keys | 113.8 | 1169.5 | 4.6 | 0.9 | 91.7 | **24.5x** | **1.2x** |
| top_k(100) | 8.2 | 16.5 | 10.9 | 3.8 | 79.2 | 0.7x | 0.1x |
| sort | 74.0 | 705.9 | 153.6 | 1.1 | 304.0 | 0.5x | 0.2x |
| `count(s like '%user1%')` | 110.2 | 1104.3 | 5.3 | 0.5 | 152.5 | **20.9x** | 0.7x |
| `avg(f)` | 6.1 | 50.3 | 3.1 | 0.5 | 63.7 | **2.0x** | 0.1x |

### 10M rows

| operation | DuckDB | CPU-ms | GPU (resident) | CPU-ms | GPU + crossing | resident | one-shot |
|---|---:|---:|---:|---:|---:|---:|---:|
| filter + sum | 0.6 | 3.3 | 1.3 | 0.6 | 8.8 | 0.4x | 0.1x |
| filter + sum, fused query | 0.5 | 3.3 | 0.4 | 0.4 | 7.1 | **1.3x** | 0.1x |
| group-by sum, 1k keys | 3.1 | 26.0 | 1.3 | 0.6 | 19.2 | **2.4x** | 0.2x |
| group-by sum, 100k keys | 33.5 | 338.9 | 1.6 | 0.7 | 18.9 | **20.9x** | **1.8x** |
| top_k(100) | 3.2 | 6.9 | 6.1 | 3.5 | 16.6 | 0.5x | 0.2x |
| sort | 14.3 | 130.4 | 29.4 | 1.0 | 38.5 | 0.5x | 0.4x |
| `count(s like '%user1%')` | 18.1 | 183.3 | 3.0 | 0.5 | 26.2 | **6.1x** | 0.7x |
| `avg(f)` | 2.6 | 12.9 | 1.6 | 0.6 | 16.2 | **1.6x** | 0.2x |

Look at the CPU-ms columns as well as the wall-ms. DuckDB's 100k-key group-by at 50M rows costs
1,169 CPU-ms to produce 113.8 ms of wall time - it is using about ten cores. The GPU's costs 0.9
CPU-ms. If anything else on the machine wants those cores, that difference is the real one.

### What the crossing costs

At 50M rows, one BIGINT column (406 MB):

```
to_arrow_table      110.8 ms   (407 chunks - DuckDB handing over pointers, not bytes)
combine_chunks      165.2 ms   (a real copy: 407 chunks into one buffer)
am_import             9.6 ms   (buffer shared, not copied - the pointer is unchanged)
                    -------
total               285.6 ms
```

The Metal import is genuinely free; **`combine_chunks` is not**. It is the price of ArrowMetal
wanting one contiguous array per column and DuckDB producing many. `am.duckdb_batches` avoids it
entirely by keeping the chunks separate, which is why the streaming sum at 50M rows (91 ms) beats
the whole-table path's crossing (286 ms).

This is also why the "one-shot" column above is mostly below 1.0x: for a single operation, the
crossing dominates and DuckDB was going to win anyway. The GPU pays off when you do several things
to one dataset, or one expensive thing.

### 50M rows out of Parquet

```python
con.execute("copy (select i::BIGINT amount, (hash(i) % 1000)::INTEGER region "
            "from range(50000000) r(i)) to 'big.parquet' (format parquet)")   # 265 MB

con.sql("select sum(amount) from read_parquet('big.parquet')")        #  30 ms
am.from_duckdb(con.sql("select amount from read_parquet('big.parquet')"))["amount"].sum()   # 89 ms
am.duckdb_aggregate(con.sql("select amount from read_parquet('big.parquet')"),
                    {"total": ("sum", "amount")}, rows_per_batch=1 << 22)    # 289 ms
```

All three give `1249999975000000`. **DuckDB wins this one**, and it is not close: a single sum over a
Parquet file is exactly the case where DuckDB aggregates during the scan and never materialises
anything. Reach for the GPU when the aggregate is the expensive part, not the scan.

---

## 6. When this is worth it, and when it is not

**Worth it**

- High-cardinality group-by over a resident column. 100,000 keys at 50M rows: 24x, at a thousandth of
  the CPU time; a single cold query, crossing included, is 1.2x.
- String matching over a column you are keeping resident. `LIKE '%...%'` at 50M rows: 21x resident;
  a single cold query is 0.7x.
- Several operations over one dataset. The crossing is paid once; every kernel after that is free of
  it. This is the single biggest lever - `from_duckdb` once, then loop.
- Anything where you want the cores back. The GPU group-by uses 0.9 CPU-ms where DuckDB uses 1,169.
- Fused filter-and-aggregate expressions, which read the column once no matter how long they are.

**Not worth it**

- A single `sum`/`avg` over one column, whether from memory or Parquet. Memory-bound, and DuckDB
  does it while scanning.
- Sorting. DuckDB's parallel sort beats the GPU radix sort at these sizes (0.5x).
- Small data. Below a few million rows the dispatch latency is most of the time — about 60-70 µs
  measured ([RESIDENT.md](RESIDENT.md)), 110-230 µs as an all-in per-call floor in the matrix's
  latency family; use
  `am.batch()` to amortise it, or do not bother.
- Anything the extension can do, if speed is the reason. See §4.

**The rule of thumb**: the GPU wins when the compute is large relative to the bytes. Group-by with
many keys and string matching are compute-heavy per byte; `sum` is not.

---

## 7. Limits

**Tier 1**

- One `MetalArray` is one contiguous array, so a multi-chunk DuckDB result is concatenated on the
  way in. At 50M rows that is 165 ms. `am.duckdb_batches` skips it.
- `to_duckdb` registers a view on one connection; other connections to the same database will not
  see it. Use `CREATE TABLE ... AS SELECT * FROM the_view` to make it durable.
- Nested columns (list, struct, map) import and round-trip, but only the structural kernels operate
  on them - you cannot group by a struct.
- Streaming refuses `median`, `quantile`, `stddev`, `mode` and `tdigest` by design (§3).

**Tier 2**

- Slower than DuckDB today (§4).
- The eight integer widths, `FLOAT`, `DOUBLE`, `DATE` and `TIMESTAMP` only -- exactly what
  `arrow_format_for` in `duckdb-extension/src/arrowmetal_extension.cpp` lists. `BOOLEAN` is **not**
  among them despite what an earlier draft of this table said; nor are `VARCHAR`, `DECIMAL`, `TIME`,
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
