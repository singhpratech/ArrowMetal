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
extension works and matches DuckDB's answers exactly; it is **behind DuckDB's own SQL** as measured, and §4 says
why.

A third piece needs no change to the SQL at all. The **rewrite extension** is a DuckDB optimizer
extension: loaded into a connection, it moves eligible aggregates of ordinary queries onto the GPU, with
DuckDB's exact answers, and in its default mode only for the shape classes whose every benchmarked
query was ahead of DuckDB's own operators, from the size where that held. §4b has what it rewrites,
the feasibility finding behind it, and where it is ahead and where it is not.

- [1. Which tier you want](#1-which-tier-you-want)
- [2. Install](#2-install)
- [3. Tier 1: the Python bridge](#3-tier-1-the-python-bridge)
- [4. Tier 2: the loadable extension](#4-tier-2-the-loadable-extension)
- [4b. Tier 3: ordinary SQL, rewritten](#4b-tier-3-ordinary-sql-rewritten)
- [5. Numbers](#5-numbers)
- [6. When this is worth it, and when it is not](#6-when-this-is-worth-it-and-when-it-is-not)
- [7. Limits](#7-limits)

---

## 1. Which tier you want

| | Tier 1: Python bridge | Tier 2: loadable extension | Tier 3: rewrite extension |
|---|---|---|---|
| You write | Python around SQL | SQL only | SQL only, unchanged |
| Install | `pip install duckdb`, nothing else | build a `.duckdb_extension`, connect with `allow_unsigned_extensions` | build it against the installed DuckDB release (`build_rewrite.sh`), connect with `allow_unsigned_extensions`, or `am.duckdb_connect()` |
| Data crossing | zero-copy where DuckDB returns one chunk | DataChunks assembled into one buffer (a copy) | DuckDB's scan output copied into page-aligned buffers the GPU reads in place |
| Types | everything DuckDB emits, including strings, decimals, lists, structs, maps | the fixed-width numeric types, `DATE`, `TIMESTAMP` | integer columns (`MIN`/`MAX` also `DATE`, `TIMESTAMP`); an integer, `DATE`, `TIMESTAMP` or `VARCHAR` group key |
| Speed | resident: 16.0x on 100k-key group-by, 9.3x on `LIKE`, 2.2x on sort; one-shot (crossing included): 1.1x, 0.6x, 0.7x (§5) | **behind DuckDB** in 0.1.0, see §4 | in `auto`, rewrites only the shape classes whose every benchmarked query was ahead of DuckDB, from that size (§4b) |
| Larger than memory | yes, `am.duckdb_batches` | no | no: the aggregate's input is gathered in memory (a streamed plan holds only the blocks in flight, §4b) |

If you are reading this to make something faster: **use tier 1**, or tier 3 when the SQL must stay as it
is and its aggregates fall in §4b's table. Tier 2 exists because "call it from SQL" is a real
requirement for some people.

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

### Tier 3

```
PYTHON=/path/to/the/python/with/duckdb ./duckdb-extension/build_rewrite.sh
```

It reads the DuckDB release and `source_id` from that Python's `duckdb` module, shallow-clones the
matching DuckDB tag into `duckdb-extension/build/duckdb-<version>/` on first run (only the headers are
used; nothing of DuckDB is compiled), checks the clone's commit against the `source_id`, compiles
`src/arrowmetal_rewrite.cpp`, checks that the module exports every DuckDB symbol the extension needs,
and stamps the footer with the `CPP` ABI and that exact release. The result is
`duckdb-extension/build/arrowmetal_rewrite.duckdb_extension`. Built and tested against DuckDB 1.5.5
(`source_id d8cdaa33fd`) in the duckdb 1.5.5 Python module.

```python
import arrowmetal as am
con = am.duckdb_connect()     # duckdb.connect + allow_unsigned_extensions + LOAD + SET arrowmetal_rewrite = 'auto'
```

or by hand: `duckdb.connect(config={"allow_unsigned_extensions": "true"})`, then
`LOAD '.../arrowmetal_rewrite.duckdb_extension'`.

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
SELECT arrowmetal_version();                          -- '0.2.0'
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
   before the extension has touched them.
3. **The extension re-reads the table on every call**, because a table function has nowhere to cache.
   The bridge pays the crossing once and then runs twenty operations on resident data.

---

## 4b. Tier 3: ordinary SQL, rewritten

`duckdb-extension/src/arrowmetal_rewrite.cpp` is a DuckDB **optimizer extension**. The query does not
change. After DuckDB's own optimizers have run, the extension looks at every aggregate in the plan, and
where the shape is one it answers exactly (and, in the default mode, the size is one where the GPU was
measured ahead) it puts `ARROWMETAL_AGGREGATE` where DuckDB's `HASH_GROUP_BY` or `UNGROUPED_AGGREGATE`
would have been. The scan, the pushed-down filters and the projections below it are still DuckDB's,
planned and run exactly as before; so is everything above it.

```sql
LOAD 'duckdb-extension/build/arrowmetal_rewrite.duckdb_extension';

SELECT region, sum(amount), count(*), max(amount) FROM sales GROUP BY region;   -- as written

EXPLAIN SELECT region, sum(amount) FROM sales GROUP BY region;   -- ARROWMETAL_AGGREGATE when rewritten
SELECT * FROM arrowmetal_rewrites();    -- every decision, with the reason, the threshold and the GPU path

SET arrowmetal_rewrite = 'auto';        -- the default: supported shapes, at the sizes measured faster
SET arrowmetal_rewrite = 'off';         -- plain DuckDB
SET arrowmetal_rewrite = 'force';       -- every supported shape, whatever its size
```

The three values are read in any letter case, and `'off'` is also spelled `'false'` or `'0'`. Any other
value (`'on'`, `'true'`, `' force'` with a space, `NULL`) is an error from the `SET` itself,
`arrowmetal_rewrite: unrecognised value 'on'; expected 'auto', 'off' (also 'false' or '0') or 'force'`,
and the mode stays what it was. `RESET arrowmetal_rewrite` returns to `'auto'`.

From Python, `am.duckdb_connect()` opens a connection with the extension loaded,
`am.duckdb_is_rewritten(con, sql)` says whether the plan of `sql` has an `ARROWMETAL_AGGREGATE`, and
`am.duckdb_rewrites(con)` returns the decision log as a `pyarrow.Table`.

### Why a C++ extension: the feasibility finding

- **DuckDB's C extension API has no optimizer hook.** Tier 2 is built on it. Its header
  (`duckdb_extension.h`, the v1.2.0 C API that DuckDB 1.5.5 ships) has one planner-adjacent entry
  point, `duckdb_add_replacement_scan`, which swaps a table function in for a table name before
  binding; nothing in it sees or changes a logical plan.
- **The C++ API has one.** `OptimizerExtension` (`duckdb/optimizer/optimizer_extension.hpp`) takes an
  `optimize_function` that DuckDB calls with the whole logical plan after its own optimizers, registered
  with `OptimizerExtension::Register(DBConfig &, ...)`. `LogicalExtensionOperator` and
  `PhysicalOperator` let the extension put its own operator into the plan.
- **A C++ extension loads into the Python module.** It leaves DuckDB's symbols unresolved and takes
  them from the process that loads it, and the duckdb 1.5.5 Python module exports DuckDB's C++
  symbols, `OptimizerExtension::Register` among them. Compiled against the v1.5.5 headers (the module's
  `source_id`, `d8cdaa33fd`) with `-undefined dynamic_lookup` and stamped `CPP` / `v1.5.5`, the
  extension loads into that module with `allow_unsigned_extensions`. No DuckDB build is needed, only
  its headers. `build_rewrite.sh` checks the symbols on every build, because a missing one would abort
  the process at its first call instead of failing the `LOAD`.
- **So the Python-level alternative is not needed.** Intercepting queries in a wrapper and routing them
  by their `EXPLAIN` would work only from Python and only for queries sent through the wrapper; the
  optimizer route works for any client of the loaded connection. `am.duckdb_connect` only loads it.

The price of the C++ route is the pin: DuckDB loads a `CPP` extension only into the exact release in
its footer, where the C-API extension of tier 2 loads into any DuckDB with C API 1.2 or later. A new
DuckDB release means rebuilding with `build_rewrite.sh`, which fetches that release's headers, and
possibly changing the source, since DuckDB's C++ API is not stable across releases: against the 1.4.5
headers it does not compile (`OptimizerExtension::Register` and `PhysicalOperator::GetDataInternal`
are not there yet).

### What is rewritten

| | Eligible |
|---|---|
| Aggregates | none (`SELECT k FROM t GROUP BY k`), or any of: `sum` and `avg` over integer columns of every width and signedness except `UBIGINT` (which DuckDB sums through a cast to `HUGEINT`); `min` and `max` over integer, `DATE` and `TIMESTAMP` columns; `count(x)`, `count(*)`. The argument is a column, or a column under the widening integer cast DuckDB inserts itself (`sum` over `TINYINT` is `sum(CAST(x AS BIGINT))`). No `DISTINCT`, `FILTER` or `ORDER BY` inside the aggregate. |
| Grouping | none, or one column that is an integer, `DATE`, `TIMESTAMP` or `VARCHAR` column. No `GROUPING SETS`, `ROLLUP` or `CUBE`. |
| Input | projections and filters over one table function whose row count DuckDB's planner knows: a table's `seq_scan`, `read_parquet`. A join, a window or another aggregate below the aggregate, or a source that does not report its size (a Python-registered Arrow table's `arrow_scan`), leaves it to DuckDB. |

What DuckDB has already done to the plan stays done: the filters it pushed into the scan, its
compressed materialization of the group key, its rewrite of `sum(x + 1)` into `sum(x)` plus a count,
and its switch from `sum` to `sum_no_overflow` where its statistics prove the total fits in 64 bits.
`test_unsupported_shapes_are_left_alone` covers the shapes left to DuckDB and the reason the log gives
for each; `test_a_parquet_scan` and `test_a_registered_arrow_table_is_left_to_duckdb` the two sources.

### Exactly DuckDB's answers

`python/tests/test_duckdb_rewrite.py` runs every query twice on the same connection, with
`arrowmetal_rewrite` off and forced, and requires the same column types and the same values bit for
bit, `avg` included; only the order of an unordered `GROUP BY` may differ, since SQL does not define
it, and with `ORDER BY` the order is compared too. Every test that takes a connection runs twice,
once where its table fits one block and once with 2,048-row blocks, so the streamed path is covered
by the same queries. The tables come from `hash()` of the row number and reach both ends of every
integer type.

- **Integer sums are exact to `HUGEINT`.** DuckDB's `sum` over an integer column returns `HUGEINT`. A
  64-bit column is summed as its high and low 32-bit halves, each of which fits in 64 bits over fewer
  than 2^31 rows, and the halves are recombined in 128 bits; where DuckDB's statistics already proved
  the total fits in 64 bits (`sum_no_overflow`), it is summed directly, and narrower columns cannot
  leave 64 bits. `test_sums_past_int64_are_exact_hugeints` sums values at both ends of `BIGINT`.
- **`avg` uses DuckDB's own arithmetic**: `double(sum) / double(count)` over `SMALLINT`, and
  `Hugeint::Cast<long double>(sum) / count` over `INTEGER` and `BIGINT`, the two finalizers in DuckDB's
  `avg.cpp`, so the doubles are the same doubles.
- **NULLs**: `count(x)` skips them; `sum`, `avg`, `min` and `max` of a group with no value are NULL;
  an empty input gives one row of zero counts and NULLs without `GROUP BY` and no rows with it; the
  NULL key is one group; a table whose keys are all NULL is one group. Each has a test.
- **Floating-point `sum` and `avg` are not rewritten**: DuckDB's own float sums depend on how its
  threads split the input, so there is no single answer to match. `min` and `max` over floats are not
  rewritten either: DuckDB orders NaN above every number, where ArrowMetal's `min`/`max` skip NaN.

### How it runs

- **The sink.** `ARROWMETAL_AGGREGATE` is a parallel sink. Each DuckDB thread reserves row positions
  for its chunk with one atomic add and copies the aggregate's input columns into page-aligned
  buffers without a lock; a NULL clears its bit in an Arrow validity bitmap. A string key reserves its
  rows and bytes together under a lock, so the offsets stay in row order.
- **Slabs.** A fixed-width column's buffer comes from a pool kept across queries. Each slab is
  imported into ArrowMetal once, over its whole capacity; a query writes its rows into the same
  memory and hands ArrowMetal a zero-copy slice of the first rows, so the GPU wrapping is made once
  per slab rather than once per query. The pool keeps at most 4 GB of mappings, least recently used
  out first.
- **Three GPU paths.** Without `GROUP BY`, one fused query computes every aggregate in one pass
  (`am_query`, [EXPR.md](EXPR.md)). With an integer key whose values span at most 2^20, the fused dense
  group-by keeps one slot per key value, split into several passes when the table would not fit in
  threadgroup memory. Anything else - a `VARCHAR` key, a wider key range, a 64-bit `min`/`max` under a
  `GROUP BY` - takes the hash group-by (`am_group_by_keys` and `am_group_agg_ex`). A short `VARCHAR`
  key that DuckDB's compressed materialization has already turned into an integer arrives as that
  integer and takes the integer paths.
- **Streamed plans.** An ungrouped aggregate, and a group-by whose key DuckDB's statistics put within
  65,536 values (with `min`/`max` over at most 32-bit columns), is processed in blocks of
  `arrowmetal_rewrite_block_rows` rows (default 16,777,216). Each block goes to a GPU worker thread the
  moment its last row lands, while DuckDB is still scanning; Finalize runs the last, partly filled one
  and merges the per-block results on the host, exactly (sums in 128 bits). The other plans take up to
  2^31 - 1 rows, the GPU kernels' 32-bit row index; a streamed plan is not held to that, since each
  block is smaller and the partial sums are added in 128 bits.
- **Rows past the reservation.** The buffers are sized from the source's row count at planning time.
  Rows beyond that - a prepared statement run after the table grew - are kept aside and appended in
  Finalize (`test_a_prepared_statement_after_the_table_grew`, `test_a_streamed_plan_past_its_block_directory`).
- **One GPU.** Rewritten queries on different connections take turns on the GPU, under one
  process-wide lock (`test_concurrent_queries_on_two_connections`).

### When `auto` rewrites

Two conditions, both recorded for each decision in `arrowmetal_rewrites()`:

1. **The router's crossover.** DuckDB's estimate of the rows reaching the aggregate is at or above the
   largest crossover among the query's aggregates in `Benchmarks/results/router_2026-09-24.json`: the
   `reductions` rows without `GROUP BY`, the `group-by` rows with it, in the 1,000-group or
   100,000-group class by DuckDB's estimate of the group count (split at 10,000, the geometric middle),
   and the utf8 rows for a `VARCHAR` key. `test_crossovers_match_the_router_sweep` holds the constants to
   the JSON. The constants are read from `Benchmarks/results/router_2026-09-24.json`, which supersedes the
   2026-09-17 sweep; of the rows used here only `min(int64, 10% nulls)` moved between the two,
   50,000,000 then and 10,000,000 now. `min` without `GROUP BY` is gated by the
   ungrouped class's floor of 50,000,000 rows below (or not rewritten, with one or two aggregates), so no
   `auto` decision changes; at 10,000,000 rows the reason for such a query now reads "below the measured
   floor" where the results file, measured before the change, says "below the crossover".
2. **A measured floor.** The router's crossovers compare kernels on data already on the GPU. Here every
   row is first copied out of DuckDB's scan, and DuckDB's own aggregate runs while it scans, so the
   query's shape class must also have been measured faster than DuckDB's operators, from the size in
   this table, by `Benchmarks/duckdb_rewrite_bench.py`:

| Shape class | `auto` from |
|---|---:|
| no `GROUP BY`, with three or more of `sum`/`min`/`max`/`avg` | 50,000,000 rows |
| fused group-by, an estimated 10,000 groups or more (with no aggregates too) | 10,000,000 rows |
| fused group-by, fewer groups, three or more of `sum`/`min`/`max`/`avg` | 50,000,000 rows |
| hash group-by, an estimated 10,000 groups or more (with no aggregates too) | 50,000,000 rows |
| no `GROUP BY` with one or two of `sum`/`min`/`max`/`avg`; fused group-by with fewer groups and at most two; hash group-by with fewer groups; a `VARCHAR` key | not rewritten in `auto` |

The benchmark's queries in each class are the rows of the results file whose `shape_class` column
names it; a query the file does not list is in `auto` because of its class, not because it was timed.
`test_measured_floors_are_in_the_benchmark_results` requires every row of a rewritten class to be
ahead at the class's floor, and no row of any other class to be rewritten.

The measurements are in `Benchmarks/results/duckdb_rewrite_2026-09-24.csv`, a quiet run: for each
query and size (1M, 10M, 50M rows), DuckDB's time and the rewrite's, wall and CPU, the GPU path taken,
and what `auto` decided. The run conditions at its start are
recorded in `Benchmarks/results/bench_conditions_2026-09-24.txt`. `test_measured_floors_are_in_the_benchmark_results` requires every class in the
table to be faster in that file at its floor. The floors were first fitted to a run taken while other
work shared the machine, kept as history in `Benchmarks/results/duckdb_rewrite_2026-09-23_provisional.csv`;
the quiet run gives the same floors. Reproduce with:

```
./duckdb-extension/build_rewrite.sh
PYTHONPATH=python python Benchmarks/duckdb_rewrite_bench.py        # 1M, 10M and 50M rows
```

Each floor is the smallest of the three sizes at which every query of the class is ahead of DuckDB,
at that size and above. In the quiet run every query `auto` rewrote is ahead of DuckDB, by 1.09x
(`sum, max, avg (BIGINT)` at 50M rows) to 5.93x (`100k INTEGER keys alone (no aggregates)` at 50M
rows), and one query of each rewritten class keeps the class out of `auto` at the size below its floor:
`sum, max, avg (BIGINT)` at 10M rows (0.87x) for the ungrouped class, `100k INTEGER keys: sum, min,
max, avg` at 1M (0.68x) for the fused group-by with many groups, `1k INTEGER keys: sum, count, min,
max, avg` at 10M (0.79x) for the fused group-by with fewer groups, and `~1M wide BIGINT keys: sum,
count` at 10M (0.83x) for the hash group-by with many groups.

The classes left to DuckDB are the ones where its own operators came out ahead in that file, or not
ahead consistently: a single `sum` (DuckDB aggregates while it scans, and the rewrite has to copy the
column out first), few groups with at most two aggregates, and `VARCHAR` keys that DuckDB keeps as
strings. In each of them at least one query is behind DuckDB at every size, 50M rows included:
`sum(INTEGER)` at 0.32x, `1k INTEGER keys: sum` at 0.65x, `~3k wide BIGINT keys: sum` at 0.79x and
`1k long VARCHAR keys: sum, count` at 0.24x. Some of their queries were ahead: `sum` and `sum, avg`
over full-range `BIGINT` values (1.87x and 3.65x at 50M rows), `10k INTEGER keys: sum, count` (1.03x
at 10M, 1.69x at 50M) and `1k short VARCHAR keys: sum, count` (1.02x at 50M). In the ungrouped case
DuckDB's time depends on the values, which the plan does not show: `avg` and `sum, avg` over
non-negative `BIGINT` values are to improve (0.6x and 0.82x at 50M rows), while the same aggregates
over full-range `BIGINT` values are ahead. Below a rewritten class's floor, or below the router's
crossover, some queries were ahead as well, for example `sum, count, min, max (BIGINT, 10% NULL)` at
10M rows (1.35x) and `100k INTEGER keys alone (no aggregates)` at 1M (1.68x); the floor holds each class
to its slowest query.

### Limits

- Built for one DuckDB release and one platform string at a time; unsigned.
- The decision is made when the plan is: a prepared statement keeps the decision made at `PREPARE`,
  and `EXPLAIN` records a decision of its own.
- A plan keeps the key statistics it was made with. Where DuckDB would have run the group-by as its
  `PERFECT_HASH_GROUP_BY` (an integer key whose statistics span at most 2^`perfect_ht_threshold` - 2
  values, 12 bits by default), a key that lands past the table those statistics sized raises DuckDB's
  own error, `Perfect hash aggregate: aggregate group N exceeded total groups M. This likely means that
  the statistics in your data source are corrupt.`, as DuckDB does with the rewrite off. That happens
  when a prepared statement runs after keys outside the planned range were inserted: the plan's
  compressed materialization narrowed the key to `key - min` in a smaller unsigned type, in which
  such keys wrap. Where DuckDB would have used its hash group-by, which does not check, the
  rewrite does not either, and both answer over the same narrowed keys
  (`test_a_prepared_statement_whose_key_outgrew_its_statistics`,
  `test_where_duckdb_does_not_check_the_statistics_neither_does_the_rewrite`).
- One known difference in that error: for an 8-bit key (`TINYINT`, `UTINYINT`, which the plan does not
  narrow) below the planned minimum, the group number. Such a key's slot is negative, `key - min + 1`;
  DuckDB prints it as 2^128 plus the slot and the rewrite as 2^64 plus the slot, so statistics of
  [10, 14] and a key of 3 read `group 340282366920938463463374607431768211450` from DuckDB and
  `group 18446744073709551610` from the rewrite. The difference is deterministic; the exception class
  (`InvalidInputException`) and the rest of the text are the same, and keys of 16 bits or more, or above
  the maximum, give the same number. A key exactly one below the minimum has slot 0, which is the slot
  DuckDB keeps for the NULL key: DuckDB returns that key's rows as the NULL group, with no error, and
  the rewrite returns them under the key itself (`test_an_8_bit_key_below_the_planned_minimum`).
- The log keeps the last 1,024 decisions of the process and is shared by every connection; a
  rewritten plan's `path`, `rows_seen`, `groups` and `gpu_ms` describe its latest run.
- The aggregate's input is held in memory: the buffers are reserved for the source's row count and
  filled as rows arrive (a streamed plan holds only its blocks in flight). Freed slabs stay mapped in
  the pool, up to 4 GB.
- The shapes in "What is rewritten" only: one group key, no floating-point `sum`/`avg`, no
  `DECIMAL`, `HUGEINT` or `BOOLEAN` values, no `min`/`max` over strings.

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
  1.8x and 7.0x, where the matrix shows 3.8x (against Acero; 16.8x against Polars) and 24.0x.
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

- Behind DuckDB's own SQL as measured (§4).
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

**Tier 3**

- §4b, "Limits".

## Files

| Path | What |
|---|---|
| `python/arrowmetal/duckdb_bridge.py` | tier 1, the whole bridge; `duckdb_connect`, `duckdb_rewrites`, `duckdb_is_rewritten` for tier 3 |
| `duckdb-extension/src/arrowmetal_extension.cpp` | tier 2, the extension |
| `duckdb-extension/src/arrowmetal_rewrite.cpp` | tier 3, the optimizer extension |
| `duckdb-extension/build.sh` | builds the tier 2 extension without cmake |
| `duckdb-extension/build_rewrite.sh` | builds the tier 3 extension against the installed DuckDB release |
| `duckdb-extension/CMakeLists.txt` | the tier 2 build, for cmake |
| `duckdb-extension/scripts/append_metadata.py` | writes DuckDB's 512-byte extension footer (`C_STRUCT` or `CPP`) |
| `python/tests/test_duckdb.py` | tiers 1 and 2, with DuckDB's own SQL as the oracle |
| `python/tests/test_duckdb_rewrite.py` | tier 3, differential against DuckDB with the rewrite off |
| `Benchmarks/duckdb_bench.py` | the tables in §5 |
| `Benchmarks/duckdb_rewrite_bench.py` | tier 3 against DuckDB's own operators at 1M, 10M and 50M rows |
| `Benchmarks/results/duckdb_rewrite_2026-09-24.csv` | its quiet run, which the `auto` floors cite |
| `Benchmarks/results/duckdb_rewrite_2026-09-23_provisional.csv` | its first run, on a shared machine, kept as history |
