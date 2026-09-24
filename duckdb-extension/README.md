# ArrowMetal as a DuckDB extension

A loadable `.duckdb_extension` that puts ArrowMetal's GPU kernels behind SQL function names, and a
second one (the rewrite extension, below) that runs eligible aggregates of unchanged SQL on the GPU.
**Read [../docs/DUCKDB.md](../docs/DUCKDB.md) first** - it covers both of these and the Python bridge,
says which one you actually want, and carries the benchmark numbers.

## Build

```
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift build -c release --product ArrowMetalC
./build.sh
```

`build.sh` compiles one translation unit, links `libArrowMetalC.dylib`, and appends (through
`scripts/append_metadata.py`) the 512-byte metadata footer DuckDB checks before it will `dlopen` the
file. The first run downloads `duckdb.h` and `duckdb_extension.h` into `third_party/` from the DuckDB
source tree on GitHub; after that it needs no network. `CMakeLists.txt` drives the same compile for
cmake users, but it does not fetch the headers — run `build.sh` once first.

The result is `build/arrowmetal.duckdb_extension`. Neither `build/` nor `third_party/` is committed, so
until you have run `build.sh` the 11 `@extension` tests in `python/tests/test_duckdb.py` skip.

## Load

Unsigned extensions must be allowed when the connection is created:

```python
con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
con.execute("LOAD 'duckdb-extension/build/arrowmetal.duckdb_extension'")
```

From the CLI, `duckdb -unsigned`.

## Functions

```sql
SELECT arrowmetal_version();
SELECT arrowmetal_device();
SELECT * FROM arrowmetal_agg('t', 'v');                 -- sum, count, min, max, mean
SELECT * FROM arrowmetal_group_by('t', 'k', 'v');       -- key, count, sum, min, max
SELECT * FROM arrowmetal_top_k('t', 'v', 100);
SELECT * FROM arrowmetal_sort('t', 'v');
SELECT * FROM arrowmetal_query('t', '(query (aggregate (sum "total" (col "v"))))');
```

`arrowmetal_query` takes plans that end in an `(aggregate ...)` and returns one row per aggregate; a
plan that would produce columns is refused, with a message pointing at `arrowmetal_group_by` or the
Python bridge.

## Three things to know before you use it

- **It is correct and it is slower than the plain SQL.** The 11 `@extension` tests in
  `python/tests/test_duckdb.py` compare its answers with DuckDB's, nulls included, and they match; but
  assembling DuckDB's 2048-row DataChunks into one contiguous column is single-threaded and costs more
  than DuckDB's parallel aggregate saves. docs/DUCKDB.md §4 has the numbers and the three reasons. The
  Python bridge does not have this problem.
- **The eight integer widths, `FLOAT`, `DOUBLE`, `DATE` and `TIMESTAMP` only.** `BOOLEAN`, `VARCHAR`,
  `DECIMAL`, `TIME`, `TIMESTAMPTZ`, `HUGEINT` and the nested types are refused at bind time with a
  message pointing at the Python bridge.
- **The table functions take a table or view *name*, not a subquery**, and the extension opens its own
  connection, so a relation registered from Python is invisible to it — `CREATE VIEW` over it first.

## The rewrite extension: unchanged SQL on the GPU

`src/arrowmetal_rewrite.cpp` is a second, separate extension: a DuckDB **optimizer extension** that
replaces an eligible aggregate in the plan of an ordinary query with `ARROWMETAL_AGGREGATE`, which runs
it on the GPU and answers exactly what DuckDB would. docs/DUCKDB.md §4b has what it rewrites, the
`auto` gate and its measurements, and the limits.

```
PYTHON=/path/to/python/with/duckdb ./build_rewrite.sh
```

```python
con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
con.execute("LOAD 'duckdb-extension/build/arrowmetal_rewrite.duckdb_extension'")
con.sql("EXPLAIN SELECT k, sum(v) FROM t GROUP BY k").show()   # ARROWMETAL_AGGREGATE when rewritten
con.sql("SELECT * FROM arrowmetal_rewrites()").show()          # what it decided, and why
con.execute("SET arrowmetal_rewrite = 'off'")                  # or 'auto' (the default), 'force'
```

It needs DuckDB's C++ API, which only C++ extensions get: the C API above has no optimizer hook. A
C++ extension is tied to one DuckDB release, so `build_rewrite.sh` reads the release from the target
Python's `duckdb`, shallow-clones that tag into `build/duckdb-<version>/` for its headers (nothing of
DuckDB is compiled), and checks that the module exports every DuckDB symbol the extension uses. Its
tests are `python/tests/test_duckdb_rewrite.py`, which skip until the extension is built.

## Layout

| Path | What |
|---|---|
| `src/arrowmetal_extension.cpp` | the table-function extension (C API) |
| `src/arrowmetal_rewrite.cpp` | the rewrite extension (C++ API, one DuckDB release) |
| `build.sh` | build the table-function extension without cmake |
| `build_rewrite.sh` | build the rewrite extension |
| `CMakeLists.txt` | the table-function build, with cmake |
| `scripts/append_metadata.py` | writes DuckDB's extension footer |
| `third_party/` | DuckDB's C headers, fetched by `build.sh` (not committed) |
| `build/duckdb-<version>/` | DuckDB's source at the release tag, fetched by `build_rewrite.sh` (not committed) |
