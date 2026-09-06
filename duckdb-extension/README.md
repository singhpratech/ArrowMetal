# ArrowMetal as a DuckDB extension

A loadable `.duckdb_extension` that puts ArrowMetal's GPU kernels behind SQL function names.
**Read [../docs/DUCKDB.md](../docs/DUCKDB.md) first** - it covers both this and the Python bridge,
says which one you actually want, and carries the benchmark numbers.

## Build

```
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift build -c release --product ArrowMetalC
./build.sh
```

`build.sh` compiles one translation unit, links `libArrowMetalC.dylib`, and appends the 512-byte
metadata footer DuckDB checks before it will `dlopen` the file. The first run downloads `duckdb.h`
and `duckdb_extension.h` into `third_party/`; after that it needs no network. `CMakeLists.txt`
drives the same compile for cmake users.

The result is `build/arrowmetal.duckdb_extension`.

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

## Two things to know before you use it

- **It is correct and it is slower than the plain SQL.** The answers match DuckDB exactly, nulls
  included, but assembling DuckDB's 2048-row DataChunks into one contiguous column is single-threaded
  and costs more than DuckDB's parallel aggregate saves. docs/DUCKDB.md §4 has the numbers and the
  three reasons. The Python bridge does not have this problem.
- **Numeric, boolean, `DATE` and `TIMESTAMP` columns only**, and the table functions take a table or
  view *name*, not a subquery.

## Layout

| Path | What |
|---|---|
| `src/arrowmetal_extension.cpp` | the whole extension |
| `build.sh` | build without cmake |
| `CMakeLists.txt` | the same build, with cmake |
| `scripts/append_metadata.py` | writes DuckDB's extension footer |
| `third_party/` | DuckDB's C headers, fetched by `build.sh` (not committed) |
