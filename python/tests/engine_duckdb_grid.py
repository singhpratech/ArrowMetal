"""The DuckDB half of the engine conformance grid (see engine_conformance.py).

Each case is one query over a generated table, run on one connection with the optimizer extension
loaded, once with `SET arrowmetal_rewrite = 'off'` (DuckDB's own operators, the oracle) and once with
`'force'` (every shape the extension supports goes to the GPU, whatever its size). The answers must
have the same column types and the same values bit for bit, `avg` included; the rows of a `GROUP BY`
are compared as a multiset, since SQL gives them no order. `EXPLAIN` under `'force'` says whether the
plan has an `ARROWMETAL_AGGREGATE`; a query whose plan does not is still compared and is counted as
not taken.

The grid: the value column types the extension rewrites (the eight integer types, `DATE`,
`TIMESTAMP`) and six it leaves to DuckDB (`DOUBLE`, `DECIMAL(18,3)`, `HUGEINT`, `BOOLEAN`,
`TIMESTAMP_NS`, `TIMESTAMPTZ`), x `sum`, `avg`, `min`, `max`, `count(v)`, `count(*)`, all of the
valid ones in one query, and no aggregate at all under a key, x no key or one key of each kind
(`INTEGER` with NULLs, a wide `BIGINT`, a negative `SMALLINT`, `UTINYINT`, `DATE`, `TIMESTAMP`,
`VARCHAR` with NULLs), x no filter, `WHERE v IS NOT NULL` or a filter on another column, x the null
patterns and sizes of engine_conformance.py. A combination DuckDB itself rejects (`sum` of a `DATE`)
is not a case.
"""
import os
import sys

import numpy as np
import pyarrow as pa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_conformance as ec                                      # noqa: E402
from engine_conformance import PASS, NOT_TAKEN                         # noqa: E402

import duckdb                                                        # noqa: E402

ENGINE = "duckdb"
OPERATOR = "ARROWMETAL_AGGREGATE"

# value type name -> (test_differential generator, DuckDB type the column is cast to)
VALUE_TYPES = {
    "TINYINT": ("int8", "TINYINT"), "SMALLINT": ("int16", "SMALLINT"),
    "INTEGER": ("int32", "INTEGER"), "BIGINT": ("int64", "BIGINT"),
    "UTINYINT": ("uint8", "UTINYINT"), "USMALLINT": ("uint16", "USMALLINT"),
    "UINTEGER": ("uint32", "UINTEGER"), "UBIGINT": ("uint64", "UBIGINT"),
    "DATE": ("date32", "DATE"), "TIMESTAMP": ("ts_us", "TIMESTAMP"),
    # left to DuckDB by the extension; in the grid to check that they are
    "DOUBLE": ("float64", "DOUBLE"), "DECIMAL": ("decimal128_18_6", "DECIMAL(18,3)"),
    "HUGEINT": ("int64", "HUGEINT"), "BOOLEAN": ("bool", "BOOLEAN"),
    "TIMESTAMP_NS": ("ts_ns", "TIMESTAMP_NS"), "TIMESTAMPTZ": ("ts_us_tz", "TIMESTAMPTZ"),
}
REWRITTEN_TYPES = ["TINYINT", "SMALLINT", "INTEGER", "BIGINT", "UTINYINT", "USMALLINT", "UINTEGER",
                   "UBIGINT", "DATE", "TIMESTAMP"]
INTEGER_TYPES = REWRITTEN_TYPES[:8]

KEYS = ["none", "int", "bigint_wide", "smallint_neg", "utinyint", "date", "timestamp", "varchar"]
FILTERS = {"none": "", "v_not_null": "WHERE v IS NOT NULL", "other_column": "WHERE id % 3 <> 0"}
AGGS = ["sum", "avg", "min", "max", "count", "count_star", "all", "keys_only"]

_VARCHAR_POOL = ["", "a", "apple", "é", "Ωmega key text!", "a fairly long group key number 17",
                 "x" * 40, "0", "null", "日本語"]


def _key_array(key, n, seed=21):
    rng = np.random.default_rng([seed, n, KEYS.index(key)])
    mask = (rng.random(n) < 0.05) if n else None
    if key == "int":
        return pa.array(rng.integers(0, 13, n).astype(np.int32), mask=mask), "INTEGER"
    if key == "bigint_wide":
        vals = rng.integers(0, 500, n) * 1_000_000_007 - 250_000_000_000
        return pa.array(vals.astype(np.int64)), "BIGINT"
    if key == "smallint_neg":
        return pa.array((rng.integers(0, 300, n) - 150).astype(np.int16)), "SMALLINT"
    if key == "utinyint":
        return pa.array(rng.integers(0, 256, n).astype(np.uint8)), "UTINYINT"
    if key == "date":
        return pa.array((18_000 + rng.integers(0, 400, n)).astype(np.int32), type=pa.date32()), "DATE"
    if key == "timestamp":
        vals = 1_600_000_000_000_000 + rng.integers(0, 5000, n) * 1_000_003
        return pa.array(vals.astype(np.int64), type=pa.timestamp("us")), "TIMESTAMP"
    if key == "varchar":
        idx = rng.integers(0, len(_VARCHAR_POOL), n)
        return pa.array([None if (mask is not None and mask[i]) else _VARCHAR_POOL[j]
                         for i, j in enumerate(idx)], type=pa.string()), "VARCHAR"
    raise AssertionError(key)


def data_shapes(quick=False):
    """engine_conformance's shapes, plus the 100,000-row sparse table again with 2,048-row blocks
    (`arrowmetal_rewrite_block_rows`), so the streamed plans are in the grid."""
    out = ec.data_shapes(quick)
    if not quick:
        out.append(ec.DataShape(100_000, "sparse", "streamed"))
    return out


class Tables:
    """One connection, and the table for the current (value type, key, data shape)."""

    def __init__(self, extension=None):
        from arrowmetal import duckdb_bridge
        self.path = extension or duckdb_bridge.rewrite_extension_path()
        self.con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
        self.con.execute(f"LOAD '{self.path}'")
        self.current = None
        self.block_rows = None

    def load(self, vtype, key, ds):
        if self.current == (vtype, key, ds):
            return
        n = ds.size
        gen, sqltype = VALUE_TYPES[vtype]
        flavor = "random" if ds.flavor == "streamed" else ds.flavor
        src = ec._source_array(gen, ec.DataShape(n, ds.nulls, flavor), seed=31)
        cols = {"v": src, "id": pa.array(np.arange(n, dtype=np.int64))}
        select = [f"v::{sqltype} AS v", "id"]
        if key != "none":
            karr, ktype = _key_array(key, n)
            cols["k"] = karr
            select.insert(0, f"k::{ktype} AS k")
        arrow = pa.table(cols)
        self.con.register("am_src", arrow)
        self.con.execute(f"CREATE OR REPLACE TABLE t AS SELECT {', '.join(select)} FROM am_src")
        self.con.unregister("am_src")
        want_block = 2048 if ds.flavor == "streamed" else None
        if want_block != self.block_rows:
            if want_block:
                self.con.execute(f"SET arrowmetal_rewrite_block_rows = {want_block}")
            else:
                self.con.execute("RESET arrowmetal_rewrite_block_rows")
            self.block_rows = want_block
        self.current = (vtype, key, ds)

    def close(self):
        self.con.close()


def _agg_sql(agg, vtype):
    if agg == "count_star":
        return ["count(*)"]
    if agg == "count":
        return ["count(v)"]
    if agg == "all":
        out = ["count(v)", "count(*)", "min(v)", "max(v)"]
        if vtype not in ("DATE", "TIMESTAMP", "TIMESTAMP_NS", "TIMESTAMPTZ", "BOOLEAN"):
            out += ["sum(v)", "avg(v)"]
        return out
    if agg == "keys_only":
        return []
    return [f"{agg}(v)"]


def query(vtype, key, filt, agg):
    """The SQL of one case, or None when the combination is not one (no key for keys_only)."""
    aggs = _agg_sql(agg, vtype)
    where = FILTERS[filt]
    if key == "none":
        if not aggs:
            return None
        return f"SELECT {', '.join(aggs)} FROM t {where}".strip()
    return f"SELECT {', '.join(['k'] + aggs)} FROM t {where} GROUP BY k".strip()


def cases(quick=False, vtypes=None, keys=None, aggs=None):
    for ds in data_shapes(quick):
        for vtype in VALUE_TYPES:
            if vtypes and vtype not in vtypes:
                continue
            if ds.flavor == "special" and vtype not in INTEGER_TYPES + ["DOUBLE"]:
                continue
            for key in KEYS:
                if keys and key not in keys:
                    continue
                for filt in FILTERS:
                    # The filters matter below one key kind as much as another: run them under no
                    # key, an INTEGER key and a VARCHAR key only.
                    if filt != "none" and key not in ("none", "int", "varchar"):
                        continue
                    for agg in AGGS:
                        if aggs and agg not in aggs:
                            continue
                        sql = query(vtype, key, filt, agg)
                        if sql is None:
                            continue
                        shape = f"{agg}|{'no key' if key == 'none' else 'key ' + key}|{filt}"
                        yield {"engine": ENGINE, "shape": shape, "family": agg, "dtype": vtype,
                               "ds": ds, "key": key, "filter": filt, "agg": agg, "sql": sql}


def _sort_key(row):
    return tuple((v is None, repr(type(v)), v if v is not None else 0) for v in row)


def _run(con, sql, mode):
    con.execute(f"SET arrowmetal_rewrite = '{mode}'")
    rel = con.sql(sql)
    return [str(t) for t in rel.types], rel.fetchall()


def _cell_equal(a, b):
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:
            return True
        return a == b and np.copysign(1.0, a) == np.copysign(1.0, b)
    return type(a) is type(b) and a == b


def _rows_equal(r1, r2):
    return len(r1) == len(r2) and all(len(a) == len(b) and all(_cell_equal(x, y) for x, y in zip(a, b))
                                      for a, b in zip(r1, r2))


def run_case(case, tables):
    tables.load(case["dtype"], case["key"], case["ds"])
    con, sql = tables.con, case["sql"]
    try:
        base_types, base = _run(con, sql, "off")
    except duckdb.Error as exc:
        return "invalid", f"DuckDB rejects the query: {str(exc).splitlines()[0]}", {}
    con.execute("SET arrowmetal_rewrite = 'force'")
    last = con.sql("SELECT coalesce(max(id), 0) FROM arrowmetal_rewrites()").fetchone()[0]
    plan = "\n".join(row[1] for row in con.sql("EXPLAIN " + sql).fetchall())
    taken = OPERATOR in plan
    reason = None
    if not taken:
        kept = con.sql(f"SELECT reason FROM arrowmetal_rewrites() WHERE id > {last} AND decision = 'kept' "
                       "ORDER BY id").fetchall()
        reason = kept[0][0] if kept else "DuckDB's optimised plan has no aggregate left for the extension"
    try:
        types, rows = _run(con, sql, "force")
    except duckdb.Error as exc:
        return "mismatch", f"with the rewrite: {type(exc).__name__}: {str(exc).splitlines()[0][:300]}", \
            {"kind": "error", "taken": taken}
    extra = {"taken": taken}
    if types != base_types:
        extra["kind"] = "schema"
        return "mismatch", f"types: rewrite {types} vs DuckDB {base_types}", extra
    if case["key"] != "none":
        rows, base = sorted(rows, key=_sort_key), sorted(base, key=_sort_key)
    if not _rows_equal(rows, base):
        extra["kind"] = "values"
        for i, (a, b) in enumerate(zip(rows, base)):
            if not (len(a) == len(b) and all(_cell_equal(x, y) for x, y in zip(a, b))):
                return "mismatch", f"row {i}: rewrite {a!r} vs DuckDB {b!r}", extra
        return "mismatch", f"row counts: rewrite {len(rows)} vs DuckDB {len(base)}", extra
    if not taken:
        return NOT_TAKEN, reason, extra
    return PASS, "", extra


# ==================================================================================================
# the report's hooks

HOST_DESCRIPTION = (f"SET arrowmetal_rewrite = 'force' against 'off', duckdb {duckdb.__version__}")


def reproducer(case):
    """A line of Python that rebuilds the case's table and reruns it from python/tests."""
    return (f"import engine_duckdb_grid as g, engine_conformance as ec; t = g.Tables(); "
            f"c = [c for c in g.cases() if c['sql'] == {case['sql']!r} and c['dtype'] == {case['dtype']!r} "
            f"and c['ds'] == ec.DataShape{tuple(case['ds'])!r}][0]; print(g.run_case(c, t))")


DIVERGENCES = []
