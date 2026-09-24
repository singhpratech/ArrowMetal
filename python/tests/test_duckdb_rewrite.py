"""ArrowMetal underneath ordinary DuckDB SQL: the optimizer extension, differential against DuckDB.

duckdb-extension/src/arrowmetal_rewrite.cpp registers a DuckDB optimizer extension that replaces an
eligible LogicalAggregate with ARROWMETAL_AGGREGATE, which gathers the aggregate's input and runs it
on the GPU. The oracle is DuckDB itself: every query here runs twice on the same connection, once with
`SET arrowmetal_rewrite = 'off'` (DuckDB's own operators) and once with 'force' (the GPU path for every
supported shape, whatever its size), and the two answers must be identical - the same column types,
the same values bit for bit (AVG included), the same NULLs, the same rows. Only the order of an
unordered GROUP BY may differ, since SQL does not define it; with ORDER BY the order is compared too.

The tables are generated from hash() of the row number, so every run sees the same data, and they
reach the ends of every integer type, so sums past 2^63 and the most negative values are covered.

Run: PYTHONPATH=python python -m pytest python/tests/test_duckdb_rewrite.py -q
Skipped unless duckdb-extension/build/arrowmetal_rewrite.duckdb_extension exists; build it with
duckdb-extension/build_rewrite.sh (it needs the DuckDB release the installed duckdb module is).
"""
import csv
import json
import os
import re
import threading

import pytest

duckdb = pytest.importorskip("duckdb")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXTENSION = os.path.join(REPO, "duckdb-extension", "build", "arrowmetal_rewrite.duckdb_extension")
SOURCE = os.path.join(REPO, "duckdb-extension", "src", "arrowmetal_rewrite.cpp")
ROUTER = os.path.join(REPO, "Benchmarks", "results", "router_2026-09-17.json")
RESULTS = os.path.join(REPO, "Benchmarks", "results", "duckdb_rewrite_2026-09-23_provisional.csv")

pytestmark = pytest.mark.skipif(not os.path.exists(EXTENSION),
                                reason="build it with duckdb-extension/build_rewrite.sh")

OPERATOR = "ARROWMETAL_AGGREGATE"


def connect(threads=None, block_rows=None):
    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    con.execute(f"LOAD '{EXTENSION}'")
    if threads:
        con.execute(f"SET threads = {threads}")
    if block_rows:
        con.execute(f"SET arrowmetal_rewrite_block_rows = {block_rows}")
    return con


# Every test that takes `con` runs twice: with the default block size, where these small tables fit in
# one block, and with 2,048-row blocks, where a streamed plan (ungrouped, or a narrow integer key) is
# cut into many blocks that the GPU worker aggregates while the scan runs and Finalize merges.
@pytest.fixture(params=[None, 2048], ids=["one-block", "streamed"])
def con(request):
    connection = connect(block_rows=request.param)
    yield connection
    connection.close()


def plan(con, sql):
    return "\n".join(row[1] for row in con.sql("EXPLAIN " + sql).fetchall())


def run(con, sql, mode):
    con.execute(f"SET arrowmetal_rewrite = '{mode}'")
    rel = con.sql(sql)
    return [str(t) for t in rel.types], rel.fetchall()


def sort_key(row):
    return tuple((v is None, v if v is not None else 0) for v in row)


def last_decision(con):
    return con.sql("SELECT * FROM arrowmetal_rewrites() ORDER BY id DESC LIMIT 1").fetchone()


def check(con, sql, rewritten=True, ordered=False):
    """Runs `sql` both ways and requires identical answers; `rewritten` says whether 'force' must
    have replaced the aggregate (False: the shape is one the extension leaves alone)."""
    base_types, base = run(con, sql, "off")
    assert OPERATOR not in plan(con, sql)
    con.execute("SET arrowmetal_rewrite = 'force'")
    text = plan(con, sql)
    assert (OPERATOR in text) == rewritten, text
    types, rows = run(con, sql, "force")
    assert types == base_types, sql
    if not ordered:
        rows, base = sorted(rows, key=sort_key), sorted(base, key=sort_key)
    assert rows == base, sql
    return rows


# ---------------------------------------------------------------------------------------------------
# Generated tables
# ---------------------------------------------------------------------------------------------------

INT_RANGES = {
    "TINYINT": (-2**7, 2**7 - 1), "SMALLINT": (-2**15, 2**15 - 1),
    "INTEGER": (-2**31, 2**31 - 1), "BIGINT": (-2**63, 2**63 - 1),
    "UTINYINT": (0, 2**8 - 1), "USMALLINT": (0, 2**16 - 1),
    "UINTEGER": (0, 2**32 - 1), "UBIGINT": (0, 2**64 - 1),
}


def value_sql(type_name, seed, lo=None, hi=None, null_pct=10):
    """A deterministic column of `type_name` spread over [lo, hi] (its whole range by default)."""
    tlo, thi = INT_RANGES[type_name]
    lo = tlo if lo is None else lo
    hi = thi if hi is None else hi
    span = hi - lo + 1
    value = f"((hash(i * 7919 + {seed}) % {span}::HUGEINT) + ({lo})::HUGEINT)::{type_name}"
    if null_pct:
        value = f"CASE WHEN hash(i * 31 + {seed}) % 100 < {null_pct} THEN NULL ELSE {value} END"
    return value


def make(con, rows, key_sql, value_sqls, name="t"):
    cols = ", ".join([f"{key_sql} AS k"] + [f"{v} AS v{i}" for i, v in enumerate(value_sqls)])
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT {cols} FROM range({rows}) r(i)")


AGGS = "sum(v0), count(v0), count(*), min(v0), max(v0), avg(v0)"


# ---------------------------------------------------------------------------------------------------
# Every integer type, grouped and ungrouped, full range and nulls
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("type_name", list(INT_RANGES))
@pytest.mark.parametrize("null_pct", [0, 10])
def test_every_integer_type_matches_duckdb(con, type_name, null_pct):
    make(con, 50_000, "(hash(i) % 37)::INTEGER", [value_sql(type_name, 1, null_pct=null_pct)])
    aggs = AGGS if type_name != "UBIGINT" else "min(v0), max(v0), count(v0), count(*)"
    check(con, f"SELECT {aggs} FROM t")
    check(con, f"SELECT k, {aggs} FROM t GROUP BY k")


@pytest.mark.parametrize("type_name", ["TINYINT", "SMALLINT", "INTEGER", "BIGINT"])
def test_small_ranges_take_the_int64_state_paths(con, type_name):
    # Values that let DuckDB prove the sum fits in 64 bits (it switches to sum_no_overflow).
    make(con, 30_000, "(i % 11)::INTEGER", [value_sql(type_name, 2, -100, 100)])
    check(con, f"SELECT {AGGS} FROM t")
    check(con, f"SELECT k, {AGGS} FROM t GROUP BY k")


def test_sums_past_int64_are_exact_hugeints(con):
    # Every value near the top of BIGINT: the total is far past 2^63, and DuckDB answers a HUGEINT.
    con.execute("CREATE TABLE big AS SELECT (i % 3)::INTEGER k, (9223372036854775807 - i)::BIGINT v "
                "FROM range(20000) r(i)")
    rows = check(con, "SELECT sum(v), avg(v), min(v), max(v) FROM big")
    assert rows[0][0] > 2**63 * 19_000
    check(con, "SELECT k, sum(v), avg(v) FROM big GROUP BY k")
    con.execute("CREATE TABLE small AS SELECT (i % 3)::INTEGER k, (-9223372036854775808 + i)::BIGINT v "
                "FROM range(20000) r(i)")
    rows = check(con, "SELECT sum(v), avg(v), min(v), max(v) FROM small")
    assert rows[0][0] < -(2**63) * 19_000
    check(con, "SELECT k, sum(v), avg(v) FROM small GROUP BY k")


def test_several_value_columns_and_repeated_aggregates(con):
    make(con, 40_000, "(hash(i) % 50)::INTEGER",
         [value_sql("INTEGER", 3), value_sql("BIGINT", 4), value_sql("SMALLINT", 5, null_pct=40)])
    check(con, "SELECT k, sum(v0), sum(v1), avg(v2), min(v1), max(v0), count(v2), count(*), sum(v0), "
               "min(v0) FROM t GROUP BY k")
    check(con, "SELECT sum(v0), sum(v1), avg(v2), min(v1), max(v0), count(v2) FROM t")


def test_aggregates_over_the_key_itself(con):
    make(con, 20_000, "CASE WHEN i % 17 = 0 THEN NULL ELSE (i % 23)::INTEGER END", [value_sql("INTEGER", 6)])
    check(con, "SELECT k, count(k), sum(k), min(k), max(k), avg(k), count(*) FROM t GROUP BY k")


# ---------------------------------------------------------------------------------------------------
# Keys: dense and hashed, NULL keys, dates, timestamps, strings
# ---------------------------------------------------------------------------------------------------

KEYS = {
    "small int": "(hash(i) % 13)::INTEGER",
    "negative smallint": "((hash(i) % 300)::INTEGER - 150)::SMALLINT",
    "utinyint": "(hash(i) % 256)::UTINYINT",
    "100k int": "(hash(i) % 100000)::INTEGER",
    "wide bigint (hash path)": "((hash(i) % 500)::BIGINT * 1000000007 - 250000000000)::BIGINT",
    "ubigint": "(hash(i) % 700)::UBIGINT * 18446744073709",
    "extreme bigint": "CASE WHEN i % 3 = 0 THEN -9223372036854775808 WHEN i % 3 = 1 THEN 9223372036854775807 "
                      "ELSE 0 END::BIGINT",
    "nullable int": "CASE WHEN hash(i + 5) % 10 = 0 THEN NULL ELSE (hash(i) % 40)::INTEGER END",
    "date": "DATE '2021-03-04' + (hash(i) % 400)::INTEGER",
    "timestamp": "TIMESTAMP '2021-03-04 05:06:07' + to_microseconds((hash(i) % 5000)::BIGINT * 1000003)",
    "short varchar": "'k' || (hash(i) % 60)::VARCHAR",
    "long varchar": "'a fairly long group key number ' || (hash(i) % 300)::VARCHAR",
    "nullable varchar": "CASE WHEN i % 9 = 0 THEN NULL ELSE 'group-' || (hash(i) % 25)::VARCHAR || "
                        "'-with-a-longer-suffix' END",
    "empty and unicode varchar": "CASE hash(i) % 4 WHEN 0 THEN '' WHEN 1 THEN 'é' WHEN 2 THEN 'Ωmega key text!' "
                                 "ELSE 'x' END",
}


@pytest.mark.parametrize("key", list(KEYS))
def test_group_keys_match_duckdb(con, key):
    make(con, 60_000, KEYS[key], [value_sql("BIGINT", 7, -10**12, 10**12), value_sql("INTEGER", 8)])
    check(con, "SELECT k, sum(v0), count(v1), count(*), min(v1), max(v1), avg(v0) FROM t GROUP BY k")


def test_every_key_null(con):
    make(con, 5_000, "NULL::INTEGER", [value_sql("INTEGER", 9)])
    rows = check(con, "SELECT k, sum(v0), count(*) FROM t GROUP BY k")
    assert len(rows) == 1 and rows[0][0] is None


def test_every_value_null(con):
    make(con, 5_000, "(i % 4)::INTEGER", ["NULL::BIGINT"])
    rows = check(con, "SELECT k, sum(v0), min(v0), max(v0), avg(v0), count(v0), count(*) FROM t GROUP BY k")
    assert all(r[1] is None and r[5] == 0 for r in rows)
    check(con, "SELECT sum(v0), min(v0), max(v0), avg(v0), count(v0), count(*) FROM t")


def test_a_key_span_above_the_dense_limit(con):
    # A key range of 2^21 is past the fused group-by's slot limit; the hash group-by takes it.
    make(con, 50_000, "(hash(i) % 2097152)::INTEGER", [value_sql("INTEGER", 10)])
    check(con, "SELECT k, sum(v0), min(v0), max(v0), count(*) FROM t GROUP BY k")


# ---------------------------------------------------------------------------------------------------
# Empty inputs, WHERE clauses, ORDER BY above the aggregate, bigger tables
# ---------------------------------------------------------------------------------------------------

def test_empty_table(con):
    con.execute("CREATE TABLE e (k INTEGER, v BIGINT)")
    rows = check(con, "SELECT sum(v), count(v), count(*), min(v), max(v), avg(v) FROM e")
    assert rows == [(None, 0, 0, None, None, None)]
    assert check(con, "SELECT k, sum(v), count(*) FROM e GROUP BY k") == []


def test_a_filter_that_keeps_nothing(con):
    make(con, 10_000, "(i % 7)::INTEGER", [value_sql("INTEGER", 11, 0, 1000)])
    rows = check(con, "SELECT sum(v0), count(*), max(v0) FROM t WHERE hash(v0) = 1")
    assert rows == [(None, 0, None)]
    assert check(con, "SELECT k, sum(v0) FROM t WHERE hash(v0) = 1 GROUP BY k") == []


@pytest.mark.parametrize("where", ["v0 > 0", "v0 BETWEEN -1000 AND 50000", "k < 5 AND v1 IS NOT NULL",
                                   "k IN (1, 3, 5, 7)", "v1 % 3 = 0", "v0 IS NULL"])
def test_where_clauses(con, where):
    make(con, 80_000, "(hash(i) % 29)::INTEGER", [value_sql("INTEGER", 12), value_sql("BIGINT", 13)])
    check(con, f"SELECT k, sum(v1), count(*), min(v0), avg(v1) FROM t WHERE {where} GROUP BY k")
    check(con, f"SELECT sum(v1), count(*), min(v0), avg(v1) FROM t WHERE {where}")


def test_order_by_and_limit_above_the_aggregate(con):
    make(con, 70_000, "(hash(i) % 501)::INTEGER", [value_sql("BIGINT", 14, -10**9, 10**9)])
    check(con, "SELECT k, sum(v0) AS s, count(*) FROM t GROUP BY k ORDER BY s DESC, k", ordered=True)
    check(con, "SELECT k, sum(v0) FROM t GROUP BY k ORDER BY k LIMIT 7", ordered=True)
    check(con, "SELECT k, avg(v0) FROM t GROUP BY k HAVING count(*) > 130 ORDER BY k", ordered=True)


def test_the_aggregate_inside_a_larger_query(con):
    make(con, 30_000, "(hash(i) % 40)::INTEGER", [value_sql("INTEGER", 15, -10**6, 10**6)])
    con.execute("CREATE TABLE names AS SELECT i::INTEGER AS k, 'name ' || i AS name FROM range(40) r(i)")
    check(con, "SELECT n.name, g.s FROM (SELECT k, sum(v0) AS s FROM t GROUP BY k) g "
               "JOIN names n ON n.k = g.k")
    check(con, "SELECT k, sum(v0) FROM t GROUP BY k UNION ALL SELECT -1, sum(v0) FROM t")
    check(con, "SELECT (SELECT max(v0) FROM t) - (SELECT min(v0) FROM t)")


@pytest.mark.parametrize("block_rows", [None, 65536], ids=["one-block", "streamed"])
@pytest.mark.parametrize("threads", [1, 4, 16])
def test_thread_counts(threads, block_rows):
    con = connect(threads, block_rows)
    make(con, 500_000, "(hash(i) % 1000)::INTEGER",
         [value_sql("BIGINT", 16), value_sql("INTEGER", 17, null_pct=25)])
    check(con, "SELECT k, sum(v0), count(v1), min(v1), max(v1), avg(v0) FROM t GROUP BY k")
    check(con, "SELECT sum(v0), avg(v1), min(v0), max(v1), count(*) FROM t")
    make(con, 300_000, "'key ' || (hash(i) % 3000)::VARCHAR || ' of a table'", [value_sql("INTEGER", 18)])
    check(con, "SELECT k, sum(v0), count(*) FROM t GROUP BY k")
    con.close()


def test_a_million_rows_and_a_million_keys(con):
    make(con, 1_000_000, "(hash(i) % 900000)::INTEGER", [value_sql("BIGINT", 19)])
    check(con, "SELECT k, sum(v0), count(*), max(v0) FROM t GROUP BY k")


# ---------------------------------------------------------------------------------------------------
# Shapes the extension leaves to DuckDB
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sql, reason", [
    ("SELECT sum(DISTINCT v0) FROM t", "DISTINCT"),
    ("SELECT sum(v0) FILTER (WHERE v0 > 0) FROM t", "FILTER"),
    ("SELECT sum(f) FROM t", "DOUBLE"),
    ("SELECT avg(f) FROM t", "DOUBLE"),
    ("SELECT min(f) FROM t", "DOUBLE"),
    ("SELECT k, v0 % 2, sum(v0) FROM t GROUP BY k, v0 % 2", "GROUP BY"),
    ("SELECT sum(v0 // 2) FROM t", "expression"),
    ("SELECT k, sum(v0) FROM t GROUP BY ROLLUP (k)", "GROUPING SETS"),
    ("SELECT median(v0) FROM t", "median"),
    ("SELECT sum(d) FROM t", "DECIMAL"),
    ("SELECT count(*) FROM t", "only counts"),
    ("SELECT t.k, sum(t.v0) FROM t JOIN t u ON t.k = u.k GROUP BY t.k", "not a table scan"),
    ("SELECT min(s) FROM t", "VARCHAR"),
])
def test_unsupported_shapes_are_left_alone(con, sql, reason):
    make(con, 3_000, "(i % 5)::INTEGER", [value_sql("INTEGER", 20)])
    con.execute("ALTER TABLE t ADD COLUMN f DOUBLE DEFAULT 1.5")
    con.execute("ALTER TABLE t ADD COLUMN d DECIMAL(10, 2) DEFAULT 2.25")
    con.execute("ALTER TABLE t ADD COLUMN s VARCHAR DEFAULT 'x'")
    check(con, sql, rewritten=False)
    kept = [r for r in con.sql("SELECT decision, reason FROM arrowmetal_rewrites()").fetchall()
            if r[0] == "kept"]
    assert any(reason in r[1] for r in kept), kept


def test_a_parquet_scan(con, tmp_path):
    # read_parquet reports its row count from the file's metadata, so it takes the same path as a table.
    make(con, 70_000, "(hash(i) % 300)::INTEGER", [value_sql("BIGINT", 32), value_sql("INTEGER", 33)])
    path = str(tmp_path / "t.parquet")
    con.execute(f"COPY t TO '{path}' (FORMAT parquet)")
    check(con, f"SELECT k, sum(v0), count(v1), min(v1), avg(v0) FROM read_parquet('{path}') GROUP BY k")
    check(con, f"SELECT sum(v0), max(v1) FROM read_parquet('{path}') WHERE v1 > 0")
    assert last_decision(con)[3].startswith("read_parquet -> ")


def test_a_registered_arrow_table_is_left_to_duckdb(con):
    # DuckDB's arrow_scan does not report a row count to the planner, so the size gate cannot be applied;
    # the aggregate stays DuckDB's (the Python bridge, tier 1, is the path for Arrow data).
    pa = pytest.importorskip("pyarrow")
    con.register("arrow_t", pa.table({"k": pa.array([1, 2, 3] * 1000, pa.int32()),
                                      "v": pa.array(range(3000), pa.int64())}))
    check(con, "SELECT k, sum(v) FROM arrow_t GROUP BY k", rewritten=False)
    assert last_decision(con)[2] == "the source arrow_scan does not report its size"


# ---------------------------------------------------------------------------------------------------
# The controls: auto / off / force, and seeing what happened
# ---------------------------------------------------------------------------------------------------

def test_auto_leaves_a_table_below_the_crossover_alone(con):
    make(con, 10_000, "(i % 10)::INTEGER", [value_sql("BIGINT", 21)])
    con.execute("SET arrowmetal_rewrite = 'auto'")
    sql = "SELECT k, sum(v0), min(v0), max(v0) FROM t GROUP BY k"
    assert OPERATOR not in plan(con, sql)
    decision = last_decision(con)
    assert decision[1] == "kept" and decision[2] == "below the crossover: fused group-by, fewer groups, three or more aggregates"
    # The router's 10M crossover for a 1,000-group sum, and the 50M measured floor for this class.
    assert decision[4] == 10_000 and decision[5] == 50_000_000


def test_auto_never_rewrites_a_class_not_measured_faster(con):
    make(con, 10_000, "(i % 10)::INTEGER", [value_sql("BIGINT", 21)])
    make(con, 10_000, "'a longer VARCHAR group key ' || (i % 10)::VARCHAR", [value_sql("BIGINT", 21)], name="s")
    con.execute("SET arrowmetal_rewrite = 'auto'")
    for sql in ["SELECT k, sum(v0) FROM t GROUP BY k", "SELECT max(v0) FROM t",
                "SELECT k, min(v0), max(v0), avg(v0) FROM s GROUP BY k"]:
        assert OPERATOR not in plan(con, sql)
        decision = last_decision(con)
        assert decision[1] == "kept" and decision[2].startswith("not measured faster than DuckDB"), decision
        assert decision[5] is None


def test_auto_rewrites_at_scale(con):
    """At 10M rows 'auto' takes the many-groups and the ungrouped classes, and still matches DuckDB."""
    con.execute("CREATE TABLE big AS SELECT (hash(i) % 200000)::INTEGER k, "
                "(hash(i * 3) % 1000000000)::BIGINT v, (hash(i * 5)::HUGEINT - 9223372036854775808)::BIGINT f "
                "FROM range(10000000) r(i)")
    con.execute("SET arrowmetal_rewrite = 'auto'")
    for sql in ["SELECT k, sum(v), count(*) FROM big GROUP BY k", "SELECT sum(f), avg(f) FROM big"]:
        assert OPERATOR in plan(con, sql), sql
        assert last_decision(con)[2].startswith("at or above the threshold")
        got = sorted(con.sql(sql).fetchall(), key=sort_key)
        con.execute("SET arrowmetal_rewrite = 'off'")
        assert got == sorted(con.sql(sql).fetchall(), key=sort_key)
        con.execute("SET arrowmetal_rewrite = 'auto'")


def test_auto_is_the_default_and_off_turns_it_off(con):
    assert con.sql("SELECT current_setting('arrowmetal_rewrite')").fetchone()[0] == "auto"
    make(con, 1_000, "(i % 10)::INTEGER", [value_sql("BIGINT", 22)])
    con.execute("SET arrowmetal_rewrite = 'force'")
    assert OPERATOR in plan(con, "SELECT k, sum(v0) FROM t GROUP BY k")
    con.execute("SET arrowmetal_rewrite = 'off'")
    before = last_decision(con)[0]
    assert OPERATOR not in plan(con, "SELECT k, sum(v0) FROM t GROUP BY k")
    assert last_decision(con)[0] == before   # 'off' does not even look


def test_the_decision_log_records_the_run(con):
    make(con, 20_000, "(i % 10)::INTEGER", [value_sql("BIGINT", 23)])
    con.execute("SET arrowmetal_rewrite = 'force'")
    con.sql("SELECT k, sum(v0) FROM t GROUP BY k").fetchall()
    row = last_decision(con)
    names = [d[0] for d in con.sql("SELECT * FROM arrowmetal_rewrites() LIMIT 0").description]
    assert names == ["id", "decision", "reason", "shape", "input_rows", "threshold_rows", "path",
                     "rows_seen", "groups", "gpu_ms"]
    assert row[1] == "rewritten" and row[2] == "forced"
    assert row[6].startswith("fused dense group-by, streamed in") and row[7] == 20_000 and row[8] == 10
    assert row[9] > 0
    blocks, handed = map(int, re.search(r"streamed in (\d+) blocks?, (\d+) handed", row[6]).groups())
    if con.sql("SELECT current_setting('arrowmetal_rewrite_block_rows')").fetchone()[0] == 2048:
        # 20,000 rows are nine full 2,048-row blocks, each handed over as it filled, and a partial tenth.
        assert (blocks, handed) == (10, 9)
    else:
        assert (blocks, handed) == (1, 0)
    assert "seq_scan" in row[3] and "GROUP BY" in row[3]


MEASURED_CLASSES = {
    "UNGROUPED": "ungrouped",
    "DENSE_MANY_GROUPS": "fused group-by, an estimated 10k or more groups",
    "DENSE_FEW_GROUPS": "fused group-by, fewer groups, three or more aggregates",
    "HASH_MANY_GROUPS": "hash group-by, an estimated 10k or more groups",
}


def test_measured_floors_are_in_the_benchmark_results():
    """Every class 'auto' rewrites was measured faster than DuckDB's own operators at its floor, in the
    results file the floors cite (Benchmarks/duckdb_rewrite_bench.py writes it)."""
    source = open(SOURCE).read()
    floors = dict(re.findall(r"static constexpr int64_t (\w+) = (\d+);", source))
    with open(RESULTS) as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    for constant, shape_class in MEASURED_CLASSES.items():
        floor = int(floors[constant])
        at_floor = [r for r in rows if r["shape_class"] == shape_class and int(r["rows"]) == floor]
        assert at_floor, (shape_class, floor)
        assert all(float(r["speedup"]) > 1 for r in at_floor), at_floor
    # The classes with no floor are the rest; none of them is rewritten in auto anywhere in the file.
    for r in rows:
        if r["auto"] == "rewritten":
            assert r["shape_class"] in MEASURED_CLASSES.values(), r


def test_crossovers_match_the_router_sweep():
    """The auto gate's constants are the router's crossovers (Benchmarks/results/router_2026-09-17.json)."""
    source = open(SOURCE).read()
    constants = dict(re.findall(r"static constexpr int64_t (\w+) = (\d+);", source))
    crossover = json.load(open(ROUTER))["vs_fastest_library"]
    assert int(constants["SUM"]) == crossover["reductions: sum(int64, 10% nulls)"]
    assert int(constants["MIN"]) == crossover["reductions: min(int64, 10% nulls)"]
    assert int(constants["MAX"]) == crossover["reductions: max(int64, 10% nulls)"]
    assert int(constants["MEAN"]) == crossover["reductions: mean(int64, 10% nulls)"]
    for op in ["sum", "count", "min", "max", "mean"]:
        assert int(constants["GROUP_1K"]) == crossover[f"group-by: {op} by int32 key (1000 groups)"]
        assert int(constants["GROUP_100K"]) == crossover[f"group-by: {op} by int32 key (100000 groups)"]
    assert int(constants["GROUP_UTF8"]) == crossover["group-by: sum by utf8 key (1000 distinct)"]
    assert int(constants["GROUP_UTF8"]) == crossover["group-by: sum by utf8 key (100000 distinct)"]


# ---------------------------------------------------------------------------------------------------
# Plans that outlive their statistics, transactions, concurrency
# ---------------------------------------------------------------------------------------------------

def test_a_prepared_statement_after_the_table_grew(con):
    # A single-block plan (a BIGINT MIN under GROUP BY needs the hash group-by's whole-input array) sizes
    # its block from the table at planning time; rows added afterwards go through the overflow path. A
    # streamed plan (the ungrouped one) just takes more blocks. Every row must be counted either way.
    make(con, 20_000, "(i % 10)::INTEGER", [value_sql("BIGINT", 24)])
    con.execute("SET arrowmetal_rewrite = 'force'")
    start = con.sql("SELECT coalesce(max(id), 0) FROM arrowmetal_rewrites()").fetchone()[0]
    con.execute("PREPARE p AS SELECT k, sum(v0), count(*), min(v0) FROM t GROUP BY k")
    con.execute("PREPARE q AS SELECT sum(v0), count(*), max(v0) FROM t")
    first = sorted(con.execute("EXECUTE p").fetchall())
    con.execute(f"INSERT INTO t SELECT (i % 13)::INTEGER, {value_sql('BIGINT', 25)} FROM range(150000) r(i)")
    grown_p = sorted(con.execute("EXECUTE p").fetchall())
    grown_q = con.execute("EXECUTE q").fetchall()
    # One decision per plan; its run columns describe the plan's latest execution.
    # (The log is the process's, shared by every connection, hence the id filter.)
    runs = con.sql(f"SELECT path, rows_seen FROM arrowmetal_rewrites() WHERE id > {start} ORDER BY id").fetchall()
    assert len(runs) == 2
    assert re.fullmatch(r"hash group-by \(\d+ rows past the reserved buffers\)", runs[0][0]), runs
    assert "streamed in" in runs[1][0], runs
    assert runs[0][1] == runs[1][1] == 170_000
    con.execute("SET arrowmetal_rewrite = 'off'")
    assert grown_p == sorted(con.sql("SELECT k, sum(v0), count(*), min(v0) FROM t GROUP BY k").fetchall())
    assert grown_q == con.sql("SELECT sum(v0), count(*), max(v0) FROM t").fetchall()
    assert grown_p != first


def test_a_streamed_plan_past_its_block_directory():
    # A streamed plan keeps room for 1,024 blocks beyond the planned input; with 2,048-row blocks a
    # 20,000-row table planned once can take about 2.1M more rows before the rest is kept aside and
    # run through fresh blocks in Finalize.
    con = connect(block_rows=2048)
    make(con, 20_000, "(i % 10)::INTEGER", [value_sql("BIGINT", 30)])
    con.execute("SET arrowmetal_rewrite = 'force'")
    start = con.sql("SELECT coalesce(max(id), 0) FROM arrowmetal_rewrites()").fetchone()[0]
    con.execute("PREPARE q AS SELECT k, sum(v0), count(v0), avg(v0), count(*) FROM t GROUP BY k")
    con.execute(f"INSERT INTO t SELECT (i % 10)::INTEGER, {value_sql('BIGINT', 31)} FROM range(2300000) r(i)")
    got = sorted(con.execute("EXECUTE q").fetchall())
    path, seen = con.sql(f"SELECT path, rows_seen FROM arrowmetal_rewrites() WHERE id > {start}").fetchone()
    assert "rows past the reserved buffers" in path and "streamed in" in path and seen == 2_320_000, path
    con.execute("SET arrowmetal_rewrite = 'off'")
    assert got == sorted(con.sql("SELECT k, sum(v0), count(v0), avg(v0), count(*) FROM t GROUP BY k").fetchall())
    con.close()


def test_string_keys_past_the_reserved_bytes(con):
    # 400-byte keys, far above the 64 bytes a row the string buffer reserves: the overflow path.
    make(con, 30_000, "repeat('x', 390) || (hash(i) % 70)::VARCHAR", [value_sql("INTEGER", 26)])
    check(con, "SELECT k, sum(v0), count(*) FROM t GROUP BY k")
    assert "rows past the reserved buffers" in last_decision(con)[6]


def test_uncommitted_rows_in_a_transaction(con):
    make(con, 10_000, "(i % 10)::INTEGER", [value_sql("INTEGER", 27)])
    con.execute("BEGIN")
    con.execute(f"INSERT INTO t SELECT (i % 10)::INTEGER, {value_sql('INTEGER', 28)} FROM range(40000) r(i)")
    con.execute("DELETE FROM t WHERE k = 3")
    check(con, "SELECT k, sum(v0), count(*) FROM t GROUP BY k")
    check(con, "SELECT sum(v0), count(*), min(v0) FROM t")
    con.execute("ROLLBACK")
    check(con, "SELECT k, sum(v0), count(*) FROM t GROUP BY k")


def test_concurrent_queries_on_two_connections():
    db = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    db.execute(f"LOAD '{EXTENSION}'")
    make(db, 300_000, "(hash(i) % 700)::INTEGER", [value_sql("BIGINT", 29)])
    db.execute("SET arrowmetal_rewrite = 'off'")
    expected = sorted(db.sql("SELECT k, sum(v0), avg(v0) FROM t GROUP BY k").fetchall())
    errors, results = [], []

    def worker():
        try:
            c = db.cursor()
            c.execute("SET arrowmetal_rewrite = 'force'")
            for _ in range(5):
                results.append(sorted(c.sql("SELECT k, sum(v0), avg(v0) FROM t GROUP BY k").fetchall()))
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(results) == 20 and all(r == expected for r in results)
    db.close()


# ---------------------------------------------------------------------------------------------------
# The Python side: am.duckdb_connect and friends
# ---------------------------------------------------------------------------------------------------

def test_python_duckdb_connect():
    am = pytest.importorskip("arrowmetal")
    con = am.duckdb_connect(rewrite="force")
    assert con.sql("SELECT current_setting('arrowmetal_rewrite')").fetchone()[0] == "force"
    con.execute("CREATE TABLE t AS SELECT (i % 7)::INTEGER k, i::BIGINT v FROM range(10000) r(i)")
    sql = "SELECT k, sum(v), count(*) FROM t GROUP BY k"
    assert am.duckdb_is_rewritten(con, sql)
    got = sorted(con.sql(sql).fetchall())
    log = am.duckdb_rewrites(con)
    assert log.column_names[:3] == ["id", "decision", "reason"]
    assert log.column("decision").to_pylist()[-1] == "rewritten"
    con.execute("SET arrowmetal_rewrite = 'off'")
    assert not am.duckdb_is_rewritten(con, sql)
    assert got == sorted(con.sql(sql).fetchall())
    con.close()
    assert am.duckdb_connect().sql("SELECT current_setting('arrowmetal_rewrite')").fetchone()[0] == "auto"


def test_python_duckdb_connect_refuses_what_it_cannot_do():
    am = pytest.importorskip("arrowmetal")
    with pytest.raises(am.ArrowMetalError, match="rewrite must be one of"):
        am.duckdb_connect(rewrite="sometimes")
    with pytest.raises(am.ArrowMetalError, match="build it with duckdb-extension/build_rewrite.sh"):
        am.duckdb_connect(extension="/nonexistent/arrowmetal_rewrite.duckdb_extension")
