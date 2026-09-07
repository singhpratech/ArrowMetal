"""The DuckDB bridge and the DuckDB extension, checked against DuckDB's own SQL.

The oracle throughout is DuckDB itself: every GPU answer is compared to the answer the same
connection gives for the equivalent SQL, on the same data, in the same process. Nothing here trusts
a hand-written expected value.

Run: PYTHONPATH=python python -m pytest python/tests/test_duckdb.py -q

The extension tests are skipped unless duckdb-extension/build/arrowmetal.duckdb_extension exists;
build it with duckdb-extension/build.sh.
"""
import os

import pyarrow as pa
import pytest

import arrowmetal as am

duckdb = pytest.importorskip("duckdb")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXTENSION = os.path.join(REPO, "duckdb-extension", "build", "arrowmetal.duckdb_extension")


@pytest.fixture
def con():
    connection = duckdb.connect()
    yield connection
    connection.close()


def make_table(con, rows=100_003, name="t"):
    """A table with a key, a nullable bigint, a double and a string, all deterministic."""
    con.execute(f"""
        create or replace table {name} as
        select (i % 97)::INTEGER          as k,
               case when i % 13 = 0 then null else i::BIGINT end as v,
               (i * 0.5 - 1000)::DOUBLE   as f,
               ('row' || (i % 1000))      as s
        from range({rows}) r(i)
    """)
    return name


# ---------------------------------------------------------------------------------------------------
# The types DuckDB emits
# ---------------------------------------------------------------------------------------------------

# One row per Arrow type DuckDB can produce, spanning the whole width of what a real result set
# carries: every integer width and signedness, both floats, strings, blobs, all three date/time
# families, both decimal widths, and the nested types.
TYPE_SQL = """
select 1::TINYINT a, 2::SMALLINT b, 3::INTEGER c, 4::BIGINT d,
       5::UTINYINT e, 6::USMALLINT f, 7::UINTEGER g, 8::UBIGINT h,
       1.5::FLOAT i, 2.5::DOUBLE j, true k, 'hi' l, 'ab'::BLOB m,
       DATE '2020-01-02' n, TIMESTAMP '2020-01-02 03:04:05' o, TIME '03:04:05' p,
       123.45::DECIMAL(10,2) q, 12345678901234567890.12::DECIMAL(38,2) r,
       [1,2,3] s, {'x': 1, 'y': 'z'} t, INTERVAL 1 DAY u, 9::HUGEINT v,
       TIMESTAMPTZ '2020-01-02 03:04:05+00' w, MAP{'a': 1} x,
       '8ac1b2c3-0000-4000-8000-000000000001'::UUID::VARCHAR y
"""


def test_every_duckdb_type_round_trips(con):
    """from_duckdb lifts every type DuckDB emits onto the GPU, and to_arrow gives it back intact."""
    table = con.sql(TYPE_SQL).to_arrow_table()
    columns = am.from_duckdb(con.sql(TYPE_SQL), on_unsupported="raise")
    assert set(columns) == set(table.column_names)
    for name in table.column_names:
        assert isinstance(columns[name], am.MetalArray), f"{name} did not import onto the GPU"
        original = table.column(name).combine_chunks()
        assert columns[name].to_arrow().equals(original), f"{name} did not round-trip"


def test_import_is_zero_copy(con):
    """The Metal buffer is DuckDB's own page, not a copy of it - the whole point on unified memory."""
    table = con.sql("select i::BIGINT v from range(1000000) r(i)").to_arrow_table()
    assert am.duckdb_is_zero_copy(table.column("v").combine_chunks())


def test_nulls_survive_the_crossing(con):
    values = con.sql("select unnest([1, null, 3, null, 5])::BIGINT v").to_arrow_table()
    column = am.from_duckdb(con.sql("select unnest([1, null, 3, null, 5])::BIGINT v"))["v"]
    assert column.null_count == 2
    assert column.to_arrow().equals(values.column("v").combine_chunks())


def test_empty_result(con):
    columns = am.from_duckdb(con.sql("select i::BIGINT v from range(0) r(i)"))
    assert len(columns["v"]) == 0
    assert columns["v"].sum() is None


# ---------------------------------------------------------------------------------------------------
# from_duckdb: the many things you may hand it
# ---------------------------------------------------------------------------------------------------

def test_from_duckdb_accepts_sql_relation_and_arrow(con):
    make_table(con, 5000)
    expected = con.sql("select sum(v) from t").fetchone()[0]
    for source, kwargs in [
        (con.sql("select v from t"), {}),
        ("select v from t", {"con": con}),
        (con.table("t"), {}),
        (con.sql("select v from t").to_arrow_table(), {}),
    ]:
        columns = am.from_duckdb(source, **kwargs)
        assert columns["v"].sum() == expected


def test_from_duckdb_column_subset(con):
    make_table(con, 1000)
    columns = am.from_duckdb(con.sql("select * from t"), columns=["k", "v"])
    assert set(columns) == {"k", "v"}


# ---------------------------------------------------------------------------------------------------
# Aggregates, group-by, top-k and sort against DuckDB's own SQL
# ---------------------------------------------------------------------------------------------------

def test_aggregates_match_duckdb(con):
    make_table(con)
    columns = am.from_duckdb(con.sql("select v, f from t"))
    total, count, low, high, mean = con.sql(
        "select sum(v), count(v), min(v), max(v), avg(v) from t").fetchone()
    assert columns["v"].sum() == total
    assert len(columns["v"]) - columns["v"].null_count == count
    assert columns["v"].min() == low
    assert columns["v"].max() == high
    assert columns["v"].mean() == pytest.approx(mean)

    ftotal, fmean = con.sql("select sum(f), avg(f) from t").fetchone()
    assert columns["f"].sum() == pytest.approx(ftotal)
    assert columns["f"].mean() == pytest.approx(fmean)


def test_filter_then_sum_matches_duckdb(con):
    make_table(con)
    columns = am.from_duckdb(con.sql("select v from t"))
    expected = con.sql("select sum(v) from t where v > 50000").fetchone()[0]
    assert columns["v"].filter_where(">", 50000).sum() == expected


def test_group_by_matches_duckdb(con):
    make_table(con)
    columns = am.from_duckdb(con.sql("select k, v from t"))
    gb = am.group_by([columns["k"]])
    got = dict(zip(gb.keys()[0].to_pylist(), gb.sum(columns["v"]).to_arrow().to_pylist()))
    expected = dict(con.sql("select k, sum(v) from t group by k").fetchall())
    assert got == expected


def test_top_k_matches_duckdb(con):
    make_table(con)
    column = am.from_duckdb(con.sql("select v from t"))["v"]
    got = column.take(column.top_k(10)).to_arrow().to_pylist()
    expected = [row[0] for row in
                con.sql("select v from t where v is not null order by v desc limit 10").fetchall()]
    assert got == expected


def test_sort_matches_duckdb(con):
    make_table(con, 20_000)
    column = am.from_duckdb(con.sql("select v from t"))["v"]
    got = column.sort().to_arrow().to_pylist()
    expected = [row[0] for row in
                con.sql("select v from t order by v asc nulls last").fetchall()]
    assert got == expected


def test_string_contains_matches_duckdb(con):
    make_table(con)
    column = am.from_duckdb(con.sql("select s from t"))["s"]
    got = sum(1 for x in column.str_contains("row1").to_arrow().to_pylist() if x)
    expected = con.sql("select count(*) from t where s like '%row1%'").fetchone()[0]
    assert got == expected


def test_fused_query_matches_duckdb(con):
    make_table(con)
    columns = am.from_duckdb(con.sql("select k, v from t"))
    query = am.filter(am.col("v") > 50000).aggregate([("sum", "total", am.col("v")),
                                                      ("count", "n", None)])
    got = am.query(columns, query)
    total, n = con.sql("select sum(v), count(*) from t where v > 50000").fetchone()
    assert got["total"] == total
    assert got["n"] == n


# ---------------------------------------------------------------------------------------------------
# to_duckdb and duckdb_gpu_query
# ---------------------------------------------------------------------------------------------------

def test_to_duckdb_registers_a_queryable_view(con):
    make_table(con, 10_000)
    columns = am.from_duckdb(con.sql("select k, v from t"))
    gb = am.group_by([columns["k"]])
    relation = am.to_duckdb(con, "gpu_totals", {"k": gb.keys()[0], "total": gb.sum(columns["v"])})
    assert relation.aggregate("sum(total)").fetchone()[0] == \
        con.sql("select sum(v) from t").fetchone()[0]
    # and it joins like any other relation
    joined = con.sql("select count(*) from gpu_totals g join t on t.k = g.k").fetchone()[0]
    assert joined == con.sql("select count(*) from t").fetchone()[0]


def test_to_duckdb_accepts_scalars_and_pyarrow(con):
    am.to_duckdb(con, "scalars", {"total": 42, "name": pa.array(["x"])})
    assert con.sql("select total, name from scalars").fetchone() == (42, "x")


def test_duckdb_gpu_query_round_trip(con):
    make_table(con, 50_000)
    relation = am.duckdb_gpu_query(
        con, "select k, v from t",
        then=lambda cols: {"k": am.group_by([cols["k"]]).keys()[0],
                           "total": am.group_by([cols["k"]]).sum(cols["v"])},
        name="gpu_result")
    got = dict(relation.fetchall())
    expected = dict(con.sql("select k, sum(v) from t group by k").fetchall())
    assert got == expected


def test_duckdb_gpu_query_after_a_join(con):
    """The division of labour the bridge is for: DuckDB joins, ArrowMetal aggregates."""
    con.execute("create table orders as select (i % 50)::INTEGER customer, i::BIGINT amount "
                "from range(20000) r(i)")
    con.execute("create table customers as select i::INTEGER id, 'c' || i as name "
                "from range(50) r(i)")
    relation = am.duckdb_gpu_query(
        con,
        "select o.customer, o.amount from orders o join customers c on c.id = o.customer",
        then=lambda cols: {"customer": am.group_by([cols["customer"]]).keys()[0],
                           "total": am.group_by([cols["customer"]]).sum(cols["amount"])},
        name="per_customer")
    got = dict(relation.fetchall())
    expected = dict(con.sql(
        "select o.customer, sum(o.amount) from orders o join customers c on c.id = o.customer "
        "group by o.customer").fetchall())
    assert got == expected


# ---------------------------------------------------------------------------------------------------
# Streaming: one record batch at a time
# ---------------------------------------------------------------------------------------------------

def test_streaming_sees_twenty_batches(con):
    make_table(con, 100_000)
    batches = list(am.duckdb_batches(con.sql("select v from t"), rows_per_batch=5000))
    assert len(batches) >= 20
    assert sum(len(b["v"]) for b in batches) == 100_000


def test_streaming_aggregate_equals_whole_table(con):
    make_table(con, 100_000)
    got = am.duckdb_aggregate(
        con.sql("select k, v, f from t"),
        {"total": ("sum", "v"), "n": ("count", "v"), "low": ("min", "v"), "high": ("max", "v"),
         "avg": ("mean", "f"), "keys": ("count_distinct", "k")},
        rows_per_batch=5000)
    total, n, low, high, avg, keys = con.sql(
        "select sum(v), count(v), min(v), max(v), avg(f), count(distinct k) from t").fetchone()
    assert got["total"] == total
    assert got["n"] == n
    assert got["low"] == low
    assert got["high"] == high
    assert got["avg"] == pytest.approx(avg)
    assert got["keys"] == keys


def test_streaming_aggregate_is_batch_size_independent(con):
    """Twenty batches, two batches and one batch must all give the same answer."""
    make_table(con, 60_000)
    answers = [am.duckdb_aggregate(con.sql("select v from t"), {"s": ("sum", "v")},
                                   rows_per_batch=size)["s"]
               for size in (3_000, 30_000, 1_000_000)]
    assert len(set(answers)) == 1
    assert answers[0] == con.sql("select sum(v) from t").fetchone()[0]


def test_streaming_group_by_equals_whole_table(con):
    make_table(con, 100_000)
    got = am.duckdb_group_by(con.sql("select k, v from t"), "k",
                             {"total": ("sum", "v"), "n": ("count", "v"), "avg": ("mean", "v")},
                             rows_per_batch=5000)
    expected = {row[0]: row[1:] for row in
                con.sql("select k, sum(v), count(v), avg(v) from t group by k").fetchall()}
    assert got.num_rows == len(expected)
    for key, total, n, avg in zip(got.column("k").to_pylist(), got.column("total").to_pylist(),
                                  got.column("n").to_pylist(), got.column("avg").to_pylist()):
        want_total, want_n, want_avg = expected[key]
        assert total == want_total
        assert n == want_n
        assert avg == pytest.approx(want_avg)


def test_streaming_refuses_an_aggregate_it_cannot_merge_exactly(con):
    """A median of medians is not a median; the bridge says so instead of returning one."""
    make_table(con, 1000)
    with pytest.raises(am.ArrowMetalError, match="exact"):
        am.duckdb_aggregate(con.sql("select v from t"), {"m": ("median", "v")})


# ---------------------------------------------------------------------------------------------------
# The loadable extension
# ---------------------------------------------------------------------------------------------------

extension = pytest.mark.skipif(not os.path.exists(EXTENSION),
                               reason="build it with duckdb-extension/build.sh")


@pytest.fixture
def ext_con():
    connection = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    connection.execute(f"LOAD '{EXTENSION}'")
    yield connection
    connection.close()


@extension
def test_extension_loads_and_names_the_device(ext_con):
    version, device = ext_con.sql("select arrowmetal_version(), arrowmetal_device()").fetchone()
    assert version == am.version()
    assert device == am.device_name()


@extension
def test_extension_agg_matches_duckdb(ext_con):
    make_table(ext_con)
    got = ext_con.sql("select * from arrowmetal_agg('t', 'v')").fetchone()
    expected = ext_con.sql("select sum(v), count(v), min(v), max(v), avg(v) from t").fetchone()
    assert got[:4] == expected[:4]
    assert got[4] == pytest.approx(expected[4])


@extension
def test_extension_agg_on_a_double_column(ext_con):
    make_table(ext_con)
    got = ext_con.sql("select * from arrowmetal_agg('t', 'f')").fetchone()
    expected = ext_con.sql("select sum(f), count(f), min(f), max(f), avg(f) from t").fetchone()
    for a, b in zip(got, expected):
        assert a == pytest.approx(b)


@extension
def test_extension_group_by_matches_duckdb(ext_con):
    make_table(ext_con)
    got = ext_con.sql("select * from arrowmetal_group_by('t', 'k', 'v') order by key").fetchall()
    expected = ext_con.sql(
        "select k, count(v), sum(v), min(v), max(v) from t group by k order by k").fetchall()
    assert got == expected


@extension
def test_extension_group_by_a_float_key(ext_con):
    """A DOUBLE key is declared DOUBLE and must be written as one - it used to go out as a BIGINT."""
    ext_con.execute("create table fk as select (i % 5 + 0.5)::DOUBLE k, i::BIGINT v "
                    "from range(1000) r(i)")
    got = ext_con.sql("select * from arrowmetal_group_by('fk', 'k', 'v') order by key").fetchall()
    expected = ext_con.sql(
        "select k, count(v), sum(v), min(v), max(v) from fk group by k order by k").fetchall()
    assert got == expected


@extension
def test_extension_top_k_matches_duckdb(ext_con):
    make_table(ext_con)
    got = [row[0] for row in ext_con.sql("select * from arrowmetal_top_k('t', 'v', 10)").fetchall()]
    expected = [row[0] for row in ext_con.sql(
        "select v from t where v is not null order by v desc limit 10").fetchall()]
    assert got == expected


@extension
def test_extension_sort_matches_duckdb(ext_con):
    make_table(ext_con, 20_000)
    got = [row[0] for row in ext_con.sql("select * from arrowmetal_sort('t', 'v')").fetchall()]
    expected = [row[0] for row in ext_con.sql("select v from t order by v asc nulls last").fetchall()]
    assert got == expected


@extension
def test_extension_query_matches_duckdb(ext_con):
    make_table(ext_con)
    expression = ('(query (filter (gt (col "v") (int 50000))) '
                  '(aggregate (sum "total" (col "v")) (count "n")))')
    rows = {name: (value, exact) for name, value, exact in
            ext_con.sql(f"select * from arrowmetal_query('t', $${expression}$$)").fetchall()}
    total, n = ext_con.sql("select sum(v), count(*) from t where v > 50000").fetchone()
    assert rows["total"][1] == total, "the exact column must carry the int64 total undamaged"
    assert rows["total"][0] == pytest.approx(float(total))
    assert rows["n"][1] == n


@extension
def test_extension_output_spans_several_chunks(ext_con):
    """More rows than one DataChunk holds, so the paging in main() is exercised."""
    make_table(ext_con, 50_000)
    assert ext_con.sql("select count(*) from arrowmetal_sort('t', 'v')").fetchone()[0] == 50_000


@extension
def test_extension_refuses_a_type_it_cannot_read(ext_con):
    make_table(ext_con, 100)
    with pytest.raises(duckdb.Error, match="type"):
        ext_con.sql("select * from arrowmetal_agg('t', 's')").fetchall()


@extension
def test_extension_results_join_like_any_table(ext_con):
    make_table(ext_con, 10_000)
    got = ext_con.sql("""
        select count(*) from arrowmetal_group_by('t', 'k', 'v') g join t on t.k = g.key
    """).fetchone()[0]
    assert got == ext_con.sql("select count(*) from t").fetchone()[0]


# ---------------------------------------------------------------------------------------------------
# Regression tests from the pre-release integration review.
# ---------------------------------------------------------------------------------------------------

def test_streaming_refuses_a_column_that_is_not_there(con):
    """`columns=` used to hand back the *last* column of the batch for a name that is not in it.

    `RecordBatch.schema.get_field_index(name)` answers -1 for an unknown name and
    `record_batch.column(-1)` is the last column, so a typo in an aggregate's column name summed a
    different column and said nothing. Every path that takes `columns=` goes through here.
    """
    make_table(con, 5_000)
    with pytest.raises(am.ArrowMetalError, match="no column"):
        list(am.duckdb_batches(con.sql("select k, v from t"), columns=["amuont"]))
    with pytest.raises(am.ArrowMetalError, match="no column"):
        am.duckdb_aggregate(con.sql("select k, v from t"), {"total": ("sum", "amuont")})
    # the spelled-right form is unaffected
    got = am.duckdb_aggregate(con.sql("select k, v from t"), {"total": ("sum", "v")})
    assert got["total"] == con.sql("select sum(v) from t").fetchone()[0]


def test_streaming_count_of_nothing_is_zero(con):
    """SQL's `count` over no rows is 0, and `count_distinct` here already answered 0; `count`
    answered None, because its accumulator started at None and no batch ever ran."""
    make_table(con, 1_000)
    empty = con.sql("select k, v from t where v < 0")
    got = am.duckdb_aggregate(empty, {"n": ("count", "v"), "d": ("count_distinct", "v"),
                                      "total": ("sum", "v"), "lo": ("min", "v"),
                                      "avg": ("mean", "v")})
    assert got["n"] == 0
    assert got["d"] == 0
    # sum/min/mean of nothing stay NULL, which is what SQL says too
    assert (got["total"], got["lo"], got["avg"]) == (None, None, None)
    assert con.sql("select count(v), sum(v), min(v), avg(v) from t where v < 0").fetchone() \
        == (got["n"], got["total"], got["lo"], got["avg"])


def test_to_duckdb_takes_the_scalars_a_reduction_actually_returns(con):
    """`arrow_table` recognised a scalar by `isinstance(value, (int, float, bool))`, so a numpy
    scalar (`np.int64`, what a numpy-backed reduction hands back) and a plain string both fell
    through to `pa.array(value)` and raised `TypeError: not iterable`."""
    import numpy as np
    rel = am.to_duckdb(con, "scalars", {"n": np.int64(7), "f": np.float64(1.5),
                                        "label": "total", "nothing": None, "plain": 3})
    assert rel.fetchall() == [(7, 1.5, "total", None, 3)]


def test_streaming_group_by_key_order_is_the_gpu_group_order(con):
    """`duckdb_group_by` said "ordered by first appearance". The order is the per-batch group-by's
    own: ascending by key for a numeric key, whatever the rows came in as."""
    con.execute("create table go as select * from (values (5, 1), (1, 2), (3, 3)) v(k, x)")
    got = am.duckdb_group_by(con.sql("select k, x from go"), "k", {"total": ("sum", "x")})
    assert got.column("k").to_pylist() == [1, 3, 5]
    assert got.column("total").to_pylist() == [2, 3, 1]


@pytest.mark.xfail(strict=True,
                   reason="REVIEW: include/arrowmetal.h declares am_plan_source as both a typedef "
                          "and a function, so the public C header does not compile and the DuckDB "
                          "extension cannot be built")
def test_the_public_c_header_compiles():
    """include/arrowmetal.h declares `am_plan_source` twice, as a typedef and as a function:

        typedef struct am_plan_source_t am_plan_source;
        int  am_plan_source(const char* name, ...);

    That is "redefinition as a different kind of symbol" in both C and C++, so *no* C consumer of
    the published header can compile -- including this repository's own DuckDB extension, which is
    why every `@extension` test in this file skips. Rename one of the two (the struct handle is the
    one the docs call `am_plan_source`, so the function wants to be `am_plan_source_create`) and
    drop this marker.
    """
    import shutil
    import subprocess
    import tempfile
    cc = shutil.which("cc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler on PATH")
    header = os.path.join(REPO, "include")
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "include_only.c")
        with open(src, "w") as fh:
            fh.write('#include "arrowmetal.h"\nint main(void) { return 0; }\n')
        out = subprocess.run([cc, "-fsyntax-only", "-I", header, src],
                             capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
