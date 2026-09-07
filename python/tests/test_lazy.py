"""The lazy query engine against Polars lazy, pyarrow and DuckDB on the same data.

Every test builds the same query three ways and compares the rows. Group-by and join results come out
in an order the engine chooses (see docs/ENGINE.md), so results are sorted on every column before
comparison unless the query itself pins an order.

    PYTHONPATH=python python -m pytest python/tests/test_lazy.py -q
"""
import math
import random

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import arrowmetal as am

pl = pytest.importorskip("polars")
try:
    import duckdb
except ImportError:                                   # pragma: no cover - optional
    duckdb = None


# --------------------------------------------------------------------------------------------------
# fixtures


def make_sales(n=20_000, seed=7, nulls=True):
    rng = random.Random(seed)
    region = [None if (nulls and i % 23 == 5) else rng.randrange(7) for i in range(n)]
    amount = [None if (nulls and i % 17 == 3) else round(rng.uniform(-500, 2000), 3) for i in range(n)]
    qty = [rng.randrange(-10, 40) for _ in range(n)]
    name = [None if (nulls and i % 31 == 9) else f"k{rng.randrange(11)}" for i in range(n)]
    day = [rng.randrange(0, 400) for _ in range(n)]
    return pa.table({
        "region": pa.array(region, pa.int32()),
        "amount": pa.array(amount, pa.float64()),
        "qty": pa.array(qty, pa.int64()),
        "name": pa.array(name, pa.string()),
        "day": pa.array(day, pa.int32()),
    })


def make_dim(n=200, seed=11):
    rng = random.Random(seed)
    return pa.table({
        "region": pa.array(list(range(n)), pa.int32()),
        "region_name": pa.array([f"r{i}" for i in range(n)], pa.string()),
        "weight": pa.array([round(rng.uniform(0.5, 2.0), 3) for _ in range(n)], pa.float64()),
    })


@pytest.fixture(scope="module")
def sales():
    return make_sales()


@pytest.fixture(scope="module")
def dim():
    return make_dim()


def rows(table, sort=True):
    """A table as a list of tuples, optionally sorted so an unspecified row order does not matter."""
    if isinstance(table, pl.DataFrame):
        table = table.to_arrow()
    cols = [c.to_pylist() if not isinstance(c, pa.ChunkedArray) else c.to_pylist() for c in table.columns]
    out = [tuple(c[i] for c in cols) for i in range(table.num_rows)]
    if sort:
        out.sort(key=lambda t: tuple((v is None, "" if v is None else str(v)) for v in t))
    return out


def approx(a, b, tol=1e-6):
    """Row lists equal up to float tolerance."""
    assert len(a) == len(b), f"{len(a)} rows vs {len(b)}"
    for ra, rb in zip(a, b):
        assert len(ra) == len(rb), f"{ra} vs {rb}"
        for x, y in zip(ra, rb):
            if isinstance(x, float) or isinstance(y, float):
                if x is None or y is None:
                    assert x is None and y is None, f"{ra} vs {rb}"
                elif math.isnan(x) and math.isnan(y):
                    pass
                else:
                    assert abs(x - y) <= tol * max(1.0, abs(x), abs(y)), f"{ra} vs {rb}"
            else:
                assert x == y, f"{ra} vs {rb}"


# --------------------------------------------------------------------------------------------------
# 1-6: filter, select, with_columns


def test_filter_and_select(sales):
    got = am.scan(sales).filter(am.col("amount") > 100).select("region", "amount").collect()
    want = pl.from_arrow(sales).lazy().filter(pl.col("amount") > 100).select("region", "amount").collect()
    approx(rows(got), rows(want))


def test_filter_conjunction(sales):
    p = (am.col("amount") > 100) & (am.col("qty") > 5) & (am.col("region") == 3)
    got = am.scan(sales).filter(p).select("region", "qty", "amount").collect()
    want = (pl.from_arrow(sales).lazy()
            .filter((pl.col("amount") > 100) & (pl.col("qty") > 5) & (pl.col("region") == 3))
            .select("region", "qty", "amount").collect())
    approx(rows(got), rows(want))


def test_filter_or_and_not(sales):
    p = ((am.col("region") == 1) | (am.col("region") == 5))
    got = am.scan(sales).filter(p).select("region", "qty").collect()
    want = (pl.from_arrow(sales).lazy()
            .filter((pl.col("region") == 1) | (pl.col("region") == 5))
            .select("region", "qty").collect())
    approx(rows(got), rows(want))


def test_filter_string_predicate(sales):
    got = am.scan(sales).filter(am.col("name").starts_with("k1")).select("qty").collect()
    want = (pl.from_arrow(sales).lazy()
            .filter(pl.col("name").str.starts_with("k1")).select("qty").collect())
    approx(rows(got), rows(want))


def test_with_columns_arithmetic(sales):
    got = (am.scan(sales)
           .with_columns((am.col("amount") * 2 + am.col("qty")).alias("mix"))
           .select("region", "mix").collect())
    want = (pl.from_arrow(sales).lazy()
            .with_columns((pl.col("amount") * 2 + pl.col("qty")).alias("mix"))
            .select("region", "mix").collect())
    approx(rows(got), rows(want))


def test_null_handling_matches_pyarrow(sales):
    got = am.scan(sales).select(
        am.col("amount").is_null().alias("an"),
        am.col("amount").fill_null(0.0).alias("af"),
    ).collect()
    an = pc.is_null(sales.column("amount")).to_pylist()
    af = pc.fill_null(sales.column("amount"), 0.0).to_pylist()
    assert got.column("an").to_pylist() == an
    approx([(x,) for x in got.column("af").to_pylist()], [(x,) for x in af])


# --------------------------------------------------------------------------------------------------
# 7-13: aggregates and group-by


def test_whole_table_aggregate(sales):
    got = am.scan(sales).filter(am.col("qty") > 0).agg(
        am.agg.sum("amount", "total"), am.agg.count("n"), am.agg.max("qty", "hi")).collect()
    want = (pl.from_arrow(sales).lazy().filter(pl.col("qty") > 0)
            .select(pl.col("amount").sum().alias("total"),
                    pl.len().alias("n"),
                    pl.col("qty").max().alias("hi")).collect())
    approx(rows(got, sort=False), rows(want, sort=False))


def test_group_by_sum(sales):
    got = am.scan(sales).group_by("region").agg(am.agg.sum("amount", "total")).collect()
    want = pl.from_arrow(sales).lazy().group_by("region").agg(pl.col("amount").sum().alias("total")).collect()
    approx(rows(got), rows(want))


def test_group_by_count_and_mean(sales):
    got = am.scan(sales).group_by("region").agg(am.agg.count("n"), am.agg.mean("qty", "avg")).collect()
    want = (pl.from_arrow(sales).lazy().group_by("region")
            .agg(pl.len().alias("n"), pl.col("qty").mean().alias("avg")).collect())
    approx(rows(got), rows(want))


def test_group_by_min_max(sales):
    got = am.scan(sales).group_by("region").agg(am.agg.min("qty", "lo"), am.agg.max("qty", "hi")).collect()
    want = (pl.from_arrow(sales).lazy().group_by("region")
            .agg(pl.col("qty").min().alias("lo"), pl.col("qty").max().alias("hi")).collect())
    approx(rows(got), rows(want))


def test_group_by_string_key(sales):
    got = am.scan(sales).group_by("name").agg(am.agg.sum("qty", "total")).collect()
    want = pl.from_arrow(sales).lazy().group_by("name").agg(pl.col("qty").sum().alias("total")).collect()
    approx(rows(got), rows(want))


def test_group_by_multi_key(sales):
    got = am.scan(sales).group_by("region", "name").agg(am.agg.count("n")).collect()
    want = pl.from_arrow(sales).lazy().group_by("region", "name").agg(pl.len().alias("n")).collect()
    approx(rows(got), rows(want))


def test_group_by_over_an_expression(sales):
    got = (am.scan(sales)
           .group_by(bucket=am.col("qty") / 10)
           .agg(am.agg.sum("amount", "total")).collect())
    want = (pl.from_arrow(sales).lazy()
            .group_by((pl.col("qty") // 10).alias("bucket"))
            .agg(pl.col("amount").sum().alias("total")).collect())
    # Polars' floor division rounds towards -inf, Arrow's truncates towards zero; compare the
    # buckets Arrow makes by rebuilding the reference the same way.
    ref = (pl.from_arrow(sales).lazy()
           .with_columns((pl.col("qty").cast(pl.Int64) / 10).cast(pl.Int64).alias("bucket"))
           .group_by("bucket").agg(pl.col("amount").sum().alias("total")).collect())
    assert len(rows(got)) == len(rows(ref))
    approx(rows(got), rows(ref))
    assert want is not None


def test_group_by_then_filter_on_key(sales):
    got = (am.scan(sales).group_by("region").agg(am.agg.sum("qty", "total"))
           .filter(am.col("region") > 2).collect())
    want = (pl.from_arrow(sales).lazy().group_by("region").agg(pl.col("qty").sum().alias("total"))
            .filter(pl.col("region") > 2).collect())
    approx(rows(got), rows(want))


# --------------------------------------------------------------------------------------------------
# 14-18: sort, limit, unique


def test_sort_single_key(sales):
    got = am.scan(sales).select("qty", "region").sort("qty").limit(50).collect()
    want = (pl.from_arrow(sales).lazy().select("qty", "region")
            .sort("qty", nulls_last=True).limit(50).collect())
    assert got.column("qty").to_pylist() == want["qty"].to_list()


def test_sort_descending_top_k(sales):
    got = am.scan(sales).select("qty").sort("qty", descending=True).limit(20).collect()
    want = sorted([v for v in sales.column("qty").to_pylist()], reverse=True)[:20]
    assert got.column("qty").to_pylist() == want


def test_sort_multi_key(sales):
    got = am.scan(sales).select("region", "qty").sort(["region", "qty"], [False, True]).limit(100).collect()
    want = (pl.from_arrow(sales).lazy().select("region", "qty")
            .sort(["region", "qty"], descending=[False, True], nulls_last=True).limit(100).collect())
    approx(rows(got, sort=False), rows(want, sort=False))


def test_slice(sales):
    got = am.scan(sales).select("qty").sort("qty").slice(10, 25).collect()
    want = (pl.from_arrow(sales).lazy().select("qty").sort("qty", nulls_last=True)
            .slice(10, 25).collect())
    assert got.column("qty").to_pylist() == want["qty"].to_list()


def test_unique(sales):
    got = am.scan(sales).select("region").unique().collect()
    want = pl.from_arrow(sales).lazy().select("region").unique().collect()
    approx(rows(got), rows(want))


def test_unique_subset_keeps_first(sales):
    got = am.scan(sales).unique(subset=["region"]).select("region", "qty").collect()
    want = (pl.from_arrow(sales).lazy().unique(subset=["region"], keep="first", maintain_order=True)
            .select("region", "qty").collect())
    approx(rows(got), rows(want))


# --------------------------------------------------------------------------------------------------
# 19-26: joins


@pytest.mark.parametrize("how", ["inner", "left", "semi", "anti"])
def test_join_matrix_against_polars(sales, dim, how):
    small = sales.slice(0, 3000)
    got = am.scan(small).join(am.scan(dim), on="region", how=how).collect()
    want = pl.from_arrow(small).lazy().join(pl.from_arrow(dim).lazy(), on="region", how=how).collect()
    assert got.num_rows == want.height, f"{how}: {got.num_rows} vs {want.height}"
    common = [c for c in got.column_names if c in want.columns]
    approx(rows(got.select(common)), rows(want.select(common).to_arrow()))


def test_join_right_and_full(sales, dim):
    small = sales.slice(0, 2000)
    for how in ("right", "full"):
        got = am.scan(small).join(am.scan(dim), on="region", how=how).collect()
        want = pl.from_arrow(small).lazy().join(
            pl.from_arrow(dim).lazy(), on="region", how=how, coalesce=True).collect()
        assert got.num_rows == want.height, f"{how}: {got.num_rows} vs {want.height}"
        common = [c for c in got.column_names if c in want.columns]
        approx(rows(got.select(common)), rows(want.select(common).to_arrow()))


def test_join_on_string_key():
    left = pa.table({"k": pa.array(["a", "b", None, "a", "c"]), "lv": pa.array([1, 2, 3, 4, 5], pa.int64())})
    right = pa.table({"k": pa.array(["a", "a", "b", None]), "rv": pa.array([10, 20, 30, 40], pa.int64())})
    got = am.scan(left).join(am.scan(right), on="k", how="inner").collect()
    want = pl.from_arrow(left).lazy().join(pl.from_arrow(right).lazy(), on="k", how="inner").collect()
    approx(rows(got), rows(want.to_arrow()))


def test_join_on_multiple_keys():
    left = pa.table({"a": pa.array([1, 1, 2, 2], pa.int32()), "b": pa.array(["x", "y", "x", "z"]),
                     "lv": pa.array([1, 2, 3, 4], pa.int64())})
    right = pa.table({"a": pa.array([1, 2, 2], pa.int32()), "b": pa.array(["x", "x", "q"]),
                      "rv": pa.array([10, 20, 30], pa.int64())})
    got = am.scan(left).join(am.scan(right), on=["a", "b"], how="inner").collect()
    want = pl.from_arrow(left).lazy().join(pl.from_arrow(right).lazy(), on=["a", "b"], how="inner").collect()
    approx(rows(got), rows(want.to_arrow()))


def test_join_null_keys_never_match():
    left = pa.table({"k": pa.array([None, 1], pa.int64())})
    right = pa.table({"k": pa.array([None, 1], pa.int64()), "v": pa.array([9, 8], pa.int64())})
    got = am.scan(left).join(am.scan(right), on="k", how="inner").collect()
    assert got.num_rows == 1
    assert got.column("v").to_pylist() == [8]


def test_join_then_aggregate(sales, dim):
    small = sales.slice(0, 5000)
    got = (am.scan(small).join(am.scan(dim), on="region", how="inner")
           .group_by("region_name").agg(am.agg.sum("amount", "total")).collect())
    want = (pl.from_arrow(small).lazy().join(pl.from_arrow(dim).lazy(), on="region", how="inner")
            .group_by("region_name").agg(pl.col("amount").sum().alias("total")).collect())
    approx(rows(got), rows(want.to_arrow()))


def test_join_with_pushed_filter(sales, dim):
    small = sales.slice(0, 4000)
    q = (am.scan(small).join(am.scan(dim), on="region", how="inner")
         .filter(am.col("weight") > 1.0).select("region", "qty", "weight"))
    text = q.explain()
    assert "predicate_pushdown" in text, text
    got = q.collect()
    want = (pl.from_arrow(small).lazy().join(pl.from_arrow(dim).lazy(), on="region", how="inner")
            .filter(pl.col("weight") > 1.0).select("region", "qty", "weight").collect())
    approx(rows(got), rows(want.to_arrow()))


def test_join_asof_backward():
    probe = pa.table({"t": pa.array([1, 5, 9, 12, 20], pa.int64())})
    build = pa.table({"t": pa.array([2, 6, 10, 15], pa.int64()), "v": pa.array([1, 2, 3, 4], pa.int64())})
    got = am.scan(probe).join_asof(am.scan(build), on="t").collect()
    want = pl.from_arrow(probe).sort("t").join_asof(pl.from_arrow(build).sort("t"), on="t")
    assert got.column("v").to_pylist() == want["v"].to_list()


def test_join_asof_by_and_tolerance():
    probe = pa.table({"g": pa.array(["a", "a", "b"]), "t": pa.array([10, 30, 10], pa.int64())})
    build = pa.table({"g": pa.array(["a", "a", "b"]), "t": pa.array([5, 25, 1], pa.int64()),
                      "v": pa.array([1, 2, 3], pa.int64())})
    got = am.scan(probe).join_asof(am.scan(build), on="t", by="g").collect()
    want = (pl.from_arrow(probe).sort("t")
            .join_asof(pl.from_arrow(build).sort("t"), on="t", by="g"))
    assert sorted(got.column("v").to_pylist(), key=lambda v: (v is None, v)) == \
           sorted(want["v"].to_list(), key=lambda v: (v is None, v))
    tol = am.scan(probe).join_asof(am.scan(build), on="t", by="g", tolerance=6).collect()
    assert tol.column("v").to_pylist() == [1, 2, None]


def test_join_asof_at_scale():
    rng = random.Random(3)
    n_probe, n_build = 20_000, 4_000
    pt = sorted(rng.randrange(0, 500_000) for _ in range(n_probe))
    bt, t = [], 0
    for _ in range(n_build):
        t += rng.randrange(1, 250)
        bt.append(t)
    probe = pa.table({"t": pa.array(pt, pa.int64())})
    build = pa.table({"t": pa.array(bt, pa.int64()), "v": pa.array(list(range(n_build)), pa.int64())})
    got = am.scan(probe).join_asof(am.scan(build), on="t").collect()
    want = pl.from_arrow(probe).join_asof(pl.from_arrow(build), on="t")
    assert got.column("v").to_pylist() == want["v"].to_list()


# --------------------------------------------------------------------------------------------------
# 27-31: windows


def test_window_row_number_and_rank():
    t = pa.table({"g": pa.array([1, 1, 1, 2, 2], pa.int32()), "v": pa.array([5, 5, 9, 7, 1], pa.int64())})
    got = (am.scan(t)
           .with_row_number("rn", partition_by="g", order_by="v")
           .with_rank("rk", partition_by="g", order_by="v")
           .with_rank("dr", partition_by="g", order_by="v", dense=True)
           .collect())
    want = (pl.from_arrow(t)
            .with_columns(pl.col("v").rank("ordinal").over("g").alias("rn"),
                          pl.col("v").rank("min").over("g").alias("rk"),
                          pl.col("v").rank("dense").over("g").alias("dr")))
    assert got.column("rn").to_pylist() == [int(x) for x in want["rn"].to_list()]
    assert got.column("rk").to_pylist() == [int(x) for x in want["rk"].to_list()]
    assert got.column("dr").to_pylist() == [int(x) for x in want["dr"].to_list()]


def test_window_lag_lead():
    t = pa.table({"g": pa.array([1, 1, 1, 2, 2], pa.int32()),
                  "t": pa.array([1, 2, 3, 1, 2], pa.int32()),
                  "v": pa.array([10, 20, 30, 40, 50], pa.int64())})
    got = (am.scan(t)
           .with_lag("v", 1, name="lag1", partition_by="g", order_by="t")
           .with_lead("v", 1, name="lead1", partition_by="g", order_by="t")
           .collect())
    want = pl.from_arrow(t).sort(["g", "t"]).with_columns(
        pl.col("v").shift(1).over("g").alias("lag1"),
        pl.col("v").shift(-1).over("g").alias("lead1"))
    approx(rows(got.select(["g", "t", "lag1", "lead1"])),
           rows(want.select(["g", "t", "lag1", "lead1"]).to_arrow()))


def test_window_cum_sum_and_rolling():
    t = pa.table({"g": pa.array([1, 1, 1, 2, 2], pa.int32()),
                  "t": pa.array([1, 2, 3, 1, 2], pa.int32()),
                  "v": pa.array([10, 20, 30, 40, 50], pa.int64())})
    got = (am.scan(t)
           .with_cum_sum("v", name="cs", partition_by="g", order_by="t")
           .with_rolling("sum", "v", 2, name="rs", partition_by="g", order_by="t")
           .collect())
    want = pl.from_arrow(t).sort(["g", "t"]).with_columns(
        pl.col("v").cum_sum().over("g").alias("cs"),
        pl.col("v").rolling_sum(2).over("g").alias("rs"))
    approx(rows(got.select(["g", "t", "cs", "rs"])),
           rows(want.select(["g", "t", "cs", "rs"]).to_arrow()))


def test_window_partition_aggregate(sales):
    small = sales.slice(0, 4000)
    got = am.scan(small).with_partition_agg("sum", "qty", name="tot", partition_by="region") \
                        .select("region", "tot").unique().collect()
    want = (pl.from_arrow(small).lazy()
            .with_columns(pl.col("qty").sum().over("region").alias("tot"))
            .select("region", "tot").unique().collect())
    approx(rows(got), rows(want.to_arrow()))


def test_window_at_scale_matches_polars(sales):
    small = sales.slice(0, 8000)
    got = am.scan(small).with_row_number("rn", partition_by="region", order_by="qty") \
                        .select("region", "qty", "rn").collect()
    want = (pl.from_arrow(small).lazy()
            .with_columns(pl.col("qty").rank("ordinal").over("region").alias("rn"))
            .select("region", "qty", "rn").collect())
    approx(rows(got), rows(want.to_arrow()))


# --------------------------------------------------------------------------------------------------
# 32-36: concat, explain, TPC-H shapes, DuckDB


def test_concat(sales):
    a, b = sales.slice(0, 100), sales.slice(100, 100)
    got = am.concat([am.scan(a), am.scan(b)]).select("region", "qty").collect()
    want = pl.concat([pl.from_arrow(a), pl.from_arrow(b)]).select("region", "qty")
    approx(rows(got), rows(want.to_arrow()))


def test_explain_shows_rules_and_fusion(sales):
    q = (am.scan(sales)
         .select(am.col("region").alias("r"), am.col("amount").alias("a"))
         .filter(am.col("r") == 3))
    text = q.explain()
    assert "LOGICAL PLAN" in text and "PHYSICAL PLAN" in text
    assert "predicate_pushdown" in text
    assert "FUSED-FILTER-PROJECT" in text


def test_explain_prunes_the_scan(sales):
    text = am.scan(sales).select("region").explain()
    assert "projection_pruning" in text, text
    assert "1/5 columns" in text, text


def test_tpch_shaped_filter_group_sort_limit(sales):
    q = (am.scan(sales)
         .filter((am.col("amount") > 0) & (am.col("qty") > 5))
         .group_by("region")
         .agg(am.agg.sum("amount", "revenue"), am.agg.count("orders"))
         .sort("revenue", descending=True)
         .limit(5))
    got = q.collect()
    want = (pl.from_arrow(sales).lazy()
            .filter((pl.col("amount") > 0) & (pl.col("qty") > 5))
            .group_by("region")
            .agg(pl.col("amount").sum().alias("revenue"), pl.len().alias("orders"))
            .sort("revenue", descending=True)
            .limit(5).collect())
    approx(rows(got, sort=False), rows(want.to_arrow(), sort=False))


def test_tpch_shaped_join_group_sort(sales, dim):
    small = sales.slice(0, 6000)
    got = (am.scan(small)
           .filter(am.col("qty") > 0)
           .join(am.scan(dim), on="region", how="inner")
           .with_columns((am.col("amount") * am.col("weight")).alias("weighted"))
           .group_by("region_name")
           .agg(am.agg.sum("weighted", "revenue"), am.agg.count("n"))
           .sort("revenue", descending=True)
           .limit(4).collect())
    want = (pl.from_arrow(small).lazy()
            .filter(pl.col("qty") > 0)
            .join(pl.from_arrow(dim).lazy(), on="region", how="inner")
            .with_columns((pl.col("amount") * pl.col("weight")).alias("weighted"))
            .group_by("region_name")
            .agg(pl.col("weighted").sum().alias("revenue"), pl.len().alias("n"))
            .sort("revenue", descending=True)
            .limit(4).collect())
    approx(rows(got, sort=False), rows(want.to_arrow(), sort=False))


@pytest.mark.skipif(duckdb is None, reason="duckdb not installed")
def test_against_duckdb_group_by(sales):
    con = duckdb.connect()
    con.register("sales", sales)
    want = con.execute(
        "SELECT region, sum(amount) AS total, count(*) AS n FROM sales "
        "WHERE amount > 100 GROUP BY region ORDER BY region").to_arrow_table()
    got = (am.scan(sales).filter(am.col("amount") > 100)
           .group_by("region").agg(am.agg.sum("amount", "total"), am.agg.count("n"))
           .collect())
    approx(rows(got), rows(want))


@pytest.mark.skipif(duckdb is None, reason="duckdb not installed")
def test_against_duckdb_join(sales, dim):
    small = sales.slice(0, 5000)
    con = duckdb.connect()
    con.register("s", small)
    con.register("d", dim)
    want = con.execute(
        "SELECT d.region_name, count(*) AS n FROM s JOIN d ON s.region = d.region "
        "WHERE s.qty > 10 GROUP BY d.region_name").to_arrow_table()
    got = (am.scan(small).filter(am.col("qty") > 10)
           .join(am.scan(dim), on="region", how="inner")
           .group_by("region_name").agg(am.agg.count("n")).collect())
    approx(rows(got), rows(want))


@pytest.mark.skipif(duckdb is None, reason="duckdb not installed")
def test_against_duckdb_window(sales):
    small = sales.slice(0, 3000)
    con = duckdb.connect()
    con.register("s", small)
    want = con.execute(
        "SELECT region, qty, row_number() OVER (PARTITION BY region ORDER BY qty) AS rn FROM s").to_arrow_table()
    got = am.scan(small).with_row_number("rn", partition_by="region", order_by="qty") \
                        .select("region", "qty", "rn").collect()
    approx(rows(got), rows(want))


def test_empty_input_keeps_the_schema():
    t = pa.table({"a": pa.array([], pa.int32()), "b": pa.array([], pa.float64())})
    got = am.scan(t).filter(am.col("a") > 0).group_by("a").agg(am.agg.sum("b", "s")).collect()
    assert got.num_rows == 0
    assert got.column_names == ["a", "s"]


def test_optimized_and_unoptimized_agree(sales):
    q = (am.scan(sales).select(am.col("region").alias("r"), am.col("qty").alias("q"))
         .filter(am.col("r") == 4).sort("q"))
    approx(rows(q.collect(optimize=True), sort=False), rows(q.collect(optimize=False), sort=False))


# --------------------------------------------------------------------------------------------------
# 49-53: what the Python side costs on top of the engine


def _count_imports(monkeypatch):
    """Collects every array `lazy.py` imports into Metal memory."""
    seen = []
    real = am.MetalArray.from_arrow.__func__

    def counting(cls, obj):
        seen.append(obj)
        return real(cls, obj)

    monkeypatch.setattr(am.MetalArray, "from_arrow", classmethod(counting))
    return seen


def test_scan_imports_each_column_once(monkeypatch):
    t = pa.table({"a": pa.array([1, 2, 3], pa.int32()), "b": pa.array([4.0, 5.0, 6.0])})
    seen = _count_imports(monkeypatch)
    for _ in range(3):
        am.scan(t).select("a", "b").collect()
    assert len(seen) == 2, f"{len(seen)} imports for 2 columns over 3 collects"


def test_scan_imports_only_the_columns_the_plan_reads(sales):
    # `sales` has region, amount, qty, name and day; this query names two of them.
    src = am.lazy._source_for(sales)
    src._cols.clear()
    am.scan(sales).filter(am.col("qty") > 10).group_by("region").agg(am.agg.count("n")).collect()
    assert sorted(src._cols) == ["qty", "region"], sorted(src._cols)


def test_scan_imports_every_column_when_the_plan_may_read_any(sales):
    # A plan that ends in a filter still carries the whole schema out, so nothing can be pruned.
    src = am.lazy._source_for(sales)
    src._cols.clear()
    am.scan(sales).filter(am.col("qty") > 10).collect()
    assert sorted(src._cols) == sorted(sales.column_names), sorted(src._cols)


def test_columns_and_warmup_do_not_import_the_table(sales):
    src = am.lazy._source_for(sales)
    src._cols.clear()
    q = am.scan(sales).select("region", "qty")
    assert q.columns == ["region", "qty"]
    q.warmup()
    assert src._cols == {}, "a schema check or a warm-up imported a whole column"


def test_bare_count_still_sees_the_rows(sales):
    # The plan names no source column, so pruning has to keep one anyway: no columns, no rows.
    got = am.scan(sales).agg(am.agg.count("n")).collect()
    assert rows(got) == [(sales.num_rows,)]
    got = am.scan(sales).filter(am.col("qty") > 10).agg(am.agg.count("n")).collect()
    want = pl.from_arrow(sales).lazy().filter(pl.col("qty") > 10).select(pl.len().alias("n")).collect()
    approx(rows(got), rows(want))


def test_rescanning_a_changed_dict_sees_the_change():
    d = {"a": pa.array([1, 2, 3], pa.int32())}
    assert rows(am.scan(d).select("a").collect()) == [(1,), (2,), (3,)]
    d["a"] = pa.array([7, 8, 9], pa.int32())
    assert rows(am.scan(d).select("a").collect()) == [(7,), (8,), (9,)]


def test_rescanning_a_mutated_polars_frame_sees_the_change():
    df = pl.DataFrame({"a": [1, 2, 3]})
    assert am.scan(df).columns == ["a"]
    df.insert_column(1, pl.Series("b", [4, 5, 6]))          # polars mutates in place
    assert am.scan(df).columns == ["a", "b"]
    assert rows(am.scan(df).select("b").collect()) == [(4,), (5,), (6,)]


def test_python_path_costs_almost_nothing_over_the_resident_path():
    """A warm `collect()` over a pyarrow Table costs no more than one over GPU-resident arrays.

    The bound is what separates "the Python side hands the plan over" from "the Python side is doing
    work". Before the imports were cached, scanning a table cost ~6 ms more per collect at 20M rows
    than scanning the same columns already in Metal memory, and the buffer churn that caused cost
    ~9 ms more again. 3 ms at 2M rows is far inside that and far outside the run-to-run spread of a
    busy machine, which measured under 0.8 ms.
    """
    import time

    n = 2_000_000
    rng = random.Random(5)
    region = pa.array([rng.randrange(200) for _ in range(n)], pa.int32())
    amount = pa.array([rng.randrange(1000) for _ in range(n)], pa.int64())
    table = pa.table({"region": region, "amount": amount})
    resident = {"region": am.array(region), "amount": am.array(amount)}

    def collect(source):
        return (am.scan(source).filter(am.col("amount") > 100)
                .group_by(am.col("region")).agg(am.agg.sum(am.col("amount")).alias("total"))
                .sort("total", descending=True).limit(5).collect())

    def best(source, iters=9):
        collect(source)                                # warm the imports and the pipeline cache
        out = float("inf")
        for _ in range(iters):
            t0 = time.perf_counter()
            collect(source)
            out = min(out, time.perf_counter() - t0)
        return out * 1000

    approx(rows(collect(table)), rows(collect(resident)))
    gpu, python = best(resident), best(table)
    assert python < gpu + 3.0, f"python path {python:.2f} ms vs resident path {gpu:.2f} ms"
