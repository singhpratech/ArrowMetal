"""The fused expression compiler from Python, against pyarrow.compute and Polars on the same data.

Run: PYTHONPATH=python python -m pytest python/tests/test_expr.py -q
"""
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import arrowmetal as am

try:
    import polars as pl
except ImportError:                                     # pragma: no cover
    pl = None

SIZES = [0, 1, 33, 4097, 100_003]


def rng(seed=7):
    return np.random.default_rng(seed)


def f64col(n, seed, nulls=0.1):
    r = rng(seed)
    v = r.standard_normal(n) * 100
    mask = r.random(n) < nulls if n else np.zeros(0, dtype=bool)
    return pa.array(v, mask=mask)


def i64col(n, seed, nulls=0.1, lo=-1000, hi=1000):
    r = rng(seed)
    v = r.integers(lo, hi, size=n, dtype=np.int64)
    mask = r.random(n) < nulls if n else np.zeros(0, dtype=bool)
    return pa.array(v, mask=mask)


def table(**cols):
    return pa.table(cols)


# ---------------------------------------------------------------------------- element-wise


@pytest.mark.parametrize("n", SIZES)
def test_six_operator_float64_expression_matches_pyarrow(n):
    a, b, c, d = (f64col(n, s) for s in (1, 2, 3, 4))
    t = table(a=a, b=b, c=c, d=d)
    e = (am.col("a") * 2 + am.col("b")) / (am.col("c") + 1) - am.col("d")
    got = am.query(t, e.alias("r").project())["r"]
    want = pc.subtract(pc.divide(pc.add(pc.multiply(a, pa.scalar(2.0)), b),
                                 pc.add(c, pa.scalar(1.0))), d)
    # Correctly rounded software binary64: the results are bit identical, not merely close.
    assert got.equals(want.cast(got.type))


@pytest.mark.parametrize("n", [0, 1, 33, 4097])
def test_integer_arithmetic_and_promotion(n):
    a = i64col(n, 11)
    b = pa.array(rng(12).integers(-50, 50, size=n, dtype=np.int32))
    t = table(a=a, b=b)
    r = am.query(t, am.filter(am.col("a") == am.col("a")).project(
        [(am.col("a") + am.col("b")).alias("add"),
         (am.col("a") * am.col("b")).alias("mul"),
         (am.col("a") - 1).alias("dec")]))
    keep = pc.is_valid(a)
    assert r["add"].equals(pc.filter(pc.add(a, b.cast(pa.int64())), keep))
    assert r["mul"].equals(pc.filter(pc.multiply(a, b.cast(pa.int64())), keep))
    assert r["dec"].equals(pc.filter(pc.add(a, pa.scalar(-1, pa.int64())), keep))


@pytest.mark.parametrize("n", [1, 33, 4097])
def test_comparison_and_kleene_logic(n):
    a, b = i64col(n, 21), i64col(n, 22)
    t = table(a=a, b=b)
    r = am.query(t, am.project([
        (am.col("a") > 0).alias("gt"),
        ((am.col("a") > 0) & (am.col("b") > 0)).alias("and"),
        ((am.col("a") > 0).and_kleene(am.col("b") > 0)).alias("andk"),
        ((am.col("a") > 0).or_kleene(am.col("b") > 0)).alias("ork"),
        am.col("a").is_null().alias("isnull"),
    ]))
    pa_gt, pb_gt = pc.greater(a, 0), pc.greater(b, 0)
    assert r["gt"].equals(pa_gt)
    assert r["and"].equals(pc.and_(pa_gt, pb_gt))
    assert r["andk"].equals(pc.and_kleene(pa_gt, pb_gt))
    assert r["ork"].equals(pc.or_kleene(pa_gt, pb_gt))
    assert r["isnull"].equals(pc.is_null(a))


@pytest.mark.parametrize("n", [1, 33, 4097])
def test_conditionals_and_null_functions(n):
    a, b = i64col(n, 31), i64col(n, 32)
    t = table(a=a, b=b)
    r = am.query(t, am.project([
        am.if_else(am.col("a") > 0, am.col("a"), am.col("b")).alias("ie"),
        am.coalesce_expr(am.col("a"), am.col("b"), am.lit(-1, "int64")).alias("co"),
        am.col("a").fill_null(am.lit(7, "int64")).alias("fn"),
        am.col("a").is_in([1, 2, 3]).alias("in"),
        am.col("a").is_valid().alias("iv"),
    ]))
    assert r["ie"].equals(pc.if_else(pc.greater(a, 0), a, b))
    assert r["co"].equals(pc.coalesce(a, b, pa.scalar(-1, pa.int64())))
    assert r["fn"].equals(pc.fill_null(a, pa.scalar(7, pa.int64())))
    assert r["in"].equals(pc.is_in(a, value_set=pa.array([1, 2, 3], pa.int64())))
    assert r["iv"].equals(pc.is_valid(a))


def test_math_unaries_and_round():
    v = pa.array([0.0, 0.5, 1.5, -0.5, -2.5, 4.0, 100.0, None], pa.float64())
    t = table(v=v)
    r = am.query(t, am.project([
        am.col("v").abs().alias("abs"),
        am.col("v").abs().sqrt().alias("sqrt"),
        am.col("v").round().alias("round"),
    ]))
    assert r["abs"].equals(pc.abs(v))
    assert r["round"].equals(pc.round(v, round_mode="half_towards_infinity"))
    for got, want in zip(r["sqrt"].to_pylist(), pc.sqrt(pc.abs(v)).to_pylist()):
        if want is None:
            assert got is None
        else:
            assert abs(got - want) <= 1e-9 * max(1.0, abs(want))


def test_casts():
    v = pa.array([1.7, -1.7, 0.0, 1e10, None], pa.float64())
    t = table(v=v)
    r = am.query(t, am.project([
        am.col("v").cast("int64").alias("i"),
        am.col("v").cast("float32").alias("f"),
        am.col("v").cast("int32").cast("float64").alias("rt"),
    ]))
    assert r["i"].equals(pc.cast(v, options=pc.CastOptions(pa.int64(), allow_float_truncate=True)))
    assert r["f"].equals(pc.cast(v, pa.float32()))


def test_string_predicates():
    s = pa.array(["north", "south", "northwest", None, "", "the north pole"])
    t = table(s=s)
    r = am.query(t, am.project([
        am.col("s").str_equals("north").alias("eq"),
        am.col("s").starts_with("north").alias("sw"),
        am.col("s").contains("orth").alias("ct"),
    ]))
    assert r["eq"].equals(pc.equal(s, pa.scalar("north")))
    assert r["sw"].equals(pc.starts_with(s, "north"))
    assert r["ct"].equals(pc.match_substring(s, "orth"))


# ---------------------------------------------------------------------------- terminals


@pytest.mark.parametrize("n", SIZES)
def test_filter_then_sum_matches_pyarrow_and_polars(n):
    region = pa.array(rng(41).integers(0, 5, size=n, dtype=np.int32))
    amount = f64col(n, 42)
    t = table(region=region, amount=amount)
    q = am.filter((am.col("region") == 2) & (am.col("amount") > 10)).sum(am.col("amount"))
    got = am.query(t, q)
    mask = pc.and_(pc.equal(region, 2), pc.greater(amount, 10))
    want = pc.sum(pc.filter(amount, pc.fill_null(mask, False))).as_py()
    if want is None:
        assert got is None
    else:
        assert got == pytest.approx(want, rel=1e-12, abs=1e-9)
    if pl is not None and n:
        df = pl.from_arrow(t).lazy()
        pw = df.filter((pl.col("region") == 2) & (pl.col("amount") > 10)) \
               .select(pl.col("amount").sum()).collect().item()
        assert (got or 0.0) == pytest.approx(pw, rel=1e-9, abs=1e-6)


@pytest.mark.parametrize("n", SIZES)
def test_filter_project_three_columns(n):
    a, b = i64col(n, 51), f64col(n, 52)
    c = pa.array(rng(53).integers(0, 10, size=n, dtype=np.int32))
    t = table(a=a, b=b, c=c)
    r = am.query(t, am.filter(am.col("c") < 5).project(
        [am.col("a"), (am.col("b") * 2).alias("b2"), am.col("c")]))
    keep = pc.less(c, 5)
    assert r["a"].equals(pc.filter(a, keep))
    assert r["c"].equals(pc.filter(c, keep))
    assert r["b2"].equals(pc.filter(pc.multiply(b, pa.scalar(2.0)), keep))


@pytest.mark.parametrize("n", [1, 33, 4097, 100_003])
def test_aggregates_match_pyarrow(n):
    a = i64col(n, 61)
    t = table(a=a)
    q = am.Query().aggregate([("sum", "s", am.col("a")), ("min", "mn", am.col("a")),
                              ("max", "mx", am.col("a")), ("count", "c", am.col("a")),
                              ("count", "rows", None), ("mean", "me", am.col("a"))])
    r = am.query(t, q)
    assert r["s"] == pc.sum(a).as_py()
    assert r["mn"] == pc.min(a).as_py()
    assert r["mx"] == pc.max(a).as_py()
    assert r["c"] == len(a) - a.null_count
    assert r["rows"] == n
    me = pc.mean(a).as_py()
    if me is None:
        assert r["me"] is None
    else:
        assert r["me"] == pytest.approx(me, rel=1e-12)


@pytest.mark.parametrize("n", [33, 4097, 100_003])
def test_group_by_matches_pyarrow(n):
    K = 64
    keys = pa.array(rng(71).integers(0, K, size=n, dtype=np.int32))
    vals = i64col(n, 72)
    t = table(k=keys, v=vals)
    r = am.query(t, am.group_by(am.col("k"), K).aggregate(
        [("sum", "s", am.col("v")), ("count", "c", am.col("v"))]))
    want = t.group_by("k").aggregate([("v", "sum"), ("v", "count")])
    want_sum = dict(zip(want.column("k").to_pylist(), want.column("v_sum").to_pylist()))
    want_cnt = dict(zip(want.column("k").to_pylist(), want.column("v_count").to_pylist()))
    got_s, got_c = r["s"].to_pylist(), r["c"].to_pylist()
    for k in range(K):
        assert got_s[k] == want_sum.get(k)
        assert got_c[k] == want_cnt.get(k, 0)


def test_group_by_sum_of_an_expression():
    n = 50_000
    K = 100
    keys = pa.array(rng(81).integers(0, K, size=n, dtype=np.int32))
    vals = pa.array(rng(82).integers(0, 100, size=n, dtype=np.int32))
    t = table(k=keys, v=vals)
    r = am.query(t, am.filter(am.col("v") > 10).group_by(am.col("k"), K).sum(am.col("v") * 3 + 1, "s"))
    keep = pc.greater(vals, 10)
    sub = pa.table({"k": pc.filter(keys, keep),
                    "e": pc.add(pc.multiply(pc.filter(vals, keep), 3), 1).cast(pa.int64())})
    want = sub.group_by("k").aggregate([("e", "sum")])
    wmap = dict(zip(want.column("k").to_pylist(), want.column("e_sum").to_pylist()))
    got = r["s"].to_pylist()
    for k in range(K):
        assert got[k] == wmap.get(k)


# ---------------------------------------------------------------------------- API surface


def test_repr_and_wire_format_round_trip():
    e = (am.col("amount") * 2 + am.col("fee")) > 100
    assert repr(e) == '(((col("amount") * 2) + col("fee")) > 100)'
    q = am.filter(e).sum(am.col("amount"), "total")
    assert q.sexpr() == am.query_canonical(q)
    assert "aggregate" in q.sexpr()


def test_metal_arrays_can_be_passed_directly():
    a = am.array(pa.array([1, 2, 3, None], pa.int64()))
    assert am.query({"a": a}, am.col("a").sum()) == 6


def test_errors_name_the_offending_node():
    t = table(a=pa.array([1, 2, 3], pa.int64()), s=pa.array(["x", "y", "z"]))
    with pytest.raises(am.ArrowMetalError, match="nope"):
        am.query(t, am.col("nope").sum())
    with pytest.raises(am.ArrowMetalError, match="utf8"):
        am.query(t, am.col("s").project())
    with pytest.raises(am.ArrowMetalError, match="boolean"):
        am.query(t, am.filter(am.col("a")).sum(am.col("a")))
    with pytest.raises(am.ArrowMetalError, match="unknown expression operator"):
        am.query_canonical('(query (aggregate (sum "s" (frobnicate (col "a")))))')


def test_zero_rows():
    t = table(a=pa.array([], pa.int64()))
    assert am.query(t, am.col("a").sum()) is None
    assert am.query(t, am.Query().count()) == 0
    assert len(am.query(t, am.filter(am.col("a") > 0).project([am.col("a")]))["a"]) == 0
