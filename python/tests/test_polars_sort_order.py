"""Polars' sort order, as the installed Polars sorts, and MetalEngine's translation of it.

The first half pins Polars' own rules, so an upgrade that moved one fails here first (they hold for
polars 1.44 and the 2.0 release candidate alike):

* `nulls_last` is per key and independent of the direction; its default, False, puts the nulls
  first in both directions.
* A float key orders NaN as one value above every number, +inf included, in both directions: last
  ascending, first descending (before the values, after the nulls unless `nulls_last`). A NaN with
  its sign bit set is the same NaN.
* -0.0 and +0.0 are equal: they tie, a tie keeps input order under `maintain_order=True`, and a
  multi-key sort falls through to the next key.
* A descending sort is not the reverse of an ascending one: ties keep their input order in both
  directions.
* `top_k(k, by=)` is a descending sort with `nulls_last=True` and a slice, `bottom_k` the ascending
  one; `sort(...).head(k)` keeps the sort's own `nulls_last`.
* Strings order byte-wise (UTF-8), Booleans False before True; integer and temporal keys by value.

The second half checks that MetalEngine gives Polars' answer for every combination, and that it does
so with one ArrowMetal key per Polars key: the null placement and the float order are options of the
key (`{"nulls": "first"}`, `{"float_order": "nan_largest"}`), no extra key column is computed, and a
single-key sort with a slice stays a top-k.
"""
import datetime
import json
import math
import struct

import numpy as np
import pytest

from arrowmetal import polars_engine as pe

pl = pytest.importorskip("polars")
from polars.testing import assert_frame_equal  # noqa: E402

NAN = float("nan")
NNAN = struct.unpack("<d", struct.pack("<Q", 0xFFF8000000000000))[0]
X = [1.0, NAN, None, -0.0, 0.0, -1.0, math.inf, -math.inf, NNAN, None, 2.0, -0.0]


def _order(df, *a, **k):
    return df.with_row_index("i").sort(*a, maintain_order=True, **k)["i"].to_list()


# ---------------------------------------------------------------------------------------------
# Polars' rules


def test_float_order_nan_largest_nulls_per_key():
    df = pl.DataFrame({"x": X}, schema={"x": pl.Float64})
    assert _order(df, "x") == [2, 9, 7, 5, 3, 4, 11, 0, 10, 6, 1, 8]
    assert _order(df, "x", nulls_last=True) == [7, 5, 3, 4, 11, 0, 10, 6, 1, 8, 2, 9]
    assert _order(df, "x", descending=True) == [2, 9, 1, 8, 6, 10, 0, 3, 4, 11, 5, 7]
    assert _order(df, "x", descending=True, nulls_last=True) == [1, 8, 6, 10, 0, 3, 4, 11, 5, 7, 2, 9]
    f32 = pl.DataFrame({"x": X}, schema={"x": pl.Float32})
    assert _order(f32, "x", descending=True) == [2, 9, 1, 8, 6, 10, 0, 3, 4, 11, 5, 7]


def test_zeros_and_nans_tie_and_fall_through_to_the_next_key():
    df = pl.DataFrame({"x": [0.0, -0.0, 0.0, -0.0], "y": [3, 2, 1, 0]})
    assert df.sort(["x", "y"])["y"].to_list() == [0, 1, 2, 3]
    assert df.sort(["x", "y"], descending=[True, False])["y"].to_list() == [0, 1, 2, 3]
    nans = pl.DataFrame({"x": [NAN, NNAN, NAN, NNAN], "y": [3, 2, 1, 0]})
    assert nans.sort(["x", "y"])["y"].to_list() == [0, 1, 2, 3]


def test_descending_keeps_ties_in_input_order():
    df = pl.DataFrame({"k": [1, 2, 1, 2, 1]})
    assert _order(df, "k", descending=True) == [1, 3, 0, 2, 4]


def test_top_k_bottom_k_and_head():
    df = pl.DataFrame({"x": X, "i": list(range(len(X)))}, schema={"x": pl.Float64, "i": pl.Int64})
    assert set(df.top_k(4, by="x")["i"].to_list()) == {1, 8, 6, 10}       # NaN, NaN, inf, 2.0
    assert df.bottom_k(4, by="x")["x"].to_list()[:2] == [-math.inf, -1.0]
    assert df.lazy().sort("x", descending=True).head(2).collect()["x"].to_list() == [None, None]
    plan = df.lazy().top_k(4, by="x").explain()
    assert "descending: [true]" in plan and "nulls_last: [true]" in plan
    plan = df.lazy().bottom_k(4, by="x").explain()
    assert "descending" not in plan and "nulls_last: [true]" in plan


def test_strings_bytewise_booleans_false_first():
    s = pl.DataFrame({"s": ["b", "a", None, "", "B", "ä", "a\x00", "ab"]})
    assert s.sort("s")["s"].to_list() == [None, "", "B", "a", "a\x00", "ab", "b", "ä"]
    b = pl.DataFrame({"b": [True, None, False, True]})
    assert b.sort("b")["b"].to_list() == [None, False, True, True]
    assert b.sort("b", descending=True, nulls_last=True)["b"].to_list() == [True, True, False, None]


# ---------------------------------------------------------------------------------------------
# MetalEngine


def metal():
    return pe.MetalEngine(raise_on_fail=True, min_rows=0, shapes="all")


def _frame(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    special = rng.random(n) < 0.2
    x[special] = np.array([NAN, NNAN, 0.0, -0.0, math.inf, -math.inf])[rng.integers(0, 6, special.sum())]
    ts = [datetime.datetime(2026, 1, 1) + datetime.timedelta(seconds=int(v))
          for v in rng.integers(0, 50, n)]
    df = pl.DataFrame({
        "x": x, "x32": x.astype(np.float32), "k": rng.integers(-3, 3, n).astype(np.int32),
        "u": rng.integers(0, 4, n).astype(np.uint64), "s": [f"v{v}" for v in rng.integers(0, 30, n)],
        "b": rng.random(n) < 0.5, "t": ts, "d": [t.date() for t in ts], "i": np.arange(n)})
    nulls = rng.random((n, 8)) < 0.1
    return df.with_columns([pl.when(pl.lit(pl.Series(nulls[:, j]))).then(None).otherwise(pl.col(c)).alias(c)
                            for j, c in enumerate(["x", "x32", "k", "u", "s", "b", "t", "d"])])


KEYS = ["x", "x32", "k", "u", "s", "b", "t", "d"]


def _plan(eng):
    return json.loads(eng.last_report.taken[0]["plan"])


def _sort_node(plan):
    while plan.get("op") != "sort":
        plan = plan["input"]
    return plan


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
def test_one_key_sorts_in_polars_order_with_one_plan_key(key, descending, nulls_last):
    df = _frame(3_001, seed=len(key))
    lf = df.lazy().sort(key, descending=descending, nulls_last=nulls_last, maintain_order=True)
    eng = metal()
    got = lf.collect(engine=eng)
    assert_frame_equal(got, lf.collect())
    plan = _plan(eng)
    by = _sort_node(plan)["by"]
    assert len(by) == 1 and by[0][0] == key and by[0][1] is descending
    opts = by[0][2] if len(by[0]) > 2 else {}
    assert opts.get("nulls") == (None if nulls_last else "first")
    assert opts.get("float_order") == ("nan_largest" if key in ("x", "x32") else None)
    assert "__arrowmetal_" not in json.dumps(plan)
    assert [t["shape"] for t in eng.last_report.taken] == [["sort"]]


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
@pytest.mark.parametrize("k", [1, 10, 500])
def test_top_k_stays_one_key(descending, nulls_last, k):
    df = _frame(20_011, seed=k)
    for key in ("x", "x32", "k", "t"):
        lf = df.lazy().sort(key, descending=descending, nulls_last=nulls_last).head(k)
        eng = metal()
        got = lf.collect(engine=eng)
        want = lf.collect()
        assert_frame_equal(got.select(key), want.select(key))
        plan = _plan(eng)
        assert plan["op"] == "limit" and len(_sort_node(plan)["by"]) == 1
        assert [t["shape"] for t in eng.last_report.taken] == [["top_k"]]


def test_top_k_and_bottom_k():
    df = _frame(20_011, seed=4)
    for lf, by in ((df.lazy().top_k(25, by="x"), ["x"]), (df.lazy().bottom_k(25, by="x"), ["x"]),
                   (df.lazy().top_k(25, by=["k", "x"]), ["k", "x"]), (df.lazy().bottom_k(7, by="t"), ["t"])):
        eng = metal()
        got, want = lf.collect(engine=eng), lf.collect()
        assert_frame_equal(got.select(by), want.select(by))
        assert "__arrowmetal_" not in eng.last_report.taken[0]["plan"]


@pytest.mark.parametrize("seed", range(6))
def test_multi_key_per_key_options(seed):
    rng = np.random.default_rng(seed)
    df = _frame(5_003, seed=seed)
    keys = list(rng.choice(KEYS, size=int(rng.integers(2, 4)), replace=False))
    desc = [bool(v) for v in rng.integers(0, 2, len(keys))]
    nl = [bool(v) for v in rng.integers(0, 2, len(keys))]
    lf = df.lazy().sort(keys, descending=desc, nulls_last=nl, maintain_order=True)
    eng = metal()
    assert_frame_equal(lf.collect(engine=eng), lf.collect())
    assert len(_sort_node(_plan(eng))["by"]) == len(keys)
    # With a slice: a multi-key top-k.
    lf = df.lazy().sort(keys, descending=desc, nulls_last=nl).head(40)
    eng = metal()
    assert_frame_equal(lf.collect(engine=eng).select(keys), lf.collect().select(keys))


def test_nullable_temporal_key_nulls_first_over_a_join_and_a_group_by():
    """A nullable temporal key sorted nulls first used to need a validity key taken from the scan,
    which ruled out anything but one in-memory frame below the sort; the key option has no such limit."""
    df = _frame(4_001, seed=9)
    right = pl.DataFrame({"k": np.arange(-3, 3, dtype=np.int32), "w": np.arange(6)})
    lf = df.lazy().join(right.lazy(), on="k", how="left").sort("t", descending=True)
    eng = metal()
    got = lf.collect(engine=eng)
    assert_frame_equal(got.select("t"), lf.collect().select("t"))
    assert "Sort" in eng.last_report.taken[0]["kinds"]
    lf = df.lazy().group_by("t").agg(pl.len().alias("n")).sort("t")
    eng = metal()
    assert_frame_equal(lf.collect(engine=eng).select("t"), lf.collect().select("t"))
    assert "Sort" in eng.last_report.taken[0]["kinds"]


def test_parquet_nullable_temporal_key_nulls_first(tmp_path):
    df = _frame(3_001, seed=12)
    path = tmp_path / "t.parquet"
    df.write_parquet(path)
    lf = pl.scan_parquet(path).select("t", "i").sort("t", descending=True)
    eng = metal()
    assert_frame_equal(lf.collect(engine=eng).select("t"), lf.collect().select("t"))
    assert '"nulls": "first"' in eng.last_report.taken[0]["plan"]
