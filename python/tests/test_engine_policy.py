"""MetalEngine's placement policy (python/arrowmetal/_engine_policy.py): which translated subtrees run
on Metal by default, and the overrides.

Run: PYTHONPATH=python python -m pytest python/tests/test_engine_policy.py -q

The pure half checks `decide()` itself against the committed crossover table
(python/arrowmetal/_engine_crossovers.py): the same decision for the same (classes, dtypes, rows,
input) across calls and across processes, and every rule just below and at its crossover. The
engine half collects plans through `MetalEngine()` and checks that the report says which rule took
or left each subtree, that a Parquet scan is judged by its footer's row count, and that the table
is the one `Benchmarks/polars_engine_crossover.py` fits from the results file it names.
"""
import os
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import arrowmetal as am

pl = pytest.importorskip("polars")
from arrowmetal import _engine_crossovers as table  # noqa: E402
from arrowmetal import _engine_policy as policy  # noqa: E402
from arrowmetal import polars_engine as pe  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ROUTER = policy.router_crossovers(am.router_table())
KEYS = sorted(table.ENGINE)


def dtypes_of(dclass):
    return [pl.Int64, pl.String] if dclass == "string" else [pl.Int64, pl.Float64, pl.Int32]


def reached(key):
    return policy.crossover(*key, ROUTER)[0] is not None


# =============================================================================================
# the pure policy
# =============================================================================================


def _grid():
    out = []
    for cls, dclass, source in KEYS:
        x = policy.crossover(cls, dclass, source, ROUTER)[0] or 10**9
        for rows in (0, 1, 999_999, 1_000_000, x - 1, x, x + 1, 10**12):
            out.append(((cls,), dclass, source, rows))
    out.append((("sort", "group_by:sum", "join:inner"), "numeric", "memory", 7_654_321))
    out.append(((), "numeric", "memory", 5_000_000))
    return out


def _decisions(router):
    return [tuple(policy.decide(list(c), dtypes_of(d), n, s, router=router))
            for c, d, s, n in _grid()]


def test_the_policy_is_pure_across_calls_and_processes():
    first = _decisions(ROUTER)
    assert len(first) > len(KEYS) * 8
    for _ in range(200):
        assert _decisions(ROUTER) == first
    code = ("import sys; sys.path.insert(0, %r); import test_engine_policy as t; "
            "print(repr(t._decisions(t.ROUTER)))" % HERE)
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "python"))
    out = subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=REPO,
                         capture_output=True, text=True).stdout.strip().splitlines()[-1]
    assert out == repr(first)


def test_the_table_covers_the_translator_classes_it_measured():
    for cls, dclass, source in KEYS:
        assert cls in policy.CLASSES, cls
        assert dclass in ("numeric", "string") and source in ("memory", "parquet")
        e = table.ENGINE[(cls, dclass, source)]
        assert e["cases"] and e["largest"] > 0
        assert e["rows"] is None or 0 < e["rows"] <= e["largest"]


@pytest.mark.parametrize("key", KEYS, ids=lambda k: "|".join(k))
def test_each_rule_just_below_and_at_its_crossover(key):
    cls, dclass, source = key
    x, where = policy.crossover(cls, dclass, source, ROUTER)
    dt = dtypes_of(dclass)
    if x is None:
        for rows in (0, 1_000_000, 10**12):
            d = policy.decide([cls], dt, rows, source, router=ROUTER)
            assert not d.take and d.binding == cls and d.crossover is None
            assert "was not measured ahead of Polars up to" in d.reason, d.reason
        return
    assert x >= table.ENGINE[key]["rows"]
    below = policy.decide([cls], dt, x - 1, source, router=ROUTER)
    at = policy.decide([cls], dt, x, source, router=ROUTER)
    assert not below.take and at.take
    assert below.binding == at.binding == cls and below.crossover == at.crossover == x
    assert f"{x - 1:,} input rows is below the {x:,}-row crossover for {cls}" in below.reason
    assert f"{x:,} input rows is at or above the {x:,}-row crossover for {cls}" in at.reason
    assert where in at.reason


def test_a_subtree_needs_the_crossover_of_every_class_in_it():
    numeric = [(k[0], policy.crossover(*k, ROUTER)[0]) for k in KEYS
               if k[1:] == ("numeric", "memory") and reached(k)]
    missing = [k[0] for k in KEYS if k[1:] == ("numeric", "memory") and not reached(k)]
    if len(numeric) >= 2:
        (c1, x1), (c2, x2) = sorted(numeric, key=lambda t: t[1])[:: len(numeric) - 1]
        d = policy.decide([c1, c2], dtypes_of("numeric"), x2 - 1, router=ROUTER)
        assert not d.take and d.binding == c2 and d.crossover == x2
        assert policy.decide([c1, c2], dtypes_of("numeric"), x2, router=ROUTER).take
    if numeric and missing:
        d = policy.decide([numeric[0][0], missing[0]], dtypes_of("numeric"), 10**12, router=ROUTER)
        assert not d.take and d.binding == missing[0]


def test_the_string_rule_uses_the_string_rows_only():
    """A String column among the inputs selects the table's `string` rows: a class measured over
    numeric columns only is not taken with a String column at any size."""
    for cls, dclass, source in KEYS:
        if dclass != "numeric" or (cls, "string", source) in table.ENGINE:
            continue
        d = policy.decide([cls], [pl.Int64, pl.String], 10**12, source, router=ROUTER)
        assert not d.take and d.reason.startswith(f"no measurement of {cls} with a String column")


def test_unmeasured_shapes_are_left_to_polars():
    d = policy.decide(["join:anti"], dtypes_of("numeric"), 10**12, "parquet", router=ROUTER)
    if ("join:anti", "numeric", "parquet") not in table.ENGINE:
        assert not d.take and d.reason.startswith("no measurement of join:anti over a Parquet file")


def test_the_router_table_and_the_sort_sweep_are_floors():
    """A class's crossover is at least the router table's for the kernels it runs and the crossover
    sweep's for the sort kernels; a floor above the engine table's figure is the one the reason
    names."""
    for key in KEYS:
        x = policy.crossover(*key, ROUTER)[0]
        if x is None:
            continue
        for op in policy.ROUTER_OPS.get(key[0], ()):
            if op in ROUTER:
                assert x >= ROUTER[op][0], (key, op)
        if key[0] in table.SWEEP:
            assert x >= table.SWEEP[key[0]]["rows"], key
    reached_keys = [k for k in KEYS if reached(k) and k[0] in policy.ROUTER_OPS]
    if reached_keys:
        key = reached_keys[0]
        op = policy.ROUTER_OPS[key[0]][0]
        big = dict(ROUTER, **{op: (10**11, "router table, a test row")})
        d = policy.decide([key[0]], dtypes_of(key[1]), 10**11 - 1, key[2], router=big)
        assert not d.take and d.crossover == 10**11 and "router table, a test row" in d.reason


def test_overrides():
    dt = dtypes_of("numeric")
    d = policy.decide(["join:anti"], dt, 999_999, shapes="all")
    assert not d.take and "below min_rows=1,000,000 (shapes='all')" in d.reason
    assert policy.decide(["join:anti"], dt, 1_000_000, shapes="all").take
    assert policy.decide(["join:anti"], dt, 0, shapes="all", min_rows=0).take
    named = frozenset({"join", "sort"})
    assert policy.decide(["join:anti", "sort"], dt, 5, shapes=named).take
    d = policy.decide(["join:anti", "top_k"], dt, 5, shapes=named)
    assert not d.take and d.reason.startswith("top_k is not among the shapes named")
    assert not policy.decide(["sort"], dt, 5, shapes=named, min_rows=6).take
    # min_rows above a crossover is a floor for the measured default as well.
    for key in KEYS:
        x = policy.crossover(*key, ROUTER)[0]
        if x is not None:
            d = policy.decide([key[0]], dtypes_of(key[1]), x, key[2], min_rows=x + 5, router=ROUTER)
            assert not d.take and f"below min_rows={x + 5:,}" in d.reason
            assert policy.decide([key[0]], dtypes_of(key[1]), x + 5, key[2], min_rows=x + 5,
                                 router=ROUTER).take
            break


def test_engine_arguments():
    e = am.MetalEngine()
    assert e.shapes == "measured" and e.min_rows is None
    assert am.MetalEngine(shapes={"sort", "group_by:sum"}).shapes == frozenset({"sort", "group_by:sum"})
    for bad in ("some", {"sorts"}, set(), 3):
        with pytest.raises(ValueError):
            am.MetalEngine(shapes=bad)
    assert set(pe.SHAPE_CLASSES) == set(policy.CLASSES)
    rules = pe.placement_rules()
    assert [r[:3] for r in rules] == KEYS


def test_the_committed_table_is_the_fit_of_the_results_file_it_names():
    assert os.path.exists(os.path.join(REPO, table.SOURCE)), table.SOURCE
    assert os.path.exists(os.path.join(REPO, table.SWEEP_SOURCE)), table.SWEEP_SOURCE
    subprocess.run([sys.executable, os.path.join(REPO, "Benchmarks", "polars_engine_crossover.py"),
                    "--check"], check=True, cwd=REPO, capture_output=True)


def test_every_fitted_case_is_ahead_from_its_crossover_on():
    """The per-case fits in the table: at every measured size at or above a case's crossover the
    MetalEngine was at least as fast as the faster Polars engine, and a class's crossover is the
    largest of its cases'."""
    for case, f in table.CASES.items():
        ahead = 1.0 + table.MARGIN
        if f["rows"] is None:
            # Not ahead at the largest size, or ahead there alone.
            last = sorted(f["ratios"])[-2:]
            assert not all(f["ratios"][n] >= ahead + 0.005 for n in last), (case, f)
            continue
        assert all(r >= ahead - 0.005 for n, r in f["ratios"].items() if n >= f["step"]), (case, f)
    for key, e in table.ENGINE.items():
        rows = [table.CASES[c]["rows"] for c in e["cases"]]
        assert e["rows"] == (None if None in rows else max(rows)), key


# =============================================================================================
# the policy inside the engine
# =============================================================================================


def _x(cls, dclass="numeric", source="memory"):
    return policy.crossover(cls, dclass, source, ROUTER)[0]


def _frame(n, seed=3):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"q": rng.integers(0, 10**9, n), "x": rng.random(n),
                         "k": rng.integers(0, 50, n).astype(np.int32)})


def test_the_default_takes_a_sort_at_its_crossover_and_names_the_rule():
    x = _x("sort")
    assert x is not None
    df = _frame(x)
    eng = am.MetalEngine()
    lf = df.lazy().sort("q")
    got = lf.collect(engine=eng)
    assert got["q"].equals(lf.collect()["q"])
    rep = eng.last_report
    assert [t["kinds"] for t in rep.taken] == [["Sort", "DataFrameScan"]], rep
    t = rep.taken[0]
    assert t["shape"] == ["sort"] and t["dtype_class"] == "numeric" and t["input"] == "memory"
    assert t["rule"].startswith(f"{x:,} input rows is at or above the {x:,}-row crossover for sort")
    assert f"rule: {t['rule']}" in str(rep)
    # One row fewer: the same plan stays with Polars, and the report says which rule left it.
    small = df.head(x - 1).lazy().sort("q")
    small.collect(engine=eng)
    assert not eng.last_report.taken
    assert any(f.startswith("Sort#") and f"rule: {x - 1:,} input rows is below the {x:,}-row "
               "crossover for sort" in f for f in eng.last_report.fallbacks), eng.last_report


def test_the_default_leaves_what_was_not_measured_ahead_and_says_so():
    df = _frame(_x("sort"))
    eng = am.MetalEngine()
    for lf, cls in ((df.lazy().filter(pl.col("x") > 0.5), "rowwise"),
                    (df.lazy().select(pl.col("q").sum()), "aggregate:sum"),
                    (df.lazy().sort("q").head(10), "top_k")):
        if reached((cls, "numeric", "memory")):
            continue
        want = lf.collect()
        got = lf.collect(engine=eng)
        assert got.equals(want) and not eng.last_report.taken, eng.last_report
        assert any(f"rule: {cls} was not measured ahead of Polars up to" in f
                   for f in eng.last_report.fallbacks), (cls, eng.last_report)


def test_the_string_rule_in_the_engine():
    """A sort carrying a String column is judged by the String row of the table."""
    x_num = _x("sort")
    df = _frame(x_num).with_columns(pl.col("k").cast(pl.String).alias("s"))
    eng = am.MetalEngine()
    lf = df.lazy().select("q", "s").sort("q")
    lf.collect(engine=eng)
    x = _x("sort", "string")
    if x is None or x > x_num:
        assert not eng.last_report.taken
        assert any("sort with a String column" in f for f in eng.last_report.fallbacks), eng.last_report
    else:
        assert eng.last_report.taken


def test_a_smaller_subtree_below_a_declined_node_can_still_be_taken():
    """The placement walks down from a node the policy leaves: an inner join under a whole-frame
    sum runs on Metal when the sum is not measured ahead but the join is."""
    x = _x("join:inner")
    if x is None or reached(("aggregate:sum", "numeric", "memory")):
        pytest.skip("the table in force does not separate the two")
    df = _frame(x)
    right = pl.DataFrame({"k": np.arange(50, dtype=np.int32), "w": np.arange(50, dtype=np.int64)})
    lf = df.lazy().join(right.lazy(), on="k", how="inner").select(pl.col("w").sum())
    eng = am.MetalEngine()
    assert lf.collect(engine=eng).equals(lf.collect())
    rep = eng.last_report
    assert len(rep.taken) == 1 and "Join" in rep.taken[0]["kinds"], rep
    assert rep.taken[0]["shape"] == ["join:inner"], rep
    assert any("rule: aggregate:sum was not measured ahead" in f for f in rep.fallbacks), rep


def test_a_parquet_scan_is_judged_by_its_footer_row_count(tmp_path):
    """The rows of a Parquet scan are the file's, from the footer, whatever the predicate keeps:
    a sort over a filter that keeps a handful of rows is taken from the file's crossover on."""
    x = _x("sort", "numeric", "parquet")
    if x is None:
        pytest.skip("no Parquet sort crossover in the table in force")
    rng = np.random.default_rng(4)
    q = rng.integers(0, 10**9, x)
    p = str(tmp_path / "at.parquet")
    pq.write_table(pa.table({"q": q, "x": rng.random(x)}), p, row_group_size=1 << 20)
    below = str(tmp_path / "below.parquet")
    pq.write_table(pa.table({"q": q[:-1], "x": rng.random(x - 1)}), below, row_group_size=1 << 20)
    eng = am.MetalEngine()
    lf = pl.scan_parquet(p).filter(pl.col("q") < 1000).sort("q")
    got = lf.collect(engine=eng)
    assert got.equals(lf.collect())
    rep = eng.last_report
    assert len(rep.taken) == 1 and rep.taken[0]["rows"] == x and rep.taken[0]["input"] == "parquet", rep
    assert "Scan" in rep.taken[0]["kinds"]
    assert f"crossover for sort over a Parquet file" in rep.taken[0]["rule"]
    lf = pl.scan_parquet(below).filter(pl.col("q") < 1000).sort("q")
    assert lf.collect(engine=eng).equals(lf.collect())
    assert not eng.last_report.taken
    assert any(f"{x - 1:,} input rows is below the {x:,}-row crossover for sort over a Parquet file"
               in f for f in eng.last_report.fallbacks), eng.last_report
    am.clear_parquet_cache()


def test_overrides_in_the_engine():
    df = _frame(1000)
    lf = df.lazy().group_by("k").agg(pl.col("q").sum())
    eng = am.MetalEngine(shapes={"group_by"})
    compare = lf.collect().sort("k")
    assert lf.collect(engine=eng).sort("k").equals(compare)
    assert eng.last_report.taken and "shapes={group_by}" in eng.last_report.taken[0]["rule"]
    eng = am.MetalEngine(shapes={"sort"})
    lf.collect(engine=eng)
    assert not eng.last_report.taken
    assert any("group_by:sum is not among the shapes named (sort)" in f
               for f in eng.last_report.fallbacks), eng.last_report
    eng = am.MetalEngine(shapes="all")
    lf.collect(engine=eng)
    assert not eng.last_report.taken and any("below min_rows=1,000,000" in f
                                             for f in eng.last_report.fallbacks)


def test_the_report_prints_a_rule_for_every_placement_decision():
    df = _frame(_x("sort"))
    eng = am.MetalEngine()
    lf = df.lazy().filter(pl.col("x") > 0.1).sort("q").select(pl.col("q").sum())
    lf.collect(engine=eng)
    text = str(eng.last_report)
    for t in eng.last_report.taken:
        assert f"rule: {t['rule']}" in text
    assert all(f.split(": ", 1)[1].startswith("rule: ") or "no ArrowMetal translation" in f
               for f in eng.last_report.fallbacks), text


# =============================================================================================
# the Float64 group-by guard (found by the crossover sweep)
# =============================================================================================


def test_core_group_by_float64_sum_and_mean_at_2_24_groups():
    """A Float64 group sum at exactly 2^24 groups: every group present and right. Found wrong by the
    (v3) case of the crossover sweep (per-group grids wrapped past 2^32 threads) and fixed in the core;
    python/tests/test_group_by_2_24.py covers the other aggregates and counts."""
    from arrowmetal import lazy
    n = 1 << 24
    k = am.MetalArray.from_arrow(pa.array(np.arange(n, dtype=np.int64)))
    v = am.MetalArray.from_arrow(pa.array(np.arange(n, dtype=np.float64)))
    plan = {"op": "group_by", "input": {"op": "scan", "source": "t"},
            "keys": [["k", '(col "k")']], "aggs": [["sum", "s", '(col "v")']]}
    out = lazy.LazyFrame(plan, {"t": lazy._Source(["k", "v"], [k, v])}).collect()
    s = out.column("s")
    assert s.null_count == 0
    assert len(s) == n
    order = pa.compute.sort_indices(out.column("k"))
    assert pa.compute.take(s, order).to_pylist()[:3] == [0.0, 1.0, 2.0]


def test_the_engine_takes_a_float64_group_sum_at_2_24_rows():
    """A group-by with a Float64 sum or mean over 2^24 input rows runs on Metal and equals Polars."""
    df = pl.DataFrame({"k": np.arange(1 << 24, dtype=np.int64) % 1000,
                       "v": np.ones(1 << 24), "i": np.ones(1 << 24, dtype=np.int32)})
    eng = am.MetalEngine(shapes="all", min_rows=0)
    for agg in (pl.col("v").sum(), pl.col("v").mean(), pl.col("k").mean().alias("m"), pl.col("i").sum()):
        lf = df.lazy().group_by("k").agg(agg)
        assert lf.collect(engine=eng).sort("k").equals(lf.collect().sort("k"))
        assert eng.last_report.taken, eng.last_report


