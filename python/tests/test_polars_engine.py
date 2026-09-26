"""MetalEngine (tier 4 of docs/POLARS.md) against Polars' own engine, on the same LazyFrames.

Run: PYTHONPATH=python python -m pytest python/tests/test_polars_engine.py -q
     DIFF_QUICK=1 ... drops the 100,003-row datasets.

Every differential case collects one LazyFrame twice -- `lf.collect()` (Polars, the oracle) and
`lf.collect(engine=MetalEngine(raise_on_fail=True, min_rows=0, shapes="all"))` -- checks from
`engine.last_report` that the intended nodes ran on Metal, and compares the two frames: schema first,
then values, exactly for integers, booleans, strings, nulls and element-wise floats, and with a
relative tolerance for float aggregates, whose summation order differs. Rows are compared as a
multiset unless the plan fixes an order; a sort is compared on its keys row by row plus the rows as
a multiset, since rows that tie on every key come back in an unspecified order from both engines.

The inputs are the generators of `test_differential.py` (`make_array`, its sizes, null ratios and
"sliced"/"special" flavours), plus two Polars-side shapes: a two-chunk frame (`pl.concat(...,
rechunk=False)`) and a frame sliced with `DataFrame.slice`.
"""
import json
import os
import re
import subprocess
import sys
import warnings

import numpy as np
import pyarrow as pa
import pytest

import arrowmetal as am

pl = pytest.importorskip("polars")
from polars.testing import assert_frame_equal  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_differential as td  # noqa: E402

from arrowmetal import polars_engine as pe  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
QUICK = os.environ.get("DIFF_QUICK") == "1"

INTS = td.INTEGER
FLOATS = td.FLOATING
NUMERIC = td.NUMERIC
PL_DTYPE = {"int8": pl.Int8, "int16": pl.Int16, "int32": pl.Int32, "int64": pl.Int64,
            "uint8": pl.UInt8, "uint16": pl.UInt16, "uint32": pl.UInt32, "uint64": pl.UInt64,
            "float32": pl.Float32, "float64": pl.Float64, "bool": pl.Boolean}


# ---------------------------------------------------------------------------------------------
# inputs


def _shapes():
    sizes = [0, 1, 33, 4097] + ([] if QUICK else [100_003])
    out = []
    for n in sizes:
        for nr in ([0.0, 0.3] if n > 10_000 else td.NULL_RATIOS):
            out.append(("random", n, nr))
    out += [("sliced", 4097, 0.3), ("special", 1000, 0.0), ("special", 1000, 0.3),
            ("chunked", 4097, 0.3), ("pl_sliced", 4097, 0.3)]
    return out


SHAPES = _shapes()
SHAPE_IDS = [f"{k}-{n}-{nr:g}" for k, n, nr in SHAPES]
# A narrower set for the per-operator matrices that would otherwise run thousands of cases.
CORE = [s for s in SHAPES if s[1] in (0, 33, 4097) or s[0] != "random"]
CORE_IDS = [f"{k}-{n}-{nr:g}" for k, n, nr in CORE]


def _arrow_frame(dtype, n, nr, flavor, seed):
    offset = 5 if flavor == "sliced" and n else 0
    shape = td.Shape(n, nr, "random" if flavor in ("chunked", "pl_sliced") else flavor, offset)
    cols = {
        "a": td.make_array(dtype, shape, seed=seed + 1),
        "b": td.make_array(dtype, shape, seed=seed + 2),
        "s": td.make_array("utf8", shape, seed=seed + 3),
        "g": td.make_array("bool", shape, seed=seed + 4),
    }
    rng = np.random.default_rng([seed, n, int(nr * 10)])
    keys = rng.integers(0, 9, n).astype(np.int32)
    mask = rng.random(n) < min(nr, 0.2) if n else None
    cols["k"] = pa.array(keys, mask=mask, type=pa.int32())
    return pa.table(cols)


def frame(dtype, shape, seed=0):
    """A Polars DataFrame with columns a, b (`dtype`), s (String), g (Boolean) and k (Int32 keys
    in 0..8 with up to 20% nulls), in the given shape."""
    flavor, n, nr = shape
    if flavor == "chunked":
        half = n // 2
        a = pl.from_arrow(_arrow_frame(dtype, half, nr, flavor, seed))
        b = pl.from_arrow(_arrow_frame(dtype, n - half, nr, flavor, seed + 7))
        df = pl.concat([a, b], rechunk=False)
        assert df.n_chunks() == 2 or n < 2
        return df
    if flavor == "pl_sliced":
        df = pl.from_arrow(_arrow_frame(dtype, n + 2, nr, flavor, seed))
        return df.slice(1, n)
    return pl.from_arrow(_arrow_frame(dtype, n, nr, flavor, seed))


# ---------------------------------------------------------------------------------------------
# the harness


def metal():
    return pe.MetalEngine(raise_on_fail=True, min_rows=0, shapes="all")


def _sorted(df):
    return df.sort(pl.all(), nulls_last=True) if df.width and df.height else df


def compare(got, want, order=None, rel=None):
    """`order`: None (rows as a multiset), True (exact order), or a list of sort keys (keys row by
    row, then the rows as a multiset). `rel`: relative tolerance for float columns, else exact."""
    assert got.schema == want.schema, f"schema {got.schema} != Polars {want.schema}"
    kw = {"check_exact": True} if rel is None else {"check_exact": False, "rel_tol": rel,
                                                     "abs_tol": 1e-12}
    if isinstance(order, list):
        assert_frame_equal(got.select(order), want.select(order), **kw)
        order = None
    if order is None:
        got, want = _sorted(got), _sorted(want)
    assert_frame_equal(got, want, **kw)


def check(lf, *, kinds=(), order=None, rel=None, keys_only=None):
    """Collects `lf` on Polars and on Metal (everything must run there: `raise_on_fail=True`),
    and compares. Returns the engine, for its report.

    Polars folds some plans over an empty frame down to the scan itself; then there is nothing to
    run anywhere, and the report must show exactly that. `keys_only` compares just those columns,
    in order: a sort with a slice keeps an unspecified choice of the rows that tie at the cut."""
    want = lf.collect()
    eng = metal()
    got = lf.collect(engine=eng)
    rep = eng.last_report
    if rep.taken:
        taken = rep.kinds_taken()
        for k in kinds:
            assert k in taken, f"{k} did not run on Metal:\n{rep}"
    else:
        walked = {k for _n, k in rep.walked}
        assert walked <= {"DataFrameScan", "Scan", "SimpleProjection", "Slice"}, str(rep)
    if keys_only:
        assert got.schema == want.schema
        compare(got.select(keys_only), want.select(keys_only), True, rel)
    else:
        compare(got, want, order, rel)
    return eng


def check_fallback(lf, reason, *, order=None, rel=None):
    """`lf` still collects identically, and the report names `reason` for what stayed on Polars."""
    want = lf.collect()
    eng = pe.MetalEngine(min_rows=0, shapes="all")
    got = lf.collect(engine=eng)
    rep = eng.last_report
    assert any(reason in f for f in rep.fallbacks), f"expected {reason!r} in:\n{rep}"
    compare(got, want, order, rel)
    return eng


def _lit(dtype, v):
    return pl.lit(v, dtype=PL_DTYPE[dtype])


def _float_rel(dtype):
    return 1e-3 if dtype == "float32" else 1e-9


# =============================================================================================
# M0 -- the engine, the walk, the Polars surfaces
# =============================================================================================


def test_import_arrowmetal_does_not_import_polars():
    code = "import sys, arrowmetal; assert 'polars' not in sys.modules, 'polars was imported'"
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "python"))
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=REPO)
    code = ("import sys, arrowmetal as am; e = am.MetalEngine; "
            "assert 'polars' in sys.modules and e.__name__ == 'MetalEngine'")
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=REPO)


def test_engine_identity():
    e = am.MetalEngine()
    assert e.name == "in-memory"
    assert e.plan_engine == "metal"
    assert repr(e).startswith("MetalEngine(")
    assert e.min_rows == pe.DEFAULT_MIN_ROWS and e.shapes == "measured" and not e.raise_on_fail
    assert am.MetalEngine is pe.MetalEngine
    with pytest.raises(ValueError):
        am.MetalEngine(shapes="some")


def test_polars_surfaces_this_module_uses_exist():
    """One named test fails when Polars renames an unstable surface the engine relies on."""
    from polars.lazyframe.engine import _LocalEngine
    assert hasattr(_LocalEngine, "_post_opt_callback") and hasattr(_LocalEngine, "collect")
    assert hasattr(pl.DataFrame, "_from_pydf")
    for kind in ("DataFrameScan", "Filter", "Select", "HStack", "SimpleProjection", "Slice", "Sort",
                 "GroupBy"):
        assert hasattr(pe._in, kind), kind
    for kind in ("Column", "Literal", "BinaryExpr", "Cast", "Ternary", "Function", "Agg", "Len",
                 "Operator", "BooleanFunction", "StringFunction"):
        assert hasattr(pe._xn, kind), kind
    seen = {}

    def cb(nt, duration=None):
        for m in ("version", "get_node", "set_node", "get_inputs", "view_current_node",
                  "view_expression", "get_dtype", "get_schema", "set_udf"):
            seen[m] = callable(getattr(nt, m, None))
        seen["version()"] = tuple(nt.version())

    pl.LazyFrame({"a": [1]}).collect(post_opt_callback=cb)
    assert all(v for k, v in seen.items() if k != "version()"), seen
    assert seen["version()"] == pe.TESTED_IR_VERSION == (14, 7)
    assert isinstance(pe.TESTED_POLARS, tuple) and pl.__version__ in pe.TESTED_POLARS
    # Every IR node class this Polars has is one the walk knows; a new one would make every plan
    # holding it stay with Polars (`test_an_unknown_node_kind_keeps_the_whole_plan_on_polars`).
    kinds = {k for k, v in vars(pe._in).items() if isinstance(v, type) and not k.startswith("_")}
    assert kinds == set(pe.KNOWN_NODE_KINDS), kinds ^ set(pe.KNOWN_NODE_KINDS)


def _node_kind_plans(tmp_path):
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    lf = pl.LazyFrame({"k": [1, 2, 3, None], "v": [4, 5, 6, 7], "l": [[1], [2, 3], [], None]})
    lf2 = pl.LazyFrame({"k": [1, 3], "w": [7, 8]})
    path = str(tmp_path / "a.parquet")
    pq.write_table(pa.table({"k": [1, 2]}), path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ctx = lf.select("k").with_context(lf2.select("w")).select(pl.col("k") + pl.col("w").sum())
    return {
        # kind: (plan, runs entirely on Metal)
        "DataFrameScan": (lf.select("k", "v").filter(pl.col("v") > 4), True),
        "Filter": (lf.filter(pl.col("v") > 4), False),          # carries a List column
        "Select": (lf.select((pl.col("v") * 2).alias("x")), True),
        "HStack": (lf.select("k", "v").with_columns((pl.col("v") + 1).alias("x")), True),
        "SimpleProjection": (lf.select("k", "v").filter((pl.col("v") > 4) & (pl.col("k") > 0))
                             .select((pl.col("v") + pl.col("v")).alias("x")), True),
        "Slice": (lf.select("v").filter(pl.col("v") > 4).slice(1, 1), True),
        "Sort": (lf.select("k", "v").sort("v", descending=True), True),
        "GroupBy": (lf.group_by("k").agg(pl.col("v").sum()), True),
        "Distinct": (lf.select("k", "v").unique(), True),
        "Join": (lf.select("k", "v").join(lf2, on="k"), True),
        "Cache": (lf.select("k", "v").join(lf.select("k", "v"), on="k"), False),
        "Union": (pl.concat([lf.select("k"), lf2.select("k")]), False),
        "HConcat": (pl.concat([lf.select("k"), lf2.select("w")], how="horizontal_extend"), False),
        "MapFunction": (lf.with_row_index(), False),
        "MergeSorted": (lf.select("v").merge_sorted(lf2.select(pl.col("w").alias("v")), key="v"),
                        False),
        "Scan": (pl.scan_parquet(path).filter(pl.col("k") > 1), True),
        "PythonScan": (pl.scan_pyarrow_dataset(ds.dataset(path)), False),
        "ExtContext": (ctx, False),
        "Sink": (lf.select("k").sink_parquet(str(tmp_path / "o.parquet"), lazy=True), False),
    }


def test_every_node_kind_walks_and_collects_identically(tmp_path):
    """The walk survives every plan node Polars can emit (`Reduce` is never emitted, per cudf's
    translator) and the result is always Polars' result."""
    plans = _node_kind_plans(tmp_path)
    for kind, (lf, full) in plans.items():
        eng = pe.MetalEngine(min_rows=0, shapes="all")
        got = lf.collect(engine=eng)
        walked = [k for _n, k in eng.last_report.walked]
        assert kind in walked, (kind, walked)
        if kind == "Sink":
            assert got.height == 0            # a sink returns an empty frame on both engines
            continue
        want = lf.collect()
        compare(got, want)
        if full:
            assert eng.last_report.ran_on_metal and not eng.last_report.fallbacks, (kind, eng.last_report)


def test_raise_on_fail_names_the_node_and_reason(tmp_path):
    plans = _node_kind_plans(tmp_path)
    ipc = str(tmp_path / "a.arrow")
    pl.DataFrame({"k": [1, 2]}).write_ipc(ipc)
    plans["Scan"] = (pl.scan_ipc(ipc).sort("k"), False)      # polars 1.44.1 cannot show it
    for kind in ("Cache", "Union", "HConcat", "MapFunction", "MergeSorted", "Scan",
                 "PythonScan", "ExtContext"):
        lf, _ = plans[kind]
        with pytest.raises(pl.exceptions.ComputeError) as err:
            lf.collect(engine=metal())
        text = str(err.value)
        assert "ArrowMetal MetalEngine:" in text and kind in text, text
        assert "'cuda' conversion failed" in text      # hardcoded in polars 1.44.1
        assert pe.FORCE_POLARS in text, text
    # A plan with nothing unsupported does not raise.
    plans["Sort"][0].collect(engine=metal())


def test_report_says_what_ran_where():
    df = pl.DataFrame({"k": [1, 2, 3], "v": [3, 1, 2]})
    lf = df.lazy().filter(pl.col("v") > 1).select(pl.col("v").rank().alias("r"))
    eng = pe.MetalEngine(min_rows=0, shapes="all")
    lf.collect(engine=eng)
    rep = eng.last_report
    assert [t["root"].split("#")[0] for t in rep.taken] == ["Filter"]
    assert any(f.startswith("Select#") and "rank" in f for f in rep.fallbacks), rep.fallbacks
    text = str(rep)
    assert "metal:  Filter#" in text and "polars: Select#" in text
    assert rep.taken[0]["seconds"] is not None and rep.taken[0]["rows_out"] == 2


def test_background_collection_warns_and_runs_on_polars():
    pe._warned_paths.clear()
    lf = pl.LazyFrame({"v": [1, 2, 3]}).select(pl.col("v") * 2)
    eng = metal()
    with pytest.warns(pe.MetalEngineFallbackWarning, match="background"):
        q = lf.collect(engine=eng, background=True)
    assert q.fetch_blocking().equals(lf.collect())
    assert eng.last_report.path == "background" and not eng.last_report.taken


def test_a_sink_plan_is_left_to_polars(tmp_path):
    """Polars runs `sink_*` by collecting the sink plan through the engine, and its streaming sink
    panics on a replaced subtree ("entered unreachable code"), so a plan with a sink stays whole,
    says so in the report, and warns once."""
    pe._warned_paths.clear()
    lf = pl.LazyFrame({"v": [3, 1, 2]}).sort("v")
    eng = pe.MetalEngine(min_rows=0, shapes="all")
    with pytest.warns(pe.MetalEngineFallbackWarning, match="ends in a sink"):
        lf.sink_parquet(tmp_path / "x.parquet", engine=eng)
    assert pl.read_parquet(tmp_path / "x.parquet").equals(lf.collect())
    assert not eng.last_report.taken and eng.last_report.path == "sink"
    assert any("sinks" in f for f in eng.last_report.fallbacks), eng.last_report
    with warnings.catch_warnings():
        warnings.simplefilter("error", pe.MetalEngineFallbackWarning)      # once per process
        lf.sink_csv(tmp_path / "x.csv", engine=eng)
    assert pl.read_csv(tmp_path / "x.csv").equals(lf.collect())


def test_collect_async_runs_on_polars_and_says_so():
    import asyncio
    pe._warned_paths.clear()
    lf = pl.LazyFrame({"v": [3, 1, 2]}).sort("v")
    eng = pe.MetalEngine(min_rows=0, shapes="all")

    async def go():
        return await lf.collect_async(engine=eng)

    with pytest.warns(pe.MetalEngineFallbackWarning, match="collect_async"):
        assert asyncio.run(go()).equals(lf.collect())
    assert eng.last_report.path == "collect_async" and not eng.last_report.taken
    assert "no engine callback" in eng.last_report.fallbacks[0]


def test_eager_and_background_callbacks_are_none():
    pe._warned_paths.clear()
    e = am.MetalEngine()
    assert e._post_opt_callback(background=False, eager=True) is None
    assert e.last_report.path == "eager"
    with pytest.warns(pe.MetalEngineFallbackWarning):
        assert e._post_opt_callback(background=True, eager=False) is None
    assert callable(e._post_opt_callback(background=False, eager=False))


def test_profile_shows_a_metal_row():
    lf = pl.LazyFrame({"k": [1, 2, 1, 3] * 50, "v": list(range(200))}).sort("v", descending=True)
    eng = pe.MetalEngine(min_rows=0, shapes="all")
    out, timings = eng.profile(lf)
    compare(out, lf.collect(), order=["v"])
    nodes = timings["node"].to_list()
    assert any(n.startswith("metal:Sort#") for n in nodes), nodes
    row = timings.filter(pl.col("node").str.starts_with("metal:"))
    assert (row["end"] >= row["start"]).all()


def test_callback_second_argument_is_none_under_collect_and_an_int_under_profile(monkeypatch):
    seen = []
    real = pe.execute_with_metal

    def spy(nt, duration, **kw):
        seen.append(duration)
        return real(nt, duration, **kw)

    monkeypatch.setattr(pe, "execute_with_metal", spy)
    lf = pl.LazyFrame({"v": [3, 1, 2]}).sort("v")
    eng = pe.MetalEngine(min_rows=0, shapes="all")
    lf.collect(engine=eng)
    eng.profile(lf)
    assert seen[0] is None and isinstance(seen[1], int), seen


def test_lazyframe_profile_with_engine_runs_polars_only():
    """polars 1.44.1's `LazyFrame.profile` passes the callback only for a GPUEngine, so
    `lf.profile(engine=MetalEngine())` profiles Polars; `MetalEngine.profile(lf)` is the way."""
    lf = pl.LazyFrame({"v": list(range(100))}).sort("v", descending=True)
    eng = pe.MetalEngine(min_rows=0, shapes="all")
    _out, timings = lf.profile(engine=eng)
    assert not any(n.startswith("metal:") for n in timings["node"].to_list())
    assert eng.last_report is None


# =============================================================================================
# every collect path: on Metal, or on Polars and saying so (docs/POLARS.md, "Collect paths")
# =============================================================================================


def _path_plan():
    return pl.LazyFrame({"v": [3, 1, 2] * 10, "k": list(range(30))}).sort("v")


def _run_async(start):
    """Runs `start()` (which calls a Polars async collect) inside an event loop and awaits it."""
    import asyncio

    async def go():
        return await start()

    return asyncio.run(go())


def _in_config(eng, fn):
    with pl.Config(engine_affinity=eng):
        return fn()


# path: (run it with engine `e` and a temporary directory `d`, runs on Metal, report path, warns)
COLLECT_PATHS = {
    "collect": (lambda e, d: _path_plan().collect(engine=e), True, "collect", None),
    "head().collect": (lambda e, d: _path_plan().head(5).collect(engine=e), True, "collect", None),
    "Config(engine_affinity=e) + lazy().collect()": (
        lambda e, d: _in_config(e, lambda: pl.DataFrame({"v": [3, 1, 2]}).lazy().sort("v").collect()),
        True, "collect", None),
    "Config(tbl_rows=3) + collect(engine=e)": (
        lambda e, d: _in_config_rows(lambda: _path_plan().collect(engine=e)), True, "collect", None),
    "collect_all": (lambda e, d: pl.collect_all([_path_plan(), _path_plan().head(3)], engine=e),
                    True, "collect_all", None),
    "engine.profile(lf)": (lambda e, d: e.profile(_path_plan()), True, "profile", None),
    "engine.explain(lf)": (lambda e, d: e.explain(_path_plan()), True, "explain", None),
    "collect_async": (lambda e, d: _run_async(lambda: _path_plan().collect_async(engine=e)),
                      False, "collect_async", "collect_async"),
    "collect_all_async": (lambda e, d: _run_async(lambda: pl.collect_all_async([_path_plan()],
                                                                               engine=e)),
                          False, "collect_all_async", "collect_all_async"),
    "collect_batches": (lambda e, d: list(_path_plan().collect_batches(engine=e)), False,
                        "collect_batches", "collect_batches"),
    "collect(background=True)": (lambda e, d: _path_plan().collect(engine=e, background=True)
                                 .fetch_blocking(), False, "background", "background"),
    "sink_parquet": (lambda e, d: _path_plan().sink_parquet(os.path.join(d, "o.parquet"), engine=e),
                     False, "sink", "ends in a sink"),
    "sink_ipc": (lambda e, d: _path_plan().sink_ipc(os.path.join(d, "o.arrow"), engine=e),
                 False, "sink", "ends in a sink"),
    "sink_csv": (lambda e, d: _path_plan().sink_csv(os.path.join(d, "o.csv"), engine=e),
                 False, "sink", "ends in a sink"),
    "sink_ndjson": (lambda e, d: _path_plan().sink_ndjson(os.path.join(d, "o.ndjson"), engine=e),
                    False, "sink", "ends in a sink"),
    "sink_batches": (lambda e, d: _path_plan().sink_batches(lambda df: None, engine=e),
                     False, "sink", "ends in a sink"),
    "lazy sink + collect": (lambda e, d: _path_plan().sink_parquet(os.path.join(d, "l.parquet"),
                                                                   lazy=True).collect(engine=e),
                            False, "sink", "ends in a sink"),
}


def _in_config_rows(fn):
    with pl.Config(tbl_rows=3):
        return fn()


@pytest.mark.parametrize("path", list(COLLECT_PATHS))
def test_every_collect_path_is_explicit(path, tmp_path):
    """Each path either runs on Metal and the report names the path, or runs the whole plan on
    Polars, the report says why, and (except `eager`) a MetalEngineFallbackWarning says so once."""
    run, on_metal, report_path, warns = COLLECT_PATHS[path]
    pe._warned_paths.clear()
    eng = pe.MetalEngine(min_rows=0, shapes="all")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        run(eng, str(tmp_path))
    ours = [str(w.message) for w in seen if issubclass(w.category, pe.MetalEngineFallbackWarning)]
    rep = eng.last_report
    assert rep is not None and rep.path == report_path, (path, rep)
    assert bool(rep.taken) == on_metal, (path, str(rep))
    if on_metal:
        assert not rep.fallbacks and not ours, (path, str(rep), ours)
    else:
        assert rep.fallbacks and len(ours) == 1 and warns in ours[0], (path, str(rep), ours)
        with warnings.catch_warnings(record=True) as again:          # once per process
            warnings.simplefilter("always")
            run(pe.MetalEngine(min_rows=0, shapes="all"), str(tmp_path))
        assert not [w for w in again if issubclass(w.category, pe.MetalEngineFallbackWarning)]
    assert f"report for {report_path}" in str(rep)


def test_polars_side_entry_points_that_never_call_the_engine():
    """`lf.explain(engine=)` and `lf.profile(engine=)` read nothing from the engine but its
    `plan_engine`, and an eager DataFrame method runs on Polars' in-memory engine whatever the
    configured affinity: no callback, no report."""
    lf = _path_plan()
    eng = pe.MetalEngine(min_rows=0, shapes="all")
    assert lf.explain(engine=eng) == lf.explain()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        lf.profile(engine=eng)
    pl.Config.set_engine_affinity(eng)
    try:
        pl.DataFrame({"v": [3, 1, 2]}).sort("v")
        pl.DataFrame({"v": [3, 1, 2]}).lazy().sort("v").collect(engine="in-memory")
    finally:
        pl.Config.set_engine_affinity(None)
    assert eng.last_report is None


def test_collect_all_runs_each_frame_through_the_engine():
    a = _path_plan()
    b = pl.LazyFrame({"x": [2.0, 1.0, float("nan")]}).sort("x", descending=True)
    eng = pe.MetalEngine(min_rows=0, shapes="all")
    got = pl.collect_all([a, b], engine=eng)
    want = pl.collect_all([a, b])
    for g, w in zip(got, want):
        compare(g, w, order=True if g.width == 1 else [g.columns[0]])
    assert [r.path for r in eng.last_reports] == ["collect_all", "collect_all"]
    assert all(r.taken for r in eng.last_reports) and eng.last_report is eng.last_reports[-1]


def test_explain_reports_the_placement_without_running_the_query():
    lf = pl.LazyFrame({"v": list(range(100))}).filter(pl.col("v") > 3).select(
        pl.col("v").rank().alias("r"))
    eng = pe.MetalEngine(min_rows=0, shapes="all")
    text = eng.explain(lf)
    assert lf.explain() in text and "report for explain" in text
    rep = eng.last_report
    assert [t["root"].split("#")[0] for t in rep.taken] == ["Filter"]
    assert rep.taken[0]["seconds"] is None                      # nothing ran
    eng2 = pe.MetalEngine(min_rows=0, shapes="all")
    lf.collect(engine=eng2)
    assert rep.walked == eng2.last_report.walked and rep.fallbacks == eng2.last_report.fallbacks


# =============================================================================================
# version compatibility: fail closed, and the check command
# =============================================================================================


def test_an_unknown_node_kind_keeps_the_whole_plan_on_polars(monkeypatch):
    """A node kind the walk does not know (a Polars release that adds one) is not an error: the
    whole plan stays with Polars, and the report names the node and the Polars version."""
    monkeypatch.setattr(pe, "KNOWN_NODE_KINDS", pe.KNOWN_NODE_KINDS - {"Sort"})
    lf = pl.LazyFrame({"v": [3, 1, 2], "w": [1, 2, 3]}).filter(pl.col("w") > 1).sort("v")
    eng = pe.MetalEngine(min_rows=0, shapes="all")
    assert lf.collect(engine=eng).equals(lf.collect())
    rep = eng.last_report
    assert not rep.taken, rep
    assert len(rep.fallbacks) == 1, rep
    assert f"unknown node Sort in polars {pl.__version__}" in rep.fallbacks[0], rep


def test_a_node_polars_cannot_describe_keeps_the_whole_plan_on_polars(monkeypatch):
    """polars 1.44.1 raises from `view_current_node` for a `sink_batches` Sink when cloudpickle is
    not installed. Any such failure keeps the whole plan with Polars instead of failing it."""
    real_exec = pe.execute_with_metal
    calls = []

    def spy(nt, d=None, **kw):
        class Traverser:                   # the real traverser, whose second node cannot be shown
            def __getattr__(self, name):
                return getattr(nt, name)

            def view_current_node(self):
                calls.append(nt.get_node())
                if len(calls) == 2:
                    raise ValueError("synthetic: cannot show")
                return nt.view_current_node()

        return real_exec(Traverser(), d, **kw)

    monkeypatch.setattr(pe, "execute_with_metal", spy)
    lf = pl.LazyFrame({"v": [3, 1, 2]}).filter(pl.col("v") > 1).sort("v")
    eng = pe.MetalEngine(min_rows=0, shapes="all")
    assert lf.collect(engine=eng).equals(lf.collect())
    rep = eng.last_report
    assert not rep.taken
    assert len(rep.fallbacks) == 1 and "unknown node Node#" in rep.fallbacks[0], rep
    assert "ValueError: synthetic: cannot show" in rep.fallbacks[0], rep


def test_the_off_switch_leaves_every_plan_to_polars(monkeypatch):
    monkeypatch.setenv(pe.OFF_ENV, "off")
    lf = _path_plan()
    eng = pe.MetalEngine(min_rows=0, shapes="all", raise_on_fail=True)   # does not raise
    assert lf.collect(engine=eng).equals(lf.collect())
    assert not eng.last_report.taken
    assert f"{pe.OFF_ENV}=off is set" in eng.last_report.fallbacks[0]
    monkeypatch.setenv(pe.OFF_ENV, "on")
    lf.collect(engine=eng)
    assert eng.last_report.taken


def test_the_check_command_prints_versions_and_the_capability_header():
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "python"))
    env.pop(pe.OFF_ENV, None)
    r = subprocess.run([sys.executable, "-m", "arrowmetal.polars_engine", "check"], env=env,
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    assert f"polars {pl.__version__}: a tested release" in out, out
    assert f"IR version {pe.TESTED_IR_VERSION}: the tested major" in out, out
    assert "callback API: present" in out and "unknown to the engine: none" in out, out
    assert "capability table (docs/ENGINE_CAPABILITIES.md):" in out, out
    assert re.search(r"- Polars (\S+), IR \(14, 7\)", out).group(1) in pe.TESTED_POLARS, out


# =============================================================================================
# the capability table and the messages
# =============================================================================================


def test_the_capability_table_is_current():
    """docs/ENGINE_CAPABILITIES.md is what `engine_capabilities.py` generates now (apart from the
    line naming the commit), and no cell of it is a wrong answer or an exception."""
    import engine_capabilities as ecap
    results = ecap.run_all()
    assert not ecap.bad_cells(results), ecap.bad_cells(results)
    with open(ecap.DOC, encoding="utf-8") as f:
        committed = f.read()
    fresh = ecap.render(results)
    # The table was generated on one of TESTED_POLARS; on another tested release it must be the
    # same table, apart from the line naming the Polars version.
    recorded = re.search(r"^- Polars (\S+), IR ", committed, re.M).group(1)
    assert recorded in pe.TESTED_POLARS, recorded
    if pl.__version__ != recorded:
        assert pl.__version__ in pe.TESTED_POLARS, pl.__version__
        committed = committed.replace(f"- Polars {recorded}, ", f"- Polars {pl.__version__}, ", 1)
    if ecap.comparable(committed) != ecap.comparable(fresh):
        import difflib
        diff = "\n".join(difflib.unified_diff(ecap.comparable(committed).splitlines(),
                                              ecap.comparable(fresh).splitlines(), "committed",
                                              "fresh", lineterm="", n=1))
        pytest.fail("docs/ENGINE_CAPABILITIES.md is stale; regenerate it with "
                    "`python python/tests/engine_capabilities.py --write`:\n" + diff[:6000])


_VERB = re.compile(r"\b(is|are|was|has|have|does|do|can|cannot|may|stays|stay|holds|counts|"
                   r"differ|needs|adds|keeps|returns|takes|runs)\b")


def _reason_texts():
    """Every `_Unsupported(...)` message of the engine, with each `{...}` field as `{}`."""
    import ast
    path = os.path.join(REPO, "python", "arrowmetal", "polars_engine.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_Unsupported":
            (arg,) = node.args
            if isinstance(arg, ast.Constant):
                text = arg.value
            else:
                assert isinstance(arg, ast.JoinedStr), ast.dump(arg)
                text = "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in arg.values)
            out.append((node.lineno, text))
    return out


def test_every_unsupported_reason_is_a_full_sentence():
    """The report prints each reason after its node (`Sort#3: <reason>`): a capitalised sentence
    with a verb, ending in a full stop."""
    texts = _reason_texts()
    assert len(texts) > 90
    for line, text in texts:
        assert text[:1].isupper(), (line, text)
        assert text.endswith("."), (line, text)
        assert len(text.split()) >= 4 and _VERB.search(text), (line, text)


def test_engine_errors_name_the_node_the_reason_and_how_to_force_polars(monkeypatch):
    lf = pl.LazyFrame({"v": [1, 2, 3]}).select(pl.col("v").rank())
    with pytest.raises(pl.exceptions.ComputeError) as err:
        lf.collect(engine=metal())
    text = str(err.value)
    assert "Select#" in text and "rank has no ArrowMetal translation" in text, text
    assert pe.FORCE_POLARS in text and "raise_on_fail=True" in text, text

    lf = pl.LazyFrame({"v": [1, 2, 3]}).sort("v", descending=True)
    monkeypatch.setattr(pe, "_validate", lambda sub, schema: None)

    def fail(plan, leaves, rows=None, scans=None):
        raise am.ArrowMetalError("synthetic failure")

    monkeypatch.setattr(pe, "_run_plan", fail)
    with pytest.raises(am.ArrowMetalError) as err:
        lf.collect(engine=pe.MetalEngine(min_rows=0, shapes="all"))
    text = str(err.value)
    assert "the subtree at Sort#" in text and "synthetic failure" in text, text
    assert pe.FORCE_POLARS in text, text

    monkeypatch.setattr(pe, "_run_plan", lambda plan, leaves, rows=None, scans=None:
                        pa.table({"v": pa.array([3.0, 2.0, 1.0])}))
    with pytest.raises(am.ArrowMetalError) as err:
        lf.collect(engine=pe.MetalEngine(min_rows=0, shapes="all"))
    text = str(err.value)
    assert "the subtree at Sort#" in text and "schema" in text and pe.FORCE_POLARS in text, text


def test_polars_verbose_warns_with_the_fallbacks(monkeypatch):
    monkeypatch.setenv("POLARS_VERBOSE", "1")
    lf = pl.LazyFrame({"v": [1, 2, 3]}).select(pl.col("v").rank())
    with pytest.warns(pl.exceptions.PerformanceWarning, match="rank"):
        lf.collect(engine=pe.MetalEngine(min_rows=0, shapes="all"))


def test_a_wrong_schema_from_metal_is_an_error_not_a_silent_mismatch(monkeypatch):
    """Polars does not check what an engine returns; the engine does, inside the replaced node."""
    lf = pl.LazyFrame({"v": [1, 2, 3]}).sort("v", descending=True)
    monkeypatch.setattr(pe, "_validate", lambda sub, schema: None)
    monkeypatch.setattr(pe, "_run_plan", lambda plan, leaves, rows=None, scans=None:
                        pa.table({"v": pa.array([3.0, 2.0, 1.0])}))
    with pytest.raises(am.ArrowMetalError, match="schema"):
        lf.collect(engine=pe.MetalEngine(min_rows=0, shapes="all"))


def test_a_plan_metal_rejects_falls_back_at_translation(monkeypatch):
    lf = pl.LazyFrame({"v": [1, 2, 3]}).sort("v", descending=True)

    def reject(plan, leaves, rows=None):
        raise am.ArrowMetalError("synthetic rejection")

    monkeypatch.setattr(pe, "_run_plan", reject)
    pe._validated.clear()
    try:
        eng = check_fallback(lf, "synthetic rejection", order=True)
    finally:
        pe._validated.clear()
    assert not eng.last_report.taken


def test_a_polars_export_panic_falls_back_instead_of_failing(monkeypatch):
    """pyo3's PanicException derives from BaseException, not Exception; a panic while exporting
    the input to ArrowMetal is a rejection like any other."""
    class Panic(BaseException):
        pass

    lf = pl.LazyFrame({"v": [1, 2, 3]}).sort("v", descending=True)

    def panic(plan, leaves, rows=None):
        raise Panic("synthetic panic")

    monkeypatch.setattr(pe, "_run_plan", panic)
    pe._validated.clear()
    try:
        eng = check_fallback(lf, "Panic: synthetic panic", order=True)
    finally:
        pe._validated.clear()
    assert not eng.last_report.taken


def test_a_nul_in_a_column_name_falls_back():
    """Polars' own `to_arrow` panics on a NUL in a column name; plain `collect` does not export,
    so the engine leaves such a frame to Polars."""
    df = pl.DataFrame({"a\x00b": [3, 1, 2], "v": [1, 2, 3]})
    check_fallback(df.lazy().sort("v"), "NUL byte", order=True)


def test_helper_columns_never_take_a_user_column_name():
    """The sort's helper columns (`__arrowmetal_valid1`, `__arrowmetal_nan2`, ...) skip names the
    frame already uses, so such a frame still runs on Metal."""
    df = pl.DataFrame({"a": [1.0, None, float("nan"), 3.0], "__arrowmetal_valid1": [1, 2, 3, 4],
                       "__arrowmetal_nan2": [5, 6, 7, 8], "__arrowmetal_key3": [9, 9, 9, 9]})
    eng = check(df.lazy().sort("a", descending=True), kinds=["Sort"], order=True)
    assert '"__arrowmetal_valid2"' in eng.last_report.taken[0]["plan"]
    check(df.lazy().with_columns((pl.col("a") * 2.0).alias("b")).sort("b"), kinds=["Sort"],
          order=True)


def test_a_translator_bug_falls_back_instead_of_failing(monkeypatch):
    def boom(self, *a, **k):
        raise KeyError("synthetic")

    monkeypatch.setattr(pe._Translator, "_node_Sort", boom)
    check_fallback(pl.LazyFrame({"v": [2, 1]}).sort("v"), "translation error KeyError", order=True)


def test_a_newer_ir_major_falls_back_entirely(monkeypatch):
    monkeypatch.setattr(pe, "TESTED_IR_VERSION", (13, 0))
    check_fallback(pl.LazyFrame({"v": [2, 1]}).sort("v"), "not the tested", order=True)


def test_existing_polars_suites_collect_identically(tmp_path):
    """Every LazyFrame `test_polars.py` and `test_lazy.py` collect is collected again through
    `MetalEngine(min_rows=0, shapes="all")` by the `metal_engine_everywhere` plugin, and must match."""
    stats = tmp_path / "stats.json"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(REPO, "python"), HERE]),
               AM_EVERYWHERE_STATS=str(stats))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "metal_engine_everywhere",
                        os.path.join(HERE, "test_polars.py"), os.path.join(HERE, "test_lazy.py")],
                       env=env, cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    s = json.loads(stats.read_text())
    assert s["compared"] >= 30 and s["taken"] >= 15, s


# =============================================================================================
# M1 -- element-wise subtrees
# =============================================================================================

CMP_OPS = {"eq": lambda x, y: x == y, "ne": lambda x, y: x != y, "lt": lambda x, y: x < y,
           "le": lambda x, y: x <= y, "gt": lambda x, y: x > y, "ge": lambda x, y: x >= y}


@pytest.mark.parametrize("shape", CORE, ids=CORE_IDS)
@pytest.mark.parametrize("dtype", NUMERIC)
def test_filter_comparisons(dtype, shape):
    df = frame(dtype, shape)
    for name, op in CMP_OPS.items():
        check(df.lazy().filter(op(pl.col("a"), pl.col("b"))), kinds=["Filter"])
        check(df.lazy().filter(op(pl.col("a"), _lit(dtype, 1))).select("a", "k"), kinds=["Filter"])


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("dtype", NUMERIC)
def test_arithmetic_and_comparison_columns(dtype, shape):
    df = frame(dtype, shape)
    lf = df.lazy().select(
        (pl.col("a") + pl.col("b")).alias("add"),
        (pl.col("a") - pl.col("b")).alias("sub"),
        (pl.col("a") * pl.col("b")).alias("mul"),
        (pl.col("a") / pl.col("b")).alias("div"),
        (pl.col("a") + _lit(dtype, 3)).alias("addl"),
        (pl.col("a") >= pl.col("b")).alias("ge"),
        (pl.col("a") == pl.col("b")).alias("eq"),
        pl.col("k"),
    )
    check(lf, kinds=["Select"], order=True)


@pytest.mark.parametrize("shape", CORE, ids=CORE_IDS)
@pytest.mark.parametrize("dtype", NUMERIC + ["bool"])
def test_null_logic_and_ternary(dtype, shape):
    df = frame(dtype, shape)
    lit = _lit(dtype, True if dtype == "bool" else 7)
    lf = df.lazy().with_columns(
        pl.col("a").is_null().alias("isnull"),
        pl.col("a").is_not_null().alias("notnull"),
        pl.col("a").fill_null(lit).alias("filled"),
        pl.when(pl.col("g")).then(pl.col("a")).otherwise(pl.col("b")).alias("when"),
        pl.when(pl.col("k") > 3).then(pl.col("a")).otherwise(None).alias("when_null"),
    ).drop("s")
    check(lf, kinds=["HStack"], order=True)


@pytest.mark.parametrize("shape", CORE, ids=CORE_IDS)
@pytest.mark.parametrize("dtype", INTS + FLOATS)
def test_is_in(dtype, shape):
    df = frame(dtype, shape)
    vals = [0, 1, 2, 5, 100] if dtype in INTS else [0.0, 1.5, -2.25]
    lf = df.lazy().select(pl.col("a").is_in(pl.Series(vals, dtype=PL_DTYPE[dtype])).alias("i"),
                          pl.col("k").is_in([1, 3, None]).alias("j"), pl.col("k"))
    check(lf, kinds=["Select"], order=True)


def test_is_in_matches_a_nan_in_the_list():
    """Polars matches floats in its total order, so a NaN in the list matches a NaN row (either
    sign); ArrowMetal's is_in alone would compare with IEEE equality, where NaN matches nothing."""
    nan, inf = float("nan"), float("inf")
    for dt in (pl.Float32, pl.Float64):
        df = pl.DataFrame({"f": [nan, 1.0, inf, -0.0, 0.0, None, 2.0, -nan]}, schema={"f": dt})
        for vals in ([nan, 1.0], [nan], [-nan], [inf, nan, None], [nan, -inf], [0.0], [-0.0, 2.0]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                lf = df.lazy().select(pl.col("f").is_in(pl.Series(vals, dtype=dt)).alias("i"),
                                      pl.col("f").is_in(vals).alias("j"))
                eng = check(lf, kinds=["Select"], order=True)
            if any(v is not None and v != v for v in vals):
                assert "(ne " in eng.last_report.taken[0]["plan"]


def test_true_division_by_a_literal_is_polars_reciprocal_multiply():
    """Polars divides by a scalar as `x * (1 / c)`, which differs from the correctly rounded `x / c`
    by one ulp in about a third of the rows; the engine emits the same product, so the bits match.
    A column divisor stays a true division, which is what Polars computes there."""
    rng = np.random.default_rng(11)
    n = 100_003
    x = rng.standard_normal(n)
    x[::97] = np.nan
    df = pl.DataFrame({"q": rng.integers(0, 10**9, n), "x": x, "f": x.astype(np.float32),
                       "i": rng.integers(-10**6, 10**6, n), "i8": rng.integers(-100, 100, n)
                       .astype(np.int8)}).with_columns(
        pl.when(pl.col("q") % 11 == 0).then(None).otherwise(pl.col("x")).alias("xn"))
    ieee = x / 3.0
    off = int((df.select(pl.col("x") / 3.0)["x"].to_numpy() != ieee)[~np.isnan(x)].sum())
    assert 0.30 < off / n < 0.37, off           # "about a third of Float64 rows for / 3.0"
    # "At most one ulp, in a share of rows that depends on the divisor ... none for a power of two".
    finite = np.isfinite(x)
    for d in (3.0, 7.0, 0.1, 1e-5, 3.3333333333333335, 0.3, 2.0, 0.125, 2.0 ** -40):
        got = df.select(pl.col("x") / d)["x"].to_numpy()[finite].view(np.int64)
        ulps = np.abs(got - (x[finite] / d).view(np.int64))
        assert ulps.max() <= 1, (d, ulps.max())
        if d in (2.0, 0.125, 2.0 ** -40):
            assert ulps.max() == 0, d
    for c in ("x", "f", "i", "i8", "xn"):
        exprs = [(pl.col(c) / d).alias(f"{c}/{d}")
                 for d in (3.0, 7, 0.1, 1e-310, 0.0, -0.0, float("inf"), float("nan"), -2.5)]
        exprs += [(pl.col(c) / pl.lit(3.0, dtype=pl.Float32)).alias(f"{c}/f32"),
                  (pl.col(c) / pl.lit(None, dtype=pl.Float64)).alias(f"{c}/null"),
                  (pl.col(c) / pl.col("x")).alias(f"{c}/x")]
        if c != "f":    # `3.0 / Float32`: Polars says Float64 in its schema, returns Float32
            exprs.append((3.0 / pl.col(c)).alias(f"3/{c}"))
        eng = check(df.lazy().select(exprs), kinds=["Select"], order=True)
        assert "(mul " in eng.last_report.taken[0]["plan"]
    # The shape the default engine takes: a large sort carrying the quotient.
    check(df.lazy().with_columns((pl.col("x") / 3.0).alias("y")).sort("q"), kinds=["Sort"],
          order=True)



# A finite float literal of magnitude 2**63 or more once trapped the process inside ArrowMetal's
# expression compiler (it converted every float literal to Int64 as well), and the translator declined
# it. The compiler no longer does, so these plans run on Metal. Still run in a child process: a trap
# there fails this test instead of ending the pytest run.
_HUGE_LITERALS = r"""
import json, sys, warnings
import numpy as np, polars as pl
from polars.testing import assert_frame_equal
import arrowmetal as am
warnings.simplefilter("ignore")
df = pl.DataFrame({"x": [1.0, 2.0, None, 1e19, -1e19, float("nan")],
                   "f": pl.Series([1.0, 2.0, None, 1e19, -1e19, float("nan")], dtype=pl.Float32),
                   "i": [1, 2, None, 4, 5, 6]})
x, f = pl.col("x"), pl.col("f")
cases = {
    "gt": x > 1e19, "eq": x == 1e19, "lt_neg": x < -1e19, "f32_gt": f > 1e19,
    "f32_lit": f > pl.lit(1e30, pl.Float32), "mul": x * 1e30, "add": x + 1e100, "sub": x - 1e100,
    "int_mul": pl.col("i") * 1e19, "fill_null": x.fill_null(1e19),
    "when_then": pl.when(x > 1.5).then(1e19).otherwise(x), "is_in": x.is_in([1.0, 1.7e308]),
    "recip_63": x / 2.0 ** -63, "recip_1022": x / 2.0 ** -1022, "f32_recip": f / 2.0 ** -70,
    "below": x * 9.2e18, "min_i64": x * -(2.0 ** 63), "inf": x * float("inf"),
}
out = {}
for name, e in cases.items():
    lf = df.lazy().select(e.alias("o"))
    eng = am.MetalEngine(min_rows=0, shapes="all", raise_on_fail=True)
    assert_frame_equal(lf.collect(engine=eng), lf.collect())
    out[name] = [bool(eng.last_report.taken), list(eng.last_report.fallbacks)]
# The shape the default engine takes: a large sort, here with a filter against such a literal.
n = 1_000_000
rng = np.random.default_rng(3)
big = pl.DataFrame({"q": rng.integers(0, 10**9, n), "x": rng.standard_normal(n) * 1e19})
lf = big.lazy().filter(pl.col("x") < 1e19).sort("q")
eng = am.MetalEngine()
assert_frame_equal(lf.collect(engine=eng).sort("q", "x"), lf.collect().sort("q", "x"))
out["default_sort"] = [bool(eng.last_report.taken), list(eng.last_report.fallbacks)]
print(json.dumps(out))
"""


def test_a_float_literal_of_magnitude_2_63_or_more_runs_on_metal():
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "python"))
    r = subprocess.run([sys.executable, "-c", _HUGE_LITERALS], env=env, cwd=REPO,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (r.returncode, r.stderr[-2000:])
    out = json.loads(r.stdout.strip().splitlines()[-1])
    for name, (taken, fallbacks) in out.items():
        assert taken, (name, fallbacks)
        assert not any("2**63" in fb for fb in fallbacks), (name, fallbacks)


def test_multiply_by_minus_one_is_a_negation_like_polars():
    """Polars multiplies a float by a scalar -1 (either side, or divides by -1) as a negation, so a
    NaN comes back with its sign bit flipped; the engine emits `negate` and matches the raw bits,
    which `assert_frame_equal` cannot see. Any other multiplier keeps the input NaN in both."""
    nan = float("nan")
    x = np.array([nan, -nan, 1.0, 0.0, -0.0, np.inf, 5e-324, -2.5, 3.0])
    df = pl.DataFrame({"x": x, "f": x.astype(np.float32), "i": np.arange(9) - 4}).with_columns(
        pl.when(pl.col("i") == 0).then(None).otherwise(pl.col("x")).alias("xn"))
    for c in ("x", "f", "xn", "i"):
        col = pl.col(c)
        cases = {f"{c}*-1.0": col * -1.0, f"-1.0*{c}": -1.0 * col, f"{c}/-1.0": col / -1.0,
                 f"{c}*-1": col * -1, f"{c}/-1": col / pl.lit(-1),
                 f"{c}*f32(-1)": col * pl.lit(-1.0, pl.Float32),
                 f"{c}*-2.5": col * -2.5, f"{c}*-1.0000000000000002": col * -1.0000000000000002,
                 f"{c}/3.0": col / 3.0}
        lf = df.lazy().select([e.alias(k) for k, e in cases.items()])
        want = lf.collect()
        eng = metal()
        got = lf.collect(engine=eng)
        assert got.schema == want.schema
        for k in cases:
            a, b = want[k], got[k]
            assert (a.is_null() == b.is_null()).all(), k
            an, bn = a.to_numpy(), b.to_numpy()
            if an.dtype.kind == "f":
                view = np.uint32 if an.dtype == np.float32 else np.uint64
                assert (an.view(view) == bn.view(view)).all(), (k, an, bn)
            else:
                assert a.equals(b), k
        plan = eng.last_report.taken[0]["plan"]
        assert "(negate " in plan and "(f64 -1.0)" not in plan, plan

LOSSLESS = {"int8": [pl.Int16, pl.Int64, pl.Float64, pl.Float32], "int16": [pl.Int32, pl.Float32],
            "int32": [pl.Int64, pl.Float64], "int64": [pl.Float64],
            "uint8": [pl.UInt16, pl.Int16, pl.Float32], "uint16": [pl.UInt64, pl.Int32],
            "uint32": [pl.Int64, pl.UInt64], "uint64": [pl.Float64],
            "float32": [pl.Float64], "bool": [pl.Int8, pl.UInt32]}


@pytest.mark.parametrize("shape", CORE, ids=CORE_IDS)
@pytest.mark.parametrize("dtype", list(LOSSLESS))
def test_lossless_casts(dtype, shape):
    df = frame(dtype, shape)
    lf = df.lazy().select([pl.col("a").cast(t).alias(f"c{i}") for i, t in enumerate(LOSSLESS[dtype])])
    check(lf, kinds=["Select"], order=True)


@pytest.mark.parametrize("shape", CORE, ids=CORE_IDS)
def test_boolean_logic_is_kleene(shape):
    df = frame("bool", shape)
    lf = df.lazy().select(
        (pl.col("a") & pl.col("b")).alias("and"), (pl.col("a") | pl.col("b")).alias("or"),
        (pl.col("a") ^ pl.col("b")).alias("xor"), (~pl.col("a")).alias("not"), pl.col("k"))
    check(lf, kinds=["Select"], order=True)
    check(df.lazy().filter(pl.col("a") | (pl.col("k") > 4)), kinds=["Filter"])


@pytest.mark.parametrize("shape", CORE, ids=CORE_IDS)
@pytest.mark.parametrize("dtype", ["int8", "uint16", "int64"])
def test_integer_bitwise(dtype, shape):
    df = frame(dtype, shape)
    lf = df.lazy().select((pl.col("a") & pl.col("b")).alias("and"),
                          (pl.col("a") | pl.col("b")).alias("or"),
                          (pl.col("a") ^ pl.col("b")).alias("xor"))
    check(lf, kinds=["Select"], order=True)


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_string_predicates(shape):
    df = frame("int32", shape)
    for pred in (pl.col("s").str.starts_with("a"), pl.col("s").str.contains("an", literal=True),
                 pl.col("s").str.contains("pp"), pl.col("s") == "apple", pl.col("s") != "apple",
                 pl.col("s").is_in(["a", "apple", "日本語"])):
        check(df.lazy().filter(pred), kinds=["Filter"])
    lf = df.lazy().select((pl.col("s") == "banana").alias("eq"),
                          pl.col("s").str.starts_with("ba").alias("sw"), "s", "k")
    check(lf, kinds=["Select"], order=True)


@pytest.mark.parametrize("shape", CORE, ids=CORE_IDS)
@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
def test_projection_chains_stay_one_kernel(dtype, shape):
    """with_columns -> filter -> select: the with_columns output stays virtual and is computed
    after the filter, inside the same fused kernel."""
    df = frame(dtype, shape)
    lf = (df.lazy().with_columns((pl.col("a") * 2 + pl.col("b")).alias("c"))
          .filter(pl.col("c") > pl.col("a")).select("c", "a", "k", "s"))
    eng = check(lf, kinds=["Filter", "HStack"], order=True)
    plan = eng.last_report.taken[0]["plan"]
    assert "with_columns" not in plan, plan


@pytest.mark.parametrize("shape", CORE, ids=CORE_IDS)
def test_slice_after_filter(shape):
    df = frame("int64", shape)
    check(df.lazy().filter(pl.col("a") > 0).slice(3, 17), kinds=["Slice"], order=True)
    check(df.lazy().filter(pl.col("a") > 0).head(5), kinds=["Filter"], order=True)


@pytest.mark.parametrize("shape", CORE, ids=CORE_IDS)
@pytest.mark.parametrize("carried", ["date", "datetime_us_tz", "duration_ms", "time"])
def test_temporal_columns_are_carried(carried, shape):
    df = frame("int64", shape)
    n = df.height
    base = pl.Series(np.arange(n, dtype=np.int64) * 86_400_000_000 - 10**15)
    t = {"date": base.cast(pl.Datetime("us")).cast(pl.Date),
         "datetime_us_tz": base.cast(pl.Datetime("us", "America/New_York")),
         "duration_ms": (base // 1000).cast(pl.Duration("ms")),
         "time": (base % 86_400_000_000_000).abs().cast(pl.Time)}[carried]
    df = df.with_columns(t.alias("t"))
    check(df.lazy().filter(pl.col("a") > 0).select("t", "a"), kinds=["Filter"], order=True)
    check(df.lazy().select("t", "a", "k").sort("a", descending=True), kinds=["Sort"], order=["a"])


# -- the semantics the plan marked "unverified", pinned


def test_float_comparisons_follow_polars_total_order():
    nan, inf = float("nan"), float("inf")
    for dt in (pl.Float32, pl.Float64):
        df = pl.DataFrame({"a": [1.0, nan, None, -0.0, 0.0, 2.0, nan, inf],
                           "b": [nan, nan, 1.0, 0.0, -0.0, None, 1.0, nan]},
                          schema={"a": dt, "b": dt})
        lf = df.lazy().select(**{k: op(pl.col("a"), pl.col("b")) for k, op in CMP_OPS.items()},
                              gt1=pl.col("a") > 1.0, le1=pl.col("a") <= 1.0)
        check(lf, order=True)


def test_ternary_with_a_null_condition_takes_otherwise():
    df = pl.DataFrame({"c": [True, None, False], "x": [1, 2, 3]})
    check(df.lazy().select(pl.when(pl.col("c")).then(pl.col("x")).otherwise(-1).alias("w")), order=True)


def test_integer_arithmetic_wraps_like_polars():
    df = pl.DataFrame({"a": [127, -128, 100], "b": [1, -1, 100]}, schema={"a": pl.Int8, "b": pl.Int8})
    check(df.lazy().select((pl.col("a") + pl.col("b")).alias("s"),
                           (pl.col("a") * pl.col("b")).alias("m"),
                           (pl.col("a") - pl.col("b")).alias("d")), order=True)
    big = pl.DataFrame({"a": [2**63 - 1, -(2**63)]}, schema={"a": pl.Int64})
    check(big.lazy().select((pl.col("a") + 1).alias("s"), (pl.col("a") * 3).alias("m")), order=True)


def test_is_in_of_a_null_is_null():
    df = pl.DataFrame({"k": [1, None, 3]})
    eng = check(df.lazy().select(pl.col("k").is_in([1, None]).alias("i")), order=True)
    assert "if_else" in eng.last_report.taken[0]["plan"]


def test_gpu_float32_arithmetic_flushes_subnormals_so_the_engine_computes_in_binary64():
    """The property the Float32 translation exists for: the GPU's float adds flush a subnormal to
    zero, so `(add a b)` over Float32 is not Polars' answer; the engine emits
    `(cast (add (cast a f64) (cast b f64)) f32)`, which is."""
    from arrowmetal import lazy
    tiny = pa.array(np.array([1.4e-45, 1.0], dtype=np.float32))
    zero = pa.array(np.array([0.0, 1.0], dtype=np.float32))
    plan = {"op": "select", "input": {"op": "scan", "source": "t"},
            "exprs": [["x", '(add (col "a") (col "b"))']]}
    got = lazy.LazyFrame(plan, {"t": lazy._Source(["a", "b"], [tiny, zero])}).collect()
    assert got.column("x").to_pylist()[0] == 0.0
    df = pl.DataFrame({"a": tiny, "b": zero})
    eng = check(df.lazy().select((pl.col("a") + pl.col("b")).alias("x")), order=True)
    assert "(cast (add (cast" in eng.last_report.taken[0]["plan"]


def test_true_division_by_zero_is_ieee():
    df = pl.DataFrame({"a": [1, 0, -1, None], "b": [0, 0, 0, 0]})
    check(df.lazy().select((pl.col("a") / pl.col("b")).alias("d")), order=True)


# =============================================================================================
# M2 -- aggregation, sort, top-k
# =============================================================================================


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("dtype", NUMERIC)
def test_group_by_aggregates(dtype, shape):
    if shape[0] == "special" and dtype in FLOATS:
        pytest.skip("sums over infinities and 1e308s depend on the summation order")
    df = frame(dtype, shape)
    aggs = [pl.col("a").sum().alias("sum"), pl.col("a").mean().alias("mean"),
            pl.col("a").count().alias("count"), pl.len().alias("len")]
    if dtype != "float64":
        aggs += [pl.col("a").min().alias("min"), pl.col("a").max().alias("max")]
    lf = df.lazy().group_by("k").agg(aggs)
    check(lf, kinds=["GroupBy"], rel=_float_rel(dtype) if dtype in FLOATS else 1e-12)


@pytest.mark.parametrize("shape", CORE, ids=CORE_IDS)
@pytest.mark.parametrize("dtype", NUMERIC)
def test_whole_frame_aggregates(dtype, shape):
    if shape[0] == "special" and dtype in FLOATS:
        pytest.skip("sums over infinities and 1e308s depend on the summation order")
    df = frame(dtype, shape)
    lf = df.lazy().select(pl.col("a").sum().alias("sum"), pl.col("a").mean().alias("mean"),
                          pl.col("a").min().alias("min"), pl.col("a").max().alias("max"),
                          pl.col("a").count().alias("count"), pl.len().alias("len"))
    check(lf, kinds=["Select"], rel=_float_rel(dtype) if dtype in FLOATS else 1e-12)
    check(df.lazy().filter(pl.col("k") > 3).select(pl.col("b").sum(), pl.len()), kinds=["Filter"],
          rel=_float_rel(dtype) if dtype in FLOATS else 1e-12)


@pytest.mark.parametrize("shape", CORE, ids=CORE_IDS)
def test_group_by_several_keys_and_key_types(shape):
    df = frame("int64", shape)
    check(df.lazy().group_by("k", "g").agg(pl.col("a").sum(), pl.len()), kinds=["GroupBy"])
    check(df.lazy().group_by("s").agg(pl.col("a").max(), pl.col("b").count()), kinds=["GroupBy"])
    check(df.lazy().filter(pl.col("a") > 0).group_by("k", "s").agg(pl.col("a").mean()),
          kinds=["GroupBy", "Filter"], rel=1e-12)


def test_aggregate_schema_contract():
    """Polars' dtypes, whatever ArrowMetal accumulates in: widened sums of small ints, UInt32
    count/len, Float64 mean, Float32 sum/mean of a Float32, UInt32 sum of a Boolean, and an Int32
    sum that wraps at Int32."""
    df = pl.DataFrame({"k": [1, 1, 2],
                       "i8": pl.Series([100, 100, -5], dtype=pl.Int8),
                       "u8": pl.Series([200, 200, 1], dtype=pl.UInt8),
                       "i16": pl.Series([30000, 30000, 1], dtype=pl.Int16),
                       "i32": pl.Series([2**31 - 1, 5, 1], dtype=pl.Int32),
                       "f32": pl.Series([1.5, 2.5, None], dtype=pl.Float32),
                       "b": [True, True, None]})
    aggs = [pl.col("i8").sum().alias("s_i8"), pl.col("u8").sum().alias("s_u8"),
            pl.col("i16").sum().alias("s_i16"), pl.col("i32").sum().alias("s_i32"),
            pl.col("f32").sum().alias("s_f32"), pl.col("f32").mean().alias("m_f32"),
            pl.col("b").sum().alias("s_b"), pl.col("i8").mean().alias("m_i8"),
            pl.col("i8").count().alias("c"), pl.len().alias("n")]
    check(df.lazy().group_by("k").agg(aggs), rel=1e-6)
    check(df.lazy().select(aggs), rel=1e-6)
    got = df.lazy().select(aggs).collect(engine=metal())
    assert got.schema["s_i8"] == pl.Int64 and got.schema["c"] == pl.UInt32
    assert got.schema["s_f32"] == pl.Float32 and got.schema["s_b"] == pl.UInt32
    assert got.schema["s_i32"] == pl.Int32 and got["s_i32"][0] == df["i32"].sum()


def test_count_of_a_boolean_per_group():
    """ArrowMetal's group-by does not read a Boolean even to count it; the engine counts the
    validity bits instead, as for Float64."""
    df = pl.DataFrame({"k": [1, 1, 2, 2, 3, None], "g": [True, None, False, None, None, True]})
    eng = check(df.lazy().group_by("k").agg(pl.col("g").count().alias("c"), pl.len()),
                kinds=["GroupBy"])
    assert "is_valid" in eng.last_report.taken[0]["plan"]
    check(df.lazy().select(pl.col("g").count()), kinds=["Select"])


def test_sum_over_nothing_is_zero_and_max_over_only_nan_is_nan():
    nan = float("nan")
    df = pl.DataFrame({"k": [1, 1, 2, 3], "v": [None, None, 3, None], "f": [nan, nan, 1.0, None]},
                      schema={"k": pl.Int64, "v": pl.Int64, "f": pl.Float32})
    check(df.lazy().group_by("k").agg(pl.col("v").sum().alias("s"), pl.col("v").count().alias("c"),
                                      pl.col("f").max().alias("fx"), pl.col("f").min().alias("fn"),
                                      pl.col("v").mean().alias("m")))
    check(df.lazy().filter(pl.col("k") > 5).select(pl.col("v").sum(), pl.col("f").max(), pl.len()))
    check(df.lazy().select(pl.col("f").max().alias("fx"), pl.col("f").min().alias("fn")))


SORT_DTYPES = ["int8", "uint16", "int32", "int64", "uint64", "float32", "float64", "bool"]


@pytest.mark.parametrize("shape", CORE, ids=CORE_IDS)
@pytest.mark.parametrize("dtype", SORT_DTYPES)
def test_sort_single_key(dtype, shape):
    df = frame(dtype, shape)
    for desc in (False, True):
        for nulls_last in (False, True):
            lf = df.lazy().sort("a", descending=desc, nulls_last=nulls_last)
            check(lf, kinds=["Sort"], order=["a"])


@pytest.mark.parametrize("shape", CORE, ids=CORE_IDS)
def test_sort_several_keys(shape):
    df = frame("float64", shape)
    lf = df.lazy().sort(["k", "a", "b"], descending=[True, False, True], nulls_last=[False, True, False])
    check(lf, kinds=["Sort"], order=["k", "a", "b"])
    lf = df.lazy().sort(["g", "k", "a"], descending=[True, False, True], nulls_last=[True, False, True])
    check(lf, kinds=["Sort"], order=["g", "k", "a"])
    lf = df.lazy().sort(["g", "b"], descending=[False, True])
    check(lf, kinds=["Sort"], order=["g", "b"])


@pytest.mark.parametrize("shape", CORE, ids=CORE_IDS)
def test_sort_stable_matches_maintain_order(shape):
    df = frame("int8", shape)
    lf = df.lazy().with_row_index("i").sort("k", maintain_order=True).drop("i")
    want = lf.collect()
    got = df.lazy().sort("k", maintain_order=True).collect(engine=metal())
    compare(got, want, order=True)


@pytest.mark.parametrize("shape", CORE, ids=CORE_IDS)
@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
def test_top_k(dtype, shape):
    df = frame(dtype, shape)
    for desc in (False, True):
        lf = df.lazy().sort("a", descending=desc, nulls_last=True).head(7)
        check(lf, kinds=["Sort"], keys_only=["a"])
        lf = df.lazy().sort("a", descending=desc).slice(2, 5)
        check(lf, kinds=["Sort"], keys_only=["a"])


@pytest.mark.parametrize("shape", CORE, ids=CORE_IDS)
def test_group_by_then_top_k_drops_the_dynamic_predicate(shape):
    """`sort().head()` over a group_by makes Polars insert a `dynamic_pred` hint filter between
    them (plan fact 10); the translator drops it and takes the whole plan."""
    df = frame("int64", shape)
    lf = (df.lazy().filter(pl.col("a") > 0).group_by("k").agg(pl.col("a").sum().alias("t"))
          .sort("t", descending=True).head(3))
    check(lf, kinds=["Sort", "GroupBy", "Filter"], keys_only=["t"])


@pytest.mark.parametrize("shape", CORE, ids=CORE_IDS)
def test_nan_sorts_above_numbers_in_both_directions(shape):
    nan = float("nan")
    df = pl.DataFrame({"f": [2.0, nan, None, -1.0, float("inf"), nan, 0.0]})
    for desc in (False, True):
        for nl in (False, True):
            check(df.lazy().sort("f", descending=desc, nulls_last=nl), order=True)


# =============================================================================================
# M3 -- joins and unique
# =============================================================================================

JOIN_HOWS = ["inner", "left", "semi", "anti"]


def _right_frame(dtype, shape):
    """A right side sharing k (Int32, nulls, duplicates), s (String) and g (Boolean) with `frame`,
    plus a colliding `a` and its own `w`."""
    right = frame(dtype, shape, seed=11).rename({"b": "w"})
    return right.select("k", "s", "g", "a", "w")


@pytest.mark.parametrize("shape", CORE, ids=CORE_IDS)
@pytest.mark.parametrize("how", JOIN_HOWS)
def test_join_kinds_on_an_int_key(how, shape):
    left, right = frame("int64", shape), _right_frame("int64", shape)
    check(left.lazy().join(right.lazy(), on="k", how=how), kinds=["Join"])


@pytest.mark.parametrize("shape", CORE, ids=CORE_IDS)
@pytest.mark.parametrize("how", JOIN_HOWS)
def test_join_kinds_on_string_and_multi_column_keys(how, shape):
    left, right = frame("int32", shape), _right_frame("int32", shape)
    check(left.lazy().join(right.lazy(), on="s", how=how), kinds=["Join"])
    check(left.lazy().join(right.lazy(), on=["k", "g"], how=how), kinds=["Join"])


@pytest.mark.parametrize("how", JOIN_HOWS)
def test_join_on_a_temporal_key(how):
    days = pl.Series(np.arange(-3000, 3000, 7, dtype=np.int32)).cast(pl.Date)
    left = pl.DataFrame({"d": days, "v": np.arange(len(days))}).with_columns(
        pl.when(pl.col("v") % 11 == 0).then(None).otherwise(pl.col("d")).alias("d"))
    right = pl.DataFrame({"d": days[::3], "w": np.arange(len(days[::3])) * 2})
    check(left.lazy().join(right.lazy(), on="d", how=how), kinds=["Join"])
    ts = left.with_columns(pl.col("d").cast(pl.Datetime("us", "UTC")))
    rts = right.with_columns(pl.col("d").cast(pl.Datetime("us", "UTC")))
    check(ts.lazy().join(rts.lazy(), on="d", how=how), kinds=["Join"])


@pytest.mark.parametrize("shape", CORE, ids=CORE_IDS)
def test_join_names_suffix_and_different_key_names(shape):
    left, right = frame("float32", shape), _right_frame("float32", shape)
    r2 = right.rename({"k": "kk"})
    check(left.lazy().join(r2.lazy(), left_on="k", right_on="kk", how="inner"), kinds=["Join"])
    check(left.lazy().join(r2.lazy(), left_on="k", right_on="kk", how="left", suffix="_r"),
          kinds=["Join"])


@pytest.mark.parametrize("shape", CORE, ids=CORE_IDS)
def test_join_inside_a_plan(shape):
    """Filters pushed into both sides, a projection and an aggregate above: one subtree."""
    left, right = frame("int64", shape), _right_frame("int64", shape)
    lf = (left.lazy().filter(pl.col("a") > 0)
          .join(right.lazy().filter(pl.col("w") < 0), on="k", how="inner")
          .select((pl.col("a") + pl.col("w")).alias("aw"), "k")
          .group_by("k").agg(pl.col("aw").sum(), pl.len()))
    check(lf, kinds=["Join", "GroupBy", "Filter"])


@pytest.mark.parametrize("shape", CORE, ids=CORE_IDS)
def test_unique(shape):
    df = frame("int16", shape)
    check(df.lazy().unique(subset=["k"], keep="first"), kinds=["Distinct"])
    check(df.lazy().unique(), kinds=["Distinct"])
    lf = df.lazy().unique(subset=["k", "g"], keep="any")
    want, got = lf.collect(), lf.collect(engine=metal())
    compare(got.select("k", "g"), want.select("k", "g"))        # which row of a group is Polars' choice


JOIN_FALLBACKS = {
    "right": (lambda l, r: l.join(r, on="k", how="right"), "Right join"),
    "full": (lambda l, r: l.join(r, on="k", how="full"), "Full join"),
    "cross": (lambda l, r: l.join(r.select("w"), how="cross"), "Cross join"),
    "nulls_equal": (lambda l, r: l.join(r, on="k", nulls_equal=True), "nulls_equal"),
    "float_key": (lambda l, r: l.join(r, on="a"), "float key"),
    "maintain_order": (lambda l, r: l.join(r, on="k", maintain_order="left"), "maintain_order"),
    "keep_last": (lambda l, r: l.unique(subset=["k"], keep="last"), "keep='last'"),
    "unique_ordered": (lambda l, r: l.unique(subset=["k"], maintain_order=True), "maintain_order"),
    "unique_float": (lambda l, r: l.unique(subset=["a"]), "float column"),
}


@pytest.mark.parametrize("case", list(JOIN_FALLBACKS))
def test_join_and_unique_fallbacks(case):
    build, reason = JOIN_FALLBACKS[case]
    shape = ("random", 4097, 0.3)
    left, right = frame("float64", shape), _right_frame("float64", shape)
    eng = check_fallback(build(left.lazy(), right.lazy()), reason)
    assert "Join" not in eng.last_report.kinds_taken() and "Distinct" not in eng.last_report.kinds_taken()


# =============================================================================================
# what stays with Polars, and says why
# =============================================================================================

FALLBACKS = {
    "modulus": (lambda lf: lf.select(pl.col("a") % 3), "Modulus"),
    "floor_div": (lambda lf: lf.select(pl.col("a") // 3), "FloorDivide"),
    "eq_missing": (lambda lf: lf.select(pl.col("a").eq_missing(pl.col("b"))), "EqValidity"),
    "ends_with": (lambda lf: lf.filter(pl.col("s").str.ends_with("a")), "ends_with"),
    "regex": (lambda lf: lf.filter(pl.col("s").str.contains("a.c")), "regex"),
    "narrowing_cast": (lambda lf: lf.select(pl.col("a").cast(pl.Int8, strict=False)),
                       "may lose values"),
    "is_in_nulls_equal": (lambda lf: lf.select(pl.col("k").is_in([1, None], nulls_equal=True)),
                          "nulls_equal"),
    "maintain_order_group_by": (lambda lf: lf.group_by("k", maintain_order=True).agg(pl.col("a").sum()),
                                "maintain_order"),
    "float_key": (lambda lf: lf.group_by("f").agg(pl.len()), "float key"),
    "f64_group_max": (lambda lf: lf.group_by("k").agg(pl.col("f").max()), "Float64"),
    "window": (lambda lf: lf.select(pl.col("a").sum().over("k")), "Window"),
    "rank": (lambda lf: lf.select(pl.col("a").rank()), "rank"),
    "tail": (lambda lf: lf.filter(pl.col("a") > 0).tail(3), "negative offset"),
    "sort_by_expr": (lambda lf: lf.sort(pl.col("a") * 2), "sort by an expression"),
    "stable_top_k": (lambda lf: lf.sort("a", maintain_order=True).head(3), "maintain_order"),
    "literals_only": (lambda lf: lf.select(pl.lit(1).alias("one")), "literals only"),
    "mixed_select": (lambda lf: lf.select(pl.col("a"), pl.col("a").sum().alias("t")), "mixing"),
    "median": (lambda lf: lf.group_by("k").agg(pl.col("a").median()), "median"),
    "agg_expression": (lambda lf: lf.group_by("k").agg(pl.col("a").sum() * 2), "over an aggregate"),
    "string_output": (lambda lf: lf.select(pl.col("s").str.to_uppercase()), "Function"),
    "key_expression": (lambda lf: lf.group_by(pl.col("k") * 2).agg(pl.len()), "key is an expression"),
}


@pytest.mark.parametrize("case", list(FALLBACKS))
def test_unsupported_expressions_fall_back_with_a_reason(case):
    build, reason = FALLBACKS[case]
    df = frame("int64", ("random", 4097, 0.3)).with_columns(pl.col("a").cast(pl.Float64).alias("f"))
    check_fallback(build(df.lazy()), reason, order=True if case in ("tail",) else None)


@pytest.mark.parametrize("dtype", [pl.Categorical, pl.Enum(["x", "y"]), pl.Decimal(10, 2),
                                   pl.List(pl.Int64), pl.Struct({"a": pl.Int64}), pl.Null,
                                   pl.Binary])
def test_unsupported_column_types_fall_back(dtype):
    values = {pl.Categorical: ["x", "y", None], pl.Null: [None, None, None],
              pl.Binary: [b"x", None, b"y"]}.get(dtype)
    if values is None:
        values = {"Enum": ["x", "y", None], "Decimal": [1, None, 3], "List": [[1], None, [2, 3]],
                  "Struct": [{"a": 1}, None, {"a": 3}]}[type(dtype).__name__]
    df = pl.DataFrame({"c": pl.Series(values, dtype=dtype), "v": [3, 1, 2]})
    check_fallback(df.lazy().sort("v"), "does not carry", order=True)


def test_other_file_scans_stay_with_polars(tmp_path):
    path = tmp_path / "t.parquet"
    pl.DataFrame({"v": [3, 1, 2]}).write_parquet(path)
    check(pl.scan_parquet(path).sort("v"), kinds=["Sort", "Scan"], order=True)
    check_fallback(pl.scan_pyarrow_dataset(__import__("pyarrow.dataset").dataset.dataset(path))
                   .sort("v"), "PythonScan", order=True)
    pl.DataFrame({"v": [3, 1, 2]}).write_csv(tmp_path / "t.csv")
    check_fallback(pl.scan_csv(tmp_path / "t.csv").sort("v"), "csv scan", order=True)


# =============================================================================================
# the default policy and the size gate
# =============================================================================================


def test_default_takes_a_large_simple_sort_and_leaves_the_rest():
    n = 1_000_000
    rng = np.random.default_rng(3)
    df = pl.DataFrame({"q": rng.integers(0, 10**9, n), "x": rng.random(n),
                       "k": rng.integers(0, 50, n).astype(np.int32)})
    eng = am.MetalEngine()
    lf = df.lazy().sort("q")
    compare(lf.collect(engine=eng), lf.collect(), order=["q"])
    assert [t["kinds"] for t in eng.last_report.taken] == [["Sort", "DataFrameScan"]]
    for lf, reason in ((df.lazy().group_by("k").agg(pl.col("x").sum()), "group_by"),
                       (df.lazy().filter(pl.col("x") > 0.5), "rowwise"),
                       (df.lazy().sort("q").head(10), "top_k"),
                       (df.lazy().sort("x", descending=True), "needs"),
                       (df.head(999_999).lazy().sort("q"), "needs"),
                       (df.with_columns(pl.col("k").cast(pl.String)).lazy().sort("q"), "String")):
        lf.collect(engine=eng)
        assert not eng.last_report.taken, eng.last_report
        assert any(reason in f for f in eng.last_report.fallbacks), (reason, eng.last_report)


def test_min_rows_gate():
    lf = pl.LazyFrame({"v": list(range(50))}).sort("v", descending=True)
    eng = pe.MetalEngine(min_rows=100, shapes="all")
    got = lf.collect(engine=eng)
    assert got.equals(lf.collect()) and not eng.last_report.taken
    assert any("below" in f for f in eng.last_report.fallbacks)
    eng = pe.MetalEngine(min_rows=50, shapes="all")
    lf.collect(engine=eng)
    assert eng.last_report.taken


# =============================================================================================
# the hand-off: zero copy, the import cache, and what it costs
# =============================================================================================


def test_single_chunk_numeric_input_reaches_metal_without_a_copy():
    n = 1_000_000
    df = pl.DataFrame({"v": np.arange(n, dtype=np.int64)[::-1].copy()})
    src, dst, same = am.zero_copy_report(df["v"])
    assert same, (src, dst)
    pe.clear_import_cache()
    df.lazy().sort("v").collect(engine=metal())
    arr = df["v"].to_arrow()
    cached = pe._cache.get(pe._cache.key(arr))
    assert cached is not None
    assert cached.to_arrow().buffers()[1].address == arr.buffers()[1].address


def test_the_import_cache_reuses_an_import_and_never_serves_stale_data():
    pe.clear_import_cache()
    n = 200_000
    df = pl.DataFrame({"v": np.arange(n, dtype=np.int64) % 977, "w": np.arange(n, dtype=np.int64)})
    lf = df.lazy().sort(["v", "w"])
    lf.collect(engine=metal())
    before = pe.import_cache_info()
    compare(lf.collect(engine=metal()), lf.collect(), order=True)
    after = pe.import_cache_info()
    assert after["hits"] >= before["hits"] + 2 and after["misses"] == before["misses"]
    # A new column is new memory: a miss, and the right answer.
    df2 = df.with_columns((pl.col("v") * 3 % 1013).alias("v"))
    lf2 = df2.lazy().sort(["v", "w"])
    compare(lf2.collect(engine=metal()), lf2.collect(), order=True)
    assert pe.import_cache_info()["misses"] > after["misses"]
    # A multi-chunk column is concatenated on every export, so it is never cached.
    pe.clear_import_cache()
    chunked = pl.concat([df, df], rechunk=False)
    chunked.lazy().sort("w").collect(engine=metal())
    assert pe.import_cache_info()["entries"] == 0
    # A budget of 0 disables it.
    old = pe.import_cache_limit()
    try:
        pe.import_cache_limit(0)
        lf.collect(engine=metal())
        assert pe.import_cache_info()["entries"] == 0
    finally:
        pe.import_cache_limit(old)
        pe.clear_import_cache()


def test_engine_overhead_over_the_resident_plan():
    """A warm MetalEngine collect against the same ArrowMetal plan run straight over GPU-resident
    columns: what the engine adds (the IR walk, the cache lookups, the Polars result) stays small.
    The bound is the one test_lazy.py holds the pyarrow path to; measured well under it."""
    import time
    n = 2_000_000
    rng = np.random.default_rng(9)
    df = pl.DataFrame({"q": rng.integers(0, 10**9, n), "k": rng.integers(0, 50, n).astype(np.int32)})
    lf = df.lazy().sort("q")
    eng = metal()
    lf.collect(engine=eng)
    plan = json.loads(eng.last_report.taken[0]["plan"])
    resident = {c: am.MetalArray.from_arrow(df[c].to_arrow()) for c in df.columns}
    from arrowmetal import lazy
    name = json.dumps(plan).split('"source": "')[1].split('"')[0]
    q = lazy.LazyFrame(plan, {name: lazy._Source(list(resident), list(resident.values()))})

    def best(fn, iters=9):
        fn()
        out = float("inf")
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            out = min(out, time.perf_counter() - t0)
        return out * 1000

    gpu = best(q.collect)
    engine = best(lambda: lf.collect(engine=eng))
    assert engine < gpu + 3.0, f"MetalEngine {engine:.2f} ms vs resident plan {gpu:.2f} ms"


# =============================================================================================
# core findings of this suite, fixed in ArrowMetal (the engine once worked around each)
# =============================================================================================


def _string_with_bytes_under_a_null():
    offsets = pa.py_buffer(np.array([0, 1, 7, 13], dtype=np.int64).tobytes())
    validity = pa.py_buffer(np.packbits(np.array([0, 1, 1, 0, 0, 0, 0, 0], dtype=np.uint8),
                                        bitorder="little").tobytes())
    return pa.LargeStringArray.from_buffers(3, offsets, pa.py_buffer(b"xcherrybanana"), validity,
                                            null_count=1)


def test_core_string_filter_null_slot():
    """Polars exports a null string slot with the bytes it held (valid Arrow). ArrowMetal's
    string compaction once copied them over the next value ('xanana' for 'banana')."""
    a = _string_with_bytes_under_a_null()
    got = am.MetalArray.from_arrow(a).filter(am.MetalArray.from_arrow(pa.array([True, False, True])))
    assert got.to_arrow().to_pylist() == [None, "banana"]


def test_engine_carries_bytes_under_null_strings():
    """A Polars String column whose null slots keep their bytes goes to ArrowMetal as exported."""
    s = pl.Series(np.random.default_rng(0).choice(["apple", "banana", "cherry", "x"], 5000))
    df = pl.DataFrame({"s": s.replace("x", None), "v": np.arange(5000)})
    check(df.lazy().filter(pl.col("v") > 10).select("s", "v"), kinds=["Filter"], order=True)
    check(df.lazy().filter(pl.col("v") > 10).sort("v", descending=True), kinds=["Sort"], order=True)


def test_core_bool_sort_null_count():
    from arrowmetal import lazy
    g = pa.array([None if i % 3 == 0 else i % 2 == 0 for i in range(33)], pa.bool_())
    a = pa.array(np.arange(33)[::-1].astype(np.int32))
    plan = {"op": "sort", "input": {"op": "scan", "source": "t"}, "by": [["a", False]]}
    out = lazy.LazyFrame(plan, {"t": lazy._Source(["a", "g"], [a, g])}).collect().column("g")
    assert out.null_count == g.null_count


def test_core_filter_carrying_a_date():
    from arrowmetal import lazy
    t = pa.array(range(10), pa.date32())
    a = pa.array(range(-5, 5), pa.int64())
    plan = {"op": "filter", "input": {"op": "scan", "source": "t"},
            "predicate": '(gt (col "a") (i64 0))'}
    out = lazy.LazyFrame(plan, {"t": lazy._Source(["t", "a"], [t, a])}).collect()
    assert out.num_rows == 4


def test_core_string_sort_with_nulls():
    """Run in a child process: before the fix, one run of this ended in a bus error rather than a
    wrong answer."""
    code = ("import pyarrow as pa, arrowmetal as am\n"
            "t = pa.table({'s': pa.array(['b', 'a', None, 'c', 'ab', 'xxxxxxxx'], pa.large_string())})\n"
            "got = am.scan(t).sort('s').collect().column('s').to_pylist()\n"
            "assert got == ['a', 'ab', 'b', 'c', 'xxxxxxxx', None], got\n")
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "python"))
    r = subprocess.run([sys.executable, "-c", code], env=env, cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]


@pytest.mark.parametrize("n", [33, 4097])
def test_engine_takes_the_plans_it_once_worked_around(n):
    """Each fixed finding above, now taken by the engine and equal to Polars: a sort by a String
    column holding nulls and rows of 8+ bytes, a Boolean column with nulls carried through a sort
    (its null count included), and a filter that carries temporal columns."""
    rng = np.random.default_rng(n)
    words = np.array(["b", "a", "c", "ab", "xxxxxxxx", "abcdefghij", "banana", "cherry"])
    df = pl.DataFrame({
        "s": pl.Series(rng.choice(words, n)).set(pl.Series(rng.random(n) < 0.2), None),
        "g": pl.Series([None if i % 3 == 0 else i % 2 == 0 for i in range(n)], dtype=pl.Boolean),
        "d": pl.Series(rng.integers(0, 20000, n), dtype=pl.Int32).cast(pl.Date),
        "t": pl.Series(rng.integers(0, 10**12, n)).cast(pl.Datetime("us")),
        "v": np.arange(n)[::-1].copy()})
    for desc in (False, True):
        for nl in (False, True):
            check(df.lazy().sort(["s", "v"], descending=desc, nulls_last=nl), kinds=["Sort"],
                  order=True)
    eng = check(df.lazy().sort("v"), kinds=["Sort"], order=True)
    got = df.lazy().sort("v").collect(engine=eng)["g"]
    assert got.null_count() == df["g"].null_count() > 0
    check(df.lazy().filter(pl.col("v") > n // 3), kinds=["Filter"], order=True)


# =============================================================================================
# Parquet scans: the file is read on the GPU and the subtree above it runs there
# =============================================================================================
#
# Every scan-taken shape is collected on Polars and through the engine over files written by
# pyarrow (three layouts), Polars and DuckDB, with nulls, NaN, -0.0 and infinities in the data, and
# over the flat columns of the nested fixtures. The predicate-pushdown cases check both the answer
# and what the reader skipped; the fallback cases check the reason in the report.

NESTED = os.path.join(REPO, "Tests", "Fixtures", "nested")
SCAN_N = 20_011
SCAN_RG = 4096


def _scan_table(n=SCAN_N, seed=5):
    rng = np.random.default_rng(seed)

    def mask(p):
        return rng.random(n) < p

    f64 = rng.standard_normal(n) * 4
    f64[rng.random(n) < 0.03] = np.nan
    f64[::997] = -0.0
    f64[5::1999] = np.inf
    f32 = (rng.standard_normal(n) * 4).astype(np.float32)
    f32[rng.random(n) < 0.03] = np.nan
    f32[7::1499] = -np.inf
    words = np.array([f"s{i}" for i in range(40)] + ["", "a longer string value"])
    return pa.table({
        "id": pa.array(np.arange(n, dtype=np.int64)),
        "k": pa.array(rng.integers(0, 9, n).astype(np.int32), mask=mask(0.1)),
        "i8": pa.array(rng.integers(-128, 128, n).astype(np.int8), mask=mask(0.2)),
        "i16": pa.array(rng.integers(-2**15, 2**15, n).astype(np.int16), mask=mask(0.2)),
        "i32": pa.array(rng.integers(0, 10, n).astype(np.int32), mask=mask(0.2)),
        "i64": pa.array(rng.integers(-2**40, 2**40, n), mask=mask(0.2)),
        "u8": pa.array(rng.integers(0, 256, n).astype(np.uint8), mask=mask(0.2)),
        "u16": pa.array(rng.integers(0, 2**16, n).astype(np.uint16), mask=mask(0.2)),
        "u32": pa.array(rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32), mask=mask(0.2)),
        "u64": pa.array(rng.integers(0, 2**63, n, dtype=np.uint64) * np.uint64(2), mask=mask(0.2)),
        "f32": pa.array(f32, mask=mask(0.1)),
        "f64": pa.array(f64, mask=mask(0.1)),
        "b": pa.array(rng.random(n) < 0.5, mask=mask(0.2)),
        "s": pa.array(rng.choice(words, n), mask=mask(0.2)),
        "d": pa.array(rng.integers(0, 20000, n).astype(np.int32), mask=mask(0.2)).cast(pa.date32()),
        "ts": pa.array(rng.integers(0, 10**15, n), mask=mask(0.2)).cast(pa.timestamp("us")),
    })


def _write_duckdb(table, path, row_group_size):
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.register("t", table)
    con.execute(f"COPY (SELECT * FROM t) TO '{path}' (FORMAT parquet, ROW_GROUP_SIZE {row_group_size})")
    con.close()


def _write_all(table, d, stem, rg):
    import pyarrow.parquet as pq
    paths = {}
    p = str(d / f"{stem}_pa.parquet")
    pq.write_table(table, p, row_group_size=rg, compression="snappy")
    paths["pyarrow"] = p
    p = str(d / f"{stem}_pa_index_plain.parquet")
    pq.write_table(table, p, row_group_size=rg, compression="none", use_dictionary=False,
                   write_page_index=True, data_page_size=2048)
    paths["pyarrow_page_index_plain"] = p
    p = str(d / f"{stem}_pa_v2_zstd.parquet")
    pq.write_table(table, p, row_group_size=rg, compression="zstd", data_page_version="2.0")
    paths["pyarrow_v2_zstd"] = p
    p = str(d / f"{stem}_polars.parquet")
    pl.from_arrow(table).write_parquet(p, row_group_size=rg)
    paths["polars"] = p
    p = str(d / f"{stem}_duckdb.parquet")
    _write_duckdb(table, p, rg)
    paths["duckdb"] = p
    return paths


@pytest.fixture(scope="module")
def scan_files(tmp_path_factory):
    return _write_all(_scan_table(), tmp_path_factory.mktemp("scan"), "t", SCAN_RG)


SCAN_WRITERS = ["pyarrow", "pyarrow_page_index_plain", "pyarrow_v2_zstd", "polars", "duckdb"]

# name -> (plan over a scan, check() keywords)
SCAN_SHAPES = {
    "filter_select": (lambda lf: lf.filter((pl.col("id") >= 5000) & (pl.col("i64") > 0))
                      .select("id", "i64", "s", "d", "ts", "b"), {"order": True}),
    "filter_every_dtype": (lambda lf: lf.filter(
        (pl.col("i8") < 50) & (pl.col("i16") > -1000) & (pl.col("u16") >= 100)
        & (pl.col("u32") <= 3_000_000_000) & (pl.col("u64") > 2**60) & (pl.col("f32") < 1.5)
        & (pl.col("f64") > -1.0) & pl.col("b")), {"order": True}),
    "float_total_order": (lambda lf: lf.filter((pl.col("f64") >= 2.0) | (pl.col("f32") == 0.0))
                          .select("id", "f64", "f32"), {"order": True}),
    "not_equal": (lambda lf: lf.filter((pl.col("i32") != 5) & (pl.col("f64") != 0.0))
                  .select("id", "i32", "f64"), {"order": True}),
    "strings": (lambda lf: lf.filter((pl.col("s") == "s7") | pl.col("s").str.starts_with("a l"))
                .select("id", "s"), {"order": True}),
    "with_columns": (lambda lf: lf.with_columns((pl.col("i32") * 2 + pl.col("i16")).alias("x"),
                                                (pl.col("f64") / 4.0).alias("y"))
                     .filter(pl.col("x") > 0).select("id", "x", "y"), {"order": True}),
    "group_by": (lambda lf: lf.filter(pl.col("id") < 12_000).group_by("k").agg(
        pl.col("i64").sum().alias("s64"), pl.col("i32").min().alias("mn"),
        pl.col("i16").max().alias("mx"), pl.col("f32").mean().alias("m32"),
        pl.col("u32").count().alias("c"), pl.len().alias("n")), {"rel": 1e-5}),
    "group_by_two_keys": (lambda lf: lf.group_by("k", "i32").agg(pl.col("i64").sum(), pl.len()),
                          {}),
    "aggregate": (lambda lf: lf.filter(pl.col("f64") <= 0.0).select(
        pl.col("i64").sum().alias("s"), pl.col("f64").max().alias("mx"),
        pl.col("u64").min().alias("mn"), pl.col("i16").mean().alias("m"), pl.len().alias("n")),
        {"rel": 1e-9}),
    "sort": (lambda lf: lf.select("id", "k", "f64", "s").sort(["k", "f64"], descending=[False, True]),
             {"order": ["k", "f64"]}),
    "sort_nulls_last": (lambda lf: lf.select("id", "i64", "d").sort("i64", nulls_last=True),
                        {"order": ["i64"]}),
    "top_k": (lambda lf: lf.select("id", "i64").sort("i64", descending=True).head(25),
              {"keys_only": ["i64"]}),
    "join": (lambda lf: lf.select("k", "i64").join(
        pl.LazyFrame({"k": pl.Series(range(9), dtype=pl.Int32), "w": range(10, 19)}), on="k")
        .select(pl.col("i64").sum().alias("s"), pl.col("w").sum().alias("w")), {}),
    "unique": (lambda lf: lf.select("k", "i32").unique(), {}),
}


@pytest.mark.parametrize("shape", list(SCAN_SHAPES))
@pytest.mark.parametrize("writer", SCAN_WRITERS)
def test_parquet_scan_shapes_against_polars(scan_files, writer, shape):
    build, kw = SCAN_SHAPES[shape]
    lf = build(pl.scan_parquet(scan_files[writer]))
    eng = check(lf, kinds=["Scan"], **kw)
    (entry,) = eng.last_report.taken
    (scan,) = entry["scans"]
    assert scan["path"] == scan_files[writer]


def _nested_cases():
    out = []
    for name in sorted(os.listdir(NESTED)):
        if not name.endswith(".parquet"):
            continue
        path = os.path.join(NESTED, name)
        try:
            schema = pl.scan_parquet(path).collect_schema()
        except BaseException:              # noqa: BLE001 -- a Polars panic on a crafted file
            continue
        flat = [c for c, dt in schema.items() if pe._carryable(dt)]
        numeric = [c for c in flat if pe._code(schema[c]) not in (None, "bool")]
        if numeric and flat != list(schema):
            out.append((name, flat, numeric[0]))
    return out


NESTED_CASES = _nested_cases()


@pytest.mark.parametrize("name,flat,key", NESTED_CASES, ids=[c[0] for c in NESTED_CASES])
def test_parquet_scan_over_the_nested_fixtures(name, flat, key):
    """The flat columns of every nested fixture, by every writer: a filter, a sort and an
    aggregate on the GPU equal Polars; the whole file, which holds a nested column, stays with
    Polars and says why."""
    path = os.path.join(NESTED, name)
    lf = pl.scan_parquet(path)
    try:
        want = lf.select(flat).collect()
    except BaseException:                  # noqa: BLE001
        pytest.skip("Polars cannot read this file's flat columns")
    mid = want[key].drop_nulls().sort()
    cut = mid[len(mid) // 2] if len(mid) else 0
    if isinstance(cut, float) and cut != cut:
        cut = 0.0
    check(lf.select(flat).filter(pl.col(key) >= cut), kinds=["Scan"], order=True)
    check(lf.select(flat).sort(key, descending=True, nulls_last=True), kinds=["Scan"],
          order=[key])
    check(lf.select(pl.col(key).max().alias("mx"), pl.col(key).count().alias("c"), pl.len()),
          kinds=["Scan"])
    check_fallback(lf.filter(pl.col(key) >= cut), "does not carry")


# -- predicate pushdown: what the reader skips, and that the answers stay Polars'

PUSH_RG = 2048          # DuckDB rounds a row group to 2048 rows


def _push_table():
    """Five row groups of `PUSH_RG` rows. `f`: [0, 1) with NaN rows, then [10, 11), then only NaN,
    then only null, then 1.5 everywhere but one NaN (the `!=` case of apache/arrow#51491). `c` is
    7 in the first row group and 0..9 after it; `s` is "a" in the first row group only."""
    rng = np.random.default_rng(17)
    g = PUSH_RG
    f = np.concatenate([rng.random(g), 10 + rng.random(g), np.full(g, np.nan), np.zeros(g),
                        np.full(g, 1.5)])
    f[: g : 50] = np.nan
    f[4 * g + 10] = np.nan
    fmask = np.zeros(5 * g, dtype=bool)
    fmask[3 * g: 4 * g] = True
    c = np.concatenate([np.full(g, 7), rng.integers(0, 10, 4 * g)]).astype(np.int64)
    s = np.array(["a"] * g + ["b"] * (4 * g), dtype=object)
    return pa.table({"id": pa.array(np.arange(5 * g, dtype=np.int64)),
                     "f": pa.array(f, mask=fmask),
                     "f32": pa.array(f.astype(np.float32), mask=fmask),
                     "c": pa.array(c), "s": pa.array(s, pa.string())})


@pytest.fixture(scope="module")
def push_files(tmp_path_factory):
    return _write_all(_push_table(), tmp_path_factory.mktemp("push"), "p", PUSH_RG)


def _scan_entry(lf, **kw):
    eng = check(lf, kinds=["Scan"], **kw)
    (entry,) = eng.last_report.taken
    (scan,) = entry["scans"]
    return scan


def _skipped(scan):
    return (scan["row_groups_skipped_by_statistics"] + scan["row_groups_skipped_by_page_index"]
            + scan["row_groups_skipped_by_bloom_filter"])


# (predicate, the filters the reader gets, fewest row groups it must skip)
PUSH_CASES = {
    "id_range": (pl.col("id") < PUSH_RG, [("id", "<", PUSH_RG)], 4),
    "id_flipped": (PUSH_RG * 4 <= pl.col("id"), [("id", ">=", PUSH_RG * 4)], 4),
    "int_not_equal_constant_group": (pl.col("c") != 7, [("c", "!=", 7)], 1),
    "string_equal": (pl.col("s") == "a", [("s", "==", "a")], 4),
    "string_not_equal": (pl.col("s") != "b", [("s", "!=", "b")], 4),
    "float_lt": (pl.col("f") < 5.0, [("f", "<", 5.0)], 1),
    "float_le": (pl.col("f") <= 0.5, [("f", "<=", 0.5)], 1),
    "float_eq": (pl.col("f") == 1.5, [("f", "==", 1.5)], 2),
    "float32_lt": (pl.col("f32") < 5.0, [("f32", "<", 5.0)], 1),
    # NaN is above every number in Polars' order: `>`, `>=` and `!=` keep NaN rows, which min/max
    # leave out, so none of them reaches the reader.
    "float_gt_keeps_nan": (pl.col("f") > 5.0, [], 0),
    "float_ge_keeps_nan": (pl.col("f") >= 10.0, [], 0),
    "float_ne_keeps_nan_51491": (pl.col("f") != 1.5, [], 0),
    "float32_ne_keeps_nan_51491": (pl.col("f32") != 1.5, [], 0),
    # Polars types a float literal against a Float32 column as Float32, and so does the filter;
    # a Float64 literal makes Polars compare in Float64 over a cast column, which stays on the GPU.
    "float32_literal": (pl.col("f32") < 0.1, [("f32", "<", float(np.float32(0.1)))], 1),
    "float32_against_float64": (pl.col("f32") < pl.lit(0.1, pl.Float64), [], 0),
    "conjunction": ((pl.col("id") >= PUSH_RG) & (pl.col("f") < 5.0) & (pl.col("f") > 0.25),
                    [("id", ">=", PUSH_RG), ("f", "<", 5.0)], 2),
    "or_is_not_pushed": ((pl.col("id") < 10) | (pl.col("c") == 3), [], 0),
}


@pytest.mark.parametrize("case", list(PUSH_CASES))
@pytest.mark.parametrize("writer", SCAN_WRITERS)
def test_parquet_predicate_pushdown(push_files, writer, case):
    pred, filters, skip = PUSH_CASES[case]
    lf = pl.scan_parquet(push_files[writer]).filter(pred).select("id", "f", "f32", "c", "s")
    scan = _scan_entry(lf, order=True)
    assert sorted(scan["filters"]) == sorted(filters)       # Polars may reorder the conjuncts
    assert _skipped(scan) >= skip, scan
    if "keeps_nan" in case:
        got = lf.collect(engine=metal())
        assert got["f"].is_nan().sum() > 0          # NaN rows are in Polars' answer, and in ours
        if "51491" in case:
            assert 4 * PUSH_RG + 10 in got["id"].to_list()


def test_a_filter_polars_left_above_the_scan_is_pushed_too(push_files):
    """With Polars' own predicate pushdown off, the Filter sits on the Scan; its comparisons still
    reach the reader."""
    lf = pl.scan_parquet(push_files["pyarrow"]).filter(pl.col("id") < 100)
    opts = pl.QueryOptFlags(predicate_pushdown=False)
    want = lf.collect(optimizations=opts)
    eng = metal()
    got = lf.collect(engine=eng, optimizations=opts)
    compare(got, want, order=True)
    (entry,) = eng.last_report.taken
    assert entry["kinds"] == ["Filter", "Scan"]
    assert entry["scans"][0]["filters"] == [("id", "<", 100)]
    assert _skipped(entry["scans"][0]) == 4


def test_use_statistics_false_hands_the_reader_no_filters(push_files):
    lf = pl.scan_parquet(push_files["pyarrow"], use_statistics=False).filter(pl.col("id") < 100)
    scan = _scan_entry(lf, order=True)
    assert scan["filters"] == [] and _skipped(scan) == 0


@pytest.mark.parametrize("col", ["f64", "f32"])
def test_not_equal_keeps_the_nan_in_a_constant_page_fixture(col):
    """pageindexnan__pa_constpage: a page of 5.0 with one NaN (row 10), whose page index says
    5.0 .. 5.0. Polars keeps row 10 under `!= 5.0`, and so does the engine."""
    lf = pl.scan_parquet(os.path.join(NESTED, "pageindexnan__pa_constpage.parquet"))
    lf = lf.filter(pl.col(col) != 5.0).select("id", col)
    scan = _scan_entry(lf, order=True)
    assert scan["filters"] == []
    assert 10 in lf.collect(engine=metal())["id"].to_list()
    lf = pl.scan_parquet(os.path.join(NESTED, "pageindexnan__pa_constpage.parquet"))
    scan = _scan_entry(lf.filter(pl.col("i64") != 5).select("id", "i64"), order=True)
    assert scan["filters"] == [("i64", "!=", 5)] and scan["pages_skipped"] >= 2


@pytest.mark.parametrize("name", ["pageindexnan__polars.parquet", "pageindexnan__pa_plain_none.parquet"])
@pytest.mark.parametrize("pred", [pl.col("f64") < 0.0, pl.col("f64") <= -1.5, pl.col("f64") == 2.0,
                                  pl.col("f64") > 3.0, pl.col("f32") < 1.0])
def test_nan_pages_flagged_by_polars_are_read(name, pred):
    lf = pl.scan_parquet(os.path.join(NESTED, name)).filter(pred)
    _scan_entry(lf, order=True)


# -- what stays with Polars, and why


def test_parquet_scan_fallbacks_name_the_reason(scan_files, tmp_path):
    p = scan_files["pyarrow"]
    q = str(tmp_path / "second.parquet")
    pl.scan_parquet(p).head(100).collect().write_parquet(q)
    q2 = str(tmp_path / "third.parquet")
    pl.scan_parquet(p).head(100).collect().write_parquet(q2)
    hive = tmp_path / "hive" / "part=1"
    hive.mkdir(parents=True)
    pl.scan_parquet(p).select("id", "i64").head(100).collect().write_parquet(hive / "x.parquet")
    csv = str(tmp_path / "t.csv")
    pl.scan_parquet(p).select("id", "i64").collect().write_csv(csv)
    ipc = str(tmp_path / "t.arrow")
    pl.scan_parquet(p).select("id", "i64").collect().write_ipc(ipc)
    dec = str(tmp_path / "dec.parquet")
    pl.DataFrame({"x": pl.Series([1, 2, None], dtype=pl.Decimal(10, 2)), "v": [3, 1, 2]}).write_parquet(dec)
    lst = str(tmp_path / "lst.parquet")
    pl.DataFrame({"x": [[1], None, [2, 3]], "v": [3, 1, 2]}).write_parquet(lst)
    cat = str(tmp_path / "cat.parquet")
    pl.DataFrame({"x": pl.Series(["a", None, "b"], dtype=pl.Categorical), "v": [3, 1, 2]}).write_parquet(cat)
    gt = pl.col("id") > 3
    cases = [
        (pl.scan_parquet(p).head(100).filter(gt), "row limit pushed into the scan", True),
        (pl.scan_parquet(p).tail(100).filter(gt), "row limit pushed into the scan", True),
        (pl.scan_parquet(p, row_index_name="ri").filter(gt), "row index", True),
        (pl.scan_parquet(p, include_file_paths="path").filter(gt), "include_file_paths", True),
        (pl.scan_parquet([p, q]).filter(gt), "scan of 2 files", None),
        (pl.scan_parquet(str(tmp_path / "*d.parquet")).filter(gt), "scan of 2 files", None),
        (pl.scan_parquet(str(tmp_path / "hive")).filter(gt), "Hive partition columns", None),
        (pl.scan_parquet("file://" + p).filter(gt), "is not a local file", True),
        (pl.scan_parquet(p, schema=pl.scan_parquet(p).collect_schema()).filter(gt), "schema=", True),
        (pl.scan_csv(csv).filter(gt), "csv scan", True),
        (pl.scan_ipc(ipc).filter(gt), "does not show this node", True),
        (pl.scan_parquet(dec).sort("v"), "does not carry", True),
        (pl.scan_parquet(lst).sort("v"), "does not carry", True),
        (pl.scan_parquet(cat).sort("v"), "does not carry", True),
        (pl.scan_parquet(p).filter(pl.col("i64") % 7 == 1), "Operator Modulus", True),
    ]
    for lf, reason, order in cases:
        eng = check_fallback(lf, reason, order=order)
        assert not eng.last_report.taken, eng.last_report


def test_a_stored_schema_polars_applies_and_arrowmetal_ignores_falls_back():
    """arrowschema__duckdb_fewer / _more hold an ARROW:schema with a different number of fields.
    Polars applies it (and names or types columns after it); ArrowMetal's reader ignores it, as
    Arrow's does. The engine sees the difference and leaves the scan to Polars."""
    more = pl.scan_parquet(os.path.join(NESTED, "arrowschema__duckdb_more.parquet"))
    check_fallback(more.sort("x0"), "is not a column of the file", order=True)
    fewer = pl.scan_parquet(os.path.join(NESTED, "arrowschema__duckdb_fewer.parquet"))
    eng = check_fallback(fewer.select("ts_paris").sort("ts_paris", nulls_last=True), "rejected",
                         order=True)
    assert "returned schema" in " ".join(eng.last_report.fallbacks)


def test_a_bare_scan_stays_with_polars_reader(scan_files):
    """A scan with no GPU work above it is Polars' to read: moving it would only add the export."""
    eng = pe.MetalEngine(min_rows=0, shapes="all")
    lf = pl.scan_parquet(scan_files["polars"]).select("id", "i64")
    compare(lf.collect(engine=eng), lf.collect(), order=True)
    assert not eng.last_report.taken and not eng.last_report.fallbacks


def test_a_scan_joined_with_a_scan_runs_as_one_subtree(scan_files):
    left = pl.scan_parquet(scan_files["pyarrow"]).select("id", "k", "i64")
    right = pl.scan_parquet(scan_files["duckdb"]).select("id", "f64").filter(pl.col("id") < 3000)
    eng = check(left.join(right, on="id").select(pl.col("i64").sum(), pl.len()), kinds=["Join"])
    (entry,) = eng.last_report.taken
    assert len(entry["scans"]) == 2 and entry["kinds"].count("Scan") == 2


# -- the open-file cache


def test_the_parquet_file_cache_reuses_the_open_file(scan_files):
    p = scan_files["pyarrow"]
    am.clear_parquet_cache()
    lf = pl.scan_parquet(p).filter(pl.col("id") < 100).select("id", "i64")
    lf.collect(engine=metal())
    info = am.parquet_cache_info()
    assert info["entries"] == 1 and info["files"] == [os.path.realpath(p)]
    misses, hits = info["misses"], info["hits"]
    compare(lf.collect(engine=metal()), lf.collect(), order=True)
    info = am.parquet_cache_info()
    assert info["misses"] == misses and info["hits"] > hits
    assert info["bytes"] == os.path.getsize(p)
    # read_parquet(cache=True) goes through the same entry.
    got = am.read_parquet(p, columns=["i64"], cache=True)["i64"].to_arrow()
    assert got.equals(am.read_parquet(p, columns=["i64"])["i64"].to_arrow())
    assert am.parquet_cache_info()["misses"] == misses
    am.clear_parquet_cache()
    assert am.parquet_cache_info()["entries"] == 0


def test_the_parquet_file_cache_is_invalidated_when_the_file_changes(tmp_path):
    p = str(tmp_path / "changing.parquet")

    def total():
        # A new LazyFrame each time: Polars keeps the file's metadata in the one it scanned.
        lf = pl.scan_parquet(p).filter(pl.col("v") > 1).select(pl.col("v").sum())
        got = lf.collect(engine=metal())
        compare(got, lf.collect())
        return got.item()

    pl.DataFrame({"v": [1, 2, 3]}).write_parquet(p)
    am.clear_parquet_cache()
    assert total() == 5
    before = am.parquet_cache_info()
    # New content, new size.
    pl.DataFrame({"v": [5, 6, 7, 8, 9, 10]}).write_parquet(p)
    assert total() == 45
    after = am.parquet_cache_info()
    assert after["invalidations"] == before["invalidations"] + 1 and after["entries"] == 1
    # Same size, new content: the modification time moves.
    st = os.stat(p)
    pl.DataFrame({"v": [5, 6, 7, 8, 9, 11]}).write_parquet(p)
    assert os.path.getsize(p) == st.st_size
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert total() == 46
    # Only the modification time moves: the file is opened again all the same.
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000))
    inv = am.parquet_cache_info()["invalidations"]
    assert total() == 46
    assert am.parquet_cache_info()["invalidations"] == inv + 1
    # Replaced by another file (a new inode) under the same name.
    other = str(tmp_path / "other.parquet")
    pl.DataFrame({"v": [100, 200]}).write_parquet(other)
    os.replace(other, p)
    assert total() == 300
    assert am.parquet_cache_info()["entries"] == 1
    am.clear_parquet_cache()


def test_the_parquet_file_cache_is_bounded(tmp_path):
    paths = []
    for i in range(3):
        p = str(tmp_path / f"f{i}.parquet")
        pl.DataFrame({"v": np.arange(1000 * (i + 1))}).write_parquet(p)
        paths.append(p)
    old = am.parquet_cache_limit()
    am.clear_parquet_cache()
    try:
        assert am.parquet_cache_limit(max_entries=2) == (2, old[1])
        for p in paths:
            am.read_parquet(p, cache=True)
        info = am.parquet_cache_info()
        assert info["entries"] == 2 and info["files"] == [os.path.realpath(q) for q in paths[1:]]
        # Least recently used goes first.
        am.read_parquet(paths[1], cache=True)
        am.read_parquet(paths[0], cache=True)
        assert am.parquet_cache_info()["files"] == [os.path.realpath(q) for q in (paths[1], paths[0])]
        # A byte budget below a file's size keeps it out, and the read still works.
        am.parquet_cache_limit(max_entries=16, max_bytes=os.path.getsize(paths[2]) - 1)
        assert len(am.read_parquet(paths[2], cache=True)["v"]) == 3000
        assert os.path.realpath(paths[2]) not in am.parquet_cache_info()["files"]
        # Shrinking the budget evicts.
        am.parquet_cache_limit(max_bytes=0)
        assert am.parquet_cache_info()["entries"] == 0
        # max_entries=0 turns it off.
        am.parquet_cache_limit(max_entries=0, max_bytes=old[1])
        am.read_parquet(paths[0], cache=True)
        assert am.parquet_cache_info()["entries"] == 0
    finally:
        am.parquet_cache_limit(*old)
        am.clear_parquet_cache()


# -- nullability from the footer


def test_column_null_count_reads_the_footer(scan_files):
    """`ParquetFile.column_null_count` is the footer's word, summed over row groups, and equals the
    data's null count for every writer that records it."""
    import pyarrow.parquet as pq
    exact = 0
    for writer in SCAN_WRITERS:
        p = scan_files[writer]
        f = am.ParquetFile(p)
        t = pq.read_table(p)
        for c in ("id", "k", "i8", "s", "f64", "d", "ts", "b"):
            got = f.column_null_count(c)
            assert got in (None, t[c].null_count), (writer, c, got)
            exact += got is not None
    assert exact >= 3 * 8
    f = am.ParquetFile(os.path.join(NESTED, "structs__pa_plain_none.parquet"))
    assert f.column_null_count("k") == 0 and f.column_null_count("s") is None
    with pytest.raises(am.ArrowMetalError, match="no top-level column"):
        f.column_null_count("nope")


def test_a_column_the_footer_says_has_no_nulls_needs_no_null_handling(scan_files):
    """`id` holds no null, so a descending sort by it needs no validity key: the plan is the sort
    alone, and the default engine counts it as a plain sort."""
    lf = pl.scan_parquet(scan_files["pyarrow"]).select("id", "i64").sort("id", descending=True)
    eng = check(lf, kinds=["Sort", "Scan"], order=True)
    assert "valid" not in eng.last_report.taken[0]["plan"]
    lf = pl.scan_parquet(scan_files["pyarrow"]).select("id", "i64").sort("i64", descending=True)
    eng = check(lf, kinds=["Sort", "Scan"], order=["i64"])
    assert "__arrowmetal_valid" in eng.last_report.taken[0]["plan"]


def test_statistics_that_disagree_with_the_data_are_an_error(scan_files, monkeypatch):
    """The plan trusts the footer's null counts; a file whose data holds nulls its statistics deny
    fails the query instead of answering differently from Polars."""
    monkeypatch.setattr(am.ParquetFile, "column_null_count", lambda self, c: 0)
    am.clear_parquet_cache()
    lf = pl.scan_parquet(scan_files["pyarrow"]).filter(pl.col("i64") > 0).select("id", "i64")
    with pytest.raises(am.ArrowMetalError, match="holds no null"):
        lf.collect(engine=pe.MetalEngine(min_rows=0, shapes="all"))
    am.clear_parquet_cache()


def test_the_default_takes_a_large_sort_over_a_parquet_scan_and_leaves_the_rest(tmp_path):
    """The defaults apply to scans as to in-memory frames: a full sort of at least 1,000,000 rows
    whose keys need no helper column runs on Metal; group-by, a sort carrying a String column and a
    smaller file stay with Polars, with the reason in the report."""
    import pyarrow.parquet as pq
    n = 1_000_000
    rng = np.random.default_rng(4)
    p = str(tmp_path / "big.parquet")
    pq.write_table(pa.table({"q": rng.integers(0, 10**9, n), "x": rng.random(n),
                             "k": rng.integers(0, 50, n).astype(np.int32),
                             "s": pa.array(rng.integers(0, 99, n).astype(str))}), p)
    small = str(tmp_path / "small.parquet")
    pq.write_table(pq.read_table(p).slice(0, n - 1), small)
    eng = am.MetalEngine()
    lf = pl.scan_parquet(p).select("q", "x").sort("q")
    compare(lf.collect(engine=eng), lf.collect(), order=["q"])
    assert [t["kinds"] for t in eng.last_report.taken] == [["Sort", "Scan"]]
    for lf, reason in ((pl.scan_parquet(p).group_by("k").agg(pl.col("x").sum()), "group_by"),
                       (pl.scan_parquet(p).filter(pl.col("x") > 0.5), "rowwise"),
                       (pl.scan_parquet(p).select("q", "s").sort("q"), "String"),
                       (pl.scan_parquet(small).select("q", "x").sort("q"), "below")):
        lf.collect(engine=eng)
        assert not eng.last_report.taken, eng.last_report
        assert any(reason in f for f in eng.last_report.fallbacks), (reason, eng.last_report)
