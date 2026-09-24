"""pytest plugin: every `LazyFrame.collect()` in a run is also collected through `MetalEngine` and
the two results must agree.

    PYTHONPATH=python python -m pytest python/tests/test_lazy.py python/tests/test_polars.py \
        -p metal_engine_everywhere -q

Loaded by `test_polars_engine.py::test_existing_polars_suites_collect_identically`, which runs the
tier-1..3 and lazy-engine suites under it. A collect that names an engine, runs in the background or
streams is left alone; every other one runs twice -- Polars as asked, then
`MetalEngine(min_rows=0, shapes="all")` -- and the test that called it fails if the frames differ.
The Polars result is what the test sees, so the suites' own assertions are unchanged.

Both frames are sorted by every column before comparing: Polars promises no group or join order, and
a sort leaves ties in an unspecified order on both sides (ordering is checked by
`test_polars_engine.py` itself, with keys that have no ties). Floats compare with a
relative tolerance of 1e-9 (Float64) / 1e-5 (Float32), because GPU and CPU sum in a different order.

At the end of the session the counts go to `$AM_EVERYWHERE_STATS` as JSON: how many collects were
compared and how many of them ran some subtree on Metal.
"""
import json
import re
import os

import polars as pl
from polars.testing import assert_frame_equal

import arrowmetal as am

_original_collect = pl.LazyFrame.collect
_stats = {"compared": 0, "taken": 0, "kinds": {}}


def _orderable(df):
    try:
        df.sort(pl.all(), nulls_last=True)
        return True
    except Exception:          # list / struct / object columns cannot be sorted
        return False


def _top_k_keys(lf):
    """The key columns of the root sort when the plan's root is a sort with a slice, else None."""
    first = lf.explain().splitlines()[0]
    if not first.startswith("SORT BY [slice"):
        return None
    return re.findall(r'col\("([^"]+)"\)', first) or None


def _compare(got, want):
    assert got.schema == want.schema, f"MetalEngine schema {got.schema} != Polars {want.schema}"
    if _orderable(want) and want.width:
        got = got.sort(pl.all(), nulls_last=True)
        want = want.sort(pl.all(), nulls_last=True)
    exact = not any(dt in (pl.Float32, pl.Float64) for dt in want.dtypes)
    if exact:
        assert_frame_equal(got, want, check_exact=True)
    else:
        rel = 1e-5 if pl.Float32 in want.dtypes else 1e-9
        assert_frame_equal(got, want, check_exact=False, rel_tol=rel, abs_tol=1e-9)


def _collect(self, *args, **kwargs):
    want = _original_collect(self, *args, **kwargs)
    plain = not args and set(kwargs) <= {"optimizations", "type_coercion", "predicate_pushdown",
                                         "projection_pushdown", "simplify_expression",
                                         "slice_pushdown", "comm_subplan_elim", "comm_subexpr_elim",
                                         "cluster_with_columns", "no_optimization"}
    if not plain or not isinstance(want, pl.DataFrame):
        return want
    engine = am.MetalEngine(min_rows=0, shapes="all")
    got = _original_collect(self, *args, engine=engine, **kwargs)
    report = engine.last_report
    try:
        _compare(got, want)
    except AssertionError:
        # A sort with a slice (`sort(k).head(n)`) may keep different rows of a tie at the cut on
        # the two engines; then only the sort keys have one right answer.
        keys = _top_k_keys(self)
        if not keys:
            raise
        _compare(got.select(keys), want.select(keys))
    _stats["compared"] += 1
    if report is not None and report.taken:
        _stats["taken"] += 1
        for k in report.kinds_taken():
            _stats["kinds"][k] = _stats["kinds"].get(k, 0) + 1
    return want


def pytest_configure(config):
    pl.LazyFrame.collect = _collect


def pytest_unconfigure(config):
    pl.LazyFrame.collect = _original_collect
    path = os.environ.get("AM_EVERYWHERE_STATS")
    if path:
        with open(path, "w") as fh:
            json.dump(_stats, fh)
