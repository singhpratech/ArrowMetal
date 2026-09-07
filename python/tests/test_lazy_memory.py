"""The plan C ABI must not grow the process without bound.

`am_plan_explain` / `am_plan_run` are called straight from Python, which never returns to a run
loop, so nothing drains the autorelease pool between calls. Without an explicit `autoreleasepool`
in the entry point every Foundation/Metal object a query autoreleases lived until the process
exited: a fixed `explain()` in a loop grew peak RSS by about 0.5 KB per call, for ever.

    PYTHONPATH=python python -m pytest python/tests/test_lazy_memory.py -q
"""
import resource

import pyarrow as pa
import pytest

import arrowmetal as am

# `ru_maxrss` is bytes on Darwin and kilobytes on Linux.
_MAXRSS_TO_MB = 1024.0 * 1024.0


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _MAXRSS_TO_MB


@pytest.fixture(scope="module")
def plan():
    t = pa.table({"k": pa.array([1, 2, 3] * 10, pa.int64()),
                  "v": pa.array(list(range(30)), pa.int64())})
    return am.scan(t).group_by("k").agg(am.agg.sum("v", "s"))


def test_repeated_explain_does_not_grow_the_process(plan):
    """40 000 identical `explain()` calls: everything they touch is cached or transient, so peak RSS
    must be flat. Before the `autoreleasepool` in `am_plan_explain` this grew ~18 MB."""
    for _ in range(2000):                       # warm every cache first
        plan.explain()
    before = peak_rss_mb()
    for _ in range(40_000):
        plan.explain()
    grew = peak_rss_mb() - before
    assert grew < 8.0, f"peak RSS grew {grew:.1f} MB over 40000 explain() calls"


def test_repeated_collect_does_not_grow_the_process_quickly(plan):
    """The same for the executing path. The GPU execution below `am_plan_run` still grows slowly
    (~0.4 KB a query, see the review notes), so this only pins the part the ABI controls: before the
    `autoreleasepool` this grew ~26 MB over 10 000 collects, and now grows about a sixth of that."""
    for _ in range(500):
        plan.collect()
    before = peak_rss_mb()
    for _ in range(10_000):
        plan.collect()
    grew = peak_rss_mb() - before
    assert grew < 12.0, f"peak RSS grew {grew:.1f} MB over 10000 collect() calls"
