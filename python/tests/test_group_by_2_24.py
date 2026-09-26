"""Group-by at and past 2^24 groups, against pyarrow's `Table.group_by`.

The per-group kernels (one 256-thread threadgroup per group: the Float64 and Float32-in-Float64 sum
and mean, the segmented 64-bit min/max, product, list, the wide variance) used to dispatch a
`(groups, 1, 1)` grid. The GPU holds a grid dimension's thread count in 32 bits, so at 2^24 groups the
width, 2^32 threads, wrapped: only `groups mod 2^24` threadgroups ran and the rest of the groups came
back null. The grid now folds into rows past that width. The values are small multiples of 0.5, so
every sum is exact in any order and the comparison is equality.
"""
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import arrowmetal as am

GROUP_COUNTS = [(1 << 24) - 1, 1 << 24, (1 << 24) + 1, (1 << 24) + (1 << 20)]
EXTRA_ROWS = 1 << 18


def _table(groups):
    rng = np.random.default_rng(groups)
    n = groups + EXTRA_ROWS
    keys = np.concatenate([np.arange(groups, dtype=np.int64), rng.integers(0, groups, EXTRA_ROWS)])
    keys = keys[rng.permutation(n)]
    ints = rng.integers(-1000, 1001, n)
    valid = rng.random(n) > 0.01
    return pa.table({
        "k": keys,
        "f64": pa.array(ints * 0.5, mask=~valid),
        "f32": pa.array((ints * 0.5).astype(np.float32), mask=~valid),
        "i64": pa.array(ints, mask=~valid),
    })


def _sorted_by_key(gb, result):
    order = pc.sort_indices(gb.keys()[0])
    return pc.take(result.to_arrow(), order)


@pytest.fixture(scope="module", params=GROUP_COUNTS, ids=lambda g: f"groups={g}")
def grouped(request):
    t = _table(request.param)
    return request.param, t, am.group_by([t.column("k").combine_chunks()])


@pytest.mark.parametrize("column", ["f64", "f32", "i64"])
def test_sum_mean_count_min_max(grouped, column):
    groups, t, gb = grouped
    assert len(gb) == groups
    aggs = ["sum", "mean", "count", "min", "max"]
    ref = t.group_by("k").aggregate([(column, a) for a in aggs]).sort_by("k")
    values = t.column(column).combine_chunks()
    for a in aggs:
        got = _sorted_by_key(gb, getattr(gb, a)(values))
        want = ref.column(f"{column}_{a}").combine_chunks()
        if want.type != got.type:
            want = want.cast(got.type)
        assert got.null_count == want.null_count, (a, column, groups)
        assert got.equals(want), (a, column, groups)


def test_product_and_list(grouped):
    groups, t, gb = grouped
    # Single-threaded: pyarrow's threaded hash_list does not keep row order inside a group at this size.
    ref = (t.group_by("k", use_threads=False)
           .aggregate([("f64", "product"), ("i64", "product"), ("i64", "list")]).sort_by("k"))
    for column in ("f64", "i64"):
        got = _sorted_by_key(gb, gb.product(t.column(column).combine_chunks()))
        want = ref.column(f"{column}_product").combine_chunks()
        assert got.equals(want.cast(got.type)), (column, groups)
    got = _sorted_by_key(gb, gb.list(t.column("i64").combine_chunks()))
    want = ref.column("i64_list").combine_chunks()
    assert pc.list_value_length(got).equals(pc.list_value_length(want))
    assert got.flatten().equals(want.flatten())


@pytest.mark.parametrize("agg", ["sum", "mean"])
def test_lazy_float64_group_by_at_2_24_groups(agg):
    """The shape of the Polars engine's crossover case: 2^24 distinct keys, one Float64 value each."""
    from arrowmetal import lazy
    n = 1 << 24
    k = am.MetalArray.from_arrow(pa.array(np.arange(n, dtype=np.int64)))
    v = am.MetalArray.from_arrow(pa.array(np.arange(n, dtype=np.float64)))
    plan = {"op": "group_by", "input": {"op": "scan", "source": "t"},
            "keys": [["k", '(col "k")']], "aggs": [[agg, "s", '(col "v")']]}
    out = lazy.LazyFrame(plan, {"t": lazy._Source(["k", "v"], [k, v])}).collect()
    s = out.column("s").combine_chunks()
    assert s.null_count == 0
    got = pc.take(s, pc.sort_indices(out.column("k")))
    assert got.equals(pa.array(np.arange(n, dtype=np.float64)))
