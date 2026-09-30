"""`float_order="nan_largest"`, and the sort options of window `order_by` keys and of the streaming sort.

``float_order="nan_largest"`` is Polars' float order: every NaN one value above +inf in both
directions (last ascending, first descending), -0.0 and +0.0 tied. The reference is Polars itself: a
frame of the values and their row numbers sorted with ``maintain_order=True``, which is stable, so the
row numbers are the permutation index for index. The window and streaming checks reference the same
orders -- "ieee" through pyarrow, "total" through its definition (test_sort_options.py) and
"nan_largest" through Polars.
"""
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import arrowmetal as am
from arrowmetal import stream as ams

from test_sort_options import OPTIONS, SIZES, floats32, floats64, total_order_indices

pl = pytest.importorskip("polars")

ORDERS = ["ieee", "total", "nan_largest"]


def _ids(x):
    return x.to_arrow().to_pylist()


def polars_order(arr, descending, null_placement):
    """The stable permutation Polars sorts `arr` into (its NaN-largest order)."""
    df = pl.DataFrame({"x": pl.from_arrow(arr), "i": np.arange(len(arr), dtype=np.int64)})
    out = df.sort("x", descending=descending, nulls_last=null_placement == "at_end", maintain_order=True)
    return out["i"].to_list()


def reference(arr, descending, null_placement, float_order):
    if float_order == "nan_largest":
        return polars_order(arr, descending, null_placement)
    if float_order == "total":
        return total_order_indices(arr, descending, null_placement)
    return pc.array_sort_indices(arr, order="descending" if descending else "ascending",
                                 null_placement=null_placement).to_pylist()


def tie_key(v, float_order):
    """What two values tie on under `float_order`: None for a null, one value for every NaN and for
    both zeros except under "total", which ties only identical bits."""
    if v is None:
        return None
    if float_order == "total":
        return np.float64(v).view(np.uint64).item()
    if v != v:
        return "nan"
    return 0.0 if v == 0 else v


def ranks(arr, descending, null_placement, float_order):
    order = reference(arr, descending, null_placement, float_order)
    vals = [tie_key(v, float_order) for v in arr.to_pylist()]
    rank = [0] * len(arr)
    for j, i in enumerate(order):
        rank[i] = j if j == 0 or vals[order[j - 1]] != vals[i] else rank[order[j - 1]]
    return rank


# ---------------------------------------------------------------------------------------------
# the kernels


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("null_fraction", [0.0, 0.1, 1.0])
@pytest.mark.parametrize("descending,placement", OPTIONS)
@pytest.mark.parametrize("make", [floats64, floats32], ids=["f64", "f32"])
def test_nan_largest_is_polars_order(n, null_fraction, descending, placement, make):
    arr = make(n, null_fraction, n + 7)
    want = polars_order(arr, descending, placement)
    x = am.MetalArray.from_arrow(arr)
    assert _ids(x.argsort(descending, null_placement=placement, float_order="nan_largest")) == want
    for k in (1, 7, 100, 1500):
        got = _ids(x.top_k(k, largest=descending, null_placement=placement, float_order="nan_largest"))
        assert got == want[:k], k
    # The sorted values: nulls where Polars puts them, every value equal to Polars' (NaN to NaN).
    got = x.sort(descending, null_placement=placement, float_order="nan_largest").to_arrow()
    want_vals = arr.take(pa.array(want, pa.int64()))
    assert got.is_null().to_pylist() == want_vals.is_null().to_pylist()
    g, w = got.to_pylist(), want_vals.to_pylist()
    assert all((a is None and b is None) or (a != a and b != b) or a == b for a, b in zip(g, w))


def test_nan_largest_two_million_rows():
    arr = floats64(2_000_000, 0.05, 99)
    x = am.MetalArray.from_arrow(arr)
    for descending, placement in OPTIONS:
        want = polars_order(arr, descending, placement)
        assert _ids(x.argsort(descending, null_placement=placement, float_order="nan_largest")) == want
        assert _ids(x.top_k(100, largest=descending, null_placement=placement,
                            float_order="nan_largest")) == want[:100]


@pytest.mark.parametrize("n", [8193, 70_001])
def test_lexsort_nan_largest_per_key(n):
    rng = np.random.default_rng(n)
    k1 = pa.array(rng.integers(0, 4, n), mask=rng.random(n) < 0.1)
    k2 = floats64(n, 0.1, n + 1)
    for d1, p1 in OPTIONS:
        for d2, p2 in OPTIONS:
            got = _ids(am.lexsort_indices([k1, k2], [d1, d2], null_placement=[p1, p2],
                                          float_order=["ieee", "nan_largest"]))
            df = pl.DataFrame({"a": pl.from_arrow(k1), "b": pl.from_arrow(k2),
                               "i": np.arange(n, dtype=np.int64)})
            want = df.sort(["a", "b"], descending=[d1, d2], nulls_last=[p1 == "at_end", p2 == "at_end"],
                           maintain_order=True)["i"].to_list()
            assert got == want, (d1, p1, d2, p2)


@pytest.mark.parametrize("limit", [None, 100])
def test_lazy_sort_nan_largest(limit):
    n = 70_001
    x = floats64(n, 0.05, 21)
    t = pa.table({"x": x, "row": pa.array(np.arange(n, dtype=np.int32))})
    for descending, placement in OPTIONS:
        want = polars_order(x, descending, placement)
        q = am.scan(t).sort("x", descending=descending, null_placement=placement, float_order="nan_largest")
        if limit:
            q = q.limit(limit)
        assert q.collect().column("row").to_pylist() == (want[:limit] if limit else want)


def test_the_c_abi_takes_float_order_2():
    x = am.MetalArray.from_arrow(pa.array([1.0, float("nan"), None, -0.0, 0.0, float("inf")]))
    assert _ids(x.argsort(True, null_placement="at_end", float_order="nan_largest")) == [1, 5, 0, 3, 4, 2]
    assert _ids(x.argsort(False, null_placement="at_start", float_order="nan_largest")) == [2, 3, 4, 0, 5, 1]
    with pytest.raises(am.ArrowMetalError):
        x.argsort(float_order="nan_last")


# ---------------------------------------------------------------------------------------------
# window order_by keys


@pytest.mark.parametrize("float_order", ORDERS)
@pytest.mark.parametrize("descending,placement", OPTIONS)
def test_window_order_by_options(float_order, descending, placement):
    n = 20_011
    x = floats64(n, 0.1, 5)
    rng = np.random.default_rng(6)
    p = rng.integers(0, 5, n).astype(np.int32)
    t = pa.table({"x": x, "p": pa.array(p), "row": pa.array(np.arange(n, dtype=np.int32))})
    out = (am.scan(t)
           .with_row_number("rn", partition_by="p", order_by="x", descending=descending,
                            null_placement=placement, float_order=float_order)
           .with_rank("rk", partition_by="p", order_by="x", descending=descending,
                      null_placement=placement, float_order=float_order)
           .collect())
    rank = ranks(x, descending, placement, float_order)
    order = sorted(range(n), key=lambda i: (p[i], rank[i], i))
    want_rn, want_rk = [0] * n, [0] * n
    start = tie = 0
    for j, i in enumerate(order):
        if j == 0 or p[order[j - 1]] != p[i]:
            start = tie = j
        elif rank[order[j - 1]] != rank[i]:
            tie = j
        want_rn[i], want_rk[i] = j - start + 1, tie - start + 1
    rows = out.column("row").to_pylist()
    got_rn = dict(zip(rows, out.column("rn").to_pylist()))
    got_rk = dict(zip(rows, out.column("rk").to_pylist()))
    assert [got_rn[i] for i in range(n)] == want_rn
    assert [got_rk[i] for i in range(n)] == want_rk


def test_window_order_by_defaults_unchanged():
    t = pa.table({"x": pa.array([3.0, None, 1.0, float("nan"), 1.0, None, -0.0, 0.0])})
    asc = am.scan(t).with_rank("rk", order_by="x").collect().column("rk").to_pylist()
    assert asc == [5, 7, 3, 6, 3, 7, 1, 1]
    desc = am.scan(t).with_rank("rk", order_by="x", descending=True).collect().column("rk").to_pylist()
    assert desc == [1, 7, 2, 6, 2, 7, 4, 4]


# ---------------------------------------------------------------------------------------------
# the streaming external sort


@pytest.mark.parametrize("float_order", ORDERS)
@pytest.mark.parametrize("descending,placement", OPTIONS)
@pytest.mark.parametrize("limit", [0, 1, 37, 900])
def test_stream_sort_options(tmp_path, float_order, descending, placement, limit):
    n = 9_001
    x = floats64(n, 0.1, 17)
    t = pa.table({"x": x, "id": pa.array(np.arange(n, dtype=np.int64))})
    want = reference(x, descending, placement, float_order)
    got = (ams.scan_table(t, batch_rows=700)
           .sort([("x", descending)], limit=limit, scratch=tmp_path,
                 null_placement=placement, float_order=float_order))
    assert got.column("id").to_pylist() == (want[:limit] if limit else want)


def test_stream_sort_two_keys_per_key_options(tmp_path):
    n = 6_007
    rng = np.random.default_rng(3)
    k = pa.array(rng.integers(0, 5, n), mask=rng.random(n) < 0.15)
    x = floats64(n, 0.1, 18)
    t = pa.table({"k": k, "x": x, "id": pa.array(np.arange(n, dtype=np.int64))})
    got = (ams.scan_table(t, batch_rows=500)
           .sort([("k", True), ("x", False)], scratch=tmp_path,
                 null_placement=["at_start", "at_end"], float_order=["ieee", "nan_largest"]))
    df = pl.DataFrame({"k": pl.from_arrow(k), "x": pl.from_arrow(x), "i": np.arange(n, dtype=np.int64)})
    want = df.sort(["k", "x"], descending=[True, False], nulls_last=[False, True],
                   maintain_order=True)["i"].to_list()
    assert got.column("id").to_pylist() == want


def test_stream_sort_defaults_keep_the_old_call(tmp_path):
    t = pa.table({"x": pa.array([2.0, None, 1.0, 3.0]), "id": pa.array([0, 1, 2, 3], pa.int64())})
    got = ams.scan_table(t, batch_rows=2).sort("x", scratch=tmp_path)
    assert got.column("id").to_pylist() == [2, 0, 3, 1]
