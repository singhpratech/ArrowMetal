"""Index arrays are uint32.

argsort, top_k, partition_nth_indices, lexsort_indices, the join indices and the ranks (row_number,
rank, dense_rank, the window functions of a lazy plan) come back as `pa.uint32()`: the same width the
int32 arrays had, and the type arrow-rs `sort_to_indices` and Polars' IdxSize use. pyarrow returns
uint64 for the same functions, so the values are compared here, not the types. `take` accepts int32,
int64 and uint32 indices. A row number of 2^31 or more comes back positive (the big test, which needs
ARROWMETAL_BIG_TESTS=1).
"""
import os

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import arrowmetal as am

VALUES = pa.array([5, None, 3, 9, 1, 9, 3], pa.int64())


def arrow(x):
    return x.to_arrow() if hasattr(x, "to_arrow") else x


def test_index_arrays_are_uint32_with_pyarrows_values():
    a = am.array(VALUES)
    cases = [
        (a.argsort(), pc.array_sort_indices(VALUES)),
        (a.argsort(True), pc.array_sort_indices(VALUES, order="descending")),
        (a.top_k(2), pc.array_sort_indices(VALUES, order="descending").slice(0, 2)),
        (a.top_k(3, largest=False, null_placement="at_start"),
         pc.array_sort_indices(VALUES, null_placement="at_start").slice(0, 3)),
        (am.lexsort_indices([a, a], [True, False]),
         pc.sort_indices(pa.table({"a": VALUES, "b": VALUES}), sort_keys=[("a", "descending"), ("b", "ascending")])),
        (a.row_number(), pc.rank(VALUES, tiebreaker="first")),
        (a.rank(), pc.rank(VALUES, tiebreaker="min")),
        (a.rank(tiebreaker="dense"), pc.rank(VALUES, tiebreaker="dense")),
        (a.rank(tiebreaker="max"), pc.rank(VALUES, tiebreaker="max")),
    ]
    for got, want in cases:
        got = arrow(got)
        assert got.type == pa.uint32()
        assert want.type == pa.uint64()                 # pyarrow's own type; the values must agree
        assert got.to_pylist() == want.to_pylist()

    part = arrow(a.partition_nth_indices(3))
    assert part.type == pa.uint32()
    assert sorted(part.to_pylist()) == list(range(len(VALUES)))
    assert sorted(VALUES.take(part.slice(0, 3)).to_pylist()) == [1, 3, 3]


def test_join_indices_are_uint32():
    li, ri = am.join(pa.array([1, 2, 3, 2], pa.int32()), pa.array([2, 3, 4], pa.int32()))
    li, ri = arrow(li), arrow(ri)
    assert li.type == ri.type == pa.uint32()
    assert sorted(zip(li.to_pylist(), ri.to_pylist())) == [(1, 0), (2, 1), (3, 0)]


def test_plan_window_ranks_are_uint32():
    out = (am.scan(pa.table({"k": pa.array([3, 1, 2, 1], pa.int32())}))
           .with_row_number("rn", order_by="k").with_rank("rk", order_by="k")
           .with_rank("dr", order_by="k", dense=True).collect())
    for name, want in [("rn", [4, 1, 3, 2]), ("rk", [4, 1, 3, 1]), ("dr", [3, 1, 2, 1])]:
        assert out.column(name).type == pa.uint32()
        assert out.column(name).to_pylist() == want


def test_other_integer_outputs_keep_their_types():
    """indices_nonzero is uint64 as in Arrow; dictionary codes, group ids and inverse_permutation are
    int32."""
    a = am.array(pa.array([5, 0, 3, 0], pa.int64()))
    nz = arrow((a != 0).indices_nonzero())
    assert nz.type == pa.uint64() and nz.to_pylist() == [0, 2]
    codes, _ = a.dictionary_encode()
    assert arrow(codes).type == pa.int32()
    assert arrow(am.group_by([a]).ids()).type == pa.int32()
    inv = arrow(am.array(pa.array([2, 0, 1], pa.uint32())).inverse_permutation())
    assert inv.type == pa.int32() and inv.to_pylist() == [1, 2, 0]


@pytest.mark.parametrize("index", [
    pa.array([4, 1, 0, None, 3], pa.int32()),
    pa.array([4, 1, 0, None, 3], pa.int64()),
    pa.array([4, 1, 0, None, 3], pa.uint32()),
    pa.array([4, 1, 0, None, 3], pa.int8()),
    pa.array([4, 1, 0, None, 3], pa.uint16()),
    np.array([4, 1, 0, 2, 3], dtype=np.int64),
    np.array([4, 1, 0, 2, 3], dtype=np.uint32),
    [4, 1, 0, 2, 3],
], ids=["int32", "int64", "uint32", "int8", "uint16", "numpy-int64", "numpy-uint32", "list"])
def test_take_accepts_every_index_type(index):
    src = pa.array([10, None, 30, 40, 50], pa.int64())
    strs = pa.array(["a", "bb", None, "dddd", "e"])
    want_index = index if isinstance(index, pa.Array) else pa.array(index)
    for col in (src, strs):
        got = arrow(am.array(col).take(index))
        assert got.to_pylist() == col.take(want_index).to_pylist()


def test_take_returned_indices_round_trip():
    a = am.array(VALUES)
    assert arrow(a.take(a.argsort())).to_pylist() == [1, 3, 3, 5, 9, 9, None]
    assert arrow(a.take(arrow(a.argsort()))).to_pylist() == [1, 3, 3, 5, 9, 9, None]


@pytest.mark.parametrize("index", [
    pa.array([0, -1], pa.int32()),
    pa.array([0, 7], pa.uint32()),
    pa.array([0, (1 << 32) + 1], pa.int64()),           # 2^32 + 1 would narrow onto row 1
    [0, 2**31 + 3],
], ids=["int32-negative", "uint32-at-length", "int64-past-2^32", "list-past-2^31"])
def test_take_refuses_out_of_range_indices(index):
    with pytest.raises(am.ArrowMetalError, match="out of range"):
        arrow(am.array(VALUES).take(index))


@pytest.mark.skipif(os.environ.get("ARROWMETAL_BIG_TESTS") != "1",
                    reason="set ARROWMETAL_BIG_TESTS=1 to run over 2^31 + 16 rows")
def test_row_numbers_past_2_to_31_come_back_positive():
    n = (1 << 31) + 16
    row = (1 << 31) + 7
    values = np.zeros(n, dtype=np.uint8)
    values[row] = 255
    values[5] = 200
    a = am.array(pa.array(values))
    top = arrow(a.top_k(2))
    assert top.type == pa.uint32()
    assert top.to_pylist() == [row, 5]
    assert top.to_pylist()[0] > 2**31 - 1
    for index in (pa.array([row, 5], pa.uint32()), pa.array([row, 5], pa.int64()), [row, 5]):
        assert arrow(a.take(index)).to_pylist() == [255, 200]
    nz = arrow((a == 255).indices_nonzero())
    assert nz.to_pylist() == [row]
