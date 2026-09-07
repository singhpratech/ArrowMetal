"""Optimizer soundness: `collect(optimize=True)` must agree with `collect(optimize=False)`.

docs/ENGINE.md: "Every rule preserves the result exactly — same rows, same nulls." These are the
cases where one did not, so the unoptimized plan is the oracle throughout and only a documented
invariant (row order, output arity) is compared against anything else.

    PYTHONPATH=python python -m pytest python/tests/test_lazy_optimizer.py -q
"""
import pyarrow as pa
import pytest

import arrowmetal as am


def rows(table, sort=False):
    cols = [c.to_pylist() for c in table.columns]
    out = [tuple(c[i] for c in cols) for i in range(table.num_rows)]
    if sort:
        out.sort(key=lambda r: tuple((v is None, "" if v is None else str(v)) for v in r))
    return out


def agree(q, sort=False):
    """Optimized and unoptimized must produce the same rows, and the same column names."""
    unopt = q.collect(optimize=False)
    opt = q.collect(optimize=True)
    assert opt.column_names == unopt.column_names, \
        f"schema differs: opt={opt.column_names} unopt={unopt.column_names}"
    a, b = rows(unopt, sort), rows(opt, sort)
    assert a == b, f"unoptimized={a}\n  optimized={b}"
    return opt


@pytest.fixture(scope="module")
def nullable():
    return pa.table({
        "c": pa.array([True, False, None, True], pa.bool_()),
        "x": pa.array([7, 8, 9, 10], pa.int64()),
        "y": pa.array([1, None, 3, 4], pa.int64()),
    })


# --------------------------------------------------------------------------------------------------
# constant folding


def test_and_with_a_false_literal_still_propagates_nulls(nullable):
    """`and` is null propagating: `false and null` is null, not false. Only `and_kleene` absorbs."""
    q = am.scan(nullable).select((am.lit(False) & am.col("c")).alias("r"))
    assert rows(agree(q)) == [(False,), (False,), (None,), (False,)]


def test_or_with_a_true_literal_still_propagates_nulls(nullable):
    q = am.scan(nullable).select((am.lit(True) | am.col("c")).alias("r"))
    assert rows(agree(q)) == [(True,), (True,), (None,), (True,)]


def test_kleene_forms_do_absorb(nullable):
    """The fix must not stop the Kleene rules from folding, where absorbing *is* correct."""
    q = am.scan(nullable).select(am.lit(False).and_kleene(am.col("c")).alias("a"),
                                 am.lit(True).or_kleene(am.col("c")).alias("o"))
    assert rows(agree(q)) == [(False, True)] * 4


def test_if_else_with_identical_branches_keeps_the_condition_null(nullable):
    """Arrow's if_else is null wherever the condition is, even when both branches are the same."""
    q = am.scan(nullable).select(am.if_else(am.col("c"), am.col("x"), am.col("x")).alias("r"))
    assert rows(agree(q)) == [(7,), (8,), (None,), (10,)]


def test_if_else_with_a_literal_condition_still_folds(nullable):
    q = am.scan(nullable).select(am.if_else(am.lit(True), am.col("x"), am.col("y")).alias("r"))
    assert rows(agree(q)) == [(7,), (8,), (9,), (10,)]


def test_folding_int64_min_divided_by_minus_one_does_not_trap(nullable):
    """This used to trap in `Optimizer.foldExpr` and take the whole process with it."""
    q = am.scan(nullable).select(
        (am.lit(-(2 ** 63), "int64") / am.lit(-1, "int64")).alias("d"),
        am.lit(-(2 ** 63), "int64").abs().alias("a"))
    got = rows(q.collect(optimize=True))
    assert got == [(-(2 ** 63), -(2 ** 63))] * 4


# --------------------------------------------------------------------------------------------------
# CSE


def test_duplicate_select_outputs_are_two_columns(nullable):
    """`select` names its outputs positionally, so a repeated one is a repeated column."""
    q = am.scan(nullable).select(am.col("x").alias("a"), am.col("x").alias("a"))
    out = agree(q)
    assert out.num_columns == 2, f"expected 2 columns, got {out.column_names}"


def test_with_columns_still_merges_a_repeated_name(nullable):
    """`with_columns` really does merge by name, so deduplicating it stays correct."""
    q = am.scan(nullable).with_columns((am.col("x") + 1).alias("z"), (am.col("x") + 1).alias("z"))
    out = agree(q)
    assert out.column_names.count("z") == 1


# --------------------------------------------------------------------------------------------------
# join reordering vs the documented row order


def _small_and_big():
    small = pa.table({"k": pa.array([3, 1, 2], pa.int64()), "lv": pa.array([30, 10, 20], pa.int64())})
    big = pa.table({"k": pa.array(list(range(400)), pa.int64()),
                    "rv": pa.array([i * 2 for i in range(400)], pa.int64())})
    return small, big


def test_inner_join_keeps_left_order_even_when_the_right_side_is_bigger():
    """docs/ENGINE.md: "Inner and left keep probe (left) order". Swapping the sides is a permutation,
    so `join_reorder` must not fire where the order is still observable."""
    small, big = _small_and_big()
    q = am.scan(small).join(am.scan(big), on="k", how="inner")
    out = agree(q)
    assert [r[0] for r in rows(out)] == [3, 1, 2]
    assert "join_reorder" not in q.explain()


def test_join_reorder_still_fires_under_a_group_by():
    small, big = _small_and_big()
    q = am.scan(small).join(am.scan(big), on="k", how="inner").group_by("k").agg(am.agg.sum("rv", "s"))
    assert "join_reorder" in q.explain()
    agree(q, sort=True)


def test_join_reorder_still_fires_under_a_sort():
    small, big = _small_and_big()
    q = am.scan(small).join(am.scan(big), on="k", how="inner").sort("rv")
    assert "join_reorder" in q.explain()
    agree(q)


def test_join_reorder_does_not_fire_under_a_limit():
    small, big = _small_and_big()
    q = am.scan(small).join(am.scan(big), on="k", how="inner").limit(2)
    assert "join_reorder" not in q.explain()
    assert rows(agree(q)) == [(3, 30, 6), (1, 10, 2)]


# --------------------------------------------------------------------------------------------------
# the untyped-literal type rule, through the engine rather than am.query


@pytest.mark.parametrize("expr,want", [
    (lambda: am.col("i8") > am.lit(200), [False, False, False]),
    (lambda: am.col("i8") == am.lit(1000), [False, False, False]),
    (lambda: am.col("u8") == am.lit(-1), [False, False, False]),
    (lambda: am.col("i64") >= am.lit(2.5), [False, False, True]),
])
def test_untyped_literals_that_do_not_fit_widen_the_comparison(expr, want):
    t = pa.table({"i8": pa.array([-56, 1, 100], pa.int8()),
                  "u8": pa.array([0, 1, 255], pa.uint8()),
                  "i64": pa.array([1, 2, 3], pa.int64())})
    q = am.scan(t).select(expr().alias("r"))
    assert rows(agree(q)) == [(w,) for w in want]
