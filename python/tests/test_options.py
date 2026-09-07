"""Every option value of every option-carrying function, against pyarrow.compute.

The functions here used to take one shape of call each and carried a `partial` row in
``arrowmetal.functions`` saying which of Arrow's options were missing. This file is the evidence that
they are not missing any more: for each function it walks the **full cross product** of its option
values and compares ArrowMetal's answer to pyarrow's on the same input, at several sizes and with
nulls throughout.

Where ArrowMetal deliberately differs from pyarrow the difference is asserted rather than skipped, so
the test fails if the difference ever silently changes:

* ``is_in`` / ``index_in`` default to ``null_matching_behavior="skip"``; pyarrow defaults to
  ``"match"``. pyarrow 25 only exposes those two through ``skip_nulls``, so ``emit_null`` and
  ``inconclusive`` are checked against the Arrow C++ definition written out here.
* ``cast`` defaults to ``safe=False``; pyarrow defaults to ``safe=True``.
* ``unique`` / ``value_counts`` over a **utf8** column drop the null, because the GPU string
  dictionary has no slot for one. Over every other type they keep it, as pyarrow does.
* ``rank_normal`` in float64 evaluates the inverse CDF on the host, so it agrees with pyarrow to
  about 1e-12 rather than exactly.
"""
import datetime
import math

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import arrowmetal as am

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

SIZES = [0, 1, 33, 4097]
BIG = 1_000_003
PLACEMENTS = ["at_end", "at_start"]
DIRECTIONS = ["ascending", "descending"]
TIEBREAKERS = ["min", "max", "first", "dense"]


def ints(n, null_every=7, modulus=97):
    """int64 with plenty of ties and roughly one null in `null_every`."""
    return pa.array([None if i % null_every == 3 else (i * 7919) % modulus for i in range(n)],
                    type=pa.int64())


def floats(n):
    return pa.array([None if i % 5 == 2 else float((i * 13) % 11) - 5.0 for i in range(n)],
                    type=pa.float64())


def _py(x):
    return x.to_arrow().to_pylist() if isinstance(x, am.MetalArray) else x


# ---------------------------------------------------------------------------
# null_placement on the sorts


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_array_sort_indices_null_placement(n, descending, placement):
    v = ints(n)
    got = _py(am.MetalArray.from_arrow(v).argsort(descending=descending, null_placement=placement))
    want = pc.array_sort_indices(v, order="descending" if descending else "ascending",
                                 null_placement=placement).to_pylist()
    assert got == want


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_array_sort_indices_null_placement_large(placement):
    v = ints(BIG)
    got = _py(am.MetalArray.from_arrow(v).argsort(null_placement=placement))
    assert got == pc.array_sort_indices(v, null_placement=placement).to_pylist()


@pytest.mark.parametrize("placement", PLACEMENTS)
@pytest.mark.parametrize("descending", [False, True])
def test_sort_copy_null_placement(placement, descending):
    v = ints(200)
    got = _py(am.MetalArray.from_arrow(v).sort(descending=descending, null_placement=placement))
    idx = pc.array_sort_indices(v, order="descending" if descending else "ascending",
                                null_placement=placement)
    assert got == pc.take(v, idx).to_pylist()


@pytest.mark.parametrize("descending", [False, True])
def test_sort_keeps_the_input_type(descending):
    """`sort` returns the type it was given, a dictionary column included.

    A dictionary orders by the values its codes point at, but the gather runs on the codes, so the
    answer is a dictionary of the same value array — not the decoded column."""
    words = pa.array(["pear", None, "fig", "apple", "fig", "damson"] * 40).dictionary_encode()
    got = am.MetalArray.from_arrow(words).sort(descending=descending)
    assert got.type == words.type, "a dictionary column must not decode on the way through sort"
    out = got.to_arrow()
    assert out.dictionary.to_pylist() == words.dictionary.to_pylist(), "the value array is untouched"
    idx = pc.array_sort_indices(words, order="descending" if descending else "ascending")
    assert out.to_pylist() == pc.take(words, idx).to_pylist()

    for col in (pa.array([3, None, 1, 2] * 50, type=pa.int64()),
                pa.array([1.5, None, -0.0, 2.5] * 50, type=pa.float64()),
                pa.array(["b", None, "a", "c"] * 50),
                pa.array([b"b", None, b"a"] * 50, type=pa.binary()),
                pa.array([True, None, False] * 50)):
        s = am.MetalArray.from_arrow(col).sort(descending=descending)
        assert s.type == am.MetalArray.from_arrow(col).type, f"{col.type} changed type"
        want = pc.take(col, pc.array_sort_indices(col, order="descending" if descending else "ascending"))
        assert s.to_arrow().to_pylist() == want.to_pylist()


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_lexsort_null_placement(placement):
    n = 300
    major = pa.array([None if i % 11 == 2 else i % 5 for i in range(n)], type=pa.int64())
    minor = pa.array([None if i % 13 == 4 else (i * 7) % 9 for i in range(n)], type=pa.int64())
    got = _py(am.lexsort_indices([am.MetalArray.from_arrow(major), am.MetalArray.from_arrow(minor)],
                                 [False, True], null_placement=placement))
    want = pc.sort_indices(pa.table({"a": major, "b": minor}),
                           sort_keys=[("a", "ascending"), ("b", "descending")],
                           null_placement=placement).to_pylist()
    assert got == want


# ---------------------------------------------------------------------------
# rank


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("tiebreaker", TIEBREAKERS)
@pytest.mark.parametrize("sort_keys", DIRECTIONS)
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_rank_every_option(n, tiebreaker, sort_keys, placement):
    v = ints(n)
    got = _py(am.MetalArray.from_arrow(v).rank(sort_keys=sort_keys, null_placement=placement,
                                               tiebreaker=tiebreaker))
    want = pc.rank(v, sort_keys=sort_keys, null_placement=placement,
                   tiebreaker=tiebreaker).to_pylist()
    assert got == want


@pytest.mark.parametrize("tiebreaker", TIEBREAKERS)
def test_rank_large(tiebreaker):
    v = ints(BIG)
    got = _py(am.MetalArray.from_arrow(v).rank(null_placement="at_start", tiebreaker=tiebreaker))
    assert got == pc.rank(v, sort_keys="ascending", null_placement="at_start",
                          tiebreaker=tiebreaker).to_pylist()


@pytest.mark.parametrize("sort_keys", DIRECTIONS)
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_rank_floats_with_nan(sort_keys, placement):
    v = pa.array([1.0, float("nan"), None, -0.0, 0.0, 2.5, float("nan")], type=pa.float64())
    for tiebreaker in TIEBREAKERS:
        got = _py(am.MetalArray.from_arrow(v).rank(sort_keys=sort_keys, null_placement=placement,
                                                   tiebreaker=tiebreaker))
        want = pc.rank(v, sort_keys=sort_keys, null_placement=placement,
                       tiebreaker=tiebreaker).to_pylist()
        assert got == want, tiebreaker


def test_rank_default_tiebreaker_is_min():
    """This package's `rank()` keeps SQL's `RANK()` meaning, `tiebreaker="min"`; Arrow's default is
    `"first"`. Everything else about the default call matches."""
    v = ints(129)
    a = am.MetalArray.from_arrow(v)
    assert _py(a.rank()) == pc.rank(v, sort_keys="ascending", tiebreaker="min").to_pylist()
    assert _py(a.rank()) != pc.rank(v, sort_keys="ascending").to_pylist()


# ---------------------------------------------------------------------------
# rank_quantile / rank_normal


@pytest.mark.parametrize("n", [1, 33, 4097])
@pytest.mark.parametrize("sort_keys", DIRECTIONS)
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_rank_quantile_every_option(n, sort_keys, placement):
    v = ints(n)
    got = _py(am.MetalArray.from_arrow(v).rank_quantile(sort_keys=sort_keys, null_placement=placement))
    want = pc.rank_quantile(v, sort_keys=sort_keys, null_placement=placement).to_pylist()
    assert got == pytest.approx(want, abs=1e-15)


@pytest.mark.parametrize("sort_keys", DIRECTIONS)
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_rank_normal_every_option(sort_keys, placement):
    v = ints(257)
    got = _py(am.MetalArray.from_arrow(v).rank_normal(sort_keys=sort_keys, null_placement=placement))
    want = pc.rank_normal(v, sort_keys=sort_keys, null_placement=placement).to_pylist()
    # The float64 inverse CDF runs on the host with Wichura's AS 241; Arrow's differs in the last ulp.
    assert got == pytest.approx(want, abs=1e-12)
    # float32 is the fully-GPU form, good to about 1e-6.
    f32 = _py(am.MetalArray.from_arrow(v).rank_normal(sort_keys=sort_keys, null_placement=placement,
                                                      float32=True))
    assert f32 == pytest.approx(want, abs=2e-5)


# ---------------------------------------------------------------------------
# partition_nth_indices


def _check_partition(v, pivot, placement):
    got = _py(am.MetalArray.from_arrow(v).partition_nth_indices(pivot, null_placement=placement))
    n = len(v)
    assert sorted(got) == list(range(n)), "must be a permutation"
    if pivot >= n:
        return
    rows = v.to_pylist()

    def key(i):
        x = rows[i]
        return (-1 if placement == "at_start" else 1, 0) if x is None else (0, x)

    at = key(got[pivot])
    assert all(key(got[i]) <= at for i in range(pivot))
    assert all(key(got[i]) >= at for i in range(pivot + 1, n))
    # And it is the order statistic a full sort would put there.
    ordered = pc.array_sort_indices(v, null_placement=placement).to_pylist()
    assert key(got[pivot]) == key(ordered[pivot])


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_partition_nth_every_option(n, placement):
    v = ints(n)
    for pivot in sorted({0, n // 3, n // 2, max(0, n - 1), n}):
        _check_partition(v, pivot, placement)


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_partition_nth_floats(placement):
    _check_partition(floats(1001), 500, placement)


def test_partition_nth_large():
    _check_partition(ints(BIG), BIG // 2, "at_end")


def test_partition_nth_rejects_bad_pivot():
    a = am.MetalArray.from_arrow(pa.array([1, 2, 3], type=pa.int64()))
    with pytest.raises(am.ArrowMetalError):
        a.partition_nth_indices(-1)
    with pytest.raises(am.ArrowMetalError):
        a.partition_nth_indices(4)


# ---------------------------------------------------------------------------
# is_in / index_in null_matching_behavior


def _expected_is_in(rows, value_set, behavior):
    present = {v for v in value_set if v is not None}
    set_has_null = any(v is None for v in value_set)
    out = []
    for x in rows:
        if x is None:
            out.append({"skip": False, "match": set_has_null}.get(behavior))
        elif x in present:
            out.append(True)
        else:
            out.append(None if (behavior == "inconclusive" and set_has_null) else False)
    return out


def _expected_index_in(rows, value_set, behavior):
    first = {}
    first_null = None
    for i, v in enumerate(value_set):
        if v is None:
            if first_null is None:
                first_null = i
        elif v not in first:
            first[v] = i
    return [(first_null if behavior == "match" else None) if x is None else first.get(x)
            for x in rows]


VALUE_SETS = [[2, None, 5, 2], [2, 5], [None], []]


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("behavior", ["match", "skip", "emit_null", "inconclusive"])
@pytest.mark.parametrize("value_set", VALUE_SETS)
def test_set_lookup_every_behavior(n, behavior, value_set):
    v = ints(n, null_every=5, modulus=8)
    s = pa.array(value_set, type=pa.int64())
    a = am.MetalArray.from_arrow(v)
    rows = v.to_pylist()
    assert _py(a.is_in(s, behavior)) == _expected_is_in(rows, value_set, behavior)
    assert _py(a.index_in(s, behavior)) == _expected_index_in(rows, value_set, behavior)


@pytest.mark.parametrize("value_set", VALUE_SETS)
def test_set_lookup_agrees_with_pyarrow_where_pyarrow_exposes_it(value_set):
    """pyarrow 25 exposes only `match` (skip_nulls=False) and `skip` (skip_nulls=True)."""
    v = ints(129, null_every=5, modulus=8)
    s = pa.array(value_set, type=pa.int64())
    a = am.MetalArray.from_arrow(v)
    for behavior, skip_nulls in [("match", False), ("skip", True)]:
        assert _py(a.is_in(s, behavior)) == pc.is_in(v, value_set=s, skip_nulls=skip_nulls).to_pylist()
        assert _py(a.index_in(s, behavior)) == pc.index_in(v, value_set=s,
                                                           skip_nulls=skip_nulls).to_pylist()


def test_set_lookup_default_is_skip():
    v = pa.array([1, None], type=pa.int64())
    s = pa.array([1, None], type=pa.int64())
    a = am.MetalArray.from_arrow(v)
    assert _py(a.is_in(s)) == [True, False]
    assert _py(a.is_in(s, "match")) == [True, True]


@pytest.mark.parametrize("behavior", ["match", "skip", "emit_null", "inconclusive"])
def test_set_lookup_strings(behavior):
    words = ["a", "bb", None, "ccc", "a", None, "dd"] * 40
    value_set = ["bb", None, "a"]
    a = am.MetalArray.from_arrow(pa.array(words, type=pa.string()))
    s = pa.array(value_set, type=pa.string())
    assert _py(a.is_in(s, behavior)) == _expected_is_in(words, value_set, behavior)
    assert _py(a.index_in(s, behavior)) == _expected_index_in(words, value_set, behavior)


def test_set_lookup_large():
    v = ints(BIG, null_every=7, modulus=64)
    s = pa.array([1, 2, None, 63], type=pa.int64())
    a = am.MetalArray.from_arrow(v)
    for behavior, skip_nulls in [("match", False), ("skip", True)]:
        assert _py(a.is_in(s, behavior)) == pc.is_in(v, value_set=s, skip_nulls=skip_nulls).to_pylist()


# ---------------------------------------------------------------------------
# unique / value_counts / dictionary_encode order


@pytest.mark.parametrize("n", SIZES + [BIG])
def test_unique_first_appearance_matches_pyarrow(n):
    v = ints(n, null_every=6, modulus=23)
    assert _py(am.MetalArray.from_arrow(v).unique()) == pc.unique(v).to_pylist()


@pytest.mark.parametrize("n", SIZES)
def test_unique_sorted_order(n):
    v = ints(n, null_every=6, modulus=23)
    got = _py(am.MetalArray.from_arrow(v).unique(order="sorted"))
    assert got == sorted(x for x in set(v.to_pylist()) if x is not None)


@pytest.mark.parametrize("n", SIZES)
def test_value_counts_first_appearance_matches_pyarrow(n):
    v = ints(n, null_every=6, modulus=23)
    got = _py(am.MetalArray.from_arrow(v).value_counts())
    assert got == pc.value_counts(v).to_pylist()


@pytest.mark.parametrize("n", SIZES)
def test_value_counts_sorted_order(n):
    v = ints(n, null_every=6, modulus=23)
    got = _py(am.MetalArray.from_arrow(v).value_counts(order="sorted"))
    rows = v.to_pylist()
    want = [{"values": x, "counts": rows.count(x)}
            for x in sorted(y for y in set(rows) if y is not None)]
    assert got == want


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("order", ["first_appearance", "sorted"])
def test_dictionary_encode_order(n, order):
    v = ints(n, null_every=6, modulus=23)
    codes, values = am.MetalArray.from_arrow(v).dictionary_encode(order=order)
    dictionary = _py(values)
    assert None not in dictionary, "the dictionary never holds a null"
    if order == "first_appearance":
        assert dictionary == [x for x in pc.unique(v).to_pylist() if x is not None]
        assert _py(codes) == pc.dictionary_encode(v).indices.to_pylist()
    else:
        assert dictionary == sorted(dictionary)
    # And decoding must reproduce the input.
    assert [None if c is None else dictionary[c] for c in _py(codes)] == v.to_pylist()


def test_unique_all_null():
    v = pa.array([None, None, None], type=pa.int64())
    a = am.MetalArray.from_arrow(v)
    assert _py(a.unique()) == pc.unique(v).to_pylist() == [None]
    assert _py(a.unique(order="sorted")) == []
    assert _py(a.value_counts()) == pc.value_counts(v).to_pylist()


def test_unique_strings_drops_the_null():
    """The one documented difference: the GPU string dictionary has no slot for a null."""
    v = pa.array(["pear", "apple", "pear", None, "fig", "apple"], type=pa.string())
    a = am.MetalArray.from_arrow(v)
    assert _py(a.unique()) == ["pear", "apple", "fig"]
    assert pc.unique(v).to_pylist() == ["pear", "apple", None, "fig"]
    assert _py(a.unique(order="sorted")) == ["apple", "fig", "pear"]
    assert _py(a.value_counts()) == [{"values": "pear", "counts": 2},
                                     {"values": "apple", "counts": 2},
                                     {"values": "fig", "counts": 1}]


def test_unique_rejects_unknown_order():
    a = am.MetalArray.from_arrow(pa.array([1], type=pa.int64()))
    with pytest.raises(am.ArrowMetalError):
        a.unique(order="whatever")


# ---------------------------------------------------------------------------
# cast


def _cast_or_error(fn):
    try:
        return fn()
    except Exception as e:                                    # noqa: BLE001 - the comparison is the point
        return "error"


CAST_NUMERIC = ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
                "float32", "float64"]


@pytest.mark.parametrize("target", CAST_NUMERIC)
@pytest.mark.parametrize("safe", [False, True])
def test_cast_numeric_matches_pyarrow(target, safe):
    v = pa.array([0, 1, -1, 127, 128, 300, -300, 2 ** 31, None], type=pa.int64())
    a = am.MetalArray.from_arrow(v)
    got = _cast_or_error(lambda: _py(a.cast(target, safe=safe)))
    want = _cast_or_error(lambda: pc.cast(v, pa.type_for_alias(target), safe=safe).to_pylist())
    assert got == want, f"{target} safe={safe}"


@pytest.mark.parametrize("target", ["int8", "int32", "int64", "uint16"])
@pytest.mark.parametrize("safe", [False, True])
def test_cast_float_to_int_matches_pyarrow(target, safe):
    # In range on both sides. The out-of-range unchecked value is a known divergence, pinned by
    # test_cast_out_of_range_float_stays_divergent below and by test_differential.py.
    v = pa.array([0.0, 1.0, 1.5, -2.0, 100.0, None], type=pa.float64())
    a = am.MetalArray.from_arrow(v)
    got = _cast_or_error(lambda: _py(a.cast(target, safe=safe)))
    want = _cast_or_error(lambda: pc.cast(v, pa.type_for_alias(target), safe=safe).to_pylist())
    assert got == want, f"{target} safe={safe}"


def test_cast_out_of_range_float_stays_divergent():
    """`safe=False` on a float too big for the target is undefined in C and the two libraries pick
    different answers: Arrow saturates at the target's width, this saturates at 64 bits and truncates.
    `safe=True` refuses the row instead of choosing, which is the point of the option."""
    v = pa.array([1e20, float("nan"), float("inf")], type=pa.float64())
    a = am.MetalArray.from_arrow(v)
    assert _py(a.cast("int32")) != pc.cast(v, pa.int32(), safe=False).to_pylist()
    with pytest.raises(am.ArrowMetalError):
        a.cast("int32", safe=True)


def test_cast_flags_each_open_one_class():
    over = am.MetalArray.from_arrow(pa.array([300], type=pa.int64()))
    with pytest.raises(am.ArrowMetalError):
        over.cast("int8", safe=True)
    assert _py(over.cast("int8", safe=True, allow_int_overflow=True)) == [44]
    assert _py(over.cast("int8")) == [44]

    frac = am.MetalArray.from_arrow(pa.array([1.5], type=pa.float64()))
    with pytest.raises(am.ArrowMetalError):
        frac.cast("int32", safe=True)
    assert _py(frac.cast("int32", safe=True, allow_float_truncate=True)) == [1]

    lossy = am.MetalArray.from_arrow(pa.array([2 ** 60 + 1], type=pa.int64()))
    with pytest.raises(am.ArrowMetalError):
        lossy.cast("float32", safe=True)
    assert _py(lossy.cast("float32", safe=True, allow_float_truncate=True)) == [2.0 ** 60]


def test_cast_float_to_float_never_raises():
    """Arrow lets a float overflow to infinity, and so does this."""
    v = pa.array([1e300, -1e300, 0.5], type=pa.float64())
    got = _py(am.MetalArray.from_arrow(v).cast("float32", safe=True))
    assert got == pc.cast(v, pa.float32(), safe=True).to_pylist()


def test_cast_nulls_never_raise():
    v = pa.array([None, None], type=pa.int64())
    assert _py(am.MetalArray.from_arrow(v).cast("int8", safe=True)) == [None, None]


def test_cast_bool_both_ways():
    b = pa.array([True, False, None])
    assert _py(am.MetalArray.from_arrow(b).cast("int32")) == pc.cast(b, pa.int32()).to_pylist()
    i = pa.array([0, 1, 5, None], type=pa.int32())
    assert _py(am.MetalArray.from_arrow(i).cast("bool")) == pc.cast(i, pa.bool_()).to_pylist()


def test_cast_strings_both_ways():
    i = pa.array([12, -3, None], type=pa.int32())
    assert _py(am.MetalArray.from_arrow(i).cast("string")) == pc.cast(i, pa.string()).to_pylist()
    s = pa.array(["12", "-3", None], type=pa.string())
    assert _py(am.MetalArray.from_arrow(s).cast("int32")) == pc.cast(s, pa.int32()).to_pylist()


@pytest.mark.parametrize("safe", [False, True])
def test_cast_temporal_units(safe):
    v = pa.array([1_500_000_000, 2_000_000_000, None], type=pa.timestamp("ns"))
    a = am.MetalArray.from_arrow(v)
    got = _cast_or_error(lambda: _py(a.cast(pa.timestamp("s"), safe=safe)))
    want = _cast_or_error(lambda: pc.cast(v, pa.timestamp("s"), safe=safe).to_pylist())
    assert got == want


def test_cast_temporal_flags():
    lossy = am.MetalArray.from_arrow(pa.array([1_500_000_000], type=pa.timestamp("ns")))
    with pytest.raises(am.ArrowMetalError):
        lossy.cast(pa.timestamp("s"), safe=True)
    assert _py(lossy.cast(pa.timestamp("s"), safe=True, allow_time_truncate=True)) == \
        pc.cast(pa.array([1_500_000_000], type=pa.timestamp("ns")), pa.timestamp("s"),
                safe=False).to_pylist()

    big = am.MetalArray.from_arrow(pa.array([10 ** 12], type=pa.timestamp("s")))
    with pytest.raises(am.ArrowMetalError):
        big.cast(pa.timestamp("ns"), safe=True)
    assert _py(big.cast(pa.timestamp("ns"), safe=True, allow_time_overflow=True)) == \
        pc.cast(pa.array([10 ** 12], type=pa.timestamp("s")), pa.timestamp("ns"),
                safe=False).to_pylist()


def test_cast_date_and_timestamp():
    d = pa.array([1, 0, None], type=pa.date32())
    a = am.MetalArray.from_arrow(d)
    assert _py(a.cast(pa.timestamp("s"))) == pc.cast(d, pa.timestamp("s")).to_pylist()
    assert _py(a.cast(pa.date64())) == pc.cast(d, pa.date64()).to_pylist()
    ts = pa.array([86_405, 0, None], type=pa.timestamp("s"))
    assert _py(am.MetalArray.from_arrow(ts).cast(pa.date32())) == pc.cast(ts, pa.date32()).to_pylist()


def test_cast_decimal():
    import decimal
    v = pa.array([123, -5, None], type=pa.int64())
    got = _py(am.MetalArray.from_arrow(v).cast(pa.decimal128(21, 2)))
    assert got == pc.cast(v, pa.decimal128(21, 2)).to_pylist()

    v32 = pa.array([123, -5, None], type=pa.int32())
    assert _py(am.MetalArray.from_arrow(v32).cast(pa.decimal128(12, 2))) == \
        pc.cast(v32, pa.decimal128(12, 2)).to_pylist()

    d = pa.array([decimal.Decimal("1.234"), None], type=pa.decimal128(10, 3))
    a = am.MetalArray.from_arrow(d)
    assert _py(a.cast(pa.decimal128(12, 3))) == d.to_pylist()
    with pytest.raises(am.ArrowMetalError):
        a.cast(pa.decimal128(10, 1), safe=True)
    assert _py(a.cast(pa.decimal128(10, 1), safe=True, allow_decimal_truncate=True)) == \
        pc.cast(d, options=pc.CastOptions(target_type=pa.decimal128(10, 1),
                                          allow_decimal_truncate=True)).to_pylist()


def test_cast_nested_list_and_struct():
    lst = pa.array([[1, 2], None, [3]], type=pa.list_(pa.int32()))
    got = _py(am.MetalArray.from_arrow(lst).cast(pa.list_(pa.int64())))
    assert got == pc.cast(lst, pa.list_(pa.int64())).to_pylist()

    over = pa.array([[300]], type=pa.list_(pa.int64()))
    with pytest.raises(am.ArrowMetalError):
        am.MetalArray.from_arrow(over).cast(pa.list_(pa.int8()), safe=True)

    st = pa.array([{"a": 1, "b": 2.5}, None], type=pa.struct([("a", pa.int32()), ("b", pa.float64())]))
    target = pa.struct([("a", pa.int64()), ("b", pa.float32())])
    assert _py(am.MetalArray.from_arrow(st).cast(target)) == pc.cast(st, target).to_pylist()


def test_cast_default_is_unsafe():
    """ArrowMetal's own default stays the unchecked conversion; pyarrow's default raises."""
    v = pa.array([300], type=pa.int64())
    assert _py(am.MetalArray.from_arrow(v).cast("int8")) == [44]
    with pytest.raises(Exception):
        pc.cast(v, pa.int8())


def test_cast_reports_the_first_offending_row():
    v = pa.array([1, 2, 300, 400], type=pa.int64())
    with pytest.raises(am.ArrowMetalError) as e:
        am.MetalArray.from_arrow(v).cast("int8", safe=True)
    assert "index 2" in str(e.value)


@pytest.mark.parametrize("n", [4097])
def test_cast_safe_large(n):
    v = pa.array([i % 100 for i in range(n)], type=pa.int64())
    assert _py(am.MetalArray.from_arrow(v).cast("int8", safe=True)) == v.to_pylist()


# ---------------------------------------------------------------------------
# round_temporal / ceil_temporal / floor_temporal


ROUND_UNITS = ["nanosecond", "microsecond", "millisecond", "second", "minute", "hour",
               "day", "week", "month", "quarter", "year"]


def timestamps(unit="s"):
    rows = ["2024-05-17T13:47:33", "2024-02-29T23:10:00", "1970-01-01T00:00:00",
            "1969-12-31T23:59:59", "2024-01-01T00:00:00", "2000-03-01T12:00:00",
            "1912-07-04T06:30:15"]
    return pa.array([datetime.datetime.fromisoformat(s) for s in rows] + [None]).cast(
        pa.timestamp(unit))


@pytest.mark.parametrize("unit", ROUND_UNITS)
@pytest.mark.parametrize("multiple", [1, 3, 7])
@pytest.mark.parametrize("mode", ["floor", "ceil", "round"])
@pytest.mark.parametrize("storage", ["s", "ms", "us", "ns"])
def test_temporal_rounding_units_and_multiples(unit, multiple, mode, storage):
    v = timestamps(storage)
    a = am.MetalArray.from_arrow(v)
    got = _py(getattr(a, f"{mode}_temporal")(unit, multiple))
    want = getattr(pc, f"{mode}_temporal")(v, multiple=multiple, unit=unit).to_pylist()
    assert got == want, f"{mode} {unit} x{multiple} [{storage}]"


@pytest.mark.parametrize("unit", ROUND_UNITS)
@pytest.mark.parametrize("multiple", [1, 5])
@pytest.mark.parametrize("mode", ["floor", "ceil", "round"])
def test_temporal_rounding_calendar_based_origin(unit, multiple, mode):
    if unit == "week" and multiple > 1:
        pytest.skip("see test_calendar_week_multiple_is_arrows_defect")
    v = timestamps("ns")
    a = am.MetalArray.from_arrow(v)
    got = _py(getattr(a, f"{mode}_temporal")(unit, multiple, calendar_based_origin=True))
    want = getattr(pc, f"{mode}_temporal")(v, multiple=multiple, unit=unit,
                                           calendar_based_origin=True).to_pylist()
    assert got == want, f"{mode} {unit} x{multiple} calendar_based_origin"


def test_calendar_week_multiple_is_arrows_defect():
    """`floor_temporal(unit="week", multiple>1, calendar_based_origin=True)` in Arrow 21 returns a
    value **greater than its input** for a fifth of all days — it rounds the week index instead of
    flooring it. ArrowMetal floors, so its answer is on the same grid and never exceeds the input.
    This test pins both halves of that, so it fails the day Arrow fixes it.
    """
    days = [datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(120)]
    v = pa.array(days, type=pa.date32())
    theirs = pc.floor_temporal(v, multiple=2, unit="week", calendar_based_origin=True).to_pylist()
    ours = _py(am.MetalArray.from_arrow(v).floor_temporal("week", 2, calendar_based_origin=True))
    assert any(t > d for t, d in zip(theirs, days)), "Arrow no longer overshoots; drop this test"
    assert all(o <= d for o, d in zip(ours, days)), "a floor must never exceed its input"
    grid = sorted(set(theirs) | set(ours))
    for a, b in zip(grid, grid[1:]):
        assert (b - a).days % 14 == 0, "both sit on the same 14-day grid"


@pytest.mark.parametrize("multiple", [1, 2, 3])
@pytest.mark.parametrize("week_starts_monday", [True, False])
@pytest.mark.parametrize("calendar", [False, True])
@pytest.mark.parametrize("mode", ["floor", "ceil", "round"])
def test_temporal_rounding_weeks(multiple, week_starts_monday, calendar, mode):
    if calendar and multiple > 1:
        pytest.skip("see test_calendar_week_multiple_is_arrows_defect")
    v = timestamps("s")
    a = am.MetalArray.from_arrow(v)
    got = _py(getattr(a, f"{mode}_temporal")("week", multiple,
                                             week_starts_monday=week_starts_monday,
                                             calendar_based_origin=calendar))
    want = getattr(pc, f"{mode}_temporal")(v, multiple=multiple, unit="week",
                                           week_starts_monday=week_starts_monday,
                                           calendar_based_origin=calendar).to_pylist()
    assert got == want


@pytest.mark.parametrize("unit", ROUND_UNITS)
@pytest.mark.parametrize("multiple", [1, 3])
def test_temporal_ceil_is_strictly_greater(unit, multiple):
    v = timestamps("s")
    a = am.MetalArray.from_arrow(v)
    got = _py(a.ceil_temporal(unit, multiple, ceil_is_strictly_greater=True))
    want = pc.ceil_temporal(v, multiple=multiple, unit=unit,
                            ceil_is_strictly_greater=True).to_pylist()
    assert got == want


@pytest.mark.parametrize("unit", ["day", "week", "month", "quarter", "year"])
@pytest.mark.parametrize("multiple", [1, 4])
@pytest.mark.parametrize("mode", ["floor", "ceil", "round"])
def test_temporal_rounding_date32(unit, multiple, mode):
    v = pa.array([datetime.date(2024, 5, 17), datetime.date(1969, 3, 3), datetime.date(1970, 1, 1),
                  None], type=pa.date32())
    got = _py(getattr(am.MetalArray.from_arrow(v), f"{mode}_temporal")(unit, multiple))
    want = getattr(pc, f"{mode}_temporal")(v, multiple=multiple, unit=unit).to_pylist()
    assert got == want


def test_temporal_rounding_halves_go_up():
    """Arrow rounds an exact half toward +infinity, not to even; so does this."""
    rows = [0, 1, 2, 3, 4, 5, -1, -3]
    v = pa.array(rows, type=pa.timestamp("s"))
    got = _py(am.MetalArray.from_arrow(v).round_temporal("second", 2))
    assert got == pc.round_temporal(v, multiple=2, unit="second").to_pylist()


def test_temporal_rounding_rejects_bad_arguments():
    a = am.MetalArray.from_arrow(timestamps())
    with pytest.raises(am.ArrowMetalError):
        a.floor_temporal("fortnight")
    with pytest.raises(am.ArrowMetalError):
        a.floor_temporal("day", 0)


def test_temporal_rounding_large():
    n = 200_003
    v = pa.array([None if i % 9 == 4 else i * 37 for i in range(n)], type=pa.timestamp("s"))
    a = am.MetalArray.from_arrow(v)
    for unit, mult in [("hour", 5), ("week", 2), ("month", 3)]:
        assert _py(a.floor_temporal(unit, mult)) == \
            pc.floor_temporal(v, multiple=mult, unit=unit).to_pylist()


# ---------------------------------------------------------------------------
# list_parent_indices


@pytest.mark.parametrize("width", ["int32", "int64"])
def test_list_parent_indices_widths(width):
    lst = pa.array([[1, 2, 3], [4], None, [], [5, 6]], type=pa.list_(pa.int64()))
    a = am.MetalArray.from_arrow(lst)
    want = pc.list_parent_indices(lst)
    if width == "int64":
        got = a.list_parent_indices64()
        assert got.to_arrow().type == pa.int64()
        assert _py(got) == want.to_pylist()
    else:
        got = a.list_parent_indices()
        assert got.to_arrow().type == pa.int32()
        assert _py(got) == want.cast(pa.int32()).to_pylist()


def test_list_parent_indices_large():
    n = 50_000
    lst = pa.array([[i, i + 1] if i % 3 else None for i in range(n)], type=pa.list_(pa.int64()))
    got = _py(am.MetalArray.from_arrow(lst).list_parent_indices64())
    assert got == pc.list_parent_indices(lst).to_pylist()


# ---------------------------------------------------------------------------
# the option names themselves


def test_option_tables_are_the_arrow_spellings():
    assert am.NULL_PLACEMENT == ["at_end", "at_start"]
    assert set(am.TIEBREAKERS) == {"min", "max", "first", "dense"}
    assert set(am.NULL_MATCHING) == {"match", "skip", "emit_null", "inconclusive"}
    assert set(am.VALUE_ORDERS) == {"first_appearance", "sorted"}
    assert set(am.CAST_FLAGS) == {"allow_int_overflow", "allow_time_truncate", "allow_time_overflow",
                                  "allow_decimal_truncate", "allow_float_truncate",
                                  "allow_invalid_utf8"}


@pytest.mark.parametrize("call,kwargs", [
    ("argsort", {"null_placement": "nowhere"}),
    ("rank", {"null_placement": "nowhere"}),
    ("rank", {"tiebreaker": "coin_toss"}),
    ("rank", {"sort_keys": "sideways"}),
    ("unique", {"order": "random"}),
])
def test_unknown_option_values_raise(call, kwargs):
    a = am.MetalArray.from_arrow(pa.array([1, 2], type=pa.int64()))
    with pytest.raises(am.ArrowMetalError):
        getattr(a, call)(**kwargs)


def test_unknown_cast_flag_raises():
    a = am.MetalArray.from_arrow(pa.array([1, 2], type=pa.int64()))
    with pytest.raises(am.ArrowMetalError):
        a.cast("int8", safe=True, allow_anything=True)
