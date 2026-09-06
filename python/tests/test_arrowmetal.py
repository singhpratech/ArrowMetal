"""End-to-end tests for the arrowmetal Python package.

Every result is checked against pyarrow.compute or plain Python on the same data, so the GPU
kernels are compared to a CPU oracle rather than to themselves.

    PYTHONPATH=python python -m pytest python/tests -q

Needs a real Metal device and libArrowMetalC.dylib (bundled in the wheel, or built with
`swift build -c release --product ArrowMetalC`, or pointed at by $ARROWMETAL_LIB).
"""
import math

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import arrowmetal as am

try:
    import polars as pl
except ImportError:  # pragma: no cover - exercised only where polars is absent
    pl = None


# ---------------------------------------------------------------- fixtures / helpers

INT64 = pa.array([1, None, 3, 40, -7, 0, 12, None, 5], pa.int64())
FLOAT64 = pa.array([1.5, None, 3.25, -0.5, 100.125, 0.0, None, 7.75, 2.0], pa.float64())
FLOAT32 = pa.array([1.5, None, 3.25, -0.5, 100.125, 0.0, None, 7.75, 2.0], pa.float32())
BOOL = pa.array([True, None, False, True, False, True, None, False, True], pa.bool_())
STRINGS = pa.array(["apple", "banana", None, "apricot", "cherry", "app", None, "banana", "date"], pa.string())

ALL_ARRAYS = {"int64": INT64, "float64": FLOAT64, "float32": FLOAT32, "bool": BOOL, "string": STRINGS}


def pylist(x):
    """MetalArray -> plain Python list, through the Arrow C Data Interface."""
    return x.to_arrow().to_pylist()


def approx(values):
    return [v if v is None else pytest.approx(v, rel=1e-6, abs=1e-9) for v in values]


# ---------------------------------------------------------------- import / round trip


def test_device_and_version():
    assert isinstance(am.device_name(), str) and am.device_name()
    assert am.version() == "0.1.0"
    assert am.__version__ == "0.1.0"


@pytest.mark.parametrize("name", sorted(ALL_ARRAYS))
def test_import_round_trip_preserves_values_types_and_nulls(name):
    source = ALL_ARRAYS[name]
    x = am.array(source)

    assert len(x) == len(source)
    assert x.null_count == source.null_count
    assert x.type == source.type

    out = x.to_arrow()
    assert out.type == source.type
    assert out.to_pylist() == source.to_pylist()
    assert out.equals(source)


def test_import_accepts_chunked_array_and_python_list():
    chunked = pa.chunked_array([pa.array([1, None], pa.int64()), pa.array([3], pa.int64())])
    assert pylist(am.array(chunked)) == [1, None, 3]
    assert pylist(am.array(pa.array([4, 5, 6], pa.int64()))) == [4, 5, 6]


def test_repr_mentions_type_and_device():
    r = repr(am.array(INT64))
    assert "int64" in r and am.device_name() in r


# ---------------------------------------------------------------- reductions


@pytest.mark.parametrize("name", ["int64", "float64", "float32"])
def test_sum_min_max_mean_match_pyarrow(name):
    source = ALL_ARRAYS[name]
    x = am.array(source)

    assert x.sum() == pytest.approx(pc.sum(source).as_py(), rel=1e-6)
    assert x.min() == pytest.approx(pc.min(source).as_py(), rel=1e-6)
    assert x.max() == pytest.approx(pc.max(source).as_py(), rel=1e-6)
    assert x.mean() == pytest.approx(pc.mean(source).as_py(), rel=1e-6)


def test_int64_reductions_exactly():
    valid = [v for v in INT64.to_pylist() if v is not None]
    x = am.array(INT64)
    assert x.sum() == sum(valid)
    assert x.min() == min(valid)
    assert x.max() == max(valid)
    assert x.mean() == pytest.approx(sum(valid) / len(valid))


def test_reductions_on_all_null_and_empty_return_none():
    all_null = am.array(pa.array([None, None, None], pa.int64()))
    assert all_null.sum() is None and all_null.min() is None and all_null.mean() is None

    empty = am.array(pa.array([], pa.int64()))
    assert len(empty) == 0
    assert empty.sum() is None and empty.max() is None


# ---------------------------------------------------------------- compare and filter


@pytest.mark.parametrize("op,threshold", [("==", 3), ("!=", 3), ("<", 3), ("<=", 3), (">", 3), (">=", 3)])
def test_compare_scalar_matches_pyarrow(op, threshold):
    expected = {"==": pc.equal, "!=": pc.not_equal, "<": pc.less, "<=": pc.less_equal,
                ">": pc.greater, ">=": pc.greater_equal}[op](INT64, threshold)
    got = am.array(INT64).compare(op, threshold)
    assert got.type == pa.bool_()
    assert pylist(got) == expected.to_pylist()


def test_compare_operators_and_filter_match_pyarrow():
    x = am.array(INT64)
    mask = x > 2
    assert pylist(mask) == pc.greater(INT64, 2).to_pylist()

    kept = x.filter(mask)
    assert pylist(kept) == INT64.filter(pc.greater(INT64, 2)).to_pylist()


def test_compare_array_to_array():
    other = pa.array([1, 1, 1, 1, 1, 1, 1, 1, 1], pa.int64())
    got = am.array(INT64).compare(">", am.array(other))
    assert pylist(got) == pc.greater(INT64, other).to_pylist()


def test_filter_drops_rows_where_the_mask_is_null():
    values = pa.array([1, 2, 3, 4], pa.int64())
    mask = pa.array([True, False, None, True], pa.bool_())
    got = am.array(values).filter(am.array(mask))
    assert pylist(got) == values.filter(mask).to_pylist() == [1, 4]


def test_filter_where_equals_compare_then_filter():
    x = am.array(INT64)
    fused = x.filter_where(">", 2)
    two_step = x.filter(x > 2)
    expected = INT64.filter(pc.greater(INT64, 2)).to_pylist()
    assert pylist(fused) == expected
    assert pylist(two_step) == expected
    assert fused.null_count == 0


def test_filter_where_on_float64():
    x = am.array(FLOAT64)
    got = x.filter_where(">=", 2.0)
    expected = FLOAT64.filter(pc.greater_equal(FLOAT64, 2.0))
    assert pylist(got) == approx(expected.to_pylist())


def test_filter_keeping_nothing_yields_empty_array():
    got = am.array(INT64).filter_where(">", 10_000)
    assert len(got) == 0 and pylist(got) == []


# ---------------------------------------------------------------- take and slice


def test_take_matches_pyarrow():
    indices = [8, 0, 3, 1, 4]
    got = am.array(INT64).take(indices)
    assert pylist(got) == INT64.take(pa.array(indices, pa.int32())).to_pylist()


def test_take_accepts_a_metal_index_array_and_repeats():
    idx = am.array(pa.array([2, 2, 2, 0], pa.int32()))
    assert pylist(am.array(INT64).take(idx)) == [3, 3, 3, 1]


def test_take_on_strings():
    got = am.array(STRINGS).take([1, 2, 4])
    assert pylist(got) == STRINGS.take(pa.array([1, 2, 4], pa.int32())).to_pylist()


def test_slice_matches_pyarrow():
    got = am.array(INT64).slice(2, 4)
    assert pylist(got) == INT64.slice(2, 4).to_pylist()


# ---------------------------------------------------------------- cast


@pytest.mark.parametrize("alias,arrow_type", [("int32", pa.int32()), ("float32", pa.float32()),
                                              ("float64", pa.float64()), ("int16", pa.int16())])
def test_cast_from_int64_matches_pyarrow(alias, arrow_type):
    got = am.array(INT64).cast(alias)
    assert got.type == arrow_type
    assert pylist(got) == INT64.cast(arrow_type).to_pylist()


def test_cast_float64_to_float32_and_back():
    x = am.array(FLOAT64)
    narrowed = x.cast("float32")
    assert narrowed.type == pa.float32()
    assert pylist(narrowed) == approx(FLOAT64.cast(pa.float32()).to_pylist())
    assert pylist(narrowed.cast("float64")) == approx(FLOAT64.cast(pa.float32()).cast(pa.float64()).to_pylist())


def test_cast_accepts_a_one_letter_arrow_format():
    assert am.array(INT64).cast("g").type == pa.float64()


# ---------------------------------------------------------------- arithmetic


@pytest.mark.parametrize("op,scalar,fn", [("+", 10, pc.add), ("-", 3, pc.subtract), ("*", 7, pc.multiply)])
def test_int64_scalar_arithmetic_matches_pyarrow(op, scalar, fn):
    got = am.array(INT64).arith(op, scalar)
    assert pylist(got) == fn(INT64, pa.scalar(scalar, pa.int64())).to_pylist()


def test_int64_operators():
    x = am.array(INT64)
    assert pylist(x * 2) == pc.multiply(INT64, pa.scalar(2, pa.int64())).to_pylist()
    assert pylist(x + 1) == pc.add(INT64, pa.scalar(1, pa.int64())).to_pylist()
    assert pylist(x - 1) == pc.subtract(INT64, pa.scalar(1, pa.int64())).to_pylist()


def test_array_arithmetic_matches_pyarrow():
    other = pa.array([2, 2, 2, 2, 2, 2, 2, 2, 2], pa.int64())
    got = am.array(INT64) + am.array(other)
    assert pylist(got) == pc.add(INT64, other).to_pylist()


@pytest.mark.parametrize("op,scalar,fn", [("+", 2.5, pc.add), ("-", 1.25, pc.subtract),
                                          ("*", 3.5, pc.multiply), ("/", 2.0, pc.divide)])
def test_float64_scalar_arithmetic_matches_pyarrow(op, scalar, fn):
    got = am.array(FLOAT64).arith(op, scalar)
    assert got.type == pa.float64()
    assert pylist(got) == approx(fn(FLOAT64, pa.scalar(scalar, pa.float64())).to_pylist())


def test_float64_array_arithmetic_and_reduction_round_trip():
    x = am.array(FLOAT64)
    doubled = x * 2.0
    assert pylist(doubled) == approx([None if v is None else v * 2.0 for v in FLOAT64.to_pylist()])
    assert doubled.sum() == pytest.approx(pc.sum(pc.multiply(FLOAT64, 2.0)).as_py())

    summed = x + x
    assert pylist(summed) == approx(pc.add(FLOAT64, FLOAT64).to_pylist())


def test_float32_arithmetic_matches_pyarrow():
    got = am.array(FLOAT32) * 1.5
    assert got.type == pa.float32()
    assert pylist(got) == approx(pc.multiply(FLOAT32, pa.scalar(1.5, pa.float32())).to_pylist())


def test_arithmetic_propagates_nulls():
    got = am.array(INT64) * 2
    assert [v is None for v in pylist(got)] == [v is None for v in INT64.to_pylist()]


# ---------------------------------------------------------------- booleans


def test_boolean_and_or_not_match_pyarrow():
    a = pa.array([True, None, False, True, False, True, None, False, True], pa.bool_())
    b = pa.array([True, True, True, False, None, True, False, False, None], pa.bool_())
    x, y = am.array(a), am.array(b)

    assert pylist(x & y) == pc.and_(a, b).to_pylist()
    assert pylist(x | y) == pc.or_(a, b).to_pylist()
    assert pylist(~x) == pc.invert(a).to_pylist()


def test_boolean_masks_combine_and_filter():
    x = am.array(INT64)
    mask = (x > 0) & (x < 20)
    expected_mask = pc.and_(pc.greater(INT64, 0), pc.less(INT64, 20))
    assert pylist(mask) == expected_mask.to_pylist()
    assert pylist(x.filter(mask)) == INT64.filter(expected_mask).to_pylist()


def test_boolean_or_of_two_ranges():
    x = am.array(INT64)
    mask = (x < 0) | (x > 10)
    expected = pc.or_(pc.less(INT64, 0), pc.greater(INT64, 10))
    assert pylist(mask) == expected.to_pylist()


def test_boolean_round_trip_with_nulls():
    x = am.array(BOOL)
    assert x.type == pa.bool_()
    assert x.null_count == BOOL.null_count
    assert pylist(x) == BOOL.to_pylist()


# ---------------------------------------------------------------- group-by


KEYS = pa.array([0, 1, 0, 2, 1, 0, 2, 1, 3], pa.int32())
VALUES64 = pa.array([10, 20, 30, 40, None, 60, 70, 80, 90], pa.int64())
VALUES32 = VALUES64.cast(pa.int32())
KEY_COUNT = 4


def _expected_groups(keys, values, key_count):
    """{key: [non-null values]} computed in plain Python."""
    groups = {k: [] for k in range(key_count)}
    for k, v in zip(keys.to_pylist(), values.to_pylist()):
        if k is not None and v is not None:
            groups[k].append(v)
    return groups


def test_group_by_count_counts_rows_per_key():
    got = pylist(am.array(KEYS).group_by(KEY_COUNT).count())
    expected = [KEYS.to_pylist().count(k) for k in range(KEY_COUNT)]
    assert got == expected == [3, 3, 2, 1]


def test_group_by_sum_matches_python():
    groups = _expected_groups(KEYS, VALUES64, KEY_COUNT)
    got = pylist(am.array(KEYS).group_by(KEY_COUNT).sum(am.array(VALUES64)))
    assert got == [sum(groups[k]) if groups[k] else None for k in range(KEY_COUNT)]


def test_group_by_mean_matches_python():
    groups = _expected_groups(KEYS, VALUES64, KEY_COUNT)
    got = pylist(am.array(KEYS).group_by(KEY_COUNT).mean(am.array(VALUES64)))
    expected = [sum(groups[k]) / len(groups[k]) if groups[k] else None for k in range(KEY_COUNT)]
    assert got == approx(expected)


def test_group_by_min_and_max_match_python():
    groups = _expected_groups(KEYS, VALUES32, KEY_COUNT)
    g = am.array(KEYS).group_by(KEY_COUNT)
    assert pylist(g.min(am.array(VALUES32))) == [min(groups[k]) if groups[k] else None for k in range(KEY_COUNT)]
    assert pylist(g.max(am.array(VALUES32))) == [max(groups[k]) if groups[k] else None for k in range(KEY_COUNT)]


def test_group_by_matches_pyarrow_table_aggregate():
    table = pa.table({"key": KEYS, "value": VALUES64})
    grouped = table.group_by("key").aggregate([("value", "sum"), ("value", "mean")])
    by_key = {k: (s, m) for k, s, m in zip(grouped["key"].to_pylist(),
                                           grouped["value_sum"].to_pylist(),
                                           grouped["value_mean"].to_pylist())}

    g = am.array(KEYS).group_by(KEY_COUNT)
    got_sum = pylist(g.sum(am.array(VALUES64)))
    got_mean = pylist(g.mean(am.array(VALUES64)))
    for k in range(KEY_COUNT):
        expected_sum, expected_mean = by_key.get(k, (None, None))
        assert got_sum[k] == expected_sum
        if expected_mean is None:
            assert got_mean[k] is None
        else:
            assert got_mean[k] == pytest.approx(expected_mean)


def test_group_by_float32_sum_min_max():
    keys = pa.array([0, 1, 0, 1], pa.int32())
    values = pa.array([1.5, 2.5, 3.5, 4.5], pa.float32())
    g = am.array(keys).group_by(2)
    assert pylist(g.sum(am.array(values))) == approx([5.0, 7.0])
    assert pylist(g.min(am.array(values))) == approx([1.5, 2.5])
    assert pylist(g.max(am.array(values))) == approx([3.5, 4.5])


@pytest.mark.parametrize("agg,value_type", [
    ("min", pa.int64()),        # group-by min/max are 32-bit or narrower only
    ("max", pa.int64()),
    ("min", pa.uint64()),
    ("max", pa.uint64()),
    ("mean", pa.float32()),     # group-by mean is integer-valued only
    ("sum", pa.float64()),      # float64 has no group-by kernel at all
    ("mean", pa.float64()),
    ("min", pa.float64()),
    ("max", pa.float64()),
])
def test_group_by_combinations_the_kernels_do_not_cover_raise_clearly(agg, value_type):
    """Documents the gaps in the group-by kernels; delete a row here when one is implemented."""
    keys = pa.array([0, 1, 0, 1], pa.int32())
    values = am.array(pa.array([1, 2, 3, 4], value_type))
    with pytest.raises(am.ArrowMetalError):
        getattr(am.array(keys).group_by(2), agg)(values).to_arrow()


def test_group_by_key_with_no_rows_yields_zero_count_and_null_aggregate():
    keys = pa.array([0, 0, 2], pa.int32())
    values = pa.array([1, 2, 3], pa.int64())
    g = am.array(keys).group_by(3)
    assert pylist(g.count()) == [2, 0, 1]
    assert pylist(g.sum(am.array(values))) == [3, None, 3]


# ---------------------------------------------------------------- strings


def test_byte_length_matches_pyarrow():
    got = am.array(STRINGS).byte_length()
    assert got.type == pa.int32()
    assert pylist(got) == pc.binary_length(STRINGS).cast(pa.int32()).to_pylist()


def test_starts_with_and_ends_with_match_pyarrow():
    x = am.array(STRINGS)
    assert pylist(x.starts_with("ap")) == pc.starts_with(STRINGS, "ap").to_pylist()
    assert pylist(x.ends_with("na")) == pc.ends_with(STRINGS, "na").to_pylist()


def test_str_contains_matches_pyarrow():
    got = am.array(STRINGS).str_contains("an")
    assert pylist(got) == pc.match_substring(STRINGS, "an").to_pylist()


def test_str_equals_scalar_and_filter():
    x = am.array(STRINGS)
    mask = x.str_equals("banana")
    assert pylist(mask) == pc.equal(STRINGS, "banana").to_pylist()
    assert pylist(x.filter(mask)) == ["banana", "banana"]


def test_dictionary_encode_reproduces_the_original_strings():
    codes, unique = am.array(STRINGS).dictionary_encode()
    assert codes.type == pa.int32()
    assert unique.type == pa.string()

    uniques = pylist(unique)
    assert uniques == list(dict.fromkeys(v for v in STRINGS.to_pylist() if v is not None))

    decoded = [None if c is None else uniques[c] for c in pylist(codes)]
    assert decoded == STRINGS.to_pylist()


def test_dictionary_encode_then_group_by_counts_each_string():
    codes, unique = am.array(STRINGS).dictionary_encode()
    uniques = pylist(unique)
    counts = pylist(codes.group_by(len(uniques)).count())

    expected = [STRINGS.to_pylist().count(s) for s in uniques]
    assert counts == expected
    assert sum(counts) == len(STRINGS) - STRINGS.null_count


def test_dictionary_encode_then_group_by_sum_of_a_value_column():
    strings = pa.array(["a", "b", "a", "c", "b", "a"], pa.string())
    values = pa.array([1, 2, 3, 4, 5, 6], pa.int64())
    codes, unique = am.array(strings).dictionary_encode()
    uniques = pylist(unique)
    sums = pylist(codes.group_by(len(uniques)).sum(am.array(values)))

    expected = {}
    for s, v in zip(strings.to_pylist(), values.to_pylist()):
        expected[s] = expected.get(s, 0) + v
    assert dict(zip(uniques, sums)) == expected


# ---------------------------------------------------------------- batching


def test_batch_produces_the_same_results_as_unbatched():
    x = am.array(INT64)

    unbatched_mask = pylist((x > 1) & (x < 40))
    unbatched_kept = pylist(x.filter((x > 1) & (x < 40)))
    unbatched_scaled = pylist(x * 3)

    with am.batch():
        mask = (x > 1) & (x < 40)
        kept = x.filter(mask)
        scaled = x * 3

    assert pylist(mask) == unbatched_mask
    assert pylist(kept) == unbatched_kept
    assert pylist(scaled) == unbatched_scaled


def test_batch_reduction_inside_the_context_matches_unbatched():
    x = am.array(FLOAT64)
    expected = x.filter_where(">", 1.0).sum()

    with am.batch():
        total = x.filter_where(">", 1.0).sum()

    assert total == pytest.approx(expected)
    assert total == pytest.approx(pc.sum(FLOAT64.filter(pc.greater(FLOAT64, 1.0))).as_py())


def test_batch_around_a_group_by_matches_unbatched():
    unbatched = pylist(am.array(KEYS).group_by(KEY_COUNT).sum(am.array(VALUES64)))
    with am.batch():
        batched = pylist(am.array(KEYS).group_by(KEY_COUNT).sum(am.array(VALUES64)))
    assert batched == unbatched


def test_batch_context_can_be_reentered():
    x = am.array(INT64)
    for _ in range(3):
        with am.batch():
            got = pylist(x.filter_where(">=", 3))
        assert got == INT64.filter(pc.greater_equal(INT64, 3)).to_pylist()


# ---------------------------------------------------------------- pipeline / interop


def test_query_shaped_pipeline_matches_pyarrow():
    region = pa.array([0, 1, 2, 1, 2, 0, 2, 1, 2], pa.int32())
    amount = pa.array([5.0, 150.0, 200.0, 90.0, 300.0, 10.0, None, 400.0, 50.0], pa.float64())

    mask = pc.and_(pc.equal(region, 2), pc.greater(amount, 100.0))
    expected = pc.sum(amount.filter(mask)).as_py()

    r, a = am.array(region), am.array(amount)
    with am.batch():
        total = a.filter((r == 2) & (a > 100.0)).sum()

    assert total == pytest.approx(expected)


@pytest.mark.skipif(pl is None, reason="polars is not installed")
def test_polars_round_trip_int64():
    series = pl.Series("x", [1, None, 3, 40])
    x = am.array(series.to_arrow())

    back = pl.from_arrow(x.to_arrow())
    assert back.to_list() == series.to_list()

    kept = pl.from_arrow(x.filter_where(">", 2).to_arrow())
    assert kept.to_list() == [3, 40]
    assert kept.sum() == 43


@pytest.mark.skipif(pl is None, reason="polars is not installed")
def test_polars_round_trip_float64_and_reduction():
    series = pl.Series("v", [1.5, None, 3.25, 100.125])
    x = am.array(series.to_arrow())

    assert x.sum() == pytest.approx(series.sum())
    assert x.mean() == pytest.approx(series.mean())

    doubled = pl.from_arrow((x * 2.0).to_arrow())
    assert doubled.to_list() == pytest.approx((series * 2.0).to_list(), nan_ok=True, rel=1e-9)


@pytest.mark.skipif(pl is None, reason="polars is not installed")
def test_polars_group_by_matches_arrowmetal_group_by():
    frame = pl.DataFrame({"key": pl.Series([0, 1, 0, 2, 1], dtype=pl.Int32),
                          "value": pl.Series([10, 20, 30, 40, 50], dtype=pl.Int64)})
    expected = {row[0]: row[1] for row in frame.group_by("key").agg(pl.col("value").sum()).rows()}

    keys = am.array(frame["key"].to_arrow())
    values = am.array(frame["value"].to_arrow())
    got = pylist(keys.group_by(3).sum(values))

    assert got == [expected.get(k) for k in range(3)]


@pytest.mark.skipif(pl is None, reason="polars is not installed")
def test_polars_string_column_dictionary_encode():
    series = pl.Series("s", ["x", "y", "x", "z", "y", "x"])
    codes, unique = am.array(series.to_arrow()).dictionary_encode()
    uniques = pylist(unique)
    counts = pylist(codes.group_by(len(uniques)).count())

    expected = series.value_counts().to_dict(as_series=False)
    expected = dict(zip(expected["s"], expected["count"]))
    assert dict(zip(uniques, counts)) == expected


# ---------------------------------------------------------------- trigonometry, logic, conditionals


TRIG_DOMAIN = {
    "sin": (-20.0, 20.0), "cos": (-20.0, 20.0), "tan": (-20.0, 20.0),
    "asin": (-1.0, 1.0), "acos": (-1.0, 1.0), "atan": (-50.0, 50.0),
    "sinh": (-10.0, 10.0), "cosh": (-10.0, 10.0), "tanh": (-10.0, 10.0),
    "asinh": (-50.0, 50.0), "acosh": (1.0, 50.0), "atanh": (-0.99, 0.99),
}


def _trig_column(lo, hi, ty):
    step = (hi - lo) / 40.0
    vals = [None if i % 7 == 3 else lo + step * i for i in range(41)]
    return pa.array(vals, type=ty)


@pytest.mark.parametrize("name", sorted(TRIG_DOMAIN))
@pytest.mark.parametrize("ty", [pa.float64(), pa.float32()])
def test_trig_matches_pyarrow(name, ty):
    lo, hi = TRIG_DOMAIN[name]
    src = _trig_column(lo, hi, ty)
    got = pylist(getattr(am.array(src), name)())
    want = getattr(pc, name)(src).to_pylist()
    assert len(got) == len(want)
    for g, w in zip(got, want):
        if g is None or w is None:
            assert g is w
        else:
            assert g == pytest.approx(w, rel=1e-12 if ty == pa.float64() else 1e-6, abs=1e-300)


def test_atan2_array_and_scalar():
    y = pa.array([1.0, -1.0, 0.0, 3.0, None], pa.float64())
    x = pa.array([1.0, 2.0, -1.0, 0.0, 1.0], pa.float64())
    got = pylist(am.array(y).atan2(x))
    want = pc.atan2(y, x).to_pylist()
    for g, w in zip(got, want):
        assert g is None if w is None else g == pytest.approx(w)
    scalar = pylist(am.array(y).atan2(2.0))
    for g, v in zip(scalar, y.to_pylist()):
        assert g is None if v is None else g == pytest.approx(math.atan2(v, 2.0))


@pytest.mark.parametrize("name,bad", [("sin_checked", math.inf), ("cos_checked", -math.inf),
                                      ("tan_checked", math.inf), ("asin_checked", 1.5),
                                      ("acos_checked", -1.5), ("acosh_checked", 0.5),
                                      ("atanh_checked", 1.0)])
def test_trig_checked(name, bad):
    unchecked = name[: -len("_checked")]
    lo, hi = TRIG_DOMAIN[unchecked]
    src = _trig_column(lo, hi, pa.float64())
    # In-domain (and NaN, and nulls) gives exactly the unchecked answer.
    assert pylist(getattr(am.array(src), name)()) == pylist(getattr(am.array(src), unchecked)())
    nan_ok = pa.array([float("nan"), None, (lo + hi) / 2], pa.float64())
    assert pylist(getattr(am.array(nan_ok), name)())[1] is None
    # One out-of-domain value raises, and pyarrow agrees that it should.
    vals = src.to_pylist()
    vals[4] = bad
    with pytest.raises(am.ArrowMetalError):
        getattr(am.array(pa.array(vals, pa.float64())), name)()
    with pytest.raises(pa.ArrowInvalid):
        getattr(pc, name)(pa.array(vals, pa.float64()))


def test_xor_and_not_and_kleene():
    a = pa.array([True, True, False, None, None, False], pa.bool_())
    b = pa.array([True, False, None, True, False, False], pa.bool_())
    assert pylist(am.array(a).xor(b)) == pc.xor(a, b).to_pylist()
    assert pylist(am.array(a) ^ am.array(b)) == pc.xor(a, b).to_pylist()
    assert pylist(am.array(a).and_not(b)) == pc.and_not(a, b).to_pylist()
    assert pylist(am.array(a).and_not_kleene(b)) == pc.and_not_kleene(a, b).to_pylist()


@pytest.mark.parametrize("src", [
    pa.array([1.5, None, float("nan"), float("inf"), float("-inf"), -0.0, 1e308], pa.float64()),
    pa.array([1.5, None, float("nan"), float("inf"), float("-inf"), -0.0, 1e38], pa.float32()),
    pa.array([1, None, 3, -4, 0, 7, 9], pa.int32()),
])
def test_float_classification(src):
    assert pylist(am.array(src).is_nan()) == pc.is_nan(src).to_pylist()
    assert pylist(am.array(src).is_finite()) == pc.is_finite(src).to_pylist()
    assert pylist(am.array(src).is_inf()) == pc.is_inf(src).to_pylist()


@pytest.mark.parametrize("src", [INT64, FLOAT64, FLOAT32, BOOL])
def test_fill_null_forward_and_backward(src):
    assert pylist(am.array(src).fill_null_forward()) == pc.fill_null_forward(src).to_pylist()
    assert pylist(am.array(src).fill_null_backward()) == pc.fill_null_backward(src).to_pylist()


def test_case_when_matches_pyarrow():
    c1 = pa.array([True, False, None, False, True, True, False, None, True], pa.bool_())
    c2 = pa.array([False, True, True, None, False, False, True, True, False], pa.bool_())
    v1 = pa.array([10, 20, 30, 40, 50, 60, 70, 80, 90], pa.int64())
    v2 = pa.array([1, 2, None, 4, 5, 6, 7, 8, 9], pa.int64())
    cond = pa.StructArray.from_arrays([c1, c2], ["a", "b"])
    assert pylist(am.case_when([c1, c2], [v1, v2], INT64)) == pc.case_when(cond, v1, v2, INT64).to_pylist()
    assert pylist(am.case_when([c1, c2], [v1, v2])) == pc.case_when(cond, v1, v2).to_pylist()


def test_choose_matches_pyarrow_and_rejects_out_of_range():
    idx = pa.array([0, 1, 2, None, 1, 0, 2, 1, 0], pa.int64())
    cols = [pa.array([i * 100 + j for j in range(9)], pa.int64()) for i in range(3)]
    assert pylist(am.choose(idx, cols)) == pc.choose(idx, *cols).to_pylist()
    with pytest.raises(am.ArrowMetalError):
        am.choose(pa.array([0, 3], pa.int64()), [c.slice(0, 2) for c in cols])


def test_replace_with_mask_matches_pyarrow():
    mask = pa.array([True, False, None, True, False, True, False, None, False], pa.bool_())
    repl = pa.array([-1, None, -3], pa.int64())
    assert pylist(am.array(INT64).replace_with_mask(mask, repl)) == \
        pc.replace_with_mask(INT64, mask, repl).to_pylist()
    with pytest.raises(am.ArrowMetalError):
        am.array(INT64).replace_with_mask(mask, repl.slice(0, 2))


@pytest.mark.parametrize("src", [INT64, FLOAT64, BOOL])
def test_indices_nonzero_matches_pyarrow(src):
    got = am.array(src).indices_nonzero()
    assert got.type == pa.uint64()
    assert pylist(got) == pc.indices_nonzero(src).to_pylist()


def test_hash64_is_deterministic_and_value_based():
    h = am.array(INT64).hash64()
    assert h.type == pa.uint64()
    values = pylist(h)
    assert values[1] is None and values[7] is None          # nulls stay null
    assert pylist(am.array(INT64).hash64()) == values       # deterministic
    # Equal values hash equal across signed zero and NaN payloads.
    zeros = pylist(am.array(pa.array([0.0, -0.0], pa.float64())).hash64())
    assert zeros[0] == zeros[1]
    nans = pa.array([float("nan")] * 2, pa.float64())
    assert len(set(pylist(am.array(nans).hash64()))) == 1
    # Distinct values collide rarely.
    big = pa.array(list(range(50_000)), pa.int64())
    assert len(set(pylist(am.array(big).hash64()))) == 50_000
