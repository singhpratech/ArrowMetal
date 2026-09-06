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


# ---------------------------------------------------------------- the remaining type-matrix rows
#
# null, float16, decimal32 / decimal64, the interval family, fixed_size_binary, list_view, extension
# types, list_parent_indices, list_slice, map_lookup, assume_timezone, local_timestamp and the three
# interval-returning *_between functions. Every result is compared to pyarrow.compute where pyarrow has
# the kernel, and to plain Python where it does not.

import random as _random  # noqa: E402

_TYPES_EXTRA_SIZES = [0, 1, 33, 4097]


def _nullable(values, every):
    return [None if i % every == 0 else v for i, v in enumerate(values)]


def test_null_type_round_trip_and_selection():
    a = pa.nulls(5)
    col = am.array(a)
    assert col.format == "n"
    assert col.type == pa.null()
    assert len(col) == 5 and col.null_count == 5
    assert col.to_arrow().equals(a)
    mask = am.array(pa.array([True, False, True, False, True]))
    assert len(col.filter(mask)) == 3
    assert len(col.slice(1, 2)) == 2
    assert pylist(am.nulls(3)) == [None, None, None]


def test_float16_round_trip_and_casts():
    values = [1.5, None, -3.25, 0.0, 65504.0, 1e-8]
    a = pa.array(values, pa.float64()).cast(pa.float16())
    col = am.array(a)
    assert col.format == "e"
    assert col.type == pa.float16()
    assert col.to_arrow().equals(a)
    # Widening is exact and matches pyarrow's own cast.
    assert pylist(col.to_float32()) == a.cast(pa.float32()).to_pylist()
    # Narrowing back is the identity on values that are already halves.
    assert col.to_float32().to_float16().to_arrow().equals(a)
    # Compute runs in float32: the result is only rounded back to half on an explicit cast.
    wide = col.to_float32()
    assert wide.max() == pc.max(a.cast(pa.float32())).as_py()
    assert pylist(wide > 0.0) == pylist(am.array(pc.greater(a.cast(pa.float32()), 0.0)))


@pytest.mark.parametrize("bits,arrow_type", [(32, pa.decimal32(7, 2)), (64, pa.decimal64(15, 3))])
def test_small_decimal_round_trip_and_widening(bits, arrow_type):
    a = pa.array([1, None, -3, 0], type=arrow_type)
    col = am.array(a)
    assert col.format == f"d:{arrow_type.precision},{arrow_type.scale},{bits}"
    assert col.type == arrow_type
    assert col.to_arrow().equals(a)
    wide = col.to_decimal128()
    assert wide.format == f"d:{arrow_type.precision},{arrow_type.scale}"
    assert pylist(wide) == a.to_pylist()
    assert wide.to_small_decimal(bits, arrow_type.precision).to_arrow().equals(a)


def test_fixed_size_binary_round_trip_compare_and_hash():
    a = pa.array([b"abcd", None, b"efgh", b"abcd"], type=pa.binary(4))
    col = am.array(a)
    assert col.format == "w:4"
    assert col.type == pa.binary(4)
    assert col.to_arrow().equals(a)
    assert pylist(col.fixed_binary_compare("==", b"abcd")) == \
        pc.equal(a, pa.scalar(b"abcd", pa.binary(4))).to_pylist()
    assert pylist(col.fixed_binary_compare("!=", b"abcd")) == \
        pc.not_equal(a, pa.scalar(b"abcd", pa.binary(4))).to_pylist()
    assert pylist(col.fixed_binary_compare("==", col)) == [True, None, True, True]
    hashes = pylist(col.hash64())
    assert hashes[1] is None and hashes[0] == hashes[3] and hashes[0] != hashes[2]
    with pytest.raises(am.ArrowMetalError):
        col.fixed_binary_compare("<", b"abcd")


def test_month_day_nano_interval_round_trip():
    start = pa.array([0, 86400 * 40, None], type=pa.timestamp("s"))
    end = pa.array([86400 * 400, 0, 5], type=pa.timestamp("s"))
    a = pc.month_day_nano_interval_between(start, end)
    col = am.array(a)
    assert col.format == "tin"
    assert col.type == pa.month_day_nano_interval()
    assert col.to_arrow().equals(a)


def test_interval_between_matches_pyarrow():
    _random.seed(11)
    n = 100_000
    lo, hi = -2_208_988_800, 7_258_118_400          # 1900-01-01 .. 2200-01-01
    raw_start = [_random.randint(lo, hi) for _ in range(n)]
    raw_end = [_random.randint(lo, hi) for _ in range(n)]
    start = pa.array(raw_start, type=pa.timestamp("s"))
    end = pa.array(raw_end, type=pa.timestamp("s"))
    a, b = am.array(start), am.array(end)
    expected = pc.month_day_nano_interval_between(start, end)
    assert a.month_day_nano_interval_between(b).to_arrow().equals(expected)
    # pyarrow 25 cannot wrap interval[month] / interval[day_time] arrays in Python, so those two are
    # compared field by field against the month_day_nano result and a plain Python oracle.
    months = pylist(a.month_interval_between(b).interval_field("months"))
    assert months == [v.months for v in expected.to_pylist()]
    dt = a.day_time_interval_between(b)
    assert pylist(dt.interval_field("days")) == \
        [(e // 86400) - (s // 86400) for s, e in zip(raw_start, raw_end)]
    assert pylist(dt.interval_field("nanoseconds")) == \
        [((e % 86400) - (s % 86400)) * 10 ** 9 for s, e in zip(raw_start, raw_end)]


def test_add_interval_clamps_the_day_like_arrow():
    # 2024-01-31 + 1 month is 2024-02-29, and + 1 month again is 2024-03-29.
    jan31 = 1_706_659_200
    ts = am.array(pa.array([jan31, jan31], type=pa.timestamp("s")))
    iv = pa.array([(1, 0, 0), (2, 3, 0)], type=pa.month_day_nano_interval())
    got = pylist(ts.add_interval(iv))
    # 2024-01-31 + 2 months clamps to 2024-03-31, and the 3 days then land on 2024-04-03.
    assert [d.strftime("%Y-%m-%d") for d in got] == ["2024-02-29", "2024-04-03"]
    # pyarrow has no add(timestamp, interval) kernel, so there is no pyarrow oracle for this one.
    assert not hasattr(pc, "add_interval")


@pytest.mark.parametrize("n", _TYPES_EXTRA_SIZES)
def test_list_parent_indices_and_slice_match_pyarrow(n):
    rows = [None if i % 5 == 2 else list(range(i % 4)) for i in range(n)]
    a = pa.array(rows, type=pa.list_(pa.int64()))
    col = am.array(a)
    # ArrowMetal returns int32 where pyarrow returns int64; the values are the same.
    assert pylist(col.list_parent_indices()) == pc.list_parent_indices(a).to_pylist()
    assert pylist(col.list_slice(1, 3)) == pc.list_slice(a, 1, 3).to_pylist()
    assert pylist(col.list_slice(0, None, 2)) == pc.list_slice(a, 0, step=2).to_pylist()


@pytest.mark.parametrize("occurrence", ["first", "last", "all"])
def test_map_lookup_matches_pyarrow(occurrence):
    rows = [[("a", 1), ("b", 2), ("a", 3)], None, [], [("c", 9)], [("a", 4)]]
    a = pa.array(rows, type=pa.map_(pa.string(), pa.int64()))
    col = am.array(a)
    assert pylist(col.map_lookup("a", occurrence)) == \
        pc.map_lookup(a, pa.scalar("a"), occurrence).to_pylist()
    assert pylist(col.map_lookup("zz", occurrence)) == \
        pc.map_lookup(a, pa.scalar("zz"), occurrence).to_pylist()


def test_map_lookup_integer_keys_matches_pyarrow():
    rows = [[(1, 10), (2, 20), (1, 30)], None, [(7, 70)]]
    a = pa.array(rows, type=pa.map_(pa.int32(), pa.int64()))
    col = am.array(a)
    for occurrence in ("first", "last", "all"):
        assert pylist(col.map_lookup(1, occurrence)) == \
            pc.map_lookup(a, pa.scalar(1, pa.int32()), occurrence).to_pylist()


def test_list_view_imports_as_a_list():
    offsets = pa.array([3, 0, 1], pa.int32())
    sizes = pa.array([1, 2, 2], pa.int32())
    view = pa.ListViewArray.from_arrays(offsets, sizes, pa.array([1, 2, 3, 4]))
    col = am.array(view)
    # A list view has no ArrowMetal column of its own, so it comes back as a plain list.
    assert col.format == "+l"
    assert pylist(col) == view.to_pylist()


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_assume_timezone_and_local_timestamp_match_pyarrow(unit):
    base = 1_672_574_400                             # 2023-01-01T12:00:00Z, safely away from transitions
    step = {"s": 1, "ms": 10 ** 3, "us": 10 ** 6, "ns": 10 ** 9}[unit]
    naive = pa.array(_nullable([(base + i * 86_400) * step for i in range(200)], 17),
                     type=pa.timestamp(unit))
    col = am.array(naive)
    got = col.assume_timezone("America/New_York").to_arrow()
    assert got.equals(pc.assume_timezone(naive, "America/New_York"))
    back = am.array(got).local_timestamp().to_arrow()
    assert back.equals(pc.local_timestamp(got))
    assert back.equals(naive)


def _naive_seconds(y, m, d, hh, mm):
    """The wall-clock instant as a naive timestamp value (seconds, read as if UTC)."""
    import datetime
    return int(datetime.datetime(y, m, d, hh, mm, tzinfo=datetime.timezone.utc).timestamp())


def test_assume_timezone_ambiguous_and_nonexistent():
    tz = "America/New_York"
    # 2023-11-05 01:30 local happens twice in New York (EDT then EST).
    ambiguous = pa.array([_naive_seconds(2023, 11, 5, 1, 30)], type=pa.timestamp("s"))
    col = am.array(ambiguous)
    with pytest.raises(am.ArrowMetalError):
        col.assume_timezone(tz)
    early = pylist(col.assume_timezone(tz, ambiguous="earliest"))[0]
    late = pylist(col.assume_timezone(tz, ambiguous="latest"))[0]
    assert late.timestamp() - early.timestamp() == 3600
    assert early == pc.assume_timezone(ambiguous, tz, ambiguous="earliest").to_pylist()[0]
    assert late == pc.assume_timezone(ambiguous, tz, ambiguous="latest").to_pylist()[0]

    # 2023-03-12 02:30 local never happens in New York (the spring-forward gap).
    gap = pa.array([_naive_seconds(2023, 3, 12, 2, 30)], type=pa.timestamp("s"))
    g = am.array(gap)
    with pytest.raises(am.ArrowMetalError):
        g.assume_timezone(tz)
    before = pylist(g.assume_timezone(tz, nonexistent="earliest"))[0]
    after = pylist(g.assume_timezone(tz, nonexistent="latest"))[0]
    assert after.timestamp() - before.timestamp() == 1


def test_extension_type_round_trip_and_metadata():
    storage = pa.array([bytes(range(16)), None], type=pa.binary(16))
    a = pa.ExtensionArray.from_storage(pa.uuid(), storage)
    col = am.array(a)
    assert col.extension_name == "arrow.uuid"
    assert col.extension_metadata == b""
    assert col.format == "w:16", "the format is the storage type's"
    assert col.type == pa.uuid()
    assert col.to_arrow().equals(a)
    # Selection keeps the extension tag.
    kept = col.filter(am.array(pa.array([True, False])))
    assert kept.extension_name == "arrow.uuid"
    assert kept.to_arrow().type == pa.uuid()
    assert col.extension_storage().format == "w:16"
    assert col.extension_storage().extension_name is None
    # A plain column can be tagged as extension storage.
    wrapped = am.array(storage).as_extension_type("arrow.uuid")
    assert wrapped.to_arrow().type == pa.uuid()
