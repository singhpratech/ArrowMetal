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


# ---------------------------------------------------------------- statistical aggregates

def test_statistical_aggregates_match_pyarrow():
    a = am.array(FLOAT64)
    valid = [v for v in FLOAT64.to_pylist() if v is not None]
    assert a.product() == pytest.approx(math.prod(valid), rel=1e-12)
    assert a.variance() == pytest.approx(pc.variance(FLOAT64, ddof=0).as_py(), rel=1e-12)
    assert a.variance(1) == pytest.approx(pc.variance(FLOAT64, ddof=1).as_py(), rel=1e-12)
    assert a.stddev() == pytest.approx(pc.stddev(FLOAT64, ddof=0).as_py(), rel=1e-12)
    assert a.stddev(1) == pytest.approx(pc.stddev(FLOAT64, ddof=1).as_py(), rel=1e-12)
    assert a.median() == pytest.approx(pc.approximate_median(FLOAT64).as_py(), rel=1e-12)
    for q in (0.0, 0.25, 0.5, 0.9, 1.0):
        assert a.quantile(q) == pytest.approx(pc.quantile(FLOAT64, q=q)[0].as_py(), rel=1e-12)
    assert a.count_distinct() == pc.count_distinct(FLOAT64).as_py()
    assert a.min_max() == (pc.min(FLOAT64).as_py(), pc.max(FLOAT64).as_py())
    assert a.first() == valid[0]
    assert a.last() == valid[-1]
    assert a.index(FLOAT64[2].as_py()) == 2
    assert a.index(-12345.0) == -1


def test_mode_and_boolean_any_all():
    values = pa.array([4, 4, 7, None, 7, 7], pa.int32())
    assert am.array(values).mode() == (7, 3)
    assert am.array(values).count_distinct() == 2
    assert am.array(BOOL).any() is True
    assert am.array(BOOL).all() is False
    assert am.array(pa.array([True, None, True])).all() is True
    assert am.array(pa.array([False, None, False])).any() is False


def test_run_end_encoding_round_trips_through_pyarrow():
    source = pa.array([1, 1, 1, None, None, 4, 4, 9], pa.int32())
    encoded = am.array(source).run_end_encode()
    assert encoded.format == "+r"
    assert len(encoded) == len(source)
    exported = encoded.to_arrow()
    assert pa.types.is_run_end_encoded(exported.type)
    assert exported.to_pylist() == source.to_pylist()
    assert pylist(encoded.run_end_decode()) == source.to_pylist()

    # A run-end array pyarrow produced imports and decodes just as well.
    imported = am.array(pc.run_end_encode(pa.array([2, 2, None, 3], pa.int64())))
    assert imported.format == "+r"
    assert pylist(imported.run_end_decode()) == [2, 2, None, 3]


# ---------------------------------------------------------------- temporal: week numbers, the
# struct extractors, subsecond, is_dst and every *_between. pyarrow.compute is the oracle throughout.

import itertools as _itertools
import random as _random

_LOW_SECOND = -2_208_988_800     # 1900-01-01T00:00:00Z
_HIGH_SECOND = 7_258_118_400     # 2200-01-01T00:00:00Z


def _random_seconds(n, seed):
    """`n` UTC seconds spanning 1900-2200, with the awkward instants pinned in front."""
    rng = _random.Random(seed)
    out = [rng.randint(_LOW_SECOND, _HIGH_SECOND - 1) for _ in range(n)]
    pinned = [0, -1, 1, -86_400, 86_400, -86_401, _LOW_SECOND, _HIGH_SECOND - 1,
              951_782_400, 1_388_534_400, 1_420_070_400, 1_451_606_400, 1_451_779_200]
    for i, v in enumerate(pinned[:n]):
        out[i] = v
    return out


def _dates(seconds, nulls=True):
    days = [s // 86_400 for s in seconds]
    if nulls:
        days = [None if i % 7 == 3 else d for i, d in enumerate(days)]
    return pa.array(days, pa.date32())


def test_week_matches_pyarrow_for_every_option_combination():
    values = _dates(_random_seconds(20_000, 101))
    col = am.array(values)
    for wsm, cfz, fwfy in _itertools.product([True, False], repeat=3):
        got = pylist(col.week(week_starts_monday=wsm, count_from_zero=cfz,
                              first_week_is_fully_in_year=fwfy))
        want = pc.week(values, week_starts_monday=wsm, count_from_zero=cfz,
                       first_week_is_fully_in_year=fwfy).to_pylist()
        assert got == want, (wsm, cfz, fwfy)
    assert pylist(col.us_week()) == pc.us_week(values).to_pylist()
    assert pylist(col.us_year()) == pc.us_year(values).to_pylist()
    assert col.week().type == pa.int64()


def test_iso_calendar_and_year_month_day_match_pyarrow():
    values = _dates(_random_seconds(20_000, 202))
    col = am.array(values)
    got = col.iso_calendar()
    assert got.format == "+s"
    expected = pc.iso_calendar(values)
    assert got.to_arrow().type == expected.type
    assert got.to_arrow().to_pylist() == expected.to_pylist()

    # pyarrow 25.0.1's own `year_month_day` kernel segfaults on arrays of this size, so the oracle
    # here is its year / month / day kernels, which compute exactly the same three fields.
    ymd = col.year_month_day()
    assert ymd.format == "+s"
    assert ymd.to_arrow().type == pa.struct([("year", pa.int64()), ("month", pa.int64()),
                                             ("day", pa.int64())])
    year, month, day = pc.year(values), pc.month(values), pc.day(values)
    assert pylist(ymd.struct_field("year")) == year.cast(pa.int64()).to_pylist()
    assert pylist(ymd.struct_field("month")) == month.cast(pa.int64()).to_pylist()
    assert pylist(ymd.struct_field("day")) == day.cast(pa.int64()).to_pylist()

    # struct_field reaches one child by name.
    fields = am.array(_dates([0, 86_400], nulls=False)).iso_calendar()
    assert pylist(fields.struct_field("iso_week")) == [1, 1]
    assert pylist(fields.struct_field("iso_day_of_week")) == [4, 5]


def test_day_of_week_options_match_pyarrow():
    values = _dates(_random_seconds(5_000, 303))
    col = am.array(values)
    for cfz, week_start in _itertools.product([True, False], range(1, 8)):
        got = pylist(col.day_of_week(count_from_zero=cfz, week_start=week_start))
        assert got == pc.day_of_week(values, count_from_zero=cfz, week_start=week_start).to_pylist()


def test_subsecond_matches_pyarrow_bit_for_bit():
    rng = _random.Random(404)
    ticks = [s * 1_000_000_000 + rng.randrange(1_000_000_000)
             for s in _random_seconds(20_000, 405)]
    for unit, scale in [("ns", 1), ("us", 1_000), ("ms", 1_000_000), ("s", 1_000_000_000)]:
        values = pa.array([t // scale for t in ticks], pa.timestamp(unit))
        assert pylist(am.array(values).subsecond()) == pc.subsecond(values).to_pylist()
    for values in [pa.array([1_500_000_123, None, 86_399_999_999_999], pa.time64("ns")),
                   pa.array([1_500, None, 86_399_999], pa.time32("ms"))]:
        assert pylist(am.array(values).subsecond()) == pc.subsecond(values).to_pylist()


def test_is_dst_matches_pyarrow():
    # 1900 through 2037. Past the 2038 cliff Foundation projects each zone's current DST rule
    # forward while pyarrow's bundled timezone database stops, so the two disagree there — a
    # difference in the timezone data, not in the kernel.
    rng = _random.Random(506)
    seconds = [rng.randint(_LOW_SECOND, 2_140_000_000) for _ in range(20_000)]
    for zone in ["America/New_York", "Europe/Berlin", "Australia/Sydney", "UTC"]:
        values = pa.array([None if i % 7 == 3 else s for i, s in enumerate(seconds)],
                          pa.timestamp("s", tz=zone))
        assert pylist(am.array(values).is_dst()) == pc.is_dst(values).to_pylist(), zone
    # Every unit, and a fixed offset, which never observes DST.
    for unit, scale in [("s", 1), ("ms", 10 ** 3), ("us", 10 ** 6), ("ns", 10 ** 9)]:
        values = pa.array([1_672_531_200 * scale, 1_688_000_000 * scale, None],
                          pa.timestamp(unit, tz="America/New_York"))
        assert pylist(am.array(values).is_dst()) == [False, True, None]
    fixed = pa.array([1_688_000_000], pa.timestamp("s", tz="+02:00"))
    assert pylist(am.array(fixed).is_dst()) == pc.is_dst(fixed).to_pylist() == [False]
    with pytest.raises(am.ArrowMetalError):
        am.array(pa.array([0], pa.timestamp("s"))).is_dst()


def test_every_between_matches_pyarrow_over_100k_pairs():
    n = 100_000
    a_seconds = _random_seconds(n, 607)
    b_seconds = _random_seconds(n, 708)
    a_seconds[0], b_seconds[0] = 0, -1          # a negative difference
    a_seconds[1], b_seconds[1] = -1, 0
    a = pa.array([None if i % 11 == 5 else s for i, s in enumerate(a_seconds)], pa.timestamp("s"))
    b = pa.array([None if i % 13 == 7 else s for i, s in enumerate(b_seconds)], pa.timestamp("s"))
    ma, mb = am.array(a), am.array(b)
    for name in ["years_between", "quarters_between", "weeks_between", "days_between",
                 "hours_between", "minutes_between", "seconds_between", "milliseconds_between",
                 "microseconds_between", "nanoseconds_between"]:
        assert pylist(getattr(ma, name)(mb)) == getattr(pc, name)(a, b).to_pylist(), name
    # weeks_between with every DayOfWeekOptions combination, on a smaller slice.
    sa, sb = a.slice(0, 5_000), b.slice(0, 5_000)
    msa, msb = am.array(sa), am.array(sb)
    for cfz, week_start in _itertools.product([True, False], range(1, 8)):
        got = pylist(msa.weeks_between(msb, count_from_zero=cfz, week_start=week_start))
        assert got == pc.weeks_between(sa, sb, count_from_zero=cfz,
                                       week_start=week_start).to_pylist()
    # months_between is Arrow's month_interval_between as a plain int64 count. pyarrow cannot hand a
    # month interval back to Python, so the oracle is year * 12 + month from pyarrow's own kernels.
    ay, am_, by, bm = (pc.year(a).to_pylist(), pc.month(a).to_pylist(),
                       pc.year(b).to_pylist(), pc.month(b).to_pylist())
    want = [None if ay[i] is None or by[i] is None
            else (by[i] * 12 + bm[i]) - (ay[i] * 12 + am_[i]) for i in range(n)]
    assert pylist(ma.months_between(mb)) == want


def test_between_accepts_mixed_units_and_types():
    # 2021-03-04T23:59:00Z -> 2021-03-05T00:01:00Z: one day, one hour and two minutes of boundaries.
    left = am.array(pa.array([1_614_902_340], pa.timestamp("s")))
    right = am.array(pa.array([1_614_902_460_000], pa.timestamp("ms", tz="UTC")))
    assert pylist(left.days_between(right)) == [1]
    assert pylist(left.hours_between(right)) == [1]
    assert pylist(left.minutes_between(right)) == [2]
    assert pylist(right.seconds_between(left)) == [-120]
    d32 = am.array(pa.array([0], pa.date32()))
    ns = am.array(pa.array([86_400_000_000_001], pa.timestamp("ns")))
    assert pylist(d32.hours_between(ns)) == [24]
    assert pylist(d32.nanoseconds_between(ns)) == [86_400_000_000_001]
    with pytest.raises(am.ArrowMetalError):
        am.array(pa.array([0], pa.duration("s"))).seconds_between(am.array(pa.array([1], pa.duration("s"))))


def test_temporal_extra_shapes_and_nulls():
    for n in [0, 1, 33, 4097]:
        values = _dates(_random_seconds(n, 800 + n))
        nulls = values.null_count
        col = am.array(values)
        for result in [col.week(), col.us_week(), col.us_year(), col.subsecond(),
                       col.day_of_week(count_from_zero=False, week_start=3)]:
            assert len(result) == n
            assert result.null_count == nulls
        got = col.iso_calendar()
        assert len(got) == n and got.null_count == nulls
        assert got.to_arrow().to_pylist() == pc.iso_calendar(values).to_pylist()
        ymd = col.year_month_day()
        assert len(ymd) == n and ymd.null_count == nulls
        assert pylist(ymd.struct_field("day")) == pc.day(values).cast(pa.int64()).to_pylist()
        assert pylist(col.years_between(col)) == [None if v is None else 0
                                                  for v in values.to_pylist()]
