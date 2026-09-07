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


@pytest.mark.parametrize("offset", [0, 1, 7, 31, 32, 33, 4097])
def test_slice_at_any_offset_matches_pyarrow(offset):
    """Every offset is a zero-copy view; the awkward ones (not a multiple of 32) ride on Arrow's
    `offset` field, so the values, the null count and a re-export must all still agree."""
    src = pa.array([None if i % 7 == 3 else i for i in range(8000)], pa.int64())
    length = 8000 - offset - 11
    got = am.array(src).slice(offset, length)
    want = src.slice(offset, length)
    assert pylist(got) == want.to_pylist()
    assert got.null_count == want.null_count
    assert got.to_arrow().equals(want)


@pytest.mark.parametrize("offset", [1, 7, 33])
def test_slice_then_compute_matches_pyarrow(offset):
    src = pa.array([None if i % 7 == 3 else (i * 37) % 5000 for i in range(8000)], pa.int64())
    length = 8000 - offset - 11
    got, want = am.array(src).slice(offset, length), src.slice(offset, length)
    assert got.sum() == pc.sum(want).as_py()
    assert got.min() == pc.min(want).as_py()
    assert got.first() == pc.first(want).as_py()
    assert got.last() == pc.last(want).as_py()
    assert pylist(got.sort()) == pc.take(want, pc.array_sort_indices(want)).to_pylist()
    assert pylist(got.filter(got > 2500)) == pc.filter(want, pc.greater(want, 2500)).to_pylist()


def test_slice_of_a_string_column_is_a_view():
    src = pa.array([None if i % 9 == 2 else f"row-{i}" for i in range(3000)])
    for offset in (1, 7, 33, 1024):
        got = am.array(src).slice(offset, 500)
        assert pylist(got) == src.slice(offset, 500).to_pylist()


# ---------------------------------------------------------------- join


def test_join_returns_the_matching_index_pairs():
    left = pa.array([1, 2, 3, 2, 5], pa.int64())
    right = pa.array([2, 3, 3, 7], pa.int64())
    li, ri = am.join(am.array(left), am.array(right))
    got = sorted(zip(pylist(li), pylist(ri)))
    want = sorted((l, r) for l in range(len(left)) for r in range(len(right))
                  if left[l].as_py() == right[r].as_py())
    assert got == want


def test_left_join_keeps_unmatched_rows_with_a_null_right_index():
    left = pa.array([1, 2, 9], pa.int64())
    right = pa.array([2, 9], pa.int64())
    li, ri = am.join(am.array(left), am.array(right), how="left")
    assert sorted(zip(pylist(li), pylist(ri))) == [(0, None), (1, 0), (2, 1)]


def test_join_null_keys_never_match():
    left = pa.array([None, 4], pa.int64())
    right = pa.array([None, 4], pa.int64())
    li, ri = am.join(am.array(left), am.array(right))
    assert list(zip(pylist(li), pylist(ri))) == [(1, 1)]


def test_join_rejects_an_unknown_how():
    with pytest.raises(am.ArrowMetalError):
        am.join(am.array(pa.array([1], pa.int64())), am.array(pa.array([1], pa.int64())), how="outer")


# ---------------------------------------------------------------- dictionary_encode


@pytest.mark.parametrize("values,arrow_type", [
    ([3, 1, 3, None, 2], pa.int32()),
    ([3, 1, 3, None, 2], pa.int64()),
    ([1.5, 0.5, 1.5, None], pa.float64()),
    ([True, False, True, None], pa.bool_()),
    (["b", "a", "b", None], pa.string()),
])
def test_dictionary_encode_round_trips_for_every_type(values, arrow_type):
    src = pa.array(values, arrow_type)
    codes, uniques = am.array(src).dictionary_encode()
    assert pc.take(uniques.to_arrow(), codes.to_arrow()).to_pylist() == values


def test_dictionary_encode_on_a_wide_integer_column():
    src = pa.array([(i * 7) % 1000 for i in range(50_000)], pa.int32())
    codes, uniques = am.array(src).dictionary_encode()
    assert len(uniques) == 1000
    assert pc.take(uniques.to_arrow(), codes.to_arrow()).equals(src)


# ---------------------------------------------------------------- utf8 sort


def test_utf8_argsort_matches_pyarrow_byte_order():
    src = pa.array(["banana", "Apple", "apple", None, "Zebra", "", "ab", "ab\x00", "abc"])
    assert pylist(am.array(src).argsort()) == pc.array_sort_indices(src).to_pylist()
    assert pylist(am.array(src).sort()) == pc.take(src, pc.array_sort_indices(src)).to_pylist()


def test_utf8_argsort_descending_keeps_nulls_last():
    src = pa.array(["b", None, "a", "c"])
    assert pylist(am.array(src).argsort(descending=True)) == [3, 0, 2, 1]


def test_lexsort_accepts_a_string_key():
    region = am.array(pa.array(["west", "east", "west", "east"]))
    revenue = am.array(pa.array([10, 30, 20, 5], pa.int64()))
    assert pylist(am.lexsort_indices([region, revenue])) == [3, 1, 0, 2]


# ---------------------------------------------------------------- decimal reductions


def test_decimal_sum_min_max():
    import decimal
    values = [decimal.Decimal(f"{i}.{i % 10}{i % 7}") for i in range(500)]
    src = pa.array(values, pa.decimal128(18, 4))
    a = am.array(src)
    assert a.sum() == sum(values)
    assert a.min() == min(values)
    assert a.max() == max(values)


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

    # A single unreproduced segfault was once seen a few allocations after pyarrow 25.0.1's own
    # `year_month_day`, so out of caution the oracle here is its year / month / day kernels, which
    # compute exactly the same three fields.
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


# ---- group-by over arbitrary key columns, the rest of the grouped aggregates, and the scalar
# skew / kurtosis / tdigest. Every assertion goes through pyarrow's own group_by as the oracle.


def _by_key(gb, result, n_keys=1):
    """{(key tuple): value} from an ArrowMetal grouped result, so group order never enters a test."""
    key_cols = [c.to_pylist() for c in gb.keys()]
    values = result.to_arrow().to_pylist()
    return {tuple(key_cols[j][g] for j in range(n_keys)): values[g] for g in range(len(values))}


def _pyarrow_by_key(table, cols, agg, col="v", ordered=False):
    grouped = table.group_by(list(cols), use_threads=not ordered)
    out = grouped.aggregate([(col, agg)])
    return {tuple(row[c] for c in cols): row[f"{col}_{agg}"] for row in out.to_pylist()}


def _group_fixture():
    keys = ["a", "b", None, "a", "c", "b", "a", None, "c", "b"]
    years = [2000, 2001, 2000, 2001, None, 2001, 2000, 2000, 2001, 2001]
    vals = [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    table = pa.table({"k": pa.array(keys, pa.utf8()), "y": pa.array(years, pa.int32()),
                      "v": pa.array(vals, pa.float64())})
    return table, am.array(table["k"].combine_chunks()), am.array(table["y"].combine_chunks()), \
        am.array(table["v"].combine_chunks())


def test_group_by_arbitrary_keys_matches_pyarrow():
    table, g_keys, g_years, g_vals = _group_fixture()
    gb = am.group_by([g_keys])
    assert gb.group_count == table.group_by(["k"]).aggregate([]).num_rows
    assert len(gb.keys()) == 1
    for agg in ("sum", "mean", "min", "max", "count"):
        assert _by_key(gb, getattr(gb, agg)(g_vals)) == _pyarrow_by_key(table, ("k",), agg)
    all_rows = table.group_by(["k"], use_threads=False).aggregate([("v", "count", pc.CountOptions(mode="all"))])
    assert _by_key(gb, gb.count_all()) == {(r["k"],): r["v_count"] for r in all_rows.to_pylist()}
    # A single array (not a list) is accepted too.
    assert am.group_by(g_keys).group_count == gb.group_count


def test_group_by_two_key_columns_matches_pyarrow():
    table, g_keys, g_years, g_vals = _group_fixture()
    gb = am.group_by([g_keys, g_years])
    assert gb.group_count == table.group_by(["k", "y"]).aggregate([]).num_rows
    assert _by_key(gb, gb.sum(g_vals), 2) == _pyarrow_by_key(table, ("k", "y"), "sum")
    assert _by_key(gb, gb.max(g_vals), 2) == _pyarrow_by_key(table, ("k", "y"), "max")


def test_group_by_null_keys_form_their_own_group():
    table, g_keys, _, g_vals = _group_fixture()
    gb = am.group_by([g_keys])
    sums = _by_key(gb, gb.sum(g_vals))
    assert (None,) in sums
    assert sums[(None,)] == 8.0


def test_grouped_extra_aggregates():
    table, g_keys, _, g_vals = _group_fixture()
    gb = am.group_by([g_keys])
    # min_max and first_last come back as structs.
    mm = gb.min_max(g_vals).to_arrow()
    assert mm.type.num_fields == 2 and [f.name for f in mm.type] == ["min", "max"]
    fl = gb.first_last(g_vals).to_arrow()
    assert [f.name for f in fl.type] == ["first", "last"]
    assert _by_key(gb, gb.first(g_vals)) == _pyarrow_by_key(table, ("k",), "first", ordered=True)
    assert _by_key(gb, gb.last(g_vals)) == _pyarrow_by_key(table, ("k",), "last", ordered=True)
    assert _by_key(gb, gb.one(g_vals)) == _pyarrow_by_key(table, ("k",), "one", ordered=True)
    assert _by_key(gb, gb.list(g_vals)) == _pyarrow_by_key(table, ("k",), "list", ordered=True)
    assert _by_key(gb, gb.count_distinct(g_vals)) == _pyarrow_by_key(table, ("k",), "count_distinct")
    ours = _by_key(gb, gb.distinct(g_vals))
    theirs = _pyarrow_by_key(table, ("k",), "distinct", ordered=True)
    assert {k: sorted(x for x in v if x is not None) for k, v in ours.items()} == \
           {k: sorted(x for x in v if x is not None) for k, v in theirs.items()}
    # product runs on the GPU now; ArrowMetal returns float64 for a float column.
    assert _by_key(gb, gb.product(g_vals)) == _pyarrow_by_key(table, ("k",), "product")
    # The median is exact here, not a sketch.
    assert _by_key(gb, gb.approximate_median(g_vals))[("a",)] == 4.0
    assert _by_key(gb, gb.quantile(g_vals, 0.0))[("a",)] == 1.0
    assert _by_key(gb, gb.quantile(g_vals, 1.0))[("a",)] == 7.0
    for name in ("variance", "stddev", "skew", "kurtosis"):
        assert len(getattr(gb, name)(g_vals)) == gb.group_count
    assert len(gb.tdigest(g_vals, 0.5)) == gb.group_count


def test_grouped_pivot_wider():
    keys = am.array(pa.array(["g1", "g1", "g2", "g2"], pa.utf8()))
    pivot = pa.array(["a", "b", "a", "b"], pa.utf8())
    values = am.array(pa.array([1, 2, 3, 4], pa.int64()))
    gb = am.group_by([keys])
    out = gb.pivot_wider(pivot, values, ["a", "b"]).to_arrow()
    assert [f.name for f in out.type] == ["a", "b"]
    labels = gb.keys()[0].to_pylist()
    rows = {labels[i]: out[i].as_py() for i in range(len(out))}
    assert rows == {"g1": {"a": 1, "b": 2}, "g2": {"a": 3, "b": 4}}


def test_group_by_ids_and_dense_fast_path_agree():
    keys = am.array(pa.array([10, 20, 10, 30, 20], pa.int32()))
    values = am.array(pa.array([1, 2, 3, 4, 5], pa.int64()))
    gb = am.group_by([keys])
    ids = gb.ids().to_arrow().to_pylist()
    assert len(ids) == 5 and max(ids) == gb.group_count - 1
    dense = am.array(pa.array(ids, pa.int32())).group_by(gb.group_count)
    assert dense.sum(values).to_arrow().to_pylist() == gb.sum(values).to_arrow().to_pylist()


def test_scalar_skew_kurtosis_and_tdigest():
    values = pa.array([float(i % 17) + 0.5 * (i % 3) for i in range(5000)], pa.float64())
    a = am.array(values)
    assert a.skew() == pytest.approx(pc.skew(values).as_py(), rel=1e-5)
    assert a.kurtosis() == pytest.approx(pc.kurtosis(values).as_py(), rel=1e-5)
    assert a.skew(biased=False) == pytest.approx(pc.skew(values, biased=False).as_py(), rel=1e-5)
    assert a.kurtosis(biased=False) == pytest.approx(pc.kurtosis(values, biased=False).as_py(), rel=1e-5)
    # tdigest is a sketch; the extremes are exact and the middle is close to the exact quantile.
    assert a.tdigest(0.0) == pytest.approx(pc.min(values).as_py())
    assert a.tdigest(1.0) == pytest.approx(pc.max(values).as_py())
    assert a.tdigest(0.5) == pytest.approx(pc.quantile(values, q=0.5).to_pylist()[0], abs=0.5)
    assert am.array(pa.array([], pa.float64())).skew() is None


def test_shift_view_is_the_same_answer_without_the_copy():
    """`shift(view=True)` returns a two-chunk pyarrow.ChunkedArray sharing the input's buffers.

    The values, the nulls and the order have to match the contiguous form exactly at every shift the
    contiguous form accepts -- lag, lead, zero, a fill value, and shifts at or past the length -- so the
    only difference between the two is where the bytes live."""
    import numpy as np
    for n in (0, 1, 2, 7, 31, 32, 33, 255, 256, 257, 1000):
        values = [None if i % 5 == 0 else i - 500 for i in range(n)]
        for src in (pa.array(values, pa.int64()),
                    pa.array([0 if v is None else v for v in values], pa.int64())):
            a = am.array(src)
            for by in (0, 1, 2, 5, -1, -3, n, n + 1, -n, -(n + 1)):
                for fill in (None, 7):
                    want = a.shift(by, fill).to_arrow()
                    view = a.shift(by, fill, view=True)
                    assert isinstance(view, pa.ChunkedArray)
                    assert view.type == src.type
                    assert len(view) == n
                    assert view.combine_chunks().equals(want), f"n={n} by={by} fill={fill}"
    del np


def test_shift_view_shares_the_input_buffers_and_outlives_it():
    """Zero-copy means zero-copy: the chunked view keeps the device memory alive on its own, and the
    slice chunk starts at the same address the input's values do (offset by the lead, for a lead)."""
    import gc
    src = pa.array(list(range(1000)), pa.int64())
    a = am.array(src)
    base = a.to_arrow().buffers()[1].address
    lag = a.shift(1, view=True)
    lead = a.shift(-1, view=True)
    assert lag.chunk(1).buffers()[1].address == base
    assert lead.chunk(0).buffers()[1].address == base
    assert lead.chunk(0).offset == 1
    del a, src
    gc.collect()
    assert lag.combine_chunks().to_pylist()[:3] == [None, 0, 1]
    assert lead.combine_chunks().to_pylist()[-3:] == [998, 999, None]


def test_compute_format_of_a_dictionary_is_its_values_format():
    """am_format reports a dictionary's index type; the scalar kernels compute on its values, and
    am_compute_format says so. A float64 scalar against a dictionary<int32, float64> column must be
    packed to eight bytes, not four (a Rust binding found the four-byte packing accepted and wrong)."""
    d = pa.array([1.5, 2.5, 3.5]).dictionary_encode()
    col = am.array(d)
    assert col.format == "i"
    assert am._lib.am_compute_format(col._h).decode() == "g"
    assert col.compare(">", 2.0).to_arrow().to_pylist() == [False, True, True]
    assert col.filter_where(">", 2.0).to_arrow().to_pylist() == [2.5, 3.5]
    plain = am.array(pa.array([1, 2, 3], pa.int64()))
    assert am._lib.am_compute_format(plain._h).decode() == plain.format == "l"
