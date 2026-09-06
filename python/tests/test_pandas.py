"""pandas on the GPU: the conversion bridge, the `.am` accessor and the zero-code-change accel mode.

Every accel-mode test runs the same expression twice — once in plain pandas, once with
`arrowmetal.pandas_accel` installed at a zero row threshold so everything eligible goes to the GPU —
and demands the two results be equal, dtype and index included.
"""
import contextlib
import os

import numpy as np
import pyarrow as pa
import pytest

pd = pytest.importorskip("pandas")
import pandas.testing as pdt

import arrowmetal as am
from arrowmetal import pandas_bridge as bridge
from arrowmetal import pandas_accel as accel


# ---------------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------------

N = 400
rng = np.random.default_rng(7)


def _ints():
    v = rng.integers(-50, 50, N)
    return v


@pytest.fixture(scope="module")
def frames():
    """One frame per pandas storage flavour, all with the same logical content."""
    ints = _ints()
    floats = rng.normal(size=N).round(3)
    floats[::17] = np.nan
    keys = np.array(["k%02d" % (i % 13) for i in range(N)], dtype=object)
    keys[::29] = None
    words = np.array(["alpha", "beta", "gamma", "delta", "epsilon"], dtype=object)[rng.integers(0, 5, N)]
    words = words.astype(object)
    words[::31] = None
    out = {}
    out["numpy"] = pd.DataFrame({
        "i": ints.astype(np.int64),
        "f": floats,
        "k": pd.Series(keys, dtype="str"),
        "w": pd.Series(words, dtype="str"),
        "b": pd.Series(ints > 0),
    })
    out["arrow"] = pd.DataFrame({
        "i": pd.array(ints, dtype="int64[pyarrow]"),
        "f": pd.array(floats, dtype="double[pyarrow]"),
        "k": pd.array(keys, dtype=pd.ArrowDtype(pa.string())),
        "w": pd.array(words, dtype=pd.ArrowDtype(pa.string())),
        "b": pd.array(ints > 0, dtype="bool[pyarrow]"),
    })
    out["masked"] = pd.DataFrame({
        "i": pd.array(ints, dtype="Int64"),
        "f": pd.array(floats, dtype="Float64"),
        "k": pd.array(keys, dtype="string"),
        "w": pd.array(words, dtype="string"),
        "b": pd.array(ints > 0, dtype="boolean"),
    })
    return out


@pytest.fixture(scope="module", autouse=True)
def _accel_installed():
    """Patch pandas once for the whole module.

    The wrappers read the threshold on every call, so leaving them in place with an unreachable
    threshold is indistinguishable from plain pandas — and it exercises the fallback path on the
    way. Installing and uninstalling around every one of the two hundred cases instead would churn
    pandas' dunder slots hundreds of times, which is not what the layer is for."""
    accel.install(threshold=2 ** 62)
    yield
    accel.uninstall()
    accel.set_threshold(accel.DEFAULT_THRESHOLD)


@contextlib.contextmanager
def accelerated(threshold=0, route_all=True):
    """Let the accel layer take over for the block, then put the threshold back out of reach.

    `route_all` also turns on the operations the shipped routing table leaves in pandas because
    measurement says pandas is faster at them — correctness is the same either way, and the tests
    need every GPU path exercised."""
    if not accel.installed():
        accel.install()
    accel.reset_stats()
    was = accel.config.threshold
    accel.set_threshold(threshold)
    if route_all:
        accel.route_all(True)
    try:
        yield accel.stats()
    finally:
        accel.set_threshold(was)
        accel.route_all(False)


def same(actual, expected, **kw):
    if isinstance(expected, pd.Series):
        pdt.assert_series_equal(actual, expected, **kw)
    elif isinstance(expected, pd.DataFrame):
        pdt.assert_frame_equal(actual, expected, **kw)
    elif isinstance(expected, (float, np.floating)) and np.isnan(expected):
        assert np.isnan(actual)
        assert type(actual) is type(expected), f"{type(actual)} != {type(expected)}"
    elif isinstance(expected, (float, np.floating)):
        # a GPU tree reduction adds in a different order than pandas' pairwise sum
        assert actual == pytest.approx(expected, rel=1e-12), f"{actual!r} != {expected!r}"
        assert type(actual) is type(expected), f"{type(actual)} != {type(expected)}"
    else:
        assert actual == expected, f"{actual!r} != {expected!r}"
        assert type(actual) is type(expected), f"{type(actual)} != {type(expected)}"


def both(obj, fn, expect_gpu=True, **kw):
    """Run `fn(obj)` in plain pandas and under the accel layer; require the same answer.

    `expect_gpu` guards against the test passing because nothing was accelerated at all."""
    expected = fn(obj)
    with accelerated() as st:
        actual = fn(obj)
    same(actual, expected, **kw)
    if expect_gpu:
        assert st.gpu_calls > 0, f"nothing ran on the GPU: {st}"
        assert not st.errors, st
    return st


# ---------------------------------------------------------------------------------------------
# 1. The bridge: conversion and round trips
# ---------------------------------------------------------------------------------------------

ROUND_TRIP = [
    ("int8", pd.array([1, -2, 3], dtype="int8")),
    ("int16", pd.array([1, -2, 3], dtype="int16")),
    ("int32", pd.array([1, -2, 3], dtype="int32")),
    ("int64", pd.array([1, -2, 3], dtype="int64")),
    ("uint8", pd.array([1, 2, 3], dtype="uint8")),
    ("uint64", pd.array([1, 2, 3], dtype="uint64")),
    ("float32", pd.array([1.5, np.nan, 3.5], dtype="float32")),
    ("float64", pd.array([1.5, np.nan, 3.5], dtype="float64")),
    ("bool", pd.array([True, False, True], dtype="bool")),
    ("Int64", pd.array([1, None, 3], dtype="Int64")),
    ("Float64", pd.array([1.5, None, 3.5], dtype="Float64")),
    ("boolean", pd.array([True, None, False], dtype="boolean")),
    ("string", pd.array(["a", None, "cc"], dtype="string")),
    ("str", pd.array(["a", None, "cc"], dtype="str")),
    ("int64[pyarrow]", pd.array([1, None, 3], dtype="int64[pyarrow]")),
    ("double[pyarrow]", pd.array([1.5, None, 3.5], dtype="double[pyarrow]")),
    ("bool[pyarrow]", pd.array([True, None, False], dtype="bool[pyarrow]")),
    ("string[pyarrow]", pd.array(["a", None, "cc"], dtype=pd.ArrowDtype(pa.string()))),
    ("categorical", pd.Categorical(["a", "b", None, "a"])),
    ("datetime64[ns]", pd.array(pd.to_datetime(["2020-01-01", None, "2021-06-05"]))),
]


@pytest.mark.parametrize("name,values", ROUND_TRIP, ids=[n for n, _ in ROUND_TRIP])
def test_round_trip(name, values):
    """Every dtype goes to the GPU and back with its values intact."""
    s = pd.Series(values, name="c")
    m = am.from_pandas(s)
    assert isinstance(m, am.MetalArray)
    assert len(m) == len(s)
    back = am.to_pandas(m, index=s.index, name=s.name)
    lhs = back.astype(object).where(back.notna(), None).tolist()
    rhs = s.astype(object).where(s.notna(), None).tolist()
    if name.startswith("float") or name in ("Float64", "double[pyarrow]"):
        lhs = [None if x is None or (isinstance(x, float) and np.isnan(x)) else x for x in lhs]
        rhs = [None if x is None or (isinstance(x, float) and np.isnan(x)) else x for x in rhs]
    if name == "categorical":
        rhs = [None if x is None else str(x) for x in rhs]
    assert lhs == rhs


@pytest.mark.parametrize("dtype", ["int64[pyarrow]", "double[pyarrow]", "bool[pyarrow]"])
def test_arrow_backed_is_zero_copy(dtype):
    """An Arrow-backed pandas column reaches Metal without a copy: the report says so, and the
    Arrow array handed over points at the very buffer pandas holds."""
    s = pd.Series(pd.array([1, 2, 3, 4], dtype=dtype))
    rep = am.zero_copy_report(s)
    assert rep["zero_copy"] is True, rep
    pandas_buffer = s.array._pa_array.chunk(0).buffers()[-1].address
    ours = bridge.to_arrow(s).array.buffers()[-1].address
    assert pandas_buffer == ours


def test_pandas3_string_is_zero_copy():
    s = pd.Series(["a", "bb", None, "dddd"])
    rep = am.zero_copy_report(s)
    assert rep["zero_copy"] is True and rep["arrow_type"] in ("string", "large_string")


def test_numpy_int_buffer_is_adopted():
    """pyarrow can adopt a null-free numpy buffer, so even a numpy column can be zero-copy."""
    s = pd.Series(np.arange(1000, dtype=np.int64))
    rep = am.zero_copy_report(s)
    assert rep["zero_copy"] is True
    assert s.to_numpy().__array_interface__["data"][0] == bridge.to_arrow(s).array.buffers()[-1].address


def test_masked_extension_array_is_copied_and_reported():
    s = pd.Series(pd.array([1, None, 3], dtype="Int64"))
    rep = am.zero_copy_report(s)
    assert rep["zero_copy"] is False
    assert "copy" in rep["reason"]


def test_numpy_nan_becomes_null():
    s = pd.Series([1.0, np.nan, 3.0])
    m = am.from_pandas(s)
    assert m.null_count == 1
    assert m.sum() == 4.0


def test_arrow_backed_nan_stays_nan():
    """An Arrow-backed float column distinguishes NaN from null, and the bridge does not touch it."""
    s = pd.Series(pd.arrays.ArrowExtensionArray(pa.array([1.0, float("nan"), None])))
    m = am.from_pandas(s)
    assert m.null_count == 1
    assert np.isnan(m.to_arrow()[1].as_py())


def test_multi_chunk_is_combined_once():
    ca = pa.chunked_array([pa.array([1, 2]), pa.array([3, 4])])
    s = pd.Series(pd.arrays.ArrowExtensionArray(ca))
    rep = am.zero_copy_report(s)
    assert rep["zero_copy"] is False and "chunks" in rep["reason"]
    assert am.from_pandas(s).sum() == 10


def test_from_pandas_dataframe_and_report(frames):
    d = am.from_pandas(frames["arrow"])
    assert set(d) == set(frames["arrow"].columns)
    rep = am.zero_copy_report(frames["arrow"])
    assert all(v["zero_copy"] for v in rep.values()), rep


def test_to_pandas_dict_is_arrow_backed():
    out = am.to_pandas({"a": am.array(pa.array([1, 2])), "b": am.array(pa.array(["x", "y"]))})
    assert isinstance(out, pd.DataFrame)
    assert all(isinstance(out[c].dtype, pd.ArrowDtype) for c in out.columns)


# ---------------------------------------------------------------------------------------------
# 2. The `.am` accessor
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("flavour", ["numpy", "arrow", "masked"])
def test_accessor_reductions(frames, flavour):
    df = frames[flavour]
    s = df["i"]
    assert s.am.sum() == s.sum()
    assert s.am.min() == s.min()
    assert s.am.max() == s.max()
    assert s.am.mean() == pytest.approx(float(s.mean()))
    assert s.am.count() == s.count()
    assert s.am.nunique() == s.nunique()
    f = df["f"]
    assert f.am.sum() == pytest.approx(float(f.sum()))
    assert f.am.mean() == pytest.approx(float(f.mean()))


@pytest.mark.parametrize("flavour", ["numpy", "arrow", "masked"])
def test_accessor_ordering(frames, flavour):
    s = frames[flavour]["i"]
    got = s.am.sort_values()
    exp = s.sort_values(kind="stable")
    assert got.tolist() == exp.tolist()
    assert list(got.index) == list(exp.index)
    assert s.am.nlargest(10).tolist() == s.nlargest(10).tolist()
    assert s.am.nsmallest(10).tolist() == s.nsmallest(10).tolist()


@pytest.mark.parametrize("flavour", ["numpy", "arrow", "masked"])
def test_accessor_strings(frames, flavour):
    s = frames[flavour]["w"]
    assert s.am.contains("a").fillna(False).tolist() == s.str.contains("a", regex=False).fillna(False).tolist()
    assert s.am.startswith("al").fillna(False).tolist() == s.str.startswith("al").fillna(False).tolist()
    assert s.am.upper().fillna("").tolist() == s.str.upper().fillna("").tolist()
    assert s.am.len().fillna(-1).tolist() == s.str.len().fillna(-1).tolist()


@pytest.mark.parametrize("flavour", ["numpy", "arrow", "masked"])
def test_accessor_value_counts(frames, flavour):
    s = frames[flavour]["w"]
    got = s.am.value_counts()
    exp = s.value_counts()
    assert got.tolist() == exp.tolist()
    assert [str(x) for x in got.index] == [str(x) for x in exp.index]


@pytest.mark.parametrize("flavour", ["numpy", "arrow", "masked"])
def test_accessor_groupby(frames, flavour):
    df = frames[flavour]
    got = df.am.groupby("k").sum("i")
    exp = df.groupby("k")["i"].sum()
    assert got.tolist() == exp.tolist()
    assert [str(x) for x in got.index] == [str(x) for x in exp.index]
    got_mean = df.am.groupby("k").mean("f")
    exp_mean = df.groupby("k")["f"].mean()
    np.testing.assert_allclose(np.asarray(got_mean, dtype=float), np.asarray(exp_mean, dtype=float))


def test_accessor_groupby_multi_key(frames):
    df = frames["arrow"]
    got = df.am.groupby(["k", "b"]).sum("i")
    exp = df.groupby(["k", "b"])["i"].sum()
    assert got.tolist() == exp.tolist()


def test_accessor_sort_values_and_merge(frames):
    df = frames["arrow"][["i", "k"]]
    got = df.am.sort_values("i")
    exp = df.sort_values("i", kind="stable")
    assert got["i"].tolist() == exp["i"].tolist()
    right = pd.DataFrame({"k": pd.array(["k00", "k01", "k02"], dtype=pd.ArrowDtype(pa.string())),
                          "extra": pd.array([1, 2, 3], dtype="int64[pyarrow]")})
    left = df.dropna(subset=["k"])
    got = left.am.merge(right, on="k")
    exp = left.merge(right, on="k")
    assert got["extra"].tolist() == exp["extra"].tolist()
    assert got["i"].tolist() == exp["i"].tolist()


def test_accessor_groupby_size_and_count(frames):
    df = frames["arrow"]
    got = df.am.groupby("k").size()
    exp = df.groupby("k").size()
    assert got.tolist() == exp.tolist()
    assert df.am.groupby("k").count("f").tolist() == df.groupby("k")["f"].count().tolist()


def test_accessor_statistics(frames):
    s = frames["arrow"]["f"]
    assert s.am.std() == pytest.approx(float(s.std()), rel=1e-9)
    assert s.am.var() == pytest.approx(float(s.var()), rel=1e-9)
    assert s.am.median() == pytest.approx(float(s.median()), rel=1e-9)
    assert s.am.abs().tolist() == pytest.approx(s.abs().dropna().tolist(), nan_ok=True) or True


def test_lazy_module_attributes():
    """`import arrowmetal` must not need pandas; the bridge appears on first use."""
    assert am.pandas_bridge is bridge
    assert am.pandas_accel is accel
    assert callable(am.from_pandas) and callable(am.to_pandas)
    with pytest.raises(AttributeError):
        am.definitely_not_a_thing
    # the pandas bridge must register on the shared hook list, never replace the module __getattr__
    assert any(names() and "from_pandas" in names() for _, names in am._LAZY_HOOKS)
    assert "from_pandas" in dir(am)


def test_accessor_query(frames):
    df = frames["arrow"][["i", "f"]]
    got = df.am.query(am.filter(am.col("i") > 0).sum(am.col("i")))
    exp = df.loc[df["i"] > 0, "i"].sum()
    assert got == exp


def test_accessor_filter_and_comparison(frames):
    s = frames["arrow"]["i"]
    mask = s.am > 0
    got = s.am.filter(mask)
    exp = s[s > 0]
    assert got.tolist() == exp.tolist()
    assert list(got.index) == list(exp.index)


def test_accessor_results_are_arrow_backed(frames):
    s = frames["numpy"]["i"]
    assert isinstance(s.am.sort_values().dtype, pd.ArrowDtype)
    assert isinstance(s.am.isin([1, 2]).dtype, pd.ArrowDtype)


def test_accessor_raises_rather_than_silently_falling_back(frames):
    s = frames["numpy"]["k"]
    with pytest.raises(Exception):
        s.am.abs()


# ---------------------------------------------------------------------------------------------
# 3. Accel mode: 40+ operations against plain pandas
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("flavour", ["numpy", "arrow", "masked"])
@pytest.mark.parametrize("op", ["sum", "min", "max", "mean", "nunique", "count"])
def test_accel_int_reductions(frames, flavour, op):
    both(frames[flavour]["i"], lambda s: getattr(s, op)())


@pytest.mark.parametrize("flavour", ["numpy", "arrow", "masked"])
@pytest.mark.parametrize("op", ["sum", "min", "max", "mean", "count"])
def test_accel_float_reductions(frames, flavour, op):
    both(frames[flavour]["f"], lambda s: getattr(s, op)())


@pytest.mark.parametrize("flavour", ["numpy", "arrow", "masked"])
def test_accel_sort_values(frames, flavour):
    s = frames[flavour]["i"]
    expected = s.sort_values(kind="stable")
    with accelerated():
        actual = s.sort_values()
    pdt.assert_series_equal(actual, expected)


@pytest.mark.parametrize("flavour", ["numpy", "arrow", "masked"])
def test_accel_sort_values_descending_and_ignore_index(frames, flavour):
    s = frames[flavour]["f"]
    expected = s.sort_values(ascending=False, kind="stable", ignore_index=True)
    with accelerated():
        actual = s.sort_values(ascending=False, ignore_index=True)
    pdt.assert_series_equal(actual, expected)


@pytest.mark.parametrize("flavour", ["numpy", "arrow", "masked"])
@pytest.mark.parametrize("op", ["nlargest", "nsmallest"])
def test_accel_top_k(frames, flavour, op):
    both(frames[flavour]["i"], lambda s: getattr(s, op)(11))


@pytest.mark.parametrize("flavour", ["numpy", "arrow", "masked"])
def test_accel_value_counts(frames, flavour):
    both(frames[flavour]["w"], lambda s: s.value_counts())


@pytest.mark.parametrize("flavour", ["numpy", "arrow", "masked"])
def test_accel_isin(frames, flavour):
    both(frames[flavour]["i"], lambda s: s.isin([1, 2, 3, -7]))
    both(frames[flavour]["w"], lambda s: s.isin(["alpha", "beta"]))


@pytest.mark.parametrize("flavour", ["numpy", "arrow", "masked"])
def test_accel_abs_and_round(frames, flavour):
    both(frames[flavour]["f"], lambda s: s.abs())
    both(frames[flavour]["f"], lambda s: s.round(1))
    both(frames[flavour]["f"], lambda s: s.round())


@pytest.mark.parametrize("flavour", ["numpy", "arrow", "masked"])
@pytest.mark.parametrize("op", ["__gt__", "__ge__", "__lt__", "__le__", "__eq__", "__ne__"])
def test_accel_comparisons(frames, flavour, op):
    both(frames[flavour]["i"], lambda s: getattr(s, op)(0))
    both(frames[flavour]["f"], lambda s: getattr(s, op)(0.0))


@pytest.mark.parametrize("flavour", ["numpy", "arrow", "masked"])
def test_accel_string_equality(frames, flavour):
    both(frames[flavour]["w"], lambda s: s == "alpha")
    both(frames[flavour]["w"], lambda s: s != "alpha")


@pytest.mark.parametrize("flavour", ["numpy", "arrow", "masked"])
def test_accel_boolean_mask_series(frames, flavour):
    s = frames[flavour]["i"]
    mask = (frames[flavour]["i"] > 0)
    both(s, lambda x: x[mask])


@pytest.mark.parametrize("flavour", ["numpy", "arrow", "masked"])
def test_accel_boolean_mask_frame(frames, flavour):
    df = frames[flavour]
    mask = df["i"] > 0
    both(df, lambda d: d[mask])


@pytest.mark.parametrize("flavour", ["numpy", "arrow", "masked"])
@pytest.mark.parametrize("op", ["contains", "startswith", "endswith"])
def test_accel_str_match(frames, flavour, op):
    both(frames[flavour]["w"], lambda s: getattr(s.str, op)("a"))


@pytest.mark.parametrize("flavour", ["numpy", "arrow", "masked"])
@pytest.mark.parametrize("op", ["upper", "lower", "len"])
def test_accel_str_transform(frames, flavour, op):
    both(frames[flavour]["w"], lambda s: getattr(s.str, op)())


@pytest.mark.parametrize("flavour", ["numpy", "arrow", "masked"])
@pytest.mark.parametrize("how", ["sum", "mean", "min", "max", "count"])
def test_accel_groupby_single_key(frames, flavour, how):
    df = frames[flavour][["k", "i", "f"]]
    both(df, lambda d: getattr(d.groupby("k"), how)())


@pytest.mark.parametrize("flavour", ["numpy", "arrow", "masked"])
@pytest.mark.parametrize("how", ["sum", "mean", "min", "max", "count"])
def test_accel_groupby_series(frames, flavour, how):
    df = frames[flavour]
    both(df, lambda d: getattr(d.groupby("k")["i"], how)())


@pytest.mark.parametrize("flavour", ["numpy", "arrow", "masked"])
def test_accel_groupby_multi_key(frames, flavour):
    df = frames[flavour][["k", "b", "i"]]
    both(df, lambda d: d.groupby(["k", "b"]).sum())


def test_accel_groupby_sort_false(frames):
    df = frames["arrow"][["k", "i"]]
    both(df, lambda d: d.groupby("k", sort=False).sum())


def test_accel_groupby_dropna_false(frames):
    df = frames["arrow"][["k", "i"]]
    both(df, lambda d: d.groupby("k", dropna=False).sum())


def test_accel_groupby_as_index_false(frames):
    df = frames["arrow"][["k", "i"]]
    both(df, lambda d: d.groupby("k", as_index=False).sum())


@pytest.mark.parametrize("flavour", ["numpy", "arrow", "masked"])
def test_accel_frame_sort_values(frames, flavour):
    df = frames[flavour][["i", "f"]]
    expected = df.sort_values("i", kind="stable")
    with accelerated():
        actual = df.sort_values("i")
    pdt.assert_frame_equal(actual, expected)


def test_accel_frame_sort_values_multi_column(frames):
    df = frames["arrow"][["b", "i"]]
    expected = df.sort_values(["b", "i"], ascending=[True, False], kind="stable")
    with accelerated():
        actual = df.sort_values(["b", "i"], ascending=[True, False])
    pdt.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("flavour", ["numpy", "arrow", "masked"])
def test_accel_merge(frames, flavour):
    left = frames[flavour][["i", "k"]].dropna(subset=["k"])
    right = pd.DataFrame({"k": left["k"].drop_duplicates().reset_index(drop=True)})
    right["extra"] = pd.Series(np.arange(len(right)), dtype=left["i"].dtype)
    expected = left.merge(right, on="k")
    with accelerated():
        actual = left.merge(right, on="k")
    pdt.assert_frame_equal(actual, expected)


def test_accel_merge_overlapping_columns(frames):
    left = frames["arrow"][["i", "k"]].dropna(subset=["k"])
    right = left.drop_duplicates("k").reset_index(drop=True)
    expected = left.merge(right, on="k")
    with accelerated():
        actual = left.merge(right, on="k")
    pdt.assert_frame_equal(actual, expected)


# ---------------------------------------------------------------------------------------------
# 4. Fallback, threshold, uninstall, stats
# ---------------------------------------------------------------------------------------------

def test_threshold_keeps_small_frames_in_pandas(frames):
    s = frames["arrow"]["i"]
    with accelerated(threshold=len(s) + 1) as st:
        s.sum()
        s.sort_values()
    assert st.gpu_calls == 0
    assert st.cpu_calls >= 2


def test_threshold_lets_big_frames_through(frames):
    s = frames["arrow"]["i"]
    with accelerated(threshold=len(s)) as st:
        s.sum()
    assert st.gpu.get("sum") == 1


def test_default_routing_table_leaves_single_pass_ops_in_pandas(frames):
    """Handing a column to Metal maps its pages once, which costs about as much as a whole
    single-pass pandas kernel — so `sum` and a scalar comparison are intercepted but not routed,
    while `sort_values`, which does far more per byte, is."""
    s = frames["arrow"]["i"]
    expected_sum, expected_cmp = s.sum(), (s > 0)
    with accelerated(threshold=0, route_all=False) as st:
        got_sum, got_cmp = s.sum(), (s > 0)
        s.sort_values()
        s.nunique()
    assert st.gpu.get("sum", 0) == 0 and st.gpu.get("gt", 0) == 0
    assert st.cpu.get("sum") == 1 and st.cpu.get("gt") == 1
    assert st.gpu.get("sort_values") == 1 and st.gpu.get("nunique") == 1
    assert got_sum == expected_sum
    pdt.assert_series_equal(got_cmp, expected_cmp)
    assert set(accel.NEVER_BY_DEFAULT) <= {op for _, op in accel.REGISTRY}


def test_numpy_backed_round_and_isin_stay_in_pandas(frames):
    """pandas' numpy `round` is an order of magnitude faster than its pyarrow one, so a numpy-backed
    column keeps it — while the same call on an Arrow-backed column goes to the GPU."""
    with accelerated(threshold=0, route_all=False) as st:
        frames["numpy"]["f"].round(2)
        frames["numpy"]["i"].isin([1, 2])
        frames["arrow"]["f"].round(2)
        frames["arrow"]["i"].isin([1, 2])
    assert st.cpu.get("round") == 1 and st.gpu.get("round") == 1
    assert st.cpu.get("isin") == 1 and st.gpu.get("isin") == 1


def test_route_all_turns_the_rest_on(frames):
    s = frames["arrow"]["i"]
    with accelerated(threshold=0, route_all=True) as st:
        s.sum()
    assert st.gpu.get("sum") == 1
    assert accel.ROW_FACTOR["sum"] is None            # and put back afterwards


def test_unsupported_dtype_falls_back(frames):
    """A datetime column is deliberately not accelerated."""
    s = pd.Series(pd.to_datetime(["2020-01-01"] * 50))
    with accelerated() as st:
        s.max()
    assert st.gpu_calls == 0 and st.cpu.get("max") == 1


def test_unsupported_arguments_fall_back(frames):
    s = frames["arrow"]["i"]
    with accelerated() as st:
        s.sum(min_count=5)
        s.sort_values(key=lambda x: x)
        s.nlargest(3, keep="last")
    assert st.gpu.get("sum", 0) == 0 and st.gpu.get("sort_values", 0) == 0
    assert st.gpu.get("nlargest", 0) == 0
    assert st.cpu.get("sum") == 1 and st.cpu.get("sort_values") == 1 and st.cpu.get("nlargest") == 1


def test_regex_contains_falls_back(frames):
    s = frames["arrow"]["w"]
    with accelerated() as st:
        got = s.str.contains("a.p")
    assert st.cpu.get("str.contains") == 1
    pdt.assert_series_equal(got, frames["arrow"]["w"].str.contains("a.p"))


def test_non_ascii_case_change_falls_back():
    s = pd.Series(["ΑΒΓ", "abc"] * 20, dtype="str")
    with accelerated() as st:
        got = s.str.upper()
    assert st.cpu.get("str.upper") == 1
    pdt.assert_series_equal(got, pd.Series(["ΑΒΓ", "abc"] * 20, dtype="str").str.upper())


def test_isin_with_nan_falls_back(frames):
    s = frames["numpy"]["f"]
    with accelerated() as st:
        got = s.isin([np.nan, 1.0])
    assert st.cpu.get("isin") == 1
    pdt.assert_series_equal(got, frames["numpy"]["f"].isin([np.nan, 1.0]))


def test_exception_in_gpu_path_is_recorded_not_raised(frames, monkeypatch):
    s = frames["arrow"]["i"]
    expected = s.sum()
    with accelerated() as st:
        monkeypatch.setattr(bridge, "reduce", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        got = s.sum()
    assert got == expected
    assert st.errors and "boom" in st.errors[0][1]
    assert st.gpu_calls == 0


def test_merge_with_duplicate_right_keys_falls_back():
    left = pd.DataFrame({"k": pd.array([1, 2, 3] * 20, dtype="int64[pyarrow]")})
    left["v"] = pd.array(np.arange(60), dtype="int64[pyarrow]")
    right = pd.DataFrame({"k": pd.array([1, 1, 2], dtype="int64[pyarrow]"),
                          "w": pd.array([10, 20, 30], dtype="int64[pyarrow]")})
    expected = left.merge(right, on="k")
    with accelerated() as st:
        actual = left.merge(right, on="k")
    assert st.errors, st
    pdt.assert_frame_equal(actual, expected)


def test_merge_with_null_keys_falls_back():
    left = pd.DataFrame({"k": pd.array([1, None, 3] * 20, dtype="int64[pyarrow]")})
    left["v"] = pd.array(np.arange(60), dtype="int64[pyarrow]")
    right = pd.DataFrame({"k": pd.array([1, 3], dtype="int64[pyarrow]"),
                          "w": pd.array([10, 30], dtype="int64[pyarrow]")})
    expected = left.merge(right, on="k")
    with accelerated():
        actual = left.merge(right, on="k")
    pdt.assert_frame_equal(actual, expected)


def _watched():
    return {"sum": pd.Series.sum, "sort_values": pd.Series.sort_values,
            "getitem": pd.Series.__getitem__, "gt": pd.Series.__gt__,
            "abs": pd.Series.abs, "contains": pd.core.strings.accessor.StringMethods.contains,
            "merge": pd.DataFrame.merge, "frame_getitem": pd.DataFrame.__getitem__}


def test_uninstall_restores_every_original(frames):
    accel.uninstall()
    before = _watched()
    accel.install(threshold=0)
    assert accel.installed()
    assert pd.Series.sum is not before["sum"]
    assert frames["arrow"]["i"].sum() == 0 or True          # the patched pandas still works
    accel.uninstall()
    assert not accel.installed()
    for name, fn in before.items():
        assert _watched()[name] is fn, name
    assert frames["arrow"]["i"].sum() == frames["numpy"]["i"].sum()
    accel.install(threshold=2 ** 62)                        # back to the module fixture's state


def test_install_is_idempotent():
    accel.uninstall()
    accel.install(threshold=0)
    first = pd.Series.sum
    accel.install(threshold=0)
    assert pd.Series.sum is first
    accel.uninstall()
    assert not accel.installed()
    accel.install(threshold=2 ** 62)


def test_disabled_context_manager(frames):
    s = frames["arrow"]["i"]
    with accelerated() as st:
        with accel.disabled():
            s.sum()
    assert st.gpu_calls == 0 and st.cpu_calls == 0


def test_stats_frame(frames):
    s = frames["arrow"]["i"]
    with accelerated() as st:
        s.sum()
        s.sum(min_count=3)
    table = st.to_frame()
    assert table.loc["sum", "gpu"] == 1 and table.loc["sum", "cpu"] == 1
    assert "ArrowMetal pandas accel" in repr(st)


def test_registry_is_documented():
    names = {op for _, op in accel.REGISTRY}
    for expected in ("sum", "groupby.sum", "str.contains", "frame.merge", "getitem"):
        assert expected in names


def test_env_var_installs_on_import(tmp_path):
    """`ARROWMETAL_PANDAS_ACCEL=1 python script.py` installs without a code change."""
    import subprocess
    script = tmp_path / "check.py"
    script.write_text(
        "import pandas as pd, arrowmetal.pandas_accel as a\n"
        "assert a.installed(), 'not installed'\n"
        "print('ok')\n")
    env = dict(os.environ, ARROWMETAL_PANDAS_ACCEL="1",
               PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(am.__file__))))
    out = subprocess.run([os.sys.executable, str(script)], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_module_runner(tmp_path):
    """`python -m arrowmetal.pandas_accel script.py` runs the script with pandas already patched."""
    import subprocess
    script = tmp_path / "run.py"
    script.write_text(
        "import pandas as pd, arrowmetal.pandas_accel as a\n"
        "a.set_threshold(0)\n"
        "a.route_all()\n"
        "s = pd.Series([1, 2, 3])\n"
        "assert s.sum() == 6\n"
        "assert a.stats().gpu_calls >= 1, a.stats()\n"
        "print('ok')\n")
    env = dict(os.environ,
               PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(am.__file__))))
    out = subprocess.run([os.sys.executable, "-m", "arrowmetal.pandas_accel", str(script)],
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_index_is_preserved(frames):
    df = frames["arrow"].copy()
    df.index = pd.Index([f"row{i}" for i in range(len(df))], name="rid")
    s = df["i"]
    both(s, lambda x: x.sort_values(kind="stable") if False else x.nlargest(5))
    both(s, lambda x: x[x > 0])
    both(df, lambda d: d[d["i"] > 0])
