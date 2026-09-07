"""The three Polars tiers, checked against native Polars on the same data.

Run: PYTHONPATH=python python -m pytest python/tests/test_polars.py -q

Tier 1 (the bridge and the `.arrowmetal` namespaces) needs only Polars. Tier 2 (the expression
plugin) additionally needs `polars-plugin/target/release/libarrowmetal_polars.dylib`; the plugin
tests skip when it has not been built, so a checkout without a Rust toolchain still runs green.

The timing assertions at the end use bounds generous enough to survive a busy machine -- they are
there to catch an accidental copy or a per-row Python loop (which would be 100x over), not to
police milliseconds.
"""
import time

import numpy as np
import pyarrow as pa
import pytest

import arrowmetal as am

pl = pytest.importorskip("polars")

from arrowmetal import polars_bridge  # noqa: E402  (registers the namespaces)
from arrowmetal import polars_plugin  # noqa: E402

PLUGIN = polars_plugin.available()
needs_plugin = pytest.mark.skipif(
    not PLUGIN, reason="build it with: cd polars-plugin && cargo build --release")

SIZES = [0, 1, 33, 4097, 100_003]


def rng(seed=7):
    return np.random.default_rng(seed)


def frame(n, seed=7):
    r = rng(seed)
    return pl.DataFrame({
        "k": r.integers(0, 17, n).astype(np.int32),
        "v": r.integers(-1000, 1000, n).astype(np.int64),
        "f": r.standard_normal(n) * 100.0,
        "s": pl.Series("s", [f"row-{i % 53}" for i in range(n)], dtype=pl.String),
    })


# =============================================================================================
# Tier 1 -- the bridge: round trips
# =============================================================================================

ALL_DTYPES = [
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64, pl.Boolean,
    pl.Date, pl.Datetime("us"), pl.Datetime("ms"), pl.Time, pl.Duration("us"),
    pl.String, pl.Binary,
]


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_round_trip_preserves_values_and_dtype(dtype):
    """Every dtype Polars and ArrowMetal share survives from_polars -> to_polars unchanged."""
    if dtype == pl.Boolean:
        values = [True, False, None, True]
    elif dtype == pl.String:
        values = ["a", "", None, "a longer one"]
    elif dtype == pl.Binary:
        values = [b"a", b"", None, b"\x00\xff"]
    elif dtype in (pl.Date, pl.Time) or isinstance(dtype, (pl.Datetime, pl.Duration)):
        values = [0, 1, None, 1000]
    elif dtype in (pl.Float32, pl.Float64):
        values = [1.5, -0.0, None, float("inf")]
    else:
        values = [1, 2, None, 3]
    s = pl.Series("x", values, dtype=dtype) if dtype not in (pl.Date, pl.Time) and \
        not isinstance(dtype, (pl.Datetime, pl.Duration)) else \
        pl.Series("x", values, dtype=pl.Int64).cast(dtype)

    back = am.to_polars(am.from_polars(s), "x")
    assert back.dtype == s.dtype
    assert back.to_list() == s.to_list()


def test_round_trip_empty_and_all_null():
    for s in (pl.Series("x", [], dtype=pl.Int64),
              pl.Series("x", [None, None, None], dtype=pl.Float64)):
        back = am.to_polars(am.from_polars(s), "x")
        assert back.to_list() == s.to_list()
        assert back.dtype == s.dtype


def test_round_trip_categorical_and_enum():
    """Polars encodes Categorical as dictionary<uint32> and Enum as dictionary<uint8>; ArrowMetal
    wants int32 or int64 indices, so the bridge recodes them (see _widen_dictionary_indices)."""
    for s in (pl.Series("c", ["a", "b", "a", None], dtype=pl.Categorical),
              pl.Series("e", ["a", "b", "a", None], dtype=pl.Enum(["a", "b", "c"]))):
        m = am.from_polars(s)
        assert m.format == "i"                      # dictionary, int32 indices
        back = am.to_polars(m, s.name)
        assert back.cast(pl.String).to_list() == s.cast(pl.String).to_list()


def test_round_trip_chunked_series_rechunks_once():
    a = pl.Series("x", np.arange(1000, dtype=np.int64))

    chunked = pl.concat([a, a, a], rechunk=False)
    assert chunked.n_chunks() == 3
    back = am.to_polars(am.from_polars(chunked), "x")
    assert back.to_list() == a.to_list() * 3
    # `Series.to_arrow()` combines the chunks in place, so the Series is single-chunk afterwards:
    # the rechunk really did happen once, not once per call.
    assert chunked.n_chunks() == 1

    # rechunk=False refuses rather than paying for the concatenation behind your back.
    fresh = pl.concat([a, a, a], rechunk=False)
    with pytest.raises(am.ArrowMetalError, match="chunks"):
        am.from_polars(fresh, rechunk=False)

    # A single-chunk Series goes through the same code path untouched.
    assert am.from_polars(a, rechunk=False).sum() == a.sum()


def test_dataframe_round_trip():
    df = frame(1000)
    columns = am.from_polars(df)
    assert set(columns) == set(df.columns)
    back = am.to_polars(columns)
    assert back.equals(df.select(back.columns))


# =============================================================================================
# Tier 1 -- zero copy
# =============================================================================================

@pytest.mark.parametrize("dtype", [np.int64, np.float64, np.int32])
def test_import_is_zero_copy_for_a_large_column(dtype):
    """The buffer ArrowMetal reads is the buffer Polars owns -- same address, nothing copied."""
    s = pl.Series("x", np.arange(4_000_000, dtype=dtype))
    src, dst, same = am.zero_copy_report(s)
    assert src is not None
    assert same, f"expected the same buffer, got {src:#x} and {dst:#x}"


def test_export_is_zero_copy():
    """Coming back is free too: the pyarrow array points at the Metal buffer, and Polars adopts
    it without a copy."""
    s = pl.Series("x", np.arange(4_000_000, dtype=np.int64))
    m = am.from_polars(s)
    exported = m.to_arrow()
    back = pl.from_arrow(exported)
    assert exported.buffers()[1].address == back.to_arrow().buffers()[1].address


def test_a_kernel_result_is_a_metal_buffer_not_the_input():
    """Sanity check on the other direction: a computed column is a *new* buffer, so an unchanged
    address would mean the test above was measuring nothing."""
    s = pl.Series("x", np.arange(1_000_000, dtype=np.int64))
    m = am.from_polars(s)
    doubled = m * 2
    assert doubled.to_arrow().buffers()[1].address != s.to_arrow().buffers()[1].address


# =============================================================================================
# Tier 1 -- Series namespace against native Polars
# =============================================================================================

@pytest.mark.parametrize("n", SIZES)
def test_series_reductions_match_polars(n):
    df = frame(n)
    for col in ("v", "f"):
        s = df[col]
        if n == 0:
            # An empty column has no value to answer with, so ArrowMetal reports null where
            # Polars reports 0 for sum and None for the rest. Both are defensible; this pins ours.
            assert s.arrowmetal.sum() is None
            assert s.arrowmetal.min() is None
            assert s.arrowmetal.count() == 0
            continue
        assert s.arrowmetal.sum() == pytest.approx(s.sum(), rel=1e-12)
        assert s.arrowmetal.min() == pytest.approx(s.min(), rel=1e-12)
        assert s.arrowmetal.max() == pytest.approx(s.max(), rel=1e-12)
        assert s.arrowmetal.mean() == pytest.approx(s.mean(), rel=1e-12)
        assert s.arrowmetal.count() == s.count()


def test_series_reductions_with_nulls_match_polars():
    r = rng(3)
    v = r.integers(-100, 100, 5000).astype(np.int64)
    mask = r.random(5000) < 0.2
    s = pl.Series("x", pa.array(v, mask=mask))
    assert s.arrowmetal.sum() == s.sum()
    assert s.arrowmetal.min() == s.min()
    assert s.arrowmetal.max() == s.max()
    assert s.arrowmetal.mean() == pytest.approx(s.mean())


def test_series_top_k_and_sort_match_polars():
    s = frame(10_000)["v"]
    assert sorted(s.arrowmetal.top_k(100).to_list()) == sorted(s.top_k(100).to_list())
    assert sorted(s.arrowmetal.bottom_k(100).to_list()) == sorted(s.bottom_k(100).to_list())
    assert s.arrowmetal.sort().to_list() == s.sort().to_list()
    assert s.arrowmetal.sort(descending=True).to_list() == s.sort(descending=True).to_list()
    idx = s.arrowmetal.arg_sort()
    assert s.gather(idx).to_list() == s.sort().to_list()


def test_series_string_methods_match_polars():
    s = frame(5000)["s"]
    assert s.arrowmetal.contains("row-1").to_list() == \
        s.str.contains("row-1", literal=True).to_list()
    assert s.arrowmetal.starts_with("row-1").to_list() == s.str.starts_with("row-1").to_list()
    assert s.arrowmetal.ends_with("9").to_list() == s.str.ends_with("9").to_list()
    assert s.arrowmetal.upper().to_list() == s.str.to_uppercase().to_list()
    assert s.arrowmetal.lower().to_list() == s.str.to_lowercase().to_list()


def test_series_filter_unique_cumsum_and_hash():
    df = frame(5000)
    s, k = df["v"], df["k"]
    mask = k > 8
    assert s.arrowmetal.filter(mask).to_list() == s.filter(mask).to_list()
    # unique() follows Arrow's first-appearance order since the option work landed; compare as sets
    # and check the order is first appearance.
    u = s.arrowmetal.unique().to_list()
    assert sorted(u) == sorted(s.unique().to_list())
    assert u == list(dict.fromkeys(s.to_list()))
    assert s.arrowmetal.cum_sum().to_list() == s.cum_sum().to_list()
    h = s.arrowmetal.hash64()
    assert h.dtype == pl.UInt64 and len(h) == len(s)
    # Equal values hash equal -- what a hash join needs.
    pairs = dict(zip(s.to_list(), h.to_list()))
    assert all(pairs[v] == hv for v, hv in zip(s.to_list(), h.to_list()))


def test_series_namespace_returns_polars_objects():
    s = frame(100)["v"]
    assert isinstance(s.arrowmetal.top_k(3), pl.Series)
    assert isinstance(s.arrowmetal.sort(), pl.Series)
    assert isinstance(s.arrowmetal.sum(), int)
    assert isinstance(s.arrowmetal.to_metal(), am.MetalArray)


# =============================================================================================
# Tier 1 -- DataFrame namespace against native Polars
# =============================================================================================

@pytest.mark.parametrize("n", [1, 33, 4097, 100_003])
def test_group_by_sum_matches_polars(n):
    df = frame(n)
    got = df.arrowmetal.group_by("k").sum("v").sort("k")
    want = df.group_by("k").agg(pl.col("v").sum()).sort("k")
    assert got.to_dicts() == want.to_dicts()


def test_group_by_several_aggregates_in_one_pass():
    df = frame(20_000)
    got = df.arrowmetal.group_by("k").agg(
        total=("v", "sum"), avg=("f", "mean"), lo=("v", "min"), hi=("v", "max"),
        n=(None, "len"), uniq=("k", "n_unique"),
    ).sort("k")
    want = df.group_by("k").agg(
        pl.col("v").sum().alias("total"), pl.col("f").mean().alias("avg"),
        pl.col("v").min().alias("lo"), pl.col("v").max().alias("hi"),
        pl.len().alias("n"), pl.col("k").n_unique().alias("uniq"),
    ).sort("k")
    assert got["total"].to_list() == want["total"].to_list()
    assert got["lo"].to_list() == want["lo"].to_list()
    assert got["hi"].to_list() == want["hi"].to_list()
    assert got["n"].to_list() == want["n"].to_list()
    assert got["uniq"].to_list() == want["uniq"].to_list()
    for a, b in zip(got["avg"], want["avg"]):
        assert a == pytest.approx(b)


def test_group_by_string_key_matches_polars():
    df = frame(20_000)
    got = df.arrowmetal.group_by("s").sum("v").sort("s")
    want = df.group_by("s").agg(pl.col("v").sum()).sort("s")
    assert got.to_dicts() == want.to_dicts()


def test_group_by_two_keys_matches_polars():
    df = frame(20_000)
    got = df.arrowmetal.group_by("k", "s").sum("v").sort(["k", "s"])
    want = df.group_by(["k", "s"]).agg(pl.col("v").sum()).sort(["k", "s"])
    assert got.to_dicts() == want.to_dicts()


def test_group_by_keys_and_ids():
    df = frame(1000)
    gb = df.arrowmetal.group_by("k")
    assert gb.group_count == df["k"].n_unique()
    assert len(gb.keys()) == gb.group_count
    assert len(gb.ids()) == len(df)


def test_dataframe_sort_matches_polars():
    df = frame(20_000)
    got = df.arrowmetal.sort("v")
    want = df.sort("v", nulls_last=True, maintain_order=True)
    assert got["v"].to_list() == want["v"].to_list()

    got2 = df.arrowmetal.sort(["k", "v"], descending=[False, True])
    want2 = df.sort(["k", "v"], descending=[False, True], nulls_last=True, maintain_order=True)
    assert got2.select(["k", "v"]).to_dicts() == want2.select(["k", "v"]).to_dicts()


def test_dataframe_filter_and_top_k():
    df = frame(20_000)
    got = df.arrowmetal.filter(df["k"] > 8)
    want = df.filter(pl.col("k") > 8)
    assert got.to_dicts() == want.to_dicts()

    top = df.arrowmetal.top_k(10, by="v")
    assert sorted(top["v"].to_list(), reverse=True) == sorted(df["v"].top_k(10).to_list(),
                                                              reverse=True)


def test_dataframe_query_matches_polars():
    df = frame(50_000)
    total = df.arrowmetal.query(am.filter(am.col("k") == 2).sum(am.col("v")))
    assert total == df.filter(pl.col("k") == 2)["v"].sum()

    projected = df.arrowmetal.query(am.project([("double", am.col("v") * 2)]))
    assert isinstance(projected, pl.DataFrame)
    assert projected["double"].to_list() == (df["v"] * 2).to_list()


def test_query_imports_only_the_columns_it_names():
    """Pruning is not cosmetic: on a frame with a wide string column, importing everything costs
    more than the query does."""
    df = frame(1000)
    ns = df.arrowmetal
    q = am.filter(am.col("k") == 2).sum(am.col("v"))
    assert ns._query_columns(q) == ["k", "v"]
    assert ns._query_columns(am.project([am.col("s")])) == ["s"]
    with pytest.raises(am.ArrowMetalError, match="not in the frame"):
        ns.query(am.project([am.col("nope")]))


def test_join_inner_and_left_match_polars():
    left = pl.DataFrame({"k": [1, 2, 3, 4, None], "a": [10, 20, 30, 40, 50]})
    right = pl.DataFrame({"k": [2, 3, 5], "b": ["x", "y", "z"]})

    got = left.arrowmetal.join(right, on="k").sort("k")
    want = left.join(right, on="k", how="inner").sort("k")
    assert got.to_dicts() == want.to_dicts()

    got = left.arrowmetal.join(right, on="k", how="left").sort("k", nulls_last=True)
    want = left.join(right, on="k", how="left").sort("k", nulls_last=True)
    assert got.to_dicts() == want.to_dicts()


def test_join_falls_back_when_the_right_key_repeats():
    left = pl.DataFrame({"k": [1, 2], "a": [10, 20]})
    right = pl.DataFrame({"k": [2, 2], "b": ["x", "y"]})
    got = left.arrowmetal.join(right, on="k")
    assert got.sort("b").to_dicts() == left.join(right, on="k").sort("b").to_dicts()
    with pytest.raises(am.ArrowMetalError, match="not unique"):
        left.arrowmetal.join(right, on="k", allow_cpu_fallback=False)


def test_join_string_key():
    left = pl.DataFrame({"k": ["a", "b", "c"], "v": [1, 2, 3]})
    right = pl.DataFrame({"k": ["b", "c", "d"], "w": [10, 20, 30]})
    got = left.arrowmetal.join(right, on="k").sort("k")
    want = left.join(right, on="k").sort("k")
    assert got.to_dicts() == want.to_dicts()


# =============================================================================================
# Tier 3 -- the streaming hand-off
# =============================================================================================

def test_collect_gpu_runs_a_query_over_the_collected_plan():
    df = frame(50_000)
    lf = df.lazy().filter(pl.col("k") < 8).select(["k", "v"])
    total = lf.arrowmetal.collect_gpu(am.filter(am.col("k") == 2).sum(am.col("v")))
    assert total == df.filter((pl.col("k") < 8) & (pl.col("k") == 2))["v"].sum()


def test_collect_gpu_accepts_a_callable_and_no_query():
    df = frame(10_000)
    lf = df.lazy()
    got = lf.arrowmetal.collect_gpu(lambda d: d.arrowmetal.group_by("k").sum("v")).sort("k")
    want = df.group_by("k").agg(pl.col("v").sum()).sort("k")
    assert got.to_dicts() == want.to_dicts()
    assert lf.arrowmetal.collect_gpu().equals(df)


def test_collect_gpu_keeps_polars_pushdown():
    """The Polars half still optimises: a projection on a lazy plan reaches the GPU already
    narrowed, so the collected frame has two columns, not four."""
    lf = frame(10_000).lazy().select(["k", "v"])
    assert lf.arrowmetal.collect_gpu().columns == ["k", "v"]


def test_from_polars_refuses_a_lazyframe():
    with pytest.raises(am.ArrowMetalError, match="collect"):
        am.from_polars(frame(10).lazy())


# =============================================================================================
# Tier 2 -- the expression plugin, inside lazy plans
# =============================================================================================

@needs_plugin
def test_plugin_reductions_match_polars():
    df = frame(50_000)
    lf = df.lazy()
    got = lf.select(
        pl.col("v").arrowmetal.sum().alias("sum"),
        pl.col("v").arrowmetal.min().alias("min"),
        pl.col("v").arrowmetal.max().alias("max"),
        pl.col("f").arrowmetal.mean().alias("mean"),
    ).collect()
    assert got["sum"][0] == df["v"].sum()
    assert got["min"][0] == df["v"].min()
    assert got["max"][0] == df["v"].max()
    assert got["mean"][0] == pytest.approx(df["f"].mean())
    assert got["min"].dtype == df["v"].dtype


@needs_plugin
def test_plugin_filter_sum_matches_polars():
    df = frame(50_000)
    got = df.lazy().select(
        pl.col("v").arrowmetal.filter_sum(pl.col("k") == 2).alias("total")
    ).collect()
    assert got["total"][0] == df.filter(pl.col("k") == 2)["v"].sum()


@needs_plugin
def test_plugin_elementwise_matches_polars():
    df = frame(20_000)
    got = df.lazy().select(
        pl.col("s").arrowmetal.upper().alias("up"),
        pl.col("s").arrowmetal.lower().alias("lo"),
        pl.col("s").arrowmetal.contains("row-1").alias("has"),
        pl.col("s").arrowmetal.starts_with("row-1").alias("pre"),
        pl.col("s").arrowmetal.ends_with("9").alias("post"),
        pl.col("v").arrowmetal.hash64().alias("h"),
        pl.col("v").arrowmetal.mul(3).alias("x3"),
        pl.col("f").arrowmetal.add(0.5).alias("plus"),
    ).collect()
    assert got["up"].to_list() == df["s"].str.to_uppercase().to_list()
    assert got["lo"].to_list() == df["s"].str.to_lowercase().to_list()
    assert got["has"].to_list() == df["s"].str.contains("row-1", literal=True).to_list()
    assert got["pre"].to_list() == df["s"].str.starts_with("row-1").to_list()
    assert got["post"].to_list() == df["s"].str.ends_with("9").to_list()
    assert got["h"].dtype == pl.UInt64
    assert got["x3"].to_list() == (df["v"] * 3).to_list()
    for a, b in zip(got["plus"], df["f"] + 0.5):
        assert a == pytest.approx(b)


@needs_plugin
def test_plugin_top_k_matches_polars():
    df = frame(20_000)
    got = df.lazy().select(pl.col("v").arrowmetal.top_k(50)).collect()
    assert sorted(got["v"].to_list()) == sorted(df["v"].top_k(50).to_list())
    got = df.lazy().select(pl.col("v").arrowmetal.bottom_k(50)).collect()
    assert sorted(got["v"].to_list()) == sorted(df["v"].bottom_k(50).to_list())


@needs_plugin
def test_plugin_group_by_sum_matches_polars():
    df = frame(20_000)
    got = (df.lazy()
           .select(pl.col("k").arrowmetal.group_by_sum(pl.col("v")).alias("g"))
           .collect()
           .unnest("g")
           .sort("k"))
    want = df.group_by("k").agg(pl.col("v").sum()).sort("k")
    assert got.to_dicts() == want.to_dicts()


@needs_plugin
def test_plugin_composes_with_the_rest_of_the_plan():
    """The point of tier 2: the GPU expression sits inside a plan Polars still optimises."""
    df = frame(50_000)
    got = (df.lazy()
           .filter(pl.col("k") < 8)
           .select(pl.col("v").arrowmetal.sum().alias("total"))
           .collect())
    assert got["total"][0] == df.filter(pl.col("k") < 8)["v"].sum()

    grouped = (df.lazy()
               .group_by("k")
               .agg(pl.col("v").arrowmetal.sum().alias("total"))
               .sort("k")
               .collect())
    want = df.group_by("k").agg(pl.col("v").sum().alias("total")).sort("k")
    assert grouped.to_dicts() == want.to_dicts()


@needs_plugin
def test_plugin_reports_the_device():
    got = frame(10).lazy().select(pl.col("v").arrowmetal.device()).collect()
    assert am.device_name() in got.item()


@needs_plugin
def test_plugin_errors_are_polars_errors_with_arrowmetal_wording():
    df = frame(100)
    with pytest.raises(Exception, match="arrowmetal"):
        df.lazy().select(pl.col("v").arrowmetal.upper()).collect()


# =============================================================================================
# Scale: 50M rows, wall-clock bounds generous enough not to flake
# =============================================================================================

BIG = 50_000_000


@pytest.fixture(scope="module")
def big_series():
    return pl.Series("x", np.arange(BIG, dtype=np.int64))


def test_50m_import_is_zero_copy_and_fast(big_series):
    """A 400 MB column reaches the GPU without a copy. What it does cost is mapping the pages
    into the Metal address space -- about 8 ms on an M4 Max, an order of magnitude under a real
    copy of the same bytes."""
    am.from_polars(big_series)                       # warm the device
    t0 = time.perf_counter()
    m = am.from_polars(big_series)
    ms = (time.perf_counter() - t0) * 1000
    assert len(m) == BIG
    assert am.zero_copy(big_series)
    assert ms < 200, f"importing 50M rows took {ms:.1f} ms; a copy would be the only explanation"


def test_50m_export_is_effectively_free(big_series):
    m = am.from_polars(big_series)
    t0 = time.perf_counter()
    back = am.to_polars(m, "x")
    ms = (time.perf_counter() - t0) * 1000
    assert len(back) == BIG
    assert ms < 50, f"exporting 50M rows took {ms:.1f} ms"


def test_50m_sum_matches_polars(big_series):
    expected = BIG * (BIG - 1) // 2
    assert big_series.arrowmetal.sum() == expected
    assert big_series.sum() == expected


# ---------------------------------------------------------------------------------------------
# Regression tests from the pre-release integration review.
# ---------------------------------------------------------------------------------------------

def test_group_by_agg_alias_may_not_shadow_a_key():
    """An aggregate named after a key column used to overwrite the key silently.

    `agg` builds `out = self._key_frame()` and then assigns `out[alias]`, so an `alias` equal to a
    key name replaced the key values with the aggregate: the caller got a frame of the right shape
    whose "key" column was not keys at all, with no error anywhere.
    """
    df = pl.DataFrame({"k": [1, 1, 2], "v": [10, 20, 30]})
    with pytest.raises(am.ArrowMetalError, match="key column"):
        df.arrowmetal.group_by("k").agg(k=("v", "sum"))
    with pytest.raises(am.ArrowMetalError, match="key column"):
        df.arrowmetal.group_by("k").sum("k", "v")
    out = df.arrowmetal.group_by("k").agg(total=("v", "sum"))
    assert out.columns == ["k", "total"]
    assert out["k"].to_list() == [1, 2]
    assert out["total"].to_list() == [30, 30]


def test_zero_copy_report_answers_for_a_dataframe():
    """`am.zero_copy_report` documents "a Polars Series/DataFrame"; the DataFrame half raised
    AttributeError("'DataFrame' object has no attribute 'buffers'")."""
    df = pl.DataFrame({"a": np.arange(1 << 20, dtype=np.int64),
                       "b": np.arange(1 << 20, dtype=np.float64)})
    report = am.zero_copy_report(df)
    assert set(report) == {"a", "b"}
    for name, (src, dst, same) in report.items():
        assert src is not None and dst is not None, name
        assert same is (src == dst)
    assert am.zero_copy(df) is all(v[2] for v in report.values())
    src, dst, same = am.zero_copy_report(df["a"])       # the Series form is unchanged
    assert same is (src == dst)


def test_unique_is_first_seen_and_keeps_nulls():
    """docs/POLARS.md promised `unique()` was "ascending" and the docstring said "non-null". It is
    neither: `am_unique` answers in first-seen order and carries a null through."""
    s = pl.Series("x", [3, 1, None, 3, 2])
    assert s.arrowmetal.unique().to_list() == [3, 1, None, 2]
    assert pl.Series("s", ["z", "a", "z", "m"]).arrowmetal.unique().to_list() == ["z", "a", "m"]
    rng = np.random.default_rng(0)
    big = pl.Series("x", rng.integers(0, 10, 200_000))
    assert big.arrowmetal.unique().to_list() == list(dict.fromkeys(big.to_list()))


# ---------------------------------------------------------------------------------------------
# Tier 2, adversarial: the plugin against the tier-1 bridge and against native Polars.
#
# The eight tests above only ever ran once the Rust library was built, which nobody had done, so
# everything below is the first pass over the plugin with the dylib actually loaded.
# ---------------------------------------------------------------------------------------------

_DUNDER = {"add": "__add__", "sub": "__sub__", "mul": "__mul__", "truediv": "__truediv__"}


def t1_arith(series, op, value):
    """The tier-1 bridge's answer for the same scalar arithmetic, as a list."""
    out = getattr(am.from_polars(series), _DUNDER[op])(value)
    return am.to_polars(out, series.name).to_list()


def t2_arith(series, op, value):
    """The tier-2 plugin's answer, inside a lazy plan."""
    return (pl.DataFrame({series.name: series}).lazy()
            .select(getattr(pl.col(series.name).arrowmetal, op)(value))
            .collect().to_series().to_list())


@needs_plugin
def test_plugin_scalar_arithmetic_is_exact_past_2_to_the_53():
    """`_arith` sent `float(value)`, so an Int64 operand lost its low bits before the kernel saw
    it: `add(2**60 + 1)` added `2**60`. The scalar now crosses as exact decimal digits."""
    for dtype, value in [(pl.Int64, 2 ** 60 + 1), (pl.Int64, -(2 ** 60) - 1),
                         (pl.UInt64, 2 ** 63 + 5), (pl.UInt64, 2 ** 64 - 1)]:
        s = pl.Series("x", [10, 20], dtype=dtype)
        assert t2_arith(s, "add", value) == t1_arith(s, "add", value), (dtype, value)
        assert t2_arith(s, "add", value) == (s + value).to_list(), (dtype, value)
    # ... and the exact digits survive a multiplication too, where a float would round.
    s = pl.Series("x", [3], dtype=pl.Int64)
    assert t2_arith(s, "mul", 3 ** 33) == t1_arith(s, "mul", 3 ** 33) == [3 * 3 ** 33]


@needs_plugin
def test_plugin_refuses_a_scalar_the_column_type_cannot_hold():
    """`scalar_bytes` narrowed with Rust `as`, which saturates floats and truncates integers, so
    `add(1000)` on Int8 quietly became `add(127)`, `add(-1)` on UInt8 a no-op, and `add(1.5)` on
    Int64 `add(1)`. The tier-1 bridge packs the scalar with `struct.pack` at the column's own
    width and raises on all three; the plugin now answers the same way.

    "Integers wrap" in docs/POLARS.md is about the arithmetic, not the operand: 127 + 1 is still
    -128 on an Int8 column.
    """
    refused = [
        (pl.Series("x", [1, 2, 127], dtype=pl.Int8), "add", 1000),
        (pl.Series("x", [1, 2, 127], dtype=pl.Int8), "sub", -200),
        (pl.Series("x", [1, 2, 250], dtype=pl.UInt8), "add", -1),
        (pl.Series("x", [1, 2], dtype=pl.UInt32), "mul", -1),
        (pl.Series("x", [10, 20], dtype=pl.Int64), "add", 1.5),
        (pl.Series("x", [10, 20], dtype=pl.Int32), "mul", 3.9),
        (pl.Series("x", [10, 20], dtype=pl.Int64), "add", 2 ** 64),
        (pl.Series("x", [10, 20], dtype=pl.UInt64), "add", 2 ** 200),
    ]
    for s, op, value in refused:
        with pytest.raises(Exception):                       # struct.error
            t1_arith(s, op, value)
        with pytest.raises(Exception, match="arrowmetal"):
            t2_arith(s, op, value)

    # What the two tiers do accept, they agree on -- including the wrap the docs promise.
    assert t2_arith(pl.Series("x", [1, 2, 127], dtype=pl.Int8), "add", 1) == [2, 3, -128]
    assert t1_arith(pl.Series("x", [1, 2, 127], dtype=pl.Int8), "add", 1) == [2, 3, -128]
    assert t2_arith(pl.Series("x", [0], dtype=pl.UInt8), "sub", 1) == [255]
    # A float column takes an integer, and an f32 overflow becomes an infinity in both tiers.
    f32 = pl.Series("x", [1.0, 2.0], dtype=pl.Float32)
    assert t2_arith(f32, "mul", 1e300) == t1_arith(f32, "mul", 1e300) == [float("inf")] * 2
    f64 = pl.Series("x", [1.0], dtype=pl.Float64)
    assert t2_arith(f64, "add", 2 ** 70) == t1_arith(f64, "add", 2 ** 70) == [float(2 ** 70)]
    # A dtype with no scalar form is an error in both, not a wrong answer.
    with pytest.raises(Exception, match="arrowmetal"):
        t2_arith(pl.Series("x", [True, False]), "add", 1)


@needs_plugin
def test_plugin_null_handling_matches_native_polars():
    df = pl.DataFrame({"v": [1, None, 3, None, 5],
                       "f": [1.0, None, 3.0, 4.0, None],
                       "s": ["ab", None, "cd", "ab", None]})
    got = df.lazy().select(
        pl.col("v").arrowmetal.sum().alias("sum"),
        pl.col("v").arrowmetal.min().alias("min"),
        pl.col("v").arrowmetal.max().alias("max"),
        pl.col("f").arrowmetal.mean().alias("mean"),
    ).collect()
    assert got["sum"][0] == df["v"].sum()
    assert got["min"][0] == df["v"].min()
    assert got["max"][0] == df["v"].max()
    assert got["mean"][0] == pytest.approx(df["f"].mean())

    ew = df.lazy().select(
        pl.col("v").arrowmetal.add(1).alias("plus"),
        pl.col("s").arrowmetal.upper().alias("up"),
        pl.col("s").arrowmetal.contains("a").alias("has"),
        pl.col("v").arrowmetal.hash64().alias("h"),
    ).collect()
    assert ew["plus"].to_list() == (df["v"] + 1).to_list()
    assert ew["up"].to_list() == df["s"].str.to_uppercase().to_list()
    assert ew["has"].to_list() == df["s"].str.contains("a", literal=True).to_list()
    assert ew["h"].null_count() == df["v"].null_count()          # a null hashes to null

    total = df.lazy().select(
        pl.col("v").arrowmetal.filter_sum(pl.col("f") > 2.0)).collect().item()
    assert total == df.filter(pl.col("f") > 2.0)["v"].sum()

    grouped = (df.lazy()
               .select(pl.col("s").arrowmetal.group_by_sum(pl.col("v")).alias("g"))
               .collect().unnest("g").sort("s", nulls_last=True))
    want = df.group_by("s").agg(pl.col("v").sum()).sort("s", nulls_last=True)
    assert grouped.to_dicts() == want.to_dicts()                 # the null key is its own group


@needs_plugin
def test_plugin_sum_of_an_empty_or_all_null_column_is_null_where_polars_says_zero():
    """`am_reduce` has nothing to add up and answers null; `pl.Series.sum()` answers 0. Both tiers
    do this -- it is Arrow's rule, not a plugin bug -- but docs/POLARS.md said scalar reductions
    come back "exactly as `pl.Series.sum()` does", which is untrue in this one case."""
    for s in [pl.Series("v", [], dtype=pl.Int64), pl.Series("v", [None, None], dtype=pl.Int64)]:
        got = pl.DataFrame({"v": s}).lazy().select(pl.col("v").arrowmetal.sum()).collect().item()
        assert got is None
        assert s.sum() == 0                                      # what Polars answers
        assert s.arrowmetal.sum() is None                        # tier 1 agrees with tier 2
        # min/mean are null on both sides, so only sum diverges.
        assert pl.DataFrame({"v": s}).lazy().select(
            pl.col("v").arrowmetal.min()).collect().item() is None
        assert s.min() is None


@needs_plugin
def test_plugin_handles_a_sliced_and_a_chunked_series():
    """`to_metal` rechunks, and a Polars slice is an offset into a shared buffer. Both have to
    reach the kernel as the rows the plan actually selected."""
    df = pl.DataFrame({"v": list(range(20)), "s": [f"r{i}" for i in range(20)]})

    sl = df.slice(3)
    got = sl.lazy().select(
        pl.col("v").arrowmetal.sum().alias("sum"),
        pl.col("v").arrowmetal.max().alias("max"),
    ).collect()
    assert got["sum"][0] == sl["v"].sum() == sum(range(3, 20))
    assert got["max"][0] == 19
    assert (sl.lazy().select(pl.col("v").arrowmetal.add(1)).collect()["v"].to_list()
            == (sl["v"] + 1).to_list())
    assert (sl.lazy().select(pl.col("s").arrowmetal.upper()).collect()["s"].to_list()
            == sl["s"].str.to_uppercase().to_list())

    cat = pl.concat([df.slice(0, 7), df.slice(7)], rechunk=False)
    assert cat["v"].n_chunks() == 2
    assert cat.lazy().select(pl.col("v").arrowmetal.sum()).collect().item() == cat["v"].sum()
    assert (cat.lazy().select(pl.col("s").arrowmetal.upper()).collect()["s"].to_list()
            == cat["s"].str.to_uppercase().to_list())
    assert (sorted(cat.lazy().select(pl.col("v").arrowmetal.top_k(4)).collect()["v"].to_list())
            == sorted(cat["v"].top_k(4).to_list()))


@needs_plugin
def test_plugin_on_an_empty_frame():
    """Nothing here may crash or return the wrong length; `changes_length` expressions answer 0
    rows and `returns_scalar` ones answer one null."""
    e = pl.DataFrame({"v": pl.Series("v", [], dtype=pl.Int64),
                      "s": pl.Series("s", [], dtype=pl.String)})
    empty = e.lazy().select(
        pl.col("v").arrowmetal.add(1).alias("plus"),
        pl.col("v").arrowmetal.hash64().alias("h"),
        pl.col("s").arrowmetal.upper().alias("up"),
        pl.col("s").arrowmetal.contains("a").alias("has"),
    ).collect()
    assert empty.height == 0
    assert empty.schema["plus"] == pl.Int64 and empty.schema["h"] == pl.UInt64
    assert e.lazy().select(pl.col("v").arrowmetal.top_k(3)).collect().height == 0
    assert e.lazy().select(
        pl.col("s").arrowmetal.group_by_sum(pl.col("v")).alias("g")).collect().height == 0
    assert e.lazy().select(
        pl.col("v").arrowmetal.filter_sum(pl.col("s") == "a")).collect().item() is None
    assert e.lazy().select(pl.col("v").arrowmetal.mean()).collect().item() is None


@needs_plugin
def test_plugin_accepts_a_validity_bitmap_with_no_nulls():
    """An all-set validity buffer is the shape `fill_null` and `filter` leave behind: null_count 0
    but a bitmap still attached, which a kernel that only checks `null_count` could mis-read."""
    s = pl.Series("v", [1, 2, None, 4]).fill_null(9)
    arrow = s.to_arrow()
    chunk = arrow.combine_chunks() if isinstance(arrow, pa.ChunkedArray) else arrow
    assert chunk.buffers()[0] is not None and s.null_count() == 0
    df = pl.DataFrame({"v": s})
    assert df.lazy().select(pl.col("v").arrowmetal.sum()).collect().item() == s.sum() == 16
    assert (df.lazy().select(pl.col("v").arrowmetal.add(1)).collect()["v"].to_list()
            == (s + 1).to_list())
    assert df.lazy().select(pl.col("v").arrowmetal.hash64()).collect()["v"].null_count() == 0


@needs_plugin
@pytest.mark.parametrize("dtype_name, series", [
    ("Boolean", pl.Series("x", [True, False])),
    ("Date", pl.Series("x", [1, 2], dtype=pl.Int32).cast(pl.Date)),
    ("Datetime", pl.Series("x", [1, 2], dtype=pl.Int64).cast(pl.Datetime("us"))),
    ("Duration", pl.Series("x", [1, 2], dtype=pl.Int64).cast(pl.Duration("us"))),
    ("Time", pl.Series("x", [1, 2], dtype=pl.Int64).cast(pl.Time)),
    ("String", pl.Series("x", ["a", "b"])),
    ("Binary", pl.Series("x", [b"a", b"b"])),
    ("Categorical", pl.Series("x", ["a", "b"], dtype=pl.Categorical)),
    ("List", pl.Series("x", [[1, 2], [3]])),
    ("Struct", pl.Series("x", [{"a": 1}, {"a": 2}])),
    ("Decimal", pl.Series("x", ["1.5", "2.5"]).cast(pl.Decimal(10, 2))),
    ("Null", pl.Series("x", [None, None], dtype=pl.Null)),
])
def test_plugin_refuses_an_unsupported_dtype_with_an_error_not_a_panic(dtype_name, series):
    """The numeric tier-2 expressions take the eight integer widths and Float32/64 and nothing
    else -- narrower than the dtypes the *bridge* round-trips, which is what docs/POLARS.md's
    "Types" paragraph is about. What matters is that every refusal is a Polars error carrying
    ArrowMetal's wording: a Rust panic crossing pyo3 would be a finding.
    """
    df = pl.DataFrame({"x": series})
    exprs = [pl.col("x").arrowmetal.sum(), pl.col("x").arrowmetal.min(),
             pl.col("x").arrowmetal.top_k(1), pl.col("x").arrowmetal.add(1),
             pl.col("x").arrowmetal.group_by_sum(pl.col("x")).alias("g")]
    if dtype_name != "String":
        exprs.append(pl.col("x").arrowmetal.upper())
    for expr in exprs:
        with pytest.raises(Exception, match="arrowmetal") as exc:
            df.lazy().select(expr).collect()
        assert "panicked" not in str(exc.value), (dtype_name, str(exc.value)[:200])
        assert type(exc.value).__name__ != "PanicException", dtype_name

    # `hash64` is the one that reaches further: it takes every fixed-width dtype, temporal and
    # boolean included, and refuses the rest the same way.
    fixed_width = dtype_name in {"Boolean", "Date", "Datetime", "Duration", "Time", "String"}
    if fixed_width:
        assert df.lazy().select(
            pl.col("x").arrowmetal.hash64()).collect()["x"].dtype == pl.UInt64
    else:
        with pytest.raises(Exception, match="arrowmetal"):
            df.lazy().select(pl.col("x").arrowmetal.hash64()).collect()


@needs_plugin
def test_plugin_runs_inside_group_by_agg_and_a_streaming_plan():
    """docs/POLARS.md claims both: the expression is called once per group inside `.agg(...)`, and
    it survives `collect(engine="streaming")` in a `with_columns`."""
    df = pl.DataFrame({"k": [1, 1, 2, 2, 2],
                       "v": [1, 2, 3, 4, 5],
                       "s": ["a", "b", "c", "d", "e"]})

    agg = (df.lazy().group_by("k")
           .agg(pl.col("v").arrowmetal.sum().alias("total"),
                pl.col("v").arrowmetal.min().alias("lo"),
                pl.col("v").arrowmetal.max().alias("hi"))
           .sort("k").collect())
    want = (df.group_by("k")
            .agg(pl.col("v").sum().alias("total"),
                 pl.col("v").min().alias("lo"),
                 pl.col("v").max().alias("hi"))
            .sort("k"))
    assert agg.to_dicts() == want.to_dicts()

    streamed = (df.lazy()
                .with_columns(pl.col("v").arrowmetal.add(10).alias("plus"),
                              pl.col("s").arrowmetal.upper().alias("up"))
                .collect(engine="streaming"))
    assert streamed["plus"].to_list() == (df["v"] + 10).to_list()
    assert streamed["up"].to_list() == df["s"].str.to_uppercase().to_list()

    assert df.lazy().select(
        pl.col("v").arrowmetal.sum()).collect(engine="streaming").item() == df["v"].sum()
    assert (df.lazy().filter(pl.col("k") == 2)
            .with_columns(pl.col("s").arrowmetal.upper().alias("up"))
            .collect(engine="streaming")["up"].to_list() == ["C", "D", "E"])
    assert (df.lazy().group_by("k").agg(pl.col("v").arrowmetal.sum().alias("total"))
            .sort("k").collect(engine="streaming").to_dicts() == want.select("k", "total").to_dicts())
