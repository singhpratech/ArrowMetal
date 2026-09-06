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
