"""The engine conformance grid (engine_conformance.py) under pytest, and the regression tests for the
defects it found.

    PYTHONPATH=python python -m pytest python/tests/test_engine_conformance.py -q

The grid runs here without its 100,000-row tables; `engine_report.py` runs all of it and writes the
per-shape table.
"""
import math
import os
import struct
import sys

import numpy as np
import pytest

pl = pytest.importorskip("polars")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_conformance as ec                                      # noqa: E402
import engine_report                                                 # noqa: E402

from arrowmetal import polars_engine as pe                           # noqa: E402


def _metal(lf):
    eng = pe.MetalEngine(shapes="all", min_rows=0, raise_on_fail=True)
    out = lf.collect(engine=eng)
    assert eng.last_report.taken, str(eng.last_report)
    return out, eng


def _bits(series):
    if series.dtype not in (pl.Float32, pl.Float64):
        return series.to_list()
    a = series.fill_null(0.0).to_numpy()
    view = a.view(np.uint64 if series.dtype == pl.Float64 else np.uint32)
    return [None if m else int(v) for v, m in zip(view, series.is_null().to_list())]


def _same_bits(got, want):
    assert got.schema == want.schema
    assert got.columns == want.columns
    for c in want.columns:
        assert _bits(got[c]) == _bits(want[c]), (c, got[c].to_list(), want[c].to_list())


# ---------------------------------------------------------------------------------------------------
# the grid


def _grid(name):
    import arrowmetal as am
    mod = dict(engine_report._engines(name))[name]
    router = am.get_router()
    am.set_router("gpu")
    try:
        res = engine_report.run_engine(name, mod, quick=True, quiet=True)
    finally:
        am.set_router(router)
    if res is None:
        pytest.skip("the DuckDB rewrite extension is not built (duckdb-extension/build_rewrite.sh)")
    return res


def test_polars_grid_has_no_unclassified_mismatch():
    shapes, total, documented, unclassified, not_taken, _secs = _grid("polars")
    assert total.cases > 5000 and total.passed > 0.9 * total.cases, engine_report.summary_line("polars", total)
    assert not unclassified, [(c["shape"], c["dtype"], ec.shape_id(c["ds"]), d) for c, d in unclassified[:10]]


def test_duckdb_grid_has_no_unclassified_mismatch():
    pytest.importorskip("duckdb")
    shapes, total, documented, unclassified, not_taken, _secs = _grid("duckdb")
    assert total.cases > 5000 and total.passed > 0.5 * total.cases, engine_report.summary_line("duckdb", total)
    assert not unclassified, [(c["shape"], c["dtype"], ec.shape_id(c["ds"]), d) for c, d in unclassified[:10]]


def test_every_documented_divergence_points_at_a_line_that_exists():
    import engine_polars_grid as pg
    mods = [pg]
    try:
        import engine_duckdb_grid as dg
        mods.append(dg)
    except ImportError:
        pass
    for mod in mods:
        for d in mod.DIVERGENCES:
            assert "anchor not found" not in d.doc_line(), (d.id, d.doc, d.anchor)


# ---------------------------------------------------------------------------------------------------
# regressions: defects the grid found, fixed in the engine or in ArrowMetal


_V = 21038.0                        # 21038 / 3 and 21038 * (1 / 3) differ in the last bit
_NAN = struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000001))[0]   # a NaN with a payload


@pytest.mark.parametrize("n", [1, 2, 5])
def test_scalar_division_and_minus_one_follow_polars_at_every_length(n):
    """Polars divides a column by a scalar element-wise when the column holds one row and as a
    multiply by the reciprocal otherwise; likewise a float multiplied by -1 is a multiply over one
    row (the NaN keeps its sign) and a negation over more. The engine picks by the row count of the
    node's input: from the scan when it is known, by counting the input when the plan runs when a
    filter or a join decides it."""
    df = pl.DataFrame({"x": [_V] * (n - 1) + [_NAN], "k": list(range(n)),
                       "f": pl.Series([0.7520270347595215] * n, dtype=pl.Float32)})
    x = pl.col("x")
    plans = [
        df.lazy().select((x / 3).alias("d"), (x * -1.0).alias("m"), (x / -1.0).alias("n"),
                         (pl.col("f") / 3).alias("f3"), (-1.0 * pl.col("f")).alias("fm")),
        df.lazy().filter(pl.col("k") < 1).with_columns((x / 3).alias("d"), (x * -1.0).alias("m")),
        df.lazy().filter(pl.col("k") >= n - 1).with_columns((x * -1.0).alias("m")),
        df.lazy().filter(pl.col("k") == 0).select((x / 3).sum().alias("s")),
        df.lazy().filter(pl.col("k") == 0).filter(x / 3 == _V / 3),
        df.lazy().head(1).with_columns((x / 3).alias("d")),
    ]
    for lf in plans:
        got, _eng = _metal(lf)
        _same_bits(got, lf.collect())


def test_a_row_count_known_only_at_run_time_is_counted_then():
    df = pl.DataFrame({"x": [_V, _V, 1.0], "k": [0, 1, 2]})
    lf = df.lazy().filter(pl.col("k") == 0).with_columns((pl.col("x") / 3).alias("d"))
    got, eng = _metal(lf)
    assert "@arrowmetal_one_row_0@" in eng.last_report.taken[0]["plan"]
    assert got["d"][0] == _V / 3 != _V * (1 / 3)
    lf = df.lazy().with_columns((pl.col("x") / 3).alias("d"))     # three rows, known from the scan
    got, eng = _metal(lf)
    assert "@arrowmetal_one_row" not in eng.last_report.taken[0]["plan"]
    assert got["d"][0] == _V * (1 / 3)


_TEMPORAL = {
    "date": pl.Date, "datetime_ms": pl.Datetime("ms"), "datetime_ns": pl.Datetime("ns"),
    "datetime_tz": pl.Datetime("us", "America/New_York"), "duration": pl.Duration("us"),
    "time": pl.Time,
}


@pytest.mark.parametrize("dtype", list(_TEMPORAL))
def test_a_nullable_temporal_sort_key_with_nulls_first_runs_on_metal(dtype):
    """The validity key that puts nulls first used to be an expression over the column, which
    ArrowMetal's expressions do not read for a temporal type, so the plan was rejected and the sort
    left to Polars. The key now comes from the scan."""
    rng = np.random.default_rng(4)
    raw = pl.Series(rng.integers(0, 10**6, 200)).cast(pl.Int64)
    raw = raw.scatter(rng.integers(0, 200, 30), None)
    col = raw.cast(pl.Int32).cast(pl.Date) if dtype == "date" else (
        (raw * 1000).cast(pl.Time) if dtype == "time" else raw.cast(_TEMPORAL[dtype]))
    df = pl.DataFrame({"t": col, "id": np.arange(200)})
    for lf in (df.lazy().sort("t"), df.lazy().sort("t", descending=True),
               df.lazy().sort(["t", "id"], descending=[True, False]),
               df.lazy().filter(pl.col("id") > 20).sort("t").head(15)):
        got, eng = _metal(lf)
        want = lf.collect()
        assert got["t"].to_list() == want["t"].to_list()
        assert "__arrowmetal_valid" in eng.last_report.taken[0]["plan"]


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64])
def test_min_and_max_over_both_zeros_are_polars_signed_zeros(dtype):
    """Polars orders -0.0 below 0.0 in min and max; ArrowMetal's answer depended on which zero it met
    first (whole frame) or was always 0.0 (per group)."""
    def sign(v):
        return math.copysign(1.0, v)
    for vals in ([0.0, -0.0], [-0.0, 0.0], [1.0, 0.0, -0.0, 0.0], [-0.0] * 3, [0.0] * 3):
        df = pl.DataFrame({"x": pl.Series(vals, dtype=dtype), "k": [1] * len(vals)})
        plans = [df.lazy().select(pl.col("x").min().alias("mn"), pl.col("x").max().alias("mx"))]
        if dtype == pl.Float32:        # a Float64 group-by min/max stays with Polars
            plans.append(df.lazy().group_by("k").agg(pl.col("x").min().alias("mn"),
                                                     pl.col("x").max().alias("mx")))
        for lf in plans:
            got, _eng = _metal(lf)
            want = lf.collect()
            for c in ("mn", "mx"):
                assert sign(got[c][0]) == sign(want[c][0]), (vals, c, got[c][0], want[c][0])


def test_a_u64_literal_above_int64_max_reaches_the_gpu():
    """ArrowMetal's expression parser read every integer literal as an Int64, so `(u64
    18446744073709551615)` was rejected and the plan left to Polars."""
    big = [2**64 - 1, 2**63, 2**63 - 1, 0, 1, None, 2**64 - 2]
    df = pl.DataFrame({"x": pl.Series(big, dtype=pl.UInt64)})
    for lf in (df.lazy().filter(pl.col("x").is_in(pl.lit([2**64 - 1, 2**63], dtype=pl.List(pl.UInt64)))),
               df.lazy().filter(pl.col("x") > pl.lit(2**63, dtype=pl.UInt64)),
               df.lazy().select((pl.col("x") == pl.lit(2**64 - 1, dtype=pl.UInt64)).alias("eq"),
                                pl.col("x").fill_null(pl.lit(2**64 - 1, dtype=pl.UInt64)).alias("f"))):
        got, _eng = _metal(lf)
        _same_bits(got, lf.collect())


def test_a_float32_mean_is_accumulated_in_float64_like_polars():
    """Polars averages a Float32 column in Float64 and rounds once; ArrowMetal's Float32 mean added in
    Float32, and over 100,000 rows of mixed magnitudes differed from Polars' by up to 7.7e-7 of the
    answer. The engine now averages the column cast to Float64."""
    rng = np.random.default_rng(9)
    n = 100_000
    x = (rng.standard_normal(n) * rng.choice([1.0, 1e6, 1e-6], n)).astype(np.float32)
    df = pl.DataFrame({"x": x, "k": rng.integers(0, 9, n).astype(np.int32)})
    for lf in (df.lazy().group_by("k").agg(pl.col("x").mean()),
               df.lazy().select(pl.col("x").mean())):
        got, _eng = _metal(lf)
        want = lf.collect()
        if "k" in want.columns:
            got, want = got.sort("k"), want.sort("k")
        _same_bits(got, want)
