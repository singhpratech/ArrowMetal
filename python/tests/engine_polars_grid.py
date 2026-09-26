"""The Polars half of the engine conformance grid (see engine_conformance.py).

Each case builds a frame, a LazyFrame over it for one shape, collects it on Polars and through
`MetalEngine(shapes="all", min_rows=0)`, and compares the frames: the schema first, then the values,
exactly (floats bit for bit, every NaN equal to every NaN), in order where the plan fixes one and as
a multiset of rows where it does not. A sort is compared on its keys row by row and on its rows as a
multiset, since rows that tie on every key come back in an unspecified order from both engines; a
sort with a limit on its keys only, since which of the rows tied at the cut are kept is unspecified.
"""
import math
import os
import sys

import numpy as np
import pyarrow as pa
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_conformance as ec                                      # noqa: E402

from arrowmetal import polars_engine as pe                           # noqa: E402
from engine_conformance import PASS, NOT_TAKEN                         # noqa: E402

ENGINE = "polars"

# dtype name -> (test_differential generator name, Polars dtype class for the capability checks)
DTYPES = {
    "int8": "int8", "int16": "int16", "int32": "int32", "int64": "int64",
    "uint8": "uint8", "uint16": "uint16", "uint32": "uint32", "uint64": "uint64",
    "float32": "float32", "float64": "float64", "bool": "bool", "string": "utf8",
    "date": "date32", "datetime_ms": "ts_ms", "datetime_us": "ts_us", "datetime_ns": "ts_ns",
    "datetime_us_tz": "ts_us_tz", "duration_ms": "duration_ms", "duration_us": "duration_us",
    "duration_ns": "duration_ns", "time": "time64_ns",
}
INT = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
SIGNED = INT[:4]
FLT = ["float32", "float64"]
NUM = INT + FLT
TEMPORAL = [d for d in DTYPES if d.startswith(("date", "datetime", "duration", "time"))]
ALL = list(DTYPES)
KEYABLE = [d for d in ALL if d not in FLT]

PL = {"int8": pl.Int8, "int16": pl.Int16, "int32": pl.Int32, "int64": pl.Int64,
      "uint8": pl.UInt8, "uint16": pl.UInt16, "uint32": pl.UInt32, "uint64": pl.UInt64,
      "float32": pl.Float32, "float64": pl.Float64}


# ==================================================================================================
# frames

_frames = {}


def frame(dtype, ds):
    """The left frame of a case: x (the column under test), xk (x resampled from 12 of its rows, a
    key with repeats), xj (x resampled from a quarter of its rows, at least 12: the join key, whose
    matches grow with the row count instead of its square), k (Int32 0..8), v (Int64), id (the row
    number)."""
    key = (dtype, ds)
    hit = _frames.get(key)
    if hit is not None:
        return hit
    n = ds.size
    x = ec._source_array(DTYPES[dtype], ds, seed=1)
    xk = ec._low_cardinality(x, n, seed=2)
    xj = ec._low_cardinality(x, n, seed=4, pool=_join_pool(n))
    k, v, ids = ec._helper_columns(n, seed=3)
    df = pl.from_arrow(pa.table({"x": x, "xk": xk, "xj": xj, "k": k, "v": v, "id": ids}))
    if len(_frames) > 64:
        _frames.clear()
    _frames[key] = df
    return df


def right_frame(dtype, ds):
    """The right side of a join: xj drawn from the rows the left side's xj was drawn from plus 4
    more, so some keys match several times and some not at all, and w (Int64)."""
    n = ds.size
    x = ec._source_array(DTYPES[dtype], ds, seed=1)
    pool = min(len(x), _join_pool(n) + 4)
    m = max(n // 2, 1) if n else 0
    if n:
        # A type with few distinct values (Boolean, 8-bit integers, the string pool) would make an
        # n / 2-row right side join to about n^2 / (2 * distinct) rows: keep the join about 4n.
        distinct = len(x.slice(0, pool).drop_null().unique())
        m = min(m, max(8, 4 * distinct))
    rng = np.random.default_rng([5, n])
    xj = x.take(pa.array(rng.integers(0, pool, m), type=pa.int64())) if n else x.slice(0, 0)
    w = pa.array(rng.integers(-1000, 1000, m), type=pa.int64())
    return pl.from_arrow(pa.table({"xj": xj, "w": w}))


def _join_pool(n):
    return max(12, n // 4)


def _lit(dtype, value):
    return pl.lit(value, dtype=PL[dtype])


def _some_values(df, col, count):
    """Up to `count` distinct non-null values of `col`, in first-seen order."""
    out = []
    for value in df.get_column(col).to_list():
        if value is None:
            continue
        if isinstance(value, float) and value != value:
            continue
        if value not in out:
            out.append(value)
        if len(out) == count:
            break
    return out


# ==================================================================================================
# shapes
#
# A shape is (name, family, dtypes, build). `build(df, dtype, ds)` returns (lf, spec): `spec` says
# how to compare -- "order": "multiset" | "exact" | ("keys", [cols]) | ("keys_only", [cols]), and
# "float_agg": the output columns that are float aggregates (sums and means, whose summation order
# the two engines do not share), "agg_col": the input column those aggregate.


def _filter_cmp(df, dtype, ds):
    x = pl.col("x")
    if dtype == "bool":
        pred = x
    elif dtype == "string":
        pred = x == "apple"
    else:
        pred = x > _lit(dtype, 1)
    return df.lazy().filter(pred), {"order": "exact"}


def _filter_null(df, dtype, ds):
    return df.lazy().filter(pl.col("x").is_not_null()), {"order": "exact"}


def _filter_other(df, dtype, ds):
    return df.lazy().filter(pl.col("k") > 3), {"order": "exact"}


def _filter_is_in(df, dtype, ds):
    values = _some_values(df, "x", 3) or ([1] if dtype in NUM else ["apple"])
    if dtype in FLT:
        values = values + [float("nan")]
    if dtype == "string":
        return df.lazy().filter(pl.col("x").is_in(values)), {"order": "exact"}
    return df.lazy().filter(pl.col("x").is_in(pl.lit(values, dtype=pl.List(PL[dtype])))), {"order": "exact"}


def _select_arith(df, dtype, ds):
    x = pl.col("x")
    return df.lazy().select((x + x).alias("add"), (x - _lit(dtype, 1)).alias("sub"),
                            (x * _lit(dtype, 3)).alias("mul"), pl.col("id")), {"order": "exact"}


def _select_div(df, dtype, ds):
    x = pl.col("x")
    return df.lazy().select((x / 3).alias("div_lit"), (x / x).alias("div_col"),
                            pl.col("id")), {"order": "exact"}


def _select_cmp(df, dtype, ds):
    x = pl.col("x")
    lit = pl.lit(True) if dtype == "bool" else _lit(dtype, 1)
    return df.lazy().select((x > lit).alias("gt"), (x <= lit).alias("le"), (x == x).alias("eq"),
                            (x != lit).alias("ne"), pl.col("id")), {"order": "exact"}


def _select_null(df, dtype, ds):
    x = pl.col("x")
    outs = [x.is_null().alias("is_null"), x.is_not_null().alias("is_not_null"), pl.col("id")]
    if dtype in NUM:
        outs.append(x.fill_null(_lit(dtype, 7)).alias("filled"))
    elif dtype == "bool":
        outs.append(x.fill_null(True).alias("filled"))
    return df.lazy().select(outs), {"order": "exact"}


def _select_when(df, dtype, ds):
    x = pl.col("x")
    if dtype == "bool":
        e = pl.when(x).then(pl.col("k")).otherwise(pl.lit(-1, dtype=pl.Int32))
    else:
        e = pl.when(x > _lit(dtype, 1)).then(x).otherwise(_lit(dtype, 0))
    return df.lazy().select(e.alias("when"), pl.col("id")), {"order": "exact"}


_WIDER = {"int8": pl.Int64, "int16": pl.Int64, "int32": pl.Int64, "int64": pl.Float64,
          "uint8": pl.Int16, "uint16": pl.Int32, "uint32": pl.Int64, "uint64": pl.Float64,
          "float32": pl.Float64, "bool": pl.Int32}


def _select_cast(df, dtype, ds):
    return df.lazy().select(pl.col("x").cast(_WIDER[dtype]).alias("cast"), pl.col("id")), {"order": "exact"}


def _select_logic(df, dtype, ds):
    x = pl.col("x")
    if dtype == "bool":
        o = pl.col("k") > 3
        outs = [(x & o).alias("and"), (x | o).alias("or"), (x ^ o).alias("xor"), (~x).alias("not")]
    else:
        outs = [(x & x).alias("and"), (x | _lit(dtype, 6)).alias("or"), (x ^ x).alias("xor"),
                (~x).alias("not")]
    return df.lazy().select(outs + [pl.col("id")]), {"order": "exact"}


def _select_str(df, dtype, ds):
    x = pl.col("x")
    return df.lazy().select((x == "apple").alias("eq"), (x != "apple").alias("ne"),
                            x.str.starts_with("a").alias("starts"),
                            x.str.contains("pp", literal=True).alias("contains"),
                            pl.col("id")), {"order": "exact"}


def _hstack(df, dtype, ds):
    extra = [(pl.col("k") * 2).alias("k2")]
    if dtype in NUM:
        extra.append((pl.col("x") > _lit(dtype, 1)).alias("xgt"))
    return df.lazy().with_columns(extra), {"order": "exact"}


def _slice(df, dtype, ds):
    return df.lazy().filter(pl.col("k").is_not_null()).slice(2, 20), {"order": "exact"}


def _sort(descending=False, nulls_last=False):
    def build(df, dtype, ds):
        return (df.lazy().sort("x", descending=descending, nulls_last=nulls_last),
                {"order": ("keys", ["x"])})
    return build


def _sort_multi(df, dtype, ds):
    return df.lazy().sort(["x", "id"], descending=[True, False]), {"order": "exact"}


def _top_k(descending):
    def build(df, dtype, ds):
        return df.lazy().sort("x", descending=descending).head(10), {"order": ("keys_only", ["x"])}
    return build


_AGGS = ["sum", "min", "max", "mean", "count"]


def _agg_expr(col, agg):
    return getattr(pl.col(col), agg)().alias(agg)


def _group_by_key(agg):
    def build(df, dtype, ds):
        e = pl.len().alias("len") if agg == "len" else _agg_expr("v", agg)
        return df.lazy().group_by("xk").agg(e), {"order": "multiset", "agg_col": "v",
                                                 "float_agg": ["mean"] if agg == "mean" else []}
    return build


def _group_by_value(agg):
    def build(df, dtype, ds):
        return (df.lazy().group_by("k").agg(_agg_expr("x", agg)),
                {"order": "multiset", "agg_col": "x", "float_agg": [agg] if agg == "mean" or
                 (agg == "sum" and dtype in FLT) else []})
    return build


def _aggregate(agg):
    def build(df, dtype, ds):
        return (df.lazy().select(_agg_expr("x", agg)),
                {"order": "exact", "agg_col": "x", "float_agg": [agg] if agg == "mean" or
                 (agg == "sum" and dtype in FLT) else []})
    return build


def _join(how):
    def build(df, dtype, ds):
        right = right_frame(dtype, ds)
        return (df.lazy().select("xj", "id", "v").join(right.lazy(), on="xj", how=how),
                {"order": "multiset"})
    return build


def _unique_subset(df, dtype, ds):
    return df.lazy().unique(subset=["xk"], keep="first"), {"order": "multiset"}


def _unique_all(df, dtype, ds):
    return df.lazy().select("xk", "k").unique(), {"order": "multiset"}


SHAPES = [
    ("filter_cmp", "filter", NUM + ["bool", "string"], _filter_cmp),
    ("filter_is_not_null", "filter", NUM + ["bool", "string"], _filter_null),
    ("filter_other_column", "filter", ALL, _filter_other),
    ("filter_is_in", "filter", NUM + ["string"], _filter_is_in),
    ("select_arith", "select", NUM, _select_arith),
    ("select_div", "select", NUM, _select_div),
    ("select_cmp", "select", NUM + ["bool"], _select_cmp),
    ("select_null", "select", NUM + ["bool", "string"], _select_null),
    ("select_when", "select", NUM + ["bool"], _select_when),
    ("select_cast", "select", [d for d in NUM + ["bool"] if d != "float64"], _select_cast),
    ("select_logic", "select", INT + ["bool"], _select_logic),
    ("select_str", "select", ["string"], _select_str),
    ("with_columns", "hstack", ALL, _hstack),
    ("slice", "slice", ALL, _slice),
    ("sort_asc", "sort", ALL, _sort()),
    ("sort_desc", "sort", ALL, _sort(descending=True)),
    ("sort_nulls_last", "sort", ALL, _sort(nulls_last=True)),
    ("sort_desc_nulls_last", "sort", ALL, _sort(descending=True, nulls_last=True)),
    ("sort_two_keys", "sort", ALL, _sort_multi),
    ("top_k_asc", "sort+limit", ALL, _top_k(False)),
    ("top_k_desc", "sort+limit", ALL, _top_k(True)),
]
for _agg in _AGGS + ["len"]:
    SHAPES.append((f"group_by_key_{_agg}", "group_by", KEYABLE, _group_by_key(_agg)))
for _agg in _AGGS:
    _types = {"sum": NUM + ["bool"], "mean": NUM, "count": NUM + ["bool"],
              "min": [d for d in NUM if d != "float64"], "max": [d for d in NUM if d != "float64"]}[_agg]
    SHAPES.append((f"group_by_value_{_agg}", "group_by", _types, _group_by_value(_agg)))
for _agg in _AGGS:
    _types = {"sum": NUM + ["bool"], "mean": NUM, "count": NUM + ["bool"], "min": NUM, "max": NUM}[_agg]
    SHAPES.append((f"aggregate_{_agg}", "aggregate", _types, _aggregate(_agg)))
for _how in ("inner", "left", "semi", "anti"):
    SHAPES.append((f"join_{_how}", "join", KEYABLE, _join(_how)))
SHAPES += [
    ("unique_subset", "unique", KEYABLE, _unique_subset),
    ("unique_all", "unique", KEYABLE, _unique_all),
]
SHAPE_BY_NAME = {s[0]: s for s in SHAPES}


# ==================================================================================================
# comparison


def _canon(v):
    """A sortable, exact stand-in for one value: nulls first, NaN after every number and equal to
    every NaN, -0.0 apart from 0.0."""
    if v is None:
        return (0,)
    if isinstance(v, float):
        if v != v:
            return (2,)
        return (1, v, math.copysign(1.0, v))
    return (1, v)


def _rows(df, cols, zero_equal=False):
    canon = _canon_zero_equal if zero_equal else _canon
    return [tuple(canon(v) for v in row) for row in df.select(cols).iter_rows()]


def _canon_zero_equal(v):
    """`_canon` with -0.0 equal to 0.0: Polars' sort order, where the two zeros tie."""
    if isinstance(v, float) and v == 0:
        v = 0.0
    return _canon(v)


def _close(a, b, rel):
    """Float-aggregate cell equality within `rel`: exact canon equality, or two finite numbers whose
    relative difference is at most `rel`."""
    if a == b:
        return True
    if len(a) < 2 or len(b) < 2:
        return False
    x, y = a[1], b[1]
    if not (isinstance(x, float) and isinstance(y, float)) or math.isinf(x) or math.isinf(y):
        return False
    return abs(x - y) <= rel * max(abs(x), abs(y), 1e-300)


def _first_difference(got, want, names):
    for i, (g, w) in enumerate(zip(got, want)):
        if g != w:
            cols = [n for n, a, b in zip(names, g, w) if a != b]
            return f"row {i}: {cols} Metal {_show(g)} vs Polars {_show(w)}"
    return f"row counts: Metal {len(got)} vs Polars {len(want)}"


def _show(row):
    return tuple(None if c == (0,) else (float("nan") if c == (2,) else c[1]) for c in row)


_CANON_NAN = {pl.Float32: 0x7FC00000, pl.Float64: 0x7FF8000000000000}


def _canon_frame(df, cols, zero_equal=False):
    """`df[cols]` with every float column replaced by its bit pattern (one pattern for every NaN;
    with `zero_equal`, -0.0 as 0.0), so that Polars' own equality and sort are exact."""
    out = {}
    for c in cols:
        s = df.get_column(c)
        if s.dtype in (pl.Float32, pl.Float64):
            wide = s.dtype == pl.Float64
            arr = s.to_arrow()
            if isinstance(arr, pa.ChunkedArray):
                arr = arr.combine_chunks()
            vals = np.asarray(arr.fill_null(0.0).to_numpy(zero_copy_only=False))
            if zero_equal:
                vals = np.where(vals == 0, 0.0, vals).astype(vals.dtype)
            view = vals.view(np.uint64 if wide else np.uint32).copy()
            view[np.isnan(vals)] = _CANON_NAN[s.dtype]
            mask = s.is_null().to_numpy()
            out[c] = pl.Series(c, view, dtype=pl.UInt64 if wide else pl.UInt32).scatter(
                np.nonzero(mask)[0], None) if mask.any() else pl.Series(c, view)
        else:
            out[c] = s
    return pl.DataFrame(out) if out else pl.DataFrame()


def _fast_equal(got, want, spec):
    """True when the frames certainly agree under `spec`; False when the slow path must look."""
    names = list(want.columns)
    if got.height != want.height:
        return False
    order = spec.get("order", "multiset")
    if isinstance(order, tuple):
        g, w = _canon_frame(got, order[1], True), _canon_frame(want, order[1], True)
        if not g.equals(w, null_equal=True):
            return False
        if order[0] == "keys_only":
            return True
        order = "multiset"
    g, w = _canon_frame(got, names), _canon_frame(want, names)
    if order == "multiset" and g.width and g.height:
        g = g.sort(pl.all(), nulls_last=True)
        w = w.sort(pl.all(), nulls_last=True)
    return g.equals(w, null_equal=True)


def compare(got, want, spec):
    """None when the frames agree exactly; otherwise (kind, detail, max_rel) where kind is "schema",
    "values" or "float_agg" (the only differences are in float-aggregate cells, and max_rel is the
    largest relative difference among them)."""
    if got.schema != want.schema:
        return ("schema", f"schema: Metal {dict(got.schema)} vs Polars {dict(want.schema)}", None)
    if _fast_equal(got, want, spec):
        return None
    names = list(want.columns)
    order = spec.get("order", "multiset")
    fl = [c for c in spec.get("float_agg", []) if c in names]
    if isinstance(order, tuple) and order[0] == "keys_only":
        g, w = _rows(got, order[1], True), _rows(want, order[1], True)
        if g != w:
            return ("values", _first_difference(g, w, order[1]), None)
        return None
    if isinstance(order, tuple) and order[0] == "keys":
        g, w = _rows(got, order[1], True), _rows(want, order[1], True)
        if g != w:
            return ("values", _first_difference(g, w, order[1]), None)
        order = "multiset"
    # Float-aggregate columns go last in the sort key, so rows line up by their exact columns.
    cols = [c for c in names if c not in fl] + fl
    g, w = _rows(got, cols), _rows(want, cols)
    if order == "multiset":
        g, w = sorted(g), sorted(w)
    if g == w:
        return None
    if len(g) != len(w):
        return ("values", _first_difference(g, w, cols), None)
    nexact = len(cols) - len(fl)
    max_rel = max_abs = 0.0
    for a, b in zip(g, w):
        if a[:nexact] != b[:nexact]:
            return ("values", _first_difference(g, w, cols), None)
        for ca, cb in zip(a[nexact:], b[nexact:]):
            if ca == cb:
                continue
            if not _close(ca, cb, 1.0):
                return ("values", _first_difference(g, w, cols), None)
            x, y = ca[1], cb[1]
            max_rel = max(max_rel, abs(x - y) / max(abs(x), abs(y), 1e-300))
            max_abs = max(max_abs, abs(x - y))
    return ("float_agg", _first_difference(g, w, cols), max_rel, max_abs)


# ==================================================================================================
# running a case


def _unit_magnitude(df, col, dtype, shape):
    """u * sum(|x|) over the finite values of the aggregated column, u the unit roundoff of the
    type the sum is accumulated in (Float32 for a Float32 sum; Float64 for every mean and every
    other sum). The error bound of a sum of n values is (n - 1) u sum(|x|)."""
    x = df.get_column(col).cast(pl.Float64).to_numpy()
    x = x[np.isfinite(x)]
    unit = 2.0 ** -24 if (dtype == "float32" and shape.endswith("_sum")) else 2.0 ** -53
    return unit * max(float(np.abs(x).sum()), 1e-300)


def metal_engine():
    return pe.MetalEngine(shapes="all", min_rows=0)


def cases(quick=False, shapes=None, dtypes=None):
    for ds in ec.data_shapes(quick):
        for name, family, types, _build in SHAPES:
            if shapes and name not in shapes:
                continue
            for dtype in types:
                if dtypes and dtype not in dtypes:
                    continue
                if ds.flavor == "special" and dtype in TEMPORAL + ["string", "bool"]:
                    # test_differential's special flavour is a numeric one; temporal "special" draws
                    # only move the dates, which the random flavour already covers.
                    continue
                yield {"engine": ENGINE, "shape": name, "family": family, "dtype": dtype, "ds": ds}


def run_case(case):
    """Runs one case. Returns (status, detail, extra) where status is PASS, NOT_TAKEN or "mismatch";
    extra carries what classification needs (the compare kind, max relative difference, the
    report)."""
    name = case["shape"]
    _n, _family, _types, build = SHAPE_BY_NAME[name]
    df = frame(case["dtype"], case["ds"])
    lf, spec = build(df, case["dtype"], case["ds"])
    want = lf.collect()
    eng = metal_engine()
    try:
        got = lf.collect(engine=eng)
    except Exception as exc:                      # noqa: BLE001 -- an engine failure is a mismatch
        return "mismatch", f"MetalEngine raised {type(exc).__name__}: {str(exc).splitlines()[0][:300]}", \
            {"kind": "error"}
    rep = eng.last_report
    taken = bool(rep.taken) and not rep.fallbacks
    d = compare(got, want, spec)
    extra = {"kind": None if d is None else d[0], "max_rel": None if d is None else d[2],
             "fallbacks": list(rep.fallbacks), "taken": bool(rep.taken), "spec": spec}
    if d is not None and d[0] == "float_agg":
        extra["ulps_of_magnitude"] = d[3] / _unit_magnitude(df, spec["agg_col"], case["dtype"],
                                                            case["shape"])
        extra["rows"] = df.height
    if d is None:
        if not rep.taken:
            reason = rep.fallbacks[0] if rep.fallbacks else "Polars' optimised plan has no node to run"
            return NOT_TAKEN, reason, extra
        if rep.fallbacks:
            return NOT_TAKEN, "partly on Polars: " + rep.fallbacks[0], extra
        return PASS, "", extra
    return "mismatch", d[1], extra


# ==================================================================================================
# the report's hooks

HOST_DESCRIPTION = ("lf.collect(engine=MetalEngine(shapes='all', min_rows=0)) against lf.collect(), "
                    f"polars {pl.__version__}")


def reproducer(case):
    """A line of Python that reruns one case from python/tests."""
    return (f"import engine_polars_grid as g, engine_conformance as ec; print(g.run_case("
            f"{{'engine': 'polars', 'shape': {case['shape']!r}, 'family': {case['family']!r}, "
            f"'dtype': {case['dtype']!r}, 'ds': ec.DataShape{tuple(case['ds'])!r}}}))")



def _summation_order(case, extra):
    """A float sum or mean that differs from Polars' by no more than two summations' worth of
    rounding error: |Metal - Polars| <= 2 (n - 1) u sum(|x|) for a sum of n values, and 2 u sum(|x|)
    for a mean (the same bound divided by the count, with sum(|x|) over the whole column, which is
    at least the group's)."""
    if extra.get("kind") != "float_agg" or extra.get("ulps_of_magnitude") is None:
        return False
    n = extra.get("rows", 0)
    bound = 2.0 * max(n - 1, 1) if case["shape"].endswith("_sum") else 2.0
    return extra["ulps_of_magnitude"] <= bound


DIVERGENCES = [
    ec.Divergence("float-summation-order", ENGINE,
                  "a float sum or mean adds in another order than Polars, and differs in the last bits",
                  "docs/POLARS.md", "**Float sums and means.**",
                  check=_summation_order),
]
