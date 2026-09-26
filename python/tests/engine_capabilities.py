#!/usr/bin/env python3
"""The MetalEngine capability table, generated from runs of the engine: docs/ENGINE_CAPABILITIES.md.

    PYTHONPATH=python python python/tests/engine_capabilities.py            # print the table
    PYTHONPATH=python python python/tests/engine_capabilities.py --write    # write the doc
    PYTHONPATH=python python python/tests/engine_capabilities.py --check    # exit 1 if it differs

Each cell is one run: a 64-row frame whose column `x` has the row's dtype and the column's null
pattern, one plan shape over it, collected through Polars and through
`MetalEngine(shapes="all", min_rows=0)` (everything the engine can translate, whatever the default
size and shape gates would say). The cell says where the plan ran and, when it stayed with Polars,
which reason the engine's report gave. A Metal answer that differs from Polars' is marked `DIFFERS`
and an engine exception `ERROR`; `test_polars_engine.py` requires neither to appear and the
committed file to match a fresh run, apart from the line that names the ArrowMetal commit.
"""
import argparse
import datetime as dt
import decimal
import os
import re
import subprocess
import sys
import tempfile
import warnings
from collections import OrderedDict

import polars as pl
from polars.testing import assert_frame_equal

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DOC = os.path.join(REPO, "docs", "ENGINE_CAPABILITIES.md")
DOC_REL = "docs/ENGINE_CAPABILITIES.md"

import arrowmetal as am                                               # noqa: E402
from arrowmetal import polars_engine as pe                            # noqa: E402

N = 64
NULLS = ("none", "some", "all")      # no null; every third row null; every row null


def _v(i):
    return (i * 7) % 23


_WORDS = ["apple", "pear", "fig", "kiwi", "plum"]
_D0 = dt.date(2024, 1, 1)
_T0 = dt.datetime(2024, 1, 1)

# name -> (Polars dtype, value of row i). The order is the table's row order.
DTYPES = OrderedDict([
    ("Int8", (pl.Int8, lambda i: _v(i) - 11)),
    ("Int16", (pl.Int16, lambda i: _v(i) - 11)),
    ("Int32", (pl.Int32, lambda i: _v(i) - 11)),
    ("Int64", (pl.Int64, lambda i: _v(i) - 11)),
    ("UInt8", (pl.UInt8, _v)),
    ("UInt16", (pl.UInt16, _v)),
    ("UInt32", (pl.UInt32, _v)),
    ("UInt64", (pl.UInt64, _v)),
    ("Float32", (pl.Float32, lambda i: (_v(i) - 11) / 4.0)),
    ("Float64", (pl.Float64, lambda i: (_v(i) - 11) / 4.0)),
    ("Boolean", (pl.Boolean, lambda i: i % 3 == 0)),
    ("String", (pl.String, lambda i: _WORDS[_v(i) % 5])),
    ("Date", (pl.Date, lambda i: _D0 + dt.timedelta(days=_v(i)))),
    ("Datetime(ms)", (pl.Datetime("ms"), lambda i: _T0 + dt.timedelta(hours=_v(i)))),
    ("Datetime(us)", (pl.Datetime("us"), lambda i: _T0 + dt.timedelta(hours=_v(i)))),
    ("Datetime(ns)", (pl.Datetime("ns"), lambda i: _T0 + dt.timedelta(hours=_v(i)))),
    ("Datetime(us, UTC)", (pl.Datetime("us", "UTC"),
                           lambda i: (_T0 + dt.timedelta(hours=_v(i))).replace(tzinfo=dt.timezone.utc))),
    ("Duration(ms)", (pl.Duration("ms"), lambda i: dt.timedelta(seconds=_v(i)))),
    ("Duration(us)", (pl.Duration("us"), lambda i: dt.timedelta(seconds=_v(i)))),
    ("Duration(ns)", (pl.Duration("ns"), lambda i: dt.timedelta(seconds=_v(i)))),
    ("Time", (pl.Time, lambda i: dt.time(_v(i), 0))),
    ("Decimal(10, 2)", (pl.Decimal(10, 2), lambda i: decimal.Decimal(_v(i)))),
    ("Categorical", (pl.Categorical, lambda i: _WORDS[_v(i) % 3])),
    ("Enum", (pl.Enum(_WORDS), lambda i: _WORDS[_v(i) % 3])),
    ("Binary", (pl.Binary, lambda i: bytes([65 + _v(i)]))),
    ("List(Int64)", (pl.List(pl.Int64), lambda i: [_v(i)])),
    ("Array(Int64, 2)", (pl.Array(pl.Int64, 2), lambda i: [_v(i), 1])),
    ("Struct", (pl.Struct({"a": pl.Int64}), lambda i: {"a": _v(i)})),
    ("Null", (pl.Null, lambda i: None)),
])


def _series(name, dtype, value, nulls, n):
    vals = [None if (nulls == "all" or (nulls == "some" and i % 3 == 1)) else value(i)
            for i in range(n)]
    return pl.Series(name, vals, dtype=dtype)


def frame(dtype_name, nulls):
    dtype, value = DTYPES[dtype_name]
    return pl.DataFrame([_series("x", dtype, value, nulls, N),
                         pl.Series("k", [i % 4 for i in range(N)], dtype=pl.Int32),
                         pl.Series("v", list(range(N)), dtype=pl.Int64)])


def right(dtype_name, nulls):
    dtype, value = DTYPES[dtype_name]
    return pl.DataFrame([_series("x", dtype, value, nulls, 16),
                         pl.Series("w", list(range(16)), dtype=pl.Int64)])


def literal(dtype_name):
    dtype, value = DTYPES[dtype_name]
    return pl.lit(pl.Series([value(1)], dtype=dtype)).first() if dtype_name in (
        "List(Int64)", "Array(Int64, 2)", "Struct") else pl.lit(value(1), dtype=dtype)


# ==================================================================================================
# shapes: (name, what it runs, build(lf, dtype_name, nulls, parquet_path) -> LazyFrame, compare)
# compare: "exact" (row order), "multiset", ("keys", col) (that column in order, rows as a
# multiset) or "float" (as a multiset, floats to a relative 1e-9).

x = pl.col("x")


def _agg(fn):
    return lambda e: getattr(e, fn)()


SHAPES = [
    ("filter_other_column", "`filter(k > 1)`: a Filter that carries `x`",
     lambda lf, d, n, p: lf.filter(pl.col("k") > 1), "exact"),
    ("filter_is_not_null", "`filter(x.is_not_null())`",
     lambda lf, d, n, p: lf.filter(x.is_not_null()), "exact"),
    ("filter_eq_literal", "`filter(x == <a value of x>)`",
     lambda lf, d, n, p: lf.filter(x == literal(d)), "exact"),
    ("filter_gt_literal", "`filter(x > <a value of x>)`",
     lambda lf, d, n, p: lf.filter(x > literal(d)), "exact"),
    ("select_add", "`select(x + x, v)`",
     lambda lf, d, n, p: lf.select((x + x).alias("y"), "v"), "exact"),
    ("select_is_null", "`select(x.is_null(), v)`",
     lambda lf, d, n, p: lf.select(x.is_null().alias("y"), "v"), "exact"),
    ("with_columns", "`with_columns(k * 2)`: an HStack that carries `x`",
     lambda lf, d, n, p: lf.with_columns((pl.col("k") * 2).alias("k2")), "exact"),
    ("slice", "`filter(k > 1).slice(2, 10)`: a Slice that carries `x`",
     lambda lf, d, n, p: lf.filter(pl.col("k") > 1).slice(2, 10), "exact"),
    ("sort", "`sort(x)` (nulls first)",
     lambda lf, d, n, p: lf.sort("x"), ("keys", "x")),
    ("sort_desc_nulls_last", "`sort(x, descending=True, nulls_last=True)`",
     lambda lf, d, n, p: lf.sort("x", descending=True, nulls_last=True), ("keys", "x")),
    ("top_k", "`sort(x).head(5)`",
     lambda lf, d, n, p: lf.sort("x").head(5), ("keys_only", "x")),
    ("group_by_key", "`group_by(x).agg(v.sum())`: `x` as the key",
     lambda lf, d, n, p: lf.group_by("x").agg(pl.col("v").sum()), "multiset"),
]
for _fn in ("sum", "min", "max", "mean", "count"):
    SHAPES.append((f"group_by_value_{_fn}", f"`group_by(k).agg(x.{_fn}())`: `x` as the value",
                   (lambda f: lambda lf, d, n, p: lf.group_by("k").agg(_agg(f)(x)))(_fn), "float"))
for _fn in ("sum", "min", "max", "mean", "count"):
    SHAPES.append((f"aggregate_{_fn}", f"`select(x.{_fn}())`: a whole-frame aggregate",
                   (lambda f: lambda lf, d, n, p: lf.select(_agg(f)(x)))(_fn), "float"))
for _how in ("inner", "left", "semi", "anti"):
    SHAPES.append((f"join_{_how}", f"`join(right, on=x, how=\"{_how}\")`: `x` as the key",
                   (lambda h: lambda lf, d, n, p: lf.select("x", "v").join(
                       right(d, n).lazy(), on="x", how=h))(_how), "multiset"))
SHAPES += [
    ("unique", "`unique(subset=[x])`",
     lambda lf, d, n, p: lf.unique(subset=["x"]).select("x"), "multiset"),
    ("parquet_filter", "`scan_parquet(f).filter(k > 1)`: a Parquet Scan that reads `x`",
     lambda lf, d, n, p: pl.scan_parquet(p).filter(pl.col("k") > 1), "exact"),
    ("parquet_sort", "`scan_parquet(f).sort(x)`",
     lambda lf, d, n, p: pl.scan_parquet(p).sort("x"), ("keys", "x")),
]


# ==================================================================================================
# running


def _same(got, want, how):
    try:
        if isinstance(how, tuple):
            assert_frame_equal(got.select(how[1]), want.select(how[1]))
            if how[0] == "keys_only":
                return True
            how = "multiset"
        if how == "exact":
            assert_frame_equal(got, want)
        else:
            kw = {"check_exact": False, "rel_tol": 1e-9} if how == "float" else {}
            assert_frame_equal(got, want, check_row_order=False, **kw)
        return True
    except AssertionError:
        return False
    except Exception:                                 # noqa: BLE001 -- unorderable dtypes
        return got.to_dicts() == want.to_dicts() or sorted(map(repr, got.rows())) == sorted(
            map(repr, want.rows()))


_ID = re.compile(r"#\d+")


def _reason(line, tmp):
    """A fallback line without node ids or temporary paths."""
    return _ID.sub("", line).replace(tmp, "<dir>")


def run_cell(shape, dtype_name, nulls, tmp):
    """(status, reason): status is "metal", "partly", "polars", "nothing" (Polars' optimised plan
    had nothing to run), "n/a" (Polars itself rejects the plan), "DIFFERS" or "ERROR"."""
    _name, _what, build, how = shape
    df = frame(dtype_name, nulls)
    path = os.path.join(tmp, f"{dtype_name}-{nulls}.parquet")
    try:
        if shape[0].startswith("parquet") and not os.path.exists(path):
            df.write_parquet(path)
        lf = build(df.lazy(), dtype_name, nulls, path)
        want = lf.collect()
    except Exception as e:                            # noqa: BLE001 -- Polars' own refusal
        return "n/a", f"Polars: {type(e).__name__}"
    eng = pe.MetalEngine(shapes="all", min_rows=0)
    try:
        got = lf.collect(engine=eng)
    except Exception as e:                            # noqa: BLE001
        return "ERROR", f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    rep = eng.last_report
    first = _reason(rep.fallbacks[0], tmp) if rep.fallbacks else None
    if rep.taken and not _same(got, want, how):
        return "DIFFERS", first or ""
    if rep.taken:
        return ("partly", first) if rep.fallbacks else ("metal", None)
    if first is None:
        return "nothing", None
    return "polars", first


def run_all(shapes=None, dtypes=None, progress=False):
    """{shape name: {(dtype, nulls): (status, reason)}}."""
    out = OrderedDict()
    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tmp = os.path.realpath(tmp)
        for shape in SHAPES:
            if shapes and shape[0] not in shapes:
                continue
            cells = out.setdefault(shape[0], OrderedDict())
            for d in DTYPES:
                if dtypes and d not in dtypes:
                    continue
                for n in NULLS:
                    cells[(d, n)] = run_cell(shape, d, n, tmp)
            if progress:
                print(f"  {shape[0]}", file=sys.stderr)
    return out


# ==================================================================================================
# the document


# What the table depends on: a change here that is not committed is marked in the header.
CODE_PATHS = ("python", "Sources", "include", "Package.swift")


def commit_line():
    """The ArrowMetal version and commit the table was generated from, marked when the code it runs
    (CODE_PATHS) held uncommitted changes."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short=10", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", *CODE_PATHS],
                               cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head, dirty = "unknown", ""
    return (f"- ArrowMetal {am.__version__}, commit `{head}`"
            + (" with uncommitted changes" if dirty else ""))


_LABEL = {"metal": "Metal", "nothing": "-", "n/a": "n/a"}


def render(results):
    reasons = OrderedDict()

    def cell(status, reason):
        if status in _LABEL:
            return _LABEL[status]
        key = reason or ""
        if key not in reasons:
            reasons[key] = len(reasons) + 1
        word = {"polars": "Polars", "partly": "partly"}.get(status, status)
        return f"{word} [{reasons[key]}]"

    lines = []
    add = lines.append
    ir = pe.compatibility()["ir"]
    add("# MetalEngine capability table")
    add("")
    add("Generated by `python/tests/engine_capabilities.py` from runs of the engine; "
        "`test_polars_engine.py` fails when this file differs from a fresh run.")
    add("")
    add(f"- Polars {pl.__version__}, IR {ir}")
    add(commit_line())
    add(f"- Engine: `MetalEngine(shapes=\"all\", min_rows=0)`, every node it can translate (the "
        "default `MetalEngine()` takes a subset: [POLARS.md](POLARS.md), \"Which translatable "
        "subtrees it runs: the defaults\")")
    add(f"- Input: a {N}-row frame with the column `x` under test, `k` (Int32) and `v` (Int64); null "
        "patterns: none, some (every third row null), all")
    add("")
    add("Cells: `Metal` -- the whole plan ran on Metal and matched Polars; `Polars [n]` -- the plan "
        "stayed with Polars for reason n below; `partly [n]` -- part of the plan ran on Metal, the "
        "rest stayed with Polars for reason n; `-` -- Polars' optimised plan had nothing to run; "
        "`n/a` -- Polars itself rejects the plan for this dtype.")
    add("")
    add("## Summary")
    add("")
    add("| shape | runs | Metal | partly | Polars | nothing to run | n/a |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    total = OrderedDict((k, 0) for k in ("runs", "metal", "partly", "polars", "nothing", "n/a"))
    body = []
    for name, what, _b, _h in SHAPES:
        cells = results.get(name)
        if not cells:
            continue
        counts = OrderedDict((k, 0) for k in total)
        for status, _r in cells.values():
            counts["runs"] += 1
            if status in counts:
                counts[status] += 1
        for k in total:
            total[k] += counts[k]
        add(f"| `{name}` | {counts['runs']} | {counts['metal']} | {counts['partly']} | "
            f"{counts['polars']} | {counts['nothing']} | {counts['n/a']} |")
        body.append("")
        body.append(f"## `{name}`")
        body.append("")
        body.append(what)
        body.append("")
        body.append("| dtype of `x` | " + " | ".join(f"{n} null" if n != "some" else "some null"
                                                    for n in NULLS) + " |")
        body.append("|---|" + "---|" * len(NULLS))
        for d in DTYPES:
            if (d, NULLS[0]) not in cells:
                continue
            body.append(f"| {d} | " + " | ".join(cell(*cells[(d, n)]) for n in NULLS) + " |")
    add(f"| all | {total['runs']} | {total['metal']} | {total['partly']} | {total['polars']} | "
        f"{total['nothing']} | {total['n/a']} |")
    lines += body
    lines.append("")
    lines.append("## Reasons")
    lines.append("")
    lines.append("As the engine's report gives them (`Kind: reason`, for the first node that "
                 "stayed with Polars).")
    lines.append("")
    for text, i in reasons.items():
        lines.append(f"{i}. {text.replace('|', '\\|')}")
    lines.append("")
    return "\n".join(lines)


def comparable(text):
    """The document without the line that names the ArrowMetal commit."""
    return "\n".join(line for line in text.splitlines() if not line.startswith("- ArrowMetal "))


def bad_cells(results):
    return [(name, key, st) for name, cells in results.items()
            for key, st in cells.items() if st[0] in ("DIFFERS", "ERROR")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help=f"write {DOC_REL}")
    parser.add_argument("--check", action="store_true", help=f"exit 1 if {DOC_REL} differs")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)
    results = run_all(progress=not args.quiet)
    text = render(results)
    bad = bad_cells(results)
    for name, key, st in bad:
        print(f"{st[0]}: {name} / {key}: {st[1]}", file=sys.stderr)
    if args.write:
        with open(DOC, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {DOC_REL}")
    elif args.check:
        with open(DOC, encoding="utf-8") as f:
            same = comparable(f.read()) == comparable(text)
        print(f"{DOC_REL} is " + ("current" if same else "stale: rerun with --write"))
        return 0 if same and not bad else 1
    else:
        print(text)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
