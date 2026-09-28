"""Sort options (`null_placement`, `float_order`): the default path before and after, and the one-key
option form against the helper-key form it replaces.

Two builds are loaded into one process and timed alternately, case by case (the rule and the loader
of `Benchmarks/loss_sort_gather.py`: one warm-up, then best of five, `--rounds` alternating rounds).

    PYTHONPATH=python python Benchmarks/sort_options_bench.py \\
        --baseline OLD_PYTHON_DIR:OLD_DYLIB --csv out.csv

Three groups of rows:

* **default** — every sort entry point with no options, on both builds: argsort of int64, float64,
  float32 and utf8 (1,000 distinct values), a sorted float64 copy, a two-key lexsort, top-k of int64 and
  float64, a 10%-null float64 argsort, and three plan runs over a three-column table (sort by the int64
  column, by the float64 column, and a descending float64 top-100). The two builds should agree
  within noise.
* **option** — the new options, under-test build only: float64 argsort / sorted copy / top-k in
  totalOrder with the nulls first.
* **plan: option vs helper keys** — the same order written two ways through the plan runner. The
  helper-key form is what a caller had to send before the options existed: `with_columns` adds an
  is-null key (for nulls first), a NaN key (for a descending float key, so NaN leads) and a
  signed-zero key (`1/x < 0`, so -0.0 and +0.0 split), and the sort runs over those keys around the
  value; a final `select` drops them. The option form is one key with `nulls` / `float_order`. Both
  are timed on the under-test build, and the helper-key form also on the baseline.
"""
import argparse
import csv
import gc
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa

SEED = 20260928
ITERS, BUDGET = 5, 1.2


def load_engine(pkg_dir, lib_path, alias):
    if lib_path:
        os.environ["ARROWMETAL_LIB"] = lib_path
    else:
        os.environ.pop("ARROWMETAL_LIB", None)
    root = os.path.join(pkg_dir, "arrowmetal")
    spec = importlib.util.spec_from_file_location(alias, os.path.join(root, "__init__.py"),
                                                  submodule_search_locations=[root])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PKG = os.environ.get("ARROWMETAL_PYTHON") or os.path.join(HERE, "..", "python")


def best_of(fn):
    fn()
    best, total, n = float("inf"), 0.0, 0
    while n < ITERS and (n < 2 or total < BUDGET):
        t0 = time.perf_counter()
        fn()
        w = time.perf_counter() - t0
        best = min(best, w)
        total += w
        n += 1
    return best * 1000.0


def columns(n):
    rng = np.random.default_rng(SEED)
    f64 = rng.random(n) * 2e9 - 1e9
    f64[::1000] = np.nan
    f64[1::1000] = -0.0
    f64[2::1000] = 0.0
    f32 = f64.astype(np.float32)
    i64 = rng.integers(-(2 ** 62), 2 ** 62, size=n, dtype=np.int64)
    i32 = rng.integers(0, 1000, size=n, dtype=np.int32)
    words = np.array([f"w{i:04d}" for i in range(1000)], dtype=object)
    s = pa.array(words[rng.integers(0, 1000, size=n)], pa.string())
    mask = rng.random(n) < 0.10
    return {"f64": pa.array(f64), "f32": pa.array(f32), "i64": pa.array(i64), "i32": pa.array(i32),
            "str": s, "f64_null": pa.array(f64, mask=mask)}


def plan_sort(by, extra=None, limit=None):
    """A plan over source `t` (columns x, q, k1): optional helper columns, the sort, an optional limit,
    and a select of the three data columns."""
    node = {"op": "scan", "source": "t"}
    if extra:
        node = {"op": "with_columns", "exprs": extra, "input": node}
    node = {"op": "sort", "by": by, "input": node}
    if limit:
        node = {"op": "limit", "count": limit, "input": node}
    return {"op": "select", "exprs": [[c, f'(col "{c}")'] for c in ("x", "q", "k1")], "input": node}


def helper_keys(descending, nulls_first, limit=None):
    """The helper-key form of one float key in arrow-rs order."""
    c = '(col "x")'
    extra, by = [], []
    if nulls_first:
        extra.append(["n0", f"(if_else (is_null {c}) (i32 1) (i32 0))"])
        by.append(["n0", True])
    if descending:
        extra.append(["f0", f"(if_else (ne {c} {c}) (i32 1) (i32 0))"])
        by.append(["f0", True])
    by.append(["x", descending])
    extra.append(["z0", f"(if_else (lt (div (f64 1) {c}) (f64 0)) (i32 0) (i32 1))"])
    by.append(["z0", descending])
    return plan_sort(by, extra, limit)


def option_key(descending, nulls_first, limit=None):
    opts = {"float_order": "total"}
    if nulls_first:
        opts["nulls"] = "first"
    return plan_sort([["x", descending, opts]], None, limit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="*", default=[1_000_000, 10_000_000, 50_000_000])
    ap.add_argument("--tag", default="after")
    ap.add_argument("--baseline", help="PYTHON_DIR:DYLIB of a second build, measured alternately")
    ap.add_argument("--baseline-tag", default="before")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--csv")
    args = ap.parse_args()

    under_test_lib = os.environ.get("ARROWMETAL_LIB")
    engines = []
    if args.baseline:
        pkg, _, lib = args.baseline.partition(":")
        engines.append((args.baseline_tag, load_engine(pkg, lib, "arrowmetal_baseline")))
    engines.append((args.tag, load_engine(DEFAULT_PKG, under_test_lib, "arrowmetal_under_test")))
    out = []

    def measure(group, op, n, per_engine):
        for tag, fn in per_engine.items():
            fn()
        best = {tag: float("inf") for tag in per_engine}
        for _ in range(max(1, args.rounds)):
            for tag, fn in per_engine.items():
                best[tag] = min(best[tag], best_of(fn))
        for tag in per_engine:
            print(f"{group:<9} {tag:<14} {op:<56} {n:>11,} {best[tag]:9.3f} ms", flush=True)
            out.append(dict(group=group, tag=tag, op=op, rows=n, wall_ms=round(best[tag], 3)))
        gc.collect()

    for n in args.rows:
        raw = columns(n)
        table = pa.table({"x": raw["f64"], "q": raw["i64"], "k1": raw["i32"]})
        table_null = pa.table({"x": raw["f64_null"], "q": raw["i64"], "k1": raw["i32"]})
        cols = {tag: {k: am.MetalArray.from_arrow(v) for k, v in raw.items()} for tag, am in engines}
        # One frame per build and table, reused by every run, so the columns are imported once.
        frames = {(tag, i): am.scan(tb, name="t") for tag, am in engines
                  for i, tb in enumerate((table, table_null))}

        def each(make):
            return {tag: make(am, cols[tag]) for tag, am in engines}

        # ---- default path, both builds
        measure("default", "argsort int64", n, each(lambda am, c: lambda: c["i64"].argsort()))
        measure("default", "argsort float64", n, each(lambda am, c: lambda: c["f64"].argsort()))
        measure("default", "argsort float32", n, each(lambda am, c: lambda: c["f32"].argsort()))
        measure("default", "argsort utf8 (1000 distinct)", n, each(lambda am, c: lambda: c["str"].argsort()))
        measure("default", "argsort float64 (10% nulls)", n, each(lambda am, c: lambda: c["f64_null"].argsort()))
        measure("default", "sort float64 (sorted copy)", n, each(lambda am, c: lambda: c["f64"].sort()))
        measure("default", "lexsort int32 + float64", n,
                each(lambda am, c: lambda: am.lexsort_indices([c["i32"], c["f64"]])))
        measure("default", "top_k int64 (k=100)", n, each(lambda am, c: lambda: c["i64"].top_k(100)))
        measure("default", "top_k float64 (k=100)", n, each(lambda am, c: lambda: c["f64"].top_k(100)))
        for label, by, lim in [("plan sort by int64", [["q", False]], None),
                               ("plan sort by float64", [["x", False]], None),
                               ("plan top-100 float64 desc", [["x", True]], 100)]:
            plan = plan_sort(by, None, lim)
            measure("default", label, n,
                    {etag: (lambda f=frames[(etag, 0)], plan=plan: f._with(plan).collect())
                     for etag, _ in engines})

        # ---- the options themselves, under-test build only
        c = cols[engines[-1][0]]
        tag = engines[-1][0]
        measure("option", "argsort float64 total", n, {tag: lambda: c["f64"].argsort(float_order="total")})
        measure("option", "argsort float64 desc total nulls first (10% nulls)", n,
                {tag: lambda: c["f64_null"].argsort(True, null_placement="at_start", float_order="total")})
        measure("option", "sort float64 total (sorted copy)", n,
                {tag: lambda: c["f64"].sort(float_order="total")})
        measure("option", "top_k float64 desc total nulls first (k=100)", n,
                {tag: lambda: c["f64"].top_k(100, null_placement="at_start", float_order="total")})
        measure("option", "top_k float64 desc total nulls first (k=100, 10% nulls)", n,
                {tag: lambda: c["f64_null"].top_k(100, null_placement="at_start", float_order="total")})

        # ---- one key with options against the helper-key form, through the plan runner
        for ti, tb_label in [(0, ""), (1, " (10% nulls)")]:
            for label, desc, nf, lim in [("asc", False, False, None),
                                         ("desc nulls first", True, True, None),
                                         ("desc nulls first top-100", True, True, 100)]:
                helper = helper_keys(desc, nf, lim)
                opt = option_key(desc, nf, lim)
                runs = {}
                for etag, _ in engines:
                    runs[f"{etag}:helper"] = (lambda f=frames[(etag, ti)], helper=helper:
                                              f._with(helper).collect())
                runs[f"{tag}:option"] = (lambda f=frames[(tag, ti)], opt=opt: f._with(opt).collect())
                measure("plan", f"float64 {label}{tb_label}", n, runs)
        del cols, raw, table, table_null, frames
        gc.collect()

    if args.csv:
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["group", "tag", "op", "rows", "wall_ms"])
            if new:
                w.writeheader()
            w.writerows(out)


if __name__ == "__main__":
    main()
