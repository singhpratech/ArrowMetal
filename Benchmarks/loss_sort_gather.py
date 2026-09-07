"""The sort family, before and after the `sorted()` gather went away.

A narrow stand-in for `Benchmarks/full_matrix.py`: the same seed, the same column builders and the
same rule (one warm-up, then best of five), but only ArrowMetal, so a before/after can be taken in a
few minutes rather than forty. Two modes:

    --mode matrix   the matrix's own sort-family rows, plus a nullable float64 sort (10% nulls)
    --mode sweep    a shape sweep well past the matrix: five sizes x eleven column shapes x three
                    types, so a fast path that only helps the matrix's columns shows up as such

Usage:

    PYTHONPATH=python python Benchmarks/loss_sort_gather.py --tag after --csv out.csv
    PYTHONPATH=python python Benchmarks/loss_sort_gather.py --mode sweep --tag after --csv out.csv
    python Benchmarks/loss_sort_gather.py --mode sweep --baseline OLD_PYTHON_DIR:OLD_DYLIB

`--baseline` loads a second build (its ctypes package and its dylib) into the *same* process and
measures the two alternately, case by case: a build measured in its own process is measured at a
different minute, and at 1M rows the difference between two minutes on a shared machine is larger
than the difference between the two builds. Each library gets its own Metal context and its own
buffer pool, and every case warms both before either is timed. Passing the *same* build on both
sides is how the harness's own noise floor was measured (at 1M rows it reaches 12%).

Without `--baseline` one build runs per process, which is how full_matrix measures and what the
matrix-comparable numbers come from; run the two tags one after the other, never at the same time.

Prints one line per (op, shape, rows, build) and, with `--csv`, appends them to a CSV. Never run two
of these at once: the numbers move by a factor on a busy GPU.
"""
import argparse
import csv
import gc
import importlib.util
import os
import sys
import time

import numpy as np
import pyarrow as pa

SEED = 20260906            # full_matrix.py's seed
ITERS, BUDGET = 5, 1.2     # full_matrix.py's Bench(5, 1.2)


def load_engine(pkg_dir, lib_path, alias):
    """Imports `<pkg_dir>/arrowmetal` under `alias`, bound to `lib_path`. Two of these coexist."""
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
# `ARROWMETAL_PYTHON` points at a checkout's `python/` directory, so a before/after that spans a
# change to the ctypes package (a new entry point, say) can run the old package against the old
# library; without it the package next to this script is used, as everywhere else in Benchmarks/.
DEFAULT_PKG = os.environ.get("ARROWMETAL_PYTHON") or os.path.join(HERE, "..", "python")


def best_of(fn):
    """(wall_ms, iterations): best of up to ITERS calls after one warm-up, full_matrix's rule."""
    fn()
    best, total, n = float("inf"), 0.0, 0
    while n < ITERS and (n < 2 or total < BUDGET):
        t0 = time.perf_counter()
        fn()
        w = time.perf_counter() - t0
        best = min(best, w)
        total += w
        n += 1
    return best * 1000.0, n


# ---------------------------------------------------------------------------- the matrix's own rows

def matrix_arrays(n):
    """i64_nn, f64_nn and the two lexsort keys, drawn in full_matrix.py's order from its seed, plus a
    10% null float64 column of the same values (the shape the tdigest work found the null cost on)."""
    rng = np.random.default_rng(SEED)
    i64 = rng.integers(-(2 ** 62), 2 ** 62, size=n, dtype=np.int64)
    f64 = rng.random(n) * 2e9 - 1e9
    lex_a = rng.integers(0, 1000, size=n, dtype=np.int32)
    lex_b = rng.integers(0, 100000, size=n, dtype=np.int32)
    mask = rng.random(n) < 0.10
    return {"i64": pa.array(i64), "f64": pa.array(f64), "lex_a": pa.array(lex_a),
            "lex_b": pa.array(lex_b), "f64_null": pa.array(f64, mask=mask)}


def matrix_cases(am, c, n):
    """The ops in full_matrix.py's own order — which is not cosmetic: each allocates its scratch out
    of the context's buffer pool, so a row measured first in a process pays to map the big buffers
    and one measured after `argsort` does not (see docs/TESTING.md)."""
    return [("argsort int64", lambda: c["i64"].argsort()),
            ("argsort float64", lambda: c["f64"].argsort()),
            ("sort float64", lambda: c["f64"].sort()),
            ("sort int64", lambda: c["i64"].sort()),
            ("sort float64 (10% nulls)", lambda: c["f64_null"].sort()),
            ("argsort float64 (10% nulls)", lambda: c["f64_null"].argsort()),
            ("top_k (k=100, int64)", lambda: c["i64"].top_k(100)),
            ("select_k_unstable (k=100)", lambda: c["i64"].top_k(100, largest=False)),
            ("lexsort (2 int32 keys)", lambda: am.lexsort_indices([c["lex_a"], c["lex_b"]])),
            ("partition_nth_indices (n/2)", lambda: c["i64"].partition_nth_indices(n // 2))]


# ------------------------------------------------------------------------------------- the sweep

def shape(name, n, dtype, rng):
    """One column of `n` rows in `dtype`, in the named shape. Returns a pyarrow array or None when
    the shape does not apply to the type (the float specials on an integer column)."""
    is_float = dtype in (np.float64, np.float32)
    if name == "random":
        v = (rng.random(n) * 2e9 - 1e9) if is_float else rng.integers(-(2 ** 40), 2 ** 40, n)
    elif name == "sorted":
        v = np.sort((rng.random(n) * 2e9 - 1e9) if is_float else rng.integers(-(2 ** 40), 2 ** 40, n))
    elif name == "reversed":
        v = np.sort((rng.random(n) * 2e9 - 1e9) if is_float else rng.integers(-(2 ** 40), 2 ** 40, n))[::-1].copy()
    elif name == "all-equal":
        v = np.full(n, 7.5 if is_float else 7)
    elif name == "1000-distinct":
        v = rng.integers(0, 1000, n)
        if is_float:
            v = v.astype(np.float64) / 8.0
    elif name == "with -0.0":
        if not is_float:
            return None
        v = rng.random(n) * 2e9 - 1e9
        v[::100] = -0.0
    elif name == "with NaN":
        if not is_float:
            return None
        v = rng.random(n) * 2e9 - 1e9
        v[::100] = np.nan
    elif name in ("1% nulls", "10% nulls", "50% nulls"):
        frac = {"1% nulls": 0.01, "10% nulls": 0.10, "50% nulls": 0.50}[name]
        v = (rng.random(n) * 2e9 - 1e9) if is_float else rng.integers(-(2 ** 40), 2 ** 40, n)
        return pa.array(np.asarray(v, dtype=dtype), mask=rng.random(n) < frac)
    elif name == "sliced (offset 33)":
        v = (rng.random(n) * 2e9 - 1e9) if is_float else rng.integers(-(2 ** 40), 2 ** 40, n)
        return pa.array(np.asarray(v, dtype=dtype))[33:]
    else:
        raise ValueError(name)
    return pa.array(np.asarray(v, dtype=dtype))


SHAPES = ["random", "sorted", "reversed", "all-equal", "1000-distinct", "with -0.0", "with NaN",
          "1% nulls", "10% nulls", "50% nulls", "sliced (offset 33)"]
DTYPES = [("float64", np.float64), ("int64", np.int64), ("int32", np.int32)]


# ------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["matrix", "sweep"], default="matrix")
    ap.add_argument("--rows", type=int, nargs="*")
    ap.add_argument("--tag", default="after")
    ap.add_argument("--baseline", help="PYTHON_DIR:DYLIB of a second build, measured alternately")
    ap.add_argument("--baseline-tag", default="before")
    ap.add_argument("--rounds", type=int, default=2, help="alternating rounds per case")
    ap.add_argument("--csv")
    args = ap.parse_args()
    rows = args.rows or ([10_000_000, 50_000_000] if args.mode == "matrix"
                         else [1_000_000, 3_000_000, 10_000_000, 27_000_000, 50_000_000])

    under_test_lib = os.environ.get("ARROWMETAL_LIB")
    engines = []
    if args.baseline:
        pkg, _, lib = args.baseline.partition(":")
        engines.append((args.baseline_tag, load_engine(pkg, lib, "arrowmetal_baseline")))
    engines.append((args.tag, load_engine(DEFAULT_PKG, under_test_lib, "arrowmetal_under_test")))

    out = []

    def measure(op, sh, n, per_engine):
        """Times every engine on the same case, alternately, and keeps each one's best.

        Alternating matters more than repeating: two builds timed in two processes are timed at two
        different minutes, and on a shared machine a minute is worth more than this change is."""
        for tag, _ in engines:            # warm every build before any of them is timed
            per_engine[tag]()
        best = {tag: float("inf") for tag, _ in engines}
        for _ in range(max(1, args.rounds)):
            for tag, _ in engines:
                ms, _iters = best_of(per_engine[tag])
                best[tag] = min(best[tag], ms)
        for tag, _ in engines:
            print(f"{tag:<7} {op:<28} {sh:<20} {n:>11,} {best[tag]:9.3f} ms", flush=True)
            out.append(dict(tag=tag, mode=args.mode, op=op, shape=sh, rows=n,
                            wall_ms=round(best[tag], 3), iters=ITERS * max(1, args.rounds)))
        gc.collect()

    for n in rows:
        if args.mode == "matrix":
            arrays = matrix_arrays(n)
            # Each build imports the same pyarrow columns and gets its own case list over them.
            cases = {tag: dict(matrix_cases(mod, {k: mod.array(v) for k, v in arrays.items()}, n))
                     for tag, mod in engines}
            for op in cases[engines[-1][0]]:
                measure(op, "matrix", n, {tag: cases[tag][op] for tag, _ in engines})
            del cases, arrays
        else:
            rng = np.random.default_rng(SEED)
            # One warm-up sort per size and per build, so the buffer pool already holds the big
            # scratch buffers before the first measured row (docs/TESTING.md's cold-pool caveat).
            warmcol = pa.array(rng.random(n))
            for _tag, mod in engines:
                w = mod.array(warmcol)
                w.sort()
                w.argsort()
                del w
            del warmcol
            for tname, dtype in DTYPES:
                for sh in SHAPES:
                    col = shape(sh, n, dtype, rng)
                    if col is None:
                        continue
                    g = {tag: mod.array(col) for tag, mod in engines}
                    measure(f"sort {tname}", sh, n,
                            {tag: (lambda a=a: a.sort()) for tag, a in g.items()})
                    measure(f"argsort {tname}", sh, n,
                            {tag: (lambda a=a: a.argsort()) for tag, a in g.items()})
                    del col, g
                    gc.collect()
        gc.collect()

    if args.csv:
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["tag", "mode", "op", "shape", "rows", "wall_ms", "iters"])
            if new:
                w.writeheader()
            w.writerows(out)


if __name__ == "__main__":
    main()
