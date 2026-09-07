"""The sort family, before and after the `sorted()` gather went away.

A narrow stand-in for `Benchmarks/full_matrix.py`: the same seed, the same column builders and the
same rule (one warm-up, then best of five), but only ArrowMetal, so a before/after can be taken in a
few minutes rather than forty. Two modes:

    --mode matrix   the matrix's own sort-family rows, plus a nullable float64 sort (10% nulls)
    --mode sweep    a shape sweep well past the matrix: five sizes x eleven column shapes x three
                    types, so a fast path that only helps the matrix's columns shows up as such

Usage:

    PYTHONPATH=python python Benchmarks/loss_sort_gather.py --tag before --csv out.csv
    PYTHONPATH=python python Benchmarks/loss_sort_gather.py --mode sweep --tag after --csv out.csv
    ARROWMETAL_LIB=/path/to/old/libArrowMetalC.dylib PYTHONPATH=python python ... --tag before

Prints one line per (op, shape, rows) and, with `--csv`, appends them to a CSV that a later run with
a different `--tag` can be diffed against. One process per tag, never two at once: the numbers move
by a factor on a busy GPU.
"""
import argparse
import csv
import gc
import os
import sys
import time

import numpy as np
import pyarrow as pa

# `ARROWMETAL_PYTHON` points at a checkout's `python/` directory, so a before/after that spans a
# change to the ctypes package (a new entry point, say) can run the old package against the old
# library; without it the package next to this script is used, as everywhere else in Benchmarks/.
sys.path.insert(0, os.environ.get("ARROWMETAL_PYTHON")
                or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import arrowmetal as am

SEED = 20260906            # full_matrix.py's seed
ITERS, BUDGET = 5, 1.2     # full_matrix.py's Bench(5, 1.2)


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

def matrix_columns(n):
    """i64_nn, f64_nn and the two lexsort keys, drawn in full_matrix.py's order from its seed, plus a
    10% null float64 column of the same values (the shape the tdigest work found the null cost on)."""
    rng = np.random.default_rng(SEED)
    i64 = rng.integers(-(2 ** 62), 2 ** 62, size=n, dtype=np.int64)
    f64 = rng.random(n) * 2e9 - 1e9
    lex_a = rng.integers(0, 1000, size=n, dtype=np.int32)
    lex_b = rng.integers(0, 100000, size=n, dtype=np.int32)
    mask = rng.random(n) < 0.10
    return {"i64": am.array(pa.array(i64)), "f64": am.array(pa.array(f64)),
            "lex_a": am.array(pa.array(lex_a)), "lex_b": am.array(pa.array(lex_b)),
            "f64_null": am.array(pa.array(f64, mask=mask))}


def matrix_cases(c, n):
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
    ap.add_argument("--tag", default="run")
    ap.add_argument("--csv")
    args = ap.parse_args()
    rows = args.rows or ([10_000_000, 50_000_000] if args.mode == "matrix"
                         else [1_000_000, 3_000_000, 10_000_000, 27_000_000, 50_000_000])

    out = []

    def record(op, sh, n, fn):
        wall, iters = best_of(fn)
        print(f"{args.tag:<7} {op:<28} {sh:<20} {n:>11,} {wall:9.3f} ms  ({iters})", flush=True)
        out.append(dict(tag=args.tag, mode=args.mode, op=op, shape=sh, rows=n,
                        wall_ms=round(wall, 3), iters=iters))

    for n in rows:
        if args.mode == "matrix":
            c = matrix_columns(n)
            for op, fn in matrix_cases(c, n):
                record(op, "matrix", n, fn)
            del c
        else:
            rng = np.random.default_rng(SEED)
            # One warm-up sort per size, so the buffer pool already holds the big scratch buffers
            # before the first measured row (docs/TESTING.md's cold-pool caveat).
            warm = am.array(pa.array(rng.random(n)))
            warm.sort()
            warm.argsort()
            del warm
            for tname, dtype in DTYPES:
                for sh in SHAPES:
                    col = shape(sh, n, dtype, rng)
                    if col is None:
                        continue
                    g = am.array(col)
                    record(f"sort {tname}", sh, n, lambda g=g: g.sort())
                    record(f"argsort {tname}", sh, n, lambda g=g: g.argsort())
                    del g, col
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
