"""The three rows this script owns: float64 sort/argsort, `shift`, and float64 `sqrt`.

A small stand-in for `full_matrix.py` while those three losses are worked on — same columns, same
seed, same measurement rule (one warm-up, best of 5), so the numbers line up with the matrix rows —
without paying for the other 900 comparisons or their memory.

    PYTHONPATH=python python Benchmarks/loss_sort_shift_sqrt.py            # 10M and 50M
    PYTHONPATH=python python Benchmarks/loss_sort_shift_sqrt.py --rows 10000000
    PYTHONPATH=python python Benchmarks/loss_sort_shift_sqrt.py --only sqrt
"""
import argparse, gc, os, sys, time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import polars as pl
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import arrowmetal as am

SEED = 20260906


def best_of(fn, iters=5, budget=1.2):
    """(best wall ms, iterations): one warm-up, then best of `iters`, the matrix's rule."""
    fn()
    best, total, n = float("inf"), 0.0, 0
    while n < iters and (n < 2 or total < budget):
        t0 = time.perf_counter()
        fn()
        w = time.perf_counter() - t0
        best = min(best, w)
        total += w
        n += 1
    return best * 1000.0, n


def case(label, rows, cands):
    print(f"  {label:<26} {rows:>11,}", flush=True)
    out = {}
    for lib, fn in cands.items():
        if fn is None:
            continue
        ms, n = best_of(fn)
        out[lib] = ms
        print(f"      {lib:<12} {ms:9.3f} ms  (best of {n})", flush=True)
    am_ms = out.get("arrowmetal")
    cpu = {k: v for k, v in out.items() if k != "arrowmetal"}
    if am_ms and cpu:
        lib = min(cpu, key=cpu.get)
        print(f"      -> ratio {cpu[lib] / am_ms:6.2f}x vs {lib} ({cpu[lib]:.3f} ms)", flush=True)
    gc.collect()


def run(n, only):
    rng = np.random.default_rng(SEED)
    print(f"\n=== {n:,} rows ===", flush=True)

    if only in (None, "sort"):
        v = rng.integers(-(2 ** 62), 2 ** 62, size=n, dtype=np.int64)     # i64_nn
        a = pa.array(v)
        g, p = am.array(a), pl.Series("x", a)
        case("argsort int64", n, {
            "arrowmetal": lambda: g.argsort(),
            "polars": lambda: p.arg_sort(),
            "pyarrow": lambda: pc.array_sort_indices(a),
            "numpy": lambda: np.argsort(v)})
        del v, a, g, p
        gc.collect()

        f = rng.random(n) * 2e9 - 1e9                                     # f64_nn
        fa = pa.array(f)
        fg, fp = am.array(fa), pl.Series("x", fa)
        case("argsort float64", n, {
            "arrowmetal": lambda: fg.argsort(),
            "polars": lambda: fp.arg_sort(),
            "pyarrow": lambda: pc.array_sort_indices(fa),
            "numpy": lambda: np.argsort(f)})
        case("sort float64", n, {
            "arrowmetal": lambda: fg.sort(),
            "polars": lambda: fp.sort(),
            "pyarrow": lambda: pc.take(fa, pc.array_sort_indices(fa)),
            "numpy": lambda: np.sort(f)})
        del f, fa, fg, fp
        gc.collect()

    if only in (None, "shift"):
        rng2 = np.random.default_rng(SEED)
        iv = rng2.integers(-1000, 1001, size=n, dtype=np.int64)           # i64 (10% nulls)
        ia = pa.array(iv, mask=rng2.random(n) < 0.10)
        ig, ip = am.array(ia), pl.Series("x", ia)
        idf = pd.Series(pd.arrays.ArrowExtensionArray(ia))
        case("shift (lag 1, int64)", n, {
            "arrowmetal": lambda: ig.shift(1),
            "polars": lambda: ip.shift(1),
            "pandas": lambda: idf.shift(1)})
        if hasattr(ig, "shift") and "view" in (am.MetalArray.shift.__doc__ or ""):
            case("shift (lag 1, view=True)", n, {
                "arrowmetal": lambda: ig.shift(1, view=True),
                "polars": lambda: ip.shift(1),
                "pandas": lambda: idf.shift(1)})
        del iv, ia, ig, ip, idf
        gc.collect()

    if only in (None, "sqrt"):
        rng3 = np.random.default_rng(SEED)
        bv = rng3.random(n) * 1000.0 + 1.0                                # f64b
        ba = pa.array(bv)
        bg, bp = am.array(ba), pl.Series("x", ba)
        case("sqrt (float64)", n, {
            "arrowmetal": lambda: bg.sqrt(),
            "polars": lambda: bp.sqrt(),
            "pyarrow": lambda: pc.sqrt(ba),
            "numpy": lambda: np.sqrt(bv)})
        del bv, ba, bg, bp
        gc.collect()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, action="append")
    ap.add_argument("--only", choices=["sort", "shift", "sqrt"])
    args = ap.parse_args()
    for rows in (args.rows or [10_000_000, 50_000_000]):
        run(rows, args.only)
