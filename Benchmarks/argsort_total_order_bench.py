"""Float64 argsort in the default (IEEE) order against `float_order="total"` on one build, timed alternately.

    PYTHONPATH=python python Benchmarks/argsort_total_order_bench.py --csv out.csv [--baseline DIR:DYLIB]

The column is the one `Benchmarks/sort_options_bench.py` sorts: uniform values in [-1e9, 1e9) with a NaN,
a -0.0 and a +0.0 every 1,000 rows. Per row count: an untimed warm-up of at least 100 ms of each order,
the first call of each order after a 500 ms idle (its own column), then `--rounds` rounds alternating the
two orders (and the builds, with `--baseline`), `--reps` calls each. Best and median wall time per call and
the median process CPU time per call (getrusage).
"""
import argparse
import csv
import os
import statistics
import sys

import numpy as np
import pyarrow as pa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_fold_bench import DEFAULT_PKG, first_after_idle, load_engine, reps_of, warm  # noqa: E402

SEED = 20260928


def column(n):
    rng = np.random.default_rng(SEED)
    f64 = rng.random(n) * 2e9 - 1e9
    f64[::1000] = np.nan
    f64[1::1000] = -0.0
    f64[2::1000] = 0.0
    return pa.array(f64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="*", default=[1_000_000, 10_000_000, 50_000_000])
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--baseline", help="PYTHON_DIR:DYLIB of a second build, measured alternately")
    ap.add_argument("--csv")
    args = ap.parse_args()

    under_test_lib = os.environ.get("ARROWMETAL_LIB")
    engines = []
    if args.baseline:
        pkg, _, lib = args.baseline.partition(":")
        engines.append(("before", load_engine(pkg, lib, "arrowmetal_before")))
    engines.append(("after", load_engine(DEFAULT_PKG, under_test_lib, "arrowmetal_after")))
    rows = []
    for n in args.rows:
        col = column(n)
        reps = args.reps if n <= 1_000_000 else max(10, args.reps // 10)
        fns = {}
        for tag, am in engines:
            a = am.MetalArray.from_arrow(col)
            fns[(tag, "ieee")] = (lambda a=a: a.argsort())
            fns[(tag, "total")] = (lambda a=a: a.argsort(float_order="total"))
        keys = list(fns)
        for k in keys:
            warm(fns[k])
        idle = {k: first_after_idle(fns[k]) for k in keys}
        walls = {k: [] for k in keys}
        cpus = {k: [] for k in keys}
        for r in range(args.rounds):
            for k in (keys if r % 2 == 0 else keys[::-1]):
                w, c = reps_of(fns[k], reps)
                walls[k] += w
                cpus[k] += c
        for (tag, order) in keys:
            k = (tag, order)
            row = {"case": f"argsort float64 {order}", "rows": n, "build": tag,
                   "first_after_idle_ms": round(idle[k], 3), "best_ms": round(min(walls[k]), 3),
                   "median_ms": round(statistics.median(walls[k]), 3),
                   "cpu_median_ms": round(statistics.median(cpus[k]), 3), "rounds": args.rounds, "reps": reps}
            rows.append(row)
            print(row, flush=True)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
