"""The sort-family rows that `partition_nth_indices` has to be fixed without disturbing.

A narrow stand-in for `Benchmarks/full_matrix.py`: the same seed, the same column builders and the
same rule (one warm-up, then best of five), but only ArrowMetal and only the five operations whose
timings the partition_nth work could move. It exists so a before/after can be taken in a couple of
minutes without a 40-minute matrix run.

    PYTHONPATH=python python Benchmarks/loss_partition_nth.py            # 10M and 50M
    PYTHONPATH=python python Benchmarks/loss_partition_nth.py --rows 10000000
    PYTHONPATH=python python Benchmarks/loss_partition_nth.py --tag before --csv out.csv

Prints one line per (op, rows) and, with `--csv`, appends them to a CSV that a later run with a
different `--tag` can be diffed against.
"""
import argparse, csv, gc, os, sys, time

import numpy as np
import pyarrow as pa

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
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


def columns(n):
    """i64_nn, f64_nn and the two lexsort keys, drawn in full_matrix.py's order from its seed."""
    rng = np.random.default_rng(SEED)
    i64 = rng.integers(-(2 ** 62), 2 ** 62, size=n, dtype=np.int64)
    f64 = rng.random(n) * 2e9 - 1e9
    lex_a = rng.integers(0, 1000, size=n, dtype=np.int32)
    lex_b = rng.integers(0, 100000, size=n, dtype=np.int32)
    return {"i64": am.array(pa.array(i64)), "f64": am.array(pa.array(f64)),
            "lex_a": am.array(pa.array(lex_a)), "lex_b": am.array(pa.array(lex_b))}


def cases(c, n):
    """The ops in full_matrix.py's own order.

    The order matters and is not cosmetic: every one of these allocates its scratch out of the
    context's buffer pool, so an operation that runs after `argsort` finds the big buffers already
    pooled and one that runs first pays to map them. `partition_nth_indices` is the last row of
    full_matrix's sort family and is measured last here for the same reason.
    """
    return [("argsort int64", lambda: c["i64"].argsort()),
            ("sort float64", lambda: c["f64"].sort()),
            ("top_k (k=100, int64)", lambda: c["i64"].top_k(100)),
            ("select_k_unstable (k=100)", lambda: c["i64"].top_k(100, largest=False)),
            ("lexsort (2 int32 keys)", lambda: am.lexsort_indices([c["lex_a"], c["lex_b"]])),
            ("partition_nth_indices (n/2)", lambda: c["i64"].partition_nth_indices(n // 2))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="*", default=[10_000_000, 50_000_000])
    ap.add_argument("--tag", default="run")
    ap.add_argument("--csv")
    args = ap.parse_args()

    out = []
    for n in args.rows:
        c = columns(n)
        for op, fn in cases(c, n):
            wall, iters = best_of(fn)
            print(f"{args.tag:<8} {op:<30} {n:>11,} {wall:9.3f} ms  ({iters} iters)", flush=True)
            out.append(dict(tag=args.tag, op=op, rows=n, wall_ms=round(wall, 3), iters=iters))
        del c
        gc.collect()

    if args.csv:
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["tag", "op", "rows", "wall_ms", "iters"])
            if new:
                w.writeheader()
            w.writerows(out)


if __name__ == "__main__":
    main()
