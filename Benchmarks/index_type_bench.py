"""Index arrays as UInt32 against the build where they were Int32, timed alternately in one process.

    PYTHONPATH=python python Benchmarks/index_type_bench.py \\
        --baseline OLD_PYTHON_DIR:OLD_DYLIB --csv out.csv

The calls that return or consume row numbers: argsort of int64, float64 and utf8 columns, top-k,
partition_nth_indices, the lexsort, rank, the group-by keys (dense ids plus the representative rows),
take through each build's own argsort, and the hash join's index pairs. Same method as
`index_wrap_bench.py`: per case and row count an untimed warm-up of at least 100 ms of the same call on
each build, the first call after a 500 ms idle on each build (its own column), then `--rounds` rounds
that alternate the builds, `--reps` calls per build per round (fewer for slow calls, so a round stays
near a second). Best and median wall time per call, and the median process CPU time per call
(getrusage), per build.
"""
import argparse
import csv
import os
import statistics
import sys

import numpy as np
import pyarrow as pa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from index_wrap_bench import DEFAULT_PKG, first_after_idle, load_engine, reps_of, warm  # noqa: E402


class Data:
    """Inputs shared by both builds; each build imports its own copy."""

    def __init__(self, n, seed=20261001):
        rng = np.random.default_rng(seed)
        self.n = n
        self.i64 = pa.array(rng.integers(-(2 ** 40), 2 ** 40, size=n, dtype=np.int64))
        self.f64 = pa.array(rng.random(n) * 2e6 - 1e6, mask=rng.random(n) < 0.05)
        self.i32 = pa.array(rng.integers(-(2 ** 30), 2 ** 30, size=n, dtype=np.int32))
        self.k1000 = pa.array(rng.integers(0, 1000, size=n, dtype=np.int32))
        words = pa.array([f"key-{i:07d}-{(i * 7919) % 10007:05d}" for i in range(100_000)])
        self.utf8 = words.take(pa.array(rng.integers(0, 100_000, size=n)))
        self.left = pa.array(rng.integers(0, n, size=n, dtype=np.int64))
        self.right = pa.array(rng.permutation(n // 4).astype(np.int64))


def cases(am, d):
    M = am.MetalArray.from_arrow
    i64, f64, i32, k1000 = M(d.i64), M(d.f64), M(d.i32), M(d.k1000)
    utf8 = M(d.utf8)
    left, right = M(d.left), M(d.right)
    order = i64.argsort()                       # this build's own index array, int32 or uint32
    return {
        "argsort int64": lambda: i64.argsort(),
        "argsort float64 (5% nulls)": lambda: f64.argsort(),
        "argsort utf8": lambda: utf8.argsort(),
        "top_k 10 int64": lambda: i64.top_k(10),
        "top_k 5000 float64": lambda: f64.top_k(5000),
        "partition_nth_indices int64": lambda: i64.partition_nth_indices(d.n // 2),
        "lexsort (int32, float64)": lambda: am.lexsort_indices([k1000, f64]),
        "rank int32": lambda: i32.rank(),
        "group_by keys (1000 groups)": lambda: am.group_by([k1000]).keys(),
        "take int64 by argsort": lambda: i64.take(order),
        "join indices int64 (n x n/4)": lambda: am.join(left, right),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="*", default=[10_000_000, 50_000_000])
    ap.add_argument("--baseline", required=True, help="PYTHON_DIR:DYLIB of the build before")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--only", nargs="*", help="case names to run (default all)")
    ap.add_argument("--csv")
    args = ap.parse_args()

    under_test_lib = os.environ.get("ARROWMETAL_LIB")
    pkg, _, lib = args.baseline.partition(":")
    engines = [("before", load_engine(pkg, lib, "arrowmetal_before")),
               ("after", load_engine(DEFAULT_PKG, under_test_lib, "arrowmetal_after"))]
    for _, am in engines:
        am.set_router("gpu")
    rows = []
    for n in args.rows:
        data = Data(n)
        made = {tag: cases(am, data) for tag, am in engines}
        for name in made["after"]:
            if args.only and name not in args.only:
                continue
            fns = {tag: made[tag][name] for tag, _ in engines}
            for tag, _ in engines:
                warm(fns[tag])
            idle = {tag: first_after_idle(fns[tag]) for tag, _ in engines}
            probe, _ = reps_of(fns["after"], 3)
            reps = max(5, min(args.reps, int(1000.0 / max(statistics.median(probe), 0.01))))
            walls = {tag: [] for tag, _ in engines}
            cpus = {tag: [] for tag, _ in engines}
            for r in range(args.rounds):
                order = engines if r % 2 == 0 else engines[::-1]
                for tag, _ in order:
                    w, c = reps_of(fns[tag], reps)
                    walls[tag] += w
                    cpus[tag] += c
            for tag, _ in engines:
                row = {"case": name, "rows": n, "build": tag,
                       "first_after_idle_ms": round(idle[tag], 3),
                       "best_ms": round(min(walls[tag]), 3),
                       "median_ms": round(statistics.median(walls[tag]), 3),
                       "cpu_median_ms": round(statistics.median(cpus[tag]), 3),
                       "rounds": args.rounds, "reps": reps}
                rows.append(row)
                print(row, flush=True)
            b, a = rows[-2], rows[-1]
            print(f"  {name} {n}: after/before best {a['best_ms'] / b['best_ms']:.3f} "
                  f"median {a['median_ms'] / b['median_ms']:.3f}", flush=True)
        del made, data
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
