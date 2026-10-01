"""Row-wise kernels on two builds, timed alternately in one process: the grid fold for arrays past 2^32 - 256
elements (every kernel reads its grid position through the fold) against the build before it.

    PYTHONPATH=python python Benchmarks/grid_fold_bench.py \\
        --baseline OLD_PYTHON_DIR:OLD_DYLIB --csv out.csv

Per case and row count: an untimed warm-up of at least 100 ms of the same call on each build, then the first
call after a 500 ms idle on each build (its own column), then `--rounds` rounds that alternate the builds,
each round `--reps` calls per build. Best and median wall time per call, and the median process CPU time per
call (getrusage), per build.
"""
import argparse
import csv
import importlib.util
import os
import resource
import statistics
import sys
import time

import numpy as np
import pyarrow as pa


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


def cpu_s():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def warm(fn, ms=100.0):
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1000.0 < ms:
        fn()


def first_after_idle(fn):
    time.sleep(0.5)
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000.0


def reps_of(fn, reps):
    walls, cpus = [], []
    for _ in range(reps):
        c0, t0 = cpu_s(), time.perf_counter()
        fn()
        walls.append((time.perf_counter() - t0) * 1000.0)
        cpus.append((cpu_s() - c0) * 1000.0)
    return walls, cpus


def cases(am, n, rng_seed=20261001):
    rng = np.random.default_rng(rng_seed)
    f64 = am.MetalArray.from_arrow(pa.array(rng.random(n) * 2e6 - 1e6))
    g64 = am.MetalArray.from_arrow(pa.array(rng.random(n) * 2e6 - 1e6))
    i32 = am.MetalArray.from_arrow(pa.array(rng.integers(-(2 ** 30), 2 ** 30, size=n, dtype=np.int32)))
    i64 = am.MetalArray.from_arrow(pa.array(rng.integers(-(2 ** 62), 2 ** 62, size=n, dtype=np.int64)))
    k32 = am.MetalArray.from_arrow(pa.array(rng.integers(0, 64, size=n, dtype=np.int32)))
    grouped = am.group_by(am.col("k"), 64).aggregate([("sum", "s", am.col("v"))])
    return {
        "add float64 + float64": lambda: f64 + g64,
        "bitwise_and int32 & scalar": lambda: i32.bitwise_and(0x0F0F0F0F),
        "abs int64": lambda: i64.abs(),
        "expr group_by 64 keys sum int32": lambda: am.query({"k": k32, "v": i32}, grouped),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="*", default=[10_000_000, 50_000_000])
    ap.add_argument("--baseline", required=True, help="PYTHON_DIR:DYLIB of the build before")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--only", nargs="*", help="case names to run (default all)")
    ap.add_argument("--csv")
    args = ap.parse_args()

    under_test_lib = os.environ.get("ARROWMETAL_LIB")
    pkg, _, lib = args.baseline.partition(":")
    engines = [("before", load_engine(pkg, lib, "arrowmetal_before")),
               ("after", load_engine(DEFAULT_PKG, under_test_lib, "arrowmetal_after"))]
    rows = []
    for n in args.rows:
        made = {tag: cases(am, n) for tag, am in engines}
        for name in made["after"]:
            if args.only and name not in args.only:
                continue
            fns = {tag: made[tag][name] for tag, _ in engines}
            for tag, _ in engines:
                warm(fns[tag])
            idle = {tag: first_after_idle(fns[tag]) for tag, _ in engines}
            walls = {tag: [] for tag, _ in engines}
            cpus = {tag: [] for tag, _ in engines}
            for r in range(args.rounds):
                order = engines if r % 2 == 0 else engines[::-1]
                for tag, _ in order:
                    w, c = reps_of(fns[tag], args.reps)
                    walls[tag] += w
                    cpus[tag] += c
            for tag, _ in engines:
                row = {"case": name, "rows": n, "build": tag,
                       "first_after_idle_ms": round(idle[tag], 3),
                       "best_ms": round(min(walls[tag]), 3),
                       "median_ms": round(statistics.median(walls[tag]), 3),
                       "cpu_median_ms": round(statistics.median(cpus[tag]), 3),
                       "rounds": args.rounds, "reps": args.reps}
                rows.append(row)
                print(row, flush=True)
            b, a = rows[-2], rows[-1]
            print(f"  {name} {n}: after/before best {a['best_ms'] / b['best_ms']:.3f} "
                  f"median {a['median_ms'] / b['median_ms']:.3f}", flush=True)
        del made
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
