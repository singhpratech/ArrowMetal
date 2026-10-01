"""Kernels whose loops and block ends are saturated at the end of the array (64-bit block ends, steps that
stop at `n`, a word count without `n + 31`, 64-bit byte positions in the Parquet decode), on two builds,
timed alternately in one process against the build before.

    PYTHONPATH=python python Benchmarks/index_wrap_bench.py \\
        --baseline OLD_PYTHON_DIR:OLD_DYLIB --csv out.csv

One call per touched kernel family. Per case and row count: an untimed warm-up of at least 100 ms of the
same call on each build, then the first call after a 500 ms idle on each build (its own column), then
`--rounds` rounds that alternate the builds, each round `--reps` calls per build (fewer for calls over
20 ms, so a round stays near a second). Best and median wall time per call, and the median process CPU
time per call (getrusage), per build.
"""
import argparse
import csv
import importlib.util
import os
import resource
import statistics
import sys
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


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


class Data:
    """Inputs shared by both builds (pyarrow arrays and Parquet files); each build imports its own copy."""

    def __init__(self, n, tmp, seed=20261001):
        rng = np.random.default_rng(seed)
        self.n = n
        self.i32 = pa.array(rng.integers(-(2 ** 30), 2 ** 30, size=n, dtype=np.int32))
        self.i32_nulls = pa.array(rng.integers(-(2 ** 30), 2 ** 30, size=n, dtype=np.int32),
                                  mask=rng.random(n) < 0.05)
        self.i64 = pa.array(rng.integers(-1000, 1000, size=n, dtype=np.int64))
        self.f32 = pa.array(rng.random(n, dtype=np.float32))
        self.f64 = pa.array(rng.random(n) * 2e6 - 1e6)
        self.b1 = pa.array(rng.random(n) < 0.5)
        self.b2 = pa.array(rng.random(n) < 0.5)
        self.k64 = pa.array(rng.integers(0, 64, size=n, dtype=np.int32))
        self.k1000 = pa.array(rng.integers(0, 1000, size=n, dtype=np.int32))
        self.dec = pa.array(rng.integers(-10 ** 9, 10 ** 9, size=n, dtype=np.int32)).cast(pa.decimal128(18, 2))
        self.dkey = pa.array(rng.integers(0, 1000, size=n, dtype=np.int32)).cast(pa.decimal128(18, 2))
        words = pa.array([f"value-{i:06d}-padding" for i in range(1000)])
        self.sview = words.take(pa.array(rng.integers(0, 1000, size=n))).cast(pa.string_view())
        self.stream = pa.table({"k": pa.array(rng.integers(0, 1_000_000, size=n, dtype=np.int64)), "v": self.f64})
        self.pq_dict = os.path.join(tmp, f"dict_{n}.parquet")
        self.pq_plain = os.path.join(tmp, f"plain_{n}.parquet")
        pq.write_table(pa.table({"v": pa.array(rng.integers(0, 1000, size=n, dtype=np.int64))}),
                       self.pq_dict, use_dictionary=True, compression="none")
        pq.write_table(pa.table({"v": self.i64}), self.pq_plain, use_dictionary=False, compression="none")


def cases(am, d):
    M = am.MetalArray.from_arrow
    i32, i32n, i64 = M(d.i32), M(d.i32_nulls), M(d.i64)
    f32, f64 = M(d.f32), M(d.f64)
    b1, b2 = M(d.b1), M(d.b2)
    k64, k1000 = M(d.k64), M(d.k1000)
    dec, dkey = M(d.dec), M(d.dkey)
    g1000 = am.group_by([k1000])
    reduce_q = am.Query().sum(am.col("v"))
    return {
        "sum int32 (reduce)": lambda: i32.sum(),
        "and boolean (bitmap words)": lambda: b1 & b2,
        "min_max int32 (agg)": lambda: i32.min_max(),
        "variance float64 (agg)": lambda: f64.variance(),
        "skew float32 (moments)": lambda: f32.skew(),
        "sum decimal128": lambda: dec.sum(),
        "expr sum int32 (fused reduce)": lambda: am.query({"v": i32}, reduce_q),
        "group_by 64 keys sum int32 (atomic)": lambda: k64.group_by(64).sum(i32),
        # A group-by keeps the aggregates and the group order it computed, so these build it afresh per call
        # (the key mapping is part of the time) ...
        "group_by 1000 keys min_max int32 (extrema)": lambda: am.group_by([k1000]).min_max(i32),
        "group_by 1000 keys sum float64 (exact)": lambda: am.group_by([k1000]).sum(f64),
        "group_by 1000 keys variance int64 (group order, moments)": lambda: am.group_by([k1000]).variance(i64),
        # ... and these reuse one, so its group order is built once and the per-group kernels are timed.
        "group_by 1000 keys product int64 (segmented)": lambda: g1000.product(i64),
        "group_by 1000 keys mean float32 (segmented reduce)": lambda: g1000.mean(f32),
        "group_by decimal keys count (limb gather)": lambda: am.group_by([dkey]).count_all(),
        "argsort int32 (radix)": lambda: i32.argsort(),
        "argsort int32 with nulls (partition)": lambda: i32n.argsort(),
        "partition_nth_indices int32": lambda: i32.partition_nth_indices(d.n // 2),
        "top_k 10 float64": lambda: f64.top_k(10),
        "top_k 5000 float64": lambda: f64.top_k(5000),
        "utf8_view import (byte total)": lambda: M(d.sview),
        "stream group_by 1M keys sum float64": lambda: am.scan_table(d.stream).group_by("k").agg([("sum", "v", "s")]),
        "read_parquet int64 dictionary": lambda: am.read_parquet(d.pq_dict, dictionary=False),
        "read_parquet int64 plain": lambda: am.read_parquet(d.pq_plain),
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
    rows = []
    tmp = tempfile.mkdtemp(prefix="index_wrap_bench_")
    for n in args.rows:
        data = Data(n, tmp)
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
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
