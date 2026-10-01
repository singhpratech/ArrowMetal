#!/usr/bin/env python3
"""Grouped Float64 aggregates through the array API and the plan runner, one library per process.

Run it once per build with `ARROWMETAL_LIB` pointing at that build's `libArrowMetalC.dylib`, and
alternate the builds (old, new, new, old, ...) to compare them in one session; the rows of both go
into one file and `--label` tells them apart. Recorded in
`Benchmarks/results/groupby_float64_2026-09-30.csv` (per row: best and median over the rounds) and
`…_raw_…` (every round), with the conditions in `…_conditions.txt`.

For each size, group count and null fraction a table is generated with seed 1234: `k` uniform over
the group count, `k1` and `k2` drawn from `[0, G / b) x [0, b)` with `b = min(G, 100)` (two keys over
the same number of possible groups), a Float64 value `v` (standard normal x 1000), and an Int64 and a
Float32 copy for the control rows (`count`, an Int64 `sum`, a Float32 `sum`, measured with one key
and no nulls). "rows/2" groups is half the rows.

* `array`: `getattr(am.group_by(keys), agg)(v)`, a fresh grouping per call, so no work kept on a
  grouping carries over from one call to the next.
* `plan`: `am.scan(table).group_by(keys).agg(...).collect()`, warm (the columns imported once).
* `minmax` is `min_max` on the array API and a `min` and a `max` in one plan.

Per case: an untimed warm-up of at least 100 ms, 500 ms of idle, one run recorded on its own
(`first_ms`), a second warm-up of at least 100 ms, then at least `--reps` timed runs (up to 30 within
`--budget-ms`). Wall time and process CPU time (getrusage) of each.

Usage:
  ARROWMETAL_LIB=<dylib> PYTHONPATH=python python Benchmarks/groupby_float64_bench.py --label new \\
      [--sizes 2000000,10000000,50000000] [--groups 200,10000,100000,1000000,rows/2] [--keys 1,2]
      [--nulls 0,1] [--apis array,plan] [--aggs sum,mean,min,max,minmax,count,i64sum,f32sum]
      [--reps 3] [--budget-ms 400] >> results.csv
Prints: label,rows,groups,keys,nulls,api,agg,first_ms,best_ms,median_ms,cpu_best_ms,cpu_median_ms,reps
"""
import argparse
import resource
import statistics
import time

import numpy as np
import pyarrow as pa

import arrowmetal as am


def cpu_ms():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return (r.ru_utime + r.ru_stime) * 1e3


def frame(rows, g, nulls, seed=1234):
    rng = np.random.default_rng(seed)
    n = max(2, rows // 2) if g == "rows/2" else int(g)
    b = min(n, 100)
    k = rng.integers(0, n, rows, dtype=np.int32)
    k1 = rng.integers(0, max(1, n // b), rows, dtype=np.int32)
    k2 = rng.integers(0, b, rows, dtype=np.int32)
    v = rng.standard_normal(rows) * 1000.0
    vi = rng.integers(-1_000_000, 1_000_000, rows, dtype=np.int64)
    vf = v.astype(np.float32)
    mask = (rng.random(rows) < 0.1) if nulls else None
    return pa.table({"k": k, "k1": k1, "k2": k2,
                     "v": pa.array(v, mask=mask), "vi": pa.array(vi, mask=mask),
                     "vf": pa.array(vf, mask=mask)})


def timeit(fn, reps_min, budget_ms):
    def warm():
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) * 1e3 < 100.0:
            fn()
    warm()
    time.sleep(0.5)
    t = time.perf_counter()
    fn()
    first = (time.perf_counter() - t) * 1e3
    warm()
    walls, cpus = [], []
    t0 = time.perf_counter()
    while len(walls) < reps_min or ((time.perf_counter() - t0) * 1e3 < budget_ms and len(walls) < 30):
        c = cpu_ms()
        t = time.perf_counter()
        fn()
        walls.append((time.perf_counter() - t) * 1e3)
        cpus.append(cpu_ms() - c)
    return first, min(walls), statistics.median(walls), min(cpus), statistics.median(cpus), len(walls)


CONTROLS = ("i64sum", "count", "f32sum")
METHOD = {"sum": "sum", "i64sum": "sum", "f32sum": "sum", "mean": "mean", "min": "min", "max": "max",
          "minmax": "min_max", "count": "count"}


def plan_aggs(agg, col):
    if agg in ("sum", "i64sum", "f32sum"):
        return [am.agg.sum(col, "s")]
    if agg == "mean":
        return [am.agg.mean(col, "s")]
    if agg in ("min", "max"):
        return [getattr(am.agg, agg)(col, "s")]
    if agg == "minmax":
        return [am.agg.min(col, "lo"), am.agg.max(col, "hi")]
    return [am.agg.count("s", am.col(col))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--sizes", default="2000000,10000000,50000000")
    ap.add_argument("--groups", default="200,10000,100000,1000000,rows/2")
    ap.add_argument("--aggs", default="sum,mean,min,max,minmax,i64sum,count,f32sum")
    ap.add_argument("--keys", default="1,2")
    ap.add_argument("--nulls", default="0,1")
    ap.add_argument("--apis", default="array,plan")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--budget-ms", type=float, default=400.0)
    a = ap.parse_args()
    for rows in [int(s) for s in a.sizes.split(",")]:
        for g in a.groups.split(","):
            for nulls in [int(x) for x in a.nulls.split(",")]:
                t = frame(rows, g, nulls)
                cols = {c: am.array(t.column(c).combine_chunks()) for c in ("k", "k1", "k2", "v", "vi", "vf")}
                for nk in [int(x) for x in a.keys.split(",")]:
                    keynames = ["k"] if nk == 1 else ["k1", "k2"]
                    lf = am.scan(t)
                    for api in a.apis.split(","):
                        for agg in a.aggs.split(","):
                            if agg in CONTROLS and (nk != 1 or nulls != 0):
                                continue
                            col = {"i64sum": "vi", "f32sum": "vf"}.get(agg, "v")
                            if api == "array":
                                fn = (lambda m=METHOD[agg], v=cols[col], kc=[cols[c] for c in keynames]:
                                      getattr(am.group_by(kc), m)(v))
                            else:
                                fn = lf.group_by(*keynames).agg(*plan_aggs(agg, col)).collect
                            r = timeit(fn, a.reps, a.budget_ms)
                            print(",".join(str(x) for x in [a.label, rows, g, nk, nulls, api, agg]
                                           + [f"{x:.3f}" for x in r[:5]] + [r[5]]), flush=True)
                    del lf
                del t, cols


if __name__ == "__main__":
    main()
