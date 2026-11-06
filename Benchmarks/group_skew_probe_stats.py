#!/usr/bin/env python3
"""Sample statistics of the probe on the two families the timings separate (GPU wins: log-normal group sizes;
GPU loses: Zipf, frequent keys + rare tail), same frame generators and seed as Benchmarks/group_skew_bench.py.
Per frame and sample size: d, f1, f2, f3, D2 = n(n-1)/sum c(c-1) (the effective number of keys), the
Chao-Lin estimate, z and ratio of excess singletons over a one-rate Poisson fit. One JSON line per
frame. Usage: PYTHONPATH=python python Benchmarks/group_skew_probe_stats.py ROWS OUT.jsonl"""
import sys, json, math, numpy as np, polars as pl
import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "python")); sys.path.insert(0, HERE)
from arrowmetal import polars_engine as pe
import group_skew_bench as gb
rows = int(sys.argv[1]); out = open(sys.argv[2], "a")
def lam_of(m):
    if m <= 1 + 1e-12: return 0.0
    lo, hi = 1e-9, 2 * m + 10
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mid / (1 - math.exp(-mid)) < m else (lo, mid)
    return (lo + hi) / 2
for fam, gen in (("loses", gb.frames), ("wins", gb.mild_frames)):
    rng = np.random.default_rng(2026)
    for name, k in gen(rows, rng):
        rng.integers(0, 1_000_000_000, rows, dtype=np.int64)       # the bench's value column, to keep the stream aligned
        df = pl.DataFrame({"k": k}); true = df["k"].n_unique()
        prof = []; n = 512
        while n <= min(65_536, rows // 4):
            v = np.sort(pe._sample_tuples(np, df, ["k"], pe._sample_positions(np, rows, n)))
            e = np.flatnonzero(v[1:] != v[:-1]); c = np.diff(np.concatenate(([-1], e, [len(v) - 1])))
            f = np.bincount(c, minlength=4); d, f1, f2, f3 = len(c), int(f[1]), int(f[2]), int(f[3])
            s2 = float((c * (c - 1)).sum()); D2 = n * (n - 1) / s2 if s2 else float("inf")
            est = pe._chao1(d, f1, f2, n, rows)[0]
            lam = lam_of(n / d); e1 = d * lam * math.exp(-lam) / (1 - math.exp(-lam)) if lam else d
            prof.append(dict(n=n, d=d, f1=f1, f2=f2, f3=f3, D2=D2, est=est, z=(f1 - e1) / math.sqrt(e1 + 1), R=f1 / max(e1, 1e-9)))
            n *= 4
        out.write(json.dumps({"rows": rows, "family": fam, "frame": name, "true": true, "profile": prof}) + "\n"); out.flush()
        print(fam, rows, name, true, flush=True)
