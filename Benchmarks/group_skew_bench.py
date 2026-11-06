#!/usr/bin/env python3
"""Group-bys over skewed keys: which engine is faster, and what the default chose.

For each frame size and each skewed key distribution (Zipf at several exponents; 200 frequent keys
with a tail of unique rows or of small keys; the tail's share of rows varies), a group-by over one
int64 key with sum, min + max, mean or count of an int64 column runs through Polars' in-memory and
streaming engines, through `MetalEngine(shapes="all", min_rows=0)` (every shape taken: the GPU's
time) and through `MetalEngine()` (the default, which decides from its group-count probe), best of
`--iters` after one warm-up, the caches cleared before every default run so each one probes. Every
engine's answer is checked against Polars in-memory (sorted by key). `gpu_x` is the faster Polars
engine's time over the GPU's; `table_takes_true` is what the default's crossover table decides at the
true group count; `default_took`, `default_estimate` and `default_how` are what the default did.

`--set mild` runs keys whose group sizes vary without a long tail instead: 10,000, 100,000 and
1,000,000 keys, each weighted by a log-normal draw (sigma 0.5, 1.0, 1.5).

Usage:
  PYTHONPATH=python python Benchmarks/group_skew_bench.py [--sizes 10000000,50000000] [--set skewed|mild] [--out CSV]
"""
import argparse
import csv
import sys

import numpy as np
import polars as pl

import arrowmetal as am
from arrowmetal import _engine_policy as policy
from arrowmetal import polars_engine as pe

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from polars_engine_bench import best_of  # noqa: E402

AGGS = {
    "sum": ("group_by:sum", lambda: [pl.col("q").sum().alias("s")]),
    "minmax": ("group_by:minmax", lambda: [pl.col("q").min().alias("lo"), pl.col("q").max().alias("hi")]),
    "mean": ("group_by:mean", lambda: [pl.col("q").mean().alias("m")]),
    "count": ("group_by:count", lambda: [pl.len().alias("n")]),
}


def mild_frames(rows, rng):
    """Group sizes that vary without a long tail: each of G keys weighted by a log-normal draw."""
    for g in (10_000, 100_000, 1_000_000):
        for sigma in (0.5, 1.0, 1.5):
            w = rng.lognormal(0.0, sigma, g)
            yield (f"{g:,} keys, log-normal sizes s={sigma}",
                   rng.choice(g, size=rows, p=w / w.sum()).astype(np.int64))


def frames(rows, rng):
    for a in (1.1, 1.3, 1.6, 2.0):
        yield f"zipf a={a}", rng.zipf(a, rows).astype(np.int64)
    for frac in (0.01, 0.05, 0.2):
        tail = rng.random(rows) < frac
        yield (f"200 frequent + {frac:.0%} unique rows",
               np.where(tail, 10_000_000 + np.arange(rows), rng.integers(0, 200, rows)).astype(np.int64))
    for c in (3, 100):
        tail = rng.random(rows) < 0.05
        yield (f"200 frequent + 5% rows in keys of {c}",
               np.where(tail, 10_000_000 + np.arange(rows) // c, rng.integers(0, 200, rows)).astype(np.int64))


def cold():
    pe.clear_group_estimates()
    pe.clear_import_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="10000000,50000000")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", default="skewed", choices=("skewed", "mild"))
    args = ap.parse_args()
    header = (f"ArrowMetal {am.version()} on {am.device_name()}, polars {pl.__version__} "
              f"({pl.thread_pool_size()} threads), best of {args.iters} after one warm-up")
    print(header, flush=True)
    every = am.MetalEngine(shapes="all", min_rows=0)
    default = am.MetalEngine()
    out = []
    for rows in [int(s) for s in args.sizes.split(",")]:
        rng = np.random.default_rng(2026)
        for name, k in (frames if args.set == "skewed" else mild_frames)(rows, rng):
            df = pl.DataFrame({"k": k, "q": rng.integers(0, 1_000_000_000, rows, dtype=np.int64)})
            true = df["k"].n_unique()
            for agg, (cls, exprs) in AGGS.items():
                lf = df.lazy().group_by("k").agg(exprs())
                ref = lf.collect(engine="in-memory").sort("k")
                for eng in (every, default):
                    cold()
                    got = lf.collect(engine=eng).sort("k")
                    assert got.equals(ref), f"{name} {agg}: answer differs"
                t_mem = best_of(lambda: lf.collect(engine="in-memory"), args.iters)[0]
                t_str = best_of(lambda: lf.collect(engine="streaming"), args.iters)[0]
                t_gpu = best_of(lambda: lf.collect(engine=every), args.iters, cold)[0]
                t_def = best_of(lambda: lf.collect(engine=default), args.iters, cold)[0]
                cold()
                lf.collect(engine=default)
                rep = default.last_report
                probe = [g for g in rep.groups if g["keys"] == ["k"]]
                cpu = min(t_mem, t_str)
                row = {"rows": rows, "frame": name, "agg": agg, "true_groups": true,
                       "true_bucket": policy.group_bucket(true, rows) or "",
                       "polars_in_memory_ms": f"{t_mem:.3f}", "polars_streaming_ms": f"{t_str:.3f}",
                       "metal_all_ms": f"{t_gpu:.3f}", "gpu_x": f"{cpu / t_gpu:.2f}",
                       "table_takes_true": policy.decide([cls], [pl.Int64, pl.Int64], rows, "memory",
                                                         groups=true).take,
                       "default_took": bool(rep.taken), "metal_default_ms": f"{t_def:.3f}",
                       "default_x": f"{cpu / t_def:.2f}",
                       "default_estimate": probe[0]["estimate"] if probe else "",
                       "default_how": probe[0]["how"] if probe else (rep.fallbacks[0] if rep.fallbacks else "")}
                out.append(row)
                print(f"  {rows:>11,} {name:<36} {agg:<6} true {true:>10,} est {str(row['default_estimate']):>10} "
                      f"polars {cpu:9.3f} gpu {t_gpu:9.3f} ({row['gpu_x']}x) table@true={row['table_takes_true']} "
                      f"default took={row['default_took']} {t_def:9.3f} ({row['default_x']}x)", flush=True)
            del df, k
    if args.out:
        with open(args.out, "w", newline="") as fh:
            fh.write("# " + header + "\n")
            w = csv.DictWriter(fh, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
