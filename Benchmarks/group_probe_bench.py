#!/usr/bin/env python3
"""The MetalEngine's group-count probe: its estimate against the true group count, and its cost
against the group-by it decides.

For each frame size and each group count of the group-by grid (`polars_engine_bench.py`
`GRID_GROUPS`), over one int32 key, two int32 keys and one String key, a group-by sum of an int64
column is collected through `MetalEngine()` `--probe-iters` times with its caches cleared before each
(`clear_group_estimates()`, `clear_import_cache()`), and the probe's own time is read from the
report (`last_report.groups`): `probe_us` is the median, `probe_min_us` the fastest. The same
group-by runs through Polars' in-memory and streaming engines and through `MetalEngine()` cold, best
of `--gb-iters`; `probe_pct` is the median probe as a percentage of the faster of those three.
`estimate` is the probe's answer in the engine (sampled until the decision is settled);
`free_estimate` is its answer with no decision to settle (`polars_engine._frame_groups` alone,
sampling until its denominator reaches 16), against `true_groups`.

Usage:
  PYTHONPATH=python python Benchmarks/group_probe_bench.py [--sizes 2000000,50000000] [--out CSV]
"""
import argparse
import csv
import os
import statistics
import sys

import numpy as np
import polars as pl

import arrowmetal as am
from arrowmetal import _engine_policy as policy
from arrowmetal import polars_engine as pe

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from polars_engine_bench import GRID_GROUPS, best_of  # noqa: E402


def frame(rows, g, rng):
    n = max(2, rows // 2) if g == "rows/2" else g
    b = min(n, 100)
    return pl.DataFrame({"k": rng.integers(0, n, rows, dtype=np.int32),
                         "k1": rng.integers(0, max(1, n // b), rows, dtype=np.int32),
                         "k2": rng.integers(0, b, rows, dtype=np.int32),
                         "s": pl.Series(rng.integers(0, n, rows)).cast(pl.String),
                         "q": rng.integers(0, 1_000_000_000, rows, dtype=np.int64)})


def cold():
    pe.clear_group_estimates()
    pe.clear_import_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="2000000,50000000")
    ap.add_argument("--probe-iters", type=int, default=9)
    ap.add_argument("--gb-iters", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    header = (f"ArrowMetal {am.version()} on {am.device_name()}, polars {pl.__version__} "
              f"({pl.thread_pool_size()} threads), probe median of {args.probe_iters} cold "
              f"collects, group-by best of {args.gb_iters}")
    print(header)
    out = []
    engine = am.MetalEngine()
    for rows in [int(s) for s in args.sizes.split(",")]:
        rng = np.random.default_rng(1234)
        for g in GRID_GROUPS:
            df = frame(rows, g, rng)
            for cols in (["k"], ["k1", "k2"], ["s"]):
                true = df.select(cols).n_unique()
                free, free_how, *_ = pe._frame_groups(df, cols)
                lf = df.lazy().group_by(cols).agg(pl.col("q").sum())
                probes = []
                for _ in range(args.probe_iters):
                    cold()
                    lf.collect(engine=engine)
                    probes += [x for x in engine.last_report.groups if x["keys"] == cols]
                taken = bool(engine.last_report.taken)
                t_mem = best_of(lambda: lf.collect(engine="in-memory"), args.gb_iters)[0]
                t_str = best_of(lambda: lf.collect(engine="streaming"), args.gb_iters)[0]
                t_def = best_of(lambda: lf.collect(engine=engine), args.gb_iters, cold)[0]
                fastest = min(t_mem, t_str, t_def)
                row = {"rows": rows, "grid_groups": g, "keys": "+".join(cols),
                       "dtype": "String" if cols == ["s"] else "int32", "true_groups": true,
                       "true_bucket": policy.group_bucket(true, rows) or "",
                       "free_estimate": free, "free_how": free_how,
                       "free_bucket": policy.group_bucket(free, rows) or "",
                       "estimate": probes[0]["estimate"] if probes else "",
                       "how": probes[0]["how"] if probes else "no probe (no bucket taken at these rows)",
                       "default_took": taken,
                       "probe_us": f"{statistics.median(p['seconds'] for p in probes) * 1e6:.1f}"
                                   if probes else "",
                       "probe_min_us": f"{min(p['seconds'] for p in probes) * 1e6:.1f}"
                                       if probes else "",
                       "polars_in_memory_ms": f"{t_mem:.3f}", "polars_streaming_ms": f"{t_str:.3f}",
                       "metal_default_cold_ms": f"{t_def:.3f}",
                       "probe_pct": (f"{100 * statistics.median(p['seconds'] for p in probes) * 1e3 / fastest:.2f}"
                                     if probes else "")}
                out.append(row)
                print(f"  {rows:>11,} {str(g):>9} {row['keys']:<6} true {true:>11,} "
                      f"free {free:>11,} ({row['true_bucket']}/{row['free_bucket']}) "
                      f"engine {str(row['estimate']):>11} probe {row['probe_us']:>8} us "
                      f"group-by {fastest:9.3f} ms {row['probe_pct']:>6}% took={taken}", flush=True)
            del df
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
