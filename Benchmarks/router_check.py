#!/usr/bin/env python3
"""Checks the router's choice: every routed operation timed under `gpu`, `cpu` and `auto`.

For each operation and size the script times the same call three ways through the Python binding,
on Metal-resident arrays (no import inside the timing): pinned to the GPU, pinned to the CPU loop,
and under `auto`, recording which path `auto` took (`am.last_route()`). A row says whether `auto`
picked the faster of the two measured paths; `margin` is slower / faster, so a margin near 1 is a
tie that a rerun can put on either side. int64 columns with 10% nulls, the shape the table was
measured on (Benchmarks/results/router_2026-09-17.json); 300k and 3M sit inside the brackets the
table was fitted across (Benchmarks/router_table.py).

Usage:
    PYTHONPATH=python python Benchmarks/router_check.py --out Benchmarks/results/router_check_<date>.csv
"""
import argparse
import datetime
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa

import arrowmetal as am

SIZES = [1_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000]


def best_us(fn, reps):
    fn()                                           # warm-up: pipelines, pool
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        best = min(best, (time.perf_counter_ns() - t0) / 1e3)
    return best


def cases(n, rng):
    vals = rng.integers(-1000, 1001, n, dtype=np.int64)
    col = am.array(pa.array(vals, mask=rng.random(n) < 0.1))
    other = am.array(pa.array(rng.integers(-1000, 1001, n, dtype=np.int64), mask=rng.random(n) < 0.1))
    mask = col.compare(">", 0)
    keys = am.array(pa.array(rng.integers(0, 1000, n, dtype=np.int32)))
    gb = am.GroupBy(keys, 1000)
    return [
        ("sum(int64)", col.sum),
        ("min(int64)", col.min),
        ("max(int64)", col.max),
        ("compare(int64 > 0)", lambda: col.compare(">", 0)),
        ("compare(int64 > int64)", lambda: col.compare(">", other)),
        ("add(int64, 1)", lambda: col.arith("+", 1)),
        ("add(int64 + int64)", lambda: col.arith("+", other)),
        ("subtract(int64, 1)", lambda: col.arith("-", 1)),
        ("multiply(int64, 3)", lambda: col.arith("*", 3)),
        ("filter(int64, mask)", lambda: col.filter(mask)),
        ("filter_where(int64 > 0)", lambda: col.filter_where(">", 0)),
        ("group-by sum (1000 keys)", lambda: gb.sum(col)),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    cores = os.cpu_count()
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        chip = platform.machine()
    rows = []
    rng = np.random.default_rng(2026)
    for n in sizes:
        reps = 5 if n >= 3_000_000 else 20
        for label, fn in cases(n, rng):
            t = {}
            for mode in ("gpu", "cpu", "auto"):
                with am.router(mode):
                    t[mode] = best_us(fn, reps)
                    d = am.last_route()
                if mode == "auto":
                    auto_path, reason = d.path, d.reason
            faster = "gpu" if t["gpu"] <= t["cpu"] else "cpu"
            margin = max(t["gpu"], t["cpu"]) / max(min(t["gpu"], t["cpu"]), 1e-9)
            rows.append((label, n, t["gpu"], t["cpu"], t["auto"], auto_path, faster,
                         "yes" if auto_path == faster else "no", margin, reason))
            print(f"{label:28s} {n:>10,d}  gpu {t['gpu']:9.1f}  cpu {t['cpu']:9.1f}  auto {t['auto']:9.1f} "
                  f"({auto_path}, faster {faster}, margin {margin:.2f})", flush=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(args.out, "w") as fh:
        fh.write(f"# ArrowMetal router check on {chip}, {cores} CPU cores, {am.device_name()}, best of 20 "
                 f"(5 at 3M rows and above), int64 with 10% nulls, Python binding on resident arrays, {stamp}"
                 + (f"; {args.note}" if args.note else "") + "\n")
        fh.write("op,rows,gpu_us,cpu_us,auto_us,auto_path,faster_path,auto_picked_faster,margin,auto_reason\n")
        for r in rows:
            fh.write(f"\"{r[0]}\",{r[1]},{r[2]:.1f},{r[3]:.1f},{r[4]:.1f},{r[5]},{r[6]},{r[7]},{r[8]:.2f},\"{r[9]}\"\n")
    picked = sum(r[7] == "yes" for r in rows)
    print(f"auto picked the faster path in {picked} of {len(rows)} cases; wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
