#!/usr/bin/env python3
"""Checks the router's choice: every routed operation timed under `gpu`, `cpu` and `auto`.

For each operation and size the script times the same call three ways through the Python binding,
on Metal-resident arrays (no import inside the timing): pinned to the GPU, pinned to the CPU loop,
and under `auto`, recording which path `auto` took (`am.last_route()`). A row says whether `auto`
picked the faster of the two measured paths; `margin` is slower / faster, so a margin near 1 is a
tie that a rerun can put on either side. int64 columns with 10% nulls, the shape the table was
measured on (Benchmarks/results/router_2026-09-17.json); 300k and 3M sit inside the brackets the
table was fitted across (Benchmarks/router_table.py).

The sweep itself lives in the package (python/arrowmetal/_router_calibrate.py), where
`python -m arrowmetal.router calibrate` runs the same grid and fits a per-machine table from it.

Usage:
    PYTHONPATH=python python Benchmarks/router_check.py --out Benchmarks/results/router_check_<date>.csv
"""
import argparse
import datetime
import sys

import arrowmetal as am
from arrowmetal import _router_calibrate as rc

SIZES = rc.FULL_SIZES


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    g = rc.GRIDS["full"]
    machine = rc.machine_info()
    rows = rc.sweep(sizes, g["reps"], g["reps_large"], g["modes"], log=lambda s: print(s, flush=True))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rc.write_check_csv(args.out, rows, rc.check_header(machine, g["reps"], g["reps_large"], stamp, args.note))
    picked = sum(r["auto_picked_faster"] for r in rows)
    print(f"auto picked the faster path in {picked} of {len(rows)} cases; wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
