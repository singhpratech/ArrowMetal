#!/usr/bin/env python3
"""Reads a mixed-type CSV file with ArrowMetal (GPU) and the CPU readers, and records the times.

    PYTHONPATH=python python Benchmarks/csv_bench.py [--rows 1000000,10000000] [--repeat 5]
                                                     [--dir /tmp/am-csv] [--out results.csv] [--keep]

The file has eight columns -- int64 id, int64 qty, float64 price, float64 weight, a string category,
a bool flag, a date and a timestamp -- plus a quoted free-text column with embedded commas every 50th
row, so the structure scan has quotes to track. Readers, all producing typed columns:

    arrowmetal          am.read_csv: GPU-resident columns (MetalArrays), types inferred; the file
                        is pread into a Metal buffer (file_access="read", the default)
    arrowmetal_map      the same with file_access="map": mmap + no-copy wrap
    arrowmetal_table    am.read_csv_table: the same read exported to a pyarrow.Table (zero copy)
    pyarrow             pyarrow.csv.read_csv (multithreaded, the oracle)
    polars              polars.read_csv
    pandas_pyarrow      pandas.read_csv(engine="pyarrow")
    pandas_c            pandas.read_csv (the default C engine)
    duckdb              duckdb read_csv(...).arrow()

Each reader is warmed once, then timed `--repeat` times; the median, min and max wall times are
written with the machine, row count and file size. Three lanes build on the same GPU while this
script is new, so any result it writes is provisional until a quiet rerun.
"""
import argparse
import csv
import datetime
import gc
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import arrowmetal as am        # noqa: E402


def build(path, rows, chunk=1_000_000):
    """Writes the fixture in chunks with numpy + plain string formatting (no reader involved)."""
    rng = np.random.default_rng(23)
    cats = np.array(["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"])
    with open(path, "w", newline="") as fh:
        fh.write("id,qty,price,weight,cat,flag,day,ts,note\n")
        written = 0
        while written < rows:
            n = min(chunk, rows - written)
            base = np.arange(written, written + n, dtype=np.int64)
            qty = rng.integers(0, 1000, n)
            price = np.round(rng.random(n) * 1000.0, 2)
            weight = rng.random(n) * 5.0
            cat = cats[base % 8]
            flag = np.where(base % 3 == 0, "true", "false")
            day = (np.datetime64("2020-01-01") + (base % 3650)).astype(str)
            ts = (np.datetime64("2020-01-01T00:00:00") + base * 37).astype(str)
            note = np.where(base % 50 == 0, '"a, quoted ""note"""', "plain")
            lines = [
                "%d,%d,%s,%r,%s,%s,%s,%s,%s" % row
                for row in zip(base.tolist(), qty.tolist(), price.tolist(), weight.tolist(), cat.tolist(),
                               flag.tolist(), day.tolist(), ts.tolist(), note.tolist())
            ]
            fh.write("\n".join(lines))
            fh.write("\n")
            written += n
    return os.path.getsize(path)


def readers(path):
    import pyarrow.csv as pc
    out = {
        "arrowmetal": lambda: am.read_csv(path),
        "arrowmetal_map": lambda: am.read_csv(path, file_access="map"),
        "arrowmetal_table": lambda: am.read_csv_table(path),
        "pyarrow": lambda: pc.read_csv(path),
    }
    try:
        import polars as pl
        out["polars"] = lambda: pl.read_csv(path)
    except ImportError:
        pass
    try:
        import pandas as pd
        out["pandas_pyarrow"] = lambda: pd.read_csv(path, engine="pyarrow")
        out["pandas_c"] = lambda: pd.read_csv(path)
    except ImportError:
        pass
    try:
        import duckdb
        out["duckdb"] = lambda: duckdb.connect().execute("SELECT * FROM read_csv(?)", [path]).arrow()
    except ImportError:
        pass
    return out


def timed(fn):
    gc.collect()
    t0 = time.perf_counter()
    r = fn()
    t = time.perf_counter() - t0
    del r
    return t


def machine():
    chip = platform.processor()
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True).stdout.strip() or chip
    except OSError:
        pass
    return chip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1000000,10000000")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--dir", default="/tmp/am-csv-bench")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="comma-separated reader names")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    today = datetime.date.today().isoformat()
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                                   "csv_bench_%s_provisional.csv" % today)
    rows_out = []
    chip, device = machine(), am.device_name() if hasattr(am, "device_name") else ""
    for rows in [int(r) for r in args.rows.split(",")]:
        path = os.path.join(args.dir, "mixed_%d.csv" % rows)
        if not os.path.exists(path):
            print("building %s ..." % path, flush=True)
            build(path, rows)
        size = os.path.getsize(path)
        for name, fn in readers(path).items():
            if args.only and name not in args.only.split(","):
                continue
            fn()                                             # warm: shader compile, page cache
            times = [timed(fn) for _ in range(args.repeat)]
            med = statistics.median(times)
            print("%-18s rows=%-9d median %8.1f ms  (min %.1f, max %.1f)  %6.0f MB/s"
                  % (name, rows, med * 1e3, min(times) * 1e3, max(times) * 1e3, size / med / 1e6), flush=True)
            rows_out.append(dict(date=today, machine=chip, device=device, reader=name, rows=rows,
                                 file_bytes=size, repeat=args.repeat, median_ms=round(med * 1e3, 2),
                                 min_ms=round(min(times) * 1e3, 2), max_ms=round(max(times) * 1e3, 2),
                                 mb_per_s=round(size / med / 1e6, 1)))
        if not args.keep:
            os.remove(path)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print("wrote", out)


if __name__ == "__main__":
    main()
