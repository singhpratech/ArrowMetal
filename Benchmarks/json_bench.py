#!/usr/bin/env python3
"""Reads a newline-delimited JSON file with ArrowMetal (GPU), pyarrow, Polars, pandas and DuckDB.

    PYTHONPATH=python python Benchmarks/json_bench.py [--rows 1000000,10000000] [--repeat 3]
                                                      [--shapes flat,nested] [--readers arrowmetal,...]
                                                      [--dir /tmp/arrowmetal-json-bench] [--keep]
                                                      [--out Benchmarks/results/json_bench_<date>.csv]

The fixture is seven fields per record -- an int64 id, a short string, a double, a boolean, an
ISO-8601 timestamp string, an int64 with 10% nulls and a key that is missing from 5% of the records --
which is roughly the shape of an event log (`flat`). The `nested` shape adds a struct of two doubles
and a list of zero to three strings to every record.

Every reader produces a table in memory:

    arrowmetal         am.read_json: the columns stay in Metal shared memory, ready for GPU compute
    arrowmetal-table   am.read_json_table: the same columns exported to a pyarrow.Table (zero copy)
    pyarrow            pyarrow.json.read_json (all cores)
    polars             polars.read_ndjson
    pandas             pandas.read_json(lines=True)
    duckdb             SELECT * FROM read_json(path, format='newline_delimited') as a pyarrow.Table

Reported per reader and size: the median and the minimum wall time over --repeat runs (after one
warm-up run), the process CPU time of the median run, and file megabytes per second at the median.
Before timing, the ArrowMetal table is checked against pyarrow's single-threaded read of the same file.
"""
import argparse
import csv
import datetime
import gc
import os
import platform
import random
import shutil
import statistics
import sys
import time

import pyarrow as pa
import pyarrow.json as pj

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import arrowmetal as am        # noqa: E402

WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]


def build(path, rows, nested, chunk=200_000):
    rng = random.Random(11)
    with open(path, "w") as f:
        for lo in range(0, rows, chunk):
            out = []
            for i in range(lo, min(rows, lo + chunk)):
                parts = ['"id":%d' % i,
                         '"user":"%s_%d"' % (WORDS[i % 8], rng.randrange(1000)),
                         '"score":%.6f' % (rng.random() * 1000.0),
                         '"ok":%s' % ("true" if rng.random() < 0.5 else "false"),
                         '"ts":"2024-%02d-%02dT%02d:%02d:%02d"' % (rng.randrange(1, 13), rng.randrange(1, 29),
                                                                   rng.randrange(24), rng.randrange(60), rng.randrange(60)),
                         '"n":%s' % ("null" if rng.random() < 0.1 else str(rng.randrange(-100000, 100000)))]
                if rng.random() >= 0.05:
                    parts.append('"opt":%d' % rng.randrange(10))
                if nested:
                    parts.append('"geo":{"lat":%.4f,"lon":%.4f}' % (rng.uniform(-90, 90), rng.uniform(-180, 180)))
                    parts.append('"tags":[%s]' % ",".join('"t%d"' % rng.randrange(20) for _ in range(rng.randrange(4))))
                out.append("{" + ",".join(parts) + "}\n")
            f.write("".join(out))
    return os.path.getsize(path)


def cpu_seconds():
    return time.process_time()


def readers():
    rs = {
        "arrowmetal": lambda p: am.read_json(p),
        "arrowmetal-table": lambda p: am.read_json_table(p),
        "pyarrow": lambda p: pj.read_json(p),
    }
    try:
        import polars as pl
        rs["polars"] = lambda p: pl.read_ndjson(p)
    except ImportError:
        pass
    try:
        import pandas as pd
        rs["pandas"] = lambda p: pd.read_json(p, lines=True)
    except ImportError:
        pass
    try:
        import duckdb
        def duck(p):
            # .arrow() hands back a lazy RecordBatchReader; materialise the table like the others.
            rel = duckdb.sql("SELECT * FROM read_json('%s', format='newline_delimited')" % p)
            return rel.to_arrow_table() if hasattr(rel, "to_arrow_table") else rel.fetch_arrow_table()
        rs["duckdb"] = duck
    except ImportError:
        pass
    return rs


def measure(fn, path, repeat):
    fn(path)                                   # warm-up: page cache, pipelines, thread pools
    walls, cpus = [], []
    for _ in range(repeat):
        gc.collect()
        c0, t0 = cpu_seconds(), time.perf_counter()
        out = fn(path)
        t1, c1 = time.perf_counter(), cpu_seconds()
        del out
        walls.append((t1 - t0) * 1e3)
        cpus.append((c1 - c0) * 1e3)
    order = sorted(range(repeat), key=lambda i: walls[i])
    mid = order[len(order) // 2]
    return walls[mid], min(walls), cpus[mid]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1000000,10000000")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--dir", default=os.path.join(os.sep, "tmp", "arrowmetal-json-bench"))
    ap.add_argument("--keep", action="store_true", help="keep the generated files")
    ap.add_argument("--shapes", default="flat", help="comma-separated: flat, nested")
    ap.add_argument("--readers", default="", help="comma-separated subset of readers to run")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                                                  "json_bench_%s.csv" % datetime.date.today().isoformat()))
    args = ap.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    rs = readers()
    if args.readers:
        rs = {k: v for k, v in rs.items() if k in args.readers.split(",")}
    versions = {"pyarrow": pa.__version__}
    for mod in ("polars", "pandas", "duckdb"):
        try:
            versions[mod] = __import__(mod).__version__
        except ImportError:
            pass
    header = "# json_bench.py, device %s, %s, python %s, %s, loadavg at start %.1f, %s" % (
        am.device_name(), ", ".join("%s %s" % kv for kv in versions.items()), platform.python_version(),
        platform.mac_ver()[0] and "macOS " + platform.mac_ver()[0] or platform.platform(),
        os.getloadavg()[0], datetime.datetime.now().isoformat(timespec="seconds"))
    print(header)
    rows_out = []
    try:
        for shape, rows in [(sh, int(r)) for sh in args.shapes.split(",") for r in args.rows.split(",")]:
            nested = shape == "nested"
            path = os.path.join(args.dir, "events_%d_%s.jsonl" % (rows, shape))
            if not os.path.exists(path):
                build(path, rows, nested)
            size = os.path.getsize(path)
            got = am.read_json_table(path)
            want = pj.read_json(path, read_options=pj.ReadOptions(use_threads=False))
            if not got.equals(want):
                raise SystemExit("arrowmetal and pyarrow disagree on %s" % path)
            del got, want
            print("%s, %d rows, %.1f MB" % (shape, rows, size / 1e6))
            for name, fn in rs.items():
                try:
                    med, best, cpu = measure(fn, path, args.repeat)
                    note = ""
                except Exception as e:                      # recorded, never dropped
                    med = best = cpu = float("nan")
                    note = "%s: %s" % (type(e).__name__, str(e)[:120])
                mbs = size / 1e6 / (med / 1e3) if med == med else float("nan")
                print("  %-17s median %9.1f ms  min %9.1f ms  cpu %9.1f ms  %8.1f MB/s %s" % (name, med, best, cpu, mbs, note))
                rows_out.append([rows, size, name, "%.2f" % med, "%.2f" % best, "%.2f" % cpu, "%.1f" % mbs,
                                 args.repeat, shape, note])
    finally:
        if not args.keep:
            shutil.rmtree(args.dir, ignore_errors=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        f.write(header + "\n")
        w = csv.writer(f)
        w.writerow(["rows", "bytes", "reader", "wall_ms_median", "wall_ms_min", "cpu_ms", "mb_per_s", "repeat", "shape", "note"])
        w.writerows(rows_out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
