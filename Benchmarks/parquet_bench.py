#!/usr/bin/env python3
"""Reads a large Parquet file with ArrowMetal (GPU), pyarrow, Polars and pandas, and compares them.

    PYTHONPATH=python python Benchmarks/parquet_bench.py [--rows 50000000] [--codecs snappy,lz4,none]
                                                         [--dir /tmp/am-parquet] [--keep]

The table is 8 columns wide -- two int64, one int32, two float64, one dictionary-encoded string, one
timestamp and one boolean -- which is roughly the shape of a fact table. Four numbers are reported per
reader and codec:

    wall ms            elapsed time for the read
    cpu  ms            process CPU time over the same interval (GPU work does not appear here, so a
                       small number next to a large wall time is the point of the exercise)
    MB/s               file bytes divided by wall time
    ttfc ms            time to first compute: read one column and sum it, which is what a query actually
                       needs. ArrowMetal's sum runs on the GPU over the buffers the decode just filled;
                       the CPU readers have to hand their arrays to something else first.
"""
import argparse
import gc
import os
import shutil
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import arrowmetal as am        # noqa: E402

COLUMNS = ["id", "qty", "code", "price", "weight", "cat", "ts", "flag"]


def build(path, rows, codec, chunk=2_000_000):
    """Writes the fixture in chunks so the generator never holds the whole table in memory."""
    schema = pa.schema([
        ("id", pa.int64()), ("qty", pa.int64()), ("code", pa.int32()),
        ("price", pa.float64()), ("weight", pa.float64()),
        ("cat", pa.string()), ("ts", pa.timestamp("us")), ("flag", pa.bool_()),
    ])
    cats = np.array(["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"], dtype=object)
    writer = pq.ParquetWriter(path, schema, compression=codec if codec != "none" else None,
                              use_dictionary=["cat"], data_page_size=1 << 20)
    rng = np.random.default_rng(11)
    written = 0
    while written < rows:
        n = min(chunk, rows - written)
        base = np.arange(written, written + n, dtype=np.int64)
        batch = pa.record_batch([
            pa.array(base),
            pa.array(rng.integers(0, 1000, n).astype(np.int64)),
            pa.array((base % 100000).astype(np.int32)),
            pa.array(rng.random(n) * 1000.0),
            pa.array(rng.random(n) * 5.0),
            pa.array(cats[(base % 8)], pa.string()),
            pa.array(1_600_000_000_000_000 + base * 1_000_003, pa.timestamp("us")),
            pa.array((base % 3 == 0)),
        ], schema=schema)
        writer.write_batch(batch)
        written += n
    writer.close()
    return os.path.getsize(path)


def timed(fn):
    gc.collect()
    c0, w0 = time.process_time(), time.perf_counter()
    out = fn()
    w1, c1 = time.perf_counter(), time.process_time()
    return out, (w1 - w0) * 1000.0, (c1 - c0) * 1000.0


def readers(path):
    r = {
        "arrowmetal (GPU)": lambda: am.read_parquet(path),
        "pyarrow.parquet": lambda: pq.read_table(path),
    }
    try:
        import polars as pl
        r["polars"] = lambda: pl.read_parquet(path)
    except ImportError:
        pass
    try:
        import pandas  # noqa: F401
        r["pandas"] = lambda: pandas.read_parquet(path)
    except ImportError:
        pass
    return r


def first_compute(path):
    """Read one column and sum it: the smallest useful end-to-end query."""
    out = {"arrowmetal (GPU)": lambda: am.read_parquet(path, columns=["price"])["price"].sum(),
           "pyarrow.parquet": lambda: pq.read_table(path, columns=["price"])["price"].to_numpy().sum()}
    try:
        import polars as pl
        out["polars"] = lambda: pl.read_parquet(path, columns=["price"])["price"].sum()
    except ImportError:
        pass
    try:
        import pandas
        out["pandas"] = lambda: pandas.read_parquet(path, columns=["price"])["price"].sum()
    except ImportError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50_000_000)
    ap.add_argument("--codecs", default="snappy,lz4,none")
    ap.add_argument("--dir", default=os.path.join(os.sep, "tmp", "arrowmetal-parquet-bench"))
    ap.add_argument("--keep", action="store_true", help="keep the generated files")
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    print("device: %s" % am.device_name())
    print("rows:   {:,}".format(args.rows))
    rows_out = []
    try:
        for codec in args.codecs.split(","):
            path = os.path.join(args.dir, "bench-%s-%d.parquet" % (codec, args.rows))
            if not os.path.exists(path):
                t0 = time.perf_counter()
                size = build(path, args.rows, codec)
                print("wrote %s (%.2f GB) in %.1f s" % (os.path.basename(path), size / 1e9,
                                                        time.perf_counter() - t0))
            size = os.path.getsize(path)
            mb = size / 1e6
            print("\n=== %s, %.2f GB on disk ===" % (codec, size / 1e9))
            print("%-20s %10s %10s %10s %10s" % ("reader", "wall ms", "cpu ms", "MB/s", "ttfc ms"))
            fc = first_compute(path)
            for name, fn in readers(path).items():
                best_w = best_c = None
                for _ in range(args.repeat):
                    out, w, c = timed(fn)
                    del out
                    if best_w is None or w < best_w:
                        best_w, best_c = w, c
                tt = None
                for _ in range(args.repeat):
                    _, w, _c = timed(fc[name])
                    tt = w if tt is None else min(tt, w)
                print("%-20s %10.0f %10.0f %10.0f %10.0f" % (name, best_w, best_c, mb / (best_w / 1000.0), tt))
                rows_out.append((codec, name, best_w, best_c, mb / (best_w / 1000.0), tt))
    finally:
        if not args.keep:
            shutil.rmtree(args.dir, ignore_errors=True)

    print("\nmarkdown:")
    print("| codec | reader | wall ms | CPU ms | MB/s | time to first compute (ms) |")
    print("|---|---|---:|---:|---:|---:|")
    for r in rows_out:
        print("| %s | %s | %.0f | %.0f | %.0f | %.0f |" % r)


if __name__ == "__main__":
    main()
