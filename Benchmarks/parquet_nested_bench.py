#!/usr/bin/env python3
"""Nested Parquet reads: ArrowMetal (GPU) against pyarrow, Polars and DuckDB.

    PYTHONPATH=python python Benchmarks/parquet_nested_bench.py [--rows 1000000,10000000] [--repeat 3]
                                                                [--dir /tmp/am-nested] [--out results.csv]

One file per row count, written by pyarrow (Snappy, 1 MB pages, 1M-row row groups), holding one column
of each nested shape the reader reassembles:

    s      struct<a: int64, b: string, c: float64>      (nullable struct, nullable members)
    l      list<int64>                                  (the one-level path)
    ll     list<list<int32>>
    m      map<string, int64>
    ls     list<struct<x: int32, y: float64>>

Each shape is read on its own (a one-column projection) and all five together. Per reader, the best of
`--repeat` runs of:

    wall ms    elapsed time of the read
    cpu ms     process CPU time over the same interval (GPU work does not appear in it)

ArrowMetal is timed on a file handle kept open (`am.ParquetFile(path).read(columns=...)`), returning
GPU-resident arrays; pyarrow is `pq.read_table`, Polars `pl.read_parquet`, and DuckDB
`SELECT ... FROM read_parquet(...)` fetched as an Arrow table. The first ArrowMetal read of each shape is
also checked against pyarrow's table (`match` column).

The integers are random over their full range on purpose. Sequential integers, or 64-bit values whose
high bytes are all zero, compress into long Snappy token streams, and the 1 MB dictionary page pyarrow
writes before falling back to PLAIN is then one serial stream for one SIMD group -- the Snappy weak spot
docs/PARQUET.md describes. That costs the same with or without nesting (a flat column of such values
shows it too), so it would measure decompression rather than reassembly.
"""
import argparse
import csv
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

SHAPES = {"struct": ["s"], "list": ["l"], "list<list>": ["ll"], "map": ["m"], "list<struct>": ["ls"],
          "all": ["s", "l", "ll", "m", "ls"]}


def offsets_for(lengths):
    off = np.zeros(len(lengths) + 1, dtype=np.int32)
    np.cumsum(lengths, out=off[1:])
    return pa.array(off)


def build_chunk(rng, start, n):
    i = np.arange(start, start + n, dtype=np.int64)
    valid = (i % 11) != 0
    words = pa.array(["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"])
    names = pa.DictionaryArray.from_arrays(pa.array((i % 8).astype(np.int32)), words).cast(pa.string())
    s = pa.StructArray.from_arrays(
        [pa.array(rng.integers(-(1 << 62), 1 << 62, n), mask=(i % 7) == 3), names, pa.array(rng.random(n))],
        names=["a", "b", "c"],
        mask=pa.array(~valid))
    # list<int64>: 0-3 elements.
    ln = (i % 4).astype(np.int32)
    l = pa.ListArray.from_arrays(offsets_for(ln), pa.array(rng.integers(-(1 << 62), 1 << 62, int(ln.sum()))),
                                 mask=pa.array((i % 13) == 0))
    # list<list<int32>>: 2 inner lists of 0-2 elements.
    inner_len = (np.arange(2 * n) % 3).astype(np.int32)
    inner = pa.ListArray.from_arrays(offsets_for(inner_len),
                                     pa.array(rng.integers(-(1 << 31), 1 << 31, int(inner_len.sum())).astype(np.int32)))
    ll = pa.ListArray.from_arrays(offsets_for(np.full(n, 2, dtype=np.int32)), inner,
                                  mask=pa.array((i % 17) == 0))
    # map<string, int64>: 1-3 entries.
    mn = (i % 3 + 1).astype(np.int32)
    total = int(mn.sum())
    keys = pa.DictionaryArray.from_arrays(pa.array((np.arange(total) % 8).astype(np.int32)), words).cast(pa.string())
    m = pa.MapArray.from_arrays(offsets_for(mn), keys, pa.array(rng.integers(-(1 << 62), 1 << 62, total)))
    # list<struct<x: int32, y: float64>>: 0-2 elements.
    sn = (i % 3).astype(np.int32)
    total = int(sn.sum())
    st = pa.StructArray.from_arrays([pa.array(rng.integers(-(1 << 31), 1 << 31, total).astype(np.int32)),
                                     pa.array(rng.random(total))],
                                    names=["x", "y"])
    ls = pa.ListArray.from_arrays(offsets_for(sn), st)
    return pa.table({"s": s, "l": l, "ll": ll, "m": m, "ls": ls})


def build(path, rows, chunk=1_000_000):
    rng = np.random.default_rng(23)
    writer = None
    for start in range(0, rows, chunk):
        t = build_chunk(rng, start, min(chunk, rows - start))
        if writer is None:
            writer = pq.ParquetWriter(path, t.schema, compression="snappy", data_page_size=1 << 20)
        writer.write_table(t, row_group_size=chunk)
    writer.close()


def timed(fn, repeat):
    best_wall, best_cpu, out = None, None, None
    for _ in range(repeat):
        gc.collect()
        w0, c0 = time.perf_counter(), time.process_time()
        out = fn()
        w, c = (time.perf_counter() - w0) * 1000, (time.process_time() - c0) * 1000
        if best_wall is None or w < best_wall:
            best_wall, best_cpu = w, c
        del out
        out = None
    return best_wall, best_cpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1000000,10000000")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--dir", default="/tmp/am-parquet-nested")
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    try:
        import polars as pl
    except ImportError:
        pl = None
    try:
        import duckdb
    except ImportError:
        duckdb = None

    os.makedirs(args.dir, exist_ok=True)
    results = []
    for rows in [int(r) for r in args.rows.split(",")]:
        path = os.path.join(args.dir, "nested_%d.parquet" % rows)
        if not os.path.exists(path):
            build(path, rows)
        size_mb = os.path.getsize(path) / 1e6
        f = am.ParquetFile(path)
        con = duckdb.connect() if duckdb else None
        for shape, cols in SHAPES.items():
            got = f.read_table(columns=cols)
            want = pq.read_table(path, columns=cols)
            match = all(got[c].combine_chunks().to_pylist()[:2000] == want[c].combine_chunks().to_pylist()[:2000]
                        and got[c].type == want[c].type for c in cols)
            del got, want
            readers = [("arrowmetal", lambda: f.read(columns=cols, dictionary=False)),
                       ("pyarrow", lambda: pq.read_table(path, columns=cols))]
            if pl is not None:
                readers.append(("polars", lambda: pl.read_parquet(path, columns=cols)))
            if con is not None:
                sql = "SELECT %s FROM read_parquet('%s')" % (", ".join(cols), path)
                readers.append(("duckdb", lambda: con.execute(sql).to_arrow_table()))
            for name, fn in readers:
                wall, cpu = timed(fn, args.repeat)
                row = {"rows": rows, "shape": shape, "reader": name, "wall_ms": round(wall, 2),
                       "cpu_ms": round(cpu, 2), "file_mb": round(size_mb, 1),
                       "match": match if name == "arrowmetal" else ""}
                results.append(row)
                print("%9d  %-13s %-11s wall %9.2f ms  cpu %9.2f ms" % (rows, shape, name, wall, cpu), flush=True)
        del f
    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print("wrote", args.out)
    if not args.keep:
        shutil.rmtree(args.dir, ignore_errors=True)


if __name__ == "__main__":
    main()
