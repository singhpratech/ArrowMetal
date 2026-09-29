"""Chunked import against combine-then-import, for pyarrow ChunkedArrays.

For each size, type and chunk size, three ChunkedArray columns are built from separately allocated
chunks (as a stream of record batches holds them) and imported three ways:

- `combine+import`: `combine_chunks()` per column, then `am.array` of the result (today's path
  before the chunked import; the two steps are also timed apart);
- `chunked`: `am.array(chunked_array)`, which takes the chunks through `am_import_chunks`;
- `single`: `am.array` of an already-combined column (the import step alone).

Best of `--iters` runs after one warm-up; wall time and process CPU time (all threads) of the best
run, and the 1-minute load average before each case. Output is CSV.

    PYTHONPATH=python python Benchmarks/chunked_import_bench.py [--rows 1000000,10000000,50000000]
        [--types int64,float64,utf8,utf8view] [--iters 5] [--out results/<file>.csv]
        [--label name] [--no-chunked]

Numbers from a run while other work shares the machine are provisional; publish only a quiet rerun.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import arrowmetal as am


def column(ty, rows, chunk, seed):
    rng = np.random.default_rng(seed)
    # 1,000 distinct strings, a third of them over 12 bytes (out of line in the view layout).
    words = pa.array([f"a-longer-name-{k:04d}" if k % 3 == 0 else f"name-{k:04d}" for k in range(1000)], pa.string())
    out = []
    for at in range(0, rows, chunk):
        n = min(chunk, rows - at)
        if ty == "int64":
            out.append(pa.array(rng.integers(0, 1 << 62, n)))
        elif ty == "float64":
            out.append(pa.array(rng.random(n) * 1e6))
        else:
            a = pc.take(words, pa.array(rng.integers(0, 1000, n)))      # a new allocation per chunk
            out.append(a.cast(pa.string_view()) if ty == "utf8view" else a)
    return pa.chunked_array(out)


def best(iters, fn):
    fn()
    wall, cpu, parts = float("inf"), 0.0, None
    for _ in range(iters):
        c0, t0 = time.process_time(), time.perf_counter()
        p = fn()
        w, c = (time.perf_counter() - t0) * 1e3, (time.process_time() - c0) * 1e3
        if w < wall:
            wall, cpu, parts = w, c, p
    return wall, cpu, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1000000,10000000,50000000")
    ap.add_argument("--types", default="int64,float64,utf8,utf8view")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out")
    ap.add_argument("--label", default="new")
    ap.add_argument("--chunkings", default="8192,/16",
                    help="chunk sizes: a row count, or /k for rows/k (k chunks)")
    ap.add_argument("--no-chunked", action="store_true",
                    help="skip the chunked import (for a library without am_import_chunks)")
    args = ap.parse_args()
    fields = ["label", "rows", "type", "chunk_rows", "chunks", "load", "load_end", "combine_import_wall_ms", "combine_import_cpu_ms",
              "combine_ms", "import_after_combine_ms", "chunked_wall_ms", "chunked_cpu_ms",
              "single_import_wall_ms", "single_import_cpu_ms", "speedup"]
    f = open(args.out, "w", newline="") if args.out else sys.stdout
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for n in [int(x) for x in args.rows.split(",")]:
        for ty in args.types.split(","):
            for spec in args.chunkings.split(","):
                chunk = -(-n // int(spec[1:])) if spec.startswith("/") else int(spec)
                cols = [column(ty, n, chunk, 7 + c) for c in range(3)]
                load = os.getloadavg()[0]

                def today():
                    comb = imp = 0.0
                    keep = []
                    for c in cols:
                        t = time.perf_counter()
                        flat = c.combine_chunks()
                        comb += time.perf_counter() - t
                        t = time.perf_counter()
                        keep.append(am.array(flat))
                        imp += time.perf_counter() - t
                    return comb * 1e3, imp * 1e3

                tw, tc, (comb, imp) = best(args.iters, today)
                if args.no_chunked:
                    cw = cc = float("nan")
                else:
                    cw, cc, _ = best(args.iters, lambda: [am.array(c) for c in cols])
                flats = [c.combine_chunks() for c in cols]
                sw, sc, _ = best(args.iters, lambda: [am.array(f) for f in flats])
                del flats
                w.writerow({"label": args.label, "load_end": f"{os.getloadavg()[0]:.2f}", "rows": n, "type": ty, "chunk_rows": chunk, "chunks": cols[0].num_chunks,
                            "load": f"{load:.2f}", "combine_import_wall_ms": f"{tw:.2f}",
                            "combine_import_cpu_ms": f"{tc:.1f}", "combine_ms": f"{comb:.2f}",
                            "import_after_combine_ms": f"{imp:.2f}", "chunked_wall_ms": f"{cw:.2f}",
                            "chunked_cpu_ms": f"{cc:.1f}", "single_import_wall_ms": f"{sw:.2f}",
                            "single_import_cpu_ms": f"{sc:.1f}", "speedup": f"{tw / cw:.2f}"})
                f.flush()


if __name__ == "__main__":
    main()
