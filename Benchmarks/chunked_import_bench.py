"""Chunked import against combine-then-import, for pyarrow ChunkedArrays.

For each size, type and chunk size, three ChunkedArray columns are built from separately allocated
chunks (as a stream of record batches holds them) and imported three ways:

- `combine+import`: `combine_chunks()` per column, then `am.array` of the result (the path before
  the chunked import; the two steps are also timed apart);
- `chunked`: `am.array(chunked_array)`, which takes the chunks through `am_import_chunks`;
- `single`: `am.array` of an already-combined column (the import step alone).

`--iters` runs after one warm-up; wall time and process CPU time (all threads) of the best run and
the median of the runs, and the 1-minute load average before and after each case. Output is CSV.

    PYTHONPATH=python python Benchmarks/chunked_import_bench.py [--rows 1000000,10000000,50000000]
        [--types int64,float64,utf8,utf8view] [--iters 5] [--out results/<file>.csv]
        [--label name] [--no-chunked] [--chunkings 8192,/16] [--threads auto,1,2,4,8,12]
        [--paths combine,chunked,single]

`--chunkings` lists the chunk sizes: a row count, or `/k` for k chunks of rows/k. `--threads` lists
the copy thread counts to time (`am.set_import_threads`; `auto` is the default policy; `env`, the
default, leaves the setting alone, for a build without the setter, where `ARROWMETAL_IMPORT_THREADS`
sets it). `--no-chunked` runs against a build without `am_import_chunks`, so two builds can be timed
alternately (one process each, `ARROWMETAL_LIB` and `PYTHONPATH` pointing at the build). `--out`
appends to an existing file.

Numbers from a run while other work shares the machine are provisional; publish only a quiet rerun.
"""
import argparse
import csv
import os
import statistics
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


def timed(iters, fn):
    """(best wall, its CPU, median wall, median CPU, parts of the best run) over `iters` runs.

    A run's result is dropped before the next run starts, so the arrays it imported go back to the
    buffer pool and every run after the warm-up writes into recycled memory."""
    p = fn()
    del p
    runs = []
    for _ in range(iters):
        c0, t0 = time.process_time(), time.perf_counter()
        p = fn()
        w, c = (time.perf_counter() - t0) * 1e3, (time.process_time() - c0) * 1e3
        runs.append((w, c, p if isinstance(p, tuple) else None))
        del p
    best = min(runs, key=lambda r: r[0])
    return best[0], best[1], statistics.median(r[0] for r in runs), statistics.median(r[1] for r in runs), best[2]


def set_threads(spec):
    if spec == "env":
        return
    if not hasattr(am, "set_import_threads"):
        sys.exit("this build has no am.set_import_threads; use --threads env")
    am.set_import_threads(0 if spec == "auto" else int(spec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1000000,10000000,50000000")
    ap.add_argument("--types", default="int64,float64,utf8,utf8view")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out")
    ap.add_argument("--label", default="new")
    ap.add_argument("--round", default="1")
    ap.add_argument("--chunkings", default="8192,/16",
                    help="chunk sizes: a row count, or /k for rows/k (k chunks)")
    ap.add_argument("--threads", default="env", help="copy thread counts: auto, env or a number, comma separated")
    ap.add_argument("--paths", default="combine,chunked,single")
    ap.add_argument("--no-chunked", action="store_true",
                    help="skip the chunked import (for a library without am_import_chunks)")
    args = ap.parse_args()
    paths = set(args.paths.split(","))
    if args.no_chunked:
        paths.discard("chunked")
    fields = ["label", "round", "threads", "rows", "type", "chunk_rows", "chunks", "load", "load_end",
              "combine_import_wall_ms", "combine_import_cpu_ms", "combine_ms", "import_after_combine_ms",
              "chunked_wall_ms", "chunked_cpu_ms", "chunked_wall_median_ms", "chunked_cpu_median_ms",
              "single_import_wall_ms", "single_import_cpu_ms", "single_import_wall_median_ms",
              "single_import_cpu_median_ms"]
    append = bool(args.out) and os.path.exists(args.out)
    f = open(args.out, "a" if append else "w", newline="") if args.out else sys.stdout
    w = csv.DictWriter(f, fieldnames=fields)
    if not append:
        w.writeheader()
    nan = float("nan")
    for n in [int(x) for x in args.rows.split(",")]:
        for ty in args.types.split(","):
            for spec in args.chunkings.split(","):
                chunk = -(-n // int(spec[1:])) if spec.startswith("/") else int(spec)
                cols = [column(ty, n, chunk, 7 + c) for c in range(3)]
                flats = [c.combine_chunks() for c in cols] if "single" in paths else None
                for th in args.threads.split(","):
                    set_threads(th)
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

                    tw = tc = comb = imp = nan
                    if "combine" in paths:
                        tw, tc, _, _, (comb, imp) = timed(args.iters, today)
                    cw = cc = cwm = ccm = nan
                    if "chunked" in paths:
                        cw, cc, cwm, ccm, _ = timed(args.iters, lambda: [am.array(c) for c in cols])
                    sw = sc = swm = scm = nan
                    if "single" in paths:
                        sw, sc, swm, scm, _ = timed(args.iters, lambda: [am.array(x) for x in flats])
                    w.writerow({"label": args.label, "round": args.round, "threads": th, "rows": n, "type": ty,
                                "chunk_rows": chunk, "chunks": cols[0].num_chunks, "load": f"{load:.2f}",
                                "load_end": f"{os.getloadavg()[0]:.2f}",
                                "combine_import_wall_ms": f"{tw:.2f}", "combine_import_cpu_ms": f"{tc:.1f}",
                                "combine_ms": f"{comb:.2f}", "import_after_combine_ms": f"{imp:.2f}",
                                "chunked_wall_ms": f"{cw:.2f}", "chunked_cpu_ms": f"{cc:.1f}",
                                "chunked_wall_median_ms": f"{cwm:.2f}", "chunked_cpu_median_ms": f"{ccm:.1f}",
                                "single_import_wall_ms": f"{sw:.2f}", "single_import_cpu_ms": f"{sc:.1f}",
                                "single_import_wall_median_ms": f"{swm:.2f}", "single_import_cpu_median_ms": f"{scm:.1f}"})
                    f.flush()
                del flats, cols


if __name__ == "__main__":
    main()
