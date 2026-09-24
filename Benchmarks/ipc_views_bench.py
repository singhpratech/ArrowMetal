"""Arrow IPC read cost of the view layouts against their classic counterparts.

For each layout the same values are written twice by pyarrow, once as the classic type and once as
the view type, and read back through ArrowMetal's IPC reader, batch by batch, with
`am.scan_ipc(path).to_reader()` (each batch is materialised in shared memory and handed to pyarrow
zero copy). The view layouts are converted to the engine's offsets-plus-data layout on read, so the
gap between a pair is the cost of that conversion:

- string / string_view: a CPU pass over the 16-byte views, then one copy of the bytes;
- list<int64> / list_view<int64> in order: offsets derived, child used as it is;
- list_view<int64> out of order: the child gathered with the engine's take.

pyarrow's own read of each file (`read_all`, which keeps views as views) is timed alongside for
reference. Output is a CSV row per (layout, reader).

    PYTHONPATH=python python Benchmarks/ipc_views_bench.py [rows] [iters] [--out results/<file>.csv]

Numbers from a run while other work shares the machine are provisional; publish only a quiet rerun.
"""
import argparse
import csv
import os
import tempfile
import time

import numpy as np
import pyarrow as pa

import arrowmetal as am


LAYOUTS = ["string", "string_view", "list_int64", "list_view_int64_in_order", "list_view_int64_shuffled"]


def batch_arrays(rows, rng):
    """One batch of every layout over the same values. Each batch owns its child, so a list view file
    holds exactly the child elements its rows cover, as the list file does."""
    words = np.array(["w%07d" % i for i in range(1000)] + ["a much longer string value %06d" % i for i in range(1000)])
    text = pa.array(words[rng.integers(0, len(words), rows)].tolist(), pa.string())
    sizes = rng.integers(0, 8, rows).astype(np.int32)
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int32)
    child = pa.array(rng.integers(0, 1 << 40, int(offsets[-1])), pa.int64())
    perm = rng.permutation(rows)
    return {
        "string": text,
        "string_view": text.cast(pa.string_view()),
        "list_int64": pa.ListArray.from_arrays(pa.array(offsets), child),
        "list_view_int64_in_order": pa.ListViewArray.from_arrays(pa.array(offsets[:-1]), pa.array(sizes), child),
        "list_view_int64_shuffled": pa.ListViewArray.from_arrays(pa.array(offsets[:-1][perm]),
                                                                 pa.array(sizes[perm]), child),
    }


def ours_read(path):
    return sum(b.num_rows for b in am.scan_ipc(path, prefetch=0).to_reader())


def best(fn, iters):
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return min(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="?", type=int, default=10_000_000)
    ap.add_argument("iters", nargs="?", type=int, default=5)
    ap.add_argument("--batch-rows", type=int, default=1 << 20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(42)
    rows_out = []
    with tempfile.TemporaryDirectory() as tmp:
        writers = {}
        for start in range(0, args.rows, args.batch_rows):
            arrays = batch_arrays(min(args.batch_rows, args.rows - start), rng)
            for name in LAYOUTS:
                batch = pa.record_batch([arrays[name]], names=["c"])
                if name not in writers:
                    writers[name] = pa.ipc.new_file(os.path.join(tmp, name + ".arrow"), batch.schema)
                writers[name].write_batch(batch)
        for w in writers.values():
            w.close()
        for name in LAYOUTS:
            path = os.path.join(tmp, name + ".arrow")
            size = os.path.getsize(path)
            assert ours_read(path) == args.rows, name
            ours = best(lambda: ours_read(path), args.iters)
            theirs = best(lambda: pa.ipc.open_file(path).read_all(), args.iters)
            for reader, secs in (("arrowmetal_scan_ipc", ours), ("pyarrow_read_all", theirs)):
                rows_out.append({"layout": name, "rows": args.rows, "file_mb": round(size / 1e6, 1),
                                 "reader": reader, "best_ms": round(secs * 1e3, 2), "iters": args.iters})
                print("%-28s %-28s %10.2f ms" % (name, reader, secs * 1e3))
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)


if __name__ == "__main__":
    main()
