"""Per-operation timings for the four rows this branch attacks in docs/TO_IMPROVE.md.

`full_matrix.py` measures 946 comparisons and needs the machine to itself for half an hour. This
script measures one operation at a time, with the same methodology (one warm-up, best of five, the
same generated columns, the same library idioms), so a before/after pair can be taken in a couple of
seconds without paying for the rest of the matrix.

    PYTHONPATH=python python Benchmarks/loss_bench.py --op count_distinct_by_key --rows 10000000
    PYTHONPATH=python python Benchmarks/loss_bench.py --op all --rows 10000000,50000000

One operation and one row count per process is the memory-safe way to run it; `--op all` walks them
in one process and frees each dataset before the next.
"""
import argparse
import gc
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import arrowmetal as am                                                # noqa: E402

SEED = 20260906


def best_of(fn, iters=5, budget=1.2):
    """(wall_ms, iterations): full_matrix.py's rule — one warm-up, then the best of up to `iters`."""
    fn()
    best, total, n = float("inf"), 0.0, 0
    while n < iters and (n < 2 or total < budget):
        t0 = time.perf_counter()
        fn()
        w = time.perf_counter() - t0
        best = min(best, w)
        total += w
        n += 1
    return best * 1000.0, n


def report(op, rows, impls):
    print(f"  {op:<38} {rows:>11,}")
    for library, fn in impls.items():
        if fn is None:
            print(f"    {library:<12}         -- no equivalent")
            continue
        wall, n = best_of(fn)
        print(f"    {library:<12} {wall:10.3f} ms  (best of {n})")
    gc.collect()


def masked(values, rng, null_fraction=0.10):
    return pa.array(values, mask=rng.random(len(values)) < null_fraction)


# ------------------------------------------------------------------ the four operations


def bench_count_distinct_by_key(n, distincts=(1_000, 100_000, 10_000_000)):
    import polars as pl
    rng = np.random.default_rng(SEED)
    v = rng.random(n) * 1000.0
    fl_a = masked(v, rng)
    fl_g = am.array(fl_a)
    fl_p = pl.Series("x", fl_a)
    for distinct in distincts:
        if distinct > n:
            continue
        k = rng.integers(0, min(distinct, n), size=n, dtype=np.int32)
        k_a = pa.array(k)
        k_g = am.array(k_a)
        df = pl.DataFrame({"k": pl.Series("k", k_a), "x": fl_p})
        tbl = pa.table({"k": k_a, "x": fl_a})
        report(f"count_distinct by key ({distinct} groups)", n, {
            "arrowmetal": lambda _k=k_g: am.group_by([_k]).count_distinct(fl_g),
            "polars": lambda _d=df: _d.group_by("k").agg(pl.col("x").n_unique()),
            "pyarrow": lambda _t=tbl: _t.group_by("k").aggregate([("x", "count_distinct")]),
        })
        del k, k_a, k_g, df, tbl
        gc.collect()


def bench_tdigest(n):
    rng = np.random.default_rng(SEED)
    v = rng.random(n) * 1000.0
    fl_a = masked(v, rng)
    fl_g = am.array(fl_a)
    report("tdigest(float64, q=0.5)", n, {
        "arrowmetal": lambda: fl_g.tdigest(0.5),
        "polars": None,
        "pyarrow": lambda: pc.tdigest(fl_a, q=0.5),
    })
    print(f"    values: arrowmetal {fl_g.tdigest(0.5)!r}  pyarrow "
          f"{pc.tdigest(fl_a, q=0.5).to_pylist()[0]!r}  exact {np.nanmedian(v[fl_a.is_valid().to_numpy(zero_copy_only=False)])!r}")


def bench_two_key_sum(n, distinct=1_000):
    import polars as pl
    rng = np.random.default_rng(SEED)
    iv = rng.integers(-1000, 1001, size=n, dtype=np.int64)
    vals_a = masked(iv, rng)
    vals_g = am.array(vals_a)
    vals_p = pl.Series("x", vals_a)
    side = max(2, int(np.ceil(np.sqrt(distinct))))
    ka = pa.array(rng.integers(0, side, size=n, dtype=np.int32))
    kb = pa.array(rng.integers(0, side, size=n, dtype=np.int32))
    ka_g, kb_g = am.array(ka), am.array(kb)
    df = pl.DataFrame({"a": pl.Series("a", ka), "b": pl.Series("b", kb), "x": vals_p})
    tbl = pa.table({"a": ka, "b": kb, "x": vals_a})
    report(f"sum by two int32 keys (~{side * side} groups)", n, {
        "arrowmetal": lambda: am.group_by([ka_g, kb_g]).sum(vals_g),
        "polars": lambda: df.group_by(["a", "b"]).agg(pl.col("x").sum()),
        "pyarrow": lambda: tbl.group_by(["a", "b"]).aggregate([("x", "sum")]),
    })
    # the single-key row next to it, for the floor this path is aiming at
    report("sum by int32 key (1000 groups)", n, {
        "arrowmetal": lambda: am.group_by([ka_g]).sum(vals_g),
    })


def bench_list_value_length(n):
    import polars as pl
    m = min(n, 10_000_000)
    rng = np.random.default_rng(SEED + 3)
    per_row = 4
    child = rng.integers(0, 1000, size=m * per_row, dtype=np.int64)
    offsets = np.arange(m + 1, dtype=np.int32) * per_row
    lst = pa.ListArray.from_arrays(pa.array(offsets), pa.array(child))
    g_lst = am.array(lst)
    pl_lst = pl.Series("l", lst)
    report("list_value_length", m, {
        "arrowmetal": lambda: g_lst.list_value_length(),
        "polars": lambda: pl_lst.list.len(),
        "pyarrow": lambda: pc.list_value_length(lst),
    })


OPS = {
    "count_distinct_by_key": lambda n: bench_count_distinct_by_key(n),
    "tdigest": bench_tdigest,
    "two_key_sum": bench_two_key_sum,
    "list_value_length": bench_list_value_length,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="all")
    ap.add_argument("--rows", default="10000000")
    ap.add_argument("--distinct", default=None, help="one cardinality for count_distinct_by_key")
    args = ap.parse_args()
    names = list(OPS) if args.op == "all" else args.op.split(",")
    for rows in [int(r) for r in args.rows.split(",")]:
        for name in names:
            if name == "count_distinct_by_key" and args.distinct:
                bench_count_distinct_by_key(rows, distincts=(int(args.distinct),))
            else:
                OPS[name](rows)
            gc.collect()


if __name__ == "__main__":
    main()
