"""pandas on the GPU: plain pandas vs the `.am` accessor vs zero-code-change accel mode.

The same expression is measured three ways on the same in-process frame:

  pandas   the frame exactly as a pandas user holds it, untouched
  accessor the explicit `s.am.sum()` / `df.am.groupby(...)` form
  accel    the *unchanged* pandas expression, with `arrowmetal.pandas_accel` installed

Both ArrowMetal columns are the ones pandas already owns, so the accessor and accel rows include
whatever conversion the dtype forces (nothing at all for the Arrow-backed frame, one numpy -> Arrow
pass for the numpy frame) plus the GPU work plus the trip back into a pandas object. Nothing is
pre-converted and cached between iterations.

Usage: PYTHONPATH=python python Benchmarks/pandas_bench.py [rows] [iterations] [--backend arrow|numpy|both]
"""
import argparse
import gc
import os
import resource
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import arrowmetal as am                                    # noqa: E402
from arrowmetal import pandas_accel                        # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("rows", nargs="?", type=int, default=10_000_000)
parser.add_argument("iters", nargs="?", type=int, default=5)
parser.add_argument("--backend", default="both", choices=["arrow", "numpy", "both"])
parser.add_argument("--csv", default=None, help="write the rows to this CSV as well")
args = parser.parse_args()

ROWS, ITERS = args.rows, args.iters
rng = np.random.default_rng(20240906)
results = []


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def bench(section, backend, label, fn):
    """Best of `ITERS`, after one warm-up. Wall ms and CPU ms."""
    try:
        fn()
    except Exception as exc:
        print(f"    {label:<10} skipped ({type(exc).__name__}: {exc})")
        results.append((section, backend, label, float("nan"), float("nan"), str(exc)))
        return float("nan")
    best, best_cpu = float("inf"), float("inf")
    for _ in range(ITERS):
        gc.collect()
        c0, t0 = cpu_seconds(), time.perf_counter()
        fn()
        wall, cpu = time.perf_counter() - t0, cpu_seconds() - c0
        if wall < best:
            best, best_cpu = wall, cpu
    print(f"    {label:<10} {best * 1000:9.2f} ms   {best_cpu * 1000:9.1f} CPU-ms")
    results.append((section, backend, label, best * 1000, best_cpu * 1000, ""))
    return best * 1000


def trio(section, backend, plain, accessor, accel_fn=None):
    """One row of the table: pandas, accessor, accel."""
    print(f"  {section}")
    p = bench(section, backend, "pandas", plain)
    a = bench(section, backend, "accessor", accessor)
    pandas_accel.install(threshold=0)
    pandas_accel.reset_stats()
    try:
        c = bench(section, backend, "accel", accel_fn or plain)
        routed = pandas_accel.stats().gpu_calls
    finally:
        pandas_accel.uninstall()
    if routed == 0:
        print("      (accel: nothing routed to the GPU — it ran in pandas)")
    for label, ms in (("accessor", a), ("accel", c)):
        if ms == ms and p == p and ms > 0:
            print(f"      {label} speedup: {p / ms:.2f}x")
    return p, a, c


def build(backend):
    """The benchmark frame in one of the two pandas storage flavours."""
    ints = rng.integers(-1000, 1001, ROWS, dtype=np.int64)
    floats = rng.normal(size=ROWS)
    keys_1k = rng.integers(0, 1_000, ROWS, dtype=np.int64)
    keys_100k = rng.integers(0, 100_000, ROWS, dtype=np.int64)
    words = np.array(["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
                      "golf", "hotel"], dtype=object)[rng.integers(0, 8, ROWS)]
    if backend == "arrow":
        df = pd.DataFrame({
            "i": pd.array(ints, dtype="int64[pyarrow]"),
            "f": pd.array(floats, dtype="double[pyarrow]"),
            "k1": pd.array(keys_1k, dtype="int64[pyarrow]"),
            "k2": pd.array(keys_100k, dtype="int64[pyarrow]"),
            "w": pd.array(words, dtype=pd.ArrowDtype(pa.string())),
        })
        right = pd.DataFrame({"k2": pd.array(np.arange(100_000), dtype="int64[pyarrow]"),
                              "extra": pd.array(np.arange(100_000), dtype="int64[pyarrow]")})
    else:
        df = pd.DataFrame({"i": ints, "f": floats, "k1": keys_1k, "k2": keys_100k,
                           "w": pd.Series(words, dtype="str")})
        right = pd.DataFrame({"k2": np.arange(100_000), "extra": np.arange(100_000)})
    return df, right


def run(backend):
    df, right = build(backend)
    print(f"\n=== {backend}-backed frame, {ROWS:,} rows, best of {ITERS} "
          f"({', '.join(f'{c}:{df[c].dtype}' for c in df.columns)}) ===")
    rep = am.zero_copy_report(df)
    print("    zero-copy on the way to Metal: "
          + ", ".join(f"{c}={'yes' if v['zero_copy'] else 'no'}" for c, v in rep.items()))

    s, f, w = df["i"], df["f"], df["w"]
    trio("sum(int64)", backend, lambda: s.sum(), lambda: s.am.sum())
    trio("mean(float64)", backend, lambda: f.mean(), lambda: f.am.mean())
    trio("groupby-sum 1k keys", backend,
         lambda: df.groupby("k1")["i"].sum(), lambda: df.am.groupby("k1").sum("i"))
    trio("groupby-sum 100k keys", backend,
         lambda: df.groupby("k2")["i"].sum(), lambda: df.am.groupby("k2").sum("i"))
    trio("sort_values(int64)", backend,
         lambda: df.sort_values("i"), lambda: df.am.sort_values("i"))
    trio("nlargest(100)", backend, lambda: s.nlargest(100), lambda: s.am.nlargest(100))
    trio("str.contains", backend,
         lambda: w.str.contains("ta", regex=False), lambda: w.am.contains("ta"),
         accel_fn=lambda: w.str.contains("ta"))
    trio("filter (mask + take)", backend,
         lambda: df[df["i"] > 0], lambda: df.am.filter(df["i"] > 0))
    trio("merge on 100k int keys", backend,
         lambda: df.merge(right, on="k2"), lambda: df.am.merge(right, on="k2"))


print(f"ArrowMetal {am.version()} on {am.device_name()}; pandas {pd.__version__}, "
      f"pyarrow {pa.__version__}; rows={ROWS:,}, best of {ITERS}")

for backend in (["arrow", "numpy"] if args.backend == "both" else [args.backend]):
    run(backend)

print("\n| Operation | Backend | pandas ms | accessor ms | accel ms | accessor x | accel x |")
print("|---|---|---:|---:|---:|---:|---:|")
by = {}
for section, backend, label, ms, cpu, err in results:
    by.setdefault((section, backend), {})[label] = ms
for (section, backend), row in by.items():
    p, a, c = row.get("pandas"), row.get("accessor"), row.get("accel")
    def x(v):
        return f"{p / v:.2f}x" if p and v and v == v and p == p else "-"
    def m(v):
        return f"{v:.1f}" if v is not None and v == v else "-"
    print(f"| {section} | {backend} | {m(p)} | {m(a)} | {m(c)} | {x(a)} | {x(c)} |")

if args.csv:
    with open(args.csv, "w") as fh:
        fh.write("section,backend,variant,wall_ms,cpu_ms,note\n")
        for r in results:
            fh.write(",".join(str(x).replace(",", ";") for x in r) + "\n")
    print(f"\nwrote {args.csv}")
