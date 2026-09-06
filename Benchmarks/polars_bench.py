"""Native Polars vs Polars + ArrowMetal, at 10M and 50M rows.

Three ways of reaching the GPU are timed against Polars' own answer for the same question:

  polars                    the baseline, all cores
  arrowmetal (namespace)    tier 1: `df.arrowmetal...` / `s.arrowmetal...`, GPU around Polars
  arrowmetal (plugin)       tier 2: `pl.col(...).arrowmetal...` inside a lazy plan
  arrowmetal (resident)     tier 1 with the columns already in Metal memory, so the import is
                            outside the timed region -- what a pipeline that stays on the GPU sees

Usage: PYTHONPATH=python python Benchmarks/polars_bench.py [rows] [iterations]
       PYTHONPATH=python python Benchmarks/polars_bench.py 10000000 5

Requires .build/release/libArrowMetalC.dylib. The plugin rows are skipped when
polars-plugin/target/release/libarrowmetal_polars.dylib has not been built.

Fairness rules (the same ones Benchmarks/README.md states): one process, one data set, the best
of `iterations` runs after a warm-up, wall time and process CPU time reported side by side. The
GPU rows include the Arrow C Data Interface hand-off unless the row says "resident"; nothing is
hidden in a setup phase that the Polars row does not also get.
"""
import resource
import sys
import time

import numpy as np
import polars as pl

import arrowmetal as am
from arrowmetal import polars_bridge  # noqa: F401  (registers the .arrowmetal namespaces)
from arrowmetal import polars_plugin

rows = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000_000
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 5
HAVE_PLUGIN = polars_plugin.available()

rng = np.random.default_rng(42)
results = []


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def bench(section, label, fn, *, check=None):
    """Best of `iters` after one warm-up. `check` is the value every run must agree with."""
    got = fn()
    if check is not None:
        ok = np.isclose(float(got), float(check), rtol=1e-9) if isinstance(got, (int, float)) \
            else got == check
        if not ok:
            print(f"  !! {label}: got {got!r}, expected {check!r}")
    best = float("inf")
    best_cpu = float("inf")
    for _ in range(iters):
        c0 = cpu_seconds()
        t0 = time.perf_counter()
        fn()
        wall = time.perf_counter() - t0
        cpu = cpu_seconds() - c0
        if wall < best:
            best, best_cpu = wall, cpu
    print(f"  {label:<36} {best * 1000:9.2f} ms  {best_cpu * 1000:9.1f} CPU-ms")
    results.append((section, label, best * 1000, best_cpu * 1000))
    return got


print(f"ArrowMetal {am.version()} on {am.device_name()} vs polars {pl.__version__} "
      f"({pl.thread_pool_size()} threads); rows={rows}, best of {iters}")
print(f"expression plugin: {'built' if HAVE_PLUGIN else 'NOT BUILT (rows skipped)'}\n")

# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------
keys = rng.integers(0, 1000, size=rows, dtype=np.int32)
vals = rng.integers(-1000, 1001, size=rows, dtype=np.int64)
amounts = rng.random(rows) * 500.0
names = np.array([f"customer-{i}" for i in range(4096)])[rng.integers(0, 4096, size=rows)]

df = pl.DataFrame({"k": keys, "v": vals, "amount": amounts, "name": pl.Series("name", names)})
lf = df.lazy()

# Columns held in Metal memory, for the "resident" rows.
gk = am.from_polars(df["k"])
gv = am.from_polars(df["v"])
ga = am.from_polars(df["amount"])
gn = am.from_polars(df["name"])
gb_keys = am.group_by([gk])

# ---------------------------------------------------------------------------------------------
sec = "sum(Int64)"
print(sec)
want = df["v"].sum()
bench(sec, "polars", lambda: df["v"].sum(), check=want)
bench(sec, "arrowmetal (namespace)", lambda: df["v"].arrowmetal.sum(), check=want)
bench(sec, "arrowmetal (resident)", lambda: gv.sum(), check=want)
if HAVE_PLUGIN:
    bench(sec, "arrowmetal (plugin, lazy)",
          lambda: lf.select(pl.col("v").arrowmetal.sum()).collect().item(), check=want)

# ---------------------------------------------------------------------------------------------
sec = "filter(k == 2) + sum(v)"
print("\n" + sec)
want = df.filter(pl.col("k") == 2)["v"].sum()
bench(sec, "polars", lambda: df.filter(pl.col("k") == 2)["v"].sum(), check=want)
bench(sec, "polars (lazy)",
      lambda: lf.select(pl.col("v").filter(pl.col("k") == 2).sum()).collect().item(), check=want)
bench(sec, "arrowmetal (namespace)",
      lambda: df.arrowmetal.query(am.filter(am.col("k") == 2).sum(am.col("v"))), check=want)
bench(sec, "arrowmetal (fused, resident)",
      lambda: am.query({"k": gk, "v": gv}, am.filter(am.col("k") == 2).sum(am.col("v"))),
      check=want)


def _resident_filter_sum():
    with am.batch():
        return gv.filter(gk == 2).sum()


bench(sec, "arrowmetal (resident)", _resident_filter_sum, check=want)
if HAVE_PLUGIN:
    bench(sec, "arrowmetal (plugin, lazy)",
          lambda: lf.select(pl.col("v").arrowmetal.filter_sum(pl.col("k") == 2)).collect().item(),
          check=want)

# ---------------------------------------------------------------------------------------------
sec = "group-by sum(v) by 1000 keys"
print("\n" + sec)
want = df.group_by("k").agg(pl.col("v").sum()).sort("k")["v"].to_list()
bench(sec, "polars", lambda: df.group_by("k").agg(pl.col("v").sum()))
bench(sec, "arrowmetal (namespace)", lambda: df.arrowmetal.group_by("k").sum("v"))
bench(sec, "arrowmetal (resident)", lambda: gb_keys.sum(gv))
if HAVE_PLUGIN:
    bench(sec, "arrowmetal (plugin, lazy)",
          lambda: lf.select(pl.col("k").arrowmetal.group_by_sum(pl.col("v"))).collect())

# ---------------------------------------------------------------------------------------------
sec = "top_k(100) of Int64"
print("\n" + sec)
bench(sec, "polars", lambda: df["v"].top_k(100))
bench(sec, "arrowmetal (namespace)", lambda: df["v"].arrowmetal.top_k(100))
bench(sec, "arrowmetal (resident)", lambda: gv.take(gv.top_k(100, True)))
if HAVE_PLUGIN:
    bench(sec, "arrowmetal (plugin, lazy)",
          lambda: lf.select(pl.col("v").arrowmetal.top_k(100)).collect())

# ---------------------------------------------------------------------------------------------
sec = "string contains (literal)"
print("\n" + sec)
want = df["name"].str.contains("customer-1", literal=True).sum()
bench(sec, "polars", lambda: df["name"].str.contains("customer-1", literal=True), check=None)
bench(sec, "arrowmetal (namespace)", lambda: df["name"].arrowmetal.contains("customer-1"))
bench(sec, "arrowmetal (resident)", lambda: gn.str_contains("customer-1"))
if HAVE_PLUGIN:
    bench(sec, "arrowmetal (plugin, lazy)",
          lambda: lf.select(pl.col("name").arrowmetal.contains("customer-1")).collect())

# ---------------------------------------------------------------------------------------------
sec = "the hand-off itself"
print("\n" + sec)
bench(sec, f"pl.Series -> Metal ({rows / 1e6:.0f}M x Int64)", lambda: am.from_polars(df["v"]))
bench(sec, "Metal -> pl.Series", lambda: am.to_polars(gv, "v"))
src, dst, same = am.zero_copy_report(df["v"])
print(f"  zero copy: source buffer {src:#x}, Metal buffer {dst:#x} -> {'SAME' if same else 'COPIED'}")

# ---------------------------------------------------------------------------------------------
print("\n\n| Operation | Polars | ArrowMetal namespace | ArrowMetal plugin | resident |")
print("|---|---|---|---|---|")
sections = []
for section, label, ms, _cpu in results:
    if section not in sections:
        sections.append(section)
for section in sections:
    row = {label: ms for s, label, ms, _ in results if s == section}
    base = row.get("polars")
    if base is None:            # the hand-off section has no Polars counterpart to compare with
        continue

    def cell(key):
        ms = row.get(key)
        if ms is None:
            return "-"
        speed = f" ({base / ms:.1f}x)" if base and key != "polars" else ""
        return f"{ms:.1f} ms{speed}"

    print(f"| {section} | {cell('polars')} | {cell('arrowmetal (namespace)')} | "
          f"{cell('arrowmetal (plugin, lazy)')} | {cell('arrowmetal (resident)')} |")
