"""`lf.collect(engine=am.MetalEngine())` against Polars' own engines, on the same LazyFrames.

The eight shapes of `Benchmarks/engine_bench.py` and 37 more (group-by per aggregate family over one
and two keys at few and many groups, whole-frame aggregates, each join kind, sorts, top-k, `unique`,
and the same with a String column), written as Polars LazyFrames over in-memory Polars DataFrames,
collected five ways:

* `polars in-memory`, `polars streaming` -- `lf.collect(engine=...)`
* `MetalEngine all, cold` / `warm`       -- `MetalEngine(shapes="all", min_rows=0)`: every shape it
                                          can take, with the import cache cleared first (cold) or not
* `MetalEngine default, cold`            -- `MetalEngine()`, the measured default; its `rule` column
                                          is the report's word on what it took and left

Unlike `engine_bench.py`, nothing is resident on the GPU beforehand: every MetalEngine run imports
its columns from the Polars frame, runs the plan and hands a Polars DataFrame back, which is what a
Polars user gets. Each MetalEngine row also records what the engine took (`engine.last_report`), the
shape classes, dtype class and input of each taken subtree (`shape`) and its input rows, and every
run checks that the MetalEngine result equals Polars' before timing it.

Wall time is the best of `--iters` runs, CPU time the process CPU of that run (as engine_bench.py).
`--crossover` is the sweep the default's crossovers are fitted from
(`Benchmarks/polars_engine_crossover.py`): the two Polars engines and `MetalEngine all, cold` only.
See docs/POLARS.md, "Which translatable subtrees it runs: the defaults".

`--scan` adds the Parquet scan cases: `pl.scan_parquet(file)` under a filter, a group-by, a sort
or an aggregate, over the 8-column files `Benchmarks/parquet_bench.py` writes, one per `--scan-rows` size (generated
into `--scan-dir` when it is not there yet), once per codec. There the MetalEngine reads the file
on the GPU itself; "cold" clears its open-file cache (`am.clear_parquet_cache()`) before every run,
"warm" keeps the file open between runs. Polars reads the file on every run in both of its
engines; the file stays in the OS page cache throughout.

Usage:
  PYTHONPATH=python python Benchmarks/polars_engine_bench.py [--sizes 1000000,2000000] [--iters 5]
                                                             [--cases a,t1,w3] [--crossover] [--out results.csv]
  PYTHONPATH=python python Benchmarks/polars_engine_bench.py --scan-only [--scan-rows 1000000,50000000]
                                        [--scan-codecs snappy,none] [--scan-dir DIR] [--out scan.csv]
Requires .build/release/libArrowMetalC.dylib (swift build -c release --product ArrowMetalC).
"""
import argparse
import csv
import os
import resource
import sys
import time

import numpy as np
import polars as pl
import pyarrow as pa

import arrowmetal as am
from arrowmetal import polars_engine as pe


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def best_of(fn, iters, setup=None):
    if setup:
        setup()
    fn()
    best, best_cpu = float("inf"), float("inf")
    for _ in range(iters):
        if setup:
            setup()
        c0, t0 = cpu_seconds(), time.perf_counter()
        fn()
        wall, cpu = time.perf_counter() - t0, cpu_seconds() - c0
        if wall < best:
            best, best_cpu = wall, cpu
    return best * 1e3, best_cpu * 1e3


def shapes(rows, rng):
    """The eight LazyFrames of engine_bench.py, (label, LazyFrame, order_matters)."""
    region = rng.integers(0, 200, size=rows, dtype=np.int32)
    sub = rng.integers(0, 50, size=rows, dtype=np.int32)
    amount = (rng.random(rows, dtype=np.float32) * 2000 - 500).astype(np.float32)
    qty = rng.integers(-10, 40, size=rows, dtype=np.int64)
    fact = pl.DataFrame({"region": region, "sub": sub, "amount": amount, "qty": qty})

    probe_rows = min(rows, 10_000_000)
    build_rows = 1_000_000
    probe = pl.DataFrame({"k": rng.integers(0, build_rows, size=probe_rows, dtype=np.int64),
                          "v": (rng.random(probe_rows, dtype=np.float32) * 100).astype(np.float32)})
    build = pl.DataFrame({"k": np.arange(build_rows, dtype=np.int64),
                          "w": (rng.random(build_rows, dtype=np.float32) * 2).astype(np.float32)})
    small = pl.DataFrame({"k": np.arange(1000, dtype=np.int64)})
    win_rows = min(rows, 10_000_000)
    win = pl.DataFrame({"g": rng.integers(0, 1000, size=win_rows, dtype=np.int32),
                        "v": rng.integers(0, 1_000_000, size=win_rows, dtype=np.int64)})
    span = 10_000_000_000
    quote_rows = 1_000_000
    trades = pl.DataFrame({"t": np.sort(rng.integers(0, span, size=rows).astype(np.int64))})
    quotes = pl.DataFrame({"t": np.cumsum(rng.integers(1, 2 * span // quote_rows, size=quote_rows,
                                                       dtype=np.int64)),
                           "px": rng.random(quote_rows)})
    f = fact.lazy()
    expr = (pl.col("amount") * 2 + pl.col("qty")) / (pl.col("region") + 1) - pl.col("qty")
    extra = pl.DataFrame({
        "k1": rng.integers(0, 100_000, size=rows, dtype=np.int32),
        "k2": rng.integers(0, 1_000, size=rows, dtype=np.int32),
        "name": pl.Series(rng.integers(0, 1000, size=rows)).cast(pl.String),
        "x": rng.random(rows),
        "q": rng.integers(0, 1_000_000_000, size=rows, dtype=np.int64),
    }).lazy()
    return [
        ("(a) filtered sum + count", f.filter((pl.col("region") < 20) & (pl.col("qty") > 10))
            .select(pl.col("amount").sum().alias("total"), pl.len().alias("n")), False),
        ("(b) filter + group-by 200 keys + sort desc + limit 10", f.filter(pl.col("amount") > 0)
            .group_by("region").agg(pl.col("amount").sum().alias("total"), pl.len().alias("n"))
            .sort("total", descending=True).limit(10), ["total"]),
        ("(c) group-by (region, sub) mean + max", f.group_by("region", "sub")
            .agg(pl.col("qty").mean().alias("avg"), pl.col("qty").max().alias("hi")), False),
        ("(d) projection chain then filter", f.select(expr.alias("r")).filter(pl.col("r") > 0), True),
        ("(e) inner join then sum", probe.lazy().join(build.lazy(), on="k", how="inner")
            .select(pl.col("v").sum().alias("total")), False),
        ("(f) semi join", probe.lazy().join(small.lazy(), on="k", how="semi"), False),
        ("(g) row_number over partitions", win.lazy()
            .with_columns(pl.col("v").rank("ordinal").over("g").alias("rn")), True),
        ("(h) as-of join", trades.lazy().join_asof(quotes.lazy(), on="t"), True),
        # Beyond engine_bench.py: the group-by and sort shapes the size gate has to cover.
        ("(i) group-by 1 key, 100 000 groups, sum", extra.group_by("k1")
            .agg(pl.col("q").sum().alias("s")), False),
        ("(j) group-by (k1, k2), sum + count", extra.group_by("k1", "k2")
            .agg(pl.col("q").sum().alias("s"), pl.len().alias("n")), False),
        ("(k) group-by (String, int32), sum", extra.group_by("name", "k2")
            .agg(pl.col("q").sum().alias("s")), False),
        ("(l) group-by (region, sub), Float64 sum + mean", f.with_columns(pl.col("amount")
            .cast(pl.Float64)).group_by("region", "sub")
            .agg(pl.col("amount").sum().alias("s"), pl.col("amount").mean().alias("m")), False),
        ("(m) sort 3 columns by an int64 key", extra.select("q", "k1", "x").sort("q"), ["q"]),
        ("(n) top 100 by Float64, descending", extra.select("q", "x").sort("x", descending=True)
            .head(100), ["x"]),
        ("(o) sort with a String column, by an int64 key", extra.select("q", "name", "k2")
            .sort("q"), ["q"]),
        ("(p) filter, then sort by (int32 asc, nullable Float64 desc)", extra
            .with_columns(pl.when(pl.col("k2") < 50).then(None).otherwise(pl.col("x")).alias("x"))
            .filter(pl.col("q") > 100_000_000).sort(["k2", "x"], descending=[False, True]),
            ["k2", "x"]),
        ("(q) filter, then sort by (int32 asc, int64 desc)", extra.select("q", "k1", "k2", "x")
            .filter(pl.col("x") > 0.2).sort(["k2", "q"], descending=[False, True]), ["k2", "q"]),
        ("(r) unique over (region, sub), keep first", f.unique(subset=["region", "sub"],
                                                                 keep="first"), False),
    ] + more_shapes(rows, rng, fact, extra)


def more_shapes(rows, rng, fact, extra):
    """The cases the per-shape crossovers are fitted from as well (Benchmarks/polars_engine_crossover.py):
    each aggregate family alone, over a whole frame and per group with one key and with two, at a
    few hundred to ten thousand groups and at a hundred thousand or more; each join kind with the
    probe side growing with `rows`; `unique` over many groups; a sort that needs helper keys and a
    top-k over numeric columns only; and the same operators with a String column in the input."""
    f = fact.lazy()
    cond = (pl.col("region") < 20) & (pl.col("qty") > 10)
    # Joins: the probe side has `rows` rows, the build side 1,000,000 keys, every other value of the
    # probe side's range, so half the probe rows match.
    build_rows = 1_000_000
    probe = pl.DataFrame({"k": rng.integers(0, 2 * build_rows, size=rows, dtype=np.int64),
                          "v": (rng.random(rows, dtype=np.float32) * 100).astype(np.float32)}).lazy()
    build = pl.DataFrame({"k": np.arange(0, 2 * build_rows, 2, dtype=np.int64),
                          "w": (rng.random(build_rows, dtype=np.float32) * 2).astype(np.float32)}).lazy()
    names = pl.DataFrame({"name": pl.Series(np.arange(0, 1000, 2)).cast(pl.String),
                          "w": (rng.random(500, dtype=np.float32) * 2).astype(np.float32)}).lazy()
    nullable_x = extra.select("q", pl.when(pl.col("k2") < 50).then(None).otherwise(pl.col("x"))
                              .alias("x"))
    return [
        # Whole-frame aggregates, one family each ((a) is sum + count).
        ("(a2) filtered min + max", f.filter(cond)
            .select(pl.col("qty").min().alias("lo"), pl.col("amount").max().alias("hi")), False),
        ("(a3) filtered mean", f.filter(cond)
            .select(pl.col("qty").mean().alias("m"), pl.col("amount").mean().alias("ma")), False),
        ("(a4) filtered count", f.filter(cond).select(pl.len().alias("n")), False),
        # Group-by over one int32 key: 200 groups (region) and 100,000 (k1; (i) is the sum).
        ("(t1) group-by 1 key, 200 groups, sum", f.group_by("region")
            .agg(pl.col("qty").sum().alias("s")), False),
        ("(t2) group-by 1 key, 200 groups, count", f.group_by("region")
            .agg(pl.len().alias("n")), False),
        ("(t3) group-by 1 key, 200 groups, mean", f.group_by("region")
            .agg(pl.col("qty").mean().alias("m")), False),
        ("(t4) group-by 1 key, 200 groups, min + max", f.group_by("region")
            .agg(pl.col("qty").min().alias("lo"), pl.col("qty").max().alias("hi")), False),
        ("(t5) group-by 1 key, 100 000 groups, count", extra.group_by("k1")
            .agg(pl.len().alias("n")), False),
        ("(t6) group-by 1 key, 100 000 groups, mean", extra.group_by("k1")
            .agg(pl.col("q").mean().alias("m")), False),
        ("(t7) group-by 1 key, 100 000 groups, min + max", extra.group_by("k1")
            .agg(pl.col("q").min().alias("lo"), pl.col("q").max().alias("hi")), False),
        # Group-by over two int32 keys: (region, sub), 10,000 groups, and (k1, k2), about as many
        # groups as rows up to 100,000,000 ((c), (j) and (l) mix families).
        ("(v1) group-by (region, sub), sum", f.group_by("region", "sub")
            .agg(pl.col("qty").sum().alias("s")), False),
        ("(v2) group-by (region, sub), count", f.group_by("region", "sub")
            .agg(pl.len().alias("n")), False),
        ("(v3) group-by (k1, k2), mean", extra.group_by("k1", "k2")
            .agg(pl.col("q").mean().alias("m")), False),
        ("(v4) group-by (k1, k2), min + max", extra.group_by("k1", "k2")
            .agg(pl.col("q").min().alias("lo"), pl.col("q").max().alias("hi")), False),
        # Each join kind on an int64 key, the result returned whole.
        ("(w1) inner join, 1M-row build side", probe.join(build, on="k", how="inner"), False),
        ("(w2) left join, 1M-row build side", probe.join(build, on="k", how="left"), False),
        ("(w3) semi join, 1M-row build side", probe.join(build, on="k", how="semi"), False),
        ("(w4) anti join, 1M-row build side", probe.join(build, on="k", how="anti"), False),
        # Numeric-only versions of shapes the first cases have only with a String column.
        ("(x1) unique over (k1, k2), keep first", extra.select("k1", "k2", "q")
            .unique(subset=["k1", "k2"], keep="first"), False),
        ("(x2) sort by a nullable Float64 key, descending", nullable_x
            .sort("x", descending=True), ["x"]),
        ("(x3) top 100 by an int64 key", extra.select("q", "x").sort("q").head(100), ["q"]),
        # With a String column in the input.
        ("(y1) group-by a String key (1000 values), sum", extra.group_by("name")
            .agg(pl.col("q").sum().alias("s")), False),
        ("(y2) filter by a String equality", extra.select("q", "name")
            .filter(pl.col("name") == "17"), False),
        ("(y3) filter by a String prefix, then sum", extra
            .filter(pl.col("name").str.starts_with("12")).select(pl.col("q").sum().alias("s")),
            False),
        ("(y4) inner join on a String key", extra.select("name", "q")
            .join(names, on="name", how="inner"), False),
        ("(y5) unique over (String, int32), keep first", extra.select("name", "k2", "q")
            .unique(subset=["name", "k2"], keep="first"), False),
        ("(y6) top 100 by an int64 key with a String column", extra.select("q", "name")
            .sort("q").head(100), ["q"]),
    ]


def scan_shapes(path):
    """The Parquet scan cases over parquet_bench.py's file (id, qty, code, price, weight, cat, ts,
    flag), (label, LazyFrame, order_matters)."""
    lf = pl.scan_parquet(path)
    return [
        ("(s1) scan, filter, group-by 1000 keys, sum + count", lf.filter(pl.col("price") > 500.0)
            .group_by("qty").agg(pl.col("weight").sum().alias("w"), pl.len().alias("n")), False),
        ("(s2) scan, filter on id (row groups skipped), group-by, sum", lf
            .filter(pl.col("id") < 5_000_000).group_by("qty")
            .agg(pl.col("price").sum().alias("p")), False),
        ("(s3) scan, filter, sum + count", lf.filter((pl.col("code") < 20_000) & pl.col("flag"))
            .select(pl.col("price").sum().alias("p"), pl.len().alias("n")), False),
        ("(s4) scan, sort 2 columns by a float64 key", lf.select("id", "price").sort("price"),
            ["price"]),
    ]


def ensure_scan_file(directory, rows, codec):
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import parquet_bench
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "bench-%s-%d.parquet" % (codec, rows))
    if not parquet_bench.complete(path):
        t0 = time.perf_counter()
        parquet_bench.build(path, rows, codec)
        print(f"wrote {path} in {time.perf_counter() - t0:.1f} s")
    return path


def same(a, b, order):
    """`order` is False (any row order), True (the exact order), or the sort key columns: then
    the keys must match row by row and the rows as a whole must match as a multiset, because rows
    that tie on every key come back in an unspecified order from both engines."""
    from polars.testing import assert_frame_equal
    try:
        if isinstance(order, list):
            assert_frame_equal(a.select(order), b.select(order), check_exact=False,
                               rel_tol=1e-4, abs_tol=1e-6)
            order = False
        if not order:
            a, b = a.sort(pl.all(), nulls_last=True), b.sort(pl.all(), nulls_last=True)
        assert_frame_equal(a, b, check_exact=False, rel_tol=1e-4, abs_tol=1e-6)
        return True
    except AssertionError:
        return False


def case_id(label):
    """'(a2) filtered min + max' -> 'a2'."""
    return label[1:label.index(")")]


def shape_of(report):
    """What the policy sees of the subtrees an engine took: `shape classes|dtype class|input` and
    the input rows, one per subtree, joined by ';'."""
    shapes = ";".join("+".join(t["shape"]) + "|" + t["dtype_class"] + "|" + t["input"]
                      for t in report.taken)
    rows = ";".join(str(t["rows"]) for t in report.taken)
    return shapes, rows


def rule_of(report):
    """The default policy's word on a plan: the rule of each subtree it took, and the policy lines
    of the nodes it left (the placement stops at a taken node, so these are the ones above it)."""
    lines = [t["root"] + ": " + t["rule"] for t in report.taken]
    lines += [f for f in report.fallbacks if " rule: " in f]
    return " | ".join(lines)


def run_case(label, lf, order, rows, iters, crossover, cold, results_rows):
    every = am.MetalEngine(min_rows=0, shapes="all")
    default = am.MetalEngine()
    want = lf.collect()
    got = lf.collect(engine=every)
    rep = every.last_report
    taken = ";".join(t["root"] + "[" + ">".join(t["kinds"]) + "]" for t in rep.taken)
    shape, input_rows = shape_of(rep)
    fallbacks = " | ".join(rep.fallbacks)
    ok = same(got, want, order)
    engines = [("polars in-memory", lambda: lf.collect(engine="in-memory"), None),
               ("polars streaming", lambda: lf.collect(engine="streaming"), None),
               ("MetalEngine all, cold", lambda: lf.collect(engine=every), cold)]
    rule = taken_default = ""
    if not crossover:
        lf.collect(engine=default)
        taken_default = ";".join(t["root"] for t in default.last_report.taken)
        rule = rule_of(default.last_report)
        engines += [("MetalEngine all, warm", lambda: lf.collect(engine=every), None),
                    ("MetalEngine default, cold", lambda: lf.collect(engine=default), cold)]
    results = {}
    for name, fn, setup in engines:
        results[name] = best_of(fn, iters, setup)
    cold()
    fastest = min(results["polars in-memory"][0], results["polars streaming"][0])
    for name, (wall, cpu) in results.items():
        is_metal = name.startswith("MetalEngine")
        is_default = name.startswith("MetalEngine default")
        row = {"rows": rows, "case": label, "engine": name, "wall_ms": f"{wall:.3f}",
               "cpu_ms": f"{cpu:.1f}",
               "taken": (taken_default if is_default else taken) if is_metal else "",
               "fallbacks": fallbacks if is_metal and not is_default else "",
               "equal_to_polars": ok if is_metal else "",
               "vs_fastest_polars": f"{fastest / wall:.2f}" if is_metal else "",
               "vs_polars_in_memory": (f"{results['polars in-memory'][0] / wall:.2f}"
                                       if is_metal else ""),
               "shape": shape if is_metal and not is_default else "",
               "input_rows": input_rows if is_metal and not is_default else "",
               "rule": rule if is_default else ""}
        results_rows.append(row)
        note = ""
        if is_metal:
            note = (f"  x{fastest / wall:.2f} vs fastest Polars; taken: "
                    f"{row['taken'] or 'nothing'}")
        print(f"  {label[:52]:<52} {name:<26} {wall:9.2f} ms {cpu:8.1f} CPU-ms{note}")
    if not crossover and rule:
        print(f"  {'':<52} default rule: {rule}")
    if not ok:
        print(f"  !! {label}: MetalEngine result differs from Polars")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="2000000,50000000")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--cases", default="", help="comma-separated case ids, e.g. a,c,j,t1,s4")
    ap.add_argument("--out", default=None)
    ap.add_argument("--crossover", action="store_true",
                    help="the crossover sweep: Polars' two engines and MetalEngine(shapes='all') "
                         "cold only, with the shape each subtree has (Benchmarks/"
                         "polars_engine_crossover.py fits the per-shape crossovers from it)")
    ap.add_argument("--scan", action="store_true", help="add the Parquet scan cases")
    ap.add_argument("--scan-only", action="store_true", help="only the Parquet scan cases")
    ap.add_argument("--scan-rows", default="50000000",
                    help="comma-separated file sizes in rows, one file per size and codec")
    ap.add_argument("--scan-codecs", default="snappy,none")
    ap.add_argument("--scan-dir", default=os.path.join(os.sep, "tmp", "arrowmetal-parquet-bench"))
    args = ap.parse_args()
    sizes = [] if args.scan_only else [int(s) for s in args.sizes.split(",")]
    wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
    mode = ", crossover sweep" if args.crossover else ""
    header = (f"ArrowMetal {am.version()} on {am.device_name()}, polars {pl.__version__} "
              f"({pl.thread_pool_size()} threads), pyarrow {pa.__version__}, best of {args.iters}"
              f"{mode}")
    print(header)
    rows_out = []
    for rows in sizes:
        rng = np.random.default_rng(1234)
        print(f"\nrows={rows:,}")
        for label, lf, order in shapes(rows, rng):
            if wanted and case_id(label) not in wanted:
                continue
            run_case(label, lf, order, rows, args.iters, args.crossover, pe.clear_import_cache,
                     rows_out)
    if args.scan or args.scan_only:
        for scan_rows in [int(s) for s in args.scan_rows.split(",")]:
            for codec in args.scan_codecs.split(","):
                path = ensure_scan_file(args.scan_dir, scan_rows, codec)
                size = os.path.getsize(path)
                print(f"\nscan: {os.path.basename(path)}, {size / 1e9:.2f} GB, {scan_rows:,} rows")
                for label, lf, order in scan_shapes(path):
                    if wanted and case_id(label) not in wanted:
                        continue

                    def cold():
                        am.clear_parquet_cache()
                        pe.clear_import_cache()

                    before = len(rows_out)
                    rep = run_case(f"{label} [{codec}]", lf, order, scan_rows, args.iters,
                                   args.crossover, cold, rows_out)
                    scans = [sc for t in rep.taken for sc in t.get("scans") or ()]
                    skipped = sum(sc.get("row_groups_skipped_by_statistics", 0)
                                  + sc.get("row_groups_skipped_by_page_index", 0) for sc in scans)
                    read = sum(sc.get("row_groups_read", 0) for sc in scans)
                    for r in rows_out[before:]:
                        r["row_groups_read_skipped"] = (
                            f"{read}/{skipped}" if r["engine"] == "MetalEngine all, cold"
                            or r["engine"] == "MetalEngine all, warm" else "")
    if args.out:
        fields = list(dict.fromkeys(k for r in rows_out for k in r))
        for r in rows_out:
            for k in fields:
                r.setdefault(k, "")
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            fh.write("# " + header + "\n")
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows_out)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
