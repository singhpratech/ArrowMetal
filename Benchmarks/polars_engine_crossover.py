#!/usr/bin/env python3
"""Fits the MetalEngine's per-shape crossovers (python/arrowmetal/_engine_crossovers.py) from a sweep.

The sweep is `Benchmarks/polars_engine_bench.py --crossover`: every case at several sizes, through
Polars' in-memory and streaming engines and through `MetalEngine(shapes="all", min_rows=0)` cold,
each MetalEngine row carrying the shape the engine's policy sees (`shape`: the classes of the taken
subtree, its dtype class and its input) and its input rows (`input_rows`):

    PYTHONPATH=python python Benchmarks/polars_engine_bench.py --crossover \\
        --sizes 250000,500000,1000000,2000000,5000000,10000000,20000000,50000000 \\
        --scan --scan-rows 1000000,2000000,5000000,10000000,20000000,50000000 \\
        --out Benchmarks/results/polars_engine_crossover_<date>.csv

The fit, per case, is router_table.py's (python/arrowmetal/_router_fit.py `fit`), with the
MetalEngine as the GPU side and the faster of Polars' two engines as the CPU side, over the case's
input rows: the first measured size from which the engine is ahead at every larger size by at least
`MARGIN` (its time x 1.15, or x 1.35 for a shape with a String column, no more than the faster Polars engine's),
and inside the bracket below it the point where the straight lines through the two measured points
of each side meet. A case the engine is not ahead of at its largest size has no crossover.

Per (class, dtype class, input), the crossover is the largest over the cases that measure the class:
the cases whose taken subtree has that dtype class and input and whose classes all belong to the
class's node (a group-by of sum + mean measures both group_by:sum and group_by:mean; a filter,
group-by and top-k does not measure the group-by, because the top-k is a different node). If one of
those cases has no crossover, neither has the class, and the default leaves it to Polars at every size.
A case taken as more than one subtree, or not taken at all, measures nothing.

The sort kernels' own crossovers against the fastest CPU library (`vs_fastest_library` of
Benchmarks/results/router_2026-09-24.json, docs/CROSSOVER.md) are recorded alongside as `SWEEP`;
the policy (python/arrowmetal/_engine_policy.py) takes the larger of the two, and of the router
table's rows for the kernels it routes.

Usage:
    python Benchmarks/polars_engine_crossover.py Benchmarks/results/polars_engine_crossover_<date>.csv
    python Benchmarks/polars_engine_crossover.py --check     # exit 1 if the committed table is stale
    python Benchmarks/polars_engine_crossover.py CSV --print # the per-case and per-class fits
"""
import argparse
import csv
import importlib.util
import json
import os
import pprint
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "python", "arrowmetal", "_engine_crossovers.py")
SWEEP_JSON = "Benchmarks/results/router_2026-09-24.json"
SWEEP_LABELS = {
    "sort": ["sort: argsort int64", "sort: argsort float64", "sort: lexsort (2 int32 keys)"],
    "sort_helper_keys": ["sort: argsort int64", "sort: argsort float64", "sort: lexsort (2 int32 keys)"],
    "top_k": ["sort: top_k (k=100, int64)"],
}
POLARS = ("polars in-memory", "polars streaming")
# A case counts as ahead at a size only when the MetalEngine's time, raised by this fraction, is
# still at most the faster Polars engine's: a case within that margin of Polars is not ahead. A shape
# with a String column gets the wider margin: its advantage grows slowly with size (the string sort
# case is 1.02x at 1M rows and 1.4x at 5M), so a crossover fitted at 15% lands on a coin flip.
MARGIN = {"numeric": 0.15, "string": 0.35}
# A shape with a String column is taken from this many rows at the earliest, whatever its fit says:
# below it the string sort cases sit within the margin of Polars (1.22x and 1.37x at 2,000,000 rows in
# the sweep, 1.20x and 1.32x under shapes="all" in the default benchmark), and from 5,000,000 the
# sweep has them at 1.4x and up.
STRING_FLOOR = 5_000_000
METAL = "MetalEngine all, cold"
# The group-count buckets of the group-by classes: (name, fewest groups, most groups), one per group
# count of the sweep's group-by grid (Benchmarks/polars_engine_bench.py `GRID_GROUPS`), each reaching
# half a decade either side of it (the geometric midpoints). A group count of at least the input rows
# / NEAR_ROWS is the "rows/2" bucket whatever its size: the grid's rows/2 cases hold about 0.43 x rows
# groups, and its 1,000,000-group cases fall there too below 4,000,000 rows. A group count between
# the last bucket and rows / NEAR_ROWS, or of 0, has no bucket.
GROUP_BUCKETS = (("200", 1, 447), ("1,000", 448, 3_162), ("10,000", 3_163, 31_622),
                 ("100,000", 31_623, 316_227), ("1,000,000", 316_228, 3_162_277))
NEAR_ROWS = 4
ROWS_BUCKET = "rows/2"


def group_bucket(groups, rows):
    """The bucket of a group-by with `groups` groups over `rows` input rows, or None (the same rule
    as _engine_policy.group_bucket, which reads the generated table)."""
    if groups is None or groups <= 0:
        return None
    if groups * NEAR_ROWS >= rows:
        return ROWS_BUCKET
    for name, lo, hi in GROUP_BUCKETS:
        if lo <= groups <= hi:
            return name
    return None


def _load(name):
    path = os.path.join(ROOT, "python", "arrowmetal", name + ".py")
    spec = importlib.util.spec_from_file_location("arrowmetal_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def node_of(cls):
    # The same map as _engine_policy.NODE (that module imports the generated table, so it is not
    # loaded here).
    head = cls.split(":")[0]
    return {"group_by_multi": "group_by", "sort_helper_keys": "sort", "top_k": "sort"}.get(head, head)


def read_sweep(path):
    """{case: {"shape": (classes, dtype class, input), "points": {input_rows: (metal_ms, polars_ms)}}}"""
    with open(path) as fh:
        lines = fh.read().splitlines()
    header = lines[0][1:].strip() if lines and lines[0].startswith("#") else ""
    rows = list(csv.DictReader(l for l in lines if not l.startswith("#")))
    timings = {}
    for r in rows:
        timings.setdefault((r["case"], r["rows"]), {})[r["engine"]] = r
    cases = {}
    for (case, _rows), eng in timings.items():
        m = eng.get(METAL)
        if m is None or not m["shape"] or ";" in m["shape"] or not all(p in eng for p in POLARS):
            continue
        classes, dclass, source = m["shape"].split("|")
        shape = (tuple(classes.split("+")), dclass, source)
        c = cases.setdefault(case, {"shape": shape, "points": {}, "groups": {}})
        if c["shape"] != shape:
            raise SystemExit(f"{case}: the engine took different shapes at different sizes "
                             f"({c['shape']} and {shape})")
        n = int(m["input_rows"])
        if n in c["points"]:
            continue        # a case whose input stops growing (a capped side): the first size counts
        # A size where the MetalEngine's answer differed from Polars' counts as not ahead, whatever
        # its time.
        metal = float(m["wall_ms"]) if m["equal_to_polars"] == "True" else float("inf")
        c["points"][n] = (metal, min(float(eng[p]["wall_ms"]) for p in POLARS))
        if m.get("groups"):
            c["groups"][n] = int(m["groups"])
    return header, cases


def fit_points(label, points, dclass, single=True):
    """(crossover, step, bracket low) of one series {input rows: (metal_ms, polars_ms)}, or Nones.
    `single`: whether one measured size alone can be a crossover."""
    fit = _load("_router_fit")
    sizes = sorted(points)
    bench = {}
    for n, (metal, polars) in points.items():
        bench[(label, n, "gpu")] = metal * (1 + MARGIN[dclass])
        bench[(label, n, "cpu")] = polars
    try:
        cross, step, low, _pts = fit.fit(label, "cpu", None, sizes, bench)
    except fit.FitError:
        cross = step = low = None
    if step is not None and step == sizes[-1] and (len(sizes) > 1 or not single):
        # Ahead at the largest size alone is one measurement: not a crossover.
        cross = step = low = None
    if cross is not None and dclass == "string" and cross < STRING_FLOOR:
        cross = STRING_FLOOR
        step = min((n for n in sizes if n >= STRING_FLOOR), default=step)
    return cross, step, low


def fit_cases(cases):
    out = {}
    for case, c in sorted(cases.items()):
        sizes = sorted(c["points"])
        cross, step, low = fit_points(case, c["points"], c["shape"][1])
        out[case] = {"shape": c["shape"], "rows": cross, "step": step, "low": low,
                     "largest": sizes[-1], "smallest": sizes[0],
                     "ratios": {n: round(c["points"][n][1] / c["points"][n][0], 2) for n in sizes}}
        if c["groups"]:
            out[case]["groups"] = {n: c["groups"][n] for n in sorted(c["groups"])}
    return out


def fit_groups(cases):
    """{(class, dtype class, input, bucket): fit} from the cases that are a group-by alone and
    record their group counts. Every (case, input rows) point goes to the bucket of its group count
    at that size, for each class of the case; at each size a bucket's series is the worst of its
    cases there (the lowest fastest-Polars / MetalEngine ratio), so the fitted step is the largest
    of the cases' own, and a bucket measured at one size alone has no crossover."""
    pooled = {}
    for case, c in cases.items():
        classes, dclass, source = c["shape"]
        if not c["groups"] or {node_of(x) for x in classes} != {"group_by"}:
            continue
        for n, (metal, polars) in c["points"].items():
            b = group_bucket(c["groups"].get(n), n)
            if b is None:
                continue
            for cls in classes:
                pooled.setdefault((cls, dclass, source, b), {}).setdefault(n, []).append(
                    (case, metal, polars, c["groups"][n]))
    table = {}
    for key, by_size in sorted(pooled.items()):
        points = {}
        for n, pts in by_size.items():
            _case, metal, polars, _g = min(pts, key=lambda p: (p[2] / p[1], p[0]))
            points[n] = (metal, polars)
        cross, step, _low = fit_points("|".join(key), points, key[1], single=False)
        sizes = sorted(points)
        groups = [p[3] for pts in by_size.values() for p in pts]
        table[key] = {"rows": cross, "step": step, "largest": sizes[-1], "smallest": sizes[0],
                      "cases": sorted({p[0] for pts in by_size.values() for p in pts}),
                      "groups": (min(groups), max(groups)),
                      "ratios": {n: round(points[n][1] / points[n][0], 2) for n in sizes}}
    return table


def fit_classes(per_case):
    table = {}
    for case, f in per_case.items():
        classes, dclass, source = f["shape"]
        nodes = {node_of(c) for c in classes}
        if len(nodes) != 1:
            continue
        for cls in classes:
            e = table.setdefault((cls, dclass, source), {"rows": 0, "largest": 0, "cases": []})
            e["cases"].append(case)
            e["largest"] = max(e["largest"], f["largest"])
            if f["rows"] is None or e["rows"] is None:
                e["rows"] = None
            else:
                e["rows"] = max(e["rows"], f["rows"])
    for e in table.values():
        e["cases"].sort()
    return table


def sweep():
    with open(os.path.join(ROOT, SWEEP_JSON)) as fh:
        lib = json.load(fh)["vs_fastest_library"]
    return {cls: {"rows": max(lib[l] for l in labels), "labels": [l.split(": ", 1)[1] for l in labels]}
            for cls, labels in SWEEP_LABELS.items()}


def render(source, header, per_case, table, groups):
    lines = [
        '"""The MetalEngine\'s per-shape crossover table. Generated by Benchmarks/polars_engine_crossover.py',
        f"from {source}; do not edit by hand (`--check` fails when this file and that one disagree).",
        "",
        "ENGINE: {(class, dtype class, input): {\"rows\": crossover or None (not ahead at the largest size",
        "measured), \"largest\": the largest input rows measured, \"cases\": the sweep cases fitted}}.",
        "A case is ahead at a size when MetalEngine time x (1 + MARGIN[dtype class]) <= the faster Polars engine's.",
        "CASES: each case's own fit and its ratio (fastest Polars / MetalEngine) at every input size.",
        "SWEEP: the sort kernels' crossovers against the fastest CPU library, from " + SWEEP_JSON + ".",
        "GROUPS: {(class, dtype class, input, group-count bucket): the same fit over the points of the",
        "group-by cases whose group count at that size falls in the bucket (GROUP_BUCKETS, and ROWS_BUCKET",
        "for at least input rows / NEAR_ROWS groups), the worst case at each size; \"groups\": the fewest",
        "and most groups measured there}.",
        '"""',
        "",
        f"SOURCE = {source!r}",
        f"HEADER = {header!r}",
        f"SWEEP_SOURCE = {SWEEP_JSON!r}",
        f"MARGIN = {MARGIN!r}",
        f"STRING_FLOOR = {STRING_FLOOR!r}",
        f"GROUP_BUCKETS = {GROUP_BUCKETS!r}",
        f"NEAR_ROWS = {NEAR_ROWS!r}",
        f"ROWS_BUCKET = {ROWS_BUCKET!r}",
        "",
        "ENGINE = " + pprint.pformat(table, width=100, sort_dicts=True),
        "",
        "SWEEP = " + pprint.pformat(sweep(), width=100, sort_dicts=True),
        "",
        "GROUPS = " + pprint.pformat(groups, width=100, sort_dicts=True),
        "",
        "CASES = " + pprint.pformat(per_case, width=100, sort_dicts=True),
        "",
    ]
    return "\n".join(lines)


def generate(csv_path):
    source = os.path.relpath(os.path.abspath(csv_path), ROOT)
    header, cases = read_sweep(csv_path)
    per_case = fit_cases(cases)
    groups = fit_groups(cases)
    return render(source, header, per_case, fit_classes(per_case), groups), per_case, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", help="the sweep CSV (default with --check: the one the "
                                           "committed table names)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()
    path = args.csv
    if path is None:
        with open(OUT) as fh:
            path = os.path.join(ROOT, re.search(r"^SOURCE = '([^']+)'", fh.read(), re.M).group(1))
    text, per_case, groups = generate(path)
    if args.print:
        for case, f in per_case.items():
            x = f"{f['rows']:,}" if f["rows"] else "not reached"
            print(f"{case[:60]:<60} {'+'.join(f['shape'][0]):<34} {f['shape'][1]:<8} {f['shape'][2]:<8} "
                  f"{x:>12}  {f['ratios']}")
        table = fit_classes(per_case)
        print()
        for k in sorted(table):
            e = table[k]
            x = f"{e['rows']:,}" if e["rows"] else "not reached"
            print(f"{k[0]:<24} {k[1]:<8} {k[2]:<8} {x:>12}  up to {e['largest']:,}  {', '.join(e['cases'])}")
        print()
        order = [b[0] for b in GROUP_BUCKETS] + [ROWS_BUCKET]
        for k in sorted(groups, key=lambda k: (k[:3], order.index(k[3]))):
            e = groups[k]
            x = f"{e['rows']:,}" if e["rows"] else "not reached"
            print(f"{k[0]:<22} {k[1]:<8} {k[2]:<8} {k[3]:>10} groups {x:>12}  "
                  f"({e['groups'][0]:,} to {e['groups'][1]:,} groups measured)  {e['ratios']}")
    if args.check:
        with open(OUT) as fh:
            if fh.read() != text:
                print(f"{OUT} is stale: regenerate it from {path}", file=sys.stderr)
                return 1
        print(f"{OUT} matches {path}")
        return 0
    if args.csv:
        with open(OUT, "w") as fh:
            fh.write(text)
        print(f"wrote {os.path.relpath(OUT, ROOT)} from {os.path.relpath(path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
