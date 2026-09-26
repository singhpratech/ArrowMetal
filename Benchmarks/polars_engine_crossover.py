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
# below it the string sort cases sit within run-to-run noise of Polars (1.2-1.4x in the sweep, and
# 0.9x in a repeat of the default benchmark at 2,000,000 rows).
STRING_FLOOR = 5_000_000
METAL = "MetalEngine all, cold"


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
        c = cases.setdefault(case, {"shape": shape, "points": {}})
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
    return header, cases


def fit_cases(cases):
    fit = _load("_router_fit")
    out = {}
    for case, c in sorted(cases.items()):
        sizes = sorted(c["points"])
        bench = {}
        for n, (metal, polars) in c["points"].items():
            bench[(case, n, "gpu")] = metal * (1 + MARGIN[c["shape"][1]])
            bench[(case, n, "cpu")] = polars
        try:
            cross, step, low, _pts = fit.fit(case, "cpu", None, sizes, bench)
        except fit.FitError:
            cross = step = low = None
        if step is not None and step == sizes[-1] and len(sizes) > 1:
            # Ahead at the largest size alone is one measurement: not a crossover.
            cross = step = low = None
        if cross is not None and c["shape"][1] == "string" and cross < STRING_FLOOR:
            cross = STRING_FLOOR
            step = min((n for n in sizes if n >= STRING_FLOOR), default=step)
        out[case] = {"shape": c["shape"], "rows": cross, "step": step, "low": low,
                     "largest": sizes[-1], "smallest": sizes[0],
                     "ratios": {n: round(c["points"][n][1] / c["points"][n][0], 2) for n in sizes}}
    return out


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


def render(source, header, per_case, table):
    lines = [
        '"""The MetalEngine\'s per-shape crossover table. Generated by Benchmarks/polars_engine_crossover.py',
        f"from {source}; do not edit by hand (`--check` fails when this file and that one disagree).",
        "",
        "ENGINE: {(class, dtype class, input): {\"rows\": crossover or None (not ahead at the largest size",
        "measured), \"largest\": the largest input rows measured, \"cases\": the sweep cases fitted}}.",
        "A case is ahead at a size when MetalEngine time x (1 + MARGIN[dtype class]) <= the faster Polars engine's.",
        "CASES: each case's own fit and its ratio (fastest Polars / MetalEngine) at every input size.",
        "SWEEP: the sort kernels' crossovers against the fastest CPU library, from " + SWEEP_JSON + ".",
        '"""',
        "",
        f"SOURCE = {source!r}",
        f"HEADER = {header!r}",
        f"SWEEP_SOURCE = {SWEEP_JSON!r}",
        f"MARGIN = {MARGIN!r}",
        f"STRING_FLOOR = {STRING_FLOOR!r}",
        "",
        "ENGINE = " + pprint.pformat(table, width=100, sort_dicts=True),
        "",
        "SWEEP = " + pprint.pformat(sweep(), width=100, sort_dicts=True),
        "",
        "CASES = " + pprint.pformat(per_case, width=100, sort_dicts=True),
        "",
    ]
    return "\n".join(lines)


def generate(csv_path):
    source = os.path.relpath(os.path.abspath(csv_path), ROOT)
    header, cases = read_sweep(csv_path)
    per_case = fit_cases(cases)
    return render(source, header, per_case, fit_classes(per_case)), per_case


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
    text, per_case = generate(path)
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
