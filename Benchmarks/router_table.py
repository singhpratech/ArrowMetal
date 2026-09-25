#!/usr/bin/env python3
"""Generates the CPU/GPU router's crossover table from a measured sweep.

Two inputs are accepted, and the generated file says which one it came from.

`--json Benchmarks/results/router_<date>.json` (the default, written by `Benchmarks/crossover.py`):
its `vs_own_cpu_path` section gives, per routed operation, the first swept row count at which the
GPU path is at least as fast as a single-core CPU loop, and the `arrowmetal-bench crossover` CSV it
names under `sources.bench` holds the timings behind that step. The CPU side of that sweep is the
bench's own loops in Sources/ArrowMetalBench/main.swift (`cpu-1core`; `cpu-candidate` for the
group-by), measured on 2026-09-17 before Sources/ArrowMetal/Router/RouterCPU.swift existed. They
are not the loops the router runs.

`--from-check Benchmarks/results/router_check_<date>.csv` (written by `Benchmarks/router_check.py`):
the same operations timed with the router pinned to `gpu` and to `cpu`, so the CPU side is the
shipped RouterCPU loops. The step is the first size from which the GPU was at least as fast at
every larger size in the file. This is the input to regenerate the table from once a quiet run of
router_check.py exists.

`multiply` gets its own row. The 2026-09-17 sweep timed `add` only, and a 64-bit integer multiply
costs the CPU more per element than an add, so the arithmetic row does not carry over to it. The
multiply row is fitted the same way from the `multiply(int64, 3)` rows of a router_check.py CSV
(`--multiply-from`, default Benchmarks/results/router_check_2026-09-23_provisional.csv; with
`--from-check` the same file), whose CPU side is the shipped RouterCPU loop.

The step alone would send every size between the last CPU-wins point and the first GPU-wins point to
the CPU, however close the GPU already is. So the table is fitted once, offline, as the router section
of docs/DESIGN.md describes: inside that bracket both paths are taken as straight lines between the
two measured points, and the crossover is where the lines meet. The fitted value always lies inside
the measured bracket; nothing is extrapolated.

Output: `Sources/ArrowMetal/Router/RouterTable.swift`, a Swift literal (no resource bundle to load).
With `--json-out PATH` (and `--from-check`) the same table is also written as JSON in the format the
router loads at run time (`am_router_load_table`, ARROWMETAL_ROUTER_TABLE), the one
`python -m arrowmetal.router calibrate` writes for a single machine. The fitting code is shared with
calibrate: python/arrowmetal/_router_fit.py, loaded here by path so no built library is needed.

Usage:
    python Benchmarks/router_table.py                      # regenerate from the default JSON
    python Benchmarks/router_table.py --check              # exit 1 if the committed table is stale
                                                           # (regenerated from the source it names)
    python Benchmarks/router_table.py --json Benchmarks/results/router_<date>.json
    python Benchmarks/router_table.py --from-check Benchmarks/results/router_check_<date>.csv
    python Benchmarks/router_table.py --multiply-from Benchmarks/results/router_check_<date>.csv
    python Benchmarks/router_table.py --from-check Benchmarks/results/router_check_<date>.csv \
        --json-out router_table.json               # also write the JSON form
"""
import argparse
import importlib.util
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_fit():
    path = os.path.join(ROOT, "python", "arrowmetal", "_router_fit.py")
    spec = importlib.util.spec_from_file_location("arrowmetal_router_fit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_fit = _load_fit()
DEFAULT_JSON = os.path.join(ROOT, "Benchmarks", "results", "router_2026-09-17.json")
DEFAULT_MULTIPLY = os.path.join(ROOT, "Benchmarks", "results", "router_check_2026-09-23_provisional.csv")
MULTIPLY_LABEL = "multiply(int64, 3)"
OUT = os.path.join(ROOT, "Sources", "ArrowMetal", "Router", "RouterTable.swift")

# Bench label -> (RoutedOp case, CPU path label in the bench CSV). The order is the C ABI op number.
OPS = [
    ("sum(int64)", "sum", "cpu-1core"),
    ("min(int64)", "min", "cpu-1core"),
    ("max(int64)", "max", "cpu-1core"),
    ("compare(int64 > 0)", "compare", "cpu-1core"),
    ("add(int64, 1)", "arithmetic", "cpu-1core"),
    ("filter(int64, mask)", "filter", "cpu-1core"),
    ("group-by sum (1000 keys)", "groupBySum", "cpu-candidate"),
]


def load_bench(path):
    rows, header = {}, ""
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                header = line[1:].strip()
                continue
            if line.startswith("op,"):
                continue
            op, n, p, wall, _cpu, _ = line.rstrip("\n").rsplit(",", 5)  # op names may hold commas
            rows[(op, int(n), p)] = float(wall)
    return rows, header


def fit(label, cpu_label, step, sizes, bench):
    try:
        return _fit.fit(label, cpu_label, step, sizes, bench)
    except _fit.FitError as e:
        raise SystemExit(str(e))


def entries_from_json(json_path):
    with open(json_path) as fh:
        data = json.load(fh)
    results = os.path.dirname(os.path.abspath(json_path))
    bench_name = data["sources"]["bench"]
    bench, header = load_bench(os.path.join(results, bench_name))
    sizes = data["sizes"]
    steps = data["vs_own_cpu_path"]
    entries = []
    for label, case, cpu_label in OPS:
        if steps.get(label) is None:
            raise SystemExit(f"{label}: no crossover in {json_path}")
        cross, step, lo, pts = fit(label, cpu_label, int(steps[label]), sizes, bench)
        entries.append((label, case, step, cross, lo, pts))
    rel = os.path.relpath(json_path, ROOT)
    source = [
        "// Generated by Benchmarks/router_table.py from " + rel + " and",
        "// Benchmarks/results/" + bench_name + ". Do not edit by hand: rerun the script.",
        "// Bench header: " + header,
        "//",
        "// The CPU side of these timings is the bench's own single-core loops in",
        "// Sources/ArrowMetalBench/main.swift (`cpu-1core`; `cpu-candidate` for the group-by), measured",
        "// before Sources/ArrowMetal/Router/RouterCPU.swift existed, not the RouterCPU loops the router",
        "// runs. Regenerate from a quiet run of the shipped loops with",
        "// `Benchmarks/router_table.py --from-check Benchmarks/results/router_check_<date>.csv`.",
    ]
    cpu_side = "a single-core CPU loop of the 2026-09-17 bench (not the RouterCPU loops; see the header)"
    return entries, rel, source, cpu_side, header


load_check = _fit.load_check


def multiply_entry(csv_path):
    """(label, step, crossover, bracket low, points, relative path, check header) for the multiply row."""
    header, bench, sizes = load_check(csv_path)
    if not any(k[0] == MULTIPLY_LABEL for k in bench):
        raise SystemExit(f"{MULTIPLY_LABEL}: not in {csv_path}")
    cross, step, lo, pts = fit(MULTIPLY_LABEL, "cpu", None, sizes, bench)
    return MULTIPLY_LABEL, step, cross, lo, pts, os.path.relpath(csv_path, ROOT), header


def entries_from_check(csv_path):
    header, bench, sizes = load_check(csv_path)
    entries = []
    for label, case, _ in OPS:
        if not any(k[0] == label for k in bench):
            raise SystemExit(f"{label}: not in {csv_path}")
        cross, step, lo, pts = fit(label, "cpu", None, sizes, bench)
        entries.append((label, case, step, cross, lo, pts))
    rel = os.path.relpath(csv_path, ROOT)
    source = [
        "// Generated by Benchmarks/router_table.py from " + rel + ".",
        "// Do not edit by hand: rerun the script.",
        "// Check header: " + header,
        "//",
        "// The CPU side of these timings is the shipped RouterCPU loops, timed by",
        "// Benchmarks/router_check.py with the router pinned to `cpu`, against the GPU pinned to `gpu`.",
    ]
    cpu_side = "the router's single-core CPU loop (RouterCPU), as timed by Benchmarks/router_check.py"
    return entries, rel, source, cpu_side, header


def generate(path, multiply_path=None):
    if path.endswith(".csv"):
        entries, rel, source, cpu_side, header = entries_from_check(path)
        multiply_path = path
    else:
        entries, rel, source, cpu_side, header = entries_from_json(path)
        multiply_path = multiply_path or DEFAULT_MULTIPLY
    m_label, m_step, m_cross, m_lo, m_pts, m_rel, m_header = multiply_entry(multiply_path)
    if m_rel != rel:
        source = source + [
            "//",
            "// The multiply row comes from " + m_rel + ",",
            "// since the sweep above timed add only. Its CPU side is the shipped RouterCPU multiply loop,",
            "// timed by Benchmarks/router_check.py. Check header: " + m_header,
        ]
    out = source + [
        "",
        "/// The router's crossover table: per routed operation, the row count from which the GPU path",
        "/// was measured at least as fast as " + cpu_side + ".",
        "/// Measured on int64 columns with 10% nulls; docs/DESIGN.md (CPU/GPU router) says how the table",
        "/// is used.",
        "enum RouterTable {",
        "    /// Results file the table was generated from.",
        f"    static let source = \"{rel}\"",
        "",
        "    /// Machine and protocol line of that results file.",
        f"    static let header = \"{swift_string(header)}\"",
        "",
        "    /// Fitted crossover in rows (straight lines between the two bracketing measured sizes).",
        "    static func crossoverRows(_ op: RoutedOp) -> Int {",
        "        switch op {",
    ]
    for label, case, step, cross, lo, pts in entries:
        pt = "; ".join(f"{n:,} rows: GPU {g:g} us, CPU {c:g} us" for n, g, c in pts)
        out.append(f"        case .{case}: return {cross}   // {label}; {pt}")
    out += ["        }", "    }", "",
            "    /// First measured size from which the GPU stayed ahead (the step).",
            "    static func measuredStepRows(_ op: RoutedOp) -> Int {",
            "        switch op {"]
    for label, case, step, cross, lo, pts in entries:
        out.append(f"        case .{case}: return {step}")
    out += ["        }", "    }", "",
            "    /// Largest measured size below the step (the CPU was measured ahead there).",
            "    static func bracketLowRows(_ op: RoutedOp) -> Int {",
            "        switch op {"]
    for label, case, step, cross, lo, pts in entries:
        out.append(f"        case .{case}: return {lo}")
    out += ["        }", "    }", "",
            "    /// The results file's label for the case each row was fitted from.",
            "    static func label(_ op: RoutedOp) -> String {",
            "        switch op {"]
    for label, case, step, cross, lo, pts in entries:
        out.append(f"        case .{case}: return \"{swift_string(label)}\"")
    out += ["        }", "    }", "",
            "    /// The two measured points each crossover was fitted between: (rows, GPU us, CPU us).",
            "    static func bracketPoints(_ op: RoutedOp) -> [(Int, Double, Double)] {",
            "        switch op {"]
    for label, case, step, cross, lo, pts in entries:
        out.append(f"        case .{case}: return {swift_points(pts)}")
    m_pt = "; ".join(f"{n:,} rows: GPU {g:g} us, CPU {c:g} us" for n, g, c in m_pts)
    out += ["        }", "    }", "",
            "    /// Results file the multiply row was fitted from (a router_check.py CSV).",
            f"    static let multiplySource = \"{m_rel}\"",
            "",
            "    /// `multiply`'s own crossover: the arithmetic row above was measured on add (subtract follows it).",
            f"    static let multiplyCrossoverRows = {m_cross}   // {m_label}; {m_pt}",
            f"    static let multiplyMeasuredStepRows = {m_step}",
            f"    static let multiplyBracketLowRows = {m_lo}",
            f"    static let multiplyLabel = \"{swift_string(m_label)}\"",
            f"    static let multiplyBracketPoints: [(Int, Double, Double)] = {swift_points(m_pts)}",
            "}"]
    return "\n".join(out) + "\n"


def swift_string(text):
    return text.replace("\\", "\\\\").replace('"', '\\"')


def swift_points(pts):
    return "[" + ", ".join(f"({n}, {float(g)!r}, {float(c)!r})" for n, g, c in pts) + "]"


def json_table(csv_path):
    """The table as JSON (python/arrowmetal/_router_fit.py's format), fitted from a router check CSV."""
    header, bench, sizes = load_check(csv_path)
    machine, date = {}, None
    m = re.search(r" on (.+?), (\d+) CPU cores, (.+?), ", header)
    if m:
        machine = {"chip": m.group(1), "chip_id": _fit.chip_id(m.group(1)), "cpu_cores": int(m.group(2)),
                   "metal_device": m.group(3)}
    d = re.search(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)", header)
    if d:
        date = d.group(1)
    for label in _fit.TABLE_LABELS:
        if not any(k[0] == label for k in bench):
            raise SystemExit(f"{label}: not in {csv_path}")
    table = _fit.table_json(bench, sizes, header, os.path.relpath(csv_path, ROOT), machine=machine, date=date,
                            grid={"name": "router_check", "sizes": sizes})
    missing = [op for op, e in table["crossovers"].items() if e["crossover_rows"] is None]
    if missing:
        raise SystemExit(f"no crossover for {', '.join(missing)} in {csv_path}")
    return table


def committed_source(out_path, name="source"):
    """The results file a committed table names in `static let <name>`, or None."""
    try:
        with open(out_path) as fh:
            m = re.search(r'static let ' + name + r' = "([^"]+)"', fh.read())
    except FileNotFoundError:
        return None
    return os.path.join(ROOT, m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--json", help="router_<date>.json from Benchmarks/crossover.py (default: " +
                     os.path.relpath(DEFAULT_JSON, ROOT) + ")")
    src.add_argument("--from-check", help="router_check_<date>.csv from Benchmarks/router_check.py")
    ap.add_argument("--multiply-from", help="with --json: router_check_<date>.csv to fit the multiply row from "
                    "(default: " + os.path.relpath(DEFAULT_MULTIPLY, ROOT) + ")")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--json-out", help="with --from-check: also write the table as JSON (the router's run-time format)")
    ap.add_argument("--check", action="store_true",
                    help="compare with the committed table (regenerated from the source it names) instead of writing")
    args = ap.parse_args()
    if args.from_check and args.multiply_from:
        ap.error("--multiply-from goes with --json; --from-check fits every row from its own file")
    if args.json_out and not args.from_check:
        ap.error("--json-out needs --from-check (the JSON form is fitted from a router check CSV)")
    path = args.json or args.from_check
    multiply = args.multiply_from
    if path is None:
        path = (committed_source(args.out) if args.check else None) or DEFAULT_JSON
    if multiply is None and args.check and not args.from_check:
        multiply = committed_source(args.out, "multiplySource")
    text = generate(os.path.abspath(path), os.path.abspath(multiply) if multiply else None)
    if args.check:
        try:
            with open(args.out) as fh:
                current = fh.read()
        except FileNotFoundError:
            current = ""
        if current != text:
            print(f"{os.path.relpath(args.out, ROOT)} is stale; rerun Benchmarks/router_table.py", file=sys.stderr)
            return 1
        print("router table up to date")
        return 0
    with open(args.out, "w") as fh:
        fh.write(text)
    print(f"wrote {os.path.relpath(args.out, ROOT)}")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(json_table(os.path.abspath(args.from_check)), fh, indent=1)
            fh.write("\n")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
