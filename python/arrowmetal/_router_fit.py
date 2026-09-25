"""Fitting the router's crossover table from a router check sweep, and the table's JSON form.

Standard library only, and no import of the rest of the package, so Benchmarks/router_table.py can
load this file by path without a built library. Used by:

* `Benchmarks/router_table.py`, which writes the shipped table (Sources/ArrowMetal/Router/RouterTable.swift)
  and, with `--json-out`, the same table as JSON;
* `python -m arrowmetal.router calibrate`, which writes this machine's table as JSON.

The JSON format ("arrowmetal-router-table/1") is read by the Swift side
(Sources/ArrowMetal/Router/RouterCrossovers.swift, `am_router_load_table`).
"""
import csv
import math
import re

FORMAT = "arrowmetal-router-table/1"

# Check label -> routed operation (the RoutedOp raw value; the order is the C ABI op number).
OPS = [
    ("sum(int64)", "sum"),
    ("min(int64)", "min"),
    ("max(int64)", "max"),
    ("compare(int64 > 0)", "compare"),
    ("add(int64, 1)", "arithmetic"),
    ("filter(int64, mask)", "filter"),
    ("group-by sum (1000 keys)", "group_by_sum"),
]
MULTIPLY_LABEL = "multiply(int64, 3)"
TABLE_LABELS = [label for label, _ in OPS] + [MULTIPLY_LABEL]

# The shape every row of the table was measured on.
SHAPE = {"dtype": "int64", "null_fraction": 0.1, "resident": True}


class FitError(ValueError):
    pass


def chip_id(chip):
    """'Apple M4 Max' -> 'apple-m4-max': the file name of a chip's table (RouterCrossovers.chipID)."""
    out = re.sub(r"[^a-z0-9]+", "-", chip.lower()).strip("-")
    return out or "unknown"


def parse_check(lines):
    """(header, {(label, rows, 'gpu'|'cpu'): us}, sorted sizes) from router check CSV lines."""
    header, bench, sizes = "", {}, set()
    lines = list(lines)
    if lines and lines[0].startswith("#"):
        header = lines[0][1:].strip()
        lines = lines[1:]
    for row in csv.DictReader(lines):
        n = int(row["rows"])
        sizes.add(n)
        bench[(row["op"], n, "gpu")] = float(row["gpu_us"])
        bench[(row["op"], n, "cpu")] = float(row["cpu_us"])
    return header, bench, sorted(sizes)


def load_check(path):
    with open(path) as fh:
        return parse_check(fh.read().splitlines())


def fit(label, cpu_label, step, sizes, bench):
    """(crossover rows, step, bracket low, points) for one operation. `step` None: take it from the
    timings (the first size from which the GPU is at least as fast at every larger size).

    Inside the bracket (the last size where the CPU was ahead, the step) both paths are taken as
    straight lines between the two measured points and the crossover is where they meet; it never
    leaves the bracket. Raises FitError when the GPU is not ahead at the largest measured size."""
    measured = [n for n in sizes if (label, n, "gpu") in bench and (label, n, cpu_label) in bench]
    first_gpu = next((n for n in measured
                      if all(bench[(label, m, "gpu")] <= bench[(label, m, cpu_label)] for m in measured if m >= n)),
                     None)
    if step is None:
        if first_gpu is None:
            raise FitError(f"{label}: the GPU is not ahead at the largest measured size")
        step = first_gpu
    if step not in measured:
        raise FitError(f"{label}: step crossover {step} is not a measured size")
    if first_gpu != step:
        raise FitError(f"{label}: the bench says the GPU stays ahead from {first_gpu}, the JSON says {step}")
    lower = [n for n in measured if n < step]
    if not lower:
        return step, step, step, []
    lo, hi = lower[-1], step
    g0, c0 = bench[(label, lo, "gpu")], bench[(label, lo, cpu_label)]
    g1, c1 = bench[(label, hi, "gpu")], bench[(label, hi, cpu_label)]
    d0, d1 = c0 - g0, c1 - g1          # CPU minus GPU: negative at lo (CPU ahead), >= 0 at hi
    if d0 >= 0 or d1 < 0:
        cross = hi
    else:
        cross = lo + (hi - lo) * (-d0) / (d1 - d0)
    cross = int(min(hi, max(lo + 1, math.ceil(cross))))
    return cross, step, lo, [(lo, g0, c0), (hi, g1, c1)]


def _entry(label, bench, sizes):
    measured = sorted(n for n in sizes if (label, n, "gpu") in bench and (label, n, "cpu") in bench)
    if not measured:
        return {"label": label, "crossover_rows": None, "not_reached": "not in the sweep"}
    try:
        cross, step, lo, pts = fit(label, "cpu", None, sizes, bench)
    except FitError:
        return {"label": label, "crossover_rows": None,
                "not_reached": f"the GPU was not ahead at the largest measured size ({measured[-1]:,} rows)"}
    return {"label": label, "crossover_rows": cross, "step_rows": step, "bracket_low_rows": lo,
            "points": [{"rows": n, "gpu_us": g, "cpu_us": c} for n, g, c in pts]}


def table_json(bench, sizes, header, source, machine=None, date=None, version=None, grid=None, measurements=None):
    """The JSON table (a dict) fitted from router check timings. An operation whose sweep never put
    the GPU ahead gets "crossover_rows": null, and the router keeps the shipped row for it."""
    crossovers = {op: _entry(label, bench, sizes) for label, op in OPS}
    crossovers["multiply"] = _entry(MULTIPLY_LABEL, bench, sizes)
    out = {"format": FORMAT, "machine": machine or {}, "date": date, "arrowmetal_version": version,
           "grid": grid or {"sizes": sizes}, "shape": dict(SHAPE), "source": source, "header": header,
           "crossovers": crossovers}
    if measurements is not None:
        out["measurements"] = measurements
    return out


def crossover_for(table, op, shipped):
    """The crossover a JSON table gives `op` ("sum" ... "group_by_sum", "multiply"), falling back to
    `shipped[op]` when the table has none, as the router does."""
    e = (table.get("crossovers") or {}).get(op) or {}
    c = e.get("crossover_rows")
    return (int(c), "table") if c else (shipped[op], "shipped")


def check_rows_from_measurements(measurements):
    """{(label, rows, 'gpu'|'cpu'): us} from a JSON table's "measurements" list."""
    bench = {}
    for m in measurements:
        bench[(m["op"], m["rows"], "gpu")] = m["gpu_us"]
        bench[(m["op"], m["rows"], "cpu")] = m["cpu_us"]
    return bench
