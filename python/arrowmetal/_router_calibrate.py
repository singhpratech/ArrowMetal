"""The router check sweep, per-machine calibration and `explain`, behind `python -m arrowmetal.router`.

The sweep times every routed operation through this binding on Metal-resident arrays, with the router
pinned to `gpu` and to `cpu` (and, on the full grid, under `auto`), exactly as
Benchmarks/router_check.py does; that script now calls `sweep` and `write_check_csv` here. The
fitting and the JSON format are in _router_fit.py, shared with Benchmarks/router_table.py.
"""
import datetime
import json
import os
import platform
import subprocess
import sys
import time

from . import _router_fit as fit

# The full grid is Benchmarks/router_check.py's: the sizes and repetitions the shipped table was
# fitted from, all twelve cases, and `auto` timed alongside.
FULL_SIZES = [1_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000]
# The quick grid: the cases the table is fitted from, fewer repetitions, no `auto` column.
QUICK_SIZES = [30_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000]

GRIDS = {
    "full": {"sizes": FULL_SIZES, "reps": 20, "reps_large": 5, "modes": ("gpu", "cpu", "auto"), "cases": None},
    "quick": {"sizes": QUICK_SIZES, "reps": 10, "reps_large": 5, "modes": ("gpu", "cpu"), "cases": fit.TABLE_LABELS},
}
LARGE = 3_000_000          # at and above this many rows the grid's `reps_large` applies

# Check label -> the operation the router decides it as (the table row it reads).
DECIDED_AS = {
    "sum(int64)": "sum", "min(int64)": "min", "max(int64)": "max",
    "compare(int64 > 0)": "compare", "compare(int64 > int64)": "compare",
    "add(int64, 1)": "arithmetic", "add(int64 + int64)": "arithmetic", "subtract(int64, 1)": "arithmetic",
    "multiply(int64, 3)": "multiply",
    "filter(int64, mask)": "filter", "filter_where(int64 > 0)": "filter",
    "group-by sum (1000 keys)": "group_by_sum",
}


# ---- the machine

def _sysctl(name):
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _gpu_cores():
    try:
        out = subprocess.run(["ioreg", "-rc", "AGXAccelerator", "-d", "1"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.stdout.splitlines():
        if '"gpu-core-count"' in line:
            v = line.rsplit("=", 1)[-1].strip()
            return int(v) if v.isdigit() else None
    return None


def machine_info():
    """Chip, core counts, memory, macOS and the Metal device: the header of a per-machine table."""
    import arrowmetal as am
    chip = _sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown"

    def num(name):
        v = _sysctl(name)
        return int(v) if v.isdigit() else None
    mem = num("hw.memsize")
    return {"chip": chip, "chip_id": fit.chip_id(chip), "metal_device": am.device_name(),
            "cpu_cores": os.cpu_count() or 0, "performance_cores": num("hw.perflevel0.physicalcpu"),
            "efficiency_cores": num("hw.perflevel1.physicalcpu"), "gpu_cores": _gpu_cores(),
            "memory_gb": round(mem / 2**30) if mem else None, "macos": platform.mac_ver()[0] or platform.release()}


def default_table_path(chip=None):
    """~/.arrowmetal/router/<chip id>.json, the file the router loads for this chip."""
    chip = chip or _sysctl("machdep.cpu.brand_string") or "unknown"
    return os.path.join(os.path.expanduser("~"), ".arrowmetal", "router", fit.chip_id(chip) + ".json")


# ---- the sweep

def best_us(fn, reps):
    fn()                                           # warm-up: pipelines, pool
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        best = min(best, (time.perf_counter_ns() - t0) / 1e3)
    return best


def cases(n, rng):
    """(label, call) for every routed case at `n` rows: int64 columns with 10% nulls, resident."""
    import numpy as np
    import pyarrow as pa
    import arrowmetal as am
    vals = rng.integers(-1000, 1001, n, dtype=np.int64)
    col = am.array(pa.array(vals, mask=rng.random(n) < 0.1))
    other = am.array(pa.array(rng.integers(-1000, 1001, n, dtype=np.int64), mask=rng.random(n) < 0.1))
    mask = col.compare(">", 0)
    keys = am.array(pa.array(rng.integers(0, 1000, n, dtype=np.int32)))
    gb = am.GroupBy(keys, 1000)
    return [
        ("sum(int64)", col.sum),
        ("min(int64)", col.min),
        ("max(int64)", col.max),
        ("compare(int64 > 0)", lambda: col.compare(">", 0)),
        ("compare(int64 > int64)", lambda: col.compare(">", other)),
        ("add(int64, 1)", lambda: col.arith("+", 1)),
        ("add(int64 + int64)", lambda: col.arith("+", other)),
        ("subtract(int64, 1)", lambda: col.arith("-", 1)),
        ("multiply(int64, 3)", lambda: col.arith("*", 3)),
        ("filter(int64, mask)", lambda: col.filter(mask)),
        ("filter_where(int64 > 0)", lambda: col.filter_where(">", 0)),
        ("group-by sum (1000 keys)", lambda: gb.sum(col)),
    ]


def sweep(sizes, reps=20, reps_large=5, modes=("gpu", "cpu", "auto"), only=None, log=None):
    """Times each case at each size under each mode; one dict per (case, size) with gpu_us, cpu_us,
    auto_us / auto_path / auto_reason (when `auto` is timed), faster_path and margin."""
    import numpy as np
    import arrowmetal as am
    rows = []
    rng = np.random.default_rng(2026)
    for n in sizes:
        r = reps_large if n >= LARGE else reps
        for label, fn in cases(n, rng):
            if only is not None and label not in only:
                continue
            t, auto_path, reason = {}, None, None
            for mode in modes:
                with am.router(mode):
                    t[mode] = best_us(fn, r)
                    d = am.last_route()
                if mode == "auto":
                    auto_path, reason = d.path, d.reason
            faster = "gpu" if t["gpu"] <= t["cpu"] else "cpu"
            margin = max(t["gpu"], t["cpu"]) / max(min(t["gpu"], t["cpu"]), 1e-9)
            row = {"op": label, "rows": n, "gpu_us": round(t["gpu"], 1), "cpu_us": round(t["cpu"], 1),
                   "faster_path": faster, "margin": round(margin, 2)}
            if "auto" in t:
                row.update(auto_us=round(t["auto"], 1), auto_path=auto_path, auto_reason=reason,
                           auto_picked_faster=auto_path == faster)
            rows.append(row)
            if log:
                extra = f"  auto {t['auto']:9.1f} ({auto_path})" if "auto" in t else ""
                log(f"{label:28s} {n:>10,d}  gpu {t['gpu']:9.1f}  cpu {t['cpu']:9.1f}{extra}  "
                    f"(faster {faster}, margin {margin:.2f})")
    return rows


def check_header(machine, reps, reps_large, stamp, note=""):
    return (f"ArrowMetal router check on {machine['chip']}, {machine['cpu_cores']} CPU cores, "
            f"{machine['metal_device']}, best of {reps} ({reps_large} at 3M rows and above), int64 with 10% "
            f"nulls, Python binding on resident arrays, {stamp}" + (f"; {note}" if note else ""))


def write_check_csv(path, rows, header):
    """The router check CSV (Benchmarks/router_check.py's format; `router_table.py --from-check` reads it)."""
    with open(path, "w") as fh:
        fh.write("# " + header + "\n")
        fh.write("op,rows,gpu_us,cpu_us,auto_us,auto_path,faster_path,auto_picked_faster,margin,auto_reason\n")
        for r in rows:
            picked = "" if r.get("auto_path") is None else ("yes" if r["auto_picked_faster"] else "no")
            auto_us = f"{r['auto_us']:.1f}" if "auto_us" in r else ""
            fh.write(f"\"{r['op']}\",{r['rows']},{r['gpu_us']:.1f},{r['cpu_us']:.1f},{auto_us},"
                     f"{r.get('auto_path') or ''},{r['faster_path']},{picked},{r['margin']:.2f},"
                     f"\"{r.get('auto_reason') or ''}\"\n")


# ---- calibrate

def compare_tables(rows, machine_cross, shipped_cross):
    """Decisions of the two tables on the sweep's (case, size) pairs: how often they agree, and how
    often each picked the path this sweep measured faster."""
    agree = mach_faster = ship_faster = 0
    diffs = []
    for r in rows:
        op = DECIDED_AS[r["op"]]
        m = "cpu" if r["rows"] < machine_cross[op] else "gpu"
        s = "cpu" if r["rows"] < shipped_cross[op] else "gpu"
        agree += m == s
        mach_faster += m == r["faster_path"]
        ship_faster += s == r["faster_path"]
        if m != s:
            diffs.append((r["op"], r["rows"], m, s, r["faster_path"], r["margin"]))
    return {"pairs": len(rows), "agree": agree, "machine_picked_faster": mach_faster,
            "shipped_picked_faster": ship_faster, "differences": diffs}


def calibrate(quick=False, out=None, csv_out=None, log=print, progress=True):
    """Runs the sweep on this machine, fits the table and writes it as JSON. Returns (path, table)."""
    import arrowmetal as am
    grid_name = "quick" if quick else "full"
    g = GRIDS[grid_name]
    machine = machine_info()
    out = out or default_table_path(machine["chip"])
    only = set(g["cases"]) if g["cases"] else None
    ncases = len(only) if only else 12
    log(f"arrowmetal router calibrate: {grid_name} grid, {ncases} cases x {len(g['sizes'])} sizes "
        f"({', '.join(f'{n:,}' for n in g['sizes'])} rows), best of {g['reps']} ({g['reps_large']} at "
        f"{LARGE:,} rows and above), modes {', '.join(g['modes'])}; {machine['chip']}, "
        f"{machine['cpu_cores']} CPU cores, Metal device {machine['metal_device']}")
    t0 = time.perf_counter()
    rows = sweep(g["sizes"], g["reps"], g["reps_large"], g["modes"], only, log=log if progress else None)
    elapsed = time.perf_counter() - t0
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = check_header(machine, g["reps"], g["reps_large"], stamp, f"{grid_name} grid, python -m arrowmetal.router calibrate")
    bench = fit.check_rows_from_measurements(rows)
    grid = {"name": grid_name, "sizes": g["sizes"], "reps": g["reps"], "reps_large_from_rows": LARGE,
            "reps_large": g["reps_large"], "modes": list(g["modes"]), "cases": sorted({r["op"] for r in rows}),
            "elapsed_s": round(elapsed, 1)}
    table = fit.table_json(bench, sorted({r["rows"] for r in rows}), header,
                           "python -m arrowmetal.router calibrate" + (" --quick" if quick else ""),
                           machine=machine, date=stamp, version=am.__version__, grid=grid, measurements=rows)
    info = am.router_table()
    shipped = info["shipped_crossovers"]
    machine_cross = {op: fit.crossover_for(table, op, shipped)[0] for op in shipped}
    table["comparison_with_shipped"] = {k: v for k, v in compare_tables(rows, machine_cross, shipped).items()
                                        if k != "differences"}
    timed = [r for r in rows if r.get("auto_path")]
    if timed:
        table["comparison_with_shipped"]["auto_picked_faster"] = sum(r["auto_picked_faster"] for r in timed)
        table["comparison_with_shipped"]["auto_table"] = "shipped" if info["shipped"] else info.get("path")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(table, fh, indent=1)
        fh.write("\n")
    os.replace(tmp, out)
    if csv_out:
        write_check_csv(csv_out, rows, header)
    report(table, out, shipped, rows, machine_cross, csv_out, log)
    return out, table


def report(table, out, shipped, rows, machine_cross, csv_out, log):
    g = table["grid"]
    log("")
    log(f"{g['name']} grid ran in {g['elapsed_s']:.1f} s")
    log(f"{'operation':14s} {'this machine':>14s} {'shipped':>12s}  fitted between")
    for op, e in table["crossovers"].items():
        if e["crossover_rows"] is None:
            log(f"{op:14s} {'not reached':>14s} {shipped[op]:>12,d}  {e['not_reached']}; the shipped row applies")
            continue
        pts = e.get("points") or []
        between = " and ".join(f"{p['rows']:,} rows (GPU {p['gpu_us']:g} us, CPU {p['cpu_us']:g} us)" for p in pts) \
            or f"GPU ahead from the smallest size, {e['step_rows']:,} rows"
        log(f"{op:14s} {e['crossover_rows']:>14,d} {shipped[op]:>12,d}  {between}")
    c = compare_tables(rows, machine_cross, shipped)
    log(f"decisions on this sweep's {c['pairs']} (case, size) pairs: this machine's table and the shipped "
        f"table agree on {c['agree']}; the path measured faster here is picked by this machine's table in "
        f"{c['machine_picked_faster']} and by the shipped table in {c['shipped_picked_faster']}")
    for label, n, m, s, f, margin in c["differences"]:
        log(f"  {label} at {n:,} rows: this machine {m}, shipped {s}, measured faster {f} (margin {margin:.2f})")
    timed = [r for r in rows if r.get("auto_path")]
    if timed:
        log(f"auto, timed during the sweep under the table then in force, took the faster path in "
            f"{sum(r['auto_picked_faster'] for r in timed)} of {len(timed)} cases")
    log(f"wrote {out}")
    if csv_out:
        log(f"wrote {csv_out} (router check CSV: Benchmarks/router_table.py --from-check reads it)")
    env = os.environ.get("ARROWMETAL_ROUTER_TABLE")
    if env:
        log(f"ARROWMETAL_ROUTER_TABLE is set ({env}); it takes precedence over {out} in new processes")
    elif os.path.abspath(out) == os.path.abspath(default_table_path(table["machine"].get("chip"))):
        log("new processes on this machine load it at startup; ARROWMETAL_ROUTER_TABLE=shipped keeps the shipped table")
    else:
        log(f"load it with ARROWMETAL_ROUTER_TABLE={out} or am.load_router_table({out!r})")


# ---- explain

def table_origin(t):
    if t.get("shipped"):
        return "shipped (built into the library)"
    path = t.get("path", "")
    env = os.environ.get("ARROWMETAL_ROUTER_TABLE")
    if env and os.path.abspath(env) == os.path.abspath(path):
        return f"ARROWMETAL_ROUTER_TABLE ({path})"
    if os.path.abspath(path) == os.path.abspath(default_table_path()):
        return f"this machine's ({path})"
    return f"loaded from {path}"


def explain_text(e, nulls=None):
    """The lines `explain` prints for one `am.explain_route` result."""
    t, row = e["table"], e["row"]
    mode = e["mode"]
    L = [f"{e['operation']} over {e['rows']:,} {e['dtype']} rows: mode {mode} -> {e['path']}"
         + (" (per-thread override)" if e.get("thread_override") else "")]
    L.append(f"  reason: {e['reason']}")
    L.append(f"  decided as: {e['routed_op']}" + (f" ({e['key_count']:,} keys)" if e["routed_op"] == "group_by_sum" else ""))
    pts = row.get("points") or []
    fitted = " and ".join(f"{p['rows']:,} rows (GPU {p['gpu_us']:g} us, CPU {p['cpu_us']:g} us)" for p in pts)
    consulted = e["reason"].startswith(("below the", "at or above the"))
    L.append(f"  table row{'' if consulted else ' (not consulted for this decision)'}: {row['label']}, "
             f"crossover {row['crossover_rows']:,} rows"
             + (f", fitted between {fitted}" if fitted else f", GPU ahead from {row['step_rows']:,} rows"))
    L.append(f"  table: {table_origin(t)}")
    L.append(f"  source: {t['source']}")
    if t.get("header"):
        L.append(f"  measured: {t['header']}")
    if t.get("shipped_ops"):
        L.append(f"  rows from the shipped table: {', '.join(t['shipped_ops'])}")
    if t.get("load_error"):
        L.append(f"  load error: {t['load_error']}")
    if nulls is not None:
        L.append(f"  null fraction {nulls:g}: the table has one null-fraction bucket (measured at "
                 f"{fit.SHAPE['null_fraction']:.0%} nulls), so it does not change the decision")
    L.append("  residency: routed calls read Metal-shared buffers, the state the table was measured in")
    return L


# ---- the command

def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="python -m arrowmetal.router",
                                description="The CPU/GPU router's crossover table: calibrate it on this Mac, or explain one decision.")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate", help="run the router check sweep here and write this machine's table")
    c.add_argument("--quick", action="store_true", help="the shorter grid (table cases only, fewer repetitions)")
    c.add_argument("--out", help="where to write the JSON table (default ~/.arrowmetal/router/<chip>.json)")
    c.add_argument("--csv", help="also write the sweep as a router check CSV")
    c.add_argument("--quiet", action="store_true", help="print the summary only")
    x = sub.add_parser("explain", help="print the decision for one operation and size, and where it came from")
    x.add_argument("op", help="sum, min, max, compare, add, subtract, multiply, divide, filter, filter_where, group_by_sum")
    x.add_argument("rows", type=lambda s: int(s.replace(",", "").replace("_", "")))
    x.add_argument("--dtype", default="int64", help="int8 ... uint64, float32, float64 (default int64)")
    x.add_argument("--nulls", type=float, default=None, help="null fraction of the column, 0 to 1")
    x.add_argument("--keys", type=int, default=1000, help="group_by_sum key count (default 1000)")
    x.add_argument("--json", action="store_true", help="print the decision as JSON")
    a = p.parse_args(argv)

    import arrowmetal as am
    if a.cmd == "calibrate":
        calibrate(quick=a.quick, out=a.out, csv_out=a.csv, log=lambda s: print(s, flush=True), progress=not a.quiet)
        return 0
    if a.nulls is not None and not 0 <= a.nulls <= 1:
        p.error("--nulls is a fraction between 0 and 1")
    if a.rows < 0:
        p.error("rows must be at least 0")
    try:
        e = am.explain_route(a.op, a.rows, a.dtype, a.keys)
    except ValueError as err:
        p.error(str(err))
    if a.json:
        if a.nulls is not None:
            e["null_fraction"] = a.nulls
        print(json.dumps(e, indent=1, sort_keys=True))
    else:
        print("\n".join(explain_text(e, a.nulls)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
