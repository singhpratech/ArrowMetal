"""CPU against Metal on this Mac, in 30 seconds or less: `python -m arrowmetal.bench`.

One seeded dataset (10,000,000 rows by default), four operations everyone knows (sum, filter, sort,
group-by sum), each measured the way `Benchmarks/full_matrix.py` measures the published matrix: one
warm-up, the best of up to five calls, wall milliseconds with the process CPU milliseconds of that
same call beside them. pyarrow is always measured; Polars is measured when it is installed. Every
ArrowMetal answer is checked against pyarrow's before anything is printed.

Nothing is written and nothing is sent: the script prints, and the "Share it" block at the end is
text to paste into a GitHub issue or the Discord channel if you want to.

    python -m arrowmetal.bench                # 10,000,000 rows
    python -m arrowmetal.bench --rows 2000000
    python -m arrowmetal.bench --json         # one JSON object instead of the text report
    python -m arrowmetal.bench --quiet        # the table only
    python -m arrowmetal.bench --no-share     # no "Share it" block
    python -m arrowmetal.bench --no-polars    # skip Polars even when it is installed
"""
import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
import urllib.parse

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import arrowmetal as am

SEED = 20260907
KEYS = 1_000
NULL_FRACTION = 0.10
ISSUE_URL = "https://github.com/singhpratech/ArrowMetal/issues/new?template=benchmark_result.yml&title="


# ---- timing, as Benchmarks/full_matrix.py does it

def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


class Bench:
    """One warm-up, then the best of up to `iters` calls (never fewer than two) within `budget` s."""

    def __init__(self, iters=5, budget=1.2):
        self.iters, self.budget = iters, budget

    def run(self, fn):
        """(wall_ms, cpu_ms, iterations) of the best call; cpu_ms is the process CPU time (all
        threads) of that same call, so a threaded library shows the cores it used."""
        fn()
        best_w, best_c, total, n = float("inf"), float("inf"), 0.0, 0
        while n < self.iters and (n < 2 or total < self.budget):
            c0 = cpu_seconds()
            t0 = time.perf_counter()
            fn()
            w = time.perf_counter() - t0
            c = cpu_seconds() - c0
            if w < best_w:
                best_w, best_c = w, c
            total += w
            n += 1
        return best_w * 1000.0, best_c * 1000.0, n


# ---- the machine

def _sysctl(name):
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def machine_info():
    mem = _sysctl("hw.memsize")
    return {
        "chip": _sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown CPU",
        "cores": os.cpu_count() or 0,
        "memory_gb": round(int(mem) / 2**30) if mem.isdigit() else None,
        "macos": platform.mac_ver()[0] or platform.release(),
        "gpu": am.device_name(),
    }


def machine_line(m, versions):
    mem = f"{m['memory_gb']} GB" if m["memory_gb"] else "memory unknown"
    libs = f"ArrowMetal {versions['arrowmetal']}, pyarrow {versions['pyarrow']}"
    if versions.get("polars"):
        libs += f", Polars {versions['polars']}"
    return f"{m['chip']}, {m['cores']} cores, {mem}, macOS {m['macos']}, GPU {m['gpu']}; {libs}"


# ---- the data

def make_data(rows, seed=SEED):
    """int64 `v` with ~10% nulls, float64 `f`, int32 `k` with 1,000 distinct keys."""
    rng = np.random.default_rng(seed)
    v = rng.integers(-1_000_000, 1_000_000, size=rows, dtype=np.int64)
    mask = rng.random(rows) < NULL_FRACTION
    f = rng.standard_normal(rows)
    k = rng.integers(0, KEYS, size=rows, dtype=np.int32)
    return {
        "v": pa.array(v, mask=mask, type=pa.int64()),
        "f": pa.array(f, type=pa.float64()),
        "k": pa.array(k, type=pa.int32()),
    }


def _polars(enabled):
    if not enabled:
        return None
    try:
        import polars as pl
    except Exception:
        return None
    return pl


# ---- the four operations

OPS = ["sum int64 (10% nulls)", "filter int64 > 0", "sort float64", "sum by int32 key (1000 groups)"]


def _group_dict(keys, sums):
    return dict(zip(keys.to_pylist(), sums.to_pylist()))


def run(rows, use_polars=True, bench=None):
    """Measure everything; returns a dict with the machine, the versions, the timings, the import
    time and the match flag. Raises nothing on a mismatch: `result["match"]` says."""
    bench = bench or Bench()
    t_all = time.perf_counter()
    pl = _polars(use_polars)

    t0 = time.perf_counter()
    d = make_data(rows)
    gen_s = time.perf_counter() - t0
    v, f, k = d["v"], d["f"], d["k"]
    tbl = pa.table({"k": k, "v": v})

    if pl is not None:
        pv, pf = pl.from_arrow(v), pl.from_arrow(f)
        pdf = pl.from_arrow(tbl)

    # The Metal columns are imported once, before any warm-up, and stay resident: the operations
    # are timed on resident columns, as the published matrix times them, and the import is its own
    # line in the report.
    t0 = time.perf_counter()
    gv, gf, gk = am.MetalArray.from_arrow(v), am.MetalArray.from_arrow(f), am.MetalArray.from_arrow(k)
    import_ms = (time.perf_counter() - t0) * 1000.0

    cases = {
        OPS[0]: {
            "pyarrow": lambda: pc.sum(v),
            "polars": (lambda: pv.sum()) if pl is not None else None,
            "arrowmetal": lambda: gv.sum(),
        },
        OPS[1]: {
            "pyarrow": lambda: pc.filter(v, pc.greater(v, 0)),
            "polars": (lambda: pv.filter(pv > 0)) if pl is not None else None,
            "arrowmetal": lambda: gv.filter_where(">", 0),
        },
        OPS[2]: {
            "pyarrow": lambda: pc.take(f, pc.sort_indices(f)),
            "polars": (lambda: pf.sort()) if pl is not None else None,
            "arrowmetal": lambda: gf.sort(),
        },
        OPS[3]: {
            "pyarrow": lambda: tbl.group_by("k").aggregate([("v", "sum")]),
            "polars": (lambda: pdf.group_by("k").agg(pl.sum("v"))) if pl is not None else None,
            "arrowmetal": lambda: am.group_by([gk]).sum(gv),
        },
    }

    timings = {}
    for op, impls in cases.items():
        row = {}
        for lib, fn in impls.items():
            if fn is None:
                continue
            w, c, n = bench.run(fn)
            row[lib] = {"wall_ms": round(w, 3), "cpu_ms": round(c, 3), "iterations": n}
        cpu_best = min(row[lib]["wall_ms"] for lib in ("pyarrow", "polars") if lib in row)
        row["fastest_cpu"] = min((lib for lib in ("pyarrow", "polars") if lib in row),
                                 key=lambda lib: row[lib]["wall_ms"])
        row["speedup"] = round(cpu_best / row["arrowmetal"]["wall_ms"], 2) if row["arrowmetal"]["wall_ms"] > 0 else None
        timings[op] = row

    # Every ArrowMetal answer against pyarrow's, computed once more outside the timing.
    problems = []
    if gv.sum() != pc.sum(v).as_py():
        problems.append(f"{OPS[0]}: ArrowMetal {gv.sum()} vs pyarrow {pc.sum(v).as_py()}")
    got, want = gv.filter_where(">", 0).to_arrow(), pc.filter(v, pc.greater(v, 0))
    if len(got) != len(want):
        problems.append(f"{OPS[1]}: ArrowMetal keeps {len(got)} rows vs pyarrow {len(want)}")
    elif not got.equals(want):
        problems.append(f"{OPS[1]}: same row count, different rows")
    got, want = gf.sort().to_arrow(), pc.take(f, pc.sort_indices(f))
    if not got.equals(want):
        problems.append(f"{OPS[2]}: sorted values differ from pyarrow's sort_indices order")
    gb = am.group_by([gk])
    got = _group_dict(gb.keys()[0], gb.sum(gv).to_arrow())
    pt = tbl.group_by("k").aggregate([("v", "sum")])
    want = _group_dict(pt.column("k"), pt.column("v_sum"))
    if got != want:
        bad = [key for key in want if got.get(key) != want[key]]
        problems.append(f"{OPS[3]}: {len(bad)} of {len(want)} group sums differ, first key {bad[:1]}")

    versions = {"arrowmetal": am.__version__, "pyarrow": pa.__version__,
                "polars": pl.__version__ if pl is not None else None,
                "python": platform.python_version()}
    return {
        "machine": machine_info(),
        "versions": versions,
        "rows": rows,
        "keys": KEYS,
        "import_ms": round(import_ms, 3),
        "generate_s": round(gen_s, 3),
        "timings": timings,
        "match": not problems,
        "problems": problems,
        "protocol": "one warm-up, best of up to 5 calls, wall ms with the process CPU ms of that call; "
                    "ArrowMetal columns resident on the GPU, import timed separately",
        "total_s": round(time.perf_counter() - t_all, 3),
    }


# ---- the report

def _cell(t):
    return f"{t['wall_ms']:.2f} ({t['cpu_ms']:.1f})" if t else "-"


def _speedup(x):
    return f"{x:.2f}x" if x is not None else "-"


def table_rows(result):
    have_polars = result["versions"].get("polars") is not None
    head = ["op", "rows", "pyarrow ms (cpu-ms)"]
    if have_polars:
        head.append("Polars ms (cpu-ms)")
    head += ["ArrowMetal ms (cpu-ms)", "speedup"]
    body = []
    for op, t in result["timings"].items():
        cells = [op, f"{result['rows']:,}", _cell(t.get("pyarrow"))]
        if have_polars:
            cells.append(_cell(t.get("polars")))
        cells += [_cell(t["arrowmetal"]), _speedup(t["speedup"])]
        body.append(cells)
    return head, body


def text_table(result):
    head, body = table_rows(result)
    widths = [max(len(r[i]) for r in [head] + body) for i in range(len(head))]
    lines = []
    for r in [head] + body:
        cells = [r[0].ljust(widths[0])] + [r[i].rjust(widths[i]) for i in range(1, len(r))]
        lines.append("  ".join(cells))
    return "\n".join(lines)


def markdown_table(result):
    head, body = table_rows(result)
    lines = ["| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def issue_url(result):
    title = f"bench: {result['machine']['chip']}, {result['rows']:,} rows"
    return ISSUE_URL + urllib.parse.quote(title, safe="")


def report(result, quiet=False, share=True):
    m, versions = result["machine"], result["versions"]
    line = machine_line(m, versions)
    out = []
    if not quiet:
        out.append(line)
        out.append("")
    out.append(text_table(result))
    if quiet:
        return "\n".join(out)
    out.append("")
    out.append("results match pyarrow" if result["match"] else "MISMATCH against pyarrow")
    out.append("")
    polars_note = ", Polars its thread pool" if versions.get("polars") else ""
    out.append("Protocol: one warm-up, best of up to 5 calls, wall ms with the process CPU ms of that call in "
               f"parentheses; pyarrow uses its thread pool{polars_note}; speedup is against the fastest CPU number on the row.")
    out.append(f"ArrowMetal columns resident on the GPU; import cost shown separately: importing the three columns "
               f"once took {result['import_ms']:.1f} ms. Data generation {result['generate_s']:.1f} s, "
               f"whole run {result['total_s']:.1f} s. Router pinned to the GPU for the run.")
    if share:
        out.append("")
        out.append("Share it (nothing is sent by this script; copy the block below):")
        out.append("")
        out.append("```")
        out.append(line)
        out.append("")
        out.append(markdown_table(result))
        out.append("```")
        out.append("")
        out.append(f"Open a prefilled issue: {issue_url(result)}")
        out.append("or drop it in the #benchmarks channel of the ArrowMetal Discord (link in the README).")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m arrowmetal.bench",
                                description="CPU (pyarrow, Polars if installed) against ArrowMetal on this Mac.")
    p.add_argument("--rows", type=int, default=10_000_000, help="rows in the generated dataset (default 10,000,000)")
    p.add_argument("--json", action="store_true", help="print one JSON object instead of the text report")
    p.add_argument("--quiet", action="store_true", help="print the table only")
    p.add_argument("--no-share", action="store_true", help="omit the Share it block")
    p.add_argument("--no-polars", action="store_true", help="do not measure Polars even if it is installed")
    a = p.parse_args(argv)
    if a.rows < 1:
        p.error("--rows must be at least 1")

    # The suites pin the router the same way (docs/TESTING.md): this is a CPU-against-GPU report, so
    # the ArrowMetal column is the GPU kernel at every row count. ARROWMETAL_ROUTER, when set, wins.
    if "ARROWMETAL_ROUTER" not in os.environ:
        am.set_router("gpu")

    result = run(a.rows, use_polars=not a.no_polars)
    if a.json:
        print(json.dumps(result, indent=2))
    else:
        print(report(result, quiet=a.quiet, share=not a.no_share))
    if not result["match"]:
        for line in result["problems"]:
            print("MISMATCH: " + line, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
