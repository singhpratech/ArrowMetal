#!/usr/bin/env python3
"""The crossover table: at what row count the GPU path overtakes the CPU, per operation and per family.

Two sources, both produced on this machine with the matrix protocol (best of up to five after a
warm-up, 1.2 s budget, at least two):

1. A size sweep of the full matrix, `Benchmarks/full_matrix.py --sizes 1000,...,10000000 --families
   ...`, which times ArrowMetal and every CPU library idiom on the same data at each size. The 50M
   points come from the matrix's own parallel run (`--matrix`), which used the same generators.
   From this the script derives, per operation, the first row count from which ArrowMetal stays at
   or ahead of the fastest CPU idiom at every larger measured size ("vs fastest library").

2. `arrowmetal-bench crossover` (Sources/ArrowMetalBench), which times each routable operation's
   GPU path against the CPU path the router would run instead (`CPUReference`, single core) at the
   same sizes ("vs own CPU path"). This is the router's calibration data; the first is the number a
   caller wants when deciding whether to hand a column to Metal at all.

Outputs, all generated, no hand-typed numbers:
    Benchmarks/results/crossover_<date>.csv   one row per (family, op, rows): ArrowMetal, best CPU, ratio
    Benchmarks/results/router_<date>.json     the two crossover tables keyed by operation
    docs/CROSSOVER.md                         the page

Usage:
    python Benchmarks/crossover.py --sweep Benchmarks/results/full_matrix_2026-09-17.csv \
        --matrix Benchmarks/results/full_matrix_2026-09-07-parallel.csv \
        --bench Benchmarks/results/crossover_bench_2026-09-17.csv
"""
import argparse
import collections
import csv
import datetime
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 50_000_000]


def load_matrix(path, sizes=None):
    """{(family, op, rows): {library: (wall_ms, cpu_ms)}} for status ok rows."""
    out = collections.defaultdict(dict)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["status"] != "ok":
                continue
            n = int(r["rows"])
            if sizes is not None and n not in sizes:
                continue
            out[(r["family"], r["op"], n)][r["library"]] = (float(r["wall_ms"]), float(r["cpu_ms"]))
    return out


def load_bench(path):
    """{(op, rows): {path: (wall_us, cpu_us)}} from `arrowmetal-bench crossover`; returns header too."""
    out, header = collections.defaultdict(dict), ""
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                header = line[1:].strip()
                continue
            if line.startswith("op,"):
                continue
            op, rows, p, wall, cpu, _ = line.rstrip("\n").rsplit(",", 5)  # op names may hold commas
            out[(op, int(rows))][p] = (float(wall), float(cpu))
    return out, header


def crossover(points):
    """points: [(rows, ratio)] sorted by rows; ratio = other / arrowmetal (>= 1 means ArrowMetal ahead).
    The smallest rows from which every measured point is >= 1. None when the last point is behind."""
    best = None
    for rows, ratio in reversed(points):
        if ratio >= 1.0:
            best = rows
        else:
            break
    return best


#: The CPU path the router would run, in order of preference: the tight single-core loop, the
#: candidate loop for an operation with no CPU implementation yet, and only then the tests' oracle.
OWN_CPU_PATHS = ("cpu-1core", "cpu-candidate", "cpu-ref")


def own_cpu_path(paths):
    for p in OWN_CPU_PATHS:
        if p in paths:
            return paths[p]
    return None


def fmt_rows(n):
    if n is None:
        return "not reached"
    return f"{n:,}"


def router_table_section():
    """The router's table section: how the shipped table is made and how to make one for another Mac.
    The source file and its machine line are read from the generated RouterTable.swift."""
    import re
    with open(os.path.join(ROOT, "Sources", "ArrowMetal", "Router", "RouterTable.swift")) as fh:
        src = fh.read()
    source = re.search(r'static let source = "([^"]+)"', src).group(1)
    header = re.search(r'static let header = "([^"]+)"', src).group(1)
    L = ["## The router's table: how it is made, and making one for your Mac\n"]
    L.append(f"The shipped table (`Sources/ArrowMetal/Router/RouterTable.swift`) is fitted from `{source}`: "
             f"{header}. The pipeline:\n")
    L.append("1. Sweep: `PYTHONPATH=python python Benchmarks/router_check.py --out "
             "Benchmarks/results/router_check_<date>.csv` times every routed operation with the router pinned "
             "to `gpu` and to `cpu`, and under `auto`, on int64 columns with 10% nulls resident in Metal memory "
             "(12 cases, 1,000 to 10,000,000 rows).")
    L.append("2. The CSV lands in `Benchmarks/results/`.")
    L.append("3. Fit: `python Benchmarks/router_table.py --from-check Benchmarks/results/router_check_<date>.csv` "
             "takes, per operation, the first measured size from which the GPU stays ahead and the size below it, "
             "joins the two measured points of each path with a straight line, and puts the crossover where the "
             "lines meet. It writes `RouterTable.swift`; `--json-out PATH` also writes the same table as JSON "
             "(format `arrowmetal-router-table/1`), and `--check` fails when the committed table no longer "
             "matches the file it names.")
    L.append("4. `RouterTable.swift` is a Swift literal compiled into the library.\n")
    L.append("On any Mac, one command runs the sweep and the fit for that machine:\n")
    L.append("```")
    L.append("python -m arrowmetal.router calibrate          # the router check grid: 12 cases x 6 sizes, gpu, cpu and auto")
    L.append("python -m arrowmetal.router calibrate --quick  # the 8 cases the table is fitted from, 30,000 to 10,000,000 rows")
    L.append("python -m arrowmetal.bench --calibrate         # the benchmark, then the quick calibration")
    L.append("```\n")
    L.append("It writes `~/.arrowmetal/router/<chip>.json`, where `<chip>` is `sysctl machdep.cpu.brand_string` in "
             "lower case with other characters as `-` (`apple-m4-max`). The file records the chip, the CPU core "
             "counts, the Metal device, memory, macOS, the date, the ArrowMetal version, the grid that ran and "
             "every measurement, and the output names the grid. `--out PATH` writes elsewhere; `--csv PATH` also "
             "writes the sweep as a router check CSV for `router_table.py --from-check`.\n")
    L.append("The table in force is chosen once, at load: `ARROWMETAL_ROUTER_TABLE=<path>` (or `shipped`), else "
             "`~/.arrowmetal/router/<chip>.json` when it exists, else the shipped table. An operation the sweep "
             "did not bring to a crossover keeps its shipped row, and a file that does not parse leaves the "
             "shipped table in force. `am_router_load_table` (C), `am.load_router_table` (Python) and "
             "`Router.loadTable(path:)` (Swift) replace it in a running process; `am.router_table()` and "
             "`am_router_table_info` describe it.\n")
    L.append("```")
    L.append("python -m arrowmetal.router explain sum 150000 [--dtype int64] [--nulls 0.1] [--keys 1000] [--json]")
    L.append("```\n")
    L.append("prints the decision under the mode and table in force (`mode auto -> cpu`), the reason, the table "
             "row with its crossover and the two measured points it was fitted between, and whether the table is "
             "the shipped one or this machine's.\n")
    L.append("A decision depends only on the operation, the value type, the row count, the mode, whether a batch "
             "is open and the table; nothing is timed at call time. The table has one shape, int64 with 10% nulls "
             "on resident arrays, so the null fraction and residency do not change a decision. "
             "`python/tests/test_router_calibrate.py` checks that 200 (operation, size) pairs get the same "
             "decision across 1,000 calls and across two processes, and `RouterTests.testRouteIsPure` checks "
             "the Swift rule the same way.\n")
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", required=True, help="full_matrix.py CSV from the size sweep")
    ap.add_argument("--matrix", default=None, help="the matrix's parallel CSV, for the 50M rows")
    ap.add_argument("--bench", default=None, help="arrowmetal-bench crossover CSV")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--no-doc", action="store_true", help="write the CSV and JSON only")
    args = ap.parse_args()

    data = load_matrix(args.sweep)
    if args.matrix:
        for k, v in load_matrix(args.matrix, sizes={50_000_000}).items():
            data.setdefault(k, v)

    # Per operation: the points, the best CPU idiom at each size, the crossover.
    ops = collections.defaultdict(list)  # (family, op) -> [(rows, am_wall, am_cpu, best_wall, best_lib, ratio)]
    for (family, op, n), libs in data.items():
        if "arrowmetal" not in libs:
            continue
        others = {lib: w for lib, (w, _) in libs.items() if lib != "arrowmetal"}
        if not others:
            continue
        best_lib = min(others, key=others.get)
        am_w, am_c = libs["arrowmetal"]
        ops[(family, op)].append((n, am_w, am_c, others[best_lib], best_lib, others[best_lib] / am_w if am_w > 0 else float("inf")))
    for k in ops:
        ops[k].sort()

    # An operation needs the sweep behind it (at least three sizes) before a crossover means anything;
    # the families the sweep did not cover have only the matrix's 50M point and are listed as such.
    rows_out = []
    cross_lib = {}
    swept = {family for (family, op), pts in ops.items() if len(pts) >= 3}
    not_swept = sorted({family for (family, op) in ops} - swept)
    ops = {k: v for k, v in ops.items() if len(v) >= 3}
    for (family, op), pts in sorted(ops.items()):
        c = crossover([(n, r) for n, _, _, _, _, r in pts])
        cross_lib[(family, op)] = c
        for n, am_w, am_c, bw, bl, r in pts:
            rows_out.append([family, op, n, f"{am_w:.4f}", f"{am_c:.4f}", f"{bw:.4f}", bl, f"{r:.3f}", fmt_rows(c)])

    results_dir = os.path.join(ROOT, "Benchmarks", "results")
    csv_path = os.path.join(results_dir, f"crossover_{args.date}.csv")
    with open(csv_path, "w", newline="") as fh:
        fh.write(f"# crossover from {os.path.basename(args.sweep)}"
                 + (f" + {os.path.basename(args.matrix)} (50M rows)" if args.matrix else "")
                 + (f" + {os.path.basename(args.bench)}" if args.bench else "") + f", {args.date}\n")
        w = csv.writer(fh)
        w.writerow(["family", "op", "rows", "arrowmetal_wall_ms", "arrowmetal_cpu_ms", "best_cpu_wall_ms",
                    "best_cpu_library", "ratio", "crossover_rows"])
        w.writerows(rows_out)

    # The bench: GPU path vs the CPU path the router would run.
    bench, bench_header, cross_own = {}, "", {}
    if args.bench:
        bench, bench_header = load_bench(args.bench)
        by_op = collections.defaultdict(list)
        for (op, n), paths in bench.items():
            if "gpu" not in paths:
                continue
            ref = own_cpu_path(paths)
            if ref is None:
                continue
            by_op[op].append((n, ref[0] / paths["gpu"][0]))
        for op, pts in by_op.items():
            pts.sort()
            cross_own[op] = crossover(pts)

    # Family summary.
    fam = collections.defaultdict(list)
    for (family, op), c in cross_lib.items():
        fam[family].append(c)
    family_rows = []
    for family in sorted(fam):
        cs = fam[family]
        reached = [c for c in cs if c is not None]
        family_rows.append((family, len(cs), len(reached),
                            fmt_rows(min(reached)) if reached else "not reached",
                            fmt_rows(int(statistics.median(reached))) if reached else "not reached",
                            fmt_rows(max(reached)) if reached else "not reached"))

    json_path = os.path.join(results_dir, f"router_{args.date}.json")
    with open(json_path, "w") as fh:
        json.dump({
            "generated": args.date,
            "sources": {"sweep": os.path.basename(args.sweep), "matrix": os.path.basename(args.matrix) if args.matrix else None,
                        "bench": os.path.basename(args.bench) if args.bench else None, "bench_header": bench_header},
            "sizes": SIZES,
            "not_swept": not_swept,
            "vs_fastest_library": {f"{family}: {op}": c for (family, op), c in sorted(cross_lib.items())},
            "vs_own_cpu_path": dict(sorted(cross_own.items())),
        }, fh, indent=1)

    print(f"wrote {csv_path} ({len(rows_out)} rows) and {json_path}")
    if args.no_doc:
        return

    L = []
    L.append("# The crossover: where the GPU path overtakes the CPU\n")
    L.append("Generated by `Benchmarks/crossover.py` from the files named at the top of "
             f"`Benchmarks/results/crossover_{args.date}.csv`; nothing on this page is typed by hand. "
             "Apple M4 Max, the matrix protocol (best of up to five after a warm-up, 1.2 s budget, at least two), "
             "10% nulls on the numeric columns.\n")
    L.append("Two questions, two tables. The first is the one a caller asks: from what row count is "
             "ArrowMetal at or ahead of the fastest CPU idiom the matrix knows (Polars eager and lazy, "
             "pyarrow and Acero on 16 batches, pandas, numpy), at every larger size measured? The second is "
             "the one the engine's router asks: from what row count is the GPU kernel faster than a "
             "single-core Swift loop? That loop is the bench's own (`cpu-1core`, `cpu-candidate` for the "
             "group-by, in `Sources/ArrowMetalBench/main.swift`), not the router's shipped CPU loops in "
             "`Sources/ArrowMetal/Router/RouterCPU.swift`, which `Benchmarks/router_check.py` times. "
             "A row \"not reached\" means ArrowMetal was still behind at the largest size measured.\n")
    L.append("## Per family, against the fastest CPU idiom\n")
    L.append("| family | operations | reach the crossover | earliest | median | latest |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for f, n, r, lo, med, hi in family_rows:
        L.append(f"| {f} | {n} | {r} | {lo} | {med} | {hi} |")
    L.append("")
    if not_swept:
        L.append("Not swept, so no crossover is stated: " + ", ".join(f"`{f}`" for f in not_swept)
                 + ". The matrix measures them at 10M and 50M rows only ([BENCHMARKS_MATRIX.md](BENCHMARKS_MATRIX.md)).\n")
    if cross_own:
        L.append("## The router's own crossover: GPU kernel against the CPU path it would run\n")
        L.append(f"`arrowmetal-bench crossover`: {bench_header}\n")
        L.append("| operation | crossover (rows) | " + " | ".join(f"{n:,}" for n in SIZES) + " |")
        L.append("|---|---:|" + "---:|" * len(SIZES))
        for op in sorted(cross_own):
            cells = []
            for n in SIZES:
                paths = bench.get((op, n))
                if not paths or "gpu" not in paths:
                    cells.append("")
                    continue
                cpu = own_cpu_path(paths)
                cells.append(f"{paths['gpu'][0]:.0f} / {cpu[0]:.0f} µs" if cpu else f"{paths['gpu'][0]:.0f} µs")
            L.append(f"| {op} | {fmt_rows(cross_own[op])} | " + " | ".join(cells) + " |")
        L.append("")
        L.append("Each cell is GPU / CPU-path wall microseconds, best of the run's repetitions; the CPU path "
                 "is the tight single-core loop (`cpu-1core`), or for group-by the candidate dictionary loop "
                 "(`cpu-candidate`). The bench CSV also carries `cpu-ref`, the tests' `CPUReference` oracle, "
                 "which walks a closure per element and is one to two orders slower than the tight loop: it "
                 "is the reason the router's CPU side is written as new loops rather than reused from the "
                 "oracle. `cpu-allcores`, where present, is the bench's all-core loop, recorded for "
                 "reference.\n")
    L.extend(router_table_section())
    L.append("## Per operation, against the fastest CPU idiom\n")
    L.append("| family | operation | crossover (rows) | " + " | ".join(f"{n:,}" for n in SIZES) + " |")
    L.append("|---|---|---:|" + "---:|" * len(SIZES))
    for (family, op), pts in sorted(ops.items()):
        by_n = {n: (am_w, bw, bl, r) for n, am_w, _, bw, bl, r in pts}
        cells = []
        for n in SIZES:
            if n in by_n:
                am_w, bw, bl, r = by_n[n]
                cells.append(f"{r:.2f}x")
            else:
                cells.append("")
        L.append(f"| {family} | {op} | {fmt_rows(cross_lib[(family, op)])} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("A cell is fastest-CPU wall time divided by ArrowMetal wall time at that size: above 1.00x "
             "ArrowMetal is ahead. The CSV carries the times, the library that was fastest, and the CPU "
             "milliseconds of each.\n")
    doc_path = os.path.join(ROOT, "docs", "CROSSOVER.md")
    with open(doc_path, "w") as fh:
        fh.write("\n".join(L))
    print(f"wrote {doc_path}")


if __name__ == "__main__":
    main()
