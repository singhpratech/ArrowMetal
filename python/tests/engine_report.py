#!/usr/bin/env python3
"""Run the engine conformance grid (engine_conformance.py) and print it. Exit code 1 on any
*unclassified* mismatch -- one that no documented divergence explains.

    PYTHONPATH=python python python/tests/engine_report.py                  # both engines, full grid
    ... --quick                        # without the 100,000-row tables
    ... --engine polars                # or duckdb
    ... --csv-dir Benchmarks/results   # also write engine_conformance_<date>.csv and _shapes.csv
    ... -q                             # no progress on stderr

The summary lines read, per engine:

    engine polars: total N  pass P  documented D  unclassified U  not taken T

`not taken` counts cases the engine did not run itself (Polars folded the plan away, the translation
declined it, DuckDB kept its own aggregate); they were compared too, and a mismatch among them would
be counted as unclassified. The DuckDB half needs duckdb-extension/build/arrowmetal_rewrite.duckdb_extension
(or $ARROWMETAL_DUCKDB_REWRITE_EXTENSION); without it that engine is reported as skipped.
"""
import argparse
import csv
import datetime
import os
import sys
import time
from collections import OrderedDict, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import engine_conformance as ec                                      # noqa: E402

PASS, DOCUMENTED, UNCLASSIFIED, NOT_TAKEN = ec.PASS, ec.DOCUMENTED, ec.UNCLASSIFIED, ec.NOT_TAKEN


class Tally:
    __slots__ = ("cases", "passed", "documented", "unclassified", "not_taken", "family")

    def __init__(self, family=""):
        self.cases = self.passed = self.documented = self.unclassified = self.not_taken = 0
        self.family = family

    def add(self, status):
        self.cases += 1
        if status == PASS:
            self.passed += 1
        elif status == DOCUMENTED:
            self.documented += 1
        elif status == NOT_TAKEN:
            self.not_taken += 1
        else:
            self.unclassified += 1


def _engines(which):
    out = []
    if which in ("polars", "both"):
        import engine_polars_grid as pg
        out.append(("polars", pg))
    if which in ("duckdb", "both"):
        import engine_duckdb_grid as dg
        out.append(("duckdb", dg))
    return out


def run_engine(name, mod, quick=False, quiet=False, dtypes=None):
    """Runs one engine's grid. Returns (per-shape tallies, total tally, documented hits, the
    unclassified cases, not-taken reasons, seconds), or None when the engine cannot run here.
    `dtypes` restricts the Polars grid to those column types."""
    started = time.time()
    shapes = OrderedDict()
    total = Tally()
    documented = defaultdict(list)
    unclassified = []
    not_taken = defaultdict(int)
    tables = None
    if name == "duckdb":
        from arrowmetal import duckdb_bridge
        if not os.path.exists(duckdb_bridge.rewrite_extension_path()):
            return None
        tables = mod.Tables()
    n = 0
    try:
        for case in (mod.cases(quick=quick, dtypes=dtypes) if dtypes else mod.cases(quick=quick)):
            t0 = time.time()
            if tables is not None:
                status, detail, extra = mod.run_case(case, tables)
            else:
                status, detail, extra = mod.run_case(case)
            if not quiet and time.time() - t0 > 10:
                print(f"  ... slow case ({time.time() - t0:.0f}s): {case['shape']} / {case['dtype']} / "
                      f"{ec.shape_id(case['ds'])}", file=sys.stderr)
            if status == "invalid":            # the host itself rejects the combination
                continue
            if status == "mismatch":
                d = ec.classify(mod.DIVERGENCES, case, dict(extra, detail=detail))
                if d is None:
                    status = UNCLASSIFIED
                    unclassified.append((case, detail))
                else:
                    status = DOCUMENTED
                    documented[d.id].append((case, detail, extra))
            elif status == NOT_TAKEN:
                not_taken[detail.split(": ", 1)[-1][:120] if name == "polars" else detail[:120]] += 1
            key = case["shape"]
            if key not in shapes:
                shapes[key] = Tally(case["family"])
            shapes[key].add(status)
            total.add(status)
            n += 1
            if not quiet and n % 1000 == 0:
                print(f"  ... {name}: {n} cases, {time.time() - started:.0f}s", file=sys.stderr)
    finally:
        if tables is not None:
            tables.close()
    return shapes, total, documented, unclassified, not_taken, time.time() - started


def summary_line(name, total):
    return (f"engine {name}: total {total.cases}  pass {total.passed}  documented {total.documented}  "
            f"unclassified {total.unclassified}  not taken {total.not_taken}")


def render(results, quick):
    import arrowmetal as am
    lines = []
    add = lines.append
    add("ArrowMetal engines vs their hosts -- engine conformance grid")
    add(f"arrowmetal {am.__version__} on {am.device_name()} | sizes "
        f"{', '.join(str(s) for s in (ec.SIZES_QUICK if quick else ec.SIZES_FULL))} | null patterns "
        f"{', '.join(f'{k} {v:g}' for k, v in ec.NULL_PATTERNS.items())} | flavours random, special")
    add("")
    for name, res in results.items():
        if res is None:
            add(f"engine {name}: skipped (the DuckDB rewrite extension is not built)")
            add("")
            continue
        shapes, total, documented, unclassified, not_taken, secs = res
        mod = dict(_engines(name))[name]
        add(f"{name}: {mod.HOST_DESCRIPTION}")
        width = max(len(s) for s in shapes) if shapes else 10
        add(f"{'shape'.ljust(width)} | {'cases':>6} | {'pass':>6} | {'doc':>5} | {'uncl':>5} | {'not taken':>9}")
        add("-" * width + "-+--------+--------+-------+-------+----------")
        for s, t in shapes.items():
            add(f"{s.ljust(width)} | {t.cases:>6} | {t.passed:>6} | {t.documented:>5} | "
                f"{t.unclassified:>5} | {t.not_taken:>9}")
        add("")
        if documented:
            add(f"{name}: documented divergences")
            for d in mod.DIVERGENCES:
                hits = documented.get(d.id)
                if not hits:
                    continue
                case, detail, _extra = hits[0]
                add(f"  [{d.id}] {d.title} -- {d.doc_line()}")
                # Float aggregates: the largest difference by dtype and aggregate, relative to the
                # answer and in units of u * sum(|x|) (see engine_polars_grid._summation_order).
                worst = defaultdict(lambda: [0.0, 0.0, 0])
                for c, _d, x in hits:
                    if x.get("max_rel") is None:
                        continue
                    w = worst[(c["dtype"], c["shape"].rsplit("_", 1)[-1])]
                    w[0] = max(w[0], x["max_rel"])
                    w[1] = max(w[1], x.get("ulps_of_magnitude") or 0.0)
                    w[2] += 1
                for (t, agg), (rel, ulps, count) in sorted(worst.items()):
                    add(f"    {t} {agg}: {count} case(s), largest difference {rel:.2e} of the answer, "
                        f"{ulps:.3g} u*sum(|x|)")
                add(f"    {len(hits)} case(s); e.g. {case['shape']} / {case['dtype']} / "
                    f"{ec.shape_id(case['ds'])}: {detail[:200]}")
            add("")
        if unclassified:
            add(f"{name}: UNCLASSIFIED mismatches ({len(unclassified)})")
            for case, detail in unclassified[:40]:
                add(f"  {case['shape']} / {case['dtype']} / {ec.shape_id(case['ds'])}")
                add(f"    {mod.reproducer(case)}")
                add(f"    {detail[:300]}")
            add("")
        if not_taken:
            add(f"{name}: not taken, by reason (compared, and equal)")
            for reason, count in sorted(not_taken.items(), key=lambda kv: -kv[1]):
                add(f"  {count:>6}  {reason}")
            add("")
        add(f"{name}: {secs:.1f}s")
        add("")
    for name, res in results.items():
        if res is not None:
            add(summary_line(name, res[1]))
    bad = sum(res[1].unclassified for res in results.values() if res is not None)
    add("FAIL" if bad else "PASS")
    return "\n".join(lines)


def write_csv(results, directory, stamp):
    os.makedirs(directory, exist_ok=True)
    summary = os.path.join(directory, f"engine_conformance_{stamp}.csv")
    per_shape = os.path.join(directory, f"engine_conformance_{stamp}_shapes.csv")
    fields = ["cases", "pass", "documented", "unclassified", "not_taken"]
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["engine"] + fields)
        for name, res in results.items():
            if res is not None:
                t = res[1]
                w.writerow([name, t.cases, t.passed, t.documented, t.unclassified, t.not_taken])
    with open(per_shape, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["engine", "shape", "family"] + fields)
        for name, res in results.items():
            if res is None:
                continue
            for s, t in res[0].items():
                w.writerow([name, s, t.family, t.cases, t.passed, t.documented, t.unclassified,
                            t.not_taken])
    return summary, per_shape


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", choices=["polars", "duckdb", "both"], default="both")
    parser.add_argument("--quick", action="store_true", help="drop the 100,000-row tables")
    parser.add_argument("--csv-dir", help="write the summary and per-shape CSVs here")
    parser.add_argument("--stamp", default=datetime.date.today().isoformat(),
                        help="the date in the CSV file names")
    parser.add_argument("-q", "--quiet", action="store_true", help="no progress on stderr")
    parser.add_argument("--dtypes", help="Polars grid only: comma-separated column types (e.g. string)")
    parser.add_argument("--string-layout", choices=["view", "offsets"], default="view",
                        help="how the Polars engine hands String columns over (polars_engine.STRING_LAYOUT)")
    args = parser.parse_args(argv)
    if args.dtypes and args.engine != "polars":
        parser.error("--dtypes needs --engine polars")

    import arrowmetal as am
    from arrowmetal import polars_engine
    # The grid's tables are small, so the router's `auto` would send the routed operations to their
    # CPU loops; pin the GPU as differential_report.py does. ARROWMETAL_ROUTER, when set, wins.
    if "ARROWMETAL_ROUTER" not in os.environ:
        am.set_router("gpu")
    polars_engine.STRING_LAYOUT = args.string_layout

    results = OrderedDict()
    conversions = am.string_view_conversions()
    for name, mod in _engines(args.engine):
        results[name] = run_engine(name, mod, args.quick, args.quiet,
                                   dtypes=args.dtypes.split(",") if args.dtypes else None)
    conversions = tuple(b - a for a, b in zip(conversions, am.string_view_conversions()))
    print(render(results, args.quick))
    print(f"String columns handed over as {args.string_layout}; view columns converted to offsets + "
          f"bytes during the run: {conversions[0]} ({conversions[1]} rows)")
    if args.csv_dir:
        for path in write_csv(results, args.csv_dir, args.stamp):
            print(f"wrote {path}")
    bad = sum(res[1].unclassified for res in results.values() if res is not None)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
