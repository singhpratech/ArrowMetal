#!/usr/bin/env python3
"""Generates the measured join table (datafusion/src/join_table.rs) from join CSVs.

The measurement is `examples/bench.rs --families join --contexts off,on --on joins --idle-ctx on
--idle-reps N --idle-gap-ms 500,5000`: per probe size (the bench's `--sizes`), build size (10k, 1M,
10M rows in BUILD_BUCKETS) and table layout, an inner and a left join of `probe` (int64 key) and
`build` (int64 key, every other value of the probe's key domain, so half the probe rows match),
once with count + sum over the whole result (`j_<how>_<build>`) and once with a GROUP BY of the
probe key above the join (`jg_<how>_<build>`); a key class other than int64 is a `j32` / `js` prefix
(int32, Utf8). DataFusion alone (`off`) against the default config with every translatable join
replaced (`on` with `--on joins`, `JoinChoice::ArrowMetal`; the aggregate above the join stays
DataFusion's, as under the default): warm (best over the rounds), and the first run after each idle gap
(the median of the runs after it; `def_idle_*` columns hold the `--idle-ctx` context's runs).

DataFusion plans the SQL left join of the larger table (`probe LEFT JOIN build`) as a
`HashJoinExec` with `join_type=Right`: the smaller input is the build (left) side and the probe
(right) side's rows are kept. So the SQL `left` cases fill the table's `right` rows. Nothing
measured fills its `left` rows (the build side's rows kept), which are therefore not taken.

The rule (per join type, key class, input class and build bucket):

* A probe size passes when every case of the bucket at that size, in both layouts, gave the same
  answer with the rule and has
  - warm: off_ms / on_ms >= WARM_RATIO (best runs), and
  - idle against idle at every gap of IDLE_GAPS_MS: DataFusion alone's first run after the gap /
    the rule's first run after the same gap >= IDLE_RATIO.
* The row is taken from the smallest measured probe size from which every larger measured size
  passes as well; probe sizes below it are left. A row with no such size is not taken.
* Build sizes go to BUILD_BUCKETS (half a decade either side of the measured size). A join whose
  build side falls outside every bucket is left.

A later CSV's row replaces an earlier one's for the same size, layout and case.

Usage:
    python3 scripts/join_table.py results/<a>.csv [results/<b>.csv ...]   # writes src/join_table.rs
    python3 scripts/join_table.py results/<a>.csv ... --print            # the cells and the fit
    python3 scripts/join_table.py --check                                # exit 1 if src/join_table.rs is stale
"""
import argparse
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "src", "join_table.rs")

WARM_RATIO = 1.5
IDLE_RATIO = 1.0
IDLE_GAPS_MS = (500, 5000)
# (name, lowest, highest build rows): half a decade (sqrt 10) either side of the measured size.
BUILD_BUCKETS = [("10k", 3_163, 31_622), ("1M", 316_228, 3_162_277), ("10M", 3_162_278, 31_622_776)]
MEASURED_BUILD = {"10k": 10_000, "1M": 1_000_000, "10M": 10_000_000}
CASE = re.compile(r"^(j|jg)(32|s)?_(inner|left)_(10k|1M|10M)$")
KEY_CLASS = {None: "i64", "32": "i32", "s": "string"}
# The SQL join of the bench -> the HashJoinExec's join type.
PHYSICAL = {"inner": "inner", "left": "right"}


def rows_text(n):
    if n >= 1_000_000 and n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n >= 1_000 and n % 1_000 == 0:
        return f"{n // 1_000}k"
    return str(n)


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(paths):
    latest = {}
    for p in paths:
        with open(os.path.join(HERE, p) if not os.path.isabs(p) else p) as fh:
            for r in csv.DictReader(fh):
                # Only rows whose `on` context replaced the joins alone (`--on joins`).
                if r.get("family") != "join" or r.get("on_config") != "joins":
                    continue
                m = CASE.match(r["case"])
                if not m:
                    continue
                latest[(int(r["size"]), r["layout"], r["case"])] = r
    return latest


def points(latest):
    """(how, key class, build bucket) -> {probe size: [point dict]}"""
    cells = {}
    for (size, layout, case), r in latest.items():
        kind, kc, how, build = CASE.match(case).groups()
        key = (PHYSICAL[how], KEY_CLASS[kc], build)
        warm = None
        if f(r["off_ms"]) and f(r["on_ms"]):
            warm = f(r["off_ms"]) / f(r["on_ms"])
        # The idle columns hold the rule's runs only when the bench ran with `--idle-ctx on`.
        idle = {}
        if r.get("idle_ctx") == "on":
            gaps = [(f(r.get("idle_gap_ms")), f(r.get("off_idle_ms")), f(r.get("def_idle_ms"))),
                    (f(r.get("idle_gap2_ms")), f(r.get("off_idle2_ms")), f(r.get("def_idle2_ms")))]
            for g, off, on in gaps:
                if g is not None and off and on:
                    idle[int(g)] = off / on
        equal = r["equal"] == "yes"
        cells.setdefault(key, {}).setdefault(size, []).append(
            {"case": case, "layout": layout, "warm": warm, "idle": idle, "equal": equal}
        )
    return cells


def passes(pts):
    """(passes, lowest warm, lowest idle per gap, the row that failed or None)"""
    worst_warm = min((p["warm"] for p in pts if p["warm"] is not None), default=None)
    worst_idle = {}
    # Every failure as (how far below its bar, text); the reason names the farthest below.
    fails = []
    for p in pts:
        if not p["equal"]:
            fails.append((-1.0, f"{p['case']} {p['layout']}: answers differ"))
        if p["warm"] is None or p["warm"] < WARM_RATIO:
            fails.append(((p["warm"] or 0) / WARM_RATIO, f"{p['case']} {p['layout']}: warm {p['warm'] or 0:.2f}x"))
        for g in IDLE_GAPS_MS:
            x = p["idle"].get(g)
            if x is None:
                fails.append((0.0, f"{p['case']} {p['layout']}: idle at {g} ms not measured"))
                continue
            worst_idle[g] = min(worst_idle.get(g, x), x)
            if x < IDLE_RATIO:
                fails.append((x / IDLE_RATIO, f"{p['case']} {p['layout']}: idle vs idle {x:.2f}x after {g} ms"))
    fail = min(fails)[1] if fails else None
    return fail is None, worst_warm, worst_idle, fail


def fit(cells):
    rows = []
    for key in sorted(cells):
        how, kc, build = key
        sizes = sorted(cells[key])
        verdict = {s: passes(cells[key][s]) for s in sizes}
        start = None
        for i, s in enumerate(sizes):
            if all(verdict[t][0] for t in sizes[i:]):
                start = s
                break
        parts = []
        for s in sizes:
            ok, w, idle, fail = verdict[s]
            idle_t = ", ".join(f"{g} ms {idle[g]:.2f}x" for g in IDLE_GAPS_MS if g in idle)
            parts.append(f"{rows_text(s)}: warm {w or 0:.2f}x, idle {idle_t}" + ("" if ok else f" [{fail}]"))
        if start is None:
            reason = "not taken at any measured probe size; " + "; ".join(parts)
        else:
            reason = f"taken from {rows_text(start)} probe rows; " + "; ".join(parts)
        rows.append((how, kc, "memory", build, start, reason))
    return rows


def render(rows, sources):
    out = []
    out.append("//! The measured join table. Generated by scripts/join_table.py from")
    for s in sources:
        out.append(f"//! {s}")
    out.append("//! Do not edit by hand (`--check` fails when this file and those disagree).")
    out.append("//!")
    out.append("//! One row per (join type of DataFusion's `HashJoinExec`, key class, input class, build-side")
    out.append("//! bucket) measured: `min_probe_rows` is the fewest probe-side rows from which every case, in")
    out.append(f"//! both layouts, was at least {WARM_RATIO}x faster than DataFusion alone warm and at least")
    out.append(f"//! {IDLE_RATIO}x on the first run after {' ms and '.join(str(g) for g in IDLE_GAPS_MS)} ms of idle against DataFusion alone's")
    out.append("//! first run after the same idle, at that size and every larger measured size (None: not")
    out.append("//! taken). Every row ends with the reason for its threshold.")
    out.append("")
    out.append("pub(crate) struct JoinRow {")
    out.append("    pub how: &'static str,")
    out.append("    pub key_class: &'static str,")
    out.append("    pub source: &'static str,")
    out.append("    pub build: &'static str,")
    out.append("    pub min_probe_rows: Option<u64>,")
    out.append("    pub reason: &'static str,")
    out.append("}")
    out.append("")
    out.append(f"pub(crate) const SOURCE: &str = \"{', '.join(sources)}\";")
    b = ", ".join(f"(\"{n}\", {lo}, {hi})" for n, lo, hi in BUILD_BUCKETS)
    out.append(f"pub(crate) const BUILD_BUCKETS: &[(&str, u64, u64)] = &[{b}];")
    out.append("")
    out.append("pub(crate) const TABLE: &[JoinRow] = &[")
    for how, kc, src, build, start, reason in rows:
        m = "None" if start is None else f"Some({start})"
        reason = reason.replace("\\", "\\\\").replace('"', '\\"')
        out.append(
            f"    JoinRow {{ how: \"{how}\", key_class: \"{kc}\", source: \"{src}\", build: \"{build}\", "
            f"min_probe_rows: {m}, reason: \"{reason}\" }},"
        )
    out.append("];")
    return "\n".join(out) + "\n"


def sources_of(path):
    srcs = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("//! results/"):
                srcs.append(line[4:].strip())
            elif not line.startswith("//!"):
                break
    return srcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="*")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        srcs = sources_of(OUT)
        want = render(fit(points(load(srcs))), srcs)
        have = open(OUT).read()
        if want != have:
            print("src/join_table.rs is stale: regenerate with scripts/join_table.py " + " ".join(srcs))
            sys.exit(1)
        print("src/join_table.rs is up to date")
        return
    if not a.csvs:
        ap.error("give the join CSVs")
    srcs = [os.path.relpath(os.path.abspath(p), HERE) for p in a.csvs]
    cells = points(load(srcs))
    rows = fit(cells)
    if a.print:
        for key in sorted(cells):
            print(key)
            for s in sorted(cells[key]):
                for p in cells[key][s]:
                    idle = ", ".join(f"{g}: {x:.2f}" for g, x in sorted(p["idle"].items()))
                    print(f"  {rows_text(s):>4} {p['layout']:6} {p['case']:16} warm {p['warm'] or 0:6.2f}  idle {idle}  equal {p['equal']}")
        for r in rows:
            print(r[:5], r[5])
        return
    open(OUT, "w").write(render(rows, srcs))
    print(f"wrote {OUT}: {sum(1 for r in rows if r[4] is not None)} of {len(rows)} rows taken")


if __name__ == "__main__":
    main()
