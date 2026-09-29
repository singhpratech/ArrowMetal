#!/usr/bin/env python3
"""Generates the measured aggregate table (datafusion/src/agg_table.rs) from a sweep CSV.

The sweep is `examples/bench.rs --families gsweep --contexts off,on,...`: every aggregate family
(count; sum and avg over int64; sum and avg over Float64; min + max over int64; min + max over
Float64; DISTINCT) over one and two keys, int32 and int64 keys, at five group counts in the key
domain (200, 10k, 100k, 1M, rows/2), per input size and batch layout; DataFusion alone (`off`)
against the rule with every replaced aggregate forced onto ArrowMetal (`on`), both warm, best of 5.

The fit:

* Each (case, size, layout) is a point with ratio = off_ms / on_ms (0 when the answers differed).
  It goes to the group-count bucket of the groups its data holds at that size (`out_rows`):
  BUCKETS by group count, ROWS_BUCKET for at least rows / NEAR_ROWS groups, none in between.
* Per (family, keys, key class, input, bucket) and size, the series' ratio is the worst point
  there: over both layouts and over the cases of the family (sum and avg form one family).
* A series is taken from the smallest measured size from which its ratio is at least
  MIN_RATIO x HEADROOM at that size and every larger measured size, and only when that holds at
  two or more measured sizes (one measured size alone is not a crossover).
* The take is capped at the largest measured size when the ratio falls between the two largest
  measured sizes (a falling ratio cannot be extended past the sweep); otherwise it has no upper
  bound.
* Everything else is not taken: the MetalExec hands it back to DataFusion (or the rule leaves it
  at plan time when no bucket is taken at the input's row count).

Two constraints from the default-take check CSVs (`--check`, merged in order: a later file's row
replaces an earlier one's for the same size, layout and case) then raise the thresholds:

Each constraint reads, per size, layout and case, the latest row in which the default took that path
(ran it on ArrowMetal, or handed it back), whatever a later file decided.

* First run after idle: a series (family, keys, key class, input, bucket) is taken at a size only
  if, at that size and every larger measured size, every row the default ran on ArrowMetal has
  `off_ms / def_idle_ms >= IDLE_RATIO` (the default's first run after the GPU idled, pipelines
  already compiled, against DataFusion alone's warm time). A size with no such row is not taken.
* Hand-back: a shape (family, keys, key class, input) is replaced at plan time from a size only if,
  at that size and every larger measured size, every row the default handed back at run time is at
  least HANDBACK_RATIO of DataFusion alone on the best run AND on the median of the round bests.
  Every bucket of the shape is raised to that size; a shape with no such size is not taken.

Usage:
    python3 scripts/groupby_table.py results/<sweep>.csv --check-csv results/<check>.csv ...  # writes src/agg_table.rs
    python3 scripts/groupby_table.py results/<sweep>.csv --print    # the series and the fit
    python3 scripts/groupby_table.py --check                        # exit 1 if src/agg_table.rs is stale
"""
import argparse
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "src", "agg_table.rs")

MIN_RATIO = 1.5
# The sweep ratio needed at a size is MIN_RATIO x HEADROOM: a shape measured at 1.5x sits within
# run-to-run noise of 1.5x, so the default takes only shapes measured at least 10 % above it.
HEADROOM = 1.1
# The default's first run after the GPU idled (pipelines compiled) must not be slower than
# DataFusion alone warm.
IDLE_RATIO = 1.0
# A run-time hand-back may cost at most 3 % against DataFusion alone (best and median of rounds).
HANDBACK_RATIO = 0.97
SIZES_ALL = (1_000_000, 2_000_000, 5_000_000, 10_000_000, 50_000_000)
BUCKETS = (("200", 1, 1_414), ("10k", 1_415, 31_622), ("100k", 31_623, 316_227), ("1M", 316_228, 3_162_277))
NEAR_ROWS = 4
ROWS_BUCKET = "rows/2"
ORDER = [b[0] for b in BUCKETS] + [ROWS_BUCKET]
FAMILY = {
    "count": "count",
    "sum_f64": "sum_avg_f64",
    "avg_f64": "sum_avg_f64",
    "minmax_f64": "minmax_f64",
    "sum_int": "sum_avg_int",
    "avg_int": "sum_avg_int",
    "minmax_int": "minmax_int",
    "distinct": "distinct",
}
CASE = re.compile(r"^(?P<fam>[a-z_0-9]+?)_(?P<nk>[12])(?P<kc>i32|i64)_(?P<g>[^_]+)$")


def bucket(groups, rows):
    if groups <= 0:
        return None
    if groups * NEAR_ROWS >= rows:
        return ROWS_BUCKET
    for name, lo, hi in BUCKETS:
        if lo <= groups <= hi:
            return name
    return None


def read(path):
    """{(family, keys, key class, input, bucket): {size: [(case, layout, ratio, off, on)]}}"""
    series = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["family"] != "gsweep" or not r["on_ms"]:
                continue
            m = CASE.match(r["case"])
            if not m:
                raise SystemExit(f"unexpected case id {r['case']}")
            size = int(r["size"])
            b = bucket(int(r["out_rows"]), size)
            if b is None:
                continue
            off, on = float(r["off_ms"]), float(r["on_ms"])
            ratio = off / on if r["equal"] == "yes" else 0.0
            key = (FAMILY[m["fam"]], "1" if m["nk"] == "1" else "2+", m["kc"], "memory", b)
            series.setdefault(key, {}).setdefault(size, []).append((r["case"], r["layout"], ratio, off, on))
    return series


def fit(points):
    """(min_rows or None, max_rows or None, {size: worst ratio})"""
    worst = {n: min(p[2] for p in pts) for n, pts in points.items()}
    sizes = sorted(worst)
    need = MIN_RATIO * HEADROOM
    start = None
    for i in range(len(sizes)):
        if all(worst[n] >= need for n in sizes[i:]):
            start = i
            break
    if start is None or len(sizes) - start < 2:
        return None, None, worst
    cap = None
    if len(sizes) >= 2 and worst[sizes[-1]] < worst[sizes[-2]]:
        cap = sizes[-1]
    return sizes[start], cap, worst


def read_checks(paths, kind=None):
    """The default-check rows, merged in order (a later file's row replaces an earlier one's).
    With `kind` ("gpu" or "back"), only rows the default decided that way are merged: the latest
    measurement of that path for each size, layout and case, whatever later files decided."""
    merged = {}
    for path in paths:
        with open(path) as fh:
            for r in csv.DictReader(fh):
                if r["family"] != "gsweep" or not CASE.match(r["case"]):
                    continue
                if kind is not None and state(r) != kind:
                    continue
                merged[(r["size"], r["layout"], r["case"])] = r
    return list(merged.values())


def state(r):
    if float(r.get("def_handed_back") or 0) > 0:
        return "back"
    if int(r.get("def_taken") or 0) > 0:
        return "gpu"
    return "left"


def shape_of(r):
    m = CASE.match(r["case"])
    return FAMILY[m["fam"]], "1" if m["nk"] == "1" else "2+", m["kc"], "memory"


def first_holding(ok_by_size):
    """The smallest size from which every measured size passes ({size: bool}; sizes absent pass)."""
    for n in SIZES_ALL:
        if all(ok_by_size.get(m, True) for m in SIZES_ALL if m >= n):
            return n
    return None


def constraints(paths):
    """({series key: {size: every first run after idle passes}}, {shape: {size: every hand-back
    passes}}, the rows that fail), from the latest measurement of each path per size, layout and
    case (`read_checks` with `kind`)."""
    idle, back, fails = {}, {}, []
    for r in read_checks(paths, "gpu") + read_checks(paths, "back"):
        size = int(r["size"])
        shape = shape_of(r)
        if state(r) == "gpu":
            b = bucket(int(r["out_rows"]), size)
            key = shape + (b,)
            v = r.get("def_idle_ms")
            ok = bool(v) and float(r["off_ms"]) / float(v) >= IDLE_RATIO
            idle.setdefault(key, {})
            idle[key][size] = idle[key].get(size, True) and ok
            if not ok:
                fails.append(("idle", key, size, r["layout"], r["case"], r["off_ms"], v or "-"))
        elif state(r) == "back":
            best = float(r["off_ms"]) / float(r["def_ms"])
            med = float(r["off_round_median_ms"] or r["off_ms"]) / float(r["def_round_median_ms"] or r["def_ms"])
            ok = best >= HANDBACK_RATIO and med >= HANDBACK_RATIO
            back.setdefault(shape, {})
            back[shape][size] = back[shape].get(size, True) and ok
            if not ok:
                fails.append(("handback", shape, size, r["layout"], r["case"], f"{best:.3f}", f"{med:.3f}"))
    return idle, back, fails


def table(path, checks=()):
    s = read(path)
    idle, back, _ = constraints(checks) if checks else ({}, {}, [])
    rows = []
    for key in sorted(s, key=lambda k: (k[0], k[1], k[2], k[3], ORDER.index(k[4]))):
        lo, hi, worst = fit(s[key])
        why = []
        if lo is not None and checks:
            sizes = idle.get(key, {})
            # A size at or above the threshold where the default never ran this series on the GPU
            # has no first-run measurement: not taken there.
            for n in SIZES_ALL:
                if n >= lo and n in worst and n not in sizes:
                    sizes = {**sizes, n: False}
            first = first_holding(sizes)
            if first is None or (hi is not None and first > hi):
                why.append("first run after idle below DataFusion alone at every size")
                lo = hi = None
            elif first > lo:
                why.append(f"first run after idle holds from {first // 1_000_000}M")
                lo = first
        if lo is not None and checks:
            first = first_holding(back.get(key[:4], {}))
            if first is None or (hi is not None and first > hi):
                why.append("a run-time hand-back of the shape below the limit at the largest size")
                lo = hi = None
            elif first > lo:
                why.append(f"run-time hand-backs of the shape hold from {first // 1_000_000}M")
                lo = first
        rows.append((key, lo, hi, worst, why))
    return rows


def rust(path, rows, checks=()):
    src = os.path.relpath(os.path.abspath(path), HERE)
    out = [
        "//! The measured aggregate table. Generated by scripts/groupby_table.py from",
        f"//! {src}; do not edit by hand (`--check` fails when this file and that one disagree).",
    ]
    for c in checks:
        out.append(f"//! check: {os.path.relpath(os.path.abspath(c), HERE)}")
    out += [
        "//!",
        "//! One row per (aggregate family, key count, key class, input, group-count bucket) the sweep",
        "//! measured: `min_rows` is the fewest input rows from which ArrowMetal was measured at least",
        f"//! MIN_RATIO x HEADROOM ({MIN_RATIO} x {HEADROOM}) ahead of DataFusion alone at that size and every",
        "//! larger one (None: not taken at any size); `max_rows` caps a take whose ratio fell between the",
        "//! two largest sizes; `ratios` is the worst ratio (both layouts, every case of the family) per size.",
        f"//! The `check:` files raise `min_rows` where the default's first run after idle is below",
        f"//! {IDLE_RATIO}x of DataFusion alone warm, or a run-time hand-back of the shape below {HANDBACK_RATIO}x",
        "//! (best and median of rounds); the reason follows the row.",
        "",
        "pub(crate) struct Row {",
        "    pub family: &'static str,",
        "    pub keys: &'static str,",
        "    pub key_class: &'static str,",
        "    pub source: &'static str,",
        "    pub bucket: &'static str,",
        "    pub min_rows: Option<u64>,",
        "    pub max_rows: Option<u64>,",
        "    pub ratios: &'static [(u64, f64)],",
        "}",
        "",
        f"pub(crate) const SOURCE: &str = {src!r};".replace("'", '"'),
        f"pub(crate) const MIN_RATIO: f64 = {MIN_RATIO};",
        f"pub(crate) const HEADROOM: f64 = {HEADROOM};",
        f"pub(crate) const NEAR_ROWS: u64 = {NEAR_ROWS};",
        f'pub(crate) const ROWS_BUCKET: &str = "{ROWS_BUCKET}";',
        "pub(crate) const BUCKETS: &[(&str, u64, u64)] = &["
        + ", ".join(f'("{n}", {lo}, {hi})' for n, lo, hi in BUCKETS)
        + "];",
        "",
        "pub(crate) const TABLE: &[Row] = &[",
    ]
    opt = lambda v: "None" if v is None else f"Some({v})"
    for (fam, keys, kc, source, b), lo, hi, worst, why in rows:
        ratios = ", ".join(f"({n}, {worst[n]:.2})" for n in sorted(worst))
        out.append(
            f'    Row {{ family: "{fam}", keys: "{keys}", key_class: "{kc}", source: "{source}", bucket: "{b}", '
            f"min_rows: {opt(lo)}, max_rows: {opt(hi)}, ratios: &[{ratios}] }},"
            + (f" // {'; '.join(why)}" if why else "")
        )
    out.append("];")
    out.append("")
    return "\n".join(out)


def committed_source():
    src, checks = None, []
    with open(OUT) as fh:
        for line in fh:
            m = re.match(r'pub\(crate\) const SOURCE: &str = "(.*)";', line)
            if m:
                src = os.path.join(HERE, m.group(1))
            m = re.match(r"//! check: (.*)$", line.rstrip())
            if m:
                checks.append(os.path.join(HERE, m.group(1)))
    if src is None:
        raise SystemExit("no SOURCE line in src/agg_table.rs")
    return src, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-csv", action="append", default=[])
    ap.add_argument("--fails", action="store_true", help="print the check rows that fail a constraint")
    a = ap.parse_args()
    if a.csv:
        path, checks = a.csv, a.check_csv
    else:
        path, checks = committed_source()
    rows = table(path, checks)
    text = rust(path, rows, checks)
    if a.fails:
        for f in sorted(constraints(checks)[2], key=str):
            print(*f)
        return
    if a.print:
        for (fam, keys, kc, source, b), lo, hi, worst, why in rows:
            r = "  ".join(f"{n // 1_000_000}M {worst[n]:.2}" for n in sorted(worst))
            took = "-" if lo is None else f"from {lo // 1_000_000}M" + ("" if hi is None else f" to {hi // 1_000_000}M")
            print(f"{fam:12} {keys:2} {kc} {b:7} {took:14} {r}  {'; '.join(why)}")
        return
    if a.check:
        with open(OUT) as fh:
            ok = fh.read() == text
        print("src/agg_table.rs is " + ("up to date" if ok else "STALE"))
        sys.exit(0 if ok else 1)
    with open(OUT, "w") as fh:
        fh.write(text)
    print(f"wrote {OUT}: {sum(1 for r in rows if r[1] is not None)} of {len(rows)} series taken")


if __name__ == "__main__":
    main()
