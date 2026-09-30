"""Derives the chunked import's thread table (`ImportThreads` in Sources/ArrowMetal/ChunkedImport.swift)
from a thread sweep of `Benchmarks/chunked_import_bench.py --threads`.

The rule, per import: a thread count is allowed when its CPU time is at most
max(32 CPU-ms, 2.0 x the one-thread CPU time); the policy takes the fastest allowed count, and of the
counts within 5% of its wall time, the smallest. Copies have no CPU limit from the size where the
limit costs wall time: the smallest measured size from which, at every larger size too, some
measured column (a type and chunk layout) is more than 5% slower under the limited count than at its
own fastest count. From there a copy takes the fastest count outright.

The sweep's rows time three columns (three imports) each. Rows are grouped by step kind and by the
bytes one import writes in that step:

* `copy`: int64 / float64 (8 bytes a row) and utf8 (4 bytes of offset plus 12 bytes of string a row,
  the bench's strings): every step moves bytes as they are or with a constant added;
* `views`: utf8_view, 16 bytes of view a row (the view pass; its data buffers are copies).

A group's wall and CPU times are summed over its rows (both chunk layouts, and for views the one-array
column too), the rule is applied to the sums with the 32 CPU-ms scaled by the number of imports, and a
group gets the resulting count. Between two measured sizes the boundary is their geometric mean.

    python Benchmarks/import_threads_policy.py [Benchmarks/results/import_threads_2026-09-30.csv]
        [--cases] [--check]

prints the Swift table (and with --cases each group's pick with its wall and CPU times). --check
compares it with the tables in Sources/ArrowMetal/ChunkedImport.swift and exits 1 when they differ.
"""
import csv
import math
import os
import re
import sys
from collections import defaultdict

ABS_CPU_MS = 32.0   # per import
CPU_FACTOR = 2.0
WITHIN = 0.05
COLUMNS = 3         # imports per sweep row


def per_import_bytes(ty, rows):
    if ty in ("int64", "float64"):
        return "copy", rows * 8
    if ty == "utf8":
        return "copy", rows * 16
    return "views", rows * 16


ROWS = defaultdict(lambda: defaultdict(dict))   # (kind, bytes) -> column -> threads -> (wall, cpu)


def load(path):
    groups = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))
    for r in csv.DictReader(open(path)):
        kind, b = per_import_bytes(r["type"], int(r["rows"]))
        t = int(r["threads"])
        cols = [("chunked_wall_ms", "chunked_cpu_ms", "chunked")]
        if r["type"] == "utf8view" and int(r["chunks"]) > 64:
            cols.append(("single_import_wall_ms", "single_import_cpu_ms", "single"))
        for w, c, path_name in cols:
            e = groups[(kind, b)][t]
            e[0] += float(r[w]); e[1] += float(r[c]); e[2] += 1
            ROWS[(kind, b)][(r["type"], r["chunks"], path_name)][t] = (float(r[w]), float(r[c]))
    return groups


def limited(v, imports):
    """The rule with the CPU limit over {threads: (wall, cpu, ...)} of `imports` imports."""
    c1 = v[1][1]
    cap = max(ABS_CPU_MS * imports, CPU_FACTOR * c1)
    allowed = {t: e for t, e in v.items() if e[1] <= cap}
    best = min(e[0] for e in allowed.values())
    return min(t for t, e in allowed.items() if e[0] <= best * (1 + WITHIN)), cap


def limit_costs_wall(kind, b):
    """Whether some column of this size is more than 5% slower under the limited count than at its fastest."""
    for v in ROWS[(kind, b)].values():
        t, _ = limited(v, COLUMNS)
        if v[t][0] > min(e[0] for e in v.values()) * (1 + WITHIN):
            return True
    return False


def unlimited_from(groups, kind):
    """The smallest measured size from which every size's limit costs wall time (None: nowhere)."""
    sizes = sorted(b for k, b in groups if k == kind)
    start = None
    for b in reversed(sizes):
        if not limit_costs_wall(kind, b):
            break
        start = b
    return start


def pick(threads, unlimited=False):
    # Only the counts every row of the group measured.
    n = max(e[2] for e in threads.values())
    v = {t: e for t, e in threads.items() if e[2] == n}
    if unlimited:
        t = min(v, key=lambda t: v[t][0])
        return t, v[t], v[1], float("inf")
    t, cap = limited(v, COLUMNS * n)
    return t, v[t], v[1], cap


def table(groups, kind):
    sizes = sorted(b for k, b in groups if k == kind)
    start = unlimited_from(groups, kind) if kind == "copy" else None
    picks = [(b, pick(groups[(kind, b)], start is not None and b >= start)[0]) for b in sizes]
    rows = []   # (upper bound in bytes, exclusive; threads)
    for i, (b, t) in enumerate(picks):
        upper = int(math.sqrt(b * picks[i + 1][0])) if i + 1 < len(picks) else None
        if rows and rows[-1][1] == t:
            rows[-1] = (upper, t)
        else:
            rows.append((upper, t))
    return rows


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def swift_tables():
    """The (below, threads) tables `ImportThreads` has, by name (None for Int.max)."""
    src = open(os.path.join(ROOT, "Sources/ArrowMetal/ChunkedImport.swift")).read()
    out = {}
    for name in ("copyTable", "viewTable"):
        body = re.search(name + r": \[\(below: Int, threads: Int\)\] = \[(.*?)\]", src, re.S).group(1)
        out[name] = [(None if a.strip() == "Int.max" else int(a.replace("_", "")), int(b))
                     for a, b in re.findall(r"\(([^,()]+), (\d+)\)", body)]
    return out


def main():
    path = next((a for a in sys.argv[1:] if not a.startswith("--")),
                os.path.join(ROOT, "Benchmarks/results/import_threads_2026-09-30.csv"))
    groups = load(path)
    if "--check" in sys.argv:
        have = swift_tables()
        want = {"copyTable": table(groups, "copy"), "viewTable": table(groups, "views")}
        ok = have == want
        print("tables match" if ok else f"tables differ: source {have}, derived {want}")
        sys.exit(0 if ok else 1)
    if "--cases" in sys.argv:
        starts = {k: unlimited_from(groups, k) if k == "copy" else None for k in ("copy", "views")}
        for (kind, b) in sorted(groups):
            st = starts[kind]
            t, (w, c, _), (w1, c1, _), cap = pick(groups[(kind, b)], st is not None and b >= st)
            print(f"{kind:5} {b / 2**20:9.1f} MiB  t{t:<2}  wall {w:8.2f} ms ({w1 / w:4.2f}x one thread)  "
                  f"CPU {c:7.1f} ms ({c / c1:4.2f}x)  cap {cap:6.1f}")
    for kind in ("copy", "views"):
        print(f"    // {kind}: (bytes below, threads); the last entry has no bound")
        for upper, t in table(groups, kind):
            print(f"    ({upper if upper is not None else 'Int.max'}, {t}),")


if __name__ == "__main__":
    main()
