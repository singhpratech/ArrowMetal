"""Ordinary DuckDB SQL with and without the ArrowMetal optimizer extension (docs/DUCKDB.md §4b).

    PYTHONPATH=python python Benchmarks/duckdb_rewrite_bench.py                    # 1M, 10M, 50M rows
    PYTHONPATH=python python Benchmarks/duckdb_rewrite_bench.py 10000000 --reps 7
    PYTHONPATH=python python Benchmarks/duckdb_rewrite_bench.py --out path.csv

Needs duckdb-extension/build/arrowmetal_rewrite.duckdb_extension (duckdb-extension/build_rewrite.sh).

Each query is the same SQL text run on the same connection three ways:

  duckdb     SET arrowmetal_rewrite = 'off'    DuckDB's own operators
  rewrite    SET arrowmetal_rewrite = 'force'  ARROWMETAL_AGGREGATE for every supported shape
  auto       SET arrowmetal_rewrite = 'auto'   what a user gets by default: the row records whether
                                               the auto gate rewrote the query at this size

The query is materialised into a DuckDB temp table (CREATE OR REPLACE TEMP TABLE r AS ...), so what
is timed is the engine, not an export to Python. Wall ms is the best of --reps runs after one warm-up
(which also compiles the GPU pipeline); CPU ms is the process's (getrusage), so it counts DuckDB's
worker threads. Before timing, the rewritten answer is checked against DuckDB's own, row for row.
`path` and `gpu_ms` come from arrowmetal_rewrites() for the rewritten run: which GPU path ran, and
the time spent in the operator's Finalize (import, GPU, and handing the result back).
"""
import argparse, csv, datetime, os, platform, resource, subprocess, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EXTENSION = os.path.join(ROOT, "duckdb-extension", "build", "arrowmetal_rewrite.duckdb_extension")
DEFAULT_OUT = os.path.join(ROOT, "Benchmarks", "results", "duckdb_rewrite_2026-09-23_provisional.csv")

TABLE = """
CREATE OR REPLACE TABLE t AS SELECT
    (hash(i) % 1000)::INTEGER                                     AS k1k,
    (hash(i + 7) % 100000)::INTEGER                               AS k100k,
    (hash(i + 11) % 10000)::INTEGER                               AS k10k,
    'key_' || (hash(i) % 1000)::VARCHAR                           AS kstr,
    'customer segment ' || (hash(i) % 1000)::VARCHAR || ' of the table' AS klong,
    ((hash(i) % 1000)::BIGINT * 1000003 + i % 3)                   AS kwide,
    (hash(i + 3) % 1000000)::BIGINT * 1000003                     AS kwide1m,
    (hash(i * 3) % 1000000000)::BIGINT                            AS v,
    (hash(i * 5)::HUGEINT - 9223372036854775808)::BIGINT          AS vfull,
    (i % 1000)::INTEGER                                           AS w,
    CASE WHEN hash(i * 11) % 10 = 0 THEN NULL ELSE (hash(i * 13) % 1000000)::BIGINT END AS n
FROM range({rows}) r(i)
"""

QUERIES = [
    ("ungrouped", "sum(BIGINT)", "SELECT sum(v) FROM t"),
    ("ungrouped", "sum(INTEGER)", "SELECT sum(w) FROM t"),
    ("ungrouped", "sum, min, max, avg (BIGINT)", "SELECT sum(v), min(v), max(v), avg(v) FROM t"),
    ("ungrouped", "sum, avg over full-range BIGINT (HUGEINT result)", "SELECT sum(vfull), avg(vfull) FROM t"),
    ("ungrouped", "sum, count, min, max (BIGINT, 10% NULL)", "SELECT sum(n), count(n), min(n), max(n) FROM t"),
    ("ungrouped", "sum, max, avg (BIGINT)", "SELECT sum(v), max(v), avg(v) FROM t"),
    ("ungrouped", "sum, min, max, avg (INTEGER)", "SELECT sum(w), min(w), max(w), avg(w) FROM t"),
    ("ungrouped", "avg (BIGINT)", "SELECT avg(v) FROM t"),
    ("ungrouped", "sum, avg (BIGINT)", "SELECT sum(v), avg(v) FROM t"),
    ("ungrouped", "sum over full-range BIGINT (HUGEINT result)", "SELECT sum(vfull) FROM t"),
    ("grouped", "1k INTEGER keys: sum", "SELECT k1k, sum(v) FROM t GROUP BY k1k"),
    ("grouped", "1k INTEGER keys: sum, count, min, max, avg",
     "SELECT k1k, sum(w), count(*), min(w), max(w), avg(w) FROM t GROUP BY k1k"),
    ("grouped", "1k INTEGER keys: sum, count (10% NULL)", "SELECT k1k, sum(n), count(n) FROM t GROUP BY k1k"),
    ("grouped", "1k INTEGER keys, WHERE w < 500: sum", "SELECT k1k, sum(v) FROM t WHERE w < 500 GROUP BY k1k"),
    ("grouped", "10k INTEGER keys: sum, count", "SELECT k10k, sum(v), count(*) FROM t GROUP BY k10k"),
    ("grouped", "100k INTEGER keys: sum, count", "SELECT k100k, sum(v), count(*) FROM t GROUP BY k100k"),
    ("grouped", "100k INTEGER keys: sum, min, max, avg",
     "SELECT k100k, sum(v), min(w), max(w), avg(v) FROM t GROUP BY k100k"),
    ("grouped", "1k short VARCHAR keys: sum, count", "SELECT kstr, sum(v), count(*) FROM t GROUP BY kstr"),
    ("grouped", "1k long VARCHAR keys: sum, count", "SELECT klong, sum(v), count(*) FROM t GROUP BY klong"),
    ("grouped", "~3k wide BIGINT keys: sum", "SELECT kwide, sum(v) FROM t GROUP BY kwide"),
    ("grouped", "~1M wide BIGINT keys: sum, count", "SELECT kwide1m, sum(v), count(*) FROM t GROUP BY kwide1m"),
    ("grouped", "1k INTEGER keys alone (no aggregates)", "SELECT k1k FROM t GROUP BY k1k"),
    ("grouped", "100k INTEGER keys alone (no aggregates)", "SELECT k100k FROM t GROUP BY k100k"),
    ("grouped", "~1M wide BIGINT keys alone (no aggregates)", "SELECT kwide1m FROM t GROUP BY kwide1m"),
]


def sysctl(key):
    return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()


def cpu_ms():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return (r.ru_utime + r.ru_stime) * 1000.0


def best_of(con, sql, reps):
    ctas = "CREATE OR REPLACE TEMP TABLE r AS " + sql
    con.execute(ctas)  # warm-up (and the GPU pipeline compile, for the rewrite)
    best = None
    for _ in range(reps):
        c0, t0 = cpu_ms(), time.perf_counter()
        con.execute(ctas)
        wall, cpu = (time.perf_counter() - t0) * 1000.0, cpu_ms() - c0
        if best is None or wall < best[0]:
            best = (wall, cpu)
    return best


def main():
    import duckdb
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("rows", nargs="*", type=int, default=[1_000_000, 10_000_000, 50_000_000])
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    if not os.path.exists(EXTENSION):
        sys.exit("build the extension first: duckdb-extension/build_rewrite.sh")

    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    con.execute(f"LOAD '{EXTENSION}'")
    threads = con.sql("SELECT current_setting('threads')").fetchone()[0]
    header = (f"# duckdb {duckdb.__version__}, threads={threads}, python {platform.python_version()}, "
              f"{sysctl('machdep.cpu.brand_string')}, macOS {platform.mac_ver()[0]}, "
              f"loadavg at start {os.getloadavg()[0]:.1f}, {datetime.datetime.now().isoformat(timespec='seconds')}; "
              f"PROVISIONAL: measured while other work shared the machine")
    rows_out = []
    for n in args.rows:
        con.execute(TABLE.format(rows=n))
        for family, op, sql in QUERIES:
            con.execute("SET arrowmetal_rewrite = 'off'")
            expected = sorted(con.sql(sql).fetchall(), key=repr)
            duck = best_of(con, sql, args.reps)

            con.execute("SET arrowmetal_rewrite = 'force'")
            got = sorted(con.sql(sql).fetchall(), key=repr)
            assert got == expected, (op, n)
            gpu = best_of(con, sql, args.reps)
            path, gpu_ms = con.sql("SELECT path, gpu_ms FROM arrowmetal_rewrites() ORDER BY id DESC LIMIT 1").fetchone()

            con.execute("SET arrowmetal_rewrite = 'auto'")
            con.sql("EXPLAIN " + sql).fetchall()
            decision, reason = con.sql(
                "SELECT decision, reason FROM arrowmetal_rewrites() ORDER BY id DESC LIMIT 1").fetchone()

            row = dict(family=family, op=op, rows=n, duckdb_ms=round(duck[0], 3), duckdb_cpu_ms=round(duck[1], 1),
                       rewrite_ms=round(gpu[0], 3), rewrite_cpu_ms=round(gpu[1], 1),
                       speedup=round(duck[0] / gpu[0], 2), path=path, gpu_ms=round(gpu_ms, 3),
                       auto=decision, shape_class=reason.split(": ", 1)[1] if ": " in reason else "",
                       auto_reason=reason, sql=sql)
            rows_out.append(row)
            print(f"{n:>10,d} {op:52s} duckdb {duck[0]:8.2f} ms  rewrite {gpu[0]:8.2f} ms  "
                  f"x{duck[0] / gpu[0]:5.2f}  auto={decision:9s}  {path}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        f.write(header + "\n")
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
