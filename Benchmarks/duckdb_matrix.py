"""DuckDB through the full_matrix.py protocol, for the sort, group-by, filter and sum rows.

    PYTHONPATH=python python Benchmarks/duckdb_matrix.py [rows ...]          # measure (default 10M and 50M)
    python Benchmarks/duckdb_matrix.py --report Benchmarks/results/duckdb_matrix_<date>.csv   # markdown

`full_matrix.py` has no DuckDB column because DuckDB is a query engine, not an array library: it
takes SQL over a table and hands back a result set, so the fair comparison needs a table, a query
and a decision about where the result lands. This script measures that, reusing full_matrix.py's
generators (same seed, distributions and sizes), its Bench (best of up to 5 after one warm-up, 1.2 s
budget, never fewer than 2) and its byte counts per row, so GB/s is comparable with
docs/BENCHMARKS_MATRIX.md. Four DuckDB idioms per row, each the same SQL text:

  duckdb-arrow         over the in-process Arrow table (con.register) in one chunk, result fetched as
                       an Arrow table. DuckDB parallelises an Arrow scan per record batch, so a single
                       chunk scans on about two cores.
  duckdb-arrow-<N>     the same table split into one record batch per hardware thread, the way the
                       matrix feeds Acero for its pyarrow-threaded rows.
  duckdb-native        over a DuckDB table created from the Arrow table (CREATE TABLE AS), result
                       fetched as an Arrow table: what a Python caller gets.
  duckdb-native-ctas   the same query materialised into a DuckDB temp table (CREATE OR REPLACE TEMP
                       TABLE r AS ...): engine time without the export to Arrow.

Results are fetched with fetch_arrow_table() and every row asserts its row count. In duckdb 1.5
`.arrow()` returns a lazy RecordBatchReader, and timing that measures nothing. DuckDB has no argsort,
so those rows carry an explicit int64 row index column and read 8 bytes per row more than the kernel.
CPU ms is process-wide (getrusage), so it counts DuckDB's worker threads, as the matrix does for Polars.
Nothing in the published matrix is re-measured; --report joins this CSV with the 2026-09-07 rows.
"""
import csv, datetime, os, platform, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
REF = os.path.join(ROOT, "Benchmarks", "results", "full_matrix_2026-09-07-parallel.csv")
STR_SIZES = [1_000_000, 10_000_000]
ROWS = []


def sysctl(k):
    return subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()


def rec(family, op, n, idiom, wall, cpu, it, nbytes, sql, note=""):
    ROWS.append(dict(family=family, op=op, rows=n, library=idiom, wall_ms=round(wall, 4),
                     cpu_ms=round(cpu, 4), gbs=round(nbytes / wall / 1e6, 3), cores=round(cpu / wall, 2),
                     iterations=it, sql=sql, note=note))
    print(f"{family:14s} {op:44s} {n:>9d} {idiom:18s} {wall:9.3f} ms {cpu:10.3f} cpu-ms "
          f"{cpu / wall:5.1f} cores  n={it}", flush=True)


def measure(sizes):
    sys.path.insert(0, os.path.join(ROOT, "Benchmarks"))
    import full_matrix as fm
    import duckdb, numpy as np, pyarrow as pa

    def q(con, sql, expect):
        out = con.execute(sql).fetch_arrow_table()
        assert isinstance(out, pa.Table), type(out)
        if expect is not None:
            assert out.num_rows == expect, (sql, out.num_rows, expect)
        return out

    def both(con, family, op, n, nbytes, tbl, sql, note="", expect=None):
        con.register("t", tbl)
        w, c, it = fm.BENCH.run(lambda: q(con, sql.format(t="t"), expect))
        rec(family, op, n, "duckdb-arrow", w, c, it, nbytes, sql.format(t="t"), note)
        con.unregister("t")
        parts = tbl.to_batches(max_chunksize=-(-tbl.num_rows // fm.NCHUNK))
        con.register("t", pa.Table.from_batches(parts))
        w, c, it = fm.BENCH.run(lambda: q(con, sql.format(t="t"), expect))
        rec(family, op, n, f"duckdb-arrow-{fm.NCHUNK}", w, c, it, nbytes, sql.format(t="t"),
            (note + "; " if note else "") + f"Arrow table in {len(parts)} record batches")
        con.execute("CREATE OR REPLACE TABLE tn AS SELECT * FROM t")
        w, c, it = fm.BENCH.run(lambda: q(con, sql.format(t="tn"), expect))
        rec(family, op, n, "duckdb-native", w, c, it, nbytes, sql.format(t="tn"), note)
        ctas = "CREATE OR REPLACE TEMP TABLE r AS " + sql.format(t="tn")
        w, c, it = fm.BENCH.run(lambda: con.execute(ctas))
        if expect is not None:
            got = con.execute("SELECT count(*) FROM r").fetchone()[0]
            assert got == expect, (ctas, got, expect)
        rec(family, op, n, "duckdb-native-ctas", w, c, it, nbytes, ctas,
            (note + "; " if note else "") + "materialised in DuckDB, not exported")
        con.execute("DROP TABLE r")
        con.execute("DROP TABLE tn")
        con.unregister("t")

    con = duckdb.connect()
    threads = con.execute("SELECT current_setting('threads')").fetchone()[0]
    pio = con.execute("SELECT current_setting('preserve_insertion_order')").fetchone()[0]
    hdr = (f"duckdb {duckdb.__version__}, threads={threads}, preserve_insertion_order={pio}, "
           f"pyarrow {pa.__version__}, python {platform.python_version()}, "
           f"{sysctl('machdep.cpu.brand_string')}, macOS {platform.mac_ver()[0]}, "
           f"loadavg at start {os.getloadavg()[0]:.1f}, {datetime.datetime.now().isoformat(timespec='seconds')}")
    print(hdr, flush=True)
    ARGSORT_NOTE = ("DuckDB has no argsort; the table carries an explicit int64 row index i and the "
                    "query is ORDER BY x returning i, so the input is 8 bytes/row wider than the kernel's")
    for n in sizes:
        d = fm.Data(n)
        i64, m30 = d("i64"), d("mask30")
        both(con, "reductions", "sum(int64, 10% nulls)", n, n * 8, pa.table({"x": i64.a}),
             "SELECT sum(x) FROM {t}", "DuckDB sum(BIGINT) returns HUGEINT", expect=1)
        both(con, "compare+select", "filter int64 (30% kept)", n, int(n * 8 * 1.3),
             pa.table({"x": i64.a, "m": m30.a}), "SELECT x FROM {t} WHERE m",
             expect=int(np.count_nonzero(m30.n)))
        ii, ff = d("i64_nn"), d("f64_nn")
        idx = pa.array(np.arange(n, dtype=np.int64))
        both(con, "sort", "argsort int64", n, n * 12, pa.table({"x": ii.a, "i": idx}),
             "SELECT i FROM {t} ORDER BY x", ARGSORT_NOTE, expect=n)
        both(con, "sort", "argsort float64", n, n * 12, pa.table({"x": ff.a, "i": idx}),
             "SELECT i FROM {t} ORDER BY x", ARGSORT_NOTE, expect=n)
        both(con, "sort", "sort float64", n, n * 16, pa.table({"x": ff.a}),
             "SELECT x FROM {t} ORDER BY x", expect=n)
        k1, k2 = d("keys_lex_a"), d("keys_lex_b")
        both(con, "sort", "lexsort (2 int32 keys)", n, n * 16, pa.table({"a": k1.a, "b": k2.a, "i": idx}),
             "SELECT i FROM {t} ORDER BY a, b", ARGSORT_NOTE, expect=n)
        del idx
        vals = d("i64")
        for distinct in (1_000, 100_000, 10_000_000):
            if distinct > n:
                continue
            k = d.keys(distinct)
            tbl = pa.table({"k": k.a, "x": vals.a})
            aggs = ([("sum", "sum(x)"), ("mean", "avg(x)"), ("count", "count(x)")]
                    if distinct < 10_000_000 else [("sum", "sum(x)")])
            for name, expr in aggs:
                both(con, "group-by", f"{name} by int32 key ({distinct} groups)", n, n * 12, tbl,
                     f"SELECT k, {expr} FROM {{t}} GROUP BY k")
            if distinct == 1_000:
                side = max(2, int(np.ceil(np.sqrt(distinct))))
                ka = pa.array(d.rng.integers(0, side, size=n, dtype=np.int32))
                kb = pa.array(d.rng.integers(0, side, size=n, dtype=np.int32))
                both(con, "group-by", f"sum by two int32 keys (~{side * side} groups)", n, n * 16,
                     pa.table({"a": ka, "b": kb, "x": vals.a}), "SELECT a, b, sum(x) FROM {t} GROUP BY a, b")
                del ka, kb
            del tbl
        del d
    for sn in STR_SIZES:
        sd = fm.StrData(sn)
        both(con, "sort", "sort utf8", sn, sd.s.nbytes * 2, pa.table({"s": sd.s.a}),
             "SELECT s FROM {t} ORDER BY s", expect=sn)
        del sd
    print(f"loadavg at end {os.getloadavg()[0]:.1f}", flush=True)
    out = os.path.join(ROOT, "Benchmarks", "results", f"duckdb_matrix_{datetime.date.today().isoformat()}.csv")
    with open(out, "w", newline="") as f:
        f.write("# " + hdr + "\n")
        w = csv.DictWriter(f, fieldnames=list(ROWS[0].keys()))
        w.writeheader()
        w.writerows(ROWS)
    print("wrote", out)


def report(path):
    """Markdown: this CSV joined with the published 2026-09-07 matrix rows, which are not re-measured."""
    R = {}
    with open(REF) as f:
        for r in csv.DictReader(f):
            if r["library"] in ("arrowmetal", "polars", "polars-lazy", "pyarrow", "pyarrow-threaded") \
                    and r["status"] == "ok":
                R[(r["family"], r["op"], int(r["rows"]), r["library"])] = (float(r["wall_ms"]), float(r["cpu_ms"]))
    D, order = {}, []
    with open(path) as f:
        hdr = f.readline().strip("# \n")
        for r in csv.DictReader(f):
            k = (r["family"], r["op"], int(r["rows"]))
            if k not in order:
                order.append(k)
            D.setdefault(k, {})[r["library"]] = (float(r["wall_ms"]), float(r["cpu_ms"]), r["sql"], r["note"])

    def ms(x):
        return f"{x:,.1f}" if x >= 10 else f"{x:.2f}"

    def cell(x):
        return f"{ms(x[0])} ({x[1]:,.0f})" if x else "-"

    date = os.path.basename(path)[len("duckdb_matrix_"):-len(".csv")]
    print(f"DuckDB rows measured {date}: {hdr}.")
    print("ArrowMetal, Polars and pyarrow are the published 2026-09-07 matrix rows "
          "(`Benchmarks/results/full_matrix_2026-09-07-parallel.csv`), not re-measured. Wall ms, best of up to "
          "5 after one warm-up, process CPU ms in brackets. \"Best\" is the faster of a library's eager and "
          "parallel idioms. The last column is the fastest CPU number on the row, any library or idiom, "
          "over ArrowMetal.\n")
    print("| family | op | rows | ArrowMetal | Polars best | pyarrow best | DuckDB native, to Arrow | "
          "DuckDB native, stays in DuckDB | DuckDB Arrow scan, 16 batches | DuckDB Arrow scan, 1 chunk | "
          "fastest CPU / ArrowMetal |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (fam, op, n) in order:
        am = R.get((fam, op, n, "arrowmetal"))
        pol = min((R[k] for k in R if k[:3] == (fam, op, n) and k[3].startswith("polars")), default=None)
        pya = min((R[k] for k in R if k[:3] == (fam, op, n) and k[3].startswith("pyarrow")), default=None)
        d = D[(fam, op, n)]
        nat, ct = d.get("duckdb-native"), d.get("duckdb-native-ctas")
        a16 = next((v for k, v in d.items() if k.startswith("duckdb-arrow-")), None)
        a1 = d.get("duckdb-arrow")
        cands = [x[0] for x in (pol, pya, nat, a16, ct) if x]
        ratio = f"{min(cands) / am[0]:.1f}x" if am and cands else "-"
        print(f"| {fam} | {op} | {n:,} | {cell(am) if am else '-'} | {cell(pol)} | {cell(pya)} | {cell(nat)} | "
              f"{cell(ct)} | {cell(a16)} | {ms(a1[0]) if a1 else '-'} | {ratio} |")
    print("\nSQL per operation, over the native table; the Arrow-scan idioms run the same text over the "
          "registered Arrow table, and the stays-in-DuckDB idiom wraps it in "
          "`CREATE OR REPLACE TEMP TABLE r AS`:\n")
    seen = set()
    for (fam, op, n) in order:
        d = D[(fam, op, n)].get("duckdb-native")
        if d and op not in seen:
            seen.add(op)
            print(f"- {op}: `{d[2]}`" + (f" ({d[3].split(';')[0]})" if d[3] else ""))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--report":
        report(sys.argv[2])
    else:
        measure([int(s) for s in (sys.argv[1:] or ["10000000", "50000000"])])
