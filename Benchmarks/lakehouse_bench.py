#!/usr/bin/env python3
"""Reads one Delta Lake table and one Apache Iceberg table with ArrowMetal and the other readers
available on this machine, and writes the timings to a CSV.

    PYTHONPATH=python python Benchmarks/lakehouse_bench.py [--rows 2000000] [--commits 8] [--runs 5]
                                                           [--dir DIR] [--keep] [--out CSV]

Both tables hold the same rows: a partition column (`grp`, 8 values), int64 / int32 / float64 / string /
date / timestamp columns, written in `--commits` appends so each table has several commits (Delta) or
snapshots (Iceberg) and many data files. Three reads per table:

    full        every column, every row
    project     two columns
    filtered    one partition value and a range on a data column (partition pruning plus row filter)

Readers (each skipped, with the reason in the CSV, when it is not installed or does not load):

    arrowmetal          am.read_delta / am.read_iceberg (a ColumnSet of GPU-resident columns)
    arrowmetal_table    the same read exported as a pyarrow.Table
    deltalake           DeltaTable.to_pyarrow_table
    pyiceberg           StaticTable.scan().to_arrow()
    polars              pl.scan_delta / pl.scan_iceberg, collected
    duckdb              delta_scan / iceberg_scan through DuckDB's extensions, fetched as Arrow;
                        only when the extensions load without downloading (autoinstall is switched off)

Every reader's row count is checked against the others for each read. The median of `--runs` timed
runs (after one warm-up) is reported, with the minimum. Numbers taken while other work shares the
machine or the GPU are provisional.
"""
import argparse
import csv
import datetime as dt
import os
import shutil
import statistics
import sys
import tempfile
import time

import numpy as np
import pyarrow as pa

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import arrowmetal as am        # noqa: E402

GROUPS = ["g%d" % i for i in range(8)]


def batch(start, n, rng):
    ids = np.arange(start, start + n, dtype=np.int64)
    return pa.table({
        "id": pa.array(ids),
        "grp": pa.array(np.array(GROUPS, dtype=object)[ids % len(GROUPS)], pa.string()),
        "qty": pa.array(rng.integers(-1000, 1000, n).astype(np.int32)),
        "price": pa.array(rng.random(n) * 1000.0),
        "name": pa.array(np.char.add("n", (ids % 5000).astype(str)).astype(object), pa.string()),
        "day": pa.array((19700 + ids % 365).astype(np.int32), pa.date32()),
        "ts": pa.array(1_704_067_200_000_000 + ids * 1_000_003, pa.timestamp("us", tz="UTC")),
    })


def build(root, rows, commits):
    import deltalake as dl
    delta = os.path.join(root, "delta")
    per = rows // commits
    rng = np.random.default_rng(7)
    parts = [batch(i * per, per, rng) for i in range(commits)]
    if not os.path.exists(os.path.join(delta, "_delta_log")):
        for i, p in enumerate(parts):
            dl.write_deltalake(delta, p, partition_by=["grp"], mode="append" if i else "error")
    iceberg_meta = None
    try:
        from pyiceberg.catalog.sql import SqlCatalog
        cat = SqlCatalog("bench", uri="sqlite:///" + os.path.join(root, "catalog.db"),
                         warehouse="file://" + os.path.join(root, "warehouse"))
        try:
            t = cat.load_table("bench.t")
        except Exception:
            cat.create_namespace_if_not_exists("bench")
            t = cat.create_table("bench.t", schema=parts[0].schema)
            with t.update_spec() as u:
                u.add_identity("grp")
            for p in parts:
                t.append(p)
            t = cat.load_table("bench.t")
        iceberg_meta = t.metadata_location.replace("file://", "")
    except ImportError:
        pass
    return delta, iceberg_meta


def timed(fn, runs):
    out = fn()                       # warm-up (also the result checked for its row count)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return out, statistics.median(times), min(times)


def nrows(x):
    if isinstance(x, am.ColumnSet):
        return len(x.columns[0].to_arrow()) if len(x) else 0
    if hasattr(x, "num_rows"):
        return x.num_rows
    if hasattr(x, "height"):
        return x.height
    return len(x)


def duckdb_connection(ext):
    import duckdb
    con = duckdb.connect()
    con.execute("SET autoinstall_known_extensions = false")
    con.execute("SET autoload_known_extensions = false")
    con.execute("LOAD " + ext)
    return con


def readers(kind, path, scenario):
    """(name, callable or None, note) for each reader of `kind` in `scenario`."""
    cols = ["id", "price"] if scenario == "project" else None
    flt = [("grp", "==", "g3"), ("price", ">", 500.0)] if scenario == "filtered" else None
    out = []
    if kind == "delta":
        out.append(("arrowmetal", lambda: am.read_delta(path, columns=cols, filters=flt), ""))
        out.append(("arrowmetal_table", lambda: am.read_delta_table(path, columns=cols, filters=flt), ""))
        try:
            import deltalake as dl
            out.append(("deltalake", lambda: dl.DeltaTable(path).to_pyarrow_table(columns=cols, filters=flt), ""))
        except ImportError:
            out.append(("deltalake", None, "deltalake not installed"))
    else:
        out.append(("arrowmetal", lambda: am.read_iceberg(path, columns=cols, filters=flt), ""))
        out.append(("arrowmetal_table", lambda: am.read_iceberg_table(path, columns=cols, filters=flt), ""))
        try:
            from pyiceberg.expressions import And, EqualTo, GreaterThan
            from pyiceberg.table import StaticTable

            def pyiceberg_read():
                kw = {}
                if cols:
                    kw["selected_fields"] = tuple(cols)
                if flt:
                    kw["row_filter"] = And(EqualTo("grp", "g3"), GreaterThan("price", 500.0))
                return StaticTable.from_metadata(path).scan(**kw).to_arrow()
            out.append(("pyiceberg", pyiceberg_read, ""))
        except ImportError:
            out.append(("pyiceberg", None, "pyiceberg not installed"))
    try:
        import polars as pl

        def polars_read():
            if kind == "delta":
                lf = pl.scan_delta(path)
            else:
                from pyiceberg.table import StaticTable
                lf = pl.scan_iceberg(StaticTable.from_metadata(path))
            if flt:
                lf = lf.filter((pl.col("grp") == "g3") & (pl.col("price") > 500.0))
            if cols:
                lf = lf.select(cols)
            return lf.collect()
        out.append(("polars", polars_read, ""))
    except ImportError:
        out.append(("polars", None, "polars not installed"))
    try:
        ext = "delta" if kind == "delta" else "iceberg"
        con = duckdb_connection(ext)
        fn = "delta_scan" if kind == "delta" else "iceberg_scan"
        sel = ", ".join(cols) if cols else "*"
        where = " WHERE grp = 'g3' AND price > 500.0" if flt else ""
        sql = "SELECT %s FROM %s('%s')%s" % (sel, fn, path, where)
        def duckdb_read():
            r = con.execute(sql)
            return r.to_arrow_table() if hasattr(r, "to_arrow_table") else r.fetch_arrow_table()
        out.append(("duckdb", duckdb_read, ""))
    except Exception as e:           # not installed, or the extension would have to be downloaded
        out.append(("duckdb", None, "%s extension did not load offline: %s" % (kind, str(e).splitlines()[0][:120])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2_000_000)
    ap.add_argument("--commits", type=int, default=8)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                                                  "lakehouse_%s.csv" % dt.date.today().isoformat()))
    args = ap.parse_args()
    root = args.dir or tempfile.mkdtemp(prefix="am-lakehouse-")
    os.makedirs(root, exist_ok=True)
    try:
        delta, iceberg = build(root, args.rows, args.commits)
        records = []
        for kind, path in [("delta", delta), ("iceberg", iceberg)]:
            if path is None:
                records.append([kind, "all", "", args.rows, "", "", "", 0, "skipped: pyiceberg not installed"])
                continue
            for scenario in ["full", "project", "filtered"]:
                counts = {}
                for name, fn, note in readers(kind, path, scenario):
                    if fn is None:
                        records.append([kind, scenario, name, args.rows, "", "", "", 0, "skipped: " + note])
                        print("%-8s %-9s %-17s skipped (%s)" % (kind, scenario, name, note))
                        continue
                    try:
                        res, med, lo = timed(fn, args.runs)
                    except Exception as e:
                        msg = str(e).splitlines()[0][:160]
                        records.append([kind, scenario, name, args.rows, "", "", "", 0, "error: " + msg])
                        print("%-8s %-9s %-17s error (%s)" % (kind, scenario, name, msg))
                        continue
                    counts[name] = nrows(res)
                    records.append([kind, scenario, name, args.rows, counts[name], "%.2f" % med, "%.2f" % lo,
                                    args.runs, ""])
                    print("%-8s %-9s %-17s %10d rows  median %9.2f ms  min %9.2f ms" %
                          (kind, scenario, name, counts[name], med, lo))
                if len(set(counts.values())) > 1:
                    print("ROW COUNT MISMATCH", kind, scenario, counts)
                    for r in records:
                        if r[0] == kind and r[1] == scenario and r[8] == "":
                            r[8] = "row counts differ across readers: %s" % counts
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["format", "scenario", "reader", "table_rows", "rows_out", "median_ms", "min_ms", "runs", "note"])
            w.writerows(records)
        print("wrote", args.out)
    finally:
        if not args.keep and args.dir is None:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
