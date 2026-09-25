"""The router's table on this machine: `python -m arrowmetal.router calibrate` and `explain`, the JSON
table format shared with Benchmarks/router_table.py, and determinism: a decision depends only on the
operation, the type, the row count and the table, never on a timing taken at call time."""
import json
import os
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pytest

import arrowmetal as am
from arrowmetal import _router_fit

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OPS = ["sum", "min", "max", "compare", "add", "subtract", "multiply", "divide", "filter", "filter_where",
       "group_by_sum"]
SHIPPED_CSV = os.path.join(ROOT, "Benchmarks", "results", "router_check_2026-09-24.csv")


def pairs():
    """200 (op, rows) pairs: every operation at sizes on and around each crossover."""
    shipped = am.router_table()["shipped_crossovers"]
    sizes = {0, 1, 1_000, 100_000, 1_000_000, 10_000_000, 100_000_000}
    for c in shipped.values():
        sizes |= {c - 1, c, c + 1}
    sizes = sorted(sizes)
    out = [(op, n) for n in sizes for op in OPS]
    k = 0
    while len(out) < 200:
        out.append((OPS[k % len(OPS)], 7 * k + 3))
        k += 1
    return out[:200]


def env(**extra):
    e = {k: v for k, v in os.environ.items() if k not in ("ARROWMETAL_ROUTER", "ARROWMETAL_ROUTER_TABLE")}
    e["PYTHONPATH"] = os.path.join(ROOT, "python")
    e.update(extra)
    return e


DECISIONS_CODE = """
import json, sys
sys.path.insert(0, {tests!r})
import arrowmetal as am
from test_router_calibrate import pairs
print(json.dumps([[op, n, *am.route_decision(op, n)] for op, n in pairs()]))
"""


def test_decisions_identical_across_1000_calls():
    ps = pairs()
    assert len(ps) == 200 and len(set(ps)) == 200
    with am.router("auto"):
        first = [am.route_decision(op, n) for op, n in ps]
        for _ in range(1000):
            assert [am.route_decision(op, n) for op, n in ps] == first
    # The decision is the table's comparison: below the crossover the CPU, else the GPU (divide is not routed).
    for (op, n), (path, cross) in zip(ps, first):
        assert path == ("gpu" if op == "divide" or n >= cross else "cpu"), (op, n, path, cross)


def test_decisions_identical_across_processes():
    code = DECISIONS_CODE.format(tests=os.path.join(ROOT, "python", "tests"))
    runs = []
    for _ in range(2):
        out = subprocess.run([sys.executable, "-c", code], env=env(ARROWMETAL_ROUTER="auto", ARROWMETAL_ROUTER_TABLE="shipped"),
                             capture_output=True, text=True, check=True)
        runs.append(json.loads(out.stdout))
    with am.router("auto"):
        here = [[op, n, *am.route_decision(op, n)] for op, n in pairs()]
    assert runs[0] == runs[1] == here


def test_decision_is_what_a_routed_call_records():
    rng = np.random.default_rng(3)
    with am.router("auto"):
        for n in (0, 1, 1000, 100_003):
            col = am.array(pa.array(rng.integers(-5, 5, n, dtype=np.int64)))
            fcol = am.array(pa.array(rng.random(n)))
            mask = col.compare(">", 0)
            for op, call, dtype in [("sum", col.sum, "int64"), ("max", col.max, "int64"),
                                    ("compare", lambda: col.compare(">", 1), "int64"),
                                    ("multiply", lambda: col.arith("*", 3), "int64"),
                                    ("subtract", lambda: col.arith("-", 3), "int64"),
                                    ("filter", lambda: col.filter(mask), "int64"),
                                    ("add", lambda: fcol.arith("+", 1.0), "float64")]:
                call()
                d = am.last_route()
                e = am.explain_route(op, n, dtype)
                assert (d.path, d.reason, d.rows) == (e["path"], e["reason"], n), (op, n)
                assert am.route_decision(op, n, dtype)[0] == d.path


def test_explain_says_where_the_decision_came_from():
    out = subprocess.run([sys.executable, "-m", "arrowmetal.router", "explain", "sum", "150,000", "--nulls", "0.3"],
                         env=env(ARROWMETAL_ROUTER_TABLE="shipped"), capture_output=True, text=True, check=True).stdout
    cross = am.router_table()["shipped_crossovers"]["sum"]
    assert "sum over 150,000 int64 rows: mode auto -> " + ("cpu" if 150_000 < cross else "gpu") in out
    assert f"crossover {cross:,} rows" in out and "fitted between" in out
    assert "table: shipped" in out and "router_check_2026-09-24.csv" in out
    assert "null fraction 0.3" in out
    e = json.loads(subprocess.run([sys.executable, "-m", "arrowmetal.router", "explain", "min", "10", "--dtype", "float32",
                                   "--json"], env=env(ARROWMETAL_ROUTER="cpu", ARROWMETAL_ROUTER_TABLE="shipped"),
                                  capture_output=True, text=True, check=True).stdout)
    assert (e["mode"], e["path"]) == ("cpu", "gpu") and e["reason"].startswith("no alternative")
    bad = subprocess.run([sys.executable, "-m", "arrowmetal.router", "explain", "sort", "10"], env=env(),
                         capture_output=True, text=True)
    assert bad.returncode == 2 and "unknown operation sort" in bad.stderr
    with pytest.raises(ValueError, match="unknown dtype"):
        am.explain_route("sum", 10, "decimal")


def test_json_table_from_router_table_py_loads_as_the_shipped_values(tmp_path):
    """router_table.py --json-out writes the run-time format; loaded, it gives the shipped crossovers."""
    out = tmp_path / "t.json"
    subprocess.run([sys.executable, os.path.join(ROOT, "Benchmarks", "router_table.py"), "--from-check", SHIPPED_CSV,
                    "--out", str(tmp_path / "RouterTable.swift"), "--json-out", str(out)], check=True, capture_output=True)
    table = json.loads(out.read_text())
    assert table["format"] == _router_fit.FORMAT
    assert table["machine"]["chip"] and table["machine"]["cpu_cores"] > 0
    shipped = am.router_crossovers()
    try:
        am.load_router_table(str(out))
        info = am.router_table()
        assert info["shipped"] is False and info["path"] == str(out) and info["shipped_ops"] == []
        assert am.router_crossovers() == shipped
        with pytest.raises(am.ArrowMetalError):
            am.load_router_table(str(tmp_path / "missing.json"))
        assert am.router_table()["path"] == str(out)                    # unchanged after a failed load
    finally:
        am.load_router_table(None)
    assert am.router_table()["shipped"] is True


def test_unreadable_table_leaves_the_shipped_one():
    code = "import json, arrowmetal as am; print(json.dumps(am.router_table()))"
    info = json.loads(subprocess.run([sys.executable, "-c", code], env=env(ARROWMETAL_ROUTER_TABLE="/nonexistent/t.json"),
                                     capture_output=True, text=True, check=True).stdout)
    assert info["shipped"] is True and "/nonexistent/t.json" in info["load_error"]


def test_calibrate_quick_writes_this_machines_table_and_new_processes_load_it(tmp_path):
    e = env(HOME=str(tmp_path))
    out = subprocess.run([sys.executable, "-m", "arrowmetal.router", "calibrate", "--quick", "--csv", str(tmp_path / "c.csv")],
                         env=e, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert "quick grid" in out.stdout
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    path = tmp_path / ".arrowmetal" / "router" / (_router_fit.chip_id(chip) + ".json")
    assert f"wrote {path}" in out.stdout
    t = json.loads(path.read_text())
    assert t["format"] == _router_fit.FORMAT and t["grid"]["name"] == "quick"
    assert t["machine"]["chip"] == chip and t["machine"]["cpu_cores"] == os.cpu_count()
    assert t["machine"]["metal_device"] == am.device_name()
    assert t["arrowmetal_version"] == am.__version__ and t["date"]
    assert set(t["crossovers"]) == {"sum", "min", "max", "compare", "arithmetic", "filter", "group_by_sum", "multiply"}
    for op, row in t["crossovers"].items():
        if row["crossover_rows"] is not None:
            assert row["bracket_low_rows"] <= row["crossover_rows"] <= row["step_rows"], op
    # The CSV is the router check format: router_table.py fits the same crossovers from it.
    header, bench, sizes = _router_fit.load_check(str(tmp_path / "c.csv"))
    assert "quick grid" in header
    # A new process on this machine loads it at startup.
    code = "import json, arrowmetal as am; print(json.dumps([am.router_table(), am.router_crossovers()]))"
    info, cross = json.loads(subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True,
                                            check=True).stdout)
    assert info["shipped"] is False and info["path"] == str(path) and info["grid"] == "quick"
    for op, row in t["crossovers"].items():
        assert cross[op] == (row["crossover_rows"] or info["shipped_crossovers"][op]), op
    ex = subprocess.run([sys.executable, "-m", "arrowmetal.router", "explain", "filter", "1000"], env=e,
                        capture_output=True, text=True, check=True).stdout
    assert f"table: this machine's ({path})" in ex
    # ARROWMETAL_ROUTER_TABLE=shipped keeps the shipped table.
    info = json.loads(subprocess.run([sys.executable, "-c", code], env=dict(e, ARROWMETAL_ROUTER_TABLE="shipped"),
                                     capture_output=True, text=True, check=True).stdout)[0]
    assert info["shipped"] is True


def test_bench_calibrate_flag(tmp_path):
    out = subprocess.run([sys.executable, "-m", "arrowmetal.bench", "--rows", "200000", "--quiet", "--no-share",
                          "--no-polars", "--calibrate"], env=env(HOME=str(tmp_path)), capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0, out.stderr
    line = next(ln for ln in out.stdout.splitlines() if ln.startswith("Router calibration:"))
    assert "quick grid" in line
    written = list((tmp_path / ".arrowmetal" / "router").glob("*.json"))
    assert len(written) == 1 and str(written[0]) in line


def test_fit_matches_router_table_py():
    """calibrate and router_table.py share one fit: the JSON form of the shipped CSV gives the Swift literal's rows."""
    header, bench, sizes = _router_fit.load_check(SHIPPED_CSV)
    table = _router_fit.table_json(bench, sizes, header, "x")
    shipped = am.router_table()["shipped_crossovers"]
    assert {op: e["crossover_rows"] for op, e in table["crossovers"].items()} == shipped
    # A sweep where the GPU never catches up leaves that row to the shipped table.
    few = {k: v for k, v in bench.items() if k[1] <= 100_000}
    t = _router_fit.table_json(few, [n for n in sizes if n <= 100_000], header, "x")
    assert t["crossovers"]["compare"]["crossover_rows"] is None and "not ahead" in t["crossovers"]["compare"]["not_reached"]
    assert _router_fit.crossover_for(t, "compare", shipped) == (shipped["compare"], "shipped")
