"""The CPU/GPU router from Python: both paths give byte-identical Arrow output and match pyarrow,
the decision is visible (`am.last_route()`), and the overrides work. See docs/DESIGN.md."""
import os
import re
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import arrowmetal as am

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIZES = [0, 1, 31, 32, 33, 64, 65, 1000, 4099, 100_003]
INT_TYPES = [pa.int8(), pa.uint8(), pa.int16(), pa.uint16(), pa.int32(), pa.uint32(), pa.int64(), pa.uint64()]


def column(typ, n, nulls, seed):
    rng = np.random.default_rng(seed)
    if pa.types.is_floating(typ):
        vals = rng.uniform(-1e6, 1e6, n).astype(typ.to_pandas_dtype())
    else:
        info = np.iinfo(typ.to_pandas_dtype())
        vals = rng.integers(info.min, info.max, n, dtype=typ.to_pandas_dtype(), endpoint=True)
    mask = rng.random(n) < nulls if nulls else None
    return pa.array(vals, type=typ, mask=mask)


def on(path, fn):
    """Runs fn under a per-thread router override and checks it ran where it was sent."""
    with am.router(path):
        out = fn()
        d = am.last_route()
    assert d is not None and d.path == path, d
    return out


def raw(arr):
    """The Arrow-visible bytes of an exported array: length, null count, validity bits, value bytes
    of every slot (nulls included: both paths compute them)."""
    a = arr.to_arrow() if isinstance(arr, am.MetalArray) else arr
    bufs = a.buffers()
    n, off = len(a), a.offset
    validity = None
    if bufs[0] is not None:
        bits = np.unpackbits(np.frombuffer(bufs[0], dtype=np.uint8), bitorder="little")
        validity = bits[off:off + n].tobytes()
    if pa.types.is_boolean(a.type):
        vbits = np.unpackbits(np.frombuffer(bufs[1], dtype=np.uint8), bitorder="little")
        values = vbits[off:off + n].tobytes()
    else:
        w = a.type.bit_width // 8
        values = bytes(memoryview(bufs[1])[off * w:(off + n) * w])
    return n, a.null_count, validity, values


def same(x, y):
    assert raw(x) == raw(y)


@pytest.mark.parametrize("typ", INT_TYPES + [pa.float64()], ids=str)
def test_reductions_both_paths(typ):
    for n in SIZES:
        for nulls in (0.0, 0.1):
            pa_arr = column(typ, n, nulls, n)
            a = am.array(pa_arr)
            for name in ("sum", "min", "max"):
                g = on("gpu", getattr(a, name))
                c = on("cpu", getattr(a, name))
                if isinstance(g, float):
                    assert np.float64(g).tobytes() == np.float64(c).tobytes(), (name, n)
                else:
                    assert g == c, (name, n)
                if n and name != "sum":
                    oracle = pc.min_max(pa_arr)[name].as_py()
                    assert c == oracle, (name, n)
            if pa.types.is_integer(typ):
                oracle = pc.sum(pa_arr, min_count=1)
                expected = None if oracle.as_py() is None else oracle.cast(pa.int64() if pa.types.is_signed_integer(typ) else pa.uint64()).as_py()
                assert on("cpu", a.sum) == expected


@pytest.mark.parametrize("typ", INT_TYPES + [pa.float32(), pa.float64()], ids=str)
def test_compare_filter_arithmetic_both_paths(typ):
    for n in SIZES:
        pa_a, pa_b = column(typ, n, 0.1, n + 1), column(typ, n, 0.2, n + 2)
        a, b = am.array(pa_a), am.array(pa_b)
        scalar = 3 if pa.types.is_integer(typ) else 0.5
        for op, fn in ((">", pc.greater), ("<=", pc.less_equal), ("==", pc.equal), ("!=", pc.not_equal)):
            g = on("gpu", lambda: a.compare(op, scalar))
            c = on("cpu", lambda: a.compare(op, scalar))
            same(g, c)
            assert c.to_arrow().equals(fn(pa_a, pa.scalar(scalar, type=typ)))
            g2, c2 = on("gpu", lambda: a.compare(op, b)), on("cpu", lambda: a.compare(op, b))
            same(g2, c2)
            assert c2.to_arrow().equals(fn(pa_a, pa_b))
        mask_pa = column(pa.int8(), n, 0.2, n + 3)
        mask = am.array(pc.greater(mask_pa, 0))
        g, c = on("gpu", lambda: a.filter(mask)), on("cpu", lambda: a.filter(mask))
        same(g, c)
        assert c.to_arrow().equals(pc.filter(pa_a, pc.greater(mask_pa, 0)))
        if pa.types.is_integer(typ):
            g, c = on("gpu", lambda: a.filter_where(">", scalar)), on("cpu", lambda: a.filter_where(">", scalar))
            same(g, c)
            assert c.to_arrow().equals(pc.filter(pa_a, pc.greater(pa_a, pa.scalar(scalar, type=typ))))
        if typ != pa.float32():
            for op, fn in (("+", pc.add), ("-", pc.subtract), ("*", pc.multiply)):
                g, c = on("gpu", lambda: a.arith(op, scalar)), on("cpu", lambda: a.arith(op, scalar))
                same(g, c)
                assert c.to_arrow().equals(fn(pa_a, pa.scalar(scalar, type=typ)))
                g2, c2 = on("gpu", lambda: a.arith(op, b)), on("cpu", lambda: a.arith(op, b))
                same(g2, c2)
                assert c2.to_arrow().equals(fn(pa_a, pa_b))


def test_group_by_sum_both_paths():
    for n in (0, 1, 65, 1000, 100_003):
        rng = np.random.default_rng(n)
        keys = pa.array(rng.integers(0, 1000, n, dtype=np.int32), mask=rng.random(n) < 0.05)
        vals = column(pa.int64(), n, 0.1, n + 9)
        k, v = am.array(keys), am.array(vals)
        gb = am.GroupBy(k, 1000)
        g, c = on("gpu", lambda: gb.sum(v)), on("cpu", lambda: gb.sum(v))
        same(g, c)
        assert am.last_route().op == "group_by_sum"
        # Oracle: pyarrow's hash_sum, keyed by position (a key with no valid value is null).
        t = pa.table({"k": keys, "v": vals}).group_by("k").aggregate([("v", "sum")])
        expected = [None] * 1000
        for key, s in zip(t["k"].to_pylist(), t["v_sum"].to_pylist()):
            if key is not None:
                expected[key] = s
        assert c.to_arrow().to_pylist() == expected


def test_last_route_and_reasons():
    small = am.array(pa.array([1, 2, None, 4], type=pa.int64()))
    with am.router("auto"):
        assert small.sum() == 7
        d = am.last_route()
        assert (d.op, d.path, d.rows) == ("sum", "cpu", 4)
        assert d.reason == f"below the {am.router_crossovers()['sum']}-row crossover"
        am.array(pa.array([1.0, 2.0])).max()
        assert am.last_route().reason == "no measured crossover for float64"
        small.arith("*", 2)
        assert am.last_route().reason == "no measured crossover for multiply"
        am.array(pa.array([1.0, 2.0], type=pa.float32())).min()
        assert am.last_route().path == "gpu" and am.last_route().reason.startswith("no alternative")
        with am.batch():
            small.compare(">", 1)
            assert am.last_route().reason == "batch open"
    with am.router("cpu"):
        small.max()
        assert am.last_route().reason == "forced cpu by the per-thread override"
    am.clear_last_route()
    assert am.last_route() is None


def test_auto_at_scale_takes_the_gpu():
    n = am.router_crossovers()["filter"]
    a = am.array(pa.array(np.arange(n, dtype=np.int64)))
    with am.router("auto"):
        out = a.filter_where(">", n - 3)
    assert am.last_route().path == "gpu" and am.last_route().rows == n
    assert out.to_arrow().to_pylist() == [n - 2, n - 1]


def test_modes():
    saved = am.get_router()
    try:
        am.set_router("cpu")
        assert am.get_router() == "cpu"
        am.array(pa.array([1, 2], type=pa.int32())).sum()
        assert am.last_route().reason == "forced cpu by the process mode"
        with am.router("gpu"):
            with am.router("auto"):
                pass
            am.array(pa.array([1, 2], type=pa.int32())).sum()
            assert am.last_route().path == "gpu"
        with pytest.raises(ValueError):
            am.set_router("fastest")
        with pytest.raises(ValueError):
            am.router("sometimes")
    finally:
        am.set_router(saved)


@pytest.mark.parametrize("value", ["gpu", "cpu", "auto", "CPU"])
def test_environment_variable(value):
    env = dict(os.environ, ARROWMETAL_ROUTER=value, PYTHONPATH=os.path.join(ROOT, "python"))
    out = subprocess.run([sys.executable, "-c", "import arrowmetal as am; print(am.get_router())"],
                         env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == value.lower()


def test_table_matches_the_results_file():
    """The shipped table is what Benchmarks/router_table.py generates from the results JSON."""
    subprocess.run([sys.executable, os.path.join(ROOT, "Benchmarks", "router_table.py"), "--check"], check=True)
    src = open(os.path.join(ROOT, "Sources", "ArrowMetal", "Router", "RouterTable.swift")).read()
    body = src.split("static func crossoverRows", 1)[1].split("static func", 1)[0]
    table = dict(re.findall(r"case \.(\w+): return (\d+)", body))
    names = {"groupBySum": "group_by_sum"}
    assert {names.get(k, k): int(v) for k, v in table.items()} == am.router_crossovers()


@pytest.mark.parametrize("value", [None, "cpu"])
def test_differential_report_pins_the_gpu(value):
    """differential_report.py is a plain script, so conftest.py's pin does not reach it; it pins the
    GPU itself (its datasets are below every crossover) and ARROWMETAL_ROUTER still wins."""
    env = {k: v for k, v in os.environ.items() if k != "ARROWMETAL_ROUTER"}
    env["PYTHONPATH"] = os.path.join(ROOT, "python")
    if value:
        env["ARROWMETAL_ROUTER"] = value
    tests = os.path.join(ROOT, "python", "tests")
    code = ("import sys; sys.path.insert(0, %r); import differential_report, arrowmetal as am; "
            "print(am.get_router())" % tests)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == (value or "gpu")


def test_group_by_sum_uint64_is_routed():
    """uint64 values reach GroupBy.sumUnsigned through the C ABI; it is routed like the signed sum."""
    for n in (0, 1, 65, 1000, 100_003):
        rng = np.random.default_rng(n + 5)
        keys = pa.array(rng.integers(0, 100, n, dtype=np.int32), mask=rng.random(n) < 0.05)
        vals = column(pa.uint64(), n, 0.1, n + 11)
        k, v = am.array(keys), am.array(vals)
        gb = am.GroupBy(k, 100)
        am.clear_last_route()
        g, c = on("gpu", lambda: gb.sum(v)), on("cpu", lambda: gb.sum(v))
        same(g, c)
        assert am.last_route().op == "group_by_sum"
        assert c.to_arrow().type == pa.uint64()
        t = pa.table({"k": keys, "v": vals}).group_by("k").aggregate([("v", "sum")])
        expected = [None] * 100
        for key, s in zip(t["k"].to_pylist(), t["v_sum"].to_pylist()):
            if key is not None:
                expected[key] = s
        assert c.to_arrow().to_pylist() == expected


def test_table_says_what_its_cpu_side_measured():
    """The committed table names its source and, in its header, which CPU loops that source timed:
    the 2026-09-17 bench's own loops for the JSON, the shipped RouterCPU loops for a router_check CSV."""
    src = open(os.path.join(ROOT, "Sources", "ArrowMetal", "Router", "RouterTable.swift")).read()
    source = re.search(r'static let source = "([^"]+)"', src).group(1)
    header = src.split("enum RouterTable", 1)[0]
    if source.endswith(".json"):
        assert "Sources/ArrowMetalBench/main.swift" in header and "`cpu-1core`" in header
        assert "not the RouterCPU loops the router" in header
        assert "--from-check" in header
    else:
        assert "shipped RouterCPU loops" in header


def test_table_from_router_check(tmp_path):
    """`router_table.py --from-check` fits the table from a router_check.py run (CPU side: RouterCPU):
    every crossover lies inside the bracket that file measured."""
    import csv
    check = os.path.join(ROOT, "Benchmarks", "results", "router_check_2026-09-23_provisional.csv")
    out = tmp_path / "RouterTable.swift"
    subprocess.run([sys.executable, os.path.join(ROOT, "Benchmarks", "router_table.py"),
                    "--from-check", check, "--out", str(out)], check=True, capture_output=True)
    text = out.read_text()
    assert "shipped RouterCPU loops" in text
    assert 'static let source = "Benchmarks/results/router_check_2026-09-23_provisional.csv"' in text

    def table(name):
        body = text.split(f"static func {name}", 1)[1].split("static func", 1)[0]
        return {k: int(v) for k, v in re.findall(r"case \.(\w+): return (\d+)", body)}
    cross, step, low = table("crossoverRows"), table("measuredStepRows"), table("bracketLowRows")
    labels = {"sum": "sum(int64)", "min": "min(int64)", "max": "max(int64)", "compare": "compare(int64 > 0)",
              "arithmetic": "add(int64, 1)", "filter": "filter(int64, mask)",
              "groupBySum": "group-by sum (1000 keys)"}
    with open(check) as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
    for case, label in labels.items():
        timings = {int(r["rows"]): (float(r["gpu_us"]), float(r["cpu_us"])) for r in rows if r["op"] == label}
        assert low[case] < cross[case] <= step[case], case
        assert timings[low[case]][0] > timings[low[case]][1], case          # CPU ahead at the bracket's low end
        assert all(g <= c for n, (g, c) in timings.items() if n >= step[case]), case
    # --check follows the source the table names.
    subprocess.run([sys.executable, os.path.join(ROOT, "Benchmarks", "router_table.py"), "--check",
                    "--out", str(out)], check=True, capture_output=True)
