"""`python -m arrowmetal.bench`: the 30-second CPU-against-Metal report, run as a subprocess."""
import json
import os
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from arrowmetal import bench
from arrowmetal.bench import OPS

ROWS = "200000"


def _run(args, code=None):
    env = dict(os.environ)
    cmd = [sys.executable, "-c", code] if code else [sys.executable, "-m", "arrowmetal.bench", *args]
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


def test_json_report_matches_and_times_every_operation():
    out = _run(["--rows", ROWS, "--json"])
    assert out.returncode == 0, out.stderr
    result = json.loads(out.stdout)
    assert result["match"] is True
    assert result["problems"] == []
    assert result["rows"] == int(ROWS)
    assert list(result["timings"]) == OPS
    for op in OPS:
        t = result["timings"][op]
        for lib in ("pyarrow", "arrowmetal"):
            assert t[lib]["wall_ms"] > 0, (op, lib)
            assert t[lib]["cpu_ms"] >= 0, (op, lib)
            assert t[lib]["iterations"] >= 2, (op, lib)
        assert t["speedup"] > 0
    assert result["import_ms"] > 0
    assert result["machine"]["gpu"]
    assert result["versions"]["arrowmetal"]


def test_quiet_prints_the_table_only_and_runs_without_polars():
    # Polars hidden from the import system: the table has no Polars column and the run still passes.
    code = ("import sys; sys.modules['polars'] = None; from arrowmetal.bench import main; "
            f"sys.exit(main(['--rows', '{ROWS}', '--quiet']))")
    out = _run([], code=code)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.rstrip("\n").splitlines()
    assert len(lines) == 1 + len(OPS)
    assert lines[0].split()[0] == "op"
    assert "pyarrow ms (cpu-ms)" in lines[0] and "ArrowMetal ms (cpu-ms)" in lines[0]
    assert "Polars" not in out.stdout
    for line, op in zip(lines[1:], OPS):
        assert line.startswith(op)
        assert f"{int(ROWS):,}" in line
        assert line.rstrip().endswith("x")
    assert "Share it" not in out.stdout and "results match" not in out.stdout


# ---- --parquet: the user's own file

def _write(tmp_path, columns, name="data.parquet", **kw):
    path = tmp_path / name
    pq.write_table(pa.table(columns), path, **kw)
    return str(path)


def _mixed(n=20_000, seed=7):
    rng = np.random.default_rng(seed)
    return {
        "secret_small": pa.array(rng.integers(0, 1_000, n), pa.int32()),
        "secret_amount": pa.array(rng.standard_normal(n) * 100.0, pa.float64()),
        "secret_const": pa.array(np.full(n, 7), pa.int8()),
        "secret_store": pa.array(rng.integers(0, 3, n), pa.int16()),
        "secret_region": pa.array(rng.choice(["north", "south"], n)),
        "secret_flag": pa.array(rng.random(n) < 0.5),
        "secret_nested": pa.array([{"x": int(i)} for i in range(n)]),
    }


def test_parquet_value_column_is_the_largest_numeric_one(tmp_path):
    info = bench.inspect_parquet(_write(tmp_path, _mixed()))
    kinds = [c["kind"] for c in info["columns"]]
    assert kinds == ["numeric", "numeric", "numeric", "numeric", "string", None, None]
    assert bench.choose_value_column(info) == "secret_amount"          # float64: 8 bytes a row


def test_parquet_value_column_ties_go_to_more_encoded_bytes_then_file_order(tmp_path):
    n = 10_000
    rng = np.random.default_rng(3)
    info = bench.inspect_parquet(_write(tmp_path, {
        "few": pa.array(rng.integers(0, 4, n), pa.int64()),              # dictionary pages: small
        "many": pa.array(rng.integers(0, 2**40, n), pa.int64()),         # plain pages: large
    }))
    assert bench.choose_value_column(info) == "many"
    info = bench.inspect_parquet(_write(tmp_path, {"a": pa.array([1], pa.int32()),
                                                   "b": pa.array([1], pa.int32())}, "t.parquet"))
    assert bench.choose_value_column(info) == "a"
    info = bench.inspect_parquet(_write(tmp_path, {"s": pa.array(["x"])}, "s.parquet"))
    assert bench.choose_value_column(info) is None


def test_parquet_key_column_is_the_lowest_cardinality_integer_or_string(tmp_path):
    path = _write(tmp_path, _mixed())
    info = bench.inspect_parquet(path)
    table = pq.read_table(path, columns=[c["name"] for c in info["columns"] if c["kind"]])
    # region has 2 values, store 3; the constant column (1 value) is passed over, the float is never a key
    assert bench.choose_key_column(table, info, "secret_amount") == ("secret_region", 2)
    no_strings = table.drop_columns(["secret_region"])
    assert bench.choose_key_column(no_strings, info, "secret_amount") == ("secret_store", 3)
    only_const = table.select(["secret_amount", "secret_const"])
    assert bench.choose_key_column(only_const, info, "secret_amount") == ("secret_const", 1)
    assert bench.choose_key_column(table.select(["secret_amount"]), info, "secret_amount") == (None, 0)


def test_parquet_nulls_count_as_one_group(tmp_path):
    path = _write(tmp_path, {"v": pa.array([1.0, 2.0, 3.0, 4.0]),
                             "k": pa.array([1, None, 1, None], pa.int64())})
    info = bench.inspect_parquet(path)
    assert bench.choose_key_column(pq.read_table(path), info, "v") == ("k", 2)


def test_parquet_size_guard_names_the_limit(tmp_path):
    info = bench.inspect_parquet(_write(tmp_path, _mixed()))
    assert bench.size_guard(info, 64 * 2**30) is None
    line = bench.size_guard(info, 256 * 2**10)                          # a 256 KB machine
    assert line.startswith("refusing this file")
    assert "the limit on this Mac, with 0.0 GB of memory, is 0.0 GB" in line
    # Through the command: exit status 2, the line on stderr, nothing measured.
    path = _write(tmp_path, _mixed(), "g.parquet")
    code = ("import sys, arrowmetal.bench as b; b.physical_memory = lambda: 256 * 1024; "
            f"sys.exit(b.main(['--parquet', {path!r}]))")
    out = _run([], code=code)
    assert out.returncode == 2, out.stderr
    assert "refusing this file" in out.stderr and "the limit on this Mac" in out.stderr
    assert out.stdout == ""


def test_parquet_without_a_supported_column_says_so(tmp_path):
    path = _write(tmp_path, {"flag": pa.array([True, False]),
                             "when": pa.array([1, 2], pa.timestamp("us")),
                             "items": pa.array([[1], [2, 3]])})
    out = _run(["--parquet", path])
    assert out.returncode == 2
    assert "no supported column in this file" in out.stderr
    assert "bool (1)" in out.stderr and "timestamp[us] (1)" in out.stderr
    assert out.stdout == ""


def test_both_modes_run_without_numpy(tmp_path):
    # pyarrow and Polars install without NumPy, and so does the wheel.
    path = _write(tmp_path, {"v": pa.array([float(i) for i in range(5_000)]),
                             "k": pa.array([i % 3 for i in range(5_000)], pa.int32())})
    hide = "import sys; sys.modules['numpy'] = None; from arrowmetal.bench import main; "
    out = _run([], code=hide + f"sys.exit(main(['--parquet', {path!r}, '--quiet']))")
    assert out.returncode == 0, out.stderr
    assert "sum float64 by int32 key (3 groups)" in out.stdout
    out = _run([], code=hide + f"sys.exit(main(['--rows', '{ROWS}', '--json']))")
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["match"] is True


def test_generated_dataset_has_the_documented_shape():
    n = 200_000
    d = bench.make_data(n)
    v, f, k = d["v"], d["f"], d["k"]
    assert (v.type, f.type, k.type) == (pa.int64(), pa.float64(), pa.int32())
    assert len(v) == len(f) == len(k) == n
    assert abs(v.null_count / n - 0.10) < 0.005
    mm = pc.min_max(v).as_py()
    assert -1_000_000 <= mm["min"] < -990_000 and 990_000 < mm["max"] < 1_000_000
    assert abs(pc.mean(f).as_py()) < 0.01 and abs(pc.stddev(f).as_py() - 1.0) < 0.01
    assert not pc.any(pc.is_nan(f)).as_py()
    assert pc.count_distinct(k).as_py() == 1_000
    assert pc.min_max(k).as_py() == {"min": 0, "max": 999}
    # seeded: the same rows every run, and the SplitMix64 stream matches a plain-Python reference
    again = bench.make_data(n)
    assert v.equals(again["v"]) and f.equals(again["f"]) and k.equals(again["k"])
    mask = (1 << 64) - 1
    start = bench._mix64((bench.SEED * 0x100 + 3) & mask)
    want = [bench._mix64((start + i * bench._GOLDEN) & mask) for i in range(1, 9)]
    assert bench._random_u64(8, bench.SEED, 3).to_pylist() == want


def test_parquet_string_only_file_measures_the_read_only(tmp_path):
    path = _write(tmp_path, {"s": pa.array(["a", "b", None, "c"] * 1000)})
    out = _run(["--parquet", path, "--json"])
    assert out.returncode == 0, out.stderr
    result = json.loads(out.stdout)
    assert list(result["timings"]) == ["read 1 of 1 columns"]
    assert any("no numeric column" in n for n in result["notes"])


def test_parquet_report_matches_and_shares_no_path_or_names(tmp_path):
    path = _write(tmp_path, _mixed(), "secret_file.parquet", compression="snappy", row_group_size=5_000)
    out = _run(["--parquet", path, "--json"])
    assert out.returncode == 0, out.stderr
    result = json.loads(out.stdout)
    assert result["match"] is True, result["problems"]
    f = result["file"]
    assert (f["rows"], f["columns"], f["columns_read"], f["numeric_columns"], f["string_columns"],
            f["codecs"], f["row_groups"]) == (20_000, 7, 5, 4, 1, ["SNAPPY"], 4)
    assert list(result["timings"]) == ["read 5 of 7 columns", "sum float64", "filter float64 > median",
                                       "sum float64 by string key (2 groups)"]
    for op, t in result["timings"].items():
        for lib in ("pyarrow", "arrowmetal"):
            assert t[lib]["wall_ms"] > 0 and t[lib]["iterations"] >= 2, (op, lib)

    text = _run(["--parquet", path])
    assert text.returncode == 0, text.stderr
    assert "Share it" in text.stdout and "codec SNAPPY" in text.stdout
    assert "&share=" in text.stdout and "20%2C000%20rows" in text.stdout
    assert "secret" not in text.stdout and str(tmp_path) not in text.stdout
