"""`python -m arrowmetal.bench`: the 30-second CPU-against-Metal report, run as a subprocess."""
import json
import os
import subprocess
import sys

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
