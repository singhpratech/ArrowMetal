"""The Share it block of `python -m arrowmetal.bench` and the "Benchmark result" issue form it feeds.

The block is one pasteable piece (header, `key: value` lines, the Markdown result table); these tests
check that it parses back into exactly the fields `.github/ISSUE_TEMPLATE/benchmark_result.yml`
declares, that the prefilled issue link carries those values, and that a real run prints such a block.
"""
import os
import re
import subprocess
import sys
import urllib.parse

import pytest

from arrowmetal import bench

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FORM = os.path.join(ROOT, ".github", "ISSUE_TEMPLATE", "benchmark_result.yml")
FENCE = "`" * 3


def _form():
    """(labels line, field ids) of the issue form. PyYAML is not a dependency, so the two things the
    tests need are read line by line; when PyYAML is installed the same answer is checked against it."""
    with open(FORM) as fh:
        text = fh.read()
    ids = re.findall(r"^\s+id:\s*(\S+)\s*$", text, flags=re.M)
    labels = re.search(r"^labels:\s*(.*)$", text, flags=re.M).group(1)
    try:
        import yaml
    except ImportError:
        return labels, ids
    doc = yaml.safe_load(text)
    assert [b["id"] for b in doc["body"] if "id" in b] == ids
    assert "benchmark result" in doc["labels"]
    return labels, ids


def _result(parquet=False, polars=True, match=True):
    timing = {"wall_ms": 12.5, "cpu_ms": 80.0, "iterations": 5}
    fast = {"wall_ms": 1.25, "cpu_ms": 1.5, "iterations": 5}
    ops = (["read 3 of 4 columns", "sum float64"] if parquet else bench.OPS)
    timings = {}
    for op in ops:
        row = {"pyarrow": timing, "arrowmetal": fast, "fastest_cpu": "pyarrow", "speedup": 10.0}
        if polars:
            row["polars"] = timing
        timings[op] = row
    r = {"machine": {"chip": "Apple M9 Test", "cores": 12, "memory_gb": 32, "macos": "26.1", "gpu": "Apple M9 Test"},
         "versions": {"arrowmetal": "0.5.0", "pyarrow": "25.0.1", "polars": "1.44.1" if polars else None,
                      "python": "3.13.9"},
         "rows": 2_000_000, "timings": timings, "match": match, "problems": [], "import_ms": 3.0,
         "generate_s": 0.5, "total_s": 9.0, "router_table": "shipped (`python -m arrowmetal.bench --calibrate` "
                                                            "writes this Mac's)"}
    if parquet:
        r["file"] = {"rows": 2_000_000, "columns": 4, "columns_read": 3, "numeric_columns": 2, "string_columns": 1,
                     "codecs": ["SNAPPY"], "row_groups": 2, "file_mb": 31.5}
        r["notes"] = []
    return r


def _fenced_block(text):
    """The block between the fences after the "Share it" line of a report."""
    after = text[text.index("Share it"):]
    start = after.index(FENCE + "\n") + len(FENCE) + 1
    return after[start:after.index("\n" + FENCE, start)]


def test_the_form_declares_the_fields_the_block_fills():
    labels, ids = _form()
    assert "benchmark result" in labels
    assert tuple(ids) == bench.FORM_FIELDS
    filled = bench.form_fields(bench.share_block(_result()))
    # Every field but the submitter's own notes comes from the block.
    assert set(filled) == set(bench.FORM_FIELDS) - {"notes"}


def test_the_block_parses_back_into_its_parts():
    r = _result()
    parsed = bench.parse_share_block(bench.share_block(r))
    assert parsed["chip"] == "Apple M9 Test"
    assert parsed["macos"] == "26.1"
    assert parsed["machine"] == "12 CPU cores, 32 GB, Metal device Apple M9 Test"
    assert parsed["versions"] == {"ArrowMetal": "0.5.0", "pyarrow": "25.0.1", "Polars": "1.44.1", "Python": "3.13.9"}
    assert parsed["run"].startswith("python -m arrowmetal.bench, generated data, 2,000,000 rows, router ")
    assert parsed["run"].endswith("results match pyarrow")
    assert parsed["router_table"] == r["router_table"]
    assert "file" not in parsed
    assert [row["op"] for row in parsed["table"]] == bench.OPS
    for row in parsed["table"]:
        assert row["rows"] == "2,000,000"
        assert row["pyarrow ms (cpu-ms)"] == "12.50 (80.0)"
        assert row["ArrowMetal ms (cpu-ms)"] == "1.25 (1.5)"
        assert row["speedup"] == "10.00x"


def test_a_parquet_block_and_one_without_polars_parse_too():
    parsed = bench.parse_share_block(bench.share_block(_result(parquet=True, polars=False)))
    assert parsed["file"].startswith("2,000,000 rows, 4 columns (3 read")
    assert "codec SNAPPY" in parsed["file"]
    assert "--parquet" in parsed["run"]
    assert "Polars" not in parsed["versions"]
    assert [row["op"] for row in parsed["table"]] == ["read 3 of 4 columns", "sum float64"]
    assert "Polars ms (cpu-ms)" not in parsed["table"][0]


def test_the_report_prints_the_block_and_a_link_prefilled_with_the_form_fields():
    r = _result()
    text = bench.report(r)
    block = _fenced_block(text)
    assert block == bench.share_block(r)
    link = next(ln for ln in text.splitlines() if ln.startswith("Open a prefilled issue: "))
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(link.split(": ", 1)[1]).query)
    assert query["template"] == ["benchmark_result.yml"]
    fields = bench.form_fields(block)
    for name, value in fields.items():
        assert query[name] == [value], name
    assert fields["share"] == FENCE + "\n" + block + "\n" + FENCE
    assert fields["chip"] == "Apple M9 Test" and fields["os"] == "macOS 26.1"
    # The pasted form of the block, fences included, parses the same.
    assert bench.parse_share_block(fields["share"]) == bench.parse_share_block(block)


def test_the_parquet_report_block_round_trips():
    r = _result(parquet=True)
    block = _fenced_block(bench.report_parquet(r))
    assert block == bench.share_block(r)
    assert bench.form_fields(block)["chip"] == "Apple M9 Test"


def test_paths_in_the_block_name_the_home_directory_as_a_tilde():
    home = os.path.expanduser("~")
    assert bench._home(os.path.join(home, ".arrowmetal", "router", "apple-m9.json")) == \
        os.path.join("~", ".arrowmetal", "router", "apple-m9.json")
    assert bench._home("/opt/tables/x.json") == "/opt/tables/x.json"
    line = bench.router_table_line(os.path.join(home, ".arrowmetal", "router", "apple-m9.json"))
    assert line.startswith("~") and "written by --calibrate in this run" in line and home not in line


@pytest.mark.parametrize("text", ["", "hello", bench.SHARE_HEADER + "\nchip: x\n",
                                  bench.SHARE_HEADER + "\ncolour: red\n"])
def test_text_that_is_not_a_block_is_refused(text):
    with pytest.raises(ValueError):
        bench.parse_share_block(text)


def test_a_real_run_prints_a_block_with_this_machines_chip_and_router_table(tmp_path):
    env = dict(os.environ, HOME=str(tmp_path))
    env.pop("ARROWMETAL_ROUTER_TABLE", None)
    out = subprocess.run([sys.executable, "-m", "arrowmetal.bench", "--rows", "200000", "--no-polars"],
                         env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    block = _fenced_block(out.stdout)
    parsed = bench.parse_share_block(block)
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    assert parsed["chip"] == chip
    assert parsed["macos"] == bench.machine_info()["macos"]
    assert parsed["router_table"].startswith("shipped")       # an empty HOME holds no calibrated table
    assert [row["op"] for row in parsed["table"]] == bench.OPS
    assert parsed["run"].endswith("results match pyarrow")
