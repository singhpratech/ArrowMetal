"""docs/COVERAGE.md's per-function table and docs/ARROW_FUNCTIONS.md's matrix section cannot drift.

Both pages are rendered from the per-cell record of a differential run
(docs/data/differential_coverage.json, written by `python/tests/coverage_report.py`). These tests check
that every matrix operation is assigned to Arrow functions or listed as outside them, that the record
has exactly the matrix's cells, that the committed pages are the rendering of the record, and that a
seeded sample of cells run now gives the recorded counts. `differential_report.py` compares every full
run with the record as well (exit code 3 when they differ).
"""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import coverage_report as cr                                          # noqa: E402
import test_differential as diff                                      # noqa: E402
from arrowmetal import functions as F                                 # noqa: E402

#: How many (operation, type) cells the sample reruns, and the seed that picks them.
SAMPLE_CELLS = 24
SAMPLE_SEED = 20261001


def _record():
    return cr.load_record()


def test_every_matrix_operation_is_assigned_exactly_once():
    ops = [o.name for o in diff.OPS]
    assigned = set(cr.OPERATION_FUNCTIONS) | set(cr.OUTSIDE)
    assert not set(cr.OPERATION_FUNCTIONS) & set(cr.OUTSIDE)
    unassigned = [o for o in ops if o not in assigned]
    assert not unassigned, (f"new matrix operations {unassigned}: add them to OPERATION_FUNCTIONS (or OUTSIDE) "
                            "in python/tests/coverage_report.py, then run it")
    stale = sorted(assigned - set(ops))
    assert not stale, f"coverage_report.py names operations the matrix no longer has: {stale}"


def test_every_assigned_name_is_an_arrow_function_name():
    names = set(F.list_functions())
    for op_name, functions in cr.OPERATION_FUNCTIONS.items():
        assert functions, op_name
        assert len(set(functions)) == len(functions), op_name
        unknown = [n for n in functions if n not in names]
        assert not unknown, f"{op_name}: {unknown} are not Arrow function names in arrowmetal.functions"


def test_record_has_exactly_the_matrix_cells():
    if cr.default_matrix_problem():
        pytest.skip(cr.default_matrix_problem())
    record = _record()
    assert record["fields"] == list(cr.FIELDS)
    expected = {(o.name, t) for o in diff.OPS for t in o.types}
    recorded = {(r[0], r[1]) for r in record["cells"]}
    assert recorded == expected, (
        f"the matrix changed: {sorted(expected - recorded)[:10]} new, {sorted(recorded - expected)[:10]} gone; "
        f"regenerate with {cr.COMMAND}")
    assert len(record["cells"]) == len(recorded)
    for row in record["cells"]:
        assert row[2] == len(diff.SHAPES), (row, "datasets per cell changed; regenerate the record")


def test_record_counts_add_up_and_nothing_is_unclassified():
    record = _record()
    for op_name, type_name, cases, passed, documented, unclassified, skipped in record["cells"]:
        assert passed + documented + unclassified + skipped == cases, (op_name, type_name)
        assert unclassified == 0, (op_name, type_name)
    t = cr.totals(record)
    assert t[0] == sum(r[2] for r in record["cells"])
    assert t[0] == t[1] + t[2] + t[3] + t[4]
    # Every function row sums whole operations, so a row never exceeds the matrix.
    for name, entry in cr.per_function(record).items():
        assert 0 < entry["counts"][0] <= t[0], name
        assert entry["counts"][0] == sum(cr.per_operation(record)[o][0] for o in entry["ops"]), name


def test_documented_cells_belong_to_open_findings():
    """A documented count can only sit on an operation that some open finding names."""
    named = set().union(*(f.ops for f in diff.FINDINGS))
    for op_name, type_name, _cases, _p, documented, _u, _s in _record()["cells"]:
        if documented:
            assert op_name in named, (op_name, type_name)


def test_committed_pages_are_the_rendering_of_the_record():
    for path, content in cr.render_pages(_record()).items():
        with open(path) as fh:
            on_disk = fh.read()
        assert on_disk == content, (f"{os.path.relpath(path, cr.ROOT)} is not the rendering of the record: "
                                    f"run {cr.COMMAND} --render (or the full command after a matrix change)")


def test_compare_cells_names_every_difference():
    rows = [["a", "int8", 27, 27, 0, 0, 0], ["b", "utf8", 27, 20, 0, 0, 7]]
    assert cr.compare_cells(rows, [list(r) for r in rows]) == []
    changed = [["a", "int8", 27, 26, 1, 0, 0], ["c", "bool", 27, 27, 0, 0, 0]]
    out = cr.compare_cells(changed, rows)
    assert any("a / int8" in line and "pass 27 -> 26" in line for line in out)
    assert any("b / utf8" in line and "not in this run" in line for line in out)
    assert any("c / bool" in line and "not in the record" in line for line in out)


def test_a_seeded_sample_of_cells_matches_a_fresh_run():
    """The full comparison is differential_report.py's (every gate run); this reruns a sample."""
    if cr.default_matrix_problem():
        pytest.skip(cr.default_matrix_problem())
    import differential_report
    if differential_report.VIRTUAL_DEVICE:
        pytest.skip("the record is from a real GPU; a virtual Metal device cannot build every kernel")
    record = _record()
    rows = sorted(record["cells"])
    picked = random.Random(SAMPLE_SEED).sample(rows, SAMPLE_CELLS)
    cells, _ops, _types, _total, _elapsed = differential_report.run(
        quiet=True, cell_filter={(r[0], r[1]) for r in picked})
    fresh = cr.cells_from_report(cells)
    assert cr.compare_cells(fresh, picked) == []
