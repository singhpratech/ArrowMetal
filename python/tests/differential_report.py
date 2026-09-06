#!/usr/bin/env python3
"""Run the whole ArrowMetal-vs-pyarrow.compute matrix and print it. Exit code 1 on any failure.

    PYTHONPATH=python python python/tests/differential_report.py
    DIFF_QUICK=1 ...    # drop the 100k row datasets
    DIFF_LARGE=1 ...    # add the 5,000,000 row datasets
    ... --ops sort,argsort --types int32,float64    # narrow the matrix

The cases are the ones in test_differential.py, so the report and `pytest python/tests` agree by
construction. The report exists because the pass/fail shape of ~6000 parametrised tests is easier to
read as a table than as a pytest log: it says which operation on which type diverges, how often, and
what the first divergence looked like.
"""
import argparse
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyarrow as pa                                                  # noqa: E402

import arrowmetal as am                                               # noqa: E402
import test_differential as diff                                      # noqa: E402


class Cell:
    __slots__ = ("passed", "failed", "skipped", "first_failure", "skip_reason", "findings", "new")

    def __init__(self):
        self.passed = self.failed = self.skipped = self.new = 0
        self.first_failure = None
        self.skip_reason = None
        self.findings = set()

    @property
    def total(self):
        return self.passed + self.failed + self.skipped

    def mark(self):
        if self.new:
            return f"NEW {self.new}/{self.total}"
        if self.failed:
            return f"known {self.failed}/{self.total}"
        if self.passed:
            return f"ok {self.passed}" + (f" +{self.skipped}s" if self.skipped else "")
        return f"skip {self.skipped}"


def run(op_filter=None, type_filter=None, quiet=False):
    cells = defaultdict(Cell)
    order_ops, order_types = [], []
    started = time.time()
    total = 0

    for operation, type_name, shape in diff.all_cases():
        if op_filter and operation.name not in op_filter:
            continue
        if type_filter and type_name not in type_filter:
            continue
        if operation.name not in order_ops:
            order_ops.append(operation.name)
        if type_name not in order_types:
            order_types.append(type_name)

        status, detail = diff.run_case(operation, type_name, shape)
        cell = cells[(operation.name, type_name)]
        total += 1
        if status == diff.PASS:
            cell.passed += 1
        elif status == diff.SKIP:
            cell.skipped += 1
            cell.skip_reason = cell.skip_reason or detail
        else:
            cell.failed += 1
            finding = diff.classify(operation.name, type_name, shape)
            if finding is None:
                cell.new += 1
            else:
                cell.findings.add(finding.id)
            if cell.first_failure is None or (finding is None and cell.new == 1):
                cell.first_failure = (shape.id, detail, finding)
        if not quiet and total % 250 == 0:
            print(f"  ... {total} cases, {time.time() - started:.0f}s", file=sys.stderr)

    order_types.sort(key=lambda t: diff.ALL_TYPES.index(t))
    return cells, order_ops, order_types, total, time.time() - started


def render(cells, ops, types, total, elapsed):
    lines = []
    add = lines.append

    add("ArrowMetal vs pyarrow.compute -- differential matrix")
    add(f"arrowmetal {am.__version__} on {am.device_name()} | pyarrow {pa.__version__}")
    add(f"{len(diff.SHAPES)} datasets per (operation, type): sizes "
        f"{', '.join(str(s) for s in diff.sizes())}; null ratios "
        f"{', '.join(f'{r:g}' for r in diff.NULL_RATIOS)}; flavors random, sliced, special")
    add("")

    width = max(len(o) for o in ops) if ops else 10
    col = max(12, max((len(t) for t in types), default=8) + 2)
    add("operation".ljust(width) + " | " + " | ".join(t.center(col) for t in types))
    add("-" * width + "-+-" + "-+-".join("-" * col for _ in types))
    for op_name in ops:
        row = [op_name.ljust(width)]
        for t in types:
            cell = cells.get((op_name, t))
            row.append(("-" if cell is None or cell.total == 0 else cell.mark()).center(col))
        add(" | ".join(row))
    add("")

    passed = sum(c.passed for c in cells.values())
    failed = sum(c.failed for c in cells.values())
    skipped = sum(c.skipped for c in cells.values())
    brand_new = sum(c.new for c in cells.values())
    add(f"total: {total} cases  |  pass {passed}  fail {failed} ({brand_new} unclassified)  "
        f"skip {skipped}  |  {elapsed:.1f}s")
    add("legend: 'ok N' all N datasets agree ('+Ns' = N skipped); 'known n/N' n datasets hit an open "
        "finding below; 'NEW n/N' an unclassified divergence; 'skip N' not implemented for that "
        "type; '-' out of scope")
    add("")

    by_finding = defaultdict(list)
    unclassified = []
    for key, cell in sorted(cells.items()):
        if not cell.failed:
            continue
        if cell.new:
            unclassified.append(key)
        for ident in cell.findings:
            by_finding[ident].append(key)

    if unclassified:
        add(f"UNCLASSIFIED divergences ({len(unclassified)} cells) -- these are not in "
            "docs/EVALUATION.md:")
        add("")
        for op_name, t in unclassified:
            cell = cells[(op_name, t)]
            dataset, detail, _ = cell.first_failure
            add(f"  {op_name} / {t}  ({cell.new} of {cell.total} datasets)")
            add(f"    dataset: {dataset}")
            add(f"    {detail[:300]}")
        add("")

    if by_finding:
        add("open findings (see docs/EVALUATION.md) -- first differing dataset and value:")
        add("")
        for finding in diff.FINDINGS:
            hits = by_finding.get(finding.id)
            if not hits:
                continue
            n = sum(cells[k].failed for k in hits)
            add(f"  [{finding.id}] {finding.title}")
            add(f"    {n} case(s) across {len(hits)} cell(s): "
                f"{', '.join(f'{o}/{t}' for o, t in hits)}")
            op_name, t = hits[0]
            dataset, detail, _ = cells[(op_name, t)].first_failure
            add(f"    example: {op_name} / {t} / {dataset}")
            add(f"             {detail[:280]}")
            add("")

    skips = {}
    for (op_name, t), cell in cells.items():
        if cell.skipped and not cell.passed and not cell.failed and cell.skip_reason:
            skips.setdefault(cell.skip_reason.split(";")[0][:110], []).append(f"{op_name}/{t}")
    if skips:
        add("not implemented (skipped everywhere):")
        for reason, where in sorted(skips.items()):
            add(f"  {reason}")
            add(f"    {', '.join(sorted(where))}")
        add("")

    if diff.ABSENT:
        add("operations this harness is ready for, but MetalArray does not expose yet:")
        add("  " + ", ".join(f"{name}" for name, _ in diff.ABSENT))
        add("")

    add("FAIL" if failed else "PASS")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ops", help="comma-separated operation names to include")
    parser.add_argument("--types", help="comma-separated type names to include")
    parser.add_argument("-q", "--quiet", action="store_true", help="no progress on stderr")
    args = parser.parse_args(argv)

    op_filter = set(args.ops.split(",")) if args.ops else None
    type_filter = set(args.types.split(",")) if args.types else None

    cells, ops, types, total, elapsed = run(op_filter, type_filter, args.quiet)
    print(render(cells, ops, types, total, elapsed))
    return 1 if any(c.failed for c in cells.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
