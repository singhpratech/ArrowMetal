"""Engine-level conformance: MetalEngine against Polars, and the DuckDB rewrite against DuckDB.

`test_differential.py` checks ArrowMetal's kernels against `pyarrow.compute`. This module checks the
two engines built on them against their own hosts' CPU answers, over a generated grid:

* **Polars** -- every shape `MetalEngine` translates (filter, select and `with_columns` expressions,
  slice, sort with and without a limit, group-by with each aggregate in both roles, whole-frame
  aggregates, the four join kinds, `unique`) x every dtype it carries (the eight integer widths,
  Float32/64, Boolean, String, Date, Datetime in ms/us/ns and with a time zone, Duration in
  ms/us/ns, Time) x null patterns (none, sparse, dense, all-null) x sizes (0, 1, 7, 1,000, 100,000
  rows), plus a special-value flavour (integer extremes, NaN, infinities, subnormals, -0.0). Each
  case collects the LazyFrame with `lf.collect()` and with
  `lf.collect(engine=MetalEngine(shapes="all", min_rows=0))` -- the switch that sends every
  translatable subtree to Metal whatever its size -- and compares the two frames.
* **DuckDB** -- every aggregate the optimizer extension rewrites (`sum`, `avg`, `min`, `max`,
  `count(x)`, `count(*)`, none) x the value column types it takes x no key or one key of each kind it
  takes x no filter or a pushed-down filter x the same null patterns and sizes. Each query runs on
  one connection with `SET arrowmetal_rewrite = 'off'` and with `'force'`, and the two answers are
  compared: the same column types, the same values bit for bit.

Every mismatch is classified. A mismatch that matches a `Divergence` below is *documented*: the
divergence names the line of the docs that states it. Anything else is *unclassified*, and
`engine_report.py` exits 1 on it. A case where the engine did not run the plan (Polars folded it
away, the translation declined it, DuckDB's plan kept its own aggregate) is still compared, and is
counted as *not taken*: its answer is the host's own.

    PYTHONPATH=python python python/tests/engine_report.py          # both engines, CSVs, summary
    PYTHONPATH=python python -m pytest python/tests/test_engine_conformance.py -q
"""
import os
import sys
from collections import namedtuple

import numpy as np
import pyarrow as pa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_differential as td                                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

PASS, DOCUMENTED, UNCLASSIFIED, NOT_TAKEN = "pass", "documented", "unclassified", "not_taken"


# ==================================================================================================
# the data grid, shared by both engines

SIZES_FULL = [0, 1, 7, 1000, 100_000]
SIZES_QUICK = [0, 1, 7, 1000]
#: name -> null ratio of the column under test
NULL_PATTERNS = {"none": 0.0, "sparse": 0.05, "dense": 0.7, "all": 1.0}


DataShape = namedtuple("DataShape", "size nulls flavor")


def data_shapes(quick=False):
    """(size, null pattern, flavour) triples. A size-0 frame has one pattern (it is empty); the
    special-value flavour runs at 7 and 1,000 rows with no and sparse nulls."""
    out = []
    for size in (SIZES_QUICK if quick else SIZES_FULL):
        if size == 0:
            out.append(DataShape(0, "none", "random"))
            continue
        for nulls in NULL_PATTERNS:
            out.append(DataShape(size, nulls, "random"))
    for size in (7, 1000):
        for nulls in ("none", "sparse"):
            out.append(DataShape(size, nulls, "special"))
    return out


def shape_id(ds):
    return f"{ds.size}rows-{ds.nulls}-{ds.flavor}"


def _source_array(td_name, ds, seed):
    """A pyarrow array from test_differential's generators."""
    return td.make_array(td_name, td.Shape(ds.size, NULL_PATTERNS[ds.nulls], ds.flavor), seed=seed)


def _low_cardinality(arr, n, seed, pool=12):
    """`arr` resampled from its first `pool` rows: a key column with repeats (and the nulls those
    rows hold)."""
    if n == 0:
        return arr.slice(0, 0)
    rng = np.random.default_rng([seed, n, 7])
    take = rng.integers(0, min(len(arr), pool), n)
    return arr.take(pa.array(take, type=pa.int64()))


def _helper_columns(n, seed):
    """k: Int32 keys 0..8 with 10% nulls; v: Int64 values with 10% nulls; id: the row number."""
    rng = np.random.default_rng([seed, n, 11])
    k = pa.array(rng.integers(0, 9, n).astype(np.int32), mask=(rng.random(n) < 0.1) if n else None,
                 type=pa.int32())
    v = pa.array(rng.integers(-1_000_000, 1_000_000, n), mask=(rng.random(n) < 0.1) if n else None,
                 type=pa.int64())
    ids = pa.array(np.arange(n, dtype=np.int64))
    return k, v, ids


# ==================================================================================================
# documented divergences
#
# A `Divergence` is a mismatch the docs state. `doc` is the file and `anchor` a phrase that appears
# on exactly the line that states it; `doc_line()` resolves the line number when the report is
# written, so the pointer follows edits of the docs. `matches(case, detail)` says whether a failed
# case is this divergence: by engine, shape and dtype, and by a check of the data or of the detail
# where the triple alone is too wide.


class Divergence:
    def __init__(self, ident, engine, title, doc, anchor, shapes=None, dtypes=None, check=None):
        self.id, self.engine, self.title = ident, engine, title
        self.doc, self.anchor = doc, anchor
        self.shapes = set(shapes) if shapes else None
        self.dtypes = set(dtypes) if dtypes else None
        self.check = check

    def matches(self, case, detail):
        if case["engine"] != self.engine:
            return False
        if self.shapes is not None and case["shape"] not in self.shapes:
            return False
        if self.dtypes is not None and case["dtype"] not in self.dtypes:
            return False
        return self.check is None or bool(self.check(case, detail))

    def doc_line(self):
        path = os.path.join(REPO, self.doc)
        try:
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    if self.anchor in line:
                        return f"{self.doc}:{i}"
        except OSError:
            pass
        return f"{self.doc} (anchor not found: {self.anchor!r})"


def classify(divergences, case, detail):
    for d in divergences:
        if d.matches(case, detail):
            return d
    return None
