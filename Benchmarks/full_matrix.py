"""The complete ArrowMetal vs Polars / pyarrow.compute / pandas matrix.

Every operation family the Python package exposes, measured against the fastest idiom of each CPU
library on the same in-process data, at 10M and 50M rows (1M and 10M for string columns).

    PYTHONPATH=python python Benchmarks/full_matrix.py            # the full run
    PYTHONPATH=python python Benchmarks/full_matrix.py --quick    # 1M-row smoke, a few minutes
    PYTHONPATH=python python Benchmarks/full_matrix.py --verify --sizes 1000000   # answers only

Writes `Benchmarks/results/full_matrix_<date>.csv`, `full_matrix_<date>_cores.txt` and
`docs/BENCHMARKS_MATRIX.md`.

Methodology (the same rules as Benchmarks/README.md):
- one warm-up call, then best-of-5 wall time; a call is repeated fewer times (never fewer than
  twice) once the repetitions have used the per-measurement budget, so a six-second pyarrow sort
  does not cost a minute. The iteration count actually used is recorded in the CSV.
- CPU time is the process user+system time delta across the call (`resource.getrusage`), which
  counts every thread the library spawned, so a 16-thread CPU kernel shows ~16x its wall time.
  cpu_ms / wall_ms is therefore the number of cores that row actually used, and it is reported per
  library and idiom in `full_matrix_<date>_cores.txt`.
- Every library is handed the same values. pyarrow gets the Arrow array; Polars a Series built from
  it; pandas an Arrow-backed Series when the column has nulls (the only faithful representation) and
  a numpy-backed one when it has none (pandas' fastest idiom); numpy stands in where pandas has no
  vectorised equivalent. No Python loops anywhere in a baseline.
- **Two idioms per CPU library.** The plain eager idiom above uses about one core on the
  element-wise and whole-column reductions -- that is what `Series.sum()` and `pc.add(...)` do
  however many threads the pool has -- and several on group-by, sort and join, where the library
  reaches its own parallel machinery on its own. So every operation is also measured with the most
  parallel idiom that library offers for the same answer, recorded as its own library row:
  `polars-lazy` (the same expression through `pl.LazyFrame`, collected on the in-memory or the
  streaming engine) and `pyarrow-threaded` (an Acero plan over the same values split into one
  record batch per hardware thread, `to_table(use_threads=True)`, or `pa.Table.group_by` over that
  chunked table). The `note` column names the idiom on every such row. pandas gets a
  `pandas-parallel` row recording why it has none: its kernels are single-threaded by design, and
  its two threaded paths -- numexpr, behind `pd.eval`, and the numba engine with `parallel=True`
  behind `rolling` / `groupby.agg` / `apply` -- are looked up with `importlib.util.find_spec` at
  run time and named in the note, present or absent. The default rows are untouched; the parallel
  rows are additions, and "fastest CPU" in the report is the best of every idiom of every library.
- `--verify` runs every operation once instead of timing it and asserts that each parallel idiom
  returns the same answer as that library's default idiom, within 1e-9 relative for floats (a
  threaded reduction sums in a different order). It reports the comparisons it deliberately does
  not make -- t-digest, whose sketch depends on the partitioning, and pyarrow's threaded grouped
  `list`, whose element order inside a group follows batch arrival -- separately from the rows
  that have no parallel idiom at all.
- Bytes counted are the bytes the operation must touch (input + output), so GB/s is comparable.
- An operation ArrowMetal does not have, or that raises, is recorded as an error row, never skipped.
"""
import re
import argparse, csv, datetime, decimal, gc, importlib.util, math, os, resource, statistics, sys, time

import numpy as np
import pyarrow as pa
import pyarrow.acero as ac
import pyarrow.compute as pc
import polars as pl
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import arrowmetal as am

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SEED = 20260906

# ---------------------------------------------------------------- measurement

ROWS = []           # dicts: family, op, rows, library, wall_ms, cpu_ms, gbs, iters, status, note
_ORDER = []         # (family, op, rows) in first-seen order


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


class Bench:
    def __init__(self, iters, budget):
        self.iters, self.budget = iters, budget

    def run(self, fn):
        """(wall_ms, cpu_ms, iterations) for the best of up to `iters` calls after one warm-up."""
        fn()
        best_w, best_c, total, n = float("inf"), float("inf"), 0.0, 0
        while n < self.iters and (n < 2 or total < self.budget):
            c0 = cpu_seconds()
            t0 = time.perf_counter()
            fn()
            w = time.perf_counter() - t0
            c = cpu_seconds() - c0
            if w < best_w:
                best_w, best_c = w, c
            total += w
            n += 1
        return best_w * 1000.0, best_c * 1000.0, n


BENCH = Bench(5, 1.2)


def record(family, op, rows, library, wall_ms, cpu_ms, gbs, iters, status, note):
    key = (family, op, rows)
    if key not in _ORDER:
        _ORDER.append(key)
    ROWS.append(dict(family=family, op=op, rows=rows, library=library, wall_ms=wall_ms,
                     cpu_ms=cpu_ms, gbs=gbs, iters=iters, status=status, note=note))


def case(family, op, rows, nbytes, impls, notes=None, parallel=None, compare=None):
    """Measure one operation across the libraries, in each library's plain and parallel idiom.

    `impls` maps a library name to a zero-argument callable, or to None when the library has no
    equivalent (recorded as such, not silently dropped). `notes` maps a library name to a string.

    `parallel` maps a *suffixed* library name (`polars-lazy`, `pyarrow-threaded`) to either a
    callable or a `(callable, note)` pair: the same operation, same values, same answer, through
    that library's most parallel idiom. `Par` below builds these. Every library that has a default
    row but no parallel idiom gets a `-parallel` row recording why, so the question is never left
    open. `compare` maps a parallel library name to "unordered" when the two idioms may return the
    same rows in a different order (group-by, unique, value_counts); `--verify` uses it.
    """
    notes = dict(notes or {})
    impls = dict(impls)
    parallel = dict(parallel or {})
    for lib, spec in list(parallel.items()):
        auto = PAR_NOTE.get(lib, "parallel idiom")
        if isinstance(spec, tuple):
            parallel[lib], auto = spec[0], spec[1]
        if lib in notes:                       # an explicit note wins, with the idiom appended
            notes[lib] = f"{notes[lib]} ({auto})"
        else:
            notes[lib] = auto
    # pandas and numpy: no threaded path for anything measured here. Say so rather than leave the
    # row out, so the site can show "pandas: single-threaded, no parallel idiom" from the CSV.
    for lib in ("pandas", "numpy"):
        par = lib + "-parallel"
        if impls.get(lib) is not None and par not in parallel:
            parallel[par] = None
            notes.setdefault(par, NO_PARALLEL[lib])

    if VERIFY:
        verify_case(family, op, rows, impls, parallel, compare or {})
        gc.collect()
        return

    for library, fn in list(impls.items()) + list(parallel.items()):
        note = notes.get(library, "")
        if fn is None:
            record(family, op, rows, library, None, None, None, 0, "no equivalent", note)
            print(f"  {family:<12} {op:<34} {rows:>10,} {library:<17} -- no equivalent")
            continue
        try:
            wall, cpu, iters = BENCH.run(fn)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:220]
            record(family, op, rows, library, None, None, None, 0, "error", msg)
            print(f"  {family:<12} {op:<34} {rows:>10,} {library:<17} !! {msg[:90]}")
            continue
        gbs = (nbytes / (wall / 1000.0) / 1e9) if wall > 0 else None
        record(family, op, rows, library, wall, cpu, gbs, iters, "ok", note)
        print(f"  {family:<12} {op:<34} {rows:>10,} {library:<17} {wall:9.3f} ms "
              f"{cpu:9.1f} cpu-ms {gbs:7.1f} GB/s {cpu / wall if wall else 0:5.1f} cores")
    gc.collect()


# ---------------------------------------------------------------- the parallel idioms
#
# What this exists to establish: a CPU baseline is only "on all cores" if the idiom it was
# written in actually fans out. `pl.Series.sum()` and `pyarrow.compute.add(...)` do not, however
# many threads the pool has. These helpers build, for the same values and the same answer, the
# idiom each library offers that does: a polars LazyFrame collected on the in-memory or streaming
# engine, and an Acero plan over the values split into one record batch per core.

# One record batch per hardware thread, so an Acero plan can hand every worker some rows.
NCHUNK = pa.cpu_count() or os.cpu_count() or 8

PAR_LIBS = ("polars-lazy", "pyarrow-threaded", "pandas-parallel", "numpy-parallel")


def _installed(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _pandas_no_parallel():
    """Why pandas has no parallel row, checked against this interpreter rather than asserted.

    pandas has two threaded paths, and neither is a general idiom: numexpr, which `pd.eval` /
    `DataFrame.eval` use for element-wise arithmetic on large frames, and the numba engine with
    `parallel=True`, which `rolling`, `groupby.agg` / `transform` and `apply` accept. Both are
    optional dependencies; if one is present the note says so, because then the gap is this
    harness' and not pandas'.
    """
    have = [n for n in ("numexpr", "numba") if _installed(n)]
    paths = ("its two threaded paths are numexpr (element-wise arithmetic through pd.eval / "
             "DataFrame.eval) and the numba engine with parallel=True (rolling, groupby.agg / "
             "transform, apply)")
    if not have:
        return ("pandas has no parallel idiom here: its kernels are single-threaded by design and "
                f"{paths}, neither of which is installed in this environment")
    return ("pandas has no parallel idiom recorded for this operation: its kernels are "
            f"single-threaded by design and {paths}; {' and '.join(have)} "
            f"{'is' if len(have) == 1 else 'are'} installed here, so a threaded pandas idiom for "
            "the operations that path covers should be added to this harness")


NO_PARALLEL = {
    "pandas": _pandas_no_parallel(),
    "numpy": "numpy has no parallel idiom: its ufuncs are single-threaded",
}

PAR_NOTE = {
    "polars-lazy": "parallel idiom: pl.LazyFrame(...).collect()",
    "pyarrow-threaded": "parallel idiom: Acero, to_table(use_threads=True)",
}


def chunked(arr, k=NCHUNK):
    """`arr` as a ChunkedArray of k zero-copy slices, so an Acero plan sees k record batches."""
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    n = len(arr)
    if n < k:
        return pa.chunked_array([arr], type=arr.type)
    step = -(-n // k)
    return pa.chunked_array([arr.slice(i, min(step, n - i)) for i in range(0, n, step)],
                            type=arr.type)


def _plan(build):
    """A callable running an Acero plan built on first use, so a bad expression is an error row."""
    box = {}

    def run():
        if "d" not in box:
            box["d"] = build()
        return box["d"].to_table(use_threads=True)
    return run


class Par:
    """The parallel idiom of Polars and pyarrow over one set of columns.

    Columns are pyarrow arrays -- exactly the values the default rows are handed. They are split
    into `NCHUNK` record batches once for the pyarrow side and wrapped in one LazyFrame for the
    polars side; both are built lazily, so a `Par` that is never used costs nothing.

    Each method returns the `parallel=` dict for one `case(...)`: `{library: (callable, note)}`.
    """

    def __init__(self, **cols):
        self._cols = cols
        self._ch = {}
        self._tbls = {}
        self._lf = None

    def _chunked(self, name):
        if name not in self._ch:
            self._ch[name] = chunked(self._cols[name])
        return self._ch[name]

    def table(self, cols=None):
        """The source table, optionally narrowed to `cols`.

        An Acero *filter* node passes its whole input schema through, so a plan whose source
        carries columns the operation does not read materialises them for every surviving row --
        work the eager `pc.filter` row never does. Narrowing the source is what keeps the two
        idioms doing the same amount of work; the chunked arrays are shared, so it is free.
        """
        key = tuple(cols) if cols is not None else tuple(self._cols)
        if key not in self._tbls:
            self._tbls[key] = pa.table({k: self._chunked(k) for k in key})
        return self._tbls[key]

    @property
    def tbl(self):
        return self.table()

    @property
    def lf(self):
        if self._lf is None:
            self._lf = pl.LazyFrame({k: pl.Series(k, v) for k, v in self._cols.items()})
        return self._lf

    def _src(self, cols=None):
        return ac.Declaration("table_source", ac.TableSourceNodeOptions(self.table(cols)))

    # -- polars ------------------------------------------------------------
    def _pl(self, build, streaming, shape):
        """`build(lf)` returns the LazyFrame to collect; `shape` names the idiom for the note."""
        plan = build(self.lf)
        if streaming:
            def fn():
                return plan.collect(engine="streaming")
            tail = 'collect(engine="streaming")'
        else:
            def fn():
                return plan.collect()
            tail = "collect()"
        return fn, (f"parallel idiom: pl.LazyFrame{shape}.{tail}, "
                    f"{'streaming' if streaming else 'in-memory'} engine, "
                    f"{pl.thread_pool_size()} threads")

    @staticmethod
    def _list(e):
        return list(e) if isinstance(e, (list, tuple)) else [e]

    # -- the five shapes ---------------------------------------------------
    def project(self, pl_expr=None, pa_expr=None, streaming=True):
        """Element-wise: polars streaming engine, Acero project node."""
        out = {}
        if pl_expr is not None:
            e = self._list(pl_expr)
            out["polars-lazy"] = self._pl(lambda lf: lf.select(e), streaming, ".select(...)")
        if pa_expr is not None:
            exprs = list(pa_expr) if isinstance(pa_expr, (list, tuple)) else [pa_expr]
            names = [f"r{i}" for i in range(len(exprs))]
            out["pyarrow-threaded"] = (
                _plan(lambda: ac.Declaration.from_sequence(
                    [self._src(), ac.Declaration("project", ac.ProjectNodeOptions(exprs, names))])),
                f"parallel idiom: Acero project node over {NCHUNK} record batches, "
                f"to_table(use_threads=True), {pa.cpu_count()} threads")
        return out

    def reduce(self, pl_expr=None, aggs=None, streaming=False):
        """A whole-column aggregate: polars in-memory lazy engine, Acero aggregate node."""
        out = {}
        if pl_expr is not None:
            e = self._list(pl_expr)
            out["polars-lazy"] = self._pl(lambda lf: lf.select(e), streaming, ".select(...)")
        if aggs is not None:
            spec = [(a[0], a[1], a[2] if len(a) > 2 else None, f"o{i}")
                    for i, a in enumerate(aggs)]
            out["pyarrow-threaded"] = (
                _plan(lambda: ac.Declaration.from_sequence(
                    [self._src(),
                     ac.Declaration("aggregate", ac.AggregateNodeOptions(spec))])),
                f"parallel idiom: Acero aggregate node ({', '.join(a[1] for a in aggs)}) over "
                f"{NCHUNK} record batches, to_table(use_threads=True), {pa.cpu_count()} threads")
        return out

    def group(self, keys, pl_expr=None, aggs=None, streaming=False):
        """A grouped aggregate: polars lazy group_by, pa.Table.group_by over a chunked table.

        The default pyarrow group-by rows already call `pa.Table.group_by`, which is Acero with
        `use_threads=True`, and Acero's table_source node slices even a single chunk into several
        ExecBatches -- so those rows are already partly threaded (a median of about 6 cores in the
        cores table, against 1.0 for the eager element-wise and reduction rows). Handing it one
        chunk per hardware thread simply gives it more to hand out: measured on this machine, a
        one-chunk group-by keeps about 3.8 cores busy and a `NCHUNK`-chunk one about 9.1.
        """
        out = {}
        if pl_expr is not None:
            e = self._list(pl_expr)
            out["polars-lazy"] = self._pl(lambda lf: lf.group_by(keys).agg(e), streaming,
                                          f".group_by({keys}).agg(...)")
        if aggs is not None:
            def fn2(_k=keys, _a=list(aggs)):
                return pa.TableGroupBy(self.tbl, _k, use_threads=True).aggregate(_a)
            out["pyarrow-threaded"] = (
                fn2, f"parallel idiom: pa.Table.group_by({keys}, use_threads=True) over a "
                     f"{NCHUNK}-chunk table, {pa.cpu_count()} threads")
        return out

    def filter_(self, pl_expr=None, pa_pred=None, pa_out=None, streaming=True, cols=None):
        """A filter: polars streaming engine, Acero filter node feeding a projection.

        `cols` names the columns the pyarrow plan may see; pass exactly the ones the predicate and
        the output read. Acero's filter node emits its whole input schema, so anything else in the
        source would be materialised for every surviving row and the parallel row would be doing
        strictly more work than the `pc.filter` row it is compared with. (polars needs no such
        care: its optimiser pushes the projection below the filter.)
        """
        out = {}
        if pl_expr is not None:
            keep, sel = pl_expr
            e = self._list(sel)
            out["polars-lazy"] = self._pl(lambda lf: lf.filter(keep).select(e), streaming,
                                          ".filter(...).select(...)")
        if pa_pred is not None:
            outs = list(pa_out) if isinstance(pa_out, (list, tuple)) else [pa_out]
            names = [f"r{i}" for i in range(len(outs))]
            width = len(cols) if cols is not None else len(self._cols)
            out["pyarrow-threaded"] = (
                _plan(lambda: ac.Declaration.from_sequence(
                    [self._src(cols),
                     ac.Declaration("filter", ac.FilterNodeOptions(pa_pred)),
                     ac.Declaration("project", ac.ProjectNodeOptions(outs, names))])),
                f"parallel idiom: Acero filter + project nodes over {NCHUNK} record batches of "
                f"the {width} column(s) this operation reads, to_table(use_threads=True), "
                f"{pa.cpu_count()} threads")
        return out

    def polars(self, build, streaming, shape):
        """An arbitrary lazy plan over these columns, for a chain that no shape above covers."""
        return {"polars-lazy": self._pl(build, streaming, shape)}

    def acero(self, nodes, what, cols=None):
        """An arbitrary Acero plan over these columns; `nodes()` returns the nodes after the source
        (called on first use, so a bad expression becomes an error row rather than an abort).
        `cols` narrows the source the way `filter_` does, for a plan that starts with a filter."""
        return {"pyarrow-threaded": (
            _plan(lambda: ac.Declaration.from_sequence([self._src(cols)] + list(nodes()))),
            f"parallel idiom: Acero {what} over {NCHUNK} record batches of "
            f"{len(cols) if cols is not None else len(self._cols)} column(s), "
            f"to_table(use_threads=True), {pa.cpu_count()} threads")}

    def vector(self, pl_expr=None, streaming=False, pa_sort=None):
        """A whole-column vector function (sort, rank, cumulative): polars lazy only, unless an
        Acero order_by node answers it."""
        out = {}
        if pl_expr is not None:
            e = self._list(pl_expr)
            out["polars-lazy"] = self._pl(lambda lf: lf.select(e), streaming, ".select(...)")
        if pa_sort is not None:
            out["pyarrow-threaded"] = (
                _plan(lambda: ac.Declaration.from_sequence(
                    [self._src(), ac.Declaration("order_by", ac.OrderByNodeOptions(pa_sort))])),
                f"parallel idiom: Acero order_by node over {NCHUNK} record batches, "
                f"to_table(use_threads=True), {pa.cpu_count()} threads")
        return out


def par_none(reason, libs=("polars-lazy", "pyarrow-threaded")):
    """Record, for one operation, that a library's only idiom is the eager one already measured."""
    return {lib: (None, reason) for lib in libs}


# A grouped result's rows have no defined order, so --verify sorts both sides before comparing.
UNORDERED = {"polars-lazy": "unordered", "pyarrow-threaded": "unordered"}


# ---------------------------------------------------------------- answer verification

VERIFY = False
VERIFY_STATS = {"pass": 0, "fail": 0, "no_idiom": 0, "skipped": 0, "error": 0}
VERIFY_SKIPPED = []      # (family, op, library): comparisons deliberately not made, and why
VERIFY_FAILURES = []


def _flat(x, out):
    """Append every output column of one benchmark result to `out` as a numpy array."""
    if x is None:
        return
    if isinstance(x, tuple):
        for e in x:
            _flat(e, out)
        return
    if isinstance(x, pl.DataFrame):
        for name in x.columns:
            _flat(x[name], out)
        return
    if isinstance(x, pl.Series):
        _flat(x.to_arrow(), out)
        return
    if isinstance(x, (pa.Table, pa.RecordBatch)):
        for col in x.columns:
            _flat(col, out)
        return
    if isinstance(x, pa.ChunkedArray):
        x = x.combine_chunks()
    if isinstance(x, pa.StructArray):
        for i in range(x.type.num_fields):
            _flat(x.field(i), out)
        return
    if isinstance(x, pa.StructScalar):
        for k in x.keys():
            _flat(x[k], out)
        return
    if isinstance(x, pa.Scalar):
        out.append(np.array([x.as_py()], dtype=object))
        return
    if isinstance(x, pa.Array):
        out.append(np.asarray(x.to_pylist(), dtype=object))
        return
    if isinstance(x, (pd.DataFrame,)):
        for name in x.columns:
            _flat(x[name], out)
        return
    if isinstance(x, pd.Series):
        out.append(np.asarray(x.to_list(), dtype=object))
        return
    if isinstance(x, np.ndarray):
        out.append(x)
        return
    if isinstance(x, list):
        out.append(np.asarray(x, dtype=object))
        return
    out.append(np.array([x], dtype=object))


def _cols(x):
    out = []
    _flat(x, out)
    return out


def _eq(a, b, rtol=1e-9):
    """Elementwise equality of two result columns; floats within a relative tolerance, because a
    threaded reduction adds in a different order."""
    if len(a) != len(b):
        return f"length {len(a)} vs {len(b)}"
    for i, (x, y) in enumerate(zip(a.tolist(), b.tolist())):
        if x is None or y is None:
            if x is not y:
                return f"row {i}: {x!r} vs {y!r}"
            continue
        if isinstance(x, float) or isinstance(y, float):
            try:
                fx, fy = float(x), float(y)
            except (TypeError, ValueError):
                return f"row {i}: {x!r} vs {y!r}"
            if math.isnan(fx) and math.isnan(fy):
                continue
            if not math.isclose(fx, fy, rel_tol=rtol, abs_tol=1e-12):
                return f"row {i}: {fx!r} vs {fy!r}"
            continue
        if x != y:
            return f"row {i}: {x!r} vs {y!r}"
    return None


def _reorder(cols):
    """Sort a result's rows lexicographically, for an operation with no defined row order
    (group-by, unique, value_counts): the same rows in a different order are the same answer."""
    if not cols:
        return cols
    lists = [c.tolist() for c in cols]
    # Sort on the exact-valued columns only (the group keys); a float aggregate can differ in its
    # last bit between a threaded and a single-threaded reduction and must not decide the order.
    keyed = [l for l in lists
             if all(v is None or isinstance(v, (int, str, bool, bytes)) for v in l)] or lists
    idx = sorted(range(len(lists[0])), key=lambda i: tuple(repr(l[i]) for l in keyed))
    return [c[np.asarray(idx, dtype=np.intp)] for c in cols]


def verify_case(family, op, rows, impls, parallel, compare):
    """Run each idiom once and assert every parallel row answers exactly what its default row does."""
    base = {}
    for lib, fn in parallel.items():
        if fn is None:                       # this library has no parallel idiom for this row
            VERIFY_STATS["no_idiom"] += 1
            continue
        root = lib.split("-")[0]
        try:
            if root not in base:
                bf = impls.get(root)
                if bf is None:
                    base[root] = ("missing", None)
                else:
                    base[root] = ("ok", _cols(bf()))
            state, want = base[root]
            if state != "ok":
                print(f"  SKIP {family}/{op} {lib}: no default {root} row to compare against")
                VERIFY_STATS["no_idiom"] += 1
                continue
            got = _cols(fn())
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:200]
            print(f"  ERR  {family}/{op} {lib}: {msg}")
            VERIFY_FAILURES.append((family, op, rows, lib, msg))
            VERIFY_STATS["error"] += 1
            continue
        mode = compare.get(lib, "exact")
        if mode == "skip":                   # a real answer, deliberately not compared: see the note
            VERIFY_STATS["skipped"] += 1
            VERIFY_SKIPPED.append((family, op, lib))
            continue
        if mode == "unordered":
            want, got = _reorder(want), _reorder(got)
        if len(want) != len(got):
            why = f"{len(want)} output columns vs {len(got)}"
        else:
            why = next((w for w in (_eq(a, b) for a, b in zip(want, got)) if w), None)
        if why is None:
            VERIFY_STATS["pass"] += 1
        else:
            print(f"  FAIL {family}/{op} {lib} vs {root}: {why}")
            VERIFY_FAILURES.append((family, op, rows, lib, why))
            VERIFY_STATS["fail"] += 1
    if VERIFY_STATS["pass"] % 25 == 0:
        print(f"  .. {family:<12} {op:<34} verified "
              f"({VERIFY_STATS['pass']} ok, {VERIFY_STATS['fail'] + VERIFY_STATS['error']} bad)")


# ---------------------------------------------------------------- data

class Col:
    """One column, materialised per library only when a benchmark asks for it."""

    def __init__(self, arrow, numpy_=None, has_nulls=None):
        self.a = arrow
        self._np = numpy_
        self._has_nulls = arrow.null_count > 0 if has_nulls is None else has_nulls
        self._g = self._p = self._d = None

    @property
    def g(self):                       # ArrowMetal (device memory)
        if self._g is None:
            self._g = am.array(self.a)
        return self._g

    @property
    def p(self):                       # polars
        if self._p is None:
            self._p = pl.Series("x", self.a)
        return self._p

    @property
    def d(self):                       # pandas
        if self._d is None:
            if self._np is not None and not self._has_nulls:
                self._d = pd.Series(self._np)
            else:
                self._d = pd.Series(pd.arrays.ArrowExtensionArray(self.a))
        return self._d

    @property
    def n(self):                       # numpy
        if self._np is None:
            self._np = self.a.to_numpy(zero_copy_only=False)
        return self._np

    @property
    def nbytes(self):
        return self.a.nbytes


def masked(values, rng, null_fraction=0.10):
    if null_fraction <= 0:
        return pa.array(values)
    return pa.array(values, mask=rng.random(len(values)) < null_fraction)


class Data:
    """The columns for one row count. Everything is lazy; drop the object to free the memory."""

    def __init__(self, n):
        self.n = n
        self.rng = np.random.default_rng(SEED)
        self._c = {}

    def __call__(self, name):
        if name not in self._c:
            self._c[name] = getattr(self, "_build_" + name)()
        return self._c[name]

    # -- integers
    def _build_i64(self):
        v = self.rng.integers(-1000, 1001, size=self.n, dtype=np.int64)
        return Col(masked(v, self.rng), v)

    def _build_i64b(self):
        v = self.rng.integers(1, 1001, size=self.n, dtype=np.int64)
        return Col(pa.array(v), v)

    def _build_i64_nn(self):
        v = self.rng.integers(-(2 ** 62), 2 ** 62, size=self.n, dtype=np.int64)
        return Col(pa.array(v), v)

    # -- floats
    def _build_f64(self):
        v = self.rng.random(self.n) * 1000.0
        return Col(masked(v, self.rng), v)

    def _build_f64_nn(self):
        v = self.rng.random(self.n) * 2e9 - 1e9
        return Col(pa.array(v), v)

    def _build_f64b(self):
        v = self.rng.random(self.n) * 1000.0 + 1.0
        return Col(pa.array(v), v)

    def _build_f32(self):
        v = (self.rng.random(self.n, dtype=np.float32) * 2 - 1).astype(np.float32)
        return Col(pa.array(v), v)

    # -- booleans
    def _build_bool(self):
        v = self.rng.random(self.n) < 0.5
        return Col(masked(v, self.rng), v)

    def _build_bool_nn(self):
        v = self.rng.random(self.n) < 0.5
        return Col(pa.array(v), v)

    def _build_mask30(self):
        v = self.rng.random(self.n) < 0.30
        return Col(pa.array(v), v)

    def _build_mask90(self):
        v = self.rng.random(self.n) < 0.90
        return Col(pa.array(v), v)

    # -- selection
    def _build_idx(self):
        v = self.rng.integers(0, self.n, size=self.n // 2, dtype=np.int32)
        return Col(pa.array(v), v)

    # -- temporal: microsecond timestamps across ~20 years
    def _build_ts(self):
        base = 1_100_000_000_000_000
        v = base + self.rng.integers(0, 631_000_000_000_000, size=self.n, dtype=np.int64)
        return Col(pa.array(v).cast(pa.timestamp("us")), v)

    def _build_ts2(self):
        base = 1_400_000_000_000_000
        v = base + self.rng.integers(0, 631_000_000_000_000, size=self.n, dtype=np.int64)
        return Col(pa.array(v).cast(pa.timestamp("us")), v)

    # -- decimal128(18, 4), built straight from the unscaled 128-bit values
    def _build_dec(self):
        v = self.rng.integers(0, 10 ** 10, size=self.n, dtype=np.int64)
        return Col(decimal_array(v), None, has_nulls=False)

    def _build_dec2(self):
        v = self.rng.integers(0, 10 ** 8, size=self.n, dtype=np.int64)
        return Col(decimal_array(v), None, has_nulls=False)

    # -- group-by keys
    def keys(self, distinct):
        name = f"keys_{distinct}"
        if name not in self._c:
            v = self.rng.integers(0, min(distinct, self.n), size=self.n, dtype=np.int32)
            self._c[name] = Col(pa.array(v), v)
        return self._c[name]

    def str_keys(self, distinct):
        """utf8 keys of `distinct` cardinality, built straight into Arrow buffers (12 bytes each)."""
        name = f"str_keys_{distinct}"
        if name not in self._c:
            codes = self.rng.integers(0, min(distinct, self.n), size=self.n, dtype=np.int32)
            self._c[name] = Col(fixed_width_utf8(codes), None, has_nulls=False)
        return self._c[name]


def decimal_array(unscaled, precision=18, scale=4):
    """A decimal128 Arrow array from int64 unscaled values (two 64-bit halves, little-endian)."""
    n = len(unscaled)
    buf = np.zeros((n, 2), dtype=np.int64)
    buf[:, 0] = unscaled
    buf[:, 1] = np.where(unscaled < 0, -1, 0)
    return pa.Array.from_buffers(pa.decimal128(precision, scale), n,
                                 [None, pa.py_buffer(buf.reshape(-1))])


def fixed_width_utf8(codes, width=12, lead=b"k"):
    """A utf8 Arrow array of `len(codes)` fixed-width keys, vectorised (no Python string loop)."""
    n = len(codes)
    hexdigits = np.frombuffer(b"0123456789abcdef", dtype=np.uint8)
    body = np.empty((n, width), dtype=np.uint8)
    body[:, 0] = lead[0]
    acc = codes.astype(np.uint64)
    for j in range(width - 1, 0, -1):
        body[:, j] = hexdigits[(acc & 0xF).astype(np.intp)]
        acc >>= 4
    offsets = np.arange(n + 1, dtype=np.int32) * width
    return pa.Array.from_buffers(pa.utf8(), n, [None, pa.py_buffer(offsets),
                                                pa.py_buffer(body.reshape(-1))])


class StrData:
    """utf8 columns: 1000 distinct `cust_NNN_region` values, the shape used everywhere else here."""

    REGIONS = ("north", "south", "east", "west")

    def __init__(self, n, distinct=1000):
        self.n, self.distinct = n, distinct
        self.rng = np.random.default_rng(SEED + 1)
        self.vocab = [f"cust_{i:03d}_{self.REGIONS[i % 4]}" for i in range(distinct)]
        self.codes = self.rng.integers(0, distinct, size=n, dtype=np.int32)
        arr = pa.DictionaryArray.from_arrays(pa.array(self.codes), pa.array(self.vocab)).cast(pa.string())
        self.s = Col(arr, None, has_nulls=False)
        self._c = {}

    def __call__(self, name):
        if name not in self._c:
            self._c[name] = getattr(self, "_build_" + name)()
        return self._c[name]

    def _build_s2(self):
        codes = self.rng.integers(0, self.distinct, size=self.n, dtype=np.int32)
        arr = pa.DictionaryArray.from_arrays(pa.array(codes), pa.array(self.vocab)).cast(pa.string())
        return Col(arr, None, has_nulls=False)

    def _build_padded(self):
        """The same values with leading and trailing spaces, for trim."""
        vocab = pa.array(["  " + v + "  " for v in self.vocab])
        arr = pa.DictionaryArray.from_arrays(pa.array(self.codes), vocab).cast(pa.string())
        return Col(arr, None, has_nulls=False)

    def _build_ints(self):
        v = self.rng.integers(0, 1_000_000, size=self.n, dtype=np.int64)
        return Col(pa.array(v), v)

    def _build_numeric_text(self):
        codes = self.rng.integers(0, 1_000_000, size=self.n, dtype=np.int32)
        digits = np.frombuffer(b"0123456789", dtype=np.uint8)
        width = 8
        body = np.empty((self.n, width), dtype=np.uint8)
        acc = codes.astype(np.uint64)
        for j in range(width - 1, -1, -1):
            body[:, j] = digits[(acc % 10).astype(np.intp)]
            acc //= 10
        offsets = np.arange(self.n + 1, dtype=np.int32) * width
        arr = pa.Array.from_buffers(pa.utf8(), self.n, [None, pa.py_buffer(offsets),
                                                        pa.py_buffer(body.reshape(-1))])
        return Col(arr, None, has_nulls=False)


# ---------------------------------------------------------------- families

NO_ACERO_AGG = "Acero has no aggregate node for this function; pyarrow.compute is the only idiom"


def family_reductions(d, n):
    f = "reductions"
    i, fl, b = d("i64"), d("f64"), d("bool")
    pi, pf, pb = Par(x=i.a), Par(x=fl.a), Par(x=b.a)
    B = n * 8
    case(f, "sum(int64, 10% nulls)", n, B, {
        "arrowmetal": lambda: i.g.sum(),
        "polars": lambda: i.p.sum(),
        "pyarrow": lambda: pc.sum(i.a),
        "pandas": lambda: i.d.sum()},
        parallel=pi.reduce(pl.col("x").sum(), [("x", "sum")]))
    case(f, "mean(int64, 10% nulls)", n, B, {
        "arrowmetal": lambda: i.g.mean(),
        "polars": lambda: i.p.mean(),
        "pyarrow": lambda: pc.mean(i.a),
        "pandas": lambda: i.d.mean()},
        parallel=pi.reduce(pl.col("x").mean(), [("x", "mean")]))
    case(f, "min(int64, 10% nulls)", n, B, {
        "arrowmetal": lambda: i.g.min(),
        "polars": lambda: i.p.min(),
        "pyarrow": lambda: pc.min(i.a),
        "pandas": lambda: i.d.min()},
        parallel=pi.reduce(pl.col("x").min(), [("x", "min")]))
    case(f, "max(int64, 10% nulls)", n, B, {
        "arrowmetal": lambda: i.g.max(),
        "polars": lambda: i.p.max(),
        "pyarrow": lambda: pc.max(i.a),
        "pandas": lambda: i.d.max()},
        parallel=pi.reduce(pl.col("x").max(), [("x", "max")]))
    case(f, "min_max(int64)", n, B, {
        "arrowmetal": lambda: i.g.min_max(),
        "polars": lambda: (i.p.min(), i.p.max()),
        "pyarrow": lambda: pc.min_max(i.a),
        "pandas": lambda: (i.d.min(), i.d.max())},
        notes={"polars": "no single-pass min_max; two reductions",
               "pandas": "no single-pass min_max; two reductions"},
        parallel=pi.reduce([pl.col("x").min().alias("lo"), pl.col("x").max().alias("hi")],
                           [("x", "min_max")]))
    case(f, "variance(float64, ddof=1)", n, B, {
        "arrowmetal": lambda: fl.g.variance(ddof=1),
        "polars": lambda: fl.p.var(),
        "pyarrow": lambda: pc.variance(fl.a, ddof=1),
        "pandas": lambda: fl.d.var()},
        parallel=pf.reduce(pl.col("x").var(), [("x", "variance", pc.VarianceOptions(ddof=1))]))
    case(f, "stddev(float64, ddof=1)", n, B, {
        "arrowmetal": lambda: fl.g.stddev(ddof=1),
        "polars": lambda: fl.p.std(),
        "pyarrow": lambda: pc.stddev(fl.a, ddof=1),
        "pandas": lambda: fl.d.std()},
        parallel=pf.reduce(pl.col("x").std(), [("x", "stddev", pc.VarianceOptions(ddof=1))]))
    case(f, "count_distinct(int64)", n, B, {
        "arrowmetal": lambda: i.g.count_distinct(),
        "polars": lambda: i.p.n_unique(),
        "pyarrow": lambda: pc.count_distinct(i.a),
        "pandas": lambda: i.d.nunique()},
        parallel=pi.reduce(pl.col("x").n_unique(), [("x", "count_distinct")]))
    case(f, "quantile(float64, 0.5)", n, B, {
        "arrowmetal": lambda: fl.g.quantile(0.5),
        "polars": lambda: fl.p.quantile(0.5),
        "pyarrow": lambda: pc.quantile(fl.a, q=0.5),
        "pandas": lambda: fl.d.quantile(0.5)},
        parallel={**pf.reduce(pl.col("x").quantile(0.5)),
                  **par_none(NO_ACERO_AGG, ["pyarrow-threaded"])})
    case(f, "mode(int64)", n, B, {
        "arrowmetal": lambda: i.g.mode(),
        "polars": lambda: i.p.mode(),
        "pyarrow": lambda: pc.mode(i.a),
        "pandas": lambda: i.d.mode()},
        parallel={**pi.reduce(pl.col("x").mode()),
                  **par_none(NO_ACERO_AGG, ["pyarrow-threaded"])},
        compare={"polars-lazy": "unordered"})
    case(f, "product(int64)", n, B, {
        "arrowmetal": lambda: i.g.product(),
        "polars": lambda: i.p.product(),
        "pyarrow": lambda: pc.product(i.a),
        "pandas": lambda: i.d.prod()},
        parallel=pi.reduce(pl.col("x").product(), [("x", "product")]))
    case(f, "any(bool, 10% nulls)", n, n // 8, {
        "arrowmetal": lambda: b.g.any(),
        "polars": lambda: b.p.any(),
        "pyarrow": lambda: pc.any(b.a),
        "pandas": lambda: b.d.any()},
        parallel=pb.reduce(pl.col("x").any(), [("x", "any")]))
    case(f, "all(bool, 10% nulls)", n, n // 8, {
        "arrowmetal": lambda: b.g.all(),
        "polars": lambda: b.p.all(),
        "pyarrow": lambda: pc.all(b.a),
        "pandas": lambda: b.d.all()},
        parallel=pb.reduce(pl.col("x").all(), [("x", "all")]))
    ORDERED = ("Acero refuses an ordered aggregator (first/last) in a threaded plan: "
               "\"Using ordered aggregator in multiple threaded execution is not supported\"")
    case(f, "first(int64)", n, B, {
        "arrowmetal": lambda: i.g.first(),
        "polars": lambda: i.p.drop_nulls().first(),
        "pyarrow": lambda: pc.first(i.a),
        "pandas": lambda: i.d.iloc[i.d.first_valid_index()]},
        notes={"polars": "no skip-null first(); drop_nulls().first()",
               "pandas": "no skip-null first(); iloc[first_valid_index()]"},
        parallel={**pi.reduce(pl.col("x").drop_nulls().first()),
                  **par_none(ORDERED, ["pyarrow-threaded"])})
    case(f, "last(int64)", n, B, {
        "arrowmetal": lambda: i.g.last(),
        "polars": lambda: i.p.drop_nulls().last(),
        "pyarrow": lambda: pc.last(i.a),
        "pandas": lambda: i.d.iloc[i.d.last_valid_index()]},
        parallel={**pi.reduce(pl.col("x").drop_nulls().last()),
                  **par_none(ORDERED, ["pyarrow-threaded"])})
    fl_np = pd.Series(fl.n)          # arrow-backed pandas has no skew/kurtosis kernel
    case(f, "skew(float64)", n, B, {
        "arrowmetal": lambda: fl.g.skew(),
        "polars": lambda: fl.p.skew(bias=True),
        "pyarrow": lambda: pc.skew(fl.a),
        "pandas": lambda: fl_np.skew()},
        notes={"pandas": "numpy-backed Series (arrow-backed pandas has no skew kernel); "
                         "pandas' skew is bias-corrected, the same amount of work"},
        parallel=pf.reduce(pl.col("x").skew(bias=True), [("x", "skew")]))
    case(f, "kurtosis(float64)", n, B, {
        "arrowmetal": lambda: fl.g.kurtosis(),
        "polars": lambda: fl.p.kurtosis(bias=True),
        "pyarrow": lambda: pc.kurtosis(fl.a),
        "pandas": lambda: fl_np.kurt()},
        notes={"pandas": "numpy-backed Series (arrow-backed pandas has no kurtosis kernel)"},
        parallel=pf.reduce(pl.col("x").kurtosis(bias=True), [("x", "kurtosis")]))
    case(f, "tdigest(float64, q=0.5)", n, B, {
        "arrowmetal": lambda: fl.g.tdigest(0.5),
        "polars": None,
        "pyarrow": lambda: pc.tdigest(fl.a, q=0.5),
        "pandas": None},
        notes={"polars": "polars has no t-digest sketch",
               "pandas": "pandas has no t-digest sketch"},
        parallel=pf.reduce(aggs=[("x", "tdigest", pc.TDigestOptions(q=0.5))]),
        compare={"pyarrow-threaded": "skip"})


NO_ACERO_VECTOR = ("not an Acero expression: this is a pyarrow *vector* function, and a project "
                   "node executes scalar expressions only")


def family_elementwise(d, n):
    f = "element-wise"
    i, ib, fl, flb, f32 = d("i64"), d("i64b"), d("f64"), d("f64b"), d("f32")
    cond = d("mask30")
    c2 = d("mask90")
    pi, pfl, pfb, p32 = Par(x=i.a), Par(x=fl.a), Par(x=flb.a), Par(x=f32.a)
    pii = Par(a=i.a, b=ib.a)
    pff = Par(a=fl.a, b=flb.a)
    pw = Par(a=i.a, b=ib.a, c=cond.a, c2=c2.a)
    B = n * 16
    case(f, "add scalar (int64)", n, B, {
        "arrowmetal": lambda: i.g + 1,
        "polars": lambda: i.p + 1,
        "pyarrow": lambda: pc.add(i.a, 1),
        "pandas": lambda: i.d + 1},
        parallel=pi.project(pl.col("x") + 1, pc.add(pc.field("x"), 1)))
    case(f, "multiply scalar (int64)", n, B, {
        "arrowmetal": lambda: i.g * 3,
        "polars": lambda: i.p * 3,
        "pyarrow": lambda: pc.multiply(i.a, 3),
        "pandas": lambda: i.d * 3},
        parallel=pi.project(pl.col("x") * 3, pc.multiply(pc.field("x"), 3)))
    case(f, "add array (int64 + int64)", n, n * 24, {
        "arrowmetal": lambda: i.g + ib.g,
        "polars": lambda: i.p + ib.p,
        "pyarrow": lambda: pc.add(i.a, ib.a),
        "pandas": lambda: i.d + ib.d},
        parallel=pii.project(pl.col("a") + pl.col("b"), pc.add(pc.field("a"), pc.field("b"))))
    case(f, "multiply array (int64 * int64)", n, n * 24, {
        "arrowmetal": lambda: i.g * ib.g,
        "polars": lambda: i.p * ib.p,
        "pyarrow": lambda: pc.multiply(i.a, ib.a),
        "pandas": lambda: i.d * ib.d},
        parallel=pii.project(pl.col("a") * pl.col("b"), pc.multiply(pc.field("a"), pc.field("b"))))
    case(f, "divide (float64 / float64)", n, n * 24, {
        "arrowmetal": lambda: fl.g / flb.g,
        "polars": lambda: fl.p / flb.p,
        "pyarrow": lambda: pc.divide(fl.a, flb.a),
        "pandas": lambda: fl.d / flb.d},
        parallel=pff.project(pl.col("a") / pl.col("b"), pc.divide(pc.field("a"), pc.field("b"))))
    case(f, "power (float32 ** 2)", n, n * 8, {
        "arrowmetal": lambda: f32.g.power(2.0),
        "polars": lambda: f32.p ** 2.0,
        "pyarrow": lambda: pc.power(f32.a, pa.scalar(2.0, pa.float32())),
        "pandas": lambda: f32.d ** 2.0},
        notes={"arrowmetal": "power() is not implemented for float64; float32 column used"},
        parallel=p32.project(pl.col("x") ** 2.0,
                             pc.power(pc.field("x"), pa.scalar(2.0, pa.float32()))))
    case(f, "sqrt (float64)", n, B, {
        "arrowmetal": lambda: flb.g.sqrt(),
        "polars": lambda: flb.p.sqrt(),
        "pyarrow": lambda: pc.sqrt(flb.a),
        "numpy": lambda: np.sqrt(flb.n)},
        notes={"numpy": "pandas has no Series.sqrt(); numpy is the pandas idiom"},
        parallel=pfb.project(pl.col("x").sqrt(), pc.sqrt(pc.field("x"))))
    case(f, "exp (float32)", n, n * 8, {
        "arrowmetal": lambda: f32.g.exp(),
        "polars": lambda: f32.p.exp(),
        "pyarrow": lambda: pc.exp(f32.a),
        "numpy": lambda: np.exp(f32.n)},
        parallel=p32.project(pl.col("x").exp(), pc.exp(pc.field("x"))))
    case(f, "ln (float64)", n, B, {
        "arrowmetal": lambda: flb.g.ln(),
        "polars": lambda: flb.p.log(),
        "pyarrow": lambda: pc.ln(flb.a),
        "numpy": lambda: np.log(flb.n)},
        parallel=pfb.project(pl.col("x").log(), pc.ln(pc.field("x"))))
    case(f, "sin (float64)", n, B, {
        "arrowmetal": lambda: fl.g.sin(),
        "polars": lambda: fl.p.sin(),
        "pyarrow": lambda: pc.sin(fl.a),
        "numpy": lambda: np.sin(fl.n)},
        parallel=pfl.project(pl.col("x").sin(), pc.sin(pc.field("x"))))
    case(f, "round (float64)", n, B, {
        "arrowmetal": lambda: fl.g.round(),
        "polars": lambda: fl.p.round(0),
        "pyarrow": lambda: pc.round(fl.a),
        "pandas": lambda: fl.d.round(0)},
        parallel=pfl.project(pl.col("x").round(0), pc.round(pc.field("x"))))
    case(f, "abs (float64)", n, B, {
        "arrowmetal": lambda: fl.g.abs(),
        "polars": lambda: fl.p.abs(),
        "pyarrow": lambda: pc.abs(fl.a),
        "pandas": lambda: fl.d.abs()},
        parallel=pfl.project(pl.col("x").abs(), pc.abs(pc.field("x"))))
    case(f, "negate (int64)", n, B, {
        "arrowmetal": lambda: i.g.negate(),
        "polars": lambda: -i.p,
        "pyarrow": lambda: pc.negate(i.a),
        "pandas": lambda: -i.d},
        parallel=pi.project(-pl.col("x"), pc.negate(pc.field("x"))))
    case(f, "add_checked (int64 + int64)", n, n * 24, {
        "arrowmetal": lambda: i.g.add_checked(ib.g),
        "polars": None,
        "pyarrow": lambda: pc.add_checked(i.a, ib.a),
        "pandas": None},
        notes={"polars": "polars addition is unchecked (wraps); no checked kernel",
               "pandas": "pandas/numpy addition is unchecked"},
        parallel=pii.project(pa_expr=pc.add_checked(pc.field("a"), pc.field("b"))))
    case(f, "bit_wise_and (int64)", n, n * 24, {
        "arrowmetal": lambda: i.g.bitwise_and(ib.g),
        "polars": lambda: i.p & ib.p,
        "pyarrow": lambda: pc.bit_wise_and(i.a, ib.a),
        "numpy": lambda: i.n & ib.n},
        parallel=pii.project(pl.col("a") & pl.col("b"),
                             pc.bit_wise_and(pc.field("a"), pc.field("b"))))
    case(f, "shift_left (int64 << 2)", n, B, {
        "arrowmetal": lambda: i.g.shift_left(2),
        "polars": None,
        "pyarrow": lambda: pc.shift_left(i.a, 2),
        "numpy": lambda: i.n << 2},
        notes={"polars": "polars Series has no bit-shift operator"},
        parallel=pi.project(pa_expr=pc.shift_left(pc.field("x"), 2)))
    ew = pl.DataFrame({"a": i.p, "b": ib.p, "c": cond.p, "c2": c2.p})
    case(f, "if_else (bool ? int64 : int64)", n, n * 24, {
        "arrowmetal": lambda: cond.g.if_else(i.g, ib.g),
        "polars": lambda: ew.select(pl.when(pl.col("c")).then(pl.col("a")).otherwise(pl.col("b"))),
        "pyarrow": lambda: pc.if_else(cond.a, i.a, ib.a),
        "numpy": lambda: np.where(cond.n, i.n, ib.n)},
        notes={"numpy": "numpy has no null representation; the same values without the validity bitmap"},
        parallel=pw.project(pl.when(pl.col("c")).then(pl.col("a")).otherwise(pl.col("b")),
                            pc.if_else(pc.field("c"), pc.field("a"), pc.field("b"))))
    case(f, "coalesce (2 int64 columns)", n, n * 24, {
        "arrowmetal": lambda: am.coalesce(i.g, ib.g),
        "polars": lambda: ew.select(pl.coalesce(pl.col("a"), pl.col("b"))),
        "pyarrow": lambda: pc.coalesce(i.a, ib.a),
        "pandas": lambda: i.d.fillna(ib.d)},
        parallel=pw.project(pl.coalesce(pl.col("a"), pl.col("b")),
                            pc.coalesce(pc.field("a"), pc.field("b"))))
    case(f, "fill_null (int64)", n, B, {
        "arrowmetal": lambda: i.g.fill_null(0),
        "polars": lambda: i.p.fill_null(0),
        "pyarrow": lambda: pc.fill_null(i.a, 0),
        "pandas": lambda: i.d.fillna(0)},
        parallel=pi.project(pl.col("x").fill_null(0), pc.coalesce(pc.field("x"), 0)))
    case(f, "fill_null_forward (int64)", n, B, {
        "arrowmetal": lambda: i.g.fill_null_forward(),
        "polars": lambda: i.p.fill_null(strategy="forward"),
        "pyarrow": lambda: pc.fill_null_forward(i.a),
        "pandas": lambda: i.d.ffill()},
        parallel={**pi.project(pl.col("x").fill_null(strategy="forward"), streaming=False),
                  **par_none(NO_ACERO_VECTOR, ["pyarrow-threaded"])})
    case(f, "is_nan (float64)", n, n * 9, {
        "arrowmetal": lambda: fl.g.is_nan(),
        "polars": lambda: fl.p.is_nan(),
        "pyarrow": lambda: pc.is_nan(fl.a),
        "numpy": lambda: np.isnan(fl.n)},
        parallel=pfl.project(pl.col("x").is_nan(), pc.is_nan(pc.field("x"))))
    set_np = np.arange(0, 1000, 10, dtype=np.int64)
    set_pa = pa.array(set_np)
    set_g = am.array(set_pa)
    set_pl = pl.Series(set_np)
    case(f, "is_in (int64, 100-value set)", n, n * 9, {
        "arrowmetal": lambda: i.g.is_in(set_g),
        "polars": lambda: i.p.is_in(set_pl),
        "pyarrow": lambda: pc.is_in(i.a, value_set=set_pa),
        "pandas": lambda: i.d.isin(set_np)},
        parallel=pi.project(pl.col("x").is_in(set_pl),
                            pc.is_in(pc.field("x"), value_set=set_pa)))
    case(f, "index_in (int64, 100-value set)", n, n * 12, {
        "arrowmetal": lambda: i.g.index_in(set_g),
        "polars": None,
        "pyarrow": lambda: pc.index_in(i.a, value_set=set_pa),
        "numpy": lambda: np.searchsorted(set_np, i.n)},
        notes={"polars": "no index_in; polars has is_in only",
               "numpy": "searchsorted is the closest vectorised equivalent (no membership check)"},
        parallel=pi.project(pa_expr=pc.index_in(pc.field("x"), value_set=set_pa)))
    case(f, "case_when (2 conditions)", n, n * 40, {
        "arrowmetal": lambda: am.case_when([cond.g, c2.g], [i.g, ib.g], ib.g),
        "polars": lambda: ew.select(pl.when(pl.col("c")).then(pl.col("a"))
                                      .when(pl.col("c2")).then(pl.col("b")).otherwise(pl.col("b"))),
        "pyarrow": lambda: pc.case_when(pc.make_struct(cond.a, c2.a), i.a, ib.a, ib.a),
        "numpy": lambda: np.select([cond.n, c2.n], [i.n, ib.n], ib.n)},
        parallel=pw.project(pl.when(pl.col("c")).then(pl.col("a"))
                              .when(pl.col("c2")).then(pl.col("b")).otherwise(pl.col("b")),
                            pc.case_when(pc.make_struct(pc.field("c"), pc.field("c2")),
                                         pc.field("a"), pc.field("b"), pc.field("b"))))
    case(f, "hash64 (int64)", n, B, {
        "arrowmetal": lambda: i.g.hash64(),
        "polars": lambda: i.p.hash(),
        "pyarrow": None,
        "pandas": lambda: pd.util.hash_array(ib.n)},
        notes={"pyarrow": "pyarrow.compute has no element-wise hash function",
               "pandas": "pd.util.hash_array over the no-null column"},
        parallel=pi.project(pl.col("x").hash()))


def family_select(d, n):
    f = "compare+select"
    i, ib = d("i64"), d("i64b")
    m30, m90, idx = d("mask30"), d("mask90"), d("idx")
    ps = Par(x=i.a, b=ib.a, m30=m30.a, m90=m90.a)
    B = n * 9
    case(f, "compare scalar (int64 > 0)", n, B, {
        "arrowmetal": lambda: i.g > 0,
        "polars": lambda: i.p > 0,
        "pyarrow": lambda: pc.greater(i.a, 0),
        "pandas": lambda: i.d > 0},
        parallel=ps.project(pl.col("x") > 0, pc.greater(pc.field("x"), 0)))
    case(f, "compare array (int64 > int64)", n, n * 17, {
        "arrowmetal": lambda: i.g > ib.g,
        "polars": lambda: i.p > ib.p,
        "pyarrow": lambda: pc.greater(i.a, ib.a),
        "pandas": lambda: i.d > ib.d},
        parallel=ps.project(pl.col("x") > pl.col("b"),
                            pc.greater(pc.field("x"), pc.field("b"))))
    case(f, "filter int64 (30% kept)", n, int(n * 8 * 1.3), {
        "arrowmetal": lambda: i.g.filter(m30.g),
        "polars": lambda: i.p.filter(m30.p),
        "pyarrow": lambda: pc.filter(i.a, m30.a),
        "pandas": lambda: i.d[m30.d]},
        parallel=ps.filter_((pl.col("m30"), pl.col("x")), pc.field("m30"), pc.field("x"),
                            cols=("x", "m30")))
    case(f, "filter int64 (90% kept)", n, int(n * 8 * 1.9), {
        "arrowmetal": lambda: i.g.filter(m90.g),
        "polars": lambda: i.p.filter(m90.p),
        "pyarrow": lambda: pc.filter(i.a, m90.a),
        "pandas": lambda: i.d[m90.d]},
        parallel=ps.filter_((pl.col("m90"), pl.col("x")), pc.field("m90"), pc.field("x"),
                            cols=("x", "m90")))
    case(f, "take (n/2 random indices)", n, n // 2 * 20, {
        "arrowmetal": lambda: i.g.take(idx.g),
        "polars": lambda: i.p.gather(idx.p),
        "pyarrow": lambda: pc.take(i.a, idx.a),
        "pandas": lambda: i.d.take(idx.n)},
        parallel={**ps.vector(pl.col("x").gather(idx.p)),
                  **par_none("no Acero node gathers by index; pc.take is the only idiom",
                             ["pyarrow-threaded"])})
    case(f, "drop_null (int64, 10% nulls)", n, int(n * 8 * 1.9), {
        "arrowmetal": lambda: i.g.drop_null(),
        "polars": lambda: i.p.drop_nulls(),
        "pyarrow": lambda: pc.drop_null(i.a),
        "pandas": lambda: i.d.dropna()},
        parallel=ps.filter_((pl.col("x").is_not_null(), pl.col("x")),
                            pc.is_valid(pc.field("x")), pc.field("x"), cols=("x",)))
    NOTHING = ("a zero-copy view: the call changes an offset and a length, so there is nothing "
               "to spread over cores in any idiom")
    case(f, "slice (zero-copy view)", n, 0, {
        "arrowmetal": lambda: i.g.slice(1000, n - 2000),
        "polars": lambda: i.p.slice(1000, n - 2000),
        "pyarrow": lambda: i.a.slice(1000, n - 2000),
        "pandas": lambda: i.d.iloc[1000:n - 1000]},
        parallel=par_none(NOTHING))
    repl = Col(pa.array(np.zeros(int(m30.n.sum()), dtype=np.int64)))
    zeros_pl = pl.Series(np.zeros(n, dtype=np.int64))
    case(f, "replace_with_mask (30% replaced)", n, n * 17, {
        "arrowmetal": lambda: ib.g.replace_with_mask(m30.g, repl.g),
        "polars": lambda: zeros_pl.zip_with(m30.p, ib.p),
        "pyarrow": lambda: pc.replace_with_mask(ib.a, m30.a, repl.a),
        "pandas": lambda: ib.d.mask(m30.d, 0)},
        notes={"polars": "no replace_with_mask; zip_with against a full zero column",
               "pandas": "Series.mask with a scalar (no positional replacement list)",
               "pyarrow-threaded": "`replace_with_mask` itself is a vector function with no Acero "
                                   "project node, but every replacement here is the same zero, so "
                                   "if_else over the same mask is the byte-identical answer and "
                                   "does have one"},
        parallel={**ps.project(pl.when(pl.col("m30")).then(pl.lit(0, pl.Int64))
                                 .otherwise(pl.col("b")),
                               pc.if_else(pc.field("m30"), pa.scalar(0, pa.int64()),
                                          pc.field("b")))})
    case(f, "indices_nonzero (bool)", n, n * 9, {
        "arrowmetal": lambda: m30.g.indices_nonzero(),
        "polars": lambda: m30.p.arg_true(),
        "pyarrow": lambda: pc.indices_nonzero(m30.a),
        "numpy": lambda: np.flatnonzero(m30.n)},
        parallel={**ps.vector(pl.col("m30").arg_true()),
                  **par_none(NO_ACERO_VECTOR, ["pyarrow-threaded"])})


NO_ACERO_INDICES = ("Acero's order_by node sorts rows; no node returns the sort permutation, so "
                    "pyarrow.compute is the only idiom for an index-producing sort")


def family_sort(d, n, sd):
    f = "sort"
    ii, ff = d("i64_nn"), d("f64_nn")
    pii, pff = Par(x=ii.a), Par(x=ff.a)
    case(f, "argsort int64", n, n * 12, {
        "arrowmetal": lambda: ii.g.argsort(),
        "polars": lambda: ii.p.arg_sort(),
        "pyarrow": lambda: pc.array_sort_indices(ii.a),
        "numpy": lambda: np.argsort(ii.n)},
        notes={"numpy": "pandas argsort delegates to numpy"},
        parallel={**pii.vector(pl.col("x").arg_sort()),
                  **par_none(NO_ACERO_INDICES, ["pyarrow-threaded"])})
    case(f, "argsort float64", n, n * 12, {
        "arrowmetal": lambda: ff.g.argsort(),
        "polars": lambda: ff.p.arg_sort(),
        "pyarrow": lambda: pc.array_sort_indices(ff.a),
        "numpy": lambda: np.argsort(ff.n)},
        parallel={**pff.vector(pl.col("x").arg_sort()),
                  **par_none(NO_ACERO_INDICES, ["pyarrow-threaded"])})
    if sd is not None:
        pstr = Par(x=sd.s.a)
        case(f, "argsort utf8", sd.n, sd.s.nbytes, {
            "arrowmetal": lambda: sd.s.g.argsort(),
            "polars": lambda: sd.s.p.arg_sort(),
            "pyarrow": lambda: pc.array_sort_indices(sd.s.a),
            "pandas": lambda: sd.s.d.argsort()},
            parallel={**pstr.vector(pl.col("x").arg_sort()),
                      **par_none(NO_ACERO_INDICES, ["pyarrow-threaded"])})
        case(f, "sort utf8", sd.n, sd.s.nbytes * 2, {
            "arrowmetal": lambda: sd.s.g.sort(),
            "polars": lambda: sd.s.p.sort(),
            "pyarrow": lambda: pc.take(sd.s.a, pc.array_sort_indices(sd.s.a)),
            "pandas": lambda: sd.s.d.sort_values()},
            parallel=pstr.vector(pl.col("x").sort(), pa_sort=[("x", "ascending")]))
    case(f, "sort float64", n, n * 16, {
        "arrowmetal": lambda: ff.g.sort(),
        "polars": lambda: ff.p.sort(),
        "pyarrow": lambda: pc.take(ff.a, pc.array_sort_indices(ff.a)),
        "numpy": lambda: np.sort(ff.n)},
        parallel=pff.vector(pl.col("x").sort(), pa_sort=[("x", "ascending")]))
    NO_ACERO_TOPK = ("pyarrow exposes no top-k node in Acero; pc.select_k_unstable is the "
                     "only idiom")
    case(f, "top_k (k=100, int64)", n, n * 8, {
        "arrowmetal": lambda: ii.g.top_k(100),
        "polars": lambda: ii.p.top_k(100),
        "pyarrow": lambda: pc.select_k_unstable(ii.a, k=100, sort_keys=[("", "descending")]),
        "numpy": lambda: np.argpartition(ii.n, n - 100)[n - 100:]},
        parallel={**pii.vector(pl.col("x").top_k(100)),
                  **par_none(NO_ACERO_TOPK, ["pyarrow-threaded"])})
    case(f, "top_k (k=10000, int64)", n, n * 8, {
        "arrowmetal": lambda: ii.g.top_k(10000),
        "polars": lambda: ii.p.top_k(10000),
        "pyarrow": lambda: pc.select_k_unstable(ii.a, k=10000, sort_keys=[("", "descending")]),
        "numpy": lambda: np.argpartition(ii.n, n - 10000)[n - 10000:]},
        parallel={**pii.vector(pl.col("x").top_k(10000)),
                  **par_none(NO_ACERO_TOPK, ["pyarrow-threaded"])})
    case(f, "rank (min)", n, n * 16, {
        "arrowmetal": lambda: ii.g.rank(),
        "polars": lambda: ii.p.rank("min"),
        "pyarrow": lambda: pc.rank(ii.a, sort_keys="ascending", tiebreaker="min"),
        "pandas": lambda: ii.d.rank(method="min")},
        parallel={**pii.vector(pl.col("x").rank("min")),
                  **par_none(NO_ACERO_VECTOR, ["pyarrow-threaded"])})
    case(f, "dense_rank", n, n * 16, {
        "arrowmetal": lambda: ii.g.dense_rank(),
        "polars": lambda: ii.p.rank("dense"),
        "pyarrow": lambda: pc.rank(ii.a, sort_keys="ascending", tiebreaker="dense"),
        "pandas": lambda: ii.d.rank(method="dense")},
        parallel={**pii.vector(pl.col("x").rank("dense")),
                  **par_none(NO_ACERO_VECTOR, ["pyarrow-threaded"])})
    k1, k2 = d("keys_lex_a"), d("keys_lex_b")
    tbl_lex = pa.table({"a": k1.a, "b": k2.a})
    df_lex = pl.DataFrame({"a": k1.p, "b": k2.p})
    plex = Par(a=k1.a, b=k2.a)
    case(f, "lexsort (2 int32 keys)", n, n * 16, {
        "arrowmetal": lambda: am.lexsort_indices([k1.g, k2.g]),
        "polars": lambda: df_lex.select(pl.arg_sort_by(["a", "b"])),
        "pyarrow": lambda: pc.sort_indices(tbl_lex, sort_keys=[("a", "ascending"), ("b", "ascending")]),
        "numpy": lambda: np.lexsort((k2.n, k1.n))},
        parallel={**plex.vector(pl.arg_sort_by(["a", "b"])),
                  **par_none(NO_ACERO_INDICES, ["pyarrow-threaded"])})
    case(f, "partition_nth_indices (n/2)", n, n * 12, {
        "arrowmetal": lambda: ii.g.partition_nth_indices(n // 2),
        "polars": None,
        "pyarrow": lambda: pc.partition_nth_indices(ii.a, pivot=n // 2),
        "numpy": lambda: np.argpartition(ii.n, n // 2)},
        notes={"polars": "polars has no partial-partition function"},
        parallel=par_none(NO_ACERO_VECTOR, ["pyarrow-threaded"]))
    keys = d("keys_1000")
    pk = Par(x=keys.a)
    case(f, "unique (int32, 1000 distinct)", n, n * 4, {
        "arrowmetal": lambda: keys.g.unique(),
        "polars": lambda: keys.p.unique(),
        "pyarrow": lambda: pc.unique(keys.a),
        "pandas": lambda: pd.unique(keys.n)},
        parallel={**pk.vector(pl.col("x").unique()),
                  **pk.group(["x"], aggs=[])},
        compare={"polars-lazy": "unordered", "pyarrow-threaded": "unordered"})
    case(f, "value_counts (int32, 1000 distinct)", n, n * 4, {
        "arrowmetal": lambda: keys.g.value_counts(),
        "polars": lambda: keys.p.value_counts(),
        "pyarrow": lambda: pc.value_counts(keys.a),
        "pandas": lambda: keys.d.value_counts()},
        parallel={**pk.vector(pl.col("x").value_counts()),
                  **pk.group(["x"], aggs=[([], "count_all")])},
        compare={"polars-lazy": "unordered", "pyarrow-threaded": "unordered"})
    case(f, "dictionary_encode (int32)", n, n * 8, {
        "arrowmetal": lambda: keys.g.dictionary_encode(),
        "polars": None,
        "pyarrow": lambda: pc.dictionary_encode(keys.a),
        "pandas": lambda: keys.d.astype("category")},
        notes={"polars": "polars has no eager dictionary_encode for an integer Series"},
        parallel=par_none(NO_ACERO_VECTOR, ["pyarrow-threaded"]))
    if sd is not None:
        case(f, "dictionary_encode (utf8)", sd.n, sd.s.nbytes, {
            "arrowmetal": lambda: sd.s.g.dictionary_encode(),
            "polars": lambda: sd.s.p.cast(pl.Categorical),
            "pyarrow": lambda: pc.dictionary_encode(sd.s.a),
            "pandas": lambda: sd.s.d.astype("category")},
            parallel={**pstr.project(pl.col("x").cast(pl.Categorical), streaming=False),
                      **par_none(NO_ACERO_VECTOR, ["pyarrow-threaded"])})


def _lex_builders():
    def a(self):
        v = self.rng.integers(0, 1000, size=self.n, dtype=np.int32)
        return Col(pa.array(v), v)

    def b(self):
        v = self.rng.integers(0, 100000, size=self.n, dtype=np.int32)
        return Col(pa.array(v), v)
    Data._build_keys_lex_a = a
    Data._build_keys_lex_b = b
    Data._build_keys_1000 = lambda self: self.keys(1000)


_lex_builders()


def family_groupby(d, n):
    f = "group-by"
    vals = d("i64")
    for distinct in (1_000, 100_000, 10_000_000):
        if distinct > n:
            continue
        k = d.keys(distinct)
        df = pl.DataFrame({"k": k.p, "x": vals.p})
        tbl = pa.table({"k": k.a, "x": vals.a})
        pdf = pd.DataFrame({"k": k.n, "x": vals.d})
        pg = Par(k=k.a, x=vals.a)
        B = n * 12
        for agg, am_fn, pl_agg, pa_agg, pd_agg in [
            ("sum", "sum", pl.col("x").sum(), "sum", "sum"),
            ("mean", "mean", pl.col("x").mean(), "mean", "mean"),
            ("min", "min", pl.col("x").min(), "min", "min"),
            ("max", "max", pl.col("x").max(), "max", "max"),
            ("count", "count", pl.col("x").count(), "count", "count"),  # noqa: E501
        ]:
            def make_am(name=am_fn, _k=k, _v=vals):
                return lambda: getattr(am.group_by([_k.g]), name)(_v.g)
            case(f, f"{agg} by int32 key ({distinct} groups)", n, B, {
                "arrowmetal": make_am(),
                "polars": (lambda _d=df, _a=pl_agg: _d.group_by("k").agg(_a)),
                "pyarrow": (lambda _t=tbl, _a=pa_agg: _t.group_by("k").aggregate([("x", _a)])),
                "pandas": (lambda _p=pdf, _a=pd_agg: _p.groupby("k", sort=False)["x"].agg(_a)),
            }, parallel=pg.group(["k"], pl_agg, [("x", pa_agg)]), compare=UNORDERED)
        # utf8 keys of the same cardinality
        sk = d.str_keys(distinct)
        df_s = pl.DataFrame({"k": sk.p, "x": vals.p})
        tbl_s = pa.table({"k": sk.a, "x": vals.a})
        pdf_s = pd.DataFrame({"k": sk.d, "x": vals.d})
        pgs = Par(k=sk.a, x=vals.a)
        SB = n * 20
        case(f, f"sum by utf8 key ({distinct} distinct)", n, SB, {
            "arrowmetal": lambda _k=sk: am.group_by([_k.g]).sum(vals.g),
            "polars": lambda _d=df_s: _d.group_by("k").agg(pl.col("x").sum()),
            "pyarrow": lambda _t=tbl_s: _t.group_by("k").aggregate([("x", "sum")]),
            "pandas": lambda _p=pdf_s: _p.groupby("k", sort=False)["x"].sum()},
            parallel=pgs.group(["k"], pl.col("x").sum(), [("x", "sum")]), compare=UNORDERED)
        # two key columns
        side = max(2, int(np.ceil(np.sqrt(distinct))))
        ka = Col(pa.array(d.rng.integers(0, side, size=n, dtype=np.int32)))
        kb = Col(pa.array(d.rng.integers(0, side, size=n, dtype=np.int32)))
        df_2 = pl.DataFrame({"a": ka.p, "b": kb.p, "x": vals.p})
        tbl_2 = pa.table({"a": ka.a, "b": kb.a, "x": vals.a})
        pdf_2 = pd.DataFrame({"a": ka.n, "b": kb.n, "x": vals.d})
        pg2 = Par(a=ka.a, b=kb.a, x=vals.a)
        case(f, f"sum by two int32 keys (~{side * side} groups)", n, n * 16, {
            "arrowmetal": lambda _a=ka, _b=kb: am.group_by([_a.g, _b.g]).sum(vals.g),
            "polars": lambda _d=df_2: _d.group_by(["a", "b"]).agg(pl.col("x").sum()),
            "pyarrow": lambda _t=tbl_2: _t.group_by(["a", "b"]).aggregate([("x", "sum")]),
            "pandas": lambda _p=pdf_2: _p.groupby(["a", "b"], sort=False)["x"].sum()},
            parallel=pg2.group(["a", "b"], pl.col("x").sum(), [("x", "sum")]), compare=UNORDERED)
        del ka, kb, df_2, tbl_2, pdf_2, pg, pgs, pg2
        gc.collect()

    # the remaining grouped aggregates, at every cardinality the sum/min/max rows use — these are the
    # ones the sort-free group-by work is measured by, so they need the same 1k / 100k / 10M sweep.
    fl = d("f64")
    for distinct in (1_000, 100_000, 10_000_000):
        if distinct > n:
            continue
        k = d.keys(distinct)
        df = pl.DataFrame({"k": k.p, "x": fl.p})
        tbl = pa.table({"k": k.a, "x": fl.a})
        pdf = pd.DataFrame({"k": k.n, "x": fl.d})
        pg = Par(k=k.a, x=fl.a)
        B = n * 12
        case(f, f"variance by key ({distinct} groups)", n, B, {
            "arrowmetal": (lambda _k=k: am.group_by([_k.g]).variance(fl.g, ddof=1)),
            "polars": (lambda _d=df: _d.group_by("k").agg(pl.col("x").var())),
            "pyarrow": (lambda _t=tbl: _t.group_by("k").aggregate([("x", "variance")])),
            "pandas": (lambda _p=pdf: _p.groupby("k", sort=False)["x"].var())},
            parallel=pg.group(["k"], pl.col("x").var(), [("x", "variance")]), compare=UNORDERED)
        case(f, f"stddev by key ({distinct} groups)", n, B, {
            "arrowmetal": (lambda _k=k: am.group_by([_k.g]).stddev(fl.g, ddof=1)),
            "polars": (lambda _d=df: _d.group_by("k").agg(pl.col("x").std())),
            "pyarrow": (lambda _t=tbl: _t.group_by("k").aggregate([("x", "stddev")])),
            "pandas": (lambda _p=pdf: _p.groupby("k", sort=False)["x"].std())},
            parallel=pg.group(["k"], pl.col("x").std(), [("x", "stddev")]), compare=UNORDERED)
        case(f, f"min by key, float64 ({distinct} groups)", n, B, {
            "arrowmetal": (lambda _k=k: am.group_by([_k.g]).min(fl.g)),
            "polars": (lambda _d=df: _d.group_by("k").agg(pl.col("x").min())),
            "pyarrow": (lambda _t=tbl: _t.group_by("k").aggregate([("x", "min")])),
            "pandas": (lambda _p=pdf: _p.groupby("k", sort=False)["x"].min())},
            parallel=pg.group(["k"], pl.col("x").min(), [("x", "min")]), compare=UNORDERED)
        case(f, f"max by key, float64 ({distinct} groups)", n, B, {
            "arrowmetal": (lambda _k=k: am.group_by([_k.g]).max(fl.g)),
            "polars": (lambda _d=df: _d.group_by("k").agg(pl.col("x").max())),
            "pyarrow": (lambda _t=tbl: _t.group_by("k").aggregate([("x", "max")])),
            "pandas": (lambda _p=pdf: _p.groupby("k", sort=False)["x"].max())},
            parallel=pg.group(["k"], pl.col("x").max(), [("x", "max")]), compare=UNORDERED)
        case(f, f"count_distinct by key ({distinct} groups)", n, B, {
            "arrowmetal": (lambda _k=k: am.group_by([_k.g]).count_distinct(fl.g)),
            "polars": (lambda _d=df: _d.group_by("k").agg(pl.col("x").n_unique())),
            "pyarrow": (lambda _t=tbl: _t.group_by("k").aggregate([("x", "count_distinct")])),
            "pandas": (lambda _p=pdf: _p.groupby("k", sort=False)["x"].nunique())},
            parallel=pg.group(["k"], pl.col("x").n_unique(), [("x", "count_distinct")]),
            compare=UNORDERED)
        case(f, f"first by key ({distinct} groups)", n, B, {
            "arrowmetal": (lambda _k=k: am.group_by([_k.g]).first(fl.g)),
            "polars": (lambda _d=df: _d.group_by("k").agg(pl.col("x").first())),
            "pyarrow": (lambda _t=tbl: _t.group_by("k", use_threads=False).aggregate([("x", "first")])),
            "pandas": (lambda _p=pdf: _p.groupby("k", sort=False)["x"].first())},
            parallel={**pg.group(["k"], pl.col("x").first()),
                      **par_none("Acero refuses hash_first in a threaded plan: \"Using ordered "
                                 "aggregator in multiple threaded execution is not supported\"; "
                                 "the default row already runs it with use_threads=False",
                                 ["pyarrow-threaded"])},
            compare=UNORDERED)
        case(f, f"list by key ({distinct} groups)", n, n * 20, {
            "arrowmetal": (lambda _k=k: am.group_by([_k.g]).list(fl.g)),
            "polars": (lambda _d=df: _d.group_by("k").agg(pl.col("x"))),
            "pyarrow": (lambda _t=tbl: _t.group_by("k").aggregate([("x", "list")])),
            "pandas": (lambda _p=pdf: _p.groupby("k", sort=False)["x"].apply(list))},
            notes={"pandas": "pandas has no vectorised list aggregation; apply(list) is its idiom",
                   "pyarrow-threaded": "the same lists, but the order of the values inside each "
                                       "group follows batch arrival, so --verify does not compare "
                                       "them element by element"},
            parallel=pg.group(["k"], pl.col("x"), [("x", "list")]),
            compare={"polars-lazy": "unordered", "pyarrow-threaded": "skip"})
        del df, tbl, pdf, pg
        gc.collect()


def family_join(d, n):
    f = "join"
    build_n = max(1000, n // 50)
    rng = np.random.default_rng(SEED + 7)
    right_keys = np.arange(build_n, dtype=np.int64)
    right_vals = rng.integers(0, 1000, size=build_n, dtype=np.int64)
    left_keys = rng.integers(0, build_n * 2, size=n, dtype=np.int64)   # ~50% match
    left_vals = rng.integers(0, 1000, size=n, dtype=np.int64)
    a_rk, a_rv = pa.array(right_keys), pa.array(right_vals)
    a_lk, a_lv = pa.array(left_keys), pa.array(left_vals)
    g_rk, g_rv = am.array(a_rk), am.array(a_rv)
    g_lk, g_lv = am.array(a_lk), am.array(a_lv)
    left_pl = pl.DataFrame({"k": a_lk, "v": a_lv})
    right_pl = pl.DataFrame({"k": a_rk, "w": a_rv})
    left_tb = pa.table({"k": a_lk, "v": a_lv})
    right_tb = pa.table({"k": a_rk, "w": a_rv})
    left_pd = pd.DataFrame({"k": left_keys, "v": left_vals})
    right_pd = pd.DataFrame({"k": right_keys, "w": right_vals})

    # The parallel idioms: the same join through the polars lazy engine, and pyarrow's Acero
    # hashjoin over both sides split into one record batch per core (the default rows join
    # single-chunk tables, which is one batch and so one thread however many the pool has).
    left_ch = pa.table({"k": chunked(a_lk), "v": chunked(a_lv)})
    right_ch = pa.table({"k": chunked(a_rk), "w": chunked(a_rv)})
    left_lz, right_lz = left_pl.lazy(), right_pl.lazy()

    def join_par(how_pl, how_pa):
        return {
            "polars-lazy": (lambda: left_lz.join(right_lz, on="k", how=how_pl).collect(),
                            f"parallel idiom: pl.LazyFrame.join(..., how=\"{how_pl}\").collect(), "
                            f"in-memory engine, {pl.thread_pool_size()} threads"),
            "pyarrow-threaded": (lambda: left_ch.join(right_ch, keys="k", join_type=how_pa,
                                                      use_threads=True),
                                 f"parallel idiom: pa.Table.join(join_type=\"{how_pa}\", "
                                 f"use_threads=True) over {NCHUNK}-chunk tables (Acero hashjoin), "
                                 f"{pa.cpu_count()} threads")}

    def am_join():
        li, ri = am.join(g_lk, g_rk)       # GPU hash join, index pairs
        return g_lv.take(li), g_rv.take(ri)

    case(f, f"inner hash join ({n} x {build_n} on int64)", n, n * 16 + build_n * 16, {
        "arrowmetal": am_join,
        "polars": lambda: left_pl.join(right_pl, on="k", how="inner"),
        "pyarrow": lambda: left_tb.join(right_tb, keys="k", join_type="inner"),
        "pandas": lambda: left_pd.merge(right_pd, on="k", how="inner")},
        notes={"arrowmetal": "am.join index pairs, then one take per output column"},
        parallel=join_par("inner", "inner"), compare=UNORDERED)
    case(f, f"hash join indices ({n} x {build_n} on int64)", n, n * 8 + build_n * 8, {
        "arrowmetal": lambda: am.join(g_lk, g_rk),
        "polars": lambda: left_pl.join(right_pl, on="k", how="inner"),
        "pyarrow": lambda: left_tb.join(right_tb, keys="k", join_type="inner"),
        "pandas": lambda: left_pd.merge(right_pd, on="k", how="inner")},
        notes={"arrowmetal": "the join kernel alone (am_join): the matching index pairs, no payload gather",
               "polars": "no index-only join; the full join is the closest equivalent",
               "pyarrow": "no index-only join; the full join is the closest equivalent",
               "pandas": "no index-only join; the full join is the closest equivalent"},
        parallel=join_par("inner", "inner"), compare=UNORDERED)
    case(f, f"left outer hash join ({n} x {build_n} on int64)", n, n * 8 + build_n * 8, {
        "arrowmetal": lambda: am.join(g_lk, g_rk, how="left"),
        "polars": lambda: left_pl.join(right_pl, on="k", how="left"),
        "pyarrow": lambda: left_tb.join(right_tb, keys="k", join_type="left outer"),
        "pandas": lambda: left_pd.merge(right_pd, on="k", how="left")},
        notes={"arrowmetal": "am_join join_type 1: index pairs with a null right index for an unmatched left row"},
        parallel=join_par("left", "left outer"), compare=UNORDERED)
    del g_rk, g_rv, g_lk, g_lv, left_pl, right_pl, left_tb, right_tb, left_pd, right_pd
    del left_ch, right_ch, left_lz, right_lz
    gc.collect()


def family_strings(sd):
    f = "strings"
    n = sd.n
    s, SB = sd.s, sd.s.nbytes
    s2 = sd("s2")
    padded = sd("padded")
    ints = sd("ints")
    numtext = sd("numeric_text")
    ps = Par(x=s.a, y=s2.a)
    ppad, pint, pnum = Par(x=padded.a), Par(x=ints.a), Par(x=numtext.a)
    case(f, "char_length", n, SB, {
        "arrowmetal": lambda: s.g.char_length(),
        "polars": lambda: s.p.str.len_chars(),
        "pyarrow": lambda: pc.utf8_length(s.a),
        "pandas": lambda: s.d.str.len()},
        parallel=ps.project(pl.col("x").str.len_chars(), pc.utf8_length(pc.field("x"))))
    case(f, "upper", n, SB * 2, {
        "arrowmetal": lambda: s.g.upper(),
        "polars": lambda: s.p.str.to_uppercase(),
        "pyarrow": lambda: pc.utf8_upper(s.a),
        "pandas": lambda: s.d.str.upper()},
        parallel=ps.project(pl.col("x").str.to_uppercase(), pc.utf8_upper(pc.field("x"))))
    case(f, "lower", n, SB * 2, {
        "arrowmetal": lambda: s.g.lower(),
        "polars": lambda: s.p.str.to_lowercase(),
        "pyarrow": lambda: pc.utf8_lower(s.a),
        "pandas": lambda: s.d.str.lower()},
        parallel=ps.project(pl.col("x").str.to_lowercase(), pc.utf8_lower(pc.field("x"))))
    case(f, 'contains("north")', n, SB, {
        "arrowmetal": lambda: s.g.str_contains("north"),
        "polars": lambda: s.p.str.contains("north", literal=True),
        "pyarrow": lambda: pc.match_substring(s.a, "north"),
        "pandas": lambda: s.d.str.contains("north", regex=False)},
        parallel=ps.project(pl.col("x").str.contains("north", literal=True),
                            pc.match_substring(pc.field("x"), "north")))
    case(f, 'starts_with("cust_1")', n, SB, {
        "arrowmetal": lambda: s.g.starts_with("cust_1"),
        "polars": lambda: s.p.str.starts_with("cust_1"),
        "pyarrow": lambda: pc.starts_with(s.a, "cust_1"),
        "pandas": lambda: s.d.str.startswith("cust_1")},
        parallel=ps.project(pl.col("x").str.starts_with("cust_1"),
                            pc.starts_with(pc.field("x"), "cust_1")))
    case(f, 'match_like("cust%") [pure prefix]', n, SB, {
        "arrowmetal": lambda: s.g.match_like("cust%"),
        "polars": lambda: s.p.str.starts_with("cust"),
        "pyarrow": lambda: pc.match_like(s.a, "cust%"),
        "pandas": lambda: s.d.str.match("cust")},
        notes={"arrowmetal": "a pure prefix pattern, which Regex.likePredicate maps onto the GPU "
                             "starts_with kernel",
               "polars": "no SQL LIKE; the equivalent prefix predicate"},
        parallel=ps.project(pl.col("x").str.starts_with("cust"),
                            pc.match_like(pc.field("x"), "cust%")))
    case(f, 'match_like("cust_1%") [_ wildcard]', n, SB, {
        "arrowmetal": lambda: s.g.match_like("cust_1%"),
        "polars": lambda: s.p.str.contains(r"^cust.1"),
        "pyarrow": lambda: pc.match_like(s.a, "cust_1%"),
        "pandas": lambda: s.d.str.match(r"cust.1")},
        notes={"arrowmetal": "`_` is LIKE's single-character wildcard, so this is not a pure prefix "
                             "and takes the ICU host path",
               "polars": "no SQL LIKE; the equivalent anchored regex"},
        parallel=ps.project(pl.col("x").str.contains(r"^cust.1"),
                            pc.match_like(pc.field("x"), "cust_1%")))
    case(f, "match_substring_regex (literal)", n, SB, {
        "arrowmetal": lambda: s.g.match_substring_regex("north"),
        "polars": lambda: s.p.str.contains("north"),
        "pyarrow": lambda: pc.match_substring_regex(s.a, "north"),
        "pandas": lambda: s.d.str.contains("north", regex=True)},
        parallel=ps.project(pl.col("x").str.contains("north"),
                            pc.match_substring_regex(pc.field("x"), "north")))
    RX = r"_[0-9]{2}3_(?:north|west)"
    case(f, "match_substring_regex (real regex)", n, SB, {
        "arrowmetal": lambda: s.g.match_substring_regex(RX),
        "polars": lambda: s.p.str.contains(RX),
        "pyarrow": lambda: pc.match_substring_regex(s.a, RX),
        "pandas": lambda: s.d.str.contains(RX, regex=True)},
        parallel=ps.project(pl.col("x").str.contains(RX),
                            pc.match_substring_regex(pc.field("x"), RX)))
    case(f, 'replace_substring("cust" -> "cx")', n, SB * 2, {
        "arrowmetal": lambda: s.g.replace("cust", "cx"),
        "polars": lambda: s.p.str.replace_all("cust", "cx", literal=True),
        "pyarrow": lambda: pc.replace_substring(s.a, "cust", "cx"),
        "pandas": lambda: s.d.str.replace("cust", "cx", regex=False)},
        parallel=ps.project(pl.col("x").str.replace_all("cust", "cx", literal=True),
                            pc.replace_substring(pc.field("x"), "cust", "cx")))
    case(f, 'split_pattern("_")', n, SB * 2, {
        "arrowmetal": lambda: s.g.split_pattern("_"),
        "polars": lambda: s.p.str.split("_"),
        "pyarrow": lambda: pc.split_pattern(s.a, "_"),
        "pandas": lambda: s.d.str.split("_")},
        parallel=ps.project(pl.col("x").str.split("_"),
                            pc.split_pattern(pc.field("x"), "_")))
    case(f, "trim (whitespace)", n, padded.nbytes * 2, {
        "arrowmetal": lambda: padded.g.utf8_trim(),
        "polars": lambda: padded.p.str.strip_chars(),
        "pyarrow": lambda: pc.utf8_trim_whitespace(padded.a),
        "pandas": lambda: padded.d.str.strip()},
        parallel=ppad.project(pl.col("x").str.strip_chars(),
                              pc.utf8_trim_whitespace(pc.field("x"))))
    case(f, "pad_left (width 20)", n, SB * 3, {
        "arrowmetal": lambda: s.g.pad_left(20, "*"),
        "polars": lambda: s.p.str.pad_start(20, "*"),
        "pyarrow": lambda: pc.utf8_lpad(s.a, 20, padding="*"),
        "pandas": lambda: s.d.str.rjust(20, "*")},
        parallel=ps.project(pl.col("x").str.pad_start(20, "*"),
                            pc.utf8_lpad(pc.field("x"), 20, padding="*")))
    case(f, "slice_codeunits [5:10]", n, SB, {
        "arrowmetal": lambda: s.g.slice_codeunits(5, 10),
        "polars": lambda: s.p.str.slice(5, 5),
        "pyarrow": lambda: pc.utf8_slice_codeunits(s.a, 5, 10),
        "pandas": lambda: s.d.str[5:10]},
        parallel=ps.project(pl.col("x").str.slice(5, 5),
                            pc.utf8_slice_codeunits(pc.field("x"), 5, 10)))
    case(f, "concat (a + '-' + b)", n, SB * 3, {
        "arrowmetal": lambda: s.g.str_concat(s2.g, "-"),
        "polars": lambda: s.p + "-" + s2.p,
        "pyarrow": lambda: pc.binary_join_element_wise(s.a, s2.a, "-"),
        "pandas": lambda: s.d + "-" + s2.d},
        parallel=ps.project(pl.col("x") + "-" + pl.col("y"),
                            pc.binary_join_element_wise(pc.field("x"), pc.field("y"), "-")))
    case(f, "is_alpha", n, SB, {
        "arrowmetal": lambda: s.g.utf8_is_alpha(),
        "polars": lambda: s.p.str.contains(r"^[^\W\d_]+$"),
        "pyarrow": lambda: pc.utf8_is_alpha(s.a),
        "pandas": lambda: s.d.str.isalpha()},
        notes={"polars": "no is_alpha predicate; the equivalent anchored regex"},
        parallel=ps.project(pl.col("x").str.contains(r"^[^\W\d_]+$"),
                            pc.utf8_is_alpha(pc.field("x"))))
    case(f, "to_strings (int64 -> utf8)", n, n * 16, {
        "arrowmetal": lambda: ints.g.to_strings(),
        "polars": lambda: ints.p.cast(pl.Utf8),
        "pyarrow": lambda: pc.cast(ints.a, pa.string()),
        "pandas": lambda: ints.d.astype(str)},
        parallel=pint.project(pl.col("x").cast(pl.Utf8), pc.field("x").cast(pa.string())))
    case(f, "parse (utf8 -> int64)", n, numtext.nbytes + n * 8, {
        "arrowmetal": lambda: numtext.g.parse("int64"),
        "polars": lambda: numtext.p.cast(pl.Int64),
        "pyarrow": lambda: pc.cast(numtext.a, pa.int64()),
        "pandas": lambda: numtext.d.astype("int64")},
        parallel=pnum.project(pl.col("x").cast(pl.Int64), pc.field("x").cast(pa.int64())))
    subset = pa.array(sd.vocab[:100])
    subset_g = am.array(subset)
    subset_list = sd.vocab[:100]
    case(f, "is_in (utf8, 100-value set)", n, SB, {
        "arrowmetal": lambda: s.g.is_in(subset_g),
        "polars": lambda: s.p.is_in(subset_list),
        "pyarrow": lambda: pc.is_in(s.a, value_set=subset),
        "pandas": lambda: s.d.isin(subset_list)},
        parallel=ps.project(pl.col("x").is_in(subset_list),
                            pc.is_in(pc.field("x"), value_set=subset)))


def family_temporal(d, n):
    f = "temporal"
    ts, ts2 = d("ts"), d("ts2")
    pdt = pd.Series(ts.n.astype("datetime64[us]"))
    pdt2 = pd.Series(ts2.n.astype("datetime64[us]"))
    pt = Par(x=ts.a, y=ts2.a)
    B = n * 12
    case(f, "year", n, B, {
        "arrowmetal": lambda: ts.g.year(),
        "polars": lambda: ts.p.dt.year(),
        "pyarrow": lambda: pc.year(ts.a),
        "pandas": lambda: pdt.dt.year},
        parallel=pt.project(pl.col("x").dt.year(), pc.year(pc.field("x"))))
    case(f, "month", n, B, {
        "arrowmetal": lambda: ts.g.month(),
        "polars": lambda: ts.p.dt.month(),
        "pyarrow": lambda: pc.month(ts.a),
        "pandas": lambda: pdt.dt.month},
        parallel=pt.project(pl.col("x").dt.month(), pc.month(pc.field("x"))))
    case(f, "day", n, B, {
        "arrowmetal": lambda: ts.g.day(),
        "polars": lambda: ts.p.dt.day(),
        "pyarrow": lambda: pc.day(ts.a),
        "pandas": lambda: pdt.dt.day},
        parallel=pt.project(pl.col("x").dt.day(), pc.day(pc.field("x"))))
    case(f, "floor_temporal (day)", n, n * 16, {
        "arrowmetal": lambda: ts.g.floor_temporal("day"),
        "polars": lambda: ts.p.dt.truncate("1d"),
        "pyarrow": lambda: pc.floor_temporal(ts.a, unit="day"),
        "pandas": lambda: pdt.dt.floor("D")},
        parallel=pt.project(pl.col("x").dt.truncate("1d"),
                            pc.floor_temporal(pc.field("x"), unit="day")))
    case(f, "add_duration (+1h)", n, n * 16, {
        "arrowmetal": lambda: ts.g.add_duration(3_600_000_000),
        "polars": lambda: ts.p.dt.offset_by("1h"),
        "pyarrow": lambda: pc.add(ts.a, pa.scalar(3_600_000_000, pa.duration("us"))),
        "pandas": lambda: pdt + pd.Timedelta("1h")},
        parallel=pt.project(pl.col("x").dt.offset_by("1h"),
                            pc.add(pc.field("x"), pa.scalar(3_600_000_000, pa.duration("us")))))
    case(f, "days_between", n, n * 24, {
        "arrowmetal": lambda: ts.g.days_between(ts2.g),
        "polars": lambda: (ts2.p - ts.p).dt.total_days(),
        "pyarrow": lambda: pc.days_between(ts.a, ts2.a),
        "pandas": lambda: (pdt2 - pdt).dt.days},
        notes={"polars": "no days_between; duration difference in whole days",
               "pandas": "same"},
        parallel=pt.project((pl.col("y") - pl.col("x")).dt.total_days(),
                            pc.days_between(pc.field("x"), pc.field("y"))))
    case(f, "week", n, B, {
        "arrowmetal": lambda: ts.g.week(),
        "polars": lambda: ts.p.dt.week(),
        "pyarrow": lambda: pc.week(ts.a),
        "pandas": lambda: pdt.dt.isocalendar().week},
        parallel=pt.project(pl.col("x").dt.week(), pc.week(pc.field("x"))))
    case(f, "strftime (%Y-%m-%d)", n, n * 20, {
        "arrowmetal": lambda: ts.g.strftime("%Y-%m-%d"),
        "polars": lambda: ts.p.dt.strftime("%Y-%m-%d"),
        "pyarrow": lambda: pc.strftime(ts.a, format="%Y-%m-%d"),
        "pandas": lambda: pdt.dt.strftime("%Y-%m-%d")},
        parallel=pt.project(pl.col("x").dt.strftime("%Y-%m-%d"),
                            pc.strftime(pc.field("x"), format="%Y-%m-%d")))
    # strptime is fed ArrowMetal's own strftime output: pyarrow's strftime folds the fractional second
    # into %S, and none of the three CPU libraries will then parse "%Y-%m-%d %H:%M:%S" back.
    stamps = ts.g.strftime("%Y-%m-%d %H:%M:%S").to_arrow()
    st_g = am.MetalArray.from_arrow(stamps)
    st_p = pl.Series(stamps)
    st_d = stamps.to_pandas()
    pst = Par(x=stamps)
    case(f, "strptime (%Y-%m-%d %H:%M:%S)", n, n * 28, {
        "arrowmetal": lambda: st_g.strptime("%Y-%m-%d %H:%M:%S"),
        "polars": lambda: st_p.str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"),
        "pyarrow": lambda: pc.strptime(stamps, format="%Y-%m-%d %H:%M:%S", unit="us"),
        "pandas": lambda: pd.to_datetime(st_d, format="%Y-%m-%d %H:%M:%S")},
        parallel=pst.project(pl.col("x").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"),
                             pc.strptime(pc.field("x"), format="%Y-%m-%d %H:%M:%S", unit="us")))
    case(f, "assume_timezone (America/New_York)", n, n * 16, {
        "arrowmetal": lambda: ts.g.assume_timezone("America/New_York", ambiguous="earliest",
                                                   nonexistent="latest"),
        "polars": lambda: ts.p.dt.replace_time_zone("America/New_York", ambiguous="earliest",
                                                    non_existent="null"),
        "pyarrow": lambda: pc.assume_timezone(ts.a, "America/New_York", ambiguous="earliest",
                                              nonexistent="latest"),
        "pandas": lambda: pdt.dt.tz_localize("America/New_York", ambiguous=True,
                                             nonexistent="shift_forward")},
        parallel=pt.project(pl.col("x").dt.replace_time_zone("America/New_York",
                                                             ambiguous="earliest",
                                                             non_existent="null"),
                            pc.assume_timezone(pc.field("x"), "America/New_York",
                                               ambiguous="earliest", nonexistent="latest")))
    # The tz-aware column the next two rows read; every library gets the same values.
    aware = ts.a.cast(pa.timestamp("us", "America/New_York"))
    aw_g = am.MetalArray.from_arrow(aware)
    aw_p = pl.Series(aware)
    aw_d = aware.to_pandas()
    paw = Par(x=aware)
    case(f, "local_timestamp (America/New_York)", n, n * 16, {
        "arrowmetal": lambda: aw_g.local_timestamp(),
        "polars": lambda: aw_p.dt.replace_time_zone(None),
        "pyarrow": lambda: pc.local_timestamp(aware),
        "pandas": lambda: aw_d.dt.tz_localize(None)},
        notes={"polars": "replace_time_zone(None) keeps the wall clock, which is local_timestamp"},
        parallel=paw.project(pl.col("x").dt.replace_time_zone(None),
                             pc.local_timestamp(pc.field("x"))))
    case(f, "is_dst (America/New_York)", n, n * 9, {
        "arrowmetal": lambda: aw_g.is_dst(),
        "polars": lambda: aw_p.dt.dst_offset(),
        "pyarrow": lambda: pc.is_dst(aware),
        "pandas": None},
        notes={"polars": "no is_dst; dst_offset is the same tz lookup",
               "pandas": "no vectorised is_dst on a tz-aware Series"},
        parallel=paw.project(pl.col("x").dt.dst_offset(), pc.is_dst(pc.field("x"))))


def family_window(d, n):
    f = "window"
    fl = d("f64_nn")
    i = d("i64")
    pw, pwi = Par(x=fl.a), Par(x=i.a)
    SEQ = ("a running total is sequential; polars keeps it in one pass and Acero has no node for "
           "it, so there is no parallel idiom on either side")
    B = n * 16
    case(f, "cumulative_sum (float64)", n, B, {
        "arrowmetal": lambda: fl.g.cumulative_sum(),
        "polars": lambda: fl.p.cum_sum(),
        "pyarrow": lambda: pc.cumulative_sum(fl.a),
        "pandas": lambda: fl.d.cumsum()},
        parallel={**pw.vector(pl.col("x").cum_sum()),
                  **par_none(NO_ACERO_VECTOR + "; " + SEQ, ["pyarrow-threaded"])})
    case(f, "cumulative_prod (float64)", n, B, {
        "arrowmetal": lambda: fl.g.cumulative_prod(),
        "polars": lambda: fl.p.cum_prod(),
        "pyarrow": lambda: pc.cumulative_prod(fl.a),
        "pandas": lambda: fl.d.cumprod()},
        parallel={**pw.vector(pl.col("x").cum_prod()),
                  **par_none(NO_ACERO_VECTOR + "; " + SEQ, ["pyarrow-threaded"])})
    case(f, "shift (lag 1, int64)", n, B, {
        "arrowmetal": lambda: i.g.shift(1),
        "polars": lambda: i.p.shift(1),
        "pyarrow": None,
        "pandas": lambda: i.d.shift(1)},
        notes={"pyarrow": "pyarrow.compute has no shift/lag kernel"},
        parallel=pwi.vector(pl.col("x").shift(1)))
    case(f, "pairwise_diff (float64)", n, B, {
        "arrowmetal": lambda: fl.g.pairwise_diff(),
        "polars": lambda: fl.p.diff(),
        "pyarrow": lambda: pc.pairwise_diff(fl.a),
        "pandas": lambda: fl.d.diff()},
        parallel={**pw.vector(pl.col("x").diff()),
                  **par_none(NO_ACERO_VECTOR, ["pyarrow-threaded"])})
    case(f, "rolling_sum (window 64)", n, B, {
        "arrowmetal": lambda: fl.g.rolling_sum(64),
        "polars": lambda: fl.p.rolling_sum(64),
        "pyarrow": None,
        "pandas": lambda: fl.d.rolling(64).sum()},
        notes={"pyarrow": "pyarrow.compute has no rolling-window kernels"},
        parallel=pw.vector(pl.col("x").rolling_sum(64)))
    case(f, "rolling_mean (window 64)", n, B, {
        "arrowmetal": lambda: fl.g.rolling_mean(64),
        "polars": lambda: fl.p.rolling_mean(64),
        "pyarrow": None,
        "pandas": lambda: fl.d.rolling(64).mean()},
        parallel=pw.vector(pl.col("x").rolling_mean(64)))


def family_decimal(d, n):
    f = "decimal"
    a, b = d("dec"), d("dec2")
    pl_a = pl.Series("x", a.a)
    pl_b = pl.Series("y", b.a)
    pd_a = pd.Series(pd.arrays.ArrowExtensionArray(a.a))
    pd_b = pd.Series(pd.arrays.ArrowExtensionArray(b.a))
    pdec = Par(x=a.a, y=b.a)
    B = n * 48
    case(f, "decimal add (128-bit)", n, B, {
        "arrowmetal": lambda: a.g.decimal_add(b.g),
        "polars": lambda: pl_a + pl_b,
        "pyarrow": lambda: pc.add(a.a, b.a),
        "pandas": lambda: pd_a + pd_b},
        parallel=pdec.project(pl.col("x") + pl.col("y"), pc.add(pc.field("x"), pc.field("y"))))
    dec3 = pa.scalar(decimal.Decimal("3"), pa.decimal128(2, 0))
    lim = decimal.Decimal("500000.0000")
    dec_lim = pa.scalar(lim, pa.decimal128(18, 4))
    case(f, "decimal multiply (by scalar)", n, n * 32, {
        "arrowmetal": lambda: a.g.decimal_mul(3),
        "polars": lambda: pl_a * 3,
        "pyarrow": lambda: pc.multiply(a.a, dec3),
        "pandas": lambda: pd_a * 3},
        parallel=pdec.project(pl.col("x") * 3, pc.multiply(pc.field("x"), dec3)))
    case(f, "decimal compare (> scalar)", n, n * 17, {
        "arrowmetal": lambda: a.g > lim,
        "polars": lambda: pl_a > lim,
        "pyarrow": lambda: pc.greater(a.a, dec_lim),
        "pandas": lambda: pd_a > lim},
        parallel=pdec.project(pl.col("x") > lim, pc.greater(pc.field("x"), dec_lim)))
    case(f, "decimal sum", n, n * 16, {
        "arrowmetal": lambda: a.g.sum(),
        "polars": lambda: pl_a.sum(),
        "pyarrow": lambda: pc.sum(a.a),
        "pandas": lambda: pd_a.sum()},
        parallel=pdec.reduce(pl.col("x").sum(), [("x", "sum")]))
    case(f, "decimal round (2 places)", n, n * 32, {
        "arrowmetal": lambda: a.g.decimal_round(2),
        "polars": lambda: pl_a.round(2),
        "pyarrow": lambda: pc.round(a.a, ndigits=2),
        "pandas": lambda: pd_a.round(2)},
        parallel=pdec.project(pl.col("x").round(2), pc.round(pc.field("x"), ndigits=2)))


_NESTED_SIZES_DONE = set()


def family_nested(d, n):
    f = "nested"
    m = min(n, 10_000_000)
    # The family caps its size, so two requested sizes above the cap would measure the same 10M
    # rows twice; run each capped size once per invocation.
    if m in _NESTED_SIZES_DONE:
        print(f"  nested: {m:,} rows already measured in this run, skipped for {n:,}")
        return
    _NESTED_SIZES_DONE.add(m)
    rng = np.random.default_rng(SEED + 3)
    per_row = 4
    child = rng.integers(0, 1000, size=m * per_row, dtype=np.int64)
    offsets = np.arange(m + 1, dtype=np.int32) * per_row
    lst = pa.ListArray.from_arrays(pa.array(offsets), pa.array(child))
    a_x = pa.array(rng.integers(0, 1000, size=m, dtype=np.int64))
    a_y = pa.array(rng.random(m))
    st = pa.StructArray.from_arrays([a_x, a_y], names=["x", "y"])
    g_lst, g_st = am.array(lst), am.array(st)
    pl_lst = pl.Series("l", lst)
    pl_st = pl.Series("s", st)
    pd_lst = pd.Series(pd.arrays.ArrowExtensionArray(lst))
    pd_st = pd.Series(pd.arrays.ArrowExtensionArray(st))
    pn = Par(l=lst, s=st)
    B = m * per_row * 8
    case(f, "list_value_length", m, B, {
        "arrowmetal": lambda: g_lst.list_value_length(),
        "polars": lambda: pl_lst.list.len(),
        "pyarrow": lambda: pc.list_value_length(lst),
        "pandas": lambda: pd_lst.list.len()},
        parallel=pn.project(pl.col("l").list.len(), pc.list_value_length(pc.field("l"))))
    case(f, "list_flatten", m, B * 2, {
        "arrowmetal": lambda: g_lst.list_flatten(),
        "polars": lambda: pl_lst.explode(),
        "pyarrow": lambda: pc.list_flatten(lst),
        "pandas": lambda: pd_lst.explode()},
        parallel={**pn.vector(pl.col("l").explode()),
                  **par_none(NO_ACERO_VECTOR, ["pyarrow-threaded"])})
    case(f, "list_element (index 1)", m, B, {
        "arrowmetal": lambda: g_lst.list_element(1),
        "polars": lambda: pl_lst.list.get(1),
        "pyarrow": lambda: pc.list_element(lst, 1),
        "pandas": lambda: pd_lst.list[1]},
        parallel=pn.project(pl.col("l").list.get(1), pc.list_element(pc.field("l"), 1)))
    case(f, "struct_field", m, m * 16, {
        "arrowmetal": lambda: g_st.struct_field("y"),
        "polars": lambda: pl_st.struct.field("y"),
        "pyarrow": lambda: pc.struct_field(st, "y"),
        "pandas": lambda: pd_st.struct.field("y")},
        parallel=pn.project(pl.col("s").struct.field("y"),
                            pc.struct_field(pc.field("s"), "y")))
    del g_lst, g_st, pl_lst, pl_st, pd_lst, pd_st, lst, st, child, pn
    gc.collect()


def family_chains(d, n):
    f = "chains"
    rng = np.random.default_rng(SEED + 5)
    region = rng.integers(0, 5, size=n, dtype=np.int32)
    amount = (rng.random(n) * 500)
    a_r, a_a = pa.array(region), pa.array(amount)
    g_r, g_a = am.array(a_r), am.array(a_a)
    keys = d.keys(1000)
    g_k = keys.g
    df = pl.DataFrame({"region": a_r, "amount": a_a, "k": keys.a})
    lz = df.lazy()
    tbl = pa.table({"region": a_r, "amount": a_a, "k": keys.a})
    pdf = pd.DataFrame({"region": region, "amount": amount, "k": keys.n})
    pch = Par(region=a_r, amount=a_a, k=keys.a)
    B = n * 12

    def q1():
        return g_a.filter((g_r == 2) & (g_a > 100.0)).sum()

    def q1_batched():
        with am.batch():
            return g_a.filter((g_r == 2) & (g_a > 100.0)).sum()

    q1_keep = (pl.col("region") == 2) & (pl.col("amount") > 100.0)
    case(f, "filter two columns + sum", n, B, {
        "arrowmetal": q1,
        "polars": lambda: lz.filter((pl.col("region") == 2) & (pl.col("amount") > 100.0))
                            .select(pl.col("amount").sum()).collect(),
        "pyarrow": lambda: pc.sum(pc.filter(a_a, pc.and_(pc.equal(a_r, 2), pc.greater(a_a, 100.0)))),
        "numpy": lambda: amount[(region == 2) & (amount > 100.0)].sum()},
        notes={"polars": "the default row is already a LazyFrame; the parallel row is the same "
                         "plan on the streaming engine"},
        parallel={
            **pch.polars(lambda lf: lf.filter(q1_keep).select(pl.col("amount").sum()),
                         True, ".filter(...).select(...)"),
            **pch.acero(lambda: [
                ac.Declaration("filter", ac.FilterNodeOptions(
                    (pc.field("region") == 2) & (pc.field("amount") > 100.0))),
                ac.Declaration("aggregate", ac.AggregateNodeOptions(
                    [("amount", "sum", None, "s")]))], "filter + aggregate nodes",
                cols=("region", "amount"))})
    case(f, "filter two columns + sum [batched]", n, B, {
        "arrowmetal": q1_batched,
        "polars": None, "pyarrow": None, "pandas": None},
        notes={"arrowmetal": "same chain inside `with am.batch()`; compare with the unbatched row",
               "polars": "batching is an ArrowMetal-only concept",
               "pyarrow": "batching is an ArrowMetal-only concept",
               "pandas": "batching is an ArrowMetal-only concept"})

    def q2():
        m = g_r == 2
        return am.group_by([g_k.filter(m)]).sum(g_a.filter(m))

    def q2_batched():
        with am.batch():
            m = g_r == 2
            return am.group_by([g_k.filter(m)]).sum(g_a.filter(m))

    case(f, "group-by after filter", n, n * 20, {
        "arrowmetal": q2,
        "polars": lambda: lz.filter(pl.col("region") == 2).group_by("k")
                            .agg(pl.col("amount").sum()).collect(),
        "pyarrow": lambda: tbl.filter(pc.equal(a_r, 2)).group_by("k").aggregate([("amount", "sum")]),
        "pandas": lambda: pdf[pdf.region == 2].groupby("k", sort=False)["amount"].sum()},
        notes={"polars": "the default row is already a LazyFrame; the parallel row is the same "
                         "plan on the streaming engine"},
        parallel={
            **pch.polars(lambda lf: lf.filter(pl.col("region") == 2).group_by("k")
                                      .agg(pl.col("amount").sum()),
                         True, ".filter(...).group_by(...).agg(...)"),
            **pch.acero(lambda: [
                ac.Declaration("filter", ac.FilterNodeOptions(pc.equal(pc.field("region"), 2))),
                ac.Declaration("aggregate", ac.AggregateNodeOptions(
                    [("amount", "hash_sum", None, "s")], keys=["k"]))],
                "filter + hash aggregate nodes", cols=("region", "amount", "k"))},
        compare=UNORDERED)
    case(f, "group-by after filter [batched]", n, n * 20, {
        "arrowmetal": q2_batched,
        "polars": None, "pyarrow": None, "pandas": None},
        notes={"arrowmetal": "same chain inside `with am.batch()`"})

    # Indices safely inside the ~50% that survives the filter, so no side has to measure the
    # filtered length first (which would force a sync inside the batch).
    small_idx_np = rng.integers(0, max(1, n // 4), size=max(1, n // 8), dtype=np.int32)
    small_idx_pa = pa.array(small_idx_np)
    small_idx_pl = pl.Series(small_idx_np)
    g_idx = am.array(small_idx_pa)
    pl_amount = pl.Series("a", a_a)

    def q3():
        return g_a.filter(g_a > 250.0).take(g_idx)

    def q3_batched():
        with am.batch():
            return g_a.filter(g_a > 250.0).take(g_idx)

    case(f, "compare + filter + take", n, n * 20, {
        "arrowmetal": q3,
        "polars": lambda: pl_amount.filter(pl_amount > 250.0).gather(small_idx_pl),
        "pyarrow": lambda: pc.take(pc.filter(a_a, pc.greater(a_a, 250.0)), small_idx_pa),
        "numpy": lambda: amount[amount > 250.0][small_idx_np]},
        parallel={
            **pch.polars(lambda lf: lf.filter(pl.col("amount") > 250.0)
                                      .select(pl.col("amount").gather(small_idx_pl)),
                         False, ".filter(...).select(gather(...))"),
            **par_none("the gather at the end is a vector function: no Acero node takes by "
                       "index, so pc.take is the only idiom", ["pyarrow-threaded"])})
    case(f, "compare + filter + take [batched]", n, n * 20, {
        "arrowmetal": q3_batched,
        "polars": None, "pyarrow": None, "pandas": None},
        notes={"arrowmetal": "same chain inside `with am.batch()`"})
    del g_r, g_a, df, lz, tbl, pdf, pch
    gc.collect()


def family_small(sizes):
    """The latency floor: the same three operations at small row counts, batched and unbatched."""
    f = "latency"
    for n in sizes:
        rng = np.random.default_rng(SEED + 11)
        v = rng.integers(-1000, 1001, size=n, dtype=np.int64)
        a = pa.array(v)
        g = am.array(a)
        p = pl.Series("x", a)
        dser = pd.Series(v)
        k = rng.integers(0, min(1000, n), size=n, dtype=np.int32)
        a_k = pa.array(k)
        g_k = am.array(a_k)
        df = pl.DataFrame({"k": a_k, "x": a})
        tbl = pa.table({"k": a_k, "x": a})
        pdf = pd.DataFrame({"k": k, "x": v})
        psm = Par(x=a, k=a_k)
        B = n * 8
        case(f, "sum(int64)", n, B, {
            "arrowmetal": lambda: g.sum(),
            "polars": lambda: p.sum(),
            "pyarrow": lambda: pc.sum(a),
            "pandas": lambda: dser.sum()},
            parallel=psm.reduce(pl.col("x").sum(), [("x", "sum")]))

        def sum_batched():
            with am.batch():
                return g.sum()
        case(f, "sum(int64) [batched]", n, B, {
            "arrowmetal": sum_batched, "polars": None, "pyarrow": None, "pandas": None},
            notes={"arrowmetal": "one-call batch; shows the fixed cost of the batch itself"})
        case(f, "filter(int64 > 0)", n, B, {
            "arrowmetal": lambda: g.filter_where(">", 0),
            "polars": lambda: p.filter(p > 0),
            "pyarrow": lambda: pc.filter(a, pc.greater(a, 0)),
            "pandas": lambda: dser[dser > 0]},
            parallel=psm.filter_((pl.col("x") > 0, pl.col("x")),
                                 pc.greater(pc.field("x"), 0), pc.field("x"), cols=("x",)))

        def filter_batched():
            with am.batch():
                return g.filter_where(">", 0)
        case(f, "filter(int64 > 0) [batched]", n, B, {
            "arrowmetal": filter_batched, "polars": None, "pyarrow": None, "pandas": None})
        case(f, "group-by sum (1000 keys)", n, n * 12, {
            "arrowmetal": lambda: am.group_by([g_k]).sum(g),
            "polars": lambda: df.group_by("k").agg(pl.col("x").sum()),
            "pyarrow": lambda: tbl.group_by("k").aggregate([("x", "sum")]),
            "pandas": lambda: pdf.groupby("k", sort=False)["x"].sum()},
            parallel=psm.group(["k"], pl.col("x").sum(), [("x", "sum")]), compare=UNORDERED)

        def gb_batched():
            with am.batch():
                return am.group_by([g_k]).sum(g)
        case(f, "group-by sum (1000 keys) [batched]", n, n * 12, {
            "arrowmetal": gb_batched, "polars": None, "pyarrow": None, "pandas": None})
        del g, g_k, p, dser, df, tbl, pdf, psm
        gc.collect()


# ---------------------------------------------------------------- reporting

# Numbers copied verbatim from docs/BENCHMARKS.md (round 7 and round 6, Apple M4 Max, 50M rows).
# The Swift benchmark is NOT rebuilt or re-run here; these are quoted for context only.
SWIFT_BASELINE = [
    # (family, op, rows, metal_ms, cpu16_ms, cpu16_cpu_ms, source)
    ("sort", "argsort int64", 50_000_000, 128.52, 653.97, 5017.1, "round 7"),
    ("sort", "sort float64", 50_000_000, 138.52, 591.97, 4468.6, "round 7"),
    ("sort", "top_k (k=100, int64)", 50_000_000, 128.67, 1.96, 25.6, "round 7; superseded, the sort table above measures 2.67 ms"),
    ("strings", 'contains("north")', 10_000_000, 1.65, 17.77, 250.0, "round 7"),
    ("strings", 'starts_with("cust_1")', 10_000_000, 1.75, 3.10, 42.7, "round 7"),
    ("reductions", "sum(int64, 10% nulls)", 50_000_000, 1.07, 4.87, 66.6, "round 6"),
    ("compare+select", "filter int64 (30% kept)", 50_000_000, 2.95, 3.49, 48.8, "round 6"),
    ("compare+select", "take (n/2 random indices)", 50_000_000, 5.91, 11.75, 154.6, "round 6"),
    ("chains", "filter two columns + sum", 50_000_000, 1.75, 6.27, 86.5, "round 6"),
]

BASELINE_LIBS = ("polars", "pyarrow", "pandas", "numpy")
# Every CPU idiom "fastest CPU" is chosen from: the plain eager one and the parallel one.
CPU_LIBS = BASELINE_LIBS + PAR_LIBS
# (column heading, library) pairs, in the order the per-family tables show them.
REPORT_COLUMNS = [("polars", "polars"), ("polars-lazy", "polars-lazy"),
                  ("pyarrow", "pyarrow"), ("pyarrow-threaded", "pyarrow-threaded"),
                  ("pandas / numpy", "pandas")]


def fmt_ratio(r):
    """baseline / ArrowMetal, with enough digits to show how far off a bad row is."""
    if r is None:
        return "--"
    return f"{r:.3f}x" if r < 0.1 else f"{r:.2f}x"


def fmt_ms(v):
    """Milliseconds with enough digits to stay meaningful under the dispatch floor."""
    if v is None:
        return "--"
    if v < 1:
        return f"{v:.3f}"
    if v < 100:
        return f"{v:.2f}"
    return f"{v:.0f}"


def fmt_gbs(g):
    if g is None:
        return "--"
    if g >= 100:
        return f"{g:.0f}"
    if g >= 10:
        return f"{g:.1f}"
    return f"{g:.2f}"


def fmt_cores(r):
    """cpu_ms / wall_ms: the number of cores this measurement actually kept busy."""
    if not r or not r.get("wall_ms") or r.get("cpu_ms") is None:
        return None
    return r["cpu_ms"] / r["wall_ms"]


def cpu_cell(r):
    c = fmt_cores(r)
    return f"{fmt_ms(r['wall_ms'])} / {fmt_ms(r['cpu_ms'])}" + (f" ({c:.1f}c)" if c else "")


def cores_summary(rows, context=False):
    """Per library and idiom, the median and max of cpu_ms/wall_ms over every measured row.

    This is the number the claim "on all cores" has to be checked against: 1.0 means the idiom ran
    on one core whatever the thread pool's size.

    `context` adds this process's machine and thread-pool sizes to the header. Only a run that has
    just measured the rows may pass it: `--cores <csv>` reads a CSV that may have been measured on
    another machine, with other pool sizes, and must not caption it with this one's.
    """
    per = {}
    per_fam = {}
    for r in rows:
        c = fmt_cores(r) if r["status"] == "ok" else None
        if c is None:
            continue
        per.setdefault(r["library"], []).append(c)
        per_fam.setdefault((r["family"], r["library"]), []).append(c)
    order = [lib for lib in ("arrowmetal",) + CPU_LIBS if lib in per]
    order += sorted(k for k in per if k not in order)
    out = []
    out.append("Cores actually used, per library and idiom: cpu_ms / wall_ms over every measured "
               "row of this CSV.")
    if context:
        out.append(f"Measured on {os.uname().sysname} {os.uname().release}, "
                   f"{os.cpu_count()} logical CPUs, polars {pl.thread_pool_size()} threads, "
                   f"pyarrow {pa.cpu_count()} threads, Acero fed {NCHUNK} record batches.")
    out.append("")
    out.append(f"{'library / idiom':<20}{'rows':>7}{'median':>9}{'mean':>9}{'max':>9}"
               f"{'rows >2':>9}{'%':>6}")
    out.append("-" * 69)
    for lib in order:
        v = per[lib]
        many = sum(1 for x in v if x > 2.0)
        out.append(f"{lib:<20}{len(v):>7}{statistics.median(v):>9.2f}"
                   f"{statistics.fmean(v):>9.2f}{max(v):>9.2f}"
                   f"{many:>9}{round(100 * many / len(v)):>5}%")
    if "arrowmetal" in per:
        out.append("")
        out.append("arrowmetal's number is host CPU time only -- the thread encoding the command "
                   "buffer and")
        out.append("waiting on it. Time the GPU spends executing is not counted in it, by either "
                   "side's clock.")
    out.append("")
    out.append("By family (median cores)")
    out.append("")
    fams = []
    for r in rows:
        if r["family"] not in fams:
            fams.append(r["family"])
    head = f"{'family':<16}" + "".join(f"{lib:>18}" for lib in order)
    out.append(head)
    out.append("-" * len(head))
    for fam in fams:
        cells = ""
        for lib in order:
            v = per_fam.get((fam, lib))
            cells += f"{statistics.median(v):>18.2f}" if v else f"{'--':>18}"
        out.append(f"{fam:<16}{cells}")
    out.append("")
    return "\n".join(out)


def _host_scan_note(verdicts):
    """`any` and `all` stop at the first decisive bit on the host (docs/COVERAGE.md), so their ratio
    is a property of the data, not of a GPU pass; the summary says so when they are in the run."""
    keys = [k for k in _ORDER if re.match(r"(any|all)\b", k[1])]
    ok = sum(1 for k, v in zip(_ORDER, verdicts) if k in keys and v == "OK")
    if not ok:
        return ""
    total_ok = sum(1 for v in verdicts if v == "OK")
    sizes = sorted({k[2] for k in keys})
    WORD = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight"}
    return (f" {WORD.get(ok, ok)} of the {total_ok} — `any` and `all` at "
            + " and ".join(f"{n // 1_000_000}M" if n % 1_000_000 == 0 else f"{n:,}" for n in sizes)
            + " rows — are host CPU scans rather than GPU passes: the kernel stops at the first "
            "decisive bit without a dispatch ([COVERAGE.md](COVERAGE.md)), so their ratio depends on "
            "where that bit is.")


def verdict(ratio):
    if ratio is None:
        return "n/a"
    if ratio >= 3.0:
        return "OK"
    if ratio >= 1.0:
        return "WARN"
    return "to improve"


VERDICT_MARK = {"OK": "✅", "WARN": "⚠️", "to improve": "❌", "n/a": "—"}


def build_report(csv_path, elapsed_s):
    by_key = {}
    for r in ROWS:
        by_key.setdefault((r["family"], r["op"], r["rows"]), {})[r["library"]] = r

    families = []
    for key in _ORDER:
        if key[0] not in families:
            families.append(key[0])

    shortfalls = []
    verdicts = []
    errors = 0
    lines = []
    lines.append("# The complete comparison matrix")
    lines.append("")
    lines.append(f"Generated by `Benchmarks/full_matrix.py` on {datetime.date.today().isoformat()}, "
                 f"Apple M4 Max, 16 CPU cores, 64 GB unified memory, Darwin {os.uname().release}.")
    lines.append("")
    lines.append(f"ArrowMetal {am.__version__} on `{am.device_name()}` against "
                 f"Polars {pl.__version__} ({pl.thread_pool_size()} threads), "
                 f"pyarrow {pa.__version__} ({pa.cpu_count()} threads), "
                 f"pandas {pd.__version__}, numpy {np.__version__}.")
    lines.append("")
    lines.append(f"Run time {elapsed_s / 60:.1f} minutes; raw numbers in `{os.path.relpath(csv_path, ROOT)}`.")
    lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append("- Every number is the best wall time of up to five calls after one warm-up, with the "
                 "process CPU time (all threads) of that same call beside it, and then **the cores that "
                 "call used** (cpu-ms / wall-ms). A slow call is repeated fewer times, never "
                 "fewer than twice; the CSV records the count.")
    lines.append("- **Every CPU library gets two columns: its plain eager idiom and its most parallel "
                 "idiom.** `polars-lazy` is the same expression through `pl.LazyFrame`, collected on "
                 "the in-memory or the streaming engine; `pyarrow-threaded` is an Acero plan over the "
                 f"same values split into {NCHUNK} record batches with `to_table(use_threads=True)`, or "
                 f"`pa.Table.group_by` over that {NCHUNK}-chunk table. The `note` column of every such "
                 "CSV row names the exact idiom. This exists because the eager idioms use about one "
                 "core on the element-wise and whole-column reduction rows however many threads the "
                 "pool has — see the cores table below for where that is and is not true — and a "
                 "comparison against one core is not the comparison this project wants to make.")
    lines.append("- pandas' threaded paths, numexpr and numba, were not installed for this run "
                 "(numexpr is element-wise arithmetic through `pd.eval` / `DataFrame.eval`; the "
                 "numba engine with `parallel=True` is `rolling`, `groupby.agg` / `transform`, "
                 "`apply`), so pandas is measured in its eager idiom and every `pandas-parallel` "
                 "row's note says so. numpy's ufuncs are single-threaded. Both are recorded as "
                 "`pandas-parallel` / `numpy-parallel` rows saying so, rather than left out.")
    lines.append("- **ratio** is the fastest CPU idiom's wall time divided by ArrowMetal's, taken "
                 "across *all* the idioms of all the libraries; the **fastest CPU** column names the "
                 "idiom that won. The verdict is the project's own bar: ✅ at or above 3x, "
                 "⚠️ between 1x and 3x, ❌ to improve: the fastest CPU idiom is ahead.")
    lines.append("- `--` in a library column means that library has no equivalent operation, or the "
                 "matrix has not measured a threaded idiom for it (the reason is in the CSV's `note` "
                 "column); `err` means the "
                 "call raised, and the message is in the CSV. Nothing is skipped silently.")
    # A family that caps its size (nested, at 10M) used to be measured once per requested size;
    # the CSV then carries the same (family, op, rows, library) key twice. by_key keeps the later
    # record, and the count says so.
    _timed = [r for r in ROWS if r["wall_ms"] is not None]
    _seen, _dup = set(), set()
    for r in ROWS:
        k = (r["family"], r["op"], r["rows"], r["library"])
        (_dup if k in _seen else _seen).add(k)
    if _dup:
        _fams = ", ".join(sorted({k[0] for k in _dup}))
        _sizes = ", ".join(f"{n:,}" for n in sorted({k[2] for k in _dup}))
        lines.append(f"- {len(_timed):,} timed records, "
                     f"{len({(r['family'], r['op'], r['rows'], r['library']) for r in _timed}):,} rows: "
                     f"the {_fams} family at {_sizes} rows was measured twice and the later pass is shown.")
    lines.append("- Bandwidth (GB/s) is bytes touched (input + output) over wall time. Where ArrowMetal "
                 "and the best baseline are both near the ~400 GB/s the single-pass rows of this matrix reach the "
                 "operation is memory-bound and no ratio above ~1.5x is available to either side.")
    lines.append("")
    summary_at = len(lines)          # the verdict tally is filled in once every row is scored
    lines.append("")

    for fam in families:
        lines.append(f"## {fam}")
        lines.append("")
        lines.append("| op | rows | ArrowMetal wall / CPU ms | ArrowMetal GB/s | "
                     + " | ".join(h for h, _ in REPORT_COLUMNS)
                     + " | fastest CPU idiom | ratio | verdict |")
        lines.append("|---|---:|---:|---:|" + "---:|" * len(REPORT_COLUMNS) + "---|---:|:--:|")
        for key in _ORDER:
            if key[0] != fam:
                continue
            row = by_key[key]
            amr = row.get("arrowmetal")
            cells = []
            best_lib, best_wall = None, None
            for lib in CPU_LIBS:
                r = row.get(lib)
                if r is None:
                    continue
                if r["status"] != "ok":
                    cells.append((lib, "--" if r["status"] == "no equivalent" else "err"))
                    continue
                cells.append((lib, cpu_cell(r)))
                if best_wall is None or r["wall_ms"] < best_wall:
                    best_lib, best_wall = lib, r["wall_ms"]
            cellmap = dict(cells)
            pandas_cell = cellmap.get("pandas") or cellmap.get("numpy") or "--"
            pandas_lib = "pandas" if "pandas" in cellmap else ("numpy" if "numpy" in cellmap else "")
            if pandas_lib == "numpy" and pandas_cell != "--":
                pandas_cell += " (numpy)"
            cellmap["pandas"] = pandas_cell
            if amr is None or amr["status"] != "ok":
                am_cell = "err" if (amr and amr["status"] == "error") else "--"
                am_gbs = "--"
                ratio, vd = None, "to improve" if amr and amr["status"] == "error" else "n/a"
                if amr and amr["status"] == "error":
                    shortfalls.append((key, None, best_lib, best_wall, amr, row))
                    errors += 1
            else:
                am_cell = f"**{fmt_ms(amr['wall_ms'])}** / {fmt_ms(amr['cpu_ms'])}"
                am_gbs = fmt_gbs(amr["gbs"])
                ratio = (best_wall / amr["wall_ms"]) if best_wall else None
                vd = verdict(ratio)
                if vd in ("WARN", "to improve"):
                    shortfalls.append((key, ratio, best_lib, best_wall, amr, row))
            verdicts.append(vd)
            lines.append(
                f"| {key[1]} | {key[2]:,} | {am_cell} | {am_gbs} | "
                + " | ".join(cellmap.get(lib, "--") for _h, lib in REPORT_COLUMNS)
                + f" | {best_lib or '--'} | {fmt_ratio(ratio)} | {VERDICT_MARK[vd]} |")
        lines.append("")

    tally = {"OK": 0, "WARN": 0, "to improve": 0, "n/a": 0}
    for v in verdicts:
        tally[v] = tally.get(v, 0) + 1
    # _ORDER is one entry per (family, op, rows): a row of the table, not an operation. Most
    # operations are measured at two sizes, so the two counts differ and the sentence says which
    # is which; the percentage is of rows, rounded rather than floored.
    n_rows = len(_ORDER)
    ops = {(fam, op) for fam, op, _rws in _ORDER}
    sizes_per_op = {}
    for fam, op, _rws in _ORDER:
        sizes_per_op[(fam, op)] = sizes_per_op.get((fam, op), 0) + 1
    spread = {}
    for count in sizes_per_op.values():
        spread[count] = spread.get(count, 0) + 1
    WORD = {1: "one size", 2: "two", 3: "three", 4: "four", 5: "five"}
    at = ", ".join(f"{n} measured at {WORD.get(k, k)}" if i == 0 else f"{n} at {WORD.get(k, k)}"
                   for i, (k, n) in enumerate(sorted(spread.items())))
    lines[summary_at] = (
        f"**{n_rows} rows measured, over {len(ops)} operations** ({at}). "
        f"{tally['OK']} at or above 3x (✅), {tally['WARN']} between 1x and 3x (⚠️), "
        f"{tally['to improve']} to improve, where the fastest CPU idiom is ahead (❌)"
        + (f" — of which {errors} a call that raised" if errors else " — none of them a call that "
           "raised")
        + f" — and {tally['n/a']} with no CPU equivalent to compare against (—). "
        f"{round(tally['OK'] * 100 / max(n_rows, 1))}% of the measured rows meet the bar."
        + _host_scan_note(verdicts) + "\n")

    # ---- the Swift / vDSP baselines, copied
    lines.append("## The 16-core Swift / vDSP baselines (copied from docs/BENCHMARKS.md)")
    lines.append("")
    lines.append("These are **not** re-measured here. They are the numbers already recorded in "
                 "`docs/BENCHMARKS.md` for the in-process Swift benchmark (Metal vs all 16 CPU cores on "
                 "the same Arrow buffers), quoted so the CPU-side ceiling is visible next to the Python "
                 "libraries. Rows are matched to this matrix by operation name where one exists.")
    lines.append("")
    lines.append("| op | rows | Metal (Swift bench) | 16-core Swift wall / CPU ms | source |")
    lines.append("|---|---:|---:|---:|---|")
    for fam, op, rws, metal, cpu16, cpu16cpu, src in SWIFT_BASELINE:
        lines.append(f"| {op} | {rws:,} | {metal:.2f} | {cpu16:.2f} / {cpu16cpu:.1f} | "
                     f"docs/BENCHMARKS.md {src} |")
    lines.append("")

    # ---- shortfalls
    lines.append("## Rows under the 3x bar")
    lines.append("")
    lines.append("Sorted by distance from the 3x bar, furthest first. "
                 "❌ means the fastest CPU idiom is ahead of ArrowMetal on that row. The baseline here is "
                 "the fastest of every idiom of every CPU library, named in its own column.")
    lines.append("")
    lines.append("| op | rows | ratio | ArrowMetal ms | fastest CPU idiom | baseline ms | "
                 "AM GB/s | baseline GB/s | likely cause |")
    lines.append("|---|---:|---:|---:|---|---:|---:|---:|---|")

    def sort_key(item):
        _key, ratio, _bl, _bw, _amr, _row = item
        return -1e9 if ratio is None else ratio
    for key, ratio, best_lib, best_wall, amr, row in sorted(shortfalls, key=sort_key):
        cause = diagnose(key, ratio, best_lib, amr, row)
        if amr is None or amr["status"] != "ok":
            lines.append(f"| {key[1]} | {key[2]:,} | err | -- | {best_lib or '--'} | "
                         f"{fmt_ms(best_wall)} | -- | -- | {cause} |")
            continue
        b = row.get(best_lib)
        lines.append(
            f"| {key[1]} | {key[2]:,} | {fmt_ratio(ratio)} | {fmt_ms(amr['wall_ms'])} | {best_lib} | "
            f"{fmt_ms(best_wall)} | {fmt_gbs(amr['gbs'])} | {fmt_gbs(b['gbs'])} | {cause} |")
    lines.append("")
    lines.append(f"{len(shortfalls)} of {len(_ORDER)} measured rows are below the 3x bar.")
    lines.append("")
    lines.append("## How many cores each idiom used")
    lines.append("")
    lines.append("cpu_ms / wall_ms for every measured row, per library and idiom. 1.00 means the "
                 "idiom ran on one core, whatever the size of the thread pool. This is the table "
                 "that decides whether \"on all cores\" is a true sentence about a given row.")
    lines.append("")
    lines.append("```")
    lines.append(cores_summary(ROWS))
    lines.append("```")
    lines.append("")
    lines.append("## Reproducing")
    lines.append("")
    lines.append("```")
    lines.append("DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \\")
    lines.append("  swift build -c release --product ArrowMetalC")
    lines.append("# full run")
    lines.append("PYTHONPATH=python python Benchmarks/full_matrix.py")
    lines.append("# 1M-row smoke")
    lines.append("PYTHONPATH=python python Benchmarks/full_matrix.py --quick")
    lines.append("# assert every parallel idiom answers what its default does")
    lines.append("PYTHONPATH=python python Benchmarks/full_matrix.py --verify --sizes 1000000")
    lines.append("```")
    lines.append("")
    return "\n".join(lines), shortfalls


SORT_BASED = ("count_distinct", "quantile", "mode(", "median", "tdigest", "unique", "value_counts")

CAUSE_HINTS = [
    # (predicate on (family, op), cause)
    (lambda fam, op: "match_like" in op and "_ wildcard" in op,
     "Host fallback, correctly: `Regex.likePredicate` (Sources/ArrowMetal/Kernels/Regex.swift) maps "
     "only a pure prefix / suffix / contains / equality LIKE pattern onto a GPU predicate. `_` is "
     "LIKE's single-character wildcard, so this pattern is translated to an anchored regex and matched "
     "row by row with NSRegularExpression (ICU) on the CPU. A GPU wildcard matcher is the fix."),
    (lambda fam, op: "match_like" in op,
     "`match_like` should take the GPU prefix path here (Regex.likePredicate); if this row is short of "
     "3x the GPU predicate itself is the cost, not a fallback."),
    (lambda fam, op: fam == "sort" and "utf8" in op and op.startswith(("argsort", "sort")),
     "`Kernels/StringSort.swift` sorts these on the GPU: an LSD radix over 7-byte prefix chunks, one "
     "stable radix pass per chunk, so the pass count is `ceil(longest row / 7)`. Each pass is a full "
     "64-bit radix argsort plus a gather of the next chunk's keys, which is several times the column "
     "in traffic; Polars sorts strings with one multi-threaded comparison sort over pointers. Wider "
     "chunks, or refining only the tie runs after the first pass, is the lever."),
    (lambda fam, op: "dictionary_encode (int32)" in op,
     "`am_dictionary_encode` routes an integer column to `DictionaryCompute.dictionaryEncoded()`, "
     "which is the GPU `unique()` pipeline: a full radix argsort of the column, run marks, a scan and "
     "a gather. pandas builds its categories from one hash-table pass. A hash-based dense-encoding "
     "kernel (the one `am_group_by_keys` already has for narrow ranges) is the fix."),
    (lambda fam, op: fam == "decimal" and "sum" in op,
     "`sum` on a decimal column routes to `am_decimal_op` op 18 (128-bit threadgroup partials, host "
     "combine). If this row is short of 3x the reduction itself is the cost."),
    (lambda fam, op: any(t in op for t in SORT_BASED),
     "Sort-based path: ArrowMetal answers this with a full GPU radix sort plus a run scan, where the "
     "CPU libraries use a hash table (count_distinct, mode, unique, value_counts) or a partial "
     "selection (quantile, tdigest). A hash/sketch kernel is the fix."),
    (lambda fam, op: op.startswith("min_max"),
     "`min_max` is two separate `am_reduce_ex` dispatches here (min_of_min_max then max_of_min_max), "
     "so it pays the dispatch floor twice for one pass' worth of work."),
    (lambda fam, op: fam == "strings" and ("parse" in op or "to_strings" in op),
     "Variable-length output: the kernel measures the lengths, prefix-sums them and writes in a second "
     "pass, against one streaming pass on the CPU."),
    (lambda fam, op: "top_k" in op,
     "`top_k` is a full GPU radix argsort plus a slice; a CPU running top-k touches each value once "
     "and rarely writes. Needs a partial radix / threadgroup selection kernel."),
    (lambda fam, op: op.startswith("partition_nth"),
     "`partition_nth_indices` is documented as the full stable argsort; there is no partial-partition "
     "kernel yet."),
    (lambda fam, op: fam == "latency",
     "Below the 110-160 us dispatch floor (the latency family's sum and filter rows at 1,000 rows): "
     "encode + commit + wait dominates the kernel. Batching removes "
     "most of it, but a single small call cannot beat an in-cache CPU loop."),
    (lambda fam, op: fam == "reductions" and ("first" in op or "last" in op or "any" in op or "all" in op),
     "Answered by a short-circuiting host scan of the bitmap in shared memory (no dispatch at all), so "
     "what is left is the ~1 us of ctypes marshalling around a call that itself takes a microsecond or "
     "two. The baselines answer the same question in their own process with no FFI hop."),
    (lambda fam, op: op.startswith("slice ("),
     "`MetalArray.slice` (Sources/ArrowMetal/Slice.swift) is now a zero-copy view at every offset and "
     "O(1) in the length, like the baselines: what this row measures is the ctypes hop into the C ABI "
     "and the Python wrapper object around the returned handle, roughly a microsecond, against an "
     "in-process metadata tweak on the other side. Both are constant time; neither is moving data."),
    (lambda fam, op: "dictionary_encode" in op,
     "`MetalStringArray.dictionaryEncode()` now takes the GPU hash path "
     "(`dictionaryEncodeGPU`, up to three hash rounds); what is left is the uniques buffer being "
     "rebuilt as a string array on the host."),
    (lambda fam, op: fam == "strings" and "split" in op,
     "`Regex.splitPattern` is documented **always CPU**: it builds a Swift `[String]` per row under "
     "`concurrentPerform`, then rebuilds a MetalStringArray. Nothing runs on the GPU."),
    (lambda fam, op: fam == "strings" and "regex" in op,
     "ICU host fallback: only a metacharacter-free pattern (or `^literal`) takes the GPU path, "
     "everything else is NSRegularExpression row by row across 4096-row chunks."),
    (lambda fam, op: fam == "strings" and "replace" in op,
     "Variable-length output: the kernel measures every row's new length, prefix-sums, then writes."),
    (lambda fam, op: fam == "temporal" and ("strftime" in op or "strptime" in op),
     "Two passes plus a prefix scan, and the output is 10 to 20 bytes a row: the whole cost is writing "
     "the text. `Kernels/TemporalFormat.swift` compiles the format into an op list on the host, so the "
     "kernel is generic and no format costs a shader recompile."),
    (lambda fam, op: fam == "temporal" and ("timezone" in op or "local_timestamp" in op or "is_dst" in op),
     "The zone's transition table (a few hundred instants) is uploaded once per zone and cached, so "
     "the per-row work is a ten-step binary search and an add — the pass is bandwidth-bound on the "
     "values themselves. See `Kernels/TimezoneGPU.swift`."),
    (lambda fam, op: fam == "temporal",
     "Civil-calendar arithmetic: days-from-civil and its inverse are a few dozen integer operations "
     "per row on both sides, so this is compute-bound rather than bandwidth-bound and the GPU's only "
     "advantage is its lane count. Both sides land within a factor of three."),
    (lambda fam, op: op.startswith(("rank", "dense_rank")),
     "`rank` is the full GPU argsort plus a segmented scan, so it inherits the radix sort's traffic; "
     "see the sort family above."),
    (lambda fam, op: "replace_with_mask" in op,
     "Three passes: the mask's prefix sum, a gather of the replacements, then the merge. The Polars "
     "lazy idiom (`when/then/otherwise` over the mask) is one streaming pass over the column."),
    (lambda fam, op: fam == "chains" and "group-by" in op,
     "The chain's group-by rebuilds the dense key mapping after the filter, which is most of the "
     "measured time; the filter and the aggregate themselves are each well inside the bar."),
    (lambda fam, op: fam == "join",
     "The GPU hash join (`Sources/ArrowMetal/Kernels/Join.swift`) is reached through `am_join` / "
     "`am.join`: build the table over the right keys, probe the left twice (count, GPU scan, write). "
     "If this row is short of 3x the build side no longer fits in cache and the probe's random reads "
     "into device memory are the cost."),
    (lambda fam, op: fam == "nested",
     "Nested kernels are one thread per row over an offsets buffer; the CPU equivalents are often "
     "metadata-only (a zero-copy child view) and so cannot be beaten by any amount of bandwidth."),
    (lambda fam, op: fam == "group-by" and "count_distinct" in op,
     "Two full GPU radix sorts: one to dictionary-encode the values, one over the packed `(group, "
     "code)` int64 to collapse repeats. The CPU libraries keep a hash set per group instead. A "
     "segmented sort of the values inside each group's counting-sort run, or a per-group hash, is the "
     "fix; the sort of the key column itself is already gone."),
    (lambda fam, op: fam == "group-by" and any(a in op for a in ("list ", "quantile", "median")),
     "The ordering itself is now a counting sort by group id (Sources/ArrowMetal/Kernels/GroupOrder.swift), "
     "so what is left is the gather that materialises the child column plus, at very high cardinality, "
     "one threadgroup per group in the run-concatenation kernel."),
    (lambda fam, op: fam == "group-by",
     "The dense key mapping (`am_group_by_keys`) is rebuilt on every call and, at low cardinality, "
     "costs more than the aggregation itself; the CPU libraries' hash table over a thousand keys sits "
     "in L2. Reusing one `am.group_by([...])` object across aggregates (as "
     "Benchmarks/python_gpu_bench.py does) removes that part."),
    (lambda fam, op: fam == "sort" and op.startswith(("argsort", "sort")),
     "The LSD radix sort makes one full read/write pass over the column per digit, plus a gather for "
     "`sort`; that is several times the column in traffic, so it is bandwidth-bound where Polars' "
     "multi-threaded pattern-defeating sort touches the data far fewer times. Wider digits, or an "
     "in-threadgroup first pass, is the lever."),
    (lambda fam, op: op.startswith(("sin", "cos", "tan", "ln", "divide")) and "float64" in op,
     "Metal has no `double`: float64 transcendentals and division run ArrowMetal's **software "
     "binary64** (Sources/ArrowMetal/Kernels/DoubleTranscendental.swift), tens of integer instructions "
     "per element against one vectorised hardware instruction on the CPU. Compute-bound, not "
     "bandwidth-bound."),
    (lambda fam, op: fam == "group-by" and "10000000 groups" in op,
     "At 10M groups the per-group table no longer fits in threadgroup memory, so every row's update "
     "goes to device memory: the table is tens of megabytes and each row touches a random line of it, "
     "which is a cache miss per row rather than a contended atomic."),
    (lambda fam, op: fam == "decimal",
     "128-bit decimal arithmetic is emulated from 32-bit lanes on the GPU, so each element costs "
     "several instructions where the CPU has native 128-bit adds."),
    (lambda fam, op: fam == "window" and "rolling" in op,
     "Rolling sum/mean are prefix-sum differences: two full passes over the column plus a scan, "
     "against one streaming pass on the CPU."),
]


def diagnose(key, ratio, best_lib, amr, row):
    fam, op, _rows = key
    if amr is not None and amr["status"] == "error":
        return f"ArrowMetal raised: `{amr['note']}`"
    if amr is not None and amr["status"] == "no equivalent":
        return "ArrowMetal has no equivalent operation."
    for pred, cause in CAUSE_HINTS:
        try:
            if pred(fam, op):
                return cause
        except Exception:
            pass
    # memory-bound tie?
    b = row.get(best_lib) if best_lib else None
    if amr and amr.get("gbs") and b and b.get("gbs"):
        # Anything far above the ~400 GB/s the single-pass rows reach is not moving the data at all.
        if b["gbs"] > 800:
            return (f"The baseline is not moving the data: {best_lib} reports "
                    f"{fmt_gbs(b['gbs'])} GB/s, far above the ~400 GB/s a single pass over the data reaches, so it "
                    "returns a view or a metadata change rather than a materialised column, while "
                    "ArrowMetal materialises the result.")
        if 50 < amr["gbs"] <= 800 and 50 < b["gbs"] <= 800:
            return (f"Memory-bound tie: ArrowMetal {fmt_gbs(amr['gbs'])} GB/s vs {best_lib} "
                    f"{fmt_gbs(b['gbs'])} GB/s, both within reach of the ~400 GB/s the single-pass rows "
                    "reach; there is no 3x available to either side on this operation.")
    if amr and amr["wall_ms"] < 0.5:
        return ("Under half a millisecond: the 110-160 us dispatch floor (the latency family's sum "
                "and filter rows at 1,000 rows) is a large share of the measurement.")
    return ("Not yet diagnosed from the kernel; the ArrowMetal path is doing more passes over the "
            "column than the CPU library's fused one.")


def load_csv(path):
    """Reload a results CSV into ROWS / _ORDER so the report can be rebuilt without re-measuring."""
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            record(r["family"], r["op"], int(r["rows"]), r["library"],
                   float(r["wall_ms"]) if r["wall_ms"] else None,
                   float(r["cpu_ms"]) if r["cpu_ms"] else None,
                   float(r["gb_per_s"]) if r["gb_per_s"] else None,
                   int(r["iterations"]), r["status"], r["note"])


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="1M-row smoke run")
    ap.add_argument("--iters", type=int, default=None, help="max repetitions per measurement")
    ap.add_argument("--budget", type=float, default=1.2,
                    help="seconds of repetitions before a measurement stops early")
    ap.add_argument("--sizes", type=str, default=None, help="comma-separated row counts")
    ap.add_argument("--str-sizes", type=str, default=None,
                    help="comma-separated row counts for the string family")
    ap.add_argument("--small-sizes", type=str, default=None,
                    help="comma-separated row counts for the latency family")
    ap.add_argument("--families", type=str, default=None,
                    help="comma-separated subset of families to run")
    ap.add_argument("--no-report", action="store_true", help="write the CSV only")
    ap.add_argument("--report-from", type=str, default=None,
                    help="rebuild the Markdown report from an existing results CSV and exit")
    ap.add_argument("--elapsed-min", type=float, default=0.0,
                    help="with --report-from: the run time to record, in minutes")
    ap.add_argument("--verify", action="store_true",
                    help="run each operation once instead of timing it and assert that every "
                         "parallel idiom returns the same answer as its library's default idiom")
    ap.add_argument("--cores", type=str, default=None, metavar="CSV",
                    help="print the cores-per-idiom summary for an existing results CSV and exit "
                         "(a run always writes one next to its own CSV)")
    args = ap.parse_args()

    global BENCH, VERIFY
    BENCH = Bench(args.iters or (3 if args.quick else 5), args.budget)
    VERIFY = args.verify

    if args.cores:
        load_csv(args.cores)
        print(cores_summary(ROWS))
        return

    if args.report_from:
        load_csv(args.report_from)
        report, shortfalls = build_report(args.report_from, args.elapsed_min * 60.0)
        md_path = (os.path.join(os.path.dirname(args.report_from), "BENCHMARKS_MATRIX_quick.md")
                   if args.quick else os.path.join(ROOT, "docs", "BENCHMARKS_MATRIX.md"))
        with open(md_path, "w") as fh:
            fh.write(report)
        print(f"rebuilt {md_path} from {args.report_from}; "
              f"{len(shortfalls)} of {len(_ORDER)} rows below 3x")
        return

    # --verify checks answers, not speed, so one middling size is the whole job; 1M rows is big
    # enough to exercise every chunk boundary and small enough to run in a couple of minutes.
    if args.sizes:
        sizes = [int(s) for s in args.sizes.split(",")]
    elif VERIFY:
        sizes = [1_000_000]
    elif args.quick:
        sizes = [1_000_000]
    else:
        sizes = [10_000_000, 50_000_000]
    str_sizes = ([int(s) for s in args.str_sizes.split(",")] if args.str_sizes
                 else sizes if VERIFY
                 else [1_000_000] if args.quick else [1_000_000, 10_000_000])
    small_sizes = ([int(s) for s in args.small_sizes.split(",")] if args.small_sizes
                   else [1_000, min(100_000, max(sizes))] if VERIFY
                   else [1_000, 100_000] if args.quick else [1_000, 100_000, 1_000_000])
    wanted = set(args.families.split(",")) if args.families else None

    def want(name):
        return wanted is None or name in wanted

    print(f"ArrowMetal {am.__version__} on {am.device_name()}; polars {pl.__version__} "
          f"({pl.thread_pool_size()} threads), pyarrow {pa.__version__}, pandas {pd.__version__}, "
          f"numpy {np.__version__}")
    print(f"sizes={sizes} string sizes={str_sizes} small sizes={small_sizes} "
          f"iters<={BENCH.iters} budget={BENCH.budget}s\n")

    t_start = time.perf_counter()
    for n in sizes:
        print(f"\n===== {n:,} rows =====")
        d = Data(n)
        sd = StrData(min(n, max(str_sizes)))
        if want("reductions"):
            family_reductions(d, n)
        if want("element-wise"):
            family_elementwise(d, n)
        if want("compare+select"):
            family_select(d, n)
        if want("sort"):
            family_sort(d, n, sd if n == sizes[0] else None)
        if want("group-by"):
            family_groupby(d, n)
        if want("temporal"):
            family_temporal(d, n)
        if want("window"):
            family_window(d, n)
        if want("decimal"):
            family_decimal(d, n)
        if want("nested"):
            family_nested(d, n)
        if want("chains"):
            family_chains(d, n)
        if want("join"):
            family_join(d, n)
        del d, sd
        gc.collect()

    for n in str_sizes:
        print(f"\n===== strings, {n:,} rows =====")
        sd = StrData(n)
        if want("strings"):
            family_strings(sd)
        del sd
        gc.collect()

    if want("latency"):
        print("\n===== latency floor =====")
        family_small(small_sizes)

    elapsed = time.perf_counter() - t_start

    if VERIFY:
        s = VERIFY_STATS
        print(f"\n===== --verify: {s['pass']} parallel idioms answer their default idiom within "
              f"1e-9 relative, {s['fail']} differ, {s['error']} raised, "
              f"{s['skipped']} comparisons deliberately skipped, "
              f"{s['no_idiom']} rows with no parallel idiom to compare ({elapsed:.0f}s) =====")
        for fam, op, lib in VERIFY_SKIPPED:
            print(f"  skipped  {fam:<14} {op:<40} {lib}")
        for fam, op, rws, lib, why in VERIFY_FAILURES:
            print(f"  DIFFERS  {fam:<14} {op:<40} {rws:>9,} {lib:<18} {why}")
        return 0 if not VERIFY_FAILURES else 1

    results_dir = os.path.join(ROOT, "Benchmarks", "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"full_matrix_{datetime.date.today().isoformat()}"
                                         f"{'_quick' if args.quick else ''}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "op", "rows", "library", "wall_ms", "cpu_ms",
                    "gb_per_s", "iterations", "status", "note"])
        for r in ROWS:
            w.writerow([r["family"], r["op"], r["rows"], r["library"],
                        "" if r["wall_ms"] is None else f"{r['wall_ms']:.4f}",
                        "" if r["cpu_ms"] is None else f"{r['cpu_ms']:.4f}",
                        "" if r["gbs"] is None else f"{r['gbs']:.3f}",
                        r["iters"], r["status"], r["note"]])
    print(f"\nwrote {csv_path} ({len(ROWS)} rows, {elapsed / 60:.1f} min)")

    cores_txt = cores_summary(ROWS, context=True)
    cores_path = csv_path[:-4] + "_cores.txt"
    with open(cores_path, "w") as fh:
        fh.write(cores_txt + "\n")
    print()
    print(cores_txt)
    print(f"wrote {cores_path}")

    if not args.no_report:
        report, shortfalls = build_report(csv_path, elapsed)
        md_path = (os.path.join(results_dir, "BENCHMARKS_MATRIX_quick.md") if args.quick
                   else os.path.join(ROOT, "docs", "BENCHMARKS_MATRIX.md"))
        with open(md_path, "w") as fh:
            fh.write(report)
        print(f"wrote {md_path}; {len(shortfalls)} of {len(_ORDER)} rows below 3x")


if __name__ == "__main__":
    sys.exit(main() or 0)
