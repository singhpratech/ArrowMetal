"""A lazy, optimizing query engine on the Apple GPU, with a Polars-shaped API.

    import pyarrow as pa, arrowmetal as am

    q = (am.scan(table)
           .filter(am.col("amount") > 100)
           .group_by("region").agg(am.agg.sum("amount", "total"), am.agg.count("n"))
           .sort("total", descending=True)
           .limit(10))
    print(q.explain())
    q.collect()                      # -> pyarrow.Table

Nothing runs until `collect()`. The plan is serialised as JSON (the grammar is in
`Sources/ArrowMetal/Engine/PlanJSON.swift` and `docs/ENGINE.md`), handed to the Swift engine in one
call, type-checked, optimized — predicate pushdown, projection pruning, filter fusion, constant
folding, CSE, join reordering — lowered to physical operators with maximal fused kernels, and executed
inside one Metal command buffer.

Expressions are the ones `am.col` already builds, so everything in `docs/EXPR.md` works here too.

Getting data onto the GPU costs something, so the Python side does it once. `am.scan(table)` does not
import anything; the first `collect()` imports the columns the *optimized* plan will actually read,
and caches them on the table it scanned. Scanning the same table again — in a loop, in another query —
reuses those imports, so a warm `collect()` pays only for kernels and the result.
"""

import ctypes
import json
import re
import weakref

# Bound by arrowmetal/__init__.py once the library is loaded, to keep this module import-free.
_lib = None
_P = None
_check = None
_MetalArray = None
_Expr = None
_as_expr = None
_col = None
_pa = None
_ArrowMetalError = None
_query_columns = None
_as_pa_array = None


def _bind(ns):
    """Wires this module to the loaded library and the expression types of the parent package."""
    global _lib, _P, _check, _MetalArray, _Expr, _as_expr, _col, _pa, _ArrowMetalError, _query_columns
    global _as_pa_array
    _as_pa_array = ns["as_pa_array"]
    _lib = ns["lib"]
    _P = ns["P"]
    _check = ns["check"]
    _MetalArray = ns["MetalArray"]
    _Expr = ns["Expr"]
    _as_expr = ns["as_expr"]
    _col = ns["col"]
    _pa = ns["pa"]
    _ArrowMetalError = ns["ArrowMetalError"]
    _query_columns = ns["query_columns"]

    _lib.am_plan_source.argtypes = [ctypes.c_char_p, _P, _P, ctypes.c_int64, ctypes.POINTER(_P)]
    _lib.am_plan_source.restype = ctypes.c_int
    _lib.am_plan_source_release.argtypes = [_P]
    _lib.am_plan_run.argtypes = [ctypes.c_char_p, _P, ctypes.c_int64, ctypes.c_int, ctypes.POINTER(_P)]
    _lib.am_plan_run.restype = ctypes.c_int
    _lib.am_plan_explain.argtypes = [ctypes.c_char_p, _P, ctypes.c_int64, ctypes.c_int]
    _lib.am_plan_explain.restype = ctypes.c_char_p
    _lib.am_plan_column_count.argtypes = [_P]
    _lib.am_plan_column_count.restype = ctypes.c_int64
    _lib.am_plan_row_count.argtypes = [_P]
    _lib.am_plan_row_count.restype = ctypes.c_int64
    _lib.am_plan_column_name.argtypes = [_P, ctypes.c_int64]
    _lib.am_plan_column_name.restype = ctypes.c_char_p
    _lib.am_plan_column.argtypes = [_P, ctypes.c_int64, ctypes.POINTER(_P)]
    _lib.am_plan_column.restype = ctypes.c_int
    _lib.am_plan_result_release.argtypes = [_P]


# --------------------------------------------------------------------------------------------------
# Sources: import once, per column, per scanned object
#
# Importing a column is `makeBuffer(bytesNoCopy:)` plus making the pages resident, which measured
# ~6 ms for 240 MB on an M4 Max — three quarters of the kernel time of the query that reads it. Doing
# it inside every `collect()`, as this module first did, also churned the Metal buffer pool: the
# allocation and teardown of a fresh 240 MB import per run cost more than the import itself and made a
# warm 10 ms query take 25 ms. So a scanned object owns its imports and keeps them.


class _Source:
    """The columns of one scanned object, imported into Metal memory at most once each.

    An imported `MetalArray` owns the Arrow C Data Interface release callback of the array it came
    from (`Sources/ArrowMetal/CInterop.swift`), so it keeps the producer's buffers alive by itself and
    is safe to hold after the `pyarrow.Array` it was built from is gone.
    """

    __slots__ = ("names", "_raw", "_cols", "__weakref__")

    def __init__(self, names, arrays):
        self.names = list(names)
        self._raw = dict(zip(self.names, arrays))
        self._cols = {}

    def columns(self, want, rows=None):
        """The named columns as `MetalArray`s, importing the ones not imported yet.

        `rows` asks for a prefix instead of the whole column, and is not cached: it is the cheap
        stand-in a schema check or a warm-up run scans, and it takes the prefix *before* the import so
        neither costs the full column.
        """
        out = []
        for n in want:
            a = self._cols.get(n)
            if a is None:
                a = self._raw.get(n)
                if a is None:
                    raise _ArrowMetalError(f"scan: no column named {n!r}; have {self.names}")
            if isinstance(a, _MetalArray):
                out.append(a if rows is None else a.slice(0, min(rows, len(a))))
                continue
            if rows is not None and hasattr(a, "slice"):
                out.append(_MetalArray.from_arrow(_as_pa_array(a.slice(0, rows))))
                continue
            # A chunked column is rechunked here, once, rather than on the way into every query.
            c = _MetalArray.from_arrow(_as_pa_array(a))
            self._cols[n] = c
            out.append(c if rows is None else c.slice(0, min(rows, len(c))))
        return out


# id(scanned object) -> (weakref to it, dict-contents fingerprint or None, _Source). The weakref
# callback drops the entry when the object dies, so an id is never reused behind a live entry, and the
# identity check below covers the gap between death and callback.
_sources_by_object = {}


def _dict_fingerprint(d):
    return (tuple(d.keys()), tuple(id(v) for v in d.values()))


def _source_for(data):
    """The `_Source` for a scanned object, reusing the one from an earlier `scan` of the same object.

    A `pyarrow` Table/RecordBatch and a Polars DataFrame are immutable, so identity is enough. A dict
    is not, so its keys and value identities are checked before the cache is trusted.
    """
    key = id(data)
    hit = _sources_by_object.get(key)
    fp = _dict_fingerprint(data) if isinstance(data, dict) else None
    if hit is not None and hit[0]() is data and hit[1] == fp:
        return hit[2]
    if isinstance(data, _pa.Table):
        # Keep the chunked columns as they are: rechunking is a copy, and a column no query reads
        # should not be paying for one.
        names, arrays = list(data.column_names), [data.column(i) for i in range(data.num_columns)]
    else:
        names, arrays = _query_columns(data)
    src = _Source(names, arrays)
    try:
        ref = weakref.ref(data, lambda _r, k=key: _sources_by_object.pop(k, None))
    except TypeError:                                  # not weak-referenceable: no caching, no leak
        return src
    _sources_by_object[key] = (ref, fp, src)
    return src


# --------------------------------------------------------------------------------------------------
# Which columns a plan reads
#
# The Swift optimizer already prunes the scan to the columns the plan needs (`projection_pruning` in
# `Optimizer.swift`), but by then the columns are imported. This is the same question asked one step
# earlier, in Python, so a 200-column table scanned for two of them imports two.
#
# It is answered conservatively: a superset is always safe, and anything the analysis does not
# recognise means "every column".

_COL_RE = re.compile(r'\(col "([^"\\]*)"\)')

# Operators whose output schema is exactly what their own expressions name, so nothing above them can
# reach a scan column that the plan text never mentions.
_NARROWING_OPS = frozenset({"select", "aggregate", "group_by"})
# Renaming joins would let a plan name a column (`v_right`) that no scan has, so the mention set is
# not a superset of what the scan needs. Bail out on them.
_OPAQUE_OPS = frozenset({"join", "join_asof"})


def _walk(node, out):
    """Every column name `node` and its inputs mention, and whether they narrow the scan's schema.

    False is both "does not narrow" and "the analysis does not apply"; either way the caller imports
    every column, so the two need not be told apart.
    """
    op = node.get("op")
    if op in _OPAQUE_OPS:
        return False
    for key in ("keys", "exprs"):
        for pair in node.get(key) or []:
            out.add(pair[0])
            if not _add_expr(pair[1], out):
                return False
    for a in node.get("aggs") or []:
        out.add(a[1])
        if len(a) > 2 and a[2] and not _add_expr(a[2], out):
            return False
    if node.get("predicate") is not None and not _add_expr(node["predicate"], out):
        return False
    for key in ("by", "order_by"):
        for pair in node.get(key) or []:
            out.add(pair[0] if isinstance(pair, (list, tuple)) else pair)
    for key in ("subset", "columns", "partition_by"):
        for c in node.get(key) or []:
            out.add(c)
    for spec in node.get("specs") or []:
        if spec.get("name"):
            out.add(spec["name"])
        if spec.get("column"):
            out.add(spec["column"])
        for c in spec.get("partition_by") or []:
            out.add(c)
        for pair in spec.get("order_by") or []:
            out.add(pair[0])
    kids = [node[k] for k in ("input", "left", "right") if node.get(k)] + list(node.get("inputs") or [])
    narrows = op in _NARROWING_OPS
    for kid in kids:
        child = _walk(kid, out)
        if child is False:
            return False
        narrows = narrows or child
    return narrows


def _add_expr(sexpr, out):
    """Column names in one s-expression. False when a name is escaped, which this does not decode."""
    if '\\' in sexpr:
        return False
    out.update(_COL_RE.findall(sexpr))
    return True


def _plan_columns(plan):
    """The column names the plan can read, or None when every column has to be imported."""
    names = set()
    try:
        narrows = _walk(plan, names)
    except (AttributeError, TypeError, IndexError, KeyError):   # a hand-built plan shape: import all
        return None
    return names if narrows else None


# --------------------------------------------------------------------------------------------------
# Aggregations


class Agg:
    """One aggregation in `.agg(...)` or `.select(...)` over a whole table."""

    __slots__ = ("op", "expr", "name")

    def __init__(self, op, expr, name):
        self.op = op
        self.expr = expr
        self.name = name

    def alias(self, name):
        return Agg(self.op, self.expr, name)

    def _json(self):
        return [self.op, self.name, "" if self.expr is None else self.expr.sexpr()]

    def __repr__(self):
        inner = "" if self.expr is None else repr(self.expr)
        return f"{self.op}({inner}) AS {self.name}"


def _agg_expr(e):
    """A bare column name is `col(name)`; anything else is already an expression."""
    if isinstance(e, str):
        return _col(e)
    return _as_expr(e)


def _default_agg_name(op, e):
    if isinstance(e, str):
        return e
    s = getattr(e, "_s", "")
    if s.startswith('(col "') and s.endswith('")'):
        return s[6:-2]
    return op


class _AggNamespace:
    """`am.agg.sum("amount", "total")`, `am.agg.count("n")`, `am.agg.mean(am.col("x") * 2, "m")`."""

    @staticmethod
    def sum(e, name=None):
        return Agg("sum", _agg_expr(e), name or _default_agg_name("sum", e))

    @staticmethod
    def min(e, name=None):
        return Agg("min", _agg_expr(e), name or _default_agg_name("min", e))

    @staticmethod
    def max(e, name=None):
        return Agg("max", _agg_expr(e), name or _default_agg_name("max", e))

    @staticmethod
    def mean(e, name=None):
        return Agg("mean", _agg_expr(e), name or _default_agg_name("mean", e))

    @staticmethod
    def count(name="count", e=None):
        """Rows that reach the aggregate (`count()`), or non-null values of `e`."""
        return Agg("count", None if e is None else _agg_expr(e), name)


agg = _AggNamespace()


def _as_aggs(items):
    out = []
    for x in items:
        if isinstance(x, Agg):
            out.append(x)
        elif isinstance(x, (list, tuple)) and len(x) >= 2:
            e = None if (len(x) < 3 or x[2] is None) else _agg_expr(x[2])
            out.append(Agg(x[0], e, x[1]))
        else:
            raise _ArrowMetalError(f"not an aggregation: {x!r}")
    return out


def _as_named(items, kwargs=None):
    """`[expr.alias(n)]`, `[(n, expr)]`, `"name"` and `name=expr` all become [name, sexpr] pairs."""
    out = []
    for x in items:
        if isinstance(x, str):
            out.append([x, _col(x).sexpr()])
        elif isinstance(x, (list, tuple)) and len(x) == 2:
            out.append([x[0], _as_expr(x[1]).sexpr()])
        else:
            e = _as_expr(x)
            name = getattr(e, "_name", None)
            if name is None:
                s = e.sexpr()
                if s.startswith('(col "') and s.endswith('")'):
                    name = s[6:-2]
            if name is None:
                raise _ArrowMetalError(f"select needs a name: use .alias(...) on {e!r}")
            out.append([name, e.sexpr()])
    for k, v in (kwargs or {}).items():
        out.append([k, _as_expr(v).sexpr()])
    return out


def _as_sort_keys(by, descending):
    if isinstance(by, str):
        by = [by]
    by = list(by)
    if isinstance(descending, bool):
        descending = [descending] * len(by)
    descending = list(descending)
    if len(descending) != len(by):
        raise _ArrowMetalError(f"sort: {len(descending)} descending flags for {len(by)} keys")
    return [[b, bool(d)] for b, d in zip(by, descending)]


# --------------------------------------------------------------------------------------------------
# The frame


_source_counter = [0]

# Rows per source that `LazyFrame.warmup()` runs over: enough that a kernel picking a variant by size
# picks the one a real run will, small enough that the run itself is free.
_WARMUP_ROWS = 4096


class LazyFrame:
    """A query being built. Every method returns a new frame; nothing runs until `collect()`."""

    __slots__ = ("_plan", "_sources")

    def __init__(self, plan, sources):
        self._plan = plan
        self._sources = sources

    # -- construction

    @staticmethod
    def scan(data, name=None, columns=None):
        """Starts a query over `data`. Nothing is imported until `collect()`.

        `columns` restricts the scan up front, for the rare plan whose column use this module cannot
        see (a renaming join over a very wide table); the optimizer prunes the rest by itself.
        """
        if name is None:
            _source_counter[0] += 1
            name = f"t{_source_counter[0]}"
        src = _source_for(data)
        node = {"op": "scan", "source": name}
        if columns is not None:
            node["columns"] = list(columns)
        return LazyFrame(node, {name: src})

    def _with(self, plan, extra_sources=None):
        srcs = dict(self._sources)
        if extra_sources:
            srcs.update(extra_sources)
        return LazyFrame(plan, srcs)

    # -- row / column operators

    def filter(self, predicate):
        return self._with({"op": "filter", "input": self._plan, "predicate": _as_expr(predicate).sexpr()})

    def select(self, *exprs, **named):
        items = exprs[0] if len(exprs) == 1 and isinstance(exprs[0], (list, tuple)) else list(exprs)
        return self._with({"op": "select", "input": self._plan, "exprs": _as_named(items, named)})

    def with_columns(self, *exprs, **named):
        items = exprs[0] if len(exprs) == 1 and isinstance(exprs[0], (list, tuple)) else list(exprs)
        return self._with({"op": "with_columns", "input": self._plan, "exprs": _as_named(items, named)})

    def drop(self, *columns):
        cols = columns[0] if len(columns) == 1 and isinstance(columns[0], (list, tuple)) else list(columns)
        keep = [c for c in self.columns if c not in set(cols)]
        return self.select(keep)

    def rename(self, mapping):
        return self.select([[mapping.get(c, c), _col(c)] for c in self.columns])

    # -- aggregation

    def agg(self, *aggs):
        items = aggs[0] if len(aggs) == 1 and isinstance(aggs[0], (list, tuple)) else list(aggs)
        return self._with({"op": "aggregate", "input": self._plan,
                           "aggs": [a._json() for a in _as_aggs(items)]})

    def group_by(self, *keys, **named):
        items = keys[0] if len(keys) == 1 and isinstance(keys[0], (list, tuple)) else list(keys)
        return _GroupBy(self, _as_named(items, named))

    # -- ordering and slicing

    def sort(self, by, descending=False):
        return self._with({"op": "sort", "input": self._plan, "by": _as_sort_keys(by, descending)})

    def limit(self, n):
        return self._with({"op": "limit", "input": self._plan, "count": int(n), "offset": 0})

    def head(self, n=5):
        return self.limit(n)

    def slice(self, offset, length):
        return self._with({"op": "limit", "input": self._plan, "count": int(length), "offset": int(offset)})

    def unique(self, subset=None):
        if isinstance(subset, str):
            subset = [subset]
        return self._with({"op": "unique", "input": self._plan,
                           "subset": None if subset is None else list(subset)})

    # -- combining

    def join(self, other, on=None, left_on=None, right_on=None, how="inner", suffix="_right"):
        if on is not None:
            left_on = right_on = [on] if isinstance(on, str) else list(on)
        if left_on is None or right_on is None:
            raise _ArrowMetalError("join needs `on` or both `left_on` and `right_on`")
        if isinstance(left_on, str):
            left_on = [left_on]
        if isinstance(right_on, str):
            right_on = [right_on]
        return self._with({"op": "join", "left": self._plan, "right": other._plan,
                           "left_on": list(left_on), "right_on": list(right_on),
                           "how": how, "suffix": suffix}, other._sources)

    def join_asof(self, other, on=None, left_on=None, right_on=None, by=None, by_left=None,
                  by_right=None, strategy="backward", tolerance=None, suffix="_right"):
        if on is not None:
            left_on = right_on = on
        if left_on is None or right_on is None:
            raise _ArrowMetalError("join_asof needs `on` or both `left_on` and `right_on`")
        bl = by_left if by_left is not None else by
        br = by_right if by_right is not None else bl
        if isinstance(bl, str):
            bl = [bl]
        if isinstance(br, str):
            br = [br]
        node = {"op": "join_asof", "left": self._plan, "right": other._plan,
                "left_on": left_on, "right_on": right_on, "strategy": strategy, "suffix": suffix,
                "by": list(bl or []), "by_right": list(br or [])}
        if tolerance is not None:
            node["tolerance"] = int(tolerance)
        return self._with(node, other._sources)

    def concat(self, others):
        if isinstance(others, LazyFrame):
            others = [others]
        srcs = {}
        for o in others:
            srcs.update(o._sources)
        return self._with({"op": "concat", "inputs": [self._plan] + [o._plan for o in others]}, srcs)

    def explode(self, columns):
        if isinstance(columns, str):
            columns = [columns]
        return self._with({"op": "explode", "input": self._plan, "columns": list(columns)})

    # -- windows

    def window(self, specs):
        if isinstance(specs, dict):
            specs = [specs]
        return self._with({"op": "window", "input": self._plan, "specs": list(specs)})

    def _win(self, fn, name, column=None, n=None, partition_by=None, order_by=None, descending=False):
        spec = {"name": name, "fn": fn}
        if column is not None:
            spec["column"] = column
        if n is not None:
            spec["n"] = int(n)
        if partition_by is not None:
            spec["partition_by"] = [partition_by] if isinstance(partition_by, str) else list(partition_by)
        if order_by is not None:
            spec["order_by"] = _as_sort_keys(order_by, descending)
        return self.window(spec)

    def with_row_number(self, name="row_number", partition_by=None, order_by=None, descending=False):
        return self._win("row_number", name, partition_by=partition_by, order_by=order_by, descending=descending)

    def with_rank(self, name="rank", partition_by=None, order_by=None, descending=False, dense=False):
        return self._win("dense_rank" if dense else "rank", name,
                         partition_by=partition_by, order_by=order_by, descending=descending)

    def with_lag(self, column, n=1, name=None, partition_by=None, order_by=None, descending=False):
        return self._win("lag", name or f"{column}_lag", column=column, n=n,
                         partition_by=partition_by, order_by=order_by, descending=descending)

    def with_lead(self, column, n=1, name=None, partition_by=None, order_by=None, descending=False):
        return self._win("lead", name or f"{column}_lead", column=column, n=n,
                         partition_by=partition_by, order_by=order_by, descending=descending)

    def with_cum_sum(self, column, name=None, partition_by=None, order_by=None, descending=False):
        return self._win("cum_sum", name or f"{column}_cum_sum", column=column,
                         partition_by=partition_by, order_by=order_by, descending=descending)

    def with_rolling(self, kind, column, window, name=None, partition_by=None, order_by=None, descending=False):
        if kind not in ("sum", "mean", "min", "max"):
            raise _ArrowMetalError(f"rolling kind must be sum/mean/min/max, got {kind!r}")
        return self._win(f"rolling_{kind}", name or f"{column}_rolling_{kind}", column=column, n=window,
                         partition_by=partition_by, order_by=order_by, descending=descending)

    def with_partition_agg(self, op, column, name=None, partition_by=None):
        return self._win(op, name or f"{column}_{op}", column=column, partition_by=partition_by)

    # -- running

    @property
    def columns(self):
        """Output column names. Runs the plan over zero rows, which type-checks it without work."""
        q = self.limit(0)
        try:
            return list(q.collect(_source_rows=0).column_names)
        except _ArrowMetalError:
            # An operator that cannot run over an empty source (a sliced nested column, say) gets the
            # honest answer instead.
            return list(q.collect().column_names)

    def warmup(self):
        """Compiles every kernel this plan needs, over a short prefix of each source.

        A first `collect()` in a process generates and compiles MSL for the kernels the plan lowers
        to, which measured ~45 ms of a 76 ms cold run at 20M rows on an M4 Max; every later run finds
        them in `MetalContext`'s pipeline cache. Calling `warmup()` moves that cost off the first
        query. Returns self, so it chains: `q.warmup().collect()`.
        """
        try:
            self.collect(_source_rows=_WARMUP_ROWS)
        except _ArrowMetalError:
            pass                                   # warming is best effort; the real run reports
        return self

    def plan_json(self):
        """The serialised plan, exactly as it goes over the C ABI."""
        return json.dumps(self._plan)

    def _source_handles(self, source_rows=None, prune=True):
        """Registers every source this plan scans, importing only the columns the plan can read.

        `source_rows` registers a prefix of each source instead of all of it, which is how `columns`
        type-checks a plan and how `warmup` compiles its kernels without touching the whole table.
        `prune=False` registers the full schema, so `explain` prints the scan against the real table.
        """
        want = _plan_columns(self._plan) if prune else None
        handles, boxes = [], []
        for name, src in self._sources.items():
            names = src.names if want is None else [n for n in src.names if n in want]
            cols = src.columns(names, rows=source_rows)
            n = len(cols)
            harr = (_P * max(n, 1))(*[c._h for c in cols])
            narr = (ctypes.c_char_p * max(n, 1))(*[nm.encode() for nm in names])
            out = _P()
            _check(_lib.am_plan_source(name.encode(), harr, narr, n, ctypes.byref(out)))
            handles.append(out)
            boxes.append(cols)                     # keep the imported columns alive
        return handles, boxes

    def explain(self, optimized=True):
        """The optimized logical plan and the physical plan it lowers to, as Polars prints one.

        The scan is registered against the whole table — every column, every row — so the `n/m
        columns` the plan prints is the optimizer's own pruning measured against the real schema, and
        the cardinality estimates that pick a join order are the real ones. That means `explain()` on
        a wide table imports all of it, where `collect()` imports only the columns it reads.
        """
        handles, _boxes = self._source_handles(prune=False)
        try:
            arr = (_P * max(len(handles), 1))(*[h for h in handles])
            r = _lib.am_plan_explain(self.plan_json().encode(), arr, len(handles), 1 if optimized else 0)
            if r is None:
                _check(1)
            return r.decode()
        finally:
            for h in handles:
                _lib.am_plan_source_release(h)

    def collect(self, optimize=True, _source_rows=None):
        """Runs the query and returns a `pyarrow.Table`."""
        handles, _boxes = self._source_handles(_source_rows)
        out = _P()
        try:
            arr = (_P * max(len(handles), 1))(*[h for h in handles])
            _check(_lib.am_plan_run(self.plan_json().encode(), arr, len(handles),
                                    1 if optimize else 0, ctypes.byref(out)))
        finally:
            for h in handles:
                _lib.am_plan_source_release(h)
        try:
            names, cols = [], []
            for i in range(_lib.am_plan_column_count(out)):
                names.append(_lib.am_plan_column_name(out, i).decode())
                h = _P()
                _check(_lib.am_plan_column(out, i, ctypes.byref(h)))
                cols.append(_MetalArray(h).to_arrow())
            if not names:
                return _pa.table({})
            return _pa.Table.from_arrays(cols, names=names)
        finally:
            _lib.am_plan_result_release(out)

    def to_arrow(self, optimize=True):
        """Alias for `collect()`."""
        return self.collect(optimize=optimize)

    def __repr__(self):
        return f"<am.LazyFrame\n{self.explain()}\n>"


class _GroupBy:
    """The result of `.group_by(...)`; call `.agg(...)` on it."""

    __slots__ = ("_frame", "_keys")

    def __init__(self, frame, keys):
        self._frame = frame
        self._keys = keys

    def agg(self, *aggs):
        items = aggs[0] if len(aggs) == 1 and isinstance(aggs[0], (list, tuple)) else list(aggs)
        return self._frame._with({"op": "group_by", "input": self._frame._plan, "keys": self._keys,
                                  "aggs": [a._json() for a in _as_aggs(items)]})

    def count(self, name="count"):
        return self.agg(agg.count(name))

    def sum(self, column, name=None):
        return self.agg(agg.sum(column, name))

    def mean(self, column, name=None):
        return self.agg(agg.mean(column, name))


def scan(data, name=None, columns=None):
    """Starts a lazy query over a dict of arrays, a pyarrow Table/RecordBatch or a Polars DataFrame.

    Nothing is imported here: the first `collect()` imports the columns the optimized plan reads and
    caches them on `data`, so scanning the same object again costs nothing.
    """
    return LazyFrame.scan(data, name=name, columns=columns)
