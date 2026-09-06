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
"""

import ctypes
import json

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


def _bind(ns):
    """Wires this module to the loaded library and the expression types of the parent package."""
    global _lib, _P, _check, _MetalArray, _Expr, _as_expr, _col, _pa, _ArrowMetalError, _query_columns
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


class LazyFrame:
    """A query being built. Every method returns a new frame; nothing runs until `collect()`."""

    __slots__ = ("_plan", "_sources")

    def __init__(self, plan, sources):
        self._plan = plan
        self._sources = sources

    # -- construction

    @staticmethod
    def scan(data, name=None):
        if name is None:
            _source_counter[0] += 1
            name = f"t{_source_counter[0]}"
        names, arrays = _query_columns(data)
        return LazyFrame({"op": "scan", "source": name}, {name: (names, arrays)})

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
        return list(self.limit(0).collect().column_names)

    def plan_json(self):
        """The serialised plan, exactly as it goes over the C ABI."""
        return json.dumps(self._plan)

    def _source_handles(self):
        handles, boxes = [], []
        for name, (names, arrays) in self._sources.items():
            cols = [a if isinstance(a, _MetalArray) else _MetalArray.from_arrow(a) for a in arrays]
            n = len(cols)
            harr = (_P * max(n, 1))(*[c._h for c in cols])
            narr = (ctypes.c_char_p * max(n, 1))(*[nm.encode() for nm in names])
            out = _P()
            _check(_lib.am_plan_source(name.encode(), harr, narr, n, ctypes.byref(out)))
            handles.append(out)
            boxes.append(cols)                     # keep the imported columns alive
        return handles, boxes

    def explain(self, optimized=True):
        """The optimized logical plan and the physical plan it lowers to, as Polars prints one."""
        handles, _boxes = self._source_handles()
        try:
            arr = (_P * max(len(handles), 1))(*[h for h in handles])
            r = _lib.am_plan_explain(self.plan_json().encode(), arr, len(handles), 1 if optimized else 0)
            if r is None:
                _check(1)
            return r.decode()
        finally:
            for h in handles:
                _lib.am_plan_source_release(h)

    def collect(self, optimize=True):
        """Runs the query and returns a `pyarrow.Table`."""
        handles, _boxes = self._source_handles()
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


def scan(data, name=None):
    """Starts a lazy query over a dict of arrays, a pyarrow Table/RecordBatch or a Polars DataFrame."""
    return LazyFrame.scan(data, name=name)
