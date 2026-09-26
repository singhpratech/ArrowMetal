"""MetalEngine: a Polars execution engine that runs the parts of a lazy plan it can on the GPU.

    import polars as pl, arrowmetal as am

    engine = am.MetalEngine()
    out = lf.collect(engine=engine)        # identical to lf.collect(), parts of it on Metal
    print(engine.last_report)              # which nodes ran where, and why the rest did not

This is tier 4 of docs/POLARS.md. Polars optimises the plan as usual and then hands the optimised
IR to a post-optimisation callback (the hook `pl.GPUEngine` uses for cuDF). The callback walks the
IR, translates every subtree it can into an ArrowMetal plan (`docs/ENGINE.md`, "The plan grammar"),
and replaces each such subtree with a Python function that runs the plan on the GPU and returns a
Polars DataFrame. Everything it does not take stays with Polars' in-memory engine, which also runs
whatever sits above a replaced subtree. So every plan collects: the worst case is plain Polars.

Which subtrees can be taken
---------------------------
A replaced subtree becomes a leaf of the Polars plan (it takes no input), so the only subtrees that
can move are ones whose leaves are all in-memory frames (`DataFrameScan`). Inside such a subtree the
engine takes `Filter`, `Select`, `HStack` (`with_columns`), `SimpleProjection`, `Slice`, `Sort`
(with a pushed-in slice, which ArrowMetal runs as top-k), `GroupBy`, all-aggregate `Select`s,
inner/left/semi/anti `Join`s and `Distinct` (`unique`), over the expressions and dtypes listed in
docs/POLARS.md, "Tier 4". Everything else falls back with
a one-line reason in `engine.last_report`.

Which of those it does take is a second decision. By default (`shapes="measured"`) it takes only
the shape classes the benchmark measured ahead of both Polars engines, from the row
count at which they were ahead (`MEASURED_SHAPES`, `min_rows`); `shapes="all"` takes everything it
can translate. docs/POLARS.md says how the defaults were chosen and which results files they come
from.

Results are Polars' results
---------------------------
Where ArrowMetal and Polars differ in semantics the translation emits the Polars answer, and the
cases are pinned by `python/tests/test_polars_engine.py`: float comparisons use Polars' total order
(NaN equals NaN and sorts above every number), `&`/`|` are Kleene, a null `when` condition takes the
`otherwise` branch, `is_in` of a null is null, a sum over no values is 0, a min/max over only NaN is
NaN, Float32 arithmetic keeps subnormals, and every output column is cast to the dtype Polars' own
schema says it has (Polars does not check what an engine returns, so this module does).

The Polars surfaces used
------------------------
`polars.lazyframe.engine._LocalEngine`, its `_name` and `_post_opt_callback`, and the
`NodeTraverser` methods `version`, `get_node`, `set_node`, `get_inputs`, `view_current_node`,
`view_expression`, `get_dtype`, `get_schema` and `set_udf`. They are unstable Polars API; the IR
version this module was written against is `TESTED_IR_VERSION` and a test fails loudly when an
upgrade moves it.
"""
import json
import math
import os
import re
import struct
import time
import warnings
from collections import OrderedDict
from functools import partial

import polars as pl
import pyarrow as pa
from polars._plr import _expr_nodes as _xn
from polars._plr import _ir_nodes as _in  # noqa: F401  (the node classes; tested to exist)
from polars.lazyframe.engine import _LocalEngine

from . import ArrowMetalError, MetalArray, lazy as _lazy

__all__ = ["MetalEngine", "MetalPlanReport", "TESTED_IR_VERSION", "TESTED_POLARS", "MEASURED_SHAPES",
           "clear_import_cache", "import_cache_limit", "import_cache_info"]

# `NodeTraverser.version()` on the polars this module was written and tested against. The major is
# bumped by Polars for incompatible IR changes (renamed nodes, reshaped tuples); a different major
# makes the engine decline every plan. A newer minor only adds nodes, so the engine still runs and
# warns once.
TESTED_IR_VERSION = (14, 7)
TESTED_POLARS = "1.44.1"

# Default size gate, in total rows over a subtree's in-memory inputs. How it was chosen is in
# docs/POLARS.md ("Tier 4", "Which translatable subtrees it runs"): the crossover of argsort and
# lexsort, which a full sort runs, in Benchmarks/results/router_2026-09-24.json, checked against
# Benchmarks/results/polars_engine_bench_2026-09-23_provisional.csv and the quiet rerun,
# Benchmarks/results/polars_engine_bench_2026-09-24.csv.
DEFAULT_MIN_ROWS = 1_000_000

# Shape classes of a translated subtree: "sort" (a full sort whose keys ArrowMetal orders as Polars
# does), "sort_helper_keys" (a full sort that needs extra key columns for that: nulls first on a
# nullable key, or a Float64/Float32 key descending, where NaN goes first), "top_k" (a sort with a slice),
# "group_by_multi" (a GroupBy over two or more keys), "group_by" (one key), "aggregate" (a
# whole-frame aggregate), "join", "distinct"; a subtree with none of them is "rowwise" (filters and
# projections only).
# `shapes="measured"` takes a subtree only when every class in it is a key of MEASURED_SHAPES, none
# of its inputs is a String column (a String column is copied on the way in, and that copy is what the
# sort with a String column spent its time on), and it reads at least the class's row minimum (or
# `min_rows`, whichever is larger). Why these two: docs/POLARS.md, "Tier 4".
MEASURED_SHAPES = {"sort": 0, "sort_helper_keys": 10_000_000}

_HIDDEN = "__arrowmetal_"          # prefix of the helper columns the translation adds and drops


# --------------------------------------------------------------------------------------------------
# dtypes


_INT_TYPES = {pl.Int8: "i8", pl.Int16: "i16", pl.Int32: "i32", pl.Int64: "i64",
              pl.UInt8: "u8", pl.UInt16: "u16", pl.UInt32: "u32", pl.UInt64: "u64"}
_FLOAT_TYPES = {pl.Float32: "f32", pl.Float64: "f64"}
_INT_BITS = {"i8": 8, "i16": 16, "i32": 32, "i64": 64, "u8": 8, "u16": 16, "u32": 32, "u64": 64}
# Columns the plan may carry, sort by and group by. Only numeric and boolean columns can be read by
# a fused expression; String is readable by the string predicates only.
_CARRY_TYPES = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64, pl.Boolean, pl.String, pl.Date, pl.Datetime, pl.Duration,
                pl.Time)


def _code(dt):
    """ArrowMetal's expression type name for a Polars dtype, or None when a fused kernel cannot
    read it."""
    t = type(dt) if not isinstance(dt, type) else dt
    if t in _INT_TYPES:
        return _INT_TYPES[t]
    if t in _FLOAT_TYPES:
        return _FLOAT_TYPES[t]
    if t is pl.Boolean:
        return "bool"
    return None


def _is_int(c):
    return c is not None and c[0] in "iu"


def _is_float(c):
    return c in ("f32", "f64")


def _carryable(dt):
    t = type(dt) if not isinstance(dt, type) else dt
    return t in _CARRY_TYPES


def _lossless(src, dst):
    """Whether casting ArrowMetal type `src` to `dst` can neither fail nor lose a value, so that
    Polars' strict, non-strict and overflowing casts all agree with ArrowMetal's."""
    if src == dst:
        return True
    if src == "bool":
        return dst != "bool"
    if _is_int(src) and _is_int(dst):
        sb, db = _INT_BITS[src], _INT_BITS[dst]
        if src[0] == dst[0]:
            return db >= sb
        return src[0] == "u" and dst[0] == "i" and db > sb
    if _is_int(src) and dst == "f64":
        return True          # round-to-nearest in both engines above 2**53
    if _is_int(src) and dst == "f32":
        return _INT_BITS[src] <= 16
    return src == "f32" and dst == "f64"


def _q(s):
    """A string in the s-expression grammar's quoting."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _unq(s):
    """The inverse of `_q`, on the text between the quotes."""
    return re.sub(r"\\(.)", r"\1", s)


_COL_REF = re.compile(r'\(col "((?:[^"\\]|\\.)*)"\)')


def _to_f32(x):
    """`x` rounded to the nearest float32 (ties to even), as a Python float; past the float32 range
    it is the signed infinity, as a C cast gives."""
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def _reciprocal(value, code):
    """`1 / value` rounded as Polars computes it for a scalar divisor: in `code`'s precision, with
    IEEE results for zero, infinities and NaN.

    Plain Python floats, so the engine needs no NumPy. The float32 case divides in float64 and rounds
    once to float32; float64 carries more than twice float32's 24-bit significand plus two bits, so
    that double rounding gives the correctly rounded float32 quotient.
    """
    v = float(value)
    if code == "f32":
        v = _to_f32(v)
    if math.isnan(v):
        r = math.nan
    elif v == 0.0:
        r = math.copysign(math.inf, v)
    else:
        r = 1.0 / v                     # +-inf -> +-0.0; a subnormal divisor overflows to +-inf
    return _to_f32(r) if code == "f32" else r


def _float_text(f, code):
    """`f` as an ArrowMetal float literal."""
    return f"({code} {repr(f)})"


def _is_minus_one(x):
    """`x` is a scalar literal equal to -1."""
    return (x.is_lit and not x.has_col and not isinstance(x.lit, bool)
            and isinstance(x.lit, (int, float)) and x.lit == -1)


def _num_lit(value, code):
    """A typed literal, or None when `value` is not exactly representable in `code`."""
    if value is None:
        return f"(null {code})"
    if code == "bool":
        return f"(bool {'true' if value else 'false'})"
    if _is_int(code):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if isinstance(value, float):
            if not value.is_integer():
                return None
            value = int(value)
        bits = _INT_BITS[code]
        lo, hi = (0, 2 ** bits - 1) if code[0] == "u" else (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        if not lo <= value <= hi:
            return None
        return f"({code} {value})"
    if _is_float(code):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        f = float(value)
        if isinstance(value, int) and int(f) != value:
            return None
        return _float_text(f, code)
    return None


# --------------------------------------------------------------------------------------------------
# The report


class MetalPlanReport:
    """What the last `collect` ran on Metal and what it left to Polars, and why.

    * `taken` -- one entry per subtree that ran on the GPU: the Polars node id and kind at its top,
      the node kinds inside it, the rows it read, the ArrowMetal plan it ran, and after the run the
      wall time and output rows.
    * `fallbacks` -- one line per node that stayed with Polars for a reason of its own, in the form
      `Kind#id: reason`.
    * `nodes` -- the nodes the placement visited, top down, `(id, kind, "metal" | "polars")`; it
      stops at a node that runs on Metal.
    * `walked` -- every node of the optimised plan, `(id, kind)`.
    """

    def __init__(self, polars_version, ir_version):
        self.polars_version = polars_version
        self.ir_version = ir_version
        self.taken = []
        self.fallbacks = []
        self.nodes = []
        self.walked = []          # every node of the optimised plan, (id, kind), in walk order

    @property
    def ran_on_metal(self):
        return bool(self.taken)

    def kinds_taken(self):
        return [k for t in self.taken for k in t["kinds"]]

    def __repr__(self):
        return (f"MetalPlanReport(taken={[t['root'] for t in self.taken]}, "
                f"fallbacks={len(self.fallbacks)})")

    def __str__(self):
        lines = [f"MetalEngine report (polars {self.polars_version}, IR {self.ir_version})"]
        if not self.taken:
            lines.append("  nothing ran on Metal")
        for t in self.taken:
            ran = ""
            if t.get("seconds") is not None:
                ran = f", ran in {t['seconds'] * 1e3:.2f} ms -> {t['rows_out']:,} rows"
            lines.append(f"  metal:  {t['root']} [{' > '.join(t['kinds'])}] over {t['rows']:,} rows{ran}")
        for f in self.fallbacks:
            lines.append(f"  polars: {f}")
        return "\n".join(lines)


class _Unsupported(Exception):
    pass


# --------------------------------------------------------------------------------------------------
# The translation
#
# `_Col` is what the translator knows about a column: its Polars dtype and whether it can hold a
# null. `_E` is one translated expression: its s-expression, the ArrowMetal type it evaluates to,
# whether it can be null, and, for literals, the Python value.


class _Col:
    """A column of a translated subtree. `ref` is the s-expression that computes it from the
    physical columns of the subtree's plan: `(col "name")` for a column the plan carries, or a whole
    expression for one a `select`/`with_columns` defined and nothing has needed materialised yet.
    Keeping those virtual lets a filter, the next projection and the output share one fused kernel,
    instead of writing every intermediate column."""
    __slots__ = ("dtype", "nullable", "ref")

    def __init__(self, dtype, nullable=True, ref=None):
        self.dtype = dtype
        self.nullable = nullable
        self.ref = ref


def _base(name, dtype, nullable=True):
    return _Col(dtype, nullable, f"(col {_q(name)})")


# Above this length an inlined expression is materialised instead, so a chain of with_columns that
# reuses its own outputs cannot grow the plan text geometrically.
_INLINE_MAX = 2000


class _E:
    __slots__ = ("s", "code", "nullable", "lit", "is_lit", "has_col", "dtype", "col")

    def __init__(self, s, code, nullable, *, dtype=None, lit=None, is_lit=False, has_col=True, col=None):
        self.s = s
        self.code = code
        self.nullable = nullable
        self.dtype = dtype
        self.lit = lit
        self.is_lit = is_lit
        self.has_col = has_col
        self.col = col


class _Sub:
    """A translated subtree: its ArrowMetal plan (None when it cannot run on Metal), its output
    columns, the physical columns its plan produces, its in-memory leaves, their total rows, and
    whether it does any GPU work at all."""

    __slots__ = ("plan", "cols", "phys", "leaves", "rows", "work", "kinds", "classes")

    def __init__(self, plan, cols, phys=(), leaves=(), rows=0, work=False, kinds=(), classes=()):
        self.plan = plan
        self.cols = cols
        self.phys = list(phys)
        self.leaves = list(leaves)
        self.rows = rows
        self.work = work
        self.kinds = list(kinds)
        self.classes = set(classes)         # shape classes inside: see _SHAPE_CLASSES

    def derive(self, plan=None, cols=None, phys=None, work=None, add_class=None):
        return _Sub(self.plan if plan is None else plan, dict(self.cols) if cols is None else cols,
                    self.phys if phys is None else phys, self.leaves, self.rows,
                    self.work if work is None else work, (),
                    self.classes | ({add_class} if add_class else set()))

    def final_plan(self):
        """The plan with one select on top that computes every virtual column and puts the output
        in Polars' order -- or the plan alone when it already produces exactly that."""
        names = list(self.cols)
        if names == self.phys and all(c.ref == f"(col {_q(n)})" for n, c in self.cols.items()):
            return self.plan
        return {"op": "select", "input": self.plan,
                "exprs": [[n, c.ref] for n, c in self.cols.items()]}


_ARITH = {"Plus": "add", "Minus": "sub", "Multiply": "mul"}
_CMP = {"Eq": "eq", "NotEq": "ne", "Lt": "lt", "LtEq": "le", "Gt": "gt", "GtEq": "ge"}
_REGEX_META = set(".^$*+?()[]{}|\\")


class _Translator:
    def __init__(self, nt, report):
        self.nt = nt
        self.report = report
        self.subs = {}                  # node id -> _Sub
        self.hidden = 0
        self.names = set()              # every column name in the plan, so a helper avoids them

    # -- helpers

    def collect_names(self, root):
        """Gathers the column names of every node under `root`, so `_hidden` never picks a name
        the frame already uses."""
        nt, seen, todo = self.nt, set(), [root]
        while todo:
            n = todo.pop()
            if n in seen:
                continue
            seen.add(n)
            nt.set_node(n)
            self.names.update(nt.get_schema())
            todo.extend(nt.get_inputs())

    def _hidden(self, what):
        while True:
            self.hidden += 1
            name = f"{_HIDDEN}{what}{self.hidden}"
            if name not in self.names:
                self.names.add(name)
                return name

    def _fallback(self, n, kind, reason):
        self.report.fallbacks.append(f"{kind}#{n}: {reason}")

    def _schema_cols(self):
        return {name: _base(name, dt) for name, dt in self.nt.get_schema().items()}

    # -- the walk

    def walk(self, n):
        """Translates node `n` and everything under it. Returns its `_Sub` (plan None when the node
        stays with Polars)."""
        nt = self.nt
        nt.set_node(n)
        node = nt.view_current_node()
        kind = type(node).__name__
        inputs = list(nt.get_inputs())
        if n in self.subs:                 # a subplan shared under two Cache nodes: walk it once
            return self.subs[n]
        self.report.walked.append((n, kind))
        kids = [self.walk(i) for i in inputs]
        nt.set_node(n)
        try:
            sub = self._translate(n, kind, node, inputs, kids)
        except _Unsupported as e:
            self._fallback(n, kind, str(e))
            sub = None
        except Exception as e:                        # a translator bug must not fail the query
            self._fallback(n, kind, f"translation error {type(e).__name__}: {e}")
            sub = None
        if sub is None:
            nt.set_node(n)
            sub = _Sub(None, self._schema_cols())
        if sub.plan is not None:
            sub.kinds = [kind] + [k for kid in kids for k in kid.kinds]
        self.subs[n] = sub
        return sub

    def _translate(self, n, kind, node, inputs, kids):
        method = getattr(self, "_node_" + kind, None)
        if method is None:
            raise _Unsupported(f"node {kind} has no ArrowMetal translation")
        if kind != "DataFrameScan" and any(k.plan is None for k in kids):
            # The node itself may be supported; it cannot move without its inputs. Check its own
            # expressions anyway so the report lists every reason at once.
            nt = self.nt
            try:
                if inputs:
                    method(n, node, inputs, kids, check_only=True)
            except _Unsupported as e:
                self._fallback(n, kind, str(e))
            except Exception:           # the input already fell back; its columns may be unknown
                pass
            finally:
                nt.set_node(n)
            return _Sub(None, self._schema_cols())
        return method(n, node, inputs, kids)

    # -- nodes

    def _node_DataFrameScan(self, n, node, inputs, kids, check_only=False):
        if node.selection is not None:
            raise _Unsupported("a DataFrameScan with a pushed-down selection")
        df = pl.DataFrame._from_pydf(node.df)
        names = list(node.projection) if node.projection is not None else list(df.columns)
        cols = {}
        for name in names:
            if "\x00" in name:
                # Polars' own Arrow export panics on a NUL in a column name.
                raise _Unsupported(f"column name {name!r} holds a NUL byte, which Polars cannot "
                                   "export to Arrow")
            s = df.get_column(name)
            if not _carryable(s.dtype):
                raise _Unsupported(f"column {name!r} has dtype {s.dtype}, which the Metal plan "
                                   "does not carry")
            cols[name] = _base(name, s.dtype, s.null_count() > 0)
        src = f"s{n}"
        return _Sub({"op": "scan", "source": src}, cols, names, [(src, df, names)], df.height, False)

    def _node_SimpleProjection(self, n, node, inputs, kids, check_only=False):
        kid = kids[0]
        return kid.derive(cols={name: kid.cols[name] for name in self.nt.get_schema()})

    def _node_Slice(self, n, node, inputs, kids, check_only=False):
        if node.offset < 0:
            raise _Unsupported(f"slice with a negative offset ({node.offset}) counts from the end")
        kid = kids[0]
        plan = {"op": "limit", "input": kid.plan, "count": int(node.len), "offset": int(node.offset)}
        return kid.derive(plan=plan)

    def _node_Filter(self, n, node, inputs, kids, check_only=False):
        kid = kids[0]
        self.nt.set_node(inputs[0])
        pred = self._predicate(node.predicate.node, kid.cols)
        if pred is None:                              # the predicate was only an optimiser hint
            return kid.derive()
        if pred.code != "bool":
            raise _Unsupported("filter predicate is not boolean")
        return kid.derive(plan={"op": "filter", "input": kid.plan, "predicate": pred.s}, work=True)

    def _projected(self, kid, outputs, cols):
        """Adds `outputs` [(name, _E)] to `cols` as virtual columns, materialising any whose
        expression has grown past `_INLINE_MAX`. Returns the plan and its physical columns."""
        plan, phys, extra = kid.plan, list(kid.phys), []
        for name, e in outputs:
            ref = e.s
            if len(ref) > _INLINE_MAX:
                h = self._hidden("expr")
                extra.append([h, ref])
                phys.append(h)
                ref = f"(col {_q(h)})"
            cols[name] = _Col(e.dtype, e.nullable, ref)
        if extra:
            plan = {"op": "with_columns", "input": plan, "exprs": extra}
        return plan, phys

    def _node_HStack(self, n, node, inputs, kids, check_only=False):
        kid = kids[0]
        self.nt.set_node(inputs[0])
        outputs, work = [], False
        for pe in node.exprs:
            name, e, computed = self._output(pe, kid.cols)
            outputs.append((name, e))
            work = work or computed
        cols = dict(kid.cols)
        plan, phys = self._projected(kid, outputs, cols)
        self.nt.set_node(n)
        cols = {name: cols[name] for name in self.nt.get_schema()}
        return kid.derive(plan=plan, cols=cols, phys=phys, work=kid.work or work)

    def _node_Select(self, n, node, inputs, kids, check_only=False):
        kid = kids[0]
        self.nt.set_node(inputs[0])
        exprs = list(node.expr)
        aggs = [self._is_agg(pe.node) for pe in exprs]
        if exprs and all(aggs):
            return self._aggregate(n, kid, [], exprs)
        if any(aggs):
            raise _Unsupported("a select mixing aggregates with row-wise expressions")
        outputs, work, any_col = [], False, False
        for pe in exprs:
            name, e, computed = self._output(pe, kid.cols)
            outputs.append((name, e))
            work = work or computed
            any_col = any_col or e.has_col
        if not any_col:
            raise _Unsupported("a select of literals only (Polars returns one row)")
        cols = {}
        plan, phys = self._projected(kid, outputs, cols)
        return kid.derive(plan=plan, cols=cols, phys=phys, work=kid.work or work)

    def _node_GroupBy(self, n, node, inputs, kids, check_only=False):
        kid = kids[0]
        if node.maintain_order:
            raise _Unsupported("group_by(maintain_order=True): ArrowMetal's group order is not "
                               "first-seen")
        opts = node.options
        if opts.dynamic is not None or opts.rolling is not None:
            raise _Unsupported("rolling and dynamic group_by")
        if opts.slice is not None:
            raise _Unsupported("group_by with a pushed-down slice")
        if node.apply:
            raise _Unsupported("group_by with an apply function")
        self.nt.set_node(inputs[0])
        keys = []
        for pe in node.keys:
            e = self.nt.view_expression(pe.node)
            if type(e).__name__ != "Column":
                raise _Unsupported("group_by key is an expression, not a column")
            c = kid.cols.get(e.name)
            if c is None:
                raise _Unsupported(f"group_by key {e.name!r} not found")
            if _is_float(_code(c.dtype)):
                raise _Unsupported("group_by on a float key (NaN and -0.0 grouping)")
            if pe.output_name != e.name:
                raise _Unsupported("group_by key renamed")
            keys.append(e.name)
        return self._aggregate(n, kid, keys, list(node.aggs))

    def _node_Sort(self, n, node, inputs, kids, check_only=False):
        kid = kids[0]
        stable, nulls_last, descending = node.sort_options
        keys = []
        self.nt.set_node(inputs[0])
        for pe in node.by_column:
            e = self.nt.view_expression(pe.node)
            if type(e).__name__ != "Column":
                raise _Unsupported("sort by an expression, not a column")
            keys.append(e.name)
        nulls_last = list(nulls_last) if len(nulls_last) == len(keys) else [nulls_last[0]] * len(keys)
        descending = list(descending) if len(descending) == len(keys) else [descending[0]] * len(keys)
        slc = node.slice
        if slc is not None and slc[0] < 0:
            raise _Unsupported("sort with a negative slice offset")
        if slc is not None and stable:
            raise _Unsupported("sort(maintain_order=True) with a slice: top-k does not promise "
                               "stable ties")
        hidden, by, helpers = [], [], False
        for name, nl, desc in zip(keys, nulls_last, descending):
            c = kid.cols.get(name)
            if c is None:
                raise _Unsupported(f"sort key {name!r} not found")
            if not _carryable(c.dtype):
                raise _Unsupported(f"sort key {name!r} has dtype {c.dtype}")
            ref = c.ref
            if c.nullable and not nl:
                # ArrowMetal puts nulls last in both directions; a validity key in front puts
                # them first.
                h = self._hidden("valid")
                hidden.append([h, f"(is_valid {ref})"])
                helpers = True
                by.append([h, False])
            if desc and _is_float(_code(c.dtype)):
                # Polars sorts NaN above every number in both directions; ArrowMetal's descending
                # sort puts it after the numbers. A NaN key in front restores Polars' order.
                h = self._hidden("nan")
                hidden.append([h, f"(ne {ref} {ref})"])
                helpers = True
                by.append([h, True])
            m = _COL_REF.fullmatch(ref)
            if m is not None and _unq(m.group(1)) in kid.phys:
                by.append([_unq(m.group(1)), bool(desc)])
            else:                                     # a virtual column: materialise the key
                h = self._hidden("key")
                hidden.append([h, ref])
                by.append([h, bool(desc)])
        plan, phys = kid.plan, list(kid.phys)
        if hidden:
            plan = {"op": "with_columns", "input": plan, "exprs": hidden}
            phys += [h for h, _ in hidden]
        plan = {"op": "sort", "input": plan, "by": by}
        if slc is not None:
            plan = {"op": "limit", "input": plan, "count": int(slc[1]), "offset": int(slc[0])}
        return kid.derive(plan=plan, phys=phys, work=True,
                          add_class="top_k" if slc is not None else
                          ("sort_helper_keys" if helpers else "sort"))

    def _key_columns(self, input_id, exprs, cols, what):
        self.nt.set_node(input_id)
        names = []
        for pe in exprs:
            e = self.nt.view_expression(pe.node)
            if type(e).__name__ != "Column":
                raise _Unsupported(f"{what} key is an expression, not a column")
            c = cols.get(e.name)
            if c is None:
                raise _Unsupported(f"{what} key {e.name!r} not found")
            if _is_float(_code(c.dtype)):
                raise _Unsupported(f"{what} on a float key (NaN and -0.0 equality)")
            names.append(e.name)
        return names

    def _node_Join(self, n, node, inputs, kids, check_only=False):
        how, nulls_equal, slc, suffix, coalesce, maintain = node.options
        if not isinstance(how, str) or how not in ("Inner", "Left", "Semi", "Anti"):
            raise _Unsupported(f"{how if isinstance(how, str) else how[0]} join (inner, left, semi "
                               "and anti are taken)")
        if nulls_equal:
            raise _Unsupported("join(nulls_equal=True): ArrowMetal's null keys never match")
        if slc is not None:
            raise _Unsupported("join with a pushed-down slice")
        if maintain != "none":
            raise _Unsupported(f"join(maintain_order={maintain!r})")
        left, right = self.subs[node.input_left], self.subs[node.input_right]
        lkeys = self._key_columns(node.input_left, node.left_on, left.cols, "join")
        rkeys = self._key_columns(node.input_right, node.right_on, right.cols, "join")
        for lk, rk in zip(lkeys, rkeys):
            if left.cols[lk].dtype != right.cols[rk].dtype:
                raise _Unsupported(f"join keys {lk!r} and {rk!r} differ in dtype")
        # ArrowMetal's output columns (LogicalPlan.swift, `.join`): the left columns, then each right
        # column except a key whose left partner has the same name, renamed with the suffix when the
        # name is taken (JoinExtra.swift `uniqueName`).
        meta = [(name, c.dtype, c.nullable) for name, c in left.cols.items()]
        if how in ("Inner", "Left"):
            taken = [m[0] for m in meta]
            for name, c in right.cols.items():
                if name in rkeys and lkeys[rkeys.index(name)] == name:
                    continue
                out, k = name, 2
                if out in taken:
                    out = name + suffix
                    while out in taken:
                        out, k = f"{name}{suffix}{k}", k + 1
                taken.append(out)
                meta.append((out, c.dtype, c.nullable or how == "Left"))
        by_name = {m[0]: m for m in meta}
        self.nt.set_node(n)
        cols = {}
        for name, dt in self.nt.get_schema().items():
            m = by_name.get(name)
            if m is None or m[1] != dt:
                raise _Unsupported(f"join output column {name!r} is not one ArrowMetal's join names "
                                   "that way (coalesce, suffix)")
            cols[name] = _base(name, dt, m[2])
        plan = {"op": "join", "left": left.final_plan(), "right": right.final_plan(),
                "left_on": lkeys, "right_on": rkeys, "how": how.lower(), "suffix": suffix}
        return _Sub(plan, cols, [m[0] for m in meta], left.leaves + right.leaves,
                    left.rows + right.rows, True, (), left.classes | right.classes | {"join"})

    def _node_Distinct(self, n, node, inputs, kids, check_only=False):
        keep, subset, maintain, slc = node.options
        if keep not in ("any", "first"):
            raise _Unsupported(f"unique(keep={keep!r}) (any and first are taken: ArrowMetal keeps "
                               "the first row of each group)")
        if maintain:
            raise _Unsupported("unique(maintain_order=True)")
        if slc is not None:
            raise _Unsupported("unique with a pushed-down slice")
        kid = kids[0]
        subset = list(subset) if subset else list(kid.cols)
        for c in subset:
            if c not in kid.cols:
                raise _Unsupported(f"unique subset column {c!r} not found")
            if _is_float(_code(kid.cols[c].dtype)):
                raise _Unsupported("unique over a float column (NaN and -0.0 equality)")
        cols = {name: _base(name, c.dtype, c.nullable) for name, c in kid.cols.items()}
        plan = {"op": "unique", "input": kid.final_plan(), "subset": subset}
        return _Sub(plan, cols, list(cols), kid.leaves, kid.rows, True, (),
                    kid.classes | {"distinct"})

    # -- aggregation (GroupBy, and a Select whose every output is an aggregate)

    def _is_agg(self, i):
        e = self.nt.view_expression(i)
        k = type(e).__name__
        if k == "Alias":
            return self._is_agg(e.expr)
        return k in ("Agg", "Len")

    def _aggregate(self, n, kid, keys, exprs):
        aggs, fix = [], []
        for pe in exprs:
            name = pe.output_name
            want = _code(self.nt.get_dtype(pe.node))
            e = self.nt.view_expression(pe.node)
            while type(e).__name__ == "Alias":
                e = self.nt.view_expression(e.expr)
            k = type(e).__name__
            if want is None:
                raise _Unsupported(f"aggregate {name!r} has dtype {self.nt.get_dtype(pe.node)}")
            if k == "Len":
                aggs.append(["count", name, ""])
                fix.append((name, "i64", want, "count"))
                continue
            if k != "Agg":
                raise _Unsupported("an expression over an aggregate")
            if len(e.arguments) != 1:
                raise _Unsupported(f"aggregate {e.name} with {len(e.arguments)} arguments")
            op = e.name
            if op not in ("sum", "min", "max", "mean", "count"):
                raise _Unsupported(f"aggregate {op} has no ArrowMetal plan operator")
            arg = self._expr(e.arguments[0], kid.cols)
            if arg.code is None or arg.s is None:
                raise _Unsupported(f"aggregate {op} over dtype {arg.dtype}")
            if op == "count":
                if e.options:                           # include nulls: the group's row count
                    aggs.append(["count", name, ""])
                    fix.append((name, "i64", want, "count"))
                elif keys and arg.code in ("f64", "bool"):
                    # ArrowMetal's group-by will not read Float64 or Boolean values even to count
                    # them; the validity bits summed per group are the same count.
                    aggs.append(["sum", name, f"(cast (is_valid {arg.s}) u32)"])
                    fix.append((name, "u64", want, "zero"))
                else:
                    aggs.append(["count", name, arg.s])
                    fix.append((name, "i64", want, "count"))
                continue
            if arg.code == "bool":
                if op != "sum":
                    raise _Unsupported(f"{op} of a boolean")
                arg = _E(f"(cast {arg.s} u32)", "u32", arg.nullable)
            if op == "sum":
                got = "f64" if _is_float(arg.code) else ("u64" if arg.code[0] == "u" else "i64")
                aggs.append(["sum", name, arg.s])
                fix.append((name, got, want, "zero"))
            elif op == "mean":
                s = arg.s
                if arg.code in ("i64", "u64"):
                    # ArrowMetal sums a 64-bit mean in 64-bit integers, which wraps on extreme
                    # values; Polars averages in floating point.
                    s = f"(cast {s} f64)"
                aggs.append(["mean", name, s])
                fix.append((name, "f64", want, None))
            else:
                if keys and arg.code == "f64":
                    raise _Unsupported(f"group_by {op} over Float64 (ArrowMetal's GroupBy min/max "
                                       "kernels take up to 32-bit floats)")
                aggs.append([op, name, arg.s])
                if _is_float(arg.code):
                    # Polars: min/max skip NaN, but a group of only NaN answers NaN. ArrowMetal
                    # answers null (whole frame) or an infinity (per group) there, so count the
                    # non-null and the non-NaN values and decide from the two.
                    cnt, num = self._hidden("count"), self._hidden("count")
                    x = arg.s
                    aggs.append(["count", cnt, x])
                    aggs.append(["count", num, f"(if_else (ne {x} {x}) (null {arg.code}) {x})"])
                    fix.append((name, arg.code, want, ("nan", cnt, num)))
                else:
                    fix.append((name, arg.code, want, None))
        if keys:
            plan = {"op": "group_by", "input": kid.plan,
                    "keys": [[k, kid.cols[k].ref] for k in keys], "aggs": aggs}
        else:
            plan = {"op": "aggregate", "input": kid.plan, "aggs": aggs}
        # The Polars answer from ArrowMetal's, as virtual columns over the aggregate's output: a sum
        # over no values is 0, a min/max over only NaN is NaN, and every dtype is Polars' own. The
        # helper counts stay physical and drop out at the final select.
        by_name = {f[0]: f for f in fix}
        cols = {}
        for name in keys + [pe.output_name for pe in exprs]:
            if name in keys:
                c = kid.cols[name]
                cols[name] = _base(name, c.dtype, c.nullable)
                continue
            _n, got, want, how = by_name[name]
            s = f"(col {_q(name)})"
            nullable = True
            if how == "zero":
                s = f"(fill_null {s} ({got} 0))"
                nullable = False
            elif isinstance(how, tuple):
                cnt, num = f"(col {_q(how[1])})", f"(col {_q(how[2])})"
                s = (f"(if_else (gt {num} (i64 0)) {s} "
                     f"(if_else (gt {cnt} (i64 0)) ({got} nan) (null {got})))")
            elif how == "count":
                nullable = False
            if got != want:
                s = f"(cast {s} {want})"
            cols[name] = _Col(_dtype_of(want), nullable, s)
        phys = keys + [a[1] for a in aggs]
        cls = "group_by_multi" if len(keys) > 1 else ("group_by" if keys else "aggregate")
        return _Sub(plan, cols, phys, kid.leaves, kid.rows, True, (), kid.classes | {cls})

    # -- expressions

    def _output(self, pe, cols):
        """One named output of a Select/HStack: (name, _E, computed)."""
        e = self.nt.view_expression(pe.node)
        while type(e).__name__ == "Alias":
            e = self.nt.view_expression(e.expr)
        if type(e).__name__ == "Column":
            c = cols.get(e.name)
            if c is None:
                raise _Unsupported(f"column {e.name!r} not found")
            return pe.output_name, _E(c.ref, _code(c.dtype), c.nullable, dtype=c.dtype,
                                      col=e.name), False
        x = self._expr(pe.node, cols)
        want = self.nt.get_dtype(pe.node)
        wc = _code(want)
        if x.s is None or wc is None:
            raise _Unsupported(f"output {pe.output_name!r} of dtype {want} cannot be computed on "
                               "Metal (a fused kernel writes numeric and boolean columns only)")
        x = self._to(x, wc)
        x.dtype = want
        return pe.output_name, x, True

    def _to(self, x, code):
        """`x` as ArrowMetal type `code`, through a cast that cannot lose a value."""
        if x.code == code:
            return x
        if x.is_lit:
            s = _num_lit(x.lit, code)
            if s is None:
                raise _Unsupported(f"literal {x.lit!r} is not exact as {code}")
            return _E(s, code, x.nullable, dtype=_dtype_of(code), lit=x.lit, is_lit=True, has_col=False)
        if x.code is None or not _lossless(x.code, code):
            raise _Unsupported(f"cast from {x.code or x.dtype} to {code} may lose values")
        return _E(f"(cast {x.s} {code})", code, x.nullable, dtype=_dtype_of(code), has_col=x.has_col)

    def _predicate(self, i, cols):
        """A filter predicate with Polars' `dynamic_pred` optimiser hints dropped. None when
        nothing but hints was left."""
        e = self.nt.view_expression(i)
        k = type(e).__name__
        if k == "Function" and e.function_data and e.function_data[0] == "dynamic_pred":
            return None
        if k == "BinaryExpr" and e.op in (_xn.Operator.And, _xn.Operator.LogicalAnd):
            l = self._predicate(e.left, cols)
            r = self._predicate(e.right, cols)
            if l is None:
                return r
            if r is None:
                return l
            return self._logic("and_kleene", l, r)
        return self._expr(i, cols)

    def _logic(self, op, l, r):
        if l.code != "bool" or r.code != "bool":
            raise _Unsupported("logical operator over non-boolean operands")
        return _E(f"({op} {l.s} {r.s})", "bool", l.nullable or r.nullable, dtype=pl.Boolean,
                  has_col=l.has_col or r.has_col)

    def _expr(self, i, cols):
        nt = self.nt
        e = nt.view_expression(i)
        k = type(e).__name__
        if k == "Column":
            c = cols.get(e.name)
            if c is None:
                raise _Unsupported(f"column {e.name!r} not found")
            code = _code(c.dtype)
            if code is None and c.dtype != pl.String:
                raise _Unsupported(f"column {e.name!r} of dtype {c.dtype} cannot be read by a "
                                   "fused expression")
            return _E(c.ref, code, c.nullable, dtype=c.dtype, col=e.name)
        if k == "Alias":
            return self._expr(e.expr, cols)
        if k == "Literal":
            return self._literal(e)
        if k == "Cast":
            x = self._expr(e.expr, cols)
            code = _code(e.dtype)
            if code is None or (x.code is None and not (x.is_lit and x.lit is None)):
                raise _Unsupported(f"cast from {x.dtype} to {e.dtype}")
            return self._to(x, code)
        if k == "BinaryExpr":
            return self._binary(i, e, cols)
        if k == "Ternary":
            p = self._expr(e.predicate, cols)
            if p.code != "bool":
                raise _Unsupported("when() condition is not boolean")
            want = _code(nt.get_dtype(i))
            if want is None:
                raise _Unsupported(f"when/then/otherwise of dtype {nt.get_dtype(i)}")
            t = self._to(self._expr(e.truthy, cols), want)
            f = self._to(self._expr(e.falsy, cols), want)
            ps = f"(coalesce {p.s} (bool false))" if p.nullable else p.s
            return _E(f"(if_else {ps} {t.s} {f.s})", want, t.nullable or f.nullable,
                      dtype=_dtype_of(want), has_col=p.has_col or t.has_col or f.has_col)
        if k == "Function":
            return self._function(i, e, cols)
        if k in ("Agg", "Len"):
            raise _Unsupported("an aggregate inside a row-wise expression")
        raise _Unsupported(f"expression {k} has no ArrowMetal translation")

    def _literal(self, e):
        dt = e.dtype
        v = e.value
        code = _code(dt)
        if code is not None:
            if isinstance(v, (pl.Series,)):
                raise _Unsupported("a Series literal")
            s = _num_lit(v, code)
            if s is None:
                raise _Unsupported(f"literal {v!r} of dtype {dt}")
            return _E(s, code, v is None, dtype=dt, lit=v, is_lit=True, has_col=False)
        if dt == pl.Null and v is None:
            # `then(None)`: typed by whatever it meets (`_to` writes `(null TYPE)`).
            return _E(None, None, True, dtype=dt, lit=None, is_lit=True, has_col=False)
        if dt == pl.String and isinstance(v, str):
            return _E(None, None, False, dtype=dt, lit=v, is_lit=True, has_col=False)
        if isinstance(dt, pl.List) and isinstance(v, (list, tuple)):
            return _E(None, None, False, dtype=dt, lit=list(v), is_lit=True, has_col=False)
        raise _Unsupported(f"literal of dtype {dt}")

    def _binary(self, i, e, cols):
        op = e.op
        name = str(op).rsplit(".", 1)[-1]
        l = self._expr(e.left, cols)
        r = self._expr(e.right, cols)
        has_col = l.has_col or r.has_col
        nullable = l.nullable or r.nullable
        if name in _ARITH or name == "TrueDivide":
            want = _code(self.nt.get_dtype(i))
            if want is None or want == "bool":
                raise _Unsupported(f"{name} producing {self.nt.get_dtype(i)}")
            if name == "TrueDivide" and not _is_float(want):
                raise _Unsupported("integer true division")
            if l.code is None or r.code is None:
                raise _Unsupported(f"{name} over {l.dtype} and {r.dtype}")
            if name == "TrueDivide":
                l2, r2 = self._widen(l, want), self._widen(r, want)
            else:
                l2, r2 = self._to(l, want), self._to(r, want)
            am = "div" if name == "TrueDivide" else _ARITH[name]
            if name == "TrueDivide" and not r2.has_col:
                # Polars divides by a scalar as a multiply by its reciprocal, `x * (1 / c)` in the
                # result type, which is not always the correctly rounded `x / c`; the same product
                # here gives Polars' bits. A scalar that is not a plain literal stays with Polars.
                if not r2.is_lit:
                    raise _Unsupported("true division by a scalar expression")
                if r2.lit is not None:
                    am = "mul"
                    rec = _reciprocal(r2.lit, want)
                    r2 = _E(_float_text(rec, "f64"), "f64", False, dtype=pl.Float64, lit=rec,
                            is_lit=True, has_col=False)
                    if want == "f64" and rec != -1.0:
                        return _E(f"(mul {l2.s} {r2.s})", want, nullable, dtype=_dtype_of(want),
                                  has_col=has_col)
            if am == "mul" and _is_float(want) and (_is_minus_one(l2) or _is_minus_one(r2)):
                # Polars multiplies a float by a scalar -1 (either side, or divides by -1) as a
                # negation, which flips a NaN's sign bit where a multiply keeps the input NaN.
                o64 = self._to(r2 if _is_minus_one(l2) else l2, "f64")
                s = f"(negate {o64.s})" if want == "f64" else f"(cast (negate {o64.s}) f32)"
                return _E(s, want, nullable, dtype=_dtype_of(want), has_col=has_col)
            if want == "f32":
                # The GPU's float ALUs flush subnormals to zero; Polars does not. Binary64 (software,
                # correctly rounded) and one rounding back gives the correctly rounded Float32
                # result for + - * / (53 >= 2 * 24 + 2 bits), subnormals included.
                l64, r64 = self._to(l2, "f64"), self._to(r2, "f64")
                return _E(f"(cast ({am} {l64.s} {r64.s}) f32)", want, nullable,
                          dtype=_dtype_of(want), has_col=has_col)
            return _E(f"({am} {l2.s} {r2.s})", want, nullable, dtype=_dtype_of(want), has_col=has_col)
        if name in _CMP:
            return self._compare(name, l, r)
        if name in ("And", "LogicalAnd", "Or", "LogicalOr", "Xor"):
            if l.code == "bool" and r.code == "bool":
                if name == "Xor":            # null if either side is null, as Polars' xor
                    return _E(f"(ne (cast {l.s} u8) (cast {r.s} u8))", "bool", nullable,
                              dtype=pl.Boolean, has_col=has_col)
                return self._logic("and_kleene" if "And" in name else "or_kleene", l, r)
            if _is_int(l.code) and l.code == r.code and name in ("And", "Or", "Xor"):
                bop = {"And": "bit_and", "Or": "bit_or", "Xor": "bit_xor"}[name]
                return _E(f"({bop} {l.s} {r.s})", l.code, nullable, dtype=l.dtype, has_col=has_col)
            raise _Unsupported(f"{name} over {l.dtype} and {r.dtype}")
        raise _Unsupported(f"operator {name} has no ArrowMetal kernel")

    def _widen(self, x, code):
        """For true division: Polars divides in `code` (a float type) whatever the operands."""
        if x.code == code:
            return x
        if x.is_lit:
            return self._to(x, code)
        if x.code is None:
            raise _Unsupported(f"division over {x.dtype}")
        if _is_int(x.code) or x.code == "bool" or (x.code == "f32" and code == "f64"):
            if code == "f32" and _is_int(x.code) and _INT_BITS[x.code] > 16:
                raise _Unsupported(f"division of {x.code} in f32")
            return _E(f"(cast {x.s} {code})", code, x.nullable, dtype=_dtype_of(code), has_col=x.has_col)
        raise _Unsupported(f"division of {x.code} in {code}")

    def _compare(self, name, l, r):
        has_col = l.has_col or r.has_col
        nullable = l.nullable or r.nullable
        # Strings: equality against a literal only.
        if l.dtype == pl.String or r.dtype == pl.String:
            if name not in ("Eq", "NotEq"):
                raise _Unsupported(f"string comparison {name}")
            if l.is_lit and not r.is_lit:
                l, r = r, l
            if l.s is None or l.is_lit or not (r.is_lit and isinstance(r.lit, str)):
                raise _Unsupported("string comparison other than column against a literal")
            s = f"(str_eq {l.s} {_q(r.lit)})"
            if name == "NotEq":
                s = f"(not {s})"
            return _E(s, "bool", l.nullable, dtype=pl.Boolean, has_col=True)
        if l.code is None or r.code is None:
            raise _Unsupported(f"comparison over {l.dtype} and {r.dtype}")
        # Bring both sides to one type without losing a value.
        if l.code != r.code:
            if l.is_lit and not r.is_lit:
                l = self._to(l, r.code)
            elif r.is_lit and not l.is_lit:
                r = self._to(r, l.code)
            elif not (_is_int(l.code) and _is_int(r.code)):
                raise _Unsupported(f"comparison of {l.code} with {r.code}")
            # two integer columns: ArrowMetal's promotion is exact (docs/EXPR.md rules 3-4)
        op = _CMP[name]
        if _is_float(l.code) or _is_float(r.code):
            s = self._float_compare(name, l, r)
        elif l.code == "bool":
            # The fused comparisons take numbers; false < true as 0 < 1, validity unchanged.
            s = f"({op} (cast {l.s} u8) (cast {r.s} u8))"
        else:
            s = f"({op} {l.s} {r.s})"
        return _E(s, "bool", nullable, dtype=pl.Boolean, has_col=has_col)

    @staticmethod
    def _float_compare(name, l, r):
        """Polars compares floats in a total order: NaN equals NaN and is above every number, and
        -0.0 equals 0.0. `(ne x x)` is IEEE's "x is NaN"; every term keeps the operands' validity,
        so the result is null exactly where Polars' is."""
        def nan(x):
            if x.is_lit:
                v = x.lit
                return "true" if (isinstance(v, float) and v != v) else "false"
            return f"(ne {x.s} {x.s})"

        ln, rn = nan(l), nan(r)
        if "true" in (ln, rn):
            raise _Unsupported("comparison against a NaN literal")
        if name in ("Lt", "LtEq"):
            l, r, ln, rn = r, l, rn, ln
            name = "Gt" if name == "Lt" else "GtEq"
        if name == "Gt":
            core = f"(gt {l.s} {r.s})"
            if ln == "false":
                return core
            left = ln if rn == "false" else f"(and {ln} (not {rn}))"
            return f"(or {left} {core})"
        if name == "GtEq":
            core = f"(ge {l.s} {r.s})"
            return core if ln == "false" else f"(or {ln} {core})"
        core = f"(eq {l.s} {r.s})"
        if ln == "false" or rn == "false":
            eq = core
        else:
            eq = f"(or (and {ln} {rn}) {core})"
        return eq if name == "Eq" else f"(not {eq})"

    def _function(self, i, e, cols):
        fd = e.function_data
        head = fd[0] if fd else None
        args = list(e.input)
        B, S = _xn.BooleanFunction, _xn.StringFunction
        if head == B.IsNull or head == B.IsNotNull:
            x = self._expr(args[0], cols)
            if x.s is None:
                raise _Unsupported("is_null of a literal")
            op = "is_null" if head == B.IsNull else "is_valid"
            return _E(f"({op} {x.s})", "bool", False, dtype=pl.Boolean, has_col=x.has_col)
        if head == B.Not:
            x = self._expr(args[0], cols)
            if x.code == "bool":
                return _E(f"(not {x.s})", "bool", x.nullable, dtype=pl.Boolean, has_col=x.has_col)
            if _is_int(x.code):
                return _E(f"(bit_not {x.s})", x.code, x.nullable, dtype=x.dtype, has_col=x.has_col)
            raise _Unsupported(f"not over {x.dtype}")
        if head == B.IsIn:
            if len(fd) > 1 and fd[1]:
                raise _Unsupported("is_in(nulls_equal=True)")
            x = self._expr(args[0], cols)
            lst = self._expr(args[1], cols)
            if not lst.is_lit or not isinstance(lst.lit, list):
                raise _Unsupported("is_in over something other than a literal list")
            values = [v for v in lst.lit if v is not None]
            if not values or len(values) > 64:
                raise _Unsupported(f"is_in over {len(values)} values (1 to 64 are taken)")
            if x.dtype == pl.String and x.s is not None and not x.is_lit:
                if not all(isinstance(v, str) for v in values):
                    raise _Unsupported("is_in of a string column against non-strings")
                terms = [f"(str_eq {x.s} {_q(v)})" for v in values]
                acc = terms[0]
                for t in terms[1:]:
                    acc = f"(or {acc} {t})"
                return _E(acc, "bool", x.nullable, dtype=pl.Boolean, has_col=True)
            if x.code is None or x.code == "bool" or x.is_lit:
                raise _Unsupported(f"is_in over {x.dtype}")
            # Polars matches floats in its total order, so a NaN in the list matches a NaN row;
            # ArrowMetal's is_in compares with IEEE equality, where NaN matches nothing. `(ne x x)`
            # is "x is NaN" and stands in for the NaN values.
            has_nan = _is_float(x.code) and any(isinstance(v, float) and v != v for v in values)
            values = [v for v in values if not (isinstance(v, float) and v != v)]
            lits = [s for s in (_num_lit(v, x.code) for v in values) if s is not None]
            if not lits and not has_nan:
                raise _Unsupported("is_in values not representable in the column's type")
            terms = ([f"(is_in {x.s} {' '.join(lits)})"] if lits else [])
            if has_nan:
                terms.insert(0, f"(ne {x.s} {x.s})")
            s = terms[0] if len(terms) == 1 else f"(or {terms[0]} {terms[1]})"
            if x.nullable:        # Polars: is_in of a null is null; ArrowMetal says false
                s = f"(if_else (is_valid {x.s}) {s} (null bool))"
            return _E(s, "bool", x.nullable, dtype=pl.Boolean, has_col=True)
        if head in (S.StartsWith, S.EndsWith, S.Contains):
            x = self._expr(args[0], cols)
            p = self._expr(args[1], cols)
            if x.dtype != pl.String or x.is_lit or not (p.is_lit and isinstance(p.lit, str)):
                raise _Unsupported("string predicate other than column against a literal")
            if head == S.EndsWith:
                raise _Unsupported("str.ends_with is not in the fused expression grammar")
            if head == S.Contains:
                literal = bool(fd[1]) if len(fd) > 1 else False
                if not literal and any(ch in _REGEX_META for ch in p.lit):
                    raise _Unsupported("str.contains with a regex pattern (ArrowMetal matches "
                                       "literally)")
                op = "contains"
            else:
                op = "starts_with"
            return _E(f"({op} {x.s} {_q(p.lit)})", "bool", x.nullable, dtype=pl.Boolean, has_col=True)
        if head == "fill_null":
            want = _code(self.nt.get_dtype(i))
            if want is None:
                raise _Unsupported(f"fill_null of dtype {self.nt.get_dtype(i)}")
            x = self._to(self._expr(args[0], cols), want)
            v = self._to(self._expr(args[1], cols), want)
            return _E(f"(fill_null {x.s} {v.s})", want, x.nullable and v.nullable,
                      dtype=_dtype_of(want), has_col=x.has_col or v.has_col)
        label = head if isinstance(head, str) else str(head)
        raise _Unsupported(f"function {label} has no ArrowMetal translation")


_CODE_DTYPE = {"i8": pl.Int8, "i16": pl.Int16, "i32": pl.Int32, "i64": pl.Int64, "u8": pl.UInt8,
               "u16": pl.UInt16, "u32": pl.UInt32, "u64": pl.UInt64, "f32": pl.Float32,
               "f64": pl.Float64, "bool": pl.Boolean}


def _dtype_of(code):
    return _CODE_DTYPE[code]()


# --------------------------------------------------------------------------------------------------
# Running a translated subtree


_VALIDATION_ROWS = 64
_validated = {}                     # (plan json, leaf schemas) -> None or the error text
_VALIDATED_MAX = 512


# --------------------------------------------------------------------------------------------------
# The import cache
#
# Importing a column into Metal memory maps its pages (`makeBuffer(bytesNoCopy:)` plus residency),
# and releasing the import unmaps them; on a 50M-row query over three columns the two together cost
# more than the kernels (docs/POLARS.md, "Tier 4"). So an import of a Polars column is kept and reused
# by the next query that reads the same column.
#
# The key is the column's Arrow type, length, offset and the address and size of every buffer. That is
# safe because the cached import holds the exported Arrow array, and through it Polars' own buffer:
# while an entry lives, Polars cannot free that memory (so the address cannot be reused by another
# column) and cannot mutate it in place (Polars copies a shared buffer before writing). Only columns
# whose import did not copy are cached; String columns are converted on every export and never hit.
# Entries are evicted least-recently-used above `import_cache_limit()` bytes.


class _ImportCache:
    def __init__(self):
        self.entries = OrderedDict()          # key -> (MetalArray, nbytes)
        self.bytes = 0
        self.limit = self._default_limit()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _default_limit():
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 4
        except (ValueError, OSError, AttributeError):
            return 4 << 30

    @staticmethod
    def key(arr):
        t = arr.type
        if not (pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_boolean(t)
                or pa.types.is_temporal(t)):
            return None
        return (str(t), len(arr), arr.offset,
                tuple((b.address, b.size) if b is not None else None for b in arr.buffers()))

    def get(self, key):
        hit = self.entries.get(key)
        if hit is None:
            return None
        self.entries.move_to_end(key)
        return hit[0]

    def import_(self, arr):
        """The Metal import of `arr`, from the cache when it holds one."""
        key = self.key(arr)
        if key is not None:
            m = self.get(key)
            if m is not None:
                self.hits += 1
                return m
        self.misses += 1
        m = MetalArray.from_arrow(arr)
        if key is None or self.limit <= 0:
            return m
        # Cache only a no-copy import: a copied one would not follow the source's pages.
        src = [b.address for b in arr.buffers()[1:] if b is not None]
        dst = [b.address for b in m.to_arrow().buffers()[1:] if b is not None]
        if not src or src[-1] not in dst:
            return m
        nbytes = sum(b.size for b in arr.buffers() if b is not None)
        if nbytes > self.limit:
            return m
        self.entries[key] = (m, nbytes)
        self.bytes += nbytes
        while self.bytes > self.limit and self.entries:
            _k, (_m, nb) = self.entries.popitem(last=False)
            self.bytes -= nb
        return m

    def clear(self):
        self.entries.clear()
        self.bytes = 0


_cache = _ImportCache()


def clear_import_cache():
    """Drops every cached import (and with it the Polars buffers the cache kept alive)."""
    _cache.clear()


def import_cache_limit(nbytes=None):
    """The cache's byte budget; with an argument, sets it (0 disables caching) and returns the new
    value. The default is a quarter of physical memory."""
    if nbytes is not None:
        _cache.limit = int(nbytes)
        while _cache.bytes > _cache.limit and _cache.entries:
            _k, (_m, nb) = _cache.entries.popitem(last=False)
            _cache.bytes -= nb
    return _cache.limit


def import_cache_info():
    """`{"entries", "bytes", "limit", "hits", "misses"}` for the import cache."""
    return {"entries": len(_cache.entries), "bytes": _cache.bytes, "limit": _cache.limit,
            "hits": _cache.hits, "misses": _cache.misses}


def _leaf_sources(leaves, rows=None):
    srcs = {}
    for src, df, names in leaves:
        frame = df if rows is None else df.head(rows)
        arrays = []
        for c in names:
            s = frame.get_column(c)
            a = s.to_arrow()
            # A multi-chunk column is concatenated by `to_arrow`, so its buffers are new every time.
            if rows is None and s.n_chunks() == 1 and isinstance(a, pa.Array):
                a = _cache.import_(a)
            arrays.append(a)
        srcs[src] = _lazy._Source(names, arrays)
    return srcs


def _run_plan(plan, leaves, rows=None):
    return _lazy.LazyFrame(plan, _leaf_sources(leaves, rows)).collect()


def _to_polars(table):
    if table.num_columns == 0:
        return pl.DataFrame()
    return pl.from_arrow(table, rechunk=False)


def _check_schema(df, schema, where):
    got = dict(df.schema)
    want = dict(schema)
    if list(got) != list(want) or any(got[k] != want[k] for k in want):
        raise ArrowMetalError(f"ArrowMetal MetalEngine: {where} returned schema {got}, Polars "
                              f"expects {want}")


def _validate(sub, schema):
    """Runs the plan once over a short prefix of each input, so a plan the ArrowMetal engine would
    reject, or one whose output schema is not Polars', falls back at translation time instead of
    failing inside the query. Cached per plan shape, input dtypes and nullability -- but only a
    verdict from a run that had rows to read: over zero rows nothing dispatches, so it proves
    nothing."""
    key = (json.dumps(sub.plan, sort_keys=True),
           tuple((src, tuple((c, str(df.schema[c]), df.get_column(c).null_count() > 0)
                             for c in names)) for src, df, names in sub.leaves),
           tuple((k, str(v)) for k, v in schema.items()))
    if key in _validated:
        return _validated[key]
    had_rows = all(df.height > 0 for _src, df, _names in sub.leaves)
    try:
        out = _to_polars(_run_plan(sub.plan, sub.leaves, _VALIDATION_ROWS))
        _check_schema(out, schema, "the plan")
        verdict = None
    except ArrowMetalError as e:
        verdict = str(e).splitlines()[0] if str(e) else type(e).__name__
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except BaseException as e:      # noqa: BLE001 -- a Polars export panic (pyo3 PanicException)
        # is a BaseException; the plan falls back instead of failing the query.
        text = str(e).splitlines()[0] if str(e) else ""
        verdict = f"the export to ArrowMetal failed: {type(e).__name__}: {text}"
    if had_rows:
        if len(_validated) >= _VALIDATED_MAX:
            _validated.clear()
        _validated[key] = verdict
    return verdict


def _run_subtree(sub, schema, entry, duration_since_start, with_columns, predicate, n_rows,
                 should_time=False):
    """What Polars calls in place of the replaced subtree (`PythonScan` with a Python source):
    `(with_columns, predicate, n_rows, should_time)`, the first three always None here."""
    if with_columns is not None or predicate is not None or n_rows is not None:
        raise ArrowMetalError("ArrowMetal MetalEngine: Polars pushed a projection, predicate or "
                              "row limit into a replaced subtree, which this engine does not expect")
    start = time.monotonic_ns()
    try:
        table = _run_plan(sub.plan, sub.leaves)
    except ArrowMetalError as e:
        raise ArrowMetalError(f"ArrowMetal MetalEngine: the subtree at {entry['root']} failed on "
                              f"Metal: {e}\nplan: {entry['plan']}") from e
    df = _to_polars(table)
    _check_schema(df, schema, f"the subtree at {entry['root']}")
    end = time.monotonic_ns()
    entry["seconds"] = (end - start) / 1e9
    entry["rows_out"] = df.height
    if should_time:
        # Polars wants (start, end, name) relative to the query's start, as cudf-polars' Timer does.
        origin = (entry.get("_callback_ns") or start) - (duration_since_start or 0)
        return df, [(start - origin, end - origin, f"metal:{entry['root']}")]
    return df


# --------------------------------------------------------------------------------------------------
# The callback


_warned_minor = [False]


def execute_with_metal(nt, duration_since_start, *, config):
    """The post-optimisation callback: translate what can run on Metal, replace it with a udf, and
    leave the rest of the plan to Polars. Works by mutating `nt`; returns None."""
    callback_ns = time.monotonic_ns()
    report = MetalPlanReport(pl.__version__, None)
    config.last_report = report
    version = tuple(nt.version())
    report.ir_version = version
    root = nt.get_node()
    if version[0] != TESTED_IR_VERSION[0]:
        report.fallbacks.append(f"plan#{root}: Polars IR version {version} is not the tested "
                                f"{TESTED_IR_VERSION}; the whole plan runs on Polars")
        return _finish(config, report)
    if version[1] > TESTED_IR_VERSION[1] and not _warned_minor[0]:
        _warned_minor[0] = True
        warnings.warn(f"arrowmetal MetalEngine was tested against Polars IR {TESTED_IR_VERSION} "
                      f"(polars {TESTED_POLARS}); this polars reports {version}",
                      stacklevel=2)
    tr = _Translator(nt, report)
    try:
        tr.collect_names(root)
        tr.walk(root)
    finally:
        nt.set_node(root)
    sinks = [n for n, k in report.walked if k == "Sink"]
    if sinks:
        # `sink_*` runs its plan on Polars' streaming engine, which panics on a replaced subtree
        # (`test_a_sink_plan_is_left_to_polars`); a plan with a sink stays whole.
        report.fallbacks.append(f"Sink#{sinks[0]}: a plan that sinks runs on Polars' streaming "
                                "engine, which cannot run a replaced subtree")
        return _finish(config, report)

    # Take the largest translated subtrees that do GPU work, top down.
    chosen, seen = [], set()

    def choose(n):
        if n in seen:                     # a subplan shared under two Cache nodes
            return
        seen.add(n)
        nt.set_node(n)
        kind = type(nt.view_current_node()).__name__
        inputs = list(nt.get_inputs())
        sub = tr.subs.get(n)
        if sub is not None and sub.plan is not None:
            if not sub.work:
                report.nodes.append((n, kind, "polars"))
                return
            classes = sub.classes or {"rowwise"}
            strings = any(df.schema[c] == pl.String for _s, df, names in sub.leaves for c in names)
            need = config.min_rows
            if config.shapes == "measured":
                if not classes <= set(MEASURED_SHAPES) or strings:
                    what = "+".join(sorted(classes)) + (" over a String column" if strings else "")
                    report.fallbacks.append(
                        f"{kind}#{n}: shape {what} is not one the benchmark measured ahead of "
                        "Polars (MetalEngine(shapes='all') takes it)")
                    report.nodes.append((n, kind, "polars"))
                    for i in inputs:
                        choose(i)
                    return
                need = max([need] + [MEASURED_SHAPES[c] for c in classes])
            if sub.rows < need:
                report.fallbacks.append(f"{kind}#{n}: {sub.rows:,} input rows is below the "
                                        f"{need:,} this shape needs")
                report.nodes.append((n, kind, "polars"))
                return
            nt.set_node(n)
            schema = dict(nt.get_schema())
            sub.plan = sub.final_plan()
            sub.phys = list(sub.cols)
            verdict = _validate(sub, schema)
            if verdict is None:
                chosen.append((n, kind, sub, schema))
                return
            report.fallbacks.append(f"{kind}#{n}: the ArrowMetal plan was rejected: {verdict}")
        report.nodes.append((n, kind, "polars"))
        for i in inputs:
            choose(i)

    try:
        choose(root)
        for n, kind, sub, schema in chosen:
            entry = {"root": f"{kind}#{n}", "kinds": sub.kinds, "rows": sub.rows,
                     "plan": json.dumps(sub.plan), "seconds": None, "rows_out": None,
                     "_callback_ns": callback_ns}
            report.taken.append(entry)
            report.nodes.append((n, kind, "metal"))
            nt.set_node(n)
            nt.set_udf(partial(_run_subtree, sub, schema, entry, duration_since_start))
    finally:
        nt.set_node(root)
    return _finish(config, report)


def _finish(config, report):
    if report.fallbacks and os.environ.get("POLARS_VERBOSE", "0") not in ("", "0"):
        warnings.warn("ArrowMetal MetalEngine left these to Polars:\n  "
                      + "\n  ".join(report.fallbacks), pl.exceptions.PerformanceWarning, stacklevel=2)
    if config.raise_on_fail and report.fallbacks:
        seen, lines = set(), []
        for f in report.fallbacks:
            reason = f.split(": ", 1)[-1]
            if reason not in seen:
                seen.add(reason)
                lines.append(f)
        raise NotImplementedError("ArrowMetal MetalEngine: this plan cannot run entirely on Metal:\n  "
                                  + "\n  ".join(lines))
    return None


# --------------------------------------------------------------------------------------------------
# The engine


class MetalEngine(_LocalEngine):
    """A Polars engine that runs what it can of a lazy plan on the Apple GPU through ArrowMetal.

        engine = am.MetalEngine()
        df = lf.collect(engine=engine)
        print(engine.last_report)

    `raise_on_fail=True` raises instead of falling back (Polars reports it as
    `ComputeError: 'cuda' conversion failed: NotImplementedError: ArrowMetal MetalEngine: ...`;
    the `'cuda'` is hardcoded in polars 1.44.1).

    Two gates decide whether a translatable subtree runs on Metal:

    * `shapes="measured"` (the default) takes only the shape classes in `MEASURED_SHAPES`, the ones
      the benchmark measured ahead of both Polars engines; `shapes="all"` takes every
      subtree the translator can express (for testing, or to move work off the CPU cores).
    * `min_rows`: a subtree whose in-memory inputs hold fewer rows stays with Polars. `min_rows=0`
      turns the gate off.

    The name Polars is told is "in-memory": Rust accepts only its four engine names, and the
    in-memory engine is what runs every node this engine leaves to Polars. `plan_engine` and
    `repr` say "metal".
    """

    _name = "in-memory"

    def __init__(self, *, raise_on_fail=False, min_rows=None, shapes="measured", monitoring=None):
        super().__init__(monitoring=monitoring)
        if shapes not in ("measured", "all"):
            raise ValueError(f"shapes must be 'measured' or 'all', got {shapes!r}")
        self.raise_on_fail = bool(raise_on_fail)
        self.min_rows = DEFAULT_MIN_ROWS if min_rows is None else int(min_rows)
        self.shapes = shapes
        self.last_report = None

    @property
    def plan_engine(self):
        return "metal"

    def __repr__(self):
        return (f"MetalEngine(raise_on_fail={self.raise_on_fail!r}, min_rows={self.min_rows!r}, "
                f"shapes={self.shapes!r})")

    def _post_opt_callback(self, *, background, eager):
        if background:
            warnings.warn("MetalEngine does not support background collection; running on Polars' "
                          "in-memory engine.", UserWarning, stacklevel=3)
            return None
        if eager:
            return None
        return partial(execute_with_metal, config=self)

    def profile(self, lf, **kwargs):
        """`lf.profile()` with this engine's callback. polars 1.44.1's `LazyFrame.profile` only
        honours an `engine=` that is a `GPUEngine`, so `lf.profile(engine=MetalEngine())` would
        profile plain Polars; this passes the callback through `profile`'s own
        `post_opt_callback` keyword instead. Subtrees that ran on Metal appear as `metal:<node>`
        rows of the timings frame."""
        return lf.profile(post_opt_callback=partial(execute_with_metal, config=self), **kwargs)
