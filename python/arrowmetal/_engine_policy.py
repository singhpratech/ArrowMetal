"""Which translated subtrees `MetalEngine` runs on Metal: the default policy and its overrides.

A translated subtree is described by its shape classes (below), the dtypes of the columns it reads,
its input row count (the rows of its in-memory frames, or of its Parquet file as the footer states
it) and where those rows come from. `decide()` answers from those and the crossover tables alone:
nothing is timed, sampled or guessed when a plan is placed, so the same description always gets
the same decision.

Shape classes (`polars_engine._Translator` assigns them):

* `rowwise` -- filters and projections only;
* `aggregate:<family>` -- a whole-frame aggregate, `group_by:<family>` -- a group-by over one key,
  `group_by_multi:<family>` -- over two or more; the families are `sum`, `count` (`count` and
  `len`), `mean` and `minmax` (`min` and `max`); `:keys` is a group-by with no aggregate;
* `sort` -- a full sort whose keys ArrowMetal orders as Polars does, `sort_helper_keys` -- one that
  needs extra key columns for that (nulls first on a nullable key, a float key descending),
  `top_k` -- a sort with a slice;
* `join:inner`, `join:left`, `join:semi`, `join:anti`;
* `distinct` -- `unique`.

The measured default (`shapes="measured"`) takes a subtree when its input rows are at or above the
crossover of every class in it, for the subtree's dtype class (`numeric`, or `string` when a String
column is among its inputs) and input (`memory` or `parquet`). A class's crossover is the largest of

* the engine table (`_engine_crossovers.ENGINE`): the whole subtree through `MetalEngine`, cold,
  against the faster of Polars' in-memory and streaming engines, fitted per class from
  `Benchmarks/polars_engine_crossover.py`'s sweep. A class with no measurement for that dtype class
  and input, or one that was not ahead at the largest size measured, is not taken at any size;
* the router table in force (`arrowmetal.router_table()`, the GPU kernel against the CPU loop the
  router would run) for the kernels the class runs (`ROUTER_OPS`);
* the crossover sweep (`_engine_crossovers.SWEEP`, the GPU kernel against the fastest CPU library)
  for the sort kernels, which the router table does not route.
"""
from collections import namedtuple

from . import _engine_crossovers as _table

FAMILIES = ("sum", "count", "mean", "minmax")

# Every class the translator assigns.
CLASSES = (("rowwise",)
           + tuple(f"{kind}:{f}" for kind in ("aggregate", "group_by", "group_by_multi")
                   for f in FAMILIES)
           + ("group_by:keys", "group_by_multi:keys", "sort", "sort_helper_keys", "top_k")
           + tuple(f"join:{how}" for how in ("inner", "left", "semi", "anti"))
           + ("distinct",))

# The node a class belongs to: a sweep case whose classes all belong to one node measures each of
# them (`Benchmarks/polars_engine_crossover.py`).
NODE = {"rowwise": "rowwise", "aggregate": "aggregate", "group_by": "group_by",
        "group_by_multi": "group_by", "sort": "sort", "sort_helper_keys": "sort", "top_k": "sort",
        "join": "join", "distinct": "distinct"}

# The router-table operations (`arrowmetal.router_table()["crossovers"]`) whose kernels a class runs.
ROUTER_OPS = {"rowwise": ("filter",),
              "aggregate:sum": ("sum",), "aggregate:mean": ("sum",),
              "aggregate:minmax": ("min", "max")}
for _kind in ("group_by", "group_by_multi"):
    for _f in FAMILIES + ("keys",):
        ROUTER_OPS[f"{_kind}:{_f}"] = ("group_by_sum",)

DEFAULT_MIN_ROWS_ALL = 1_000_000     # `shapes="all"` without `min_rows=`

Decision = namedtuple("Decision", "take reason binding crossover")
Decision.__doc__ = """`take`: run the subtree on Metal. `reason`: the rule that decided, as the
report prints it. `binding`: the class whose crossover decided (None under an override).
`crossover`: the row count it needed (None when no row count is enough)."""


def node_of(cls):
    return NODE[cls.split(":")[0]]


def dtype_class(dtypes):
    """'string' when a String column is among `dtypes` (Polars dtypes or their names), else
    'numeric'."""
    return "string" if any(str(d) in ("String", "Utf8") for d in dtypes) else "numeric"


def _what(cls, dclass, source):
    text = cls
    if dclass == "string":
        text += " with a String column"
    return text + (" over a Parquet file" if source == "parquet" else "")


def router_crossovers(table):
    """{op: (rows, source)} from `arrowmetal.router_table()`'s dict."""
    if not table:
        return {}
    where = table.get("source") if table.get("shipped") else (table.get("path") or "this machine's table")
    out = {}
    for op, row in (table.get("crossovers") or {}).items():
        rows = row.get("crossover_rows") or (table.get("shipped_crossovers") or {}).get(op)
        if rows:
            out[op] = (int(rows), f"router table, {row.get('label', op)}, {where}")
    return out


def crossover(cls, dclass, source, router=None):
    """(rows, where) for one class: the largest of the engine table's crossover and the kernels'
    (router table, crossover sweep), or (None, why) when the engine table has no crossover."""
    e = _table.ENGINE.get((cls, dclass, source))
    if e is None:
        return None, f"no measurement of {_what(cls, dclass, source)} ({_table.SOURCE})"
    if e["rows"] is None:
        return None, (f"{_what(cls, dclass, source)} was not measured ahead of Polars up to "
                      f"{e['largest']:,} input rows ({_table.SOURCE})")
    best = (e["rows"], f"engine table, {_table.SOURCE}")
    for op in ROUTER_OPS.get(cls, ()):
        r = (router or {}).get(op)
        if r is not None and r[0] > best[0]:
            best = r
    s = _table.SWEEP.get(cls)
    if s is not None and s["rows"] > best[0]:
        best = (s["rows"], f"crossover sweep, {', '.join(s['labels'])}, {_table.SWEEP_SOURCE}")
    return best


def _named(cls, names):
    return cls in names or cls.split(":")[0] in names


def decide(classes, dtypes, rows, source="memory", *, shapes="measured", min_rows=None, router=None):
    """The decision for one translated subtree. `classes`: its shape classes (empty means
    `rowwise`); `dtypes`: the dtypes of the columns it reads; `rows`: its input rows; `source`:
    'memory' or 'parquet'; `shapes`, `min_rows`: `MetalEngine`'s arguments; `router`: the router
    table's crossovers (`router_crossovers(arrowmetal.router_table())`)."""
    classes = sorted(classes) or ["rowwise"]
    rows = int(rows)
    if shapes == "all" or not isinstance(shapes, str):
        if shapes == "all":
            need = DEFAULT_MIN_ROWS_ALL if min_rows is None else int(min_rows)
            rule = "shapes='all'"
        else:
            missing = [c for c in classes if not _named(c, shapes)]
            if missing:
                return Decision(False, f"{missing[0]} is not among the shapes named "
                                       f"({', '.join(sorted(shapes))})", None, None)
            need = 0 if min_rows is None else int(min_rows)
            rule = f"shapes={{{', '.join(sorted(shapes))}}}"
        if rows < need:
            return Decision(False, f"{rows:,} input rows is below min_rows={need:,} ({rule})",
                            None, need)
        return Decision(True, f"{rule} takes every translatable subtree of at least {need:,} rows",
                        None, need)
    dclass = dtype_class(dtypes)
    found = [(c,) + crossover(c, dclass, source, router) for c in classes]
    for c, x, where in found:
        if x is None:
            return Decision(False, where, c, None)
    c, need, where = max(found, key=lambda f: (f[1], f[0]))
    what = _what(c, dclass, source)
    if min_rows is not None and rows < int(min_rows) and int(min_rows) > need:
        return Decision(False, f"{rows:,} input rows is below min_rows={int(min_rows):,} (the "
                               f"crossover for {what} is {need:,} rows, {where})", c, int(min_rows))
    if rows < need:
        return Decision(False, f"{rows:,} input rows is below the {need:,}-row crossover for "
                               f"{what} ({where})", c, need)
    return Decision(True, f"{rows:,} input rows is at or above the {need:,}-row crossover for "
                          f"{what} ({where})", c, need)


def rule_table(router=None):
    """Every (class, dtype class, input) the engine table knows, with the crossover the default
    uses and where it comes from: [(class, dtype class, input, rows or None, where)]."""
    out = []
    for cls, dclass, source in sorted(_table.ENGINE):
        x, where = crossover(cls, dclass, source, router)
        out.append((cls, dclass, source, x, where))
    return out
