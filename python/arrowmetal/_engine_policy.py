"""Which translated subtrees `MetalEngine` runs on Metal: the default policy and its overrides.

A translated subtree is described by its shape classes (below), the dtypes of the columns it reads,
its input row count (the rows of its in-memory frames, or of its Parquet file as the footer states
it), where those rows come from and, for a group-by, the number of groups its keys hold as the
engine's probe estimates it. `decide()` answers from those and the crossover tables alone, so the
same description always gets the same decision; the probe itself is deterministic (a fixed-seed
sample of the frame), so the same frame gets the same estimate.

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

A group-by class the sweep measured at several group counts (`_engine_crossovers.GROUPS`) is judged
by the bucket its estimated group count falls in (`group_bucket`) instead of the engine table's row:
the bucket's own crossover, with the same floors. A group count in a bucket with no crossover, or in
no bucket, is not taken; a group-by with no estimate (its keys are not columns of one input frame)
is judged by the engine table's row, which every group count measured has to be ahead for.
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

Decision = namedtuple("Decision", "take reason binding crossover groups", defaults=(None,))
Decision.__doc__ = """`take`: run the subtree on Metal. `reason`: the rule that decided, as the
report prints it. `binding`: the class whose crossover decided (None under an override).
`crossover`: the row count it needed (None when no row count is enough). `groups`: the group-count
estimate the decision used, or None when it used none."""

# The group-count buckets, in order, and the (class, dtype class, input) the sweep measured per bucket.
BUCKETS = tuple(b[0] for b in _table.GROUP_BUCKETS) + (_table.ROWS_BUCKET,)
_BUCKETED = frozenset(k[:3] for k in _table.GROUPS)
NEAR_ROWS = _table.NEAR_ROWS


def node_of(cls):
    return NODE[cls.split(":")[0]]


def is_group_by(cls):
    return node_of(cls) == "group_by"


def dtype_class(dtypes):
    """'string' when a String column is among `dtypes` (Polars dtypes or their names), else
    'numeric'."""
    return "string" if any(str(d) in ("String", "Utf8") for d in dtypes) else "numeric"


def _what(cls, dclass, source):
    text = cls
    if dclass == "string":
        text += " with a String column"
    return text + (" over a Parquet file" if source == "parquet" else "")


def group_bucket(groups, rows):
    """The bucket of a group-by with `groups` groups over `rows` input rows: the name of one of
    `_engine_crossovers.GROUP_BUCKETS` (fewest to most groups), `ROWS_BUCKET` for at least
    rows / NEAR_ROWS groups, or None (no groups, or a count between the last bucket and that)."""
    if groups is None or groups <= 0:
        return None
    if groups * _table.NEAR_ROWS >= rows:
        return _table.ROWS_BUCKET
    for name, lo, hi in _table.GROUP_BUCKETS:
        if lo <= groups <= hi:
            return name
    return None


def bucket_range(bucket, rows):
    """The group counts of a bucket at `rows` input rows, as text."""
    if bucket == _table.ROWS_BUCKET:
        return f"at least {-(-rows // _table.NEAR_ROWS):,} groups (rows / {_table.NEAR_ROWS})"
    for name, lo, hi in _table.GROUP_BUCKETS:
        if name == bucket:
            return f"{lo:,} to {hi:,} groups"
    raise KeyError(bucket)


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


def _floors(cls, best, router):
    """`best` (rows, where) raised to the router table's and the crossover sweep's floors."""
    for op in ROUTER_OPS.get(cls, ()):
        r = (router or {}).get(op)
        if r is not None and r[0] > best[0]:
            best = r
    s = _table.SWEEP.get(cls)
    if s is not None and s["rows"] > best[0]:
        best = (s["rows"], f"crossover sweep, {', '.join(s['labels'])}, {_table.SWEEP_SOURCE}")
    return best


def crossover(cls, dclass, source, router=None):
    """(rows, where) for one class: the largest of the engine table's crossover and the kernels'
    (router table, crossover sweep), or (None, why) when the engine table has no crossover."""
    e = _table.ENGINE.get((cls, dclass, source))
    if e is None:
        return None, f"no measurement of {_what(cls, dclass, source)} ({_table.SOURCE})"
    if e["rows"] is None:
        return None, (f"{_what(cls, dclass, source)} was not measured ahead of Polars up to "
                      f"{e['largest']:,} input rows ({_table.SOURCE})")
    return _floors(cls, (e["rows"], f"engine table, {_table.SOURCE}"), router)


def group_crossover(cls, dclass, source, bucket, router=None):
    """(rows, where) for one group-by class at one group-count bucket, the same way as
    `crossover`, or (None, why)."""
    e = _table.GROUPS.get((cls, dclass, source, bucket))
    at = f"the {bucket}-group bucket"
    if e is None:
        return None, f"{at} has no measurement of {_what(cls, dclass, source)} ({_table.SOURCE})"
    if e["rows"] is None:
        return None, (f"{_what(cls, dclass, source)} at {at} was not measured ahead of Polars up "
                      f"to {e['largest']:,} input rows ({_table.SOURCE})")
    return _floors(cls, (e["rows"], f"engine table, {at}, {_table.SOURCE}"), router)


def group_band(cls, dclass, source, rows, router=None):
    """The buckets in which `cls` is taken at `rows` input rows, fewest groups first."""
    return [b for b in BUCKETS
            if (group_crossover(cls, dclass, source, b, router)[0] or rows + 1) <= rows]


def _band_text(band, rows):
    """'taken at 3,163 to 316,227 groups', adjacent buckets merged."""
    if not band:
        return "taken at no group count"
    runs = []
    for b in band:
        i = BUCKETS.index(b)
        if runs and runs[-1][1] == i - 1 and b != _table.ROWS_BUCKET:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    parts = []
    for lo, hi in runs:
        if BUCKETS[lo] == _table.ROWS_BUCKET:
            parts.append(bucket_range(_table.ROWS_BUCKET, rows))
            continue
        first = _table.GROUP_BUCKETS[lo][1]
        last = _table.GROUP_BUCKETS[hi][2]
        parts.append(f"{first:,} to {last:,} groups")
    return "taken at " + " and ".join(parts)


def _named(cls, names):
    return cls in names or cls.split(":")[0] in names


def group_regions(rows):
    """The group counts at `rows` input rows as [(fewest, most, bucket or None)], in order: each
    bucket below rows / NEAR_ROWS, the counts between the last one and that (no bucket), and the
    rows bucket."""
    edge = -(-rows // _table.NEAR_ROWS)
    out, top = [], 0
    for name, lo, hi in _table.GROUP_BUCKETS:
        if lo >= edge:
            break
        top = min(hi, edge - 1)
        out.append((lo, top, name))
    if top + 1 < edge:
        out.append((top + 1, edge - 1, None))
    out.append((max(edge, 1), float("inf"), _table.ROWS_BUCKET))
    return out


def settled_for(classes, dclass, source, rows, router=None):
    """`settled(lo, hi)`: whether every group count from `lo` to `hi` gets the same answer for these
    group-by classes at `rows` input rows (each bucket that range reaches is taken, or none is).
    The probe samples until its estimate's range is settled."""
    regions = []
    for lo, hi, b in group_regions(rows):
        take = b is not None and all(
            (group_crossover(c, dclass, source, b, router)[0] or rows + 1) <= rows for c in classes)
        regions.append((lo, hi, take))

    def settled(lo, hi):
        return len({t for rlo, rhi, t in regions if rlo <= hi and rhi >= lo}) <= 1
    return settled


def _estimate(groups, settled=None):
    """(groups or None, what the estimate is, its low end, its high end) from `decide`'s `groups`
    argument."""
    if callable(groups):
        groups = groups(settled)
    if groups is None:
        return None, "no group-count estimate", None, None
    if isinstance(groups, tuple):
        if len(groups) == 4:
            return groups
        return groups[0], groups[1], groups[0], groups[0]
    return int(groups), f"{int(groups):,} groups", int(groups), int(groups)


def _band_counts(band, rows):
    """(fewest, most) group count of a band of buckets at `rows` input rows."""
    edge = -(-rows // _table.NEAR_ROWS)
    lims = {b[0]: (b[1], min(b[2], edge - 1)) for b in _table.GROUP_BUCKETS}
    lims[_table.ROWS_BUCKET] = (edge, float("inf"))
    return lims[band[0]][0], lims[band[-1]][1]


def decide(classes, dtypes, rows, source="memory", *, shapes="measured", min_rows=None, router=None,
           groups=None):
    """The decision for one translated subtree. `classes`: its shape classes (empty means
    `rowwise`); `dtypes`: the dtypes of the columns it reads; `rows`: its input rows; `source`:
    'memory' or 'parquet'; `shapes`, `min_rows`: `MetalEngine`'s arguments; `router`: the router
    table's crossovers (`router_crossovers(arrowmetal.router_table())`); `groups`: for a subtree
    with a group-by, its estimated group count -- an int, `(count or None, text)`, None, or a
    callable returning one of those, called only when a group-count bucket decides, with the
    `settled(lo, hi)` predicate of `settled_for` for these classes and rows."""
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
    grouped = [c for c in classes if is_group_by(c) and (c, dclass, source) in _BUCKETED]
    found = [(c,) + crossover(c, dclass, source, router) for c in classes if c not in grouped]
    for c, x, where in found:
        if x is None:
            return Decision(False, where, c, None)
    estimate = None
    note = ""
    if grouped:
        # Below the smallest crossover any group count has, no estimate can take the subtree.
        low = []
        for c in grouped:
            xs = [x for x in (group_crossover(c, dclass, source, b, router)[0] for b in BUCKETS)
                  if x is not None]
            low.append((c, min(xs) if xs else None))
        rest = max((x for _c, x, _w in found), default=0)
        for c, x in low:
            if x is None:
                x, where = crossover(c, dclass, source, router)
                why = where if x is None else f"the {x:,}-row crossover ({where})"
                return Decision(False, f"{_what(c, dclass, source)} was not measured ahead of "
                                       f"Polars at any group count ({why})", c, None)
        c, x = max(low, key=lambda t: (t[1], t[0]))
        if rows < rest:
            c, need, where = max(found, key=lambda f: (f[1], f[0]))
            return Decision(False, f"{rows:,} input rows is below the {need:,}-row crossover for "
                                   f"{_what(c, dclass, source)} ({where})", c, need)
        if rows < x:
            return Decision(False, f"{rows:,} input rows is below the {x:,}-row crossover for "
                                   f"{_what(c, dclass, source)} at every group count measured "
                                   f"({_table.SOURCE})", c, x)
        count, text, lo, hi = _estimate(groups, settled_for(grouped, dclass, source, rows, router))
        if count is not None and lo != hi:
            # A range: taken only when every group count in it is.
            for c in grouped:
                takes = [b is not None and (group_crossover(c, dclass, source, b, router)[0]
                                            or rows + 1) <= rows
                         for rlo, rhi, b in group_regions(rows) if rlo <= hi and rhi >= lo]
                if not all(takes):
                    band = group_band(c, dclass, source, rows, router)
                    side = "outside"
                    if band:
                        first, last = _band_counts(band, rows)
                        side = "below" if hi < first else ("above" if lo > last else "outside")
                    which = ("no count in the estimate's range is" if not any(takes) else
                             "the estimate's range reaches counts that are not")
                    return Decision(False, f"estimated {text}: {side} the measured band for "
                                           f"{_what(c, dclass, source)} at {rows:,} input rows "
                                           f"({_band_text(band, rows)}; {which}; "
                                           f"{_table.SOURCE})", c, None, text)
        if count is None:
            # No estimate: the engine table's row, which every group count measured is in.
            note = f"; {text}"
            found += [(c,) + crossover(c, dclass, source, router) for c in grouped]
            for c, x, where in found:
                if x is None:
                    return Decision(False, where + note, c, None)
        else:
            estimate = text
            bucket = group_bucket(count, rows)
            for c in grouped:
                if bucket is None:
                    x, where = None, (f"no group-count bucket holds {count:,} groups at {rows:,} "
                                      f"input rows ({_table.SOURCE})")
                else:
                    x, where = group_crossover(c, dclass, source, bucket, router)
                if x is None:
                    band = group_band(c, dclass, source, rows, router)
                    if band and bucket is not None and BUCKETS.index(bucket) < BUCKETS.index(band[0]):
                        side = "below"
                    elif band and (bucket is None or BUCKETS.index(bucket) > BUCKETS.index(band[-1])):
                        side = "above"
                    else:
                        side = "outside"
                    return Decision(False, f"estimated {text}: {side} the measured band for "
                                           f"{_what(c, dclass, source)} at {rows:,} input rows "
                                           f"({_band_text(band, rows)}; {where})", c, None, text)
                found.append((c, x, where))
    c, need, where = max(found, key=lambda f: (f[1], f[0]))
    what = _what(c, dclass, source)
    if estimate is not None and c in grouped:
        what += f" at an estimated {estimate}"
    if min_rows is not None and rows < int(min_rows) and int(min_rows) > need:
        return Decision(False, f"{rows:,} input rows is below min_rows={int(min_rows):,} (the "
                               f"crossover for {what} is {need:,} rows, {where}){note}", c,
                        int(min_rows), estimate)
    if rows < need:
        return Decision(False, f"{rows:,} input rows is below the {need:,}-row crossover for "
                               f"{what} ({where}){note}", c, need, estimate)
    return Decision(True, f"{rows:,} input rows is at or above the {need:,}-row crossover for "
                          f"{what} ({where}){note}", c, need, estimate)


def rule_table(router=None):
    """Every (class, dtype class, input) the engine table knows, with the crossover the default
    uses and where it comes from: [(class, dtype class, input, rows or None, where)]."""
    out = []
    for cls, dclass, source in sorted(_table.ENGINE):
        x, where = crossover(cls, dclass, source, router)
        out.append((cls, dclass, source, x, where))
    return out


def group_rule_table(router=None):
    """Every (class, dtype class, input, bucket) the group-count table knows, with its crossover:
    [(class, dtype class, input, bucket, rows or None, where)], buckets fewest groups first."""
    out = []
    for key in sorted(_table.GROUPS, key=lambda k: (k[:3], BUCKETS.index(k[3]))):
        x, where = group_crossover(*key, router)
        out.append(key + (x, where))
    return out
