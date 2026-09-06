"""The Arrow function-name registry: what ArrowMetal answers to, by exact Apache Arrow name.

This module mirrors the shape of ``pyarrow.compute``'s introspection API — :func:`list_functions`
and :func:`call_function` — over the Arrow v25 compute function list (283 names in the C++ docs plus
the 24 ``hash_*`` grouped aggregates). Every one of those names is present. A name either resolves
to an ArrowMetal call, or it carries an explicit record saying it is not implemented and why.

The single source of truth is :data:`_ROWS` below. :func:`function_table` reads it, the tests read it,
and ``python/tests/function_table_report.py --page`` turns it into the whole of
``docs/ARROW_FUNCTIONS.md``. Nothing here infers a status from anything else, so a row is only ever as
accurate as the line that states it — and the test suite calls every row that claims to work.

Status vocabulary, deliberately strict:

``gpu``
    A Metal kernel does the work. Host code sets up buffers and reads the answer back, nothing more.
``cpu``
    Implemented and reachable through the ArrowMetal API, but the work happens on the host.
``partial``
    Reachable, with a stated limitation — some input types, some options, or a documented
    difference from Arrow's semantics. The note says exactly what.
``missing``
    Not implemented. The note says why (out of scope, or simply unclaimed).
``pending``
    Reserved for a name whose implementation is landing on a branch that has not merged. **No row
    carries this status today**; it stays in the vocabulary so a work-in-progress name has somewhere
    honest to sit rather than being called ``missing``.

The mixed rows are the ones worth reading twice. ``partial`` covers three different things, and each
note says which one applies: an option or an input type Arrow supports and this does not; a result
that deliberately differs from Arrow's (group order, ascending ``unique``, an int32 where Arrow
returns int64); or an evaluation that is genuinely split between the GPU and the host — the Unicode
string predicates, which fall back to the CPU only for the rows carrying a byte >= 0x80, are the
clearest example.
"""
import math

import pyarrow as pa
import pyarrow.compute as pc

from . import (ArrowMetalError, MetalArray, case_when, choose, group_by, lexsort_indices,
               make_struct, pivot_wider, random)

GPU = "gpu"
CPU = "cpu"
PARTIAL = "partial"
MISSING = "missing"
PENDING = "pending"

_RUNNABLE = (GPU, CPU, PARTIAL)


class Function:
    """One Arrow function name and what ArrowMetal does about it."""

    __slots__ = ("name", "section", "status", "swift_file", "method", "notes", "call", "example", "oracle")

    def __init__(self, name, section, status, swift_file, method, notes, call=None, example=None, oracle=None):
        self.name = name
        self.section = section
        self.status = status
        self.swift_file = swift_file
        self.method = method
        self.notes = notes
        self.call = call
        self.example = example
        self.oracle = oracle

    @property
    def runnable(self):
        return self.status in _RUNNABLE

    def __repr__(self):
        return f"<Function {self.name} {self.status} {self.method or '-'}>"


# ---------------------------------------------------------------------------
# Argument plumbing


def _a(x):
    """A MetalArray from anything: one already, or something with the Arrow C array protocol."""
    return x if isinstance(x, MetalArray) else MetalArray.from_arrow(x)


def _out(x):
    """Back to pyarrow (or a plain Python value) so a caller can compare against pyarrow.compute."""
    return x.to_arrow() if isinstance(x, MetalArray) else x


def _u(method, **defaults):
    """A unary call: ``args[0].<method>(**options)``."""
    def run(args, options):
        opts = dict(defaults)
        opts.update(options)
        return _out(getattr(_a(args[0]), method)(**opts))
    return run


def _b(method, **defaults):
    """A binary call: ``args[0].<method>(args[1], **options)``, both operands lifted onto the GPU."""
    def run(args, options):
        opts = dict(defaults)
        opts.update(options)
        return _out(getattr(_a(args[0]), method)(_a(args[1]), **opts))
    return run


def _split(method, *positional):
    """A splitting call. ArrowMetal returns `(offsets, values)`; Arrow returns a list column, so the
    two are stitched back together here, nulls included."""
    def run(args, options):
        offsets, values = getattr(_a(args[0]), method)(*[options[p] for p in positional])
        mask = pc.is_null(args[0]) if isinstance(args[0], pa.Array) else None
        return pa.ListArray.from_arrays(pa.array(offsets.to_arrow().to_pylist(), type=pa.int32()),
                                        values.to_arrow(), mask=mask)
    return run


def _op(symbol):
    """A binary operator on two columns, e.g. ``a + b``."""
    def run(args, options):
        left, right = _a(args[0]), _a(args[1])
        return _out({"+": lambda: left + right, "-": lambda: left - right,
                     "*": lambda: left * right, "/": lambda: left / right,
                     "&": lambda: left & right, "|": lambda: left | right,
                     "==": lambda: left.compare("==", right), "!=": lambda: left.compare("!=", right),
                     "<": lambda: left.compare("<", right), "<=": lambda: left.compare("<=", right),
                     ">": lambda: left.compare(">", right), ">=": lambda: left.compare(">=", right)}[symbol]())
    return run


def _pos(method, *names, **defaults):
    """A call whose options arrive as positional arguments in `names` order."""
    def run(args, options):
        opts = dict(defaults)
        opts.update(options)
        return _out(getattr(_a(args[0]), method)(*[opts[n] for n in names]))
    return run


def _hashk(agg, values=True, **option_names):
    """A grouped aggregate over an **arbitrary** key column, keyed by the group's own key value.

    ArrowMetal's group order is deterministic but is not pyarrow's first-seen order, so both sides
    are returned as `{key: value}` rather than as two parallel arrays. The comparison is then over
    the mapping, which is what Arrow actually specifies.
    """
    def run(args, options):
        gb = group_by([_a(args[0])])
        keys = gb.keys()[0].to_pylist()
        opts = {py: options[arrow] for arrow, py in option_names.items() if arrow in options}
        out = gb.count_all() if not values else getattr(gb, agg)(_a(args[1]), **opts)
        return dict(zip(keys, out.to_arrow().to_pylist()))
    return run


def _oracle_hashk(agg, values=True, unwrap=False):
    """The same grouped aggregate through pyarrow's own `hash_*` kernels, keyed the same way.

    `use_threads=False` because pyarrow refuses to run its ordered aggregators (`first`, `last`,
    `first_last`) multi-threaded.
    """
    def run(args, options):
        table = pa.table({"key": args[0]} if not values else {"key": args[0], "value": args[1]})
        grouped = table.group_by("key", use_threads=False)
        if not values:
            result, column = grouped.aggregate([([], "count_all")]), "count_all"
        else:
            result, column = grouped.aggregate([("value", agg)]), "value_" + agg
        got = result[column].to_pylist()
        if unwrap:
            got = [None if v is None else v[0] for v in got]
        return dict(zip(result["key"].to_pylist(), got))
    return run


def _interval(kind, field):
    """One field of an interval-valued difference, as a plain integer column.

    pyarrow 25 can build a Python type for `month_day_nano_interval` only, so `interval[month]` and
    `interval[day_time]` results are read field by field — see `MetalArray.interval_field`.
    """
    def run(args, options):
        return _out(getattr(_a(args[0]), kind)(_a(args[1])).interval_field(field))
    return run


# Rows whose answer is right but not bit-identical to Arrow's, with the relative tolerance the
# difference justifies. Four families:
#
# * **1e-6** — a float32 evaluation of a float64 column. Only the tdigest sketch is left here: `exp`,
#   the plain logarithms, `sqrt`, `power` and their `_checked` twins used to take that detour and now
#   run in software binary64 (`Kernels/DoublePower.swift`).
# * **1e-15** — the 1-ulp software binary64 routines: `exp`, `ln`, `log2`, `log10`, `power` and their
#   `_checked` twins. `sqrt` is not listed at all, because it is correctly rounded and therefore
#   bit-identical to Arrow's.
# * **1e-14** — the rest of the software binary64 routines: the trigonometric and hyperbolic family,
#   `atan2`, `expm1`, `log1p`, `logb` and `hypot` are within 5 ulp of the host libm.
# * **1e-5** — the *grouped* moments, whose per-group deviations are formed in float32 about a
#   float64 mean (see `Kernels/AggregatesExtra.swift`).
#
# The note on each row says so; these are the numbers the test suite holds them to.
_FLOAT32_EVAL = 1e-6
_BINARY64 = 1e-14
_BINARY64_1ULP = 1e-15
TOLERANCE = {"exp": _BINARY64_1ULP, "ln": _BINARY64_1ULP, "log10": _BINARY64_1ULP,
             "log2": _BINARY64_1ULP, "power": _BINARY64_1ULP,
             "ln_checked": _BINARY64_1ULP, "log10_checked": _BINARY64_1ULP,
             "log2_checked": _BINARY64_1ULP, "power_checked": _BINARY64_1ULP,
             "hash_mean": 1e-12, "mean": 1e-12, "stddev": 1e-12, "variance": 1e-12,
             "skew": 1e-12, "kurtosis": 1e-12, "tdigest": _FLOAT32_EVAL,
             "hash_variance": 1e-5, "hash_stddev": 1e-5, "hash_skew": 1e-5, "hash_kurtosis": 1e-5,
             "hash_tdigest": 1e-5,
             "expm1": _BINARY64, "log1p": _BINARY64, "log1p_checked": _BINARY64,
             "logb": _BINARY64, "logb_checked": _BINARY64, "hypot": _BINARY64,
             "atan2": _BINARY64, "subsecond": _BINARY64}
TOLERANCE.update({name: _BINARY64 for name in
                  ("sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
                   "asinh", "acosh", "atanh", "sin_checked", "cos_checked", "tan_checked",
                   "asin_checked", "acos_checked", "acosh_checked", "atanh_checked")})


# ---------------------------------------------------------------------------
# Example inputs. Every runnable row is exercised on one of these by the test suite.

_INT = pa.array([3, 1, 4, 1, 5, None], type=pa.int64())
_INT_NN = pa.array([3, 1, 4, 1, 5, 9], type=pa.int64())
_INT2 = pa.array([2, 7, 2, 3, 1, None], type=pa.int64())
_UINT = pa.array([3, 1, 4, 1, 5, None], type=pa.uint32())
_FLT = pa.array([3.5, -1.25, 4.0, 0.5, 2.75, None], type=pa.float64())
_FLT_POS = pa.array([3.5, 1.25, 4.0, 0.5, 2.75, None], type=pa.float64())
_FLT2 = pa.array([2.0, 3.0, 0.5, 1.5, 2.5, None], type=pa.float64())
_BOOL = pa.array([True, False, True, None, False, True])
_BOOL2 = pa.array([True, True, False, False, None, True])
_STR = pa.array(["Hello", "wORLD", None, "abc123", "", "Zz"])
_STR_PAD = pa.array(["  pad  ", "x", None, "\tab\n", "", "yy"])
_STR_NUM = pa.array(["12", "-3", "+4", None, "", "123456"])
_IDX = pa.array([2, 0, 1, 5, 4, 3], type=pa.int32())
_TS = pa.array([1_600_000_000_000_000, 1_700_000_123_456_789 // 1000, None, 0],
               type=pa.timestamp("us"))
_TS2 = pa.array([1_600_000_000_000_000 + 86_400_000_000, 1_700_000_123_456, None, 86_400_000_000],
                type=pa.timestamp("us"))
_LIST = pa.array([[1, 2, 3], [4], None, []], type=pa.list_(pa.int64()))
_STRUCT = pa.StructArray.from_arrays([pa.array([1, 2, None]), pa.array(["a", "b", "c"])], ["n", "s"])
_DICT = pa.array(["a", "b", "a", None, "c", "b"]).dictionary_encode()
_FLT_ROUND = pa.array([3.4, -1.2, 4.6, 0.2, 2.75, None], type=pa.float64())
_BIN = pa.array([b"abc", b"z", None, b"", b"xy", b"qrst"], type=pa.binary())
_STR_WS = pa.array(["a b", "x", None, "c  d e", "q"])
_LIST_NE = pa.array([[1, 2, 3], [4, 5], None, [7]], type=pa.list_(pa.int64()))
_F32 = pa.array([3.5, 1.25, 4.0, 0.5, 2.75, None], type=pa.float32())
_F32B = pa.array([2.0, 3.0, 0.5, 1.5, 2.5, None], type=pa.float32())
# A column with content above U+007F, so the Unicode string rows exercise their host path as well as
# their GPU one: a Latin-1 word, a titlecase digraph, Unicode whitespace, fullwidth letters, a Roman
# numeral and a vulgar fraction.
_STR_UNI = pa.array(["Hello", "Ünïcödé", None, "abc123", "",
                     "ǅungla", "   x  ", "ＡＢ", "ⅷ", "½"])
_UNIT = pa.array([0.5, -0.25, 0.75, 0.0, -0.9, None], type=pa.float64())      # |x| <= 1: asin / acos / atanh
_GE1 = pa.array([1.0, 2.5, 4.0, 1.5, 3.25, None], type=pa.float64())          # x >= 1: acosh
_SHIFT = pa.array([1, 2, 3, 1, 0, None], type=pa.int64())
_NDIGITS = pa.array([0, 1, 2, 0, 1, None], type=pa.int32())
_LIST_STR = pa.array([["a", "b"], ["c"], None, []], type=pa.list_(pa.string()))
_MAP = pa.array([[("a", 1), ("b", 2)], [("a", 3)], None, []], type=pa.map_(pa.string(), pa.int64()))
_TS_TZ = pa.array([1_600_000_000_000_000, 1_700_000_123_456_789 // 1000, None, 0],
                  type=pa.timestamp("us", tz="America/New_York"))
_REGEX_IN = pa.array(["a1", "b22", None, "zz"])
# Group-by example: four groups (one of them the null key) of three values each, so every moment and
# every order statistic is defined on every group and pyarrow's t-digest lands on the exact median.
_GKEY_STR = pa.array(["red", "blue", "red", "blue", "red", "blue",
                      "green", "green", "green", None, None, None])
_GVAL_INT = pa.array([10, 20, 30, 40, 50, 60, 70, 80, 95, 5, 15, 25], type=pa.int64())
_GVAL_BOOL = pa.array([True, False, True, True, False, False, True, True, True, None, False, True])
_PKEY = pa.array(["red", "red", "blue", "blue"])
_PPIVOT = pa.array(["x", "y", "x", "y"])
_PVAL = pa.array([1, 2, 3, 4], type=pa.int64())


# ---------------------------------------------------------------------------
# Custom oracles: the handful of names where pyarrow's answer needs shaping, or where ArrowMetal
# deliberately differs and the test checks a property instead.


def _oracle_count_all(args, options):
    return len(args[0])


def _oracle_sorted_unique(args, options):
    """ArrowMetal returns unique values ascending; Arrow returns them by first appearance."""
    return pa.array(sorted(v for v in pc.unique(args[0]).to_pylist() if v is not None))


def _oracle_sorted_value_counts(args, options):
    want = pc.value_counts(args[0]).to_pylist()
    rows = sorted(((r["values"], r["counts"]) for r in want if r["values"] is not None))
    return [{"values": v, "counts": c} for v, c in rows]


def _oracle_first_last(args, options):
    return pc.first_last(args[0], **options).as_py()


def _oracle_min_max(args, options):
    d = pc.min_max(args[0], **options).as_py()
    return (d["min"], d["max"])


def _oracle_mode(args, options):
    row = pc.mode(args[0]).to_pylist()[0]
    return (row["mode"], row["count"])


def _oracle_top_k(args, options):
    return pc.select_k_unstable(args[0], k=options["k"], sort_keys=[("", "descending")])


def _oracle_bottom_k(args, options):
    return pc.select_k_unstable(args[0], k=options["k"], sort_keys=[("", "ascending")])


def _oracle_select_k(args, options):
    order = "descending" if options.get("largest") else "ascending"
    return pc.select_k_unstable(args[0], k=options["k"], sort_keys=[("", order)])


def _oracle_partition_nth(args, options):
    """ArrowMetal answers with the full sort, so check Arrow's actual contract, not Arrow's output:
    the first n positions must name the n smallest values."""
    return "property:partition_nth"


def _oracle_random(args, options):
    """No two implementations share a random stream; the test checks range and determinism."""
    return "property:random"


def _oracle_pivot(args, options):
    return pc.pivot_wider(args[0], args[1], **options).as_py()


def _oracle_make_struct(args, options):
    return pc.make_struct(*args, **options)


def _oracle_quantile(args, options):
    return pc.quantile(args[0], q=options["q"]).to_pylist()[0]


def _oracle_index(args, options):
    return pc.index(args[0], options["value"]).as_py()


def _oracle_positional(fn, *names):
    """pyarrow spells some options as positional arguments; feed them in that order."""
    def run(args, options):
        return getattr(pc, fn)(*args, *[options[n] for n in names])
    return run


def _oracle_keyword(fn, **extra):
    def run(args, options):
        opts = dict(options)
        opts.update(extra)
        return getattr(pc, fn)(*args, **opts)
    return run


def _oracle_extract_regex(args, options):
    """ArrowMetal takes ICU's `(?<name>...)`; pyarrow takes RE2's `(?P<name>...)`."""
    return pc.extract_regex(args[0], options["pattern"].replace("(?<", "(?P<"))


def _oracle_select_property(args, options):
    """`select_k_unstable` and friends are explicitly *unstable*: two implementations may pick
    different rows out of a tie. The test checks the selected values instead of the row numbers."""
    return "property:select_k"


def _oracle_case_when(args, options):
    """pyarrow takes the conditions as one struct column; ArrowMetal takes them as a list."""
    conds = pa.StructArray.from_arrays(list(args[0]), [f"c{i}" for i in range(len(args[0]))])
    return pc.case_when(conds, *args[1], args[2])


def _oracle_choose(args, options):
    return pc.choose(args[0], *args[1])


def _call_case_when(args, options):
    return _out(case_when(args[0], args[1], args[2]))


def _call_choose(args, options):
    return _out(choose(args[0], args[1]))


def _call_extract_regex_span(args, options):
    """ArrowMetal returns `{group: (start, length)}`; Arrow returns a struct of `[start, length]`
    pairs. Reshaped here into Arrow's row-wise form so the two can be compared directly."""
    spans = _a(args[0]).extract_regex_span(options["pattern"])
    columns = {name: (s.to_arrow().to_pylist(), n.to_arrow().to_pylist()) for name, (s, n) in spans.items()}
    rows = []
    for i in range(len(args[0])):
        row = {name: None if s[i] is None else [s[i], n[i]] for name, (s, n) in columns.items()}
        rows.append(None if all(v is None for v in row.values()) else row)
    return rows


def _oracle_extract_regex_span(args, options):
    """pyarrow spells the named groups RE2's way; the span shape is the same otherwise."""
    return pc.extract_regex_span(args[0], options["pattern"].replace("(?<", "(?P<")).to_pylist()


def _oracle_month_interval(args, options):
    """`interval[month]` has no pyarrow Python type, so the months come from the one interval
    difference pyarrow *can* express."""
    return pa.array([None if v is None else v.months
                     for v in pc.month_day_nano_interval_between(args[0], args[1]).to_pylist()],
                    type=pa.int32())


def _oracle_day_time_days(args, options):
    return pc.days_between(args[0], args[1]).cast(pa.int32())


def _oracle_month_day_nano(args, options):
    return pc.month_day_nano_interval_between(args[0], args[1])


def _oracle_group_pivot(args, options):
    table = pa.table({"key": args[0], "pivot": args[1], "value": args[2]})
    result = table.group_by("key", use_threads=False).aggregate(
        [(["pivot", "value"], "pivot_wider", pc.PivotWiderOptions(key_names=options["key_names"]))])
    return dict(zip(result["key"].to_pylist(), result["pivot_value_pivot_wider"].to_pylist()))


def _call_group_pivot(args, options):
    gb = group_by([_a(args[0])])
    values = gb.pivot_wider(_a(args[1]), _a(args[2]), options["key_names"]).to_arrow().to_pylist()
    return dict(zip(gb.keys()[0].to_pylist(), values))


# ---------------------------------------------------------------------------
# The table. (name, section, status, swift file, ArrowMetal method, notes, call, example, oracle)
#
# `example` is (args, options) — what the test suite feeds the row.

_ROWS = [
    # ---- Aggregations ------------------------------------------------------
    ("all", "Aggregations", GPU, "Kernels/Aggregates.swift", "all()",
     "One GPU pass over the values and validity bitmaps. Null-only input gives None, as Arrow does.",
     _u("all"), (( _BOOL,), {})),
    ("any", "Aggregations", GPU, "Kernels/Aggregates.swift", "any()",
     "Same pass as `all`.", _u("any"), ((_BOOL,), {})),
    ("approximate_median", "Aggregations", GPU, "Kernels/Aggregates.swift", "median()",
     "Exact, not approximate: a GPU sort and an interpolated read, not a sketch. Answers are therefore "
     "at least as good as Arrow's.", _u("median"), ((_FLT,), {})),
    ("count", "Aggregations", CPU, "Sources/ArrowMetal/MetalArray.swift", "count(mode)",
     "O(1) metadata: the null count already rides on the column. `mode` accepts only_valid / only_null / all.",
     _u("count"), ((_INT,), {})),
    ("count_all", "Aggregations", CPU, "Kernels/Selection.swift", "count_all()",
     "The row count, valid or not. O(1) metadata; inside an open batch, reading it forces a sync point.",
     _u("count_all"), ((_INT,), {}), _oracle_count_all),
    ("count_distinct", "Aggregations", GPU, "Kernels/Unique.swift", "count_distinct()",
     "The length of `unique()`: one GPU sort and a run scan.", _u("count_distinct"), ((_INT,), {})),
    ("first", "Aggregations", GPU, "Kernels/Aggregates.swift", "first()",
     "A GPU pass takes the atomic minimum valid index, then one host read fetches the value.",
     _u("first"), ((_INT,), {})),
    ("first_last", "Aggregations", GPU, "Kernels/Selection.swift", "first_last()",
     "`first()` and `last()` packaged as the one-row struct Arrow returns. `min_count` is not implemented.",
     lambda args, options: _a(args[0]).first_last(**options).to_arrow().to_pylist()[0],
     ((_INT,), {}), _oracle_first_last),
    ("index", "Aggregations", GPU, "Kernels/Aggregates.swift", "index(value)",
     "Row of the first occurrence, -1 when absent. The value crosses the C ABI as a double, so an "
     "integer above 2^53 cannot be expressed exactly.",
     lambda args, options: _a(args[0]).index(options["value"]), ((_INT,), {"value": 4}), _oracle_index),
    ("kurtosis", "Aggregations", GPU, "Kernels/AggregatesExtra.swift", "kurtosis(biased)",
     "Excess kurtosis, biased by default as Arrow's is: two GPU passes, the same per-type deviation "
     "machinery `variance` uses, so about 1e-15 relative on a float64 column.",
     _u("kurtosis"), ((_FLT,), {})),
    ("last", "Aggregations", GPU, "Kernels/Aggregates.swift", "last()",
     "The atomic maximum valid index, mirroring `first`.", _u("last"), ((_INT,), {})),
    ("max", "Aggregations", GPU, "Kernels/Reductions.swift", "max()",
     "Threadgroup partials, host finalise, no atomics.", _u("max"), ((_INT,), {})),
    ("mean", "Aggregations", GPU, "Kernels/Reductions.swift", "mean()",
     "The sum kernel over a valid-count, divided on the host.", _u("mean"), ((_FLT,), {})),
    ("min", "Aggregations", GPU, "Kernels/Reductions.swift", "min()",
     "Threadgroup partials, host finalise.", _u("min"), ((_INT,), {})),
    ("min_max", "Aggregations", GPU, "Kernels/Aggregates.swift", "min_max()",
     "One kernel produces both. Returned as a `(min, max)` tuple rather than Arrow's struct scalar.",
     _u("min_max"), ((_INT,), {}), _oracle_min_max),
    ("mode", "Aggregations", GPU, "Kernels/Aggregates.swift", "mode()",
     "A GPU sort plus a run scan. Returned as `(value, count)`; only the single most common value, "
     "not Arrow's top-n list.", _u("mode"), ((_INT,), {}), _oracle_mode),
    ("pivot_wider", "Aggregations", CPU, "Sources/ArrowMetal/PivotWider.swift", "am.pivot_wider(...)",
     "One host pass over the key column, then one single-row `take` per field. The output is one row "
     "wide however long the input is, so there is no parallel work worth a kernel.",
     lambda args, options: pivot_wider(args[0], args[1], **options).to_arrow().to_pylist()[0],
     ((pa.array(["w", "x", "y"]), pa.array([1, 2, 3])), {"key_names": ["w", "x", "z"]}), _oracle_pivot),
    ("product", "Aggregations", GPU, "Kernels/Aggregates.swift", "product()",
     "Integers wrap in 64 bits; a float32 product over thousands of factors reassociates.",
     _u("product"), ((_INT,), {})),
    ("quantile", "Aggregations", GPU, "Kernels/Aggregates.swift", "quantile(q)",
     "Exact, with linear interpolation, from a GPU sort. Arrow's other `interpolation` modes and its "
     "multi-q form are not implemented.",
     lambda args, options: _a(args[0]).quantile(options["q"]), ((_FLT,), {"q": 0.5}), _oracle_quantile),
    ("skew", "Aggregations", GPU, "Kernels/AggregatesExtra.swift", "skew(biased)",
     "The third standardised central moment, biased by default as Arrow's is. Same two GPU passes as "
     "`kurtosis`.", _u("skew"), ((_FLT,), {})),
    ("stddev", "Aggregations", GPU, "Kernels/Aggregates.swift", "stddev(ddof)",
     "Compensated squared deviations on the GPU: about 1e-7 relative for float32, 1e-15 for float64.",
     _u("stddev", ddof=0), ((_FLT,), {})),
    ("sum", "Aggregations", GPU, "Kernels/Reductions.swift", "sum()",
     "Integers accumulate in int64/uint64 and wrap; float64 uses the software binary64 adder.",
     _u("sum"), ((_INT,), {})),
    ("tdigest", "Aggregations", PARTIAL, "Kernels/AggregatesExtra.swift", "tdigest(q)",
     "A GPU sort feeding a **host** centroid merge (delta 100), so the work is mixed: this is a "
     "sketch and agrees with Arrow's to within the sketch's error rather than exactly. Returns one q "
     "as a float, not Arrow's list; `quantile()` is the exact answer and needs no sketch.",
     lambda args, options: _a(args[0]).tdigest(options.get("q", 0.5)), ((_FLT,), {"q": 0.5}),
     lambda args, options: pc.tdigest(args[0], q=options.get("q", 0.5)).to_pylist()[0]),
    ("variance", "Aggregations", GPU, "Kernels/Aggregates.swift", "variance(ddof)",
     "Same pass as `stddev`.", _u("variance", ddof=0), ((_FLT,), {})),

    # ---- Arithmetic --------------------------------------------------------
    ("abs", "Arithmetic", GPU, "Kernels/Rounding.swift", "abs()", "One thread per element.",
     _u("abs"), ((_INT,), {})),
    ("add", "Arithmetic", GPU, "Kernels/Arithmetic.swift", "a + b",
     "Wrapping on integers, which is Arrow's unchecked `add`.", _op("+"), ((_INT, _INT2), {})),
    ("divide", "Arithmetic", GPU, "Kernels/Arithmetic.swift", "a / b",
     "Integer division by zero is undefined here rather than an error; that is `divide_checked`'s job.",
     _op("/"), ((_INT, _INT2), {})),
    ("exp", "Arithmetic", GPU, "Kernels/Rounding.swift", "exp()",
     "Software binary64 on a float64 column (`Kernels/DoublePower.swift`): argument reduction against "
     "a 107-bit ln 2 pair, then the exp series. Measured **1 ulp** against Foundation over 10^6 inputs "
     "spanning -745.2 to 709.78, subnormal results and the overflow edge included.",
     _u("exp"), ((_FLT2,), {})),
    ("multiply", "Arithmetic", GPU, "Kernels/Arithmetic.swift", "a * b", "Wrapping on integers.",
     _op("*"), ((_INT, _INT2), {})),
    ("negate", "Arithmetic", GPU, "Kernels/Rounding.swift", "negate()", "Wrapping on integers.",
     _u("negate"), ((_INT,), {})),
    ("power", "Arithmetic", GPU, "Kernels/Rounding.swift", "power(other)",
     "Element-wise `pow`, scalar or column exponent. Integers use repeated squaring and wrap. float64 "
     "runs in software binary64 (`Kernels/DoublePower.swift`): log2 of the base as an unevaluated "
     "hi/lo pair against a split exponent, so the 61 bits the product needs survive. Measured **1 "
     "ulp** over 10^6 random pairs, and the C99 edge table (x^0, 0^y, 1^y, (-1)^int, inf/NaN) matches "
     "libm exactly.",
     _b("power"), ((_FLT_POS, _FLT2), {})),
    ("sign", "Arithmetic", GPU, "Kernels/Rounding.swift", "sign()", "-1 / 0 / 1.",
     _u("sign"), ((_INT,), {})),
    ("sqrt", "Arithmetic", GPU, "Kernels/Rounding.swift", "sqrt()",
     "A negative input gives NaN, as Arrow's unchecked `sqrt` does. On float64 this is "
     "`DoubleMath.d_sqrt`, a digit-by-digit extraction in integers and therefore **correctly rounded** "
     "— bit-identical to Foundation over 10^6 random bit patterns, subnormals included.",
     _u("sqrt"), ((_FLT_POS,), {})),
    ("subtract", "Arithmetic", GPU, "Kernels/Arithmetic.swift", "a - b", "Wrapping on integers.",
     _op("-"), ((_INT, _INT2), {})),
    ("abs_checked", "Arithmetic", GPU, "Kernels/Checked.swift", "abs_checked()",
     "The unchecked kernel plus a read-only check pass in the same command buffer, so a checked op "
     "costs one GPU round trip. Raises only for INT_MIN on a signed integer column.",
     _u("abs_checked"), ((_INT,), {})),
    ("add_checked", "Arithmetic", GPU, "Kernels/Checked.swift", "add_checked(other)",
     "Wrapping is an error rather than a result: an `ArrowMetalError` naming the Arrow message and "
     "the first offending row. Float columns never raise, as in Arrow.",
     _b("add_checked"), ((_INT, _INT2), {})),
    ("divide_checked", "Arithmetic", GPU, "Kernels/Checked.swift", "divide_checked(other)",
     "Raises `divide by zero` for a zero divisor on any type, and `overflow` for INT_MIN / -1.",
     _b("divide_checked"), ((_INT, _INT2), {})),
    ("multiply_checked", "Arithmetic", GPU, "Kernels/Checked.swift", "multiply_checked(other)",
     "As `add_checked`.", _b("multiply_checked"), ((_INT, _INT2), {})),
    ("negate_checked", "Arithmetic", GPU, "Kernels/Checked.swift", "negate_checked()",
     "Raises for INT_MIN on a signed column and — unlike pyarrow, which has no unsigned kernel at "
     "all — for every non-zero value on an unsigned one.", _u("negate_checked"), ((_INT,), {})),
    ("power_checked", "Arithmetic", GPU, "Kernels/Checked.swift", "power_checked(other)",
     "Raises for a negative integer exponent and for any repeated-squaring step that would wrap. A "
     "float column never raises, as in Arrow, and is bit-identical to the unchecked `power` — so on "
     "float64 it is the same 1-ulp software binary64 answer.",
     _b("power_checked"), ((_INT, _INT2), {})),
    ("sqrt_checked", "Arithmetic", GPU, "Kernels/Checked.swift", "sqrt_checked()",
     "Raises `square root of negative number`; NaN, -0.0 and +inf do not raise. Bit-identical to the "
     "unchecked `sqrt`, and so correctly rounded on a float64 column too.",
     _u("sqrt_checked"), ((_FLT_POS,), {})),
    ("subtract_checked", "Arithmetic", GPU, "Kernels/Checked.swift", "subtract_checked(other)",
     "As `add_checked`.", _b("subtract_checked"), ((_INT, _INT2), {})),
    ("expm1", "Arithmetic", GPU, "Kernels/MathExtra.swift", "expm1()",
     "exp(x) - 1, accurate for small x, through the software binary64 routine on a float64 column "
     "(within about 5 ulp of the host libm). Float columns only, as in Arrow.",
     _u("expm1"), ((_FLT2,), {})),
    ("hypot", "Arithmetic", GPU, "Kernels/MathExtra.swift", "hypot(other)",
     "sqrt(x^2 + y^2), scaled so a large or tiny pair neither overflows nor underflows on the way. An "
     "infinite operand gives inf even opposite a NaN, as IEEE-754 prescribes.",
     _b("hypot"), ((_FLT, _FLT2), {})),

    # ---- Bit-wise ----------------------------------------------------------
    ("bit_wise_and", "Bitwise", GPU, "Kernels/Bitwise.swift", "bitwise_and(other)", "One thread per element.",
     _b("bitwise_and"), ((_INT, _INT2), {})),
    ("bit_wise_not", "Bitwise", GPU, "Kernels/Bitwise.swift", "bitwise_not()", "One thread per element.",
     _u("bitwise_not"), ((_INT,), {})),
    ("bit_wise_or", "Bitwise", GPU, "Kernels/Bitwise.swift", "bitwise_or(other)", "One thread per element.",
     _b("bitwise_or"), ((_INT, _INT2), {})),
    ("bit_wise_xor", "Bitwise", GPU, "Kernels/Bitwise.swift", "bitwise_xor(other)", "One thread per element.",
     _b("bitwise_xor"), ((_INT, _INT2), {})),
    ("shift_left", "Bitwise", GPU, "Kernels/Bitwise.swift", "shift_left(other)",
     "A shift at or past the width is undefined here rather than an error.",
     _b("shift_left"), ((_INT, pa.array([1, 2, 3, 1, 0, None], type=pa.int64())), {})),
    ("shift_right", "Bitwise", GPU, "Kernels/Bitwise.swift", "shift_right(other)",
     "Arithmetic for signed types, logical for unsigned.",
     _b("shift_right"), ((_INT, pa.array([1, 2, 3, 1, 0, None], type=pa.int64())), {})),
    ("shift_left_checked", "Bitwise", GPU, "Kernels/Checked.swift", "shift_left_checked(other)",
     "Raises when the shift amount is negative or at least the precision of the type (the bit width, "
     "less one on a signed column). Bits shifted off the top are not an error, as in Arrow.",
     _b("shift_left_checked"), ((_INT, _SHIFT), {})),
    ("shift_right_checked", "Bitwise", GPU, "Kernels/Checked.swift", "shift_right_checked(other)",
     "The same amount check as `shift_left_checked`.", _b("shift_right_checked"), ((_INT, _SHIFT), {})),

    # ---- Rounding ----------------------------------------------------------
    ("ceil", "Rounding", GPU, "Kernels/Rounding.swift", "ceil()", "One thread per element.",
     _u("ceil"), ((_FLT,), {})),
    ("floor", "Rounding", GPU, "Kernels/Rounding.swift", "floor()", "One thread per element.",
     _u("floor"), ((_FLT,), {})),
    ("trunc", "Rounding", GPU, "Kernels/Rounding.swift", "trunc()", "One thread per element.",
     _u("trunc"), ((_FLT,), {})),
    ("round", "Rounding", GPU, "Kernels/MathExtra.swift", "round(ndigits, mode)",
     "All ten Arrow round modes and any `ndigits`, evaluated as round_int(x * 10^ndigits) / 10^ndigits. "
     "`round()` with no argument keeps its historical meaning (halves away from zero); passing either "
     "option selects Arrow's kernel, whose defaults are ndigits=0 and half_to_even.",
     lambda args, options: _out(_a(args[0]).round(options.get("ndigits", 0),
                                                  options.get("round_mode", "half_to_even"))),
     ((_FLT_ROUND,), {})),
    ("round_binary", "Rounding", GPU, "Kernels/MathExtra.swift", "round_binary(ndigits, mode)",
     "`round` with one `ndigits` per row, from an int32 column. Null wherever either column is.",
     lambda args, options: _out(_a(args[0]).round_binary(_a(args[1]),
                                                         options.get("round_mode", "half_to_even"))),
     ((_FLT_ROUND, _NDIGITS), {})),
    ("round_to_multiple", "Rounding", GPU, "Kernels/MathExtra.swift", "round_to_multiple(multiple, mode)",
     "round_int(x / multiple) * multiple, for any positive scalar multiple and any Arrow round mode.",
     lambda args, options: _out(_a(args[0]).round_to_multiple(options["multiple"],
                                                              options.get("round_mode", "half_to_even"))),
     ((_FLT_ROUND,), {"multiple": 0.5})),

    # ---- Logarithmic -------------------------------------------------------
    ("ln", "Logarithmic", GPU, "Kernels/Rounding.swift", "ln()",
     "Metal's `log` on float32; on float64 the software binary64 of `Kernels/DoublePower.swift`, which "
     "carries log2(x) as an unevaluated hi/lo pair and scales it by a split ln 2 so the leading product "
     "is exact. Measured **1 ulp** against Foundation over 10^6 inputs from 5e-324 to 1e308, plus "
     "passes concentrated near 1 and over the subnormals.", _u("ln"), ((_FLT_POS,), {})),
    ("log10", "Logarithmic", GPU, "Kernels/Rounding.swift", "log10()",
     "Metal's `log10` on float32; on float64 the same software binary64 reduction as `ln`, scaled by a "
     "split log10 2 instead. Measured **1 ulp** over 10^6 inputs.", _u("log10"), ((_FLT_POS,), {})),
    ("log2", "Logarithmic", GPU, "Kernels/Rounding.swift", "log2()",
     "Metal's `log2` on float32; on float64 the software binary64 reduction rounded once, so an exact "
     "power of two comes back exactly. Measured **1 ulp** over 10^6 inputs.", _u("log2"), ((_FLT_POS,), {})),
    ("ln_checked", "Logarithmic", GPU, "Kernels/Checked.swift", "ln_checked()",
     "Raises `logarithm of zero` / `logarithm of negative number` — the boundary is exact, so +5e-324 "
     "passes and -5e-324 does not, and NaN and +inf never raise. Bit-identical to the unchecked `ln`, "
     "and so the same 1-ulp software binary64 answer on a float64 column.",
     _u("ln_checked"), ((_FLT_POS,), {})),
    ("log10_checked", "Logarithmic", GPU, "Kernels/Checked.swift", "log10_checked()",
     "Same domain check and the same 1-ulp software binary64 evaluation as `ln_checked`.",
     _u("log10_checked"), ((_FLT_POS,), {})),
    ("log2_checked", "Logarithmic", GPU, "Kernels/Checked.swift", "log2_checked()",
     "Same domain check and the same 1-ulp software binary64 evaluation as `ln_checked`.",
     _u("log2_checked"), ((_FLT_POS,), {})),
    ("log1p", "Logarithmic", GPU, "Kernels/MathExtra.swift", "log1p()",
     "ln(1 + x), accurate for small x, through the software binary64 routine — full float64 precision, "
     "unlike the plain `ln`. x == -1 gives -inf and x < -1 gives NaN.",
     _u("log1p"), ((_FLT_POS,), {})),
    ("log1p_checked", "Logarithmic", GPU, "Kernels/Checked.swift", "log1p_checked()",
     "As `log1p`, raising at the domain boundary: -1 gives `logarithm of zero` and anything below it "
     "`logarithm of negative number`.", _u("log1p_checked"), ((_FLT_POS,), {})),
    ("logb", "Logarithmic", GPU, "Kernels/MathExtra.swift", "logb(base)",
     "ln(x) / ln(base) in software binary64, with a scalar base or a column of bases.",
     _b("logb"), ((_FLT_POS, _FLT2), {})),
    ("logb_checked", "Logarithmic", GPU, "Kernels/Checked.swift", "logb_checked(base)",
     "As `logb`, raising when the value or the base is zero or negative.",
     _b("logb_checked"), ((_FLT_POS, _FLT2), {})),

    # ---- Trigonometric -----------------------------------------------------
    # float32 runs Metal's library functions, with the six hyperbolics written out from well-conditioned
    # identities because Metal's own lose accuracy and get +/-infinity wrong; float64 runs a software
    # binary64 implementation on the GPU, since Metal has no `double`. Measured against the host libm
    # over a million random arguments per function the worst case is 4 ulp (float32) and 5 ulp (float64).
    # The `_checked` twins are the same values, raising on a domain violation instead of giving NaN.
    ("acos", "Trigonometric", GPU, "Kernels/Trig.swift", "acos()",
     "Needs |x| <= 1; outside that the unchecked form gives NaN.", _u("acos"), ((_UNIT,), {})),
    ("acos_checked", "Trigonometric", GPU, "Kernels/Trig.swift", "acos_checked()",
     "As `acos`, raising for |x| > 1 on a non-null row. NaN never raises.",
     _u("acos_checked"), ((_UNIT,), {})),
    ("acosh", "Trigonometric", GPU, "Kernels/Trig.swift", "acosh()",
     "Needs x >= 1.", _u("acosh"), ((_GE1,), {})),
    ("acosh_checked", "Trigonometric", GPU, "Kernels/Trig.swift", "acosh_checked()",
     "As `acosh`, raising for x < 1.", _u("acosh_checked"), ((_GE1,), {})),
    ("asin", "Trigonometric", GPU, "Kernels/Trig.swift", "asin()",
     "Needs |x| <= 1.", _u("asin"), ((_UNIT,), {})),
    ("asin_checked", "Trigonometric", GPU, "Kernels/Trig.swift", "asin_checked()",
     "As `asin`, raising for |x| > 1.", _u("asin_checked"), ((_UNIT,), {})),
    ("asinh", "Trigonometric", GPU, "Kernels/Trig.swift", "asinh()",
     "Defined on the whole real line, so Arrow publishes no checked twin.", _u("asinh"), ((_FLT,), {})),
    ("atan", "Trigonometric", GPU, "Kernels/Trig.swift", "atan()", "Whole real line.",
     _u("atan"), ((_FLT,), {})),
    ("atan2", "Trigonometric", GPU, "Kernels/Trig.swift", "atan2(other)",
     "The angle of (x, y) in [-pi, pi], following the C99 special-value table including the four "
     "+/-0 and four +/-infinity cases. A scalar second argument is broadcast.",
     _b("atan2"), ((_FLT, _FLT2), {})),
    ("atanh", "Trigonometric", GPU, "Kernels/Trig.swift", "atanh()",
     "Needs |x| < 1.", _u("atanh"), ((_UNIT,), {})),
    ("atanh_checked", "Trigonometric", GPU, "Kernels/Trig.swift", "atanh_checked()",
     "As `atanh`, raising for |x| >= 1.", _u("atanh_checked"), ((_UNIT,), {})),
    ("cos", "Trigonometric", GPU, "Kernels/Trig.swift", "cos()", "Whole real line.",
     _u("cos"), ((_FLT,), {})),
    ("cos_checked", "Trigonometric", GPU, "Kernels/Trig.swift", "cos_checked()",
     "As `cos`, rejecting +/-infinity.", _u("cos_checked"), ((_FLT,), {})),
    ("cosh", "Trigonometric", GPU, "Kernels/Trig.swift", "cosh()",
     "Written out from a well-conditioned identity rather than Metal's own.", _u("cosh"), ((_FLT,), {})),
    ("sin", "Trigonometric", GPU, "Kernels/Trig.swift", "sin()", "Whole real line.",
     _u("sin"), ((_FLT,), {})),
    ("sin_checked", "Trigonometric", GPU, "Kernels/Trig.swift", "sin_checked()",
     "As `sin`, rejecting +/-infinity.", _u("sin_checked"), ((_FLT,), {})),
    ("sinh", "Trigonometric", GPU, "Kernels/Trig.swift", "sinh()", "As `cosh`.",
     _u("sinh"), ((_FLT,), {})),
    ("tan", "Trigonometric", GPU, "Kernels/Trig.swift", "tan()", "Whole real line.",
     _u("tan"), ((_FLT,), {})),
    ("tan_checked", "Trigonometric", GPU, "Kernels/Trig.swift", "tan_checked()",
     "As `tan`, rejecting +/-infinity.", _u("tan_checked"), ((_FLT,), {})),
    ("tanh", "Trigonometric", GPU, "Kernels/Trig.swift", "tanh()", "As `cosh`.",
     _u("tanh"), ((_FLT,), {})),

    # ---- Comparisons -------------------------------------------------------
    ("equal", "Comparisons", GPU, "Kernels/Compare.swift", "compare('==', other)",
     "Bit-exact for float32 through an order-preserving integer key; validity bitmaps are ANDed on the GPU.",
     _op("=="), ((_INT, _INT2), {})),
    ("greater", "Comparisons", GPU, "Kernels/Compare.swift", "compare('>', other)", "One thread per element.",
     _op(">"), ((_INT, _INT2), {})),
    ("greater_equal", "Comparisons", GPU, "Kernels/Compare.swift", "compare('>=', other)", "One thread per element.",
     _op(">="), ((_INT, _INT2), {})),
    ("less", "Comparisons", GPU, "Kernels/Compare.swift", "compare('<', other)", "One thread per element.",
     _op("<"), ((_INT, _INT2), {})),
    ("less_equal", "Comparisons", GPU, "Kernels/Compare.swift", "compare('<=', other)", "One thread per element.",
     _op("<="), ((_INT, _INT2), {})),
    ("not_equal", "Comparisons", GPU, "Kernels/Compare.swift", "compare('!=', other)", "One thread per element.",
     _op("!="), ((_INT, _INT2), {})),
    ("max_element_wise", "Comparisons", GPU, "Kernels/Rounding.swift", "max_element_wise(other)",
     "Arrow's default `skip_nulls=True` is what this does; `skip_nulls=False` is not implemented.",
     _b("max_element_wise"), ((_INT, _INT2), {})),
    ("min_element_wise", "Comparisons", GPU, "Kernels/Rounding.swift", "min_element_wise(other)",
     "Same null rule as `max_element_wise`.", _b("min_element_wise"), ((_INT, _INT2), {})),

    # ---- Logical -----------------------------------------------------------
    ("and_", "Logical", GPU, "Kernels/Compare.swift", "a & b",
     "Bitmap AND on the GPU, null-propagating (Arrow's `and`, not `and_kleene`).",
     _op("&"), ((_BOOL, _BOOL2), {})),
    ("or_", "Logical", GPU, "Kernels/Compare.swift", "a | b", "Bitmap OR, null-propagating.",
     _op("|"), ((_BOOL, _BOOL2), {})),
    ("invert", "Logical", GPU, "Kernels/Compare.swift", "~a",
     "Validity is shared zero-copy with the input.", _u("invert"), ((_BOOL,), {})),
    ("and_kleene", "Logical", GPU, "Kernels/Structural.swift", "and_kleene(other)",
     "Three-valued AND: false wins over null.", _b("and_kleene"), ((_BOOL, _BOOL2), {})),
    ("or_kleene", "Logical", GPU, "Kernels/Structural.swift", "or_kleene(other)",
     "Three-valued OR: true wins over null.", _b("or_kleene"), ((_BOOL, _BOOL2), {})),
    ("and_not", "Logical", GPU, "Kernels/LogicalExtra.swift", "and_not(other)",
     "`a AND NOT b`, word-wise over the packed bitmaps, one thread per 32-bit output word. Nulls "
     "propagate.", _b("and_not"), ((_BOOL, _BOOL2), {})),
    ("and_not_kleene", "Logical", GPU, "Kernels/LogicalExtra.swift", "and_not_kleene(other)",
     "Three-valued `a AND NOT b`: a valid false on the left or a valid true on the right gives false "
     "even when the other side is null.", _b("and_not_kleene"), ((_BOOL, _BOOL2), {})),
    ("xor", "Logical", GPU, "Kernels/LogicalExtra.swift", "a ^ b",
     "Word-wise bitmap XOR; the output validity is the AND of the inputs'.",
     _b("xor"), ((_BOOL, _BOOL2), {})),

    # ---- String predicates -------------------------------------------------
    ("ascii_is_alnum", "StringPredicates", GPU, "Kernels/StringTransforms.swift", "is_alnum()",
     "One thread per row over the bytes. An empty string is false, as in Arrow.",
     _u("is_alnum"), ((_STR,), {})),
    ("ascii_is_alpha", "StringPredicates", GPU, "Kernels/StringTransforms.swift", "is_alpha()",
     "One thread per row.", _u("is_alpha"), ((_STR,), {})),
    ("ascii_is_decimal", "StringPredicates", GPU, "Kernels/StringTransforms.swift", "is_digit()",
     "One thread per row; `0`-`9` only, which is exactly `ascii_is_decimal`.",
     _u("is_digit"), ((_STR,), {})),
    ("ascii_is_lower", "StringPredicates", GPU, "Kernels/StringTransforms.swift", "is_lower()",
     "Needs at least one cased character and none of the opposite case; non-ASCII bytes count as uncased.",
     _u("is_lower"), ((_STR,), {})),
    ("ascii_is_space", "StringPredicates", GPU, "Kernels/StringTransforms.swift", "is_space()",
     "Space and `\\t`-`\\r`.", _u("is_space"), ((_STR_PAD,), {})),
    ("ascii_is_upper", "StringPredicates", GPU, "Kernels/StringTransforms.swift", "is_upper()",
     "Mirror of `ascii_is_lower`.", _u("is_upper"), ((_STR,), {})),
    ("ascii_is_printable", "StringPredicates", GPU, "Kernels/StringExtra.swift", "ascii_is_printable()",
     "Every byte in 0x20-0x7E. The empty string is true, and every other predicate here is false on "
     "it. One thread per row.", _u("ascii_is_printable"), ((_STR_UNI,), {})),
    ("ascii_is_title", "StringPredicates", GPU, "Kernels/StringExtra.swift", "ascii_is_title()",
     "Byte-wise title case over runs of ASCII letters; needs at least one letter.",
     _u("ascii_is_title"), ((_STR_UNI,), {})),
    ("string_is_ascii", "StringPredicates", GPU, "Kernels/StringExtra.swift", "string_is_ascii()",
     "Every byte < 0x80. The empty string is true.", _u("string_is_ascii"), ((_STR_UNI,), {})),
    # The ten `utf8_is_*` predicates split per row: one GPU pass answers every row and, in a second
    # bitmap, reports which rows carry a byte >= 0x80; only those rows are re-decided on the host with
    # Swift's `Unicode.Scalar.Properties`. An all-ASCII column therefore never leaves the device, and a
    # column with any non-ASCII content is a mixed GPU/CPU evaluation — which is what `partial` records
    # here. The answers agree with pyarrow's utf8proc classification (see `UnicodeClass` for the three
    # rules that had to be reconstructed: cased, whitespace, printable).
    ("utf8_is_alnum", "StringPredicates", PARTIAL, "Kernels/StringExtra.swift", "utf8_is_alnum()",
     "Non-empty and every code point a letter or a number. GPU for the rows whose bytes are all "
     "< 0x80; rows with a byte >= 0x80 are re-decided on the host, sharded over 4096-row chunks.",
     _u("utf8_is_alnum"), ((_STR_UNI,), {})),
    ("utf8_is_alpha", "StringPredicates", PARTIAL, "Kernels/StringExtra.swift", "utf8_is_alpha()",
     "Non-empty and every code point in an L* category. Same GPU/host split as `utf8_is_alnum`.",
     _u("utf8_is_alpha"), ((_STR_UNI,), {})),
    ("utf8_is_decimal", "StringPredicates", PARTIAL, "Kernels/StringExtra.swift", "utf8_is_decimal()",
     "Non-empty and every code point category Nd. Same GPU/host split.",
     _u("utf8_is_decimal"), ((_STR_UNI,), {})),
    ("utf8_is_digit", "StringPredicates", PARTIAL, "Kernels/StringExtra.swift", "utf8_is_digit()",
     "Non-empty and every code point category Nd or No. Same GPU/host split.",
     _u("utf8_is_digit"), ((_STR_UNI,), {})),
    ("utf8_is_lower", "StringPredicates", PARTIAL, "Kernels/StringExtra.swift", "utf8_is_lower()",
     "At least one cased code point and no upper-case one. Same GPU/host split.",
     _u("utf8_is_lower"), ((_STR_UNI,), {})),
    ("utf8_is_numeric", "StringPredicates", PARTIAL, "Kernels/StringExtra.swift", "utf8_is_numeric()",
     "Non-empty and every code point category Nd, Nl or No. Same GPU/host split.",
     _u("utf8_is_numeric"), ((_STR_UNI,), {})),
    ("utf8_is_printable", "StringPredicates", PARTIAL, "Kernels/StringExtra.swift", "utf8_is_printable()",
     "No Cc/Cf/Cs/Co/Cn/Zs/Zl/Zp code point, except U+0020; the empty string is true. Same GPU/host "
     "split.", _u("utf8_is_printable"), ((_STR_UNI,), {})),
    ("utf8_is_space", "StringPredicates", PARTIAL, "Kernels/StringExtra.swift", "utf8_is_space()",
     "Non-empty and every code point Unicode whitespace (U+200B deliberately is not). Same GPU/host "
     "split.", _u("utf8_is_space"), ((_STR_UNI,), {})),
    ("utf8_is_title", "StringPredicates", PARTIAL, "Kernels/StringExtra.swift", "utf8_is_title()",
     "At least one cased code point, in title case. Same GPU/host split.",
     _u("utf8_is_title"), ((_STR_UNI,), {})),
    ("utf8_is_upper", "StringPredicates", PARTIAL, "Kernels/StringExtra.swift", "utf8_is_upper()",
     "At least one cased code point and no lower-case one. Same GPU/host split.",
     _u("utf8_is_upper"), ((_STR_UNI,), {})),

    # ---- String transforms -------------------------------------------------
    ("ascii_capitalize", "StringTransforms", GPU, "Kernels/StringTransforms.swift", "capitalize()",
     "First byte upper-cased, the rest lower-cased, ASCII only.", _u("capitalize"), ((_STR,), {})),
    ("ascii_lower", "StringTransforms", GPU, "Kernels/StringTransforms.swift", "ascii_lower()",
     "Byte-wise `A`-`Z` to `a`-`z`; every other byte is copied through, so the output stays valid UTF-8.",
     _u("ascii_lower"), ((_STR,), {})),
    ("ascii_upper", "StringTransforms", GPU, "Kernels/StringTransforms.swift", "ascii_upper()",
     "Byte-wise `a`-`z` to `A`-`Z`.", _u("ascii_upper"), ((_STR,), {})),
    ("ascii_swapcase", "StringTransforms", GPU, "Kernels/StringTransforms.swift", "swapcase()",
     "Byte-wise ASCII case flip.", _u("ascii_swapcase"), ((_STR,), {})),
    ("ascii_reverse", "StringTransforms", PARTIAL, "Kernels/StringTransforms.swift", "str_reverse()",
     "Reverses **code points**, not bytes. Identical to Arrow for ASCII input — which is what "
     "`ascii_reverse` is defined on — but Arrow's byte reversal of non-ASCII input (which produces "
     "invalid UTF-8) is not reproduced.", _u("str_reverse"), ((_STR,), {})),
    ("utf8_reverse", "StringTransforms", GPU, "Kernels/StringTransforms.swift", "str_reverse()",
     "Reverses code points, walking the UTF-8 lead bytes.", _u("str_reverse"), ((_STR,), {})),
    ("binary_reverse", "StringTransforms", PARTIAL, "Kernels/StringTransforms.swift", "str_reverse()",
     "Same kernel as `utf8_reverse`, so it reverses code points rather than bytes; equal to Arrow only "
     "for single-byte content, which is what the check below feeds it. A `binary` column is refused, "
     "so the input has to be utf8.",
     _u("str_reverse"), ((_STR,), {}),
     lambda args, options: pc.binary_reverse(args[0].cast(pa.binary())).cast(pa.string())),
    ("binary_length", "StringTransforms", PARTIAL, "Sources/ArrowMetal/MetalStringArray.swift", "byte_length()",
     "The offsets difference; no data read at all. Takes a **utf8** column only — a `binary` column "
     "is refused, where Arrow's `binary_length` accepts both. The same holds for `binary_repeat` and "
     "`binary_reverse`; `is_in` / `index_in` / `take` / `filter` / `binary_replace_slice` do accept "
     "binary.", _u("byte_length"), ((_STR,), {})),
    ("utf8_length", "StringTransforms", GPU, "Sources/ArrowMetal/MetalStringArray.swift", "char_length()",
     "Counts non-continuation bytes.", _u("char_length"), ((_STR,), {})),
    ("binary_repeat", "StringTransforms", PARTIAL, "Kernels/StringTransforms.swift", "repeat(n)",
     "Two-pass on the GPU: a length kernel, a scan into offsets, a byte kernel. Two limits: `n` is "
     "one scalar for the whole column, where Arrow also takes a per-row `num_repeats` array; and the "
     "input must be utf8, not `binary`.",
     lambda args, options: _out(_a(args[0]).repeat(options["num_repeats"])),
     ((_STR,), {"num_repeats": 3}), _oracle_positional("binary_repeat", "num_repeats")),
    ("replace_substring", "StringTransforms", GPU, "Kernels/StringTransforms.swift",
     "replace(pattern, replacement, max_replacements)",
     "Literal replacement on the GPU, two-pass.",
     lambda args, options: _out(_a(args[0]).replace(options["pattern"], options["replacement"],
                                                    options.get("max_replacements", -1) or -1)),
     ((_STR,), {"pattern": "l", "replacement": "L"})),
    ("replace_substring_regex", "StringTransforms", CPU, "Kernels/Regex.swift",
     "replace_substring_regex(pattern, replacement)",
     "The regex engine is a backtracker on the host; a literal pattern is routed to the GPU kernel instead.",
     lambda args, options: _out(_a(args[0]).replace_substring_regex(options["pattern"], options["replacement"])),
     ((_STR,), {"pattern": "l+", "replacement": "L"})),
    ("utf8_lower", "StringTransforms", PARTIAL, "Kernels/StringTransforms.swift", "lower()",
     "Simple 1:1 case mapping over Basic Latin, Latin-1 Supplement and Latin Extended-A. Everything "
     "above U+017F passes through unchanged; there is no Unicode case table on the GPU.",
     _u("lower"), ((_STR,), {})),
    ("utf8_upper", "StringTransforms", PARTIAL, "Kernels/StringTransforms.swift", "upper()",
     "Same coverage as `utf8_lower`. U+00DF, whose full uppercase is `SS`, passes through unchanged.",
     _u("upper"), ((_STR,), {})),
    ("utf8_swapcase", "StringTransforms", PARTIAL, "Kernels/StringExtra.swift", "utf8_swapcase()",
     "Same coverage as `utf8_upper` / `utf8_lower`: Basic Latin, Latin-1 Supplement and Latin "
     "Extended-A, including the length-changing pairs. U+00DF (which Arrow swaps to U+1E9E) and every "
     "code point above U+017F pass through unchanged rather than being mangled.",
     _u("utf8_swapcase"), ((_STR,), {})),
    ("ascii_title", "StringTransforms", GPU, "Kernels/StringExtra.swift", "ascii_title()",
     "Byte-wise: the first ASCII letter of every run of ASCII letters is upper-cased and the rest "
     "lower-cased. Bytes >= 0x80 are copied through and end a word, so `\"ünïcödé\"` becomes "
     "`\"üNïCöDé\"` — exactly what Arrow does.", _u("ascii_title"), ((_STR_UNI,), {})),
    ("utf8_capitalize", "StringTransforms", PARTIAL, "Kernels/StringExtra.swift", "utf8_capitalize()",
     "First code point upper-cased, every later one lower-cased, through Unicode's **simple** 1:1 "
     "mappings as utf8proc uses. GPU (the `ascii_capitalize` kernel) when the whole column is ASCII "
     "and host-side otherwise — an output length that depends on a Unicode table cannot be computed "
     "in the GPU length pass.", _u("utf8_capitalize"), ((_STR_UNI,), {})),
    ("utf8_title", "StringTransforms", PARTIAL, "Kernels/StringExtra.swift", "utf8_title()",
     "The first cased code point of every word upper-cased and the rest lower-cased, a word being a "
     "maximal run of cased code points. Same per-column GPU/host split as `utf8_capitalize`.",
     _u("utf8_title"), ((_STR_UNI,), {})),
    ("utf8_normalize", "StringTransforms", CPU, "Kernels/StringExtra.swift", "utf8_normalize(form)",
     "NFC / NFKC / NFD / NFKD through Foundation, on the host: a full Unicode normalisation table in "
     "MSL buys nothing over it. Deliberate difference: pyarrow 25 never *composes*, so its NFC equals "
     "its NFD and its NFKC equals its NFKD; this follows the Unicode standard and agrees with "
     "Python's `unicodedata.normalize` on all four forms. The check below uses NFD, where the two "
     "agree.",
     lambda args, options: _out(_a(args[0]).utf8_normalize(options["form"])),
     ((_STR_UNI,), {"form": "NFD"}), _oracle_positional("utf8_normalize", "form")),
    ("utf8_replace_slice", "StringTransforms", GPU, "Kernels/StringExtra.swift",
     "utf8_replace_slice(start, stop, replacement)",
     "Replaces the code points in [start, stop). Negative indices count from the end, both ends "
     "clamp, and a stop below start inserts without deleting. The cut always lands on a code-point "
     "boundary.",
     lambda args, options: _out(_a(args[0]).utf8_replace_slice(options["start"], options["stop"],
                                                               options["replacement"])),
     ((_STR_UNI,), {"start": 1, "stop": 3, "replacement": "XY"}),
     _oracle_positional("utf8_replace_slice", "start", "stop", "replacement")),
    ("binary_replace_slice", "StringTransforms", GPU, "Kernels/StringExtra.swift",
     "binary_replace_slice(start, stop, replacement)",
     "The same substitution indexed in **bytes**, which can split a UTF-8 sequence and is why the "
     "result is `binary` rather than `utf8`.",
     lambda args, options: _out(_a(args[0]).binary_replace_slice(options["start"], options["stop"],
                                                                 options["replacement"])),
     ((_STR_UNI,), {"start": 1, "stop": 3, "replacement": "XY"}),
     lambda args, options: pc.binary_replace_slice(args[0].cast(pa.binary()), options["start"],
                                                   options["stop"], options["replacement"])),

    # ---- String padding ----------------------------------------------------
    ("ascii_lpad", "StringPadding", GPU, "Kernels/StringTransforms.swift", "pad_left(width, pad)",
     "`width` counts code points and `pad` must be one character; a string already at or over `width` "
     "is returned unchanged.",
     lambda args, options: _out(_a(args[0]).pad_left(options["width"], options.get("padding", " "))),
     ((_STR,), {"width": 8})),
    ("ascii_rpad", "StringPadding", GPU, "Kernels/StringTransforms.swift", "pad_right(width, pad)",
     "Mirror of `ascii_lpad`.",
     lambda args, options: _out(_a(args[0]).pad_right(options["width"], options.get("padding", " "))),
     ((_STR,), {"width": 8})),
    ("utf8_lpad", "StringPadding", GPU, "Kernels/StringTransforms.swift", "pad_left(width, pad)",
     "The same kernel: `width` has always counted code points.",
     lambda args, options: _out(_a(args[0]).pad_left(options["width"], options.get("padding", " "))),
     ((_STR,), {"width": 8})),
    ("utf8_rpad", "StringPadding", GPU, "Kernels/StringTransforms.swift", "pad_right(width, pad)",
     "The same kernel as `ascii_rpad`.",
     lambda args, options: _out(_a(args[0]).pad_right(options["width"], options.get("padding", " "))),
     ((_STR,), {"width": 8})),
    ("utf8_zero_fill", "StringPadding", GPU, "Kernels/StringExtra.swift", "utf8_zero_fill(width, padding)",
     "Left-pads to `width` code points, inserting the padding after a leading `+` or `-`. The content "
     "need not be numeric.",
     lambda args, options: _out(_a(args[0]).utf8_zero_fill(options["width"], options.get("padding", "0"))),
     ((_STR_NUM,), {"width": 5})),
    ("ascii_center", "StringPadding", PARTIAL, "Kernels/StringExtra.swift", "utf8_center(width, padding)",
     "The same kernel as `utf8_center`, so `width` counts **code points** where Arrow's ASCII form "
     "counts bytes — identical on ASCII input, which is what `ascii_center` is defined on.",
     lambda args, options: _out(_a(args[0]).utf8_center(options["width"], options.get("padding", " "))),
     ((_STR,), {"width": 8, "padding": "*"}),
     lambda args, options: pc.ascii_center(args[0], options["width"], options.get("padding", " "))),
    ("utf8_center", "StringPadding", GPU, "Kernels/StringExtra.swift", "utf8_center(width, padding)",
     "Pads on both sides to `width` code points, the odd pad character going on the **right** (`\"a\"` "
     "centred in 4 is `\"*a**\"`). Strings already at or over `width` come back unchanged.",
     lambda args, options: _out(_a(args[0]).utf8_center(options["width"], options.get("padding", " "))),
     ((_STR_UNI,), {"width": 8, "padding": "*"}),
     _oracle_positional("utf8_center", "width", "padding")),

    # ---- String trimming ---------------------------------------------------
    ("ascii_trim_whitespace", "StringTrimming", GPU, "Kernels/StringTransforms.swift", "trim()",
     "Space and `\\t`-`\\r` from both ends.", _u("trim"), ((_STR_PAD,), {})),
    ("ascii_ltrim_whitespace", "StringTrimming", GPU, "Kernels/StringTransforms.swift", "ltrim()",
     "Leading ASCII whitespace.", _u("ltrim"), ((_STR_PAD,), {})),
    ("ascii_rtrim_whitespace", "StringTrimming", GPU, "Kernels/StringTransforms.swift", "rtrim()",
     "Trailing ASCII whitespace.", _u("rtrim"), ((_STR_PAD,), {})),
    ("ascii_trim", "StringTrimming", GPU, "Kernels/StringTransforms.swift", "trim(characters)",
     "Trims any byte in `characters` from both ends.",
     lambda args, options: _out(_a(args[0]).trim(options["characters"])), ((_STR,), {"characters": "Hlo"})),
    ("ascii_ltrim", "StringTrimming", GPU, "Kernels/StringTransforms.swift", "ltrim(characters)",
     "Leading bytes only.",
     lambda args, options: _out(_a(args[0]).ltrim(options["characters"])), ((_STR,), {"characters": "Hlo"})),
    ("ascii_rtrim", "StringTrimming", GPU, "Kernels/StringTransforms.swift", "rtrim(characters)",
     "Trailing bytes only.",
     lambda args, options: _out(_a(args[0]).rtrim(options["characters"])), ((_STR,), {"characters": "Hlo"})),
    ("utf8_trim_whitespace", "StringTrimming", PARTIAL, "Kernels/StringExtra.swift", "utf8_trim()",
     "Strips the full **Unicode** whitespace class from both ends (Zs/Zl/Zp plus U+0009-U+000D, "
     "U+001C-U+001F and U+0085; U+200B deliberately is not whitespace). GPU when the column is all "
     "ASCII — that set restricted to ASCII is a ten-byte set the existing trim kernel handles — and "
     "host-side otherwise.", _u("utf8_trim"), ((_STR_UNI,), {})),
    ("utf8_ltrim_whitespace", "StringTrimming", PARTIAL, "Kernels/StringExtra.swift", "utf8_ltrim()",
     "Leading-only form of `utf8_trim_whitespace`, with the same GPU/host split.",
     _u("utf8_ltrim"), ((_STR_UNI,), {})),
    ("utf8_rtrim_whitespace", "StringTrimming", PARTIAL, "Kernels/StringExtra.swift", "utf8_rtrim()",
     "Trailing-only form, with the same GPU/host split.", _u("utf8_rtrim"), ((_STR_UNI,), {})),
    ("utf8_trim", "StringTrimming", PARTIAL, "Kernels/StringExtra.swift", "utf8_trim(characters)",
     "Strips leading and trailing **code points** that appear in `characters`. An ASCII set runs on "
     "the GPU (byte-wise trimming can never split a UTF-8 sequence, since every continuation byte is "
     ">= 0x80); a set with a non-ASCII character runs on the host.",
     lambda args, options: _out(_a(args[0]).utf8_trim(options["characters"])),
     ((_STR_UNI,), {"characters": "éH  "})),
    ("utf8_ltrim", "StringTrimming", PARTIAL, "Kernels/StringExtra.swift", "utf8_ltrim(characters)",
     "Leading-only form of `utf8_trim`, with the same GPU/host split.",
     lambda args, options: _out(_a(args[0]).utf8_ltrim(options["characters"])),
     ((_STR_UNI,), {"characters": "éH  "})),
    ("utf8_rtrim", "StringTrimming", PARTIAL, "Kernels/StringExtra.swift", "utf8_rtrim(characters)",
     "Trailing-only form, with the same GPU/host split.",
     lambda args, options: _out(_a(args[0]).utf8_rtrim(options["characters"])),
     ((_STR_UNI,), {"characters": "éH  "})),

    # ---- String splitting --------------------------------------------------
    ("ascii_split_whitespace", "StringSplitting", PARTIAL, "Kernels/Regex.swift", "split_whitespace()",
     "Splits on runs of ASCII whitespace, on the host. Two differences from Arrow: leading and "
     "trailing whitespace produce no empty piece (Arrow emits one at each end, and one for an empty "
     "string), and the ArrowMetal call returns `(offsets, values)` rather than a list column — the "
     "registry stitches those into one. `max_splits` and `reverse` are not implemented.",
     _split("split_whitespace"), ((_STR_WS,), {})),
    ("utf8_split_whitespace", "StringSplitting", PARTIAL, "Kernels/Regex.swift", "split_whitespace()",
     "The same host split, with the same dropped end pieces as `ascii_split_whitespace`, and **ASCII** "
     "whitespace only rather than the Unicode whitespace class.",
     _split("split_whitespace"), ((_STR_WS,), {})),
    ("split_pattern", "StringSplitting", CPU, "Kernels/Regex.swift", "split_pattern(pattern)",
     "Literal split on the host. `max_splits` and `reverse` are not implemented.",
     _split("split_pattern", "pattern"), ((_STR,), {"pattern": "l"})),
    ("split_pattern_regex", "StringSplitting", CPU, "Kernels/Regex.swift", "split_pattern(pattern, regex=True)",
     "The host backtracking engine. `max_splits` and `reverse` are not implemented.",
     lambda args, options: _split_regex(args, options), ((_STR,), {"pattern": "l+"})),

    # ---- String extraction / joining / slicing -----------------------------
    ("extract_regex", "StringExtraction", CPU, "Kernels/Regex.swift", "extract_regex(pattern)",
     "Named groups on the host, returned as a struct column. ICU spelling `(?<name>...)`; RE2's "
     "`(?P<name>...)`, which pyarrow uses, is rewritten by the Python wrapper.",
     lambda args, options: _extract_regex_struct(args, options),
     ((pa.array(["a1", "b22", None, "zz"]),), {"pattern": r"(?<letter>[a-z])(?<digits>\d+)"}),
     _oracle_extract_regex),
    ("extract_regex_span", "StringExtraction", CPU, "Kernels/StringExtra.swift", "extract_regex_span(pattern)",
     "One `(start, length)` pair of int32 columns per **named** group, counted in bytes as Arrow's "
     "are; the registry reshapes them into Arrow's row-wise struct. A row that does not match, a null "
     "row and a group that took part in no alternative are null in both. Host-side throughout (ICU, "
     "sharded over 4096-row chunks), and ICU's `(?<name>...)` spelling rather than RE2's `(?P<name>...)`.",
     _call_extract_regex_span,
     ((_REGEX_IN,), {"pattern": r"(?<letter>[a-z])(?<digits>\d+)"}), _oracle_extract_regex_span),
    ("binary_join", "StringJoining", GPU, "Kernels/StringContainment.swift", "binary_join(separator)",
     "Joins the child strings of every row of a `list<utf8>`, two-pass on the GPU. The separator is a "
     "scalar or a per-row column. An empty row joins to the empty string; a null row, a null element "
     "inside a row and a null separator all give a null output row.",
     lambda args, options: _out(_a(args[0]).binary_join(args[1])), ((_LIST_STR, "-"), {}),
     lambda args, options: pc.binary_join(args[0], args[1])),
    ("binary_join_element_wise", "StringJoining", PARTIAL, "Kernels/StringTransforms.swift",
     "str_concat(other, separator)",
     "Two columns and a scalar separator on the GPU. Arrow's N-column form (the last argument being "
     "the separator column) and its `null_handling` options are not implemented.",
     lambda args, options: _out(_a(args[0]).str_concat(_a(args[1]), options.get("separator", ""))),
     ((_STR, _STR), {"separator": "-"}),
     lambda args, options: pc.binary_join_element_wise(*args, pa.scalar(options["separator"]))),
    ("utf8_slice_codeunits", "StringSlicing", GPU, "Kernels/StringTransforms.swift",
     "slice_codeunits(start, stop)",
     "Code-point slicing, two-pass on the GPU. A negative `start`/`stop` and Arrow's `step` are not "
     "implemented.",
     lambda args, options: _out(_a(args[0]).slice_codeunits(options["start"], options.get("stop"))),
     ((_STR,), {"start": 1, "stop": 4})),
    ("binary_slice", "StringSlicing", MISSING, "-", "-",
     "The one Arrow compute name with no implementation here. The slicing kernel counts code points "
     "and there is no byte-offset variant of it; `binary_replace_slice` does index in bytes, so the "
     "machinery exists and this is unclaimed rather than out of scope.", None, None),

    # ---- Containment / matching -------------------------------------------
    ("count_substring", "Containment", GPU, "Kernels/StringTransforms.swift", "count_substring(pattern)",
     "Byte-wise search, one thread per row.",
     lambda args, options: _out(_a(args[0]).count_substring(options["pattern"])), ((_STR,), {"pattern": "l"})),
    ("find_substring", "Containment", GPU, "Kernels/StringTransforms.swift", "find_substring(pattern)",
     "First byte offset, -1 when absent.",
     lambda args, options: _out(_a(args[0]).find_substring(options["pattern"])), ((_STR,), {"pattern": "l"})),
    ("match_substring", "Containment", GPU, "Sources/ArrowMetal/MetalStringArray.swift", "str_contains(pattern)",
     "Byte-wise containment. `ignore_case` is not implemented.",
     lambda args, options: _out(_a(args[0]).str_contains(options["pattern"])), ((_STR,), {"pattern": "l"})),
    ("starts_with", "Containment", GPU, "Sources/ArrowMetal/MetalStringArray.swift", "starts_with(pattern)",
     "Byte-wise prefix test.",
     lambda args, options: _out(_a(args[0]).starts_with(options["pattern"])), ((_STR,), {"pattern": "He"})),
    ("ends_with", "Containment", GPU, "Sources/ArrowMetal/MetalStringArray.swift", "ends_with(pattern)",
     "Byte-wise suffix test.",
     lambda args, options: _out(_a(args[0]).ends_with(options["pattern"])), ((_STR,), {"pattern": "lo"})),
    ("count_substring_regex", "Containment", CPU, "Kernels/Regex.swift", "count_substring_regex(pattern)",
     "Host engine; a literal pattern routes to the GPU counter instead.",
     lambda args, options: _out(_a(args[0]).count_substring_regex(options["pattern"])), ((_STR,), {"pattern": "l+"})),
    ("find_substring_regex", "Containment", CPU, "Kernels/Regex.swift", "find_substring_regex(pattern)",
     "Host engine.",
     lambda args, options: _out(_a(args[0]).find_substring_regex(options["pattern"])), ((_STR,), {"pattern": "l+"})),
    ("match_substring_regex", "Containment", CPU, "Kernels/Regex.swift", "match_substring_regex(pattern)",
     "Host engine; literal and `^literal` patterns route to the GPU kernels.",
     lambda args, options: _out(_a(args[0]).match_substring_regex(options["pattern"])), ((_STR,), {"pattern": "l+"})),
    ("match_like", "Containment", CPU, "Kernels/Regex.swift", "match_like(pattern)",
     "SQL LIKE, translated to the host regex engine; a pattern with no metacharacter routes to the GPU.",
     lambda args, options: _out(_a(args[0]).match_like(options["pattern"])), ((_STR,), {"pattern": "%ll%"})),
    ("is_in", "Containment", PARTIAL, "Kernels/StringContainment.swift", "is_in(value_set)",
     "GPU throughout: a sorted set plus a binary search per row for primitive and temporal columns, "
     "the GPU string hash table for utf8 and binary ones. Nulls in the set are ignored and a null "
     "element is never in the set, so Arrow's `null_matching_behavior` is fixed at `skip` (pyarrow's "
     "default); `match` / `emit_null` / `inconclusive` are not implemented.",
     lambda args, options: _out(_a(args[0]).is_in(options["value_set"])),
     ((_STR,), {"value_set": pa.array(["Hello", "Zz"])})),
    ("index_in", "Containment", PARTIAL, "Kernels/StringContainment.swift", "index_in(value_set)",
     "The same two paths, returning the int32 position of the first occurrence in the set and null "
     "where the element is null or absent. Same `null_matching_behavior` limitation as `is_in`.",
     lambda args, options: _out(_a(args[0]).index_in(options["value_set"])),
     ((_STR,), {"value_set": pa.array(["Hello", "Zz"])})),

    # ---- Categorizations ---------------------------------------------------
    ("is_null", "Categorizations", GPU, "Kernels/Structural.swift", "is_null()",
     "The validity bitmap inverted on the GPU. Arrow's `nan_is_null` is not implemented.",
     _u("is_null"), ((_INT,), {})),
    ("is_valid", "Categorizations", GPU, "Kernels/Structural.swift", "is_valid()",
     "The validity bitmap copied out as a boolean column.", _u("is_valid"), ((_INT,), {})),
    ("true_unless_null", "Categorizations", CPU, "Kernels/Selection.swift", "true_unless_null()",
     "A host `memset` of the values bitmap; the validity bitmap is shared with the input with no copy, "
     "so no kernel runs. Union and run-end encoded columns are refused.",
     _u("true_unless_null"), ((_INT,), {})),
    ("indices_nonzero", "Categorizations", GPU, "Kernels/Conditional.swift", "indices_nonzero()",
     "`iota` through the existing stream compaction: the uint64 row numbers where the value is valid "
     "and not zero. `-0.0` counts as zero and every NaN as non-zero, as in Arrow.",
     _u("indices_nonzero"), ((pa.array([0, 1, 0, 5, None, 7], type=pa.int64()),), {})),
    ("is_finite", "Categorizations", GPU, "Kernels/FloatClass.swift", "is_finite()",
     "A raw bit-pattern test, one thread per 32-bit output word, so float64 needs no software "
     "binary64. True everywhere on an integer column, and null where the input is null.",
     _u("is_finite"), ((_FLT,), {})),
    ("is_inf", "Categorizations", GPU, "Kernels/FloatClass.swift", "is_inf()",
     "As `is_finite`; false everywhere on an integer column.", _u("is_inf"), ((_FLT,), {})),
    ("is_nan", "Categorizations", GPU, "Kernels/FloatClass.swift", "is_nan()",
     "As `is_finite`; false everywhere on an integer column.", _u("is_nan"), ((_FLT,), {})),

    # ---- Selecting ---------------------------------------------------------
    ("if_else", "Selecting", GPU, "Kernels/Structural.swift", "if_else(left, right)",
     "One thread per element; a null condition gives a null output.",
     lambda args, options: _out(_a(args[0]).if_else(args[1], args[2])), ((_BOOL, _INT, _INT2), {})),
    ("coalesce", "Selecting", GPU, "Kernels/Structural.swift", "am.coalesce(*arrays)",
     "First non-null across the inputs, one thread per element.",
     lambda args, options: _out(_coalesce(*[_a(x) for x in args])), ((_INT, _INT2), {})),
    ("case_when", "Selecting", GPU, "Kernels/Conditional.swift", "am.case_when(conds, values, default)",
     "A fold of the existing `if_else` kernel, one GPU pass per branch. A **null condition counts as "
     "false** and the row falls through, as in Arrow. ArrowMetal takes the conditions as a list of "
     "boolean columns where Arrow takes one struct column of them.",
     _call_case_when,
     (([_BOOL, _BOOL2], [_INT, _INT2], _IDX.cast(pa.int64())), {}), _oracle_case_when),
    ("choose", "Selecting", GPU, "Kernels/Conditional.swift", "am.choose(indices, values)",
     "`values[indices[i]][i]`, element-wise, as the same fold. A null index gives a null output and "
     "an index outside [0, len(values)) raises, as in Arrow.",
     _call_choose,
     ((pa.array([0, 1, 0, 1, None, 0], type=pa.int32()), [_INT, _INT2]), {}), _oracle_choose),

    # ---- Conversions -------------------------------------------------------
    ("cast", "Conversions", PARTIAL, "Kernels/Cast.swift", "cast(target)",
     "Numeric to numeric on the GPU (wrapping, C-style, no overflow check), numeric to/from utf8 on "
     "the GPU, temporal unit changes on the GPU. The float16 and decimal conversions exist but under "
     "their own names — `to_float32()` / `to_float16()`, `to_decimal128()` / `to_small_decimal()` — "
     "rather than through `cast()`. Overflow-erroring casts and casts between nested types are not "
     "implemented.",
     lambda args, options: _out(_a(args[0]).cast(options["target_type"])),
     ((_INT,), {"target_type": "float64"})),
    ("ceil_temporal", "Conversions", GPU, "Kernels/TemporalMath.swift", "ceil_temporal(unit, multiple)",
     "Integer arithmetic in the value's own resolution, plus the civil-date algorithm for month / "
     "quarter / year. `calendar_based_origin` and week units are not implemented.",
     lambda args, options: _out(_a(args[0]).ceil_temporal(options["unit"], options.get("multiple", 1))),
     ((_TS,), {"unit": "day"})),
    ("floor_temporal", "Conversions", GPU, "Kernels/TemporalMath.swift", "floor_temporal(unit, multiple)",
     "As `ceil_temporal`.",
     lambda args, options: _out(_a(args[0]).floor_temporal(options["unit"], options.get("multiple", 1))),
     ((_TS,), {"unit": "day"})),
    ("round_temporal", "Conversions", PARTIAL, "Kernels/TemporalMath.swift", "round_temporal(unit, multiple)",
     "As `ceil_temporal`, but an exact half rounds **up** (toward +infinity) where Arrow rounds half "
     "to even.",
     lambda args, options: _out(_a(args[0]).round_temporal(options["unit"], options.get("multiple", 1))),
     ((_TS,), {"unit": "day"})),
    ("run_end_encode", "Conversions", GPU, "Sources/ArrowMetal/RunEndEncoded.swift", "run_end_encode()",
     "Bit equality collapses adjacent values; nulls form runs of their own. Run ends are int32.",
     _u("run_end_encode"), ((_INT,), {})),
    ("run_end_decode", "Conversions", GPU, "Sources/ArrowMetal/RunEndEncoded.swift", "run_end_decode()",
     "Expands a run-end encoded column back to a flat one.",
     lambda args, options: _out(_a(args[0]).run_end_encode().run_end_decode()), ((_INT,), {}),
     lambda args, options: args[0]),

    # ---- Temporal extraction ----------------------------------------------
    ("year", "TemporalExtraction", GPU, "Sources/ArrowMetal/Temporal.swift", "year()",
     "The civil-date algorithm on the GPU, UTC. A timestamp's timezone is carried as metadata and "
     "never applied.", _u("year"), ((_TS,), {})),
    ("month", "TemporalExtraction", GPU, "Sources/ArrowMetal/Temporal.swift", "month()", "UTC.",
     _u("month"), ((_TS,), {})),
    ("day", "TemporalExtraction", GPU, "Sources/ArrowMetal/Temporal.swift", "day()", "UTC.",
     _u("day"), ((_TS,), {})),
    ("day_of_week", "TemporalExtraction", GPU, "Kernels/TemporalExtra.swift",
     "day_of_week(count_from_zero, week_start)",
     "Arrow's full `DayOfWeekOptions`; `week_start` uses the ISO numbering (1 = Monday ... 7 = "
     "Sunday). The default options keep the int32 result (Monday = 0); any other combination returns "
     "int64, as pyarrow does.",
     lambda args, options: _out(_a(args[0]).day_of_week(**options)), ((_TS,), {})),
    ("hour", "TemporalExtraction", GPU, "Sources/ArrowMetal/Temporal.swift", "hour()", "UTC.",
     _u("hour"), ((_TS,), {})),
    ("minute", "TemporalExtraction", GPU, "Sources/ArrowMetal/Temporal.swift", "minute()", "UTC.",
     _u("minute"), ((_TS,), {})),
    ("second", "TemporalExtraction", GPU, "Sources/ArrowMetal/Temporal.swift", "second()", "UTC.",
     _u("second"), ((_TS,), {})),
    ("day_of_year", "TemporalExtraction", GPU, "Kernels/TemporalMath.swift", "day_of_year()", "UTC.",
     _u("day_of_year"), ((_TS,), {})),
    ("quarter", "TemporalExtraction", GPU, "Kernels/TemporalMath.swift", "quarter()", "UTC.",
     _u("quarter"), ((_TS,), {})),
    ("iso_week", "TemporalExtraction", GPU, "Kernels/TemporalMath.swift", "iso_week()", "UTC.",
     _u("iso_week"), ((_TS,), {})),
    ("iso_year", "TemporalExtraction", GPU, "Kernels/TemporalMath.swift", "iso_year()", "UTC.",
     _u("iso_year"), ((_TS,), {})),
    ("millisecond", "TemporalExtraction", GPU, "Kernels/TemporalMath.swift", "millisecond()", "UTC.",
     _u("millisecond"), ((_TS,), {})),
    ("microsecond", "TemporalExtraction", GPU, "Kernels/TemporalMath.swift", "microsecond()", "UTC.",
     _u("microsecond"), ((_TS,), {})),
    ("nanosecond", "TemporalExtraction", GPU, "Kernels/TemporalMath.swift", "nanosecond()", "UTC.",
     _u("nanosecond"), ((_TS,), {})),
    ("is_leap_year", "TemporalExtraction", GPU, "Kernels/TemporalMath.swift", "is_leap_year()", "UTC.",
     _u("is_leap_year"), ((_TS,), {})),
    ("strftime", "TemporalExtraction", CPU, "Kernels/TemporalMath.swift", "strftime(format)",
     "The C library's `strftime` against a `gmtime_r` struct on the host, plus a `%f` extension for "
     "microseconds. Not a Unicode date pattern, and always UTC.",
     lambda args, options: _out(_a(args[0]).strftime(options["format"])),
     ((_TS,), {"format": "%Y-%m-%d"})),
    ("strptime", "TemporalExtraction", CPU, "Kernels/TemporalMath.swift", "strptime(format)",
     "The C library's `strptime` on the host, UTC. `error_is_null` is not implemented — an unparseable "
     "row is null either way.",
     lambda args, options: _out(_a(args[0]).strptime(options["format"])),
     ((pa.array(["2020-01-02", None, "1999-12-31"]),), {"format": "%Y-%m-%d"}),
     _oracle_keyword("strptime", unit="us")),
    ("is_dst", "TemporalExtraction", CPU, "Kernels/TemporalExtra.swift", "is_dst()",
     "The one function here that does apply a timestamp's timezone, and so the one that needs the "
     "IANA tz database — host data with no GPU-resident form. Runs on the host, sharded over "
     "`DispatchQueue.concurrentPerform`. A naive timestamp is an error, as in Arrow.",
     _u("is_dst"), ((_TS_TZ,), {})),
    ("iso_calendar", "TemporalExtraction", GPU, "Kernels/TemporalExtra.swift", "iso_calendar()",
     "A struct of int64 `iso_year`, `iso_week` and `iso_day_of_week` (1 = Monday), UTC.",
     lambda args, options: _out(_a(args[0]).iso_calendar()), ((_TS,), {})),
    ("subsecond", "TemporalExtraction", GPU, "Kernels/TemporalExtra.swift", "subsecond()",
     "The fraction of a second in [0, 1) as float64, UTC. date32 and date64 answer 0 (pyarrow has no "
     "kernel for them) and duration is rejected.", _u("subsecond"), ((_TS,), {})),
    ("us_week", "TemporalExtraction", GPU, "Kernels/TemporalExtra.swift", "us_week()",
     "The US week number: Sunday-start weeks and the majority rule, 1-53. UTC.",
     _u("us_week"), ((_TS,), {})),
    ("us_year", "TemporalExtraction", GPU, "Kernels/TemporalExtra.swift", "us_year()",
     "The US epidemiological week-numbering year — the year owning the Wednesday of this date's "
     "Sunday-start week. UTC.", _u("us_year"), ((_TS,), {})),
    ("week", "TemporalExtraction", GPU, "Kernels/TemporalExtra.swift",
     "week(week_starts_monday, count_from_zero, first_week_is_fully_in_year)",
     "Arrow's full `WeekOptions`, int64, UTC. The defaults reproduce `iso_week`.",
     lambda args, options: _out(_a(args[0]).week(**options)), ((_TS,), {})),
    ("year_month_day", "TemporalExtraction", GPU, "Kernels/TemporalExtra.swift", "year_month_day()",
     "A struct of int64 `year`, `month` and `day`, UTC.",
     lambda args, options: _out(_a(args[0]).year_month_day()), ((_TS,), {})),

    # ---- Temporal differences ---------------------------------------------
    ("days_between", "TemporalDifference", GPU, "Kernels/TemporalMath.swift", "days_between(other)",
     "Whole days between two temporal columns, UTC.",
     lambda args, options: _out(_a(args[0]).days_between(_a(args[1]))), ((_TS, _TS2), {})),
    # Every `*_between` counts boundaries crossed from the first argument to the second: each side is
    # truncated to the unit first and the difference taken afterwards, so it is not the truncated
    # difference. Positive when the second is later. UTC throughout, and more permissive than pyarrow
    # in one direction — the two sides may differ in unit and in type.
    ("day_time_interval_between", "TemporalDifference", PARTIAL, "Sources/ArrowMetal/IntervalBetween.swift",
     "day_time_interval_between(other).interval_field(...)",
     "GPU. pyarrow 25 has no Python type or Array class for `interval[day_time]`, so the result "
     "cannot be handed to pyarrow at all: read the fields with `interval_field('days' | "
     "'nanoseconds')`, which is what the check below compares against `days_between`.",
     _interval("day_time_interval_between", "days"), ((_TS, _TS2), {}), _oracle_day_time_days),
    ("hours_between", "TemporalDifference", GPU, "Kernels/TemporalExtra.swift", "hours_between(other)",
     "Hour boundaries crossed, int64, UTC.",
     _b("hours_between"), ((_TS, _TS2), {})),
    ("microseconds_between", "TemporalDifference", GPU, "Kernels/TemporalExtra.swift", "microseconds_between(other)",
     "Microsecond boundaries crossed.", _b("microseconds_between"), ((_TS, _TS2), {})),
    ("milliseconds_between", "TemporalDifference", GPU, "Kernels/TemporalExtra.swift", "milliseconds_between(other)",
     "Millisecond boundaries crossed.", _b("milliseconds_between"), ((_TS, _TS2), {})),
    ("minutes_between", "TemporalDifference", GPU, "Kernels/TemporalExtra.swift", "minutes_between(other)",
     "Minute boundaries crossed.", _b("minutes_between"), ((_TS, _TS2), {})),
    ("month_day_nano_interval_between", "TemporalDifference", GPU, "Sources/ArrowMetal/IntervalBetween.swift",
     "month_day_nano_interval_between(other)",
     "The one interval difference pyarrow can express in Python, so this one round-trips as an "
     "`interval[month_day_nano]` column. Every field is the difference of the corresponding truncated "
     "field, so the day and sub-day parts may carry the opposite sign.",
     _b("month_day_nano_interval_between"), ((_TS, _TS2), {}), _oracle_month_day_nano),
    ("month_interval_between", "TemporalDifference", PARTIAL, "Sources/ArrowMetal/IntervalBetween.swift",
     "month_interval_between(other).interval_field('months')",
     "GPU: month boundaries crossed. Same pyarrow gap as `day_time_interval_between` — there is no "
     "Python type for `interval[month]`, so the months are read with `interval_field('months')` and "
     "checked against the `months` field of `month_day_nano_interval_between`.",
     _interval("month_interval_between", "months"), ((_TS, _TS2), {}), _oracle_month_interval),
    ("nanoseconds_between", "TemporalDifference", GPU, "Kernels/TemporalExtra.swift", "nanoseconds_between(other)",
     "Nanosecond boundaries crossed; wraps in int64 past about 292 years, as Arrow's does.",
     _b("nanoseconds_between"), ((_TS, _TS2), {})),
    ("quarters_between", "TemporalDifference", GPU, "Kernels/TemporalExtra.swift", "quarters_between(other)",
     "The difference of year * 4 + quarter.", _b("quarters_between"), ((_TS, _TS2), {})),
    ("seconds_between", "TemporalDifference", GPU, "Kernels/TemporalExtra.swift", "seconds_between(other)",
     "Second boundaries crossed.", _b("seconds_between"), ((_TS, _TS2), {})),
    ("weeks_between", "TemporalDifference", GPU, "Kernels/TemporalExtra.swift",
     "weeks_between(other, count_from_zero, week_start)",
     "Week boundaries crossed, both sides floored to the start of their week first. `week_start` is "
     "1 = Monday ... 7 = Sunday; `count_from_zero` is accepted for signature parity and does not "
     "change the answer, as in Arrow.", _b("weeks_between"), ((_TS, _TS2), {})),
    ("years_between", "TemporalDifference", GPU, "Kernels/TemporalExtra.swift", "years_between(other)",
     "The difference of the two calendar years.", _b("years_between"), ((_TS, _TS2), {})),

    # ---- Timezone ----------------------------------------------------------
    ("assume_timezone", "Timezone", CPU, "Sources/ArrowMetal/Timezone.swift",
     "assume_timezone(tz, ambiguous, nonexistent)",
     "Reads a naive column as wall-clock times in `tz` and returns the instants they name. Host-side "
     "deliberately: the offsets are a lookup in the IANA tz database, which has no GPU-resident form, "
     "so uploading the transition table per call would cost more than the arithmetic saves. Sharded "
     "over `DispatchQueue.concurrentPerform` with a per-shard offset cache. A local time that occurs "
     "twice or never raises by default; `\"earliest\"` / `\"latest\"` pick one, as Arrow does.",
     lambda args, options: _out(_a(args[0]).assume_timezone(options["timezone"],
                                                            options.get("ambiguous", "raise"),
                                                            options.get("nonexistent", "raise"))),
     ((_TS,), {"timezone": "America/New_York"}),
     lambda args, options: pc.assume_timezone(args[0], options["timezone"])),
    ("local_timestamp", "Timezone", CPU, "Sources/ArrowMetal/Timezone.swift", "local_timestamp()",
     "The wall-clock time each instant names in the column's own timezone, as a naive timestamp of "
     "the same unit. Host-side for the same reason as `assume_timezone`; a column with no timezone "
     "comes back unchanged.", _u("local_timestamp"), ((_TS_TZ,), {})),

    # ---- Random ------------------------------------------------------------
    ("random", "Random", GPU, "Kernels/Selection.swift", "am.random(n, initializer)",
     "Philox4x32-10 keyed by the seed, one counter per element, so the stream depends only on the seed. "
     "The top 53 bits of each draw become a multiple of 2^-53 in [0, 1). ArrowMetal's own stream: it "
     "does not reproduce Arrow C++'s pcg32_fast numbers for the same seed.",
     lambda args, options: _out(random(options["n"], options.get("initializer", 0))),
     ((), {"n": 64, "initializer": 7}), _oracle_random),

    # ---- Associative transforms -------------------------------------------
    ("unique", "Associative", PARTIAL, "Kernels/Unique.swift", "unique()",
     "One GPU sort plus a run scan. The values come back **ascending**; Arrow returns them in order of "
     "first appearance, and nulls are dropped rather than kept.",
     _u("unique"), ((_INT,), {}), _oracle_sorted_unique),
    ("value_counts", "Associative", PARTIAL, "Kernels/Unique.swift", "value_counts()",
     "The same pass, returned as a struct of `values` and `counts`. Ascending order, not first "
     "appearance; nulls are dropped.",
     lambda args, options: _a(args[0]).value_counts().to_arrow().to_pylist(), ((_INT,), {}),
     _oracle_sorted_value_counts),
    ("dictionary_encode", "Associative", GPU, "Kernels/StringDictionary.swift", "dictionary_encode()",
     "GPU hashing for utf8, a GPU sort for primitives. Returns `(codes, values)` rather than Arrow's "
     "dictionary-typed array.",
     lambda args, options: _a(args[0]).dictionary_encode()[0].to_arrow(), ((_STR,), {}),
     lambda args, options: pc.dictionary_encode(args[0]).indices),
    ("dictionary_decode", "Associative", GPU, "Sources/ArrowMetal/DictionaryArray.swift", "dictionary_decode()",
     "A `take` of the values through the codes.", _u("dictionary_decode"), ((_DICT,), {})),

    # ---- Selections --------------------------------------------------------
    ("filter", "Selections", GPU, "Kernels/Filter.swift", "filter(mask)",
     "Per-block popcount, GPU scan, scatter and validity pack, all in one command buffer. A null mask "
     "entry drops the row (Arrow's `null_selection_behavior=\"drop\"`); `\"emit_null\"` is not implemented.",
     lambda args, options: _out(_a(args[0]).filter(_a(args[1]))), ((_INT, _BOOL), {})),
    ("array_filter", "Selections", GPU, "Kernels/Filter.swift", "array_filter(mask)",
     "The same kernel under Arrow's array-only name.",
     lambda args, options: _out(_a(args[0]).array_filter(_a(args[1]))), ((_INT, _BOOL), {})),
    ("take", "Selections", GPU, "Kernels/Take.swift", "take(indices)",
     "One gather kernel. A null index yields a null row; an out-of-range index sets a GPU error flag "
     "raised after the dispatch.",
     lambda args, options: _out(_a(args[0]).take(_a(args[1]))), ((_INT, _IDX), {})),
    ("array_take", "Selections", GPU, "Kernels/Take.swift", "array_take(indices)",
     "The same kernel under Arrow's array-only name.",
     lambda args, options: _out(_a(args[0]).array_take(_a(args[1]))), ((_INT, _IDX), {})),
    ("drop_null", "Selections", GPU, "Kernels/Structural.swift", "drop_null()",
     "The filter kernel driven by the validity bitmap.", _u("drop_null"), ((_INT,), {})),
    ("inverse_permutation", "Selections", GPU, "Kernels/Selection.swift", "inverse_permutation(max_index)",
     "An atomic scatter: for the i-th index the index-th output is i. Unassigned slots are null and "
     "duplicates resolve to the last source position, deterministically (the scatter is an atomic "
     "maximum). Always int32 — Arrow's `output_type` is not implemented.",
     lambda args, options: _out(_a(args[0]).inverse_permutation(options.get("max_index", -1))),
     ((pa.array([1, 1, None, 3], type=pa.int32()),), {"max_index": 4})),
    ("scatter", "Selections", GPU, "Kernels/Selection.swift", "scatter(indices, max_index)",
     "The inverse permutation used as a `take`, so it works for every column type. Unassigned "
     "positions are null and duplicate indices resolve to the last value.",
     lambda args, options: _out(_a(args[0]).scatter(args[1], options.get("max_index", -1))),
     ((pa.array([10, 20, 30], type=pa.int64()), pa.array([1, 1, 3], type=pa.int32())), {"max_index": 4})),

    # ---- Sorts and partitions ---------------------------------------------
    ("array_sort_indices", "Sorts", GPU, "Kernels/Sort.swift", "array_sort_indices(descending)",
     "LSD radix sort, stable, nulls last, total order for floats (NaN after +inf). Arrow's "
     "`null_placement=\"at_start\"` is not implemented.",
     lambda args, options: _out(_a(args[0]).array_sort_indices(options.get("order", "ascending") == "descending")),
     ((_INT,), {})),
    ("sort_indices", "Sorts", PARTIAL, "Kernels/MultiSort.swift", "sort_indices() / am.lexsort_indices(cols)",
     "Single key through the radix argsort; multiple keys through `lexsort_indices`, which is "
     "successive stable argsorts from the least significant key upwards. utf8, binary and dictionary "
     "key columns are not sortable, and `null_placement=\"at_start\"` is not implemented.",
     lambda args, options: _out(_a(args[0]).sort_indices()), ((_INT,), {})),
    ("partition_nth_indices", "Sorts", PARTIAL, "Kernels/MultiSort.swift", "partition_nth_indices(pivot)",
     "Answered with the full stable GPU argsort, which satisfies Arrow's contract (the n smallest "
     "first) but does more work than a true partial partition and returns a different permutation.",
     lambda args, options: _out(_a(args[0]).partition_nth_indices(options["pivot"])),
     ((_INT,), {"pivot": 3}), _oracle_partition_nth),
    ("select_k_unstable", "Sorts", GPU, "Kernels/TopK.swift", "select_k_unstable(k, largest)",
     "For k <= 1024 each threadgroup keeps the best k of its own block and one radix sort orders the "
     "survivors; larger k falls back to the full sort. Single key.",
     lambda args, options: _out(_a(args[0]).select_k_unstable(options["k"], options.get("largest", False))),
     ((_INT,), {"k": 3}), _oracle_select_property),
    ("top_k_unstable", "Sorts", GPU, "Kernels/TopK.swift", "top_k_unstable(k)",
     "`select_k_unstable` with the descending order.",
     lambda args, options: _out(_a(args[0]).top_k_unstable(options["k"])), ((_INT,), {"k": 3}),
     _oracle_select_property),
    ("bottom_k_unstable", "Sorts", GPU, "Kernels/TopK.swift", "bottom_k_unstable(k)",
     "`select_k_unstable` with the ascending order.",
     lambda args, options: _out(_a(args[0]).bottom_k_unstable(options["k"])), ((_INT,), {"k": 3}),
     _oracle_select_property),
    ("rank", "Sorts", PARTIAL, "Kernels/Window.swift", "rank() / dense_rank() / row_number()",
     "One argsort, run marks, a scan and a scatter back to the original rows. Arrow's `tiebreaker` "
     "options are separate calls here — `rank()` is `min`, `dense_rank()` is `dense`, `row_number()` "
     "is `first`; `max` is not implemented, and neither is `null_placement=\"at_start\"`.",
     _u("rank"), ((_INT,), {}), lambda args, options: pc.rank(args[0], sort_keys="ascending", tiebreaker="min")),
    ("rank_quantile", "Sorts", PARTIAL, "Kernels/Selection.swift", "rank_quantile()",
     "(average 1-based rank of the tie group - 0.5) / n, computed on the GPU as (s + e) / (2n) over "
     "the run's sorted positions. Arrow's `sort_keys` and `null_placement` options are not "
     "implemented: always one ascending key with nulls at the end.",
     _u("rank_quantile"), ((_INT,), {})),
    ("rank_normal", "Sorts", PARTIAL, "Kernels/Selection.swift", "rank_normal(float32=False)",
     "The normal percent-point function of `rank_quantile`. float64 evaluates the inverse CDF on the "
     "host with Wichura's AS 241 (about 1e-16 relative) because Metal has no `double` and the software "
     "binary64 has no log/exp/erfc; `float32=True` runs Acklam plus one Halley refinement entirely on "
     "the GPU, within about 1e-6. Arrow's option set is not implemented.",
     _u("rank_normal"), ((_INT,), {})),
    ("winsorize", "Sorts", GPU, "Kernels/Selection.swift", "winsorize(lower_limit, upper_limit)",
     "One GPU sort for the two nearest quantiles, then a clamp kernel. Nulls stay null and NaNs pass "
     "through, taking part in neither the limits nor the comparison.",
     lambda args, options: _out(_a(args[0]).winsorize(options["lower_limit"], options["upper_limit"])),
     ((_FLT,), {"lower_limit": 0.2, "upper_limit": 0.8})),

    # ---- Null filling ------------------------------------------------------
    ("fill_null", "NullFilling", GPU, "Kernels/Structural.swift", "fill_null(value)",
     "One thread per element; the result carries no validity bitmap when the fill removes every null.",
     lambda args, options: _out(_a(args[0]).fill_null(options["fill_value"])), ((_INT,), {"fill_value": 0})),
    ("fill_null_forward", "NullFilling", GPU, "Kernels/Conditional.swift", "fill_null_forward()",
     "One GPU max-scan over the last valid row index plus a gather. Leading nulls stay null.",
     _u("fill_null_forward"), ((_INT,), {})),
    ("fill_null_backward", "NullFilling", GPU, "Kernels/Conditional.swift", "fill_null_backward()",
     "The same scan run the other way. Trailing nulls stay null.",
     _u("fill_null_backward"), ((_INT,), {})),

    # ---- Structural --------------------------------------------------------
    ("list_value_length", "Structural", GPU, "Sources/ArrowMetal/Nested.swift", "list_value_length()",
     "The offsets difference; null lists give null.", _u("list_value_length"), ((_LIST,), {})),
    ("list_flatten", "Structural", GPU, "Sources/ArrowMetal/Nested.swift", "list_flatten()",
     "Concatenates the child ranges of the valid rows.", _u("list_flatten"), ((_LIST,), {})),
    ("list_element", "Structural", GPU, "Sources/ArrowMetal/Nested.swift", "list_element(index)",
     "A gather through offsets + index; a row too short gives null where Arrow raises.",
     lambda args, options: _out(_a(args[0]).list_element(options["index"])), ((_LIST_NE,), {"index": 0}),
     _oracle_positional("list_element", "index")),
    ("struct_field", "Structural", GPU, "Sources/ArrowMetal/Nested.swift", "struct_field(name)",
     "The named child with the struct's own nulls propagated into it. Only a single-level field name, "
     "not Arrow's nested index path.",
     lambda args, options: _out(_a(args[0]).struct_field(options["indices"])), ((_STRUCT,), {"indices": "n"})),
    ("make_struct", "Structural", CPU, "Sources/ArrowMetal/Nested.swift", "am.make_struct(arrays, names)",
     "Metadata only: the children are shared, nothing is copied and no kernel runs.",
     lambda args, options: make_struct(list(args), options["field_names"]).to_arrow(),
     ((_INT, _STR), {"field_names": ["a", "b"]}), _oracle_make_struct),
    ("list_parent_indices", "Structural", PARTIAL, "Sources/ArrowMetal/NestedExtra.swift", "list_parent_indices()",
     "GPU, one binary search per child element. Documented difference: the result is **int32** where "
     "pyarrow's is int64, because list offsets are int32 throughout this package; the values are the "
     "same, and the check below casts.",
     _u("list_parent_indices"), ((_LIST,), {}),
     lambda args, options: pc.list_parent_indices(args[0]).cast(pa.int32())),
    ("list_slice", "Structural", GPU, "Sources/ArrowMetal/NestedExtra.swift", "list_slice(start, stop, step)",
     "`row[start:stop:step]` for every row, as a variable-length list. `start` must be >= 0 and "
     "`step` >= 1, as Arrow requires; a null row stays null and a row shorter than `start` becomes "
     "empty.",
     lambda args, options: _out(_a(args[0]).list_slice(options["start"], options.get("stop"),
                                                       options.get("step", 1))),
     ((_LIST,), {"start": 1, "stop": 3}),
     _oracle_positional("list_slice", "start", "stop")),
    ("map_lookup", "Structural", GPU, "Sources/ArrowMetal/NestedExtra.swift", "map_lookup(key, occurrence)",
     "One key compare per entry inside each row's range. `occurrence` is `first`, `last` or `all` "
     "(which returns a list of the item type); the result is null where the row is null or the key is "
     "absent. Keys may be utf8 / binary or any integer type.",
     lambda args, options: _out(_a(args[0]).map_lookup(options["query_key"],
                                                       options.get("occurrence", "first"))),
     ((_MAP,), {"query_key": "a", "occurrence": "first"}),
     _oracle_positional("map_lookup", "query_key", "occurrence")),
    ("replace_with_mask", "Structural", GPU, "Kernels/Conditional.swift",
     "replace_with_mask(mask, replacements)",
     "Rows where the mask is true take the next value from `replacements`, in order; rows where the "
     "mask is null become null; every other row keeps its own value. A GPU scan supplies the "
     "replacement index. Fewer replacements than valid trues raises, a surplus is ignored, both as "
     "in pyarrow.",
     lambda args, options: _out(_a(args[0]).replace_with_mask(args[1], args[2])),
     ((_INT, pa.array([True, False, None, True, False, False]),
       pa.array([100, 200], type=pa.int64())), {})),

    # ---- Pairwise and cumulative ------------------------------------------
    ("pairwise_diff", "Pairwise", GPU, "Kernels/Window.swift", "pairwise_diff(period)",
     "One thread per element; null where either side is null or outside the array. Integers wrap.",
     lambda args, options: _out(_a(args[0]).pairwise_diff(options.get("period", 1))), ((_INT,), {})),
    ("pairwise_diff_checked", "Pairwise", GPU, "Kernels/Checked.swift", "pairwise_diff_checked(period)",
     "`self[i] - self[i - period]`, raising where that step would wrap.",
     lambda args, options: _out(_a(args[0]).pairwise_diff_checked(options.get("period", 1))),
     ((_INT,), {})),
    ("cumulative_sum", "Cumulative", GPU, "Kernels/Cumulative.swift", "cumulative_sum()",
     "A two-level scan. Null in, null out, the running value carrying across nulls (Arrow's "
     "`skip_nulls=True`). `start` is not implemented.", _u("cumulative_sum"), ((_INT,), {})),
    ("cumulative_prod", "Cumulative", GPU, "Kernels/Window.swift", "cumulative_prod()",
     "The same scan with a multiply; float products reassociate.", _u("cumulative_prod"), ((_INT,), {})),
    ("cumulative_max", "Cumulative", GPU, "Kernels/Cumulative.swift", "cumulative_max()",
     "The same scan with a maximum.", _u("cumulative_max"), ((_INT,), {})),
    ("cumulative_min", "Cumulative", GPU, "Kernels/Cumulative.swift", "cumulative_min()",
     "The same scan with a minimum.", _u("cumulative_min"), ((_INT,), {})),
    ("cumulative_mean", "Cumulative", GPU, "Kernels/Window.swift", "cumulative_mean()",
     "A binary64 running sum over a running count of non-null rows; int64 magnitudes above 2^53 round "
     "on the way in.", _u("cumulative_mean"), ((_INT,), {})),
    ("cumulative_sum_checked", "Cumulative", GPU, "Kernels/Checked.swift", "cumulative_sum_checked()",
     "The running sum, raising where a step would wrap. Null rows are skipped and the running value "
     "carries across them (this package's `cumulative_sum` behaviour, Arrow's `skip_nulls=True`); "
     "pyarrow's default instead makes every row after a null null.",
     _u("cumulative_sum_checked"), ((_INT,), {})),
    ("cumulative_prod_checked", "Cumulative", GPU, "Kernels/Checked.swift", "cumulative_prod_checked()",
     "The running product, raising where a step would wrap. Same null rule as `cumulative_sum_checked`.",
     _u("cumulative_prod_checked"), ((_INT,), {})),

    # ---- Grouped aggregates (hash_*) ---------------------------------------
    #
    # `am.group_by([...])` maps **arbitrary** key columns to dense group ids on the GPU — integers
    # sparse or negative, floats, booleans, temporal values, utf8, binary, dictionary, decimal, and
    # several columns folded together — and every aggregate below runs against those ids. A null key
    # forms its own group, as in Arrow.
    #
    # One difference applies to every row here and is not repeated on each: the group ORDER is
    # deterministic but is not pyarrow's first-seen order, so the registry returns `{key: value}` and
    # the check compares the mapping rather than two parallel arrays.
    ("hash_count", "GroupedAggregations", GPU, "Kernels/GroupByKeys.swift", "group_by(keys).count(values)",
     "Non-null values per group, for any value type. Arrow's `mode` option is not implemented — this "
     "is always `only_valid`; `hash_count_all` is the `all` mode.",
     _hashk("count"), ((_GKEY_STR, _GVAL_INT), {}), _oracle_hashk("count")),
    ("hash_count_all", "GroupedAggregations", GPU, "Kernels/AggregatesExtra.swift", "group_by(keys).count_all()",
     "Rows per group, null values included. A group with no row is zero, never null.",
     _hashk("count_all", values=False), ((_GKEY_STR,), {}), _oracle_hashk("count_all", values=False)),
    ("hash_sum", "GroupedAggregations", GPU, "Kernels/GroupByKeys.swift", "group_by(keys).sum(values)",
     "A segmented reduction over the dense ids. Integers accumulate in 64 bits and wrap; `min_count` "
     "is not implemented.", _hashk("sum"), ((_GKEY_STR, _GVAL_INT), {}), _oracle_hashk("sum")),
    ("hash_mean", "GroupedAggregations", GPU, "Kernels/GroupByKeys.swift", "group_by(keys).mean(values)",
     "The grouped sum over the grouped valid count.",
     _hashk("mean"), ((_GKEY_STR, _GVAL_INT), {}), _oracle_hashk("mean")),
    ("hash_min", "GroupedAggregations", GPU, "Kernels/GroupByKeys.swift", "group_by(keys).min(values)",
     "A segmented minimum.", _hashk("min"), ((_GKEY_STR, _GVAL_INT), {}), _oracle_hashk("min")),
    ("hash_max", "GroupedAggregations", GPU, "Kernels/GroupByKeys.swift", "group_by(keys).max(values)",
     "A segmented maximum.", _hashk("max"), ((_GKEY_STR, _GVAL_INT), {}), _oracle_hashk("max")),
    ("hash_min_max", "GroupedAggregations", GPU, "Kernels/AggregatesExtra.swift", "group_by(keys).min_max(values)",
     "Fused: one segmented kernel produces both extremes from one read of the values, as a "
     "`struct<min, max>` column.",
     _hashk("min_max"), ((_GKEY_STR, _GVAL_INT), {}), _oracle_hashk("min_max")),
    ("hash_all", "GroupedAggregations", GPU, "Kernels/AggregatesExtra.swift", "group_by(keys).all(values)",
     "Three-valued AND per group over a boolean column, matching Arrow's null handling.",
     _hashk("all"), ((_GKEY_STR, _GVAL_BOOL), {}), _oracle_hashk("all")),
    ("hash_any", "GroupedAggregations", GPU, "Kernels/AggregatesExtra.swift", "group_by(keys).any(values)",
     "Three-valued OR per group.", _hashk("any"), ((_GKEY_STR, _GVAL_BOOL), {}), _oracle_hashk("any")),
    ("hash_approximate_median", "GroupedAggregations", GPU, "Kernels/AggregatesExtra.swift",
     "group_by(keys).approximate_median(values)",
     "**Exact**, not approximate: a GPU sort by (group, value) and a GPU per-group pick, because "
     "sorting on the GPU is cheaper than sketching. Arrow's is a t-digest, so on a group whose size "
     "makes the sketch inexact this answers the true median and pyarrow does not.",
     _hashk("approximate_median"), ((_GKEY_STR, _GVAL_INT), {}), _oracle_hashk("approximate_median")),
    ("hash_count_distinct", "GroupedAggregations", GPU, "Kernels/AggregatesExtra.swift",
     "group_by(keys).count_distinct(values)",
     "Dictionary-encode, packed `unique`, then a segmented count.",
     _hashk("count_distinct"), ((_GKEY_STR, _GVAL_INT), {}), _oracle_hashk("count_distinct")),
    ("hash_distinct", "GroupedAggregations", GPU, "Kernels/AggregatesExtra.swift", "group_by(keys).distinct(values)",
     "The distinct non-null values of each group as a list column, **ascending** — Arrow returns them "
     "in order of first appearance, as it does for the scalar `unique`.",
     _hashk("distinct"), ((_GKEY_STR, _GVAL_INT), {}), _oracle_hashk("distinct")),
    ("hash_first", "GroupedAggregations", GPU, "Kernels/AggregatesExtra.swift", "group_by(keys).first(values)",
     "A group-by extreme over a masked row index plus a gather.",
     _hashk("first"), ((_GKEY_STR, _GVAL_INT), {}), _oracle_hashk("first")),
    ("hash_first_last", "GroupedAggregations", GPU, "Kernels/AggregatesExtra.swift",
     "group_by(keys).first_last(values)",
     "Both ends in one `struct<first, last>` column: two group-by extremes over the masked row index "
     "plus two gathers.",
     _hashk("first_last"), ((_GKEY_STR, _GVAL_INT), {}), _oracle_hashk("first_last")),
    ("hash_last", "GroupedAggregations", GPU, "Kernels/AggregatesExtra.swift", "group_by(keys).last(values)",
     "Mirror of `hash_first`.", _hashk("last"), ((_GKEY_STR, _GVAL_INT), {}), _oracle_hashk("last")),
    ("hash_list", "GroupedAggregations", GPU, "Kernels/AggregatesExtra.swift", "group_by(keys).list(values)",
     "Every value of the group in row order, as a list column: a segmented gather after the stable "
     "sort by group id.", _hashk("list"), ((_GKEY_STR, _GVAL_INT), {}), _oracle_hashk("list")),
    ("hash_one", "GroupedAggregations", GPU, "Kernels/AggregatesExtra.swift", "group_by(keys).one(values)",
     "One value per group — here always the group's lowest row, null included. Arrow leaves which "
     "one unspecified.", _hashk("one"), ((_GKEY_STR, _GVAL_INT), {}), _oracle_hashk("one")),
    ("hash_product", "GroupedAggregations", GPU, "Kernels/AggregatesExtra.swift", "group_by(keys).product(values)",
     "A segmented multiply reduction. Integers wrap in 64 bits exactly as the scalar `product` does, "
     "and float products reassociate across the threads of a group.",
     _hashk("product"), ((_GKEY_STR, _GVAL_INT), {}), _oracle_hashk("product")),
    ("hash_stddev", "GroupedAggregations", GPU, "Kernels/GroupByKeys.swift", "group_by(keys).stddev(values, ddof)",
     "The square root of `hash_variance`, and so carries the same float32 deviations: expect about "
     "1e-5 relative.", _hashk("stddev"), ((_GKEY_STR, _GVAL_INT), {}), _oracle_hashk("stddev")),
    ("hash_variance", "GroupedAggregations", GPU, "Kernels/GroupByKeys.swift", "group_by(keys).variance(values, ddof)",
     "Two GPU passes: per-group means, then the squared deviations. The deviations are formed in "
     "float32 about a float64 mean, so expect about 1e-5 relative on well-conditioned data rather "
     "than the 1e-15 of the scalar `variance`.",
     _hashk("variance"), ((_GKEY_STR, _GVAL_INT), {}), _oracle_hashk("variance")),
    ("hash_pivot_wider", "GroupedAggregations", GPU, "Kernels/AggregatesExtra.swift",
     "group_by(keys).pivot_wider(pivot_keys, values, names)",
     "One masked `hash_one` per pivot key, giving a struct with one field per name. A (group, key) "
     "pair carrying more than one non-null value takes the lowest row here where pyarrow raises.",
     _call_group_pivot, ((_PKEY, _PPIVOT, _PVAL), {"key_names": ["x", "y"]}), _oracle_group_pivot),
    ("hash_kurtosis", "GroupedAggregations", GPU, "Kernels/AggregatesExtra.swift",
     "group_by(keys).kurtosis(values)",
     "Excess kurtosis, biased, from two GPU passes over the per-group means. Same float32 deviations "
     "as `hash_variance`, so about 1e-5 relative. A group with too few values is null here where "
     "pyarrow returns NaN.",
     _hashk("kurtosis"), ((_GKEY_STR, _GVAL_INT), {}), _oracle_hashk("kurtosis")),
    ("hash_skew", "GroupedAggregations", GPU, "Kernels/AggregatesExtra.swift", "group_by(keys).skew(values)",
     "The third standardised central moment, biased. Same passes, precision and null-on-degenerate "
     "group as `hash_kurtosis`.", _hashk("skew"), ((_GKEY_STR, _GVAL_INT), {}), _oracle_hashk("skew")),
    ("hash_tdigest", "GroupedAggregations", PARTIAL, "Kernels/AggregatesExtra.swift",
     "group_by(keys).tdigest(values, q)",
     "Mixed: a GPU sort by (group, value) and a **host** merge of each group's centroids. Returns one "
     "q per group as a scalar column where Arrow returns a list, and being a sketch it agrees with "
     "Arrow's to within the sketch's error. `group_by(keys).quantile(values, q)` is the exact answer.",
     _hashk("tdigest", q="q"), ((_GKEY_STR, _GVAL_INT), {"q": 0.5}),
     _oracle_hashk("tdigest", unwrap=True)),
]
def _extract_regex_struct(args, options):
    """`extract_regex` hands back one column per named group; Arrow hands back a struct column."""
    groups = _a(args[0]).extract_regex(options["pattern"])
    names = list(groups)
    children = [groups[n].to_arrow() for n in names]
    valid = [all(c[i].is_valid for c in children) for i in range(len(children[0]))]
    return pa.StructArray.from_arrays(children, names, mask=pa.array([not v for v in valid]))


def _split_regex(args, options):
    """`split_pattern_regex`: the same stitching as `_split`, with the regex flag set."""
    offsets, values = _a(args[0]).split_pattern(options["pattern"], regex=True)
    mask = pc.is_null(args[0]) if isinstance(args[0], pa.Array) else None
    return pa.ListArray.from_arrays(pa.array(offsets.to_arrow().to_pylist(), type=pa.int32()),
                                    values.to_arrow(), mask=mask)


def _coalesce(*arrays):
    """Local import shim so the table can name `coalesce` without a circular import at module load."""
    from . import coalesce as _c
    return _c(*arrays)


_REGISTRY = {}
for _row in _ROWS:
    _name, _section, _status, _file, _method, _notes = _row[0], _row[1], _row[2], _row[3], _row[4], _row[5]
    _call = _row[6] if len(_row) > 6 else None
    _example = _row[7] if len(_row) > 7 else None
    _oracle = _row[8] if len(_row) > 8 else None
    if _name in _REGISTRY:
        raise RuntimeError(f"duplicate registry entry for {_name}")
    _REGISTRY[_name] = Function(_name, _section, _status, _file, _method, _notes, _call, _example, _oracle)
del _row, _name, _section, _status, _file, _method, _notes, _call, _example, _oracle

# Order the sections the way the Arrow docs do, so the generated Markdown reads like the reference.
SECTIONS = []
for _f in _REGISTRY.values():
    if _f.section not in SECTIONS:
        SECTIONS.append(_f.section)
del _f


# ---------------------------------------------------------------------------
# Public API


def list_functions():
    """Every Arrow function name this registry knows, sorted — the shape of `pc.list_functions()`."""
    return sorted(_REGISTRY)


def get_function(name):
    """The :class:`Function` record for `name`."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"no Arrow function named {name!r} in the ArrowMetal registry") from None


def function_table():
    """One dict per Arrow name: section, status, the Swift file behind it, the method, and notes."""
    return [{"name": f.name, "section": f.section, "status": f.status, "swift_file": f.swift_file,
             "method": f.method, "notes": f.notes} for f in _REGISTRY.values()]


def status_counts():
    """{section: {status: count}} plus a "Total" row, for the report and the launch numbers."""
    out = {}
    for f in _REGISTRY.values():
        out.setdefault(f.section, {}).setdefault(f.status, 0)
        out[f.section][f.status] += 1
    total = {}
    for counts in out.values():
        for status, n in counts.items():
            total[status] = total.get(status, 0) + n
    out["Total"] = total
    return out


def call_function(name, args, options=None):
    """Run an Arrow-named function through ArrowMetal, mirroring `pc.call_function`.

    `args` is the positional argument list Arrow's signature takes; `options` is a dict using Arrow's
    own option names. The result comes back as a pyarrow object (or a plain Python value for a scalar
    aggregate), so it can be compared against `pyarrow.compute` directly.

    Raises :class:`NotImplementedError` for a name ArrowMetal does not implement, with the registry's
    reason attached.
    """
    f = get_function(name)
    if f.call is None:
        raise NotImplementedError(f"{name}: {f.status} - {f.notes}")
    return f.call(list(args), dict(options or {}))


def markdown_table(sections=None):
    """The registry as a Markdown table — the body of docs/ARROW_FUNCTIONS.md."""
    lines = ["| Arrow function | Section | Status | ArrowMetal call | Implementation | Notes |",
             "|---|---|---|---|---|---|"]
    wanted = sections or SECTIONS
    for section in wanted:
        for f in _REGISTRY.values():
            if f.section != section:
                continue
            status = {GPU: "**GPU**", CPU: "**CPU**", PARTIAL: "**Partial**",
                      MISSING: "Missing", PENDING: "Pending"}[f.status]
            # A literal pipe would end the cell, even inside a code span, so escape it everywhere.
            def cell(text):
                return text.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{cell(f.name)}` | {cell(f.section)} | {status} | `{cell(f.method)}` | "
                         f"`{cell(f.swift_file)}` | {cell(f.notes)} |")
    return "\n".join(lines)


def summary_markdown():
    """A per-section count table: gpu / cpu / partial / missing / pending."""
    counts = status_counts()
    lines = ["| Section | GPU | CPU | Partial | Missing | Pending | Rows |", "|---|---:|---:|---:|---:|---:|---:|"]
    for section in SECTIONS + ["Total"]:
        c = counts.get(section, {})
        row = [c.get(GPU, 0), c.get(CPU, 0), c.get(PARTIAL, 0), c.get(MISSING, 0), c.get(PENDING, 0)]
        lines.append(f"| {section} | " + " | ".join(str(x) for x in row) + f" | {sum(row)} |")
    return "\n".join(lines)


__all__ = ["Function", "GPU", "CPU", "PARTIAL", "MISSING", "PENDING", "SECTIONS", "TOLERANCE",
           "call_function", "function_table", "get_function", "list_functions",
           "markdown_table", "status_counts", "summary_markdown"]
