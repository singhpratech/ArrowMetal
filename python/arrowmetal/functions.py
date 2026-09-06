"""The Arrow function-name registry: what ArrowMetal answers to, by exact Apache Arrow name.

This module mirrors the shape of ``pyarrow.compute``'s introspection API — :func:`list_functions`
and :func:`call_function` — over the Arrow v25 compute function list (283 names in the C++ docs plus
the 24 ``hash_*`` grouped aggregates). Every one of those names is present. A name either resolves
to an ArrowMetal call, or it carries an explicit record saying it is not implemented and why.

The single source of truth is :data:`_TABLE` below. :func:`function_table` reads it, the tests read
it, and ``python/tests/function_table_report.py`` turns it into the Markdown that goes into
``docs/COVERAGE.md``. Nothing here infers a status from anything else, so a row is only ever as
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
    Not reachable from this registry yet, and a named feature owns landing it — either because it is
    being implemented on a concurrent branch, or because a Swift kernel exists but has no C ABI entry
    point and so no Python binding. The note says which. These flip to their real status when that
    work merges; nothing here can call them, so the tests report them rather than fail on them.
"""
import math

import pyarrow as pa
import pyarrow.compute as pc

from . import ArrowMetalError, MetalArray, lexsort_indices, make_struct, pivot_wider, random

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


def _hash(agg):
    """A grouped aggregate over dense integer keys: `keys.group_by(k).<agg>(values)`."""
    def run(args, options):
        keys = _a(args[0])
        group = keys.group_by(options["key_count"])
        if agg == "count_all":
            return group.count().to_arrow().to_pylist()
        if agg == "min_max":
            return [group.min(_a(args[1])).to_arrow().to_pylist(),
                    group.max(_a(args[1])).to_arrow().to_pylist()]
        return getattr(group, agg)(_a(args[1])).to_arrow().to_pylist()
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


# Rows whose answer is right but not bit-identical to Arrow's, with the relative tolerance the
# difference justifies. Every one of these is a float32 evaluation of a float64 column: Metal has no
# `double` transcendentals, so `exp`, the logarithms, `sqrt` and `power` compute in `float` and widen
# the result. The note on each row says so; this is the number the test holds them to.
TOLERANCE = {"exp": 1e-6, "ln": 1e-6, "log10": 1e-6, "log2": 1e-6, "sqrt": 1e-6, "power": 1e-6,
             "hash_mean": 1e-12, "mean": 1e-12, "stddev": 1e-12, "variance": 1e-12}


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
_GKEY = pa.array([0, 1, 0, 2, 1, 2], type=pa.int32())
_GVAL = pa.array([10, 20, 30, None, 50, 60], type=pa.int32())
_STR_WS = pa.array(["a b", "x", None, "c  d e", "q"])
_LIST_NE = pa.array([[1, 2, 3], [4, 5], None, [7]], type=pa.list_(pa.int64()))
_F32 = pa.array([3.5, 1.25, 4.0, 0.5, 2.75, None], type=pa.float32())
_F32B = pa.array([2.0, 3.0, 0.5, 1.5, 2.5, None], type=pa.float32())


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


def _oracle_hash(agg):
    """The same grouped aggregate through pyarrow's own hash kernels, laid out by dense key."""
    def run(args, options):
        k = options["key_count"]
        if agg == "count_all":
            counts = {r["values"]: r["counts"] for r in pc.value_counts(args[0]).to_pylist()}
            return [counts.get(i, 0) for i in range(k)]
        else:
            table = pa.table({"key": args[0], "value": args[1]})
            result = table.group_by("key").aggregate([("value", agg)])
            column = "value_" + agg
        if agg == "min_max":
            rows = dict(zip(result["key"].to_pylist(), result[column].to_pylist()))
            return [[(rows.get(i) or {}).get("min") for i in range(k)],
                    [(rows.get(i) or {}).get("max") for i in range(k)]]
        rows = dict(zip(result["key"].to_pylist(), result[column].to_pylist()))
        return [rows.get(i) for i in range(k)]
    return run


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
    ("kurtosis", "Aggregations", MISSING, "-", "-",
     "Not implemented and not on the roadmap. `variance` and `stddev` ship; the third and fourth "
     "moments do not.", None, None),
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
    ("skew", "Aggregations", MISSING, "-", "-",
     "Not implemented and not on the roadmap, like `kurtosis`.", None, None),
    ("stddev", "Aggregations", GPU, "Kernels/Aggregates.swift", "stddev(ddof)",
     "Compensated squared deviations on the GPU: about 1e-7 relative for float32, 1e-15 for float64.",
     _u("stddev", ddof=0), ((_FLT,), {})),
    ("sum", "Aggregations", GPU, "Kernels/Reductions.swift", "sum()",
     "Integers accumulate in int64/uint64 and wrap; float64 uses the software binary64 adder.",
     _u("sum"), ((_INT,), {})),
    ("tdigest", "Aggregations", MISSING, "-", "-",
     "Out of scope: `quantile` here is exact, so there is no sketch to approximate it with.", None, None),
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
    ("exp", "Arithmetic", PARTIAL, "Kernels/Rounding.swift", "exp()",
     "Evaluated in `float` even for a float64 column: Metal has no `double` transcendentals, so the "
     "answer is correct to about float32 precision (1e-7 relative) rather than to a float64 ulp.",
     _u("exp"), ((_FLT2,), {})),
    ("multiply", "Arithmetic", GPU, "Kernels/Arithmetic.swift", "a * b", "Wrapping on integers.",
     _op("*"), ((_INT, _INT2), {})),
    ("negate", "Arithmetic", GPU, "Kernels/Rounding.swift", "negate()", "Wrapping on integers.",
     _u("negate"), ((_INT,), {})),
    ("power", "Arithmetic", PARTIAL, "Kernels/Rounding.swift", "power(other)",
     "Element-wise `pow` on **float32 only**: a float64 column raises rather than losing precision "
     "silently (Metal has no `double` transcendental to call). Cast first.",
     _b("power"), ((_F32, _F32B), {})),
    ("sign", "Arithmetic", GPU, "Kernels/Rounding.swift", "sign()", "-1 / 0 / 1.",
     _u("sign"), ((_INT,), {})),
    ("sqrt", "Arithmetic", PARTIAL, "Kernels/Rounding.swift", "sqrt()",
     "A negative input gives NaN, as Arrow's unchecked `sqrt` does. Evaluated in `float` for float64 "
     "columns too, so the last digits differ from a float64 square root.",
     _u("sqrt"), ((_FLT_POS,), {})),
    ("subtract", "Arithmetic", GPU, "Kernels/Arithmetic.swift", "a - b", "Wrapping on integers.",
     _op("-"), ((_INT, _INT2), {})),
    ("abs_checked", "Arithmetic", PENDING, "-", "-", "Owned by the checked-arithmetic feature.", None, None),
    ("add_checked", "Arithmetic", PENDING, "-", "-", "Owned by the checked-arithmetic feature.", None, None),
    ("divide_checked", "Arithmetic", PENDING, "-", "-", "Owned by the checked-arithmetic feature.", None, None),
    ("multiply_checked", "Arithmetic", PENDING, "-", "-", "Owned by the checked-arithmetic feature.", None, None),
    ("negate_checked", "Arithmetic", PENDING, "-", "-", "Owned by the checked-arithmetic feature.", None, None),
    ("power_checked", "Arithmetic", PENDING, "-", "-", "Owned by the checked-arithmetic feature.", None, None),
    ("sqrt_checked", "Arithmetic", PENDING, "-", "-", "Owned by the checked-arithmetic feature.", None, None),
    ("subtract_checked", "Arithmetic", PENDING, "-", "-", "Owned by the checked-arithmetic feature.", None, None),
    ("expm1", "Arithmetic", PENDING, "-", "-", "Owned by the checked-arithmetic feature.", None, None),
    ("hypot", "Arithmetic", PENDING, "-", "-", "Owned by the checked-arithmetic feature.", None, None),

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
    ("shift_left_checked", "Bitwise", PENDING, "-", "-", "Owned by the checked-arithmetic feature.", None, None),
    ("shift_right_checked", "Bitwise", PENDING, "-", "-", "Owned by the checked-arithmetic feature.", None, None),

    # ---- Rounding ----------------------------------------------------------
    ("ceil", "Rounding", GPU, "Kernels/Rounding.swift", "ceil()", "One thread per element.",
     _u("ceil"), ((_FLT,), {})),
    ("floor", "Rounding", GPU, "Kernels/Rounding.swift", "floor()", "One thread per element.",
     _u("floor"), ((_FLT,), {})),
    ("trunc", "Rounding", GPU, "Kernels/Rounding.swift", "trunc()", "One thread per element.",
     _u("trunc"), ((_FLT,), {})),
    ("round", "Rounding", PARTIAL, "Kernels/Rounding.swift", "round()",
     "Rounds half **away from zero**. Arrow's `round_mode` (whose default is `half_to_even`) and its "
     "`ndigits` option are not implemented, so a value exactly on a half differs from pyarrow.",
     _u("round"), ((_FLT_ROUND,), {})),
    ("round_binary", "Rounding", PENDING, "-", "-", "Owned by the checked-arithmetic feature.", None, None),
    ("round_to_multiple", "Rounding", PENDING, "-", "-", "Owned by the checked-arithmetic feature.", None, None),

    # ---- Logarithmic -------------------------------------------------------
    ("ln", "Logarithmic", PARTIAL, "Kernels/Rounding.swift", "ln()",
     "Metal's `log`, evaluated in `float` even for a float64 column, so the answer is correct to about float32 "
     "precision (1e-7 relative) rather than to a float64 ulp.", _u("ln"), ((_FLT_POS,), {})),
    ("log10", "Logarithmic", PARTIAL, "Kernels/Rounding.swift", "log10()",
     "Metal's `log10`, evaluated in `float` even for a float64 column, so the answer is correct to about float32 "
     "precision (1e-7 relative) rather than to a float64 ulp.", _u("log10"), ((_FLT_POS,), {})),
    ("log2", "Logarithmic", PARTIAL, "Kernels/Rounding.swift", "log2()",
     "Metal's `log2`, evaluated in `float` even for a float64 column, so the answer is correct to about float32 "
     "precision (1e-7 relative) rather than to a float64 ulp.", _u("log2"), ((_FLT_POS,), {})),
    ("ln_checked", "Logarithmic", PENDING, "-", "-", "Owned by the checked-arithmetic feature.", None, None),
    ("log10_checked", "Logarithmic", PENDING, "-", "-", "Owned by the checked-arithmetic feature.", None, None),
    ("log2_checked", "Logarithmic", PENDING, "-", "-", "Owned by the checked-arithmetic feature.", None, None),
    ("log1p", "Logarithmic", PENDING, "-", "-", "Owned by the checked-arithmetic feature.", None, None),
    ("log1p_checked", "Logarithmic", PENDING, "-", "-", "Owned by the checked-arithmetic feature.", None, None),
    ("logb", "Logarithmic", PENDING, "-", "-", "Owned by the checked-arithmetic feature.", None, None),
    ("logb_checked", "Logarithmic", PENDING, "-", "-", "Owned by the checked-arithmetic feature.", None, None),

    # ---- Trigonometric (all owned by the trig / conditional feature) -------
    ("acos", "Trigonometric", PENDING, "-", "-", "Owned by the trig feature.", None, None),
    ("acos_checked", "Trigonometric", PENDING, "-", "-", "Owned by the trig feature.", None, None),
    ("acosh", "Trigonometric", PENDING, "-", "-", "Owned by the trig feature.", None, None),
    ("acosh_checked", "Trigonometric", PENDING, "-", "-", "Owned by the trig feature.", None, None),
    ("asin", "Trigonometric", PENDING, "-", "-", "Owned by the trig feature.", None, None),
    ("asin_checked", "Trigonometric", PENDING, "-", "-", "Owned by the trig feature.", None, None),
    ("asinh", "Trigonometric", PENDING, "-", "-", "Owned by the trig feature.", None, None),
    ("atan", "Trigonometric", PENDING, "-", "-", "Owned by the trig feature.", None, None),
    ("atan2", "Trigonometric", PENDING, "-", "-", "Owned by the trig feature.", None, None),
    ("atanh", "Trigonometric", PENDING, "-", "-", "Owned by the trig feature.", None, None),
    ("atanh_checked", "Trigonometric", PENDING, "-", "-", "Owned by the trig feature.", None, None),
    ("cos", "Trigonometric", PENDING, "-", "-", "Owned by the trig feature.", None, None),
    ("cos_checked", "Trigonometric", PENDING, "-", "-", "Owned by the trig feature.", None, None),
    ("cosh", "Trigonometric", PENDING, "-", "-", "Owned by the trig feature.", None, None),
    ("sin", "Trigonometric", PENDING, "-", "-", "Owned by the trig feature.", None, None),
    ("sin_checked", "Trigonometric", PENDING, "-", "-", "Owned by the trig feature.", None, None),
    ("sinh", "Trigonometric", PENDING, "-", "-", "Owned by the trig feature.", None, None),
    ("tan", "Trigonometric", PENDING, "-", "-", "Owned by the trig feature.", None, None),
    ("tan_checked", "Trigonometric", PENDING, "-", "-", "Owned by the trig feature.", None, None),
    ("tanh", "Trigonometric", PENDING, "-", "-", "Owned by the trig feature.", None, None),

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
    ("and_not", "Logical", PENDING, "-", "-", "Owned by the trig / conditional feature.", None, None),
    ("and_not_kleene", "Logical", PENDING, "-", "-", "Owned by the trig / conditional feature.", None, None),
    ("xor", "Logical", PENDING, "-", "-", "Owned by the trig / conditional feature.", None, None),

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
    ("ascii_is_printable", "StringPredicates", PENDING, "-", "-", "Owned by the string-predicates feature.", None, None),
    ("ascii_is_title", "StringPredicates", PENDING, "-", "-", "Owned by the string-predicates feature.", None, None),
    ("string_is_ascii", "StringPredicates", PENDING, "-", "-", "Owned by the string-predicates feature.", None, None),
    ("utf8_is_alnum", "StringPredicates", PENDING, "-", "-", "Owned by the string-predicates feature.", None, None),
    ("utf8_is_alpha", "StringPredicates", PENDING, "-", "-", "Owned by the string-predicates feature.", None, None),
    ("utf8_is_decimal", "StringPredicates", PENDING, "-", "-", "Owned by the string-predicates feature.", None, None),
    ("utf8_is_digit", "StringPredicates", PENDING, "-", "-", "Owned by the string-predicates feature.", None, None),
    ("utf8_is_lower", "StringPredicates", PENDING, "-", "-", "Owned by the string-predicates feature.", None, None),
    ("utf8_is_numeric", "StringPredicates", PENDING, "-", "-", "Owned by the string-predicates feature.", None, None),
    ("utf8_is_printable", "StringPredicates", PENDING, "-", "-", "Owned by the string-predicates feature.", None, None),
    ("utf8_is_space", "StringPredicates", PENDING, "-", "-", "Owned by the string-predicates feature.", None, None),
    ("utf8_is_title", "StringPredicates", PENDING, "-", "-", "Owned by the string-predicates feature.", None, None),
    ("utf8_is_upper", "StringPredicates", PENDING, "-", "-", "Owned by the string-predicates feature.", None, None),

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
     "for single-byte content, which is what the check below feeds it.",
     _u("str_reverse"), ((_STR,), {}),
     lambda args, options: pc.binary_reverse(args[0].cast(pa.binary())).cast(pa.string())),
    ("binary_length", "StringTransforms", GPU, "Sources/ArrowMetal/MetalStringArray.swift", "byte_length()",
     "The offsets difference; no data read at all.", _u("byte_length"), ((_STR,), {})),
    ("utf8_length", "StringTransforms", GPU, "Sources/ArrowMetal/MetalStringArray.swift", "char_length()",
     "Counts non-continuation bytes.", _u("char_length"), ((_STR,), {})),
    ("binary_repeat", "StringTransforms", GPU, "Kernels/StringTransforms.swift", "repeat(n)",
     "Two-pass: a length kernel, a GPU scan into offsets, a byte kernel. `n` is one scalar for the "
     "whole column; Arrow's per-row `num_repeats` array is not implemented.",
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
    ("ascii_title", "StringTransforms", PENDING, "-", "-", "Owned by the string-transforms feature.", None, None),
    ("utf8_capitalize", "StringTransforms", PENDING, "-", "-", "Owned by the string-transforms feature.", None, None),
    ("utf8_title", "StringTransforms", PENDING, "-", "-", "Owned by the string-transforms feature.", None, None),
    ("utf8_normalize", "StringTransforms", PENDING, "-", "-", "Owned by the string-transforms feature.", None, None),
    ("utf8_replace_slice", "StringTransforms", PENDING, "-", "-", "Owned by the string-transforms feature.", None, None),
    ("binary_replace_slice", "StringTransforms", PENDING, "-", "-", "Owned by the string-transforms feature.", None, None),

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
    ("ascii_center", "StringPadding", PENDING, "-", "-", "Owned by the string-transforms feature.", None, None),
    ("utf8_center", "StringPadding", PENDING, "-", "-", "Owned by the string-transforms feature.", None, None),

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
    ("utf8_trim_whitespace", "StringTrimming", PARTIAL, "Kernels/StringTransforms.swift", "trim()",
     "Trims **ASCII** whitespace only. Arrow's `utf8_trim_whitespace` also strips U+00A0, U+2000-U+200A, "
     "U+3000 and the rest of the Unicode whitespace class, which this does not.",
     _u("trim"), ((_STR_PAD,), {})),
    ("utf8_ltrim_whitespace", "StringTrimming", PARTIAL, "Kernels/StringTransforms.swift", "ltrim()",
     "ASCII whitespace only, as `utf8_trim_whitespace`.", _u("ltrim"), ((_STR_PAD,), {})),
    ("utf8_rtrim_whitespace", "StringTrimming", PARTIAL, "Kernels/StringTransforms.swift", "rtrim()",
     "ASCII whitespace only.", _u("rtrim"), ((_STR_PAD,), {})),
    ("utf8_trim", "StringTrimming", PARTIAL, "Kernels/StringTransforms.swift", "trim(characters)",
     "The character set is matched **byte by byte**, so a multi-byte character in `characters` trims "
     "its individual bytes rather than the whole code point. Equal to Arrow for an ASCII set.",
     lambda args, options: _out(_a(args[0]).trim(options["characters"])), ((_STR,), {"characters": "Hlo"})),
    ("utf8_ltrim", "StringTrimming", PARTIAL, "Kernels/StringTransforms.swift", "ltrim(characters)",
     "Byte-wise character set, as `utf8_trim`.",
     lambda args, options: _out(_a(args[0]).ltrim(options["characters"])), ((_STR,), {"characters": "Hlo"})),
    ("utf8_rtrim", "StringTrimming", PARTIAL, "Kernels/StringTransforms.swift", "rtrim(characters)",
     "Byte-wise character set, as `utf8_trim`.",
     lambda args, options: _out(_a(args[0]).rtrim(options["characters"])), ((_STR,), {"characters": "Hlo"})),

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
    ("extract_regex_span", "StringExtraction", PENDING, "-", "-", "Owned by the string-transforms feature.", None, None),
    ("binary_join", "StringJoining", PENDING, "-", "-", "Owned by the string-transforms feature.", None, None),
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
     "Not implemented: the slice kernel counts code points, and there is no byte-offset variant. "
     "Unclaimed rather than out of scope.", None, None),

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
    ("is_in", "Containment", PARTIAL, "Kernels/Structural.swift", "is_in(value_set)",
     "GPU for primitive and temporal columns (a sorted set plus a binary search per row). String and "
     "binary value sets are not implemented; Arrow's `null_matching_behavior` is fixed at `match`.",
     lambda args, options: _out(_a(args[0]).is_in(options["value_set"])),
     ((_INT,), {"value_set": pa.array([1, 4], type=pa.int64())})),
    ("index_in", "Containment", PARTIAL, "Kernels/Structural.swift", "index_in(value_set)",
     "Same kernel as `is_in`, returning the int32 position in the set. String value sets are not "
     "implemented.",
     lambda args, options: _out(_a(args[0]).index_in(options["value_set"])),
     ((_INT,), {"value_set": pa.array([1, 4], type=pa.int64())})),

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
    ("indices_nonzero", "Categorizations", PENDING, "-", "-", "Owned by the trig / conditional feature.", None, None),
    ("is_finite", "Categorizations", PENDING, "-", "-", "Owned by the trig / conditional feature.", None, None),
    ("is_inf", "Categorizations", PENDING, "-", "-", "Owned by the trig / conditional feature.", None, None),
    ("is_nan", "Categorizations", PENDING, "-", "-", "Owned by the trig / conditional feature.", None, None),

    # ---- Selecting ---------------------------------------------------------
    ("if_else", "Selecting", GPU, "Kernels/Structural.swift", "if_else(left, right)",
     "One thread per element; a null condition gives a null output.",
     lambda args, options: _out(_a(args[0]).if_else(args[1], args[2])), ((_BOOL, _INT, _INT2), {})),
    ("coalesce", "Selecting", GPU, "Kernels/Structural.swift", "am.coalesce(*arrays)",
     "First non-null across the inputs, one thread per element.",
     lambda args, options: _out(_coalesce(*[_a(x) for x in args])), ((_INT, _INT2), {})),
    ("case_when", "Selecting", PENDING, "-", "-", "Owned by the trig / conditional feature.", None, None),
    ("choose", "Selecting", PENDING, "-", "-", "Owned by the trig / conditional feature.", None, None),

    # ---- Conversions -------------------------------------------------------
    ("cast", "Conversions", PARTIAL, "Kernels/Cast.swift", "cast(target)",
     "Numeric to numeric on the GPU (wrapping, C-style, no overflow check), numeric to/from utf8 on "
     "the GPU, temporal unit changes on the GPU. Overflow-erroring casts, decimal casts and casts "
     "between nested types are not implemented.",
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
    ("day_of_week", "TemporalExtraction", GPU, "Sources/ArrowMetal/Temporal.swift", "day_of_week()",
     "Monday = 0, matching Arrow's default `week_start=1, count_from_zero=True`.",
     _u("day_of_week"), ((_TS,), {})),
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
    ("is_dst", "TemporalExtraction", PENDING, "-", "-", "Owned by the timezone / types feature.", None, None),
    ("iso_calendar", "TemporalExtraction", PENDING, "-", "-", "Owned by the temporal feature.", None, None),
    ("subsecond", "TemporalExtraction", PENDING, "-", "-", "Owned by the temporal feature.", None, None),
    ("us_week", "TemporalExtraction", PENDING, "-", "-", "Owned by the temporal feature.", None, None),
    ("us_year", "TemporalExtraction", PENDING, "-", "-", "Owned by the temporal feature.", None, None),
    ("week", "TemporalExtraction", PENDING, "-", "-", "Owned by the temporal feature.", None, None),
    ("year_month_day", "TemporalExtraction", PENDING, "-", "-", "Owned by the temporal feature.", None, None),

    # ---- Temporal differences ---------------------------------------------
    ("days_between", "TemporalDifference", GPU, "Kernels/TemporalMath.swift", "days_between(other)",
     "Whole days between two temporal columns, UTC.",
     lambda args, options: _out(_a(args[0]).days_between(_a(args[1]))), ((_TS, _TS2), {})),
    ("day_time_interval_between", "TemporalDifference", PENDING, "-", "-", "Owned by the temporal feature.", None, None),
    ("hours_between", "TemporalDifference", PENDING, "-", "-", "Owned by the temporal feature.", None, None),
    ("microseconds_between", "TemporalDifference", PENDING, "-", "-", "Owned by the temporal feature.", None, None),
    ("milliseconds_between", "TemporalDifference", PENDING, "-", "-", "Owned by the temporal feature.", None, None),
    ("minutes_between", "TemporalDifference", PENDING, "-", "-", "Owned by the temporal feature.", None, None),
    ("month_day_nano_interval_between", "TemporalDifference", PENDING, "-", "-", "Owned by the temporal feature.", None, None),
    ("month_interval_between", "TemporalDifference", PENDING, "-", "-", "Owned by the temporal feature.", None, None),
    ("nanoseconds_between", "TemporalDifference", PENDING, "-", "-", "Owned by the temporal feature.", None, None),
    ("quarters_between", "TemporalDifference", PENDING, "-", "-", "Owned by the temporal feature.", None, None),
    ("seconds_between", "TemporalDifference", PENDING, "-", "-", "Owned by the temporal feature.", None, None),
    ("weeks_between", "TemporalDifference", PENDING, "-", "-", "Owned by the temporal feature.", None, None),
    ("years_between", "TemporalDifference", PENDING, "-", "-", "Owned by the temporal feature.", None, None),

    # ---- Timezone ----------------------------------------------------------
    ("assume_timezone", "Timezone", PENDING, "-", "-", "Owned by the timezone / types feature.", None, None),
    ("local_timestamp", "Timezone", PENDING, "-", "-", "Owned by the timezone / types feature.", None, None),

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
    ("fill_null_forward", "NullFilling", PENDING, "-", "-", "Owned by the trig / conditional feature.", None, None),
    ("fill_null_backward", "NullFilling", PENDING, "-", "-", "Owned by the trig / conditional feature.", None, None),

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
    ("list_parent_indices", "Structural", MISSING, "-", "-",
     "Not implemented. The offsets are on the device and the kernel would be a scan, so this is "
     "unclaimed rather than out of scope.", None, None),
    ("list_slice", "Structural", PENDING, "-", "-", "Owned by the remaining-types feature.", None, None),
    ("map_lookup", "Structural", PENDING, "-", "-", "Owned by the remaining-types feature.", None, None),
    ("replace_with_mask", "Structural", PENDING, "-", "-", "Owned by the trig / conditional feature.", None, None),

    # ---- Pairwise and cumulative ------------------------------------------
    ("pairwise_diff", "Pairwise", GPU, "Kernels/Window.swift", "pairwise_diff(period)",
     "One thread per element; null where either side is null or outside the array. Integers wrap.",
     lambda args, options: _out(_a(args[0]).pairwise_diff(options.get("period", 1))), ((_INT,), {})),
    ("pairwise_diff_checked", "Pairwise", PENDING, "-", "-", "Owned by the checked-arithmetic feature.", None, None),
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
    ("cumulative_sum_checked", "Cumulative", PENDING, "-", "-", "Owned by the checked-arithmetic feature.", None, None),
    ("cumulative_prod_checked", "Cumulative", PENDING, "-", "-", "Owned by the checked-arithmetic feature.", None, None),

    # ---- Grouped aggregates (hash_*) ---------------------------------------
    ("hash_count", "GroupedAggregations", PARTIAL, "Kernels/GroupBy.swift", "group_by(k).count()",
     "GPU, but only over **dense integer keys** in [0, k): the caller supplies k. Arbitrary keys go "
     "through `dictionary_encode` first. `mode` is not implemented.", _hash("count_values"),
     ((_GKEY, _GVAL), {"key_count": 3}), _oracle_hash("count")),
    ("hash_count_all", "GroupedAggregations", PARTIAL, "Kernels/GroupBy.swift", "group_by(k).count()",
     "Rows per key, dense integer keys only.", _hash("count_all"),
     ((_GKEY,), {"key_count": 3}), _oracle_hash("count_all")),
    ("hash_sum", "GroupedAggregations", PARTIAL, "Kernels/GroupBy.swift", "group_by(k).sum(values)",
     "Dense integer keys only; `min_count` is not implemented.", _hash("sum"),
     ((_GKEY, _GVAL), {"key_count": 3}), _oracle_hash("sum")),
    ("hash_mean", "GroupedAggregations", PARTIAL, "Kernels/GroupBy.swift", "group_by(k).mean(values)",
     "Dense integer keys only.", _hash("mean"),
     ((_GKEY, _GVAL), {"key_count": 3}), _oracle_hash("mean")),
    ("hash_min", "GroupedAggregations", PARTIAL, "Kernels/GroupBy.swift", "group_by(k).min(values)",
     "Dense integer keys only.", _hash("min"),
     ((_GKEY, _GVAL), {"key_count": 3}), _oracle_hash("min")),
    ("hash_max", "GroupedAggregations", PARTIAL, "Kernels/GroupBy.swift", "group_by(k).max(values)",
     "Dense integer keys only.", _hash("max"),
     ((_GKEY, _GVAL), {"key_count": 3}), _oracle_hash("max")),
    ("hash_min_max", "GroupedAggregations", PARTIAL, "Kernels/GroupBy.swift", "group_by(k).min/.max",
     "The two calls; there is no fused form and no struct output.", _hash("min_max"),
     ((_GKEY, _GVAL), {"key_count": 3}), _oracle_hash("min_max")),
    ("hash_all", "GroupedAggregations", PENDING, "-", "-",
     "`GroupBy.all(_:)` is GPU over dense integer keys in Swift today, with no C ABI entry point yet. The grouped-aggregates feature owns wiring it through.", None, None),
    ("hash_any", "GroupedAggregations", PENDING, "-", "-",
     "`GroupBy.any(_:)` is GPU over dense integer keys in Swift today, but there is no C ABI entry point and so no Python binding. The grouped-aggregates feature owns wiring it through.", None, None),
    ("hash_approximate_median", "GroupedAggregations", PENDING, "-", "-",
     "`GroupBy.approximateMedian(_:)` exists but throws: a grouped median needs a segmented sort kernel this tree does not have. The grouped-aggregates feature owns it.", None, None),
    ("hash_count_distinct", "GroupedAggregations", PENDING, "-", "-",
     "`GroupBy.countDistinct(_:)` is GPU over dense integer keys in Swift today, with no C ABI entry point yet. The grouped-aggregates feature owns it.", None, None),
    ("hash_distinct", "GroupedAggregations", PENDING, "-", "-",
     "Not implemented; `hash_count_distinct` counts them but does not return them. The grouped-aggregates feature owns it.", None, None),
    ("hash_first", "GroupedAggregations", PENDING, "-", "-",
     "`GroupBy.first(_:)` is GPU over dense integer keys in Swift today, with no C ABI entry point yet. The grouped-aggregates feature owns it.", None, None),
    ("hash_first_last", "GroupedAggregations", PENDING, "-", "-", "Owned by the grouped-aggregates feature.", None, None),
    ("hash_last", "GroupedAggregations", PENDING, "-", "-",
     "`GroupBy.last(_:)` is GPU over dense integer keys in Swift today, with no C ABI entry point yet. The grouped-aggregates feature owns it.", None, None),
    ("hash_list", "GroupedAggregations", PENDING, "-", "-", "Owned by the grouped-aggregates feature.", None, None),
    ("hash_one", "GroupedAggregations", PENDING, "-", "-", "Owned by the grouped-aggregates feature.", None, None),
    ("hash_product", "GroupedAggregations", PENDING, "-", "-",
     "`GroupBy.product(_:)` exists in Swift and runs on the host (Metal has no 64-bit atomic multiply), with no C ABI entry point yet. The grouped-aggregates feature owns it.", None, None),
    ("hash_stddev", "GroupedAggregations", PENDING, "-", "-",
     "`GroupBy.stddev(_:)` is GPU over dense integer keys in Swift today (float32 deviations, about 1e-6 relative), with no C ABI entry point yet. The grouped-aggregates feature owns it.", None, None),
    ("hash_variance", "GroupedAggregations", PENDING, "-", "-",
     "`GroupBy.variance(_:)` is GPU over dense integer keys in Swift today, with no C ABI entry point yet. The grouped-aggregates feature owns it.", None, None),
    ("hash_pivot_wider", "GroupedAggregations", PENDING, "-", "-", "Owned by the grouped-aggregates feature.", None, None),
    ("hash_kurtosis", "GroupedAggregations", MISSING, "-", "-",
     "Not implemented and not on the roadmap, like the scalar `kurtosis`.", None, None),
    ("hash_skew", "GroupedAggregations", MISSING, "-", "-",
     "Not implemented and not on the roadmap, like the scalar `skew`.", None, None),
    ("hash_tdigest", "GroupedAggregations", MISSING, "-", "-",
     "Out of scope, like the scalar `tdigest`.", None, None),
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
    """The registry as a Markdown table, ready to paste into docs/COVERAGE.md."""
    lines = ["| Arrow function | Section | Status | Implementation | ArrowMetal call | Notes |",
             "|---|---|---|---|---|---|"]
    wanted = sections or SECTIONS
    for section in wanted:
        for f in _REGISTRY.values():
            if f.section != section:
                continue
            status = {GPU: "**GPU**", CPU: "**CPU**", PARTIAL: "**Partial**",
                      MISSING: "Missing", PENDING: "Pending"}[f.status]
            notes = f.notes.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{f.name}` | {f.section} | {status} | `{f.swift_file}` | "
                         f"`{f.method}` | {notes} |")
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
