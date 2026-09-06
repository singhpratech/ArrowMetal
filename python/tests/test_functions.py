"""The Arrow function-name registry, checked against pyarrow.compute name by name.

Four things are asserted here, and the third one is the point of the file:

1. Every Arrow compute function name exists in the registry — measured against
   ``pyarrow.compute.list_functions()`` itself, so the list cannot drift as Arrow grows.
2. Every row that claims to work (``gpu`` / ``cpu`` / ``partial``) actually runs through
   :func:`arrowmetal.functions.call_function` on a small array, and its answer matches
   ``pyarrow.compute`` for at least one input type. A row marked ``pending`` would be skipped with
   its owning feature named — no row carries that status today — and ``missing`` rows are asserted
   *not* to run.
3. A second input is fed to the rows whose claim spans several Arrow **type families**, so
   "runs on numeric columns" is not asserted from one int64 array: unsigned and float32 numerics,
   booleans, binary, temporal, dictionary and list inputs each get a pass through the same
   :func:`call_function` entry point.
4. The whole table prints as Markdown, so the maintainer can paste it into ``docs/ARROW_FUNCTIONS.md``
   (``python/tests/function_table_report.py`` writes the same thing to stdout).

    PYTHONPATH=python python -m pytest python/tests/test_functions.py -q -s
"""
import math

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import arrowmetal as am
from arrowmetal import functions as F

# pyarrow registers two internal meta-functions that have no Arrow docs entry and no user-facing
# signature; and the Arrow docs name three functions that pyarrow exposes only as Python wrappers.
_PYARROW_ONLY = {"index_in_meta_binary", "is_in_meta_binary"}
_DOCS_ONLY = {"top_k_unstable", "bottom_k_unstable", "fill_null"}
_PYARROW_ALIASES = {"and": "and_", "or": "or_"}


def _arrow_names():
    return {_PYARROW_ALIASES.get(n, n) for n in pc.list_functions()} - _PYARROW_ONLY


# ---------------------------------------------------------------- (a) every name is present


def test_registry_covers_every_arrow_name():
    missing = sorted(_arrow_names() - set(F.list_functions()))
    assert not missing, f"Arrow names with no registry row: {missing}"


def test_registry_invents_no_names():
    extra = sorted(set(F.list_functions()) - _arrow_names() - _DOCS_ONLY)
    assert not extra, f"registry rows that are not Arrow function names: {extra}"


def test_every_row_is_well_formed():
    for f in F.function_table():
        rec = F.get_function(f["name"])
        assert rec.notes and rec.notes.strip(), f"{rec.name} has no note"
        assert rec.status in (F.GPU, F.CPU, F.PARTIAL, F.MISSING, F.PENDING), rec.status
        if rec.runnable:
            assert rec.call is not None, f"{rec.name} claims {rec.status} but has no call"
            assert rec.example is not None, f"{rec.name} claims {rec.status} but has no example"
            assert rec.swift_file != "-", f"{rec.name} claims {rec.status} but names no source file"
        else:
            assert rec.call is None, f"{rec.name} is {rec.status} but carries a call"


def test_missing_and_pending_rows_raise():
    for rec in (F.get_function(n) for n in F.list_functions()):
        if rec.runnable:
            continue
        with pytest.raises(NotImplementedError):
            F.call_function(rec.name, [pa.array([1, 2, 3])])


# ---------------------------------------------------------------- (b) every runnable row runs


def _py(x):
    """Anything pyarrow hands back, as plain Python."""
    if isinstance(x, (pa.Array, pa.ChunkedArray)):
        return x.to_pylist()
    if isinstance(x, pa.Scalar):
        return x.as_py()
    return x


def _close(a, b, tol=1e-9):
    """Structural equality with a float tolerance, over nested lists and dicts."""
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        if isinstance(a, float) and math.isnan(a):
            return isinstance(b, float) and math.isnan(b)
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol * 1e-3)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_close(a[k], b[k], tol) for k in a)
    return a == b


def _check_property(name, got, args, options):
    """The rows whose answer legitimately differs from Arrow's, checked by their contract instead.

    `select_k_unstable`, `top_k_unstable` and `bottom_k_unstable` are *unstable* by name: two
    implementations may pick different rows out of a tie, so the selected **values** are what has to
    agree. `partition_nth_indices` returns a full sort here rather than Arrow's partial partition, so
    its contract — the n smallest first — is what gets checked. `random` shares no stream with Arrow,
    so range and determinism are what matter.
    """
    if name in ("select_k_unstable", "top_k_unstable", "bottom_k_unstable"):
        order = "descending" if (name == "top_k_unstable" or options.get("largest")) else "ascending"
        want = pc.select_k_unstable(args[0], k=options["k"], sort_keys=[("", order)])
        values = args[0].to_pylist()
        assert sorted(values[i] for i in got.to_pylist()) == sorted(values[i] for i in want.to_pylist()), \
            f"{name} picked a different multiset of values than pyarrow"
        return
    if name == "partition_nth_indices":
        idx = [i for i in got.to_pylist()]
        n = options["pivot"]
        values = args[0].to_pylist()
        head = [values[i] for i in idx[:n] if values[i] is not None]
        tail = [values[i] for i in idx[n:] if values[i] is not None]
        assert not head or not tail or max(head) <= min(tail), \
            f"partition_nth_indices({n}) did not separate the {n} smallest"
        assert sorted(idx) == list(range(len(values))), "not a permutation"
        return
    if name == "random":
        values = got.to_pylist()
        assert len(values) == options["n"]
        assert all(0.0 <= v < 1.0 for v in values), "random values must live in [0, 1)"
        again = am.random(options["n"], options["initializer"]).to_arrow().to_pylist()
        assert values == again, "the same seed must give the same stream"
        assert values != am.random(options["n"], options["initializer"] + 1).to_arrow().to_pylist()
        return
    raise AssertionError(f"no property check for {name}")


RUNNABLE = [n for n in F.list_functions() if F.get_function(n).runnable]
NOT_RUNNABLE = [n for n in F.list_functions() if not F.get_function(n).runnable]


@pytest.mark.parametrize("name", RUNNABLE)
def test_runnable_row_matches_pyarrow(name):
    rec = F.get_function(name)
    args, options = rec.example
    got = F.call_function(name, args, options)

    if rec.oracle is not None:
        want = rec.oracle(list(args), dict(options))
        if isinstance(want, str) and want.startswith("property:"):
            _check_property(name, got, list(args), dict(options))
            return
    else:
        want = getattr(pc, name)(*args, **options)

    tol = F.TOLERANCE.get(name, 1e-9)
    assert _close(_py(got), _py(want), tol), f"{name}: ArrowMetal {_py(got)!r} != pyarrow {_py(want)!r}"


# ------------------------------------------- (c) a second type family for the rows that claim one

_UINT = pa.array([3, 1, 4, 1, 5, None], type=pa.uint32())
_UINT2 = pa.array([2, 7, 2, 3, 1, None], type=pa.uint32())
_F32 = pa.array([3.5, -1.25, 4.0, 0.5, 2.75, None], type=pa.float32())
_F32B = pa.array([2.0, 3.0, 0.5, 1.5, 2.5, None], type=pa.float32())
_BOOLC = pa.array([True, False, True, None, False, True])
_BIN = pa.array([b"abc", b"z", None, b"", b"xy", b"qrst"], type=pa.binary())
_DATE = pa.array([18000, 19000, None, 0], type=pa.date32())
_DATE2 = pa.array([18100, 19000, None, 366], type=pa.date32())
_DICT = pa.array(["a", "b", "a", None, "c", "b"]).dictionary_encode()
_LIST = pa.array([[1, 2, 3], [4], None, []], type=pa.list_(pa.int64()))
_MASK = pa.array([True, False, True, None, True, False])
_IDX32 = pa.array([2, 0, 1, 5, 4, 3], type=pa.int32())

# (Arrow name, args, options, tolerance) — one extra representative input per row, in a type family
# the row's note claims but the row's own example does not exercise. `tolerance` is None to use the
# registry's own; a float32 column needs its own number, because a kernel that accumulates or
# evaluates in `float` is accurate to float32 rather than to the float64 the row's example measures.
# Everything here goes through the same `call_function` the parametrised test above uses, and is
# compared against pyarrow the same way.
_F32_ACCUM = 1e-7        # float32 accumulation (variance, mean) and the float32 transcendentals
_SECOND_FAMILY = [
    # unsigned integers
    ("add", (_UINT, _UINT2), {}, None), ("subtract", (_UINT2, _UINT), {}, None),
    ("multiply", (_UINT, _UINT2), {}, None), ("sum", (_UINT,), {}, None), ("min", (_UINT,), {}, None),
    ("max", (_UINT,), {}, None), ("less", (_UINT, _UINT2), {}, None),
    ("bit_wise_and", (_UINT, _UINT2), {}, None), ("cumulative_sum", (_UINT,), {}, None),
    ("unique", (_UINT,), {}, None), ("is_finite", (_UINT,), {}, None),
    ("hash_sum", (_DICT, _UINT), {}, None),
    # float32
    ("add", (_F32, _F32B), {}, None), ("abs", (_F32,), {}, None), ("mean", (_F32,), {}, _F32_ACCUM),
    ("variance", (_F32,), {}, _F32_ACCUM), ("min_max", (_F32,), {}, None),
    ("sin", (_F32,), {}, _F32_ACCUM), ("is_nan", (_F32,), {}, None), ("round", (_F32,), {}, None),
    ("array_sort_indices", (_F32,), {}, None), ("cumulative_max", (_F32,), {}, None),
    # boolean
    ("count", (_BOOLC,), {}, None), ("filter", (_UINT, _MASK), {}, None),
    ("fill_null", (_BOOLC,), {"fill_value": False}, None), ("is_null", (_BOOLC,), {}, None),
    ("indices_nonzero", (_BOOLC,), {}, None),
    # binary (the two string rows that do take a binary column; see `binary_length`'s note for the
    # ones that do not)
    ("is_in", (_BIN,), {"value_set": pa.array([b"z", b"xy"])}, None),
    ("index_in", (_BIN,), {"value_set": pa.array([b"z", b"xy"])}, None),
    ("take", (_BIN, _IDX32), {}, None),
    # temporal (date32 rather than timestamp[us])
    ("year", (_DATE,), {}, None), ("day_of_year", (_DATE,), {}, None), ("is_leap_year", (_DATE,), {}, None),
    ("days_between", (_DATE, _DATE2), {}, None), ("years_between", (_DATE, _DATE2), {}, None),
    ("week", (_DATE,), {}, None), ("iso_calendar", (_DATE,), {}, None),
    # dictionary and list
    ("dictionary_decode", (_DICT,), {}, None), ("take", (_DICT, _IDX32), {}, None),
    ("list_value_length", (_LIST,), {}, None), ("list_flatten", (_LIST,), {}, None),
    ("list_parent_indices", (_LIST,), {}, None), ("list_slice", (_LIST,), {"start": 0, "stop": 2}, None),
]


@pytest.mark.parametrize("name,args,options,tolerance", _SECOND_FAMILY,
                         ids=[f"{n}-{a[0].type}" for n, a, _, _ in _SECOND_FAMILY])
def test_second_type_family_matches_pyarrow(name, args, options, tolerance):
    rec = F.get_function(name)
    assert rec.runnable, f"{name} is {rec.status}; it has no second family to check"
    got = F.call_function(name, args, options)
    want = rec.oracle(list(args), dict(options)) if rec.oracle is not None else getattr(pc, name)(*args, **options)
    tol = tolerance if tolerance is not None else F.TOLERANCE.get(name, 1e-9)
    assert _close(_py(got), _py(want), tol), \
        f"{name} on {args[0].type}: ArrowMetal {_py(got)!r} != pyarrow {_py(want)!r}"


@pytest.mark.parametrize("name", NOT_RUNNABLE)
def test_not_runnable_row_names_its_reason(name):
    rec = F.get_function(name)
    if rec.status == F.PENDING:
        assert "feature" in rec.notes, f"{name} is pending but does not name the owning feature"
    else:
        assert rec.status == F.MISSING
        assert len(rec.notes) > 20, f"{name} is missing but gives no reason"


# ---------------------------------------------------------------- (c) the Markdown table


def test_print_markdown_table(capsys):
    """Print the per-section counts, and check the full table is complete and well formed.

    The counts are what a reader wants on every run; the full 307-row table is the body of
    docs/ARROW_FUNCTIONS.md, so it comes out of `python/tests/function_table_report.py` instead of
    scrolling past on every test run.
    """
    table = F.markdown_table()
    assert table.count("\n") == len(F.list_functions()) + 1, "one row per Arrow name, plus the header"
    for name in F.list_functions():
        assert f"| `{name}` |" in table, f"{name} is missing from the Markdown table"
    with capsys.disabled():
        print()
        print(F.summary_markdown())
        print("\nFull 307-row table: PYTHONPATH=python python python/tests/function_table_report.py")
