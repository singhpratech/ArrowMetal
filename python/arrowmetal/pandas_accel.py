"""Zero-code-change GPU acceleration for pandas, in the shape of `cudf.pandas`.

    import arrowmetal.pandas_accel
    arrowmetal.pandas_accel.install()

    # ... unchanged pandas code from here on ...

or, without touching the script at all:

    python -m arrowmetal.pandas_accel my_script.py
    ARROWMETAL_PANDAS_ACCEL=1 python my_script.py     # auto-installs on import

`install()` replaces a small, documented set of pandas methods with wrappers that route the work to
the Apple silicon GPU **only** when all three of these hold:

1. the column's dtype is in the registry below (numeric, boolean, or utf8 — Arrow-backed or numpy),
2. the frame is at least `threshold` rows (2,000,000 by default), because below that the GPU never
   wins against pandas after the launch latency, and
3. the arguments are ones the GPU kernel implements exactly (no `key=`, no `min_count`, no regex
   `str.contains`, ...).

Everything else runs in pandas, untouched. **Any exception inside the GPU path is caught, recorded
in `stats()`, and the original pandas method is run instead** — the accel layer must never be the
reason a program fails.

`stats()` reports what ran where. `uninstall()` puts every original method back.

Result fidelity. The wrappers reproduce pandas' own answer, index and dtype: the result dtype is
taken from pandas itself (each wrapper runs the original method over a two-row probe of the same
dtype to learn it), the index is carried along by the same permutation as the values, and a null
coming back from Arrow is turned into whatever pandas would have put there (`NaN`, `False`, `True`
after `!=`, or the dtype's own NA). The one deliberate difference is documented in docs/PANDAS.md:
`sort_values` is *stable* here, where pandas' default `kind="quicksort"` is not, so rows with equal
keys can come out in a different order than pandas (the same order as `kind="stable"`).
"""

import functools
import os
import sys
import threading

import numpy as np
import pandas as pd
import pyarrow as pa

from . import MetalArray
from . import pandas_bridge as _b

__all__ = ["install", "uninstall", "installed", "stats", "reset_stats", "config",
           "threshold", "set_threshold", "disabled", "Stats"]

DEFAULT_THRESHOLD = 2_000_000
_UNSUPPORTED = object()          # "this call is not for the GPU"; the wrapper falls through


# ---------------------------------------------------------------------------------------------
# State and statistics
# ---------------------------------------------------------------------------------------------

class Stats:
    """What ran where since the last `reset_stats()`."""

    def __init__(self):
        self.gpu, self.cpu, self.errors = {}, {}, []
        self.intercepted = 0          # every call that reached a patched method, routed or not

    def _bump(self, table, op):
        table[op] = table.get(op, 0) + 1

    @property
    def gpu_calls(self):
        return sum(self.gpu.values())

    @property
    def cpu_calls(self):
        return sum(self.cpu.values())

    def to_frame(self):
        """A DataFrame with one row per intercepted operation: gpu, cpu (fell back), errors."""
        ops = sorted(set(self.gpu) | set(self.cpu) | {op for op, _ in self.errors})
        errs = {}
        for op, _ in self.errors:
            errs[op] = errs.get(op, 0) + 1
        return pd.DataFrame({"gpu": [self.gpu.get(o, 0) for o in ops],
                             "cpu": [self.cpu.get(o, 0) for o in ops],
                             "errors": [errs.get(o, 0) for o in ops]}, index=pd.Index(ops, name="op"))

    def __repr__(self):
        lines = [f"ArrowMetal pandas accel: {self.gpu_calls} on the GPU, {self.cpu_calls} fell back "
                 f"to pandas, {len(self.errors)} errors"]
        for op in sorted(set(self.gpu) | set(self.cpu)):
            lines.append(f"  {op:<32} gpu={self.gpu.get(op, 0):<8} cpu={self.cpu.get(op, 0)}")
        for op, exc in self.errors[:20]:
            lines.append(f"  ! {op}: {exc}")
        if len(self.errors) > 20:
            lines.append(f"  ! ... and {len(self.errors) - 20} more")
        return "\n".join(lines)


class _Config:
    """Knobs; read `am.pandas_accel.config`."""

    def __init__(self):
        self.threshold = int(os.environ.get("ARROWMETAL_PANDAS_ACCEL_THRESHOLD", DEFAULT_THRESHOLD))
        self.enabled = True
        self.raise_on_error = False      # tests flip this to see the real exception


config = _Config()
_stats = Stats()
_originals = []                          # [(owner, attribute, original)]
_local = threading.local()


def stats():
    """The `Stats` collected so far."""
    return _stats


def reset_stats():
    global _stats
    _stats = Stats()
    return _stats


def threshold():
    return config.threshold


def set_threshold(rows):
    """Rows below which everything stays in pandas. The default is 2,000,000."""
    config.threshold = int(rows)


class disabled:
    """`with am.pandas_accel.disabled(): ...` — run a block in plain pandas."""

    def __enter__(self):
        self._was = config.enabled
        config.enabled = False
        return self

    def __exit__(self, *exc):
        config.enabled = self._was
        return False


def _busy():
    return getattr(_local, "depth", 0) > 0


class _working:
    def __enter__(self):
        _local.depth = getattr(_local, "depth", 0) + 1

    def __exit__(self, *exc):
        _local.depth -= 1
        return False


# ---------------------------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------------------------

_MASKED_NUMERIC = {"Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64",
                   "Float32", "Float64"}


def dtype_family(dtype):
    """`"numeric"`, `"bool"`, `"string"` or None (meaning: leave it to pandas).

    Categoricals, datetimes, timedeltas, objects, intervals and periods are deliberately not
    accelerated: their pandas semantics are richer than the kernels here."""
    if isinstance(dtype, pd.CategoricalDtype):
        return None
    if isinstance(dtype, pd.ArrowDtype):
        t = dtype.pyarrow_dtype
        if pa.types.is_boolean(t):
            return "bool"
        if pa.types.is_integer(t) or pa.types.is_floating(t):
            return "numeric"
        if pa.types.is_string(t) or pa.types.is_large_string(t):
            return "string"
        return None
    if isinstance(dtype, np.dtype):
        if dtype.kind == "b":
            return "bool"
        if dtype.kind in "iuf":
            return "numeric"
        return None
    name = str(dtype)
    if name in _MASKED_NUMERIC:
        return "numeric"
    if name == "boolean":
        return "bool"
    if name in ("string", "str"):
        return "string"
    return None


#: Rows an operation needs before the GPU is worth it, as a multiple of `config.threshold` —
#: or **None, meaning it is intercepted but never routed**, because measurement says pandas does it
#: better. Handing a column to Metal maps its pages once, and that map costs about as much as a
#: whole single-pass pandas kernel (6.7 ms for a 400 MB column against 4.9 ms for pandas' `sum`),
#: while the kernel itself takes 1.1 ms. So an operation that reads each byte once and writes at
#: most one byte back — `sum`, `min`, `max`, `mean`, `count`, `abs`, a scalar comparison — cannot
#: pay for the map, and an operation that does more per byte — a sort, a hash group-by, a merge,
#: a string scan, `round`, `isin`, `nunique` — pays for it many times over.
#:
#: The table is data: edit it, or call `route_all()` to route everything anyway (which is what the
#: `.am` accessor does — see docs/PANDAS.md for the measured ratios behind every entry).
NEVER_BY_DEFAULT = ("sum", "min", "max", "mean", "count", "abs",
                    "eq", "ne", "lt", "le", "gt", "ge")
ROW_FACTOR = {op: None for op in NEVER_BY_DEFAULT}


#: Operations that win on an Arrow-backed column but lose on a numpy-backed one, because pandas'
#: numpy kernel for them is far faster than its pyarrow kernel *and* the column has to be converted
#: to Arrow first. `round` at 10M rows: 252 ms in pandas on an Arrow-backed float column against
#: 4 ms on a numpy one, while the GPU needs about 25 ms either way. Same table, same rules — edit it
#: or clear it. `route_all()` clears this too.
_NUMPY_NEVER_DEFAULT = frozenset({"round", "isin"})
NUMPY_NEVER = set(_NUMPY_NEVER_DEFAULT)


def _numpy_blocked(op, series):
    return op in NUMPY_NEVER and isinstance(series.dtype, np.dtype)


def route_all(on=True):
    """Route every operation in the registry, including the ones pandas does better on its own.

    Correctness is unaffected — this only changes where the work runs. The test suite uses it to
    exercise every GPU path."""
    for op in NEVER_BY_DEFAULT:
        ROW_FACTOR[op] = 1 if on else None
    NUMPY_NEVER.clear() if on else NUMPY_NEVER.update(_NUMPY_NEVER_DEFAULT)


def _big(obj, factor=1):
    return factor is not None and len(obj) >= config.threshold * factor


def _eligible(series, families, factor=1):
    return _big(series, factor) and dtype_family(series.dtype) in families


# ---------------------------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------------------------

def _wrap(op, impl):
    """Builds the replacement for one pandas method."""
    def make(orig):
        @functools.wraps(orig)
        def wrapper(self, *a, **k):
            _stats.intercepted += 1
            if not config.enabled or _busy():
                return orig(self, *a, **k)
            try:
                with _working():
                    out = impl(orig, self, *a, **k)
            except Exception as exc:                      # never surface a GPU failure
                _stats.errors.append((op, f"{type(exc).__name__}: {exc}"))
                if config.raise_on_error:
                    raise
                return orig(self, *a, **k)
            if out is _UNSUPPORTED:
                _stats._bump(_stats.cpu, op)
                return orig(self, *a, **k)
            _stats._bump(_stats.gpu, op)
            return out
        wrapper.__arrowmetal_accel__ = True
        wrapper.__arrowmetal_original__ = orig
        return wrapper
    return make


#: One wrapper object per patched method, reused across install/uninstall cycles so that the
#: interpreter sees a stable function identity for each slot. (Install once. Cycling install and
#: uninstall a hundred-odd times in one process makes CPython's adaptive specializer keep an
#: already-specialized `x[mask]` call site bound to whichever `__getitem__` it first saw — a
#: interpreter-level artifact of repeatedly rewriting a dunder, not something this layer can fix.)
_WRAPPERS = {}


def _patch(owner, name, op, impl):
    current = owner.__dict__.get(name, _MISSING)
    orig = getattr(owner, name) if isinstance(current, _MissingType) else current
    if getattr(orig, "__arrowmetal_accel__", False):
        return
    key = (owner, name)
    cached = _WRAPPERS.get(key)
    if cached is not None and cached[0] is orig:
        wrapper = cached[1]
    else:
        wrapper = _wrap(op, impl)(orig)
        _WRAPPERS[key] = (orig, wrapper)
    _originals.append((owner, name, current))
    setattr(owner, name, wrapper)


class _MissingType:
    pass


_MISSING = _MissingType()


# ---------------------------------------------------------------------------------------------
# Helpers shared by the implementations
# ---------------------------------------------------------------------------------------------

def _probe(series, n=2):
    """A tiny Series of the same dtype as `series`, carrying a null when `series` may have one.

    Running the *original* pandas method over this tells us the exact dtype and NA convention pandas
    would produce, so the GPU result can be cast to match without a hand-written dtype table."""
    head = series.iloc[:n]
    try:
        if series.isna().iloc[:512].any() or getattr(series.dtype, "na_value", None) is not None:
            head = pd.concat([head, pd.Series([None], dtype=series.dtype, index=[0])])
    except Exception:
        pass
    head.name = series.name
    return head


def _positions(idx):
    """int64 numpy positions from a MetalArray of indices."""
    return idx.to_arrow().to_numpy(zero_copy_only=False)


def _finish(arrow_array, index, name, dtype, fill=None):
    """Arrow result -> a pandas Series with pandas' own dtype and NA convention."""
    if fill is not None:
        m = arrow_array if isinstance(arrow_array, MetalArray) else MetalArray.from_arrow(arrow_array)
        arrow_array = m.fill_null(fill).to_arrow()
    ser = _b._arrow_series(arrow_array, index, name)
    return _b.astype_like(ser, dtype)


def _mask_positions(mask_series):
    """(MetalArray mask with nulls as False, int64 numpy positions of the True rows)."""
    m = _b._metal(mask_series).fill_null(False)
    return m, _positions(m.indices_nonzero().cast("int64"))


# ---------------------------------------------------------------------------------------------
# Series: scalar reductions
# ---------------------------------------------------------------------------------------------

def _impl_reduce(how, families=("numeric",)):
    def impl(orig, self, *a, **k):
        if a or k.get("skipna", True) is not True or k.get("min_count", 0) or k.get("level") is not None:
            return _UNSUPPORTED
        if not isinstance(self, pd.Series) or not _eligible(self, families, ROW_FACTOR.get(how, 1)):
            return _UNSUPPORTED
        v = _b.reduce(self, how)
        if v is None:
            return orig(pd.Series([None], dtype=self.dtype))
        return _like_probe(orig(_probe(self)), v)
    return impl


def _like_probe(probe_value, v):
    """Return `v` typed the way pandas types its own answer (np.int64 rather than int, ...)."""
    if isinstance(probe_value, np.generic):
        try:
            return type(probe_value)(v)
        except Exception:
            return v
    return v


def _impl_nunique(orig, self, dropna=True):
    if not isinstance(self, pd.Series) or not _eligible(self, ("numeric", "bool", "string")):
        return _UNSUPPORTED
    return _like_probe(orig(_probe(self), dropna), _b.nunique(self, dropna))


def _impl_count(orig, self, *a, **k):
    if a or k or not isinstance(self, pd.Series):
        return _UNSUPPORTED
    if not _eligible(self, ("numeric", "bool", "string"), ROW_FACTOR["count"]):
        return _UNSUPPORTED
    return _like_probe(orig(_probe(self)), int(len(self) - _b._metal(self).null_count))


def _impl_value_counts(orig, self, normalize=False, sort=True, ascending=False, bins=None, dropna=True):
    if normalize or bins is not None:
        return _UNSUPPORTED
    if not isinstance(self, pd.Series) or not _eligible(self, ("numeric", "bool", "string")):
        return _UNSUPPORTED
    keys, counts = _b.value_counts(self, sort=sort, ascending=ascending, dropna=dropna)
    probe = orig(_probe(self), normalize, sort, ascending, bins, dropna)
    index = pd.Index(pd.arrays.ArrowExtensionArray(keys), name=probe.index.name)
    try:
        index = index.astype(probe.index.dtype)
    except Exception:
        pass
    return _finish(counts, index, probe.name, probe.dtype)


# ---------------------------------------------------------------------------------------------
# Series: ordering
# ---------------------------------------------------------------------------------------------

def _impl_sort_values(orig, self, *a, **k):
    if a:
        return _UNSUPPORTED
    if (k.get("inplace") or k.get("key") is not None or k.get("na_position", "last") != "last"
            or k.get("axis", 0) not in (0, "index")):
        return _UNSUPPORTED
    ascending = k.get("ascending", True)
    if not isinstance(ascending, bool):
        return _UNSUPPORTED
    if not isinstance(self, pd.Series) or not _eligible(self, ("numeric", "bool", "string")):
        return _UNSUPPORTED
    idx = _b.sort_indices(self, ascending)
    pos = _positions(idx)
    values = _b._take(self, idx).to_arrow()
    index = pd.RangeIndex(len(pos)) if k.get("ignore_index") else self.index.take(pos)
    return _finish(values, index, self.name, self.dtype)


def _impl_top_k(largest):
    def impl(orig, self, n=5, keep="first"):
        if keep != "first" or not isinstance(self, pd.Series) or not _eligible(self, ("numeric",)):
            return _UNSUPPORTED
        n = min(int(n), len(self))
        if n < 0:
            return _UNSUPPORTED
        idx = _b.top_k_indices(self, n, largest)
        pos = _positions(idx)
        values = _b._take(self, idx).to_arrow()
        return _finish(values, self.index.take(pos), self.name, self.dtype)
    return impl


# ---------------------------------------------------------------------------------------------
# Series: element-wise
# ---------------------------------------------------------------------------------------------

def _impl_isin(orig, self, values):
    if not isinstance(self, pd.Series) or not _eligible(self, ("numeric", "string")):
        return _UNSUPPORTED
    if _numpy_blocked("isin", self):
        return _UNSUPPORTED
    try:
        vals = list(values)
    except TypeError:
        return _UNSUPPORTED
    for v in vals:
        if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
            return _UNSUPPORTED       # pandas matches NaN against NaN; Arrow's is_in does not
    arr = _b.isin(self, vals)
    probe = orig(_probe(self), vals)
    return _finish(arr, self.index, self.name, probe.dtype)


def _impl_unary(name):
    def impl(orig, self, *a, **k):
        if a or k or not isinstance(self, pd.Series):
            return _UNSUPPORTED
        if not _eligible(self, ("numeric",), ROW_FACTOR.get(name, 1)):
            return _UNSUPPORTED
        arr = getattr(_b._metal(self), name)().to_arrow()
        return _finish(arr, self.index, self.name, self.dtype)
    return impl


def _impl_round(orig, self, decimals=0, *a, **k):
    if a or k or not isinstance(self, pd.Series) or not _eligible(self, ("numeric",)):
        return _UNSUPPORTED
    if _numpy_blocked("round", self):
        return _UNSUPPORTED
    if dtype_family(self.dtype) == "numeric" and _b._metal(self).format not in ("f", "g"):
        return _UNSUPPORTED                     # rounding an integer column is the identity
    arr = _b._metal(self).round(int(decimals), mode="half_to_even").to_arrow()
    return _finish(arr, self.index, self.name, self.dtype)


_COMPARISONS = {"__eq__": "eq", "__ne__": "ne", "__lt__": "lt",
                "__le__": "le", "__gt__": "gt", "__ge__": "ge"}


def _impl_compare(op):
    def impl(orig, self, other):
        if not isinstance(self, pd.Series) or not _big(self, ROW_FACTOR.get(op, 1)):
            return _UNSUPPORTED
        fam = dtype_family(self.dtype)
        if fam == "string":
            if op not in ("eq", "ne") or not isinstance(other, str):
                return _UNSUPPORTED
        elif fam == "numeric":
            if isinstance(other, bool) or not isinstance(other, (int, float, np.integer, np.floating)):
                return _UNSUPPORTED
            if isinstance(other, float) and np.isnan(other):
                return _UNSUPPORTED
        else:
            return _UNSUPPORTED
        arr = _b.compare(self, op, other)
        probe = orig(_probe(self), other)
        # pandas' numpy path has no null: `NaN != x` is True and every other comparison is False.
        fill = None
        if isinstance(probe.dtype, np.dtype) and probe.dtype.kind == "b":
            fill = (op == "ne")
        return _finish(arr, self.index, self.name, probe.dtype, fill=fill)
    return impl


def _bool_mask(key, length, index):
    """The boolean mask behind `obj[key]`, or None when this is not a full-length boolean mask."""
    if isinstance(key, pd.Series):
        if dtype_family(key.dtype) != "bool" or len(key) != length or not key.index.equals(index):
            return None
        return key
    if isinstance(key, np.ndarray) and key.dtype == bool and len(key) == length:
        return pd.Series(key, index=index)
    return None


def _impl_series_getitem(orig, self, key):
    if type(key) is not pd.Series and not isinstance(key, np.ndarray):
        return _UNSUPPORTED
    if not _big(self) or dtype_family(self.dtype) is None:
        return _UNSUPPORTED
    mask = _bool_mask(key, len(self), self.index)
    if mask is None:
        return _UNSUPPORTED
    m, pos = _mask_positions(mask)
    values = _b._metal(self).filter(m).to_arrow()
    return _finish(values, self.index.take(pos), self.name, self.dtype)


# ---------------------------------------------------------------------------------------------
# Series.str
# ---------------------------------------------------------------------------------------------

def _str_series(sm):
    return sm._orig


def _impl_str_match(kind):
    def impl(orig, self, pat, *a, **k):
        s = _str_series(self)
        if not isinstance(s, pd.Series) or not _eligible(s, ("string",)):
            return _UNSUPPORTED
        if not isinstance(pat, str):
            return _UNSUPPORTED
        if a or "na" in k:            # an explicit `na=` changes what a null value maps to
            return _UNSUPPORTED
        if kind == "contains":
            if not k.get("case", True) or k.get("flags", 0):
                return _UNSUPPORTED
            if k.get("regex", True) and not _b.is_plain_pattern(pat):
                return _UNSUPPORTED
        arr = _b.str_op(s, kind, pat)
        probe = orig(_probe(s).str, pat)
        fill = None
        if isinstance(probe.dtype, np.dtype) and probe.dtype.kind == "b":
            fill = False
        return _finish(arr, s.index, s.name, probe.dtype, fill=fill)
    return impl


def _impl_str_case(kind):
    def impl(orig, self, *a, **k):
        s = _str_series(self)
        if a or k or not isinstance(s, pd.Series) or not _eligible(s, ("string",)):
            return _UNSUPPORTED
        if not _b.is_ascii(s):
            return _UNSUPPORTED       # the GPU case kernels only match Python's for ASCII
        arr = _b.str_op(s, kind)
        probe = orig(_probe(s).str)
        return _finish(arr, s.index, s.name, probe.dtype)
    return impl


def _impl_str_len(orig, self, *a, **k):
    s = _str_series(self)
    if a or k or not isinstance(s, pd.Series) or not _eligible(s, ("string",)):
        return _UNSUPPORTED
    arr = _b.str_op(s, "len")
    probe = orig(_probe(s).str)
    return _finish(arr, s.index, s.name, probe.dtype)


# ---------------------------------------------------------------------------------------------
# DataFrame
# ---------------------------------------------------------------------------------------------

def _frame_take(df, pos, index):
    data = {}
    idx = _b._metal(pa.array(np.asarray(pos, dtype=np.int64)))
    for c in df.columns:
        col = df[c]
        if dtype_family(col.dtype) is None:
            raise TypeError(f"column {c!r} of dtype {col.dtype} is not on the GPU path")
        data[c] = _b.astype_like(_b._arrow_series(_b._take(col, idx).to_arrow()), col.dtype).array
    return pd.DataFrame(data, index=index, copy=False)


def _impl_frame_sort_values(orig, self, by=None, *a, **k):
    if a or by is None:
        return _UNSUPPORTED
    if (k.get("inplace") or k.get("key") is not None or k.get("na_position", "last") != "last"
            or k.get("axis", 0) not in (0, "index")):
        return _UNSUPPORTED
    cols = [by] if isinstance(by, str) else list(by)
    if not all(isinstance(c, str) and c in self.columns for c in cols):
        return _UNSUPPORTED
    ascending = k.get("ascending", True)
    if not isinstance(ascending, (bool, list, tuple)):
        return _UNSUPPORTED
    if not _big(self) or any(dtype_family(self[c].dtype) is None for c in self.columns):
        return _UNSUPPORTED
    idx = _b.sort_indices([self[c] for c in cols], ascending)
    pos = _positions(idx)
    index = pd.RangeIndex(len(pos)) if k.get("ignore_index") else self.index.take(pos)
    return _frame_take(self, pos, index)


def _impl_frame_getitem(orig, self, key):
    if type(key) is str or not isinstance(key, (pd.Series, np.ndarray)):
        return _UNSUPPORTED
    if not _big(self):
        return _UNSUPPORTED
    mask = _bool_mask(key, len(self), self.index)
    if mask is None or any(dtype_family(self[c].dtype) is None for c in self.columns):
        return _UNSUPPORTED
    _, pos = _mask_positions(mask)
    return _frame_take(self, pos, self.index.take(pos))


def _impl_merge(orig, self, right, how="inner", on=None, *a, **k):
    if a or how != "inner" or on is None or not isinstance(on, str):
        return _UNSUPPORTED
    for bad in ("left_on", "right_on", "left_index", "right_index", "validate", "indicator"):
        if k.get(bad):
            return _UNSUPPORTED
    if k.get("sort"):
        return _UNSUPPORTED
    suffixes = k.get("suffixes", ("_x", "_y"))
    if not isinstance(right, pd.DataFrame) or on not in self.columns or on not in right.columns:
        return _UNSUPPORTED
    if not (_big(self) or _big(right)):
        return _UNSUPPORTED
    if dtype_family(self[on].dtype) not in ("numeric", "string"):
        return _UNSUPPORTED
    if any(dtype_family(self[c].dtype) is None for c in self.columns):
        return _UNSUPPORTED
    if any(dtype_family(right[c].dtype) is None for c in right.columns):
        return _UNSUPPORTED
    lrows, rrows = _b.inner_join_indices(self[on], right[on])
    lpos = lrows.to_numpy(zero_copy_only=False)
    rpos = rrows.to_numpy(zero_copy_only=False)
    lm = _b._metal(pa.array(lpos))
    rm = _b._metal(pa.array(rpos))
    overlap = (set(self.columns) & set(right.columns)) - {on}
    data = {}
    for c in self.columns:
        name = f"{c}{suffixes[0]}" if c in overlap else c
        data[name] = _b.astype_like(_b._arrow_series(_b._take(self[c], lm).to_arrow()),
                                    self[c].dtype).array
    for c in right.columns:
        if c == on:
            continue
        name = f"{c}{suffixes[1]}" if c in overlap else c
        data[name] = _b.astype_like(_b._arrow_series(_b._take(right[c], rm).to_arrow()),
                                    right[c].dtype).array
    return pd.DataFrame(data, copy=False)


# ---------------------------------------------------------------------------------------------
# groupby
# ---------------------------------------------------------------------------------------------

def _grouper_of(gb):
    return getattr(gb, "_grouper", None) or getattr(gb, "grouper", None)


def _group_keys(gb):
    """(names, key columns as pandas Series) when every key is a plain column, else None."""
    gr = _grouper_of(gb)
    if gr is None or getattr(gb, "level", None) is not None:
        return None
    names, cols = list(gr.names), []
    for g in gr.groupings:
        vec = g.grouping_vector
        if isinstance(vec, (pd.Index, pd.Categorical)) or not hasattr(vec, "dtype"):
            return None
        s = vec if isinstance(vec, pd.Series) else pd.Series(vec, copy=False)
        if dtype_family(s.dtype) is None:
            return None
        cols.append(s)
    if any(n is None for n in names) or not cols:
        return None
    return names, cols


def _impl_group_agg(how):
    def impl(orig, self, *a, **k):
        if a or k.get("numeric_only") or k.get("min_count", 0) or k.get("engine") is not None:
            return _UNSUPPORTED
        if getattr(self, "group_keys", True) is False:
            return _UNSUPPORTED
        keys = _group_keys(self)
        if keys is None:
            return _UNSUPPORTED
        names, key_cols = keys
        obj = self._obj_with_exclusions
        as_series = isinstance(obj, pd.Series)
        value_cols = [obj] if as_series else [obj[c] for c in obj.columns]
        if not value_cols or not _big(value_cols[0]):
            return _UNSUPPORTED
        if any(dtype_family(c.dtype) != "numeric" for c in value_cols):
            return _UNSUPPORTED
        if len(key_cols[0]) != len(value_cols[0]):
            return _UNSUPPORTED
        key_arrays, agg_arrays = _b.groupby_aggregate(
            key_cols, value_cols, how, dropna=getattr(self, "dropna", True),
            sort=getattr(self, "sort", True))

        # Learn pandas' own key and value dtypes from a two-row groupby of the same columns.
        probe_frame = pd.DataFrame(
            {n: c.iloc[:2].reset_index(drop=True) for n, c in zip(names, key_cols)})
        for c in value_cols:
            probe_frame[c.name] = c.iloc[:2].reset_index(drop=True)
        probe = getattr(probe_frame.groupby(names, dropna=getattr(self, "dropna", True)), how)()

        levels = []
        for i, k in enumerate(key_arrays):
            level = pd.Index(pd.arrays.ArrowExtensionArray(k), name=names[i])
            try:
                level = level.astype(probe.index.get_level_values(i).dtype)
            except Exception:
                pass
            levels.append(level)
        index = levels[0] if len(levels) == 1 else pd.MultiIndex.from_arrays(levels, names=names)

        cols = {}
        for c, arr in zip(value_cols, agg_arrays):
            want = probe[c.name].dtype if c.name in probe.columns else c.dtype
            cols[c.name] = _b.astype_like(_b._arrow_series(arr), want).array
        if as_series:
            name = obj.name
            return pd.Series(cols[name], index=index, name=name, copy=False)
        out = pd.DataFrame(cols, index=index, copy=False)
        if getattr(self, "as_index", True):
            return out
        return out.reset_index()
    return impl


# ---------------------------------------------------------------------------------------------
# install / uninstall
# ---------------------------------------------------------------------------------------------

#: Everything the accel layer touches. Read it, print it, put it in a doc — it is the contract.
REGISTRY = [
    ("Series.sum", "sum"), ("Series.min", "min"), ("Series.max", "max"), ("Series.mean", "mean"),
    ("Series.nunique", "nunique"), ("Series.count", "count"), ("Series.value_counts", "value_counts"),
    ("Series.sort_values", "sort_values"), ("Series.nlargest", "nlargest"),
    ("Series.nsmallest", "nsmallest"), ("Series.isin", "isin"), ("Series.abs", "abs"),
    ("Series.round", "round"), ("Series.__getitem__", "getitem"),
    ("Series.__eq__", "eq"), ("Series.__ne__", "ne"), ("Series.__lt__", "lt"),
    ("Series.__le__", "le"), ("Series.__gt__", "gt"), ("Series.__ge__", "ge"),
    ("Series.str.contains", "str.contains"), ("Series.str.startswith", "str.startswith"),
    ("Series.str.endswith", "str.endswith"), ("Series.str.upper", "str.upper"),
    ("Series.str.lower", "str.lower"), ("Series.str.len", "str.len"),
    ("DataFrame.sort_values", "frame.sort_values"), ("DataFrame.merge", "frame.merge"),
    ("DataFrame.__getitem__", "frame.getitem"),
    ("DataFrameGroupBy.sum", "groupby.sum"), ("DataFrameGroupBy.mean", "groupby.mean"),
    ("DataFrameGroupBy.min", "groupby.min"), ("DataFrameGroupBy.max", "groupby.max"),
    ("DataFrameGroupBy.count", "groupby.count"),
    ("SeriesGroupBy.sum", "groupby.sum"), ("SeriesGroupBy.mean", "groupby.mean"),
    ("SeriesGroupBy.min", "groupby.min"), ("SeriesGroupBy.max", "groupby.max"),
    ("SeriesGroupBy.count", "groupby.count"),
]


def installed():
    return bool(_originals)


def install(threshold=None):
    """Patch pandas. Idempotent; returns the `Stats` object."""
    if threshold is not None:
        set_threshold(threshold)
    if installed():
        return _stats
    from pandas.core.strings.accessor import StringMethods
    from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy

    S = pd.Series
    _patch(S, "sum", "sum", _impl_reduce("sum"))
    _patch(S, "min", "min", _impl_reduce("min"))
    _patch(S, "max", "max", _impl_reduce("max"))
    _patch(S, "mean", "mean", _impl_reduce("mean"))
    _patch(S, "nunique", "nunique", _impl_nunique)
    _patch(S, "count", "count", _impl_count)
    _patch(S, "value_counts", "value_counts", _impl_value_counts)
    _patch(S, "sort_values", "sort_values", _impl_sort_values)
    _patch(S, "nlargest", "nlargest", _impl_top_k(True))
    _patch(S, "nsmallest", "nsmallest", _impl_top_k(False))
    _patch(S, "isin", "isin", _impl_isin)
    _patch(S, "abs", "abs", _impl_unary("abs"))
    _patch(S, "round", "round", _impl_round)
    _patch(S, "__getitem__", "getitem", _impl_series_getitem)
    for dunder, op in _COMPARISONS.items():
        _patch(S, dunder, op, _impl_compare(op))

    _patch(StringMethods, "contains", "str.contains", _impl_str_match("contains"))
    _patch(StringMethods, "startswith", "str.startswith", _impl_str_match("startswith"))
    _patch(StringMethods, "endswith", "str.endswith", _impl_str_match("endswith"))
    _patch(StringMethods, "upper", "str.upper", _impl_str_case("upper"))
    _patch(StringMethods, "lower", "str.lower", _impl_str_case("lower"))
    _patch(StringMethods, "len", "str.len", _impl_str_len)

    _patch(pd.DataFrame, "sort_values", "frame.sort_values", _impl_frame_sort_values)
    _patch(pd.DataFrame, "merge", "frame.merge", _impl_merge)
    _patch(pd.DataFrame, "__getitem__", "frame.getitem", _impl_frame_getitem)

    for cls in (DataFrameGroupBy, SeriesGroupBy):
        for how in ("sum", "mean", "min", "max", "count"):
            _patch(cls, how, f"groupby.{how}", _impl_group_agg(how))
    return _stats


def uninstall():
    """Put every original pandas method back."""
    while _originals:
        owner, name, orig = _originals.pop()
        if isinstance(orig, _MissingType):
            try:
                delattr(owner, name)
            except AttributeError:
                pass
        else:
            setattr(owner, name, orig)
    return _stats


def _main(argv):
    """`python -m arrowmetal.pandas_accel script.py [args]`."""
    if not argv:
        print(__doc__)
        return 0
    install()
    script = argv[0]
    sys.argv = list(argv)
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)) or ".")
    code = compile(open(script).read(), script, "exec")
    globs = {"__name__": "__main__", "__file__": script, "__builtins__": __builtins__}
    try:
        exec(code, globs)
    finally:
        if os.environ.get("ARROWMETAL_PANDAS_ACCEL_REPORT"):
            print(_stats, file=sys.stderr)
    return 0


if os.environ.get("ARROWMETAL_PANDAS_ACCEL") in ("1", "true", "True", "yes"):
    install()

if __name__ == "__main__":
    # `python -m` runs this file as "__main__", which would give the script a second, separate copy
    # of the module (and of its stats). Hand the work to the canonical `arrowmetal.pandas_accel`.
    import importlib

    sys.exit(importlib.import_module("arrowmetal.pandas_accel")._main(sys.argv[1:]))
