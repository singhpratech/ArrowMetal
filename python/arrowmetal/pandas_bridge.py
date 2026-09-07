"""pandas <-> ArrowMetal: zero-copy conversion and an explicit `.am` accessor.

Two things live here.

1. **Conversion.** `am.from_pandas(obj)` lifts a pandas Series or DataFrame onto the GPU and
   `am.to_pandas(obj)` brings the result back. When the pandas column is already Arrow-backed
   (`pd.ArrowDtype`, `pd.array(..., dtype="int64[pyarrow]")`, or pandas 3's default `str` dtype,
   which is an `ArrowStringArray`) the Arrow buffers are handed straight to Metal and nothing is
   copied: `Series.array._pa_array` is a `pyarrow.ChunkedArray`, a single chunk of which is taken
   as-is. A column that pandas has split into several chunks is combined once (one copy, reported).
   A numpy-backed column has no Arrow buffers, so it is converted once (see `zero_copy_report`).

2. **The `.am` accessor.** `s.am.sum()`, `s.am.top_k(100)`, `s.am.contains("x")`,
   `df.am.groupby("k").sum("v")`, `df.am.sort_values("c")`, `df.am.merge(other, on="k")`,
   `df.am.query(am.filter(...).sum(...))`. Every result comes back as a pandas object with an
   Arrow-backed dtype, so it goes back to the GPU zero-copy too.

Nulls. Arrow distinguishes null from NaN; numpy-backed pandas does not. Converting a numpy float
column therefore maps NaN to null (which is what `skipna=True`, pandas' default, means anyway).
An Arrow-backed column is passed through untouched, so a NaN stays a NaN and a null stays a null.

Importing this module registers the accessors; it is safe to import twice.
"""

import re as _re

import numpy as np
import pandas as pd
import pyarrow as pa

from . import (ArrowMetalError, Expr as _Expr, MetalArray, Query as _Query,
               group_by as _am_group_by, lexsort_indices as _am_lexsort, query as _am_query,
               _as_pa_array)

__all__ = ["from_pandas", "to_pandas", "to_arrow", "zero_copy_report", "Conversion",
           "SeriesAccessor", "DataFrameAccessor", "register", "GPUGroupBy"]


# ---------------------------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------------------------

class Conversion:
    """The Arrow array behind a pandas column, and how it got there.

    `zero_copy` is decided by comparing buffer addresses, not by guessing: it is true only when the
    Arrow array handed to Metal points at the very bytes pandas was already holding.
    """

    __slots__ = ("array", "zero_copy", "reason", "source_dtype")

    def __init__(self, array, zero_copy, reason, source_dtype):
        self.array, self.zero_copy, self.reason, self.source_dtype = array, zero_copy, reason, source_dtype

    def __repr__(self):
        return (f"Conversion({self.array.type}, zero_copy={self.zero_copy}, "
                f"reason={self.reason!r}, from={self.source_dtype})")


_NUMPY_EA = getattr(pd.arrays, "NumpyExtensionArray", getattr(pd.arrays, "PandasArray", ()))


def _addresses(obj):
    """The non-null buffer addresses of a pyarrow array, or of a numpy array."""
    if isinstance(obj, np.ndarray):
        return (obj.__array_interface__["data"][0],)
    if isinstance(obj, pa.ChunkedArray):
        return tuple(a for c in obj.chunks for a in _addresses(c))
    try:
        return tuple(b.address for b in obj.buffers() if b is not None)
    except Exception:
        return ()


def _values_address(obj):
    """The address of the values buffer (the last one, after validity/offsets), or None."""
    a = _addresses(obj)
    return a[-1] if a else None


def _widen_dictionary(arr):
    """ArrowMetal reads dictionary indices as int32 or int64; a pandas Categorical often gets int8
    or int16 from pyarrow, so widen those (the array was being copied anyway)."""
    if pa.types.is_dictionary(arr.type) and arr.type.index_type.bit_width < 32:
        return pa.DictionaryArray.from_arrays(arr.indices.cast(pa.int32()), arr.dictionary)
    return arr


def _pandas_values(obj):
    """(ExtensionArray or ndarray, dtype) for a Series, Index, ExtensionArray or ndarray."""
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.array if hasattr(obj, "array") else np.asarray(obj), obj.dtype
    if isinstance(obj, np.ndarray):
        return obj, obj.dtype
    return obj, getattr(obj, "dtype", None)


def to_arrow(obj):
    """A pandas column as a `pyarrow.Array`, with a `Conversion` describing the cost.

        conv = am.pandas_bridge.to_arrow(df["price"])
        conv.zero_copy, conv.reason
    """
    if isinstance(obj, MetalArray):
        return Conversion(obj, True, "already on the GPU", None)
    if isinstance(obj, (pa.Array, pa.ChunkedArray)):
        arr = _as_pa_array(obj)
        return Conversion(arr, True, "already Arrow", None)

    values, dtype = _pandas_values(obj)

    # 1. Arrow-backed pandas: ArrowExtensionArray / ArrowStringArray keep a pyarrow.ChunkedArray.
    chunked = getattr(values, "_pa_array", None)
    if chunked is not None:
        before = _values_address(chunked)
        arr = _as_pa_array(chunked)
        after = _values_address(arr)
        n = getattr(chunked, "num_chunks", 1)
        if n <= 1 or (before is not None and before == after):
            return Conversion(arr, True, "Arrow-backed pandas column, buffers shared", dtype)
        return Conversion(arr, False, f"Arrow-backed but split into {n} chunks; combined once", dtype)

    # 2. pandas nullable masked extension arrays (Int64, Float64, boolean, string[python]) and
    #    Categorical: pyarrow reads them through __arrow_array__. One copy.
    #    NumpyExtensionArray is an ExtensionArray too but holds a bare ndarray, so it falls through
    #    to the numpy path below (where NaN -> null happens).
    if isinstance(values, pd.api.extensions.ExtensionArray) and not isinstance(values, _NUMPY_EA):
        arr = _widen_dictionary(_as_pa_array(pa.array(values)))
        kind = "categorical" if isinstance(values, pd.Categorical) else "pandas masked extension array"
        return Conversion(arr, False, f"{kind} -> Arrow, one copy", dtype)

    # 3. numpy-backed. Integers and booleans without nulls can still come across without a copy;
    #    floats and datetimes need NaN/NaT -> null, which builds a validity bitmap.
    np_arr = np.asarray(values)
    from_pandas = np_arr.dtype.kind in "fMmO"
    arr = _as_pa_array(pa.array(np_arr, from_pandas=from_pandas))
    shared = _values_address(np_arr) in _addresses(arr)
    if shared:
        return Conversion(arr, True, "numpy buffer adopted by Arrow", dtype)
    reason = ("numpy float/datetime -> Arrow, one copy (NaN/NaT become null)" if from_pandas
              else "numpy -> Arrow, one copy")
    return Conversion(arr, False, reason, dtype)


def _metal(obj):
    """Anything -> MetalArray, going through `to_arrow` for pandas input."""
    if isinstance(obj, MetalArray):
        return obj
    if isinstance(obj, (pa.Array, pa.ChunkedArray)):
        return MetalArray.from_arrow(_as_pa_array(obj))
    return MetalArray.from_arrow(to_arrow(obj).array)


def from_pandas(obj):
    """A pandas Series -> `MetalArray`; a pandas DataFrame -> `{column name: MetalArray}`.

    Zero-copy for Arrow-backed columns (see the module docstring); one conversion otherwise.
    Use `zero_copy_report` to see which happened.
    """
    if isinstance(obj, pd.DataFrame):
        return {c: _metal(obj[c]) for c in obj.columns}
    return _metal(obj)


def zero_copy_report(obj):
    """`{column: {"zero_copy": bool, "reason": str, "dtype": str, "arrow_type": str}}` for a
    DataFrame, or the single dict for a Series. Evidence, not a guess: the flag comes from comparing
    the pandas buffer address with the Arrow one."""
    def one(col):
        c = to_arrow(col)
        return {"zero_copy": c.zero_copy, "reason": c.reason,
                "dtype": str(c.source_dtype), "arrow_type": str(getattr(c.array, "type", None))}
    if isinstance(obj, pd.DataFrame):
        return {c: one(obj[c]) for c in obj.columns}
    return one(obj)


# ---------------------------------------------------------------------------------------------
# Back to pandas
# ---------------------------------------------------------------------------------------------

def _arrow(obj):
    """MetalArray / pyarrow -> pyarrow.Array."""
    if isinstance(obj, MetalArray):
        return obj.to_arrow()
    return _as_pa_array(obj)


def _arrow_series(arr, index=None, name=None):
    """An Arrow-backed pandas Series over `arr` — no copy of the values."""
    return pd.Series(pd.arrays.ArrowExtensionArray(_arrow(arr)), index=index, name=name, copy=False)


def to_pandas(obj, index=None, name=None, dtype=None):
    """A `MetalArray`, pyarrow array or dict of them, as a pandas Series or DataFrame with
    Arrow-backed dtypes. `dtype` casts the result (see `astype_like` for the accel-mode rules)."""
    if isinstance(obj, dict):
        return pd.DataFrame({k: _arrow_series(v, index) for k, v in obj.items()}, index=index)
    s = _arrow_series(obj, index, name)
    return s if dtype is None else astype_like(s, dtype)


def astype_like(series, dtype):
    """Cast an Arrow-backed result to the dtype pandas itself would have produced.

    Arrow keeps nulls where numpy pandas keeps NaN or False, so the cast is done in the direction
    pandas uses: a null becomes `False` for a plain `bool` result, `NaN` for a float one, and the
    dtype's own NA everywhere else."""
    if dtype is None or series.dtype == dtype:
        return series
    try:
        return series.astype(dtype)
    except Exception:
        pass
    try:
        np_dtype = np.dtype(dtype)
    except Exception:
        np_dtype = None
    if isinstance(np_dtype, np.dtype) and np_dtype.kind == "b":
        return series.fillna(False).astype(bool)
    if isinstance(np_dtype, np.dtype) and np_dtype.kind in "fiu":
        out = series.astype("float64[pyarrow]").to_numpy(dtype="float64", na_value=np.nan)
        return pd.Series(out, index=series.index, name=series.name)
    if isinstance(np_dtype, np.dtype) and np_dtype.kind == "O":
        return pd.Series(series.to_numpy(dtype=object, na_value=np.nan),
                         index=series.index, name=series.name)
    return series.astype(dtype)


# ---------------------------------------------------------------------------------------------
# The GPU kernels the accessor and the accel layer share.
# Every one of these takes pandas / Arrow input and returns pyarrow arrays.
# ---------------------------------------------------------------------------------------------

_REDUCTIONS = ("sum", "min", "max", "mean", "product", "median", "any", "all")


def reduce(col, how, skipna=True):
    """One scalar aggregate on the GPU. `sum` of an empty or all-null column is 0, as in pandas."""
    m = _metal(col)
    v = getattr(m, how)()
    if v is None and how == "sum":
        return 0
    return v


def nunique(col, dropna=True):
    """pandas `nunique`: distinct non-null values."""
    m = _metal(col)
    n = m.count_distinct()
    if not dropna and m.null_count > 0:
        n += 1
    return int(n)


def _row_numbers(n):
    return pa.array(np.arange(n, dtype=np.int64))


def _take(arr, indices):
    m = arr if isinstance(arr, MetalArray) else _metal(arr)
    idx = indices if isinstance(indices, MetalArray) else _metal(indices)
    if idx.format not in ("i", "l"):
        idx = idx.cast("int64")
    return m.take(idx)


_STRING_FORMATS = ("u", "U")


def ordinal(col):
    """A sortable integer column with the same order as `col`.

    The GPU radix sort takes fixed-width keys, so a utf8 column is dictionary-encoded on the GPU,
    its (much smaller) dictionary is ranked, and the codes are mapped through that ranking — the
    row-level work stays on the GPU and only the distinct values are ordered on the host. Any other
    column is already sortable and comes back unchanged. Nulls stay null, so they still sort last.
    """
    m = _metal(col)
    if m.format not in _STRING_FORMATS:
        return m
    codes, uniques = m.dictionary_encode()
    import pyarrow.compute as pc
    order = pc.sort_indices(uniques.to_arrow()).cast(pa.int32())
    rank = MetalArray.from_arrow(order).inverse_permutation()
    return rank.take(codes)


def sort_indices(cols, ascending=True):
    """Stable GPU sort indices over one or more columns; nulls last, as pandas' `na_position`
    default is. Returned as int64."""
    if not isinstance(cols, (list, tuple)):
        cols = [cols]
    flags = ascending if isinstance(ascending, (list, tuple)) else [ascending] * len(cols)
    desc = [not bool(a) for a in flags]
    metal = [ordinal(c) for c in cols]
    if len(metal) == 1:
        idx = metal[0].argsort(descending=desc[0])
    else:
        idx = _am_lexsort(metal, desc)
    return idx.cast("int64")


def top_k_indices(col, k, largest=True):
    """Indices of the k largest (or smallest) values, in pandas' `nlargest` order."""
    return _metal(col).top_k(int(k), largest=largest).cast("int64")


def group_ids(keys):
    """A `GroupByKeys` over the given key columns."""
    return _am_group_by([_metal(k) for k in keys])


def groupby_aggregate(keys, values, how, dropna=True, sort=True):
    """pandas `groupby(...).agg` on the GPU.

    Returns `(key_arrays, value_arrays)`, one row per group. Follows pandas: null keys are dropped
    unless `dropna=False`, groups come back sorted by key unless `sort=False` (then in first-seen
    order), and a group whose values are all null sums to 0.
    """
    mkeys = [_metal(k) for k in keys]
    n = len(mkeys[0])
    gb = _am_group_by(mkeys)
    mvals = [_metal(v) for v in values]
    if how == "count_all":                      # pandas `size`: rows per group, nulls included
        aggs = [gb.count_all() for _ in mvals]
    else:
        aggs = [getattr(gb, how)(v) for v in mvals]
    if how == "sum":
        aggs = [a.fill_null(0) for a in aggs]
    key_cols = [MetalArray.from_arrow(k) for k in gb.keys()]

    if sort:
        ranked = [ordinal(k) for k in key_cols]
        order = (ranked[0].argsort() if len(ranked) == 1
                 else _am_lexsort(ranked, [False] * len(ranked)))
    else:
        order = gb.min(MetalArray.from_arrow(_row_numbers(n))).argsort()
    key_cols = [_take(k, order) for k in key_cols]
    aggs = [_take(a, order) for a in aggs]

    if dropna and any(k.null_count for k in key_cols):
        keep = None
        for k in key_cols:
            v = ordinal(k).is_valid()          # is_valid needs a primitive column, so rank strings
            keep = v if keep is None else (keep & v)
        key_cols = [k.filter(keep) for k in key_cols]
        aggs = [a.filter(keep) for a in aggs]
    return [k.to_arrow() for k in key_cols], [a.to_arrow() for a in aggs]


def value_counts(col, sort=True, ascending=False, dropna=True):
    """pandas `value_counts` on the GPU, including pandas' tie order: counts descending, ties in
    first-seen order."""
    m = _metal(col)
    n = len(m)
    gb = _am_group_by([m])
    counts = gb.count_all()
    keys = MetalArray.from_arrow(gb.keys()[0])
    first = gb.min(MetalArray.from_arrow(_row_numbers(n)))
    order = first.argsort()                       # first-seen order
    keys, counts = _take(keys, order), _take(counts, order)
    if dropna and keys.null_count:
        keep = ordinal(keys).is_valid()
        keys, counts = keys.filter(keep), counts.filter(keep)
    if sort:
        # a stable argsort of the counts keeps the first-seen order inside a tie group
        order = counts.argsort(descending=not ascending)
        keys, counts = _take(keys, order), _take(counts, order)
    return keys.to_arrow(), counts.to_arrow()


def inner_join_indices(left_key, right_key):
    """`(left_rows, right_rows)` int64 index arrays for an inner join on one key column.

    Uses the GPU `index_in` hash probe, so it needs the right key to be unique and neither side to
    hold nulls — exactly the `validate="m:1"` case. Raises otherwise; the caller falls back."""
    l, r = _metal(left_key), _metal(right_key)
    if l.null_count or r.null_count:
        raise ArrowMetalError("GPU merge needs join keys without nulls")
    if len(_am_group_by([r])) != len(r):        # GPU hash build: distinct count of any key type
        raise ArrowMetalError("GPU merge needs unique keys on the right frame")
    pos = l.index_in(r)                    # int32 row of the right frame, null where absent
    mask = pos.is_valid()
    left_rows = mask.indices_nonzero().cast("int64")
    right_rows = pos.filter(mask).cast("int64")
    return left_rows.to_arrow(), right_rows.to_arrow()


_STR_MATCH = {"contains": "str_contains", "startswith": "starts_with", "endswith": "ends_with"}
_REGEX_META = set(".^$*+?{}[]\\|()")


def is_plain_pattern(pattern):
    """True when a `str.contains` pattern is a literal, so the GPU substring kernel is exact."""
    return isinstance(pattern, str) and not (_REGEX_META & set(pattern))


def is_ascii(col):
    """True when every value is pure ASCII — checked on the GPU by comparing byte length with code
    point count. `upper`/`lower` are only guaranteed to match Python's for ASCII."""
    m = _metal(col)
    if len(m) == 0:
        return True
    diff = (m.byte_length() == m.char_length())
    return bool(diff.fill_null(True).all())


def str_op(col, op, pattern=None):
    """One string kernel by pandas name."""
    m = _metal(col)
    if op in _STR_MATCH:
        return getattr(m, _STR_MATCH[op])(pattern).to_arrow()
    if op == "upper":
        return m.upper().to_arrow()
    if op == "lower":
        return m.lower().to_arrow()
    if op == "len":
        return m.char_length().to_arrow()
    raise ArrowMetalError(f"no GPU kernel for str.{op}")


#: `(col "name")` in a serialised query, with the grammar's \" and \\ escapes. The same scan
#: `polars_bridge._COL_REF` does: the wire form names every column the query touches, so one pass
#: over it answers exactly which columns have to be lifted -- no walking the expression tree.
_COL_REF = _re.compile(r'\(col\s+"((?:[^"\\]|\\.)*)"\s*\)')


def _query_columns(q, available):
    """The frame columns `q` reads, in frame order.

    Lifting a column costs a page map and can fail outright (an object or struct column is not
    something the expression compiler reads), so a query must not pay for columns it never names.
    A query that names nothing at all falls back to every column, which is the only thing that can
    give the kernel a row count.
    """
    text = q.sexpr() if isinstance(q, (_Query, _Expr)) else str(q)
    wanted = {n.replace('\\"', '"').replace("\\\\", "\\") for n in _COL_REF.findall(text)}
    if not wanted:
        return list(available)
    missing = wanted - set(available)
    if missing:
        raise ArrowMetalError(f"query names column(s) not in the frame: {sorted(missing)}")
    return [c for c in available if c in wanted]


_COMPARE = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}


def compare(col, op, other):
    """A comparison against a scalar or another column, as an Arrow boolean array."""
    m = _metal(col)
    sym = _COMPARE.get(op, op)
    if isinstance(other, (pd.Series, pa.Array, pa.ChunkedArray, MetalArray, np.ndarray)):
        return m.compare(sym, _metal(other)).to_arrow()
    if m.format in ("u", "U"):
        if sym == "==":
            return m.str_equals(other).to_arrow()
        if sym == "!=":
            return (~m.str_equals(other)).to_arrow()
        raise ArrowMetalError("only == and != run on the GPU for string columns")
    return m.compare(sym, other).to_arrow()


def isin(col, values):
    m = _metal(col)
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return m.is_in(pa.array(vals, type=m.type)).to_arrow()


# ---------------------------------------------------------------------------------------------
# The `.am` accessor
# ---------------------------------------------------------------------------------------------

class SeriesAccessor:
    """`s.am.<op>()` — explicit, always on the GPU, always Arrow-backed on the way out.

    Unlike the accel layer this never falls back: an unsupported dtype raises, so you always know
    where the work ran."""

    def __init__(self, obj):
        self._s = obj

    # -- conversion
    def to_metal(self):
        """The column as a `MetalArray` (zero-copy when the column is Arrow-backed)."""
        return _metal(self._s)

    def to_arrow(self):
        return to_arrow(self._s).array

    def zero_copy(self):
        """The conversion report for this column."""
        return zero_copy_report(self._s)

    def _wrap(self, arr, index=None, name="keep"):
        idx = self._s.index if index is None else index
        return _arrow_series(arr, idx, self._s.name if name == "keep" else name)

    # -- reductions
    def sum(self): return reduce(self._s, "sum")
    def min(self): return reduce(self._s, "min")
    def max(self): return reduce(self._s, "max")
    def mean(self): return reduce(self._s, "mean")
    def product(self): return reduce(self._s, "product")
    def median(self): return reduce(self._s, "median")
    def any(self): return reduce(self._s, "any")
    def all(self): return reduce(self._s, "all")
    def std(self, ddof=1): return _metal(self._s).stddev(ddof=ddof)
    def var(self, ddof=1): return _metal(self._s).variance(ddof=ddof)
    def count(self): return len(self._s) - _metal(self._s).null_count
    def nunique(self, dropna=True): return nunique(self._s, dropna)

    def value_counts(self, sort=True, ascending=False, dropna=True):
        keys, counts = value_counts(self._s, sort, ascending, dropna)
        idx = pd.Index(pd.arrays.ArrowExtensionArray(keys), name=self._s.name)
        return _arrow_series(counts, idx, "count")

    # -- ordering
    def sort_values(self, ascending=True):
        """Stable sort, nulls last. The index is taken along, as pandas does."""
        idx = sort_indices(self._s, ascending)
        arr = _take(self._s, idx)
        pos = idx.to_arrow().to_numpy(zero_copy_only=False)
        return self._wrap(arr, self._s.index[pos])

    def argsort(self, ascending=True):
        return self._wrap(sort_indices(self._s, ascending).to_arrow(), None)

    def top_k(self, k, largest=True):
        """The k largest (or smallest) values, with their index — pandas `nlargest`/`nsmallest`."""
        idx = top_k_indices(self._s, k, largest)
        pos = idx.to_arrow().to_numpy(zero_copy_only=False)
        return self._wrap(_take(self._s, idx), self._s.index[pos])

    def nlargest(self, n=5): return self.top_k(n, True)
    def nsmallest(self, n=5): return self.top_k(n, False)

    # -- element-wise
    def isin(self, values): return self._wrap(isin(self._s, values))
    def abs(self): return self._wrap(_metal(self._s).abs().to_arrow())
    def round(self, ndigits=0): return self._wrap(_metal(self._s).round(ndigits).to_arrow())
    def compare(self, op, other): return self._wrap(compare(self._s, op, other))
    def __gt__(self, o): return self.compare("gt", o)
    def __ge__(self, o): return self.compare("ge", o)
    def __lt__(self, o): return self.compare("lt", o)
    def __le__(self, o): return self.compare("le", o)
    def __eq__(self, o): return self.compare("eq", o)
    def __ne__(self, o): return self.compare("ne", o)
    __hash__ = object.__hash__

    def filter(self, mask):
        """Boolean-mask selection on the GPU, index taken along. A null in the mask drops the row,
        as `df[mask]` does in pandas."""
        m = _metal(mask).fill_null(False)
        arr = _metal(self._s).filter(m)
        pos = m.indices_nonzero().to_arrow().to_numpy(zero_copy_only=False)
        return self._wrap(arr, self._s.index[pos])

    # -- strings
    def contains(self, pattern): return self._wrap(str_op(self._s, "contains", pattern))
    def startswith(self, pattern): return self._wrap(str_op(self._s, "startswith", pattern))
    def endswith(self, pattern): return self._wrap(str_op(self._s, "endswith", pattern))
    def upper(self): return self._wrap(str_op(self._s, "upper"))
    def lower(self): return self._wrap(str_op(self._s, "lower"))
    def len(self): return self._wrap(str_op(self._s, "len"))


class GPUGroupBy:
    """What `df.am.groupby(keys)` returns."""

    def __init__(self, frame, keys, dropna=True, sort=True, as_index=True):
        self._df = frame
        self._keys = [keys] if isinstance(keys, str) else list(keys)
        self._dropna, self._sort, self._as_index = dropna, sort, as_index

    def _value_columns(self, columns):
        if columns is None:
            return [c for c in self._df.columns if c not in self._keys]
        return [columns] if isinstance(columns, str) else list(columns)

    def _agg(self, how, columns=None):
        vcols = self._value_columns(columns)
        keys, vals = groupby_aggregate([self._df[k] for k in self._keys],
                                       [self._df[c] for c in vcols], how,
                                       dropna=self._dropna, sort=self._sort)
        data = {c: pd.arrays.ArrowExtensionArray(a) for c, a in zip(vcols, vals)}
        if self._as_index:
            if len(keys) == 1:
                index = pd.Index(pd.arrays.ArrowExtensionArray(keys[0]), name=self._keys[0])
            else:
                index = pd.MultiIndex.from_arrays(
                    [pd.arrays.ArrowExtensionArray(k) for k in keys], names=self._keys)
            out = pd.DataFrame(data, index=index)
        else:
            cols = {k: pd.arrays.ArrowExtensionArray(a) for k, a in zip(self._keys, keys)}
            cols.update(data)
            out = pd.DataFrame(cols)
        if isinstance(columns, str):
            return out[columns]
        return out

    def sum(self, columns=None): return self._agg("sum", columns)
    def mean(self, columns=None): return self._agg("mean", columns)
    def min(self, columns=None): return self._agg("min", columns)
    def max(self, columns=None): return self._agg("max", columns)
    def count(self, columns=None): return self._agg("count", columns)

    def size(self):
        """Rows per group, counting nulls — pandas `groupby(...).size()`."""
        keys, counts = groupby_aggregate([self._df[k] for k in self._keys],
                                         [self._df[self._keys[0]]], "count_all",
                                         dropna=self._dropna, sort=self._sort)
        index = (pd.Index(pd.arrays.ArrowExtensionArray(keys[0]), name=self._keys[0])
                 if len(keys) == 1 else
                 pd.MultiIndex.from_arrays([pd.arrays.ArrowExtensionArray(k) for k in keys],
                                           names=self._keys))
        return _arrow_series(counts[0], index, None)


class DataFrameAccessor:
    """`df.am.<op>()`."""

    def __init__(self, obj):
        self._df = obj

    def to_metal(self):
        return from_pandas(self._df)

    def zero_copy(self):
        return zero_copy_report(self._df)

    def groupby(self, by, dropna=True, sort=True, as_index=True):
        return GPUGroupBy(self._df, by, dropna=dropna, sort=sort, as_index=as_index)

    def query(self, q):
        """Runs a fused ArrowMetal expression query over this frame:

            df.am.query(am.filter(am.col("x") > 3).sum(am.col("y")))

        Only the columns the query names are lifted, zero-copy where the dtype allows: a frame that
        also carries an object or struct column the expression compiler cannot read is fine as long
        as the query does not name it. Column results come back as an Arrow-backed DataFrame;
        scalar results come back as Python scalars."""
        cols = {c: _metal(self._df[c]) for c in _query_columns(q, self._df.columns)}
        out = _am_query(cols, q)
        if isinstance(out, dict) and out and all(isinstance(v, (pa.Array, pa.ChunkedArray))
                                                 for v in out.values()):
            return pd.DataFrame({k: pd.arrays.ArrowExtensionArray(_as_pa_array(v))
                                 for k, v in out.items()})
        return out

    def sort_values(self, by, ascending=True):
        """Multi-column stable GPU sort, nulls last; the index is taken along."""
        cols = [by] if isinstance(by, str) else list(by)
        idx = sort_indices([self._df[c] for c in cols], ascending)
        return self._take(idx)

    def _take(self, idx):
        pos = idx.to_arrow().to_numpy(zero_copy_only=False) if isinstance(idx, MetalArray) else idx
        data = {c: pd.arrays.ArrowExtensionArray(_take(self._df[c], _metal(pa.array(pos))).to_arrow())
                for c in self._df.columns}
        return pd.DataFrame(data, index=self._df.index[pos])

    def filter(self, mask):
        """Boolean-mask selection of rows on the GPU; a null in the mask drops the row."""
        m = _metal(mask).fill_null(False)
        pos = m.indices_nonzero().to_arrow().to_numpy(zero_copy_only=False)
        return self._take(pos.astype(np.int64))

    def merge(self, right, on=None, left_on=None, right_on=None, how="inner", suffixes=("_x", "_y")):
        """Inner merge on one key column, using the GPU hash probe.

        Only `how="inner"` with a unique, null-free key on the right frame runs here (pandas'
        `validate="m:1"` case); anything else raises so the caller can fall back. Left row order and
        pandas' `_x`/`_y` suffixing are preserved; the result gets a fresh RangeIndex, as pandas'
        inner merge does."""
        if how != "inner":
            raise ArrowMetalError("only how='inner' runs on the GPU")
        lk = left_on if left_on is not None else on
        rk = right_on if right_on is not None else on
        if lk is None or rk is None or not isinstance(lk, str) or not isinstance(rk, str):
            raise ArrowMetalError("GPU merge needs exactly one key column")
        lrows, rrows = inner_join_indices(self._df[lk], right[rk])
        lpos = lrows.to_numpy(zero_copy_only=False)
        rpos = rrows.to_numpy(zero_copy_only=False)
        lm, rm = _metal(pa.array(lpos)), _metal(pa.array(rpos))
        overlap = (set(self._df.columns) & set(right.columns)) - ({lk} if lk == rk else set())
        data = {}
        for c in self._df.columns:
            name = f"{c}{suffixes[0]}" if c in overlap else c
            data[name] = pd.arrays.ArrowExtensionArray(_take(self._df[c], lm).to_arrow())
        for c in right.columns:
            if c == rk and lk == rk:
                continue
            name = f"{c}{suffixes[1]}" if c in overlap else c
            data[name] = pd.arrays.ArrowExtensionArray(_take(right[c], rm).to_arrow())
        return pd.DataFrame(data)


_registered = False


def register():
    """Registers the `.am` accessors on Series and DataFrame. Idempotent."""
    global _registered
    if _registered:
        return
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pd.api.extensions.register_series_accessor("am")(SeriesAccessor)
        pd.api.extensions.register_dataframe_accessor("am")(DataFrameAccessor)
    _registered = True


register()
