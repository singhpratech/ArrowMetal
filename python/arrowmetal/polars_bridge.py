"""Polars <-> ArrowMetal: a zero-copy bridge and a `.arrowmetal` namespace on Series, DataFrame
and LazyFrame.

Two functions and three namespaces, and that is the whole surface:

    import polars as pl, arrowmetal as am

    am.from_polars(series_or_df)      # -> MetalArray, or a dict of them, in Metal memory
    am.to_polars(array_or_dict)       # -> pl.Series / pl.DataFrame

    s.arrowmetal.sum()                            # a Python scalar, computed on the GPU
    s.arrowmetal.top_k(100)                       # -> pl.Series
    df.arrowmetal.group_by("k").sum("v")          # -> pl.DataFrame
    df.arrowmetal.query(am.filter(...).sum(...))  # one fused kernel, -> pl.DataFrame or scalar
    df.arrowmetal.sort("v", descending=True)      # -> pl.DataFrame
    df.arrowmetal.join(other, on="k")             # -> pl.DataFrame
    lf.arrowmetal.collect_gpu(q)                  # collect the plan, then one GPU query

Zero copy, and how to check it
------------------------------
`pl.Series.to_arrow()` hands out the Series' own Arrow buffers, and `am_import` maps those pages
into the Metal address space with `bytesNoCopy` when they are page aligned -- which a Polars
column of any real size always is. Nothing is copied in either direction, and you can prove it:

    s = pl.Series("x", np.arange(50_000_000, dtype=np.int64))
    a = s.to_arrow()
    m = am.from_polars(s)
    assert a.buffers()[1].address == m.to_arrow().buffers()[1].address   # the same 400 MB

`am.zero_copy(series)` runs exactly that check and returns True/False, and
`am.zero_copy_report(series)` returns the two addresses so a failure says why.

What still costs something
--------------------------
* **Chunked Series.** ArrowMetal takes one Arrow array. A Series with several chunks (read from
  several files, or built by `pl.concat`) is rechunked once, which does copy. `from_polars` does
  it for you and `rechunk=False` makes it raise instead, so the copy is never silent.
* **Categorical / Enum.** Polars encodes those as a dictionary with uint32 (Categorical) or uint8
  (Enum) indices; ArrowMetal's dictionary support wants int32 or int64. The bridge casts the index
  buffer, which copies 4 bytes a row -- the dictionary values themselves are untouched.
* **Strings.** Free in this direction: Polars' `to_arrow()` already produces `large_string`
  (offsets + bytes), which is what ArrowMetal reads.
* **Wiring the pages.** Mapping a 400 MB buffer into the Metal address space costs about 8 ms on
  an M4 Max -- page-table work, not a copy, and one order of magnitude under the ~35 ms a real
  copy of the same bytes takes. See docs/POLARS.md for the measurements.
"""
import polars as pl
import pyarrow as pa

from . import (
    ArrowMetalError,
    Expr,
    MetalArray,
    Query,
    device_name,
    group_by as _am_group_by,
    lexsort_indices,
    query as _am_query,
    version,
)

__all__ = [
    "from_polars",
    "to_polars",
    "zero_copy",
    "zero_copy_report",
    "register_namespaces",
]


# ---------------------------------------------------------------------------------------------
# from_polars / to_polars
# ---------------------------------------------------------------------------------------------

def _series_to_arrow(s, rechunk=True):
    """One pyarrow.Array for a Polars Series, rechunking (and only then copying) when it has to.

    `Series.to_arrow()` already combines chunks, so the check is explicit: with `rechunk=False` a
    multi-chunk column raises rather than paying for a silent concatenation.
    """
    n = s.n_chunks()
    if n > 1 and not rechunk:
        raise ArrowMetalError(
            f"Series {s.name!r} has {n} chunks; ArrowMetal takes one Arrow array. "
            "Call .rechunk() first, or pass rechunk=True to have the bridge do it (one copy)."
        )
    arr = s.to_arrow()
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
        if isinstance(arr, pa.ChunkedArray):        # pyarrow versions differ on what it returns
            arr = pa.concat_arrays(list(arr.chunks)) if arr.num_chunks else pa.array([], arr.type)
    return _widen_dictionary_indices(arr)


def _widen_dictionary_indices(arr):
    """Polars' Categorical is dictionary<uint32> and its Enum is dictionary<uint8>; ArrowMetal's
    dictionary support wants int32 or int64 indices. Recode the indices (the values ride along
    untouched) so both land as dictionary<int32>."""
    if not pa.types.is_dictionary(arr.type):
        return arr
    if pa.types.is_int32(arr.type.index_type) or pa.types.is_int64(arr.type.index_type):
        return arr
    return pa.DictionaryArray.from_arrays(arr.indices.cast(pa.int32()), arr.dictionary)


def from_polars(obj, *, rechunk=True):
    """Move a Polars object into Metal memory, zero-copy where the buffers allow it.

    * `pl.Series`     -> `MetalArray`
    * `pl.DataFrame`  -> `dict[str, MetalArray]`, one entry per column
    * anything else with `.to_arrow()` (a `pl.LazyFrame` is *not* accepted -- collect it first)

    `rechunk=False` raises on a multi-chunk Series instead of concatenating it, so a copy can
    never happen behind your back.
    """
    if isinstance(obj, pl.Series):
        return MetalArray.from_arrow(_series_to_arrow(obj, rechunk))
    if isinstance(obj, pl.DataFrame):
        return {name: MetalArray.from_arrow(_series_to_arrow(obj[name], rechunk))
                for name in obj.columns}
    if isinstance(obj, pl.LazyFrame):
        raise ArrowMetalError("from_polars needs data: call .collect() on the LazyFrame first, "
                              "or use lf.arrowmetal.collect_gpu(query)")
    if isinstance(obj, MetalArray):
        return obj
    return MetalArray.from_arrow(obj)


def to_polars(obj, name=None):
    """Bring an ArrowMetal result back as a Polars object.

    * `MetalArray` / `pyarrow.Array` -> `pl.Series` (named `name`, or "" )
    * `dict` of those                -> `pl.DataFrame`, keys as column names
    * a Python scalar                -> itself, unchanged, so a reduction can pass straight through
    """
    if isinstance(obj, dict):
        return pl.DataFrame({k: to_polars(v, k) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return [to_polars(v) for v in obj]
    if isinstance(obj, MetalArray):
        obj = obj.to_arrow()
    if isinstance(obj, (pa.Array, pa.ChunkedArray)):
        s = pl.from_arrow(obj)
        return s.rename(name) if name is not None else s
    return obj


# ---------------------------------------------------------------------------------------------
# Zero-copy evidence
# ---------------------------------------------------------------------------------------------

def _data_buffer_address(arr):
    """The address of the values buffer of a pyarrow array, or None when there is not one."""
    bufs = arr.buffers()
    # buffers()[0] is the validity bitmap (often None); the values buffer is the last one for a
    # primitive array and the middle one (offsets, bytes) for a string array.
    for b in bufs[1:]:
        if b is not None:
            return b.address
    return None


def zero_copy_report(obj):
    """`(source_address, metal_address, same)` for one Series or pyarrow array.

    `same` is True when ArrowMetal mapped the producer's own pages rather than copying them.
    Small arrays are often copied (their allocation is not page aligned), which is why the check
    is worth running on the size you actually care about.
    """
    arr = _series_to_arrow(obj) if isinstance(obj, pl.Series) else obj
    src = _data_buffer_address(arr)
    m = MetalArray.from_arrow(arr)
    dst = _data_buffer_address(m.to_arrow())
    return src, dst, (src is not None and src == dst)


def zero_copy(obj):
    """True when importing `obj` into Metal memory copies nothing. See `zero_copy_report`."""
    return zero_copy_report(obj)[2]


# ---------------------------------------------------------------------------------------------
# Series namespace
# ---------------------------------------------------------------------------------------------

@pl.api.register_series_namespace("arrowmetal")
class ArrowMetalSeries:
    """`s.arrowmetal.<op>()` -- one GPU kernel per call, Polars objects in and out.

    Scalar reductions return Python scalars (like `pl.Series.sum()` does); everything else
    returns a `pl.Series`.
    """

    def __init__(self, s: pl.Series):
        self._s = s

    # -- interop
    def to_metal(self, *, rechunk=True) -> MetalArray:
        """This Series as a Metal-resident array. Hold on to it to run several kernels without
        re-importing."""
        return from_polars(self._s, rechunk=rechunk)

    def _m(self):
        return from_polars(self._s)

    def _back(self, m):
        return to_polars(m, self._s.name)

    # -- reductions (Python scalars)
    def sum(self):
        """Arrow `sum` on the GPU. Integers widen to 64 bits, as Arrow's does."""
        return self._m().sum()

    def min(self):
        return self._m().min()

    def max(self):
        return self._m().max()

    def mean(self):
        return self._m().mean()

    def count(self):
        """The number of non-null values."""
        return len(self._s) - self._m().null_count

    # -- selection
    def top_k(self, k: int, *, largest=True) -> pl.Series:
        """The `k` largest values, in ArrowMetal's sort order (stable, nulls last, NaN after
        +inf). `largest=False` gives the k smallest."""
        m = self._m()
        idx = m.top_k(int(k), largest)
        return self._back(m.take(idx))

    def bottom_k(self, k: int) -> pl.Series:
        return self.top_k(k, largest=False)

    def sort(self, *, descending=False) -> pl.Series:
        """A GPU LSD radix sort. Nulls last and NaN after +inf in **both** directions -- a
        descending sort does not mirror them to the front."""
        return self._back(self._m().sort(descending))

    def arg_sort(self, *, descending=False) -> pl.Series:
        """The int32 indices that sort this Series."""
        return to_polars(self._m().argsort(descending), self._s.name)

    def filter(self, mask) -> pl.Series:
        """Keep the rows where `mask` (a boolean `pl.Series`) is true; a null drops the row, which
        is Arrow's and Polars' behaviour."""
        m = self._m()
        return self._back(m.filter(from_polars(mask)))

    def unique(self) -> pl.Series:
        """The distinct non-null values, **ascending** (Polars' `unique()` does not promise an
        order; `unique(maintain_order=True)` gives first-seen instead)."""
        return self._back(self._m().unique())

    # -- element-wise
    def hash64(self) -> pl.Series:
        """A 64-bit hash per value (uint64). Arrow-equal values hash equal, a null hashes to 0 and
        stays null. Strings use the murmur3 kernel, widened to 64 bits."""
        m = self._m()
        h = m.hash32().cast("uint64") if self._s.dtype == pl.String else m.hash64()
        return self._back(h)

    def contains(self, pattern: str) -> pl.Series:
        return self._back(self._m().str_contains(pattern))

    def starts_with(self, pattern: str) -> pl.Series:
        return self._back(self._m().starts_with(pattern))

    def ends_with(self, pattern: str) -> pl.Series:
        return self._back(self._m().ends_with(pattern))

    def upper(self) -> pl.Series:
        """Simple 1:1 case mapping over Basic Latin, Latin-1 Supplement and Latin Extended-A.
        Everything above U+017F passes through, and the multi-character expansions (U+00DF -> SS)
        are not applied; `pl.Series.str.to_uppercase()` does apply them."""
        return self._back(self._m().upper())

    def lower(self) -> pl.Series:
        return self._back(self._m().lower())

    def cum_sum(self) -> pl.Series:
        """A two-level GPU scan. The running value carries across nulls (Arrow's
        `skip_nulls=True`), and the output is null exactly where the input is."""
        return self._back(self._m().cumulative_sum())

    def device(self) -> str:
        return f"ArrowMetal {version()} on {device_name()}"


# ---------------------------------------------------------------------------------------------
# DataFrame namespace
# ---------------------------------------------------------------------------------------------

# Grouped aggregate name -> the GroupByKeys method that answers it.
_GROUP_AGGS = {
    "sum": "sum", "mean": "mean", "min": "min", "max": "max",
    "count": "count", "len": "count_all", "n_unique": "count_distinct",
    "first": "first", "last": "last", "median": "approximate_median",
    "std": "stddev", "var": "variance", "product": "product",
    "any": "any", "all": "all",
}


class GpuGroupBy:
    """The object `df.arrowmetal.group_by(...)` returns.

    One `am_group_by_keys` pass builds the dense group ids; every aggregate after that is one
    segmented GPU kernel over the same mapping, so `.agg(...)` with several outputs costs one
    group-by, not one per aggregate.

    The result carries the key columns first, then the aggregates. **Group order is ArrowMetal's**
    -- ascending by key for numeric, boolean, temporal and decimal keys, first-seen for utf8 and
    binary, lexicographic in column order for several keys -- so sort both sides before comparing
    with `df.group_by(...).agg(...)`, which makes no order promise at all.
    """

    def __init__(self, df: pl.DataFrame, keys):
        self._df = df
        self._keys = list(keys)
        missing = [k for k in self._keys if k not in df.columns]
        if missing:
            raise ArrowMetalError(f"group_by: no such column(s): {missing}")
        self._columns = {}
        self._gb = _am_group_by([self._column(k) for k in self._keys])

    def _column(self, name):
        m = self._columns.get(name)
        if m is None:
            m = self._columns[name] = from_polars(self._df[name])
        return m

    def __len__(self):
        return self._gb.group_count

    @property
    def group_count(self):
        return self._gb.group_count

    def _key_frame(self):
        return {name: to_polars(arr, name) for name, arr in zip(self._keys, self._gb.keys())}

    def agg(self, *args, **named):
        """Several aggregates over one group-by.

        Two spellings, both of which name every output column:

            gb.agg(total=("v", "sum"), n=("v", "count"))
            gb.agg({"total": ("v", "sum")})

        The aggregate name is one of: sum, mean, min, max, count, len, n_unique, first, last,
        median, std, var, product, any, all. `len` counts rows and takes no column, so pass
        `n=("", "len")` or `n=(None, "len")`.
        """
        spec = {}
        for a in args:
            if not isinstance(a, dict):
                raise ArrowMetalError("agg() takes a dict or keyword arguments")
            spec.update(a)
        spec.update(named)
        if not spec:
            raise ArrowMetalError("agg() needs at least one aggregate")

        out = self._key_frame()
        for alias, item in spec.items():
            column, how = item if isinstance(item, (tuple, list)) else (item, "sum")
            method = _GROUP_AGGS.get(how)
            if method is None:
                raise ArrowMetalError(
                    f"unknown aggregate {how!r}; known: {', '.join(sorted(_GROUP_AGGS))}")
            if method == "count_all":
                res = self._gb.count_all()
            else:
                if column not in self._df.columns:
                    raise ArrowMetalError(f"agg: no such column: {column!r}")
                res = getattr(self._gb, method)(self._column(column))
            out[alias] = to_polars(res, alias)
        return pl.DataFrame(out)

    def _simple(self, how, columns):
        columns = list(columns) or [c for c in self._df.columns if c not in self._keys]
        return self.agg({c: (c, how) for c in columns})

    def sum(self, *columns):
        """`gb.sum("v")`, or `gb.sum()` for every non-key column."""
        return self._simple("sum", columns)

    def mean(self, *columns):
        return self._simple("mean", columns)

    def min(self, *columns):
        return self._simple("min", columns)

    def max(self, *columns):
        return self._simple("max", columns)

    def median(self, *columns):
        """Exact: a GPU sort by (group, value), not a sketch."""
        return self._simple("median", columns)

    def n_unique(self, *columns):
        return self._simple("n_unique", columns)

    def count(self):
        """One row per group with the number of rows in it, as `len`."""
        return self.agg(len=(None, "len"))

    def len(self):
        return self.count()

    def quantile(self, column, q):
        """Exact per-group quantile with linear interpolation."""
        out = self._key_frame()
        out[column] = to_polars(self._gb.quantile(self._column(column), q), column)
        return pl.DataFrame(out)

    def keys(self) -> pl.DataFrame:
        """Just the distinct key rows, in group order."""
        return pl.DataFrame(self._key_frame())

    def ids(self) -> pl.Series:
        """The dense group id of every input row (int32, never null)."""
        return to_polars(self._gb.ids(), "group_id")


@pl.api.register_dataframe_namespace("arrowmetal")
class ArrowMetalFrame:
    """`df.arrowmetal.<op>()` -- whole-frame operations that run on the GPU and hand back Polars."""

    def __init__(self, df: pl.DataFrame):
        self._df = df

    # -- interop
    def to_metal(self, *, rechunk=True):
        """`dict[str, MetalArray]`, one per column, in Metal memory."""
        return from_polars(self._df, rechunk=rechunk)

    def device(self) -> str:
        return f"ArrowMetal {version()} on {device_name()}"

    # -- group-by
    def group_by(self, *keys) -> GpuGroupBy:
        """`df.arrowmetal.group_by("k").sum("v")`. Keys may be any supported type -- integers
        sparse or negative, floats, bool, temporal, utf8, binary, dictionary, decimal -- and
        several keys fold into one mapping."""
        if len(keys) == 1 and isinstance(keys[0], (list, tuple)):
            keys = tuple(keys[0])
        if not keys:
            raise ArrowMetalError("group_by needs at least one key column")
        return GpuGroupBy(self._df, keys)

    # -- fused queries
    def query(self, q):
        """Run one fused ArrowMetal expression query over this frame (docs/EXPR.md).

        The whole expression DAG compiles into a single Metal kernel, so the columns are read
        once however many operators the query has:

            df.arrowmetal.query(am.filter(am.col("region") == 2).sum(am.col("amount")))

        A `project` or `group_by` query comes back as a `pl.DataFrame`; an `aggregate` query
        comes back as a Python scalar (or a dict of them, for several aggregates).
        """
        columns = from_polars(self._df)
        res = _am_query(columns, q)
        if isinstance(res, dict) and res and all(isinstance(v, (pa.Array, pa.ChunkedArray))
                                                 for v in res.values()):
            return pl.DataFrame({k: to_polars(v, k) for k, v in res.items()})
        return res

    # -- ordering
    def sort(self, by, *, descending=False) -> pl.DataFrame:
        """Sort the whole frame on the GPU: one lexicographic radix argsort, then one take per
        column. `descending` is a bool or one flag per key.

        Nulls come last in every key, in both directions -- Polars' default is
        `nulls_last=False` for an ascending sort, so pass `nulls_last=True` when comparing.
        """
        keys = [by] if isinstance(by, str) else list(by)
        flags = [descending] * len(keys) if isinstance(descending, bool) else list(descending)
        if len(flags) != len(keys):
            raise ArrowMetalError(f"descending has {len(flags)} entries for {len(keys)} keys")
        columns = from_polars(self._df)
        idx = lexsort_indices([columns[k] for k in keys], flags)
        return pl.DataFrame({name: to_polars(m.take(idx), name) for name, m in columns.items()})

    def top_k(self, k: int, *, by: str, largest=True) -> pl.DataFrame:
        """The `k` rows with the largest (or smallest) `by` value, as a frame."""
        columns = from_polars(self._df)
        idx = columns[by].top_k(int(k), largest)
        return pl.DataFrame({name: to_polars(m.take(idx), name) for name, m in columns.items()})

    def filter(self, mask) -> pl.DataFrame:
        """Keep the rows where `mask` is true. `mask` is a boolean `pl.Series`, a boolean column
        name, or a `MetalArray`; a null drops the row."""
        if isinstance(mask, str):
            mask = self._df[mask]
        m = from_polars(mask)
        columns = from_polars(self._df)
        return pl.DataFrame({name: to_polars(c.filter(m), name) for name, c in columns.items()})

    # -- joins
    def join(self, other: pl.DataFrame, *, on=None, left_on=None, right_on=None,
             how="inner", suffix="_right", allow_cpu_fallback=True) -> pl.DataFrame:
        """Join on the GPU where the shape allows it, otherwise fall back to Polars.

        ArrowMetal publishes no join kernel, so this is built out of the ones it does have:
        `am_index_in` finds, for every left key, the row of the right key column it matches, and
        `am_take` / `am_filter` then gather both sides. That is a complete inner or left join
        **when the right key is unique** -- the usual dimension-table shape -- and the whole thing
        (uniqueness check included) runs on the GPU with no row-by-row work on the host.

        Everything else -- a duplicated right key (which changes the row count), several join
        keys, an outer/semi/anti join -- falls back to `pl.DataFrame.join`, which is correct and
        fast but on the CPU. `allow_cpu_fallback=False` raises instead of falling back, so a
        benchmark can be sure of what it measured.

        Null keys never match, matching Polars' default `join_nulls=False`.
        """
        if how not in ("inner", "left"):
            return self._fallback_join(other, on, left_on, right_on, how, suffix,
                                       allow_cpu_fallback, f"how={how!r} has no GPU path")
        lk, rk = self._join_keys(on, left_on, right_on)
        if len(lk) != 1:
            return self._fallback_join(other, on, left_on, right_on, how, suffix,
                                       allow_cpu_fallback,
                                       "a multi-column join key has no GPU path")

        left_key = from_polars(self._df[lk[0]])
        right_key = from_polars(other[rk[0]])
        # One GPU group-by decides uniqueness: as many groups as rows means every key is distinct.
        if _am_group_by([right_key]).group_count != len(other):
            return self._fallback_join(other, on, left_on, right_on, how, suffix,
                                       allow_cpu_fallback,
                                       "the right key is not unique, so the join changes the "
                                       "row count in a way index_in cannot express")

        idx = left_key.index_in(right_key)          # int32 into `other`, null where absent
        left_cols = from_polars(self._df)
        right_cols = {n: from_polars(other[n]) for n in other.columns if n not in rk}

        if how == "left":
            take_idx, keep = idx, None
        else:
            keep = idx.is_valid()
            take_idx = idx.filter(keep)

        out = {}
        for name, col in left_cols.items():
            out[name] = to_polars(col if keep is None else col.filter(keep), name)
        for name, col in right_cols.items():
            alias = name + suffix if name in left_cols else name
            out[alias] = to_polars(col.take(take_idx), alias)
        return pl.DataFrame(out)

    def _join_keys(self, on, left_on, right_on):
        if on is not None:
            keys = [on] if isinstance(on, str) else list(on)
            return keys, keys
        if left_on is None or right_on is None:
            raise ArrowMetalError("join needs on=..., or both left_on=... and right_on=...")
        left = [left_on] if isinstance(left_on, str) else list(left_on)
        right = [right_on] if isinstance(right_on, str) else list(right_on)
        if len(left) != len(right):
            raise ArrowMetalError("left_on and right_on must have the same length")
        return left, right

    def _fallback_join(self, other, on, left_on, right_on, how, suffix, allowed, why):
        if not allowed:
            raise ArrowMetalError(f"no GPU join for this shape: {why}")
        kwargs = {"how": how, "suffix": suffix}
        if on is not None:
            kwargs["on"] = on
        else:
            kwargs["left_on"], kwargs["right_on"] = left_on, right_on
        return self._df.join(other, **kwargs)


# ---------------------------------------------------------------------------------------------
# LazyFrame namespace
# ---------------------------------------------------------------------------------------------

@pl.api.register_lazyframe_namespace("arrowmetal")
class ArrowMetalLazy:
    """`lf.arrowmetal.collect_gpu(...)` -- the streaming hand-off.

    Polars' own `engine="gpu"` hook is cuDF-only today (see docs/POLARS.md for what a Metal
    backend would have to implement), so the hand-off is explicit: Polars runs the plan, then
    ArrowMetal runs one GPU pass over the collected frame.
    """

    def __init__(self, lf: pl.LazyFrame):
        self._lf = lf

    def collect_gpu(self, q=None, *, engine="auto", **collect_kwargs):
        """Collect the Polars plan, then run `q` over the result in one GPU batch.

        `q` is an ArrowMetal `Query` (`am.filter(...).sum(...)`), an `Expr`, or a callable taking
        the collected `pl.DataFrame` and returning whatever you want -- which is the escape hatch
        for the `.arrowmetal` namespace methods:

            lf.arrowmetal.collect_gpu(lambda df: df.arrowmetal.group_by("k").sum("v"))

        With `q=None` this is just `lf.collect(engine=engine)`.

        `engine` and any other keyword go to `LazyFrame.collect`, so the Polars half can still
        use the streaming engine: everything Polars can push into the scan (projection pushdown,
        predicate pushdown, `slice`) happens before a single byte reaches the GPU.
        """
        df = self._lf.collect(engine=engine, **collect_kwargs)
        if q is None:
            return df
        if callable(q) and not isinstance(q, (Query, Expr)):
            return q(df)
        return df.arrowmetal.query(q)

    def device(self) -> str:
        return f"ArrowMetal {version()} on {device_name()}"


def register_namespaces():
    """Idempotent no-op kept for symmetry: importing this module registers the three namespaces.

    Returns the names it registered, so a caller can assert on them."""
    return ("Series.arrowmetal", "DataFrame.arrowmetal", "LazyFrame.arrowmetal")
