"""Out-of-core streaming execution: datasets larger than memory, answered on the Apple GPU.

    import arrowmetal as am

    s = am.scan_ipc("/data/events")                  # a file or a directory of Arrow IPC files
    s.filter(am.col("amount") > 100).sum("amount")   # one pass, bounded memory

A scan is a source of Arrow record batches plus the filter and projection built on it. A terminal
(`.sum`, `.group_by(...).agg`, `.top_k`, `.sort`, `.quantile`, `.sink_ipc`, `.to_reader`, ...) runs a
three-stage pipeline in the Swift core: the reader thread maps and imports batch i + 1 while the GPU
runs batch i and the merge thread folds batch i - 1's result into the running state. Nothing but the
answer grows with the input, so 100+ GB is answered in a few GB of RAM.

Parquet, CSV and partitioned datasets arrive through `scan_arrow`, which takes anything that speaks
the Arrow C Stream ABI: a `pyarrow.RecordBatchReader`, a `pyarrow.dataset.Dataset` or its scanner, a
`pyarrow.Table`, or a Polars `LazyFrame`.

Every terminal leaves the run's statistics on `.stats`: batches, rows, bytes read, the time each
pipeline stage was busy, and `overlap` = (read + gpu + merge) / wall, which is 1.0 for a serial
pipeline and approaches 3.0 for three fully overlapped stages.
"""
import ctypes
import os

import pyarrow as pa

from . import _lib, _P, _check, ArrowMetalError, MetalArray, Expr, _as_expr

__all__ = ["scan_ipc", "scan_arrow", "scan_table", "Stream", "GroupedStream"]

# Aggregate op codes: the C ABI contract (see include/arrowmetal.h).
_STREAM_AGG = {"sum": 0, "count": 1, "min": 2, "max": 3, "mean": 4,
               "variance": 5, "var": 5, "stddev": 6, "std": 6,
               "count_distinct_approx": 7, "n_unique_approx": 7}
_JOIN_KIND = {"inner": 0, "left": 1}


class _ArrowArrayStream(ctypes.Structure):
    pass


_ArrowArrayStream._fields_ = [
    ("get_schema", ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(_ArrowArrayStream), ctypes.c_void_p)),
    ("get_next", ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(_ArrowArrayStream), ctypes.c_void_p)),
    ("get_last_error", ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.POINTER(_ArrowArrayStream))),
    ("release", ctypes.CFUNCTYPE(None, ctypes.POINTER(_ArrowArrayStream))),
    ("private_data", ctypes.c_void_p),
]

_S = _P            # am_stream*
_R = _P            # am_stream_result*

_lib.am_stream_last_error.restype = ctypes.c_char_p
_lib.am_stream_open_ipc.argtypes = [ctypes.c_char_p, ctypes.c_int]
_lib.am_stream_open_ipc.restype = _S
_lib.am_stream_from_c_stream.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.am_stream_from_c_stream.restype = _S
_lib.am_stream_release.argtypes = [_S]
_lib.am_stream_filter.argtypes = [_S, ctypes.c_char_p]
_lib.am_stream_filter.restype = ctypes.c_int
_lib.am_stream_select.argtypes = [_S, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int64]
_lib.am_stream_select.restype = ctypes.c_int
_lib.am_stream_project.argtypes = [_S, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p),
                                   ctypes.c_int64]
_lib.am_stream_project.restype = ctypes.c_int
_PROGRESS_FN = ctypes.CFUNCTYPE(None, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                                ctypes.c_void_p)
_lib.am_stream_set_progress.argtypes = [_S, _PROGRESS_FN, ctypes.c_void_p]
_lib.am_stream_set_progress.restype = ctypes.c_int

_lib.am_stream_query.argtypes = [_S, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(_R)]
_lib.am_stream_query.restype = ctypes.c_int
_lib.am_stream_aggregate.argtypes = [_S, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_char_p),
                                     ctypes.POINTER(ctypes.c_char_p), ctypes.c_int64, ctypes.c_int,
                                     ctypes.c_int, ctypes.POINTER(_R)]
_lib.am_stream_aggregate.restype = ctypes.c_int
_lib.am_stream_group_by.argtypes = [_S, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int64,
                                    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_char_p),
                                    ctypes.POINTER(ctypes.c_char_p), ctypes.c_int64,
                                    ctypes.c_int64, ctypes.c_int, ctypes.POINTER(_R)]
_lib.am_stream_group_by.restype = ctypes.c_int
_lib.am_stream_top_k.argtypes = [_S, ctypes.c_char_p, ctypes.c_int64, ctypes.c_int, ctypes.POINTER(_R)]
_lib.am_stream_top_k.restype = ctypes.c_int
_lib.am_stream_quantiles.argtypes = [_S, ctypes.c_char_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int64,
                                     ctypes.c_int64, ctypes.POINTER(_R)]
_lib.am_stream_quantiles.restype = ctypes.c_int
_lib.am_stream_sort.argtypes = [_S, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int),
                                ctypes.c_int64, ctypes.c_int64, ctypes.c_char_p, ctypes.c_char_p,
                                ctypes.POINTER(_R)]
_lib.am_stream_sort.restype = ctypes.c_int
_lib.am_stream_sink_ipc.argtypes = [_S, ctypes.c_char_p, ctypes.POINTER(_R)]
_lib.am_stream_sink_ipc.restype = ctypes.c_int
_lib.am_stream_join_broadcast.argtypes = [_S, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
                                          ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(_R)]
_lib.am_stream_join_broadcast.restype = ctypes.c_int
_lib.am_stream_join_grace.argtypes = [_S, _S, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
                                      ctypes.c_int64, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(_R)]
_lib.am_stream_join_grace.restype = ctypes.c_int
_lib.am_stream_export_c.argtypes = [_S, ctypes.c_void_p]
_lib.am_stream_export_c.restype = ctypes.c_int

_lib.am_stream_result_column_count.argtypes = [_R]
_lib.am_stream_result_column_count.restype = ctypes.c_int64
_lib.am_stream_result_column_name.argtypes = [_R, ctypes.c_int64]
_lib.am_stream_result_column_name.restype = ctypes.c_char_p
_lib.am_stream_result_column.argtypes = [_R, ctypes.c_int64, ctypes.POINTER(_P)]
_lib.am_stream_result_column.restype = ctypes.c_int
_lib.am_stream_result_scalar_count.argtypes = [_R]
_lib.am_stream_result_scalar_count.restype = ctypes.c_int64
_lib.am_stream_result_scalar_name.argtypes = [_R, ctypes.c_int64]
_lib.am_stream_result_scalar_name.restype = ctypes.c_char_p
_lib.am_stream_result_scalar.argtypes = [_R, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64),
                                         ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
                                         ctypes.POINTER(ctypes.c_int)]
_lib.am_stream_result_scalar.restype = ctypes.c_int
_lib.am_stream_result_rows_out.argtypes = [_R]
_lib.am_stream_result_rows_out.restype = ctypes.c_int64
_lib.am_stream_result_stats.argtypes = [_R] + [ctypes.POINTER(ctypes.c_int64)] * 3 + \
                                       [ctypes.POINTER(ctypes.c_double)] * 5
_lib.am_stream_result_stats.restype = ctypes.c_int
_lib.am_stream_result_release.argtypes = [_R]


def _stream_error():
    return (_lib.am_stream_last_error() or b"unknown streaming error").decode()


def _check_stream(rc):
    if rc != 0:
        raise ArrowMetalError(_stream_error())


def _c_strings(values):
    """A NULL-terminated-free `const char**` from a list of str (or None for a NULL slot)."""
    n = max(len(values), 1)
    arr = (ctypes.c_char_p * n)()
    for i, v in enumerate(values):
        arr[i] = None if v is None else v.encode()
    return arr


class _Result:
    """One terminal's answer: columns, scalars and the pipeline statistics."""

    def __init__(self, handle):
        self._h = handle

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            _lib.am_stream_result_release(h)

    def table(self):
        n = _lib.am_stream_result_column_count(self._h)
        if n <= 0:
            return None
        names, cols = [], []
        for i in range(n):
            names.append(_lib.am_stream_result_column_name(self._h, i).decode())
            out = _P()
            _check_stream(_lib.am_stream_result_column(self._h, i, ctypes.byref(out)))
            cols.append(MetalArray(out).to_arrow())
        return pa.table(cols, names=names)

    def scalars(self):
        out = {}
        for i in range(_lib.am_stream_result_scalar_count(self._h)):
            name = _lib.am_stream_result_scalar_name(self._h, i).decode()
            iv, fv = ctypes.c_int64(), ctypes.c_double()
            kind, isnull = ctypes.c_int(), ctypes.c_int()
            _check_stream(_lib.am_stream_result_scalar(self._h, i, ctypes.byref(iv), ctypes.byref(fv),
                                                       ctypes.byref(kind), ctypes.byref(isnull)))
            if isnull.value:
                out[name] = None
            elif kind.value == 2:
                out[name] = fv.value
            elif kind.value == 1:
                out[name] = iv.value & 0xFFFFFFFFFFFFFFFF
            else:
                out[name] = iv.value
        return out

    def rows_out(self):
        return _lib.am_stream_result_rows_out(self._h)

    def stats(self):
        b, r, by = ctypes.c_int64(), ctypes.c_int64(), ctypes.c_int64()
        wall, read, gpu, merge, overlap = (ctypes.c_double() for _ in range(5))
        _check_stream(_lib.am_stream_result_stats(self._h, ctypes.byref(b), ctypes.byref(r), ctypes.byref(by),
                                                  ctypes.byref(wall), ctypes.byref(read), ctypes.byref(gpu),
                                                  ctypes.byref(merge), ctypes.byref(overlap)))
        wall_s = wall.value
        return {"batches": b.value, "rows": r.value, "bytes_read": by.value,
                "wall_s": wall_s, "read_s": read.value, "gpu_s": gpu.value, "merge_s": merge.value,
                "overlap": overlap.value,
                "gb_per_s": (by.value / 1e9 / wall_s) if wall_s > 0 else 0.0}


class Stream:
    """A streamed dataset with a filter and a projection on it. Terminals consume the stream."""

    def __init__(self, handle, description=""):
        if not handle:
            raise ArrowMetalError(_stream_error())
        self._h = handle
        self._desc = description
        self._consumed = False
        self._progress_cb = None      # kept alive: ctypes would otherwise free the trampoline
        self.stats = {}
        self.rows_out = 0

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            _lib.am_stream_release(h)

    def __repr__(self):
        return f"Stream({self._desc!r})"

    # ---- building

    def filter(self, pred):
        """Keep the rows where `pred` is true and not null. Several calls AND together."""
        _check_stream(_lib.am_stream_filter(self._h, _as_expr(pred).sexpr().encode()))
        return self

    def select(self, names):
        """Keep only these columns."""
        if isinstance(names, str):
            names = [names]
        arr = _c_strings(list(names))
        _check_stream(_lib.am_stream_select(self._h, arr, len(names)))
        return self

    def project(self, exprs):
        """Materialise computed columns: a dict {name: expr} or a list of (name, expr) pairs."""
        items = list(exprs.items()) if isinstance(exprs, dict) else list(exprs)
        names = _c_strings([n for n, _ in items])
        texts = _c_strings([_as_expr(e).sexpr() for _, e in items])
        _check_stream(_lib.am_stream_project(self._h, names, texts, len(items)))
        return self

    def progress(self, fn):
        """Call `fn(batches, rows, bytes_read, total_bytes)` once per batch. Pass None to clear."""
        if fn is None:
            self._progress_cb = None
            _check_stream(_lib.am_stream_set_progress(self._h, ctypes.cast(None, _PROGRESS_FN), None))
            return self

        def trampoline(batches, rows, read, total, _user):
            fn(batches, rows, read, total)

        self._progress_cb = _PROGRESS_FN(trampoline)
        _check_stream(_lib.am_stream_set_progress(self._h, self._progress_cb, None))
        return self

    # ---- terminals

    def _terminal(self, call):
        if self._consumed:
            raise ArrowMetalError("this stream has already been consumed; open a new scan")
        self._consumed = True
        out = _R()
        _check_stream(call(ctypes.byref(out)))
        r = _Result(out)
        self.stats = r.stats()
        self.rows_out = r.rows_out()
        return r

    def aggregate(self, specs, hll_precision=14, ddof=0):
        """Whole-dataset aggregates. `specs` is a list of (op, column, name); `column` may be None
        for `count`. Ops: sum, count, min, max, mean, variance, stddev, count_distinct_approx.

        Every op but `count_distinct_approx` is exact: it is decomposed into pieces each batch can
        produce on its own and the merge combines associatively."""
        specs = [tuple(s) if not isinstance(s, str) else (s, None, s) for s in specs]
        ops = (ctypes.c_int * max(len(specs), 1))()
        for i, (op, _, _) in enumerate(specs):
            if op not in _STREAM_AGG:
                raise ArrowMetalError(f"unknown streaming aggregate {op!r}")
            ops[i] = _STREAM_AGG[op]
        cols = _c_strings([c for _, c, _ in specs])
        names = _c_strings([n for _, _, n in specs])
        r = self._terminal(lambda out: _lib.am_stream_aggregate(self._h, ops, cols, names, len(specs),
                                                                hll_precision, ddof, out))
        return r.scalars()

    def sum(self, column):
        return self.aggregate([("sum", column, "sum")])["sum"]

    def count(self):
        return self.aggregate([("count", None, "count")])["count"]

    def mean(self, column):
        return self.aggregate([("mean", column, "mean")])["mean"]

    def min(self, column):
        return self.aggregate([("min", column, "min")])["min"]

    def max(self, column):
        return self.aggregate([("max", column, "max")])["max"]

    def variance(self, column, ddof=0):
        return self.aggregate([("variance", column, "var")], ddof=ddof)["var"]

    def stddev(self, column, ddof=0):
        return self.aggregate([("stddev", column, "std")], ddof=ddof)["std"]

    def count_distinct_approx(self, column, precision=14):
        """Approximate distinct count from a GPU HyperLogLog sketch merged across batches.

        The relative standard error is 1.04 / sqrt(2^precision): 0.81% at the default 14, 0.41% at
        16. The sketch is 2^precision bytes whatever the dataset's size."""
        return self.aggregate([("count_distinct_approx", column, "ndv")], hll_precision=precision)["ndv"]

    def group_by(self, keys, dense_key_count=0):
        """Group by one or more key columns of any type. `dense_key_count` promises a single integer
        key already inside `[0, dense_key_count)` and keeps the global aggregate table on the GPU."""
        if isinstance(keys, str):
            keys = [keys]
        return GroupedStream(self, list(keys), dense_key_count)

    def top_k(self, column, k, largest=True):
        """The k rows with the largest (or smallest) value of `column`, as a pyarrow.Table. Exact:
        the k best of a union are always inside the union of each part's k best."""
        r = self._terminal(lambda out: _lib.am_stream_top_k(self._h, column.encode(), k,
                                                            1 if largest else 0, out))
        return r.table()

    def quantile(self, column, q, compression=1000):
        """Approximate quantiles from a GPU digest merged across batches. `q` is a float or a list;
        the return matches. Rank error is roughly 1 / compression at the median and far smaller in
        the tails (the digest samples on the t-digest k1 scale)."""
        want = [float(q)] if isinstance(q, (int, float)) else [float(x) for x in q]
        arr = (ctypes.c_double * len(want))(*want)
        r = self._terminal(lambda out: _lib.am_stream_quantiles(self._h, column.encode(), arr, len(want),
                                                                compression, out))
        vals = r.scalars()
        ordered = [vals.get(f"q{x}") for x in want]
        return ordered[0] if isinstance(q, (int, float)) else ordered

    def sort(self, by, limit=0, scratch=None):
        """External sort, collected into a pyarrow.Table.

        `by` is a column name, a (name, descending) pair, or a list of either. Sorted runs go to
        `scratch` (a temporary directory by default) and are merged k-way. Pass `limit` unless the
        whole sorted result really fits in memory; without it every row is collected."""
        cols, desc = _sort_keys(by)
        names = _c_strings(cols)
        flags = (ctypes.c_int * max(len(cols), 1))(*[1 if d else 0 for d in desc])
        r = self._terminal(lambda out: _lib.am_stream_sort(
            self._h, names, flags, len(cols), limit,
            None if scratch is None else str(scratch).encode(), None, out))
        return r.table()

    def sort_to_ipc(self, by, path, scratch=None):
        """External sort straight into an Arrow IPC stream file; returns the run's statistics."""
        cols, desc = _sort_keys(by)
        names = _c_strings(cols)
        flags = (ctypes.c_int * max(len(cols), 1))(*[1 if d else 0 for d in desc])
        self._terminal(lambda out: _lib.am_stream_sort(
            self._h, names, flags, len(cols), 0,
            None if scratch is None else str(scratch).encode(), str(path).encode(), out))
        return self.stats

    def sink_ipc(self, path):
        """Stream the filtered and projected rows into an Arrow IPC stream file. Returns the run's
        statistics; `rows_out` on this object is the row count written."""
        self._terminal(lambda out: _lib.am_stream_sink_ipc(self._h, str(path).encode(), out))
        return self.stats

    def collect(self):
        """Collect the filtered and projected rows into one pyarrow.Table. Only for small results —
        use `sink_ipc` or `to_reader` otherwise."""
        reader = self.to_reader()
        return reader.read_all()

    def to_reader(self):
        """A `pyarrow.RecordBatchReader` over the filtered and projected rows.

        Pull based and zero copy: each `read_next_batch` drives exactly one source batch through the
        whole pipeline, so pyarrow or Polars can consume an out-of-core query lazily."""
        if self._consumed:
            raise ArrowMetalError("this stream has already been consumed; open a new scan")
        self._consumed = True
        cstream = _ArrowArrayStream()
        _check_stream(_lib.am_stream_export_c(self._h, ctypes.addressof(cstream)))
        return pa.RecordBatchReader._import_from_c(ctypes.addressof(cstream))

    def join(self, other, on, right_on=None, how="inner", broadcast=True, partitions=16,
             scratch=None, sink=None):
        """Join this streamed dataset with `other`.

        With `broadcast=True` (the default) `other` is the build side: it is read into memory once
        and every probe batch is joined against it on the GPU. Use it whenever one side fits.

        With `broadcast=False` a grace hash join runs: both sides are streamed and partitioned by a
        hash of the key into Arrow IPC files under `scratch`, then joined partition by partition, so
        neither side ever has to fit. `other` must then be another `Stream`.

        Returns a pyarrow.Table, or the run's statistics when `sink` names an IPC file."""
        right_on = right_on or on
        kind = _JOIN_KIND.get(how)
        if kind is None:
            raise ArrowMetalError(f"unsupported join type {how!r}; use 'inner' or 'left'")
        if broadcast:
            build = _export_c_stream(other)
            r = self._terminal(lambda out: _lib.am_stream_join_broadcast(
                self._h, ctypes.addressof(build), on.encode(), right_on.encode(), kind,
                None if sink is None else str(sink).encode(), out))
        else:
            if not isinstance(other, Stream):
                other = scan_arrow(other)
            other._consumed = True
            r = self._terminal(lambda out: _lib.am_stream_join_grace(
                self._h, other._h, on.encode(), right_on.encode(), kind, partitions,
                None if scratch is None else str(scratch).encode(),
                None if sink is None else str(sink).encode(), out))
        return self.stats if sink is not None else r.table()


class GroupedStream:
    """`stream.group_by(keys)`: waiting for `.agg(...)`."""

    def __init__(self, stream, keys, dense_key_count=0):
        self._stream = stream
        self._keys = keys
        self._dense = dense_key_count

    def agg(self, specs, ddof=0):
        """Aggregate each group. `specs` is a list of (op, column, name) with `column` None for
        `count`; the result is a pyarrow.Table with the key columns first, one row per group, in
        ascending key order.

        The global table survives across batches: on the GPU for a dense integer key, on the host
        (keyed by the key value itself) for arbitrary keys. Both are exact."""
        specs = [tuple(s) for s in specs]
        ops = (ctypes.c_int * max(len(specs), 1))()
        for i, (op, _, _) in enumerate(specs):
            if op not in _STREAM_AGG:
                raise ArrowMetalError(f"unknown streaming aggregate {op!r}")
            ops[i] = _STREAM_AGG[op]
        keys = _c_strings(self._keys)
        cols = _c_strings([c for _, c, _ in specs])
        names = _c_strings([n for _, _, n in specs])
        s = self._stream
        r = s._terminal(lambda out: _lib.am_stream_group_by(s._h, keys, len(self._keys), ops, cols, names,
                                                            len(specs), self._dense, ddof, out))
        return r.table()

    def count(self, name="count"):
        return self.agg([("count", None, name)])

    def sum(self, column, name=None):
        return self.agg([("sum", column, name or f"sum_{column}")])


def _sort_keys(by):
    """Normalises `by` into (column names, descending flags)."""
    if isinstance(by, str):
        return [by], [False]
    if isinstance(by, tuple) and len(by) == 2 and isinstance(by[0], str) and isinstance(by[1], bool):
        return [by[0]], [by[1]]
    cols, desc = [], []
    for item in by:
        if isinstance(item, str):
            cols.append(item)
            desc.append(False)
        else:
            cols.append(item[0])
            desc.append(bool(item[1]))
    return cols, desc


def _as_reader(obj):
    """A `pyarrow.RecordBatchReader` over anything this module accepts as a source."""
    if isinstance(obj, pa.RecordBatchReader):
        return obj
    if isinstance(obj, pa.Table):
        return obj.to_reader()
    if isinstance(obj, pa.RecordBatch):
        return pa.RecordBatchReader.from_batches(obj.schema, [obj])
    # pyarrow.dataset.Dataset / Scanner, and anything else exposing to_reader / scanner.
    if hasattr(obj, "to_reader"):
        return obj.to_reader()
    if hasattr(obj, "scanner"):
        return obj.scanner().to_reader()
    # Polars LazyFrame: prefer a streaming collect so the source itself stays out of core.
    if hasattr(obj, "collect") and hasattr(obj, "lazy"):
        try:
            df = obj.collect(engine="streaming")
        except TypeError:
            df = obj.collect(streaming=True)
        return df.to_arrow().to_reader()
    if hasattr(obj, "to_arrow"):                      # Polars DataFrame, DuckDB relation, ...
        t = obj.to_arrow()
        return t.to_reader() if isinstance(t, pa.Table) else pa.table(t).to_reader()
    if hasattr(obj, "__arrow_c_stream__"):
        return pa.RecordBatchReader.from_stream(obj)
    raise ArrowMetalError(f"cannot stream from {type(obj).__name__}; pass a pyarrow reader, dataset, "
                          "table, or a Polars LazyFrame")


def _export_c_stream(obj):
    """Exports `obj` as a C ArrowArrayStream struct the caller keeps alive for the call."""
    reader = _as_reader(obj)
    out = _ArrowArrayStream()
    reader._export_to_c(ctypes.addressof(out))
    return out


def scan_ipc(path, prefetch=3, batch_rows=None):
    """Stream an Arrow IPC file, or a directory of them, off disk.

    The file is memory mapped and read one record batch at a time, with `prefetch` batches read
    ahead on their own thread and the unified buffer cache warmed ahead of the read cursor. Pass
    `prefetch=0` to read synchronously.

    `batch_rows` is accepted for symmetry with `scan_arrow` but the IPC file's own record batch
    boundaries decide the batch size; it is ignored (with no error) when the file already has them."""
    del batch_rows      # the IPC file's own record batches are the unit of streaming
    path = os.fspath(path)
    return Stream(_lib.am_stream_open_ipc(str(path).encode(), int(prefetch)), description=str(path))


def scan_arrow(obj, prefetch=3):
    """Stream from any Arrow producer: a `pyarrow.RecordBatchReader`, a `pyarrow.dataset.Dataset`
    or scanner (so Parquet, CSV and partitioned datasets work today through pyarrow's readers), a
    `pyarrow.Table`, a Polars `LazyFrame`, or anything exposing `__arrow_c_stream__`.

    Ownership of the stream moves into ArrowMetal; do not use the reader afterwards."""
    cstream = _export_c_stream(obj)
    h = _lib.am_stream_from_c_stream(ctypes.addressof(cstream), int(prefetch))
    return Stream(h, description=type(obj).__name__)


def scan_table(table, batch_rows=1 << 20):
    """Stream an in-memory pyarrow Table in `batch_rows`-row chunks (useful for tests)."""
    if isinstance(table, pa.Table):
        table = table.combine_chunks()
        batches = table.to_batches(max_chunksize=batch_rows)
        reader = pa.RecordBatchReader.from_batches(table.schema, batches)
    else:
        reader = _as_reader(table)
    return scan_arrow(reader)
