"""DuckDB <-> ArrowMetal bridge: run the SQL in DuckDB, run the heavy compute on the GPU.

DuckDB hands results out over the Arrow C Data / C Stream interfaces, and ArrowMetal imports Arrow
buffers into Metal without copying them (Apple silicon has one physical memory pool, so a Metal
buffer can wrap DuckDB's own pages). The two therefore meet at zero cost: a 40 MB int64 column
crosses in under a millisecond and the pointer on the far side is the same pointer.

    import duckdb, arrowmetal as am
    con = duckdb.connect()
    cols = am.from_duckdb(con.sql("select k, v from t where v > 0"))   # dict of MetalArray
    gb = am.group_by([cols["k"]])
    totals = gb.sum(cols["v"]).to_arrow()

Four entry points, in rising order of convenience:

  * ``am.from_duckdb(source)``     - pull a relation / SQL / Arrow stream in as ``{name: MetalArray}``
  * ``am.to_duckdb(con, name, x)`` - register GPU output back as a DuckDB view (zero-copy)
  * ``am.duckdb_gpu_query(con, sql, then=...)`` - do both around one callable, get a relation back
  * ``am.duckdb_batches(...)`` / ``am.duckdb_aggregate(...)`` / ``am.duckdb_group_by(...)``
    - stream a table that does not fit in memory through the GPU one record batch at a time

Nothing here holds a DuckDB lock while the GPU runs, and nothing here copies a column that DuckDB
was willing to hand over as a plain Arrow buffer.
"""
import pyarrow as pa

from . import ArrowMetalError, MetalArray, group_by as _group_by

__all__ = ["from_duckdb", "to_duckdb", "duckdb_gpu_query", "duckdb_batches", "duckdb_aggregate",
           "duckdb_group_by", "duckdb_reader", "duckdb_table", "is_zero_copy",
           "DEFAULT_ROWS_PER_BATCH"]

#: Rows per record batch when streaming. DuckDB's own vector size is 2048; a few hundred thousand
#: rows per batch keeps the GPU busy without materialising the whole table.
DEFAULT_ROWS_PER_BATCH = 1 << 20


# ---------------------------------------------------------------------------------------------------
# Pulling out of DuckDB
# ---------------------------------------------------------------------------------------------------

def _is_relation(obj):
    return hasattr(obj, "to_arrow_table") or hasattr(obj, "fetch_arrow_table")


def _resolve(source, con=None):
    """Normalise the many things a caller may pass into a DuckDB relation (or a pyarrow object).

    Accepts a ``DuckDBPyRelation``, a SQL string plus ``con``, a ``DuckDBPyConnection`` that has a
    pending result, a ``pyarrow.Table`` / ``RecordBatchReader``, or anything exposing
    ``__arrow_c_stream__``.
    """
    if isinstance(source, str):
        if con is None:
            raise ArrowMetalError("a SQL string needs a connection: from_duckdb(sql, con=con)")
        return con.sql(source)
    if isinstance(source, (pa.Table, pa.RecordBatch, pa.RecordBatchReader)):
        return source
    if _is_relation(source):
        return source
    # A DuckDBPyConnection after .execute(...): it answers the same fetch methods.
    if hasattr(source, "fetch_record_batch") or hasattr(source, "__arrow_c_stream__"):
        return source
    raise ArrowMetalError("from_duckdb needs a DuckDB relation, a SQL string with con=, a "
                          "connection with a pending result, or an Arrow table/stream; got "
                          + type(source).__name__)


def duckdb_table(source, con=None):
    """The result of `source` as a ``pyarrow.Table``, without touching the GPU.

    Uses DuckDB's Arrow export, which is the C Data interface underneath, so the table's buffers are
    DuckDB's own where DuckDB can share them."""
    rel = _resolve(source, con)
    if isinstance(rel, pa.Table):
        return rel
    if isinstance(rel, pa.RecordBatch):
        return pa.Table.from_batches([rel])
    if isinstance(rel, pa.RecordBatchReader):
        return rel.read_all()
    for name in ("to_arrow_table", "fetch_arrow_table", "arrow"):
        fn = getattr(rel, name, None)
        if fn is not None:
            return fn()
    if hasattr(rel, "__arrow_c_stream__"):
        return pa.table(rel)
    raise ArrowMetalError("this object does not expose an Arrow result")


def duckdb_reader(source, con=None, rows_per_batch=DEFAULT_ROWS_PER_BATCH):
    """A ``pyarrow.RecordBatchReader`` over `source`, streamed `rows_per_batch` rows at a time.

    This is DuckDB's ``fetch_record_batch``: an Arrow C Stream that produces batches lazily, so the
    whole result never has to exist at once."""
    rel = _resolve(source, con)
    if isinstance(rel, pa.RecordBatchReader):
        return rel
    if isinstance(rel, pa.Table):
        return pa.RecordBatchReader.from_batches(rel.schema, rel.to_batches(rows_per_batch))
    if isinstance(rel, pa.RecordBatch):
        return pa.RecordBatchReader.from_batches(rel.schema, [rel])
    for name in ("fetch_record_batch", "fetch_arrow_reader", "to_arrow_reader"):
        fn = getattr(rel, name, None)
        if fn is None:
            continue
        try:
            return fn(rows_per_batch)
        except TypeError:
            return fn()
    if hasattr(rel, "__arrow_c_stream__"):
        return pa.RecordBatchReader.from_stream(rel)
    raise ArrowMetalError("this object cannot stream record batches")


def _flatten(column):
    """One ``pyarrow.Array`` from a Table column, which may be a ChunkedArray of several chunks."""
    if isinstance(column, pa.ChunkedArray):
        if column.num_chunks == 1:
            return column.chunk(0)
        if column.num_chunks == 0:
            return pa.array([], column.type)
        return pa.concat_arrays(list(column.chunks))
    return column


def _lift(array, on_unsupported):
    """A pyarrow array as a MetalArray, or the array itself when its type will not import."""
    try:
        return MetalArray.from_arrow(array)
    except Exception:
        if on_unsupported == "keep":
            return array
        raise


def from_duckdb(source, con=None, on_unsupported="keep", columns=None):
    """Pull a DuckDB result onto the GPU: ``{column_name: MetalArray}``.

    `source` is a ``DuckDBPyRelation`` (``con.sql(...)``, ``con.table(...)``, ``con.read_parquet(...)``),
    a SQL string with ``con=``, a connection holding a pending result, or any Arrow table/stream.

    The import is zero-copy for every fixed-width, string, binary, decimal and temporal column DuckDB
    produces: the Metal buffer wraps DuckDB's Arrow pages in place (see ``is_zero_copy``). Nested
    columns (list, struct, map) import too, though only the structural kernels operate on them.

    `on_unsupported` says what to do with a column ArrowMetal cannot lift - ``"keep"`` (the default)
    leaves it as a ``pyarrow.Array`` in the same dict, ``"raise"`` propagates the error. `columns`
    restricts which columns are pulled at all.

    The returned dict is exactly what ``am.query`` wants as its data argument, so a fused query over
    a DuckDB result is ``am.query(am.from_duckdb(rel), q)``.
    """
    table = duckdb_table(source, con)
    names = list(table.column_names) if columns is None else list(columns)
    out = {}
    for name in names:
        out[name] = _lift(_flatten(table.column(name)), on_unsupported)
    return out


def is_zero_copy(array, metal=None):
    """True when lifting `array` onto the GPU shared its buffers rather than copying them.

    Compares the address of the value buffer before and after the Metal round trip. Handy for
    proving what the bridge costs; the benchmark uses it, and so does the test suite."""
    if not isinstance(array, pa.Array):
        array = _flatten(array)
    metal = metal if metal is not None else MetalArray.from_arrow(array)
    before = [b.address for b in array.buffers() if b is not None]
    after = [b.address for b in metal.to_arrow().buffers() if b is not None]
    if not before or not after:
        return False
    return bool(set(before) & set(after))


# ---------------------------------------------------------------------------------------------------
# Pushing back into DuckDB
# ---------------------------------------------------------------------------------------------------

def _as_arrow(value):
    """A pyarrow.Array from a MetalArray, a pyarrow array, or a plain Python sequence."""
    if isinstance(value, MetalArray):
        return value.to_arrow()
    if isinstance(value, pa.ChunkedArray):
        return _flatten(value)
    if isinstance(value, pa.Array):
        return value
    return pa.array(value)


def arrow_table(arrays, names=None):
    """A ``pyarrow.Table`` from ArrowMetal output: a dict, a list of arrays plus `names`, a Table.

    Scalars (what ``am.query`` returns for an aggregate terminal) become one-row columns, so a GPU
    aggregate can be registered as a DuckDB view just like a projection."""
    if isinstance(arrays, pa.Table):
        return arrays
    if isinstance(arrays, pa.RecordBatch):
        return pa.Table.from_batches([arrays])
    if isinstance(arrays, dict):
        items = list(arrays.items())
    else:
        seq = list(arrays)
        if names is None:
            names = [f"col{i}" for i in range(len(seq))]
        items = list(zip(names, seq))
    cols, out_names = [], []
    for name, value in items:
        if isinstance(value, (int, float, bool)) or value is None:
            cols.append(pa.array([value]))
        else:
            cols.append(_as_arrow(value))
        out_names.append(name)
    lengths = {len(c) for c in cols}
    if len(lengths) > 1:
        raise ArrowMetalError(f"columns have different lengths: {sorted(lengths)}")
    return pa.Table.from_arrays(cols, names=out_names)


def to_duckdb(con, name, arrays, names=None):
    """Register ArrowMetal output with `con` under `name` and return the DuckDB relation.

    `arrays` is a dict of ``{name: MetalArray}``, a list of arrays with `names`, a ``pyarrow.Table``,
    or the dict/scalar that ``am.query`` returned. Registration goes through ``con.register``, which
    takes the Arrow table over the C Data interface - DuckDB reads the GPU-produced buffers in place,
    with no copy and no serialisation.

    The view stays registered until ``con.unregister(name)``; SQL can join against it immediately.
    """
    if not isinstance(arrays, (dict, list, tuple, pa.Table, pa.RecordBatch)):
        arrays = {name: arrays}                      # a bare scalar or array
    table = arrow_table(arrays, names)
    con.register(name, table)
    return con.table(name)


def duckdb_gpu_query(con, sql, then=None, name="arrowmetal_result", into=None):
    """Run `sql` in DuckDB, hand the result to `then` on the GPU, register what comes back.

    The division of labour the bridge exists for: DuckDB does the scan, the projection pushdown, the
    predicate pushdown and the joins - the things a query engine with statistics and a Parquet reader
    is good at - and ArrowMetal does the aggregate, the filter, the sort or the group-by that follows,
    on thousands of GPU threads.

        rel = am.duckdb_gpu_query(
            con,
            "select o.customer, o.amount from orders o join customers c on c.id = o.customer",
            then=lambda cols: {"customer": am.group_by([cols["customer"]]).keys()[0],
                               "total": am.group_by([cols["customer"]]).sum(cols["amount"])},
        )
        rel.order("total desc").limit(10).show()

    `then` receives ``{name: MetalArray}`` and returns anything ``to_duckdb`` accepts - a dict of
    MetalArrays, a ``pyarrow.Table``, or the result of ``am.query``. With `then` left as None the
    relation is simply round-tripped through the GPU, which is a useful sanity check and not much
    else. `into` names a different connection to register the result on.
    """
    columns = from_duckdb(sql, con)
    result = columns if then is None else then(columns)
    if not isinstance(result, (dict, list, tuple, pa.Table, pa.RecordBatch)):
        result = {name: result}
    return to_duckdb(into if into is not None else con, name, result)


# ---------------------------------------------------------------------------------------------------
# Streaming: tables larger than memory
# ---------------------------------------------------------------------------------------------------

def duckdb_batches(source, con=None, rows_per_batch=DEFAULT_ROWS_PER_BATCH, on_unsupported="keep",
                   columns=None):
    """Iterate a DuckDB result as ``{name: MetalArray}``, one record batch at a time.

    Only one batch is ever resident, so a table far larger than memory - or than the GPU's working
    set - can be pushed through the same kernels. Each batch is imported zero-copy and released as
    soon as the loop moves on.

        total = 0
        for cols in am.duckdb_batches(con.sql("select amount from big_parquet"), rows_per_batch=1 << 20):
            with am.batch():
                total += cols["amount"].sum()
    """
    reader = duckdb_reader(source, con, rows_per_batch)
    for record_batch in reader:
        names = list(record_batch.schema.names) if columns is None else list(columns)
        yield {name: _lift(record_batch.column(record_batch.schema.get_field_index(name)),
                           on_unsupported) for name in names}


# Which streaming aggregates are exact when merged from per-batch partials, and how they merge.
#
#   sum, count, min, max   - exact: the merge is the same associative operator as the aggregate
#   mean                   - exact up to float64 rounding: sum/count over the whole stream, not a
#                            mean of means (the classic wrong answer for unequal batches)
#   count_distinct, unique - exact, but the host keeps every distinct value seen, so memory is
#                            proportional to cardinality rather than to the batch
#
# Anything order- or distribution-dependent (median, quantile, stddev, tdigest, mode) is NOT
# offered here: it cannot be merged exactly from independent partials, and this module will not
# quietly return an approximation.
_EXACT_STREAMING = ("sum", "count", "min", "max", "mean", "count_distinct")


def _merge_scalar(op, acc, value):
    if value is None:
        return acc
    if acc is None:
        return value
    if op == "sum" or op == "count":
        return acc + value
    if op == "min":
        return min(acc, value)
    if op == "max":
        return max(acc, value)
    raise ArrowMetalError(f"no exact host merge for {op!r}")


def duckdb_aggregate(source, aggs, con=None, rows_per_batch=DEFAULT_ROWS_PER_BATCH):
    """Aggregate a DuckDB result on the GPU one batch at a time, merging partials on the host.

    `aggs` maps an output name to ``(op, column)``, or is a list of ``(op, column)`` pairs named
    ``op_column``. Supported ops are sum, count, min, max, mean and count_distinct - the ones whose
    per-batch partials merge to an *exact* whole-table answer. sum/count/min/max merge by their own
    operator; mean is computed as total sum over total count, never as a mean of batch means;
    count_distinct unions the batches' distinct values on the host, so it costs memory proportional
    to the cardinality rather than to the batch.

    Order- and distribution-dependent aggregates (median, quantile, stddev, mode, tdigest) are
    deliberately absent: they have no exact merge, and returning an approximation under the same name
    as the exact aggregate would be a lie. Compute those over a whole batch, or over a sample.

        am.duckdb_aggregate(con.sql("select amount from big"), {"total": ("sum", "amount")})
    """
    if isinstance(aggs, (list, tuple)):
        aggs = {f"{op}_{col}": (op, col) for op, col in aggs}
    for name, (op, _col) in aggs.items():
        if op not in _EXACT_STREAMING:
            raise ArrowMetalError(
                f"{op!r} has no exact merge across batches; streaming supports {_EXACT_STREAMING}")
    needed = sorted({col for _op, col in aggs.values()})
    state = {name: None for name in aggs}
    sums = {name: None for name in aggs}
    counts = {name: 0 for name in aggs}
    distinct = {name: set() for name in aggs if aggs[name][0] == "count_distinct"}

    for cols in duckdb_batches(source, con, rows_per_batch, columns=needed):
        for name, (op, col) in aggs.items():
            column = cols[col]
            if not isinstance(column, MetalArray):
                raise ArrowMetalError(f"column {col!r} did not import onto the GPU")
            if op == "count":
                state[name] = (state[name] or 0) + (len(column) - column.null_count)
            elif op == "mean":
                sums[name] = _merge_scalar("sum", sums[name], column.sum())
                counts[name] += len(column) - column.null_count
            elif op == "count_distinct":
                distinct[name].update(v for v in column.unique().to_arrow().to_pylist() if v is not None)
            else:
                state[name] = _merge_scalar(op, state[name], getattr(column, op)())

    out = {}
    for name, (op, _col) in aggs.items():
        if op == "mean":
            out[name] = None if not counts[name] else sums[name] / counts[name]
        elif op == "count_distinct":
            out[name] = len(distinct[name])
        else:
            out[name] = state[name]
    return out


def duckdb_group_by(source, keys, aggs, con=None, rows_per_batch=DEFAULT_ROWS_PER_BATCH):
    """Group-by over a stream: a GPU group-by per batch, partials merged on the host by key.

    `keys` is a column name or a list of them; `aggs` maps an output name to ``(op, column)`` with
    op in sum / count / min / max / mean. Every one of those is exact: the per-batch group-by is the
    GPU hash aggregate, and merging two partials for the same key is the same associative operator
    again (mean once more being total sum over total count).

    Returns a ``pyarrow.Table`` with one row per distinct key combination, ordered by first
    appearance. Host memory is proportional to the number of distinct keys, not to the table.

        am.duckdb_group_by(rel, "region", {"total": ("sum", "amount"), "n": ("count", "amount")})
    """
    if isinstance(keys, str):
        keys = [keys]
    keys = list(keys)
    for name, (op, _col) in aggs.items():
        if op not in ("sum", "count", "min", "max", "mean"):
            raise ArrowMetalError(f"{op!r} has no exact per-key merge across batches")
    value_columns = sorted({col for _op, col in aggs.values()})
    needed = keys + [c for c in value_columns if c not in keys]

    acc = {}           # key tuple -> {output name: partial}
    order = []
    for cols in duckdb_batches(source, con, rows_per_batch, columns=needed):
        gb = _group_by([cols[k] for k in keys])
        key_arrays = [k.to_pylist() for k in gb.keys()]
        per_agg = {}
        for name, (op, col) in aggs.items():
            if op == "mean":
                per_agg[name] = (gb.sum(cols[col]).to_arrow().to_pylist(),
                                 gb.count(cols[col]).to_arrow().to_pylist())
            elif op == "count":
                per_agg[name] = gb.count(cols[col]).to_arrow().to_pylist()
            else:
                per_agg[name] = getattr(gb, op)(cols[col]).to_arrow().to_pylist()
        for row in range(len(gb)):
            key = tuple(ka[row] for ka in key_arrays)
            slot = acc.get(key)
            if slot is None:
                slot = acc[key] = {}
                order.append(key)
            for name, (op, _col) in aggs.items():
                if op == "mean":
                    s, c = per_agg[name]
                    prev = slot.get(name, (None, 0))
                    slot[name] = (_merge_scalar("sum", prev[0], s[row]), prev[1] + (c[row] or 0))
                else:
                    slot[name] = _merge_scalar("count" if op == "count" else op,
                                               slot.get(name), per_agg[name][row])

    key_columns = [[k[i] for k in order] for i in range(len(keys))]
    out_names = list(keys)
    out_columns = [pa.array(c) for c in key_columns]
    for name, (op, _col) in aggs.items():
        if op == "mean":
            values = [None if not acc[k][name][1] else acc[k][name][0] / acc[k][name][1] for k in order]
        else:
            values = [acc[k].get(name) for k in order]
        out_names.append(name)
        out_columns.append(pa.array(values))
    return pa.Table.from_arrays(out_columns, names=out_names)
