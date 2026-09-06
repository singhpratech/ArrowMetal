"""Out-of-core streaming: ArrowMetal's Stream API vs Polars, DuckDB and pyarrow on a dataset that
does not fit in RAM.

Generates a multi-gigabyte Arrow IPC dataset on disk (8 columns, one file per ~1 GB), then runs seven
whole-dataset queries against every engine and reports wall time, CPU time, disk throughput and peak
RSS. The point of the benchmark is what happens when the data is larger than memory: an engine that
finishes at a flat few hundred MB of RSS is streaming, one whose RSS tracks the dataset is not, and
one that is killed is recorded as such rather than dropped.

    PYTHONPATH=python python Benchmarks/streaming_bench.py --data-dir /Volumes/scratch/ambench
    PYTHONPATH=python python Benchmarks/streaming_bench.py --data-dir DIR --size-gb 8 --reuse --keep

Every (workload, engine) measurement runs in its own subprocess, so peak RSS is that engine's alone
and an out-of-memory kill costs one cell of the matrix instead of the run. A cell that raises, times
out or is killed is reported with the reason.

Correctness is cross-checked: ArrowMetal's answer for filter_sum, groupby_1k, topk and sort_limit is
compared against the reference engine (Polars, or pyarrow when Polars is unavailable) with a 1e-9
relative tolerance on float sums, and the approximate distinct count is reported against the true
cardinality, which the generator knows exactly.

Output is a Markdown table per workload on stdout, plus --json for the raw numbers.
"""
import argparse
import glob
import json
import os
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

SEED = 20260906

# 8 columns: int64 + int32 + int64 + ~12-byte utf8 + float64 + int32 + int64 + int8.
BYTES_PER_ROW = 8 + 4 + 8 + 12 + 8 + 4 + 8 + 1

LABELS = 50                 # distinct values in `label`
REGIONS = 1000              # distinct values in `region`, and the dense group-by key space
BIGKEYS = 10_000_000        # distinct values in `bigkey`
DIM_ROWS = 1000             # the broadcast join's dimension table
JOIN_CUTOFF = 50            # broadcast_join keeps region < 50, so the joined result stays streamable
TOPK = 100
SORT_LIMIT = 1000
HLL_PRECISION = 14
SAMPLE_REGIONS = (0, 1, 7, 42, 500, 999)   # the groups the cross-check compares value by value

WORKLOADS = ["filter_sum", "groupby_1k", "groupby_10m", "topk", "count_distinct",
             "sort_limit", "broadcast_join"]
ENGINES = ["arrowmetal", "polars", "duckdb", "pyarrow"]
CHECKED = ("filter_sum", "groupby_1k", "topk", "sort_limit")

WORKLOAD_DESC = {
    "filter_sum": "sum(amount) where region < 100",
    "groupby_1k": "group by region -> sum(amount), count  (1000 groups)",
    "groupby_10m": "group by bigkey -> sum(amount)  (10M groups)",
    "topk": f"top {TOPK} rows by amount",
    "count_distinct": "distinct count of id (approximate where the engine has a sketch)",
    "sort_limit": f"order by amount desc limit {SORT_LIMIT}",
    "broadcast_join": f"join a {DIM_ROWS}-row dimension on region (region < {JOIN_CUTOFF}), then sum",
}

RESULT_MARK = "__RESULT__ "


# ---------------------------------------------------------------- measurement


def maxrss_mb(usage):
    # ru_maxrss is BYTES on macOS and kilobytes on Linux; both are converted to MB here.
    return usage.ru_maxrss / 1e6 if sys.platform == "darwin" else usage.ru_maxrss / 1e3


def measure(fn):
    """Run one engine callable and return the timing record for it."""
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    c0, t0 = time.process_time(), time.perf_counter()
    value, extra = fn()
    wall = time.perf_counter() - t0
    cpu = time.process_time() - c0
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    rusage_cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    return dict(status="ok", wall_ms=wall * 1000.0, cpu_ms=cpu * 1000.0,
                rusage_cpu_ms=rusage_cpu * 1000.0, peak_rss_mb=maxrss_mb(r1),
                value=value, extra=extra or {})


# ---------------------------------------------------------------- dataset


def part_files(data_dir):
    return sorted(glob.glob(os.path.join(data_dir, "part-*.arrows")))


def manifest_path(data_dir):
    return os.path.join(data_dir, "manifest.json")


def open_ipc(path):
    """A reader for one part file. The format is taken from the magic bytes rather than assumed, so
    a dataset written with --ipc-format file still reads back here."""
    with open(path, "rb") as fh:
        magic = fh.read(6)
    src = pa.memory_map(path, "rb")
    return pa.ipc.open_file(src) if magic == b"ARROW1" else pa.ipc.open_stream(src)


def iter_batches(files):
    for path in files:
        reader = open_ipc(path)
        if hasattr(reader, "num_record_batches"):
            for i in range(reader.num_record_batches):
                yield reader.get_batch(i)
        else:
            for batch in reader:
                yield batch


def dataset_schema(files):
    return open_ipc(files[0]).schema


def batch_reader(files):
    """One RecordBatchReader over the whole dataset, for engines that consume a stream directly."""
    return pa.RecordBatchReader.from_batches(dataset_schema(files), iter_batches(files))


def make_batch(index, rows, id_start, schema):
    """One deterministic record batch. Seeded per batch so generation stays chunked and repeatable."""
    rng = np.random.default_rng(SEED + index)
    ids = np.arange(id_start, id_start + rows, dtype=np.int64)
    region = rng.integers(0, REGIONS, size=rows, dtype=np.int32)
    bigkey = rng.integers(0, BIGKEYS, size=rows, dtype=np.int64)
    codes = rng.integers(0, LABELS, size=rows, dtype=np.int32)
    vocab = pa.array([f"label_{i:02d}" for i in range(LABELS)])
    label = pa.DictionaryArray.from_arrays(pa.array(codes), vocab).cast(pa.string())
    amount = rng.random(rows) * 1000.0
    qty = rng.integers(0, 100, size=rows, dtype=np.int32)
    ts = 1_600_000_000 + rng.integers(0, 200_000_000, size=rows, dtype=np.int64)
    flag = (rng.random(rows) < 0.5).astype(np.int8)
    return pa.record_batch([pa.array(ids), pa.array(region), pa.array(bigkey), label,
                            pa.array(amount), pa.array(qty), pa.array(ts), pa.array(flag)],
                           schema=schema)


def target_bytes(args, data_dir):
    """The dataset size actually used: the request, capped by the disk budget and by 80% of free
    space, because a benchmark that fills the volume takes the machine down with it."""
    probe = data_dir if os.path.isdir(data_dir) else os.path.dirname(os.path.abspath(data_dir))
    free = shutil.disk_usage(probe or ".").free
    wanted = args.size_gb * 1e9
    capped = min(wanted, args.disk_budget_gb * 1e9, free * 0.80)
    return capped, wanted, free


def generate(args, data_dir):
    """Write the dataset as part-NNNN.arrows, one file per ~1 GB. Returns the manifest."""
    capped, wanted, free = target_bytes(args, data_dir)
    batch_bytes = args.batch_rows * BYTES_PER_ROW
    batches_per_file = max(1, int(round(1e9 / batch_bytes)))
    file_rows = batches_per_file * args.batch_rows
    n_files = max(1, int(round(capped / (file_rows * BYTES_PER_ROW))))
    total_rows = n_files * file_rows
    print(f"dataset: requested {wanted / 1e9:.1f} GB, free {free / 1e9:.1f} GB, "
          f"budget {args.disk_budget_gb:.1f} GB -> generating {total_rows * BYTES_PER_ROW / 1e9:.1f} GB "
          f"({total_rows:,} rows, {n_files} files x {batches_per_file} batches x {args.batch_rows:,})")

    schema = pa.schema([("id", pa.int64()), ("region", pa.int32()), ("bigkey", pa.int64()),
                        ("label", pa.string()), ("amount", pa.float64()), ("qty", pa.int32()),
                        ("ts", pa.int64()), ("flag", pa.int8())])
    os.makedirs(data_dir, exist_ok=True)
    for stale in part_files(data_dir):
        os.remove(stale)

    t0 = time.perf_counter()
    index = 0
    for f in range(n_files):
        path = os.path.join(data_dir, f"part-{f:04d}.arrows")
        with pa.OSFile(path, "wb") as sink:
            new = pa.ipc.new_file if args.ipc_format == "file" else pa.ipc.new_stream
            with new(sink, schema) as writer:
                for _ in range(batches_per_file):
                    writer.write_batch(make_batch(index, args.batch_rows,
                                                  index * args.batch_rows, schema))
                    index += 1
        print(f"  wrote {os.path.basename(path)} "
              f"({os.path.getsize(path) / 1e9:.2f} GB, {time.perf_counter() - t0:.0f}s elapsed)")

    files = part_files(data_dir)
    manifest = dict(rows=total_rows, files=len(files), batch_rows=args.batch_rows,
                    bytes=sum(os.path.getsize(p) for p in files), ipc_format=args.ipc_format,
                    seed=SEED, distinct_ids=total_rows)
    with open(manifest_path(data_dir), "w") as fh:
        json.dump(manifest, fh)
    print(f"dataset ready: {manifest['bytes'] / 1e9:.2f} GB on disk in {len(files)} files, "
          f"{time.perf_counter() - t0:.0f}s\n")
    return manifest


def load_manifest(data_dir):
    with open(manifest_path(data_dir)) as fh:
        return json.load(fh)


def can_reuse(args, data_dir):
    """Reuse only when the manifest matches what is on disk; a partial dataset is regenerated."""
    if not os.path.exists(manifest_path(data_dir)):
        return None
    try:
        manifest = load_manifest(data_dir)
    except (OSError, ValueError):
        return None
    files = part_files(data_dir)
    if len(files) != manifest.get("files") or not files:
        return None
    if manifest.get("batch_rows") != args.batch_rows:
        return None
    return manifest


def remove_dataset(data_dir):
    for path in part_files(data_dir):
        os.remove(path)
    if os.path.exists(manifest_path(data_dir)):
        os.remove(manifest_path(data_dir))
    try:
        os.rmdir(data_dir)
    except OSError:
        pass                                  # the directory was not ours to remove; leave it


def dim_table():
    """The broadcast join's right side: one row per region, with a weight to prove the join attached
    the right-hand values rather than merely counting matches."""
    rng = np.random.default_rng(SEED + 99)
    return pa.table({"region": pa.array(np.arange(DIM_ROWS, dtype=np.int32)),
                     "weight": pa.array(rng.random(DIM_ROWS))})


# ---------------------------------------------------------------- result shapes


def num(v):
    return None if v is None else float(v)


def group_summary(table, key, sample=SAMPLE_REGIONS, value_cols=("total", "n")):
    """A comparable digest of a group-by result: group count, the sum of the totals, and the exact
    values of a handful of named groups. Comparing whole 10M-row results across engines would cost
    more than the query."""
    cols = [c for c in value_cols if c in table.column_names]
    out = {"groups": table.num_rows}
    if "total" in table.column_names:
        out["total_sum"] = num(pc.sum(table["total"]).as_py())
    if sample is None:
        return out
    keys = table[key].to_pylist()
    values = {c: table[c].to_pylist() for c in cols}
    index = {k: i for i, k in enumerate(keys)}
    out["sample"] = {str(s): [num(values[c][index[s]]) for c in cols]
                     for s in sample if s in index}
    return out


def amounts(table, k):
    """The `amount` column of a top-k / sorted-limit result, largest first, so engines that return a
    different row order still compare equal."""
    values = sorted((v for v in table["amount"].to_pylist() if v is not None), reverse=True)
    return [float(v) for v in values[:k]]


def rename(table, mapping):
    return pa.table({new: table[old] for old, new in mapping.items()})


def compare(reference, candidate, tol=1e-9):
    """Recursive PASS/FAIL over the result digests. Returns None when both sides agree, otherwise the
    first disagreement as text."""
    if isinstance(reference, dict):
        if not isinstance(candidate, dict):
            return f"expected an object, got {type(candidate).__name__}"
        for key in reference:
            if key not in candidate:
                return f"missing {key}"
            bad = compare(reference[key], candidate[key], tol)
            if bad:
                return f"{key}: {bad}"
        return None
    if isinstance(reference, (list, tuple)):
        if not isinstance(candidate, (list, tuple)):
            return f"expected a list, got {type(candidate).__name__}"
        if len(reference) != len(candidate):
            return f"length {len(candidate)} != {len(reference)}"
        for i, (a, b) in enumerate(zip(reference, candidate)):
            bad = compare(a, b, tol)
            if bad:
                return f"[{i}] {bad}"
        return None
    if isinstance(reference, bool) or reference is None:
        return None if reference == candidate else f"{candidate!r} != {reference!r}"
    if isinstance(reference, (int, float)):
        if candidate is None:
            return "null"
        scale = max(abs(float(reference)), abs(float(candidate)), 1e-300)
        if abs(float(reference) - float(candidate)) <= tol * scale:
            return None
        return f"{candidate!r} != {reference!r}"
    return None if reference == candidate else f"{candidate!r} != {reference!r}"


# ---------------------------------------------------------------- ArrowMetal


class Ctx:
    def __init__(self, data_dir, manifest, args):
        self.data_dir = data_dir
        self.manifest = manifest
        self.rows = manifest["rows"]
        self.bytes = manifest["bytes"]
        self.files = part_files(data_dir)
        self.prefetch = args.prefetch


def am_stream(ctx):
    import arrowmetal as am
    return am, am.scan_ipc(ctx.data_dir, prefetch=ctx.prefetch)


def am_stats(stream):
    try:
        stats = dict(stream.stats or {})
    except Exception:                          # stats are diagnostics; never fail a run over them
        return {}
    return {k: v for k, v in stats.items() if isinstance(v, (int, float, str, bool))}


def am_table(result):
    """The streaming API returns a pyarrow.Table for the small terminals; anything else is drained
    through to_reader() so a sink-shaped result is still comparable."""
    if isinstance(result, pa.Table):
        return result
    if hasattr(result, "to_reader"):
        return pa.Table.from_batches(list(result.to_reader()))
    if isinstance(result, pa.RecordBatchReader):
        return pa.Table.from_batches(list(result))
    raise TypeError(f"unexpected streaming result {type(result).__name__}")


def am_filter_sum(ctx):
    am, s = am_stream(ctx)
    total = s.filter(am.col("region") < 100).sum("amount")
    return num(total), am_stats(s)


def am_groupby_1k(ctx):
    _am, s = am_stream(ctx)
    table = am_table(s.group_by("region", dense_key_count=REGIONS)
                     .agg([("sum", "amount", "total"), ("count", None, "n")]))
    return group_summary(table, "region"), am_stats(s)


def am_groupby_10m(ctx):
    _am, s = am_stream(ctx)
    table = am_table(s.group_by(["bigkey"]).agg([("sum", "amount", "total")]))
    return group_summary(table, "bigkey", sample=None), am_stats(s)


def am_topk(ctx):
    _am, s = am_stream(ctx)
    return amounts(am_table(s.top_k("amount", TOPK, largest=True)), TOPK), am_stats(s)


def am_count_distinct(ctx):
    _am, s = am_stream(ctx)
    value = int(s.count_distinct_approx("id", precision=HLL_PRECISION))
    stats = am_stats(s)
    stats["exact"] = False
    return value, stats


def am_sort_limit(ctx):
    _am, s = am_stream(ctx)
    # (column, descending): the sort keys follow Polars' convention, largest first here.
    table = am_table(s.sort([("amount", True)], limit=SORT_LIMIT))
    return amounts(table, SORT_LIMIT), am_stats(s)


def am_broadcast_join(ctx):
    am, s = am_stream(ctx)
    joined = s.filter(am.col("region") < JOIN_CUTOFF).join(
        dim_table(), on="region", right_on="region", how="inner", broadcast=True)
    if isinstance(joined, pa.Table):
        total = num(pc.sum(joined["amount"]).as_py())
        weight = num(pc.sum(joined["weight"]).as_py())
    else:
        total = weight = 0.0
        reader = joined.to_reader() if hasattr(joined, "to_reader") else joined
        for batch in reader:                   # a sink-shaped join result is folded in one pass
            total += float(pc.sum(batch["amount"]).as_py() or 0.0)
            weight += float(pc.sum(batch["weight"]).as_py() or 0.0)
    return [total, weight], am_stats(s)


AM = dict(filter_sum=am_filter_sum, groupby_1k=am_groupby_1k, groupby_10m=am_groupby_10m,
          topk=am_topk, count_distinct=am_count_distinct, sort_limit=am_sort_limit,
          broadcast_join=am_broadcast_join)


# ---------------------------------------------------------------- Polars


def polars_lazy(ctx):
    """A streaming LazyFrame over the dataset. scan_ipc is the fast path; it wants the IPC file
    format, so a stream-format dataset falls back to a pyarrow dataset scan."""
    import polars as pl
    pattern = os.path.join(ctx.data_dir, "part-*.arrows")
    try:
        lf = pl.scan_ipc(pattern)
        lf.collect_schema()                    # forces the scan to open a file, so a format
        return lf                              # mismatch surfaces here rather than mid-query
    except Exception as exc:
        first = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:160]
    try:
        import pyarrow.dataset as pads
        lf = pl.scan_pyarrow_dataset(pads.dataset(ctx.files, format="ipc"))
        lf.collect_schema()
        return lf
    except Exception as exc:
        second = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:160]
    raise RuntimeError(f"polars cannot scan this dataset: scan_ipc -> {first}; "
                       f"scan_pyarrow_dataset -> {second}")


def pl_collect(lf):
    """Polars renamed the streaming switch; the new keyword first, the old one for older versions."""
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)


def pl_filter_sum(ctx):
    import polars as pl
    lf = polars_lazy(ctx)
    out = pl_collect(lf.filter(pl.col("region") < 100).select(pl.col("amount").sum()))
    return num(out.item()), {}


def pl_groupby_1k(ctx):
    import polars as pl
    lf = polars_lazy(ctx)
    out = pl_collect(lf.group_by("region").agg(pl.col("amount").sum().alias("total"),
                                               pl.len().alias("n")))
    return group_summary(out.to_arrow(), "region"), {}


def pl_groupby_10m(ctx):
    import polars as pl
    lf = polars_lazy(ctx)
    out = pl_collect(lf.group_by("bigkey").agg(pl.col("amount").sum().alias("total")))
    return group_summary(out.to_arrow(), "bigkey", sample=None), {}


def pl_topk(ctx):
    import polars as pl
    lf = polars_lazy(ctx).select("amount")
    try:
        out = pl_collect(lf.top_k(TOPK, by="amount"))
    except (TypeError, AttributeError):        # top_k's signature moved across 0.20/1.x
        out = pl_collect(lf.sort("amount", descending=True).head(TOPK))
    return amounts(out.to_arrow(), TOPK), {}


def pl_count_distinct(ctx):
    import polars as pl
    out = pl_collect(polars_lazy(ctx).select(pl.col("id").n_unique()))
    return int(out.item()), {"exact": True}


def pl_sort_limit(ctx):
    lf = polars_lazy(ctx).select("amount").sort("amount", descending=True).head(SORT_LIMIT)
    return amounts(pl_collect(lf).to_arrow(), SORT_LIMIT), {}


def pl_broadcast_join(ctx):
    import polars as pl
    dim = pl.from_arrow(dim_table()).lazy()
    lf = (polars_lazy(ctx).filter(pl.col("region") < JOIN_CUTOFF)
          .join(dim, on="region", how="inner")
          .select(pl.col("amount").sum().alias("total"), pl.col("weight").sum().alias("weight")))
    out = pl_collect(lf)
    return [num(out["total"][0]), num(out["weight"][0])], {}


POLARS = dict(filter_sum=pl_filter_sum, groupby_1k=pl_groupby_1k, groupby_10m=pl_groupby_10m,
              topk=pl_topk, count_distinct=pl_count_distinct, sort_limit=pl_sort_limit,
              broadcast_join=pl_broadcast_join)


# ---------------------------------------------------------------- DuckDB


def duck_relation(ctx):
    """A DuckDB connection with the dataset registered as `data`. DuckDB has no IPC reader in every
    build, so this tries read_ipc, then a pyarrow dataset, then a plain RecordBatchReader."""
    import duckdb
    con = duckdb.connect()
    notes = []
    if hasattr(duckdb, "read_ipc"):
        try:
            con.register("data", duckdb.read_ipc(os.path.join(ctx.data_dir, "part-*.arrows")))
            return con, "read_ipc"
        except Exception as exc:
            notes.append(f"read_ipc: {type(exc).__name__}")
    try:
        import pyarrow.dataset as pads
        dataset = pads.dataset(ctx.files, format="ipc")
        dataset.schema                          # opening the dataset is lazy; touch it to fail early
        con.register("data", dataset)
        return con, "pyarrow.dataset"
    except Exception as exc:
        notes.append(f"dataset: {type(exc).__name__}")
    # Last resort: a single-use reader over the raw IPC streams. One query per subprocess, so a
    # reader that can only be scanned once is enough.
    con.register("data", batch_reader(ctx.files))
    return con, "record batch reader (" + ", ".join(notes) + " unavailable)"


def duck_query(ctx, sql, register_dim=False):
    con, source = duck_relation(ctx)
    if register_dim:
        con.register("dim", dim_table())
    return con.sql(sql).arrow(), {"source": source}


def dk_filter_sum(ctx):
    table, extra = duck_query(ctx, "SELECT sum(amount) AS total FROM data WHERE region < 100")
    return num(table["total"][0].as_py()), extra


def dk_groupby_1k(ctx):
    table, extra = duck_query(
        ctx, "SELECT region, sum(amount) AS total, count(*) AS n FROM data GROUP BY region")
    return group_summary(table, "region"), extra


def dk_groupby_10m(ctx):
    table, extra = duck_query(
        ctx, "SELECT bigkey, sum(amount) AS total FROM data GROUP BY bigkey")
    return group_summary(table, "bigkey", sample=None), extra


def dk_topk(ctx):
    table, extra = duck_query(ctx, f"SELECT amount FROM data ORDER BY amount DESC LIMIT {TOPK}")
    return amounts(table, TOPK), extra


def dk_count_distinct(ctx):
    table, extra = duck_query(ctx, "SELECT approx_count_distinct(id) AS n FROM data")
    extra["exact"] = False
    return int(table["n"][0].as_py()), extra


def dk_sort_limit(ctx):
    table, extra = duck_query(ctx,
                              f"SELECT amount FROM data ORDER BY amount DESC LIMIT {SORT_LIMIT}")
    return amounts(table, SORT_LIMIT), extra


def dk_broadcast_join(ctx):
    table, extra = duck_query(
        ctx,
        "SELECT sum(d.amount) AS total, sum(m.weight) AS weight "
        f"FROM data d JOIN dim m USING (region) WHERE d.region < {JOIN_CUTOFF}",
        register_dim=True)
    return [num(table["total"][0].as_py()), num(table["weight"][0].as_py())], extra


DUCKDB = dict(filter_sum=dk_filter_sum, groupby_1k=dk_groupby_1k, groupby_10m=dk_groupby_10m,
              topk=dk_topk, count_distinct=dk_count_distinct, sort_limit=dk_sort_limit,
              broadcast_join=dk_broadcast_join)


# ---------------------------------------------------------------- pyarrow


def pa_batches(ctx, columns=None):
    """Batches from a pyarrow.dataset scanner when the dataset opens, otherwise straight from the IPC
    readers. Either way one batch is live at a time."""
    try:
        import pyarrow.dataset as pads
        scanner = pads.dataset(ctx.files, format="ipc").scanner(columns=columns)
        return scanner.to_batches()
    except Exception:
        pass
    if columns is None:
        return iter_batches(ctx.files)
    return (b.select(columns) for b in iter_batches(ctx.files))


class Rollup:
    """Per-batch group-by partials, compacted every `compact_every` batches so the accumulator stays
    a small multiple of the distinct-key count instead of growing with the dataset."""

    def __init__(self, key, count=False, compact_every=8):
        self.key, self.count, self.every = key, count, compact_every
        self.parts, self.acc = [], None

    def _aggregate(self, table, first):
        # pyarrow names an aggregate output "<column>_<function>" and its position relative to the
        # key has moved between releases, so the columns are picked by name, never by index.
        if first:
            aggs, mapping = [("amount", "sum")], {"amount_sum": "total"}
            if self.count:
                aggs.append((self.key, "count"))
                mapping[f"{self.key}_count"] = "n"
        else:
            aggs, mapping = [("total", "sum")], {"total_sum": "total"}
            if self.count:
                aggs.append(("n", "sum"))
                mapping["n_sum"] = "n"
        out = table.group_by(self.key).aggregate(aggs)
        mapping[self.key] = self.key
        return rename(out, mapping)

    def add(self, batch):
        self.parts.append(self._aggregate(pa.Table.from_batches([batch]), True))
        if len(self.parts) >= self.every:
            self.compact()

    def compact(self):
        tables = ([self.acc] if self.acc is not None else []) + self.parts
        self.parts = []
        if not tables:
            return
        self.acc = self._aggregate(pa.concat_tables(tables), False)

    def result(self):
        self.compact()
        return self.acc if self.acc is not None else pa.table({self.key: [], "total": []})


class TopK:
    """A running top-k: concatenate, and re-select whenever the buffer grows past a few k."""

    def __init__(self, k):
        self.k, self.acc = k, None

    def add(self, table):
        self.acc = table if self.acc is None else pa.concat_tables([self.acc, table])
        if self.acc.num_rows > self.k * 8:
            self.trim()

    def trim(self):
        if self.acc is None or self.acc.num_rows <= self.k:
            return
        idx = pc.select_k_unstable(self.acc, k=self.k, sort_keys=[("amount", "descending")])
        self.acc = self.acc.take(idx)

    def result(self):
        self.trim()
        return self.acc if self.acc is not None else pa.table({"amount": pa.array([], pa.float64())})


def pa_filter_sum(ctx):
    total = 0.0
    for batch in pa_batches(ctx, ["region", "amount"]):
        kept = pc.filter(batch["amount"], pc.less(batch["region"], 100))
        total += float(pc.sum(kept).as_py() or 0.0)
    return total, {}


def pa_groupby_1k(ctx):
    roll = Rollup("region", count=True)
    for batch in pa_batches(ctx, ["region", "amount"]):
        roll.add(batch)
    return group_summary(roll.result(), "region"), {}


def pa_groupby_10m(ctx):
    roll = Rollup("bigkey", count=False, compact_every=4)
    for batch in pa_batches(ctx, ["bigkey", "amount"]):
        roll.add(batch)
    return group_summary(roll.result(), "bigkey", sample=None), {}


def pa_topk(ctx):
    top = TopK(TOPK)
    for batch in pa_batches(ctx, ["amount"]):
        top.add(pa.Table.from_batches([batch]))
    return amounts(top.result(), TOPK), {}


def pa_count_distinct(ctx):
    """pyarrow has no distinct sketch, so this is the exact answer: the running set of unique values,
    which on a high-cardinality column is the memory-hungry outcome the benchmark is looking for."""
    acc = None
    pending = []
    for batch in pa_batches(ctx, ["id"]):
        pending.append(pc.unique(batch["id"]))
        if len(pending) >= 8:
            acc = pc.unique(pa.concat_arrays(([acc] if acc is not None else []) + pending))
            pending = []
    if pending or acc is None:
        acc = pc.unique(pa.concat_arrays(([acc] if acc is not None else []) + pending))
    return int(len(acc)), {"exact": True}


def pa_sort_limit(ctx):
    top = TopK(SORT_LIMIT)
    for batch in pa_batches(ctx, ["amount"]):
        top.add(pa.Table.from_batches([batch]))
    return amounts(top.result(), SORT_LIMIT), {}


def pa_broadcast_join(ctx):
    dim = dim_table()
    total = weight = 0.0
    for batch in pa_batches(ctx, ["region", "amount"]):
        table = pa.Table.from_batches([batch])
        table = table.filter(pc.less(table["region"], JOIN_CUTOFF))
        if table.num_rows == 0:
            continue
        joined = table.join(dim, keys="region", join_type="inner")
        total += float(pc.sum(joined["amount"]).as_py() or 0.0)
        weight += float(pc.sum(joined["weight"]).as_py() or 0.0)
    return [total, weight], {}


PYARROW = dict(filter_sum=pa_filter_sum, groupby_1k=pa_groupby_1k, groupby_10m=pa_groupby_10m,
               topk=pa_topk, count_distinct=pa_count_distinct, sort_limit=pa_sort_limit,
               broadcast_join=pa_broadcast_join)


IMPLS = dict(arrowmetal=AM, polars=POLARS, duckdb=DUCKDB, pyarrow=PYARROW)


# ---------------------------------------------------------------- child process


def engine_version(engine):
    try:
        if engine == "arrowmetal":
            import arrowmetal as am
            return f"{am.__version__} on {am.device_name()}"
        if engine == "polars":
            import polars as pl
            return pl.__version__
        if engine == "duckdb":
            import duckdb
            return duckdb.__version__
        if engine == "pyarrow":
            return pa.__version__
    except Exception as exc:
        return f"unavailable ({type(exc).__name__}: {exc})"[:120]
    return "?"


def run_one(engine, workload, data_dir, args):
    """The child process: one engine, one workload, one JSON line on stdout."""
    manifest = load_manifest(data_dir)
    ctx = Ctx(data_dir, manifest, args)
    fn = IMPLS[engine][workload]
    try:
        record = measure(lambda: fn(ctx))
    except BaseException as exc:               # BaseException so a MemoryError is recorded, not raised
        record = dict(status="error", reason=f"{type(exc).__name__}: {exc}".replace("\n", " ")[:300])
    record.update(engine=engine, workload=workload, version=engine_version(engine))
    sys.stdout.write(RESULT_MARK + json.dumps(record, default=str) + "\n")
    sys.stdout.flush()


def child_command(engine, workload, data_dir, args):
    cmd = [sys.executable, os.path.abspath(__file__), "--data-dir", data_dir,
           "--run-one", engine, workload, "--reuse", "--keep",
           "--batch-rows", str(args.batch_rows), "--prefetch", str(args.prefetch)]
    return cmd


def run_measurement(engine, workload, data_dir, args):
    """Spawn the child, parse its result line, and turn every other outcome into a failure record."""
    started = time.perf_counter()
    try:
        proc = subprocess.run(child_command(engine, workload, data_dir, args),
                              capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return dict(engine=engine, workload=workload, status="timeout",
                    reason=f"did not finish within {args.timeout:.0f}s")
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_MARK):
            record = json.loads(line[len(RESULT_MARK):])
            record["subprocess_s"] = time.perf_counter() - started
            return record
    if proc.returncode < 0:
        return dict(engine=engine, workload=workload, status="killed",
                    reason=f"did not finish (killed by signal {-proc.returncode}, likely OOM)")
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    reason = tail[-1][:300] if tail else f"exit code {proc.returncode}, no output"
    return dict(engine=engine, workload=workload, status="error", reason=reason)


# ---------------------------------------------------------------- report


def fmt(value, spec):
    return "-" if value is None else format(value, spec)


def report(records, ctx_bytes, rows, engines, workloads, checks):
    gb = ctx_bytes / 1e9
    print(f"\n# Streaming benchmark: {gb:.2f} GB, {rows:,} rows, 8 columns\n")
    for workload in workloads:
        print(f"## {workload} — {WORKLOAD_DESC[workload]}\n")
        print("| Engine | Wall (ms) | GB/s | CPU (ms) | rusage CPU (ms) | Peak RSS (MB) | "
              "Finished | Check |")
        print("|---|---:|---:|---:|---:|---:|---|---|")
        for engine in engines:
            r = records.get((workload, engine))
            if r is None:
                continue
            if r["status"] != "ok":
                print(f"| {engine} | - | - | - | - | - | no ({r['status']}) | "
                      f"{r.get('reason', '')[:80]} |")
                continue
            wall_s = r["wall_ms"] / 1000.0
            gbs = ctx_bytes / wall_s / 1e9 if wall_s > 0 else None
            check = checks.get((workload, engine), "")
            print(f"| {engine} | {fmt(r['wall_ms'], '.0f')} | {fmt(gbs, '.2f')} | "
                  f"{fmt(r['cpu_ms'], '.0f')} | {fmt(r['rusage_cpu_ms'], '.0f')} | "
                  f"{fmt(r['peak_rss_mb'], '.0f')} | yes | {check} |")
        print()
    print("## ArrowMetal pipeline\n")
    print("| Workload | Overlap | Batches | Rows | Read (s) | GPU (s) | Merge (s) | Read GB/s |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for workload in workloads:
        r = records.get((workload, "arrowmetal"))
        if r is None or r["status"] != "ok":
            continue
        s = r.get("extra") or {}
        read_s = s.get("read_s")
        read_gbs = (s.get("bytes_read", 0) / read_s / 1e9) if read_s else None
        print(f"| {workload} | {fmt(s.get('overlap'), '.2f')} | {s.get('batches', '-')} | "
              f"{s.get('rows', '-')} | {fmt(s.get('read_s'), '.1f')} | {fmt(s.get('gpu_s'), '.1f')} | "
              f"{fmt(s.get('merge_s'), '.1f')} | {fmt(read_gbs, '.2f')} |")
    print()


def cross_check(records, rows, workloads, engines):
    """Compare ArrowMetal against the reference engine on the checked workloads. Returns the per-cell
    labels for the report."""
    checks = {}
    reference = "polars" if "polars" in engines else "pyarrow"
    for workload in workloads:
        if workload not in CHECKED:
            continue
        ref = records.get((workload, reference))
        cand = records.get((workload, "arrowmetal"))
        if ref is None or ref["status"] != "ok":
            checks[(workload, "arrowmetal")] = f"no {reference} reference"
            continue
        checks[(workload, reference)] = "reference"
        if cand is None or cand["status"] != "ok":
            continue
        bad = compare(ref["value"], cand["value"])
        checks[(workload, "arrowmetal")] = "PASS" if bad is None else f"FAIL ({bad[:60]})"
    if "count_distinct" in workloads:
        cand = records.get(("count_distinct", "arrowmetal"))
        print("## count_distinct accuracy\n")
        print(f"true distinct ids (by construction): {rows:,}")
        for engine in engines:
            r = records.get(("count_distinct", engine))
            if r is None or r["status"] != "ok":
                continue
            exact = (r.get("extra") or {}).get("exact")
            kind = "exact" if exact else "approximate"
            err = abs(r["value"] - rows) / rows if rows else 0.0
            print(f"  {engine}: {r['value']:,} ({kind}, relative error {err * 100:.4f}%)")
        if cand is not None and cand["status"] == "ok":
            checks[("count_distinct", "arrowmetal")] = \
                f"{abs(cand['value'] - rows) / rows * 100:.3f}% error"
        print()
    return checks


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True,
                    help="directory for the generated dataset (required; it is deleted unless --keep)")
    ap.add_argument("--size-gb", type=float, default=30.0, help="target dataset size")
    ap.add_argument("--disk-budget-gb", type=float, default=40.0,
                    help="hard cap on the dataset size, whatever --size-gb asks for")
    ap.add_argument("--batch-rows", type=int, default=1_000_000, help="rows per record batch")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse an existing dataset in --data-dir when its manifest matches")
    ap.add_argument("--keep", action="store_true", help="do not delete the dataset at the end")
    ap.add_argument("--engines", default=",".join(ENGINES))
    ap.add_argument("--workloads", default=",".join(WORKLOADS))
    ap.add_argument("--json", dest="json_out", default=None, help="write the raw numbers here")
    ap.add_argument("--timeout", type=float, default=900.0, help="seconds per measurement")
    ap.add_argument("--prefetch", type=int, default=3, help="ArrowMetal scan prefetch depth")
    ap.add_argument("--ipc-format", choices=("stream", "file"), default="stream",
                    help="IPC framing to write; 'file' adds a footer for scanners that need one")
    ap.add_argument("--run-one", nargs=2, metavar=("ENGINE", "WORKLOAD"), default=None,
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    if args.run_one:
        run_one(args.run_one[0], args.run_one[1], data_dir, args)
        return

    engines = [e for e in args.engines.split(",") if e]
    workloads = [w for w in args.workloads.split(",") if w]
    for name in engines:
        if name not in IMPLS:
            ap.error(f"unknown engine {name!r}; choose from {', '.join(ENGINES)}")
    for name in workloads:
        if name not in WORKLOADS:
            ap.error(f"unknown workload {name!r}; choose from {', '.join(WORKLOADS)}")

    print("engines: " + ", ".join(f"{e} {engine_version(e)}" for e in engines))
    print(f"workloads: {', '.join(workloads)}; timeout {args.timeout:.0f}s per measurement\n")

    reused = None
    if args.reuse:
        reused = can_reuse(args, data_dir)
        if reused is not None:
            print(f"reusing {reused['bytes'] / 1e9:.2f} GB in {data_dir} "
                  f"({reused['files']} files, {reused['rows']:,} rows)\n")
    manifest = reused if reused is not None else generate(args, data_dir)

    records, order = {}, []
    try:
        for workload in workloads:
            for engine in engines:
                r = run_measurement(engine, workload, data_dir, args)
                r.setdefault("workload", workload)
                r.setdefault("engine", engine)
                records[(workload, engine)] = r
                order.append(r)
                if r["status"] == "ok":
                    gbs = manifest["bytes"] / (r["wall_ms"] / 1000.0) / 1e9
                    print(f"  {workload:<16} {engine:<11} {r['wall_ms'] / 1000.0:8.1f} s "
                          f"{gbs:6.2f} GB/s  {r['peak_rss_mb']:8.0f} MB peak RSS")
                else:
                    print(f"  {workload:<16} {engine:<11} !! {r['status']}: "
                          f"{r.get('reason', '')[:80]}")
    finally:
        if not args.keep and reused is None:
            # Only the dataset this run generated is removed; a reused one is the caller's.
            remove_dataset(data_dir)
            print(f"\nremoved the generated dataset in {data_dir}")

    checks = cross_check(records, manifest["rows"], workloads, engines)
    report(records, manifest["bytes"], manifest["rows"], engines, workloads, checks)

    if args.json_out:
        payload = dict(dataset=manifest, timeout_s=args.timeout,
                       checks={f"{w}/{e}": v for (w, e), v in checks.items()},
                       results=order)
        with open(args.json_out, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
