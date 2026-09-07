"""Out-of-core streaming execution from Python: results equal to pyarrow / Polars on the same data.

Every test builds a dataset with pyarrow, streams it through ArrowMetal one batch at a time, and
compares against the whole-dataset answer computed by pyarrow (or Polars where it is installed).

The last test generates a multi-gigabyte Arrow IPC directory at test time and streams it; it skips
with a reason when the disk cannot spare the space.
"""
import math
import os
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import arrowmetal as am

try:
    import polars as pl
except ImportError:                                   # pragma: no cover - optional
    pl = None


ROWS = 60_000
BATCH = 4_096


def make_table(rows=ROWS, seed=7):
    """Eight columns: dense and sparse keys, a string label, values with nulls, and a float."""
    rng = np.random.default_rng(seed)
    ids = np.arange(rows, dtype=np.int64)
    region = (ids % 137).astype(np.int32)
    bigkey = rng.integers(0, 50_000, size=rows, dtype=np.int64)
    labels = np.array([f"L{i % 23}" for i in ids])
    value = rng.integers(-1000, 1000, size=rows).astype(np.int64)
    mask = (ids % 11) == 3
    amount = rng.random(rows) * 1000.0
    qty = rng.integers(1, 50, size=rows).astype(np.int32)
    flag = (ids % 2).astype(np.int8)
    return pa.table({
        "id": pa.array(ids),
        "region": pa.array(region),
        "bigkey": pa.array(bigkey),
        "label": pa.array(labels),
        "value": pa.array(value, mask=mask),
        "amount": pa.array(amount),
        "qty": pa.array(qty),
        "flag": pa.array(flag),
    })


@pytest.fixture(scope="module")
def table():
    return make_table()


@pytest.fixture(scope="module")
def ipc_dir(tmp_path_factory, table):
    """The same table on disk as a directory of Arrow IPC stream files with ragged batches."""
    d = tmp_path_factory.mktemp("stream-ipc")
    sizes = [BATCH, BATCH, BATCH // 3, BATCH]           # ragged, and the tail is whatever is left
    offset, part = 0, 0
    while offset < table.num_rows:
        path = d / f"part-{part:04d}.arrows"
        with pa.ipc.new_stream(path, table.schema) as w:
            for _ in range(4):
                if offset >= table.num_rows:
                    break
                n = min(sizes[(offset // BATCH) % len(sizes)], table.num_rows - offset)
                w.write_batch(table.slice(offset, n).combine_chunks().to_batches()[0])
                offset += n
        part += 1
    return d


def scan(ipc_dir):
    return am.scan_ipc(str(ipc_dir))


def test_scan_ipc_counts_every_row(ipc_dir, table):
    s = scan(ipc_dir)
    assert s.count() == table.num_rows
    assert s.stats["batches"] > 1
    assert s.stats["rows"] == table.num_rows
    assert s.stats["bytes_read"] > 0


def test_sum_min_max_mean_match_pyarrow(ipc_dir, table):
    got = scan(ipc_dir).aggregate([
        ("sum", "value", "sum"),
        ("count", "value", "nv"),
        ("min", "value", "mn"),
        ("max", "value", "mx"),
        ("mean", "amount", "avg"),
        ("variance", "amount", "var"),
    ])
    assert got["sum"] == pc.sum(table["value"]).as_py()
    assert got["nv"] == table.num_rows - table["value"].null_count
    assert got["mn"] == pc.min(table["value"]).as_py()
    assert got["mx"] == pc.max(table["value"]).as_py()
    assert got["avg"] == pytest.approx(pc.mean(table["amount"]).as_py(), rel=1e-9)
    assert got["var"] == pytest.approx(pc.variance(table["amount"], ddof=0).as_py(), rel=1e-6)


def test_filtered_sum_matches_pyarrow(ipc_dir, table):
    s = scan(ipc_dir).filter(am.col("region") < 10)
    got = s.sum("amount")
    kept = table.filter(pc.less(table["region"], 10))
    assert got == pytest.approx(pc.sum(kept["amount"]).as_py(), rel=1e-9)


def test_group_by_arbitrary_keys_matches_pyarrow(ipc_dir, table):
    out = scan(ipc_dir).group_by("label").agg([
        ("sum", "value", "s"),
        ("count", None, "n"),
        ("mean", "amount", "avg"),
    ])
    want = table.group_by("label").aggregate([("value", "sum"), ("label", "count"), ("amount", "mean")])
    want = want.sort_by("label").to_pydict()
    got = out.sort_by("label").to_pydict()
    assert got["label"] == want["label"]
    assert got["s"] == want["value_sum"]
    assert got["n"] == want["label_count"]
    for a, b in zip(got["avg"], want["amount_mean"]):
        assert a == pytest.approx(b, rel=1e-9)


def test_group_by_multiple_keys(ipc_dir, table):
    out = scan(ipc_dir).group_by(["flag", "label"]).agg([("count", None, "n")]).to_pydict()
    want = table.group_by(["flag", "label"]).aggregate([([], "count_all")]).to_pydict()
    got = {(f, l): n for f, l, n in zip(out["flag"], out["label"], out["n"])}
    exp = {(f, l): n for f, l, n in zip(want["flag"], want["label"], want["count_all"])}
    assert got == exp


def test_group_by_dense_keys_keeps_state_on_the_gpu(ipc_dir, table):
    out = scan(ipc_dir).group_by("region", dense_key_count=137).agg([("sum", "amount", "s")])
    want = table.group_by("region").aggregate([("amount", "sum")]).to_pydict()
    exp = dict(zip(want["region"], want["amount_sum"]))
    got = dict(zip(out.to_pydict()["region"], out.to_pydict()["s"]))
    assert set(got) == set(exp)
    for k, v in got.items():
        assert v == pytest.approx(exp[k], rel=1e-9)


def test_top_k_matches_a_whole_dataset_sort(ipc_dir, table):
    out = scan(ipc_dir).top_k("amount", 25, largest=True)
    want = pc.sort_indices(table["amount"], sort_keys=[("", "descending")])[:25]
    assert out["amount"].to_pylist() == pytest.approx(table["amount"].take(want).to_pylist())


def test_sort_with_limit_matches_pyarrow(ipc_dir, table):
    out = scan(ipc_dir).sort([("amount", True)], limit=40)
    want = table.sort_by([("amount", "descending")]).slice(0, 40)
    assert out["amount"].to_pylist() == pytest.approx(want["amount"].to_pylist())


def test_sort_to_ipc_round_trips_in_order(tmp_path, ipc_dir, table):
    out = tmp_path / "sorted.arrows"
    stats = scan(ipc_dir).select(["id", "amount"]).sort_to_ipc([("amount", False)], out)
    assert stats["rows"] == table.num_rows
    with pa.ipc.open_stream(out) as r:
        got = r.read_all()
    assert got.num_rows == table.num_rows
    values = got["amount"].to_pylist()
    assert values == sorted(values)


def test_quantiles_are_close_to_exact(ipc_dir, table):
    qs = [0.1, 0.5, 0.9, 0.99]
    got = scan(ipc_dir).quantile("amount", qs)
    exact = np.quantile(np.asarray(table["amount"]), qs)
    values = np.sort(np.asarray(table["amount"]))
    for q, g, e in zip(qs, got, exact):
        # Compare in rank space: how far off is the reported value's true rank?
        rank = np.searchsorted(values, g) / len(values)
        assert abs(rank - q) < 0.01, f"q={q} got={g} exact={e}"


def test_count_distinct_approx_is_within_its_error_bound(ipc_dir, table):
    got = scan(ipc_dir).count_distinct_approx("id", precision=14)
    exact = pc.count_distinct(table["id"]).as_py()
    assert abs(got - exact) / exact < 0.05
    assert scan(ipc_dir).count_distinct_approx("label") == pytest.approx(23, abs=2)


def test_filter_project_to_ipc_matches_pyarrow(tmp_path, ipc_dir, table):
    out = tmp_path / "filtered.arrows"
    s = scan(ipc_dir).filter(am.col("value") > 500).project({"id": am.col("id"),
                                                             "doubled": am.col("value") * 2})
    stats = s.sink_ipc(out)
    want = table.filter(pc.greater(table["value"], 500))
    assert stats["rows"] > 0
    with pa.ipc.open_stream(out) as r:
        got = r.read_all()
    assert got.column_names == ["id", "doubled"]
    assert got["id"].to_pylist() == want["id"].to_pylist()
    assert got["doubled"].to_pylist() == pc.multiply(want["value"], 2).to_pylist()


def test_to_reader_is_a_lazy_pyarrow_reader(ipc_dir, table):
    reader = scan(ipc_dir).filter(am.col("flag") == 1).select(["id", "flag"]).to_reader()
    assert isinstance(reader, pa.RecordBatchReader)
    seen, batches = 0, 0
    for batch in reader:
        seen += batch.num_rows
        batches += 1
    assert seen == table.num_rows // 2
    assert batches > 1


def test_parallel_readers_give_the_same_answers(ipc_dir, table):
    """readers > 1 reads several files at once; the batch order changes, the answers do not."""
    serial = am.scan_ipc(str(ipc_dir), readers=1).aggregate([("sum", "amount", "s"),
                                                             ("count", None, "n")])
    parallel = am.scan_ipc(str(ipc_dir), readers=4).aggregate([("sum", "amount", "s"),
                                                               ("count", None, "n")])
    assert parallel["n"] == serial["n"] == table.num_rows
    assert parallel["s"] == pytest.approx(serial["s"], rel=1e-12)

    a = am.scan_ipc(str(ipc_dir), readers=4).group_by("label").agg([("sum", "value", "s")])
    b = am.scan_ipc(str(ipc_dir), readers=1).group_by("label").agg([("sum", "value", "s")])
    assert a.sort_by("label").to_pydict() == b.sort_by("label").to_pydict()


def test_scan_arrow_from_a_pyarrow_dataset(tmp_path, table):
    ds = pytest.importorskip("pyarrow.dataset")
    path = tmp_path / "ds"
    path.mkdir()
    pa.parquet = pytest.importorskip("pyarrow.parquet")
    pa.parquet.write_table(table, path / "part-0.parquet", row_group_size=5_000)
    dataset = ds.dataset(str(path), format="parquet")
    got = am.scan_arrow(dataset.scanner(batch_size=5_000)).sum("amount")
    assert got == pytest.approx(pc.sum(table["amount"]).as_py(), rel=1e-9)


def test_scan_arrow_from_a_record_batch_reader(table):
    reader = pa.RecordBatchReader.from_batches(table.schema, table.to_batches(max_chunksize=3_000))
    assert am.scan_arrow(reader).count() == table.num_rows


def test_broadcast_join_matches_pyarrow(ipc_dir, table):
    dim = pa.table({"region": pa.array(np.arange(137, dtype=np.int32)),
                    "weight": pa.array(np.arange(137, dtype=np.int64) * 10)})
    out = scan(ipc_dir).join(dim, on="region", how="inner", broadcast=True)
    want = table.join(dim, keys="region", join_type="inner")
    assert out.num_rows == want.num_rows
    got = sorted(zip(out["id"].to_pylist(), out["weight"].to_pylist()))
    exp = sorted(zip(want["id"].to_pylist(), want["weight"].to_pylist()))
    assert got == exp


def test_grace_join_equals_a_broadcast_join(tmp_path, ipc_dir, table):
    dim_path = tmp_path / "dim.arrows"
    dim = pa.table({"region": pa.array(np.arange(137, dtype=np.int32)),
                    "weight": pa.array(np.arange(137, dtype=np.int64) * 10)})
    with pa.ipc.new_stream(dim_path, dim.schema) as w:
        w.write_table(dim)
    out = scan(ipc_dir).join(am.scan_ipc(str(dim_path)), on="region", how="inner",
                             broadcast=False, partitions=8, scratch=str(tmp_path / "grace"))
    want = table.join(dim, keys="region", join_type="inner")
    assert out.num_rows == want.num_rows
    got = sorted(zip(out["id"].to_pylist(), out["weight"].to_pylist()))
    exp = sorted(zip(want["id"].to_pylist(), want["weight"].to_pylist()))
    assert got == exp


def test_progress_callback_sees_every_batch(ipc_dir):
    seen = []
    s = scan(ipc_dir)
    s.progress(lambda batches, rows, read, total: seen.append((batches, rows)))
    s.count()
    assert seen
    assert [b for b, _ in seen] == list(range(1, len(seen) + 1))
    assert seen[-1][1] == s.stats["rows"]


def test_pipeline_stats_report_every_stage(ipc_dir):
    s = scan(ipc_dir)
    s.group_by("label").agg([("count", None, "n")])
    st = s.stats
    assert st["wall_s"] > 0
    assert st["gpu_s"] > 0
    assert st["read_s"] >= 0
    assert st["overlap"] > 0


@pytest.mark.skipif(pl is None, reason="polars is not installed")
def test_matches_polars_on_the_same_data(ipc_dir, table):
    df = pl.from_arrow(table)
    want = (df.filter(pl.col("region") < 20)
              .group_by("label")
              .agg(pl.col("amount").sum().alias("s"))
              .sort("label"))
    got = (scan(ipc_dir).filter(am.col("region") < 20)
           .group_by("label").agg([("sum", "amount", "s")]))
    got = pl.from_arrow(got).sort("label")
    assert got["label"].to_list() == want["label"].to_list()
    for a, b in zip(got["s"].to_list(), want["s"].to_list()):
        assert a == pytest.approx(b, rel=1e-9)


# ---- a dataset larger than a batch cache, generated at test time when the disk allows

def _free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def test_two_gigabyte_ipc_directory(tmp_path):
    """Streams a ~2 GB Arrow IPC directory and checks the aggregate against the generator's own sums.

    Skipped with a reason when the volume cannot spare 6 GB, or when ARROWMETAL_SKIP_BIG is set."""
    if os.environ.get("ARROWMETAL_SKIP_BIG"):
        pytest.skip("ARROWMETAL_SKIP_BIG is set")
    if _free_gb(tmp_path) < 6:
        pytest.skip(f"only {_free_gb(tmp_path):.1f} GB free; this test needs about 6 GB")

    target_bytes = 2_000_000_000
    row_bytes = 8 + 4 + 8 + 8 + 4 + 1        # id, region, bigkey, amount, qty, flag
    rows_total = target_bytes // row_bytes
    rows_per_batch = 2_000_000
    batches_per_file = 8
    d = tmp_path / "big"
    d.mkdir()

    schema = pa.schema([("id", pa.int64()), ("region", pa.int32()), ("bigkey", pa.int64()),
                        ("amount", pa.float64()), ("qty", pa.int32()), ("flag", pa.int8())])
    rng = np.random.default_rng(11)
    expected_sum = 0.0
    expected_rows = 0
    written, part = 0, 0
    while written < rows_total:
        with pa.ipc.new_stream(d / f"part-{part:04d}.arrows", schema) as w:
            for _ in range(batches_per_file):
                if written >= rows_total:
                    break
                n = int(min(rows_per_batch, rows_total - written))
                ids = np.arange(written, written + n, dtype=np.int64)
                amount = rng.random(n) * 100.0
                w.write_batch(pa.record_batch([
                    pa.array(ids),
                    pa.array((ids % 1000).astype(np.int32)),
                    pa.array((ids * 2654435761 % 10_000_000).astype(np.int64)),
                    pa.array(amount),
                    pa.array((ids % 47 + 1).astype(np.int32)),
                    pa.array((ids % 2).astype(np.int8)),
                ], schema=schema))
                expected_sum += float(amount.sum())
                expected_rows += n
                written += n
        part += 1

    try:
        size_gb = sum(f.stat().st_size for f in d.iterdir()) / 1e9
        s = am.scan_ipc(str(d))
        got = s.aggregate([("sum", "amount", "s"), ("count", None, "n")])
        assert got["n"] == expected_rows
        assert got["s"] == pytest.approx(expected_sum, rel=1e-9)
        assert s.stats["bytes_read"] > 0
        assert size_gb > 1.0

        groups = am.scan_ipc(str(d)).group_by("region", dense_key_count=1000).agg([("count", None, "n")])
        assert groups.num_rows == 1000
        assert sum(groups["n"].to_pylist()) == expected_rows
    finally:
        shutil.rmtree(d, ignore_errors=True)
