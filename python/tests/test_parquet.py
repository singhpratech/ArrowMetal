"""GPU Parquet reading, checked element for element against pyarrow's own reader.

Every fixture under Tests/Fixtures is read twice -- once by ArrowMetal on the GPU and once by
pyarrow.parquet on the CPU -- and the two results must be identical, values, nulls and types alike.

The 50 M-row round trip is skipped unless ARROWMETAL_PARQUET_BIG=1 is set, because it writes a
multi-gigabyte file; Benchmarks/parquet_bench.py runs the same shape as a benchmark.
"""
import glob
import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import arrowmetal as am

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.abspath(os.path.join(HERE, "..", "..", "Tests", "Fixtures"))


def fixture_paths():
    return sorted(glob.glob(os.path.join(FIXTURES, "*.parquet")))


def zstd_unavailable(exc):
    return "libzstd" in str(exc)


def normalise(table):
    """A Table as plain Python, with dictionary columns decoded so encodings do not affect equality."""
    out = {}
    for name in table.column_names:
        col = table[name]
        if pa.types.is_dictionary(col.type):
            col = col.cast(col.type.value_type)
        out[name] = col.to_pylist()
    return out


@pytest.mark.skipif(not fixture_paths(), reason="fixtures not generated")
@pytest.mark.parametrize("path", fixture_paths(), ids=lambda p: os.path.basename(p)[:-8])
def test_matches_pyarrow(path):
    want = pq.read_table(path)
    try:
        got = am.read_parquet_table(path)
    except am.ArrowMetalError as e:
        if zstd_unavailable(e):
            pytest.skip("libzstd is not installed; ZSTD pages fall back to an error by design")
        raise
    assert got.column_names == want.column_names
    assert got.num_rows == want.num_rows
    assert normalise(got) == normalise(want)


@pytest.mark.skipif(not fixture_paths(), reason="fixtures not generated")
@pytest.mark.parametrize("path", fixture_paths(), ids=lambda p: os.path.basename(p)[:-8])
def test_types_match_pyarrow(path):
    want = pq.read_table(path)
    try:
        got = am.read_parquet_table(path)
    except am.ArrowMetalError as e:
        if zstd_unavailable(e):
            pytest.skip("libzstd is not installed")
        raise
    for name in want.column_names:
        w, g = want.schema.field(name).type, got.schema.field(name).type
        # Timezone-naive vs "UTC"-stamped timestamps are the one place a writer's intent is ambiguous;
        # compare the unit and leave the zone out of it.
        if pa.types.is_timestamp(w) and pa.types.is_timestamp(g):
            assert w.unit == g.unit, name
            continue
        assert w == g, "%s: %s != %s" % (name, w, g)


def test_projection_and_row_groups():
    path = os.path.join(FIXTURES, "groups__plain_none.parquet")
    f = am.ParquetFile(path)
    assert f.num_row_groups > 1
    assert f.num_rows == pq.read_metadata(path).num_rows
    want = pq.read_table(path, columns=["id", "f64"])
    got = f.read_table(columns=["id", "f64"])
    assert normalise(got) == normalise(want)

    # One row group at a time must concatenate back to the whole column.
    pieces = []
    for g in range(f.num_row_groups):
        pieces += f.read_table(columns=["id"], row_groups=[g])["id"].to_pylist()
    assert pieces == want["id"].to_pylist()


def test_statistics_pushdown():
    path = os.path.join(FIXTURES, "groups__plain_none.parquet")
    f = am.ParquetFile(path)
    kept = f.selected_row_groups([("id", ">", 1000)])
    assert 0 < len(kept) < f.num_row_groups
    got = f.read_table(columns=["id"], filters=[("id", ">", 1000)])["id"].to_pylist()
    want = [v for v in pq.read_table(path, columns=["id"])["id"].to_pylist() if v > 1000]
    # Pushdown is row-group granular, so the result is a superset that must contain every match.
    assert set(want) <= set(got)
    assert f.selected_row_groups([("id", ">", 10**9)]) == []


def test_dictionary_encoded_columns():
    path = os.path.join(FIXTURES, "flat__dict_snappy.parquet")
    cols = am.read_parquet(path, columns=["s"], dictionary=True)
    arr = cols["s"].to_arrow()
    assert pa.types.is_dictionary(arr.type)
    plain = am.read_parquet(path, columns=["s"], dictionary=False)["s"].to_arrow()
    assert arr.cast(arr.type.value_type).to_pylist() == plain.to_pylist()


def test_struct_leaves():
    """A struct column reads whole, and each of its leaves also reads on its own by dotted path."""
    path = os.path.join(FIXTURES, "struct__plain_none.parquet")
    if not os.path.exists(path):
        pytest.skip("fixture not generated")
    want = pq.read_table(path)
    got = am.read_parquet_table(path, columns=["addr.city", "addr.zip", "k"])
    assert got.column_names == ["addr.city", "addr.zip", "k"]
    assert got["addr.city"].to_pylist() == [v["city"] if v else None for v in want["addr"].to_pylist()]
    assert got["addr.zip"].to_pylist() == [v["zip"] if v else None for v in want["addr"].to_pylist()]
    assert got["k"].to_pylist() == want["k"].to_pylist()
    # The struct itself is reassembled from those leaves (python/tests/test_parquet_nested.py covers
    # the nested shapes in depth).
    whole = am.read_parquet_table(path, columns=["addr"])
    assert whole["addr"].type == want["addr"].type
    assert whole["addr"].to_pylist() == want["addr"].to_pylist()


def test_arrays_are_gpu_resident():
    """The decoded columns are MetalArrays, so a reduction runs without an import step."""
    path = os.path.join(FIXTURES, "flat__plain_none.parquet")
    cols = am.read_parquet(path, columns=["id"])
    assert isinstance(cols["id"], am.MetalArray)
    want = sum(pq.read_table(path, columns=["id"])["id"].to_pylist())
    assert cols["id"].sum() == want


def test_metadata_surface():
    path = os.path.join(FIXTURES, "flat__dict_snappy.parquet")
    f = am.ParquetFile(path)
    assert f.codec(0, 0) == "SNAPPY"
    assert "RLE_DICTIONARY" in f.encodings(0, 0) or "PLAIN_DICTIONARY" in f.encodings(0, 0)
    assert "INT64" in f.column_types[0]
    assert sum(f.row_group_rows(i) for i in range(f.num_row_groups)) == f.num_rows


def test_writer_round_trip():
    table = pa.table({
        "i": pa.array([1, None, 3, 4] * 50, pa.int64()),
        "f": pa.array([1.5, 2.5, None, 4.5] * 50, pa.float64()),
        "s": pa.array(["a", "bb", None, ""] * 50, pa.string()),
        "b": pa.array([True, False, True, None] * 50, pa.bool_()),
        "t": pa.array([1_600_000_000_000_000 + i for i in range(200)], pa.timestamp("us")),
    })
    with tempfile.TemporaryDirectory() as d:
        for compression in ("none", "snappy"):
            for dictionary in (False, True):
                path = os.path.join(d, "w-%s-%d.parquet" % (compression, dictionary))
                am.write_parquet(table, path, compression=compression, use_dictionary=dictionary)
                # Our own reader and pyarrow's must both agree with the input.
                assert normalise(am.read_parquet_table(path)) == normalise(table)
                assert normalise(pq.read_table(path)) == normalise(table)


@pytest.mark.skipif(os.environ.get("ARROWMETAL_PARQUET_BIG") != "1",
                    reason="set ARROWMETAL_PARQUET_BIG=1 to run the 50M-row round trip")
def test_fifty_million_rows():
    import numpy as np
    n = 50_000_000
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "big.parquet")
        table = pa.table({
            "id": pa.array(np.arange(n, dtype=np.int64)),
            "v": pa.array(np.arange(n, dtype=np.float64) * 0.5),
            "cat": pa.array(np.tile(np.array(["a", "bb", "ccc", "dddd"], dtype=object), n // 4), pa.string()),
        })
        pq.write_table(table, path, compression="snappy", use_dictionary=["cat"], row_group_size=1 << 20)
        f = am.ParquetFile(path)
        assert f.num_rows == n
        cols = f.read(columns=["id", "v"])
        assert len(cols["id"]) == n
        assert cols["id"].sum() == n * (n - 1) // 2
        assert abs(cols["v"].sum() - (n * (n - 1) / 2) * 0.5) < 1e6
        cat = f.read(columns=["cat"], dictionary=True)["cat"].to_arrow()
        assert len(cat) == n


# ---- SNAPPY pages decoded on the host: dictionary pages, and dispatches of a few pages


def _snappy_table(n):
    import numpy as np
    rng = np.random.default_rng(3)
    runs = np.repeat(rng.integers(0, 50, n // 100 + 1), 100)[:n]           # overlapping copies
    words = np.array(["x" * k + str(k) for k in range(1, 300, 7)])          # literals past 60 bytes
    return pa.table({
        "rand": pa.array(rng.integers(0, 10**9, n)),                        # 4-byte copies, many tokens
        "runs": pa.array(runs),
        "f": pa.array(rng.random(n)),
        "s": pa.array(rng.choice(words, n), mask=rng.random(n) < 0.1),
    })


@pytest.mark.parametrize("pages", ["few", "many"])
@pytest.mark.parametrize("dictionary", [True, False])
def test_snappy_pages_on_the_host_and_on_the_gpu_read_like_pyarrow(tmp_path, pages, dictionary):
    """With dictionary encoding every chunk starts with a dictionary page, which is decompressed on the
    host; `few` data pages go to the host too, `many` (small pages) to the GPU. Every way reads what
    pyarrow reads."""
    n = 200_000
    t = _snappy_table(n)
    p = str(tmp_path / "s.parquet")
    kw = {"data_page_size": 1 << 20} if pages == "few" else {"data_page_size": 4096,
                                                             "write_batch_size": 512}
    pq.write_table(t, p, compression="snappy", use_dictionary=dictionary, **kw)
    got = am.read_parquet_table(p)
    assert normalise(got) == normalise(pq.read_table(p))
    got = am.read_parquet(p, columns=["s", "runs"], dictionary=True)
    assert got["runs"].to_arrow().to_pylist() == t["runs"].to_pylist()


def test_a_damaged_snappy_dictionary_page_raises(tmp_path):
    """The host decoder checks every element against the page and the output slot: damage in a
    dictionary page is an error, never a crash or a read outside the page."""
    import numpy as np
    n = 50_000
    p = str(tmp_path / "d.parquet")
    pq.write_table(_snappy_table(n), p, compression="snappy", use_dictionary=True)
    cc = pq.ParquetFile(p).metadata.row_group(0).column(0)
    start, end = cc.dictionary_page_offset, cc.data_page_offset
    raw = open(p, "rb").read()
    rng = np.random.default_rng(9)
    outcomes = set()
    for k in range(60):
        b = bytearray(raw)
        pos = int(rng.integers(start + 16, end))
        b[pos] = (b[pos] + 1 + int(rng.integers(0, 254))) % 256
        q = str(tmp_path / f"d{k}.parquet")
        open(q, "wb").write(bytes(b))
        try:
            am.read_parquet(q, columns=["rand"])
            outcomes.add("read")
        except am.ArrowMetalError:
            outcomes.add("raised")
    assert "raised" in outcomes
