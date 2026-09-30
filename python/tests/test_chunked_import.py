"""Chunked import: `am.array(pa.chunked_array)` against `am.array` of the concatenation, value for
value and null for null, over types and chunk layouts (empty chunks, one-row chunks, sliced chunks
with non-zero offsets, chunks without validity next to chunks with it, all-null chunks, one chunk,
10,000 chunks), then through sum, sort and group-by, and through the Table and Polars paths."""
import datetime
import decimal
import random

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import arrowmetal as am


def _values(ty, n, rng, null_every):
    def null(i):
        return null_every and i % null_every == 0

    def word():
        k = rng.randrange(13, 40) if rng.random() < 0.3 else rng.randrange(0, 13)
        return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(k))

    gen = {
        "int8": lambda: rng.randrange(-128, 128),
        "uint16": lambda: rng.randrange(0, 1 << 16),
        "int32": lambda: rng.randrange(-(1 << 31), 1 << 31),
        "int64": lambda: rng.randrange(-(1 << 40), 1 << 40),
        "uint64": lambda: rng.randrange(0, 1 << 64),
        "float32": lambda: rng.uniform(-1e3, 1e3),
        "float64": lambda: rng.uniform(-1e6, 1e6),
        "float16": lambda: np.float16(rng.uniform(-100, 100)),
        "bool": lambda: rng.random() < 0.5,
        "date32": lambda: datetime.date(2000, 1, 1) + datetime.timedelta(days=rng.randrange(0, 20000)),
        "date64": lambda: datetime.date(2000, 1, 1) + datetime.timedelta(days=rng.randrange(0, 20000)),
        "timestamp": lambda: rng.randrange(0, 2 * 10 ** 15),
        "duration": lambda: rng.randrange(-10 ** 12, 10 ** 12),
        "time32": lambda: rng.randrange(0, 86400),
        "decimal128": lambda: decimal.Decimal(rng.randrange(-10 ** 15, 10 ** 15)).scaleb(-3),
        "decimal256": lambda: decimal.Decimal(rng.randrange(-10 ** 30, 10 ** 30)).scaleb(-2),
        "fixed5": lambda: bytes(rng.randrange(256) for _ in range(5)),
        "utf8": word, "large_utf8": word, "utf8_view": word,
        "binary": lambda: word().encode(), "large_binary": lambda: word().encode(),
        "binary_view": lambda: word().encode(),
        "null": lambda: None,
    }[ty]
    return [None if null(i) else gen() for i in range(n)]


TYPES = {
    "int8": pa.int8(), "uint16": pa.uint16(), "int32": pa.int32(), "int64": pa.int64(), "uint64": pa.uint64(),
    "float32": pa.float32(), "float64": pa.float64(), "float16": pa.float16(), "bool": pa.bool_(),
    "date32": pa.date32(), "date64": pa.date64(), "timestamp": pa.timestamp("us", tz="UTC"),
    "duration": pa.duration("ns"), "time32": pa.time32("s"),
    "decimal128": pa.decimal128(20, 3), "decimal256": pa.decimal256(40, 2), "fixed5": pa.binary(5),
    "utf8": pa.string(), "large_utf8": pa.large_string(), "utf8_view": pa.string_view(),
    "binary": pa.binary(), "large_binary": pa.large_binary(), "binary_view": pa.binary_view(),
    "null": pa.null(),
}


def _array(ty, n, rng, null_every):
    return pa.array(_values(ty, n, rng, null_every), type=TYPES[ty])


def _chunks(ty, sizes, seed):
    """Chunks of `sizes` rows, each a slice at a random offset of a longer array; every fourth is
    all null, every fourth another has no nulls (so no validity buffer)."""
    rng = random.Random(seed)
    out = []
    for i, n in enumerate(sizes):
        pad = 0 if rng.random() < 0.3 else rng.randrange(70)
        null_every = {3: 1, 1: 0}.get(i % 4, 5)
        out.append(_array(ty, pad + n + 3, rng, null_every).slice(pad, n))
    return out


@pytest.fixture
def counted(monkeypatch):
    """Counts the chunked-import calls `am.array` makes."""
    calls = []
    real = am._import_chunks

    def wrapper(chunks):
        calls.append(len(chunks))
        return real(chunks)

    monkeypatch.setattr(am, "_import_chunks", wrapper)
    return calls


def _check(ty, chunks):
    ca = pa.chunked_array(chunks, type=TYPES[ty])
    flat = ca.combine_chunks()
    got, want = am.array(ca), am.array(flat)
    assert len(got) == len(flat)
    assert got.null_count == flat.null_count
    g, w = got.to_arrow(), want.to_arrow()
    assert g.type == w.type, ty
    assert g.equals(w), ty
    assert g.to_pylist() == flat.to_pylist(), ty
    return got


LAYOUTS = {
    "empty and one-row chunks": [0, 1, 7, 0, 64, 1, 100, 13, 0, 3, 33, 1],
    "two chunks": [37, 91],
    "one non-empty among empty": [0, 57, 0],
    "only empty chunks": [0, 0],
}


@pytest.mark.parametrize("ty", list(TYPES))
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_chunked_equals_concatenation(ty, layout, counted):
    _check(ty, _chunks(ty, LAYOUTS[layout], hash((ty, layout)) & 0xFFFF))
    assert counted == [len(LAYOUTS[layout])], "the chunked import was used"


@pytest.mark.parametrize("ty", ["int64", "float64", "bool", "utf8", "large_utf8", "utf8_view", "decimal128"])
def test_ten_thousand_chunks(ty):
    rng = random.Random(3)
    _check(ty, _chunks(ty, [rng.randrange(4) for _ in range(10_000)], 11))


def test_one_chunk_is_the_single_import(counted):
    a = pa.array(np.arange(1 << 16, dtype=np.int64))
    m = am.array(pa.chunked_array([a]))
    assert counted == []
    assert m.to_arrow().equals(a)


def test_import_threads_setting():
    saved = am.get_import_threads()
    try:
        am.set_import_threads(3)
        assert am.get_import_threads() == 3
        am.set_import_threads(0)
        assert am.get_import_threads() == 0
        with pytest.raises(ValueError):
            am.set_import_threads(-1)
    finally:
        am.set_import_threads(saved)


@pytest.mark.parametrize("ty", ["int64", "bool", "utf8", "utf8_view", "utf8_view_many"])
def test_thread_counts_give_the_same_array(ty):
    """One thread, several and the default policy import the same column (1.2M rows, so every
    copy step splits over the threads it is given). utf8_view_many has more data buffers than are
    wrapped, so they are copied into merged buffers."""
    rng = np.random.default_rng(17)
    words = pa.array([f"a-longer-string-{k:05d}" if k % 3 else f"s{k}" for k in range(5000)], pa.string())
    chunks = []
    for n in (rng.integers(1, 30_000, 80) if ty == "utf8_view_many" else rng.integers(1, 60_000, 40)):
        idx = pa.array(rng.integers(0, 5000, int(n)), mask=rng.random(int(n)) < 0.1)
        if ty == "int64":
            chunks.append(pa.array(rng.integers(-(1 << 62), 1 << 62, int(n)), mask=rng.random(int(n)) < 0.1))
        elif ty == "bool":
            chunks.append(pa.array(rng.random(int(n)) < 0.5, mask=rng.random(int(n)) < 0.1))
        else:
            s = pc.take(words, idx)
            chunks.append(s.cast(pa.string_view()) if ty.startswith("utf8_view") else s)
    ca = pa.chunked_array(chunks)
    saved = am.get_import_threads()
    try:
        results = []
        for t in [1, 2, 5, 16, 0]:
            am.set_import_threads(t)
            results.append(am.array(ca).to_arrow())
    finally:
        am.set_import_threads(saved)
    flat = ca.combine_chunks()
    for r in results:
        assert r.equals(results[0])
    assert results[0].to_pylist() == flat.to_pylist()


def test_slices_of_one_view_array_share_data_buffers():
    rng = random.Random(5)
    base = _array("utf8_view", 5000, rng, 7)
    cuts = sorted(rng.sample(range(1, 5000), 60))
    chunks = [base.slice(a, b - a) for a, b in zip([0] + cuts, cuts + [5000])]
    _check("utf8_view", chunks)


def test_dictionary_chunks_fall_back_to_combine(counted):
    d = pa.chunked_array([pa.array(["a", "b", "a"]).dictionary_encode(),
                          pa.array(["b", "c"]).dictionary_encode()])
    m = am.array(d)
    assert m.to_arrow().cast(pa.string()).to_pylist() == ["a", "b", "a", "b", "c"]


def test_kernels_on_chunked_arrays():
    rng = random.Random(21)
    n = 200_000
    sizes, left = [], n
    while left:
        k = min(left, 1 + rng.randrange(8192))
        sizes.append(k)
        left -= k

    def cut(a):
        out, at = [], 0
        for k in sizes:
            out.append(a.slice(at, k))
            at += k
        return pa.chunked_array(out, type=a.type)

    ints = pa.array([None if i % 13 == 0 else rng.randrange(-10 ** 6, 10 ** 6) for i in range(n)], pa.int64())
    dbls = pa.array([None if i % 17 == 0 else rng.uniform(-1e3, 1e3) for i in range(n)], pa.float64())
    keys = pa.array([rng.randrange(100) for _ in range(n)], pa.int32())
    strs = pa.array([None if i % 19 == 0 else f"w{rng.randrange(5000)}" + ("-a-longer-suffix" if i % 3 == 0 else "")
                     for i in range(n)], pa.string_view())
    ci, cd, ck, cs = (am.array(cut(x)) for x in (ints, dbls, keys, strs))
    wi, wd, wk, ws = (am.array(x) for x in (ints, dbls, keys, strs))
    assert ci.sum() == wi.sum() == pc.sum(ints).as_py()
    assert cd.sum() == wd.sum()
    assert ci.sort().to_arrow().equals(wi.sort().to_arrow())
    assert cd.argsort(descending=True).to_arrow().equals(wd.argsort(descending=True).to_arrow())
    assert cs.argsort().to_arrow().equals(ws.argsort().to_arrow())
    assert ck.group_by(100).sum(ci).to_arrow().equals(wk.group_by(100).sum(wi).to_arrow())


def test_table_paths_keep_chunks(counted):
    rng = random.Random(8)
    n = 30_000
    t = pa.table({"k": pa.array([rng.randrange(10) for _ in range(n)], pa.int64()),
                  "s": pa.array([f"s{rng.randrange(100)}" for _ in range(n)], pa.string())})
    t = pa.Table.from_batches(t.to_batches(max_chunksize=4096))
    assert t.column("k").num_chunks > 1
    got = am.scan(t).group_by("s").agg(am.agg.sum("k", "total")).sort("s").collect()
    want = t.group_by("s").aggregate([("k", "sum")]).sort_by("s")
    assert got.column("total").to_pylist() == want.column("k_sum").to_pylist()
    assert len(counted) >= 1
    assert am.query(t, am.col("k").sum("total")) == pc.sum(t.column("k")).as_py()


def test_polars_engine_multi_chunk_frame():
    pl = pytest.importorskip("polars")
    a = pl.DataFrame({"k": [1, 2, 3] * 1000, "s": ["x", "y", "z-long-string-here"] * 1000})
    b = pl.DataFrame({"k": [4, 5] * 500, "s": ["u", None] * 500})
    df = pl.concat([a, b], rechunk=False)
    assert df["k"].n_chunks() > 1
    from arrowmetal import polars_engine as pe
    got = df.lazy().group_by("s").agg(pl.col("k").sum()).sort("s").collect(engine=pe.MetalEngine(raise_on_fail=True, min_rows=0, shapes="all"))
    want = df.lazy().group_by("s").agg(pl.col("k").sum()).sort("s").collect()
    assert got.equals(want)
