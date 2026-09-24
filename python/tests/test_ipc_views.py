"""Arrow IPC beyond the classic layouts, from Python: view types, big-endian files and the
arrow.fixed_shape_tensor extension type, read through `am.scan_ipc`.

pyarrow 25 is the oracle. It writes the view-typed and tensor inputs; the big-endian inputs are Arrow's
own integration files under Tests/Fixtures/ipc (see the README there), compared with pyarrow reading
their little-endian twins and with the values in their JSON files.
"""
import gzip
import json
import os
import struct

import numpy as np
import pyarrow as pa
import pytest

import arrowmetal as am

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Tests", "Fixtures", "ipc")


def read(path):
    return am.scan_ipc(path, prefetch=0).collect()


def write(path, batches, fmt, **kw):
    open_ = pa.ipc.new_file if fmt == "file" else pa.ipc.new_stream
    with open_(path, batches[0].schema, **kw) as w:
        for b in batches:
            w.write_batch(b)


# ---- view types

N = 40
LONG = "a string longer than twelve bytes, row %d"


def view_batch():
    sv = pa.array([None if i % 5 == 1 else ("s%d" % i if i % 3 else LONG % i) for i in range(N)], pa.string_view())
    bv = pa.array([None if i % 4 == 2 else bytes([i % 256]) * (i % 20) for i in range(N)], pa.binary_view())
    child = pa.array([None if k % 9 == 4 else k * 10 for k in range(70)], pa.int32())
    lv = pa.ListViewArray.from_arrays(pa.array([(i * 7) % 60 for i in range(N)], pa.int32()),
                                      pa.array([i % 5 for i in range(N)], pa.int32()), child,
                                      mask=pa.array([i % 6 == 3 for i in range(N)]))
    llv = pa.array([None if i % 7 == 2 else ["w%d" % k for k in range(i % 4)] for i in range(N)],
                   pa.large_list_view(pa.string()))
    st = pa.StructArray.from_arrays([sv, lv], names=["a", "b"], mask=pa.array([i % 8 == 5 for i in range(N)]))
    ls = pa.array([None if i % 6 == 0 else [LONG % (i + k) for k in range(i % 3)] for i in range(N)],
                  pa.list_(pa.string_view()))
    dv = pa.array(["red", "green", None, "a colour name past twelve bytes"] * (N // 4), pa.string_view()).dictionary_encode()
    n = pa.array(range(N), pa.int64())
    return pa.record_batch([sv, bv, lv, llv, st, ls, dv, n], names=["sv", "bv", "lv", "llv", "st", "ls", "dv", "n"])


@pytest.mark.parametrize("fmt", ["file", "stream"])
def test_view_types_read_as_their_classic_types(tmp_path, fmt):
    full = view_batch()
    path = str(tmp_path / ("views." + fmt))
    write(path, [full, full.slice(3, 25)], fmt)
    got = read(path)
    want = pa.Table.from_batches([full, full.slice(3, 25)])
    assert got.num_rows == N + 25
    assert got.schema.field("sv").type == pa.string()
    assert got.schema.field("bv").type == pa.binary()
    assert got.schema.field("lv").type == pa.list_(pa.int32())
    # large_list_view offsets are narrowed to int32, as large_list's are.
    assert got.schema.field("llv").type == pa.list_(pa.string())
    assert got.schema.field("st").type == pa.struct([("a", pa.string()), ("b", pa.list_(pa.int32()))])
    assert got.schema.field("ls").type == pa.list_(pa.string())
    for name in want.column_names:
        g = got.column(name).to_pylist()
        w = want.column(name).to_pylist()
        assert g == w, name


def test_view_types_in_a_compressed_stream(tmp_path):
    full = view_batch()
    path = str(tmp_path / "views.arrows")
    write(path, [full], "stream", options=pa.ipc.IpcWriteOptions(compression="lz4"))
    got = read(path)
    for name in full.schema.names:
        assert got.column(name).to_pylist() == full.column(name).to_pylist(), name


def test_group_by_on_a_string_view_key(tmp_path):
    full = view_batch()
    path = str(tmp_path / "views.arrow")
    write(path, [full, full.slice(3, 25)], "file")
    out = am.scan_ipc(path, prefetch=0).group_by("sv").agg([("sum", "n", "s"), ("count", None, "c")])
    table = pa.Table.from_batches([full, full.slice(3, 25)]).select(["sv", "n"])
    table = table.set_column(0, "sv", table.column("sv").cast(pa.string()))
    want = table.group_by("sv").aggregate([("n", "sum"), ([], "count_all")])
    got = {k: (s, c) for k, s, c in zip(out.column("sv").to_pylist(), out.column("s").to_pylist(),
                                        out.column("c").to_pylist())}
    exp = {k: (s, c) for k, s, c in zip(want.column("sv").to_pylist(), want.column("n_sum").to_pylist(),
                                        want.column("count_all").to_pylist())}
    assert got == exp


def test_view_offsets_past_2gb_are_a_clear_error(tmp_path):
    size, rows = 1 << 20, 2100
    view = struct.pack("<i4sii", size, b"zzzz", 0, 0)
    v = pa.Array.from_buffers(pa.string_view(), rows, [None, pa.py_buffer(view * rows), pa.py_buffer(b"z" * size)])
    path = str(tmp_path / "big.arrows")
    write(path, [pa.record_batch([v], names=["v"])], "stream")
    with pytest.raises(Exception, match="view column 'v' over 2 GB"):
        read(path)


# ---- big-endian

NAMES = ["custom_metadata", "datetime", "dictionary", "dictionary_unsigned", "extension", "interval", "map",
         "nested", "nested_large_offsets", "null", "primitive", "primitive_large_offsets", "recursive_nested"]
# Not listed: `union` repeats its column names, and `scan_ipc(...).collect()` addresses columns by name
# (the Swift suite IPCViewTests compares it through ArrowIPCReader directly); `primitive_zerolength`
# holds no rows, which `collect()` returns as an empty table without a schema.


def same(got, want):
    if got.type != want.type:
        return got.to_pylist() == want.to_pylist()
    return got.equals(want)


@pytest.mark.parametrize("ext", ["arrow_file", "stream"])
@pytest.mark.parametrize("name", NAMES)
def test_big_endian_files_match_their_little_endian_twins(name, ext):
    big = os.path.join(FIXTURES, "bigendian", "generated_%s.%s" % (name, ext))
    little = pa.ipc.open_stream(os.path.join(FIXTURES, "littleendian", "generated_%s.stream" % name)).read_all()
    got = read(big)
    assert got.column_names == little.column_names
    for i in range(little.num_columns):
        # Chunked comparison: pyarrow has no Python wrapper for a day_time interval chunk.
        assert same(got.column(i), little.column(i)), (name, little.column_names[i])


@pytest.mark.parametrize("name", ["decimal", "decimal256"])
def test_big_endian_decimals_match_pyarrow(name):
    path = os.path.join(FIXTURES, "bigendian", "generated_%s.stream" % name)
    want = pa.ipc.open_stream(path).read_all()      # pyarrow swaps to native byte order itself
    got = read(path)
    for i in range(want.num_columns):
        assert got.column(i).combine_chunks().equals(want.column(i).combine_chunks()), i


def json_values(column):
    """The values of one flat integration-JSON column: DATA with VALIDITY applied."""
    return [v if ok else None for v, ok in zip(column["DATA"], column["VALIDITY"])]


def test_big_endian_primitive_matches_its_json():
    with gzip.open(os.path.join(FIXTURES, "bigendian", "generated_primitive.json.gz")) as f:
        doc = json.load(f)
    got = read(os.path.join(FIXTURES, "bigendian", "generated_primitive.stream"))
    checked = 0
    row = 0
    for batch in doc["batches"]:
        for column in batch["columns"]:
            field = next(f for f in doc["schema"]["fields"] if f["name"] == column["name"])
            kind = field["type"]["name"]
            if kind not in ("int", "utf8", "bool"):
                continue
            want = json_values(column)
            if kind == "int":
                want = [None if v is None else int(v) for v in want]
            have = got.column(column["name"]).to_pylist()[row:row + batch["count"]]
            assert have == want, column["name"]
            checked += 1
        row += batch["count"]
    assert checked >= 10


# ---- fixed_shape_tensor and tensor messages

def test_fixed_shape_tensor_comes_back_as_itself(tmp_path):
    plain = pa.FixedShapeTensorArray.from_numpy_ndarray(np.arange(24, dtype=np.float32).reshape(4, 2, 3))
    t = pa.fixed_shape_tensor(pa.int64(), [2, 2], dim_names=["r", "c"], permutation=[1, 0])
    named = pa.ExtensionArray.from_storage(t, pa.array([[1, 2, 3, 4], None, [5, 6, 7, 8], [9, 10, 11, 12]],
                                                       pa.list_(pa.int64(), 4)))
    for fmt in ["file", "stream"]:
        path = str(tmp_path / ("tensor." + fmt))
        write(path, [pa.record_batch([plain, named], names=["plain", "named"])], fmt)
        got = read(path)
        assert got.column("plain").type == plain.type
        assert got.column("named").type == t
        assert got.column("named").type.dim_names == ["r", "c"]
        assert got.column("named").type.permutation == [1, 0]
        assert got.column("plain").combine_chunks().equals(plain)
        assert got.column("named").combine_chunks().equals(named)
        np.testing.assert_array_equal(got.column("plain").combine_chunks().to_numpy_ndarray(),
                                      np.arange(24, dtype=np.float32).reshape(4, 2, 3))


def test_ipc_tensor_messages_are_refused(tmp_path):
    path = str(tmp_path / "tensor.bin")
    with pa.OSFile(path, "wb") as f:
        pa.ipc.write_tensor(pa.Tensor.from_numpy(np.arange(6, dtype=np.int32).reshape(2, 3)), f)
    with pytest.raises(Exception, match="IPC Tensor and SparseTensor messages are not read"):
        read(path)
