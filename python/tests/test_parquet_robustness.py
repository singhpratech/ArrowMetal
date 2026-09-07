"""Adversarial Parquet reading: corrupt files must raise, never crash and never hang.

Every case here starts from a file pyarrow wrote and then damages it. The contract under test is
narrow and absolute: whatever ArrowMetal does with the damaged bytes, the *process* must survive it
and must finish in bounded time. Garbage values out of a garbage file are acceptable; a segmentation
fault, a Swift trap, a GPU that never returns, or a read outside the mapped file are not.

Each case runs in a child process, so a crash shows up as a signal naming the case rather than
taking the whole test run with it, and a hang shows up as a timeout instead of a stalled CI job.
"""
import os
import struct
import subprocess
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import arrowmetal as am

HERE = os.path.dirname(os.path.abspath(__file__))
PY_ROOT = os.path.abspath(os.path.join(HERE, ".."))

# Reading one file, printed as a single line the parent can classify.
_CHILD = r"""
import sys
import arrowmetal as am
try:
    t = am.read_parquet_table(sys.argv[1])
    print("READ-OK rows=%d" % t.num_rows)
except am.ArrowMetalError as e:
    print("ERROR %s" % str(e)[:200])
except Exception as e:
    print("PYERROR %s: %s" % (type(e).__name__, str(e)[:200]))
"""

# Reading a whole directory of files in one process: a crash aborts the run and the last line
# printed names the file that did it.
_CHILD_MANY = r"""
import os, sys
import arrowmetal as am
for name in sorted(os.listdir(sys.argv[1])):
    if not name.endswith(".parquet"):
        continue
    sys.stdout.write("AT %s\n" % name); sys.stdout.flush()
    try:
        am.read_parquet_table(os.path.join(sys.argv[1], name))
    except Exception:
        pass
print("ALL-DONE")
"""


def read_in_child(path, timeout=45):
    """`(status, detail)` for reading `path` in a fresh process.

    status is one of "READ-OK", "ERROR", "PYERROR" (all fine), "CRASH" or "TIMEOUT" (both bugs).
    """
    env = dict(os.environ, PYTHONPATH=PY_ROOT, MallocScribble="1")
    try:
        r = subprocess.run([sys.executable, "-c", _CHILD, path],
                           capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", "still running after %ds" % timeout
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()[-3:]
        return "CRASH", "rc=%d %s" % (r.returncode, " | ".join(tail))
    out = (r.stdout or "").strip()
    return (out.split(" ", 1) + [""])[:2]


# --------------------------------------------------------------------------------------------
# Fixtures written here rather than committed: the damaged files are derived, not data.

def _base_table():
    return pa.table({
        "i": pa.array(list(range(400)), pa.int64()),
        "s": pa.array(["value-%04d" % i for i in range(400)]),
        "l": pa.array([[1, 2], [3], None, []] * 100, pa.list_(pa.int64())),
    })


@pytest.fixture(scope="module")
def workdir():
    with tempfile.TemporaryDirectory(prefix="am-pq-robust") as d:
        yield d


def _write(workdir, name, table, **kw):
    path = os.path.join(workdir, name)
    pq.write_table(table, path, **kw)
    return path


# --------------------------------------------------------------------------------------------
# BYTE_ARRAY length fields: a corrupt one used to point the GPU gather at 4 GiB of memory.

def _corrupt_byte_array_length(src, dst, marker):
    """Rewrite the 4-byte PLAIN length prefix in front of `marker` to 0xFFFFFFFF."""
    raw = bytearray(open(src, "rb").read())
    needle = struct.pack("<I", len(marker)) + marker
    at = raw.find(needle)
    assert at >= 0, "PLAIN value %r not found in the file" % marker
    raw[at:at + 4] = struct.pack("<I", 0xFFFFFFFF)
    open(dst, "wb").write(bytes(raw))


def test_corrupt_plain_byte_array_length_terminates(workdir):
    """A 4 GiB length in a PLAIN BYTE_ARRAY page must not hang the GPU or read out of bounds.

    `pq_plain_bytes_scan` copied the page's 32-bit length fields into the per-value table verbatim,
    and `pq_gather_bytes` then copied that many bytes, one thread, one byte at a time, from wherever
    the offset landed. A single flipped byte was enough to turn a read into an unbounded
    out-of-bounds copy that never returned.
    """
    src = _write(workdir, "ba-plain.parquet",
                 pa.table({"s": pa.array(["marker-value-%04d" % i for i in range(200)])}),
                 compression="none", use_dictionary=False)
    dst = os.path.join(workdir, "ba-plain-corrupt.parquet")
    _corrupt_byte_array_length(src, dst, b"marker-value-0100")
    status, detail = read_in_child(dst)
    assert status in ("READ-OK", "ERROR", "PYERROR"), "%s: %s" % (status, detail)


def test_corrupt_dictionary_byte_array_length_terminates(workdir):
    """The same clamp is needed on the dictionary page, which `pq_dict_bytes_scan` walks."""
    src = _write(workdir, "ba-dict.parquet",
                 pa.table({"s": pa.array(["dictvalue-%03d" % (i % 50) for i in range(400)])}),
                 compression="none", use_dictionary=True)
    dst = os.path.join(workdir, "ba-dict-corrupt.parquet")
    _corrupt_byte_array_length(src, dst, b"dictvalue-017")
    status, detail = read_in_child(dst)
    assert status in ("READ-OK", "ERROR", "PYERROR"), "%s: %s" % (status, detail)


# --------------------------------------------------------------------------------------------
# Hand-built footers. `PAR1 <thrift FileMetaData> <4-byte length> PAR1` is the whole container, so
# a footer can be written by hand and handed to the reader without pyarrow's help.

def _footer_file(path, thrift_bytes):
    body = b"PAR1" + thrift_bytes + struct.pack("<I", len(thrift_bytes)) + b"PAR1"
    open(path, "wb").write(body)
    return path


def _varint(v):
    out = bytearray()
    while True:
        if v < 0x80:
            out.append(v)
            return bytes(out)
        out.append((v & 0x7F) | 0x80)
        v >>= 7


def _zigzag(v):
    return _varint((v << 1) ^ (v >> 63) if v >= 0 else ((-v - 1) * 2 + 1))


def test_deeply_nested_thrift_footer_is_rejected(workdir):
    """A footer nesting structs 60,000 deep must not recurse `skip` off the end of the stack."""
    depth = 60000
    # field id 7 (nothing the reader consumes), type 12 = struct, then `depth` nested structs.
    thrift = bytes([0x7C]) + bytes([0x1C]) * depth + bytes([0x00]) * (depth + 2)
    path = _footer_file(os.path.join(workdir, "deep.parquet"), thrift)
    status, detail = read_in_child(path)
    assert status in ("ERROR", "PYERROR", "READ-OK"), "%s: %s" % (status, detail)


def test_huge_thrift_binary_length_is_rejected(workdir):
    """A string field claiming 2^64-1 bytes must raise: `Int(UInt64)` traps on overflow."""
    # field id 6 (created_by), type 8 = binary, then a 10-byte varint holding 2^64-1.
    thrift = bytes([0x68]) + b"\xff" * 9 + b"\x01" + bytes([0x00])
    path = _footer_file(os.path.join(workdir, "hugelen.parquet"), thrift)
    status, detail = read_in_child(path)
    assert status in ("ERROR", "PYERROR"), "%s: %s" % (status, detail)


def test_huge_thrift_list_count_is_rejected(workdir):
    """A `list<struct>` header claiming 2^64-1 elements must raise rather than trap."""
    # field id 2 (schema), type 9 = list; list header 0xFC = count-in-varint, element type 12.
    thrift = bytes([0x29, 0xFC]) + b"\xff" * 9 + b"\x01" + bytes([0x00])
    path = _footer_file(os.path.join(workdir, "hugelist.parquet"), thrift)
    status, detail = read_in_child(path)
    assert status in ("ERROR", "PYERROR"), "%s: %s" % (status, detail)


def _schema_footer(num_children):
    """FileMetaData with one SchemaElement whose `num_children` is `num_children`."""
    name = b"root"
    element = (bytes([0x48]) + _varint(len(name)) + name          # 4: name
               + bytes([0x15]) + _zigzag(num_children)            # 5: num_children
               + bytes([0x00]))
    return (bytes([0x29, 0x1C])                                   # 2: list<struct> of 1
            + element
            + bytes([0x00]))


def test_num_children_beyond_the_schema_is_rejected(workdir):
    """`num_children` larger than the schema list must not index past the array."""
    path = _footer_file(os.path.join(workdir, "children.parquet"), _schema_footer(500))
    status, detail = read_in_child(path)
    assert status in ("ERROR", "PYERROR"), "%s: %s" % (status, detail)


def test_negative_num_children_is_rejected(workdir):
    """A negative `num_children` must not build a reversed Range, which is a trap."""
    path = _footer_file(os.path.join(workdir, "negchildren.parquet"), _schema_footer(-5))
    status, detail = read_in_child(path)
    assert status in ("ERROR", "PYERROR", "READ-OK"), "%s: %s" % (status, detail)


# --------------------------------------------------------------------------------------------
# Bulk mutation sweep: truncation, magic, footer length, and single-byte damage everywhere.

def test_damaged_files_never_crash_the_process(workdir):
    """220 damaged variants of one file, read back to back in one process.

    Reading them in a single child is the point: a trap or a segfault kills that process, and the
    last `AT <name>` line it printed names the file that did it.
    """
    import random

    base = _write(workdir, "sweep-base.parquet", _base_table(), compression="snappy")
    raw = open(base, "rb").read()
    mlen = struct.unpack("<I", raw[-8:-4])[0]
    footer_start = len(raw) - 8 - mlen

    d = os.path.join(workdir, "sweep")
    os.makedirs(d, exist_ok=True)

    def put(name, data):
        open(os.path.join(d, name + ".parquet"), "wb").write(data)

    for n in (0, 1, 4, 8, 12, 100, len(raw) // 2, len(raw) - 9, len(raw) - 1):
        put("trunc-%08d" % n, raw[:n])
    put("magic-head", b"XXXX" + raw[4:])
    put("magic-tail", raw[:-4] + b"XXXX")
    for v in (0, 1, len(raw), len(raw) * 4, 0x7FFFFFFF, 0xFFFFFFFF, 0x80000000):
        put("flen-%010d" % v, raw[:-8] + struct.pack("<I", v) + b"PAR1")

    rnd = random.Random(20260906)
    for i in range(120):
        b = bytearray(raw)
        for _ in range(rnd.choice([1, 1, 2, 5])):
            b[rnd.randrange(footer_start, len(raw) - 8)] = rnd.randrange(256)
        put("footer-%03d" % i, bytes(b))
    for i in range(80):
        b = bytearray(raw)
        for _ in range(rnd.choice([1, 3])):
            b[rnd.randrange(4, footer_start)] = rnd.randrange(256)
        put("data-%03d" % i, bytes(b))

    env = dict(os.environ, PYTHONPATH=PY_ROOT, MallocScribble="1")
    try:
        r = subprocess.run([sys.executable, "-c", _CHILD_MANY, d],
                           capture_output=True, text=True, timeout=900, env=env)
    except subprocess.TimeoutExpired as e:
        seen = [l for l in (e.stdout or b"").decode(errors="replace").splitlines() if l.startswith("AT ")]
        pytest.fail("read hung on %s" % (seen[-1] if seen else "the first file"))
    seen = [l for l in r.stdout.splitlines() if l.startswith("AT ")]
    assert r.returncode == 0 and "ALL-DONE" in r.stdout, (
        "crashed (rc=%d) on %s: %s" % (r.returncode, seen[-1] if seen else "?",
                                       "\n".join(r.stderr.strip().splitlines()[-3:])))
    assert len(seen) == len([f for f in os.listdir(d) if f.endswith(".parquet")])


# --------------------------------------------------------------------------------------------
# Projection

def test_empty_projection_selects_no_columns(workdir):
    """`columns=[]` means "none of them"; only `columns=None` means "all of them".

    The C entry point folded a NULL column array and a zero count together, so an explicit empty
    projection silently read the whole table.
    """
    path = _write(workdir, "proj.parquet", pa.table({
        "a": pa.array([1, 2, 3], pa.int64()),
        "b": pa.array(["x", "y", "z"]),
    }))
    with am.ParquetFile(path) as f:
        assert f.read(columns=[]) == {}
        assert set(f.read(columns=None)) == {"a", "b"}
        assert list(f.read(columns=["b"])) == ["b"]


def test_projection_keeps_the_requested_order(workdir):
    path = _write(workdir, "order.parquet", pa.table({
        "a": pa.array([1, 2, 3], pa.int64()),
        "b": pa.array(["x", "y", "z"]),
        "c": pa.array([1.0, 2.0, 3.0]),
    }))
    with am.ParquetFile(path) as f:
        assert f.read_table(columns=["c", "a"]).column_names == ["c", "a"]


def test_unknown_column_and_row_group_raise(workdir):
    path = _write(workdir, "bad-proj.parquet", pa.table({"a": pa.array([1, 2, 3], pa.int64())}))
    with am.ParquetFile(path) as f:
        with pytest.raises(am.ArrowMetalError):
            f.read(columns=["nope"])
        with pytest.raises(am.ArrowMetalError):
            f.read(row_groups=[5])


def test_duplicate_projection_names(workdir):
    path = _write(workdir, "dup.parquet", pa.table({"a": pa.array([1, 2, 3], pa.int64())}))
    with am.ParquetFile(path) as f:
        assert f.read_table(columns=["a", "a"]).column_names == ["a", "a"]


def test_duplicate_projection_reads_the_column_twice(workdir):
    """`read` is positional underneath, so a name asked for twice comes back twice, as pyarrow does."""
    path = _write(workdir, "dup2.parquet", pa.table({
        "a": pa.array([1, 2, 3], pa.int64()),
        "b": pa.array(["x", "y", "z"]),
    }))
    with am.ParquetFile(path) as f:
        cols = f.read(columns=["a", "b", "a"])
        assert len(cols) == 3
        assert cols.names == ["a", "b", "a"]
        assert list(cols) == ["a", "b", "a"]
        assert [n for n, _ in cols.items()] == ["a", "b", "a"]
        assert cols[0].to_arrow().to_pylist() == [1, 2, 3]
        assert cols[2].to_arrow().to_pylist() == [1, 2, 3]
        assert cols["a"].to_arrow().to_pylist() == [1, 2, 3]      # by name: the first one
        assert len(cols.columns_named("a")) == 2
        assert f.read_table(columns=["a", "b", "a"]).column_names == ["a", "b", "a"]
    # Same through the module-level entry point.
    assert am.read_parquet(path, columns=["a", "a"]).names == ["a", "a"]
    assert am.read_parquet_table(path, columns=["a", "a"]).column_names == ["a", "a"]


def test_a_file_with_two_columns_of_the_same_name_reads_both(workdir):
    """pyarrow will write one, and dropping the second silently would lose data."""
    table = pa.table([pa.array([1, 2, 3], pa.int64()), pa.array([4, 5, 6], pa.int64())],
                     names=["a", "a"])
    path = _write(workdir, "dupfile.parquet", table)
    with am.ParquetFile(path) as f:
        assert f.column_names == ["a", "a"]
        cols = f.read()
        assert len(cols) == 2
        assert cols.names == ["a", "a"]
        assert cols[0].to_arrow().to_pylist() == [1, 2, 3]
        assert cols[1].to_arrow().to_pylist() == [4, 5, 6]
        t = f.read_table()
        assert t.column_names == ["a", "a"]
        assert t.column(1).to_pylist() == [4, 5, 6]


def test_column_set_still_behaves_like_the_mapping_it_replaces(workdir):
    """Everything that worked when `read` returned a plain dict keeps working."""
    path = _write(workdir, "mapping.parquet", pa.table({
        "a": pa.array([1, 2, 3], pa.int64()),
        "b": pa.array([1.5, 2.5, 3.5]),
    }))
    cols = am.read_parquet(path)
    assert isinstance(cols, dict)
    assert set(cols) == {"a", "b"} and len(cols) == 2
    assert "a" in cols and "zz" not in cols
    assert cols.get("zz") is None
    assert sorted(dict(cols)) == ["a", "b"]
    assert list(cols.values())[0] is cols["a"]
    with pytest.raises(KeyError):
        cols["zz"]
    # A dict of arrays is what query() and write_parquet() take. (The fused expression compiler still
    # wants materialised columns, which is why this reads with dictionary=False; see docs/PARQUET.md.)
    plain = am.read_parquet(path, dictionary=False)
    assert am.query(plain, am.Query().aggregate([("sum", "s", am.col("a"))])) == 6
    out = am.write_parquet(plain, os.path.join(workdir, "mapping-out.parquet"))
    assert pq.read_table(out).column_names == ["a", "b"]


# --------------------------------------------------------------------------------------------
# Two contracts the docs state that the code did not keep.

def test_reduction_on_a_dictionary_encoded_column(workdir):
    """The example in `read_parquet`'s docstring and in docs/PARQUET.md, run as written."""
    path = _write(workdir, "dictred.parquet",
                  pa.table({"price": pa.array([float(i % 97) for i in range(20000)], pa.float64())}),
                  compression="snappy", use_dictionary=True)
    cols = am.read_parquet(path, columns=["price"])
    assert cols["price"].sum() == sum(float(i % 97) for i in range(20000))


def _dictionary_column(workdir, name="dictops.parquet"):
    """A float column pyarrow dictionary-encoded, read back still encoded (the default)."""
    values = [float(i % 7) for i in range(4096)]
    path = _write(workdir, name, pa.table({"v": pa.array(values, pa.float64())}),
                  compression="snappy", use_dictionary=True)
    col = am.read_parquet(path, columns=["v"], dictionary=True)["v"]
    assert pa.types.is_dictionary(col.type), "not dictionary encoded: %s" % col.type
    return values, col


def test_every_entry_point_that_needs_values_decodes_a_dictionary(workdir):
    """One dictionary-encoded column through each C entry point that cannot work on the codes.

    The oracle is the same column materialised: `am_reduce` (sum/min/max/mean), `am_reduce_ex` (the
    statistical aggregates), `am_compare_scalar`, `am_compare_array`, `am_arith_scalar`,
    `am_arith_array`, `am_cast` / `am_cast_ex`, `am_filter_where`, `am_argsort` / `am_argsort_ex`,
    `am_sort`, `am_top_k` and the maths kernels (`am_unary`, `am_binary`, `am_cumulative`) all decode
    first, so every one of them answers what the materialised column answers.

    `sort` (`am_sort_ex`) is the one that decodes only to *order*: it gathers the codes, so it answers
    the same values in a dictionary of its own rather than a decoded column. `test_options.py`'s
    `test_sort_keeps_the_input_type` pins that; here only the values are compared.
    """
    values, col = _dictionary_column(workdir)
    plain = am.array(values)
    listed = lambda a: a.to_arrow().to_pylist()

    assert col.sum() == plain.sum() == pytest.approx(sum(values))        # am_reduce
    assert col.min() == plain.min() and col.max() == plain.max()
    assert col.mean() == pytest.approx(plain.mean())
    assert col.stddev() == pytest.approx(plain.stddev())                 # am_reduce_ex
    assert col.quantile(0.5) == pytest.approx(plain.quantile(0.5))
    assert col.mode() == plain.mode()

    # am_compare_scalar / am_compare_array, with the dictionary on either side.
    assert listed(col > 3.0) == listed(plain > 3.0)
    assert listed(col == plain) == [True] * len(values)
    assert listed(plain == col) == [True] * len(values)
    assert listed(col != plain) == [False] * len(values)
    assert listed(col + 1.0) == listed(plain + 1.0)                      # am_arith_scalar
    assert listed(col + plain) == listed(plain + plain)                  # am_arith_array
    assert listed(plain * col) == listed(plain * plain)
    assert listed(col.cast("int64")) == listed(plain.cast("int64"))      # am_cast_ex
    assert listed(col.filter_where(">", 5.0)) == listed(plain.filter_where(">", 5.0))
    assert listed(col.filter(col > 5.0)) == listed(plain.filter(plain > 5.0))
    assert listed(col.argsort()) == listed(plain.argsort())              # am_argsort_ex
    assert listed(col.argsort(null_placement="at_start")) == \
        listed(plain.argsort(null_placement="at_start"))
    assert listed(col.sort()) == sorted(values)
    # `argsort` / `sort` are the _ex spellings; the plain entry points decode too.
    from arrowmetal import _call, _lib
    assert listed(_call(_lib.am_argsort, col._h, 0)) == listed(plain.argsort())    # am_argsort
    assert listed(_call(_lib.am_sort, col._h, 0)) == sorted(values)                # am_sort
    assert listed(col.top_k(3)) == listed(plain.top_k(3))                # am_top_k
    assert listed(col.abs()) == listed(plain.abs())                      # am_unary
    assert listed(col.power(2.0)) == listed(plain.power(2.0))            # am_binary (scalar)
    assert listed(col.power(plain)) == listed(plain.power(plain))        # am_binary (array)
    assert listed(col.cumulative_sum()) == listed(plain.cumulative_sum())  # am_cumulative


def test_a_dictionary_string_column_is_still_a_dictionary(workdir):
    """Decoding happens only where the codes cannot be used; string kernels keep the encoding."""
    path = _write(workdir, "dictstr.parquet",
                  pa.table({"s": pa.array(["aa", "bb", "aa", "cc"] * 64)}),
                  compression="snappy", use_dictionary=True)
    col = am.read_parquet(path, columns=["s"], dictionary=True)["s"]
    assert pa.types.is_dictionary(col.type)
    assert col.to_arrow().to_pylist() == ["aa", "bb", "aa", "cc"] * 64


def test_argument_errors_set_an_error_message(workdir):
    """`return 2` must leave a message behind, or the C caller reads a stale one."""
    import ctypes
    from arrowmetal import _lib, _P

    path = _write(workdir, "errmsg.parquet", pa.table({"a": pa.array([1, 2, 3], pa.int64())}))
    with am.ParquetFile(path) as f:
        # Seed the thread-local slot with an unrelated failure.
        with pytest.raises(am.ArrowMetalError):
            f.read(columns=["nope"])
        # Now fail on a NULL out-pointer, which is a different failure entirely.
        rc = _lib.am_parquet_read_ex(f._h, None, 0, None, 0, None, 1, None)
        assert rc != 0
        message = (_lib.am_last_error() or b"").decode()
        assert "no column named nope" not in message, "stale message: %r" % message
        assert message, "no message at all for a NULL out-pointer"


def test_every_parquet_entry_point_names_the_argument_it_rejected(workdir):
    """Each `am_parquet_*` failure path names its own function and the argument at fault.

    The message is what the Python layer raises, so a NULL or out-of-range argument has to overwrite
    the thread-local slot rather than leave whatever failed last in it.
    """
    import ctypes
    from arrowmetal import _lib, _P

    path = _write(workdir, "argerr.parquet", pa.table({"a": pa.array([1, 2, 3], pa.int64())}))
    names = (ctypes.c_char_p * 1)(b"a")
    groups = (ctypes.c_int64 * 1)(0)
    out = _P()

    def seed_an_unrelated_failure(f):
        """Leave a message about a different call in the thread-local slot."""
        with pytest.raises(am.ArrowMetalError):
            f.read(columns=["nope"])
        assert "no column named nope" in (_lib.am_last_error() or b"").decode()

    def check(f, name, expect, call):
        seed_an_unrelated_failure(f)
        rc = call()
        assert rc in (-1, 2), "%s: expected a failure, got rc=%r" % (name, rc)
        message = (_lib.am_last_error() or b"").decode()
        assert "no column named nope" not in message, "%s: stale message %r" % (name, message)
        assert name in message, "%s: message does not name the function: %r" % (name, message)
        for word in expect:
            assert word in message, "%s: message does not name %r: %r" % (name, word, message)

    with am.ParquetFile(path) as f:
        cases = [
            # am_parquet_open
            ("am_parquet_open", ["path"], lambda: _lib.am_parquet_open(None, ctypes.byref(out))),
            ("am_parquet_open", ["out"], lambda: _lib.am_parquet_open(str(path).encode(), None)),
            # am_parquet_read_ex, one case per argument it validates
            ("am_parquet_read_ex", ["f"],
             lambda: _lib.am_parquet_read_ex(None, names, 1, None, 0, None, 1, ctypes.byref(out))),
            ("am_parquet_read_ex", ["out"],
             lambda: _lib.am_parquet_read_ex(f._h, names, 1, None, 0, None, 1, None)),
            ("am_parquet_read_ex", ["n_columns"],
             lambda: _lib.am_parquet_read_ex(f._h, names, -1, None, 0, None, 1, ctypes.byref(out))),
            ("am_parquet_read_ex", ["columns", "n_columns"],
             lambda: _lib.am_parquet_read_ex(f._h, None, 2, None, 0, None, 1, ctypes.byref(out))),
            ("am_parquet_read_ex", ["n_row_groups"],
             lambda: _lib.am_parquet_read_ex(f._h, names, 1, groups, -1, None, 1, ctypes.byref(out))),
            ("am_parquet_read_ex", ["row_groups", "n_row_groups"],
             lambda: _lib.am_parquet_read_ex(f._h, names, 1, None, 1, None, 1, ctypes.byref(out))),
            # am_parquet_selected_row_groups
            ("am_parquet_selected_row_groups", ["f"],
             lambda: _lib.am_parquet_selected_row_groups(None, None, None, 0)),
            ("am_parquet_selected_row_groups", ["cap"],
             lambda: _lib.am_parquet_selected_row_groups(f._h, None, None, -1)),
            ("am_parquet_selected_row_groups", ["out", "cap"],
             lambda: _lib.am_parquet_selected_row_groups(f._h, None, None, 4)),
            # the file accessors, whose -1 is a failure like any other
            ("am_parquet_num_rows", ["f"], lambda: _lib.am_parquet_num_rows(None)),
            ("am_parquet_num_row_groups", ["f"], lambda: _lib.am_parquet_num_row_groups(None)),
            ("am_parquet_num_columns", ["f"], lambda: _lib.am_parquet_num_columns(None)),
            ("am_parquet_row_group_rows", ["f"], lambda: _lib.am_parquet_row_group_rows(None, 0)),
            ("am_parquet_row_group_rows", ["row group 7"],
             lambda: _lib.am_parquet_row_group_rows(f._h, 7)),
            # the batch accessors
            ("am_parquet_batch_columns", ["b"], lambda: _lib.am_parquet_batch_columns(None)),
            ("am_parquet_batch_rows", ["b"], lambda: _lib.am_parquet_batch_rows(None)),
            ("am_parquet_batch_column", ["b"],
             lambda: _lib.am_parquet_batch_column(None, 0, ctypes.byref(out))),
        ]
        for name, expect, call in cases:
            check(f, name, expect, call)

        # A live batch, for the two am_parquet_batch_column cases that need one.
        batch = _P()
        assert _lib.am_parquet_read_ex(f._h, names, 1, None, 0, None, 1, ctypes.byref(batch)) == 0
        try:
            check(f, "am_parquet_batch_column", ["out"],
                  lambda: _lib.am_parquet_batch_column(batch, 0, None))
            check(f, "am_parquet_batch_column", ["column index 9"],
                  lambda: _lib.am_parquet_batch_column(batch, 9, ctypes.byref(out)))
        finally:
            _lib.am_parquet_batch_release(batch)

        # The writer half of the ABI validates the same way.
        one = am.array([1, 2, 3])
        handles = (_P * 1)(one._h)
        wnames = (ctypes.c_char_p * 1)(b"a")
        target = os.path.join(workdir, "argerr-out.parquet").encode()
        for expect, call in [
            (["path"], lambda: _lib.am_parquet_write(None, handles, wnames, 1, b"none", 0, 0)),
            (["columns"], lambda: _lib.am_parquet_write(target, None, wnames, 1, b"none", 0, 0)),
            (["names"], lambda: _lib.am_parquet_write(target, handles, None, 1, b"none", 0, 0)),
            (["n_columns"], lambda: _lib.am_parquet_write(target, handles, wnames, 0, b"none", 0, 0)),
        ]:
            check(f, "am_parquet_write", expect, call)


def test_a_bad_filter_string_is_reported_not_ignored(workdir):
    """`selected_row_groups` used to swallow a filter it could not parse and keep every row group."""
    path = _write(workdir, "badfilter.parquet", pa.table({"a": pa.array([1, 2, 3], pa.int64())}))
    with am.ParquetFile(path) as f:
        with pytest.raises(am.ArrowMetalError) as e:
            f.selected_row_groups("a is 3")
        assert "comparison operator" in str(e.value)
