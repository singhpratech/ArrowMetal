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


@pytest.mark.xfail(strict=True, reason="REVIEW: ParquetFile.read returns a dict, so a column named "
                                       "twice collapses to one entry instead of being read twice "
                                       "or rejected")
def test_duplicate_projection_names(workdir):
    path = _write(workdir, "dup.parquet", pa.table({"a": pa.array([1, 2, 3], pa.int64())}))
    with am.ParquetFile(path) as f:
        assert f.read_table(columns=["a", "a"]).column_names == ["a", "a"]
