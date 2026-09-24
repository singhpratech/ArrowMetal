"""Nested Parquet columns, the ARROW:schema metadata and page-index skipping, checked against pyarrow.

Every fixture under Tests/Fixtures/nested (written by Tests/Fixtures/generate_parquet_nested.py with
pyarrow, DuckDB and Polars) is read twice -- by ArrowMetal on the GPU and by
`pyarrow.parquet.read_table` -- and the two must agree value for value and type for type. Where they
cannot agree by design, the difference is spelled out in one place (`KNOWN_TYPE_DIFFERENCES` below) and
a test checks that the difference is exactly that and nothing more.
"""
import glob
import os
import random
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
NESTED = os.path.abspath(os.path.join(HERE, "..", "..", "Tests", "Fixtures", "nested"))


def nested_paths(prefixes=("structs", "maps", "lists", "reqstruct")):
    return sorted(p for p in glob.glob(os.path.join(NESTED, "*.parquet"))
                  if os.path.basename(p).split("__")[0] in prefixes)


def ids(p):
    return os.path.basename(p)[:-8]


# ------------------------------------------------------------------------------ type comparison
#
# The two places ArrowMetal's type differs from pyarrow's by design:
#
# 1. Offsets are 32-bit everywhere in ArrowMetal. A writer that records `large_string`,
#    `large_binary` or `large_list` in its ARROW:schema (Polars does, for every string and list) reads
#    as `string` / `binary` / `list` here, with the same values.
# 2. The members of a struct (and the element of a list) are exported as nullable. A writer that
#    declares a struct member `required` gets `not null` on that member from pyarrow and a nullable
#    member here, with the same values.

def narrow(t):
    """`t` with every 64-bit-offset type replaced by its 32-bit twin, recursively."""
    if pa.types.is_large_string(t):
        return pa.string()
    if pa.types.is_large_binary(t):
        return pa.binary()
    if pa.types.is_large_list(t) or pa.types.is_list(t):
        return pa.list_(pa.field(t.value_field.name, narrow(t.value_type), t.value_field.nullable))
    if pa.types.is_map(t):
        return pa.map_(narrow(t.key_type), pa.field(t.item_field.name, narrow(t.item_type), t.item_field.nullable),
                       t.keys_sorted)
    if pa.types.is_struct(t):
        return pa.struct([pa.field(f.name, narrow(f.type), f.nullable, f.metadata) for f in t])
    return t


def relax(t):
    """`t` with every struct member and list element nullable, recursively."""
    if pa.types.is_list(t):
        return pa.list_(pa.field(t.value_field.name, relax(t.value_type), True))
    if pa.types.is_map(t):
        return pa.map_(relax(t.key_type), pa.field(t.item_field.name, relax(t.item_type), True), t.keys_sorted)
    if pa.types.is_struct(t):
        return pa.struct([pa.field(f.name, relax(f.type), True) for f in t])
    return t


def assert_same_table(got, want):
    assert got.column_names == want.column_names
    assert got.num_rows == want.num_rows
    for name in want.column_names:
        g, w = got[name], want[name]
        assert g.to_pylist() == w.to_pylist(), name
        assert g.type == relax(narrow(w.type)), "%s: %s != %s" % (name, g.type, w.type)


# ------------------------------------------------------------------------ 1-3. nested columns

@pytest.mark.skipif(not nested_paths(), reason="nested fixtures not generated")
@pytest.mark.parametrize("path", nested_paths(), ids=ids)
def test_nested_matches_pyarrow(path):
    assert_same_table(am.read_parquet_table(path), pq.read_table(path))


@pytest.mark.parametrize("path", nested_paths(), ids=ids)
def test_nested_types_match_exactly_outside_the_known_differences(path):
    """Only Polars files (large offsets) and the required-member fixture may differ at all."""
    got, want = am.read_parquet_table(path), pq.read_table(path)
    writer = ids(path).split("__")[1]
    for name in want.column_names:
        g, w = got[name].type, want[name].type
        if writer == "polars":
            assert g == narrow(w), name
        elif ids(path).startswith("reqstruct") and name == "rs":
            assert g != w and g == relax(w), name
        else:
            assert g == w, "%s: %s != %s" % (name, g, w)


def test_required_struct_members_differ_only_in_nullability():
    path = os.path.join(NESTED, "reqstruct__pa_plain_none.parquet")
    want = pq.read_table(path)["rs"].type
    got = am.read_parquet_table(path)["rs"].type
    assert [f.nullable for f in want] == [False, False]
    assert [f.nullable for f in got] == [True, True]
    assert [(f.name, f.type) for f in got] == [(f.name, f.type) for f in want]


def test_polars_files_differ_only_in_offset_width():
    path = os.path.join(NESTED, "structs__polars.parquet")
    want = pq.read_table(path)
    got = am.read_parquet_table(path)
    assert want["s"].type.field("b").type == pa.large_string()
    assert got["s"].type.field("b").type == pa.string()
    assert got["s"].type == narrow(want["s"].type)


def test_every_shape_comes_back_as_its_arrow_type():
    got = am.read_parquet_table(os.path.join(NESTED, "lists__pa_plain_none.parquet"))
    assert got["ll"].type == pa.list_(pa.field("element", pa.list_(pa.field("element", pa.int64()))))
    assert pa.types.is_list(got["lll"].type) and pa.types.is_list(got["lll"].type.value_type)
    assert pa.types.is_struct(got["ls"].type.value_type)
    assert pa.types.is_list(got["sl"].type.field("xs").type)
    assert pa.types.is_map(got["lm"].type.value_type)
    maps = am.read_parquet_table(os.path.join(NESTED, "maps__pa_plain_none.parquet"))
    assert maps["m"].type == pa.map_(pa.string(), pa.int64())
    assert pa.types.is_struct(maps["ms"].type.item_type)
    assert pa.types.is_list(maps["ml"].type.item_type)


@pytest.mark.parametrize("base", ["structs", "maps", "lists"])
def test_row_groups_and_projection(base):
    path = os.path.join(NESTED, base + "__pa_dict_snappy.parquet")
    pf = pq.ParquetFile(path)
    assert pf.metadata.num_row_groups > 1
    f = am.ParquetFile(path)
    cols = [c for c in pf.schema_arrow.names if c != "k"][::-1]
    for g in range(pf.metadata.num_row_groups):
        want = pf.read_row_group(g, columns=cols)
        got = f.read_table(columns=cols, row_groups=[g])
        assert_same_table(got, want)


def test_dictionary_encoded_leaves_inside_nested_columns():
    path = os.path.join(NESTED, "lists__pa_dict_snappy.parquet")
    want = pq.read_table(path, columns=["lls", "ls", "lm"])
    cols = am.read_parquet(path, columns=["lls", "ls", "lm"], dictionary=True)
    for name in ("lls", "ls", "lm"):
        assert cols[name].to_arrow().to_pylist() == want[name].to_pylist(), name


def test_struct_leaves_still_read_flat_by_dotted_path():
    path = os.path.join(NESTED, "structs__pa_plain_none.parquet")
    want = pq.read_table(path)["s"].to_pylist()
    got = am.read_parquet_table(path, columns=["s.b"])["s.b"].to_pylist()
    assert got == [None if v is None else v["b"] for v in want]


_CHILD_MANY = r"""
import os, sys
import arrowmetal as am
for name in sorted(os.listdir(sys.argv[1])):
    sys.stdout.write("AT %s\n" % name); sys.stdout.flush()
    try:
        am.read_parquet_table(os.path.join(sys.argv[1], name))
    except Exception:
        pass
print("ALL-DONE")
"""


def test_damaged_nested_files_never_crash_the_process():
    """Single-byte damage in the pages of nested columns: levels that disagree across leaves, runs
    that overshoot, counts that do not add up. Every read must raise or return; none may crash."""
    rnd = random.Random(20260923)
    with tempfile.TemporaryDirectory() as d:
        n = 0
        for base in ("structs__pa_plain_none", "maps__pa_plain_none", "lists__pa_plain_none",
                     "lists__pa_v2_snappy", "maps__duckdb"):
            raw = open(os.path.join(NESTED, base + ".parquet"), "rb").read()
            mlen = struct.unpack("<I", raw[-8:-4])[0]
            footer = len(raw) - 8 - mlen
            for i in range(40):
                b = bytearray(raw)
                for _ in range(rnd.choice([1, 2, 4])):
                    b[rnd.randrange(4, footer)] = rnd.randrange(256)
                open(os.path.join(d, "%s-%03d.parquet" % (base, i)), "wb").write(bytes(b))
                n += 1
        env = dict(os.environ, PYTHONPATH=PY_ROOT, MallocScribble="1")
        r = subprocess.run([sys.executable, "-c", _CHILD_MANY, d], capture_output=True, text=True,
                           timeout=900, env=env)
        seen = [l for l in r.stdout.splitlines() if l.startswith("AT ")]
        assert r.returncode == 0 and "ALL-DONE" in r.stdout, (
            "crashed (rc=%d) on %s: %s" % (r.returncode, seen[-1] if seen else "?",
                                           "\n".join(r.stderr.strip().splitlines()[-3:])))
        assert len(seen) == n
