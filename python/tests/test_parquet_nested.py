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


def nested_paths(prefixes=("structs", "maps", "lists", "reqstruct", "nullleaves")):
    return sorted(p for p in glob.glob(os.path.join(NESTED, "*.parquet"))
                  if os.path.basename(p).split("__")[0] in prefixes)


def ids(p):
    return os.path.basename(p)[:-8]


# ------------------------------------------------------------------------------ type comparison
#
# The two places ArrowMetal's type differs from pyarrow's by design:
#
# 1. Offsets are 32-bit everywhere in ArrowMetal, and it has no view layouts. A writer that records
#    `large_string`, `large_binary`, `large_list`, `string_view`, `binary_view` or `list_view` in its
#    ARROW:schema (Polars records the large ones for every string and list) reads as `string` /
#    `binary` / `list` here, with the same values.
# 2. The members of a struct (and the element of a list) are exported as nullable. A writer that
#    declares a struct member `required` gets `not null` on that member from pyarrow and a nullable
#    member here, with the same values.

def narrow(t):
    """`t` with every 64-bit-offset or view type replaced by its 32-bit twin, recursively."""
    if pa.types.is_large_string(t) or pa.types.is_string_view(t):
        return pa.string()
    if pa.types.is_large_binary(t) or pa.types.is_binary_view(t):
        return pa.binary()
    if pa.types.is_large_list(t) or pa.types.is_list(t) or pa.types.is_list_view(t) or pa.types.is_large_list_view(t):
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


# --------------------------------------------------------------------------- 4. ARROW:schema

# `arrowschema__pa_types` and `arrowschema__pa_fslnull` have tests of their own below, and so do the two
# files whose ARROW:schema does not decode, which pyarrow refuses to open.
ARROW_SCHEMA = sorted(p for p in glob.glob(os.path.join(NESTED, "arrowschema__*.parquet"))
                      if ids(p) not in ("arrowschema__pa_types", "arrowschema__pa_fslnull",
                                        "arrowschema__duckdb_corrupt", "arrowschema__duckdb_truncated"))


def assert_same_schema_metadata(got, want):
    for name in want.column_names:
        assert got.schema.field(name).metadata == want.schema.field(name).metadata, name
    assert got.schema.metadata == want.schema.metadata


@pytest.mark.skipif(not ARROW_SCHEMA, reason="nested fixtures not generated")
@pytest.mark.parametrize("path", ARROW_SCHEMA, ids=ids)
def test_arrow_schema_matches_pyarrow(path):
    """Values, types, field metadata and schema metadata, as pyarrow.parquet.read_table has them."""
    got, want = am.read_parquet_table(path), pq.read_table(path)
    assert_same_table(got, want)
    for name in want.column_names:
        if "polars" not in path:
            assert got[name].type == want[name].type, name
    assert_same_schema_metadata(got, want)


def test_time_zones_come_back():
    got = am.read_parquet_table(os.path.join(NESTED, "arrowschema__pa_plain_none.parquet"))
    assert got["ts_paris"].type == pa.timestamp("us", tz="Europe/Paris")
    # Parquet has no seconds or other units to lose here, but the zone is restored on the stored unit.
    assert got["ts_ny_ms"].type == pa.timestamp("ms", tz="America/New_York")
    assert got["ts_off_ns"].type == pa.timestamp("ns", tz="+05:30")
    assert got["ts_utc"].type == pa.timestamp("us", tz="UTC")
    assert got["ts_naive"].type == pa.timestamp("us")
    # Inside a struct, a list and a map too.
    assert got["inner_tz"].type.field("t").type == pa.timestamp("ms", tz="Asia/Tokyo")
    assert got["inner_tz"].type.field("d").type == pa.duration("ms")
    assert got["dur_list"].type.value_type == pa.duration("ns")
    assert got["tz_map"].type.item_type == pa.timestamp("us", tz="Australia/Sydney")
    # A column read dictionary-encoded carries the zone on its dictionary.
    enc = am.read_parquet(os.path.join(NESTED, "arrowschema__pa_dict_snappy.parquet"), columns=["ts_paris"],
                          dictionary=True)["ts_paris"].to_arrow()
    assert enc.type == pa.dictionary(pa.int32(), pa.timestamp("us", tz="Europe/Paris"))


def test_null_typed_column_reads_as_null():
    for name in ("arrowschema__pa_plain_none", "arrowschema__pa_dict_snappy"):
        got = am.read_parquet_table(os.path.join(NESTED, name + ".parquet"))
        assert got["nothing"].type == pa.null()
        assert got["nothing"].null_count == got.num_rows == 300
    # The Parquet UNKNOWN annotation is what marks it, so it needs no ARROW:schema at all.
    cols = am.read_parquet(os.path.join(NESTED, "arrowschema__pa_plain_none.parquet"), columns=["nothing"])
    assert cols["nothing"].to_arrow().type == pa.null()


def test_durations_come_back():
    got = am.read_parquet_table(os.path.join(NESTED, "arrowschema__pa_plain_none.parquet"))
    want = pq.read_table(os.path.join(NESTED, "arrowschema__pa_plain_none.parquet"))
    assert got["dur_s"].type == pa.duration("s") and got["dur_us"].type == pa.duration("us")
    assert got["dur_s"].to_pylist() == want["dur_s"].to_pylist()


def test_field_metadata_and_field_id():
    path = os.path.join(NESTED, "arrowschema__pa_plain_none.parquet")
    f = am.ParquetFile(path)
    assert f.field_metadata("price") == {b"currency": b"EUR", b"precision": b"cents"}
    assert f.field_metadata("fid")[b"PARQUET:field_id"] == b"42"
    assert f.field_metadata("fid")[b"note"] == b"has a field id"
    assert f.field_metadata("ts_naive") == {}
    assert f.schema_metadata == {b"source": b"generate_parquet_nested.py"}
    got = f.read_table(columns=["price", "fid"])
    want = pq.read_table(path, columns=["price", "fid"])
    assert_same_schema_metadata(got, want)


def test_extension_types():
    path = os.path.join(NESTED, "arrowschema__pa_plain_none.parquet")
    got, want = am.read_parquet_table(path), pq.read_table(path)
    # arrow.uuid is registered in pyarrow, so both readers hand back the extension type.
    assert got["u"].type == pa.uuid() == want["u"].type
    # example.label is registered only by the generator: both readers see the storage type and keep
    # the extension keys in the field metadata.
    assert got["label"].type == pa.string() == want["label"].type
    md = got.schema.field("label").metadata
    assert md[b"ARROW:extension:name"] == b"example.label" and md[b"owner"] == b"fixtures"
    assert got["rat"].type == want["rat"].type


def test_metadata_on_a_nested_field_is_not_carried():
    """The engine's nested arrays hold no per-field metadata, so metadata on a struct member (or a list
    element, or a map value) is dropped; a top-level field's is kept. The types still compare equal."""
    path = os.path.join(NESTED, "arrowschema__pa_plain_none.parquet")
    got, want = am.read_parquet_table(path, columns=["inner_meta"]), pq.read_table(path, columns=["inner_meta"])
    assert want["inner_meta"].type.field("x").metadata == {b"inner": b"kept by pyarrow"}
    assert got["inner_meta"].type.field("x").metadata is None
    assert got["inner_meta"].type == want["inner_meta"].type
    assert got["inner_meta"].to_pylist() == want["inner_meta"].to_pylist()


class _Label(pa.ExtensionType):
    def __init__(self):
        super().__init__(pa.string(), "example.label")

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return cls()


def test_registered_extension_type_is_rebuilt():
    path = os.path.join(NESTED, "arrowschema__pa_plain_none.parquet")
    pa.register_extension_type(_Label())
    try:
        got, want = am.read_parquet_table(path, columns=["label"]), pq.read_table(path, columns=["label"])
        assert isinstance(got["label"].type, _Label) and isinstance(want["label"].type, _Label)
        assert got.schema.field("label").metadata == want.schema.field("label").metadata == {b"owner": b"fixtures"}
        assert got["label"].to_pylist() == want["label"].to_pylist()
    finally:
        pa.unregister_extension_type("example.label")


@pytest.mark.parametrize("name", ["arrowschema__pa_nostore", "arrowschema__duckdb_nostore"])
def test_absent_arrow_schema(name):
    """No ARROW:schema: the file reads as the Parquet schema alone describes it, exactly as pyarrow reads it."""
    path = os.path.join(NESTED, name + ".parquet")
    got, want = am.read_parquet_table(path), pq.read_table(path)
    assert b"ARROW:schema" not in open(path, "rb").read()
    assert_same_table(got, want)
    assert got["ts_paris"].type == pa.timestamp("us", tz="UTC") == want["ts_paris"].type


@pytest.mark.parametrize("kind,value", [("corrupt", b"not base64 at all!"), ("truncated", b"/////w==")])
def test_malformed_arrow_schema_is_ignored(kind, value):
    """An ARROW:schema that is not base64, or is base64 of a truncated message. pyarrow refuses to open
    such a file; ArrowMetal reads it as the same columns without the key read, keeping the undecodable
    value in the schema metadata, since it was not applied."""
    path = os.path.join(NESTED, "arrowschema__duckdb_%s.parquet" % kind)
    raw = open(path, "rb").read()
    assert b"ARROW:schema" in raw and value in raw
    with pytest.raises(pa.ArrowInvalid):
        pq.read_table(path)
    got = am.read_parquet_table(path)
    want = pq.read_table(os.path.join(NESTED, "arrowschema__duckdb_nostore.parquet"))
    assert_same_table(got, want)
    assert got["ts_paris"].type == pa.timestamp("us", tz="UTC")
    assert got.schema.metadata == {b"ARROW:schema": value}


def test_stored_schema_of_another_width_is_ignored_like_pyarrow():
    """Two or nine stored fields against five columns: pyarrow ignores the stored schema and keeps the key
    in the schema metadata; five fields apply by position whatever the names say."""
    for kind in ("fewer", "more"):
        path = os.path.join(NESTED, "arrowschema__duckdb_%s.parquet" % kind)
        got, want = am.read_parquet_table(path), pq.read_table(path)
        assert got["id"].type == want["id"].type == pa.int64()
        assert got["ts_paris"].type == want["ts_paris"].type == pa.timestamp("us", tz="UTC")
        assert b"ARROW:schema" in got.schema.metadata
        assert got.schema.metadata == want.schema.metadata
    path = os.path.join(NESTED, "arrowschema__duckdb_renamed.parquet")
    got, want = am.read_parquet_table(path), pq.read_table(path)
    assert got["id"].type == want["id"].type == pa.duration("s")
    assert got["ts_paris"].type == want["ts_paris"].type == pa.timestamp("us", tz="Asia/Tokyo")


def test_dictionary_claim_over_a_struct_is_ignored():
    path = os.path.join(NESTED, "arrowschema__duckdb_dictstruct.parquet")
    got, want = am.read_parquet_table(path), pq.read_table(path)
    assert pa.types.is_struct(got["s"].type) and got["s"].type == want["s"].type
    assert got["s"].to_pylist() == want["s"].to_pylist()


@pytest.mark.parametrize("name", ["arrowschema__pa_categoricals", "arrowschema__pa_categoricals_plain"])
def test_only_string_and_binary_categoricals_are_restored(name):
    """pyarrow restores a stored dictionary type only over string and binary values; a categorical of
    integers, timestamps or dates reads back as its value type. ArrowMetal does the same."""
    path = os.path.join(NESTED, name + ".parquet")
    got, want = am.read_parquet_table(path), pq.read_table(path)
    for col in want.column_names:
        assert got[col].type == want[col].type, col
        assert got[col].to_pylist() == want[col].to_pylist(), col
    assert got["cat_str"].type == pa.dictionary(pa.int32(), pa.string())
    assert got["cat_bin"].type == pa.dictionary(pa.int32(), pa.binary())
    assert got["cat_int"].type == pa.int64()
    assert got["cat_ts"].type == pa.timestamp("us", tz="UTC")
    assert got["cat_date"].type == pa.date32()


def test_arrow_types_parquet_cannot_name():
    """decimal32 / decimal64, fixed_size_list, a dictionary (categorical) column and a zone on a
    seconds timestamp come back as their stored Arrow types; the view and 64-bit-offset layouts come
    back as their 32-bit, non-view twins with the same values -- and nothing else differs."""
    path = os.path.join(NESTED, "arrowschema__pa_types.parquet")
    got, want = am.read_parquet_table(path), pq.read_table(path)
    for name in want.column_names:
        assert got[name].to_pylist() == want[name].to_pylist(), name
    assert got["d32"].type == pa.decimal32(7, 2) == want["d32"].type
    assert got["d64"].type == pa.decimal64(15, 3) == want["d64"].type
    assert got["fsl"].type == want["fsl"].type == pa.list_(pa.int32(), 3)
    assert got["cat"].type == want["cat"].type == pa.dictionary(pa.int32(), pa.string())
    assert got["ts_s_tz"].type == want["ts_s_tz"].type == pa.timestamp("ms", tz="Europe/Berlin")
    assert (want["sv"].type, got["sv"].type) == (pa.string_view(), pa.string())
    assert (want["ls"].type, got["ls"].type) == (pa.large_string(), pa.string())
    assert (want["ll"].type, got["ll"].type) == (pa.large_list(pa.field("element", pa.int64())), pa.list_(pa.int64()))
    assert (want["lv"].type, got["lv"].type) == (pa.list_view(pa.field("element", pa.int64())), pa.list_(pa.int64()))
    for name in want.column_names:
        assert got[name].type == narrow(want[name].type), name


def test_fixed_size_list_with_null_rows():
    """pyarrow 25 writes this file but its own reader rejects it; ArrowMetal reads the fixed-size
    list back with its null rows (each padded with three null child slots, as the layout requires)."""
    path = os.path.join(NESTED, "arrowschema__pa_fslnull.parquet")
    with pytest.raises(pa.ArrowInvalid):
        pq.read_table(path)
    got = am.read_parquet_table(path)["fsl"]
    assert got.type == pa.list_(pa.int32(), 3)
    assert got.to_pylist() == [None if i % 7 == 0 else [i, -i, None if i % 5 == 0 else i * 2] for i in range(60)]


# --------------------------------------------------------------- 5. page-level skipping

import pyarrow.compute as pc  # noqa: E402

PAGE_INDEX = sorted(glob.glob(os.path.join(NESTED, "pageindex__*.parquet")))
_OPS = {"==": pc.equal, "!=": pc.not_equal, "<": pc.less, "<=": pc.less_equal, ">": pc.greater,
        ">=": pc.greater_equal}
FILTERS = [
    [("id", ">", 20000)], [("id", "<", 500)], [("id", "==", 12017)], [("id", "!=", 17)],
    [("i32", "==", 77)], [("i32", ">=", 140)], [("f64", ">", 150.0)], [("f64", "<=", -200.0)],
    [("cat", "==", "c0010")], [("cat", "<", "c0003")], [("v", "<", 3.0)], [("v", "==", 996.0)],
    [("id", ">=", 10000), ("i32", "<", 120)], [("noise", "==", 5)],
    # Row group 1 passes both row-group statistics, but no page passes both filters.
    [("id", ">=", 20000), ("i32", "<", 70)],
]


def _exact(table, flt):
    mask = None
    for col, op, val in flt:
        m = _OPS[op](table[col], val)
        mask = m if mask is None else pc.and_(mask, m)
    return table.filter(mask)


@pytest.mark.skipif(not PAGE_INDEX, reason="nested fixtures not generated")
@pytest.mark.parametrize("path", PAGE_INDEX, ids=ids)
@pytest.mark.parametrize("flt", FILTERS, ids=lambda f: ";".join("%s%s%s" % t for t in f))
def test_page_skipping_gives_identical_results(path, flt):
    """The rows that match are the same with and without the page index, and the same as pyarrow's;
    the page-index read returns no more rows than the row-group-granular one."""
    f = am.ParquetFile(path)
    f.use_page_index = True
    with_index = f.read_table(filters=flt)
    stats = f.last_read_stats
    f.use_page_index = False
    without = f.read_table(filters=flt)
    assert f.last_read_stats["pages_skipped"] == 0
    want = _exact(pq.read_table(path), flt)
    assert _exact(with_index, flt).to_pylist() == _exact(without, flt).to_pylist() == want.to_pylist()
    assert with_index.num_rows <= without.num_rows
    assert stats["rows"] == with_index.num_rows
    # Every column of the page-index read covers the same rows, nested ones included.
    assert with_index.column_names == without.column_names


def test_pages_are_skipped_and_counted():
    path = os.path.join(NESTED, "pageindex__pa_plain_none.parquet")
    f = am.ParquetFile(path)
    f.use_page_index = False
    f.read_table(columns=["id", "f64", "cat"], filters=[("id", "==", 12017)])
    whole = f.last_read_stats
    f.use_page_index = True
    got = f.read_table(columns=["id", "f64", "cat"], filters=[("id", "==", 12017)])
    part = f.last_read_stats
    assert whole["pages_skipped"] == 0 and part["pages_skipped"] > 0
    # Flat columns: every page of the row groups read is either decoded or skipped.
    assert part["pages_decoded"] + part["pages_skipped"] == whole["pages_decoded"]
    assert part["pages_decoded"] < whole["pages_decoded"] // 2
    assert part["rows"] < whole["rows"]
    assert 12017 in got["id"].to_pylist()


def test_a_row_group_can_be_ruled_out_by_pages_alone():
    path = os.path.join(NESTED, "pageindex__pa_plain_none.parquet")
    f = am.ParquetFile(path)
    flt = [("id", ">=", 20000), ("i32", "<", 70)]
    assert 1 in f.selected_row_groups(flt)          # the row-group statistics keep it
    f.read_table(columns=["id"], filters=flt)
    assert f.last_read_stats["row_groups_skipped_by_page_index"] >= 1


NAN_FILTERS = [
    [("f64", ">", 100.0)], [("f64", "<", -150.0)], [("f64", "<=", 0.0)], [("f32", "<", 5.0)],
    [("f32", ">=", 50.0), ("id", "<", 3500)], [("f64", ">=", 0.0), ("f64", "<=", 1.0)],
]


@pytest.mark.parametrize("name", ["pageindexnan__polars", "pageindexnan__pa_plain_none"])
@pytest.mark.parametrize("flt", NAN_FILTERS, ids=lambda f: ";".join("%s%s%s" % t for t in f))
def test_nan_pages_are_never_skipped_wrongly(name, flt):
    """Polars marks each page holding a NaN as a null page with a null count of 0. Such a page holds
    matching rows; the page-index read must keep it and agree with the read without the index and with
    the exact matches of the whole file."""
    path = os.path.join(NESTED, name + ".parquet")
    f = am.ParquetFile(path)
    f.use_page_index = True
    with_index = f.read_table(filters=flt)
    f.use_page_index = False
    without = f.read_table(filters=flt)
    want = _exact(pq.read_table(path), flt)
    assert want.num_rows > 0
    assert _exact(with_index, flt).to_pylist() == _exact(without, flt).to_pylist() == want.to_pylist()
    # pyarrow's own filtered read trusts Polars' row-group min / max, which leave the flagged pages out:
    # it returns every match or, when the filter falls outside the unflagged pages' range, none.
    by_pyarrow = _exact(pq.read_table(path, filters=flt), flt).num_rows
    if name.endswith("pa_plain_none"):
        assert by_pyarrow == want.num_rows
    else:
        assert by_pyarrow in (want.num_rows, 0)



@pytest.mark.parametrize("col", ["f64", "f32"])
def test_not_equal_keeps_a_nan_hidden_in_a_constant_page(col):
    """Pages of 64 rows; the first is 5.0 everywhere but one NaN (row 10). pyarrow leaves NaN out of a
    page's min / max, so its index entry says 5.0 .. 5.0, like the all-5.0 page after it. NaN != 5.0 is
    true, so `!= 5.0` must keep that page: the read with the index, the read without it and pyarrow's
    filtered read all return the same rows."""
    path = os.path.join(NESTED, "pageindexnan__pa_constpage.parquet")
    ci = pq.ParquetFile(path).metadata.row_group(0).column(1)
    assert ci.has_column_index
    flt = [(col, "!=", 5.0)]
    f = am.ParquetFile(path)
    f.use_page_index = True
    on = _exact(f.read_table(filters=flt), flt)
    f.use_page_index = False
    off = _exact(f.read_table(filters=flt), flt)
    want = _exact(pq.read_table(path), flt)
    assert 10 in on["id"].to_pylist()
    assert on["id"].to_pylist() == off["id"].to_pylist() == want["id"].to_pylist()
    assert pq.read_table(path, filters=flt).num_rows == want.num_rows
    # An integer column has no NaN: its two constant pages are still skipped.
    f.use_page_index = True
    got = f.read_table(columns=["id", "i64"], filters=[("i64", "!=", 5)])
    assert f.last_read_stats["pages_skipped"] >= 2
    assert _exact(got, [("i64", "!=", 5)]).to_pylist() == _exact(pq.read_table(path, columns=["id", "i64"]), [("i64", "!=", 5)]).to_pylist()


def test_not_equal_keeps_a_nan_row_group_where_pyarrow_drops_it():
    """The same at row-group level, where pyarrow's filtered read differs: a row group of 5.0 with one
    NaN has min == max == 5.0 in its statistics, and `pyarrow.parquet.read_table(filters=[("x", "!=",
    5.0)])` rules it out, dropping the NaN row that `pyarrow.compute.not_equal` keeps. ArrowMetal keeps
    the row group and returns the exact matches."""
    import numpy as np
    x = np.full(256, 5.0)
    x[10] = np.nan
    x = np.concatenate([x, np.arange(256.0)])
    tbl = pa.table({"id": pa.array(range(len(x)), pa.int64()), "x": pa.array(x)})
    flt = [("x", "!=", 5.0)]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "nan_rg.parquet")
        pq.write_table(tbl, p, row_group_size=256, use_dictionary=False)
        st = pq.ParquetFile(p).metadata.row_group(0).column(1).statistics
        assert (st.min, st.max) == (5.0, 5.0)
        want = _exact(pq.read_table(p), flt)
        by_pyarrow = pq.read_table(p, filters=flt)
        f = am.ParquetFile(p)
        assert f.selected_row_groups(flt) == [0, 1]
        got = _exact(f.read_table(filters=flt), flt)
        assert got["id"].to_pylist() == want["id"].to_pylist()
        assert 10 in got["id"].to_pylist() and 10 not in by_pyarrow["id"].to_pylist()
        assert by_pyarrow.num_rows == want.num_rows - 1


@pytest.mark.parametrize("op,value", [(">=", 2**63), ("==", 2**63 + 7), (">", 2**63 + 4990), ("<", 2**63 + 10),
                                      ("!=", 2**63 + 7), ("<=", 2**64 - 1), (">", 2**64 + 5), ("<", 0),
                                      (">", 9.3e18), (">=", 0)])
def test_uint64_literals_past_the_signed_range(op, value):
    """uint64 values 2^63 .. 2^63 + 4999 in five row groups. The statistics are unsigned and a literal
    above the int64 range stays exact, so the row groups kept are those that can match, and the rows are
    pyarrow's."""
    path = os.path.join(NESTED, "unsigned__pa_plain_none.parquet")
    flt = [("u64", op, value)]
    import operator
    pyop = {"==": operator.eq, "!=": operator.ne, "<": operator.lt, "<=": operator.le, ">": operator.gt,
            ">=": operator.ge}[op]
    table = pq.read_table(path)
    # Python compares an int with an int or a float exactly.
    want = [i for i, u in zip(table["id"].to_pylist(), table["u64"].to_pylist()) if pyop(u, value)]
    f = am.ParquetFile(path)
    got = f.read_table(filters=flt)
    got_exact = [i for i, u in zip(got["id"].to_pylist(), got["u64"].to_pylist()) if pyop(u, value)]
    assert got_exact == want
    if isinstance(value, int) and 0 <= value < 2**64:
        by_pyarrow = pq.read_table(path, filters=[("u64", op, pa.scalar(value, pa.uint64()))])
        assert by_pyarrow["id"].to_pylist() == want
    kept = f.selected_row_groups(flt)
    need = sorted({i // 1000 for i in want})
    assert set(need) <= set(kept)
    if op != "!=":
        assert kept == need


def test_null_type_below_lists_maps_and_structs():
    """A pyarrow `list<null>` (Parquet UNKNOWN below a LIST) used to fail the import with "Length spanned by
    list offsets larger than values array"; every null-typed leaf inside a nested column now has a slot
    per element."""
    for v in ("pa_plain_none", "pa_dict_snappy", "pa_v2_snappy", "pa_v2_lz4"):
        path = os.path.join(NESTED, "nullleaves__%s.parquet" % v)
        got, want = am.read_parquet_table(path), pq.read_table(path)
        assert_same_table(got, want)
        assert got["ln"].type == want["ln"].type == pa.list_(pa.field("element", pa.null()))
        assert pa.types.is_null(got["mn"].type.item_type)
        for name in ("ln", "lln", "lsn", "sln"):
            one = am.read_parquet_table(path, columns=[name])
            assert one[name].to_pylist() == want[name].to_pylist(), name


# The one place a restored dictionary type differs from pyarrow's: ArrowMetal's dictionary arrays have
# int32 indices and no ordered flag, so a stored dictionary<int8 | uint8 | uint32, ..., ordered> comes back
# as dictionary<int32, ...>, unordered, with the same values.
@pytest.mark.parametrize("name,expect", [
    ("catwidth__pandas", {"c": (pa.int8(), False), "o": (pa.int8(), True)}),
    ("catwidth__polars", {"cat": (pa.uint32(), False), "enum": (pa.uint8(), True)}),
])
def test_restored_dictionaries_have_int32_indices_and_no_ordered_flag(name, expect):
    path = os.path.join(NESTED, name + ".parquet")
    got, want = am.read_parquet_table(path), pq.read_table(path)
    for col, (index, ordered) in expect.items():
        w, g = want[col].type, got[col].type
        assert (w.index_type, w.ordered) == (index, ordered), col
        assert g == pa.dictionary(pa.int32(), pa.string()) and not g.ordered, col
        assert narrow(w.value_type) == g.value_type == pa.string(), col
        assert got[col].to_pylist() == want[col].to_pylist(), col
        # Everything but the index type and the flag is pyarrow's.
        assert got[col].cast(pa.string()).to_pylist() == want[col].cast(pa.string()).to_pylist(), col


def test_string_literals_with_quotes_semicolons_and_backslashes():
    """A string filter value travels to the C parser quoted, with `"` and `\\` escaped, so a quote, a
    semicolon, a backslash or an operator inside it is part of the value."""
    vals = ['a"b;c', 'x;y', '\\', 'p\\"q', 'k==v', 'plain', '"', ';', 'end\\', '<>!=']
    n = 400
    tbl = pa.table({"id": pa.array(range(n), pa.int64()), "s": pa.array([vals[i // 40] for i in range(n)])})
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "quotes.parquet")
        pq.write_table(tbl, p, row_group_size=40, write_page_index=True)
        f = am.ParquetFile(p)
        for k, v in enumerate(vals):
            for op in ("==", "!=", "<"):
                flt = [("s", op, v), ("id", ">=", 0)]
                got = _exact(f.read_table(filters=flt), flt)
                want = _exact(tbl, flt)
                assert got.to_pylist() == want.to_pylist(), (v, op)
            assert f.selected_row_groups([("s", "==", v)]) == [k], v
        with pytest.raises(am.ArrowMetalError, match="column name cannot contain"):
            f.read_table(filters=[("a;b", "==", 1)])


def test_a_stored_field_nested_past_64_levels_is_ignored_like_pyarrow():
    """A stored Arrow field nested 70 levels deep is kept to 64 levels and does not match its int64
    column, so that column reads as the Parquet schema says, the other stored fields still apply, and the
    schema metadata is pyarrow's (no ARROW:schema key)."""
    import base64
    duckdb = pytest.importorskip("duckdb")
    n = 50
    base = pa.table({"i": pa.array(range(n), pa.int64()),
                     "ts": pa.array([k * 1000 for k in range(n)], pa.timestamp("us", tz="UTC"))})
    deep = pa.int32()
    for _ in range(70):
        deep = pa.struct([pa.field("c", deep)])
    stored = pa.schema([pa.field("i", deep), pa.field("ts", pa.timestamp("us", tz="Asia/Tokyo"))])
    kv = base64.b64encode(stored.serialize().to_pybytes()).decode()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "deep.parquet")
        con = duckdb.connect()
        con.register("t", base)
        con.execute("COPY (SELECT * FROM t) TO '%s' (FORMAT parquet, KV_METADATA {'ARROW:schema': '%s'})" % (p, kv))
        con.close()
        got, want = am.read_parquet_table(p), pq.read_table(p)
        assert_same_table(got, want)
        assert got["i"].type == want["i"].type == pa.int64()
        assert got["ts"].type == want["ts"].type == pa.timestamp("us", tz="Asia/Tokyo")
        assert got.schema.metadata == want.schema.metadata


def test_polars_row_group_statistics_leave_nan_pages_out():
    """The one place ArrowMetal and pyarrow's filtered read differ on these files: Polars' row-group
    min is the least value of the pages it did not flag, so `f64 < -150` rules the row group out for
    pyarrow, which returns no rows, while ArrowMetal sees the flagged pages in the column index, keeps
    the row group and returns every match."""
    path = os.path.join(NESTED, "pageindexnan__polars.parquet")
    flt = [("f64", "<", -150.0)]
    stats = pq.ParquetFile(path).metadata.row_group(0).column(1).statistics
    exact = _exact(pq.read_table(path), flt).num_rows
    assert stats.min > -150.0 and exact > 0
    assert pq.read_table(path, filters=flt).num_rows == 0
    f = am.ParquetFile(path)
    assert f.selected_row_groups(flt) == [0]
    assert _exact(f.read_table(filters=flt), flt).num_rows == exact


def test_polars_flags_nan_pages_as_null_pages():
    """The fixture really has the shape the test above guards against: pages flagged null with a zero
    null count, and a column (f32) with no row-group statistics to fall back on."""
    path = os.path.join(NESTED, "pageindexnan__polars.parquet")
    md = pq.ParquetFile(path).metadata.row_group(0)
    assert md.column(1).statistics.null_count == 0
    assert md.column(2).statistics is None or not md.column(2).statistics.has_min_max
    f = am.ParquetFile(path)
    f.read_table(columns=["id"], filters=[("f64", ">", 100.0)])
    stats = f.last_read_stats
    # Pages without a NaN still skip; the flagged ones are decoded.
    assert stats["pages_skipped"] > 0 and stats["pages_decoded"] > 2


def test_filter_values_of_other_types_raise():
    """A date, datetime or Decimal has no filter text; it used to rule out every row group without a
    word. It now raises, and the stored integer filters a date or timestamp column the way pyarrow's
    date and datetime values do."""
    import datetime
    import decimal
    path = os.path.join(NESTED, "pageindex__pa_plain_none.parquet")
    for val in (datetime.date(2021, 1, 1), datetime.datetime(2021, 1, 1), decimal.Decimal("3.5")):
        with pytest.raises(am.ArrowMetalError, match="must be a str, bool, int or float"):
            am.read_parquet_table(path, filters=[("id", ">", val)])
    # numpy scalars are numbers.
    import numpy as np
    got = am.read_parquet_table(path, filters=[("id", "<", np.int64(500))])
    assert _exact(got, [("id", "<", 500)]).num_rows == 97
    tbl = pa.table({"d": pa.array(range(18_000, 22_000), pa.date32()),
                    "t": pa.array([v * 86_400_000_000 for v in range(18_000, 22_000)], pa.timestamp("us"))})
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "dates.parquet")
        pq.write_table(tbl, p, row_group_size=500)
        day = (datetime.date(2021, 1, 1) - datetime.date(1970, 1, 1)).days
        want = pq.read_table(p, filters=[("d", ">=", datetime.date(2021, 1, 1))])
        f = am.ParquetFile(p)
        got = f.read_table(filters=[("d", ">=", day)])
        assert f.last_read_stats["row_groups_skipped_by_statistics"] > 0
        assert _exact(got, [("d", ">=", datetime.date(2021, 1, 1))]).to_pylist() == want.to_pylist()
        got = f.read_table(filters=[("t", "<", day * 86_400_000_000)])
        want = pq.read_table(p, filters=[("t", "<", datetime.datetime(2021, 1, 1))])
        assert _exact(got, [("t", "<", pa.scalar(day * 86_400_000_000, pa.timestamp("us")))]).to_pylist() == want.to_pylist()


def test_no_index_means_no_page_skipping():
    path = os.path.join(NESTED, "pageindex__pa_noindex.parquet")
    f = am.ParquetFile(path)
    f.read_table(filters=[("id", "==", 12017)])
    assert f.last_read_stats["pages_skipped"] == 0


def test_dictionary_encoded_and_nested_columns_are_trimmed_too():
    path = os.path.join(NESTED, "pageindex__pa_dict_snappy.parquet")
    flt = [("id", "<", 500)]
    cols = am.read_parquet(path, columns=["cat", "xs", "id"], filters=flt, dictionary=True)
    ids_ = cols["id"].to_arrow().to_pylist()
    n = len(ids_)
    assert n < 2500
    cat = cols["cat"].to_arrow()
    assert pa.types.is_dictionary(cat.type) and len(cat) == n
    xs = cols["xs"].to_arrow()
    assert len(xs) == n
    want = pq.read_table(path, columns=["id", "cat", "xs"]).to_pylist()
    by_id = {r["id"]: r for r in want}
    assert [by_id[i]["xs"] for i in ids_] == xs.to_pylist()
    assert [by_id[i]["cat"] for i in ids_] == cat.cast(pa.string()).to_pylist()


_CHILD_FILTERED = r"""
import os, sys
import arrowmetal as am
for name in sorted(os.listdir(sys.argv[1])):
    sys.stdout.write("AT %s\n" % name); sys.stdout.flush()
    for flt in ([("id", ">", 20000)], [("cat", "==", "c0010"), ("i32", "<", 90)], [("v", "<", 3.0)]):
        try:
            am.read_parquet_table(os.path.join(sys.argv[1], name), filters=flt)
        except Exception:
            pass
print("ALL-DONE")
"""


def test_damaged_page_indexes_never_crash_the_process():
    """Single-byte damage in the column and offset indexes (the bytes between the last page and the
    footer), read with filters so the indexes are used: every read must raise or return."""
    rnd = random.Random(20260924)
    with tempfile.TemporaryDirectory() as d:
        n = 0
        for base in ("pageindex__pa_plain_none", "pageindex__pa_v2_snappy", "pageindex__polars"):
            path = os.path.join(NESTED, base + ".parquet")
            raw = open(path, "rb").read()
            md = pq.ParquetFile(path).metadata
            pages_end = max(md.row_group(g).column(c).data_page_offset + md.row_group(g).column(c).total_compressed_size
                            for g in range(md.num_row_groups) for c in range(md.num_columns))
            footer = len(raw) - 8 - struct.unpack("<I", raw[-8:-4])[0]
            assert pages_end < footer
            for i in range(60):
                b = bytearray(raw)
                for _ in range(rnd.choice([1, 2, 6])):
                    b[rnd.randrange(pages_end, footer)] = rnd.randrange(256)
                open(os.path.join(d, "%s-%03d.parquet" % (base, i)), "wb").write(bytes(b))
                n += 1
        env = dict(os.environ, PYTHONPATH=PY_ROOT, MallocScribble="1")
        r = subprocess.run([sys.executable, "-c", _CHILD_FILTERED, d], capture_output=True, text=True,
                           timeout=900, env=env)
        seen = [l for l in r.stdout.splitlines() if l.startswith("AT ")]
        assert r.returncode == 0 and "ALL-DONE" in r.stdout, (
            "crashed (rc=%d) on %s: %s" % (r.returncode, seen[-1] if seen else "?",
                                           "\n".join(r.stderr.strip().splitlines()[-3:])))
        assert len(seen) == n


# Strings whose UTF-8 byte order differs from Unicode canonical order: a composed and a decomposed
# accent (canonically equal, different bytes), upper case before lower case, the Angstrom sign and the
# two spellings of Å, a ligature, U+FFFD and an emoji (four UTF-8 bytes, above every BMP character).
_NON_ASCII = ["Z", "a", "é", "é", "\U0001F600", "ﬁ", "�", "zz", "Å", "Å",
              "Å", "été", "été"]


@pytest.mark.parametrize("layout", ["row_groups", "pages"])
def test_string_statistics_are_ordered_by_bytes(layout):
    """Parquet orders BYTE_ARRAY statistics by unsigned byte comparison. Every filter on a column of
    non-ASCII strings returns the rows pyarrow's filtered read returns and the exact matches, whether
    the statistics are the row groups' (one row each) or the pages' column index (one row per page)."""
    table = pa.table({"s": _NON_ASCII, "i": list(range(len(_NON_ASCII)))})
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "strings.parquet")
        if layout == "row_groups":
            pq.write_table(table, path, row_group_size=1, use_dictionary=False)
        else:
            pq.write_table(table, path, use_dictionary=False, write_page_index=True, max_rows_per_page=1)
        f = am.ParquetFile(path)
        f.use_page_index = layout == "pages"
        whole = pq.read_table(path)
        for lit in _NON_ASCII:
            for op in _OPS:
                flt = [("s", op, lit)]
                want = _exact(whole, flt).to_pylist()
                assert _exact(f.read_table(filters=flt), flt).to_pylist() == want, (ascii(lit), op)
                assert _exact(pq.read_table(path, filters=flt), flt).to_pylist() == want
        # The statistics still rule groups and pages out: one value matches, the rest are skipped.
        got = f.read_table(filters=[("s", "==", "é")])
        st = f.last_read_stats
        assert got["s"].to_pylist() == ["é"]
        if layout == "row_groups":
            assert st["row_groups_skipped_by_statistics"] == len(_NON_ASCII) - 1
        else:
            assert st["pages_skipped"] >= len(_NON_ASCII) - 1


def test_int64_statistics_against_a_double_literal_near_2_pow_53():
    """An int64 statistic and a double literal are compared exactly, never through a rounded double:
    2^53 + 1 is greater than 2^53 as a double even though it rounds to it. pyarrow refuses these
    filters, so the reference is the exact comparison Python makes between an int and a float."""
    import operator
    ints = [2 ** 53 - 1, 2 ** 53, 2 ** 53 + 1, 2 ** 53 + 2, 2 ** 53 + 3, -(2 ** 53) - 1]
    py_ops = {"==": operator.eq, "!=": operator.ne, "<": operator.lt, "<=": operator.le, ">": operator.gt,
              ">=": operator.ge}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ints.parquet")
        pq.write_table(pa.table({"x": pa.array(ints, pa.int64())}), path, row_group_size=1,
                       write_page_index=True)
        f = am.ParquetFile(path)
        for lit in [2.0 ** 53, 2.0 ** 53 + 2, 9007199254740991.5, -(2.0 ** 53), 2.0 ** 53 + 4, 2.0 ** 63,
                    -(2.0 ** 63), 1e300]:
            for op, fn in py_ops.items():
                want = [v for v in ints if fn(v, lit)]
                got = f.read_table(filters=[("x", op, lit)])["x"].to_pylist()
                assert [v for v in got if fn(v, lit)] == want, (lit, op)
                if op == "==":
                    assert len(got) == len(want)    # only the row group holding the value is read


def test_decimal_statistics_stored_as_integers_are_not_compared_unscaled():
    """A decimal stored as INT32 / INT64 has its unscaled integer as statistics (12345 for 123.45);
    comparing that with a literal in the column's units ruled out a row group that matches."""
    import decimal
    table = pa.table({"x": pa.array([decimal.Decimal("123.45"), decimal.Decimal("300.00")], pa.decimal128(9, 2))})
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dec.parquet")
        pq.write_table(table, path, row_group_size=1, store_decimal_as_integer=True)
        want = pq.read_table(path, filters=[("x", "<", decimal.Decimal(200))])["x"].to_pylist()
        assert want == [decimal.Decimal("123.45")]
        got = am.ParquetFile(path).read_table(filters=[("x", "<", 200)])["x"].to_pylist()
        assert want[0] in got


# ---------------------------------------------------------------------------- 6. bloom filters

BLOOM_COLUMNS = ("i64", "i32", "u32", "f64", "s", "long", "cat")


@pytest.mark.parametrize("name", ["bloom__pa_snappy", "bloom__duckdb", "bloom__pa_nobloom"])
def test_bloom_filters_never_drop_a_matching_row_group(name):
    """Every row group holds values spanning nearly the same min/max, so only a bloom filter can tell
    them apart. Looking up values that are present must always find them, with or without bloom
    filters, and give pyarrow's exact matches."""
    path = os.path.join(NESTED, name + ".parquet")
    want = pq.read_table(path)
    f = am.ParquetFile(path)
    rnd = random.Random(7)
    for col in BLOOM_COLUMNS:
        values = want[col].to_pylist()
        for v in rnd.sample(values, 25):
            flt = [(col, "==", v)]
            f.use_bloom_filters = True
            with_bloom = _exact(f.read_table(filters=flt), flt)
            f.use_bloom_filters = False
            without = _exact(f.read_table(filters=flt), flt)
            assert with_bloom.to_pylist() == without.to_pylist() == _exact(want, flt).to_pylist(), (col, v)
            assert with_bloom.num_rows > 0


def test_bloom_filters_skip_row_groups():
    path = os.path.join(NESTED, "bloom__pa_snappy.parquet")
    want = pq.read_table(path)
    f = am.ParquetFile(path)
    rnd = random.Random(8)
    for col in BLOOM_COLUMNS:
        lookups = skipped = 0
        for v in rnd.sample(want[col].to_pylist(), 20):
            f.read_table(columns=[col], filters=[(col, "==", v)])
            st = f.last_read_stats
            skipped += st["row_groups_skipped_by_bloom_filter"]
            lookups += 1
            assert st["row_groups_read"] >= 1
        # Each value lives in one of the four row groups; the other three are ruled out bar the odd
        # false positive (the filters were written for a 1% false-positive rate).
        assert skipped >= 0.9 * 3 * lookups, (col, skipped, lookups)
    f.use_bloom_filters = False
    f.read_table(columns=["i64"], filters=[("i64", "==", 4001)])
    assert f.last_read_stats["row_groups_skipped_by_bloom_filter"] == 0
    assert f.last_read_stats["row_groups_read"] == 4


def test_bloom_filters_from_duckdb():
    """DuckDB writes bloom filters for its dictionary-encoded columns (here `cat`)."""
    path = os.path.join(NESTED, "bloom__duckdb.parquet")
    f = am.ParquetFile(path)
    assert f.num_row_groups == 2
    skipped = 0
    for k in range(200):
        value = "c%03d" % k
        got = f.read_table(columns=["cat"], filters=[("cat", "==", value)])
        skipped += f.last_read_stats["row_groups_skipped_by_bloom_filter"]
        assert value in got["cat"].to_pylist()
    # Each value is in one of the two row groups; DuckDB's small filters let a few false positives by.
    assert skipped >= 150, skipped


def test_a_value_absent_but_within_the_statistics_is_ruled_out():
    path = os.path.join(NESTED, "bloom__pa_snappy.parquet")
    f = am.ParquetFile(path)
    # 0.25 lies inside row group 0's [min, max] of f64 but is not one of its values.
    t = f.read_table(columns=["f64"], filters=[("f64", "==", 0.25)])
    st = f.last_read_stats
    assert st["row_groups_skipped_by_statistics"] == 3 and st["row_groups_skipped_by_bloom_filter"] == 1
    assert t.num_rows == 0


# ------------------------------------------------------------------ larger files, written here

def test_repeated_columns_past_one_allocation_page():
    """Tens of thousands of level entries per column, several list and nested columns in one read.
    Repetition levels used to be decoded with a 4-byte scratch rank buffer that the kernel wrote a rank
    per level into -- harmless up to the 16 KB allocation padding (4,096 levels), memory corruption
    past it. This read has 120,000+ levels per repeated column."""
    import numpy as np
    n = 40_000
    i = np.arange(n)
    table = pa.table({
        "l": pa.array([None if k % 13 == 0 else list(range(k % 5)) for k in range(n)], pa.list_(pa.int64())),
        "ll": pa.array([None if k % 17 == 0 else [[k] * (k % 3), []] for k in range(n)],
                       pa.list_(pa.list_(pa.int32()))),
        "m": pa.array([[("k%d" % (k % 7), k)] * (k % 4) for k in range(n)], pa.map_(pa.string(), pa.int64())),
        "s": pa.array([None if k % 11 == 0 else {"a": int(k), "b": "v%d" % k} for k in range(n)],
                      pa.struct([("a", pa.int64()), ("b", pa.string())])),
        "x": pa.array(i, pa.int64()),
    })
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "big-nested.parquet")
        pq.write_table(table, path, compression="snappy", row_group_size=15_000)
        for _ in range(3):
            assert_same_table(am.read_parquet_table(path), pq.read_table(path))
