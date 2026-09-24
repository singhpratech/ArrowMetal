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

# `arrowschema__pa_types` and `arrowschema__pa_fslnull` have tests of their own below.
ARROW_SCHEMA = sorted(p for p in glob.glob(os.path.join(NESTED, "arrowschema__*.parquet"))
                      if ids(p) not in ("arrowschema__pa_types", "arrowschema__pa_fslnull"))


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


@pytest.mark.parametrize("name", ["arrowschema__pa_nostore", "arrowschema__pa_corrupt", "arrowschema__pa_truncated"])
def test_absent_or_malformed_arrow_schema_is_ignored(name):
    """No ARROW:schema, one that is not base64, one that is base64 of a truncated message: each reads
    as the Parquet schema alone describes it, exactly as pyarrow does."""
    path = os.path.join(NESTED, name + ".parquet")
    got, want = am.read_parquet_table(path), pq.read_table(path)
    assert_same_table(got, want)
    assert got["ts_paris"].type == pa.timestamp("us", tz="UTC") == want["ts_paris"].type


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
