"""Delta Lake and Apache Iceberg reads, checked against deltalake and pyiceberg.

Three layers:

* every read recorded in Tests/Fixtures/lakehouse/expected.json (the same cases LakehouseTests.swift
  replays), through the Python binding;
* the committed fixture tables against the reference readers live, types included;
* larger tables generated here (several thousand rows, many commits and files, partitions, schema
  evolution, time travel), read with filters of every comparison over every column type and compared
  with deltalake's `to_pyarrow_table(filters=...)` and pyiceberg's `scan(row_filter=...)`.

Neither format promises a row order, so rows are compared sorted. The reference readers are optional
dependencies: without them the live comparisons are skipped and the expected.json replay still runs.
"""
import ctypes
import datetime as dt
import decimal
import glob
import json
import os
import shutil

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import arrowmetal as am

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIXTURES = os.path.join(ROOT, "Tests", "Fixtures", "lakehouse")

try:
    import deltalake as dl
except ImportError:          # pragma: no cover - optional reference reader
    dl = None
try:
    import pyiceberg  # noqa: F401
    from pyiceberg.table import StaticTable
except ImportError:          # pragma: no cover - optional reference reader
    StaticTable = None

needs_delta = pytest.mark.skipif(dl is None, reason="deltalake is not installed")
needs_iceberg = pytest.mark.skipif(StaticTable is None, reason="pyiceberg is not installed")


def load_cases():
    path = os.path.join(FIXTURES, "expected.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)["cases"]


CASES = load_cases()


# ---- fingerprints, as in generate_lakehouse.py

def fingerprint_value(v, typ):
    if v is None:
        return "null"
    if pa.types.is_boolean(typ):
        return "true" if v else "false"
    if pa.types.is_integer(typ):
        return str(int(v))
    if pa.types.is_floating(typ):
        return repr(float(v))
    if pa.types.is_date32(typ):
        return str((v - dt.date(1970, 1, 1)).days)
    if pa.types.is_timestamp(typ):
        if v.tzinfo is None:
            v = v.replace(tzinfo=dt.timezone.utc)
        d = v - dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
        return str((d.days * 86400 + d.seconds) * 1_000_000 + d.microseconds)
    if pa.types.is_decimal(typ):
        return str(int(v.scaleb(typ.scale)))
    return v


def rows_of(table):
    cols = [[fingerprint_value(v, table.schema.field(i).type) for v in table.column(i).to_pylist()]
            for i in range(table.num_columns)]
    return sorted(list(r) for r in zip(*cols)) if cols else []


def filters_of(json_filters):
    return [tuple(f) for f in json_filters]


def case_id(c):
    where = "v%s" % c["version"] if c["kind"] == "delta" else "s%s" % c.get("snapshot_id")
    return "%s-%s-%s" % (c["table"], where, "f%d" % len(c["filters"]))


def am_read(c):
    if c["kind"] == "delta":
        return am.read_delta_table(os.path.join(FIXTURES, c["table"]), version=c["version"],
                                   columns=c["columns"], filters=filters_of(c["filters"]))
    return am.read_iceberg_table(os.path.join(FIXTURES, c["metadata"]), snapshot_id=c["snapshot_id"],
                                 columns=c["columns"], filters=filters_of(c["filters"]))


def test_library_is_a_development_or_bundled_build():
    # The binding loads a dylib; this suite is meant to run against the checkout it lives in.
    assert os.path.exists(am._find_library())


@pytest.mark.skipif(not CASES, reason="Tests/Fixtures/lakehouse/expected.json is missing")
@pytest.mark.parametrize("case", CASES, ids=[case_id(c) for c in CASES])
def test_recorded_reference_reads(case):
    t = am_read(case)
    assert t.column_names == case["expected"]["names"]
    assert rows_of(t) == sorted(case["expected"]["rows"])


# ---- the committed tables against the reference readers, live

def normalise(table):
    """Plain values, sorted by row; large_string and string compare equal."""
    return rows_of(table)


@needs_delta
@pytest.mark.parametrize("name", ["basic", "partitioned", "by_day", "evolution", "multipart"])
def test_delta_fixtures_match_deltalake(name):
    path = os.path.join(FIXTURES, "delta", name)
    ref = dl.DeltaTable(path)
    for v in range(ref.version() + 1):
        # Every version is compared; a version deltalake cannot read fails the test rather than
        # dropping out of the comparison.
        expected = dl.DeltaTable(path, version=v).to_pyarrow_table()
        got = am.read_delta_table(path, version=v)
        assert got.schema == expected.schema, (name, v)
        assert normalise(got) == normalise(expected), (name, v)
    assert am.delta_latest_version(path) == ref.version()


@needs_iceberg
@pytest.mark.parametrize("name", ["v2_partitioned", "v1_plain", "transforms", "by_day"])
def test_iceberg_fixtures_match_pyiceberg(name, monkeypatch):
    # The fixtures record relative locations, which pyiceberg resolves against the working directory.
    monkeypatch.chdir(FIXTURES)
    meta = sorted(glob.glob(os.path.join("iceberg", name, "metadata", "*.metadata.json")))[-1]
    static = StaticTable.from_metadata(meta)
    for s in static.metadata.snapshots:
        expected = static.scan(snapshot_id=s.snapshot_id).to_arrow()
        got = am.read_iceberg_table(os.path.join(FIXTURES, meta), snapshot_id=s.snapshot_id)
        assert got.column_names == expected.column_names
        for f_got, f_exp in zip(got.schema, expected.schema):
            assert_iceberg_type(f_got.type, f_exp.type, (name, f_got, f_exp))
        assert normalise(got) == normalise(expected), (name, s.snapshot_id)
    assert am.iceberg_current_snapshot(os.path.join(FIXTURES, "iceberg", name)) == static.metadata.current_snapshot_id


def assert_iceberg_type(got, expected, where=None):
    """pyiceberg returns large_string / large_binary for some string and binary columns; ArrowMetal
    returns string / binary. Every other type must match exactly."""
    if pa.types.is_large_string(expected):
        assert got == pa.string(), where
    elif pa.types.is_large_binary(expected):
        assert got == pa.binary(), where
    else:
        assert got == expected, where


# ---- generated tables, larger, filtered every way

N_PER_COMMIT = 3000


def gen_rows(start, n):
    ids = list(range(start, start + n))
    return pa.table({
        "id": pa.array(ids, pa.int64()),
        "grp": [["a", "b", "c", "d", "e"][i % 5] for i in ids],
        "x": [None if i % 17 == 3 else (i * 7919 % 10007) / 7.0 - 500 for i in ids],
        "k": pa.array([None if i % 23 == 4 else (i * 31) % 1000 - 300 for i in ids], pa.int32()),
        "b": [None if i % 19 == 7 else i % 4 == 0 for i in ids],
        "d": [dt.date(2023, 12, 25) + dt.timedelta(days=i % 40) for i in ids],
        "t": pa.array([dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(minutes=13 * i) for i in ids],
                      pa.timestamp("us", tz="UTC")),
        "s": [None if i % 29 == 1 else "k%05d" % (i * 97 % 20000) for i in ids],
        "m": pa.array([None if i % 31 == 2 else decimal.Decimal(i % 5000) / 100 for i in ids], pa.decimal128(12, 2)),
    })


FILTERS = [
    [("id", "<", 1234)], [("id", ">=", 7000)], [("id", "==", 4321)], [("id", "!=", 5)],
    [("x", ">", 100.5)], [("x", "<=", -250.0)], [("k", "==", 12)], [("k", "!=", 12)], [("k", ">", 2.5)],
    [("b", "==", True)], [("grp", "==", "c")], [("grp", ">", "b")], [("grp", "!=", "a")],
    [("s", ">=", "k10000")], [("s", "<", "k00500")],
    [("d", ">=", dt.date(2024, 1, 20))], [("d", "==", dt.date(2023, 12, 31))],
    [("t", "<", dt.datetime(2024, 1, 10, 12, 30, tzinfo=dt.timezone.utc))],
    [("t", ">=", dt.datetime(2024, 1, 20, tzinfo=dt.timezone.utc))],
    [("id", ">", 2000), ("grp", "<=", "c"), ("x", "<", 0.0)],
]


@pytest.fixture(scope="module")
def delta_table(tmp_path_factory):
    if dl is None:
        pytest.skip("deltalake is not installed")
    p = str(tmp_path_factory.mktemp("delta") / "t")
    dl.write_deltalake(p, gen_rows(0, N_PER_COMMIT), partition_by=["grp"])
    dl.write_deltalake(p, gen_rows(N_PER_COMMIT, N_PER_COMMIT), mode="append")
    dl.DeltaTable(p).delete("k < -200")
    dl.DeltaTable(p).create_checkpoint()
    dl.write_deltalake(p, gen_rows(2 * N_PER_COMMIT, N_PER_COMMIT), mode="append")
    dl.write_deltalake(p, gen_rows(3 * N_PER_COMMIT, 50).filter(pc.equal(gen_rows(3 * N_PER_COMMIT, 50)["grp"], "e")),
                       mode="overwrite", predicate="grp = 'e'")
    return p


@needs_delta
def test_generated_delta_time_travel(delta_table):
    for v in range(dl.DeltaTable(delta_table).version() + 1):
        expected = dl.DeltaTable(delta_table, version=v).to_pyarrow_table()
        got = am.read_delta_table(delta_table, version=v)
        assert got.schema == expected.schema
        assert normalise(got) == normalise(expected), v


@needs_delta
@pytest.mark.parametrize("flt", FILTERS, ids=[str(f) for f in FILTERS])
def test_generated_delta_filters(delta_table, flt):
    expected = dl.DeltaTable(delta_table).to_pyarrow_table(filters=flt)
    got = am.read_delta_table(delta_table, filters=flt)
    assert normalise(got) == normalise(expected)
    projected = am.read_delta_table(delta_table, columns=["s", "id"], filters=flt)
    assert projected.column_names == ["s", "id"]
    assert normalise(projected) == normalise(expected.select(["s", "id"]))


@needs_delta
def test_generated_delta_pruning_counters(delta_table):
    cols, stats = am.read_delta(delta_table, filters=[("grp", "==", "c")], with_stats=True)
    assert stats["files_pruned_by_partition"] > 0
    assert stats["files_read"] + stats["files_pruned_by_partition"] + stats["files_pruned_by_statistics"] \
        == stats["files_total"]
    assert set(cols["grp"].to_arrow().to_pylist()) == {"c"}


@pytest.fixture(scope="module")
def iceberg_table(tmp_path_factory):
    if StaticTable is None:
        pytest.skip("pyiceberg is not installed")
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.expressions import LessThan
    from pyiceberg.types import StringType
    d = tmp_path_factory.mktemp("iceberg")
    cat = SqlCatalog("t", uri="sqlite:///%s" % (d / "cat.db"), warehouse="file://%s" % d)
    cat.create_namespace("ns")
    t = cat.create_table("ns.t", schema=gen_rows(0, 1).schema)
    with t.update_spec() as u:
        u.add_identity("grp")
    t.append(gen_rows(0, N_PER_COMMIT))
    t.append(gen_rows(N_PER_COMMIT, N_PER_COMMIT))
    t.delete(LessThan("k", -200))
    with t.update_schema() as u:
        u.add_column("extra", StringType())
        u.rename_column("x", "x2")
    t = cat.load_table("ns.t")
    more = gen_rows(2 * N_PER_COMMIT, N_PER_COMMIT).rename_columns(
        ["id", "grp", "x2", "k", "b", "d", "t", "s", "m"])
    more = more.append_column("extra", pa.array(["e%d" % (i % 7) for i in range(N_PER_COMMIT)]))
    t.append(more)
    return cat.load_table("ns.t").metadata_location


def iceberg_expr(flt):
    from pyiceberg.expressions import (And, EqualTo, GreaterThan, GreaterThanOrEqual, LessThan,
                                       LessThanOrEqual, NotEqualTo)
    ops = {"==": EqualTo, "!=": NotEqualTo, "<": LessThan, "<=": LessThanOrEqual, ">": GreaterThan,
           ">=": GreaterThanOrEqual}
    e = None
    for c, op, v in flt:
        if isinstance(v, dt.datetime) and v.tzinfo is None:
            v = v.replace(tzinfo=dt.timezone.utc)
        x = ops[op](c, v)
        e = x if e is None else And(e, x)
    return e


@needs_iceberg
def test_generated_iceberg_time_travel(iceberg_table):
    static = StaticTable.from_metadata(iceberg_table)
    for s in static.metadata.snapshots:
        expected = static.scan(snapshot_id=s.snapshot_id).to_arrow()
        got = am.read_iceberg_table(iceberg_table, snapshot_id=s.snapshot_id)
        assert got.column_names == expected.column_names
        assert normalise(got) == normalise(expected), s.snapshot_id
    got = am.read_iceberg_table(os.path.dirname(os.path.dirname(iceberg_table.replace("file://", ""))))
    assert normalise(got) == normalise(static.scan().to_arrow())


@needs_iceberg
@pytest.mark.parametrize("flt", FILTERS, ids=[str(f) for f in FILTERS])
def test_generated_iceberg_filters(iceberg_table, flt):
    flt = [("x2" if c == "x" else c, op, v) for c, op, v in flt]
    static = StaticTable.from_metadata(iceberg_table)
    if flt == [("k", ">", 2.5)]:
        # pyiceberg will not bind a float literal to an int column; pyarrow's comparison is the reference.
        full = static.scan().to_arrow()
        expected = full.filter(pc.greater(full["k"], 2.5))
    else:
        expected = static.scan(row_filter=iceberg_expr(flt)).to_arrow()
    got = am.read_iceberg_table(iceberg_table, filters=flt)
    assert normalise(got) == normalise(expected)


@needs_iceberg
def test_generated_iceberg_pruning_counters(iceberg_table):
    cols, stats = am.read_iceberg(iceberg_table, filters=[("grp", "==", "c")], with_stats=True)
    assert stats["files_pruned_by_partition"] > 0
    assert stats["manifests_total"] >= 1
    cols, stats = am.read_iceberg(iceberg_table, filters=[("id", ">=", 2 * N_PER_COMMIT)], with_stats=True)
    assert stats["files_pruned_by_statistics"] > 0
    assert len(cols["id"].to_arrow()) == N_PER_COMMIT


# ---- behaviour the reference readers do not cover

def test_delta_errors_name_the_feature():
    with pytest.raises(am.ArrowMetalError, match="deletionVectors"):
        am.read_delta(os.path.join(FIXTURES, "delta", "unsupported_deletion_vectors"))
    with pytest.raises(am.ArrowMetalError, match="columnMapping mode 'id'"):
        am.read_delta(os.path.join(FIXTURES, "delta", "unsupported_column_mapping_id"))
    with pytest.raises(am.ArrowMetalError, match="someFutureFeature"):
        am.read_delta(os.path.join(FIXTURES, "delta", "unsupported_unknown_feature"))
    with pytest.raises(am.ArrowMetalError, match="version 99"):
        am.read_delta(os.path.join(FIXTURES, "delta", "basic"), version=99)
    with pytest.raises(am.ArrowMetalError, match="no column named zz"):
        am.read_delta(os.path.join(FIXTURES, "delta", "basic"), columns=["zz"])
    with pytest.raises(am.ArrowMetalError, match="_delta_log"):
        am.read_delta(FIXTURES)


def test_iceberg_errors_name_the_feature():
    with pytest.raises(am.ArrowMetalError, match="position delete files"):
        am.read_iceberg(os.path.join(FIXTURES, "iceberg", "position_deletes"))
    with pytest.raises(am.ArrowMetalError, match="snapshot 42"):
        am.read_iceberg(os.path.join(FIXTURES, "iceberg", "v1_plain"), snapshot_id=42)


def test_column_mapping_reads_the_source_data():
    # deltalake 1.6.5 and polars 1.44.1 return nulls for this table's mapped columns (see
    # docs/LAKEHOUSE.md); the reference is the data the files were written from.
    t = am.read_delta_table(os.path.join(FIXTURES, "delta", "column_mapping"))
    assert t.column_names == ["id", "region", "total"]
    assert sorted(t["id"].to_pylist()) == list(range(9))
    assert t.sort_by("id")["region"].to_pylist()[:3] == ["north", "south", "north"]


@needs_delta
def test_column_mapping_reference_readers_return_nulls():
    """The difference docs/LAKEHOUSE.md records, pinned to the versions it names."""
    path = os.path.join(FIXTURES, "delta", "column_mapping")
    if dl.__version__ != "1.6.5":
        pytest.skip("recorded against deltalake 1.6.5")
    ref = dl.DeltaTable(path).to_pyarrow_table()
    assert ref.num_rows == 9 and ref["id"].null_count == 9 and ref["total"].null_count == 9
    pl = pytest.importorskip("polars")
    if pl.__version__ == "1.44.1":
        df = pl.read_delta(path)
        assert df.height == 9 and df["id"].null_count() == 9
    assert am.read_delta_table(path)["id"].null_count == 0


@needs_iceberg
def test_time_travel_filters_use_the_snapshot_schema(monkeypatch):
    """`amount` was renamed to `total` after the first snapshots: pyiceberg binds a time-travel filter
    to the current schema and does not find `amount`; ArrowMetal resolves it in the snapshot's schema."""
    monkeypatch.chdir(FIXTURES)
    meta = sorted(glob.glob(os.path.join("iceberg", "v2_partitioned", "metadata", "*.metadata.json")))[-1]
    static = StaticTable.from_metadata(meta)
    first = static.metadata.snapshots[1].snapshot_id
    from pyiceberg.expressions import GreaterThanOrEqual
    with pytest.raises(ValueError, match="amount"):
        static.scan(snapshot_id=first, row_filter=GreaterThanOrEqual("amount", 10.0)).to_arrow()
    got = am.read_iceberg_table(os.path.join(FIXTURES, meta), snapshot_id=first, filters=[("amount", ">=", 10.0)])
    full = static.scan(snapshot_id=first).to_arrow()
    expected = full.filter(pc.greater_equal(full["amount"], 10.0))
    assert normalise(got) == normalise(expected)


def test_iceberg_v1_manifests_listed_in_the_snapshot(tmp_path):
    """A format v1 snapshot may list its manifests directly instead of through a manifest list."""
    pytest.importorskip("pyiceberg")
    from pyiceberg.io.pyarrow import PyArrowFileIO
    from pyiceberg.manifest import read_manifest_list
    src = os.path.join(FIXTURES, "iceberg", "v1_plain")
    dst = tmp_path / "iceberg" / "v1_plain"
    shutil.copytree(src, dst)
    meta_path = sorted(glob.glob(str(dst / "metadata" / "*.metadata.json")))[-1]
    meta = json.load(open(meta_path))
    io = PyArrowFileIO()
    for s in meta["snapshots"]:
        listing = str(tmp_path / s.pop("manifest-list"))
        s["manifests"] = [m.manifest_path for m in read_manifest_list(io.new_input(listing))]
    json.dump(meta, open(meta_path, "w"))
    for s in meta["snapshots"]:
        a = am.read_iceberg_table(meta_path, snapshot_id=s["snapshot-id"])
        b = am.read_iceberg_table(os.path.join(FIXTURES, "iceberg", "v1_plain"), snapshot_id=s["snapshot-id"])
        assert normalise(a) == normalise(b)


def test_filter_literals():
    path = os.path.join(FIXTURES, "delta", "basic")
    base = am.read_delta_table(path)
    # A tz-aware datetime literal is converted to UTC; a date literal is a day.
    when = dt.datetime(2024, 1, 5, 1, 0, tzinfo=dt.timezone(dt.timedelta(hours=1)))
    got = am.read_delta_table(path, filters=[("ts", ">=", when)])
    ts = base["ts"].to_pylist()
    assert got.num_rows == sum(1 for v in ts if v is not None and v >= when)
    got = am.read_delta_table(path, filters=[("price", ">=", decimal.Decimal("5.00"))])
    assert got.num_rows == sum(1 for v in base["price"].to_pylist() if v is not None and v >= decimal.Decimal(5))
    with pytest.raises(am.ArrowMetalError, match="cannot contain"):
        am.read_delta(path, filters=[("name", "==", 'a"b')])
    with pytest.raises(am.ArrowMetalError, match="does not fit column day"):
        am.read_delta(path, filters=[("day", "==", 1.5)])


def test_empty_result_keeps_the_schema():
    t = am.read_delta_table(os.path.join(FIXTURES, "delta", "basic"), filters=[("id", ">", 10 ** 9)])
    assert t.num_rows == 0
    assert t.schema.field("price").type == pa.decimal128(10, 2)
    assert t.schema.field("ts").type == pa.timestamp("us", tz="UTC")


def test_c_abi_argument_errors():
    lib = am._lib
    out = ctypes.c_void_p()
    assert lib.am_delta_read(None, -1, None, 0, None, ctypes.byref(out)) == 2
    assert b"`path` is NULL" in lib.am_last_error()
    assert lib.am_iceberg_read(b"x", 0, 0, None, 3, None, ctypes.byref(out)) == 2
    assert b"`columns` is NULL but `n_columns` is 3" in lib.am_last_error()



# ---- review findings: values the first fixtures did not cover, and inputs that used to crash

NFD_E = "e\u0301"          # "é" decomposed, as macOS file names store it: byte-wise below "f"


@needs_delta
def test_delta_empty_string_partition_is_null(tmp_path):
    """The Delta protocol writes a null partition value as "" for every type; deltalake reads it as
    null, and so does ArrowMetal."""
    p = str(tmp_path / "t")
    grp = ["", "a", None, "x=y", "ü"]
    t = pa.table({"id": pa.array(range(50), pa.int64()), "grp": pa.array([grp[i % 5] for i in range(50)])})
    dl.write_deltalake(p, t, partition_by=["grp"])
    log = open(sorted(glob.glob(p + "/_delta_log/*.json"))[0]).read()
    assert '"grp":""' in log.replace(" ", "")        # the empty partition value is in the log
    got = am.read_delta_table(p)
    assert normalise(got) == normalise(dl.DeltaTable(p).to_pyarrow_table())
    assert got["grp"].null_count == 20
    for flt in [[("grp", "==", "")], [("grp", "!=", "a")], [("grp", "<", "b")], [("grp", ">=", "")]]:
        expected = dl.DeltaTable(p).to_pyarrow_table(filters=flt)
        assert normalise(am.read_delta_table(p, filters=flt)) == normalise(expected), flt


@needs_delta
def test_delta_decomposed_strings_survive_row_group_pruning(tmp_path):
    from deltalake import WriterProperties
    p = str(tmp_path / "t")
    s = ["%s%02d" % (NFD_E, i) for i in range(100)] + ["g%02d" % i for i in range(20)]
    t = pa.table({"id": pa.array(range(120), pa.int64()), "s": s})
    dl.write_deltalake(p, t, writer_properties=WriterProperties(max_row_group_size=16))
    for flt in [[("s", "<", "f")], [("s", "<=", "f")], [("s", ">", "f")], [("s", ">=", NFD_E + "50")],
                [("s", "==", NFD_E + "07")], [("s", "!=", NFD_E + "07")]]:
        expected = dl.DeltaTable(p).to_pyarrow_table(filters=flt)
        got = am.read_delta_table(p, filters=flt)
        assert normalise(got) == normalise(expected), flt
    assert am.read_delta_table(p, filters=[("s", "<", "f")]).num_rows == 100


def iceberg_catalog(tmp_path):
    from pyiceberg.catalog.sql import SqlCatalog
    cat = SqlCatalog("t", uri="sqlite:///%s" % (tmp_path / "cat.db"), warehouse="file://%s" % tmp_path)
    cat.create_namespace("ns")
    return cat


@needs_iceberg
def test_iceberg_partition_values_that_need_escaping(tmp_path):
    """pyiceberg writes the directory of partition value "x=y" as `grp=x%3Dy` and records that path;
    the reader opens it as written. The table also has a binary column and decomposed strings in small
    row groups."""
    cat = iceberg_catalog(tmp_path)
    grp = ["x=y", "ü", "a b", "plain", NFD_E]
    n = 200
    t = pa.table({
        "id": pa.array(range(n), pa.int64()),
        "grp": pa.array([grp[i % 5] for i in range(n)]),
        "s": pa.array(["%s%03d" % (NFD_E, i) if i % 4 else "g%03d" % i for i in range(n)]),
        "bin": pa.array([bytes([i % 3, 0x61, 0x62]) if i % 7 else None for i in range(n)], pa.binary()),
    })
    tbl = cat.create_table("ns.t", schema=t.schema, properties={"write.parquet.row-group-limit": "8"})
    with tbl.update_spec() as u:
        u.add_identity("grp")
    tbl.append(t)
    tbl = cat.load_table("ns.t")
    tbl.append(t.slice(0, 50))
    tbl = cat.load_table("ns.t")
    dirs = {os.path.basename(d) for d in glob.glob(str(tmp_path / "ns" / "t" / "data" / "*"))}
    assert "grp=x%3Dy" in dirs and "grp=%C3%BC" in dirs
    meta = tbl.metadata_location
    static = StaticTable.from_metadata(meta)
    for snap in static.metadata.snapshots:
        expected = static.scan(snapshot_id=snap.snapshot_id).to_arrow()
        got = am.read_iceberg_table(meta, snapshot_id=snap.snapshot_id)
        assert got.column_names == expected.column_names
        for f_got, f_exp in zip(got.schema, expected.schema):
            assert_iceberg_type(f_got.type, f_exp.type, (f_got, f_exp))
        assert normalise(got) == normalise(expected), snap.snapshot_id
    from pyiceberg.expressions import EqualTo, GreaterThan, LessThan, LessThanOrEqual, NotEqualTo
    for flt, expr in [([("grp", "==", "x=y")], EqualTo("grp", "x=y")), ([("grp", "!=", "ü")], NotEqualTo("grp", "ü")),
                      ([("s", "<", "f")], LessThan("s", "f")), ([("s", "<=", "f")], LessThanOrEqual("s", "f")),
                      ([("s", ">", "f")], GreaterThan("s", "f")), ([("grp", "<", "f")], LessThan("grp", "f"))]:
        expected = static.scan(row_filter=expr).to_arrow()
        got = am.read_iceberg_table(meta, filters=flt)
        assert normalise(got) == normalise(expected), flt
    # A bytes literal that is valid UTF-8 filters a binary column; one that is not is refused by name.
    full = static.scan().to_arrow()
    got = am.read_iceberg_table(meta, filters=[("bin", "==", b"\x01ab")])
    assert normalise(got) == normalise(full.filter(pc.equal(full["bin"], pa.scalar(b"\x01ab", pa.large_binary()))))
    with pytest.raises(am.ArrowMetalError, match="must be valid UTF-8"):
        am.read_iceberg(meta, filters=[("bin", "==", b"\x01\x00\xff")])
    # Projection: ArrowMetal returns the columns in the order asked for, pyiceberg in schema order.
    got = am.read_iceberg_table(meta, columns=["s", "id"])
    assert got.column_names == ["s", "id"]
    assert static.scan(selected_fields=("s", "id")).to_arrow().column_names == ["id", "s"]
    assert normalise(got) == normalise(full.select(["s", "id"]))


PA_OPS = {"==": pc.equal, "!=": pc.not_equal, "<": pc.less, "<=": pc.less_equal, ">": pc.greater,
          ">=": pc.greater_equal}


# The comparisons for which deltalake 1.6.5 also returns a NaN row (docs/LAKEHOUSE.md).
DELTALAKE_NAN_ROWS = {(">", 0.1), (">=", 0.10000000149011612), ("<", 1e300), (">", -1e300)}


@needs_delta
def test_float32_filters_compare_exactly(tmp_path):
    """A float32 column against a double literal is the exact comparison, as pyarrow does (it widens the
    column). deltalake agrees on the numbers but also returns the NaN row for some ordering comparisons;
    pyiceberg rounds the literal to float32 first."""
    p = str(tmp_path / "t")
    t = pa.table({"f": pa.array([0.1, 0.2, 0.5, None, float("nan")], pa.float32())})
    dl.write_deltalake(p, t)
    dl.write_deltalake(p, pa.table({"f": pa.array([0.1], pa.float32())}), mode="append")
    full = dl.DeltaTable(p).to_pyarrow_table()
    for col, op, lit in [("f", "==", 0.1), ("f", ">", 0.1), ("f", "<=", 0.1), ("f", "!=", 0.1),
                         ("f", ">=", 0.10000000149011612), ("f", "<", 0.5), ("f", "<", 1e300), ("f", ">", -1e300)]:
        got = am.read_delta_table(p, filters=[(col, op, lit)])
        assert normalise(got) == normalise(full.filter(PA_OPS[op](full["f"], lit))), (op, lit)
        reference = dl.DeltaTable(p).to_pyarrow_table(filters=[(col, op, lit)])
        if op != "!=":
            # deltalake returns the NaN row for some ordering comparisons (the ones recorded below);
            # the numbers agree.
            nan_rows = pc.sum(pc.is_nan(reference["f"])).as_py() or 0
            assert nan_rows == (1 if (op, lit) in DELTALAKE_NAN_ROWS else 0), (op, lit)
            reference = reference.filter(pc.invert(pc.is_nan(reference["f"])))
        assert normalise(got) == normalise(reference), (op, lit)
    assert am.read_delta_table(p, filters=[("f", ">", 0.1)]).num_rows == 4        # both 0.1f rows, 0.2, 0.5


@needs_iceberg
def test_float32_literal_rounding_differs_from_pyiceberg(tmp_path):
    cat = iceberg_catalog(tmp_path)
    t = pa.table({"f": pa.array([0.1, 0.2, 0.5, None], pa.float32())})
    tbl = cat.create_table("ns.t", schema=t.schema)
    tbl.append(t)
    meta = cat.load_table("ns.t").metadata_location
    from pyiceberg.expressions import GreaterThan
    ref = StaticTable.from_metadata(meta).scan(row_filter=GreaterThan("f", 0.1)).to_arrow()
    assert ref.num_rows == 2                        # pyiceberg: 0.1 rounded to 0.1f, which is not > 0.1f
    got = am.read_iceberg_table(meta, filters=[("f", ">", 0.1)])
    assert normalise(got) == normalise(t.filter(pc.greater(t["f"], 0.1)))
    assert got.num_rows == 3                        # exact: 0.1f is 0.10000000149... > 0.1


def test_filter_text_that_would_change_meaning_is_refused():
    path = os.path.join(FIXTURES, "delta", "basic")
    with pytest.raises(am.ArrowMetalError, match="would be read first"):
        am.read_delta(path, filters=[("name", "<", "a==b")])
    with pytest.raises(am.ArrowMetalError, match="version must be 0 or more"):
        am.read_delta(path, version=-5)
    lib = am._lib
    out = ctypes.c_void_p()
    assert lib.am_delta_read(path.encode(), -5, None, 0, None, ctypes.byref(out)) == 1
    assert b"version -5" in lib.am_last_error()


def run_isolated(code, timeout=120):
    """Runs `code` in a fresh interpreter, so a crash is a failed test rather than a dead suite."""
    import subprocess
    import sys
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "python"))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout,
                          cwd=ROOT, env=env)


EDGE_LITERALS = [
    ("delta/basic", "id", ">", 9.25e18, 0),
    ("delta/basic", "id", "<", -9.25e18, 0),
    ("delta/basic", "qty", "<", 9.25e18, None),       # None: every non-null qty
    ("iceberg/v2_partitioned", "id", ">", 9.25e18, 0),
    ("delta/basic", "day", ">", "100000000000000000-01-01", "does not fit column day"),
    ("delta/basic", "ts", ">", "2024-01-01 99999999999:00:00", "does not fit column ts"),
    ("delta/basic", "price", ">", "\u00bd", "does not fit column price"),
    ("delta/basic", "price", ">", "\u0967", "does not fit column price"),
    ("delta/basic", "price", ">", "9" * 200, "does not fit column price"),
]


@pytest.mark.parametrize("table,col,op,lit,want", EDGE_LITERALS, ids=[repr(e[1:4]) for e in EDGE_LITERALS])
def test_filter_literals_at_type_edges(table, col, op, lit, want):
    fn = "read_delta_table" if table.startswith("delta") else "read_iceberg_table"
    code = ("import arrowmetal as am\ntry:\n    print(am.%s(%r, filters=[(%r, %r, %r)]).num_rows)\n"
            "except am.ArrowMetalError as e:\n    print('ERROR', e)\n"
            % (fn, os.path.join(FIXTURES, table), col, op, lit))
    r = run_isolated(code)
    assert r.returncode == 0, (r.returncode, r.stderr[-500:])
    out = r.stdout.strip()
    if isinstance(want, str):
        assert out.startswith("ERROR") and want in out, out
    elif want is None:
        base = am.read_delta_table(os.path.join(FIXTURES, table))
        assert out == str(len(base[col]) - base[col].null_count)
    else:
        assert out == str(want)


def _zz(v):
    u = ((v << 1) ^ (v >> 63)) & 0xFFFFFFFFFFFFFFFF
    out = bytearray()
    while u >= 0x80:
        out.append((u & 0x7F) | 0x80)
        u >>= 7
    out.append(u)
    return bytes(out)


def _avro(schema, block):
    def s(x):
        b = x.encode()
        return _zz(len(b)) + b
    sync = b"\xab" * 16
    return b"Obj\x01" + _zz(2) + s("avro.schema") + s(schema) + s("avro.codec") + s("null") + _zz(0) + sync + block + sync


MALFORMED_AVRO = {
    "block size near Int64.max": ('{"type":"record","name":"r","fields":[{"name":"a","type":"long"}]}',
                                  _zz(1) + _zz(2 ** 63 - 1)),
    "self-recursive record": ('{"type":"record","name":"r","fields":[{"name":"a","type":"r"}]}', _zz(1) + _zz(1) + b"\x00"),
    "huge block count, null schema": ('"null"', _zz(2 ** 63 - 1) + _zz(0)),
    "array count Int64.min": ('{"type":"array","items":"long"}', _zz(1) + _zz(11) + _zz(-2 ** 63) + _zz(0)),
}


@pytest.mark.parametrize("name", sorted(MALFORMED_AVRO))
def test_malformed_manifest_list_is_an_error(tmp_path, name):
    schema, block = MALFORMED_AVRO[name]
    dst = tmp_path / "iceberg" / "v1_plain"
    shutil.copytree(os.path.join(FIXTURES, "iceberg", "v1_plain"), dst)
    meta = json.load(open(sorted(glob.glob(str(dst / "metadata" / "*.metadata.json")))[-1]))
    listing = tmp_path / meta["snapshots"][-1]["manifest-list"]
    listing.write_bytes(_avro(schema, block))
    code = ("import arrowmetal as am\ntry:\n    am.read_iceberg(%r)\n    print('READ')\n"
            "except am.ArrowMetalError as e:\n    print('ERROR', e)\n" % str(dst))
    r = run_isolated(code, timeout=60)
    assert r.returncode == 0, (r.returncode, r.stderr[-500:])
    assert r.stdout.startswith("ERROR") and "manifest list" in r.stdout, r.stdout


def test_delta_malformed_partition_values_are_errors(tmp_path):
    dst = tmp_path / "by_day"
    shutil.copytree(os.path.join(FIXTURES, "delta", "by_day"), dst)
    last = sorted(glob.glob(str(dst / "_delta_log" / "*.json")))[-1]
    text = open(last).read()
    assert '"partitionValues":{' in text
    open(last, "w").write(text.replace('"partitionValues":{', '"partitionValues":[{').replace('},"size"', '}],"size"'))
    with pytest.raises(am.ArrowMetalError, match="is not a JSON object"):
        am.read_delta(str(dst))


def test_column_mapping_duckdb_reads_data_columns_not_the_partition():
    """DuckDB's delta_scan reads the mapped data columns of the column-mapping fixture exactly as
    ArrowMetal does, and returns null for the mapped partition column (docs/LAKEHOUSE.md)."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect(config={"autoinstall_known_extensions": False, "autoload_known_extensions": False})
    try:
        con.execute("LOAD delta")
    except Exception as e:           # pragma: no cover - depends on the local extension cache
        pytest.skip("DuckDB's delta extension does not load offline: %s" % str(e).splitlines()[0])
    path = os.path.join(FIXTURES, "delta", "column_mapping")
    ref = con.execute("SELECT * FROM delta_scan('%s')" % path).arrow()
    if hasattr(ref, "read_all"):
        ref = ref.read_all()
    got = am.read_delta_table(path)
    assert ref.column_names == got.column_names == ["id", "region", "total"]
    assert normalise(got.select(["id", "total"])) == normalise(ref.select(["id", "total"]))
    assert ref["region"].null_count == ref.num_rows
    assert got["region"].null_count == 0


@needs_delta
@pytest.mark.parametrize("op", ["==", "!=", "<", "<=", ">", ">="])
def test_delta_nan_partition_matches_not_equal(tmp_path, op):
    """A float partition value of NaN (deltalake writes `fp=NaN`) is unequal to every literal: the file is
    kept for `!=` and pruned for every other comparison, as deltalake and pyarrow filter it."""
    p = str(tmp_path / "t")
    src = pa.table({"id": pa.array([1, 2, 3], pa.int64()), "fp": pa.array([1.0, float("nan"), 2.0])})
    dl.write_deltalake(p, src, partition_by=["fp"])
    got = am.read_delta_table(p, filters=[("fp", op, 1.0)])
    ref = dl.DeltaTable(p).to_pyarrow_table(filters=[("fp", op, 1.0)])
    assert sorted(got["id"].to_pylist()) == sorted(ref["id"].to_pylist())
    if op == "!=":
        assert sorted(got["id"].to_pylist()) == [2, 3]


@needs_iceberg
def test_null_rows_under_always_true_filters_differ_from_pyiceberg(tmp_path):
    """pyiceberg 0.12.0 turns a comparison it can decide without the data into "every row", nulls
    included: `!=` on a column that older files predate returns those files' null-filled rows, and a
    literal beyond the column's range returns the null and NaN rows. ArrowMetal keeps its null rule (a
    null never matches, NaN matches only `!=`) and returns what pyarrow's filter does (docs/LAKEHOUSE.md)."""
    from pyiceberg.expressions import LessThan, NotEqualTo
    from pyiceberg.types import StringType
    cat = iceberg_catalog(tmp_path)
    t1 = pa.table({"id": pa.array([1, 2, 3], pa.int64()), "i": pa.array([1, None, 3], pa.int32()),
                   "f": pa.array([1.0, None, float("nan")], pa.float32())})
    tbl = cat.create_table("ns.t", schema=t1.schema)
    tbl.append(t1)
    with tbl.update_schema() as u:
        u.add_column("note", StringType())
    t2 = pa.table({"id": pa.array([4, 5, 6], pa.int64()), "i": pa.array([4, 5, None], pa.int32()),
                   "f": pa.array([2.0, 3.0, None], pa.float32()), "note": pa.array(["n0", "n1", None])})
    cat.load_table("ns.t").append(t2)
    meta = cat.load_table("ns.t").metadata_location
    full = pa.concat_tables([t1.append_column("note", pa.nulls(3, pa.string())), t2])
    cases = [(("note", "!=", "n0"), NotEqualTo("note", "n0"), pc.not_equal(full["note"], "n0"), [1, 2, 3, 5]),
             (("i", "<", 2 ** 63 - 1), LessThan("i", 2 ** 63 - 1), pc.less(full["i"], 2 ** 63 - 1), [1, 2, 3, 4, 5, 6]),
             (("f", "<", 1e308), LessThan("f", 1e308), pc.less(full["f"].cast(pa.float64()), 1e308), [1, 2, 3, 4, 5, 6])]
    for flt, expr, mask, pyiceberg_ids in cases:
        got = sorted(am.read_iceberg_table(meta, filters=[flt])["id"].to_pylist())
        assert got == sorted(full.filter(mask)["id"].to_pylist()), flt
        ref = sorted(StaticTable.from_metadata(meta).scan(row_filter=expr).to_arrow()["id"].to_pylist())
        assert ref == pyiceberg_ids, flt
        assert got != ref, flt
