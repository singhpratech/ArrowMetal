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
        try:
            expected = dl.DeltaTable(path, version=v).to_pyarrow_table()
        except Exception:        # versions below a multi-part checkpoint are still readable by both
            continue
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
            if pa.types.is_large_string(f_exp.type):
                # pyiceberg returns large_string for some string columns; ArrowMetal returns string.
                assert f_got.type == pa.string()
            else:
                assert f_got.type == f_exp.type, (name, f_got, f_exp)
        assert normalise(got) == normalise(expected), (name, s.snapshot_id)
    assert am.iceberg_current_snapshot(os.path.join(FIXTURES, "iceberg", name)) == static.metadata.current_snapshot_id


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
