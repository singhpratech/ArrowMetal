"""Writes the Delta Lake and Apache Iceberg fixture tables under Tests/Fixtures/lakehouse, and
expected.json: what the reference readers (deltalake's to_pyarrow_table, pyiceberg's scan().to_arrow())
return for a list of reads over them. LakehouseTests.swift replays every case in expected.json against
ArrowMetal's reader; python/tests/test_lakehouse.py also re-checks the tables against the reference
readers live.

    python Tests/Fixtures/lakehouse/generate_lakehouse.py        # needs deltalake and pyiceberg[sql-sqlite]

The tables are small on purpose (tens of rows, a few files each): they exercise the metadata paths --
several commits, overwrites, deletes, a checkpoint, partitions, schema evolution, time travel -- and
the unsupported-feature errors. Iceberg tables are written with a *relative* warehouse path, so the
metadata records locations like `iceberg/v2_partitioned` and no machine-specific directory; the reader
re-roots paths under the table's recorded location at wherever the table is found.
"""
import datetime as dt
import decimal
import json
import os
import shutil
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))

SCHEMA = pa.schema([
    ("id", pa.int64()),
    ("region", pa.string()),
    ("amount", pa.float64()),
    ("qty", pa.int32()),
    ("flag", pa.bool_()),
    ("day", pa.date32()),
    ("ts", pa.timestamp("us", tz="UTC")),
    ("price", pa.decimal128(10, 2)),
    ("name", pa.string()),
])

REGIONS = ["north", "south", "east", "west"]


def rows(start, n, regions=REGIONS, null_region_every=0, schema=SCHEMA):
    """A deterministic batch of `n` rows with ids start..start+n-1 (a few nulls in most columns)."""
    ids = list(range(start, start + n))
    region = [None if null_region_every and i % null_region_every == 0 else regions[i % len(regions)] for i in ids]
    data = {
        "id": ids,
        "region": region,
        "amount": [None if i % 7 == 3 else round(i * 1.25 - 10, 2) for i in ids],
        "qty": [None if i % 11 == 5 else (i * 37) % 100 - 20 for i in ids],
        "flag": [None if i % 13 == 6 else i % 3 == 0 for i in ids],
        "day": [dt.date(2024, 1, 1) + dt.timedelta(days=i % 5) for i in ids],
        "ts": [dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(hours=7 * i, microseconds=i) for i in ids],
        "price": [None if i % 9 == 4 else decimal.Decimal(i * 13 % 1000) / 100 for i in ids],
        "name": [None if i % 8 == 2 else "item-%03d" % (i * 7 % 50) for i in ids],
    }
    return pa.table({k: data[k] for k in schema.names}, schema=schema)


# ---------------------------------------------------------------------------------------------------
# Fingerprints: one string per value, identical to what LakehouseTests.swift prints for a column.

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
        delta = v - dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
        return str((delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds)
    if pa.types.is_decimal(typ):
        return str(int(v.scaleb(typ.scale)))
    if pa.types.is_string(typ) or pa.types.is_large_string(typ) or pa.types.is_string_view(typ):
        return v
    raise TypeError("no fingerprint for %s" % typ)


def json_filters(filters):
    """Filters as JSON: dates and datetimes become ISO 8601 strings, which is also how the Python
    binding passes them to the reader."""
    out = []
    for c, op, v in filters or []:
        if isinstance(v, dt.datetime):
            v = v.isoformat()
        elif isinstance(v, dt.date):
            v = v.isoformat()
        out.append([c, op, v])
    return out


def fingerprint(table):
    names = table.column_names
    cols = [[fingerprint_value(v, table.schema.field(n).type) for v in table.column(n).to_pylist()] for n in names]
    out = [list(r) for r in zip(*cols)] if cols else []
    out.sort()
    return {"names": names, "rows": out}


# ---------------------------------------------------------------------------------------------------
# Delta Lake

def delta_tables(cases):
    import deltalake as dl

    root = os.path.join(HERE, "delta")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)

    def reference(name, version=None, columns=None, filters=None):
        t = dl.DeltaTable(os.path.join(root, name), version=version)
        tbl = t.to_pyarrow_table(columns=columns, filters=filters or None)
        cases.append({"kind": "delta", "table": "delta/" + name, "version": version, "columns": columns,
                      "filters": json_filters(filters), "expected": fingerprint(tbl)})

    # basic: unpartitioned; append, delete, predicate overwrite, checkpoint, append.
    p = os.path.join(root, "basic")
    dl.write_deltalake(p, rows(0, 30))                                               # v0
    dl.write_deltalake(p, rows(30, 20), mode="append")                               # v1
    dl.DeltaTable(p).delete("id < 10")                                               # v2
    dl.write_deltalake(p, rows(40, 5).set_column(2, "amount", pa.array([1000.5] * 5)),
                       mode="overwrite", predicate="id >= 40 AND id < 45")             # v3
    dl.DeltaTable(p).create_checkpoint()                                             # checkpoint at v3
    dl.write_deltalake(p, rows(50, 10), mode="append")                               # v4
    for v in [0, 1, 2, 3, 4, None]:
        reference("basic", version=v)
    reference("basic", columns=["name", "id"])
    reference("basic", filters=[("id", ">=", 20), ("amount", "<", 40.0)])
    reference("basic", filters=[("name", "==", "item-014")])
    reference("basic", filters=[("qty", "!=", 16)])
    reference("basic", filters=[("flag", "==", True)])
    reference("basic", version=1, filters=[("id", "<", 5)])

    # partitioned by a string column with a null partition; partition delete and replace.
    p = os.path.join(root, "partitioned")
    dl.write_deltalake(p, rows(0, 40, null_region_every=9), partition_by=["region"])  # v0
    dl.write_deltalake(p, rows(40, 12), mode="append")                                # v1
    dl.DeltaTable(p).delete("region = 'east'")                                        # v2
    dl.DeltaTable(p).create_checkpoint()                                              # checkpoint at v2
    dl.write_deltalake(p, rows(52, 8), mode="append")                                 # v3
    dl.write_deltalake(p, rows(100, 4, regions=["north"]), mode="overwrite",
                       predicate="region = 'north'")                                   # v4
    for v in [0, 1, 2, 3, 4]:
        reference("partitioned", version=v)
    reference("partitioned", filters=[("region", "==", "south")])
    reference("partitioned", filters=[("region", ">", "north")])
    reference("partitioned", filters=[("region", "!=", "west"), ("id", ">", 20)])
    reference("partitioned", columns=["region"], filters=[("region", "<=", "south")])

    # partitioned by a date column.
    p = os.path.join(root, "by_day")
    dl.write_deltalake(p, rows(0, 25), partition_by=["day"])
    dl.write_deltalake(p, rows(25, 10), mode="append")
    reference("by_day")
    reference("by_day", filters=[("day", ">=", dt.date(2024, 1, 3))])
    reference("by_day", filters=[("day", "==", dt.date(2024, 1, 2)), ("amount", ">", 0.0)])

    # schema evolution: a column added by a later commit.
    p = os.path.join(root, "evolution")
    base = rows(0, 10).select(["id", "region", "amount"])
    dl.write_deltalake(p, base)
    wider = rows(10, 10).select(["id", "region", "amount", "name"])
    dl.write_deltalake(p, wider, mode="append", schema_mode="merge")
    dl.write_deltalake(p, rows(20, 5).select(["id", "region", "amount", "name"]), mode="append")
    for v in [0, 1, 2]:
        reference("evolution", version=v)
    reference("evolution", filters=[("name", ">=", "item-010")])

    # multi-part checkpoint: the single checkpoint of a copy of `partitioned` split into two parts.
    src = os.path.join(root, "partitioned")
    p = os.path.join(root, "multipart")
    shutil.copytree(src, p)
    log = os.path.join(p, "_delta_log")
    cp = os.path.join(log, "%020d.checkpoint.parquet" % 2)
    t = pq.read_table(cp)
    half = t.num_rows // 2
    pq.write_table(t.slice(0, half), os.path.join(log, "%020d.checkpoint.%010d.%010d.parquet" % (2, 1, 2)))
    pq.write_table(t.slice(half), os.path.join(log, "%020d.checkpoint.%010d.%010d.parquet" % (2, 2, 2)))
    os.remove(cp)
    last = json.load(open(os.path.join(log, "_last_checkpoint")))
    last["parts"] = 2
    last.pop("sizeInBytes", None)
    json.dump(last, open(os.path.join(log, "_last_checkpoint"), "w"))
    for v in [2, 4]:
        reference("multipart", version=v)

    handcrafted_delta(root, cases, reference)


def handcrafted_delta(root, cases, reference):
    """Tables the Python writer does not produce: column mapping (name mode, with a rename) and three
    protocol cases the reader must refuse."""
    def write_log(path, version, actions):
        os.makedirs(os.path.join(path, "_delta_log"), exist_ok=True)
        with open(os.path.join(path, "_delta_log", "%020d.json" % version), "w") as f:
            for a in actions:
                f.write(json.dumps(a, separators=(",", ":")) + "\n")

    def field(name, typ, phys, fid):
        return {"name": name, "type": typ, "nullable": True,
                "metadata": {"delta.columnMapping.id": fid, "delta.columnMapping.physicalName": phys}}

    def add(path, rel, table, partition=None):
        full = os.path.join(path, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        pq.write_table(table, full)
        return {"add": {"path": rel, "partitionValues": partition or {}, "size": os.path.getsize(full),
                        "modificationTime": 1700000000000, "dataChange": True,
                        "stats": json.dumps({"numRecords": table.num_rows})}}

    # Column mapping, name mode. Physical names differ from logical ones; commit 1 renames
    # `amount` -> `total` and appends a file.
    p = os.path.join(root, "column_mapping")
    fields = [field("id", "long", "col-1a", 1), field("region", "string", "col-2b", 2),
              field("amount", "double", "col-3c", 3)]
    meta = {"id": "00000000-0000-0000-0000-000000000001", "format": {"provider": "parquet", "options": {}},
            "schemaString": json.dumps({"type": "struct", "fields": fields}),
            "partitionColumns": ["region"], "createdTime": 1700000000000,
            "configuration": {"delta.columnMapping.mode": "name", "delta.columnMapping.maxColumnId": "3"}}
    proto = {"protocol": {"minReaderVersion": 2, "minWriterVersion": 5}}
    def physical(t):
        """Data under physical names, each field carrying its column mapping id as the Parquet field id
        (as Spark writes column-mapped tables)."""
        schema = pa.schema([pa.field("col-1a", pa.int64(), metadata={"PARQUET:field_id": "1"}),
                            pa.field("col-3c", pa.float64(), metadata={"PARQUET:field_id": "3"})])
        return pa.table([t["id"], t["amount"]], schema=schema)

    r0 = rows(0, 6, regions=["north", "south"])
    acts = [proto, {"metaData": meta}]
    # Column-mapped tables do not name partition directories after the (physical) column.
    for reg, prefix in [("north", "a1"), ("south", "b2")]:
        part = r0.filter(pa.compute.equal(r0["region"], reg))
        acts.append(add(p, "%s/part-0.parquet" % prefix, physical(part), {"col-2b": reg}))
    write_log(p, 0, acts)
    fields[2] = field("total", "double", "col-3c", 3)
    meta1 = dict(meta, schemaString=json.dumps({"type": "struct", "fields": fields}))
    r1 = rows(6, 3, regions=["south"])
    write_log(p, 1, [{"metaData": meta1}, add(p, "b2/part-1.parquet", physical(r1), {"col-2b": "south"})])
    # deltalake 1.6.5 (and polars, which reads through it) returns nulls for every mapped column of
    # this table, so the reference here is the pyarrow data the files were written from.
    logical0 = r0.select(["id", "region", "amount"])
    logical1 = pa.concat_tables([logical0, r1.select(["id", "region", "amount"])]).rename_columns(
        ["id", "region", "total"])
    for version, tbl, filters in [(0, logical0, None), (None, logical1, None),
                                  (None, logical1.filter(pa.compute.greater(logical1["total"], -5.0)),
                                   [("total", ">", -5.0)])]:
        cases.append({"kind": "delta", "table": "delta/column_mapping", "version": version, "columns": None,
                      "filters": json_filters(filters), "reference": "pyarrow source data",
                      "expected": fingerprint(tbl)})

    # Protocol cases the reader refuses (each also refused by name in the tests).
    simple = {"id": "00000000-0000-0000-0000-000000000002", "format": {"provider": "parquet", "options": {}},
              "schemaString": json.dumps({"type": "struct", "fields": [
                  {"name": "id", "type": "long", "nullable": True, "metadata": {}}]}),
              "partitionColumns": [], "createdTime": 1700000000000, "configuration": {}}
    one = pa.table({"id": pa.array([1, 2, 3], pa.int64())})
    p = os.path.join(root, "unsupported_deletion_vectors")
    write_log(p, 0, [{"protocol": {"minReaderVersion": 3, "minWriterVersion": 7,
                                   "readerFeatures": ["deletionVectors"], "writerFeatures": ["deletionVectors"]}},
                     {"metaData": dict(simple, configuration={"delta.enableDeletionVectors": "true"})},
                     add(p, "part-0.parquet", one)])
    p = os.path.join(root, "unsupported_column_mapping_id")
    idfields = [field("id", "long", "col-1a", 1)]
    write_log(p, 0, [{"protocol": {"minReaderVersion": 2, "minWriterVersion": 5}},
                     {"metaData": dict(simple, schemaString=json.dumps({"type": "struct", "fields": idfields}),
                                       configuration={"delta.columnMapping.mode": "id"})},
                     add(p, "part-0.parquet", pa.table({"col-1a": one["id"]}))])
    p = os.path.join(root, "unsupported_unknown_feature")
    write_log(p, 0, [{"protocol": {"minReaderVersion": 3, "minWriterVersion": 7,
                                   "readerFeatures": ["someFutureFeature"], "writerFeatures": ["someFutureFeature"]}},
                     {"metaData": simple}, add(p, "part-0.parquet", one)])


# ---------------------------------------------------------------------------------------------------
# Apache Iceberg

def iceberg_tables(cases):
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.expressions import (And, EqualTo, GreaterThan, GreaterThanOrEqual, LessThan,
                                       LessThanOrEqual, NotEqualTo)
    from pyiceberg.table import StaticTable
    from pyiceberg.transforms import DayTransform, MonthTransform, TruncateTransform
    from pyiceberg.types import LongType, StringType

    root = os.path.join(HERE, "iceberg")
    shutil.rmtree(root, ignore_errors=True)
    catalog_dir = tempfile.mkdtemp()
    cwd = os.getcwd()
    os.chdir(HERE)        # the warehouse path is relative, so the metadata records relative locations
    try:
        cat = SqlCatalog("fixtures", uri="sqlite:///" + os.path.join(catalog_dir, "catalog.db"), warehouse="iceberg")
        cat.create_namespace("fixtures")
        ops = {"==": EqualTo, "!=": NotEqualTo, "<": LessThan, "<=": LessThanOrEqual, ">": GreaterThan,
               ">=": GreaterThanOrEqual}

        def reference(tbl, name, snapshot_index=None, columns=None, filters=None):
            meta = tbl.metadata_location
            static = StaticTable.from_metadata(meta)
            snap = None if snapshot_index is None else static.metadata.snapshots[snapshot_index].snapshot_id
            expr = None
            for c, op, v in filters or []:
                e = ops[op](c, v)
                expr = e if expr is None else And(expr, e)
            kwargs = {}
            if expr is not None:
                kwargs["row_filter"] = expr
            if columns is not None:
                kwargs["selected_fields"] = tuple(columns)
            out = static.scan(snapshot_id=snap, **kwargs).to_arrow()
            if columns is not None:
                out = out.select(columns)
            cases.append({"kind": "iceberg", "table": "iceberg/" + name,
                          "metadata": os.path.relpath(meta.replace("file://", ""), HERE),
                          "snapshot_id": snap, "columns": columns, "filters": json_filters(filters),
                          "expected": fingerprint(out)})

        # v2, partitioned by identity(region); appends, a copy-on-write delete, schema evolution (added
        # column, rename, int -> long promotion), a partial overwrite.
        t = cat.create_table("fixtures.v2_partitioned", schema=SCHEMA, location="iceberg/v2_partitioned")
        with t.update_spec() as u:
            u.add_identity("region")
        t.append(rows(0, 30))                                                        # s0
        t.append(rows(30, 20))                                                       # s1
        t.delete(LessThan("id", 5))                                                  # s2
        with t.update_schema() as u:
            u.add_column("note", StringType())
            u.rename_column("amount", "total")
            u.update_column("qty", LongType())
        t = cat.load_table("fixtures.v2_partitioned")
        extra = rows(50, 10).rename_columns(["id", "region", "total", "qty", "flag", "day", "ts", "price", "name"])
        extra = extra.set_column(3, "qty", extra["qty"].cast(pa.int64()))
        extra = extra.append_column("note", pa.array(["n%d" % i for i in range(10)]))
        t.append(extra)                                                              # s3
        t.overwrite(extra.filter(pa.compute.equal(extra["region"], "west")).slice(0, 1),
                    overwrite_filter=EqualTo("region", "west"))                       # s4 (+ s5)
        t = cat.load_table("fixtures.v2_partitioned")
        n = len(t.metadata.snapshots)
        for i in range(n):
            reference(t, "v2_partitioned", snapshot_index=i)
        reference(t, "v2_partitioned")
        reference(t, "v2_partitioned", columns=["note", "id", "total"])
        reference(t, "v2_partitioned", filters=[("region", "==", "south")])
        reference(t, "v2_partitioned", filters=[("ts", ">=", dt.datetime(2024, 1, 8, 3, 0, 0))])
        reference(t, "v2_partitioned", filters=[("total", ">", 20.0), ("qty", "<", 50)])
        reference(t, "v2_partitioned", filters=[("name", "<", "item-020")])
        reference(t, "v2_partitioned", filters=[("day", "!=", dt.date(2024, 1, 2))])
        reference(t, "v2_partitioned", snapshot_index=1, filters=[("id", ">=", 10)])

        # v1, unpartitioned, manifests written without compression.
        t = cat.create_table("fixtures.v1_plain", schema=SCHEMA, location="iceberg/v1_plain",
                             properties={"format-version": "1", "write.avro.compression-codec": "null"})
        t.append(rows(0, 15))
        t.append(rows(15, 15))
        t.overwrite(rows(100, 5))
        t = cat.load_table("fixtures.v1_plain")
        for i in range(len(t.metadata.snapshots)):
            reference(t, "v1_plain", snapshot_index=i)
        reference(t, "v1_plain", filters=[("id", ">", 101)])

        # truncate / month / day partitions. pyiceberg's writer needs its optional Rust extension for
        # non-identity transforms, so these files are written by pyarrow (one file per partition,
        # without field ids) and registered with add_files, which records the partition values and a
        # name mapping -- the reader's path for files that carry no field ids. (add_files cannot infer
        # a bucket partition, so no fixture is bucket-partitioned.)
        def internal(v):
            """A Python value in Iceberg's internal form (what Transform.transform expects)."""
            if isinstance(v, dt.datetime):
                return int(fingerprint_value(v, pa.timestamp("us", tz="UTC")))
            if isinstance(v, dt.date):
                return (v - dt.date(1970, 1, 1)).days
            return v

        def add_partitioned(t, table, transforms, tag):
            groups = {}
            for r in table.to_pylist():
                key = tuple(tr.transform(t.schema().find_field(col).field_type)(internal(r[col]))
                            for col, tr in transforms)
                groups.setdefault(key, []).append(r)
            paths = []
            for i, (_, rs) in enumerate(sorted(groups.items(), key=lambda kv: str(kv[0]))):
                path = "%s/data/%s-%03d.parquet" % (t.location(), tag, i)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                pq.write_table(pa.Table.from_pylist(rs, schema=table.schema), path)
                paths.append(path)
            t.add_files(paths)

        dense = rows(0, 50)
        dense = dense.set_column(8, "name", pa.array(["item-%03d" % (i * 7 % 50) for i in range(50)]))
        t = cat.create_table("fixtures.transforms", schema=SCHEMA, location="iceberg/transforms")
        with t.update_spec() as u:
            u.add_field("id", TruncateTransform(25), "id_trunc")
            u.add_field("name", TruncateTransform(6), "name_trunc")
            u.add_field("day", MonthTransform(), "day_month")
        t = cat.load_table("fixtures.transforms")
        tr = [("id", TruncateTransform(25)), ("name", TruncateTransform(6)), ("day", MonthTransform())]
        add_partitioned(t, dense.slice(0, 40), tr, "a")
        t = cat.load_table("fixtures.transforms")
        add_partitioned(t, dense.slice(40), tr, "b")
        t = cat.load_table("fixtures.transforms")
        reference(t, "transforms")
        reference(t, "transforms", filters=[("id", "==", 17)])
        reference(t, "transforms", filters=[("id", ">", 30)])
        reference(t, "transforms", filters=[("name", ">=", "item-03")])
        reference(t, "transforms", filters=[("name", "<", "item-01")])
        reference(t, "transforms", filters=[("day", "<", dt.date(2024, 1, 3))])
        reference(t, "transforms", filters=[("day", ">=", dt.date(2024, 2, 1))])

        t = cat.create_table("fixtures.by_day", schema=SCHEMA, location="iceberg/by_day")
        with t.update_spec() as u:
            u.add_field("ts", DayTransform(), "ts_day")
        t = cat.load_table("fixtures.by_day")
        add_partitioned(t, rows(0, 20), [("ts", DayTransform())], "d")
        t = cat.load_table("fixtures.by_day")
        reference(t, "by_day")
        reference(t, "by_day", filters=[("ts", ">=", dt.datetime(2024, 1, 3, 12, 0, 0))])
        reference(t, "by_day", filters=[("ts", "<", dt.datetime(2024, 1, 2))])

        position_deletes(cat)
    finally:
        os.chdir(cwd)
        shutil.rmtree(catalog_dir, ignore_errors=True)


def position_deletes(cat):
    """A v2 table whose latest snapshot adds a position delete file. pyiceberg does not write delete
    files, so the delete manifest and the snapshot are assembled from its manifest writers."""
    from pyiceberg.manifest import (DataFile, DataFileContent, FileFormat, ManifestContent, ManifestEntry,
                                    ManifestEntryStatus, ManifestWriterV2, write_manifest_list)
    from pyiceberg.typedef import Record

    t = cat.create_table("fixtures.position_deletes", schema=SCHEMA, location="iceberg/position_deletes")
    t.append(rows(0, 10))
    t = cat.load_table("fixtures.position_deletes")
    snap = t.current_snapshot()
    data_path = next(iter(t.scan().plan_files())).file.file_path
    del_path = "iceberg/position_deletes/data/deletes-0.parquet"
    pq.write_table(pa.table({"file_path": [data_path, data_path], "pos": pa.array([0, 3], pa.int64())}), del_path)

    class DeleteManifestWriter(ManifestWriterV2):
        def content(self):
            return ManifestContent.DELETES

        @property
        def _meta(self):
            return {**super()._meta, "content": "deletes"}

    new_id = snap.snapshot_id + 1
    io = t.io
    mpath = "iceberg/position_deletes/metadata/deletes-m0.avro"
    df = DataFile.from_args(content=DataFileContent.POSITION_DELETES, file_path=del_path,
                            file_format=FileFormat.PARQUET, partition=Record(), record_count=2,
                            file_size_in_bytes=os.path.getsize(del_path))
    df.spec_id = 0
    with DeleteManifestWriter(t.spec(), t.schema(), io.new_output(mpath), new_id, "deflate") as w:
        w.add_entry(ManifestEntry.from_args(status=ManifestEntryStatus.ADDED, snapshot_id=new_id,
                                            sequence_number=snap.sequence_number + 1, data_file=df))
    manifests = snap.manifests(io) + [w.to_manifest_file()]
    lpath = "iceberg/position_deletes/metadata/snap-%d-deletes.avro" % new_id
    with write_manifest_list(2, io.new_output(lpath), new_id, snap.snapshot_id, snap.sequence_number + 1,
                             "deflate") as lw:
        lw.add_manifests(manifests)
    meta_path = t.metadata_location
    meta = json.load(open(meta_path))
    s = dict(meta["snapshots"][-1])
    s.update({"snapshot-id": new_id, "parent-snapshot-id": snap.snapshot_id,
              "sequence-number": snap.sequence_number + 1, "manifest-list": lpath,
              "summary": {"operation": "delete", "added-delete-files": "1", "added-position-deletes": "2"}})
    meta["snapshots"].append(s)
    meta["current-snapshot-id"] = new_id
    meta["last-sequence-number"] = snap.sequence_number + 1
    meta.setdefault("refs", {})["main"] = {"snapshot-id": new_id, "type": "branch"}
    meta["snapshot-log"] = meta.get("snapshot-log", []) + [{"snapshot-id": new_id,
                                                              "timestamp-ms": s.get("timestamp-ms", 0) + 1}]
    base = os.path.basename(meta_path)
    num = int(base.split("-")[0])
    out = os.path.join(os.path.dirname(meta_path), "%05d-deletes.metadata.json" % (num + 1))
    json.dump(meta, open(out, "w"), indent=1)


def main():
    cases = []
    delta_tables(cases)
    iceberg_tables(cases)
    import deltalake
    import pyiceberg
    header = {"generated_by": "Tests/Fixtures/lakehouse/generate_lakehouse.py",
              "deltalake": deltalake.__version__, "pyiceberg": pyiceberg.__version__, "pyarrow": pa.__version__}
    # One case per line keeps the file small and its diffs readable.
    with open(os.path.join(HERE, "expected.json"), "w") as f:
        f.write("{\n")
        for k, v in header.items():
            f.write(" %s: %s,\n" % (json.dumps(k), json.dumps(v)))
        f.write(' "cases": [\n')
        f.write(",\n".join("  " + json.dumps(c, separators=(",", ":")) for c in cases))
        f.write("\n ]\n}\n")
    print("wrote %d cases" % len(cases), file=sys.stderr)


if __name__ == "__main__":
    main()
