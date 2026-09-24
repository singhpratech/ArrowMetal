# Delta Lake and Apache Iceberg tables

ArrowMetal reads Delta Lake and Apache Iceberg tables from the local filesystem. The table metadata is
resolved on the CPU, the data files are pruned by partition values and column statistics, and the Parquet
files that remain are decoded by the GPU Parquet reader ([PARQUET.md](PARQUET.md)) with a projection. The
result is the same thing a Parquet read returns: GPU-resident columns (a `ColumnSet` in Python, a
`MetalRecordBatch` in Swift), or a `pyarrow.Table` exported from them without a copy.

Source: `Sources/ArrowMetal/Lakehouse/` (`DeltaTable.swift`, `IcebergTable.swift`, `Avro.swift`,
`LakehouseCommon.swift`, `LakehouseParquet.swift`), the C ABI in `Sources/ArrowMetalC/ArrowMetalC_Lakehouse.swift`
and the last section of `include/arrowmetal.h`, the Python functions in the "Lakehouse tables" section of
`python/arrowmetal/__init__.py`.

## Using it

```python
import datetime as dt
import arrowmetal as am

cols = am.read_delta("events", version=3, columns=["user", "amount"],
                     filters=[("day", ">=", dt.date(2024, 1, 1)), ("amount", ">", 100)])
total = cols["amount"].sum()                                # on the GPU, no import step

t = am.read_iceberg_table("warehouse/db/orders", snapshot_id=5593686759590374981)   # a pyarrow.Table
cols, stats = am.read_iceberg("warehouse/db/orders/metadata/00004-....metadata.json",
                              filters=[("region", "==", "west")], with_stats=True)
stats["files_pruned_by_partition"], stats["manifests_pruned"]
am.delta_latest_version("events"), am.iceberg_current_snapshot("warehouse/db/orders")
```

| Python | Returns |
|---|---|
| `read_delta(path, version=None, columns=None, filters=None, with_stats=False)` | `ColumnSet`, or `(ColumnSet, stats)` |
| `read_delta_table(path, version=None, columns=None, filters=None)` | `pyarrow.Table` |
| `read_iceberg(metadata_path_or_table_dir, snapshot_id=None, columns=None, filters=None, with_stats=False)` | `ColumnSet`, or `(ColumnSet, stats)` |
| `read_iceberg_table(metadata_path_or_table_dir, snapshot_id=None, columns=None, filters=None)` | `pyarrow.Table` |
| `delta_latest_version(path)`, `iceberg_current_snapshot(path)` | the newest Delta version; the current Iceberg snapshot id or `None` |

Swift: `DeltaTable(path:)` with `read(version:columns:filters:)`, `scan(...)` (rows plus
`LakehouseScanStats`), `snapshot(version:)` and `latestVersion()`; `IcebergTable(path:)` with
`read(snapshotId:columns:filters:)`, `scan(...)`, `plan(snapshotId:filters:)`, `schema(forSnapshot:)` and
the parsed metadata (`schemas`, `specs`, `snapshots`). Filters are `ParquetFilter` values, the type the
Parquet reader takes. C: `am_delta_read`, `am_iceberg_read`, `am_delta_latest_version`,
`am_iceberg_current_snapshot` and the `am_lakehouse_batch_*` accessors.

### Filters

`filters` is a list of `(column, op, value)` triples with `op` one of `== != < <= > >=`; all of them must
hold. The column is named as the table's schema names it (the snapshot's schema when time-travelling in
Iceberg). They are used three times: to skip files (partition values; Delta's per-file `minValues` /
`maxValues` / `nullCount`; Iceberg's manifest partition summaries, partition tuples and column bounds), to
skip row groups (Parquet footer statistics), and then on the rows. The result holds exactly the matching
rows, as `deltalake`'s `to_pyarrow_table(filters=...)` and pyiceberg's `scan(row_filter=...)` return them.
This differs from `read_parquet`, whose filters only skip row groups.

A null never matches, including under `!=` (SQL and Arrow semantics); NaN is unequal to everything, so
it matches `!=` and nothing else. Literals are read in the column's type: an integer column compared with
`2.5` behaves as the exact comparison (`> 2.5` is `>= 3`), a literal outside an integer column's range
matches everything or nothing as the comparison says, a date literal is a `datetime.date` or
`"YYYY-MM-DD"`, a timestamp literal is a `datetime.datetime` (converted to UTC when it carries a zone, taken
as UTC when it does not) or an ISO 8601 string, a decimal literal is an integer, a `decimal.Decimal` or a
string that is exact at the column's scale. A literal that does not fit its column is an error naming both
(`test_filter_literals`, `testRowFilterSemantics`, `testDeltaUnknownColumnAndBadLiteral`).
String comparisons are byte-wise UTF-8 and run on the CPU over the unified-memory buffers; the other types
compare on the GPU.

## Delta Lake

| Feature | Support | Test |
|---|---|---|
| JSON commit replay (`add`, `remove`, `metaData`, `protocol`) | Yes | `testReadsMatchReferenceReaders` (every version of every fixture) |
| Single-file checkpoints, read with the GPU Parquet reader (the `map` and `list` columns of the checkpoint included) | Yes | `testCheckpointAgreesWithLogReplay` |
| Multi-part checkpoints (`N.checkpoint.P.T.parquet`) | Yes | the `multipart` fixture in `expected.json` |
| Time travel to any version reconstructable from the log | Yes; a version whose commits were cleaned up and that no checkpoint covers is an error saying so | `testLogCleanup` |
| Partition values of every primitive type, null partitions | Yes | `partitioned` (null partition), `by_day` (date partition) |
| Schema evolution: columns added later read as null from older files | Yes | `evolution` |
| Column mapping `none` and `name` (renamed columns, physical names in files and partition values) | Yes | `column_mapping`, `testDeltaColumnMappingUsesPhysicalNames` |
| Partition pruning and per-file statistics pruning | Yes; the counters are in `stats` | `testDeltaPruning`, `test_generated_delta_pruning_counters` |
| Reader protocol versions 1 to 3; reader features `columnMapping`, `timestampNtz`, `vacuumProtocolCheck` | Yes | `testDeltaReaderFeaturesItImplements` |
| Deletion vectors, column mapping mode `id`, any other reader feature (type widening, v2 checkpoints, variant, unknown names) | Rejected with an error naming the feature | `testDeltaRejectsUnsupportedFeaturesByName`, `test_delta_errors_name_the_feature` |
| Nested columns (struct, array, map) | Rejected when projected, with an error naming the column; project the other columns | `testNestedColumnsAndRemotePaths` |
| Files on remote storage (`s3://` and other schemes) | Rejected with an error naming the scheme; local paths and `file:` URIs are read | `testNestedColumnsAndRemotePaths` |
| Change data feed, `_last_checkpoint` hints, log compaction files | Not used; the log directory listing is the source of truth | |

Timestamp statistics are written at millisecond precision and string statistics may be truncated, so file
pruning skips timestamp columns and strings of 32 characters or more; the row filter still applies.

## Apache Iceberg

| Feature | Support | Test |
|---|---|---|
| Format versions 1 and 2 | Yes | `v1_plain`, `v2_partitioned` |
| The table named by its metadata file, its directory (newest `NNNNN-*.metadata.json` or `vN.metadata.json`), or `version-hint.text` | Yes | `testIcebergMetadataLocation` |
| Current snapshot and any listed snapshot id; a table with no snapshot reads as empty | Yes | every snapshot of every fixture in `expected.json`, `testIcebergSnapshotsAndSchemas`, `testNestedColumnsAndRemotePaths` |
| Manifest lists and manifests (Avro, `null` / `deflate` / `snappy` codecs), v1 snapshots that list manifests directly | Yes | `testAvroManifestCodecs`, `testAvroSnappyContainer`, `test_iceberg_v1_manifests_listed_in_the_snapshot` |
| Columns resolved by field id: renamed columns, added columns (null in older files), `int -> long` promotion | Yes | `v2_partitioned` (rename, added column, promotion) |
| Files without field ids, through `schema.name-mapping.default` | Yes | `transforms`, `by_day` (files registered with `add_files`) |
| Manifest pruning by partition summaries, file pruning by partition tuples and column bounds | identity, `year`, `month`, `day`, `hour`, `truncate[W]` partitions; `bucket[N]` and `void` partitions are read and not pruned | `testIcebergPruning`, `testIcebergTransformProjection`, `test_generated_iceberg_pruning_counters` |
| Relocated tables: paths under the recorded `location` are re-rooted where the metadata was found | Yes | the committed fixtures record relative locations and are read from any working directory |
| Position and equality delete files (v2) | Rejected with an error naming the kind and a delete file | `testIcebergRejectsDeleteFiles` |
| gzip-compressed metadata files | Rejected with an error | `testNestedColumnsAndRemotePaths` |
| Nested columns | Rejected when projected, as for Delta | `testNestedColumnsAndRemotePaths` |

## Where the results differ from the reference readers

Checked by `python/tests/test_lakehouse.py`:

- Types match `deltalake` column for column. pyiceberg returns `large_string` for some string columns;
  ArrowMetal returns `string` (`test_iceberg_fixtures_match_pyiceberg`).
- pyiceberg binds a time-travel scan's row filter against the current schema, so a column renamed after
  the snapshot is not found under its old name; ArrowMetal resolves filters in the snapshot's schema, like
  its projection (`test_time_travel_filters_use_the_snapshot_schema`). pyiceberg also refuses a float
  literal against an integer column; ArrowMetal compares exactly, as pyarrow does
  (`test_generated_iceberg_filters`).
- `deltalake` 1.6.5 and polars 1.44.1 return nulls for every mapped column of the column-mapping fixture
  (Parquet files with physical names and field ids, partition values keyed by physical name, as the Delta
  protocol specifies); the reference for that table is the pyarrow data it was written from
  (`test_column_mapping_reference_readers_return_nulls`, `test_column_mapping_reads_the_source_data`).
  The fixture is written by hand: pyarrow data files and JSON commits.
- Row order is not defined by either format; every comparison sorts.

## Tests and fixtures

`Tests/Fixtures/lakehouse/generate_lakehouse.py` writes the fixture tables with `deltalake` 1.6.5 and
pyiceberg 0.12.0 (the SQL catalog on SQLite, pyarrow 25.0.1): several commits, appends, deletes,
predicate overwrites, a checkpoint, a multi-part checkpoint, partitions of several types, schema evolution,
a hand-written column-mapping table, three tables with reader features the reader refuses, and an Iceberg
table whose last snapshot adds a position delete file. It also writes `expected.json`: the rows the
reference readers return for 62 reads, which `LakehouseTests.swift` replays. `python/tests/test_lakehouse.py`
replays the same file, compares the committed tables with the reference readers live, and generates larger
tables (thousands of rows, many files, a checkpoint, deletes, schema evolution) to compare every filter
comparison over every column type. See [TESTING.md](TESTING.md).

## Performance

`Benchmarks/lakehouse_bench.py` times full, projected and filtered reads of one Delta and one Iceberg table
against `deltalake`, pyiceberg, polars (`scan_delta`, `scan_iceberg`) and DuckDB's `delta_scan` /
`iceberg_scan` (used only when the extensions load with auto-install switched off); the first run used
`deltalake` 1.6.5, pyiceberg 0.12.0, polars 1.44.1 and DuckDB 1.5.5 with its extensions from the local
extension cache. That run,
`Benchmarks/results/lakehouse_2026-09-23_provisional.csv`, was taken while other work shared the machine
and the GPU; it is provisional. On it, the CPU readers are ahead on every read. Timing the phases of
those reads put the time in the per-file Parquet decode, not in the metadata or the concatenation. Files
are decoded several at a time, each thread with its own command buffer; the per-file decode itself (a chain
of small kernels with CPU round trips in between) is the part to improve, and belongs to the Parquet
reader ([PARQUET.md](PARQUET.md)).
