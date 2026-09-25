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
it matches `!=` and nothing else, as in pyarrow. Literals are read in the column's type and compared as
exact values, as pyarrow compares them:

- an integer column compared with `2.5` behaves as `>= 3` for `> 2.5`; a literal outside the column's
  range, including doubles beyond the Int64 range such as `9.25e18`, matches everything or nothing as the
  comparison says (`testFilterLiteralsAtTypeEdges`, `test_filter_literals_at_type_edges`);
- a float32 column compared with a double literal is the exact comparison: `0.1` as a float32 is
  0.10000000149..., so it is `> 0.1` and not `== 0.1` (`testFloat32ComparesExactly`,
  `test_float32_filters_compare_exactly`);
- a date literal is a `datetime.date` or `"YYYY-MM-DD"` (ASCII digits, a year from 0 to 5,000,000, a
  value that fits date32); a timestamp literal is a `datetime.datetime` (converted to UTC when it carries
  a zone, taken as UTC when it does not) or an ISO 8601 string with two-digit hour, minute and second
  fields that fits the column's unit;
- a decimal literal is an integer, a `decimal.Decimal` or a string of ASCII digits that is exact at the
  column's scale and has at most 38 significant digits;
- a string literal is written into the filter text double-quoted, with `"` and `\` escaped as `\"` and
  `\\` (the grammar of the Parquet filter text, `include/arrowmetal.h`), so it may hold `;`, `"`, `\` and
  operator characters (`("s", "<", "a==b")`); a `bytes` literal (for a binary column) is passed the same
  way as its UTF-8 text, so it must be valid UTF-8. A NUL byte ends the C string and is refused in both,
  and so are bytes that are not UTF-8, each with an error saying so. Data columns, partition columns and
  binary columns filter as deltalake and pyiceberg filter them
  (`test_delta_string_literals_holding_quotes_and_semicolons`,
  `test_iceberg_string_literals_holding_quotes_and_semicolons`). A column name cannot hold `=`, `!`, `<`,
  `>`, `;` or `"`, since the filter text's name ends at the first operator character; Python refuses one
  that does (`test_operators_in_a_literal_are_kept_and_in_a_column_name_refused`).

A literal that does not fit its column is an error naming both, never a crash (`test_filter_literals`,
`testRowFilterSemantics`, `testDeltaUnknownColumnAndBadLiteral`, `testFilterLiteralsAtTypeEdges`).
Integer, float, temporal and decimal columns compare on the GPU; string, binary and boolean columns compare
on the CPU over the unified-memory buffers, strings and binary byte-wise, and a comparison whose answer is
known without the data (a literal outside the column's range, a NaN literal) is a constant mask built on the
CPU. Row groups are pruned by the same order: the readers decide string and binary
row-group pruning themselves, byte-wise on the footer's `min_value` / `max_value`, so a decomposed "é"
(which byte-wise sorts below "f") is kept for `< "f"` (`testStringRowGroupPruningIsByteWise`,
`test_delta_decomposed_strings_survive_row_group_pruning`). Numeric, date, timestamp and boolean filters
use the Parquet reader's row-group filter, only where its comparison is exact (an integer column against a
double literal of any magnitude, compared without rounding the integer; a timestamp stored in the table's
unit; floats never under `!=`).

Malformed metadata is an error naming the file, never a crash or a hang: Avro blocks whose sizes or counts
run past the data, records that contain themselves with nothing optional in between, a Delta
`partitionValues` that is not an object of strings and nulls, a Delta reader protocol 3 whose
`readerFeatures` is missing or not a list of strings, an Iceberg snapshot with neither a `manifest-list` nor
`manifests`, and a data file that holds none of the table's columns (a Delta table without column mapping,
or an Iceberg file without field ids in a table without a name mapping) instead of rows of nulls
(`testAvroMalformedContainersAreErrors`, `test_malformed_manifest_list_is_an_error`,
`testDeltaMalformedPartitionValues`, `testDeltaReaderFeaturesMustBeAListOfStrings`,
`testIcebergSnapshotWithoutManifestsIsAnError`, `testDataFilesWithoutTheTableColumnsAreErrors`). `version` is 0 or more
(`None` reads the latest; in C, -1).

## Delta Lake

| Feature | Support | Test |
|---|---|---|
| JSON commit replay (`add`, `remove`, `metaData`, `protocol`) | Yes | `testReadsMatchReferenceReaders` (every version of every fixture) |
| Single-file checkpoints, read with the GPU Parquet reader (the `map` and `list` columns of the checkpoint included) | Yes | `testCheckpointAgreesWithLogReplay` |
| Multi-part checkpoints (`N.checkpoint.P.T.parquet`) | Yes | the `multipart` fixture in `expected.json` |
| Time travel to any version reconstructable from the log | Yes; a version whose commits were cleaned up and that no checkpoint covers is an error saying so | `testLogCleanup` |
| Partition values of every primitive type, null partitions | Yes; an empty partition value is null for every type, strings included, as the protocol specifies and `deltalake` reads it | `partitioned` (null partition), `by_day` (date partition), `testDeltaEmptyStringPartitionIsNull`, `test_delta_empty_string_partition_is_null` |
| Schema evolution: columns added later read as null from older files | Yes | `evolution` |
| Column mapping `none` and `name` (renamed columns, physical names in files and partition values) | Yes | `column_mapping`, `testDeltaColumnMappingUsesPhysicalNames` |
| Partition pruning and per-file statistics pruning | Yes; the counters are in `stats`. A NaN float partition value is kept for `!=` only | `testDeltaPruning`, `test_generated_delta_pruning_counters`, `testDeltaNaNPartitionMatchesNotEqual`, `test_delta_nan_partition_matches_not_equal` |
| Reader protocol versions 1 to 3; reader features `columnMapping`, `timestampNtz`, `vacuumProtocolCheck` | Yes | `testDeltaReaderFeaturesItImplements` |
| Deletion vectors, column mapping mode `id`, any other reader feature (type widening, v2 checkpoints, variant, unknown names) | Rejected with an error naming the feature | `testDeltaRejectsUnsupportedFeaturesByName`, `test_delta_errors_name_the_feature` |
| Nested columns (struct, array, map) | Rejected when projected, with an error naming the column; project the other columns | `testNestedColumnsAndRemotePaths` |
| Files on remote storage (`s3://` and other schemes) | Rejected with an error naming the scheme; local paths and `file:` URIs are read | `testNestedColumnsAndRemotePaths` |
| Change data feed, `_last_checkpoint` hints, log compaction files | Not used; the log directory listing is the source of truth | |

File pruning by the per-file statistics uses integer, float, date and string columns. Timestamp
statistics are written at millisecond precision and string statistics may be truncated, so timestamp
columns and strings of 32 characters or more are not used; boolean, decimal and binary statistics are not
used either. The row filter still applies to every column.

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
| Paths as written: pyiceberg writes the directory of partition value `x=y` as `grp=x%3Dy` and records that path; it is opened as written, and the percent-decoded path is tried only when that does not exist | Yes | `testIcebergPathsAreNotPercentDecodedFirst`, `test_iceberg_partition_values_that_need_escaping` |
| Position and equality delete files (v2) | Rejected with an error naming the kind and a delete file | `testIcebergRejectsDeleteFiles` |
| gzip-compressed metadata files | Rejected with an error | `testNestedColumnsAndRemotePaths` |
| Nested columns | Rejected when projected, as for Delta | `testNestedColumnsAndRemotePaths` |

## Where the results differ from the reference readers

Checked by `python/tests/test_lakehouse.py`:

- Types match `deltalake` column for column. pyiceberg returns `large_string` and `large_binary` for some
  string and binary columns; ArrowMetal returns `string` and `binary` (`test_iceberg_fixtures_match_pyiceberg`,
  `test_iceberg_partition_values_that_need_escaping`, which has a binary column).
- A projection comes back in the order the columns were asked for; pyiceberg's `selected_fields` returns
  them in schema order (`test_iceberg_partition_values_that_need_escaping`).
- Float comparisons follow pyarrow. `deltalake` 1.6.5 agrees on every number but also returns the NaN row
  for some ordering comparisons on a float32 column (`> 0.1`, `< 1e300`); pyiceberg 0.12.0 rounds a double
  literal to float32 before comparing, so its `f > 0.1` leaves out the rows holding `0.1` as a float32,
  which ArrowMetal and pyarrow return (`test_float32_filters_compare_exactly`,
  `test_float32_literal_rounding_differs_from_pyiceberg`).
- pyiceberg binds a time-travel scan's row filter against the current schema, so a column renamed after
  the snapshot is not found under its old name; ArrowMetal resolves filters in the snapshot's schema, like
  its projection (`test_time_travel_filters_use_the_snapshot_schema`). pyiceberg also refuses a float
  literal against an integer column; ArrowMetal compares exactly, as pyarrow does
  (`test_generated_iceberg_filters`).
- pyiceberg 0.12.0 returns every row, nulls included, for a comparison it can decide without the data:
  `!=` on a column added after some files were written returns those files' null-filled rows, and a
  literal beyond the column's range (`i32 < 2**63 - 1`, `f32 < 1e308`) returns the null and NaN rows.
  ArrowMetal keeps its rule (a null never matches; NaN matches only `!=`) and returns the rows pyarrow's
  filter returns (`test_null_rows_under_always_true_filters_differ_from_pyiceberg`).
- `deltalake` 1.6.5 and polars 1.44.1 return nulls for every mapped column of the column-mapping fixture
  (Parquet files with physical names and field ids, partition values keyed by physical name, as the Delta
  protocol specifies). DuckDB 1.5.5's `delta_scan` reads the mapped data columns (`id`, `total`) exactly
  as ArrowMetal does, which checks the file-level mapping, and returns null for the mapped partition column
  `region`, which ArrowMetal fills from `partitionValues` keyed by the physical name. No independent reader
  confirms that partition keying; it follows the protocol text, and the reference for that column is the
  pyarrow data the table was written from (`test_column_mapping_reference_readers_return_nulls`,
  `test_column_mapping_duckdb_reads_data_columns_not_the_partition`,
  `test_column_mapping_reads_the_source_data`). The fixture is written by hand: pyarrow data files and
  JSON commits.
- Row order is not defined by either format; every comparison sorts.

## Tests and fixtures

`Tests/Fixtures/lakehouse/generate_lakehouse.py` writes the fixture tables with `deltalake` 1.6.5 and
pyiceberg 0.12.0 (the SQL catalog on SQLite, pyarrow 25.0.1): several commits, appends, deletes,
predicate overwrites, a checkpoint, a multi-part checkpoint, partitions of several types, schema evolution,
a hand-written column-mapping table, three tables with reader features the reader refuses, an Iceberg
table whose last snapshot adds a position delete file, and `parquet/nfd_strings.parquet` (decomposed
strings in three row groups, for the byte-wise row-group pruning test). It also writes `expected.json`: the rows the
reference readers return for 62 reads, which `LakehouseTests.swift` replays. `python/tests/test_lakehouse.py`
replays the same file, compares the committed tables with the reference readers live, and generates larger
tables (thousands of rows, many files, a checkpoint, deletes, schema evolution) to compare every filter
comparison over every column type. See [TESTING.md](TESTING.md).

## Performance

`Benchmarks/lakehouse_bench.py` times full, projected and filtered reads of one Delta and one Iceberg table
against `deltalake`, pyiceberg, polars (`scan_delta`, `scan_iceberg`) and DuckDB's `delta_scan` /
`iceberg_scan` (used only when the extensions load with auto-install switched off); the first run used
`deltalake` 1.6.5, pyiceberg 0.12.0, polars 1.44.1 and DuckDB 1.5.5 with its extensions from the local
extension cache. The numbers here are from a quiet run over a 2,000,000-row table,
`Benchmarks/results/lakehouse_2026-09-24.csv`; the run conditions at its
start are recorded in `Benchmarks/results/bench_conditions_2026-09-24.txt`. Median wall time in that
file:

| read | `arrowmetal` | `deltalake` / `pyiceberg` | `polars` | `duckdb` |
|---|---:|---:|---:|---:|
| Delta, full | 1788.18 ms | 18.15 ms | 7.16 ms | 66.82 ms |
| Delta, projected | 740.68 ms | 16.17 ms | 4.17 ms | 21.64 ms |
| Delta, filtered | 185.79 ms | 9.93 ms | 3.31 ms | 7.59 ms |
| Iceberg, full | 155.20 ms | 27.66 ms | 12.87 ms | 91.18 ms |
| Iceberg, projected | 46.29 ms | 22.19 ms | 10.81 ms | 36.04 ms |
| Iceberg, filtered | 32.93 ms | 9.95 ms | 7.72 ms | 7.39 ms |

Every CPU reader is ahead of ArrowMetal on every read, so table reads are to improve. The gap is
largest on Delta: a full read takes 1788.18 ms against Polars' 7.16 ms. On Iceberg it is narrower: a
projected read takes 46.29 ms against 36.04 ms for DuckDB and 10.81 ms for Polars. The file also
times `arrowmetal_table`, the same read handed to pyarrow. An earlier run
over a 1,000,000-row table, taken while other work shared the machine and the GPU, is kept as history
in `Benchmarks/results/lakehouse_2026-09-23_provisional.csv`. The benchmark times whole reads; data
files are decoded several at a time by the GPU Parquet reader ([PARQUET.md](PARQUET.md)), each thread
with its own command buffer.
