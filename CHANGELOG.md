# Changelog

## 0.1.0
Everything below is in 0.1.0, the first public release.

Core
- Metal shared-memory Arrow buffers (page aligned, pooled) and primitive/boolean arrays.
- Kernels: sum/min/max/mean, compare, add/sub/mul/div (vectorised, defined integer division by zero),
  filter (one command buffer, GPU scan) and fused filter(where:), take, cast, slice, boolean and/or/not/count/any/all.
- Float64 compare/min/max/filter/take/slice on the GPU via order-preserving bit patterns; NaN semantics.
- GroupBy over dense integer keys: count, sum, mean, min, max (privatised and device-atomic paths), plus a
  sort-based segmented path with no atomics — sumDouble/meanDouble/sumFloatAsDouble/meanFloat and
  min64/max64 — which covers the Float64 and 64-bit min/max cases 32-bit atomics cannot express.
- The key-value-to-dense-id mapping reads an integer key column as the type it is rather than widening it
  to int64 first, and its three dispatches share one command buffer; `hash_mean` reuses the counts the
  accumulation already produced and divides on the GPU, and `hash_sum` returns the accumulator's own
  buffer with a GPU-built validity bitmap instead of a host loop. Same ids, same answers to the bit;
  `sum` by int32 key at 50M rows and a thousand groups went 8.75 -> 4.81 ms and `mean` 9.78 -> 4.80 in
  the A/B run recorded in docs/TO_IMPROVE.md; the published matrix reads 4.89 and 4.91 ms for those cells.
- MetalRecordBatch with filter/take/slice/selecting; struct (+s) C Data import/export; ArrowArrayStream import.
- Batched execution (`MetalContext.batch { }`) and its non-blocking form: `batchAsync` (Swift `async`
  and completion-handler), with `MetalArray.sumAsync`/`meanAsync` for scalars, so the calling thread is
  free while the GPU works.
- Arrow C Data Interface and C Device Data Interface (ARROW_DEVICE_METAL) import and export.
- CPU/GPU router: `sum`, `min`, `max`, `compare`, `add`/`subtract`/`multiply`, `filter` (by mask and
  fused `filter(where:)`) and the group-by sum over at most 1,024 keys run a single-threaded CPU loop
  below the crossover table's row count and the GPU kernel at or above it, with byte-identical Arrow
  output on both paths (float sums reproduce the GPU's summation order bit for bit). The crossover table
  is generated from `Benchmarks/results/router_2026-09-17.json` by `Benchmarks/router_table.py`, whose
  CPU side is the bench's single-core loops rather than the router's own; `router_table.py --from-check`
  fits it from a `Benchmarks/router_check.py` run of the shipped loops instead. `multiply` has its own
  row, fitted from the RouterCPU multiply loop in `Benchmarks/results/router_check_2026-09-23_provisional.csv`
  (the sweep timed `add` only). The group-by sum is routed for uint64 values kept unsigned
  (`GroupBy.sumUnsigned`) as well. A batch always keeps the GPU, and so does `auto` for float columns,
  which have no measured crossover yet. `Benchmarks/router_check.py` times each routed operation under gpu, cpu and auto. `ARROWMETAL_ROUTER=auto|gpu|cpu`, `Router.mode` / `Router.withMode` in Swift,
  `am_router_*` in C, and `am.set_router`, `with am.router(...)`, `am.last_route()` in Python
  (docs/DESIGN.md, "CPU/GPU router").
- The Swift and Python test harnesses and `python/tests/differential_report.py` pin the router to the
  GPU unless `ARROWMETAL_ROUTER` is set, so the suites keep exercising the kernels and
  `ARROWMETAL_ROUTER=cpu` runs them over the CPU loops; the other bindings' suites run under `auto`
  (docs/TESTING.md).

Strings and sorting
- `MetalStringArray` (utf8): byte/char length, equals/starts_with/ends_with/contains, MurmurHash3, GPU filter/take,
  GPU dictionary encoding (hash, argsort, byte-comparing boundaries, rank scan and gather; collisions detected
  and re-hashed, with the host path as the final fallback); import/export through the C Data Interface
  (large_utf8 narrowed on import).
- GPU LSD radix sort: `argsort`, `sorted`, `MetalRecordBatch.sorted(by:)`; stable, nulls last by default
  (`null_placement` puts them at either end), IEEE total order.
  The nulls are partitioned out before the sort rather than lifted out of the permutation afterwards, and
  `sorted()` rebuilds the values from the sort's own keys instead of gathering them through it.
- `topK` for any k: a GPU radix select (digit histogram over the order-preserving key, one compaction pass
  keeping only the rows that can still win, a refinement round, then a bitonic or radix ordering) reads the
  column twice whatever k is, and matches the full sort index for index. The per-threadgroup selection
  (threshold plus a bitonic compaction in threadgroup memory) still serves small inputs at k <= 1024, and
  the sort covers k near n and the case where fewer than k rows are non-null.
- `kthElement(k, largest:)`: the exact k-th smallest or largest value, by the same selection with the winners
  counted but never written. `quantile` and `approximateMedian` run on it instead of a full sort.

Temporal, binary and dictionary types
- `MetalTemporalArray`: date32/date64, time32/time64, timestamp (unit + optional timezone) and duration,
  forwarding compare/filter/take/slice/min/max/sort/argsort to the int32 or int64 array underneath.
- GPU calendar fields in UTC (`year`, `month`, `day`, `dayOfWeek`, `hour`, `minute`, `second`), plus
  `toDate32()` and `castUnit(to:)`; civil-from-days arithmetic, correct for negative epochs.
- `binary` and `large_binary` share the utf8 layout (`MetalStringArray.isBinary`, exported as "z").
- Dictionary-encoded arrays import as int32 codes plus a value array; selection runs on the codes,
  `decode()` materialises with `take`, and export writes `schema.dictionary` / `array.dictionary`.
- C ABI: `am_temporal_extract`, `am_temporal_cast_unit`, `am_dictionary_decode`; Python `year()` ...
  `second()`, `cast_unit()`, `decode()` and temporal/binary/dictionary types on `MetalArray.type`.

Execution model
- `MetalContext.batch { }`: one command buffer per chain; kernels read lengths from device buffers so pending
  filter results chain without a CPU sync; pool parks buffers while a batch is open.
- Software IEEE-754 Float64 on the GPU: `add`/`sub`/`mul`/`div` are correctly rounded and bit-exact
  against Swift's `Double` (`DoubleMathTests`); `sum` uses the same adder but reassociates over
  threadgroups, so it is checked to a tolerance rather than bit for bit.

Numerics
- Checked arithmetic: `add_checked` … `power_checked`, `negate`/`abs`/`sqrt`/`ln`/`log10`/`log2`/`log1p`,
  the checked shifts, `logb_checked`, and checked `cumulative_sum`/`cumulative_prod`/`pairwise_diff`.
  Each is the unchecked kernel plus a read-only check pass in the same command buffer, so the values are
  bit-identical and the cost is one GPU round trip; a failure names the Arrow message and the first row.
- Trigonometry: the twelve trigonometric and hyperbolic functions, their seven `_checked` twins and
  `atan2` — Metal's library functions for float32 (hyperbolics rewritten from well-conditioned
  identities), a software binary64 implementation for float64. Measured within 5 ulp of the host libm
  over 1,000,003 random arguments per function (float64 worst 5, `tan`; float32 worst 4 — `TrigTests`).
  Above \|x\| ≈ 5e13 the argument reduction degrades in step with the argument's own ulp (7 ulp measured
  at 5e13) and \|x\| ≥ 2^62 returns NaN — the one documented difference from libm.
- The remaining element-wise math: `expm1`, `log1p`, `logb`, `hypot`, `round_to_multiple`, `round_binary`
  and `round` with all ten Arrow round modes and any `ndigits`.
- Float64 `sqrt`/`exp`/`ln`/`log2`/`log10`/`power` (and their `_checked` twins) in real software
  binary64, not a `float` evaluation widened back: `sqrt` is correctly rounded — bit-identical to
  Foundation over 10^6 random bit patterns plus the extremes — and the rest measure 1 ulp over 10^6
  inputs each across the whole domain, against a 2-ulp bound the tests assert.
  `power` on a float64 column is new; it used to raise. The trade is throughput, stated in
  `docs/BENCHMARKS.md`: a binary64 logarithm costs about 25x what the seven-digit one did.
- Software binary64 arithmetic made faster without losing a bit: `clz` normalisation instead of shift
  loops, four 32x32 partial products instead of an emulated 64x64, and a Newton reciprocal with an exact
  128-bit remainder correction instead of a 57-step restoring division. Float64 `add` and `multiply`
  now run at about 390 GB/s at 50M rows, the fastest these single-pass float64 rows reach, and `divide` within 15% of it
  (331 GB/s) — docs/DESIGN.md.
- Float classification (`is_nan`, `is_finite`, `is_inf`) as raw bit-pattern tests, and the boolean
  operators the bitmap family lacked: `xor`, `and_not`, `and_not_kleene`.
- Statistical aggregates `skew`, `kurtosis` and `tdigest` (GPU sort, host centroid merge), with grouped
  forms of all three.

Grouped aggregation and windows
- `GroupByKeys`: group-by over arbitrary key columns — sparse or negative integers, floats (with
  `-0.0 == 0.0` and one NaN group), booleans, temporal values, utf8, binary, dictionary and decimal, and
  several columns folded pairwise into an injective composite. A range path (mark, scan, rank) for narrow
  integer-like columns and the dictionary-encode sort path for everything else.
- All 24 Arrow `hash_*` names against those ids: the fused `hash_min_max`, `hash_count_all`,
  `hash_first`/`hash_last`/`hash_first_last`, `hash_one`, `hash_list`, `hash_distinct`,
  `hash_approximate_median` and `hash_quantile` (exact, from a segmented sort), `hash_product`,
  `hash_variance`/`hash_stddev`, `hash_skew`/`hash_kurtosis`, `hash_tdigest` and `hash_pivot_wider`.
- `hash_count_distinct` is a GPU hash **set** over the (key, value) pair — one insert pass over the rows,
  then a histogram over the group ids of the occupied slots — instead of a dictionary encoding, a packed
  int64 column and a `unique()` over it. 7.1 ms at 10M rows and 41.8 ms at 50M for a thousand groups,
  and 7.3 ms at 10M for a hundred thousand — 60.7 ms at 50M rows and ten million groups is the worst
  case (docs/TO_IMPROVE.md).
- Several integer key columns whose ranges multiply out to at most 2^24 (and at most the row count) are
  packed into one key in a single pass instead of folded pairwise through a range encoding each. The
  dense ids are the fold's own, value for value, so the group order is unchanged.
- Window and ordering functions: `rank`, `dense_rank`, `row_number`, `rank_quantile`, `rank_normal`,
  `winsorize`, `shift`, rolling `sum`/`mean`/`min`/`max`, `cumulative_prod`/`cumulative_mean`,
  `pairwise_diff`, multi-column `lexsort_indices`, `partition_nth_indices`, `inverse_permutation`,
  `scatter`, `unique`, `value_counts`, `count_all`, `true_unless_null`, `first_last` and `random`
  (Philox4x32-10, seed-only stream).

Strings and Unicode
- The `utf8_is_*` predicate family plus `ascii_is_printable`, `ascii_is_title` and `string_is_ascii`. Each
  answers every row on the GPU and re-decides on the host only the rows carrying a byte >= 0x80, so an
  all-ASCII column never leaves the device.
- Case and layout transforms: `ascii_title`, `utf8_capitalize`, `utf8_title`, `utf8_swapcase`,
  `utf8_center`, `utf8_zero_fill`, `utf8_replace_slice`, `binary_replace_slice`, the `utf8_trim*` family
  against the full Unicode whitespace class, and `utf8_normalize` (NFC/NFKC/NFD/NFKD, host-side).
- `extract_regex_span`, `binary_join` over `list<utf8>`, and string/binary value sets for `is_in` and
  `index_in` through the GPU string hash table.

Temporal, timezones and the rest of the type matrix
- `week` with all its `WeekOptions`, `day_of_week` with `DayOfWeekOptions`, `us_week`, `us_year`,
  `iso_calendar`, `year_month_day`, `subsecond`, and every `*_between` difference from `years_between`
  down to `nanoseconds_between`.
- The three interval layouts (`tiM`, `tiD`, `tin`) with `add_interval` and the three interval-valued
  differences; `assume_timezone`, `local_timestamp` and `is_dst` on the host, where the IANA database is.
- `null`, `float16`, `decimal32`/`decimal64` (GPU widening to decimal128 and narrowing back),
  `fixed_size_binary`, `list_view`/`large_list_view` import, `map` with `map_lookup`, `list_slice`,
  `list_parent_indices`, run-end encoding and extension types.
- Conditional transforms: `case_when`, `choose`, `replace_with_mask`, `fill_null_forward`/`_backward`,
  `indices_nonzero`, `make_struct`, `pivot_wider`, and a 64-bit element-wise `hash64` (`am_hash64`, plus
  an FNV-1a form for `fixed_size_binary`) — an ArrowMetal extension, not one of the 307 Arrow names.
- The Arrow IPC writer takes every one of these types, nested children recursively: decimal32/64/128/256,
  `float16`, `fixed_size_binary`, the three interval units, `null`, `list`/`large_list`/`fixed_size_list`,
  `struct`, `map`, dense and sparse unions, run-end encoded columns and extension types (whose
  `ARROW:extension:*` keys ride in the field's `custom_metadata`). Field nodes and buffers are written in
  Arrow's pre-order with the type metadata the spec prescribes — decimal precision/scale/bitWidth, the
  list child field, a map's `entries` struct with `keysSorted` and a non-nullable key, struct and union
  child names, union mode and typeIds, the interval unit — and pyarrow 25 reads each one back with the
  right type and the right values from both the file and the stream encapsulation. A sliced utf8, binary
  or list column now rebases its offsets and writes only the bytes and child elements its own rows cover,
  instead of the prefix it shares with its parent.
- The Arrow IPC reader builds every one of those types too. It walks a batch's field nodes and buffers in
  the pre-order the spec defines, recursing into children, so `list`/`large_list`/`fixed_size_list`,
  `struct`, `map`, both unions and run-end encoding come back as the engine's own nested arrays, and
  decimal32/64/128/256, `fixed_size_binary`, `float16`, the three interval units and `null` come back as
  theirs; 64-bit offsets are narrowed to the 32-bit ones the engine stores, with a clear error above 2 GB.
  Every file pyarrow 25 writes for a type the engine can hold now reads back with pyarrow's values, in
  both encapsulations.
- The IPC reader decompresses **LZ4_FRAME and ZSTD** bodies, per buffer, including the -1 marker for a
  buffer a writer left uncompressed, in record batches and dictionary batches alike. ZSTD goes through
  the same `dlopen` of libzstd the Parquet reader uses and names the missing library rather than
  returning wrong data; LZ4 frames are decoded into one contiguous output, so linked blocks decode as
  well as independent ones. The writer still emits uncompressed bodies only.
- IPC dictionaries follow message position: a dictionary applies to the batches after it, so a stream may
  replace one part way through or extend it with a delta, while the file format — which indexes every
  dictionary in its footer — still refuses a replacement. Reading batches out of order replays the
  dictionary messages from the first.
- Three IPC reader fixes: a batch whose field nodes or buffers are not all consumed is rejected as
  malformed rather than read with the surplus ignored; a field is classified by its type before it is
  judged for having children, so an unsupported type is named for what it is; and dictionary
  materialisation no longer reads every `DictionaryBatch` in the source on the first batch read.
- The IPC reader reads the view types: `utf8_view` and `binary_view` materialise to the utf8 / binary
  layout in a CPU pass sharded over the cores, `list_view` and `large_list_view` to a list (child used as
  it is when the rows are in order, gathered with `take` otherwise). The writer writes their classic
  counterparts.
- The IPC reader reads big-endian sources, byte swapping every buffer by element width (decimal limbs
  reordered, interval parts and view headers swapped one by one); fixtures are Arrow's 1.0.0 big-endian
  integration files. The writer stays little-endian.
- The IPC reader keeps field `custom_metadata` on its schema and returns `arrow.fixed_shape_tensor` columns
  as `.extended`, round-tripping with pyarrow's FixedShapeTensorArray (`ArrowFixedShapeTensorType`); tensor
  metadata whose shape product overflows is a malformed-data error. Columns naming any other extension type
  read as their storage, as before. IPC Tensor and SparseTensor messages are refused with an error that
  names them.

Arrow function coverage
- `arrowmetal.functions`: a registry with one entry per Arrow v25 compute function name — all 307, the
  283 in the C++ docs plus the 24 `hash_*` — each carrying its status, Swift file, ArrowMetal call, note,
  an executable call adapter under Arrow's own option names and a `pyarrow.compute` oracle. `list_functions()`
  and `call_function()` mirror pyarrow's introspection API.
- `docs/ARROW_FUNCTIONS.md`, generated from that registry, is the by-name coverage page: 283 gpu, 17 cpu,
  7 partial, 0 missing, 0 pending — all 307 names.

Fused expression compiler and lazy query engine
- Expression compiler (`Sources/ArrowMetal/Expr`, docs/EXPR.md): a whole Arrow compute expression — arithmetic,
  comparisons, null logic, casts, string predicates — lowered to one runtime-generated MSL kernel, so the
  inputs are read once and the outputs written once; `batch.query(query().filter(...).sum(...))` in Swift,
  `am.query(table, am.filter(pred).sum(am.col("x")))` in Python. Filter + aggregate, project and dense-key
  group-by fuse into one dispatch; a five-operator expression at 50M rows goes from five kernels to one.
- Lazy query engine (`Sources/ArrowMetal/Engine`, docs/ENGINE.md): `am.scan(table).filter().group_by().agg()
  .sort().limit().collect()` and the Swift `LazyFrame`; an optimizer (predicate pushdown, projection pruning,
  filter fusion, constant folding, expression CSE, join type-check, `explain()`), inner/left/right/outer/
  semi/anti joins on one or several keys incl. utf8, `join_asof` with by-keys and tolerance, window
  functions, `unique`, `explode`, `concat`; the plan runs inside one Metal command buffer. A JSON plan
  grammar over the C ABI (`am_plan_source_create`, `am_plan_run`) for other front ends.

Parquet on the GPU
- A Parquet reader whose column data stays off the CPU for the encodings and codecs listed below
  (docs/PARQUET.md): the host parses only
  the Thrift footer and page headers; decompression (Snappy, LZ4, LZ4_RAW), definition levels, RLE /
  dictionary indices, DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY and BYTE_STREAM_SPLIT
  decode as Metal kernels straight into shared-memory Arrow arrays. ZSTD/GZIP/BROTLI pages decompress on
  the host. Row-group and column selection, nested lists; a small host-side writer for round trips.
  `am.read_parquet(path)` in Python, `ParquetReader` in Swift.
- Nested Parquet columns reassembled from their leaves at any depth: structs (nullable, structs of
  structs, structs of strings), maps (`map<K, V>` with nullable and nested values), and lists nested in
  lists, in structs and in maps (`list<list<T>>`, `list<struct<...>>`, `struct<list<...>>`,
  `list<map<...>>`). Three levels per field from the schema, one flag kernel, one prefix sum and one scatter
  per field (`Parquet/ParquetNested.swift`). Checked value for value and type for type against
  `pyarrow.parquet.read_table` on files written by pyarrow, DuckDB and Polars
  (`Tests/Fixtures/generate_parquet_nested.py`, `ParquetNestedTests`, `python/tests/test_parquet_nested.py`).
- The Parquet reader applies the file's `ARROW:schema` metadata: timestamp time zones, durations,
  decimal32 / decimal64, fixed-size lists, string and binary dictionary (categorical) columns and
  extension types come back as their stored Arrow types, at any depth for zones, durations and decimals;
  a categorical of any other value type reads as that type, as in pyarrow; a column annotated `UNKNOWN`
  reads as the `null` type rather than an all-null `int32`. Field and schema metadata are served by
  `arrowFieldMetadata(column:)` / `arrowSchemaMetadata`, `am_parquet_field_metadata` /
  `am_parquet_schema_metadata`, and carried on `read_parquet_table`'s Table. The stored fields match the
  columns by position; a stored schema of another width is ignored, as pyarrow ignores it, and one that
  is not base64 or not a Schema message is ignored where pyarrow refuses the file
  (`ParquetArrowSchemaTests`).
- Parquet filter values: Python raises on a filter value that is not a str, bool, int or float (a
  `datetime.date` or `Decimal` used to rule out every row group without an error), and a literal of
  another kind than its column's never rules a row group or page out.
- Page-level skipping for Parquet statistics filters: with a column index and offset index in the file,
  the pages whose min/max cannot match (or that hold only nulls, by their null count as well as their
  flag, since Polars flags pages holding a NaN) are never read, decompressed or decoded; the row-group
  min/max of a column whose index shows such a flagged page do not drop the row group, since Polars
  leaves those pages out of them; a row group every page of which is ruled out is dropped, and every
  column is trimmed to the same candidate rows. The matching rows are identical with and without the
  index; `usePageIndex` / `use_page_index` / `am_parquet_set_page_index` turn it off and
  `lastReadStatistics` / `last_read_stats` / `am_parquet_last_read_stats` count the pages decoded and
  skipped (`ParquetPageIndexTests`).
- Parquet split-block bloom filters (pyarrow's `bloom_filter_options`, DuckDB's): an `==` filter drops
  the row groups whose bloom filter rules its literal out, before any page is read; `useBloomFilters` /
  `use_bloom_filters` / `am_parquet_set_bloom_filters` turn it off (`ParquetBloomFilterTests`).
- Parquet repetition levels were decoded with a 4-byte scratch buffer for the per-level ranks the kernel
  writes, so a list column with more than 4,096 level entries in one read wrote past that buffer into
  host memory (the allocation is `posix_memalign` memory wrapped for the GPU); the scratch buffer now
  has a slot per level. `test_parquet_nested.py::test_repeated_columns_past_one_allocation_page` fails
  without the fix and passes with it.
- A one-level Parquet list column read from row groups that a filter removed entirely now comes back
  empty instead of raising "a list column must have definition levels".
- Parquet `!=` filters on a `float` or `double` column never rule out a row group or page: writers leave
  NaN out of min / max, so a page of one value with a NaN in it was skipped and its NaN row lost with
  the page index on. pyarrow's filtered read still rules out such a row group; ArrowMetal returns its NaN
  rows (`test_not_equal_keeps_a_nan_hidden_in_a_constant_page`, `ParquetFilterEdgeTests`).
- A pyarrow `list<null>` column (and any `null`-typed leaf below a list or map) reads as its Arrow type,
  the null child with one slot per element (`test_null_type_below_lists_maps_and_structs`).
- `uint64` statistics are read as unsigned, and an integer filter literal above the int64 range stays
  exact: `u64 >= 2**63` used to rule out every row group (`test_uint64_literals_past_the_signed_range`).
- Python quotes a string filter value with `"` and `\` escaped, so a `;` or quote inside it is part of
  the value; a column name holding `= ! < > ;` raises.
- A restored Parquet dictionary type has `int32` indices and no ordered flag, where pyarrow keeps the
  stored index type and flag; listed under Limits in docs/PARQUET.md and tested.
- Delta Lake and Apache Iceberg tables (docs/LAKEHOUSE.md): `am.read_delta` / `am.read_iceberg` (and
  `*_table` for a pyarrow.Table), `DeltaTable` / `IcebergTable` in Swift, `am_delta_read` /
  `am_iceberg_read` in C. The Delta log (JSON commits, single and multi-part checkpoints read with the GPU
  Parquet reader) and the Iceberg metadata (v1 and v2, Avro manifest lists and manifests through a small
  CPU Avro reader with the null, deflate and snappy codecs) are resolved on the CPU; time travel, partition
  columns, Delta column mapping `none`/`name`, Iceberg columns by field id (renames, added columns, int to
  long); filters prune files by partition values and statistics and are applied to the rows. Unimplemented
  reader features (deletion vectors, column mapping `id`, Iceberg delete files, unknown features) are
  refused with an error naming them. Checked against `deltalake` 1.6.5 and pyiceberg 0.12.0
  (`LakehouseTests`, `python/tests/test_lakehouse.py`); `Benchmarks/lakehouse_bench.py` for timings.
- Lakehouse reads: a Delta empty-string partition value reads as null for every type, as the protocol
  and `deltalake` have it; string row-group pruning is byte-wise, the row filter's order, so decomposed
  strings are no longer pruned away; Iceberg data-file paths are opened as written (pyiceberg's
  `grp=x%3Dy` directories); a float32 column compares with a double literal exactly, as pyarrow does.
  Filter literals at the edges of their types (doubles past the Int64 range, huge years, non-ASCII digits,
  decimals over 38 digits) and malformed Avro manifests or Delta `partitionValues` are answers or errors,
  never a crash or a hang; a negative Delta version other than -1 (C) is an error.
- Lakehouse reads: a NaN Delta float partition is kept for `!=` (it was pruned for every comparison); a
  Delta reader protocol 3 whose `readerFeatures` is missing or not a list of strings, an Iceberg snapshot
  with neither a manifest list nor manifests, and a data file holding none of the table's columns are
  errors instead of reads.

CSV on the GPU
- A CSV reader that parses on the GPU (docs/CSV.md): a quote-aware structure scan (the RFC 4180 parser as
  a state table, run per block from every start state and prefix-composed), pyarrow's type inference
  (null, int64, bool, date32, time32, timestamp with and without a zone, float64, string, binary) from a
  sample and checked on every row while converting, and one row-major kernel converting every column.
  Options follow pyarrow's ReadOptions / ParseOptions / ConvertOptions. `CSVReader` in Swift,
  `am_csv_open` / `am_csv_read` in C, `am.read_csv` / `am.read_csv_table` in Python; differential-tested
  against `pyarrow.csv.read_csv`.
- `MetalStringArray.parse(Double.self)` / `parse(Float.self)` run on the GPU (Eisel-Lemire in integer
  arithmetic), bit-identical to the Swift initialisers they replace, which still parse the rows the GPU
  cannot decide exactly.
- CSV reader review fixes: `scan_block_bytes` of 2^32 or more no longer traps (any positive size only
  changes the speed); a fractional timestamp outside int64 nanoseconds is not inferred as timestamp[ns]
  and raises when forced, as in pyarrow; `skip_rows_after_names` skips rows without a width check and
  counts empty lines, as pyarrow does; ragged-row errors quote at most 100 bytes of the row, as pyarrow's
  do; `delimiter` equal to `quote_char` is accepted; `am_csv_batch_column_name_length` and
  `am_csv_last_error` carry names and messages that hold NUL bytes.
- CSV reader second-round fixes: a ragged row that runs to the end of the file inside an open quote is
  quoted without its last line terminator, as pyarrow quotes it; option strings (`include_columns`,
  `column_types` names, `column_names`, `null_values`, `true_values`, `false_values`) travel with their
  byte lengths (`am_csv_options.*_lengths`), so a NUL byte inside one matches as in pyarrow; a NUL
  `delimiter`, `quote_char` or `decimal_point` and a bytes path are refused, as pyarrow refuses them.

JSON on the GPU
- A newline-delimited JSON reader that parses on the GPU (docs/JSON.md): bit-mask structure passes find
  the records, one thread per record validates the grammar with RapidJSON's error texts, keys match
  fields by byte compare and GPU dictionary encoding, and strings and ISO-8601 timestamps decode as
  kernels; number text goes through `MetalStringArray.parse`. Type inference, field order, missing keys,
  nested structs and lists, `explicit_schema` and `unexpected_field_behavior` follow
  `pyarrow.json.read_json`, compared input by input in `python/tests/test_json.py`, with the documented
  differences each tested. `am.read_json` / `am.read_json_table` in Python, `JSONReader` in Swift,
  `am_json_open` / `am_json_read` in C; `Benchmarks/json_bench.py` against pyarrow, Polars, pandas and
  DuckDB.

Out-of-core streaming
- A streaming executor for datasets larger than memory (docs/STREAMING.md): Arrow IPC files/directories
  (parallel readers with readahead and backpressure) or any Arrow C Stream flow through the GPU one
  record batch at a time; filter/project to a sink, sum/count/min/max/mean/variance, group-by with the
  state resident on the GPU, top-k, sort with a bounded k-way merge, broadcast and grace hash joins,
  a HyperLogLog `count_distinct_approx` (0.0073 % error on 570M rows) and a t-digest quantile. On a
  30.21 GB IPC directory: filter + sum in 0.53 s (ties Polars' 0.515), the approximate count-distinct
  7.2x faster than Polars (0.532 s against 3.825) on 8.8 GB of peak RSS against Polars' 17.0; top-k,
  sort + limit and joins are rows where Polars and DuckDB are ahead, and §9 says so.

Integrations
- Polars (docs/POLARS.md): a zero-copy bridge with `.arrowmetal` namespaces on Series/DataFrame/LazyFrame,
  a Rust expression plugin (`polars-plugin/`) that runs inside a lazy plan, and a streaming hand-off.
- DuckDB (docs/DUCKDB.md): a Python bridge that is copy-free where DuckDB returns one chunk and
  assembles the DataChunks into one buffer otherwise (`duckdb_aggregate`, `duckdb_group_by`, streaming
  `duckdb_batches`) and a loadable C-API extension (`duckdb-extension/`) exposing seven kernels as SQL
  functions; its answers are checked against DuckDB's by the 11 `@extension` tests in
  `python/tests/test_duckdb.py`, which run once `duckdb-extension/build.sh` has built it, and it is not
  yet a speedup (2048-row vectors).
- DuckDB rewrite extension (docs/DUCKDB.md §4b): `duckdb-extension/src/arrowmetal_rewrite.cpp`, a C++
  optimizer extension for DuckDB 1.5.5 (the C extension API has no optimizer hook; the duckdb Python
  module exports the C++ symbols a `CPP` extension needs), built by `duckdb-extension/build_rewrite.sh`.
  It replaces an eligible aggregate of unchanged SQL - `sum`/`avg` over integers, `min`/`max` over
  integers, `DATE` and `TIMESTAMP`, `count`, with no key or one integer, `DATE`, `TIMESTAMP` or `VARCHAR`
  key, over a table or Parquet scan - with `ARROWMETAL_AGGREGATE`: a parallel sink into pooled,
  page-aligned slabs imported into ArrowMetal once, then a fused aggregate, a fused dense group-by or the
  hash group-by, with ungrouped and narrow-key plans streamed to the GPU in blocks while DuckDB scans.
  Answers are DuckDB's exactly (`HUGEINT` sums through 32-bit halves, `avg` with DuckDB's own finalizer
  arithmetic, NULL groups, empty inputs), checked by `python/tests/test_duckdb_rewrite.py` with the
  rewrite off and forced. `SET arrowmetal_rewrite = 'auto'` rewrites only at or above the router's
  crossover and the shape class's measured floor (`Benchmarks/duckdb_rewrite_bench.py`, provisional
  results in `Benchmarks/results/duckdb_rewrite_2026-09-23_provisional.csv`); `'off'` and `'force'`
  too; `arrowmetal_rewrites()` and `EXPLAIN` show what happened. Python: `am.duckdb_connect()`,
  `am.duckdb_rewrites(con)`, `am.duckdb_is_rewritten(con, sql)`. In `auto` an ungrouped query is
  rewritten only with three or more of `sum`/`min`/`max`/`avg`, from 50M rows; a `GROUP BY` with no
  aggregates is rewritten like any other group-by. Where DuckDB would have run the group-by as its
  `PERFECT_HASH_GROUP_BY`, a key past the table its planning-time statistics sized (a prepared statement
  run after out-of-range keys were inserted) raises DuckDB's own error, as it does with the rewrite off.
- pandas (docs/PANDAS.md): an `.am` accessor on Series/DataFrame, and an opt-in accel mode that patches a
  documented set of pandas methods, routes to the GPU only when dtype, size and arguments qualify, and
  restores the originals exactly on `uninstall()`.
- Polars engine (docs/POLARS.md, tier 4): `lf.collect(engine=am.MetalEngine())` translates the
  subtrees of Polars' optimised plan that read in-memory frames -- filters, projections, slices,
  sorts, group-bys, aggregates, inner/left/semi/anti joins and `unique`, over a documented set of
  expressions and dtypes -- into ArrowMetal
  plans and runs them on the GPU through Polars' post-optimisation callback, leaving every other node
  to Polars; `engine.last_report` says what ran where and why. Results are Polars' own (float total
  order, `is_in` matching NaN, Kleene logic, Polars' aggregate dtypes and empty-group answers, Float32
  arithmetic without subnormal flushing, division by a literal as Polars' reciprocal multiply, a float
  multiply by -1 as Polars' negation, NaN sign bits included), checked by `python/tests/test_polars_engine.py` against Polars across sizes,
  null ratios, dtypes and chunked and sliced frames. By default it takes the shapes the provisional
  benchmark (`Benchmarks/polars_engine_bench.py`) measured ahead of both Polars engines -- full sorts
  from 1M rows -- and `shapes="all"` takes everything it can translate. Imports of Polars columns are
  cached by buffer address across queries. A float literal of magnitude 2^63 or more, which traps the
  process inside ArrowMetal's expression compiler, is left to Polars.
- Four engine behaviours found by that suite and worked around in `polars_engine.py`, each pinned by a
  strict xfail: String compaction reading bytes under a null slot, a Boolean column's null count lost
  through the plan's sort, a filter rejecting a plan that carries a `date32` column, and a String sort
  returning wrong rows when the column holds a null and a value of 8 bytes or more.
- `import arrowmetal` imports none of the bridges; each loads on first use through one chained
  PEP 562 hook (`_LAZY_HOOKS`).
- The public C header compiles as C, which `test_the_public_c_header_compiles` now holds it to (a
  typedef/function name clash, `am_plan_source`, was a redefinition in both C and C++ and blocked every
  C consumer including the DuckDB extension until the reviewer caught it).

Bindings
- libArrowMetalC C ABI (include/arrowmetal.h) and python/arrowmetal ctypes package (Arrow PyCapsule protocol).
- Four language bindings over that ABI, each with its own suite: `rust/` (`arrowmetal` + `arrowmetal-sys`,
  48 tests and 4 `no_run` doc-tests, docs/RUST.md), `go/arrowmetal` (45 test functions and one runnable example, docs/GO.md),
  `node/` (N-API addon, 62 tests, docs/TYPESCRIPT.md) and `r/arrowmetal` (266 testthat expectations over
  the 34 header entry points it wraps, docs/R.md).
- `python/build_wheel.sh` packages that ctypes package as a macOS arm64 wheel with the dylib bundled in
  `arrowmetal/_lib/`, so an install needs no Swift toolchain; extras `polars`, `duckdb`, `pandas`, `test`.
  The publication steps are in docs/RELEASE.md (step 4 is the PyPI upload).

Fixed
- `am.scan_ipc(...)` (and every stream with no explicit projection) no longer replaces the second of two
  same-named columns with a copy of the first: the default projection looked each column up by name.
  Such a batch now stays positional; batches with unique names take the fused path as before. Found by
  the review of the IPC lane.
- A float literal whose value is 2^63 or more in magnitude no longer ends the host process: the fused
  expression compiler converted every float literal to Int64 even for float targets, and `Int64(Double)`
  traps outside its range. Float targets no longer compute the integer; an integer-typed float literal that
  does not fit 64 bits is an `ExprError` naming it. Found by the review of the Polars engine lane.
- `GroupByKeys._agg` in the Python package took the device handle of a temporary `MetalArray` that was
  released before the C call read it, segfaulting every grouped aggregate whose values arrived as a
  pyarrow array rather than a `MetalArray`.
- `tdigest` built its digest by walking every value on the host. That walk never merged anything: the
  weight limit is scaled by the weight seen so far rather than by the column's final weight, and the k1
  scale function's inverse is bounded by 1, so every centroid holds exactly one value at any
  compression. The digest of a sorted column is that column, so the quantile is read out of it directly
  and the nulls are compacted away before the sort. Same answer to the last bit, 653 ms -> 29 ms at 10M
  rows.
- `list_value_length` ran at 70 GB/s: one thread per row read every offset twice and stored four bytes at
  a time. Eight rows per thread through vector loads and stores, 1.05 ms -> 0.35 ms at 10M rows.
- `sorted()` was `take(argsort())`, and the gather was most of what it cost above the sort. The radix
  sort's own keys are an order-preserving map of the values that is a bijection except on -0.0 and NaN,
  so the sorted values are inverted straight out of the sorted keys — and the sort then drops the
  row-number payload it only carried for the gather. A column that does hold a -0.0 or a NaN copies back
  only the runs they occupy. `sort float64` 9.585 -> 6.503 ms at 10M rows and 50.281 -> 31.948 ms at 50M;
  `sort int64` 9.390 -> 6.142 and 49.409 -> 30.626. The Python `MetalArray.sort` reaches it through the new
  `am_sort_ex`; it used to be `take(argsort())` in the ctypes package and never called this path, and
  it keeps returning the input's type, a dictionary column included.
- Sorting a column with a validity bitmap lifted the nulls out of the finished permutation on the *host*:
  three passes over the whole index array and a sort of the null row numbers, which cost an order of
  magnitude more than the GPU sort it followed. A stable three-way partition (values, NaNs, nulls) now
  runs before the sort instead, so the nulls never enter it and keep their input order by construction,
  and the passes are planned around the value block rather than the column. `sort float64` with 10%
  nulls 139.745 -> 6.837 ms at 10M rows and 725.935 -> 33.669 ms at 50M; with 50% nulls at 50M,
  3,436.130 -> 22.879 ms. (Every figure in these two bullets and in the matching section of
  docs/TO_IMPROVE.md comes from one set of runs, recorded with the scripts that produced it: the 10% rows
  and the plain sorts from the matrix-conditions harness, the 50% row from the shape sweep. TO_IMPROVE.md
  names the file each one is in.)
- `partition_nth_indices` split the column around the selected key with three `compare` + `filter`
  compactions and a concatenation: seven command buffers, and two of its steps ran on the host — the row
  numbers it compacted were filled by a CPU loop and the output was allocated zeroed, a write and a
  memset over 40 MB at 10M rows. One stable counting sort over five buckets in a single command buffer
  instead, 5.13 ms -> 3.02 ms at 10M rows and 19.8 ms -> 12.3 ms at 50M.
- Found by the pre-release review pass, each with a regression test:
  - Expression compiler: an untyped literal that did not fit the other operand was truncated to it
    (`int8 > 200` was true for every row, `uint8 == -1` matched 255); a float literal against an integer
    column was truncated to an integer. Both now widen the pair, also in `if_else`/`fill_null`/`coalesce`/`is_in`.
  - Optimizer: constant folding absorbed a literal into the null-propagating `and`/`or` (only the Kleene
    forms may); `if_else(c, a, a)` dropped the condition's nulls; folding `Int64.min / -1` and
    `abs(Int64.min)` trapped the process; expression CSE deduplicated `select` outputs by name;
    `join_reorder` permuted rows where the order is observable (now only below a sort, a reduction or a
    group-by). `am_plan_explain`/`am_plan_run` leaked autoreleased objects when called from Python.
  - Parquet: nine ways a corrupt file could hang, trap or read out of bounds (unbounded BYTE_ARRAY
    lengths, varint overflow traps, unbounded Thrift nesting, unchecked schema cursors, negative offsets
    and sizes in footer and page headers, oversized dictionaries and FLBA lengths) now error; an explicit
    empty projection returned every column. A projection naming a column twice (and a file with two
    columns of one name) lost one of them, because the Python read returned a plain dict; it now returns
    a positional `ColumnSet`. A dictionary-encoded column — what `read_parquet` returns by default —
    was refused by every reduction, arithmetic, comparison, cast and sort entry point, so the documented
    `read_parquet(p)["price"].sum()` raised; those entry points now decode the codes on the way in. The
    argument-validation paths returned 2 without setting the error string, so `am_last_error()` handed
    the caller an unrelated earlier failure; every non-zero return now names the function and the
    argument, and a filter string `selected_row_groups` could not parse is reported instead of dropped.
  - Kernels: `lexsort` ignored `null_placement` on a utf8/binary key; the regex pre-filter claimed a
    literal no-match for three pattern shapes; `partition_nth_indices` left a NaN behind when nulls
    moved to the front.
  - Integrations: an unknown column name in the DuckDB streaming helpers silently used the last column;
    a Polars aggregate alias equal to a key replaced the keys; `df.am.query` lifted every column of the
    frame; `zero_copy_report` failed on a Polars DataFrame; `count` over an empty DuckDB stream was
    None; numpy scalars and strings were refused by `arrow_table`.
  - The public C header declared `am_plan_source` as both a typedef and a function; no C consumer
    (including the DuckDB extension) could compile it.
  - Polars tier-2 plugin (its tests had only ever skipped, because nobody had built the Rust
    library): the scalar operand of `.add/.sub/.mul/.truediv` crossed as an `f64` and was narrowed
    with Rust `as`, which saturates and truncates instead of refusing — `add(1000)` on an Int8
    column silently became `add(127)`, `add(-1)` on a UInt8 one a no-op, `add(1.5)` on Int64
    `add(1)`, and any integer past 2^53 lost its low bits (`add(2**60 + 1)` added `2**60`). The
    operand now travels exactly and is range-checked against the column dtype, which is what the
    tier-1 bridge's `struct.pack` does.

Quality
- A CPU oracle behind every kernel test — `Sources/ArrowMetal/CPUReference.swift`, a hand-computed
  vector, or a pyarrow 25.0.1 answer pinned as a literal; 769 XCTest cases in 61 files, run in release
  (all 769 executed, 7 skipped, in the last gated run), including a scenario
  matrix over every type, null density, size and sliced input; concurrency and pool tests (docs/TESTING.md).
  CI on GitHub's hosted Apple silicon runs the suite in debug and release, but its GPU is virtual, so the
  GPU tests skip there and only the build, interop and CPU paths are proved (CONTRIBUTING.md).
- `python/tests/test_functions.py` executes the Arrow-name registry: every runnable row is called through
  `call_function` and compared to `pyarrow.compute`, with a second input in a different Arrow type family
  for the rows whose claim spans several, and float tolerances recorded per row in `functions.TOLERANCE`.
  It found that `binary_length`, `binary_repeat` and `binary_reverse` refuse a `binary` column — all
  three now accept `binary` and `large_binary`.
- Differential matrix against `pyarrow.compute` (docs/EVALUATION.md): 39,069 cases per run over 45 column types, 0 unclassified
  divergences. Fixed from its findings: stable null order in argsort and top_k, the -0.0 tie in the sort
  keys (top_k included, which had its own key mapping), NaN kept at the end of a descending sort, float32
  sums in software double, unsigned group-by sums, exact float32 comparison, float32 `sign`/`ceil`/`floor`/
  `trunc`/`round` and element-wise `min`/`max` on subnormals and signed zeros.
- Benchmarks: Swift vs all-core CPU vs Accelerate; Polars/pyarrow/pandas; ArrowMetal from Python in-process; latency mode.
- `Benchmarks/full_matrix.py` measures every CPU library twice: the plain eager idiom and the most parallel idiom
  that library has for the same answer (`polars-lazy` through `pl.LazyFrame` on the in-memory or streaming engine,
  `pyarrow-threaded` through an Acero plan over one record batch per hardware thread), because the eager idioms use
  about one core on the element-wise and reduction rows whatever the pool size. `--verify` asserts the two answer
  the same within 1e-9 relative, reporting the comparisons deliberately skipped (t-digest's partition-dependent
  sketch, pyarrow's threaded grouped `list`) separately; `--cores` and `full_matrix_<date>_cores.txt` report
  cpu_ms/wall_ms per idiom; "fastest CPU" in the report is the best of every idiom, named.
- 2026-09-07: the published baseline is now that parallel one (`Benchmarks/results/full_matrix_2026-09-07-parallel.csv`,
  cores per idiom in `full_matrix_2026-09-07-parallel_cores.txt`: Polars lazy a median of 11.5 cores, pyarrow through
  Acero 11.1, the eager idioms 1.0 on most families). Of 339 measured rows: **145 at or above 3x, 102 between 1x and
  3x, 77 where the fastest CPU idiom is ahead, 15 with no CPU equivalent** — against the eager idioms alone the same
  build read 247 / 62 / 15 / 15, and that CSV (`full_matrix_2026-09-07.csv`) is kept. Ten ArrowMetal rows a
  regression check flagged were re-measured on a quieter machine and eight spliced in place, each saying so in its
  `note`. docs/BENCHMARKS_MATRIX.md, docs/BENCHMARKS.md, docs/TO_IMPROVE.md and the README are written against the
  parallel baseline; the 77 rows to improve are grouped by measured cause in docs/TO_IMPROVE.md, each with what would
  change it.
- Adversarial review pass before release (four independent reviewers over the integrations, the engine and
  expression compiler, the GPU kernels, and the C ABI and Parquet reader): every finding carries a
  regression test; the fixes are the "Fixed" bullets above and the entries in docs/EVALUATION.md.
- The crossover table (docs/CROSSOVER.md, `Benchmarks/crossover.py`, `arrowmetal-bench crossover`): a size sweep of
  the matrix from a thousand rows to fifty million over six families, and the GPU kernel timed against the
  single-core loop a CPU/GPU router would run instead, so the row count from which the GPU path is ahead is
  stated per operation instead of bracketed. Measured 2026-09-17; the router itself is not implemented.
