# Changelog

## 0.1.0 (unreleased, in development)
Everything below ships together as the first public release.

Core
- Metal shared-memory Arrow buffers (page aligned, pooled) and primitive/boolean arrays.
- Kernels: sum/min/max/mean, compare, add/sub/mul/div (vectorised, defined integer division by zero),
  filter (one command buffer, GPU scan) and fused filter(where:), take, cast, slice, boolean and/or/not/count/any/all.
- Float64 compare/min/max/filter/take/slice on the GPU via order-preserving bit patterns; NaN semantics.
- GroupBy over dense integer keys: count, sum, mean, min, max (privatised and device-atomic paths), plus a
  sort-based segmented path with no atomics — sumDouble/meanDouble/sumFloatAsDouble/meanFloat and
  min64/max64 — which covers the Float64 and 64-bit min/max cases 32-bit atomics cannot express.
- MetalRecordBatch with filter/take/slice/selecting; struct (+s) C Data import/export; ArrowArrayStream import.
- Batched execution (`MetalContext.batch { }`) and its non-blocking form: `batchAsync` (Swift `async`
  and completion-handler), with `MetalArray.sumAsync`/`meanAsync` for scalars, so the calling thread is
  free while the GPU works.
- Arrow C Data Interface and C Device Data Interface (ARROW_DEVICE_METAL) import and export.

Strings and sorting
- `MetalStringArray` (utf8): byte/char length, equals/starts_with/ends_with/contains, MurmurHash3, GPU filter/take,
  GPU dictionary encoding (hash, argsort, byte-comparing boundaries, rank scan and gather; collisions detected
  and re-hashed, with the host path as the final fallback); import/export through the C Data Interface
  (large_utf8 narrowed on import).
- GPU LSD radix sort: `argsort`, `sorted`, `MetalRecordBatch.sorted(by:)`; stable, nulls last, IEEE total order.
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
- Software IEEE-754 Float64 (add/sub/mul/div/sum) on the GPU, bit-exact against the CPU.

Numerics
- Checked arithmetic: `add_checked` … `power_checked`, `negate`/`abs`/`sqrt`/`ln`/`log10`/`log2`/`log1p`,
  the checked shifts, `logb_checked`, and checked `cumulative_sum`/`cumulative_prod`/`pairwise_diff`.
  Each is the unchecked kernel plus a read-only check pass in the same command buffer, so the values are
  bit-identical and the cost is one GPU round trip; a failure names the Arrow message and the first row.
- Trigonometry: the twelve trigonometric and hyperbolic functions, their seven `_checked` twins and
  `atan2` — Metal's library functions for float32 (hyperbolics rewritten from well-conditioned
  identities), a software binary64 implementation for float64, within 4-5 ulp of the host libm.
- The remaining element-wise math: `expm1`, `log1p`, `logb`, `hypot`, `round_to_multiple`, `round_binary`
  and `round` with all ten Arrow round modes and any `ndigits`.
- Float64 `sqrt`/`exp`/`ln`/`log2`/`log10`/`power` (and their `_checked` twins) in real software
  binary64, not a `float` evaluation widened back: `sqrt` is correctly rounded — bit-identical to
  Foundation — and the rest are within 1 ulp, measured over 10^6 inputs each across the whole domain.
  `power` on a float64 column is new; it used to raise. The trade is throughput, stated in
  `docs/BENCHMARKS.md`: a binary64 logarithm costs about 25x what the seven-digit one did.
- Software binary64 arithmetic made faster without losing a bit: `clz` normalisation instead of shift
  loops, four 32x32 partial products instead of an emulated 64x64, and a Newton reciprocal with an exact
  128-bit remainder correction instead of a 57-step restoring division. Float64 `add`, `multiply` and
  `divide` now all run at the machine's memory ceiling.
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
  int64 column and a `unique()` over it. 8.6 ms at 10M rows and 45 ms at 50M, whatever the cardinality.
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
  `indices_nonzero`, `make_struct`, `pivot_wider`, and a 64-bit element-wise `hash64` (an ArrowMetal
  extension; Arrow has no such function).

Arrow function coverage
- `arrowmetal.functions`: a registry with one entry per Arrow v25 compute function name — all 307, the
  283 in the C++ docs plus the 24 `hash_*` — each carrying its status, Swift file, ArrowMetal call, note,
  an executable call adapter under Arrow's own option names and a `pyarrow.compute` oracle. `list_functions()`
  and `call_function()` mirror pyarrow's introspection API.
- `docs/ARROW_FUNCTIONS.md`, generated from that registry, is the by-name coverage page: 283 gpu, 17 cpu,
  7 partial, 0 missing, 0 planned — all 307 names.

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
- A Parquet reader whose column data never passes through the CPU (docs/PARQUET.md): the host parses only
  the Thrift footer and page headers; decompression (Snappy, LZ4, LZ4_RAW), definition levels, RLE /
  dictionary indices, DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY and BYTE_STREAM_SPLIT
  decode as Metal kernels straight into shared-memory Arrow arrays. ZSTD/GZIP/BROTLI pages decompress on
  the host. Row-group and column selection, nested lists; a small host-side writer for round trips.
  `am.read_parquet(path)` in Python, `ParquetReader` in Swift.

Out-of-core streaming
- A streaming executor for datasets larger than memory (docs/STREAMING.md): Arrow IPC files/directories
  (parallel readers with readahead and backpressure) or any Arrow C Stream flow through the GPU one
  record batch at a time; filter/project to a sink, sum/count/min/max/mean/variance, group-by with the
  state resident on the GPU, top-k, sort with a bounded k-way merge, broadcast and grace hash joins,
  a HyperLogLog `count_distinct_approx` (0.007 % error on 570M rows) and a t-digest quantile. On a 30 GB
  IPC directory: filter + sum in 0.53 s (ties Polars), count-distinct 7x faster than Polars at a third of
  the memory; top-k, sort + limit and joins are slower than Polars/DuckDB and say so in §9.

Integrations
- Polars (docs/POLARS.md): a zero-copy bridge with `.arrowmetal` namespaces on Series/DataFrame/LazyFrame,
  a Rust expression plugin (`polars-plugin/`) that runs inside a lazy plan, and a streaming hand-off.
- DuckDB (docs/DUCKDB.md): a zero-copy Python bridge (`duckdb_aggregate`, `duckdb_group_by`, streaming
  `duckdb_batches`) and a loadable C-API extension (`duckdb-extension/`) exposing the kernels as SQL
  functions; the extension matches DuckDB's answers and is not yet a speedup (2048-row vectors).
- pandas (docs/PANDAS.md): an `.am` accessor on Series/DataFrame, and an opt-in accel mode that patches a
  documented set of pandas methods, routes to the GPU only when dtype, size and arguments qualify, and
  restores the originals exactly on `uninstall()`.
- `import arrowmetal` imports none of the three; each bridge loads on first use through one chained
  PEP 562 hook (`_LAZY_HOOKS`).
- The public C header compiles as C and C++ (a typedef/function name clash, `am_plan_source`, blocked
  every C consumer including the DuckDB extension until the reviewer caught it).

Bindings
- libArrowMetalC C ABI (include/arrowmetal.h) and python/arrowmetal ctypes package (Arrow PyCapsule protocol).

Fixed
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
  - Polars tier-2 plugin (its 8 tests had only ever skipped, because nobody had built the Rust
    library): the scalar operand of `.add/.sub/.mul/.truediv` crossed as an `f64` and was narrowed
    with Rust `as`, which saturates and truncates instead of refusing — `add(1000)` on an Int8
    column silently became `add(127)`, `add(-1)` on a UInt8 one a no-op, `add(1.5)` on Int64
    `add(1)`, and any integer past 2^53 lost its low bits (`add(2**60 + 1)` added `2**60`). The
    operand now travels exactly and is range-checked against the column dtype, which is what the
    tier-1 bridge's `struct.pack` does.

Quality
- CPU reference for every kernel; 630 XCTest cases in release including a scenario matrix over every type, null density, size
  and sliced input; concurrency and pool tests; CI on hosted Apple silicon in debug and release.
- `python/tests/test_functions.py` executes the Arrow-name registry: every runnable row is called through
  `call_function` and compared to `pyarrow.compute`, with a second input in a different Arrow type family
  for the rows whose claim spans several, and float tolerances recorded per row in `functions.TOLERANCE`.
  It found that `binary_length`, `binary_repeat` and `binary_reverse` refuse a `binary` column.
- Differential matrix against `pyarrow.compute` (docs/EVALUATION.md): 39,069 cases per run over 45 column types, 0 unclassified
  divergences. Fixed from its findings: stable null order in argsort and top_k, the -0.0 tie in the sort
  keys (top_k included, which had its own key mapping), NaN kept at the end of a descending sort, float32
  sums in software double, unsigned group-by sums, exact float32 comparison, float32 `sign`/`ceil`/`floor`/
  `trunc`/`round` and element-wise `min`/`max` on subnormals and signed zeros.
- Benchmarks: Swift vs all-core CPU vs Accelerate; Polars/pyarrow/pandas; ArrowMetal from Python in-process; latency mode.
- Adversarial review pass before release (four independent reviewers over the integrations, the engine and
  expression compiler, the GPU kernels, and the C ABI and Parquet reader): every finding carries a
  regression test; the fixes are the "Fixed" bullets above and the entries in docs/EVALUATION.md.
