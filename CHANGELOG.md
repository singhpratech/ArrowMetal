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
- `topK`: per-threadgroup selection for k <= 1024 (threshold plus a bitonic compaction in threadgroup memory,
  then one radix sort of the candidates), matching the full sort index for index; the sort path above that.

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
- `docs/ARROW_FUNCTIONS.md`, generated from that registry, is the by-name coverage page: 231 gpu, 20 cpu,
  55 partial, 1 missing (`binary_slice`), 0 pending.

Bindings
- libArrowMetalC C ABI (include/arrowmetal.h) and python/arrowmetal ctypes package (Arrow PyCapsule protocol).

Fixed
- `GroupByKeys._agg` in the Python package took the device handle of a temporary `MetalArray` that was
  released before the C call read it, segfaulting every grouped aggregate whose values arrived as a
  pyarrow array rather than a `MetalArray`.

Quality
- CPU reference for every kernel; 26 tests including a scenario matrix over every type, null density, size
  and sliced input; concurrency and pool tests; CI on hosted Apple silicon in debug and release.
- `python/tests/test_functions.py` executes the Arrow-name registry: every runnable row is called through
  `call_function` and compared to `pyarrow.compute`, with a second input in a different Arrow type family
  for the rows whose claim spans several, and float tolerances recorded per row in `functions.TOLERANCE`.
  It found that `binary_length`, `binary_repeat` and `binary_reverse` refuse a `binary` column.
- Differential matrix against `pyarrow.compute` (docs/EVALUATION.md): 13,176 cases per run, 0 unclassified
  divergences. Fixed from its findings: stable null order in argsort and top_k, the -0.0 tie in the sort
  keys (top_k included, which had its own key mapping), NaN kept at the end of a descending sort, float32
  sums in software double, unsigned group-by sums, exact float32 comparison, float32 `sign`/`ceil`/`floor`/
  `trunc`/`round` and element-wise `min`/`max` on subnormals and signed zeros.
- Benchmarks: Swift vs all-core CPU vs Accelerate; Polars/pyarrow/pandas; ArrowMetal from Python in-process; latency mode.
