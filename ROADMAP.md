# Roadmap

Ordered roughly by impact divided by effort. Each item is a self-contained contribution. Open an issue to claim one.

Where each item sits against the full Apache Arrow compute and type lists is tracked by name in
[docs/ARROW_FUNCTIONS.md](docs/ARROW_FUNCTIONS.md) and by family in [docs/COVERAGE.md](docs/COVERAGE.md).

## Delivered in 0.1.0

Everything below is on `main`, reachable from Swift, the C ABI and Python, and checked against
`pyarrow.compute` in `python/tests/test_functions.py`. It is listed rather than deleted so a reader can
see what the earlier roadmap promised and where it landed.

- **Take / gather**, **count / any / all** on the GPU, **cast** between primitive types, **slicing with
  offsets** without materialising, and Arrow's **NaN semantics** for float min/max.
- **Float64 on the GPU** throughout: compare/min/max/filter/take on order-preserving bit patterns, and
  `sum`, the four arithmetic operations, the cumulative scans and the whole transcendental family through
  a correctly rounded software binary64 implementation. `exp` / `ln` / `log10` / `log2` / `sqrt` / `power`
  remain float32-evaluated and widened (about 1e-7 relative) — Metal has no `double` transcendentals.
- **Checked arithmetic**: `add_checked` … `logb_checked`, the checked shifts, the checked cumulative and
  pairwise forms — the unchecked kernel plus one read-only check pass in the same command buffer, so a
  checked op costs one GPU round trip and raises naming the Arrow message and the first offending row.
- **Group-by over arbitrary keys**: `GroupByKeys` maps any key column — sparse or negative integers,
  floats, booleans, temporal values, `utf8`, `binary`, dictionary, decimal, several columns folded
  together — to dense ids on the GPU, and all 24 Arrow `hash_*` aggregates run against them. The
  dense-integer `GroupBy` fast path is kept underneath.
- **Sort / argsort / top-k**, plus multi-column `lexsort_indices`, `partition_nth_indices`, `rank`,
  `dense_rank`, `row_number`, `rank_quantile`, `rank_normal` and `winsorize`.
- **The full Unicode string surface**: the `utf8_is_*` predicates, case and title mapping, centring,
  padding, slicing, replace-slice, trimming against the Unicode whitespace class, normalisation, the regex
  family, SQL `LIKE`, splitting, joining, and set lookup over strings and binary. GPU wherever bytes are
  enough; host only where a Unicode table or ICU is.
- **Temporal**: every extractor including `week` with all its options, `iso_calendar`, `year_month_day`,
  `subsecond`, `us_week` / `us_year`; every `*_between` difference; the three interval layouts and their
  interval-valued differences; `add_interval`; and the timezone functions `assume_timezone`,
  `local_timestamp` and `is_dst` (host-side — the IANA database is host data).
- **The rest of the type matrix**: `null`, `float16`, `decimal32` / `decimal64` (widening to decimal128 on
  the GPU and narrowing back), `fixed_size_binary`, `list_view` / `large_list_view` import, `map` with
  `map_lookup`, `run_end_encoded`, and extension types.
- **Nested and conditional compute**: `list_slice`, `list_parent_indices`, `case_when`, `choose`,
  `replace_with_mask`, `fill_null_forward` / `_backward`, `indices_nonzero`, `is_nan` / `is_finite` /
  `is_inf`, `pivot_wider` and `hash_pivot_wider`.
- **Statistical aggregates**: `variance`, `stddev`, `quantile`, `mode`, `count_distinct`, `first` / `last`,
  `index`, `skew`, `kurtosis` and `tdigest`, with the grouped forms of all of them.
- **Windows**: `pairwise_diff`, the cumulative family, `shift`, and rolling `sum` / `mean` / `min` / `max`.
- **RecordBatch** with C Data, C Device and C Stream interop, **batched and async execution**
  (`batch { }`, `batchAsync`), **Arrow IPC** read and write including `DictionaryBatch`, and a **GPU hash
  join** over int32 / int64 keys.
- **Python package** over the C ABI speaking `__arrow_c_array__`.

## Near term

- [ ] **`binary_slice`** — the one Arrow compute name with no implementation. The slicing kernel counts
      code points and needs a byte-offset variant; `binary_replace_slice` already indexes in bytes, so the
      machinery is there.
- [ ] **`binary` input to the string kernels.** `binary_length`, `binary_repeat` and `binary_reverse` ask
      for `utf8` and refuse a `binary` column, where Arrow accepts both. The layout is identical; only the
      importer check and the char-length kernel are utf8-specific.
- [x] **Overflow-erroring casts** (`safe=true`), casts between decimal precisions, and casts between
      nested types — `Sources/ArrowMetal/CastOptions.swift` and `CastDispatch.swift`. `safe=true` is one
      read-only GPU pass that converts each value back and raises on the first row that loses something;
      `cast(to: format, options:)` is one entry point for numeric, bool, utf8, temporal, decimal, list
      and struct targets. Still open under it: utf8 → temporal (that is `strptime`) and dictionary,
      union, run-end and interval targets.
- [x] **The remaining Arrow options** — `null_placement` on the sorts and the rank family, all four
      `rank` tiebreakers, `null_matching_behavior` on `is_in` / `index_in`, the distinct-value order on
      `unique` / `value_counts` / `dictionary_encode`, the whole of `CastOptions` and the whole of
      `RoundTemporalOptions`. `python/tests/test_options.py` walks the cross product against
      `pyarrow.compute`. Still open: `max_splits` and `reverse` on the splits, N-column
      `binary_join_element_wise`, per-row `num_repeats` on `binary_repeat`, and utf8 / binary /
      dictionary **key columns** for `sort_indices` and `lexsort_indices`.
- [ ] Benchmarks on M1/M2/M3 and on iPhone/iPad; a results table per chip.

## Medium term

- [ ] **`utf8_view` / `binary_view`** import and export.
- [ ] **decimal256 arithmetic.** Import, export, comparisons, `filter` / `take` / `slice` and `sum` are
      there; `min` / `max`, arithmetic, the rounding family and the casts throw rather than compute
      something wrong.
- [ ] **Compute over union values**, which a type-id-dispatched layout makes awkward for the uniform-thread
      model, and **aggregates over list values**.
- [ ] **Arrow IPC for nested and decimal columns**, and compressed bodies.
- [ ] **C Stream export and C Device Stream**, the two interop rows still open.
- [ ] **Right and full outer joins**, and join keys beyond int32 / int64.
- [ ] **Metal 4** command-encoding path and residency sets for very large columns.
- [ ] **Pipelining for small columns.** Under about a million rows the fixed cost of a dispatch dominates;
      see [docs/DESIGN.md](docs/DESIGN.md).

## Integrations

- [ ] `ArrowMetalSwiftArrow`: convenience conversion to and from `apache/arrow-swift` arrays.
- [ ] `ArrowMetalMLX`: zero-copy bridge to `MLXArray` for feeding columns into models.
- [ ] Wheel packaging with the dylib inside, and `__arrow_c_device_array__` in the Python package.
- [ ] A DuckDB or DataFusion user-defined function that offloads a scan+filter+aggregate to ArrowMetal.

## Project

- [ ] GitHub Actions on `macos-15` runners (Metal works on the hosted Apple silicon runners).
- [ ] DocC documentation site.
