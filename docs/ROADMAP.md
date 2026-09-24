# Roadmap

What comes after 0.1.0, in the order it is likely to happen. Everything here is open to a contributor;
the measured state behind each item is in [TO_IMPROVE.md](TO_IMPROVE.md) and the benchmark matrix.

This is the only roadmap: the older `../ROADMAP.md` now points here.

## Kernels

- **Delta Lake and Iceberg read speed.** The table readers are correct but slow: on the 2026-09-24 run
  every CPU reader measured is ahead on every read, by the widest margin on a full Delta read
  (`Benchmarks/results/lakehouse_2026-09-24.csv`, [LAKEHOUSE.md](LAKEHOUSE.md)). The metadata walk and
  the per-file Parquet reads are the places to look first.
- **IPC view layouts and nested Parquet reads.** pyarrow is ahead on every IPC view layout
  (`Benchmarks/results/ipc_views_2026-09-24.csv`), and plain lists, small lists of structs and the
  all-columns read are behind the fastest CPU reader
  (`Benchmarks/results/parquet_nested_2026-09-24.csv`, [PARQUET.md](PARQUET.md)).
- **The router for float columns and array-to-array compares.** The router's table covers integer
  columns; float columns have no measured crossover yet and stay on the GPU. The compare row is fitted on
  the scalar compare, and an array-to-array compare near 3M rows is the one case in the router check where
  `auto` picks the slower path (`Benchmarks/results/router_check_2026-09-24_after_refit.csv`).

- **Grouped moments at a few groups.** Variance at a thousand groups is 0.96x of pyarrow at 50M rows
  (0.78x of it in the eager baseline), and stddev 1.24x of the fastest parallel idiom, because Metal
  has no 64-bit atomics and the software-binary64 accumulators cannot be privatised per threadgroup.
  A simdgroup-per-group accumulator is costed at about 6 ms of the 30.
- **A chunked `MetalArray`.** `shift(view=True)` already returns a two-chunk view; making chunked arrays
  a first-class input to every kernel turns slices, concatenations and shifts into pointers.
- **A GPU regular-expression engine.** A compiled automaton kernel behind the existing pre-filter, so a
  real regex stops being the one string operation that runs on the host.
- **Faster correct doubles.** `ln` is measured at 1 ulp and `sin` at 2, and they run 0.10x to 0.17x of
  the fastest parallel CPU idiom (1.04x-1.99x of the eager one), `days_between` included; a better
  range reduction or polynomial is a direct path to 3x.
- **Float64 sum and mean with very few groups** take the one-threadgroup-per-group segmented path
  (35 ms for 10M rows in one group against 4.6 ms at a thousand); a work-stealing split is the fix.

## Engine

- **Plan-level batching** so a whole query from an engine pays one dispatch below a million rows.
- **Sort-free `unique` order** where the caller does not need it sorted.
- **A JSON plan test suite** shared with the integrations, so a Polars or DuckDB plan can be replayed
  against the engine without either library present.

## Types and interop

The gaps a caller meets at the border. Each one is a type or an interface ArrowMetal does not answer
to yet; the by-family status of every one of them is in [COVERAGE.md](COVERAGE.md).

- **`utf8_view` / `binary_view` through the C Data Interface.** The IPC reader takes the view types on
  `main` (materialised to the classic layouts), but the C Data importer and exporter still do not read
  the `vu` / `vz` formats, and the list views import and re-export as a plain list.
- **decimal256 beyond selection.** Import, export, the comparisons, `filter` / `take` / `slice` and
  `sum` are there; `min` / `max`, arithmetic, the rounding family and the casts throw rather than
  compute something wrong (`Sources/ArrowMetal/Decimal.swift`).
- **Compute over union values**, which a type-id-dispatched layout makes awkward for the uniform-thread
  model, and **aggregates over list values**. Both are import, export and selection only today.
- **Arrow IPC *writing* of compressed bodies.** The reader takes LZ4_FRAME and ZSTD, per buffer; the
  writer emits uncompressed bodies only, so a file this package writes is larger than the one pyarrow
  writes for the same batch. Reading is no longer the narrow side: every type the engine holds is read
  as well as written, dictionaries follow message order, and deltas and stream replacements are taken.
- **The C Device Stream** (`ArrowDeviceArrayStream`, declared in `arrow_abi.h` and referenced nowhere
  else) and **`__arrow_c_device_array__`** in the Python package, which exports only
  `__arrow_c_array__`. Plain C Stream import and export both work (`am_stream_from_c_stream`,
  `am_stream_export_c`).
- **The cast targets still refused:** utf8 → temporal (that is `strptime`, which exists as its own
  function but is not wired up as a cast), and dictionary, union, run-end and interval targets.
- **The last two Arrow options:** a per-row `num_repeats` on `binary_repeat`, and dictionary or nested
  **key columns** for `sort_indices` / `lexsort_indices` (utf8 and binary keys sort today, through
  `Kernels/StringSort.swift`).
- **The low-level `am_join`** index entry point is inner and left over int32 / int64 keys only. The
  engine's join already covers inner/left/right/full/semi/anti over arbitrary key types
  (`Kernels/JoinExtra.swift`); the C index form has not caught up.
- **Metal 4** command encoding and residency sets for very large columns.

## Integrations

- **Plan errors carry the right kind.** `am_plan_run` reports an engine type-check failure (an unknown column, say) as `Invalid ArrowArray: ...`; the message is right and the prefix is wrong. One enum case in the C shim.
- **`am_import_ex` with a copy report.** `am_import` discards the Swift side's `ImportResult.zeroCopy`, so a binding cannot tell its caller whether the import wrapped or copied; the Rust and Node bindings infer it from pointer alignment. One extra out-parameter on a new entry point, kept ABI-compatible.
- **arrow-swift and MLX:** `ArrowMetalSwiftArrow`, conversion to and from `apache/arrow-swift` arrays,
  and `ArrowMetalMLX`, a bridge to `MLXArray` for feeding columns into models. Neither target exists
  yet; both would go through the Arrow C interfaces rather than becoming dependencies
  ([DECISIONS.md](DECISIONS.md)).
- **Polars:** `lf.collect(engine=am.MetalEngine())` runs the parts of an optimised plan it can take on
  `main` ([POLARS.md](POLARS.md), tier 4). Still open: plans that start from a file scan, and a named
  `engine="metal"`, which needs a change in Polars itself.
- **DuckDB:** an optimizer extension on `main` runs eligible aggregates of unchanged SQL on the GPU
  ([DUCKDB.md](DUCKDB.md) §4b). Still open: shapes beyond those aggregates, and a signed build, since an
  unsigned C++ extension is tied to one DuckDB release and must be loaded with
  `allow_unsigned_extensions`.
- **pandas:** a stable extension-array hook for compute kernels, so the accelerator stops patching.
- **Apache Arrow:** a round-trip test between ArrowMetal and nanoarrow's experimental Metal device implementation (memory only today), so the two agree on `ARROW_DEVICE_METAL`.
- **DataFusion, Ibis, Lance, Hugging Face datasets, MLX:** one operator, one backend, one scan, one
  map step, one zero-copy handoff, respectively.

## Languages

In the order they matter for analytics and ML work on a Mac:

- **Python** and **Swift** are in 0.1.0. **Rust** (`rust/arrowmetal` over `include/arrowmetal.h`, which the
  Polars plugin already uses), **R** (`r/arrowmetal`, exchanging columns with the `arrow` R package
  through the C Data Interface), **TypeScript / JavaScript** (`node/`, an N-API addon over Apache
  Arrow JS) and **Go** (`go/arrowmetal`, cgo over the same header) are in 0.1.0 too, each with its own test
  suite ([RUST.md](RUST.md), [R.md](R.md), [TYPESCRIPT.md](TYPESCRIPT.md), [GO.md](GO.md)). The browser
  is out of scope: Metal is not there.
- **Java, C#, Julia** through each language's FFI: the header needs nothing language-specific.
- Per-language timings of one operation from each binding, measured, on the site (the per-binding
  tables already exist in `docs/`).

## Release mechanics

- The macOS arm64 wheel on PyPI, and the Polars plugin through maturin. `scripts/build_wheel.sh`
  already builds the wheel with `libArrowMetalC.dylib` inside it and a `macosx_*_arm64` tag; what is
  left is publishing.
- The Swift package tagged, the crate on crates.io, a GitHub organisation.
- The site at arrowmetal.org, built from this repository.
- A DocC documentation site. There is no `.docc` catalogue today; the documentation is the hand-written
  Markdown in `docs/`.
- Benchmarks on M1 / M2 / M3 and on iPhone and iPad, with a results table per chip. Every number
  published so far is from one M4 Max.

## Non-goals

- Anything but Apple silicon: NVIDIA has cuDF, and a discrete GPU makes the zero-copy argument moot.
- A dataframe API of its own: ArrowMetal is the engine under Polars, DuckDB and pandas, not a fourth
  frontend.
- Approximate answers where Arrow specifies exact ones. The software-binary64 paths stay correct.
