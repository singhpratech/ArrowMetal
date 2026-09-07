# Roadmap

What comes after 0.1.0, in the order it is likely to happen. Everything here is open to a contributor;
the measured state behind each item is in [LOSSES.md](LOSSES.md) and the benchmark matrix.

This is the only roadmap: the older `../ROADMAP.md` now points here.

## Kernels

- **Grouped moments at a few groups.** Variance at a thousand groups is 0.96x of pyarrow at 50M rows
  (0.78x of it in the eager baseline), and stddev 1.24x of the fastest parallel idiom, because Metal
  has no 64-bit atomics and the software-binary64 accumulators cannot be privatised per threadgroup.
  A simdgroup-per-group accumulator is costed at about 6 ms of the 30.
- **The float64 sort's last gather.** Sorting values by inverting the sort key instead of gathering
  through the permutation; in progress.
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

- **`utf8_view` / `binary_view`** import and export. Nothing in `Sources/` reads the `vu` / `vz`
  formats today; the list views import and re-export as a plain list.
- **decimal256 beyond selection.** Import, export, the comparisons, `filter` / `take` / `slice` and
  `sum` are there; `min` / `max`, arithmetic, the rounding family and the casts throw rather than
  compute something wrong (`Sources/ArrowMetal/Decimal.swift`).
- **Compute over union values**, which a type-id-dispatched layout makes awkward for the uniform-thread
  model, and **aggregates over list values**. Both are import, export and selection only today.
- **Arrow IPC for nested and decimal columns**, and compressed bodies — the reader and writer refuse
  all three by name rather than mis-decoding them.
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
- **Polars:** an `engine="metal"` behind `collect()` that hands the optimised plan to the C ABI, the way
  the GPU engine hands it to cuDF. Needs Polars.
- **DuckDB:** an optimizer rule that pushes scan, filter and aggregate down into the extension. Needs
  DuckDB.
- **pandas:** a stable extension-array hook for compute kernels, so the accelerator stops patching.
- **Apache Arrow:** a round-trip test between ArrowMetal and nanoarrow's experimental Metal device implementation (memory only today), so the two agree on `ARROW_DEVICE_METAL`.
- **DataFusion, Ibis, Lance, Hugging Face datasets, MLX:** one operator, one backend, one scan, one
  map step, one zero-copy handoff, respectively.

## Languages

In the order they matter for analytics and ML work on a Mac:

- **Python** ships. **Swift** ships. **Rust** (`rust/arrowmetal` over `include/arrowmetal.h`, which the
  Polars plugin already uses), **R** (`r/arrowmetal`, exchanging columns with the `arrow` R package
  through the C Data Interface), **TypeScript / JavaScript** (`node/`, an N-API addon over Apache
  Arrow JS) and **Go** (`go/arrowmetal`, cgo over the same header) ship too, each with its own test
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
