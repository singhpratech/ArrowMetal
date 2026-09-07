# Roadmap

What comes after 0.1.0, in the order it is likely to happen. Everything here is open to a contributor;
the measured state behind each item is in [LOSSES.md](LOSSES.md) and the benchmark matrix.

## Kernels

- **Grouped moments at a few groups.** Variance and stddev at a thousand groups are 0.64x of pyarrow at
  50M rows because Metal has no 64-bit atomics and the software-binary64 accumulators cannot be
  privatised per threadgroup. A simdgroup-per-group accumulator is costed at about 6 ms of the 15.
- **The float64 sort's last gather.** Sorting values by inverting the sort key instead of gathering
  through the permutation; in progress.
- **A chunked `MetalArray`.** `shift(view=True)` already returns a two-chunk view; making chunked arrays
  a first-class input to every kernel turns slices, concatenations and shifts into pointers.
- **A GPU regular-expression engine.** A compiled automaton kernel behind the existing pre-filter, so a
  real regex stops being the one string operation that runs on the host.
- **Faster correct doubles.** `ln`, `sin` and `days_between` are correct to 2 ulp and only 1.04x to 1.99x
  over the fastest CPU library in the 2026-09-07 matrix; a better range reduction or polynomial is a
  direct path to 3x.
- **Float64 sum and mean with very few groups** take the one-threadgroup-per-group segmented path
  (35 ms for 10M rows in one group against 4.6 ms at a thousand); a work-stealing split is the fix.

## Engine

- **Plan-level batching** so a whole query from an engine pays one dispatch below a million rows.
- **Sort-free `unique` order** where the caller does not need it sorted.
- **A JSON plan test suite** shared with the integrations, so a Polars or DuckDB plan can be replayed
  against the engine without either library present.

## Integrations

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

- **Python** ships. **Swift** ships.
- **Rust:** a crate over `include/arrowmetal.h`; the Polars plugin already uses the ABI from Rust.
- **R:** a package over the same header, exchanging columns with the `arrow` R package through the
  C Data Interface, so nothing is copied on the way in.
- **TypeScript / JavaScript:** a Node binding with N-API, exchanging columns with Apache Arrow JS through
  the C Data Interface, for the tooling around analytics and ML that is written in TypeScript. The
  browser is out of scope: Metal is not there.
- **Go, Java, C#, Julia** through each language's FFI: the header needs nothing language-specific.
- Per-language timings of one operation from each binding, measured, on the site.

## Release mechanics

- The macOS arm64 wheel on PyPI, and the Polars plugin through maturin.
- The Swift package tagged, the crate on crates.io, a GitHub organisation.
- The site at arrowmetal.org, built from this repository.

## Non-goals

- Anything but Apple silicon: NVIDIA has cuDF, and a discrete GPU makes the zero-copy argument moot.
- A dataframe API of its own: ArrowMetal is the engine under Polars, DuckDB and pandas, not a fourth
  frontend.
- Approximate answers where Arrow specifies exact ones. The software-binary64 paths stay correct.
