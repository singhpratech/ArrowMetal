# ArrowMetal for Rust

Apache Arrow compute on Apple silicon GPUs, for [arrow-rs](https://docs.rs/arrow).

Two crates in one workspace:

| Crate | What it is |
|---|---|
| [`arrowmetal-sys`](arrowmetal-sys) | Raw `extern "C"` declarations over `include/arrowmetal.h`, plus the `build.rs` that finds and links `libArrowMetalC.dylib`. |
| [`arrowmetal`](arrowmetal) | The safe crate. `arrow::array::ArrayRef` in, `ArrayRef` out, every failure a `Result` carrying `am_last_error()`'s message. |

User documentation — installing, the example, what is and is not wrapped — is in
[`docs/RUST.md`](../docs/RUST.md). This file is the build and the measurement.

## Building and testing

The crates link a dylib that is not in the repository, so it has to exist first:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift build -c release --product ArrowMetalC     # from the repository root

cd rust
cargo test --release
```

`build.rs` looks for `libArrowMetalC.dylib` in `$ARROWMETAL_LIB` (the full path to the dylib), then
`$ARROWMETAL_LIB_DIR`, then `<repo>/.build/release`, `<repo>/.build/debug`, `/usr/local/lib` and
`/opt/homebrew/lib`, and fails with the list it searched otherwise. From outside the repository:

```sh
ARROWMETAL_LIB=/path/to/libArrowMetalC.dylib cargo test --release
```

`cargo test --release` runs 44 tests, plus 4 doc-tests that are **`no_run`: they are compiled and
type-checked, not executed** (they would need a GPU inside a doctest binary). Release matters: this
project has hit one release-only miscompile on the Swift side, and the sweeps run at 1,000,001
elements, which is slow to build and run unoptimised.

| File | Tests | Oracle |
|---|---|---|
| `arrowmetal/tests/compute.rs` | 29 | `arrow::compute` on the same array — `sum`/`min`/`max`, the six comparisons, `filter`, `sort`, `take`, `slice`, `cast`. `group_by` has no counterpart in the `arrow` crate (hash aggregation lives in DataFusion), so its oracle is a plain `HashMap` fold. |
| `arrowmetal/tests/plan.rs` | 7 | the same plans assembled by hand from arrow-rs kernels |
| `arrowmetal/tests/copy_rule.rs` | 4 | measured pointer alignments, not assumptions — the pointers `am_import` actually receives, read back out of an `arrow::ffi` export |
| `arrowmetal/tests/signatures.rs` | 2 | `include/arrowmetal.h`, re-parsed at test time |
| `arrowmetal-sys/src/lib.rs` | 2 | the C Data Interface's normative struct sizes; one live `am_version()` call |

Every kernel sweep runs lengths 0, 1, 33, 1024, 1025, 100,001 and **1,000,001** — an odd length past
a million that crosses a threadgroup boundary and leaves a partial tail — with and without nulls,
and the selection kernels also run on a producer-sliced array (`offset != 0`). No array in the suite
exceeds 10M elements.

`cargo test --test copy_rule -- --nocapture` prints the alignment table the copy rule is written
from. The short version: arrow-rs values buffers are page aligned from a few thousand rows up, but a
**validity bitmap** is eight times smaller and only gets there around 131,072 rows — so a nullable
column below that normally has its bitmap copied (a kilobyte or so) while its values are wrapped.
Full table and caveats in [`docs/RUST.md`](../docs/RUST.md#the-copy-rule).

## The measured timing

One run, on the machine below, on 2026-09-07. Reproduce it with:

```sh
ARROWMETAL_LIB=/path/to/libArrowMetalC.dylib cargo run --release --example bench
```

**Machine and method.** Apple M4 Max, macOS 26.6.2, `rustc 1.95.0`, arrow-rs 59.3.0, ArrowMetal 0.1.0.
One 10,000,000-element `Int64Array` of pseudo-random values in `[-1_000_000, 1_000_000)`, no nulls,
built once and shared by every row. The filter predicate is `x > 0`; 5,000,125 of the 10,000,000 rows
survive (50.0%). `std::time::Instant` around the call, wall time, single-threaded, nothing
subtracted, `std::hint::black_box` on every input and result. Three untimed warm-up iterations, then
five timed ones; the table is the **best of the five**. Both libraries' answers are asserted equal
before anything is timed. Outside a batch every ArrowMetal call commits its command buffer and waits,
so a "kernel" number is a complete GPU round trip, not an enqueue. The source is
[`arrowmetal/examples/bench.rs`](arrowmetal/examples/bench.rs).

| Operation, 10M Int64 | arrow-rs | ArrowMetal, kernel | ArrowMetal, end to end |
|---|---|---|---|
| `sum` | 0.91 ms | **0.28 ms** | 2.11 ms |
| `filter`, mask already built | 3.55 ms | **0.64 ms** | — |
| compare + `filter` | 4.58 ms | — | **2.97 ms** |

* **kernel** — the GPU call on an array already imported, mask already on the GPU. This is what each
  step of a longer chain costs.
* **end to end** — what *one* operation on an arrow-rs array costs, import included. What that covers
  differs by row: the `sum` row is import + the reduction and has **no export**, because a reduction
  returns a scalar through out-parameters rather than an array; the compare + `filter` row is
  import + compare + filter + `to_arrow`.

Supporting numbers, best of five: import 1.110 ms, export 0.000 ms (0.001 ms median). Medians for the
table above: 0.91 / 0.30 / 2.14 for `sum`, 3.59 / 0.66 for `filter`, 4.62 / 2.97 for compare +
`filter` — within a few percent of the bests, so the run was not noisy.

### Where ArrowMetal loses

**A single `sum` on an arrow-rs array is 2.3× slower than arrow-rs**: 2.11 ms against 0.91 ms. The
kernel itself is 3.3× *faster* (0.28 ms); the loss is entirely the cost of handing 80 MB to Metal —
1.11 ms for the import measured on its own, and the remaining ~0.7 ms in handle setup and the GPU's
first touch of the newly mapped pages.

That import is copy-free at this size: the values buffer came back aligned to 4 MiB, well past the
16 KiB page `makeBuffer(bytesNoCopy:)` needs. So the 1.1 ms is Metal mapping pages into the GPU's
address space, not a `memcpy`. It is still 1.1 ms, and one cheap kernel does not earn it back.

The break-even is roughly "more than one pass over the data". `compare + filter` is two passes and
ArrowMetal is already 1.5× faster end to end (2.97 ms against 4.58 ms), and every further operation on
the same imported handle costs the kernel column, not the end-to-end column.

So: import once, chain, export once. Wrapping a single reduction is a loss.

## Layout

```
rust/
  Cargo.toml              workspace
  arrowmetal-sys/
    build.rs              finds and links libArrowMetalC.dylib
    src/lib.rs            extern "C" declarations, the two C Data Interface structs
  arrowmetal/
    build.rs              repeats the -rpath for this crate's tests and examples
    src/lib.rs            the safe crate
    examples/bench.rs     the timing above
    examples/quickstart.rs  the example in docs/RUST.md, kept compilable
    tests/                compute.rs, plan.rs, copy_rule.rs, signatures.rs
```

`polars-plugin/arrowmetal-sys` is a separate, older `-sys` crate that the Polars expression plugin
depends on; it is untouched by this workspace and keeps building on its own.

> **One crate cannot depend on both `-sys` crates.** They both declare `links = "ArrowMetalC"`, and
> Cargo refuses a dependency graph containing two packages that link the same native library
> (*"multiple packages link to native library `ArrowMetalC`"*). That is Cargo working as intended —
> the `links` key exists to make exactly that collision an error. Nothing in this repository hits it,
> because the Polars plugin and this workspace are built separately and never appear in one graph.
> If you are writing a crate that wants both the Polars plugin and this binding, depend on
> `rust/arrowmetal-sys` only and reach the plugin through Polars at run time.
