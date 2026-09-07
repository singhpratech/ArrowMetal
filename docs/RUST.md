# ArrowMetal for Rust users

[arrow-rs](https://docs.rs/arrow) is where Rust keeps Arrow-shaped data. This document is how you
point that data at the GPU.

Two crates live in [`rust/`](../rust):

| Crate | What it is |
|---|---|
| `arrowmetal-sys` | Raw `extern "C"` declarations over [`include/arrowmetal.h`](../include/arrowmetal.h), plus the `build.rs` that finds and links `libArrowMetalC.dylib`. |
| `arrowmetal` | The safe crate. An `arrow::array::ArrayRef` goes in, an `ArrayRef` comes out; every failure is a `Result` carrying `am_last_error()`'s message. |

Neither is published to crates.io yet (`publish = false`), so both are used by path or by git.

Everything below was run in this repository on 2026-09-07 on an Apple M4 Max, macOS 26.6.2,
`rustc 1.95.0`, arrow-rs 59.3.0, ArrowMetal 0.1.0. **No number on this page was not measured in that
session.**

---

## Install

ArrowMetal is a Metal library. It needs an **Apple silicon Mac**; there is no other target.

```bash
# 1. Build the GPU library. This is the only non-cargo step.
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
    swift build -c release --product ArrowMetalC
```

That produces `.build/release/libArrowMetalC.dylib`. Then, in your own crate:

```toml
# Cargo.toml
[dependencies]
arrow = "59"
arrowmetal = { path = "../ArrowMetal/rust/arrowmetal" }
# or, once you have a checkout somewhere fixed:
# arrowmetal = { git = "https://github.com/singhpratech/ArrowMetal", branch = "main" }
```

and build with the dylib's location known:

```bash
ARROWMETAL_LIB=/path/to/ArrowMetal/.build/release/libArrowMetalC.dylib cargo build --release
```

### Finding the dylib

`arrowmetal-sys/build.rs` searches, first hit wins:

1. `$ARROWMETAL_LIB` — the full path to the dylib. This is the one to set.
2. `$ARROWMETAL_LIB_DIR` — a directory holding it.
3. `<repo>/.build/release`, then `<repo>/.build/debug` — a SwiftPM build in the ArrowMetal checkout.
4. `/usr/local/lib`, `/opt/homebrew/lib`.

Anything else is a hard build error naming every path it looked in and the `swift build` line, rather
than a link that succeeds and a `dyld` failure at run time.

The dylib's install name is `@rpath/libArrowMetalC.dylib`, so the directory that was found is baked
into your binary as an `LC_RPATH` entry. **You do not need `DYLD_LIBRARY_PATH` at run time.** Because
Cargo hands a build script's link arguments only to the crate that owns it, `arrowmetal-sys`
republishes that directory through its `links = "ArrowMetalC"` key (`DEP_ARROWMETALC_LIB_DIR`) and
`arrowmetal/build.rs` repeats the `-rpath` for its own binaries, examples and tests. A crate that
depends on `arrowmetal` and builds a binary should do the same:

```rust
// your-crate/build.rs
fn main() {
    let dir = std::env::var("DEP_ARROWMETALC_LIB_DIR").unwrap();
    println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
}
```

If you move the dylib after building, set `DYLD_LIBRARY_PATH` or rebuild.

---

## Example

This is [`rust/arrowmetal/examples/quickstart.rs`](../rust/arrowmetal/examples/quickstart.rs) verbatim,
so it is compiled by every `cargo build --examples`. Run it with
`cargo run --release --example quickstart`; it prints `2 groups: [0, 1] -> [10, 27]` and
`sum of everything: Some(Int64(42))`.

```rust
use arrow::array::Int64Array;
use arrowmetal::{group_by, Array, CompareOp};

fn main() -> Result<(), arrowmetal::Error> {
    let region = Int64Array::from(vec![0i64, 1, 0, 2, 1]);
    let amount = Int64Array::from(vec![Some(10i64), Some(20), None, Some(5), Some(7)]);

    let region = Array::from_arrow(&region)?;   // to the GPU
    let amount = Array::from_arrow(&amount)?;

    let big = amount.compare_scalar(CompareOp::Gt, 5i64)?;   // boolean mask
    let kept_amount = amount.filter(&big)?;                  // 10, 20, 7
    let kept_region = region.filter(&big)?;

    let gb = group_by(&[&kept_region])?;
    let totals = gb.sum(&kept_amount)?.to_arrow()?;          // back to arrow-rs
    let keys = gb.keys(0)?.to_arrow()?;

    let ints = |a: &dyn arrow::array::Array| -> Vec<i64> {
        a.as_any().downcast_ref::<Int64Array>().unwrap().values().to_vec()
    };
    println!("{} groups: {:?} -> {:?}", gb.group_count(), ints(&keys), ints(&totals));
    println!("sum of everything: {:?}", amount.sum()?);
    Ok(())
}
```

---

## The copy rule

**Copy-free out always; copy-free in when the producer's buffers are page aligned, one copy
otherwise.**

The decision is made **per buffer**, so a nullable column can have its values wrapped and its
validity bitmap copied. That case is common; see the table.

Out ([`Array::to_arrow`]) is always copy-free. ArrowMetal's buffers are `MTLBuffer`s in shared
memory; it hands their pointers straight to the C Data Interface with a release callback, and
arrow-rs reads the GPU's memory in place. Measured over inputs that all carry nulls, so both buffers
are exercised: the exported **values and validity** pointers were page aligned in every case
(`tests/copy_rule.rs::arrowmetal_exported_buffers_are_page_aligned`), and the export call itself took
0.001 ms for a 10M-row column.

In ([`Array::from_arrow`]) is copy-free only when the buffer pointer is aligned to a **16 KiB page**,
because that is what `MTLDevice.makeBuffer(bytesNoCopy:)` requires; otherwise that buffer is copied
once. Whether an arrow-rs array clears the bar is a property of the *allocator*, not of arrow-rs, so
it is measured rather than assumed. 32 allocations at each size, exported through `arrow::ffi` and
read back as the pointers `am_import` actually receives:

| Column length | Values bytes | Values page aligned | Validity bitmap page aligned |
|---|---|---|---|
| 16 | 128 B | 1/32 | 0/32 |
| 512 | 4 KiB | 8/32 | 0/32 |
| 8,192 | 64 KiB | 32/32 | 2/32 |
| 131,072 | 1 MiB | 32/32 | 32/32 |
| 1,250,000 | 10 MB | 32/32 | 32/32 |
| 10,000,000 | 80 MB | 32/32 | 32/32 |

So in practice:

* **Values are wrapped from a few thousand rows up**, and copied below that.
* **A nullable column of fewer than roughly 130,000 rows normally has its validity bitmap copied**,
  even when its values are wrapped. The bitmap is one bit per row, so the copy is about a kilobyte at
  8,192 rows and about 16 KB at the point it stops happening — small, but real.

Both are the system allocator serving a large enough request with `mmap` and handing back
page-aligned memory. The bitmap is eight times smaller than the values, so it crosses that threshold
eight times later. None of it is a guarantee — a custom global allocator, a different macOS, or a
buffer sliced out of an arena can all change it. Run `cargo test --test copy_rule -- --nocapture` in
`rust/` to print the table for your own machine.

Copy-free is not free: at 10M int64 rows the import still cost **1.11 ms**, which is Metal mapping
80 MB of pages into the GPU's address space. See the timing below.

Two more claims, **inferred rather than tested**:

* **A slice does not cost a copy either way.** An Arrow `offset` is carried on the handle rather than
  applied, so `array.slice(7, n)` should import the same buffers the unsliced array would.
* **An array that came out of ArrowMetal and goes back in is never copied in either buffer**,
  whatever the original alignment: ArrowMetal recognises its own exported buffers and shares the
  `MTLBuffer` objects directly.

Both follow from reading `Sources/ArrowMetal/CInterop.swift` plus the alignment measurement above,
and both have tests behind their *observable* halves — a sliced array round-trips exactly
(`round_trip_of_a_sliced_array`, `a_slice_survives_the_round_trip`), and an ArrowMetal export is
always page aligned, which is the precondition for the no-copy path
(`arrowmetal_exported_buffers_are_page_aligned`). But **whether a given import actually copied is not
observable through the C ABI**: the Swift side computes an `ImportResult.zeroCopy` flag and
`am_import` discards it. Until the ABI reports it — an `am_import_ex` with an `int* out_zero_copy`
would do — these two are arguments, not measurements, and are labelled as such.

---

## What is covered

Each row is backed by a test in `rust/arrowmetal/tests/` against arrow-rs's own kernel on the same
data, at lengths 0, 1, 33, 1024, 1025, 100,001 and 1,000,001, with and without nulls.

| Rust | ArrowMetal ABI | Oracle |
|---|---|---|
| `Array::from_arrow` / `to_arrow` | `am_import` / `am_export` | round trip equals the input, including sliced arrays |
| `len`, `null_count`, `format` | `am_length`, `am_null_count`, `am_format` | `arrow::array::Array` |
| `sum`, `min`, `max`, `mean` | `am_reduce` | `arrow::compute::{sum, min, max}`; `mean` against arrow's exact sum over the valid count |
| `compare_scalar`, `compare` (6 ops) | `am_compare_scalar`, `am_compare_array` | `arrow::compute::kernels::cmp` |
| `filter` | `am_filter` | `arrow::compute::filter`, including a mask with nulls |
| `take` | `am_take` | `arrow::compute::take`, with repeated, out-of-order and null indices |
| `slice` | `am_slice` | `arrow::array::Array::slice` |
| `sort`, `argsort` | `am_sort`, `am_argsort` | `arrow::compute::sort` with `nulls_first: false` |
| `cast` | `am_cast` | `arrow::compute::cast` |
| `group_by(keys)` with `sum`, `min`, `max`, `mean`, `count`, `count_all`; `keys(i)`, `ids()`, `agg_raw` | `am_group_by_keys`, `am_group_agg_ex` | a plain `HashMap` fold — arrow-rs's `arrow` crate has no hash aggregation (it lives in DataFusion) |
| `Source`, `run_plan`, `explain_plan`, `PlanResult` | `am_plan_source_create`, `am_plan_run`, `am_plan_explain`, `am_plan_column*` | the same plan assembled by hand from arrow-rs kernels |
| `batch(\|\| …)` | `am_batch_begin` / `am_batch_end` | the same chain unbatched, and arrow's answer; plus deferred-failure, nesting and panic-unwind cases |

Element types: **Int64 and Float64** are swept against arrow-rs. Boolean arrays are exercised as
comparison output and filter masks, and Int32 as sort indices. Every other Arrow type the ABI
supports reaches the GPU through the same entry points but has **no test in this crate** — the Swift
and Python suites cover them ([TESTING.md](TESTING.md)).

Errors are `Result<T, arrowmetal::Error>`, and the message is `am_last_error()`'s, read on the spot
because that slot is thread-local.

Handles (`Array`, `GroupBy`, `Source`, `PlanResult`) are `!Send` and `!Sync` on purpose: the ABI's
error slot and its command-buffer batching are both thread-local. Use one set of handles per thread.
That is asserted at compile time (`assert_not_impl_any!`), not just documented.

### Scalar operands, and the one type that is refused

The ABI reads `sizeof(element type)` bytes through a `void*`, so a scalar of the wrong width would be
an out-of-bounds read. Every scalar entry point therefore checks the Rust type against the array's
own `am_format` string first: `compare_scalar(Gt, 1i32)` on an `Int64Array` is an `Err`, not a read
off the end.

That check is only as good as `am_format`, and there is exactly one Arrow type for which `am_format`
does not describe what the kernels compute on. **A dictionary-encoded array reports its *index* type
(`"i"`) while every kernel decodes the dictionary and computes on the *value* type.** A 4-byte `i32`
scalar would be accepted for a `Dictionary(Int32, Float64)` column whose kernel then reads 8 bytes —
unsound from safe Rust, and wrong besides (`compare_scalar(Lt, 2i32)` returns `[false, false, false]`
where arrow-rs says `[true, false, false]`, and the *correct* `2.0f64` scalar is rejected).

**`Array::from_arrow` therefore refuses `DataType::Dictionary` outright**, with an error naming the
key and value types and telling you to decode first:

```rust
let decoded = arrow::compute::cast(&dict, &DataType::Float64)?;
let gpu = arrowmetal::Array::from_arrow(decoded.as_ref())?;   // now type-checks correctly
```

This is a limitation of the C ABI, not of the Arrow type — the kernels handle dictionaries correctly,
and Swift and Python callers use them. It is an ABI defect on ArrowMetal's side (`am_format` should
report the compute type) and is tracked to be fixed there; until it is, refusing the type is what
keeps the sentence above true for everything this crate accepts.
`tests/compute.rs::dictionary_arrays_are_refused_at_import` pins the rejection and the decode path.

### One divergence from arrow-rs, found and pinned

`min` / `max` on a float column containing NaN. The two libraries use different, each internally
consistent, rules — neither is a bug:

| Input (float64) | ArrowMetal | arrow-rs 59.3 | pyarrow |
|---|---|---|---|
| `[NaN, 2.0, null, -1.0, NaN]` — `min` | `-1.0` | `-1.0` | `-1.0` |
| the same — `max` | `2.0` | `NaN` | `2.0` |
| `[NaN, NaN]` — `min` and `max` | null | `NaN` | `NaN` |

**arrow-rs orders NaN at the top of a total order.** Both `min` and `max` say so in the same
sentence — *"For floating point arrays any NaN values are considered to be greater than any other
non-null value"* (`arrow_arith::aggregate`, 59.3.0) — so `min` returns the smallest non-NaN and
`max` returns NaN. That was reported as [apache/arrow-rs#101](https://github.com/apache/arrow-rs/issues/101)
and closed as intended behaviour in 2022.

**ArrowMetal skips NaN in both**, the way it skips a null, and reports "no valid value" when every
valid element is NaN. That follows Arrow C++ / pyarrow on the mixed case; on an all-NaN column
pyarrow returns NaN where ArrowMetal returns null, which the C header already documents.

`tests/compute.rs::nan_handling_diverges_from_arrow_rs_and_is_pinned` asserts the **ArrowMetal and
arrow-rs columns** and fails if either moves. The pyarrow column is not asserted from Rust — pyarrow
is not a dependency of this crate; it is pinned in the Python suite, by
`test_nan_is_skipped_by_min_and_max_in_both` and
`test_all_nan_min_max_is_null_in_arrowmetal_and_nan_in_pyarrow` in
`python/tests/test_differential.py`. With no NaN in the data the two agree exactly, which is what
every other reduction test relies on.

---

## What is not wrapped

`include/arrowmetal.h` has 220 entry points. `arrowmetal-sys` declares 35 of them — every one called
by the safe crate, none declared and unused — and the safe crate covers the list above. Everything
below is reachable from Swift, Python and the C ABI, and **not** from this crate. There is no
technical obstacle to any of it; it is unwrapped because it is untested here, and an untested wrapper
is not a shipped one.

| ABI area | Entry points |
|---|---|
| Strings | `am_str_unary`, `am_str_match`, `am_str_transform`, `am_str_concat`, `am_str_equals_array`, `am_str_dictionary_encode`, `am_string_predicate`, `am_string_transform`, `am_string_is_in`, `am_string_index_in`, `am_str_extra`, `am_to_strings`, `am_parse`, `am_byte_transform`, `am_binary_join`, `am_join_element_wise` |
| Regex, LIKE, splitting | `am_regex`, `am_split`, `am_extract_struct` |
| Temporal | `am_temporal_extract`, `am_temporal_cast_unit`, `am_temporal_math`, `am_temporal_extra`, `am_round_temporal_ex`, `am_assume_timezone`, `am_local_timestamp`, `am_utc_offset`, `am_to_timezone`, `am_add_interval`, `am_interval_between`, `am_interval_field` |
| Decimals | `am_decimal_op`, `am_decimal_widen`, `am_decimal_narrow` |
| Nested types (list, struct, map, union) | `am_list_value_length`, `am_list_flatten`, `am_list_element`, `am_list_slice`, `am_list_parent_indices`, `am_list_parent_indices64`, `am_struct_field`, `am_make_struct`, `am_child`, `am_child_count`, `am_map_lookup` |
| Element-wise arithmetic and math | `am_arith_scalar`, `am_arith_array`, `am_unary`, `am_binary`, `am_trig`, `am_logical`, `am_float_class`, `am_math_extra`, `am_unary_checked`, `am_binary_checked`, `am_cumulative`, `am_cumulative_checked` |
| Selection and casting variants of what *is* wrapped | `am_filter_where` (fused compare + filter), `am_group_by` (the dense-key group-by; `am_group_by_keys` + `am_group_agg_ex` are wrapped instead), `am_cast_ex` (cast with child formats and a safety flag) |
| Boolean and Kleene logic | `am_bool_and`, `am_bool_or`, `am_bool_not`, `am_and_kleene`, `am_or_kleene` |
| Structural and conditional | `am_is_null`, `am_is_valid`, `am_fill_null`, `am_fill_null_direction`, `am_drop_null`, `am_if_else`, `am_coalesce`, `am_case_when`, `am_choose`, `am_replace_with_mask`, `am_indices_nonzero`, `am_true_unless_null` |
| Set lookup and hashing | `am_is_in`, `am_index_in`, `am_is_in_ex`, `am_index_in_ex`, `am_hash64`, `am_fixed_binary_hash64`, `am_fixed_binary_compare` |
| Sorting and selection beyond `sort`/`argsort` | `am_top_k`, `am_lexsort`, `am_lexsort_ex`, `am_argsort_ex`, `am_partition_nth_indices`, `am_partition_nth_ex`, `am_rank`, `am_rank_ex`, `am_rank_quantile_ex`, `am_inverse_permutation`, `am_scatter` |
| Window functions and rolling | `am_window` |
| The rest of the aggregates | `am_reduce_ex`, `am_reduce_ex2`, `am_count_all`, `am_first_last`, `am_winsorize`, `am_group_pivot_wider`, `am_pivot_wider` |
| Dictionaries, run-end, uniqueness | `am_dictionary_encode`, `am_dictionary_encode_ex`, `am_dictionary_decode`, `am_run_end_encode`, `am_run_end_decode`, `am_unique`, `am_unique_ex`, `am_value_counts`, `am_value_counts_ex` |
| Joins | `am_join` |
| The fused expression query runner | `am_query`, `am_query_column*`, `am_query_scalar*`, `am_query_canonical` |
| Parquet | `am_parquet_open`, `am_parquet_read`, `am_parquet_read_ex`, `am_parquet_write` and the 15 metadata accessors |
| The streaming engine | all 30 `am_stream_*` entry points |
| Device interop | `am_import_device`, `am_export_device` (CPU `am_import`/`am_export` are wrapped) |
| Extension types, float16, null arrays, random | `am_extension_*`, `am_cast_float16`, `am_null_array`, `am_random` |
| Execution-mode knobs | `am_resident_mode`, `am_resident_mode_available`, `am_resident_mode_reason`, `am_low_latency_wait`, `am_spin_microseconds` |

Wrapping any of them is mechanical: add the `extern "C"` line to `arrowmetal-sys/src/lib.rs`, the
method to `arrowmetal/src/lib.rs`, and a test to `arrowmetal/tests/compute.rs` against arrow-rs (or a
hand oracle where arrow-rs has no counterpart). `tests/signatures.rs` will check the new declaration
against the header on the next `cargo test`.

---

## The measured timing

One run, 2026-09-07, Apple M4 Max, macOS 26.6.2, `rustc 1.95.0`, arrow-rs 59.3.0, ArrowMetal 0.1.0.

**Method.** One 10,000,000-element `Int64Array` of pseudo-random values in `[-1_000_000, 1_000_000)`,
no nulls, built once and shared by every row. Filter predicate `x > 0`, which keeps 5,000,125 of the
10,000,000 rows (50.0%). `std::time::Instant` around the call, wall time, single-threaded, nothing
subtracted, `std::hint::black_box` on every input and result. Three untimed warm-up iterations, then
five timed ones; the table is the **best of the five** (medians were within a few percent, and are in
[`rust/README.md`](../rust/README.md)). Both libraries' answers are asserted equal before anything is
timed. Outside a batch every ArrowMetal call commits its command buffer and waits, so a "kernel"
number is a complete GPU round trip, not an enqueue. Source:
[`rust/arrowmetal/examples/bench.rs`](../rust/arrowmetal/examples/bench.rs); reproduce with
`cargo run --release --example bench` in `rust/`.

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
  import + compare + filter + `to_arrow`. Supporting numbers: import 1.11 ms, export 0.000 ms.

**Where ArrowMetal loses.** A single `sum` on an arrow-rs array is **2.3× slower** than
`arrow::compute::sum`: 2.11 ms against 0.91 ms. The kernel is 3.3× faster; the loss is entirely the
1.11 ms it takes to hand 80 MB to Metal, plus about 0.7 ms of handle setup and first-touch. The
import is copy-free at this size (the values buffer came back aligned to 4 MiB), so that is page
mapping, not a `memcpy` — and it is still 1.1 ms that one cheap kernel does not earn back.

The break-even is roughly "more than one pass over the data": `compare + filter` is two passes and
ArrowMetal is already 1.5× faster end to end. **Import once, chain, export once.** Wrapping a single
reduction is a loss.

---

## Limits

* **Apple silicon only.** `arrowmetal-sys/build.rs` refuses to build anywhere else.
* **Two element types are tested against arrow-rs**: Int64 and Float64, plus Boolean masks and Int32
  indices. Anything else works through the same entry points but is untested from Rust.
* **One thread per handle set.** Handles are `!Send` and `!Sync`; `am_last_error` and batching are
  thread-local.
* **No `RecordBatch` type.** Columns go across one at a time. The plan runner's `Source` takes a
  named set of columns, which is the closest thing here to a table.
* **`arrowmetal-sys` declares 35 of the ABI's 220 entry points**, and every one of them is called by the safe crate (nothing is declared and unused).
  The table above lists what is missing.
* **The crates are not published.** `publish = false` on both; use a path or git dependency.
* **`arrow` is pinned to major version 59.** The C Data Interface structs are ABI-stable, so a
  different arrow-rs major would very likely work, but it is not tested here.
* **A batch defers validation.** Inside `batch(|| …)` an operation that would fail — `take` with an
  out-of-range index, say — returns `Ok` at the call and fails the whole batch at the end, which
  `batch` reports as an `Err`. The value the closure produced is discarded: a readback (a reduction,
  an export) forces a mid-batch flush, so some kernels in a failed batch have run and some have not,
  and the ABI does not say where the line fell. Nested `batch` calls are safe — only the outermost
  opens and commits one — but that is counted in this crate, not in the ABI: `am_batch_end` closes
  whatever is open regardless of nesting.
* **No async.** Every call is synchronous: outside a batch each one commits its command buffer and
  waits. The ABI has an async path (`Sources/ArrowMetal/Async.swift`); this crate does not use it.

## Testing

```bash
cd rust
ARROWMETAL_LIB=/path/to/libArrowMetalC.dylib cargo test --release
```

48 tests, plus 4 `no_run` doc-tests (compiled, not executed), all green as of this writing. What each file compares against is in
[`rust/README.md`](../rust/README.md) and in [TESTING.md](TESTING.md).
