# ArrowMetal

[![CI](https://github.com/singhpratech/ArrowMetal/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/singhpratech/ArrowMetal/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/arrowmetal)](https://pypi.org/project/arrowmetal/)
[![Go Reference](https://pkg.go.dev/badge/github.com/singhpratech/ArrowMetal/go/arrowmetal.svg)](https://pkg.go.dev/github.com/singhpratech/ArrowMetal/go/arrowmetal)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Apache Arrow Powered By](https://img.shields.io/badge/Apache_Arrow-Powered_By-0B7285)](https://arrow.apache.org/powered_by/)

**Apache Arrow columnar data on Apple silicon GPUs.** Unified-memory, Metal-accelerated kernels that
understand Arrow's layout natively: validity bitmaps, packed booleans, the C Data Interface and the
C Device Data Interface (`ARROW_DEVICE_METAL`).

Listed on Apache Arrow's [Powered By](https://arrow.apache.org/powered_by/) page. Site and docs: <https://arrowmetal.org> · Python: `pip install arrowmetal` · Rust: `arrowmetal = "0.1.0"` · Release: [v0.1.0](https://github.com/singhpratech/ArrowMetal/releases/tag/v0.1.0)

## The pitch in one paragraph

Every array is Arrow layout in memory the GPU already shares, so there is nothing to upload. Gathers,
group-by, sorts and string scans run on the GPU 3.8x to 24.2x faster than the fastest CPU idiom in the
table below. That idiom is Polars' lazy engine or pyarrow's Acero, using a median of eleven of the
sixteen cores and up to fifteen. While they run the CPU is free for the rest of the application: every table here reports CPU time per operation next to wall time, and the
same rows cost 26 to 1,284 CPU-ms on the other side (the `shift` view apart, at 0.06) against 0.4 to 1.7 here. Chains of operations
share one GPU round trip. The whole thing is reachable from Swift, Python, and any language with Arrow
bindings through one C ABI. The full distribution over all 339 measured rows — including the 77 where
the CPU idiom is ahead — is in [docs/BENCHMARKS_MATRIX.md](docs/BENCHMARKS_MATRIX.md) and
[docs/TO_IMPROVE.md](docs/TO_IMPROVE.md).

## What was measured

- **339 benchmark rows over 173 operations against four CPU libraries at their most parallel idiom** —
  Polars' lazy engine, pyarrow's Acero, pandas and numpy on an idle M4 Max, with the cores each call
  used recorded beside it. 145 rows at or above 3x, 102 between 1x and 3x, 77 where the CPU idiom is
  ahead, 15 with no CPU equivalent; each of the 77 has its cause written down.
  [`docs/BENCHMARKS_MATRIX.md`](docs/BENCHMARKS_MATRIX.md), [`docs/TO_IMPROVE.md`](docs/TO_IMPROVE.md)
- **307 of Apache Arrow v25's 307 compute function names** — 283 on the GPU, 17 on the host, 7 with a
  stated limitation. The table is generated from a registry the test suite executes against
  `pyarrow.compute`. [`docs/ARROW_FUNCTIONS.md`](docs/ARROW_FUNCTIONS.md)
- **39,069 differential cases against pyarrow over 45 column types**, 769 Swift tests against a CPU
  oracle and 2,473 Python cases (2,452 passed, 21 skipped), all run in release.
  [`docs/TESTING.md`](docs/TESTING.md)
- **Seven languages on one C ABI** — Swift, Python, C, Rust, Go, TypeScript and R, each binding with its
  own suite against that language's Arrow library. [`docs/README.md`](docs/README.md)
- **18 findings in other projects** — pyarrow, Apple Metal, the Arrow JS, R and Go libraries, and the
  Swift compiler — each with the evidence behind it and where its report stands.
  [`docs/UPSTREAM.md`](docs/UPSTREAM.md)

## Why this exists

Apple silicon has one physical memory shared by CPU and GPU. An Arrow buffer placed in a `MTLBuffer` with
`storageModeShared` is *simultaneously* a valid CPU Arrow buffer and a valid GPU buffer. No upload and no
download: a page-aligned producer's buffers are used in place, and a buffer that is not page aligned is
copied once on the way in. No Arrow project we surveyed took advantage of that:

| Project | What it has | What it lacks |
|---|---|---|
| [arrow-swift](https://github.com/apache/arrow-swift) | Types, IPC, Flight, C Data Interface | Zero compute kernels, no Metal |
| [nanoarrow device](https://arrow.apache.org/nanoarrow/latest/reference/device.html) | C wrapper for Metal buffers | No compute, C only |
| [cuDF](https://github.com/rapidsai/cudf) | Full GPU dataframe | CUDA only |
| [MLX](https://github.com/ml-explore/mlx) | Fast unified-memory tensors | No nulls, no columnar semantics, ML focused |

ArrowMetal fills that hole: Arrow arrays whose buffers live in Metal shared memory, GPU kernels that honour
Arrow semantics, and standard Arrow C interfaces in and out so it plugs into arrow-rs, pyarrow, DuckDB, Polars,
arrow-swift or anything else that speaks the C Data Interface.

## Benchmarks

Apple M4 Max (16 CPU cores), 50,000,000 rows (10,000,000 for the string row), best of up to five calls
after a warm-up, release build. Full history and methodology in [docs/BENCHMARKS.md](docs/BENCHMARKS.md)
and [Benchmarks/README.md](Benchmarks/README.md).

**Called from Python, same in-process data.** Every CPU library is measured twice — its plain eager
idiom, and the most parallel idiom it has for the same answer: `polars-lazy` through `pl.LazyFrame` on
the in-memory or streaming engine, `pyarrow-threaded` through an Acero plan over 16 record batches
(pandas' threaded paths, numexpr and numba, were not installed for this run; numpy's ufuncs are single-threaded). The baseline below is the **fastest
of all of them**, named, with the cores that call used (its CPU-ms over its wall ms). Every
row is taken from `Benchmarks/results/full_matrix_2026-09-07-parallel.csv`, the raw output behind
[docs/BENCHMARKS_MATRIX.md](docs/BENCHMARKS_MATRIX.md); the eager-only run of the same build is kept
beside it as `full_matrix_2026-09-07.csv`. Wall time in milliseconds, with the CPU time each call
consumed in parentheses. The last three rows are ones where the CPU idiom is ahead:

| Operation | Rows | ArrowMetal ms (CPU-ms) | fastest CPU idiom | its ms (CPU-ms) | its cores | speedup |
|---|---:|---:|---|---:|---:|---:|
| sum int64, 10% nulls | 50,000,000 | 1.11 (0.4) | Polars lazy | 2.30 (26) | 11.1 | 2.1x |
| filter int64 (30% kept) | 50,000,000 | 2.23 (0.7) | Polars lazy | 2.85 (36) | 12.6 | 1.3x |
| take 25M random indices | 50,000,000 | 5.90 (0.8) | pyarrow | 143 (143) | 1.0 | **24.2x** |
| group-by sum, 1000 keys | 50,000,000 | 4.89 (1.2) | pyarrow Acero | 18.45 (250) | 13.6 | **3.8x** |
| filter two columns + sum | 50,000,000 | 3.53 (1.7) | Polars lazy | 2.92 (41) | 13.9 | 0.83x |
| sort float64 | 50,000,000 | 32.12 (1.3) | Polars | 133 (1,284) | 9.7 | **4.1x** |
| string contains, literal | 10,000,000 | 1.64 (0.4) | Polars lazy | 11.47 (173) | 15.0 | **7.0x** |
| ln (float64) | 50,000,000 | 81.31 (0.5) | Polars lazy | 8.46 (120) | 14.2 | 0.10x |
| temporal year | 50,000,000 | 26.82 (0.4) | pyarrow Acero | 17.91 (246) | 13.7 | 0.67x |
| shift (lag 1, int64) | 50,000,000 | 3.58 (0.9) | Polars lazy | 0.03 (0) | 2.2 | 0.01x |

**Swift, against all 16 CPU cores** (tight typed loops over the same Arrow layout) and Accelerate. These
are the Swift-level baselines of rounds 2, 3 and 7 in [docs/BENCHMARKS.md](docs/BENCHMARKS.md), measured
2026-09-06; the sort row predates the 2026-09-07 sort rewrite that the matrix above measures:

| Operation | Metal | 16-core CPU / Accelerate |
|---|---:|---:|
| sum Int64, 10% nulls | **1.40 ms** | 4.89 |
| min Int64 | **1.34** | 3.73 |
| compare Int64 > 0 to bitmap | **1.07** | 2.12 |
| filter Int64 (45% kept) | **2.90** | 3.49 |
| take 25M random indices | **5.85** | 11.93 |
| group-by sum, 5 keys | **1.96** | 5.92 |
| query: filter two columns + sum | **2.33** | 6.36 |
| multiply Int64 * 3 | **2.13** | 2.26 |
| cast Int64 to Float32 | 3.14 | **1.80** |
| sum Float32 | 0.90 | **0.84** (vDSP) |
| multiply Float32 * 2.5 | 1.72 | **1.63** (vDSP) |
| Float32 compare + filter | **2.14** | 5.73 |
| sort Float64, 50M rows | **138.52** | 591.97 (chunk sort + merge tree) |
| string `contains`, 10M utf8 | **1.65** | 17.77 |

Takeaways, all against the fastest idiom of any CPU library. **The rows at or above 3x are gathers,
group-by, sorts and GPU string predicates:** gathers (`take`, 24.2x), multi-key and high-cardinality group-by (`lexsort` 24.0x,
`sum by int32 key` 3.8x at a thousand groups and 6.3x at a hundred thousand), sorts (`argsort int64`
7.7x, `sort float64` 4.1x) and GPU string predicates (`contains` 7.0x at 10M rows; the four GPU
predicates measured at ten million rows — `match_like` on a pure prefix, `contains`, `starts_with` and
`match_substring_regex` on a literal — run 1.6x to 7.0x against Polars lazy and Acero). The group-by family is 65 of 84 rows at or
above 3x. **A single pass that reads one column and writes one is memory bound on both sides:** eleven
to fifteen cores reach the same unified memory the GPU does.

**Across all 339 rows the verdicts are 145 at or above 3x, 102 between 1x and 3x, 77 where the
fastest CPU idiom is ahead, and 15 with no CPU equivalent.** The 77 by measured cause, as
[docs/TO_IMPROVE.md](docs/TO_IMPROVE.md) groups them: 16 at a million rows and below, where the dispatch
floor is the operation; 28 memory-bound single passes; 6 software binary64 transcendentals and calendar
arithmetic; 2 regular expressions matched on the host; 4 where the CPU library returns a view and
ArrowMetal materialises a column; 8 temporal extractions; 4 `unique` and `value_counts` rows against a
threaded hash aggregation; 1 grouped moment; 4 string conversions with variable-length output; and 4
whole-query chains against a streaming engine. `shift (lag 1)` at 0.01x is the lowest ratio in the matrix outside the latency family. Every one of them is listed with its cause and with what would change it in
[docs/TO_IMPROVE.md](docs/TO_IMPROVE.md), and [docs/DESIGN.md](docs/DESIGN.md) has the pipelining plan for the dispatch floor.

## From Python

Two ways to install, both from this checkout; the PyPI upload of 0.1.0 is step 4 of
[docs/RELEASE.md](docs/RELEASE.md).

```
# From source (needs the Swift toolchain)
swift build -c release --product ArrowMetalC        # .build/release/libArrowMetalC.dylib
PYTHONPATH=python python -c "import arrowmetal as am; print(am.device_name())"

# From a wheel that carries the dylib (no Swift toolchain at install time)
pip install build && scripts/build_wheel.sh         # or python/build_wheel.sh, if the dylib is built
pip install python/dist/arrowmetal-0.1.0-*.whl      # macOS arm64 only, pyarrow comes with it
python -c "import arrowmetal as am; print(am.device_name())"
```
```python
import pyarrow as pa, arrowmetal as am               # pyarrow is the only import dependency
col = am.array(pa.array([1, None, 3, 40]))
print(col.filter_where(">", 2).sum())              # 43, computed on the GPU
with am.batch():                                   # several kernels, one GPU round trip
    total = col.filter((col > 1) & (col < 40)).sum()
```
With the `polars` extra installed (`pip install 'arrowmetal[polars]'`, or just `pip install polars`), a Polars
Series crosses the same way:
```python
import polars as pl
col = am.array(pl.Series([1, None, 3, 40]).to_arrow())
print(pl.from_arrow(col.filter_where(">", 2).to_arrow()))
```
```python
from decimal import Decimal

# Group-by over arbitrary key columns — strings here, several columns if you pass several.
# A null key forms its own group, as in Arrow.
region = am.array(pa.array(["emea", "apac", "emea", None, "apac"]))
amount = am.array(pa.array([10.0, 20.0, 30.0, 40.0, 50.0]))
gb = am.group_by([region])
dict(zip(gb.keys()[0].to_pylist(), gb.sum(amount).to_arrow().to_pylist()))
# {'emea': 40.0, 'apac': 70.0, None: 40.0}

# Checked arithmetic raises exactly where the unchecked kernel wraps, naming the row.
big = am.array(pa.array([2**62, 2**62], type=pa.int64()))
(big + big).to_arrow()                             # wraps, which is Arrow's unchecked `add`
big.add_checked(big)                               # ArrowMetalError: add_checked: overflow at index 0

# Regex runs on the host behind the same API; a pattern that is really a literal
# routes back to the GPU kernel.
names = am.array(pa.array(["ArrowMetal", "polars", "pyarrow"]))
names.match_substring_regex("^p").to_arrow()       # [false, true, true]

# decimal128 arithmetic on the GPU; decimal32/64 widen into it and narrow back.
price = am.array(pa.array([Decimal("19.99"), Decimal("5.01")], type=pa.decimal128(10, 2)))
price.decimal_add(price).to_arrow()                # [39.98, 10.02]
```
See [python/README.md](python/README.md).

## Bindings

Four language bindings over the same C ABI (`include/arrowmetal.h`) are in 0.1.0, in this repository, each
exchanging columns through the Arrow C Data Interface and each with its own test suite:

| Binding | Where | Tests | Doc |
|---|---|---|---|
| Rust | `rust/arrowmetal`, `rust/arrowmetal-sys`; [crates.io](https://crates.io/crates/arrowmetal) | 48 tests plus 4 `no_run` doc-tests | [docs/RUST.md](docs/RUST.md) |
| Go | `go/arrowmetal` | 45 test functions and one example, 46 runnable | [docs/GO.md](docs/GO.md) |
| TypeScript / JavaScript | `node/` (N-API addon) | 62 tests | [docs/TYPESCRIPT.md](docs/TYPESCRIPT.md) |
| R | `r/arrowmetal` | 64 `test_that()` blocks in the sources (65 as testthat runs them: the one in test-dispatch.R runs once per attach order), 266 expectations | [docs/R.md](docs/R.md) |

C and C++ callers use the header directly. Java, C# and Julia are on the
[roadmap](docs/ROADMAP.md).

## Quick start (Swift)

```swift
import ArrowMetal

let prices = try MetalArray<Float>([9.5, nil, 12.0, 3.25])       // nullable, lives in Metal shared memory
let mask   = try prices.compare(.gt, 5)                          // GPU, Arrow boolean bitmap out
let picked = try prices.filter(mask)                             // GPU stream compaction
print(try picked.sum(), try picked.max(), picked.nullCount)      // .float(21.5), 12.0, 0

// Multi-column batches behave like an Arrow RecordBatch.
let orders = try MetalRecordBatch(names: ["region", "amount"], columns: [.int32(region), .float32(amount)])
let hits = try orders.filter(try region.compare(.eq, 2).and(try amount.compare(.gt, 100)))
let sample = try hits.take(try MetalArray<Int32>([0, 5, 9]))
let window = try hits.slice(offset: 32, length: 1000)             // zero-copy view
```

Chains of operations can share one GPU round trip:

```swift
let total = try MetalContext.shared.batch {
    try amount.filter(try region.compare(.eq, 2).and(try amount.compare(.gt, 100))).sum()   // one command buffer
}
```

`batchAsync` is the same thing without the wait, so the CPU is free while the GPU scans. Return a pending
array from the body (anything that reads a scalar on the CPU would sync inside it) and use the async
accessors for scalars:

```swift
let ctx = MetalContext.shared
let hits = try await ctx.batchAsync {                          // records, commits, releases the thread
    try amount.filter(try region.compare(.eq, 2).and(try amount.compare(.gt, 100)))
}
print(hits.length)                                             // already resolved: no GPU round trip
let total = try await hits.sumAsync()                          // reduction, still without blocking

// Callback form, for code that is not async:
ctx.batchAsync({ try amount.filter(where: .gt, 100) }) { result in
    switch result { case .success(let hits): print(hits.length); case .failure(let e): print(e) }
}
```

Six end-to-end scenarios (analytics query, feature preparation, Float64 with NaN, C Data Interface
interop, sliced windows, one batched command buffer) live in `Sources/ArrowMetalExamples`: `swift run -c release arrowmetal-examples`.

Interop with any Arrow implementation through the C Data Interface:

```swift
import CArrowABI

var schema = ArrowSchema(); var array = ArrowArray()
producer.export(into: &schema, &array)                            // arrow-rs, pyarrow, nanoarrow, arrow-swift ...
let imported = try importArrowArray(schema: &schema, array: &array)
// imported.zeroCopy == true when the producer's buffers were page aligned

var device = ArrowDeviceArray()
picked.exportArrowDeviceArray(into: &device)                      // device_type == ARROW_DEVICE_METAL
```

Requirements: macOS 14+ / iOS 17+, Swift 5.10+. Shaders compile at runtime so the Command Line Tools are
enough to build and use it. Running the test suite needs Xcode (for XCTest):
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test`.

## Reading Arrow files

The Arrow IPC streaming and file formats are read and written directly, with no dependencies: a small
FlatBuffers codec lives in `Sources/ArrowMetal/IPC`. Buffers land straight in Metal shared memory, so a
file read is ready for the GPU with no further copy.

```swift
let batches = try ArrowIPCReader(url: url).readAll()          // .arrow or .arrows, memory mapped
try ArrowIPCWriter.write(batches, to: url)                     // file format, readable by pyarrow

let reader = try ArrowIPCReader(url: url)                      // random access via the file footer
print(reader.schema.names, reader.batchCount)
let first = try reader.batch(at: 0)

let stream = try ArrowIPCWriter.encode(batches, format: .stream)   // Data, for sockets or Flight
```

Int8 to UInt64, Float32/64, Bool, Utf8, LargeUtf8 and Binary are read into their `MetalArray` types, and
dictionary-encoded columns round trip through complete `DictionaryBatch` messages (`isDelta = false`).
Temporal columns (`date32/64`, `time32/64`, `timestamp`, `duration`) are carried by their storage integer
array while the logical type stays visible in `reader.schema`. Nested columns (list, struct, map, union),
decimals, compressed bodies and big-endian data are rejected with a clear error — IPC is the one place
those types do not go, and [docs/COVERAGE.md](docs/COVERAGE.md) says so on each row.

## What is implemented

**307 of Apache Arrow v25's 307 compute function names** — 283 entirely on the GPU, 17 on the host, 7
with a stated limitation, none missing. Every one of those names has a row in
[docs/ARROW_FUNCTIONS.md](docs/ARROW_FUNCTIONS.md) naming the Swift file behind it, the ArrowMetal call
that reaches it and what it does differently, and that table is generated from a registry the test suite
*executes*: `python/tests/test_functions.py` calls every runnable row and compares the answer to
`pyarrow.compute`. [docs/COVERAGE.md](docs/COVERAGE.md) is the same ground arranged by family, with the
Arrow type matrix and the interop status.

- `MetalArrowBuffer`: page-aligned shared-memory buffers, zero-copy wrap of foreign page-aligned memory.
- `MetalArray<T>` for Int8/16/32/64, UInt8/16/32/64, Float16/32/64; `MetalBooleanArray` with packed bits;
  `MetalStringArray` (`utf8`, `binary`), `MetalDecimalArray` (decimal128/256), `MetalSmallDecimalArray`
  (decimal32/64), `MetalTemporalArray` (`date32/64`, `time32/64`, `timestamp` with timezone, `duration`),
  `MetalIntervalArray` (all three layouts), `MetalFixedBinaryArray`, `MetalListArray`, `MetalStructArray`,
  `MetalMapArray`, `MetalUnionArray`, dictionary, run-end-encoded and extension arrays.
- `MetalRecordBatch`: named equal-length columns with `filter`, `take`, `slice`, `selecting`, plus a GPU
  hash join over int32/int64 keys (`Kernels/Join.swift`).
- **Element-wise**: the six comparisons, wrapping and **checked** (overflow-raising) `add`/`subtract`/
  `multiply`/`divide`/`power`/`negate`/`abs`/`sqrt`/the logarithms/the shifts, `bit_wise_*`, boolean
  `and`/`or`/`not`/`xor`/`and_not` and the Kleene forms, all ten Arrow round modes with `ndigits`,
  `min`/`max` element-wise, `is_nan`/`is_finite`/`is_inf`.
- **Transcendentals**: the twelve trigonometric and hyperbolic functions, their seven `_checked` twins,
  `atan2`, `expm1`, `log1p`, `logb`, `hypot` — float32 through Metal's library functions, float64 through
  a software binary64 implementation on the GPU; measured worst case 5 ulp (`tan`) on float64 and 4 on
  float32 — the per-function table is in [docs/COVERAGE.md](docs/COVERAGE.md).
- **Aggregates**: `sum`, `min`, `max`, `mean`, `product`, `variance`, `stddev`, `quantile`, `median`,
  `mode`, `count_distinct`, `first`/`last`, `index`, `min_max`, `skew`, `kurtosis`, `tdigest`.
- **Grouped aggregates**: all 24 `hash_*` names over **arbitrary** key columns — integers sparse or
  negative, floats, booleans, temporal values, `utf8`, `binary`, dictionary and decimal, several columns
  folded together — mapped to dense ids on the GPU by `GroupByKeys`, with `GroupBy`'s dense-integer fast
  path kept underneath.
- **Selection and ordering**: `filter` (including a fused predicate form), `take`, `slice`, `drop_null`,
  `scatter`, `inverse_permutation`, single- and multi-key `sort_indices` / `lexsort_indices`, `top_k` /
  `select_k_unstable` (a GPU radix select for any k, with a per-threadgroup path at k ≤ 1024), `partition_nth_indices`, `rank`,
  `dense_rank`, `row_number`, `rank_quantile`, `rank_normal`, `winsorize`.
- **Strings**: length, equality, prefix/suffix/containment, `count_substring`, `find_substring`, murmur3
  hash, `dictionary_encode`, the case and title transforms (ASCII and Unicode), padding and centring,
  trimming (ASCII and the Unicode whitespace class), slicing, replace-slice, repeat, reverse, joining,
  `is_in`/`index_in`, and the full `ascii_is_*` / `utf8_is_*` predicate set. The regex family, SQL `LIKE`,
  splitting, Unicode normalisation and the float/boolean casts run on the host behind the same API, with
  a GPU fast path for patterns that are really literals.
- **Temporal**: every extractor (`year` … `nanosecond`, `iso_calendar`, `year_month_day`, `subsecond`,
  `week` with all its options, `us_week`, `us_year`), `ceil`/`floor`/`round_temporal`, every `*_between`
  difference including the three interval-valued ones, `add_interval`, `strftime`/`strptime`, and the
  three timezone functions (`assume_timezone`, `local_timestamp`, `is_dst`) on the host, where the IANA
  database lives.
- **Structural and conditional**: `if_else`, `case_when`, `choose`, `coalesce`, `replace_with_mask`,
  `fill_null`, `fill_null_forward`/`_backward`, `is_null`/`is_valid`/`true_unless_null`,
  `indices_nonzero`, `make_struct`, `struct_field`, `list_value_length`/`list_flatten`/`list_element`/
  `list_slice`/`list_parent_indices`, `map_lookup`, `pivot_wider`, `run_end_encode`/`run_end_decode`.
- **Windows**: `pairwise_diff` (and its checked form), the cumulative family (`sum`, `prod`, `min`, `max`,
  `mean`, and the checked sums and products), `shift`, and rolling `sum`/`mean`/`min`/`max`.
- `libArrowMetalC`: a C ABI over everything above, and a ctypes Python package that speaks the Arrow
  PyCapsule protocol.
- Float64 on the GPU even though Metal has no `double`: compare, min, max, filter, take and slice use an
  order-preserving map of the IEEE bit pattern; sum, add, subtract, multiply, divide, the segmented
  group-by sums and means and the whole transcendental family use a software IEEE-754 binary64
  implementation on 64-bit integers. The arithmetic (`+ − × ÷`) is bit-exact
  against Swift's `Double` over millions of random and edge-case inputs, subnormals and NaN included;
  `sum` and the grouped sums use the same adder but reassociate over threadgroups, so they are checked
  to a tolerance.
  `sqrt` is correctly rounded; `exp`, `ln`, `log10`, `log2` and `power` measure a worst case of 1 ulp
  against libm and are asserted within 2; the trigonometric and hyperbolic families are within 5.
- NaN: `min`/`max` skip NaN and return null if only NaN remains; `sum` propagates NaN; comparisons follow IEEE.
- C Data Interface import/export for every type above and for struct (`+s`) record batches, C Stream
  Interface import, C Device Data Interface import/export, `MTLBuffer` recovery from our own exports.
- `ArrowIPCReader` / `ArrowIPCWriter`: the Arrow IPC streaming and file formats, including a minimal
  FlatBuffers reader and builder, with no dependencies. Cross-checked against pyarrow in both directions.
- A CPU reference implementation behind the kernels, used as the oracle in tests.

## Design notes

- **Runtime shader compilation.** Kernels are MSL strings generated per element type and cached. No `.metal`
  files, no `metal` toolchain, no Xcode required to build.
- **No atomics in reductions.** Each threadgroup writes a partial; the host finalises a few thousand values.
  Deterministic results, works for 64-bit integers where Metal atomics are limited.
- **Bitmaps as 32-bit words.** Compare writes one word per thread; filter uses `popcount` + simdgroup
  prefix sums. Buffers are always padded so whole-word reads are in bounds.
- **Page-aligned allocation.** Every buffer ArrowMetal hands out can be re-wrapped with
  `makeBuffer(bytesNoCopy:)` by another Metal consumer.
- **Lifetime discipline.** Raw pointers are only valid while their owning object lives. Prefer the closure
  accessors (`withValues`, `withTyped`).

See [ROADMAP.md](ROADMAP.md) for what is next and [CONTRIBUTING.md](CONTRIBUTING.md) to get involved.

Questions, and timings from Macs we have not measured: [Discord](https://discord.gg/MEH7QQABUR) or a GitHub issue, whichever you prefer.

## License

Apache License 2.0. Apache Arrow is a trademark of the Apache Software Foundation; this project is
independent and not endorsed by the ASF or Apple.
