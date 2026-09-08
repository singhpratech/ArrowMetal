# TypeScript / Node.js

ArrowMetal from TypeScript: an N-API addon over the one C ABI (`include/arrowmetal.h`), an Arrow
C Data Interface bridge to and from Apache Arrow JS, and a typed API with `.d.ts`. The
package lives in [`node/`](../node); this page is the whole contract.

Every number on this page was measured on 2026-09-07 on an Apple M4 Max, macOS, node v24.9.0,
apache-arrow 21.2.0, ArrowMetal 0.1.0 built `-c release`.

**The browser is out of scope.** There is no Metal in a browser, and this package loads a `.dylib`
through `dlopen`. It is macOS on Apple silicon, in Node, or nothing. There is no WASM fallback and
none is planned.

## Install

The package is not published; build it from the checkout.

```
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
swift build -c release --product ArrowMetalC     # produces .build/release/libArrowMetalC.dylib

cd node
npm install                                      # node-gyp builds the addon during install
npm run build:ts                                 # tsc -> dist/, with .d.ts
npm test
```

`npm install` needs a C++ toolchain (Xcode command line tools) and python3 for node-gyp; both come
with Xcode. `apache-arrow` is a **peer dependency**: everything except `fromArrow` / `toArrow`
works without it.

### Finding the dylib

The addon resolves `libArrowMetalC.dylib` at runtime, in this order:

1. `$ARROWMETAL_LIB` — a full path to the dylib, not a directory
2. `<package>/../.build/release/libArrowMetalC.dylib`

If neither exists the error names both paths and tells you the `swift build` line. There is no
bundled binary and no download step.

```
export ARROWMETAL_LIB=/opt/arrowmetal/libArrowMetalC.dylib
```

`import { info } from 'arrowmetal'` reports which one this process bound to, plus the ArrowMetal
version, the Metal device name and the page size.

## Example

```ts
import { MetalArray, groupBy, PlanSource, runPlan } from 'arrowmetal';
import { vectorFromArray, Int64, Utf8 } from 'apache-arrow';

const amount = MetalArray.fromArrow(vectorFromArray([100n, 250n, 300n, 50n], new Int64()));
const region = MetalArray.fromArrow(vectorFromArray(['e', 'w', 'e', 'w'], new Utf8()));

amount.sum();                                    // 700n   (BigInt: int64 is BigInt everywhere)
amount.mean();                                   // 175    (mean is always a number)
amount.filter(amount.gt(60n)).toArrow();         // Vector<Int64> [100n, 250n, 300n]
amount.sort(true).toArrow();                     // Vector<Int64> [300n, 250n, 100n, 50n]
amount.take(amount.argsort()).toArrow();         // the same as sort()

const gb = groupBy(region);
[...gb.keys(0).toArrow()];                       // ['e', 'w']
[...gb.sum(amount).toArrow()];                   // [400n, 300n]
```

## The copy rule

**Copy-free out, always. Copy-free in when the producer's buffers are page aligned, one copy
otherwise.** That is the project's rule and this binding does not add a copy of its own to either
side.

### In

Arrow JS has no C Data Interface export, so the addon builds the `ArrowSchema` and `ArrowArray`
structs itself:

* **Wrapped, never copied by this binding.** Every buffer pointer is the V8 backing store's own
  address (`ArrayBuffer.Data() + byteOffset`). The addon holds a JS reference to each producer
  buffer in the `ArrowArray`'s own `private_data`, and drops it **when ArrowMetal calls the release
  callback, and at no other time**. That is later than you might expect: on a page-aligned import
  ArrowMetal wraps the V8 pages with `makeBuffer(bytesNoCopy:)`, and a slice, a group-by or a
  registered plan source built from that array retains the import, so the pages stay pinned until
  the last of those is gone. Releasing the *handle* does not unpin them — see
  [Lifetime](#lifetime) below.
* **Then ArrowMetal decides.** It wraps those pages when they are page aligned and copies once when
  they are not. `MetalArray#wrappedProducerBuffers` reports which happened, per handle.
* Buffers involved: the values buffer, the validity bitmap when there are nulls, and the int32
  offsets buffer for `utf8`. All three are wrapped by the addon; a copy, when there is one, is
  ArrowMetal's.

One wrinkle, handled here rather than left to you. Arrow JS's `Data.slice` advances the numeric
values buffer and the utf8 offsets buffer, but leaves the validity bitmap alone and keeps the row
offset in `Data.offset`. The C Data Interface applies one offset to *every* buffer. So on import
this binding rewinds the two advanced buffers to their row-0 origin (a typed-array view at a
different `byteOffset` — no bytes move) and lets the C offset do the work; on export it advances
them again. Sliced and doubly sliced vectors, with and without nulls, are tested.

If a hand-built `Data` has `offset > 0` and a buffer that cannot be rewound (`byteOffset` smaller
than `offset × BYTES_PER_ELEMENT`), the import throws and tells you to materialise the vector. No
Arrow-JS-produced `Data` reaches that path.

### Out

`am_export` hands back an `ArrowArray`; each of its buffers becomes an external `ArrayBuffer`
sharing one refcount. The release callback runs when the last of them is garbage collected. So
`toArrow()` and `toTypedArray()` return views over GPU-resident memory and copy nothing. (`toArray()`
copies, by construction — it builds a JS array.)

### Lifetime

Who keeps what alive, precisely:

| Object | Pinned until |
|---|---|
| The producer's V8 `TypedArray` (import) | ArrowMetal calls the `ArrowArray` release callback — i.e. when the imported array *and everything derived from it* is gone |
| A `MetalArray` handle | garbage collected, or `release()` |
| ArrowMetal's own result buffers (export) | the last external `ArrayBuffer` of that result is garbage collected |

The first row is the one with teeth. `col.slice(...)`, `groupBy(col)` and
`PlanSource.create(name, { col })` all retain the imported array inside ArrowMetal, and on a
page-aligned import that array *is* the V8 pages. So the JS references must outlive the handle, and
they do: they live in the `ArrowArray`'s `private_data`, not on the handle. `col.release()` frees
the handle and nothing else; the derived slice, group-by or plan source keeps reading the right
bytes. Four tests in `test/lifetime.test.js` drop every JS reference to the source typed array,
force two collections, allocate 40 more 1M-element arrays over the freed pages, and then read
through the derived object.

### Measured: are V8's typed-array buffers page aligned?

Page size 16,384 bytes on this machine. "wrapped" is `wrappedProducerBuffers` — ArrowMetal kept the
buffers instead of handing them back during `am_import`.

| Elements | `BigInt64Array` aligned | wrapped | `Float64Array` aligned | wrapped |
|---|---|---|---|---|
| 1,000 | true | true | false | false |
| 100,000 | true | true | true | true |
| **1,000,000** | **true** | **true** | **true** | **true** |
| **10,000,000** | **true** | **true** | **true** | **true** |

**At 1M and 10M elements, both types, the answer is yes** — V8 backs a large typed array with its
own `mmap`ed pages, so the import is a wrap and not a copy. Small typed arrays are packed into the
heap and land wherever they land; the 1,000-element `Float64Array` above was not aligned and was
copied. The test suite does not pin an address: it pins the consequence. A wrapped import sees a
later write through the typed array (`sum` changes); a copied one does not.

A view whose `byteOffset` is not a multiple of the page size is never page aligned and is always
copied. A `byteOffset` that *is* a multiple of the page size, into an already-aligned buffer, still
lands on a page boundary and is still wrapped.

## Covered

| Area | Surface | Backed by |
|---|---|---|
| Interop | `MetalArray.fromArrow`, `fromTypedArray`, `toArrow`, `toTypedArray`, `toArray` | `am_import` / `am_export` |
| Types carried | Int8/16/32/64, Uint8/16/32/64, Float32, Float64, Bool, Utf8 | round-trip test per type, with nulls |
| Reductions | `sum`, `min`, `max`, `mean` | `am_reduce` |
| Compare | `compare(op, scalar)`, `compareWith(op, array)`, `eq ne lt le gt ge` | `am_compare_scalar` / `am_compare_array` |
| Arithmetic | `arith('+' \| '-' \| '*' \| '/', scalar)` | `am_arith_scalar` |
| Cast | `cast(format)` | `am_cast` |
| Selection | `filter`, `take`, `slice` | `am_filter`, `am_take`, `am_slice` |
| Sorting | `argsort`, `sort`, `lexsort([...])` | `am_argsort`, `am_sort`, `am_lexsort` |
| Group-by | `groupBy(keys).sum / mean / min / max / count`, `.keys(i)`, `.groups` | `am_group_by_keys` + `am_group_agg_ex` |
| Query engine | `PlanSource.create`, `runPlan`, `explainPlan`, `PlanResult#column` | `am_plan_source_create`, `am_plan_run`, `am_plan_explain` |
| Diagnostics | `info`, `bufferAddress`, `isPageAligned`, `wrappedProducerBuffers` | — |

Errors from ArrowMetal are thrown as JS `Error`s carrying the `am_last_error()` message verbatim,
except argument-guard rejections (return code 2), which carry the binding's own message: the C
ABI's guards return 2 without setting `am_last_error`, so reading it there would report the
previous call's failure.
Handles are freed by the garbage collector; `release()` frees one now.

## Not covered

The C ABI has 222 entry points. This binding wraps the ones above and no others. Not wrapped:

* strings beyond `utf8` import/export — no `am_str_unary`, `am_str_match`, `am_str_transform`,
  `am_regex`, `am_to_strings`, `am_parse`
* temporal, decimal128/256, nested (list, struct, map, union), dictionary and run-end types
* window functions, cumulative ops, `am_reduce_ex` / `am_reduce_ex2` statistics, `am_top_k`,
  `am_unary` / `am_binary` op tables, Kleene logic and conditionals
* joins (`am_join`), IPC, Parquet, the streaming API (`am_stream_*`), the device C interface
  (`am_import_device` / `am_export_device`), batching (`am_batch_begin` / `am_batch_end`)
* Arrow JS `Table` and `RecordBatch`; only a single-chunk `Vector` or `Data` is accepted. A chunked
  vector is rejected with a message saying to concatenate it first, not silently concatenated.
* every call is synchronous on the JS thread. There is no worker or async variant, so a long kernel
  blocks the event loop.

## Timing

`node bench/spread.mjs` on 2026-09-07. Apple M4 Max, macOS, node v24.9.0, apache-arrow 21.2.0,
ArrowMetal 0.1.0 release. 10,000,000 Int64 rows, no nulls, values `i % 1000`.

**Method**: five fresh processes, each building the dataset once, warming each row up 3 times and
then timing 5 calls (`process.hrtime.bigint()` around the call) and keeping its best. The table
reports the **min, max and median of those five per-process bests**. Every row's answer is checked
against the plain-loop answer inside each process before it reports.

The spread is the point. A single process's best-of-5 is not a stable measurement here — an earlier
one-process run of this same benchmark put `filter` end to end at 8.62 ms, roughly double the
median below — so anything reported as one number would be over-claiming. Two rows whose ranges
overlap cannot be ordered by this measurement, and the benchmark prints those pairs itself.

Two ArrowMetal rows: one with the column already on the device, one that re-imports the
`BigInt64Array` on every call (a page-aligned wrap at this size, so no copy) and, for `filter`,
wraps the result back as a typed array.

### `sum` over 10,000,000 Int64 rows (answer 4995000000)

| Method | Min (ms) | Max (ms) | Median (ms) |
|---|---:|---:|---:|
| ArrowMetal, column already on the device | 0.27 | 0.36 | 0.30 |
| ArrowMetal, end to end from a JS `BigInt64Array` | 1.88 | 2.06 | 2.02 |
| plain typed-array loop | 16.38 | 19.81 | 16.47 |
| Arrow JS, `Vector.get(i)` | 173.52 | 181.75 | 177.97 |
| Arrow JS, `vector.toArray()` then loop | 16.35 | 16.66 | 16.43 |

### `filter x >= 500` over 10,000,000 Int64 rows (5,000,000 kept)

| Method | Min (ms) | Max (ms) | Median (ms) |
|---|---:|---:|---:|
| ArrowMetal, column already on the device | 2.12 | 2.32 | 2.16 |
| ArrowMetal, end to end from a JS `BigInt64Array` | 3.57 | 4.53 | 3.92 |
| plain typed-array loop | 8.08 | 9.52 | 8.44 |
| Arrow JS, `Vector.get(i)` into a new `Vector` | 170.57 | 183.74 | 179.41 |

The only overlapping pair the benchmark found is `plain typed-array loop` and
`Arrow JS, vector.toArray() then loop` in the `sum` table, and they overlap because they are the
same code: `toArray()` on an unsliced `Int64` vector hands back the same `BigInt64Array`.

Reading the tables:

* **The two `filter` rows are not measuring the same amount of work.** ArrowMetal's end-to-end row
  finishes by wrapping the result as a typed-array view — the 5,000,000 output rows are written
  once by the GPU and never touched again on the host — whereas the plain loop writes 5,000,000
  `BigInt`s into a JS array as it goes. Some of the 2.2x is the GPU and some of it is that the
  comparison hands the CPU a materialisation the GPU path does not have to repeat.
* **`Vector.get(i)` is 170–184 ms** because Arrow JS allocates a `BigInt` per row. It is in the
  table because it is what a naive Arrow JS user writes, not because it is a fair kernel.
* The 0.27–0.36 ms resident `sum` is 80 MB read in about 0.3 ms, roughly 265 GB/s, which is in
  range for an M4 Max. It is not a cached answer: a test mutates the wrapped buffer and watches the
  sum change.
* **On these two operations at this size ArrowMetal is ahead in every row.** The small-array
  crossover is not measured here.

Numbers are from one machine on one day. `node bench/spread.mjs` re-runs the whole thing;
`node bench/bench.mjs` runs a single process and prints tables.

## Limits

* **macOS on Apple silicon, in Node.** No browser, no WASM, no Intel, no Linux.
* **int64 is `BigInt`.** `sum`, `min`, `max` on an Int64 or Uint64 column return a `BigInt`;
  `mean` always returns a `number`. Scalars passed to `compare` / `arith` on an int64 column should
  be `BigInt`s (`amount.gt(60n)`, not `amount.gt(60)`); a `number` is accepted and truncated toward
  zero.
  Arrow JS's `Int64` vectors are `BigInt64Array`-backed, which matches.
* **Single-chunk only.** `Table`, `RecordBatch` and chunked `Vector`s are not accepted.
* **Synchronous.** Every call blocks the event loop for the length of the kernel.
* **Handles are GC-managed.** A `MetalArray` holds GPU memory until it is collected; call
  `release()` in a loop that makes many of them. `release()` is safe at any point: it frees the
  handle only, and anything derived from it keeps working, because the producer's buffers stay
  pinned until ArrowMetal itself lets go. See [Lifetime](#lifetime).
* **No prebuilt binary.** `npm install` compiles the addon locally, and the dylib must already
  exist.
* **Not publishable as it stands.** `binding.gyp` adds `../include` so the addon can include the
  repository's `arrow_abi.h`; that path is outside the package, so an `npm pack` tarball installs
  and then fails to compile. The package is therefore marked `"private": true` and the `files`,
  `os` and `cpu` fields have been removed rather than left as a promise the tarball cannot keep.
  Publishing needs either a vendored `arrow_abi.h` under `node/` or a prebuilt binary; neither is
  done here. `main` and `types` are kept, so a local `file:` or `npm link` install resolves.

## Tests

62 tests, `node:test`, oracles are Apache Arrow JS and plain JS over the same rows.
`npm test` sets `NODE_OPTIONS=--expose-gc`, which the lifetime tests need.

```
cd node
ARROWMETAL_LIB=/path/to/libArrowMetalC.dylib npm test
```

| File | Tests | What it pins |
|---|---:|---|
| `test/interop.test.js` | 21 | round trip for all 12 carried types with nulls; sliced, doubly sliced, sliced-with-nulls, sliced utf8 and bool; chunked and unsupported input rejected by message; the alignment table above; wrapped-vs-copied proved by mutating the source buffer; short-buffer rejection with byte counts for validity, values, utf8 offsets, utf8 values and bool; utf8 offsets that decrease or start below zero, named by index; impossible `nullCount`s; type-tagged handles rejected across kinds; an argument-guard rejection never reporting a stale message; 10,000 import/compute/export cycles |
| `test/reductions.test.js` | 10 | sum/min/max/mean against plain-JS oracles for Int64 and Float64, with nulls, all-null, empty; 1,000,001 rows; Kahan-summed float oracle to 1e-9 relative; validity bitmaps with a known and an unknown null count |
| `test/compute.test.js` | 18 | all six comparison ops against JS; null masks; filter on empty and at 1,000,001 rows; sort and argsort with nulls last and stable ties; sort at 1,000,001 rows against `Array.prototype.sort`; take, slice, arith, cast; groupBy sum/mean/min/max/count against a JS `Map`, including a null key group and 1,000,001 rows; lexsort |
| `test/plan.test.js` | 7 | a filter → group_by → sort plan against the same steps in JS; optimized vs unoptimized agree; explain; a plan that does not type-check throws the engine's own message; an out-of-range column named by index and by name |
| `test/lifetime.test.js` | 4 | a slice, a plan source and a group-by all still read the right bytes after the parent handle is released, every JS reference to the source array dropped, two collections forced and the freed pages trampled; and an exported `Vector` after its handle is released |
| `test/workers.test.js` | 2 | four `worker_threads` workers importing, deriving and releasing concurrently, each answering correctly; and workers that exit with references still parked, so the env cleanup hook has to drain them |

Sizes are 0, small, and 1,000,001 — a length that crosses a threadgroup boundary. Nothing above
10,000,000 elements.
