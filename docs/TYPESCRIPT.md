# TypeScript / Node.js

ArrowMetal from TypeScript: an N-API addon over the one C ABI (`include/arrowmetal.h`), an Arrow
C Data Interface bridge to and from Apache Arrow JS, and a typed API shipped with `.d.ts`. The
package lives in [`node/`](../node); this page is the whole contract.

Every number on this page was measured on 2026-09-07 on an Apple M4 Max, macOS, node v24.9.0,
apache-arrow 21.2.0, ArrowMetal 0.1.0 built `-c release`. Nothing here is estimated.

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
  address (`ArrayBuffer.Data() + byteOffset`). The addon holds a JS reference to each buffer for the
  whole life of the handle, so ArrowMetal may keep them for as long as it likes. The `ArrowArray`
  release callback we install only records that ArrowMetal let go.
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

An `ArrayBuffer` view with a non-zero `byteOffset` is never page aligned and is always copied.

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

Errors from ArrowMetal are thrown as JS `Error`s carrying the `am_last_error()` message verbatim.
Handles are freed by the garbage collector; `release()` frees one now.

## Not covered

The C ABI has 220 entry points. This binding wraps the ones above and no others. Not wrapped:

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

One run of `node bench/bench.mjs` on 2026-09-07. Apple M4 Max, macOS, node v24.9.0,
apache-arrow 21.2.0, ArrowMetal 0.1.0 release. 10,000,000 Int64 rows, no nulls, values `i % 1000`.

**Method**: one process, one dataset. Each row is warmed up 3 times, then run 5 times; the table
reports the **best of 5** wall time, `process.hrtime.bigint()` around the call. Every row's answer
is checked against the plain-loop answer before the table prints. Two ArrowMetal rows: one with the
column already on the device, one that re-imports the `BigInt64Array` on every call (a page-aligned
wrap at this size, so no copy) and, for `filter`, wraps the result back as a typed array.

### `sum` over 10,000,000 Int64 rows (answer 4995000000)

| Method | Best of 5 (ms) | vs fastest |
|---|---:|---:|
| ArrowMetal, column already on the device | 0.28 | 1.00x |
| ArrowMetal, end to end from a JS `BigInt64Array` | 2.04 | 7.34x |
| plain typed-array loop | 16.55 | 59.59x |
| Arrow JS, `Vector.get(i)` | 175.21 | 630.92x |
| Arrow JS, `vector.toArray()` then loop | 16.67 | 60.03x |

### `filter x >= 500` over 10,000,000 Int64 rows (5,000,000 kept)

| Method | Best of 5 (ms) | vs fastest |
|---|---:|---:|
| ArrowMetal, column already on the device | 3.10 | 1.00x |
| ArrowMetal, end to end from a JS `BigInt64Array` | 8.62 | 2.78x |
| plain typed-array loop | 9.77 | 3.15x |
| Arrow JS, `Vector.get(i)` into a new `Vector` | 184.38 | 59.41x |

What these say, including the parts that are not flattering:

* **`filter` end to end barely wins.** 8.62 ms against a plain typed-array loop's 9.77 ms: 1.13x.
  Filter materialises 5,000,000 rows, and that write is most of the work on both sides. If your
  data starts and ends in a JS typed array and you filter once, ArrowMetal is not worth the call.
  It pays when the column stays on the device across several operations (3.10 ms there, 3.2x).
* **`Arrow JS, toArray() then loop` ties the plain loop** at 16.6 ms, because `toArray()` on an
  unsliced `Int64` vector hands back the same `BigInt64Array`. It *is* the plain loop.
* **`Vector.get(i)` is 175–184 ms** because Arrow JS allocates a `BigInt` per row. It is in the
  table because it is what a naive Arrow JS user writes, not because it is a fair kernel.
* The 0.28 ms resident `sum` is 80 MB in 0.28 ms, about 285 GB/s — in range for an M4 Max, and not
  a cached answer: a test mutates the wrapped buffer and watches the sum change.

Numbers are from one machine on one day. `node bench/bench.mjs` re-runs the whole thing.

## Limits

* **macOS on Apple silicon, in Node.** No browser, no WASM, no Intel, no Linux.
* **int64 is `BigInt`.** `sum`, `min`, `max` on an Int64 or Uint64 column return a `BigInt`;
  `mean` always returns a `number`. Scalars passed to `compare` / `arith` on an int64 column must
  be `BigInt`s (`amount.gt(60n)`, not `amount.gt(60)`) — a `number` is accepted and truncated.
  Arrow JS's `Int64` vectors are `BigInt64Array`-backed, which matches.
* **Single-chunk only.** `Table`, `RecordBatch` and chunked `Vector`s are not accepted.
* **Synchronous.** Every call blocks the event loop for the length of the kernel.
* **Handles are GC-managed.** A `MetalArray` holds GPU memory until it is collected; call
  `release()` in a loop that makes many of them.
* **No prebuilt binary.** `npm install` compiles the addon locally, and the dylib must already
  exist.

## Tests

50 tests, `node:test`, oracles are Apache Arrow JS and plain JS over the same rows.

```
cd node
ARROWMETAL_LIB=/path/to/libArrowMetalC.dylib npm test
```

| File | Tests | What it pins |
|---|---:|---|
| `test/interop.test.js` | 15 | round trip for all 12 carried types with nulls; sliced, doubly sliced, sliced-with-nulls, sliced utf8 and bool; chunked and unsupported input rejected by message; the alignment table above; wrapped-vs-copied proved by mutating the source buffer; 10,000 import/compute/export cycles |
| `test/reductions.test.js` | 10 | sum/min/max/mean against plain-JS oracles for Int64 and Float64, with nulls, all-null, empty; 1,000,001 rows; Kahan-summed float oracle to 1e-9 relative; validity bitmaps with a known and an unknown null count |
| `test/compute.test.js` | 18 | all six comparison ops against JS; null masks; filter on empty and at 1,000,001 rows; sort and argsort with nulls last and stable ties; sort at 1,000,001 rows against `Array.prototype.sort`; take, slice, arith, cast; groupBy sum/mean/min/max/count against a JS `Map`, including a null key group and 1,000,001 rows; lexsort |
| `test/plan.test.js` | 7 | a filter → group_by → sort plan against the same steps in JS; optimized vs unoptimized agree; explain; a plan that does not type-check throws the engine's message |

Sizes are 0, small, and 1,000,001 — a length that crosses a threadgroup boundary. Nothing above
10,000,000 elements.
