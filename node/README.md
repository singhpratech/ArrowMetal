# ArrowMetal for Node.js

Apache Arrow compute on Apple silicon GPUs, from TypeScript and JavaScript. A thin N-API addon over
the one C ABI (`include/arrowmetal.h`), speaking the Arrow C Data Interface in both directions, with
an `apache-arrow` (Arrow JS) `Vector` on either end.

macOS on Apple silicon only. **The browser is out of scope**: there is no Metal there, and this
package loads a `.dylib` through `dlopen`.

Full documentation: [`../docs/TYPESCRIPT.md`](../docs/TYPESCRIPT.md).

**Not publishable as it stands.** `binding.gyp` reaches `../include` for the repository's
`arrow_abi.h`, which does not exist inside an installed package, so a tarball would not build. The
package is marked `private` and `files` / `os` / `cpu` have been removed rather than left as a
promise the tarball cannot keep; see [the note below](#packaging).

## Install

```
cd node
npm install          # builds the addon (node-gyp) as part of the install
npm run build:ts     # compiles src/*.ts to dist/, with .d.ts
npm test
```

`libArrowMetalC.dylib` is found at runtime in this order:

1. `$ARROWMETAL_LIB`, a full path to the dylib
2. `<package>/../.build/release/libArrowMetalC.dylib`

Neither present is an error naming both paths.

```
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
swift build -c release --product ArrowMetalC      # from the repository root
```

## Example

```ts
import { MetalArray, groupBy } from 'arrowmetal';
import { vectorFromArray, Int64, Utf8 } from 'apache-arrow';

const amount = MetalArray.fromArrow(vectorFromArray([100n, 250n, 300n, 50n], new Int64()));
const region = MetalArray.fromArrow(vectorFromArray(['e', 'w', 'e', 'w'], new Utf8()));

amount.sum();                                  // 700n
amount.filter(amount.gt(60n)).toArrow();       // Vector<Int64> [100n, 250n, 300n]

const gb = groupBy(region);
[...gb.keys(0).toArrow()];                     // ['e', 'w']
[...gb.sum(amount).toArrow()];                 // [400n, 300n]
```

## The copy rule

Copy-free out, always. Copy-free in when the producer's buffers are page aligned; one copy
otherwise. This binding adds no copy of its own in either direction:

* **In.** The addon builds the `ArrowArray` / `ArrowSchema` structs over the V8 backing store and
  keeps a JS reference to every producer buffer in the `ArrowArray`'s own `private_data`. Those
  references are dropped when ArrowMetal calls the release callback and at no other time — not when
  the handle is released. ArrowMetal then either wraps those pages
  (`MetalArray#wrappedProducerBuffers === true`) or copies once. Nothing is staged through an
  intermediate buffer here.
* **Out.** Each `ArrowArray` buffer ArrowMetal returns becomes an external `ArrayBuffer` sharing a
  refcount; the array's release callback runs when the last of them is collected.
* Import validates every buffer against the size the Arrow layout requires for `offset + length`
  rows, and throws with the byte counts rather than reading past the end of a short view.

### Lifetime

| Object | Pinned until |
|---|---|
| The producer's V8 `TypedArray` (import) | ArrowMetal calls the `ArrowArray` release callback — when the imported array *and everything derived from it* is gone |
| A `MetalArray` handle | garbage collected, or `release()` |
| ArrowMetal's own result buffers (export) | the last external `ArrayBuffer` of that result is collected |

`col.slice(...)`, `groupBy(col)` and `PlanSource.create(name, { col })` all retain the imported
array inside ArrowMetal, and on a page-aligned import that array *is* the V8 pages. So `release()`
frees the handle and nothing else: the derived object keeps reading the right bytes.
`test/lifetime.test.js` proves it by dropping every JS reference to the source array, forcing two
collections and trampling the freed pages.

### Measured: V8 typed-array page alignment

`node bench/bench.mjs`, node v24.9.0, macOS on an Apple M4 Max, page size 16,384 bytes. "wrapped"
is `wrappedProducerBuffers`: ArrowMetal did not hand the buffers back during `am_import`.

| Elements | `BigInt64Array` aligned | wrapped | `Float64Array` aligned | wrapped |
|---|---|---|---|---|
| 1,000 | true | true | false | false |
| 100,000 | true | true | true | true |
| **1,000,000** | **true** | **true** | **true** | **true** |
| **10,000,000** | **true** | **true** | **true** | **true** |

At 1M and 10M elements V8 gives the backing store its own `mmap`ed pages, so it is page aligned and
the import is free. Small typed arrays are packed together and land wherever they land: the 1,000
element `Float64Array` above was not aligned and was copied. A view whose `byteOffset` is not a
multiple of the page size is never aligned; one that is a multiple, into an aligned buffer, still
lands on a page boundary and is still wrapped. Two tests pin the consequence rather than the
address: a wrapped import sees a later write through the typed array; a copied one does not.

## Measured: `sum` and `filter` at 10,000,000 Int64 rows

`node bench/spread.mjs` on 2026-09-07. Apple M4 Max, macOS, node v24.9.0, apache-arrow 21.2.0,
ArrowMetal 0.1.0 release build. 10,000,000 Int64 rows, no nulls, values `i % 1000`.

**Method**: five fresh processes; each builds the dataset once, warms each row up 3 times, times 5
calls with `process.hrtime.bigint()` and keeps its best. The tables report the **min, max and
median of those five per-process bests**. Every row's answer is checked against the plain-loop
answer inside each process before it reports. A single process's best-of-5 is not stable enough
here to order two methods within ~20% of each other, so the benchmark prints any pair of rows whose
ranges overlap.

### `sum`, answer 4995000000

| Method | Min (ms) | Max (ms) | Median (ms) |
|---|---:|---:|---:|
| ArrowMetal, column already on the device | 0.27 | 0.36 | 0.30 |
| ArrowMetal, end to end from a JS `BigInt64Array` | 1.88 | 2.06 | 2.02 |
| plain typed-array loop | 16.38 | 19.81 | 16.47 |
| Arrow JS, `Vector.get(i)` | 173.52 | 181.75 | 177.97 |
| Arrow JS, `vector.toArray()` then loop | 16.35 | 16.66 | 16.43 |

### `filter x >= 500`, 5,000,000 rows kept

| Method | Min (ms) | Max (ms) | Median (ms) |
|---|---:|---:|---:|
| ArrowMetal, column already on the device | 2.12 | 2.32 | 2.16 |
| ArrowMetal, end to end from a JS `BigInt64Array` | 3.57 | 4.53 | 3.92 |
| plain typed-array loop | 8.08 | 9.52 | 8.44 |
| Arrow JS, `Vector.get(i)` into a new `Vector` | 170.57 | 183.74 | 179.41 |

Read it honestly:

* **The two `filter` rows do not do the same amount of work.** ArrowMetal's end-to-end row finishes
  by wrapping the result as a typed-array view — the 5,000,000 output rows are written once by the
  GPU and never touched on the host — while the plain loop writes 5,000,000 `BigInt`s as it goes.
* The only overlapping pair is `plain typed-array loop` and `Arrow JS, vector.toArray() then loop`
  in the `sum` table; they overlap because `toArray()` on an unsliced `Int64` vector hands back the
  same `BigInt64Array`, so they are the same code.
* `Vector.get(i)` is 170–184 ms because Arrow JS allocates a `BigInt` per row. It is what a naive
  Arrow JS user writes, not a fair kernel.
* The 0.27–0.36 ms resident `sum` is 80 MB in about 0.3 ms, roughly 265 GB/s, in range for an
  M4 Max. Not a cached answer: a test mutates the wrapped buffer and the sum changes.
* An earlier single-process run put `filter` end to end at 8.62 ms against 9.77 ms and called it a
  1.13x win. That was inside the noise; the claim is withdrawn and this table replaces it.

Numbers are from one machine on one day. `node bench/spread.mjs` re-runs it.

## Tests

```
cd node
ARROWMETAL_LIB=/path/to/libArrowMetalC.dylib npm test    # 62 tests, node:test
```

Oracles are Apache Arrow JS and plain JS over the same rows. Sizes include 0, small, and 1,000,001
(a length that crosses a threadgroup boundary), with nulls, all-null columns, empty columns and
sliced input.

## Packaging

`npm publish` is blocked (`"private": true`) and the `files`, `os` and `cpu` fields have been
removed. The reason is `binding.gyp`, which adds `../include` so the addon can include the
repository's `arrow_abi.h`. That path is outside the package, so `npm pack` produces a tarball that
installs and then fails to compile. Publishing needs either a vendored copy of `arrow_abi.h` inside
`node/` or a prebuilt binary; neither is done here, and pretending otherwise with a `files` list
would only make the failure later and stranger. `main` and `types` are kept, since they are what
make a local `file:` or `npm link` install resolve to `dist/`.
