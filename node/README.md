# ArrowMetal for Node.js

Apache Arrow compute on Apple silicon GPUs, from TypeScript and JavaScript. A thin N-API addon over
the one C ABI (`include/arrowmetal.h`), speaking the Arrow C Data Interface in both directions, with
an `apache-arrow` (Arrow JS) `Vector` on either end.

macOS on Apple silicon only. **The browser is out of scope**: there is no Metal there, and this
package loads a `.dylib` through `dlopen`.

Full documentation: [`../docs/TYPESCRIPT.md`](../docs/TYPESCRIPT.md).

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
  keeps a JS reference to every buffer for the life of the handle. ArrowMetal then either wraps
  those pages (`MetalArray#wrappedProducerBuffers === true`) or copies once. Nothing is staged
  through an intermediate buffer here.
* **Out.** Each `ArrowArray` buffer ArrowMetal returns becomes an external `ArrayBuffer` sharing a
  refcount; the array's release callback runs when the last of them is collected.

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
element `Float64Array` above was not aligned and was copied. Two tests pin the consequence rather
than the address: a wrapped import sees a later write through the typed array; a copied one does
not.

## Measured: `sum` and `filter` at 10,000,000 Int64 rows

One run of `node bench/bench.mjs` on 2026-09-07. Apple M4 Max, macOS, node v24.9.0,
apache-arrow 21.2.0, ArrowMetal 0.1.0 release build. 10,000,000 Int64 rows, no nulls,
values `i % 1000`. Method: one process, one dataset, 3 warm-up calls then 5 timed calls per row,
**best of 5**, wall time from `process.hrtime.bigint()` around the call. Every row's answer is
checked against the plain-loop answer before the table is printed.

The two ArrowMetal rows are the honest pair: the first is the kernel with the column already on the
device, the second includes importing a fresh `BigInt64Array` on every call (which at this size is a
page-aligned wrap, not a copy) and, for `filter`, wrapping the result back as a typed array.

### `sum`, answer 4995000000

| Method | Best of 5 (ms) | vs fastest |
|---|---:|---:|
| ArrowMetal, column already on the device | 0.28 | 1.00x |
| ArrowMetal, end to end from a JS `BigInt64Array` | 2.04 | 7.34x |
| plain typed-array loop | 16.55 | 59.59x |
| Arrow JS, `Vector.get(i)` | 175.21 | 630.92x |
| Arrow JS, `vector.toArray()` then loop | 16.67 | 60.03x |

### `filter x >= 500`, 5,000,000 rows kept

| Method | Best of 5 (ms) | vs fastest |
|---|---:|---:|
| ArrowMetal, column already on the device | 3.10 | 1.00x |
| ArrowMetal, end to end from a JS `BigInt64Array` | 8.62 | 2.78x |
| plain typed-array loop | 9.77 | 3.15x |
| Arrow JS, `Vector.get(i)` into a new `Vector` | 184.38 | 59.41x |

Read it honestly:

* `filter` end to end is 8.62 ms against a plain loop's 9.77 ms. That is a 1.13x win, not a
  headline. Filter has to write 5,000,000 rows out; the GPU's advantage on the scan is mostly
  spent on the materialisation.
* `Arrow JS, vector.toArray()` for `sum` is the same 16.6 ms as the plain loop, because
  `toArray()` on an unsliced `Int64` vector hands back the same `BigInt64Array` — it is the plain
  loop, reached through Arrow JS.
* `Vector.get(i)` is 175–184 ms because Arrow JS allocates a `BigInt` per row. It is what a naive
  Arrow JS user actually writes, so it is in the table, but it is not the interesting comparison.
* The 0.28 ms resident `sum` is 80 MB read in 0.28 ms, about 285 GB/s, which is in range for an
  M4 Max. It is not a cached answer: a test mutates the wrapped buffer and the sum changes.

Numbers are from one machine on one day. Re-run `node bench/bench.mjs` on yours.

## Tests

```
cd node
ARROWMETAL_LIB=/path/to/libArrowMetalC.dylib npm test    # 50 tests, node:test
```

Oracles are Apache Arrow JS and plain JS over the same rows. Sizes include 0, small, and 1,000,001
(a length that crosses a threadgroup boundary), with nulls, all-null columns, empty columns and
sliced input.
