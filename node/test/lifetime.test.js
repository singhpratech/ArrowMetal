// Who keeps the producer's V8 buffers alive, and for how long.
//
// On a page-aligned import ArrowMetal wraps the V8 pages rather than copying them. Anything derived
// from the imported array — a slice, a group-by, a registered plan source — retains that import, so
// the pages must stay reachable from JS after the original handle has been released. These tests
// drop every JS reference to the source typed array, force a collection, churn the heap, and then
// read through the derived object. Needs --expose-gc (see package.json).
const test = require('node:test');
const assert = require('node:assert/strict');
const A = require('apache-arrow');
const { MetalArray, PlanSource, runPlan, groupBy } = require('../dist/index.js');

const N = 1_000_000;

// Makes the typed array unreachable from the caller: it exists only inside this frame.
function importAndDrop(build) {
  const values = new BigInt64Array(N).fill(1n);
  const col = MetalArray.fromTypedArray(values);
  assert.equal(col.wrappedProducerBuffers, true, 'expected a page-aligned wrap at 1M rows');
  const derived = build(col);
  col.release();
  return derived;
}

// Collect, then trample whatever the collector freed.
function collectAndChurn() {
  global.gc();
  global.gc();
  const junk = [];
  for (let i = 0; i < 40; i++) junk.push(new BigInt64Array(N).fill(BigInt(0xdead0000 + i)));
  junk.length = 0;
  global.gc();
}

test('a slice still reads the right bytes after the parent handle is released and collected', () => {
  const slice = importAndDrop((col) => col.slice(0, N));
  collectAndChurn();
  assert.equal(slice.length, N);
  assert.equal(slice.sum(), BigInt(N));
});

test('a plan source still reads the right bytes after its column is released and collected', () => {
  const source = importAndDrop((col) => PlanSource.create('t', { amount: col }));
  collectAndChurn();
  const r = runPlan({ op: 'scan', source: 't' }, [source]);
  assert.equal(r.rows, N);
  assert.equal(r.column('amount').sum(), BigInt(N));
});

test('a group-by still reads the right bytes after its key column is released and collected', () => {
  const gb = importAndDrop((col) => groupBy(col));
  collectAndChurn();
  assert.equal(gb.groups, 1);
  assert.deepEqual([...gb.keys(0).toArrow()], [1n]);
  assert.deepEqual([...gb.count().toArrow()], [BigInt(N)]);
});

test('an exported vector still reads the right bytes after its handle is released', () => {
  // The other direction: toArrow() wraps ArrowMetal's buffers, and releasing the handle must not
  // free them out from under the Vector.
  let vector;
  {
    const col = MetalArray.fromArrow(A.vectorFromArray([3n, 1n, 2n], new A.Int64()));
    const sorted = col.sort();
    vector = sorted.toArrow();
    sorted.release();
    col.release();
  }
  collectAndChurn();
  assert.deepEqual([...vector], [1n, 2n, 3n]);
});
