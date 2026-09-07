// Reductions against Apache Arrow JS / plain JS on the same data.
const test = require('node:test');
const assert = require('node:assert/strict');
const A = require('apache-arrow');
const { MetalArray } = require('../dist/index.js');

// Plain-JS oracles over the same rows, nulls skipped, matching Arrow's semantics.
const oracle = {
  sum: (xs) => xs.filter((x) => x !== null).reduce((a, b) => a + b, typeof xs.find((x) => x !== null) === 'bigint' ? 0n : 0),
  min: (xs) => xs.filter((x) => x !== null).reduce((a, b) => (b < a ? b : a)),
  max: (xs) => xs.filter((x) => x !== null).reduce((a, b) => (b > a ? b : a)),
};

test('int64 sum/min/max/mean match a plain-JS oracle', () => {
  const rows = [1n, -5n, 1000n, 42n, 7n];
  const m = MetalArray.fromArrow(A.vectorFromArray(rows, new A.Int64()));
  assert.equal(m.sum(), oracle.sum(rows));
  assert.equal(m.min(), oracle.min(rows));
  assert.equal(m.max(), oracle.max(rows));
  assert.equal(m.mean(), Number(oracle.sum(rows)) / rows.length);
});

test('int64 with nulls skips them, as Arrow does', () => {
  const rows = [1n, null, 3n, null, 8n];
  const m = MetalArray.fromArrow(A.vectorFromArray(rows, new A.Int64()));
  assert.equal(m.nullCount, 2);
  assert.equal(m.sum(), 12n);
  assert.equal(m.min(), 1n);
  assert.equal(m.max(), 8n);
  assert.equal(m.mean(), 4);
});

test('float64 sum/min/max/mean match a plain-JS oracle', () => {
  const rows = [1.5, -2.25, 100.125, 0.0];
  const m = MetalArray.fromArrow(A.vectorFromArray(rows, new A.Float64()));
  assert.equal(m.sum(), oracle.sum(rows));
  assert.equal(m.min(), oracle.min(rows));
  assert.equal(m.max(), oracle.max(rows));
  assert.equal(m.mean(), oracle.sum(rows) / rows.length);
});

test('float64 with nulls', () => {
  const rows = [1.5, null, 2.5, null];
  const m = MetalArray.fromArrow(A.vectorFromArray(rows, new A.Float64()));
  assert.equal(m.nullCount, 2);
  assert.equal(m.sum(), 4);
  assert.equal(m.mean(), 2);
});

test('an all-null column reduces to null', () => {
  const m = MetalArray.fromArrow(A.vectorFromArray([null, null, null], new A.Int64()));
  assert.equal(m.sum(), null);
  assert.equal(m.min(), null);
  assert.equal(m.max(), null);
  assert.equal(m.mean(), null);
});

test('an empty column reduces to null and exports empty', () => {
  const m = MetalArray.fromArrow(A.vectorFromArray([], new A.Int64()));
  assert.equal(m.length, 0);
  assert.equal(m.sum(), null);
  assert.deepEqual([...m.toArrow()], []);
});

test('int64 sum at 1,000,001 rows, a length that crosses a threadgroup boundary', () => {
  const n = 1_000_001;
  const values = new BigInt64Array(n);
  let expected = 0n;
  for (let i = 0; i < n; i++) {
    const v = BigInt(i % 977);
    values[i] = v;
    expected += v;
  }
  const m = MetalArray.fromTypedArray(values);
  assert.equal(m.length, n);
  assert.equal(m.sum(), expected);
  assert.equal(m.max(), 976n);
  assert.equal(m.min(), 0n);
});

test('float64 sum at 1,000,001 rows agrees with a Kahan-summed oracle to 1e-9 relative', () => {
  const n = 1_000_001;
  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) values[i] = (i % 1000) * 0.5;
  // Kahan summation on the host: the GPU reassociates, so compare relatively.
  let sum = 0;
  let c = 0;
  for (let i = 0; i < n; i++) {
    const y = values[i] - c;
    const t = sum + y;
    c = t - sum - y;
    sum = t;
  }
  const m = MetalArray.fromTypedArray(values);
  assert.ok(Math.abs(m.sum() - sum) / sum < 1e-9, `${m.sum()} vs ${sum}`);
});

test('uint64 sums answer with a BigInt', () => {
  const m = MetalArray.fromTypedArray(new BigUint64Array([1n, 2n, 3n]));
  assert.equal(m.format, 'L');
  assert.equal(m.sum(), 6n);
});
