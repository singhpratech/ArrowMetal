// compare / filter / argsort / sort / take, against plain JS on the same rows.
const test = require('node:test');
const assert = require('node:assert/strict');
const A = require('apache-arrow');
const { MetalArray, groupBy, lexsort } = require('../dist/index.js');

test('compare against a scalar gives the same mask as JS', () => {
  const rows = [1n, 5n, 3n, 5n, 9n];
  const m = MetalArray.fromArrow(A.vectorFromArray(rows, new A.Int64()));
  const ops = { '==': (a, b) => a === b, '!=': (a, b) => a !== b, '<': (a, b) => a < b,
    '<=': (a, b) => a <= b, '>': (a, b) => a > b, '>=': (a, b) => a >= b };
  for (const [op, fn] of Object.entries(ops)) {
    assert.deepEqual([...m.compare(op, 5n).toArrow()], rows.map((r) => fn(r, 5n)), op);
  }
});

test('compare with nulls leaves the mask null there, and filter drops those rows', () => {
  const rows = [1n, null, 3n, null, 9n];
  const m = MetalArray.fromArrow(A.vectorFromArray(rows, new A.Int64()));
  const mask = m.gt(2n);
  assert.deepEqual([...mask.toArrow()], [false, null, true, null, true]);
  assert.deepEqual([...m.filter(mask).toArrow()], [3n, 9n]);
});

test('compare between two arrays', () => {
  const a = MetalArray.fromTypedArray(new Float64Array([1, 2, 3, 4]));
  const b = MetalArray.fromTypedArray(new Float64Array([4, 3, 2, 1]));
  assert.deepEqual([...a.compareWith('<', b).toArrow()], [true, true, false, false]);
});

test('filter on an empty array stays empty', () => {
  const m = MetalArray.fromArrow(A.vectorFromArray([], new A.Int64()));
  assert.equal(m.filter(m.gt(0n)).length, 0);
});

test('filter at 1,000,001 rows matches a JS loop', () => {
  const n = 1_000_001;
  const values = new BigInt64Array(n);
  for (let i = 0; i < n; i++) values[i] = BigInt(i % 1000);
  const m = MetalArray.fromTypedArray(values);
  const kept = m.filter(m.ge(500n));
  let expected = 0;
  for (let i = 0; i < n; i++) if (values[i] >= 500n) expected++;
  assert.equal(kept.length, expected);
  assert.equal(kept.min(), 500n);
  assert.equal(kept.max(), 999n);
});

test('sort and argsort agree with JS, nulls last', () => {
  const rows = [5n, null, 1n, 9n, 3n];
  const m = MetalArray.fromArrow(A.vectorFromArray(rows, new A.Int64()));
  assert.deepEqual([...m.sort().toArrow()], [1n, 3n, 5n, 9n, null]);
  assert.deepEqual([...m.sort(true).toArrow()], [9n, 5n, 3n, 1n, null]);
  const ix = [...m.argsort().toArrow()];
  assert.deepEqual(ix.map((i) => rows[i]), [1n, 3n, 5n, 9n, null]);
});

test('argsort is stable across equal keys', () => {
  const rows = [2n, 1n, 2n, 1n, 2n];
  const m = MetalArray.fromArrow(A.vectorFromArray(rows, new A.Int64()));
  assert.deepEqual([...m.argsort().toArrow()], [1, 3, 0, 2, 4]);
});

test('sort at 1,000,001 rows matches Array.prototype.sort', () => {
  const n = 1_000_001;
  const values = new Float64Array(n);
  let seed = 12345;
  for (let i = 0; i < n; i++) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    values[i] = seed / 0x7fffffff;
  }
  const m = MetalArray.fromTypedArray(values);
  const got = m.sort().toTypedArray();
  const want = Float64Array.from(values).sort();
  assert.equal(got.length, want.length);
  for (let i = 0; i < n; i += 9973) assert.equal(got[i], want[i], `row ${i}`);
  assert.equal(got[0], want[0]);
  assert.equal(got[n - 1], want[n - 1]);
});

test('take gathers by Int32 indices', () => {
  const m = MetalArray.fromArrow(A.vectorFromArray([10n, 20n, 30n, 40n], new A.Int64()));
  const ix = MetalArray.fromTypedArray(new Int32Array([3, 0, 2]));
  assert.deepEqual([...m.take(ix).toArrow()], [40n, 10n, 30n]);
});

test('take by argsort reproduces sort', () => {
  const rows = [5.5, 1.5, 9.5, 3.5];
  const m = MetalArray.fromTypedArray(Float64Array.from(rows));
  assert.deepEqual([...m.take(m.argsort()).toArrow()], [...rows].sort((a, b) => a - b));
});

test('slice is a view with the right rows', () => {
  const m = MetalArray.fromArrow(A.vectorFromArray([1n, 2n, 3n, 4n, 5n], new A.Int64()));
  assert.deepEqual([...m.slice(1, 3).toArrow()], [2n, 3n, 4n]);
  assert.equal(m.slice(0, 0).length, 0);
});

test('arithmetic against a scalar', () => {
  const m = MetalArray.fromTypedArray(new Float64Array([1, 2, 3]));
  assert.deepEqual([...m.arith('*', 2).toArrow()], [2, 4, 6]);
  assert.deepEqual([...m.arith('+', 0.5).toArrow()], [1.5, 2.5, 3.5]);
});

test('cast int64 to float64', () => {
  const m = MetalArray.fromTypedArray(new BigInt64Array([1n, 2n, 3n]));
  const f = m.cast('g');
  assert.equal(f.format, 'g');
  assert.deepEqual([...f.toArrow()], [1, 2, 3]);
});

test('groupBy(keys).sum(values) matches a JS Map', () => {
  const keys = ['b', 'a', 'b', 'c', 'a', 'b'];
  const vals = [1n, 2n, 3n, 4n, 5n, 6n];
  const gb = groupBy(MetalArray.fromArrow(A.vectorFromArray(keys, new A.Utf8())));
  const sums = gb.sum(MetalArray.fromArrow(A.vectorFromArray(vals, new A.Int64())));
  const gotKeys = [...gb.keys(0).toArrow()];
  const gotSums = [...sums.toArrow()];
  const want = new Map();
  keys.forEach((k, i) => want.set(k, (want.get(k) ?? 0n) + vals[i]));
  assert.equal(gotKeys.length, want.size);
  gotKeys.forEach((k, i) => assert.equal(gotSums[i], want.get(k), k));
});

test('groupBy over an int key column, with mean, min, max and count', () => {
  const keys = [2, 1, 2, 1, 3];
  const vals = [10, 20, 30, 40, 50];
  const gb = groupBy(MetalArray.fromTypedArray(Int32Array.from(keys)));
  const v = MetalArray.fromTypedArray(Float64Array.from(vals));
  assert.deepEqual([...gb.keys(0).toArrow()], [1, 2, 3]);
  assert.deepEqual([...gb.sum(v).toArrow()], [60, 40, 50]);
  assert.deepEqual([...gb.mean(v).toArrow()], [30, 20, 50]);
  assert.deepEqual([...gb.min(v).toArrow()], [20, 10, 50]);
  assert.deepEqual([...gb.max(v).toArrow()], [40, 30, 50]);
  assert.deepEqual([...gb.count().toArrow()], [2n, 2n, 1n]);
});

test('groupBy skips nothing: a null key forms its own group', () => {
  const keys = A.vectorFromArray([1, null, 1, null], new A.Int32());
  const gb = groupBy(MetalArray.fromArrow(keys));
  assert.equal(gb.groups, 2);
  const sums = gb.sum(MetalArray.fromTypedArray(new BigInt64Array([1n, 2n, 3n, 4n])));
  assert.deepEqual([...sums.toArrow()], [4n, 6n]);
});

test('groupBy at 1,000,001 rows matches a JS Map', () => {
  const n = 1_000_001;
  const keys = new Int32Array(n);
  const vals = new BigInt64Array(n);
  const want = new Map();
  for (let i = 0; i < n; i++) {
    const k = i % 97;
    keys[i] = k;
    vals[i] = BigInt(i % 13);
    want.set(k, (want.get(k) ?? 0n) + vals[i]);
  }
  const gb = groupBy(MetalArray.fromTypedArray(keys));
  const gotKeys = [...gb.keys(0).toArrow()];
  const gotSums = [...gb.sum(MetalArray.fromTypedArray(vals)).toArrow()];
  assert.equal(gotKeys.length, 97);
  gotKeys.forEach((k, i) => assert.equal(gotSums[i], want.get(k), String(k)));
});

test('lexsort orders by two columns', () => {
  const a = MetalArray.fromTypedArray(new Int32Array([1, 1, 2, 2]));
  const b = MetalArray.fromTypedArray(new Int32Array([9, 3, 8, 1]));
  assert.deepEqual([...lexsort([a, b]).toArrow()], [1, 0, 3, 2]);
});
