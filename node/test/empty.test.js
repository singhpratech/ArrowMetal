// Zero-row inputs through every import path (fromTypedArray, fromArrow, fromChunks) and back out
// through the C Data export (toArrow, toTypedArray). A zero-length typed array has no backing store,
// so its view has a null base address; the import must still hand over a valid empty array.
const test = require('node:test');
const assert = require('node:assert/strict');
const A = require('apache-arrow');
const { MetalArray } = require('../dist/index.js');

const TYPED = [
  [Int8Array, 'c', A.Int8],
  [Uint8Array, 'C', A.Uint8],
  [Int16Array, 's', A.Int16],
  [Uint16Array, 'S', A.Uint16],
  [Int32Array, 'i', A.Int32],
  [Uint32Array, 'I', A.Uint32],
  [BigInt64Array, 'l', A.Int64],
  [BigUint64Array, 'L', A.Uint64],
  [Float32Array, 'f', A.Float32],
  [Float64Array, 'g', A.Float64],
];
const ARROW_TYPES = [...TYPED.map(([, f, T]) => [f, () => new T()]), ['b', () => new A.Bool()], ['u', () => new A.Utf8()]];

// An empty MetalArray of `format`: length 0, no nulls, and an empty Arrow JS vector of the same type
// on the way out.
function checkEmpty(m, format, what) {
  assert.equal(m.length, 0, what);
  assert.equal(m.nullCount, 0, what);
  assert.equal(m.format, format, what);
  const v = m.toArrow();
  assert.equal(v.length, 0, what);
  assert.equal(v.nullCount, 0, what);
  assert.deepEqual([...v], [], what);
  assert.equal(m.toArray().length, 0, what);
  if (format !== 'b' && format !== 'u') {
    const t = m.toTypedArray();
    assert.equal(t.length, 0, what);
    assert.equal(t.constructor, TYPED.find(([, f]) => f === format)[0], what);
  }
}

test('fromTypedArray takes an empty array of every typed-array type', () => {
  for (const [C, format] of TYPED) {
    const zero = new C(0);
    checkEmpty(MetalArray.fromTypedArray(zero), format, `${C.name}(0)`);
    // an empty validity bitmap, with the null count given and left to be counted
    checkEmpty(MetalArray.fromTypedArray(zero, { validity: new Uint8Array(0) }), format, `${C.name} + validity`);
    checkEmpty(MetalArray.fromTypedArray(zero, { validity: new Uint8Array(0), nullCount: 0 }), format, `${C.name} + nullCount`);
    // a zero-length view into a non-empty buffer, at its start and past it
    checkEmpty(MetalArray.fromTypedArray(new C(4).subarray(0, 0)), format, `${C.name} subarray(0, 0)`);
    checkEmpty(MetalArray.fromTypedArray(new C(4).subarray(4)), format, `${C.name} subarray(4)`);
    // a view over a zero-byte ArrayBuffer, and an empty array still computes
    const m = MetalArray.fromTypedArray(new C(new ArrayBuffer(0)));
    checkEmpty(m, format, `${C.name}(ArrayBuffer(0))`);
    checkEmpty(m.argsort(), 'I', `${C.name} argsort`);
    checkEmpty(m.sort({ descending: true, nulls: 'first', floatOrder: 'nan_largest' }), format, `${C.name} sort`);
    checkEmpty(m.topK(3), 'I', `${C.name} topK`);
    assert.equal(Number(m.sum() ?? 0), 0, `${C.name} sum`);
  }
});

test('fromArrow takes an empty vector and an empty Data of every type', () => {
  for (const [format, type] of ARROW_TYPES) {
    checkEmpty(MetalArray.fromArrow(A.vectorFromArray([], type())), format, `vectorFromArray ${format}`);
    // makeData with no buffers: empty values, and for utf8 an empty offsets buffer
    checkEmpty(MetalArray.fromArrow(A.makeData({ type: type(), length: 0 })), format, `makeData ${format}`);
    // a zero-length slice of a non-empty vector
    const full = A.vectorFromArray(format === 'b' ? [true, null, false] : format === 'u' ? ['a', null, 'bc'] : format === 'l' || format === 'L' ? [1n, null, 2n] : [1, null, 2], type());
    checkEmpty(MetalArray.fromArrow(full.slice(1, 1)), format, `slice ${format}`);
  }
});

test('utf8 rows that are all empty strings import with a zero-byte values buffer', () => {
  const d = A.makeData({ type: new A.Utf8(), length: 3, valueOffsets: new Int32Array(4), data: new Uint8Array(0) });
  const m = MetalArray.fromArrow(d);
  assert.equal(m.length, 3);
  assert.deepEqual([...m.toArrow()], ['', '', '']);
});

test('fromChunks takes empty chunks alone and next to non-empty ones', () => {
  for (const [format, type] of ARROW_TYPES) {
    const empty = () => A.vectorFromArray([], type());
    const made = () => A.makeData({ type: type(), length: 0 });
    checkEmpty(MetalArray.fromChunks([empty(), empty()]), format, `two empty ${format}`);
    checkEmpty(MetalArray.fromChunks([made(), empty(), made()]), format, `made + empty ${format}`);
    const vals = format === 'b' ? [true, null] : format === 'u' ? ['x', null] : format === 'l' || format === 'L' ? [7n, null] : [7, null];
    const m = MetalArray.fromChunks([made(), A.vectorFromArray(vals, type()), empty()]);
    assert.equal(m.length, 2, format);
    assert.equal(m.nullCount, 1, format);
    assert.deepEqual([...m.toArrow()], vals, format);
    // the column of a table with no batches
    const table = new A.Table(new A.Schema([new A.Field('x', type(), true)]));
    checkEmpty(MetalArray.fromChunks([table.getChild('x')]), format, `empty table ${format}`);
  }
});
