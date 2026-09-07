// Import and export across the C Data Interface, and what they cost.
const test = require('node:test');
const assert = require('node:assert/strict');
const A = require('apache-arrow');
const { MetalArray, info, isPageAligned, bufferAddress } = require('../dist/index.js');

test('the loader reports the dylib it bound to', () => {
  assert.ok(info.path.endsWith('libArrowMetalC.dylib'), info.path);
  assert.match(info.version, /^\d+\.\d+\.\d+$/);
  assert.ok(info.device.length > 0);
  assert.equal(info.pageSize, 16384);
});

test('round trip preserves values, nulls and type for every carried type', () => {
  const cases = [
    [new A.Int8(), [1, -2, null, 127]],
    [new A.Int16(), [1, -2, null, 32767]],
    [new A.Int32(), [1, -2, null, 2147483647]],
    [new A.Int64(), [1n, -2n, null, 9007199254740993n]],
    [new A.Uint8(), [0, 255, null, 7]],
    [new A.Uint16(), [0, 65535, null, 7]],
    [new A.Uint32(), [0, 4294967295, null, 7]],
    [new A.Uint64(), [0n, 18446744073709551615n, null, 7n]],
    [new A.Float32(), [1.5, -2.5, null, 0]],
    [new A.Float64(), [1.5, -2.5, null, 1e300]],
    [new A.Bool(), [true, false, null, true]],
    [new A.Utf8(), ['a', '', null, 'hello world']],
  ];
  for (const [type, rows] of cases) {
    const v = A.vectorFromArray(rows, type);
    const back = MetalArray.fromArrow(v).toArrow();
    assert.equal(back.type.toString(), v.type.toString(), String(type));
    assert.deepEqual([...back], rows, String(type));
  }
});

test('a sliced Arrow JS vector imports the right rows', () => {
  const v = A.vectorFromArray([1n, 2n, 3n, 4n, 5n, 6n], new A.Int64());
  for (const [off, len] of [[0, 6], [1, 3], [3, 3], [2, 1], [5, 1]]) {
    const s = v.slice(off, off + len);
    const m = MetalArray.fromArrow(s);
    assert.equal(m.length, len, `${off}/${len}`);
    assert.deepEqual([...m.toArrow()], [...s], `${off}/${len}`);
  }
});

test('a sliced vector with nulls imports the right rows and null positions', () => {
  const rows = [1n, null, 3n, null, 5n, 6n];
  const v = A.vectorFromArray(rows, new A.Int64());
  for (const [off, len] of [[1, 4], [2, 3], [0, 5], [3, 2]]) {
    const s = v.slice(off, off + len);
    const m = MetalArray.fromArrow(s);
    assert.deepEqual([...m.toArrow()], rows.slice(off, off + len), `${off}/${len}`);
    assert.equal(m.sum(), rows.slice(off, off + len).filter((x) => x !== null).reduce((a, b) => a + b, 0n));
  }
});

test('a doubly sliced vector still imports the right rows', () => {
  const v = A.vectorFromArray([1n, 2n, 3n, 4n, 5n, 6n], new A.Int64()).slice(1, 5).slice(1, 3);
  assert.deepEqual([...MetalArray.fromArrow(v).toArrow()], [3n, 4n]);
});

test('a sliced utf8 and a sliced bool vector import correctly', () => {
  const s = A.vectorFromArray(['a', 'bb', 'ccc', 'dddd'], new A.Utf8()).slice(1, 3);
  assert.deepEqual([...MetalArray.fromArrow(s).toArrow()], ['bb', 'ccc']);
  const b = A.vectorFromArray([true, false, true, true, false, true, true, true, false], new A.Bool());
  const bs = b.slice(2, 9);
  assert.deepEqual([...MetalArray.fromArrow(bs).toArrow()], [...bs]);
});

test('a chunked vector is rejected with a message that says what to do', () => {
  const chunked = new A.Vector([
    A.vectorFromArray([1n, 2n], new A.Int64()).data[0],
    A.vectorFromArray([3n], new A.Int64()).data[0],
  ]);
  assert.throws(() => MetalArray.fromArrow(chunked), /single-chunk Vector, got 2 chunks/);
});

test('an unsupported Arrow type is rejected by name', () => {
  const v = A.vectorFromArray([new Date(0)], new A.DateMillisecond());
  assert.throws(() => MetalArray.fromArrow(v), /is not carried by this binding/);
});

test('an ArrowMetal error surfaces as a JS Error carrying am_last_error', () => {
  const a = MetalArray.fromTypedArray(new BigInt64Array([1n, 2n, 3n]));
  const b = MetalArray.fromTypedArray(new BigInt64Array([1n, 2n]));
  assert.throws(() => a.compareWith('==', b), (e) => e instanceof Error && e.message.length > 0);
});

test('exported buffers wrap ArrowMetal memory, not a copy of it', () => {
  const m = MetalArray.fromTypedArray(new BigInt64Array([5n, 1n, 9n]));
  const sorted = m.sort();
  const ta = sorted.toTypedArray();
  assert.deepEqual([...ta], [1n, 5n, 9n]);
  // Two exports of the same handle hand back the same address: nothing was staged through a copy.
  assert.equal(bufferAddress(sorted.toTypedArray()), bufferAddress(ta));
});

test('V8 typed-array page alignment at 1M and 10M elements', () => {
  // The claim under test: copy-free in when the producer's buffers are page aligned, one copy
  // otherwise. These are the sizes the timing table uses.
  for (const n of [1_000_000, 10_000_000]) {
    for (const Ctor of [BigInt64Array, Float64Array]) {
      const a = new Ctor(n);
      assert.ok(isPageAligned(a), `${Ctor.name}(${n}) is not page aligned`);
      const m = MetalArray.fromTypedArray(a);
      assert.equal(m.wrappedProducerBuffers, true, `${Ctor.name}(${n}) was copied at import`);
    }
  }
});

test('a small typed array is not always page aligned, and is then copied', () => {
  // Not an assertion about any one allocation: V8 packs small backing stores into a heap page, so
  // some are aligned by luck and some are not. What must hold is the implication.
  let sawUnaligned = false;
  for (let i = 0; i < 64; i++) {
    const a = new Float64Array(100 + i);
    if (!isPageAligned(a)) {
      sawUnaligned = true;
      assert.equal(MetalArray.fromTypedArray(a).wrappedProducerBuffers, false);
    }
  }
  assert.ok(sawUnaligned, 'expected at least one unaligned small allocation out of 64');
});

test('10,000 import/compute/export cycles do not crash', () => {
  // The export path hands out external ArrayBuffers whose finalizers release the ArrowMetal
  // allocation. This churns them hard enough that a mistake there shows up as a crash.
  for (let i = 0; i < 10_000; i++) {
    const m = MetalArray.fromTypedArray(new BigInt64Array([BigInt(i), 2n, 3n]));
    const s = m.sort();
    assert.equal(s.toTypedArray().length, 3);
    if (i % 1000 === 0) m.release();
  }
});
