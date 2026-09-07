// Import and export across the C Data Interface, and what they cost.
const test = require('node:test');
const assert = require('node:assert/strict');
const A = require('apache-arrow');
const { MetalArray, PlanSource, lexsort, info, isPageAligned, bufferAddress } = require('../dist/index.js');
const { native } = require('../dist/native.js');

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

test('an ArrowMetal error surfaces as a JS Error carrying am_last_error verbatim', () => {
  const a = MetalArray.fromTypedArray(new BigInt64Array([1n, 2n, 3n]));
  const b = MetalArray.fromTypedArray(new BigInt64Array([1n, 2n]));
  assert.throws(() => a.compareWith('==', b), /Array length mismatch: 3 vs 2/);
});

test('an argument-guard rejection (rc 2) never reports a previous call\'s message', () => {
  // The C ABI's guards return 2 without setting am_last_error, so reading it there would report
  // whatever the last failing call left behind. Provoke a real error first, then a guard.
  const a = MetalArray.fromTypedArray(new BigInt64Array([1n, 2n, 3n]));
  const b = MetalArray.fromTypedArray(new BigInt64Array([1n, 2n]));
  assert.throws(() => a.compareWith('==', b), /Array length mismatch/);
  for (const [what, fn] of [
    ['lexsort([])', () => lexsort([])],
    ['PlanSource.create with no columns', () => PlanSource.create('empty', {})],
  ]) {
    assert.throws(fn, (e) => {
      assert.ok(!/Array length mismatch/.test(e.message), `${what} leaked a stale message`);
      assert.match(e.message, /ArrowMetal \(Node\)/);
      return true;
    });
  }
});

test('handles of one kind are rejected where another is expected', () => {
  const a = MetalArray.fromTypedArray(new BigInt64Array([1n, 2n, 3n]));
  const source = PlanSource.create('t', { x: a });
  // Reach past the typed API and hand the raw externals to the wrong entry points.
  assert.throws(() => native.filter(a.handle, source.handle), /expected an array handle/);
  assert.throws(() => native.groupCount(a.handle), /expected a group-by handle/);
  assert.throws(() => native.planColumn(source.handle, 0), /expected a plan result handle/);
  assert.throws(() => native.length({}), /expected an array handle/);
});

test('a buffer shorter than the Arrow layout requires is rejected with the byte counts', () => {
  // A one-byte validity bitmap for 64 rows would read seven bytes past the view.
  assert.throws(
    () => MetalArray.fromTypedArray(new BigInt64Array(64), { validity: new Uint8Array(1) }),
    /the validity buffer is 1 bytes but Arrow format "l" needs at least 8 for offset 0 plus 64 rows/,
  );
  // A values buffer shorter than offset + length.
  assert.throws(
    () => native.importArray('l', 64, 0, 0, null, new BigInt64Array(8), null),
    /the values buffer is 64 bytes but Arrow format "l" needs at least 512/,
  );
  // offset + length, not length alone.
  assert.throws(
    () => native.importArray('g', 4, 4, 0, null, new Float64Array(4), null),
    /the values buffer is 32 bytes but Arrow format "g" needs at least 64 for offset 4 plus 4 rows/,
  );
  // A utf8 offsets buffer needs rows + 1 entries.
  assert.throws(
    () => native.importArray('u', 4, 0, 0, null, new Uint8Array(16), new Int32Array(4)),
    /the utf8 offsets buffer is 16 bytes but Arrow format "u" needs at least 20/,
  );
  // A utf8 values buffer must reach the last offset.
  assert.throws(
    () => native.importArray('u', 2, 0, 0, null, new Uint8Array(3), new Int32Array([0, 2, 9])),
    /the utf8 values buffer is 3 bytes but Arrow format "u" needs at least 9/,
  );
  // A boolean data buffer is a bitmap, so it is sized in bits.
  assert.throws(
    () => native.importArray('b', 64, 0, 0, null, new Uint8Array(2), null),
    /the boolean values buffer is 2 bytes but Arrow format "b" needs at least 8/,
  );
});

test('utf8 offsets that decrease or start below zero are rejected, naming the index', () => {
  assert.throws(
    () => native.importArray('u', 3, 0, 0, null, new Uint8Array(8), new Int32Array([0, 2, 1, 4])),
    /utf8 offsets must not decrease, but offsets\[2\] is 1 after offsets\[1\] = 2/,
  );
  assert.throws(
    () => native.importArray('u', 2, 0, 0, null, new Uint8Array(8), new Int32Array([-1, 0, 1])),
    /utf8 offsets\[0\] is -1; Arrow offsets start at 0 or above/,
  );
  // The check runs over offset + length, not length alone.
  assert.throws(
    () => native.importArray('u', 2, 1, 0, null, new Uint8Array(8), new Int32Array([0, 1, 0, 2])),
    /offsets\[2\] is 0 after offsets\[1\] = 1/,
  );
  // A flat, non-decreasing run is fine (empty strings).
  const ok = native.importArray('u', 3, 0, 0, null, new Uint8Array(2), new Int32Array([0, 1, 1, 2]));
  assert.equal(native.length(ok), 3);
});

test('an impossible nullCount is rejected before it reaches the ABI', () => {
  const v = new BigInt64Array(8);
  assert.throws(
    () => MetalArray.fromTypedArray(v, { validity: new Uint8Array(1), nullCount: 9 }),
    /nullCount 9 is larger than the 8 rows in the array/,
  );
  assert.throws(
    () => MetalArray.fromTypedArray(v, { nullCount: 3 }),
    /nullCount is 3 but no validity bitmap was given/,
  );
  assert.throws(
    () => MetalArray.fromTypedArray(v, { validity: new Uint8Array(1), nullCount: -2 }),
    /nullCount must be a non-negative integer or -1 for unknown, got -2/,
  );
  assert.throws(
    () => MetalArray.fromTypedArray(v, { validity: new Uint8Array(1), nullCount: 1.5 }),
    /nullCount must be a non-negative integer or -1 for unknown, got 1.5/,
  );
  // nullCount equal to the row count is legal: every row null.
  const allNull = MetalArray.fromTypedArray(v, { validity: new Uint8Array(1), nullCount: 8 });
  assert.equal(allNull.nullCount, 8);
  assert.equal(allNull.sum(), null);
});

test('a buffer exactly the required size is accepted', () => {
  const ok = native.importArray('l', 64, 0, 0, new Uint8Array(8).fill(0xff), new BigInt64Array(64).fill(2n), null);
  assert.equal(native.reduce(ok, 0), 128n);
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

test('a wrapped import really is the same memory: mutating the JS buffer changes the answer', () => {
  // The sharpest statement of "copy-free in": at 1M elements the backing store is page aligned,
  // ArrowMetal wraps it, and a later write through the typed array is visible to the kernel.
  const v = new BigInt64Array(1_000_000).fill(1n);
  const m = MetalArray.fromTypedArray(v);
  assert.equal(m.wrappedProducerBuffers, true);
  assert.equal(m.sum(), 1_000_000n);
  v[0] = 1000n;
  assert.equal(m.sum(), 1_000_999n);
});

test('an unaligned import is a copy: mutating the JS buffer does not change the answer', () => {
  const base = new ArrayBuffer(8 * 34);
  const v = new BigInt64Array(base, 8, 33).fill(1n); // byteOffset 8, so never page aligned
  assert.equal(isPageAligned(v), false);
  const m = MetalArray.fromTypedArray(v);
  assert.equal(m.wrappedProducerBuffers, false);
  assert.equal(m.sum(), 33n);
  v[0] = 100n;
  assert.equal(m.sum(), 33n);
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
