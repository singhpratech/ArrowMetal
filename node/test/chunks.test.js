// The chunked import: several Arrow JS chunks, one MetalArray, equal to the import of the
// concatenation.
const test = require('node:test');
const assert = require('node:assert/strict');
const A = require('apache-arrow');
const { MetalArray, PlanSource, runPlan } = require('../dist/index.js');

function rng(seed) {
  let x = BigInt(seed);
  return () => {
    x = (x * 6364136223846793005n + 1442695040888963407n) & 0xffffffffffffffffn;
    return Number(x >> 40n);
  };
}

const TYPES = {
  Int8: [() => new A.Int8(), (x) => x % 100],
  Uint16: [() => new A.Uint16(), (x) => x & 0xffff],
  Int32: [() => new A.Int32(), (x) => x],
  Int64: [() => new A.Int64(), (x) => BigInt(x)],
  Uint64: [() => new A.Uint64(), (x) => BigInt(x) & 0xffffffffffffn],
  Float32: [() => new A.Float32(), (x) => x / 8],
  Float64: [() => new A.Float64(), (x) => x / 16],
  Bool: [() => new A.Bool(), (x) => (x & 1) === 1],
  Utf8: [() => new A.Utf8(), (x) => `s${x % 1000}`],
};

// n rows of one type with a null every nullEvery rows (0: none; 1: all null).
function chunk(name, n, nullEvery, seed) {
  const [type, val] = TYPES[name];
  const r = rng(seed);
  const rows = Array.from({ length: n }, (_, i) =>
    nullEvery > 0 && i % nullEvery === 0 ? null : val(r() - (1 << 23)),
  );
  return A.vectorFromArray(rows, type());
}

// Empty, one-row, sliced (odd offsets, bit offsets), all-null, no-null, one-row slice, many small.
const LAYOUTS = {
  mixed: (t) => {
    const big = chunk(t, 5000, 7, 1);
    return [
      chunk(t, 0, 0, 2), chunk(t, 1, 0, 3), big.slice(3, 1203), chunk(t, 64, 1, 4),
      chunk(t, 1000, 0, 5), big.slice(4001, 4002), chunk(t, 0, 0, 6), big.slice(1999),
      chunk(t, 1, 1, 7), chunk(t, 333, 3, 8),
    ];
  },
  oneChunk: (t) => [chunk(t, 777, 5, 9)],
  allEmpty: (t) => [chunk(t, 0, 0, 1), chunk(t, 0, 0, 2)],
  allNull: (t) => [chunk(t, 10, 1, 1), chunk(t, 1, 1, 2), chunk(t, 100, 1, 3)],
  manySmall: (t) => Array.from({ length: 300 }, (_, i) => chunk(t, i % 5, 3, i)),
};

const values = (v) => [...v];

test('fromChunks equals the import of the concatenation, every type and layout', () => {
  for (const t of Object.keys(TYPES)) {
    for (const [ln, build] of Object.entries(LAYOUTS)) {
      const parts = build(t);
      const concat = parts.reduce((a, b) => a.concat(b));
      const want = values(MetalArray.fromArrow(A.vectorFromArray([...concat], TYPES[t][0]())).toArrow());
      const got = MetalArray.fromChunks(parts);
      assert.equal(got.length, concat.length, `${t} ${ln}`);
      assert.equal(got.nullCount, concat.nullCount, `${t} ${ln}`);
      assert.deepEqual(values(got.toArrow()), want, `${t} ${ln} fromChunks`);
      // A Vector of several Data chunks through fromArrow takes the same path.
      const chunked = new A.Vector(parts.flatMap((p) => p.data));
      assert.deepEqual(values(MetalArray.fromArrow(chunked).toArrow()), want, `${t} ${ln} fromArrow`);
    }
  }
});

test('chunked columns answer the kernels like the concatenation', () => {
  const parts = Array.from({ length: 30 }, (_, i) => {
    const c = chunk('Int64', 20000 + i, 9, i + 1);
    return i % 4 === 1 ? c.slice(17, c.length - 3) : c;
  });
  const got = MetalArray.fromChunks(parts);
  const ref = MetalArray.fromArrow(A.vectorFromArray([...parts.reduce((a, b) => a.concat(b))], new A.Int64()));
  assert.equal(got.sum(), ref.sum());
  assert.equal(got.max(), ref.max());
  for (const nulls of ['last', 'first']) {
    assert.deepEqual([...got.argsort({ descending: true, nulls }).toTypedArray()],
      [...ref.argsort({ descending: true, nulls }).toTypedArray()]);
  }
});

test('a Table of several record batches: its columns import through the chunked path', () => {
  const batches = [0, 1, 2, 3, 4].map((i) => {
    const n = 1000 * i;
    return new A.RecordBatch({ k: chunk('Int64', n, 5, i).data[0], s: chunk('Utf8', n, 3, i + 9).data[0] });
  });
  const table = new A.Table(batches);
  assert.ok(table.getChild('k').data.length > 1);
  const k = MetalArray.fromArrow(table.getChild('k'));
  const s = MetalArray.fromArrow(table.getChild('s'));
  const src = PlanSource.create('b', { k, s });
  const res = runPlan(
    { op: 'aggregate', aggs: [['sum', 't', '(col "k")'], ['count', 'c', '(col "s")']], input: { op: 'scan', source: 'b' } },
    [src],
  );
  let sum = 0n;
  let count = 0n;
  for (const x of table.getChild('k')) if (x !== null) sum += x;
  for (const x of table.getChild('s')) if (x !== null) count += 1n;
  assert.equal(res.column('t').sum(), sum);
  assert.equal(res.column('c').sum(), count);
});

test('empty input, mixed types and unsupported types are refused or handled by name', () => {
  const empty = new A.Vector([A.makeData({ type: new A.Float64(), length: 0, data: new Float64Array(0) })]);
  assert.equal(MetalArray.fromChunks([empty, empty]).length, 0);
  assert.equal(MetalArray.fromChunks([empty, empty]).format, 'g');
  assert.throws(() => MetalArray.fromChunks([]), /at least one chunk/);
  assert.throws(
    () => MetalArray.fromChunks([chunk('Int64', 3, 0, 1), chunk('Float64', 3, 0, 1)]),
    /every chunk must have one type/,
  );
  const dates = A.vectorFromArray([new Date(0)], new A.DateMillisecond());
  assert.throws(() => MetalArray.fromChunks([dates, dates]), /is not carried by this binding/);
});

test('chunk buffers stay pinned until ArrowMetal releases them, and the import outlives them', () => {
  const parts = [chunk('Float64', 3000, 4, 1), chunk('Float64', 5000, 0, 2)];
  const want = values(parts[0].concat(parts[1]));
  let h = MetalArray.fromChunks(parts);
  parts.length = 0;
  global.gc?.();
  global.gc?.();
  // trample freed memory
  const junk = [];
  for (let i = 0; i < 64; i++) junk.push(new Float64Array(1 << 14).fill(1234.5));
  assert.deepEqual(values(h.toArrow()), want);
  for (let i = 0; i < 500; i++) {
    h = MetalArray.fromChunks([chunk('Int32', 50, 3, i), chunk('Int32', 7, 2, i + 1)]);
    if (i % 100 === 0) global.gc?.();
  }
  global.gc?.();
  assert.equal(h.length, 57);
});
