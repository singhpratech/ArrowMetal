// Sort options (null placement; the ieee, total and nan_largest float orders) and top-k, against a
// stable plain-JS reference sort of the same rows.
const test = require('node:test');
const assert = require('node:assert/strict');
const A = require('apache-arrow');
const { MetalArray, lexsort, runPlan, PlanSource } = require('../dist/index.js');

const f64 = new Float64Array(1);
const u64 = new BigUint64Array(f64.buffer);
function bitsOf(x) {
  f64[0] = x;
  return u64[0];
}
function fromBits(b) {
  u64[0] = b;
  return f64[0];
}

// NaN of both signs and several payloads, both zeros, both infinities, subnormals.
const AWKWARD = [
  NaN,
  fromBits(0xfff8000000000000n),
  fromBits(0x7ff0000000000001n),
  fromBits(0xfff8000000000042n),
  fromBits(0x7ff8000000000007n),
  0,
  -0,
  Infinity,
  -Infinity,
  5e-324,
  -1.5e-323,
  1.5,
  -1.5,
];

function rng(seed) {
  let x = BigInt(seed);
  return () => {
    x = (x * 6364136223846793005n + 1442695040888963407n) & 0xffffffffffffffffn;
    return Number(x >> 40n);
  };
}

// The rows (null, or the value's bits as a BigInt, so no NaN payload depends on how the engine
// boxes a double) and the column built from the same Float64Array and a validity bitmap.
function awkwardColumn(n, nullEvery, seed) {
  const r = rng(seed);
  const values = new Float64Array(n);
  const valid = new Uint8Array(Math.ceil(n / 8) || 1);
  const rows = new Array(n);
  let nulls = 0;
  for (let i = 0; i < n; i++) {
    const x = r();
    values[i] = x % 3 === 0 ? AWKWARD[x % AWKWARD.length] : (x % 200) / 4;
    if (nullEvery > 0 && i % nullEvery === 0) {
      rows[i] = null;
      nulls++;
    } else {
      valid[i >> 3] |= 1 << (i & 7);
    }
  }
  const bits = new BigUint64Array(values.buffer);
  for (let i = 0; i < n; i++) if (rows[i] !== null) rows[i] = bits[i];
  return { rows, col: MetalArray.fromTypedArray(values, { validity: valid, nullCount: nulls }) };
}

// Rows hold bits (BigInt); the Int32 key of the lexsort test holds plain numbers.
function totalKey(b) {
  return b >> 63n ? ~b & 0xffffffffffffffffn : b | (1n << 63n);
}
const num = (x) => (typeof x === 'bigint' ? fromBits(x) : x);
const asBits = (x) => (typeof x === 'bigint' ? x : bitsOf(x));

// The documented order of one key: nulls where `nulls` says in both directions; with 'ieee' the NaN
// rows sit next to the nulls in both directions and -0 ties +0; with 'total' a descending sort is
// the mirror; with 'nan_largest' -0 ties +0 and every NaN is one value above +Infinity.
function keyCmp(a, b, o) {
  const first = o.nulls === 'first';
  if (a === null || b === null) {
    if (a === null && b === null) return 0;
    return (a === null) === first ? -1 : 1;
  }
  let c;
  if ((o.floatOrder ?? 'ieee') === 'ieee') {
    const x = num(a);
    const y = num(b);
    const na = Number.isNaN(x);
    const nb = Number.isNaN(y);
    if (na || nb) {
      if (na && nb) return 0;
      return na === first ? -1 : 1;
    }
    c = x < y ? -1 : x > y ? 1 : 0;
  } else if (o.floatOrder === 'nan_largest') {
    const x = num(a);
    const y = num(b);
    const na = Number.isNaN(x);
    const nb = Number.isNaN(y);
    c = na || nb ? (na && nb ? 0 : na ? 1 : -1) : x < y ? -1 : x > y ? 1 : 0;
  } else {
    const ka = totalKey(asBits(a));
    const kb = totalKey(asBits(b));
    c = ka < kb ? -1 : ka > kb ? 1 : 0;
  }
  return o.descending ? -c : c;
}

function refArgsort(rows, o) {
  const idx = rows.map((_, i) => i);
  return idx.sort((i, j) => keyCmp(rows[i], rows[j], o) || i - j); // stable: ties by input order
}

const COMBOS = [];
for (const descending of [false, true])
  for (const nulls of ['last', 'first'])
    for (const floatOrder of ['ieee', 'total', 'nan_largest']) COMBOS.push({ descending, nulls, floatOrder });

const idxOf = (m) => [...m.toTypedArray()];
// A Float64 result's validity and bits, read from its buffers rather than through JS numbers.
function resultBits(m) {
  const v = m.toArrow();
  const vals = m.toTypedArray();
  const bits = new BigUint64Array(vals.buffer, vals.byteOffset, vals.length);
  return [...bits].map((b, i) => (v.isValid(i) ? b : null));
}

test('argsort with options matches the reference for every combination and size', () => {
  for (const n of [0, 1, 33, 1025, 100001]) {
    const { rows, col } = awkwardColumn(n, 7, n + 3);
    for (const o of COMBOS) {
      assert.deepEqual(idxOf(col.argsort(o)), refArgsort(rows, o), `n=${n} ${JSON.stringify(o)}`);
    }
  }
});

test('sort with options is bit-exact against the reference order', () => {
  const { rows, col } = awkwardColumn(20011, 9, 5);
  for (const o of COMBOS) {
    const ref = refArgsort(rows, o);
    assert.deepEqual(resultBits(col.sort(o)), ref.map((i) => rows[i]), JSON.stringify(o));
  }
});

test('an Int64 column places its nulls in both directions (float order ignored)', () => {
  const r = rng(4);
  const rows = Array.from({ length: 50001 }, (_, i) => (i % 6 === 0 ? null : BigInt((r() % 500) - 250)));
  const col = MetalArray.fromArrow(A.vectorFromArray(rows, new A.Int64()));
  for (const o of COMBOS) {
    const ref = rows.map((_, i) => i).sort((i, j) => {
      const a = rows[i];
      const b = rows[j];
      if (a === null || b === null) {
        if (a === null && b === null) return i - j;
        return (a === null) === (o.nulls === 'first') ? -1 : 1;
      }
      const c = a < b ? -1 : a > b ? 1 : 0;
      return (o.descending ? -c : c) || i - j;
    });
    assert.deepEqual(idxOf(col.argsort(o)), ref, JSON.stringify(o));
  }
});

test('Float32 totalOrder and nan_largest with NaN of both signs and both zeros', () => {
  const specials = [NaN, fromBits(0xfff8000000000000n), 0, -0, Infinity, -Infinity, 2.5];
  const r = rng(8);
  const n = 5003;
  const v32 = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = r();
    v32[i] = x % 2 ? specials[x % specials.length] : (x % 40) / 2;
  }
  // Widened to float64 bits by hand from the stored float32 bits (no subnormals here), so neither
  // a NaN's sign nor its payload depends on how the engine loads a float.
  const rows = [...new Uint32Array(v32.buffer)].map((w) => {
    const sign = BigInt(w >>> 31) << 63n;
    const exp = (w >>> 23) & 0xff;
    const mant = BigInt(w & 0x7fffff) << 29n;
    if (exp === 0xff) return sign | (0x7ffn << 52n) | mant;
    if (exp === 0) return sign; // zeros
    return sign | (BigInt(exp - 127 + 1023) << 52n) | mant;
  });
  const col = MetalArray.fromTypedArray(v32);
  for (const o of COMBOS) {
    if (o.floatOrder === 'ieee') continue;
    assert.deepEqual(idxOf(col.argsort(o)), refArgsort(rows, o), JSON.stringify(o));
  }
});

test('the defaults are the plain calls, and a boolean still means descending', () => {
  const { rows, col } = awkwardColumn(30001, 13, 2);
  for (const d of [false, true]) {
    assert.deepEqual(idxOf(col.argsort({ descending: d })), idxOf(col.argsort(d)));
    assert.deepEqual(idxOf(col.argsort(d)), refArgsort(rows, { descending: d }));
    assert.deepEqual(idxOf(col.topK(50, { largest: d })), idxOf(col.topK(50, d)));
  }
  assert.deepEqual(idxOf(col.argsort()), refArgsort(rows, {}));
  assert.deepEqual(idxOf(col.topK(50)), refArgsort(rows, { descending: true }).slice(0, 50));
});

test('topK with options is the head of argsort with the same options', () => {
  const { rows, col } = awkwardColumn(100001, 8, 21);
  for (const o of COMBOS) {
    const full = refArgsort(rows, o);
    for (const k of [0, 1, 10, 1000, 30000, 100010]) {
      const got = idxOf(col.topK(k, { largest: o.descending, nulls: o.nulls, floatOrder: o.floatOrder }));
      assert.deepEqual(got, full.slice(0, k), `k=${k} ${JSON.stringify(o)}`);
    }
  }
  assert.throws(() => col.topK(-1), /non-negative integer/);
  assert.throws(() => col.topK(1.5), /non-negative integer/);
  assert.throws(() => col.argsort({ nulls: 'at_start' }), /nulls must be 'last' or 'first'/);
  assert.throws(() => col.argsort({ floatOrder: 'totalOrder' }), /floatOrder must be 'ieee', 'total' or 'nan_largest'/);
});

test('lexsort with per-key SortOptions matches a stable reference', () => {
  const n = 20011;
  const r = rng(0xabc);
  const k1 = Array.from({ length: n }, (_, i) => (i % 10 === 3 ? null : r() % 6));
  const k1col = MetalArray.fromArrow(A.vectorFromArray(k1, new A.Int32()));
  const { rows: k2, col: k2col } = awkwardColumn(n, 9, 0x5eed);
  for (const o1 of COMBOS) {
    if (o1.floatOrder !== 'ieee') continue; // an integer key ignores the float order
    for (const o2 of COMBOS) {
      const ref = k1
        .map((_, i) => i)
        .sort((i, j) => keyCmp(k1[i], k1[j], o1) || keyCmp(k2[i], k2[j], o2) || i - j);
      assert.deepEqual(idxOf(lexsort([k1col, k2col], [o1, o2])), ref, JSON.stringify([o1, o2]));
    }
  }
  // booleans and options mixed: a boolean is a descending flag with the defaults
  const mixed = k1.map((_, i) => i).sort((i, j) => keyCmp(k1[i], k1[j], { descending: true }) ||
    keyCmp(k2[i], k2[j], { nulls: 'first' }) || i - j);
  assert.deepEqual(idxOf(lexsort([k1col, k2col], [true, { nulls: 'first' }])), mixed);
  assert.throws(() => lexsort([k1col, k2col], [{}]), /2 columns but 1/);
});

test('a plan sort key takes nulls and float_order', () => {
  const { rows, col } = awkwardColumn(5003, 4, 31);
  const src = PlanSource.create('t', { x: col });
  const plan = {
    op: 'sort',
    by: [{ column: 'x', descending: true, nulls: 'first', float_order: 'total' }],
    input: { op: 'scan', source: 't' },
  };
  const ref = refArgsort(rows, { descending: true, nulls: 'first', floatOrder: 'total' });
  assert.deepEqual(resultBits(runPlan(plan, [src]).column('x')), ref.map((i) => rows[i]));
});

test('nan_largest on a hand-picked column: argsort, sort, topK and lexsort', () => {
  //                                  0  1    2     3         4   5          6  7     8
  const negNaN = fromBits(0xfff8000000000000n);
  const vals = [2, NaN, null, Infinity, -1, -Infinity, 0, negNaN, -0];
  const x = MetalArray.fromArrow(A.vectorFromArray(vals, new A.Float64()));
  const cases = [
    // -Infinity, -1, 0 and -0 tied in input order, 2, Infinity, the NaNs (sign ignored), the null
    [{ floatOrder: 'nan_largest' }, [5, 4, 6, 8, 0, 3, 1, 7, 2]],
    [{ nulls: 'first', floatOrder: 'nan_largest' }, [2, 5, 4, 6, 8, 0, 3, 1, 7]],
    [{ descending: true, floatOrder: 'nan_largest' }, [1, 7, 3, 0, 6, 8, 4, 5, 2]],
    [{ descending: true, nulls: 'first', floatOrder: 'nan_largest' }, [2, 1, 7, 3, 0, 6, 8, 4, 5]],
  ];
  const rows = [...new BigUint64Array(new Float64Array(vals.map((v) => v ?? 0)).buffer)].map((b, i) =>
    vals[i] === null ? null : b,
  );
  for (const [o, want] of cases) {
    assert.deepEqual(refArgsort(rows, o), want, `reference ${JSON.stringify(o)}`);
    assert.deepEqual(idxOf(x.argsort(o)), want, JSON.stringify(o));
    assert.deepEqual(resultBits(x.sort(o)), want.map((i) => rows[i]), JSON.stringify(o));
    for (const k of [0, 1, 3, 9, 20]) {
      const got = idxOf(x.topK(k, { largest: o.descending === true, nulls: o.nulls, floatOrder: o.floatOrder }));
      assert.deepEqual(got, want.slice(0, k), `k=${k} ${JSON.stringify(o)}`);
    }
  }
  // A group key splitting the rows in two, then the float key descending under nan_largest.
  const g = MetalArray.fromTypedArray(new Int32Array([1, 0, 1, 0, 1, 0, 1, 0, 1]));
  assert.deepEqual(
    idxOf(lexsort([g, x], [{}, { descending: true, floatOrder: 'nan_largest' }])),
    [1, 7, 3, 5, 0, 6, 8, 4, 2],
  );
});

test('a plan sort takes float_order nan_largest per key and as the sort default, with a limit', () => {
  const { rows, col } = awkwardColumn(20011, 4, 37);
  const src = PlanSource.create('t', { x: col });
  const scan = { op: 'scan', source: 't' };
  for (const descending of [false, true]) {
    for (const nulls of ['last', 'first']) {
      const forms = {
        object: { op: 'sort', by: [{ column: 'x', descending, nulls, float_order: 'nan_largest' }], input: scan },
        array: { op: 'sort', by: [['x', descending, { nulls, float_order: 'nan_largest' }]], input: scan },
        level: { op: 'sort', by: [['x', descending]], nulls, float_order: 'nan_largest', input: scan },
      };
      const ref = refArgsort(rows, { descending, nulls, floatOrder: 'nan_largest' }).map((i) => rows[i]);
      for (const [form, plan] of Object.entries(forms)) {
        const what = `${form} descending=${descending} nulls=${nulls}`;
        assert.deepEqual(resultBits(runPlan(plan, [src]).column('x')), ref, what);
        const top = { op: 'limit', count: 100, input: plan };
        assert.deepEqual(resultBits(runPlan(top, [src]).column('x')), ref.slice(0, 100), `${what} limit 100`);
      }
    }
  }
});

test('the docs/TYPESCRIPT.md sort-options and chunked-column examples', () => {
  const x = MetalArray.fromArrow(A.vectorFromArray([2, null, NaN, -0, 7], new A.Float64()));
  assert.deepEqual(idxOf(x.argsort(true)), [4, 0, 3, 2, 1]);
  assert.deepEqual(idxOf(x.argsort({ descending: true, nulls: 'first', floatOrder: 'total' })), [1, 2, 4, 0, 3]);
  assert.deepEqual(idxOf(x.topK(2, { nulls: 'first', floatOrder: 'total' })), [1, 2]);
  assert.deepEqual(idxOf(x.argsort({ descending: true, floatOrder: 'nan_largest' })), [2, 4, 0, 3, 1]);
  const c = MetalArray.fromChunks([
    A.vectorFromArray([1n, 2n], new A.Int64()),
    A.vectorFromArray([3n, null, 5n], new A.Int64()),
  ]);
  assert.equal(c.length, 5);
  assert.equal(c.sum(), 11n);
});
