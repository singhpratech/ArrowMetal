// The binding's call overhead on the existing import and sort calls, one row per call and size, so
// two builds can be compared row by row in one session.
//
//   node --expose-gc bench/overhead.mjs [label] [reps]
//
// Every row: calls of the same shape run untimed for at least 100 ms, then the process sleeps
// 500 ms and times one call on its own (first_after_idle_ms), then `reps` timed calls give best and
// median wall time (process.hrtime.bigint) and process CPU time (process.cpuUsage, user + system)
// per call. The result handle is released inside the timed call. CSV on stdout. Uses only calls
// that exist since 0.3.0.

import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { MetalArray, lexsort } = require('../dist/index.js');
const A = require('apache-arrow');

const label = process.argv[2] ?? '';
const REPS = Number(process.argv[3] ?? 30);
const only = process.argv[4] ?? '';

const sleeper = new Int32Array(new SharedArrayBuffer(4));
const sleep = (ms) => Atomics.wait(sleeper, 0, 0, ms);

function row(name, n, fn) {
  if (only && !name.includes(only)) return;
  const t = process.hrtime.bigint();
  while (Number(process.hrtime.bigint() - t) < 100e6) fn();
  sleep(500);
  let t0 = process.hrtime.bigint();
  fn();
  const idle = Number(process.hrtime.bigint() - t0) / 1e6;
  const d = [];
  const c0 = process.cpuUsage();
  for (let i = 0; i < REPS; i++) {
    t0 = process.hrtime.bigint();
    fn();
    d.push(Number(process.hrtime.bigint() - t0) / 1e6);
  }
  const c = process.cpuUsage(c0);
  d.sort((a, b) => a - b);
  const cpu = (c.user + c.system) / 1e3 / REPS;
  console.log(
    `${label},node,${name},${n},${idle.toFixed(4)},${d[0].toFixed(4)},${d[d.length >> 1].toFixed(4)},${cpu.toFixed(4)}`,
  );
}

function int64s(n) {
  const v = new BigInt64Array(n);
  let x = 1n;
  for (let i = 0; i < n; i++) {
    x = (x * 6364136223846793005n + 1442695040888963407n) & 0xffffffffffffffffn;
    v[i] = BigInt.asIntN(64, x) >> 40n;
  }
  return v;
}

function float64s(n) {
  const v = new Float64Array(n);
  let x = 7;
  for (let i = 0; i < n; i++) {
    x = (x * 1103515245 + 12345) % 2147483648;
    v[i] = x / 1024;
  }
  return v;
}

function validityEvery(n, k) {
  const b = new Uint8Array(Math.ceil(n / 8)).fill(0xff);
  let nulls = 0;
  for (let i = 0; i < n; i += k) {
    b[i >> 3] &= ~(1 << (i & 7));
    nulls++;
  }
  return { validity: b, nullCount: nulls };
}

console.log('label,binding,row,rows,first_after_idle_ms,best_ms,median_ms,cpu_ms_per_call');

for (const n of [1000, 1_000_000, 10_000_000]) {
  const vec = A.makeVector(int64s(n));
  row('import_int64_fromArrow', n, () => MetalArray.fromArrow(vec).release());
}
{
  const n = 10_000_000;
  const vals = float64s(n);
  const v = validityEvery(n, 10);
  row('import_float64_nulls_fromTypedArray', n, () => MetalArray.fromTypedArray(vals, v).release());
}

for (const n of [1000, 1_000_000, 10_000_000]) {
  const f = MetalArray.fromTypedArray(float64s(n), validityEvery(n, 10));
  const i = MetalArray.fromTypedArray(int64s(n));
  row('argsort_float64_nulls', n, () => f.argsort(false).release());
  row('argsort_int64_desc', n, () => i.argsort(true).release());
  row('sort_float64_nulls', n, () => f.sort(false).release());
  row('lexsort_int64_float64', n, () => lexsort([i, f], [false, true]).release());
  f.release();
  i.release();
  globalThis.gc?.();
}
