// The chunked import (MetalArray.fromChunks) against what a program does without it: copy the
// chunks into one typed array and one validity bitmap, then import that. Both timed end to end,
// handles released inside the timed call.
//
//   node --expose-gc bench/chunks.mjs [label] [reps] [rows,...]
//
// Each chunk is its own Arrow JS Data, as the batches of a Table read from IPC are. Method as in
// bench/overhead.mjs (100 ms warm-up, a 500 ms idle and one call on its own, then `reps` calls for
// best, median and CPU per call). CSV on stdout.

import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { MetalArray } = require('../dist/index.js');
const A = require('apache-arrow');

const label = process.argv[2] ?? '';
const REPS = Number(process.argv[3] ?? 10);
const SIZES = (process.argv[4] ?? '10000000,50000000').split(',').map(Number);

const sleeper = new Int32Array(new SharedArrayBuffer(4));
const sleep = (ms) => Atomics.wait(sleeper, 0, 0, ms);

function row(name, n, chunkRows, fn) {
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
  console.log(
    `${label},node,${name},${n},${chunkRows},${idle.toFixed(3)},${d[0].toFixed(3)},` +
      `${d[d.length >> 1].toFixed(3)},${((c.user + c.system) / 1e3 / REPS).toFixed(3)}`,
  );
}

function makeChunks(kind, n, chunkRows) {
  const out = [];
  let x = 3;
  for (let done = 0; done < n; done += chunkRows) {
    const m = Math.min(chunkRows, n - done);
    if (kind === 'int64') {
      const v = new BigInt64Array(m);
      for (let i = 0; i < m; i++) {
        x = (x * 1103515245 + 12345) % 2147483648;
        v[i] = BigInt(x);
      }
      out.push(A.makeData({ type: new A.Int64(), length: m, data: v }));
    } else {
      const v = new Float64Array(m);
      const bm = new Uint8Array(Math.ceil(m / 8)).fill(0xff);
      let nulls = 0;
      for (let i = 0; i < m; i++) {
        x = (x * 1103515245 + 12345) % 2147483648;
        v[i] = x / 1024;
        if ((done + i) % 10 === 0) {
          bm[i >> 3] &= ~(1 << (i & 7));
          nulls++;
        }
      }
      out.push(A.makeData({ type: new A.Float64(), length: m, nullCount: nulls, nullBitmap: bm, data: v }));
    }
  }
  return out;
}

// The concatenation a program writes without the chunked import: one values array (TypedArray.set
// per chunk) and, when there are nulls, one bitmap assembled bit by bit.
function concatenate(chunks) {
  const n = chunks.reduce((s, c) => s + c.length, 0);
  const Ctor = chunks[0].values.constructor;
  const values = new Ctor(n);
  let at = 0;
  for (const c of chunks) {
    values.set(c.values.subarray(0, c.length), at);
    at += c.length;
  }
  const nullCount = chunks.reduce((s, c) => s + c.nullCount, 0);
  if (nullCount === 0) return MetalArray.fromTypedArray(values);
  const validity = new Uint8Array(Math.ceil(n / 8));
  at = 0;
  for (const c of chunks) {
    const bm = c.nullBitmap;
    for (let i = 0; i < c.length; i++, at++) {
      if (bm === undefined || bm.length === 0 || (bm[(c.offset + i) >> 3] >> ((c.offset + i) & 7)) & 1) {
        validity[at >> 3] |= 1 << (at & 7);
      }
    }
  }
  return MetalArray.fromTypedArray(values, { validity, nullCount });
}

console.log('label,binding,row,rows,chunk_rows,first_after_idle_ms,best_ms,median_ms,cpu_ms_per_call');
for (const n of SIZES) {
  for (const kind of ['int64', 'float64_nulls']) {
    for (const chunkRows of [65_536, 1_000_000]) {
      const chunks = makeChunks(kind, n, chunkRows);
      row(`chunked_import_${kind}`, n, chunkRows, () => MetalArray.fromChunks(chunks).release());
      row(`concatenate_then_import_${kind}`, n, chunkRows, () => concatenate(chunks).release());
      globalThis.gc?.();
    }
  }
}
