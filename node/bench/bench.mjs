// sum and filter at 10M Int64 rows: ArrowMetal from Node, a plain typed-array loop, and Arrow JS.
//
// Method: one process, one dataset. Each variant is warmed up 3 times, then run 5 times and the
// best wall time is reported (process.hrtime.bigint around the call). No number here is estimated.
//
//   node bench/bench.mjs           tables on stdout
//   node bench/bench.mjs --json    one JSON object, for bench/spread.mjs
//
// One process is not enough for filter: its cost varies enough between processes that a single
// best-of-5 is inside the noise. bench/spread.mjs runs this file five times and reports the range.

import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { MetalArray, info, isPageAligned } = require('../dist/index.js');
const A = require('apache-arrow');
const arrowVersion = JSON.parse(
  require('node:fs').readFileSync(
    new URL('../node_modules/apache-arrow/package.json', import.meta.url),
    'utf8',
  ),
).version;

const JSON_OUT = process.argv.includes('--json');
const log = (...a) => { if (!JSON_OUT) console.log(...a); };

const N = 10_000_000;
const THRESHOLD = 500n;

// ---------------------------------------------------------------------------------------------

function best(label, fn, { warmup = 3, runs = 5 } = {}) {
  for (let i = 0; i < warmup; i++) fn();
  let bestNs = Infinity;
  let last;
  for (let i = 0; i < runs; i++) {
    const t0 = process.hrtime.bigint();
    last = fn();
    const ns = Number(process.hrtime.bigint() - t0);
    if (ns < bestNs) bestNs = ns;
  }
  return { label, ms: bestNs / 1e6, result: last };
}

// ---------------------------------------------------------------------------------------------
// Data: 10M Int64 rows, no nulls.

const values = new BigInt64Array(N);
for (let i = 0; i < N; i++) values[i] = BigInt(i % 1000);

const vector = A.makeVector(values); // Arrow JS Int64 vector over the same bytes

log(`ArrowMetal ${info.version} on ${info.device}`);
log(`node ${process.version}, apache-arrow ${arrowVersion}`);
log(`page size ${info.pageSize}; the 10M BigInt64Array is page aligned: ${isPageAligned(values)}`);
log(`rows ${N.toLocaleString('en-US')}, Int64, no nulls, filter predicate x >= ${THRESHOLD}`);
log('');

// ---------------------------------------------------------------------------------------------
// sum

const resident = MetalArray.fromTypedArray(values); // imported once, stays on the device
log(`import wrapped the producer buffers (no copy): ${resident.wrappedProducerBuffers}`);

const sumRows = [
  best('ArrowMetal, column already on the device', () => resident.sum()),
  best('ArrowMetal, end to end from a JS BigInt64Array', () =>
    MetalArray.fromTypedArray(values).sum()),
  best('plain typed-array loop', () => {
    let s = 0n;
    for (let i = 0; i < N; i++) s += values[i];
    return s;
  }),
  best('Arrow JS, Vector.get(i)', () => {
    let s = 0n;
    for (let i = 0; i < N; i++) s += vector.get(i);
    return s;
  }),
  best('Arrow JS, vector.toArray() then loop', () => {
    const a = vector.toArray();
    let s = 0n;
    for (let i = 0; i < a.length; i++) s += a[i];
    return s;
  }),
];

const expectedSum = sumRows[2].result;
for (const r of sumRows) {
  if (r.result !== expectedSum) throw new Error(`sum mismatch in "${r.label}": ${r.result}`);
}

// ---------------------------------------------------------------------------------------------
// filter

const filterRows = [
  best('ArrowMetal, column already on the device', () => {
    const kept = resident.filter(resident.ge(THRESHOLD));
    return kept.toTypedArray().length;
  }),
  best('ArrowMetal, end to end from a JS BigInt64Array', () => {
    const m = MetalArray.fromTypedArray(values);
    const kept = m.filter(m.ge(THRESHOLD));
    return kept.toTypedArray().length;
  }),
  best('plain typed-array loop', () => {
    const out = new BigInt64Array(N);
    let k = 0;
    for (let i = 0; i < N; i++) if (values[i] >= THRESHOLD) out[k++] = values[i];
    return out.subarray(0, k).length;
  }),
  best('Arrow JS, Vector.get(i) into a new Vector', () => {
    const out = new BigInt64Array(N);
    let k = 0;
    for (let i = 0; i < N; i++) {
      const v = vector.get(i);
      if (v >= THRESHOLD) out[k++] = v;
    }
    return A.makeVector(out.subarray(0, k)).length;
  }),
];

const expectedKept = filterRows[2].result;
for (const r of filterRows) {
  if (r.result !== expectedKept) throw new Error(`filter mismatch in "${r.label}": ${r.result}`);
}

// ---------------------------------------------------------------------------------------------

function table(title, rows) {
  log(`\n${title}`);
  log('| Method | Best of 5 (ms) | vs fastest |');
  log('|---|---:|---:|');
  const fastest = Math.min(...rows.map((r) => r.ms));
  for (const r of rows) {
    log(`| ${r.label} | ${r.ms.toFixed(2)} | ${(r.ms / fastest).toFixed(2)}x |`);
  }
}

table(`sum over ${N.toLocaleString('en-US')} Int64 rows (answer ${expectedSum})`, sumRows);
table(`filter x >= ${THRESHOLD} over ${N.toLocaleString('en-US')} Int64 rows (${expectedKept.toLocaleString('en-US')} kept)`, filterRows);

// ---------------------------------------------------------------------------------------------
// Alignment survey, for the copy rule in the docs.

log('\nV8 typed-array page alignment (page size ' + info.pageSize + ')');
log('| Elements | BigInt64Array aligned | wrapped by ArrowMetal | Float64Array aligned | wrapped |');
log('|---|---|---|---|---|');
for (const n of [1_000, 100_000, 1_000_000, 10_000_000]) {
  const a = new BigInt64Array(n);
  const b = new Float64Array(n);
  const ma = MetalArray.fromTypedArray(a);
  const mb = MetalArray.fromTypedArray(b);
  log(
    `| ${n.toLocaleString('en-US')} | ${isPageAligned(a)} | ${ma.wrappedProducerBuffers} | ` +
      `${isPageAligned(b)} | ${mb.wrappedProducerBuffers} |`,
  );
}

if (JSON_OUT) {
  process.stdout.write(
    JSON.stringify({
      device: info.device,
      version: info.version,
      node: process.version,
      arrow: arrowVersion,
      rows: N,
      sum: sumRows.map((r) => ({ label: r.label, ms: r.ms })),
      filter: filterRows.map((r) => ({ label: r.label, ms: r.ms })),
    }) + '\n',
  );
}
