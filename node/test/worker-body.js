// Body of the worker_threads test. Not a test file itself (no .test.js suffix).
const { workerData, parentPort } = require('node:worker_threads');
const { MetalArray, PlanSource, runPlan } = require('../dist/index.js');

const { id, iterations } = workerData;
const N = 1_000_000;

// Handles deliberately left alive so the env is torn down with references still parked.
const leftBehind = [];

try {
  const values = new BigInt64Array(N).fill(BigInt(id + 1));
  const col = MetalArray.fromTypedArray(values);
  const wrapped = col.wrappedProducerBuffers;
  const expected = BigInt(N) * BigInt(id + 1);

  // Churn: import, derive, release the parent, keep the derived object. This is exactly the path
  // whose references outlive the handle.
  for (let i = 0; i < iterations; i++) {
    const small = MetalArray.fromTypedArray(new BigInt64Array([BigInt(i), 2n, 3n]));
    const slice = small.slice(0, 3);
    small.release();
    if (slice.sum() !== BigInt(i) + 5n) throw new Error(`iteration ${i} sum wrong`);
    if (i % 25 === 0) leftBehind.push(slice);
  }

  const source = PlanSource.create('t', { amount: col });
  col.release();
  const r = runPlan({ op: 'scan', source: 't' }, [source]);
  const sum = r.column('amount').sum();
  leftBehind.push(source, r);

  parentPort.postMessage({ ok: true, id, wrapped, sum: String(sum), expected: String(expected) });
} catch (e) {
  parentPort.postMessage({ ok: false, id, error: e.stack ?? String(e) });
}
