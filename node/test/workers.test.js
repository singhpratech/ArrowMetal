// worker_threads: each env owns its own napi references.
//
// An napi_ref belongs to the env that created it, so the queue of references waiting to be deleted
// after ArrowMetal lets go of an imported array is per-env, not per-process. If it were global, one
// worker's drain could run napi_delete_reference on another worker's references, on the wrong
// thread. These tests load the addon in several workers at once, churn imports and releases in each,
// and require every answer to come back right and every worker to exit cleanly.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { Worker } = require('node:worker_threads');

const WORKER = path.join(__dirname, 'worker-body.js');

function runWorkers(count, iterations) {
  return Promise.all(
    Array.from({ length: count }, (_, id) =>
      new Promise((resolve, reject) => {
        const w = new Worker(WORKER, { workerData: { id, iterations } });
        let message;
        w.on('message', (m) => { message = m; });
        w.on('error', reject);
        w.on('exit', (code) => {
          if (code !== 0) reject(new Error(`worker ${id} exited with code ${code}`));
          else if (message === undefined) reject(new Error(`worker ${id} sent no result`));
          else resolve(message);
        });
      })),
  );
}

test('four workers import, compute and release concurrently without crossing envs', async () => {
  const results = await runWorkers(4, 200);
  for (const r of results) {
    assert.equal(r.ok, true, r.error);
    assert.equal(r.sum, String(r.expected), `worker ${r.id} sum`);
    assert.equal(r.wrapped, true, `worker ${r.id} expected a page-aligned wrap`);
  }
});

test('a worker that exits with references still parked does not crash the process', async () => {
  // Each worker deliberately leaves live handles behind when it finishes, so its env is torn down
  // with references still pinned. The env cleanup hook must drain them while the env is usable.
  const results = await runWorkers(3, 50);
  for (const r of results) assert.equal(r.ok, true, r.error);
});
