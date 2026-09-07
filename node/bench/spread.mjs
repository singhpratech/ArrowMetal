// Runs bench/bench.mjs in five fresh processes and reports the range, not a single number.
//
//   node bench/spread.mjs
//
// A single process's best-of-5 is not a stable measurement for `filter`: the cost moves enough
// between processes (allocator state, GPU clocks, page placement) that two methods within ~20% of
// each other cannot be ordered from one run. So each row here is best-of-5 within a process,
// reported as min–max across 5 processes, and the summary says plainly when two rows overlap.

import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const BENCH = fileURLToPath(new URL('./bench.mjs', import.meta.url));
const PROCESSES = 5;

const runs = [];
for (let i = 0; i < PROCESSES; i++) {
  const r = spawnSync(process.execPath, [BENCH, '--json'], { encoding: 'utf8', maxBuffer: 1 << 24 });
  if (r.status !== 0) {
    process.stderr.write(r.stderr ?? '');
    throw new Error(`bench.mjs exited ${r.status} on process ${i + 1}`);
  }
  runs.push(JSON.parse(r.stdout.trim().split('\n').pop()));
  process.stderr.write(`process ${i + 1}/${PROCESSES} done\n`);
}

const head = runs[0];
console.log(`ArrowMetal ${head.version} on ${head.device}`);
console.log(`node ${head.node}, apache-arrow ${head.arrow}`);
console.log(
  `${head.rows.toLocaleString('en-US')} Int64 rows, ${PROCESSES} processes x (3 warm-ups + best of 5)`,
);

function table(title, key) {
  console.log(`\n${title}`);
  console.log('| Method | Min (ms) | Max (ms) | Median (ms) |');
  console.log('|---|---:|---:|---:|');
  const rows = head[key].map((r, i) => {
    const xs = runs.map((run) => run[key][i].ms).sort((a, b) => a - b);
    return { label: r.label, min: xs[0], max: xs[xs.length - 1], median: xs[(xs.length - 1) >> 1] };
  });
  for (const r of rows) {
    console.log(
      `| ${r.label} | ${r.min.toFixed(2)} | ${r.max.toFixed(2)} | ${r.median.toFixed(2)} |`,
    );
  }
  return rows;
}

const sum = table('sum over 10,000,000 Int64 rows', 'sum');
const filter = table('filter x >= 500 over 10,000,000 Int64 rows', 'filter');

// Any two rows whose [min, max] ranges overlap cannot be ordered by this measurement.
console.log('\nOverlapping ranges (these two cannot be ordered by this measurement):');
let any = false;
for (const rows of [sum, filter]) {
  for (let i = 0; i < rows.length; i++) {
    for (let j = i + 1; j < rows.length; j++) {
      if (rows[i].min <= rows[j].max && rows[j].min <= rows[i].max) {
        console.log(`  ${rows[i].label}  <->  ${rows[j].label}`);
        any = true;
      }
    }
  }
}
if (!any) console.log('  none');
