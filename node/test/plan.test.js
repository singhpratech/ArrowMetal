// The JSON plan runner (am_plan_run), against the same answer computed step by step in JS.
const test = require('node:test');
const assert = require('node:assert/strict');
const A = require('apache-arrow');
const { MetalArray, PlanSource, runPlan, explainPlan } = require('../dist/index.js');

const REGIONS = ['east', 'west', 'east', 'west', 'north', 'east'];
const AMOUNTS = [100n, 250n, 300n, 50n, 400n, 20n];

function source() {
  return PlanSource.create('sales', {
    region: MetalArray.fromArrow(A.vectorFromArray(REGIONS, new A.Utf8())),
    amount: MetalArray.fromArrow(A.vectorFromArray(AMOUNTS, new A.Int64())),
  });
}

const PLAN = {
  op: 'sort',
  by: [['total', true]],
  input: {
    op: 'group_by',
    keys: [['region', '(col "region")']],
    aggs: [['sum', 'total', '(col "amount")']],
    input: {
      op: 'filter',
      predicate: '(gt (col "amount") (int 60))',
      input: { op: 'scan', source: 'sales' },
    },
  },
};

test('a filter/group_by/sort plan matches the same steps in JS', () => {
  const r = runPlan(PLAN, [source()]);
  assert.deepEqual(r.names, ['region', 'total']);

  const want = new Map();
  REGIONS.forEach((k, i) => {
    if (AMOUNTS[i] > 60n) want.set(k, (want.get(k) ?? 0n) + AMOUNTS[i]);
  });
  const expected = [...want.entries()].sort((a, b) => (b[1] > a[1] ? 1 : b[1] < a[1] ? -1 : 0));

  const gotRegions = [...r.column('region').toArrow()];
  const gotTotals = [...r.column('total').toArrow()];
  assert.equal(r.rows, expected.length);
  assert.deepEqual(gotTotals, expected.map((e) => e[1]));
  assert.deepEqual(new Set(gotRegions), new Set(expected.map((e) => e[0])));
});

test('the same plan run unoptimized gives the same answer', () => {
  const a = runPlan(PLAN, [source()], { optimize: true });
  const b = runPlan(PLAN, [source()], { optimize: false });
  assert.deepEqual([...a.column('total').toArrow()], [...b.column('total').toArrow()]);
});

test('a plan can be handed in as a JSON string', () => {
  const r = runPlan(JSON.stringify(PLAN), [source()]);
  assert.equal(r.rows, 3);
});

test('explain returns the logical and physical plans', () => {
  const text = explainPlan(PLAN, [source()]);
  assert.match(text, /LOGICAL PLAN/);
  assert.match(text, /PHYSICAL PLAN/);
  assert.match(text, /SCAN sales/);
});

test('a plan that does not type-check throws with the engine message', () => {
  assert.throws(() => runPlan({ op: 'scan', source: 'nosuchtable' }, [source()]),
    /unknown source "nosuchtable"/);
  assert.throws(() => runPlan({ op: 'filter', predicate: '(gt (col "nope") (int 1))',
    input: { op: 'scan', source: 'sales' } }, [source()]), /nope/);
});

test('asking for a column that is not in the result names it, by name or by index', () => {
  const r = runPlan(PLAN, [source()]);
  assert.throws(() => r.column('nope'), /no column "nope" in the result; it has 2 columns: region, total/);
  assert.throws(() => r.column(99), /no column 99 in the result; it has 2 columns/);
  assert.throws(() => r.column(-1), /no column -1 in the result/);
  // And never a stale message from an earlier failure.
  assert.throws(() => r.column(99), (e) => !/unknown source/.test(e.message));
});

test('a select plan with an expression', () => {
  const r = runPlan(
    {
      op: 'select',
      exprs: [['doubled', '(mul (col "amount") (int 2))']],
      input: { op: 'scan', source: 'sales' },
    },
    [source()],
  );
  assert.deepEqual([...r.column('doubled').toArrow()], AMOUNTS.map((a) => a * 2n));
});
