//! The same SQL on DataFusion 55.1 with the ArrowMetal rule off and on.
//!
//! Build and run (release, thin LTO; run the binary itself, not through `cargo run`, so no cargo
//! process is alive during the timed blocks):
//!
//! ```text
//! ARROWMETAL_LIB=<dylib> cargo build --release --example bench
//! ARROWMETAL_LIB=<dylib> target/release/examples/bench --sizes 1000000,10000000,50000000 \
//!     --out results/datafusion_rule_2026-09-27.csv --parquet-dir <dir with bench-<codec>-<rows>.parquet>
//! ```
//!
//! Per case and size:
//!
//! * Three `SessionContext`s over the same tables: **off** (DataFusion's defaults), **on** (the
//!   rule with `min_rows: 0, accept_inexact: true`, so it takes every node it can translate and
//!   the timing shows the crossover), and **default** (the rule's `ArrowMetalConfig::default()`,
//!   planned only: its decisions are recorded, not timed).
//! * `target_partitions` is DataFusion's default (the number of cores). Tables are `MemTable`s, in
//!   one of two layouts (`--modes`): `b8192` -- 8192-row RecordBatches dealt round-robin over the
//!   partitions (what DataFusion's scans produce); `part1` -- one batch per partition.
//! * Parquet cases register the files with `register_parquet` (DataFusion's own reader feeds both
//!   runs; ArrowMetal's GPU Parquet reader is not in this path).
//! * One untimed run each for off and on, whose results are compared (order-insensitive where SQL
//!   leaves the order open; ORDER BY keys row by row; floats within 1e-9 relative), then the best of
//!   `--iters` wall-clock runs (best of 2 when the untimed run took over `--slow-s` seconds), with the
//!   process CPU time (getrusage, user + system, all threads) of that best run. Timed: SQL to
//!   logical plan, physical planning (the rule runs there), and `collect`.
//! * `MetalExec`'s metrics from the best on-run: input wait, concat, import, plan run, export.
//! * Before each block (one family at one size and layout): wait until the 1-minute load is below
//!   `--max-load` and no cargo / rustc / swift-build / swift-frontend / swiftc / pytest runs,
//!   checking every 20 s. The load at the start and end of each block goes in every row.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{Array, ArrayRef, AsArray, Float64Array, Int32Array, Int64Array, StringArray};
use arrow::compute::{cast, concat_batches, lexsort_to_indices, take, SortColumn};
use arrow::datatypes::{DataType, Field, Float64Type, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::{collect, ExecutionPlan};
use datafusion::prelude::{ParquetReadOptions, SessionConfig};
use datafusion_arrowmetal::{session_context, ArrowMetalConfig, ArrowMetalRule, MetalExec};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const BATCH: usize = 8192;
const BUSY: &[&str] = &["cargo", "rustc", "swift-build", "swift-frontend", "swiftc", "pytest"];

// -------------------------------------------------------------------------------------------------
// arguments

struct Args {
    sizes: Vec<usize>,
    modes: Vec<String>,
    families: Vec<String>,
    cases: Vec<String>,
    iters: usize,
    slow_s: f64,
    max_load: f64,
    out: String,
    parquet_dir: Option<String>,
    parquet_rows: Vec<usize>,
    codecs: Vec<String>,
    /// Print each case's physical plan (off and on) and skip the timing.
    explain: bool,
}

fn args() -> Args {
    let mut a = Args {
        sizes: vec![1_000_000, 10_000_000, 50_000_000],
        modes: vec!["b8192".into(), "part1".into()],
        families: ["sort", "join", "groupby", "distinct", "filter", "parquet"].map(String::from).to_vec(),
        cases: vec![],
        iters: 5,
        slow_s: 2.0,
        max_load: 3.5,
        out: "results/datafusion_rule.csv".into(),
        parquet_dir: None,
        parquet_rows: vec![10_000_000, 50_000_000],
        codecs: vec!["snappy".into(), "zstd".into()],
        explain: false,
    };
    let v: Vec<String> = std::env::args().skip(1).collect();
    let list = |s: &str| s.split(',').filter(|x| !x.is_empty()).map(String::from).collect::<Vec<_>>();
    let nums = |s: &str| s.split(',').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect();
    let mut i = 0;
    while i < v.len() {
        let val = v.get(i + 1).cloned().unwrap_or_default();
        match v[i].as_str() {
            "--sizes" => a.sizes = nums(&val),
            "--modes" => a.modes = list(&val),
            "--families" => a.families = list(&val),
            "--cases" => a.cases = list(&val),
            "--iters" => a.iters = val.parse().unwrap(),
            "--slow-s" => a.slow_s = val.parse().unwrap(),
            "--max-load" => a.max_load = val.parse().unwrap(),
            "--out" => a.out = val,
            "--parquet-dir" => a.parquet_dir = Some(val),
            "--parquet-rows" => a.parquet_rows = nums(&val),
            "--codecs" => a.codecs = list(&val),
            "--explain" => {
                a.explain = true;
                i += 1;
                continue;
            }
            x => panic!("unknown argument {x}"),
        }
        i += 2;
    }
    a
}

// -------------------------------------------------------------------------------------------------
// quiet machine

fn load1() -> f64 {
    let mut l = [0f64; 3];
    unsafe { libc::getloadavg(l.as_mut_ptr(), 3) };
    l[0]
}

fn busy() -> Vec<&'static str> {
    BUSY.iter()
        .copied()
        .filter(|n| {
            let flag = if *n == "pytest" { "-f" } else { "-x" };
            std::process::Command::new("pgrep")
                .args([flag, n])
                .output()
                .map(|o| !o.stdout.is_empty())
                .unwrap_or(false)
        })
        .collect()
}

fn wait_quiet(max_load: f64, what: &str) -> f64 {
    let mut waited = 0;
    loop {
        let b = busy();
        let l = load1();
        if b.is_empty() && l < max_load {
            if waited > 0 {
                println!("  [quiet after {waited} s: load {l:.2}]");
            }
            return l;
        }
        if waited % 120 == 0 {
            println!("  [waiting before {what}: load {l:.2}, busy {b:?}]");
        }
        std::thread::sleep(Duration::from_secs(20));
        waited += 20;
    }
}

fn cpu_s() -> f64 {
    let mut r: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut r) };
    let tv = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
    tv(r.ru_utime) + tv(r.ru_stime)
}

// -------------------------------------------------------------------------------------------------
// data (the distributions of Benchmarks/datafusion_bench.py, with Float64 value columns)

fn i32s(rng: &mut StdRng, n: usize, hi: i32) -> ArrayRef {
    Arc::new(Int32Array::from_iter_values((0..n).map(|_| rng.random_range(0..hi))))
}
fn i64s(rng: &mut StdRng, n: usize, hi: i64) -> ArrayRef {
    Arc::new(Int64Array::from_iter_values((0..n).map(|_| rng.random_range(0..hi))))
}
fn f64s(rng: &mut StdRng, n: usize, scale: f64, shift: f64) -> ArrayRef {
    Arc::new(Float64Array::from_iter_values((0..n).map(|_| rng.random::<f64>() * scale + shift)))
}

fn batch(cols: Vec<(&str, ArrayRef)>) -> RecordBatch {
    let schema = Schema::new(
        cols.iter().map(|(n, a)| Field::new(*n, a.data_type().clone(), false)).collect::<Vec<_>>(),
    );
    RecordBatch::try_new(Arc::new(schema), cols.into_iter().map(|(_, a)| a).collect()).unwrap()
}

/// k1 int32 [0, 100k), k2 int32 [0, 1000), name utf8 (1000 values), x f64 [0, 1), q int64 [0, 1e9).
fn data_extra(rng: &mut StdRng, n: usize) -> RecordBatch {
    let k1 = i32s(rng, n, 100_000);
    let k2 = i32s(rng, n, 1000);
    let name: ArrayRef =
        Arc::new(StringArray::from_iter_values((0..n).map(|_| rng.random_range(0..1000).to_string())));
    let x = f64s(rng, n, 1.0, 0.0);
    let q = i64s(rng, n, 1_000_000_000);
    batch(vec![("k1", k1), ("k2", k2), ("name", name), ("x", x), ("q", q)])
}

/// region int32 [0, 200), sub int32 [0, 50), amount f64 [-500, 1500), qty int64 [-10, 40).
fn data_fact(rng: &mut StdRng, n: usize) -> RecordBatch {
    let region = i32s(rng, n, 200);
    let sub = i32s(rng, n, 50);
    let amount = f64s(rng, n, 2000.0, -500.0);
    let qty: ArrayRef = Arc::new(Int64Array::from_iter_values((0..n).map(|_| rng.random_range(-10..40i64))));
    batch(vec![("region", region), ("sub", sub), ("amount", amount), ("qty", qty)])
}

/// Probe keys from [0, 2 build); the build side holds every other value, so half the probe rows match.
fn data_join(rng: &mut StdRng, n: usize, build: usize) -> (RecordBatch, RecordBatch) {
    let probe = batch(vec![("k", i64s(rng, n, 2 * build as i64)), ("v", f64s(rng, n, 100.0, 0.0))]);
    let bk: ArrayRef = Arc::new(Int64Array::from_iter_values((0..build as i64).map(|i| 2 * i)));
    let b = batch(vec![("k", bk), ("w", f64s(rng, build, 2.0, 0.0))]);
    (probe, b)
}

/// One key from [0, G); two keys from [0, G/b) x [0, b), b = min(G, 100); x f64 [0, 1).
fn data_grid(rng: &mut StdRng, n: usize, g: usize) -> RecordBatch {
    let b = g.min(100);
    let k = i32s(rng, n, g as i32);
    let k1 = i32s(rng, n, (g / b).max(1) as i32);
    let k2 = i32s(rng, n, b as i32);
    let x = f64s(rng, n, 1.0, 0.0);
    batch(vec![("k", k), ("k1", k1), ("k2", k2), ("x", x)])
}

// -------------------------------------------------------------------------------------------------
// cases

#[derive(Clone)]
enum Order {
    /// Any row order.
    Any,
    /// ORDER BY these output columns (keys row by row, rows as a multiset within tied keys).
    Keys(Vec<usize>),
    /// ORDER BY ... LIMIT: keys row by row; the rows of the last tie run may differ.
    KeysLimit(Vec<usize>),
}

struct Case {
    id: String,
    label: String,
    sql: String,
    order: Order,
}

fn case(id: &str, label: &str, sql: &str, order: Order) -> Case {
    Case { id: id.into(), label: label.into(), sql: sql.into(), order }
}

enum Source {
    Mem(Vec<(&'static str, RecordBatch)>),
    Parquet(String),
}

struct Block {
    family: &'static str,
    source: Source,
    cases: Vec<Case>,
}

fn gname(g: usize, rows: usize) -> String {
    match g {
        200 => "200".into(),
        10_000 => "10k".into(),
        100_000 => "100k".into(),
        1_000_000 => "1M".into(),
        10_000_000 => "10M".into(),
        _ if g == rows / 2 => "R2".into(),
        _ => g.to_string(),
    }
}

/// The blocks of one size, generated lazily (only one block's tables alive at a time).
fn blocks(rows: usize, fam: &str, rng: &mut StdRng) -> Vec<Box<dyn FnOnce(&mut StdRng) -> Block>> {
    let mut out: Vec<Box<dyn FnOnce(&mut StdRng) -> Block>> = Vec::new();
    match fam {
        "sort" => out.push(Box::new(move |rng| Block {
            family: "sort",
            source: Source::Mem(vec![("extra", data_extra(rng, rows))]),
            cases: vec![
                case("srt_i64", "sort 3 columns by an int64 key", "SELECT q, k1, x FROM extra ORDER BY q", Order::Keys(vec![0])),
                case("srt_f64", "sort 3 columns by a Float64 key", "SELECT q, k1, x FROM extra ORDER BY x", Order::Keys(vec![2])),
                case("srt_str", "sort 3 columns by a String key (1000 values)", "SELECT name, q, k1 FROM extra ORDER BY name", Order::Keys(vec![0])),
                case("top_i64", "top 100 by an int64 key (control)", "SELECT q, k1, x FROM extra ORDER BY q LIMIT 100", Order::KeysLimit(vec![0])),
                case("top_f64", "top 100 by Float64, descending (control)", "SELECT q, k1, x FROM extra ORDER BY x DESC LIMIT 100", Order::KeysLimit(vec![2])),
            ],
        })),
        "join" => {
            for b in [10_000usize, 1_000_000, 10_000_000] {
                out.push(Box::new(move |rng| {
                    let (probe, build) = data_join(rng, rows, b);
                    let bn = gname(b, 0);
                    let mut cases = Vec::new();
                    for (how, sql_how) in [("inner", "JOIN"), ("left", "LEFT JOIN")] {
                        cases.push(case(
                            &format!("j_{how}_{bn}"),
                            &format!("{how} join, {bn}-row build, count + sum over the whole result"),
                            &format!("SELECT count(*) AS n, sum(b.w) AS s FROM probe p {sql_how} build b ON p.k = b.k"),
                            Order::Any,
                        ));
                        cases.push(case(
                            &format!("jg_{how}_{bn}"),
                            &format!("{how} join, {bn}-row build, then group by the probe key: count + sum"),
                            &format!("SELECT p.k, count(*) AS n, sum(p.v) AS s FROM probe p {sql_how} build b ON p.k = b.k GROUP BY p.k"),
                            Order::Any,
                        ));
                    }
                    Block { family: "join", source: Source::Mem(vec![("probe", probe), ("build", build)]), cases }
                }));
            }
        }
        "groupby" => {
            let mut gs = vec![200usize, 10_000, 100_000, 1_000_000, rows / 2];
            gs.dedup();
            for g in gs {
                out.push(Box::new(move |rng| {
                    let t = data_grid(rng, rows, g);
                    let gn = gname(g, rows);
                    let mut cases = Vec::new();
                    for (fam, agg) in [
                        ("count", "count(*) AS n"),
                        ("sum", "sum(x) AS s"),
                        ("mean", "avg(x) AS m"),
                        ("minmax", "min(x) AS lo, max(x) AS hi"),
                    ] {
                        cases.push(case(
                            &format!("g1{fam}{gn}"),
                            &format!("group-by 1 int32 key, {gn} groups, {agg}"),
                            &format!("SELECT k, {agg} FROM grid GROUP BY k"),
                            Order::Any,
                        ));
                        cases.push(case(
                            &format!("g2{fam}{gn}"),
                            &format!("group-by 2 int32 keys, {gn} groups, {agg}"),
                            &format!("SELECT k1, k2, {agg} FROM grid GROUP BY k1, k2"),
                            Order::Any,
                        ));
                    }
                    Block { family: "groupby", source: Source::Mem(vec![("grid", t)]), cases }
                }));
            }
        }
        "distinct" => out.push(Box::new(move |rng| Block {
            family: "distinct",
            source: Source::Mem(vec![("fact", data_fact(rng, rows))]),
            cases: vec![case("u_small", "distinct (int32, int32), 10k groups", "SELECT DISTINCT region, sub FROM fact", Order::Any)],
        })),
        "filter" => out.push(Box::new(move |rng| Block {
            family: "filter",
            source: Source::Mem(vec![("fact", data_fact(rng, rows))]),
            cases: vec![case(
                "a_sumcnt",
                "filtered sum + count over the whole table (control)",
                "SELECT sum(amount) AS total, count(*) AS n FROM fact WHERE region < 20 AND qty > 10",
                Order::Any,
            )],
        })),
        _ => {}
    }
    let _ = rng;
    out
}

fn parquet_cases() -> Vec<Case> {
    vec![
        case("p_sort", "Parquet scan, sort 2 columns by a Float64 key (s4)", "SELECT id, price FROM f ORDER BY price", Order::Keys(vec![1])),
        case("p_group", "Parquet scan, group-by int32 code (100k groups), sum + count", "SELECT code, sum(price) AS p, count(*) AS n FROM f GROUP BY code", Order::Any),
        case("p_filtgroup", "Parquet scan, filter, group-by qty (1000 keys), sum + count (s1)", "SELECT qty, sum(weight) AS w, count(*) AS n FROM f WHERE price > 500.0 GROUP BY qty", Order::Any),
    ]
}

// -------------------------------------------------------------------------------------------------
// contexts

fn layout(b: &RecordBatch, mode: &str, parts: usize) -> Vec<Vec<RecordBatch>> {
    let n = b.num_rows();
    let mut out = vec![Vec::new(); parts];
    match mode {
        "b8192" => {
            let mut i = 0;
            let mut off = 0;
            while off < n {
                let len = BATCH.min(n - off);
                out[i % parts].push(b.slice(off, len));
                off += len;
                i += 1;
            }
        }
        "part1" => {
            let per = n.div_ceil(parts);
            for (p, slot) in out.iter_mut().enumerate() {
                let off = p * per;
                if off < n {
                    slot.push(b.slice(off, per.min(n - off)));
                }
            }
        }
        m => panic!("unknown mode {m}"),
    }
    out
}

async fn context(source: &Source, mode: &str, rule: Option<ArrowMetalRule>) -> SessionContext {
    let config = SessionConfig::new();
    let parts = config.target_partitions();
    let ctx = match rule {
        Some(r) => session_context(config, r),
        None => SessionContext::new_with_config(config),
    };
    match source {
        Source::Mem(tables) => {
            for (name, b) in tables {
                let t = MemTable::try_new(b.schema(), layout(b, mode, parts)).unwrap();
                ctx.register_table(*name, Arc::new(t)).unwrap();
            }
        }
        Source::Parquet(path) => {
            ctx.register_parquet("f", path, ParquetReadOptions::default()).await.unwrap();
        }
    }
    ctx
}

struct Run {
    out: Vec<RecordBatch>,
    wall: f64,
    cpu: f64,
    metal: BTreeMap<&'static str, f64>,
}

const METRICS: &[&str] =
    &["input_time", "concat_time", "import_time", "kernel_time", "export_time", "input_batches"];

fn metal_metrics(plan: &Arc<dyn ExecutionPlan>, acc: &mut BTreeMap<&'static str, f64>) {
    if let Some(m) = plan.downcast_ref::<MetalExec>() {
        *acc.entry("nodes").or_default() += 1.0;
        if let Some(ms) = m.metrics() {
            for &k in METRICS {
                if let Some(v) = ms.sum_by_name(k) {
                    let scale = if k == "input_batches" { 1.0 } else { 1e-6 };
                    *acc.entry(k).or_default() += v.as_usize() as f64 * scale;
                }
            }
        }
    }
    for c in plan.children() {
        metal_metrics(c, acc);
    }
}

async fn run(ctx: &SessionContext, sql: &str) -> Run {
    let c0 = cpu_s();
    let t0 = Instant::now();
    let df = ctx.sql(sql).await.unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    let out = collect(Arc::clone(&plan), ctx.task_ctx()).await.unwrap();
    let wall = t0.elapsed().as_secs_f64() * 1e3;
    let cpu = (cpu_s() - c0) * 1e3;
    let mut metal = BTreeMap::new();
    metal_metrics(&plan, &mut metal);
    Run { out, wall, cpu, metal }
}

async fn best(ctx: &SessionContext, sql: &str, first_ms: f64, a: &Args) -> (Run, usize) {
    let n = if first_ms > a.slow_s * 1e3 { a.iters.min(2) } else { a.iters };
    let mut best: Option<Run> = None;
    for _ in 0..n {
        let r = run(ctx, sql).await;
        let r = Run { out: Vec::new(), ..r };
        if best.as_ref().is_none_or(|b| r.wall < b.wall) {
            best = Some(r);
        }
    }
    (best.unwrap(), n)
}

// -------------------------------------------------------------------------------------------------
// result equality

fn one(schema: SchemaRef, b: &[RecordBatch]) -> RecordBatch {
    concat_batches(&schema, b).unwrap()
}

fn canon(b: &RecordBatch) -> RecordBatch {
    if b.num_rows() <= 1 {
        return b.clone();
    }
    let cols: Vec<SortColumn> =
        b.columns().iter().map(|c| SortColumn { values: Arc::clone(c), options: None }).collect();
    let idx = lexsort_to_indices(&cols, None).unwrap();
    let cols = b.columns().iter().map(|c| take(c.as_ref(), &idx, None).unwrap()).collect();
    RecordBatch::try_new(b.schema(), cols).unwrap()
}

/// Equal columns: floats within 1e-9 relative (NaN = NaN, null = null), everything else exact.
fn col_eq(a: &ArrayRef, b: &ArrayRef, dev: &mut f64) -> Result<(), String> {
    if a.data_type() != b.data_type() {
        return Err(format!("type {} vs {}", a.data_type(), b.data_type()));
    }
    if matches!(a.data_type(), DataType::Float64 | DataType::Float32) {
        let a = cast(a, &DataType::Float64).unwrap();
        let b = cast(b, &DataType::Float64).unwrap();
        let (a, b) = (a.as_primitive::<Float64Type>(), b.as_primitive::<Float64Type>());
        for i in 0..a.len() {
            match (a.is_null(i), b.is_null(i)) {
                (true, true) => continue,
                (false, false) => {}
                _ => return Err(format!("row {i}: null vs value")),
            }
            let (x, y) = (a.value(i), b.value(i));
            if x.is_nan() && y.is_nan() {
                continue;
            }
            let d = (x - y).abs() / 1f64.max(x.abs()).max(y.abs());
            *dev = dev.max(d);
            if d > 1e-9 {
                return Err(format!("row {i}: {x} vs {y}"));
            }
        }
        return Ok(());
    }
    if a.to_data() != b.to_data() {
        return Err("values differ".into());
    }
    Ok(())
}

fn batch_eq(a: &RecordBatch, b: &RecordBatch, dev: &mut f64) -> Result<(), String> {
    if a.num_columns() != b.num_columns() || a.num_rows() != b.num_rows() {
        return Err(format!("shape {}x{} vs {}x{}", a.num_rows(), a.num_columns(), b.num_rows(), b.num_columns()));
    }
    for i in 0..a.num_columns() {
        col_eq(a.column(i), b.column(i), dev).map_err(|e| format!("column {i}: {e}"))?;
    }
    Ok(())
}

/// Off vs on. Returns the largest relative float deviation seen.
fn same(off: &[RecordBatch], on: &[RecordBatch], order: &Order) -> Result<f64, String> {
    let schema = match (off.first(), on.first()) {
        (Some(b), _) | (None, Some(b)) => b.schema(),
        (None, None) => return Ok(0.0),
    };
    let (a, b) = (one(Arc::clone(&schema), off), one(schema, on));
    if a.num_rows() != b.num_rows() {
        return Err(format!("rows {} vs {}", a.num_rows(), b.num_rows()));
    }
    let mut dev = 0.0;
    match order {
        Order::Any => batch_eq(&canon(&a), &canon(&b), &mut dev)?,
        Order::Keys(keys) => {
            for &k in keys {
                col_eq(a.column(k), b.column(k), &mut dev).map_err(|e| format!("key column {k}: {e}"))?;
            }
            batch_eq(&canon(&a), &canon(&b), &mut dev)?;
        }
        Order::KeysLimit(keys) => {
            for &k in keys {
                col_eq(a.column(k), b.column(k), &mut dev).map_err(|e| format!("key column {k}: {e}"))?;
            }
            // Rows tied with the last key may be any of the tied rows: compare the rows before it.
            let n = a.num_rows();
            let mut start = n;
            if n > 0 {
                let last: Vec<ArrayRef> = keys.iter().map(|&k| a.column(k).slice(n - 1, 1)).collect();
                start = n - 1;
                while start > 0 {
                    let here: Vec<ArrayRef> = keys.iter().map(|&k| a.column(k).slice(start - 1, 1)).collect();
                    if here.iter().zip(&last).all(|(x, y)| x.to_data() == y.to_data()) {
                        start -= 1;
                    } else {
                        break;
                    }
                }
            }
            batch_eq(&canon(&a.slice(0, start)), &canon(&b.slice(0, start)), &mut dev)?;
        }
    }
    Ok(dev)
}

// -------------------------------------------------------------------------------------------------
// the report column

fn decisions(rule: &ArrowMetalRule) -> (usize, String) {
    let r = rule.report();
    let taken = r.taken().count();
    let s: Vec<String> = r
        .decisions()
        .iter()
        .map(|d| {
            let node = d.node.split(':').next().unwrap_or("").trim();
            let tag = if d.runtime_fallback { "FALLBACK" } else if d.taken { "TAKEN" } else { "LEFT" };
            let mut reason = d.reason.clone();
            if let Some(i) = reason.find("; output re-partitioned") {
                reason.truncate(i);
                reason.push_str("; re-partitioned");
            }
            format!("{tag} {node} ({reason})")
        })
        .collect();
    (taken, if s.is_empty() { "no candidate node".into() } else { s.join(" | ") })
}

fn csv(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

// -------------------------------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let a = args();
    if let Some(dir) = std::path::Path::new(&a.out).parent() {
        std::fs::create_dir_all(dir).ok();
    }
    let new_file = !std::path::Path::new(&a.out).exists();
    let mut out = std::fs::OpenOptions::new().create(true).append(true).open(&a.out).unwrap();
    use std::io::Write;
    if new_file {
        writeln!(
            out,
            "size,layout,family,case,label,sql,off_ms,on_ms,ratio,off_cpu_ms,on_cpu_ms,iters_off,iters_on,\
             equal,max_rel_dev,nodes_taken,rule_on,rule_default,runtime_fallbacks,metal_nodes,input_ms,concat_ms,\
             import_ms,kernel_ms,export_ms,input_batches,out_rows,block_load_start,block_load_end"
        )
        .unwrap();
    }
    let parts = SessionConfig::new().target_partitions();
    println!("target_partitions {parts}; iters {}; slow {} s; max load {}", a.iters, a.slow_s, a.max_load);

    // (size, layout, family, block builder)
    let mut work: Vec<(usize, String, Box<dyn FnOnce(&mut StdRng) -> Block>)> = Vec::new();
    for &rows in &a.sizes {
        for mode in &a.modes {
            for fam in &a.families {
                if fam == "parquet" {
                    continue;
                }
                let mut dummy = StdRng::seed_from_u64(0);
                for b in blocks(rows, fam, &mut dummy) {
                    work.push((rows, mode.clone(), b));
                }
            }
        }
        if a.families.iter().any(|f| f == "parquet") && a.parquet_rows.contains(&rows) {
            if let Some(dir) = &a.parquet_dir {
                for codec in &a.codecs {
                    let path = format!("{dir}/bench-{codec}-{rows}.parquet");
                    if !std::path::Path::new(&path).exists() {
                        println!("  [missing {path}]");
                        continue;
                    }
                    let c = codec.clone();
                    work.push((
                        rows,
                        format!("parquet-{c}"),
                        Box::new(move |_| Block { family: "parquet", source: Source::Parquet(path), cases: parquet_cases() }),
                    ));
                }
            }
        }
    }

    for (rows, mode, build) in work {
        let mut rng = StdRng::seed_from_u64(1234);
        let t = Instant::now();
        let block = build(&mut rng);
        let cases: Vec<&Case> =
            block.cases.iter().filter(|c| a.cases.is_empty() || a.cases.contains(&c.id)).collect();
        if cases.is_empty() {
            continue;
        }
        println!("== {} rows, {}, {} ({} cases; data {:.1} s)", rows, mode, block.family, cases.len(), t.elapsed().as_secs_f64());
        let l0 = wait_quiet(a.max_load, &format!("{} {rows} {mode}", block.family));
        println!("  [load {l0:.2} at block start]");
        let mem_mode = if mode.starts_with("parquet") { "b8192" } else { mode.as_str() };
        let rule_on = ArrowMetalRule::new(ArrowMetalConfig { min_rows: 0, accept_inexact: true, ..Default::default() });
        let rule_def = ArrowMetalRule::new(ArrowMetalConfig::default());
        let ctx_off = context(&block.source, mem_mode, None).await;
        let ctx_on = context(&block.source, mem_mode, Some(rule_on.clone())).await;
        let ctx_def = context(&block.source, mem_mode, Some(rule_def.clone())).await;
        let mut rows_out = Vec::new();
        for c in cases {
            // Decisions under the default config (planned, not run).
            rule_def.clear_report();
            let _ = ctx_def.sql(&c.sql).await.unwrap().create_physical_plan().await.unwrap();
            let (_, def_s) = decisions(&rule_def);

            if a.explain {
                for (tag, ctx) in [("off", &ctx_off), ("on", &ctx_on)] {
                    let plan = ctx.sql(&c.sql).await.unwrap().create_physical_plan().await.unwrap();
                    let shown = datafusion::physical_plan::displayable(plan.as_ref()).indent(false).to_string();
                    println!("--- {} {tag}\n{shown}", c.id);
                }
                println!("    default: {def_s}");
                continue;
            }
            let off0 = run(&ctx_off, &c.sql).await;
            rule_on.clear_report();
            let on0 = run(&ctx_on, &c.sql).await;
            let (taken, on_s) = decisions(&rule_on);
            let (equal, dev) = match same(&off0.out, &on0.out, &c.order) {
                Ok(d) => ("yes".to_string(), d),
                Err(e) => {
                    println!("  !! {} {}: results differ: {e}", c.id, rows);
                    (format!("NO: {e}"), f64::NAN)
                }
            };
            let out_rows: usize = off0.out.iter().map(|b| b.num_rows()).sum();
            let (off_first, on_first) = (off0.wall, on0.wall);
            drop(off0);
            drop(on0);
            let (off, n_off) = best(&ctx_off, &c.sql, off_first, &a).await;
            rule_on.clear_report();
            let (on, n_on) = best(&ctx_on, &c.sql, on_first, &a).await;
            let fallbacks = rule_on.report().runtime_fallbacks().count();
            let m = |k: &str| on.metal.get(k).copied().unwrap_or(0.0);
            let ratio = off.wall / on.wall;
            let mark = if taken == 0 { "  (rule took nothing)" } else { "" };
            println!(
                "  {:14} off {:9.1} ms  on {:9.1} ms  {:6.2}x  cpu {:8.0}/{:8.0}  concat {:6.1} import {:6.1} kernel {:7.1}{}",
                c.id, off.wall, on.wall, ratio, off.cpu, on.cpu, m("concat_time"), m("import_time"), m("kernel_time"), mark
            );
            rows_out.push(format!(
                "{rows},{mode},{},{},{},{},{:.2},{:.2},{:.3},{:.1},{:.1},{n_off},{n_on},{},{:.3e},{taken},{},{},{fallbacks},{},{:.2},{:.2},{:.2},{:.2},{:.2},{},{out_rows}",
                block.family,
                c.id,
                csv(&c.label),
                csv(&c.sql),
                off.wall,
                on.wall,
                ratio,
                off.cpu,
                on.cpu,
                csv(&equal),
                dev,
                csv(&on_s),
                csv(&def_s),
                m("nodes"),
                m("input_time"),
                m("concat_time"),
                m("import_time"),
                m("kernel_time"),
                m("export_time"),
                m("input_batches"),
            ));
        }
        let l1 = load1();
        println!("  [load {l1:.2} at block end]");
        for r in rows_out {
            writeln!(out, "{r},{l0:.2},{l1:.2}").unwrap();
        }
        out.flush().unwrap();
    }
}
