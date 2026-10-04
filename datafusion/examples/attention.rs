//! Single-head attention written as one SQL query over coordinate-format matrices, on DataFusion 55.1
//! alone and with the ArrowMetal rule.
//!
//! The query (`ATTENTION`, copied verbatim) and the tables' value function are those of the
//! attention test of the ddx project (`crates/ddx-datafusion/tests/attention.rs`,
//! <https://github.com/xqlsystems/ddx>, Apache-2.0, query and tables by Alexander Merose and the ddx
//! authors): every matrix is a table of rows `(i BIGINT, j BIGINT, val DOUBLE)` -- `x[t, d]`,
//! `wq`/`wk`/`wv[d, e]`, `tgt[t, e]` -- and every matrix product is `SUM(a.val * b.val)` over a
//! `JOIN ... GROUP BY`. Q, K and V are `t * d * e` joined rows each before their grouped sums; the
//! scores `s` and the output `o` are `t * t * e` each.
//!
//! Build and run (release, thin LTO; run the binary itself, so no cargo process is alive during
//! the timed blocks):
//!
//! ```text
//! ARROWMETAL_LIB=<dylib> cargo build --release --example attention
//! ARROWMETAL_LIB=<dylib> target/release/examples/attention --shapes 256x128x32,512x256x64 \
//!     --out results/attention_2026-10-03.csv
//! ```
//!
//! Per shape `t x d x e`:
//!
//! * Configurations (`--configs`, default `a,b,c`): **a** `SessionContext::new()` (DataFusion
//!   alone); **b** `session_context(SessionConfig::new(), rule)` with `ArrowMetalConfig::default()`;
//!   **c** the rule forced: `ArrowMetalConfig::all().with_min_rows(0)` with
//!   `AggregateChoice::ArrowMetal` and `JoinChoice::ArrowMetal`; **d** (on request) **c** with
//!   `accept_inexact(true)`, the benchmark's `on` context.
//! * Tables: `MemTable`s of 8,192-row batches dealt round-robin over `target_partitions`.
//! * Each configuration's first run (untimed; the first GPU query of the process also compiles
//!   pipelines) gives the loss compared with **a**'s and the rule's report for one plan (nodes
//!   taken and left at plan time, run-time choices, runtime fallbacks).
//! * Timed: `--rounds` rounds (default 5); in each, every configuration in an order rotated by one
//!   per round runs untimed until `--warm-ms` (default 100) of wall time, then once timed. Best and
//!   median wall ms over the rounds, with the process CPU time (getrusage, user + system, all
//!   threads) of each run: the CPU ms of the best run and the median CPU ms. Timed: SQL to logical
//!   plan, physical planning and `collect`.
//! * Then `--idle-reps` (default 3) runs of each configuration after `--idle-ms` (default 500) of
//!   sleep, configurations alternating; the median and the maximum.
//! * A configuration whose first run, times the runs the timing needs, exceeds `--max-config-s`
//!   (default 300) is not timed for that shape; its row says so.
//! * After every shape is timed: the row counts of the joins and grouped sums of q, s and o,
//!   counted with SQL on DataFusion alone (untimed).
//! * Before each shape's timed block: wait until the 1-minute load is below `--max-load` (3.5), no
//!   cargo / rustc / swift / pytest process runs, and (with `--lock-dir`) no `BUILDING*` or
//!   `TIMING*` file exists; then hold `<lock-dir>/TIMING.<name>` (`BENCH_LANE`, default `attention`) for
//!   the block. Load and time at the block's start and end and the last `Sleep` entry of
//!   `pmset -g log` go in every row.

use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{Array, ArrayRef, Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::collect;
use datafusion::prelude::SessionConfig;
use datafusion_arrowmetal::{session_context, AggregateChoice, ArrowMetalConfig, ArrowMetalRule, JoinChoice};

const BATCH: usize = 8192;
const BUSY: &[&str] = &["cargo", "rustc", "swift-build", "swift-frontend", "swiftc", "pytest"];

/// The query of the ddx attention test, verbatim.
pub const ATTENTION: &str = "\
WITH q AS (SELECT x.t, w.e, SUM(x.val * w.val) AS val \
           FROM x JOIN wq w ON x.d = w.d GROUP BY x.t, w.e), \
     k AS (SELECT x.t, w.e, SUM(x.val * w.val) AS val \
           FROM x JOIN wk w ON x.d = w.d GROUP BY x.t, w.e), \
     v AS (SELECT x.t, w.e, SUM(x.val * w.val) AS val \
           FROM x JOIN wv w ON x.d = w.d GROUP BY x.t, w.e), \
     s AS (SELECT q.t, k.t AS u, SUM(q.val * k.val) * 0.7071067811865476 AS val \
           FROM q JOIN k ON q.e = k.e GROUP BY q.t, k.t), \
     m AS (SELECT t, MAX(val) AS m FROM s GROUP BY t), \
     ex AS (SELECT s.t, s.u, exp(s.val - m.m) AS val \
            FROM s JOIN m ON s.t = m.t), \
     z AS (SELECT t, SUM(val) AS val FROM ex GROUP BY t), \
     a AS (SELECT ex.t, ex.u, ex.val / z.val AS val FROM ex JOIN z ON ex.t = z.t), \
     o AS (SELECT a.t, v.e, SUM(a.val * v.val) AS val \
           FROM a JOIN v ON a.u = v.t GROUP BY a.t, v.e) \
SELECT SUM(0.5 * power(o.val - tgt.val, 2)) AS loss \
FROM o JOIN tgt ON o.t = tgt.t AND o.e = tgt.e";

/// The CTEs of `ATTENTION` up to `o` (the part before the final SELECT), for the row counts.
fn ctes() -> &'static str {
    let at = ATTENTION.find("SELECT SUM(0.5").unwrap();
    ATTENTION[..at].trim_end()
}

// -------------------------------------------------------------------------------------------------
// tables (the ddx test's value function and seeds)

fn value(seed: usize) -> f64 {
    ((seed * 7919 + 13) % 1000) as f64 / 1000.0 - 0.5
}

fn matrix(dims: [&str; 2], shape: [usize; 2], seed: usize) -> RecordBatch {
    let n = shape[0] * shape[1];
    let (mut is, mut js, mut vs) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            is.push(i as i64);
            js.push(j as i64);
            vs.push(value(seed + i * shape[1] + j));
        }
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new(dims[0], DataType::Int64, false),
        Field::new(dims[1], DataType::Int64, false),
        Field::new("val", DataType::Float64, false),
    ]));
    let cols: Vec<ArrayRef> =
        vec![Arc::new(Int64Array::from(is)), Arc::new(Int64Array::from(js)), Arc::new(Float64Array::from(vs))];
    RecordBatch::try_new(schema, cols).unwrap()
}

fn tables(t: usize, d: usize, e: usize) -> Vec<(&'static str, RecordBatch)> {
    vec![
        ("x", matrix(["t", "d"], [t, d], 1)),
        ("wq", matrix(["d", "e"], [d, e], 20)),
        ("wk", matrix(["d", "e"], [d, e], 40)),
        ("wv", matrix(["d", "e"], [d, e], 60)),
        ("tgt", matrix(["t", "e"], [t, e], 80)),
    ]
}

/// 8,192-row batches dealt round-robin over the partitions (the benchmark's `b8192` layout).
fn layout(b: &RecordBatch, parts: usize) -> Vec<Vec<RecordBatch>> {
    let n = b.num_rows();
    let mut out = vec![Vec::new(); parts];
    let (mut i, mut off) = (0, 0);
    while off < n {
        let len = BATCH.min(n - off);
        out[i % parts].push(b.slice(off, len));
        off += len;
        i += 1;
    }
    out
}

// -------------------------------------------------------------------------------------------------
// configurations

fn rule_for(name: &str) -> Option<ArrowMetalRule> {
    let forced = ArrowMetalConfig::all()
        .with_min_rows(0)
        .with_aggregate_choice(AggregateChoice::ArrowMetal)
        .with_join_choice(JoinChoice::ArrowMetal);
    match name {
        "a" => None,
        "b" => Some(ArrowMetalRule::new(ArrowMetalConfig::default())),
        "c" => Some(ArrowMetalRule::new(forced)),
        "d" => Some(ArrowMetalRule::new(forced.with_accept_inexact(true))),
        c => panic!("unknown configuration {c}"),
    }
}

fn describe(name: &str) -> &'static str {
    match name {
        "a" => "SessionContext::new() (DataFusion alone)",
        "b" => "session_context(SessionConfig::new(), ArrowMetalRule::new(ArrowMetalConfig::default()))",
        "c" => "ArrowMetalConfig::all().with_min_rows(0), AggregateChoice::ArrowMetal, JoinChoice::ArrowMetal",
        "d" => "as c, with_accept_inexact(true)",
        _ => "",
    }
}

fn context(name: &str, rule: Option<ArrowMetalRule>, data: &[(&'static str, RecordBatch)]) -> SessionContext {
    let ctx = match (name, rule) {
        ("a", _) => SessionContext::new(),
        (_, Some(r)) => session_context(SessionConfig::new(), r),
        (n, None) => panic!("configuration {n} needs a rule"),
    };
    let parts = ctx.copied_config().target_partitions();
    for (tname, b) in data {
        let t = MemTable::try_new(b.schema(), layout(b, parts)).unwrap();
        ctx.register_table(*tname, Arc::new(t)).unwrap();
    }
    ctx
}

// -------------------------------------------------------------------------------------------------
// quiet machine, locks

fn load1() -> f64 {
    let mut l = [0f64; 3];
    // SAFETY: `l` holds the 3 doubles getloadavg writes at most.
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

fn sh(cmd: &str) -> String {
    std::process::Command::new("sh")
        .args(["-c", cmd])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

fn now_text() -> String {
    sh("date '+%Y-%m-%d %H:%M:%S'")
}

fn last_sleep() -> String {
    let l = sh("pmset -g log | grep ' Sleep  ' | tail -1");
    l.split_whitespace().take(2).collect::<Vec<_>>().join(" ")
}

fn lock_name() -> String {
    std::env::var("BENCH_LANE").unwrap_or_else(|_| "attention".into())
}

/// Any `BUILDING*` or `TIMING*` file in the lock dir other than `own`.
fn others(d: &std::path::Path, own: &str) -> Vec<String> {
    std::fs::read_dir(d)
        .map(|it| {
            it.flatten()
                .map(|e| e.file_name().to_string_lossy().to_string())
                .filter(|n| (n.starts_with("BUILDING") || n.starts_with("TIMING")) && n != own)
                .collect()
        })
        .unwrap_or_default()
}

/// Holds `<dir>/TIMING.<name>` for one block; removed on drop while it holds this process's tag.
struct TimingLock(Option<std::path::PathBuf>, String);

impl Drop for TimingLock {
    fn drop(&mut self) {
        if let Some(p) = &self.0 {
            if std::fs::read_to_string(p).map(|s| s.trim_end() == self.1).unwrap_or(false) {
                std::fs::remove_file(p).ok();
            }
        }
    }
}

fn acquire(a: &Args, what: &str) -> (f64, TimingLock) {
    let own = format!("TIMING.{}", lock_name());
    let tag = format!("{} {}", lock_name(), std::process::id());
    let mut waited = 0u64;
    loop {
        let o = a.lock_dir.as_deref().map(|d| others(std::path::Path::new(d), &own)).unwrap_or_default();
        let b = busy();
        let l = load1();
        if o.is_empty() && b.is_empty() && l < a.max_load {
            let Some(d) = &a.lock_dir else { return (l, TimingLock(None, tag)) };
            let p = std::path::Path::new(d).join(&own);
            let created = std::fs::OpenOptions::new().write(true).create_new(true).open(&p).and_then(|mut f| {
                use std::io::Write;
                writeln!(f, "{tag}")
            });
            if created.is_ok() {
                if waited > 0 {
                    println!("  [quiet after {waited} s: load {l:.2}]");
                }
                return (l, TimingLock(Some(p), tag));
            }
        } else if waited % 120 == 0 {
            println!("  [waiting before {what}: load {l:.2}, busy {b:?}, lock files {o:?}]");
        }
        std::thread::sleep(Duration::from_secs(20));
        waited += 20;
    }
}

fn cpu_s() -> f64 {
    // SAFETY: `rusage` is plain integers and timevals, for which all-zero bytes are a valid value.
    let mut r: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: `r` is a valid, writable rusage for getrusage to fill.
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut r) };
    let tv = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
    tv(r.ru_utime) + tv(r.ru_stime)
}

// -------------------------------------------------------------------------------------------------
// runs

struct Run {
    wall: f64,
    cpu: f64,
    loss: f64,
}

async fn run(ctx: &SessionContext, sql: &str) -> Run {
    let c0 = cpu_s();
    let t0 = Instant::now();
    let df = ctx.sql(sql).await.unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    let out = collect(plan, ctx.task_ctx()).await.unwrap();
    let wall = t0.elapsed().as_secs_f64() * 1e3;
    let cpu = (cpu_s() - c0) * 1e3;
    let col = out.iter().find(|b| b.num_rows() > 0).map(|b| Arc::clone(b.column(0)));
    let loss = col.and_then(|c| c.as_any().downcast_ref::<Float64Array>().map(|f| f.value(0))).unwrap_or(f64::NAN);
    Run { wall, cpu, loss }
}

async fn count(ctx: &SessionContext, sql: &str) -> i64 {
    let out = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    out[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap().value(0)
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n == 0 {
        f64::NAN
    } else if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// The rule's report for the plans since the last clear: (taken, left, run-time on GPU,
/// run-time handed back, runtime fallbacks, the decisions as text).
fn report(rule: &ArrowMetalRule) -> (usize, usize, usize, usize, usize, Vec<String>) {
    let r = rule.report();
    let gpu = r.runtime_choices().filter(|d| d.taken).count();
    let back = r.runtime_choices().filter(|d| !d.taken).count();
    let text = r.decisions().iter().map(|d| d.to_string()).collect();
    (r.taken().count(), r.left().count(), gpu, back, r.runtime_fallbacks().count(), text)
}

// -------------------------------------------------------------------------------------------------
// arguments

struct Args {
    shapes: Vec<(usize, usize, usize)>,
    configs: Vec<String>,
    rounds: usize,
    warm_ms: f64,
    idle_ms: u64,
    idle_reps: usize,
    max_load: f64,
    max_config_s: f64,
    out: String,
    lock_dir: Option<String>,
    counts: bool,
}

fn args() -> Args {
    let mut a = Args {
        shapes: vec![(256, 128, 32), (512, 256, 64), (1024, 512, 64), (2048, 512, 64)],
        configs: vec!["a".into(), "b".into(), "c".into()],
        rounds: 5,
        warm_ms: 100.0,
        idle_ms: 500,
        idle_reps: 3,
        max_load: 3.5,
        max_config_s: 300.0,
        out: "results/attention.csv".into(),
        lock_dir: None,
        counts: true,
    };
    let v: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < v.len() {
        let val = v.get(i + 1).cloned().unwrap_or_default();
        match v[i].as_str() {
            "--shapes" => {
                a.shapes = val
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| {
                        let n: Vec<usize> = s.split('x').map(|x| x.parse().unwrap()).collect();
                        (n[0], n[1], n[2])
                    })
                    .collect()
            }
            "--configs" => a.configs = val.split(',').filter(|s| !s.is_empty()).map(String::from).collect(),
            "--rounds" => a.rounds = val.parse().unwrap(),
            "--warm-ms" => a.warm_ms = val.parse().unwrap(),
            "--idle-ms" => a.idle_ms = val.parse().unwrap(),
            "--idle-reps" => a.idle_reps = val.parse().unwrap(),
            "--max-load" => a.max_load = val.parse().unwrap(),
            "--max-config-s" => a.max_config_s = val.parse().unwrap(),
            "--out" => a.out = val,
            "--lock-dir" => a.lock_dir = Some(val),
            "--counts" => a.counts = val != "no",
            f => panic!("unknown flag {f}"),
        }
        i += 2;
    }
    assert_eq!(a.configs.first().map(String::as_str), Some("a"), "--configs must start with a (the reference)");
    a
}

fn csv(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

// -------------------------------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let a = args();
    let date = sh("date '+%Y-%m-%d'");
    let new_file = !std::path::Path::new(&a.out).exists();
    let mut out = std::fs::OpenOptions::new().create(true).append(true).open(&a.out).unwrap();
    let cond_path = a.out.trim_end_matches(".csv").to_string() + "_conditions.txt";
    let mut cond = std::fs::OpenOptions::new().create(true).append(true).open(&cond_path).unwrap();
    use std::io::Write;
    if new_file {
        writeln!(
            out,
            "date,t,d,e,config,config_desc,metric,value,unit,runs,block_load_start,block_load_end,block_t0,block_t1,last_sleep,note"
        )
        .unwrap();
    }
    let parts = SessionConfig::new().target_partitions();
    let head = format!(
        "attention: target_partitions {parts}; batch {BATCH}; rounds {}; warm {} ms; idle {} ms x {}; max load {}; configs {:?}",
        a.rounds, a.warm_ms, a.idle_ms, a.idle_reps, a.max_load, a.configs
    );
    println!("{head}");
    writeln!(cond, "\n# run started {} (pid {})\n{head}", now_text(), std::process::id()).unwrap();

    for &(t, d, e) in &a.shapes {
        println!("== shape t={t} d={d} e={e}");
        let data = tables(t, d, e);
        let mut ctxs: Vec<(String, SessionContext, Option<ArrowMetalRule>)> = a
            .configs
            .iter()
            .map(|c| {
                let r = rule_for(c);
                (c.clone(), context(c, r.clone(), &data), r)
            })
            .collect();

        let (l0, lock) = acquire(&a, &format!("shape {t}x{d}x{e}"));
        let (t0, sleep0) = (now_text(), last_sleep());
        println!("  [load {l0:.2} at block start, {t0}]");

        // First run of each configuration: the loss, the report of one plan, the time (untimed
        // in the statistics; the first GPU query of the process also compiles pipelines).
        struct Cfg {
            first: Run,
            report: Option<(usize, usize, usize, usize, usize, Vec<String>)>,
            skip: Option<String>,
            walls: Vec<f64>,
            cpus: Vec<f64>,
            losses: Vec<f64>,
            idle: Vec<f64>,
            idle_cpu: Vec<f64>,
            warm_runs: usize,
        }
        let mut cfg: Vec<Cfg> = Vec::new();
        for (name, ctx, rule) in &ctxs {
            if let Some(r) = rule {
                r.clear_report();
            }
            let first = run(ctx, ATTENTION).await;
            let report = rule.as_ref().map(report);
            // The runs the timing needs: per round a warm-up of at least one run and the timed run,
            // plus the idle runs.
            let needed = (2 * a.rounds + a.idle_reps) as f64 * first.wall / 1e3;
            let skip = (needed > a.max_config_s).then(|| {
                format!(
                    "not run: the first run took {:.1} s; {} rounds with warm-ups and {} idle runs need about {:.0} s, over the {:.0} s limit per configuration and shape",
                    first.wall / 1e3, a.rounds, a.idle_reps, needed, a.max_config_s
                )
            });
            println!(
                "  {name}: first {:.1} ms, cpu {:.1} ms, loss {:.17e}{}",
                first.wall,
                first.cpu,
                first.loss,
                report
                    .as_ref()
                    .map(|r| format!("; taken {} left {} gpu {} handback {} fallbacks {}", r.0, r.1, r.2, r.3, r.4))
                    .unwrap_or_default()
            );
            if let Some(s) = &skip {
                println!("  {name}: {s}");
            }
            cfg.push(Cfg {
                first,
                report,
                skip,
                walls: vec![],
                cpus: vec![],
                losses: vec![],
                idle: vec![],
                idle_cpu: vec![],
                warm_runs: 0,
            });
        }

        // Timed rounds, the order rotated by one per round.
        let n = ctxs.len();
        for round in 0..a.rounds {
            for k in 0..n {
                let i = (k + round) % n;
                if cfg[i].skip.is_some() {
                    continue;
                }
                let ctx = &ctxs[i].1;
                let mut spent = 0.0;
                while spent < a.warm_ms {
                    spent += run(ctx, ATTENTION).await.wall;
                    cfg[i].warm_runs += 1;
                }
                let r = run(ctx, ATTENTION).await;
                if std::env::var("BENCH_TRACE").is_ok() {
                    println!("    round {round} {}: wall {:.2} cpu {:.1}", ctxs[i].0, r.wall, r.cpu);
                }
                cfg[i].walls.push(r.wall);
                cfg[i].cpus.push(r.cpu);
                cfg[i].losses.push(r.loss);
            }
        }
        // First run after an idle gap, the configurations alternating.
        for rep in 0..a.idle_reps {
            for k in 0..n {
                let i = (k + rep) % n;
                if cfg[i].skip.is_some() {
                    continue;
                }
                std::thread::sleep(Duration::from_millis(a.idle_ms));
                let r = run(&ctxs[i].1, ATTENTION).await;
                cfg[i].idle.push(r.wall);
                cfg[i].idle_cpu.push(r.cpu);
                cfg[i].losses.push(r.loss);
            }
        }
        let l1 = load1();
        let (t1, sleep1) = (now_text(), last_sleep());
        drop(lock);
        println!("  [load {l1:.2} at block end, {t1}; last sleep {sleep1}]");
        let slept = if sleep0 != sleep1 { "; a Sleep entry was logged during the block" } else { "" };

        // Rows.
        let ref_loss = cfg[0].first.loss;
        let mut row = |c: &str, metric: &str, value: String, unit: &str, runs: usize, note: &str| {
            writeln!(
                out,
                "{date},{t},{d},{e},{c},{},{metric},{value},{unit},{runs},{l0:.2},{l1:.2},{t0},{t1},{},{}",
                csv(describe(c)),
                csv(&sleep1),
                csv(&format!("{note}{slept}"))
            )
            .unwrap();
        };
        for (i, c) in cfg.iter().enumerate() {
            let name = ctxs[i].0.as_str();
            row(name, "first_run_ms", format!("{:.2}", c.first.wall), "ms", 1, "the configuration's first run of the shape (untimed in best/median); the process's first GPU query also compiles pipelines");
            row(name, "first_run_cpu_ms", format!("{:.1}", c.first.cpu), "ms", 1, "");
            row(name, "loss", format!("{:.17e}", c.first.loss), "", 1, "first run");
            let rel = |x: f64| ((x - ref_loss) / ref_loss).abs();
            row(name, "loss_rel_diff_vs_a", format!("{:.3e}", rel(c.first.loss)), "", 1, "first run");
            if let Some(r) = &c.report {
                row(name, "report_taken", r.0.to_string(), "nodes", 1, "plan-time nodes replaced, one plan");
                row(name, "report_left", r.1.to_string(), "nodes", 1, "plan-time nodes left, one plan");
                row(name, "report_runtime_gpu", r.2.to_string(), "nodes", 1, "replaced aggregates that ran on ArrowMetal");
                row(name, "report_runtime_handed_back", r.3.to_string(), "nodes", 1, "replaced aggregates handed back to DataFusion");
                row(name, "report_runtime_fallbacks", r.4.to_string(), "nodes", 1, "");
                writeln!(cond, "\n[{t}x{d}x{e} {name}] {} -- report of the first run's plan:", describe(name)).unwrap();
                for line in &r.5 {
                    writeln!(cond, "  {line}").unwrap();
                }
            }
            if let Some(s) = &c.skip {
                row(name, "wall_best_ms", "".into(), "ms", 0, s);
                continue;
            }
            if c.walls.is_empty() {
                continue;
            }
            let k = c.walls.len();
            let (bi, best) = c.walls.iter().copied().enumerate().fold((0, f64::INFINITY), |acc, (j, w)| if w < acc.1 { (j, w) } else { acc });
            row(name, "wall_best_ms", format!("{best:.2}"), "ms", k, "");
            row(name, "wall_median_ms", format!("{:.2}", median(&c.walls)), "ms", k, "");
            row(name, "cpu_ms_of_best_run", format!("{:.1}", c.cpus[bi]), "ms", k, "");
            row(name, "cpu_median_ms", format!("{:.1}", median(&c.cpus)), "ms", k, "");
            row(name, "wall_runs_ms", csv(&c.walls.iter().map(|w| format!("{w:.2}")).collect::<Vec<_>>().join(" ")), "ms", k, "timed runs in round order");
            row(name, "cpu_runs_ms", csv(&c.cpus.iter().map(|w| format!("{w:.1}")).collect::<Vec<_>>().join(" ")), "ms", k, "the timed runs' CPU ms, in round order");
            row(name, "warmup_runs", c.warm_runs.to_string(), "runs", k, &format!("untimed runs before the timed runs, at least {} ms per round", a.warm_ms));
            if !c.idle.is_empty() {
                let m = c.idle.len();
                row(name, &format!("idle{}_first_median_ms", a.idle_ms), format!("{:.2}", median(&c.idle)), "ms", m, &format!("run after {} ms of sleep", a.idle_ms));
                row(name, &format!("idle{}_first_max_ms", a.idle_ms), format!("{:.2}", c.idle.iter().copied().fold(0.0, f64::max)), "ms", m, "");
                row(name, &format!("idle{}_cpu_median_ms", a.idle_ms), format!("{:.1}", median(&c.idle_cpu)), "ms", m, "");
            }
            let worst = c.losses.iter().map(|&x| ((x - ref_loss) / ref_loss).abs()).fold(0.0, f64::max);
            row(name, "loss_rel_diff_vs_a_max_all_runs", format!("{worst:.3e}"), "", c.losses.len(), "against a's first run");
            println!(
                "  {name}: best {best:.2} ms, median {:.2} ms, cpu(best) {:.1} ms, cpu median {:.1} ms, idle median {:.2} ms, max rel loss diff {worst:.2e}",
                median(&c.walls),
                c.cpus[bi],
                median(&c.cpus),
                median(&c.idle)
            );
        }
        out.flush().unwrap();
        writeln!(cond, "\n[{t}x{d}x{e}] block {t0} - {t1}, load {l0:.2} - {l1:.2}, last sleep {sleep1}").unwrap();
        ctxs.clear();
    }

    // Row counts of the intermediates, on DataFusion alone, after every timed block.
    if a.counts {
        for &(t, d, e) in &a.shapes {
            let data = tables(t, d, e);
            let ctx = context("a", None, &data);
            let c = ctes();
            let qs: [(&str, String, usize); 6] = [
                ("rows_q_join", "SELECT COUNT(*) FROM x JOIN wq w ON x.d = w.d".into(), t * d * e),
                ("rows_q", format!("{c} SELECT COUNT(*) FROM q"), t * e),
                ("rows_s_join", format!("{c} SELECT COUNT(*) FROM q JOIN k ON q.e = k.e"), t * t * e),
                ("rows_s", format!("{c} SELECT COUNT(*) FROM s"), t * t),
                ("rows_o_join", format!("{c} SELECT COUNT(*) FROM a JOIN v ON a.u = v.t"), t * t * e),
                ("rows_o", format!("{c} SELECT COUNT(*) FROM o"), t * e),
            ];
            for (metric, sql, expect) in qs {
                let n = count(&ctx, &sql).await;
                let note = if n as usize == expect { "counted with SQL; equals the shape's product" } else { "counted with SQL; DIFFERS from the shape's product" };
                println!("  {t}x{d}x{e} {metric} {n} (product {expect})");
                writeln!(
                    out,
                    "{date},{t},{d},{e},a,{},{metric},{n},rows,1,,,,,,{}",
                    csv(describe("a")),
                    csv(note)
                )
                .unwrap();
            }
        }
    }
    writeln!(cond, "\n# run ended {}", now_text()).unwrap();
}
