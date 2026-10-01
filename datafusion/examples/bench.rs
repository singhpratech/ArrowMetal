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
//! * Warm-up (`--warm-ms`, default 100): before the timed runs of each context, untimed runs
//!   repeat until they add up to `--warm-ms` of wall time (the first, compared run not counted).
//!   The GPU steps up to its fast state only after about 20-25 ms of back-to-back queries. On the
//!   M4 Max, a 250,000-row Float64 ORDER BY runs its plan in 2.3-3.0 ms in the first seven queries
//!   after the GPU idled, and in 0.8 ms from the eighth on. Without the warm-up, the best of 5 sits
//!   on either side of that step depending on how long each run is. The CSV records the setting,
//!   the untimed runs of each context, and `on_first_ms`: the first rule-on run, untimed, with the
//!   GPU idle before it (the first case of a process also compiles its pipelines there).
//!   `--warm-ms 0` is the earlier method. `BENCH_TRACE=1` prints every timed run.
//! * `MetalExec`'s metrics from the best on-run: input wait, import, plan run, export (the
//!   `concat_ms` column is 0 since the crate hands the batches to the chunked import; it stays in
//!   the CSV so the files of the earlier crate state line up).
//! * Before each block (one family at one size and layout): wait until the 1-minute load is below
//!   `--max-load` and no cargo / rustc / swift-build / swift-frontend / swiftc / pytest runs,
//!   checking every 20 s. The load at the start and end of each block goes in every row.
//!   With `--lock-dir <dir>`, also wait while a `<dir>/BUILDING*` file exists (another process
//!   compiles or runs tests) or `<dir>/TIMING` exists (anyone else's timing), and hold
//!   `<dir>/TIMING` for the block, tagged `<BENCH_LANE> <pid>`; only that tag is ever removed.
//!   Each row records the block's start and end
//!   time and the last `Sleep` entry of `pmset -g log` at its end, so a block that spans a sleep
//!   can be found and rerun.
//! * Contexts (`--contexts`, default `off,on`): `off` DataFusion alone (always timed, the
//!   reference); `on` the rule taking every node it can translate, a replaced aggregate forced onto
//!   ArrowMetal (`AggregateChoice::ArrowMetal`); `back` the same with every replaced aggregate
//!   handed back to DataFusion at run time (`AggregateChoice::DataFusion`: the cost of the
//!   hand-back itself); `def` the default config, timed (the group-count probe and the measured
//!   table decide). Each context's first run is compared with `off`'s and its time is recorded as
//!   `<ctx>_first_ms` (the GPU idle before it: the previous context ran on the CPU).
//! * `--idle-ctx <ctx>`: the context timed after each idle gap next to `off` (default `def`; its
//!   times go to the `def_idle*` columns, and the `idle_ctx` column names it). `--on joins`: the
//!   `on` context is the default config with every translatable join replaced (`on_config` column).
//!   `--join-keys i64,i32,str`: the key types of the join family's cases.
//! * `--families gsweep`: the aggregate sweep. One block per group count (200, 10k, 100k, 1M,
//!   rows/2 in the key domain); per block, count / sum / avg / min+max over a Float64 and an int64
//!   value column, and DISTINCT, each over one key and over two keys, int32 and int64 keys.
//!   `out_rows` is the group count the data holds.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{Array, ArrayRef, AsArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray};
use arrow::compute::{cast, concat_batches, lexsort_to_indices, take, SortColumn};
use arrow::datatypes::{DataType, Field, Float64Type, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::{collect, ExecutionPlan};
use datafusion::prelude::{ParquetReadOptions, SessionConfig};
use datafusion_arrowmetal::{session_context, AggregateChoice, ArrowMetalConfig, ArrowMetalRule, JoinChoice, MetalExec};
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
    warm_ms: f64,
    max_load: f64,
    out: String,
    parquet_dir: Option<String>,
    parquet_rows: Vec<usize>,
    codecs: Vec<String>,
    /// Print each case's physical plan (off and on) and skip the timing.
    explain: bool,
    contexts: Vec<String>,
    lock_dir: Option<String>,
    /// Timed rounds per case; each round warms and times every context once, in an order rotated
    /// by one per round; a context's figure is its best over the rounds.
    rounds: usize,
    /// For a case no context ran on the GPU: after the warm-up, this many runs of each context,
    /// alternating run by run (the order rotated each time) instead of the rounds.
    alternate: usize,
    /// After the timed runs: per idle gap (`--idle-gap-ms`, one or two values), this many runs of
    /// `off` and `def`, each after that gap of sleep (the GPU idles; pipelines are compiled by then).
    /// The order of the contexts alternates run by run and the order of the gaps by repetition.
    /// The first gap goes to the `*_idle_*` columns, the second to the `*_idle2_*` columns.
    idle_reps: usize,
    idle_gaps: Vec<u64>,
    /// `--idle-gpu-only`: the idle runs only for a case some context ran on the GPU.
    idle_gpu_only: bool,
    /// The context timed after each idle gap next to `off` (default `def`); its times go to the
    /// `def_idle*` columns, and the `idle_ctx` column names it.
    idle_ctx: String,
    /// Key classes of the join family's cases: `i64` (`j_*`), `i32` (`j32_*`), `str` (`js_*`).
    join_keys: Vec<String>,
    /// `--on joins`: the `on` context is the default config with every translatable join
    /// replaced (column `on_config` = `joins`); `--on all` (default): every node forced.
    on_joins: bool,
    /// Only these (size, layout, case) triples, one `size,layout,case` per line.
    select: Option<std::collections::HashSet<(usize, String, String)>>,
}

fn args() -> Args {
    let mut a = Args {
        sizes: vec![1_000_000, 10_000_000, 50_000_000],
        modes: vec!["b8192".into(), "part1".into()],
        families: ["sort", "join", "groupby", "distinct", "filter", "parquet"].map(String::from).to_vec(),
        cases: vec![],
        iters: 5,
        slow_s: 2.0,
        warm_ms: 100.0,
        max_load: 3.5,
        out: "results/datafusion_rule.csv".into(),
        parquet_dir: None,
        parquet_rows: vec![10_000_000, 50_000_000],
        codecs: vec!["snappy".into(), "zstd".into()],
        explain: false,
        contexts: vec!["off".into(), "on".into()],
        rounds: 1,
        alternate: 0,
        idle_reps: 0,
        idle_gaps: vec![500],
        idle_gpu_only: false,
        idle_ctx: "def".into(),
        join_keys: vec!["i64".into()],
        on_joins: false,
        select: None,
        lock_dir: None,
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
            "--warm-ms" => a.warm_ms = val.parse().unwrap(),
            "--max-load" => a.max_load = val.parse().unwrap(),
            "--out" => a.out = val,
            "--parquet-dir" => a.parquet_dir = Some(val),
            "--parquet-rows" => a.parquet_rows = nums(&val),
            "--codecs" => a.codecs = list(&val),
            "--contexts" => a.contexts = list(&val),
            "--rounds" => a.rounds = val.parse().unwrap(),
            "--alternate" => a.alternate = val.parse().unwrap(),
            "--idle-reps" => a.idle_reps = val.parse().unwrap(),
            "--idle-gap-ms" => {
                a.idle_gaps = nums(&val).into_iter().map(|g: usize| g as u64).collect();
                assert!((1..=2).contains(&a.idle_gaps.len()), "--idle-gap-ms takes one or two values");
            }
            "--select" => {
                let text = std::fs::read_to_string(&val).unwrap();
                a.select = Some(
                    text.lines()
                        .filter(|l| !l.trim().is_empty())
                        .map(|l| {
                            let f: Vec<&str> = l.trim().split(',').collect();
                            (f[0].parse().unwrap(), f[1].to_string(), f[2].to_string())
                        })
                        .collect(),
                );
            }
            "--lock-dir" => a.lock_dir = Some(val),
            "--idle-ctx" => a.idle_ctx = val,
            "--join-keys" => a.join_keys = list(&val),
            "--on" => a.on_joins = match val.as_str() {
                "joins" => true,
                "all" => false,
                v => panic!("--on takes joins or all, got {v}"),
            },
            "--explain" => {
                a.explain = true;
                i += 1;
                continue;
            }
            "--idle-gpu-only" => {
                a.idle_gpu_only = true;
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

/// The timestamp of the last `Sleep` entry in `pmset -g log`.
fn last_sleep() -> String {
    let l = sh("pmset -g log | grep ' Sleep  ' | tail -1");
    l.split_whitespace().take(2).collect::<Vec<_>>().join(" ")
}

/// Holds `<dir>/TIMING` for one block. The file holds this process's tag (`<lane> <pid>`) and is
/// removed on drop only while it still holds exactly that tag: a TIMING file another lane or
/// process wrote (with its own tag, or none) is never removed here.
struct TimingLock(Option<std::path::PathBuf>, String);

impl TimingLock {
    fn holds(path: &std::path::Path, tag: &str) -> bool {
        std::fs::read_to_string(path).map(|s| s.trim_end() == tag).unwrap_or(false)
    }
}

impl Drop for TimingLock {
    fn drop(&mut self) {
        if let Some(p) = &self.0 {
            if Self::holds(p, &self.1) {
                std::fs::remove_file(p).ok();
            }
        }
    }
}

/// Whether a `BUILDING*` file (`BUILDING.<lane>`: a build or a test run) exists in the lock dir.
fn building(d: &std::path::Path) -> bool {
    std::fs::read_dir(d)
        .map(|it| it.flatten().any(|e| e.file_name().to_string_lossy().starts_with("BUILDING")))
        .unwrap_or(false)
}

/// This process's TIMING tag: `<BENCH_LANE> <pid>` (`BENCH_LANE` defaults to `bench`).
fn timing_tag() -> String {
    format!("{} {}", std::env::var("BENCH_LANE").unwrap_or_else(|_| "bench".into()), std::process::id())
}

/// Waits for a quiet machine and, with a lock dir, for no `BUILDING*` file and no TIMING file
/// at all (another lane's, another process's, or one made with `touch`), then creates TIMING with
/// this process's tag (create-new, so two processes cannot both take it). Returns the load at
/// the start and the lock.
fn acquire(a: &Args, what: &str) -> (f64, TimingLock) {
    let tag = timing_tag();
    loop {
        if let Some(d) = &a.lock_dir {
            let d = std::path::Path::new(d);
            let mut waited = 0;
            loop {
                let building = building(d);
                let timing = d.join("TIMING").exists();
                if !building && !timing {
                    break;
                }
                if waited % 120 == 0 {
                    println!("  [waiting before {what}: BUILDING {building}, TIMING held {timing}]");
                }
                std::thread::sleep(Duration::from_secs(10));
                waited += 10;
            }
        }
        let l = wait_quiet(a.max_load, what);
        let Some(d) = &a.lock_dir else { return (l, TimingLock(None, tag)) };
        let d = std::path::Path::new(d);
        let t = d.join("TIMING");
        // Created only if absent: another process may have taken TIMING while this one waited
        // for the load to settle.
        let created = std::fs::OpenOptions::new().write(true).create_new(true).open(&t).and_then(|mut f| {
            use std::io::Write;
            writeln!(f, "{tag}")
        });
        if created.is_err() {
            // Someone took it between the check and the create: wait again.
            continue;
        }
        if building(d) {
            if TimingLock::holds(&t, &tag) {
                std::fs::remove_file(&t).ok();
            }
            continue;
        }
        return (l, TimingLock(Some(t), tag));
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

/// k1 int32 [0, 100k), k2 int32 [0, 1000), name utf8 (1000 values), x f64 [0, 1), q int64 [0, 1e9),
/// y f32 [0, 1) (drawn last, so the other columns are the earlier runs' values).
fn data_extra(rng: &mut StdRng, n: usize) -> RecordBatch {
    let k1 = i32s(rng, n, 100_000);
    let k2 = i32s(rng, n, 1000);
    let name: ArrayRef =
        Arc::new(StringArray::from_iter_values((0..n).map(|_| rng.random_range(0..1000).to_string())));
    let x = f64s(rng, n, 1.0, 0.0);
    let q = i64s(rng, n, 1_000_000_000);
    let y: ArrayRef = Arc::new(Float32Array::from_iter_values((0..n).map(|_| rng.random::<f32>())));
    batch(vec![("k1", k1), ("k2", k2), ("name", name), ("x", x), ("q", q), ("y", y)])
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
/// The same keys as int32 (`k32`) and as Utf8 decimal text (`ks`) when `keys` asks for them.
fn data_join(rng: &mut StdRng, n: usize, build: usize, keys: &[String]) -> (RecordBatch, RecordBatch) {
    let pk = i64s(rng, n, 2 * build as i64);
    let v = f64s(rng, n, 100.0, 0.0);
    let bk: ArrayRef = Arc::new(Int64Array::from_iter_values((0..build as i64).map(|i| 2 * i)));
    let w = f64s(rng, build, 2.0, 0.0);
    let mut pc = vec![("k", Arc::clone(&pk)), ("v", v)];
    let mut bc = vec![("k", Arc::clone(&bk)), ("w", w)];
    if keys.iter().any(|k| k == "i32") {
        pc.push(("k32", cast(&pk, &DataType::Int32).unwrap()));
        bc.push(("k32", cast(&bk, &DataType::Int32).unwrap()));
    }
    if keys.iter().any(|k| k == "str") {
        pc.push(("ks", cast(&pk, &DataType::Utf8).unwrap()));
        bc.push(("ks", cast(&bk, &DataType::Utf8).unwrap()));
    }
    (batch(pc), batch(bc))
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

/// The aggregate sweep's table: `data_grid`'s int32 keys, the same keys as int64 (`kl`, `kl1`,
/// `kl2`), a Float64 value `x` [0, 1) and an int64 value `q` [0, 1e9).
fn data_sweep(rng: &mut StdRng, n: usize, g: usize) -> RecordBatch {
    let t = data_grid(rng, n, g);
    let q = i64s(rng, n, 1_000_000_000);
    let wide = |i: usize| cast(t.column(i), &DataType::Int64).unwrap();
    batch(vec![
        ("k", Arc::clone(t.column(0))),
        ("k1", Arc::clone(t.column(1))),
        ("k2", Arc::clone(t.column(2))),
        ("kl", wide(0)),
        ("kl1", wide(1)),
        ("kl2", wide(2)),
        ("x", Arc::clone(t.column(3))),
        ("q", q),
    ])
}

/// The sweep's cases for one group count: (id, label, SQL).
fn sweep_cases(gn: &str) -> Vec<Case> {
    let mut out = Vec::new();
    for (kc, one, two) in [("i32", "k", "k1, k2"), ("i64", "kl", "kl1, kl2")] {
        for (nk, keys) in [("1", one), ("2", two)] {
            for (fam, agg) in [
                ("count", Some("count(*) AS n")),
                ("sum_f64", Some("sum(x) AS s")),
                ("avg_f64", Some("avg(x) AS m")),
                ("minmax_f64", Some("min(x) AS lo, max(x) AS hi")),
                ("sum_int", Some("sum(q) AS s")),
                ("avg_int", Some("avg(q) AS m")),
                ("minmax_int", Some("min(q) AS lo, max(q) AS hi")),
                ("distinct", None),
            ] {
                let sql = match agg {
                    Some(agg) => format!("SELECT {keys}, {agg} FROM grid GROUP BY {keys}"),
                    None => format!("SELECT DISTINCT {keys} FROM grid"),
                };
                out.push(case(
                    &format!("{fam}_{nk}{kc}_{gn}"),
                    &format!("{fam}, {nk} {kc} key(s), {gn} groups in the key domain"),
                    &sql,
                    Order::Any,
                ));
            }
        }
    }
    out
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

/// Builds one block's tables and cases (run when the block's turn comes).
type BlockFn = Box<dyn FnOnce(&mut StdRng) -> Block>;

/// The blocks of one size, generated lazily (only one block's tables alive at a time).
fn blocks(rows: usize, fam: &str, join_keys: &[String], rng: &mut StdRng) -> Vec<BlockFn> {
    let mut out: Vec<BlockFn> = Vec::new();
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
                case("srt_f32", "sort 3 columns by a Float32 key", "SELECT q, k1, y FROM extra ORDER BY y", Order::Keys(vec![2])),
                case("top_f32", "top 100 by Float32, descending", "SELECT q, k1, y FROM extra ORDER BY y DESC LIMIT 100", Order::KeysLimit(vec![2])),
            ],
        })),
        "join" => {
            for b in [10_000usize, 1_000_000, 10_000_000] {
                let keys = join_keys.to_vec();
                out.push(Box::new(move |rng| {
                    let (probe, build) = data_join(rng, rows, b, &keys);
                    let bn = gname(b, 0);
                    let mut cases = Vec::new();
                    for kc in &keys {
                        let (pfx, col, kt) = match kc.as_str() {
                            "i64" => ("", "k", "int64"),
                            "i32" => ("32", "k32", "int32"),
                            "str" => ("s", "ks", "Utf8"),
                            k => panic!("unknown join key class {k}"),
                        };
                        for (how, sql_how) in [("inner", "JOIN"), ("left", "LEFT JOIN")] {
                            cases.push(case(
                                &format!("j{pfx}_{how}_{bn}"),
                                &format!("{how} join on an {kt} key, {bn}-row build, count + sum over the whole result"),
                                &format!("SELECT count(*) AS n, sum(b.w) AS s FROM probe p {sql_how} build b ON p.{col} = b.{col}"),
                                Order::Any,
                            ));
                            cases.push(case(
                                &format!("jg{pfx}_{how}_{bn}"),
                                &format!("{how} join on an {kt} key, {bn}-row build, then group by the probe key: count + sum"),
                                &format!(
                                    "SELECT p.{col}, count(*) AS n, sum(p.v) AS s FROM probe p {sql_how} build b ON p.{col} = b.{col} GROUP BY p.{col}"
                                ),
                                Order::Any,
                            ));
                        }
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
                    // A Float32 value column `y` [0, 1), drawn after the others (so they keep the
                    // values of the files without it).
                    let y: ArrayRef = Arc::new(Float32Array::from_iter_values((0..rows).map(|_| rng.random::<f32>())));
                    let mut cols: Vec<(&str, ArrayRef)> =
                        ["k", "k1", "k2", "x"].iter().enumerate().map(|(i, n)| (*n, Arc::clone(t.column(i)))).collect();
                    cols.push(("y", y));
                    let t = batch(cols);
                    let gn = gname(g, rows);
                    let mut cases = Vec::new();
                    for (fam, agg) in [
                        ("count", "count(*) AS n"),
                        ("sum", "sum(x) AS s"),
                        ("mean", "avg(x) AS m"),
                        ("minmax", "min(x) AS lo, max(x) AS hi"),
                        ("minmax32", "min(y) AS lo, max(y) AS hi"),
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
        // The same grid with an int64 value column (D1's `q`): whether the rule's take-list must
        // tell aggregates over Float64 from those over integers.
        "groupby_i64" => {
            let mut gs = vec![200usize, 10_000, 100_000, 1_000_000, rows / 2];
            gs.dedup();
            for g in gs {
                out.push(Box::new(move |rng| {
                    let t = data_grid(rng, rows, g);
                    let q = i64s(rng, rows, 1_000_000_000);
                    let mut cols: Vec<(&str, ArrayRef)> =
                        ["k", "k1", "k2"].iter().enumerate().map(|(i, n)| (*n, Arc::clone(t.column(i)))).collect();
                    cols.push(("q", q));
                    let t = batch(cols);
                    let gn = gname(g, rows);
                    let mut cases = Vec::new();
                    for (fam, agg) in
                        [("sum", "sum(q) AS s"), ("mean", "avg(q) AS m"), ("minmax", "min(q) AS lo, max(q) AS hi")]
                    {
                        cases.push(case(
                            &format!("gi1{fam}{gn}"),
                            &format!("group-by 1 int32 key, {gn} groups, {agg} (int64 values)"),
                            &format!("SELECT k, {agg} FROM grid GROUP BY k"),
                            Order::Any,
                        ));
                        cases.push(case(
                            &format!("gi2{fam}{gn}"),
                            &format!("group-by 2 int32 keys, {gn} groups, {agg} (int64 values)"),
                            &format!("SELECT k1, k2, {agg} FROM grid GROUP BY k1, k2"),
                            Order::Any,
                        ));
                    }
                    Block { family: "groupby_i64", source: Source::Mem(vec![("grid", t)]), cases }
                }));
            }
        }
        "gsweep" => {
            let mut gs = vec![200usize, 10_000, 100_000, 1_000_000, rows / 2];
            gs.dedup();
            for g in gs {
                out.push(Box::new(move |rng| {
                    let gn = gname(g, rows);
                    Block { family: "gsweep", source: Source::Mem(vec![("grid", data_sweep(rng, rows, g))]), cases: sweep_cases(&gn) }
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
            cases: vec![
                case(
                    "a_sumcnt",
                    "filtered sum + count over the whole table (control)",
                    "SELECT sum(amount) AS total, count(*) AS n FROM fact WHERE region < 20 AND qty > 10",
                    Order::Any,
                ),
                // Float comparisons against a literal (totalOrder, zeros equal): the rows that pass.
                case("f_gt", "rows with a Float64 above a finite literal (25%)", "SELECT region, amount FROM fact WHERE amount > 1000.0", Order::Any),
                case("f_lt0", "rows with a Float64 below zero (25%)", "SELECT region, amount FROM fact WHERE amount < 0.0", Order::Any),
                case("f_ne0", "rows with a Float64 not equal to zero (all)", "SELECT region, qty FROM fact WHERE amount <> 0.0", Order::Any),
                case(
                    "f_sum_ge0",
                    "sum + count of the rows with a Float64 at or above zero (75%)",
                    "SELECT sum(qty) AS q, count(*) AS n FROM fact WHERE amount >= 0.0",
                    Order::Any,
                ),
            ],
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

const METRICS: &[&str] = &[
    "input_time",
    "concat_time",
    "import_time",
    "kernel_time",
    "export_time",
    "input_batches",
    "probe_time",
    "handed_back",
    "groups_estimate",
];

fn metal_metrics(plan: &Arc<dyn ExecutionPlan>, acc: &mut BTreeMap<&'static str, f64>) {
    if let Some(m) = plan.downcast_ref::<MetalExec>() {
        *acc.entry("nodes").or_default() += 1.0;
        if let Some(ms) = m.metrics() {
            for &k in METRICS {
                if let Some(v) = ms.sum_by_name(k) {
                    let scale = if matches!(k, "input_batches" | "handed_back" | "groups_estimate") { 1.0 } else { 1e-6 };
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

/// Untimed runs until they add up to `--warm-ms` of wall time (the first, compared run is not
/// counted: in a process's first case it also compiles pipelines while the GPU idles), then the best
/// of `--iters` timed runs. Returns the best run, the timed count and the untimed count (the first
/// run included).
async fn best(ctx: &SessionContext, sql: &str, first_ms: f64, a: &Args) -> (Run, usize, usize) {
    let mut spent = 0.0;
    let mut untimed = 1;
    while spent < a.warm_ms {
        spent += run(ctx, sql).await.wall;
        untimed += 1;
    }
    let n = if first_ms > a.slow_s * 1e3 { a.iters.min(2) } else { a.iters };
    let mut best: Option<Run> = None;
    for _ in 0..n {
        let r = run(ctx, sql).await;
        let r = Run { out: Vec::new(), ..r };
        if std::env::var("BENCH_TRACE").is_ok() {
            println!("    iter wall {:6.2} kernel {:6.2}", r.wall, r.metal.get("kernel_time").copied().unwrap_or(0.0));
        }
        if best.as_ref().is_none_or(|b| r.wall < b.wall) {
            best = Some(r);
        }
    }
    (best.unwrap(), n, untimed)
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

/// One timed context of a case.
struct Ctx {
    name: String,
    ctx: SessionContext,
    rule: Option<ArrowMetalRule>,
}

/// One context's result for one case.
struct Res {
    best: Run,
    iters: usize,
    untimed: usize,
    first_ms: f64,
    equal: String,
    dev: f64,
    taken: usize,
    decisions: String,
    fallbacks: usize,
}

fn rule_for(name: &str, on_joins: bool) -> Option<ArrowMetalRule> {
    let all = ArrowMetalConfig::all().with_min_rows(0).with_accept_inexact(true);
    match name {
        "off" => None,
        // `--on joins`: the default config with every translatable join replaced, so only the
        // join differs from DataFusion alone (an aggregate above it stays DataFusion's, as under
        // the default).
        "on" if on_joins => Some(ArrowMetalRule::new(
            ArrowMetalConfig::default().with_min_rows(0).with_join_choice(JoinChoice::ArrowMetal),
        )),
        "on" => Some(ArrowMetalRule::new(
            all.clone().with_aggregate_choice(AggregateChoice::ArrowMetal).with_join_choice(JoinChoice::ArrowMetal),
        )),
        "back" => Some(ArrowMetalRule::new(all.clone().with_aggregate_choice(AggregateChoice::DataFusion))),
        "def" => Some(ArrowMetalRule::new(ArrowMetalConfig::default())),
        c => panic!("unknown context {c}"),
    }
}

#[tokio::main]
async fn main() {
    let a = args();
    assert_eq!(a.contexts.first().map(String::as_str), Some("off"), "--contexts must start with off");
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
             import_ms,kernel_ms,export_ms,input_batches,out_rows,block_load_start,block_load_end,warm_ms,untimed_off,\
             untimed_on,on_first_ms,back_ms,back_cpu_ms,back_first_ms,equal_back,def_ms,def_cpu_ms,def_first_ms,\
             equal_def,def_taken,def_handed_back,def_groups_est,def_probe_ms,def_input_ms,def_kernel_ms,def_choice,\
             off_first_ms,block_t0,block_t1,last_sleep,rounds,off_round_median_ms,def_round_median_ms,\
             back_round_median_ms,on_round_median_ms,method,def_gpu,def_idle_ms,def_idle_max_ms,off_idle_ms,\
             idle_gap_ms,off_idle_max_ms,def_idle_cpu_ms,off_idle_cpu_ms,idle_reps,idle_gap2_ms,def_idle2_ms,\
             def_idle2_max_ms,off_idle2_ms,off_idle2_max_ms,def_idle2_cpu_ms,off_idle2_cpu_ms,idle_ctx,on_config"
        )
        .unwrap();
    }
    let parts = SessionConfig::new().target_partitions();
    println!(
        "target_partitions {parts}; iters {}; slow {} s; warm {} ms; max load {}; contexts {:?}",
        a.iters, a.slow_s, a.warm_ms, a.max_load, a.contexts
    );

    // (size, layout, family, block builder)
    let mut work: Vec<(usize, String, BlockFn)> = Vec::new();
    for &rows in &a.sizes {
        for mode in &a.modes {
            for fam in &a.families {
                if fam == "parquet" || fam == "psort" {
                    continue;
                }
                let mut dummy = StdRng::seed_from_u64(0);
                for b in blocks(rows, fam, &a.join_keys, &mut dummy) {
                    work.push((rows, mode.clone(), b));
                }
            }
        }
        // A Parquet file with a String column, written on first use: DataFusion reads its strings as
        // Utf8View, so the sorts carry (or sort by) a Utf8View column.
        if a.families.iter().any(|f| f == "psort") {
            if let Some(dir) = &a.parquet_dir {
                let path = format!("{dir}/psort-{rows}.parquet");
                work.push((
                    rows,
                    "parquet-psort".to_string(),
                    Box::new(move |rng| {
                        if !std::path::Path::new(&path).exists() {
                            let b = data_extra(rng, rows);
                            let f = std::fs::File::create(&path).unwrap();
                            let mut w = datafusion::parquet::arrow::ArrowWriter::try_new(f, b.schema(), None).unwrap();
                            let mut off = 0;
                            while off < rows {
                                let n = (1usize << 20).min(rows - off);
                                w.write(&b.slice(off, n)).unwrap();
                                off += n;
                            }
                            w.close().unwrap();
                        }
                        Block {
                            family: "psort",
                            source: Source::Parquet(path),
                            cases: vec![
                                case("ps_str", "Parquet scan, sort 3 columns by a String key (Utf8View)", "SELECT name, q, k1 FROM f ORDER BY name", Order::Keys(vec![0])),
                                case("ps_i64_str", "Parquet scan, sort by int64, a String column carried", "SELECT q, name, k1 FROM f ORDER BY q", Order::Keys(vec![0])),
                            ],
                        }
                    }),
                ));
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
        let cases: Vec<&Case> = block
            .cases
            .iter()
            .filter(|c| a.cases.is_empty() || a.cases.contains(&c.id))
            .filter(|c| a.select.as_ref().is_none_or(|s| s.contains(&(rows, mode.clone(), c.id.clone()))))
            .collect();
        if cases.is_empty() {
            continue;
        }
        println!("== {} rows, {}, {} ({} cases; data {:.1} s)", rows, mode, block.family, cases.len(), t.elapsed().as_secs_f64());
        let mem_mode = if mode.starts_with("parquet") { "b8192" } else { mode.as_str() };
        let rule_def = ArrowMetalRule::new(ArrowMetalConfig::default());
        let ctx_def = context(&block.source, mem_mode, Some(rule_def.clone())).await;
        let mut ctxs = Vec::new();
        for name in &a.contexts {
            let rule = rule_for(name, a.on_joins);
            let ctx = context(&block.source, mem_mode, rule.clone()).await;
            ctxs.push(Ctx { name: name.clone(), ctx, rule });
        }
        let (l0, lock) = acquire(&a, &format!("{} {rows} {mode}", block.family));
        let t0 = now_text();
        println!("  [load {l0:.2} at block start, {t0}]");
        let mut rows_out = Vec::new();
        for c in cases {
            // Decisions under the default config (planned, not run).
            rule_def.clear_report();
            let _ = ctx_def.sql(&c.sql).await.unwrap().create_physical_plan().await.unwrap();
            let (_, def_s) = decisions(&rule_def);

            if a.explain {
                for x in &ctxs {
                    let plan = x.ctx.sql(&c.sql).await.unwrap().create_physical_plan().await.unwrap();
                    let shown = datafusion::physical_plan::displayable(plan.as_ref()).indent(false).to_string();
                    println!("--- {} {}\n{shown}", c.id, x.name);
                }
                println!("    default: {def_s}");
                continue;
            }
            // First runs, in context order: each compared with off's.
            let mut firsts: Vec<(Run, usize, String, usize)> = Vec::new();
            let mut gpu = false;
            for x in &ctxs {
                if let Some(r) = &x.rule {
                    r.clear_report();
                }
                let r = run(&x.ctx, &c.sql).await;
                let (taken, s, fb) = match &x.rule {
                    Some(rule) => {
                        let (t, s) = decisions(rule);
                        let rep = rule.report();
                        // A replaced sort or filter always runs on the GPU; an aggregate when its
                        // run-time choice says so.
                        let metal_nodes = r.metal.get("nodes").copied().unwrap_or(0.0) > 0.0;
                        let handed = rep.runtime_choices().count() > 0 && rep.runtime_choices().all(|d| !d.taken);
                        gpu |= metal_nodes && !handed && rep.runtime_fallbacks().count() == 0;
                        (t, s, rep.runtime_fallbacks().count())
                    }
                    None => (0, String::new(), 0),
                };
                firsts.push((r, taken, s, fb));
            }
            let out_rows: usize = firsts[0].0.out.iter().map(|b| b.num_rows()).sum();
            let mut res: Vec<Res> = Vec::new();
            for (i, x) in ctxs.iter().enumerate() {
                let (equal, dev) = if i == 0 {
                    ("yes".to_string(), 0.0)
                } else {
                    match same(&firsts[0].0.out, &firsts[i].0.out, &c.order) {
                        Ok(d) => ("yes".to_string(), d),
                        Err(e) => {
                            println!("  !! {} {} {}: results differ: {e}", c.id, rows, x.name);
                            (format!("NO: {e}"), f64::NAN)
                        }
                    }
                };
                let first_ms = firsts[i].0.wall;
                let (_, taken, ref s, fb) = firsts[i];
                res.push(Res {
                    best: Run { out: Vec::new(), wall: first_ms, cpu: 0.0, metal: BTreeMap::new() },
                    iters: 0,
                    untimed: 0,
                    first_ms,
                    equal,
                    dev,
                    taken,
                    decisions: s.clone(),
                    fallbacks: fb,
                });
            }
            drop(firsts);
            let mut round_bests: Vec<Vec<f64>> = vec![Vec::new(); ctxs.len()];
            let alternate = a.alternate > 0 && !gpu;
            let idle_reps = if a.idle_gpu_only && !gpu { 0 } else { a.idle_reps };
            if alternate {
                // CPU only in every context: warm each, then run them in turn, run by run.
                for x in &ctxs {
                    let mut spent = 0.0;
                    while spent < a.warm_ms {
                        spent += run(&x.ctx, &c.sql).await.wall;
                    }
                }
                for x in &ctxs {
                    if let Some(r) = &x.rule {
                        r.clear_report();
                    }
                }
                for k in 0..a.alternate {
                    for j in 0..ctxs.len() {
                        let i = (j + k) % ctxs.len();
                        let r = run(&ctxs[i].ctx, &c.sql).await;
                        let r = Run { out: Vec::new(), ..r };
                        round_bests[i].push(r.wall);
                        if res[i].iters == 0 || r.wall < res[i].best.wall {
                            res[i].best = r;
                        }
                        res[i].iters += 1;
                    }
                }
                for (i, x) in ctxs.iter().enumerate() {
                    if let Some(r) = &x.rule {
                        res[i].fallbacks = res[i].fallbacks.max(r.report().runtime_fallbacks().count());
                    }
                }
            }
            for round in 0..if alternate { 0 } else { a.rounds.max(1) } {
                for k in 0..ctxs.len() {
                    let i = (k + round) % ctxs.len();
                    let x = &ctxs[i];
                    if let Some(r) = &x.rule {
                        r.clear_report();
                    }
                    let (b, n, u) = best(&x.ctx, &c.sql, res[i].first_ms, &a).await;
                    if let Some(r) = &x.rule {
                        res[i].fallbacks = res[i].fallbacks.max(r.report().runtime_fallbacks().count());
                    }
                    round_bests[i].push(b.wall);
                    if round == 0 || b.wall < res[i].best.wall {
                        res[i].best = b;
                    }
                    res[i].iters += n;
                    res[i].untimed += u;
                }
            }
            let median = |v: &[f64]| {
                let mut v = v.to_vec();
                v.sort_by(|x, y| x.partial_cmp(y).unwrap());
                v[v.len() / 2]
            };
            // First runs after the GPU idled, pipelines compiled (every context has run by now).
            // The order alternates by repetition so neither context always follows the other.
            // (gap index, context) -> wall and CPU times of the runs after that gap.
            let mut idle: BTreeMap<(usize, &str), Vec<f64>> = BTreeMap::new();
            let mut idle_cpu: BTreeMap<(usize, &str), Vec<f64>> = BTreeMap::new();
            let mut turn = 0;
            for rep in 0..idle_reps {
                let mut gaps: Vec<usize> = (0..a.idle_gaps.len()).collect();
                if rep % 2 == 1 {
                    gaps.reverse();
                }
                for gi in gaps {
                    for k in 0..2 {
                        let name = if (turn + k) % 2 == 0 { "off" } else { a.idle_ctx.as_str() };
                        if let Some(x) = ctxs.iter().find(|x| x.name == name) {
                            std::thread::sleep(Duration::from_millis(a.idle_gaps[gi]));
                            let r = run(&x.ctx, &c.sql).await;
                            idle.entry((gi, name)).or_default().push(r.wall);
                            idle_cpu.entry((gi, name)).or_default().push(r.cpu);
                        }
                    }
                    turn += 1;
                }
            }
            let idle_med =
                |g: usize, n: &str| idle.get(&(g, n)).map(|v| format!("{:.2}", median(v))).unwrap_or_default();
            let idle_max = |g: usize, n: &str| {
                idle.get(&(g, n)).map(|v| format!("{:.2}", v.iter().cloned().fold(0.0, f64::max))).unwrap_or_default()
            };
            let idle_cpu_med =
                |g: usize, n: &str| idle_cpu.get(&(g, n)).map(|v| format!("{:.1}", median(v))).unwrap_or_default();
            let gap = |g: usize| {
                if idle_reps > 0 { a.idle_gaps.get(g).map(|x| x.to_string()).unwrap_or_default() } else { String::new() }
            };
            let by = |n: &str| ctxs.iter().position(|x| x.name == n).map(|i| &res[i]);
            let off = &res[0];
            let on = by("on");
            let back = by("back");
            let def = by("def");
            let f = |r: Option<&Res>, g: &dyn Fn(&Res) -> String| r.map(g).unwrap_or_default();
            let mm = |r: &Res, k: &str| r.best.metal.get(k).copied().unwrap_or(0.0);
            let ms = |r: Option<&Res>| f(r, &|r| format!("{:.2}", r.best.wall));
            let cpu = |r: Option<&Res>| f(r, &|r| format!("{:.1}", r.best.cpu));
            let first = |r: Option<&Res>| f(r, &|r| format!("{:.2}", r.first_ms));
            let eq = |r: Option<&Res>| f(r, &|r| csv(&r.equal));
            let ratio = |r: Option<&Res>| f(r, &|r| format!("{:.3}", off.best.wall / r.best.wall));
            print!("  {:22} off {:8.1}", c.id, off.best.wall);
            for (name, r) in [("on", on), ("back", back), ("def", def)] {
                if let Some(r) = r {
                    print!("  {name} {:8.1} {:5.2}x", r.best.wall, off.best.wall / r.best.wall);
                }
            }
            if let Some(d) = def {
                print!("  [def: {} probe {:.3} ms]", if mm(d, "handed_back") > 0.0 { "handed back" } else if mm(d, "nodes") > 0.0 { "gpu" } else { "left" }, mm(d, "probe_time"));
            }
            println!();
            let equal_all = res.iter().skip(1).map(|r| r.equal.clone()).find(|e| e != "yes").unwrap_or_else(|| "yes".into());
            let dev = res.iter().skip(1).map(|r| r.dev).fold(0.0f64, |x, y| if y.is_nan() { f64::NAN } else { x.max(y) });
            let tail = format!("{},{},{},{}", a.warm_ms, off.untimed, f(on, &|r| r.untimed.to_string()), first(on));
            let onr = on.unwrap_or(off);
            let row = format!(
                "{rows},{mode},{},{},{},{},{:.2},{},{},{:.1},{},{},{},{},{:.3e},{},{},{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{},{out_rows}",
                block.family,
                c.id,
                csv(&c.label),
                csv(&c.sql),
                off.best.wall,
                ms(on),
                ratio(on),
                off.best.cpu,
                cpu(on),
                off.iters,
                f(on, &|r| r.iters.to_string()),
                csv(&equal_all),
                dev,
                f(on, &|r| r.taken.to_string()),
                f(on, &|r| csv(&r.decisions)),
                csv(&def_s),
                f(on, &|r| r.fallbacks.to_string()),
                mm(onr, "nodes"),
                mm(onr, "input_time"),
                mm(onr, "concat_time"),
                mm(onr, "import_time"),
                mm(onr, "kernel_time"),
                mm(onr, "export_time"),
                mm(onr, "input_batches"),
            );
            let extra = format!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.2}",
                ms(back),
                cpu(back),
                first(back),
                eq(back),
                ms(def),
                cpu(def),
                first(def),
                eq(def),
                f(def, &|r| r.taken.to_string()),
                f(def, &|r| format!("{}", mm(r, "handed_back"))),
                f(def, &|r| format!("{}", mm(r, "groups_estimate"))),
                f(def, &|r| format!("{:.4}", mm(r, "probe_time"))),
                f(def, &|r| format!("{:.2}", mm(r, "input_time"))),
                f(def, &|r| format!("{:.2}", mm(r, "kernel_time"))),
                f(def, &|r| csv(&r.decisions)),
                off.first_ms,
            );
            let med = |n: &str| {
                ctxs.iter().position(|x| x.name == n).map(|i| format!("{:.2}", median(&round_bests[i]))).unwrap_or_default()
            };
            let rounds_s = format!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                if alternate { a.alternate } else { a.rounds.max(1) },
                med("off"),
                med("def"),
                med("back"),
                med("on"),
                if alternate { "alternate" } else { "rounds" },
                gpu,
                idle_med(0, a.idle_ctx.as_str()),
                idle_max(0, a.idle_ctx.as_str()),
                idle_med(0, "off"),
                gap(0),
                idle_max(0, "off"),
                idle_cpu_med(0, a.idle_ctx.as_str()),
                idle_cpu_med(0, "off"),
                idle_reps,
                gap(1),
                idle_med(1, a.idle_ctx.as_str()),
                idle_max(1, a.idle_ctx.as_str()),
                idle_med(1, "off"),
                idle_max(1, "off"),
                idle_cpu_med(1, a.idle_ctx.as_str()),
                idle_cpu_med(1, "off"),
            );
            rows_out.push((row, tail, extra, rounds_s));
        }
        let l1 = load1();
        let t1 = now_text();
        drop(lock);
        let sleep = last_sleep();
        println!("  [load {l1:.2} at block end, {t1}; last sleep {sleep}]");
        for (r, tail, extra, rounds_s) in rows_out {
            writeln!(
                out,
                "{r},{l0:.2},{l1:.2},{tail},{extra},{t0},{t1},{sleep},{rounds_s},{},{}",
                a.idle_ctx,
                if a.on_joins { "joins" } else { "all" }
            )
            .unwrap();
        }
        out.flush().unwrap();
    }
}
