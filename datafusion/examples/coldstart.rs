//! What the first GPU query of a process costs: pipeline compilation, apart from the GPU's idle
//! state. Run one fresh process per measurement:
//!
//! ```text
//! ARROWMETAL_LIB=<dylib> target/release/examples/coldstart <query> [rows] [gap_ms]
//! ```
//!
//! `query` is one of `count`, `sum`, `minmax`, `distinct` (two int32 keys, 100,000 groups in the
//! key domain, an int64 value) or `sort` (an int64 key, three columns). The table is a `MemTable`
//! of 8,192-row batches over DataFusion's default partitions; the rule forces the node onto
//! ArrowMetal. Printed, in ms: DataFusion alone warm (best of 5); the rule's first run in the
//! process (pipelines compiled there, the GPU idle); three runs each after `gap_ms` of sleep
//! (compiled, the GPU idle); the warm best of 5.

use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{ArrayRef, Int32Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_arrowmetal::{session_context, AggregateChoice, ArrowMetalConfig, ArrowMetalRule};

async fn time(ctx: &SessionContext, sql: &str) -> f64 {
    let t = Instant::now();
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
    t.elapsed().as_secs_f64() * 1e3
}

async fn warm(ctx: &SessionContext, sql: &str) -> f64 {
    let mut spent = 0.0;
    while spent < 100.0 {
        spent += time(ctx, sql).await;
    }
    let mut best = f64::MAX;
    for _ in 0..5 {
        best = best.min(time(ctx, sql).await);
    }
    best
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let query = args.first().cloned().unwrap_or_else(|| "count".into());
    let rows: usize = args.get(1).map(|x| x.parse().unwrap()).unwrap_or(10_000_000);
    let gap: u64 = args.get(2).map(|x| x.parse().unwrap()).unwrap_or(500);
    let sql = match query.as_str() {
        "count" => "SELECT k1, k2, count(*) AS n FROM t GROUP BY k1, k2",
        "sum" => "SELECT k1, k2, sum(q) AS s FROM t GROUP BY k1, k2",
        "minmax" => "SELECT k1, k2, min(q) AS lo, max(q) AS hi FROM t GROUP BY k1, k2",
        "distinct" => "SELECT DISTINCT k1, k2 FROM t",
        "sort" => "SELECT q, k1, k2 FROM t ORDER BY q",
        q => panic!("unknown query {q}"),
    };
    // splitmix64 keys: k1 in [0, 1000), k2 in [0, 100), q in [0, 1e9).
    let mix = |mut z: u64| {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let k1: Int32Array = (0..rows as u64).map(|i| (mix(i) % 1000) as i32).collect();
    let k2: Int32Array = (0..rows as u64).map(|i| (mix(i ^ 0x55) % 100) as i32).collect();
    let q: Int64Array = (0..rows as u64).map(|i| (mix(i ^ 0xAA) % 1_000_000_000) as i64).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("k1", DataType::Int32, false),
        Field::new("k2", DataType::Int32, false),
        Field::new("q", DataType::Int64, false),
    ]));
    let b = RecordBatch::try_new(schema.clone(), vec![Arc::new(k1) as ArrayRef, Arc::new(k2), Arc::new(q)]).unwrap();
    let parts = SessionConfig::new().target_partitions();
    let mut p = vec![Vec::new(); parts];
    let (mut off, mut i) = (0, 0);
    while off < rows {
        let n = 8192.min(rows - off);
        p[i % parts].push(b.slice(off, n));
        off += n;
        i += 1;
    }
    let plain = SessionContext::new_with_config(SessionConfig::new());
    let rule = ArrowMetalRule::new(
        ArrowMetalConfig::all().with_min_rows(0).with_aggregate_choice(AggregateChoice::ArrowMetal),
    );
    let gpu = session_context(SessionConfig::new(), rule);
    for c in [&plain, &gpu] {
        c.register_table("t", Arc::new(MemTable::try_new(schema.clone(), p.clone()).unwrap())).unwrap();
    }
    let df = warm(&plain, sql).await;
    std::thread::sleep(Duration::from_millis(gap));
    let first = time(&gpu, sql).await;
    let mut idle = Vec::new();
    for _ in 0..3 {
        std::thread::sleep(Duration::from_millis(gap));
        idle.push(time(&gpu, sql).await);
    }
    let w = warm(&gpu, sql).await;
    idle.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "query={query} rows={rows} gap_ms={gap} datafusion_warm={df:.2} first_in_process={first:.2} idle_compiled={:.2},{:.2},{:.2} warm={w:.2}",
        idle[0], idle[1], idle[2]
    );
}
