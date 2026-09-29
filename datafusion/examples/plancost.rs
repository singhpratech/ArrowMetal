//! What registering the rule costs a query it leaves: planning time (SQL to physical plan) and
//! planning + execution, DataFusion alone against DataFusion with `ArrowMetalConfig::default()`,
//! alternating run by run. Arguments: rows (default 1,000,000), rounds (default 200).

use std::sync::Arc;
use std::time::Instant;

use arrow::array::{ArrayRef, Float64Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::physical_plan::collect;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_arrowmetal::{session_context, ArrowMetalConfig, ArrowMetalRule};

#[tokio::main]
async fn main() {
    let a: Vec<usize> = std::env::args().skip(1).map(|x| x.parse().unwrap()).collect();
    let rows = a.first().copied().unwrap_or(1_000_000);
    let rounds = a.get(1).copied().unwrap_or(200);
    let k: Int32Array = (0..rows as i32).map(|i| Some(i.wrapping_mul(7_919) % 200)).collect();
    let x: Float64Array = (0..rows).map(|i| Some(i as f64 / 7.0)).collect();
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("x", DataType::Float64, false)]));
    let b = RecordBatch::try_new(schema.clone(), vec![Arc::new(k) as ArrayRef, Arc::new(x)]).unwrap();
    let parts = SessionConfig::new().target_partitions();
    let mut p = vec![Vec::new(); parts];
    let mut off = 0;
    let mut i = 0;
    while off < rows {
        let n = 8192.min(rows - off);
        p[i % parts].push(b.slice(off, n));
        off += n;
        i += 1;
    }
    let plain = SessionContext::new_with_config(SessionConfig::new());
    let rule = ArrowMetalRule::new(ArrowMetalConfig::default());
    let with = session_context(SessionConfig::new(), rule.clone());
    for c in [&plain, &with] {
        c.register_table("t", Arc::new(MemTable::try_new(schema.clone(), p.clone()).unwrap())).unwrap();
    }
    for sql in ["SELECT k, sum(x) AS s FROM t GROUP BY k", "SELECT k, min(x), max(x) FROM t GROUP BY k"] {
        let (mut plan_ms, mut all_ms) = ([Vec::new(), Vec::new()], [Vec::new(), Vec::new()]);
        for r in 0..rounds {
            for j in [r % 2, 1 - r % 2] {
                let c = if j == 0 { &plain } else { &with };
                rule.clear_report();
                let t = Instant::now();
                let plan = c.sql(sql).await.unwrap().create_physical_plan().await.unwrap();
                let tp = t.elapsed().as_secs_f64() * 1e3;
                collect(plan, c.task_ctx()).await.unwrap();
                let ta = t.elapsed().as_secs_f64() * 1e3;
                plan_ms[j].push(tp);
                all_ms[j].push(ta);
            }
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (v[v.len() / 2], v[0])
        };
        let (po, pw) = (med(&mut plan_ms[0]), med(&mut plan_ms[1]));
        let (ao, aw) = (med(&mut all_ms[0]), med(&mut all_ms[1]));
        println!("{sql}\n  rows {rows}: plan ms median/min alone {:.3}/{:.3}, with the rule {:.3}/{:.3}; plan+run alone {:.3}/{:.3}, with {:.3}/{:.3}", po.0, po.1, pw.0, pw.1, ao.0, ao.1, aw.0, aw.1);
        println!("  report: {}", rule.report().decisions().iter().map(|d| d.reason.clone()).collect::<Vec<_>>().join(" | "));
    }
}
