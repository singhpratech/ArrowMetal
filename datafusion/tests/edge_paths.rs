//! Paths the other tests do not reach: the ordering a replaced aggregate declares,
//! a join's result after the query ends, the hand-back when the result's reservation is refused, and
//! a top-k under an OFFSET.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow::util::pretty::pretty_format_batches;
use datafusion::datasource::MemTable;
use datafusion::physical_plan::{collect, displayable};
use datafusion::prelude::{col, SessionConfig, SessionContext};
use datafusion_arrowmetal::{session_context, AggregateChoice, ArrowMetalConfig, ArrowMetalRule, JoinChoice};

fn gpu_all() -> ArrowMetalConfig {
    ArrowMetalConfig::all().with_min_rows(0).with_aggregate_choice(AggregateChoice::ArrowMetal)
}

fn text(out: &[RecordBatch]) -> String {
    pretty_format_batches(out).unwrap().to_string()
}

/// A MemTable declared sorted by `k` descending: DataFusion's aggregate over it is ordered by `k`,
/// so `ORDER BY k DESC` above it needs no sort. The replaced aggregate keeps the aggregate's
/// equivalence properties; the answer must still come out in `k DESC` order.
#[tokio::test(flavor = "multi_thread")]
async fn aggregate_over_a_declared_sort_order_keeps_it() {
    let n = 300_000usize;
    let k: Int64Array = (0..n).map(|i| Some(((n - i) / 3) as i64)).collect();
    let v: Int64Array = (0..n).map(|i| Some((i % 97) as i64)).collect();
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false), Field::new("v", DataType::Int64, false)]));
    let b = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(k) as ArrayRef, Arc::new(v)]).unwrap();
    let mem = || {
        MemTable::try_new(Arc::clone(&schema), vec![vec![b.clone()]])
            .unwrap()
            .with_sort_order(vec![vec![col("k").sort(false, false)]])
    };
    let sql = "SELECT k, count(*) AS n, max(v) AS m FROM t GROUP BY k ORDER BY k DESC";
    let rule = ArrowMetalRule::new(gpu_all());
    let ctx = session_context(SessionConfig::new().with_target_partitions(1), rule.clone());
    ctx.register_table("t", Arc::new(mem())).unwrap();
    let df = ctx.sql(sql).await.unwrap();
    let plan = df.clone().create_physical_plan().await.unwrap();
    let shown = displayable(plan.as_ref()).indent(true).to_string();
    let got = text(&df.collect().await.unwrap());
    let reference = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(1));
    reference.register_table("t", Arc::new(mem())).unwrap();
    let want = text(&reference.sql(sql).await.unwrap().collect().await.unwrap());
    let r = rule.report();
    println!("{r}\n{shown}");
    assert!(got == want, "rows out of the declared order\nplan:\n{shown}\nfirst rows got:\n{}\nwant:\n{}",
        got.lines().take(8).collect::<Vec<_>>().join("\n"), want.lines().take(8).collect::<Vec<_>>().join("\n"));
}

fn join_tables(n_probe: usize, n_build: usize) -> (RecordBatch, RecordBatch) {
    let k: Int64Array = (0..n_probe).map(|i| Some(((i as u64).wrapping_mul(2_654_435_761) % (2 * n_build as u64)) as i64)).collect();
    let v: Float64Array = (0..n_probe).map(|i| if i % 13 == 0 { None } else { Some(i as f64 / 7.0) }).collect();
    let p = RecordBatch::try_from_iter([("k", Arc::new(k) as ArrayRef), ("v", Arc::new(v) as ArrayRef)]).unwrap();
    let bk: Int64Array = (0..n_build).map(|i| if i % 101 == 0 { None } else { Some(2 * i as i64) }).collect();
    let w: Float64Array = (0..n_build).map(|i| Some(i as f64 * 0.5)).collect();
    let b = RecordBatch::try_from_iter([("k", Arc::new(bk) as ArrayRef), ("w", Arc::new(w) as ArrayRef)]).unwrap();
    (p, b)
}

fn split(b: &RecordBatch, parts: usize) -> Vec<Vec<RecordBatch>> {
    let n = b.num_rows();
    let per = n.div_ceil(parts);
    (0..parts).map(|i| vec![b.slice(i * per, per.min(n - i * per))]).collect()
}

fn sorted_lines(out: &[RecordBatch]) -> Vec<String> {
    let t = text(out);
    let mut lines: Vec<String> = t.lines().filter(|l| l.starts_with('|')).skip(1).map(String::from).collect();
    lines.sort();
    lines
}

/// A replaced join's physical plan collected twice: both answers are DataFusion's, and once a
/// collect has returned, the session's memory pool holds nothing for the finished query.
#[tokio::test(flavor = "multi_thread")]
async fn a_join_plan_collected_twice_and_its_memory_after_the_query() {
    let rule = ArrowMetalRule::new(ArrowMetalConfig::all().with_min_rows(0).with_join_choice(JoinChoice::ArrowMetal));
    let tp = 4;
    let ctx = session_context(SessionConfig::new().with_target_partitions(tp), rule.clone());
    let (p, b) = join_tables(300_000, 200_000);
    ctx.register_table("p", Arc::new(MemTable::try_new(p.schema(), split(&p, tp)).unwrap())).unwrap();
    ctx.register_table("b", Arc::new(MemTable::try_new(b.schema(), split(&b, tp)).unwrap())).unwrap();
    let sql = "SELECT p.k, p.v, b.w FROM p JOIN b ON p.k = b.k";
    let reference = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(tp));
    reference.register_table("p", Arc::new(MemTable::try_new(p.schema(), split(&p, tp)).unwrap())).unwrap();
    reference.register_table("b", Arc::new(MemTable::try_new(b.schema(), split(&b, tp)).unwrap())).unwrap();
    let want = sorted_lines(&reference.sql(sql).await.unwrap().collect().await.unwrap());

    let df = ctx.sql(sql).await.unwrap();
    let task = Arc::new(df.task_ctx());
    let plan = df.create_physical_plan().await.unwrap();
    let shown = displayable(plan.as_ref()).indent(true).to_string();
    assert!(shown.contains("MetalExec: join="), "{shown}");
    let first = sorted_lines(&collect(Arc::clone(&plan), Arc::clone(&task)).await.unwrap());
    let held_after_first = ctx.runtime_env().memory_pool.reserved();
    let second = sorted_lines(&collect(Arc::clone(&plan), Arc::clone(&task)).await.unwrap());
    assert!(first == want, "first collect differs");
    assert!(second == want, "second collect differs");
    assert_eq!(held_after_first, 0, "bytes still reserved after the query returned, with the plan alive\n{shown}");
}

fn batch(n: usize, seed: usize) -> RecordBatch {
    let k: Int64Array = (0..n).map(|i| Some(((i + seed) % 17) as i64)).collect();
    let v: Float64Array = (0..n).map(|i| if (i + seed) % 11 == 0 { None } else { Some((i + seed) as f64 / 3.0) }).collect();
    let s: StringArray = (0..n).map(|i| Some(format!("s{}", (i + seed) % 5))).collect();
    RecordBatch::try_from_iter([("k", Arc::new(k) as ArrayRef), ("v", Arc::new(v) as ArrayRef), ("s", Arc::new(s) as ArrayRef)]).unwrap()
}

/// A pool that holds the collected input but not the input and the result together: the result's
/// reservation is refused, the sort is handed back, and the answer is DataFusion's.
#[tokio::test(flavor = "multi_thread")]
async fn a_refused_result_reservation_hands_the_sort_back() {
    use datafusion::execution::memory_pool::GreedyMemoryPool;
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;
    use datafusion::execution::session_state::SessionStateBuilder;
    let parts: Vec<Vec<RecordBatch>> = (0..4).map(|i| vec![batch(50_000, i * 50_000)]).collect();
    let schema = parts[0][0].schema();
    let bytes: usize = parts.iter().flatten().map(|b| b.get_array_memory_size()).sum();
    let rule = ArrowMetalRule::new(gpu_all());
    let rt = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::new(GreedyMemoryPool::new(bytes + bytes / 2)))
        .build_arc()
        .unwrap();
    let state = datafusion_arrowmetal::with_arrowmetal(
        SessionStateBuilder::new()
            .with_config(SessionConfig::new().with_target_partitions(4).with_sort_spill_reservation_bytes(256 << 10))
            .with_runtime_env(rt)
            .with_default_features(),
        rule.clone(),
    )
    .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_table("t", Arc::new(MemTable::try_new(Arc::clone(&schema), parts.clone()).unwrap())).unwrap();
    let reference = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));
    reference.register_table("t", Arc::new(MemTable::try_new(schema, parts).unwrap())).unwrap();
    let sql = "SELECT k, v, s FROM t ORDER BY v DESC NULLS LAST, k, s";
    let got = text(&ctx.sql(sql).await.unwrap().collect().await.unwrap());
    let want = text(&reference.sql(sql).await.unwrap().collect().await.unwrap());
    assert!(got == want, "results differ");
    let r = rule.report();
    assert!(
        r.runtime_fallbacks().any(|d| d.reason.contains("reservation for the result")),
        "expected a hand-back of the result\n{r}"
    );
}

/// One partition: an ORDER BY over a right join whose probe side DataFusion sorts below the join
/// (as for the inner join in tests/rule.rs). The replaced right join keeps the probe order, with
/// the unmatched probe rows in place.
#[tokio::test(flavor = "multi_thread")]
async fn an_ordered_right_join_keeps_the_probe_order() {
    let rule = ArrowMetalRule::new(ArrowMetalConfig::all().with_min_rows(0).with_join_choice(JoinChoice::ArrowMetal).with_sort(false));
    let ctx = session_context(SessionConfig::new().with_target_partitions(1), rule.clone());
    let reference = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(1));
    let (p, b) = join_tables(300_000, 200_000);
    for c in [&ctx, &reference] {
        c.register_table("p", Arc::new(MemTable::try_new(p.schema(), vec![vec![p.clone()]]).unwrap())).unwrap();
        c.register_table("b", Arc::new(MemTable::try_new(b.schema(), vec![vec![b.clone()]]).unwrap())).unwrap();
    }
    for sql in [
        "SELECT * FROM (SELECT p.k, p.v, b.w FROM b RIGHT JOIN p ON p.k = b.k) ORDER BY 1, 2",
        "SELECT * FROM (SELECT p.k, p.v, b.w FROM p LEFT JOIN b ON p.k = b.k) ORDER BY 1, 2",
    ] {
        rule.clear_report();
        let df = ctx.sql(sql).await.unwrap();
        let shown = displayable(df.clone().create_physical_plan().await.unwrap().as_ref()).indent(true).to_string();
        let got = text(&df.collect().await.unwrap());
        let want = text(&reference.sql(sql).await.unwrap().collect().await.unwrap());
        assert!(got == want, "{sql}: results differ\n{shown}");
        println!("{sql}\n{shown}\n{}", rule.report());
    }
}

/// `ORDER BY ... LIMIT n OFFSET m` with top-k on: the sort's fetch is n + m, the skip stays above.
#[tokio::test(flavor = "multi_thread")]
async fn topk_under_an_offset() {
    let parts: Vec<Vec<RecordBatch>> = (0..4).map(|i| vec![batch(50_000, i * 50_000)]).collect();
    let schema = parts[0][0].schema();
    let rule = ArrowMetalRule::new(gpu_all());
    let ctx = session_context(SessionConfig::new().with_target_partitions(4), rule.clone());
    ctx.register_table("t", Arc::new(MemTable::try_new(Arc::clone(&schema), parts.clone()).unwrap())).unwrap();
    let reference = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));
    reference.register_table("t", Arc::new(MemTable::try_new(schema, parts).unwrap())).unwrap();
    for sql in [
        "SELECT k, v, s FROM t ORDER BY v DESC NULLS LAST, k, s LIMIT 10 OFFSET 5",
        "SELECT k, v, s FROM t ORDER BY v NULLS FIRST, k, s LIMIT 7 OFFSET 20000",
    ] {
        rule.clear_report();
        let got = text(&ctx.sql(sql).await.unwrap().collect().await.unwrap());
        let want = text(&reference.sql(sql).await.unwrap().collect().await.unwrap());
        assert!(got == want, "{sql}: results differ\n{got}\n{want}");
        let r = rule.report();
        assert!(r.taken().count() >= 1, "{sql}\n{r}");
        assert_eq!(r.runtime_fallbacks().count(), 0, "{sql}\n{r}");
    }
}
