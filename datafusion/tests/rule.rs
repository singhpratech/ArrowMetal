//! The rule's mechanics: the row threshold, what it does with unknown statistics, the config
//! switches, EXPLAIN, and plans where a replaced node feeds a partitioned join.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow::util::pretty::pretty_format_batches;
use datafusion::datasource::MemTable;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_arrowmetal::{session_context, ArrowMetalConfig, ArrowMetalRule};

fn batch(n: usize) -> RecordBatch {
    let k: Int64Array = (0..n).map(|i| Some((i % 17) as i64)).collect();
    let v: Float64Array = (0..n).map(|i| if i % 11 == 0 { None } else { Some(i as f64 / 3.0) }).collect();
    let s: StringArray = (0..n).map(|i| Some(format!("s{}", i % 5))).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("v", DataType::Float64, true),
        Field::new("s", DataType::Utf8, true),
    ]));
    RecordBatch::try_new(schema, vec![Arc::new(k) as ArrayRef, Arc::new(v), Arc::new(s)]).unwrap()
}

async fn ctx_with(rule: &ArrowMetalRule, n: usize, tp: usize) -> SessionContext {
    let ctx = session_context(SessionConfig::new().with_target_partitions(tp), rule.clone());
    let b = batch(n);
    ctx.register_table("t", Arc::new(MemTable::try_new(b.schema(), vec![vec![b]]).unwrap())).unwrap();
    ctx
}

async fn plain(n: usize, tp: usize) -> SessionContext {
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(tp));
    let b = batch(n);
    ctx.register_table("t", Arc::new(MemTable::try_new(b.schema(), vec![vec![b]]).unwrap())).unwrap();
    ctx
}

async fn sorted_text(ctx: &SessionContext, sql: &str) -> String {
    let out = ctx.sql(&format!("SELECT * FROM ({sql}) ORDER BY 1, 2")).await.unwrap().collect().await.unwrap();
    pretty_format_batches(&out).unwrap().to_string()
}

#[tokio::test]
async fn below_min_rows_is_left_with_the_count_in_the_reason() {
    let rule = ArrowMetalRule::new(ArrowMetalConfig::default()); // min_rows 1,000,000
    let ctx = ctx_with(&rule, 5_000, 1).await;
    ctx.sql("SELECT * FROM t ORDER BY v").await.unwrap().collect().await.unwrap();
    let r = rule.report();
    assert_eq!(r.taken().count(), 0);
    let d = r.left().next().expect("the sort is reported");
    assert!(d.reason.contains("input rows 5000 (exact) vs min_rows 1000000"), "{d}");
}

#[tokio::test]
async fn at_min_rows_is_taken() {
    let rule = ArrowMetalRule::new(ArrowMetalConfig { min_rows: 5_000, ..Default::default() });
    let ctx = ctx_with(&rule, 5_000, 1).await;
    ctx.sql("SELECT * FROM t ORDER BY v").await.unwrap().collect().await.unwrap();
    assert_eq!(rule.report().taken().count(), 1, "{}", rule.report());
}

/// A filter's output row count is an estimate, so a sort above it is left by default and taken
/// with `accept_inexact`.
#[tokio::test]
async fn inexact_statistics_are_left_unless_accepted() {
    let sql = "SELECT * FROM t WHERE k = 3 ORDER BY v";
    let rule = ArrowMetalRule::new(ArrowMetalConfig { min_rows: 10, filter: false, ..Default::default() });
    let ctx = ctx_with(&rule, 5_000, 1).await;
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let r = rule.report();
    assert_eq!(r.taken().count(), 0, "{r}");
    assert!(r.left().any(|d| d.reason.contains("estimate")), "{r}");

    let rule = ArrowMetalRule::new(ArrowMetalConfig {
        min_rows: 10,
        filter: false,
        accept_inexact: true,
        ..Default::default()
    });
    let ctx = ctx_with(&rule, 5_000, 1).await;
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
    assert_eq!(rule.report().taken().count(), 1, "{}", rule.report());
}

#[tokio::test]
async fn config_switches_leave_the_node() {
    let rule = ArrowMetalRule::new(ArrowMetalConfig { min_rows: 0, sort: false, ..Default::default() });
    let ctx = ctx_with(&rule, 100, 1).await;
    ctx.sql("SELECT * FROM t ORDER BY v").await.unwrap().collect().await.unwrap();
    let r = rule.report();
    assert_eq!(r.taken().count(), 0);
    assert!(r.left().any(|d| d.reason == "sort disabled in config"), "{r}");
}

#[tokio::test]
async fn unsupported_shapes_are_left_with_a_reason() {
    let rule = ArrowMetalRule::new(ArrowMetalConfig { min_rows: 0, ..Default::default() });
    let ctx = ctx_with(&rule, 100, 1).await;
    for (sql, why) in [
        ("SELECT * FROM t ORDER BY v + 1", "is an expression"),
        ("SELECT k, array_agg(v) FROM t GROUP BY k", "aggregate function array_agg"),
        ("SELECT k, sum(v) FILTER (WHERE v > 1) FROM t GROUP BY k", "FILTER"),
        ("SELECT sum(v) FROM t", "whole-table aggregate"),
        ("SELECT k, median(v) FROM t GROUP BY k", "aggregate function median"),
        ("SELECT * FROM t WHERE s < 'b'", "string comparison"),
    ] {
        rule.clear_report();
        ctx.sql(sql).await.unwrap().collect().await.unwrap();
        let r = rule.report();
        assert!(r.left().any(|d| d.reason.contains(why)), "{sql}: expected a node left for '{why}'\n{r}");
    }
}

/// count(DISTINCT v) plans as an inner GROUP BY k, v with no aggregates under an outer count;
/// the inner one is taken (a dummy row count stands in for the missing aggregate).
#[tokio::test]
async fn count_distinct_takes_the_inner_group_by() {
    let sql = "SELECT k, count(DISTINCT v) AS d FROM t GROUP BY k";
    for tp in [1, 4] {
        let rule = ArrowMetalRule::new(ArrowMetalConfig { min_rows: 0, accept_inexact: true, ..Default::default() });
        let ctx = ctx_with(&rule, 20_000, tp).await;
        let got = sorted_text(&ctx, sql).await;
        assert_eq!(got, sorted_text(&plain(20_000, tp).await, sql).await, "tp={tp}");
        let r = rule.report();
        assert!(r.taken().count() >= 1, "{r}");
        assert_eq!(r.runtime_fallbacks().count(), 0, "{r}");
    }
}

/// The default config (min_rows 1,000,000, exact statistics only) at 2,000,000 rows: the sort,
/// the group-by and the filter are taken and the answers match. Slow in a debug build, so ignored;
/// run with `cargo test --test rule -- --ignored`.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn default_config_at_two_million_rows() {
    let n = 2_000_000;
    for (sql, text) in [
        ("SELECT k, v, s FROM t ORDER BY v DESC NULLS LAST, s LIMIT 20", false),
        ("SELECT k, max(v), min(v), count(v), count(*) FROM t GROUP BY k", true),
        ("SELECT k, s FROM t WHERE k = 3 AND s <> 's1'", true),
    ] {
        let rule = ArrowMetalRule::new(ArrowMetalConfig::default());
        let ctx = ctx_with(&rule, n, 4).await;
        let (got, want) = if text {
            (sorted_text(&ctx, sql).await, sorted_text(&plain(n, 4).await, sql).await)
        } else {
            let run = |c: SessionContext| async move {
                pretty_format_batches(&c.sql(sql).await.unwrap().collect().await.unwrap()).unwrap().to_string()
            };
            (run(ctx.clone()).await, run(plain(n, 4).await).await)
        };
        assert_eq!(got, want, "{sql}");
        let r = rule.report();
        assert_eq!(r.taken().count(), 1, "{sql}\n{r}");
        assert_eq!(r.runtime_fallbacks().count(), 0, "{sql}\n{r}");
        println!("{sql}\n{r}");
    }
}

#[tokio::test]
async fn explain_shows_metal_exec() {
    let rule = ArrowMetalRule::new(ArrowMetalConfig { min_rows: 0, ..Default::default() });
    let ctx = ctx_with(&rule, 100, 4).await;
    let out = ctx
        .sql("EXPLAIN SELECT k, sum(v) FROM t GROUP BY k ORDER BY k")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let text = pretty_format_batches(&out).unwrap().to_string();
    assert!(text.contains("MetalExec: group_by=[k], aggr=[Sum(v)]"), "{text}");
}

/// Two replaced aggregates feed a partitioned hash join; the rule re-partitions each MetalExec's
/// output by the same hash so the join's inputs keep their distribution.
#[tokio::test]
async fn replaced_aggregates_under_a_partitioned_join() {
    let sql = "SELECT a.k, a.top, b.n FROM \
               (SELECT k, max(v) AS top FROM t GROUP BY k) a \
               JOIN (SELECT k, count(*) AS n FROM t GROUP BY k) b ON a.k = b.k";
    for tp in [1, 4] {
        let rule = ArrowMetalRule::new(ArrowMetalConfig { min_rows: 0, sort: false, ..Default::default() });
        let ctx = ctx_with(&rule, 50_000, tp).await;
        let got = sorted_text(&ctx, sql).await;
        let want = sorted_text(&plain(50_000, tp).await, sql).await;
        assert_eq!(got, want, "tp={tp}");
        let taken = rule.report().taken().count();
        assert!(taken >= 2, "tp={tp}: {}", rule.report());
        assert_eq!(rule.report().runtime_fallbacks().count(), 0);
    }
}
