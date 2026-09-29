//! The rule's mechanics: the row threshold, what it does with unknown statistics, the config
//! switches, EXPLAIN, and plans where a replaced node feeds a partitioned join.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow::util::pretty::pretty_format_batches;
use datafusion::datasource::MemTable;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_arrowmetal::{session_context, AggregateChoice, ArrowMetalConfig, ArrowMetalRule};

/// Every shape the rule can translate, a replaced aggregate forced onto ArrowMetal.
fn gpu_all() -> ArrowMetalConfig {
    ArrowMetalConfig::all().with_min_rows(0).with_aggregate_choice(AggregateChoice::ArrowMetal)
}

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
    let rule = ArrowMetalRule::new(ArrowMetalConfig::default()); // min_rows 250,000
    let ctx = ctx_with(&rule, 5_000, 1).await;
    ctx.sql("SELECT * FROM t ORDER BY v").await.unwrap().collect().await.unwrap();
    let r = rule.report();
    assert_eq!(r.taken().count(), 0);
    let d = r.left().next().expect("the sort is reported");
    assert!(d.reason.contains("input rows 5000 (exact) vs min_rows 250000"), "{d}");
}

#[tokio::test]
async fn at_min_rows_is_taken() {
    let rule = ArrowMetalRule::new(ArrowMetalConfig::default().with_min_rows(5_000));
    let ctx = ctx_with(&rule, 5_000, 1).await;
    let sql = "SELECT * FROM t ORDER BY v, k, s";
    let got = pretty_format_batches(&ctx.sql(sql).await.unwrap().collect().await.unwrap()).unwrap().to_string();
    let want = pretty_format_batches(&plain(5_000, 1).await.sql(sql).await.unwrap().collect().await.unwrap()).unwrap().to_string();
    assert_eq!(got, want);
    let r = rule.report();
    assert_eq!(r.taken().count(), 1, "{r}");
    assert!(r.taken().any(|d| d.reason.contains("input rows 5000 (exact) vs min_rows 5000")), "{r}");
    assert_eq!(r.runtime_fallbacks().count(), 0, "{r}");
}

/// A filter's output row count is an estimate, so a sort above it is left by default and taken
/// with `accept_inexact`.
#[tokio::test]
async fn inexact_statistics_are_left_unless_accepted() {
    let sql = "SELECT * FROM t WHERE k = 3 ORDER BY v";
    let rule = ArrowMetalRule::new(ArrowMetalConfig::default().with_min_rows(10).with_filter(false));
    let ctx = ctx_with(&rule, 5_000, 1).await;
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let r = rule.report();
    assert_eq!(r.taken().count(), 0, "{r}");
    assert!(r.left().any(|d| d.reason.contains("estimate")), "{r}");

    let rule = ArrowMetalRule::new(ArrowMetalConfig::default().with_min_rows(10).with_filter(false).with_accept_inexact(true));
    let ctx = ctx_with(&rule, 5_000, 1).await;
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
    assert_eq!(rule.report().taken().count(), 1, "{}", rule.report());
}

#[tokio::test]
async fn config_switches_leave_the_node() {
    let rule = ArrowMetalRule::new(ArrowMetalConfig::all().with_min_rows(0).with_sort(false));
    let ctx = ctx_with(&rule, 100, 1).await;
    ctx.sql("SELECT * FROM t ORDER BY v").await.unwrap().collect().await.unwrap();
    let r = rule.report();
    assert_eq!(r.taken().count(), 0);
    assert!(r.left().any(|d| d.reason == "sort disabled in config"), "{r}");
}

#[tokio::test]
async fn unsupported_shapes_are_left_with_a_reason() {
    let rule = ArrowMetalRule::new(ArrowMetalConfig::all().with_min_rows(0));
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
/// the inner one is taken (sent as a group_by with no aggregates, which returns the distinct keys).
#[tokio::test]
async fn count_distinct_takes_the_inner_group_by() {
    let sql = "SELECT k, count(DISTINCT v) AS d FROM t GROUP BY k";
    for tp in [1, 4] {
        let rule = ArrowMetalRule::new(gpu_all().with_accept_inexact(true));
        let ctx = ctx_with(&rule, 20_000, tp).await;
        let got = sorted_text(&ctx, sql).await;
        assert_eq!(got, sorted_text(&plain(20_000, tp).await, sql).await, "tp={tp}");
        let r = rule.report();
        assert!(r.taken().count() >= 1, "{r}");
        assert_eq!(r.runtime_fallbacks().count(), 0, "{r}");
    }
}

/// The default config at 2,000,000 rows: the full sort is taken, top-k and the filter are left,
/// the group-by never runs on ArrowMetal, and the answers match DataFusion's. Slow in a debug
/// build, so ignored; run with `cargo test --test rule -- --ignored`.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn default_config_at_two_million_rows() {
    let n = 2_000_000;
    // The default take-list: the full sort is taken; top-k and the filter are left; the group-by
    // (17 groups, Float64 min/max) is left at plan time or handed back at run time by the measured
    // table, never run on ArrowMetal.
    for (sql, text, taken) in [
        ("SELECT s, v, k FROM t ORDER BY v DESC NULLS LAST, s, k", false, 1),
        ("SELECT k, v, s FROM t ORDER BY v DESC NULLS LAST, s LIMIT 20", false, 0),
        ("SELECT k, max(v), min(v), count(v), count(*) FROM t GROUP BY k", true, usize::MAX),
        ("SELECT k, s FROM t WHERE k = 3 AND s <> 's1'", true, 0),
    ] {
        let rule = ArrowMetalRule::new(ArrowMetalConfig::default());
        let ctx = ctx_with(&rule, n, 4).await;
        let (got, want) = if text {
            (sorted_text(&ctx, sql).await, sorted_text(&plain(n, 4).await, sql).await)
        } else {
            let run = |c: SessionContext| async move {
                let out = c.sql(sql).await.unwrap().collect().await.unwrap();
                let one = arrow::compute::concat_batches(&out[0].schema(), &out).unwrap();
                format!("{one:?}")
            };
            (run(ctx.clone()).await, run(plain(n, 4).await).await)
        };
        assert!(got == want, "{sql}: results differ");
        let r = rule.report();
        if taken == usize::MAX {
            assert_eq!(r.runtime_choices().filter(|d| d.taken).count(), 0, "{sql}\n{r}");
        } else {
            assert_eq!(r.taken().count(), taken, "{sql}\n{r}");
        }
        assert_eq!(r.runtime_fallbacks().count(), 0, "{sql}\n{r}");
        println!("{sql}\n{r}");
    }
}

/// What the default take-list leaves, with the reason in the report: top-k and filters by the
/// config, an aggregate by the measured table (it takes no aggregate shape at 1,000 rows).
#[tokio::test]
async fn default_take_list_leaves_topk_aggregates_and_filters() {
    let rule = ArrowMetalRule::new(ArrowMetalConfig::default().with_min_rows(0));
    let ctx = ctx_with(&rule, 1_000, 1).await;
    for (sql, why) in [
        ("SELECT * FROM t ORDER BY v LIMIT 5", "top-k (sort with fetch 5) disabled in config"),
        ("SELECT k, count(*) FROM t GROUP BY k", "the measured table takes"),
        ("SELECT * FROM t WHERE k = 3", "filter disabled in config"),
    ] {
        rule.clear_report();
        ctx.sql(sql).await.unwrap().collect().await.unwrap();
        let r = rule.report();
        assert_eq!(r.taken().count(), 0, "{sql}\n{r}");
        assert!(r.left().any(|d| d.reason.contains(why)), "{sql}\n{r}");
    }
    rule.clear_report();
    ctx.sql("SELECT * FROM t ORDER BY v").await.unwrap().collect().await.unwrap();
    assert_eq!(rule.report().taken().count(), 1, "{}", rule.report());
}

#[tokio::test]
async fn explain_shows_metal_exec() {
    let rule = ArrowMetalRule::new(gpu_all());
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
        let rule = ArrowMetalRule::new(gpu_all().with_sort(false));
        let ctx = ctx_with(&rule, 50_000, tp).await;
        let got = sorted_text(&ctx, sql).await;
        let want = sorted_text(&plain(50_000, tp).await, sql).await;
        assert_eq!(got, want, "tp={tp}");
        let taken = rule.report().taken().count();
        assert!(taken >= 2, "tp={tp}: {}", rule.report());
        assert_eq!(rule.report().runtime_fallbacks().count(), 0);
    }
}

/// The table split over `parts` MemTable partitions (DataFusion plans a merge over per-partition
/// sorts only when there are several).
async fn ctx_parts(rule: Option<&ArrowMetalRule>, n: usize, parts: usize, tp: usize) -> SessionContext {
    let config = SessionConfig::new().with_target_partitions(tp);
    let ctx = match rule {
        Some(r) => session_context(config, r.clone()),
        None => SessionContext::new_with_config(config),
    };
    let b = batch(n);
    let per = n.div_ceil(parts);
    let p: Vec<Vec<RecordBatch>> =
        (0..parts).map(|i| vec![b.slice(i * per, per.min(n - i * per))]).collect();
    ctx.register_table("t", Arc::new(MemTable::try_new(b.schema(), p).unwrap())).unwrap();
    ctx
}

/// An ORDER BY whose SELECT list reorders the columns plans as `SortPreservingMergeExec ->
/// ProjectionExec -> SortExec(preserve_partitioning)`. The rule replaces the merge and the sorts
/// with one MetalExec and keeps the projection above it; with and without LIMIT, the rows match.
#[tokio::test]
async fn order_by_through_a_projection_is_taken() {
    for sql in [
        "SELECT s, v, k FROM t ORDER BY k, v DESC, s",
        "SELECT s, v, k FROM t ORDER BY k DESC, v NULLS FIRST, s LIMIT 37",
        "SELECT v * 2 AS w, k, s FROM t ORDER BY k, s, w",
    ] {
        for (parts, tp) in [(3, 4), (1, 1)] {
            let rule = ArrowMetalRule::new(ArrowMetalConfig::all().with_min_rows(0));
            let ctx = ctx_parts(Some(&rule), 20_000, parts, tp).await;
            let run = |c: SessionContext| async move {
                pretty_format_batches(&c.sql(sql).await.unwrap().collect().await.unwrap()).unwrap().to_string()
            };
            let got = run(ctx).await;
            let want = run(ctx_parts(None, 20_000, parts, tp).await).await;
            let r = rule.report();
            // `v * 2` is computed in the projection, so its sort key is an expression: left.
            if sql.contains("v * 2") {
                assert_eq!(got, want, "{sql} parts={parts}");
                continue;
            }
            assert_eq!(got, want, "{sql} parts={parts}\n{r}");
            assert_eq!(r.taken().count(), 1, "{sql} parts={parts}\n{r}");
            assert_eq!(r.runtime_fallbacks().count(), 0, "{sql}\n{r}");
            if parts > 1 {
                assert!(r.taken().any(|d| d.reason.contains("projection kept above it")), "{sql}\n{r}");
            }
        }
    }
}

/// A replaced aggregate with only projections above it keeps MetalExec's single output
/// partition (no hash re-partitioning of the result); below a join it is still re-partitioned
/// (`replaced_aggregates_under_a_partitioned_join`).
#[tokio::test]
async fn top_level_aggregate_is_not_repartitioned() {
    // Exact aggregates only (a float sum differs in its last bits with the summation order).
    let sql = "SELECT k, max(v) AS top, count(v) AS c, count(*) AS n FROM t GROUP BY k";
    let rule = ArrowMetalRule::new(gpu_all());
    let ctx = ctx_parts(Some(&rule), 20_000, 3, 4).await;
    let got = sorted_text(&ctx, sql).await;
    // sorted_text wraps the query in an outer ORDER BY, so plan the bare query for the report.
    rule.clear_report();
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let r = rule.report();
    assert!(r.taken().any(|d| d.reason.contains("kept at one partition")), "{r}");
    assert_eq!(got, sorted_text(&ctx_parts(None, 20_000, 3, 4).await, sql).await);
}

/// A table whose i64 key `k` holds `groups` distinct values over `n` rows (dealt round-robin, so
/// every batch holds many groups), a Float64 `v` and an int64 `q`, in `parts` partitions.
async fn grouped_ctx(rule: Option<&ArrowMetalRule>, n: usize, groups: i64, parts: usize) -> SessionContext {
    let config = SessionConfig::new().with_target_partitions(4);
    let ctx = match rule {
        Some(r) => session_context(config, r.clone()),
        None => SessionContext::new_with_config(config),
    };
    let k: Int64Array = (0..n as i64).map(|i| Some((i * 7_919) % groups)).collect();
    let v: Float64Array = (0..n).map(|i| Some((i % 1_000) as f64 / 8.0)).collect();
    let q: Int64Array = (0..n as i64).map(|i| Some(i % 977)).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("v", DataType::Float64, true),
        Field::new("q", DataType::Int64, true),
    ]));
    let b = RecordBatch::try_new(schema.clone(), vec![Arc::new(k) as ArrayRef, Arc::new(v), Arc::new(q)]).unwrap();
    let per = n.div_ceil(parts);
    let p: Vec<Vec<RecordBatch>> = (0..parts).map(|i| vec![b.slice(i * per, per.min(n - i * per))]).collect();
    ctx.register_table("g", Arc::new(MemTable::try_new(schema, p).unwrap())).unwrap();
    ctx
}

/// The forced hand-back (`AggregateChoice::DataFusion`) gives DataFusion's answer, over one and
/// several input partitions, and records its run-time choice in the report.
#[tokio::test]
async fn forced_hand_back_gives_datafusions_answer() {
    let sql = "SELECT k, count(*) AS n, sum(q) AS s, min(q) AS lo, max(q) AS hi FROM g GROUP BY k";
    for parts in [1, 3] {
        let rule = ArrowMetalRule::new(gpu_all().with_aggregate_choice(AggregateChoice::DataFusion));
        let ctx = grouped_ctx(Some(&rule), 30_000, 5_000, parts).await;
        let got = sorted_text(&ctx, sql).await;
        let want = sorted_text(&grouped_ctx(None, 30_000, 5_000, parts).await, sql).await;
        assert_eq!(got, want, "parts={parts}");
        let r = rule.report();
        assert!(r.runtime_choices().count() >= 1, "{r}");
        assert!(r.runtime_choices().all(|d| !d.taken && d.reason.contains("aggregate_choice is DataFusion")), "{r}");
        assert_eq!(r.runtime_fallbacks().count(), 0, "{r}");
    }
}

/// The default choice decides at run time from the probe's estimate. With the table looked up at
/// 50,000,000 rows (`table_rows`): the decision follows the table for the estimated bucket, the
/// report carries the estimate, and the answer is DataFusion's either way.
#[tokio::test]
async fn measured_choice_records_the_estimate_and_matches() {
    let sql = "SELECT k, count(*) AS n FROM g GROUP BY k";
    for (groups, n) in [(150i64, 40_000usize), (60_000, 200_000)] {
        let rule = ArrowMetalRule::new(ArrowMetalConfig::default().with_min_rows(0).with_table_rows(Some(50_000_000)));
        let ctx = grouped_ctx(Some(&rule), n, groups, 3).await;
        let got = sorted_text(&ctx, sql).await;
        let want = sorted_text(&grouped_ctx(None, n, groups, 3).await, sql).await;
        assert_eq!(got, want, "groups={groups}");
        let r = rule.report();
        assert_eq!(r.runtime_fallbacks().count(), 0, "{r}");
        for d in r.runtime_choices() {
            let e = d.groups.as_ref().unwrap().estimate.expect("an estimate");
            assert!(e.low <= groups as u64 * 5 / 4 && e.high >= groups as u64 * 3 / 4, "groups={groups}: {d}");
            assert!(d.reason.contains("estimated") || d.reason.contains("counted"), "{d}");
        }
        println!("groups={groups}:\n{r}");
    }
}

/// A session whose memory pool holds 3 MiB: the MetalExec's reservation for its input is refused,
/// and the node is handed back to DataFusion (whose sort spills) with the reason in the report;
/// the answers are DataFusion's.
#[tokio::test(flavor = "multi_thread")]
async fn a_refused_memory_reservation_hands_the_node_back() {
    use datafusion::execution::memory_pool::GreedyMemoryPool;
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;
    use datafusion::execution::session_state::SessionStateBuilder;
    let n = 200_000;
    for sql in [
        "SELECT k, count(*) AS n, max(v) AS top FROM t GROUP BY k",
        "SELECT k, v, s FROM t ORDER BY v DESC NULLS LAST, k, s",
    ] {
        let rule = ArrowMetalRule::new(gpu_all());
        let rt = RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(GreedyMemoryPool::new(3 << 20)))
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
        // Four batches of their own (not slices of one: DataFusion's sort reserves a slice at the
        // size of the buffer it shares).
        let p: Vec<Vec<RecordBatch>> = (0..4).map(|_| vec![batch(n / 4)]).collect();
        let schema = p[0][0].schema();
        ctx.register_table("t", Arc::new(MemTable::try_new(Arc::clone(&schema), p.clone()).unwrap())).unwrap();
        let reference = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));
        reference.register_table("t", Arc::new(MemTable::try_new(schema, p).unwrap())).unwrap();
        let text = |out: Vec<RecordBatch>| pretty_format_batches(&out).unwrap().to_string();
        let grouped = sql.contains("GROUP BY");
        let got = if grouped {
            sorted_text(&ctx, sql).await
        } else {
            text(ctx.sql(sql).await.unwrap().collect().await.unwrap())
        };
        let want = if grouped {
            sorted_text(&reference, sql).await
        } else {
            text(reference.sql(sql).await.unwrap().collect().await.unwrap())
        };
        assert!(got == want, "{sql}: results differ");
        let r = rule.report();
        assert!(
            r.runtime_fallbacks().any(|d| d.reason.contains("memory pool refused")),
            "{sql}: expected a memory hand-back\n{r}"
        );
    }
}
