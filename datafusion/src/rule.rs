//! `ArrowMetalRule`: the `PhysicalOptimizerRule` that swaps supported nodes for `MetalExec`.

use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use datafusion::common::config::ConfigOptions;
use datafusion::common::JoinType;
use datafusion::common::stats::Precision;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::Result;
use datafusion::physical_optimizer::sanity_checker::SanityCheckPlan;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::joins::HashJoinExec;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::{
    displayable, Distribution, ExecutionPlan, ExecutionPlanProperties, Partitioning, StatisticsArgs,
    StatisticsContext,
};

use crate::exec::{MetalExec, MetalOp};
use crate::translate;

/// When the rule takes a node.
///
/// [`Default`] is the measured take-list; [`ArrowMetalConfig::all`] takes every shape the rule can
/// translate (what the differential grid and the benchmark use).
///
/// The fields are public to read and to set on a value (`let mut c = ArrowMetalConfig::all();
/// c.min_rows = 0;`); outside this crate a config is built from [`Default`] or
/// [`ArrowMetalConfig::all`] and the `with_*` methods.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ArrowMetalConfig {
    /// Take a node only when its input has at least this many rows.
    pub min_rows: usize,
    /// Use an inexact (estimated) row count as if it were exact. Default: no.
    pub accept_inexact: bool,
    /// Take a node whose input row count is unknown. Default: no (leave it).
    pub take_when_unknown: bool,
    /// Full sorts: `ORDER BY` without `LIMIT`.
    pub sort: bool,
    /// Top-k: `ORDER BY ... LIMIT` (a sort with a fetch).
    pub topk: bool,
    /// Aggregates (GROUP BY, DISTINCT). Who runs a replaced one is `aggregate_choice`.
    pub aggregate: bool,
    /// Filters: a `FilterExec` whose predicate translates.
    pub filter: bool,
    /// Who runs a replaced aggregate. Default: [`AggregateChoice::Measured`].
    pub aggregate_choice: AggregateChoice,
    /// Under [`AggregateChoice::Measured`], look the measured table up at this row count instead of
    /// the input's, at plan time and at run time (tests and experiments; default `None`).
    pub table_rows: Option<usize>,
    /// Hash joins (`HashJoinExec`: inner, left, right on equal keys). Which ones is `join_choice`.
    pub join: bool,
    /// Which translatable joins are replaced. Default: [`JoinChoice::Measured`].
    pub join_choice: JoinChoice,
    /// The report keeps the decisions of the last this many plans the rule optimized (an EXPLAIN
    /// and each execution plan count one each), with the run-time decisions of their nodes.
    /// Default 64; 0 keeps one.
    pub report_plans: usize,
}

/// Who runs an aggregate the rule replaced.
///
/// The rule sees the input's row count at plan time but not its group count, and the same
/// aggregate SQL is faster on ArrowMetal at some group counts and slower at others. So a replaced
/// aggregate is decided when it runs: `MetalExec` collects its input, and under `Measured` it
/// estimates the group count from a sample of the key columns on the CPU and looks the shape up in
/// the measured table (`src/agg_table.rs`). It runs on ArrowMetal only where the sweep measured it
/// ahead; otherwise it hands the node back to DataFusion's own operators over the same batches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AggregateChoice {
    /// Estimate the group count and look it up in the measured table (the default). At plan time
    /// the rule replaces an aggregate only when the table takes its shape at some group count at
    /// the input's exact row count.
    #[default]
    Measured,
    /// Always run a replaced aggregate on ArrowMetal (the benchmark's crossover sweep, tests).
    ArrowMetal,
    /// Always hand a replaced aggregate back to DataFusion (measures the hand-back itself, tests).
    DataFusion,
}

/// Which hash joins the rule replaces (decided at plan time, from both inputs' exact row counts).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum JoinChoice {
    /// The joins the measured table (`src/join_table.rs`) takes: by join type, key type, the
    /// build (left) input's rows and the probe (right) input's rows (the default).
    #[default]
    Measured,
    /// Every join the rule can translate, from `min_rows` (the benchmark's sweep, tests).
    ArrowMetal,
}

/// The default take-list, from the rule on/off measurement against DataFusion 55.1 on an M4 Max
/// (16 partitions; MemTables of 8192-row batches and of one batch per partition; this crate with
/// one totalOrder key per ORDER BY key and the chunked import;
/// `datafusion/results/datafusion_rule_2026-09-29.csv`, rows `crate = now`; DataFusion alone /
/// DataFusion with the rule, wall, best of 5, both layouts):
///
/// | shape | 1M | 10M | 50M | default |
/// |---|---:|---:|---:|---|
/// | ORDER BY int64 / Float64 / String / Float32 key, 3 columns | 5.9x - 12.8x | 19.2x - 28.7x | 18.9x - 27.2x | taken |
/// | the same over DataFusion's Parquet scan (snappy, zstd) | | 13.6x, 10.8x | 16.5x, 13.8x | taken |
/// | ORDER BY ... LIMIT 100 (int64, Float64 DESC, Float32 DESC) | 0.14x - 0.49x | 0.24x - 0.41x | 0.17x - 0.29x | left |
/// | WHERE + whole-table sum/count (the filter is what is taken) | 0.56x - 0.59x | 0.62x - 0.72x | 0.74x | left |
/// | Parquet WHERE + GROUP BY | | 0.52x - 0.53x | 0.54x - 0.55x | left |
///
/// Full sorts at smaller inputs (every key type, both layouts): 100k rows 1.34x - 2.63x, 250k
/// 2.64x - 4.80x, 500k 4.32x - 7.49x; the 1x crossover is below 100k for every key, and 250k is
/// the first measured size at which every key type is at or above 2.6x; hence `min_rows` 250,000.
///
/// Top-k is left: DataFusion's TopK answers LIMIT 100 over 50M rows in 4.6 - 6.5 ms; the rule's
/// GPU top-k takes 20 - 29 ms there, of which collecting the input stream alone is 6.5 - 8 ms.
///
/// Aggregates (GROUP BY, DISTINCT) are decided per shape and group count
/// ([`AggregateChoice::Measured`]): the same SQL is ahead of DataFusion at some group counts and
/// behind at others, and the rule sees the row count but not the group count (DataFusion's
/// MemTable and Parquet statistics carry no distinct counts). The rule replaces an aggregate when
/// the measured table (`src/agg_table.rs`, generated by `scripts/groupby_table.py` from the sweep
/// CSV it names) takes its shape at some group count at the input's exact row count; the
/// `MetalExec` then estimates the group count from a sample of the keys and runs on ArrowMetal only
/// at a bucket the table takes, handing the node back to DataFusion otherwise. Hash joins are
/// replaced where the measured join table (`src/join_table.rs`, generated by
/// `scripts/join_table.py` from the join CSV it names) takes their join type, key type and build
/// and probe row counts ([`JoinChoice::Measured`]).
impl Default for ArrowMetalConfig {
    fn default() -> Self {
        Self {
            min_rows: 250_000,
            accept_inexact: false,
            take_when_unknown: false,
            sort: true,
            topk: false,
            aggregate: true,
            filter: false,
            aggregate_choice: AggregateChoice::Measured,
            table_rows: None,
            join: true,
            join_choice: JoinChoice::Measured,
            report_plans: 64,
        }
    }
}

impl ArrowMetalConfig {
    /// Every shape the rule can translate (sorts, top-k, aggregates, filters, joins), at the
    /// default `min_rows`. Aggregates and joins are still decided by their measured tables unless
    /// [`aggregate_choice`](Self::aggregate_choice) and [`join_choice`](Self::join_choice) say
    /// otherwise.
    pub fn all() -> Self {
        Self { topk: true, aggregate: true, filter: true, ..Self::default() }
    }

    /// Sets [`min_rows`](Self::min_rows).
    pub fn with_min_rows(mut self, rows: usize) -> Self {
        self.min_rows = rows;
        self
    }
    /// Sets [`accept_inexact`](Self::accept_inexact).
    pub fn with_accept_inexact(mut self, on: bool) -> Self {
        self.accept_inexact = on;
        self
    }
    /// Sets [`take_when_unknown`](Self::take_when_unknown).
    pub fn with_take_when_unknown(mut self, on: bool) -> Self {
        self.take_when_unknown = on;
        self
    }
    /// Sets [`sort`](Self::sort).
    pub fn with_sort(mut self, on: bool) -> Self {
        self.sort = on;
        self
    }
    /// Sets [`topk`](Self::topk).
    pub fn with_topk(mut self, on: bool) -> Self {
        self.topk = on;
        self
    }
    /// Sets [`aggregate`](Self::aggregate).
    pub fn with_aggregate(mut self, on: bool) -> Self {
        self.aggregate = on;
        self
    }
    /// Sets [`filter`](Self::filter).
    pub fn with_filter(mut self, on: bool) -> Self {
        self.filter = on;
        self
    }
    /// Sets [`aggregate_choice`](Self::aggregate_choice).
    pub fn with_aggregate_choice(mut self, choice: AggregateChoice) -> Self {
        self.aggregate_choice = choice;
        self
    }
    /// Sets [`table_rows`](Self::table_rows).
    pub fn with_table_rows(mut self, rows: Option<usize>) -> Self {
        self.table_rows = rows;
        self
    }
    /// Sets [`join`](Self::join).
    pub fn with_join(mut self, on: bool) -> Self {
        self.join = on;
        self
    }
    /// Sets [`join_choice`](Self::join_choice).
    pub fn with_join_choice(mut self, choice: JoinChoice) -> Self {
        self.join_choice = choice;
        self
    }
    /// Sets [`report_plans`](Self::report_plans).
    pub fn with_report_plans(mut self, plans: usize) -> Self {
        self.report_plans = plans;
        self
    }
}

/// One node the rule looked at, one replaced aggregate's run-time choice, or one runtime fallback.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Decision {
    /// The node, as DataFusion prints it on one line.
    pub node: String,
    /// At plan time: the node was replaced by a `MetalExec`. For a run-time choice: the aggregate
    /// ran on ArrowMetal.
    pub taken: bool,
    /// Why it was taken or left.
    pub reason: String,
    /// True for a `MetalExec` that hit an ArrowMetal error at run time and ran DataFusion instead.
    pub runtime_fallback: bool,
    /// For a replaced aggregate's run-time choice: what it was decided from. `taken` is then
    /// whether it ran on ArrowMetal.
    pub groups: Option<GroupChoice>,
}

/// What a replaced aggregate's run-time choice was made from.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct GroupChoice {
    /// Rows of the collected input.
    pub rows: usize,
    /// The group-count probe's answer (`None` when `aggregate_choice` forced the choice).
    pub estimate: Option<crate::probe::GroupEstimate>,
}

impl Decision {
    pub(crate) fn runtime_fallback(op: &MetalOp, msg: &str) -> Self {
        let reason = if msg.starts_with(crate::gpu::DATA_DEPENDENT) {
            format!("{msg}; ran the DataFusion plan instead")
        } else {
            format!("ArrowMetal error at run time, ran the DataFusion plan instead: {msg}")
        };
        Decision { node: format!("MetalExec {op:?}"), taken: false, reason, runtime_fallback: true, groups: None }
    }

    pub(crate) fn memory_hand_back(op: &MetalOp, what: &str, err: &str) -> Self {
        Decision {
            node: format!("MetalExec {op:?}"),
            taken: false,
            reason: format!("the memory pool refused the reservation for {what}, ran the DataFusion plan instead: {err}"),
            runtime_fallback: true,
            groups: None,
        }
    }

    pub(crate) fn runtime_choice(
        op: &MetalOp,
        on_arrowmetal: bool,
        reason: String,
        rows: usize,
        estimate: Option<crate::probe::GroupEstimate>,
    ) -> Self {
        let reason = format!("{}: {reason}", if on_arrowmetal { "ran on ArrowMetal" } else { "handed back to DataFusion" });
        Decision {
            node: format!("MetalExec {op:?}"),
            taken: on_arrowmetal,
            reason,
            runtime_fallback: false,
            groups: Some(GroupChoice { rows, estimate }),
        }
    }

    /// A replaced aggregate's run-time choice (see [`AggregateChoice`]).
    pub fn is_runtime_choice(&self) -> bool {
        self.groups.is_some()
    }

    /// A runtime fallback (the `runtime_fallback` field) caused by the data, not an error.
    pub fn is_data_dependent(&self) -> bool {
        self.runtime_fallback && self.reason.starts_with(crate::gpu::DATA_DEPENDENT)
    }
}

impl fmt::Display for Decision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tag = if self.runtime_fallback {
            "FALLBACK"
        } else if self.groups.is_some() {
            if self.taken { "GPU" } else { "HANDBACK" }
        } else if self.taken {
            "TAKEN"
        } else {
            "LEFT"
        };
        write!(f, "{tag:8} {} -- {}", self.node, self.reason)
    }
}

/// What the rule decided for the last `report_plans` plans since it was created or last cleared.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct Report(Vec<Decision>);

impl Report {
    /// Every decision, oldest first.
    pub fn decisions(&self) -> &[Decision] {
        &self.0
    }
    /// Nodes the rule replaced at plan time.
    pub fn taken(&self) -> impl Iterator<Item = &Decision> {
        self.0.iter().filter(|d| d.taken && d.groups.is_none())
    }
    /// Nodes the rule left at plan time.
    pub fn left(&self) -> impl Iterator<Item = &Decision> {
        self.0.iter().filter(|d| !d.taken && !d.runtime_fallback && d.groups.is_none())
    }
    /// Replaced aggregates' run-time choices (ArrowMetal or handed back).
    pub fn runtime_choices(&self) -> impl Iterator<Item = &Decision> {
        self.0.iter().filter(|d| d.groups.is_some())
    }
    /// `MetalExec`s that ran DataFusion's plan instead after an ArrowMetal error or on data the
    /// GPU path cannot answer exactly (see [`Decision::is_data_dependent`]).
    pub fn runtime_fallbacks(&self) -> impl Iterator<Item = &Decision> {
        self.0.iter().filter(|d| d.runtime_fallback)
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for d in &self.0 {
            writeln!(f, "{d}")?;
        }
        Ok(())
    }
}

/// The shared decision log: the decisions of the last `keep` plans, each tagged with its plan.
#[derive(Debug)]
pub(crate) struct Log {
    entries: VecDeque<(u64, Decision)>,
    plan: u64,
    keep: u64,
}

impl Log {
    fn new(keep: usize) -> Self {
        Self { entries: VecDeque::new(), plan: 0, keep: keep.max(1) as u64 }
    }

    /// Starts a new plan: drops the decisions of plans outside the last `keep`.
    fn begin_plan(&mut self) -> u64 {
        self.plan += 1;
        let first = self.plan.saturating_sub(self.keep - 1);
        while self.entries.front().is_some_and(|(p, _)| *p < first) {
            self.entries.pop_front();
        }
        self.plan
    }

    /// Records a decision of plan `plan` (dropped when that plan is no longer kept).
    pub(crate) fn push(&mut self, plan: u64, d: Decision) {
        if plan + self.keep > self.plan {
            self.entries.push_back((plan, d));
        }
    }
}

pub(crate) type SharedLog = Arc<Mutex<Log>>;

/// The log, also after a panic elsewhere poisoned its mutex (the log stays consistent: every
/// update is a single push or pop).
pub(crate) fn lock(log: &SharedLog) -> MutexGuard<'_, Log> {
    log.lock().unwrap_or_else(|p| p.into_inner())
}

/// Replaces `SortExec` (+ its `SortPreservingMergeExec`), hash `AggregateExec` (a `Single` node, or
/// a `Final`/`Partial` pair) and `FilterExec` with [`MetalExec`] when ArrowMetal gives the same
/// answer and the input is big enough. Cloning shares the report.
#[derive(Debug, Clone)]
pub struct ArrowMetalRule {
    config: ArrowMetalConfig,
    log: SharedLog,
}

struct Candidate {
    op: std::result::Result<MetalOp, String>,
    /// The input `MetalExec` reads (the replaced chain's leaf input; a join's left input).
    input: Arc<dyn ExecutionPlan>,
    /// A join's right input.
    right: Option<Arc<dyn ExecutionPlan>>,
    /// The runtime fallback, when it is not the node itself (it must have `MetalExec`'s schema).
    original: Option<Arc<dyn ExecutionPlan>>,
    /// A node to put back above `MetalExec` (the projection of `SPM -> Projection -> SortExec`).
    reparent: Option<Arc<dyn ExecutionPlan>>,
}

impl Candidate {
    fn new(op: std::result::Result<MetalOp, String>, input: Arc<dyn ExecutionPlan>) -> Self {
        Self { op, input, right: None, original: None, reparent: None }
    }
}

/// `p` without the exchanges directly above its data: `RepartitionExec`s and
/// `CoalescePartitionsExec`s. A join's `MetalExec` collects every partition of both inputs, so a
/// re-partitioning below it only moves rows it reads anyway. (The fallback plan keeps them.)
fn below_exchanges(p: &Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let mut cur = Arc::clone(p);
    loop {
        let next = if let Some(r) = cur.downcast_ref::<RepartitionExec>() {
            Arc::clone(r.input())
        } else if let Some(c) = cur.downcast_ref::<CoalescePartitionsExec>() {
            if c.fetch().is_some() {
                return cur;
            }
            Arc::clone(c.input())
        } else {
            return cur;
        };
        cur = next;
    }
}

/// `spm_expr` (written against the projection's output) mapped through `proj` onto the
/// projection's input, or `None` when a merge key is not a plain column of the projection.
fn map_through_projection(
    spm_expr: &datafusion::physical_expr::LexOrdering,
    proj: &ProjectionExec,
) -> Option<Vec<(usize, arrow::compute::SortOptions)>> {
    let mut out = Vec::new();
    for s in spm_expr.iter() {
        let c = s.expr.downcast_ref::<Column>()?;
        let pe = proj.expr().get(c.index())?;
        let inner = pe.expr.downcast_ref::<Column>()?;
        out.push((inner.index(), s.options));
    }
    Some(out)
}

fn sort_keys(expr: &datafusion::physical_expr::LexOrdering) -> Option<Vec<(usize, arrow::compute::SortOptions)>> {
    expr.iter().map(|s| s.expr.downcast_ref::<Column>().map(|c| (c.index(), s.options))).collect()
}

impl ArrowMetalRule {
    /// A rule with this config and an empty report.
    pub fn new(config: ArrowMetalConfig) -> Self {
        let keep = config.report_plans;
        Self { config, log: Arc::new(Mutex::new(Log::new(keep))) }
    }

    /// The config the rule was made with.
    pub fn config(&self) -> &ArrowMetalConfig {
        &self.config
    }

    /// A snapshot of the decisions of the last `report_plans` plans (plan time, run-time choices
    /// and runtime fallbacks).
    pub fn report(&self) -> Report {
        Report(lock(&self.log).entries.iter().map(|(_, d)| d.clone()).collect())
    }

    /// Empties the report.
    pub fn clear_report(&self) {
        lock(&self.log).entries.clear();
    }

    fn record(&self, plan: u64, node: &Arc<dyn ExecutionPlan>, taken: bool, reason: String) {
        let node = displayable(node.as_ref()).one_line().to_string().trim_end().to_string();
        lock(&self.log).push(plan, Decision { node, taken, reason, runtime_fallback: false, groups: None });
    }

    /// Why a sort with this `fetch` is switched off in the config, if it is.
    fn sort_disabled(&self, fetch: Option<usize>) -> Option<String> {
        match fetch {
            None if !self.config.sort => Some("sort disabled in config".into()),
            Some(n) if !self.config.topk => Some(format!("top-k (sort with fetch {n}) disabled in config")),
            _ => None,
        }
    }

    /// The node this rule would replace, as an operation over an input, or `None` when the node is
    /// not one it handles at all (a scan, a projection, ...).
    fn candidate(&self, node: &Arc<dyn ExecutionPlan>) -> Option<Candidate> {
        if let Some(spm) = node.downcast_ref::<SortPreservingMergeExec>() {
            // `SPM -> SortExec`, or `SPM -> ProjectionExec -> SortExec`: DataFusion 55 plans the
            // latter for any ORDER BY whose SELECT list reorders or computes columns (the
            // projection stays above the per-partition sorts). The pair is replaced by one
            // MetalExec sort, and the projection is put back above it.
            let (sort_node, proj) = if spm.input().downcast_ref::<SortExec>().is_some() {
                (Arc::clone(spm.input()), None)
            } else {
                let p = spm.input().downcast_ref::<ProjectionExec>()?;
                p.input().downcast_ref::<SortExec>()?;
                (Arc::clone(p.input()), Some(Arc::clone(spm.input())))
            };
            let sort = sort_node.downcast_ref::<SortExec>()?;
            let fetch = match (spm.fetch(), sort.fetch()) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
            if let Some(why) = self.sort_disabled(fetch) {
                return Some(Candidate::new(Err(why), Arc::clone(sort.input())));
            }
            let same_order = match &proj {
                None => sort.expr() == spm.expr(),
                Some(p) => {
                    let p = p.downcast_ref::<ProjectionExec>()?;
                    let mapped = map_through_projection(spm.expr(), p);
                    mapped.is_some() && mapped == sort_keys(sort.expr())
                }
            };
            if !same_order {
                return Some(Candidate::new(
                    Err("merge ordering differs from the sort's".into()),
                    Arc::clone(sort.input()),
                ));
            }
            let input = Arc::clone(sort.input());
            let op = translate::sort_op(sort.expr(), &input.schema(), fetch);
            let mut c = Candidate::new(op, input);
            if proj.is_some() {
                // The fallback is the merge over the sorts, without the projection (which stays).
                c.original = Some(Arc::new(
                    SortPreservingMergeExec::new(sort.expr().clone(), Arc::clone(&sort_node)).with_fetch(fetch),
                ));
                c.reparent = proj;
            }
            return Some(c);
        }
        if let Some(sort) = node.downcast_ref::<SortExec>() {
            let input = Arc::clone(sort.input());
            if let Some(why) = self.sort_disabled(sort.fetch()) {
                return Some(Candidate::new(Err(why), input));
            }
            if sort.preserve_partitioning() && input.output_partitioning().partition_count() > 1 {
                return Some(Candidate::new(Err("per-partition sort (preserve_partitioning) with no replaced merge above it".into()), input));
            }
            let op = translate::sort_op(sort.expr(), &input.schema(), sort.fetch());
            return Some(Candidate::new(op, input));
        }
        if let Some(agg) = node.downcast_ref::<AggregateExec>() {
            if !self.config.aggregate {
                return Some(Candidate::new(Err("aggregate disabled in config".into()), Arc::clone(agg.input())));
            }
            if node.output_ordering().is_some() {
                return Some(Candidate::new(
                    Err("the aggregate's output carries an ordering (sorted input), which the GPU group-by does not keep".into()),
                    Arc::clone(agg.input()),
                ));
            }
            return Some(match agg.mode() {
                AggregateMode::Single | AggregateMode::SinglePartitioned => {
                    Candidate::new(translate::aggregate_op(agg), Arc::clone(agg.input()))
                }
                AggregateMode::Final | AggregateMode::FinalPartitioned => {
                    // Walk down through the exchange to the Partial that feeds this Final; the pair
                    // is replaced as one, reading the Partial's input.
                    let mut cur = Arc::clone(agg.input());
                    loop {
                        if cur.downcast_ref::<RepartitionExec>().is_some()
                            || cur.downcast_ref::<CoalescePartitionsExec>().is_some()
                        {
                            let next = Arc::clone(cur.children()[0]);
                            cur = next;
                            continue;
                        }
                        break;
                    }
                    match cur.downcast_ref::<AggregateExec>() {
                        Some(p) if *p.mode() == AggregateMode::Partial && translate::same_aggregates(agg, p) => {
                            Candidate::new(translate::aggregate_op(p), Arc::clone(p.input()))
                        }
                        _ => Candidate::new(Err("Final aggregate without a matching Partial below its exchange".into()), Arc::clone(agg.input())),
                    }
                }
                AggregateMode::Partial => Candidate::new(Err("Partial aggregate whose Final was not replaced".into()), Arc::clone(agg.input())),
                AggregateMode::PartialReduce => Candidate::new(Err("PartialReduce aggregate".into()), Arc::clone(agg.input())),
            });
        }
        if let Some(f) = node.downcast_ref::<FilterExec>() {
            let input = Arc::clone(f.input());
            if !self.config.filter {
                return Some(Candidate::new(Err("filter disabled in config".into()), input));
            }
            if node.fetch().is_some() {
                return Some(Candidate::new(Err("filter with a fetch limit".into()), input));
            }
            if node.output_ordering().is_some() && input.output_partitioning().partition_count() > 1 {
                return Some(Candidate::new(Err("order-preserving filter over several partitions".into()), input));
            }
            let projection = f.projection().as_ref().map(|p| p.iter().copied().collect::<Vec<usize>>());
            let op = translate::filter_op(f.predicate(), &input.schema(), projection);
            return Some(Candidate::new(op, input));
        }
        if let Some(j) = node.downcast_ref::<HashJoinExec>() {
            let (left, right) = (below_exchanges(j.left()), below_exchanges(j.right()));
            // DataFusion's hash join keeps the probe (right) side's order per partition for inner
            // and right joins, and plans may rely on it (a sort pushed below the join). The plan
            // runner's join keeps its left input's order, which is the probe side for both; so an
            // ordered join is taken when its probe input is one partition read as it is.
            let ordered_ok = matches!(j.join_type(), JoinType::Inner | JoinType::Right)
                && Arc::ptr_eq(&right, j.right())
                && j.right().output_partitioning().partition_count() == 1;
            let op = if !self.config.join {
                Err("join disabled in config".into())
            } else if node.output_ordering().is_some() && !ordered_ok {
                Err("the join's output carries an ordering of a probe side split over partitions".into())
            } else if Arc::ptr_eq(&left, &right) {
                Err("both join inputs are one plan node".into())
            } else {
                translate::join_op(j)
            };
            let mut c = Candidate::new(op, left);
            c.right = Some(right);
            return Some(c);
        }
        None
    }

    /// The input's row count as DataFusion's statistics give it.
    fn statistics_rows(input: &Arc<dyn ExecutionPlan>) -> Precision<usize> {
        match StatisticsContext::new().compute(input.as_ref(), &StatisticsArgs::new()) {
            Ok(s) => s.num_rows,
            Err(_) => Precision::Absent,
        }
    }

    /// A join input's row count, when the rule may go by it (exact, or an accepted estimate), and
    /// the phrase the report uses for it.
    fn row_count(&self, input: &Arc<dyn ExecutionPlan>) -> (Option<usize>, String) {
        match Self::statistics_rows(input) {
            Precision::Exact(n) => (Some(n), format!("{n} (exact)")),
            Precision::Inexact(n) if self.config.accept_inexact => (Some(n), format!("~{n} (inexact, accepted)")),
            Precision::Inexact(n) => (None, format!("~{n} (an estimate; accept_inexact is off)")),
            Precision::Absent => (None, "unknown".into()),
        }
    }

    /// Whether the input is big enough, the phrase the report uses for its size, and the row count
    /// it went by (None when unknown).
    fn size_ok(&self, input: &Arc<dyn ExecutionPlan>) -> (bool, String, Option<usize>) {
        let rows = Self::statistics_rows(input);
        let min = self.config.min_rows;
        match rows {
            Precision::Exact(n) => (n >= min, format!("input rows {n} (exact) vs min_rows {min}"), Some(n)),
            Precision::Inexact(n) if self.config.accept_inexact => {
                (n >= min, format!("input rows ~{n} (inexact, accepted) vs min_rows {min}"), Some(n))
            }
            Precision::Inexact(n) => (false, format!("input rows ~{n} are an estimate (accept_inexact is off)"), None),
            Precision::Absent => (
                self.config.take_when_unknown,
                format!("input row count unknown (take_when_unknown = {})", self.config.take_when_unknown),
                None,
            ),
        }
    }

    /// `top` is the node under the root's chain of projections (by address): nothing above it
    /// requires a distribution, so a replacement there keeps MetalExec's single output partition.
    fn visit(&self, plan: u64, node: Arc<dyn ExecutionPlan>, top: usize) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
        if node.downcast_ref::<MetalExec>().is_some() {
            return Ok(Transformed::no(node));
        }
        let Some(c) = self.candidate(&node) else {
            return Ok(Transformed::no(node));
        };
        let op = match c.op {
            Ok(op) => op,
            Err(why) => {
                self.record(plan, &node, false, why);
                return Ok(Transformed::no(node));
            }
        };
        // Keep the replaced node's partition count, so every parent's distribution requirement
        // (fixed by EnsureRequirements before this rule runs) still holds. At the top of the plan
        // (only projections above) there is no such requirement: re-partitioning the output there
        // would only hash every result row to split it, for `collect` to merge it again.
        let at_top = Arc::as_ptr(&node) as *const () as usize == top;
        let wrap = match node.output_partitioning() {
            p if p.partition_count() <= 1 => None,
            _ if at_top => None,
            Partitioning::Hash(exprs, n) => Some(Partitioning::Hash(exprs.clone(), *n)),
            Partitioning::RoundRobinBatch(n) | Partitioning::UnknownPartitioning(n) => {
                Some(Partitioning::RoundRobinBatch(*n))
            }
            Partitioning::Range(_) => {
                self.record(plan, &node, false, "range-partitioned output".into());
                return Ok(Transformed::no(node));
            }
        };
        // MetalExec coalesces its input, so a round-robin split right below it is pure overhead:
        // read what feeds the split instead. (The fallback plan keeps the split.)
        let mut input = Arc::clone(&c.input);
        while let Some(r) = input.downcast_ref::<RepartitionExec>() {
            if !matches!(r.partitioning(), Partitioning::RoundRobinBatch(_)) {
                break;
            }
            let next = Arc::clone(r.input());
            input = next;
        }
        let mut inputs = vec![Arc::clone(&input)];
        let (ok, mut size, rows) = match &c.right {
            None => self.size_ok(&input),
            Some(right) => {
                // A join: both inputs need a row count; `min_rows` is compared with the larger.
                inputs.push(Arc::clone(right));
                let (lrows, ltext) = self.row_count(&input);
                let (rrows, rtext) = self.row_count(right);
                let rows_text = format!("left (build) rows {ltext}, right (probe) rows {rtext}");
                match (lrows, rrows) {
                    (Some(l), Some(r)) => {
                        let min = self.config.min_rows;
                        let n = l.max(r);
                        let mut why = format!("{rows_text}; the larger vs min_rows {min}");
                        let mut ok = n >= min;
                        if ok && self.config.join_choice == JoinChoice::Measured {
                            match crate::choice::join_takes(&op, input.as_ref(), right.as_ref(), l as u64, r as u64) {
                                Ok(w) => why = format!("{why}; {w}"),
                                Err(w) => {
                                    why = format!("{why}; {w}");
                                    ok = false;
                                }
                            }
                        }
                        (ok, why, Some(l + r))
                    }
                    _ => (false, format!("{rows_text}: a join needs both row counts"), None),
                }
            }
        };
        if !ok {
            self.record(plan, &node, false, size);
            return Ok(Transformed::no(node));
        }
        // A replaced aggregate is decided at run time; under `Measured` it is replaced only when
        // the measured table takes its shape at some group count at this row count.
        if matches!(op, MetalOp::Aggregate { .. }) && self.config.aggregate_choice == AggregateChoice::Measured {
            let Some(shape) = crate::choice::shape(&op, input.as_ref()) else {
                self.record(plan, &node, false, "aggregate without a shape".into());
                return Ok(Transformed::no(node));
            };
            if let Some(n) = self.config.table_rows.or(rows) {
                match crate::choice::any_bucket(&shape, n as u64) {
                    Ok(why) => size = format!("{size}; {why}"),
                    Err(why) => {
                        self.record(plan, &node, false, format!("{size}; {why}"));
                        return Ok(Transformed::no(node));
                    }
                }
            }
        }
        let original = c.original.unwrap_or_else(|| Arc::clone(&node));
        let mut wrap = wrap;
        let mut out_parts = 1;
        if c.right.is_some() && !at_top {
            // A join keeps the replaced node's partition count itself (one GPU execution dealt
            // out to the partitions); only a hash partitioning still needs a re-partition above.
            out_parts = node.output_partitioning().partition_count();
            if !matches!(wrap, Some(Partitioning::Hash(..))) {
                wrap = None;
            }
        }
        let metal: Arc<dyn ExecutionPlan> = Arc::new(
            MetalExec::new(
                op,
                inputs,
                original,
                (Arc::clone(&self.log), plan),
                crate::exec::AggSettings {
                    choice: self.config.aggregate_choice,
                    table_rows: self.config.table_rows,
                    rows_hint: rows,
                },
            )
            .with_output_partitions(out_parts),
        );
        let mut reason = size;
        if out_parts > 1 {
            reason.push_str(&format!("; output in {out_parts} partitions"));
        }
        if at_top && node.output_partitioning().partition_count() > 1 {
            reason.push_str("; output kept at one partition (only projections above it)");
        }
        let metal = match c.reparent {
            #[allow(deprecated)] // `replace_children` is the 55 name; `with_new_children` still works
            Some(p) => {
                reason.push_str("; projection kept above it");
                p.with_new_children(vec![metal])?
            }
            None => metal,
        };
        let out: Arc<dyn ExecutionPlan> = match wrap {
            None => metal,
            Some(p) => {
                if c.right.is_some() && matches!(p, Partitioning::Hash(..)) {
                    reason.push_str(&format!(
                        "; output re-partitioned to {p} where the parent requires it (to keep the parent's distribution)"
                    ));
                } else {
                    reason.push_str(&format!("; output re-partitioned to {p} to keep the parent's distribution"));
                }
                Arc::new(RepartitionExec::try_new(metal, p)?)
            }
        };
        self.record(plan, &node, true, reason);
        Ok(Transformed::yes(out))
    }
}

/// A replaced hash-partitioned join gets a hash `RepartitionExec` above it, so a parent that
/// requires that distribution still has it. Where the parent does not require a hash
/// distribution of that input (a partial aggregate, a projection, a filter), hashing every joined
/// row only to split it is dropped: the join's own output partitions feed the parent.
#[allow(deprecated)] // `required_input_distribution` and `with_new_children`: the 55 names still work
fn drop_hash_where_not_required(node: Arc<dyn ExecutionPlan>) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
    let required = node.required_input_distribution();
    let children: Vec<Arc<dyn ExecutionPlan>> = node.children().into_iter().cloned().collect();
    let mut changed = false;
    let mut new_children = Vec::with_capacity(children.len());
    for (i, c) in children.into_iter().enumerate() {
        let swap = match c.downcast_ref::<RepartitionExec>() {
            Some(r) => {
                let over_join = r
                    .input()
                    .downcast_ref::<MetalExec>()
                    .is_some_and(|m| matches!(m.op(), MetalOp::Join { .. }));
                match r.partitioning() {
                    Partitioning::Hash(..)
                        if over_join && matches!(required.get(i), Some(Distribution::UnspecifiedDistribution)) =>
                    {
                        Some(Arc::clone(r.input()))
                    }
                    _ => None,
                }
            }
            None => None,
        };
        match swap {
            Some(s) => {
                changed = true;
                new_children.push(s);
            }
            None => new_children.push(c),
        }
    }
    if !changed {
        return Ok(Transformed::no(node));
    }
    Ok(Transformed::yes(node.with_new_children(new_children)?))
}

impl PhysicalOptimizerRule for ArrowMetalRule {
    fn optimize(&self, plan: Arc<dyn ExecutionPlan>, config: &ConfigOptions) -> Result<Arc<dyn ExecutionPlan>> {
        let mut top = Arc::clone(&plan);
        while let Some(p) = top.downcast_ref::<ProjectionExec>() {
            let next = Arc::clone(p.input());
            top = next;
        }
        let top = Arc::as_ptr(&top) as *const () as usize;
        let id = lock(&self.log).begin_plan();
        let plan = plan.transform_down(|n| self.visit(id, n, top))?.data;
        // Dropped only when DataFusion's own check of every node's required input distribution
        // still passes (a projection between a join and a hash-partitioned aggregate passes the
        // join's partitioning through).
        let relaxed = Arc::clone(&plan).transform_up(drop_hash_where_not_required)?;
        if relaxed.transformed && SanityCheckPlan::new().optimize(Arc::clone(&relaxed.data), config).is_ok() {
            return Ok(relaxed.data);
        }
        Ok(plan)
    }

    fn name(&self) -> &str {
        "ArrowMetalRule"
    }

    fn schema_check(&self) -> bool {
        true
    }
}
