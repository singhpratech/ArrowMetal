//! `ArrowMetalRule`: the `PhysicalOptimizerRule` that swaps supported nodes for `MetalExec`.

use std::fmt;
use std::sync::{Arc, Mutex};

use datafusion::common::config::ConfigOptions;
use datafusion::common::stats::Precision;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::Result;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::{
    displayable, ExecutionPlan, ExecutionPlanProperties, Partitioning, StatisticsArgs,
    StatisticsContext,
};

use crate::exec::{MetalExec, MetalOp};
use crate::translate;

/// When the rule takes a node.
///
/// [`Default`] is the measured take-list; [`ArrowMetalConfig::all`] takes every shape the rule can
/// translate (what the differential grid and the benchmark use).
#[derive(Debug, Clone)]
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
    pub aggregate: bool,
    pub filter: bool,
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
/// | GROUP BY, count(*) only, 100k and 1M groups | 1.08x - 1.26x | 1.41x - 2.63x | 1.62x - 3.23x | left (see below) |
/// | GROUP BY, count(*) only, 200 groups | 0.43x - 0.63x | 0.76x - 1.39x | 0.95x - 1.72x | left |
/// | GROUP BY, count(*) only, rows/2 groups | 1.08x - 1.29x | 1.61x - 1.90x | 0.87x - 0.92x | left |
/// | GROUP BY, sum / avg over Float64, 200 .. rows/2 groups | 0.22x - 1.06x | 0.46x - 1.47x | 0.42x - 1.42x | left |
/// | GROUP BY, min + max over Float64, 200 .. rows/2 groups | 0.16x - 0.71x | 0.26x - 1.07x | 0.26x - 0.88x | left |
/// | SELECT DISTINCT (int32, int32), 10k groups | 0.72x - 0.85x | 1.60x - 1.80x | 1.51x - 1.65x | left (see below) |
/// | GROUP BY over DataFusion's Parquet scan, 100k groups, sum + count | | 1.08x - 1.39x | 1.29x - 1.57x | left |
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
/// Aggregates are off: count(*)-only group-bys with 100k-1M groups clear 1.5x at 10M and 50M
/// (one exception, 1.41x), and so does the one DISTINCT shape measured, but the same count(*) SQL
/// with 200 groups is 0.76x - 1.72x and with rows/2 groups at 50M 0.87x - 0.92x, and every
/// sum / avg / min+max family has shapes below 1x. The rule sees the row count but not the group
/// count (DataFusion's MemTable and Parquet statistics carry no distinct counts), so it cannot tell
/// these apart. Hash joins are not replaced at all (the rule has no join operator).
impl Default for ArrowMetalConfig {
    fn default() -> Self {
        Self {
            min_rows: 250_000,
            accept_inexact: false,
            take_when_unknown: false,
            sort: true,
            topk: false,
            aggregate: false,
            filter: false,
        }
    }
}

impl ArrowMetalConfig {
    /// Every shape the rule can translate (sorts, top-k, aggregates, filters), at the default
    /// `min_rows`.
    pub fn all() -> Self {
        Self { topk: true, aggregate: true, filter: true, ..Self::default() }
    }
}

/// One node the rule looked at, or one runtime fallback.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    /// The node, as DataFusion prints it on one line.
    pub node: String,
    pub taken: bool,
    /// Why it was taken or left.
    pub reason: String,
    /// True for a `MetalExec` that hit an ArrowMetal error at run time and ran DataFusion instead.
    pub runtime_fallback: bool,
}

impl Decision {
    pub(crate) fn runtime_fallback(op: &MetalOp, msg: &str) -> Self {
        let reason = if msg.starts_with(crate::gpu::DATA_DEPENDENT) {
            format!("{msg}; ran the DataFusion plan instead")
        } else {
            format!("ArrowMetal error at run time, ran the DataFusion plan instead: {msg}")
        };
        Decision { node: format!("MetalExec {op:?}"), taken: false, reason, runtime_fallback: true }
    }

    /// A runtime fallback caused by the data (see [`Decision::runtime_fallback`]), not an error.
    pub fn is_data_dependent(&self) -> bool {
        self.runtime_fallback && self.reason.starts_with(crate::gpu::DATA_DEPENDENT)
    }
}

impl fmt::Display for Decision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tag = if self.runtime_fallback {
            "FALLBACK"
        } else if self.taken {
            "TAKEN"
        } else {
            "LEFT"
        };
        write!(f, "{tag:8} {} -- {}", self.node, self.reason)
    }
}

/// Everything the rule decided since it was created or last cleared.
#[derive(Debug, Clone, Default)]
pub struct Report(Vec<Decision>);

impl Report {
    pub fn decisions(&self) -> &[Decision] {
        &self.0
    }
    pub fn taken(&self) -> impl Iterator<Item = &Decision> {
        self.0.iter().filter(|d| d.taken)
    }
    pub fn left(&self) -> impl Iterator<Item = &Decision> {
        self.0.iter().filter(|d| !d.taken && !d.runtime_fallback)
    }
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

/// Replaces `SortExec` (+ its `SortPreservingMergeExec`), hash `AggregateExec` (a `Single` node, or
/// a `Final`/`Partial` pair) and `FilterExec` with [`MetalExec`] when ArrowMetal gives the same
/// answer and the input is big enough. Cloning shares the report.
#[derive(Debug, Clone)]
pub struct ArrowMetalRule {
    config: ArrowMetalConfig,
    log: Arc<Mutex<Vec<Decision>>>,
}

struct Candidate {
    op: std::result::Result<MetalOp, String>,
    /// The input `MetalExec` reads (the replaced chain's leaf input).
    input: Arc<dyn ExecutionPlan>,
    /// The runtime fallback, when it is not the node itself (it must have `MetalExec`'s schema).
    original: Option<Arc<dyn ExecutionPlan>>,
    /// A node to put back above `MetalExec` (the projection of `SPM -> Projection -> SortExec`).
    reparent: Option<Arc<dyn ExecutionPlan>>,
}

impl Candidate {
    fn new(op: std::result::Result<MetalOp, String>, input: Arc<dyn ExecutionPlan>) -> Self {
        Self { op, input, original: None, reparent: None }
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
    pub fn new(config: ArrowMetalConfig) -> Self {
        Self { config, log: Arc::new(Mutex::new(Vec::new())) }
    }

    pub fn config(&self) -> &ArrowMetalConfig {
        &self.config
    }

    /// A snapshot of every decision so far (planning and runtime fallbacks).
    pub fn report(&self) -> Report {
        Report(self.log.lock().unwrap().clone())
    }

    pub fn clear_report(&self) {
        self.log.lock().unwrap().clear();
    }

    fn record(&self, node: &Arc<dyn ExecutionPlan>, taken: bool, reason: String) {
        let node = displayable(node.as_ref()).one_line().to_string().trim_end().to_string();
        self.log.lock().unwrap().push(Decision { node, taken, reason, runtime_fallback: false });
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
        None
    }

    /// Whether the input is big enough, and the phrase the report uses for its size.
    fn size_ok(&self, input: &Arc<dyn ExecutionPlan>) -> (bool, String) {
        let stats = StatisticsContext::new().compute(input.as_ref(), &StatisticsArgs::new());
        let rows = match stats {
            Ok(s) => s.num_rows,
            Err(_) => Precision::Absent,
        };
        let min = self.config.min_rows;
        match rows {
            Precision::Exact(n) => (n >= min, format!("input rows {n} (exact) vs min_rows {min}")),
            Precision::Inexact(n) if self.config.accept_inexact => {
                (n >= min, format!("input rows ~{n} (inexact, accepted) vs min_rows {min}"))
            }
            Precision::Inexact(n) => (false, format!("input rows ~{n} are an estimate (accept_inexact is off)")),
            Precision::Absent => (
                self.config.take_when_unknown,
                format!("input row count unknown (take_when_unknown = {})", self.config.take_when_unknown),
            ),
        }
    }

    /// `top` is the node under the root's chain of projections (by address): nothing above it
    /// requires a distribution, so a replacement there keeps MetalExec's single output partition.
    fn visit(&self, node: Arc<dyn ExecutionPlan>, top: usize) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
        if node.downcast_ref::<MetalExec>().is_some() {
            return Ok(Transformed::no(node));
        }
        let Some(c) = self.candidate(&node) else {
            return Ok(Transformed::no(node));
        };
        let op = match c.op {
            Ok(op) => op,
            Err(why) => {
                self.record(&node, false, why);
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
                self.record(&node, false, "range-partitioned output".into());
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
        let (ok, size) = self.size_ok(&input);
        if !ok {
            self.record(&node, false, size);
            return Ok(Transformed::no(node));
        }
        let original = c.original.unwrap_or_else(|| Arc::clone(&node));
        let metal: Arc<dyn ExecutionPlan> = Arc::new(MetalExec::new(op, input, original, Arc::clone(&self.log)));
        let mut reason = size;
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
                reason.push_str(&format!("; output re-partitioned to {p} to keep the parent's distribution"));
                Arc::new(RepartitionExec::try_new(metal, p)?)
            }
        };
        self.record(&node, true, reason);
        Ok(Transformed::yes(out))
    }
}

impl PhysicalOptimizerRule for ArrowMetalRule {
    fn optimize(&self, plan: Arc<dyn ExecutionPlan>, _config: &ConfigOptions) -> Result<Arc<dyn ExecutionPlan>> {
        let mut top = Arc::clone(&plan);
        while let Some(p) = top.downcast_ref::<ProjectionExec>() {
            let next = Arc::clone(p.input());
            top = next;
        }
        let top = Arc::as_ptr(&top) as *const () as usize;
        Ok(plan.transform_down(|n| self.visit(n, top))?.data)
    }

    fn name(&self) -> &str {
        "ArrowMetalRule"
    }

    fn schema_check(&self) -> bool {
        true
    }
}
