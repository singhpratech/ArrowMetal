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
use datafusion::physical_plan::filter::FilterExec;
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
#[derive(Debug, Clone)]
pub struct ArrowMetalConfig {
    /// Take a node only when its input has at least this many rows.
    pub min_rows: usize,
    /// Use an inexact (estimated) row count as if it were exact. Default: no.
    pub accept_inexact: bool,
    /// Take a node whose input row count is unknown. Default: no (leave it).
    pub take_when_unknown: bool,
    pub sort: bool,
    pub aggregate: bool,
    pub filter: bool,
}

impl Default for ArrowMetalConfig {
    fn default() -> Self {
        // 1,000,000 is the order of ArrowMetal's measured crossover for sorts and group-bys
        // against CPU engines on this hardware (docs/CROSSOVER.md); lane D1 measures DataFusion's.
        Self {
            min_rows: 1_000_000,
            accept_inexact: false,
            take_when_unknown: false,
            sort: true,
            aggregate: true,
            filter: true,
        }
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
            format!("{msg}: DataFusion's answer depends on its row order, ran the DataFusion plan instead")
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

    /// The node this rule would replace, as an operation over an input, or `None` when the node is
    /// not one it handles at all (a scan, a projection, ...).
    fn candidate(&self, node: &Arc<dyn ExecutionPlan>) -> Option<Candidate> {
        if let Some(spm) = node.downcast_ref::<SortPreservingMergeExec>() {
            let sort = spm.input().downcast_ref::<SortExec>()?;
            if !self.config.sort {
                return Some(Candidate { op: Err("sort disabled in config".into()), input: Arc::clone(spm.input()) });
            }
            if sort.expr() != spm.expr() {
                return Some(Candidate {
                    op: Err("merge ordering differs from the sort's".into()),
                    input: Arc::clone(sort.input()),
                });
            }
            let fetch = match (spm.fetch(), sort.fetch()) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
            let input = Arc::clone(sort.input());
            let op = translate::sort_op(sort.expr(), &input.schema(), fetch);
            return Some(Candidate { op, input });
        }
        if let Some(sort) = node.downcast_ref::<SortExec>() {
            let input = Arc::clone(sort.input());
            if !self.config.sort {
                return Some(Candidate { op: Err("sort disabled in config".into()), input });
            }
            if sort.preserve_partitioning() && input.output_partitioning().partition_count() > 1 {
                return Some(Candidate {
                    op: Err("per-partition sort (preserve_partitioning) with no replaced merge above it".into()),
                    input,
                });
            }
            let op = translate::sort_op(sort.expr(), &input.schema(), sort.fetch());
            return Some(Candidate { op, input });
        }
        if let Some(agg) = node.downcast_ref::<AggregateExec>() {
            if !self.config.aggregate {
                return Some(Candidate { op: Err("aggregate disabled in config".into()), input: Arc::clone(agg.input()) });
            }
            return Some(match agg.mode() {
                AggregateMode::Single | AggregateMode::SinglePartitioned => {
                    Candidate { op: translate::aggregate_op(agg), input: Arc::clone(agg.input()) }
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
                            Candidate { op: translate::aggregate_op(p), input: Arc::clone(p.input()) }
                        }
                        _ => Candidate {
                            op: Err("Final aggregate without a matching Partial below its exchange".into()),
                            input: Arc::clone(agg.input()),
                        },
                    }
                }
                AggregateMode::Partial => Candidate {
                    op: Err("Partial aggregate whose Final was not replaced".into()),
                    input: Arc::clone(agg.input()),
                },
                AggregateMode::PartialReduce => Candidate {
                    op: Err("PartialReduce aggregate".into()),
                    input: Arc::clone(agg.input()),
                },
            });
        }
        if let Some(f) = node.downcast_ref::<FilterExec>() {
            let input = Arc::clone(f.input());
            if !self.config.filter {
                return Some(Candidate { op: Err("filter disabled in config".into()), input });
            }
            if node.fetch().is_some() {
                return Some(Candidate { op: Err("filter with a fetch limit".into()), input });
            }
            if node.output_ordering().is_some() && input.output_partitioning().partition_count() > 1 {
                return Some(Candidate {
                    op: Err("order-preserving filter over several partitions".into()),
                    input,
                });
            }
            let projection = f.projection().as_ref().map(|p| p.iter().copied().collect::<Vec<usize>>());
            let op = translate::filter_op(f.predicate(), &input.schema(), projection);
            return Some(Candidate { op, input });
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

    fn visit(&self, node: Arc<dyn ExecutionPlan>) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
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
        // (fixed by EnsureRequirements before this rule runs) still holds.
        let wrap = match node.output_partitioning() {
            p if p.partition_count() <= 1 => None,
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
        let metal: Arc<dyn ExecutionPlan> =
            Arc::new(MetalExec::new(op, input, Arc::clone(&node), Arc::clone(&self.log)));
        let mut reason = size;
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
        Ok(plan.transform_down(|n| self.visit(n))?.data)
    }

    fn name(&self) -> &str {
        "ArrowMetalRule"
    }

    fn schema_check(&self) -> bool {
        true
    }
}
