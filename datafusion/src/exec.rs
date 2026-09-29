//! `MetalExec`: the `ExecutionPlan` that runs a replaced node on ArrowMetal.

use std::fmt;
use std::sync::{Arc, Mutex};

use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{internal_err, DataFusionError, Result};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::execution::memory_pool::MemoryConsumer;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricBuilder, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    collect, DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties, SendableRecordBatchStream,
};
use futures::{StreamExt, TryStreamExt};

use crate::rule::Decision;

/// One sort key: an input column, a direction, and where its nulls go.
#[derive(Debug, Clone, PartialEq)]
pub struct SortKey {
    pub column: usize,
    pub descending: bool,
    pub nulls_first: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggKind {
    Sum,
    Min,
    Max,
    /// `count(col)`: non-null values.
    Count,
    /// `count(*)`: rows.
    CountAll,
    /// `avg`.
    Mean,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AggSpec {
    pub kind: AggKind,
    pub column: Option<usize>,
    /// The argument column is floating point (min/max get the NaN / signed-zero fix-up).
    pub float: bool,
    /// The type DataFusion's schema gives the result.
    pub out_type: DataType,
}

/// What a `MetalExec` computes, in terms of its input's column indices.
#[derive(Debug, Clone, PartialEq)]
pub enum MetalOp {
    Sort { keys: Vec<SortKey>, fetch: Option<usize> },
    Aggregate { keys: Vec<usize>, aggs: Vec<AggSpec> },
    /// `predicate` is an ArrowMetal s-expression over columns named `c{i}`; `float_compared` are
    /// the float columns it compares with a literal (checked for negative NaN at run time).
    Filter { predicate: String, projection: Option<Vec<usize>>, float_compared: Vec<usize> },
}

/// Runs one [`MetalOp`] on ArrowMetal.
///
/// Collects every partition of its input (coalesced), hands the batches to ArrowMetal's chunked
/// import (no `concat_batches`), runs the operation on the GPU through ArrowMetal's plan runner, and
/// emits the result in `batch_size` slices as a single partition. If ArrowMetal returns an error at run time, the replaced subtree
/// (`original`) is run instead on the collected batches, and that is recorded in the report.
pub struct MetalExec {
    op: MetalOp,
    input: Arc<dyn ExecutionPlan>,
    /// The subtree this node replaced, with `input` as its leaf: the runtime fallback.
    original: Arc<dyn ExecutionPlan>,
    props: Arc<PlanProperties>,
    log: Arc<Mutex<Vec<Decision>>>,
    metrics: ExecutionPlanMetricsSet,
}

impl fmt::Debug for MetalExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalExec").field("op", &self.op).finish()
    }
}

impl MetalExec {
    /// `original` is the node (or node chain) being replaced; its equivalence properties (ordering,
    /// constants) are kept, its partitioning becomes a single partition.
    pub(crate) fn new(
        op: MetalOp,
        input: Arc<dyn ExecutionPlan>,
        original: Arc<dyn ExecutionPlan>,
        log: Arc<Mutex<Vec<Decision>>>,
    ) -> Self {
        let mut eq = original.equivalence_properties().clone();
        eq.clear_per_partition_constants();
        let props = PlanProperties::new(
            eq,
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );
        Self { op, input, original, props: Arc::new(props), log, metrics: ExecutionPlanMetricsSet::new() }
    }

    pub fn op(&self) -> &MetalOp {
        &self.op
    }

    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }
}

impl DisplayAs for MetalExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        let schema = self.input.schema();
        let name = |i: usize| schema.field(i).name().clone();
        match &self.op {
            MetalOp::Sort { keys, fetch } => {
                let ks: Vec<String> = keys
                    .iter()
                    .map(|k| {
                        format!(
                            "{} {} NULLS {}",
                            name(k.column),
                            if k.descending { "DESC" } else { "ASC" },
                            if k.nulls_first { "FIRST" } else { "LAST" }
                        )
                    })
                    .collect();
                write!(f, "MetalExec: sort=[{}]", ks.join(", "))?;
                if let Some(n) = fetch {
                    write!(f, ", fetch={n}")?;
                }
                Ok(())
            }
            MetalOp::Aggregate { keys, aggs } => {
                let ks: Vec<String> = keys.iter().map(|&k| name(k)).collect();
                let asx: Vec<String> = aggs
                    .iter()
                    .map(|a| format!("{:?}({})", a.kind, a.column.map(name).unwrap_or_else(|| "*".into())))
                    .collect();
                write!(f, "MetalExec: group_by=[{}], aggr=[{}]", ks.join(", "), asx.join(", "))
            }
            MetalOp::Filter { predicate, projection, .. } => {
                write!(f, "MetalExec: filter={predicate}")?;
                if let Some(p) = projection {
                    write!(f, ", projection={p:?}")?;
                }
                Ok(())
            }
        }
    }
}

/// `plan` with every occurrence of the node `old` (by pointer) replaced by `new`.
fn replace_leaf(
    plan: &Arc<dyn ExecutionPlan>,
    old: &Arc<dyn ExecutionPlan>,
    new: &Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    if Arc::ptr_eq(plan, old) {
        return Ok(Arc::clone(new));
    }
    Ok(Arc::clone(plan)
        .transform_down(|n| {
            if Arc::ptr_eq(&n, old) {
                Ok(Transformed::new(Arc::clone(new), true, TreeNodeRecursion::Jump))
            } else {
                Ok(Transformed::no(n))
            }
        })?
        .data)
}

impl ExecutionPlan for MetalExec {
    fn name(&self) -> &str {
        "MetalExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.props
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return internal_err!("MetalExec takes one child, got {}", children.len());
        }
        let input = children.swap_remove(0);
        let original = replace_leaf(&self.original, &self.input, &input)?;
        Ok(Arc::new(MetalExec {
            op: self.op.clone(),
            input,
            original,
            props: Arc::clone(&self.props),
            log: Arc::clone(&self.log),
            metrics: ExecutionPlanMetricsSet::new(),
        }))
    }

    fn execute(&self, partition: usize, ctx: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return internal_err!("MetalExec has one partition, asked for {partition}");
        }
        let input = Arc::clone(&self.input);
        let original = Arc::clone(&self.original);
        let op = self.op.clone();
        let schema = self.schema();
        let log = Arc::clone(&self.log);
        let batch_size = ctx.session_config().batch_size();

        // Coalescing runs the input partitions concurrently, as DataFusion's own merge does.
        let source: Arc<dyn ExecutionPlan> = if input.output_partitioning().partition_count() > 1 {
            Arc::new(CoalescePartitionsExec::new(Arc::clone(&input)))
        } else {
            Arc::clone(&input)
        };
        let mut stream = source.execute(0, Arc::clone(&ctx))?;

        // Where the time goes (EXPLAIN ANALYZE shows these): waiting for and collecting the input
        // stream (includes the upstream operators' own work), and the GPU call split into import
        // (the chunked copy into Metal buffers) / plan run / export.
        let m = |name: &'static str| MetricBuilder::new(&self.metrics).subset_time(name, partition);
        let input_time = m("input_time");
        let (import_time, kernel_time, export_time) = (m("import_time"), m("kernel_time"), m("export_time"));
        let input_batches = MetricBuilder::new(&self.metrics).counter("input_batches", partition);

        let out_schema = Arc::clone(&schema);
        let fut = async move {
            let reservation = MemoryConsumer::new("MetalExec").register(ctx.memory_pool());
            let mut batches = Vec::new();
            let t = std::time::Instant::now();
            while let Some(b) = stream.next().await {
                let b = b?;
                reservation.try_grow(b.get_array_memory_size())?;
                batches.push(b);
            }
            input_time.add_duration(t.elapsed());
            input_batches.add(batches.len());
            let in_schema = input.schema();

            let op2 = op.clone();
            let s2 = Arc::clone(&out_schema);
            let (res, batches, times) = tokio::task::spawn_blocking(move || {
                let r = crate::gpu::run(&op2, &batches, &s2);
                (r, batches, crate::gpu::take_times())
            })
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
            import_time.add_duration(times.import);
            kernel_time.add_duration(times.kernel);
            export_time.add_duration(times.export);

            let out: Vec<RecordBatch> = match res {
                Ok(b) => {
                    drop(batches);
                    let mut v = Vec::new();
                    let mut off = 0;
                    while off < b.num_rows() {
                        let n = batch_size.min(b.num_rows() - off);
                        v.push(b.slice(off, n));
                        off += n;
                    }
                    v
                }
                Err(msg) => {
                    log.lock().unwrap().push(Decision::runtime_fallback(&op, &msg));
                    let mem: Arc<dyn ExecutionPlan> =
                        MemorySourceConfig::try_new_exec(&[batches], in_schema, None)?;
                    let plan = replace_leaf(&original, &input, &mem)?;
                    collect(plan, ctx).await?
                }
            };
            drop(reservation);
            Ok::<_, DataFusionError>(futures::stream::iter(out.into_iter().map(Ok)))
        };
        let s = futures::stream::once(fut).try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, s)))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}
