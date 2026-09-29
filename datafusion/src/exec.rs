//! `MetalExec`: the `ExecutionPlan` that runs a replaced node on ArrowMetal.

use std::fmt;
use std::sync::{Arc, Mutex};

use arrow::array::{Array, AsArray};
use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::common::runtime::SpawnedTask;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{internal_err, DataFusionError, Result};
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{Count, ExecutionPlanMetricsSet, Gauge, MetricBuilder, MetricsSet, Time};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    execute_stream, DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties, SendableRecordBatchStream,
};
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};

use crate::rule::{lock, AggregateChoice, Decision, SharedLog};

/// One sort key: an input column, a direction, and where its nulls go.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SortKey {
    /// The input column.
    pub column: usize,
    /// Descending order.
    pub descending: bool,
    /// Nulls before every value.
    pub nulls_first: bool,
}

/// An aggregate function `MetalExec` computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AggKind {
    /// `sum`.
    Sum,
    /// `min`.
    Min,
    /// `max`.
    Max,
    /// `count(col)`: non-null values.
    Count,
    /// `count(*)`: rows.
    CountAll,
    /// `avg`.
    Mean,
}

/// One aggregate of a [`MetalOp::Aggregate`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AggSpec {
    /// The function.
    pub kind: AggKind,
    /// The argument column (`None` for `count(*)`).
    pub column: Option<usize>,
    /// The argument column is floating point (min/max get the NaN / signed-zero fix-up).
    pub float: bool,
    /// The type DataFusion's schema gives the result.
    pub out_type: DataType,
}

/// What a `MetalExec` computes, in terms of its input's column indices.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum MetalOp {
    /// A sort (`fetch`: under a LIMIT).
    Sort {
        /// The ORDER BY keys, first to last.
        keys: Vec<SortKey>,
        /// The LIMIT, if any.
        fetch: Option<usize>,
    },
    /// A GROUP BY over column keys (no aggregates: DISTINCT).
    Aggregate {
        /// The key columns.
        keys: Vec<usize>,
        /// The aggregates, in output order after the keys.
        aggs: Vec<AggSpec>,
    },
    /// A filter, with the projection DataFusion's `FilterExec` carried.
    Filter {
        /// An ArrowMetal s-expression over columns named `c{i}`.
        predicate: String,
        /// The output columns, if the filter projects.
        projection: Option<Vec<usize>>,
        /// The float columns the predicate compares with a literal (checked for negative NaN at
        /// run time).
        float_compared: Vec<usize>,
    },
}

/// Runs one [`MetalOp`] on ArrowMetal.
///
/// Collects its input, hands the batches to ArrowMetal's chunked import (no `concat_batches`),
/// runs the operation on the GPU through ArrowMetal's plan runner, and emits the result in
/// `batch_size` slices as a single partition.
///
/// The replaced subtree (`original`) runs instead, over the batches already collected and the
/// rest of the input, when:
///
/// * a replaced aggregate's run-time choice hands it back ([`AggregateChoice`]);
/// * ArrowMetal returns an error, or the data holds values the GPU path cannot answer exactly
///   (a runtime fallback);
/// * the session's memory pool refuses the reservation for the input or the result (DataFusion's
///   own operators can spill).
///
/// Each of these is recorded in the rule's report. The GPU call runs on a blocking thread and is
/// not interrupted when the query is dropped: it finishes, and its result is discarded.
pub struct MetalExec {
    op: MetalOp,
    input: Arc<dyn ExecutionPlan>,
    /// The subtree this node replaced, with `input` as its leaf: the runtime fallback.
    original: Arc<dyn ExecutionPlan>,
    props: Arc<PlanProperties>,
    log: SharedLog,
    /// The plan (of the rule's log) this node belongs to.
    plan: u64,
    metrics: ExecutionPlanMetricsSet,
    /// For an aggregate: who runs it (decided at run time under `AggregateChoice::Measured`).
    choice: AggregateChoice,
    /// Look the run-time decision up at this row count instead of the input's.
    table_rows: Option<usize>,
    /// The input's row count as the plan's statistics give it (exact, or an accepted estimate).
    rows_hint: Option<usize>,
}

/// How a replaced aggregate is decided (see [`AggregateChoice`]).
#[derive(Debug, Clone, Copy)]
pub(crate) struct AggSettings {
    pub choice: AggregateChoice,
    /// Look the table up at this row count instead of the input's.
    pub table_rows: Option<usize>,
    /// The input's row count as the plan's statistics give it.
    pub rows_hint: Option<usize>,
}

/// The per-phase metrics of one execution.
struct PhaseMetrics {
    input_time: Time,
    import_time: Time,
    kernel_time: Time,
    export_time: Time,
    probe_time: Time,
    input_batches: Count,
    handed_back: Count,
    groups_estimate: Gauge,
}

impl PhaseMetrics {
    fn add(&self, t: &crate::gpu::GpuTimes) {
        self.import_time.add_duration(t.import);
        self.kernel_time.add_duration(t.kernel);
        self.export_time.add_duration(t.export);
    }
}

/// `b` in slices of at most `batch_size` rows.
fn slices(b: RecordBatch, batch_size: usize) -> Vec<RecordBatch> {
    let mut v = Vec::new();
    let mut off = 0;
    while off < b.num_rows() {
        let n = batch_size.min(b.num_rows() - off);
        v.push(b.slice(off, n));
        off += n;
    }
    v
}

/// The bytes a batch's columns reference: for a slice of a larger buffer, only the slice (so the
/// slices of one table are not each counted at the table's size), without allocating.
pub(crate) fn batch_bytes(b: &RecordBatch) -> usize {
    b.columns().iter().map(|a| array_bytes(a.as_ref())).sum()
}

fn array_bytes(a: &dyn Array) -> usize {
    let n = a.len();
    let nulls = a.nulls().map_or(0, |_| n.div_ceil(8));
    let values = match a.data_type() {
        DataType::Boolean => n.div_ceil(8),
        DataType::Utf8 => {
            let o = a.as_string::<i32>().value_offsets();
            (n + 1) * 4 + (o[n] - o[0]) as usize
        }
        DataType::LargeUtf8 => {
            let o = a.as_string::<i64>().value_offsets();
            (n + 1) * 8 + (o[n] - o[0]) as usize
        }
        DataType::Utf8View => {
            let v = a.as_string_view();
            n * 16 + v.views().iter().map(|w| *w as u32 as usize).filter(|&l| l > 12).sum::<usize>()
        }
        t => match t.primitive_width() {
            Some(w) => n * w,
            None => return a.get_array_memory_size(),
        },
    };
    values + nulls
}

/// What collecting one or more input streams gave.
struct Collected {
    /// The batches of each stream, in order.
    parts: Vec<Vec<RecordBatch>>,
    /// The rest of each stream, where the memory pool refused a batch.
    rest: Vec<Option<SendableRecordBatchStream>>,
    /// The reservations holding `parts`.
    held: Vec<MemoryReservation>,
    /// Why the pool refused, if it did.
    refused: Option<String>,
}

impl Collected {
    fn rows(&self) -> usize {
        self.parts.iter().flatten().map(|b| b.num_rows()).sum()
    }
}

/// Rows of the prefix a replaced aggregate decides from under `AggregateChoice::Measured`: the
/// first batches of each input partition, together at least this many rows (the probe samples at
/// most a quarter of them).
const PREFIX_ROWS: usize = 262_144;

/// The largest sample the run-time choice draws. A range still not settled at this size hands the
/// node back: at 5M and 10M rows with about rows / 2 groups, growing the sample to 32,768 rows took
/// 0.35 to 0.53 ms, 4 % to 8 % of DataFusion's time for the query.
const DECIDE_SAMPLE: usize = 8_192;

/// The largest sample of the whole input that confirms a take decided from the prefix.
const CONFIRM_SAMPLE: usize = 2_048;

/// Drains `streams`, reserving each batch in the memory pool: every stream to its end (one task
/// each), or with `quota`, each until it has given at least `quota` rows (polled together in
/// this task; the rest of each stream is kept). A stream whose reservation is refused stops there
/// and keeps the rest of its stream.
async fn collect_streams(
    streams: Vec<SendableRecordBatchStream>,
    ctx: &TaskContext,
    quota: Option<usize>,
) -> Result<Collected> {
    let drain = |mut s: SendableRecordBatchStream, r: MemoryReservation| async move {
        let mut v = Vec::new();
        let mut rows = 0usize;
        while let Some(b) = s.next().await {
            let b = b?;
            rows += b.num_rows();
            let refused = r.try_grow(batch_bytes(&b)).err().map(|e| e.to_string());
            v.push(b);
            if refused.is_some() || quota.is_some_and(|q| rows >= q) {
                return Ok::<_, DataFusionError>((v, Some(s), r, refused));
            }
        }
        Ok((v, None, r, None))
    };
    let reservation = || MemoryConsumer::new("MetalExec").register(ctx.memory_pool());
    let results: Vec<_> = if quota.is_some() || streams.len() == 1 {
        futures::future::join_all(streams.into_iter().map(|s| drain(s, reservation())))
            .await
            .into_iter()
            .collect::<Result<_>>()?
    } else {
        let tasks: Vec<_> = streams.into_iter().map(|s| SpawnedTask::spawn(drain(s, reservation()))).collect();
        let mut v = Vec::with_capacity(tasks.len());
        for t in tasks {
            v.push(t.join().await.map_err(|e| DataFusionError::External(Box::new(e)))??);
        }
        v
    };
    let mut out = Collected { parts: Vec::new(), rest: Vec::new(), held: Vec::new(), refused: None };
    for (v, rest, r, refused) in results {
        out.parts.push(v);
        out.rest.push(rest);
        out.held.push(r);
        if out.refused.is_none() {
            out.refused = refused;
        }
    }
    if out.parts.is_empty() {
        out.parts.push(Vec::new());
        out.rest.push(None);
    }
    Ok(out)
}

/// `c` with the rest of each of its streams collected (a stream refused by the pool stays open,
/// and `refused` says why).
async fn finish(mut c: Collected, ctx: &TaskContext) -> Result<Collected> {
    let open: Vec<usize> = (0..c.rest.len()).filter(|&i| c.rest[i].is_some()).collect();
    if open.is_empty() {
        return Ok(c);
    }
    let streams: Vec<SendableRecordBatchStream> = open.iter().filter_map(|&i| c.rest[i].take()).collect();
    let more = collect_streams(streams, ctx, None).await?;
    for ((i, v), rest) in open.into_iter().zip(more.parts).zip(more.rest) {
        c.parts[i].extend(v);
        c.rest[i] = rest;
    }
    c.held.extend(more.held);
    if c.refused.is_none() {
        c.refused = more.refused;
    }
    Ok(c)
}

/// A leaf that replays collected batches, then the rest of their streams: the input of the
/// replaced subtree when `MetalExec` hands a node back. One partition per collected stream; each
/// partition can be executed once.
/// One partition of a [`ReplayExec`]: the collected batches and the rest of their stream.
type ReplaySlot = Option<(Vec<RecordBatch>, Option<SendableRecordBatchStream>)>;

struct ReplayExec {
    schema: SchemaRef,
    slots: Mutex<Vec<ReplaySlot>>,
    props: Arc<PlanProperties>,
}

impl fmt::Debug for ReplayExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReplayExec").finish()
    }
}

impl ReplayExec {
    fn new(schema: SchemaRef, parts: Vec<Vec<RecordBatch>>, rest: Vec<Option<SendableRecordBatchStream>>) -> Self {
        let n = parts.len();
        let props = PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(n),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        let slots = parts.into_iter().zip(rest).map(Some).collect();
        Self { schema, slots: Mutex::new(slots), props: Arc::new(props) }
    }
}

impl DisplayAs for ReplayExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ReplayExec")
    }
}

impl ExecutionPlan for ReplayExec {
    fn name(&self) -> &str {
        "ReplayExec"
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        &self.props
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }
    fn with_new_children(self: Arc<Self>, children: Vec<Arc<dyn ExecutionPlan>>) -> Result<Arc<dyn ExecutionPlan>> {
        if children.is_empty() {
            Ok(self)
        } else {
            internal_err!("ReplayExec has no children")
        }
    }
    fn execute(&self, partition: usize, _ctx: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
        let slot = self.slots.lock().unwrap_or_else(|p| p.into_inner()).get_mut(partition).and_then(Option::take);
        let Some((batches, rest)) = slot else {
            return internal_err!("ReplayExec partition {partition} executed twice or out of range");
        };
        let head = futures::stream::iter(batches.into_iter().map(Ok));
        let s: BoxStream<'static, Result<RecordBatch>> = match rest {
            Some(r) => Box::pin(head.chain(r)),
            None => Box::pin(head),
        };
        Ok(Box::pin(RecordBatchStreamAdapter::new(Arc::clone(&self.schema), s)))
    }
}

impl fmt::Debug for MetalExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalExec").field("op", &self.op).finish()
    }
}

/// Everything one execution needs, moved into its future.
struct Job {
    op: MetalOp,
    input: Arc<dyn ExecutionPlan>,
    original: Arc<dyn ExecutionPlan>,
    out_schema: SchemaRef,
    log: SharedLog,
    plan: u64,
    choice: AggregateChoice,
    table_rows: Option<usize>,
    rows_hint: Option<usize>,
    mm: PhaseMetrics,
    ctx: Arc<TaskContext>,
}

impl Job {
    fn record(&self, d: Decision) {
        lock(&self.log).push(self.plan, d);
    }

    /// The replaced subtree run by DataFusion over `c` (the collected batches, then the rest of
    /// each stream), one partition per collected stream, coalesced to one stream. `c`'s
    /// reservations are released first: DataFusion's operators reserve what they buffer as they
    /// consume the replayed batches (and can spill), which they could not with this node still
    /// holding the pool.
    fn hand_back(&self, c: Collected) -> Result<BoxStream<'static, Result<RecordBatch>>> {
        drop(c.held);
        let leaf: Arc<dyn ExecutionPlan> = Arc::new(ReplayExec::new(self.input.schema(), c.parts, c.rest));
        let plan = replace_leaf(&self.original, &self.input, &leaf)?;
        Ok(Box::pin(execute_stream(plan, Arc::clone(&self.ctx))?))
    }

    /// Runs the op on ArrowMetal over `c`; on an error, a panic, a data-dependent refusal or a
    /// refused result reservation, hands the node back instead (recorded).
    async fn run_gpu(self, c: Collected) -> Result<BoxStream<'static, Result<RecordBatch>>> {
        let op = self.op.clone();
        let s2 = Arc::clone(&self.out_schema);
        let parts = c.parts;
        let (res, parts, times) = tokio::task::spawn_blocking(move || {
            let refs: Vec<&RecordBatch> = parts.iter().flatten().collect();
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| crate::gpu::run(&op, &refs, &s2)))
                .unwrap_or_else(|p| {
                    let msg = p
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_default();
                    Err(format!("panic in the GPU path: {msg}"))
                });
            (r, parts, crate::gpu::take_times())
        })
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
        self.mm.add(&times);
        let c = Collected { parts, ..c };
        match res {
            Ok(b) => {
                let out = MemoryConsumer::new("MetalExec output").register(self.ctx.memory_pool());
                if let Err(e) = out.try_grow(batch_bytes(&b)) {
                    self.record(Decision::memory_hand_back(&self.op, "the result", &e.to_string()));
                    self.mm.handed_back.add(1);
                    return self.hand_back(c);
                }
                // The input is released; the result's reservation is held until the stream ends.
                drop(c);
                let batch_size = self.ctx.session_config().batch_size();
                Ok(Box::pin(futures::stream::iter(slices(b, batch_size).into_iter().map(Ok)).map(move |b| {
                    let _held = &out;
                    b
                })))
            }
            Err(msg) => {
                self.record(Decision::runtime_fallback(&self.op, &msg));
                self.hand_back(c)
            }
        }
    }

    /// Sort and filter: the input coalesced into one stream, collected, run on ArrowMetal.
    async fn run_sort_or_filter(self, stream: SendableRecordBatchStream) -> Result<BoxStream<'static, Result<RecordBatch>>> {
        let t = std::time::Instant::now();
        let c = collect_streams(vec![stream], &self.ctx, None).await?;
        self.mm.input_time.add_duration(t.elapsed());
        self.mm.input_batches.add(c.parts.iter().map(|p| p.len()).sum());
        if let Some(e) = &c.refused {
            self.record(Decision::memory_hand_back(&self.op, "the input", e));
            self.mm.handed_back.add(1);
            return self.hand_back(c);
        }
        self.run_gpu(c).await
    }

    /// An aggregate: the run-time choice, then ArrowMetal or the hand-back.
    ///
    /// * `ArrowMetal`: every input partition collected (concurrently, each kept as its own
    ///   partition), then the GPU.
    /// * `DataFusion`: handed back at once; the replaced subtree reads the input streams.
    /// * `Measured`, with the input's row count known from the plan: the first batches of each
    ///   partition (`PREFIX_ROWS` together) are collected and probed. A hand-back replays them
    ///   and streams the rest of each partition into DataFusion's own operators. A take collects
    ///   the rest and confirms the choice with a probe over the whole input (a prefix of data
    ///   ordered by its keys shows fewer groups than the input holds) before the GPU runs.
    /// * `Measured` without a row count: everything collected, probed, then either.
    async fn run_aggregate(self, streams: Vec<SendableRecordBatchStream>) -> Result<BoxStream<'static, Result<RecordBatch>>> {
        let MetalOp::Aggregate { keys, .. } = &self.op else {
            return internal_err!("run_aggregate on {:?}", self.op);
        };
        let keys = keys.clone();
        let t = std::time::Instant::now();
        if self.choice == AggregateChoice::DataFusion {
            let n = streams.len();
            let c = Collected {
                parts: vec![Vec::new(); n],
                rest: streams.into_iter().map(Some).collect(),
                held: Vec::new(),
                refused: None,
            };
            self.record(Decision::runtime_choice(&self.op, false, "aggregate_choice is DataFusion".into(), self.rows_hint.unwrap_or(0), None));
            self.mm.handed_back.add(1);
            return self.hand_back(c);
        }
        let prefix = match (self.choice, self.rows_hint) {
            (AggregateChoice::Measured, Some(_)) => Some(PREFIX_ROWS.div_ceil(streams.len().max(1))),
            _ => None,
        };
        let mut c = collect_streams(streams, &self.ctx, prefix).await?;
        if let Some(e) = &c.refused {
            self.mm.input_time.add_duration(t.elapsed());
            self.record(Decision::memory_hand_back(&self.op, "the input", e));
            self.mm.handed_back.add(1);
            return self.hand_back(c);
        }
        let mut from_prefix: Option<(String, crate::probe::GroupEstimate)> = None;
        if let (Some(_), Some(total)) = (prefix, self.rows_hint) {
            if c.rest.iter().any(Option::is_some) {
                // Decide from the prefix.
                self.mm.input_time.add_duration(t.elapsed());
                let t = std::time::Instant::now();
                let refs: Vec<&RecordBatch> = c.parts.iter().flatten().collect();
                let (take, reason, estimate) = decide(&self.op, self.input.as_ref(), &refs, &keys, total, self.table_rows);
                self.mm.probe_time.add_duration(t.elapsed());
                if !take {
                    if let Some(e) = &estimate {
                        self.mm.groups_estimate.set(e.estimate as usize);
                    }
                    let reason = format!("{reason} (from the first {} rows of each partition)", prefix.unwrap_or(0));
                    self.record(Decision::runtime_choice(&self.op, false, reason, total, estimate));
                    self.mm.handed_back.add(1);
                    return self.hand_back(c);
                }
                if let Some(e) = estimate {
                    from_prefix = Some((reason, e));
                }
                let t = std::time::Instant::now();
                c = finish(c, &self.ctx).await?;
                self.mm.input_time.add_duration(t.elapsed());
                if let Some(e) = &c.refused {
                    self.record(Decision::memory_hand_back(&self.op, "the input", e));
                    self.mm.handed_back.add(1);
                    return self.hand_back(c);
                }
            } else {
                self.mm.input_time.add_duration(t.elapsed());
            }
        } else {
            self.mm.input_time.add_duration(t.elapsed());
        }
        self.mm.input_batches.add(c.parts.iter().map(|p| p.len()).sum());
        let rows = c.rows();
        let t = std::time::Instant::now();
        let (on_gpu, reason, estimate) = match (self.choice, from_prefix) {
            (AggregateChoice::Measured, Some((why, e))) => {
                // Taken from the prefix: a small sample of the whole input must agree (its range
                // must meet the prefix's). Data ordered by its keys shows fewer groups in a prefix
                // than the input holds; a sample spread over the whole input sees them.
                let refs: Vec<&RecordBatch> = c.parts.iter().flatten().collect();
                let f = crate::probe::estimate_up_to(&refs, &keys, None, Some(rows), CONFIRM_SAMPLE);
                let agree = f.low <= e.high && f.high >= e.low;
                let check = format!(
                    "a {}-row sample of the whole input puts it at {} to {} groups",
                    f.sample_rows, f.low, f.high
                );
                if agree {
                    (true, format!("{why} (from the first rows of each partition; {check})"), Some(e))
                } else {
                    (false, format!("{why} from the first rows of each partition, but {check}; left to DataFusion"), Some(f))
                }
            }
            (AggregateChoice::Measured, None) => {
                let refs: Vec<&RecordBatch> = c.parts.iter().flatten().collect();
                decide(&self.op, self.input.as_ref(), &refs, &keys, rows, self.table_rows)
            }
            _ => (true, "aggregate_choice is ArrowMetal".to_string(), None),
        };
        self.mm.probe_time.add_duration(t.elapsed());
        if let Some(e) = &estimate {
            self.mm.groups_estimate.set(e.estimate as usize);
        }
        self.record(Decision::runtime_choice(&self.op, on_gpu, reason, rows, estimate));
        if !on_gpu {
            self.mm.handed_back.add(1);
            return self.hand_back(c);
        }
        self.run_gpu(c).await
    }
}

impl MetalExec {
    /// `original` is the node (or node chain) being replaced; its equivalence properties (ordering,
    /// constants) are kept, its partitioning becomes a single partition.
    pub(crate) fn new(
        op: MetalOp,
        input: Arc<dyn ExecutionPlan>,
        original: Arc<dyn ExecutionPlan>,
        (log, plan): (SharedLog, u64),
        agg: AggSettings,
    ) -> Self {
        let AggSettings { choice, table_rows, rows_hint } = agg;
        let mut eq = original.equivalence_properties().clone();
        eq.clear_per_partition_constants();
        let props = PlanProperties::new(
            eq,
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );
        Self {
            op,
            input,
            original,
            props: Arc::new(props),
            log,
            plan,
            metrics: ExecutionPlanMetricsSet::new(),
            choice,
            table_rows,
            rows_hint,
        }
    }

    /// The operation this node runs.
    pub fn op(&self) -> &MetalOp {
        &self.op
    }

    /// The input it reads.
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// The per-phase metrics (EXPLAIN ANALYZE shows them): waiting for and collecting the input
    /// (includes the upstream operators' own work), the group-count probe, and the GPU call split
    /// into import (the chunked copy into Metal buffers) / plan run / export.
    fn phase_metrics(&self) -> PhaseMetrics {
        let m = |name: &'static str| MetricBuilder::new(&self.metrics).subset_time(name, 0);
        PhaseMetrics {
            input_time: m("input_time"),
            import_time: m("import_time"),
            kernel_time: m("kernel_time"),
            export_time: m("export_time"),
            probe_time: m("probe_time"),
            input_batches: MetricBuilder::new(&self.metrics).counter("input_batches", 0),
            handed_back: MetricBuilder::new(&self.metrics).counter("handed_back", 0),
            groups_estimate: MetricBuilder::new(&self.metrics).gauge("groups_estimate", 0),
        }
    }
}

/// The run-time decision under `AggregateChoice::Measured`: (on ArrowMetal, why, the estimate).
fn decide(
    op: &MetalOp,
    input: &dyn ExecutionPlan,
    refs: &[&RecordBatch],
    keys: &[usize],
    rows: usize,
    table_rows: Option<usize>,
) -> (bool, String, Option<crate::probe::GroupEstimate>) {
    let Some(shape) = crate::choice::shape(op, input) else {
        return (false, "not an aggregate".into(), None);
    };
    let at = table_rows.unwrap_or(rows) as u64;
    let settled = crate::choice::settled_for(&shape, at);
    let e = crate::probe::estimate_up_to(refs, keys, Some(&settled), Some(rows), DECIDE_SAMPLE);
    let how = if e.exact {
        format!("{} groups, counted over all {} rows", e.estimate, e.rows)
    } else {
        format!("an estimated {} groups ({} to {}, Chao1 over a {}-row sample)", e.estimate, e.low, e.high, e.sample_rows)
    };
    let at_text = match table_rows {
        Some(n) => format!("{} rows (table_rows; input {rows})", crate::choice::rows_text(n as u64)),
        None => format!("{rows} rows"),
    };
    if !settled(e.low, e.high) {
        return (
            false,
            format!("{how} at {at_text}: the range reaches group counts the table decides differently; left to DataFusion"),
            Some(e),
        );
    }
    let b = crate::choice::bucket(e.estimate, at);
    match crate::choice::takes(&shape, b, at) {
        Ok(why) => (true, format!("{how} at {at_text}: {why}"), Some(e)),
        Err(why) => (false, format!("{how} at {at_text}: {why}; left to DataFusion"), Some(e)),
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
            plan: self.plan,
            metrics: ExecutionPlanMetricsSet::new(),
            choice: self.choice,
            table_rows: self.table_rows,
            rows_hint: self.rows_hint,
        }))
    }

    fn execute(&self, partition: usize, ctx: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return internal_err!("MetalExec has one partition, asked for {partition}");
        }
        let schema = self.schema();
        let job = Job {
            op: self.op.clone(),
            input: Arc::clone(&self.input),
            original: Arc::clone(&self.original),
            out_schema: Arc::clone(&schema),
            log: Arc::clone(&self.log),
            plan: self.plan,
            choice: self.choice,
            table_rows: self.table_rows,
            rows_hint: self.rows_hint,
            mm: self.phase_metrics(),
            ctx: Arc::clone(&ctx),
        };
        let fut: futures::future::BoxFuture<'static, Result<BoxStream<'static, Result<RecordBatch>>>> =
            if matches!(self.op, MetalOp::Aggregate { .. }) {
                let parts = self.input.output_partitioning().partition_count();
                let mut streams = Vec::with_capacity(parts);
                for p in 0..parts {
                    streams.push(self.input.execute(p, Arc::clone(&ctx))?);
                }
                Box::pin(job.run_aggregate(streams))
            } else {
                // Coalescing runs the input partitions concurrently, as DataFusion's own merge does.
                let source: Arc<dyn ExecutionPlan> = if self.input.output_partitioning().partition_count() > 1 {
                    Arc::new(CoalescePartitionsExec::new(Arc::clone(&self.input)))
                } else {
                    Arc::clone(&self.input)
                };
                let stream = source.execute(0, Arc::clone(&ctx))?;
                Box::pin(job.run_sort_or_filter(stream))
            };
        let s = futures::stream::once(fut).try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, s)))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}
