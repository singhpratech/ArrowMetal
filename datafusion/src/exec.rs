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
use futures::{FutureExt, StreamExt, TryStreamExt};

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
    /// The argument column is floating point (min/max keep DataFusion's NaN, signed-zero and
    /// infinity semantics).
    pub float: bool,
    /// The type DataFusion's schema gives the result.
    pub out_type: DataType,
}

/// Which rows of an equi-join come out (the `JoinType`s of DataFusion's `HashJoinExec` that
/// `MetalExec` runs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum JoinHow {
    /// The pairs of rows whose keys are equal.
    Inner,
    /// The inner pairs, and every left (build-side) row without a match, with nulls on the right.
    Left,
    /// The inner pairs, and every right (probe-side) row without a match, with nulls on the left.
    Right,
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
    },
    /// An equi-join over column keys (DataFusion's `HashJoinExec`), both inputs collected. The
    /// output is DataFusion's: the left input's columns, then the right input's, through
    /// `projection`.
    Join {
        /// Which unmatched rows are kept.
        how: JoinHow,
        /// The key columns of the left (build) input, in the join's `on` order.
        left_keys: Vec<usize>,
        /// The key columns of the right (probe) input, in the same order.
        right_keys: Vec<usize>,
        /// The output columns, as indices into the left input's columns followed by the right's.
        projection: Vec<usize>,
        /// The number of columns of the left input.
        left_columns: usize,
    },
}

/// Runs one [`MetalOp`] on ArrowMetal.
///
/// Collects its input (a join: both inputs), hands the batches to ArrowMetal's chunked import (no
/// `concat_batches`; the columns of one call are imported at the same time, one thread each),
/// runs the operation on the GPU through ArrowMetal's plan runner, and emits the result in
/// `batch_size` slices as a single partition (a join: dealt out to as many partitions as the join
/// it replaced, from one execution).
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
    /// What it reads: one input, or a join's left and right inputs.
    inputs: Vec<Arc<dyn ExecutionPlan>>,
    /// The subtree this node replaced, with `inputs` as its leaves: the runtime fallback.
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
    /// A join's execution, shared by its output partitions.
    join_slot: JoinSlot,
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
    /// How many of the streams belong to each input, in order (one input but for a join).
    per_input: Vec<usize>,
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
    let mut out = Collected { parts: Vec::new(), rest: Vec::new(), held: Vec::new(), refused: None, per_input: Vec::new() };
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
    out.per_input = vec![out.parts.len()];
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
    inputs: Vec<Arc<dyn ExecutionPlan>>,
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
    /// holding the pool. A join's streams are its left input's partitions, then its right's
    /// (`c.per_input`).
    fn hand_back(&self, c: Collected) -> Result<BoxStream<'static, Result<RecordBatch>>> {
        let plan = self.hand_back_plan(c)?;
        Ok(Box::pin(execute_stream(plan, Arc::clone(&self.ctx))?))
    }

    /// The replaced subtree over `c` (see [`hand_back`](Self::hand_back)), not yet executed.
    fn hand_back_plan(&self, c: Collected) -> Result<Arc<dyn ExecutionPlan>> {
        drop(c.held);
        let per_input = if c.per_input.len() == self.inputs.len() { c.per_input } else { vec![c.parts.len()] };
        let (mut parts, mut rest) = (c.parts.into_iter(), c.rest.into_iter());
        let mut plan = Arc::clone(&self.original);
        for (input, n) in self.inputs.iter().zip(per_input) {
            let p: Vec<Vec<RecordBatch>> = parts.by_ref().take(n).collect();
            let r: Vec<Option<SendableRecordBatchStream>> = rest.by_ref().take(n).collect();
            let leaf: Arc<dyn ExecutionPlan> = Arc::new(ReplayExec::new(input.schema(), p, r));
            plan = replace_leaf(&plan, input, &leaf)?;
        }
        Ok(plan)
    }

    /// Runs the op on ArrowMetal over `c`; on an error, a panic, a data-dependent refusal or a
    /// refused result reservation, hands the node back instead (recorded).
    async fn run_gpu(self, c: Collected) -> Result<BoxStream<'static, Result<RecordBatch>>> {
        let ctx = Arc::clone(&self.ctx);
        match self.run_gpu_outcome(c).await? {
            Outcome::Gpu(batches, held) => Ok(Box::pin(futures::stream::iter(batches.into_iter().map(Ok)).map(move |b| {
                let _held = &held;
                b
            }))),
            Outcome::Back(plan) => Ok(Box::pin(execute_stream(plan, ctx)?)),
        }
    }

    /// [`run_gpu`](Self::run_gpu), with the result as `batch_size` slices and the reservation
    /// holding them, or the plan DataFusion runs instead.
    async fn run_gpu_outcome(self, c: Collected) -> Result<Outcome> {
        let op = self.op.clone();
        let s2 = Arc::clone(&self.out_schema);
        let parts = c.parts;
        let per_input = c.per_input.clone();
        let (res, parts, times) = tokio::task::spawn_blocking(move || {
            // The batches of each input (one input but for a join).
            let mut groups: Vec<Vec<&RecordBatch>> = Vec::new();
            let mut it = parts.iter();
            for &n in &per_input {
                groups.push(it.by_ref().take(n).flatten().collect());
            }
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| crate::gpu::run(&op, &groups, &s2)))
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
                    return Ok(Outcome::Back(self.hand_back_plan(c)?));
                }
                // The input is released; the result's reservation is held until the stream ends.
                drop(c);
                let batch_size = self.ctx.session_config().batch_size();
                Ok(Outcome::Gpu(slices(b, batch_size), out))
            }
            Err(msg) => {
                self.record(Decision::runtime_fallback(&self.op, &msg));
                Ok(Outcome::Back(self.hand_back_plan(c)?))
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
                per_input: vec![n],
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
                let (take, reason, estimate) = decide(&self.op, self.inputs[0].as_ref(), &refs, &keys, total, self.table_rows);
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
                let (agree, f) = confirm(&refs, &keys, rows, &e);
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
                decide(&self.op, self.inputs[0].as_ref(), &refs, &keys, rows, self.table_rows)
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

    /// A join: every partition of both inputs collected, each in its own task (all at the same
    /// time), then the GPU. `left` is the number of `streams` that belong to the left input.
    async fn run_join(self, streams: Vec<SendableRecordBatchStream>, left: usize) -> Result<Outcome> {
        let t = std::time::Instant::now();
        let total = streams.len();
        let mut c = collect_streams(streams, &self.ctx, None).await?;
        c.per_input = vec![left, total - left];
        self.mm.input_time.add_duration(t.elapsed());
        self.mm.input_batches.add(c.parts.iter().map(|p| p.len()).sum());
        if let Some(e) = &c.refused {
            self.record(Decision::memory_hand_back(&self.op, "the input", e));
            self.mm.handed_back.add(1);
            return Ok(Outcome::Back(self.hand_back_plan(c)?));
        }
        self.run_gpu_outcome(c).await
    }
}

/// What one execution of a `MetalExec` produced.
enum Outcome {
    /// The GPU's result in `batch_size` slices, and the reservation that holds it.
    Gpu(Vec<RecordBatch>, MemoryReservation),
    /// The replaced subtree, over the collected batches, for DataFusion to run instead.
    Back(Arc<dyn ExecutionPlan>),
}

/// A join's one execution, shared by its output partitions: started by the first partition
/// executed, and taken by each partition once.
type SharedOutcome = futures::future::Shared<futures::future::BoxFuture<'static, std::result::Result<Arc<Outcome>, Arc<DataFusionError>>>>;

/// The shared execution and how many output partitions have taken it.
type JoinSlot = Arc<Mutex<Option<(SharedOutcome, usize)>>>;

/// Output partition `p` of `n` of a shared join execution: every `n`-th GPU slice from the `p`-th,
/// or partition `p` of the replaced subtree.
fn partition_of(shared: SharedOutcome, p: usize, n: usize, ctx: Arc<TaskContext>) -> BoxStream<'static, Result<RecordBatch>> {
    let s = futures::stream::once(shared).map(move |r| -> Result<BoxStream<'static, Result<RecordBatch>>> {
        let out = r.map_err(DataFusionError::Shared)?;
        match &*out {
            Outcome::Gpu(batches, _) => {
                let mine: Vec<RecordBatch> = batches.iter().skip(p).step_by(n).cloned().collect();
                let held = Arc::clone(&out);
                Ok(Box::pin(futures::stream::iter(mine.into_iter().map(Ok)).map(move |b| {
                    let _held = &held;
                    b
                })))
            }
            // `fit_partitions` gave it `n` partitions.
            Outcome::Back(plan) => Ok(Box::pin(plan.execute(p, Arc::clone(&ctx))?)),
        }
    });
    Box::pin(s.try_flatten())
}

/// `plan` with `n` output partitions: as it is, coalesced to one, or split round-robin.
fn fit_partitions(plan: Arc<dyn ExecutionPlan>, n: usize) -> Result<Arc<dyn ExecutionPlan>> {
    let have = plan.output_partitioning().partition_count();
    Ok(if have == n {
        plan
    } else if n == 1 {
        Arc::new(CoalescePartitionsExec::new(plan))
    } else {
        Arc::new(datafusion::physical_plan::repartition::RepartitionExec::try_new(plan, Partitioning::RoundRobinBatch(n))?)
    })
}

impl MetalExec {
    /// `original` is the node (or node chain) being replaced; its equivalence properties (ordering,
    /// constants) are kept, its partitioning becomes a single partition.
    pub(crate) fn new(
        op: MetalOp,
        inputs: Vec<Arc<dyn ExecutionPlan>>,
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
            inputs,
            original,
            props: Arc::new(props),
            log,
            plan,
            metrics: ExecutionPlanMetricsSet::new(),
            choice,
            table_rows,
            rows_hint,
            join_slot: Arc::new(Mutex::new(None)),
        }
    }

    /// A join's `MetalExec` with `n` output partitions: one execution on the GPU, its result
    /// dealt out to the partitions batch by batch, so the operators above consume it in parallel
    /// as they did the `HashJoinExec`'s partitions.
    pub(crate) fn with_output_partitions(mut self, n: usize) -> Self {
        if matches!(self.op, MetalOp::Join { .. }) && n > 1 {
            let props = PlanProperties::new(
                self.props.eq_properties.clone(),
                Partitioning::UnknownPartitioning(n),
                EmissionType::Final,
                Boundedness::Bounded,
            );
            self.props = Arc::new(props);
        }
        self
    }

    /// Everything one execution needs.
    fn job(&self, ctx: &Arc<TaskContext>) -> Job {
        Job {
            op: self.op.clone(),
            inputs: self.inputs.clone(),
            original: Arc::clone(&self.original),
            out_schema: self.schema(),
            log: Arc::clone(&self.log),
            plan: self.plan,
            choice: self.choice,
            table_rows: self.table_rows,
            rows_hint: self.rows_hint,
            mm: self.phase_metrics(),
            ctx: Arc::clone(ctx),
        }
    }

    /// The operation this node runs.
    pub fn op(&self) -> &MetalOp {
        &self.op
    }

    /// The input it reads (a join's left input).
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.inputs[0]
    }

    /// Every input it reads: one, or a join's left and right inputs.
    pub fn inputs(&self) -> &[Arc<dyn ExecutionPlan>] {
        &self.inputs
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

/// Whether a sample of at most `CONFIRM_SAMPLE` rows of the whole input (`refs`, `rows` rows)
/// agrees with the estimate `e` a take was decided from: their ranges must meet.
pub(crate) fn confirm(
    refs: &[&RecordBatch],
    keys: &[usize],
    rows: usize,
    e: &crate::probe::GroupEstimate,
) -> (bool, crate::probe::GroupEstimate) {
    let f = crate::probe::estimate_up_to(refs, keys, None, Some(rows), CONFIRM_SAMPLE);
    (f.low <= e.high && f.high >= e.low, f)
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
        let schema = self.inputs[0].schema();
        let name = |i: usize| schema.field(i).name().clone();
        match &self.op {
            MetalOp::Join { how, left_keys, right_keys, projection, .. } => {
                let right = self.inputs.get(1).map(|r| r.schema()).unwrap_or_else(|| Arc::clone(&schema));
                let on: Vec<String> = left_keys
                    .iter()
                    .zip(right_keys)
                    .map(|(&l, &r)| format!("({}, {})", name(l), right.field(r).name()))
                    .collect();
                write!(f, "MetalExec: join={how:?}, on=[{}], projection={projection:?}", on.join(", "))
            }
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
        self.inputs.iter().collect()
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false; self.inputs.len()]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != self.inputs.len() {
            return internal_err!("MetalExec takes {} children, got {}", self.inputs.len(), children.len());
        }
        let mut original = Arc::clone(&self.original);
        for (old, new) in self.inputs.iter().zip(&children) {
            original = replace_leaf(&original, old, new)?;
        }
        Ok(Arc::new(MetalExec {
            op: self.op.clone(),
            inputs: children,
            original,
            props: Arc::clone(&self.props),
            log: Arc::clone(&self.log),
            plan: self.plan,
            metrics: ExecutionPlanMetricsSet::new(),
            choice: self.choice,
            table_rows: self.table_rows,
            rows_hint: self.rows_hint,
            join_slot: Arc::new(Mutex::new(None)),
        }))
    }

    fn execute(&self, partition: usize, ctx: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
        let n = self.props.partitioning.partition_count();
        if partition >= n {
            return internal_err!("MetalExec has {n} partition(s), asked for {partition}");
        }
        let schema = self.schema();
        if matches!(self.op, MetalOp::Join { .. }) {
            // One execution for all output partitions: the first partition executed starts it,
            // and a new one starts once every partition has taken the last one.
            let mut slot = self.join_slot.lock().unwrap_or_else(|p| p.into_inner());
            let fresh = slot.as_ref().is_none_or(|(_, taken)| *taken >= n);
            if fresh {
                let job = self.job(&ctx);
                let mut streams = Vec::new();
                let mut left = 0;
                for (i, input) in self.inputs.iter().enumerate() {
                    for p in 0..input.output_partitioning().partition_count() {
                        streams.push(input.execute(p, Arc::clone(&ctx))?);
                    }
                    if i == 0 {
                        left = streams.len();
                    }
                }
                let fut: futures::future::BoxFuture<'static, std::result::Result<Arc<Outcome>, Arc<DataFusionError>>> =
                    Box::pin(async move {
                        let out = match job.run_join(streams, left).await {
                            Ok(Outcome::Back(plan)) => fit_partitions(plan, n).map(Outcome::Back),
                            other => other,
                        };
                        out.map(Arc::new).map_err(Arc::new)
                    });
                *slot = Some((fut.shared(), 0));
            }
            let Some((shared, taken)) = slot.as_mut() else {
                return internal_err!("MetalExec join slot empty");
            };
            *taken += 1;
            let s = partition_of(shared.clone(), partition, n, ctx);
            return Ok(Box::pin(RecordBatchStreamAdapter::new(schema, s)));
        }
        let job = self.job(&ctx);
        let fut: futures::future::BoxFuture<'static, Result<BoxStream<'static, Result<RecordBatch>>>> =
            if matches!(self.op, MetalOp::Aggregate { .. }) {
                // Every partition of the input, each its own stream.
                let input = &self.inputs[0];
                let mut streams = Vec::new();
                for p in 0..input.output_partitioning().partition_count() {
                    streams.push(input.execute(p, Arc::clone(&ctx))?);
                }
                Box::pin(job.run_aggregate(streams))
            } else {
                // Coalescing runs the input partitions concurrently, as DataFusion's own merge does.
                let input = &self.inputs[0];
                let source: Arc<dyn ExecutionPlan> = if input.output_partitioning().partition_count() > 1 {
                    Arc::new(CoalescePartitionsExec::new(Arc::clone(input)))
                } else {
                    Arc::clone(input)
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use arrow::datatypes::{Field, Schema};

    fn batches(keys: Vec<i64>) -> Vec<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
        keys.chunks(8192)
            .map(|c| RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(c.to_vec()))]).unwrap())
            .collect()
    }

    /// A prefix that repeats a few thousand keys, followed by keys seen once: the prefix's estimate
    /// is far below the input's groups, and the whole-input sample disagrees with it.
    #[test]
    fn a_prefix_that_under_counts_is_not_confirmed() {
        let n = 1_200_000usize;
        let head = 262_144usize;
        let keys: Vec<i64> = (0..n).map(|i| if i < head { (i % 20_000) as i64 } else { i as i64 }).collect();
        let all = batches(keys);
        let refs: Vec<&RecordBatch> = all.iter().collect();
        let prefix: Vec<&RecordBatch> = refs[..head / 8192].to_vec();
        let e = crate::probe::estimate_up_to(&prefix, &[0], None, Some(n), 8_192);
        assert!(e.high < 40_000, "{e:?}");
        let (agree, f) = confirm(&refs, &[0], n, &e);
        assert!(!agree, "prefix {e:?} whole {f:?}");
        // The same data in random order: the prefix's range meets the whole input's.
        let mut shuffled: Vec<i64> = (0..n).map(|i| if i < head { (i % 20_000) as i64 } else { i as i64 }).collect();
        let mut state = 0x1234_5678u64;
        for i in (1..n).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            shuffled.swap(i, (state % (i as u64 + 1)) as usize);
        }
        let all = batches(shuffled);
        let refs: Vec<&RecordBatch> = all.iter().collect();
        let prefix: Vec<&RecordBatch> = refs[..head / 8192].to_vec();
        let e = crate::probe::estimate_up_to(&prefix, &[0], None, Some(n), 8_192);
        let (agree, f) = confirm(&refs, &[0], n, &e);
        assert!(agree, "prefix {e:?} whole {f:?}");
    }
}
