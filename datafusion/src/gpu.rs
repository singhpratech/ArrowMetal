//! The GPU half of `MetalExec`: the collected `RecordBatch`es in, one batch out, through
//! ArrowMetal's plan runner.
//!
//! Synchronous. The ArrowMetal handles of one call live and die inside it: the columns are
//! imported on threads of their own (one per column, see [`import`]), and every other call on a
//! handle runs on the calling thread.

use std::sync::{Arc, Mutex};

use arrow::array::{new_null_array, Array as _, ArrayRef, AsArray, Float32Array, Float64Array, Int64Array};
use arrow::compute::{cast, concat};
use arrow::datatypes::{DataType, Float32Type, Float64Type, Int64Type, SchemaRef};
use arrow::record_batch::{RecordBatch, RecordBatchOptions};

use crate::exec::{AggKind, JoinHow, MetalOp};

/// One GPU job at a time in this process. The plan runner is thread-safe per handle set, but
/// serialising keeps concurrent `MetalExec`s (parallel queries, parallel tests) from contending
/// for the one GPU queue.
static GPU: Mutex<()> = Mutex::new(());

/// `inputs` holds the batches of each input: one input, or a join's left and right inputs.
pub(crate) fn run(op: &MetalOp, inputs: &[Vec<&RecordBatch>], out_schema: &SchemaRef) -> Result<RecordBatch, String> {
    let _guard = GPU.lock().unwrap_or_else(|p| p.into_inner());
    take_times();
    // The chunked import takes every non-empty batch as one chunk of each column.
    fn nonempty<'a>(v: &[&'a RecordBatch]) -> Vec<&'a RecordBatch> {
        v.iter().copied().filter(|b| b.num_rows() > 0).collect()
    }
    if let MetalOp::Join { how, left_keys, right_keys, projection, left_columns } = op {
        let (Some(l), Some(r)) = (inputs.first(), inputs.get(1)) else {
            return Err("a join needs two inputs".into());
        };
        let spec = JoinSpec { how: *how, left_keys, right_keys, projection, left_columns: *left_columns };
        return join(&spec, &nonempty(l), &nonempty(r), out_schema);
    }
    let input = inputs.first().map(|v| nonempty(v)).unwrap_or_default();
    if input.is_empty() {
        // A sort, a filter and a grouped aggregate of nothing are all empty.
        return Ok(RecordBatch::new_empty(out_schema.clone()));
    }
    let num_columns = input[0].num_columns();
    let all: Vec<usize> = (0..num_columns).collect();
    match op {
        MetalOp::Sort { keys, fetch } => {
            // One plan key per ORDER BY key, with DataFusion's null placement and arrow-rs's float
            // order (IEEE 754 totalOrder: -NaN < -inf < ... < -0.0 < +0.0 < ... < +inf < +NaN, NaNs
            // by payload, the exact mirror when descending). Integer and string keys ignore
            // `float_order`. A one-key sort under a limit stays a GPU top-k with these options
            // (docs/ENGINE.md, "Sort key options").
            let by: Vec<String> = keys
                .iter()
                .map(|k| {
                    format!(
                        r#"["c{}",{},{{"nulls":"{}","float_order":"total"}}]"#,
                        k.column,
                        k.descending,
                        if k.nulls_first { "first" } else { "last" }
                    )
                })
                .collect();
            let mut node = format!(r#"{{"op":"sort","by":[{}],"input":{}}}"#, by.join(","), scan("t"));
            if let Some(f) = fetch {
                node = format!(r#"{{"op":"limit","count":{f},"input":{node}}}"#);
            }
            node = select_all(num_columns, node);
            let cols = run_plan(&node, &[Src::new("t", &input, "c", &all)])?;
            assemble(cols, out_schema)
        }
        MetalOp::Filter { predicate, projection } => {
            // Float comparisons arrive in totalOrder form (translate.rs, `total_order_comparison`).
            let node = format!(
                r#"{{"op":"filter","predicate":"{}","input":{}}}"#,
                json_escape(predicate),
                scan("t")
            );
            let node = select_all(num_columns, node);
            let cols = run_plan(&node, &[Src::new("t", &input, "c", &all)])?;
            let cols = match projection {
                Some(p) => p.iter().map(|&i| cols[i].clone()).collect(),
                None => cols,
            };
            assemble(cols, out_schema)
        }
        MetalOp::Aggregate { keys, aggs } => aggregate(keys, aggs, &input, out_schema),
        MetalOp::Join { .. } => unreachable!("handled above"),
    }
}

/// True when the non-null values of float column `c` hold NaNs of more than one bit pattern.
/// DataFusion keeps each NaN bit pattern as its own group (`HashValue::canonicalize` folds only
/// -0.0 into +0.0); ArrowMetal puts every NaN in one group.
fn has_distinct_nans(input: &[&RecordBatch], c: usize) -> bool {
    let mut first: Option<u64> = None;
    let mut seen = |x: u64| -> bool {
        match first {
            None => {
                first = Some(x);
                false
            }
            Some(f) => f != x,
        }
    };
    for b in input {
        let a = b.column(c);
        match a.data_type() {
            DataType::Float64 => {
                let p = a.as_primitive::<Float64Type>();
                for i in 0..p.len() {
                    if p.is_valid(i) && p.value(i).is_nan() && seen(p.value(i).to_bits()) {
                        return true;
                    }
                }
            }
            DataType::Float32 => {
                let p = a.as_primitive::<Float32Type>();
                for i in 0..p.len() {
                    if p.is_valid(i) && p.value(i).is_nan() && seen(p.value(i).to_bits() as u64) {
                        return true;
                    }
                }
            }
            _ => return false,
        }
    }
    false
}

/// The values of a float column on which the plan runner's grouped `min`/`max` and DataFusion's
/// can differ: a NaN, -0.0 and +0.0 (non-null values only).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct FloatSpecials {
    pub nan: bool,
    pub neg_zero: bool,
    pub pos_zero: bool,
}

impl FloatSpecials {
    fn or(self, o: Self) -> Self {
        Self { nan: self.nan || o.nan, neg_zero: self.neg_zero || o.neg_zero, pos_zero: self.pos_zero || o.pos_zero }
    }
}

/// Bit tests over one slice of values: (NaN seen, -0.0 seen, +0.0 seen) as 0/1 accumulators, with
/// no branch per value so the loop vectorises. `abs` clears the sign bit, `inf` is +inf's pattern.
fn scan_bits<T: Copy>(v: &[T], bits: impl Fn(T) -> u64, abs: u64, inf: u64, sign: u64) -> FloatSpecials {
    let (mut nan, mut nz, mut pz) = (0u64, 0u64, 0u64);
    for &x in v {
        let b = bits(x);
        nan |= ((b & abs) > inf) as u64;
        nz |= (b == sign) as u64;
        pz |= (b == 0) as u64;
    }
    FloatSpecials { nan: nan != 0, neg_zero: nz != 0, pos_zero: pz != 0 }
}

fn specials_of(a: &dyn arrow::array::Array) -> FloatSpecials {
    // A column with nulls is read value by value (the slots under a null hold any value).
    match a.data_type() {
        DataType::Float64 => {
            let p = a.as_primitive::<Float64Type>();
            if p.null_count() == 0 {
                scan_bits(p.values(), |x: f64| x.to_bits(), 0x7FFF_FFFF_FFFF_FFFF, 0x7FF0_0000_0000_0000, 1 << 63)
            } else {
                let mut s = FloatSpecials::default();
                for x in p.iter().flatten() {
                    s = s.or(FloatSpecials { nan: x.is_nan(), neg_zero: x == 0.0 && x.is_sign_negative(), pos_zero: x == 0.0 && x.is_sign_positive() });
                }
                s
            }
        }
        DataType::Float32 => {
            let p = a.as_primitive::<Float32Type>();
            if p.null_count() == 0 {
                scan_bits(p.values(), |x: f32| x.to_bits() as u64, 0x7FFF_FFFF, 0x7F80_0000, 1 << 31)
            } else {
                let mut s = FloatSpecials::default();
                for x in p.iter().flatten() {
                    s = s.or(FloatSpecials { nan: x.is_nan(), neg_zero: x == 0.0 && x.is_sign_negative(), pos_zero: x == 0.0 && x.is_sign_positive() });
                }
                s
            }
        }
        _ => FloatSpecials::default(),
    }
}

/// [`FloatSpecials`] of column `c` over every batch. Above 2,000,000 rows the batches are split
/// among up to 8 threads (about 2,000,000 rows each).
pub(crate) fn float_specials(input: &[&RecordBatch], c: usize) -> FloatSpecials {
    let rows: usize = input.iter().map(|b| b.num_rows()).sum();
    let threads = (rows / 2_000_000).clamp(1, 8);
    // Pieces of at most rows / threads rows (a large batch is sliced), dealt out in order.
    let per = rows.div_ceil(threads).max(1);
    let mut pieces: Vec<ArrayRef> = Vec::new();
    for b in input {
        let a = b.column(c);
        let mut off = 0;
        while off < a.len() {
            let n = per.min(a.len() - off);
            pieces.push(a.slice(off, n));
            off += n;
        }
    }
    if threads == 1 {
        return pieces.iter().fold(FloatSpecials::default(), |s, a| s.or(specials_of(a.as_ref())));
    }
    let chunk = pieces.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        let hs: Vec<_> = pieces
            .chunks(chunk)
            .map(|ps| s.spawn(move || ps.iter().fold(FloatSpecials::default(), |s, a| s.or(specials_of(a.as_ref())))))
            .collect();
        hs.into_iter().fold(FloatSpecials::default(), |s, h| s.or(h.join().unwrap_or(FloatSpecials { nan: true, ..Default::default() })))
    })
}

/// A JSON string body: quotes, backslashes and control characters escaped.
fn json_escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

fn scan(source: &str) -> String {
    format!(r#"{{"op":"scan","source":"{source}"}}"#)
}

fn select_all(n: usize, input: String) -> String {
    let exprs: Vec<String> = (0..n).map(|i| format!("[\"c{i}\",\"(col \\\"c{i}\\\")\"]")).collect();
    format!(r#"{{"op":"select","exprs":[{}],"input":{input}}}"#, exprs.join(","))
}

/// Where one [`run`] spent its time: importing the input into Metal memory, the plan runner, and
/// exporting the result. Read with [`take_times`] on the thread that called `run`.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct GpuTimes {
    pub import: std::time::Duration,
    pub kernel: std::time::Duration,
    pub export: std::time::Duration,
}

thread_local! {
    static TIMES: std::cell::Cell<GpuTimes> = const { std::cell::Cell::new(GpuTimes {
        import: std::time::Duration::ZERO,
        kernel: std::time::Duration::ZERO,
        export: std::time::Duration::ZERO,
    }) };
}

/// The split of the last [`run`] on this thread, and resets it.
pub(crate) fn take_times() -> GpuTimes {
    TIMES.with(|t| t.replace(GpuTimes::default()))
}

/// One plan source: a name, its batches, and the columns to import (input column `i` is named
/// `{prefix}{i}` in the plan).
struct Src<'a> {
    name: &'a str,
    batches: &'a [&'a RecordBatch],
    prefix: &'a str,
    columns: Vec<usize>,
}

impl<'a> Src<'a> {
    fn new(name: &'a str, batches: &'a [&'a RecordBatch], prefix: &'a str, columns: &[usize]) -> Self {
        Self { name, batches, prefix, columns: columns.to_vec() }
    }
}

/// An imported column handed from its import thread to the thread that runs the plan.
struct Imported(arrowmetal::Array);

// SAFETY: an `am_array` handle is a reference-counted object of the ArrowMetal library, not tied
// to the thread that made it: the library itself imports batch i + 1 on a reader thread while the
// GPU runs batch i (the streaming executor). What is thread-local in the C ABI is the error slot,
// read by `from_arrow_chunks` on the import thread before it returns, and command-buffer batching,
// which an import does not use. Each handle moves once, after its import returned, to the one
// thread that uses it from then on; no handle is used from two threads at the same time.
unsafe impl Send for Imported {}

/// Input rows from which the columns of one call are imported at the same time.
pub(crate) const CONCURRENT_IMPORT_ROWS: usize = 1_000_000;

/// Imports `(name, chunks)` columns with `Array::from_arrow_chunks` (`am_import_chunks`): the
/// chunks are copied straight into Metal buffers, with no `concat_batches` copy in between. The
/// import's own thread policy caps the copy threads per column; from
/// [`CONCURRENT_IMPORT_ROWS`] rows the columns are imported at the same time, one thread each.
fn import(jobs: Vec<(String, Vec<&dyn arrow::array::Array>)>, rows: usize) -> Result<Vec<(String, arrowmetal::Array)>, String> {
    let one = |chunks: &[&dyn arrow::array::Array]| arrowmetal::Array::from_arrow_chunks(chunks).map_err(|x| x.message().to_string());
    if jobs.len() <= 1 || rows < CONCURRENT_IMPORT_ROWS {
        return jobs.into_iter().map(|(n, c)| one(&c).map(|a| (n, a))).collect();
    }
    std::thread::scope(|s| {
        let hs: Vec<_> = jobs
            .into_iter()
            .map(|(n, c)| s.spawn(move || one(&c).map(|a| (n, Imported(a)))))
            .collect();
        hs.into_iter()
            .map(|h| match h.join() {
                Ok(r) => r.map(|(n, a)| (n, a.0)),
                Err(_) => Err("panic in a column import".to_string()),
            })
            .collect()
    })
}

/// Imports the sources' columns, runs `plan`, exports every output column.
///
/// A single batch is imported as it is (copy-free when its buffers allow).
fn run_plan(plan: &str, sources: &[Src]) -> Result<Vec<ArrayRef>, String> {
    let e = |x: arrowmetal::Error| x.message().to_string();
    let rows: usize = sources.iter().map(|s| s.batches.iter().map(|b| b.num_rows()).sum::<usize>()).sum();
    let t0 = std::time::Instant::now();
    let mut jobs = Vec::new();
    for s in sources {
        for &i in &s.columns {
            let chunks: Vec<&dyn arrow::array::Array> = s.batches.iter().map(|b| b.column(i).as_ref()).collect();
            jobs.push((format!("{}{i}", s.prefix), chunks));
        }
    }
    let mut imported = import(jobs, rows)?.into_iter();
    let mut srcs = Vec::with_capacity(sources.len());
    for s in sources {
        let cols: Vec<(String, arrowmetal::Array)> = imported.by_ref().take(s.columns.len()).collect();
        srcs.push(arrowmetal::Source::new(s.name, cols).map_err(e)?);
    }
    let t1 = std::time::Instant::now();
    let refs: Vec<&arrowmetal::Source> = srcs.iter().collect();
    let out = arrowmetal::run_plan(plan, &refs, true).map_err(|x| format!("{} [rows {rows}; plan {plan}]", x.message()))?;
    let t2 = std::time::Instant::now();
    let cols = (0..out.column_count())
        .map(|i| out.column(i).and_then(|a| a.to_arrow()).map_err(e))
        .collect();
    let t3 = std::time::Instant::now();
    TIMES.with(|t| {
        let mut v = t.get();
        v.import += t1 - t0;
        v.kernel += t2 - t1;
        v.export += t3 - t2;
        t.set(v);
    });
    cols
}

/// Casts each column to the type DataFusion's schema says (a no-op when it already matches).
fn assemble(cols: Vec<ArrayRef>, schema: &SchemaRef) -> Result<RecordBatch, String> {
    if cols.len() != schema.fields().len() {
        return Err(format!("plan returned {} columns, schema has {}", cols.len(), schema.fields().len()));
    }
    let cols = cols
        .into_iter()
        .zip(schema.fields())
        .map(|(c, f)| {
            if c.data_type() == f.data_type() {
                Ok(c)
            } else {
                cast(&c, f.data_type()).map_err(|e| format!("cast {} -> {}: {e}", c.data_type(), f.data_type()))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(schema.clone(), cols).map_err(|e| e.to_string())
}

fn aggregate(
    keys: &[usize],
    aggs: &[crate::exec::AggSpec],
    input: &[&RecordBatch],
    out_schema: &SchemaRef,
) -> Result<RecordBatch, String> {
    let schema = input[0].schema();
    for &k in keys {
        if matches!(schema.field(k).data_type(), DataType::Float32 | DataType::Float64) && has_distinct_nans(input, k) {
            return Err(format!(
                "{DATA_DEPENDENT}: float group key {} holds NaNs of more than one bit pattern \
                 (DataFusion groups each pattern apart, ArrowMetal groups every NaN together)",
                schema.field(k).name()
            ));
        }
    }
    let mut key_json = Vec::new();
    for (j, k) in keys.iter().enumerate() {
        key_json.push(format!("[\"k{j}\",\"(col \\\"c{k}\\\")\"]"));
    }
    // A float min/max column's special values, scanned once per column.
    let mut specials: Vec<(usize, FloatSpecials)> = Vec::new();
    let mut agg_json = Vec::new();
    let mut helper_json = Vec::new();
    let mut fixes: Vec<MinMaxFix> = Vec::new();
    let mut next = keys.len() + aggs.len();
    for (j, a) in aggs.iter().enumerate() {
        let c = a.column.map(|c| format!("(col \\\"c{c}\\\")"));
        let row = match a.kind {
            AggKind::CountAll => format!("[\"count\",\"a{j}\"]"),
            k => {
                let op = match k {
                    AggKind::Sum => "sum",
                    AggKind::Min => "min",
                    AggKind::Max => "max",
                    AggKind::Mean => "mean",
                    // The engine's `count(x)` counts the non-null values of every type on every
                    // path (core a387a2a).
                    AggKind::Count => "count",
                    AggKind::CountAll => "count",
                };
                let Some(c) = &c else { return Err(format!("{op} without an argument column")) };
                format!("[\"{op}\",\"a{j}\",\"{c}\"]")
            }
        };
        agg_json.push(row);
        let fix = match (a.kind, a.float, a.column) {
            (AggKind::Min | AggKind::Max, true, Some(col)) => {
                let s = match specials.iter().find(|(x, _)| *x == col) {
                    Some((_, s)) => *s,
                    None => {
                        let s = float_specials(input, col);
                        specials.push((col, s));
                        s
                    }
                };
                if s.nan || (s.neg_zero && s.pos_zero) {
                    // The column holds a NaN or both zero signs: per-group helper counts decide
                    // where DataFusion's answer depends on row order (fix_float_min_max).
                    // Helpers go after every main aggregate, so the first keys+aggs outputs line up.
                    let Some(c) = c else { return Err("min/max without an argument column".into()) };
                    let lit = if schema.field(col).data_type() == &DataType::Float32 { "f32" } else { "f64" };
                    helper_json.push(format!("[\"sum\",\"nan{j}\",\"(if_else (ne {c} {c}) (i64 1) (i64 0))\"]"));
                    helper_json.push(format!("[\"count\",\"cnt{j}\",\"{c}\"]"));
                    helper_json.push(format!("[\"sum\",\"z{j}\",\"(if_else (eq {c} ({lit} 0)) (i64 1) (i64 0))\"]"));
                    helper_json.push(format!(
                        "[\"sum\",\"nz{j}\",\"(if_else (and (eq {c} ({lit} 0)) (lt (div ({lit} 1) {c}) ({lit} 0))) (i64 1) (i64 0))\"]"
                    ));
                    let h = [next, next + 1, next + 2, next + 3];
                    next += 4;
                    MinMaxFix::Helpers(h)
                } else {
                    // No NaN and at most one zero sign in the column: the plan runner's own
                    // min/max is DataFusion's answer up to the sign of a zero and the infinities.
                    MinMaxFix::Direct(s)
                }
            }
            _ => MinMaxFix::None,
        };
        fixes.push(fix);
    }
    // A GROUP BY with no aggregates (DISTINCT, or the inner half of count(DISTINCT x)) is sent as
    // it is: the plan runner returns the distinct keys.
    agg_json.extend(helper_json);
    let plan = format!(
        r#"{{"op":"group_by","keys":[{}],"aggs":[{}],"input":{}}}"#,
        key_json.join(","),
        agg_json.join(","),
        scan("t")
    );
    let mut needed: Vec<usize> = keys.to_vec();
    needed.extend(aggs.iter().filter_map(|a| a.column));
    needed.sort_unstable();
    needed.dedup();
    let cols = run_plan(&plan, &[Src::new("t", input, "c", &needed)])?;
    if cols.len() != next {
        return Err(format!("group_by returned {} columns, expected {next}", cols.len()));
    }
    let mut out: Vec<ArrayRef> = cols[..keys.len() + aggs.len()].to_vec();
    // DataFusion folds -0.0 into +0.0 in a group key (`normalize_float_zero`); ArrowMetal groups
    // them together too but reports the first-seen zero as the key.
    for (j, &k) in keys.iter().enumerate() {
        match schema.field(k).data_type() {
            DataType::Float64 => {
                let a = cast(&out[j], &DataType::Float64).map_err(|e| e.to_string())?;
                let p = a.as_primitive::<Float64Type>();
                let fixed: Float64Array = p.iter().map(|v| v.map(|x| if x == 0.0 { 0.0 } else { x })).collect();
                out[j] = Arc::new(fixed);
            }
            DataType::Float32 => {
                let a = cast(&out[j], &DataType::Float32).map_err(|e| e.to_string())?;
                let p = a.as_primitive::<Float32Type>();
                let fixed: Float32Array = p.iter().map(|v| v.map(|x| if x == 0.0 { 0.0 } else { x })).collect();
                out[j] = Arc::new(fixed);
            }
            _ => {}
        }
    }
    for (j, fix) in fixes.iter().enumerate() {
        let pos = keys.len() + j;
        let f32 = aggs[j].column.is_some_and(|c| schema.field(c).data_type() == &DataType::Float32);
        match *fix {
            MinMaxFix::None => {}
            MinMaxFix::Helpers([nan, cnt, z, nz]) => {
                out[pos] = fix_float_min_max(aggs[j].kind, f32, &out[pos], &cols[nan], &cols[cnt], &cols[z], &cols[nz])?;
            }
            MinMaxFix::Direct(s) => {
                out[pos] = direct_float_min_max(aggs[j].kind, f32, &out[pos], s)?;
            }
        }
    }
    assemble(out, out_schema)
}

/// How a grouped float `min`/`max` gets DataFusion's answer from the plan runner's.
#[derive(Debug, Clone, Copy)]
enum MinMaxFix {
    /// Not a float min/max.
    None,
    /// The column holds no NaN and at most one zero sign: [`direct_float_min_max`].
    Direct(FloatSpecials),
    /// Output positions of the helper counts (NaN, valid, zero, negative zero):
    /// [`fix_float_min_max`].
    Helpers([usize; 4]),
}

/// Prefix of a `run` error that is not an ArrowMetal failure but data on which the GPU path cannot
/// give DataFusion's answer (DataFusion's answer depends on row order, or the data holds a value
/// the GPU path orders differently), so only DataFusion can give it.
pub(crate) const DATA_DEPENDENT: &str = "data-dependent";

/// DataFusion's grouped float MIN/MAX (functions-aggregate min_max.rs, `PrimitiveGroupsAccumulator`)
/// starts every group at the type's largest finite value (MIN) or lowest finite value (MAX) and
/// replaces the running value when the new one compares lower (higher) or does not compare. So:
///
/// * a group whose MAX is -inf reads `f64::MIN` (`f32::MIN`), one whose MIN is +inf reads
///   `f64::MAX` (`f32::MAX`): -inf never compares higher than the starting value;
/// * with a NaN in a group, the answer is whatever follows the last NaN in arrival order;
/// * -0.0 and +0.0 compare equal, so the first zero seen is kept.
///
/// The last two are not functions of the group's values. The plan runner's min/max skip NaN and
/// return the extreme otherwise.
fn datafusion_infinity(kind: AggKind, f32: bool, v: f64) -> f64 {
    match kind {
        AggKind::Max if v == f64::NEG_INFINITY => {
            if f32 {
                f32::MIN as f64
            } else {
                f64::MIN
            }
        }
        AggKind::Min if v == f64::INFINITY => {
            if f32 {
                f32::MAX as f64
            } else {
                f64::MAX
            }
        }
        _ => v,
    }
}

/// The plan runner's min/max of a column with no NaN and at most one zero sign, as DataFusion
/// gives it (see [`datafusion_infinity`]): a zero result takes the column's one zero sign.
fn direct_float_min_max(kind: AggKind, f32: bool, got: &ArrayRef, s: FloatSpecials) -> Result<ArrayRef, String> {
    let got = cast(got, &DataType::Float64).map_err(|e| e.to_string())?;
    let got = got.as_primitive::<Float64Type>();
    let out: Float64Array = got
        .iter()
        .map(|v| {
            v.map(|x| {
                if x == 0.0 {
                    if s.neg_zero {
                        -0.0
                    } else {
                        0.0
                    }
                } else {
                    datafusion_infinity(kind, f32, x)
                }
            })
        })
        .collect();
    Ok(Arc::new(out))
}

/// The plan runner's min/max of a column holding a NaN or both zero signs, from the per-group
/// helper counts: a group with a NaN, or with both zero signs where the answer is zero, sends the
/// whole node back to DataFusion (its answer there depends on row order); everything else is the
/// plan runner's answer with the sign of a zero result set from the group's zero counts and
/// DataFusion's infinities (see [`datafusion_infinity`]).
fn fix_float_min_max(
    kind: AggKind,
    f32: bool,
    got: &ArrayRef,
    nan: &ArrayRef,
    cnt: &ArrayRef,
    z: &ArrayRef,
    nz: &ArrayRef,
) -> Result<ArrayRef, String> {
    let as_i64 = |a: &ArrayRef| -> Result<Int64Array, String> {
        Ok(cast(a, &DataType::Int64).map_err(|e| e.to_string())?.as_primitive::<Int64Type>().clone())
    };
    let got = cast(got, &DataType::Float64).map_err(|e| e.to_string())?;
    let got = got.as_primitive::<Float64Type>();
    let (nan, cnt, z, nz) = (as_i64(nan)?, as_i64(cnt)?, as_i64(z)?, as_i64(nz)?);
    let n = |a: &Int64Array, i: usize| if a.is_null(i) { 0 } else { a.value(i) };
    let mut out = Vec::with_capacity(got.len());
    for i in 0..got.len() {
        if n(&cnt, i) == 0 {
            out.push(None);
            continue;
        }
        if n(&nan, i) > 0 {
            return Err(format!("{DATA_DEPENDENT}: {kind:?} over a float group containing NaN"));
        }
        let v = got.value(i);
        if v == 0.0 {
            let (zeros, neg) = (n(&z, i), n(&nz, i));
            if neg > 0 && zeros > neg {
                return Err(format!("{DATA_DEPENDENT}: {kind:?} over a float group holding both -0.0 and +0.0"));
            }
            out.push(Some(if neg > 0 { -0.0 } else { 0.0 }));
        } else {
            out.push(Some(datafusion_infinity(kind, f32, v)));
        }
    }
    Ok(Arc::new(Float64Array::from(out)))
}

/// A join's parts, borrowed from its [`MetalOp::Join`].
struct JoinSpec<'a> {
    how: JoinHow,
    left_keys: &'a [usize],
    right_keys: &'a [usize],
    projection: &'a [usize],
    left_columns: usize,
}

/// DataFusion's `HashJoinExec` (inner, left, right; `NullEqualsNothing`) on the plan runner's join.
///
/// The plan runner builds its hash table from its right input and keeps its left input's rows
/// (`how: "left"`), so: an inner join probes with DataFusion's right (probe) input against its
/// left (build) input; a left join keeps DataFusion's left input; a right join keeps its right
/// input. A null key matches nothing on either side, as in DataFusion. Columns are named `l{i}` and
/// `r{i}` after the DataFusion input they come from, and a `select` puts the output in
/// DataFusion's order (the left input's columns, then the right's, through the projection).
fn join(spec: &JoinSpec, l: &[&RecordBatch], r: &[&RecordBatch], out_schema: &SchemaRef) -> Result<RecordBatch, String> {
    let nl = spec.left_columns;
    let mut lcols: Vec<usize> = spec.left_keys.to_vec();
    let mut rcols: Vec<usize> = spec.right_keys.to_vec();
    for &p in spec.projection {
        if p < nl {
            lcols.push(p);
        } else {
            rcols.push(p - nl);
        }
    }
    lcols.sort_unstable();
    lcols.dedup();
    rcols.sort_unstable();
    rcols.dedup();
    if l.is_empty() || r.is_empty() {
        return join_with_an_empty_side(spec, l, r, out_schema);
    }
    let name = |p: usize| if p < nl { format!("l{p}") } else { format!("r{}", p - nl) };
    let lk: Vec<String> = spec.left_keys.iter().map(|k| format!("\"l{k}\"")).collect();
    let rk: Vec<String> = spec.right_keys.iter().map(|k| format!("\"r{k}\"")).collect();
    // (plan-left source, plan-right source, its keys, the other's keys, how)
    let (a, b, a_on, b_on, how) = match spec.how {
        JoinHow::Inner => ("r", "l", &rk, &lk, "inner"),
        JoinHow::Left => ("l", "r", &lk, &rk, "left"),
        JoinHow::Right => ("r", "l", &rk, &lk, "left"),
    };
    let node = format!(
        r#"{{"op":"join","left":{},"right":{},"left_on":[{}],"right_on":[{}],"how":"{how}"}}"#,
        scan(a),
        scan(b),
        a_on.join(","),
        b_on.join(",")
    );
    // An empty projection (count(*) over the join) still needs the row count: select one key.
    let outs: Vec<String> = if spec.projection.is_empty() {
        vec![format!("[\"o0\",\"(col \\\"{}\\\")\"]", name(spec.left_keys[0]))]
    } else {
        spec.projection.iter().enumerate().map(|(j, &p)| format!("[\"o{j}\",\"(col \\\"{}\\\")\"]", name(p))).collect()
    };
    let plan = format!(r#"{{"op":"select","exprs":[{}],"input":{node}}}"#, outs.join(","));
    let cols = run_plan(&plan, &[Src::new("l", l, "l", &lcols), Src::new("r", r, "r", &rcols)])?;
    if spec.projection.is_empty() {
        let n = cols.first().map_or(0, |c| c.len());
        return RecordBatch::try_new_with_options(out_schema.clone(), vec![], &RecordBatchOptions::new().with_row_count(Some(n)))
            .map_err(|e| e.to_string());
    }
    assemble(cols, out_schema)
}

/// A join with no rows on one side (or both): the kept side's rows with nulls for the other's
/// columns (left or right join), or nothing.
fn join_with_an_empty_side(spec: &JoinSpec, l: &[&RecordBatch], r: &[&RecordBatch], out_schema: &SchemaRef) -> Result<RecordBatch, String> {
    let nl = spec.left_columns;
    let (kept, kept_is_left) = match spec.how {
        JoinHow::Left if r.is_empty() && !l.is_empty() => (l, true),
        JoinHow::Right if l.is_empty() && !r.is_empty() => (r, false),
        _ => return Ok(RecordBatch::new_empty(out_schema.clone())),
    };
    let n: usize = kept.iter().map(|b| b.num_rows()).sum();
    let mut cols = Vec::with_capacity(spec.projection.len());
    for (j, &p) in spec.projection.iter().enumerate() {
        let from_kept = if kept_is_left { p < nl } else { p >= nl };
        let field = out_schema.field(j);
        if from_kept {
            let i = if kept_is_left { p } else { p - nl };
            let chunks: Vec<&dyn arrow::array::Array> = kept.iter().map(|b| b.column(i).as_ref()).collect();
            cols.push(concat(&chunks).map_err(|e| e.to_string())?);
        } else {
            cols.push(new_null_array(field.data_type(), n));
        }
    }
    if cols.is_empty() {
        return RecordBatch::try_new_with_options(out_schema.clone(), vec![], &RecordBatchOptions::new().with_row_count(Some(n)))
            .map_err(|e| e.to_string());
    }
    assemble(cols, out_schema)
}
