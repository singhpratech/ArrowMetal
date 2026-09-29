//! The GPU half of `MetalExec`: the collected `RecordBatch`es in, one batch out, through
//! ArrowMetal's plan runner.
//!
//! Synchronous, and every ArrowMetal handle lives and dies inside one call on one thread (the
//! handles are `!Send`; the ABI's error slot and batching are thread-local).

use std::sync::{Arc, Mutex};

use arrow::array::{Array as _, ArrayRef, AsArray, Float32Array, Float64Array, Int64Array};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Float32Type, Float64Type, Int64Type, SchemaRef};
use arrow::record_batch::RecordBatch;

use crate::exec::{AggKind, MetalOp};

/// One GPU job at a time in this process. The plan runner is thread-safe per handle set, but
/// serialising keeps concurrent `MetalExec`s (a join's two sides, parallel tests) from contending
/// for the one GPU queue; it costs nothing in a spike that runs one query at a time.
static GPU: Mutex<()> = Mutex::new(());

pub(crate) fn run(op: &MetalOp, input: &[RecordBatch], out_schema: &SchemaRef) -> Result<RecordBatch, String> {
    let _guard = GPU.lock().unwrap_or_else(|p| p.into_inner());
    take_times();
    // The chunked import takes every non-empty batch as one chunk of each column.
    let input: Vec<&RecordBatch> = input.iter().filter(|b| b.num_rows() > 0).collect();
    if input.is_empty() {
        // A sort, a filter and a grouped aggregate of nothing are all empty.
        return Ok(RecordBatch::new_empty(out_schema.clone()));
    }
    let num_columns = input[0].num_columns();
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
            let mut node = format!(r#"{{"op":"sort","by":[{}],"input":{}}}"#, by.join(","), scan());
            if let Some(f) = fetch {
                node = format!(r#"{{"op":"limit","count":{f},"input":{node}}}"#);
            }
            node = select_all(num_columns, node);
            let cols = run_plan(&node, &input, None)?;
            assemble(cols, out_schema)
        }
        MetalOp::Filter { predicate, projection, float_compared } => {
            // The fused comparisons are IEEE; the rewrite in translate.rs matches arrow-rs's
            // totalOrder for every value except a NaN with the sign bit set, which totalOrder puts
            // below -inf (the expression grammar has no sign-bit test). A compared float column
            // holding one sends the node back to DataFusion.
            for &c in float_compared {
                if has_negative_nan(&input, c) {
                    return Err(format!(
                        "{DATA_DEPENDENT}: a compared float column holds a NaN with the sign bit set \
                         (totalOrder puts it below -inf, the GPU comparison cannot)"
                    ));
                }
            }
            let node = format!(
                r#"{{"op":"filter","predicate":"{}","input":{}}}"#,
                json_escape(predicate),
                scan()
            );
            let node = select_all(num_columns, node);
            let cols = run_plan(&node, &input, None)?;
            let cols = match projection {
                Some(p) => p.iter().map(|&i| cols[i].clone()).collect(),
                None => cols,
            };
            assemble(cols, out_schema)
        }
        MetalOp::Aggregate { keys, aggs } => aggregate(keys, aggs, &input, out_schema),
    }
}

/// True when column `c` of any batch holds a NaN with the sign bit set (null slots included: a
/// false positive only costs the fallback). Negative NaNs are exactly the bit patterns above -inf's.
fn has_negative_nan(input: &[&RecordBatch], c: usize) -> bool {
    input.iter().any(|b| {
        let a = b.column(c);
        match a.data_type() {
            DataType::Float64 => {
                a.as_primitive::<Float64Type>().values().iter().any(|v| v.to_bits() > 0xFFF0_0000_0000_0000)
            }
            DataType::Float32 => a.as_primitive::<Float32Type>().values().iter().any(|v| v.to_bits() > 0xFF80_0000),
            _ => false,
        }
    })
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

fn scan() -> String {
    r#"{"op":"scan","source":"t"}"#.to_string()
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

/// Imports the columns (all of them, or `only`), runs `plan`, exports every output column.
///
/// Each column is imported from its chunks, one per batch, with `Array::from_arrow_chunks`
/// (`am_import_chunks`): the chunks are copied straight into the Metal buffers, with no
/// `concat_batches` copy in between. A single batch is imported as it is (copy-free when its
/// buffers allow).
fn run_plan(plan: &str, input: &[&RecordBatch], only: Option<&[usize]>) -> Result<Vec<ArrayRef>, String> {
    let e = |x: arrowmetal::Error| x.message().to_string();
    let wanted: Vec<usize> = match only {
        Some(o) => o.to_vec(),
        None => (0..input[0].num_columns()).collect(),
    };
    let rows: usize = input.iter().map(|b| b.num_rows()).sum();
    let t0 = std::time::Instant::now();
    let mut cols = Vec::with_capacity(wanted.len());
    for &i in &wanted {
        let chunks: Vec<&dyn arrow::array::Array> = input.iter().map(|b| b.column(i).as_ref()).collect();
        cols.push((format!("c{i}"), arrowmetal::Array::from_arrow_chunks(&chunks).map_err(e)?));
    }
    let src = arrowmetal::Source::new("t", cols).map_err(e)?;
    let t1 = std::time::Instant::now();
    let out = arrowmetal::run_plan(plan, &[&src], true)
        .map_err(|x| format!("{} [rows {rows}; plan {plan}]", x.message()))?;
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
    let mut agg_json = Vec::new();
    let mut helper_json = Vec::new();
    // Output position of each helper aggregate for a Float64 min/max: (nan, valid, zero, negzero).
    let mut helpers: Vec<Option<[usize; 4]>> = Vec::new();
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
                    AggKind::CountAll => unreachable!(),
                };
                format!("[\"{op}\",\"a{j}\",\"{}\"]", c.clone().unwrap())
            }
        };
        agg_json.push(row);
        if matches!(a.kind, AggKind::Min | AggKind::Max) && a.float {
            // Helpers go after every main aggregate, so the first keys+aggs outputs line up.
            let c = c.unwrap();
            helper_json.push(format!("[\"sum\",\"nan{j}\",\"(if_else (ne {c} {c}) (i64 1) (i64 0))\"]"));
            helper_json.push(format!("[\"count\",\"cnt{j}\",\"{c}\"]"));
            helper_json.push(format!("[\"sum\",\"z{j}\",\"(if_else (eq {c} (f64 0)) (i64 1) (i64 0))\"]"));
            helper_json.push(format!(
                "[\"sum\",\"nz{j}\",\"(if_else (and (eq {c} (f64 0)) (lt (div (f64 1) {c}) (f64 0))) (i64 1) (i64 0))\"]"
            ));
            helpers.push(Some([next, next + 1, next + 2, next + 3]));
            next += 4;
        } else {
            helpers.push(None);
        }
    }
    // A GROUP BY with no aggregates (DISTINCT, or the inner half of count(DISTINCT x)) is sent as
    // it is: the plan runner returns the distinct keys.
    agg_json.extend(helper_json);
    let plan = format!(
        r#"{{"op":"group_by","keys":[{}],"aggs":[{}],"input":{}}}"#,
        key_json.join(","),
        agg_json.join(","),
        scan()
    );
    let mut needed: Vec<usize> = keys.to_vec();
    needed.extend(aggs.iter().filter_map(|a| a.column));
    needed.sort_unstable();
    needed.dedup();
    let cols = run_plan(&plan, input, Some(&needed))?;
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
    for (j, h) in helpers.iter().enumerate() {
        let Some([nan, cnt, z, nz]) = *h else { continue };
        let pos = keys.len() + j;
        out[pos] = fix_float_min_max(aggs[j].kind, &out[pos], &cols[nan], &cols[cnt], &cols[z], &cols[nz])?;
    }
    assemble(out, out_schema)
}

/// Prefix of a `run` error that is not an ArrowMetal failure but data on which the GPU path cannot
/// give DataFusion's answer (DataFusion's answer depends on row order, or the data holds a value
/// the GPU path orders differently), so only DataFusion can give it.
pub(crate) const DATA_DEPENDENT: &str = "data-dependent";

/// DataFusion's grouped Float64 MIN/MAX folds with `partial_cmp` and replaces the running value
/// whenever the comparison is `None` (functions-aggregate min_max.rs), so with a NaN in a group the
/// answer is whatever follows the last NaN in arrival order; and -0.0 vs +0.0 compare equal, so the
/// first one seen wins. Neither is a function of the group's values. ArrowMetal's min/max skip NaN
/// and are exact otherwise, so: a group with a NaN, or with both zero signs where the answer is
/// zero, sends the whole node back to DataFusion; everything else is ArrowMetal's answer with the
/// sign of a zero result set from the per-group zero counts.
fn fix_float_min_max(
    kind: AggKind,
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
            return Err(format!("{DATA_DEPENDENT}: {kind:?} over a Float64 group containing NaN"));
        }
        let v = got.value(i);
        if v == 0.0 {
            let (zeros, neg) = (n(&z, i), n(&nz, i));
            if neg > 0 && zeros > neg {
                return Err(format!("{DATA_DEPENDENT}: {kind:?} over a Float64 group holding both -0.0 and +0.0"));
            }
            out.push(Some(if neg > 0 { -0.0 } else { 0.0 }));
        } else {
            out.push(Some(v));
        }
    }
    Ok(Arc::new(Float64Array::from(out)))
}
