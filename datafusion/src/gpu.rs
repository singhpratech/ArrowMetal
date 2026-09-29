//! The GPU half of `MetalExec`: one `RecordBatch` in, one out, through ArrowMetal's plan runner.
//!
//! Synchronous, and every ArrowMetal handle lives and dies inside one call on one thread (the
//! handles are `!Send`; the ABI's error slot and batching are thread-local).

use std::sync::{Arc, Mutex};

use arrow::array::{Array as _, ArrayRef, AsArray, Float64Array, Int64Array};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Float64Type, Int64Type, SchemaRef};
use arrow::record_batch::RecordBatch;

use crate::exec::{AggKind, MetalOp};

/// One GPU job at a time in this process. The plan runner is thread-safe per handle set, but
/// serialising keeps concurrent `MetalExec`s (a join's two sides, parallel tests) from contending
/// for the one GPU queue; it costs nothing in a spike that runs one query at a time.
static GPU: Mutex<()> = Mutex::new(());

pub(crate) fn run(op: &MetalOp, input: &RecordBatch, out_schema: &SchemaRef) -> Result<RecordBatch, String> {
    let _guard = GPU.lock().unwrap_or_else(|p| p.into_inner());
    take_times();
    if input.num_rows() == 0 {
        // A sort, a filter and a grouped aggregate of nothing are all empty.
        return Ok(RecordBatch::new_empty(out_schema.clone()));
    }
    match op {
        MetalOp::Sort { keys, fetch } => {
            let mut flags = Vec::new();
            let mut by = Vec::new();
            for (i, k) in keys.iter().enumerate() {
                let c = format!("(col \\\"c{}\\\")", k.column);
                // ArrowMetal's sort puts nulls last in both directions. NULLS FIRST is a leading
                // is-null key sorted descending.
                if k.nulls_first {
                    flags.push(format!("[\"n{i}\",\"(if_else (is_null {c}) (i32 1) (i32 0))\"]"));
                    by.push(format!("[\"n{i}\",true]"));
                }
                // ArrowMetal keeps NaN after the values in both directions; arrow-rs orders NaN
                // above +inf, so a descending float key puts NaN first. A NaN key sorted
                // descending ahead of the value does that (null rows get a null flag: last).
                if k.float && k.descending {
                    flags.push(format!("[\"f{i}\",\"(if_else (ne {c} {c}) (i32 1) (i32 0))\"]"));
                    by.push(format!("[\"f{i}\",true]"));
                }
                by.push(format!("[\"c{}\",{}]", k.column, k.descending));
                // ArrowMetal's sort ties -0.0 with +0.0 (IEEE equality); arrow-rs's totalOrder puts
                // -0.0 first ascending. 1/x is -inf exactly for -0.0 among the zeros, so a key
                // "1/x < 0" (0 for -0.0, 1 otherwise) right after the value orders each zero tie.
                if k.float {
                    flags.push(format!("[\"z{i}\",\"(if_else (lt (div (f64 1) {c}) (f64 0)) (i32 0) (i32 1))\"]"));
                    by.push(format!("[\"z{i}\",{}]", k.descending));
                }
            }
            let mut node = scan();
            if !flags.is_empty() {
                node = format!(r#"{{"op":"with_columns","exprs":[{}],"input":{node}}}"#, flags.join(","));
            }
            node = format!(r#"{{"op":"sort","by":[{}],"input":{node}}}"#, by.join(","));
            if let Some(f) = fetch {
                node = format!(r#"{{"op":"limit","count":{f},"input":{node}}}"#);
            }
            node = select_all(input.num_columns(), node);
            let cols = run_plan(&node, input, None)?;
            assemble(cols, out_schema)
        }
        MetalOp::Filter { predicate, projection } => {
            let node = format!(
                r#"{{"op":"filter","predicate":"{}","input":{}}}"#,
                json_escape(predicate),
                scan()
            );
            let node = select_all(input.num_columns(), node);
            let cols = run_plan(&node, input, None)?;
            let cols = match projection {
                Some(p) => p.iter().map(|&i| cols[i].clone()).collect(),
                None => cols,
            };
            assemble(cols, out_schema)
        }
        MetalOp::Aggregate { keys, aggs } => aggregate(keys, aggs, input, out_schema),
    }
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
fn run_plan(plan: &str, input: &RecordBatch, only: Option<&[usize]>) -> Result<Vec<ArrayRef>, String> {
    let e = |x: arrowmetal::Error| x.message().to_string();
    let wanted: Vec<usize> = match only {
        Some(o) => o.to_vec(),
        None => (0..input.num_columns()).collect(),
    };
    let t0 = std::time::Instant::now();
    let mut cols = Vec::with_capacity(wanted.len());
    for &i in &wanted {
        cols.push((format!("c{i}"), arrowmetal::Array::from_arrow(input.column(i).as_ref()).map_err(e)?));
    }
    let src = arrowmetal::Source::new("t", cols).map_err(e)?;
    let t1 = std::time::Instant::now();
    let out = arrowmetal::run_plan(plan, &[&src], true)
        .map_err(|x| format!("{} [rows {}; plan {plan}]", x.message(), input.num_rows()))?;
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
    input: &RecordBatch,
    out_schema: &SchemaRef,
) -> Result<RecordBatch, String> {
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
            // count(x) as sum(is_valid(x) ? 1 : 0). The engine's own `count` fails or miscounts
            // when the group-by takes the per-aggregate path (see SPIKE.md, ArrowMetal defects):
            // over Float64 it throws "group-by over Float64 values", over other non-numeric types
            // it counts rows.
            AggKind::Count => {
                format!("[\"sum\",\"a{j}\",\"(if_else (is_valid {}) (i64 1) (i64 0))\"]", c.clone().unwrap())
            }
            k => {
                let op = match k {
                    AggKind::Sum => "sum",
                    AggKind::Min => "min",
                    AggKind::Max => "max",
                    AggKind::Mean => "mean",
                    AggKind::Count | AggKind::CountAll => unreachable!(),
                };
                format!("[\"{op}\",\"a{j}\",\"{}\"]", c.clone().unwrap())
            }
        };
        agg_json.push(row);
        if matches!(a.kind, AggKind::Min | AggKind::Max) && a.float {
            // Helpers go after every main aggregate, so the first keys+aggs outputs line up.
            let c = c.unwrap();
            helper_json.push(format!("[\"sum\",\"nan{j}\",\"(if_else (ne {c} {c}) (i64 1) (i64 0))\"]"));
            helper_json.push(format!("[\"sum\",\"cnt{j}\",\"(if_else (is_valid {c}) (i64 1) (i64 0))\"]"));
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
    // A GROUP BY with no aggregates (DISTINCT, or the inner half of count(DISTINCT x)): the plan
    // runner wants at least one, so count rows and drop the column afterwards.
    let dummy = aggs.is_empty();
    if dummy {
        agg_json.push("[\"count\",\"__distinct\"]".to_string());
        next += 1;
    }
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
        if input.schema().field(k).data_type() == &DataType::Float64 {
            let a = cast(&out[j], &DataType::Float64).map_err(|e| e.to_string())?;
            let p = a.as_primitive::<Float64Type>();
            let fixed: Float64Array =
                p.iter().map(|v| v.map(|x| if x == 0.0 { 0.0 } else { x })).collect();
            out[j] = Arc::new(fixed);
        }
    }
    for (j, h) in helpers.iter().enumerate() {
        let Some([nan, cnt, z, nz]) = *h else { continue };
        let pos = keys.len() + j;
        out[pos] = fix_float_min_max(aggs[j].kind, &out[pos], &cols[nan], &cols[cnt], &cols[z], &cols[nz])?;
    }
    assemble(out, out_schema)
}

/// Prefix of a `run` error that is not an ArrowMetal failure but data on which DataFusion's own
/// answer depends on row order, so only DataFusion can give it.
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
