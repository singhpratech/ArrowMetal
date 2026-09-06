//! ArrowMetal as a Polars expression plugin: GPU kernels usable *inside* a lazy plan.
//!
//! Every function here is a `#[polars_expr]`, which pyo3-polars turns into a C entry point Polars
//! `dlopen`s. Polars hands each one its inputs as `Series`; this crate moves them into Metal
//! memory over the Arrow C Data Interface (see `bridge.rs`), runs one ArrowMetal kernel, and moves
//! the answer back. Nothing here re-implements a kernel -- the GPU code all lives in the Swift
//! package and is reached through `arrowmetal-sys`.
//!
//! The Python side (`python/arrowmetal/polars_plugin.py`) registers these under the `arrowmetal`
//! expression namespace, so a user writes `pl.col("x").arrowmetal.sum()`.
//!
//! Thread safety: ArrowMetal serialises command-buffer commits behind its own lock
//! (`Sources/ArrowMetal/MetalContext.swift`) and keeps its error state thread-local, so Polars is
//! free to call these from several worker threads. No lock is taken here.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

mod bridge;
use bridge::{amerr, from_metal, scalar_bytes, scalar_series, to_metal};

// ---------------------------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------------------------

/// The dtype `am_reduce` answers a `sum` in: int64 for signed, uint64 for unsigned, float64 for
/// floats -- Arrow's own widening rule, and Polars' for `sum` over the narrow integer types.
fn sum_dtype(dt: &DataType) -> PolarsResult<DataType> {
    Ok(match dt {
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => DataType::Int64,
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            DataType::UInt64
        },
        DataType::Float32 | DataType::Float64 => DataType::Float64,
        other => polars_bail!(InvalidOperation: "arrowmetal: sum is not defined on {other:?}"),
    })
}

fn sum_output(fields: &[Field]) -> PolarsResult<Field> {
    let f = &fields[0];
    Ok(Field::new(f.name().clone(), sum_dtype(f.dtype())?))
}

/// `filter_sum(values, predicate)` sums its **first** input -- values first so that Polars, which
/// names a plugin's output after its first argument, names the column after the values.
fn filter_sum_output(fields: &[Field]) -> PolarsResult<Field> {
    let f = &fields[0];
    Ok(Field::new(f.name().clone(), sum_dtype(f.dtype())?))
}

fn same_output(fields: &[Field]) -> PolarsResult<Field> {
    Ok(fields[0].clone())
}

fn float64_output(fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(fields[0].name().clone(), DataType::Float64))
}

/// `group_by_sum(key, value)` returns one row per group as a struct, because a plugin function
/// answers with a single Series.
fn group_by_sum_output(fields: &[Field]) -> PolarsResult<Field> {
    let key = &fields[0];
    let value = &fields[1];
    Ok(Field::new(
        PlSmallStr::from_static("arrowmetal_group_by_sum"),
        DataType::Struct(vec![
            Field::new(key.name().clone(), key.dtype().clone()),
            Field::new(value.name().clone(), sum_dtype(value.dtype())?),
        ]),
    ))
}

// ---------------------------------------------------------------------------------------------
// Scalar aggregates
// ---------------------------------------------------------------------------------------------

/// `am_reduce` op codes.
const OP_SUM: i32 = 0;
const OP_MIN: i32 = 1;
const OP_MAX: i32 = 2;
const OP_MEAN: i32 = 3;

#[polars_expr(output_type_func=sum_output)]
fn arrowmetal_sum(inputs: &[Series]) -> PolarsResult<Series> {
    let a = to_metal(&inputs[0])?;
    let v = a.reduce(OP_SUM).map_err(amerr)?;
    scalar_series(inputs[0].name(), v, None)
}

#[polars_expr(output_type_func=same_output)]
fn arrowmetal_min(inputs: &[Series]) -> PolarsResult<Series> {
    let a = to_metal(&inputs[0])?;
    let v = a.reduce(OP_MIN).map_err(amerr)?;
    scalar_series(inputs[0].name(), v, Some(inputs[0].dtype()))
}

#[polars_expr(output_type_func=same_output)]
fn arrowmetal_max(inputs: &[Series]) -> PolarsResult<Series> {
    let a = to_metal(&inputs[0])?;
    let v = a.reduce(OP_MAX).map_err(amerr)?;
    scalar_series(inputs[0].name(), v, Some(inputs[0].dtype()))
}

#[polars_expr(output_type_func=float64_output)]
fn arrowmetal_mean(inputs: &[Series]) -> PolarsResult<Series> {
    let a = to_metal(&inputs[0])?;
    let v = a.reduce(OP_MEAN).map_err(amerr)?;
    scalar_series(inputs[0].name(), v, Some(&DataType::Float64))
}

/// `filter_sum(values, predicate)`: one GPU compaction plus one reduction, with no intermediate
/// Series crossing back into Polars. This is the shape that pays for itself -- a
/// `pl.col("v").filter(pl.col("k") == 2).sum()` in native Polars materialises the filtered column.
#[polars_expr(output_type_func=filter_sum_output)]
fn arrowmetal_filter_sum(inputs: &[Series]) -> PolarsResult<Series> {
    let values = &inputs[0];
    let mask = &inputs[1];
    polars_ensure!(
        matches!(mask.dtype(), DataType::Boolean),
        InvalidOperation: "arrowmetal: filter_sum wants a boolean predicate, got {:?}", mask.dtype()
    );
    polars_ensure!(
        mask.len() == values.len(),
        ShapeMismatch: "arrowmetal: filter_sum predicate has {} rows, values {}", mask.len(), values.len()
    );
    let m = to_metal(mask)?;
    let v = to_metal(values)?;
    let kept = v.filter(&m).map_err(amerr)?;
    let out = kept.reduce(OP_SUM).map_err(amerr)?;
    scalar_series(values.name(), out, None)
}

// ---------------------------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------------------------

#[derive(Deserialize)]
struct TopKKwargs {
    k: i64,
    largest: bool,
}

/// `top_k(k)`: the GPU radix sort's top-k indices, then a GPU take. Changes length, so the Python
/// side registers it with `changes_length=True`.
///
/// Ties and nulls follow ArrowMetal's sort order (stable, nulls last, NaN after +inf), which is
/// the order `am_argsort` uses -- not necessarily Polars' `top_k` tie order.
#[polars_expr(output_type_func=same_output)]
fn arrowmetal_top_k(inputs: &[Series], kwargs: TopKKwargs) -> PolarsResult<Series> {
    let s = &inputs[0];
    let k = kwargs.k.clamp(0, s.len() as i64);
    let a = to_metal(s)?;
    let idx = a.top_k(k, kwargs.largest).map_err(amerr)?;
    let picked = a.take(&idx).map_err(amerr)?;
    from_metal(&picked, s.name())
}

// ---------------------------------------------------------------------------------------------
// Element-wise
// ---------------------------------------------------------------------------------------------

/// A 64-bit hash of every value (`am_hash64`): the ArrowMetal extension defined in the header, so
/// Arrow-equal values hash equal and a null hashes to 0 and stays null.
#[polars_expr(output_type=UInt64)]
fn arrowmetal_hash64(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let a = to_metal(s)?;
    let h = if matches!(s.dtype(), DataType::String) {
        // Strings have their own kernel; kind 2 is the murmur3 hash, which is uint32.
        a.str_unary(2).map_err(amerr)?.cast("L").map_err(amerr)?
    } else {
        a.hash64().map_err(amerr)?
    };
    from_metal(&h, s.name())
}

#[derive(Deserialize)]
struct PatternKwargs {
    pattern: String,
}

/// `am_str_match` predicate codes.
const PRED_STARTS_WITH: i32 = 1;
const PRED_ENDS_WITH: i32 = 2;
const PRED_CONTAINS: i32 = 3;

fn str_match(inputs: &[Series], pred: i32, pattern: &str) -> PolarsResult<Series> {
    let s = &inputs[0];
    polars_ensure!(
        matches!(s.dtype(), DataType::String),
        InvalidOperation: "arrowmetal: string predicates want a String column, got {:?}", s.dtype()
    );
    let a = to_metal(s)?;
    let out = a.str_match(pred, pattern.as_bytes()).map_err(amerr)?;
    from_metal(&out, s.name())
}

#[polars_expr(output_type=Boolean)]
fn arrowmetal_contains(inputs: &[Series], kwargs: PatternKwargs) -> PolarsResult<Series> {
    str_match(inputs, PRED_CONTAINS, &kwargs.pattern)
}

#[polars_expr(output_type=Boolean)]
fn arrowmetal_starts_with(inputs: &[Series], kwargs: PatternKwargs) -> PolarsResult<Series> {
    str_match(inputs, PRED_STARTS_WITH, &kwargs.pattern)
}

#[polars_expr(output_type=Boolean)]
fn arrowmetal_ends_with(inputs: &[Series], kwargs: PatternKwargs) -> PolarsResult<Series> {
    str_match(inputs, PRED_ENDS_WITH, &kwargs.pattern)
}

/// `am_str_transform` op codes: 2 = utf8_upper, 3 = utf8_lower (simple 1:1 case mapping over
/// Basic Latin, Latin-1 Supplement and Latin Extended-A; see the header for what is out of scope).
const STR_UPPER: i32 = 2;
const STR_LOWER: i32 = 3;

fn str_case(inputs: &[Series], op: i32) -> PolarsResult<Series> {
    let s = &inputs[0];
    polars_ensure!(
        matches!(s.dtype(), DataType::String),
        InvalidOperation: "arrowmetal: upper/lower want a String column, got {:?}", s.dtype()
    );
    let a = to_metal(s)?;
    let out = a.str_transform(op, b"", b"", 0, 0).map_err(amerr)?;
    from_metal(&out, s.name())
}

#[polars_expr(output_type=String)]
fn arrowmetal_upper(inputs: &[Series]) -> PolarsResult<Series> {
    str_case(inputs, STR_UPPER)
}

#[polars_expr(output_type=String)]
fn arrowmetal_lower(inputs: &[Series]) -> PolarsResult<Series> {
    str_case(inputs, STR_LOWER)
}

#[derive(Deserialize)]
struct ArithKwargs {
    /// "add", "sub", "mul" or "div".
    op: String,
    value: f64,
}

/// Arithmetic against a scalar, on the GPU. The result keeps the column's own type, which is
/// Arrow's unchecked behaviour: integer arithmetic wraps and integer division by zero yields 0.
#[polars_expr(output_type_func=same_output)]
fn arrowmetal_arith_scalar(inputs: &[Series], kwargs: ArithKwargs) -> PolarsResult<Series> {
    let s = &inputs[0];
    let op = match kwargs.op.as_str() {
        "add" => 0,
        "sub" => 1,
        "mul" => 2,
        "div" => 3,
        other => polars_bail!(InvalidOperation: "arrowmetal: unknown arithmetic op {other:?}"),
    };
    let bytes = scalar_bytes(s.dtype(), kwargs.value)?;
    let a = to_metal(s)?;
    let out = unsafe { a.arith_scalar(op, bytes.as_ptr() as *const std::ffi::c_void) }
        .map_err(amerr)?;
    from_metal(&out, s.name())
}

// ---------------------------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------------------------

/// `group_by_sum(key, value)`: one GPU group-by over an arbitrary key column plus one segmented
/// sum, returned as a struct column of `n_groups` rows with the key and the sum.
///
/// A plugin expression answers with one Series, so a grouped result has to be a struct; unnest it
/// on the Python side. Group order is ArrowMetal's -- ascending by key for numeric, boolean,
/// temporal and decimal keys, first-seen for utf8 -- so sort both sides before comparing with
/// Polars' own `group_by`.
///
/// This is a plugin *expression*, not a Polars aggregation: Polars' plugin API has no hook for
/// contributing a hash aggregate to the group-by engine, so this runs as a projection over the
/// whole frame (`df.select(...)`) rather than inside `df.group_by(...).agg(...)`.
#[polars_expr(output_type_func=group_by_sum_output)]
fn arrowmetal_group_by_sum(inputs: &[Series]) -> PolarsResult<Series> {
    let keys = &inputs[0];
    let values = &inputs[1];
    polars_ensure!(
        keys.len() == values.len(),
        ShapeMismatch: "arrowmetal: group_by_sum key has {} rows, values {}", keys.len(), values.len()
    );
    let k = to_metal(keys)?;
    let v = to_metal(values)?;
    let gb = arrowmetal_sys::GroupBy::new(&[&k]).map_err(amerr)?;
    let key_out = gb.keys(0).map_err(amerr)?;
    // agg op 0 is hash_sum.
    let sum_out = gb.agg(Some(&v), 0, 0.0).map_err(amerr)?;

    let key_series = from_metal(&key_out, keys.name())?;
    let sum_series = from_metal(&sum_out, values.name())?;
    let n = key_series.len();
    let st = StructChunked::from_series(
        PlSmallStr::from_static("arrowmetal_group_by_sum"),
        n,
        [&key_series, &sum_series].into_iter(),
    )?;
    Ok(st.into_series())
}

// ---------------------------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------------------------

/// `am_version()` and `am_device_name()` as a one-row String column, so a lazy plan can prove it
/// really reached the GPU.
#[polars_expr(output_type=String)]
fn arrowmetal_device(_inputs: &[Series]) -> PolarsResult<Series> {
    let s = format!(
        "ArrowMetal {} on {}",
        arrowmetal_sys::version(),
        arrowmetal_sys::device_name()
    );
    Ok(Series::new(PlSmallStr::from_static("arrowmetal_device"), [s]))
}
