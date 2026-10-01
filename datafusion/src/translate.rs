//! DataFusion physical nodes -> the operation `MetalExec` runs, or the reason it cannot.
//!
//! Everything here is a pure check on plan metadata (types, expression shapes, modes); nothing
//! touches data or the GPU. A `Err(String)` is a reason to leave the node to DataFusion.

use std::sync::Arc;

use arrow::datatypes::{DataType, Schema};
use datafusion::common::{JoinType, NullEquality, ScalarValue};
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::expressions::{
    BinaryExpr, CastExpr, Column, InListExpr, IsNotNullExpr, IsNullExpr, Literal, NotExpr,
};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::aggregates::AggregateExec;
use datafusion::physical_plan::expressions::PhysicalSortExpr;
use datafusion::physical_plan::joins::HashJoinExec;
use datafusion::physical_plan::ExecutionPlan;

use crate::exec::{AggKind, AggSpec, JoinHow, MetalOp, SortKey};

pub(crate) type Why = String;

fn is_int(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
    )
}

pub(crate) fn is_float(t: &DataType) -> bool {
    matches!(t, DataType::Float32 | DataType::Float64)
}

/// Column types a `MetalExec` will carry through a sort or filter (imported, `take`n, exported).
/// A whitelist: only what the differential grid covers, so everything else fails closed.
fn carried(t: &DataType) -> bool {
    is_int(t) || is_float(t) || matches!(t, DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8 | DataType::Boolean)
}

fn column_of(e: &Arc<dyn PhysicalExpr>) -> Option<&Column> {
    e.downcast_ref::<Column>()
}

fn check_schema(schema: &Schema) -> Result<(), Why> {
    for f in schema.fields() {
        if !carried(f.data_type()) {
            return Err(format!("column {} has type {} (not in the carried-type whitelist)", f.name(), f.data_type()));
        }
    }
    Ok(())
}

// -------------------------------------------------------------------------------------------------
// Sort
// -------------------------------------------------------------------------------------------------

pub(crate) fn sort_op(
    exprs: &[PhysicalSortExpr],
    input_schema: &Schema,
    fetch: Option<usize>,
) -> Result<MetalOp, Why> {
    check_schema(input_schema)?;
    let mut keys = Vec::new();
    for s in exprs {
        let Some(c) = column_of(&s.expr) else {
            return Err(format!("sort key {} is an expression, not a column", s.expr));
        };
        let t = input_schema.field(c.index()).data_type();
        // Float keys sort by IEEE 754 totalOrder in the core (the plan key's `float_order`), the
        // order arrow-rs uses, so Float32 and Float64 both qualify.
        if !(is_int(t) || is_float(t) || matches!(t, DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8)) {
            return Err(format!("sort key {} has type {t}", c.name()));
        }
        keys.push(SortKey { column: c.index(), descending: s.options.descending, nulls_first: s.options.nulls_first });
    }
    if keys.is_empty() {
        return Err("sort with no keys".into());
    }
    Ok(MetalOp::Sort { keys, fetch })
}

// -------------------------------------------------------------------------------------------------
// Aggregate
// -------------------------------------------------------------------------------------------------

/// `agg` is the aggregate whose `group_expr` / `aggr_expr` are written against the raw input (a
/// `Single`, `SinglePartitioned` or `Partial` node).
pub(crate) fn aggregate_op(agg: &AggregateExec) -> Result<MetalOp, Why> {
    let input_schema = agg.input().schema();
    let g = agg.group_expr();
    if g.has_grouping_set() || !g.is_single() {
        return Err("GROUPING SETS / ROLLUP / CUBE".into());
    }
    if g.expr().is_empty() {
        return Err("no GROUP BY keys (a whole-table aggregate)".into());
    }
    if agg.limit_options().is_some() {
        return Err("aggregate carries a TopK limit".into());
    }
    let mut keys = Vec::new();
    for (e, _) in g.expr() {
        let Some(c) = column_of(e) else {
            return Err(format!("group key {e} is an expression, not a column"));
        };
        let t = input_schema.field(c.index()).data_type();
        if !(is_int(t) || is_float(t) || *t == DataType::Utf8) {
            return Err(format!("group key {} has type {t}", c.name()));
        }
        keys.push(c.index());
    }
    if agg.filter_expr().iter().any(|f| f.is_some()) {
        return Err("aggregate FILTER (WHERE ...) clause".into());
    }
    let mut aggs = Vec::new();
    for a in agg.aggr_expr() {
        let name = a.fun().name().to_ascii_lowercase();
        if a.is_distinct() {
            return Err(format!("{}: DISTINCT", a.name()));
        }
        if !a.order_bys().is_empty() {
            return Err(format!("{}: ORDER BY inside the aggregate", a.name()));
        }
        let args = a.expressions();
        if args.len() != 1 {
            return Err(format!("{}: {} arguments", a.name(), args.len()));
        }
        let arg = &args[0];
        let out_type = a.field().data_type().clone();
        let kind = match name.as_str() {
            "sum" => AggKind::Sum,
            "min" => AggKind::Min,
            "max" => AggKind::Max,
            "count" => AggKind::Count,
            "avg" | "mean" => AggKind::Mean,
            other => return Err(format!("aggregate function {other}")),
        };
        // count(*) arrives as count(Int64(1)): a non-null literal counts rows.
        if kind == AggKind::Count {
            if let Some(l) = arg.downcast_ref::<Literal>() {
                if l.value().is_null() {
                    return Err("count(NULL)".into());
                }
                aggs.push(AggSpec { kind: AggKind::CountAll, column: None, float: false, out_type });
                continue;
            }
        }
        // DataFusion's type coercion wraps the argument in a widening cast (sum(Int32) reads
        // CAST(c AS Int64), avg(Int64) reads CAST(c AS Float64)). A cast that cannot change a
        // value is looked through; ArrowMetal widens the same way inside the aggregate.
        let (col_expr, cast_to) = match arg.downcast_ref::<CastExpr>() {
            Some(cast) => (cast.expr(), Some(cast.cast_type().clone())),
            None => (arg, None),
        };
        let Some(c) = column_of(col_expr) else {
            return Err(format!("{}: argument {arg} is an expression, not a column", a.name()));
        };
        let t = input_schema.field(c.index()).data_type();
        if let Some(to) = &cast_to {
            if !widening(t, to, kind) {
                return Err(format!("{}: argument cast {t} -> {to}", a.name()));
            }
        }
        let ok = match kind {
            AggKind::Sum | AggKind::Mean => is_int(t) || is_float(t),
            // Float32 and Float64 get DataFusion's NaN, signed-zero and infinity semantics in
            // gpu.rs (`aggregate`).
            AggKind::Min | AggKind::Max => is_int(t) || is_float(t),
            AggKind::Count => carried(t),
            AggKind::CountAll => true,
        };
        if !ok {
            return Err(format!("{}: argument type {t}", a.name()));
        }
        aggs.push(AggSpec { kind, column: Some(c.index()), float: is_float(t), out_type });
    }
    Ok(MetalOp::Aggregate { keys, aggs })
}

/// A cast inside an aggregate argument that the ArrowMetal aggregate makes itself: integers to the
/// 64-bit integer of the same signedness for `sum`, anything numeric to Float64 for `avg`.
fn widening(from: &DataType, to: &DataType, kind: AggKind) -> bool {
    use DataType::*;
    match (kind, to) {
        (AggKind::Sum, Int64) => matches!(from, Int8 | Int16 | Int32 | Int64),
        (AggKind::Sum, UInt64) => matches!(from, UInt8 | UInt16 | UInt32 | UInt64),
        (AggKind::Sum, Float64) => matches!(from, Float32 | Float64),
        (AggKind::Mean, Float64) => is_int(from) || is_float(from),
        _ => from == to,
    }
}

/// True when two aggregates compute the same thing (a `Final` over the `Partial` feeding it).
pub(crate) fn same_aggregates(final_: &AggregateExec, partial: &AggregateExec) -> bool {
    let (f, p) = (final_.aggr_expr(), partial.aggr_expr());
    f.len() == p.len()
        && f.iter().zip(p).all(|(a, b)| a.name() == b.name() && a.fun().name() == b.fun().name())
        && final_.group_expr().expr().len() == partial.group_expr().expr().len()
}

// -------------------------------------------------------------------------------------------------
// Filter predicate -> ArrowMetal s-expression
// -------------------------------------------------------------------------------------------------

/// Translates a DataFusion predicate into the s-expression grammar of docs/EXPR.md, column `i`
/// named `c{i}`. Only shapes whose semantics match arrow-rs exactly are translated.
pub(crate) fn predicate_sexpr(
    e: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
) -> Result<String, Why> {
    let any = e.as_ref();
    if let Some(c) = any.downcast_ref::<Column>() {
        let t = schema.field(c.index()).data_type();
        if *t != DataType::Boolean {
            return Err(format!("bare non-boolean column {} in a predicate", c.name()));
        }
        return Ok(format!("(col \"c{}\")", c.index()));
    }
    if let Some(n) = any.downcast_ref::<IsNullExpr>() {
        return Ok(format!("(is_null {})", value_sexpr(n.arg(), schema)?.0));
    }
    if let Some(n) = any.downcast_ref::<IsNotNullExpr>() {
        return Ok(format!("(is_valid {})", value_sexpr(n.arg(), schema)?.0));
    }
    if let Some(n) = any.downcast_ref::<NotExpr>() {
        return Ok(format!("(not {})", predicate_sexpr(n.arg(), schema)?));
    }
    if let Some(l) = any.downcast_ref::<Literal>() {
        return match l.value() {
            ScalarValue::Boolean(Some(b)) => Ok(format!("(bool {b})")),
            v => Err(format!("literal {v} as a predicate")),
        };
    }
    if any.downcast_ref::<InListExpr>().is_some() {
        return Err("IN list".into());
    }
    if let Some(b) = any.downcast_ref::<BinaryExpr>() {
        let op = b.op();
        match op {
            // SQL AND / OR are three-valued; the Kleene forms are the ones that match.
            Operator::And | Operator::Or => {
                let l = predicate_sexpr(b.left(), schema)?;
                let r = predicate_sexpr(b.right(), schema)?;
                let f = if *op == Operator::And { "and_kleene" } else { "or_kleene" };
                return Ok(format!("({f} {l} {r})"));
            }
            Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq => return comparison_sexpr(b, schema),
            other => return Err(format!("operator {other} in a predicate")),
        }
    }
    Err(format!("predicate node {e}"))
}

/// A value-producing leaf: a column or a literal. Returns the text and its type.
fn value_sexpr(e: &Arc<dyn PhysicalExpr>, schema: &Schema) -> Result<(String, DataType), Why> {
    let any = e.as_ref();
    if let Some(c) = any.downcast_ref::<Column>() {
        let t = schema.field(c.index()).data_type().clone();
        if !carried(&t) {
            return Err(format!("column {} of type {t}", c.name()));
        }
        return Ok((format!("(col \"c{}\")", c.index()), t));
    }
    if let Some(l) = any.downcast_ref::<Literal>() {
        return literal_sexpr(l.value());
    }
    if any.downcast_ref::<CastExpr>().is_some() {
        return Err(format!("CAST in a predicate ({e})"));
    }
    Err(format!("value expression {e}"))
}

fn literal_sexpr(v: &ScalarValue) -> Result<(String, DataType), Why> {
    let t = v.data_type();
    let s = match v {
        ScalarValue::Int8(Some(x)) => format!("(i8 {x})"),
        ScalarValue::Int16(Some(x)) => format!("(i16 {x})"),
        ScalarValue::Int32(Some(x)) => format!("(i32 {x})"),
        ScalarValue::Int64(Some(x)) => format!("(i64 {x})"),
        ScalarValue::UInt8(Some(x)) => format!("(u8 {x})"),
        ScalarValue::UInt16(Some(x)) => format!("(u16 {x})"),
        ScalarValue::UInt32(Some(x)) => format!("(u32 {x})"),
        ScalarValue::UInt64(Some(x)) => format!("(u64 {x})"),
        ScalarValue::Float64(Some(x)) if x.is_finite() => format!("(f64 {x:e})"),
        ScalarValue::Float32(Some(x)) if x.is_finite() => format!("(f32 {x:e})"),
        ScalarValue::Float64(Some(x)) if x.is_infinite() => format!("(f64 {})", if *x > 0.0 { "inf" } else { "-inf" }),
        ScalarValue::Float32(Some(x)) if x.is_infinite() => format!("(f32 {})", if *x > 0.0 { "inf" } else { "-inf" }),
        ScalarValue::Utf8(Some(x)) => format!("(str \"{}\")", escape(x)),
        ScalarValue::Boolean(Some(b)) => format!("(bool {b})"),
        other => return Err(format!("literal {other}")),
    };
    Ok((s, t))
}

/// A string literal body in the s-expression grammar (its lexer reads `\n`, `\t`, and `\x` as x).
pub(crate) fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n").replace('\t', "\\t")
}

fn comparison_sexpr(b: &BinaryExpr, schema: &Schema) -> Result<String, Why> {
    let (l, lt) = value_sexpr(b.left(), schema)?;
    let (r, rt) = value_sexpr(b.right(), schema)?;
    if lt != rt {
        return Err(format!("comparison between {lt} and {rt} (no coercion done here)"));
    }
    let name = match b.op() {
        Operator::Eq => "eq",
        Operator::NotEq => "ne",
        Operator::Lt => "lt",
        Operator::LtEq => "le",
        Operator::Gt => "gt",
        Operator::GtEq => "ge",
        _ => unreachable!(),
    };
    if lt == DataType::Utf8 {
        // Only equality against a literal has a string kernel in the fused grammar.
        let lit_right = b.right().downcast_ref::<Literal>().is_some();
        let lit_left = b.left().downcast_ref::<Literal>().is_some();
        let (col, lit) = match (lit_left, lit_right) {
            (false, true) => (l, b.right()),
            (true, false) => (r, b.left()),
            _ => return Err("string comparison not between a column and a literal".into()),
        };
        let Some(ScalarValue::Utf8(Some(p))) = lit.downcast_ref::<Literal>().map(|l| l.value()) else {
            return Err("string literal is null".into());
        };
        let eq = format!("(str_eq {col} \"{}\")", escape(p));
        return match b.op() {
            Operator::Eq => Ok(eq),
            Operator::NotEq => Ok(format!("(not {eq})")),
            op => Err(format!("string comparison {op}")),
        };
    }
    if lt == DataType::Boolean {
        return Err("boolean comparison".into());
    }
    if is_float(&lt) {
        let lit_right = b.right().downcast_ref::<Literal>();
        let lit_left = b.left().downcast_ref::<Literal>();
        let (col, lit, flipped) = match (lit_left, lit_right) {
            (None, Some(v)) => (l, v, false),
            (Some(v), None) => (r, v, true),
            _ => return Err("float comparison not between a column and a literal".into()),
        };
        let v = match lit.value() {
            ScalarValue::Float64(Some(x)) => *x,
            ScalarValue::Float32(Some(x)) => *x as f64,
            _ => return Err("float literal is null".into()),
        };
        // Normalise to `col OP lit`.
        let op = match (b.op(), flipped) {
            (o, false) => *o,
            (Operator::Lt, true) => Operator::Gt,
            (Operator::LtEq, true) => Operator::GtEq,
            (Operator::Gt, true) => Operator::Lt,
            (Operator::GtEq, true) => Operator::LtEq,
            (o, true) => *o,
        };
        return total_order_comparison(op, &col, lit.value(), v);
    }
    Ok(format!("({name} {l} {r})"))
}

/// `col OP c` for a float column and a non-NaN literal `c` as DataFusion 55.1 compares: IEEE 754
/// totalOrder (-NaN < -inf < ... < +inf < +NaN) with -0.0 and +0.0 equal. One fused expression: the
/// plan runner's comparisons are IEEE (a NaN compares false, -0.0 == +0.0), and `signbit` /
/// `is_nan` (docs/EXPR.md, "totalOrder comparisons") place the NaNs. Null where the column is null,
/// so a filter drops those rows as DataFusion does.
fn total_order_comparison(op: Operator, col: &str, lit: &ScalarValue, v: f64) -> Result<String, Why> {
    if v.is_nan() {
        return Err("float comparison against a NaN literal (totalOrder tells NaN payloads apart)".into());
    }
    let c = match lit {
        ScalarValue::Float64(Some(x)) if x.is_infinite() => format!("(f64 {})", if *x > 0.0 { "inf" } else { "-inf" }),
        ScalarValue::Float32(Some(x)) if x.is_infinite() => format!("(f32 {})", if *x > 0.0 { "inf" } else { "-inf" }),
        // `{:e}` keeps the sign of a zero ("-0e0").
        ScalarValue::Float64(Some(x)) => format!("(f64 {x:e})"),
        ScalarValue::Float32(Some(x)) => format!("(f32 {x:e})"),
        other => return Err(format!("literal {other}")),
    };
    // DataFusion 55.1 compares with arrow-rs's totalOrder kernels after rewriting -0.0 to +0.0 on
    // both sides (`normalize_float_zero` in datum.rs: SQL's `-0.0 = +0.0`). So a zero literal
    // compares like any other non-NaN literal: the IEEE comparison already treats the zeros as
    // equal, and only the NaNs need placing (a NaN with the sign bit set below every value, one
    // without above). This is docs/EXPR.md's "finite non-zero, ±inf" row for every non-NaN `c`.
    let s = format!("(signbit {col})");
    let n = format!("(is_nan {col})");
    let cmp = |name: &str| format!("({name} {col} {c})");
    Ok(match op {
        Operator::Eq => cmp("eq"),
        Operator::NotEq => format!("(not {})", cmp("eq")),
        Operator::Lt => format!("(or {} (and {n} {s}))", cmp("lt")),
        Operator::LtEq => format!("(or {} (and {n} {s}))", cmp("le")),
        Operator::Gt => format!("(or {} (and {n} (not {s})))", cmp("gt")),
        Operator::GtEq => format!("(or {} (and {n} (not {s})))", cmp("ge")),
        o => return Err(format!("float comparison {o}")),
    })
}

// -------------------------------------------------------------------------------------------------
// Join
// -------------------------------------------------------------------------------------------------

/// Key types a join is taken on: the ones the differential grid joins on.
fn join_key(t: &DataType) -> bool {
    matches!(t, DataType::Int32 | DataType::Int64 | DataType::Utf8)
}

/// A `HashJoinExec` the plan runner's join gives DataFusion's answer for: inner, left or right,
/// equal keys only (no join filter), `NullEqualsNothing` (SQL's `=`), column keys of the same type
/// on both sides, and carried columns of the whitelisted types.
pub(crate) fn join_op(j: &HashJoinExec) -> Result<MetalOp, Why> {
    let how = match j.join_type() {
        JoinType::Inner => JoinHow::Inner,
        JoinType::Left => JoinHow::Left,
        JoinType::Right => JoinHow::Right,
        t => return Err(format!("{t} join")),
    };
    if j.filter().is_some() {
        return Err("join filter (a condition besides the equal keys)".into());
    }
    if j.null_equality() != NullEquality::NullEqualsNothing {
        return Err("nulls compare equal in the keys (IS NOT DISTINCT FROM)".into());
    }
    if j.null_aware {
        return Err("null-aware anti join".into());
    }
    if ExecutionPlan::fetch(j).is_some() {
        return Err("join with a fetch limit".into());
    }
    let (ls, rs) = (j.left().schema(), j.right().schema());
    let mut left_keys = Vec::new();
    let mut right_keys = Vec::new();
    for (l, r) in j.on() {
        let (Some(lc), Some(rc)) = (column_of(l), column_of(r)) else {
            return Err(format!("join key {l} = {r} is an expression, not a column"));
        };
        let (lt, rt) = (ls.field(lc.index()).data_type(), rs.field(rc.index()).data_type());
        if lt != rt {
            return Err(format!("join key types differ: {lt} and {rt}"));
        }
        if !join_key(lt) {
            return Err(format!("join key {} has type {lt}", lc.name()));
        }
        left_keys.push(lc.index());
        right_keys.push(rc.index());
    }
    if left_keys.is_empty() {
        return Err("join without equal keys".into());
    }
    let nl = ls.fields().len();
    let projection: Vec<usize> = match &j.projection {
        Some(p) => p.iter().copied().collect(),
        None => (0..nl + rs.fields().len()).collect(),
    };
    for &p in &projection {
        let f = if p < nl { ls.field(p) } else { rs.field(p - nl) };
        if !carried(f.data_type()) {
            return Err(format!("column {} has type {} (not in the carried-type whitelist)", f.name(), f.data_type()));
        }
    }
    Ok(MetalOp::Join { how, left_keys, right_keys, projection, left_columns: nl })
}

pub(crate) fn filter_op(
    predicate: &Arc<dyn PhysicalExpr>,
    input_schema: &Schema,
    projection: Option<Vec<usize>>,
) -> Result<MetalOp, Why> {
    check_schema(input_schema)?;

    let sexpr = predicate_sexpr(predicate, input_schema)?;
    Ok(MetalOp::Filter { predicate: sexpr, projection })
}
