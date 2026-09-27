//! Minimal ArrowMetal-only reproductions of what the differential grid found, through the
//! `arrowmetal` crate and the plan runner (no DataFusion involved). Each test pins what ArrowMetal
//! does today, so a change on either side shows up here first. The crate works around every one of
//! them; SPIKE.md classifies each.

use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray, Float64Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{Float64Type, Int64Type};
use arrowmetal::{run_plan, Array, Source};

fn src(cols: Vec<(&str, ArrayRef)>) -> Source {
    Source::new(
        "t",
        cols.into_iter().map(|(n, a)| (n.to_string(), Array::from_arrow(a.as_ref()).unwrap())).collect(),
    )
    .unwrap()
}

fn f64_bits(a: &ArrayRef) -> Vec<u64> {
    let p = a.as_primitive::<Float64Type>();
    (0..p.len()).map(|i| p.value(i).to_bits()).collect()
}

fn group_by(aggs: &str) -> String {
    format!(r#"{{"op":"group_by","keys":[["k","(col \"k\")"]],"aggs":[{aggs}],"input":{{"op":"scan","source":"t"}}}}"#)
}

/// ArrowMetal defect. `count(v)` over a Float64 column errors whenever the group-by also has
/// another aggregate over it that the fused kernel cannot take (sum/min/max/mean over Float64):
/// the whole group-by then runs per aggregate (`Executor.aggregateOne`), whose `count` case sends
/// Float64 to `GroupBy.count(MetalArray<Double>)`, which throws at Kernels/GroupBy.swift:200.
/// `count(v)` alone and `sum(v)` alone both work.
#[test]
fn count_over_float64_next_to_a_float64_aggregate_errors() {
    let k: ArrayRef = Arc::new(Int32Array::from(vec![0, 1, 0, 1]));
    let v: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0]));
    let s = src(vec![("k", k), ("v", v)]);
    let c = r#"(col \"v\")"#;
    for alone in [format!(r#"["count","n","{c}"]"#), format!(r#"["sum","a","{c}"]"#)] {
        assert!(run_plan(&group_by(&alone), &[&s], true).is_ok(), "{alone}");
    }
    for other in ["sum", "min", "max", "mean"] {
        let aggs = format!(r#"["{other}","a","{c}"],["count","n","{c}"]"#);
        let err = run_plan(&group_by(&aggs), &[&s], true).unwrap_err();
        assert!(err.message().contains("group-by over Float64 values: cast to Float32 first"), "{other}: {err}");
    }
}

/// ArrowMetal defect. `count(s)` over a utf8 column with nulls: alone it is rejected; next to a
/// Float64 `sum` (per-aggregate path) it returns the row count, nulls included — a wrong answer,
/// not an error: `Executor.aggregateOne`'s `count` falls to `default: gb.count()` for any type it
/// does not list. SQL's count(s) below is 2.
#[test]
fn count_over_utf8_is_rejected_alone_and_counts_nulls_on_the_per_aggregate_path() {
    let k: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 1, 1, 1]));
    let s: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), None, Some("b"), None]));
    let v: ArrayRef = Arc::new(Float64Array::from(vec![1.0f64, 2.0, 3.0, 4.0]));
    let source = src(vec![("k", k), ("s", s), ("v", v)]);
    let err = run_plan(&group_by(r#"["count","n","(col \"s\")"]"#), &[&source], true).unwrap_err();
    assert!(err.message().contains("group_by count of a utf8 expression"), "{err}");

    let out = run_plan(&group_by(r#"["count","n","(col \"s\")"],["sum","t","(col \"v\")"]"#), &[&source], true)
        .unwrap();
    let n = out.column(1).unwrap().to_arrow().unwrap();
    assert_eq!(n.as_primitive::<Int64Type>().value(0), 4, "ArrowMetal counts the two null rows");

    // The workaround the crate uses: sum of a validity indicator.
    let out = run_plan(
        &group_by(r#"["sum","n","(if_else (is_valid (col \"s\")) (i64 1) (i64 0))"],["sum","t","(col \"v\")"]"#),
        &[&source],
        true,
    )
    .unwrap();
    assert_eq!(out.column(1).unwrap().to_arrow().unwrap().as_primitive::<Int64Type>().value(0), 2);
}

/// Semantics difference, not a defect. ArrowMetal's sort compares -0.0 and +0.0 as equal (IEEE),
/// so a stable sort keeps them in input order; arrow-rs (and DataFusion) sort by IEEE totalOrder,
/// -0.0 first. The crate adds a signed-zero key.
#[test]
fn sort_ties_signed_zeros() {
    let x: ArrayRef = Arc::new(Float64Array::from(vec![0.0f64, -0.0, 0.0, -0.0]));
    let s = src(vec![("x", x)]);
    let out = run_plan(r#"{"op":"sort","by":[["x",false]],"input":{"op":"scan","source":"t"}}"#, &[&s], true)
        .unwrap();
    let got = f64_bits(&out.column(0).unwrap().to_arrow().unwrap());
    assert_eq!(got, vec![0, 1 << 63, 0, 1 << 63]);
}

/// Semantics difference, not a defect. A Float64 group key: -0.0 and +0.0 form one group (as in
/// DataFusion 55), and the key reported is the first zero seen; DataFusion reports +0.0
/// (`normalize_float_zero`). The crate rewrites a zero key to +0.0.
#[test]
fn float_group_key_is_the_first_zero_seen() {
    for (input, want) in [
        (vec![0.0f64], vec![0u64]),
        (vec![-0.0], vec![1 << 63]),
        (vec![-0.0, 0.0, 2.5], vec![1 << 63, 2.5f64.to_bits()]),
    ] {
        let k: ArrayRef = Arc::new(Float64Array::from(input.clone()));
        let v: ArrayRef = Arc::new(Int64Array::from(vec![1i64; input.len()]));
        let s = src(vec![("k", k), ("v", v)]);
        let out = run_plan(&group_by(r#"["count","n"]"#), &[&s], true).unwrap();
        assert_eq!(f64_bits(&out.column(0).unwrap().to_arrow().unwrap()), want, "{input:?}");
    }
}

/// What the crate relies on for its signed-zero keys: 1 / -0.0 is -inf in the fused compiler.
#[test]
fn fused_division_keeps_the_sign_of_zero() {
    let x: ArrayRef = Arc::new(Float64Array::from(vec![0.0f64, -0.0, 2.0]));
    let s = src(vec![("x", x)]);
    let plan = r#"{"op":"select","exprs":[["r","(div (f64 1) (col \"x\"))"]],"input":{"op":"scan","source":"t"}}"#;
    let out = run_plan(plan, &[&s], true).unwrap();
    let got = out.column(0).unwrap().to_arrow().unwrap();
    let p = got.as_primitive::<Float64Type>();
    assert_eq!((p.value(0), p.value(1), p.value(2)), (f64::INFINITY, f64::NEG_INFINITY, 0.5));
}

/// The plan runner rejects a group_by with no aggregates (a DISTINCT); the crate adds a row count
/// and drops it.
#[test]
fn group_by_needs_an_aggregate() {
    let k: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 1, 2]));
    let s = src(vec![("k", k)]);
    let err = run_plan(&group_by(""), &[&s], true).unwrap_err();
    assert!(err.message().contains("at least one aggregation"), "{err}");
}
