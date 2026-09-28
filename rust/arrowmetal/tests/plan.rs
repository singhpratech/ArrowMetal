//! The JSON plan runner (`am_plan_run`), against arrow-rs kernels doing the same work by hand.

mod common;

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array as _, ArrayRef, Int64Array, Scalar as ArrowScalar};
use arrow::compute::kernels::cmp;
use arrow::compute::{filter, sort, SortOptions};
use arrowmetal::{explain_plan, run_plan, Array, Source};

use common::int64;

/// Builds a one-table source named `sales` with an `amount` column.
fn sales(amount: &Int64Array) -> Source {
    let a = Array::from_arrow(amount).unwrap();
    Source::new("sales", vec![("amount".to_string(), a)]).unwrap()
}

/// `filter` in a plan against `arrow::compute::filter` over the same predicate.
#[test]
fn plan_filter_matches_arrow() {
    let amount = int64(1_000_001, 4, 201);
    let src = sales(&amount);
    let plan = r#"{"op":"filter","predicate":"(gt (col \"amount\") (int 0))",
                   "input":{"op":"scan","source":"sales"}}"#;

    let out = run_plan(plan, &[&src], true).unwrap();
    assert_eq!(out.column_count(), 1);
    assert_eq!(out.column_name(0).unwrap(), "amount");

    let mask = cmp::gt(&amount, &ArrowScalar::new(&Int64Array::from(vec![0i64]))).unwrap();
    let want = filter(&amount, &mask).unwrap();
    assert_eq!(out.row_count(), want.len());

    let got = out.column(0).unwrap().to_arrow().unwrap();
    assert_eq!(&got, &want);
}

/// A whole filter + group_by + sort plan, against the same answer assembled from arrow-rs kernels
/// and a `HashMap` fold (arrow-rs has no hash aggregation of its own).
#[test]
fn plan_group_by_and_sort_matches_a_hand_built_answer() {
    let n = 250_001usize;
    let mut r = common::rng(211);
    let region: Int64Array =
        (0..n).map(|_| Some(rand::Rng::random_range(&mut r, 0i64..13))).collect();
    let amount = int64(n, 0, 212);

    let g_region = Array::from_arrow(&region).unwrap();
    let g_amount = Array::from_arrow(&amount).unwrap();
    let src = Source::new(
        "sales",
        vec![("region".to_string(), g_region), ("amount".to_string(), g_amount)],
    )
    .unwrap();

    let plan = r#"{"op":"sort","by":[["total",true]],"input":
                    {"op":"group_by","keys":[["region","(col \"region\")"]],
                     "aggs":[["sum","total","(col \"amount\")"]],"input":
                      {"op":"filter","predicate":"(gt (col \"amount\") (int 0))","input":
                        {"op":"scan","source":"sales"}}}}"#;

    let out = run_plan(plan, &[&src], true).unwrap();

    // Oracle: arrow's filter for the predicate, a HashMap for the grouping, a sort for the order.
    let mask = cmp::gt(&amount, &ArrowScalar::new(&Int64Array::from(vec![0i64]))).unwrap();
    let kept_amount = filter(&amount, &mask).unwrap();
    let kept_region = filter(&region, &mask).unwrap();
    let kept_amount = kept_amount.as_any().downcast_ref::<Int64Array>().unwrap();
    let kept_region = kept_region.as_any().downcast_ref::<Int64Array>().unwrap();
    let mut want: HashMap<i64, i64> = HashMap::new();
    for i in 0..kept_amount.len() {
        *want.entry(kept_region.value(i)).or_insert(0) += kept_amount.value(i);
    }
    assert_eq!(out.row_count(), want.len());

    // Find the total column by name rather than by position.
    let mut totals = None;
    let mut regions = None;
    for i in 0..out.column_count() {
        match out.column_name(i).unwrap().as_str() {
            "total" => totals = Some(out.column(i).unwrap().to_arrow().unwrap()),
            "region" => regions = Some(out.column(i).unwrap().to_arrow().unwrap()),
            _ => {}
        }
    }
    let totals: ArrayRef = totals.expect("the plan produced a `total` column");
    let regions: ArrayRef = regions.expect("the plan produced a `region` column");
    let totals_i = totals.as_any().downcast_ref::<Int64Array>().unwrap();
    let regions_i = regions.as_any().downcast_ref::<Int64Array>().unwrap();

    for i in 0..totals_i.len() {
        assert_eq!(totals_i.value(i), want[&regions_i.value(i)], "region {}", regions_i.value(i));
    }

    // The plan asked for descending order by total; arrow's own sort of the same values is the
    // oracle for the ordering.
    let want_sorted = sort(
        &(Arc::new(Int64Array::from(want.values().copied().collect::<Vec<i64>>())) as ArrayRef),
        Some(SortOptions { descending: true, nulls_first: false }),
    )
    .unwrap();
    assert_eq!(&totals, &want_sorted);
}

/// The optimizer must not change the answer. Same plan, `optimize` on and off.
#[test]
fn optimizing_does_not_change_the_answer() {
    let amount = int64(100_001, 3, 221);
    let src = sales(&amount);
    let plan = r#"{"op":"limit","count":10,"input":
                    {"op":"sort","by":[["amount",false]],"input":
                      {"op":"filter","predicate":"(gt (col \"amount\") (int 100))","input":
                        {"op":"scan","source":"sales"}}}}"#;

    let optimized = run_plan(plan, &[&src], true).unwrap().column(0).unwrap().to_arrow().unwrap();
    let plain = run_plan(plan, &[&src], false).unwrap().column(0).unwrap().to_arrow().unwrap();
    assert_eq!(&optimized, &plain);
    assert_eq!(optimized.len(), 10);

    // And it is arrow's answer: the ten smallest values above 100.
    let mask = cmp::gt(&amount, &ArrowScalar::new(&Int64Array::from(vec![100i64]))).unwrap();
    let kept = filter(&amount, &mask).unwrap();
    let sorted = sort(&kept, Some(SortOptions { descending: false, nulls_first: false })).unwrap();
    assert_eq!(&optimized, &sorted.slice(0, 10));
}

/// `am_plan_explain` must return the two plans as text, and must error rather than return an empty
/// string when the plan does not type-check.
#[test]
fn explain_returns_text_and_errors_on_a_bad_plan() {
    let amount = int64(1000, 0, 231);
    let src = sales(&amount);

    let good = r#"{"op":"scan","source":"sales"}"#;
    let text = explain_plan(good, &[&src], true).unwrap();
    assert!(!text.is_empty(), "explain returned an empty string");

    let bad = r#"{"op":"scan","source":"no_such_table"}"#;
    let err = explain_plan(bad, &[&src], true).unwrap_err();
    assert!(!err.message().is_empty());
    assert!(!err.message().contains("(no message)"), "empty am_last_error: {err}");
}

/// A plan that does not parse or does not type-check must come back as an `Err` carrying the ABI's
/// own message, never as a panic or a wrong answer.
#[test]
fn a_bad_plan_is_an_error() {
    let amount = int64(1000, 0, 241);
    let src = sales(&amount);

    for bad in [
        "not json at all",
        r#"{"op":"scan","source":"no_such_table"}"#,
        r#"{"op":"filter","predicate":"(gt (col \"no_such_column\") (int 0))",
            "input":{"op":"scan","source":"sales"}}"#,
    ] {
        let err = run_plan(bad, &[&src], true).unwrap_err();
        assert!(err.message().starts_with("am_plan_run:"), "{err}");
        assert!(!err.message().contains("(no message)"), "empty am_last_error for {bad}: {err}");
    }
}

#[test]
fn a_source_with_no_columns_is_an_error() {
    assert!(Source::new("empty", vec![]).is_err());
}

/// An empty input table must run the plan and produce zero rows rather than fail.
#[test]
fn a_plan_over_an_empty_table_produces_no_rows() {
    let amount = Int64Array::from(Vec::<i64>::new());
    let src = sales(&amount);
    let plan = r#"{"op":"filter","predicate":"(gt (col \"amount\") (int 0))",
                   "input":{"op":"scan","source":"sales"}}"#;
    let out = run_plan(plan, &[&src], true).unwrap();
    assert_eq!(out.row_count(), 0);
    assert_eq!(out.column(0).unwrap().to_arrow().unwrap().len(), 0);
}

fn group_count_source(cols: Vec<(&str, ArrayRef)>) -> Source {
    Source::new(
        "t",
        cols.into_iter().map(|(n, a)| (n.to_string(), Array::from_arrow(a.as_ref()).unwrap())).collect(),
    )
    .unwrap()
}

fn group_by_k(aggs: &str) -> String {
    format!(r#"{{"op":"group_by","keys":[["k","(col \"k\")"]],"aggs":[{aggs}],"input":{{"op":"scan","source":"t"}}}}"#)
}

fn int64_column(out: &arrowmetal::PlanResult, i: usize) -> Vec<Option<i64>> {
    let a = out.column(i).unwrap().to_arrow().unwrap();
    a.as_any().downcast_ref::<Int64Array>().unwrap().iter().collect()
}

/// `count(v)` over Float64 next to a Float64 sum / min / max / mean (the per-aggregate group-by)
/// counts the non-null values; it used to fail with "cast to Float32 first".
#[test]
fn group_by_count_of_float64_next_to_a_float64_aggregate() {
    use arrow::array::{Float64Array, Int32Array};
    let k: ArrayRef = Arc::new(Int32Array::from(vec![0, 1, 0, 1]));
    let v: ArrayRef = Arc::new(Float64Array::from(vec![Some(1.0), Some(2.0), None, Some(4.0)]));
    let s = group_count_source(vec![("k", k), ("v", v)]);
    for other in ["sum", "min", "max", "mean"] {
        let aggs = format!(r#"["{other}","a","(col \"v\")"],["count","n","(col \"v\")"]"#);
        let out = run_plan(&group_by_k(&aggs), &[&s], true).unwrap();
        let keys = out.column(0).unwrap().to_arrow().unwrap();
        let keys = keys.as_any().downcast_ref::<Int32Array>().unwrap();
        let n = int64_column(&out, 2);
        assert_eq!(keys.len(), 2, "{other}");
        for (i, key) in keys.iter().enumerate() {
            let want = if key == Some(0) { 1 } else { 2 };
            assert_eq!(n[i], Some(want), "{other} key {key:?}");
        }
    }
}

/// `count(s)` over utf8 is the number of non-null strings: alone, and next to a Float64 sum, where
/// it used to return the row count (4) instead of 2.
#[test]
fn group_by_count_of_utf8_counts_non_null_strings() {
    use arrow::array::{Float64Array, StringArray};
    let k: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 1, 1, 1]));
    let s: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), None, Some("b"), None]));
    let v: ArrayRef = Arc::new(Float64Array::from(vec![1.0f64, 2.0, 3.0, 4.0]));
    let source = group_count_source(vec![("k", k), ("s", s), ("v", v)]);

    let alone = run_plan(&group_by_k(r#"["count","n","(col \"s\")"]"#), &[&source], true).unwrap();
    assert_eq!(int64_column(&alone, 1), vec![Some(2)]);

    let with_sum =
        run_plan(&group_by_k(r#"["count","n","(col \"s\")"],["sum","t","(col \"v\")"]"#), &[&source], true)
            .unwrap();
    assert_eq!(int64_column(&with_sum, 1), vec![Some(2)]);
    let t = with_sum.column(2).unwrap().to_arrow().unwrap();
    assert_eq!(t.as_any().downcast_ref::<Float64Array>().unwrap().value(0), 10.0);
}
