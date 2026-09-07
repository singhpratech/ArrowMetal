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
