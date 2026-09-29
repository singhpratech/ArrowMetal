//! Minimal ArrowMetal-only reproductions of what the differential grid found, through the
//! `arrowmetal` crate and the plan runner (no DataFusion involved). Each test pins what ArrowMetal
//! does today, so a change on either side shows up here first. The crate works around every one of
//! them; SPIKE.md classifies each.

use std::sync::Arc;

use arrow::array::{Array as _, ArrayRef, AsArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Float32Type, Float64Type, Int64Type};
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

fn as_i64(a: &ArrayRef) -> Vec<i64> {
    let a = arrow::compute::cast(a, &arrow::datatypes::DataType::Int64).unwrap();
    a.as_primitive::<Int64Type>().values().to_vec()
}

/// Fixed in the core (a387a2a). `count(v)` over a Float64 column next to another Float64 aggregate
/// (sum/min/max/mean, which send the group-by down the per-aggregate path) used to throw "group-by
/// over Float64 values: cast to Float32 first". It now counts the non-null values, as SQL does.
#[test]
fn count_over_float64_next_to_a_float64_aggregate() {
    let k: ArrayRef = Arc::new(Int32Array::from(vec![0, 1, 0, 1]));
    let v: ArrayRef = Arc::new(Float64Array::from(vec![Some(1.0), Some(2.0), None, Some(4.0)]));
    let s = src(vec![("k", k), ("v", v)]);
    let c = r#"(col \"v\")"#;
    for other in ["sum", "min", "max", "mean"] {
        let aggs = format!(r#"["{other}","a","{c}"],["count","n","{c}"]"#);
        let out = run_plan(&group_by(&aggs), &[&s], true).unwrap_or_else(|e| panic!("{other}: {e}"));
        let keys = as_i64(&out.column(0).unwrap().to_arrow().unwrap());
        let n = as_i64(&out.column(2).unwrap().to_arrow().unwrap());
        let mut got: Vec<(i64, i64)> = keys.into_iter().zip(n).collect();
        got.sort();
        assert_eq!(got, vec![(0, 1), (1, 2)], "{other}");
    }
}

/// Fixed in the core (a387a2a). `count(s)` over utf8 with nulls used to be rejected alone, and next
/// to a Float64 `sum` (per-aggregate path) it counted the null rows (4, a silent wrong answer).
/// Both paths now give SQL's 2.
#[test]
fn count_over_utf8_counts_non_null_values_on_every_path() {
    let k: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 1, 1, 1]));
    let s: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), None, Some("b"), None]));
    let v: ArrayRef = Arc::new(Float64Array::from(vec![1.0f64, 2.0, 3.0, 4.0]));
    let source = src(vec![("k", k), ("s", s), ("v", v)]);
    let out = run_plan(&group_by(r#"["count","n","(col \"s\")"]"#), &[&source], true).unwrap();
    assert_eq!(as_i64(&out.column(1).unwrap().to_arrow().unwrap()), vec![2], "alone");

    let out = run_plan(&group_by(r#"["count","n","(col \"s\")"],["sum","t","(col \"v\")"]"#), &[&source], true)
        .unwrap();
    assert_eq!(as_i64(&out.column(1).unwrap().to_arrow().unwrap()), vec![2], "next to a Float64 sum");
}

/// The IEEE float order (the plan key's default) ties -0.0 with +0.0 and keeps NaN next to the
/// nulls; the crate sends `float_order: "total"`, IEEE 754 totalOrder, which is arrow-rs's order
/// (and so DataFusion's): -NaN < -inf < ... < -0.0 < +0.0 < ... < +inf < +NaN, NaNs by payload,
/// with the nulls first or last, and the exact mirror when descending. Checked against
/// `arrow::compute::sort_to_indices`, bit for bit, for Float64 and Float32, as a full sort and as a
/// top-k (sort + limit).
#[test]
fn sort_float_order_total_is_arrow_rs_order() {
    let neg_nan = f64::from_bits(0xFFF8_0000_0000_0000);
    let payload_nan = f64::from_bits(0x7FF0_0000_0000_0001);
    let neg_payload_nan = f64::from_bits(0xFFF4_0000_0000_0003);
    let vals: Vec<Option<f64>> = vec![
        Some(0.0),
        Some(-0.0),
        Some(f64::NAN),
        None,
        Some(neg_nan),
        Some(1.5),
        Some(f64::NEG_INFINITY),
        Some(payload_nan),
        Some(-2.0),
        None,
        Some(f64::INFINITY),
        Some(neg_payload_nan),
        Some(-0.0),
        Some(0.0),
        Some(f64::MIN_POSITIVE / 2.0),
    ];
    // IEEE (the default): the zeros tie and stay in input order.
    let zeros: ArrayRef = Arc::new(Float64Array::from(vec![0.0f64, -0.0, 0.0, -0.0]));
    let s = src(vec![("x", zeros)]);
    let out = run_plan(r#"{"op":"sort","by":[["x",false]],"input":{"op":"scan","source":"t"}}"#, &[&s], true)
        .unwrap();
    assert_eq!(f64_bits(&out.column(0).unwrap().to_arrow().unwrap()), vec![0, 1 << 63, 0, 1 << 63]);

    // Float32 from the same values; f32 NaN payloads of their own.
    let mut f32_vals: Vec<Option<f32>> = vals.iter().map(|v| v.map(|x| x as f32)).collect();
    f32_vals.push(Some(f32::from_bits(0xFFC0_0001)));
    f32_vals.push(Some(f32::from_bits(0x7F80_0002)));
    let arrays: [(&str, ArrayRef); 2] = [
        ("f64", Arc::new(Float64Array::from(vals.clone()))),
        ("f32", Arc::new(Float32Array::from(f32_vals))),
    ];
    for (ty, x) in arrays {
        let s = src(vec![("x", Arc::clone(&x))]);
        for descending in [false, true] {
            for nulls_first in [false, true] {
                let want_idx = arrow::compute::sort_to_indices(
                    x.as_ref(),
                    Some(arrow::compute::SortOptions { descending, nulls_first }),
                    None,
                )
                .unwrap();
                let want = arrow::compute::take(x.as_ref(), &want_idx, None).unwrap();
                let nulls = if nulls_first { "first" } else { "last" };
                let key = format!(r#"["x",{descending},{{"nulls":"{nulls}","float_order":"total"}}]"#);
                let sort = format!(r#"{{"op":"sort","by":[{key}],"input":{{"op":"scan","source":"t"}}}}"#);
                for limit in [None, Some(5usize)] {
                    let plan = match limit {
                        Some(n) => format!(r#"{{"op":"limit","count":{n},"input":{sort}}}"#),
                        None => sort.clone(),
                    };
                    let got = run_plan(&plan, &[&s], true).unwrap().column(0).unwrap().to_arrow().unwrap();
                    let want = want.slice(0, limit.unwrap_or(want.len()));
                    assert_eq!(
                        float_bits(&got),
                        float_bits(&want),
                        "{ty} descending={descending} nulls_first={nulls_first} limit={limit:?}"
                    );
                }
            }
        }
    }
}

/// Float values by bit pattern, nulls as None.
fn float_bits(a: &ArrayRef) -> Vec<Option<u64>> {
    if a.data_type() == &DataType::Float32 {
        let p = a.as_primitive::<Float32Type>();
        return (0..p.len()).map(|i| p.is_valid(i).then(|| p.value(i).to_bits() as u64)).collect();
    }
    let p = a.as_primitive::<Float64Type>();
    (0..p.len()).map(|i| p.is_valid(i).then(|| p.value(i).to_bits())).collect()
}

/// Semantics difference, not a defect. DataFusion 55 keeps every NaN bit pattern of a float group
/// key as its own group (`HashValue::canonicalize` folds only -0.0 into +0.0). The plan runner
/// puts every NaN in one group; with one NaN bit pattern the key keeps its bits, so the two agree.
/// The crate sends a node back to DataFusion when a float key column holds NaNs of more than one
/// bit pattern.
#[test]
fn float_group_key_nans() {
    let neg_nan = f64::from_bits(0xFFF8_0000_0000_0000);
    let payload_nan = f64::from_bits(0x7FF0_0000_0000_0001);
    let k: ArrayRef = Arc::new(Float64Array::from(vec![f64::NAN, neg_nan, payload_nan, f64::NAN, 1.0]));
    let v: ArrayRef = Arc::new(Int64Array::from(vec![1i64; 5]));
    let s = src(vec![("k", k), ("v", v)]);
    let out = run_plan(&group_by(r#"["count","n"]"#), &[&s], true).unwrap();
    let keys = f64_bits(&out.column(0).unwrap().to_arrow().unwrap());
    let n = as_i64(&out.column(1).unwrap().to_arrow().unwrap());
    let mut got: Vec<(u64, i64)> = keys.into_iter().zip(n).collect();
    got.sort();
    assert_eq!(got, vec![(1f64.to_bits(), 1), (f64::NAN.to_bits(), 4)]);
    // One NaN bit pattern alone: which key comes back.
    for nan in [neg_nan, payload_nan] {
        let k: ArrayRef = Arc::new(Float64Array::from(vec![nan, nan, 2.0]));
        let v: ArrayRef = Arc::new(Int64Array::from(vec![1i64; 3]));
        let s = src(vec![("k", k), ("v", v)]);
        let out = run_plan(&group_by(r#"["count","n"]"#), &[&s], true).unwrap();
        let mut keys = f64_bits(&out.column(0).unwrap().to_arrow().unwrap());
        keys.sort();
        assert_eq!(keys, vec![2f64.to_bits(), nan.to_bits()], "{:x}", nan.to_bits());
    }
}

/// The chunked import the crate now uses: a column held as many arrow-rs arrays imports as one
/// array, with each chunk's nulls and offsets, and runs through the plan runner.
#[test]
fn chunked_import_of_a_column() {
    let a = Int64Array::from(vec![Some(3i64), None, Some(1)]);
    let b = Int64Array::from(vec![Some(2i64), Some(5), None, Some(4)]).slice(1, 3); // [5, null, 4]
    let x = Array::from_arrow_chunks(&[&a, &b]).unwrap();
    assert_eq!(x.len(), 6);
    let s = Source::new("t", vec![("x".to_string(), x)]).unwrap();
    let out = run_plan(
        r#"{"op":"sort","by":[["x",false,{"nulls":"first"}]],"input":{"op":"scan","source":"t"}}"#,
        &[&s],
        true,
    )
    .unwrap();
    let got = out.column(0).unwrap().to_arrow().unwrap();
    let got: Vec<Option<i64>> = got.as_primitive::<Int64Type>().iter().collect();
    assert_eq!(got, vec![None, None, Some(1), Some(3), Some(4), Some(5)]);
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

/// The plan runner used to reject a group_by with no aggregates (a DISTINCT); on the current core it
/// returns the distinct keys, and the crate sends DISTINCT that way.
#[test]
fn group_by_without_aggregates_returns_the_distinct_keys() {
    let k: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 1, 2]));
    let s = src(vec![("k", k)]);
    let out = run_plan(&group_by(""), &[&s], true).unwrap();
    assert_eq!(out.column_count(), 1);
    let mut got = as_i64(&out.column(0).unwrap().to_arrow().unwrap());
    got.sort();
    assert_eq!(got, vec![1, 2]);
}
