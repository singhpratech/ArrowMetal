//! Every wrapped kernel against arrow-rs's own compute kernels on the same data.
//!
//! The oracle is `arrow::compute` wherever arrow-rs has the kernel. It has no hash aggregation (that
//! lives in DataFusion, not in the `arrow` crate), so `group_by` is checked against a plain
//! `HashMap` fold over the same arrays instead; the test says so where that happens.
//!
//! Every sweep runs the degenerate lengths (0, 1), a word boundary, a threadgroup boundary and
//! 1,000,001 -- an odd length past a million with a partial tail -- with and without nulls, and the
//! selection kernels also run on an array the producer sliced (`offset != 0`).

mod common;

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{
    Array as _, ArrayRef, BooleanArray, DictionaryArray, Float64Array, Int32Array, Int64Array,
    Scalar as ArrowScalar, StructArray,
};
use arrow::compute::kernels::cmp;
use arrow::compute::{filter, sort, take, SortOptions};
use arrow::datatypes::{Field, Int32Type};
use arrowmetal::{group_by, Agg, Array, CompareOp, Scalar};

use common::{arc, close, float64, int64, LENGTHS};

// =================================================================================================
// Identity and interop
// =================================================================================================

#[test]
fn reports_a_version_and_a_device() {
    assert!(!arrowmetal::version().is_empty());
    assert!(!arrowmetal::device_name().is_empty());
}

/// Import then export must give back exactly the array arrow-rs handed over, at every length, with
/// and without nulls, for both element types the safe crate covers.
#[test]
fn c_data_interface_round_trip() {
    for &n in LENGTHS {
        for &null_every in &[0usize, 3] {
            let i: ArrayRef = arc(int64(n, null_every, 1));
            let out = common::roundtrip(&i);
            assert_eq!(&out, &i, "int64 round trip at n={n} null_every={null_every}");

            let f: ArrayRef = arc(float64(n, null_every, 2));
            let out = common::roundtrip(&f);
            assert_eq!(&out, &f, "float64 round trip at n={n} null_every={null_every}");
        }
    }
}

/// A producer-sliced array (Arrow `offset != 0`) must import as the rows it names, not as the rows
/// the underlying buffer starts with. ArrowMetal carries the offset rather than applying it, so this
/// is the test that catches an off-by-offset.
#[test]
fn round_trip_of_a_sliced_array() {
    let base: ArrayRef = arc(int64(100_001, 3, 7));
    for &(off, len) in &[(1usize, 10usize), (7, 1000), (31, 99_000), (64, 1), (33, 0)] {
        let sliced = base.slice(off, len);
        let out = common::roundtrip(&sliced);
        assert_eq!(&out, &sliced, "slice({off}, {len})");
    }
}

#[test]
fn length_and_null_count_agree_with_arrow() {
    let a: ArrayRef = arc(int64(1025, 4, 11));
    let gpu = Array::from_arrow(a.as_ref()).unwrap();
    assert_eq!(gpu.len(), a.len());
    assert_eq!(gpu.null_count(), a.null_count());
    assert_eq!(gpu.format(), "l");

    let sliced = a.slice(7, 500);
    let gpu = Array::from_arrow(sliced.as_ref()).unwrap();
    assert_eq!(gpu.len(), 500);
    assert_eq!(gpu.null_count(), sliced.null_count());
}

// =================================================================================================
// Reductions
// =================================================================================================

/// `sum` / `min` / `max` on int64 against `arrow::compute`, which is exact for all three, so these
/// are equality assertions. The values are bounded so an i64 sum cannot overflow at these lengths;
/// ArrowMetal and Arrow both wrap, but a test that relied on the wrap would be testing nothing.
#[test]
fn int64_reductions_match_arrow() {
    for &n in LENGTHS {
        for &null_every in &[0usize, 3, 1] {
            let a = int64(n, null_every, 21 + n as u64);
            let gpu = Array::from_arrow(&a).unwrap();

            let want = arrow::compute::sum(&a);
            let got = gpu.sum().unwrap().map(|s| s.as_i64().expect("int64 sum is Int64"));
            assert_eq!(got, want, "sum at n={n} null_every={null_every}");

            let want = arrow::compute::min(&a);
            let got = gpu.min().unwrap().and_then(Scalar::as_i64);
            assert_eq!(got, want, "min at n={n} null_every={null_every}");

            let want = arrow::compute::max(&a);
            let got = gpu.max().unwrap().and_then(Scalar::as_i64);
            assert_eq!(got, want, "max at n={n} null_every={null_every}");
        }
    }
}

/// `mean` has no arrow-rs kernel in the `arrow` crate, so the oracle is arrow's own exact `sum` and
/// `len - null_count`, divided in f64. That division is one rounding; ArrowMetal accumulates on the
/// GPU, so this is a relative tolerance, not an equality.
#[test]
fn int64_mean_matches_arrow_sum_over_count() {
    for &n in LENGTHS {
        for &null_every in &[0usize, 3] {
            let a = int64(n, null_every, 31 + n as u64);
            let gpu = Array::from_arrow(&a).unwrap();
            let valid = a.len() - a.null_count();
            let want = arrow::compute::sum(&a).map(|s| s as f64 / valid as f64);
            let got = gpu.mean().unwrap().map(Scalar::as_f64);
            match (got, want) {
                (None, None) => {}
                (Some(g), Some(w)) => assert!(
                    close(g, w, 1e-12),
                    "mean at n={n} null_every={null_every}: got {g}, arrow {w}"
                ),
                other => panic!("mean nullness disagrees at n={n}: {other:?}"),
            }
        }
    }
}

/// float64 `min` and `max` are exact comparisons, so they must match arrow exactly. `sum` and `mean`
/// reassociate on the GPU, so they get a relative tolerance -- 1e-12 over a million values, which is
/// far inside float64's own accumulation error.
#[test]
fn float64_reductions_match_arrow() {
    for &n in LENGTHS {
        for &null_every in &[0usize, 3, 1] {
            let a = float64(n, null_every, 41 + n as u64);
            let gpu = Array::from_arrow(&a).unwrap();

            assert_eq!(
                gpu.min().unwrap().map(Scalar::as_f64),
                arrow::compute::min(&a),
                "min at n={n} null_every={null_every}"
            );
            assert_eq!(
                gpu.max().unwrap().map(Scalar::as_f64),
                arrow::compute::max(&a),
                "max at n={n} null_every={null_every}"
            );

            let want = arrow::compute::sum(&a);
            let got = gpu.sum().unwrap().map(Scalar::as_f64);
            match (got, want) {
                (None, None) => {}
                (Some(g), Some(w)) => {
                    assert!(close(g, w, 1e-12), "sum at n={n}: got {g}, arrow {w}")
                }
                other => panic!("sum nullness disagrees at n={n}: {other:?}"),
            }
        }
    }
}

/// NaN in `min` / `max` is the one place the two libraries disagree, so it is pinned rather than
/// compared.
///
/// The two rules are each self-consistent and neither is a bug:
///
/// * **arrow-rs** puts NaN at the top of a total order. Both `min` and `max` document it in the same
///   sentence -- "For floating point arrays any NaN values are considered to be greater than any
///   other non-null value" (`arrow_arith::aggregate`, 59.3.0) -- so `min` returns the smallest
///   non-NaN and `max` returns NaN. This was reported as apache/arrow-rs#101 and closed as
///   intended behaviour in 2022.
/// * **ArrowMetal** skips NaN in both, the way it skips a null, and reports "no valid value" when
///   every valid element is NaN. That follows Arrow C++ / pyarrow, which also skips NaN -- though
///   pyarrow returns NaN rather than null for an all-NaN column, so ArrowMetal differs from it in
///   that case too.
///
/// `docs/RUST.md` carries the same note.
#[test]
fn nan_handling_diverges_from_arrow_rs_and_is_pinned() {
    let mixed =
        Float64Array::from(vec![Some(f64::NAN), Some(2.0), None, Some(-1.0), Some(f64::NAN)]);
    let gpu = Array::from_arrow(&mixed).unwrap();

    // ArrowMetal: NaN is skipped by both.
    assert_eq!(gpu.min().unwrap().map(Scalar::as_f64), Some(-1.0));
    assert_eq!(gpu.max().unwrap().map(Scalar::as_f64), Some(2.0));

    // arrow-rs, NaN ordered greatest: `min` agrees here by coincidence, `max` returns NaN.
    assert_eq!(arrow::compute::min(&mixed), Some(-1.0));
    assert!(
        arrow::compute::max(&mixed).is_some_and(f64::is_nan),
        "arrow-rs stopped ordering NaN greatest; re-check the divergence note in docs/RUST.md"
    );

    // Every valid value NaN: ArrowMetal reports no valid value, arrow-rs returns NaN from both
    // (NaN is simply the greatest and the least element present).
    let all_nan = Float64Array::from(vec![Some(f64::NAN), Some(f64::NAN)]);
    let gpu = Array::from_arrow(&all_nan).unwrap();
    assert_eq!(gpu.min().unwrap(), None);
    assert_eq!(gpu.max().unwrap(), None);
    assert!(arrow::compute::min(&all_nan).is_some_and(f64::is_nan));
    assert!(arrow::compute::max(&all_nan).is_some_and(f64::is_nan));

    // With no NaN anywhere the two libraries agree exactly, which is what the sweep above relies on.
    let clean = Float64Array::from(vec![Some(3.0), None, Some(-4.5), Some(0.0)]);
    let gpu = Array::from_arrow(&clean).unwrap();
    assert_eq!(gpu.min().unwrap().map(Scalar::as_f64), arrow::compute::min(&clean));
    assert_eq!(gpu.max().unwrap().map(Scalar::as_f64), arrow::compute::max(&clean));
}

// =================================================================================================
// Compare and filter
// =================================================================================================

/// `compare_scalar(Gt) + filter` against `arrow::compute::kernels::cmp::gt` + `arrow::compute::filter`
/// on the same array, including the boolean mask itself.
#[test]
fn compare_scalar_and_filter_match_arrow() {
    for &n in LENGTHS {
        for &null_every in &[0usize, 3] {
            let a = int64(n, null_every, 51 + n as u64);
            let a_ref: ArrayRef = arc(a.clone());
            let gpu = Array::from_arrow(&a).unwrap();

            let threshold = 0i64;
            let want_mask = cmp::gt(
                &a,
                &ArrowScalar::new(&Int64Array::from(vec![threshold])),
            )
            .unwrap();
            let gpu_mask = gpu.compare_scalar(CompareOp::Gt, threshold).unwrap();
            let got_mask = gpu_mask.to_arrow().unwrap();
            assert_eq!(
                got_mask.as_ref(),
                &want_mask as &dyn arrow::array::Array,
                "mask at n={n} null_every={null_every}"
            );

            let want = filter(a_ref.as_ref(), &want_mask).unwrap();
            let got = gpu.filter(&gpu_mask).unwrap().to_arrow().unwrap();
            assert_eq!(&got, &want, "filter at n={n} null_every={null_every}");
        }
    }
}

/// Every comparison operator, array against array, on a pair that shares nulls in some rows and not
/// in others.
#[test]
fn every_compare_operator_matches_arrow() {
    let a = int64(100_001, 3, 61);
    let b = int64(100_001, 5, 62);
    let ga = Array::from_arrow(&a).unwrap();
    let gb = Array::from_arrow(&b).unwrap();

    let cases: &[(CompareOp, fn(&Int64Array, &Int64Array) -> BooleanArray)] = &[
        (CompareOp::Eq, |x, y| cmp::eq(x, y).unwrap()),
        (CompareOp::Ne, |x, y| cmp::neq(x, y).unwrap()),
        (CompareOp::Lt, |x, y| cmp::lt(x, y).unwrap()),
        (CompareOp::Le, |x, y| cmp::lt_eq(x, y).unwrap()),
        (CompareOp::Gt, |x, y| cmp::gt(x, y).unwrap()),
        (CompareOp::Ge, |x, y| cmp::gt_eq(x, y).unwrap()),
    ];
    for (op, oracle) in cases {
        let want = oracle(&a, &b);
        let got = ga.compare(*op, &gb).unwrap().to_arrow().unwrap();
        assert_eq!(got.as_ref(), &want as &dyn arrow::array::Array, "{op:?}");
    }
}

/// Filtering a producer-sliced array: both sides see the same rows.
#[test]
fn filter_on_a_sliced_array_matches_arrow() {
    let base: ArrayRef = arc(int64(200_003, 4, 71));
    let sliced = base.slice(33, 150_000);
    let ints = sliced.as_any().downcast_ref::<Int64Array>().unwrap();
    let gpu = Array::from_arrow(ints).unwrap();

    let want_mask = cmp::gt(ints, &ArrowScalar::new(&Int64Array::from(vec![100i64]))).unwrap();
    let gpu_mask = gpu.compare_scalar(CompareOp::Gt, 100i64).unwrap();

    let want = filter(sliced.as_ref(), &want_mask).unwrap();
    let got = gpu.filter(&gpu_mask).unwrap().to_arrow().unwrap();
    assert_eq!(&got, &want);
}

/// A mask with nulls in it. arrow-rs's `filter` drops a null mask entry; this pins that ArrowMetal
/// agrees, which is the sentence the crate docs make about `filter`.
#[test]
fn filter_with_a_null_mask_matches_arrow() {
    let values: ArrayRef = arc(Int64Array::from((0..1025).collect::<Vec<i64>>()));
    let mask: BooleanArray = (0..1025)
        .map(|i| match i % 3 {
            0 => None,
            1 => Some(true),
            _ => Some(false),
        })
        .collect();
    let gpu = Array::from_arrow(values.as_ref()).unwrap();
    let gpu_mask = Array::from_arrow(&mask).unwrap();

    let want = filter(values.as_ref(), &mask).unwrap();
    let got = gpu.filter(&gpu_mask).unwrap().to_arrow().unwrap();
    assert_eq!(&got, &want);
}

// =================================================================================================
// Sorting and take
// =================================================================================================

/// `sort` against `arrow::compute::sort` with nulls last, which is ArrowMetal's placement in both
/// directions.
#[test]
fn sort_matches_arrow() {
    for &n in LENGTHS {
        for &null_every in &[0usize, 4] {
            for &descending in &[false, true] {
                let a: ArrayRef = arc(int64(n, null_every, 81 + n as u64));
                let gpu = Array::from_arrow(a.as_ref()).unwrap();

                let opts = SortOptions { descending, nulls_first: false };
                let want = sort(a.as_ref(), Some(opts)).unwrap();
                let got = gpu.sort(descending).unwrap().to_arrow().unwrap();
                assert_eq!(&got, &want, "sort n={n} desc={descending} nulls={null_every}");
            }
        }
    }
}

/// `argsort` gives indices, and both libraries' sorts are stable, but a tie could still in principle
/// be broken differently -- so this asserts the *permutation applied* equals arrow's sorted array,
/// which is the property that actually matters, and additionally that the indices are a permutation
/// of `0..n`.
#[test]
fn argsort_indices_reproduce_arrows_sorted_order() {
    for &n in &[0usize, 1, 1024, 1025, 1_000_001] {
        for &descending in &[false, true] {
            let a: ArrayRef = arc(int64(n, 4, 91 + n as u64));
            let gpu = Array::from_arrow(a.as_ref()).unwrap();

            let idx = gpu.argsort(descending).unwrap();
            let idx_arrow = idx.to_arrow().unwrap();
            let idx_i32 = idx_arrow.as_any().downcast_ref::<Int32Array>().unwrap();
            assert_eq!(idx_i32.len(), n, "argsort length at n={n}");
            assert_eq!(idx_i32.null_count(), 0, "argsort indices are never null");
            let mut seen = vec![false; n];
            for v in idx_i32.values() {
                let v = *v as usize;
                assert!(v < n && !seen[v], "argsort indices are not a permutation at n={n}");
                seen[v] = true;
            }

            let opts = SortOptions { descending, nulls_first: false };
            let want = sort(a.as_ref(), Some(opts)).unwrap();
            let got = take(a.as_ref(), idx_i32, None).unwrap();
            assert_eq!(&got, &want, "take(argsort) at n={n} desc={descending}");
        }
    }
}

/// `take` against `arrow::compute::take`, with out-of-order and repeated indices and a null index.
#[test]
fn take_matches_arrow() {
    let a: ArrayRef = arc(int64(100_001, 3, 101));

    let mut r = common::rng(102);
    let indices: Int32Array = (0..250_000)
        .map(|i| {
            if i % 1000 == 0 {
                None
            } else {
                Some(rand::Rng::random_range(&mut r, 0i32..100_001))
            }
        })
        .collect();

    let gpu = Array::from_arrow(a.as_ref()).unwrap();
    let gpu_idx = Array::from_arrow(&indices).unwrap();

    let want = take(a.as_ref(), &indices, None).unwrap();
    let got = gpu.take(&gpu_idx).unwrap().to_arrow().unwrap();
    assert_eq!(&got, &want);
}

/// The empty and single-element cases for take, which are their own code paths on the GPU.
#[test]
fn take_degenerate_lengths_match_arrow() {
    let a: ArrayRef = arc(Int64Array::from(vec![Some(5i64), None, Some(7)]));
    let gpu = Array::from_arrow(a.as_ref()).unwrap();
    for indices in [
        Int32Array::from(Vec::<i32>::new()),
        Int32Array::from(vec![1]),
        Int32Array::from(vec![2, 2, 0, 1]),
    ] {
        let gpu_idx = Array::from_arrow(&indices).unwrap();
        let want = take(a.as_ref(), &indices, None).unwrap();
        let got = gpu.take(&gpu_idx).unwrap().to_arrow().unwrap();
        assert_eq!(&got, &want, "take({indices:?})");
    }
}

/// `slice` on the handle, against arrow-rs's own slice.
#[test]
fn slice_matches_arrow() {
    let a: ArrayRef = arc(int64(100_001, 3, 111));
    let gpu = Array::from_arrow(a.as_ref()).unwrap();
    for &(off, len) in &[(0usize, 0usize), (0, 1), (1, 99_999), (64, 32_768), (100_000, 1)] {
        let want = a.slice(off, len);
        let got = gpu.slice(off, len).unwrap().to_arrow().unwrap();
        assert_eq!(&got, &want, "slice({off}, {len})");
    }
}

// =================================================================================================
// Group-by
// =================================================================================================

/// `group_by(keys).sum(values)`.
///
/// arrow-rs's `arrow` crate has no hash aggregation, so the oracle here is a plain `HashMap` fold
/// over the same two arrays. The comparison is order-independent: ArrowMetal's group order is
/// documented as ascending-by-key, not first-seen, and this test does not depend on either.
#[test]
fn group_by_sum_matches_a_hashmap_fold() {
    for &n in &[0usize, 1, 1025, 1_000_001] {
        for &null_every in &[0usize, 7] {
            let mut r = common::rng(121 + n as u64);
            let keys: Int64Array = (0..n)
                .map(|i| {
                    if null_every != 0 && i % null_every == 0 {
                        None
                    } else {
                        Some(rand::Rng::random_range(&mut r, 0i64..17))
                    }
                })
                .collect();
            let values = int64(n, 5, 122 + n as u64);

            // Oracle: sum the non-null values per key, nulls forming their own group.
            let mut want: HashMap<Option<i64>, Option<i64>> = HashMap::new();
            for i in 0..n {
                let k = if keys.is_null(i) { None } else { Some(keys.value(i)) };
                let slot = want.entry(k).or_insert(None);
                if !values.is_null(i) {
                    *slot = Some(slot.unwrap_or(0) + values.value(i));
                }
            }

            let gk = Array::from_arrow(&keys).unwrap();
            let gv = Array::from_arrow(&values).unwrap();
            let gb = group_by(&[&gk]).unwrap();
            assert_eq!(gb.group_count(), want.len(), "group count at n={n}");
            assert_eq!(gb.key_column_count(), 1);

            let out_keys = gb.keys(0).unwrap().to_arrow().unwrap();
            let out_keys = out_keys.as_any().downcast_ref::<Int64Array>().unwrap();
            let out_sums = gb.sum(&gv).unwrap().to_arrow().unwrap();
            let out_sums = out_sums.as_any().downcast_ref::<Int64Array>().unwrap();
            assert_eq!(out_keys.len(), want.len());
            assert_eq!(out_sums.len(), want.len());

            let mut got: HashMap<Option<i64>, Option<i64>> = HashMap::new();
            for i in 0..out_keys.len() {
                let k = if out_keys.is_null(i) { None } else { Some(out_keys.value(i)) };
                let v = if out_sums.is_null(i) { None } else { Some(out_sums.value(i)) };
                assert!(got.insert(k, v).is_none(), "duplicate group key at n={n}");
            }
            assert_eq!(got, want, "group_by sum at n={n} null_every={null_every}");
        }
    }
}

/// The other grouped aggregates this crate names, against the same fold. `count_all` counts rows
/// including nulls; `count` counts non-null values; `mean` is float64.
#[test]
fn group_by_min_max_count_and_mean_match_a_hashmap_fold() {
    let n = 100_001usize;
    let mut r = common::rng(131);
    let keys: Int64Array = (0..n)
        .map(|_| Some(rand::Rng::random_range(&mut r, 0i64..11)))
        .collect();
    let values = int64(n, 6, 132);

    let mut rows: HashMap<i64, Vec<Option<i64>>> = HashMap::new();
    for i in 0..n {
        rows.entry(keys.value(i))
            .or_default()
            .push(if values.is_null(i) { None } else { Some(values.value(i)) });
    }

    let gk = Array::from_arrow(&keys).unwrap();
    let gv = Array::from_arrow(&values).unwrap();
    let gb = group_by(&[&gk]).unwrap();

    let out_keys = gb.keys(0).unwrap().to_arrow().unwrap();
    let out_keys = out_keys.as_any().downcast_ref::<Int64Array>().unwrap();

    let as_i64 = |a: ArrayRef| -> Vec<Option<i64>> {
        let a = a.as_any().downcast_ref::<Int64Array>().unwrap();
        (0..a.len()).map(|i| if a.is_null(i) { None } else { Some(a.value(i)) }).collect()
    };
    let min = as_i64(gb.agg(Agg::Min, Some(&gv)).unwrap().to_arrow().unwrap());
    let max = as_i64(gb.agg(Agg::Max, Some(&gv)).unwrap().to_arrow().unwrap());
    let count_all = as_i64(gb.agg(Agg::CountAll, None).unwrap().to_arrow().unwrap());
    let count = as_i64(gb.agg(Agg::Count, Some(&gv)).unwrap().to_arrow().unwrap());
    let mean = gb.agg(Agg::Mean, Some(&gv)).unwrap().to_arrow().unwrap();
    let mean = mean.as_any().downcast_ref::<Float64Array>().unwrap();

    for i in 0..out_keys.len() {
        let k = out_keys.value(i);
        let vs = &rows[&k];
        let valid: Vec<i64> = vs.iter().flatten().copied().collect();
        assert_eq!(min[i], valid.iter().copied().min(), "min for key {k}");
        assert_eq!(max[i], valid.iter().copied().max(), "max for key {k}");
        assert_eq!(count_all[i], Some(vs.len() as i64), "count_all for key {k}");
        assert_eq!(count[i], Some(valid.len() as i64), "count for key {k}");
        let want_mean = valid.iter().sum::<i64>() as f64 / valid.len() as f64;
        assert!(
            close(mean.value(i), want_mean, 1e-12),
            "mean for key {k}: got {}, want {want_mean}",
            mean.value(i)
        );
    }
}

/// Two key columns, so the pairwise key folding runs.
#[test]
fn group_by_two_key_columns() {
    let n = 50_000usize;
    let mut r = common::rng(141);
    let k1: Int64Array = (0..n).map(|_| Some(rand::Rng::random_range(&mut r, 0i64..5))).collect();
    let k2: Int64Array = (0..n).map(|_| Some(rand::Rng::random_range(&mut r, 0i64..3))).collect();
    let values = int64(n, 0, 142);

    let mut want: HashMap<(i64, i64), i64> = HashMap::new();
    for i in 0..n {
        *want.entry((k1.value(i), k2.value(i))).or_insert(0) += values.value(i);
    }

    let g1 = Array::from_arrow(&k1).unwrap();
    let g2 = Array::from_arrow(&k2).unwrap();
    let gv = Array::from_arrow(&values).unwrap();
    let gb = group_by(&[&g1, &g2]).unwrap();
    assert_eq!(gb.key_column_count(), 2);
    assert_eq!(gb.group_count(), want.len());

    let ok1 = gb.keys(0).unwrap().to_arrow().unwrap();
    let ok1 = ok1.as_any().downcast_ref::<Int64Array>().unwrap();
    let ok2 = gb.keys(1).unwrap().to_arrow().unwrap();
    let ok2 = ok2.as_any().downcast_ref::<Int64Array>().unwrap();
    let sums = gb.sum(&gv).unwrap().to_arrow().unwrap();
    let sums = sums.as_any().downcast_ref::<Int64Array>().unwrap();

    for i in 0..ok1.len() {
        assert_eq!(sums.value(i), want[&(ok1.value(i), ok2.value(i))]);
    }
}

/// `ids()` labels every input row, so re-summing by hand through the ids must reproduce `sum()`.
#[test]
fn group_ids_label_every_row() {
    let n = 20_001usize;
    let mut r = common::rng(151);
    let keys: Int64Array = (0..n).map(|_| Some(rand::Rng::random_range(&mut r, 0i64..9))).collect();
    let values = int64(n, 0, 152);

    let gk = Array::from_arrow(&keys).unwrap();
    let gv = Array::from_arrow(&values).unwrap();
    let gb = group_by(&[&gk]).unwrap();

    let ids = gb.ids().unwrap().to_arrow().unwrap();
    let ids = ids.as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(ids.len(), n);
    assert_eq!(ids.null_count(), 0);

    let mut by_id = vec![0i64; gb.group_count()];
    for i in 0..n {
        by_id[ids.value(i) as usize] += values.value(i);
    }
    let sums = gb.sum(&gv).unwrap().to_arrow().unwrap();
    let sums = sums.as_any().downcast_ref::<Int64Array>().unwrap();
    for (i, want) in by_id.iter().enumerate() {
        assert_eq!(sums.value(i), *want, "group {i}");
    }
}

// =================================================================================================
// Errors
// =================================================================================================

/// A scalar of the wrong width must be an error, not a four-byte read past the end of an i32.
#[test]
fn a_mismatched_scalar_type_is_an_error() {
    let a = Int64Array::from(vec![1i64, 2, 3]);
    let gpu = Array::from_arrow(&a).unwrap();
    let err = gpu.compare_scalar(CompareOp::Gt, 1i32).unwrap_err();
    assert!(err.message().contains("does not match"), "{err}");
    assert!(gpu.compare_scalar(CompareOp::Gt, 1i64).is_ok());
}

/// A failing ABI call must surface `am_last_error()`'s message, not a bare code.
#[test]
fn an_abi_error_carries_its_message() {
    let a = Int64Array::from(vec![1i64, 2, 3]);
    let b = Int64Array::from(vec![1i64, 2]);
    let ga = Array::from_arrow(&a).unwrap();
    let gb = Array::from_arrow(&b).unwrap();
    let err = ga.compare(CompareOp::Eq, &gb).unwrap_err();
    assert!(!err.message().is_empty());
    assert!(err.message().starts_with("am_compare_array:"), "{err}");
    assert!(!err.message().contains("(no message)"), "empty am_last_error: {err}");
}

#[test]
fn group_by_with_no_keys_is_an_error() {
    assert!(group_by(&[]).is_err());
}

// =================================================================================================
// Batching
// =================================================================================================

/// A chain inside a batch must give the same answer as the same chain outside one.
#[test]
fn a_batched_chain_equals_an_unbatched_one() {
    let a: ArrayRef = arc(int64(1_000_001, 3, 161));
    let gpu = Array::from_arrow(a.as_ref()).unwrap();

    let unbatched = {
        let mask = gpu.compare_scalar(CompareOp::Gt, 0i64).unwrap();
        gpu.filter(&mask).unwrap().sum().unwrap()
    };
    let batched = arrowmetal::batch(|| {
        let mask = gpu.compare_scalar(CompareOp::Gt, 0i64).unwrap();
        gpu.filter(&mask).unwrap().sum().unwrap()
    })
    .unwrap();
    assert_eq!(batched, unbatched);

    // And it is arrow's answer.
    let want_mask = cmp::gt(
        a.as_any().downcast_ref::<Int64Array>().unwrap(),
        &ArrowScalar::new(&Int64Array::from(vec![0i64])),
    )
    .unwrap();
    let want = filter(a.as_ref(), &want_mask).unwrap();
    let want = arrow::compute::sum(want.as_any().downcast_ref::<Int64Array>().unwrap());
    assert_eq!(batched.and_then(Scalar::as_i64), want);
}

/// A batch defers validation, so a bad operation inside one succeeds at the call and fails the whole
/// batch at the end. This pins that route, and is the mechanism the nesting test below relies on.
#[test]
fn a_failure_inside_a_batch_surfaces_at_the_end() {
    let a = Int64Array::from(vec![1i64, 2, 3]);
    let out_of_range = Int32Array::from(vec![0, 1, 999_999]);
    let ga = Array::from_arrow(&a).unwrap();
    let gi = Array::from_arrow(&out_of_range).unwrap();

    // Unbatched, the same take fails at the call.
    let unbatched = ga.take(&gi);
    assert!(unbatched.is_err(), "expected an immediate error when unbatched");
    assert!(unbatched.unwrap_err().message().contains("out of range"));

    // Batched, it succeeds at the call and the batch reports the failure.
    let mut took_ok = false;
    let batched = arrowmetal::batch(|| {
        took_ok = ga.take(&gi).is_ok();
    });
    assert!(took_ok, "inside a batch the take should be deferred, not validated at the call");
    let err = batched.unwrap_err();
    assert!(err.message().starts_with("am_batch_end:"), "{err}");

    // The thread is usable afterwards.
    assert_eq!(ga.sum().unwrap(), Some(Scalar::Int64(6)));
}

/// Nested `batch` calls must not collapse the outer batch.
///
/// `am_batch_begin` is a no-op when a batch is already open, but `am_batch_end` closes whatever is
/// open regardless of nesting — so without a depth counter the inner scope's guard commits the outer
/// scope's batch and the rest of the outer body runs unbatched, silently.
///
/// The discriminator is deferral, not timing: a failing `take` issued **after** the inner batch
/// returns must still be deferred (so it returns `Ok` at the call) and must fail the *outer* batch.
/// If the inner scope had closed the batch, that same take would have been validated immediately and
/// returned `Err` inside the closure, and the outer batch would have come back `Ok`.
#[test]
fn a_nested_batch_keeps_the_outer_one_open() {
    let a = Int64Array::from(vec![1i64, 2, 3]);
    let out_of_range = Int32Array::from(vec![0, 1, 999_999]);
    let ga = Array::from_arrow(&a).unwrap();
    let gi = Array::from_arrow(&out_of_range).unwrap();

    let mut inner_ok = false;
    let mut took_ok = false;
    let outer = arrowmetal::batch(|| {
        // An inner batch does no ABI work and always succeeds.
        inner_ok = arrowmetal::batch(|| ()).is_ok();
        // Issued after the inner scope has ended: still deferred, so the outer batch is still open.
        took_ok = ga.take(&gi).is_ok();
    });

    assert!(inner_ok, "the inner batch should return Ok");
    assert!(
        took_ok,
        "the take was validated at the call, so the inner batch had already closed the outer one"
    );
    let err = outer.unwrap_err();
    assert!(err.message().starts_with("am_batch_end:"), "{err}");

    // Depth is back to zero: a fresh batch still works.
    assert!(arrowmetal::batch(|| ga.sum().unwrap()).unwrap() == Some(Scalar::Int64(6)));
}

/// Three levels deep, and a panic in the middle, must still leave the depth counter at zero.
#[test]
fn batch_depth_unwinds_correctly_on_panic() {
    let a = Int64Array::from(vec![1i64, 2, 3]);
    let ga = Array::from_arrow(&a).unwrap();

    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = arrowmetal::batch(|| {
            let _ = arrowmetal::batch(|| {
                let _ = arrowmetal::batch(|| panic!("boom"));
            });
        });
    }));
    assert!(r.is_err());

    // If any level had leaked its depth, this batch would never commit and the sum would be stale
    // or the call would hang on an uncommitted buffer.
    assert_eq!(arrowmetal::batch(|| ga.sum().unwrap()).unwrap(), Some(Scalar::Int64(6)));
    assert_eq!(ga.sum().unwrap(), Some(Scalar::Int64(6)));
}

/// A panic inside a batch must still close it, or every later call on this thread is appended to a
/// command buffer nobody commits.
#[test]
fn a_panicking_batch_still_closes() {
    let r = std::panic::catch_unwind(|| {
        let _ = arrowmetal::batch(|| panic!("boom"));
    });
    assert!(r.is_err());
    // The next call on this thread must behave normally.
    let a = Int64Array::from(vec![1i64, 2, 3]);
    let gpu = Array::from_arrow(&a).unwrap();
    assert_eq!(gpu.sum().unwrap(), Some(Scalar::Int64(6)));
}

/// `cast` between the two element types the crate covers, against `arrow::compute::cast`.
#[test]
fn cast_matches_arrow() {
    let a: ArrayRef = arc(int64(100_001, 3, 171));
    let gpu = Array::from_arrow(a.as_ref()).unwrap();
    let want = arrow::compute::cast(a.as_ref(), &arrow::datatypes::DataType::Float64).unwrap();
    let got = gpu.cast("g").unwrap().to_arrow().unwrap();
    assert_eq!(&got, &want);
}

/// Handles must stay `!Send` and `!Sync`: the ABI's error slot and its command-buffer batching are
/// both thread-local, so a handle belongs to the thread that made it.
///
/// These are real compile-time assertions — `assert_not_impl_any!` fails to compile if the type ever
/// gains the trait — not a runtime check and not a comment. The `assert_impl_all!` line is the
/// control: it proves the macro can see `Send` when it is there.
#[test]
fn handles_are_not_send_or_sync() {
    use static_assertions::{assert_impl_all, assert_not_impl_any};

    assert_not_impl_any!(Array: Send, Sync);
    assert_not_impl_any!(arrowmetal::GroupBy: Send, Sync);
    assert_not_impl_any!(arrowmetal::Source: Send, Sync);
    assert_not_impl_any!(arrowmetal::PlanResult: Send, Sync);

    assert_impl_all!(Arc<Int64Array>: Send, Sync);
    // `Error` carries only a String, so it may cross threads; that is deliberate.
    assert_impl_all!(arrowmetal::Error: Send, Sync);
}

/// A dictionary-encoded array must be refused by `from_arrow`, not imported.
///
/// The ABI's `am_format` reports a dictionary's *index* type (`"i"`) while every kernel decodes the
/// dictionary and computes on the *value* type. This crate's scalar type check reads `am_format`, so
/// importing one would let a 4-byte `i32` scalar reach a kernel reading 8 bytes off a
/// `Dictionary(Int32, Float64)` column — unsound from safe Rust, and a wrong answer besides
/// (`compare_scalar(Lt, 2i32)` returned `[false, false, false]` where arrow-rs says
/// `[true, false, false]`). Rejecting at the door is what keeps the type check honest.
#[test]
fn dictionary_arrays_are_refused_at_import() {
    let values = Float64Array::from(vec![1.0, 5.0, 9.0]);
    let keys = Int32Array::from(vec![0, 1, 2]);
    let dict = DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values)).unwrap();

    let err = Array::from_arrow(&dict).unwrap_err();
    assert!(
        err.message().contains("dictionary-encoded arrays are not accepted"),
        "unexpected message: {err}"
    );
    assert!(err.message().contains("Dictionary(Int32, Float64)"), "{err}");

    // The documented way through: decode, then import. The decoded column type-checks correctly,
    // and a 4-byte scalar is now refused for what is a Float64 column.
    let decoded = arrow::compute::cast(&dict, &arrow::datatypes::DataType::Float64).unwrap();
    let gpu = Array::from_arrow(decoded.as_ref()).unwrap();
    assert_eq!(gpu.format(), "g");
    assert!(gpu.compare_scalar(CompareOp::Lt, 2i32).is_err());

    // And the answer is arrow-rs's.
    let want = cmp::lt(
        decoded.as_any().downcast_ref::<Float64Array>().unwrap(),
        &ArrowScalar::new(&Float64Array::from(vec![2.0f64])),
    )
    .unwrap();
    let got = gpu.compare_scalar(CompareOp::Lt, 2.0f64).unwrap().to_arrow().unwrap();
    assert_eq!(got.as_ref(), &want as &dyn arrow::array::Array);
    assert_eq!(want.values().iter().collect::<Vec<_>>(), vec![true, false, false]);
}

/// Refusing dictionaries at `from_arrow` closes the hole only if a dictionary nested inside an
/// accepted array cannot be pulled out into a handle of its own.
///
/// A `Struct{d: Dictionary(Int32, Float64)}` imports fine — nested types go through the C Data
/// Interface unchanged — and is harmless only because the struct handle reports `"+s"`, which no
/// `NativeType` matches, so every scalar entry point refuses it. This test pins that, and is the
/// property the "do not reopen this" note on `NativeType` rests on: wrapping `am_child`,
/// `am_struct_field`, `am_dictionary_decode`, `am_list_flatten` or `am_cast_ex` would hand back a
/// bare dictionary handle reporting `"i"` and reopen the hole.
#[test]
fn a_nested_dictionary_is_unreachable_from_the_safe_surface() {
    let values = Float64Array::from(vec![1.0, 5.0, 9.0]);
    let keys = Int32Array::from(vec![0, 1, 2]);
    let dict = DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values)).unwrap();

    let field = Arc::new(Field::new("d", dict.data_type().clone(), true));
    let st = StructArray::from(vec![(field, Arc::new(dict) as ArrayRef)]);

    let gpu = Array::from_arrow(&st).expect("a struct carrying a dictionary should import");
    assert_eq!(gpu.len(), 3);

    // The handle must never report a format any `NativeType` claims. `"+s"` is what it does report;
    // the assertion is on the property, not the spelling, so this still holds if the string changes.
    let format = gpu.format();
    for scalar_format in ["c", "C", "s", "S", "i", "I", "l", "L", "f", "g"] {
        assert_ne!(
            format, scalar_format,
            "a struct wrapping a dictionary reported the scalar format {scalar_format:?}; a scalar \
             operand would now be accepted for it"
        );
    }

    // Every scalar width is refused, so no scalar can reach the dictionary's kernel.
    assert!(gpu.compare_scalar(CompareOp::Lt, 2i32).is_err());
    assert!(gpu.compare_scalar(CompareOp::Lt, 2i64).is_err());
    assert!(gpu.compare_scalar(CompareOp::Lt, 2.0f64).is_err());
    assert!(gpu.compare_scalar(CompareOp::Lt, 2.0f32).is_err());
}
