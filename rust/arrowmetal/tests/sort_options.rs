//! Sort options (null placement, IEEE 754 totalOrder) against arrow-rs's own sort.
//!
//! arrow-rs orders floats with `total_cmp` and places nulls by `SortOptions::nulls_first`, so
//! `SortOptions::from(arrow::compute::SortOptions)` must reproduce `arrow::compute::sort` bit for bit
//! on columns full of the values the two float orders disagree on: NaN of both signs and several
//! payloads, -0.0 and +0.0, the infinities and subnormals.

mod common;

use std::sync::Arc;

use arrow::array::{Array as _, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array};
use arrow::compute::{lexsort_to_indices, sort, take, SortColumn};
use arrowmetal::{lexsort, run_plan, Array, FloatOrder, SortOptions, Source};
use rand::Rng;

use common::rng;

const SIZES: &[usize] = &[0, 1, 33, 1025, 8193, 100_001, 300_007];

/// Float64 values drawn mostly from the awkward set, with a null every `null_every` rows.
fn awkward_f64(n: usize, null_every: usize, seed: u64) -> Float64Array {
    let special = [
        f64::NAN,
        -f64::NAN,
        f64::from_bits(0x7FF0_0000_0000_0001),
        f64::from_bits(0xFFF8_0000_0000_0042),
        f64::from_bits(0x7FF8_0000_0000_0007),
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MIN_POSITIVE / 4.0,
        -f64::MIN_POSITIVE / 8.0,
        1.5,
        -1.5,
    ];
    let mut r = rng(seed);
    (0..n)
        .map(|i| {
            if null_every != 0 && i % null_every == 0 {
                None
            } else if r.random_range(0..3) == 0 {
                Some(special[r.random_range(0..special.len())])
            } else {
                Some(r.random_range(-100.0f64..100.0).round() / 4.0)
            }
        })
        .collect()
}

fn awkward_f32(n: usize, null_every: usize, seed: u64) -> Float32Array {
    let special = [
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7F80_0001),
        f32::from_bits(0xFFC0_0042),
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MIN_POSITIVE / 4.0,
        2.5,
    ];
    let mut r = rng(seed);
    (0..n)
        .map(|i| {
            if null_every != 0 && i % null_every == 0 {
                None
            } else if r.random_range(0..3) == 0 {
                Some(special[r.random_range(0..special.len())])
            } else {
                Some(r.random_range(-50.0f32..50.0).round() / 2.0)
            }
        })
        .collect()
}

/// Validity and bit patterns, so NaN payloads and zero signs are compared exactly.
fn bits(a: &ArrayRef) -> Vec<Option<u64>> {
    if let Some(f) = a.as_any().downcast_ref::<Float64Array>() {
        return (0..f.len()).map(|i| f.is_valid(i).then(|| f.value(i).to_bits())).collect();
    }
    if let Some(f) = a.as_any().downcast_ref::<Float32Array>() {
        return (0..f.len()).map(|i| f.is_valid(i).then(|| f.value(i).to_bits() as u64)).collect();
    }
    let f = a.as_any().downcast_ref::<Int64Array>().expect("f64, f32 or i64");
    (0..f.len()).map(|i| f.is_valid(i).then(|| f.value(i) as u64)).collect()
}

fn options() -> Vec<arrow::compute::SortOptions> {
    let mut v = Vec::new();
    for descending in [false, true] {
        for nulls_first in [false, true] {
            v.push(arrow::compute::SortOptions { descending, nulls_first });
        }
    }
    v
}

fn check_column(a: ArrayRef, what: &str) {
    let n = a.len();
    let gpu = Array::from_arrow(a.as_ref()).unwrap();
    let all = bits(&a);
    for o in options() {
        let want = bits(&sort(a.as_ref(), Some(o)).unwrap());
        let opts = SortOptions::from(o);
        assert_eq!(opts.float_order, FloatOrder::Total);

        let got = bits(&gpu.sort_with(opts).unwrap().to_arrow().unwrap());
        assert_eq!(got, want, "{what} sort n={n} {o:?}");

        let idx = gpu.argsort_with(opts).unwrap().to_arrow().unwrap();
        let idx = idx.as_any().downcast_ref::<Int32Array>().unwrap().clone();
        let applied = bits(&take(a.as_ref(), &idx, None).unwrap());
        assert_eq!(applied, want, "{what} take(argsort) n={n} {o:?}");
        // Stable: equal keys (identical bits, or both null) keep their input order.
        for w in idx.values().windows(2) {
            let (i, j) = (w[0] as usize, w[1] as usize);
            if all[i] == all[j] {
                assert!(i < j, "{what} argsort not stable at n={n} {o:?}");
            }
        }

        for k in [1usize, 17, 100] {
            let top = gpu.top_k_with(k, opts).unwrap().to_arrow().unwrap();
            let top = top.as_any().downcast_ref::<Int32Array>().unwrap();
            let kk = k.min(n);
            assert_eq!(top.len(), kk, "{what} top_k len n={n} k={k} {o:?}");
            assert_eq!(top.values()[..], idx.values()[..kk], "{what} top_k n={n} k={k} {o:?}");
        }
    }
}

#[test]
fn float64_total_order_and_null_placement_match_arrow_rs() {
    for &n in SIZES {
        for null_every in [0usize, 10, 1] {
            check_column(Arc::new(awkward_f64(n, null_every, 7 + n as u64)), "f64");
        }
    }
}

#[test]
fn float32_total_order_and_null_placement_match_arrow_rs() {
    for &n in SIZES {
        for null_every in [0usize, 7] {
            check_column(Arc::new(awkward_f32(n, null_every, 11 + n as u64)), "f32");
        }
    }
}

#[test]
fn int64_null_placement_matches_arrow_rs() {
    for &n in &[0usize, 1, 1025, 100_001] {
        check_column(Arc::new(common::int64(n, 5, 13 + n as u64)), "i64");
    }
}

/// The default options are the plain sort, bit for bit.
#[test]
fn default_options_are_the_plain_sort() {
    let a: ArrayRef = Arc::new(awkward_f64(100_001, 9, 3));
    let gpu = Array::from_arrow(a.as_ref()).unwrap();
    for descending in [false, true] {
        let opts = SortOptions { descending, ..SortOptions::default() };
        let plain = gpu.argsort(descending).unwrap().to_arrow().unwrap();
        let with = gpu.argsort_with(opts).unwrap().to_arrow().unwrap();
        assert_eq!(&plain, &with);
    }
}

/// Two keys, each with its own options, against `lexsort_to_indices`, compared as the applied rows.
#[test]
fn lexsort_with_per_key_options_matches_arrow_rs() {
    for &n in &[1usize, 1025, 100_001] {
        let mut r = rng(n as u64);
        let k1: ArrayRef = Arc::new(
            (0..n)
                .map(|i| if i % 11 == 0 { None } else { Some(r.random_range(0..5i64)) })
                .collect::<Int64Array>(),
        );
        let k2: ArrayRef = Arc::new(awkward_f64(n, 13, 5 + n as u64));
        let g1 = Array::from_arrow(k1.as_ref()).unwrap();
        let g2 = Array::from_arrow(k2.as_ref()).unwrap();
        for o1 in options() {
            for o2 in options() {
                let cols = vec![
                    SortColumn { values: k1.clone(), options: Some(o1) },
                    SortColumn { values: k2.clone(), options: Some(o2) },
                ];
                let want_idx = lexsort_to_indices(&cols, None).unwrap();
                let got = lexsort(&[&g1, &g2], &[o1.into(), o2.into()]).unwrap().to_arrow().unwrap();
                let got = got.as_any().downcast_ref::<Int32Array>().unwrap();
                let got_idx: Vec<u32> = got.values().iter().map(|&v| v as u32).collect();
                let want_idx: Vec<u32> = want_idx.values().to_vec();
                for c in [&k1, &k2] {
                    let w = bits(&take(c.as_ref(), &arrow::array::UInt32Array::from(want_idx.clone()), None).unwrap());
                    let g = bits(&take(c.as_ref(), &arrow::array::UInt32Array::from(got_idx.clone()), None).unwrap());
                    assert_eq!(g, w, "lexsort n={n} {o1:?} {o2:?}");
                }
            }
        }
    }
}

/// The plan grammar's per-key fields: a descending totalOrder key with nulls first, then a limit
/// (the fused top-k), against arrow-rs.
#[test]
fn plan_sort_key_options_match_arrow_rs() {
    let x = awkward_f64(200_003, 17, 99);
    let a: ArrayRef = Arc::new(x.clone());
    let src = Source::new("t", vec![("x".to_string(), Array::from_arrow(&x).unwrap())]).unwrap();
    let o = arrow::compute::SortOptions { descending: true, nulls_first: true };
    let want = bits(&sort(a.as_ref(), Some(o)).unwrap());
    for (limit, plan) in [
        (None, r#"{"op":"sort","by":[["x",true,{"nulls":"first","float_order":"total"}]],"input":{"op":"scan","source":"t"}}"#.to_string()),
        (Some(100usize), r#"{"op":"limit","count":100,"input":{"op":"sort","by":[{"column":"x","descending":true,"nulls":"first"}],"float_order":"total","input":{"op":"scan","source":"t"}}}"#.to_string()),
    ] {
        let out = run_plan(&plan, &[&src], true).unwrap();
        let got = bits(&out.column(0).unwrap().to_arrow().unwrap());
        let n = limit.unwrap_or(want.len());
        assert_eq!(got, want[..n].to_vec(), "plan {plan}");
    }
}
