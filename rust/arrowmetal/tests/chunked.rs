//! `Array::from_arrow_chunks` against `Array::from_arrow` of the concatenation, value for value and
//! null for null, over types and chunk layouts (empty chunks, one-row chunks, slices at non-zero
//! offsets, chunks without validity next to chunks with it, all-null chunks, one chunk, 10,000
//! chunks), then through sum, sort and group-by; and `Source::from_batches`.

mod common;

use arrow::array::{
    Array as _, ArrayRef, BinaryArray, BooleanArray, Date32Array, Decimal128Array, Float64Array,
    Int32Array, Int64Array, LargeStringArray, StringArray, StringViewArray, TimestampMicrosecondArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrowmetal::{Array, Source};
use rand::Rng;
use std::sync::Arc;

/// Rows of every tested type, with a null roughly every `null_every` rows (0: none).
fn column(ty: &str, n: usize, null_every: usize, seed: u64) -> ArrayRef {
    let mut r = common::rng(seed);
    let null = |i: usize| null_every != 0 && i % null_every == 0;
    match ty {
        "int64" => Arc::new((0..n).map(|i| (!null(i)).then(|| r.random_range(-1_000_000i64..1_000_000))).collect::<Int64Array>()),
        "int32" => Arc::new((0..n).map(|i| (!null(i)).then(|| r.random_range(0i32..100))).collect::<Int32Array>()),
        "float64" => Arc::new((0..n).map(|i| (!null(i)).then(|| r.random_range(-1e6f64..1e6))).collect::<Float64Array>()),
        "bool" => Arc::new((0..n).map(|i| (!null(i)).then(|| r.random_bool(0.5))).collect::<BooleanArray>()),
        "date32" => Arc::new((0..n).map(|i| (!null(i)).then(|| r.random_range(-20_000i32..20_000))).collect::<Date32Array>()),
        "timestamp" => Arc::new(
            (0..n).map(|i| (!null(i)).then(|| r.random_range(0i64..2_000_000_000_000_000))).collect::<TimestampMicrosecondArray>(),
        ),
        "decimal128" => Arc::new(
            (0..n)
                .map(|i| (!null(i)).then(|| r.random_range(-10i128.pow(18)..10i128.pow(18))))
                .collect::<Decimal128Array>()
                .with_precision_and_scale(20, 3)
                .unwrap(),
        ),
        "utf8" | "large_utf8" | "utf8_view" | "binary" => {
            let words: Vec<Option<String>> = (0..n)
                .map(|i| {
                    (!null(i)).then(|| {
                        let len = if r.random_bool(0.3) { 13 + r.random_range(0..30) } else { r.random_range(0..13) };
                        (0..len).map(|_| (b'a' + r.random_range(0..26u8)) as char).collect()
                    })
                })
                .collect();
            let refs: Vec<Option<&str>> = words.iter().map(|w| w.as_deref()).collect();
            match ty {
                "utf8" => Arc::new(StringArray::from(refs)),
                "large_utf8" => Arc::new(LargeStringArray::from(refs)),
                "utf8_view" => Arc::new(StringViewArray::from(refs)),
                _ => Arc::new(BinaryArray::from(refs.iter().map(|w| w.map(|s| s.as_bytes())).collect::<Vec<_>>())),
            }
        }
        _ => unreachable!("{ty}"),
    }
}

const TYPES: &[&str] =
    &["int64", "int32", "float64", "bool", "date32", "timestamp", "decimal128", "utf8", "large_utf8", "utf8_view", "binary"];

/// Chunks of `sizes` rows cut from fresh columns, each a slice at a random offset of a longer array,
/// with every fourth chunk all null and every third without nulls (so without validity).
fn chunks(ty: &str, sizes: &[usize], seed: u64) -> Vec<ArrayRef> {
    let mut r = common::rng(seed ^ 0x5eed);
    sizes
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            let pad = if r.random_bool(0.3) { 0 } else { r.random_range(0..70) };
            let null_every = match i % 4 {
                3 => 1,
                1 => 0,
                _ => 5,
            };
            let base = column(ty, pad + n + 3, null_every, seed.wrapping_add(i as u64));
            base.slice(pad, n)
        })
        .collect()
}

fn check(ty: &str, parts: &[ArrayRef], what: &str) -> Array {
    let refs: Vec<&dyn arrow::array::Array> = parts.iter().map(|a| a.as_ref()).collect();
    let got = Array::from_arrow_chunks(&refs).unwrap_or_else(|e| panic!("{ty} {what}: {e}"));
    let concat = arrow::compute::concat(&refs).unwrap();
    let want = Array::from_arrow(concat.as_ref()).unwrap();
    assert_eq!(got.len(), concat.len(), "{ty} {what}: length");
    assert_eq!(got.null_count(), concat.null_count(), "{ty} {what}: null count");
    assert_eq!(got.format(), want.format(), "{ty} {what}: format");
    let (g, w) = (got.to_arrow().unwrap(), want.to_arrow().unwrap());
    assert_eq!(g.to_data(), w.to_data(), "{ty} {what}: values against the concatenation's import");
    // large_utf8 comes back as utf8 (ArrowMetal narrows the offsets on import, both paths alike).
    let concat = arrow::compute::cast(&concat, g.data_type()).unwrap();
    assert_eq!(g.to_data(), concat.to_data(), "{ty} {what}: values against the concatenation");
    got
}

#[test]
fn every_type_over_chunk_layouts() {
    let layouts: &[(&str, Vec<usize>)] = &[
        ("one chunk", vec![300]),
        ("empty and one-row chunks", vec![0, 1, 7, 0, 64, 1, 100, 13, 0, 3, 33, 1]),
        ("two chunks", vec![37, 91]),
        ("one non-empty among empty", vec![0, 57, 0]),
        ("only empty chunks", vec![0, 0]),
    ];
    for (k, ty) in TYPES.iter().enumerate() {
        for (j, (what, sizes)) in layouts.iter().enumerate() {
            check(ty, &chunks(ty, sizes, (k * 31 + j) as u64), what);
        }
    }
}

#[test]
fn ten_thousand_chunks() {
    let mut r = common::rng(99);
    let sizes: Vec<usize> = (0..10_000).map(|_| r.random_range(0..4)).collect();
    for ty in ["int64", "bool", "utf8", "utf8_view", "decimal128"] {
        check(ty, &chunks(ty, &sizes, 7), "10,000 chunks");
    }
}

/// One copy thread, several and the default policy import the same column. The setting is
/// process-wide, so the other tests may run under any of these counts; their results do not change.
#[test]
fn thread_counts_give_the_same_array() {
    let saved = arrowmetal::import_threads();
    let mut r = common::rng(5);
    let sizes: Vec<usize> = (0..90).map(|_| r.random_range(1..25_000)).collect();
    for ty in ["int64", "bool", "utf8", "utf8_view"] {
        let parts = chunks(ty, &sizes, 3);
        let refs: Vec<&dyn arrow::array::Array> = parts.iter().map(|a| a.as_ref()).collect();
        let mut first = None;
        for t in [1, 2, 5, 16, 0] {
            arrowmetal::set_import_threads(t);
            assert_eq!(arrowmetal::import_threads(), t);
            let got = Array::from_arrow_chunks(&refs).unwrap().to_arrow().unwrap().to_data();
            match &first {
                None => first = Some(got),
                Some(f) => assert_eq!(&got, f, "{ty}: {t} threads against one"),
            }
        }
        check(ty, &parts, "90 chunks");
    }
    arrowmetal::set_import_threads(saved);
}

#[test]
fn nested_types_fall_back_to_concat() {
    let a: ArrayRef = Arc::new(arrow::array::ListArray::from_iter_primitive::<arrow::datatypes::Int64Type, _, _>(vec![
        Some(vec![Some(1i64), None]),
        None,
    ]));
    let b: ArrayRef = Arc::new(arrow::array::ListArray::from_iter_primitive::<arrow::datatypes::Int64Type, _, _>(vec![
        Some(vec![Some(3i64)]),
    ]));
    let got = Array::from_arrow_chunks(&[a.as_ref(), b.as_ref()]).unwrap();
    let want = arrow::compute::concat(&[a.as_ref(), b.as_ref()]).unwrap();
    assert_eq!(got.to_arrow().unwrap().to_data(), want.to_data());
}

#[test]
fn mismatched_types_and_no_chunks_are_errors() {
    let a = Int64Array::from(vec![1i64]);
    let b = Int32Array::from(vec![1i32]);
    assert!(Array::from_arrow_chunks(&[&a, &b]).is_err());
    assert!(Array::from_arrow_chunks(&[]).is_err());
}

#[test]
fn kernels_on_chunked_arrays() {
    let n = 200_000;
    let mut r = common::rng(5);
    let mut sizes = Vec::new();
    let mut left = n;
    while left > 0 {
        let k = left.min(1 + r.random_range(0..8192));
        sizes.push(k);
        left -= k;
    }
    let cut = |a: &ArrayRef| -> Vec<ArrayRef> {
        let mut at = 0;
        sizes
            .iter()
            .map(|&k| {
                let s = a.slice(at, k);
                at += k;
                s
            })
            .collect()
    };
    let ints = column("int64", n, 13, 1);
    let dbls = column("float64", n, 17, 2);
    let keys = column("int32", n, 0, 3);
    let strs = column("utf8_view", n, 19, 4);
    let (ci, cd, ck, cs) = (check("int64", &cut(&ints), "k"), check("float64", &cut(&dbls), "k"), check("int32", &cut(&keys), "k"), check("utf8_view", &cut(&strs), "k"));
    let (wi, wd, wk, ws) =
        (Array::from_arrow(ints.as_ref()).unwrap(), Array::from_arrow(dbls.as_ref()).unwrap(), Array::from_arrow(keys.as_ref()).unwrap(), Array::from_arrow(strs.as_ref()).unwrap());
    assert_eq!(format!("{:?}", ci.sum().unwrap()), format!("{:?}", wi.sum().unwrap()));
    assert_eq!(format!("{:?}", cd.sum().unwrap()), format!("{:?}", wd.sum().unwrap()));
    assert_eq!(ci.sort(false).unwrap().to_arrow().unwrap().to_data(), wi.sort(false).unwrap().to_arrow().unwrap().to_data());
    assert_eq!(cd.argsort(true).unwrap().to_arrow().unwrap().to_data(), wd.argsort(true).unwrap().to_arrow().unwrap().to_data());
    assert_eq!(cs.argsort(false).unwrap().to_arrow().unwrap().to_data(), ws.argsort(false).unwrap().to_arrow().unwrap().to_data());
    let (gc, gw) = (arrowmetal::group_by(&[&ck]).unwrap(), arrowmetal::group_by(&[&wk]).unwrap());
    assert_eq!(gc.group_count(), gw.group_count());
    assert_eq!(gc.sum(&ci).unwrap().to_arrow().unwrap().to_data(), gw.sum(&wi).unwrap().to_arrow().unwrap().to_data());
}

#[test]
fn source_from_batches() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("s", DataType::Utf8, true),
    ]));
    let n = 50_000;
    let (k, s) = (column("int64", n, 7, 11), column("utf8", n, 9, 12));
    let mut batches = Vec::new();
    let mut at = 0;
    while at < n {
        let len = (n - at).min(8192);
        batches.push(RecordBatch::try_new(schema.clone(), vec![k.slice(at, len), s.slice(at, len)]).unwrap());
        at += len;
    }
    let src = Source::from_batches("t", &batches).unwrap();
    let plan = r#"{"op":"scan","source":"t"}"#;
    let out = arrowmetal::run_plan(plan, &[&src], false).unwrap();
    assert_eq!(out.row_count(), n);
    assert_eq!(out.column(0).unwrap().to_arrow().unwrap().to_data(), k.to_data());
    assert_eq!(out.column(1).unwrap().to_arrow().unwrap().to_data(), s.to_data());
    assert!(Source::from_batches("t", &[]).is_err());
}
