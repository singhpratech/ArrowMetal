//! Differential grid: every query runs twice, once in a plain DataFusion session and once with
//! `ArrowMetalRule` (min_rows 0, so every supported node is taken), and the answers must match.
//!
//! * ORDER BY: the sort-key sequence must match exactly; rows are compared as multisets within
//!   each run of equal keys (SQL leaves the order of ties unspecified). With LIMIT, the last tie
//!   run is excluded from the row comparison, since which of its rows make the cut is unspecified.
//! * GROUP BY and WHERE: order-insensitive (both sides sorted by the full row).
//! * Values compare exactly — floats by bit pattern, so -0.0 != +0.0 and NaNs of different sign
//!   or payload differ — except `sum` over a float column and every `avg`, which compare within
//!   |a - b| <= 1e-9 * max(1, |a|, |b|) (floating-point addition is not associative; the two
//!   engines add in different orders), any NaN equal to any NaN there (a NaN sum's payload is the
//!   adder's choice).
//! * A GROUP BY runs three times with the rule, once per run-time choice of a replaced aggregate:
//!   forced onto ArrowMetal (`AggregateChoice::ArrowMetal`), forced back to DataFusion
//!   (`AggregateChoice::DataFusion`: the hand-back over the collected batches), and the default
//!   (`Measured`: the group-count probe and the measured table, looked up at 50,000,000 rows with
//!   `table_rows` so the table decides at these small sizes). Sorts and filters run once (forced
//!   onto ArrowMetal as before).
//! * The float columns hold both zero signs, ±inf and NaNs: `kf` one NaN bit pattern (so a
//!   GROUP BY over it runs on the GPU), `kn` (Float64) and `kf32` / `vf32` (Float32) negative NaNs
//!   and NaN payloads, `vf` positive NaNs with payloads.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray};
use arrow::datatypes::{DataType, Field, Float32Type, Float64Type, Int32Type, Int64Type, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_arrowmetal::{session_context, AggregateChoice, ArrowMetalConfig, ArrowMetalRule};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// -------------------------------------------------------------------------------------------------
// Data
// -------------------------------------------------------------------------------------------------

const FLOAT_KEYS: [f64; 8] = [-1.5, -0.0, 0.0, 2.25, f64::NAN, 7.0, f64::INFINITY, f64::NEG_INFINITY];
/// NaN-heavy Float64 keys: negative NaN, NaN payloads of both signs, ±0.0, ±inf.
fn nan_keys() -> [f64; 9] {
    [
        f64::NAN,
        f64::from_bits(0xFFF8_0000_0000_0000),
        f64::from_bits(0x7FF0_0000_0000_0001),
        f64::from_bits(0xFFF4_0000_0000_0003),
        -0.0,
        0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -3.5,
    ]
}
/// Float32 keys: one NaN bit pattern, ±0.0, ±inf.
const F32_KEYS: [f32; 8] = [-1.5, -0.0, 0.0, 2.25, f32::NAN, 7.0, f32::INFINITY, f32::NEG_INFINITY];
const STR_KEYS: [&str; 7] = ["a", "b", "B", "ab", "", "\u{fc}", "zz"];

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k32", DataType::Int32, true),
        Field::new("k64", DataType::Int64, true),
        Field::new("kf", DataType::Float64, true),
        Field::new("ks", DataType::Utf8, true),
        Field::new("v32", DataType::Int32, true),
        Field::new("v64", DataType::Int64, true),
        Field::new("vf", DataType::Float64, true),
        Field::new("vg", DataType::Float64, true),
        Field::new("kn", DataType::Float64, true),
        Field::new("kf32", DataType::Float32, true),
        Field::new("vf32", DataType::Float32, true),
        Field::new("ksv", DataType::Utf8View, true),
        Field::new("ksl", DataType::LargeUtf8, true),
    ]))
}

fn table(n: usize, null_frac: f64, seed: u64) -> RecordBatch {
    let mut r = StdRng::seed_from_u64(seed);
    let null = |r: &mut StdRng| null_frac >= 1.0 || (null_frac > 0.0 && r.random_bool(null_frac));
    let k64_domain: Vec<i64> = (0..50).map(|i| (i as i64 - 25) * 40_000_000_007).collect();
    let mut k32 = Vec::new();
    let mut k64 = Vec::new();
    let mut kf = Vec::new();
    let mut ks = Vec::new();
    let mut v32 = Vec::new();
    let mut v64 = Vec::new();
    let mut vf = Vec::new();
    let mut vg = Vec::new();
    let mut kn = Vec::new();
    let mut kf32 = Vec::new();
    let mut vf32 = Vec::new();
    let nk = nan_keys();
    for _ in 0..n {
        k32.push(if null(&mut r) { None } else { Some(r.random_range(-3i32..=3)) });
        k64.push(if null(&mut r) { None } else { Some(k64_domain[r.random_range(0..50)]) });
        kf.push(if null(&mut r) { None } else { Some(FLOAT_KEYS[r.random_range(0..FLOAT_KEYS.len())]) });
        ks.push(if null(&mut r) { None } else { Some(STR_KEYS[r.random_range(0..STR_KEYS.len())]) });
        v32.push(if null(&mut r) { None } else { Some(r.random_range(-1000i32..=1000)) });
        v64.push(if null(&mut r) { None } else { Some(r.random_range(-1_000_000_000i64..=1_000_000_000)) });
        vf.push(if null(&mut r) {
            None
        } else {
            Some(match r.random_range(0..40) {
                0 | 1 => f64::NAN,
                2 => f64::from_bits(0x7FF0_0000_0000_0005),
                3 | 4 => -0.0,
                5 | 6 => 0.0,
                _ => (r.random_range(-4000i32..=4000) as f64) * 0.25,
            })
        });
        // No NaN; values not exactly representable, so float sums round; a rare -0.0 (never +0.0).
        vg.push(if null(&mut r) {
            None
        } else if r.random_range(0..100) == 0 {
            Some(-0.0)
        } else {
            Some(r.random_range(-1000.0f64..1000.0) / 3.0)
        });
        kn.push(if null(&mut r) { None } else { Some(nk[r.random_range(0..nk.len())]) });
        kf32.push(if null(&mut r) { None } else { Some(F32_KEYS[r.random_range(0..F32_KEYS.len())]) });
        vf32.push(if null(&mut r) {
            None
        } else {
            Some(match r.random_range(0..40) {
                0 => f32::NAN,
                1 => f32::from_bits(0xFFC0_0000),
                2 => f32::from_bits(0x7F80_0007),
                3 => -0.0,
                4 => 0.0,
                5 => f32::NEG_INFINITY,
                _ => (r.random_range(-4000i32..=4000) as f32) * 0.25,
            })
        });
    }
    let cols: Vec<ArrayRef> = vec![
        Arc::new(arrow::array::Int32Array::from(k32)),
        Arc::new(arrow::array::Int64Array::from(k64)),
        Arc::new(arrow::array::Float64Array::from(kf)),
        Arc::new(arrow::array::StringArray::from(ks)),
        Arc::new(arrow::array::Int32Array::from(v32)),
        Arc::new(arrow::array::Int64Array::from(v64)),
        Arc::new(arrow::array::Float64Array::from(vf)),
        Arc::new(arrow::array::Float64Array::from(vg)),
        Arc::new(arrow::array::Float64Array::from(kn)),
        Arc::new(arrow::array::Float32Array::from(kf32)),
        Arc::new(arrow::array::Float32Array::from(vf32)),
    ];
    // `ks` again as Utf8View (DataFusion's Parquet reader's string type) and LargeUtf8.
    let mut cols = cols;
    let ks = std::sync::Arc::clone(&cols[3]);
    cols.push(arrow::compute::cast(&ks, &DataType::Utf8View).unwrap());
    cols.push(arrow::compute::cast(&ks, &DataType::LargeUtf8).unwrap());
    RecordBatch::try_new(schema(), cols).unwrap()
}

fn mem_table(b: &RecordBatch, parts: usize) -> Arc<MemTable> {
    let n = b.num_rows();
    let mut partitions = Vec::new();
    let step = n.div_ceil(parts.max(1)).max(1);
    let mut off = 0;
    for _ in 0..parts {
        let len = step.min(n - off.min(n));
        partitions.push(vec![b.slice(off.min(n), len)]);
        off += len;
    }
    Arc::new(MemTable::try_new(b.schema(), partitions).unwrap())
}

// -------------------------------------------------------------------------------------------------
// Values
// -------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum V {
    Null,
    I(i64),
    /// Float64 by bit pattern.
    F(u64),
    /// Float32 by bit pattern.
    F32(u32),
    S(String),
    B(bool),
}

fn f_of(v: &V) -> Option<f64> {
    match v {
        V::F(b) => Some(f64::from_bits(*b)),
        V::F32(b) => Some(f32::from_bits(*b) as f64),
        V::I(i) => Some(*i as f64),
        _ => None,
    }
}

fn column_values(a: &ArrayRef) -> Vec<V> {
    (0..a.len())
        .map(|i| {
            if a.is_null(i) {
                return V::Null;
            }
            match a.data_type() {
                DataType::Int32 => V::I(a.as_primitive::<Int32Type>().value(i) as i64),
                DataType::Int64 => V::I(a.as_primitive::<Int64Type>().value(i)),
                DataType::Float64 => V::F(a.as_primitive::<Float64Type>().value(i).to_bits()),
                DataType::Float32 => V::F32(a.as_primitive::<Float32Type>().value(i).to_bits()),
                DataType::Utf8 => V::S(a.as_string::<i32>().value(i).to_string()),
                DataType::Utf8View => V::S(a.as_string_view().value(i).to_string()),
                DataType::LargeUtf8 => V::S(a.as_string::<i64>().value(i).to_string()),
                DataType::Boolean => V::B(a.as_boolean().value(i)),
                t => panic!("value type {t} not handled by the grid"),
            }
        })
        .collect()
}

fn rows(batches: &[RecordBatch]) -> Vec<Vec<V>> {
    let mut out = Vec::new();
    for b in batches {
        let cols: Vec<Vec<V>> = b.columns().iter().map(column_values).collect();
        for i in 0..b.num_rows() {
            out.push(cols.iter().map(|c| c[i].clone()).collect());
        }
    }
    out
}

fn close(a: &V, b: &V) -> bool {
    match (f_of(a), f_of(b)) {
        (Some(x), Some(y)) => {
            if x.is_nan() || y.is_nan() {
                return x.is_nan() && y.is_nan();
            }
            // An infinite sum or mean must be the same infinity (inf - inf is NaN).
            if x.is_infinite() || y.is_infinite() {
                return x == y;
            }
            (x - y).abs() <= 1e-9 * 1f64.max(x.abs()).max(y.abs())
        }
        _ => a == b,
    }
}

fn rows_equal(a: &[Vec<V>], b: &[Vec<V>], tol_cols: &[usize]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.len() == y.len()
                && x.iter().zip(y).enumerate().all(|(i, (p, q))| if tol_cols.contains(&i) { close(p, q) } else { p == q })
        })
}

// -------------------------------------------------------------------------------------------------
// Queries
// -------------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Kind {
    /// ORDER BY with the key columns' positions in the SELECT list; `limited` when there is a LIMIT.
    Ordered { key_cols: Vec<usize>, limited: bool },
    /// Order-insensitive; `tol_cols` compared with the float tolerance.
    Unordered { tol_cols: Vec<usize> },
}

struct Query {
    class: String,
    sql: String,
    kind: Kind,
    /// (sum, min, max) column positions of a Float64 min/max, whose DataFusion answer depends on
    /// row arrival order when a group holds a NaN or both zero signs (see `relax_order_dependent`).
    float_min_max: Option<(usize, usize, usize)>,
}

/// DataFusion's grouped Float64 MIN/MAX is order-dependent for a group with a NaN (the running
/// value is replaced on every NaN comparison) and for the sign of a zero result (the first zero
/// seen wins). With several partitions the arrival order is not fixed, so two runs of the same
/// plain DataFusion query can differ there. For multi-partition layouts those cells are compared
/// with NaN groups blanked and zero signs ignored; returns how many cells were relaxed.
fn relax_order_dependent(r: &mut [Vec<V>], (sum, min, max): (usize, usize, usize)) -> usize {
    let mut n = 0;
    for row in r.iter_mut() {
        let nan_group = matches!(f_of(&row[sum]), Some(x) if x.is_nan());
        for c in [min, max] {
            if nan_group {
                row[c] = V::Null;
                n += 1;
            } else if matches!(f_of(&row[c]), Some(x) if x == 0.0) {
                row[c] = match row[c] {
                    V::F32(_) => V::F32(0),
                    _ => V::F(0),
                };
                n += 1;
            }
        }
    }
    n
}

fn queries() -> Vec<Query> {
    let mut q = Vec::new();
    let cols = ["k32", "k64", "kf", "ks", "v32", "v64", "vf", "vg", "kn", "kf32", "vf32", "ksv", "ksl"];
    let pos = |c: &str| cols.iter().position(|x| *x == c).unwrap();
    // Single-column ORDER BY, every direction x null placement, with and without LIMIT.
    for c in ["k32", "k64", "kf", "kn", "kf32", "ks", "ksv", "ksl", "v64", "vf", "vf32", "vg"] {
        for dir in ["ASC", "DESC"] {
            for nulls in ["NULLS FIRST", "NULLS LAST", ""] {
                for limit in [None, Some(7)] {
                    let lim = limit.map(|n| format!(" LIMIT {n}")).unwrap_or_default();
                    q.push(Query {
                        class: format!("order by {c} {dir} {nulls}{}", if limit.is_some() { " limit" } else { "" }),
                        sql: format!("SELECT * FROM t ORDER BY {c} {dir} {nulls}{lim}"),
                        kind: Kind::Ordered { key_cols: vec![pos(c)], limited: limit.is_some() },
                        float_min_max: None,
                    });
                }
            }
        }
    }
    // Multi-column ORDER BY.
    for (keys, lim) in [
        ("ks ASC, vf DESC NULLS FIRST, k64", ""),
        ("kf DESC, k32 ASC NULLS FIRST", ""),
        ("k32 DESC NULLS LAST, ks DESC, v64", " LIMIT 11"),
        ("vf, kf DESC NULLS LAST", " LIMIT 5"),
        ("kn DESC NULLS FIRST, kf32 ASC", " LIMIT 9"),
        ("kf32 DESC NULLS LAST, kn", ""),
        ("vf32 ASC NULLS FIRST, k64 DESC", ""),
        ("kn, vf DESC NULLS FIRST", " LIMIT 40"),
        ("ksv DESC, k32 NULLS FIRST, v64", ""),
        ("ksl, kf DESC", " LIMIT 13"),
    ] {
        let key_cols = keys.split(',').map(|k| pos(k.split_whitespace().next().unwrap())).collect();
        q.push(Query {
            class: format!("order by {keys}{}", if lim.is_empty() { "" } else { " limit" }),
            sql: format!("SELECT * FROM t ORDER BY {keys}{lim}"),
            kind: Kind::Ordered { key_cols, limited: !lim.is_empty() },
            float_min_max: None,
        });
    }
    // GROUP BY with every aggregate, per key and value column.
    for key in ["k32", "k64", "kf", "ks", "k32, ks", "kf, k64", "kn", "kf32", "kf32, k32"] {
        let nk = key.split(',').count();
        for v in ["v32", "v64", "vf", "vg", "vf32"] {
            let float = v == "vf" || v == "vg" || v == "vf32";
            // sum over a float column and every avg take the tolerance.
            let mut tol = vec![nk + 5];
            if float {
                tol.push(nk);
            }
            q.push(Query {
                class: format!("group by {key} aggs({v})"),
                sql: format!(
                    "SELECT {key}, sum({v}), min({v}), max({v}), count({v}), count(*), avg({v}) FROM t GROUP BY {key}"
                ),
                kind: Kind::Unordered { tol_cols: tol },
                float_min_max: if float { Some((nk, nk + 1, nk + 2)) } else { None },
            });
        }
    }
    // count over a string column, alone and next to Float64 aggregates.
    for sql in [
        "SELECT k32, count(ks) FROM t GROUP BY k32",
        "SELECT k32, count(ks), count(vf), sum(vf) FROM t GROUP BY k32",
        "SELECT ks, count(ks), count(k64), max(v64) FROM t GROUP BY ks",
        "SELECT k32, count(ksv), count(ksl) FROM t GROUP BY k32",
    ] {
        q.push(Query {
            class: sql.to_string(),
            sql: sql.to_string(),
            kind: Kind::Unordered { tol_cols: if sql.contains("sum(vf)") { vec![3] } else { vec![] } },
            float_min_max: None,
        });
    }
    // WHERE.
    for pred in [
        "v32 > 5",
        "v64 <= 0 AND k32 = 2",
        "vf > 1.5",
        "vf <= -2.5 OR ks = 'b'",
        "k64 IS NULL",
        "vf IS NOT NULL AND NOT (k32 = 3)",
        "ks <> 'a'",
        "vf >= 100.25 AND v64 < 0",
        "vf = 0.0",
        "vf < 1.5",
        "vg > 1.5",
        "vg <= -2.5 OR k32 = 1",
        "vf32 > 1.5",
        "vf32 < -2.5 AND k64 IS NOT NULL",
        "vf32 > CAST(1.5 AS REAL)",
        "vf32 <= CAST(-2.5 AS REAL) OR k32 = 1",
        "kn > 1.5",
        "kn <= -2.5",
    ] {
        q.push(Query {
            class: format!("where {pred}"),
            sql: format!("SELECT * FROM t WHERE {pred}"),
            kind: Kind::Unordered { tol_cols: vec![] },
            float_min_max: None,
        });
    }
    q
}

// -------------------------------------------------------------------------------------------------
// Comparison
// -------------------------------------------------------------------------------------------------

/// Rows grouped into runs of equal keys (as `want` orders them); each run sorted by full row.
fn tie_runs(r: &[Vec<V>], key_cols: &[usize]) -> Vec<Vec<Vec<V>>> {
    let mut runs: Vec<Vec<Vec<V>>> = Vec::new();
    let mut last: Option<Vec<V>> = None;
    for row in r {
        let k: Vec<V> = key_cols.iter().map(|&i| row[i].clone()).collect();
        if last.as_ref() != Some(&k) {
            runs.push(Vec::new());
            last = Some(k);
        }
        runs.last_mut().unwrap().push(row.clone());
    }
    for run in &mut runs {
        run.sort();
    }
    runs
}

fn compare(kind: &Kind, want: &[Vec<V>], got: &[Vec<V>]) -> Result<(), String> {
    if want.len() != got.len() {
        return Err(format!("row count: want {}, got {}", want.len(), got.len()));
    }
    match kind {
        Kind::Ordered { key_cols, limited } => {
            let keys = |r: &[Vec<V>]| -> Vec<Vec<V>> {
                r.iter().map(|row| key_cols.iter().map(|&i| row[i].clone()).collect()).collect()
            };
            let (kw, kg) = (keys(want), keys(got));
            if kw != kg {
                let at = kw.iter().zip(&kg).position(|(a, b)| a != b).unwrap_or(0);
                return Err(format!("key order differs at row {at}: want {:?}, got {:?}", kw[at], kg[at]));
            }
            let (mut rw, mut rg) = (tie_runs(want, key_cols), tie_runs(got, key_cols));
            if *limited {
                rw.pop();
                rg.pop();
            }
            if rw != rg {
                return Err("rows within a run of equal keys differ".into());
            }
            Ok(())
        }
        Kind::Unordered { tol_cols } => {
            let (mut w, mut g) = (want.to_vec(), got.to_vec());
            w.sort();
            g.sort();
            if !rows_equal(&w, &g, tol_cols) {
                let at = w
                    .iter()
                    .zip(&g)
                    .position(|(a, b)| !rows_equal(std::slice::from_ref(a), std::slice::from_ref(b), tol_cols))
                    .unwrap_or(0);
                let cols: Vec<usize> = (0..w[at].len())
                    .filter(|&i| if tol_cols.contains(&i) { !close(&w[at][i], &g[at][i]) } else { w[at][i] != g[at][i] })
                    .collect();
                return Err(format!(
                    "row {at} of {} (sorted), columns {cols:?} differ: want {:?}, got {:?}",
                    w.len(),
                    w[at],
                    g[at]
                ));
            }
            Ok(())
        }
    }
}

fn max_float_dev(want: &[Vec<V>], got: &[Vec<V>], tol_cols: &[usize]) -> f64 {
    let (mut w, mut g) = (want.to_vec(), got.to_vec());
    w.sort();
    g.sort();
    let mut m = 0f64;
    for (a, b) in w.iter().zip(&g) {
        for &c in tol_cols {
            if let (Some(x), Some(y)) = (f_of(&a[c]), f_of(&b[c])) {
                if x.is_finite() && y.is_finite() {
                    m = m.max((x - y).abs() / 1f64.max(x.abs()).max(y.abs()));
                }
            }
        }
    }
    m
}

// -------------------------------------------------------------------------------------------------
// The grid
// -------------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn differential_grid() {
    let sizes = [0usize, 1, 1000, 20_000];
    let null_fracs = [0.0, 0.1, 1.0];
    // (MemTable partitions, target_partitions): the single-partition plan (Single aggregates, a
    // plain SortExec) and the multi-partition one (Partial/FinalPartitioned pairs, sort + merge).
    let layouts = [(1usize, 1usize), (3, 4)];
    let queries = queries();

    let mut pairs = 0usize;
    let mut pairs_taken = 0usize;
    let mut nodes_taken = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let mut fallbacks: Vec<String> = Vec::new();
    let mut left_reasons: BTreeMap<String, usize> = BTreeMap::new();
    let mut per_class_fail: BTreeMap<String, usize> = BTreeMap::new();
    let mut max_dev = 0f64;
    let mut data_dependent: BTreeMap<String, usize> = BTreeMap::new();
    let mut relaxed_cells = 0usize;

    // Per variant: (pairs, pairs with a node on ArrowMetal, run-time choices on ArrowMetal, handed back).
    let mut per_variant: BTreeMap<&str, [usize; 4]> = BTreeMap::new();
    let variants = [
        ("take", AggregateChoice::ArrowMetal, None),
        ("back", AggregateChoice::DataFusion, None),
        ("def", AggregateChoice::Measured, Some(50_000_000usize)),
    ];

    let mut seed = 1u64;
    for &n in &sizes {
        for &nf in &null_fracs {
            for &(parts, tp) in &layouts {
                seed += 1;
                let batch = table(n, nf, seed);
                let cfg = || SessionConfig::new().with_target_partitions(tp).with_batch_size(1024);
                let plain = SessionContext::new_with_config(cfg());
                plain.register_table("t", mem_table(&batch, parts)).unwrap();
                let mut wants = Vec::with_capacity(queries.len());
                for q in &queries {
                    wants.push(plain.sql(&q.sql).await.unwrap().collect().await.unwrap());
                }

                for &(variant, choice, table_rows) in &variants {
                    let rule = ArrowMetalRule::new(ArrowMetalConfig::all().with_min_rows(0).with_aggregate_choice(choice).with_table_rows(table_rows));
                    let metal = session_context(cfg(), rule.clone());
                    metal.register_table("t", mem_table(&batch, parts)).unwrap();
                    for (q, want) in queries.iter().zip(&wants) {
                        let aggregate = q.sql.contains("GROUP BY");
                        if variant != "take" && !aggregate {
                            continue;
                        }
                        pairs += 1;
                        let pv = per_variant.entry(variant).or_default();
                        pv[0] += 1;
                        let label = format!("[{variant} n={n} nulls={nf} parts={parts} tp={tp}] {}", q.sql);
                        rule.clear_report();
                        let got = match metal.sql(&q.sql).await.unwrap().collect().await {
                            Ok(b) => b,
                            Err(e) => {
                                failures.push(format!("{label}: error with the rule: {e}"));
                                *per_class_fail.entry(q.class.clone()).or_default() += 1;
                                continue;
                            }
                        };
                        let report = rule.report();
                        let taken = report.taken().count();
                        nodes_taken += taken;
                        let on_gpu = report.runtime_choices().filter(|d| d.taken).count();
                        let back = report.runtime_choices().filter(|d| !d.taken).count();
                        pv[2] += on_gpu;
                        pv[3] += back;
                        if taken > 0 {
                            pairs_taken += 1;
                            pv[1] += 1;
                        }
                        if choice == AggregateChoice::ArrowMetal && back > 0 {
                            failures.push(format!("{label}: a forced ArrowMetal aggregate was handed back"));
                        }
                        if choice == AggregateChoice::DataFusion && on_gpu > 0 {
                            failures.push(format!("{label}: a forced hand-back ran on ArrowMetal"));
                        }
                        for d in report.left() {
                            let reason = d.reason.split(" (exact)").next().unwrap().to_string();
                            let key = format!("{variant} {} :: {}", d.node.split(':').next().unwrap(), reason);
                            *left_reasons.entry(key).or_default() += 1;
                        }
                        for d in report.runtime_fallbacks() {
                            if d.is_data_dependent() {
                                *data_dependent.entry(d.reason.clone()).or_default() += 1;
                            } else {
                                fallbacks.push(format!("{label}: {}", d.reason));
                            }
                        }
                        for d in report.runtime_choices() {
                            if variant == "def" {
                                let est = d.groups.as_ref().and_then(|g| g.estimate).map(|e| e.estimate);
                                let key = format!(
                                    "def run-time choice: {} (estimate {})",
                                    if d.taken { "ArrowMetal" } else { "handed back" },
                                    est.map(|e| if e <= 1414 { "in the 200 bucket".to_string() } else { format!("{e}") }).unwrap_or_default()
                                );
                                *left_reasons.entry(key).or_default() += 1;
                            }
                        }
                        let (mut w, mut g) = (rows(want), rows(&got));
                        if let (Some(cols), true) = (q.float_min_max, tp > 1) {
                            relaxed_cells += relax_order_dependent(&mut w, cols);
                            relax_order_dependent(&mut g, cols);
                        }
                        if let Kind::Unordered { tol_cols } = &q.kind {
                            if w.len() == g.len() {
                                max_dev = max_dev.max(max_float_dev(&w, &g, tol_cols));
                            }
                        }
                        if let Err(msg) = compare(&q.kind, &w, &g) {
                            failures.push(format!("{label}: {msg}"));
                            *per_class_fail.entry(q.class.clone()).or_default() += 1;
                        }
                    }
                }
            }
        }
    }

    println!("grid: {pairs} query pairs, {pairs_taken} with at least one node replaced, {nodes_taken} nodes replaced");
    for (v, [p, t, g, b]) in &per_variant {
        println!("  variant {v}: {p} pairs, {t} with a node replaced, run-time choices: {g} on ArrowMetal, {b} handed back");
    }
    println!("largest relative deviation in a toleranced float column: {max_dev:e}");
    println!("left (node :: reason -> count):");
    for (k, v) in &left_reasons {
        println!("  {v:5}  {k}");
    }
    println!("data-dependent fallbacks to DataFusion (reason -> count):");
    for (k, v) in &data_dependent {
        println!("  {v:5}  {k}");
    }
    println!("order-dependent min/max(vf) cells relaxed in multi-partition layouts: {relaxed_cells}");
    println!("runtime fallbacks on an ArrowMetal error: {}", fallbacks.len());
    for f in fallbacks.iter().take(20) {
        println!("  {f}");
    }
    println!("mismatches: {}", failures.len());
    for (c, k) in &per_class_fail {
        println!("  {k:4}  {c}");
    }
    for f in failures.iter().take(40) {
        println!("  {f}");
    }
    assert!(fallbacks.is_empty(), "{} runtime fallbacks", fallbacks.len());
    assert!(failures.is_empty(), "{} mismatches", failures.len());
}
