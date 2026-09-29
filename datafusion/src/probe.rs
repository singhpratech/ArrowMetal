//! The group-count probe: how many groups a GROUP BY's keys hold, estimated on the CPU from a
//! sample of the collected input, before anything is imported to the GPU.
//!
//! A port of the Polars engine's probe (`python/arrowmetal/polars_engine.py`, "The group-count
//! probe"). The distinct key tuples of a fixed sample of n of the input's N rows are scaled to the
//! input by the bias-corrected Chao1 estimator (Chao, Biometrics 2005),
//!
//! ```text
//!     D = d + f1 (f1 - 1) / (2 (f2 + 1)),
//! ```
//!
//! where d is the number of distinct tuples in the sample and f1, f2 the number seen exactly once
//! and exactly twice, clipped to [d, N]. The sample is stratified: row `i * (N / n) + h(i)` for
//! `i < n`, `h` a fixed-seed splitmix64 hash of `i`, so the same input always gets the same sample
//! and the same estimate.
//!
//! The samples are 512, 2,048, 8,192, ... rows, at most 65,536 and a quarter of the input, and the
//! probe stops at the first whose range settles the decision: the range is D with f2 moved by two
//! of its Poisson standard deviations (and one or two more) either way, and the caller's
//! `settled(lo, hi)` says whether every group count in it gets the same answer. An input of at most
//! 4,096 rows is counted exactly.

use std::time::{Duration, Instant};

use arrow::array::{Array, AsArray};
use arrow::datatypes::{
    DataType, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type,
    UInt64Type, UInt8Type,
};
use arrow::record_batch::RecordBatch;

const FIRST: usize = 512;
const GROWTH: usize = 4;
const SAMPLE_MAX: usize = 65_536;
const EXACT: usize = 4_096;
const SEED: u64 = 0xA6_5EED;
const SETTLED_F2: u64 = 8;
/// What a null key hashes to (a null is a group of its own, as in SQL GROUP BY).
const NULL_KEY: u64 = 0x6E75_6C6C_6B65_7931;

/// One probe's answer.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct GroupEstimate {
    /// The Chao1 estimate (or the exact count), in groups.
    pub estimate: u64,
    /// The low end of its range (equal to `estimate` when counted exactly).
    pub low: u64,
    /// The high end of its range.
    pub high: u64,
    /// Rows in the last sample (the whole input when counted exactly).
    pub sample_rows: usize,
    /// Rows of the input (the whole input, also when the sample was drawn from a part of it).
    pub rows: usize,
    /// True when every row was counted (inputs of at most 4,096 rows).
    pub exact: bool,
    /// Wall time of the probe.
    pub time: Duration,
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn fnv1a(b: &[u8]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x0100_0000_01B3);
    }
    h
}

/// The sample's row positions: stratified, increasing, fixed for (rows, n).
fn positions(rows: usize, n: usize) -> Vec<usize> {
    let stride = (rows / n).max(1);
    (0..n)
        .map(|i| i * stride + (splitmix(i as u64 + SEED) % stride as u64) as usize)
        .filter(|&p| p < rows)
        .collect()
}

/// Folds the key values of rows `idx` of column `a` into `out` (one slot per row): each value
/// becomes a u64 that is equal for equal keys (DataFusion's grouping: -0.0 and +0.0 one group, each
/// NaN bit pattern its own, null its own), mixed, and combined with what `out` holds unless
/// `first`. One type dispatch per column and batch, not per value.
fn fold_column(a: &dyn Array, idx: &[usize], out: &mut [u64], first: bool) {
    fn put(out: &mut [u64], j: usize, v: u64, first: bool) {
        let v = splitmix(v);
        out[j] = if first { v } else { out[j].rotate_left(23).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ v };
    }
    let nulls = a.nulls();
    let valid = |i: usize| nulls.is_none_or(|n| n.is_valid(i));
    macro_rules! prim {
        ($t:ty, $f:expr) => {{
            let v = a.as_primitive::<$t>().values();
            for (j, &i) in idx.iter().enumerate() {
                put(out, j, if valid(i) { $f(v[i]) } else { NULL_KEY }, first);
            }
        }};
    }
    macro_rules! strs {
        ($arr:expr) => {{
            let s = $arr;
            for (j, &i) in idx.iter().enumerate() {
                put(out, j, if valid(i) { fnv1a(s.value(i).as_bytes()) } else { NULL_KEY }, first);
            }
        }};
    }
    match a.data_type() {
        DataType::Int8 => prim!(Int8Type, |x: i8| x as i64 as u64),
        DataType::Int16 => prim!(Int16Type, |x: i16| x as i64 as u64),
        DataType::Int32 => prim!(Int32Type, |x: i32| x as i64 as u64),
        DataType::Int64 => prim!(Int64Type, |x: i64| x as u64),
        DataType::UInt8 => prim!(UInt8Type, |x: u8| x as u64),
        DataType::UInt16 => prim!(UInt16Type, |x: u16| x as u64),
        DataType::UInt32 => prim!(UInt32Type, |x: u32| x as u64),
        DataType::UInt64 => prim!(UInt64Type, |x: u64| x),
        DataType::Float64 => prim!(Float64Type, |x: f64| (if x == 0.0 { 0.0 } else { x }).to_bits()),
        DataType::Float32 => prim!(Float32Type, |x: f32| (if x == 0.0 { 0.0f32 } else { x }).to_bits() as u64),
        DataType::Utf8 => strs!(a.as_string::<i32>()),
        DataType::LargeUtf8 => strs!(a.as_string::<i64>()),
        DataType::Utf8View => strs!(a.as_string_view()),
        // Other key types are not taken by the rule (the caller never gets here): only nulls
        // are told apart.
        _ => {
            for (j, &i) in idx.iter().enumerate() {
                put(out, j, if valid(i) { 0 } else { NULL_KEY }, first);
            }
        }
    }
}

/// One u64 per sampled row (`pos`: increasing positions over the batches in order), equal for
/// equal key tuples.
fn sample(batches: &[&RecordBatch], keys: &[usize], pos: &[usize]) -> Vec<u64> {
    let mut out = vec![0u64; pos.len()];
    let mut start = 0usize;
    let mut at = 0usize;
    let mut idx = Vec::new();
    for b in batches {
        let end = start + b.num_rows();
        idx.clear();
        let from = at;
        while at < pos.len() && pos[at] < end {
            idx.push(pos[at] - start);
            at += 1;
        }
        if !idx.is_empty() {
            for (j, &k) in keys.iter().enumerate() {
                fold_column(b.column(k).as_ref(), &idx, &mut out[from..at], j == 0);
            }
        }
        start = end;
        if at == pos.len() {
            break;
        }
    }
    out
}

/// (d, f1, f2) of a sample of key tuples: an open-addressing count (the values are already mixed
/// hashes), about a quarter of the time of sorting them.
fn stats(v: Vec<u64>) -> (u64, u64, u64) {
    let size = (v.len() * 2).next_power_of_two().max(16);
    let mask = size - 1;
    let mut keys = vec![0u64; size];
    let mut counts = vec![0u32; size];
    let mut d = 0u64;
    for x in v {
        let mut i = (x as usize) & mask;
        loop {
            if counts[i] == 0 {
                keys[i] = x;
                counts[i] = 1;
                d += 1;
                break;
            }
            if keys[i] == x {
                counts[i] += 1;
                break;
            }
            i = (i + 1) & mask;
        }
    }
    let f1 = counts.iter().filter(|&&c| c == 1).count() as u64;
    let f2 = counts.iter().filter(|&&c| c == 2).count() as u64;
    (d, f1, f2)
}

/// (estimate, low end, high end), each clipped to [d, rows]. The high end is the whole input while
/// f2 could still be 0 (below about 6 pairs seen), because Chao1 cannot see a group count much past
/// n^2 / 2 without pairs.
fn chao1(d: u64, f1: u64, f2: u64, rows: usize) -> (f64, f64, f64) {
    let (d, f1, f2, rows) = (d as f64, f1 as f64, f2 as f64, rows as f64);
    let s = f2.sqrt();
    let at = |f: f64| rows.min(d.max(d + f1 * (f1 - 1.0) / (2.0 * (f + 1.0))));
    let low_f2 = f2 - 2.0 * s - 1.0;
    let hi = if low_f2 > 0.0 {
        at(low_f2)
    } else if f1 == 0.0 {
        d
    } else {
        rows
    };
    (at(f2), at(f2 + 2.0 * s + 2.0), hi)
}

/// Estimates the groups of `keys` (column indices) over an input of `total` rows (`None`: the rows
/// of `batches`) from `batches`: the whole input, or a part of it (the first batches of each
/// partition), which the sample is drawn from. `settled(lo, hi)`: whether every group count from
/// `lo` to `hi` gets the same decision; `None` stops once the estimate is within twice d, f2
/// reaches 8, or the low end is a quarter of the input. (The run-time choice uses
/// [`estimate_up_to`] with a smaller largest sample.)
#[cfg(test)]
pub(crate) fn estimate(
    batches: &[&RecordBatch],
    keys: &[usize],
    settled: Option<&dyn Fn(u64, u64) -> bool>,
    total: Option<usize>,
) -> GroupEstimate {
    estimate_up_to(batches, keys, settled, total, SAMPLE_MAX)
}

/// [`estimate`] with samples of at most `max_sample` rows.
pub(crate) fn estimate_up_to(
    batches: &[&RecordBatch],
    keys: &[usize],
    settled: Option<&dyn Fn(u64, u64) -> bool>,
    total: Option<usize>,
    max_sample: usize,
) -> GroupEstimate {
    let t = Instant::now();
    let batches: Vec<&RecordBatch> = batches.iter().copied().filter(|b| b.num_rows() > 0).collect();
    let avail: usize = batches.iter().map(|b| b.num_rows()).sum();
    let rows = total.unwrap_or(avail).max(avail);
    if avail == rows && rows <= EXACT {
        let all: Vec<usize> = (0..rows).collect();
        let (d, _, _) = stats(sample(&batches, keys, &all));
        return GroupEstimate {
            estimate: d,
            low: d,
            high: d,
            sample_rows: rows,
            rows,
            exact: true,
            time: t.elapsed(),
        };
    }
    let cap = max_sample.min(SAMPLE_MAX).min(avail / 4).max(FIRST.min(avail));
    let mut n = FIRST.min(avail);
    loop {
        let (d, f1, f2) = stats(sample(&batches, keys, &positions(avail, n)));
        let (est, lo, hi) = chao1(d, f1, f2, rows);
        let done = n * GROWTH > cap
            || match settled {
                Some(s) => s(lo.round() as u64, hi.round() as u64),
                None => est <= 2.0 * d as f64 || f2 >= SETTLED_F2 || lo * 4.0 >= rows as f64,
            };
        if done {
            return GroupEstimate {
                estimate: est.round() as u64,
                low: lo.round() as u64,
                high: hi.round() as u64,
                sample_rows: n,
                rows,
                exact: false,
                time: t.elapsed(),
            };
        }
        n *= GROWTH;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    fn batches(keys: Vec<i32>, chunk: usize) -> Vec<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, true)]));
        keys.chunks(chunk)
            .map(|c| RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(c.to_vec()))]).unwrap())
            .collect()
    }

    fn uniform(rows: usize, groups: u64) -> Vec<i32> {
        (0..rows as u64).map(|i| (splitmix(i ^ 0x55) % groups) as i32).collect()
    }

    #[test]
    fn small_inputs_are_counted_exactly() {
        let b = batches((0..4000).map(|i| i % 37).collect(), 1000);
        let r: Vec<&RecordBatch> = b.iter().collect();
        let e = estimate(&r, &[0], None, None);
        assert!(e.exact);
        assert_eq!((e.estimate, e.low, e.high), (37, 37, 37));
    }

    #[test]
    fn estimates_land_near_the_truth() {
        // (groups in the key domain, rows); the realised distinct count is what is compared.
        for (groups, rows) in [(200u64, 1_000_000usize), (10_000, 1_000_000), (100_000, 2_000_000), (1_000_000, 4_000_000)] {
            let keys = uniform(rows, groups);
            let mut seen = keys.clone();
            seen.sort_unstable();
            seen.dedup();
            let truth = seen.len() as f64;
            let b = batches(keys, 8192);
            let r: Vec<&RecordBatch> = b.iter().collect();
            let e = estimate(&r, &[0], None, None);
            let ratio = e.estimate as f64 / truth;
            assert!((0.5..2.0).contains(&ratio), "groups {groups}, rows {rows}: estimate {e:?}, truth {truth}");
            assert!(e.low as f64 <= truth * 1.2 && e.high as f64 >= truth * 0.8, "{e:?} vs {truth}");
        }
    }

    #[test]
    fn sorted_keys_are_not_underestimated() {
        // Keys in order (each value twice in a row): a stratified sample sees them spread out.
        let rows = 1_000_000;
        let keys: Vec<i32> = (0..rows as i32).map(|i| i / 2).collect();
        let b = batches(keys, 8192);
        let r: Vec<&RecordBatch> = b.iter().collect();
        let e = estimate(&r, &[0], None, None);
        assert!(e.high as usize >= rows / 4, "{e:?}");
    }

    #[test]
    fn two_keys_and_strings_hash_as_tuples() {
        let n = 20_000;
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("s", DataType::Utf8, true),
        ]));
        let a = Int32Array::from((0..n).map(|i| i % 10).collect::<Vec<i32>>());
        let s = StringArray::from((0..n).map(|i| if i % 7 == 0 { None } else { Some(format!("s{}", i % 3)) }).collect::<Vec<_>>());
        let b = RecordBatch::try_new(schema, vec![Arc::new(a), Arc::new(s)]).unwrap();
        let e = estimate(&[&b], &[0, 1], None, None);
        // 10 x (3 strings + null) = 40 tuples.
        assert!((30..=48).contains(&e.estimate), "{e:?}");
    }

    #[test]
    fn a_prefix_estimates_the_whole_input() {
        // The first 262,144 of 2,000,000 rows, keys uniform over 100,000 values.
        let keys = uniform(2_000_000, 100_000);
        let b = batches(keys[..262_144].to_vec(), 8192);
        let r: Vec<&RecordBatch> = b.iter().collect();
        let e = estimate(&r, &[0], None, Some(2_000_000));
        assert_eq!(e.rows, 2_000_000);
        assert!((50_000..200_000).contains(&e.estimate), "{e:?}");
    }

    /// Probe timing (release build): `cargo test --release --lib probe_timing -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn probe_timing() {
        let keys: Vec<i64> = (0..262_144u64).map(|i| (splitmix(i) % 1_000_000) as i64).collect();
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
        let b: Vec<RecordBatch> = keys
            .chunks(8192)
            .map(|c| RecordBatch::try_new(schema.clone(), vec![Arc::new(arrow::array::Int64Array::from(c.to_vec()))]).unwrap())
            .collect();
        let r: Vec<&RecordBatch> = b.iter().collect();
        for n in [512usize, 2048, 8192, 32768] {
            let pos = positions(262_144, n);
            let t = Instant::now();
            let mut v = Vec::new();
            for _ in 0..100 {
                v = sample(&r, &[0], &pos);
            }
            let ts = t.elapsed() / 100;
            let t = Instant::now();
            for _ in 0..100 {
                std::hint::black_box(stats(v.clone()));
            }
            let tt = t.elapsed() / 100;
            println!("n {n}: sample {ts:?}, stats {tt:?}");
        }
    }

    #[test]
    fn same_input_same_estimate() {
        let b = batches(uniform(500_000, 50_000), 8192);
        let r: Vec<&RecordBatch> = b.iter().collect();
        let (x, y) = (estimate(&r, &[0], None, None), estimate(&r, &[0], None, None));
        assert_eq!((x.estimate, x.low, x.high, x.sample_rows), (y.estimate, y.low, y.high, y.sample_rows));
    }
}
