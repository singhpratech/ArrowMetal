//! Shared fixtures for the integration tests.
//!
//! The sizes are the ones the project tests everything at: 0 and 1 (degenerate), a word boundary,
//! a threadgroup boundary, and 1,000,001 -- an odd length past a million that crosses a threadgroup
//! boundary and leaves a partial tail. Nothing here exceeds 10M elements.

#![allow(dead_code)]

use arrow::array::{ArrayRef, Float64Array, Int64Array};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;

/// The lengths every kernel test sweeps.
pub const LENGTHS: &[usize] = &[0, 1, 33, 1024, 1025, 100_001, 1_000_001];

/// A deterministic RNG, so a failure is reproducible from the seed alone.
pub fn rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// `n` int64s in a range that cannot overflow an i64 sum at these lengths, with roughly one null in
/// `null_every` (0 = no nulls).
pub fn int64(n: usize, null_every: usize, seed: u64) -> Int64Array {
    let mut r = rng(seed);
    (0..n)
        .map(|i| {
            if null_every != 0 && i % null_every == 0 {
                None
            } else {
                Some(r.random_range(-1_000_000i64..1_000_000))
            }
        })
        .collect()
}

/// `n` float64s, no NaN and no infinity (NaN semantics are pinned separately), with nulls as above.
pub fn float64(n: usize, null_every: usize, seed: u64) -> Float64Array {
    let mut r = rng(seed);
    (0..n)
        .map(|i| {
            if null_every != 0 && i % null_every == 0 {
                None
            } else {
                Some(r.random_range(-1000.0f64..1000.0))
            }
        })
        .collect()
}

/// A round trip through ArrowMetal, so a kernel's input is the array arrow-rs was handed.
pub fn roundtrip(a: &ArrayRef) -> ArrayRef {
    let gpu = arrowmetal::Array::from_arrow(a.as_ref()).expect("import");
    gpu.to_arrow().expect("export")
}

pub fn arc(a: impl arrow::array::Array + 'static) -> ArrayRef {
    Arc::new(a)
}

/// Compares two f64 answers that were accumulated in different orders. Float addition does not
/// associate, so a GPU tree reduction and a CPU sequential sum differ in the last bits; this is a
/// relative tolerance, not an exact match, and the test that uses it says so.
pub fn close(a: f64, b: f64, rel: f64) -> bool {
    if a == b {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(1.0);
    (a - b).abs() / scale <= rel
}
