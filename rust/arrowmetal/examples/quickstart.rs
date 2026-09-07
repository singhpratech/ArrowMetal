//! The example in `docs/RUST.md`, kept here so `cargo build --examples` proves it compiles and
//! `cargo run --example quickstart` proves it runs. Edit the two together.
//!
//! ```text
//! ARROWMETAL_LIB=/path/to/libArrowMetalC.dylib cargo run --release --example quickstart
//! ```

use arrow::array::{ArrayRef, Int64Array};
use arrowmetal::{group_by, Array, CompareOp};
use std::sync::Arc;

fn main() -> Result<(), arrowmetal::Error> {
    let region: ArrayRef = Arc::new(Int64Array::from(vec![0i64, 1, 0, 2, 1]));
    let amount: ArrayRef =
        Arc::new(Int64Array::from(vec![Some(10i64), Some(20), None, Some(5), Some(7)]));

    let region = Array::from_arrow(region.as_ref())?; // to the GPU
    let amount = Array::from_arrow(amount.as_ref())?;

    let big = amount.compare_scalar(CompareOp::Gt, 5i64)?; // boolean mask
    let kept_amount = amount.filter(&big)?; // 10, 20, 7
    let kept_region = region.filter(&big)?;

    let gb = group_by(&[&kept_region])?;
    let totals: ArrayRef = gb.sum(&kept_amount)?.to_arrow()?; // back to arrow-rs
    let keys: ArrayRef = gb.keys(0)?.to_arrow()?;

    println!("{} groups: {keys:?} -> {totals:?}", gb.group_count());
    println!("sum of everything: {:?}", amount.sum()?);
    Ok(())
}
