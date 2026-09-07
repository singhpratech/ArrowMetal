//! The example in `docs/RUST.md`, kept here so `cargo build --examples` proves it compiles and
//! `cargo run --example quickstart` proves it runs. Edit the two together.
//!
//! ```text
//! ARROWMETAL_LIB=/path/to/libArrowMetalC.dylib cargo run --release --example quickstart
//! ```

use arrow::array::Int64Array;
use arrowmetal::{group_by, Array, CompareOp};

fn main() -> Result<(), arrowmetal::Error> {
    let region = Int64Array::from(vec![0i64, 1, 0, 2, 1]);
    let amount = Int64Array::from(vec![Some(10i64), Some(20), None, Some(5), Some(7)]);

    let region = Array::from_arrow(&region)?; // to the GPU
    let amount = Array::from_arrow(&amount)?;

    let big = amount.compare_scalar(CompareOp::Gt, 5i64)?; // boolean mask
    let kept_amount = amount.filter(&big)?; // 10, 20, 7
    let kept_region = region.filter(&big)?;

    let gb = group_by(&[&kept_region])?;
    let totals = gb.sum(&kept_amount)?.to_arrow()?; // back to arrow-rs
    let keys = gb.keys(0)?.to_arrow()?;

    let ints = |a: &dyn arrow::array::Array| -> Vec<i64> {
        a.as_any().downcast_ref::<Int64Array>().unwrap().values().to_vec()
    };
    println!("{} groups: {:?} -> {:?}", gb.group_count(), ints(&keys), ints(&totals));
    println!("sum of everything: {:?}", amount.sum()?);
    Ok(())
}
