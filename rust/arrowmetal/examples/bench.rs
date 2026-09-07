//! The one measured timing in `rust/README.md`: `sum` and `filter` at 10M Int64 rows, ArrowMetal
//! from Rust against arrow-rs's own kernels on the same data.
//!
//! ```text
//! ARROWMETAL_LIB=/path/to/libArrowMetalC.dylib cargo run --release --example bench
//! ```
//!
//! Method, so the numbers can be reproduced or disputed:
//!
//! * One 10M-element `Int64Array` of pseudo-random values in `[-1_000_000, 1_000_000)`, no nulls,
//!   built once and shared by every row. The filter predicate is `x > 0`, so roughly half the rows
//!   survive; the exact selectivity is printed.
//! * `std::time::Instant` around the call, wall time, single-threaded. Nothing is subtracted.
//! * 3 untimed warm-up iterations, then 5 timed ones; the table reports the **best** of the 5, which
//!   is the least noisy statistic on a shared machine. The median is printed too.
//! * `std::hint::black_box` on every input and every result, so nothing is optimised away.
//! * Outside a batch, every ArrowMetal call commits its command buffer and waits, so a "kernel"
//!   timing is a complete GPU round trip, not an enqueue.
//!
//! Three columns per operation, because a Rust user pays a different price depending on where the
//! data already lives:
//!
//! * **arrow-rs** -- `arrow::compute::sum` / `arrow::compute::filter`.
//! * **ArrowMetal, kernel** -- the GPU call on an array already imported, mask already on the GPU.
//!   This is what a chain of several operations pays per step.
//! * **ArrowMetal, end to end** -- what a single operation on an arrow-rs array costs, import
//!   included. Exactly what that covers differs by row, so each says: the `sum` row is import + the
//!   reduction and has **no export** (a reduction hands back a scalar through out-parameters, not an
//!   array), while the compare + `filter` row is import + compare + filter + `to_arrow`.
//!
//!   At this size the import is **copy-free**, not a copy: the values buffer comes back page
//!   aligned, which is what `makeBuffer(bytesNoCopy:)` needs. The report prints the alignment it
//!   measured next to the import time so the two are read together. The cost is Metal mapping 80 MB
//!   of pages into the GPU's address space. (`tests/copy_rule.rs` has the alignment-by-size table;
//!   below about 4 KiB a buffer usually is copied.)

use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{Array as _, ArrayRef, BooleanArray, Int64Array, Scalar as ArrowScalar};
use arrow::compute::kernels::cmp;
use arrowmetal::CompareOp;

const N: usize = 10_000_000;
const WARMUP: usize = 3;
const TIMED: usize = 5;

/// Runs `f` `WARMUP` times untimed then `TIMED` times timed, returning (best, median).
fn measure(mut f: impl FnMut()) -> (Duration, Duration) {
    for _ in 0..WARMUP {
        f();
    }
    let mut times = Vec::with_capacity(TIMED);
    for _ in 0..TIMED {
        let t = Instant::now();
        f();
        times.push(t.elapsed());
    }
    times.sort();
    (times[0], times[TIMED / 2])
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// A cheap deterministic generator; `rand` is a dev-dependency and examples do not get those.
fn data(n: usize) -> Int64Array {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            Some((state % 2_000_000) as i64 - 1_000_000)
        })
        .collect()
}

fn main() {
    println!("ArrowMetal {} on {}", arrowmetal::version(), arrowmetal::device_name());
    println!("rustc {}, arrow-rs 59, {N} Int64 rows, no nulls", rustc_version());
    println!("best of {TIMED} after {WARMUP} warm-up iterations, wall time, single-threaded\n");

    let values = data(N);
    let values_ref: ArrayRef = Arc::new(values.clone());

    // The shared predicate and its two mask representations.
    let arrow_mask: BooleanArray =
        cmp::gt(&values, &ArrowScalar::new(&Int64Array::from(vec![0i64]))).unwrap();
    let selected = arrow_mask.true_count();
    println!("filter predicate: x > 0, {selected} of {N} rows kept ({:.1}%)\n",
             100.0 * selected as f64 / N as f64);

    let gpu = arrowmetal::Array::from_arrow(&values).unwrap();
    let gpu_mask = gpu.compare_scalar(CompareOp::Gt, 0i64).unwrap();

    // Correctness first: a benchmark of two different answers is worth nothing.
    let arrow_sum = arrow::compute::sum(&values).unwrap();
    let gpu_sum = gpu.sum().unwrap().unwrap().as_i64().unwrap();
    assert_eq!(arrow_sum, gpu_sum, "sum disagrees; the timing below would be meaningless");
    let arrow_filtered = arrow::compute::filter(&values, &arrow_mask).unwrap();
    let gpu_filtered = gpu.filter(&gpu_mask).unwrap().to_arrow().unwrap();
    assert_eq!(&arrow_filtered, &gpu_filtered, "filter disagrees");
    println!("both libraries agree on both answers\n");

    // ---- sum ------------------------------------------------------------------------------------
    let (a_sum, a_sum_med) = measure(|| {
        black_box(arrow::compute::sum(black_box(&values)));
    });
    let (g_sum, g_sum_med) = measure(|| {
        black_box(black_box(&gpu).sum().unwrap());
    });
    let (e_sum, e_sum_med) = measure(|| {
        let a = arrowmetal::Array::from_arrow(black_box(&values)).unwrap();
        black_box(a.sum().unwrap());
    });

    // ---- filter ---------------------------------------------------------------------------------
    let (a_filter, a_filter_med) = measure(|| {
        black_box(arrow::compute::filter(black_box(&values), black_box(&arrow_mask)).unwrap());
    });
    let (g_filter, g_filter_med) = measure(|| {
        black_box(black_box(&gpu).filter(black_box(&gpu_mask)).unwrap());
    });
    let (e_filter, e_filter_med) = measure(|| {
        let a = arrowmetal::Array::from_arrow(black_box(&values)).unwrap();
        let m = a.compare_scalar(CompareOp::Gt, 0i64).unwrap();
        black_box(a.filter(&m).unwrap().to_arrow().unwrap());
    });

    // The end-to-end filter row above pays for a compare arrow-rs's row does not, so time the
    // arrow-rs equivalent too rather than compare unlike things.
    let (a_cmp_filter, a_cmp_filter_med) = measure(|| {
        let m = cmp::gt(black_box(&values), &ArrowScalar::new(&Int64Array::from(vec![0i64]))).unwrap();
        black_box(arrow::compute::filter(black_box(&values), &m).unwrap());
    });

    // Import and export on their own, so the end-to-end rows can be read.
    let (imp, imp_med) = measure(|| {
        black_box(arrowmetal::Array::from_arrow(black_box(&values)).unwrap());
    });
    let (exp, exp_med) = measure(|| {
        black_box(black_box(&gpu).to_arrow().unwrap());
    });

    println!("| Operation, 10M Int64 | arrow-rs | ArrowMetal, kernel | ArrowMetal, end to end |");
    println!("|---|---|---|---|");
    println!(
        "| `sum` | {:.2} ms | {:.2} ms | {:.2} ms |",
        ms(a_sum),
        ms(g_sum),
        ms(e_sum)
    );
    println!(
        "| `filter` (mask ready) | {:.2} ms | {:.2} ms | -- |",
        ms(a_filter),
        ms(g_filter)
    );
    println!(
        "| compare + `filter` | {:.2} ms | -- | {:.2} ms |",
        ms(a_cmp_filter),
        ms(e_filter)
    );
    // Whether that import copied depends on the buffer's alignment, which at this size is measured
    // by tests/copy_rule.rs rather than assumed here; the report prints the alignment so the two
    // can be read together.
    let values_alignment = {
        let data = values.to_data();
        let p = data.buffers()[0].as_ptr() as usize;
        1usize << p.trailing_zeros()
    };
    println!("\nSupporting numbers (best of {TIMED}):");
    println!(
        "  import (arrow-rs -> ArrowMetal): {:.3} ms   [values buffer aligned to {values_alignment} B; \
         copy-free needs 16384]",
        ms(imp)
    );
    println!("  export (ArrowMetal -> arrow-rs, no copy): {:.3} ms", ms(exp));

    println!("\nMedians, for comparison with the bests above:");
    println!("  arrow-rs sum {:.2} ms | ArrowMetal kernel {:.2} ms | end to end {:.2} ms",
             ms(a_sum_med), ms(g_sum_med), ms(e_sum_med));
    println!("  arrow-rs filter {:.2} ms | ArrowMetal kernel {:.2} ms", ms(a_filter_med), ms(g_filter_med));
    println!("  arrow-rs cmp+filter {:.2} ms | ArrowMetal end to end {:.2} ms",
             ms(a_cmp_filter_med), ms(e_filter_med));
    println!("  import {:.3} ms | export {:.3} ms", ms(imp_med), ms(exp_med));

    println!("\nvalues live: {} rows, {} filtered", values_ref.len(), gpu_filtered.len());
}

fn rustc_version() -> String {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}
