//! Chunked import against concatenate-then-import, for a column held as many arrow-rs arrays.
//!
//! For each size, type and chunk size, three columns of that type are built as separate arrow-rs
//! arrays of `chunk` rows each (as a stream of `RecordBatch`es holds them), then imported three ways:
//!
//! * `concat+import`: `arrow::compute::concat` per column, then `Array::from_arrow` (split into the
//!   two steps);
//! * `chunked`: `Array::from_arrow_chunks` per column;
//! * `single`: `Array::from_arrow` of an already-concatenated column (the import step alone).
//!
//! Best of `--iters` (default 5) runs after one warm-up; wall time and process CPU time (user +
//! system, all threads) of the best run; the 1-minute load average before each case. It also
//! reports whether the imported buffers of the concatenated column were mapped without a copy.
//!
//! `--no-chunked` skips the chunked import (for a library without `am_import_chunks`), and
//! `--label` names the run in the first column.
//!
//! ```text
//! cargo run --release --example chunked_import -- [--rows 1000000,10000000,50000000] [--iters 5]
//! ```

use arrow::array::{ArrayRef, Float64Array, Int64Array, StringBuilder, StringViewBuilder};
use arrowmetal::Array;
use std::sync::Arc;
use std::time::Instant;

#[repr(C)]
#[derive(Default)]
struct Timeval {
    sec: i64,
    usec: i32,
}
#[repr(C)]
#[derive(Default)]
struct Rusage {
    utime: Timeval,
    stime: Timeval,
    rest: [i64; 14],
}
unsafe extern "C" {
    fn getrusage(who: i32, usage: *mut Rusage) -> i32;
    fn getloadavg(loads: *mut f64, n: i32) -> i32;
}

fn cpu_ms() -> f64 {
    let mut r = Rusage::default();
    unsafe { getrusage(0, &mut r) };
    (r.utime.sec + r.stime.sec) as f64 * 1e3 + (r.utime.usec + r.stime.usec) as f64 / 1e3
}

fn load() -> f64 {
    let mut l = [0f64; 3];
    unsafe { getloadavg(l.as_mut_ptr(), 3) };
    l[0]
}

/// One column of `rows` rows as arrays of `chunk` rows, each built on its own.
fn column(ty: &str, rows: usize, chunk: usize, seed: u64) -> Vec<ArrayRef> {
    let mut out = Vec::new();
    let mut at = 0usize;
    let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    while at < rows {
        let n = chunk.min(rows - at);
        let a: ArrayRef = match ty {
            "int64" => Arc::new(Int64Array::from_iter_values((0..n).map(|_| (next() >> 1) as i64))),
            "float64" => Arc::new(Float64Array::from_iter_values((0..n).map(|_| (next() >> 11) as f64 / 1e6))),
            "utf8" | "utf8view" => {
                // 1,000 distinct values; a third of them longer than 12 bytes (out of line as views).
                let word = |v: u64| {
                    let k = v % 1000;
                    if k % 3 == 0 { format!("a-longer-name-{k:04}") } else { format!("name-{k:04}") }
                };
                if ty == "utf8" {
                    let mut b = StringBuilder::with_capacity(n, n * 12);
                    for _ in 0..n {
                        b.append_value(word(next()));
                    }
                    Arc::new(b.finish())
                } else {
                    let mut b = StringViewBuilder::with_capacity(n);
                    for _ in 0..n {
                        b.append_value(word(next()));
                    }
                    Arc::new(b.finish())
                }
            }
            _ => unreachable!(),
        };
        out.push(a);
        at += n;
    }
    out
}

struct Best {
    wall: f64,
    cpu: f64,
    parts: (f64, f64),
}

fn best(iters: usize, mut run: impl FnMut() -> (f64, f64)) -> Best {
    run(); // warm-up
    let mut b = Best { wall: f64::MAX, cpu: 0.0, parts: (0.0, 0.0) };
    for _ in 0..iters {
        let (c0, t0) = (cpu_ms(), Instant::now());
        let parts = run();
        let (wall, cpu) = (t0.elapsed().as_secs_f64() * 1e3, cpu_ms() - c0);
        if wall < b.wall {
            b = Best { wall, cpu, parts };
        }
    }
    b
}

fn page_aligned(a: &dyn arrow::array::Array) -> String {
    let (ffi, _s) = arrow::ffi::to_ffi(&a.to_data()).unwrap();
    // Validity, values or offsets / views, and the first data buffer; then the buffer count.
    let marks: String = (0..ffi.num_buffers().min(3))
        .map(|i| {
            let p = ffi.buffer(i) as usize;
            if p == 0 { "-" } else if p % 16384 == 0 { "Y" } else { "n" }
        })
        .collect();
    format!("{marks}/{}", ffi.num_buffers())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let rows: Vec<usize> = arg("--rows")
        .unwrap_or_else(|| "1000000,10000000,50000000".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let iters: usize = arg("--iters").map(|s| s.parse().unwrap()).unwrap_or(5);
    let label = arg("--label").unwrap_or_else(|| "new".into());
    let with_chunked = !args.iter().any(|a| a == "--no-chunked");
    let types: Vec<String> = arg("--types")
        .unwrap_or_else(|| "int64,float64,utf8,utf8view".into())
        .split(',')
        .map(String::from)
        .collect();
    println!("label,rows,type,chunk_rows,chunks,load,load_end,concat_import_wall_ms,concat_import_cpu_ms,concat_ms,import_after_concat_ms,chunked_wall_ms,chunked_cpu_ms,single_import_wall_ms,single_import_cpu_ms,speedup,concat_page_aligned");
    for &n in &rows {
        for ty in &types {
            for chunk in [8192usize, n.div_ceil(16)] {
                let cols: Vec<Vec<ArrayRef>> = (0..3).map(|c| column(ty, n, chunk, 7 + c)).collect();
                let flat: Vec<ArrayRef> = cols
                    .iter()
                    .map(|c| arrow::compute::concat(&c.iter().map(|a| a.as_ref()).collect::<Vec<_>>()).unwrap())
                    .collect();
                let aligned = page_aligned(flat[0].as_ref());
                drop(flat);
                let l = load();
                let today = best(iters, || {
                    let mut concat_ms = 0.0;
                    let mut import_ms = 0.0;
                    let mut keep = Vec::new();
                    for c in &cols {
                        let t = Instant::now();
                        let merged = arrow::compute::concat(&c.iter().map(|a| a.as_ref()).collect::<Vec<_>>()).unwrap();
                        concat_ms += t.elapsed().as_secs_f64() * 1e3;
                        let t = Instant::now();
                        keep.push(Array::from_arrow(merged.as_ref()).unwrap());
                        import_ms += t.elapsed().as_secs_f64() * 1e3;
                    }
                    (concat_ms, import_ms)
                });
                let chunked = if !with_chunked {
                    Best { wall: f64::NAN, cpu: f64::NAN, parts: (0.0, 0.0) }
                } else {
                    best(iters, || {
                    let keep: Vec<Array> = cols
                        .iter()
                        .map(|c| Array::from_arrow_chunks(&c.iter().map(|a| a.as_ref()).collect::<Vec<_>>()).unwrap())
                        .collect();
                    assert_eq!(keep[0].len(), n);
                    (0.0, 0.0)
                    })
                };
                let flat: Vec<ArrayRef> = cols
                    .iter()
                    .map(|c| arrow::compute::concat(&c.iter().map(|a| a.as_ref()).collect::<Vec<_>>()).unwrap())
                    .collect();
                let single = best(iters, || {
                    let keep: Vec<Array> = flat.iter().map(|a| Array::from_arrow(a.as_ref()).unwrap()).collect();
                    assert_eq!(keep[0].len(), n);
                    (0.0, 0.0)
                });
                println!(
                    "{label},{n},{ty},{chunk},{},{l:.2},{:.2},{:.2},{:.1},{:.2},{:.2},{:.2},{:.1},{:.2},{:.1},{:.2},{aligned}",
                    cols[0].len(),
                    load(),
                    today.wall,
                    today.cpu,
                    today.parts.0,
                    today.parts.1,
                    chunked.wall,
                    chunked.cpu,
                    single.wall,
                    single.cpu,
                    today.wall / chunked.wall,
                );
            }
        }
    }
}
