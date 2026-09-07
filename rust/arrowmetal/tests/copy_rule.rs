//! The copy rule, measured rather than assumed.
//!
//! ArrowMetal's import is copy-free only when every buffer pointer is page aligned, because that is
//! what `MTLDevice.makeBuffer(bytesNoCopy:)` requires. Whether an arrow-rs array clears that bar is
//! not a property of arrow-rs alone -- it depends on the size, on which of arrow-rs's two allocation
//! paths built the buffer, and on what the platform allocator does with that request. So this file
//! measures it across sizes and construction paths and prints a table, and `docs/RUST.md` quotes
//! that table rather than a guess.
//!
//! ```text
//! cargo test --test copy_rule -- --nocapture
//! ```

use arrow::array::{Array as _, ArrayRef, Float64Array, Int64Array};
use std::sync::Arc;

/// Apple silicon's page size, and what `makeBuffer(bytesNoCopy:)` demands.
const PAGE: usize = 16384;

fn alignment_of(p: *const u8) -> usize {
    let addr = p as usize;
    if addr == 0 {
        return 0;
    }
    1usize << addr.trailing_zeros().min(usize::BITS - 1)
}

fn buffer_alignments(a: &dyn arrow::array::Array) -> Vec<usize> {
    a.to_data().buffers().iter().map(|b| alignment_of(b.as_ptr())).collect()
}

/// arrow-rs builds a values buffer two different ways, and they do not use the same allocator.
#[derive(Clone, Copy, Debug)]
enum Path {
    /// `Int64Array::from(Vec<i64>)` adopts the `Vec`'s own allocation: Rust's global allocator, an
    /// alignment request of 8.
    FromVec,
    /// Collecting from an iterator of `Option` fills an arrow-rs `MutableBuffer`, which requests
    /// arrow-buffer's own `ALIGNMENT` (64) and carries a validity bitmap as a second buffer.
    Collected,
}

fn build(path: Path, n: usize, seed: i64) -> ArrayRef {
    match path {
        Path::FromVec => Arc::new(Int64Array::from((0..n as i64).map(|i| i + seed).collect::<Vec<_>>())),
        Path::Collected => Arc::new(
            (0..n as i64)
                .map(|i| if i % 7 == 0 { None } else { Some(i + seed) })
                .collect::<Int64Array>(),
        ),
    }
}

/// The measurement the crate docs and `docs/RUST.md` quote: how often does an arrow-rs buffer land
/// on a page boundary, by size and by construction path?
///
/// This asserts only what it can honestly assert -- that the alignments are sane, and that the small
/// sizes really do miss the page boundary sometimes, so the copy path is genuinely exercised
/// elsewhere in the suite. The fractions themselves are reported, not pinned: they are the platform
/// allocator's business and may move.
#[test]
fn arrow_rs_buffer_page_alignment_by_size_and_path() {
    const TRIALS: usize = 32;
    let sizes: &[usize] = &[16, 512, 8_192, 131_072, 1_250_000, 10_000_000];

    println!("\narrow-rs values-buffer alignment, {TRIALS} allocations each (page = {PAGE} B)");
    println!("| elements | bytes | path | page aligned | min alignment seen |");
    println!("|---|---|---|---|---|");

    let mut small_miss = false;
    let mut large_from_vec_page_aligned = 0usize;

    for &path in &[Path::FromVec, Path::Collected] {
        for &n in sizes {
            let mut live = Vec::with_capacity(TRIALS); // hold them so no two reuse one address
            let mut aligned = 0usize;
            let mut min_seen = usize::MAX;
            for t in 0..TRIALS {
                let a = build(path, n, t as i64);
                // Buffer 0 of a `Collected` array is the validity bitmap; the values buffer is the
                // last one either way, and it is the big one the copy rule turns on.
                let als = buffer_alignments(a.as_ref());
                let values_alignment = *als.last().unwrap();
                min_seen = min_seen.min(values_alignment);
                if values_alignment % PAGE == 0 {
                    aligned += 1;
                }
                live.push(a);
            }
            println!(
                "| {n} | {} | {path:?} | {aligned}/{TRIALS} | {min_seen} |",
                n * 8
            );
            if n <= 512 && aligned < TRIALS {
                small_miss = true;
            }
            if n == 10_000_000 && matches!(path, Path::FromVec) {
                large_from_vec_page_aligned = aligned;
            }
            drop(live);
        }
    }

    assert!(
        small_miss,
        "every small arrow-rs buffer was page aligned on this machine; the copy path in \
         `am_import` is then never exercised by these tests and docs/RUST.md needs re-measuring"
    );
    println!(
        "\n10M-element Int64Array::from(Vec): {large_from_vec_page_aligned}/{TRIALS} page aligned \
         -- this is the size the benchmark imports, so its import cost follows from this row"
    );
}

/// Copy-free out: an array ArrowMetal exported is backed by an `MTLBuffer` in shared memory, whose
/// contents pointer is page aligned. That is both the evidence for "copy-free out" and the reason an
/// exported array re-imports without a copy whatever the original alignment was.
#[test]
fn arrowmetal_exported_buffers_are_page_aligned() {
    for n in [1usize, 1000, 100_001, 1_000_001] {
        let a: ArrayRef = Arc::new(
            (0..n as i64).map(|i| if i % 5 == 0 { None } else { Some(i) }).collect::<Int64Array>(),
        );
        let out = arrowmetal::Array::from_arrow(a.as_ref()).unwrap().to_arrow().unwrap();

        let alignments = buffer_alignments(out.as_ref());
        println!("n={n}: ArrowMetal exported buffer alignments {alignments:?} (page = {PAGE})");
        for al in &alignments {
            assert_eq!(al % PAGE, 0, "n={n}: an exported buffer was aligned to only {al} bytes");
        }
    }
}

/// Both sides of the copy rule must give the same answer. A small array misses the page boundary and
/// is copied in; its own export is page aligned, so sending it straight back takes the copy-free
/// path. Both round trips must reproduce the input exactly.
#[test]
fn the_copy_and_no_copy_paths_agree() {
    for n in [100usize, 4096, 250_001] {
        let a: ArrayRef = Arc::new(
            (0..n)
                .map(|i| if i % 5 == 0 { None } else { Some(i as f64 * 0.5) })
                .collect::<Float64Array>(),
        );

        // First trip: whatever arrow-rs's allocator gave us.
        let once = arrowmetal::Array::from_arrow(a.as_ref()).unwrap().to_arrow().unwrap();
        assert_eq!(&once, &a, "n={n}, first trip");

        // Second trip: the input is now ArrowMetal's own page-aligned export, so the import takes
        // the "buffers we already own" fast path.
        let twice = arrowmetal::Array::from_arrow(once.as_ref()).unwrap().to_arrow().unwrap();
        assert_eq!(&twice, &a, "n={n}, second trip");
    }
}

/// A sliced input imports without a copy of its own -- the offset rides on the handle rather than
/// being applied -- so a slice must survive the trip with the same values and null count, including
/// the two offsets either side of a page boundary.
#[test]
fn a_slice_survives_the_round_trip() {
    let base: ArrayRef = Arc::new(
        (0..300_007i64).map(|i| if i % 11 == 0 { None } else { Some(i) }).collect::<Int64Array>(),
    );
    for &(off, len) in &[(1usize, 300_000usize), (16383, 1024), (16384, 1024), (2047, 8193)] {
        let sliced = base.slice(off, len);
        let out = arrowmetal::Array::from_arrow(sliced.as_ref()).unwrap().to_arrow().unwrap();
        assert_eq!(&out, &sliced, "slice({off}, {len})");
        assert_eq!(out.null_count(), sliced.null_count());
    }
}
