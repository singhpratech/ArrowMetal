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

/// The alignment of every buffer **as ArrowMetal will see it**, validity bitmap included.
///
/// `ArrayData::buffers()` is the wrong thing to measure: it excludes the null bitmap, which lives in
/// `ArrayData::nulls()` and is buffer 0 of the C Data Interface array. An earlier version of this
/// file used it and therefore never looked at validity at all. Exporting through `arrow::ffi` and
/// reading `FFI_ArrowArray::buffer(i)` gives exactly the pointers `am_import` receives, in the
/// order it receives them, so there is nothing left to get wrong.
///
/// Index 0 is the validity bitmap for a primitive array (null when the array has no nulls) and
/// index 1 the values.
fn ffi_buffer_alignments(a: &dyn arrow::array::Array) -> Vec<usize> {
    let (ffi_array, _schema) = arrow::ffi::to_ffi(&a.to_data()).expect("to_ffi");
    (0..ffi_array.num_buffers()).map(|i| alignment_of(ffi_array.buffer(i))).collect()
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

    println!("\narrow-rs buffer alignment as exported through the C Data Interface,");
    println!("{TRIALS} allocations each (page = {PAGE} B). A copied buffer is one that is not");
    println!("page aligned. `Collected` arrays carry a validity bitmap; `FromVec` ones do not.");
    println!("| elements | values bytes | path | values page aligned | validity page aligned |");
    println!("|---|---|---|---|---|");

    let mut small_values_miss = false;
    let mut large_from_vec_page_aligned = 0usize;
    let mut validity_rows: Vec<(usize, usize)> = Vec::new();

    for &path in &[Path::FromVec, Path::Collected] {
        for &n in sizes {
            let mut live = Vec::with_capacity(TRIALS); // hold them so no two reuse one address
            let mut values_aligned = 0usize;
            let mut validity_aligned = 0usize;
            let mut validity_present = 0usize;
            for t in 0..TRIALS {
                let a = build(path, n, t as i64);
                let als = ffi_buffer_alignments(a.as_ref());
                // Buffer 0 is validity (0 when absent), buffer 1 the values.
                let validity = als.first().copied().unwrap_or(0);
                let values = als.get(1).copied().unwrap_or(0);
                if values != 0 && values % PAGE == 0 {
                    values_aligned += 1;
                }
                if validity != 0 {
                    validity_present += 1;
                    if validity % PAGE == 0 {
                        validity_aligned += 1;
                    }
                }
                live.push(a);
            }
            let validity_cell = if validity_present == 0 {
                "no validity buffer".to_string()
            } else {
                format!("{validity_aligned}/{validity_present}")
            };
            println!("| {n} | {} | {path:?} | {values_aligned}/{TRIALS} | {validity_cell} |", n * 8);

            if n <= 512 && values_aligned < TRIALS {
                small_values_miss = true;
            }
            if n == 10_000_000 && matches!(path, Path::FromVec) {
                large_from_vec_page_aligned = values_aligned;
            }
            if validity_present > 0 {
                validity_rows.push((n, validity_aligned));
            }
            drop(live);
        }
    }

    assert!(
        small_values_miss,
        "every small arrow-rs values buffer was page aligned on this machine; the copy path in \
         `am_import` is then never exercised by these tests and docs/RUST.md needs re-measuring"
    );
    println!(
        "\n10M-element Int64Array::from(Vec): {large_from_vec_page_aligned}/{TRIALS} values page \
         aligned -- this is the size the benchmark imports, so its import cost follows from this row"
    );
    println!("validity bitmaps, page aligned out of {TRIALS}: {validity_rows:?}");
    println!(
        "A validity bitmap is n/8 bytes, so it reaches the allocator's page-granted sizes far later \
         than the values buffer does: a nullable column of well under ~130k rows normally has its \
         bitmap copied (about a kilobyte) while its values stay wrapped."
    );
}

/// Copy-free out: an array ArrowMetal exported is backed by an `MTLBuffer` in shared memory, whose
/// contents pointer is page aligned. That is both the evidence for "copy-free out" and the reason an
/// exported array re-imports without a copy whatever the original alignment was.
///
/// Every input here has nulls, so **both** buffers are exercised — the validity bitmap as well as
/// the values. That matters: the bitmap is the buffer arrow-rs is least likely to hand over page
/// aligned on the way in, so the fact that ArrowMetal always hands it back aligned is what makes an
/// ArrowMetal-to-ArrowMetal hop free in both buffers.
#[test]
fn arrowmetal_exported_buffers_are_page_aligned() {
    for n in [1usize, 1000, 100_001, 1_000_001] {
        let a: ArrayRef = Arc::new(
            (0..n as i64).map(|i| if i % 5 == 0 { None } else { Some(i) }).collect::<Int64Array>(),
        );
        assert!(a.null_count() > 0, "n={n}: this test needs a validity bitmap to measure");

        let out = arrowmetal::Array::from_arrow(a.as_ref()).unwrap().to_arrow().unwrap();

        let alignments = ffi_buffer_alignments(out.as_ref());
        println!(
            "n={n}: ArrowMetal exported buffer alignments [validity, values] = {alignments:?} \
             (page = {PAGE})"
        );
        assert_eq!(alignments.len(), 2, "n={n}: expected a validity and a values buffer");
        for (i, al) in alignments.iter().enumerate() {
            let which = if i == 0 { "validity" } else { "values" };
            assert_ne!(*al, 0, "n={n}: the {which} buffer was null on export");
            assert_eq!(
                al % PAGE,
                0,
                "n={n}: the exported {which} buffer was aligned to only {al} bytes"
            );
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
