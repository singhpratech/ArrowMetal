//! Raw FFI declarations for ArrowMetal's C ABI (`include/arrowmetal.h`).
//!
//! Everything here is `unsafe extern "C"` and nothing here is safe to call by hand; the checked
//! wrappers live in the [`arrowmetal`](https://docs.rs/arrowmetal) crate one level up. Each
//! declaration below is a transcription of one line of the header, with the header's own comment
//! for the op numbering kept alongside it.
//!
//! Hand-written rather than bindgen-generated: a checked-in file needs no libclang on the build
//! machine and no `build.rs` codegen step, and the header is stable. `tests/signatures.rs` in the
//! `arrowmetal` crate re-parses `include/arrowmetal.h` and checks every declaration here against it,
//! so a drift is a test failure rather than a silent ABI mismatch.
//!
//! # Conventions, from the header
//!
//! * Every `int`-returning function returns 0 on success and non-zero on error.
//! * [`ffi::am_last_error`] holds the message. It is thread-local and valid only until the next call
//!   on that thread, so read it immediately.
//! * A function that writes an `am_array**` out-parameter hands over a new handle the caller must
//!   free with [`ffi::am_release`].
//! * `am_import` **moves** the `ArrowArray` it is given (it marks the caller's struct released), and
//!   only borrows the `ArrowSchema`.
//!
//! This crate declares the subset of the ABI the safe crate uses, not all 220 entry points; see
//! `docs/RUST.md` for what is and is not covered.

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_void};

/// The directory `build.rs` linked against, baked in for diagnostics.
pub const LIB_DIR: &str = env!("ARROWMETAL_SYS_LIB_DIR");

// -------------------------------------------------------------------------------------------------
// Arrow C Data Interface (include/arrow_abi.h). Layout is normative and matches `arrow::ffi`'s
// `FFI_ArrowSchema` / `FFI_ArrowArray` field for field, which is what lets the safe crate pass an
// arrow-rs struct straight through as one of these.
// -------------------------------------------------------------------------------------------------

/// ABI-compatible with `struct ArrowSchema`.
#[repr(C)]
#[derive(Debug)]
pub struct ArrowSchema {
    pub format: *const c_char,
    pub name: *const c_char,
    pub metadata: *const c_char,
    pub flags: i64,
    pub n_children: i64,
    pub children: *mut *mut ArrowSchema,
    pub dictionary: *mut ArrowSchema,
    pub release: Option<unsafe extern "C" fn(*mut ArrowSchema)>,
    pub private_data: *mut c_void,
}

/// ABI-compatible with `struct ArrowArray`.
#[repr(C)]
#[derive(Debug)]
pub struct ArrowArray {
    pub length: i64,
    pub null_count: i64,
    pub offset: i64,
    pub n_buffers: i64,
    pub n_children: i64,
    pub buffers: *mut *const c_void,
    pub children: *mut *mut ArrowArray,
    pub dictionary: *mut ArrowArray,
    pub release: Option<unsafe extern "C" fn(*mut ArrowArray)>,
    pub private_data: *mut c_void,
}

impl ArrowSchema {
    /// An unset schema, ready to receive an export.
    pub const fn empty() -> Self {
        Self {
            format: std::ptr::null(),
            name: std::ptr::null(),
            metadata: std::ptr::null(),
            flags: 0,
            n_children: 0,
            children: std::ptr::null_mut(),
            dictionary: std::ptr::null_mut(),
            release: None,
            private_data: std::ptr::null_mut(),
        }
    }
}

impl ArrowArray {
    /// An unset array, ready to receive an export.
    pub const fn empty() -> Self {
        Self {
            length: 0,
            null_count: 0,
            offset: 0,
            n_buffers: 0,
            n_children: 0,
            buffers: std::ptr::null_mut(),
            children: std::ptr::null_mut(),
            dictionary: std::ptr::null_mut(),
            release: None,
            private_data: std::ptr::null_mut(),
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Opaque handles
// -------------------------------------------------------------------------------------------------

/// Opaque, Metal-resident Arrow array. Freed with [`ffi::am_release`].
#[repr(C)]
pub struct am_array {
    _private: [u8; 0],
}

/// Opaque key mapping produced by [`ffi::am_group_by_keys`]. Freed with [`ffi::am_group_by_release`].
#[repr(C)]
pub struct am_groupby {
    _private: [u8; 0],
}

/// One table registered with the plan runner. Freed with [`ffi::am_plan_source_release`].
#[repr(C)]
pub struct am_plan_source {
    _private: [u8; 0],
}

/// The output of one [`ffi::am_plan_run`]. Freed with [`ffi::am_plan_result_release`].
#[repr(C)]
pub struct am_plan_result {
    _private: [u8; 0],
}

// -------------------------------------------------------------------------------------------------
// The C ABI itself
// -------------------------------------------------------------------------------------------------

pub mod ffi {
    use super::*;

    unsafe extern "C" {
        // -- identity ---------------------------------------------------------------------------
        pub fn am_version() -> *const c_char;
        pub fn am_device_name() -> *const c_char;
        /// Thread-local, valid until the next call on this thread.
        pub fn am_last_error() -> *const c_char;

        // -- lifecycle and interop --------------------------------------------------------------
        // Zero-copy when the producer's buffers are page aligned; otherwise one copy.
        // `array` is moved: on success the caller's struct is left released.
        pub fn am_import(
            schema: *const ArrowSchema,
            array: *mut ArrowArray,
            out: *mut *mut am_array,
        ) -> c_int;
        pub fn am_export(
            a: *mut am_array,
            schema: *mut ArrowSchema,
            array: *mut ArrowArray,
        ) -> c_int;
        pub fn am_release(a: *mut am_array);
        pub fn am_length(a: *mut am_array) -> i64;
        pub fn am_null_count(a: *mut am_array) -> i64;
        /// Arrow format string: `c C s S i I l L f g b u z t... d:p,s`.
        pub fn am_format(a: *mut am_array) -> *const c_char;

        // -- reductions -------------------------------------------------------------------------
        // op: 0 sum, 1 min, 2 max, 3 mean.
        // out_kind: 0 = int64 in out_i64, 1 = uint64 in out_i64's slot, 2 = float64 in out_f64.
        pub fn am_reduce(
            a: *mut am_array,
            op: c_int,
            out_i64: *mut i64,
            out_f64: *mut f64,
            out_kind: *mut c_int,
            is_null: *mut c_int,
        ) -> c_int;

        // -- element-wise -----------------------------------------------------------------------
        // cmp op: 0 eq, 1 ne, 2 lt, 3 le, 4 gt, 5 ge. arith op: 0 add, 1 sub, 2 mul, 3 div.
        pub fn am_compare_scalar(
            a: *mut am_array,
            op: c_int,
            scalar: *const c_void,
            out: *mut *mut am_array,
        ) -> c_int;
        pub fn am_compare_array(
            a: *mut am_array,
            op: c_int,
            b: *mut am_array,
            out: *mut *mut am_array,
        ) -> c_int;
        pub fn am_arith_scalar(
            a: *mut am_array,
            op: c_int,
            scalar: *const c_void,
            out: *mut *mut am_array,
        ) -> c_int;
        pub fn am_arith_array(
            a: *mut am_array,
            op: c_int,
            b: *mut am_array,
            out: *mut *mut am_array,
        ) -> c_int;
        pub fn am_cast(a: *mut am_array, format: *const c_char, out: *mut *mut am_array) -> c_int;

        // -- selection --------------------------------------------------------------------------
        pub fn am_filter(a: *mut am_array, mask: *mut am_array, out: *mut *mut am_array) -> c_int;
        pub fn am_take(a: *mut am_array, indices: *mut am_array, out: *mut *mut am_array) -> c_int;
        pub fn am_slice(
            a: *mut am_array,
            offset: i64,
            length: i64,
            out: *mut *mut am_array,
        ) -> c_int;

        // -- sorting (GPU LSD radix sort; stable, nulls last, NaN after +inf) ---------------------
        /// int32 indices.
        pub fn am_argsort(a: *mut am_array, descending: c_int, out: *mut *mut am_array) -> c_int;
        /// A sorted copy, same type.
        pub fn am_sort(a: *mut am_array, descending: c_int, out: *mut *mut am_array) -> c_int;

        // -- group-by over arbitrary key columns -------------------------------------------------
        pub fn am_group_by_keys(
            columns: *mut *mut am_array,
            count: i64,
            out: *mut *mut am_groupby,
        ) -> c_int;
        /// Number of groups, or -1 for a null handle.
        pub fn am_group_by_group_count(gb: *mut am_groupby) -> i64;
        /// The i-th key column, one row per group, in group order.
        pub fn am_group_by_keys_result(
            gb: *mut am_groupby,
            i: i64,
            out: *mut *mut am_array,
        ) -> c_int;
        /// The dense group id of every row (int32, never null).
        pub fn am_group_by_ids(gb: *mut am_groupby, out: *mut *mut am_array) -> c_int;
        pub fn am_group_by_release(gb: *mut am_groupby);
        // agg op: 0 sum, 1 count_all, 2 count, 3 mean, 4 min, 5 max, ... (the header has the table).
        // `values` may be null only for op 1. `p1` is the quantile for ops 22 and 25.
        pub fn am_group_agg_ex(
            gb: *mut am_groupby,
            values: *mut am_array,
            op: c_int,
            p1: f64,
            out: *mut *mut am_array,
        ) -> c_int;

        // -- the JSON plan runner (docs/ENGINE.md) -----------------------------------------------
        /// Registers one table. The source retains the `am_array` handles it is given.
        pub fn am_plan_source_create(
            name: *const c_char,
            columns: *mut *mut am_array,
            names: *mut *const c_char,
            n_columns: i64,
            out: *mut *mut am_plan_source,
        ) -> c_int;
        pub fn am_plan_source_release(s: *mut am_plan_source);
        pub fn am_plan_run(
            plan_json: *const c_char,
            sources: *mut *mut am_plan_source,
            n_sources: i64,
            optimize: c_int,
            out: *mut *mut am_plan_result,
        ) -> c_int;
        /// The optimized logical plan and the physical plan, as text valid until the next call on
        /// this thread; null plus `am_last_error()` when the plan does not type-check.
        pub fn am_plan_explain(
            plan_json: *const c_char,
            sources: *mut *mut am_plan_source,
            n_sources: i64,
            optimize: c_int,
        ) -> *const c_char;
        pub fn am_plan_column_count(r: *mut am_plan_result) -> i64;
        pub fn am_plan_row_count(r: *mut am_plan_result) -> i64;
        pub fn am_plan_column_name(r: *mut am_plan_result, i: i64) -> *const c_char;
        /// Hands out a new `am_array` handle the caller releases with `am_release`.
        pub fn am_plan_column(r: *mut am_plan_result, i: i64, out: *mut *mut am_array) -> c_int;
        pub fn am_plan_result_release(r: *mut am_plan_result);

        // -- batching ---------------------------------------------------------------------------
        // Between begin and end every call on this thread appends to one GPU command buffer.
        pub fn am_batch_begin() -> c_int;
        pub fn am_batch_end() -> c_int;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two C Data Interface structs must be exactly the sizes the Arrow spec fixes them at, or
    /// every import and export is silently reading the wrong fields.
    #[test]
    fn c_data_interface_layout() {
        assert_eq!(std::mem::size_of::<ArrowSchema>(), 9 * 8);
        assert_eq!(std::mem::size_of::<ArrowArray>(), 10 * 8);
        assert_eq!(std::mem::align_of::<ArrowSchema>(), 8);
        assert_eq!(std::mem::align_of::<ArrowArray>(), 8);
    }

    /// Proves the dylib is linked and loadable at all: the cheapest possible call.
    #[test]
    fn links_and_loads() {
        let v = unsafe { std::ffi::CStr::from_ptr(ffi::am_version()) };
        assert!(!v.to_bytes().is_empty(), "am_version() returned an empty string");
    }
}
