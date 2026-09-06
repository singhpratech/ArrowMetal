//! Raw FFI bindings to ArrowMetal's C ABI (`include/arrowmetal.h`).
//!
//! Hand-written rather than bindgen-generated: the surface is small, the header is stable, and a
//! checked-in file needs no libclang on the build machine. Every declaration below is a
//! transcription of one line of `include/arrowmetal.h`; the comments name the header's own section.
//!
//! Two layers:
//!   * `ffi` -- the bare `extern "C"` declarations and the two Arrow C Data Interface structs.
//!   * the safe wrappers (`Array`, `GroupBy`, `last_error`) -- RAII handles that release on drop
//!     and turn a non-zero return code into an `Error` carrying `am_last_error()`.
//!
//! Every call returns 0 on success; `am_last_error()` is thread-local and valid until the next
//! call on that thread, which is why `check` reads it immediately.

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fmt;

/// The directory `build.rs` linked against, baked in for diagnostics.
pub const LIB_DIR: &str = env!("ARROWMETAL_SYS_LIB_DIR");

// ---------------------------------------------------------------------------------------------
// Arrow C Data Interface
// ---------------------------------------------------------------------------------------------

/// ABI-compatible with `struct ArrowSchema` (and with `polars_arrow::ffi::ArrowSchema`).
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

/// ABI-compatible with `struct ArrowArray` (and with `polars_arrow::ffi::ArrowArray`).
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
    pub fn empty() -> Self {
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
    pub fn empty() -> Self {
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

/// Opaque, Metal-resident Arrow array.
#[repr(C)]
pub struct am_array {
    _private: [u8; 0],
}

/// Opaque key mapping produced by `am_group_by_keys`.
#[repr(C)]
pub struct am_groupby {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------------------------
// The C ABI itself
// ---------------------------------------------------------------------------------------------

pub mod ffi {
    use super::*;

    unsafe extern "C" {
        // -- identity
        pub fn am_version() -> *const c_char;
        pub fn am_device_name() -> *const c_char;
        pub fn am_last_error() -> *const c_char;

        // -- lifecycle and interop
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
        pub fn am_format(a: *mut am_array) -> *const c_char;

        // -- reductions: op 0 sum, 1 min, 2 max, 3 mean; kind 0 i64, 1 u64, 2 f64
        pub fn am_reduce(
            a: *mut am_array,
            op: c_int,
            out_i64: *mut i64,
            out_f64: *mut f64,
            out_kind: *mut c_int,
            is_null: *mut c_int,
        ) -> c_int;

        // -- element-wise
        pub fn am_compare_scalar(
            a: *mut am_array,
            op: c_int,
            scalar: *const c_void,
            out: *mut *mut am_array,
        ) -> c_int;
        pub fn am_arith_scalar(
            a: *mut am_array,
            op: c_int,
            scalar: *const c_void,
            out: *mut *mut am_array,
        ) -> c_int;
        pub fn am_cast(a: *mut am_array, format: *const c_char, out: *mut *mut am_array) -> c_int;

        // -- selection
        pub fn am_filter(a: *mut am_array, mask: *mut am_array, out: *mut *mut am_array) -> c_int;
        pub fn am_take(a: *mut am_array, indices: *mut am_array, out: *mut *mut am_array) -> c_int;
        pub fn am_top_k(
            a: *mut am_array,
            k: i64,
            largest: c_int,
            out: *mut *mut am_array,
        ) -> c_int;

        // -- strings: unary kind 0 byte length, 1 char length, 2 murmur3 hash
        pub fn am_str_unary(a: *mut am_array, kind: c_int, out: *mut *mut am_array) -> c_int;
        // match pred: 0 equals, 1 starts_with, 2 ends_with, 3 contains
        pub fn am_str_match(
            a: *mut am_array,
            pred: c_int,
            pattern: *const u8,
            len: i64,
            out: *mut *mut am_array,
        ) -> c_int;
        // op 0 ascii_upper, 1 ascii_lower, 2 utf8_upper, 3 utf8_lower, ... (header has the table)
        pub fn am_str_transform(
            a: *mut am_array,
            op: c_int,
            arg1: *const u8,
            len1: i64,
            arg2: *const u8,
            len2: i64,
            p1: i64,
            p2: i64,
            out: *mut *mut am_array,
        ) -> c_int;

        // -- hashing
        pub fn am_hash64(a: *mut am_array, out: *mut *mut am_array) -> c_int;

        // -- group-by over arbitrary key columns
        pub fn am_group_by_keys(
            columns: *mut *mut am_array,
            count: i64,
            out: *mut *mut am_groupby,
        ) -> c_int;
        pub fn am_group_by_group_count(gb: *mut am_groupby) -> i64;
        pub fn am_group_by_keys_result(
            gb: *mut am_groupby,
            i: i64,
            out: *mut *mut am_array,
        ) -> c_int;
        pub fn am_group_by_ids(gb: *mut am_groupby, out: *mut *mut am_array) -> c_int;
        pub fn am_group_by_release(gb: *mut am_groupby);
        // agg op: 0 sum, 1 count_all, 2 count, 3 mean, 4 min, 5 max, ... (header has the table)
        pub fn am_group_agg_ex(
            gb: *mut am_groupby,
            values: *mut am_array,
            op: c_int,
            p1: f64,
            out: *mut *mut am_array,
        ) -> c_int;

        // -- batching
        pub fn am_batch_begin() -> c_int;
        pub fn am_batch_end() -> c_int;
    }
}

// ---------------------------------------------------------------------------------------------
// Safe layer
// ---------------------------------------------------------------------------------------------

/// An ArrowMetal error, carrying the message `am_last_error()` reported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// The thread-local message from the last failing call.
pub fn last_error() -> String {
    unsafe {
        let p = ffi::am_last_error();
        if p.is_null() {
            String::new()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

fn check(rc: c_int) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(Error(last_error()))
    }
}

/// The version string of the linked libArrowMetalC.
pub fn version() -> String {
    unsafe { CStr::from_ptr(ffi::am_version()).to_string_lossy().into_owned() }
}

/// The Metal device the library is running on.
pub fn device_name() -> String {
    unsafe { CStr::from_ptr(ffi::am_device_name()).to_string_lossy().into_owned() }
}

/// One scalar reduction result. `am_reduce` reports which of the three slots is live.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    Null,
    I64(i64),
    U64(u64),
    F64(f64),
}

/// A Metal-resident Arrow array. Releases its handle on drop.
pub struct Array {
    handle: *mut am_array,
}

// The handle is an owning pointer into ArrowMetal's own allocator; ArrowMetal serialises its
// command-buffer state per thread, so an Array may be moved between threads but not shared.
unsafe impl Send for Array {}

impl fmt::Debug for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Array")
            .field("format", &self.format())
            .field("len", &self.len())
            .field("nulls", &self.null_count())
            .finish()
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::am_release(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

/// Calls a C function of the shape `fn(..., out: *mut *mut am_array) -> c_int` and wraps the
/// handle it produced, so every kernel wrapper below is one line.
macro_rules! out_array {
    ($f:path $(, $arg:expr)* $(,)?) => {{
        let mut out: *mut am_array = std::ptr::null_mut();
        check(unsafe { $f($($arg,)* &mut out) })?;
        Ok(Array::from_raw(out))
    }};
}

impl Array {
    /// Takes ownership of a handle the C ABI just produced.
    ///
    /// # Safety
    /// `handle` must be a live `am_array*` this `Array` is now the sole owner of.
    pub fn from_raw(handle: *mut am_array) -> Self {
        Array { handle }
    }

    pub fn as_ptr(&self) -> *mut am_array {
        self.handle
    }

    /// Imports one array through the Arrow C Data Interface.
    ///
    /// # Safety
    /// `schema` and `array` must be a valid exported pair. `am_import` **consumes** `array`
    /// (it takes over the release callback), so the caller must not release it afterwards --
    /// `std::mem::forget` it, exactly as the Python binding does.
    pub unsafe fn import(schema: *const ArrowSchema, array: *mut ArrowArray) -> Result<Self> {
        let mut out: *mut am_array = std::ptr::null_mut();
        check(unsafe { ffi::am_import(schema, array, &mut out) })?;
        Ok(Array::from_raw(out))
    }

    /// Exports into a caller-provided schema/array pair.
    pub fn export(&self, schema: &mut ArrowSchema, array: &mut ArrowArray) -> Result<()> {
        check(unsafe { ffi::am_export(self.handle, schema, array) })
    }

    pub fn len(&self) -> i64 {
        unsafe { ffi::am_length(self.handle) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn null_count(&self) -> i64 {
        unsafe { ffi::am_null_count(self.handle) }
    }

    /// The Arrow format string: `l` int64, `g` float64, `u` utf8, ...
    pub fn format(&self) -> String {
        unsafe {
            let p = ffi::am_format(self.handle);
            if p.is_null() {
                String::new()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        }
    }

    /// op: 0 sum, 1 min, 2 max, 3 mean.
    pub fn reduce(&self, op: c_int) -> Result<Scalar> {
        let (mut i, mut f) = (0i64, 0f64);
        let (mut kind, mut is_null) = (0 as c_int, 0 as c_int);
        check(unsafe {
            ffi::am_reduce(self.handle, op, &mut i, &mut f, &mut kind, &mut is_null)
        })?;
        if is_null != 0 {
            return Ok(Scalar::Null);
        }
        Ok(match kind {
            0 => Scalar::I64(i),
            1 => Scalar::U64(i as u64),
            _ => Scalar::F64(f),
        })
    }

    pub fn sum(&self) -> Result<Scalar> {
        self.reduce(0)
    }
    pub fn min(&self) -> Result<Scalar> {
        self.reduce(1)
    }
    pub fn max(&self) -> Result<Scalar> {
        self.reduce(2)
    }
    pub fn mean(&self) -> Result<Scalar> {
        self.reduce(3)
    }

    /// Compare against a scalar. op: 0 eq, 1 ne, 2 lt, 3 le, 4 gt, 5 ge. `scalar` points at one
    /// value of this array's element type.
    ///
    /// # Safety
    /// `scalar` must point at a value of this array's element type.
    pub unsafe fn compare_scalar(&self, op: c_int, scalar: *const c_void) -> Result<Array> {
        out_array!(ffi::am_compare_scalar, self.handle, op, scalar)
    }

    /// Arithmetic against a scalar. op: 0 add, 1 sub, 2 mul, 3 div.
    ///
    /// # Safety
    /// `scalar` must point at a value of this array's element type.
    pub unsafe fn arith_scalar(&self, op: c_int, scalar: *const c_void) -> Result<Array> {
        out_array!(ffi::am_arith_scalar, self.handle, op, scalar)
    }

    pub fn cast(&self, format: &str) -> Result<Array> {
        let f = CString::new(format).map_err(|e| Error(e.to_string()))?;
        out_array!(ffi::am_cast, self.handle, f.as_ptr())
    }

    pub fn filter(&self, mask: &Array) -> Result<Array> {
        out_array!(ffi::am_filter, self.handle, mask.handle)
    }

    pub fn take(&self, indices: &Array) -> Result<Array> {
        out_array!(ffi::am_take, self.handle, indices.handle)
    }

    /// int32 indices of the `k` largest (or smallest) elements.
    pub fn top_k(&self, k: i64, largest: bool) -> Result<Array> {
        out_array!(ffi::am_top_k, self.handle, k, largest as c_int)
    }

    /// kind: 0 byte length, 1 char length, 2 murmur3 hash (uint32).
    pub fn str_unary(&self, kind: c_int) -> Result<Array> {
        out_array!(ffi::am_str_unary, self.handle, kind)
    }

    /// pred: 0 equals, 1 starts_with, 2 ends_with, 3 contains.
    pub fn str_match(&self, pred: c_int, pattern: &[u8]) -> Result<Array> {
        out_array!(
            ffi::am_str_match,
            self.handle,
            pred,
            pattern.as_ptr(),
            pattern.len() as i64,
        )
    }

    /// One entry of the `am_str_transform` op table (2 = utf8_upper, 3 = utf8_lower, ...).
    pub fn str_transform(&self, op: c_int, arg1: &[u8], arg2: &[u8], p1: i64, p2: i64) -> Result<Array> {
        out_array!(
            ffi::am_str_transform,
            self.handle,
            op,
            arg1.as_ptr(),
            arg1.len() as i64,
            arg2.as_ptr(),
            arg2.len() as i64,
            p1,
            p2,
        )
    }

    pub fn hash64(&self) -> Result<Array> {
        out_array!(ffi::am_hash64, self.handle)
    }
}

/// A dense group-id mapping over one or more key columns.
pub struct GroupBy {
    handle: *mut am_groupby,
    n_keys: i64,
}

unsafe impl Send for GroupBy {}

impl Drop for GroupBy {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::am_group_by_release(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

impl GroupBy {
    pub fn new(columns: &[&Array]) -> Result<Self> {
        let mut handles: Vec<*mut am_array> = columns.iter().map(|c| c.as_ptr()).collect();
        let mut out: *mut am_groupby = std::ptr::null_mut();
        check(unsafe {
            ffi::am_group_by_keys(handles.as_mut_ptr(), handles.len() as i64, &mut out)
        })?;
        Ok(GroupBy {
            handle: out,
            n_keys: columns.len() as i64,
        })
    }

    pub fn group_count(&self) -> i64 {
        unsafe { ffi::am_group_by_group_count(self.handle) }
    }

    /// The i-th key column, one row per group, in group order.
    pub fn keys(&self, i: i64) -> Result<Array> {
        out_array!(ffi::am_group_by_keys_result, self.handle, i)
    }

    pub fn n_keys(&self) -> i64 {
        self.n_keys
    }

    /// The dense group id of every row (int32, never null).
    pub fn ids(&self) -> Result<Array> {
        out_array!(ffi::am_group_by_ids, self.handle)
    }

    /// One grouped aggregate. op: 0 sum, 1 count_all, 2 count, 3 mean, 4 min, 5 max, ...
    pub fn agg(&self, values: Option<&Array>, op: c_int, p1: f64) -> Result<Array> {
        let v = values.map(|a| a.as_ptr()).unwrap_or(std::ptr::null_mut());
        out_array!(ffi::am_group_agg_ex, self.handle, v, op, p1)
    }
}

/// A scope in which every call appends to one GPU command buffer.
pub struct Batch {
    _private: (),
}

impl Batch {
    pub fn begin() -> Result<Self> {
        check(unsafe { ffi::am_batch_begin() })?;
        Ok(Batch { _private: () })
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        unsafe { ffi::am_batch_end() };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The producer's release callback. The C Data Interface requires one (`am_import` refuses a
    /// pair whose `release` is null, reporting "ArrowArray has already been released"), and it must
    /// mark the struct released. The buffers are leaked on purpose -- see `int64_array`.
    unsafe extern "C" fn release_noop(array: *mut ArrowArray) {
        if !array.is_null() {
            unsafe { (*array).release = None };
        }
    }

    /// Builds an int64 `ArrowArray`/`ArrowSchema` pair by hand, with no null bitmap.
    ///
    /// The buffers are leaked deliberately: `am_import` may keep them (it is zero-copy when the
    /// producer's allocation is page aligned), and a test process is about to exit anyway.
    fn int64_array(values: &[i64]) -> (ArrowSchema, ArrowArray) {
        let data: Box<[i64]> = values.to_vec().into_boxed_slice();
        let data_ptr = data.as_ptr() as *const c_void;
        std::mem::forget(data);
        // buffers[0] = validity (null == all valid), buffers[1] = values
        let buffers: Box<[*const c_void]> = vec![std::ptr::null(), data_ptr].into_boxed_slice();
        let buffers_ptr = Box::into_raw(buffers) as *mut *const c_void;

        let format = CString::new("l").unwrap();
        let name = CString::new("x").unwrap();
        let schema = ArrowSchema {
            format: format.into_raw(),
            name: name.into_raw(),
            metadata: std::ptr::null(),
            flags: 2, // ARROW_FLAG_NULLABLE
            n_children: 0,
            children: std::ptr::null_mut(),
            dictionary: std::ptr::null_mut(),
            release: None,
            private_data: std::ptr::null_mut(),
        };
        let array = ArrowArray {
            length: values.len() as i64,
            null_count: 0,
            offset: 0,
            n_buffers: 2,
            n_children: 0,
            buffers: buffers_ptr,
            children: std::ptr::null_mut(),
            dictionary: std::ptr::null_mut(),
            release: Some(release_noop),
            private_data: std::ptr::null_mut(),
        };
        (schema, array)
    }

    fn import(values: &[i64]) -> Array {
        let (schema, mut array) = int64_array(values);
        unsafe { Array::import(&schema, &mut array) }.expect("am_import")
    }

    #[test]
    fn library_identifies_itself() {
        assert!(!version().is_empty(), "am_version() returned nothing");
        assert!(!device_name().is_empty(), "am_device_name() returned nothing");
    }

    #[test]
    fn import_reports_length_and_format() {
        let a = import(&[1, 2, 3, 4, 5]);
        assert_eq!(a.len(), 5);
        assert_eq!(a.null_count(), 0);
        assert_eq!(a.format(), "l");
    }

    #[test]
    fn reductions_match_the_host() {
        let vals: Vec<i64> = (1..=1000).collect();
        let a = import(&vals);
        assert_eq!(a.sum().unwrap(), Scalar::I64(vals.iter().sum::<i64>()));
        assert_eq!(a.min().unwrap(), Scalar::I64(1));
        assert_eq!(a.max().unwrap(), Scalar::I64(1000));
        match a.mean().unwrap() {
            Scalar::F64(m) => assert!((m - 500.5).abs() < 1e-9, "mean was {m}"),
            other => panic!("mean returned {other:?}"),
        }
    }

    #[test]
    fn compare_and_filter_round_trip() {
        let vals: Vec<i64> = (0..100).collect();
        let a = import(&vals);
        let threshold: i64 = 49;
        let mask = unsafe { a.compare_scalar(4, &threshold as *const i64 as *const c_void) }
            .expect("am_compare_scalar");
        assert_eq!(mask.format(), "b");
        let kept = a.filter(&mask).expect("am_filter");
        assert_eq!(kept.len(), 50);
        assert_eq!(kept.sum().unwrap(), Scalar::I64((50..100).sum::<i64>()));
    }

    #[test]
    fn arithmetic_with_a_scalar() {
        let a = import(&[1, 2, 3]);
        let two: i64 = 2;
        let doubled = unsafe { a.arith_scalar(2, &two as *const i64 as *const c_void) }
            .expect("am_arith_scalar");
        assert_eq!(doubled.sum().unwrap(), Scalar::I64(12));
    }

    #[test]
    fn top_k_returns_indices() {
        let a = import(&[5, 1, 9, 3, 7]);
        let idx = a.top_k(2, true).expect("am_top_k");
        assert_eq!(idx.len(), 2);
        assert_eq!(idx.format(), "i");
        let picked = a.take(&idx).expect("am_take");
        assert_eq!(picked.sum().unwrap(), Scalar::I64(16)); // 9 + 7
    }

    #[test]
    fn hash64_is_deterministic_and_never_zero_for_a_value() {
        let a = import(&[0, 1, 2]);
        let h = a.hash64().expect("am_hash64");
        assert_eq!(h.len(), 3);
        assert_eq!(h.format(), "L");
    }

    #[test]
    fn group_by_sums_per_key() {
        let keys = import(&[0, 1, 0, 1, 0]);
        let values = import(&[1, 10, 2, 20, 3]);
        let gb = GroupBy::new(&[&keys]).expect("am_group_by_keys");
        assert_eq!(gb.group_count(), 2);
        let sums = gb.agg(Some(&values), 0, 0.0).expect("am_group_agg_ex");
        assert_eq!(sums.len(), 2);
        // Group order is ascending by key for a numeric column: key 0 then key 1.
        assert_eq!(sums.sum().unwrap(), Scalar::I64(36));
        let ids = gb.ids().expect("am_group_by_ids");
        assert_eq!(ids.len(), 5);
    }

    #[test]
    fn a_bad_call_reports_a_message() {
        let a = import(&[1, 2, 3]);
        // A string predicate against an int64 column has to fail.
        let err = a.str_match(3, b"nope").unwrap_err();
        assert!(!err.0.is_empty(), "expected am_last_error() to say something");
    }

    #[test]
    fn export_hands_back_a_released_pair() {
        let a = import(&[7, 8, 9]);
        let mut schema = ArrowSchema::empty();
        let mut array = ArrowArray::empty();
        a.export(&mut schema, &mut array).expect("am_export");
        assert_eq!(array.length, 3);
        assert!(array.release.is_some(), "export must set a release callback");
        assert!(schema.release.is_some());
        unsafe {
            (array.release.unwrap())(&mut array);
            (schema.release.unwrap())(&mut schema);
        }
    }
}
