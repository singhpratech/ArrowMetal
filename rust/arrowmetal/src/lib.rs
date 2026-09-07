//! Apache Arrow compute on Apple silicon GPUs, for [arrow-rs].
//!
//! ArrowMetal is a Swift library that runs Arrow's compute kernels as Metal shaders on the GPU of an
//! Apple silicon Mac. This crate is a safe Rust binding over its C ABI. It speaks the [Arrow C Data
//! Interface][cdi], so an [`arrow::array::ArrayRef`] goes in and an `ArrayRef` comes back out; the two
//! libraries share the same buffers wherever the alignment allows it.
//!
//! ```no_run
//! use std::sync::Arc;
//! use arrow::array::{ArrayRef, Int64Array};
//!
//! let values: ArrayRef = Arc::new(Int64Array::from(vec![Some(3), None, Some(4)]));
//! let gpu = arrowmetal::Array::from_arrow(&values)?;
//! assert_eq!(gpu.sum()?, Some(arrowmetal::Scalar::Int64(7)));
//! # Ok::<(), arrowmetal::Error>(())
//! ```
//!
//! # Requirements
//!
//! An Apple silicon Mac and `libArrowMetalC.dylib`. Point `ARROWMETAL_LIB` at the dylib before
//! building, or leave it in `<repo>/.build/release`; `docs/RUST.md` has the details.
//!
//! # The copy rule
//!
//! **Copy-free out always; copy-free in when the producer's buffers are page aligned, one copy
//! otherwise.**
//!
//! Going out ([`Array::to_arrow`]) is always copy-free: ArrowMetal's own buffers are `MTLBuffer`s in
//! shared memory, and it hands their pointers straight to the C Data Interface with a release
//! callback, so arrow-rs reads the GPU's memory in place. Measured, values **and** validity bitmap:
//! every exported buffer was page aligned at every size tested.
//!
//! Going in ([`Array::from_arrow`]) is copy-free only when a buffer pointer is page aligned, because
//! that is what `MTLDevice.makeBuffer(bytesNoCopy:)` requires; otherwise ArrowMetal copies that
//! buffer once. A page here is 16 KiB on Apple silicon. The decision is **per buffer**, so a
//! nullable column can have its values wrapped and its bitmap copied.
//!
//! Whether an arrow-rs array clears that bar is a property of the *allocator*, not of arrow-rs, so
//! it is measured rather than assumed. `tests/copy_rule.rs` exports through `arrow::ffi` and reads
//! the pointers `am_import` actually receives, 32 allocations at each of six sizes; on
//! macOS 26.6.2 / arm64 with the system allocator, in this repository's run:
//!
//! | Column length | Values buffer page aligned | Validity bitmap page aligned |
//! |---|---|---|
//! | 16 | 1/32 | 0/32 |
//! | 512 | 8/32 | 0/32 |
//! | 8,192 | 32/32 | 2/32 |
//! | 131,072 and above | 32/32 | 32/32 |
//!
//! So in practice: **values are wrapped from a few thousand rows up; a nullable column of fewer than
//! roughly 130,000 rows normally has its validity bitmap copied.** That copy is small — a bitmap is
//! one bit per row, so about 16 KB at the point it stops happening — but it is a copy, and the
//! earlier version of this note missed it entirely by measuring `ArrayData::buffers()`, which
//! excludes the null buffer.
//!
//! The pattern is just the system allocator handing back page-aligned memory once a request is large
//! enough to be served by `mmap`; the bitmap is eight times smaller than the values, so it crosses
//! that threshold eight times later. None of it is guaranteed — a custom global allocator or another
//! platform can change it. Re-run `cargo test --test copy_rule -- --nocapture` for your own machine.
//!
//! An array that came *out* of ArrowMetal and is sent back in is recognised and re-imported without
//! a copy in either buffer, whatever the original alignment was.
//!
//! Neither direction copies for a slice: an Arrow `offset` is carried on the handle rather than
//! applied, so `array.slice(7, n)` imports the same buffers the unsliced array would.
//!
//! # Threading
//!
//! [`Array`], [`GroupBy`], [`Source`] and [`PlanResult`] are `!Send` and `!Sync` on purpose. The C
//! ABI's error slot (`am_last_error`) and its command-buffer batching are both thread-local, so a
//! handle belongs to the thread that made it. Use one set of handles per thread.
//!
//! [arrow-rs]: https://docs.rs/arrow
//! [cdi]: https://arrow.apache.org/docs/format/CDataInterface.html

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

use std::cell::{Cell, RefCell};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fmt;
use std::marker::PhantomData;
use std::ptr::NonNull;

use arrow::array::{make_array, Array as ArrowArrayTrait, ArrayRef};
use arrow::datatypes::DataType;
use arrow::ffi::{from_ffi, to_ffi, FFI_ArrowArray, FFI_ArrowSchema};

use arrowmetal_sys as sys;
use sys::ffi;

/// The directory `libArrowMetalC.dylib` was linked from, baked in at build time.
///
/// This is the resolved output of the search in `arrowmetal-sys/build.rs`, not the `ARROWMETAL_LIB`
/// or `ARROWMETAL_LIB_DIR` you may have set to steer it.
pub const LIB_DIR: &str = env!("ARROWMETAL_LINKED_LIB_DIR");

// =================================================================================================
// Errors
// =================================================================================================

/// An error from ArrowMetal, from arrow-rs, or from this crate's own type checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    message: String,
}

impl Error {
    /// The message ArrowMetal reported.
    pub fn message(&self) -> &str {
        &self.message
    }

    fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }

    /// Reads `am_last_error()`, which is thread-local and valid only until the next call on this
    /// thread -- so this runs immediately after the failing call and copies the string.
    fn last(context: &str) -> Self {
        let raw = unsafe { ffi::am_last_error() };
        let detail = if raw.is_null() {
            String::from("(no message)")
        } else {
            unsafe { CStr::from_ptr(raw) }.to_string_lossy().into_owned()
        };
        Self::new(format!("{context}: {detail}"))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

impl From<arrow::error::ArrowError> for Error {
    fn from(e: arrow::error::ArrowError) -> Self {
        Self::new(format!("arrow: {e}"))
    }
}

/// The result type every fallible call in this crate returns.
pub type Result<T> = std::result::Result<T, Error>;

/// Turns the ABI's `0 = ok` convention into a `Result`, capturing `am_last_error()` on the spot.
fn check(code: c_int, context: &str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(Error::last(context))
    }
}

/// A `NUL`-terminated copy of `s`, rejecting an interior `NUL` rather than truncating.
fn cstring(s: &str, what: &str) -> Result<CString> {
    CString::new(s).map_err(|_| Error::new(format!("{what} contains an interior NUL byte")))
}

// =================================================================================================
// Identity
// =================================================================================================

/// ArrowMetal's version string, e.g. `"0.1.0"`.
pub fn version() -> &'static str {
    unsafe { CStr::from_ptr(ffi::am_version()) }.to_str().unwrap_or("unknown")
}

/// The Metal device the kernels run on, e.g. `"Apple M4 Max"`.
pub fn device_name() -> String {
    unsafe { CStr::from_ptr(ffi::am_device_name()) }.to_string_lossy().into_owned()
}

// =================================================================================================
// Element types
// =================================================================================================

mod sealed {
    pub trait Sealed {}
}

/// A Rust type that can be handed to ArrowMetal as a scalar operand.
///
/// The C ABI reads `sizeof(element type)` bytes through a `void*`, so passing an `i32` scalar to an
/// `Int64Array` would read four bytes past the end of the value. Every entry point on this trait
/// therefore checks [`FORMAT`](NativeType::FORMAT) against the array's own Arrow format string first
/// and returns an error on a mismatch.
///
/// That check is only as good as `am_format`, and there is one Arrow type for which `am_format`
/// does **not** describe what the kernels compute on: a dictionary-encoded array reports its *index*
/// type (`"i"`) while every kernel decodes the dictionary and operates on the *value* type. A
/// 4-byte scalar would then be accepted for a `Dictionary(Int32, Float64)` column whose kernel reads
/// 8 bytes. [`Array::from_arrow`] therefore refuses dictionary arrays outright, which is what keeps
/// the sentence above true for everything this crate accepts. See [`Array::from_arrow`].
///
/// # Do not reopen this without fixing `am_format` first
///
/// Refusing at [`Array::from_arrow`] works only because that is the **sole** way an [`Array`] handle
/// is made from outside. A dictionary can still sit *inside* an accepted array — a
/// `Struct{d: Dictionary(Int32, Float64)}` imports fine — and is harmless today only because nothing
/// in this crate can pull the child out into a handle of its own: the struct's own `am_format` is
/// `"+s"`, which no `NativeType` matches, so every scalar entry point refuses it.
///
/// Wrapping any of these ABI entry points would hand back a bare dictionary handle reporting `"i"`
/// and reopen the hole, so each needs `am_format` fixed first (or its own guard):
///
/// * `am_child` — child `i` of a struct, list, map or dictionary.
/// * `am_struct_field` — a struct field by name.
/// * `am_dictionary_decode` — materialises a dictionary; its *output* is safe, but it takes a
///   dictionary handle, which today cannot be built.
/// * `am_list_flatten` — a list's child values, which may themselves be dictionary-encoded.
/// * `am_cast_ex` — casts *to* a dictionary type via its child-format argument.
///
/// `tests/compute.rs::a_nested_dictionary_is_unreachable_from_the_safe_surface` pins the property
/// this argument rests on.
pub trait NativeType: Copy + sealed::Sealed {
    /// The Arrow C Data Interface format string for this type.
    const FORMAT: &'static str;
}

macro_rules! native {
    ($($t:ty => $f:literal),* $(,)?) => {$(
        impl sealed::Sealed for $t {}
        impl NativeType for $t {
            const FORMAT: &'static str = $f;
        }
    )*};
}

native! {
    i8 => "c", u8 => "C",
    i16 => "s", u16 => "S",
    i32 => "i", u32 => "I",
    i64 => "l", u64 => "L",
    f32 => "f", f64 => "g",
}

/// The value a reduction came back with. Which variant you get follows Arrow's own rules: a signed
/// integer column sums to `Int64`, an unsigned one to `UInt64`, a float column to `Float64`, and
/// `mean` is always `Float64`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    /// A signed 64-bit result.
    Int64(i64),
    /// An unsigned 64-bit result.
    UInt64(u64),
    /// A 64-bit float result.
    Float64(f64),
}

impl Scalar {
    /// The value widened to `f64`, whichever variant it is.
    pub fn as_f64(self) -> f64 {
        match self {
            Scalar::Int64(v) => v as f64,
            Scalar::UInt64(v) => v as f64,
            Scalar::Float64(v) => v,
        }
    }

    /// The value if it is an `Int64`, otherwise `None`.
    pub fn as_i64(self) -> Option<i64> {
        match self {
            Scalar::Int64(v) => Some(v),
            _ => None,
        }
    }
}

/// The comparison [`Array::compare_scalar`] and [`Array::compare`] apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    /// `a == b`
    Eq,
    /// `a != b`
    Ne,
    /// `a < b`
    Lt,
    /// `a <= b`
    Le,
    /// `a > b`
    Gt,
    /// `a >= b`
    Ge,
}

impl CompareOp {
    fn code(self) -> c_int {
        match self {
            CompareOp::Eq => 0,
            CompareOp::Ne => 1,
            CompareOp::Lt => 2,
            CompareOp::Le => 3,
            CompareOp::Gt => 4,
            CompareOp::Ge => 5,
        }
    }
}

// =================================================================================================
// Array
// =================================================================================================

/// A Metal-resident Arrow array.
///
/// Made with [`Array::from_arrow`], read back with [`Array::to_arrow`], and released on drop. `!Send`
/// and `!Sync`; see the [threading](crate#threading) note.
pub struct Array {
    ptr: NonNull<sys::am_array>,
    /// The raw pointer already makes this `!Send`/`!Sync`; this states the intent for a reader.
    _not_send: PhantomData<*const ()>,
}

impl Drop for Array {
    fn drop(&mut self) {
        unsafe { ffi::am_release(self.ptr.as_ptr()) }
    }
}

impl fmt::Debug for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("arrowmetal::Array")
            .field("len", &self.len())
            .field("null_count", &self.null_count())
            .field("format", &self.format())
            .finish()
    }
}

impl Array {
    /// Wraps a handle an ABI call just produced, taking ownership of it.
    ///
    /// # Safety
    /// `ptr` must be a live `am_array*` this `Array` becomes the sole owner of.
    unsafe fn from_raw(ptr: *mut sys::am_array, context: &str) -> Result<Self> {
        match NonNull::new(ptr) {
            Some(ptr) => Ok(Self { ptr, _not_send: PhantomData }),
            None => Err(Error::new(format!("{context}: succeeded but returned a null handle"))),
        }
    }

    /// Runs a call of the common `(a, .., out) -> int` shape and wraps the handle it wrote.
    fn produce(context: &str, call: impl FnOnce(*mut *mut sys::am_array) -> c_int) -> Result<Self> {
        let mut out: *mut sys::am_array = std::ptr::null_mut();
        check(call(&mut out), context)?;
        unsafe { Self::from_raw(out, context) }
    }

    fn as_ptr(&self) -> *mut sys::am_array {
        self.ptr.as_ptr()
    }

    // -- interop ----------------------------------------------------------------------------------

    /// Imports an arrow-rs array through the C Data Interface.
    ///
    /// Copy-free when the array's buffers are page aligned, one copy per buffer otherwise; see [the
    /// copy rule](crate#the-copy-rule). A sliced array (`offset != 0`) imports without copying: the
    /// offset rides on the handle.
    ///
    /// # Dictionary arrays are refused
    ///
    /// A `DataType::Dictionary` array is rejected with an error rather than imported. The ABI's
    /// `am_format` reports a dictionary's *index* type (`"i"`), but every ArrowMetal kernel decodes
    /// the dictionary first and computes on the *value* type. This crate type-checks scalar operands
    /// against `am_format`, so importing one would let a 4-byte `i32` scalar through to a kernel
    /// reading 8 bytes off a `Dictionary(Int32, Float64)` column — an out-of-bounds read reachable
    /// from safe Rust, and a wrong answer besides.
    ///
    /// This is a limitation of the C ABI, not of the Arrow type: the kernels themselves handle
    /// dictionaries correctly. Until `am_format` reports the compute type, decode first:
    ///
    /// ```no_run
    /// # use arrow::array::{ArrayRef, Int64Array};
    /// # use arrow::datatypes::DataType;
    /// # let dict: ArrayRef = std::sync::Arc::new(Int64Array::from(vec![1i64]));
    /// let decoded = arrow::compute::cast(dict.as_ref(), &DataType::Float64)?;
    /// let gpu = arrowmetal::Array::from_arrow(decoded.as_ref())?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// ```no_run
    /// use arrow::array::Int64Array;
    /// let a = Int64Array::from(vec![1i64, 2, 3]);
    /// let gpu = arrowmetal::Array::from_arrow(&a)?;
    /// assert_eq!(gpu.len(), 3);
    /// # Ok::<(), arrowmetal::Error>(())
    /// ```
    pub fn from_arrow(array: &dyn ArrowArrayTrait) -> Result<Self> {
        // See the section above: `am_format` would lie about this array's element type, and the
        // scalar type check that keeps `compare_scalar` and friends sound is built on `am_format`.
        if let DataType::Dictionary(key, value) = array.data_type() {
            return Err(Error::new(format!(
                "dictionary-encoded arrays are not accepted (this array is \
                 Dictionary({key}, {value})): ArrowMetal's am_format reports a dictionary's index \
                 type while its kernels compute on the value type, so a scalar operand cannot be \
                 type-checked against it. Decode first, e.g. \
                 `arrow::compute::cast(&array, &DataType::{value})`."
            )));
        }
        let (ffi_array, ffi_schema) = to_ffi(&array.to_data())?;
        // `am_import` moves the array: on success it nulls `release` in our struct, so the drop
        // below is a no-op. On failure `release` may still be set, and the drop frees the export.
        // Either way, letting it drop normally is exactly right -- do not `mem::forget` it.
        let mut ffi_array = ffi_array;
        Self::produce("am_import", |out| unsafe {
            ffi::am_import(
                (&ffi_schema as *const FFI_ArrowSchema).cast::<sys::ArrowSchema>(),
                (&mut ffi_array as *mut FFI_ArrowArray).cast::<sys::ArrowArray>(),
                out,
            )
        })
    }

    /// Exports back to arrow-rs through the C Data Interface. Always copy-free: arrow-rs reads the
    /// Metal buffers in place and releases them when the last reference to them goes.
    pub fn to_arrow(&self) -> Result<ArrayRef> {
        let mut schema = FFI_ArrowSchema::empty();
        let mut array = FFI_ArrowArray::empty();
        check(
            unsafe {
                ffi::am_export(
                    self.as_ptr(),
                    (&mut schema as *mut FFI_ArrowSchema).cast::<sys::ArrowSchema>(),
                    (&mut array as *mut FFI_ArrowArray).cast::<sys::ArrowArray>(),
                )
            },
            "am_export",
        )?;
        // SAFETY: `am_export` filled both structs per the C Data Interface and handed us ownership
        // of the array; `from_ffi` consumes it and takes over the release callback.
        let data = unsafe { from_ffi(array, &schema) }?;
        Ok(make_array(data))
    }

    // -- metadata ---------------------------------------------------------------------------------

    /// The number of elements.
    pub fn len(&self) -> usize {
        let n = unsafe { ffi::am_length(self.as_ptr()) };
        if n < 0 {
            0
        } else {
            n as usize
        }
    }

    /// Whether the array holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The number of nulls.
    pub fn null_count(&self) -> usize {
        let n = unsafe { ffi::am_null_count(self.as_ptr()) };
        if n < 0 {
            0
        } else {
            n as usize
        }
    }

    /// The Arrow C Data Interface format string, e.g. `"l"` for int64 or `"g"` for float64.
    pub fn format(&self) -> String {
        let raw = unsafe { ffi::am_format(self.as_ptr()) };
        if raw.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(raw) }.to_string_lossy().into_owned()
        }
    }

    /// Errors unless this array's element type is exactly `T`, so that a scalar pointer handed to
    /// the ABI is read at the right width.
    fn require_format<T: NativeType>(&self, what: &str) -> Result<()> {
        let got = self.format();
        if got == T::FORMAT {
            Ok(())
        } else {
            Err(Error::new(format!(
                "{what}: scalar type '{}' does not match the array's element type '{got}'",
                T::FORMAT
            )))
        }
    }

    // -- reductions -------------------------------------------------------------------------------

    /// `sum`. `None` when there is no valid value (an empty or all-null array).
    pub fn sum(&self) -> Result<Option<Scalar>> {
        self.reduce(0, "am_reduce(sum)")
    }

    /// `min`, skipping nulls and NaN. `None` when there is no valid value.
    pub fn min(&self) -> Result<Option<Scalar>> {
        self.reduce(1, "am_reduce(min)")
    }

    /// `max`, skipping nulls and NaN. `None` when there is no valid value.
    pub fn max(&self) -> Result<Option<Scalar>> {
        self.reduce(2, "am_reduce(max)")
    }

    /// `mean`, always a [`Scalar::Float64`]. `None` when there is no valid value.
    pub fn mean(&self) -> Result<Option<Scalar>> {
        self.reduce(3, "am_reduce(mean)")
    }

    fn reduce(&self, op: c_int, context: &str) -> Result<Option<Scalar>> {
        let mut out_i64: i64 = 0;
        let mut out_f64: f64 = 0.0;
        let mut kind: c_int = -1;
        let mut is_null: c_int = 0;
        check(
            unsafe {
                ffi::am_reduce(
                    self.as_ptr(),
                    op,
                    &mut out_i64,
                    &mut out_f64,
                    &mut kind,
                    &mut is_null,
                )
            },
            context,
        )?;
        if is_null != 0 {
            return Ok(None);
        }
        Ok(Some(match kind {
            0 => Scalar::Int64(out_i64),
            // The header shares one slot: an unsigned result arrives in `out_i64`'s bits.
            1 => Scalar::UInt64(out_i64 as u64),
            2 => Scalar::Float64(out_f64),
            other => {
                return Err(Error::new(format!("{context}: unknown out_kind {other}")));
            }
        }))
    }

    // -- element-wise -----------------------------------------------------------------------------

    /// Compares every element against `scalar`, giving a boolean array. A null element gives a null.
    ///
    /// `T` must be exactly the array's element type; anything else is an error, not a wrong answer.
    pub fn compare_scalar<T: NativeType>(&self, op: CompareOp, scalar: T) -> Result<Array> {
        self.require_format::<T>("compare_scalar")?;
        let scalar = scalar;
        Self::produce("am_compare_scalar", |out| unsafe {
            ffi::am_compare_scalar(
                self.as_ptr(),
                op.code(),
                (&scalar as *const T).cast::<c_void>(),
                out,
            )
        })
    }

    /// Compares element for element against `other`, giving a boolean array. A null on either side
    /// gives a null.
    pub fn compare(&self, op: CompareOp, other: &Array) -> Result<Array> {
        Self::produce("am_compare_array", |out| unsafe {
            ffi::am_compare_array(self.as_ptr(), op.code(), other.as_ptr(), out)
        })
    }

    /// Casts to another Arrow type, named by its C Data Interface format string (`"l"`, `"g"`, ...).
    pub fn cast(&self, format: &str) -> Result<Array> {
        let format = cstring(format, "cast format")?;
        Self::produce("am_cast", |out| unsafe {
            ffi::am_cast(self.as_ptr(), format.as_ptr(), out)
        })
    }

    // -- selection --------------------------------------------------------------------------------

    /// Keeps the elements where `mask` is true. A null in the mask drops the element, as Arrow's
    /// `filter` with the default null selection behaviour does.
    pub fn filter(&self, mask: &Array) -> Result<Array> {
        Self::produce("am_filter", |out| unsafe {
            ffi::am_filter(self.as_ptr(), mask.as_ptr(), out)
        })
    }

    /// Gathers by index. `indices` is an int32 array; a null index gives a null output element.
    pub fn take(&self, indices: &Array) -> Result<Array> {
        Self::produce("am_take", |out| unsafe {
            ffi::am_take(self.as_ptr(), indices.as_ptr(), out)
        })
    }

    /// A `length`-element view starting at `offset`.
    pub fn slice(&self, offset: usize, length: usize) -> Result<Array> {
        Self::produce("am_slice", |out| unsafe {
            ffi::am_slice(self.as_ptr(), offset as i64, length as i64, out)
        })
    }

    // -- sorting ----------------------------------------------------------------------------------

    /// The int32 indices that would sort the array. Stable, nulls last, NaN after `+inf`; nulls and
    /// NaN stay at the end when `descending` is set rather than mirroring to the front.
    pub fn argsort(&self, descending: bool) -> Result<Array> {
        Self::produce("am_argsort", |out| unsafe {
            ffi::am_argsort(self.as_ptr(), descending as c_int, out)
        })
    }

    /// A sorted copy, same type, same ordering rules as [`argsort`](Array::argsort).
    pub fn sort(&self, descending: bool) -> Result<Array> {
        Self::produce("am_sort", |out| unsafe {
            ffi::am_sort(self.as_ptr(), descending as c_int, out)
        })
    }
}

// =================================================================================================
// Group-by
// =================================================================================================

/// A grouped aggregate, from the header's `am_group_agg_ex` table.
///
/// Only the aggregates this crate has a test for are listed. The ABI has 26; the rest are reachable
/// through [`GroupBy::agg_raw`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Agg {
    /// `hash_sum`. Integers wrap in 64 bits.
    Sum,
    /// `hash_count_all`: rows per group, nulls included. Needs no values column.
    CountAll,
    /// `hash_count`: non-null values per group.
    Count,
    /// `hash_mean`, always float64.
    Mean,
    /// `hash_min`, NaN skipped.
    Min,
    /// `hash_max`, NaN skipped.
    Max,
}

impl Agg {
    fn code(self) -> c_int {
        match self {
            Agg::Sum => 0,
            Agg::CountAll => 1,
            Agg::Count => 2,
            Agg::Mean => 3,
            Agg::Min => 4,
            Agg::Max => 5,
        }
    }
}

/// Dense group ids for one or more key columns, plus the key values per group.
///
/// Built by [`group_by`]. Group order is deterministic but is **not** first-seen order: it is
/// ascending by key for numeric, boolean, temporal and decimal columns (nulls last), first-seen for
/// utf8 and binary, and lexicographic in column order for several columns. Read the labels back with
/// [`GroupBy::keys`] rather than assuming an order.
pub struct GroupBy {
    ptr: NonNull<sys::am_groupby>,
    n_keys: usize,
    _not_send: PhantomData<*const ()>,
}

impl Drop for GroupBy {
    fn drop(&mut self) {
        unsafe { ffi::am_group_by_release(self.ptr.as_ptr()) }
    }
}

impl fmt::Debug for GroupBy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("arrowmetal::GroupBy")
            .field("groups", &self.group_count())
            .field("key_columns", &self.n_keys)
            .finish()
    }
}

/// Maps one or more key columns to dense group ids on the GPU.
///
/// A null key is not skipped: it forms its own group, as Arrow's hash aggregation does.
pub fn group_by(keys: &[&Array]) -> Result<GroupBy> {
    if keys.is_empty() {
        return Err(Error::new("group_by: needs at least one key column"));
    }
    let mut raw: Vec<*mut sys::am_array> = keys.iter().map(|k| k.as_ptr()).collect();
    let mut out: *mut sys::am_groupby = std::ptr::null_mut();
    check(
        unsafe { ffi::am_group_by_keys(raw.as_mut_ptr(), raw.len() as i64, &mut out) },
        "am_group_by_keys",
    )?;
    match NonNull::new(out) {
        Some(ptr) => Ok(GroupBy { ptr, n_keys: keys.len(), _not_send: PhantomData }),
        None => Err(Error::new("am_group_by_keys: succeeded but returned a null handle")),
    }
}

impl GroupBy {
    /// The number of groups.
    pub fn group_count(&self) -> usize {
        let n = unsafe { ffi::am_group_by_group_count(self.ptr.as_ptr()) };
        if n < 0 {
            0
        } else {
            n as usize
        }
    }

    /// How many key columns this grouping was built from.
    pub fn key_column_count(&self) -> usize {
        self.n_keys
    }

    /// The `i`-th key column, one row per group, in group order, with the input column's type.
    pub fn keys(&self, i: usize) -> Result<Array> {
        if i >= self.n_keys {
            return Err(Error::new(format!(
                "keys({i}): this grouping has {} key columns",
                self.n_keys
            )));
        }
        Array::produce("am_group_by_keys_result", |out| unsafe {
            ffi::am_group_by_keys_result(self.ptr.as_ptr(), i as i64, out)
        })
    }

    /// The dense group id of every input row (int32, never null).
    pub fn ids(&self) -> Result<Array> {
        Array::produce("am_group_by_ids", |out| unsafe {
            ffi::am_group_by_ids(self.ptr.as_ptr(), out)
        })
    }

    /// One grouped aggregate, one row per group in group order. A group with no value to answer
    /// with is null.
    ///
    /// `values` is required for every [`Agg`] except [`Agg::CountAll`], which ignores it.
    pub fn agg(&self, op: Agg, values: Option<&Array>) -> Result<Array> {
        if values.is_none() && op != Agg::CountAll {
            return Err(Error::new(format!("{op:?}: needs a values column")));
        }
        self.agg_raw(op.code(), values, 0.0)
    }

    /// `sum(values)` per group. The common case.
    pub fn sum(&self, values: &Array) -> Result<Array> {
        self.agg(Agg::Sum, Some(values))
    }

    /// One grouped aggregate by its raw op number from the header's `am_group_agg_ex` table, for the
    /// aggregates [`Agg`] does not name. `p1` is the quantile for ops 22 and 25 and ignored
    /// elsewhere.
    ///
    /// The op numbers are not checked here; an unknown one comes back as an error from the ABI.
    pub fn agg_raw(&self, op: c_int, values: Option<&Array>, p1: f64) -> Result<Array> {
        let values = values.map_or(std::ptr::null_mut(), |v| v.as_ptr());
        Array::produce("am_group_agg_ex", |out| unsafe {
            ffi::am_group_agg_ex(self.ptr.as_ptr(), values, op, p1, out)
        })
    }
}

// =================================================================================================
// The JSON plan runner
// =================================================================================================

/// One table a plan can `scan`, registered by name.
///
/// The ABI retains the column handles, so the [`Array`]s may be dropped while the source lives; this
/// type keeps them anyway so a reader does not have to know that.
pub struct Source {
    ptr: NonNull<sys::am_plan_source>,
    _columns: Vec<Array>,
    _not_send: PhantomData<*const ()>,
}

impl Drop for Source {
    fn drop(&mut self) {
        unsafe { ffi::am_plan_source_release(self.ptr.as_ptr()) }
    }
}

impl fmt::Debug for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("arrowmetal::Source").field("columns", &self._columns.len()).finish()
    }
}

impl Source {
    /// Registers `columns` as a table named `name`. Each column is a `(name, array)` pair; all of
    /// them must be the same length.
    pub fn new(name: &str, columns: Vec<(String, Array)>) -> Result<Self> {
        if columns.is_empty() {
            return Err(Error::new("Source::new: needs at least one column"));
        }
        let c_name = cstring(name, "source name")?;
        let c_col_names: Vec<CString> = columns
            .iter()
            .map(|(n, _)| cstring(n, "column name"))
            .collect::<Result<_>>()?;
        let mut name_ptrs: Vec<*const c_char> = c_col_names.iter().map(|n| n.as_ptr()).collect();
        let mut col_ptrs: Vec<*mut sys::am_array> = columns.iter().map(|(_, a)| a.as_ptr()).collect();

        let mut out: *mut sys::am_plan_source = std::ptr::null_mut();
        check(
            unsafe {
                ffi::am_plan_source_create(
                    c_name.as_ptr(),
                    col_ptrs.as_mut_ptr(),
                    name_ptrs.as_mut_ptr(),
                    col_ptrs.len() as i64,
                    &mut out,
                )
            },
            "am_plan_source_create",
        )?;
        match NonNull::new(out) {
            Some(ptr) => Ok(Source {
                ptr,
                _columns: columns.into_iter().map(|(_, a)| a).collect(),
                _not_send: PhantomData,
            }),
            None => Err(Error::new("am_plan_source_create: returned a null handle")),
        }
    }
}

/// The columns one [`run_plan`] produced.
pub struct PlanResult {
    ptr: NonNull<sys::am_plan_result>,
    _not_send: PhantomData<*const ()>,
}

impl Drop for PlanResult {
    fn drop(&mut self) {
        unsafe { ffi::am_plan_result_release(self.ptr.as_ptr()) }
    }
}

impl fmt::Debug for PlanResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("arrowmetal::PlanResult")
            .field("columns", &self.column_count())
            .field("rows", &self.row_count())
            .finish()
    }
}

impl PlanResult {
    /// How many columns the plan produced.
    pub fn column_count(&self) -> usize {
        let n = unsafe { ffi::am_plan_column_count(self.ptr.as_ptr()) };
        if n < 0 {
            0
        } else {
            n as usize
        }
    }

    /// How many rows the plan produced.
    pub fn row_count(&self) -> usize {
        let n = unsafe { ffi::am_plan_row_count(self.ptr.as_ptr()) };
        if n < 0 {
            0
        } else {
            n as usize
        }
    }

    /// The name of the `i`-th output column.
    pub fn column_name(&self, i: usize) -> Result<String> {
        if i >= self.column_count() {
            return Err(Error::new(format!(
                "column_name({i}): the result has {} columns",
                self.column_count()
            )));
        }
        let raw = unsafe { ffi::am_plan_column_name(self.ptr.as_ptr(), i as i64) };
        if raw.is_null() {
            Err(Error::new(format!("am_plan_column_name({i}) returned null")))
        } else {
            Ok(unsafe { CStr::from_ptr(raw) }.to_string_lossy().into_owned())
        }
    }

    /// The `i`-th output column, as a new handle.
    pub fn column(&self, i: usize) -> Result<Array> {
        Array::produce("am_plan_column", |out| unsafe {
            ffi::am_plan_column(self.ptr.as_ptr(), i as i64, out)
        })
    }
}

/// Runs a whole plan, given as JSON, over the registered `sources`.
///
/// The grammar is in `include/arrowmetal.h` and `docs/ENGINE.md`. `optimize` turns on predicate
/// pushdown, projection pruning, filter fusion, constant folding, expression CSE and join
/// reordering; pass `false` to run the plan as written.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use arrow::array::Int64Array;
/// # let amount = arrowmetal::Array::from_arrow(&Int64Array::from(vec![50i64, 150, 250]))?;
/// let src = arrowmetal::Source::new("sales", vec![("amount".into(), amount)])?;
/// let plan = r#"{"op":"filter","predicate":"(gt (col \"amount\") (int 100))",
///                "input":{"op":"scan","source":"sales"}}"#;
/// let out = arrowmetal::run_plan(plan, &[&src], true)?;
/// assert_eq!(out.row_count(), 2);
/// # Ok::<(), arrowmetal::Error>(())
/// ```
pub fn run_plan(plan_json: &str, sources: &[&Source], optimize: bool) -> Result<PlanResult> {
    let plan = cstring(plan_json, "plan JSON")?;
    let mut raw: Vec<*mut sys::am_plan_source> = sources.iter().map(|s| s.ptr.as_ptr()).collect();
    let mut out: *mut sys::am_plan_result = std::ptr::null_mut();
    check(
        unsafe {
            ffi::am_plan_run(
                plan.as_ptr(),
                raw.as_mut_ptr(),
                raw.len() as i64,
                optimize as c_int,
                &mut out,
            )
        },
        "am_plan_run",
    )?;
    match NonNull::new(out) {
        Some(ptr) => Ok(PlanResult { ptr, _not_send: PhantomData }),
        None => Err(Error::new("am_plan_run: succeeded but returned a null handle")),
    }
}

/// The optimized logical plan and the physical plan it lowers to, as text. Errors when the plan does
/// not type-check.
pub fn explain_plan(plan_json: &str, sources: &[&Source], optimize: bool) -> Result<String> {
    let plan = cstring(plan_json, "plan JSON")?;
    let mut raw: Vec<*mut sys::am_plan_source> = sources.iter().map(|s| s.ptr.as_ptr()).collect();
    let text = unsafe {
        ffi::am_plan_explain(
            plan.as_ptr(),
            raw.as_mut_ptr(),
            raw.len() as i64,
            optimize as c_int,
        )
    };
    if text.is_null() {
        Err(Error::last("am_plan_explain"))
    } else {
        // Valid only until the next call on this thread, so copy it out now.
        Ok(unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned())
    }
}

// =================================================================================================
// Batching
// =================================================================================================

thread_local! {
    /// How many [`batch`] scopes are open on this thread.
    ///
    /// Thread-local because the ABI's batching is: `am_batch_begin` opens a command buffer for the
    /// calling thread only. Counting is what makes nested `batch` calls safe — see [`batch`].
    static BATCH_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Runs `body` with every ArrowMetal call on this thread appended to one GPU command buffer.
///
/// The GPU runs once at the end of the scope, or earlier at the first call that must read a result
/// back (a reduction, an export). Chaining several kernels inside one batch saves one command-buffer
/// round trip per kernel.
///
/// The batch is closed even if `body` panics.
///
/// # Nesting
///
/// `batch` calls nest safely: only the outermost one opens and commits a batch, and only it can
/// report an end-of-batch error. An inner call runs its body inside the batch that is already open
/// and always returns `Ok`.
///
/// That has to be counted here rather than left to the ABI. `am_batch_begin` is a no-op when a batch
/// is already open, but `am_batch_end` closes **whatever is open**, without regard to nesting — so
/// an inner scope's guard would commit the outer scope's batch and every call in the rest of the
/// outer body would run unbatched, silently and with no error anywhere. A thread-local depth counter
/// is what prevents that.
///
/// # Errors
///
/// `am_batch_end` commits the command buffer and reports a non-zero code when the GPU work failed.
/// A batch defers most validation, so this is where a failure normally surfaces: `take` with an
/// out-of-range index, for instance, returns `Err` immediately when unbatched but returns `Ok` inside
/// a batch and fails the whole batch at the end.
///
/// That failure is returned as an `Err` **even though `body` already produced a value**, so a batch
/// that comes back `Ok` is one whose GPU work committed successfully. The value is discarded — not
/// because nothing ran, but because there is no way to know **how much** ran: a call that must read
/// a result back (a reduction, an export) forces a mid-batch flush, so some kernels in a failed batch
/// have executed and some have not, and the ABI does not say where the line fell.
///
/// The close itself happens in a `Drop`, which cannot fail, so the guard records the code and this
/// function reads it once the closure has returned and the guard has run.
pub fn batch<T>(body: impl FnOnce() -> T) -> Result<T> {
    // Only the outermost scope talks to the ABI; see the "Nesting" section above.
    let outermost = BATCH_DEPTH.with(|d| d.get() == 0);
    if outermost {
        check(unsafe { ffi::am_batch_begin() }, "am_batch_begin")?;
    }
    BATCH_DEPTH.with(|d| d.set(d.get() + 1));

    /// Leaves the batch on the way out, panic or not, and — if this was the outermost scope —
    /// commits it and records what `am_batch_end` said.
    struct Guard<'a> {
        failure: &'a RefCell<Option<Error>>,
        outermost: bool,
    }
    impl Drop for Guard<'_> {
        fn drop(&mut self) {
            BATCH_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
            if !self.outermost {
                return;
            }
            let rc = unsafe { ffi::am_batch_end() };
            if rc != 0 {
                // `am_last_error` is thread-local and the next call on this thread overwrites it,
                // so it is read here rather than after the guard has gone out of scope.
                *self.failure.borrow_mut() = Some(Error::last("am_batch_end"));
            }
        }
    }

    let failure: RefCell<Option<Error>> = RefCell::new(None);
    // The guard must be dropped -- and so must have run `am_batch_end` and recorded its code --
    // before `failure` is read, which is what this inner scope is for.
    let value = {
        let _guard = Guard { failure: &failure, outermost };
        body()
    };
    match failure.into_inner() {
        Some(e) => Err(e),
        None => Ok(value),
    }
}
