//! Polars `Series` <-> ArrowMetal array, through the Arrow C Data Interface.
//!
//! Both sides already speak that interface, so the bridge is a pointer hand-off, not a conversion:
//!
//!   Series --rechunk--> polars_arrow::ffi::ArrowArray --am_import--> am_array (Metal-resident)
//!   am_array --am_export--> ArrowArray --import_array_from_c--> Series
//!
//! `polars_arrow::ffi::ArrowArray` and `arrowmetal_sys::ArrowArray` are both `#[repr(C)]`
//! transcriptions of the same C struct, so the only thing the hand-off needs is a pointer cast --
//! polars' copy keeps its fields private, which is why the cast is spelled out rather than a
//! field-by-field copy.
//!
//! Ownership follows the C Data Interface's rule that the consumer releases:
//!   * `am_import` takes over the exported `ArrowArray`, so we `mem::forget` our copy (the same
//!     thing `python/arrowmetal/__init__.py` does after `_export_to_c`). The `ArrowSchema` is
//!     ours to drop, and its `Drop` calls the release callback.
//!   * `am_export` fills a fresh pair we then hand to `import_array_from_c`, which takes the
//!     release callback over; the array's Metal buffers stay alive until Polars drops the Series.
//!
//! Strings go through `CompatLevel::oldest()`, i.e. Arrow `LargeUtf8`/`Utf8` rather than the
//! `Utf8View` layout Polars uses natively: ArrowMetal's kernels read offsets + bytes. That one
//! conversion is the plugin's only copy, and it is why the string expressions are slower relative
//! to native Polars than the numeric ones.

use polars::prelude::*;
use pyo3_polars::export::polars_arrow::ffi;

use arrowmetal_sys as am;

/// Turns an ArrowMetal error into a Polars one, keeping `am_last_error()`'s wording.
pub fn amerr(e: am::Error) -> PolarsError {
    polars_err!(ComputeError: "arrowmetal: {}", e.0)
}

/// Imports a Series into Metal memory.
///
/// The Series is rechunked first: ArrowMetal takes one Arrow array, and a Polars column read from
/// several files or built by `concat` carries several chunks. Rechunking a single-chunk Series is
/// free (it clones the `Arc`), so this only costs on genuinely chunked input.
pub fn to_metal(s: &Series) -> PolarsResult<am::Array> {
    let s = s.rechunk();
    let compat = CompatLevel::oldest();
    let field = s.dtype().to_arrow_field(s.name().clone(), compat);
    let array = s.to_arrow(0, compat);

    let mut c_array = ffi::export_array_to_c(array);
    let c_schema = ffi::export_field_to_c(&field);

    let out = unsafe {
        am::Array::import(
            &c_schema as *const ffi::ArrowSchema as *const am::ArrowSchema,
            &mut c_array as *mut ffi::ArrowArray as *mut am::ArrowArray,
        )
    };
    // `am_import` is the consumer: it owns the release callback now, whether it kept the buffers
    // zero-copy or copied them. Dropping our copy would release a second time.
    std::mem::forget(c_array);
    // `c_schema` is ours; its Drop calls the release callback, which is what we want.
    out.map_err(amerr)
}

/// Exports a Metal-resident array back into a Polars Series under `name`.
pub fn from_metal(a: &am::Array, name: &str) -> PolarsResult<Series> {
    let mut c_schema = ffi::ArrowSchema::empty();
    let mut c_array = ffi::ArrowArray::empty();
    unsafe {
        a.export_raw(
            &mut c_schema as *mut ffi::ArrowSchema as *mut am::ArrowSchema,
            &mut c_array as *mut ffi::ArrowArray as *mut am::ArrowArray,
        )
    }
    .map_err(amerr)?;

    let field = unsafe { ffi::import_field_from_c(&c_schema) }?;
    let array = unsafe { ffi::import_array_from_c(c_array, field.dtype) }?;
    Series::from_arrow(PlSmallStr::from_str(name), array)
}

/// A scalar operand as it arrived from Python, before it is narrowed to a column's element type.
///
/// Keeping the integer/float distinction (and the exact integer digits) is the whole point. The
/// tier-1 bridge builds the same 8 bytes with `struct.pack(code, v)`
/// (`python/arrowmetal/__init__.py`), and `struct.pack` **refuses** a scalar the column's type
/// cannot hold exactly: `struct.pack("b", 1000)` and `struct.pack("B", -1)` raise, and so does a
/// float against any of the integer codes. Rust's `as` does the opposite -- it saturates
/// out-of-range floats and truncates integers bit-wise -- so an `f64`-only operand would silently
/// turn `add(1000)` on an `Int8` column into `add(127)` and `add(-1)` on a `UInt8` column into a
/// no-op. It would also drop the low bits of any integer past 2^53. Hence this enum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScalarArg {
    /// An exact integer. `i128` covers the whole of `i64` and `u64`.
    Int(i128),
    /// An integer too large even for `i128`, carrying the `f64` it rounds to. Only a float column
    /// can take one -- `struct.pack("q", 2**200)` raises, `struct.pack("d", 2**200)` does not.
    WideInt(f64),
    /// A float.
    Float(f64),
}

impl ScalarArg {
    fn as_f64(self) -> f64 {
        match self {
            ScalarArg::Int(i) => i as f64,
            ScalarArg::WideInt(f) | ScalarArg::Float(f) => f,
        }
    }

    /// The exact integer, or the error `struct.pack` would have raised in tier 1.
    fn integer(self, dtype: &DataType) -> PolarsResult<i128> {
        match self {
            ScalarArg::Int(i) => Ok(i),
            ScalarArg::WideInt(f) => polars_bail!(InvalidOperation:
                "arrowmetal: scalar {f:e} is out of range for a {dtype:?} column"),
            ScalarArg::Float(f) => polars_bail!(InvalidOperation:
                "arrowmetal: a {dtype:?} column takes an integer scalar, got {f}"),
        }
    }
}

/// The inclusive value range and byte width of an integer dtype: exactly what `struct.pack`'s
/// `"b"`/`"B"`/`"h"`/`"H"`/`"i"`/`"I"`/`"q"`/`"Q"` codes enforce for the tier-1 bridge.
fn int_range(dtype: &DataType) -> Option<(usize, i128, i128)> {
    Some(match dtype {
        DataType::Int8 => (1, i8::MIN as i128, i8::MAX as i128),
        DataType::Int16 => (2, i16::MIN as i128, i16::MAX as i128),
        DataType::Int32 => (4, i32::MIN as i128, i32::MAX as i128),
        DataType::Int64 => (8, i64::MIN as i128, i64::MAX as i128),
        DataType::UInt8 => (1, 0, u8::MAX as i128),
        DataType::UInt16 => (2, 0, u16::MAX as i128),
        DataType::UInt32 => (4, 0, u32::MAX as i128),
        DataType::UInt64 => (8, 0, u64::MAX as i128),
        _ => return None,
    })
}

/// The little-endian bytes of `v` at the width of `dtype`, ready to hand to `am_compare_scalar` /
/// `am_arith_scalar`, which read a pointer to one value of the array's element type.
///
/// Only 8 bytes are ever needed: the C ABI's scalar forms cover the primitive types, and the
/// decimal path (16 bytes) is not exposed by this plugin.
///
/// A scalar the column's type cannot hold is an error, not a silently narrowed value -- see
/// `ScalarArg`. "Integers wrap" in the docs is about the *arithmetic*: `127i8 + 1 == -128` is the
/// kernel's answer and stays so. It never meant that the operand itself may be out of range.
pub fn scalar_bytes(dtype: &DataType, v: ScalarArg) -> PolarsResult<[u8; 8]> {
    let mut b = [0u8; 8];
    if let Some((width, lo, hi)) = int_range(dtype) {
        let i = v.integer(dtype)?;
        polars_ensure!(i >= lo && i <= hi, InvalidOperation:
            "arrowmetal: scalar {i} is out of range for a {dtype:?} column ({lo} to {hi})");
        // Two's complement: the low `width` bytes of the i128 are the value at that width.
        b[..width].copy_from_slice(&i.to_le_bytes()[..width]);
        return Ok(b);
    }
    match dtype {
        // f64 -> f32 rounds, and overflows to an infinity, which is what `struct.pack("f", 1e300)`
        // gives the tier-1 bridge.
        DataType::Float32 => b[..4].copy_from_slice(&(v.as_f64() as f32).to_le_bytes()),
        DataType::Float64 => b[..8].copy_from_slice(&v.as_f64().to_le_bytes()),
        other => {
            polars_bail!(InvalidOperation: "arrowmetal: no scalar form for dtype {other:?}")
        },
    }
    Ok(b)
}

/// One `am_reduce` result as a length-1 Series, cast to `want` when a dtype is asked for.
///
/// `am_reduce` answers in one of three widths (int64, uint64, float64) whatever the column's own
/// type is, so `min`/`max` narrow back to the input dtype here and `sum`/`mean` keep the wide one.
pub fn scalar_series(name: &str, v: am::Scalar, want: Option<&DataType>) -> PolarsResult<Series> {
    let name = PlSmallStr::from_str(name);
    let s = match v {
        am::Scalar::Null => match want {
            Some(dt) => Series::full_null(name, 1, dt),
            None => Series::full_null(name, 1, &DataType::Float64),
        },
        am::Scalar::I64(i) => Series::new(name, [i]),
        am::Scalar::U64(u) => Series::new(name, [u]),
        am::Scalar::F64(f) => Series::new(name, [f]),
    };
    match want {
        Some(dt) if s.dtype() != dt => s.cast(dt),
        _ => Ok(s),
    }
}
