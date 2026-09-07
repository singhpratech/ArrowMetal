#' Move a column onto the GPU
#'
#' Imports an Arrow array through the Arrow C Data Interface and returns an ArrowMetal handle.
#' Anything the `arrow` package can turn into an [arrow::Array] is accepted; an
#' [arrow::ChunkedArray] has its chunks concatenated first, and an `am_array` is returned
#' unchanged.
#'
#' The transfer is copy-free when the producer's buffers are page aligned and one copy otherwise.
#' Use [am_buffer_alignment()] to see which case a given array falls into.
#'
#' @param x An `arrow` Array, ChunkedArray, an R vector, or an `am_array`.
#' @param type Optional [arrow::DataType] used when `x` has to be converted to an Arrow array
#'   first, e.g. `arrow::int64()`.
#' @return An `am_array` handle.
#' @examples
#' if (am_available()) {
#'   a <- am_array(c(1, 2, NA, 4))
#'   am_sum(a)
#' }
#' @export
am_array <- function(x, type = NULL) {
  am_require()
  if (is_am_array(x)) return(x)
  a <- as_input_array(x, type)
  array_ptr <- .Call(C_alloc_arrow_array)
  schema_ptr <- .Call(C_alloc_arrow_schema)
  a$export_to_c(array_ptr, schema_ptr)
  .Call(C_am_import, schema_ptr, array_ptr)
}

# arrow's own generic already turns an Array, ChunkedArray (concatenating the chunks), Scalar,
# data frame or plain R vector into one Array, so let it do the work rather than reimplementing it.
# ChunkedArray has no $combine_chunks() method in arrow 25.0.0.
as_input_array <- function(x, type = NULL) {
  if (is_am_array(x)) return(as_arrow_array(x, type = type))
  arrow::as_arrow_array(x, type = type)
}

#' Is this an ArrowMetal handle?
#' @param x Any object.
#' @return `TRUE` or `FALSE`.
#' @export
is_am_array <- function(x) inherits(x, "am_array")

check_am <- function(x, what = "x") {
  if (!is_am_array(x)) stop("`", what, "` must be an am_array (see am_array())", call. = FALSE)
  x
}

# Move a column back to the arrow package.
#
# This is a METHOD on arrow's own exported `as_arrow_array` generic (formals x, ..., type = NULL),
# not a new generic: defining a second generic of that name would mask arrow's and break its eight
# methods, or be masked by it and never dispatch, depending on library() order.
#
# Exporting an ArrowMetal handle is always copy-free: the resulting arrow Array points at the Metal
# buffers and keeps them alive through its release callback.
as_arrow_array.am_array <- function(x, ..., type = NULL) {
  am_require()
  array_ptr <- .Call(C_alloc_arrow_array)
  schema_ptr <- .Call(C_alloc_arrow_schema)
  .Call(C_am_export, x, schema_ptr, array_ptr)
  out <- arrow::Array$import_from_c(array_ptr, schema_ptr)
  if (!is.null(type)) out <- out$cast(type)
  out
}

#' Number of elements
#' @param x An `am_array`.
#' @return An integer-valued numeric.
#' @export
length.am_array <- function(x) {
  am_require()
  n <- .Call(C_am_length, x)
  if (n <= .Machine$integer.max) as.integer(n) else n
}

#' Number of null elements
#' @param x An `am_array`.
#' @return An integer-valued numeric.
#' @export
am_null_count <- function(x) {
  am_require()
  .Call(C_am_null_count, check_am(x))
}

#' Arrow format string of the element type
#'
#' The one-or-more character Arrow C Data Interface format code, e.g. `"g"` for float64,
#' `"l"` for int64, `"i"` for int32 and `"b"` for boolean.
#'
#' @param x An `am_array`.
#' @return A single string.
#' @export
am_format <- function(x) {
  am_require()
  .Call(C_am_format, check_am(x))
}

#' @export
as.vector.am_array <- function(x, mode = "any") as.vector(as_arrow_array(x), mode)

#' @export
print.am_array <- function(x, ...) {
  am_require()
  cat(sprintf("<am_array %s len=%s nulls=%s device=%s>\n",
              am_format(x), format(length(x)), format(am_null_count(x)), am_device_name()))
  invisible(x)
}

#' Buffer alignment of an Arrow array
#'
#' Exports `x` through the C Data Interface and reports where each of its buffers starts. ArrowMetal
#' imports a buffer without a copy when it starts on a page boundary and copies it otherwise, so this
#' is the measurement that decides which of the two import paths a producer gets.
#'
#' @param x An `arrow` Array or anything [arrow::Array$create()] accepts.
#' @return A data frame with one row per buffer: the buffer index, its address, its offset within a
#'   page, and whether it is page aligned. A null (absent) buffer has address 0 and is reported as
#'   `NA`.
#' @export
am_buffer_alignment <- function(x) {
  a <- as_input_array(x, NULL)
  array_ptr <- .Call(C_alloc_arrow_array)
  schema_ptr <- .Call(C_alloc_arrow_schema)
  a$export_to_c(array_ptr, schema_ptr)
  addr <- .Call(C_arrow_array_buffers, array_ptr)
  page <- .Call(C_page_size)
  offset <- ifelse(addr == 0, NA_real_, addr %% page)
  data.frame(
    buffer = seq_along(addr) - 1L,
    address = addr,
    page_size = page,
    offset_in_page = offset,
    page_aligned = ifelse(is.na(offset), NA, offset == 0)
  )
}
