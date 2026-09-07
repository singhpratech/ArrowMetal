reduce_op <- c(sum = 0L, min = 1L, max = 2L, mean = 3L)

# out_kind: 0 int64, 1 uint64, 2 float64.
reduce_result <- function(r, integer64) {
  kind <- r[[1]]
  is_null <- r[[2]]
  if (kind == 2L || !integer64) {
    v <- if (is_null) NA_real_ else r[[4]]
    if (!is_null && kind != 2L && abs(v) > 2^53) {
      warning("ArrowMetal: the exact 64-bit integer result is not representable as an R double; ",
              "call with integer64 = TRUE for an exact bit64::integer64.", call. = FALSE)
    }
    return(v)
  }
  if (!requireNamespace("bit64", quietly = TRUE))
    stop("integer64 = TRUE needs the bit64 package", call. = FALSE)
  if (kind == 1L)
    warning("ArrowMetal: an unsigned 64-bit result is being returned as a signed bit64::integer64.",
            call. = FALSE)
  v <- if (is_null) NA_real_ else r[[3]]
  class(v) <- "integer64"
  v
}

am_reduce <- function(x, op, integer64 = FALSE) {
  am_require()
  reduce_result(.Call(C_am_reduce, check_am(x), reduce_op[[op]]), integer64)
}

#' Scalar reductions
#'
#' `am_sum()`, `am_min()`, `am_max()` and `am_mean()` reduce a whole column on the GPU. Nulls are
#' skipped, as in Arrow; a column with no valid value gives `NA`. `am_min()` and `am_max()` also skip
#' `NaN` and report `NA` when every valid value is `NaN`.
#'
#' R has no native 64-bit integer, so a sum over an int64 column comes back as a double and is exact
#' only to 2^53. Pass `integer64 = TRUE` for an exact `bit64::integer64`; a result outside 2^53
#' warns otherwise.
#'
#' @param x An `am_array`, or anything [am_array()] accepts.
#' @param integer64 Return `bit64::integer64` instead of a double for integer results.
#' @return A length-1 double, or a `bit64::integer64` when `integer64 = TRUE` and the result is
#'   an integer. `am_mean()` is always a double.
#' @examples
#' if (am_available()) am_sum(am_array(c(1, 2, NA, 4)))
#' @export
am_sum <- function(x, integer64 = FALSE) am_reduce(am_array(x), "sum", integer64)

#' @rdname am_sum
#' @export
am_min <- function(x, integer64 = FALSE) am_reduce(am_array(x), "min", integer64)

#' @rdname am_sum
#' @export
am_max <- function(x, integer64 = FALSE) am_reduce(am_array(x), "max", integer64)

#' @rdname am_sum
#' @export
am_mean <- function(x) am_reduce(am_array(x), "mean", FALSE)

compare_ops <- c("==" = 0L, "!=" = 1L, "<" = 2L, "<=" = 3L, ">" = 4L, ">=" = 5L,
                 eq = 0L, ne = 1L, lt = 2L, le = 3L, gt = 4L, ge = 5L)

#' Element-wise comparison
#'
#' Compares a column with a scalar or with another column of the same type and returns a boolean
#' `am_array`. Null in, null out.
#'
#' @param x An `am_array`, or anything [am_array()] accepts.
#' @param op One of `"=="`, `"!="`, `"<"`, `"<="`, `">"`, `">="`.
#' @param y A length-1 R value (compared as a scalar) or an `am_array` of the same length.
#' @return A boolean `am_array`.
#' @examples
#' if (am_available()) {
#'   a <- am_array(c(1, 5, 9))
#'   as.vector(am_filter(a, am_compare(a, ">", 3)))
#' }
#' @export
am_compare <- function(x, op, y) {
  am_require()
  x <- am_array(x)
  code <- if (is.character(op) && length(op) == 1L) compare_ops[op] else NA_integer_
  if (is.na(code)) {
    stop("unknown comparison operator: ", paste(format(op), collapse = " "),
         " (expected one of ==, !=, <, <=, >, >=)", call. = FALSE)
  }
  code <- unname(code)
  if (is_am_array(y) || inherits(y, "Array") || length(y) > 1L) {
    return(.Call(C_am_compare_array, x, code, am_array(y)))
  }
  .Call(C_am_compare_scalar, x, code, y, scalar_format(x))
}

# A dictionary-encoded column reports its index format but takes a scalar in the values' type.
scalar_format <- function(x) {
  fmt <- am_format(x)
  if (identical(fmt, "i") && .Call(C_am_child_count, x) == 1)
    fmt <- am_format(.Call(C_am_child, x, 0))
  fmt
}

#' Keep the rows a boolean mask selects
#'
#' @param x An `am_array`, or anything [am_array()] accepts.
#' @param mask A boolean `am_array` (or an R logical vector) of the same length. A null in the mask
#'   drops the row, as `arrow`'s default `null_selection_behavior = "drop"` does.
#' @return An `am_array` of the selected rows.
#' @export
am_filter <- function(x, mask) {
  am_require()
  .Call(C_am_filter, am_array(x), am_array(mask))
}

#' Gather rows by index
#'
#' @param x An `am_array`, or anything [am_array()] accepts.
#' @param indices Zero-based indices: an `am_array` of integers, or an R integer/numeric vector.
#'   Arrow indices are zero based, so `am_take(x, am_argsort(x))` is the sorted column.
#' @return An `am_array`.
#' @export
am_take <- function(x, indices) {
  am_require()
  if (!is_am_array(indices) && !inherits(indices, "Array"))
    indices <- arrow::Array$create(as.integer(indices))
  .Call(C_am_take, am_array(x), am_array(indices))
}

#' Sort indices
#'
#' A stable GPU radix sort. Nulls go last and `NaN` sorts after `+Inf`; neither is mirrored to the
#' front when `descending = TRUE`.
#'
#' @param x An `am_array`, or anything [am_array()] accepts.
#' @param descending Sort largest first.
#' @return An int32 `am_array` of zero-based indices.
#' @export
am_argsort <- function(x, descending = FALSE) {
  am_require()
  .Call(C_am_argsort, am_array(x), isTRUE(descending))
}

#' Sorted copy of a column
#' @inheritParams am_argsort
#' @return An `am_array` of the same type as `x`.
#' @export
am_sort <- function(x, descending = FALSE) {
  am_require()
  .Call(C_am_sort, am_array(x), isTRUE(descending))
}

#' A zero-copy slice of a column
#' @param x An `am_array`, or anything [am_array()] accepts.
#' @param offset Zero-based first row.
#' @param length Number of rows.
#' @return An `am_array`.
#' @export
am_slice <- function(x, offset, length) {
  am_require()
  .Call(C_am_slice, am_array(x), as.double(offset), as.double(length))
}
