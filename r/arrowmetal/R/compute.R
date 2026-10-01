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
  # NA_real_ reclassed as integer64 is that double's bit pattern read as an integer, i.e. garbage;
  # bit64 has its own NA whose bit pattern is the int64 NA sentinel.
  if (is_null) return(bit64::NA_integer64_)
  if (kind == 1L)
    warning("ArrowMetal: an unsigned 64-bit result is being returned as a signed bit64::integer64.",
            call. = FALSE)
  v <- r[[3]]
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
  if (is_am_array(y) || inherits(y, "Array") || inherits(y, "ChunkedArray") || length(y) > 1L) {
    return(.Call(C_am_compare_array, x, code, am_array(y)))
  }
  if (length(y) != 1L)
    stop("`y` must be a length-1 scalar or a column the same length as `x`", call. = FALSE)
  # The ABI scalar is a raw value with no validity flag, so an NA scalar cannot be handed to the
  # kernel: every comparison with a null is null, which is what base R and arrow both give.
  if (is.na(y) && !(is.double(y) && is.nan(y)))
    return(am_array(arrow::Array$create(rep(NA, length(x)), type = arrow::bool())))
  if (!is.numeric(y) && !is.logical(y) && !inherits(y, "integer64")) {
    stop("scalar operations need a numeric array and a numeric scalar; got a scalar of type '",
         typeof(y), "'", call. = FALSE)
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
#' @param indices Zero-based indices: an `am_array` of int32, int64 or uint32 integers (the uint32
#'   index arrays [am_argsort()] returns included), or an R integer/numeric vector of whole numbers in
#'   `[0, 2^32 - 1]`. Arrow indices are zero based, so `am_take(x, am_argsort(x))` is the sorted
#'   column.
#' @return An `am_array`.
#' @export
am_take <- function(x, indices) {
  am_require()
  if (!is_am_array(indices) && !inherits(indices, "Array") && !inherits(indices, "ChunkedArray")) {
    if (!is.numeric(indices))
      stop("`indices` must be numeric, an arrow Array or an am_array", call. = FALSE)
    # as.integer() silently turns anything outside int32 into NA with a warning; refuse instead.
    # Row numbers go up to 2^32 - 1, the uint32 index type; past int32 the indices go in as uint32.
    max_row <- 4294967295
    bad <- !is.na(indices) & (indices > max_row | indices < 0 | indices != trunc(indices))
    if (any(bad)) {
      stop("`indices` must be whole numbers in [0, ", format(max_row, scientific = FALSE), "]; ",
           "offending value: ", format(indices[which(bad)[1]], scientific = FALSE), call. = FALSE)
    }
    if (any(!is.na(indices) & indices > .Machine$integer.max)) {
      indices <- arrow::Array$create(as.double(indices), type = arrow::uint32())
    } else {
      indices <- arrow::Array$create(as.integer(indices))
    }
  }
  .Call(C_am_take, am_array(x), am_array(indices))
}

null_placements <- c(at_end = 0L, at_start = 1L)
float_orders <- c(ieee = 0L, total = 1L)

sort_code <- function(value, table, what) {
  if (!is.character(value) || length(value) != 1L || is.na(value) || !(value %in% names(table)))
    stop("`", what, "` must be one of ", paste0('"', names(table), '"', collapse = ", "),
         call. = FALSE)
  table[[value]]
}

# The plain calls keep their own entry points, so the defaults run exactly the code they always did.
sort_defaults <- function(null_placement, float_order)
  identical(null_placement, "at_end") && identical(float_order, "ieee")

#' Sort indices
#'
#' A stable GPU radix sort. By default nulls go last and `NaN` sorts after `+Inf`; neither is
#' mirrored to the front when `descending = TRUE`.
#'
#' `null_placement = "at_start"` puts the null rows first, in either direction. `float_order =
#' "total"` orders a float column by IEEE 754 totalOrder, as arrow-rs and Rust's `total_cmp` define
#' it: `-NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN`, and a descending sort is its exact mirror.
#' The default `"ieee"` is Arrow C++'s order: `-0` ties `+0`, every `NaN` is one value, and the `NaN`
#' rows sit next to the nulls in both directions. Integer, string and temporal columns ignore
#' `float_order`. Neither option adds a pass to the GPU sort.
#'
#' @param x An `am_array`, or anything [am_array()] accepts.
#' @param descending Sort largest first.
#' @param null_placement `"at_end"` (the default) or `"at_start"`, as in Arrow's sort options.
#' @param float_order `"ieee"` (the default) or `"total"`.
#' @return A uint32 `am_array` of zero-based indices. Row numbers go up to 2^32 - 1; `as.vector()`
#'   gives an integer vector when every index fits a 32-bit R integer and a double vector when one
#'   passes 2^31 - 1, which is how the `arrow` package reads uint32.
#' @examples
#' if (am_available()) {
#'   x <- c(2, NA, -0, NaN, 0, -Inf)
#'   as.vector(am_argsort(x, null_placement = "at_start", float_order = "total"))
#' }
#' @export
am_argsort <- function(x, descending = FALSE, null_placement = "at_end", float_order = "ieee") {
  am_require()
  if (sort_defaults(null_placement, float_order))
    return(.Call(C_am_argsort, am_array(x), isTRUE(descending)))
  .Call(C_am_argsort_ex, am_array(x), isTRUE(descending),
        sort_code(null_placement, null_placements, "null_placement"),
        sort_code(float_order, float_orders, "float_order"))
}

#' Sorted copy of a column
#' @inheritParams am_argsort
#' @return An `am_array` of the same type as `x`.
#' @export
am_sort <- function(x, descending = FALSE, null_placement = "at_end", float_order = "ieee") {
  am_require()
  if (sort_defaults(null_placement, float_order))
    return(.Call(C_am_sort, am_array(x), isTRUE(descending)))
  .Call(C_am_sort_ex, am_array(x), isTRUE(descending),
        sort_code(null_placement, null_placements, "null_placement"),
        sort_code(float_order, float_orders, "float_order"))
}

#' Indices of the k largest or smallest values
#'
#' The first `k` indices [am_argsort()] gives with `descending = largest` and the same options,
#' found by GPU selection rather than a whole sort.
#'
#' @inheritParams am_argsort
#' @param k Number of rows (a whole number, 0 or more; past the length gives every row).
#' @param largest The `k` largest (default) or, with `FALSE`, the `k` smallest.
#' @return A uint32 `am_array` of zero-based indices, in sorted order (read back as [am_argsort()]'s).
#' @examples
#' if (am_available()) as.vector(am_top_k(c(5, NA, 9, 1), 2))
#' @export
am_top_k <- function(x, k, largest = TRUE, null_placement = "at_end", float_order = "ieee") {
  am_require()
  if (!is.numeric(k) || length(k) != 1L || is.na(k) || k < 0 || k != trunc(k))
    stop("`k` must be a whole number, 0 or more", call. = FALSE)
  if (sort_defaults(null_placement, float_order))
    return(.Call(C_am_top_k, am_array(x), as.double(k), isTRUE(largest)))
  .Call(C_am_top_k_ex, am_array(x), as.double(k), isTRUE(largest),
        sort_code(null_placement, null_placements, "null_placement"),
        sort_code(float_order, float_orders, "float_order"))
}

#' Sort indices over several columns
#'
#' Orders the rows by each column in turn, the first column the most significant. Each of
#' `descending`, `null_placement` and `float_order` is one value for every column or a vector with
#' one per column, so each key can have its own direction, null placement and float order.
#'
#' @param columns A list of columns: `am_array`s or anything [am_array()] accepts, all one length.
#' @param descending Logical, one per column or one for all.
#' @param null_placement `"at_end"` or `"at_start"`, one per column or one for all.
#' @param float_order `"ieee"` or `"total"`, one per column or one for all.
#' @return A uint32 `am_array` of zero-based indices (read back as [am_argsort()]'s).
#' @examples
#' if (am_available()) {
#'   as.vector(am_lexsort(list(c(1, 1, 2), c(3, NA, 1)), descending = c(FALSE, TRUE),
#'                        null_placement = c("at_end", "at_start")))
#' }
#' @export
am_lexsort <- function(columns, descending = FALSE, null_placement = "at_end", float_order = "ieee") {
  am_require()
  columns <- as.list(columns)
  n <- length(columns)
  if (!n) stop("am_lexsort needs at least one column", call. = FALSE)
  per_key <- function(v, what) {
    if (length(v) == 1L) return(rep(v, n))
    if (length(v) != n)
      stop("`", what, "` has ", length(v), " entries for ", n, " columns", call. = FALSE)
    v
  }
  desc <- per_key(descending, "descending")
  if (!is.logical(desc) || anyNA(desc)) stop("`descending` must be TRUE or FALSE", call. = FALSE)
  handles <- lapply(columns, am_array)
  places <- per_key(null_placement, "null_placement")
  orders <- per_key(float_order, "float_order")
  if (all(places == "at_end") && all(orders == "ieee"))
    return(.Call(C_am_lexsort, handles, as.integer(desc), NULL, NULL))
  .Call(C_am_lexsort, handles, as.integer(desc),
        vapply(places, sort_code, integer(1), table = null_placements, what = "null_placement",
               USE.NAMES = FALSE),
        vapply(orders, sort_code, integer(1), table = float_orders, what = "float_order",
               USE.NAMES = FALSE))
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
