group_agg_ops <- c(sum = 0L, count_all = 1L, count = 2L, mean = 3L, min = 4L, max = 5L,
                   first = 7L, last = 8L, product = 16L,
                   var = 18L, sd = 20L, median = 21L, quantile = 22L)

#' Group a set of key columns
#'
#' Maps one or more key columns to dense group ids on the GPU and returns an object whose methods
#' run one grouped aggregate each. Every Arrow type ArrowMetal supports works as a key; a null key
#' is its own group, as Arrow's hash aggregation does.
#'
#' Group order is deterministic but is not Arrow's first-seen order: it is ascending by key for
#' numeric, boolean, temporal and decimal columns (nulls last), first-seen for string and binary
#' columns, and lexicographic in column order when there are several. Use `$keys()` to label the
#' rows rather than assuming an order.
#'
#' @param ... One or more key columns: `am_array`s, `arrow` Arrays or R vectors, all the same
#'   length. A single `list` or `data.frame` is also accepted.
#' @return An `am_groupby` object with these members:
#'   \describe{
#'     \item{`$n_groups`}{the number of groups}
#'     \item{`$keys(i = 1)`}{the `i`-th key column, one row per group, in group order}
#'     \item{`$ids()`}{the int32 group id of every input row}
#'     \item{`$sum(v)`, `$min(v)`, `$max(v)`, `$mean(v)`, `$count(v)`, `$first(v)`, `$last(v)`,
#'           `$product(v)`, `$var(v)`, `$sd(v)`, `$median(v)`}{one grouped aggregate}
#'     \item{`$count_all()`}{rows per group, nulls included}
#'     \item{`$quantile(v, q)`}{the `q` quantile per group}
#'     \item{`$agg(v, op, p1 = 0)`}{any op of `am_group_agg_ex` by its number}
#'   }
#' @examples
#' if (am_available()) {
#'   g <- am_group_by(c("a", "b", "a", "b"))
#'   as.vector(g$keys())
#'   as.vector(g$sum(c(1, 10, 2, 20)))
#' }
#' @export
am_group_by <- function(...) {
  am_require()
  cols <- list(...)
  if (length(cols) == 1L && (is.data.frame(cols[[1]]) ||
                             (is.list(cols[[1]]) && !inherits(cols[[1]], "Array")))) {
    cols <- as.list(cols[[1]])
  }
  if (!length(cols)) stop("am_group_by() needs at least one key column", call. = FALSE)
  handles <- lapply(cols, am_array)
  gb <- .Call(C_am_group_by_keys, handles)
  n_cols <- length(handles)

  agg <- function(values, op, p1 = 0) {
    code <- if (is.character(op)) unname(group_agg_ops[op]) else as.integer(op)
    if (length(code) != 1L || is.na(code))
      stop("unknown grouped aggregate: ", paste(format(op), collapse = " "), call. = FALSE)
    v <- if (is.null(values)) NULL else am_array(values)
    .Call(C_am_group_agg, gb, v, code, as.double(p1))
  }

  self <- list(
    n_groups = .Call(C_am_group_by_group_count, gb),
    n_keys = n_cols,
    keys = function(i = 1) {
      if (i < 1 || i > n_cols) stop("key column ", i, " does not exist", call. = FALSE)
      .Call(C_am_group_by_keys_result, gb, as.double(i - 1))
    },
    ids = function() .Call(C_am_group_by_ids, gb),
    agg = agg,
    sum = function(v) agg(v, "sum"),
    min = function(v) agg(v, "min"),
    max = function(v) agg(v, "max"),
    mean = function(v) agg(v, "mean"),
    count = function(v) agg(v, "count"),
    count_all = function() agg(NULL, "count_all"),
    first = function(v) agg(v, "first"),
    last = function(v) agg(v, "last"),
    product = function(v) agg(v, "product"),
    var = function(v) agg(v, "var"),
    sd = function(v) agg(v, "sd"),
    median = function(v) agg(v, "median"),
    quantile = function(v, q) agg(v, "quantile", q),
    handle = gb
  )
  class(self) <- "am_groupby"
  self
}

#' @export
print.am_groupby <- function(x, ...) {
  cat(sprintf("<am_groupby %s key column(s), %s group(s)>\n",
              format(x$n_keys), format(x$n_groups)))
  invisible(x)
}
