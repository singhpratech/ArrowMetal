#' Register a table for the plan runner
#'
#' A plan source names a set of columns the JSON plan can `scan`. It retains the columns, so the
#' `am_array` handles may go out of scope afterwards.
#'
#' @param name The table name used in `{"op":"scan","source":...}`.
#' @param columns A named list (or data frame) of columns: `am_array`s, `arrow` Arrays or R vectors.
#' @return An `am_plan_source` handle.
#' @export
am_plan_source <- function(name, columns) {
  am_require()
  columns <- as.list(columns)
  if (!length(columns)) stop("a plan source needs at least one column", call. = FALSE)
  nms <- names(columns)
  if (is.null(nms) || any(!nzchar(nms))) stop("every column needs a name", call. = FALSE)
  .Call(C_am_plan_source_create, as.character(name), lapply(columns, am_array), as.character(nms))
}

as_source_list <- function(sources) {
  if (inherits(sources, "am_plan_source")) sources <- list(sources)
  sources <- as.list(sources)
  for (s in sources)
    if (!inherits(s, "am_plan_source")) stop("`sources` must be am_plan_source handles", call. = FALSE)
  sources
}

#' Run a whole query plan in one call
#'
#' Sends a JSON logical plan to the ArrowMetal engine, which type-checks it, optimizes it and runs
#' it as fused Metal kernels inside one command buffer. The plan grammar is documented in the C
#' header (`include/arrowmetal.h`) and in `docs/ENGINE.md`.
#'
#' @param plan A plan: a JSON string, or a nested R `list` which is converted with
#'   `jsonlite::toJSON(auto_unbox = TRUE)` when jsonlite is installed.
#' @param sources One `am_plan_source`, or a list of them.
#' @param optimize Run the optimizer (default `TRUE`).
#' @return An `am_plan_result`: a list of `am_array` columns with `names()` and `nrow()`.
#' @examples
#' if (am_available()) {
#'   src <- am_plan_source("t", list(region = c("a", "b", "a"), amount = c(1, 2, 3)))
#'   r <- am_plan_run('{"op":"scan","source":"t"}', src)
#'   names(r)
#' }
#' @export
am_plan_run <- function(plan, sources, optimize = TRUE) {
  am_require()
  res <- .Call(C_am_plan_run, plan_json(plan), as_source_list(sources), isTRUE(optimize))
  dim <- .Call(C_am_plan_result_dim, res)
  nms <- .Call(C_am_plan_column_names, res)
  cols <- lapply(seq_along(nms), function(i) .Call(C_am_plan_column, res, as.double(i - 1)))
  names(cols) <- nms
  structure(cols, class = "am_plan_result", nrow = dim[1], result = res)
}

#' The optimized and physical plans, as text
#' @inheritParams am_plan_run
#' @return A single string.
#' @export
am_plan_explain <- function(plan, sources, optimize = TRUE) {
  am_require()
  .Call(C_am_plan_explain, plan_json(plan), as_source_list(sources), isTRUE(optimize))
}

plan_json <- function(plan) {
  if (is.character(plan) && length(plan) == 1L) return(plan)
  if (!requireNamespace("jsonlite", quietly = TRUE))
    stop("pass the plan as a JSON string, or install jsonlite to pass an R list", call. = FALSE)
  as.character(jsonlite::toJSON(plan, auto_unbox = TRUE))
}

#' @export
print.am_plan_result <- function(x, ...) {
  cat(sprintf("<am_plan_result %s row(s), %d column(s): %s>\n",
              format(attr(x, "nrow")), length(x), paste(names(x), collapse = ", ")))
  invisible(x)
}
