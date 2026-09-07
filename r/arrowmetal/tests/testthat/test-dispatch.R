# arrow exports an `as_arrow_array(x, ..., type = NULL)` generic with eight methods of its own.
# arrowmetal must add a method to THAT generic; a second generic of the same name would either mask
# arrow's (breaking arrow's methods) or be masked by it (never dispatching for am_array), depending
# on the order the packages are attached. These tests run the README example in both orders in a
# fresh R process, which is the only way to see the bug: the rest of the suite reaches arrow through
# `arrow::` and so never exercises dispatch from the search path.

run_in_fresh_r <- function(code) {
  script <- tempfile(fileext = ".R")
  on.exit(unlink(script), add = TRUE)
  writeLines(code, script)
  out <- suppressWarnings(system2(
    file.path(R.home("bin"), "Rscript"), c("--vanilla", shQuote(script)),
    stdout = TRUE, stderr = TRUE
  ))
  status <- attr(out, "status")
  list(status = if (is.null(status)) 0L else as.integer(status),
       output = paste(out, collapse = "\n"))
}

lib_preamble <- function(order) {
  c(sprintf('.libPaths(c(%s))', paste(sprintf('"%s"', .libPaths()), collapse = ", ")),
    sprintf('Sys.setenv(ARROWMETAL_LIB = "%s")', Sys.getenv("ARROWMETAL_LIB", "")),
    if (identical(order, "arrowmetal_first"))
      c('suppressMessages(library(arrowmetal))', 'suppressMessages(library(arrow))')
    else
      c('suppressMessages(library(arrow))', 'suppressMessages(library(arrowmetal))'))
}

for (order in c("arrowmetal_first", "arrow_first")) {
  local({
    ord <- order
    test_that(paste0("the README example works with ", ord), {
      skip_without_gpu()
      r <- run_in_fresh_r(c(
        lib_preamble(ord),
        'x <- Array$create(c(1, 5, 9, 2))',
        'stopifnot(am_sum(x) == 17)',
        # The README's own line: an unqualified as_arrow_array() on an am_array.
        'v <- as.vector(as_arrow_array(am_filter(x, am_compare(x, ">", 3))))',
        'stopifnot(identical(v, c(5, 9)))',
        # arrow\'s own methods must still dispatch from the search path.
        'stopifnot(inherits(as_arrow_array(c(1, 2, 3)), "Array"))',
        'stopifnot(inherits(as_arrow_array(ChunkedArray$create(c(1, 2), c(3, 4))), "Array"))',
        'stopifnot(inherits(as_arrow_array(Scalar$create(1)), "Array"))',
        'cat("OK\\n")'
      ))
      expect_equal(r$status, 0L, info = r$output)
      expect_match(r$output, "OK")
    })
  })
}

test_that("arrowmetal adds a method to arrow's generic instead of defining its own", {
  # The generic the package re-exports is the identical object arrow exports, so neither package
  # masks the other.
  expect_identical(arrowmetal::as_arrow_array, arrow::as_arrow_array)
  expect_true(any(grepl("am_array", as.character(suppressMessages(methods(arrow::as_arrow_array))))))
  # arrow's own eight methods are still registered alongside ours.
  expect_true(all(c("as_arrow_array.Array", "as_arrow_array.ChunkedArray",
                    "as_arrow_array.data.frame", "as_arrow_array.default") %in%
                    as.character(suppressMessages(methods(arrow::as_arrow_array)))))
})

test_that("a type argument on the method casts, matching the generic's contract", {
  skip_without_gpu()
  a <- am_array(c(1, 2, 3))
  expect_equal(as_arrow_array(a, type = arrow::int32())$type$ToString(), "int32")
  expect_equal(as.vector(as_arrow_array(a, type = arrow::int32())), 1:3)
})
