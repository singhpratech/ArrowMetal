# Index arrays are uint32 ("I"). R has no unsigned 32-bit type, so they read back the way the arrow
# package reads uint32: an integer vector while every value fits, a double vector past 2^31 - 1.

test_that("argsort, top_k and lexsort return uint32 indices that read back as integers", {
  skip_without_gpu()
  x <- c(30, 10, 20, 10, NA, 40)
  checks <- list(
    list(am_argsort(x), c(1L, 3L, 2L, 0L, 5L, 4L)),
    list(am_argsort(x, descending = TRUE, null_placement = "at_start"), c(4L, 5L, 0L, 2L, 1L, 3L)),
    list(am_top_k(x, 2), c(5L, 0L)),
    list(am_top_k(x, 2, largest = FALSE, float_order = "total"), c(1L, 3L)),
    list(am_lexsort(list(x), descending = TRUE), c(5L, 0L, 2L, 1L, 3L, 4L)),
    list(am_lexsort(list(x), null_placement = "at_start"), c(4L, 1L, 3L, 2L, 0L, 5L))
  )
  for (ck in checks) {
    expect_identical(am_format(ck[[1]]), "I")
    expect_identical(as_arrow_array(ck[[1]])$type$ToString(), "uint32")
    expect_identical(as.vector(ck[[1]]), ck[[2]])
  }
  # Empty input: still uint32, still an integer vector.
  e <- am_argsort(numeric(0))
  expect_identical(am_format(e), "I")
  expect_identical(as.vector(e), integer(0))
})

test_that("a uint32 index past 2^31 - 1 reads back as a double, as arrow reads uint32", {
  skip_without_gpu()
  big <- arrow::Array$create(c(0, 2147483647, 2147483648, 4294967295), type = arrow::uint32())
  a <- am_array(big)
  expect_identical(am_format(a), "I")
  v <- as.vector(a)
  expect_type(v, "double")
  expect_identical(v, c(0, 2147483647, 2147483648, 4294967295))
  small <- am_array(arrow::Array$create(c(0, 2147483647), type = arrow::uint32()))
  expect_type(as.vector(small), "integer")
})

test_that("plan window row_number, rank and dense_rank are uint32", {
  skip_without_gpu()
  src <- am_plan_source("t", list(g = c(1L, 1L, 1L, 2L, 2L), v = c(5L, 5L, 9L, 7L, 1L)))
  spec <- function(name, fn) {
    sprintf('{"name":"%s","fn":"%s","partition_by":["g"],"order_by":[["v",false]]}', name, fn)
  }
  plan <- paste0('{"op":"window","input":{"op":"scan","source":"t"},"specs":[',
                 spec("rn", "row_number"), ",", spec("rk", "rank"), ",", spec("dr", "dense_rank"), "]}")
  r <- am_plan_run(plan, src)
  want <- list(rn = c(1L, 2L, 3L, 2L, 1L), rk = c(1L, 1L, 3L, 2L, 1L), dr = c(1L, 1L, 2L, 2L, 1L))
  for (nm in names(want)) {
    expect_identical(am_format(r[[nm]]), "I")
    expect_identical(as.vector(r[[nm]]), want[[nm]])
  }
})

test_that("am_take takes int32, int64 and uint32 indices and R integer or double vectors", {
  skip_without_gpu()
  x <- c(10, 20, NA, 40)
  want <- c(40, 10, NA, NA, 20)
  idx <- c(3, 0, 2, NA, 1)
  expect_equal(rvec(am_take(x, as.integer(idx))), want)
  expect_equal(rvec(am_take(x, idx)), want)
  expect_equal(rvec(am_take(x, arrow::Array$create(idx, type = arrow::int32()))), want)
  expect_equal(rvec(am_take(x, arrow::Array$create(idx, type = arrow::int64()))), want)
  expect_equal(rvec(am_take(x, arrow::Array$create(idx, type = arrow::uint32()))), want)
  expect_equal(rvec(am_take(x, am_array(arrow::Array$create(idx, type = arrow::uint32())))), want)
  # The uint32 indices am_argsort returns feed am_take directly.
  expect_equal(rvec(am_take(x, am_argsort(x))), c(10, 20, 40, NA))
  # An index past 2^31 - 1 is accepted as a uint32 row number, so it is out of range here on the GPU
  # rather than refused in R; past 2^32 - 1 it is refused in R.
  expect_error(am_take(x, 2147483648), "out of range")
  expect_error(am_take(x, 4294967295), "out of range")
  expect_error(am_take(x, 4294967296), "whole numbers")
})
