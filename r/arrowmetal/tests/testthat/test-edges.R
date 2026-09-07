test_that("a ChunkedArray is concatenated on the way in", {
  # arrow 25.0.0's ChunkedArray has no $combine_chunks(); the conversion goes through arrow's own
  # as_arrow_array generic, which concatenates the chunks.
  skip_without_gpu()
  ca <- arrow::ChunkedArray$create(c(1, 2, NA), c(4, 5))
  expect_false("combine_chunks" %in% names(ca))
  a <- am_array(ca)
  expect_equal(length(a), 5L)
  expect_equal(rvec(a), c(1, 2, NA, 4, 5))
  expect_equal(am_sum(ca), 12)
  expect_equal(am_null_count(a), 1)
  # A single-chunk and a zero-chunk ChunkedArray too.
  expect_equal(rvec(am_array(arrow::ChunkedArray$create(c(7, 8)))), c(7, 8))
  expect_equal(length(am_array(arrow::ChunkedArray$create(numeric(0)))), 0L)
})

test_that("an NA scalar comparison gives an all-null mask, as base R and arrow do", {
  skip_without_gpu()
  cases <- list(
    # `arrow_scalar` is FALSE where arrow cannot even build the Scalar: Scalar$create(NA,
    # type = float64()) raises "Invalid: cannot convert", so only base R can be the oracle there.
    double  = list(x = c(1, 5, 9),    na = NA_real_,    arrow_scalar = TRUE),
    integer = list(x = c(1L, 5L, 9L), na = NA_integer_, arrow_scalar = TRUE),
    logical = list(x = c(1, 5, 9),    na = NA,          arrow_scalar = FALSE)
  )
  for (nm in names(cases)) {
    x <- cases[[nm]]$x
    na <- cases[[nm]]$na
    a <- arrow::Array$create(x)
    got <- rvec(am_compare(a, ">", na))
    expect_equal(got, rep(NA, length(x)), info = nm)
    expect_equal(got, x > na, info = nm)                       # base R
    if (cases[[nm]]$arrow_scalar) {                            # arrow's own kernel
      expect_equal(got, as.vector(arrow::call_function(
        "greater", a, arrow::Scalar$create(na, type = a$type))), info = nm)
    }
    expect_equal(am_null_count(am_compare(a, ">", na)), length(x), info = nm)
  }
  # Every operator behaves the same way.
  for (op in c("==", "!=", "<", "<=", ">", ">="))
    expect_equal(rvec(am_compare(c(1, 2), op, NA_real_)), c(NA, NA), info = op)
})

test_that("NaN is a value, not a null, in a scalar comparison", {
  skip_without_gpu()
  # is.na(NaN) is TRUE, so NaN must not be routed down the NA path.
  got <- rvec(am_compare(arrow::Array$create(c(1, NaN, 3)), ">", NaN))
  expect_false(all(is.na(got)))
})

test_that("integer64 = TRUE on an empty or all-null column gives bit64's NA, not garbage", {
  skip_without_gpu()
  skip_if_not_installed("bit64")
  empty <- arrow::Array$create(numeric(0), type = arrow::int64())
  allna <- arrow::Array$create(c(NA_real_, NA_real_), type = arrow::int64())
  for (a in list(empty, allna)) {
    for (f in list(am_sum, am_min, am_max)) {
      v <- f(a, integer64 = TRUE)
      expect_s3_class(v, "integer64")
      expect_true(is.na(v))
      expect_identical(v, bit64::NA_integer64_)
      # and the plain double path agrees with arrow's own answer
      expect_true(is.na(f(a)))
    }
  }
  expect_true(is.na(arrow::call_function("sum", empty)$as_vector()))
  expect_true(is.na(arrow::call_function("sum", allna)$as_vector()))
})

test_that("sliced boolean and string columns import as the rows they name", {
  skip_without_gpu()
  b <- arrow::Array$create(c(TRUE, FALSE, NA, TRUE, FALSE, TRUE))
  s <- arrow::Array$create(c("alpha", NA, "gamma", "delta", "epsilon", "zeta"))
  for (off in c(0L, 1L, 3L)) {
    bs <- b$Slice(off, 3L)
    ss <- s$Slice(off, 3L)
    expect_equal(rvec(am_array(bs)), as.vector(b)[(off + 1):(off + 3)], info = off)
    expect_equal(rvec(am_array(ss)), as.vector(s)[(off + 1):(off + 3)], info = off)
    expect_equal(am_null_count(am_array(ss)), sum(is.na(as.vector(s)[(off + 1):(off + 3)])))
  }
  # A slice of a slice, and selection on a sliced string column.
  expect_equal(rvec(am_slice(am_array(s$Slice(1L, 4L)), 1, 2)), c("gamma", "delta"))
  expect_equal(rvec(am_sort(am_array(s$Slice(2L, 3L)))), c("delta", "epsilon", "gamma"))
  expect_equal(rvec(am_take(am_array(b$Slice(1L, 3L)), c(2L, 0L))), c(TRUE, FALSE))
})

test_that("bad take indices are an error, not a silent NA", {
  skip_without_gpu()
  expect_error(am_take(c(1, 2, 3), 1e10), "whole numbers")
  expect_error(am_take(c(1, 2, 3), -1), "whole numbers")
  expect_error(am_take(c(1, 2, 3), 1.5), "whole numbers")
  expect_error(am_take(c(1, 2, 3), "a"), "must be numeric")
  # A numeric index that does fit is still fine, and so is a null index.
  expect_equal(rvec(am_take(c(1, 2, 3), c(2, 0))), c(3, 1))
  expect_equal(rvec(am_take(c(1, 2, 3), c(1L, NA))), c(2, NA))
})

test_that("a non-numeric scalar in am_compare says what is wrong", {
  skip_without_gpu()
  expect_error(am_compare(am_array(c(1, 2)), ">", "abc"),
               "scalar operations need a numeric array")
  expect_error(am_compare(am_array(c(1, 2)), ">", list(1)),
               "scalar operations need a numeric array")
})

test_that("a plan source with mismatched names and columns is an error, not a crash", {
  skip_without_gpu()
  h <- list(am_array(c(1, 2)), am_array(c(3, 4)))
  expect_error(.Call(arrowmetal:::C_am_plan_source_create, "t", h, c("only_one")),
               "2 column handles but 1 names")
  expect_error(.Call(arrowmetal:::C_am_plan_source_create, "t", h, character(0)),
               "2 column handles but 0 names")
  expect_error(.Call(arrowmetal:::C_am_plan_source_create, "t", h, list("a", "b")),
               "column handles but")
})

test_that("am_version and am_device_name always give a string", {
  skip_without_gpu()
  expect_type(am_version(), "character")
  expect_length(am_version(), 1L)
  expect_type(am_device_name(), "character")
  expect_length(am_device_name(), 1L)
})
