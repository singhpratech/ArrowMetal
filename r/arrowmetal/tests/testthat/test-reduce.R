# arrow returns an int64 aggregate as a bit64::integer64 when the value does not fit an R integer,
# and `as.vector()` on a bit64::integer64 reinterprets its bit pattern as a double (2474723182
# comes back as 1.22e-314), so the class has to be converted, never stripped.
arrow_scalar <- function(fn, x) {
  v <- arrow::call_function(fn, x)$as_vector()
  if (inherits(v, "integer64")) v <- as.numeric(v)
  as.vector(v)
}

test_that("double reductions match base R and arrow, with and without NAs", {
  skip_without_gpu()
  set.seed(1)
  for (n in c(1L, 33L, 1024L, 65537L)) {
    x <- rnorm(n) * 1000
    if (n > 3) x[c(2, n - 1)] <- NA
    a <- arrow::Array$create(x)
    expect_equal(am_sum(a), sum(x, na.rm = TRUE))
    expect_equal(am_min(a), min(x, na.rm = TRUE))
    expect_equal(am_max(a), max(x, na.rm = TRUE))
    expect_equal(am_mean(a), mean(x, na.rm = TRUE))
    # And against arrow's own kernels on the identical Array.
    expect_equal(am_sum(a), arrow_scalar("sum", a))
    expect_equal(am_mean(a), arrow_scalar("mean", a))
  }
})

test_that("double reductions match at a threadgroup-crossing length", {
  skip_without_gpu()
  set.seed(2)
  x <- runif(BIG) * 1e6
  x[seq(1, BIG, by = 1000)] <- NA
  a <- arrow::Array$create(x)
  expect_equal(am_sum(a), arrow_scalar("sum", a))
  expect_equal(am_min(a), min(x, na.rm = TRUE))
  expect_equal(am_max(a), max(x, na.rm = TRUE))
  expect_equal(am_mean(a), arrow_scalar("mean", a))
})

test_that("int64 reductions match arrow", {
  skip_without_gpu()
  set.seed(3)
  x <- as.numeric(sample.int(1e6, 5000))
  x[c(5, 100)] <- NA
  a <- i64(x)
  expect_equal(am_sum(a), sum(x, na.rm = TRUE))
  expect_equal(am_min(a), min(x, na.rm = TRUE))
  expect_equal(am_max(a), max(x, na.rm = TRUE))
  expect_equal(am_mean(a), mean(x, na.rm = TRUE))
  expect_equal(am_sum(a), arrow_scalar("sum", a))
})

test_that("int32 reductions match base R", {
  skip_without_gpu()
  x <- c(5L, NA, -7L, 100L, 0L)
  a <- arrow::Array$create(x)
  expect_equal(am_sum(a), sum(x, na.rm = TRUE))
  expect_equal(am_min(a), min(x, na.rm = TRUE))
  expect_equal(am_max(a), max(x, na.rm = TRUE))
})

test_that("an int64 sum beyond 2^53 is exact with bit64 and warns without it", {
  skip_without_gpu()
  skip_if_not_installed("bit64")
  big <- bit64::as.integer64(2)^53 + 1L
  a <- arrow::Array$create(rep(big, 4))
  expect_equal(am_format(am_array(a)), "l")
  exact <- am_sum(a, integer64 = TRUE)
  expect_s3_class(exact, "integer64")
  expect_true(exact == big * 4L)
  expect_warning(am_sum(a), "not representable")
})

test_that("min and max skip NaN, as Arrow does", {
  skip_without_gpu()
  x <- c(1, NaN, 3)
  a <- arrow::Array$create(x)
  expect_equal(am_min(a), 1)
  expect_equal(am_max(a), 3)
  expect_true(is.na(am_min(arrow::Array$create(c(NaN, NaN)))))
})

test_that("float32 sums accumulate in float64, like Arrow", {
  skip_without_gpu()
  x <- rep(0.1, 1000)
  a <- arrow::Array$create(x)$cast(arrow::float32())
  expect_equal(am_sum(a), arrow_scalar("sum", a), tolerance = 1e-9)
})
