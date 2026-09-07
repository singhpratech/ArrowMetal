test_that("the library loads and identifies itself", {
  skip_without_gpu()
  expect_type(am_version(), "character")
  expect_true(nzchar(am_device_name()))
  expect_true(file.exists(am_lib_path()))
})

test_that("a double column round trips through the C Data Interface", {
  skip_without_gpu()
  x <- c(1, 2.5, -3, 1e300, 0)
  expect_equal(rvec(am_array(x)), x)
  expect_equal(am_format(am_array(x)), "g")
})

test_that("nulls survive the round trip and are counted", {
  skip_without_gpu()
  x <- c(1, NA, 3, NA, 5)
  a <- am_array(x)
  expect_equal(am_null_count(a), 2)
  expect_equal(rvec(a), x)
})

test_that("an empty array round trips", {
  skip_without_gpu()
  a <- am_array(numeric(0))
  expect_equal(length(a), 0L)
  expect_equal(am_null_count(a), 0)
  expect_equal(rvec(a), numeric(0))
  expect_true(is.na(am_sum(a)))
  expect_true(is.na(am_mean(a)))
})

test_that("an all-null array reduces to NA", {
  skip_without_gpu()
  a <- am_array(as.numeric(c(NA, NA, NA)))
  expect_equal(am_null_count(a), 3)
  expect_true(is.na(am_sum(a)))
  expect_true(is.na(am_min(a)))
  expect_true(is.na(am_max(a)))
  expect_true(is.na(am_mean(a)))
})

test_that("a sliced arrow array imports as the same rows", {
  skip_without_gpu()
  x <- c(10, 20, NA, 40, 50, 60, 70)
  full <- arrow::Array$create(x)
  for (off in c(0L, 1L, 3L)) {
    sliced <- full$Slice(off, 3L)
    a <- am_array(sliced)
    expect_equal(length(a), 3L)
    expect_equal(rvec(a), x[(off + 1):(off + 3)])
    expect_equal(am_sum(a), sum(x[(off + 1):(off + 3)], na.rm = TRUE))
  }
})

test_that("am_slice matches R subsetting", {
  skip_without_gpu()
  x <- as.numeric(1:100)
  expect_equal(rvec(am_slice(x, 10, 5)), x[11:15])
  expect_equal(rvec(am_slice(x, 0, 0)), numeric(0))
})

test_that("int32, int64, bool and string columns round trip", {
  skip_without_gpu()
  expect_equal(rvec(am_array(c(1L, NA, 3L))), c(1L, NA, 3L))
  expect_equal(am_format(am_array(c(1L, 2L))), "i")
  a <- am_array(i64(c(1, NA, 3)))
  expect_equal(am_format(a), "l")
  expect_equal(rvec(a), c(1, NA, 3))
  expect_equal(rvec(am_array(c(TRUE, NA, FALSE))), c(TRUE, NA, FALSE))
  expect_equal(am_format(am_array(c(TRUE, FALSE))), "b")
  expect_equal(rvec(am_array(c("a", NA, "ccc"))), c("a", NA, "ccc"))
})

test_that("int64 values above 2^31 round trip exactly", {
  skip_without_gpu()
  skip_if_not_installed("bit64")
  x <- bit64::as.integer64(c(2^40, -2^40 - 1, 0))
  a <- am_array(arrow::Array$create(x))
  expect_equal(am_format(a), "l")
  # arrow hands an out-of-int32-range int64 column back as a bit64::integer64.
  back <- as_arrow_array(a)$as_vector()
  expect_s3_class(back, "integer64")
  expect_true(all(back == x))
})

test_that("am_array is idempotent and as_arrow_array gives an arrow Array", {
  skip_without_gpu()
  a <- am_array(c(1, 2))
  expect_identical(am_array(a), a)
  expect_s3_class(as_arrow_array(a), "Array")
  expect_true(is_am_array(a))
  expect_false(is_am_array(1:3))
})

test_that("a length crossing a threadgroup boundary round trips exactly", {
  skip_without_gpu()
  set.seed(11)
  x <- runif(BIG)
  x[c(1, 7, BIG)] <- NA
  a <- am_array(x)
  expect_equal(length(a), BIG)
  expect_equal(am_null_count(a), 3)
  expect_equal(rvec(a), x)
})

test_that("errors from the C ABI reach R as R errors with the ArrowMetal message", {
  skip_without_gpu()
  # A boolean column has no defined minimum, so am_reduce fails inside the library.
  expect_error(am_min(am_array(c(TRUE, FALSE))), "ArrowMetal")
  # A mask of the wrong length is rejected by the filter kernel.
  expect_error(am_filter(am_array(c(1, 2, 3)), am_array(c(TRUE, FALSE))), "ArrowMetal")
  expect_error(am_compare(am_array(c(1, 2)), "~=", 1), "unknown comparison operator")
})

test_that("print methods say something useful", {
  skip_without_gpu()
  expect_output(print(am_array(c(1, 2, NA))), "am_array g len=3 nulls=1")
  expect_output(print(am_group_by(c(1L, 1L, 2L))), "am_groupby 1 key column")
})
