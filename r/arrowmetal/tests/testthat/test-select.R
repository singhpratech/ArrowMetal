test_that("am_compare against a scalar matches R, nulls included", {
  skip_without_gpu()
  x <- c(1, 5, NA, 9, 5)
  a <- am_array(x)
  expect_equal(rvec(am_compare(a, ">", 4)), x > 4)
  expect_equal(rvec(am_compare(a, ">=", 5)), x >= 5)
  expect_equal(rvec(am_compare(a, "<", 5)), x < 5)
  expect_equal(rvec(am_compare(a, "<=", 5)), x <= 5)
  expect_equal(rvec(am_compare(a, "==", 5)), x == 5)
  expect_equal(rvec(am_compare(a, "!=", 5)), x != 5)
})

test_that("am_compare against a scalar matches arrow's own kernel", {
  skip_without_gpu()
  set.seed(4)
  x <- round(runif(5000) * 100)
  x[c(3, 4000)] <- NA
  a <- arrow::Array$create(x)
  expected <- arrow::call_function("greater", a, arrow::Scalar$create(50))
  expect_equal(rvec(am_compare(a, ">", 50)), as.vector(expected))
})

test_that("am_compare against another column matches R", {
  skip_without_gpu()
  x <- c(1, 5, NA, 9)
  y <- c(2, 5, 3, NA)
  expect_equal(rvec(am_compare(am_array(x), "<", am_array(y))), x < y)
})

test_that("am_compare works on int64 and int32 columns", {
  skip_without_gpu()
  expect_equal(rvec(am_compare(i64(c(1, 5, 9)), ">", 4)), c(FALSE, TRUE, TRUE))
  expect_equal(rvec(am_compare(arrow::Array$create(c(1L, 5L, 9L)), ">", 4)), c(FALSE, TRUE, TRUE))
})

test_that("am_filter matches arrow's filter, dropping null mask rows", {
  skip_without_gpu()
  set.seed(5)
  x <- runif(70001)
  x[c(1, 500)] <- NA
  a <- arrow::Array$create(x)
  mask <- am_compare(a, ">", 0.5)
  got <- rvec(am_filter(a, mask))
  expected <- as.vector(arrow::call_function("filter", a,
                                             arrow::Array$create(x > 0.5)))
  expect_equal(got, expected)
  expect_equal(got, x[which(x > 0.5)])
})

test_that("am_filter on an empty and an all-false mask gives an empty column", {
  skip_without_gpu()
  expect_equal(length(am_filter(numeric(0), logical(0))), 0L)
  expect_equal(length(am_filter(c(1, 2, 3), c(FALSE, FALSE, FALSE))), 0L)
})

test_that("am_take matches R subsetting and accepts an R index vector", {
  skip_without_gpu()
  x <- c(10, 20, NA, 40)
  expect_equal(rvec(am_take(x, c(3L, 0L, 1L))), x[c(4, 1, 2)])
  expect_equal(rvec(am_take(x, am_array(arrow::Array$create(c(0L, 0L))))), x[c(1, 1)])
})

test_that("am_argsort matches order() with nulls last", {
  skip_without_gpu()
  set.seed(6)
  x <- round(runif(50000) * 1000)
  x[c(2, 10, 49999)] <- NA
  a <- arrow::Array$create(x)
  idx <- rvec(am_argsort(a)) + 1L
  expect_equal(x[idx], sort(x, na.last = TRUE))
  # A stable sort agrees with order()'s own stable tie-breaking.
  expect_equal(idx, order(x, na.last = TRUE))
})

test_that("am_argsort descending keeps nulls at the end", {
  skip_without_gpu()
  x <- c(3, NA, 1, 2)
  idx <- rvec(am_argsort(am_array(x), descending = TRUE)) + 1L
  expect_equal(x[idx], c(3, 2, 1, NA))
})

test_that("am_sort matches sort() and am_take(am_argsort())", {
  skip_without_gpu()
  set.seed(7)
  x <- runif(BIG)
  x[c(1, BIG)] <- NA
  a <- arrow::Array$create(x)
  s <- rvec(am_sort(a))
  expect_equal(s, sort(x, na.last = TRUE))
  expect_equal(s, rvec(am_take(a, am_argsort(a))))
})

test_that("selection works on an empty column", {
  skip_without_gpu()
  e <- am_array(numeric(0))
  expect_equal(length(am_sort(e)), 0L)
  expect_equal(length(am_argsort(e)), 0L)
})
