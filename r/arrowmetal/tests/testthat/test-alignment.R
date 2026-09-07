test_that("am_buffer_alignment reports one row per Arrow buffer", {
  skip_without_gpu()
  d <- am_buffer_alignment(arrow::Array$create(c(1, 2, 3)))
  expect_s3_class(d, "data.frame")
  expect_equal(nrow(d), 2L)          # validity + values
  expect_equal(d$buffer, 0:1)
  expect_true(all(d$page_size > 0))
  # An array with no nulls has no validity buffer; arrow reports it as a NULL pointer.
  expect_true(is.na(d$page_aligned[1]))
  expect_false(is.na(d$page_aligned[2]))
})

test_that("the values buffer offset is inside a page", {
  skip_without_gpu()
  d <- am_buffer_alignment(arrow::Array$create(runif(1e6)))
  off <- d$offset_in_page[d$buffer == 1L]
  expect_true(off >= 0 && off < d$page_size[1])
})

test_that("a copied import still gives the right answer", {
  # This is the whole point of measuring alignment: whichever path the buffers take, the values
  # must be identical. arrow R's buffers are not page aligned (see README.md), so this exercises
  # the copying path.
  skip_without_gpu()
  set.seed(13)
  x <- runif(1e5)
  expect_equal(rvec(am_array(arrow::Array$create(x))), x)
})
