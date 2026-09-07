sales_source <- function(n = 10000L, seed = 12) {
  set.seed(seed)
  region <- sample(c("north", "south", "east"), n, replace = TRUE)
  amount <- as.numeric(sample.int(500L, n, replace = TRUE))
  list(
    source = am_plan_source("sales", list(region = region, amount = amount)),
    region = region, amount = amount
  )
}

test_that("a scan plan returns the columns unchanged", {
  skip_without_gpu()
  s <- sales_source(1000L)
  r <- am_plan_run('{"op":"scan","source":"sales"}', s$source)
  expect_s3_class(r, "am_plan_result")
  expect_equal(names(r), c("region", "amount"))
  expect_equal(attr(r, "nrow"), 1000)
  expect_equal(rvec(r$amount), s$amount)
})

test_that("filter + group_by + sort in one plan matches the same thing in R", {
  skip_without_gpu()
  s <- sales_source()
  plan <- paste0(
    '{"op":"sort","by":[["total",true]],"input":',
    '{"op":"group_by","keys":[["region","(col \\"region\\")"]],',
    '"aggs":[["sum","total","(col \\"amount\\")"]],"input":',
    '{"op":"filter","predicate":"(gt (col \\"amount\\") (float 100.0))","input":',
    '{"op":"scan","source":"sales"}}}}')
  r <- am_plan_run(plan, s$source)
  keep <- s$amount > 100
  expected <- tapply(s$amount[keep], s$region[keep], sum)
  expected <- sort(expected, decreasing = TRUE)
  expect_equal(rvec(r$region), names(expected))
  expect_equal(rvec(r$total), as.numeric(expected))
})

test_that("a limit plan truncates", {
  skip_without_gpu()
  s <- sales_source(1000L)
  r <- am_plan_run('{"op":"limit","count":7,"input":{"op":"scan","source":"sales"}}', s$source)
  expect_equal(attr(r, "nrow"), 7)
  expect_equal(rvec(r$amount), s$amount[1:7])
})

test_that("a plan can be given as an R list when jsonlite is installed", {
  skip_without_gpu()
  skip_if_not_installed("jsonlite")
  s <- sales_source(500L)
  r <- am_plan_run(list(op = "limit", count = 3,
                        input = list(op = "scan", source = "sales")), s$source)
  expect_equal(attr(r, "nrow"), 3)
})

test_that("am_plan_explain returns the optimized and physical plans", {
  skip_without_gpu()
  s <- sales_source(100L)
  txt <- am_plan_explain('{"op":"scan","source":"sales"}', s$source)
  expect_type(txt, "character")
  expect_true(nchar(txt) > 0)
})

test_that("a plan that does not type-check is an R error", {
  skip_without_gpu()
  s <- sales_source(100L)
  expect_error(am_plan_run('{"op":"scan","source":"nope"}', s$source))
  expect_error(am_plan_run('{"op":"filter","predicate":"(gt (col \\"missing\\") (int 1))",
                             "input":{"op":"scan","source":"sales"}}', s$source))
})

test_that("plan sources are validated", {
  skip_without_gpu()
  expect_error(am_plan_source("t", list()), "at least one column")
  expect_error(am_plan_source("t", list(c(1, 2))), "needs a name")
  expect_error(am_plan_run('{"op":"scan","source":"t"}', list(1)), "am_plan_source")
})

test_that("a plan result prints its shape", {
  skip_without_gpu()
  s <- sales_source(10L)
  r <- am_plan_run('{"op":"scan","source":"sales"}', s$source)
  expect_output(print(r), "am_plan_result 10 row")
})
