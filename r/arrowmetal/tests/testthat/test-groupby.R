test_that("a grouped sum over string keys matches tapply", {
  skip_without_gpu()
  set.seed(8)
  keys <- sample(c("north", "south", "east", "west"), 20000, replace = TRUE)
  vals <- runif(20000) * 100
  g <- am_group_by(keys)
  k <- rvec(g$keys())
  expect_equal(g$n_groups, 4)
  expect_equal(sort(k), sort(unique(keys)))
  expect_equal(rvec(g$sum(vals)), as.numeric(tapply(vals, keys, sum)[k]))
  expect_equal(rvec(g$mean(vals)), as.numeric(tapply(vals, keys, mean)[k]))
  expect_equal(rvec(g$min(vals)), as.numeric(tapply(vals, keys, min)[k]))
  expect_equal(rvec(g$max(vals)), as.numeric(tapply(vals, keys, max)[k]))
  expect_equal(rvec(g$count_all()), unname(as.numeric(table(keys)[k])))
})

test_that("a grouped sum over integer keys matches tapply and is in ascending key order", {
  skip_without_gpu()
  set.seed(9)
  keys <- sample(0:9, 100000, replace = TRUE)
  vals <- as.numeric(sample.int(1000, 100000, replace = TRUE))
  g <- am_group_by(arrow::Array$create(keys))
  expect_equal(rvec(g$keys()), 0:9)
  expect_equal(rvec(g$sum(vals)), as.numeric(tapply(vals, keys, sum)))
  expect_equal(rvec(g$count_all()), unname(as.numeric(table(keys))))
})

test_that("nulls in the values are skipped and nulls in the keys form their own group", {
  skip_without_gpu()
  keys <- c("a", "b", NA, "a", NA)
  vals <- c(1, 2, 3, NA, 5)
  g <- am_group_by(keys)
  k <- rvec(g$keys())
  s <- rvec(g$sum(vals))
  names(s) <- ifelse(is.na(k), "<null>", k)
  expect_equal(g$n_groups, 3)
  expect_equal(unname(s[["a"]]), 1)
  expect_equal(unname(s[["b"]]), 2)
  expect_equal(unname(s[["<null>"]]), 8)
  expect_equal(rvec(g$count(vals))[match("a", k)], 1)
})

test_that("two key columns group on the pair", {
  skip_without_gpu()
  a <- c("x", "x", "y", "y", "x")
  b <- c(1L, 2L, 1L, 1L, 1L)
  g <- am_group_by(a, b)
  expect_equal(g$n_groups, 3)
  k <- paste(rvec(g$keys(1)), rvec(g$keys(2)))
  s <- rvec(g$sum(c(1, 10, 100, 1000, 2)))
  names(s) <- k
  expect_equal(unname(s[["x 1"]]), 3)
  expect_equal(unname(s[["x 2"]]), 10)
  expect_equal(unname(s[["y 1"]]), 1100)
})

test_that("$ids() labels every row and matches the key order", {
  skip_without_gpu()
  keys <- c("b", "a", "b", "c")
  g <- am_group_by(keys)
  ids <- rvec(g$ids())
  expect_equal(length(ids), 4L)
  expect_equal(rvec(g$keys())[ids + 1L], keys)
})

test_that("one group per row and one group for everything both work", {
  skip_without_gpu()
  n <- 50000L
  every <- am_group_by(seq_len(n) - 1L)
  expect_equal(every$n_groups, n)
  expect_equal(rvec(every$sum(as.numeric(seq_len(n)))), as.numeric(seq_len(n)))
  one <- am_group_by(rep(0L, n))
  expect_equal(one$n_groups, 1)
  expect_equal(rvec(one$sum(rep(1, n))), n)
})

test_that("a grouped aggregate over a threadgroup-crossing column matches tapply", {
  skip_without_gpu()
  set.seed(10)
  keys <- sample(0:31, BIG, replace = TRUE)
  vals <- runif(BIG)
  g <- am_group_by(arrow::Array$create(keys))
  expect_equal(rvec(g$sum(vals)), as.numeric(tapply(vals, keys, sum)))
  expect_equal(rvec(g$mean(vals)), as.numeric(tapply(vals, keys, mean)), tolerance = 1e-9)
})

test_that("am_group_by accepts a data frame and rejects nothing", {
  skip_without_gpu()
  g <- am_group_by(data.frame(a = c("p", "q", "p"), b = c(1L, 1L, 1L)))
  expect_equal(g$n_keys, 2L)
  expect_equal(g$n_groups, 2)
  expect_error(am_group_by(), "at least one key column")
})
