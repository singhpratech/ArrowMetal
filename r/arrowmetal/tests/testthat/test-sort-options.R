# Sort options: null placement and IEEE 754 totalOrder, against arrow's own sort and a byte-key
# reference built from the values' bits.

dbl_bits <- function(hex) readBin(as.raw(strtoi(substring(hex, seq(1, 15, 2), seq(2, 16, 2)), 16L)),
                                  "double", endian = "big")

# NaN of both signs and several payloads, both zeros, both infinities, subnormals.
awkward <- c(NaN, dbl_bits("fff8000000000000"), dbl_bits("7ff0000000000001"),
             dbl_bits("fff8000000000042"), dbl_bits("7ff8000000000007"),
             0, -0, Inf, -Inf, 5e-324, -1.5e-323, 1.5, -1.5)

awkward_col <- function(n, null_every, seed) {
  set.seed(seed)
  x <- ifelse(runif(n) < 1 / 3, sample(awkward, n, replace = TRUE), round(runif(n, -50, 50)) / 4)
  x[seq_len(n) %% null_every == 1] <- NA
  arrow::Array$create(as.double(x), type = arrow::float64())
}

# Bits of each value as a raw matrix, one column per value (big-endian: byte 1 is the sign byte).
bits_of <- function(x) matrix(writeBin(as.double(x), raw(), endian = "big"), nrow = 8)

# Zero-based reference order under totalOrder for the valid rows, nulls placed as asked.
ref_total <- function(a, descending, null_placement) {
  x <- as.vector(a)
  valid <- which(!as.vector(arrow::call_function("is_null", a)))
  b <- bits_of(x[valid])
  neg <- b[1, ] >= as.raw(0x80)
  keys <- lapply(1:8, function(r) {
    v <- as.integer(b[r, ])
    ifelse(neg, 255L - v, if (r == 1) v + 128L else v)
  })
  o <- do.call(order, c(keys, list(seq_along(valid), method = "radix",
                                   decreasing = c(rep(descending, 8), FALSE))))
  nulls <- setdiff(seq_along(x), valid)
  idx <- valid[o]
  (if (null_placement == "at_start") c(nulls, idx) else c(idx, nulls)) - 1L
}

# ieee reference from arrow's own sort (Arrow C++ order, nulls and NaN at the end): at_start moves
# the null rows, then the NaN rows, in front, each keeping its order.
ref_ieee <- function(a, descending, null_placement) {
  at_end <- as.vector(arrow::call_function("array_sort_indices", a,
                                           options = list(order = descending)))
  if (null_placement == "at_end") return(at_end)
  x <- as.vector(a)
  isnull <- as.vector(arrow::call_function("is_null", a))
  nulls <- at_end[isnull[at_end + 1L]]
  nans <- at_end[!isnull[at_end + 1L] & is.nan(x[at_end + 1L])]
  c(nulls, nans, setdiff(at_end, c(nulls, nans)))
}

combos <- expand.grid(descending = c(FALSE, TRUE), null_placement = c("at_end", "at_start"),
                      float_order = c("ieee", "total"), stringsAsFactors = FALSE)

test_that("am_argsort with options matches the reference for every combination", {
  skip_without_gpu()
  for (n in c(0L, 1L, 33L, 1025L, 100001L)) {
    a <- awkward_col(n, 7L, n + 3L)
    h <- am_array(a)
    for (i in seq_len(nrow(combos))) {
      cb <- combos[i, ]
      got <- as.vector(as_arrow_array(am_argsort(h, cb$descending, cb$null_placement, cb$float_order)))
      ref <- if (cb$float_order == "total") ref_total(a, cb$descending, cb$null_placement) else
        ref_ieee(a, cb$descending, cb$null_placement)
      expect_identical(got, as.integer(ref),
                       label = sprintf("n=%d %s", n, paste(unlist(cb), collapse = "/")))
    }
  }
})

test_that("am_sort with options is bit-exact against the reference order", {
  skip_without_gpu()
  a <- awkward_col(20011L, 9L, 5L)
  x <- as.vector(a)
  isnull <- as.vector(arrow::call_function("is_null", a))
  for (i in seq_len(nrow(combos))) {
    cb <- combos[i, ]
    ref <- if (cb$float_order == "total") ref_total(a, cb$descending, cb$null_placement) else
      ref_ieee(a, cb$descending, cb$null_placement)
    out <- as_arrow_array(am_sort(a, cb$descending, cb$null_placement, cb$float_order))
    got_null <- as.vector(arrow::call_function("is_null", out))
    expect_identical(got_null, isnull[ref + 1L])
    keep <- !got_null
    expect_identical(bits_of(as.vector(out)[keep]), bits_of(x[ref + 1L][keep]),
                     label = paste(unlist(cb), collapse = "/"))
  }
})

test_that("integer columns: null placement in both directions against arrow", {
  skip_without_gpu()
  set.seed(4)
  v <- sample(-300:300, 50001L, replace = TRUE)
  v[sample.int(50001L, 5000L)] <- NA
  a <- arrow::Array$create(v, type = arrow::int64())
  for (desc in c(FALSE, TRUE)) {
    at_end <- as.vector(arrow::call_function("array_sort_indices", a, options = list(order = desc)))
    nulls <- at_end[is.na(v[at_end + 1L])]
    at_start <- c(nulls, setdiff(at_end, nulls))
    for (fo in c("ieee", "total")) {
      expect_identical(as.vector(as_arrow_array(am_argsort(a, desc, "at_end", fo))), at_end)
      expect_identical(as.vector(as_arrow_array(am_argsort(a, desc, "at_start", fo))), at_start)
    }
  }
})

test_that("float32 columns order by totalOrder", {
  skip_without_gpu()
  set.seed(8)
  x <- ifelse(runif(5003) < 0.5, sample(c(NaN, dbl_bits("fff8000000000000"), 0, -0, Inf, -Inf, 2.5),
                                        5003, replace = TRUE), round(runif(5003, -20, 20)) / 2)
  x[seq_along(x) %% 11 == 0] <- NA
  a32 <- arrow::Array$create(x, type = arrow::float32())
  a64 <- arrow::Array$create(x, type = arrow::float64())  # widening keeps sign, class and order
  for (i in seq_len(nrow(combos))) {
    cb <- combos[i, ]
    if (cb$float_order != "total") next
    expect_identical(as.vector(as_arrow_array(am_argsort(a32, cb$descending, cb$null_placement, "total"))),
                     as.integer(ref_total(a64, cb$descending, cb$null_placement)))
  }
})

test_that("the defaults are the plain calls", {
  skip_without_gpu()
  a <- awkward_col(30001L, 13L, 2L)
  for (desc in c(FALSE, TRUE)) {
    expect_identical(as.vector(as_arrow_array(am_argsort(a, desc))),
                     as.integer(ref_ieee(a, desc, "at_end")))
    expect_identical(as.vector(as_arrow_array(am_top_k(a, 50, largest = desc))),
                     as.vector(as_arrow_array(am_argsort(a, desc)))[1:50])
  }
})

test_that("am_top_k with options is the head of am_argsort with the same options", {
  skip_without_gpu()
  a <- awkward_col(100001L, 8L, 21L)
  h <- am_array(a)
  for (i in seq_len(nrow(combos))) {
    cb <- combos[i, ]
    full <- as.vector(as_arrow_array(am_argsort(h, cb$descending, cb$null_placement, cb$float_order)))
    for (k in c(0, 1, 10, 1000, 30000, 100010)) {
      got <- as.vector(as_arrow_array(am_top_k(h, k, cb$descending, cb$null_placement, cb$float_order)))
      expect_identical(got, full[seq_len(min(k, length(full)))],
                       label = sprintf("k=%d %s", k, paste(unlist(cb), collapse = "/")))
    }
  }
  expect_error(am_top_k(h, -1), "whole number")
  expect_error(am_top_k(h, 2.5), "whole number")
  expect_error(am_argsort(h, null_placement = "first"), "null_placement")
  expect_error(am_argsort(h, float_order = "totalorder"), "float_order")
})

test_that("am_lexsort with per-key options matches a stable reference", {
  skip_without_gpu()
  set.seed(12)
  n <- 20011L
  k1 <- sample(0:5, n, replace = TRUE)
  k1[seq_len(n) %% 10 == 3] <- NA
  a1 <- arrow::Array$create(k1, type = arrow::int32())
  a2 <- awkward_col(n, 9L, 77L)
  # rank of each row under one key's options: position in that key's own argsort, ties shared
  key_rank <- function(a, desc, np, fo) {
    idx <- as.vector(as_arrow_array(am_argsort(a, desc, np, fo))) + 1L
    x <- as.vector(a)
    isnull <- as.vector(arrow::call_function("is_null", a))
    b <- if (is.double(x)) apply(bits_of(ifelse(isnull, 0, x)), 2, paste, collapse = "") else
      as.character(x)
    if (fo == "ieee" && is.double(x)) b[is.nan(x) & !isnull] <- "nan"
    if (fo == "ieee" && is.double(x)) b[!isnull & !is.nan(x) & x == 0] <- "zero"
    b[isnull] <- "null"
    sorted <- b[idx]
    grp <- cumsum(c(TRUE, sorted[-1] != sorted[-length(sorted)]))
    r <- integer(length(x))
    r[idx] <- grp
    r
  }
  for (i in seq_len(nrow(combos))) {
    for (j in seq_len(nrow(combos))) {
      c1 <- combos[i, ]
      c2 <- combos[j, ]
      if (c1$float_order == "total") next  # an integer key ignores the float order
      got <- as.vector(as_arrow_array(am_lexsort(list(a1, a2),
                                                 descending = c(c1$descending, c2$descending),
                                                 null_placement = c(c1$null_placement, c2$null_placement),
                                                 float_order = c(c1$float_order, c2$float_order))))
      r1 <- key_rank(a1, c1$descending, c1$null_placement, c1$float_order)
      r2 <- key_rank(a2, c2$descending, c2$null_placement, c2$float_order)
      expect_identical(got, order(r1, r2, method = "radix") - 1L,
                       label = paste(c(unlist(c1), unlist(c2)), collapse = "/"))
    }
  }
  expect_error(am_lexsort(list(a1, a2), descending = c(TRUE, FALSE, TRUE)), "3 entries")
  # the plain form, all defaults, agrees with base R's order on integer keys without nulls
  p1 <- sample(0:3, 1000, replace = TRUE)
  p2 <- sample(0:50, 1000, replace = TRUE)
  expect_identical(as.vector(as_arrow_array(am_lexsort(list(p1, p2), c(FALSE, TRUE)))),
                   order(p1, -p2, method = "radix") - 1L)
})

test_that("a plan's sort key takes nulls and float_order", {
  skip_without_gpu()
  a <- awkward_col(5003L, 4L, 31L)
  src <- am_plan_source("t", list(x = a))
  plan <- paste0('{"op":"sort","by":[{"column":"x","descending":true,"nulls":"first",',
                 '"float_order":"total"}],"input":{"op":"scan","source":"t"}}')
  out <- as_arrow_array(am_plan_run(plan, src)$x)
  want <- as_arrow_array(am_sort(a, TRUE, "at_start", "total"))
  expect_identical(as.vector(arrow::call_function("is_null", out)),
                   as.vector(arrow::call_function("is_null", want)))
  keep <- !as.vector(arrow::call_function("is_null", out))
  expect_identical(bits_of(as.vector(out)[keep]), bits_of(as.vector(want)[keep]))
})

test_that("the docs/R.md sort-options lines", {
  skip_without_gpu()
  x <- c(2, NA, NaN, -0, 7)
  expect_identical(as.vector(am_argsort(x, descending = TRUE)), c(4L, 0L, 3L, 2L, 1L))
  expect_identical(as.vector(am_argsort(x, descending = TRUE, null_placement = "at_start",
                                        float_order = "total")), c(1L, 2L, 4L, 0L, 3L))
  expect_identical(as.vector(am_top_k(x, 2, null_placement = "at_start", float_order = "total")),
                   c(1L, 2L))
  expect_identical(as.vector(am_lexsort(list(c(1, 1, 2), c(3, NA, 1)), descending = c(FALSE, TRUE),
                                        null_placement = c("at_end", "at_start"))), c(1L, 0L, 2L))
})
