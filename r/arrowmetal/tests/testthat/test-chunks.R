# The chunked import: a ChunkedArray or a list of arrays, one am_array, equal to the import of the
# concatenation.

chunk_types <- list(
  int8 = arrow::int8(), uint16 = arrow::uint16(), int32 = arrow::int32(), int64 = arrow::int64(),
  float32 = arrow::float32(), float64 = arrow::float64(), bool = arrow::boolean(),
  date32 = arrow::date32(), timestamp = arrow::timestamp("us"), decimal = arrow::decimal128(18, 2),
  utf8 = arrow::utf8(), large_utf8 = arrow::large_utf8(), binary = arrow::binary()
)

gen_values <- function(type_name, n, seed) {
  set.seed(seed)
  v <- sample(-5000:5000, n, replace = TRUE)
  switch(type_name,
    int8 = v %% 100L, uint16 = abs(v), int32 = v, int64 = v,
    float32 = v / 8, float64 = v / 16, bool = v %% 2L == 0L,
    date32 = as.Date(v, origin = "1970-01-01"),
    timestamp = as.POSIXct(v * 1000, origin = "1970-01-01", tz = "UTC"),
    decimal = v / 100,
    utf8 = paste0("s", v %% 1000L), large_utf8 = paste0("L", v %% 777L),
    binary = lapply(v, function(i) as.raw(c(i %% 256L, (i %/% 7L) %% 256L)))
  )
}

gen_chunk <- function(type_name, n, null_every, seed) {
  if (n == 0) return(arrow::concat_arrays(type = chunk_types[[type_name]]))
  v <- gen_values(type_name, n, seed)
  if (n > 0 && null_every > 0) {
    is_na <- seq_len(n) %% null_every == 1L | null_every == 1L
    if (type_name == "binary") v[is_na] <- list(NULL) else v[is_na] <- NA
  }
  if (type_name == "binary") {
    return(arrow::Array$create(if (n == 0) list() else v, type = arrow::binary()))
  }
  arrow::Array$create(v, type = chunk_types[[type_name]])
}

layouts <- list(
  mixed = function(tn) {
    big <- gen_chunk(tn, 5000L, 7L, 1L)
    list(gen_chunk(tn, 0L, 0L, 2L), gen_chunk(tn, 1L, 0L, 3L), big$Slice(3, 1200),
         gen_chunk(tn, 64L, 1L, 4L), gen_chunk(tn, 1000L, 0L, 5L), big$Slice(4001, 1),
         gen_chunk(tn, 0L, 0L, 6L), big$Slice(1999), gen_chunk(tn, 1L, 1L, 7L),
         gen_chunk(tn, 333L, 3L, 8L))
  },
  one_chunk = function(tn) list(gen_chunk(tn, 777L, 5L, 9L)),
  all_empty = function(tn) list(gen_chunk(tn, 0L, 0L, 1L), gen_chunk(tn, 0L, 0L, 2L)),
  all_null = function(tn) list(gen_chunk(tn, 10L, 1L, 1L), gen_chunk(tn, 1L, 1L, 2L),
                               gen_chunk(tn, 100L, 1L, 3L)),
  many_small = function(tn) lapply(0:299, function(i) gen_chunk(tn, i %% 5L, 3L, i))
)

test_that("am_array_chunks equals the import of the concatenation, every type and layout", {
  skip_without_gpu()
  for (tn in names(chunk_types)) {
    for (ln in names(layouts)) {
      chunks <- layouts[[ln]](tn)
      merged <- do.call(arrow::concat_arrays, chunks)
      ca <- do.call(arrow::chunked_array, c(chunks, list(type = chunk_types[[tn]])))
      label <- paste(tn, ln)

      from_list <- as_arrow_array(am_array_chunks(chunks))
      from_ca <- as_arrow_array(am_array_chunks(ca))
      via_am_array <- as_arrow_array(am_array(ca))
      ref <- as_arrow_array(am_array(merged))

      expect_true(from_list$Equals(ref), label = paste(label, "list"))
      expect_true(from_ca$Equals(ref), label = paste(label, "ChunkedArray"))
      expect_true(via_am_array$Equals(ref), label = paste(label, "am_array"))
      if (from_list$type$Equals(merged$type))
        expect_true(from_list$Equals(merged), label = paste(label, "concatenation"))
      expect_identical(from_list$null_count, merged$null_count, label = label)
    }
  }
})

test_that("chunked columns give the same answers through the kernels", {
  skip_without_gpu()
  chunks <- lapply(1:30, function(i) {
    c <- gen_chunk("int64", 20000L + i, 9L, i)
    if (i %% 4L == 1L) c$Slice(17, 20000L - 20L) else c
  })
  ca <- do.call(arrow::chunked_array, chunks)
  h <- am_array_chunks(ca)
  r <- am_array(do.call(arrow::concat_arrays, chunks))
  expect_identical(am_sum(h), am_sum(r))
  expect_identical(am_max(h), am_max(r))
  for (np in c("at_end", "at_start")) {
    expect_identical(as.vector(as_arrow_array(am_argsort(h, TRUE, np))),
                     as.vector(as_arrow_array(am_argsort(r, TRUE, np))))
  }
})

test_that("a Table's chunked columns feed a plan source", {
  skip_without_gpu()
  batches <- lapply(0:5, function(i) {
    n <- 1000L * i
    arrow::record_batch(k = gen_chunk("int64", n, 5L, i), s = gen_chunk("utf8", n, 3L, i + 100L))
  })
  tbl <- do.call(arrow::Table$create, batches)
  expect_gt(tbl$k$num_chunks, 1L)
  src <- am_plan_source("b", lapply(setNames(tbl$columns, names(tbl)), am_array_chunks))
  res <- am_plan_run(paste0('{"op":"aggregate","aggs":[["sum","t","(col \\"k\\")"],',
                            '["count","c","(col \\"s\\")"]],"input":{"op":"scan","source":"b"}}'), src)
  k <- as.vector(tbl$k)
  expect_equal(am_sum(res$t), sum(k, na.rm = TRUE))
  expect_equal(am_sum(res$c), sum(!is.na(as.vector(tbl$s))))
})

test_that("empty, one-chunk, dictionary and mixed-type inputs", {
  skip_without_gpu()
  empty <- arrow::chunked_array(type = arrow::float64())
  h <- am_array_chunks(empty)
  expect_identical(length(h), 0L)
  expect_identical(am_format(h), "g")

  one <- arrow::chunked_array(c(1, NA, 3))
  expect_identical(as.vector(am_array_chunks(one)), c(1, NA, 3))

  # dictionary: not taken by the chunked import, concatenated instead, same answer
  # int32 indices: an R factor becomes int8 indices, which ArrowMetal's import does not take
  dict <- arrow::dictionary(arrow::int32(), arrow::utf8())
  d1 <- arrow::Array$create(factor(c("a", "b", NA, "a")))$cast(dict)
  d2 <- arrow::Array$create(factor(c("b", "b", "c")))$cast(dict)
  hd <- am_array_chunks(list(d1, d2))
  expect_true(as_arrow_array(hd)$Equals(as_arrow_array(am_array(arrow::concat_arrays(d1, d2)))))

  expect_error(am_array_chunks(list(arrow::Array$create(1:3), arrow::Array$create(c(1.5, 2)))),
               "one type")
  expect_error(am_array_chunks(list()), "at least one chunk")
  expect_error(am_array_chunks(1:3), "ChunkedArray or a list")

  # plain R vectors in a list become arrays first
  expect_identical(as.vector(am_array_chunks(list(c(1, 2), c(NA, 4)))), c(1, 2, NA, 4))
})

test_that("chunk carriers are released once and the import outlives the chunks", {
  skip_without_gpu()
  chunks <- list(gen_chunk("float64", 3000L, 4L, 1L), gen_chunk("float64", 5000L, 0L, 2L))
  want <- as.vector(do.call(arrow::concat_arrays, chunks))
  h <- am_array_chunks(chunks)
  rm(chunks)
  invisible(gc())
  invisible(gc())
  expect_identical(as.vector(h), want)
  # repeated imports and collections: a double release would crash the process
  for (i in 1:200) {
    x <- am_array_chunks(list(gen_chunk("int32", 50L, 3L, i), gen_chunk("int32", 7L, 2L, i + 1L)))
    if (i %% 50 == 0) invisible(gc())
  }
  invisible(gc())
  expect_identical(length(x), 57L)
})

test_that("the docs/R.md chunked-column lines", {
  skip_without_gpu()
  ca <- arrow::chunked_array(c(1, NA), numeric(0), c(3, 4, 5))
  h <- am_array_chunks(ca)
  expect_identical(length(h), 5L)
  expect_equal(am_null_count(h), 1)
  expect_equal(am_sum(h), 13)
  expect_equal(am_sum(am_array(ca)), 13)
})
