# The chunked import (am_array_chunks) against concatenating first (arrow::concat_arrays, then
# am_array), both end to end. Not run by the test suite.
#
#   Rscript chunks.R <label> [reps] [rows,...]
#
# Each chunk is its own Array, as the batches of a Table are. Method as in overhead.R (100 ms of
# untimed calls, a 500 ms idle and one call on its own, then `reps` calls for best, median and
# CPU time per call). Prints CSV.

suppressMessages(library(arrowmetal))
suppressMessages(library(arrow))

args <- commandArgs(TRUE)
label <- if (length(args) >= 1) args[1] else ""
REPS <- if (length(args) >= 2) as.integer(args[2]) else 10L
SIZES <- if (length(args) >= 3) as.numeric(strsplit(args[3], ",")[[1]]) else c(1e7, 5e7)
now <- microbenchmark::get_nanotime

row <- function(name, n, chunk_rows, expr) {
  f <- function() eval(expr, parent.frame(2))
  t <- now()
  while (now() - t < 1e8) f()
  Sys.sleep(0.5)
  t0 <- now()
  f()
  idle <- (now() - t0) / 1e6
  d <- numeric(REPS)
  c0 <- proc.time()
  for (i in seq_len(REPS)) {
    t0 <- now()
    f()
    d[i] <- (now() - t0) / 1e6
  }
  c1 <- proc.time()
  cpu <- ((c1[["user.self"]] - c0[["user.self"]]) + (c1[["sys.self"]] - c0[["sys.self"]])) * 1e3 / REPS
  cat(sprintf("%s,r,%s,%d,%d,%.3f,%.3f,%.3f,%.3f\n", label, name, as.integer(n),
              as.integer(chunk_rows), idle, min(d), stats::median(d), cpu))
}

make_chunks <- function(kind, n, chunk_rows) {
  starts <- seq(0, n - 1, by = chunk_rows)
  lapply(starts, function(s) {
    m <- min(chunk_rows, n - s)
    if (kind == "int64") {
      Array$create(sample.int(1e6, m, replace = TRUE), type = int64())
    } else {
      x <- runif(m)
      x[((s + seq_len(m) - 1) %% 10) == 0] <- NA
      Array$create(x)
    }
  })
}

cat("label,binding,row,rows,chunk_rows,first_after_idle_ms,best_ms,median_ms,cpu_ms_per_call\n")
set.seed(3)
for (n in SIZES) {
  for (kind in c("int64", "float64_nulls")) {
    for (chunk_rows in c(65536, 1e6)) {
      chunks <- make_chunks(kind, n, chunk_rows)
      ca <- do.call(chunked_array, chunks)
      row(paste0("chunked_import_", kind), n, chunk_rows, quote(invisible(am_array_chunks(ca))))
      row(paste0("concatenate_then_import_", kind), n, chunk_rows,
          quote(invisible(am_array(do.call(concat_arrays, chunks)))))
      rm(chunks, ca)
      invisible(gc())
    }
  }
}
