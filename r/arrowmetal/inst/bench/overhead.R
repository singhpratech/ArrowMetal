# The binding's call overhead on the existing import and sort calls, one row per call and size, so
# two builds of the package can be compared row by row in one session. Not run by the test suite.
#
#   Rscript overhead.R <label> [reps] [only]
#
# Every row: calls of the same shape run untimed for at least 100 ms, then the process sleeps
# 500 ms and times one call on its own (first_after_idle_ms), then `reps` timed calls give best and
# median wall time (microbenchmark::get_nanotime) and process CPU time (proc.time, user + system)
# per call. Prints CSV. Uses only calls that exist since 0.3.0.

suppressMessages(library(arrowmetal))
suppressMessages(library(arrow))

args <- commandArgs(TRUE)
label <- if (length(args) >= 1) args[1] else ""
REPS <- if (length(args) >= 2) as.integer(args[2]) else 30L
ONLY <- if (length(args) >= 3) args[3] else ""   # run only the rows whose name contains this
now <- microbenchmark::get_nanotime

row <- function(name, n, expr) {
  if (nzchar(ONLY) && !grepl(ONLY, name, fixed = TRUE)) return(invisible())
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
  cat(sprintf("%s,r,%s,%d,%.4f,%.4f,%.4f,%.4f\n", label, name, as.integer(n), idle, min(d),
              stats::median(d), cpu))
}

cat("label,binding,row,rows,first_after_idle_ms,best_ms,median_ms,cpu_ms_per_call\n")
set.seed(1)
for (n in c(1e3, 1e6, 1e7)) {
  a <- Array$create(sample.int(1e6, n, replace = TRUE) - 5e5, type = int64())
  row("import_int64", n, quote(invisible(am_array(a))))
  rm(a)
  invisible(gc())
}
{
  n <- 1e7
  x <- runif(n)
  x[seq(1, n, 10)] <- NA
  a <- Array$create(x)
  row("import_float64_nulls", n, quote(invisible(am_array(a))))
  ca <- chunked_array(a)  # one chunk: the path a Table column of one batch takes
  row("import_chunked_one_chunk", n, quote(invisible(am_array(ca))))
  rm(a, ca, x)
  invisible(gc())
}
for (n in c(1e3, 1e6, 1e7)) {
  x <- runif(n)
  x[seq(1, n, 10)] <- NA
  f <- am_array(x)
  i <- am_array(Array$create(sample.int(1e6, n, replace = TRUE), type = int64()))
  row("argsort_float64_nulls", n, quote(invisible(am_argsort(f))))
  row("argsort_int64_desc", n, quote(invisible(am_argsort(i, TRUE))))
  row("sort_float64_nulls", n, quote(invisible(am_sort(f))))
  rm(f, i, x)
  invisible(gc())
}
