# One benchmark process for the tables in README.md and docs/R.md. Not run by the test suite.
#
#   Rscript timing.R <seed>
#
# Prints CSV: mode,method,min_ms,median_ms
#
# Two modes, because they do not give the same answer and the difference is large enough to change
# a headline:
#
#   isolated     -- each expression gets its OWN microbenchmark() call, preceded by one warm-up
#                   call of that expression. This is the most favourable measurement.
#   interleaved  -- every expression in ONE microbenchmark() call, so the runs are shuffled
#                   together and each expression sees the cache and allocator state the others
#                   leave behind. This is the conservative measurement, and the honest one for a
#                   table that compares rows against each other.
#
# Run it from several fresh processes (see run.R): the per-process spread is real, in particular
# for the resident sum.

suppressMessages(library(arrowmetal))
suppressMessages(library(arrow))
suppressMessages(library(microbenchmark))

TIMES <- 20L
N <- 1e7

seed <- as.integer(commandArgs(TRUE)[1])
if (is.na(seed)) seed <- 1L
set.seed(seed)

x <- runif(N)
a <- Array$create(x)
h <- am_array(a)
thr <- Scalar$create(0.5)

exprs <- list(
  `sum:ArrowMetal import`      = quote(am_sum(a)),
  `sum:ArrowMetal resident`    = quote(am_sum(h)),
  `sum:arrow`                  = quote(call_function("sum", a)),
  `sum:base R`                 = quote(sum(x)),
  `filter:ArrowMetal import`   = quote(am_filter(a, am_compare(a, ">", 0.5))),
  `filter:ArrowMetal resident` = quote(am_filter(h, am_compare(h, ">", 0.5))),
  `filter:arrow`               = quote(call_function("filter", a, call_function("greater", a, thr))),
  `filter:base R`              = quote(x[x > 0.5]),
  `import:am_array`            = quote(am_array(a))
)

emit <- function(mode, s) {
  for (i in seq_len(nrow(s)))
    cat(sprintf("%s,%s,%.3f,%.3f\n", mode, s$expr[i], s$min[i], s$median[i]))
}

# isolated: one warm-up then one microbenchmark() call per expression
for (nm in names(exprs)) {
  e <- exprs[[nm]]
  invisible(eval(e))
  s <- summary(microbenchmark(list = setNames(list(e), nm), times = TIMES, unit = "ms"),
               unit = "ms")
  emit("isolated", s)
}

# interleaved: one warm-up of each, then a single microbenchmark() call over all of them
for (e in exprs) invisible(eval(e))
emit("interleaved",
     summary(microbenchmark(list = exprs, times = TIMES, unit = "ms"), unit = "ms"))
