# Driver for timing.R: runs it in several fresh R processes and prints the aggregated tables that
# README.md and docs/R.md quote. Not run by the test suite.
#
#   ARROWMETAL_LIB=/path/to/libArrowMetalC.dylib \
#     Rscript r/arrowmetal/inst/bench/run.R [n_processes] [n_replicates]
#
# Each replicate is an independent set of processes, so the spread between replicates is visible:
# the resident sum in particular is bimodal across processes and a single replicate can put it
# either side of arrow's number.

args <- commandArgs(TRUE)
n_proc <- if (length(args) >= 1) as.integer(args[1]) else 5L
n_rep <- if (length(args) >= 2) as.integer(args[2]) else 2L

here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) NULL)
if (is.null(here) || !file.exists(file.path(here, "timing.R"))) {
  here <- getwd()
  cand <- c(here, file.path(here, "r", "arrowmetal", "inst", "bench"),
            system.file("bench", package = "arrowmetal"))
  here <- cand[file.exists(file.path(cand, "timing.R"))][1]
}
stopifnot(!is.na(here))
script <- file.path(here, "timing.R")

rows <- list()
for (rep in seq_len(n_rep)) {
  for (p in seq_len(n_proc)) {
    out <- system2(file.path(R.home("bin"), "Rscript"),
                   c("--vanilla", shQuote(script), as.character(rep * 1000L + p)),
                   stdout = TRUE, stderr = FALSE)
    con <- textConnection(out)
    d <- utils::read.csv(con, header = FALSE,
                         col.names = c("mode", "method", "min_ms", "median_ms"),
                         stringsAsFactors = FALSE)
    close(con)
    d$replicate <- rep
    d$process <- p
    rows[[length(rows) + 1]] <- d
  }
  cat("replicate", rep, "done\n", file = stderr())
}
d <- do.call(rbind, rows)

cat("\n", n_rep, "replicate(s) x ", n_proc, " fresh processes x microbenchmark(times = 20)\n",
    sep = "")

for (m in c("isolated", "interleaved")) {
  cat("\n== ", m, " ==\n", sep = "")
  s <- d[d$mode == m, ]
  agg <- do.call(rbind, lapply(split(s, s$method), function(g) data.frame(
    method            = g$method[1],
    min               = min(g$min_ms),
    median_of_medians = median(g$median_ms),
    lo_median         = min(g$median_ms),
    hi_median         = max(g$median_ms))))
  print(agg[order(agg$method), ], row.names = FALSE, digits = 3)
}

cat("\n== per-process medians, resident sum (the bimodal row) ==\n")
r <- d[d$mode == "isolated" & d$method == "sum:ArrowMetal resident", ]
for (rp in sort(unique(r$replicate)))
  cat(" replicate", rp, ":", paste(sprintf("%.2f", sort(r$median_ms[r$replicate == rp])),
                                   collapse = " "), "ms\n")
a <- d[d$mode == "isolated" & d$method == "sum:arrow", ]
cat(" arrow, all processes:", paste(sprintf("%.2f", sort(a$median_ms)), collapse = " "), "ms\n")
