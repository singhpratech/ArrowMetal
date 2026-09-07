# src/arrowmetal.h and src/arrow_abi.h are copies of the repository's include/ headers. The shim
# takes its function-pointer types from those prototypes with __typeof__, so a stale copy would
# compile against the wrong ABI. When the test runs from a checkout, check the copies are current.

repo_include <- function() {
  d <- normalizePath(".", mustWork = FALSE)
  for (i in 1:8) {
    cand <- file.path(d, "include", "arrowmetal.h")
    if (file.exists(cand)) return(dirname(cand))
    parent <- dirname(d)
    if (identical(parent, d)) break
    d <- parent
  }
  NULL
}

pkg_src <- function() {
  for (p in c("../../src", "../../../src", "src")) if (dir.exists(p)) return(normalizePath(p))
  NULL
}

test_that("the vendored C headers match the repository's include/", {
  inc <- repo_include()
  src <- pkg_src()
  skip_if(is.null(inc), "not running from a checkout of the repository")
  skip_if(is.null(src), "package src/ not available (installed package, not the source tree)")
  for (h in c("arrowmetal.h", "arrow_abi.h")) {
    expect_equal(
      tools::md5sum(file.path(src, h))[[1]],
      tools::md5sum(file.path(inc, h))[[1]],
      info = paste0("r/arrowmetal/src/", h, " is out of date; copy include/", h, " over it")
    )
  }
})
