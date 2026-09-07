# src/arrowmetal.h and src/arrow_abi.h are copies of the repository's include/ headers. The shim
# takes its function-pointer types from those prototypes with __typeof__, so a stale copy would
# compile against the wrong ABI. When the test runs from a checkout, check the copies are current.

repo_root <- function() {
  d <- normalizePath(".", mustWork = FALSE)
  for (i in 1:10) {
    if (file.exists(file.path(d, "include", "arrowmetal.h")) &&
        dir.exists(file.path(d, "r", "arrowmetal", "src"))) {
      return(d)
    }
    parent <- dirname(d)
    if (identical(parent, d)) break
    d <- parent
  }
  NULL
}

test_that("the vendored C headers match the repository's include/", {
  root <- repo_root()
  skip_if(is.null(root), "not running from a checkout of the repository")
  inc <- file.path(root, "include")
  src <- file.path(root, "r", "arrowmetal", "src")
  for (h in c("arrowmetal.h", "arrow_abi.h")) {
    expect_equal(
      tools::md5sum(file.path(src, h))[[1]],
      tools::md5sum(file.path(inc, h))[[1]],
      info = paste0("r/arrowmetal/src/", h, " is out of date; copy include/", h, " over it")
    )
  }
})
