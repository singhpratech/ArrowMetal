# arrow exports an `as_arrow_array(x, ..., type = NULL)` generic with eight methods of its own.
# arrowmetal must add a method to THAT generic; a second generic of the same name would either mask
# arrow's (breaking arrow's methods) or be masked by it (never dispatching for am_array), depending
# on the order the packages are attached. These tests run the README example in both orders in a
# fresh R process, which is the only way to see the bug: the rest of the suite reaches arrow through
# `arrow::` and so never exercises dispatch from the search path.

#
# The subprocesses must be able to `library(arrowmetal)`, which is NOT true just because this test
# is running: `testthat::test_local()` loads the source package with pkgload without installing it,
# so a clean environment has nothing on .libPaths() to attach. The helpers below therefore find a
# library that a fresh process really can load from -- reusing an existing installation when there
# is one (which is the case under `R CMD check`, where the package sits in <pkg>.Rcheck), and
# otherwise installing the package source into a temporary library once. They never skip: skipping
# would hide exactly the regression these two tests exist to catch.

.dispatch <- new.env(parent = emptyenv())

run_in_fresh_r <- function(code, env = character()) {
  script <- tempfile(fileext = ".R")
  on.exit(unlink(script), add = TRUE)
  writeLines(code, script)
  out <- suppressWarnings(system2(
    file.path(R.home("bin"), "Rscript"), c("--vanilla", shQuote(script)),
    stdout = TRUE, stderr = TRUE, env = env
  ))
  status <- attr(out, "status")
  list(status = if (is.null(status)) 0L else as.integer(status),
       output = paste(out, collapse = "\n"))
}

libpaths_line <- function(libs) {
  sprintf('.libPaths(c(%s))', paste(sprintf('"%s"', libs), collapse = ", "))
}

# Can a fresh process attach arrowmetal with these libraries?
subprocess_can_attach <- function(libs) {
  r <- run_in_fresh_r(c(libpaths_line(libs),
                        'suppressMessages(library(arrowmetal))',
                        'cat("ATTACH_OK\n")'))
  isTRUE(r$status == 0L) && grepl("ATTACH_OK", r$output, fixed = TRUE)
}

# The package source directory, found by walking up to the DESCRIPTION that names this package.
find_pkg_source <- function() {
  d <- normalizePath(".", mustWork = FALSE)
  for (i in 1:10) {
    desc <- file.path(d, "DESCRIPTION")
    if (file.exists(desc) && dir.exists(file.path(d, "src"))) {
      p <- tryCatch(unname(read.dcf(desc, "Package")[1, 1]), error = function(e) NA_character_)
      if (identical(p, "arrowmetal")) return(d)
    }
    parent <- dirname(d)
    if (identical(parent, d)) break
    d <- parent
  }
  NULL
}

# R CMD INSTALL into `lib`. If Makeconf names a compiler that is not on PATH (a conda-built R whose
# environment was not activated), point R_MAKEVARS_USER at a temporary Makevars using clang. That
# env var is the documented override and never touches any home directory.
install_into <- function(src, lib) {
  rbin <- file.path(R.home("bin"), "R")
  cc <- suppressWarnings(system2(rbin, c("CMD", "config", "CC"), stdout = TRUE, stderr = FALSE))
  cc1 <- if (length(cc)) strsplit(trimws(cc[1]), "[[:space:]]+")[[1]][1] else ""
  env <- character()
  if (!nzchar(cc1) || !nzchar(Sys.which(cc1))) {
    mk <- file.path(tempdir(), "arrowmetal-dispatch-Makevars")
    writeLines(c("CC = clang", "CXX = clang++"), mk)
    env <- c(env, paste0("R_MAKEVARS_USER=", mk))
  }
  out <- suppressWarnings(system2(
    rbin, c("CMD", "INSTALL", "--no-docs", "--no-multiarch", "-l", shQuote(lib), shQuote(src)),
    stdout = TRUE, stderr = TRUE, env = env))
  status <- attr(out, "status")
  list(status = if (is.null(status)) 0L else as.integer(status),
       output = paste(out, collapse = "\n"))
}

# Libraries a fresh process can load arrowmetal from. Computed once per session.
dispatch_libs <- function() {
  if (!is.null(.dispatch$libs)) return(.dispatch$libs)
  if (subprocess_can_attach(.libPaths())) {
    .dispatch$libs <- .libPaths()
    return(.dispatch$libs)
  }
  src <- find_pkg_source()
  if (is.null(src)) {
    stop("arrowmetal is not installed on .libPaths() and its source directory could not be found ",
         "from ", normalizePath(".", mustWork = FALSE),
         "; these tests need a real installation to attach in a subprocess.", call. = FALSE)
  }
  lib <- file.path(tempdir(), "arrowmetal-dispatch-lib")
  dir.create(lib, showWarnings = FALSE, recursive = TRUE)
  inst <- install_into(src, lib)
  libs <- c(lib, .libPaths())
  if (inst$status != 0L || !subprocess_can_attach(libs)) {
    stop("could not install arrowmetal from ", src, " into a temporary library, so the ",
         "attach-order tests cannot run.\nR CMD INSTALL said:\n", inst$output, call. = FALSE)
  }
  .dispatch$libs <- libs
  .dispatch$libs
}

lib_preamble <- function(order, libs) {
  env <- Sys.getenv("ARROWMETAL_LIB", "")
  c(libpaths_line(libs),
    if (nzchar(env)) sprintf('Sys.setenv(ARROWMETAL_LIB = "%s")', env),
    if (identical(order, "arrowmetal_first"))
      c('suppressMessages(library(arrowmetal))', 'suppressMessages(library(arrow))')
    else
      c('suppressMessages(library(arrow))', 'suppressMessages(library(arrowmetal))'))
}

for (order in c("arrowmetal_first", "arrow_first")) {
  local({
    ord <- order
    test_that(paste0("the README example works with ", ord), {
      skip_without_gpu()
      r <- run_in_fresh_r(c(
        lib_preamble(ord, dispatch_libs()),
        'x <- Array$create(c(1, 5, 9, 2))',
        'stopifnot(am_sum(x) == 17)',
        # The README's own line: an unqualified as_arrow_array() on an am_array.
        'v <- as.vector(as_arrow_array(am_filter(x, am_compare(x, ">", 3))))',
        'stopifnot(identical(v, c(5, 9)))',
        # arrow\'s own methods must still dispatch from the search path.
        'stopifnot(inherits(as_arrow_array(c(1, 2, 3)), "Array"))',
        'stopifnot(inherits(as_arrow_array(ChunkedArray$create(c(1, 2), c(3, 4))), "Array"))',
        'stopifnot(inherits(as_arrow_array(Scalar$create(1)), "Array"))',
        'cat("OK\\n")'
      ))
      expect_equal(r$status, 0L, info = r$output)
      expect_match(r$output, "OK")
    })
  })
}

test_that("arrowmetal adds a method to arrow's generic instead of defining its own", {
  # The generic the package re-exports is the identical object arrow exports, so neither package
  # masks the other.
  expect_identical(arrowmetal::as_arrow_array, arrow::as_arrow_array)
  expect_true(any(grepl("am_array", as.character(suppressMessages(methods(arrow::as_arrow_array))))))
  # arrow's own eight methods are still registered alongside ours.
  expect_true(all(c("as_arrow_array.Array", "as_arrow_array.ChunkedArray",
                    "as_arrow_array.data.frame", "as_arrow_array.default") %in%
                    as.character(suppressMessages(methods(arrow::as_arrow_array)))))
})

test_that("a type argument on the method casts, matching the generic's contract", {
  skip_without_gpu()
  a <- am_array(c(1, 2, 3))
  expect_equal(as_arrow_array(a, type = arrow::int32())$type$ToString(), "int32")
  expect_equal(as.vector(as_arrow_array(a, type = arrow::int32())), 1:3)
})
