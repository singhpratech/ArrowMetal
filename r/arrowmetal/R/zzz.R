.am <- new.env(parent = emptyenv())

AM_RELATIVE_CANDIDATE <- file.path("..", "..", ".build", "release", "libArrowMetalC.dylib")

am_lib_candidates <- function() {
  out <- character()
  env <- Sys.getenv("ARROWMETAL_LIB", "")
  if (nzchar(env)) out <- c(out, env)
  recorded <- .Call(C_am_default_lib_path)
  if (!is.null(recorded)) out <- c(out, recorded)
  # Last resort: ../../.build/release/libArrowMetalC.dylib resolved against the WORKING DIRECTORY,
  # which finds the dylib when R was started from r/arrowmetal/ in a checkout. `configure` records
  # the path relative to the package *source* at install time, which is the candidate above.
  out <- c(out, AM_RELATIVE_CANDIDATE)
  unique(out)
}

# `tried` is what each candidate that exists reported. The message always names all three places
# the loader looks, including the ones that contributed no candidate at all, so a user who set
# nothing still learns what to set.
am_load_message <- function(tried) {
  env <- Sys.getenv("ARROWMETAL_LIB", "")
  recorded <- .Call(C_am_default_lib_path)
  paste0(
    "ArrowMetal: could not open libArrowMetalC.dylib.\n",
    "Looked in three places, in order:\n",
    "  1. $ARROWMETAL_LIB -- ",
    if (nzchar(env)) paste0("set to ", env) else "not set; set it to the dylib's absolute path",
    "\n  2. the path ./configure recorded at install time -- ",
    if (!is.null(recorded)) recorded else
      paste0("nothing was recorded (there was no ", AM_RELATIVE_CANDIDATE,
             " next to the package source when it was installed)"),
    "\n  3. ", AM_RELATIVE_CANDIDATE, " relative to the working directory (",
    getwd(), ")\n",
    "Tried:\n", paste0("  - ", tried, collapse = "\n")
  )
}

.onLoad <- function(libname, pkgname) {
  tried <- character()
  for (cand in am_lib_candidates()) {
    full <- suppressWarnings(normalizePath(cand, mustWork = FALSE))
    if (!file.exists(full)) {
      tried <- c(tried, paste0(full, " (not found)"))
      next
    }
    err <- .Call(C_am_load, full)
    if (is.null(err)) {
      .am$error <- NULL
      return(invisible(NULL))
    }
    tried <- c(tried, paste0(full, " (", err, ")"))
  }
  .am$error <- am_load_message(tried)
  invisible(NULL)
}

.onAttach <- function(libname, pkgname) {
  if (!is.null(.am$error)) packageStartupMessage(.am$error)
}

am_require <- function() {
  if (!is.null(.am$error)) stop(.am$error, call. = FALSE)
  invisible(TRUE)
}

#' Is the ArrowMetal library available?
#'
#' @return `TRUE` when the dynamic library was opened at package load.
#' @export
am_available <- function() is.null(.am$error)

#' The message explaining why the library could not be opened
#'
#' @return A single string, or `NULL` when the library loaded.
#' @export
am_load_error <- function() .am$error

#' Path of the dynamic library that was opened
#'
#' @return The absolute path, or `NULL` when nothing was opened.
#' @export
am_lib_path <- function() .Call(C_am_lib_path)

#' ArrowMetal version string
#' @return A single string.
#' @export
am_version <- function() {
  am_require()
  .Call(C_am_version)
}

#' Name of the Metal device the kernels run on
#' @return A single string.
#' @export
am_device_name <- function() {
  am_require()
  .Call(C_am_device_name)
}
