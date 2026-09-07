.am <- new.env(parent = emptyenv())

am_lib_candidates <- function() {
  out <- character()
  env <- Sys.getenv("ARROWMETAL_LIB", "")
  if (nzchar(env)) out <- c(out, env)
  recorded <- .Call(C_am_default_lib_path)
  if (!is.null(recorded)) out <- c(out, recorded)
  # ../../.build/release/libArrowMetalC.dylib relative to the package source directory, which is
  # where it sits in a checkout of the repository.
  out <- c(out, file.path("..", "..", ".build", "release", "libArrowMetalC.dylib"))
  unique(out)
}

am_load_message <- function(tried) {
  paste0(
    "ArrowMetal: could not open libArrowMetalC.dylib.\n",
    "Set ARROWMETAL_LIB to its absolute path, or build it in the repository so that\n",
    "../../.build/release/libArrowMetalC.dylib exists relative to r/arrowmetal/.\n",
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
