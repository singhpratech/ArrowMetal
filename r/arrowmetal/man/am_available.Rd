\name{am_available}
\alias{am_available}
\alias{am_load_error}
\alias{am_lib_path}
\alias{am_version}
\alias{am_device_name}
\title{Is the ArrowMetal library there, and which one}
\usage{
am_available()

am_load_error()

am_lib_path()

am_version()

am_device_name()
}
\value{
\code{am_available()} is \code{TRUE} when the dynamic library was opened at package load.
\code{am_load_error()} is the message explaining why it was not, or \code{NULL}.
\code{am_lib_path()} is the absolute path of the library that was opened, or \code{NULL}.
\code{am_version()} and \code{am_device_name()} are single strings and require the library.
}
\description{
The Swift core ships as a separate dynamic library, \file{libArrowMetalC.dylib}, which the
package opens with \code{dlopen} at load time. It looks, in order, at
\env{ARROWMETAL_LIB}, then at the absolute path \code{configure} recorded at install time from
\file{../../.build/release/libArrowMetalC.dylib} relative to the package source, then at that
relative path itself. When none of them works the package still loads and every compute function
raises an error naming all the paths it tried.
}
\examples{
am_available()
if (am_available()) {
  am_version()
  am_device_name()
}
}
