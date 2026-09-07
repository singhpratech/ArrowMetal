\docType{import}
\name{reexports}
\alias{reexports}
\alias{as_arrow_array}
\title{Objects exported from other packages}
\keyword{internal}
\description{
These objects are imported from other packages. Follow the links below to see their
documentation.

\describe{
  \item{arrow}{\code{\link[arrow]{as_arrow_array}}}
}

\code{as_arrow_array()} is \pkg{arrow}'s own generic, re-exported here so that
\code{library(arrowmetal)} alone is enough to convert a handle back. \pkg{arrowmetal} adds one
method to it, for \code{am_array}; see \code{\link{as_arrow_array.am_array}}.
}
