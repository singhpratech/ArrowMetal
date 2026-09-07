\name{as_arrow_array}
\alias{as_arrow_array}
\alias{as_arrow_array.am_array}
\alias{as_arrow_array.default}
\title{Move a column back to the arrow package}
\usage{
as_arrow_array(x, ...)
}
\arguments{
\item{x}{An \code{am_array}, or anything \code{arrow::Array$create()} accepts.}

\item{...}{Unused.}
}
\value{
An \code{arrow} \code{Array}.
}
\description{
Exports an ArrowMetal handle through the Arrow C Data Interface. This direction is always
copy-free: the resulting \code{Array} points straight at the Metal buffers and keeps them alive
through its release callback, so the handle may go out of scope first.
}
\details{
\code{as.vector()} on the result converts to a plain R vector. Beware that an int64 column whose
values do not fit an R integer comes back as a \code{bit64::integer64}, and calling
\code{as.vector()} on \emph{that} strips the class and reinterprets its bit pattern as a double;
use \code{as.numeric()} on an \code{integer64} instead.
}
\examples{
if (am_available()) {
  a <- am_array(c(3, 1, 2))
  as.vector(as_arrow_array(am_sort(a)))
}
}
\seealso{\code{\link{am_array}}}
