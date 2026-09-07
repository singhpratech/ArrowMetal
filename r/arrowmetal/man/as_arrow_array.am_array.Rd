\name{as_arrow_array.am_array}
\alias{as_arrow_array.am_array}
\title{Move a column back to the arrow package}
\usage{
\method{as_arrow_array}{am_array}(x, ..., type = NULL)
}
\arguments{
\item{x}{An \code{am_array}.}

\item{...}{Unused.}

\item{type}{Optional Arrow \code{DataType} to cast the result to.}
}
\value{
An \code{arrow} \code{Array}.
}
\description{
A method on \pkg{arrow}'s \code{\link[arrow]{as_arrow_array}} generic. It exports an ArrowMetal
handle through the Arrow C Data Interface; this direction is always copy-free, because the
resulting \code{Array} points straight at the Metal buffers and keeps them alive through its
release callback, so the handle may go out of scope first.
}
\details{
\pkg{arrowmetal} deliberately adds a method to \pkg{arrow}'s generic rather than defining one of
its own, so that dispatch does not depend on the order the two packages are attached in.

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
