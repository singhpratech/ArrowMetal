\name{am_buffer_alignment}
\alias{am_buffer_alignment}
\title{Buffer alignment of an Arrow array}
\usage{
am_buffer_alignment(x)
}
\arguments{
\item{x}{An \code{arrow} Array or anything \code{arrow::Array$create()} accepts.}
}
\value{
A data frame with one row per Arrow buffer: \code{buffer} (the index), \code{address},
\code{page_size}, \code{offset_in_page}, and \code{page_aligned}. A buffer Arrow does not
materialise (the validity buffer of a column with no nulls) has address 0 and \code{NA} for the
last two columns.
}
\description{
Exports \code{x} through the Arrow C Data Interface and reports where each of its buffers starts.
ArrowMetal imports a buffer without a copy when it starts on a page boundary and copies it
otherwise, so this is the measurement that decides which of the two import paths a producer gets.
}
\details{
Export in the other direction, \code{\link{as_arrow_array}}, is always copy-free regardless of
this.
}
\examples{
if (am_available()) {
  am_buffer_alignment(arrow::Array$create(runif(1000)))
}
}
