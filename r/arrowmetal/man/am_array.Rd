\name{am_array}
\alias{am_array}
\alias{is_am_array}
\alias{am_null_count}
\alias{am_format}
\title{Move a column onto the GPU}
\usage{
am_array(x, type = NULL)

is_am_array(x)

am_null_count(x)

am_format(x)
}
\arguments{
\item{x}{An \code{arrow} Array, ChunkedArray, an R vector, or an \code{am_array}.}

\item{type}{Optional Arrow \code{DataType} used when \code{x} has to be converted to an
Arrow array first, e.g. \code{arrow::int64()}.}
}
\value{
\code{am_array()} returns an \code{am_array} handle: an external pointer that releases the GPU
buffers when it is garbage collected. \code{is_am_array()} returns \code{TRUE} or \code{FALSE},
\code{am_null_count()} an integer-valued numeric, and \code{am_format()} the Arrow C Data
Interface format string of the element type (\code{"g"} float64, \code{"l"} int64, \code{"i"}
int32, \code{"b"} boolean, \code{"u"} string).
}
\description{
Imports an Arrow array through the Arrow C Data Interface and returns an ArrowMetal handle.
Anything \pkg{arrow}'s own \code{\link[arrow]{as_arrow_array}} generic accepts is accepted here:
an \code{Array}, a \code{ChunkedArray} (its chunks are concatenated), a \code{Scalar}, a data
frame, or a plain R vector. An \code{am_array} is returned unchanged. \code{length()} on the
handle gives the number of elements.
}
\details{
The transfer is copy-free when the producer's buffers are page aligned and one copy otherwise.
Use \code{\link{am_buffer_alignment}} to see which case a given array falls into; as measured with
arrow 25.0.0, an R-side array is not page aligned, so the import copies.
}
\examples{
if (am_available()) {
  a <- am_array(c(1, 2, NA, 4))
  am_sum(a)
  am_format(a)
}
}
\seealso{\code{\link{as_arrow_array}}, \code{\link{am_buffer_alignment}}}
