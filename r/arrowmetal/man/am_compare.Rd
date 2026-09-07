\name{am_compare}
\alias{am_compare}
\alias{am_filter}
\alias{am_take}
\alias{am_slice}
\title{Comparison and selection}
\usage{
am_compare(x, op, y)

am_filter(x, mask)

am_take(x, indices)

am_slice(x, offset, length)
}
\arguments{
\item{x}{An \code{am_array}, or anything \code{\link{am_array}} accepts.}

\item{op}{One of \code{"=="}, \code{"!="}, \code{"<"}, \code{"<="}, \code{">"}, \code{">="}.}

\item{y}{A length-1 R value, compared as a scalar, or a column of the same length.}

\item{mask}{A boolean column of the same length as \code{x}. A null in the mask drops the row,
which is Arrow's default \code{null_selection_behavior = "drop"}.}

\item{indices}{Zero-based indices: an \code{am_array} of integers, an \code{arrow} Array, or an R
integer or numeric vector.}

\item{offset}{Zero-based first row.}

\item{length}{Number of rows.}
}
\value{
\code{am_compare()} returns a boolean \code{am_array}; the others return an \code{am_array} of
\code{x}'s type.
}
\description{
Element-wise comparison against a scalar or another column, and the three ways to pick rows.
Null in, null out. Arrow indices are zero based, so \code{am_take(x, am_argsort(x))} is the sorted
column and \code{am_slice(x, 0, 5)} is the first five rows.
}
\examples{
if (am_available()) {
  a <- am_array(c(1, 5, NA, 9))
  as.vector(as_arrow_array(am_filter(a, am_compare(a, ">", 3))))
  as.vector(as_arrow_array(am_take(a, c(3L, 0L))))
  as.vector(as_arrow_array(am_slice(a, 1, 2)))
}
}
