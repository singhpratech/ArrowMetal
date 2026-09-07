\name{am_sort}
\alias{am_sort}
\alias{am_argsort}
\title{GPU sort}
\usage{
am_argsort(x, descending = FALSE)

am_sort(x, descending = FALSE)
}
\arguments{
\item{x}{An \code{am_array}, or anything \code{\link{am_array}} accepts.}

\item{descending}{Sort largest first.}
}
\value{
\code{am_argsort()} returns an int32 \code{am_array} of zero-based indices;
\code{am_sort()} returns a sorted \code{am_array} of \code{x}'s type.
}
\description{
A stable GPU LSD radix sort. Nulls go last and \code{NaN} sorts after \code{+Inf}; neither is
mirrored to the front when \code{descending = TRUE}, so a descending sort is not the exact
reverse of an ascending one. Float keys are canonicalised first: \code{-0.0} sorts as \code{0.0}
and every \code{NaN} is one value.
}
\details{
Because the indices are zero based, use \code{am_argsort(x) + 1L} (after
\code{as.vector(as_arrow_array(...))}) to index an R vector, and \code{order(x, na.last = TRUE)}
is the matching base R answer.
}
\examples{
if (am_available()) {
  a <- am_array(c(3, 1, NA, 2))
  as.vector(as_arrow_array(am_sort(a)))
  as.vector(as_arrow_array(am_argsort(a)))
}
}
