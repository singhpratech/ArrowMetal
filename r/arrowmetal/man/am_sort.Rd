\name{am_sort}
\alias{am_sort}
\alias{am_argsort}
\alias{am_top_k}
\title{GPU sort}
\usage{
am_argsort(x, descending = FALSE, null_placement = "at_end", float_order = "ieee")

am_sort(x, descending = FALSE, null_placement = "at_end", float_order = "ieee")

am_top_k(x, k, largest = TRUE, null_placement = "at_end", float_order = "ieee")
}
\arguments{
\item{x}{An \code{am_array}, or anything \code{\link{am_array}} accepts.}

\item{descending}{Sort largest first.}

\item{null_placement}{\code{"at_end"} (the default) or \code{"at_start"}, as in Arrow's sort
options; it holds in both directions.}

\item{float_order}{\code{"ieee"} (the default), \code{"total"} or \code{"nan_largest"}.}

\item{k}{Number of rows: a whole number, 0 or more; past the length gives every row.}

\item{largest}{The \code{k} largest (default) or, with \code{FALSE}, the \code{k} smallest.}
}
\value{
\code{am_argsort()} returns a uint32 \code{am_array} of zero-based indices (row numbers up to
2^32 - 1; \code{as.vector()} gives an integer vector when every index fits a 32-bit R integer and a
double vector when one passes 2^31 - 1, which is how the \code{arrow} package reads uint32);
\code{am_sort()} returns a sorted \code{am_array} of \code{x}'s type; \code{am_top_k()} returns the
first \code{k} indices \code{am_argsort(x, descending = largest, ...)} gives, found by GPU selection.
}
\description{
A stable GPU LSD radix sort. Nulls go last and \code{NaN} sorts after \code{+Inf}; neither is
mirrored to the front when \code{descending = TRUE}, so a descending sort is not the exact
reverse of an ascending one. Float keys are canonicalised first: \code{-0.0} sorts as \code{0.0}
and every \code{NaN} is one value.

\code{null_placement = "at_start"} puts the nulls first in either direction.
\code{float_order = "ieee"} is Arrow C++'s order: \code{-0} ties \code{0}, every \code{NaN} is one
value, and the \code{NaN} rows sit next to the nulls in both directions. \code{"total"} is IEEE 754
totalOrder, as arrow-rs and Rust's \code{total_cmp} define it:
\code{-NaN < -Inf < ... < -0 < 0 < ... < Inf < NaN}, and a descending sort is its exact mirror.
\code{"nan_largest"} is Polars' and NumPy's order: every \code{NaN} is one value above \code{+Inf} in
both directions (last ascending, first descending) and \code{-0} ties \code{0}.
Integer, string and temporal columns ignore \code{float_order}. With the defaults each function runs
the call it always ran.
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
  as.vector(am_argsort(c(2, NA, -0, NaN, 0), null_placement = "at_start", float_order = "total"))
  as.vector(am_top_k(a, 2))
}
}
