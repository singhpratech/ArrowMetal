\name{am_lexsort}
\alias{am_lexsort}
\title{Sort indices over several columns}
\usage{
am_lexsort(
  columns,
  descending = FALSE,
  null_placement = "at_end",
  float_order = "ieee"
)
}
\arguments{
\item{columns}{A list of columns: `am_array`s or anything [am_array()] accepts, all one length.}

\item{descending}{Logical, one per column or one for all.}

\item{null_placement}{`"at_end"` or `"at_start"`, one per column or one for all.}

\item{float_order}{`"ieee"` or `"total"`, one per column or one for all.}
}
\value{
A uint32 `am_array` of zero-based indices, read back as \code{am_argsort()}'s.
}
\description{
Orders the rows by each column in turn, the first column the most significant. Each of
`descending`, `null_placement` and `float_order` is one value for every column or a vector with
one per column, so each key can have its own direction, null placement and float order.
}
\examples{
if (am_available()) {
  as.vector(am_lexsort(list(c(1, 1, 2), c(3, NA, 1)), descending = c(FALSE, TRUE),
                       null_placement = c("at_end", "at_start")))
}
}
