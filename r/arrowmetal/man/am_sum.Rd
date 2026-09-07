\name{am_sum}
\alias{am_sum}
\alias{am_min}
\alias{am_max}
\alias{am_mean}
\title{Scalar reductions on the GPU}
\usage{
am_sum(x, integer64 = FALSE)

am_min(x, integer64 = FALSE)

am_max(x, integer64 = FALSE)

am_mean(x)
}
\arguments{
\item{x}{An \code{am_array}, or anything \code{\link{am_array}} accepts.}

\item{integer64}{Return an exact \code{bit64::integer64} instead of a double for an integer
result. Ignored for a floating-point column, whose result is a double either way.}
}
\value{
A length-1 double, or a length-1 \code{bit64::integer64} when \code{integer64 = TRUE} and the
result is an integer. \code{am_mean()} is always a double.
}
\description{
Reduce a whole column in one GPU pass. Nulls are skipped, as in Arrow; a column with no valid
value gives \code{NA}. \code{am_min()} and \code{am_max()} also skip \code{NaN} and give
\code{NA} when every valid value is \code{NaN}. A float32 sum accumulates in float64, like Arrow.
}
\details{
R has no native 64-bit integer, so a sum over an int64 column comes back as a double and is exact
only to 2^53; a result outside that range warns. Pass \code{integer64 = TRUE} for an exact
\code{bit64::integer64} (the \pkg{bit64} package must be installed). An unsigned 64-bit result
above 2^63 cannot be represented by \code{bit64::integer64} and warns.
}
\examples{
if (am_available()) {
  a <- am_array(c(1, 2, NA, 4))
  am_sum(a)
  am_mean(a)
}
}
