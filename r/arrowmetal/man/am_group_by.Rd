\name{am_group_by}
\alias{am_group_by}
\title{Group a set of key columns on the GPU}
\usage{
am_group_by(...)
}
\arguments{
\item{...}{One or more key columns: \code{am_array}s, \code{arrow} Arrays or R vectors, all the
same length. A single \code{list} or \code{data.frame} is also accepted.}
}
\value{
An \code{am_groupby} object with these members:
\describe{
  \item{\code{$n_groups}}{the number of groups}
  \item{\code{$n_keys}}{the number of key columns}
  \item{\code{$keys(i = 1)}}{the \code{i}-th key column, one row per group, in group order}
  \item{\code{$ids()}}{the int32 group id of every input row}
  \item{\code{$sum(v)}, \code{$min(v)}, \code{$max(v)}, \code{$mean(v)}, \code{$count(v)},
        \code{$first(v)}, \code{$last(v)}, \code{$product(v)}, \code{$var(v)}, \code{$sd(v)},
        \code{$median(v)}}{one grouped aggregate, one row per group}
  \item{\code{$count_all()}}{rows per group, nulls included}
  \item{\code{$quantile(v, q)}}{the \code{q} quantile per group}
  \item{\code{$agg(v, op, p1 = 0)}}{any op of \code{am_group_agg_ex} by its number, see the
        table in \file{include/arrowmetal.h}}
}
}
\description{
Maps one or more key columns to dense group ids on the GPU and returns an object whose members
each run one grouped aggregate. Every Arrow type ArrowMetal supports works as a key, including
strings; a null key is its own group, as Arrow's hash aggregation does, and a null value is
skipped by the aggregate.
}
\details{
Group order is deterministic but is \emph{not} Arrow's first-seen order: it is ascending by key
for numeric, boolean, temporal and decimal columns (nulls last), first-seen for string and binary
columns, and lexicographic in column order when there are several. Label the rows with
\code{$keys()} rather than assuming an order.
}
\examples{
if (am_available()) {
  g <- am_group_by(c("a", "b", "a", "b"))
  as.vector(as_arrow_array(g$keys()))
  as.vector(as_arrow_array(g$sum(c(1, 10, 2, 20))))
}
}
