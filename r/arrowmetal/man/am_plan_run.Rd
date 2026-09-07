\name{am_plan_run}
\alias{am_plan_run}
\alias{am_plan_explain}
\alias{am_plan_source}
\title{Run a whole query plan in one call}
\usage{
am_plan_source(name, columns)

am_plan_run(plan, sources, optimize = TRUE)

am_plan_explain(plan, sources, optimize = TRUE)
}
\arguments{
\item{name}{The table name the plan's \code{scan} refers to.}

\item{columns}{A named list (or data frame) of columns: \code{am_array}s, \code{arrow} Arrays or
R vectors.}

\item{plan}{A JSON string, or a nested R \code{list} which is converted with
\code{jsonlite::toJSON(auto_unbox = TRUE)} when \pkg{jsonlite} is installed.}

\item{sources}{One \code{am_plan_source}, or a list of them.}

\item{optimize}{Run the optimizer (default \code{TRUE}).}
}
\value{
\code{am_plan_source()} returns an \code{am_plan_source} handle. \code{am_plan_run()} returns an
\code{am_plan_result}: a named list of \code{am_array} columns, with the row count in
\code{attr(result, "nrow")}. \code{am_plan_explain()} returns the optimized logical plan and the
physical plan it lowers to, as one string.
}
\description{
Sends a JSON logical plan to the ArrowMetal engine, which type-checks it, optimizes it (predicate
pushdown, projection pruning, filter fusion, constant folding, CSE, join reordering), lowers it
to fused Metal kernels and runs the whole thing inside one command buffer.
}
\details{
The plan grammar (\code{scan}, \code{filter}, \code{select}, \code{with_columns},
\code{aggregate}, \code{group_by}, \code{sort}, \code{limit}, \code{unique}, \code{join},
\code{join_asof}, \code{concat}, \code{window}, \code{explode}) and the s-expression grammar its
predicates use are documented in \file{include/arrowmetal.h} and \file{docs/ENGINE.md}. A plan
that does not type-check raises an R error carrying the engine's message.

A plan source retains the columns it is given, so the \code{am_array} handles may go out of scope
afterwards.
}
\examples{
if (am_available()) {
  src <- am_plan_source("sales", list(region = c("a", "b", "a"), amount = c(1, 2, 3)))
  r <- am_plan_run('{"op":"limit","count":2,"input":{"op":"scan","source":"sales"}}', src)
  as.vector(as_arrow_array(r$amount))
}
# The list form avoids escaping the quotes the s-expressions need.
if (am_available() && requireNamespace("jsonlite", quietly = TRUE)) {
  src <- am_plan_source("sales", list(region = c("a", "b", "a"), amount = c(1, 2, 3)))
  r <- am_plan_run(list(
    op = "group_by",
    keys = list(list("region", '(col "region")')),
    aggs = list(list("sum", "total", '(col "amount")')),
    input = list(op = "scan", source = "sales")), src)
  as.vector(as_arrow_array(r$total))
}
}
