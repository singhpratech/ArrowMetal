\name{am_array_chunks}
\alias{am_array_chunks}
\title{Move a column held in chunks onto the GPU}
\usage{
am_array_chunks(x)
}
\arguments{
\item{x}{An [arrow::ChunkedArray], or a list of [arrow::Array]s (or R vectors) of one type.}
}
\value{
An `am_array` handle.
}
\description{
Imports several Arrow arrays of one type -- the chunks of an [arrow::ChunkedArray], or a list of
[arrow::Array]s such as one column of several record batches -- as one `am_array` of their total
length (`am_import_chunks`). The column crosses the C Data Interface once, as a stream of
one-column record batches, and each chunk's buffers are copied straight into the final GPU
buffers, on the CPU cores in parallel, with no concatenated copy first. Each chunk's offset,
length and validity are honoured.
}
\details{
One chunk is [am_array()]. A type the chunked import does not take (dictionary, nested, run-end
encoded, extension) is concatenated and imported, as [am_array()] does. A list of arrays is made
a [arrow::ChunkedArray] first, which checks that every chunk has one type.
}
\examples{
if (am_available()) {
  ca <- arrow::chunked_array(c(1, NA), numeric(0), c(3, 4, 5))
  am_sum(am_array_chunks(ca))
}
}
