skip_without_gpu <- function() {
  testthat::skip_if_not(am_available(), "libArrowMetalC.dylib not available (set ARROWMETAL_LIB)")
}

# Round trip back to plain R through the C Data Interface.
rvec <- function(x) as.vector(as_arrow_array(x))

i64 <- function(x) arrow::Array$create(x, type = arrow::int64())

# The threadgroup-crossing length used throughout.
BIG <- 1000001L
