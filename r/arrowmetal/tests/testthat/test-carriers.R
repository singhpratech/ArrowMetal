# The ArrowArray / ArrowSchema carriers the shim allocates on the published C Data Interface:
# ownership rules, not data. A wrong answer here is a leak or a crash, not a wrong number.

test_that("a filled pair dropped without an import is released by the finalizer, not leaked or crashed", {
  skip_if_not(am_available())
  a <- arrow::Array$create(c(1.5, NA, 3))
  array_ptr <- .Call(C_alloc_arrow_array)
  schema_ptr <- .Call(C_alloc_arrow_schema)
  a$export_to_c(array_ptr, schema_ptr)
  rm(array_ptr, schema_ptr)
  gc()
  # the process is still healthy and a later round trip still works
  expect_equal(rvec(am_array(c(1, 2, 3))), c(1, 2, 3))
})

test_that("a pair that was already moved by am_import is refused, not imported twice", {
  skip_if_not(am_available())
  a <- arrow::Array$create(c(1, 2, 3))
  array_ptr <- .Call(C_alloc_arrow_array)
  schema_ptr <- .Call(C_alloc_arrow_schema)
  a$export_to_c(array_ptr, schema_ptr)
  h <- .Call(C_am_import, schema_ptr, array_ptr)
  expect_s3_class(h, "am_array")
  expect_error(.Call(C_am_import, schema_ptr, array_ptr), "not filled in by a producer")
})

test_that("an unfilled pair is refused before anything is dereferenced", {
  skip_if_not(am_available())
  array_ptr <- .Call(C_alloc_arrow_array)
  schema_ptr <- .Call(C_alloc_arrow_schema)
  expect_error(.Call(C_am_import, schema_ptr, array_ptr), "not filled in by a producer")
})

test_that("the carriers are tag-checked, so swapped or foreign pointers are an error", {
  skip_if_not(am_available())
  array_ptr <- .Call(C_alloc_arrow_array)
  schema_ptr <- .Call(C_alloc_arrow_schema)
  expect_error(.Call(C_am_import, array_ptr, schema_ptr), "expected an ArrowSchema pointer")
  expect_error(.Call(C_am_import, schema_ptr, schema_ptr), "expected an ArrowArray pointer")
  expect_error(.Call(C_am_import, 42, array_ptr), "expected an ArrowSchema pointer")
})
