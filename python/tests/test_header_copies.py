"""The R and Go bindings compile against their own copies of the C ABI headers.

An R source package and a Go module can only see files inside their own directory, so
`r/arrowmetal/src/` and `go/arrowmetal/include/` carry copies of `include/arrowmetal.h` and
`include/arrow_abi.h`. Both shims type their function pointers from those prototypes, so a stale copy
compiles against the wrong ABI. Nothing syncs the copies; this test fails when one drifts. The fix is
to copy the header across, e.g. `cp include/arrowmetal.h r/arrowmetal/src/arrowmetal.h`.
"""
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HEADERS = ["arrowmetal.h", "arrow_abi.h"]
COPY_DIRS = ["r/arrowmetal/src", "go/arrowmetal/include"]


@pytest.mark.parametrize("copy_dir", COPY_DIRS)
@pytest.mark.parametrize("header", HEADERS)
def test_binding_header_copy_matches_include(copy_dir, header):
    with open(os.path.join(ROOT, "include", header), "rb") as f:
        want = f.read()
    with open(os.path.join(ROOT, copy_dir, header), "rb") as f:
        got = f.read()
    assert got == want, (
        f"{copy_dir}/{header} differs from include/{header}; "
        f"refresh it: cp include/{header} {copy_dir}/{header}"
    )
