"""numpy crossing: every claim in docs/NUMPY.md sections 1 and 2, asserted."""
import numpy as np
import pyarrow as pa
import pytest

import arrowmetal as am


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.float64, np.float32, np.uint16])
def test_import_wraps_the_numpy_buffer_in_place(dtype):
    x = np.arange(1_000_000, dtype=dtype)
    a = pa.array(x)
    assert a.buffers()[1].address == x.ctypes.data, "pyarrow did not adopt the numpy buffer"
    col = am.array(a)
    x[5] = 7                      # written after the import
    assert col.to_arrow()[5].as_py() == 7, "the Metal column does not see a later numpy write: it was copied"


def test_export_is_a_view_over_the_metal_buffer():
    col = am.array(pa.array(np.arange(1_000_000, dtype=np.int64)))
    out = col.sort().to_arrow()
    y = out.to_numpy(zero_copy_only=True)
    assert y.ctypes.data == out.buffers()[1].address
    assert y[0] == 0 and y[-1] == 999_999


def test_nan_is_a_value_not_a_null():
    x = np.array([1.0, np.nan, 3.0])
    assert pa.array(x).null_count == 0
    col = am.array(pa.array(x))
    assert np.isnan(col.sum()) and np.isnan(col.mean())
    skipped = am.array(pa.array(x, from_pandas=True))
    assert skipped.null_count == 1 and skipped.sum() == 4.0


def test_bool_costs_one_packing_copy_and_still_agrees():
    b = np.array([True, False, True] * 10_000)
    ab = pa.array(b)
    assert ab.buffers()[1].address != b.ctypes.data   # packed to bits: a new buffer
    col = am.array(ab)
    assert col.to_arrow().to_pylist() == b.tolist()


def test_small_arrays_are_copied_but_answer_the_same():
    for n in (10, 100, 1_000, 4_096):
        x = np.arange(n, dtype=np.int64)
        assert am.array(pa.array(x)).sum() == int(x.sum())


def test_float64_results_match_numpy_bit_for_bit_on_arithmetic():
    rng = np.random.default_rng(0)
    x = rng.random(200_000) * 1000
    col = am.array(pa.array(x))
    assert np.array_equal((col + 1.5).to_arrow().to_numpy(zero_copy_only=True), x + 1.5)
    assert np.array_equal((col * 3.25).to_arrow().to_numpy(zero_copy_only=True), x * 3.25)
    assert np.array_equal(col.sqrt().to_arrow().to_numpy(zero_copy_only=True), np.sqrt(x))
