"""Parquet columns whose decoded bytes pass 4 GB, against the values written.

The fixed-width decode kernels addressed the output by byte, as `row * width` in 32 bits. A
dictionary-encoded Int64 column of 2^29 + 2^22 rows and an INT96 column of 358,962,517 rows came back
with their first rows overwritten by later ones (their byte positions passed 2^32 and wrapped). The
byte positions are now 64-bit. Dictionary encoding keeps those two files under 1 GB while the decoded
columns pass 4 GB. The decimal column (PLAIN, 4-byte values widened to 16 bytes, 4 GB decoded) checks
the widening at the same size. These need about 20 GB of memory and run only with
`ARROWMETAL_BIG_TESTS=1`.
"""
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import arrowmetal as am

pytestmark = pytest.mark.skipif(os.environ.get("ARROWMETAL_BIG_TESTS") != "1",
                                reason="set ARROWMETAL_BIG_TESTS=1 for the 4 GB Parquet columns")

TAIL = 1 << 20


def _pattern(i):
    """Value of rows `i` (an index array): a hash of the row number, 1,000 distinct values."""
    i = np.asarray(i, dtype=np.uint64)
    return ((i * np.uint64(2654435761)) >> np.uint64(7)) % np.uint64(1000)


def _check(path, n, expected_at):
    cols = am.read_parquet(path, dictionary=False)
    got = cols["v"].to_arrow()
    assert len(got) == n
    # The head (where wrapped rows used to land), the tail (the rows past the wrap) and a stride.
    for idx in (np.arange(TAIL), np.arange(n - TAIL, n), np.arange(0, n, 9973)):
        want = expected_at(idx)
        have = got.take(pa.array(idx))
        assert have.equals(want), f"rows {idx[0]}..{idx[-1]} differ"


def test_int64_column_past_4_gb(tmp_path):
    n = (1 << 29) + (1 << 22)
    v = _pattern(np.arange(n)).astype(np.int64) - 500
    path = tmp_path / "i64.parquet"
    pq.write_table(pa.table({"v": v}), path, use_dictionary=True, compression="none")
    del v
    _check(path, n, lambda idx: pa.array(_pattern(idx).astype(np.int64) - 500))


def test_decimal_widened_past_4_gb(tmp_path):
    n = (1 << 28) + (1 << 20)
    t = pa.decimal128(9, 2)
    v = pa.array(_pattern(np.arange(n)).astype(np.int16) - 500).cast(t)
    path = tmp_path / "dec.parquet"
    # PLAIN: a dictionary-encoded column widens only its dictionary; the plain values widen row by row.
    pq.write_table(pa.table({"v": v}), path, use_dictionary=False, compression="none")
    assert pq.ParquetFile(path).schema.column(0).physical_type == "FIXED_LEN_BYTE_ARRAY"
    del v
    _check(path, n, lambda idx: pa.array(_pattern(idx).astype(np.int16) - 500).cast(t))


def test_int96_timestamps_past_4_gb(tmp_path):
    n = (1 << 32) // 12 + (1 << 20)
    base = np.int64(1_700_000_000_000_000_000)
    v = pa.array(base + _pattern(np.arange(n)).astype(np.int64) * 1_000_003, type=pa.timestamp("ns"))
    path = tmp_path / "i96.parquet"
    pq.write_table(pa.table({"v": v}), path, use_dictionary=True, compression="none",
                   use_deprecated_int96_timestamps=True)
    assert pq.ParquetFile(path).schema.column(0).physical_type == "INT96"
    del v
    _check(path, n, lambda idx: pa.array(base + _pattern(idx).astype(np.int64) * 1_000_003,
                                         type=pa.timestamp("ns")))
