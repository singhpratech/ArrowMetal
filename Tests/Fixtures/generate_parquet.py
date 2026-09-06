#!/usr/bin/env python3
"""Writes the small Parquet fixtures the Swift and Python tests read.

Every logical dataset is written once per encoding/codec variant, so a test can compare the variants
against each other element for element as well as against pyarrow's own reader. Files stay small
(the whole directory is well under 2 MB) so they can live in the repository.

    python3 Tests/Fixtures/generate_parquet.py [output-dir]
"""
import decimal
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
N = 700
rng = np.random.default_rng(7)


def flat_table(with_nulls):
    n = N
    ids = np.arange(n, dtype=np.int64) * 3 - 1000
    i32 = (rng.integers(-2_000_000, 2_000_000, n)).astype(np.int32)
    i16 = (rng.integers(-30000, 30000, n)).astype(np.int16)
    u32 = (rng.integers(0, 4_000_000_000, n)).astype(np.uint32)
    u64 = (rng.integers(0, 2**62, n)).astype(np.uint64)
    f32 = rng.standard_normal(n).astype(np.float32)
    f64 = rng.standard_normal(n) * 1e6
    b = rng.integers(0, 2, n).astype(bool)
    # A low-cardinality string column (dictionary friendly) and a high-cardinality one.
    cats = ["alpha", "beta", "gamma", "delta", "epsilon", "", "a much longer category value here"]
    s = np.array([cats[i % len(cats)] for i in range(n)], dtype=object)
    uniq = np.array(["row-%06d-%s" % (i, "x" * (i % 37)) for i in range(n)], dtype=object)
    blob = np.array([bytes([(i + j) % 251 for j in range(i % 19)]) for i in range(n)], dtype=object)
    fx = np.array([bytes([(i >> 8) & 0xFF, i & 0xFF, 7, 9]) for i in range(n)], dtype=object)
    ts_us = (np.arange(n, dtype=np.int64) * 1_000_003 + 1_600_000_000_000_000)
    ts_ms = (np.arange(n, dtype=np.int64) * 1_000 + 1_600_000_000_000)
    ts_ns = (np.arange(n, dtype=np.int64) * 1_000_000_007 + 1_600_000_000_000_000_000)
    date = (np.arange(n, dtype=np.int32) % 20000)
    time_ms = (np.arange(n, dtype=np.int32) * 37) % 86_400_000
    time_us = (np.arange(n, dtype=np.int64) * 37_000) % 86_400_000_000
    dec9 = (np.arange(n, dtype=np.int64) % 9999999 - 4999999)
    dec18 = (np.arange(n, dtype=np.int64) * 7919 - 10**9)
    dec38 = (np.arange(n, dtype=np.int64) * 104729 - 10**12)

    cols = {
        "id": pa.array(ids, pa.int64()),
        "i32": pa.array(i32, pa.int32()),
        "i16": pa.array(i16, pa.int16()),
        "u32": pa.array(u32, pa.uint32()),
        "u64": pa.array(u64, pa.uint64()),
        "f32": pa.array(f32, pa.float32()),
        "f64": pa.array(f64, pa.float64()),
        "b": pa.array(b, pa.bool_()),
        "s": pa.array(s, pa.string()),
        "uniq": pa.array(uniq, pa.string()),
        "blob": pa.array(blob, pa.binary()),
        "fx": pa.array(fx, pa.binary(4)),
        "ts_us": pa.array(ts_us, pa.timestamp("us")),
        "ts_ms": pa.array(ts_ms, pa.timestamp("ms")),
        "ts_ns": pa.array(ts_ns, pa.timestamp("ns")),
        "date": pa.array(date, pa.date32()),
        "time_ms": pa.array(time_ms, pa.time32("ms")),
        "time_us": pa.array(time_us, pa.time64("us")),
        "dec9": pa.array([decimal.Decimal(int(v)).scaleb(-2) for v in dec9], pa.decimal128(9, 2)),
        "dec18": pa.array([decimal.Decimal(int(v)).scaleb(-4) for v in dec18], pa.decimal128(18, 4)),
        "dec38": pa.array([decimal.Decimal(int(v)).scaleb(-10) for v in dec38], pa.decimal128(38, 10)),
    }
    if with_nulls:
        mask = rng.integers(0, 10, n) < 3
        for k, v in list(cols.items()):
            cols[k] = pa.array(v.to_pylist(), v.type, mask=mask if k != "id" else (mask & (np.arange(n) % 2 == 0)))
    return pa.table(cols)


def strings_table():
    """Values from empty to 64 KB, so the byte-array paths see long runs."""
    vals = []
    for i in range(48):
        n = [0, 1, 7, 100, 331, 4096][i % 6]
        vals.append(("v%03d" % i) * (max(n, 1) // 4) if n else "")
    # Two values at Parquet's awkward size: one just under and one just over 64 KB.
    vals[7] = "z" * 65535   # exactly the size a 16-bit length would truncate
    vals[5] = None
    return pa.table({"s": pa.array(vals, pa.string()),
                     "n": pa.array(list(range(len(vals))), pa.int64())})


def lists_table():
    n = 400
    vals, strs = [], []
    for i in range(n):
        if i % 17 == 0:
            vals.append(None)
            strs.append(None)
        elif i % 7 == 0:
            vals.append([])
            strs.append([])
        else:
            k = i % 5 + 1
            vals.append([None if (i + j) % 11 == 0 else (i * 100 + j) for j in range(k)])
            strs.append(["s%d" % (i + j) for j in range(k)])
    return pa.table({"xs": pa.array(vals, pa.list_(pa.int64())),
                     "ss": pa.array(strs, pa.list_(pa.string())),
                     "k": pa.array(list(range(n)), pa.int32())})


def write(table, name, **kw):
    path = os.path.join(OUT, name + ".parquet")
    pq.write_table(table, path, **kw)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    flat = flat_table(False)
    nulls = flat_table(True)
    small = flat.select(["id", "i32", "f32", "f64", "b", "s", "ts_us"])

    for base, tbl in (("flat", flat), ("nulls", nulls)):
        write(tbl, base + "__plain_none", compression="none", use_dictionary=False, data_page_size=8192)
        write(tbl, base + "__dict_none", compression="none", use_dictionary=True, data_page_size=8192)
        write(tbl, base + "__plain_snappy", compression="snappy", use_dictionary=False, data_page_size=8192)
        write(tbl, base + "__dict_snappy", compression="snappy", use_dictionary=True, data_page_size=8192)
        write(tbl, base + "__v2_snappy", compression="snappy", use_dictionary=True,
              data_page_version="2.0", data_page_size=8192)
        write(tbl, base + "__v2_none", compression="none", use_dictionary=False,
              data_page_version="2.0", data_page_size=8192)

    for codec in ("gzip", "lz4", "zstd", "brotli"):
        try:
            write(small, "flat__plain_" + codec, compression=codec, use_dictionary=False, data_page_size=8192)
            write(small, "flat__v2_" + codec, compression=codec, use_dictionary=True,
                  data_page_version="2.0", data_page_size=8192)
        except Exception as e:                                   # a codec the local pyarrow lacks
            print("skipping %s: %s" % (codec, e))

    # BYTE_STREAM_SPLIT over the float columns.
    floats = flat.select(["f32", "f64", "id"])
    write(floats, "floats__bss_none", compression="none", use_dictionary=False,
          use_byte_stream_split=["f32", "f64"], data_page_size=8192)
    write(floats, "floats__bss_snappy", compression="snappy", use_dictionary=False,
          use_byte_stream_split=["f32", "f64"], data_page_size=8192)
    write(floats, "floats__plain_none", compression="none", use_dictionary=False, data_page_size=8192)

    # DELTA encodings.
    ints = flat.select(["id", "i32", "s", "uniq"])
    write(ints, "delta__none", compression="none", use_dictionary=False, data_page_size=8192,
          column_encoding={"id": "DELTA_BINARY_PACKED", "i32": "DELTA_BINARY_PACKED",
                           "s": "DELTA_LENGTH_BYTE_ARRAY", "uniq": "DELTA_BYTE_ARRAY"})
    write(ints, "delta__snappy", compression="snappy", use_dictionary=False, data_page_size=8192,
          column_encoding={"id": "DELTA_BINARY_PACKED", "i32": "DELTA_BINARY_PACKED",
                           "s": "DELTA_LENGTH_BYTE_ARRAY", "uniq": "DELTA_BYTE_ARRAY"})
    write(ints, "delta__plain_none", compression="none", use_dictionary=False, data_page_size=8192)

    intn = nulls.select(["id", "i32", "s", "uniq"])
    write(intn, "deltanulls__none", compression="none", use_dictionary=False, data_page_size=8192,
          column_encoding={"id": "DELTA_BINARY_PACKED", "i32": "DELTA_BINARY_PACKED",
                           "s": "DELTA_LENGTH_BYTE_ARRAY", "uniq": "DELTA_BYTE_ARRAY"})
    write(intn, "deltanulls__plain_none", compression="none", use_dictionary=False, data_page_size=8192)

    # Several row groups (each with its own dictionary page).
    write(flat, "groups__dict_snappy", compression="snappy", use_dictionary=True, row_group_size=180,
          data_page_size=4096)
    write(flat, "groups__plain_none", compression="none", use_dictionary=False, row_group_size=180,
          data_page_size=4096)

    # Long strings and empty values.
    st = strings_table()
    write(st, "strings__plain_none", compression="none", use_dictionary=False, data_page_size=8192)
    write(st, "strings__dict_snappy", compression="snappy", use_dictionary=True, data_page_size=8192)
    write(st, "strings__v2_zstd", compression="zstd", use_dictionary=False,
          data_page_version="2.0", data_page_size=8192)

    # Lists.
    ls = lists_table()
    write(ls, "lists__plain_none", compression="none", use_dictionary=False, data_page_size=8192)
    write(ls, "lists__dict_snappy", compression="snappy", use_dictionary=True, data_page_size=8192)

    # Decimals stored as INT32 / INT64 rather than FIXED_LEN_BYTE_ARRAY.
    decs = flat.select(["dec9", "dec18", "id"])
    try:
        write(decs, "decint__plain_none", compression="none", use_dictionary=False,
              store_decimal_as_integer=True, data_page_size=8192)
        write(decs, "decint__dict_snappy", compression="snappy", use_dictionary=True,
              store_decimal_as_integer=True, data_page_size=8192)
        write(decs, "decint__plain_fixed", compression="none", use_dictionary=False, data_page_size=8192)
    except TypeError as e:                                    # older pyarrow
        print("skipping integer decimals: %s" % e)

    # A struct column: its leaves are read by dotted path.
    n = 300
    st2 = pa.table({
        "addr": pa.array([None if i % 23 == 0
                          else {"city": ["London", "Paris", "Tokyo"][i % 3], "zip": i * 7}
                          for i in range(n)],
                         pa.struct([("city", pa.string()), ("zip", pa.int32())])),
        "k": pa.array(list(range(n)), pa.int64()),
    })
    write(st2, "struct__plain_none", compression="none", use_dictionary=False, data_page_size=8192)
    write(st2, "struct__dict_snappy", compression="snappy", use_dictionary=True, data_page_size=8192)

    # Degenerate shapes.
    write(flat.slice(0, 0), "empty__plain_none", compression="none", use_dictionary=False)
    write(pa.table({"x": pa.array([None] * 500, pa.int64()),
                    "y": pa.array([None] * 500, pa.string())}),
          "allnull__plain_none", compression="none", use_dictionary=False)
    write(pa.table({"x": pa.array([1], pa.int64())}), "one__plain_none", compression="none", use_dictionary=False)
    # INT96 timestamps (the legacy Spark shape).
    write(pa.table({"t": pa.array([1_600_000_000_000_000_000 + i * 10**9 for i in range(200)],
                                  pa.timestamp("ns")),
                    "i": pa.array(list(range(200)), pa.int64())}),
          "int96__plain_none", compression="none", use_dictionary=False, use_deprecated_int96_timestamps=True)

    total = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT) if f.endswith(".parquet"))
    print("wrote %d files, %.1f KB total" % (
        len([f for f in os.listdir(OUT) if f.endswith(".parquet")]), total / 1024))


if __name__ == "__main__":
    main()
