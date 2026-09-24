#!/usr/bin/env python3
"""Writes the nested-column, Arrow-schema and page-index Parquet fixtures under Tests/Fixtures/nested.

The sibling of generate_parquet.py. Every logical dataset is written by up to three writers --
pyarrow, DuckDB (`COPY ... TO ... (FORMAT parquet)`) and Polars -- wherever that writer can express
the shape, and by pyarrow in several encoding / codec / page-version variants with small pages, so
repetition and definition levels cross page boundaries many times. The tests read every file with
ArrowMetal and with `pyarrow.parquet.read_table` and require the two to agree.

    python3 Tests/Fixtures/generate_parquet_nested.py [output-dir]

Writers that are not installed are skipped with a message; the files are committed, so the tests do
not need them.
"""
import datetime
import decimal
import json
import os
import shutil
import sys
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "nested")
N = 600


def rng():
    return np.random.default_rng(11)


# ---------------------------------------------------------------------------------------------- data

def structs_table():
    r = rng()
    s, ss, sss, sb = [], [], [], []
    for i in range(N):
        s.append(None if i % 11 == 0 else {
            "a": None if i % 7 == 3 else int(r.integers(-10**6, 10**6)),
            "b": None if i % 5 == 1 else "str-%d-%s" % (i, "y" * (i % 13)),
            "c": None if i % 9 == 2 else float(r.standard_normal()),
            "d": None if i % 4 == 0 else bool(i % 3),
        })
        inner = None if i % 6 == 5 else {"x": None if i % 8 == 1 else i * 1000 - 7,
                                         "y": None if i % 10 == 4 else "in%d" % (i % 37)}
        ss.append(None if i % 13 == 0 else {"inner": inner, "z": None if i % 3 == 2 else i})
        q = None if i % 5 == 3 else {"r": None if i % 7 == 6 else (i % 30000) - 15000}
        p = None if i % 9 == 8 else {"q": q}
        sss.append(None if i % 17 == 0 else {"p": p})
        sb.append(None if i % 19 == 0 else {"blob": bytes([(i + j) % 256 for j in range(i % 23)]),
                                            "name": "" if i % 29 == 0 else "n%05d" % i})
    return pa.table({
        "k": pa.array(range(N), pa.int64()),
        "s": pa.array(s, pa.struct([("a", pa.int32()), ("b", pa.string()), ("c", pa.float64()),
                                    ("d", pa.bool_())])),
        "ss": pa.array(ss, pa.struct([("inner", pa.struct([("x", pa.int64()), ("y", pa.string())])),
                                      ("z", pa.int32())])),
        "sss": pa.array(sss, pa.struct([("p", pa.struct([("q", pa.struct([("r", pa.int16())]))]))])),
        "sb": pa.array(sb, pa.struct([("blob", pa.binary()), ("name", pa.string())])),
    })


def required_struct_table():
    """A struct whose members are declared non-nullable, so the Parquet schema says `required`."""
    typ = pa.struct([pa.field("a", pa.int32(), nullable=False), pa.field("b", pa.string(), nullable=False)])
    vals = [None if i % 7 == 0 else {"a": i, "b": "r%d" % i} for i in range(200)]
    return pa.table({"rs": pa.array(vals, typ), "k": pa.array(range(200), pa.int32())})


def maps_table():
    m, mi, ms, ml = [], [], [], []
    for i in range(N):
        if i % 13 == 0:
            m.append(None)
        elif i % 7 == 0:
            m.append([])
        else:
            m.append([("key%d" % ((i + j) % 50), None if (i + j) % 6 == 0 else (i * 10 + j))
                      for j in range(i % 4 + 1)])
        mi.append(None if i % 17 == 0 else [(i + j, None if j == 1 else "v%d" % (i - j)) for j in range(i % 3)])
        ms.append(None if i % 19 == 0 else
                  [("s%d" % j, None if (i + j) % 5 == 0 else {"a": i + j, "b": None if j % 2 else "b%d" % i})
                   for j in range(i % 3 + 1)])
        ml.append(None if i % 23 == 0 else
                  [("l%d" % j, None if (i + j) % 7 == 0 else [i + j + t for t in range((i + j) % 4)])
                   for j in range(i % 2 + 1)])
    return pa.table({
        "k": pa.array(range(N), pa.int32()),
        "m": pa.array(m, pa.map_(pa.string(), pa.int64())),
        "mi": pa.array(mi, pa.map_(pa.int32(), pa.string())),
        "ms": pa.array(ms, pa.map_(pa.string(), pa.struct([("a", pa.int32()), ("b", pa.string())]))),
        "ml": pa.array(ml, pa.map_(pa.string(), pa.list_(pa.int32()))),
    })


def lists_table():
    ll, lls, lll, ls, sl, lm = [], [], [], [], [], []
    for i in range(N):
        def inner(j):
            if (i + j) % 9 == 0:
                return None
            if (i + j) % 5 == 0:
                return []
            return [None if (i + j + t) % 11 == 0 else (i * 100 + j * 10 + t) for t in range((i + j) % 4 + 1)]
        ll.append(None if i % 13 == 0 else ([] if i % 10 == 0 else [inner(j) for j in range(i % 4 + 1)]))
        lls.append(None if i % 15 == 0 else [None if (i + j) % 6 == 0 else
                                             ["w%d" % (i + t) if (i + t) % 8 else None for t in range(j % 3)]
                                             for j in range(i % 3 + 1)])
        lll.append(None if i % 21 == 0 else [[None if (i + j + t) % 7 == 0 else [i + j + t + u for u in range(t % 3)]
                                              for t in range((i + j) % 3)] for j in range(i % 2 + 1)])
        ls.append(None if i % 11 == 0 else [None if (i + j) % 8 == 0 else
                                            {"a": None if j == 2 else i + j, "b": "e%d" % (i * j)}
                                            for j in range(i % 4)])
        sl.append(None if i % 12 == 0 else {"xs": None if i % 5 == 0 else [i + t for t in range(i % 6)],
                                            "name": None if i % 7 == 0 else "row%d" % i})
        lm.append(None if i % 16 == 0 else [None if j == 1 else [("q%d" % t, i + t) for t in range((i + j) % 3)]
                                            for j in range(i % 3)])
    return pa.table({
        "k": pa.array(range(N), pa.int64()),
        "ll": pa.array(ll, pa.list_(pa.list_(pa.int64()))),
        "lls": pa.array(lls, pa.list_(pa.list_(pa.string()))),
        "lll": pa.array(lll, pa.list_(pa.list_(pa.list_(pa.int32())))),
        "ls": pa.array(ls, pa.list_(pa.struct([("a", pa.int32()), ("b", pa.string())]))),
        "sl": pa.array(sl, pa.struct([("xs", pa.list_(pa.int64())), ("name", pa.string())])),
        "lm": pa.array(lm, pa.list_(pa.map_(pa.string(), pa.int64()))),
    })


# ------------------------------------------------------------------ ARROW:schema metadata fixtures

class RationalType(pa.ExtensionType):
    """An extension type registered only inside this script, so a reader sees it as unregistered."""

    def __init__(self):
        super().__init__(pa.struct([("num", pa.int64()), ("den", pa.int64())]), "example.rational")

    def __arrow_ext_serialize__(self):
        return b'{"normalised":true}'

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return cls()


class LabelType(pa.ExtensionType):
    def __init__(self):
        super().__init__(pa.string(), "example.label")

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return cls()


def metadata_table():
    n = 300
    base = 1_700_000_000_000_000
    ts = [None if i % 10 == 3 else base + i * 3_600_000_123 for i in range(n)]
    uuids = [None if i % 9 == 0 else uuid.UUID(int=(i * 0x9E3779B97F4A7C15) % (1 << 128)).bytes for i in range(n)]
    fields = [
        pa.field("id", pa.int64(), metadata={"comment": "row id", "unit": "none"}),
        pa.field("ts_paris", pa.timestamp("us", tz="Europe/Paris")),
        pa.field("ts_ny_ms", pa.timestamp("ms", tz="America/New_York")),
        pa.field("ts_off_ns", pa.timestamp("ns", tz="+05:30")),
        pa.field("ts_utc", pa.timestamp("us", tz="UTC")),
        pa.field("ts_naive", pa.timestamp("us")),
        pa.field("nothing", pa.null()),
        pa.field("dur_s", pa.duration("s")),
        pa.field("dur_us", pa.duration("us")),
        pa.field("price", pa.float64(), metadata={"currency": "EUR", "precision": "cents"}),
        pa.field("tag", pa.string(), metadata={"PII": "no"}),
        pa.field("u", pa.uuid()),
        pa.field("label", LabelType(), metadata={"owner": "fixtures"}),
        pa.field("rat", RationalType()),
        pa.field("fid", pa.int32(), metadata={"PARQUET:field_id": "42", "note": "has a field id"}),
        pa.field("inner_tz", pa.struct([("t", pa.timestamp("ms", tz="Asia/Tokyo")), ("d", pa.duration("ms"))])),
        pa.field("dur_list", pa.list_(pa.duration("ns"))),
        pa.field("tz_map", pa.map_(pa.string(), pa.timestamp("us", tz="Australia/Sydney"))),
    ]
    cols = [
        pa.array(range(n), pa.int64()),
        pa.array(ts, pa.timestamp("us", tz="Europe/Paris")),
        pa.array([None if v is None else v // 1000 for v in ts], pa.timestamp("ms", tz="America/New_York")),
        pa.array([None if v is None else v * 1000 + 7 for v in ts], pa.timestamp("ns", tz="+05:30")),
        pa.array(ts, pa.timestamp("us", tz="UTC")),
        pa.array(ts, pa.timestamp("us")),
        pa.array([None] * n, pa.null()),
        pa.array([None if i % 8 == 0 else i * 61 - 1000 for i in range(n)], pa.duration("s")),
        pa.array([None if i % 6 == 0 else i * 1_000_003 for i in range(n)], pa.duration("us")),
        pa.array([None if i % 5 == 0 else i * 0.25 for i in range(n)], pa.float64()),
        pa.array(["t%d" % (i % 17) for i in range(n)], pa.string()),
        pa.ExtensionArray.from_storage(pa.uuid(), pa.array(uuids, pa.binary(16))),
        pa.ExtensionArray.from_storage(LabelType(), pa.array([None if i % 4 == 0 else "L%d" % i for i in range(n)])),
        pa.ExtensionArray.from_storage(RationalType(), pa.array(
            [None if i % 7 == 0 else {"num": i, "den": i % 5 + 1} for i in range(n)],
            pa.struct([("num", pa.int64()), ("den", pa.int64())]))),
        pa.array([None if i % 3 == 0 else i for i in range(n)], pa.int32()),
        pa.array([None if i % 11 == 0 else {"t": None if i % 4 == 0 else base // 1000 + i, "d": i * 7}
                  for i in range(n)],
                 pa.struct([("t", pa.timestamp("ms", tz="Asia/Tokyo")), ("d", pa.duration("ms"))])),
        pa.array([None if i % 10 == 0 else [i * 1000 + t for t in range(i % 3)] for i in range(n)],
                 pa.list_(pa.duration("ns"))),
        pa.array([None if i % 12 == 0 else [("k%d" % t, base + i * t) for t in range(i % 2 + 1)] for i in range(n)],
                 pa.map_(pa.string(), pa.timestamp("us", tz="Australia/Sydney"))),
    ]
    schema = pa.schema(fields, metadata={"source": "generate_parquet_nested.py"})
    return pa.Table.from_arrays(cols, schema=schema)


def arrow_types_table():
    """Arrow types Parquet cannot name, which only ARROW:schema carries: the first group comes back as
    the original type, the second (view and 64-bit-offset layouts) as its 32-bit, non-view twin."""
    n = 120
    return pa.table({
        "d32": pa.array([None if i % 9 == 0 else decimal.Decimal(i * 37 - 2000).scaleb(-2) for i in range(n)],
                        pa.decimal32(7, 2)),
        "d64": pa.array([None if i % 8 == 0 else decimal.Decimal(i * 1234567 - 10**8).scaleb(-3) for i in range(n)],
                        pa.decimal64(15, 3)),
        "fsl": pa.array([[i, None if i % 5 == 0 else -i, i * 2] for i in range(n)], pa.list_(pa.int32(), 3)),
        "cat": pa.array([None if i % 6 == 0 else ["red", "green", "blue"][i % 3] for i in range(n)]).dictionary_encode(),
        "ts_s_tz": pa.array([None if i % 4 == 0 else 1_700_000_000 + i for i in range(n)], pa.timestamp("s", tz="Europe/Berlin")),
        "sv": pa.array([None if i % 5 == 0 else "view%d" % i for i in range(n)], pa.string_view()),
        "ls": pa.array([None if i % 5 == 0 else "large%d" % i for i in range(n)], pa.large_string()),
        "ll": pa.array([None if i % 4 == 0 else list(range(i % 3)) for i in range(n)], pa.large_list(pa.int64())),
        "lv": pa.array([None if i % 3 == 0 else list(range(i % 4)) for i in range(n)], pa.list_view(pa.int64())),
    })


# ----------------------------------------------------------------------------- page index fixtures

def pageindex_table():
    """Sorted and clustered columns, so a page's min/max says something about the rows in it."""
    n = 6_000
    r = rng()
    ids = np.arange(n, dtype=np.int64) * 5 + 17
    return pa.table({
        "id": pa.array(ids, pa.int64()),
        "i32": pa.array((np.arange(n) // 40).astype(np.int32), pa.int32()),
        "f64": pa.array(np.sort(r.standard_normal(n)) * 100, pa.float64()),
        "cat": pa.array(["c%04d" % (i // 250) for i in range(n)], pa.string()),
        "v": pa.array([None if i % 13 == 0 else float(i % 997) for i in range(n)], pa.float64()),
        "noise": pa.array(r.integers(0, 1000, n).astype(np.int32), pa.int32()),
        "xs": pa.array([None if i % 31 == 0 else [i + t for t in range(i % 3)] for i in range(n)],
                       pa.list_(pa.int64())),
    })


# ------------------------------------------------------------------------------ bloom filter fixtures

BLOOM_GROUPS = 4
BLOOM_ROWS = 1024


def bloom_table():
    """Row group g holds only values congruent to g mod 4, so every row group's min/max spans nearly the
    same range and statistics cannot tell them apart; a bloom filter can."""
    v = [BLOOM_GROUPS * j + g for g in range(BLOOM_GROUPS) for j in range(BLOOM_ROWS)]
    return pa.table({
        "i64": pa.array(v, pa.int64()),
        "i32": pa.array(v, pa.int32()),
        "u32": pa.array([x + 3_000_000_000 for x in v], pa.uint32()),
        "f64": pa.array([x * 0.5 for x in v], pa.float64()),
        "s": pa.array(["s%06d" % x for x in v], pa.string()),
        # Past 32 bytes, so xxHash64's four-lane loop is exercised too.
        "long": pa.array(["a value long enough to fill the stripes %06d" % x for x in v], pa.string()),
        "cat": pa.array(["c%03d" % (BLOOM_GROUPS * (j % 50) + g) for g in range(BLOOM_GROUPS)
                         for j in range(BLOOM_ROWS)], pa.string()),
    })


# ----------------------------------------------------------------------------------------- writers

def write_pyarrow(table, name, **kw):
    path = os.path.join(OUT, name + ".parquet")
    pq.write_table(table, path, **kw)
    return path


def write_duckdb(table, name, **opts):
    try:
        import duckdb
    except ImportError:
        print("duckdb not installed; skipping %s" % name)
        return None
    path = os.path.join(OUT, name + ".parquet")
    con = duckdb.connect()
    con.register("t", table)
    extra = "".join(", %s %s" % (k, v) for k, v in opts.items())
    con.execute("COPY (SELECT * FROM t) TO '%s' (FORMAT parquet%s)" % (path, extra))
    con.close()
    return path


def write_polars(table, name, **kw):
    try:
        import polars as pl
    except ImportError:
        print("polars not installed; skipping %s" % name)
        return None
    path = os.path.join(OUT, name + ".parquet")
    pl.from_arrow(table).write_parquet(path, **kw)
    return path


def pyarrow_variants(table, base):
    write_pyarrow(table, base + "__pa_plain_none", compression="none", use_dictionary=False,
                  data_page_size=512)
    write_pyarrow(table, base + "__pa_dict_snappy", compression="snappy", use_dictionary=True,
                  data_page_size=512, row_group_size=250)
    write_pyarrow(table, base + "__pa_v2_snappy", compression="snappy", use_dictionary=False,
                  data_page_version="2.0", data_page_size=512)
    write_pyarrow(table, base + "__pa_v2_lz4", compression="lz4", use_dictionary=True,
                  data_page_version="2.0", data_page_size=1024, row_group_size=200)


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    structs = structs_table()
    maps = maps_table()
    lists = lists_table()
    for base, tbl in (("structs", structs), ("maps", maps), ("lists", lists)):
        pyarrow_variants(tbl, base)
        write_duckdb(tbl, base + "__duckdb", row_group_size=256)
    # Polars has no map type, so the map dataset has no Polars file.
    write_polars(structs, "structs__polars", row_group_size=256, data_page_size=512)
    write_polars(lists, "lists__polars", row_group_size=256, data_page_size=512)
    write_pyarrow(required_struct_table(), "reqstruct__pa_plain_none", compression="none",
                  use_dictionary=False)

    # ARROW:schema: time zones, the null type, durations, field metadata and extension types.
    pa.register_extension_type(RationalType())
    pa.register_extension_type(LabelType())
    try:
        meta = metadata_table()
        write_pyarrow(meta, "arrowschema__pa_plain_none", compression="none", use_dictionary=False)
        write_pyarrow(meta, "arrowschema__pa_dict_snappy", compression="snappy", use_dictionary=True)
    finally:
        pa.unregister_extension_type("example.rational")
        pa.unregister_extension_type("example.label")
    write_pyarrow(arrow_types_table(), "arrowschema__pa_types", compression="snappy")
    # A fixed_size_list with null rows: pyarrow 25 writes it but its own reader rejects the file
    # ("Expected all lists to be of size=3"); the test pins what ArrowMetal reads instead.
    write_pyarrow(pa.table({"fsl": pa.array([None if i % 7 == 0 else [i, -i, None if i % 5 == 0 else i * 2]
                                             for i in range(60)], pa.list_(pa.int32(), 3))}),
                  "arrowschema__pa_fslnull", compression="none")
    plain = pa.table({c: metadata_table()[c] for c in ("id", "ts_paris", "ts_utc", "ts_naive", "price")})
    write_polars(plain, "arrowschema__polars")
    # The same columns with no ARROW:schema at all, and with a corrupt one.
    write_pyarrow(plain, "arrowschema__pa_nostore", compression="none", store_schema=False)
    corrupt = plain.replace_schema_metadata({"ARROW:schema": "not base64 at all!"})
    write_pyarrow(corrupt, "arrowschema__pa_corrupt", compression="none", store_schema=False)
    truncated = plain.replace_schema_metadata({"ARROW:schema": "/////w=="})
    write_pyarrow(truncated, "arrowschema__pa_truncated", compression="none", store_schema=False)

    # Column and offset indexes: small pages, several row groups.
    pidx = pageindex_table()
    write_pyarrow(pidx, "pageindex__pa_plain_none", compression="none", use_dictionary=False,
                  write_page_index=True, data_page_size=1024, row_group_size=2500)
    write_pyarrow(pidx, "pageindex__pa_dict_snappy", compression="snappy", use_dictionary=True,
                  write_page_index=True, data_page_size=1024, row_group_size=2500)
    write_pyarrow(pidx, "pageindex__pa_v2_snappy", compression="snappy", use_dictionary=False,
                  write_page_index=True, data_page_version="2.0", data_page_size=1024, row_group_size=2500)
    write_pyarrow(pidx, "pageindex__pa_noindex", compression="none", use_dictionary=False,
                  write_page_index=False, data_page_size=1024, row_group_size=2500)
    write_polars(pidx, "pageindex__polars", row_group_size=2500, data_page_size=1024,
                 statistics=True)

    # Split-block bloom filters, from pyarrow (every column) and DuckDB (its dictionary-encoded columns).
    bloom = bloom_table()
    write_pyarrow(bloom, "bloom__pa_snappy", compression="snappy", row_group_size=BLOOM_ROWS,
                  bloom_filter_options={c: {"ndv": BLOOM_ROWS, "fpp": 0.01} for c in bloom.column_names})
    write_pyarrow(bloom, "bloom__pa_nobloom", compression="snappy", row_group_size=BLOOM_ROWS)
    write_duckdb(bloom, "bloom__duckdb", row_group_size=2048)          # two row groups of two residues

    total = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
    print("wrote %d files, %.1f KB total" % (len(os.listdir(OUT)), total / 1024))


if __name__ == "__main__":
    main()
