"""am.read_csv against pyarrow.csv.read_csv, the oracle, over generated files.

Every case runs both readers on the same bytes with the same options. Either both succeed and the
tables are identical -- names, types, validity, and values (floats bit for bit) -- or both fail and the
messages are identical (pyarrow's serial reader, whose ragged-row message carries the row number).
pyarrow always runs with `newlines_in_values=True`, which is the grammar ArrowMetal parses.
"""
import io
import os
import random

import numpy as np
import pyarrow as pa
import pyarrow.csv as pc
import pytest

import arrowmetal as am

P = pc.ParseOptions
R = pc.ReadOptions
C = pc.ConvertOptions


@pytest.fixture
def tmp_csv(tmp_path):
    counter = [0]

    def write(data):
        counter[0] += 1
        p = tmp_path / ("t%d.csv" % counter[0])
        p.write_bytes(data.encode() if isinstance(data, str) else data)
        return str(p)
    return write


def _column_equal(a, b):
    """Exact equality; floating point compared bit for bit (NaN payloads and -0.0 included)."""
    a = a.combine_chunks() if isinstance(a, pa.ChunkedArray) else a
    b = b.combine_chunks() if isinstance(b, pa.ChunkedArray) else b
    if a.type != b.type or len(a) != len(b):
        return False
    if a.null_count != b.null_count:
        return False
    if not a.is_null().equals(b.is_null()):
        return False
    if pa.types.is_floating(a.type):
        width = np.uint64 if a.type == pa.float64() else np.uint32
        av = a.fill_null(0).to_numpy(zero_copy_only=False).view(width)
        bv = b.fill_null(0).to_numpy(zero_copy_only=False).view(width)
        return bool(np.array_equal(av, bv))
    if pa.types.is_null(a.type):
        return True
    if pa.types.is_string(a.type):
        # Byte-wise, so a check_utf8=False column holding invalid UTF-8 compares too.
        return a.view(pa.binary()).to_pylist() == b.view(pa.binary()).to_pylist()
    return a.to_pylist() == b.to_pylist()


def assert_tables_equal(got, exp):
    assert got.schema.names == exp.schema.names
    assert [str(f.type) for f in got.schema] == [str(f.type) for f in exp.schema]
    assert got.num_rows == exp.num_rows
    for i, name in enumerate(exp.schema.names):
        assert _column_equal(got.column(i), exp.column(i)), (name, got.column(i), exp.column(i))


def _opts(read_options=None, parse_options=None, convert_options=None):
    ro = read_options or R()
    po = parse_options or P()
    po.newlines_in_values = True
    return ro, po, convert_options or C()


def check(path, read_options=None, parse_options=None, convert_options=None, **am_kw):
    """Reads `path` with both readers and asserts the same table or the same error message."""
    ro, po, co = _opts(read_options, parse_options, convert_options)
    try:
        serial = R(skip_rows=ro.skip_rows, skip_rows_after_names=ro.skip_rows_after_names,
                   column_names=ro.column_names, autogenerate_column_names=ro.autogenerate_column_names,
                   use_threads=False)
        exp = pc.read_csv(path, read_options=serial, parse_options=po, convert_options=co)
        exp_err = None
    except (pa.ArrowInvalid, pa.ArrowKeyError, KeyError) as e:
        exp, exp_err = None, str(e)
    try:
        got = am.read_csv_table(path, read_options=ro, parse_options=po, convert_options=co, **am_kw)
        got_err = None
    except am.ArrowMetalError as e:
        got, got_err = None, str(e)
    if exp_err is not None or got_err is not None:
        assert got_err == exp_err, (got_err, exp_err)
        return None
    assert_tables_equal(got, exp)
    return got


# ---------------------------------------------------------------------------------------------------
# Hand-written cases: one per rule established by probing pyarrow (docs/CSV.md lists them).

CASES = [
    # integers
    ("int", "a\n1\n-2\n007\n9223372036854775807\n-9223372036854775808\n"),
    ("int_plus_is_float", "a\n1\n+3\n"),
    ("int_ws_trimmed", "a\n 1\n2 \n\t3\n"),
    ("int_hex", "a\n0x1F\n0XfF\n0xFFFFFFFFFFFFFFFF\n0x8000000000000000\n"),
    ("int_hex_too_long", "a\n0x10000000000000000\n"),
    ("int_hex_neg_is_string", "a\n-0x1F\n"),
    ("int_hex_empty", "a\n0x\n"),
    ("int_overflow_is_float", "a\n9223372036854775808\n"),
    ("int_neg_overflow", "a\n-9223372036854775809\n"),
    ("int_lone_minus", "a\n-\n"),
    ("int_quoted", 'a\n"1"\n"2"\n'),
    # booleans
    ("bool", "a\ntrue\nFalse\nTRUE\nfalse\n"),
    ("bool_01_is_int", "a\n0\n1\n"),
    ("bool_1_true", "a\n1\ntrue\n"),
    ("bool_2_true_is_string", "a\n2\ntrue\n"),
    ("bool_case", "a\ntRue\n"),
    ("bool_ws_not_trimmed", "a\n true\n"),
    ("bool_quoted", 'a\n"true"\n'),
    # floats
    ("float", "a\n1.5\n2\n-0\n-0.0\n.5\n5.\n1.e5\n-.5\n"),
    ("float_exp", "a\n1e5\n1E+05\n1e-5\n1e400\n1e-400\n"),
    ("float_plus_ws", "a\n+1.5\n 2.5 \n"),
    ("float_inf", "a\ninf\n-inf\nInf\nINF\ninfinity\n+inf\nInfinity\n-Infinity\n"),
    ("float_nan_spellings", "a\nNAN\n-NAN\n+NAN\nnan(123)\n-nan(1)\nNaN(x)\nnan()\n"),
    ("float_nan_is_null", "a\n1.0\nnan\nNaN\n-nan\n"),
    ("float_bad", "a\n1e\n"),
    ("float_hex_is_string", "a\n0x1p3\n"),
    ("float_double_sign", "a\n+-1\n"),
    ("float_infin", "a\ninfin\n"),
    ("float_long_mantissa", "a\n0.1000000000000000055511151231257827021181583404541015625\n"
                            "123456789012345678901234567890\n9007199254740993\n"),
    ("float_boundaries", "a\n2.2250738585072011e-308\n4.9406564584124654e-324\n2.4703282292062327e-324\n"
                         "2.4703282292062328e-324\n1.7976931348623157e308\n1.7976931348623158e308\n"
                         "1.7976931348623159e308\n"),
    ("float_quoted_ws", 'a\n" 1.5 "\n'),
    # dates, times, timestamps
    ("date", "a\n2020-01-01\n2020-02-29\n1900-03-01\n9999-12-31\n1970-01-01\n"),
    ("date_invalid_is_string", "a\n2021-02-29\n"),
    ("date_month13", "a\n2020-13-01\n"),
    ("date_ws", "a\n 2020-01-01\n2020-01-02 \n"),
    ("date_quoted", 'a\n"2020-01-01"\n'),
    ("time", "a\n12:34:56\n00:00\n23:59:59\n"),
    ("time_frac_is_string", "a\n12:34:56.123\n"),
    ("time_24", "a\n24:00:00\n"),
    ("time_ws", "a\n 12:34:56\n"),
    ("time_one_digit", "a\n1:34:56\n"),
    ("ts", "a\n2020-01-01 12:34:56\n2020-01-01T01:02:03\n2020-01-01 12\n2020-01-01 12:34\n"),
    ("ts_date_mix", "a\n2020-01-01\n2020-01-01 12:34:56\n"),
    ("ts_frac_ns", "a\n2020-01-01 12:34:56.5\n2020-01-01 12:34:56.123456789\n2020-01-01\n"),
    ("ts_frac_10_digits", "a\n2020-01-01 12:34:56.1234567891\n"),
    ("ts_zone", "a\n2020-01-01 12:34:56Z\n2020-01-01 12:34:56+01:00\n2020-01-01 12:34:56-05:30\n"
                "2020-01-01 12:34:56+0100\n2020-01-01 12:34:56+01\n2020-01-01 12Z\n2020-01-01 12+01\n"),
    ("ts_zone_ns", "a\n2020-01-01 00:00:00.5Z\n2020-01-01 00:00:00Z\n"),
    ("ts_zone_mix_is_string", "a\n2020-01-01 12:34:56Z\n2020-01-01 12:34:56\n"),
    ("ts_ws_not_trimmed", "a\n2020-01-01 12:34:56 \n"),
    ("ts_leading_ws", "a\n 2020-01-01 12:34:56\n"),
    ("ts_leap_second", "a\n2020-01-01 23:59:60\n"),
    ("ts_bad_forms", "a\n2020-01-01Z\n"),
    ("ts_offset_24", "a\n2020-01-01 12:00:00+24:00\n"),
    ("ts_1900", "a\n1900-01-01 00:00:00.5\n1850-06-30 23:59:59\n"),
    # nulls and strings
    ("nulls", "a,b,c\nNA,,1\nnull,x,\n"),
    ("all_null", "a\n\nNA\n"),
    ("all_null_2", "a,b\n,\n,\n"),
    ("null_spellings", "a\n#N/A\n#N/A N/A\n#NA\n-1.#IND\n-1.#QNAN\n-NaN\n-nan\n1.#IND\n1.#QNAN\nN/A\nNA\n"
                       "NULL\nNaN\nn/a\nnan\nnull\n\n1\n"),
    ("null_spellings_in_strings", "a\nNA\nx\n<NA>\nNone\n"),
    ("quoted_empty_is_null", 'a\n""\n1\n'),
    ("quoted_na_in_int", 'a\n"NA"\n1\n'),
    ("quoted_na_string", 'a\n"NA"\nx\n'),
    ("ws_na_is_string", "a\n NA\n1\n"),
    ("ws_only_is_string", "a\n \n1\n"),
    ("null_then_bool", "a\ntrue\n\n"),
    ("null_then_date", "a\n\n2020-01-01\n"),
    ("mixed_int_date", "a\n1\n2020-01-01\n"),
    ("mixed_float_ts", "a\n1.5\n2020-01-01\n"),
    ("mixed_time_date", "a\n12:34:56\n2020-01-01\n"),
    ("int_then_quoted_float", 'a\n1\n"2.5"\n'),
    ("utf8", "a\nhéllo\n日本\n\U0001F600\n"),
    ("invalid_utf8_is_binary", b"a,b\n\xff\xfe,x\n"),
    ("binary_after_null", b"a\n\n\xff\n"),
    ("nul_byte", b"a\n1\x00\n"),
    ("vertical_tab_not_trimmed", "a\n\x0b1\n"),
    # quoting and structure
    ("quoted_delims", 'a,b\n"x,y",1\n'),
    ("quoted_newlines", 'a,b\n"x\ny",1\n"p\r\nq",2\n"r\rs",3\n'),
    ("doubled_quotes", 'a\n"x""y"\n""""\n"""x"""\n'),
    ("quote_mid_field", 'a,b\nab"c,d\n'),
    ("text_after_quote", 'a,b\n"ab"cd,e\n"ab"c"d,e\n'),
    ("quote_then_space", 'a,b\n"x" ,1\n'),
    ("unterminated_quote", 'a\n"abc\n'),
    ("crlf", "a,b\r\n1,2\r\n3,4\r\n"),
    ("cr_only", "a,b\r1,2\r3,4\r"),
    ("mixed_line_ends", "a,b\r\n1,2\r\r\n3,4\n5,6\r"),
    ("empty_lines", "a,b\n1,2\n\n3,4\n\n\n"),
    ("empty_line_first", "\n\na,b\n1,2\n"),
    ("no_final_newline", "a,b\n1,2"),
    ("no_final_newline_quoted", 'a,b\n1,"2"'),
    ("bom", b"\xef\xbb\xbfa,b\n1,2\n"),
    ("bom_quoted_header", b'\xef\xbb\xbf"a"\n1\n'),
    ("quoted_header", '"a,b",c\n1,2\n'),
    ("empty_header_name", ",b\n1,2\n"),
    ("trailing_delimiter", "a,b,\n1,2,\n"),
    ("duplicate_names", "a,a\n1,2\n"),
    ("one_column", "a\n1\n2\n3\n"),
    ("one_row", "a,b,c\n1,x,2.5\n"),
    ("header_only", "a,b\n"),
    ("header_only_crlf", "a,b\r\n"),
    ("trailing_empty_quoted", 'a,b\n1,""\n'),
    ("only_quote_header", '""\n1\n'),
    # errors
    ("empty", ""),
    ("bom_only", b"\xef\xbb\xbf"),
    ("newlines_only", "\n\n"),
    ("crlf_only", "\r\n"),
    ("header_no_newline", "a,b"),
    ("ragged_short", "a,b,c\n1,2,3\n4,5\n"),
    ("ragged_long", "a,b\n1,2\n3,4,5\n"),
    ("ragged_after_empty_line", "a,b,c\n\n1,2,3\n4,5\n"),
    ("ragged_after_quoted_newline", 'a,b,c\n"1\n2",2,3\n4,5\n'),
    ("ragged_first_data_row", "a,b,c\n4,5\n"),
    ("ragged_whitespace_line", "a,b\n1,2\n \n"),
]


@pytest.mark.parametrize("name,data", CASES, ids=[c[0] for c in CASES])
def test_case(tmp_csv, name, data):
    check(tmp_csv(data))


# ---------------------------------------------------------------------------------------------------
# Options.

OPTION_CASES = [
    ("skip_rows", "junk\n\njunk2\na,b\n1,2\n", dict(read_options=R(skip_rows=3))),
    ("skip_rows_ignores_quotes", '"x\ny"\na,b\n1,2\n', dict(read_options=R(skip_rows=2))),
    ("skip_rows_quote_split", '"x\ny"\na,b\n1,2\n', dict(read_options=R(skip_rows=1))),
    ("skip_rows_crlf", "j\r\na,b\r\n1,2\r\n", dict(read_options=R(skip_rows=1))),
    ("skip_rows_bom", b"\xef\xbb\xbfj\na,b\n1,2\n", dict(read_options=R(skip_rows=1))),
    ("skip_rows_too_many", "j\nk\n", dict(read_options=R(skip_rows=5))),
    ("skip_rows_unterminated", "j\nk", dict(read_options=R(skip_rows=2))),
    ("skip_rows_all", "j\nk\n", dict(read_options=R(skip_rows=2))),
    ("skip_rows_to_last", "j\na,b", dict(read_options=R(skip_rows=1))),
    ("skip_rows_ragged", "x\na,b,c\n1,2,3\n4,5\n", dict(read_options=R(skip_rows=1))),
    ("skip_rows_after_names", "a,b\n1,2\n3,4\n5,6\n", dict(read_options=R(skip_rows_after_names=2))),
    ("skip_rows_after_names_all", "a,b\n1,2\n", dict(read_options=R(skip_rows_after_names=5))),
    ("autogenerate", "1,2\n3,4\n", dict(read_options=R(autogenerate_column_names=True))),
    ("autogenerate_no_newline", "1,2", dict(read_options=R(autogenerate_column_names=True))),
    ("column_names", "1,2\n3,4\n", dict(read_options=R(column_names=["x", "y"]))),
    ("column_names_no_newline", "1,2", dict(read_options=R(column_names=["x", "y"]))),
    ("column_names_wrong_count", "1,2\n", dict(read_options=R(column_names=["x"]))),
    ("include", "a,b,c\n1,x,2.5\n", dict(convert_options=C(include_columns=["c", "a"]))),
    ("include_dup", "a,b\n1,2\n", dict(convert_options=C(include_columns=["a", "a"]))),
    ("include_dup_header", "a,a\n1,2\n", dict(convert_options=C(include_columns=["a"]))),
    ("include_missing", "a,b\n1,2\n", dict(convert_options=C(include_columns=["z"]))),
    ("include_missing_ok", "a,b\n1,2\n", dict(convert_options=C(include_columns=["a", "z"],
                                                                 include_missing_columns=True))),
    ("include_missing_typed", "a,b\n1,2\n", dict(convert_options=C(
        include_columns=["a", "z"], include_missing_columns=True, column_types={"z": pa.int32()}))),
    ("types", "a,b,c,d,e\n1,2,3,4,5\n", dict(convert_options=C(column_types={
        "a": pa.float64(), "b": pa.string(), "c": pa.int8(), "d": pa.uint16(), "e": pa.float32()}))),
    ("types_int8_hex", "a\n0xFF\n0x7f\n-128\n", dict(convert_options=C(column_types={"a": pa.int8()}))),
    ("types_int8_hex3", "a\n0x1FF\n", dict(convert_options=C(column_types={"a": pa.int8()}))),
    ("types_int8_over", "a\n1\n300\n", dict(convert_options=C(column_types={"a": pa.int8()}))),
    ("types_uint8_neg0", "a\n-0\n", dict(convert_options=C(column_types={"a": pa.uint8()}))),
    ("types_uint8_plus", "a\n+1\n", dict(convert_options=C(column_types={"a": pa.uint8()}))),
    ("types_uint64_max", "a\n18446744073709551615\n0xFFFFFFFFFFFFFFFF\n",
     dict(convert_options=C(column_types={"a": pa.uint64()}))),
    ("types_int_bad", "a,b\n1,x\n", dict(convert_options=C(column_types={"b": pa.int64()}))),
    ("types_int32_float", "a,b\n1,2\n3,4.5\n", dict(convert_options=C(column_types={"b": pa.int32()}))),
    ("types_float32", "a\n0.1\n3.4028235677973366e38\n1e-46\n", dict(convert_options=C(column_types={"a": pa.float32()}))),
    ("types_bool_bad", "a\nyes\n", dict(convert_options=C(column_types={"a": pa.bool_()}))),
    ("types_date_bad", "a\nx\n", dict(convert_options=C(column_types={"a": pa.date32()}))),
    ("types_null_bad", "a\n1\n", dict(convert_options=C(column_types={"a": pa.null()}))),
    ("types_null_ok", "a\n\nNA\n", dict(convert_options=C(column_types={"a": pa.null()}))),
    ("types_string_keeps_na", "a\n1\nNA\n", dict(convert_options=C(column_types={"a": pa.string()}))),
    ("types_binary", b"a\n\xff\nNA\n", dict(convert_options=C(column_types={"a": pa.binary()}))),
    ("types_string_invalid_utf8", b"a\nx\n\xff\n", dict(convert_options=C(column_types={"a": pa.string()}))),
    ("types_string_invalid_utf8_unchecked", b"a\n\xff\n", dict(convert_options=C(column_types={"a": pa.string()},
                                                                                     check_utf8=False))),
    ("types_ts_ms", "a\n2020-01-01 12:34:56.123\n2020-01-01\n",
     dict(convert_options=C(column_types={"a": pa.timestamp("ms")}))),
    ("types_ts_us", "a\n2020-01-01 12:34:56.123456\n", dict(convert_options=C(column_types={"a": pa.timestamp("us")}))),
    ("types_ts_ms_too_precise", "a\n2020-01-01 12:34:56.1234\n",
     dict(convert_options=C(column_types={"a": pa.timestamp("ms")}))),
    ("types_ts_expect_zone", "a\n2020-01-01 12:34:56\n",
     dict(convert_options=C(column_types={"a": pa.timestamp("s", tz="UTC")}))),
    ("types_ts_expect_no_zone", "a\n2020-01-01 12:34:56Z\n",
     dict(convert_options=C(column_types={"a": pa.timestamp("s")}))),
    ("types_ts_tz", "a\n2020-01-01 12:34:56+02:00\n",
     dict(convert_options=C(column_types={"a": pa.timestamp("s", tz="UTC")}))),
    ("types_time_ms", "a\n12:34:56.5\n12:34:56.\n12:34\n", dict(convert_options=C(column_types={"a": pa.time32("ms")}))),
    ("types_time_ns", "a\n12:34:56.123456789\n", dict(convert_options=C(column_types={"a": pa.time64("ns")}))),
    ("types_missing_column_ignored", "a\n1\n", dict(convert_options=C(column_types={"zz": pa.float32()}))),
    ("types_quoted_empty_not_null", 'a\n""\n', dict(convert_options=C(column_types={"a": pa.int64()},
                                                                        quoted_strings_can_be_null=False))),
    ("quoted_strings_can_be_null_off", 'a\n""\n', dict(convert_options=C(quoted_strings_can_be_null=False))),
    ("strings_can_be_null", "a\nx\nNA\n\n\"\"\n", dict(convert_options=C(strings_can_be_null=True))),
    ("strings_can_be_null_binary", b"a\nNA\n\xff\n", dict(convert_options=C(strings_can_be_null=True))),
    ("custom_null", "a\n1\n-\n", dict(convert_options=C(null_values=["-"]))),
    ("no_null_values", "a\n1\n\n", dict(convert_options=C(null_values=[]))),
    ("custom_bool", "a\nyes\nno\n", dict(convert_options=C(true_values=["yes"], false_values=["no"]))),
    ("check_utf8_off", b"a\n\xff\n", dict(convert_options=C(check_utf8=False))),
    ("decimal_point", "a;b\n1,5;2\n", dict(parse_options=P(delimiter=";"), convert_options=C(decimal_point=","))),
    ("delimiter_tab", "a\tb\n1\t2\n", dict(parse_options=P(delimiter="\t"))),
    ("delimiter_pipe_quoted", 'a|b\n"x|y"|2\n', dict(parse_options=P(delimiter="|"))),
    ("quote_off", 'a,b\n"x",1\n', dict(parse_options=P(quote_char=False))),
    ("quote_single", "a,b\n'x,y',1\n", dict(parse_options=P(quote_char="'"))),
    ("double_quote_off", 'a\n"x""y"\n', dict(parse_options=P(double_quote=False))),
]


@pytest.mark.parametrize("name,data,kw", OPTION_CASES, ids=[c[0] for c in OPTION_CASES])
def test_options(tmp_csv, name, data, kw):
    check(tmp_csv(data), **kw)


def test_keywords_match_option_objects(tmp_csv):
    path = tmp_csv("x;y;z\nskip;me;now\na;1,5;NA\n")
    kw = dict(skip_rows_after_names=1, delimiter=";", decimal_point=",", include_columns=["y", "x"],
              column_types={"x": pa.string()}, null_values=["NA", ""])
    got = am.read_csv_table(path, **kw)
    exp = pc.read_csv(path, read_options=R(skip_rows_after_names=1), parse_options=P(delimiter=";"),
                      convert_options=C(decimal_point=",", include_columns=["y", "x"],
                                        column_types={"x": pa.string()}, null_values=["NA", ""]))
    assert_tables_equal(got, exp)


def test_read_csv_returns_metal_arrays(tmp_csv):
    cols = am.read_csv(tmp_csv("a,b\n1,x\n2,y\n3,\n"))
    assert cols.names == ["a", "b"]
    assert isinstance(cols["a"], am.MetalArray)
    assert cols["a"].to_arrow().to_pylist() == [1, 2, 3]
    assert cols["b"].to_arrow().to_pylist() == ["x", "y", ""]


def test_unsupported_options_raise(tmp_csv):
    path = tmp_csv("a\n1\n")
    with pytest.raises(NotImplementedError):
        am.read_csv(path, parse_options=P(escape_char="\\"))
    with pytest.raises(NotImplementedError):
        am.read_csv(path, convert_options=C(timestamp_parsers=["%Y"]))
    with pytest.raises(NotImplementedError):
        am.read_csv(path, convert_options=C(auto_dict_encode=True))
    with pytest.raises(NotImplementedError):
        am.read_csv(path, parse_options=P(ignore_empty_lines=False))
    with pytest.raises(NotImplementedError):
        am.read_csv(path, read_options=R(encoding="latin1"))
    with pytest.raises(NotImplementedError):
        am.read_csv(path, parse_options=P(invalid_row_handler=lambda row: "skip"))
    for t in (pa.decimal128(10, 2), pa.large_string(), pa.dictionary(pa.int32(), pa.string())):
        with pytest.raises(NotImplementedError):
            am.read_csv(path, column_types={"a": t})
    with pytest.raises(NotImplementedError):
        am.read_csv(io.BytesIO(b"a\n1\n"))


def test_compressed_extensions_are_refused(tmp_path):
    """pyarrow decompresses .gz / .bz2 / .lz4 / .zst / .br by extension; this reader refuses them."""
    import gzip
    p = tmp_path / "t.csv.gz"
    p.write_bytes(gzip.compress(b"a\n1\n"))
    assert pc.read_csv(str(p)).column("a").to_pylist() == [1]
    with pytest.raises(NotImplementedError, match="uncompressed"):
        am.read_csv(str(p))


def test_autogenerate_with_names_is_an_error(tmp_csv):
    path = tmp_csv("1,2\n")
    with pytest.raises(am.ArrowMetalError, match="autogenerate_column_names cannot be true"):
        am.read_csv(path, column_names=["x", "y"], autogenerate_column_names=True)


# ---------------------------------------------------------------------------------------------------
# Generated files.

def _rand_field(rng, kind):
    if rng.random() < 0.08:
        return rng.choice(["", "NA", "null", "N/A", '""'])
    if kind == "int":
        return str(rng.randint(-10**18, 10**18)) if rng.random() < 0.5 else str(rng.randint(-99, 99))
    if kind == "float":
        r = rng.random()
        if r < 0.3:
            return repr(rng.uniform(-1e6, 1e6))
        if r < 0.5:
            return "%.17g" % (rng.random() * 10 ** rng.randint(-320, 300))
        if r < 0.6:
            return rng.choice(["inf", "-inf", "1e308", "5e-324", "0", "-0.0"])
        return "%.*f" % (rng.randint(0, 6), rng.uniform(-1000, 1000))
    if kind == "bool":
        return rng.choice(["true", "false", "True", "FALSE", "1", "0"])
    if kind == "date":
        return "%04d-%02d-%02d" % (rng.randint(1900, 2100), rng.randint(1, 12), rng.randint(1, 28))
    if kind == "ts":
        sep = rng.choice([" ", "T"])
        return "%04d-%02d-%02d%s%02d:%02d:%02d" % (rng.randint(1900, 2100), rng.randint(1, 12),
                                                  rng.randint(1, 28), sep, rng.randint(0, 23),
                                                  rng.randint(0, 59), rng.randint(0, 59))
    if kind == "tsz":
        return _rand_field(rng, "ts") + rng.choice(["Z", "+01:00", "-0530", "+02"])
    if kind == "tsns":
        return _rand_field(rng, "ts") + "." + str(rng.randint(0, 999999999)).zfill(rng.randint(1, 9))[:9]
    if kind == "time":
        return "%02d:%02d:%02d" % (rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59))
    # strings with every structural hazard
    alphabet = "abcXYZ 012,\"\n\r;|\té日"
    s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
    return s


def _quote(rng, s, force=False):
    needs = any(ch in s for ch in ',"\n\r') or s.strip() != s or s == ""
    if needs or force or rng.random() < 0.1:
        return '"' + s.replace('"', '""') + '"'
    return s


def _gen_csv(rng, rows, kinds):
    ends = [rng.choice(["\n", "\r\n", "\r"])]
    if rng.random() < 0.3:
        ends = ["\n", "\r\n", "\r"]
    out = []
    if rng.random() < 0.2:
        out.append("﻿")
    out.append(",".join(_quote(rng, "c%d" % i) for i in range(len(kinds))) + rng.choice(ends))
    for r in range(rows):
        if rng.random() < 0.03:
            out.append(rng.choice(ends))                   # an empty line
        fields = []
        for k in kinds:
            f = _rand_field(rng, k)
            if f == '""':
                fields.append(f)
            else:
                fields.append(_quote(rng, f, force=(k == "str" and rng.random() < 0.3)))
        line = ",".join(fields)
        if len(kinds) == 1 and line == "":
            line = '""'
        out.append(line)
        if r < rows - 1 or rng.random() < 0.7:
            out.append(rng.choice(ends))
    return "".join(out)


KINDS = ["int", "float", "bool", "date", "ts", "tsz", "tsns", "time", "str"]


@pytest.mark.parametrize("seed", range(60))
def test_random_files(tmp_csv, seed):
    rng = random.Random(1000 + seed)
    kinds = [rng.choice(KINDS) for _ in range(rng.randint(1, 8))]
    data = _gen_csv(rng, rng.randint(1, 400), kinds)
    path = tmp_csv(data)
    block = rng.choice([None, 1, 3, 16, 100])
    check(path, scan_block_bytes=block)


@pytest.mark.parametrize("block", [1, 2, 7, 64, 1024])
def test_block_boundary_straddles(tmp_csv, block):
    """Quotes, doubled quotes and CRLF pairs placed across every multiple of the scan block size."""
    rng = random.Random(block)
    rows = []
    for i in range(3000):
        pad = "p" * rng.randint(0, 9)
        rows.append('%d,"%s""q\r\n%s",%s' % (i, pad, pad[::-1], rng.choice(['"x,y"', "z", '""', "1.5"])))
    data = "a,b,c\r\n" + "\r\n".join(rows) + "\r\n"
    check(tmp_csv(data), scan_block_bytes=block)


# The reader infers each column's type from its first 8192 rows and converts every row against that
# guess, checking as it goes; a value past the sample that does not fit sends the column through
# full inference. These put the value that decides the type far past the sample.
LATE = [
    ("int_then_float", "1", "1.5"),
    ("int_then_string", "1", "abc"),
    ("int_then_bool_is_string", "7", "true"),
    ("bool01_then_true", "1", "true"),
    ("null_then_int", "", "5"),
    ("null_then_string", "NA", "x"),
    ("date_then_timestamp", "2020-01-01", "2020-01-01 00:00:01"),
    ("ts_then_fraction", "2020-01-01 00:00:01", "2020-01-01 00:00:01.25"),
    ("ts_then_zone_is_string", "2020-01-01 00:00:01", "2020-01-01 00:00:01Z"),
    ("float_then_string", "1.5", "x1.5"),
    ("string_then_invalid_utf8", "x", b"\xff"),
    ("time_then_date_is_string", "12:00", "2020-01-01"),
    ("int_then_overflow", "1", "9223372036854775808"),
]


@pytest.mark.parametrize("name,early,late", LATE, ids=[c[0] for c in LATE])
def test_type_decided_after_the_sample(tmp_csv, name, early, late):
    early = early.encode() if isinstance(early, str) else early
    late = late.encode() if isinstance(late, str) else late
    rows = [early] * 20000 + [late] + [early] * 10
    check(tmp_csv(b"a,b\n" + b"\n".join(r + b",%d" % i for i, r in enumerate(rows)) + b"\n"))


def test_forced_type_error_after_the_sample(tmp_csv):
    rows = ["1"] * 20000 + ["x"]
    check(tmp_csv("a\n" + "\n".join(rows) + "\n"), convert_options=C(column_types={"a": pa.int64()}))


def test_quoted_newline_across_pyarrow_blocks(tmp_csv):
    """ArrowMetal parses quoted newlines anywhere, as pyarrow does with newlines_in_values=True.
    pyarrow's default (False) raises when a quoted newline straddles one of its blocks."""
    data = "a,b\n1," + "y" * 52 + '\n2,"p\nq"\n3,r\n'
    path = tmp_csv(data)
    ro = R(block_size=64)
    with pytest.raises(pa.ArrowInvalid, match="Expected 2 columns"):
        pc.read_csv(path, read_options=ro)
    exp = pc.read_csv(path, read_options=ro, parse_options=P(newlines_in_values=True))
    assert_tables_equal(am.read_csv_table(path), exp)
    assert exp.column("b").to_pylist() == ["y" * 52, "p\nq", "r"]


def test_default_spellings_follow_pyarrow(tmp_csv):
    """The default null, true and false spellings are pyarrow's, read from pyarrow itself."""
    co = C()
    lines = co.null_values + [" " + v for v in co.null_values if v] + ["1"]
    check(tmp_csv("a\n" + "\n".join(lines) + "\n"))
    lines = co.true_values + co.false_values
    check(tmp_csv("a\n" + "\n".join(lines) + "\n"))
    check(tmp_csv("a\n" + "\n".join(lines + ["2"]) + "\n"))


def test_file_access_map(tmp_csv):
    rng = random.Random(5)
    for seed in range(5):
        kinds = [rng.choice(KINDS) for _ in range(rng.randint(1, 6))]
        check(tmp_csv(_gen_csv(rng, rng.randint(1, 300), kinds)), file_access="map")


def test_wide_file(tmp_csv):
    rng = random.Random(7)
    kinds = [rng.choice(KINDS) for _ in range(300)]
    check(tmp_csv(_gen_csv(rng, 50, kinds)))


def test_larger_mixed_file(tmp_csv):
    rng = random.Random(11)
    kinds = ["int", "float", "bool", "date", "ts", "str", "float", "int"]
    got = check(tmp_csv(_gen_csv(rng, 20000, kinds)))
    assert all(c.num_chunks == 1 for c in got.columns)


def test_polars_and_duckdb_written_files(tmp_path):
    """Files written by two other CSV writers read back identically to pyarrow."""
    pl = pytest.importorskip("polars")
    duckdb = pytest.importorskip("duckdb")
    rng = np.random.default_rng(3)
    n = 5000
    df = pl.DataFrame({
        "i": rng.integers(-10**9, 10**9, n),
        "f": rng.normal(size=n) * 10.0 ** rng.integers(-30, 30, n),
        "s": ['a,"b"\n%d' % k if k % 7 == 0 else "v%d" % k for k in range(n)],
        "b": rng.integers(0, 2, n).astype(bool),
    })
    p1 = tmp_path / "polars.csv"
    df.write_csv(p1)
    check(str(p1))
    p2 = tmp_path / "duckdb.csv"
    con = duckdb.connect()
    con.register("df", df.to_arrow())
    con.execute("COPY (SELECT * FROM df) TO '%s' (HEADER, DELIMITER ',')" % p2)
    check(str(p2))
