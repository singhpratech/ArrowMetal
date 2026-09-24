"""GPU newline-delimited JSON reading (am.read_json), checked against pyarrow.json.read_json.

Every input is read twice -- by ArrowMetal on the GPU and by pyarrow on the CPU -- and the two must
agree: the same schema, the same values and nulls, or the same error text. The probes below are the
inputs docs/JSON.md was written from; the generated, truncated and mutated files cover what the
probes do not. The documented differences from pyarrow each have a test of their own at the end,
which checks that the difference is exactly the one docs/JSON.md describes.
"""
import io
import json
import os
import random
import subprocess
import sys
import tempfile

import pyarrow as pa
import pyarrow.json as pj
import pytest

import arrowmetal as am


def outcome(fn):
    try:
        return ("ok", fn())
    except (am.ArrowMetalError, pa.ArrowInvalid, pa.ArrowNotImplementedError) as e:
        return ("err", str(e))


def pa_read(data, parse_options=None, read_options=None):
    return outcome(lambda: pj.read_json(io.BytesIO(data), read_options=read_options, parse_options=parse_options))


def am_read(data, parse_options=None):
    return outcome(lambda: am.read_json_table(data, parse_options=parse_options))


def same(a, b):
    if a[0] != b[0]:
        return False
    if a[0] == "err":
        return a[1] == b[1]
    ta, tb = a[1], b[1]
    if ta.schema != tb.schema or ta.num_rows != tb.num_rows:
        return False
    # NaN != NaN in Table.equals; the Python values compare them by their repr.
    return ta.equals(tb) or repr(ta.to_pylist()) == repr(tb.to_pylist())


def check(data, parse_options=None, read_options=None):
    """Asserts ArrowMetal reads `data` as pyarrow does.

    pyarrow drops the leading nulls of a list whose element type it has not seen yet
    (test_pyarrow_list_leading_null_defect); for such files the schemas must still agree, and the
    values are compared with pyarrow reading the same file under the inferred schema, which it
    reads correctly.
    """
    if isinstance(data, str):
        data = data.encode()
    want, got = pa_read(data, parse_options, read_options), am_read(data, parse_options)
    if same(want, got):
        return
    if want[0] == got[0] == "ok" and want[1].schema == got[1].schema and parse_options is None:
        oracle = pa_read(data, pj.ParseOptions(explicit_schema=want[1].schema), read_options)
        if same(oracle, got):
            return
    raise AssertionError("pyarrow: %r\narrowmetal: %r" % (
        want[1] if want[0] == "err" else want[1].to_pylist()[:5],
        got[1] if got[0] == "err" else got[1].to_pylist()[:5]))


def S(*fields, **kw):
    return pj.ParseOptions(explicit_schema=pa.schema(list(fields)), **kw)


# ---------------------------------------------------------------------------------------------------
# Probes: one input per behaviour docs/JSON.md describes.

PROBES = {
    # types and inference
    "basic": '{"a":1,"b":"x","c":true,"d":null,"e":1.5}\n',
    "int_then_double": '{"a":1}\n{"a":2.5}\n',
    "double_then_int": '{"a":2.5}\n{"a":1}\n',
    "null_only": '{"a":null}\n{"a":null}\n',
    "missing_keys": '{"a":1}\n{"b":2}\n',
    "int64_max": '{"a":9223372036854775807}\n',
    "int64_min": '{"a":-9223372036854775808}\n',
    "int_overflow_is_double": '{"a":9223372036854775808}\n',
    "int_underflow_is_double": '{"a":-9223372036854775809}\n',
    "uint64_max_is_double": '{"a":18446744073709551615}\n',
    "int_then_bigint": '{"a":1}\n{"a":123456789012345678901234567890}\n',
    "exponent_is_double": '{"a":1e3}\n{"a":1E+2}\n',
    "negative_zero": '{"a":-0}\n{"b":-0.0}\n',
    "nan_inf": '{"a":NaN}\n{"a":Inf}\n{"a":-Inf}\n{"a":Infinity}\n{"a":-Infinity}\n{"a":-NaN}\n',
    "nan_after_int": '{"a":1}\n{"a":NaN}\n',
    "tiny_double": '{"a":1e-400}\n{"a":5e-324}\n',
    "long_fraction": '{"a":0.' + "1" * 400 + '}\n',
    "big_integer_digits": '{"a":' + "9" * 400 + '}\n',
    "number_too_big_1e309": '{"a":1e309}\n',
    "number_1e308": '{"a":1e308}\n',
    "number_long_mantissa_1e309": '{"a":1.7976931348623157e309}\n',
    "number_10e308": '{"a":10e308}\n',
    "number_0.5e309": '{"a":0.5e309}\n',
    "number_0.5e310": '{"a":0.5e310}\n',
    "number_small_fraction_big_exp": '{"a":0.00001e312}\n{"b":0.00001e314}\n',
    "number_long_zero_fraction_exp": '{"a":1.' + "0" * 30 + 'e320}\n',
    "number_long_zero_fraction_exp2": '{"a":1.' + "0" * 30 + 'e330}\n',
    "number_huge_negative_exp": '{"a":1e-99999999999}\n',
    "number_huge_exp": '{"a":1e999999999999}\n',
    "string_of_digits": '{"a":"1"}\n',
    "bools": '{"b":true}\n{"b":null}\n{}\n{"b":false}\n',
    "strings_and_nulls": '{"s":null}\n{"s":"a"}\n{}\n{"s":""}\n',
    # timestamps: pyarrow infers timestamp[s] when every string of a column is ISO-8601
    "ts_space": '{"t":"2020-01-01 00:00:00"}\n',
    "ts_T": '{"t":"2020-01-01T00:00:00"}\n',
    "ts_date": '{"t":"2020-01-01"}\n',
    "ts_Z": '{"t":"2020-01-01T00:00:00Z"}\n',
    "ts_fraction_is_string": '{"t":"2020-01-01 00:00:00.123"}\n',
    "ts_minutes": '{"t":"2020-01-01T00:00"}\n',
    "ts_offset_hh_mm": '{"t":"2020-01-01T00:00:00+01:00"}\n',
    "ts_offset_hh": '{"t":"2020-01-01T00:00:00+01"}\n',
    "ts_offset_hhmm": '{"t":"2020-01-01T00:00:00+0130"}\n',
    "ts_offset_negative": '{"t":"2020-01-01T00:00:00-05:30"}\n',
    "ts_offset_on_minutes": '{"t":"2020-01-01T10:00+02:00"}\n',
    "ts_hour_only": '{"t":"2020-01-01 00"}\n{"t":"2020-01-01 10"}\n',
    "ts_hour_Z": '{"t":"2020-01-01T00Z"}\n',
    "ts_mixed_forms": '{"t":"2020-01-01"}\n{"t":"2020-01-01 00:00:00"}\n',
    "ts_then_string": '{"t":"2020-01-01"}\n{"t":"hello"}\n',
    "ts_then_null": '{"t":"2020-01-01"}\n{"t":null}\n',
    "ts_hhmm_no_colon": '{"t":"2020-01-01T0000"}\n',
    "ts_date_Z": '{"t":"2020-01-01Z"}\n',
    "ts_bad_day": '{"t":"2020-02-30"}\n',
    "ts_leap_day": '{"t":"2020-02-29"}\n{"t":"2000-02-29T12"}\n',
    "ts_not_leap": '{"t":"2019-02-29"}\n',
    "ts_1900_not_leap": '{"t":"1900-02-29"}\n',
    "ts_hour_24": '{"t":"2020-01-01 24:00:00"}\n',
    "ts_second_60": '{"t":"2020-01-01 23:59:60"}\n',
    "ts_lowercase_t": '{"t":"2020-01-01t00:00:00"}\n',
    "ts_year_9999": '{"t":"9999-12-31 23:59:59"}\n',
    "ts_old_dates": '{"t":"1066-10-14 09:00"}\n{"t":"1900-02-28"}\n',
    "ts_five_digit_year": '{"t":"12020-01-01"}\n',
    "ts_short_month": '{"t":"2020-1-01"}\n',
    "ts_trailing_space": '{"t":"2020-01-01 "}\n',
    "ts_fraction_zero": '{"t":"2020-01-01 00:00:00.000"}\n',
    "ts_bare_dot": '{"t":"2020-01-01 00:00:00."}\n',
    "ts_escaped": '{"t":"2020\\u002d01-01"}\n',
    "ts_offset_hour_25": '{"t":"2020-01-01T10:00+25:00"}\n',
    "ts_offset_minute_60": '{"t":"2020-01-01T10:00+02:60"}\n',
    "ts_offset_one_digit": '{"t":"2020-01-01T10:00+2"}\n',
    "ts_negative_year": '{"t":"-2020-01-01"}\n',
    "ts_month_13": '{"t":"2020-13-01"}\n',
    "ts_minute_60": '{"t":"2020-01-01 10:60"}\n',
    "ts_in_list": '{"l":["2020-01-01",null]}\n',
    "ts_in_struct": '{"s":{"t":"2020-01-01"}}\n{"s":{"t":null}}\n',
    "ts_and_string_in_struct": '{"s":{"t":"2020-01-01"}}\n{"s":{"t":"x"}}\n',
    # records, lines and whitespace
    "empty_lines": '\n{"a":1}\n\n\n{"a":2}\n\n',
    "whitespace_lines": '  \n{"a":1}\n \t \n{"a":2}\n',
    "no_final_newline": '{"a":1}\n{"a":2}',
    "crlf": '{"a":1}\r\n{"a":2}\r\n',
    "cr_only": '{"a":1}\r{"a":2}\r',
    "two_objects_one_line": '{"a":1}{"a":2}\n',
    "objects_separated_by_spaces": '{"a":1}   {"a":2}\n{"a":3}',
    "tab_separated_objects": '{"a":1}\t{"a":2}\n',
    "multiline_object": '{"a":\n1}\n{"a":2}\n',
    "whitespace_inside": '   {"a" :  1 ,  "b":2 }   \n{\t"c"\r:\n1\r\n}\n',
    "empty_objects": '{}\n{}\n',
    "empty_object_with_space": '{ }\n',
    "only_newlines": '\n\n\n',
    "only_spaces": '   ',
    "empty_file": '',
    "utf8_bom": b'\xef\xbb\xbf{"a":1}\n',
    "bom_mid_file": b'{"a":1}\n\xef\xbb\xbf{"a":2}\n',
    "null_row_after_first": '{"a":1}\nnull\n{"a":2}\n',
    "two_null_rows": '{"a":1}\nnullnull\n',
    "null_then_garbage": '{"a":1}\nnullx\n',
    # keys
    "key_escape": '{"a\\u0062":1}\n',
    "key_unicode": '{"\u00e9":1,"\\u00e9x":2}\n',
    "empty_key": '{"":1}\n',
    "duplicate_key": '{"a":1,"a":2}\n',
    "duplicate_key_later": '{"a":1,"b":3,"a":2}\n{"a":5,"b":1}\n',
    "duplicate_key_escaped": '{"a":1,"\\u0061":2}\n',
    "duplicate_nested_key": '{"s":{"x":1,"x":2}}\n',
    # strings
    "escapes_and_surrogates": '{"a":"\\u00e9\\ud83d\\ude00\\n\\t\\"\\\\\\/\\b\\f\\r"}\n',
    "raw_utf8": '{"a":"h\u00e9llo \U0001F600"}\n',
    "hex_upper": '{"a":"\\u00E9"}\n',
    "escaped_nul": '{"a":"\\u0000x"}\n',
    "delete_char": '{"a":"x\x7fy"}\n',
    "backslash_run_40": '{"a":"' + "\\\\" * 40 + '"}\n',
    "backslash_run_400": '{"a":"' + "\\\\" * 400 + '","b":"' + 'x\\"' * 100 + '"}\n',
    "escaped_quote_run": '{"a":"x\\\\\\"y"}\n',
    "empty_strings": '{"s":""}\n{"s":""}\n',
    # nested
    "struct": '{"s":{"x":1,"y":"a"}}\n{"s":{"y":"b","z":true}}\n',
    "struct_null_first": '{"s":null}\n{"s":{"x":1}}\n',
    "struct_empty": '{"s":{}}\n',
    "struct_null_then_empty": '{"s":null}\n{"s":{}}\n',
    "struct_missing_in_row": '{"s":{"x":1}}\n{}\n',
    "deep_struct": '{"a":{"b":{"c":{"d":1}}}}\n',
    "list": '{"l":[1,2,3]}\n{"l":[]}\n{"l":null}\n',
    "list_int_double": '{"l":[1,2.5]}\n',
    "list_null_after_value": '{"l":[1,null]}\n',
    "list_null_first_known_type": '{"l":[3]}\n{"l":[null,1]}\n{"l":[2]}\n',
    "list_only_empty": '{"l":[]}\n',
    "list_only_null_element": '{"l":[null]}\n{"l":[2]}\n',
    "list_of_lists": '{"l":[[1],[2,3]]}\n',
    "list_of_empty_lists": '{"l":[[],[]]}\n',
    "list_of_structs": '{"l":[{"x":1},{"y":2}]}\n',
    "list_of_structs_with_null": '{"l":[{"x":5},null,{"x":7}]}\n',
    "list_of_bools": '{"l":[true,false,null]}\n',
    "list_null_then_list": '{"l":null}\n{"l":[1]}\n',
    "list_of_struct_nested_list": '{"l":[{"a":[1,2]},{"a":[]},{"b":"x"}]}\n{"l":[]}\n',
    "struct_with_list_of_struct": '{"s":{"l":[{"x":1},{"x":2,"y":true}]}}\n{"s":{"l":null}}\n',
    "nested_60_lists": '{"a":' + "[" * 60 + "1" + "]" * 60 + '}\n',
    "nested_60_objects": '{"a":' + '{"b":' * 59 + "1" + "}" * 59 + '}\n',
    # type conflicts (pyarrow's message and row)
    "conflict_number_string": '{"a":1}\n{"a":"x"}\n',
    "conflict_string_number": '{"a":"x"}\n{"a":1}\n',
    "conflict_boolean_number": '{"a":true}\n{"a":1}\n',
    "conflict_object_number": '{"s":{"x":1}}\n{"s":1}\n',
    "conflict_array_number": '{"l":[1]}\n{"l":1}\n',
    "conflict_row_after_nulls": '{"a":1}\n{"a":null}\n{}\n\n{"a":2}\n{"a":3}\n{"a":"x"}\n',
    "conflict_nested": '{"s":{"x":1}}\n{"s":{"x":"a"}}\n',
    "conflict_in_list": '{"l":[1,"a"]}\n',
    "conflict_across_lists": '{"l":[1]}\n{"l":["a"]}\n',
    "conflict_first_of_two": '{"a":1,"b":1}\n{"b":"x","a":"y"}\n',
    "conflict_before_duplicate": '{"a":1,"b":1}\n{"b":"x","a":2,"a":3}\n',
    "conflict_then_later_syntax": '{"a":1}\n{"a":"x"}\n{"a":}\n',
    "syntax_then_later_conflict": '{"a":1}\n{"a":}\n{"a":"x"}\n',
    "conflict_before_syntax_same_row": '{"a":1}\n{"a":"x","b":}\n',
    "duplicate_before_syntax_same_row": '{"a":1,"a":1,}\n',
    # top-level values that are not objects
    "top_array": '[1,2]\n',
    "top_number": '1\n',
    "top_negative": '-1\n',
    "top_string": '"x"\n',
    "top_true": 'true\n',
    "top_array_after_record": '{"a":1}\n[1]\n',
    "top_close_brace": '{"a":1}}\n',
    "top_lone_close": '}\n',
    "top_colon": '{"a":1}:{"a":2}\n',
    "top_comma": '{"a":1},{"a":2}\n',
    "top_garbage": 'x\n',
    "top_invalid_string": '"x\n',
    "top_invalid_number": '-x\n',
    "trailing_garbage": '{"a":1} x\n',
    # syntax errors (RapidJSON's texts)
    "leading_zero": '{"a":01}\n',
    "truncated_record": '{"a":1\n',
    "truncated_string": '{"a":"x\n',
    "bad_escape": '{"a":"\\q"}\n',
    "lone_high_surrogate": '{"a":"\\ud83d"}\n',
    "lone_low_surrogate": '{"a":"\\ude00"}\n',
    "two_high_surrogates": '{"a":"\\ud83d\\ud83d"}\n',
    "bad_hex": '{"a":"\\u00G9"}\n',
    "control_char_in_string": '{"a":"x\ty"}\n',
    "nul_in_string": b'{"a":"x\x00y"}\n',
    "trailing_comma": '{"a":1,}\n',
    "single_quotes": "{'a':1}\n",
    "capitalised_true": '{"a":True}\n',
    "lowercase_nan": '{"a":nan}\n',
    "minus_only": '{"a":-}\n',
    "fraction_missing": '{"a":1.}\n',
    "leading_dot": '{"a":.5}\n',
    "leading_plus": '{"a":+1}\n',
    "exponent_missing": '{"a":1e}\n',
    "exponent_sign_only": '{"a":1e+}\n',
    "infinity_misspelt": '{"a":Infin}\n',
    "nan_trailing_letter": '{"a":NaNx}\n',
    "true_misspelt": '{"a":tru}\n',
    "missing_colon": '{"a" 1}\n',
    "array_missing_comma": '{"a":[1 2]}\n',
    "bracket_mismatch": '{"a":[1}\n',
    "brace_mismatch": '{"a":{"b":1]}\n',
    "eof_in_string": '{"a":"x',
    "eof_in_key": '{"a',
    "eof_after_key": '{"a"',
    "eof_after_colon": '{"a":',
    "eof_in_array": '{"a":[1',
    "eof_after_array_comma": '{"a":[1,',
    "eof_after_brace": '{',
}

EXPLICIT = {
    "infer_unexpected": ('{"a":"x","b":1,"c":true}\n{"a":"y","c":false,"d":1}\n', S(("b", pa.int32()), ("a", pa.string()))),
    "ignore_unexpected": ('{"a":"x","b":1,"c":true}\n{"a":"y","c":false,"d":1}\n',
                          S(("b", pa.int32()), ("a", pa.string()), unexpected_field_behavior="ignore")),
    "error_unexpected": ('{"a":"x","b":1,"c":true}\n{"a":"y","c":false,"d":1}\n',
                         S(("b", pa.int32()), ("a", pa.string()), unexpected_field_behavior="error")),
    "int32_from_fraction": ('{"b":1.5}\n', S(("b", pa.int32()))),
    "int8_overflow": ('{"b":300}\n', S(("b", pa.int8()))),
    "uint8_negative": ('{"b":-1}\n', S(("b", pa.uint8()))),
    "uint8_negative_zero": ('{"a":-0}\n', S(("a", pa.uint8()))),
    "uint16_max": ('{"b":65535}\n{"b":0}\n', S(("b", pa.uint16()))),
    "uint16_overflow": ('{"b":65536}\n', S(("b", pa.uint16()))),
    "int64_min": ('{"b":-9223372036854775808}\n', S(("b", pa.int64()))),
    "int64_overflow": ('{"a":9223372036854775808}\n', S(("a", pa.int64()))),
    "uint64_max": ('{"a":18446744073709551615}\n', S(("a", pa.uint64()))),
    "uint64_overflow": ('{"a":18446744073709551616}\n', S(("a", pa.uint64()))),
    "int_from_exponent": ('{"a":1e2}\n', S(("a", pa.int64()))),
    "int_from_nan": ('{"b":NaN}\n', S(("b", pa.int32()))),
    "string_from_number": ('{"b":1}\n', S(("b", pa.string()))),
    "string_from_bool": ('{"a":true}\n', S(("a", pa.string()))),
    "string_from_object": ('{"a":{"x":1}}\n', S(("a", pa.string()))),
    "int_from_string": ('{"a":"12"}\n', S(("a", pa.int64()))),
    "bool_from_number": ('{"b":1}\n', S(("b", pa.bool_()))),
    "bool": ('{"b":true}\n{"b":null}\n{"b":false}\n', S(("b", pa.bool_()))),
    "double_from_int": ('{"b":1}\n', S(("b", pa.float64()))),
    "double_nan": ('{"a":NaN}\n', S(("a", pa.float64()))),
    "float32": ('{"b":1.1}\n{"b":3}\n{"b":NaN}\n{"b":-Infinity}\n', S(("b", pa.float32()))),
    "float32_overflow": ('{"a":1e300}\n', S(("a", pa.float32()))),
    "timestamp_ms": ('{"b":"2020-01-01 00:00:00.123"}\n', S(("b", pa.timestamp("ms")))),
    "timestamp_ms_4_digits": ('{"a":"2020-01-01 00:00:00.1234"}\n', S(("a", pa.timestamp("ms")))),
    "timestamp_us": ('{"b":"2020-01-01 00:00:00.1"}\n', S(("b", pa.timestamp("us")))),
    "timestamp_ns": ('{"b":"2020-01-01 00:00:00.123456789"}\n', S(("b", pa.timestamp("ns")))),
    "timestamp_ns_1900": ('{"b":"1900-01-01 00:00:00.000000001"}\n', S(("b", pa.timestamp("ns")))),
    "timestamp_unparseable": ('{"b":"hello"}\n', S(("b", pa.timestamp("s")))),
    "timestamp_utc": ('{"b":"2020-01-01 00:00:00"}\n', S(("b", pa.timestamp("s", tz="UTC")))),
    "timestamp_zone_offset": ('{"b":"2020-01-01 00:00:00+01:00"}\n', S(("b", pa.timestamp("s", tz="Asia/Tokyo")))),
    "timestamp_us_zone": ('{"b":"2020-01-01T00:00:00.123456Z"}\n', S(("b", pa.timestamp("us", tz="America/New_York")))),
    "timestamp_null": ('{"b":null}\n', S(("b", pa.timestamp("ms")))),
    "timestamp_from_number": ('{"a":0}\n', S(("a", pa.timestamp("s")))),
    "nulls": ('{"a":null}\n{}\n', S(("a", pa.int8()))),
    "struct": ('{"s":{"x":1,"y":2}}\n', S(("s", pa.struct([("x", pa.int16())])))),
    "struct_ignore": ('{"s":{"x":1,"y":2}}\n', S(("s", pa.struct([("x", pa.int16())])), unexpected_field_behavior="ignore")),
    "struct_error": ('{"s":{"x":1,"y":2}}\n', S(("s", pa.struct([("x", pa.int16())])), unexpected_field_behavior="error")),
    "struct_nulls": ('{"s":null}\n{}\n{"s":{}}\n', S(("s", pa.struct([("x", pa.float32())])))),
    "struct_from_number": ('{"s":1}\n', S(("s", pa.struct([("x", pa.int8())])))),
    "struct_empty_schema_infers": ('{"s":{"l":[1]}}\n', S(("s", pa.struct([])))),
    "struct_conversion": ('{"s":{"x":1.5}}\n', S(("s", pa.struct([("x", pa.int8())])))),
    "list": ('{"l":[1,2]}\n', S(("l", pa.list_(pa.int8())))),
    "list_nulls": ('{"l":[null,1]}\n{"l":null}\n{}\n', S(("l", pa.list_(pa.int16())))),
    "list_of_lists": ('{"l":[[1],[],null]}\n', S(("l", pa.list_(pa.list_(pa.int32()))))),
    "list_of_struct": ('{"l":[{"x":1,"y":2}]}\n', S(("l", pa.list_(pa.struct([("x", pa.int8())]))))),
    "list_from_object": ('{"s":{}}\n', S(("s", pa.list_(pa.int8())))),
    "missing_field": ('{"a":1}\n', S(("z", pa.int8()))),
    "schema_order": ('{"x":1,"a":"q","y":2}\n', S(("a", pa.string()))),
    "schema_name_twice": ('{"a":1}\n', S(("a", pa.int8()), ("a", pa.int16()))),
    "empty_file": ('', S(("a", pa.string()))),
    "only_newline": ('\n', S(("a", pa.string()))),
    "conversion_after_conflict": ('{"a":1.5}\n{"a":"x"}\n', S(("a", pa.int8()))),
    "conversion_rows": ('{"a":1}\n{"a":2.5}\n{"a":3.5}\n', S(("a", pa.int8()))),
    "error_behaviour_without_schema": ('{"a":1}\n', pj.ParseOptions(unexpected_field_behavior="error")),
    "ignore_behaviour_without_schema": ('{"a":1}\n', pj.ParseOptions(unexpected_field_behavior="ignore")),
    "error_mode_nested_known": ('{"a":1,"s":{"x":1}}\n',
                                S(("a", pa.int64()), ("s", pa.struct([("x", pa.int64())])), unexpected_field_behavior="error")),
    "error_mode_nested_new": ('{"a":1,"s":{"x":1,"y":2}}\n',
                              S(("a", pa.int64()), ("s", pa.struct([("x", pa.int64())])), unexpected_field_behavior="error")),
    "ignored_object": ('{"a":1,"z":{"q":1}}\n', S(("a", pa.int64()), unexpected_field_behavior="ignore")),
    "ignored_conflict": ('{"a":1,"z":[1,"x"]}\n', S(("a", pa.int64()), unexpected_field_behavior="ignore")),
    "ignored_duplicate": ('{"a":1,"z":1,"z":2}\n', S(("a", pa.int64()), unexpected_field_behavior="ignore")),
    "ignored_duplicate_between_kept": ('{"a":1,"z":1,"z":2,"c":1}\n',
                                       S(("a", pa.int64()), ("c", pa.int64()), unexpected_field_behavior="ignore")),
    "duplicate_in_error_mode": ('{"a":1,"a":2}\n', S(("a", pa.int64()), unexpected_field_behavior="error")),
    "duplicate_inferred_with_schema": ('{"z":1,"z":2}\n', S(("a", pa.int64()))),
    "unexpected_then_conflict": ('{"a":1}\n{"a":"x","z":1}\n', S(("a", pa.int64()), unexpected_field_behavior="error")),
    "conflict_then_unexpected": ('{"a":1}\n{"z":1,"a":"x"}\n', S(("a", pa.int64()), unexpected_field_behavior="error")),
}


@pytest.mark.parametrize("name", sorted(PROBES))
def test_probe(name):
    check(PROBES[name])


@pytest.mark.parametrize("name", sorted(EXPLICIT))
def test_explicit_schema(name):
    data, po = EXPLICIT[name]
    check(data, po)


def test_messages_are_pyarrows():
    # Spot checks of the texts themselves, so a regression in both readers cannot hide.
    cases = {
        '{"a":1}\n{"a":"x"}\n': "JSON parse error: Column(/a) changed from number to string in row 1",
        '{"a":1,"a":2}\n': "JSON parse error: Column(/a) was specified twice in row 0",
        '{"a":1\n': "JSON parse error: Missing a comma or '}' after an object member. in row 0",
        '{"a":1}}\n': "JSON parse error: The document is empty.",
        "": "Empty JSON file",
    }
    for data, message in cases.items():
        with pytest.raises(am.ArrowMetalError) as e:
            am.read_json_table(data.encode())
        assert str(e.value) == message


# ---------------------------------------------------------------------------------------------------
# Generated files

WORDS = ["a", "b", "key", "name", "x", "y", "\u00e9", 'k"q', "tab\t", "\u00fc\U0001F600"]


def rstr(rng):
    k = rng.random()
    if k < 0.2:
        return "2020-%02d-%02d %02d:%02d:%02d" % (rng.randrange(1, 13), rng.randrange(1, 29), rng.randrange(24),
                                                   rng.randrange(60), rng.randrange(60))
    if k < 0.3:
        return "".join(chr(rng.choice([rng.randrange(32, 127), rng.randrange(0x80, 0x800), rng.randrange(0x800, 0xd000),
                                       rng.randrange(0x10000, 0x10ffff), rng.randrange(0, 32)]))
                       for _ in range(rng.randrange(0, 12)))
    return rng.choice(WORDS) * rng.randrange(0, 5)


def rvalue(rng, kind):
    if rng.random() < 0.15:
        return None
    if kind == "int":
        return rng.randrange(-10 ** rng.randrange(1, 19), 10 ** rng.randrange(1, 19))
    if kind == "float":
        return rng.choice([rng.random() * 10 ** rng.randrange(-5, 30), float(rng.randrange(-100, 100)), -0.0, 1e300, 5e-324])
    if kind == "num":
        return rng.choice([rng.randrange(-1000, 1000), rng.random()])
    if kind == "bool":
        return rng.random() < 0.5
    if kind == "str":
        return rstr(rng)
    if kind == "ts":
        return "2021-%02d-%02dT%02d:%02d" % (rng.randrange(1, 13), rng.randrange(1, 29), rng.randrange(24), rng.randrange(60))
    if kind == "list":
        return [rvalue(rng, "int") for _ in range(rng.randrange(0, 5))]
    if kind == "obj":
        return {k: rvalue(rng, t) for k, t in [("p", "int"), ("q", "str"), ("r", "list")] if rng.random() < 0.7}
    if kind == "lobj":
        return [rvalue(rng, "obj") or {} for _ in range(rng.randrange(0, 3))]
    return None


def rfile(rng, rows):
    kinds = ["int", "float", "num", "bool", "str", "ts", "list", "obj", "lobj"]
    fields = [("f%d" % i, rng.choice(kinds)) for i in range(rng.randrange(1, 9))]
    lines = []
    for _ in range(rows):
        order = fields[:]
        if rng.random() < 0.3:
            rng.shuffle(order)
        obj = {name: rvalue(rng, kind) for name, kind in order if rng.random() < 0.85}
        lines.append(json.dumps(obj, ensure_ascii=rng.random() < 0.5,
                                separators=rng.choice([(",", ":"), (", ", ": ")])))
        if rng.random() < 0.05:
            lines.append(rng.choice(["", "  ", "\t"]))
    sep = rng.choice(["\n", "\r\n"])
    text = sep.join(lines)
    if rng.random() < 0.7:
        text += sep
    return text.encode()


@pytest.mark.parametrize("seed", range(60))
def test_random_files(seed):
    rng = random.Random(seed)
    check(rfile(rng, rng.randrange(1, 300)))


@pytest.mark.parametrize("seed", range(8))
def test_every_prefix_of_a_file(seed):
    # Truncation at every byte: each prefix either reads as pyarrow reads it or fails with its text.
    rng = random.Random(1000 + seed)
    data = rfile(rng, 3)
    for i in range(1, len(data) + 1):
        check(data[:i])


@pytest.mark.parametrize("seed", range(20))
def test_mutated_files(seed):
    # One or two bytes replaced with a structural character, a digit or a letter.
    rng = random.Random(2000 + seed)
    data = rfile(rng, rng.randrange(1, 40))[:400]
    for _ in range(20):
        b = bytearray(data)
        for _ in range(rng.randrange(1, 3)):
            b[rng.randrange(len(b))] = ord(rng.choice('{}[]:,"\\ \n0123456789-.eEtfnu\x01aNI'))
        check(bytes(b))


def test_every_type_with_nulls_and_missing_keys():
    rows = []
    for i in range(500):
        r = {"i": i, "f": i / 7, "b": i % 3 == 0, "s": "v%d" % (i % 11), "t": "2024-01-%02d 10:00:00" % (i % 28 + 1),
             "l": list(range(i % 4)), "o": {"x": i, "y": "q%d" % i}}
        for k in list(r):
            if (i + len(k)) % 5 == 0:
                r[k] = None
            elif (i + len(k)) % 7 == 0:
                del r[k]
        rows.append(json.dumps(r))
    check("\n".join(rows) + "\n")


@pytest.mark.parametrize("offset", range(0, 300, 7))
def test_escapes_straddling_blocks(offset):
    # The structure pass works in 256-byte blocks; backslash runs and escaped quotes must be read the
    # same wherever a block boundary falls.
    pad = "x" * offset
    check('{"a":"%s%s\\"q"}\n{"a":"%s","b":"\\\\"}\n{"a":"%s"}\n' % (pad, "\\\\" * 40, "\\\\" * 33, "\\\\" * 64 + "y"))


@pytest.mark.parametrize("offset", range(0, 260, 37))
def test_backslash_runs_longer_than_a_block(offset):
    # Whole blocks of backslashes pass the escape state through; the quote after the run is escaped
    # or not by the run's parity, whatever the alignment.
    pad = "x" * offset
    for run in (600, 601, 1024):
        body = "\\" * run
        text = '{"a":"%s%s"}\n' % (pad, body + ("\\" if run % 2 else ""))
        check(text + '{"a":"%s\\"%s"}\n' % (body[:(run // 2) * 2], pad))


def test_wide_file():
    rng = random.Random(7)
    keys = ["k%04d" % i for i in range(1500)]
    lines = []
    for _ in range(40):
        ks = [k for k in keys if rng.random() < 0.9]
        rng.shuffle(ks)
        lines.append(json.dumps({k: rng.randrange(1000) for k in ks}))
    check("\n".join(lines) + "\n")


def test_single_key_file():
    check("".join('{"a":%d}\n' % i for i in range(20000)))


def test_one_record_with_many_fields():
    check("{" + ",".join('"f%d":"%d"' % (i, i) for i in range(5000)) + "}\n")


def test_every_record_a_different_key():
    check("".join('{"u%d":%d,"common":%d}\n' % (i, i, i) for i in range(2000)))


def test_long_string_with_escapes():
    rng = random.Random(3)
    s = "".join(rng.choice(["a", '\\"', "\\\\", "\\n", "\\u00e9", "\\ud83d\\ude00", "xyz"]) for _ in range(20000))
    check('{"s":"%s"}\n{"s":"b"}\n' % s)


def test_file_larger_than_a_pyarrow_block():
    # Beyond pyarrow's 1 MiB block, compare with its single-threaded read (see test_field_order_is_first_appearance).
    rng = random.Random(11)
    data = rfile(rng, 12000)
    assert len(data) > (1 << 20)
    check(data, read_options=pj.ReadOptions(use_threads=False))


# ---------------------------------------------------------------------------------------------------
# API

def test_read_json_returns_metal_arrays(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"x":1,"y":"a"}\n{"x":2,"y":null}\n')
    cols = am.read_json(str(p))
    assert isinstance(cols, am.ColumnSet)
    assert cols.names == ["x", "y"]
    assert isinstance(cols["x"], am.MetalArray)
    assert cols["x"].sum() == 3
    assert cols["y"].to_arrow().to_pylist() == ["a", None]


def test_sources_path_bytes_and_file_object(tmp_path):
    data = b'{"x":1}\n{"x":2}\n'
    p = tmp_path / "a.jsonl"
    p.write_bytes(data)
    want = pj.read_json(io.BytesIO(data))
    assert am.read_json_table(str(p)).equals(want)
    assert am.read_json_table(p).equals(want)
    assert am.read_json_table(data).equals(want)
    assert am.read_json_table(bytearray(data)).equals(want)
    assert am.read_json_table(io.BytesIO(data)).equals(want)


def test_rows_without_columns():
    t = am.read_json_table(b"{}\n{}\n{}\n")
    assert t.num_rows == 3 and t.num_columns == 0
    assert pj.read_json(io.BytesIO(b"{}\n{}\n{}\n")).num_rows == 3


def test_keyword_options_match_parse_options():
    data = b'{"a":"x","b":1,"c":true}\n'
    schema = pa.schema([("b", pa.int32())])
    by_options = am.read_json_table(data, parse_options=pj.ParseOptions(explicit_schema=schema,
                                                                        unexpected_field_behavior="ignore"))
    by_keywords = am.read_json_table(data, explicit_schema=schema, unexpected_field_behavior="ignore")
    assert by_options.equals(by_keywords)
    assert by_keywords.column_names == ["b"]
    # A list of fields is accepted as a schema.
    assert am.read_json_table(data, explicit_schema=[("b", pa.int32())],
                              unexpected_field_behavior="ignore").equals(by_keywords)


def test_newlines_in_values_does_not_change_the_result():
    data = b'{"a":\n1}\n{"a":2}\n'
    want = pj.read_json(io.BytesIO(data))
    for flag in (False, True):
        assert am.read_json_table(data, parse_options=pj.ParseOptions(newlines_in_values=flag)).equals(want)


def test_read_options_are_accepted():
    data = b'{"a":1}\n'
    t = am.read_json_table(data, read_options=pj.ReadOptions(block_size=1 << 16, use_threads=False))
    assert t.equals(pj.read_json(io.BytesIO(data)))


def test_bad_unexpected_field_behavior():
    with pytest.raises(am.ArrowMetalError, match="unexpected_field_behavior"):
        am.read_json_table(b'{"a":1}\n', unexpected_field_behavior="drop")


@pytest.mark.parametrize("typ", [pa.date32(), pa.decimal128(10, 2), pa.binary(), pa.large_string(),
                                 pa.float16(), pa.dictionary(pa.int32(), pa.string()), pa.null()])
def test_explicit_types_outside_the_supported_set_are_rejected(typ):
    with pytest.raises(am.ArrowMetalError, match="/a"):
        am.read_json_table(b'{"a":"1"}\n', explicit_schema=pa.schema([("a", typ)]))


@pytest.mark.parametrize("typ", [pa.decimal128(10, 2), pa.binary(), pa.large_string()])
def test_pyarrow_converts_to_types_this_reader_rejects(typ):
    # The documented difference: pyarrow reads these explicit types; ArrowMetal rejects them.
    data = b'{"a":"1"}\n'
    assert pj.read_json(io.BytesIO(data), parse_options=S(("a", typ))).num_rows == 1
    with pytest.raises(am.ArrowMetalError):
        am.read_json_table(data, explicit_schema=pa.schema([("a", typ)]))


def test_conversion_error_names_the_first_failing_value():
    # With several explicit-schema values failing, ArrowMetal names the one earliest in the file;
    # pyarrow names one of them, which one depending on its conversion order.
    data = b'{"a":1.5,"b":2.5}\n'
    po = S(("b", pa.int8()), ("a", pa.int8()))
    with pytest.raises(am.ArrowMetalError) as e:
        am.read_json_table(data, parse_options=po)
    assert str(e.value) == "Failed to convert JSON to int8, couldn't parse:1.5"
    with pytest.raises(am.ArrowMetalError) as e:
        am.read_json_table(b'{"a":1,"t":"x"}\n{"a":2.5}\n', parse_options=S(("a", pa.int8()), ("t", pa.timestamp("s"))))
    assert str(e.value) == "Failed to convert JSON to timestamp[s], couldn't parse:x"
    with pytest.raises(pa.ArrowInvalid) as p:
        pj.read_json(io.BytesIO(data), parse_options=po)
    assert str(p.value) in ("Failed to convert JSON to int8, couldn't parse:1.5",
                            "Failed to convert JSON to int8, couldn't parse:2.5")


def test_missing_file():
    with pytest.raises(am.ArrowMetalError, match="cannot open"):
        am.read_json("/nonexistent/file.jsonl")


# ---------------------------------------------------------------------------------------------------
# Documented differences from pyarrow 25.0.1 (docs/JSON.md, "Differences from pyarrow")

def run_pyarrow_in_subprocess(data):
    """pyarrow.json.read_json of `data` in a child process: (returncode, stdout)."""
    code = ("import io, sys, pyarrow.json as pj\n"
            "t = pj.read_json(io.BytesIO(sys.stdin.buffer.read()))\n"
            "print(t.to_pylist())\n")
    p = subprocess.run([sys.executable, "-c", code], input=data, capture_output=True, timeout=120)
    return p.returncode, p.stdout.decode(errors="replace")


def test_pyarrow_list_leading_null_defect():
    # pyarrow drops the nulls that open a list before the list's element type is known and shifts
    # the following values; ArrowMetal keeps them. pyarrow reads the same file correctly once the
    # type is given.
    data = b'{"l":[null,1]}\n{"l":[2]}\n'
    assert pj.read_json(io.BytesIO(data)).to_pylist() == [{"l": [1, 2]}, {"l": [0]}]
    want = [{"l": [None, 1]}, {"l": [2]}]
    assert am.read_json_table(data).to_pylist() == want
    schema = pa.schema([("l", pa.list_(pa.int64()))])
    assert pj.read_json(io.BytesIO(data), parse_options=pj.ParseOptions(explicit_schema=schema)).to_pylist() == want
    # The same defect with a struct element can take pyarrow's process down; checked in a child.
    data = b'{"l":[null,{"x":1,"y":"a"}]}\n'
    rc, _ = run_pyarrow_in_subprocess(data)
    assert rc != 0
    assert am.read_json_table(data).to_pylist() == [{"l": [None, {"x": 1, "y": "a"}]}]


def test_leading_null_record():
    # A top-level `null` is a row whose fields are all null (pyarrow does this after the first
    # object); a file that opens with one takes pyarrow's process down.
    data = b'null\n{"a":1}\n'
    rc, _ = run_pyarrow_in_subprocess(data)
    assert rc != 0
    assert am.read_json_table(data).to_pylist() == [{"a": None}, {"a": 1}]
    assert am.read_json_table(b'null\n').num_rows == 1


def test_error_row_counts_from_the_start_of_the_file():
    # pyarrow numbers rows from the start of its 1 MiB block; ArrowMetal from the start of the file.
    good = b'{"a":1}\n' * 200000
    data = good + b'{"a":"x"}\n'
    with pytest.raises(am.ArrowMetalError) as e:
        am.read_json_table(data)
    assert str(e.value) == "JSON parse error: Column(/a) changed from number to string in row 200000"
    with pytest.raises(pa.ArrowInvalid) as p:
        pj.read_json(io.BytesIO(data))
    assert "changed from number to string in row" in str(p.value)
    assert "row 200000" not in str(p.value)
    # With the file in one block pyarrow's row is the file row too.
    with pytest.raises(pa.ArrowInvalid) as p:
        pj.read_json(io.BytesIO(data), read_options=pj.ReadOptions(block_size=len(data) + 1))
    assert str(p.value) == str(e.value)


def test_multiline_object_across_a_block_boundary():
    # With newlines_in_values=False pyarrow splits blocks at newlines and fails on an object that
    # spans one; ArrowMetal has no blocks and reads it, as pyarrow does when the file fits a block.
    data = b'{"a":1}\n' * 7 + b'{"a":\n2}\n' + b'{"a":1}\n' * 7
    small = pj.ReadOptions(block_size=64)
    with pytest.raises(pa.ArrowInvalid):
        pj.read_json(io.BytesIO(data), read_options=small)
    assert am.read_json_table(data).equals(pj.read_json(io.BytesIO(data)))


def test_field_order_is_first_appearance():
    # pyarrow's threaded read of a multi-block file can order fields by a later block; ArrowMetal
    # always orders by first appearance, which is pyarrow's single-threaded order.
    rng = random.Random(7)
    keys = ["k%04d" % i for i in range(2000)]
    lines = []
    for _ in range(300):
        ks = [k for k in keys if rng.random() < 0.9]
        rng.shuffle(ks)
        lines.append(json.dumps({k: rng.randrange(1000) for k in ks}))
    data = ("\n".join(lines) + "\n").encode()
    first = list(dict.fromkeys(k for line in lines for k in json.loads(line)))
    got = am.read_json_table(data)
    assert got.column_names == first
    assert got.equals(pj.read_json(io.BytesIO(data), read_options=pj.ReadOptions(use_threads=False)))


def test_nesting_deeper_than_pyarrows_import_limit():
    # 100 levels read on the GPU; pyarrow's C Data importer stops at 64 levels, so the Table export
    # is the step that fails, not the read.
    data = ('{"a":' + "[" * 100 + "]" * 100 + '}\n').encode()
    cols = am.read_json(data)
    assert len(cols["a"]) == 1
    with pytest.raises(Exception, match="[Rr]ecursion"):
        am.read_json_table(data)


def test_nesting_limit():
    # Past 1024 levels the read stops with an error; pyarrow's process does not survive 2000 levels.
    data = ('{"a":' + "[" * 1100 + "]" * 1100 + '}\n').encode()
    with pytest.raises(am.ArrowMetalError, match="Nesting deeper than 1024 levels"):
        am.read_json(data)
    rc, _ = run_pyarrow_in_subprocess(('{"a":' + "[" * 2000 + "]" * 2000 + '}\n').encode())
    assert rc != 0


def test_invalid_utf8_passes_through_as_pyarrow_does():
    # Neither reader validates UTF-8: the bytes arrive as they are in the file.
    data = b'{"a":"\xff\xfe"}\n'
    got = am.read_json_table(data)["a"].chunk(0)
    want = pj.read_json(io.BytesIO(data))["a"].chunk(0)
    assert got.buffers()[2].to_pybytes()[:2] == want.buffers()[2].to_pybytes()[:2] == b"\xff\xfe"
    with pytest.raises(pa.ArrowInvalid):
        got.validate(full=True)
