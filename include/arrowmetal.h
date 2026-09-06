// ArrowMetal C ABI. Handle-based, speaks the Arrow C Data Interface for input and output.
// Every function returns 0 on success or non-zero on error; am_last_error() has the message.
// Scalars are passed as a pointer to a value of the array's element type (e.g. int64_t* for "l").
#ifndef ARROWMETAL_H
#define ARROWMETAL_H
#include <stdint.h>
#include "arrow_abi.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct am_array am_array;            // opaque, Metal-resident Arrow array

const char* am_version(void);
const char* am_device_name(void);
const char* am_last_error(void);             // thread-local, valid until the next call on this thread

// Lifecycle and interop (zero-copy when the producer's buffers are page aligned; otherwise one copy)
int  am_import(const struct ArrowSchema* schema, struct ArrowArray* array, am_array** out);
int  am_import_device(const struct ArrowSchema* schema, struct ArrowDeviceArray* array, am_array** out);
int  am_export(am_array* a, struct ArrowSchema* schema, struct ArrowArray* array);
int  am_export_device(am_array* a, struct ArrowSchema* schema, struct ArrowDeviceArray* array);
void am_release(am_array* a);
int64_t     am_length(am_array* a);
int64_t     am_null_count(am_array* a);
const char* am_format(am_array* a);          // Arrow format string: c C s S i I l L f g b u z t... d:p,s

// Reductions. out_kind: 0 = int64 in out_i64, 1 = uint64 in out_u64 (same slot), 2 = float64 in out_f64.
// op: 0 sum, 1 min, 2 max, 3 mean. *is_null is set when there is no valid value.
int  am_reduce(am_array* a, int op, int64_t* out_i64, double* out_f64, int* out_kind, int* is_null);
// Notes: min/max skip NaN and report is_null when every valid value is NaN (pyarrow returns NaN there).
// float32 sums accumulate in float64 like Arrow. Float32 arithmetic kernels run in hardware float, which flushes
// subnormal results to zero on Apple GPUs; comparisons and Float64 math are exact.

// Element-wise. cmp op: 0 eq 1 ne 2 lt 3 le 4 gt 5 ge. arith op: 0 add 1 sub 2 mul 3 div.
int  am_compare_scalar(am_array* a, int op, const void* scalar, am_array** out);
int  am_compare_array(am_array* a, int op, am_array* b, am_array** out);
int  am_arith_scalar(am_array* a, int op, const void* scalar, am_array** out);
int  am_arith_array(am_array* a, int op, am_array* b, am_array** out);
int  am_cast(am_array* a, const char* format, am_array** out);
int  am_bool_and(am_array* a, am_array* b, am_array** out);
int  am_bool_or(am_array* a, am_array* b, am_array** out);
int  am_bool_not(am_array* a, am_array** out);

// Selection
int  am_filter(am_array* a, am_array* mask, am_array** out);
int  am_filter_where(am_array* a, int op, const void* scalar, am_array** out);
int  am_take(am_array* a, am_array* indices, am_array** out);
int  am_slice(am_array* a, int64_t offset, int64_t length, am_array** out);

// Sorting (GPU LSD radix sort; stable, nulls last, NaN after +inf).
// Float keys are canonicalised first: -0.0 sorts as 0.0 (they are equal, so the tie keeps input order)
// and every NaN is one value. Nulls and NaN stay at the end when `descending` is set -- a reversed order
// does not mirror them to the front -- and am_top_k maps its keys the same way, so it agrees with
// am_argsort element for element.
int  am_argsort(am_array* a, int descending, am_array** out);   // int32 indices
int  am_sort(am_array* a, int descending, am_array** out);      // sorted copy, same type
int  am_top_k(am_array* a, int64_t k, int largest, am_array** out);  // int32 indices of the k largest/smallest

// Group-by over dense int32/int64 keys in [0, key_count). agg: 0 sum, 1 count(rows), 2 min, 3 max, 4 mean, 5 count(values)
int  am_group_by(am_array* keys, int64_t key_count, int agg, am_array* values /* may be NULL for count rows */, am_array** out);

// Strings (utf8). unary kind: 0 byte length (int32), 1 char length (int32), 2 murmur3 hash (uint32).
// match pred: 0 equals, 1 starts_with, 2 ends_with, 3 contains; pattern is UTF-8 bytes.
int  am_str_unary(am_array* a, int kind, am_array** out);
int  am_str_match(am_array* a, int pred, const uint8_t* pattern, int64_t len, am_array** out);
int  am_str_equals_array(am_array* a, am_array* b, am_array** out);
int  am_str_dictionary_encode(am_array* a, am_array** codes, am_array** unique);

// Temporal (date32 "tdD", date64 "tdm", time32 "tts"/"ttm", time64 "ttu"/"ttn", timestamp "tss:"/"tsm:"/
// "tsu:"/"tsn:" plus an optional timezone, duration "tDs"/"tDm"/"tDu"/"tDn"), binary ("z"/"Z") and
// dictionary-encoded arrays (am_format reports the index format "i"; the values ride in schema.dictionary).
// field: 0 year, 1 month, 2 day, 3 day of week (Monday = 0), 4 hour, 5 minute, 6 second. UTC, int32 out.
int  am_temporal_extract(am_array* a, int field, am_array** out);
// unit: 0 s, 1 ms, 2 us, 3 ns. Timestamps keep their timezone; narrowing truncates toward zero.
int  am_temporal_cast_unit(am_array* a, int unit, am_array** out);
// Materialises a dictionary-encoded array (take of the values by the codes).
int  am_dictionary_decode(am_array* a, am_array** out);

// String transforms (utf8 -> utf8, int32 or bool). One entry point with an op table.
//
// arg1/arg2 are UTF-8 byte arguments and p1/p2 the integer arguments; anything an op does not use may be
// NULL / 0. Output strings have data-dependent lengths, so each op runs two GPU passes (byte lengths, then
// a prefix scan into the offsets buffer, then the bytes). Null rows stay null and produce no bytes.
//
//  op  name                     args                                       output  notes
//  --  -----------------------  -----------------------------------------  ------  -------------------------------
//   0  ascii_upper              -                                          utf8    byte-wise a-z -> A-Z
//   1  ascii_lower              -                                          utf8    byte-wise A-Z -> a-z
//   2  utf8_upper               -                                          utf8    simple case mapping, see below
//   3  utf8_lower               -                                          utf8    simple case mapping, see below
//   4  ascii_swapcase           -                                          utf8    byte-wise ASCII case flip
//   5  ascii_capitalize         -                                          utf8    first byte upper, rest lower
//   6  ascii_trim_whitespace    -                                          utf8    space \t \n \v \f \r, both ends
//   7  ascii_ltrim_whitespace   -                                          utf8    leading only
//   8  ascii_rtrim_whitespace   -                                          utf8    trailing only
//   9  ascii_trim               arg1 = character set (ASCII only)          utf8    both ends; empty set = no-op
//  10  ascii_ltrim              arg1 = character set                       utf8    leading only
//  11  ascii_rtrim              arg1 = character set                       utf8    trailing only
//  12  replace_substring        arg1 = pattern, arg2 = replacement,        utf8    non-overlapping, left to right;
//                               p1 = max replacements (-1 = all)                   empty pattern = identity
//  13  binary_repeat            p1 = count (>= 0)                          utf8    0 gives empty strings
//  14  utf8_slice_codeunits     p1 = start, p2 = stop                      utf8    code point indices, negative
//                                                                                 counts from the end, step 1 only
//  15  utf8_lpad                p1 = width (code points), arg1 = pad char  utf8    empty arg1 pads with a space
//  16  utf8_rpad                p1 = width, arg1 = pad char                utf8    as lpad
//  17  utf8_reverse             -                                          utf8    reverses code points
//  18  count_substring          arg1 = pattern                             int32   non-overlapping; empty pattern
//                                                                                 counts code points + 1
//  19  find_substring           arg1 = pattern                             int32   byte offset, -1 when absent
//  20  ascii_is_alnum           -                                          bool    empty string is false
//  21  ascii_is_alpha           -                                          bool
//  22  ascii_is_decimal         -                                          bool
//  23  ascii_is_space           -                                          bool
//  24  ascii_is_upper           -                                          bool    >= 1 uppercase, no lowercase
//  25  ascii_is_lower           -                                          bool    >= 1 lowercase, no uppercase
//
// Case mapping coverage for ops 2 and 3: simple (1:1 code point) mapping over Basic Latin, Latin-1 Supplement
// (U+00C0-U+00FE minus the x and / signs, plus U+00FF <-> U+0178) and Latin Extended-A (U+0100-U+017F,
// including U+0130/U+0131 and U+017F -> S, which change the byte length). Everything above U+017F is copied
// through unchanged, and the multi-character expansions are not applied: U+00DF (ss), U+0149 and U+00B5 stay
// as they are. Full Unicode case folding and normalisation are out of scope.
int  am_str_transform(am_array* a, int op, const uint8_t* arg1, int64_t len1,
                      const uint8_t* arg2, int64_t len2, int64_t p1, int64_t p2, am_array** out);

// binary_join_element_wise: a[i] + separator + b[i]. Null on either side gives a null output.
int  am_str_concat(am_array* a, am_array* b, const uint8_t* separator, int64_t sep_len, am_array** out);

// Batching: between begin and end, every call on this thread appends to one GPU command buffer. The GPU runs
// once at end (or at the first call that must read a result, such as am_reduce or am_export).
int  am_batch_begin(void);
int  am_batch_end(void);

// Structural and conditional transforms, set lookup, and three-valued (Kleene) logic.
// am_is_null / am_is_valid / am_fill_null / am_drop_null accept primitive and boolean arrays;
// everything else is primitive only except am_if_else, which also takes two boolean branches.
int  am_is_null(am_array* a, am_array** out);      // boolean array, true where a is null; never null itself
int  am_is_valid(am_array* a, am_array** out);     // complement of am_is_null
// scalar points at one value of a's element type; for a boolean array it points at one byte (non-zero = true).
int  am_fill_null(am_array* a, const void* scalar, am_array** out);
int  am_drop_null(am_array* a, am_array** out);    // the non-null elements, in order
// cond ? left : right. cond is boolean; left and right share a type and cond's length. Null cond -> null out.
int  am_if_else(am_array* cond, am_array* left, am_array* right, am_array** out);
// First non-null across `count` arrays of one type and length.
int  am_coalesce(am_array** arrays, int64_t count, am_array** out);
// Set lookup against the non-null values of set_array. Nulls in the set are ignored and a null element
// never matches, so am_is_in never returns nulls; am_index_in returns int32 indices into set_array
// (first occurrence) and null where the element is null or absent. Both search the same total order
// am_unique and the sort use, so on a float column every NaN is one value and -0.0 matches 0.0
// (pyarrow's hash lookup agrees on NaN and keeps the two zeros apart).
int  am_is_in(am_array* a, am_array* set_array, am_array** out);
int  am_index_in(am_array* a, am_array* set_array, am_array** out);
// Three-valued logic over boolean arrays: false AND null = false, true OR null = true.
int  am_and_kleene(am_array* a, am_array* b, am_array** out);
int  am_or_kleene(am_array* a, am_array* b, am_array** out);

// ---------------------------------------------------------------------------------------------------
// Element-wise math, bit-wise ops and cumulative functions (all GPU, all null-aware).
//
// am_unary op numbering:
//    0 negate   1 abs    2 sign    3 sqrt    4 exp     5 ln      6 log10   7 log2
//    8 floor    9 ceil  10 round  11 trunc  12 bit_wise_not
// Notes: 3-7 need a float32/float64 column (cast an integer one first). 8-11 are the identity on an
// integer column and keep its type; `round` rounds halves away from zero. Integer `negate` and `abs`
// wrap, so abs(INT8_MIN) is INT8_MIN. `sign` returns -1/0/1 and leaves NaN and both signed zeros alone.
// (pyarrow's `sign` normalises -0.0 to 0.0 and narrows an integer column to int8; these keep the column's
// own type at every width.) On float32, `sign` and 8-11 read the bit pattern rather than comparing, so a
// subnormal operand is not flushed: sign(1.4e-45) is 1, ceil(1.4e-45) is 1.0 and round(-0.4) is -0.0.
// On float64, ops 0-2 and 8-11 are exact (bit-pattern kernels); 3-7 are evaluated in float and widened,
// so expect about 7 correct significant decimal digits.
int  am_unary(am_array* a, int op, am_array** out);

// am_binary op numbering:
//    0 bit_wise_and   1 bit_wise_or   2 bit_wise_xor   3 shift_left   4 shift_right
//    5 modulo         6 power         7 min_element_wise             8 max_element_wise
// Pass exactly one of `b` (array form) or `scalar` (scalar form, a pointer to a value of the array's
// element type); ops 7 and 8 have no scalar form. Ops 0-4 need an integer column; `shift_right` is
// arithmetic on a signed one and logical on an unsigned one, and a shift count outside [0, bit width)
// yields 0 (or the sign fill for a signed `shift_right`) rather than raising as Arrow does.
// (pyarrow's unchecked shifts return the operand unchanged there, and its range excludes the sign bit.)
// Element-wise min/max skip NaN and break a ±0 tie the way fmin/fmax do: min keeps -0.0 and max keeps 0.0,
// whichever side it came from, so the pair is commutative.
// `modulo` is C remainder (the sign follows the dividend) and defines x % 0 as 0, as `divide` does;
// `power` uses repeated squaring on integers, wraps, and defines a negative exponent as 0. `power` and
// `modulo` are not implemented for float64. Ops 0-6 propagate nulls; 7 and 8 skip them, so a null on one
// side yields the other side's value and only two nulls make a null.
int  am_binary(am_array* a, int op, am_array* b /* or NULL */, const void* scalar /* or NULL */, am_array** out);

// am_cumulative op numbering: 0 cumulative_sum, 1 cumulative_min, 2 cumulative_max.
// Output is null exactly where the input is, and the running value carries across nulls unchanged --
// pyarrow's `skip_nulls=True`, not its default, which propagates the first null to the end of the column.
// Two-level GPU scan; integer sums wrap and are exact, float sums reassociate.
int  am_cumulative(am_array* a, int op, am_array** out);

// ---------------------------------------------------------------------------------------------------
// Decimals: decimal128 (am_format "d:precision,scale") and decimal256 ("d:precision,scale,256").
//
// An element is a fixed-width two's-complement little-endian integer -- 16 bytes for decimal128, 32 for
// decimal256 -- holding the *unscaled* value; `scale` says where the point sits. Import and export go
// through the ordinary am_import / am_export with that format string, and am_filter, am_take and am_slice
// work on both widths. Comparisons also work through am_compare_scalar / am_compare_array (the scalar is
// 16 little-endian bytes), which is what the Python `==`, `<`, ... operators use.
//
// One entry point with an op table. Pass exactly one of `b` (array form) or `scalar` (scalar form);
// anything an op does not use may be NULL / 0. Unless a row says otherwise the scalar is 16 little-endian
// bytes read at the array's own scale, both sides must have the same scale (a mismatch is an error, not a
// silent rescale), nulls propagate, and arithmetic wraps modulo 2^128 exactly as Arrow's unchecked kernels
// do. Everything except the casts runs on the GPU on 64-bit limbs.
//
//  op  name              b / scalar                       p1            output        notes
//  --  ----------------  -------------------------------  ------------  ------------  ----------------------
//   0  equal             either                           -             bool
//   1  not_equal         either                           -             bool
//   2  less              either                           -             bool
//   3  less_equal        either                           -             bool
//   4  greater           either                           -             bool
//   5  greater_equal     either                           -             bool          256-bit scalars are
//                                                                                     sign-extended from 16 B
//   6  add               either                           -             same decimal  wraps
//   7  subtract          either                           -             same decimal  wraps
//   8  multiply          b: element-wise; scalar: int64_t  -             decimal       array form gives
//                                                                                     precision p1+p2+1 and
//                                                                                     scale s1+s2; the int64
//                                                                                     form keeps the type
//   9  negate            -                                -             same decimal
//  10  abs               -                                -             same decimal
//  11  sign              -                                -             int32         -1 / 0 / 1
//  12  round             -                                target scale  decimal       halves away from zero
//  13  ceil              -                                target scale  decimal       toward +inf
//  14  floor             -                                target scale  decimal       toward -inf
//  15  truncate          -                                target scale  decimal       toward zero
//  16  cast to float64   -                                -             float64       CPU; 53-bit precision
//  17  cast to decimal   -                                scale         decimal128    CPU; input is float64
//                                                                                     or int64, precision 38
//  18  sum               -                                -             decimal[1]    null when no valid value
//  19  min               -                                -             decimal[1]
//  20  max               -                                -             decimal[1]
//
// Ops 12-15 return a decimal with scale = p1: scaling up multiplies (exact, widening the precision) and
// scaling down divides with the named rounding mode. Reductions come back as a length-1 array because a
// 128-bit result does not fit an int64_t out-parameter; read it with am_export.
//
// decimal256 supports import/export, ops 0-5, am_filter / am_take / am_slice and op 18 (sum). The other
// ops raise an error naming decimal128 rather than computing something wrong.
int  am_decimal_op(am_array* a, int op, am_array* b /* or NULL */, const void* scalar /* or NULL */,
                   int64_t p1, am_array** out);
// Nested types: list ("+l"), large_list ("+L", offsets narrowed to int32 on import), fixed_size_list
// ("+w:N"), struct ("+s"), map ("+m") and dense/sparse union ("+ud:", "+us:").
//
// am_import / am_export carry all of them through the C Data Interface, recursively, and am_filter /
// am_take / am_slice work on them unchanged. The calls below are the nested-specific surface.
// am_list_* accept a list, large_list, fixed_size_list or map array; a map is a list of
// struct<key, value>, so am_list_flatten on one yields its entries struct.
int  am_list_value_length(am_array* a, am_array** out);            // int32 per-row child count, null in / null out
int  am_list_flatten(am_array* a, am_array** out);                 // the child, restricted to the referenced range
int  am_list_element(am_array* a, int64_t index, am_array** out);  // element `index` of every row; null when absent
int  am_struct_field(am_array* a, const char* name, am_array** out);
// Child navigation: 1 child for a list (its values), a map (its entries struct) or a dictionary (its
// values), one per field for a struct, one per variant for a union, 0 for a flat array.
int64_t am_child_count(am_array* a);                               // -1 for a null handle
int  am_child(am_array* a, int64_t i, am_array** out);
// Regular expressions, SQL LIKE and splitting (utf8 in).
//
// Matching runs on the CPU (ICU, via NSRegularExpression), sharded across cores. A pattern with no
// metacharacter (none of \ . [ ] { } ( ) * + ? ^ $ |) is routed to the existing byte-wise GPU kernels
// instead, as is a `^literal` pattern (ICU's `^` is exactly "start of input") and a LIKE pattern whose
// only wildcards are a leading and/or trailing `%`. A trailing `$` deliberately stays on the CPU:
// ICU's `$` also matches immediately before a final line terminator, so `abc$` matches "abc\n" while
// ends_with("abc") does not.
//
// `pattern` is the regular expression, the literal separator (ops 5/6) or the LIKE pattern (op 4).
// `repl` is the ICU replacement template for op 3 and the capture-group name for op 9. `flags` bit 0
// requests case-insensitive matching. Anything an op does not use may be NULL / 0.
//
//  op  name                        args                                  output  notes
//  --  --------------------------  ------------------------------------  ------  --------------------
//   0  match_substring_regex       pattern                               bool    unanchored search
//   1  count_substring_regex       pattern                               int32   non-overlapping
//   2  find_substring_regex        pattern                               int32   byte offset, -1 absent
//   3  replace_substring_regex     pattern, repl = ICU template          utf8    every match; groups
//                                                                                are $1, $2 (not \1)
//   4  match_like                  pattern = SQL LIKE (% and _)          bool    \ escapes % _ \
//   5  split_pattern values        pattern = literal separator           utf8    the flattened pieces
//   6  split_pattern offsets       pattern = literal separator           int32   n+1 list offsets
//   7  split_whitespace values     -                                     utf8    runs of ASCII space
//   8  split_whitespace offsets    -                                     int32   n+1 list offsets
//   9  extract_regex               pattern, repl = group name            utf8    one named group;
//                                                                                null where no match
//  10  split_pattern_regex values  pattern                               utf8
//  11  split_pattern_regex offsets pattern                               int32   n+1 list offsets
//
// ArrowMetal has no list type, so a split is returned as the (values, offsets) pair of an Arrow
// list<utf8>: row i owns values[offsets[i] .. offsets[i+1]). Call the op twice, once for each half.
// A null input row owns no pieces and its two offsets are equal.
int  am_regex(am_array* a, int op, const uint8_t* pattern, int64_t len,
              const uint8_t* repl, int64_t rlen, int flags, am_array** out);

// Casts between utf8 and the numeric and boolean types.
//
// am_to_strings is Arrow `cast(utf8)` over a primitive or boolean array. Integers are formatted on the
// GPU (two passes: digit counts, prefix scan into the offsets buffer, then the digits) with no leading
// zeros and no separators; INT64_MIN and UINT64_MAX are exact. Floats are formatted on the CPU as the
// shortest decimal string that round-trips, which differs from Arrow in two documented ways: a whole
// value keeps a ".0" (1.0, not 1) and the exponent form is Swift's (1e+20). Booleans give "true" /
// "false". Nulls stay null and emit no bytes.
int  am_to_strings(am_array* a, am_array** out);

// am_parse does three things, chosen by the input type and the format string:
//   * a is utf8 and format is one of "c C s S i I l L f g b" -> Arrow `cast` to that type. Integers
//     parse on the GPU; the whole value must match [+-]?[0-9]+ (leading zeros fine, no whitespace, no
//     radix prefix, no exponent) and a "-" is rejected for an unsigned target. Floats and bools parse
//     on the CPU; a bool is "true"/"false"/"1"/"0", case-insensitively. A value that does not parse,
//     or that is out of the target's range, comes back null unless `strict` is non-zero, which makes
//     it an error instead.
//   * a is utf8 and format is anything else -> `strptime` with that C format, UTC, producing
//     timestamp[us]; rescale afterwards with am_temporal_cast_unit. The whole value must be consumed.
//     Fields the format does not mention default to 1970-01-01 00:00:00.
//   * a is temporal -> `strftime` with that C format, UTC, producing utf8. %f is an ArrowMetal
//     extension expanding to the six-digit fractional second. `strict` is ignored.
int  am_parse(am_array* a, const char* format, int strict, am_array** out);

// Temporal rounding, arithmetic and the calendar fields am_temporal_extract does not cover. All GPU,
// all UTC; a timestamp's timezone rides along as metadata and is never applied.
//
//  op  name             extra args                        output       notes
//  --  ---------------  --------------------------------  -----------  ----------------------------
//   0  floor_temporal   p1 = unit | (multiple << 8)        same type    largest multiple <= value
//   1  ceil_temporal    p1 = unit | (multiple << 8)        same type    a value on a boundary stays
//   2  round_temporal   p1 = unit | (multiple << 8)        same type    halves go up (+infinity)
//   3  add_duration     b = duration column, or p1 ticks   same type    b is rescaled to a's unit
//   4  subtract         b = same family                    duration     finer of the two units
//   5  days_between     b = date-carrying column           int64        whole UTC days, a -> b
//   6  quarter          -                                  int32        1-4
//   7  day_of_year      -                                  int32        1-based
//   8  iso_week         -                                  int32        ISO 8601, 1-53
//   9  iso_year         -                                  int32        ISO 8601 week-numbering year
//  10  is_leap_year     -                                  bool
//  11  millisecond      -                                  int32        since the last full second
//  12  microsecond      -                                  int32        since the last full ms
//  13  nanosecond       -                                  int32        since the last full us
//
// Rounding units for p1's low byte: 0 nanosecond, 1 microsecond, 2 millisecond, 3 second, 4 minute,
// 5 hour, 6 day, 7 month, 8 quarter, 9 year. A multiple of 0 means 1. Units 0-6 are fixed length and
// round by integer arithmetic in the value's own resolution — rounding to a unit finer than that
// resolution is the identity. Units 7-9 go through the civil calendar and need a column that carries
// a date (date32, date64, timestamp); a duration or a time-of-day column is an error. add_duration is
// an error on date32, whose tick is a whole day.
int  am_temporal_math(am_array* a, int op, int64_t p1, am_array* b /* or NULL */, am_array** out);

// ---------------------------------------------------------------------------------------------------
// Window functions, shifts, pairwise differences, running products/means and rolling windows (all GPU).
//
// am_window op numbering, and what p1 / p2 / scalar_or_null mean for each:
//
//   op  name              p1              p2            scalar_or_null   output   notes
//   --  ----------------  --------------  ------------  ---------------  -------  --------------------
//    0  row_number        -               -             -                int32    1-based, sort order
//    1  rank              -               -             -                int32    min rank of a tie
//    2  dense_rank        -               -             -                int32    no gaps
//    3  percent_rank      -               -             -                float64  (rank - 1)/(n - 1)
//    4  cume_dist         -               -             -                float64  rows <= value, / n
//    5  shift             by (lag > 0)    -             fill or NULL     input    NULL fill = nulls
//    6  pairwise_diff     period          -             -                input    a[i] - a[i - period]
//    7  cumulative_prod   -               -             -                input    running product
//    8  cumulative_mean   -               -             -                float64  running mean
//    9  rolling_sum       window          min_periods   -                input    trailing window
//   10  rolling_min       window          min_periods   -                input
//   11  rolling_max       window          min_periods   -                input
//   12  rolling_mean      window          min_periods   -                float64
//
// Ranking (ops 0-4) is one GPU argsort plus a scan, and the answer comes back aligned to the original
// rows. Nulls follow SQL `ORDER BY x NULLS LAST`: they sort after every value and form one tie group, so
// row_number numbers them last in row order, rank and dense_rank give them all one rank, and no ranking
// result is itself null. Float ties use Arrow value equality: every NaN is one value (after +inf) and
// -0.0 equals 0.0.
//
// Ops 5 and 6 are null-propagating: a row that reads outside the array is null (or takes the shift's
// fill scalar, a pointer to one value of the array's element type), and pairwise_diff is null wherever
// either side is. A negative p1 leads rather than lags.
//
// Ops 7 and 8 skip nulls the way am_cumulative does: the output is null exactly where the input is and
// the running value carries across unchanged. Integer products wrap; float32 and float64 reassociate.
// cumulative_mean converts to binary64 first, so int64 magnitudes above 2^53 round on the way in.
//
// Rolling windows (ops 9-12) are trailing: the window ending at row i covers rows [i - window + 1, i].
// p2 = min_periods is how many non-null rows the window needs before a value is produced; 0 or less
// means "the whole window". min and max scan the window (O(n * window), NaN skipped, an all-NaN window
// giving +/-inf); sum and mean are the difference of two prefix sums, so they are O(n) but a NaN or
// infinity anywhere in a float column poisons every later window.
int  am_window(am_array* a, int op, int64_t p1, int64_t p2, const void* scalar_or_null, am_array** out);

// Multi-column (lexicographic) sort: int32 indices ordering the rows by each column in turn, the first
// column being the most significant. `descending` has one entry per column, or may be NULL for all
// ascending. Successive stable radix argsorts from the least significant key upwards; nulls come last in
// every key whichever direction it is sorted in. utf8, binary and dictionary columns are not sortable.
int  am_lexsort(am_array** columns, const int* descending, int64_t count, am_array** out);
// Statistical and positional aggregates, run-end encoding.
//
// am_reduce_ex writes one scalar, exactly like am_reduce: out_kind is 0 for an int64 in *out_i64,
// 1 for a uint64 in the same slot, 2 for a double in *out_f64, and *is_null is set when the column has
// no value to answer with (every row null, or fewer values than the degrees of freedom).
//
//  op  function                  p1                    out_kind          notes
//  --  ------------------------  --------------------  ----------------  ---------------------------------
//   0  product                   -                     column's kind     integers wrap in int64 / uint64
//   1  variance (population)     -                     double            ddof = 0
//   2  variance (sample)         -                     double            ddof = 1
//   3  stddev (population)       -                     double
//   4  stddev (sample)           -                     double
//   5  quantile                  q in [0, 1] (clamped) double            linear interpolation, exact
//   6  approximate_median        -                     double            exact: a GPU sort, not a sketch
//   7  mode (value)              -                     column's kind     ties go to the smallest value
//   8  count_distinct            -                     int64             non-null values only
//   9  first                     -                     column's kind     skips nulls
//  10  last                      -                     column's kind     skips nulls
//  11  index                     the value to find     int64             first row holding it, -1 when absent
//  12  any                       -                     int64 (0 or 1)    boolean columns only
//  13  all                       -                     int64 (0 or 1)    boolean columns only
//  14  min_max -> min            -                     column's kind     one kernel produces both
//  15  min_max -> max            -                     column's kind
//  16  mode (count)              -                     int64             how often the mode occurs
//
// Ops 0-11 and 14-16 accept primitive and temporal columns (a temporal column aggregates its storage
// integers); ops 8, 12 and 13 accept boolean columns. `p1` for op 11 is a double, so an integer value
// above 2^53 cannot be expressed exactly; ops that do not use it ignore it.
//
// Precision: float32 statistics accumulate squared deviations in compensated float pairs and float64
// ones in software binary64, so expect about 1e-7 and 1e-15 relative error respectively. A float32
// `product` over thousands of factors reassociates and rounds in float (about 1e-5 relative).
// `tdigest` is out of scope: `quantile` here is exact, so there is no sketch to approximate it with.
int  am_reduce_ex(am_array* a, int op, double p1, int64_t* out_i64, double* out_f64, int* out_kind, int* is_null);

// Run-end encoding ("+r"): two children, run_ends (int32 here; int16/int32/int64 are accepted on import)
// and values. am_format reports "+r"; am_length reports the decoded, logical length.
// am_run_end_encode takes a primitive, boolean or temporal array and collapses adjacent equal values
// (bit equality, and nulls form runs of their own). am_run_end_decode expands one back.
// take, filter and slice on a run-end encoded array decode first; IPC writing needs a decode as well.
int  am_run_end_encode(am_array* a, am_array** out);
int  am_run_end_decode(am_array* a, am_array** out);
// The temporal functions beyond am_temporal_extract and am_temporal_math: the option-carrying week
// numbers, the struct-valued extractors, subsecond, is_dst and every *_between difference. UTC
// throughout — a timestamp's timezone rides along as metadata and is never applied to the value —
// with is_dst the one exception, since its whole job is to ask what that timezone was doing.
//
// `b` is the second column for the *_between ops (a is the start, b the end, so the answer is
// positive when b is later) and is ignored otherwise. Anything an op does not use may be NULL / 0.
//
//  op  name                   p1                    p2          output   notes
//  --  ---------------------  --------------------  ----------  -------  ------------------------------
//   0  week                   WeekOptions bits      -           int64    see the bits below
//   1  us_week                -                     -           int64    Sunday weeks, 1-53
//   2  us_year                -                     -           int64    US epidemiological year
//   3  iso_calendar           -                     -           struct   iso_year/iso_week/iso_day_of_week
//   4  year_month_day         -                     -           struct   year/month/day
//   5  is_dst                 -                     -           bool     CPU; needs a timezone
//   6  day_of_week            count_from_zero       week_start  int64    week_start 1 = Mon ... 7 = Sun
//   7  subsecond              -                     -           float64  fraction of a second, [0, 1)
//   8  years_between          -                     -           int64    calendar years crossed
//   9  quarters_between       -                     -           int64    difference of year*4 + quarter
//  10  months_between         -                     -           int64    difference of year*12 + month
//  11  weeks_between          count_from_zero       week_start  int64    week boundaries crossed
//  12  hours_between          -                     -           int64    hour boundaries crossed
//  13  minutes_between        -                     -           int64
//  14  seconds_between        -                     -           int64
//  15  milliseconds_between   -                     -           int64
//  16  microseconds_between   -                     -           int64
//  17  nanoseconds_between    -                     -           int64    wraps in int64 past ~292 years
//
// Op 0's p1 packs WeekOptions: bit 0 week_starts_monday, bit 1 count_from_zero, bit 2
// first_week_is_fully_in_year. The default (bit 0 alone, so p1 = 1) is iso_week; p1 = 0 is us_week.
// count_from_zero numbers the weeks against the value's own calendar year, so a date at the start of
// a year belonging to the previous year's last week comes out as 0 rather than 52 or 53.
// first_week_is_fully_in_year makes week 1 the first week lying wholly inside January; without it the
// ISO majority rule applies and a week beginning on 29, 30 or 31 December is week 1 of the next year.
//
// p2 = week_start uses the ISO numbering (1 = Monday ... 7 = Sunday) and is unaffected by
// count_from_zero, which only decides whether op 6 counts from 0 or from 1. Op 11 accepts
// count_from_zero for signature parity with Arrow but the value does not change its answer.
//
// Ops 3 and 4 return a struct array; read its fields with am_struct_field. A null input row makes the
// struct row null and leaves the three children valid, which is the shape Arrow produces.
//
// Ops 0-6 and 8-11 need a column that carries a date (date32, date64 or timestamp); op 7 and ops
// 12-17 also accept time32 / time64 (ticks since midnight). duration is rejected everywhere here, and
// op 5 needs a timestamp whose format string carries a timezone — a naive one is an error, exactly as
// in Arrow. Every *_between counts *boundaries crossed*: each side is truncated to the unit first and
// the difference taken afterwards, so it is not the truncated difference. The two sides may differ in
// unit and in type (a date32 against a timestamp[ns], say); both are mapped onto the op's own ruler
// first, which is more permissive than Arrow, where both arguments must have the same type.
int  am_temporal_extra(am_array* a, int op, int64_t p1, int64_t p2, am_array* b_or_null, am_array** out);

// ---------------------------------------------------------------------------------------------------
// The remaining Arrow string surface: character-class predicates, capitalize / title / center /
// replace-slice / trim / normalize, extract_regex_span, binary_join and string is_in / index_in.
//
// am_string_predicate returns a boolean array; nulls propagate. Ops 0-2 are byte-wise and entirely
// GPU. Ops 3-12 are Arrow's Unicode utf8_is_* family: one GPU pass answers every row and reports, in a
// second bitmap, which rows carry a byte >= 0x80; only those rows are re-decided on the CPU, so an
// ASCII-only column never leaves the device. The empty string is true for the two printable predicates
// and for string_is_ascii, and false for every other op.
//
//  op  name                 GPU/CPU   notes
//  --  -------------------  --------  ----------------------------------------------------------------
//   0  ascii_is_printable   GPU       every byte in 0x20-0x7E
//   1  ascii_is_title       GPU       byte-wise title case over runs of ASCII letters
//   2  string_is_ascii      GPU       every byte < 0x80
//   3  utf8_is_alnum        GPU/CPU   every code point is a letter or a number
//   4  utf8_is_alpha        GPU/CPU   category L*
//   5  utf8_is_decimal      GPU/CPU   category Nd
//   6  utf8_is_digit        GPU/CPU   category Nd or No
//   7  utf8_is_lower        GPU/CPU   >= 1 cased code point, none of them upper case
//   8  utf8_is_numeric      GPU/CPU   category Nd, Nl or No
//   9  utf8_is_printable    GPU/CPU   not Cc/Cf/Cs/Co/Cn/Zs/Zl/Zp, with U+0020 added back
//  10  utf8_is_space        GPU/CPU   Zs/Zl/Zp plus U+0009-U+000D, U+001C-U+001F and U+0085
//  11  utf8_is_title        GPU/CPU   >= 1 cased code point, in title case
//  12  utf8_is_upper        GPU/CPU   >= 1 cased code point, none of them lower case
//
// pyarrow classifies with utf8proc; this uses Swift's Unicode.Scalar.Properties, reconstructing
// utf8proc's rules: a code point is upper case when its simple lower-case mapping changes it or it is
// category Lt, and lower case when its simple upper-case mapping changes it or it is category Ll, minus
// the Roman numerals U+2160-U+216F. A titlecase letter is therefore both, which is why utf8_is_upper
// and utf8_is_lower are both false on U+01C5. Note U+001C-U+001F are utf8_is_space but not
// ascii_is_space, and U+200B (zero-width space) is not whitespace at all.
int  am_string_predicate(am_array* a, int op, am_array** out);

// am_string_transform. arg1/arg2 are UTF-8 byte arguments, p1/p2 the integer ones; anything an op does
// not use may be NULL / 0.
//
//  op  name                    args                              output  GPU/CPU  notes
//  --  ----------------------  --------------------------------  ------  -------  ---------------------
//   0  ascii_title             -                                 utf8    GPU      byte-wise
//   1  utf8_capitalize         -                                 utf8    GPU/CPU  first cp up, rest down
//   2  utf8_title              -                                 utf8    GPU/CPU  first cased cp of a word
//   3  utf8_center             p1 = width, arg1 = pad            utf8    GPU      odd pad on the right
//   4  utf8_replace_slice      p1 = start, p2 = stop, arg1 = new utf8    GPU      code point indices
//   5  binary_replace_slice    p1 = start, p2 = stop, arg1 = new binary  GPU      byte indices
//   6  utf8_trim               arg1 = character set              utf8    GPU/CPU
//   7  utf8_ltrim              arg1 = character set              utf8    GPU/CPU
//   8  utf8_rtrim              arg1 = character set              utf8    GPU/CPU
//   9  utf8_trim_whitespace    -                                 utf8    GPU/CPU  Unicode whitespace
//  10  utf8_ltrim_whitespace   -                                 utf8    GPU/CPU
//  11  utf8_rtrim_whitespace   -                                 utf8    GPU/CPU
//  12  utf8_normalize          p1 = 0 NFC, 1 NFKC, 2 NFD, 3 NFKD utf8    CPU      Foundation
//  13  extract_regex_span      arg1 = pattern, arg2 = group      int32   CPU      byte start offset
//  14  extract_regex_span len  arg1 = pattern, arg2 = group      int32   CPU      byte length
//
// Ops 1, 2, 6-11 take the GPU byte kernel when the whole column is ASCII (ops 6-8 whenever the
// character set itself is ASCII, since a byte-wise trim can never split a UTF-8 sequence) and the CPU,
// sharded over 4096-row chunks, otherwise. `utf8_center` counts code points and needs a one-code-point
// pad; `ascii_center`, which counts bytes, is not exposed separately, exactly as `ascii_lpad` is not.
// Slice indices count code points (op 4) or bytes (op 5): negative values count from the end, both ends
// clamp into range, and a stop below start inserts without deleting. Ops 13/14 need a pattern with at
// least one (?<name>...) group and report byte offsets; a row that does not match, and a group that
// took part in no alternative, are null in both arrays. p2 bit 0 requests case-insensitive matching.
// Two documented differences from pyarrow: op 12 follows the Unicode standard (and Python's
// unicodedata.normalize), while pyarrow's utf8_normalize never composes, so its NFC output equals its
// NFD output; and ops 13/14 use ICU, whose \d matches every Unicode decimal digit, where pyarrow's RE2
// \d is ASCII only — the same difference am_regex already documents.
int  am_string_transform(am_array* a, int op, int64_t p1, int64_t p2,
                         const uint8_t* arg1, int64_t len1,
                         const uint8_t* arg2, int64_t len2, am_array** out);

// Arrow is_in / index_in over utf8, on the GPU. The value set is hashed with the 64-bit key
// dictionary_encode uses and inserted into an open-addressing table of row indices; every probe
// confirms its candidate by comparing bytes, so a hash collision costs one extra probe and never a
// wrong answer, and duplicates in the set collapse onto the lowest row index, which is what index_in
// reports. Nulls in the value set are ignored and a null value is never in the set, so is_in never
// returns a null and index_in is null exactly where the value is null or absent. pyarrow's default is
// the opposite (skip_nulls=False, where a null value matches a null in the set); that option is not
// implemented — pass skip_nulls=True to pyarrow to compare.
int  am_string_is_in(am_array* a, am_array* set, am_array** out);     // bool
int  am_string_index_in(am_array* a, am_array* set, am_array** out);  // int32

// Arrow binary_join over a list<utf8>: two GPU passes, one summing the child byte lengths plus
// count - 1 separators into a per-row output length and one copying the bytes. Pass a scalar separator
// in sep / sep_len, or a per-row utf8 column in sep_array (which then wins; pass NULL for the scalar
// form). An empty row joins to the empty string; a null row, any null element inside a row, and a null
// separator all give a null output row (Arrow's EMIT_NULL; the REPLACE / SKIP options are not
// implemented). For the two-column form see am_str_concat (binary_join_element_wise).
int  am_binary_join(am_array* list, const uint8_t* sep, int64_t sep_len,
                    am_array* sep_array /* or NULL */, am_array** out);

// ---------------------------------------------------------------------------------------------------
// The remaining Arrow type-matrix rows and the type-adjacent functions.
//
// am_import / am_export already carry every one of these through the C Data Interface, and am_filter /
// am_take / am_slice already work on them, so what follows is only the compute each type has:
//
//   format        type                      notes
//   ------------  ------------------------  ----------------------------------------------------------
//   "n"           null                      length only, no buffers; exports with n_buffers = 0
//   "e"           float16                   binary16 patterns; compute goes through float32
//   "d:p,s,32"    decimal32                 widens to decimal128 for compute
//   "d:p,s,64"    decimal64                 widens to decimal128 for compute
//   "tiM"         interval[month]           int32 months
//   "tiD"         interval[day_time]        int32 days + int32 milliseconds
//   "tin"         interval[month_day_nano]  int32 months + int32 days + int64 nanoseconds
//   "w:N"         fixed_size_binary         N raw bytes per element
//   "+vl" / "+vL" list_view / large_list_view   imported as "+l" (see below), never exported as a view
//
// A "+vl" / "+vL" array is converted on import to the contiguous int32 offsets MetalListArray uses: rows
// that already lie back to back keep the producer's child untouched, and anything else (out of order,
// overlapping or with gaps) materialises a contiguous child with one GPU gather. Either way the array
// exports as a plain list ("+l"), because ArrowMetal has no view-shaped column.
//
// An extension type is a storage type plus the schema metadata keys ARROW:extension:name and
// ARROW:extension:metadata. am_import decodes them (the metadata blob is the C Data Interface's own
// int32-count, length-prefixed key/value encoding) and keeps them beside the storage array, am_format
// reports the *storage* format, and am_export writes both keys back so a consumer that knows the type —
// pyarrow with the extension registered — reconstructs it. Every other metadata key survives too.

// float16 <-> float32 on the GPU. `to_half` non-zero narrows a float32 column (round to nearest-even,
// overflow to +/-infinity); zero widens a float16 column (exact). Arithmetic is never done in half
// precision: widen, compute with the float32 kernels, and cast back here when you want a half result.
int  am_cast_float16(am_array* a, int to_half, am_array** out);

// decimal32 / decimal64 -> decimal128 on the GPU (sign extension into two limbs), so am_decimal_op can
// run on the result. A decimal128 column passes through unchanged.
int  am_decimal_widen(am_array* a, am_array** out);
// decimal128 -> decimal32 (`bit_width` 32) or decimal64 (64) on the GPU, keeping the scale. `precision`
// 0 means "as much as the target width allows". A value that does not fit wraps (Arrow's unchecked cast).
int  am_decimal_narrow(am_array* a, int bit_width, int64_t precision, am_array** out);

// Arrow equal (`op` 0) / not_equal (`op` 1) over a fixed_size_binary column, on the GPU as a byte
// compare. Pass `b_or_null` for the array form, or `scalar_bytes` / `len` (exactly the element width)
// for the scalar form. Null in, null out; the array form ANDs the two validity bitmaps. Ordering
// comparisons are not defined for this type and are an error.
int  am_fixed_binary_compare(am_array* a, int op, am_array* b_or_null,
                             const uint8_t* scalar_bytes, int64_t len, am_array** out);
// FNV-1a 64 over each element's bytes, on the GPU: an ArrowMetal extension, not Arrow's `hash64`.
// Null in, null out; the output is uint64.
int  am_fixed_binary_hash64(am_array* a, am_array** out);

// Arrow add(timestamp | date, interval) on the GPU. `interval` is a "tiM", "tiD" or "tin" column of a's
// length, or of length 1 to broadcast. Month arithmetic goes through the civil calendar and clamps the
// day to the target month's length (2024-01-31 + 1 month = 2024-02-29), which is what Arrow does; days
// are whole UTC days, and the interval's sub-day field is converted to the column's own resolution,
// truncating toward zero when the column is coarser. date32 counts whole days, so an interval with a
// non-zero sub-day part is rejected there rather than silently dropped. The result has a's type and is
// null wherever either side is.
//
// pyarrow has no `add` kernel for (timestamp, interval), so this one has no pyarrow oracle to compare
// against; it is checked against a host civil-calendar oracle instead.
int  am_add_interval(am_array* a, am_array* interval, am_array** out);

// The three Arrow difference functions that return an interval, on the GPU. `kind` is 0
// month_interval_between ("tiM"), 1 day_time_interval_between ("tiD"), 2
// month_day_nano_interval_between ("tin"); `a` is Arrow's `start` and `b` its `end`.
//
// Every field is the difference of the corresponding *truncated* field, which is how Arrow defines
// these: months are month boundaries crossed ((y2 - y1) * 12 + (m2 - m1), so 2020-01-31 -> 2020-02-01 is
// one month), the day field is the difference of the day-of-month fields (month_day_nano) or of the
// whole days (day_time), and the sub-day field is the difference of the two times of day. The day and
// sub-day fields may therefore have the opposite sign to the month count. Both columns must be date or
// timestamp columns of the same length; they are brought to a common resolution first. Null in, null out.
int  am_interval_between(am_array* a, am_array* b, int kind, am_array** out);

// One field of an interval column as a plain integer column: `field` 0 months (int32), 1 days (int32),
// 2 nanoseconds (int64). A field the layout does not carry comes back as zeros. This is how to read an
// interval[month] or interval[day_time] column from a binding whose Arrow library cannot represent those
// types (pyarrow 25 cannot wrap them in Python).
int  am_interval_field(am_array* a, int field, am_array** out);

// Arrow list_parent_indices: for every child element the list references — the same range
// am_list_flatten returns — the index of the row that covers it. One GPU binary search per element, so
// empty and null rows cost nothing. Accepts list, large_list, fixed_size_list and map. The result is
// **int32** where pyarrow returns int64, because list offsets are int32 throughout this package.
int  am_list_parent_indices(am_array* a, am_array** out);

// Arrow list_slice: row[start:stop:step] for every row, as a variable-length list ("+l") whatever the
// input layout was. A negative `stop` means "to the end of each row"; `start` must be >= 0 and `step`
// >= 1, as Arrow requires. A null row stays null and a row shorter than `start` becomes empty. GPU: one
// kernel for the new row lengths, the shared scan for the offsets, one kernel to expand the indices.
int  am_list_slice(am_array* a, int64_t start, int64_t stop, int64_t step, am_array** out);

// Arrow map_lookup: the value(s) whose key matches, per row. `occurrence` is 0 first, 1 last, 2 all.
// For a map with utf8 or binary keys the key is `key_bytes` / `len`; for a map with integer keys it is
// the first 8 bytes of `key_bytes` read as a little-endian int64 (`len` must be 8). `first` / `last`
// return the map's item type, `all` returns a list of it, and all three are null where the row is null
// or the key is absent — an empty list never stands for "not found", matching pyarrow. GPU: one kernel
// scans each row's entry range reporting the first match, the last match and the match count; `all`
// scans the counts into offsets and a second kernel writes the matching entry indices to gather.
// Float and nested key types are rejected.
int  am_map_lookup(am_array* a, const uint8_t* key_bytes, int64_t len, int occurrence, am_array** out);

// Arrow assume_timezone: reads a naive timestamp column as wall-clock times in `tz` and returns the
// instants they name, tagged with that timezone. The unit and the sub-second part are unchanged.
// `tz` is an IANA name ("America/New_York") or a fixed offset ("+02:00").
//
// **CPU**, deliberately: the tz database is host data (Foundation's TimeZone) with no GPU-resident form,
// so the per-value offsets are computed on the host, sharded over DispatchQueue.concurrentPerform with a
// per-shard cache of the current offset's validity interval.
//
// `ambiguous` and `nonexistent` are 0 raise (Arrow's default), 1 earliest, 2 latest. A local time that
// occurs twice (a DST fall-back) picks the earlier / later instant; one that never occurs (a
// spring-forward gap) becomes the last instant before / the first instant after the gap.
int  am_assume_timezone(am_array* a, const char* tz, int ambiguous, int nonexistent, am_array** out);

// Arrow local_timestamp: the wall-clock time each instant names in the column's own timezone, as a naive
// timestamp of the same unit. A column with no timezone comes back unchanged. **CPU**, for the same
// reason as am_assume_timezone.
int  am_local_timestamp(am_array* a, am_array** out);

// ARROW:extension:name of an extension column, or NULL when the column is not an extension type. The
// pointer is owned by the library and stays valid for the process's lifetime.
const char* am_extension_name(am_array* a);
// ARROW:extension:metadata as raw bytes, or NULL when there is none; `out_len` receives the byte count.
// Same ownership as am_extension_name.
const char* am_extension_metadata(am_array* a, int64_t* out_len);
// The storage column of an extension array (the array itself for every other type), so the ordinary
// kernels can run on it without an export/import round trip.
int  am_extension_storage(am_array* a, am_array** out);
// Tags a column as the storage of an extension type, so am_export writes the two metadata keys.
// `metadata` may be NULL (with `metadata_len` 0).
int  am_extension_wrap(am_array* a, const char* name, const uint8_t* metadata, int64_t metadata_len,
                       am_array** out);

// A null column of `length` elements: every value null, no buffers. There is nothing to import for this
// type, so this is how a caller makes one.
int  am_null_array(int64_t length, am_array** out);

// ---------------------------------------------------------------------------------------------------
// Trigonometry, the remaining logical operators, float classification, the conditional transforms and
// a 64-bit value hash (all GPU).
//
// am_trig op numbering. Ops 0-11 are unary and need `b` to be NULL; op 12 is atan2 and needs `b`; ops
// 13-19 are Arrow's `_checked` twins, which raise a domain error instead of returning NaN.
//
//   op  name           op  name           op  name              op  name
//   --  -------------  --  -------------  --  ----------------  --  ----------------
//    0  sin             6  sinh           12  atan2(a, b)       16  asin_checked
//    1  cos             7  cosh           13  sin_checked       17  acos_checked
//    2  tan             8  tanh           14  cos_checked       18  acosh_checked
//    3  asin            9  asinh          15  tan_checked       19  atanh_checked
//    4  acos           10  acosh
//    5  atan           11  atanh
//
// Float columns only: an integer column is an error rather than being promoted to float64, as with
// sqrt / exp / ln. Nulls propagate (op 12 ANDs both validity bitmaps; every other op shares the
// input's zero-copy). atan2(y, x) takes this array as y and `b` as x, and follows the C99 special
// value table including the four ±0 and four ±infinity cases.
//
// float32 runs Metal's library functions, with the six hyperbolics written out from well-conditioned
// identities because Metal's own are inaccurate and get ±infinity wrong. float64 runs a **software
// binary64** implementation on the GPU (Metal has no double): Cody-Waite reduction against a 128-bit
// pi/2 plus Taylor series over the correctly rounded software adder and multiplier. Measured against
// the host libm over a million random arguments per function, the maximum error is 4 ulp for float32
// and 5 ulp for float64 (sin 2, cos 3, tan 5, asin 4, acos 4, atan 2, sinh 4, cosh 2, tanh 3,
// asinh 3, acosh 3, atanh 4). sin / cos / tan reduce exactly for |x| <= 2^45 * pi/2 ~ 5.5e13; beyond
// that the reduction degrades in step with the argument's own ulp, and |x| >= 2^62 returns NaN.
//
// The domains the `_checked` ops enforce, matching pyarrow: asin and acos need |x| <= 1, acosh needs
// x >= 1, atanh needs |x| < 1, and sin / cos / tan reject +/-infinity. A NaN input never raises and a
// null row is never inspected. The check is a device flag plus an atomic min of the offending index,
// read back after the dispatch, so a clean column costs nothing and the error names the first bad row.
int  am_trig(am_array* a, int op, am_array* b_or_null, am_array** out);

// Boolean logic beyond and / or / not and the Kleene pair. op: 0 xor, 1 and_not, 2 and_not_kleene.
// xor and and_not propagate nulls (output validity is the AND of both inputs). and_not_kleene is
// three-valued: a valid false on the left or a valid true on the right gives false even when the
// other side is null. Word-wise over the packed bitmaps, one thread per 32-bit output word.
int  am_logical(am_array* a, int op, am_array* b, am_array** out);

// Float classification. op: 0 is_nan, 1 is_finite, 2 is_inf. Defined on every numeric type, as in
// Arrow: an integer column answers the constant (is_finite everywhere true, the other two false).
// Nulls propagate — a null element gives a **null** predicate, matching pyarrow.compute.is_nan.
// The float tests read the raw bit pattern, so float64 needs no software binary64.
int  am_float_class(am_array* a, int op, am_array** out);

// Arrow fill_null_forward (forward != 0) / fill_null_backward. Every null takes the value of the
// nearest non-null element on the chosen side; nulls with none there stay null. One GPU max-scan over
// "index of the last valid row so far", then a gather. Primitive and boolean arrays; an array with no
// validity bitmap is returned unchanged.
int  am_fill_null_direction(am_array* a, int forward, am_array** out);

// Arrow case_when: `count` boolean conditions and `count` value columns, all of one length, the value
// columns all of one type, plus an optional default. Each row takes the value of the first condition
// that is true. **A null condition counts as false** (the row falls through), which is what Arrow
// does; a null in the chosen branch's values does make the output null. With else_or_null = NULL a
// row that matches no condition is null. Implemented as a right-to-left fold of the GPU if_else
// kernel, so k branches cost k passes.
int  am_case_when(am_array** conds, am_array** values, int64_t count, am_array* else_or_null, am_array** out);

// Arrow choose: out[i] = values[indices[i]][i]. Indices are int32, int64 or uint32; a null index
// gives a null output, and an index outside [0, count) is an error, as in Arrow. The range check is
// one GPU min and one GPU max over the index column (both skip nulls).
int  am_choose(am_array* indices, am_array** values, int64_t count, am_array** out);

// Arrow replace_with_mask: rows where `mask` is true take the next value from `replacements`, in
// order; rows where the mask is null become null; every other row keeps its own value. `replacements`
// must hold at least as many elements as the mask has valid trues (fewer is an error, a surplus is
// ignored, both as in pyarrow). One GPU sum-scan of the mask gives each selected row its position in
// `replacements`, then one gather. Primitive and boolean arrays.
int  am_replace_with_mask(am_array* a, am_array* mask, am_array* replacements, am_array** out);

// Arrow indices_nonzero: the **uint64** row numbers where the value is valid and not zero, in order.
// -0.0 counts as zero and every NaN counts as non-zero (the IEEE != 0 test), as in Arrow. iota put
// through the existing GPU stream compaction. The result never has nulls.
int  am_indices_nonzero(am_array* a, am_array** out);

// A 64-bit hash of primitive values (uint64 out). Arrow publishes no element-wise hash function, so
// this is an ArrowMetal extension, defined so it is reproducible from the specification alone:
//
//     hash64(v) = fmix64(normalise(v) ^ 0x9E3779B97F4A7C15)
//
// with fmix64 the MurmurHash3 128-bit finaliser (k ^= k >> 33; k *= 0xFF51AFD7ED558CCD;
// k ^= k >> 33; k *= 0xC4CEB9FE1A85EC53; k ^= k >> 33) and `normalise` the value's own bytes read as
// the unsigned type of the same width and zero-extended — for floats, after mapping -0.0 to +0.0 and
// every NaN to the canonical quiet NaN, so that **Arrow-equal values always hash equal**, which is
// what a hash join needs. Booleans normalise to 0 or 1. The golden-ratio seed keeps the value 0 from
// hashing to 0, which leaves 0 free for nulls: a null hashes to 0 **and stays null**, matching the
// existing am_str_unary(kind: 2) hash over utf8. Deterministic, and identical for a column and any
// slice of it.
int  am_hash64(am_array* a, am_array** out);

#ifdef __cplusplus
}
#endif
#endif
