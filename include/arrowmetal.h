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
// (first occurrence) and null where the element is null or absent.
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
// `modulo` is C remainder (the sign follows the dividend) and defines x % 0 as 0, as `divide` does;
// `power` uses repeated squaring on integers, wraps, and defines a negative exponent as 0. `power` and
// `modulo` are not implemented for float64. Ops 0-6 propagate nulls; 7 and 8 skip them, so a null on one
// side yields the other side's value and only two nulls make a null.
int  am_binary(am_array* a, int op, am_array* b /* or NULL */, const void* scalar /* or NULL */, am_array** out);

// am_cumulative op numbering: 0 cumulative_sum, 1 cumulative_min, 2 cumulative_max.
// Output is null exactly where the input is, and the running value carries across nulls unchanged.
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

// Remaining selection, sort, random and aggregate functions.
//
// am_inverse_permutation / am_scatter: for the i-th index the index-th output element is i, and
// scatter is that inverse permutation used as a take. The output has max_index + 1 elements, or the
// input's length when max_index is negative. A position no index names comes back **null**; when
// several positions name the same one the **last** wins (Arrow's rule, and deterministic here: the
// scatter takes an atomic maximum over the source positions, and a maximum does not depend on thread
// order). Null indices are skipped; an index outside [0, max_index] is an error. The inverse
// permutation is always int32 (Arrow's output_type option is not implemented); scatter accepts any
// column type, nested ones included, because take already turns a null index into a null row.
int  am_inverse_permutation(am_array* a, int64_t max_index, am_array** out);
int  am_scatter(am_array* values, am_array* indices, int64_t max_index, am_array** out);

// am_winsorize: values below the lower quantile take the lower quantile's value and values above the
// upper one take the upper quantile's value. The limits are Arrow's *nearest* quantiles, not
// interpolated ones: with m non-null, non-NaN values sorted ascending, a limit q picks
// sorted[round(q * (m - 1))] with the halfway case going to the even index. Nulls stay null and NaNs
// pass through unchanged (they take part in neither the limits nor the comparison). Requires
// 0 <= lower <= upper <= 1. One GPU sort plus one clamp kernel.
int  am_winsorize(am_array* a, double lower_limit, double upper_limit, am_array** out);

// am_rank: op 0 rank_quantile (float64), op 1 rank_normal (float64), op 2 rank_normal (float32).
//
// rank_quantile gives each row (average 1-based rank of its tie group - 0.5) / n, pyarrow's
// definition, computed on the GPU as (s + e) / (2n) over the tie group's sorted positions [s, e)
// with the correctly rounded software binary64 divide. Nulls sort last and form one tie group
// (pyarrow's default null_placement = "at_end") and NaN is one value after +inf; no result is null.
// Arrow's sort_keys / null_placement options are not implemented.
//
// rank_normal is the normal percent-point function of that quantile. Op 1 evaluates it on the host
// with Wichura's AS 241 (about 1e-16 relative); op 2 evaluates it on the GPU in float32 with
// Acklam's rational approximation plus one Halley refinement, which lands within about 1e-6 of the
// float64 answer. Metal has no `double` and the software binary64 has no log/exp/erfc, so the
// float64 inverse CDF is the one part of this that runs on the CPU.
int  am_rank(am_array* a, int op, am_array** out);

// am_random: `count` uniform float64 values in [0, 1), generated on the GPU with Philox4x32-10
// (Salmon, Moraes, Dror & Shaw, SC'11) keyed by `seed`. Element i comes from the counter (i,0,0,0),
// so the stream depends only on the seed — not on the device or the launch geometry — and the top
// 53 bits of each draw become a multiple of 2^-53, so no value is ever 1.0. The stream is
// ArrowMetal's own: it does not reproduce Arrow C++'s pcg32_fast numbers for the same seed.
int  am_random(int64_t count, uint64_t seed, am_array** out);

// am_true_unless_null: true for every valid row, null for every null one. The values bitmap is a
// host memset and the validity bitmap is shared with the input with no copy, so no kernel runs.
// Union and run-end encoded columns are refused: neither has a top-level validity bitmap to share.
int  am_true_unless_null(am_array* a, am_array** out);

// am_count_all: the number of rows, valid or not. -1 for a NULL handle.
int64_t am_count_all(am_array* a);

// am_first_last: a one-row struct ("+s") with fields `first` and `last` of the column's own type.
// With skip_nulls != 0 the first and last non-null values are used (both null when no row is valid);
// with skip_nulls == 0 the first and last rows are taken as they are.
int  am_first_last(am_array* a, int skip_nulls, am_array** out);

// am_str_extra: op 0 utf8_swapcase, op 1 utf8_zero_fill (p1 = width in code points, arg = the pad
// character, defaulting to "0"). Both are two-pass GPU transforms.
//
// utf8_swapcase covers the same blocks utf8_upper / utf8_lower do — Basic Latin, Latin-1 Supplement
// and Latin Extended-A, including the length-changing pairs. Everything above U+017F, and U+00DF
// (which Arrow swaps to U+1E9E), passes through unchanged rather than being mangled.
//
// utf8_zero_fill left-pads to `width` code points, inserting the padding after a leading + or -, and
// leaves a string already at or over `width` alone. The content need not be numeric.
int  am_str_extra(am_array* a, int op, const char* arg, int64_t arg_len, int64_t p1, am_array** out);

// am_pivot_wider: one struct row with a field per entry of key_names, each of the value column's
// type. A key that never appears, or appears only with a null value, gives a null field; a key that
// carries more than one non-null value is an error, as in Arrow. raise_unexpected != 0 is Arrow's
// unexpected_key_behavior = "raise". The key column may be utf8, binary, dictionary or any integer
// type (integers match by their decimal rendering). This one runs on the host: the output is one row
// wide however long the input is.
int  am_pivot_wider(am_array* keys, am_array* values, const char* const* key_names, int64_t n_keys,
                    int raise_unexpected, am_array** out);

// am_make_struct: composes equal-length columns into one struct-typed column ("+s"), with the given
// field names. Metadata only — the children are shared, nothing is copied and no kernel runs.
int  am_make_struct(am_array** arrays, const char* const* names, int64_t count, am_array** out);

// am_unique / am_value_counts / am_partition_nth_indices.
//
// unique returns the distinct non-null values; ArrowMetal orders them **ascending** (one GPU sort
// plus a run scan) where Arrow orders them by first appearance. value_counts is the same pass,
// returned as a struct ("+s") with fields `values` and `counts` (int64), in the same ascending
// order. partition_nth_indices answers with the full stable argsort, which satisfies Arrow's
// contract (the n smallest first) at the cost of one radix sort.
int  am_unique(am_array* a, am_array** out);
int  am_value_counts(am_array* a, am_array** out);
int  am_partition_nth_indices(am_array* a, int64_t n, am_array** out);

#ifdef __cplusplus
}
#endif
#endif
