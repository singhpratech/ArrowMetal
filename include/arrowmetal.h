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
const char* am_format(am_array* a);          // Arrow format string: c C s S i I l L f g b u z t...

// Reductions. out_kind: 0 = int64 in out_i64, 1 = uint64 in out_u64 (same slot), 2 = float64 in out_f64.
// op: 0 sum, 1 min, 2 max, 3 mean. *is_null is set when there is no valid value.
int  am_reduce(am_array* a, int op, int64_t* out_i64, double* out_f64, int* out_kind, int* is_null);

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

// Batching: between begin and end, every call on this thread appends to one GPU command buffer. The GPU runs
// once at end (or at the first call that must read a result, such as am_reduce or am_export).
int  am_batch_begin(void);
int  am_batch_end(void);

#ifdef __cplusplus
}
#endif
#endif
