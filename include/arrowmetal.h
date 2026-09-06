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
const char* am_format(am_array* a);          // Arrow format string: c C s S i I l L f g b

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

#ifdef __cplusplus
}
#endif
#endif
