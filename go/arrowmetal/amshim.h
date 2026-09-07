// Thin dlopen shim over the ArrowMetal C ABI.
//
// The Go binding does not link against libArrowMetalC at build time: the dylib is a build product of
// the Swift package, not something `go get` can fetch, so the location has to be a runtime decision.
// amshim_load() dlopens it and resolves exactly the symbols this binding uses; every amx_* function
// below forwards to the resolved pointer. Calling one before a successful load is a programming error
// in the Go layer, which gates every call behind the package-level load result.
#ifndef ARROWMETAL_GO_SHIM_H
#define ARROWMETAL_GO_SHIM_H

#include <stdint.h>
#include "include/arrowmetal.h"

#ifdef __cplusplus
extern "C" {
#endif

// Loads `path` and resolves the symbols. Returns 0 on success, non-zero on failure; on failure
// amshim_error() has the reason (dlerror text, or the name of the first missing symbol).
int amshim_load(const char* path);
const char* amshim_error(void);
// The page size the Swift side uses to decide whether an imported buffer can be borrowed instead of
// copied (`getpagesize()`); exposed so the Go layer can report the copy rule without guessing.
long amshim_page_size(void);

// --- info -------------------------------------------------------------------------------------
const char* amx_version(void);
const char* amx_device_name(void);
const char* amx_last_error(void);

// --- lifecycle and interop --------------------------------------------------------------------
int     amx_import(const struct ArrowSchema* schema, struct ArrowArray* array, am_array** out);
int     amx_export(am_array* a, struct ArrowSchema* schema, struct ArrowArray* array);
void    amx_release(am_array* a);
int64_t amx_length(am_array* a);
int64_t amx_null_count(am_array* a);
const char* amx_format(am_array* a);
int     amx_slice(am_array* a, int64_t offset, int64_t length, am_array** out);

// --- reductions ---------------------------------------------------------------------------------
int amx_reduce(am_array* a, int op, int64_t* out_i64, double* out_f64, int* out_kind, int* is_null);

// --- element-wise and selection -----------------------------------------------------------------
int amx_compare_scalar(am_array* a, int op, const void* scalar, am_array** out);
int amx_compare_array(am_array* a, int op, am_array* b, am_array** out);
int amx_filter(am_array* a, am_array* mask, am_array** out);
int amx_take(am_array* a, am_array* indices, am_array** out);
int amx_argsort(am_array* a, int descending, am_array** out);
int amx_sort(am_array* a, int descending, am_array** out);
int amx_lexsort(am_array** columns, const int* descending, int64_t count, am_array** out);

// --- group by -------------------------------------------------------------------------------------
int     amx_group_by_keys(am_array** columns, int64_t count, am_groupby** out);
int64_t amx_group_by_group_count(am_groupby* gb);
int     amx_group_by_keys_result(am_groupby* gb, int64_t i, am_array** out);
int     amx_group_by_ids(am_groupby* gb, am_array** out);
void    amx_group_by_release(am_groupby* gb);
int     amx_group_agg_ex(am_groupby* gb, am_array* values, int op, double p1, am_array** out);

// --- the JSON plan runner ---------------------------------------------------------------------------
int         amx_plan_source_create(const char* name, am_array** columns, const char** names,
                                   int64_t n_columns, am_plan_source** out);
void        amx_plan_source_release(am_plan_source* s);
int         amx_plan_run(const char* plan_json, am_plan_source** sources, int64_t n_sources,
                         int optimize, am_plan_result** out);
const char* amx_plan_explain(const char* plan_json, am_plan_source** sources, int64_t n_sources,
                             int optimize);
int64_t     amx_plan_column_count(am_plan_result* r);
int64_t     amx_plan_row_count(am_plan_result* r);
const char* amx_plan_column_name(am_plan_result* r, int64_t i);
int         amx_plan_column(am_plan_result* r, int64_t i, am_array** out);
void        amx_plan_result_release(am_plan_result* r);

#ifdef __cplusplus
}
#endif
#endif  // ARROWMETAL_GO_SHIM_H
