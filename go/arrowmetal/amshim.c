#include "amshim.h"

#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static void* g_handle;
static char g_error[1024];

// One function pointer per symbol, typed from the real prototype in include/arrowmetal.h, so a
// signature change on the Swift side is a compile error here rather than a crash at run time.
#define AM_SYM(name) static __typeof__(&name) p_##name;
#define AM_BIND(name)                                                        \
    do {                                                                     \
        p_##name = (__typeof__(&name))dlsym(g_handle, #name);                \
        if (p_##name == 0) {                                                 \
            snprintf(g_error, sizeof(g_error),                               \
                     "symbol %s not found in %s", #name, path);              \
            dlclose(g_handle);                                               \
            g_handle = 0;                                                    \
            return 2;                                                        \
        }                                                                    \
    } while (0)

AM_SYM(am_version)
AM_SYM(am_device_name)
AM_SYM(am_last_error)
AM_SYM(am_import)
AM_SYM(am_export)
AM_SYM(am_release)
AM_SYM(am_length)
AM_SYM(am_null_count)
AM_SYM(am_format)
AM_SYM(am_slice)
AM_SYM(am_reduce)
AM_SYM(am_compare_scalar)
AM_SYM(am_compare_array)
AM_SYM(am_filter)
AM_SYM(am_take)
AM_SYM(am_argsort)
AM_SYM(am_sort)
AM_SYM(am_lexsort)
AM_SYM(am_group_by_keys)
AM_SYM(am_group_by_group_count)
AM_SYM(am_group_by_keys_result)
AM_SYM(am_group_by_ids)
AM_SYM(am_group_by_release)
AM_SYM(am_group_agg_ex)
AM_SYM(am_plan_source_create)
AM_SYM(am_plan_source_release)
AM_SYM(am_plan_run)
AM_SYM(am_plan_explain)
AM_SYM(am_plan_column_count)
AM_SYM(am_plan_row_count)
AM_SYM(am_plan_column_name)
AM_SYM(am_plan_column)
AM_SYM(am_plan_result_release)

int amshim_load(const char* path) {
    if (g_handle) return 0;
    g_error[0] = 0;
    g_handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    if (!g_handle) {
        const char* e = dlerror();
        snprintf(g_error, sizeof(g_error), "%s", e ? e : "dlopen failed");
        return 1;
    }
    AM_BIND(am_version);
    AM_BIND(am_device_name);
    AM_BIND(am_last_error);
    AM_BIND(am_import);
    AM_BIND(am_export);
    AM_BIND(am_release);
    AM_BIND(am_length);
    AM_BIND(am_null_count);
    AM_BIND(am_format);
    AM_BIND(am_slice);
    AM_BIND(am_reduce);
    AM_BIND(am_compare_scalar);
    AM_BIND(am_compare_array);
    AM_BIND(am_filter);
    AM_BIND(am_take);
    AM_BIND(am_argsort);
    AM_BIND(am_sort);
    AM_BIND(am_lexsort);
    AM_BIND(am_group_by_keys);
    AM_BIND(am_group_by_group_count);
    AM_BIND(am_group_by_keys_result);
    AM_BIND(am_group_by_ids);
    AM_BIND(am_group_by_release);
    AM_BIND(am_group_agg_ex);
    AM_BIND(am_plan_source_create);
    AM_BIND(am_plan_source_release);
    AM_BIND(am_plan_run);
    AM_BIND(am_plan_explain);
    AM_BIND(am_plan_column_count);
    AM_BIND(am_plan_row_count);
    AM_BIND(am_plan_column_name);
    AM_BIND(am_plan_column);
    AM_BIND(am_plan_result_release);
    return 0;
}

const char* amshim_error(void) { return g_error; }

long amshim_page_size(void) { return (long)getpagesize(); }

const char* amx_version(void) { return p_am_version(); }
const char* amx_device_name(void) { return p_am_device_name(); }
const char* amx_last_error(void) { return p_am_last_error(); }

int amx_import(const struct ArrowSchema* schema, struct ArrowArray* array, am_array** out) {
    return p_am_import(schema, array, out);
}
int amx_export(am_array* a, struct ArrowSchema* schema, struct ArrowArray* array) {
    return p_am_export(a, schema, array);
}
void    amx_release(am_array* a) { p_am_release(a); }
int64_t amx_length(am_array* a) { return p_am_length(a); }
int64_t amx_null_count(am_array* a) { return p_am_null_count(a); }
const char* amx_format(am_array* a) { return p_am_format(a); }
int amx_slice(am_array* a, int64_t offset, int64_t length, am_array** out) {
    return p_am_slice(a, offset, length, out);
}

int amx_reduce(am_array* a, int op, int64_t* out_i64, double* out_f64, int* out_kind, int* is_null) {
    return p_am_reduce(a, op, out_i64, out_f64, out_kind, is_null);
}

int amx_compare_scalar(am_array* a, int op, const void* scalar, am_array** out) {
    return p_am_compare_scalar(a, op, scalar, out);
}
int amx_compare_array(am_array* a, int op, am_array* b, am_array** out) {
    return p_am_compare_array(a, op, b, out);
}
int amx_filter(am_array* a, am_array* mask, am_array** out) { return p_am_filter(a, mask, out); }
int amx_take(am_array* a, am_array* indices, am_array** out) { return p_am_take(a, indices, out); }
int amx_argsort(am_array* a, int descending, am_array** out) { return p_am_argsort(a, descending, out); }
int amx_sort(am_array* a, int descending, am_array** out) { return p_am_sort(a, descending, out); }
int amx_lexsort(am_array** columns, const int* descending, int64_t count, am_array** out) {
    return p_am_lexsort(columns, descending, count, out);
}

int amx_group_by_keys(am_array** columns, int64_t count, am_groupby** out) {
    return p_am_group_by_keys(columns, count, out);
}
int64_t amx_group_by_group_count(am_groupby* gb) { return p_am_group_by_group_count(gb); }
int amx_group_by_keys_result(am_groupby* gb, int64_t i, am_array** out) {
    return p_am_group_by_keys_result(gb, i, out);
}
int amx_group_by_ids(am_groupby* gb, am_array** out) { return p_am_group_by_ids(gb, out); }
void amx_group_by_release(am_groupby* gb) { p_am_group_by_release(gb); }
int amx_group_agg_ex(am_groupby* gb, am_array* values, int op, double p1, am_array** out) {
    return p_am_group_agg_ex(gb, values, op, p1, out);
}

int amx_plan_source_create(const char* name, am_array** columns, const char** names,
                           int64_t n_columns, am_plan_source** out) {
    return p_am_plan_source_create(name, columns, names, n_columns, out);
}
void amx_plan_source_release(am_plan_source* s) { p_am_plan_source_release(s); }
int amx_plan_run(const char* plan_json, am_plan_source** sources, int64_t n_sources, int optimize,
                 am_plan_result** out) {
    return p_am_plan_run(plan_json, sources, n_sources, optimize, out);
}
const char* amx_plan_explain(const char* plan_json, am_plan_source** sources, int64_t n_sources,
                             int optimize) {
    return p_am_plan_explain(plan_json, sources, n_sources, optimize);
}
int64_t amx_plan_column_count(am_plan_result* r) { return p_am_plan_column_count(r); }
int64_t amx_plan_row_count(am_plan_result* r) { return p_am_plan_row_count(r); }
const char* amx_plan_column_name(am_plan_result* r, int64_t i) { return p_am_plan_column_name(r, i); }
int amx_plan_column(am_plan_result* r, int64_t i, am_array** out) { return p_am_plan_column(r, i, out); }
void amx_plan_result_release(am_plan_result* r) { p_am_plan_result_release(r); }
