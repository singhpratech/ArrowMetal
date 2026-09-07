/* ArrowMetal R binding: a thin C shim over include/arrowmetal.h.
 *
 * The Swift core ships as one dylib (libArrowMetalC.dylib) which this shim opens with dlopen at
 * package load and resolves symbol by symbol. Nothing here links against the dylib, so the R
 * package installs on a machine that does not have it and fails with a message when it is used.
 *
 * arrowmetal.h and arrow_abi.h in this directory are verbatim copies of the repository's
 * include/ headers (tests/testthat/test-header-copy.R checks that when the repo is present).
 * Every entry point is called through a pointer whose type comes from the real prototype with
 * __typeof__, so a signature change in the header is a compile error here rather than a crash.
 */

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>

#include "arrowmetal.h"

/* ------------------------------------------------------------------ symbol table */

#define AM_FNS(X)                                                                   \
  X(am_version) X(am_device_name) X(am_last_error)                                   \
  X(am_import) X(am_export) X(am_release)                                            \
  X(am_length) X(am_null_count) X(am_format)                                         \
  X(am_child_count) X(am_child)                                                      \
  X(am_reduce) X(am_compare_scalar) X(am_compare_array)                              \
  X(am_filter) X(am_take) X(am_slice)                                                \
  X(am_argsort) X(am_sort)                                                           \
  X(am_group_by_keys) X(am_group_by_group_count) X(am_group_by_keys_result)          \
  X(am_group_by_ids) X(am_group_by_release) X(am_group_agg_ex)                       \
  X(am_plan_source_create) X(am_plan_source_release) X(am_plan_run)                  \
  X(am_plan_explain) X(am_plan_column_count) X(am_plan_row_count)                    \
  X(am_plan_column_name) X(am_plan_column) X(am_plan_result_release)

#define AM_DECL(f) static __typeof__(f) *p_##f = NULL;
AM_FNS(AM_DECL)
#undef AM_DECL

static void *g_lib = NULL;
static char g_lib_path[4096] = "";

/* ------------------------------------------------------------------ helpers */

static void am_stop(void) {
  const char *m = p_am_last_error ? p_am_last_error() : NULL;
  Rf_error("ArrowMetal: %s", (m && *m) ? m : "unknown error");
}

static void require_lib(void) {
  if (g_lib == NULL) Rf_error("ArrowMetal: the library is not loaded (see arrowmetal::am_lib_path())");
}

/* --- am_array handles ------------------------------------------------------- */

static void finalize_am_array(SEXP xp) {
  am_array *h = (am_array *)R_ExternalPtrAddr(xp);
  if (h != NULL && p_am_release != NULL) p_am_release(h);
  R_ClearExternalPtr(xp);
}

static SEXP wrap_am_array(am_array *h) {
  SEXP xp = PROTECT(R_MakeExternalPtr(h, Rf_install("am_array"), R_NilValue));
  R_RegisterCFinalizerEx(xp, finalize_am_array, TRUE);
  Rf_setAttrib(xp, R_ClassSymbol, Rf_mkString("am_array"));
  UNPROTECT(1);
  return xp;
}

static am_array *unwrap_am_array(SEXP xp) {
  if (TYPEOF(xp) != EXTPTRSXP || R_ExternalPtrTag(xp) != Rf_install("am_array"))
    Rf_error("ArrowMetal: expected an am_array handle");
  am_array *h = (am_array *)R_ExternalPtrAddr(xp);
  if (h == NULL) Rf_error("ArrowMetal: this am_array handle has already been released");
  return h;
}

/* --- am_groupby handles ------------------------------------------------------ */

static void finalize_groupby(SEXP xp) {
  am_groupby *g = (am_groupby *)R_ExternalPtrAddr(xp);
  if (g != NULL && p_am_group_by_release != NULL) p_am_group_by_release(g);
  R_ClearExternalPtr(xp);
}

static am_groupby *unwrap_groupby(SEXP xp) {
  if (TYPEOF(xp) != EXTPTRSXP || R_ExternalPtrTag(xp) != Rf_install("am_groupby"))
    Rf_error("ArrowMetal: expected an am_groupby handle");
  am_groupby *g = (am_groupby *)R_ExternalPtrAddr(xp);
  if (g == NULL) Rf_error("ArrowMetal: this group-by handle has already been released");
  return g;
}

/* --- plan handles ------------------------------------------------------------ */

static void finalize_plan_source(SEXP xp) {
  am_plan_source *s = (am_plan_source *)R_ExternalPtrAddr(xp);
  if (s != NULL && p_am_plan_source_release != NULL) p_am_plan_source_release(s);
  R_ClearExternalPtr(xp);
}

static void finalize_plan_result(SEXP xp) {
  am_plan_result *r = (am_plan_result *)R_ExternalPtrAddr(xp);
  if (r != NULL && p_am_plan_result_release != NULL) p_am_plan_result_release(r);
  R_ClearExternalPtr(xp);
}

static void *unwrap_tagged(SEXP xp, const char *tag) {
  if (TYPEOF(xp) != EXTPTRSXP || R_ExternalPtrTag(xp) != Rf_install(tag))
    Rf_error("ArrowMetal: expected a %s handle", tag);
  void *p = R_ExternalPtrAddr(xp);
  if (p == NULL) Rf_error("ArrowMetal: this %s handle has already been released", tag);
  return p;
}

/* --- ArrowArray / ArrowSchema carriers --------------------------------------- */
/* We allocate the C Data Interface structs ourselves rather than reaching into arrow's
 * unexported allocate_arrow_array(); arrow's ExportArray/ImportArray accept any external
 * pointer to the struct. The finalizer calls the producer's release callback if the struct
 * still owns anything, so an error between export and import cannot leak Arrow buffers. */

static void finalize_arrow_array(SEXP xp) {
  struct ArrowArray *a = (struct ArrowArray *)R_ExternalPtrAddr(xp);
  if (a != NULL) {
    if (a->release != NULL) a->release(a);
    free(a);
  }
  R_ClearExternalPtr(xp);
}

static void finalize_arrow_schema(SEXP xp) {
  struct ArrowSchema *s = (struct ArrowSchema *)R_ExternalPtrAddr(xp);
  if (s != NULL) {
    if (s->release != NULL) s->release(s);
    free(s);
  }
  R_ClearExternalPtr(xp);
}

static struct ArrowArray *unwrap_arrow_array(SEXP xp) {
  if (TYPEOF(xp) != EXTPTRSXP || R_ExternalPtrTag(xp) != Rf_install("ArrowArray"))
    Rf_error("ArrowMetal: expected an ArrowArray pointer");
  struct ArrowArray *a = (struct ArrowArray *)R_ExternalPtrAddr(xp);
  if (a == NULL) Rf_error("ArrowMetal: this ArrowArray pointer has been freed");
  return a;
}

static struct ArrowSchema *unwrap_arrow_schema(SEXP xp) {
  if (TYPEOF(xp) != EXTPTRSXP || R_ExternalPtrTag(xp) != Rf_install("ArrowSchema"))
    Rf_error("ArrowMetal: expected an ArrowSchema pointer");
  struct ArrowSchema *s = (struct ArrowSchema *)R_ExternalPtrAddr(xp);
  if (s == NULL) Rf_error("ArrowMetal: this ArrowSchema pointer has been freed");
  return s;
}

/* ------------------------------------------------------------------ loading */

SEXP C_am_load(SEXP path) {
  if (g_lib != NULL) return Rf_mkString(g_lib_path);
  const char *p = CHAR(STRING_ELT(path, 0));
  void *h = dlopen(p, RTLD_LAZY | RTLD_LOCAL);
  if (h == NULL) {
    const char *e = dlerror();
    return Rf_mkString(e ? e : "dlopen failed");
  }
#define AM_RESOLVE(f)                                                      \
  {                                                                        \
    void *s = dlsym(h, #f);                                                \
    if (s == NULL) {                                                       \
      dlclose(h);                                                          \
      return Rf_mkString("missing symbol " #f);                            \
    }                                                                      \
    p_##f = (__typeof__(f) *)s;                                            \
  }
  AM_FNS(AM_RESOLVE)
#undef AM_RESOLVE
  g_lib = h;
  snprintf(g_lib_path, sizeof(g_lib_path), "%s", p);
  return R_NilValue;
}

SEXP C_am_lib_path(void) {
  return g_lib == NULL ? R_NilValue : Rf_mkString(g_lib_path);
}

SEXP C_am_version(void) {
  require_lib();
  return Rf_mkString(p_am_version());
}

SEXP C_am_device_name(void) {
  require_lib();
  return Rf_mkString(p_am_device_name());
}

/* ------------------------------------------------------------------ interop */

SEXP C_alloc_arrow_array(void) {
  struct ArrowArray *a = (struct ArrowArray *)calloc(1, sizeof(struct ArrowArray));
  if (a == NULL) Rf_error("ArrowMetal: out of memory");
  SEXP xp = PROTECT(R_MakeExternalPtr(a, Rf_install("ArrowArray"), R_NilValue));
  R_RegisterCFinalizerEx(xp, finalize_arrow_array, TRUE);
  UNPROTECT(1);
  return xp;
}

SEXP C_alloc_arrow_schema(void) {
  struct ArrowSchema *s = (struct ArrowSchema *)calloc(1, sizeof(struct ArrowSchema));
  if (s == NULL) Rf_error("ArrowMetal: out of memory");
  SEXP xp = PROTECT(R_MakeExternalPtr(s, Rf_install("ArrowSchema"), R_NilValue));
  R_RegisterCFinalizerEx(xp, finalize_arrow_schema, TRUE);
  UNPROTECT(1);
  return xp;
}

/* Imports a filled-in (schema, array) pair. am_import moves the array; the schema stays ours
 * and is released here, exactly as the Python binding does. */
SEXP C_am_import(SEXP schema_xp, SEXP array_xp) {
  require_lib();
  struct ArrowSchema *s = unwrap_arrow_schema(schema_xp);
  struct ArrowArray *a = unwrap_arrow_array(array_xp);
  if (s->release == NULL || a->release == NULL)
    Rf_error("ArrowMetal: the (schema, array) pair was not filled in by a producer");
  am_array *out = NULL;
  int rc = p_am_import(s, a, &out);
  /* am_import moves the array (it nulls the source's release, see CInterop.swift) but the schema
   * stays the caller's, so it is released here whether or not the import succeeded. */
  if (s->release != NULL) s->release(s);
  if (rc != 0) am_stop();
  return wrap_am_array(out);
}

SEXP C_am_export(SEXP h, SEXP schema_xp, SEXP array_xp) {
  require_lib();
  am_array *x = unwrap_am_array(h);
  struct ArrowSchema *s = unwrap_arrow_schema(schema_xp);
  struct ArrowArray *a = unwrap_arrow_array(array_xp);
  if (p_am_export(x, s, a) != 0) am_stop();
  return R_NilValue;
}

/* Buffer addresses of a filled-in ArrowArray, so R can measure the producer's alignment. */
SEXP C_arrow_array_buffers(SEXP array_xp) {
  struct ArrowArray *a = unwrap_arrow_array(array_xp);
  R_xlen_t n = (R_xlen_t)a->n_buffers;
  SEXP out = PROTECT(Rf_allocVector(REALSXP, n));
  for (R_xlen_t i = 0; i < n; i++)
    REAL(out)[i] = (double)(uintptr_t)a->buffers[i]; /* NULL validity buffer shows as 0 */
  UNPROTECT(1);
  return out;
}

SEXP C_page_size(void) { return Rf_ScalarInteger((int)getpagesize()); }

/* ------------------------------------------------------------------ metadata */

SEXP C_am_length(SEXP h) {
  require_lib();
  return Rf_ScalarReal((double)p_am_length(unwrap_am_array(h)));
}

SEXP C_am_null_count(SEXP h) {
  require_lib();
  return Rf_ScalarReal((double)p_am_null_count(unwrap_am_array(h)));
}

SEXP C_am_format(SEXP h) {
  require_lib();
  const char *f = p_am_format(unwrap_am_array(h));
  return Rf_mkString(f ? f : "");
}

/* ------------------------------------------------------------------ scalars */

/* Packs an R scalar into the array's element type. `fmt` is the Arrow format string. */
static void pack_scalar(const char *fmt, SEXP v, unsigned char *buf) {
  double d;
  if (Rf_inherits(v, "integer64")) {
    int64_t i64;
    memcpy(&i64, &REAL(v)[0], 8);
    d = (double)i64;
    if (fmt[0] == 'l' || fmt[0] == 'L') {
      memcpy(buf, &i64, 8);
      return;
    }
  } else if (TYPEOF(v) == LGLSXP) {
    d = (double)LOGICAL(v)[0];
  } else {
    d = Rf_asReal(v);
  }
  if (ISNA(d) || ISNAN(d)) {
    if (fmt[0] != 'f' && fmt[0] != 'g')
      Rf_error("ArrowMetal: NA is not a valid scalar for an array of type '%s'", fmt);
  }
  switch (fmt[0]) {
    case 'b': { uint8_t x = d != 0.0; memcpy(buf, &x, 1); return; }
    case 'c': { int8_t x = (int8_t)d; memcpy(buf, &x, 1); return; }
    case 'C': { uint8_t x = (uint8_t)d; memcpy(buf, &x, 1); return; }
    case 's': { int16_t x = (int16_t)d; memcpy(buf, &x, 2); return; }
    case 'S': { uint16_t x = (uint16_t)d; memcpy(buf, &x, 2); return; }
    case 'i': { int32_t x = (int32_t)d; memcpy(buf, &x, 4); return; }
    case 'I': { uint32_t x = (uint32_t)d; memcpy(buf, &x, 4); return; }
    case 'l': { int64_t x = (int64_t)d; memcpy(buf, &x, 8); return; }
    case 'L': { uint64_t x = (uint64_t)d; memcpy(buf, &x, 8); return; }
    case 'f': { float x = (float)d; memcpy(buf, &x, 4); return; }
    case 'g': { memcpy(buf, &d, 8); return; }
    default:
      Rf_error("ArrowMetal: scalars of Arrow type '%s' are not supported from R", fmt);
  }
}

/* ------------------------------------------------------------------ compute */

SEXP C_am_reduce(SEXP h, SEXP op) {
  require_lib();
  am_array *x = unwrap_am_array(h);
  int64_t i64 = 0;
  double f64 = 0;
  int kind = 0, is_null = 0;
  if (p_am_reduce(x, Rf_asInteger(op), &i64, &f64, &kind, &is_null) != 0) am_stop();
  SEXP out = PROTECT(Rf_allocVector(VECSXP, 4));
  SET_VECTOR_ELT(out, 0, Rf_ScalarInteger(kind));
  SET_VECTOR_ELT(out, 1, Rf_ScalarLogical(is_null));
  /* Slot 2 carries the int64/uint64 bit pattern as a bit64-compatible double; slot 3 the float64
   * and, for integer kinds, the same value rounded to a double so plain R code has a number. */
  {
    double bits;
    memcpy(&bits, &i64, 8);
    SET_VECTOR_ELT(out, 2, Rf_ScalarReal(bits));
  }
  if (kind == 2) {
    SET_VECTOR_ELT(out, 3, Rf_ScalarReal(f64));
  } else if (kind == 1) {
    uint64_t u;
    memcpy(&u, &i64, 8);
    SET_VECTOR_ELT(out, 3, Rf_ScalarReal((double)u));
  } else {
    SET_VECTOR_ELT(out, 3, Rf_ScalarReal((double)i64));
  }
  UNPROTECT(1);
  return out;
}

SEXP C_am_compare_scalar(SEXP h, SEXP op, SEXP scalar, SEXP fmt) {
  require_lib();
  am_array *x = unwrap_am_array(h);
  unsigned char buf[16];
  memset(buf, 0, sizeof(buf));
  pack_scalar(CHAR(STRING_ELT(fmt, 0)), scalar, buf);
  am_array *out = NULL;
  if (p_am_compare_scalar(x, Rf_asInteger(op), buf, &out) != 0) am_stop();
  return wrap_am_array(out);
}

SEXP C_am_compare_array(SEXP h, SEXP op, SEXP other) {
  require_lib();
  am_array *out = NULL;
  if (p_am_compare_array(unwrap_am_array(h), Rf_asInteger(op), unwrap_am_array(other), &out) != 0)
    am_stop();
  return wrap_am_array(out);
}

SEXP C_am_filter(SEXP h, SEXP mask) {
  require_lib();
  am_array *out = NULL;
  if (p_am_filter(unwrap_am_array(h), unwrap_am_array(mask), &out) != 0) am_stop();
  return wrap_am_array(out);
}

SEXP C_am_take(SEXP h, SEXP idx) {
  require_lib();
  am_array *out = NULL;
  if (p_am_take(unwrap_am_array(h), unwrap_am_array(idx), &out) != 0) am_stop();
  return wrap_am_array(out);
}

SEXP C_am_slice(SEXP h, SEXP offset, SEXP length) {
  require_lib();
  am_array *out = NULL;
  if (p_am_slice(unwrap_am_array(h), (int64_t)Rf_asReal(offset), (int64_t)Rf_asReal(length), &out) != 0)
    am_stop();
  return wrap_am_array(out);
}

SEXP C_am_argsort(SEXP h, SEXP descending) {
  require_lib();
  am_array *out = NULL;
  if (p_am_argsort(unwrap_am_array(h), Rf_asLogical(descending) == TRUE, &out) != 0) am_stop();
  return wrap_am_array(out);
}

SEXP C_am_sort(SEXP h, SEXP descending) {
  require_lib();
  am_array *out = NULL;
  if (p_am_sort(unwrap_am_array(h), Rf_asLogical(descending) == TRUE, &out) != 0) am_stop();
  return wrap_am_array(out);
}

SEXP C_am_child_count(SEXP h) {
  require_lib();
  return Rf_ScalarReal((double)p_am_child_count(unwrap_am_array(h)));
}

SEXP C_am_child(SEXP h, SEXP i) {
  require_lib();
  am_array *out = NULL;
  if (p_am_child(unwrap_am_array(h), (int64_t)Rf_asReal(i), &out) != 0) am_stop();
  return wrap_am_array(out);
}

/* ------------------------------------------------------------------ group by */

SEXP C_am_group_by_keys(SEXP handles) {
  require_lib();
  R_xlen_t n = Rf_xlength(handles);
  if (n < 1) Rf_error("ArrowMetal: at least one key column is required");
  am_array **cols = (am_array **)R_alloc(n, sizeof(am_array *));
  for (R_xlen_t i = 0; i < n; i++) cols[i] = unwrap_am_array(VECTOR_ELT(handles, i));
  am_groupby *gb = NULL;
  if (p_am_group_by_keys(cols, (int64_t)n, &gb) != 0) am_stop();
  SEXP xp = PROTECT(R_MakeExternalPtr(gb, Rf_install("am_groupby"), R_NilValue));
  R_RegisterCFinalizerEx(xp, finalize_groupby, TRUE);
  Rf_setAttrib(xp, R_ClassSymbol, Rf_mkString("am_groupby_handle"));
  UNPROTECT(1);
  return xp;
}

SEXP C_am_group_by_group_count(SEXP gb) {
  require_lib();
  return Rf_ScalarReal((double)p_am_group_by_group_count(unwrap_groupby(gb)));
}

SEXP C_am_group_by_keys_result(SEXP gb, SEXP i) {
  require_lib();
  am_array *out = NULL;
  if (p_am_group_by_keys_result(unwrap_groupby(gb), (int64_t)Rf_asReal(i), &out) != 0) am_stop();
  return wrap_am_array(out);
}

SEXP C_am_group_by_ids(SEXP gb) {
  require_lib();
  am_array *out = NULL;
  if (p_am_group_by_ids(unwrap_groupby(gb), &out) != 0) am_stop();
  return wrap_am_array(out);
}

SEXP C_am_group_agg(SEXP gb, SEXP values, SEXP op, SEXP p1) {
  require_lib();
  am_array *v = Rf_isNull(values) ? NULL : unwrap_am_array(values);
  am_array *out = NULL;
  if (p_am_group_agg_ex(unwrap_groupby(gb), v, Rf_asInteger(op), Rf_asReal(p1), &out) != 0) am_stop();
  return wrap_am_array(out);
}

/* ------------------------------------------------------------------ plan runner */

SEXP C_am_plan_source_create(SEXP name, SEXP handles, SEXP names) {
  require_lib();
  R_xlen_t n = Rf_xlength(handles);
  if (n < 1) Rf_error("ArrowMetal: a plan source needs at least one column");
  am_array **cols = (am_array **)R_alloc(n, sizeof(am_array *));
  const char **cn = (const char **)R_alloc(n, sizeof(char *));
  for (R_xlen_t i = 0; i < n; i++) {
    cols[i] = unwrap_am_array(VECTOR_ELT(handles, i));
    cn[i] = CHAR(STRING_ELT(names, i));
  }
  am_plan_source *s = NULL;
  if (p_am_plan_source_create(CHAR(STRING_ELT(name, 0)), cols, cn, (int64_t)n, &s) != 0) am_stop();
  SEXP xp = PROTECT(R_MakeExternalPtr(s, Rf_install("am_plan_source"), R_NilValue));
  R_RegisterCFinalizerEx(xp, finalize_plan_source, TRUE);
  Rf_setAttrib(xp, R_ClassSymbol, Rf_mkString("am_plan_source"));
  UNPROTECT(1);
  return xp;
}

static am_plan_source **collect_sources(SEXP sources, R_xlen_t *n_out) {
  R_xlen_t n = Rf_xlength(sources);
  am_plan_source **ss = (am_plan_source **)R_alloc(n < 1 ? 1 : n, sizeof(am_plan_source *));
  for (R_xlen_t i = 0; i < n; i++)
    ss[i] = (am_plan_source *)unwrap_tagged(VECTOR_ELT(sources, i), "am_plan_source");
  *n_out = n;
  return ss;
}

SEXP C_am_plan_run(SEXP plan_json, SEXP sources, SEXP optimize) {
  require_lib();
  R_xlen_t n;
  am_plan_source **ss = collect_sources(sources, &n);
  am_plan_result *r = NULL;
  if (p_am_plan_run(CHAR(STRING_ELT(plan_json, 0)), ss, (int64_t)n,
                    Rf_asLogical(optimize) == TRUE, &r) != 0)
    am_stop();
  SEXP xp = PROTECT(R_MakeExternalPtr(r, Rf_install("am_plan_result"), R_NilValue));
  R_RegisterCFinalizerEx(xp, finalize_plan_result, TRUE);
  Rf_setAttrib(xp, R_ClassSymbol, Rf_mkString("am_plan_result"));
  UNPROTECT(1);
  return xp;
}

SEXP C_am_plan_explain(SEXP plan_json, SEXP sources, SEXP optimize) {
  require_lib();
  R_xlen_t n;
  am_plan_source **ss = collect_sources(sources, &n);
  const char *t = p_am_plan_explain(CHAR(STRING_ELT(plan_json, 0)), ss, (int64_t)n,
                                    Rf_asLogical(optimize) == TRUE);
  if (t == NULL) am_stop();
  return Rf_mkString(t);
}

SEXP C_am_plan_result_dim(SEXP r) {
  require_lib();
  am_plan_result *pr = (am_plan_result *)unwrap_tagged(r, "am_plan_result");
  SEXP out = PROTECT(Rf_allocVector(REALSXP, 2));
  REAL(out)[0] = (double)p_am_plan_row_count(pr);
  REAL(out)[1] = (double)p_am_plan_column_count(pr);
  UNPROTECT(1);
  return out;
}

SEXP C_am_plan_column_names(SEXP r) {
  require_lib();
  am_plan_result *pr = (am_plan_result *)unwrap_tagged(r, "am_plan_result");
  int64_t n = p_am_plan_column_count(pr);
  SEXP out = PROTECT(Rf_allocVector(STRSXP, (R_xlen_t)n));
  for (int64_t i = 0; i < n; i++) {
    const char *nm = p_am_plan_column_name(pr, i);
    SET_STRING_ELT(out, (R_xlen_t)i, Rf_mkChar(nm ? nm : ""));
  }
  UNPROTECT(1);
  return out;
}

SEXP C_am_plan_column(SEXP r, SEXP i) {
  require_lib();
  am_plan_result *pr = (am_plan_result *)unwrap_tagged(r, "am_plan_result");
  am_array *out = NULL;
  if (p_am_plan_column(pr, (int64_t)Rf_asReal(i), &out) != 0) am_stop();
  return wrap_am_array(out);
}

/* ------------------------------------------------------------------ registration */

static const R_CallMethodDef CallEntries[] = {
    {"C_am_load", (DL_FUNC)&C_am_load, 1},
    {"C_am_lib_path", (DL_FUNC)&C_am_lib_path, 0},
    {"C_am_version", (DL_FUNC)&C_am_version, 0},
    {"C_am_device_name", (DL_FUNC)&C_am_device_name, 0},
    {"C_alloc_arrow_array", (DL_FUNC)&C_alloc_arrow_array, 0},
    {"C_alloc_arrow_schema", (DL_FUNC)&C_alloc_arrow_schema, 0},
    {"C_am_import", (DL_FUNC)&C_am_import, 2},
    {"C_am_export", (DL_FUNC)&C_am_export, 3},
    {"C_arrow_array_buffers", (DL_FUNC)&C_arrow_array_buffers, 1},
    {"C_page_size", (DL_FUNC)&C_page_size, 0},
    {"C_am_length", (DL_FUNC)&C_am_length, 1},
    {"C_am_null_count", (DL_FUNC)&C_am_null_count, 1},
    {"C_am_format", (DL_FUNC)&C_am_format, 1},
    {"C_am_reduce", (DL_FUNC)&C_am_reduce, 2},
    {"C_am_compare_scalar", (DL_FUNC)&C_am_compare_scalar, 4},
    {"C_am_compare_array", (DL_FUNC)&C_am_compare_array, 3},
    {"C_am_filter", (DL_FUNC)&C_am_filter, 2},
    {"C_am_take", (DL_FUNC)&C_am_take, 2},
    {"C_am_slice", (DL_FUNC)&C_am_slice, 3},
    {"C_am_argsort", (DL_FUNC)&C_am_argsort, 2},
    {"C_am_sort", (DL_FUNC)&C_am_sort, 2},
    {"C_am_child_count", (DL_FUNC)&C_am_child_count, 1},
    {"C_am_child", (DL_FUNC)&C_am_child, 2},
    {"C_am_group_by_keys", (DL_FUNC)&C_am_group_by_keys, 1},
    {"C_am_group_by_group_count", (DL_FUNC)&C_am_group_by_group_count, 1},
    {"C_am_group_by_keys_result", (DL_FUNC)&C_am_group_by_keys_result, 2},
    {"C_am_group_by_ids", (DL_FUNC)&C_am_group_by_ids, 1},
    {"C_am_group_agg", (DL_FUNC)&C_am_group_agg, 4},
    {"C_am_plan_source_create", (DL_FUNC)&C_am_plan_source_create, 3},
    {"C_am_plan_run", (DL_FUNC)&C_am_plan_run, 3},
    {"C_am_plan_explain", (DL_FUNC)&C_am_plan_explain, 3},
    {"C_am_plan_result_dim", (DL_FUNC)&C_am_plan_result_dim, 1},
    {"C_am_plan_column_names", (DL_FUNC)&C_am_plan_column_names, 1},
    {"C_am_plan_column", (DL_FUNC)&C_am_plan_column, 2},
    {NULL, NULL, 0}};

void R_init_arrowmetal(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
