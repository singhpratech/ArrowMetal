// ArrowMetal N-API addon.
//
// The dylib is opened at runtime with dlopen (ARROWMETAL_LIB first, then the in-tree
// .build/release copy), so this addon links against nothing but Node. Every entry point below is a
// thin, synchronous pass-through: it builds the Arrow C Data Interface structs over buffers that
// JavaScript already owns, hands them to libArrowMetalC, and wraps the result handle in an External.
//
// Ownership rules, which the TypeScript layer relies on:
//   * Import wraps the producer's bytes; it never copies on this side. The ArrowArray we hand to
//     am_import points straight at the V8 backing store. We hold a JS reference to every producer
//     buffer in the ArrowArray's own private_data (ImportPriv), and those references live exactly
//     as long as ArrowMetal holds the array -- they are dropped by the ArrowArray release callback
//     and by nothing else. Releasing the *handle* does not drop them: on a page-aligned import
//     ArrowMetal wraps the V8 pages with makeBuffer(bytesNoCopy:), and a slice, a group-by or a
//     registered plan source built from that array retains the import, so the pages must stay
//     reachable from JS after the original handle is gone.
//   * Export wraps the ArrowMetal result; it never copies on this side either. Each output buffer
//     becomes an external ArrayBuffer whose finalizer decrements a refcount; the ArrowArray's own
//     release runs when the last one is collected.
//   * Every handle we hand to JS is an External carrying a napi type tag, so passing a plan source
//     where an array is expected throws instead of being reinterpreted.

#include <napi.h>

#include <dlfcn.h>
#include <unistd.h>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

extern "C" {
#include "arrow_abi.h"
}

// ---------------------------------------------------------------------------------------------
// Dynamic loader
// ---------------------------------------------------------------------------------------------

namespace {

struct am_array;
struct am_groupby;
struct am_plan_source;
struct am_plan_result;

struct Lib {
  void* handle = nullptr;
  std::string path;

  const char* (*am_version)(void) = nullptr;
  const char* (*am_device_name)(void) = nullptr;
  const char* (*am_last_error)(void) = nullptr;

  int (*am_import)(const struct ArrowSchema*, struct ArrowArray*, am_array**) = nullptr;
  int (*am_export)(am_array*, struct ArrowSchema*, struct ArrowArray*) = nullptr;
  void (*am_release)(am_array*) = nullptr;
  int64_t (*am_length)(am_array*) = nullptr;
  int64_t (*am_null_count)(am_array*) = nullptr;
  const char* (*am_format)(am_array*) = nullptr;

  int (*am_reduce)(am_array*, int, int64_t*, double*, int*, int*) = nullptr;
  int (*am_compare_scalar)(am_array*, int, const void*, am_array**) = nullptr;
  int (*am_compare_array)(am_array*, int, am_array*, am_array**) = nullptr;
  int (*am_arith_scalar)(am_array*, int, const void*, am_array**) = nullptr;
  int (*am_cast)(am_array*, const char*, am_array**) = nullptr;
  int (*am_filter)(am_array*, am_array*, am_array**) = nullptr;
  int (*am_take)(am_array*, am_array*, am_array**) = nullptr;
  int (*am_slice)(am_array*, int64_t, int64_t, am_array**) = nullptr;
  int (*am_argsort)(am_array*, int, am_array**) = nullptr;
  int (*am_sort)(am_array*, int, am_array**) = nullptr;
  int (*am_lexsort)(am_array**, const int*, int64_t, am_array**) = nullptr;

  int (*am_group_by_keys)(am_array**, int64_t, am_groupby**) = nullptr;
  int64_t (*am_group_by_group_count)(am_groupby*) = nullptr;
  int (*am_group_by_keys_result)(am_groupby*, int64_t, am_array**) = nullptr;
  void (*am_group_by_release)(am_groupby*) = nullptr;
  int (*am_group_agg_ex)(am_groupby*, am_array*, int, double, am_array**) = nullptr;

  int (*am_plan_source_create)(const char*, am_array**, const char**, int64_t,
                               am_plan_source**) = nullptr;
  void (*am_plan_source_release)(am_plan_source*) = nullptr;
  int (*am_plan_run)(const char*, am_plan_source**, int64_t, int, am_plan_result**) = nullptr;
  const char* (*am_plan_explain)(const char*, am_plan_source**, int64_t, int) = nullptr;
  int64_t (*am_plan_column_count)(am_plan_result*) = nullptr;
  int64_t (*am_plan_row_count)(am_plan_result*) = nullptr;
  const char* (*am_plan_column_name)(am_plan_result*, int64_t) = nullptr;
  int (*am_plan_column)(am_plan_result*, int64_t, am_array**) = nullptr;
  void (*am_plan_result_release)(am_plan_result*) = nullptr;
};

Lib g;

template <typename T>
void bind(T& slot, const char* name, std::vector<std::string>& missing) {
  slot = reinterpret_cast<T>(dlsym(g.handle, name));
  if (slot == nullptr) missing.push_back(name);
}

bool fileExists(const std::string& p) {
  if (p.empty()) return false;
  FILE* f = fopen(p.c_str(), "rb");
  if (f == nullptr) return false;
  fclose(f);
  return true;
}

// Resolves and dlopens the dylib. Order: $ARROWMETAL_LIB, then ../.build/release relative to the
// package directory that JavaScript hands us.
void loadLibrary(const std::string& packageDir) {
  if (g.handle != nullptr) return;

  const char* env = getenv("ARROWMETAL_LIB");
  std::string envPath = env != nullptr ? std::string(env) : std::string();
  std::string relPath = packageDir + "/../.build/release/libArrowMetalC.dylib";

  std::string chosen;
  if (!envPath.empty() && fileExists(envPath)) {
    chosen = envPath;
  } else if (fileExists(relPath)) {
    chosen = relPath;
  } else {
    std::string msg = "ArrowMetal: libArrowMetalC.dylib not found. Looked at ARROWMETAL_LIB=";
    msg += envPath.empty() ? "(unset)" : envPath;
    msg += " and " + relPath +
           ". Build it with `swift build -c release --product ArrowMetalC` in the ArrowMetal "
           "checkout, or point ARROWMETAL_LIB at an existing copy.";
    throw std::runtime_error(msg);
  }

  g.handle = dlopen(chosen.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (g.handle == nullptr) {
    std::string msg = "ArrowMetal: dlopen(" + chosen + ") failed: ";
    const char* e = dlerror();
    msg += e != nullptr ? e : "unknown error";
    throw std::runtime_error(msg);
  }
  g.path = chosen;

  std::vector<std::string> missing;
  bind(g.am_version, "am_version", missing);
  bind(g.am_device_name, "am_device_name", missing);
  bind(g.am_last_error, "am_last_error", missing);
  bind(g.am_import, "am_import", missing);
  bind(g.am_export, "am_export", missing);
  bind(g.am_release, "am_release", missing);
  bind(g.am_length, "am_length", missing);
  bind(g.am_null_count, "am_null_count", missing);
  bind(g.am_format, "am_format", missing);
  bind(g.am_reduce, "am_reduce", missing);
  bind(g.am_compare_scalar, "am_compare_scalar", missing);
  bind(g.am_compare_array, "am_compare_array", missing);
  bind(g.am_arith_scalar, "am_arith_scalar", missing);
  bind(g.am_cast, "am_cast", missing);
  bind(g.am_filter, "am_filter", missing);
  bind(g.am_take, "am_take", missing);
  bind(g.am_slice, "am_slice", missing);
  bind(g.am_argsort, "am_argsort", missing);
  bind(g.am_sort, "am_sort", missing);
  bind(g.am_lexsort, "am_lexsort", missing);
  bind(g.am_group_by_keys, "am_group_by_keys", missing);
  bind(g.am_group_by_group_count, "am_group_by_group_count", missing);
  bind(g.am_group_by_keys_result, "am_group_by_keys_result", missing);
  bind(g.am_group_by_release, "am_group_by_release", missing);
  bind(g.am_group_agg_ex, "am_group_agg_ex", missing);
  bind(g.am_plan_source_create, "am_plan_source_create", missing);
  bind(g.am_plan_source_release, "am_plan_source_release", missing);
  bind(g.am_plan_run, "am_plan_run", missing);
  bind(g.am_plan_explain, "am_plan_explain", missing);
  bind(g.am_plan_column_count, "am_plan_column_count", missing);
  bind(g.am_plan_row_count, "am_plan_row_count", missing);
  bind(g.am_plan_column_name, "am_plan_column_name", missing);
  bind(g.am_plan_column, "am_plan_column", missing);
  bind(g.am_plan_result_release, "am_plan_result_release", missing);

  if (!missing.empty()) {
    std::string msg = "ArrowMetal: " + chosen + " is missing entry points:";
    for (const auto& m : missing) msg += " " + m;
    throw std::runtime_error(msg);
  }
}

void requireLoaded(const Napi::Env& env) {
  if (g.handle == nullptr) {
    throw Napi::Error::New(env, "ArrowMetal: library not loaded; call load(packageDir) first");
  }
}

// Turns a non-zero return code into a JS exception.
//
// rc 2 is the C ABI's argument guard: the Swift side rejects the call before it starts and does
// NOT set am_last_error, so reading it there reports whatever the previous call on this thread left
// behind. We give rc 2 its own message rather than a stale one.
void check(const Napi::Env& env, int rc) {
  if (rc == 0) return;
  if (rc == 2) {
    throw Napi::Error::New(
        env,
        "ArrowMetal: the call was rejected by an argument guard (rc 2) — a null handle, an empty "
        "column list, or an index out of range. The C ABI sets no message for this case, so there "
        "is nothing more specific to report.");
  }
  const char* e = g.am_last_error();
  throw Napi::Error::New(env, e != nullptr && *e != 0 ? e : "ArrowMetal: unknown error");
}

// ---------------------------------------------------------------------------------------------
// Format helpers
// ---------------------------------------------------------------------------------------------

// Bytes per element for a fixed-width primitive format; -1 for bool (bitmap), -2 for utf8, 0 for
// anything this binding does not carry.
int elementWidth(const char* f) {
  if (f == nullptr || f[0] == 0) return 0;
  if (f[1] != 0) return 0;
  switch (f[0]) {
    case 'c': case 'C': return 1;
    case 's': case 'S': return 2;
    case 'i': case 'I': case 'f': return 4;
    case 'l': case 'L': case 'g': return 8;
    case 'b': return -1;
    case 'u': return -2;
    default: return 0;
  }
}

// ---------------------------------------------------------------------------------------------
// Handles
// ---------------------------------------------------------------------------------------------

// References to V8 TypedArrays that ArrowMetal has just let go of, waiting to be destroyed on the
// JS thread. napi_delete_reference is not safe from an arbitrary thread, and an ArrowArray release
// callback may in principle run from one, so the callback parks the references here and the next
// N-API entry point drains them. See drainPending().
std::mutex gPendingMutex;
std::vector<Napi::Reference<Napi::Value>> gPending;

void drainPending() {
  std::vector<Napi::Reference<Napi::Value>> local;
  {
    std::lock_guard<std::mutex> lock(gPendingMutex);
    local.swap(gPending);
  }
  local.clear();  // ~Reference runs here, on the JS thread
}

// Private data of an ArrowArray we build over JS memory.
//
// This owns the JS references that pin the producer's TypedArrays, and it is freed by our own
// release callback and by nothing else. That is the whole point: on a page-aligned import
// ArrowMetal wraps the V8 pages with makeBuffer(bytesNoCopy:), and objects derived from the
// imported array — a slice, a group-by, a registered plan source — keep those pages alive through
// their own retain of the imported array. So the producer's buffers must stay reachable from JS
// until ArrowMetal calls release, which is exactly when the last of those derived objects is gone.
// Dropping the references when the *handle* is released instead would leave the GPU reading freed
// V8 memory.
struct ImportPriv {
  const void* buffers[3];
  std::vector<Napi::Reference<Napi::Value>> keepAlive;
  // Shared with the importing frame, because priv itself may be gone by the time we want to read
  // it: a copying import calls release before am_import returns.
  std::shared_ptr<std::atomic<bool>> released;
};

void importArrayRelease(struct ArrowArray* a) {
  auto* priv = static_cast<ImportPriv*>(a->private_data);
  a->release = nullptr;
  a->private_data = nullptr;
  if (priv == nullptr) return;
  if (priv->released) priv->released->store(true);
  {
    std::lock_guard<std::mutex> lock(gPendingMutex);
    for (auto& r : priv->keepAlive) gPending.push_back(std::move(r));
  }
  delete priv;
}

void importSchemaRelease(struct ArrowSchema* s) { s->release = nullptr; }

struct ArrayHandle {
  am_array* p = nullptr;
  // No keepAlive here. The producer's buffers are pinned by ImportPriv, which outlives this handle
  // whenever ArrowMetal still holds the imported array through a derived object.
  bool retained = false;  // ArrowMetal kept our buffers rather than releasing them at import

  ~ArrayHandle() {
    if (p != nullptr) g.am_release(p);
    p = nullptr;
  }
};

struct GroupByHandle {
  am_groupby* p = nullptr;
  ~GroupByHandle() {
    if (p != nullptr) g.am_group_by_release(p);
  }
};

struct PlanSourceHandle {
  am_plan_source* p = nullptr;
  ~PlanSourceHandle() {
    if (p != nullptr) g.am_plan_source_release(p);
  }
};

struct PlanResultHandle {
  am_plan_result* p = nullptr;
  ~PlanResultHandle() {
    if (p != nullptr) g.am_plan_result_release(p);
  }
};

// Type tags, so an External of one kind passed where another is expected throws instead of being
// reinterpreted. Every handle we hand to JS carries one; every unwrap checks it.
const napi_type_tag kArrayTag = {0x9a1f4c2d7b3e4051ULL, 0xa7c6d8e920f14b35ULL};
const napi_type_tag kGroupByTag = {0x4d2b8e1f60a34c77ULL, 0xb3f5178c9d2e4a60ULL};
const napi_type_tag kPlanSourceTag = {0x1c7e35a8f4b9426dULL, 0x8e50d63b2af7194cULL};
const napi_type_tag kPlanResultTag = {0x76b04e93c1d8452aULL, 0x2f9ac514e07b638dULL};

template <typename T>
void finalizeHandle(Napi::Env, T* h) {
  delete h;
}

template <typename T>
Napi::Value wrapHandle(Napi::Env env, T* h, const napi_type_tag& tag) {
  Napi::External<T> ext = Napi::External<T>::New(env, h, finalizeHandle<T>);
  ext.TypeTag(&tag);
  return ext;
}

template <typename T>
T* unwrapTagged(const Napi::Env& env, const Napi::Value& v, const napi_type_tag& tag,
                const char* what) {
  if (!v.IsExternal()) {
    throw Napi::TypeError::New(env, std::string("ArrowMetal: expected ") + what +
                                        ", got a different kind of value");
  }
  Napi::External<T> ext = v.As<Napi::External<T>>();
  if (!ext.CheckTypeTag(&tag)) {
    throw Napi::TypeError::New(env, std::string("ArrowMetal: expected ") + what +
                                        ", got a handle of a different kind");
  }
  return ext.Data();
}

Napi::Value wrapArray(Napi::Env env, am_array* p) {
  auto* h = new ArrayHandle();
  h->p = p;
  return wrapHandle(env, h, kArrayTag);
}

ArrayHandle* unwrapArray(const Napi::Env& env, const Napi::Value& v) {
  auto* h = unwrapTagged<ArrayHandle>(env, v, kArrayTag, "an array handle");
  if (h == nullptr || h->p == nullptr) {
    throw Napi::Error::New(env, "ArrowMetal: array handle is released");
  }
  return h;
}

// Base address of a TypedArray / DataView / Buffer, honouring its byteOffset.
uint8_t* viewBase(const Napi::Env& env, const Napi::Value& v, size_t* byteLength) {
  if (v.IsTypedArray()) {
    Napi::TypedArray t = v.As<Napi::TypedArray>();
    *byteLength = t.ByteLength();
    return static_cast<uint8_t*>(t.ArrayBuffer().Data()) + t.ByteOffset();
  }
  if (v.IsDataView()) {
    Napi::DataView d = v.As<Napi::DataView>();
    *byteLength = d.ByteLength();
    return static_cast<uint8_t*>(d.ArrayBuffer().Data()) + d.ByteOffset();
  }
  if (v.IsArrayBuffer()) {
    Napi::ArrayBuffer b = v.As<Napi::ArrayBuffer>();
    *byteLength = b.ByteLength();
    return static_cast<uint8_t*>(b.Data());
  }
  throw Napi::TypeError::New(env, "ArrowMetal: expected a TypedArray, DataView or ArrayBuffer");
}

// ---------------------------------------------------------------------------------------------
// Export: wrap ArrowMetal buffers as external ArrayBuffers
// ---------------------------------------------------------------------------------------------

struct ExportHold {
  struct ArrowSchema schema {};
  struct ArrowArray array {};
  std::atomic<int> rc{0};
};

void exportBufferFinalizer(Napi::Env, void*, ExportHold* hold) {
  if (hold->rc.fetch_sub(1) == 1) {
    if (hold->array.release != nullptr) hold->array.release(&hold->array);
    if (hold->schema.release != nullptr) hold->schema.release(&hold->schema);
    delete hold;
  }
}

Napi::Value wrapExportedBuffer(Napi::Env env, ExportHold* hold, const void* data, size_t len) {
  if (data == nullptr) return env.Null();
  hold->rc.fetch_add(1);
  return Napi::ArrayBuffer::New(env, const_cast<void*>(data), len, exportBufferFinalizer, hold);
}

// ---------------------------------------------------------------------------------------------
// Scalars
// ---------------------------------------------------------------------------------------------

// Encodes a JS number/bigint/boolean into 8 bytes of the array's own element type.
void encodeScalar(const Napi::Env& env, const char* fmt, const Napi::Value& v, uint8_t* out) {
  memset(out, 0, 8);
  auto asI64 = [&]() -> int64_t {
    if (v.IsBigInt()) {
      bool lossless = false;
      return v.As<Napi::BigInt>().Int64Value(&lossless);
    }
    if (v.IsBoolean()) return v.As<Napi::Boolean>().Value() ? 1 : 0;
    return static_cast<int64_t>(v.As<Napi::Number>().DoubleValue());
  };
  auto asF64 = [&]() -> double {
    if (v.IsBigInt()) {
      bool lossless = false;
      return static_cast<double>(v.As<Napi::BigInt>().Int64Value(&lossless));
    }
    return v.As<Napi::Number>().DoubleValue();
  };
  switch (fmt[0]) {
    case 'c': { int8_t x = static_cast<int8_t>(asI64()); memcpy(out, &x, 1); return; }
    case 'C': { uint8_t x = static_cast<uint8_t>(asI64()); memcpy(out, &x, 1); return; }
    case 's': { int16_t x = static_cast<int16_t>(asI64()); memcpy(out, &x, 2); return; }
    case 'S': { uint16_t x = static_cast<uint16_t>(asI64()); memcpy(out, &x, 2); return; }
    case 'i': { int32_t x = static_cast<int32_t>(asI64()); memcpy(out, &x, 4); return; }
    case 'I': { uint32_t x = static_cast<uint32_t>(asI64()); memcpy(out, &x, 4); return; }
    case 'l': { int64_t x = asI64(); memcpy(out, &x, 8); return; }
    case 'L': { uint64_t x = static_cast<uint64_t>(asI64()); memcpy(out, &x, 8); return; }
    case 'f': { float x = static_cast<float>(asF64()); memcpy(out, &x, 4); return; }
    case 'g': { double x = asF64(); memcpy(out, &x, 8); return; }
    case 'b': { uint8_t x = v.ToBoolean().Value() ? 1 : 0; memcpy(out, &x, 1); return; }
    default:
      throw Napi::Error::New(
          env, std::string("ArrowMetal: no scalar encoding for Arrow format \"") + fmt + "\"");
  }
}

// ---------------------------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------------------------

Napi::Value Load(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  std::string dir = info[0].As<Napi::String>();
  try {
    loadLibrary(dir);
  } catch (const std::runtime_error& e) {
    throw Napi::Error::New(env, e.what());
  }
  Napi::Object out = Napi::Object::New(env);
  out.Set("path", Napi::String::New(env, g.path));
  out.Set("version", Napi::String::New(env, g.am_version()));
  out.Set("device", Napi::String::New(env, g.am_device_name()));
  out.Set("pageSize", Napi::Number::New(env, static_cast<double>(getpagesize())));
  return out;
}

// importArray(format, length, offset, nullCount, validity|null, data|null, offsets|null)
Napi::Value ImportArray(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  requireLoaded(env);

  std::string format = info[0].As<Napi::String>();
  int64_t length = info[1].As<Napi::Number>().Int64Value();
  int64_t offset = info[2].As<Napi::Number>().Int64Value();
  int64_t nullCount = info[3].As<Napi::Number>().Int64Value();

  int width = elementWidth(format.c_str());
  if (width == 0) {
    throw Napi::Error::New(
        env, "ArrowMetal (Node): Arrow format \"" + format +
                 "\" is not carried by this binding. Supported: c C s S i I l L f g b u.");
  }

  if (length < 0 || offset < 0) {
    throw Napi::Error::New(env, "ArrowMetal (Node): length and offset must be >= 0, got length " +
                                    std::to_string(length) + " and offset " +
                                    std::to_string(offset) + ".");
  }
  const int64_t rows = offset + length;  // the C Data Interface addresses rows [offset, offset+length)

  auto* priv = new ImportPriv();
  // am_import moves the ArrowArray, which per the C Data Interface nulls our copy's release pointer
  // whether it wrapped the buffers or copied them. This flag is the only way to tell the two apart:
  // it is set if and only if ArrowMetal actually ran the release callback.
  auto releasedFlag = std::make_shared<std::atomic<bool>>(false);
  priv->released = releasedFlag;

  // Every buffer is checked against the size the Arrow layout requires for `rows` rows. Without
  // this a short validity bitmap (one byte for 64 rows, say) reads past the end of the V8 view and
  // the answer depends on whatever V8 put next to it.
  auto addBuffer = [&](int slot, const Napi::Value& v, const char* name, int64_t needed) {
    if (v.IsNull() || v.IsUndefined()) {
      priv->buffers[slot] = nullptr;
      return;
    }
    size_t len = 0;
    uint8_t* base = viewBase(env, v, &len);
    if (static_cast<int64_t>(len) < needed) {
      throw Napi::Error::New(
          env, std::string("ArrowMetal (Node): the ") + name + " buffer is " + std::to_string(len) +
                   " bytes but Arrow format \"" + format + "\" needs at least " +
                   std::to_string(needed) + " for offset " + std::to_string(offset) + " plus " +
                   std::to_string(length) + " rows.");
    }
    priv->buffers[slot] = base;
    priv->keepAlive.push_back(Napi::Reference<Napi::Value>::New(v, 1));
  };

  const int64_t bitmapBytes = (rows + 7) / 8;
  int64_t nBuffers = (width == -2) ? 3 : 2;
  try {
    addBuffer(0, info[4], "validity", bitmapBytes);
    if (width == -2) {
      addBuffer(1, info[6], "utf8 offsets", (rows + 1) * 4);
      // The values buffer must reach the last offset, which we can only know once the offsets
      // buffer has been checked.
      int64_t neededBytes = 0;
      if (priv->buffers[1] != nullptr && rows >= 0) {
        const int32_t* offs = static_cast<const int32_t*>(priv->buffers[1]);
        const int32_t last = offs[rows];
        if (last < 0) {
          throw Napi::Error::New(env, "ArrowMetal (Node): the utf8 offsets buffer ends at " +
                                          std::to_string(last) + ", which is negative.");
        }
        neededBytes = last;
      }
      addBuffer(2, info[5], "utf8 values", neededBytes);
    } else if (width == -1) {
      addBuffer(1, info[5], "boolean values", bitmapBytes);
    } else {
      addBuffer(1, info[5], "values", rows * width);
    }
  } catch (...) {
    delete priv;
    throw;
  }

  struct ArrowSchema schema {};
  schema.format = format.c_str();
  schema.name = "";
  schema.metadata = nullptr;
  schema.flags = ARROW_FLAG_NULLABLE;
  schema.n_children = 0;
  schema.children = nullptr;
  schema.dictionary = nullptr;
  schema.release = importSchemaRelease;
  schema.private_data = nullptr;

  struct ArrowArray array {};
  array.length = length;
  array.null_count = nullCount;
  array.offset = offset;
  array.n_buffers = nBuffers;
  array.n_children = 0;
  array.buffers = priv->buffers;
  array.children = nullptr;
  array.dictionary = nullptr;
  array.release = importArrayRelease;
  array.private_data = priv;

  am_array* out = nullptr;
  int rc = g.am_import(&schema, &array, &out);
  if (schema.release != nullptr) schema.release(&schema);
  if (rc != 0) {
    // importArrayRelease frees priv and parks the JS references; if it already ran, array.release
    // is null and priv is gone. Either way there is nothing left for us to delete.
    if (array.release != nullptr) array.release(&array);
    if (rc == 2) check(env, rc);
    const char* e = g.am_last_error();
    throw Napi::Error::New(env, e != nullptr && *e != 0 ? e : "ArrowMetal: am_import failed");
  }

  auto* h = new ArrayHandle();
  h->p = out;
  h->retained = !releasedFlag->load();
  return wrapHandle(env, h, kArrayTag);
}

Napi::Value ImportRetained(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* h = unwrapArray(env, info[0]);
  return Napi::Boolean::New(env, h->retained);
}

Napi::Value ExportArray(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  requireLoaded(env);
  auto* h = unwrapArray(env, info[0]);

  auto* hold = new ExportHold();
  int rc = g.am_export(h->p, &hold->schema, &hold->array);
  if (rc != 0) {
    delete hold;
    const char* e = g.am_last_error();
    throw Napi::Error::New(env, e != nullptr && *e != 0 ? e : "ArrowMetal: am_export failed");
  }

  std::string format = hold->schema.format != nullptr ? hold->schema.format : "";
  int width = elementWidth(format.c_str());
  int64_t length = hold->array.length;
  int64_t offset = hold->array.offset;

  if (width == 0) {
    if (hold->array.release != nullptr) hold->array.release(&hold->array);
    if (hold->schema.release != nullptr) hold->schema.release(&hold->schema);
    delete hold;
    throw Napi::Error::New(
        env, "ArrowMetal (Node): result has Arrow format \"" + format +
                 "\", which this binding does not export to Arrow JS.");
  }

  Napi::Object out = Napi::Object::New(env);
  out.Set("format", Napi::String::New(env, format));
  out.Set("length", Napi::Number::New(env, static_cast<double>(length)));
  out.Set("offset", Napi::Number::New(env, static_cast<double>(offset)));
  out.Set("nullCount", Napi::Number::New(env, static_cast<double>(hold->array.null_count)));

  // Refcount starts at 1 so an early finalizer cannot free the hold before we finish here.
  hold->rc.store(1);
  const void* validity = hold->array.n_buffers > 0 ? hold->array.buffers[0] : nullptr;
  size_t validityLen = static_cast<size_t>((offset + length + 7) / 8);
  out.Set("validity", wrapExportedBuffer(env, hold, validity, validityLen));

  if (width == -2) {
    const void* offsetsBuf = hold->array.n_buffers > 1 ? hold->array.buffers[1] : nullptr;
    size_t offsetsLen = static_cast<size_t>((offset + length + 1) * 4);
    const int32_t* offs = static_cast<const int32_t*>(offsetsBuf);
    size_t dataLen = offs != nullptr ? static_cast<size_t>(offs[offset + length]) : 0;
    const void* dataBuf = hold->array.n_buffers > 2 ? hold->array.buffers[2] : nullptr;
    out.Set("offsets", wrapExportedBuffer(env, hold, offsetsBuf, offsetsLen));
    out.Set("data", wrapExportedBuffer(env, hold, dataBuf, dataLen));
  } else {
    const void* dataBuf = hold->array.n_buffers > 1 ? hold->array.buffers[1] : nullptr;
    size_t dataLen = (width == -1) ? static_cast<size_t>((offset + length + 7) / 8)
                                   : static_cast<size_t>((offset + length) * width);
    out.Set("offsets", env.Null());
    out.Set("data", wrapExportedBuffer(env, hold, dataBuf, dataLen));
  }

  // Drop the construction reference; if every buffer was NULL this releases immediately.
  exportBufferFinalizer(env, nullptr, hold);
  return out;
}

Napi::Value ArrayLength(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  return Napi::Number::New(env, static_cast<double>(g.am_length(unwrapArray(env, info[0])->p)));
}

Napi::Value ArrayNullCount(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  return Napi::Number::New(env, static_cast<double>(g.am_null_count(unwrapArray(env, info[0])->p)));
}

Napi::Value ArrayFormat(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  const char* f = g.am_format(unwrapArray(env, info[0])->p);
  return Napi::String::New(env, f != nullptr ? f : "");
}

Napi::Value ReleaseArray(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* h = unwrapTagged<ArrayHandle>(env, info[0], kArrayTag, "an array handle");
  if (h != nullptr && h->p != nullptr) {
    // Releases only the handle. The producer's buffers stay pinned until ArrowMetal calls the
    // ArrowArray release callback, which may be much later: a slice, a group-by or a registered
    // plan source built from this array retains the import.
    g.am_release(h->p);
    h->p = nullptr;
  }
  drainPending();
  return env.Undefined();
}

// reduce(handle, op) -> number | bigint | null
Napi::Value Reduce(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  requireLoaded(env);
  auto* h = unwrapArray(env, info[0]);
  int op = info[1].As<Napi::Number>().Int32Value();
  int64_t i64 = 0;
  double f64 = 0;
  int kind = 0;
  int isNull = 0;
  check(env, g.am_reduce(h->p, op, &i64, &f64, &kind, &isNull));
  if (isNull != 0) return env.Null();
  if (kind == 0) return Napi::BigInt::New(env, i64);
  if (kind == 1) return Napi::BigInt::New(env, static_cast<uint64_t>(i64));
  return Napi::Number::New(env, f64);
}

Napi::Value CompareScalar(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  requireLoaded(env);
  auto* h = unwrapArray(env, info[0]);
  int op = info[1].As<Napi::Number>().Int32Value();
  uint8_t scalar[8];
  encodeScalar(env, g.am_format(h->p), info[2], scalar);
  am_array* out = nullptr;
  check(env, g.am_compare_scalar(h->p, op, scalar, &out));
  return wrapArray(env, out);
}

Napi::Value CompareArray(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  requireLoaded(env);
  auto* a = unwrapArray(env, info[0]);
  int op = info[1].As<Napi::Number>().Int32Value();
  auto* b = unwrapArray(env, info[2]);
  am_array* out = nullptr;
  check(env, g.am_compare_array(a->p, op, b->p, &out));
  return wrapArray(env, out);
}

Napi::Value ArithScalar(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  requireLoaded(env);
  auto* h = unwrapArray(env, info[0]);
  int op = info[1].As<Napi::Number>().Int32Value();
  uint8_t scalar[8];
  encodeScalar(env, g.am_format(h->p), info[2], scalar);
  am_array* out = nullptr;
  check(env, g.am_arith_scalar(h->p, op, scalar, &out));
  return wrapArray(env, out);
}

Napi::Value Cast(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  requireLoaded(env);
  auto* h = unwrapArray(env, info[0]);
  std::string fmt = info[1].As<Napi::String>();
  am_array* out = nullptr;
  check(env, g.am_cast(h->p, fmt.c_str(), &out));
  return wrapArray(env, out);
}

Napi::Value Filter(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  requireLoaded(env);
  auto* a = unwrapArray(env, info[0]);
  auto* m = unwrapArray(env, info[1]);
  am_array* out = nullptr;
  check(env, g.am_filter(a->p, m->p, &out));
  return wrapArray(env, out);
}

Napi::Value Take(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  requireLoaded(env);
  auto* a = unwrapArray(env, info[0]);
  auto* i = unwrapArray(env, info[1]);
  am_array* out = nullptr;
  check(env, g.am_take(a->p, i->p, &out));
  return wrapArray(env, out);
}

Napi::Value Slice(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  requireLoaded(env);
  auto* a = unwrapArray(env, info[0]);
  int64_t off = info[1].As<Napi::Number>().Int64Value();
  int64_t len = info[2].As<Napi::Number>().Int64Value();
  am_array* out = nullptr;
  check(env, g.am_slice(a->p, off, len, &out));
  return wrapArray(env, out);
}

Napi::Value Argsort(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  requireLoaded(env);
  auto* a = unwrapArray(env, info[0]);
  int desc = info[1].As<Napi::Boolean>().Value() ? 1 : 0;
  am_array* out = nullptr;
  check(env, g.am_argsort(a->p, desc, &out));
  return wrapArray(env, out);
}

Napi::Value Sort(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  requireLoaded(env);
  auto* a = unwrapArray(env, info[0]);
  int desc = info[1].As<Napi::Boolean>().Value() ? 1 : 0;
  am_array* out = nullptr;
  check(env, g.am_sort(a->p, desc, &out));
  return wrapArray(env, out);
}

Napi::Value Lexsort(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  requireLoaded(env);
  Napi::Array cols = info[0].As<Napi::Array>();
  Napi::Array descs = info[1].As<Napi::Array>();
  std::vector<am_array*> ptrs;
  std::vector<int> desc;
  for (uint32_t i = 0; i < cols.Length(); i++) {
    ptrs.push_back(unwrapArray(env, cols.Get(i))->p);
    desc.push_back(descs.Get(i).ToBoolean().Value() ? 1 : 0);
  }
  am_array* out = nullptr;
  check(env, g.am_lexsort(ptrs.data(), desc.data(), static_cast<int64_t>(ptrs.size()), &out));
  return wrapArray(env, out);
}

Napi::Value GroupByKeys(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  requireLoaded(env);
  Napi::Array cols = info[0].As<Napi::Array>();
  std::vector<am_array*> ptrs;
  for (uint32_t i = 0; i < cols.Length(); i++) ptrs.push_back(unwrapArray(env, cols.Get(i))->p);
  am_groupby* gb = nullptr;
  check(env, g.am_group_by_keys(ptrs.data(), static_cast<int64_t>(ptrs.size()), &gb));
  auto* h = new GroupByHandle();
  h->p = gb;
  return wrapHandle(env, h, kGroupByTag);
}

GroupByHandle* unwrapGroupBy(const Napi::Env& env, const Napi::Value& v) {
  auto* h = unwrapTagged<GroupByHandle>(env, v, kGroupByTag, "a group-by handle");
  if (h == nullptr || h->p == nullptr) throw Napi::Error::New(env, "ArrowMetal: group-by released");
  return h;
}

Napi::Value GroupCount(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  return Napi::Number::New(
      env, static_cast<double>(g.am_group_by_group_count(unwrapGroupBy(env, info[0])->p)));
}

Napi::Value GroupKeysResult(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* h = unwrapGroupBy(env, info[0]);
  int64_t i = info[1].As<Napi::Number>().Int64Value();
  am_array* out = nullptr;
  check(env, g.am_group_by_keys_result(h->p, i, &out));
  return wrapArray(env, out);
}

Napi::Value GroupAgg(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* h = unwrapGroupBy(env, info[0]);
  am_array* values = nullptr;
  if (!info[1].IsNull() && !info[1].IsUndefined()) values = unwrapArray(env, info[1])->p;
  int op = info[2].As<Napi::Number>().Int32Value();
  double p1 = info[3].As<Napi::Number>().DoubleValue();
  am_array* out = nullptr;
  check(env, g.am_group_agg_ex(h->p, values, op, p1, &out));
  return wrapArray(env, out);
}

Napi::Value PlanSourceCreate(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  requireLoaded(env);
  std::string name = info[0].As<Napi::String>();
  Napi::Array cols = info[1].As<Napi::Array>();
  Napi::Array names = info[2].As<Napi::Array>();
  std::vector<am_array*> ptrs;
  std::vector<std::string> owned;
  for (uint32_t i = 0; i < cols.Length(); i++) {
    ptrs.push_back(unwrapArray(env, cols.Get(i))->p);
    owned.push_back(names.Get(i).As<Napi::String>());
  }
  std::vector<const char*> cnames;
  for (const auto& s : owned) cnames.push_back(s.c_str());
  am_plan_source* src = nullptr;
  check(env, g.am_plan_source_create(name.c_str(), ptrs.data(), cnames.data(),
                                     static_cast<int64_t>(ptrs.size()), &src));
  auto* h = new PlanSourceHandle();
  h->p = src;
  return wrapHandle(env, h, kPlanSourceTag);
}

std::vector<am_plan_source*> unwrapSources(const Napi::Env& env, const Napi::Value& v) {
  Napi::Array arr = v.As<Napi::Array>();
  std::vector<am_plan_source*> out;
  for (uint32_t i = 0; i < arr.Length(); i++) {
    Napi::Value e = arr.Get(i);
    auto* h = unwrapTagged<PlanSourceHandle>(env, e, kPlanSourceTag, "a plan source handle");
    if (h == nullptr || h->p == nullptr) {
      throw Napi::Error::New(env, "ArrowMetal: plan source released");
    }
    out.push_back(h->p);
  }
  return out;
}

Napi::Value PlanRun(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  requireLoaded(env);
  std::string json = info[0].As<Napi::String>();
  auto sources = unwrapSources(env, info[1]);
  int optimize = info[2].ToBoolean().Value() ? 1 : 0;
  am_plan_result* r = nullptr;
  check(env, g.am_plan_run(json.c_str(), sources.data(), static_cast<int64_t>(sources.size()),
                           optimize, &r));
  auto* h = new PlanResultHandle();
  h->p = r;
  return wrapHandle(env, h, kPlanResultTag);
}

Napi::Value PlanExplain(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  requireLoaded(env);
  std::string json = info[0].As<Napi::String>();
  auto sources = unwrapSources(env, info[1]);
  int optimize = info[2].ToBoolean().Value() ? 1 : 0;
  const char* text = g.am_plan_explain(json.c_str(), sources.data(),
                                       static_cast<int64_t>(sources.size()), optimize);
  if (text == nullptr) {
    const char* e = g.am_last_error();
    throw Napi::Error::New(env, e != nullptr && *e != 0 ? e : "ArrowMetal: am_plan_explain failed");
  }
  return Napi::String::New(env, text);
}

PlanResultHandle* unwrapResult(const Napi::Env& env, const Napi::Value& v) {
  auto* h = unwrapTagged<PlanResultHandle>(env, v, kPlanResultTag, "a plan result handle");
  if (h == nullptr || h->p == nullptr) throw Napi::Error::New(env, "ArrowMetal: result released");
  return h;
}

Napi::Value PlanResultInfo(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* h = unwrapResult(env, info[0]);
  int64_t n = g.am_plan_column_count(h->p);
  Napi::Array names = Napi::Array::New(env, static_cast<size_t>(n));
  for (int64_t i = 0; i < n; i++) {
    const char* nm = g.am_plan_column_name(h->p, i);
    names.Set(static_cast<uint32_t>(i), Napi::String::New(env, nm != nullptr ? nm : ""));
  }
  Napi::Object out = Napi::Object::New(env);
  out.Set("names", names);
  out.Set("rows", Napi::Number::New(env, static_cast<double>(g.am_plan_row_count(h->p))));
  return out;
}

Napi::Value PlanColumn(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* h = unwrapResult(env, info[0]);
  int64_t i = info[1].As<Napi::Number>().Int64Value();
  am_array* out = nullptr;
  check(env, g.am_plan_column(h->p, i, &out));
  return wrapArray(env, out);
}

Napi::Value LastError(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  requireLoaded(env);
  const char* e = g.am_last_error();
  return Napi::String::New(env, e != nullptr ? e : "");
}

// Address of a typed array's first byte, for the page-alignment measurement.
Napi::Value BufferAddress(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  size_t len = 0;
  uint8_t* p = viewBase(env, info[0], &len);
  return Napi::BigInt::New(env, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p)));
}

// Every N-API entry point drains the references ArrowMetal parked when it released an imported
// array. ~Reference must run on the JS thread, and this is the first place we are guaranteed to be
// on it after a release callback fired.
template <Napi::Value (*Fn)(const Napi::CallbackInfo&)>
Napi::Value Entry(const Napi::CallbackInfo& info) {
  drainPending();
  return Fn(info);
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set("load", Napi::Function::New(env, Entry<Load>));
  exports.Set("importArray", Napi::Function::New(env, Entry<ImportArray>));
  exports.Set("importRetained", Napi::Function::New(env, Entry<ImportRetained>));
  exports.Set("exportArray", Napi::Function::New(env, Entry<ExportArray>));
  exports.Set("length", Napi::Function::New(env, Entry<ArrayLength>));
  exports.Set("nullCount", Napi::Function::New(env, Entry<ArrayNullCount>));
  exports.Set("format", Napi::Function::New(env, Entry<ArrayFormat>));
  exports.Set("release", Napi::Function::New(env, Entry<ReleaseArray>));
  exports.Set("reduce", Napi::Function::New(env, Entry<Reduce>));
  exports.Set("compareScalar", Napi::Function::New(env, Entry<CompareScalar>));
  exports.Set("compareArray", Napi::Function::New(env, Entry<CompareArray>));
  exports.Set("arithScalar", Napi::Function::New(env, Entry<ArithScalar>));
  exports.Set("cast", Napi::Function::New(env, Entry<Cast>));
  exports.Set("filter", Napi::Function::New(env, Entry<Filter>));
  exports.Set("take", Napi::Function::New(env, Entry<Take>));
  exports.Set("slice", Napi::Function::New(env, Entry<Slice>));
  exports.Set("argsort", Napi::Function::New(env, Entry<Argsort>));
  exports.Set("sort", Napi::Function::New(env, Entry<Sort>));
  exports.Set("lexsort", Napi::Function::New(env, Entry<Lexsort>));
  exports.Set("groupByKeys", Napi::Function::New(env, Entry<GroupByKeys>));
  exports.Set("groupCount", Napi::Function::New(env, Entry<GroupCount>));
  exports.Set("groupKeysResult", Napi::Function::New(env, Entry<GroupKeysResult>));
  exports.Set("groupAgg", Napi::Function::New(env, Entry<GroupAgg>));
  exports.Set("planSourceCreate", Napi::Function::New(env, Entry<PlanSourceCreate>));
  exports.Set("planRun", Napi::Function::New(env, Entry<PlanRun>));
  exports.Set("planExplain", Napi::Function::New(env, Entry<PlanExplain>));
  exports.Set("planResultInfo", Napi::Function::New(env, Entry<PlanResultInfo>));
  exports.Set("planColumn", Napi::Function::New(env, Entry<PlanColumn>));
  exports.Set("lastError", Napi::Function::New(env, Entry<LastError>));
  exports.Set("bufferAddress", Napi::Function::New(env, Entry<BufferAddress>));
  return exports;
}

}  // namespace

NODE_API_MODULE(arrowmetal_native, Init)
