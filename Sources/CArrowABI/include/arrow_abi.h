// Apache Arrow C Data Interface, C Stream Interface and C Device Data Interface.
// Struct layouts are normative and copied verbatim from
// https://arrow.apache.org/docs/format/CDataInterface.html and
// https://arrow.apache.org/docs/format/CDeviceDataInterface.html
#ifndef ARROW_ABI_H
#define ARROW_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

#define ARROW_FLAG_DICTIONARY_ORDERED 1
#define ARROW_FLAG_NULLABLE 2
#define ARROW_FLAG_MAP_KEYS_SORTED 4

struct ArrowSchema {
  const char* format;
  const char* name;
  const char* metadata;
  int64_t flags;
  int64_t n_children;
  struct ArrowSchema** children;
  struct ArrowSchema* dictionary;
  void (*release)(struct ArrowSchema*);
  void* private_data;
};

struct ArrowArray {
  int64_t length;
  int64_t null_count;
  int64_t offset;
  int64_t n_buffers;
  int64_t n_children;
  const void** buffers;
  struct ArrowArray** children;
  struct ArrowArray* dictionary;
  void (*release)(struct ArrowArray*);
  void* private_data;
};

#endif  // ARROW_C_DATA_INTERFACE

#ifndef ARROW_C_STREAM_INTERFACE
#define ARROW_C_STREAM_INTERFACE

struct ArrowArrayStream {
  int (*get_schema)(struct ArrowArrayStream*, struct ArrowSchema* out);
  int (*get_next)(struct ArrowArrayStream*, struct ArrowArray* out);
  const char* (*get_last_error)(struct ArrowArrayStream*);
  void (*release)(struct ArrowArrayStream*);
  void* private_data;
};

#endif  // ARROW_C_STREAM_INTERFACE

#ifndef ARROW_C_DEVICE_DATA_INTERFACE
#define ARROW_C_DEVICE_DATA_INTERFACE

#define ARROW_DEVICE_CPU 1
#define ARROW_DEVICE_CUDA 2
#define ARROW_DEVICE_CUDA_HOST 3
#define ARROW_DEVICE_OPENCL 4
#define ARROW_DEVICE_VULKAN 7
#define ARROW_DEVICE_METAL 8
#define ARROW_DEVICE_VPI 9
#define ARROW_DEVICE_ROCM 10
#define ARROW_DEVICE_ROCM_HOST 11
#define ARROW_DEVICE_EXT_DEV 12
#define ARROW_DEVICE_CUDA_MANAGED 13
#define ARROW_DEVICE_ONEAPI 14
#define ARROW_DEVICE_WEBGPU 15
#define ARROW_DEVICE_HEXAGON 16

typedef int32_t ArrowDeviceType;

struct ArrowDeviceArray {
  struct ArrowArray array;
  int64_t device_id;
  ArrowDeviceType device_type;
  int64_t reserved[3];
  void* sync_event;
};

#endif  // ARROW_C_DEVICE_DATA_INTERFACE

#ifndef ARROW_C_DEVICE_STREAM_INTERFACE
#define ARROW_C_DEVICE_STREAM_INTERFACE

struct ArrowDeviceArrayStream {
  ArrowDeviceType device_type;
  int (*get_schema)(struct ArrowDeviceArrayStream*, struct ArrowSchema*);
  int (*get_next)(struct ArrowDeviceArrayStream*, struct ArrowDeviceArray*);
  const char* (*get_last_error)(struct ArrowDeviceArrayStream*);
  void (*release)(struct ArrowDeviceArrayStream*);
  void* private_data;
};

#endif  // ARROW_C_DEVICE_STREAM_INTERFACE

#ifdef __cplusplus
}
#endif

#endif  // ARROW_ABI_H
