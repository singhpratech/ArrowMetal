// The CPU pass of the chunked utf8_view / binary_view import (Sources/ArrowMetal/ChunkedImport.swift):
// each chunk's 16-byte views copied into the merged views buffer, the out-of-line ones pointed at the
// merged data buffers, checked, and their byte lengths summed, in one branch-free pass.
#ifndef ARROWMETAL_VIEW_REWRITE_H
#define ARROWMETAL_VIEW_REWRITE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Where one of a chunk's data buffers went: `base` is added to the offset of a view into it, and
/// `size` is its byte size (capped at UINT32_MAX; 0 for a missing buffer, so no view into it passes).
typedef struct {
  uint32_t base;
  uint32_t size;
} am_view_target;

/// How a chunk's out-of-line views get their merged buffer index: `index_keep` 1 adds `index_add` to
/// the view's own index (the chunk's buffers are merged buffers index_add, index_add + 1, ... as they
/// are); `index_keep` 0 sets it to `index_add` (every buffer of the chunk was copied into that one
/// merged buffer).
typedef struct {
  uint32_t index_keep;
  uint32_t index_add;
  uint32_t count;  // the chunk's data buffer count; `targets` has this many entries
  uint32_t pad;
  const am_view_target* targets;
} am_view_map;

/// Copies `n` views from `src` to `dst`. An inline view (length <= 12) is copied as it is. An
/// out-of-line view gets its merged buffer index from `map` and its buffer's base added to its
/// offset; it passes the check when its length and offset are below 2^31, its buffer index is below
/// `map->count`, and offset + length is at most its buffer's size. `validity` is the merged bitmap
/// (64-bit words, bit i for merged row i; NULL when every row is valid) and `first_row` the merged row
/// of `src[0]`. Returns the byte total of the valid rows, or -1 when a view fails the check (`dst`
/// then holds unspecified views for these rows and the caller rewrites them view by view). Internal to
/// the library: not exported from libArrowMetalC.
__attribute__((visibility("hidden")))
int64_t am_rewrite_views(const void* src, void* dst, size_t n, const am_view_map* map, const uint64_t* validity,
                         size_t first_row);

#ifdef __cplusplus
}
#endif

#endif
