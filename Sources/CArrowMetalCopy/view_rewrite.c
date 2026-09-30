// The view pass of the chunked utf8_view / binary_view import (include/view_rewrite.h).
//
// With NEON, four views per step: `vld4q_u32` deinterleaves them into lengths, prefixes, buffer
// indices and offsets, every lane is rewritten and checked with vector selects (no branch per view),
// and `vst4q_u32` interleaves them back. A chunk with one data buffer uses its target as a constant;
// with several, each lane's target is loaded by its (clamped) buffer index. Without NEON, and for the
// last rows of a call, the same rule runs one view at a time.

#include "view_rewrite.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/// Validity bits of rows row .. row + 3 (bit 0: row). The caller guarantees the four rows exist, so
/// the word after `row`'s is read only when they reach into it.
static inline uint32_t valid4(const uint64_t* v, size_t row) {
  if (!v) return 15u;
  size_t w = row >> 6;
  unsigned sh = (unsigned)(row & 63);
  uint64_t x = v[w] >> sh;
  if (sh > 60) x |= v[w + 1] << (64 - sh);
  return (uint32_t)x & 15u;
}

static inline int valid1(const uint64_t* v, size_t row) {
  return v ? (int)((v[row >> 6] >> (row & 63)) & 1u) : 1;
}

/// One view; returns 1 when it fails the check.
static inline int one_view(const uint32_t* s, uint32_t* d, const am_view_map* m, int valid, uint64_t* acc) {
  uint32_t len = s[0], prefix = s[1], idx = s[2], off = s[3];
  d[0] = len;
  d[1] = prefix;
  if (len <= 12) {
    d[2] = idx;
    d[3] = off;
    if (valid) *acc += len;
    return 0;
  }
  if (((len | off) >> 31) != 0 || idx >= m->count) return 1;
  am_view_target t = m->targets[idx];
  if ((uint64_t)len + off > t.size) return 1;
  d[2] = m->index_keep ? idx + m->index_add : m->index_add;
  d[3] = off + t.base;
  if (valid) *acc += len;
  return 0;
}

int64_t am_rewrite_views(const void* src, void* dst, size_t n, const am_view_map* map, const uint64_t* validity,
                         size_t first_row) {
  const uint32_t* s = (const uint32_t*)src;
  uint32_t* d = (uint32_t*)dst;
  uint64_t total = 0;
  int bad = 0;
  size_t j = 0;
#if defined(__ARM_NEON)
  {
    const uint32x4_t twelve = vdupq_n_u32(12), top = vdupq_n_u32(0x80000000u);
    const uint32x4_t count = vdupq_n_u32(map->count), last = vdupq_n_u32(map->count ? map->count - 1 : 0);
    const uint32x4_t keep = vdupq_n_u32(map->index_keep ? 0xFFFFFFFFu : 0u), add = vdupq_n_u32(map->index_add);
    static const uint32_t bitsel[4] = {1, 2, 4, 8};
    const uint32x4_t sel = vld1q_u32(bitsel);
    const am_view_target* t = map->targets;
    const int gather = map->count > 1;
    const uint32x4_t base1 = vdupq_n_u32(map->count == 1 ? t[0].base : 0u);
    const uint32x4_t size1 = vdupq_n_u32(map->count == 1 ? t[0].size : 0u);
    uint32x4_t fails = vdupq_n_u32(0);
    uint64x2_t acc = vdupq_n_u64(0);
    for (; j + 4 <= n; j += 4) {
      uint32x4x4_t v = vld4q_u32(s + j * 4);
      const uint32x4_t len = v.val[0], idx = v.val[2], off = v.val[3];
      const uint32x4_t lg = vcgtq_u32(len, twelve);
      uint32x4_t base = base1, size = size1;
      if (gather) {
        const uint32x4_t c = vminq_u32(idx, last);
        const uint64x2_t e01 = vcombine_u64(vld1_u64((const uint64_t*)&t[vgetq_lane_u32(c, 0)]),
                                            vld1_u64((const uint64_t*)&t[vgetq_lane_u32(c, 1)]));
        const uint64x2_t e23 = vcombine_u64(vld1_u64((const uint64_t*)&t[vgetq_lane_u32(c, 2)]),
                                            vld1_u64((const uint64_t*)&t[vgetq_lane_u32(c, 3)]));
        base = vuzp1q_u32(vreinterpretq_u32_u64(e01), vreinterpretq_u32_u64(e23));
        size = vuzp2q_u32(vreinterpretq_u32_u64(e01), vreinterpretq_u32_u64(e23));
      }
      // len and off below 2^31 (else the lane has failed already), so len + off does not wrap.
      uint32x4_t f = vorrq_u32(vcgeq_u32(vorrq_u32(len, off), top), vcgeq_u32(idx, count));
      f = vorrq_u32(f, vcgtq_u32(vaddq_u32(len, off), size));
      fails = vorrq_u32(fails, vandq_u32(lg, f));
      v.val[2] = vbslq_u32(lg, vaddq_u32(vandq_u32(idx, keep), add), idx);
      v.val[3] = vaddq_u32(off, vandq_u32(lg, base));
      vst4q_u32(d + j * 4, v);
      const uint32x4_t valid = vtstq_u32(vdupq_n_u32(valid4(validity, first_row + j)), sel);
      acc = vpadalq_u32(acc, vandq_u32(len, valid));
    }
    bad = vmaxvq_u32(fails) != 0;
    total = vgetq_lane_u64(acc, 0) + vgetq_lane_u64(acc, 1);
  }
#endif
  for (; j < n; j++) bad |= one_view(s + j * 4, d + j * 4, map, valid1(validity, first_row + j), &total);
  return bad ? -1 : (int64_t)total;
}
