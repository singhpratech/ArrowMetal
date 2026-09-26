import Foundation

// MSL for block decompression of Parquet pages.
//
// PARALLEL STRATEGY
//
// Snappy and LZ4 are both byte-oriented LZ77 formats: a stream of tokens, each either "copy N literal
// bytes from the input" or "copy N bytes from N' bytes back in the *output*". Token boundaries are only
// known after the previous token has been parsed, so a single block cannot be parsed in parallel — but a
// Parquet file has thousands of pages, each an independent block, and that is where the parallelism is.
//
// One threadgroup of 256 threads owns one page. Thread 0 walks the token stream and publishes each token
// through threadgroup memory; all 256 threads then move that token's bytes together. Two things make
// this fast, and both were measured, not assumed:
//
//   * The tag bytes thread 0 reads are a dependency chain of device-memory loads, and one of those costs
//     hundreds of cycles. So all 256 threads first stage the next 8 KB of the compressed stream into
//     threadgroup memory (`dec_fill`), and the parse reads from there. Payload bytes are still read
//     straight from device memory: those reads are wide and coalesced and are not on the chain.
//   * A page's bytes are overwhelmingly *literal* runs, and an incompressible page is one enormous
//     literal. Moving those with a single 32-lane SIMD group left the GPU almost idle (32 pages x 32
//     lanes = 1024 threads for 32 MB); giving the whole threadgroup the copy, four bytes deep per
//     thread, is what turns the copy into a memory-speed operation.
//
// Back-references that overlap the region they write (offset < length, which is how both formats encode
// runs) are still copied in parallel: the copy repeats a pattern of `offset` bytes, so output byte i is
// simply `out[dst - offset + (i % offset)]`, and every one of those source bytes was already written
// before this token began. The barrier at the end of each token makes those stores visible.
//
// Status codes are written per block so the host can turn corrupt input into an error instead of
// silently wrong data.
enum DecompressSource {
    static let source = KernelSource.prelude + """

    #define DEC_OK        0u
    #define DEC_OVERRUN   1u
    #define DEC_BAD_TOKEN 2u
    #define DEC_SHORT     3u
    #define DEC_WIN       8192u
    #define DEC_MARGIN    32u
    #define DEC_TG        32u

    // Descriptor per block: source offset/length and destination offset/length.
    struct BlockDesc { uint srcOffset; uint srcLength; uint dstOffset; uint dstLength; };

    // Cooperative move of `n` bytes by the whole threadgroup. `fromOutput` selects a back-reference
    // (pattern repeat) over a literal run.
    //
    // The literal path takes four bytes per thread per iteration -- thread `t` takes i, i+TG, i+2TG,
    // i+3TG -- so consecutive threads still touch consecutive bytes (one coalesced line per instruction)
    // while four independent loads are in flight to cover device-memory latency.
    inline void dec_move(device uchar* dst, device const uchar* src, uint dBase, uint op,
                         uint sOff, uint n, uint backOff, bool fromOutput, uint tid) {
        if (fromOutput) {
            uint m = backOff;
            for (uint i = tid; i < n; i += DEC_TG) {
                dst[dBase + op + i] = dst[dBase + op - m + (i % m)];
            }
            return;
        }
        device uchar* d = dst + dBase + op;
        device const uchar* s = src + sOff;
        uint i = tid;
        for (; i + 3u * DEC_TG < n; i += 4u * DEC_TG) {
            uchar a = s[i], b = s[i + DEC_TG], c = s[i + 2u * DEC_TG], e = s[i + 3u * DEC_TG];
            d[i] = a; d[i + DEC_TG] = b; d[i + 2u * DEC_TG] = c; d[i + 3u * DEC_TG] = e;
        }
        for (; i < n; i += DEC_TG) d[i] = s[i];
    }

    // Stages [at, at + DEC_WIN) of the compressed stream in threadgroup memory. All threads must call.
    inline void dec_fill(threadgroup uchar* win, device const uchar* src, uint sBase, uint sLen,
                         uint at, uint tid, thread uint& winBase, thread uint& winCount) {
        uint n = (at < sLen) ? min(DEC_WIN, sLen - at) : 0u;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < n; i += DEC_TG) win[i] = src[sBase + at + i];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        winBase = at;
        winCount = n;
    }
    // One byte of the compressed stream, from the window when it is inside and from device memory when
    // a long literal has carried the cursor past the staged range.
    inline uchar dec_rd(threadgroup const uchar* win, device const uchar* src, uint sBase,
                        uint at, uint winBase, uint winCount) {
        uint rel = at - winBase;
        return (at >= winBase && rel < winCount) ? win[rel] : src[sBase + at];
    }

    // ---------------------------------------------------------------- Snappy
    //
    // Preamble: a varint holding the uncompressed length. Then tags:
    //   xxxxxx00  literal, length-1 in the top 6 bits (>= 60 means 1..4 extra little-endian length bytes)
    //   xxxyyy01  copy, length = 4 + yyy, offset = (xxx << 8) | next byte
    //   xxxxxx10  copy, length = 1 + xxxxxx, offset = next 2 bytes little-endian
    //   xxxxxx11  copy, length = 1 + xxxxxx, offset = next 4 bytes little-endian
    kernel void snappy_decompress(device const uchar* src [[buffer(0)]],
                                  device uchar* dst [[buffer(1)]],
                                  device const BlockDesc* blocks [[buffer(2)]],
                                  constant uint& nBlocks [[buffer(3)]],
                                  device uint* status [[buffer(4)]],
                                  uint b [[threadgroup_position_in_grid]],
                                  uint tid [[thread_position_in_threadgroup]]) {
        threadgroup uchar win[DEC_WIN];
        if (b >= nBlocks) return;
        BlockDesc d = blocks[b];
        uint sBase = d.srcOffset, sLen = d.srcLength, dBase = d.dstOffset, dLen = d.dstLength;
        uint sp = 0u, op = 0u, err = DEC_OK, winBase = 0u, winCount = 0u;
        dec_fill(win, src, sBase, sLen, 0u, tid, winBase, winCount);
        if (tid == 0u) {
            while (sp < sLen) { uchar c = dec_rd(win, src, sBase, sp, winBase, winCount); sp++; if ((c & 0x80u) == 0u) break; }
        }
        sp = simd_broadcast(sp, 0u);

        while (op < dLen && err == DEC_OK) {
            if (sp + DEC_MARGIN > winBase + winCount && winBase + winCount < sLen) {
                dec_fill(win, src, sBase, sLen, sp, tid, winBase, winCount);
            }
            uint tn = 0u, tsOff = 0u, tback = 0u, tcopy = 0u;
            if (tid == 0u) {
                uint n = 0u, sOff = 0u, backOff = 0u, isCopy = 0u;
                if (sp >= sLen) { err = DEC_SHORT; }
                else {
                    uint tag = (uint)dec_rd(win, src, sBase, sp, winBase, winCount); sp++;
                    uint t = tag & 3u;
                    if (t == 0u) {
                        uint len = tag >> 2;
                        if (len >= 60u) {
                            uint extra = len - 59u;
                            if (sp + extra > sLen) { err = DEC_SHORT; }
                            else {
                                uint v = 0u;
                                for (uint k = 0u; k < extra; k++) {
                                    v |= ((uint)dec_rd(win, src, sBase, sp + k, winBase, winCount)) << (8u * k);
                                }
                                sp += extra;
                                len = v;
                            }
                        }
                        n = len + 1u;
                        sOff = sBase + sp;
                        sp += n;
                        if (sp > sLen) err = DEC_SHORT;
                    } else if (t == 1u) {
                        if (sp + 1u > sLen) { err = DEC_SHORT; }
                        else {
                            n = 4u + ((tag >> 2) & 7u);
                            backOff = (((tag >> 5) & 7u) << 8) | (uint)dec_rd(win, src, sBase, sp, winBase, winCount);
                            sp += 1u;
                            isCopy = 1u;
                        }
                    } else {
                        uint w = (t == 2u) ? 2u : 4u;
                        if (sp + w > sLen) { err = DEC_SHORT; }
                        else {
                            n = 1u + (tag >> 2);
                            uint v = 0u;
                            for (uint k = 0u; k < w; k++) {
                                v |= ((uint)dec_rd(win, src, sBase, sp + k, winBase, winCount)) << (8u * k);
                            }
                            sp += w;
                            backOff = v;
                            isCopy = 1u;
                        }
                    }
                    if (err == DEC_OK) {
                        if (op + n > dLen) err = DEC_OVERRUN;
                        else if (isCopy != 0u && (backOff == 0u || backOff > op)) err = DEC_BAD_TOKEN;
                    }
                }
                tn = n; tsOff = sOff; tback = backOff; tcopy = isCopy;
            }
            err = simd_broadcast(err, 0u);
            if (err != DEC_OK) break;
            uint n = simd_broadcast(tn, 0u);
            uint sOff = simd_broadcast(tsOff, 0u);
            uint backOff = simd_broadcast(tback, 0u);
            uint isCopy = simd_broadcast(tcopy, 0u);
            sp = simd_broadcast(sp, 0u);
            // A back-reference reads bytes the other lanes wrote for earlier tokens, so it needs the
            // stores ordered; a literal reads only the input, so it does not. Skipping the barrier on
            // literal tokens is worth about 15% on a page of mixed tokens.
            if (isCopy != 0u) simdgroup_barrier(mem_flags::mem_device);
            dec_move(dst, src, dBase, op, sOff, n, backOff, isCopy != 0u, tid);
            op += n;
        }
        if (tid == 0u) status[b] = err;
    }

    // ---------------------------------------------------------------- LZ4 (raw block format)
    //
    // Sequences of: token byte (high nibble = literal length, low nibble = match length - 4, either
    // extended by 0xFF-terminated byte runs), literals, a 2-byte little-endian match offset, the match.
    // The last sequence ends after its literals. Parquet's legacy LZ4 codec wraps the raw block in the
    // "Hadoop" framing (big-endian uncompressed size, big-endian compressed size); it is detected here
    // rather than on the host so the CPU still never reads a compressed byte.
    kernel void lz4_decompress(device const uchar* src [[buffer(0)]],
                               device uchar* dst [[buffer(1)]],
                               device const BlockDesc* blocks [[buffer(2)]],
                               constant uint& nBlocks [[buffer(3)]],
                               device uint* status [[buffer(4)]],
                               uint b [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]]) {
        threadgroup uchar win[DEC_WIN];
        if (b >= nBlocks) return;
        BlockDesc d = blocks[b];
        uint sBase = d.srcOffset, sLen = d.srcLength, dBase = d.dstOffset, dLen = d.dstLength;
        if (sLen >= 8u) {
            uint u = ((uint)src[sBase] << 24) | ((uint)src[sBase+1] << 16) | ((uint)src[sBase+2] << 8) | (uint)src[sBase+3];
            uint c = ((uint)src[sBase+4] << 24) | ((uint)src[sBase+5] << 16) | ((uint)src[sBase+6] << 8) | (uint)src[sBase+7];
            if (u == dLen && c == sLen - 8u) { sBase += 8u; sLen -= 8u; }
        }
        uint sp = 0u, op = 0u, err = DEC_OK, winBase = 0u, winCount = 0u;
        dec_fill(win, src, sBase, sLen, 0u, tid, winBase, winCount);
        while (op < dLen && err == DEC_OK) {
            if (sp + DEC_MARGIN > winBase + winCount && winBase + winCount < sLen) {
                dec_fill(win, src, sBase, sLen, sp, tid, winBase, winCount);
            }
            uint tlit = 0u, tlsrc = 0u, tmlen = 0u, tback = 0u, thas = 0u;
            if (tid == 0u) {
                uint litLen = 0u, matchLen = 0u, backOff = 0u, litSrc = 0u, hasMatch = 0u;
                if (sp >= sLen) { err = DEC_SHORT; }
                else {
                    uint token = (uint)dec_rd(win, src, sBase, sp, winBase, winCount); sp++;
                    litLen = token >> 4;
                    if (litLen == 15u) {
                        uint c2 = 255u;
                        while (c2 == 255u && sp < sLen) { c2 = (uint)dec_rd(win, src, sBase, sp, winBase, winCount); sp++; litLen += c2; }
                    }
                    litSrc = sBase + sp;
                    sp += litLen;
                    if (sp > sLen || op + litLen > dLen) { err = DEC_OVERRUN; }
                    else if (sp + 2u <= sLen) {
                        backOff = (uint)dec_rd(win, src, sBase, sp, winBase, winCount)
                                | ((uint)dec_rd(win, src, sBase, sp + 1u, winBase, winCount) << 8);
                        sp += 2u;
                        matchLen = token & 15u;
                        if (matchLen == 15u) {
                            uint c2 = 255u;
                            while (c2 == 255u && sp < sLen) { c2 = (uint)dec_rd(win, src, sBase, sp, winBase, winCount); sp++; matchLen += c2; }
                        }
                        matchLen += 4u;
                        hasMatch = 1u;
                        if (backOff == 0u || backOff > op + litLen || matchLen > dLen - op - litLen) err = DEC_BAD_TOKEN;
                    }
                }
                tlit = litLen; tlsrc = litSrc; tmlen = matchLen; tback = backOff; thas = hasMatch;
            }
            err = simd_broadcast(err, 0u);
            if (err != DEC_OK) break;
            uint litLen = simd_broadcast(tlit, 0u);
            uint litSrc = simd_broadcast(tlsrc, 0u);
            uint matchLen = simd_broadcast(tmlen, 0u);
            uint backOff = simd_broadcast(tback, 0u);
            uint hasMatch = simd_broadcast(thas, 0u);
            sp = simd_broadcast(sp, 0u);
            if (litLen > 0u) {
                dec_move(dst, src, dBase, op, litSrc, litLen, 0u, false, tid);
                op += litLen;
            }
            if (hasMatch == 0u) break;
            simdgroup_barrier(mem_flags::mem_device);
            dec_move(dst, src, dBase, op, 0u, matchLen, backOff, true, tid);
            op += matchLen;
        }
        if (tid == 0u) status[b] = (op == dLen || err != DEC_OK) ? err : DEC_SHORT;
    }

    // ---------------------------------------------------------------- one page per thread
    //
    // The kernels above spend a SIMD group on a page, and a page of short tokens -- sequential or
    // low-cardinality integers, timestamps -- is tens of thousands of tokens of a few bytes each, so 31
    // of the 32 lanes wait while lane 0 parses and every token costs a broadcast and a barrier. These
    // two decode a page per *thread*: each lane walks its own page's token stream and copies its own
    // bytes, so a SIMD group advances 32 token streams at once. They are used for pages whose output is
    // noticeably larger than their input (many tokens); a page that is mostly one long literal still
    // goes to the SIMD-group kernels, which copy it 32 lanes wide. Same checks, in the same order, and
    // the same status codes as the SIMD-group kernels.

    // Each lane keeps the last LRING bytes it wrote in its own slice of threadgroup memory. A
    // back-reference inside that window -- nearly all of them in token-dense pages, whose matches are
    // a value or two back -- reads from there instead of from device memory, where a byte this thread
    // stored a moment ago is a round trip to L2 away.
    #define LRING 256u
    #define LMASK (LRING - 1u)
    #define LANES 32u

    // The token stream is read 16 aligned bytes at a time into registers, so parsing a token is not a
    // chain of one-byte device loads. `abs` is the byte's offset from the binding point, which is page
    // aligned, and a page is a multiple of 16 bytes, so the aligned block never leaves the mapping.
    inline uchar lane_rd(device const uchar* src, thread uint& wBase, thread uint4& w, uint abs) {
        uint blk = abs & ~15u;
        if (blk != wBase) { wBase = blk; w = *((device const uint4*)(src + blk)); }
        uint i = abs & 15u;
        return (uchar)((w[i >> 2] >> ((i & 3u) * 8u)) & 0xFFu);
    }

    inline void lane_literal(device uchar* o, threadgroup uchar* ring, uint op, device const uchar* f, uint n) {
        for (uint i = 0u; i < n; i++) {
            uchar c = f[i];
            o[op + i] = c;
            ring[(op + i) & LMASK] = c;
        }
    }
    inline void lane_backcopy(device uchar* o, threadgroup uchar* ring, uint op, uint back, uint n) {
        if (back <= LRING) {
            for (uint i = 0u; i < n; i++) {
                uchar c = ring[(op - back + i) & LMASK];
                o[op + i] = c;
                ring[(op + i) & LMASK] = c;
            }
        } else {
            // A thread's own earlier stores are visible to its later loads.
            for (uint i = 0u; i < n; i++) {
                uchar c = o[op - back + i];
                o[op + i] = c;
                ring[(op + i) & LMASK] = c;
            }
        }
    }

    kernel void snappy_decompress_lane(device const uchar* src [[buffer(0)]],
                                       device uchar* dst [[buffer(1)]],
                                       device const BlockDesc* blocks [[buffer(2)]],
                                       constant uint& nBlocks [[buffer(3)]],
                                       device uint* status [[buffer(4)]],
                                       uint b [[thread_position_in_grid]],
                                       uint lane [[thread_index_in_threadgroup]]) {
        threadgroup uchar rings[LANES * LRING];
        threadgroup uchar* ring = rings + lane * LRING;
        uint wBase = 1u;                 // not a multiple of 16: no block loaded yet
        uint4 w = uint4(0u);
        if (b >= nBlocks) return;
        BlockDesc d = blocks[b];
        uint sBase = d.srcOffset;
        device const uchar* s = src + sBase;
        device uchar* o = dst + d.dstOffset;
        uint sLen = d.srcLength, dLen = d.dstLength;
        uint sp = 0u, op = 0u, err = DEC_OK;
        while (sp < sLen) { uchar c = lane_rd(src, wBase, w, sBase + sp); sp++; if ((c & 0x80u) == 0u) break; }
        while (op < dLen) {
            if (sp >= sLen) { err = DEC_SHORT; break; }
            uint tag = (uint)lane_rd(src, wBase, w, sBase + sp); sp++;
            uint t = tag & 3u;
            uint n = 0u, backOff = 0u;
            if (t == 0u) {
                uint len = tag >> 2;
                if (len >= 60u) {
                    uint extra = len - 59u;
                    if (sp + extra > sLen) { err = DEC_SHORT; break; }
                    uint v = 0u;
                    for (uint k = 0u; k < extra; k++) v |= ((uint)lane_rd(src, wBase, w, sBase + sp + k)) << (8u * k);
                    sp += extra;
                    len = v;
                }
                n = len + 1u;
                // Compared as differences so that a length near 2^32 cannot wrap past the checks.
                if (n > sLen - sp) { err = DEC_SHORT; break; }
                uint from = sp;
                sp += n;
                if (n > dLen - op) { err = DEC_OVERRUN; break; }
                lane_literal(o, ring, op, s + from, n);
                op += n;
                continue;
            } else if (t == 1u) {
                if (sp + 1u > sLen) { err = DEC_SHORT; break; }
                n = 4u + ((tag >> 2) & 7u);
                backOff = (((tag >> 5) & 7u) << 8) | (uint)lane_rd(src, wBase, w, sBase + sp);
                sp += 1u;
            } else {
                uint nw = (t == 2u) ? 2u : 4u;
                if (sp + nw > sLen) { err = DEC_SHORT; break; }
                n = 1u + (tag >> 2);
                uint v = 0u;
                for (uint k = 0u; k < nw; k++) v |= ((uint)lane_rd(src, wBase, w, sBase + sp + k)) << (8u * k);
                sp += nw;
                backOff = v;
            }
            if (op + n > dLen) { err = DEC_OVERRUN; break; }
            if (backOff == 0u || backOff > op) { err = DEC_BAD_TOKEN; break; }
            lane_backcopy(o, ring, op, backOff, n);
            op += n;
        }
        status[b] = err;
    }

    kernel void lz4_decompress_lane(device const uchar* src [[buffer(0)]],
                                    device uchar* dst [[buffer(1)]],
                                    device const BlockDesc* blocks [[buffer(2)]],
                                    constant uint& nBlocks [[buffer(3)]],
                                    device uint* status [[buffer(4)]],
                                    uint b [[thread_position_in_grid]],
                                    uint lane [[thread_index_in_threadgroup]]) {
        threadgroup uchar rings[LANES * LRING];
        threadgroup uchar* ring = rings + lane * LRING;
        uint wBase = 1u;                 // not a multiple of 16: no block loaded yet
        uint4 w = uint4(0u);
        if (b >= nBlocks) return;
        BlockDesc d = blocks[b];
        uint sBase = d.srcOffset, sLen = d.srcLength, dLen = d.dstLength;
        if (sLen >= 8u) {
            uint u = ((uint)src[sBase] << 24) | ((uint)src[sBase+1] << 16) | ((uint)src[sBase+2] << 8) | (uint)src[sBase+3];
            uint c = ((uint)src[sBase+4] << 24) | ((uint)src[sBase+5] << 16) | ((uint)src[sBase+6] << 8) | (uint)src[sBase+7];
            if (u == dLen && c == sLen - 8u) { sBase += 8u; sLen -= 8u; }
        }
        device const uchar* s = src + sBase;
        device uchar* o = dst + d.dstOffset;
        uint sp = 0u, op = 0u, err = DEC_OK;
        while (op < dLen && err == DEC_OK) {
            if (sp >= sLen) { err = DEC_SHORT; break; }
            uint token = (uint)lane_rd(src, wBase, w, sBase + sp); sp++;
            uint litLen = token >> 4;
            if (litLen == 15u) {
                uint c2 = 255u;
                while (c2 == 255u && sp < sLen) { c2 = (uint)lane_rd(src, wBase, w, sBase + sp); sp++; litLen += c2; }
            }
            // Compared as differences so that a length near 2^32 cannot wrap past the checks.
            if (litLen > sLen - sp || litLen > dLen - op) { err = DEC_OVERRUN; break; }
            uint litSrc = sp;
            sp += litLen;
            uint matchLen = 0u, backOff = 0u;
            bool hasMatch = false;
            if (sp + 2u <= sLen) {
                backOff = (uint)lane_rd(src, wBase, w, sBase + sp) | ((uint)lane_rd(src, wBase, w, sBase + sp + 1u) << 8);
                sp += 2u;
                matchLen = token & 15u;
                if (matchLen == 15u) {
                    uint c2 = 255u;
                    while (c2 == 255u && sp < sLen) { c2 = (uint)lane_rd(src, wBase, w, sBase + sp); sp++; matchLen += c2; }
                }
                matchLen += 4u;
                hasMatch = true;
                if (backOff == 0u || backOff > op + litLen || op + litLen + matchLen > dLen) err = DEC_BAD_TOKEN;
            }
            if (err != DEC_OK) break;
            if (litLen > 0u) {
                lane_literal(o, ring, op, s + litSrc, litLen);
                op += litLen;
            }
            if (!hasMatch) break;
            lane_backcopy(o, ring, op, backOff, matchLen);
            op += matchLen;
        }
        status[b] = (op == dLen || err != DEC_OK) ? err : DEC_SHORT;
    }

    // ---------------------------------------------------------------- plain copy
    //
    // Used when pages must be gathered into one contiguous buffer but carry no compression (and for the
    // host-decompressed codecs, whose output is already staged in shared memory).
    kernel void block_copy(device const uchar* src [[buffer(0)]],
                           device uchar* dst [[buffer(1)]],
                           device const BlockDesc* blocks [[buffer(2)]],
                           constant uint& nBlocks [[buffer(3)]],
                           uint tgid [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint tgSize [[threads_per_threadgroup]]) {
        if (tgid >= nBlocks) return;
        BlockDesc d = blocks[tgid];
        uint n = min(d.srcLength, d.dstLength);
        for (uint i = tid; i < n; i += tgSize) dst[d.dstOffset + i] = src[d.srcOffset + i];
    }
    """
}
