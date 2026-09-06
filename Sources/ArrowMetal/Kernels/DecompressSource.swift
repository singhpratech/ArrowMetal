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
// One SIMD group (32 lanes) owns one page. Lane 0 walks the token stream; the parsed token is broadcast
// to the other 31 lanes with `simd_broadcast`, and all 32 lanes then move the token's bytes together.
// A threadgroup of 256 threads therefore decompresses 8 pages at once, and a dispatch of a few hundred
// threadgroups saturates the GPU. Lane 0 does the serial work of one page while its neighbours do the
// bulk memory traffic, so the serial part is only the ~10 instructions per token that parse a tag byte.
//
// Back-references that overlap the region they write (offset < length, which is how both formats encode
// runs) are still copied in parallel: the copy repeats a pattern of `offset` bytes, so output byte i is
// simply `out[dst - offset + (i % offset)]`, and every one of those source bytes was already written
// before this token began. `simdgroup_barrier(mem_flags::mem_device)` at the end of each token makes the
// previous token's stores visible to the lanes that read them next.
//
// Status codes are written per block so the host can turn corrupt input into an error instead of
// silently wrong data.
enum DecompressSource {
    static let source = KernelSource.prelude + """

    #define DEC_OK        0u
    #define DEC_OVERRUN   1u
    #define DEC_BAD_TOKEN 2u
    #define DEC_SHORT     3u

    // Descriptor per block: source offset/length and destination offset/length.
    struct BlockDesc { uint srcOffset; uint srcLength; uint dstOffset; uint dstLength; };

    // Cooperative move of `n` bytes. `fromOutput` selects a back-reference (pattern repeat) over a
    // literal run. Every lane of the SIMD group participates.
    inline void dec_move(device uchar* dst, device const uchar* src, uint dBase, uint op,
                         uint sOff, uint n, uint backOff, bool fromOutput, uint lane) {
        if (fromOutput) {
            uint m = backOff;
            for (uint i = lane; i < n; i += 32u) {
                dst[dBase + op + i] = dst[dBase + op - m + (i % m)];
            }
        } else {
            for (uint i = lane; i < n; i += 32u) {
                dst[dBase + op + i] = src[sOff + i];
            }
        }
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
                                  uint tgid [[threadgroup_position_in_grid]],
                                  uint sgid [[simdgroup_index_in_threadgroup]],
                                  uint sgCount [[simdgroups_per_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
        uint b = tgid * sgCount + sgid;
        if (b >= nBlocks) return;
        BlockDesc d = blocks[b];
        uint sBase = d.srcOffset, sLen = d.srcLength, dBase = d.dstOffset, dLen = d.dstLength;
        uint sp = 0u, op = 0u, err = DEC_OK;
        if (lane == 0u) {
            // Skip the varint preamble.
            uint shift = 0u;
            while (sp < sLen) { uchar c = src[sBase + sp]; sp++; shift += 7u; if ((c & 0x80u) == 0u) break; }
        }
        sp = simd_broadcast(sp, 0u);
        while (op < dLen && err == DEC_OK) {
            uint n = 0u, sOff = 0u, backOff = 0u, isCopy = 0u;
            if (lane == 0u) {
                if (sp >= sLen) { err = DEC_SHORT; }
                else {
                    uint tag = (uint)src[sBase + sp]; sp++;
                    uint t = tag & 3u;
                    if (t == 0u) {
                        uint len = tag >> 2;
                        if (len >= 60u) {
                            uint extra = len - 59u;
                            if (sp + extra > sLen) { err = DEC_SHORT; }
                            else {
                                uint v = 0u;
                                for (uint k = 0u; k < extra; k++) v |= ((uint)src[sBase + sp + k]) << (8u * k);
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
                            backOff = (((tag >> 5) & 7u) << 8) | (uint)src[sBase + sp];
                            sp += 1u;
                            isCopy = 1u;
                        }
                    } else {
                        uint w = (t == 2u) ? 2u : 4u;
                        if (sp + w > sLen) { err = DEC_SHORT; }
                        else {
                            n = 1u + (tag >> 2);
                            uint v = 0u;
                            for (uint k = 0u; k < w; k++) v |= ((uint)src[sBase + sp + k]) << (8u * k);
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
            }
            err = simd_broadcast(err, 0u);
            if (err != DEC_OK) break;
            n = simd_broadcast(n, 0u);
            sOff = simd_broadcast(sOff, 0u);
            backOff = simd_broadcast(backOff, 0u);
            isCopy = simd_broadcast(isCopy, 0u);
            sp = simd_broadcast(sp, 0u);
            dec_move(dst, src, dBase, op, sOff, n, backOff, isCopy != 0u, lane);
            simdgroup_barrier(mem_flags::mem_device);
            op += n;
        }
        if (lane == 0u) status[b] = err;
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
                               uint tgid [[threadgroup_position_in_grid]],
                               uint sgid [[simdgroup_index_in_threadgroup]],
                               uint sgCount [[simdgroups_per_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
        uint b = tgid * sgCount + sgid;
        if (b >= nBlocks) return;
        BlockDesc d = blocks[b];
        uint sBase = d.srcOffset, sLen = d.srcLength, dBase = d.dstOffset, dLen = d.dstLength;
        if (sLen >= 8u) {
            uint u = ((uint)src[sBase] << 24) | ((uint)src[sBase+1] << 16) | ((uint)src[sBase+2] << 8) | (uint)src[sBase+3];
            uint c = ((uint)src[sBase+4] << 24) | ((uint)src[sBase+5] << 16) | ((uint)src[sBase+6] << 8) | (uint)src[sBase+7];
            if (u == dLen && c == sLen - 8u) { sBase += 8u; sLen -= 8u; }
        }
        uint sp = 0u, op = 0u, err = DEC_OK;
        while (op < dLen && err == DEC_OK) {
            uint litLen = 0u, matchLen = 0u, backOff = 0u, litSrc = 0u, hasMatch = 0u;
            if (lane == 0u) {
                if (sp >= sLen) { err = DEC_SHORT; }
                else {
                    uint token = (uint)src[sBase + sp]; sp++;
                    litLen = token >> 4;
                    if (litLen == 15u) {
                        uint c2 = 255u;
                        while (c2 == 255u && sp < sLen) { c2 = (uint)src[sBase + sp]; sp++; litLen += c2; }
                    }
                    litSrc = sBase + sp;
                    sp += litLen;
                    if (sp > sLen || op + litLen > dLen) { err = DEC_OVERRUN; }
                    else if (sp + 2u <= sLen) {
                        backOff = (uint)src[sBase + sp] | ((uint)src[sBase + sp + 1] << 8);
                        sp += 2u;
                        matchLen = token & 15u;
                        if (matchLen == 15u) {
                            uint c2 = 255u;
                            while (c2 == 255u && sp < sLen) { c2 = (uint)src[sBase + sp]; sp++; matchLen += c2; }
                        }
                        matchLen += 4u;
                        hasMatch = 1u;
                        if (backOff == 0u || backOff > op + litLen || op + litLen + matchLen > dLen) err = DEC_BAD_TOKEN;
                    }
                }
            }
            err = simd_broadcast(err, 0u);
            if (err != DEC_OK) break;
            litLen = simd_broadcast(litLen, 0u);
            litSrc = simd_broadcast(litSrc, 0u);
            matchLen = simd_broadcast(matchLen, 0u);
            backOff = simd_broadcast(backOff, 0u);
            hasMatch = simd_broadcast(hasMatch, 0u);
            sp = simd_broadcast(sp, 0u);
            if (litLen > 0u) {
                dec_move(dst, src, dBase, op, litSrc, litLen, 0u, false, lane);
                simdgroup_barrier(mem_flags::mem_device);
                op += litLen;
            }
            if (hasMatch == 0u) break;
            dec_move(dst, src, dBase, op, 0u, matchLen, backOff, true, lane);
            simdgroup_barrier(mem_flags::mem_device);
            op += matchLen;
        }
        if (lane == 0u) status[b] = (op == dLen || err != DEC_OK) ? err : DEC_SHORT;
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
