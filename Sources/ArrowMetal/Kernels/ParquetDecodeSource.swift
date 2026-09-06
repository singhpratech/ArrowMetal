import Foundation

// MSL for decoding Parquet pages into Arrow buffers.
//
// THE PARALLEL RLE STRATEGY
//
// Definition levels, repetition levels and dictionary indices all use Parquet's "RLE / bit-packing
// hybrid": a sequence of runs, each introduced by a varint header whose low bit picks the run kind
// (0 = a repeated value, 1 = a group of 8*n bit-packed values). Run boundaries are only known after the
// previous header has been read, so the *headers* cannot be parsed in parallel — but the values inside
// them can, and that is where all the work is.
//
// One threadgroup (256 threads) owns one page and walks it in batches of up to 2048 values:
//
//   1. Thread 0 scans forward over run headers only — it never touches the packed value bytes — until it
//      has covered 2048 values or filled its table of 256 run records. Each record is (kind, value or
//      byte offset of the packed data, how many values of this run were already emitted, where this run
//      starts inside the batch). A run that spills past the batch is left half-consumed and picked up by
//      the next batch, so runs of any length work.
//   2. A threadgroup barrier publishes the table.
//   3. Every thread takes a contiguous slice of 8 batch positions, binary-searches the run table for the
//      run each position falls in (the table is tiny and in threadgroup memory, so the search is a few
//      cycles), and decodes its level: a constant for an RLE run, an LSB-first bit-field read for a
//      bit-packed one. This is where the bit unpacking happens and it is fully parallel.
//   4. The same pass scans "is this level the max definition level?" across the batch — 8 values serially
//      per thread, then `simd_prefix_exclusive_sum` across the 32 lanes of each SIMD group, then a
//      256-wide combine — which gives every row its *rank*: the index of its value in the page's dense,
//      null-free value section. Ranks are what let every later decoder be a pure random-access gather:
//      row i reads value `rank[i]`, and no compaction pass is needed anywhere.
//
// A page therefore costs one serial header walk (a few instructions per run) and everything else runs
// 256-wide, while thousands of pages run at once across the GPU.
enum ParquetDecodeSource {
    /// Shared declarations: the page descriptor, varint and bit-field readers, and the scan helpers.
    static let prelude = KernelSource.prelude + """

    #define PQ_TG      256u
    #define PQ_MAXRUNS 256u
    #define PQ_CHUNK   8u
    #define PQ_BATCH   2048u

    // Mirrors `ParquetPageInfo` in Swift: 18 x uint32.
    struct PageInfo {
        uint dataOffset;     // byte offset of the page body inside the page-data buffer
        uint dataLength;
        uint numValues;      // levels (== rows for a flat column) in this page
        uint levelOffset;    // index of this page's first level within the column chunk
        uint nonNullOffset;  // index of this page's first value in the chunk's dense value section
        uint nonNullCount;   // values that are not null
        uint repOffset;      // byte offsets/lengths of the three sections, filled by pq_page_layout
        uint repLength;
        uint defOffset;
        uint defLength;
        uint valuesOffset;
        uint valuesLength;
        uint encoding;       // ParquetEncoding raw value
        uint bitWidth;       // dictionary index width, read from the page by pq_page_layout
        uint flags;          // bit 0: data page v2
        uint numRows;        // rows in this page (differs from numValues only for repeated columns)
        uint dictBase;       // offset of this page's row group's dictionary in the merged dictionary
        uint pad0;
    };

    // LSB-first varint out of device memory.
    inline uint pq_varint(device const uchar* p, thread uint& pos, uint end) {
        uint result = 0u, shift = 0u;
        for (uint k = 0u; k < 5u; k++) {
            if (pos >= end) break;
            uchar c = p[pos]; pos++;
            result |= ((uint)(c & 0x7Fu)) << shift;
            if ((c & 0x80u) == 0u) break;
            shift += 7u;
        }
        return result;
    }
    inline ulong pq_varint64(device const uchar* p, thread uint& pos, uint end) {
        ulong result = 0ul; uint shift = 0u;
        for (uint k = 0u; k < 10u; k++) {
            if (pos >= end) break;
            uchar c = p[pos]; pos++;
            result |= ((ulong)(c & 0x7Fu)) << shift;
            if ((c & 0x80u) == 0u) break;
            shift += 7u;
        }
        return result;
    }
    inline long pq_zigzag(ulong u) { return (long)(u >> 1) ^ -(long)(u & 1ul); }

    inline uint pq_le32(device const uchar* p, uint at) {
        return (uint)p[at] | ((uint)p[at+1] << 8) | ((uint)p[at+2] << 16) | ((uint)p[at+3] << 24);
    }

    // Value `idx` of an LSB-first bit-packed run of `bw`-bit fields starting at byte `base`.
    inline uint pq_bp_get(device const uchar* p, uint base, uint idx, uint bw) {
        if (bw == 0u) return 0u;
        uint bit = idx * bw;
        uint at = base + (bit >> 3);
        uint sh = bit & 7u;
        uint nb = (sh + bw + 7u) >> 3;      // at most 5 for bw <= 32
        ulong v = 0ul;
        for (uint k = 0u; k < nb; k++) v |= ((ulong)p[at + k]) << (8u * k);
        v >>= sh;
        if (bw >= 32u) return (uint)v;
        return (uint)(v & ((1ul << bw) - 1ul));
    }
    // 64-bit variant (DELTA_BINARY_PACKED on int64 uses widths up to 64).
    inline ulong pq_bp_get64(device const uchar* p, uint base, uint idx, uint bw) {
        if (bw == 0u) return 0ul;
        ulong bit = (ulong)idx * (ulong)bw;
        uint at = base + (uint)(bit >> 3);
        uint sh = (uint)(bit & 7ul);
        uint nb = (sh + bw + 7u) >> 3;      // at most 9
        ulong lo = 0ul, hi = 0ul;
        uint nlo = min(nb, 8u);
        for (uint k = 0u; k < nlo; k++) lo |= ((ulong)p[at + k]) << (8u * k);
        if (nb > 8u) hi = (ulong)p[at + 8];
        ulong v = (sh == 0u) ? lo : ((lo >> sh) | (hi << (64u - sh)));
        if (bw >= 64u) return v;
        return v & ((1ul << bw) - 1ul);
    }

    // Exclusive prefix sum of `v` across a 256-thread threadgroup. Returns this thread's exclusive
    // prefix; `total` receives the sum over the whole threadgroup. `sg` is scratch of >= 32 uints.
    inline uint pq_tg_scan(uint v, threadgroup uint* sg, uint lane, uint sgid, uint nsg, thread uint& total) {
        uint sp = simd_prefix_exclusive_sum(v);
        if (lane == 31u) sg[sgid] = sp + v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint off = 0u, tot = 0u;
        for (uint s = 0u; s < nsg; s++) { uint x = sg[s]; tot += x; if (s < sgid) off += x; }
        total = tot;
        return off + sp;
    }
    """

    /// Page layout, level decoding, and the value decoders.
    static let source = prelude + """

    // ------------------------------------------------------------------ page layout
    //
    // A v1 data page stores <4-byte length><rep levels><4-byte length><def levels><values>, so the
    // section boundaries live *inside* the (possibly compressed) page and can only be found after
    // decompression -- which is why this runs on the GPU rather than on the host. A v2 page carries the
    // two lengths in its Thrift header, so the host has already filled them in.
    kernel void pq_page_layout(device const uchar* data [[buffer(0)]],
                               device PageInfo* pages [[buffer(1)]],
                               constant uint& nPages [[buffer(2)]],
                               constant uint& maxDef [[buffer(3)]],
                               constant uint& maxRep [[buffer(4)]],
                               uint i [[thread_position_in_grid]]) {
        if (i >= nPages) return;
        PageInfo p = pages[i];
        uint at = p.dataOffset, end = p.dataOffset + p.dataLength;
        if ((p.flags & 1u) != 0u) {
            // v2: rep levels, then def levels, both with lengths from the header.
            p.repOffset = at; at += p.repLength;
            p.defOffset = at; at += p.defLength;
        } else {
            if (maxRep > 0u) {
                p.repLength = (at + 4u <= end) ? pq_le32(data, at) : 0u;
                p.repOffset = at + 4u;
                at += 4u + p.repLength;
            } else { p.repOffset = at; p.repLength = 0u; }
            if (maxDef > 0u) {
                p.defLength = (at + 4u <= end) ? pq_le32(data, at) : 0u;
                p.defOffset = at + 4u;
                at += 4u + p.defLength;
            } else { p.defOffset = at; p.defLength = 0u; }
        }
        p.valuesOffset = at;
        p.valuesLength = (end > at) ? (end - at) : 0u;
        // RLE_DICTIONARY / PLAIN_DICTIONARY: a single byte holding the index bit width comes first.
        if (p.encoding == 8u || p.encoding == 2u) {
            if (p.valuesLength > 0u) {
                p.bitWidth = (uint)data[p.valuesOffset];
                p.valuesOffset += 1u;
                p.valuesLength -= 1u;
            }
        } else if (p.encoding == 3u) {
            // RLE-encoded booleans: a 4-byte little-endian length, then the hybrid stream.
            if (p.valuesLength >= 4u) {
                uint n = pq_le32(data, p.valuesOffset);
                p.valuesOffset += 4u;
                p.valuesLength = min(n, p.valuesLength - 4u);
            }
            p.bitWidth = 1u;
        }
        pages[i] = p;
    }

    // ------------------------------------------------------------------ level decoding
    //
    // `which` selects definition (0) or repetition (1) levels; `matchLevel` is the level whose
    // occurrences are counted and ranked; `countSlot` says where the per-page count lands
    // (0 = nonNullCount, 1 = numRows). `outRank` may alias a dummy buffer when ranks are not wanted.
    kernel void pq_decode_levels(device const uchar* data [[buffer(0)]],
                                 device PageInfo* pages [[buffer(1)]],
                                 constant uint& nPages [[buffer(2)]],
                                 constant uint& levelBitWidth [[buffer(3)]],
                                 constant uint& matchLevel [[buffer(4)]],
                                 constant uint& which [[buffer(5)]],
                                 constant uint& countSlot [[buffer(6)]],
                                 device uchar* outLevels [[buffer(7)]],
                                 device uint* outRank [[buffer(8)]],
                                 uint tgid [[threadgroup_position_in_grid]],
                                 uint tid [[thread_position_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sgid [[simdgroup_index_in_threadgroup]],
                                 uint nsg [[simdgroups_per_threadgroup]]) {
        threadgroup uint rKind[PQ_MAXRUNS], rVal[PQ_MAXRUNS], rSkip[PQ_MAXRUNS], rStart[PQ_MAXRUNS + 1u];
        threadgroup uint sg[32];
        threadgroup uint sPos, sSkip, sRuns, sBatch, sBase, sEmitted;

        if (tgid >= nPages) return;
        PageInfo pg = pages[tgid];
        uint bw = levelBitWidth;
        uint start = (which == 0u) ? pg.defOffset : pg.repOffset;
        uint len   = (which == 0u) ? pg.defLength : pg.repLength;
        uint end = start + len;

        if (tid == 0u) { sPos = start; sSkip = 0u; sBase = 0u; sEmitted = 0u; sRuns = 0u; sBatch = 0u; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        while (true) {
            if (tid == 0u) {
                uint pos = sPos, skip = sSkip, emitted = sEmitted;
                uint nr = 0u, batchN = 0u;
                uint cap = min(PQ_BATCH, pg.numValues - emitted);
                while (batchN < cap && nr < PQ_MAXRUNS && pos < end) {
                    uint hpos = pos;
                    uint header = pq_varint(data, pos, end);
                    uint kind, total, val;
                    if ((header & 1u) != 0u) {
                        kind = 1u;
                        total = (header >> 1) * 8u;
                        val = pos;
                        pos += (total * bw + 7u) >> 3;
                    } else {
                        kind = 0u;
                        total = header >> 1;
                        uint nb = (bw + 7u) >> 3;
                        uint v = 0u;
                        for (uint k = 0u; k < nb && pos + k < end; k++) v |= ((uint)data[pos + k]) << (8u * k);
                        val = v;
                        pos += nb;
                    }
                    if (total == 0u || total <= skip) { skip = 0u; continue; }
                    uint avail = total - skip;
                    uint take = min(avail, cap - batchN);
                    rKind[nr] = kind; rVal[nr] = val; rSkip[nr] = skip; rStart[nr] = batchN;
                    nr++; batchN += take;
                    if (take < avail) { skip += take; pos = hpos; break; }   // resume mid-run next batch
                    skip = 0u;
                }
                sPos = pos; sSkip = skip;
                rStart[nr] = batchN;
                sRuns = nr; sBatch = batchN;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint bn = sBatch, nr = sRuns, base = sBase, emitted = sEmitted;
            if (bn == 0u) break;

            uint localCount = 0u;
            uchar mine[PQ_CHUNK];
            for (uint c = 0u; c < PQ_CHUNK; c++) {
                uint k = tid * PQ_CHUNK + c;
                uchar v = 0u;
                if (k < bn) {
                    uint lo = 0u, hi = nr - 1u;
                    while (lo < hi) { uint mid = (lo + hi + 1u) >> 1; if (rStart[mid] <= k) lo = mid; else hi = mid - 1u; }
                    uint idx = rSkip[lo] + (k - rStart[lo]);
                    uint value = (rKind[lo] != 0u) ? pq_bp_get(data, rVal[lo], idx, bw) : rVal[lo];
                    v = (uchar)value;
                    if (value == matchLevel) localCount++;
                }
                mine[c] = v;
            }
            uint total = 0u;
            uint myBase = base + pq_tg_scan(localCount, sg, lane, sgid, nsg, total);

            uint run = 0u;
            for (uint c = 0u; c < PQ_CHUNK; c++) {
                uint k = tid * PQ_CHUNK + c;
                if (k < bn) {
                    uint gi = pg.levelOffset + emitted + k;
                    outLevels[gi] = mine[c];
                    outRank[gi] = myBase + run;
                    if ((uint)mine[c] == matchLevel) run++;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) { sBase = base + total; sEmitted = emitted + bn; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sEmitted >= pg.numValues) break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            if (countSlot == 0u) pages[tgid].nonNullCount = sBase;
            else if (countSlot == 1u) pages[tgid].numRows = sBase;
        }
    }

    // Dictionary indices (and RLE booleans): the same hybrid stream, decoded straight into the page's
    // dense value slots. No ranking is needed here -- the values are already dense.
    kernel void pq_decode_rle_values(device const uchar* data [[buffer(0)]],
                                     device const PageInfo* pages [[buffer(1)]],
                                     constant uint& nPages [[buffer(2)]],
                                     device uint* out [[buffer(3)]],
                                     uint tgid [[threadgroup_position_in_grid]],
                                     uint tid [[thread_position_in_threadgroup]]) {
        threadgroup uint rKind[PQ_MAXRUNS], rVal[PQ_MAXRUNS], rSkip[PQ_MAXRUNS], rStart[PQ_MAXRUNS + 1u];
        threadgroup uint sPos, sSkip, sRuns, sBatch, sEmitted;

        if (tgid >= nPages) return;
        PageInfo pg = pages[tgid];
        uint bw = pg.bitWidth;
        uint start = pg.valuesOffset, end = pg.valuesOffset + pg.valuesLength;
        uint count = pg.nonNullCount;

        if (tid == 0u) { sPos = start; sSkip = 0u; sEmitted = 0u; sRuns = 0u; sBatch = 0u; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        while (true) {
            if (tid == 0u) {
                uint pos = sPos, skip = sSkip, emitted = sEmitted;
                uint nr = 0u, batchN = 0u;
                uint cap = min(PQ_BATCH, count - emitted);
                while (batchN < cap && nr < PQ_MAXRUNS && pos < end) {
                    uint hpos = pos;
                    uint header = pq_varint(data, pos, end);
                    uint kind, total, val;
                    if ((header & 1u) != 0u) {
                        kind = 1u; total = (header >> 1) * 8u; val = pos;
                        pos += (total * bw + 7u) >> 3;
                    } else {
                        kind = 0u; total = header >> 1;
                        uint nb = (bw + 7u) >> 3;
                        uint v = 0u;
                        for (uint k = 0u; k < nb && pos + k < end; k++) v |= ((uint)data[pos + k]) << (8u * k);
                        val = v; pos += nb;
                    }
                    if (total == 0u || total <= skip) { skip = 0u; continue; }
                    uint avail = total - skip;
                    uint take = min(avail, cap - batchN);
                    rKind[nr] = kind; rVal[nr] = val; rSkip[nr] = skip; rStart[nr] = batchN;
                    nr++; batchN += take;
                    if (take < avail) { skip += take; pos = hpos; break; }
                    skip = 0u;
                }
                sPos = pos; sSkip = skip;
                rStart[nr] = batchN;
                sRuns = nr; sBatch = batchN;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint bn = sBatch, nr = sRuns, emitted = sEmitted;
            if (bn == 0u) break;
            for (uint k = tid; k < bn; k += PQ_TG) {
                uint lo = 0u, hi = nr - 1u;
                while (lo < hi) { uint mid = (lo + hi + 1u) >> 1; if (rStart[mid] <= k) lo = mid; else hi = mid - 1u; }
                uint idx = rSkip[lo] + (k - rStart[lo]);
                uint value = (rKind[lo] != 0u) ? pq_bp_get(data, rVal[lo], idx, bw) : rVal[lo];
                out[pg.nonNullOffset + emitted + k] = value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) sEmitted = emitted + bn;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sEmitted >= count) break;
        }
    }

    // ------------------------------------------------------------------ per-page prefix sums
    //
    // Turns per-page non-null counts into the offsets of each page's slice of the chunk's dense value
    // section. One threadgroup, chunked, so it works for any number of pages.
    kernel void pq_page_scan(device PageInfo* pages [[buffer(0)]],
                             constant uint& nPages [[buffer(1)]],
                             constant uint& slot [[buffer(2)]],
                             device uint* grandTotal [[buffer(3)]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sgid [[simdgroup_index_in_threadgroup]],
                             uint nsg [[simdgroups_per_threadgroup]]) {
        threadgroup uint sg[32];
        threadgroup uint sBase;
        if (tid == 0u) sBase = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint start = 0u; start < nPages; start += PQ_TG) {
            uint k = start + tid;
            uint v = 0u;
            if (k < nPages) v = (slot == 0u) ? pages[k].nonNullCount : pages[k].numRows;
            uint total = 0u;
            uint pre = pq_tg_scan(v, sg, lane, sgid, nsg, total);
            uint base = sBase;
            if (k < nPages) {
                if (slot == 0u) pages[k].nonNullOffset = base + pre;
                else pages[k].levelOffset = base + pre;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) sBase = base + total;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) grandTotal[0] = sBase;
    }

    // ------------------------------------------------------------------ generic uint32 prefix sum
    kernel void pq_scan_block(device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]],
                              device uint* blockSums [[buffer(2)]], constant uint& n [[buffer(3)]],
                              uint tgid [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sgid [[simdgroup_index_in_threadgroup]],
                              uint nsg [[simdgroups_per_threadgroup]]) {
        threadgroup uint sg[32];
        uint base = tgid * (PQ_TG * 4u) + tid * 4u;
        uint v[4];
        for (uint c = 0u; c < 4u; c++) { uint k = base + c; v[c] = (k < n) ? in[k] : 0u; }
        uint local = v[0] + v[1] + v[2] + v[3];
        uint total = 0u;
        uint run = pq_tg_scan(local, sg, lane, sgid, nsg, total);
        for (uint c = 0u; c < 4u; c++) { uint k = base + c; if (k < n) out[k] = run; run += v[c]; }
        if (tid == 0u) blockSums[tgid] = total;
    }
    kernel void pq_scan_sums(device uint* blockSums [[buffer(0)]], constant uint& nBlocks [[buffer(1)]],
                             device uint* grandTotal [[buffer(2)]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sgid [[simdgroup_index_in_threadgroup]],
                             uint nsg [[simdgroups_per_threadgroup]]) {
        threadgroup uint sg[32];
        threadgroup uint sBase;
        if (tid == 0u) sBase = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint start = 0u; start < nBlocks; start += PQ_TG) {
            uint k = start + tid;
            uint v = (k < nBlocks) ? blockSums[k] : 0u;
            uint total = 0u;
            uint pre = pq_tg_scan(v, sg, lane, sgid, nsg, total);
            uint base = sBase;
            if (k < nBlocks) blockSums[k] = base + pre;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) sBase = base + total;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) grandTotal[0] = sBase;
    }
    kernel void pq_scan_add(device uint* out [[buffer(0)]], device const uint* blockSums [[buffer(1)]],
                            constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        out[i] += blockSums[i / (PQ_TG * 4u)];
    }


    // ------------------------------------------------------------------ value decoders
    //
    // Every value decoder writes the page's values *densely*: value j of a page lands at
    // `nonNullOffset + j`, with no gaps for nulls. One `pq_scatter` at the end moves the dense values
    // into their row positions using the ranks the level decoder produced. Keeping the two apart means
    // each encoding needs exactly one kernel, mixed-encoding chunks (the dictionary pages plus the PLAIN
    // fallback pages a writer emits when a dictionary grows too large) just run two kernels over two
    // slices of the page list, and a column with no nulls skips the scatter entirely because dense
    // positions and row positions coincide.

    // PLAIN fixed-width: the page's value section already *is* the dense array, so this is a copy.
    kernel void pq_plain_fixed(device const uchar* data [[buffer(0)]],
                               device const PageInfo* pages [[buffer(1)]],
                               constant uint& nPages [[buffer(2)]],
                               constant uint& width [[buffer(3)]],
                               device uchar* out [[buffer(4)]],
                               uint tgid [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]]) {
        if (tgid >= nPages) return;
        PageInfo pg = pages[tgid];
        uint n = pg.nonNullCount * width;
        uint src = pg.valuesOffset, dst = pg.nonNullOffset * width;
        for (uint i = tid; i < n; i += PQ_TG) out[dst + i] = data[src + i];
    }

    // PLAIN booleans: one bit per value, LSB first, expanded to one byte per value.
    kernel void pq_plain_bool(device const uchar* data [[buffer(0)]],
                              device const PageInfo* pages [[buffer(1)]],
                              constant uint& nPages [[buffer(2)]],
                              device uchar* out [[buffer(3)]],
                              uint tgid [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]]) {
        if (tgid >= nPages) return;
        PageInfo pg = pages[tgid];
        for (uint j = tid; j < pg.nonNullCount; j += PQ_TG) {
            out[pg.nonNullOffset + j] = (data[pg.valuesOffset + (j >> 3)] >> (j & 7u)) & 1u;
        }
    }

    // BYTE_STREAM_SPLIT: `width` planes of n bytes, so byte k of value j sits at plane k, slot j.
    kernel void pq_byte_stream_split(device const uchar* data [[buffer(0)]],
                                     device const PageInfo* pages [[buffer(1)]],
                                     constant uint& nPages [[buffer(2)]],
                                     constant uint& width [[buffer(3)]],
                                     device uchar* out [[buffer(4)]],
                                     uint tgid [[threadgroup_position_in_grid]],
                                     uint tid [[thread_position_in_threadgroup]]) {
        if (tgid >= nPages) return;
        PageInfo pg = pages[tgid];
        uint n = pg.nonNullCount;
        for (uint j = tid; j < n; j += PQ_TG) {
            uint dst = (pg.nonNullOffset + j) * width;
            for (uint k = 0u; k < width; k++) out[dst + k] = data[pg.valuesOffset + k * n + j];
        }
    }

    // Materialises dictionary-encoded fixed-width values: dense[v] = dictionary[codes[v]].
    kernel void pq_dict_gather_fixed(device const uchar* dict [[buffer(0)]],
                                     device const uint* codes [[buffer(1)]],
                                     device const PageInfo* pages [[buffer(2)]],
                                     constant uint& nPages [[buffer(3)]],
                                     constant uint& width [[buffer(4)]],
                                     constant uint& dictCount [[buffer(5)]],
                                     device uchar* out [[buffer(6)]],
                                     uint tgid [[threadgroup_position_in_grid]],
                                     uint tid [[thread_position_in_threadgroup]]) {
        if (tgid >= nPages) return;
        PageInfo pg = pages[tgid];
        for (uint j = tid; j < pg.nonNullCount; j += PQ_TG) {
            uint v = pg.nonNullOffset + j;
            uint c = codes[v];
            uint src = (c < dictCount ? c : 0u) * width;
            uint dst = v * width;
            for (uint k = 0u; k < width; k++) out[dst + k] = dict[src + k];
        }
    }

    // Adds each page's dictionary base to its codes, so several row groups (each with its own dictionary
    // page) can share one concatenated dictionary and one code array.
    kernel void pq_dict_rebase(device uint* codes [[buffer(0)]],
                               device const PageInfo* pages [[buffer(1)]],
                               constant uint& nPages [[buffer(2)]],
                               uint tgid [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]]) {
        if (tgid >= nPages) return;
        PageInfo pg = pages[tgid];
        if (pg.dictBase == 0u) return;
        for (uint j = tid; j < pg.nonNullCount; j += PQ_TG) codes[pg.nonNullOffset + j] += pg.dictBase;
    }

    // Dense values -> row positions, using the ranks from the level decoder. Null rows keep whatever the
    // output buffer already held (Arrow leaves them undefined).
    kernel void pq_scatter(device const uchar* dense [[buffer(0)]],
                           device const PageInfo* pages [[buffer(1)]],
                           constant uint& nPages [[buffer(2)]],
                           constant uint& width [[buffer(3)]],
                           constant uint& maxDef [[buffer(4)]],
                           device const uchar* defLevels [[buffer(5)]],
                           device const uint* ranks [[buffer(6)]],
                           device uchar* out [[buffer(7)]],
                           uint tgid [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]]) {
        if (tgid >= nPages) return;
        PageInfo pg = pages[tgid];
        for (uint j = tid; j < pg.numValues; j += PQ_TG) {
            uint gi = pg.levelOffset + j;
            if ((uint)defLevels[gi] != maxDef) continue;
            uint src = (pg.nonNullOffset + ranks[gi]) * width;
            uint dst = gi * width;
            for (uint k = 0u; k < width; k++) out[dst + k] = dense[src + k];
        }
    }

    // Validity bitmap from definition levels: 32 rows per thread.
    kernel void pq_levels_to_bitmap(device const uchar* defLevels [[buffer(0)]],
                                    constant uint& n [[buffer(1)]],
                                    constant uint& maxDef [[buffer(2)]],
                                    device uint* out [[buffer(3)]],
                                    uint w [[thread_position_in_grid]]) {
        uint base = w * 32u;
        if (base >= n) return;
        uint limit = min(32u, n - base), bits = 0u;
        for (uint j = 0u; j < limit; j++) if ((uint)defLevels[base + j] == maxDef) bits |= (1u << j);
        out[w] = bits;
    }

    // Bytes (0/1) -> packed bitmap, for boolean value buffers.
    kernel void pq_bytes_to_bitmap(device const uchar* bytes [[buffer(0)]],
                                   constant uint& n [[buffer(1)]],
                                   device uint* out [[buffer(2)]],
                                   uint w [[thread_position_in_grid]]) {
        uint base = w * 32u;
        if (base >= n) return;
        uint limit = min(32u, n - base), bits = 0u;
        for (uint j = 0u; j < limit; j++) if (bytes[base + j] != 0u) bits |= (1u << j);
        out[w] = bits;
    }

    // ------------------------------------------------------------------ BYTE_ARRAY
    //
    // The variable-length path never copies a value twice: each encoding only has to fill, for every
    // dense value slot, *where its bytes are* and *how many* -- always as offsets into the same page-data
    // buffer. The offsets buffer then comes from one prefix sum over per-row lengths, and one gather
    // moves every byte of the column in parallel.

    // PLAIN: <4-byte little-endian length><bytes>, back to back, so the walk is sequential per page.
    kernel void pq_plain_bytes_scan(device const uchar* data [[buffer(0)]],
                                    device const PageInfo* pages [[buffer(1)]],
                                    constant uint& nPages [[buffer(2)]],
                                    device uint* valOffset [[buffer(3)]],
                                    device uint* valLength [[buffer(4)]],
                                    uint i [[thread_position_in_grid]]) {
        if (i >= nPages) return;
        PageInfo pg = pages[i];
        uint at = pg.valuesOffset, end = pg.valuesOffset + pg.valuesLength;
        for (uint j = 0u; j < pg.nonNullCount; j++) {
            if (at + 4u > end) { valOffset[pg.nonNullOffset + j] = at; valLength[pg.nonNullOffset + j] = 0u; continue; }
            uint L = pq_le32(data, at);
            at += 4u;
            valOffset[pg.nonNullOffset + j] = at;
            valLength[pg.nonNullOffset + j] = L;
            at += L;
        }
    }

    // FIXED_LEN_BYTE_ARRAY seen as variable-length: constant length, computable position.
    kernel void pq_flba_bytes_map(device const PageInfo* pages [[buffer(0)]],
                                  constant uint& nPages [[buffer(1)]],
                                  constant uint& width [[buffer(2)]],
                                  device uint* valOffset [[buffer(3)]],
                                  device uint* valLength [[buffer(4)]],
                                  uint tgid [[threadgroup_position_in_grid]],
                                  uint tid [[thread_position_in_threadgroup]]) {
        if (tgid >= nPages) return;
        PageInfo pg = pages[tgid];
        for (uint j = tid; j < pg.nonNullCount; j += PQ_TG) {
            valOffset[pg.nonNullOffset + j] = pg.valuesOffset + j * width;
            valLength[pg.nonNullOffset + j] = width;
        }
    }

    // Dictionary-encoded byte arrays: point each dense slot at the dictionary entry its code selects.
    kernel void pq_dict_bytes_map(device const uint* codes [[buffer(0)]],
                                  device const uint* dictOffset [[buffer(1)]],
                                  device const uint* dictLength [[buffer(2)]],
                                  device const PageInfo* pages [[buffer(3)]],
                                  constant uint& nPages [[buffer(4)]],
                                  constant uint& dictCount [[buffer(5)]],
                                  device uint* valOffset [[buffer(6)]],
                                  device uint* valLength [[buffer(7)]],
                                  uint tgid [[threadgroup_position_in_grid]],
                                  uint tid [[thread_position_in_threadgroup]]) {
        if (tgid >= nPages) return;
        PageInfo pg = pages[tgid];
        for (uint j = tid; j < pg.nonNullCount; j += PQ_TG) {
            uint v = pg.nonNullOffset + j;
            uint c = codes[v];
            if (c >= dictCount) c = 0u;
            valOffset[v] = dictOffset[c];
            valLength[v] = dictLength[c];
        }
    }

    // Per-row byte length (0 for nulls): the input to the offsets prefix sum.
    kernel void pq_row_lengths(device const PageInfo* pages [[buffer(0)]],
                               constant uint& nPages [[buffer(1)]],
                               constant uint& hasDef [[buffer(2)]],
                               constant uint& maxDef [[buffer(3)]],
                               device const uchar* defLevels [[buffer(4)]],
                               device const uint* ranks [[buffer(5)]],
                               device const uint* valLength [[buffer(6)]],
                               device uint* out [[buffer(7)]],
                               uint tgid [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]]) {
        if (tgid >= nPages) return;
        PageInfo pg = pages[tgid];
        for (uint j = tid; j < pg.numValues; j += PQ_TG) {
            uint gi = pg.levelOffset + j;
            uint v = j;
            if (hasDef != 0u) {
                if ((uint)defLevels[gi] != maxDef) { out[gi] = 0u; continue; }
                v = ranks[gi];
            }
            out[gi] = valLength[pg.nonNullOffset + v];
        }
    }

    // Moves the bytes: one row per thread, sources from the per-value table, destinations from the scan.
    kernel void pq_gather_bytes(device const uchar* data [[buffer(0)]],
                                device const PageInfo* pages [[buffer(1)]],
                                constant uint& nPages [[buffer(2)]],
                                constant uint& hasDef [[buffer(3)]],
                                constant uint& maxDef [[buffer(4)]],
                                device const uchar* defLevels [[buffer(5)]],
                                device const uint* ranks [[buffer(6)]],
                                device const uint* valOffset [[buffer(7)]],
                                device const uint* valLength [[buffer(8)]],
                                device const uint* offsets [[buffer(9)]],
                                device uchar* out [[buffer(10)]],
                                uint tgid [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]]) {
        if (tgid >= nPages) return;
        PageInfo pg = pages[tgid];
        for (uint j = tid; j < pg.numValues; j += PQ_TG) {
            uint gi = pg.levelOffset + j;
            uint v = j;
            if (hasDef != 0u) {
                if ((uint)defLevels[gi] != maxDef) continue;
                v = ranks[gi];
            }
            uint src = valOffset[pg.nonNullOffset + v];
            uint n = valLength[pg.nonNullOffset + v];
            uint dst = offsets[gi];
            for (uint k = 0u; k < n; k++) out[dst + k] = data[src + k];
        }
    }

    // ------------------------------------------------------------------ dictionary pages
    //
    // A dictionary page is PLAIN-encoded, so a BYTE_ARRAY dictionary is the same sequential walk (once
    // per row group, over a few thousand values) and a fixed-width dictionary is a straight copy.
    kernel void pq_dict_bytes_scan(device const uchar* data [[buffer(0)]],
                                   constant uint& start [[buffer(1)]],
                                   constant uint& end [[buffer(2)]],
                                   constant uint& count [[buffer(3)]],
                                   constant uint& base [[buffer(4)]],
                                   device uint* valOffset [[buffer(5)]],
                                   device uint* valLength [[buffer(6)]],
                                   uint i [[thread_position_in_grid]]) {
        if (i != 0u) return;
        uint at = start;
        for (uint j = 0u; j < count; j++) {
            if (at + 4u > end) { valOffset[base + j] = at; valLength[base + j] = 0u; continue; }
            uint L = pq_le32(data, at);
            at += 4u;
            valOffset[base + j] = at; valLength[base + j] = L;
            at += L;
        }
    }
    kernel void pq_dict_fixed(device const uchar* data [[buffer(0)]],
                              constant uint& start [[buffer(1)]],
                              constant uint& nBytes [[buffer(2)]],
                              constant uint& dstStart [[buffer(3)]],
                              device uchar* out [[buffer(4)]],
                              uint i [[thread_position_in_grid]]) {
        if (i >= nBytes) return;
        out[dstStart + i] = data[start + i];
    }

    // ------------------------------------------------------------------ type fix-ups

    kernel void pq_u32_to_i32(device const uint* in [[buffer(0)]], device int* out [[buffer(1)]],
                              constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i < n) out[i] = (int)in[i];
    }

    // INT96 (12 bytes: 8 bytes of nanoseconds within the day, then a 4-byte Julian day) to an Arrow
    // timestamp in nanoseconds since the Unix epoch. Julian day 2440588 is 1970-01-01.
    kernel void pq_int96_to_ns(device const uchar* in [[buffer(0)]], device long* out [[buffer(1)]],
                               constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        uint at = i * 12u;
        ulong nanos = 0ul;
        for (uint k = 0u; k < 8u; k++) nanos |= ((ulong)in[at + k]) << (8u * k);
        uint julian = (uint)in[at + 8] | ((uint)in[at + 9] << 8) | ((uint)in[at + 10] << 16) | ((uint)in[at + 11] << 24);
        long days = (long)julian - 2440588L;
        out[i] = days * 86400000000000L + (long)nanos;
    }

    // A big-endian, sign-extended FIXED_LEN_BYTE_ARRAY of `width` bytes to a little-endian 16-byte
    // two's-complement decimal128 value.
    kernel void pq_flba_to_decimal128(device const uchar* in [[buffer(0)]], device uchar* out [[buffer(1)]],
                                      constant uint& n [[buffer(2)]], constant uint& width [[buffer(3)]],
                                      uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        uint src = i * width, dst = i * 16u;
        uchar fill = (width > 0u && (in[src] & 0x80u) != 0u) ? 0xFFu : 0x00u;
        for (uint k = 0u; k < 16u; k++) {
            out[dst + k] = (k < width) ? in[src + width - 1u - k] : fill;
        }
    }
    // int32 / int64 decimal storage widened to decimal128.
    kernel void pq_int_to_decimal128(device const uchar* in [[buffer(0)]], device uchar* out [[buffer(1)]],
                                     constant uint& n [[buffer(2)]], constant uint& width [[buffer(3)]],
                                     uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        uint src = i * width, dst = i * 16u;
        uchar fill = (in[src + width - 1u] & 0x80u) != 0u ? 0xFFu : 0x00u;
        for (uint k = 0u; k < 16u; k++) out[dst + k] = (k < width) ? in[src + k] : fill;
    }

    // ------------------------------------------------------------------ DELTA_BINARY_PACKED
    //
    // Header: block size, miniblocks per block, total values, first value. Then blocks of
    // <min delta><one bit width per miniblock><bit-packed deltas>. A value is the running sum of
    // (min delta + unpacked delta), which is the format's only serial dependency.
    //
    // One threadgroup per page. Thread 0 parses block headers -- a handful of bytes each, and it never
    // touches the packed delta bytes -- until it has covered 1024 values; every thread then unpacks its
    // share of those deltas in parallel, a log-step threadgroup scan turns them into values, and a
    // running total carries across iterations. `endPos` reports where the stream ended, which is how the
    // two delta-based byte-array encodings find the section that follows.
    #define PQ_DELTA_VALUES 1024u

    inline void pq_scan_long(threadgroup long* s, uint tid, uint n) {
        for (uint d = 1u; d < n; d <<= 1) {
            long v = (tid >= d && tid < n) ? s[tid - d] : 0L;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid >= d && tid < n) s[tid] += v;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    kernel void pq_delta_binary_packed(device const uchar* data [[buffer(0)]],
                                       device const PageInfo* pages [[buffer(1)]],
                                       constant uint& nPages [[buffer(2)]],
                                       constant uint& width [[buffer(3)]],
                                       device const uint* startPos [[buffer(4)]],
                                       device uchar* out [[buffer(5)]],
                                       device uint* endPos [[buffer(6)]],
                                       uint tgid [[threadgroup_position_in_grid]],
                                       uint tid [[thread_position_in_threadgroup]]) {
        threadgroup long deltas[PQ_DELTA_VALUES];
        threadgroup long sums[PQ_TG];
        threadgroup long sMinDelta[16];
        threadgroup uint sMiniOff[64], sMiniWidth[64], sBlockOfMini[64];
        threadgroup uint sPos, sNMini, sValuesInIter;
        threadgroup long sRunning;
        threadgroup uint sEmitted, sBlockSize, sMiniPerBlock, sMiniValues, sTotal;

        if (tgid >= nPages) return;
        PageInfo pg = pages[tgid];
        uint end = pg.valuesOffset + pg.valuesLength;

        if (tid == 0u) {
            uint pos = startPos[tgid];
            sBlockSize = pq_varint(data, pos, end);
            sMiniPerBlock = pq_varint(data, pos, end);
            sTotal = min(pq_varint(data, pos, end), pg.nonNullCount);
            sRunning = pq_zigzag(pq_varint64(data, pos, end));
            sMiniValues = (sMiniPerBlock > 0u) ? (sBlockSize / sMiniPerBlock) : 32u;
            sPos = pos;
            sEmitted = 0u;
            if (sTotal > 0u) {
                uint d = pg.nonNullOffset * width;
                long v = sRunning;
                for (uint k = 0u; k < width; k++) out[d + k] = (uchar)((v >> (8u * k)) & 0xFFL);
                sEmitted = 1u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        while (sEmitted < sTotal) {
            if (tid == 0u) {
                uint pos = sPos;
                uint nMini = 0u, values = 0u, blocks = 0u;
                uint want = min(PQ_DELTA_VALUES, sTotal - sEmitted);
                while (values < want && nMini + sMiniPerBlock <= 64u && blocks < 16u && pos < end) {
                    sMinDelta[blocks] = pq_zigzag(pq_varint64(data, pos, end));
                    uint wpos = pos;
                    pos += sMiniPerBlock;
                    for (uint m = 0u; m < sMiniPerBlock; m++) {
                        uint bw = (wpos + m < end) ? (uint)data[wpos + m] : 0u;
                        sMiniWidth[nMini] = bw;
                        sMiniOff[nMini] = pos;
                        sBlockOfMini[nMini] = blocks;
                        nMini++;
                        pos += (sMiniValues * bw + 7u) >> 3;
                        values += sMiniValues;
                    }
                    blocks++;
                }
                sPos = pos;
                sNMini = nMini;
                sValuesInIter = min(values, sTotal - sEmitted);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint nv = sValuesInIter, nMini = sNMini, emitted = sEmitted;
            if (nv == 0u || nMini == 0u) break;

            for (uint i = tid; i < nv; i += PQ_TG) {
                uint m = min(i / sMiniValues, nMini - 1u);
                uint idx = i % sMiniValues;
                long d = (long)pq_bp_get64(data, sMiniOff[m], idx, sMiniWidth[m]);
                deltas[i] = d + sMinDelta[sBlockOfMini[m]];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint per = (nv + PQ_TG - 1u) / PQ_TG;
            uint lo = min(tid * per, nv), hi = min(lo + per, nv);
            long acc = 0L;
            for (uint i = lo; i < hi; i++) { acc += deltas[i]; deltas[i] = acc; }
            sums[tid] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            pq_scan_long(sums, tid, PQ_TG);
            long before = (tid == 0u) ? 0L : sums[tid - 1u];
            long running = sRunning;
            for (uint i = lo; i < hi; i++) {
                long v = running + before + deltas[i];
                uint d = (pg.nonNullOffset + emitted + i) * width;
                for (uint k = 0u; k < width; k++) out[d + k] = (uchar)((v >> (8u * k)) & 0xFFL);
            }
            long grand = sums[PQ_TG - 1u];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) { sRunning = running + grand; sEmitted = emitted + nv; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) endPos[tgid] = sPos;
    }

    // DELTA_LENGTH_BYTE_ARRAY: delta-packed lengths followed by every byte, back to back. `denseStart`
    // is the running byte position of each value across the whole dense array, so subtracting the page's
    // own first entry gives the value's offset inside the page's byte region, which starts at `endPos`.
    kernel void pq_delta_lengths_to_table(device const uint* lengths [[buffer(0)]],
                                          device const uint* denseStart [[buffer(1)]],
                                          device const uint* endPos [[buffer(2)]],
                                          device const PageInfo* pages [[buffer(3)]],
                                          constant uint& nPages [[buffer(4)]],
                                          device uint* valOffset [[buffer(5)]],
                                          device uint* valLength [[buffer(6)]],
                                          uint tgid [[threadgroup_position_in_grid]],
                                          uint tid [[thread_position_in_threadgroup]]) {
        if (tgid >= nPages) return;
        PageInfo pg = pages[tgid];
        if (pg.nonNullCount == 0u) return;
        uint pageStart = denseStart[pg.nonNullOffset];
        uint bytesBase = endPos[tgid];
        for (uint j = tid; j < pg.nonNullCount; j += PQ_TG) {
            uint v = pg.nonNullOffset + j;
            valOffset[v] = bytesBase + (denseStart[v] - pageStart);
            valLength[v] = lengths[v];
        }
    }

    // DELTA_BYTE_ARRAY: value i is the first `prefix[i]` bytes of value i-1 followed by its own suffix.
    // The dependency is real, so one threadgroup walks its page's values in order while all 256 of its
    // threads move the bytes of the value in hand; pages still run in parallel.
    kernel void pq_delta_byte_array(device const uchar* data [[buffer(0)]],
                                    device const uint* prefixLen [[buffer(1)]],
                                    device const uint* suffixLen [[buffer(2)]],
                                    device const uint* suffixStart [[buffer(3)]],
                                    device const uint* endPos [[buffer(4)]],
                                    device const PageInfo* pages [[buffer(5)]],
                                    constant uint& nPages [[buffer(6)]],
                                    device const uint* denseOffsets [[buffer(7)]],
                                    device uchar* out [[buffer(8)]],
                                    uint tgid [[threadgroup_position_in_grid]],
                                    uint tid [[thread_position_in_threadgroup]]) {
        if (tgid >= nPages) return;
        PageInfo pg = pages[tgid];
        if (pg.nonNullCount == 0u) return;
        uint suffixBase = suffixStart[pg.nonNullOffset];
        uint bytesBase = endPos[tgid];
        for (uint j = 0u; j < pg.nonNullCount; j++) {
            uint v = pg.nonNullOffset + j;
            uint dst = denseOffsets[v];
            uint p = prefixLen[v], s = suffixLen[v];
            uint prev = (j == 0u) ? dst : denseOffsets[v - 1u];
            for (uint k = tid; k < p; k += PQ_TG) out[dst + k] = out[prev + k];
            uint src = bytesBase + (suffixStart[v] - suffixBase);
            for (uint k = tid; k < s; k += PQ_TG) out[dst + p + k] = data[src + k];
            threadgroup_barrier(mem_flags::mem_device);
        }
    }

    // Element-wise sum, for total = prefix + suffix lengths.
    kernel void pq_add_u32(device uint* a [[buffer(0)]], device const uint* b [[buffer(1)]],
                           constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i < n) a[i] += b[i];
    }
    // A dense value table over a freshly built dense byte buffer.
    kernel void pq_dense_bytes_table(device const uint* offsets [[buffer(0)]],
                                     device const uint* lengths [[buffer(1)]],
                                     constant uint& n [[buffer(2)]],
                                     device uint* valOffset [[buffer(3)]],
                                     device uint* valLength [[buffer(4)]],
                                     uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        valOffset[i] = offsets[i];
        valLength[i] = lengths[i];
    }

    // ------------------------------------------------------------------ flat helpers

    kernel void pq_copy_u32(device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]],
                            constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i < n) out[i] = in[i];
    }

    // uint32 values -> one byte each, over one subset of pages (RLE-encoded booleans).
    kernel void pq_narrow_u32_pages(device const uint* in [[buffer(0)]],
                                    device const PageInfo* pages [[buffer(1)]],
                                    constant uint& nPages [[buffer(2)]],
                                    device uchar* out [[buffer(3)]],
                                    uint tgid [[threadgroup_position_in_grid]],
                                    uint tid [[thread_position_in_threadgroup]]) {
        if (tgid >= nPages) return;
        PageInfo pg = pages[tgid];
        for (uint j = tid; j < pg.nonNullCount; j += PQ_TG) {
            uint v = pg.nonNullOffset + j;
            out[v] = (uchar)(in[v] & 0xFFu);
        }
    }

    // Moves bytes for a flat (page-free) value table: used to build a dictionary's Arrow data buffer.
    kernel void pq_gather_flat(device const uchar* data [[buffer(0)]],
                               device const uint* valOffset [[buffer(1)]],
                               device const uint* valLength [[buffer(2)]],
                               device const uint* offsets [[buffer(3)]],
                               constant uint& n [[buffer(4)]],
                               device uchar* out [[buffer(5)]],
                               uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        uint src = valOffset[i], len = valLength[i], dst = offsets[i];
        for (uint k = 0u; k < len; k++) out[dst + k] = data[src + k];
    }

    // ------------------------------------------------------------------ list assembly
    //
    // Dremel, for one level of repetition. A new row starts wherever the repetition level is 0; an
    // element exists wherever the definition level reaches the repeated node's level. Two prefix sums
    // over those two flags number the rows and the elements, and the row starts then write the offsets.
    kernel void pq_list_flags(device const uchar* defLevels [[buffer(0)]],
                              device const uchar* repLevels [[buffer(1)]],
                              constant uint& n [[buffer(2)]],
                              constant uint& dRep [[buffer(3)]],
                              device uint* elemFlag [[buffer(4)]],
                              device uint* rowFlag [[buffer(5)]],
                              device uchar* elemByte [[buffer(6)]],
                              uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        uint isElem = ((uint)defLevels[i] >= dRep) ? 1u : 0u;
        elemFlag[i] = isElem;
        elemByte[i] = (uchar)isElem;
        rowFlag[i] = (repLevels[i] == 0u) ? 1u : 0u;
    }

    kernel void pq_list_offsets(device const uchar* defLevels [[buffer(0)]],
                                device const uchar* repLevels [[buffer(1)]],
                                device const uint* rowScan [[buffer(2)]],
                                device const uint* elemScan [[buffer(3)]],
                                constant uint& n [[buffer(4)]],
                                constant uint& dRep [[buffer(5)]],
                                device const uint* rowTotal [[buffer(6)]],
                                device uint* offsets [[buffer(7)]],
                                device uchar* validBytes [[buffer(8)]],
                                uint i [[thread_position_in_grid]]) {
        if (i > n) return;
        if (i == n) { offsets[rowTotal[0]] = elemScan[n]; return; }
        if (repLevels[i] != 0u) return;
        uint r = rowScan[i];
        offsets[r] = elemScan[i];
        uint listDef = (dRep >= 1u) ? (dRep - 1u) : 0u;
        validBytes[r] = ((uint)defLevels[i] >= listDef) ? 1u : 0u;
    }
    """
}
