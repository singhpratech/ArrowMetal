import Foundation
import Metal

/// MSL for GPU dictionary encoding of `utf8` columns.
///
/// Sorting the rows by a hash puts equal strings next to each other, and `sd_mark` then decides run
/// boundaries by comparing the **bytes** of adjacent sorted strings — never the hashes. Two different
/// strings that hash alike therefore never share a boundary decision.
///
/// What byte comparison alone cannot fix is *placement*: a stable sort by hash leaves two colliding
/// strings interleaved (`A B A B`), and adjacent comparison would then cut that bucket into four runs
/// and give the two `A`s different codes. So `sd_mark` also counts key runs beside content runs. A
/// content run is always a subdivision of a key run, so the two totals are equal exactly when every
/// bucket holds a single distinct string — precisely the condition under which the marks are the true
/// distinct-value boundaries. When they differ, the caller re-hashes with different seeds; if that keeps
/// failing, the host implementation finishes the job.
enum StringDictionarySource {
    static let source: String = KernelSource.prelude + """

    inline uint sd_rotl32(uint x, uint r) { return (x << r) | (x >> (32u - r)); }
    inline uint sd_fmix32(uint h) {
        h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16; return h;
    }
    // MurmurHash3 x86_32 with an explicit seed. Seed 0 is exactly what `hash32()` computes.
    kernel void sd_hash32_seed(device const int* offsets [[buffer(0)]], device const uchar* data [[buffer(1)]],
                               device const uint* nPtr [[buffer(2)]], constant uint& seed [[buffer(3)]],
                               device uint* out [[buffer(4)]], uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        int start = offsets[i], len = offsets[i + 1] - offsets[i];
        uint h = seed; const uint c1 = 0xcc9e2d51u, c2 = 0x1b873593u;
        int nblocks = len / 4;
        for (int b = 0; b < nblocks; b++) {
            int p = start + b * 4;
            uint k = (uint)data[p] | ((uint)data[p + 1] << 8) | ((uint)data[p + 2] << 16) | ((uint)data[p + 3] << 24);
            k *= c1; k = sd_rotl32(k, 15); k *= c2;
            h ^= k; h = sd_rotl32(h, 13); h = h * 5u + 0xe6546b64u;
        }
        uint k1 = 0u; int tail = start + nblocks * 4;
        switch (len & 3) {
            case 3: k1 ^= (uint)data[tail + 2] << 16;
            case 2: k1 ^= (uint)data[tail + 1] << 8;
            case 1: k1 ^= (uint)data[tail]; k1 *= c1; k1 = sd_rotl32(k1, 15); k1 *= c2; h ^= k1;
        }
        h ^= (uint)len;
        out[i] = sd_fmix32(h);
    }

    // Packs two independent 32-bit hashes into the 64-bit key the rows are grouped by.
    kernel void sd_compose(device const uint* a [[buffer(0)]], device const uint* b [[buffer(1)]],
                           device const uint* nPtr [[buffer(2)]], device ulong* out [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
        if (i < *nPtr) out[i] = ((ulong)a[i] << 32) | (ulong)b[i];
    }

    // Run boundaries over the key-sorted order, decided by comparing the full bytes of adjacent
    // strings. `markBytes` feeds the bitmap packer behind `filter`; `markInts` feeds the rank scan and
    // drops the mark at position 0 so the first run gets code 0. `counters[0]` totals content runs and
    // `counters[1]` key runs; see the type comment for what their equality proves.
    kernel void sd_mark(device const int* offsets [[buffer(0)]], device const uchar* data [[buffer(1)]],
                        device const ulong* keys [[buffer(2)]], device const int* ord [[buffer(3)]],
                        device const uint* nPtr [[buffer(4)]], device uchar* markBytes [[buffer(5)]],
                        device int* markInts [[buffer(6)]], device atomic_uint* counters [[buffer(7)]],
                        uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
        uint n = *nPtr;
        uint contentRun = 0u, keyRun = 0u;
        if (i < n) {
            if (i == 0u) { contentRun = 1u; keyRun = 1u; }
            else {
                int a = ord[i], b = ord[i - 1u];
                if (keys[a] != keys[b]) { contentRun = 1u; keyRun = 1u; }
                else {
                    int a0 = offsets[a], len = offsets[a + 1] - a0;
                    int b0 = offsets[b], blen = offsets[b + 1] - b0;
                    bool eq = (len == blen);
                    for (int t = 0; eq && t < len; t++) if (data[a0 + t] != data[b0 + t]) eq = false;
                    contentRun = eq ? 0u : 1u;
                }
            }
            markBytes[i] = (uchar)contentRun;
            markInts[i] = (i == 0u) ? 0 : (int)contentRun;
        }
        // Every thread of the simdgroup joins the reduction, including those past the end (adding zero).
        uint cs = simd_sum(contentRun), ks = simd_sum(keyRun);
        if (lane == 0u) {
            atomic_fetch_add_explicit(&counters[0], cs, memory_order_relaxed);
            atomic_fetch_add_explicit(&counters[1], ks, memory_order_relaxed);
        }
    }
    """
}

extension MetalStringArray {
    /// Arrow `dictionary_encode` on the GPU: dense Int32 codes in [0, unique.count) plus the unique
    /// strings in first-seen order — the shape `GroupBy` wants. A null string gets a null code.
    ///
    /// Hash every string, argsort the hashes, mark run boundaries by comparing bytes, turn the marks
    /// into ranks with the same GPU scan `unique()` uses, and gather one representative per run with the
    /// existing string gather. The codes are then relabelled into first-seen order — two argsorts over
    /// the dictionary, which is small — so the result is identical to the host implementation rather
    /// than a relabelling of it.
    public func dictionaryEncodeGPU() throws -> (codes: MetalArray<Int32>, unique: MetalStringArray) {
        let n = length
        guard n - nullCount > 0 else {
            // Empty, or every row null: every code is null and there are no uniques.
            let buf = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * 4, context: context)
            let codes = MetalArray<Int32>(length: n, nullCount: n, validity: validity, values: buf, context: context)
            return (codes, try MetalStringArray([], context: context))
        }
        try Dispatch.checkLength(n)
        for round in UInt32(0)..<3 {
            let high = round == 0 ? try hash32() : try seededHash32(0x51ED_270B &+ round)
            let low = try seededHash32(0x9E37_79B9 &+ round)
            if let r = try encode(keys: try Self.compose(high, low)) { return r }
        }
        return try dictionaryEncodeCPU()
    }

    /// The 64-bit grouping key: two independent 32-bit hashes side by side.
    ///
    /// One 32-bit hash is not wide enough at this scale. 200k distinct strings already collide about
    /// five times by the birthday bound, and every collision costs a whole retry — so a single hash
    /// would send realistic inputs down the retry path every time. 64 bits makes a collision rare
    /// enough that the retry is what it should be: an unreached safety net.
    static func compose(_ high: MetalArray<UInt32>, _ low: MetalArray<UInt32>) throws -> MetalArray<UInt64> {
        let ctx = high.context
        let n = high.length
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * 8, zeroed: false, context: ctx)
        let p = try ctx.pipeline(source: StringDictionarySource.source, function: "sd_compose", cacheKey: "strdict/sd_compose")
        if n > 0 {
            try ctx.run { enc in
                enc.setComputePipelineState(p)
                enc.setBuffer(high.values.mtl, offset: high.values.offset, index: 0)
                enc.setBuffer(low.values.mtl, offset: low.values.offset, index: 1)
                Dispatch.setLength(enc, n, nil, index: 2)
                enc.setBuffer(out.mtl, offset: out.offset, index: 3)
                Dispatch.dispatch1D(enc, p, count: n)
            }
            ctx.retainUntilFlush(high); ctx.retainUntilFlush(low)
        }
        return MetalArray<UInt64>(length: n, nullCount: high.nullCount, validity: high.validity, values: out, context: ctx)
    }

    /// MurmurHash3 x86_32 with a seed other than 0. `hash32()` is the seed-0 case and stays the entry
    /// point; this covers the second half of the key and the retry rounds.
    func seededHash32(_ seed: UInt32) throws -> MetalArray<UInt32> {
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(length, 1) * 4, zeroed: false, context: context)
        let p = try context.pipeline(source: StringDictionarySource.source, function: "sd_hash32_seed",
                                     cacheKey: "strdict/sd_hash32_seed")
        if length > 0 {
            try context.run { enc in
                enc.setComputePipelineState(p)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                Dispatch.setLength(enc, length, nil, index: 2)
                Dispatch.setUInt(enc, Int(seed), index: 3)
                enc.setBuffer(out.mtl, offset: out.offset, index: 4)
                Dispatch.dispatch1D(enc, p, count: length)
            }
        }
        return MetalArray<UInt32>(length: length, nullCount: nullCount, validity: validity, values: out, context: context)
    }

    /// One attempt at a given grouping key. Returns nil when that key put two distinct strings in one
    /// bucket, which is the one thing this arrangement cannot resolve without re-hashing.
    func encode(keys: MetalArray<UInt64>) throws -> (codes: MetalArray<Int32>, unique: MetalStringArray)? {
        let ctx = context
        let nonNull = length - nullCount
        // Key order, nulls last, so the first `nonNull` positions are the rows that carry a string.
        let ord = try keys.argsort().slice(offset: 0, length: nonNull)
        let markBytes = try MetalArrowBuffer.allocate(byteCount: nonNull, zeroed: false, context: ctx)
        let markInts = try MetalArrowBuffer.allocate(byteCount: nonNull * 4, zeroed: false, context: ctx)
        let counters = try MetalArrowBuffer.allocate(byteCount: 8, context: ctx)
        let p = try ctx.pipeline(source: StringDictionarySource.source, function: "sd_mark", cacheKey: "strdict/sd_mark")
        try ctx.run { enc in
            enc.setComputePipelineState(p)
            enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
            enc.setBuffer(data.mtl, offset: data.offset, index: 1)
            enc.setBuffer(keys.values.mtl, offset: keys.values.offset, index: 2)
            enc.setBuffer(ord.values.mtl, offset: ord.values.offset, index: 3)
            Dispatch.setLength(enc, nonNull, nil, index: 4)
            enc.setBuffer(markBytes.mtl, offset: markBytes.offset, index: 5)
            enc.setBuffer(markInts.mtl, offset: markInts.offset, index: 6)
            enc.setBuffer(counters.mtl, offset: counters.offset, index: 7)
            Dispatch.dispatch1D(enc, p, count: nonNull)
        }
        try ctx.syncPoint()
        let (contentRuns, keyRuns) = withExtendedLifetime((counters, ord, keys)) {
            (counters.typed(UInt32.self)[0], counters.typed(UInt32.self)[1])
        }
        guard contentRuns == keyRuns else { return nil }

        let runs = SortedRuns(ord: ord, markBytes: markBytes, markInts: markInts, count: nonNull,
                              source: UniqueSource.source(U: "ulong"), unsignedType: "ulong")
        // `firstIdx[g]` is the row that opens run g in the sorted order. The sort is stable and a run
        // shares one key, so that row is the lowest row index in the run: its first-seen occurrence.
        let (firstIdx, _) = try keys.runStarts(runs)
        let keyOrderCodes = try keys.scatterRanks(runs, rows: length)
        let reps = try gather(firstIdx)
        // Sorting the first-seen rows gives the dictionary order; argsorting that permutation inverts
        // it, which is exactly the old-code -> new-code relabelling.
        let perm = try firstIdx.argsort()
        let relabel = try perm.argsort()
        let codes = try relabel.take(keyOrderCodes)
        let ordered = try reps.take(perm)
        // The dictionary has no nulls, so it carries no bitmap (the gather leaves an all-ones one).
        let unique = ordered.nullCount == 0 && ordered.validity != nil
            ? MetalStringArray(length: ordered.length, nullCount: 0, validity: nil,
                               offsets: ordered.offsets, data: ordered.data, context: ctx)
            : ordered
        return (codes, unique)
    }

    /// The host implementation: a hash map over the string bytes. The oracle the tests compare against,
    /// and the last resort when several hash seeds all collide.
    func dictionaryEncodeCPU() throws -> (codes: MetalArray<Int32>, unique: MetalStringArray) {
        var map: [ArraySlice<UInt8>: Int32] = [:]
        var uniques: [String?] = []
        var codes: [Int32?] = []
        codes.reserveCapacity(length)
        let o = offsets.typed(Int32.self), d = data.typed(UInt8.self)
        let bytes = UnsafeBufferPointer(start: d, count: totalBytes)
        for i in 0..<length {
            guard isValid(i) else { codes.append(nil); continue }
            let slice = Array(bytes[Int(o[i])..<Int(o[i + 1])])[...]
            if let c = map[slice] { codes.append(c) } else {
                let c = Int32(uniques.count); map[slice] = c; uniques.append(String(decoding: slice, as: UTF8.self)); codes.append(c)
            }
        }
        return (try MetalArray<Int32>(codes, context: context), try MetalStringArray(uniques, context: context))
    }
}
