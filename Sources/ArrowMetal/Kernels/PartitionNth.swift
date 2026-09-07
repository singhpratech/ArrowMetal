import Foundation
import Metal

/// Arrow `partition_nth_indices`, as a real selection rather than a sort.
///
/// ## The algorithm
///
/// The contract is weak on purpose: the index at output position `pivot` must be the one a sorted order
/// would put there, everything before it must be no greater and everything after it no smaller. Nothing
/// says the three blocks are themselves ordered. So a full sort is overkill; all that is needed is the
/// value that *would* sit at position `pivot` — one order statistic — and then a three-way split around it.
///
/// The order statistic comes out of an **MSB-first radix select**. Every value is mapped to the same
/// order-preserving unsigned key the radix argsort uses (`SortSource`'s `key_from_*`), so "smaller key"
/// means "smaller value" for signed integers and for floats alike. Then, one byte at a time from the top:
///
/// 1. Histogram the 256 possible values of the current digit, counting **only** the keys whose higher
///    bytes already equal the prefix found so far (`pn_hist`, one threadgroup histogram per group folded
///    into 256 global counters).
/// 2. Read the 256 counts on the host, walk them until the running total passes the rank still being
///    looked for, and append that digit to the prefix; the rank drops by the counts skipped over.
///
/// After 4 rounds (32-bit keys) or 8 (64-bit) the prefix *is* the key of the `pivot`-th smallest value.
/// Each round is one read of the array, so the whole search is O(n) with a small constant — against the
/// 4 or 8 histogram + scan + scatter rounds a full radix sort pays, plus the sort's two extra buffers.
///
/// The split around that key is a **single stable partition on the GPU** (`pn_count`, `pn_scan`,
/// `pn_scatter`) that writes the finished index array straight out. Every row is sorted into one of five
/// categories — null, NaN, below the key, equal to it, above it — the host says in which order the
/// categories occupy the output, and the three kernels are the textbook count / scan / scatter of a
/// stable counting sort over those five buckets. Blocks keep their rows in input order, and the scatter
/// ranks a chunk's rows against the other lanes of their SIMD group with five `simd_ballot`s, the same
/// trick `SortSource`'s radix scatter uses.
///
/// Doing it in one pass matters more than it looks. The obvious version — three `compare` + `filter`
/// stream compactions concatenated afterwards — reads the keys three times, writes three worst-case
/// index arrays, needs an `iota` of the row numbers to compact and a zeroed output to copy the blocks
/// into, and costs seven command buffers; two of those steps (building the row numbers, clearing the
/// output) were *CPU* passes over 40 MB at 10M rows, and together they were most of the operation.
///
/// ## Nulls
///
/// `nullPlacement` decides which end the null rows occupy, exactly as in `argsort`, and the selection
/// runs over the non-null rows only — `pn_hist` skips them, so no filtered copy of the keys is made.
/// When `pivot` falls inside the null block every arrangement of the values satisfies the contract (a
/// null compares past every value), so that case skips the select entirely and the partition puts every
/// value in one bucket.
///
/// A NaN travels with the nulls, not with the values: Arrow's `PartitionNthToIndices` sends both to
/// `null_placement`'s end, and `argsort` does the same. `.atEnd` needs no separate bucket — the
/// ascending keys already leave the NaNs at the tail of the value block, just before the nulls — so only
/// `.atStart` on a float column asks for the NaN category. There a NaN is recognised from its key
/// alone: `key_from_f32`/`key_from_f64` canonicalise every NaN to the single largest key, so `key ==
/// nanKey` is the whole test.
enum PartitionNthSource {
    /// Rows are counted into five categories; the host decides the order they occupy the output in.
    static let slots = 5

    /// Category of a row, before the host's ordering is applied. Keep in step with `slotBits` below.
    enum Category: Int { case null = 0, nan = 1, less = 2, equal = 3, greater = 4 }

    /// The five categories packed three bits each, category-indexed: `(slotBits >> 3 * category) & 7` is
    /// the output bucket that category lands in.
    static func slotBits(nullPlacement: NullPlacement) -> Int {
        // `.atEnd`:   values, then nulls.               `.atStart`: nulls, NaNs, then values.
        let order: [Category: Int] = nullPlacement == .atEnd
            ? [.less: 0, .equal: 1, .greater: 2, .null: 3, .nan: 3]
            : [.null: 0, .nan: 1, .less: 2, .equal: 3, .greater: 4]
        return order.reduce(0) { $0 | ($1.value << (3 * $1.key.rawValue)) }
    }

    static func source(K: String) -> String { KernelSource.prelude + """

    #define PN_SLOTS \(slots)u
    // Apple GPUs are 32 lanes wide, which the scatter's ballots already assume.
    #define SIMDS (TG / 32u)

    // The output bucket of row `i`: null, NaN, below the key, equal to it, above it, mapped through the
    // host's ordering. `useThreshold` is 0 when the pivot fell inside the null block, where every value
    // may stay where it is and so every value is one bucket.
    inline uint pn_slot(\(K) key, uint i, device const uchar* validity, uint hasValidity,
                        \(K) threshold, uint useThreshold, \(K) nanKey, uint hasNaN, uint slotBits) {
        uint cat;
        if (hasValidity && !bit_get(validity, i)) cat = 0u;
        else if (hasNaN && key == nanKey) cat = 1u;
        else if (!useThreshold) cat = 3u;
        else cat = key < threshold ? 2u : (key == threshold ? 3u : 4u);
        return (slotBits >> (3u * cat)) & 7u;
    }

    // `pn_count` and `pn_scatter` take the same eleven bucketing arguments in the same buffer slots, so
    // the Swift side binds them once for both.
    #define PN_SLOT_OF(i) pn_slot(keys[i], i, validity, hasValidity, threshold, useThreshold, nanKey, hasNaN, slotBits)

    // The digit histogram of the select, restricted to keys whose high bytes match `prefix`. Null rows
    // and (when `.atStart` asks for it) NaN rows are not candidates and are skipped, which is why the
    // select needs no filtered copy of the key array.
    kernel void pn_hist(device const \(K)* keys [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                        constant \(K)& prefix [[buffer(2)]], constant \(K)& highMask [[buffer(3)]],
                        constant uint& shift [[buffer(4)]], constant uint& gridSize [[buffer(5)]],
                        device const uchar* validity [[buffer(6)]], constant uint& hasValidity [[buffer(7)]],
                        constant \(K)& nanKey [[buffer(8)]], constant uint& hasNaN [[buffer(9)]],
                        device atomic_uint* counts [[buffer(10)]],
                        uint lid [[thread_index_in_threadgroup]], uint gid [[thread_position_in_grid]]) {
        threadgroup atomic_uint hist[256];
        for (uint d = lid; d < 256u; d += TG) atomic_store_explicit(&hist[d], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint n = *nPtr;
        for (uint i = gid; i < n; i += gridSize) {
            if (hasValidity && !bit_get(validity, i)) continue;
            \(K) key = keys[i];
            if (hasNaN && key == nanKey) continue;
            if ((key & highMask) == prefix) {
                uint d = (uint)((key >> shift) & (\(K))0xFF);
                atomic_fetch_add_explicit(&hist[d], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = lid; d < 256u; d += TG) {
            uint c = atomic_load_explicit(&hist[d], memory_order_relaxed);
            if (c != 0u) atomic_fetch_add_explicit(&counts[d], c, memory_order_relaxed);
        }
    }

    // Bucket histogram of one block, laid out slot-major: counts[s * blocks + b]. Five counters fit in
    // registers, so the per-element work is five compares and no atomic at all; the threadgroup total
    // is one `simd_sum` per slot plus a walk over the SIMD groups.
    kernel void pn_count(device const \(K)* keys [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                         device const uchar* validity [[buffer(2)]], constant uint& hasValidity [[buffer(3)]],
                         constant \(K)& threshold [[buffer(4)]], constant uint& useThreshold [[buffer(5)]],
                         constant \(K)& nanKey [[buffer(6)]], constant uint& hasNaN [[buffer(7)]],
                         constant uint& slotBits [[buffer(8)]], constant uint& elemsPerBlock [[buffer(9)]],
                         constant uint& blocks [[buffer(10)]], device uint* counts [[buffer(11)]],
                         uint lid [[thread_index_in_threadgroup]], uint tgid [[threadgroup_position_in_grid]],
                         uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
        threadgroup uint tot[PN_SLOTS * SIMDS];
        uint n = *nPtr, start = tgid * elemsPerBlock, end = min(n, start + elemsPerBlock);
        uint c0 = 0u, c1 = 0u, c2 = 0u, c3 = 0u, c4 = 0u;
        for (uint i = start + lid; i < end; i += TG) {
            uint s = PN_SLOT_OF(i);
            c0 += s == 0u; c1 += s == 1u; c2 += s == 2u; c3 += s == 3u; c4 += s == 4u;
        }
        uint s0 = simd_sum(c0), s1 = simd_sum(c1), s2 = simd_sum(c2), s3 = simd_sum(c3), s4 = simd_sum(c4);
        if (lane == 0u) {
            tot[0u * SIMDS + sgid] = s0; tot[1u * SIMDS + sgid] = s1; tot[2u * SIMDS + sgid] = s2;
            tot[3u * SIMDS + sgid] = s3; tot[4u * SIMDS + sgid] = s4;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < PN_SLOTS) {
            uint sum = 0u;
            for (uint g = 0u; g < SIMDS; g++) sum += tot[lid * SIMDS + g];
            counts[lid * blocks + tgid] = sum;
        }
    }

    // Exclusive scan over the slot-major counts table, single thread: it is PN_SLOTS * blocks entries,
    // a few hundred, and the scan sits between two dispatches that each move tens of megabytes.
    kernel void pn_scan(device uint* counts [[buffer(0)]], constant uint& total [[buffer(1)]],
                        uint lid [[thread_index_in_threadgroup]]) {
        if (lid != 0u) return;
        uint run = 0u;
        for (uint i = 0u; i < total; i++) { uint c = counts[i]; counts[i] = run; run += c; }
    }

    // Stable scatter of the row numbers into their buckets. A block walks its rows in chunks of TG, in
    // order; within a chunk a row's rank among the earlier rows of the same bucket is a popcount of the
    // SIMD lanes below it holding that bucket, plus the counts the earlier SIMD groups of the chunk
    // published in `simdCount`. Five ballots cover the five buckets, and every lane reaches all of them.
    kernel void pn_scatter(device const \(K)* keys [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                           device const uchar* validity [[buffer(2)]], constant uint& hasValidity [[buffer(3)]],
                           constant \(K)& threshold [[buffer(4)]], constant uint& useThreshold [[buffer(5)]],
                           constant \(K)& nanKey [[buffer(6)]], constant uint& hasNaN [[buffer(7)]],
                           constant uint& slotBits [[buffer(8)]], constant uint& elemsPerBlock [[buffer(9)]],
                           constant uint& blocks [[buffer(10)]], device const uint* offsets [[buffer(11)]],
                           device int* out [[buffer(12)]],
                           uint lid [[thread_index_in_threadgroup]], uint tgid [[threadgroup_position_in_grid]],
                           uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
        threadgroup uint base[PN_SLOTS];
        threadgroup uint simdCount[PN_SLOTS * SIMDS];
        if (lid < PN_SLOTS) base[lid] = offsets[lid * blocks + tgid];
        for (uint j = lid; j < PN_SLOTS * SIMDS; j += TG) simdCount[j] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint n = *nPtr, start = tgid * elemsPerBlock, end = min(n, start + elemsPerBlock);
        for (uint chunk = start; chunk < end; chunk += TG) {
            uint i = chunk + lid;
            bool active = i < end;
            uint s = active ? PN_SLOT_OF(i) : 0u;
            uint b0 = (uint)((simd_vote::vote_t)simd_ballot(active && s == 0u));
            uint b1 = (uint)((simd_vote::vote_t)simd_ballot(active && s == 1u));
            uint b2 = (uint)((simd_vote::vote_t)simd_ballot(active && s == 2u));
            uint b3 = (uint)((simd_vote::vote_t)simd_ballot(active && s == 3u));
            uint b4 = (uint)((simd_vote::vote_t)simd_ballot(active && s == 4u));
            uint peers = s == 0u ? b0 : (s == 1u ? b1 : (s == 2u ? b2 : (s == 3u ? b3 : b4)));
            uint rank = popcount(peers & ((1u << lane) - 1u));
            bool leader = active && rank == 0u;
            if (leader) simdCount[s * SIMDS + sgid] = popcount(peers);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint before = 0u, total = 0u;
            if (active) {
                for (uint g = 0u; g < SIMDS; g++) {
                    uint c = simdCount[s * SIMDS + g];
                    before += (g < sgid) ? c : 0u;
                    total += c;
                }
                out[base[s] + before + rank] = (int)i;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (leader) {
                simdCount[s * SIMDS + sgid] = 0u;       // ready for the next chunk, no bulk clear needed
                if (before == 0u) base[s] += total;     // the bucket's first SIMD group advances the base
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    """ }
}

extension MetalArray {

    /// Arrow `partition_nth_indices`: indices arranged so that the element at position `pivot` is the one
    /// a sorted order would put there, everything before it no greater and everything after it no smaller.
    ///
    /// One GPU radix select for the pivot value plus one stable three-way partition — O(length), not a
    /// sort. `nullPlacement` puts the null rows at the end (Arrow's default) or the start, as in `argsort`.
    ///
    /// The permutation is *not* the sorted one, and Arrow does not promise it is: only the partition
    /// property holds. Within each of the blocks the rows keep their original relative order, because the
    /// partition that produces them is stable.
    public func partitionNthIndices(_ pivot: Int,
                                    nullPlacement: NullPlacement = .atEnd) throws -> MetalArray<Int32> {
        guard pivot >= 0, pivot <= length else {
            throw ArrowMetalError.invalidArrowArray("partition index \(pivot) is outside 0...\(length)")
        }
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        if n == 0 { return try MetalArray<Int32>([Int32](), context: ctx) }

        // The order-preserving key the radix argsort would build, ascending.
        let wide = T.byteWidth == 8
        let keyBuf = try orderKeyBuffer()
        if wide {
            let keys = MetalArray<UInt64>(length: n, nullCount: 0, validity: nil, values: keyBuf, context: ctx)
            return try partition(keys: keys, pivot: pivot, nullPlacement: nullPlacement)
        }
        let keys = MetalArray<UInt32>(length: n, nullCount: 0, validity: nil, values: keyBuf, context: ctx)
        return try partition(keys: keys, pivot: pivot, nullPlacement: nullPlacement)
    }

    /// The ascending order-preserving unsigned key of every row, straight out of `SortSource`. Narrow
    /// integer types widen to int32 first, exactly as `argsort` does.
    private func orderKeyBuffer() throws -> MetalArrowBuffer {
        let ctx = context, n = length
        let wide = T.byteWidth == 8
        let keyType = wide ? "ulong" : "uint"
        let mapFn: String
        let source: MetalArrowBuffer
        var keepAlive: AnyObject? = nil
        switch T.self {
        case is Int32.Type: mapFn = "key_from_i32"; source = values
        case is UInt32.Type: mapFn = "key_from_u32"; source = values
        case is Float.Type: mapFn = "key_from_f32"; source = values
        case is Int64.Type: mapFn = "key_from_i64"; source = values
        case is UInt64.Type: mapFn = "key_from_u64"; source = values
        case is Double.Type: mapFn = "key_from_f64"; source = values
        default:
            let widened = try cast(to: Int32.self)
            keepAlive = widened
            mapFn = "key_from_i32"
            source = widened.values
        }
        let pso = try ctx.pipeline(source: SortSource.source(K: keyType), function: mapFn,
                                   cacheKey: "sort/\(keyType)/\(SortSource.digitBits)/\(mapFn)")
        let out = try MetalArrowBuffer.allocate(byteCount: n * (wide ? 8 : 4), zeroed: false, context: ctx)
        // The map's -0.0 / NaN report is only of use to `sorted()`; the split reads the keys themselves.
        let flags = try MetalArrowBuffer.allocate(byteCount: 4, zeroed: true, context: ctx)
        try withExtendedLifetime(keepAlive) {
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(source.mtl, offset: source.offset, index: 0)
                Dispatch.setLength(enc, n, nil, index: 1)
                enc.setBuffer(out.mtl, offset: out.offset, index: 2)
                Dispatch.setUInt(enc, 0, index: 3)               // ascending; the split handles direction
                enc.setBuffer(flags.mtl, offset: flags.offset, index: 4)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
            ctx.retainUntilFlush(self)
        }
        return out
    }

    /// How many valid rows carry a NaN. Only the `.atStart` float path needs the count, to know where
    /// the value block begins, and only then does it pay for the scan.
    private func nanRows() -> Int {
        guard T.isFloatingPoint else { return 0 }
        return withExtendedLifetime(self) { () -> Int in
            let p = valuePointer
            var count = 0
            if let bm = validity?.typed(UInt8.self) {
                for i in 0..<length where Bitmap.isSet(bm, i) && p[i] != p[i] { count += 1 }
            } else {
                for i in 0..<length where p[i] != p[i] { count += 1 }
            }
            return count
        }
    }

    /// The split itself, once the keys exist.
    private func partition<K>(keys: MetalArray<K>, pivot: Int,
                              nullPlacement: NullPlacement) throws -> MetalArray<Int32>
        where K: ArrowPrimitive & FixedWidthInteger & UnsignedInteger {
        let n = length, nulls = nullCount, m = n - nulls
        let bitmap = nulls > 0 ? validity : nil

        // Only `.atStart` on a float column gives the NaNs a block of their own. `key_from_f32`/`f64`
        // canonicalise every NaN to one bit pattern, one step past +inf, so a single key is the whole
        // test: 0x7F800001 / 0x7FF0000000000001 with the sign bit flipped on by the ascending map.
        // `T.isFloatingPoint` holds for Float and Double only, the two types whose keys those produce.
        let nanKey: K? = (nullPlacement == .atStart && T.isFloatingPoint)
            ? (K.bitWidth == 64 ? K(truncatingIfNeeded: 0xFFF0_0000_0000_0001 as UInt64)
                                : K(truncatingIfNeeded: 0xFF80_0001 as UInt32))
            : nil
        let nans = nanKey != nil ? nanRows() : 0
        let values = m - nans

        // Where does output position `pivot` land? nil means "inside the null (or NaN) block", where
        // any arrangement of the values is already correct.
        let rank: Int?
        switch nullPlacement {
        case .atEnd: rank = pivot < m ? pivot : nil
        case .atStart:
            let front = nulls + nans
            rank = (pivot >= front && pivot - front < values) ? pivot - front : nil
        }

        var threshold = K.zero
        if let rank {
            threshold = try PartitionNth.select(keys, rank: rank, candidates: values,
                                                validity: bitmap, nanKey: nanKey, owner: self)
        }
        return try PartitionNth.partition(keys, validity: bitmap, threshold: rank != nil ? threshold : nil,
                                          nanKey: nanKey, nullPlacement: nullPlacement, owner: self)
    }
}

/// The radix select and the stable partition, shared by every element type.
enum PartitionNth {

    /// Blocks for the count/scatter pair, on the same rule as the radix sort: a few big blocks, because
    /// each one loads and clears a whole bucket table and the scan is a single walk over
    /// `PN_SLOTS * blocks`, but never so few that a small input leaves most of the GPU idle.
    static func layout(_ n: Int) -> (elemsPerBlock: Int, blocks: Int) {
        var elemsPerBlock = Swift.max(4096, ((n + 127) / 128 + 255) / 256 * 256)
        while elemsPerBlock > 256 && (n + elemsPerBlock - 1) / elemsPerBlock < 64 { elemsPerBlock >>= 1 }
        return (elemsPerBlock, (n + elemsPerBlock - 1) / elemsPerBlock)
    }

    /// The `rank`-th smallest key (0-based) among the candidate rows — the valid ones, minus the NaNs
    /// when `nanKey` names them — found MSB-first with `rank + 1` never materialised as a sort.
    static func select<K>(_ keys: MetalArray<K>, rank: Int, candidates: Int,
                          validity: MetalArrowBuffer?, nanKey: K?, owner: AnyObject) throws -> K
        where K: ArrowPrimitive & FixedWidthInteger & UnsignedInteger {
        let ctx = keys.context, m = keys.length
        precondition(rank >= 0 && rank < candidates, "radix select rank out of range")
        let bits = K.bitWidth
        let type = bits == 64 ? "ulong" : "uint"
        let pso = try Dispatch.pipeline(ctx, family: "partition-nth", source: PartitionNthSource.source(K: type),
                                        function: "pn_hist", type: type)
        let counts = try MetalArrowBuffer.allocate(byteCount: 256 * 4, zeroed: true, context: ctx)
        let groups = Swift.max(1, Swift.min(1024, (m + Dispatch.threadgroupSize - 1) / Dispatch.threadgroupSize))
        let gridSize = groups * Dispatch.threadgroupSize

        var prefix = K.zero
        var highMask = K.zero
        var remaining = rank
        for round in 0..<(bits / 8) {
            let shift = bits - 8 * (round + 1)
            memset(counts.mutableContents, 0, 256 * 4)
            var pfx = prefix, mask = highMask
            try withExtendedLifetime(owner) {
                try ctx.run { enc in
                    enc.setComputePipelineState(pso)
                    enc.setBuffer(keys.values.mtl, offset: keys.values.offset, index: 0)
                    Dispatch.setLength(enc, m, nil, index: 1)
                    enc.setBytes(&pfx, length: bits / 8, index: 2)
                    enc.setBytes(&mask, length: bits / 8, index: 3)
                    Dispatch.setUInt(enc, shift, index: 4)
                    Dispatch.setUInt(enc, gridSize, index: 5)
                    enc.setBuffer((validity ?? keys.values).mtl, offset: (validity ?? keys.values).offset, index: 6)
                    Dispatch.setUInt(enc, validity != nil ? 1 : 0, index: 7)
                    Dispatch.setScalar(enc, nanKey ?? K.zero, index: 8)
                    Dispatch.setUInt(enc, nanKey != nil ? 1 : 0, index: 9)
                    enc.setBuffer(counts.mtl, offset: counts.offset, index: 10)
                    enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1))
                }
            }
            try ctx.syncPoint()
            let c = counts.typed(UInt32.self)
            var acc = 0
            var digit = 255
            for d in 0..<256 {
                let cd = Int(c[d])
                if acc + cd > remaining { digit = d; break }
                acc += cd
            }
            remaining -= acc
            prefix |= K(UInt(digit)) << K(shift)
            highMask |= K(255) << K(shift)
        }
        return prefix
    }

    /// The stable partition: count the buckets per block, scan the table, scatter the row numbers.
    /// One command buffer, one pass of writes, and the result is the finished index array.
    ///
    /// `threshold` is nil when the pivot landed inside the null (or NaN) block, where every value may
    /// stay where it is; the values then share one bucket and the partition is just the null move.
    static func partition<K>(_ keys: MetalArray<K>, validity: MetalArrowBuffer?, threshold: K?,
                             nanKey: K?, nullPlacement: NullPlacement,
                             owner: AnyObject) throws -> MetalArray<Int32>
        where K: ArrowPrimitive & FixedWidthInteger & UnsignedInteger {
        let ctx = keys.context, n = keys.length
        let type = K.bitWidth == 64 ? "ulong" : "uint"
        func pso(_ f: String) throws -> MTLComputePipelineState {
            try Dispatch.pipeline(ctx, family: "partition-nth", source: PartitionNthSource.source(K: type),
                                  function: f, type: type)
        }
        let countPSO = try pso("pn_count"), scanPSO = try pso("pn_scan"), scatterPSO = try pso("pn_scatter")
        let (elemsPerBlock, blocks) = layout(n)
        let slots = PartitionNthSource.slots
        let counts = try MetalArrowBuffer.allocate(byteCount: slots * blocks * 4, zeroed: false, context: ctx)
        // Every output position is written exactly once: the bucket counts add up to `n`.
        let out = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: false, context: ctx)
        let slotBits = PartitionNthSource.slotBits(nullPlacement: nullPlacement)
        let tg = MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1)
        let grid = MTLSize(width: blocks, height: 1, depth: 1)

        func bindSlotArgs(_ enc: MTLComputeCommandEncoder) {
            enc.setBuffer(keys.values.mtl, offset: keys.values.offset, index: 0)
            Dispatch.setLength(enc, n, nil, index: 1)
            enc.setBuffer((validity ?? keys.values).mtl, offset: (validity ?? keys.values).offset, index: 2)
            Dispatch.setUInt(enc, validity != nil ? 1 : 0, index: 3)
            Dispatch.setScalar(enc, threshold ?? K.zero, index: 4)
            Dispatch.setUInt(enc, threshold != nil ? 1 : 0, index: 5)
            Dispatch.setScalar(enc, nanKey ?? K.zero, index: 6)
            Dispatch.setUInt(enc, nanKey != nil ? 1 : 0, index: 7)
            Dispatch.setUInt(enc, slotBits, index: 8)
            Dispatch.setUInt(enc, elemsPerBlock, index: 9)
            Dispatch.setUInt(enc, blocks, index: 10)
        }

        try withExtendedLifetime(owner) {
            try ctx.run { enc in
                enc.setComputePipelineState(countPSO)
                bindSlotArgs(enc)
                enc.setBuffer(counts.mtl, offset: 0, index: 11)
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(scanPSO)
                enc.setBuffer(counts.mtl, offset: 0, index: 0)
                Dispatch.setUInt(enc, slots * blocks, index: 1)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(scatterPSO)
                bindSlotArgs(enc)
                enc.setBuffer(counts.mtl, offset: 0, index: 11)
                enc.setBuffer(out.mtl, offset: 0, index: 12)
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
            }
        }
        ctx.retainUntilFlush(keys)
        ctx.retainUntilFlush(counts)
        ctx.retainUntilFlush(owner)
        return MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: out, context: ctx)
    }
}
