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
/// The split is then three GPU stream compactions of the index array (`< key`, `== key`, `> key`) with
/// the existing filter kernel, concatenated by a copy kernel. The `< key` block is by construction no
/// longer than `pivot`, so position `pivot` lands inside the `== key` block: the contract holds.
///
/// ## Nulls
///
/// `nullPlacement` decides which end the null rows occupy, exactly as in `argsort`, and the selection
/// runs over the non-null rows only. When `pivot` falls inside the null block every arrangement of the
/// values satisfies the contract (a null compares past every value), so that case skips the select
/// entirely and just concatenates the two blocks.
enum PartitionNthSource {
    /// The digit histogram, restricted to keys whose high bytes match `prefix`.
    static func source(K: String) -> String { KernelSource.prelude + """

    kernel void pn_hist(device const \(K)* keys [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                        constant \(K)& prefix [[buffer(2)]], constant \(K)& highMask [[buffer(3)]],
                        constant uint& shift [[buffer(4)]], constant uint& gridSize [[buffer(5)]],
                        device atomic_uint* counts [[buffer(6)]],
                        uint lid [[thread_index_in_threadgroup]], uint gid [[thread_position_in_grid]]) {
        threadgroup atomic_uint hist[256];
        for (uint d = lid; d < 256u; d += TG) atomic_store_explicit(&hist[d], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint n = *nPtr;
        for (uint i = gid; i < n; i += gridSize) {
            \(K) key = keys[i];
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
    // Copies one int32 block into `dst` at `dstOffset`; called once per block of the concatenation.
    kernel void pn_copy(device const int* src [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                        constant uint& dstOffset [[buffer(2)]], device int* dst [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
        if (i < *nPtr) dst[dstOffset + i] = src[i];
    }
    """ }
}

extension MetalArray {

    /// Arrow `partition_nth_indices`: indices arranged so that the element at position `pivot` is the one
    /// a sorted order would put there, everything before it no greater and everything after it no smaller.
    ///
    /// One GPU radix select for the pivot value plus three stream compactions — O(length), not a sort.
    /// `nullPlacement` puts the null rows at the end (Arrow's default) or the start, as in `argsort`.
    ///
    /// The permutation is *not* the sorted one, and Arrow does not promise it is: only the partition
    /// property holds. Within each of the three blocks the rows keep their original relative order,
    /// because the compaction that produces them is stable.
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
        let src = SortSource.source(K: keyType)
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
        let pso = try ctx.pipeline(source: src, function: mapFn, cacheKey: "sort/\(keyType)/\(mapFn)")
        let out = try MetalArrowBuffer.allocate(byteCount: n * (wide ? 8 : 4), zeroed: false, context: ctx)
        try withExtendedLifetime(keepAlive) {
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(source.mtl, offset: source.offset, index: 0)
                Dispatch.setLength(enc, n, nil, index: 1)
                enc.setBuffer(out.mtl, offset: out.offset, index: 2)
                Dispatch.setUInt(enc, 0, index: 3)               // ascending; the split handles direction
                Dispatch.dispatch1D(enc, pso, count: n)
            }
            ctx.retainUntilFlush(self)
        }
        return out
    }

    /// The split itself, once the keys exist.
    private func partition<K>(keys: MetalArray<K>, pivot: Int,
                              nullPlacement: NullPlacement) throws -> MetalArray<Int32>
        where K: ArrowPrimitive & FixedWidthInteger & UnsignedInteger {
        let ctx = context, n = length
        let nulls = nullCount, m = n - nulls
        let rows = try MetalArray<Int32>.iota(n, context: ctx)

        var validIdx = rows
        var nullIdx: MetalArray<Int32>? = nil
        if nulls > 0 {
            validIdx = try rows.filter(try isValid())
            nullIdx = try rows.filter(try isNull())
        }

        // A NaN travels with the nulls, not with the values: Arrow's `PartitionNthToIndices` sends
        // both to `null_placement`'s end, and `argsort` above does the same. `.atEnd` needs no move —
        // the ascending keys already leave the NaNs at the tail of the value block, just before the
        // nulls — so only `.atStart` pays for the scan, and only for a float column.
        var nanIdx: MetalArray<Int32>? = nil
        var nans = 0
        if nullPlacement == .atStart, T.isFloatingPoint {
            var nanRows: [Int32] = [], valueRows: [Int32] = []
            withExtendedLifetime(self) {
                let p = valuePointer, bm = validity?.typed(UInt8.self)
                for i in 0..<n {
                    if let bm, !Bitmap.isSet(bm, i) { continue }
                    if p[i] != p[i] { nanRows.append(Int32(i)) } else { valueRows.append(Int32(i)) }
                }
            }
            if !nanRows.isEmpty {
                nans = nanRows.count
                nanIdx = try MetalArray<Int32>(nanRows, context: ctx)
                validIdx = try MetalArray<Int32>(valueRows, context: ctx)
            }
        }
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

        var blocks: [MetalArray<Int32>]
        if let rank {
            let selKeys = (nulls > 0 || nans > 0) ? try keys.take(validIdx) : keys
            let threshold = try PartitionNth.select(selKeys, rank: rank)
            let lower = try validIdx.filter(try selKeys.compare(.lt, threshold))
            let equal = try validIdx.filter(try selKeys.compare(.eq, threshold))
            let upper = try validIdx.filter(try selKeys.compare(.gt, threshold))
            blocks = [lower, equal, upper]
        } else {
            blocks = [validIdx]
        }
        // Front to back for `.atStart`: nulls, then the NaNs, then the values — `argsort`'s own order.
        if let nanIdx { blocks.insert(nanIdx, at: 0) }
        if let nullIdx {
            if nullPlacement == .atStart { blocks.insert(nullIdx, at: 0) } else { blocks.append(nullIdx) }
        }
        return try PartitionNth.concatenate(blocks, total: n, context: ctx)
    }
}

/// The radix select and the block concatenation, shared by every element type.
enum PartitionNth {

    /// The `rank`-th smallest key (0-based), found MSB-first with `rank + 1` never materialised as a sort.
    static func select<K>(_ keys: MetalArray<K>, rank: Int) throws -> K
        where K: ArrowPrimitive & FixedWidthInteger & UnsignedInteger {
        let ctx = keys.context, m = keys.length
        precondition(rank >= 0 && rank < m, "radix select rank out of range")
        let bits = K.bitWidth
        let src = PartitionNthSource.source(K: bits == 64 ? "ulong" : "uint")
        let pso = try Dispatch.pipeline(ctx, family: "partition-nth", source: src, function: "pn_hist",
                                        type: bits == 64 ? "ulong" : "uint")
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
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(keys.values.mtl, offset: keys.values.offset, index: 0)
                Dispatch.setLength(enc, m, nil, index: 1)
                enc.setBytes(&pfx, length: bits / 8, index: 2)
                enc.setBytes(&mask, length: bits / 8, index: 3)
                Dispatch.setUInt(enc, shift, index: 4)
                Dispatch.setUInt(enc, gridSize, index: 5)
                enc.setBuffer(counts.mtl, offset: counts.offset, index: 6)
                enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1))
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

    /// Writes the blocks end to end into one int32 index array, on the GPU, one dispatch per block.
    static func concatenate(_ blocks: [MetalArray<Int32>], total: Int,
                            context ctx: MetalContext) throws -> MetalArray<Int32> {
        if blocks.count == 1, blocks[0].length == total { return blocks[0] }
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(total, 1) * 4, zeroed: true, context: ctx)
        let src = PartitionNthSource.source(K: "uint")
        let pso = try Dispatch.pipeline(ctx, family: "partition-nth", source: src, function: "pn_copy", type: "uint")
        var offset = 0
        try ctx.run { enc in
            for b in blocks where b.length > 0 {
                enc.setComputePipelineState(pso)
                enc.setBuffer(b.values.mtl, offset: b.values.offset, index: 0)
                Dispatch.setLength(enc, b.length, nil, index: 1)
                Dispatch.setUInt(enc, offset, index: 2)
                enc.setBuffer(out.mtl, offset: out.offset, index: 3)
                Dispatch.dispatch1D(enc, pso, count: b.length)
                offset += b.length
            }
        }
        for b in blocks { ctx.retainUntilFlush(b) }
        return MetalArray<Int32>(length: total, nullCount: 0, validity: nil, values: out, context: ctx)
    }
}
