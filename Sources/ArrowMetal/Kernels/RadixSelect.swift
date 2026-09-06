import Foundation
import Metal

/// GPU radix select: the k best rows of a column without sorting, for any k.
///
/// The per-threadgroup selection in `TopK.swift` reads the column once but spends most of its time in the
/// threshold check and the bitonic compaction, so at 50M rows it runs at a fraction of memory bandwidth, and
/// it only works for k <= 1024. Radix select reasons about the *key* instead:
///
///  1. **Histogram.** One pass over the column counts every value of the top 8 bits of the order-preserving
///     key, per simdgroup-sized sub-block (`rs_histogram`). Summed over sub-blocks that is the global
///     histogram; kept per sub-block it is exactly the offset table the compaction below needs, which is why
///     the fast path reads the column only twice.
///  2. **Pick the bin.** On the CPU, walk the 256 counts to the first digit whose running total reaches k.
///     Rows with a smaller digit are guaranteed winners (there are fewer than k of them); the rest of the
///     answer lives in that one bin.
///  3. **Compact.** A second pass writes the winners and the bin's rows, in row order, into one array
///     (`rs_scatter`). Every winner's key is smaller than every bin key and both halves are in row order, so
///     one *stable* sort by key puts the array into (key, row) order — the total order `argsort` uses.
///  4. **Refine.** The bin still holds about n/256 rows, which is more than the answer needs and enough to
///     make the final sort the dominant cost. So the same two steps run again, now over the compacted array
///     and on the next digit: a couple of microseconds of work that shrinks the bin by another factor of 256.
///  5. **Order.** A bitonic sort in a single threadgroup when the survivors fit in one (k up to ~1024), and
///     the stable radix sort otherwise. The answer is the first k rows.
///
/// Heavily tied data can put nearly every row in the chosen bin, which would make step 3 copy the whole
/// column. Step 2 therefore repeats on the next digit *without* compacting while the bin is over budget, and
/// once the key has no digits left every remaining row is an exact tie: ties are settled by row order, so
/// only the first `k - winners` of them are written and the answer is complete.
extension TopK {
    /// Value types the radix-select kernels map to a sort key. Narrow integers get a key of their own width
    /// so that the top digit still discriminates.
    static func radixKind<T: ArrowPrimitive>(_: T.Type) -> (kind: String, valueType: String, keyType: String, keyBits: Int)? {
        switch T.self {
        case is Int8.Type: return ("i8", "char", "uint", 8)
        case is UInt8.Type: return ("u8", "uchar", "uint", 8)
        case is Int16.Type: return ("i16", "short", "uint", 16)
        case is UInt16.Type: return ("u16", "ushort", "uint", 16)
        case is Int32.Type: return ("i32", "int", "uint", 32)
        case is UInt32.Type: return ("u32", "uint", "uint", 32)
        case is Float.Type: return ("f32", "float", "uint", 32)
        case is Int64.Type: return ("i64", "long", "ulong", 64)
        case is UInt64.Type: return ("u64", "ulong", "ulong", 64)
        case is Double.Type: return ("f64", "ulong", "ulong", 64)
        default: return nil
        }
    }

    /// How an input is cut into sub-blocks: one sub-block per simdgroup, `subsPerGroup` per threadgroup,
    /// enough of them to fill the GPU and few enough that the count table and its scan stay small.
    static func radixPlan(n: Int) -> (eps: Int, subBlocks: Int, groups: Int) {
        let wanted = 8192
        let eps = Swift.max(32, ((n + wanted - 1) / wanted + 31) / 32 * 32)
        let groups = Swift.max(1, ((n + eps - 1) / eps + RadixSelectSource.subsPerGroup - 1) / RadixSelectSource.subsPerGroup)
        return (eps, groups * RadixSelectSource.subsPerGroup, groups)
    }

    /// How many rows a bin may hold before narrowing another digit beats compacting it. Only applies to the
    /// first source, the column itself, where compacting means writing that many `(key, row)` pairs.
    static func candidateCap(k: Int) -> Int { Swift.max(1 << 20, k) }

    /// Bin size at which refining stops: below this the survivors add little to the final sort, and one more
    /// round would cost a command buffer round trip to save less than it spends.
    static let refineCap = 4096

    /// Largest selection the single-threadgroup bitonic finish can order (32 KB of threadgroup memory holds
    /// 2048 pairs of a 64-bit key and a 32-bit slot).
    static let bitonicCap = 2048

    /// Which selection path `topK` tries first. The per-threadgroup selection wins on small inputs, where one
    /// dispatch beats radix select's two passes plus its mid-flight readback; radix select wins from a few
    /// hundred thousand rows up, and is the only path at all for k above 1024.
    static func preferRadixSelect(n: Int, k: Int) -> Bool { k > 1024 || n >= 1 << 19 }
}

extension MetalArray {
    /// Row indices of the k best rows via radix select, or nil when another path should handle it.
    ///
    /// Returns nil when the type has no key mapping, when fewer than k rows are non-null (the answer then has
    /// to reach into the null rows, which only the sort places), and when k is more than half the selectable
    /// rows — at that point the compaction copies most of the column and the full argsort does the same work
    /// in one go.
    func topKRadixSelect(_ k: Int, largest: Bool) throws -> MetalArray<Int32>? {
        guard k > 0, let kd = TopK.radixKind(T.self) else { return nil }
        let n = length
        let valid = n - nullCount
        guard valid >= k, k <= valid / 2 else { return nil }
        try Dispatch.checkLength(n)

        let ctx = context
        let wide = kd.keyType == "ulong"
        let keyBytes = wide ? 8 : 4
        let radix = RadixSelectSource.radix
        let tg = MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1)
        func setKey(_ enc: MTLComputeCommandEncoder, _ v: UInt64, index: Int) {
            if wide { var u = v; enc.setBytes(&u, length: 8, index: index) }
            else { var u = UInt32(truncatingIfNeeded: v); enc.setBytes(&u, length: 4, index: index) }
        }
        // Pipelines are specialised per key mapping. A refinement round reads keys directly, which is the
        // identity mapping — the same source the unsigned-integer column of that width already generates.
        func pipelines(_ kind: String, _ V: String) throws -> (String) throws -> MTLComputePipelineState {
            let src = RadixSelectSource.source(kind: kind, V: V, K: kd.keyType)
            return { f in try Dispatch.pipeline(ctx, family: "radixselect", source: src, function: f, type: kind) }
        }

        // The input of the current round: the column to begin with, then the compacted candidates.
        var srcKind = kd.kind, srcType = kd.valueType
        var srcValues = values
        var srcRows: MetalArrowBuffer? = nil        // nil: a slot's index is the row number
        var srcValidity = validity
        var srcN = n
        var srcInv = largest ? 1 : 0

        var prefix: UInt64 = 0
        var shift = kd.keyBits - 8
        var winners = 0                             // rows already known to beat the whole candidate window
        var outKeys = srcValues, outRows = srcValues, outLength = 0

        while true {
            let pso = try pipelines(srcKind, srcType)
            let (eps, subBlocks, groups) = TopK.radixPlan(n: srcN)
            let hasPrefix = shift + 8 < kd.keyBits
            let vv = srcValidity ?? srcValues
            let hasV = srcValidity != nil
            let grid = MTLSize(width: groups, height: 1, depth: 1)

            // --- 1: digit histogram of every sub-block, and the global count per digit ---
            let counts = try MetalArrowBuffer.allocate(byteCount: radix * subBlocks * 4, zeroed: false, context: ctx)
            let totals = try MetalArrowBuffer.allocate(byteCount: radix * 4, zeroed: false, context: ctx)
            let histPSO = try pso("rs_histogram"), totalsPSO = try pso("rs_totals")
            try ctx.run { enc in
                enc.setComputePipelineState(histPSO)
                enc.setBuffer(srcValues.mtl, offset: srcValues.offset, index: 0)
                enc.setBuffer(vv.mtl, offset: vv.offset, index: 1)
                Dispatch.setLength(enc, srcN, nil, index: 2)
                Dispatch.setUInt(enc, hasV ? 1 : 0, index: 3)
                Dispatch.setUInt(enc, srcInv, index: 4)
                Dispatch.setUInt(enc, shift, index: 5)
                setKey(enc, prefix, index: 6)
                Dispatch.setUInt(enc, hasPrefix ? shift + 8 : 0, index: 7)
                Dispatch.setUInt(enc, hasPrefix ? 1 : 0, index: 8)
                Dispatch.setUInt(enc, eps, index: 9)
                Dispatch.setUInt(enc, subBlocks, index: 10)
                enc.setBuffer(counts.mtl, offset: counts.offset, index: 11)
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(totalsPSO)
                enc.setBuffer(counts.mtl, offset: counts.offset, index: 0)
                Dispatch.setUInt(enc, subBlocks, index: 1)
                enc.setBuffer(totals.mtl, offset: totals.offset, index: 2)
                enc.dispatchThreadgroups(MTLSize(width: radix, height: 1, depth: 1), threadsPerThreadgroup: tg)
            }
            ctx.retainUntilFlush(counts); ctx.retainUntilFlush(totals); ctx.retainUntilFlush(self)
            try ctx.syncPoint()      // the 256 counts are read here; everything after depends on them

            // --- 2: the first digit whose running total reaches k ---
            var below = 0, target = 0, binCount = 0
            var found = false
            withExtendedLifetime(totals) {
                let t = totals.typed(UInt32.self)
                for d in 0..<radix {
                    let c = Int(t[d])
                    if winners + below + c >= k { target = d; binCount = c; found = true; break }
                    below += c
                }
            }
            // The counted rows always include at least k of them, so a bin is always found. If that invariant
            // ever broke, hand the query to the sort rather than answer it wrongly.
            guard found else { return nil }
            winners += below

            let collapsed = shift == 0                  // the key is fully determined: the bin is all ties
            if !collapsed && srcRows == nil && binCount > TopK.candidateCap(k: k) {
                // Compacting this bin would copy most of the column. Narrow another digit in place instead.
                prefix = (prefix << 8) | UInt64(target)
                shift -= 8
                continue
            }

            // --- 3: compact the winners and the bin into one array, in row order ---
            let full = (prefix << 8) | UInt64(target)
            let loKey = collapsed ? full : full << UInt64(shift)
            let hiKey = collapsed ? loKey : loKey | ((UInt64(1) << UInt64(shift)) - 1)
            let needed = k - winners
            let tail = collapsed ? needed : binCount    // ties beyond the k-th can never be in the answer
            let m = winners + tail

            let blockLt = try MetalArrowBuffer.allocate(byteCount: subBlocks * 4, zeroed: false, context: ctx)
            let blockEq = try MetalArrowBuffer.allocate(byteCount: subBlocks * 4, zeroed: false, context: ctx)
            let keysOut = try MetalArrowBuffer.allocate(byteCount: m * keyBytes, zeroed: false, context: ctx)
            let rowsOut = try MetalArrowBuffer.allocate(byteCount: m * 4, zeroed: false, context: ctx)
            let scanPSO = try pso("rs_scan"), scatterPSO = try pso("rs_scatter")
            // The histogram table already holds the per-sub-block counts of "digit < target" and
            // "digit == target" — but only when it counted every row, so only on an unprefixed first round.
            let fromTable = !hasPrefix
            let countPSO = try pso(fromTable ? "rs_block_counts" : "rs_count")
            try ctx.run { enc in
                enc.setComputePipelineState(countPSO)
                if fromTable {
                    enc.setBuffer(counts.mtl, offset: counts.offset, index: 0)
                    Dispatch.setUInt(enc, subBlocks, index: 1)
                    Dispatch.setUInt(enc, target, index: 2)
                    enc.setBuffer(blockLt.mtl, offset: blockLt.offset, index: 3)
                    enc.setBuffer(blockEq.mtl, offset: blockEq.offset, index: 4)
                    Dispatch.dispatch1D(enc, countPSO, count: subBlocks)
                } else {
                    enc.setBuffer(srcValues.mtl, offset: srcValues.offset, index: 0)
                    enc.setBuffer(vv.mtl, offset: vv.offset, index: 1)
                    Dispatch.setLength(enc, srcN, nil, index: 2)
                    Dispatch.setUInt(enc, hasV ? 1 : 0, index: 3)
                    Dispatch.setUInt(enc, srcInv, index: 4)
                    setKey(enc, loKey, index: 5)
                    setKey(enc, hiKey, index: 6)
                    Dispatch.setUInt(enc, eps, index: 7)
                    enc.setBuffer(blockLt.mtl, offset: blockLt.offset, index: 8)
                    enc.setBuffer(blockEq.mtl, offset: blockEq.offset, index: 9)
                    enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
                }
                enc.memoryBarrier(scope: .buffers)
                // The winners take [0, winners); the bin's rows start right after them.
                for (buf, base) in [(blockLt, 0), (blockEq, winners)] {
                    enc.setComputePipelineState(scanPSO)
                    enc.setBuffer(buf.mtl, offset: buf.offset, index: 0)
                    Dispatch.setUInt(enc, subBlocks, index: 1)
                    Dispatch.setUInt(enc, base, index: 2)
                    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
                }
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(scatterPSO)
                enc.setBuffer(srcValues.mtl, offset: srcValues.offset, index: 0)
                enc.setBuffer(vv.mtl, offset: vv.offset, index: 1)
                Dispatch.setLength(enc, srcN, nil, index: 2)
                Dispatch.setUInt(enc, hasV ? 1 : 0, index: 3)
                Dispatch.setUInt(enc, srcInv, index: 4)
                setKey(enc, loKey, index: 5)
                setKey(enc, hiKey, index: 6)
                Dispatch.setUInt(enc, eps, index: 7)
                Dispatch.setUInt(enc, m, index: 8)
                enc.setBuffer(blockLt.mtl, offset: blockLt.offset, index: 9)
                enc.setBuffer(blockEq.mtl, offset: blockEq.offset, index: 10)
                enc.setBuffer(keysOut.mtl, offset: keysOut.offset, index: 11)
                enc.setBuffer(rowsOut.mtl, offset: rowsOut.offset, index: 12)
                enc.setBuffer((srcRows ?? srcValues).mtl, offset: (srcRows ?? srcValues).offset, index: 13)
                Dispatch.setUInt(enc, srcRows == nil ? 0 : 1, index: 14)
                Dispatch.setUInt(enc, 1, index: 15)       // the winners are part of the answer here
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
            }
            ctx.retainUntilFlush(blockLt); ctx.retainUntilFlush(blockEq)
            ctx.retainUntilFlush(keysOut); ctx.retainUntilFlush(rowsOut)

            outKeys = keysOut; outRows = rowsOut; outLength = m
            // --- 4: refine on the compacted array while the bin is still worth shrinking ---
            if collapsed || tail <= TopK.refineCap || m <= TopK.bitonicCap { break }
            prefix = (prefix << 8) | UInt64(target)
            shift -= 8
            srcKind = wide ? "u64" : "u32"          // reading keys back is the identity mapping
            srcType = kd.keyType
            srcValues = keysOut
            srcRows = rowsOut
            srcValidity = nil
            srcN = m
            srcInv = 0
        }

        // --- 5: one stable ordering by key turns row order into (key, row) order ---
        let m = outLength
        if m <= TopK.bitonicCap {
            var cap = 1
            while cap < m { cap <<= 1 }
            let pso = try pipelines(srcKind, srcType)("rs_sort_small")
            let out = try MetalArrowBuffer.allocate(byteCount: k * 4, zeroed: false, context: ctx)
            let keys = outKeys, rows = outRows
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(keys.mtl, offset: keys.offset, index: 0)
                enc.setBuffer(rows.mtl, offset: rows.offset, index: 1)
                Dispatch.setUInt(enc, m, index: 2)
                Dispatch.setUInt(enc, cap, index: 3)
                Dispatch.setUInt(enc, k, index: 4)
                enc.setBuffer(out.mtl, offset: out.offset, index: 5)
                enc.setThreadgroupMemoryLength(roundUp(cap * keyBytes, to: 16), index: 0)
                enc.setThreadgroupMemoryLength(roundUp(cap * 4, to: 16), index: 1)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
            }
            return MetalArray<Int32>(length: k, nullCount: 0, validity: nil, values: out, context: ctx)
        }
        let ord: MetalArray<Int32>
        if wide {
            ord = try MetalArray<UInt64>(length: m, nullCount: 0, validity: nil, values: outKeys, context: ctx).argsort()
        } else {
            ord = try MetalArray<UInt32>(length: m, nullCount: 0, validity: nil, values: outKeys, context: ctx).argsort()
        }
        let rows = MetalArray<UInt32>(length: m, nullCount: 0, validity: nil, values: outRows, context: ctx)
        return try rows.take(try ord.slice(offset: 0, length: k)).cast(to: Int32.self)
    }
}
