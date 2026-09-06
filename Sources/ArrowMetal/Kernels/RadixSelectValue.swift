import Foundation
import Metal

/// Order statistics: the value at a given rank, without sorting the column.
///
/// `topKRadixSelect` has to materialise the k winning rows; a quantile does not — it only wants the *key* at
/// one rank, so the compaction can throw the winners away and keep the candidate bin alone. That makes the
/// median of 50M rows two passes over the column plus a few thousand keys read back to the host, instead of
/// the eight radix passes and the 400 MB gather a full sort costs.
///
/// The narrowing is the same as `RadixSelect.swift`: histogram the top digit, find the bin the rank falls in,
/// then repeat on the next digit restricted to that bin. Once the bin fits in a few thousand keys it is read
/// back and finished on the host, which ends the round trips immediately rather than after another six
/// digits. Keys carry the whole ordering (`TopKSource.keyMap`), so nulls, NaN and -0.0 land exactly where
/// `argsort` puts them.
extension TopK {
    /// Bin size at which the search stops narrowing on the GPU and finishes on the host. A few thousand keys
    /// is well under a millisecond to sort, and it saves a command buffer round trip per remaining digit.
    static let hostFinishCap = 4096

    /// The value whose order-preserving key is `stored`, undoing `TopKSource.keyMap`.
    ///
    /// Exact for every integer type. For floats it is exact except on the two values the key deliberately
    /// merges: -0.0 reads back as +0.0, and a NaN reads back as a canonical NaN rather than with its original
    /// payload. Those are the values Arrow's total order calls equal anyway.
    static func value<T: ArrowPrimitive>(fromKey stored: UInt64, largest: Bool, _: T.Type) -> T {
        guard let kd = radixKind(T.self) else { return 0 }
        let mask: UInt64 = kd.keyBits == 64 ? .max : (UInt64(1) << UInt64(kd.keyBits)) - 1
        var k = stored
        if largest {
            // Only a NaN can invert to the maximum key (see `SortSource.key_from_f32`).
            if k == mask && (kd.kind == "f32" || kd.kind == "f64") { return T(Float.nan) }
            k = mask - k
        }
        switch kd.kind {
        case "i8": return Int8(truncatingIfNeeded: Int(k) &- 128) as! T
        case "u8": return UInt8(truncatingIfNeeded: k) as! T
        case "i16": return Int16(truncatingIfNeeded: Int(k) &- 32768) as! T
        case "u16": return UInt16(truncatingIfNeeded: k) as! T
        case "i32": return Int32(bitPattern: UInt32(truncatingIfNeeded: k) ^ 0x8000_0000) as! T
        case "u32": return UInt32(truncatingIfNeeded: k) as! T
        case "f32":
            let key = UInt32(truncatingIfNeeded: k)
            return Float(bitPattern: (key & 0x8000_0000) != 0 ? key & 0x7FFF_FFFF : ~key) as! T
        case "i64": return Int64(bitPattern: k ^ 0x8000_0000_0000_0000) as! T
        case "u64": return k as! T
        default:
            return Double(bitPattern: (k & 0x8000_0000_0000_0000) != 0 ? k & 0x7FFF_FFFF_FFFF_FFFF : ~k) as! T
        }
    }
}

extension MetalArray {
    /// The order-preserving keys at ranks `rank` and `rank + 1` among the non-null values, ordered by
    /// `largest`. The second key is nil when the two ranks fall in different bins, which only happens when
    /// `rank` is the last row of the bin the search settled on; the caller re-runs for it.
    ///
    /// Returns nil when the type has no key mapping or the rank is out of range.
    func radixSelectKey(rank: Int, largest: Bool) throws -> (key: UInt64, next: UInt64?)? {
        guard let kd = TopK.radixKind(T.self) else { return nil }
        let n = length
        let valid = n - nullCount
        guard rank >= 0, rank < valid else { return nil }
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

        var srcKind = kd.kind, srcType = kd.valueType
        var srcValues = values
        var srcValidity = validity
        var srcN = n
        var srcInv = largest ? 1 : 0

        var prefix: UInt64 = 0
        var shift = kd.keyBits - 8
        var below = 0                                // rows whose key is below the current window
        let k = rank + 1

        while true {
            let src = RadixSelectSource.source(kind: srcKind, V: srcType, K: kd.keyType)
            func pso(_ f: String) throws -> MTLComputePipelineState {
                try Dispatch.pipeline(ctx, family: "radixselect", source: src, function: f, type: srcKind)
            }
            let (eps, subBlocks, groups) = TopK.radixPlan(n: srcN)
            let hasPrefix = shift + 8 < kd.keyBits
            let vv = srcValidity ?? srcValues
            let hasV = srcValidity != nil
            let grid = MTLSize(width: groups, height: 1, depth: 1)

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
            try ctx.syncPoint()

            var cum = 0, target = 0, binCount = 0
            var found = false
            withExtendedLifetime(totals) {
                let t = totals.typed(UInt32.self)
                for d in 0..<radix {
                    let c = Int(t[d])
                    if below + cum + c >= k { target = d; binCount = c; found = true; break }
                    cum += c
                }
            }
            guard found else { return nil }
            below += cum
            let full = (prefix << 8) | UInt64(target)

            if shift == 0 {
                // Every row left has this exact key, so the next rank shares it unless it is past the bin.
                let key = full
                return (key, rank + 1 < below + binCount ? key : nil)
            }
            let loKey = full << UInt64(shift)
            let hiKey = loKey | ((UInt64(1) << UInt64(shift)) - 1)

            if binCount > TopK.candidateCap(k: k) {
                // Compacting would copy most of the input; narrow another digit in place instead.
                prefix = full; shift -= 8
                continue
            }

            // Compact the bin's keys — and only those: the winners are counted, never written.
            let blockLt = try MetalArrowBuffer.allocate(byteCount: subBlocks * 4, zeroed: false, context: ctx)
            let blockEq = try MetalArrowBuffer.allocate(byteCount: subBlocks * 4, zeroed: false, context: ctx)
            let keysOut = try MetalArrowBuffer.allocate(byteCount: Swift.max(binCount, 1) * keyBytes, zeroed: false, context: ctx)
            let rowsOut = try MetalArrowBuffer.allocate(byteCount: Swift.max(binCount, 1) * 4, zeroed: false, context: ctx)
            let scanPSO = try pso("rs_scan"), scatterPSO = try pso("rs_scatter")
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
                enc.setComputePipelineState(scanPSO)
                enc.setBuffer(blockEq.mtl, offset: blockEq.offset, index: 0)
                Dispatch.setUInt(enc, subBlocks, index: 1)
                Dispatch.setUInt(enc, 0, index: 2)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
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
                Dispatch.setUInt(enc, binCount, index: 8)
                enc.setBuffer(blockLt.mtl, offset: blockLt.offset, index: 9)
                enc.setBuffer(blockEq.mtl, offset: blockEq.offset, index: 10)
                enc.setBuffer(keysOut.mtl, offset: keysOut.offset, index: 11)
                enc.setBuffer(rowsOut.mtl, offset: rowsOut.offset, index: 12)
                enc.setBuffer(srcValues.mtl, offset: srcValues.offset, index: 13)
                Dispatch.setUInt(enc, 0, index: 14)
                Dispatch.setUInt(enc, 0, index: 15)      // winners are counted, not written
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
            }
            ctx.retainUntilFlush(blockLt); ctx.retainUntilFlush(blockEq)
            ctx.retainUntilFlush(keysOut); ctx.retainUntilFlush(rowsOut)

            if binCount <= TopK.hostFinishCap {
                // Few enough keys that sorting them on the host beats another round trip per digit. These are
                // whole keys, so this ends the search however many digits were left.
                try ctx.syncPoint()
                var bin = [UInt64]()
                bin.reserveCapacity(binCount)
                withExtendedLifetime(keysOut) {
                    if wide {
                        let p = keysOut.typed(UInt64.self)
                        for i in 0..<binCount { bin.append(p[i]) }
                    } else {
                        let p = keysOut.typed(UInt32.self)
                        for i in 0..<binCount { bin.append(UInt64(p[i])) }
                    }
                }
                bin.sort()
                let i = rank - below
                guard i >= 0, i < bin.count else { return nil }
                return (bin[i], i + 1 < bin.count ? bin[i + 1] : nil)
            }

            prefix = full
            shift -= 8
            srcKind = wide ? "u64" : "u32"        // reading keys back is the identity mapping
            srcType = kd.keyType
            srcValues = keysOut
            srcValidity = nil
            srcN = binCount
            srcInv = 0
        }
    }

    /// The k-th smallest (or largest) non-null value, 1-based: `kthElement(1)` is the minimum,
    /// `kthElement(1, largest: true)` the maximum. Nil when k is outside the non-null rows.
    ///
    /// A GPU radix select on the order-preserving key — two passes over the column rather than the eight
    /// passes plus a gather that sorting costs. The ordering is `argsort`'s: nulls are skipped and NaN counts
    /// as the largest value. See `TopK.value(fromKey:)` for the one way the answer can differ from indexing a
    /// sorted copy: -0.0 comes back as +0.0 and a NaN loses its payload, values Arrow's order calls equal.
    public func kthElement(_ k: Int, largest: Bool = false) throws -> T? {
        guard k >= 1, k <= validCount else { return nil }
        guard let found = try radixSelectKey(rank: k - 1, largest: largest) else {
            // No key mapping for this type: fall back to sorting.
            let s = try sorted(descending: largest)
            return withExtendedLifetime(s) { s.valuePointer[k - 1] }
        }
        return TopK.value(fromKey: found.key, largest: largest, T.self)
    }
}
