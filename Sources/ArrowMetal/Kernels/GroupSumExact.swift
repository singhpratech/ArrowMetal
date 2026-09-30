import Foundation
import Metal

/// Grouped Float64 `sum` and `mean`, correctly rounded, with no group order.
///
/// See `GroupSumExactSource` for the kernels. One call produces the sum, the mean and their validity
/// together, so a `sum` and a `mean` of the same column over the same grouping share one pass pair
/// (`GroupBy.cache`). The few groups the GPU window cannot settle are summed exactly on the host.
extension GroupBy {

    /// The per-group sum and mean of a Float64 column, both correctly rounded.
    struct ExactSumMean {
        let sum: MetalArray<Double>
        let mean: MetalArray<Double>
    }

    /// Guard bits of the fixed-point window for `rows` rows: 53 significand bits + G + bitlength(rows)
    /// stay at or below 127, the magnitude bits of the 128-bit accumulator.
    static func exactSumGuardBits(rows: Int) -> Int {
        let bits = rows <= 0 ? 1 : (Int.bitWidth - rows.leadingZeroBitCount)
        return 74 - bits
    }

    func exactSumMean(_ values: MetalArray<Double>) throws -> ExactSumMean {
        guard values.length == keys.length else { throw ArrowMetalError.lengthMismatch(keys.length, values.length) }
        if let hit = cache.lookup(.exact, values) as? ExactSumMean { return hit }
        try Dispatch.checkLength(keys.length)
        let ctx = keys.context
        let n = keys.length
        let kc = keyCount
        let G = Self.exactSumGuardBits(rows: n)
        let src = GroupSumExactSource.source(KT: K.mslType)
        func pso(_ f: String) throws -> MTLComputePipelineState {
            try Dispatch.pipeline(ctx, family: "groupsumexact", source: src, function: f, type: K.mslType)
        }
        let st = try MetalArrowBuffer.allocate(byteCount: kc * GroupSumExactSource.words * 4, zeroed: false, context: ctx)
        let outSum = try MetalArrowBuffer.allocate(byteCount: Swift.max(kc, 1) * 8, zeroed: false, context: ctx)
        let outMean = try MetalArrowBuffer.allocate(byteCount: Swift.max(kc, 1) * 8, zeroed: false, context: ctx)
        let validBytes = try MetalArrowBuffer.allocate(byteCount: Swift.max(kc, 1), zeroed: false, context: ctx)
        let hostGroup = try MetalArrowBuffer.allocate(byteCount: Swift.max(kc, 1), zeroed: false, context: ctx)
        let hostCount = try MetalArrowBuffer.allocate(byteCount: 8, context: ctx)

        let priv = kc <= GroupSumExactSource.maxPrivateKeys
        let numTG = Swift.max(1, Swift.min(priv ? 1024 : 4096, (n + 4095) / 4096))
        let chunk = (n + numTG - 1) / numTG
        let flags = (keys.validity == nil ? 0 : 1) | (values.validity == nil ? 0 : 2)
        let initPSO = try pso("gs_init"), finPSO = try pso("gs_finalize")
        let aPSO = try pso(priv ? "gs_a_priv" : "gs_a_dev"), bPSO = try pso(priv ? "gs_b_priv" : "gs_b_dev")

        func bindRows(_ enc: MTLComputeCommandEncoder) {
            enc.setBuffer(keys.values.mtl, offset: keys.values.offset, index: 0)
            let kv = keys.validity ?? keys.values
            enc.setBuffer(kv.mtl, offset: kv.offset, index: 1)
            enc.setBuffer(values.values.mtl, offset: values.values.offset, index: 2)
            let vv = values.validity ?? values.values
            enc.setBuffer(vv.mtl, offset: vv.offset, index: 3)
            Dispatch.setUInt(enc, n, index: 4)
            Dispatch.setUInt(enc, flags, index: 5)
            Dispatch.setUInt(enc, kc, index: 6)
            Dispatch.setUInt(enc, chunk, index: 7)
            Dispatch.setUInt(enc, G, index: 8)
            enc.setBuffer(st.mtl, offset: 0, index: 9)
        }

        var bmSum: MetalArrowBuffer! = nil, bmMean: MetalArrowBuffer! = nil
        try ctx.batch {
            try ctx.run { enc in
                enc.setComputePipelineState(initPSO)
                enc.setBuffer(st.mtl, offset: 0, index: 0)
                Dispatch.setUInt(enc, kc, index: 1)
                Dispatch.dispatch1D(enc, initPSO, count: kc)
                enc.memoryBarrier(scope: .buffers)
                if n > 0 {
                    let grid = MTLSize(width: numTG, height: 1, depth: 1)
                    let tg = MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1)
                    enc.setComputePipelineState(aPSO)
                    bindRows(enc)
                    enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
                    enc.memoryBarrier(scope: .buffers)
                    enc.setComputePipelineState(bPSO)
                    bindRows(enc)
                    enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
                    enc.memoryBarrier(scope: .buffers)
                }
                enc.setComputePipelineState(finPSO)
                enc.setBuffer(st.mtl, offset: 0, index: 0)
                Dispatch.setUInt(enc, kc, index: 1)
                Dispatch.setUInt(enc, G, index: 2)
                enc.setBuffer(outSum.mtl, offset: 0, index: 3)
                enc.setBuffer(outMean.mtl, offset: 0, index: 4)
                enc.setBuffer(validBytes.mtl, offset: 0, index: 5)
                enc.setBuffer(hostGroup.mtl, offset: 0, index: 6)
                enc.setBuffer(hostCount.mtl, offset: 0, index: 7)
                Dispatch.dispatch1D(enc, finPSO, count: kc)
            }
            ctx.retainUntilFlush(keys); ctx.retainUntilFlush(values); ctx.retainUntilFlush(st)
            ctx.retainUntilFlush(validBytes); ctx.retainUntilFlush(hostGroup); ctx.retainUntilFlush(hostCount)
            bmSum = try BitmapOps.packBits(ctx, bytes: validBytes, bits: kc)
            bmMean = try BitmapOps.packBits(ctx, bytes: validBytes, bits: kc)
        }
        try ctx.syncPoint()
        let (hostGroups, hostRows) = withExtendedLifetime(hostCount) {
            let p = hostCount.typed(UInt32.self)
            return (Int(p[0]), Int(p[1]))
        }
        if hostGroups > 0 {
            try exactSumHostFallback(values, rows: hostRows, hostGroup: hostGroup, outSum: outSum, outMean: outMean)
        }
        let sum = MetalArray<Double>(length: kc, nullCount: 0, validity: bmSum, values: outSum, context: ctx)
        let mean = MetalArray<Double>(length: kc, nullCount: 0, validity: bmMean, values: outMean, context: ctx)
        sum.recomputeNullCount(); mean.recomputeNullCount()
        let result = ExactSumMean(sum: sum, mean: mean)
        cache.store(.exact, values, result)
        return result
    }

    /// Sums the flagged groups exactly on the host: their rows come from one compaction kernel, then an
    /// exact fixed-point accumulator per group (`ExactHostSum`) rounds once.
    private func exactSumHostFallback(_ values: MetalArray<Double>, rows total: Int, hostGroup: MetalArrowBuffer,
                                      outSum: MetalArrowBuffer, outMean: MetalArrowBuffer) throws {
        let ctx = keys.context
        let n = keys.length
        let rows = try MetalArrowBuffer.allocate(byteCount: Swift.max(total, 1) * 4, zeroed: false, context: ctx)
        let cursor = try MetalArrowBuffer.allocate(byteCount: 4, context: ctx)
        let src = GroupSumExactSource.source(KT: K.mslType)
        let pso = try Dispatch.pipeline(ctx, family: "groupsumexact", source: src, function: "gs_host_rows", type: K.mslType)
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(keys.values.mtl, offset: keys.values.offset, index: 0)
            let kv = keys.validity ?? keys.values
            enc.setBuffer(kv.mtl, offset: kv.offset, index: 1)
            let vv = values.validity ?? values.values
            enc.setBuffer(vv.mtl, offset: vv.offset, index: 2)
            Dispatch.setUInt(enc, n, index: 3)
            Dispatch.setUInt(enc, (keys.validity == nil ? 0 : 1) | (values.validity == nil ? 0 : 2), index: 4)
            Dispatch.setUInt(enc, keyCount, index: 5)
            enc.setBuffer(hostGroup.mtl, offset: 0, index: 6)
            enc.setBuffer(cursor.mtl, offset: 0, index: 7)
            enc.setBuffer(rows.mtl, offset: 0, index: 8)
            Dispatch.dispatch1D(enc, pso, count: n)
        }
        ctx.retainUntilFlush(rows); ctx.retainUntilFlush(cursor)
        try ctx.syncPoint()
        withExtendedLifetime((rows, cursor, outSum, outMean, keys, values)) {
            let got = Swift.min(Int(cursor.typed(UInt32.self)[0]), total)
            let r = rows.typed(UInt32.self)
            // The rows listed all have a key in [0, keyCount), so its bits read as a non-negative value.
            let kp = UnsafeRawPointer(keys.valuePointer)
            let wide = K.byteWidth == 8
            let vp = UnsafeRawPointer(values.valuePointer).assumingMemoryBound(to: UInt64.self)
            var acc: [Int: ExactHostSum] = [:]
            for j in 0..<got {
                let row = Int(r[j])
                let k = wide ? Int(kp.load(fromByteOffset: row * 8, as: Int64.self))
                             : Int(kp.load(fromByteOffset: row * 4, as: UInt32.self))
                acc[k, default: ExactHostSum()].add(vp[row], row: row)
            }
            let s = outSum.mutableTyped(UInt64.self), m = outMean.mutableTyped(UInt64.self)
            for (k, a) in acc {
                s[k] = a.value(mean: false)
                m[k] = a.value(mean: true)
            }
        }
    }
}

/// An exact sum of binary64 values: a two's-complement fixed-point integer whose bit 0 weighs 2^-1074
/// (the smallest subnormal), wide enough for 2^32 terms of the largest finite value, plus the special
/// values seen. Rounded once, to nearest-even, by `value(mean:)`.
struct ExactHostSum {
    static let limbCount = 35                  // 2,240 bits: 2,098 value bits + 33 of headroom + sign
    var limbs = [UInt64](repeating: 0, count: ExactHostSum.limbCount)
    var count = 0
    var nonzero = false, posZero = false, posInf = false, negInf = false
    var nanRow = Int.max
    var nanBits: UInt64 = 0

    mutating func add(_ bits: UInt64, row: Int) {
        count += 1
        let e = Int((bits >> 52) & 0x7FF)
        let mag = bits & 0x7FFF_FFFF_FFFF_FFFF
        if e == 0x7FF {
            if mag == 0x7FF0_0000_0000_0000 { if bits >> 63 != 0 { negInf = true } else { posInf = true } }
            else if row < nanRow { nanRow = row; nanBits = bits }
            return
        }
        if mag == 0 { if bits == 0 { posZero = true }; return }
        nonzero = true
        let m = (bits & 0xF_FFFF_FFFF_FFFF) | (e > 0 ? 1 << 52 : 0)
        let off = Swift.max(e, 1) - 1
        let limb = off / 64, sh = off % 64
        let lo = m << UInt64(sh)
        let hi = sh == 0 ? 0 : m >> UInt64(64 - sh)
        if bits >> 63 == 0 {
            let (r, c) = limbs[limb].addingReportingOverflow(lo)
            limbs[limb] = r
            var carry: UInt64 = c ? 1 : 0
            var j = limb + 1
            var add = hi
            while j < ExactHostSum.limbCount && (add != 0 || carry != 0) {
                let (a, o1) = limbs[j].addingReportingOverflow(add)
                let (b, o2) = a.addingReportingOverflow(carry)
                limbs[j] = b; carry = (o1 || o2) ? 1 : 0; add = 0; j += 1
            }
        } else {
            let (r, b0) = limbs[limb].subtractingReportingOverflow(lo)
            limbs[limb] = r
            var borrow: UInt64 = b0 ? 1 : 0
            var j = limb + 1
            var sub = hi
            while j < ExactHostSum.limbCount && (sub != 0 || borrow != 0) {
                let (a, o1) = limbs[j].subtractingReportingOverflow(sub)
                let (b, o2) = a.subtractingReportingOverflow(borrow)
                limbs[j] = b; borrow = (o1 || o2) ? 1 : 0; sub = 0; j += 1
            }
        }
    }

    /// The correctly rounded sum, or (mean) the correctly rounded quotient of the exact sum by the count.
    func value(mean: Bool) -> UInt64 {
        if nanRow != Int.max { return nanBits | (1 << 51) }
        if posInf && negInf { return 0x7FF8_0000_0000_0000 }
        if posInf { return 0x7FF0_0000_0000_0000 }
        if negInf { return 0xFFF0_0000_0000_0000 }
        let neg = limbs[ExactHostSum.limbCount - 1] >> 63 != 0
        var mag = limbs
        if neg {
            var carry: UInt64 = 1
            for i in 0..<mag.count { let (s, o) = (~mag[i]).addingReportingOverflow(carry); mag[i] = s; carry = o ? 1 : 0 }
        }
        if mag.allSatisfy({ $0 == 0 }) { return (nonzero || posZero) ? 0 : 0x8000_0000_0000_0000 }
        if !mean { return ExactHostSum.round(mag, lsbPos: 0, sticky: false, neg: neg) }
        // (mag << 128) / count, remainder sticky.
        let d = UInt64(count)
        var q = [UInt64](repeating: 0, count: mag.count + 2)
        var rem: UInt64 = 0
        let dividend = [0, 0] + mag
        for i in stride(from: dividend.count - 1, through: 0, by: -1) {
            let (qq, rr) = d.dividingFullWidth((high: rem, low: dividend[i]))
            q[i] = qq; rem = rr
        }
        return ExactHostSum.round(q, lsbPos: -128, sticky: rem != 0, neg: neg)
    }

    /// Round-to-nearest-even of (w + sticky fraction) * 2^(lsbPos - 1074) as binary64 bits.
    static func round(_ w: [UInt64], lsbPos: Int, sticky: Bool, neg: Bool) -> UInt64 {
        var top = -1
        for j in stride(from: w.count - 1, through: 0, by: -1) where w[j] != 0 {
            top = j * 64 + 63 - w[j].leadingZeroBitCount; break
        }
        let sgn: UInt64 = neg ? 1 << 63 : 0
        let P = top + lsbPos
        if P - 52 >= 2046 { return sgn | 0x7FF0_0000_0000_0000 }
        let s = P >= 52 ? top - 52 : -lsbPos
        func bits64(_ pos: Int) -> UInt64 {           // bits [pos, pos + 64), pos >= 0
            let j = pos / 64, b = pos % 64
            var r: UInt64 = j < w.count ? w[j] >> UInt64(b) : 0
            if b > 0 && j + 1 < w.count { r |= w[j + 1] << UInt64(64 - b) }
            return r
        }
        func anyBelow(_ pos: Int) -> Bool {
            if pos <= 0 { return false }
            let j = pos / 64, b = pos % 64
            for t in 0..<Swift.min(j, w.count) where w[t] != 0 { return true }
            return b > 0 && j < w.count && (w[j] & ((1 << UInt64(b)) - 1)) != 0
        }
        var r: UInt64
        if s <= 0 { r = bits64(0) << UInt64(-s) }
        else {
            let x = bits64(s - 1)
            let rb = x & 1 != 0
            r = (x >> 1) & ((1 << 54) - 1)
            if rb && (sticky || anyBelow(s - 1) || r & 1 != 0) { r += 1 }
        }
        var bits = P >= 52 ? r &+ (UInt64(P - 52) << 52) : r
        if bits >= 0x7FF0_0000_0000_0000 { bits = 0x7FF0_0000_0000_0000 }
        return sgn | bits
    }
}
