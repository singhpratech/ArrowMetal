import Foundation
import Metal

/// Group-by over dense integer keys `0 ..< keyCount`. Keys outside the range and null keys are skipped.
/// This is the shape produced by dictionary encoding or by small categorical columns; hashing of
/// arbitrary keys is a separate (future) step that produces such codes.
public struct GroupBy<K: ArrowIndex> {
    public let keys: MetalArray<K>
    public let keyCount: Int

    public init(keys: MetalArray<K>, keyCount: Int) throws {
        guard keyCount > 0, keyCount <= Int(UInt32.max) / 4 else { throw ArrowMetalError.invalidArrowArray("keyCount out of range") }
        self.keys = keys
        self.keyCount = keyCount
    }

    /// Number of rows per key (rows with a null key are not counted).
    public func count() throws -> MetalArray<Int64> {
        let (_, counts) = try run(values: keys, kind: 3)
        return MetalArray<Int64>(length: keyCount, nullCount: 0, validity: nil, values: counts, context: keys.context)
    }

    /// Sum of non-null values per key. Integer sums are exact 64-bit (wrapping); Float32 sums accumulate in Float32.
    /// Keys with no valid value are null.
    public func sum<T: ArrowPrimitive>(_ values: MetalArray<T>) throws -> MetalArray<Int64> where T: FixedWidthInteger {
        try check(values)
        let (out, counts) = try run(values: values, kind: T.minValue < 0 ? 0 : 1)
        return finish(Int64.self, out: out, counts: counts, ctx: values.context)
    }

    public func sumFloat(_ values: MetalArray<Float>) throws -> MetalArray<Double> {
        try check(values)
        let (out, counts) = try run(values: values, kind: 2)
        let res = try MetalArray<Double>.allocate(length: keyCount, withValidity: true, context: values.context)
        withExtendedLifetime((out, counts)) {
            let o = out.typed(UInt64.self), c = counts.typed(UInt64.self)
            let d = res.mutableValuePointer, v = res.validity!.mutableTyped(UInt8.self)
            for k in 0..<keyCount where c[k] > 0 { d[k] = Double(Float(bitPattern: UInt32(truncatingIfNeeded: o[k]))); Bitmap.set(v, k) }
        }
        res.recomputeNullCount()
        return res
    }

    /// Mean of non-null values per key.
    public func mean<T: ArrowPrimitive>(_ values: MetalArray<T>) throws -> MetalArray<Double> where T: FixedWidthInteger {
        let s = try sum(values), c = try count(values)
        let res = try MetalArray<Double>.allocate(length: keyCount, withValidity: true, context: values.context)
        let d = res.mutableValuePointer, v = res.validity!.mutableTyped(UInt8.self)
        for k in 0..<keyCount where c.valuePointer[k] > 0 { d[k] = Double(s.valuePointer[k]) / Double(c.valuePointer[k]); Bitmap.set(v, k) }
        res.recomputeNullCount()
        return res
    }

    /// Number of non-null values per key.
    public func count<T: ArrowPrimitive>(_ values: MetalArray<T>) throws -> MetalArray<Int64> {
        try check(values)
        let (_, counts) = try run(values: values, kind: 3, countValues: true)
        return MetalArray<Int64>(length: keyCount, nullCount: 0, validity: nil, values: counts, context: values.context)
    }

    /// Min / max per key for 32-bit-or-narrower types (Int8/16/32, UInt8/16/32, Float32). NaN is skipped.
    public func min<T: ArrowPrimitive>(_ values: MetalArray<T>) throws -> MetalArray<T> { try minMax(values, isMin: true) }
    public func max<T: ArrowPrimitive>(_ values: MetalArray<T>) throws -> MetalArray<T> { try minMax(values, isMin: false) }

    private func minMax<T: ArrowPrimitive>(_ values: MetalArray<T>, isMin: Bool) throws -> MetalArray<T> {
        try check(values)
        guard T.byteWidth <= 4 else { throw ArrowMetalError.unsupportedType("group-by min/max needs a 32-bit or narrower type (64-bit atomics are not available); cast first") }
        let kind: Int
        if T.isFloatingPoint { kind = isMin ? 8 : 9 } else if T.minValue < 0 as T { kind = isMin ? 4 : 5 } else { kind = isMin ? 6 : 7 }
        let (out, counts) = try run(values: values, kind: kind)
        let res = try MetalArray<T>.allocate(length: keyCount, withValidity: true, context: values.context)
        withExtendedLifetime((out, counts)) {
            let o = out.typed(UInt64.self), c = counts.typed(UInt64.self)
            let d = res.mutableValuePointer, v = res.validity!.mutableTyped(UInt8.self)
            for k in 0..<keyCount where c[k] > 0 {
                let w = UInt32(truncatingIfNeeded: o[k])
                if T.isFloatingPoint {
                    let key = Int32(bitPattern: w)
                    let bits = key < 0 ? key ^ 0x7FFF_FFFF : key
                    d[k] = T(Float(bitPattern: UInt32(bitPattern: bits)))
                } else if T.minValue < 0 as T {
                    d[k] = T(truncatingIfNeededInt64: Int64(Int32(bitPattern: w)))
                } else {
                    d[k] = T(truncatingIfNeededUInt64: UInt64(w))
                }
                Bitmap.set(v, k)
            }
        }
        res.recomputeNullCount()
        return res
    }

    private func check<T>(_ values: MetalArray<T>) throws {
        guard values.length == keys.length else { throw ArrowMetalError.lengthMismatch(keys.length, values.length) }
    }

    private func finish(_: Int64.Type, out: MetalArrowBuffer, counts: MetalArrowBuffer, ctx: MetalContext) -> MetalArray<Int64> {
        let res = try! MetalArray<Int64>.allocate(length: keyCount, withValidity: true, context: ctx)
        withExtendedLifetime((out, counts)) {
            let o = out.typed(Int64.self), c = counts.typed(UInt64.self)
            let d = res.mutableValuePointer, v = res.validity!.mutableTyped(UInt8.self)
            for k in 0..<keyCount where c[k] > 0 { d[k] = o[k]; Bitmap.set(v, k) }
        }
        res.recomputeNullCount()
        return res
    }

    /// Runs accumulation (+ finalize) and returns (out[K] as ulong, counts[K] as ulong).
    private func run<T: ArrowPrimitive>(values: MetalArray<T>, kind: Int, countValues: Bool = false) throws -> (MetalArrowBuffer, MetalArrowBuffer) {
        try Dispatch.checkLength(keys.length)
        let ctx = keys.context
        let n = keys.length
        let K = keyCount
        let valueType = Dispatch.moveType(T.self) == "long" && T.self == Double.self ? "float" : T.mslType  // Double is not supported on GPU here
        if T.self == Double.self { throw ArrowMetalError.unsupportedType("group-by over Float64 values: cast to Float32 first") }
        let src = GroupBySource.source(T: valueType, KT: K_.mslType)
        let out = try MetalArrowBuffer.allocate(byteCount: K * 8, zeroed: false, context: ctx)
        let counts = try MetalArrowBuffer.allocate(byteCount: K * 8, zeroed: false, context: ctx)
        let flags = (keys.validity == nil ? 0 : 1) | ((values.validity == nil || (kind == 3 && !countValues)) ? 0 : 2)
        let effectiveKind = kind == 3 ? 3 : kind
        let tg = MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1)
        if K <= GroupBySource.maxPrivateKeys {
            let numTG = Swift.max(1, Swift.min(1024, (n + 4095) / 4096))
            let chunk = (n + numTG - 1) / numTG
            let partials = try MetalArrowBuffer.allocate(byteCount: numTG * K * 8, zeroed: false, context: ctx)
            let pcounts = try MetalArrowBuffer.allocate(byteCount: numTG * K * 4, zeroed: false, context: ctx)
            let accPSO = try Dispatch.pipeline(ctx, family: "groupby", source: src, function: "gb_accumulate_priv", type: "\(valueType)/\(K_.mslType)")
            let finPSO = try Dispatch.pipeline(ctx, family: "groupby", source: src, function: "gb_finalize", type: "\(valueType)/\(K_.mslType)")
            try ctx.run { enc in
                enc.setComputePipelineState(accPSO)
                bindInputs(enc, values: values, n: n, flags: flags, K: K, kind: effectiveKind, chunk: chunk)
                enc.setBuffer(partials.mtl, offset: 0, index: 12)
                enc.setBuffer(pcounts.mtl, offset: 0, index: 13)
                enc.dispatchThreadgroups(MTLSize(width: numTG, height: 1, depth: 1), threadsPerThreadgroup: tg)
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(finPSO)
                enc.setBuffer(partials.mtl, offset: 0, index: 0)
                enc.setBuffer(pcounts.mtl, offset: 0, index: 1)
                Dispatch.setUInt(enc, K, index: 2)
                Dispatch.setUInt(enc, numTG, index: 3)
                Dispatch.setUInt(enc, effectiveKind, index: 4)
                enc.setBuffer(out.mtl, offset: 0, index: 5)
                enc.setBuffer(counts.mtl, offset: 0, index: 6)
                Dispatch.dispatch1D(enc, finPSO, count: K)
            }
        } else {
            let numTG = Swift.max(1, Swift.min(4096, (n + 4095) / 4096))
            let chunk = (n + numTG - 1) / numTG
            let lo = try MetalArrowBuffer.allocate(byteCount: K * 4, zeroed: false, context: ctx)
            let hi = try MetalArrowBuffer.allocate(byteCount: K * 4, context: ctx)
            let cnt = try MetalArrowBuffer.allocate(byteCount: K * 4, context: ctx)
            // Initialise lo with the identity for the kind.
            withExtendedLifetime(lo) {
                let p = lo.mutableTyped(UInt32.self)
                let initVal: UInt32
                switch effectiveKind {
                case 4: initVal = UInt32(bitPattern: Int32.max)
                case 5: initVal = UInt32(bitPattern: Int32.min)
                case 6: initVal = UInt32.max
                case 8: initVal = UInt32(bitPattern: 0x7F80_0000)          // f_key(+inf)
                case 9: initVal = UInt32(bitPattern: Int32(bitPattern: 0xFF80_0000) ^ 0x7FFF_FFFF) // f_key(-inf)
                default: initVal = 0
                }
                for k in 0..<K { p[k] = initVal }
            }
            let accPSO = try Dispatch.pipeline(ctx, family: "groupby", source: src, function: "gb_accumulate_dev", type: "\(valueType)/\(K_.mslType)")
            let packPSO = try Dispatch.pipeline(ctx, family: "groupby", source: src, function: "gb_pack_dev", type: "\(valueType)/\(K_.mslType)")
            try ctx.run { enc in
                enc.setComputePipelineState(accPSO)
                bindInputs(enc, values: values, n: n, flags: flags, K: K, kind: effectiveKind, chunk: chunk)
                enc.setBuffer(lo.mtl, offset: 0, index: 9)
                enc.setBuffer(hi.mtl, offset: 0, index: 10)
                enc.setBuffer(cnt.mtl, offset: 0, index: 11)
                enc.setBuffer(out.mtl, offset: 0, index: 12)     // unused on this path
                enc.setBuffer(counts.mtl, offset: 0, index: 13)
                enc.dispatchThreadgroups(MTLSize(width: numTG, height: 1, depth: 1), threadsPerThreadgroup: tg)
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(packPSO)
                enc.setBuffer(lo.mtl, offset: 0, index: 0)
                enc.setBuffer(hi.mtl, offset: 0, index: 1)
                enc.setBuffer(cnt.mtl, offset: 0, index: 2)
                Dispatch.setUInt(enc, K, index: 3)
                enc.setBuffer(out.mtl, offset: 0, index: 4)
                enc.setBuffer(counts.mtl, offset: 0, index: 5)
                Dispatch.dispatch1D(enc, packPSO, count: K)
            }
        }
        try ctx.syncPoint()   // results are read on the CPU next
        return (out, counts)
    }

    private func bindInputs<T>(_ enc: MTLComputeCommandEncoder, values: MetalArray<T>, n: Int, flags: Int, K: Int, kind: Int, chunk: Int) {
        enc.setBuffer(keys.values.mtl, offset: keys.values.offset, index: 0)
        let kv = keys.validity ?? keys.values
        enc.setBuffer(kv.mtl, offset: kv.offset, index: 1)
        enc.setBuffer(values.values.mtl, offset: values.values.offset, index: 2)
        let vv = values.validity ?? values.values
        enc.setBuffer(vv.mtl, offset: vv.offset, index: 3)
        Dispatch.setUInt(enc, n, index: 4)
        Dispatch.setUInt(enc, flags, index: 5)
        Dispatch.setUInt(enc, K, index: 6)
        Dispatch.setUInt(enc, kind, index: 7)
        Dispatch.setUInt(enc, chunk, index: 8)
    }

    private typealias K_ = K
}

extension MetalArray where T: ArrowIndex {
    /// Convenience: `keys.groupBy(keyCount:)`.
    public func groupBy(keyCount: Int) throws -> GroupBy<T> { try GroupBy(keys: self, keyCount: keyCount) }
}
