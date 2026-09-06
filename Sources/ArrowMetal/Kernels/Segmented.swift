import Foundation
import Metal

/// The sorted key order of a `GroupBy` plus `[start, end)` per key.
///
/// Building it is the expensive part of a segmented aggregate (one argsort of the keys), so it is a
/// value the caller can hold and pass to several aggregates.
public final class GroupSegments {
    /// Row index of each non-null key, ascending by key (stable, so rows inside a group keep their order).
    let ord: MetalArray<Int32>
    /// First and one-past-last sorted position of each key. A key with no rows keeps `start == end == 0`.
    let segStart: MetalArrowBuffer
    let segEnd: MetalArrowBuffer
    public let keyCount: Int
    public let rows: Int

    init(ord: MetalArray<Int32>, segStart: MetalArrowBuffer, segEnd: MetalArrowBuffer, keyCount: Int, rows: Int) {
        self.ord = ord; self.segStart = segStart; self.segEnd = segEnd; self.keyCount = keyCount; self.rows = rows
    }
}

/// Sort-based segmented aggregation for `GroupBy`.
///
/// `GroupBy`'s atomic path is bounded by what Metal's atomics can express: 32-bit only, so no 64-bit
/// min/max and no Float64 values. This path removes atomics from the aggregation entirely. The keys are
/// argsorted once, which makes every group one contiguous run of the sorted order; a threadgroup then
/// reduces one run privately and writes a single result. Float64 arithmetic uses the same software
/// binary64 implementation as `sum` (`DoubleMath`), so each addition is correctly rounded.
///
/// Nulls follow Arrow: a null key or a key outside `[0, keyCount)` contributes nothing, null values are
/// skipped, and a key with no valid value comes back null.
extension GroupBy {
    /// Argsorts the keys and marks the run of each key in the sorted order.
    public func segments() throws -> GroupSegments {
        try Dispatch.checkLength(keys.length)
        let ctx = keys.context
        let n = keys.length
        let m = n - keys.nullCount                       // argsort puts nulls last: the first m are the keys
        let segStart = try MetalArrowBuffer.allocate(byteCount: keyCount * 4, context: ctx)
        let segEnd = try MetalArrowBuffer.allocate(byteCount: keyCount * 4, context: ctx)
        let ord = m > 0 ? try keys.argsort().slice(offset: 0, length: m)
                        : try MetalArray<Int32>([Int32](), context: ctx)
        if m > 0 {
            let src = SegmentedSource.boundsSource(KT: K.mslType)
            let pso = try Dispatch.pipeline(ctx, family: "segmented", source: src, function: "seg_bounds", type: K.mslType)
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(keys.values.mtl, offset: keys.values.offset, index: 0)
                enc.setBuffer(ord.values.mtl, offset: ord.values.offset, index: 1)
                Dispatch.setLength(enc, m, nil, index: 2)
                Dispatch.setUInt(enc, keyCount, index: 3)
                enc.setBuffer(segStart.mtl, offset: segStart.offset, index: 4)
                enc.setBuffer(segEnd.mtl, offset: segEnd.offset, index: 5)
                Dispatch.dispatch1D(enc, pso, count: m)
            }
            ctx.retainUntilFlush(ord); ctx.retainUntilFlush(keys)
        }
        return GroupSegments(ord: ord, segStart: segStart, segEnd: segEnd, keyCount: keyCount, rows: n)
    }

    // MARK: - Public aggregates

    /// Sum of the non-null Float64 values of each key, through the software binary64 adder on the GPU.
    /// Keys with no valid value are null.
    public func sumDouble(_ values: MetalArray<Double>, segments s: GroupSegments? = nil) throws -> MetalArray<Double> {
        try reduceToDouble(values, op: SegmentedSource.sumDouble(), segments: s)
    }

    /// Mean of the non-null Float64 values of each key (GPU sum, GPU division).
    public func meanDouble(_ values: MetalArray<Double>, segments s: GroupSegments? = nil) throws -> MetalArray<Double> {
        try reduceToDouble(values, op: SegmentedSource.meanDouble(), segments: s)
    }

    /// Sum of the non-null Float32 values of each key, accumulated in Float64 rather than the Float32
    /// accumulation `sumFloat` uses.
    public func sumFloatAsDouble(_ values: MetalArray<Float>, segments s: GroupSegments? = nil) throws -> MetalArray<Double> {
        try reduceToDouble(values, op: SegmentedSource.sumFloatAsDouble(), segments: s)
    }

    /// Mean of the non-null Float32 values of each key, accumulated and divided in Float64.
    public func meanFloat(_ values: MetalArray<Float>, segments s: GroupSegments? = nil) throws -> MetalArray<Double> {
        try reduceToDouble(values, op: SegmentedSource.meanFloat(), segments: s)
    }

    /// Minimum non-null value per key for 64-bit types (Int64, UInt64, Float64); narrower types forward
    /// to the atomic `min`. NaN is skipped, so a key whose only values are NaN is null, as in Arrow.
    public func min64<T: ArrowPrimitive>(_ values: MetalArray<T>, segments s: GroupSegments? = nil) throws -> MetalArray<T> {
        try minMax64(values, isMin: true, segments: s)
    }

    /// Maximum non-null value per key for 64-bit types; see `min64`.
    public func max64<T: ArrowPrimitive>(_ values: MetalArray<T>, segments s: GroupSegments? = nil) throws -> MetalArray<T> {
        try minMax64(values, isMin: false, segments: s)
    }

    // MARK: - Plumbing

    private func minMax64<T: ArrowPrimitive>(_ values: MetalArray<T>, isMin: Bool, segments s: GroupSegments?) throws -> MetalArray<T> {
        guard T.byteWidth == 8 else { return isMin ? try min(values) : try max(values) }
        // Sort-free two-pass atomics unless the caller already paid for the sorted order.
        if s == nil { let e = try extrema(values); return isMin ? e.min : e.max }
        let kind = T.isFloatingPoint ? "double" : (T.minValue < 0 as T ? "signed" : "unsigned")
        let op = SegmentedSource.minMax64(isMin: isMin, kind: kind)
        let (out, valid) = try run(values, op: op, segments: s)
        return try wrap(T.self, out: out, validBytes: valid, ctx: values.context)
    }

    private func reduceToDouble<V: ArrowPrimitive>(_ values: MetalArray<V>, op: SegmentedSource.Op,
                                                   segments s: GroupSegments?) throws -> MetalArray<Double> {
        let (out, valid) = try run(values, op: op, segments: s)
        return try wrap(Double.self, out: out, validBytes: valid, ctx: values.context)
    }

    private func wrap<T: ArrowPrimitive>(_: T.Type, out: MetalArrowBuffer, validBytes: MetalArrowBuffer,
                                         ctx: MetalContext) throws -> MetalArray<T> {
        let bm = try BitmapOps.packBits(ctx, bytes: validBytes, bits: keyCount)
        try ctx.syncPoint()
        let res = MetalArray<T>(length: keyCount, nullCount: 0, validity: bm, values: out, context: ctx)
        res.recomputeNullCount()
        return res
    }

    /// Records the segmented reduction and its finalizer, returning (values, validity bytes).
    private func run<V: ArrowPrimitive>(_ values: MetalArray<V>, op: SegmentedSource.Op,
                                        segments s: GroupSegments?) throws -> (MetalArrowBuffer, MetalArrowBuffer) {
        guard values.length == keys.length else { throw ArrowMetalError.lengthMismatch(keys.length, values.length) }
        guard V.byteWidth == (op.valueType == "float" ? 4 : 8) else {
            throw ArrowMetalError.unsupportedType("segmented \(op.name) does not take \(V.self)")
        }
        let ctx = keys.context
        let seg = try s ?? segments()
        guard seg.keyCount == keyCount, seg.rows == keys.length else {
            throw ArrowMetalError.invalidArrowArray("segments were built for a different group-by")
        }
        let kc = keyCount
        let src = SegmentedSource.source(op)
        let redPSO = try Dispatch.pipeline(ctx, family: "segmented", source: src, function: "seg_reduce", type: op.name)
        let finPSO = try Dispatch.pipeline(ctx, family: "segmented", source: src, function: "seg_finalize", type: op.name)
        let raws = try MetalArrowBuffer.allocate(byteCount: kc * 8, zeroed: false, context: ctx)
        let counts = try MetalArrowBuffer.allocate(byteCount: kc * 4, context: ctx)
        let out = try MetalArrowBuffer.allocate(byteCount: kc * 8, zeroed: false, context: ctx)
        let validBytes = try MetalArrowBuffer.allocate(byteCount: Swift.max(kc, 1), context: ctx)
        let vv = values.validity ?? values.values
        try ctx.run { enc in
            enc.setComputePipelineState(redPSO)
            enc.setBuffer(seg.segStart.mtl, offset: seg.segStart.offset, index: 0)
            enc.setBuffer(seg.segEnd.mtl, offset: seg.segEnd.offset, index: 1)
            let o = seg.ord.length > 0 ? seg.ord.values : seg.segStart
            enc.setBuffer(o.mtl, offset: o.offset, index: 2)
            enc.setBuffer(values.values.mtl, offset: values.values.offset, index: 3)
            enc.setBuffer(vv.mtl, offset: vv.offset, index: 4)
            Dispatch.setLength(enc, kc, nil, index: 5)
            Dispatch.setUInt(enc, values.validity == nil ? 0 : 1, index: 6)
            enc.setBuffer(raws.mtl, offset: raws.offset, index: 7)
            enc.setBuffer(counts.mtl, offset: counts.offset, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: kc, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(finPSO)
            enc.setBuffer(raws.mtl, offset: raws.offset, index: 0)
            enc.setBuffer(counts.mtl, offset: counts.offset, index: 1)
            Dispatch.setLength(enc, kc, nil, index: 2)
            enc.setBuffer(out.mtl, offset: out.offset, index: 3)
            enc.setBuffer(validBytes.mtl, offset: validBytes.offset, index: 4)
            Dispatch.dispatch1D(enc, finPSO, count: kc)
        }
        ctx.retainUntilFlush(seg.ord); ctx.retainUntilFlush(values)
        ctx.retainUntilFlush(raws); ctx.retainUntilFlush(counts)
        return (out, validBytes)
    }
}
