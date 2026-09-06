import Foundation
import Metal

extension MetalBooleanArray {
    /// Selection bitmap for Arrow `filter` semantics with null selection behaviour "drop":
    /// an element is selected when the mask is true AND the mask is valid.
    func selectionBitmap() throws -> MetalArrowBuffer {
        guard let v = validity else { return values }
        return try BitmapOps.binary(context, "bitmap_and", values, v, bits: length)
    }
}

extension MetalArray {
    /// Arrow `filter`: returns the elements where `mask` is true. Null mask entries drop the element.
    ///
    /// Two GPU passes: per-block popcount, then a scatter using an in-threadgroup prefix scan;
    /// the block offsets are scanned on the CPU (one entry per 8192 elements).
    public func filter(_ mask: MetalBooleanArray) throws -> MetalArray<T> {
        guard mask.length == length else { throw ArrowMetalError.lengthMismatch(length, mask.length) }
        let sel = try mask.selectionBitmap()
        return try compact(selection: sel, prepare: nil)
    }

    /// Fused `filter(where: self OP scalar)`: the predicate is evaluated inside the counting pass, so no
    /// boolean array is materialised. Equivalent to `filter(compare(op, scalar))`.
    public func filter(where op: CompareOp, _ scalar: T) throws -> MetalArray<T> {
        if T.self == Double.self { return try filter(try compare(op, scalar)) }
        let ctx = context
        let sel = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: length), zeroed: false, context: ctx)
        let opIndex = UInt32(CompareOp.allCases.firstIndex(of: op)!)
        let vals = values, vld = validity ?? values, hasV = validity != nil, n = length
        return try compact(selection: sel) { enc, pso, blockCounts, grid, tg in
            enc.setComputePipelineState(pso)
            enc.setBuffer(vals.mtl, offset: vals.offset, index: 0)
            enc.setBuffer(vld.mtl, offset: vld.offset, index: 1)
            Dispatch.setUInt(enc, n, index: 2)
            Dispatch.setUInt(enc, hasV ? 1 : 0, index: 3)
            Dispatch.setUInt(enc, Int(opIndex), index: 4)
            Dispatch.setScalar(enc, scalar, index: 5)
            enc.setBuffer(sel.mtl, offset: sel.offset, index: 6)
            enc.setBuffer(blockCounts.mtl, offset: 0, index: 7)
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        }
    }

    /// Stream compaction driven by a selection bitmap, entirely on the GPU in one command buffer:
    /// count per block (or fused predicate + count), scan block counts, scatter, pack validity.
    /// The output length is only known after the GPU finishes, so the output buffer is sized for the
    /// worst case (input length) and trimmed to the real length afterwards.
    func compact(selection sel: MetalArrowBuffer,
                 prepare: ((MTLComputeCommandEncoder, MTLComputePipelineState, MetalArrowBuffer, MTLSize, MTLSize) throws -> Void)?) throws -> MetalArray<T> {
        try Dispatch.checkLength(length)
        let ctx = context
        let words = BitmapOps.words(bits: length)
        let blocks = Swift.max(1, (words + Dispatch.threadgroupSize - 1) / Dispatch.threadgroupSize)
        let blockCounts = try MetalArrowBuffer.allocate(byteCount: blocks * 4, zeroed: false, context: ctx)
        let total = try MetalArrowBuffer.allocate(byteCount: 4, zeroed: false, context: ctx)
        let mslT = Dispatch.moveType(T.self)
        let src = KernelSource.filter(T: mslT)
        let countPSO = try Dispatch.pipeline(ctx, family: "filter", source: src, function: prepare == nil ? "filter_count" : "filter_pred_count", type: mslT)
        let scanPSO = try Dispatch.pipeline(ctx, family: "filter", source: src, function: "filter_scan", type: mslT)
        let scatterPSO = try Dispatch.pipeline(ctx, family: "filter", source: src, function: "filter_scatter", type: mslT)
        let tg = MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1)
        let grid = MTLSize(width: blocks, height: 1, depth: 1)
        // Worst-case sized outputs; they come from the pool so this is cheap.
        let outValues = try MetalArrowBuffer.allocate(byteCount: length * T.byteWidth, zeroed: false, context: ctx)
        let hasValidity = validity != nil
        let validBytes = hasValidity ? try MetalArrowBuffer.allocate(byteCount: Swift.max(length, 1), zeroed: false, context: ctx) : nil

        try ctx.run { enc in
            if let prepare {
                try prepare(enc, countPSO, blockCounts, grid, tg)
            } else {
                enc.setComputePipelineState(countPSO)
                enc.setBuffer(sel.mtl, offset: sel.offset, index: 0)
                Dispatch.setUInt(enc, length, index: 1)
                enc.setBuffer(blockCounts.mtl, offset: 0, index: 2)
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
            }
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(scanPSO)
            enc.setBuffer(blockCounts.mtl, offset: 0, index: 0)
            Dispatch.setUInt(enc, blocks, index: 1)
            enc.setBuffer(total.mtl, offset: 0, index: 2)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(scatterPSO)
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            enc.setBuffer((validity ?? values).mtl, offset: (validity ?? values).offset, index: 1)
            enc.setBuffer(sel.mtl, offset: sel.offset, index: 2)
            Dispatch.setUInt(enc, length, index: 3)
            Dispatch.setUInt(enc, hasValidity ? 1 : 0, index: 4)
            enc.setBuffer(blockCounts.mtl, offset: 0, index: 5)
            enc.setBuffer(outValues.mtl, offset: 0, index: 6)
            enc.setBuffer((validBytes ?? outValues).mtl, offset: 0, index: 7)
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        }
        let outLen = withExtendedLifetime(total) { Int(total.typed(UInt32.self)[0]) }
        var outValidity: MetalArrowBuffer? = nil
        if let vb = validBytes, outLen > 0 {
            outValidity = try BitmapOps.packBits(ctx, bytes: vb, bits: outLen)
        }
        let trimmed = outValues.view(byteOffset: 0, byteCount: outLen * T.byteWidth)
        let res = MetalArray<T>(length: outLen, nullCount: 0, validity: outValidity, values: trimmed, context: ctx)
        res.recomputeNullCount()
        return res
    }
}

extension MetalBooleanArray {
    /// Arrow `filter` on a boolean array: unpack to bytes, compact, repack.
    public func filter(_ mask: MetalBooleanArray) throws -> MetalBooleanArray {
        let bytes = try toUInt8Array()
        let kept = try bytes.filter(mask)
        return try MetalBooleanArray.fromUInt8Array(kept)
    }

    /// One byte (0/1) per element with the same validity (shared, zero-copy).
    public func toUInt8Array() throws -> MetalArray<UInt8> {
        let out = try BitmapOps.unpackBits(context, bits: values, count: length)
        return MetalArray<UInt8>(length: length, nullCount: nullCount, validity: validity, values: out, context: context)
    }

    /// Inverse of `toUInt8Array`: non-zero bytes become true.
    public static func fromUInt8Array(_ a: MetalArray<UInt8>) throws -> MetalBooleanArray {
        let packed = a.length == 0 ? try MetalArrowBuffer.allocate(byteCount: 0, context: a.context)
                                   : try BitmapOps.packBits(a.context, bytes: a.values, bits: a.length)
        return MetalBooleanArray(length: a.length, nullCount: a.nullCount, validity: a.validity, values: packed, context: a.context)
    }
}
