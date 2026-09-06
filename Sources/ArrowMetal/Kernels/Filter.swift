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
        guard Dispatch.runsOnGPU(T.self) else { return try CPUReference.filter(self, mask) }
        try Dispatch.checkLength(length)
        let ctx = context
        let sel = try mask.selectionBitmap()
        let words = BitmapOps.words(bits: length)
        let blocks = Swift.max(1, (words + Dispatch.threadgroupSize - 1) / Dispatch.threadgroupSize)
        let blockCounts = try MetalArrowBuffer.allocate(byteCount: blocks * 4, context: ctx)
        let src = KernelSource.filter(T: T.mslType)
        let countPSO = try Dispatch.pipeline(ctx, family: "filter", source: src, function: "filter_count", type: T.mslType)
        let scatterPSO = try Dispatch.pipeline(ctx, family: "filter", source: src, function: "filter_scatter", type: T.mslType)
        let tg = MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1)
        let grid = MTLSize(width: blocks, height: 1, depth: 1)

        try ctx.run { enc in
            enc.setComputePipelineState(countPSO)
            enc.setBuffer(sel.mtl, offset: sel.offset, index: 0)
            Dispatch.setUInt(enc, length, index: 1)
            enc.setBuffer(blockCounts.mtl, offset: 0, index: 2)
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        }

        // Exclusive scan of block counts on the CPU.
        let outLen: Int = withExtendedLifetime(blockCounts) {
            let bc = blockCounts.mutableTyped(UInt32.self)
            var total: UInt32 = 0
            for b in 0..<blocks { let c = bc[b]; bc[b] = total; total &+= c }
            return Int(total)
        }

        let outValues = try MetalArrowBuffer.allocate(byteCount: outLen * T.byteWidth, context: ctx)
        let hasValidity = validity != nil
        let validBytes = hasValidity ? try MetalArrowBuffer.allocate(byteCount: Swift.max(outLen, 1), context: ctx) : nil

        try ctx.run { enc in
            enc.setComputePipelineState(scatterPSO)
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            if let v = validity { enc.setBuffer(v.mtl, offset: v.offset, index: 1) } else { enc.setBuffer(values.mtl, offset: 0, index: 1) }
            enc.setBuffer(sel.mtl, offset: sel.offset, index: 2)
            Dispatch.setUInt(enc, length, index: 3)
            Dispatch.setUInt(enc, hasValidity ? 1 : 0, index: 4)
            enc.setBuffer(blockCounts.mtl, offset: 0, index: 5)
            enc.setBuffer(outValues.mtl, offset: 0, index: 6)
            enc.setBuffer((validBytes ?? outValues).mtl, offset: 0, index: 7)
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        }

        var outValidity: MetalArrowBuffer? = nil
        if let vb = validBytes, outLen > 0 {
            outValidity = try BitmapOps.packBits(ctx, bytes: vb, bits: outLen)
        }
        let res = MetalArray<T>(length: outLen, nullCount: 0, validity: outValidity, values: outValues, context: ctx)
        res.recomputeNullCount()
        return res
    }
}
