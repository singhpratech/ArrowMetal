import Foundation
import Metal

/// GPU bitmap word operations. Bitmaps are treated as arrays of 32-bit words; buffers are always
/// padded so that reading a full trailing word is in bounds.
enum BitmapOps {
    static func words(bits: Int) -> Int { (bits + 31) / 32 }

    static func binary(_ ctx: MetalContext, _ fn: String, _ a: MetalArrowBuffer, _ b: MetalArrowBuffer, bits: Int,
                       lengthBuffer: MetalArrowBuffer? = nil) throws -> MetalArrowBuffer {
        let w = words(bits: bits)
        let out = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: bits), zeroed: false, context: ctx)
        let pso = try ctx.pipeline(source: KernelSource.bitmap, function: fn, cacheKey: "bitmap/\(fn)")
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(a.mtl, offset: a.offset, index: 0)
            enc.setBuffer(b.mtl, offset: b.offset, index: 1)
            Dispatch.setLength(enc, bits, lengthBuffer, index: 2)
            enc.setBuffer(out.mtl, offset: out.offset, index: 3)
            Dispatch.dispatch1D(enc, pso, count: w)
        }
        return out
    }

    /// Combined validity of two optional bitmaps: AND when both present, the present one otherwise (shared, zero-copy).
    static func combineValidity(_ ctx: MetalContext, _ a: MetalArrowBuffer?, _ b: MetalArrowBuffer?, bits: Int,
                                lengthBuffer: MetalArrowBuffer? = nil) throws -> MetalArrowBuffer? {
        switch (a, b) {
        case (nil, nil): return nil
        case (let x?, nil): return x
        case (nil, let y?): return y
        case (let x?, let y?): return try binary(ctx, "bitmap_and", x, y, bits: bits, lengthBuffer: lengthBuffer)
        }
    }

    static func packBits(_ ctx: MetalContext, bytes: MetalArrowBuffer, bits: Int, lengthBuffer: MetalArrowBuffer? = nil) throws -> MetalArrowBuffer {
        let w = words(bits: bits)
        let out = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: bits), zeroed: false, context: ctx)
        let pso = try ctx.pipeline(source: KernelSource.bitmap, function: "pack_bits", cacheKey: "bitmap/pack_bits")
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(bytes.mtl, offset: bytes.offset, index: 0)
            Dispatch.setLength(enc, bits, lengthBuffer, index: 1)
            enc.setBuffer(out.mtl, offset: out.offset, index: 2)
            Dispatch.dispatch1D(enc, pso, count: w)
        }
        return out
    }

    static func unpackBits(_ ctx: MetalContext, bits: MetalArrowBuffer, count: Int, lengthBuffer: MetalArrowBuffer? = nil) throws -> MetalArrowBuffer {
        let out = try MetalArrowBuffer.allocate(byteCount: count, zeroed: false, context: ctx)
        guard count > 0 else { return out }
        let pso = try ctx.pipeline(source: KernelSource.bitmap, function: "unpack_bits", cacheKey: "bitmap/unpack_bits")
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(bits.mtl, offset: bits.offset, index: 0)
            Dispatch.setLength(enc, count, lengthBuffer, index: 1)
            enc.setBuffer(out.mtl, offset: out.offset, index: 2)
            Dispatch.dispatch1D(enc, pso, count: count)
        }
        return out
    }
}
