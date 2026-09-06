import Foundation
import Metal

/// The three Arrow logical functions the bitmap kernels in `BitmapOps.swift` and the Kleene pair in
/// `Structural.swift` do not cover: `xor`, `and_not` and `and_not_kleene`.
///
/// All three are word-wise over packed Arrow boolean bitmaps, one thread per 32-bit output word, so
/// they cost one pass over `length / 8` bytes.
///
/// **Null handling.** `xor` and `and_not` propagate nulls (the output validity is the AND of the
/// inputs'), matching Arrow's plain `and` / `or`. `and_not_kleene` is three-valued and is exactly
/// `and_kleene(a, not(b))` — which cannot be written that way here, because Kleene `not` of a null
/// is null and `andKleene` would then need `b`'s value bits, so it gets its own kernel:
///
///     value = a & ~b
///     valid = (a.valid & ~a) | (b.valid & b) | (a.valid & b.valid)
///
/// that is, the result is known when `a` is a valid false, or `b` is a valid true (either forces
/// false), or both sides are known. The value word reads the value bits of null slots, which Arrow
/// leaves undefined; that is safe because every term that can select such a bit is masked by the
/// other side's validity, exactly as in `and_kleene`.
public enum LogicalExtraOp: String, CaseIterable, Sendable {
    case xor
    case andNot = "and_not"
    case andNotKleene = "and_not_kleene"
}

/// MSL for the two kernels the existing bitmap family does not already provide. (`and_not` reuses
/// `bitmap_and_not` from `KernelSource.bitmap`.)
enum LogicalExtraSource {
    static let source = KernelSource.prelude + """

    kernel void lx_xor(device const uint* a [[buffer(0)]], device const uint* b [[buffer(1)]],
                       device const uint* nPtr [[buffer(2)]], device uint* out [[buffer(3)]],
                       uint w [[thread_position_in_grid]]) {
        if (w < (*nPtr + 31u) / 32u) out[w] = a[w] ^ b[w];
    }
    // flags: bit0 = a has a validity bitmap, bit1 = b has one.
    kernel void lx_and_not_kleene(device const uint* aVal [[buffer(0)]], device const uint* aValid [[buffer(1)]],
                                  device const uint* bVal [[buffer(2)]], device const uint* bValid [[buffer(3)]],
                                  device const uint* nPtr [[buffer(4)]], constant uint& flags [[buffer(5)]],
                                  device uint* outVal [[buffer(6)]], device uint* outValid [[buffer(7)]],
                                  uint w [[thread_position_in_grid]]) {
        if (w >= (*nPtr + 31u) / 32u) return;
        uint av = aVal[w], bv = bVal[w];
        uint ava = (flags & 1u) ? aValid[w] : 0xFFFFFFFFu;
        uint bva = (flags & 2u) ? bValid[w] : 0xFFFFFFFFu;
        outVal[w] = av & ~bv;
        outValid[w] = (ava & ~av) | (bva & bv) | (ava & bva);
    }
    """
}

extension MetalBooleanArray {
    /// Arrow `xor`: exclusive or, nulls propagating.
    public func xor(_ other: MetalBooleanArray) throws -> MetalBooleanArray {
        try checkSameLength(other)
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: n), zeroed: false, context: ctx)
        let words = BitmapOps.words(bits: n)
        if words > 0 {
            let pso = try ctx.pipeline(source: LogicalExtraSource.source, function: "lx_xor", cacheKey: "logicalx/lx_xor")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                enc.setBuffer(other.values.mtl, offset: other.values.offset, index: 1)
                Dispatch.setLength(enc, n, lengthBuffer, index: 2)
                enc.setBuffer(out.mtl, offset: out.offset, index: 3)
                Dispatch.dispatch1D(enc, pso, count: words)
            }
            ctx.retainUntilFlush(self); ctx.retainUntilFlush(other)
        }
        let v = try BitmapOps.combineValidity(ctx, validity, other.validity, bits: n, lengthBuffer: lengthBuffer)
        let res = inheritPending(MetalBooleanArray(length: knownLength, nullCount: 0, validity: v, values: out, context: ctx))
        res.recomputeNullCount()
        return res
    }

    /// Arrow `and_not`: `a AND NOT b`, nulls propagating.
    public func andNot(_ other: MetalBooleanArray) throws -> MetalBooleanArray {
        try checkSameLength(other)
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let ctx = context
        let out = try BitmapOps.binary(ctx, "bitmap_and_not", values, other.values, bits: n, lengthBuffer: lengthBuffer)
        let v = try BitmapOps.combineValidity(ctx, validity, other.validity, bits: n, lengthBuffer: lengthBuffer)
        let res = inheritPending(MetalBooleanArray(length: knownLength, nullCount: 0, validity: v, values: out, context: ctx))
        res.recomputeNullCount()
        return res
    }

    /// Arrow `and_not_kleene`: three-valued `a AND NOT b`. A valid `false` on the left or a valid
    /// `true` on the right makes the answer `false` even when the other side is null.
    public func andNotKleene(_ other: MetalBooleanArray) throws -> MetalBooleanArray {
        try checkSameLength(other)
        // With no nulls anywhere, Kleene logic is ordinary logic.
        if validity == nil && other.validity == nil { return try andNot(other) }
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let ctx = context
        let words = BitmapOps.words(bits: n)
        let outValues = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: n), zeroed: false, context: ctx)
        let outValidity = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: n), zeroed: false, context: ctx)
        if words > 0 {
            let pso = try ctx.pipeline(source: LogicalExtraSource.source, function: "lx_and_not_kleene",
                                       cacheKey: "logicalx/lx_and_not_kleene")
            let av = validity ?? values, bv = other.validity ?? other.values
            let flags = (validity == nil ? 0 : 1) | (other.validity == nil ? 0 : 2)
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                enc.setBuffer(av.mtl, offset: av.offset, index: 1)
                enc.setBuffer(other.values.mtl, offset: other.values.offset, index: 2)
                enc.setBuffer(bv.mtl, offset: bv.offset, index: 3)
                Dispatch.setLength(enc, n, lengthBuffer, index: 4)
                Dispatch.setUInt(enc, flags, index: 5)
                enc.setBuffer(outValues.mtl, offset: outValues.offset, index: 6)
                enc.setBuffer(outValidity.mtl, offset: outValidity.offset, index: 7)
                Dispatch.dispatch1D(enc, pso, count: words)
            }
            ctx.retainUntilFlush(self); ctx.retainUntilFlush(other)
        }
        let res = inheritPending(MetalBooleanArray(length: knownLength, nullCount: 0, validity: outValidity, values: outValues, context: ctx))
        res.recomputeNullCount()
        return res
    }

    /// One of the three by name, for the C ABI's op table.
    public func logicalExtra(_ op: LogicalExtraOp, _ other: MetalBooleanArray) throws -> MetalBooleanArray {
        switch op {
        case .xor: return try xor(other)
        case .andNot: return try andNot(other)
        case .andNotKleene: return try andNotKleene(other)
        }
    }
}
