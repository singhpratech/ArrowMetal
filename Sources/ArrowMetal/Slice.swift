import Foundation

extension MetalArrowBuffer {
    /// A view of `[byteOffset, byteOffset + byteCount)` sharing the same `MTLBuffer`.
    public func view(byteOffset: Int, byteCount: Int) -> MetalArrowBuffer {
        MetalArrowBuffer(mtl: mtl, byteCount: byteCount, offset: offset + byteOffset, keepAlive: self)
    }
}

extension MetalArray {
    /// Arrow `slice`. Zero-copy when `offset` is a multiple of 32 (keeps bitmap words and values aligned for
    /// the kernels); otherwise the slice is materialised with one copy.
    public func slice(offset: Int, length newLength: Int) throws -> MetalArray<T> {
        precondition(offset >= 0 && newLength >= 0 && offset + newLength <= length, "slice out of range")
        if offset % 32 == 0 {
            let v = values.view(byteOffset: offset * T.byteWidth, byteCount: newLength * T.byteWidth)
            let bm = validity?.view(byteOffset: offset / 8, byteCount: Bitmap.byteCount(bits: newLength))
            let res = MetalArray<T>(length: newLength, nullCount: 0, validity: bm, values: v, context: context)
            res.recomputeNullCount()
            return res
        }
        let out = try MetalArray<T>.allocate(length: newLength, withValidity: validity != nil, context: context)
        withExtendedLifetime(self) {
            let src = valuePointer, dst = out.mutableValuePointer
            for i in 0..<newLength { dst[i] = src[offset + i] }
            if let ov = out.validity, let iv = validity {
                let sp = iv.typed(UInt8.self), dp = ov.mutableTyped(UInt8.self)
                for i in 0..<newLength where Bitmap.isSet(sp, offset + i) { Bitmap.set(dp, i) }
            }
        }
        out.recomputeNullCount()
        return out
    }
}

extension MetalBooleanArray {
    public func slice(offset: Int, length newLength: Int) throws -> MetalBooleanArray {
        precondition(offset >= 0 && newLength >= 0 && offset + newLength <= length, "slice out of range")
        if offset % 32 == 0 {
            let v = values.view(byteOffset: offset / 8, byteCount: Bitmap.byteCount(bits: newLength))
            let bm = validity?.view(byteOffset: offset / 8, byteCount: Bitmap.byteCount(bits: newLength))
            let res = MetalBooleanArray(length: newLength, nullCount: 0, validity: bm, values: v, context: context)
            res.recomputeNullCount()
            return res
        }
        let out = try MetalBooleanArray.allocate(length: newLength, withValidity: validity != nil, context: context)
        withExtendedLifetime(self) {
            let sp = values.typed(UInt8.self), dp = out.values.mutableTyped(UInt8.self)
            for i in 0..<newLength where Bitmap.isSet(sp, offset + i) { Bitmap.set(dp, i) }
            if let ov = out.validity, let iv = validity {
                let s = iv.typed(UInt8.self), d = ov.mutableTyped(UInt8.self)
                for i in 0..<newLength where Bitmap.isSet(s, offset + i) { Bitmap.set(d, i) }
            }
        }
        out.recomputeNullCount()
        return out
    }

    /// Number of true (and valid) values.
    public var count: Int { trueCount }
    /// True if any valid value is true.
    public var any: Bool { trueCount > 0 }
    /// True if every valid value is true (true for an all-null or empty array, matching Arrow's `all` with skip_nulls).
    public var all: Bool { trueCount == length - nullCount }
}
