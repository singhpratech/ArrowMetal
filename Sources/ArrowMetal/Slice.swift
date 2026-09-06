import Foundation

extension MetalArrowBuffer {
    /// A view of `[byteOffset, byteOffset + byteCount)` sharing the same `MTLBuffer`.
    public func view(byteOffset: Int, byteCount: Int) -> MetalArrowBuffer {
        MetalArrowBuffer(mtl: mtl, byteCount: byteCount, offset: offset + byteOffset, keepAlive: self)
    }
}

extension MetalArray {
    /// Arrow `slice`, zero-copy at every offset and O(1) in the length.
    ///
    /// When the resulting offset is a multiple of 32 the slice is a pair of `MetalArrowBuffer` views (bitmap
    /// words stay 4-byte aligned and the values stay aligned for the kernels' 4-wide vector loads), so the
    /// slice has no offset of its own and every kernel runs on it unchanged. At any other offset it carries
    /// the parent's buffers plus Arrow's element `offset`: host reads, the null count and export apply the
    /// offset directly, and only a kernel dispatch normalises it (once, cached — see `MetalArray`).
    ///
    /// Slicing a slice adds the offsets, so a chain never costs more than a single slice.
    public func slice(offset sliceOffset: Int, length newLength: Int) throws -> MetalArray<T> {
        precondition(sliceOffset >= 0 && newLength >= 0 && sliceOffset + newLength <= length, "slice out of range")
        // The init folds a 32-aligned offset into buffer views and carries anything else; with no bitmap
        // there are no nulls, and with one the count is deferred to whoever asks for it.
        return MetalArray<T>(offset: offset + sliceOffset, length: newLength,
                             validity: rawValidity, values: rawValues, context: context)
    }
}

extension MetalBooleanArray {
    /// Arrow `slice` on a boolean column: the same zero-copy rules as `MetalArray.slice`, over two bitmaps.
    public func slice(offset sliceOffset: Int, length newLength: Int) throws -> MetalBooleanArray {
        precondition(sliceOffset >= 0 && newLength >= 0 && sliceOffset + newLength <= length, "slice out of range")
        return MetalBooleanArray(offset: offset + sliceOffset, length: newLength,
                                 validity: rawValidity, values: rawValues, context: context)
    }

    /// Number of true (and valid) values.
    public var count: Int { trueCount }
    /// True if any valid value is true.
    public var any: Bool { (try? anyTrue()) ?? (trueCount > 0) }
    /// True if every valid value is true (true for an all-null or empty array, matching Arrow's `all` with skip_nulls).
    public var all: Bool { (try? allTrue()) ?? (trueCount == length - nullCount) }
}

extension MetalStringArray {
    /// Arrow `slice` on a utf8 / binary column, without touching the character data.
    ///
    /// The offsets are a buffer view (Arrow offsets are absolute, so the shared data buffer stays valid as
    /// is) and the validity bitmap is a view too whenever the offset is byte aligned; otherwise the bitmap
    /// alone — one bit per row, never the data — is shifted into a fresh buffer. Nothing else is copied.
    public func slice(offset: Int, length newLength: Int) throws -> MetalStringArray {
        precondition(offset >= 0 && newLength >= 0 && offset + newLength <= length, "slice out of range")
        let off = offsets.view(byteOffset: offset * 4, byteCount: (newLength + 1) * 4)
        var bm: MetalArrowBuffer? = nil
        if let v = validity {
            if offset % 8 == 0 {
                bm = v.view(byteOffset: offset / 8, byteCount: Bitmap.byteCount(bits: newLength))
            } else {
                let b = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: newLength), 1), context: context)
                withExtendedLifetime(v) { Bitmap.copyBits(v.typed(UInt8.self), from: offset, into: b.mutableTyped(UInt8.self), bits: newLength) }
                bm = b
            }
        }
        let res = MetalStringArray(length: newLength, nullCount: 0, validity: bm, offsets: off, data: data, context: context)
        res.isBinary = isBinary
        res.recomputeNullCount()
        return res
    }
}
