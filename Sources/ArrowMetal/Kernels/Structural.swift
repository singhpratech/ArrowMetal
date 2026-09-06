import Foundation
import Metal

/// Arrow's structural, conditional and set-lookup compute functions on the GPU:
/// `is_null`, `is_valid`, `fill_null`, `drop_null`, `if_else`, `coalesce`, `is_in`, `index_in`,
/// and the three-valued (Kleene) `and_kleene` / `or_kleene`.
///
/// Everything here works for every primitive element type. Float64 has no Metal `double`, but none
/// of these kernels does arithmetic on a value — they only move it, or order it — so Float64 runs on
/// the GPU as a raw 64-bit move, and set lookup orders it through the same order-preserving bit-pattern
/// key the radix sort uses.
enum Structural {
    static func pipeline(_ ctx: MetalContext, _ source: String, _ function: String, type: String) throws -> MTLComputePipelineState {
        try Dispatch.pipeline(ctx, family: "structural", source: source, function: function, type: type)
    }

    /// A bitmap with every bit in `[0, bits)` set (used as `is_valid` of an array that has no bitmap).
    static func filledBitmap(_ ctx: MetalContext, bits: Int, lengthBuffer: MetalArrowBuffer?) throws -> MetalArrowBuffer {
        let out = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: bits), context: ctx)
        let words = BitmapOps.words(bits: bits)
        guard words > 0 else { return out }
        let pso = try pipeline(ctx, StructuralSource.common, "st_fill_words", type: "common")
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            Dispatch.setLength(enc, bits, lengthBuffer, index: 0)
            Dispatch.setUInt(enc, Int(UInt32.max), index: 1)
            enc.setBuffer(out.mtl, offset: out.offset, index: 2)
            Dispatch.dispatch1D(enc, pso, count: words)
        }
        return out
    }

    /// Word-wise NOT of a bitmap (used as `is_null` of a validity bitmap).
    static func invertedBitmap(_ ctx: MetalContext, _ a: MetalArrowBuffer, bits: Int, lengthBuffer: MetalArrowBuffer?) throws -> MetalArrowBuffer {
        let out = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: bits), zeroed: false, context: ctx)
        let words = BitmapOps.words(bits: bits)
        guard words > 0 else { return out }
        let pso = try ctx.pipeline(source: KernelSource.bitmap, function: "bitmap_not", cacheKey: "bitmap/bitmap_not")
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(a.mtl, offset: a.offset, index: 0)
            Dispatch.setLength(enc, bits, lengthBuffer, index: 2)
            enc.setBuffer(out.mtl, offset: out.offset, index: 3)
            Dispatch.dispatch1D(enc, pso, count: words)
        }
        return out
    }

    /// The values buffer of `is_null` (or, with `wantNull` false, of `is_valid`) for a validity bitmap.
    static func nullMask(_ ctx: MetalContext, validity: MetalArrowBuffer?, bits: Int,
                         lengthBuffer: MetalArrowBuffer?, wantNull: Bool) throws -> MetalArrowBuffer {
        guard let v = validity else {
            return wantNull ? try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: bits), context: ctx)
                            : try filledBitmap(ctx, bits: bits, lengthBuffer: lengthBuffer)
        }
        // is_valid is the validity bitmap itself, shared zero-copy.
        return wantNull ? try invertedBitmap(ctx, v, bits: bits, lengthBuffer: lengthBuffer) : v
    }

    /// Binds a scalar of the move type: Float64 goes in as its raw bit pattern.
    static func setMoveScalar<T: ArrowPrimitive>(_ enc: MTLComputeCommandEncoder, _ v: T, index: Int) {
        if let d = v as? Double { Dispatch.setScalar(enc, Int64(bitPattern: d.bitPattern), index: index) }
        else { Dispatch.setScalar(enc, v, index: index) }
    }

    /// Element type and key mapping the set-lookup kernels read a column with.
    ///
    /// `cacheKey` is the Arrow element type, not the MSL one: Float32 is read as `int` and Float64 as
    /// `long`, so keying the compiled pipeline on the MSL type alone would hand Int32 the float kernel.
    static func lookupKey<T: ArrowPrimitive>(_: T.Type) -> (K: String, toKey: String, cacheKey: String) {
        if T.self == Double.self { return ("long", "d_key_n($0)", "f64") }
        if T.self == Float.self { return ("int", "f_key($0)", "f32") }
        return (T.mslType, "$0", T.mslType)
    }
}

// MARK: - is_null / is_valid

extension MetalArray {
    /// Arrow `is_null`: true where the element is null. The result never contains nulls.
    public func isNull() throws -> MetalBooleanArray {
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let out = try Structural.nullMask(context, validity: validity, bits: n, lengthBuffer: lengthBuffer, wantNull: true)
        return inheritPending(MetalBooleanArray(length: knownLength, nullCount: 0, validity: nil, values: out, context: context))
    }

    /// Arrow `is_valid`: true where the element is not null. The result never contains nulls.
    /// When the array has a validity bitmap the result shares it zero-copy.
    public func isValid() throws -> MetalBooleanArray {
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let out = try Structural.nullMask(context, validity: validity, bits: n, lengthBuffer: lengthBuffer, wantNull: false)
        return inheritPending(MetalBooleanArray(length: knownLength, nullCount: 0, validity: nil, values: out, context: context))
    }
}

extension MetalBooleanArray {
    /// Arrow `is_null` on a boolean column.
    public func isNull() throws -> MetalBooleanArray {
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let out = try Structural.nullMask(context, validity: validity, bits: n, lengthBuffer: lengthBuffer, wantNull: true)
        return inheritPending(MetalBooleanArray(length: knownLength, nullCount: 0, validity: nil, values: out, context: context))
    }

    /// Arrow `is_valid` on a boolean column.
    public func isValid() throws -> MetalBooleanArray {
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let out = try Structural.nullMask(context, validity: validity, bits: n, lengthBuffer: lengthBuffer, wantNull: false)
        return inheritPending(MetalBooleanArray(length: knownLength, nullCount: 0, validity: nil, values: out, context: context))
    }
}

// MARK: - fill_null / drop_null

extension MetalArray {
    /// Arrow `fill_null`: every null element becomes `value`; the result has no nulls.
    ///
    /// Named `fillingNull` because `MetalArray` already carries an internal host-side `fillNull(_:)`
    /// used by the string gather path; this one is the public GPU function.
    public func fillingNull(_ value: T) throws -> MetalArray<T> {
        guard validity != nil else { return self }
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * T.byteWidth, zeroed: false, context: ctx)
        if n > 0 {
            let mslT = Dispatch.moveType(T.self)
            let pso = try Structural.pipeline(ctx, StructuralSource.moves(T: mslT), "st_fill_null", type: mslT)
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                enc.setBuffer(validity!.mtl, offset: validity!.offset, index: 1)
                Dispatch.setLength(enc, n, lengthBuffer, index: 2)
                Structural.setMoveScalar(enc, value, index: 3)
                Dispatch.setUInt(enc, 1, index: 4)
                enc.setBuffer(out.mtl, offset: out.offset, index: 5)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
            ctx.retainUntilFlush(self)
        }
        return inheritPending(MetalArray<T>(length: knownLength, nullCount: 0, validity: nil, values: out, context: ctx))
    }

    /// Arrow `drop_null`: the non-null elements, in order. The result has no nulls.
    public func dropNull() throws -> MetalArray<T> {
        guard validity != nil else { return self }
        return try filter(try isValid())
    }
}

extension MetalBooleanArray {
    /// Arrow `fill_null` on a boolean column.
    public func fillingNull(_ value: Bool) throws -> MetalBooleanArray {
        guard validity != nil else { return self }
        return try MetalBooleanArray.fromUInt8Array(try toUInt8Array().fillingNull(value ? 1 : 0))
    }

    /// Arrow `drop_null` on a boolean column.
    public func dropNull() throws -> MetalBooleanArray {
        guard validity != nil else { return self }
        return try filter(try isValid())
    }
}

// MARK: - if_else

extension MetalArray {
    /// Arrow `if_else`: `cond ? left : right`, element-wise. A null condition yields a null output;
    /// otherwise the chosen side's value (and its null-ness) is copied through.
    public static func ifElse(_ cond: MetalBooleanArray, _ left: MetalArray<T>, _ right: MetalArray<T>) throws -> MetalArray<T> {
        try cond.checkSameLength(left)
        try cond.checkSameLength(right)
        return try ifElseImpl(cond, left: left, leftScalar: nil, right: right, rightScalar: nil)
    }

    /// `cond ? left : rightScalar`.
    public static func ifElse(_ cond: MetalBooleanArray, _ left: MetalArray<T>, _ right: T) throws -> MetalArray<T> {
        try cond.checkSameLength(left)
        return try ifElseImpl(cond, left: left, leftScalar: nil, right: nil, rightScalar: right)
    }

    /// `cond ? leftScalar : right`.
    public static func ifElse(_ cond: MetalBooleanArray, _ left: T, _ right: MetalArray<T>) throws -> MetalArray<T> {
        try cond.checkSameLength(right)
        return try ifElseImpl(cond, left: nil, leftScalar: left, right: right, rightScalar: nil)
    }

    /// `cond ? leftScalar : rightScalar`.
    public static func ifElse(_ cond: MetalBooleanArray, _ left: T, _ right: T) throws -> MetalArray<T> {
        try ifElseImpl(cond, left: nil, leftScalar: left, right: nil, rightScalar: right)
    }

    private static func ifElseImpl(_ cond: MetalBooleanArray,
                                   left: MetalArray<T>?, leftScalar: T?,
                                   right: MetalArray<T>?, rightScalar: T?) throws -> MetalArray<T> {
        let ctx = cond.context
        let n = cond.dispatchLength
        try Dispatch.checkLength(n)
        let outValues = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * T.byteWidth, zeroed: false, context: ctx)
        let validBytes = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1), zeroed: false, context: ctx)
        if n > 0 {
            let mslT = Dispatch.moveType(T.self)
            let pso = try Structural.pipeline(ctx, StructuralSource.moves(T: mslT), "st_if_else", type: mslT)
            var flags = 0
            if cond.validity != nil { flags |= 1 }
            if left?.validity != nil { flags |= 2 }
            if right?.validity != nil { flags |= 4 }
            if left == nil { flags |= 8 }
            if right == nil { flags |= 16 }
            let leftValues = left?.values ?? outValues
            let leftValidity = left?.validity ?? leftValues
            let rightValues = right?.values ?? outValues
            let rightValidity = right?.validity ?? rightValues
            let condValidity = cond.validity ?? cond.values
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(cond.values.mtl, offset: cond.values.offset, index: 0)
                enc.setBuffer(condValidity.mtl, offset: condValidity.offset, index: 1)
                enc.setBuffer(leftValues.mtl, offset: leftValues.offset, index: 2)
                enc.setBuffer(leftValidity.mtl, offset: leftValidity.offset, index: 3)
                enc.setBuffer(rightValues.mtl, offset: rightValues.offset, index: 4)
                enc.setBuffer(rightValidity.mtl, offset: rightValidity.offset, index: 5)
                Structural.setMoveScalar(enc, leftScalar ?? 0, index: 6)
                Structural.setMoveScalar(enc, rightScalar ?? 0, index: 7)
                Dispatch.setLength(enc, n, cond.lengthBuffer, index: 8)
                Dispatch.setUInt(enc, flags, index: 9)
                enc.setBuffer(outValues.mtl, offset: outValues.offset, index: 10)
                enc.setBuffer(validBytes.mtl, offset: validBytes.offset, index: 11)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
            ctx.retainUntilFlush(cond)
            if let l = left { ctx.retainUntilFlush(l) }
            if let r = right { ctx.retainUntilFlush(r) }
            ctx.retainUntilFlush(validBytes)
        }
        // Nothing can be null when the condition and both branches are null-free.
        let canBeNull = cond.validity != nil || left?.validity != nil || right?.validity != nil
        guard canBeNull else {
            return cond.inheritPending(MetalArray<T>(length: cond.knownLength, nullCount: 0, validity: nil, values: outValues, context: ctx))
        }
        let outValidity = n > 0 ? try BitmapOps.packBits(ctx, bytes: validBytes, bits: n, lengthBuffer: cond.lengthBuffer) : nil
        let res = cond.inheritPending(MetalArray<T>(length: cond.knownLength, nullCount: 0, validity: outValidity, values: outValues, context: ctx))
        res.recomputeNullCount()
        return res
    }
}

extension MetalBooleanArray {
    /// Arrow `if_else` with this array as the condition.
    public func ifElse<T: ArrowPrimitive>(_ left: MetalArray<T>, _ right: MetalArray<T>) throws -> MetalArray<T> {
        try MetalArray<T>.ifElse(self, left, right)
    }

    /// Arrow `if_else` over boolean values: unpack to bytes, choose, repack.
    public static func ifElse(_ cond: MetalBooleanArray, _ left: MetalBooleanArray, _ right: MetalBooleanArray) throws -> MetalBooleanArray {
        try cond.checkSameLength(left)
        try cond.checkSameLength(right)
        let l = try left.toUInt8Array(), r = try right.toUInt8Array()
        return try MetalBooleanArray.fromUInt8Array(try MetalArray<UInt8>.ifElse(cond, l, r))
    }

    /// Arrow `if_else` over boolean values with two scalar branches.
    public static func ifElse(_ cond: MetalBooleanArray, _ left: Bool, _ right: Bool) throws -> MetalBooleanArray {
        try MetalBooleanArray.fromUInt8Array(try MetalArray<UInt8>.ifElse(cond, left ? 1 : 0, right ? 1 : 0))
    }

    /// Arrow `if_else` with this array as the condition, over boolean branches.
    public func ifElse(_ left: MetalBooleanArray, _ right: MetalBooleanArray) throws -> MetalBooleanArray {
        try MetalBooleanArray.ifElse(self, left, right)
    }
}

// MARK: - coalesce

extension MetalArray {
    /// Arrow `coalesce`: the first non-null value across `arrays`, element-wise. An element is null
    /// only when it is null in every input. All inputs must have the same length.
    public static func coalesce(_ arrays: [MetalArray<T>]) throws -> MetalArray<T> {
        guard let first = arrays.first else {
            throw ArrowMetalError.invalidArrowArray("coalesce needs at least one array")
        }
        var acc = first
        for next in arrays.dropFirst() {
            if acc.validity == nil { break }        // already non-null everywhere
            try acc.checkSameLength(next)
            acc = try coalescePair(acc, next)
        }
        return acc
    }

    private static func coalescePair(_ a: MetalArray<T>, _ b: MetalArray<T>) throws -> MetalArray<T> {
        let ctx = a.context
        let n = a.dispatchLength
        try Dispatch.checkLength(n)
        let outValues = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * T.byteWidth, zeroed: false, context: ctx)
        let validBytes = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1), zeroed: false, context: ctx)
        if n > 0 {
            let mslT = Dispatch.moveType(T.self)
            let pso = try Structural.pipeline(ctx, StructuralSource.moves(T: mslT), "st_coalesce2", type: mslT)
            let flags = (a.validity == nil ? 0 : 1) | (b.validity == nil ? 0 : 2)
            let av = a.validity ?? a.values, bv = b.validity ?? b.values
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(a.values.mtl, offset: a.values.offset, index: 0)
                enc.setBuffer(av.mtl, offset: av.offset, index: 1)
                enc.setBuffer(b.values.mtl, offset: b.values.offset, index: 2)
                enc.setBuffer(bv.mtl, offset: bv.offset, index: 3)
                Dispatch.setLength(enc, n, a.lengthBuffer, index: 4)
                Dispatch.setUInt(enc, flags, index: 5)
                enc.setBuffer(outValues.mtl, offset: outValues.offset, index: 6)
                enc.setBuffer(validBytes.mtl, offset: validBytes.offset, index: 7)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
            ctx.retainUntilFlush(a); ctx.retainUntilFlush(b); ctx.retainUntilFlush(validBytes)
        }
        // b may have no bitmap, in which case nothing is null any more and the result carries none.
        if b.validity == nil {
            return a.inheritPending(MetalArray<T>(length: a.knownLength, nullCount: 0, validity: nil, values: outValues, context: ctx))
        }
        let outValidity = n > 0 ? try BitmapOps.packBits(ctx, bytes: validBytes, bits: n, lengthBuffer: a.lengthBuffer) : nil
        let res = a.inheritPending(MetalArray<T>(length: a.knownLength, nullCount: 0, validity: outValidity, values: outValues, context: ctx))
        res.recomputeNullCount()
        return res
    }
}

// MARK: - Kleene logic

extension MetalBooleanArray {
    /// Arrow `and_kleene`: three-valued AND. `false AND null` is `false`, `true AND null` is null.
    public func andKleene(_ other: MetalBooleanArray) throws -> MetalBooleanArray {
        try kleene("st_and_kleene", other)
    }

    /// Arrow `or_kleene`: three-valued OR. `true OR null` is `true`, `false OR null` is null.
    public func orKleene(_ other: MetalBooleanArray) throws -> MetalBooleanArray {
        try kleene("st_or_kleene", other)
    }

    private func kleene(_ fn: String, _ other: MetalBooleanArray) throws -> MetalBooleanArray {
        try checkSameLength(other)
        // With no nulls on either side, Kleene logic is ordinary boolean logic.
        if validity == nil && other.validity == nil {
            return fn == "st_and_kleene" ? try and(other) : try or(other)
        }
        let ctx = context
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let words = BitmapOps.words(bits: n)
        let outValues = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: n), zeroed: false, context: ctx)
        let outValidity = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: n), zeroed: false, context: ctx)
        if words > 0 {
            let pso = try Structural.pipeline(ctx, StructuralSource.common, fn, type: "common")
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
}

// MARK: - is_in / index_in

/// A set prepared for GPU lookup: the distinct non-null values of the caller's set in ascending order,
/// plus (for `index_in`) the position in the caller's set of the first occurrence of each of them.
struct LookupSet<T: ArrowPrimitive> {
    let values: MetalArray<T>
    let firstIndex: MetalArray<Int32>?
    var count: Int { values.length }
}

extension MetalArray {
    /// Arrow `is_in`: true where the element appears in `set`.
    ///
    /// Nulls in `set` are ignored and a null element is not in the set, so the result never contains
    /// nulls (Arrow's `null_matching_behavior = "skip"`). Float equality is Arrow value equality, as
    /// in `unique()`: every NaN is one value and `-0.0` equals `0.0`.
    public func isIn(_ set: MetalArray<T>) throws -> MetalBooleanArray {
        try isIn(prepared: LookupSet(values: try set.unique(), firstIndex: nil))
    }

    /// Arrow `is_in` against a host-side set.
    public func isIn(_ set: [T]) throws -> MetalBooleanArray {
        try isIn(try MetalArray<T>(set, context: context))
    }

    /// Arrow `index_in`: for each element, the position in `set` of its first occurrence there, or
    /// null when the element is null or not in the set.
    public func indexIn(_ set: MetalArray<T>) throws -> MetalArray<Int32> {
        try indexIn(prepared: try Self.prepareIndexed(set))
    }

    /// Arrow `index_in` against a host-side set.
    public func indexIn(_ set: [T]) throws -> MetalArray<Int32> {
        try indexIn(try MetalArray<T>(set, context: context))
    }

    /// Sorted distinct values of `set`, plus the first row of `set` carrying each of them.
    /// The rank of a row in the sorted distinct values is exactly its dictionary code, so the first
    /// row per value is a group-by min over the codes.
    static func prepareIndexed(_ set: MetalArray<T>) throws -> LookupSet<T> {
        let (gb, unique) = try set.groupBy()
        guard unique.length > 0 else { return LookupSet(values: unique, firstIndex: nil) }
        let rows = try MetalArray<Int32>.iota(set.length, context: set.context)
        return LookupSet(values: unique, firstIndex: try gb.min(rows))
    }

    func isIn(prepared set: LookupSet<T>) throws -> MetalBooleanArray {
        let ctx = context
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let words = BitmapOps.words(bits: n)
        let setCount = set.count
        let out = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: n), context: ctx)
        if words > 0 && setCount > 0 {
            let key = Structural.lookupKey(T.self)
            let pso = try Structural.pipeline(ctx, StructuralSource.lookup(K: key.K, toKey: key.toKey), "st_is_in", type: key.cacheKey)
            let vld = validity ?? values
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                enc.setBuffer(vld.mtl, offset: vld.offset, index: 1)
                Dispatch.setLength(enc, n, lengthBuffer, index: 2)
                enc.setBuffer(set.values.values.mtl, offset: set.values.values.offset, index: 3)
                Dispatch.setUInt(enc, setCount, index: 4)
                Dispatch.setUInt(enc, validity == nil ? 0 : 1, index: 5)
                enc.setBuffer(out.mtl, offset: out.offset, index: 6)
                Dispatch.dispatch1D(enc, pso, count: words)
            }
            ctx.retainUntilFlush(self); ctx.retainUntilFlush(set.values)
        }
        return inheritPending(MetalBooleanArray(length: knownLength, nullCount: 0, validity: nil, values: out, context: ctx))
    }

    func indexIn(prepared set: LookupSet<T>) throws -> MetalArray<Int32> {
        let ctx = context
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let setCount = set.count
        let outValues = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * 4, context: ctx)
        let validBytes = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1), context: ctx)
        if n > 0 && setCount > 0, let firstIndex = set.firstIndex {
            let key = Structural.lookupKey(T.self)
            let pso = try Structural.pipeline(ctx, StructuralSource.lookup(K: key.K, toKey: key.toKey), "st_index_in", type: key.cacheKey)
            let vld = validity ?? values
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                enc.setBuffer(vld.mtl, offset: vld.offset, index: 1)
                Dispatch.setLength(enc, n, lengthBuffer, index: 2)
                enc.setBuffer(set.values.values.mtl, offset: set.values.values.offset, index: 3)
                Dispatch.setUInt(enc, setCount, index: 4)
                Dispatch.setUInt(enc, validity == nil ? 0 : 1, index: 5)
                enc.setBuffer(firstIndex.values.mtl, offset: firstIndex.values.offset, index: 6)
                enc.setBuffer(outValues.mtl, offset: outValues.offset, index: 7)
                enc.setBuffer(validBytes.mtl, offset: validBytes.offset, index: 8)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
            ctx.retainUntilFlush(self); ctx.retainUntilFlush(set.values)
            ctx.retainUntilFlush(firstIndex); ctx.retainUntilFlush(validBytes)
        }
        let outValidity = n > 0 ? try BitmapOps.packBits(ctx, bytes: validBytes, bits: n, lengthBuffer: lengthBuffer) : nil
        let res = inheritPending(MetalArray<Int32>(length: knownLength, nullCount: 0, validity: outValidity, values: outValues, context: ctx))
        res.recomputeNullCount()
        return res
    }
}
