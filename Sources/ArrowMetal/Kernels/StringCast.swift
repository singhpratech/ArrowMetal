import Foundation
import Metal

/// Arrow `cast` between `utf8` and the numeric and boolean types.
///
/// | direction | types | where |
/// |---|---|---|
/// | number → string | `int8/16/32/64`, `uint8/16/32/64` | **GPU** (`str_itoa_*`, two-pass) |
/// | number → string | `float32`, `float64` | CPU |
/// | boolean → string | `bool` | CPU |
/// | string → number | `int8/16/32/64`, `uint8/16/32/64` | **GPU** (`str_parse_int`) |
/// | string → number | `float32`, `float64` | CPU |
/// | string → boolean | `bool` | CPU |
///
/// Nulls always propagate; a null row of a `toStrings()` result emits no bytes and shares the input's
/// validity bitmap zero-copy.
extension MetalArray {

    /// Arrow `cast(utf8)`: the decimal text of every value.
    ///
    /// **Integers** are formatted on the GPU: an optional `-` then the shortest decimal digit string,
    /// no leading zeros, no thousands separators. `Int64.min` and `UInt64.max` are exact.
    ///
    /// **Floats** are formatted on the CPU with Swift's `Double`/`Float` description, which is the
    /// *shortest decimal string that round-trips* — the same guarantee `%.17g` gives without its
    /// trailing noise (`0.1` rather than `0.10000000000000001`). Two documented differences from
    /// Arrow's float→string cast: a value with no fractional part keeps a `.0` (`1.0`, not `1`), and
    /// the exponent form is Swift's (`1e+20`). Infinities and NaN are `inf`, `-inf`, `nan`.
    public func toStrings() throws -> MetalStringArray {
        if T.isFloatingPoint { return try floatToStrings() }
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let isSigned = T.minValue < (0 as T)
        let src = StringCastSource.itoa(T: T.mslType, isSigned: isSigned)
        let key = "strcast/\(T.mslType)"
        let lens = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 4, 4), zeroed: true, context: ctx)
        let vb = validity ?? values                              // never read when hasValidity is 0
        let hasValidity = validity == nil ? 0 : 1

        if n > 0 {
            let pLen = try ctx.pipeline(source: src, function: "str_itoa_len", cacheKey: "\(key)/itoa_len")
            try ctx.run { enc in
                enc.setComputePipelineState(pLen)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 1)
                Dispatch.setLength(enc, n, nil, index: 2)
                Dispatch.setUInt(enc, hasValidity, index: 3)
                enc.setBuffer(lens.mtl, offset: lens.offset, index: 4)
                Dispatch.dispatch1D(enc, pLen, count: n)
            }
        }
        let outOffsets = try MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: lens, context: ctx)
            .exclusiveScanToOffsets()
        let total = Int(withExtendedLifetime(outOffsets) { outOffsets.typed(Int32.self)[n] })
        let outData = try MetalArrowBuffer.allocate(byteCount: Swift.max(total, 1), zeroed: false, context: ctx)
        if n > 0 {
            let pWrite = try ctx.pipeline(source: src, function: "str_itoa_write", cacheKey: "\(key)/itoa_write")
            try ctx.run { enc in
                enc.setComputePipelineState(pWrite)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 1)
                Dispatch.setLength(enc, n, nil, index: 2)
                Dispatch.setUInt(enc, hasValidity, index: 3)
                enc.setBuffer(outOffsets.mtl, offset: outOffsets.offset, index: 4)
                enc.setBuffer(outData.mtl, offset: outData.offset, index: 5)
                Dispatch.dispatch1D(enc, pWrite, count: n)
            }
        }
        ctx.retainUntilFlush(lens)
        return MetalStringArray(length: n, nullCount: nullCount, validity: validity,
                                offsets: outOffsets, data: outData, context: ctx)
    }

    /// Float formatting on the host: Swift's shortest round-tripping description.
    private func floatToStrings() throws -> MetalStringArray {
        let n = length
        var rows = [String?](repeating: nil, count: n)
        for i in 0..<n {
            guard let v = self[i] else { continue }
            rows[i] = T.self == Float.self ? "\(v as! Float)" : "\(v.asDouble)"
        }
        return try MetalStringArray(rows, context: context)
    }
}

extension MetalBooleanArray {
    /// Arrow `cast(utf8)` on a boolean column: `"true"` / `"false"`, matching Arrow's spelling. CPU.
    public func toStrings() throws -> MetalStringArray {
        let n = length
        var rows = [String?](repeating: nil, count: n)
        for i in 0..<n { rows[i] = self[i].map { $0 ? "true" : "false" } }
        return try MetalStringArray(rows, context: context)
    }
}

extension MetalStringArray {

    /// Arrow `cast` from `utf8` to a numeric type.
    ///
    /// **Integers** parse on the GPU. The grammar is the whole value matching `[+-]?[0-9]+`: no
    /// surrounding whitespace, no radix prefix, no exponent, no separators; leading zeros are fine.
    /// A row that does not parse — empty, malformed, or out of the target type's range — comes back
    /// **null**, which is Arrow's `safe=false` behaviour. A `-` sign is rejected for an unsigned
    /// target, `"-0"` included. Pass `strict: true` to throw instead of nulling those rows.
    ///
    /// **Floats** parse on the CPU with Swift's `Float`/`Double` initialiser, which accepts decimal
    /// and exponent forms plus `inf`, `-inf` and `nan` and rejects trailing junk.
    public func parse<T: ArrowPrimitive>(_ type: T.Type, strict: Bool = false) throws -> MetalArray<T> {
        let result: MetalArray<T> = T.isFloatingPoint ? try parseFloat(T.self) : try parseInteger(T.self)

        if strict, result.nullCount > nullCount {
            throw ArrowMetalError.invalidArrowArray(
                "\(result.nullCount - nullCount) of \(length) values are not a valid \(T.arrowFormat)")
        }
        return result
    }

    private func parseInteger<T: ArrowPrimitive>(_: T.Type) throws -> MetalArray<T> {
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let isSigned = T.minValue < (0 as T)
        var limPos: UInt64 = isSigned ? UInt64(T.maxValue.asInt64) : T.maxValue.asUInt64
        var limNeg: UInt64 = isSigned ? UInt64(T.maxValue.asInt64) + 1 : 0
        let words = BitmapOps.words(bits: n)
        let outVals = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * T.byteWidth, T.byteWidth),
                                                    zeroed: true, context: ctx)
        let outValid = try MetalArrowBuffer.allocate(byteCount: Swift.max(words * 4, 4), zeroed: true, context: ctx)
        let vb = validity ?? offsets                              // never read when hasValidity is 0
        if n > 0 {
            let src = StringCastSource.parse(T: T.mslType, isSigned: isSigned)
            let pso = try ctx.pipeline(source: src, function: "str_parse_int",
                                       cacheKey: "strcast/\(T.mslType)/parse_int")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                Dispatch.setLength(enc, n, nil, index: 2)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 3)
                Dispatch.setUInt(enc, validity == nil ? 0 : 1, index: 4)
                enc.setBytes(&limPos, length: 8, index: 5)
                enc.setBytes(&limNeg, length: 8, index: 6)
                enc.setBuffer(outVals.mtl, offset: outVals.offset, index: 7)
                enc.setBuffer(outValid.mtl, offset: outValid.offset, index: 8)
                Dispatch.dispatch1D(enc, pso, count: words)
            }
        }
        let out = MetalArray<T>(length: n, nullCount: 0, validity: outValid, values: outVals, context: ctx)
        out.recomputeNullCount()
        return out
    }

    private func parseFloat<T: ArrowPrimitive>(_: T.Type) throws -> MetalArray<T> {
        let n = length
        let ctx = context
        let outVals = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * T.byteWidth, T.byteWidth),
                                                    zeroed: true, context: ctx)
        let outValid = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 1),
                                                     zeroed: true, context: ctx)
        let vals = outVals.mutableTyped(T.self), bits = outValid.mutableTyped(UInt8.self)
        let isFloat32 = T.self == Float.self
        forEachRowConcurrently { i, s in
            guard let s else { return }
            if isFloat32 {
                guard let v = Float(s) else { return }
                vals[i] = v as! T
            } else {
                guard let v = Double(s) else { return }
                vals[i] = v as! T
            }
            Bitmap.set(bits, i)
        }
        let out = MetalArray<T>(length: n, nullCount: 0, validity: outValid, values: outVals, context: ctx)
        out.recomputeNullCount()
        return out
    }

    /// Arrow `cast` from `utf8` to `bool`: `"true"` / `"false"` / `"1"` / `"0"`, case-insensitively.
    /// Anything else is null (or throws when `strict` is set). CPU.
    public func parseBool(strict: Bool = false) throws -> MetalBooleanArray {
        let n = length, ctx = context
        let outVals = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 1),
                                                    zeroed: true, context: ctx)
        let outValid = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 1),
                                                     zeroed: true, context: ctx)
        let vp = outVals.mutableTyped(UInt8.self), bp = outValid.mutableTyped(UInt8.self)
        forEachRowConcurrently { i, s in
            guard let s else { return }
            switch s.lowercased() {
            case "true", "1": Bitmap.set(vp, i); Bitmap.set(bp, i)
            case "false", "0": Bitmap.set(bp, i)
            default: return
            }
        }
        let out = MetalBooleanArray(length: n, nullCount: 0, validity: outValid, values: outVals, context: ctx)
        out.recomputeNullCount()
        if strict, out.nullCount > nullCount {
            throw ArrowMetalError.invalidArrowArray("\(out.nullCount - nullCount) of \(n) values are not a valid bool")
        }
        return out
    }
}
