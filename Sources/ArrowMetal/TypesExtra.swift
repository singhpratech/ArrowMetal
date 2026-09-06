import Foundation
import Metal
import CArrowABI

// The remaining rows of the Arrow type matrix: `null` ("n"), `float16` ("e"), `decimal32` / `decimal64`
// ("d:p,s,32" / "d:p,s,64"), the three `interval` layouts ("tiM", "tiD", "tin") and `fixed_size_binary`
// ("w:N"). Each one gets C Data Interface import and export (zero-copy on a page-aligned, offset-free
// producer, exactly as the existing importer), `filter` / `take` / `slice` on the GPU, and the compute
// each type actually has:
//
//   * `float16` computes by casting to `float32` on the GPU (Metal's native `half`), so compare, sum, min
//     and max are the Float32 kernels. An arithmetic result is only rounded back to half when the caller
//     asks for it with `toFloat16()`; there is no fused half arithmetic.
//   * `decimal32` / `decimal64` widen to `decimal128` on the GPU and compute there, then narrow back.
//   * an `interval` adds to a timestamp / date column through `MetalTemporalArray.addInterval(_:)`.
//   * `fixed_size_binary` compares byte-wise against a scalar or another array, and hashes.
//
// Everything that only moves bytes (interval, fixed_size_binary) shares one gather kernel parameterised
// by the record width, in `Kernels/TypesExtraSource.swift`.

// MARK: - Shared fixed-width machinery

/// `take` / `filter` / `slice` / compare / hash over opaque fixed-width records of `byteWidth` bytes.
enum FixedWidth {
    /// GPU gather of `length` records of `byteWidth` bytes through `indices`. A null index yields a null
    /// record; an out-of-range index raises after the dispatch, as `MetalArray.take` does.
    static func take(_ ctx: MetalContext, values: MetalArrowBuffer, validity: MetalArrowBuffer?,
                     length: Int, byteWidth: Int, indices: MetalArray<Int32>)
        throws -> (values: MetalArrowBuffer, validity: MetalArrowBuffer?, length: Int) {
        try Dispatch.checkLength(length)
        let n = indices.length
        try Dispatch.checkLength(n)
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * byteWidth, 1), zeroed: false, context: ctx)
        let hasV = validity != nil, hasIV = indices.validity != nil
        let validBytes = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1), zeroed: false, context: ctx)
        let errorFlag = try MetalArrowBuffer.allocate(byteCount: 4, context: ctx)
        if n > 0 {
            let pso = try ctx.pipeline(source: TypesExtraSource.fixedWidth, function: "fw_take", cacheKey: "fw/take")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                let vb = validity ?? values
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 1)
                enc.setBuffer(indices.values.mtl, offset: indices.values.offset, index: 2)
                let ib = indices.validity ?? indices.values
                enc.setBuffer(ib.mtl, offset: ib.offset, index: 3)
                Dispatch.setUInt(enc, n, index: 4)
                Dispatch.setUInt(enc, length, index: 5)
                Dispatch.setUInt(enc, (hasV ? 1 : 0) | (hasIV ? 2 : 0), index: 6)
                Dispatch.setUInt(enc, byteWidth, index: 7)
                enc.setBuffer(out.mtl, offset: out.offset, index: 8)
                enc.setBuffer(validBytes.mtl, offset: validBytes.offset, index: 9)
                enc.setBuffer(errorFlag.mtl, offset: errorFlag.offset, index: 10)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        try ctx.afterFlush { [errorFlag] in
            if errorFlag.typed(UInt32.self)[0] != 0 {
                throw ArrowMetalError.invalidArrowArray("take: index out of range (array length \(length))")
            }
        }
        ctx.retainUntilFlush(errorFlag); ctx.retainUntilFlush(indices); ctx.retainUntilFlush(validBytes)
        var outValidity: MetalArrowBuffer? = nil
        if (hasV || hasIV) && n > 0 { outValidity = try BitmapOps.packBits(ctx, bytes: validBytes, bits: n) }
        return (out, outValidity, n)
    }

    /// Byte equality against one scalar record (`ne` gives `not_equal`). Validity is shared with the input.
    static func compareScalar(_ ctx: MetalContext, values: MetalArrowBuffer, length: Int, byteWidth: Int,
                              scalar: [UInt8], notEqual: Bool) throws -> MetalArrowBuffer {
        let pat = try MetalArrowBuffer.allocate(byteCount: Swift.max(byteWidth, 1), context: ctx)
        let pp = pat.mutableTyped(UInt8.self)
        for i in 0..<Swift.min(byteWidth, scalar.count) { pp[i] = scalar[i] }
        let bytes = try MetalArrowBuffer.allocate(byteCount: Swift.max(length, 1), zeroed: true, context: ctx)
        if length > 0 {
            let pso = try ctx.pipeline(source: TypesExtraSource.fixedWidth, function: "fw_cmp_scalar", cacheKey: "fw/cmp_scalar")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                enc.setBuffer(pat.mtl, offset: pat.offset, index: 1)
                Dispatch.setLength(enc, length, nil, index: 2)
                Dispatch.setUInt(enc, byteWidth, index: 3)
                Dispatch.setUInt(enc, notEqual ? 1 : 0, index: 4)
                enc.setBuffer(bytes.mtl, offset: bytes.offset, index: 5)
                Dispatch.dispatch1D(enc, pso, count: length)
            }
        }
        ctx.retainUntilFlush(pat)
        return try BitmapOps.packBits(ctx, bytes: bytes, bits: Swift.max(length, 1))
    }

    /// Element-wise byte equality between two equal-width arrays.
    static func compareArray(_ ctx: MetalContext, _ a: MetalArrowBuffer, _ b: MetalArrowBuffer,
                             length: Int, byteWidth: Int, notEqual: Bool) throws -> MetalArrowBuffer {
        let bytes = try MetalArrowBuffer.allocate(byteCount: Swift.max(length, 1), zeroed: true, context: ctx)
        if length > 0 {
            let pso = try ctx.pipeline(source: TypesExtraSource.fixedWidth, function: "fw_cmp_array", cacheKey: "fw/cmp_array")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(a.mtl, offset: a.offset, index: 0)
                enc.setBuffer(b.mtl, offset: b.offset, index: 1)
                Dispatch.setLength(enc, length, nil, index: 2)
                Dispatch.setUInt(enc, byteWidth, index: 3)
                Dispatch.setUInt(enc, notEqual ? 1 : 0, index: 4)
                enc.setBuffer(bytes.mtl, offset: bytes.offset, index: 5)
                Dispatch.dispatch1D(enc, pso, count: length)
            }
        }
        return try BitmapOps.packBits(ctx, bytes: bytes, bits: Swift.max(length, 1))
    }

    /// FNV-1a 64 over each record's bytes.
    static func hash64(_ ctx: MetalContext, values: MetalArrowBuffer, length: Int, byteWidth: Int) throws -> MetalArrowBuffer {
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(length * 8, 1), zeroed: true, context: ctx)
        if length > 0 {
            let pso = try ctx.pipeline(source: TypesExtraSource.fixedWidth, function: "fw_hash64", cacheKey: "fw/hash64")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                Dispatch.setLength(enc, length, nil, index: 1)
                Dispatch.setUInt(enc, byteWidth, index: 2)
                enc.setBuffer(out.mtl, offset: out.offset, index: 3)
                Dispatch.dispatch1D(enc, pso, count: length)
            }
        }
        return out
    }

    /// Host-side bitmap slice (a nested slice is metadata work, not kernel work).
    static func sliceBitmap(_ v: MetalArrowBuffer?, offset: Int, length: Int, _ ctx: MetalContext) throws -> MetalArrowBuffer? {
        guard let v, length > 0 else { return nil }
        if offset % 8 == 0 { return v.view(byteOffset: offset / 8, byteCount: Bitmap.byteCount(bits: length)) }
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: length), 1), context: ctx)
        let sp = v.typed(UInt8.self), dp = out.mutableTyped(UInt8.self)
        for i in 0..<length where Bitmap.isSet(sp, offset + i) { Bitmap.set(dp, i) }
        return out
    }
}

// MARK: - null

/// The Arrow `null` type ("n"): a length and nothing else. Every element is null, there are no buffers,
/// and `filter` / `take` / `slice` only have to work out the new length.
public final class MetalNullArray: @unchecked Sendable {
    public let length: Int
    public let context: MetalContext

    public init(length: Int, context: MetalContext = .shared) {
        self.length = length
        self.context = context
    }

    public var arrowFormat: String { "n" }
    /// Every element of a null array is null.
    public var nullCount: Int { length }
    public func isValid(_ i: Int) -> Bool { false }

    public func filter(_ mask: MetalBooleanArray) throws -> MetalNullArray {
        guard mask.length == length else { throw ArrowMetalError.lengthMismatch(length, mask.length) }
        return MetalNullArray(length: mask.trueCount, context: context)
    }
    public func take<I: ArrowIndex>(_ indices: MetalArray<I>) throws -> MetalNullArray {
        MetalNullArray(length: indices.length, context: context)
    }
    public func slice(offset: Int, length n: Int) throws -> MetalNullArray {
        guard offset >= 0, n >= 0, offset + n <= length else {
            throw ArrowMetalError.invalidArrowArray("null slice \(offset)..<\(offset + n) is out of range (length \(length))")
        }
        return MetalNullArray(length: n, context: context)
    }
}

// MARK: - float16

/// An Arrow `float16` ("e") array: IEEE-754 binary16 bit patterns in a `MetalArray<UInt16>`.
///
/// Compute goes through `float32`: `toFloat32()` runs Metal's native `half` widening on the GPU, the
/// Float32 kernels do the work, and `MetalArray<Float>.toFloat16()` rounds back (nearest-even) only when
/// the caller asks. Nothing here computes in half precision, so a sum of many halves is a Float32 sum.
public final class MetalFloat16Array: @unchecked Sendable {
    /// The raw binary16 bit patterns. Selection kernels move these directly.
    public let bits: MetalArray<UInt16>

    public init(bits: MetalArray<UInt16>) { self.bits = bits }

    /// Builds from Swift `Float` values, rounding each to binary16 on the host.
    public convenience init(_ values: [Float?], context: MetalContext = .shared) throws {
        let raw: [UInt16?] = values.map { $0.map { Float16Bits.encode($0) } }
        self.init(bits: try MetalArray<UInt16>(raw, context: context))
    }

    public var arrowFormat: String { "e" }
    public var length: Int { bits.length }
    public var nullCount: Int { bits.nullCount }
    public var validity: MetalArrowBuffer? { bits.validity }
    public var values: MetalArrowBuffer { bits.values }
    public var context: MetalContext { bits.context }
    public func isValid(_ i: Int) -> Bool { bits.isValid(i) }
    /// Element `i` decoded to `Float` on the host, or nil when it is null.
    public subscript(i: Int) -> Float? { bits[i].map { Float16Bits.decode($0) } }
    public func toArray() -> [Float?] { (0..<length).map { self[$0] } }

    /// Arrow `cast(float32)`: exact, one GPU pass through Metal's `half`.
    public func toFloat32() throws -> MetalArray<Float> {
        let ctx = context
        let n = length
        try Dispatch.checkLength(n)
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 4, 1), zeroed: false, context: ctx)
        if n > 0 {
            let pso = try ctx.pipeline(source: TypesExtraSource.float16, function: "f16_to_f32", cacheKey: "f16/to_f32")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(bits.values.mtl, offset: bits.values.offset, index: 0)
                Dispatch.setLength(enc, n, nil, index: 1)
                enc.setBuffer(out.mtl, offset: out.offset, index: 2)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        return MetalArray<Float>(length: n, nullCount: nullCount, validity: validity, values: out, context: ctx)
    }

    // Compute: widen, run the Float32 kernel, and (for value-producing ops) let the caller decide when to
    // round back to half.
    public func compare(_ op: CompareOp, _ scalar: Float) throws -> MetalBooleanArray {
        try toFloat32().compare(op, scalar)
    }
    public func compare(_ op: CompareOp, _ other: MetalFloat16Array) throws -> MetalBooleanArray {
        try toFloat32().compare(op, try other.toFloat32())
    }
    /// Sum in `float32` (never in half), matching what a widened column would give.
    public func sum() throws -> Double? {
        guard let s = try toFloat32().sum() else { return nil }
        switch s { case .float(let v): return v; case .int(let v): return Double(v); case .uint(let v): return Double(v) }
    }
    public func min() throws -> Float? { try toFloat32().min() }
    public func max() throws -> Float? { try toFloat32().max() }

    public func filter(_ mask: MetalBooleanArray) throws -> MetalFloat16Array {
        MetalFloat16Array(bits: try bits.filter(mask))
    }
    public func take<I: ArrowIndex>(_ indices: MetalArray<I>) throws -> MetalFloat16Array {
        MetalFloat16Array(bits: try bits.take(indices))
    }
    public func slice(offset: Int, length n: Int) throws -> MetalFloat16Array {
        MetalFloat16Array(bits: try bits.slice(offset: offset, length: n))
    }
}

extension MetalArray where T == Float {
    /// Arrow `cast(float16)`: rounds to nearest-even on the GPU, overflowing to +/-infinity.
    public func toFloat16() throws -> MetalFloat16Array {
        let ctx = context
        let n = length
        try Dispatch.checkLength(n)
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 2, 1), zeroed: false, context: ctx)
        if n > 0 {
            let pso = try ctx.pipeline(source: TypesExtraSource.float16, function: "f32_to_f16", cacheKey: "f16/from_f32")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                Dispatch.setLength(enc, n, nil, index: 1)
                enc.setBuffer(out.mtl, offset: out.offset, index: 2)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        return MetalFloat16Array(bits: MetalArray<UInt16>(length: n, nullCount: nullCount, validity: validity,
                                                          values: out, context: ctx))
    }
}

/// Host-side IEEE-754 binary16 <-> `Float`, used by the Swift constructors and accessors (the GPU has a
/// native `half`, so nothing on the kernel path goes through this). Written out rather than using Swift's
/// `Float16`, which is unavailable on x86_64, and rounding to nearest-even so it agrees with the GPU cast.
enum Float16Bits {
    static func decode(_ b: UInt16) -> Float {
        let negative = (b & 0x8000) != 0
        let exp = Int((b >> 10) & 0x1F)
        let frac = UInt32(b & 0x03FF)
        var value: Float
        if exp == 0 {
            value = Float(frac) * 0x1p-24                      // zero or subnormal
        } else if exp == 0x1F {
            value = frac == 0 ? .infinity : Float.nan
        } else {
            value = Float(bitPattern: (UInt32(exp - 15 + 127) << 23) | (frac << 13))
        }
        return negative ? -value : value
    }

    static func encode(_ f: Float) -> UInt16 {
        let bits = f.bitPattern
        let sign = UInt16(truncatingIfNeeded: (bits >> 16) & 0x8000)
        let exp = Int((bits >> 23) & 0xFF)
        let frac = bits & 0x007F_FFFF
        if exp == 0xFF { return frac == 0 ? (sign | 0x7C00) : (sign | 0x7E00) }   // infinity / quiet NaN
        let e = exp - 127 + 15
        if e >= 0x1F { return sign | 0x7C00 }                  // overflow to infinity
        if e > 0 {
            var half = sign | UInt16(e << 10) | UInt16(truncatingIfNeeded: frac >> 13)
            let rest = frac & 0x1FFF
            if rest > 0x1000 || (rest == 0x1000 && (frac >> 13) & 1 == 1) { half &+= 1 }
            return half
        }
        if e < -10 { return sign }                             // underflow to zero
        // Subnormal half: shift the implicit leading 1 back in, then round to nearest-even.
        let m = frac | 0x0080_0000
        let shift = UInt32(14 - e)
        var half = UInt16(truncatingIfNeeded: m >> shift)
        let rest = m & ((1 << shift) - 1)
        let halfway: UInt32 = 1 << (shift - 1)
        if rest > halfway || (rest == halfway && half & 1 == 1) { half &+= 1 }
        return sign | half
    }
}

// MARK: - decimal32 / decimal64

/// An Arrow `decimal32` ("d:p,s,32") or `decimal64` ("d:p,s,64") type.
public struct ArrowSmallDecimalType: Hashable, Sendable, CustomStringConvertible {
    public let precision: Int
    public let scale: Int
    /// 32 or 64.
    public let bitWidth: Int

    public init(precision: Int, scale: Int, bitWidth: Int) throws {
        guard bitWidth == 32 || bitWidth == 64 else {
            throw ArrowMetalError.unsupportedType("decimal\(bitWidth) is not decimal32 or decimal64")
        }
        let maxP = bitWidth == 32 ? 9 : 18
        guard precision >= 1, precision <= maxP else {
            throw ArrowMetalError.unsupportedType("decimal\(bitWidth) precision \(precision) is outside 1...\(maxP)")
        }
        guard scale >= -maxP, scale <= maxP else {
            throw ArrowMetalError.unsupportedType("decimal\(bitWidth) scale \(scale) is outside -\(maxP)...\(maxP)")
        }
        self.precision = precision; self.scale = scale; self.bitWidth = bitWidth
    }

    /// Parses `d:p,s,32` / `d:p,s,64`; nil for anything else (including decimal128 and decimal256).
    public init?(format f: String) {
        guard f.hasPrefix("d:") else { return nil }
        let parts = f.dropFirst(2).split(separator: ",", omittingEmptySubsequences: false).map(String.init)
        guard parts.count == 3, let p = Int(parts[0]), let s = Int(parts[1]), let w = Int(parts[2]),
              w == 32 || w == 64, let t = try? ArrowSmallDecimalType(precision: p, scale: s, bitWidth: w) else { return nil }
        self = t
    }

    public var arrowFormat: String { "d:\(precision),\(scale),\(bitWidth)" }
    public var byteWidth: Int { bitWidth / 8 }
    public var description: String { "decimal\(bitWidth)(\(precision), \(scale))" }
    /// The decimal128 type the widening cast produces (same scale, precision raised to fit).
    public func widened() throws -> ArrowDecimalType {
        try ArrowDecimalType(precision: precision, scale: scale, bitWidth: 128)
    }
}

/// An Arrow `decimal32` / `decimal64` array: the unscaled value in an int32 or int64 column.
///
/// There are no narrow-decimal kernels: compute widens to `decimal128` on the GPU (`toDecimal128()`),
/// runs the existing decimal kernels there, and narrows back with `MetalDecimalArray.narrowed(to:)`.
public final class MetalSmallDecimalArray: @unchecked Sendable {
    public enum Storage {
        case int32(MetalArray<Int32>)
        case int64(MetalArray<Int64>)
    }

    public let type: ArrowSmallDecimalType
    public let storage: Storage

    public init(type: ArrowSmallDecimalType, _ values: MetalArray<Int32>) throws {
        guard type.bitWidth == 32 else { throw ArrowMetalError.unsupportedType("\(type) is stored as int64, not int32") }
        self.type = type; self.storage = .int32(values)
    }
    public init(type: ArrowSmallDecimalType, _ values: MetalArray<Int64>) throws {
        guard type.bitWidth == 64 else { throw ArrowMetalError.unsupportedType("\(type) is stored as int32, not int64") }
        self.type = type; self.storage = .int64(values)
    }
    /// Builds from unscaled integer values.
    public convenience init(type: ArrowSmallDecimalType, _ values: [Int64?], context: MetalContext = .shared) throws {
        if type.bitWidth == 64 {
            try self.init(type: type, try MetalArray<Int64>(values, context: context))
        } else {
            var narrow: [Int32?] = []
            narrow.reserveCapacity(values.count)
            for v in values {
                guard let v else { narrow.append(nil); continue }
                guard let n = Int32(exactly: v) else { throw ArrowMetalError.invalidArrowArray("\(v) does not fit \(type)") }
                narrow.append(n)
            }
            try self.init(type: type, try MetalArray<Int32>(narrow, context: context))
        }
    }

    public var arrowFormat: String { type.arrowFormat }
    public var length: Int { switch storage { case .int32(let a): return a.length; case .int64(let a): return a.length } }
    public var nullCount: Int { switch storage { case .int32(let a): return a.nullCount; case .int64(let a): return a.nullCount } }
    public var validity: MetalArrowBuffer? { switch storage { case .int32(let a): return a.validity; case .int64(let a): return a.validity } }
    public var values: MetalArrowBuffer { switch storage { case .int32(let a): return a.values; case .int64(let a): return a.values } }
    public var context: MetalContext { switch storage { case .int32(let a): return a.context; case .int64(let a): return a.context } }
    public func isValid(_ i: Int) -> Bool {
        switch storage { case .int32(let a): return a.isValid(i); case .int64(let a): return a.isValid(i) }
    }
    /// The unscaled value of element `i`, or nil when it is null.
    public subscript(i: Int) -> Int64? {
        switch storage { case .int32(let a): return a[i].map(Int64.init); case .int64(let a): return a[i] }
    }
    public func toArray() -> [Int64?] { (0..<length).map { self[$0] } }

    /// GPU widening cast to `decimal128` (sign extension into two limbs). Exact for every value.
    public func toDecimal128() throws -> MetalDecimalArray {
        let ctx = context
        let n = length
        try Dispatch.checkLength(n)
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 16, 1), zeroed: false, context: ctx)
        if n > 0 {
            let fn = type.bitWidth == 32 ? "dec32_widen" : "dec64_widen"
            let pso = try ctx.pipeline(source: TypesExtraSource.smallDecimal, function: fn, cacheKey: "smalldec/\(fn)")
            let src = values
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(src.mtl, offset: src.offset, index: 0)
                Dispatch.setLength(enc, n, nil, index: 1)
                enc.setBuffer(out.mtl, offset: out.offset, index: 2)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        return MetalDecimalArray(type: try type.widened(), length: n, nullCount: nullCount,
                                 validity: validity, values: out, context: ctx)
    }

    public func filter(_ mask: MetalBooleanArray) throws -> MetalSmallDecimalArray {
        switch storage {
        case .int32(let a): return try MetalSmallDecimalArray(type: type, try a.filter(mask))
        case .int64(let a): return try MetalSmallDecimalArray(type: type, try a.filter(mask))
        }
    }
    public func take<I: ArrowIndex>(_ indices: MetalArray<I>) throws -> MetalSmallDecimalArray {
        switch storage {
        case .int32(let a): return try MetalSmallDecimalArray(type: type, try a.take(indices))
        case .int64(let a): return try MetalSmallDecimalArray(type: type, try a.take(indices))
        }
    }
    public func slice(offset: Int, length n: Int) throws -> MetalSmallDecimalArray {
        switch storage {
        case .int32(let a): return try MetalSmallDecimalArray(type: type, try a.slice(offset: offset, length: n))
        case .int64(let a): return try MetalSmallDecimalArray(type: type, try a.slice(offset: offset, length: n))
        }
    }
}

extension MetalDecimalArray {
    /// GPU narrowing cast back to `decimal32` / `decimal64`. The low 32 or 64 bits are kept, so a value
    /// that does not fit wraps — Arrow's unchecked cast. The scale is carried across unchanged.
    public func narrowed(to target: ArrowSmallDecimalType) throws -> MetalSmallDecimalArray {
        guard type.bitWidth == 128 else {
            throw ArrowMetalError.unsupportedType("narrowing to \(target) needs a decimal128 column, got \(type)")
        }
        guard target.scale == type.scale else {
            throw ArrowMetalError.unsupportedType("narrowing \(type) to \(target) would change the scale; rescale first")
        }
        let ctx = context
        let n = length
        try Dispatch.checkLength(n)
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * target.byteWidth, 1), zeroed: false, context: ctx)
        if n > 0 {
            let fn = target.bitWidth == 32 ? "dec32_narrow" : "dec64_narrow"
            let pso = try ctx.pipeline(source: TypesExtraSource.smallDecimal, function: fn, cacheKey: "smalldec/\(fn)")
            let src = values
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(src.mtl, offset: src.offset, index: 0)
                Dispatch.setLength(enc, n, nil, index: 1)
                enc.setBuffer(out.mtl, offset: out.offset, index: 2)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        if target.bitWidth == 32 {
            return try MetalSmallDecimalArray(type: target, MetalArray<Int32>(length: n, nullCount: nullCount,
                                                                              validity: validity, values: out, context: ctx))
        }
        return try MetalSmallDecimalArray(type: target, MetalArray<Int64>(length: n, nullCount: nullCount,
                                                                          validity: validity, values: out, context: ctx))
    }
}

// MARK: - interval

/// The three Arrow interval layouts.
public enum ArrowIntervalUnit: String, Sendable, CaseIterable {
    /// `interval[month]` ("tiM"): one int32 month count.
    case months = "tiM"
    /// `interval[day_time]` ("tiD"): two int32s, days then milliseconds.
    case dayTime = "tiD"
    /// `interval[month_day_nano]` ("tin"): int32 months, int32 days, int64 nanoseconds.
    case monthDayNano = "tin"

    public var arrowFormat: String { rawValue }
    public var byteWidth: Int {
        switch self {
        case .months: return 4
        case .dayTime: return 8
        case .monthDayNano: return 16
        }
    }
    /// The kernel's `ivUnit` code.
    var code: Int {
        switch self {
        case .months: return 0
        case .dayTime: return 1
        case .monthDayNano: return 2
        }
    }
}

/// One interval value in the widest form (a month interval has days = nanoseconds = 0, a day_time
/// interval has months = 0 and its milliseconds scaled to nanoseconds).
public struct ArrowInterval: Hashable, Sendable {
    public var months: Int32
    public var days: Int32
    public var nanoseconds: Int64
    public init(months: Int32 = 0, days: Int32 = 0, nanoseconds: Int64 = 0) {
        self.months = months; self.days = days; self.nanoseconds = nanoseconds
    }
}

/// An Arrow interval array: a raw fixed-width values buffer (4, 8 or 16 bytes per element) plus a
/// validity bitmap, exactly the Arrow layout, in Metal shared memory.
///
/// Import, export and `filter` / `take` / `slice` (one GPU gather over the records), plus
/// `MetalTemporalArray.addInterval(_:)` for the arithmetic.
public final class MetalIntervalArray: @unchecked Sendable {
    public let unit: ArrowIntervalUnit
    public let length: Int
    public internal(set) var nullCount: Int
    public let validity: MetalArrowBuffer?
    public let values: MetalArrowBuffer
    public let context: MetalContext

    public init(unit: ArrowIntervalUnit, length: Int, nullCount: Int, validity: MetalArrowBuffer?,
                values: MetalArrowBuffer, context: MetalContext = .shared) {
        precondition(values.byteCount >= length * unit.byteWidth)
        if let v = validity { precondition(v.byteCount >= Bitmap.byteCount(bits: length)) }
        self.unit = unit; self.length = length; self.nullCount = nullCount
        self.validity = validity; self.values = values; self.context = context
    }

    /// Builds from Swift values. A month interval keeps `months`, a day_time interval keeps `days` and
    /// `nanoseconds / 1_000_000`, and month_day_nano keeps all three.
    public convenience init(unit: ArrowIntervalUnit, _ vals: [ArrowInterval?], context: MetalContext = .shared) throws {
        let n = vals.count, w = unit.byteWidth
        let vb = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * w, 1), context: context)
        let bm = vals.contains(where: { $0 == nil })
            ? try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 1), context: context) : nil
        let raw = vb.mutableTyped(UInt8.self)
        var nulls = 0
        for (i, v) in vals.enumerated() {
            guard let v else { nulls += 1; continue }
            if let bm { Bitmap.set(bm.mutableTyped(UInt8.self), i) }
            let base = raw + i * w
            switch unit {
            case .months:
                base.withMemoryRebound(to: Int32.self, capacity: 1) { $0[0] = v.months }
            case .dayTime:
                base.withMemoryRebound(to: Int32.self, capacity: 2) { $0[0] = v.days; $0[1] = Int32(truncatingIfNeeded: v.nanoseconds / 1_000_000) }
            case .monthDayNano:
                base.withMemoryRebound(to: Int32.self, capacity: 2) { $0[0] = v.months; $0[1] = v.days }
                (base + 8).withMemoryRebound(to: Int64.self, capacity: 1) { $0[0] = v.nanoseconds }
            }
        }
        self.init(unit: unit, length: n, nullCount: nulls, validity: nulls == 0 ? nil : bm, values: vb, context: context)
    }

    public var arrowFormat: String { unit.arrowFormat }
    public func isValid(_ i: Int) -> Bool { validity.map { Bitmap.isSet($0.typed(UInt8.self), i) } ?? true }
    func setNullCount(_ n: Int) { nullCount = n }
    func recomputeNullCount() {
        guard let v = validity else { nullCount = 0; return }
        nullCount = length - Bitmap.popcount(v.typed(UInt8.self), bits: length)
    }

    /// Element `i` in the widest form, or nil when it is null.
    public subscript(i: Int) -> ArrowInterval? {
        guard isValid(i) else { return nil }
        return withExtendedLifetime(values) {
            let raw = values.typed(UInt8.self) + i * unit.byteWidth
            switch unit {
            case .months:
                return ArrowInterval(months: raw.withMemoryRebound(to: Int32.self, capacity: 1) { $0[0] })
            case .dayTime:
                let p = raw.withMemoryRebound(to: Int32.self, capacity: 2) { ($0[0], $0[1]) }
                return ArrowInterval(days: p.0, nanoseconds: Int64(p.1) * 1_000_000)
            case .monthDayNano:
                let p = raw.withMemoryRebound(to: Int32.self, capacity: 2) { ($0[0], $0[1]) }
                let ns = (raw + 8).withMemoryRebound(to: Int64.self, capacity: 1) { $0[0] }
                return ArrowInterval(months: p.0, days: p.1, nanoseconds: ns)
            }
        }
    }
    public func toArray() -> [ArrowInterval?] { (0..<length).map { self[$0] } }

    public func take<I: ArrowIndex>(_ indices: MetalArray<I>) throws -> MetalIntervalArray {
        let idx32: MetalArray<Int32> = try (indices as? MetalArray<Int32>) ?? indices.cast(to: Int32.self)
        let r = try FixedWidth.take(context, values: values, validity: validity, length: length,
                                    byteWidth: unit.byteWidth, indices: idx32)
        let out = MetalIntervalArray(unit: unit, length: r.length, nullCount: 0, validity: r.validity,
                                     values: r.values, context: context)
        out.recomputeNullCount()
        return out
    }
    public func filter(_ mask: MetalBooleanArray) throws -> MetalIntervalArray {
        guard mask.length == length else { throw ArrowMetalError.lengthMismatch(length, mask.length) }
        return try take(try MetalArray<Int32>.iota(length, context: context).filter(mask))
    }
    public func slice(offset: Int, length n: Int) throws -> MetalIntervalArray {
        guard offset >= 0, n >= 0, offset + n <= length else {
            throw ArrowMetalError.invalidArrowArray("interval slice \(offset)..<\(offset + n) is out of range (length \(length))")
        }
        if offset % 32 == 0 {
            let v = values.view(byteOffset: offset * unit.byteWidth, byteCount: n * unit.byteWidth)
            let bm = try FixedWidth.sliceBitmap(validity, offset: offset, length: n, context)
            let out = MetalIntervalArray(unit: unit, length: n, nullCount: 0, validity: bm, values: v, context: context)
            out.recomputeNullCount()
            return out
        }
        return try take(try NestedSupport.iota(offset, n, context))
    }
}

extension MetalTemporalArray {
    /// Arrow `add(timestamp | date, interval)` on the GPU.
    ///
    /// Month arithmetic goes through the civil calendar and clamps the day to the target month's length
    /// (2024-01-31 + 1 month = 2024-02-29), which is what Arrow does. Days are whole calendar days in UTC
    /// — this package has no timezone-aware arithmetic — and the sub-day part of the interval is converted
    /// to the column's own resolution, truncating toward zero when the column is coarser (a
    /// month_day_nano interval on a `timestamp[s]` column drops the nanoseconds below a second).
    ///
    /// `date32` counts whole days, so an interval with a non-zero sub-day part is rejected there rather
    /// than silently dropped. The interval array must have the receiver's length, or length 1 to broadcast.
    public func addInterval(_ iv: MetalIntervalArray) throws -> MetalTemporalArray {
        let n = length
        let broadcast = iv.length == 1 && n != 1
        guard broadcast || iv.length == n else { throw ArrowMetalError.lengthMismatch(n, iv.length) }
        let ctx = context

        // Ticks per day and the multiplier that takes the interval's sub-day field into the column's unit.
        let ticksPerDay: Int64
        let mode: Int
        var subNum: Int64 = 1, subDen: Int64 = 1
        switch type {
        case .date32:
            ticksPerDay = 1; mode = 0
        case .date64:
            ticksPerDay = 86_400_000; mode = 1
        case .timestamp(let u, _):
            ticksPerDay = 86_400 * u.perSecond; mode = 1
        case .time32, .time64, .duration:
            throw ArrowMetalError.unsupportedType("add_interval needs a date or timestamp column, got \(type.arrowFormat)")
        }
        if mode == 1 {
            // The interval's sub-day field is milliseconds for day_time and nanoseconds for month_day_nano.
            let perSecond: Int64
            if case .timestamp(let u, _) = type { perSecond = u.perSecond } else { perSecond = 1_000 }
            let fieldPerSecond: Int64 = iv.unit == .dayTime ? 1_000 : 1_000_000_000
            if perSecond >= fieldPerSecond { subNum = perSecond / fieldPerSecond; subDen = 1 }
            else { subNum = 1; subDen = fieldPerSecond / perSecond }
        } else if iv.unit != .months {
            // date32 has no sub-day resolution; only reject when the interval actually carries one.
            let carries = (0..<iv.length).contains { iv[$0].map { $0.nanoseconds != 0 } ?? false }
            if carries {
                throw ArrowMetalError.unsupportedType("add_interval on date32 cannot apply the interval's sub-day part")
            }
        }

        let wide = try int64Values()
        try Dispatch.checkLength(n)
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 8, 1), zeroed: false, context: ctx)
        if n > 0 {
            let pso = try ctx.pipeline(source: TypesExtraSource.interval, function: "temporal_add_interval",
                                       cacheKey: "interval/add")
            let ivBuf = iv.values
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(wide.values.mtl, offset: wide.values.offset, index: 0)
                enc.setBuffer(ivBuf.mtl, offset: ivBuf.offset, index: 1)
                enc.setBuffer(ivBuf.mtl, offset: ivBuf.offset, index: 2)
                Dispatch.setLength(enc, n, nil, index: 3)
                Dispatch.setUInt(enc, iv.unit.code, index: 4)
                Dispatch.setUInt(enc, broadcast ? 1 : 0, index: 5)
                var tpd = ticksPerDay; enc.setBytes(&tpd, length: 8, index: 6)
                var sn = subNum; enc.setBytes(&sn, length: 8, index: 7)
                var sd = subDen; enc.setBytes(&sd, length: 8, index: 8)
                Dispatch.setUInt(enc, mode, index: 9)
                enc.setBuffer(out.mtl, offset: out.offset, index: 10)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        // The result is null wherever either side is.
        let ivValidity = broadcast ? (iv.nullCount > 0 ? try allNullBitmap(n, ctx) : nil) : iv.validity
        let combined = try BitmapOps.combineValidity(ctx, validity, ivValidity, bits: Swift.max(n, 1))
        let wideOut = MetalArray<Int64>(length: n, nullCount: 0, validity: combined, values: out, context: ctx)
        wideOut.recomputeNullCount()
        if type.usesInt64 { return try MetalTemporalArray(type: type, wideOut) }
        return try MetalTemporalArray(type: type, try wideOut.cast(to: Int32.self))
    }

    private func allNullBitmap(_ n: Int, _ ctx: MetalContext) throws -> MetalArrowBuffer {
        try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 1), context: ctx)
    }
}

// MARK: - fixed_size_binary

/// An Arrow `fixed_size_binary` ("w:N") array: `byteWidth` bytes per element in one values buffer.
///
/// GPU: `filter` / `take` / `slice` through the shared record gather, `equal` / `not_equal` against a
/// scalar and between arrays (byte compare), and an FNV-1a `hash64`.
public final class MetalFixedBinaryArray: @unchecked Sendable {
    public let byteWidth: Int
    public let length: Int
    public internal(set) var nullCount: Int
    public let validity: MetalArrowBuffer?
    public let values: MetalArrowBuffer
    public let context: MetalContext

    public init(byteWidth: Int, length: Int, nullCount: Int, validity: MetalArrowBuffer?,
                values: MetalArrowBuffer, context: MetalContext = .shared) {
        precondition(byteWidth >= 0)
        precondition(values.byteCount >= length * byteWidth)
        if let v = validity { precondition(v.byteCount >= Bitmap.byteCount(bits: length)) }
        self.byteWidth = byteWidth; self.length = length; self.nullCount = nullCount
        self.validity = validity; self.values = values; self.context = context
    }

    /// Builds from byte strings, each of which must be exactly `byteWidth` long.
    public convenience init(byteWidth: Int, _ vals: [[UInt8]?], context: MetalContext = .shared) throws {
        let n = vals.count
        for v in vals where v != nil && v!.count != byteWidth {
            throw ArrowMetalError.invalidArrowArray("fixed_size_binary<\(byteWidth)> needs \(byteWidth) bytes per value")
        }
        let vb = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * byteWidth, 1), context: context)
        let bm = vals.contains(where: { $0 == nil })
            ? try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 1), context: context) : nil
        let p = vb.mutableTyped(UInt8.self)
        var nulls = 0
        for (i, v) in vals.enumerated() {
            guard let v else { nulls += 1; continue }
            for (k, b) in v.enumerated() { p[i * byteWidth + k] = b }
            if let bm { Bitmap.set(bm.mutableTyped(UInt8.self), i) }
        }
        self.init(byteWidth: byteWidth, length: n, nullCount: nulls, validity: nulls == 0 ? nil : bm,
                  values: vb, context: context)
    }

    public var arrowFormat: String { "w:\(byteWidth)" }
    public func isValid(_ i: Int) -> Bool { validity.map { Bitmap.isSet($0.typed(UInt8.self), i) } ?? true }
    func setNullCount(_ n: Int) { nullCount = n }
    func recomputeNullCount() {
        guard let v = validity else { nullCount = 0; return }
        nullCount = length - Bitmap.popcount(v.typed(UInt8.self), bits: length)
    }

    /// The bytes of element `i`, or nil when it is null.
    public func bytes(at i: Int) -> [UInt8]? {
        guard isValid(i) else { return nil }
        return withExtendedLifetime(values) {
            let p = values.typed(UInt8.self) + i * byteWidth
            return Array(UnsafeBufferPointer(start: p, count: byteWidth))
        }
    }
    public func toByteArrays() -> [[UInt8]?] { (0..<length).map { bytes(at: $0) } }

    /// Arrow `equal` / `not_equal` against one scalar record. Null in, null out.
    public func compare(_ op: CompareOp, _ scalar: [UInt8]) throws -> MetalBooleanArray {
        guard op == .eq || op == .ne else {
            throw ArrowMetalError.unsupportedType("fixed_size_binary supports equal and not_equal only, got \(op)")
        }
        guard scalar.count == byteWidth else {
            throw ArrowMetalError.invalidArrowArray("scalar has \(scalar.count) bytes, expected \(byteWidth)")
        }
        let bits = try FixedWidth.compareScalar(context, values: values, length: length, byteWidth: byteWidth,
                                                scalar: scalar, notEqual: op == .ne)
        let out = MetalBooleanArray(length: length, nullCount: 0, validity: validity, values: bits, context: context)
        out.recomputeNullCount()
        return out
    }

    /// Arrow `equal` / `not_equal` between two arrays of the same width and length.
    public func compare(_ op: CompareOp, _ other: MetalFixedBinaryArray) throws -> MetalBooleanArray {
        guard op == .eq || op == .ne else {
            throw ArrowMetalError.unsupportedType("fixed_size_binary supports equal and not_equal only, got \(op)")
        }
        guard other.byteWidth == byteWidth else {
            throw ArrowMetalError.unsupportedType("cannot compare w:\(byteWidth) with w:\(other.byteWidth)")
        }
        guard other.length == length else { throw ArrowMetalError.lengthMismatch(length, other.length) }
        let bits = try FixedWidth.compareArray(context, values, other.values, length: length,
                                               byteWidth: byteWidth, notEqual: op == .ne)
        let v = try BitmapOps.combineValidity(context, validity, other.validity, bits: Swift.max(length, 1))
        let out = MetalBooleanArray(length: length, nullCount: 0, validity: v, values: bits, context: context)
        out.recomputeNullCount()
        return out
    }

    /// FNV-1a 64 over each element's bytes (an ArrowMetal extension, not Arrow's `hash64`). Null in, null out.
    public func hash64() throws -> MetalArray<UInt64> {
        let out = try FixedWidth.hash64(context, values: values, length: length, byteWidth: byteWidth)
        return MetalArray<UInt64>(length: length, nullCount: nullCount, validity: validity, values: out, context: context)
    }

    public func take<I: ArrowIndex>(_ indices: MetalArray<I>) throws -> MetalFixedBinaryArray {
        let idx32: MetalArray<Int32> = try (indices as? MetalArray<Int32>) ?? indices.cast(to: Int32.self)
        let r = try FixedWidth.take(context, values: values, validity: validity, length: length,
                                    byteWidth: byteWidth, indices: idx32)
        let out = MetalFixedBinaryArray(byteWidth: byteWidth, length: r.length, nullCount: 0,
                                        validity: r.validity, values: r.values, context: context)
        out.recomputeNullCount()
        return out
    }
    public func filter(_ mask: MetalBooleanArray) throws -> MetalFixedBinaryArray {
        guard mask.length == length else { throw ArrowMetalError.lengthMismatch(length, mask.length) }
        return try take(try MetalArray<Int32>.iota(length, context: context).filter(mask))
    }
    public func slice(offset: Int, length n: Int) throws -> MetalFixedBinaryArray {
        guard offset >= 0, n >= 0, offset + n <= length else {
            throw ArrowMetalError.invalidArrowArray("fixed_size_binary slice \(offset)..<\(offset + n) is out of range (length \(length))")
        }
        if offset % 32 == 0 {
            let v = values.view(byteOffset: offset * byteWidth, byteCount: n * byteWidth)
            let bm = try FixedWidth.sliceBitmap(validity, offset: offset, length: n, context)
            let out = MetalFixedBinaryArray(byteWidth: byteWidth, length: n, nullCount: 0, validity: bm,
                                            values: v, context: context)
            out.recomputeNullCount()
            return out
        }
        return try take(try NestedSupport.iota(offset, n, context))
    }
}

// MARK: - AnyMetalArray accessors

extension AnyMetalArray {
    public var asNull: MetalNullArray? { if case .null(let a) = self { return a } else { return nil } }
    public var asFloat16: MetalFloat16Array? { if case .float16(let a) = self { return a } else { return nil } }
    public var asSmallDecimal: MetalSmallDecimalArray? { if case .smallDecimal(let a) = self { return a } else { return nil } }
    public var asInterval: MetalIntervalArray? { if case .interval(let a) = self { return a } else { return nil } }
    public var asFixedBinary: MetalFixedBinaryArray? { if case .fixedBinary(let a) = self { return a } else { return nil } }
}

// MARK: - Import

/// The C Data Interface format strings this file imports.
func isExtraTypeFormat(_ f: String) -> Bool {
    if f == "n" || f == "e" { return true }
    if f == "tiM" || f == "tiD" || f == "tin" { return true }
    if f.hasPrefix("w:") { return true }
    if ArrowSmallDecimalType(format: f) != nil { return true }
    return false
}

/// Imports one of the types this file owns, or returns nil when `format` is not one of them.
///
/// `float16` and the narrow decimals reuse the primitive importer (so they keep its zero-copy path and
/// offset handling) by re-describing the buffers with the matching integer format; the interval and
/// fixed_size_binary layouts are raw records and go through the fixed-width importer below.
func importExtraTypeArray(format fmt: String, array: UnsafeMutablePointer<ArrowArray>,
                          context: MetalContext) throws -> ImportResult? {
    if fmt == "n" { return try importNullArray(array: array, context: context) }
    if fmt == "e" {
        let r = try importAsPrimitiveFormat("S", array: array, context: context)
        guard case .uint16(let bits) = r.array else { throw ArrowMetalError.invalidArrowArray("float16 values must be 2 bytes") }
        return ImportResult(array: .float16(MetalFloat16Array(bits: bits)), zeroCopy: r.zeroCopy)
    }
    if let t = ArrowSmallDecimalType(format: fmt) {
        let r = try importAsPrimitiveFormat(t.bitWidth == 32 ? "i" : "l", array: array, context: context)
        switch r.array {
        case .int32(let a): return ImportResult(array: .smallDecimal(try MetalSmallDecimalArray(type: t, a)), zeroCopy: r.zeroCopy)
        case .int64(let a): return ImportResult(array: .smallDecimal(try MetalSmallDecimalArray(type: t, a)), zeroCopy: r.zeroCopy)
        default: throw ArrowMetalError.invalidArrowArray("\(t) values must be int32 or int64")
        }
    }
    if let unit = ArrowIntervalUnit(rawValue: fmt) {
        let b = try importFixedWidthBuffers(byteWidth: unit.byteWidth, format: fmt, array: array, context: context)
        let arr = MetalIntervalArray(unit: unit, length: b.length, nullCount: 0, validity: b.validity,
                                     values: b.values, context: context)
        if b.declaredNulls < 0 || b.rebased { arr.recomputeNullCount() } else { arr.setNullCount(b.declaredNulls) }
        return ImportResult(array: .interval(arr), zeroCopy: b.zeroCopy)
    }
    if fmt.hasPrefix("w:") {
        guard let w = Int(fmt.dropFirst(2)), w >= 0 else { throw ArrowMetalError.unsupportedType(fmt) }
        let b = try importFixedWidthBuffers(byteWidth: w, format: fmt, array: array, context: context)
        let arr = MetalFixedBinaryArray(byteWidth: w, length: b.length, nullCount: 0, validity: b.validity,
                                        values: b.values, context: context)
        if b.declaredNulls < 0 || b.rebased { arr.recomputeNullCount() } else { arr.setNullCount(b.declaredNulls) }
        return ImportResult(array: .fixedBinary(arr), zeroCopy: b.zeroCopy)
    }
    return nil
}

/// Imports the array through the primitive path by describing it with `format` (which must be a
/// primitive format of the same width). Keeps the importer's zero-copy and offset handling.
func importAsPrimitiveFormat(_ format: String, array: UnsafeMutablePointer<ArrowArray>,
                             context: MetalContext) throws -> ImportResult {
    let fmt = strdup(format)!
    defer { free(fmt) }
    let schema = UnsafeMutablePointer<ArrowSchema>.allocate(capacity: 1)
    schema.initialize(to: ArrowSchema())
    defer { schema.deallocate() }
    schema.pointee.format = UnsafePointer(fmt)
    return try importArrowArray(schema: schema, array: array, context: context)
}

/// A `null` array carries a length and no buffers. Producers differ on whether they still declare a
/// (null) validity buffer, so 0 and 1 buffers are both accepted.
private func importNullArray(array: UnsafeMutablePointer<ArrowArray>, context: MetalContext) throws -> ImportResult {
    guard array.pointee.n_buffers <= 1 else {
        throw ArrowMetalError.invalidArrowArray("a null array has no buffers, got \(array.pointee.n_buffers)")
    }
    let owner = ImportedCArray(moving: array)
    let length = Int(owner.array.length)
    withExtendedLifetime(owner) {}
    return ImportResult(array: .null(MetalNullArray(length: length, context: context)), zeroCopy: true)
}

/// Two-buffer import of an opaque fixed-width layout: validity plus `byteWidth` bytes per element.
/// Zero-copy when the producer's buffers are page aligned and the array has no offset.
func importFixedWidthBuffers(byteWidth: Int, format: String, array: UnsafeMutablePointer<ArrowArray>,
                             context: MetalContext)
    throws -> (length: Int, validity: MetalArrowBuffer?, values: MetalArrowBuffer, zeroCopy: Bool,
               declaredNulls: Int, rebased: Bool) {
    guard array.pointee.n_buffers == 2, array.pointee.buffers != nil else {
        throw ArrowMetalError.invalidArrowArray("expected 2 buffers for \(format), got \(array.pointee.n_buffers)")
    }
    let owner = ImportedCArray(moving: array)
    let a = owner.array
    let length = Int(a.length), offset = Int(a.offset)
    let validityPtr = a.buffers[0].map { UnsafeRawPointer($0) }
    guard let valuesPtr = a.buffers[1].map({ UnsafeRawPointer($0) }) else {
        throw ArrowMetalError.invalidArrowArray("values buffer is null")
    }
    let (vals, zc1) = try MetalArrowBuffer.wrapOrCopy(valuesPtr.advanced(by: offset * byteWidth),
                                                      byteCount: Swift.max(length * byteWidth, 1),
                                                      keepAlive: owner, context: context)
    var validity: MetalArrowBuffer? = nil
    var zc2 = true
    if let vp = validityPtr {
        let bytes = Swift.max(Bitmap.byteCount(bits: length), 1)
        if offset == 0 {
            (validity, zc2) = try MetalArrowBuffer.wrapOrCopy(vp, byteCount: bytes, keepAlive: owner, context: context)
        } else {
            zc2 = false
            let buf = try MetalArrowBuffer.allocate(byteCount: bytes, context: context)
            let sp = vp.assumingMemoryBound(to: UInt8.self)
            let dp = buf.mutableTyped(UInt8.self)
            if offset % 8 == 0 { memcpy(dp, sp + offset / 8, bytes) }
            else { for i in 0..<length where Bitmap.isSet(sp, i + offset) { Bitmap.set(dp, i) } }
            validity = buf
        }
    }
    return (length, validity, vals, zc1 && zc2, Int(a.null_count), offset != 0)
}

// MARK: - Export

/// Holder for an exported `null` array: no buffers, nothing to keep alive but the buffer table.
private final class NullExportHolder {
    let buffers: UnsafeMutablePointer<UnsafeRawPointer?>
    init() { buffers = .allocate(capacity: 1); buffers[0] = nil }
    deinit { buffers.deallocate() }
}

private func releaseNullArray(_ p: UnsafeMutablePointer<ArrowArray>?) {
    guard let p = p, let pd = p.pointee.private_data else { return }
    Unmanaged<NullExportHolder>.fromOpaque(pd).release()
    p.pointee.release = nil; p.pointee.private_data = nil
}

extension MetalNullArray {
    /// A `null` array exports with zero buffers, which is what the Arrow spec prescribes for the type.
    public func exportArrowArray(into out: UnsafeMutablePointer<ArrowArray>) {
        let h = NullExportHolder()
        out.pointee.length = Int64(length)
        out.pointee.null_count = Int64(length)
        out.pointee.offset = 0
        out.pointee.n_buffers = 0
        out.pointee.n_children = 0
        out.pointee.buffers = UnsafeMutablePointer<UnsafeRawPointer?>(h.buffers)
        out.pointee.children = nil
        out.pointee.dictionary = nil
        out.pointee.release = releaseNullArray
        out.pointee.private_data = Unmanaged.passRetained(h).toOpaque()
    }
    public func exportArrowSchema(name: String = "", into out: UnsafeMutablePointer<ArrowSchema>) {
        ArrowMetal.exportArrowSchema(format: "n", name: name, into: out)
    }
    public func exportArrowDeviceArray(into out: UnsafeMutablePointer<ArrowDeviceArray>) {
        withUnsafeMutablePointer(to: &out.pointee.array) { exportArrowArray(into: $0) }
        out.pointee.device_type = ARROW_DEVICE_METAL
        out.pointee.device_id = -1
        out.pointee.reserved = (0, 0, 0)
        out.pointee.sync_event = nil
    }
}

extension MetalFloat16Array {
    public func exportArrowArray(into out: UnsafeMutablePointer<ArrowArray>) { bits.exportArrowArray(into: out) }
    public func exportArrowSchema(name: String = "", into out: UnsafeMutablePointer<ArrowSchema>) {
        ArrowMetal.exportArrowSchema(format: "e", name: name, into: out)
    }
    public func exportArrowDeviceArray(into out: UnsafeMutablePointer<ArrowDeviceArray>) {
        bits.exportArrowDeviceArray(into: out)
    }
}

extension MetalSmallDecimalArray {
    public func exportArrowArray(into out: UnsafeMutablePointer<ArrowArray>) {
        switch storage {
        case .int32(let a): a.exportArrowArray(into: out)
        case .int64(let a): a.exportArrowArray(into: out)
        }
    }
    public func exportArrowSchema(name: String = "", into out: UnsafeMutablePointer<ArrowSchema>) {
        ArrowMetal.exportArrowSchema(format: type.arrowFormat, name: name, into: out)
    }
    public func exportArrowDeviceArray(into out: UnsafeMutablePointer<ArrowDeviceArray>) {
        switch storage {
        case .int32(let a): a.exportArrowDeviceArray(into: out)
        case .int64(let a): a.exportArrowDeviceArray(into: out)
        }
    }
}

extension MetalIntervalArray {
    public func exportArrowArray(into out: UnsafeMutablePointer<ArrowArray>) {
        // Same two-buffer shape as a primitive array; the proxy describes the records as bytes.
        let proxy = MetalArray<UInt8>(length: length, nullCount: nullCount, validity: validity,
                                      values: values, context: context)
        proxy.exportArrowArray(into: out)
    }
    public func exportArrowSchema(name: String = "", into out: UnsafeMutablePointer<ArrowSchema>) {
        ArrowMetal.exportArrowSchema(format: unit.arrowFormat, name: name, into: out)
    }
    public func exportArrowDeviceArray(into out: UnsafeMutablePointer<ArrowDeviceArray>) {
        withUnsafeMutablePointer(to: &out.pointee.array) { exportArrowArray(into: $0) }
        out.pointee.device_type = ARROW_DEVICE_METAL
        out.pointee.device_id = -1
        out.pointee.reserved = (0, 0, 0)
        out.pointee.sync_event = nil
    }
}

extension MetalFixedBinaryArray {
    public func exportArrowArray(into out: UnsafeMutablePointer<ArrowArray>) {
        let proxy = MetalArray<UInt8>(length: length, nullCount: nullCount, validity: validity,
                                      values: values, context: context)
        proxy.exportArrowArray(into: out)
    }
    public func exportArrowSchema(name: String = "", into out: UnsafeMutablePointer<ArrowSchema>) {
        ArrowMetal.exportArrowSchema(format: arrowFormat, name: name, into: out)
    }
    public func exportArrowDeviceArray(into out: UnsafeMutablePointer<ArrowDeviceArray>) {
        withUnsafeMutablePointer(to: &out.pointee.array) { exportArrowArray(into: $0) }
        out.pointee.device_type = ARROW_DEVICE_METAL
        out.pointee.device_id = -1
        out.pointee.reserved = (0, 0, 0)
        out.pointee.sync_event = nil
    }
}

// MARK: - Importer hook

/// The one hook the C Data Interface importer calls before its own dispatch: the type-matrix types this
/// file owns (`n`, `e`, `d:p,s,32`, `d:p,s,64`, `tiM`, `tiD`, `tin`, `w:N`) plus the two list-view
/// layouts from `NestedExtra.swift`. Returns nil when `format` is none of them, so the existing dispatch
/// runs unchanged.
func importExtraFormats(format fmt: String, schema: UnsafePointer<ArrowSchema>,
                        array: UnsafeMutablePointer<ArrowArray>, context: MetalContext) throws -> ImportResult? {
    if isListViewFormat(fmt) {
        return try importListViewArray(format: fmt, schema: schema, array: array, context: context)
    }
    return try importExtraTypeArray(format: fmt, array: array, context: context)
}
