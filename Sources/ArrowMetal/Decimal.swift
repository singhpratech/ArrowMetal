import Foundation
import Metal
import CArrowABI

// Arrow decimal types. A decimal column is a fixed-width integer column: each element is a two's-complement
// little-endian integer of 16 bytes (decimal128) or 32 bytes (decimal256) holding the *unscaled* value, and
// the type's `scale` says where the decimal point sits. Metal has no 128-bit integer, so every kernel works
// on 64-bit limbs (`Kernels/DecimalSource.swift`); this file owns the type metadata, the buffers, the C Data
// Interface glue and the host halves of the reductions and casts.

/// An Arrow decimal type together with its C Data Interface format string (`d:precision,scale` for
/// decimal128, `d:precision,scale,256` for decimal256).
public struct ArrowDecimalType: Hashable, Sendable, CustomStringConvertible {
    public let precision: Int
    public let scale: Int
    /// 128 or 256.
    public let bitWidth: Int

    /// Largest precision Arrow allows for a given width.
    public static func maxPrecision(bitWidth: Int) -> Int { bitWidth == 256 ? 76 : 38 }

    public init(precision: Int, scale: Int, bitWidth: Int = 128) throws {
        guard bitWidth == 128 || bitWidth == 256 else {
            throw ArrowMetalError.unsupportedType("decimal\(bitWidth): only decimal128 and decimal256 are supported")
        }
        let maxP = Self.maxPrecision(bitWidth: bitWidth)
        guard precision >= 1, precision <= maxP else {
            throw ArrowMetalError.unsupportedType("decimal\(bitWidth) precision \(precision) is outside 1...\(maxP)")
        }
        guard scale >= -maxP, scale <= maxP else {
            throw ArrowMetalError.unsupportedType("decimal\(bitWidth) scale \(scale) is outside -\(maxP)...\(maxP)")
        }
        self.precision = precision
        self.scale = scale
        self.bitWidth = bitWidth
    }

    /// Parses `d:precision,scale` or `d:precision,scale,bitWidth`; nil when the string is not a decimal format.
    public init?(format f: String) {
        guard f.hasPrefix("d:") else { return nil }
        let parts = f.dropFirst(2).split(separator: ",", omittingEmptySubsequences: false).map(String.init)
        guard parts.count == 2 || parts.count == 3,
              let p = Int(parts[0]), let s = Int(parts[1]) else { return nil }
        let bits = parts.count == 3 ? Int(parts[2]) : 128
        guard let bits else { return nil }
        guard let t = try? ArrowDecimalType(precision: p, scale: s, bitWidth: bits) else { return nil }
        self = t
    }

    /// Parses a format string or throws `unsupportedType` with the reason.
    public static func parse(_ f: String) throws -> ArrowDecimalType {
        guard f.hasPrefix("d:") else { throw ArrowMetalError.unsupportedType(f) }
        let parts = f.dropFirst(2).split(separator: ",", omittingEmptySubsequences: false).map(String.init)
        guard parts.count == 2 || parts.count == 3, let p = Int(parts[0]), let s = Int(parts[1]) else {
            throw ArrowMetalError.unsupportedType("\(f) is not a decimal format (expected d:precision,scale[,bitWidth])")
        }
        guard let bits = parts.count == 3 ? Int(parts[2]) : 128 else {
            throw ArrowMetalError.unsupportedType("\(f): bit width must be an integer")
        }
        return try ArrowDecimalType(precision: p, scale: s, bitWidth: bits)
    }

    /// `d:p,s` for decimal128 (Arrow's default width is implied), `d:p,s,256` for decimal256.
    public var arrowFormat: String { bitWidth == 128 ? "d:\(precision),\(scale)" : "d:\(precision),\(scale),\(bitWidth)" }
    public var byteWidth: Int { bitWidth / 8 }
    /// Number of 64-bit limbs per element (2 or 4).
    public var limbCount: Int { bitWidth / 64 }
    public var description: String { "decimal\(bitWidth)(\(precision), \(scale))" }
}

/// A signed 128-bit integer as two little-endian limbs: the unscaled value of one `decimal128` element.
///
/// Only the operations ArrowMetal needs on the host are here (the GPU has its own limb arithmetic in
/// `Kernels/DecimalSource.swift`): construction, wrapping add / subtract / negate, ordering, and the
/// conversions the CPU casts use. Every arithmetic operation wraps modulo 2^128, matching the kernels.
public struct ArrowDecimal128: Hashable, Comparable, Sendable, CustomStringConvertible {
    public var lo: UInt64
    public var hi: UInt64

    public init(lo: UInt64, hi: UInt64) { self.lo = lo; self.hi = hi }
    /// Sign-extends a 64-bit value.
    public init(_ v: Int64) { lo = UInt64(bitPattern: v); hi = v < 0 ? ~0 : 0 }
    public init(_ v: Int) { self.init(Int64(v)) }

    public static let zero = ArrowDecimal128(lo: 0, hi: 0)
    public var isNegative: Bool { Int64(bitPattern: hi) < 0 }
    public var isZero: Bool { lo == 0 && hi == 0 }

    public var negated: ArrowDecimal128 {
        let l = ~lo &+ 1
        let h = ~hi &+ (l == 0 ? 1 : 0)
        return ArrowDecimal128(lo: l, hi: h)
    }
    /// Absolute value (wrapping: the most negative value negates to itself).
    public var magnitude128: ArrowDecimal128 { isNegative ? negated : self }

    public static func + (a: ArrowDecimal128, b: ArrowDecimal128) -> ArrowDecimal128 {
        let l = a.lo &+ b.lo
        let carry: UInt64 = l < a.lo ? 1 : 0
        return ArrowDecimal128(lo: l, hi: a.hi &+ b.hi &+ carry)
    }
    public static func - (a: ArrowDecimal128, b: ArrowDecimal128) -> ArrowDecimal128 { a + b.negated }
    /// Low 128 bits of the product (wraps).
    public static func * (a: ArrowDecimal128, b: ArrowDecimal128) -> ArrowDecimal128 {
        let p = a.lo.multipliedFullWidth(by: b.lo)
        return ArrowDecimal128(lo: p.low, hi: p.high &+ (a.lo &* b.hi) &+ (a.hi &* b.lo))
    }
    public static func < (a: ArrowDecimal128, b: ArrowDecimal128) -> Bool {
        if a.hi != b.hi { return Int64(bitPattern: a.hi) < Int64(bitPattern: b.hi) }
        return a.lo < b.lo
    }

    /// Unsigned division of the magnitude by a 64-bit divisor.
    func divided(byUnsigned d: UInt64) -> (quotient: ArrowDecimal128, remainder: UInt64) {
        precondition(d != 0)
        let (qh, rh) = hi.quotientAndRemainder(dividingBy: d)
        let (ql, r) = d.dividingFullWidth((high: rh, low: lo))
        return (ArrowDecimal128(lo: ql, hi: qh), r)
    }

    /// The value as a Double, dividing by 10^`scale`. Precision is the Double's 53 bits.
    public func doubleValue(scale: Int) -> Double {
        let m = magnitude128
        var d = Double(m.hi) * 0x1p64 + Double(m.lo)
        if scale != 0 { d /= pow(10.0, Double(scale)) }
        return isNegative ? -d : d
    }

    /// Decimal digits of the unscaled value (no decimal point).
    public var description: String {
        if isZero { return "0" }
        var m = magnitude128
        var digits = ""
        while !m.isZero {
            let (q, r) = m.divided(byUnsigned: 10)
            digits.append(Character(UnicodeScalar(UInt8(48 + r))))
            m = q
        }
        return (isNegative ? "-" : "") + String(digits.reversed())
    }

    /// Little-endian two's-complement bytes.
    public var littleEndianBytes: [UInt8] {
        var out = [UInt8](repeating: 0, count: 16)
        for i in 0..<8 { out[i] = UInt8((lo >> (8 * UInt64(i))) & 0xFF) }
        for i in 0..<8 { out[8 + i] = UInt8((hi >> (8 * UInt64(i))) & 0xFF) }
        return out
    }
    /// Sign-extended limbs for a wider decimal.
    public func limbs(count: Int) -> [UInt64] {
        var l = [UInt64](repeating: isNegative ? ~0 : 0, count: count)
        if count >= 1 { l[0] = lo }
        if count >= 2 { l[1] = hi }
        return l
    }
}

/// Rounding mode for `MetalDecimalArray.rescaled(to:mode:)`.
public enum DecimalRoundMode: Int, Sendable, CaseIterable {
    /// Halves away from zero (Arrow's `HALF_UP`), matching this package's float `round`.
    case round = 0
    /// Toward +infinity.
    case ceil = 1
    /// Toward -infinity.
    case floor = 2
    /// Toward zero.
    case truncate = 3
}

/// An Arrow `decimal128` or `decimal256` array: a raw fixed-width values buffer plus an optional validity
/// bitmap, exactly the Arrow layout, in Metal shared memory.
///
/// GPU: compare (scalar and array), `sum` / `min` / `max`, `add` / `subtract` / `negate` / `abs` / `sign`,
/// multiply (by an integer scalar and element-wise), rescaling with `round` / `ceil` / `floor` / `truncate`,
/// `filter`, `take` and `slice`. CPU: the casts to and from `float64` / `int64`, and the final combine of the
/// reduction partials. decimal256 supports import/export, compare, `filter`, `take`, `slice` and `sum`;
/// the other kernels throw `unsupportedType` naming the operation.
public final class MetalDecimalArray: @unchecked Sendable {
    public let type: ArrowDecimalType
    public let length: Int
    public internal(set) var nullCount: Int
    public let validity: MetalArrowBuffer?
    public let values: MetalArrowBuffer
    public let context: MetalContext

    public init(type: ArrowDecimalType, length: Int, nullCount: Int, validity: MetalArrowBuffer?,
                values: MetalArrowBuffer, context: MetalContext = .shared) {
        precondition(values.byteCount >= length * type.byteWidth)
        if let v = validity { precondition(v.byteCount >= Bitmap.byteCount(bits: length)) }
        self.type = type
        self.length = length
        self.nullCount = nullCount
        self.validity = validity
        self.values = values
        self.context = context
    }

    /// Builds from unscaled 128-bit values (sign-extended for decimal256).
    public convenience init(type: ArrowDecimalType, _ vals: [ArrowDecimal128?], context: MetalContext = .shared) throws {
        let n = vals.count, w = type.limbCount
        let vb = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * type.byteWidth, 1), context: context)
        let bm = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 1), context: context)
        let vp = vb.mutableTyped(UInt64.self)
        let bp = bm.mutableTyped(UInt8.self)
        var nulls = 0
        for i in 0..<n {
            guard let v = vals[i] else { nulls += 1; continue }
            let limbs = v.limbs(count: w)
            for k in 0..<w { vp[i * w + k] = limbs[k] }
            Bitmap.set(bp, i)
        }
        self.init(type: type, length: n, nullCount: nulls, validity: nulls == 0 ? nil : bm, values: vb, context: context)
    }

    /// Builds from unscaled 64-bit values (a convenience for tests and small columns).
    public convenience init(type: ArrowDecimalType, unscaled vals: [Int64?], context: MetalContext = .shared) throws {
        try self.init(type: type, vals.map { $0.map(ArrowDecimal128.init) }, context: context)
    }

    /// Allocates an uninitialised array with `length` slots.
    public static func allocate(type: ArrowDecimalType, length: Int, withValidity: Bool,
                                context: MetalContext = .shared) throws -> MetalDecimalArray {
        let vb = try MetalArrowBuffer.allocate(byteCount: Swift.max(length * type.byteWidth, 1), context: context)
        let bm = withValidity ? try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: length), 1), context: context) : nil
        return MetalDecimalArray(type: type, length: length, nullCount: 0, validity: bm, values: vb, context: context)
    }

    public var arrowFormat: String { type.arrowFormat }
    public var precision: Int { type.precision }
    public var scale: Int { type.scale }
    public var validCount: Int { length - nullCount }

    /// Forces any open batch on this thread to run, so the buffers can be read on the CPU.
    @inline(__always) func ensure() { if context.isBatching { try? context.syncPoint() } }

    public func recomputeNullCount() {
        guard let v = validity else { nullCount = 0; return }
        nullCount = length - Bitmap.popcount(v.typed(UInt8.self), bits: length)
    }
    func setNullCount(_ n: Int) { nullCount = n }

    public func isValid(_ i: Int) -> Bool {
        ensure()
        guard let v = validity else { return true }
        return Bitmap.isSet(v.typed(UInt8.self), i)
    }

    /// The raw limbs of element `i`, least significant first (the null slot's bytes are returned as stored).
    public func limbs(at i: Int) -> [UInt64] {
        ensure()
        precondition(i >= 0 && i < length)
        let w = type.limbCount
        return withExtendedLifetime(values) {
            let p = values.typed(UInt64.self)
            return (0..<w).map { p[i * w + $0] }
        }
    }

    /// The raw little-endian two's-complement bytes of element `i` (16 or 32 of them).
    public func rawBytes(at i: Int) -> [UInt8] {
        ensure()
        precondition(i >= 0 && i < length)
        return withExtendedLifetime(values) {
            let p = values.typed(UInt8.self)
            return (0..<type.byteWidth).map { p[i * type.byteWidth + $0] }
        }
    }

    /// The unscaled value of element `i`, or nil when it is null. For decimal256 the value must fit 128 bits.
    public subscript(i: Int) -> ArrowDecimal128? {
        guard isValid(i) else { return nil }
        let l = limbs(at: i)
        return ArrowDecimal128(lo: l[0], hi: l.count > 1 ? l[1] : (Int64(bitPattern: l[0]) < 0 ? ~0 : 0))
    }
    public func toArray() -> [ArrowDecimal128?] { (0..<length).map { self[$0] } }
    /// Every element as a Double (nulls stay nil). CPU.
    public func toDoubleArray() -> [Double?] { (0..<length).map { self[$0]?.doubleValue(scale: type.scale) } }

    // MARK: - Internal helpers

    private func require128(_ what: String) throws {
        guard type.bitWidth == 128 else {
            throw ArrowMetalError.unsupportedType("\(what) is only implemented for decimal128, not \(type)")
        }
    }
    private func pipeline(_ function: String) throws -> MTLComputePipelineState {
        try context.pipeline(source: DecimalSource.source(limbs: type.limbCount), function: function,
                             cacheKey: "decimal/\(type.bitWidth)/\(function)")
    }
    /// A constant buffer holding one scalar's limbs (sign-extended to this type's width).
    private func scalarLimbs(_ s: ArrowDecimal128) -> [UInt64] { s.limbs(count: type.limbCount) }

    /// 10^k as this type's limbs. `k` must be in 0...(2 * maxPrecision).
    static func pow10(_ k: Int, limbCount: Int) -> [UInt64] {
        var v = [UInt64](repeating: 0, count: limbCount)
        v[0] = 1
        for _ in 0..<k {
            var carry: UInt64 = 0
            for i in 0..<limbCount {
                let p = v[i].multipliedFullWidth(by: 10)
                let (s, o) = p.low.addingReportingOverflow(carry)
                v[i] = s
                carry = p.high &+ (o ? 1 : 0)
            }
        }
        return v
    }

    // MARK: - Compare (GPU)

    /// Element-wise comparison with an unscaled scalar. Nulls propagate (the validity bitmap is shared).
    /// The scalar is interpreted at this array's scale; no rescaling is performed.
    public func compare(_ op: CompareOp, _ scalar: ArrowDecimal128) throws -> MetalBooleanArray {
        try compare(op, limbs: scalarLimbs(scalar))
    }

    /// Comparison against a full-width scalar given as little-endian limbs.
    public func compare(_ op: CompareOp, limbs scalar: [UInt64]) throws -> MetalBooleanArray {
        guard scalar.count == type.limbCount else {
            throw ArrowMetalError.invalidArrowArray("scalar needs \(type.limbCount) limbs for \(type), got \(scalar.count)")
        }
        try Dispatch.checkLength(length)
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: length), 1), zeroed: false, context: ctx)
        let pso = try pipeline("dec_cmp_scalar_\(op.rawValue)")
        if length > 0 {
            let s = scalar
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                s.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 1) }
                Dispatch.setLength(enc, length, nil, index: 2)
                enc.setBuffer(out.mtl, offset: out.offset, index: 3)
                Dispatch.dispatch1D(enc, pso, count: BitmapOps.words(bits: length))
            }
        }
        return MetalBooleanArray(length: length, nullCount: nullCount, validity: validity, values: out, context: ctx)
    }

    /// Element-wise comparison with another decimal array of the same type and length.
    public func compare(_ op: CompareOp, _ other: MetalDecimalArray) throws -> MetalBooleanArray {
        try checkCompatible(other, "compare")
        let ctx = context
        try Dispatch.checkLength(length)
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: length), 1), zeroed: false, context: ctx)
        let pso = try pipeline("dec_cmp_array_\(op.rawValue)")
        if length > 0 {
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                enc.setBuffer(other.values.mtl, offset: other.values.offset, index: 1)
                Dispatch.setLength(enc, length, nil, index: 2)
                enc.setBuffer(out.mtl, offset: out.offset, index: 3)
                Dispatch.dispatch1D(enc, pso, count: BitmapOps.words(bits: length))
            }
        }
        let v = try BitmapOps.combineValidity(ctx, validity, other.validity, bits: length)
        let res = MetalBooleanArray(length: length, nullCount: 0, validity: v, values: out, context: ctx)
        res.recomputeNullCount()
        return res
    }

    private func checkCompatible(_ other: MetalDecimalArray, _ what: String) throws {
        guard other.length == length else { throw ArrowMetalError.lengthMismatch(length, other.length) }
        guard other.type.bitWidth == type.bitWidth else {
            throw ArrowMetalError.unsupportedType("cannot \(what) \(type) with \(other.type): different widths")
        }
        guard other.type.scale == type.scale else {
            throw ArrowMetalError.unsupportedType("cannot \(what) \(type) with \(other.type): scale \(type.scale) vs \(other.type.scale); rescale one side first")
        }
    }

    // MARK: - Reductions (GPU accumulate, CPU combine)

    /// Sum of the non-null unscaled values, nil when there is no valid value (Arrow semantics).
    /// Overflow wraps modulo 2^128, as Arrow's unchecked `sum` does.
    public func sum() throws -> ArrowDecimal128? {
        guard let (partials, counts, groups) = try reduce("dec_reduce_sum") else { return nil }
        return withExtendedLifetime((partials, counts)) {
            let p = partials.typed(UInt64.self), c = counts.typed(UInt32.self)
            let w = type.limbCount
            var acc = ArrowDecimal128.zero
            for g in 0..<groups where c[g] > 0 {
                acc = acc + ArrowDecimal128(lo: p[g * w], hi: p[g * w + 1])
            }
            return acc
        }
    }

    /// Smallest non-null value, nil when there is none.
    public func min() throws -> ArrowDecimal128? { try minMax("dec_reduce_min", Swift.min) }
    /// Largest non-null value, nil when there is none.
    public func max() throws -> ArrowDecimal128? { try minMax("dec_reduce_max", Swift.max) }

    private func minMax(_ fn: String, _ pick: (ArrowDecimal128, ArrowDecimal128) -> ArrowDecimal128) throws -> ArrowDecimal128? {
        guard let (partials, counts, groups) = try reduce(fn) else { return nil }
        return withExtendedLifetime((partials, counts)) { () -> ArrowDecimal128? in
            let p = partials.typed(UInt64.self), c = counts.typed(UInt32.self)
            let w = type.limbCount
            var acc: ArrowDecimal128? = nil
            for g in 0..<groups where c[g] > 0 {
                let v = ArrowDecimal128(lo: p[g * w], hi: p[g * w + 1])
                acc = acc.map { pick($0, v) } ?? v
            }
            return acc
        }
    }

    /// Records one reduction kernel and waits. Returns nil when there is no valid value at all.
    private func reduce(_ fn: String) throws -> (MetalArrowBuffer, MetalArrowBuffer, Int)? {
        if fn != "dec_reduce_sum" { try require128("\(fn) on decimal256") }
        guard validCount > 0 else { return nil }
        try Dispatch.checkLength(length)
        let ctx = context
        let pso = try pipeline(fn)
        let w = type.limbCount
        let groups = Swift.max(1, Swift.min(2048, (length + Dispatch.threadgroupSize - 1) / Dispatch.threadgroupSize))
        let partials = try MetalArrowBuffer.allocate(byteCount: groups * w * 8, zeroed: false, context: ctx)
        let counts = try MetalArrowBuffer.allocate(byteCount: groups * 4, zeroed: false, context: ctx)
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            let vb = validity ?? values
            enc.setBuffer(vb.mtl, offset: vb.offset, index: 1)
            Dispatch.setLength(enc, length, nil, index: 2)
            Dispatch.setUInt(enc, validity == nil ? 0 : 1, index: 3)
            enc.setBuffer(partials.mtl, offset: 0, index: 4)
            enc.setBuffer(counts.mtl, offset: 0, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1))
        }
        try ctx.syncPoint()
        return (partials, counts, groups)
    }

    // MARK: - Arithmetic (GPU)

    /// Element-wise sum with another decimal of the same type. Overflow wraps.
    public func adding(_ other: MetalDecimalArray) throws -> MetalDecimalArray {
        try require128("add")
        try checkCompatible(other, "add")
        return try binary(0, other: other, scalar: nil, type: type)
    }
    /// Element-wise difference. Overflow wraps.
    public func subtracting(_ other: MetalDecimalArray) throws -> MetalDecimalArray {
        try require128("subtract")
        try checkCompatible(other, "subtract")
        return try binary(1, other: other, scalar: nil, type: type)
    }
    /// Adds one unscaled scalar to every element (the scalar is read at this array's scale).
    public func adding(_ scalar: ArrowDecimal128) throws -> MetalDecimalArray {
        try require128("add")
        return try binary(0, other: nil, scalar: scalarLimbs(scalar), type: type)
    }
    public func subtracting(_ scalar: ArrowDecimal128) throws -> MetalDecimalArray {
        try require128("subtract")
        return try binary(1, other: nil, scalar: scalarLimbs(scalar), type: type)
    }

    /// Multiplies every element by an integer scalar, keeping the type (so the scale is unchanged).
    /// Overflow wraps.
    public func multiplied(by scalar: Int64) throws -> MetalDecimalArray {
        try require128("multiply")
        return try binary(2, other: nil, scalar: ArrowDecimal128(scalar).limbs(count: type.limbCount), type: type)
    }

    /// Arrow's decimal multiply: the unscaled values are multiplied and the result's scale is the sum of the
    /// input scales (precision `p1 + p2 + 1`, which must fit the width). Overflow of the product wraps.
    public func multiplied(by other: MetalDecimalArray) throws -> MetalDecimalArray {
        try require128("multiply")
        guard other.length == length else { throw ArrowMetalError.lengthMismatch(length, other.length) }
        guard other.type.bitWidth == 128 else { throw ArrowMetalError.unsupportedType("multiply is only implemented for decimal128, not \(other.type)") }
        let p = type.precision + other.type.precision + 1
        let s = type.scale + other.type.scale
        guard p <= ArrowDecimalType.maxPrecision(bitWidth: 128) else {
            throw ArrowMetalError.unsupportedType("multiply result precision \(p) exceeds decimal128's 38 (\(type) * \(other.type))")
        }
        let rt = try ArrowDecimalType(precision: p, scale: s, bitWidth: 128)
        return try binary(2, other: other, scalar: nil, type: rt)
    }

    /// Two's-complement negation of every element (wrapping).
    public func negated() throws -> MetalDecimalArray { try unary(0, "negate") }
    /// Absolute value of every element (wrapping: the most negative value maps to itself).
    public func absoluteValue() throws -> MetalDecimalArray { try unary(1, "abs") }

    /// -1, 0 or 1 per element, as an int32 array sharing this array's validity.
    public func sign() throws -> MetalArray<Int32> {
        try require128("sign")
        let ctx = context
        try Dispatch.checkLength(length)
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(length * 4, 1), zeroed: false, context: ctx)
        if length > 0 {
            let pso = try pipeline("dec_sign")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                Dispatch.setLength(enc, length, nil, index: 1)
                enc.setBuffer(out.mtl, offset: out.offset, index: 2)
                Dispatch.dispatch1D(enc, pso, count: length)
            }
        }
        return MetalArray<Int32>(length: length, nullCount: nullCount, validity: validity, values: out, context: ctx)
    }

    private func unary(_ op: Int, _ name: String) throws -> MetalDecimalArray {
        try require128(name)
        let ctx = context
        try Dispatch.checkLength(length)
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(length * type.byteWidth, 1), zeroed: false, context: ctx)
        if length > 0 {
            let pso = try pipeline("dec_unary")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                Dispatch.setLength(enc, length, nil, index: 1)
                Dispatch.setUInt(enc, op, index: 2)
                enc.setBuffer(out.mtl, offset: out.offset, index: 3)
                Dispatch.dispatch1D(enc, pso, count: length)
            }
        }
        return MetalDecimalArray(type: type, length: length, nullCount: nullCount, validity: validity, values: out, context: ctx)
    }

    private func binary(_ op: Int, other: MetalDecimalArray?, scalar: [UInt64]?, type rt: ArrowDecimalType) throws -> MetalDecimalArray {
        let ctx = context
        try Dispatch.checkLength(length)
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(length * rt.byteWidth, 1), zeroed: false, context: ctx)
        if length > 0 {
            let pso = try pipeline("dec_binary")
            let s = scalar ?? [UInt64](repeating: 0, count: type.limbCount)
            let b = other?.values ?? values
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                enc.setBuffer(b.mtl, offset: b.offset, index: 1)
                s.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 2) }
                Dispatch.setLength(enc, length, nil, index: 3)
                Dispatch.setUInt(enc, op, index: 4)
                Dispatch.setUInt(enc, scalar == nil ? 0 : 1, index: 5)
                enc.setBuffer(out.mtl, offset: out.offset, index: 6)
                Dispatch.dispatch1D(enc, pso, count: length)
            }
        }
        let v = other == nil ? validity : try BitmapOps.combineValidity(ctx, validity, other!.validity, bits: length)
        let res = MetalDecimalArray(type: rt, length: length, nullCount: 0, validity: v, values: out, context: ctx)
        res.recomputeNullCount()
        return res
    }

    // MARK: - Rescaling: round / ceil / floor / truncate (GPU)

    /// Rounds to `scale` decimal places, returning a decimal with that scale. Halves go away from zero.
    public func rounded(toScale s: Int) throws -> MetalDecimalArray { try rescaled(to: s, mode: .round) }
    /// Rounds toward +infinity at `scale` decimal places.
    public func ceiled(toScale s: Int) throws -> MetalDecimalArray { try rescaled(to: s, mode: .ceil) }
    /// Rounds toward -infinity at `scale` decimal places.
    public func floored(toScale s: Int) throws -> MetalDecimalArray { try rescaled(to: s, mode: .floor) }
    /// Drops the digits below `scale` (toward zero).
    public func truncated(toScale s: Int) throws -> MetalDecimalArray { try rescaled(to: s, mode: .truncate) }

    /// Changes the scale to `target`, rounding with `mode` when digits are dropped. Scaling up is exact
    /// (a multiply by a power of ten) and widens the precision; scaling down rounds and narrows it.
    public func rescaled(to target: Int, mode: DecimalRoundMode) throws -> MetalDecimalArray {
        try require128("round / ceil / floor / truncate")
        guard type.scale >= 0 else {
            throw ArrowMetalError.unsupportedType("rescaling a negative-scale decimal (\(type)) is not implemented")
        }
        let maxP = ArrowDecimalType.maxPrecision(bitWidth: type.bitWidth)
        guard target >= 0, target <= maxP else {
            throw ArrowMetalError.invalidArrowArray("target scale \(target) is outside 0...\(maxP)")
        }
        let delta = target - type.scale
        if delta == 0 { return self }
        let newPrecision = Swift.max(1, Swift.min(maxP, type.precision + delta))
        guard type.precision + delta <= maxP else {
            throw ArrowMetalError.unsupportedType("rescaling \(type) to scale \(target) needs precision \(type.precision + delta), over decimal\(type.bitWidth)'s \(maxP)")
        }
        let rt = try ArrowDecimalType(precision: newPrecision, scale: target, bitWidth: type.bitWidth)
        let ctx = context
        try Dispatch.checkLength(length)
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(length * rt.byteWidth, 1), zeroed: false, context: ctx)
        if length > 0 {
            let factor = Self.pow10(abs(delta), limbCount: type.limbCount)
            if delta > 0 {
                let pso = try pipeline("dec_scale_up")
                try ctx.run { enc in
                    enc.setComputePipelineState(pso)
                    enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                    factor.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 1) }
                    Dispatch.setLength(enc, length, nil, index: 2)
                    enc.setBuffer(out.mtl, offset: out.offset, index: 3)
                    Dispatch.dispatch1D(enc, pso, count: length)
                }
            } else {
                let pso = try pipeline("dec_scale_down")
                try ctx.run { enc in
                    enc.setComputePipelineState(pso)
                    enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                    factor.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 1) }
                    Dispatch.setLength(enc, length, nil, index: 2)
                    Dispatch.setUInt(enc, mode.rawValue, index: 3)
                    enc.setBuffer(out.mtl, offset: out.offset, index: 4)
                    Dispatch.dispatch1D(enc, pso, count: length)
                }
            }
        }
        return MetalDecimalArray(type: rt, length: length, nullCount: nullCount, validity: validity, values: out, context: ctx)
    }

    // MARK: - Selection (GPU)

    /// Index vector `base ..< base + count` as an int32 array, produced on the GPU.
    private func iota(count: Int, base: Int) throws -> MetalArray<Int32> {
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(count * 4, 1), zeroed: false, context: ctx)
        if count > 0 {
            let pso = try ctx.pipeline(source: DecimalSource.gather(limbs: type.limbCount, I: "int"),
                                       function: "dec_iota", cacheKey: "decimal/\(type.bitWidth)/int/dec_iota")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(out.mtl, offset: out.offset, index: 0)
                Dispatch.setLength(enc, count, nil, index: 1)
                Dispatch.setUInt(enc, base, index: 2)
                Dispatch.dispatch1D(enc, pso, count: count)
            }
        }
        return MetalArray<Int32>(length: count, nullCount: 0, validity: nil, values: out, context: ctx)
    }

    /// Arrow `filter`: the elements where `mask` is true. Null mask entries drop the element.
    /// Runs entirely on the GPU: the existing int32 filter compacts an index vector, then a gather over the
    /// 16/32-byte elements moves the values.
    public func filter(_ mask: MetalBooleanArray) throws -> MetalDecimalArray {
        guard mask.length == length else { throw ArrowMetalError.lengthMismatch(length, mask.length) }
        let kept = try iota(count: length, base: 0).filter(mask)
        return try take(kept)
    }

    /// Arrow `take`. A null index yields a null element; an out-of-range index raises after the dispatch.
    public func take<I: ArrowIndex>(_ indices: MetalArray<I>) throws -> MetalDecimalArray {
        try Dispatch.checkLength(length)
        let n = indices.length
        try Dispatch.checkLength(n)
        let ctx = context
        let w = type.limbCount
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * type.byteWidth, 1), zeroed: false, context: ctx)
        let hasV = validity != nil, hasIV = indices.validity != nil
        let validBytes = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1), zeroed: false, context: ctx)
        let errorFlag = try MetalArrowBuffer.allocate(byteCount: 4, context: ctx)
        if n > 0 {
            let pso = try ctx.pipeline(source: DecimalSource.gather(limbs: w, I: I.mslType), function: "dec_take",
                                       cacheKey: "decimal/\(type.bitWidth)/\(I.mslType)/dec_take")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                let vb = validity ?? values
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 1)
                enc.setBuffer(indices.values.mtl, offset: indices.values.offset, index: 2)
                let ib = indices.validity ?? indices.values
                enc.setBuffer(ib.mtl, offset: ib.offset, index: 3)
                Dispatch.setUInt(enc, n, index: 4)
                Dispatch.setLength(enc, length, nil, index: 5)
                Dispatch.setUInt(enc, (hasV ? 1 : 0) | (hasIV ? 2 : 0), index: 6)
                enc.setBuffer(out.mtl, offset: out.offset, index: 7)
                enc.setBuffer(validBytes.mtl, offset: validBytes.offset, index: 8)
                enc.setBuffer(errorFlag.mtl, offset: errorFlag.offset, index: 9)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        let srcLen = length
        try ctx.afterFlush { [errorFlag] in
            if errorFlag.typed(UInt32.self)[0] != 0 {
                throw ArrowMetalError.invalidArrowArray("take: index out of range (array length \(srcLen))")
            }
        }
        ctx.retainUntilFlush(errorFlag); ctx.retainUntilFlush(self); ctx.retainUntilFlush(indices); ctx.retainUntilFlush(validBytes)
        var outValidity: MetalArrowBuffer? = nil
        if (hasV || hasIV) && n > 0 { outValidity = try BitmapOps.packBits(ctx, bytes: validBytes, bits: n) }
        let res = MetalDecimalArray(type: type, length: n, nullCount: 0, validity: outValidity, values: out, context: ctx)
        res.recomputeNullCount()
        return res
    }

    /// Arrow `slice`. Zero-copy when the offset is a multiple of 32 (keeping bitmap words aligned);
    /// otherwise the slice is gathered on the GPU.
    public func slice(offset: Int, length newLength: Int) throws -> MetalDecimalArray {
        guard offset >= 0, newLength >= 0, offset + newLength <= length else {
            throw ArrowMetalError.invalidArrowArray("slice \(offset)..<\(offset + newLength) is outside 0..<\(length)")
        }
        if offset % 32 == 0 {
            let v = values.view(byteOffset: offset * type.byteWidth, byteCount: newLength * type.byteWidth)
            let bm = validity?.view(byteOffset: offset / 8, byteCount: Bitmap.byteCount(bits: newLength))
            let res = MetalDecimalArray(type: type, length: newLength, nullCount: 0, validity: bm, values: v, context: context)
            res.recomputeNullCount()
            return res
        }
        return try take(try iota(count: newLength, base: offset))
    }

    // MARK: - Casts (CPU)

    /// The values as `float64` (unscaled / 10^scale). CPU: one host pass, Double's 53 bits of precision.
    public func toFloat64() throws -> MetalArray<Double> {
        ensure()
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(length * 8, 1), zeroed: false, context: ctx)
        withExtendedLifetime(values) {
            let src = values.typed(UInt64.self)
            let dst = out.mutableTyped(Double.self)
            let w = type.limbCount, s = type.scale
            let divisor = s == 0 ? 1.0 : pow(10.0, Double(s))
            for i in 0..<length {
                let v = ArrowDecimal128(lo: src[i * w], hi: w > 1 ? src[i * w + 1] : (Int64(bitPattern: src[i * w]) < 0 ? ~0 : 0))
                let m = v.magnitude128
                let d = (Double(m.hi) * 0x1p64 + Double(m.lo)) / divisor
                dst[i] = v.isNegative ? -d : d
            }
        }
        return MetalArray<Double>(length: length, nullCount: nullCount, validity: validity, values: out, context: ctx)
    }

    /// Builds a decimal column from `float64` values, multiplying by 10^scale and rounding halves away from
    /// zero. CPU: the conversion is a host pass and only Double's 53 significant bits survive.
    /// A non-finite or out-of-range value becomes null.
    public static func fromFloat64(_ a: MetalArray<Double>, type: ArrowDecimalType,
                                   context: MetalContext? = nil) throws -> MetalDecimalArray {
        guard type.bitWidth == 128 else { throw ArrowMetalError.unsupportedType("cast to \(type) is only implemented for decimal128") }
        guard type.scale >= 0 else { throw ArrowMetalError.unsupportedType("cast to a negative-scale decimal (\(type)) is not implemented") }
        let ctx = context ?? a.context
        let n = a.length
        let out = try MetalDecimalArray.allocate(type: type, length: n, withValidity: true, context: ctx)
        let factor = pow(10.0, Double(type.scale))
        withExtendedLifetime((a, out)) {
            let src = a.valuePointer
            let dst = out.values.mutableTyped(UInt64.self)
            let bp = out.validity!.mutableTyped(UInt8.self)
            var nulls = 0
            for i in 0..<n {
                guard a.isValid(i) else { nulls += 1; continue }
                guard let v = decimal128(fromDouble: src[i] * factor) else { nulls += 1; continue }
                dst[i * 2] = v.lo; dst[i * 2 + 1] = v.hi
                Bitmap.set(bp, i)
            }
            out.setNullCount(nulls)
        }
        return out
    }

    /// Builds a decimal column from `int64` values, multiplying by 10^scale (exact, wrapping past 128 bits).
    /// CPU: one host pass.
    public static func fromInt64(_ a: MetalArray<Int64>, type: ArrowDecimalType,
                                 context: MetalContext? = nil) throws -> MetalDecimalArray {
        guard type.bitWidth == 128 else { throw ArrowMetalError.unsupportedType("cast to \(type) is only implemented for decimal128") }
        guard type.scale >= 0 else { throw ArrowMetalError.unsupportedType("cast to a negative-scale decimal (\(type)) is not implemented") }
        let ctx = context ?? a.context
        let n = a.length
        let out = try MetalDecimalArray.allocate(type: type, length: n, withValidity: a.validity != nil, context: ctx)
        let p10 = pow10(type.scale, limbCount: 2)
        let factor = ArrowDecimal128(lo: p10[0], hi: p10[1])
        withExtendedLifetime((a, out)) {
            let src = a.valuePointer
            let dst = out.values.mutableTyped(UInt64.self)
            for i in 0..<n {
                let v = ArrowDecimal128(src[i]) * factor
                dst[i * 2] = v.lo; dst[i * 2 + 1] = v.hi
            }
            if let bp = out.validity?.mutableTyped(UInt8.self), let vv = a.validity {
                memcpy(bp, vv.typed(UInt8.self), Bitmap.byteCount(bits: n))
            }
        }
        out.setNullCount(a.nullCount)
        return out
    }

    /// Rounds a Double (already scaled) to a 128-bit integer, nil when it is not finite or does not fit.
    static func decimal128(fromDouble d: Double) -> ArrowDecimal128? {
        guard d.isFinite else { return nil }
        let r = d.rounded(.toNearestOrAwayFromZero)
        let m = r.magnitude
        guard m < 0x1p127 else { return nil }
        let hiPart = (m / 0x1p64).rounded(.down)
        let loPart = m - hiPart * 0x1p64
        let v = ArrowDecimal128(lo: UInt64(loPart), hi: UInt64(hiPart))
        return r < 0 ? v.negated : v
    }
}

// MARK: - C Data Interface

extension MetalDecimalArray {
    /// Exports through the CPU C Data Interface (zero-copy: validity + a raw fixed-width values buffer).
    public func exportArrowArray(into out: UnsafeMutablePointer<ArrowArray>) {
        // A decimal array has the same two-buffer shape as a primitive one; reuse the primitive exporter by
        // describing the values buffer as bytes. Length and null count come from this array, so the exported
        // struct is exactly the decimal array.
        ensure()
        let proxy = MetalArray<UInt8>(length: length, nullCount: nullCount, validity: validity, values: values, context: context)
        proxy.exportArrowArray(into: out)
    }
    public func exportArrowSchema(name: String = "", into out: UnsafeMutablePointer<ArrowSchema>) {
        ArrowMetal.exportArrowSchema(format: type.arrowFormat, name: name, into: out)
    }
    public func exportArrowDeviceArray(into out: UnsafeMutablePointer<ArrowDeviceArray>) {
        withUnsafeMutablePointer(to: &out.pointee.array) { exportArrowArray(into: $0) }
        out.pointee.device_type = ARROW_DEVICE_METAL
        out.pointee.device_id = -1
        out.pointee.reserved = (0, 0, 0)
        out.pointee.sync_event = nil
    }
}

/// Imports a decimal128 / decimal256 array: two buffers, the values a raw fixed-width block.
func importDecimalArray(type: ArrowDecimalType, array: UnsafeMutablePointer<ArrowArray>,
                        context: MetalContext) throws -> ImportResult {
    guard array.pointee.n_buffers == 2, array.pointee.buffers != nil else {
        throw ArrowMetalError.invalidArrowArray("expected 2 buffers for a decimal array, got \(array.pointee.n_buffers)")
    }
    // Fast path: an array this process exported. Share the MTLBuffer objects directly.
    if let own = ownedExportBuffers(array) {
        let owner = ImportedCArray(moving: array)
        let a = owner.array
        let arr = MetalDecimalArray(type: type, length: Int(a.length), nullCount: 0,
                                    validity: own.validity, values: own.values, context: context)
        if a.null_count < 0 { arr.recomputeNullCount() } else { arr.setNullCount(Int(a.null_count)) }
        return ImportResult(array: .decimal(arr), zeroCopy: true)
    }
    let owner = ImportedCArray(moving: array)
    let a = owner.array
    let length = Int(a.length), offset = Int(a.offset)
    let validityPtr = a.buffers[0].map { UnsafeRawPointer($0) }
    guard let valuesPtr = a.buffers[1].map({ UnsafeRawPointer($0) }) else {
        throw ArrowMetalError.invalidArrowArray("values buffer is null")
    }
    let (vals, zc1) = try MetalArrowBuffer.wrapOrCopy(valuesPtr.advanced(by: offset * type.byteWidth),
                                                      byteCount: length * type.byteWidth, keepAlive: owner, context: context)
    var validity: MetalArrowBuffer? = nil
    var zc2 = true
    if let vp = validityPtr {
        let bytes = Bitmap.byteCount(bits: length)
        if offset == 0 {
            (validity, zc2) = try MetalArrowBuffer.wrapOrCopy(vp, byteCount: bytes, keepAlive: owner, context: context)
        } else {
            zc2 = false
            let buf = try MetalArrowBuffer.allocate(byteCount: Swift.max(bytes, 1), context: context)
            let sp = vp.assumingMemoryBound(to: UInt8.self)
            let dp = buf.mutableTyped(UInt8.self)
            if offset % 8 == 0 { memcpy(dp, sp + offset / 8, bytes) }
            else { for i in 0..<length where Bitmap.isSet(sp, i + offset) { Bitmap.set(dp, i) } }
            validity = buf
        }
    }
    let arr = MetalDecimalArray(type: type, length: length, nullCount: 0, validity: validity, values: vals, context: context)
    if a.null_count < 0 || offset != 0 { arr.recomputeNullCount() } else { arr.setNullCount(Int(a.null_count)) }
    return ImportResult(array: .decimal(arr), zeroCopy: zc1 && zc2)
}
