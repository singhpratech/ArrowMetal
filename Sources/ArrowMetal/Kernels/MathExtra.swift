import Foundation
import Metal

/// Arrow's ten rounding modes, in Arrow's own numbering — the C ABI and the Python layer pass the index,
/// so the order here is part of the contract.
public enum RoundMode: Int, CaseIterable, Sendable {
    /// Toward negative infinity (`floor`).
    case down = 0
    /// Toward positive infinity (`ceil`).
    case up
    /// Toward zero (`trunc`).
    case towardsZero
    /// Away from zero.
    case towardsInfinity
    /// Halves toward negative infinity; everything else to the nearer value.
    case halfDown
    /// Halves toward positive infinity.
    case halfUp
    /// Halves toward zero.
    case halfTowardsZero
    /// Halves away from zero — the mode the no-argument `round()` in `Kernels/Rounding.swift` uses.
    case halfTowardsInfinity
    /// Halves to the even neighbour. Arrow's default, and the default here.
    case halfToEven
    /// Halves to the odd neighbour.
    case halfToOdd

    /// The `pyarrow.compute` spelling, so a caller can pass Arrow's own name straight through.
    public var arrowName: String {
        switch self {
        case .down: return "down"
        case .up: return "up"
        case .towardsZero: return "towards_zero"
        case .towardsInfinity: return "towards_infinity"
        case .halfDown: return "half_down"
        case .halfUp: return "half_up"
        case .halfTowardsZero: return "half_towards_zero"
        case .halfTowardsInfinity: return "half_towards_infinity"
        case .halfToEven: return "half_to_even"
        case .halfToOdd: return "half_to_odd"
        }
    }

    public static func named(_ s: String) -> RoundMode? { allCases.first { $0.arrowName == s } }
}

/// The element-wise math Arrow defines beyond `Kernels/Rounding.swift`: `expm1`, `log1p`, `logb`,
/// `hypot`, and the rounding family in all ten `RoundMode`s (`round(ndigits:mode:)`,
/// `round_to_multiple`, `round_binary`).
///
/// All GPU, all null-aware in the usual way: a unary op and a scalar-right-hand-side op share the input's
/// validity bitmap zero-copy, and a two-column op takes the AND of both bitmaps.
///
/// **Precision.** `float32` uses the MSL library functions (`expm1` and `log1p`, which MSL does not have,
/// are built from `exp`/`log` with Kahan's correction so that they keep full relative accuracy near
/// zero). `float64` runs entirely in software binary64 — see `Kernels/DoubleTranscendental.swift` for the
/// methods and the ulp bounds measured against Foundation. That makes these four functions considerably
/// more accurate on `float64` than the older `sqrt`/`exp`/`ln`/`log2`/`log10`, which still take the
/// `float` detour documented on `RoundingSource`; the same machinery could lift those later.
///
/// **Integers.** Arrow defines the rounding family on integer columns and it is implemented here, exactly,
/// on the quotient and remainder (so an `int64` above 2^53 rounds without ever touching a float).
/// `expm1`, `log1p`, `logb` and `hypot` need a floating point column and throw on an integer one — Arrow
/// promotes to `float64` instead, which ArrowMetal leaves to an explicit `cast`, as it already does for
/// `sqrt` and `ln`.
extension MetalArray {
    /// Generated source for the extra math kernels of this element type, plus its pipeline cache key.
    static var mathExtraSource: (source: String, type: String) {
        if T.self == Double.self {
            return (MathExtraSource.source(T: "ulong", U: "ulong", width: 64, kind: .float64), "double")
        }
        if T.self == Float.self {
            return (MathExtraSource.source(T: "float", U: "float", width: 32, kind: .float32), "float")
        }
        let t = T.mslType
        return (MathExtraSource.source(T: t, U: MathTypes.unsigned(t), width: MathTypes.bitWidth(T.self),
                                       kind: MathTypes.isSigned(T.self) ? .signedInt : .unsignedInt), t)
    }

    private func mathExtraPipeline(_ function: String) throws -> MTLComputePipelineState {
        let (src, ty) = Self.mathExtraSource
        return try Dispatch.pipeline(context, family: "mathextra", source: src, function: function, type: ty)
    }

    /// One-input kernel over the values, sharing the input's validity bitmap.
    private func extraUnary(_ function: String, requiresFloat: Bool, name: String) throws -> MetalArray<T> {
        if requiresFloat { try MathTypes.requireFloat(T.self, name) }
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * T.byteWidth, zeroed: false, context: ctx)
        if n > 0 {
            let pso = try mathExtraPipeline(function)
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                Dispatch.setLength(enc, n, lengthBuffer, index: 1)
                enc.setBuffer(out.mtl, offset: out.offset, index: 2)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        return inheritPending(MetalArray<T>(length: knownLength, nullCount: _nullCount, validity: validity,
                                            values: out, context: ctx))
    }

    /// Two-input kernel with a scalar right-hand side, sharing the input's validity bitmap.
    private func extraScalar(_ function: String, _ scalar: T, requiresFloat: Bool, name: String) throws -> MetalArray<T> {
        if requiresFloat { try MathTypes.requireFloat(T.self, name) }
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * T.byteWidth, zeroed: false, context: ctx)
        if n > 0 {
            let pso = try mathExtraPipeline(function)
            let isDouble = T.self == Double.self
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                if isDouble { Dispatch.setScalar(enc, (scalar as! Double).bitPattern, index: 1) }
                else { Dispatch.setScalar(enc, scalar, index: 1) }
                Dispatch.setLength(enc, n, lengthBuffer, index: 2)
                enc.setBuffer(out.mtl, offset: out.offset, index: 3)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        return inheritPending(MetalArray<T>(length: knownLength, nullCount: _nullCount, validity: validity,
                                            values: out, context: ctx))
    }

    /// Two-column kernel; the result is null wherever either side is, as for every Arrow binary kernel.
    private func extraArray(_ function: String, _ other: MetalArray<T>, requiresFloat: Bool, name: String) throws -> MetalArray<T> {
        if requiresFloat { try MathTypes.requireFloat(T.self, name) }
        try checkSameLength(other)
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * T.byteWidth, zeroed: false, context: ctx)
        if n > 0 {
            let pso = try mathExtraPipeline(function)
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                enc.setBuffer(other.values.mtl, offset: other.values.offset, index: 1)
                Dispatch.setLength(enc, n, lengthBuffer, index: 2)
                enc.setBuffer(out.mtl, offset: out.offset, index: 3)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        let v = try BitmapOps.combineValidity(ctx, validity, other.validity, bits: n, lengthBuffer: lengthBuffer)
        let res = inheritPending(MetalArray<T>(length: knownLength, nullCount: 0, validity: v, values: out, context: ctx))
        res.recomputeNullCount()
        ctx.retainUntilFlush(self)
        ctx.retainUntilFlush(other)
        return res
    }

    // MARK: - Transcendentals

    /// Arrow `expm1`: `exp(x) - 1`, accurate for small `x` where `exp(x) - 1` would cancel away.
    public func expm1() throws -> MetalArray<T> { try extraUnary("mx_unary_expm1", requiresFloat: true, name: "expm1") }

    /// Arrow `log1p`: `ln(1 + x)`, accurate for small `x`. `x == -1` gives `-inf` and `x < -1` gives NaN;
    /// `log1p_checked` raises on both instead.
    public func log1p() throws -> MetalArray<T> { try extraUnary("mx_unary_log1p", requiresFloat: true, name: "log1p") }

    /// Arrow `logb(x, base)`: `ln(x) / ln(base)`, with a single base for the whole column.
    public func logb(_ base: T) throws -> MetalArray<T> { try extraScalar("mx_scalar_logb", base, requiresFloat: true, name: "logb") }

    /// Arrow `logb(x, base)` with a per-row base.
    public func logb(_ base: MetalArray<T>) throws -> MetalArray<T> { try extraArray("mx_array_logb", base, requiresFloat: true, name: "logb") }

    /// Arrow `hypot`: `sqrt(x² + y²)`, computed with scaling so that a large or tiny pair neither
    /// overflows nor underflows on the way. An infinite operand gives `inf` even opposite a NaN, which is
    /// what IEEE-754 prescribes and what Arrow does.
    public func hypot(_ other: T) throws -> MetalArray<T> { try extraScalar("mx_scalar_hypot", other, requiresFloat: true, name: "hypot") }

    /// Arrow `hypot` against another column.
    public func hypot(_ other: MetalArray<T>) throws -> MetalArray<T> { try extraArray("mx_array_hypot", other, requiresFloat: true, name: "hypot") }

    // MARK: - Rounding

    /// Arrow `round(x, ndigits, round_mode)`.
    ///
    /// On a float column this is `round_int(x · 10^ndigits) / 10^ndigits` (and the reciprocal form for a
    /// negative `ndigits`), the expression Arrow evaluates — so, like Arrow, it inherits the rounding of
    /// the scaling: `round(123.456, ndigits: 2)` is `123.46`, because `123.456 · 100` is `12345.6` only
    /// to within a rounding. On an integer column a non-negative `ndigits` is the identity and a negative
    /// one rounds to a multiple of `10^-ndigits`.
    ///
    /// Differences from Arrow at the extremes, both defined rather than raised here: a float `ndigits`
    /// past the type's decimal range is the identity (Arrow raises "overflow occurred during rounding"
    /// above about 10^308), and an integer `ndigits` whose multiple does not fit the column type gives 0
    /// (Arrow raises "Rounding to -N digits is out of range").
    public func round(ndigits: Int, mode: RoundMode = .halfToEven) throws -> MetalArray<T> {
        try roundKernel("mx_round", mode: mode) { enc in
            Dispatch.setScalar(enc, Int32(clamping: ndigits), index: 1)
        }
    }

    /// Arrow `round_to_multiple(x, multiple, round_mode)`: `round_int(x / multiple) · multiple`.
    /// `multiple` must be strictly positive, as it must be in Arrow.
    public func roundToMultiple(_ multiple: T, mode: RoundMode = .halfToEven) throws -> MetalArray<T> {
        guard multiple > .zero else {
            throw ArrowMetalError.invalidArrowArray("round_to_multiple: rounding multiple must be positive")
        }
        let isDouble = T.self == Double.self
        return try roundKernel("mx_round_multiple", mode: mode) { enc in
            if isDouble { Dispatch.setScalar(enc, (multiple as! Double).bitPattern, index: 1) }
            else { Dispatch.setScalar(enc, multiple, index: 1) }
        }
    }

    /// Arrow `round_binary(x, ndigits)`: `round` with one `ndigits` per row. The result is null wherever
    /// either column is.
    public func roundBinary(_ ndigits: MetalArray<Int32>, mode: RoundMode = .halfToEven) throws -> MetalArray<T> {
        try checkSameLength(ndigits)
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * T.byteWidth, zeroed: false, context: ctx)
        if n > 0 {
            let pso = try mathExtraPipeline("mx_round_binary")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                enc.setBuffer(ndigits.values.mtl, offset: ndigits.values.offset, index: 1)
                Dispatch.setUInt(enc, mode.rawValue, index: 2)
                Dispatch.setLength(enc, n, lengthBuffer, index: 3)
                enc.setBuffer(out.mtl, offset: out.offset, index: 4)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        let v = try BitmapOps.combineValidity(ctx, validity, ndigits.validity, bits: n, lengthBuffer: lengthBuffer)
        let res = inheritPending(MetalArray<T>(length: knownLength, nullCount: 0, validity: v, values: out, context: ctx))
        res.recomputeNullCount()
        ctx.retainUntilFlush(self)
        ctx.retainUntilFlush(ndigits)
        return res
    }

    /// Shared body of `round` and `round_to_multiple`: buffer 1 is the op's own argument.
    private func roundKernel(_ function: String, mode: RoundMode,
                             _ bindArgument: (MTLComputeCommandEncoder) -> Void) throws -> MetalArray<T> {
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * T.byteWidth, zeroed: false, context: ctx)
        if n > 0 {
            let pso = try mathExtraPipeline(function)
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                bindArgument(enc)
                Dispatch.setUInt(enc, mode.rawValue, index: 2)
                Dispatch.setLength(enc, n, lengthBuffer, index: 3)
                enc.setBuffer(out.mtl, offset: out.offset, index: 4)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        return inheritPending(MetalArray<T>(length: knownLength, nullCount: _nullCount, validity: validity,
                                            values: out, context: ctx))
    }
}
