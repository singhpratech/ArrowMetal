import Foundation
import Metal

/// Element-wise unary arithmetic beyond `add`/`sub`/`mul`/`div`.
///
/// `sqrt`, `exp`, `ln`, `log10` and `log2` need a floating point column and throw on an integer one —
/// Arrow promotes integers to `float64` for these, which ArrowMetal leaves to an explicit `cast`.
/// `floor`, `ceil`, `round` and `trunc` are the identity on an integer column and keep its type.
/// `round` rounds halfway cases away from zero.
public enum UnaryMathOp: String, CaseIterable, Sendable {
    case negate, abs, sign
    case sqrt, exp, ln, log10, log2
    case floor, ceil, round, trunc

    /// True for the ops that are only defined on `float32` / `float64` columns here.
    public var requiresFloat: Bool {
        switch self {
        case .sqrt, .exp, .ln, .log10, .log2: return true
        default: return false
        }
    }
}

/// Element-wise binary arithmetic beyond `add`/`sub`/`mul`/`div`.
public enum BinaryMathOp: String, CaseIterable, Sendable {
    /// Integers use repeated squaring and wrap; a negative exponent is defined as 0. Floats use `pow`.
    case power
    /// `%` with C remainder semantics: the sign follows the dividend. Integer `x % 0` is defined as 0.
    case modulo
    /// Arrow `min_element_wise` with `skip_nulls` (the default): a null on one side yields the other
    /// side's value, and only two nulls make a null.
    case minElementWise
    /// Arrow `max_element_wise`, same null handling.
    case maxElementWise

    var kernelSuffix: String {
        switch self {
        case .power: return "power"
        case .modulo: return "modulo"
        case .minElementWise: return "min_ew"
        case .maxElementWise: return "max_ew"
        }
    }
    var isMinMax: Bool { self == .minElementWise || self == .maxElementWise }
}

extension MetalArray {
    /// Source and pipeline cache key for the element-wise math kernels of this element type.
    static var mathSource: (source: String, type: String) {
        if T.self == Double.self { return (RoundingSource.double, "double") }
        return (RoundingSource.source(T: T.mslType, U: MathTypes.unsigned(T.mslType),
                                      isFloat: T.isFloatingPoint, isSigned: MathTypes.isSigned(T.self)),
                T.mslType)
    }

    /// One element-wise unary math op. Validity is shared with the input, zero-copy: null in, null out.
    ///
    /// On `float64`, `negate`/`abs`/`sign`/`floor`/`ceil`/`round`/`trunc` are exact (bit-pattern work on
    /// the GPU). `sqrt`/`exp`/`ln`/`log10`/`log2` are evaluated in `float` and widened, so they carry
    /// about 7 correct significant decimal digits; see `RoundingSource`.
    public func unaryMath(_ op: UnaryMathOp) throws -> MetalArray<T> {
        if op.requiresFloat { try MathTypes.requireFloat(T.self, op.rawValue) }
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let ctx = context
        let (src, cacheType) = Self.mathSource
        let out = try MetalArrowBuffer.allocate(byteCount: n * T.byteWidth, zeroed: false, context: ctx)
        let pso = try Dispatch.pipeline(ctx, family: "math", source: src, function: "math_unary_\(op.rawValue)", type: cacheType)
        if n > 0 {
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                Dispatch.setLength(enc, n, lengthBuffer, index: 1)
                enc.setBuffer(out.mtl, offset: out.offset, index: 2)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        return inheritPending(MetalArray<T>(length: knownLength, nullCount: _nullCount, validity: validity, values: out, context: ctx))
    }

    /// `power` or `modulo` against a scalar. Validity is shared with the input.
    public func binaryMath(_ op: BinaryMathOp, _ scalar: T) throws -> MetalArray<T> {
        guard !op.isMinMax else {
            throw ArrowMetalError.unsupportedType("\(op.rawValue) takes two arrays; it has no scalar form")
        }
        try Self.requireBinarySupported(op)
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let ctx = context
        let (src, cacheType) = Self.mathSource
        let out = try MetalArrowBuffer.allocate(byteCount: n * T.byteWidth, zeroed: false, context: ctx)
        let pso = try Dispatch.pipeline(ctx, family: "math", source: src, function: "math_scalar_\(op.kernelSuffix)", type: cacheType)
        if n > 0 {
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                Dispatch.setScalar(enc, scalar, index: 1)
                Dispatch.setLength(enc, n, lengthBuffer, index: 2)
                enc.setBuffer(out.mtl, offset: out.offset, index: 3)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        return inheritPending(MetalArray<T>(length: knownLength, nullCount: _nullCount, validity: validity, values: out, context: ctx))
    }

    /// `power`, `modulo`, `min_element_wise` or `max_element_wise` against another column.
    ///
    /// `power` and `modulo` propagate nulls (output validity is the AND of both inputs); the element-wise
    /// min and max skip them (output validity is the OR), which is Arrow's `skip_nulls` default.
    public func binaryMath(_ op: BinaryMathOp, _ other: MetalArray<T>) throws -> MetalArray<T> {
        try Self.requireBinarySupported(op)
        try checkSameLength(other)
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let ctx = context
        let (src, cacheType) = Self.mathSource
        let out = try MetalArrowBuffer.allocate(byteCount: n * T.byteWidth, zeroed: false, context: ctx)
        let fn = op.isMinMax ? "math_minmax" : "math_array_\(op.kernelSuffix)"
        let pso = try Dispatch.pipeline(ctx, family: "math", source: src, function: fn, type: cacheType)
        let hasA = validity != nil, hasB = other.validity != nil
        if n > 0 {
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                enc.setBuffer(other.values.mtl, offset: other.values.offset, index: 1)
                if op.isMinMax {
                    let va = validity ?? values, vb = other.validity ?? other.values
                    enc.setBuffer(va.mtl, offset: va.offset, index: 2)
                    enc.setBuffer(vb.mtl, offset: vb.offset, index: 3)
                    Dispatch.setLength(enc, n, lengthBuffer, index: 4)
                    Dispatch.setUInt(enc, (hasA ? 1 : 0) | (hasB ? 2 : 0), index: 5)
                    Dispatch.setUInt(enc, op == .maxElementWise ? 1 : 0, index: 6)
                    enc.setBuffer(out.mtl, offset: out.offset, index: 7)
                } else {
                    Dispatch.setLength(enc, n, lengthBuffer, index: 2)
                    enc.setBuffer(out.mtl, offset: out.offset, index: 3)
                }
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        // Null-skipping min/max keeps a value wherever *either* side had one.
        var v: MetalArrowBuffer? = nil
        if op.isMinMax {
            if let a = validity, let b = other.validity {
                v = try BitmapOps.binary(ctx, "bitmap_or", a, b, bits: n, lengthBuffer: lengthBuffer)
            }
        } else {
            v = try BitmapOps.combineValidity(ctx, validity, other.validity, bits: n, lengthBuffer: lengthBuffer)
        }
        let res = inheritPending(MetalArray<T>(length: knownLength, nullCount: 0, validity: v, values: out, context: ctx))
        res.recomputeNullCount()
        ctx.retainUntilFlush(self)
        ctx.retainUntilFlush(other)
        return res
    }

    /// `power` and `modulo` on `float64` would have to run through the `float` path, which is far too
    /// coarse to be worth shipping; say so instead of returning a bad answer.
    private static func requireBinarySupported(_ op: BinaryMathOp) throws {
        if T.self == Double.self && !op.isMinMax {
            throw ArrowMetalError.unsupportedType("\(op.rawValue) is not implemented for float64; cast to float32 first")
        }
    }

    // MARK: - Named forms

    public func negate() throws -> MetalArray<T> { try unaryMath(.negate) }
    public func abs() throws -> MetalArray<T> { try unaryMath(.abs) }
    public func sign() throws -> MetalArray<T> { try unaryMath(.sign) }
    public func sqrt() throws -> MetalArray<T> { try unaryMath(.sqrt) }
    public func exp() throws -> MetalArray<T> { try unaryMath(.exp) }
    public func ln() throws -> MetalArray<T> { try unaryMath(.ln) }
    public func log10() throws -> MetalArray<T> { try unaryMath(.log10) }
    public func log2() throws -> MetalArray<T> { try unaryMath(.log2) }
    public func floor() throws -> MetalArray<T> { try unaryMath(.floor) }
    public func ceil() throws -> MetalArray<T> { try unaryMath(.ceil) }
    public func round() throws -> MetalArray<T> { try unaryMath(.round) }
    public func trunc() throws -> MetalArray<T> { try unaryMath(.trunc) }

    public func power(_ s: T) throws -> MetalArray<T> { try binaryMath(.power, s) }
    public func power(_ o: MetalArray<T>) throws -> MetalArray<T> { try binaryMath(.power, o) }
    public func modulo(_ s: T) throws -> MetalArray<T> { try binaryMath(.modulo, s) }
    public func modulo(_ o: MetalArray<T>) throws -> MetalArray<T> { try binaryMath(.modulo, o) }
    public func minElementWise(_ o: MetalArray<T>) throws -> MetalArray<T> { try binaryMath(.minElementWise, o) }
    public func maxElementWise(_ o: MetalArray<T>) throws -> MetalArray<T> { try binaryMath(.maxElementWise, o) }
}
