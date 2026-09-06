import Foundation
import Metal

/// Arrow's trigonometric, inverse-trigonometric and hyperbolic functions.
///
/// All twelve are unary; `atan2` is binary and lives on its own below. Like Arrow's `sin`/`cos`/…
/// they need a floating point column: an integer column throws rather than being promoted to
/// `float64`, matching the rest of ArrowMetal's transcendental family (`sqrt`, `exp`, `ln`).
public enum TrigOp: String, CaseIterable, Sendable {
    case sin, cos, tan, asin, acos, atan
    case sinh, cosh, tanh, asinh, acosh, atanh
}

/// The seven of them Arrow also publishes in a `_checked` form, which raises on a domain violation
/// instead of returning NaN. Arrow has no checked `atan`, `sinh`, `cosh`, `tanh` or `asinh`: those
/// are defined on the whole real line.
public enum TrigCheckedOp: String, CaseIterable, Sendable {
    case sin, cos, tan, asin, acos, acosh, atanh

    /// The unchecked function this one produces the values of.
    public var unchecked: TrigOp { TrigOp(rawValue: rawValue)! }
    /// The Arrow compute function name (`"asin_checked"`, …), used in error messages.
    public var arrowName: String { "\(rawValue)_checked" }
}

extension MetalArray {
    /// One element-wise trigonometric or hyperbolic function. Validity is shared with the input,
    /// zero-copy: null in, null out.
    ///
    /// `float32` runs the MSL library functions (`precise::sin`, `precise::cos`, `precise::tan` and
    /// the accurate default `asin` / `acos` / `atan`, with fast math off); the six hyperbolics are
    /// written out in `TrigSource` because Metal's own lose accuracy and mishandle ±∞. `float64`
    /// runs a **software
    /// binary64** implementation on the GPU — Cody-Waite reduction against a 128-bit π/2 plus Taylor
    /// series over the correctly rounded `d_add` / `d_mul` / `d_div` of `DoubleMath.swift` — so the
    /// results carry full double precision rather than a widened `float`. See `TrigSource.swift` for
    /// the identities, the special-value table and the reduction's range limit.
    public func trig(_ op: TrigOp) throws -> MetalArray<T> {
        try MathTypes.requireFloat(T.self, op.rawValue)
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * T.byteWidth, zeroed: false, context: ctx)
        if n > 0 {
            let pso = try Self.trigPipeline(ctx, "tg_\(op.rawValue)")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                Dispatch.setLength(enc, n, lengthBuffer, index: 1)
                enc.setBuffer(out.mtl, offset: out.offset, index: 2)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
            ctx.retainUntilFlush(self)
        }
        return inheritPending(MetalArray<T>(length: knownLength, nullCount: _nullCount, validity: validity, values: out, context: ctx))
    }

    /// Arrow `atan2(y, x)`: the angle of the point `(x, y)` in `[-π, π]`, with this column as `y`
    /// and `other` as `x`. Output validity is the AND of both inputs, as for every binary kernel.
    public func atan2(_ other: MetalArray<T>) throws -> MetalArray<T> {
        try MathTypes.requireFloat(T.self, "atan2")
        try checkSameLength(other)
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * T.byteWidth, zeroed: false, context: ctx)
        if n > 0 {
            let pso = try Self.trigPipeline(ctx, "tg_atan2_array")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                enc.setBuffer(other.values.mtl, offset: other.values.offset, index: 1)
                Dispatch.setLength(enc, n, lengthBuffer, index: 2)
                enc.setBuffer(out.mtl, offset: out.offset, index: 3)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
            ctx.retainUntilFlush(self); ctx.retainUntilFlush(other)
        }
        let v = try BitmapOps.combineValidity(ctx, validity, other.validity, bits: n, lengthBuffer: lengthBuffer)
        let res = inheritPending(MetalArray<T>(length: knownLength, nullCount: 0, validity: v, values: out, context: ctx))
        res.recomputeNullCount()
        return res
    }

    /// `atan2(y, scalar)` with this column as `y`. Validity is shared with the input.
    public func atan2(_ scalar: T) throws -> MetalArray<T> {
        try MathTypes.requireFloat(T.self, "atan2")
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * T.byteWidth, zeroed: false, context: ctx)
        if n > 0 {
            let pso = try Self.trigPipeline(ctx, "tg_atan2_scalar")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                if let d = scalar as? Double { Dispatch.setScalar(enc, d.bitPattern, index: 1) }
                else { Dispatch.setScalar(enc, scalar, index: 1) }
                Dispatch.setLength(enc, n, lengthBuffer, index: 2)
                enc.setBuffer(out.mtl, offset: out.offset, index: 3)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
            ctx.retainUntilFlush(self)
        }
        return inheritPending(MetalArray<T>(length: knownLength, nullCount: _nullCount, validity: validity, values: out, context: ctx))
    }

    /// The generated MSL for this element type, compiled once and cached.
    static func trigPipeline(_ ctx: MetalContext, _ function: String) throws -> MTLComputePipelineState {
        let isDouble = T.self == Double.self
        return try Dispatch.pipeline(ctx, family: "trig", source: isDouble ? TrigSource.double : TrigSource.float32,
                                     function: function, type: isDouble ? "double" : "float")
    }

    // MARK: - Checked forms

    /// One of Arrow's `_checked` trigonometric functions: the same values as the unchecked op, but a
    /// **domain violation on a non-null row raises** instead of producing NaN.
    ///
    /// The domain is Arrow's: `asin`/`acos` need `|x| ≤ 1`, `acosh` needs `x ≥ 1`, `atanh` needs
    /// `|x| < 1`, and `sin`/`cos`/`tan` reject ±∞. A NaN input never raises (it is unordered, not out
    /// of domain) and a null row is not inspected, both matching `pyarrow.compute`.
    ///
    /// The check rides along inside the same kernel: one device flag set with a relaxed atomic store
    /// and one `atomic_fetch_min` of the offending row, read back after the dispatch, so a clean
    /// column costs nothing beyond the unchecked op and the error names the first bad row.
    public func trigChecked(_ op: TrigCheckedOp) throws -> MetalArray<T> {
        try MathTypes.requireFloat(T.self, op.arrowName)
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * T.byteWidth, zeroed: false, context: ctx)
        let errFlag = try MetalArrowBuffer.allocate(byteCount: 4, context: ctx)
        let errIndex = try MetalArrowBuffer.allocate(byteCount: 4, zeroed: false, context: ctx)
        errIndex.mutableTyped(UInt32.self)[0] = .max
        if n > 0 {
            let pso = try Self.trigPipeline(ctx, "tg_\(op.rawValue)_checked")
            let vld = validity ?? values
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                enc.setBuffer(vld.mtl, offset: vld.offset, index: 1)
                Dispatch.setLength(enc, n, lengthBuffer, index: 2)
                Dispatch.setUInt(enc, validity == nil ? 0 : 1, index: 3)
                enc.setBuffer(out.mtl, offset: out.offset, index: 4)
                enc.setBuffer(errFlag.mtl, offset: errFlag.offset, index: 5)
                enc.setBuffer(errIndex.mtl, offset: errIndex.offset, index: 6)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
            ctx.retainUntilFlush(self)
        }
        let vals = values
        try ctx.afterFlush { [errFlag, errIndex] in
            guard errFlag.typed(UInt32.self)[0] != 0 else { return }
            let i = Int(errIndex.typed(UInt32.self)[0])
            throw ArrowMetalError.invalidArrowArray(
                "\(op.arrowName): domain error at index \(i) (value \(vals.typed(T.self)[i]))")
        }
        ctx.retainUntilFlush(errFlag); ctx.retainUntilFlush(errIndex)
        return inheritPending(MetalArray<T>(length: knownLength, nullCount: _nullCount, validity: validity, values: out, context: ctx))
    }

    public func sinChecked() throws -> MetalArray<T> { try trigChecked(.sin) }
    public func cosChecked() throws -> MetalArray<T> { try trigChecked(.cos) }
    public func tanChecked() throws -> MetalArray<T> { try trigChecked(.tan) }
    public func asinChecked() throws -> MetalArray<T> { try trigChecked(.asin) }
    public func acosChecked() throws -> MetalArray<T> { try trigChecked(.acos) }
    public func acoshChecked() throws -> MetalArray<T> { try trigChecked(.acosh) }
    public func atanhChecked() throws -> MetalArray<T> { try trigChecked(.atanh) }

    // MARK: - Named forms

    public func sin() throws -> MetalArray<T> { try trig(.sin) }
    public func cos() throws -> MetalArray<T> { try trig(.cos) }
    public func tan() throws -> MetalArray<T> { try trig(.tan) }
    public func asin() throws -> MetalArray<T> { try trig(.asin) }
    public func acos() throws -> MetalArray<T> { try trig(.acos) }
    public func atan() throws -> MetalArray<T> { try trig(.atan) }
    public func sinh() throws -> MetalArray<T> { try trig(.sinh) }
    public func cosh() throws -> MetalArray<T> { try trig(.cosh) }
    public func tanh() throws -> MetalArray<T> { try trig(.tanh) }
    public func asinh() throws -> MetalArray<T> { try trig(.asinh) }
    public func acosh() throws -> MetalArray<T> { try trig(.acosh) }
    public func atanh() throws -> MetalArray<T> { try trig(.atanh) }
}
