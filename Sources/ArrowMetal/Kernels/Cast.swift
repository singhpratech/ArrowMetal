import Foundation
import Metal

extension MetalArray {
    /// Arrow `cast` between primitive types. Integer narrowing wraps; float to integer truncates toward zero
    /// with unspecified results out of range (Arrow's unchecked cast). Validity is shared zero-copy.
    /// Casts involving Float64 run on the CPU.
    public func cast<U: ArrowPrimitive>(to _: U.Type) throws -> MetalArray<U> {
        if U.self == T.self { return self as! MetalArray<U> }
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: n * U.byteWidth, zeroed: false, context: ctx)
        if Dispatch.runsOnGPU(T.self) && Dispatch.runsOnGPU(U.self) {
            let src = KernelSource.cast(From: T.mslType, To: U.mslType)
            let pso = try Dispatch.pipeline(ctx, family: "cast", source: src, function: "cast_kernel", type: "\(T.mslType)->\(U.mslType)")
            if n > 0 {
                try ctx.run { enc in
                    enc.setComputePipelineState(pso)
                    enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                    Dispatch.setLength(enc, n, lengthBuffer, index: 1)
                    enc.setBuffer(out.mtl, offset: out.offset, index: 2)
                    Dispatch.dispatch1D(enc, pso, count: (n + 3) / 4)
                }
            }
            return inheritPending(MetalArray<U>(length: knownLength, nullCount: _nullCount, validity: validity, values: out, context: ctx))
        } else {
            let src = valuePointer, dst = out.mutableTyped(U.self)   // valuePointer syncs if pending
            for i in 0..<length { dst[i] = U.convert(src[i]) }
        }
        return MetalArray<U>(length: length, nullCount: nullCount, validity: validity, values: out, context: ctx)
    }
}

extension ArrowPrimitive {
    /// C-style numeric conversion used by the CPU cast path.
    static func convert<S: ArrowPrimitive>(_ v: S) -> Self {
        if let f = Self.self as? any BinaryFloatingPoint.Type {
            return f.init(v.asDouble) as! Self
        }
        let i = Self.self as! any FixedWidthInteger.Type
        if S.isFloatingPoint {
            let d = v.asDouble
            guard d.isFinite else { return i.init(truncatingIfNeeded: 0) as! Self }
            let t = d.rounded(.towardZero)
            // Clamp like the Metal conversion effectively does on Apple GPUs. This saturates at 64
            // bits and then truncates, where Arrow saturates at the *target's* width for a four-byte
            // or wider target — so `float64(1e20) -> int32` is -1 here and Int32.max in Arrow. The
            // difference is deliberate and pinned by `test_out_of_range_float_to_int_cast_diverges`:
            // it is the unchecked cast, which C leaves undefined, and `options: .safe` refuses the
            // row outright rather than choosing between the two answers.
            if t <= -9.3e18 { return i.init(truncatingIfNeeded: Int64.min) as! Self }
            if t >= 1.9e19 { return i.init(truncatingIfNeeded: UInt64.max) as! Self }
            if t >= 9.3e18 { return i.init(truncatingIfNeeded: UInt64(t)) as! Self }
            return i.init(truncatingIfNeeded: Int64(t)) as! Self
        }
        if S.minValue < 0 as S { return i.init(truncatingIfNeeded: v.asInt64) as! Self }
        return i.init(truncatingIfNeeded: v.asUInt64) as! Self
    }
}
