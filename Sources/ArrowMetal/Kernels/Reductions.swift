import Foundation
import Metal

/// Result of a sum over an integer column is widened to Int64/UInt64; floats sum to Double.
public enum SumResult: Equatable {
    case int(Int64)
    case uint(UInt64)
    case float(Double)
    public var asDouble: Double {
        switch self { case .int(let v): return Double(v); case .uint(let v): return Double(v); case .float(let v): return v }
    }
}

extension MetalArray {
    /// Number of non-null values.
    public var validCount: Int { length - nullCount }

    /// Sum of non-null values. Returns nil when there are no valid values (Arrow semantics).
    /// Integer sums are exact (Int64/UInt64 accumulation, wrapping on overflow like Arrow's `sum`).
    /// Float32 sums accumulate per-thread in float and finalise in double.
    public func sum() throws -> SumResult? {
        if validCount == 0 { return nil }
        guard Dispatch.runsOnGPU(T.self) else { return CPUReference.sum(self) }
        let (partials, counts, groups) = try runReduction("reduce_sum")
        // The raw pointers below must not outlive the buffer objects (release builds shorten lifetimes).
        return withExtendedLifetime((partials, counts)) {
            if T.isFloatingPoint {
                let p = partials.typed(Float.self)
                var acc = 0.0
                for g in 0..<groups { acc += Double(p[g]) }
                return .float(acc)
            } else if T.minValue < 0 as T {
                let p = partials.typed(Int64.self)
                var acc: Int64 = 0
                for g in 0..<groups { acc &+= p[g] }
                return .int(acc)
            } else {
                let p = partials.typed(UInt64.self)
                var acc: UInt64 = 0
                for g in 0..<groups { acc &+= p[g] }
                return .uint(acc)
            }
        }
    }

    /// Minimum non-null value, or nil if none.
    public func min() throws -> T? {
        if validCount == 0 { return nil }
        let (partials, counts, groups) = try runReduction("reduce_min")
        return finaliseMinMax(partials, counts, groups, initial: T.maxValue) { Swift.min($0, $1) }
    }

    /// Maximum non-null value, or nil if none.
    public func max() throws -> T? {
        if validCount == 0 { return nil }
        let (partials, counts, groups) = try runReduction("reduce_max")
        return finaliseMinMax(partials, counts, groups, initial: T.minValue) { Swift.max($0, $1) }
    }

    /// Arithmetic mean of non-null values.
    public func mean() throws -> Double? {
        guard let s = try sum() else { return nil }
        return s.asDouble / Double(validCount)
    }

    private func finaliseMinMax(_ partials: MetalArrowBuffer, _ counts: MetalArrowBuffer, _ groups: Int,
                                initial: T, _ f: (T, T) -> T) -> T? {
        // Partials use the accumulator type: long/ulong for integers, float for Float.
        var acc = initial
        var any = false
        withExtendedLifetime((partials, counts)) {
            let c = counts.typed(UInt32.self)
            if T.self == Double.self {
                // Partials are order-preserving keys of the double bit patterns; NaNs were skipped.
                let p = partials.typed(Int64.self)
                for g in 0..<groups where c[g] > 0 { any = true; acc = f(acc, Dispatch.doubleFromKey(p[g]) as! T) }
            } else if T.isFloatingPoint {
                let p = partials.typed(Float.self)
                for g in 0..<groups where c[g] > 0 { any = true; acc = f(acc, T(p[g])) }
            } else if T.minValue < 0 as T {
                let p = partials.typed(Int64.self)
                for g in 0..<groups where c[g] > 0 { any = true; acc = f(acc, T(truncatingIfNeededInt64: p[g])) }
            } else {
                let p = partials.typed(UInt64.self)
                for g in 0..<groups where c[g] > 0 { any = true; acc = f(acc, T(truncatingIfNeededUInt64: p[g])) }
            }
        }
        // All valid values were NaN (floating point only): Arrow returns null.
        return any ? acc : nil
    }

    /// Runs a reduction kernel and returns (partials, counts, threadgroupCount).
    private func runReduction(_ fn: String) throws -> (MetalArrowBuffer, MetalArrowBuffer, Int) {
        try Dispatch.checkLength(length)
        let ctx = context
        let acc: String, mslT: String
        let minInit: String, maxInit: String
        var load = "(ACC)vals[i]", extra = "true"
        if T.self == Double.self {
            mslT = "long"; acc = "long"; minInit = "LONG_MAX"; maxInit = "LONG_MIN"
            load = "d_key(vals[i])"; extra = "!d_isnan(vals[i])"
        } else if T.isFloatingPoint { mslT = T.mslType; acc = "float"; minInit = "INFINITY"; maxInit = "-INFINITY"; extra = "!isnan(vals[i])" }
        else if T.minValue < 0 as T { mslT = T.mslType; acc = "long"; minInit = "LONG_MAX"; maxInit = "LONG_MIN" }
        else { mslT = T.mslType; acc = "ulong"; minInit = "ULONG_MAX"; maxInit = "0" }
        // Float sum must include NaN (it propagates); only min/max skip NaN.
        if fn == "reduce_sum" { extra = "true" }
        let src = KernelSource.reductions(T: mslT, ACC: acc, minInit: minInit, maxInit: maxInit, load: load, extra: extra)
        let pso = try Dispatch.pipeline(ctx, family: "reduce", source: src, function: fn, type: mslT + (extra == "true" ? "" : "/skipnan"))
        // Enough threadgroups to saturate the GPU, but few enough that the CPU finalise is trivial.
        let groups = Swift.max(1, Swift.min(2048, (length + Dispatch.threadgroupSize - 1) / Dispatch.threadgroupSize))
        let partials = try MetalArrowBuffer.allocate(byteCount: groups * 8, context: ctx)
        let counts = try MetalArrowBuffer.allocate(byteCount: groups * 4, context: ctx)
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            if let v = validity { enc.setBuffer(v.mtl, offset: v.offset, index: 1) } else { enc.setBuffer(values.mtl, offset: 0, index: 1) }
            Dispatch.setUInt(enc, length, index: 2)
            Dispatch.setUInt(enc, validity == nil ? 0 : 1, index: 3)
            enc.setBuffer(partials.mtl, offset: 0, index: 4)
            enc.setBuffer(counts.mtl, offset: 0, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1))
        }
        return (partials, counts, groups)
    }
}

extension ArrowPrimitive {
    init(truncatingIfNeededInt64 v: Int64) {
        if let b = Self.self as? any BinaryInteger.Type { self = b.init(truncatingIfNeeded: v) as! Self }
        else { self = Self(exactly: v) ?? 0 }
    }
    init(truncatingIfNeededUInt64 v: UInt64) {
        if let b = Self.self as? any BinaryInteger.Type { self = b.init(truncatingIfNeeded: v) as! Self }
        else { self = Self(exactly: v) ?? 0 }
    }
}
