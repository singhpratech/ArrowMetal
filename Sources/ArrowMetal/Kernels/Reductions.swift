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
        if !pending && validCount == 0 { return nil }
        let (partials, counts, groups) = try runReduction("reduce_sum")
        if validCount == 0 { return nil }   // (now synced) all-null after a pending filter
        return finaliseSum(partials, counts, groups)
    }

    /// CPU half of `sum`: combines the per-threadgroup partials. Pure CPU, no GPU work, no sync.
    func finaliseSum(_ partials: MetalArrowBuffer, _ counts: MetalArrowBuffer, _ groups: Int) -> SumResult {
        // The raw pointers below must not outlive the buffer objects (release builds shorten lifetimes).
        return withExtendedLifetime((partials, counts)) {
            if T.isFloatingPoint {
                // Partials are IEEE doubles produced by software d_add on the GPU (Float32 widened exactly); combine on the CPU.
                let p = partials.typed(UInt64.self)
                var acc = 0.0
                for g in 0..<groups { acc += Double(bitPattern: p[g]) }
                return .float(acc)
            } else if false {
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
        if !pending && validCount == 0 { return nil }
        let (partials, counts, groups) = try runReduction("reduce_min")
        if validCount == 0 { return nil }
        return finaliseMinMax(partials, counts, groups, initial: T.maxValue) { Swift.min($0, $1) }
    }

    /// Maximum non-null value, or nil if none.
    public func max() throws -> T? {
        if !pending && validCount == 0 { return nil }
        let (partials, counts, groups) = try runReduction("reduce_max")
        if validCount == 0 { return nil }
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

    /// Records a reduction kernel and waits for it, returning (partials, counts, threadgroupCount).
    private func runReduction(_ fn: String) throws -> (MetalArrowBuffer, MetalArrowBuffer, Int) {
        let r = try recordReduction(fn)
        try context.syncPoint()   // the partials are read on the CPU next
        return r
    }

    /// Records a reduction kernel *without* syncing. The partials are only valid once the command buffer
    /// they were recorded into has completed (see `sumAsync`).
    func recordReduction(_ fn: String) throws -> (MetalArrowBuffer, MetalArrowBuffer, Int) {
        try Dispatch.checkLength(dispatchLength)
        let ctx = context
        let n = dispatchLength
        let acc: String, mslT: String
        let minInit: String, maxInit: String
        var load = "(ACC)vals[i]", extra = "true"
        var combineSum = "acc + v", extraPrelude = ""
        if T.self == Double.self && fn == "reduce_sum" {
            // Software IEEE double accumulation on raw bit patterns.
            mslT = "long"; acc = "ulong"; minInit = "0"; maxInit = "0"
            load = "(ulong)vals[i]"; combineSum = "d_add(acc, v)"; extraPrelude = DoubleMath.msl
        } else if T.self == Float.self && fn == "reduce_sum" {
            // Arrow sums float32 into float64: widen each value exactly and accumulate in software double.
            mslT = "float"; acc = "ulong"; minInit = "0"; maxInit = "0"
            load = "d_from_float(vals[i])"; combineSum = "d_add(acc, v)"; extraPrelude = DoubleMath.msl
        } else if T.self == Double.self {
            mslT = "long"; acc = "long"; minInit = "LONG_MAX"; maxInit = "LONG_MIN"
            load = "d_key(vals[i])"; extra = "!d_isnan(vals[i])"
        } else if T.isFloatingPoint { mslT = T.mslType; acc = "float"; minInit = "INFINITY"; maxInit = "-INFINITY"; extra = "!isnan(vals[i])" }
        else if T.minValue < 0 as T { mslT = T.mslType; acc = "long"; minInit = "LONG_MAX"; maxInit = "LONG_MIN" }
        else { mslT = T.mslType; acc = "ulong"; minInit = "ULONG_MAX"; maxInit = "0" }
        // Float sum must include NaN (it propagates); only min/max skip NaN.
        if fn == "reduce_sum" { extra = "true" }
        // The source generator is passed inline (not via a `let`) so a warm pipeline cache never
        // builds the MSL: that string work measured ~20 µs per call, a fifth of a 1,000-row sum.
        let pso = try Dispatch.pipeline(ctx, family: "reduce",
                                        source: KernelSource.reductions(T: mslT, ACC: acc, minInit: minInit, maxInit: maxInit,
                                                                        load: load, extra: extra, combineSum: combineSum,
                                                                        extraPrelude: extraPrelude),
                                        function: fn, type: mslT + (extra == "true" ? "" : "/skipnan") + (extraPrelude.isEmpty ? "" : "/dd"))
        // Enough threadgroups to saturate the GPU, but few enough that the CPU finalise is trivial.
        let groups = Swift.max(1, Swift.min(2048, (n + Dispatch.threadgroupSize - 1) / Dispatch.threadgroupSize))
        let partials = try MetalArrowBuffer.allocate(byteCount: groups * 8, zeroed: false, context: ctx)
        let counts = try MetalArrowBuffer.allocate(byteCount: groups * 4, zeroed: false, context: ctx)
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            if let v = validity { enc.setBuffer(v.mtl, offset: v.offset, index: 1) } else { enc.setBuffer(values.mtl, offset: 0, index: 1) }
            Dispatch.setLength(enc, n, lengthBuffer, index: 2)
            Dispatch.setUInt(enc, validity == nil ? 0 : 1, index: 3)
            enc.setBuffer(partials.mtl, offset: 0, index: 4)
            enc.setBuffer(counts.mtl, offset: 0, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1))
        }
        ctx.retainUntilFlush(partials); ctx.retainUntilFlush(counts); ctx.retainUntilFlush(self)
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
